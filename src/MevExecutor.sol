// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "./interfaces/IWETH.sol";
import "./interfaces/IERC20.sol";
import "./interfaces/IUniswapV2Pair.sol";
import "./interfaces/IUniswapV2Factory.sol";
import "./interfaces/IUniswapV3Pool.sol";
import "./interfaces/IUniswapV3Factory.sol";

/// @title  MevExecutor
/// @notice Atomic MEV sniper: buy a new token with honeypot detection and
///         validator-bribe support.
///
/// Honeypot detection (V2):
///   The contract performs a full round-trip simulation *inside the same
///   transaction* using a sub-call that intentionally reverts to undo all
///   state changes from the test-sell.  If the simulation shows a sell tax
///   above `maxTaxBps`, the entire transaction reverts.  The bought tokens
///   are kept in the contract; a separate `sell` transaction is used later.
///
/// Bribe:
///   `block.coinbase.transfer(bribeAmount)` is executed at the end of each
///   snipe so the validator who includes our tx gets a tip.
contract MevExecutor {
    // ─── Errors ───────────────────────────────────────────────────────────────
    error Unauthorized();
    error HoneypotDetected(address token, uint256 taxBps);
    error InsufficientOutput(uint256 got, uint256 min);
    error InsufficientFunds();
    error TransferFailed();
    error SimulationFailed();

    // ─── Events ───────────────────────────────────────────────────────────────
    event Sniped(
        address indexed token,
        address indexed pair,
        uint256 ethIn,
        uint256 tokensOut,
        uint256 taxBps,
        uint256 bribePaid
    );
    event Sold(address indexed token, uint256 tokensIn, uint256 ethOut);

    // ─── Immutables ───────────────────────────────────────────────────────────
    address private immutable _owner;

    // Well-known mainnet addresses (immutable saves 2100 gas vs SLOAD per use)
    address internal constant WETH        = 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2;
    address internal constant V2_FACTORY  = 0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f;
    address internal constant V3_FACTORY  = 0x1F98431c8aD98523631AE4a59f267346ea31F984;

    // V3 sqrt price limits (full range)
    uint160 internal constant MIN_SQRT_RATIO = 4295128739;
    uint160 internal constant MAX_SQRT_RATIO = 1461446703485210103287273052203988822378723970342;

    uint256 private constant BPS = 10_000;

    // ─── Constructor ──────────────────────────────────────────────────────────
    constructor() payable {
        _owner = msg.sender;
    }

    receive() external payable {}

    // ─── Access Control ───────────────────────────────────────────────────────
    function owner() external view returns (address) {
        return _owner;
    }

    modifier onlyOwner() {
        assembly {
            // cheaper than reading the immutable through Solidity's slot system
            if iszero(eq(caller(), sload(_owner.slot))) {
                mstore(0x00, 0x82b42900) // Unauthorized()
                revert(0x1c, 0x04)
            }
        }
        _;
    }

    // ═════════════════════════════════════════════════════════════════════════
    //                         V2  SNIPE
    // ═════════════════════════════════════════════════════════════════════════

    /// @notice Atomically snipe a new Uniswap-V2 token with on-chain honeypot
    ///         protection.
    /// @param pair          The V2 pair address (tokenOut / WETH).
    /// @param tokenOut      The token we are buying.
    /// @param amountIn      ETH to spend (must be <= msg.value + contract balance).
    /// @param minAmountOut  Minimum tokens to receive (slippage guard).
    /// @param maxTaxBps     Maximum acceptable sell-tax in basis points.
    /// @param bribeAmount   ETH to send to block.coinbase after success.
    function snipeV2(
        address pair,
        address tokenOut,
        uint256 amountIn,
        uint256 minAmountOut,
        uint256 maxTaxBps,
        uint256 bribeAmount
    ) external payable onlyOwner {
        if (address(this).balance < amountIn + bribeAmount) revert InsufficientFunds();

        // ── 1. Wrap ETH ───────────────────────────────────────────────────────
        IWETH(WETH).deposit{value: amountIn}();

        // ── 2. Determine swap direction ───────────────────────────────────────
        bool wethIsToken0 = WETH < tokenOut; // addresses are sorted in the pair

        (uint112 r0, uint112 r1,) = IUniswapV2Pair(pair).getReserves();
        uint256 reserveWeth  = wethIsToken0 ? uint256(r0) : uint256(r1);
        uint256 reserveToken = wethIsToken0 ? uint256(r1) : uint256(r0);

        uint256 amountOut = _getAmountOut(amountIn, reserveWeth, reserveToken);
        if (amountOut < minAmountOut) revert InsufficientOutput(amountOut, minAmountOut);

        // ── 3. Execute BUY via pair directly (saves ~10k gas vs router) ───────
        _safeTransfer(WETH, pair, amountIn);
        {
            (uint256 out0, uint256 out1) = wethIsToken0
                ? (uint256(0), amountOut)
                : (amountOut, uint256(0));
            IUniswapV2Pair(pair).swap(out0, out1, address(this), new bytes(0));
        }

        // ── 4. Verify tokens received ─────────────────────────────────────────
        uint256 tokensBought = IERC20(tokenOut).balanceOf(address(this));
        if (tokensBought < minAmountOut) revert InsufficientOutput(tokensBought, minAmountOut);

        // ── 5. Honeypot simulation (sub-call that always reverts) ─────────────
        //      Encodes a test-sell, executes it in a child call.
        //      The child call ALWAYS reverts, undoing the test-sell state.
        //      We decode the result from the revert payload.
        uint256 taxBps = _simulateSellV2(pair, tokenOut, tokensBought, wethIsToken0);
        if (taxBps > maxTaxBps) revert HoneypotDetected(tokenOut, taxBps);

        // ── 6. Pay validator bribe ────────────────────────────────────────────
        if (bribeAmount > 0) _payBribe(bribeAmount);

        emit Sniped(tokenOut, pair, amountIn, tokensBought, taxBps, bribeAmount);
    }

    // ═════════════════════════════════════════════════════════════════════════
    //                         V3  SNIPE
    // ═════════════════════════════════════════════════════════════════════════

    struct SnipeV3Params {
        address pool;       // V3 pool address
        address tokenOut;   // token to buy
        uint24  fee;        // pool fee tier (500 / 3000 / 10000)
        uint256 amountIn;   // ETH to spend
        uint256 minAmountOut;
        uint256 maxTaxBps;
        uint256 bribeAmount;
    }

    /// @notice Atomically snipe a new Uniswap-V3 token.
    function snipeV3(SnipeV3Params calldata p) external payable onlyOwner {
        if (address(this).balance < p.amountIn + p.bribeAmount) revert InsufficientFunds();

        IWETH(WETH).deposit{value: p.amountIn}();

        IUniswapV3Pool pool = IUniswapV3Pool(p.pool);

        // token0 < token1 always; if WETH is token0 we sell WETH→token (zeroForOne=true)
        bool zeroForOne = WETH < p.tokenOut;

        uint256 tokensBefore = IERC20(p.tokenOut).balanceOf(address(this));

        // BUY via V3 swap; callback _transfers_ WETH to the pool
        pool.swap(
            address(this),
            zeroForOne,
            int256(p.amountIn),
            zeroForOne ? MIN_SQRT_RATIO + 1 : MAX_SQRT_RATIO - 1,
            abi.encode(V3CallbackData({tokenIn: WETH, fee: p.fee, payer: address(this)}))
        );

        uint256 tokensBought = IERC20(p.tokenOut).balanceOf(address(this)) - tokensBefore;
        if (tokensBought < p.minAmountOut) revert InsufficientOutput(tokensBought, p.minAmountOut);

        // Honeypot simulation for V3
        uint256 taxBps = _simulateSellV3(p.pool, p.tokenOut, tokensBought, !zeroForOne, p.fee);
        if (taxBps > p.maxTaxBps) revert HoneypotDetected(p.tokenOut, taxBps);

        if (p.bribeAmount > 0) _payBribe(p.bribeAmount);

        emit Sniped(p.tokenOut, p.pool, p.amountIn, tokensBought, taxBps, p.bribeAmount);
    }

    // ═════════════════════════════════════════════════════════════════════════
    //                         SELL  (post-snipe exit)
    // ═════════════════════════════════════════════════════════════════════════

    /// @notice Sell previously sniped tokens back via V2.
    function sellV2(
        address pair,
        address token,
        uint256 amount,
        uint256 minEthOut,
        uint256 bribeAmount
    ) external onlyOwner {
        bool wethIsToken0 = WETH < token;
        (uint112 r0, uint112 r1,) = IUniswapV2Pair(pair).getReserves();
        uint256 reserveToken = wethIsToken0 ? uint256(r1) : uint256(r0);
        uint256 reserveWeth  = wethIsToken0 ? uint256(r0) : uint256(r1);

        uint256 wethOut = _getAmountOut(amount, reserveToken, reserveWeth);
        if (wethOut < minEthOut) revert InsufficientOutput(wethOut, minEthOut);

        _safeTransfer(token, pair, amount);
        {
            (uint256 out0, uint256 out1) = wethIsToken0
                ? (wethOut, uint256(0))
                : (uint256(0), wethOut);
            IUniswapV2Pair(pair).swap(out0, out1, address(this), new bytes(0));
        }

        uint256 totalWeth = IWETH(WETH).balanceOf(address(this));
        IWETH(WETH).withdraw(totalWeth);

        if (bribeAmount > 0) _payBribe(bribeAmount);

        emit Sold(token, amount, totalWeth);
    }

    // ═════════════════════════════════════════════════════════════════════════
    //                      UNISWAP V3 CALLBACK
    // ═════════════════════════════════════════════════════════════════════════

    struct V3CallbackData {
        address tokenIn;
        uint24  fee;
        address payer;
    }

    function uniswapV3SwapCallback(int256 amount0Delta, int256 amount1Delta, bytes calldata data)
        external
    {
        V3CallbackData memory d = abi.decode(data, (V3CallbackData));

        // Validate caller: must be the V3 pool for (tokenIn, WETH/tokenOut, fee).
        // We store both possible orderings; one of them will match msg.sender.
        // This prevents a malicious contract from draining us via the callback.
        address expectedPool = IUniswapV3Factory(V3_FACTORY).getPool(d.tokenIn, WETH, d.fee);
        if (expectedPool == address(0)) {
            // Try the reverse ordering (tokenIn may be the bought token during test-sell)
            expectedPool = IUniswapV3Factory(V3_FACTORY).getPool(WETH, d.tokenIn, d.fee);
        }
        if (msg.sender != expectedPool) revert Unauthorized();

        uint256 amountToPay = amount0Delta > 0 ? uint256(amount0Delta) : uint256(amount1Delta);
        _safeTransfer(d.tokenIn, msg.sender, amountToPay);
    }

    // ═════════════════════════════════════════════════════════════════════════
    //                      ADMIN / RESCUE
    // ═════════════════════════════════════════════════════════════════════════

    function withdrawETH() external onlyOwner {
        (bool ok,) = _owner.call{value: address(this).balance}("");
        if (!ok) revert TransferFailed();
    }

    function withdrawToken(address token, uint256 amount) external onlyOwner {
        _safeTransfer(token, _owner, amount);
    }

    // ═════════════════════════════════════════════════════════════════════════
    //                     HONEYPOT  SIMULATION  HELPERS
    // ═════════════════════════════════════════════════════════════════════════

    // Packed error selector for the simulation result carrier.
    // keccak256("SimulationResult(uint256)") truncated to 4 bytes.
    bytes4 private constant SIM_RESULT_SELECTOR = 0xb4e7aea5;

    /// @dev Runs a test-sell inside a sub-call that always reverts.
    ///      Returns the effective sell-tax in BPS (0 = no tax, 10000 = 100%).
    function _simulateSellV2(
        address pair,
        address token,
        uint256 tokenAmount,
        bool wethIsToken0
    ) internal returns (uint256 taxBps) {
        // Encode the sub-call
        bytes memory callData = abi.encodeWithSelector(
            this._testSellV2AndRevert.selector,
            pair,
            token,
            tokenAmount,
            wethIsToken0
        );

        (bool success, bytes memory result) = address(this).call(callData);
        // success should ALWAYS be false (we intentionally revert in the sub-call)
        if (success) revert SimulationFailed();

        // Decode result: first 4 bytes = selector, next 32 bytes = wethReceived
        if (result.length >= 36 && bytes4(result) == SIM_RESULT_SELECTOR) {
            uint256 wethReceived;
            assembly { wethReceived := mload(add(result, 0x24)) }

            // Load amountIn from the buy context via WETH balance delta.
            // We approximate original amountIn from current WETH contract balance;
            // for tax calculation we use actual received vs theoretical reserves.
            // Here we reconstruct from pool state.
            (uint112 r0, uint112 r1,) = IUniswapV2Pair(pair).getReserves();
            uint256 reserveToken = wethIsToken0 ? uint256(r1) : uint256(r0);
            uint256 reserveWeth  = wethIsToken0 ? uint256(r0) : uint256(r1);
            uint256 theoreticalWeth = _getAmountOut(tokenAmount, reserveToken, reserveWeth);

            if (theoreticalWeth == 0) return BPS; // degenerate pool
            taxBps = theoreticalWeth > wethReceived
                ? ((theoreticalWeth - wethReceived) * BPS) / theoreticalWeth
                : 0;
        } else {
            // Unexpected revert inside the test-sell (e.g. blacklist, transfer revert)
            // → treat as 100% honeypot
            taxBps = BPS;
        }
    }

    /// @notice INTERNAL — called only by _simulateSellV2 via address(this).call.
    ///         Always reverts, carrying the test-sell result in the revert payload.
    /// @dev    onlyOwner is intentionally NOT applied; access is enforced by the
    ///         fact that only address(this) can call this (msg.sender == address(this)).
    function _testSellV2AndRevert(
        address pair,
        address token,
        uint256 tokenAmount,
        bool wethIsToken0
    ) external {
        if (msg.sender != address(this)) revert Unauthorized();

        uint256 wethBefore = IWETH(WETH).balanceOf(address(this));

        (uint112 r0, uint112 r1,) = IUniswapV2Pair(pair).getReserves();
        uint256 reserveToken = wethIsToken0 ? uint256(r1) : uint256(r0);
        uint256 reserveWeth  = wethIsToken0 ? uint256(r0) : uint256(r1);
        uint256 expectedWeth = _getAmountOut(tokenAmount, reserveToken, reserveWeth);

        // Send tokens to pair
        _safeTransfer(token, pair, tokenAmount);

        // Attempt swap (sell tokens → WETH)
        {
            (uint256 out0, uint256 out1) = wethIsToken0
                ? (expectedWeth, uint256(0))
                : (uint256(0), expectedWeth);
            // If this reverts (blacklist, broken token), the whole sub-call reverts
            // and the parent sees an unexpected revert → taxBps = BPS (honeypot).
            IUniswapV2Pair(pair).swap(out0, out1, address(this), new bytes(0));
        }

        uint256 wethReceived = IWETH(WETH).balanceOf(address(this)) - wethBefore;

        // Encode result and REVERT (this unwinds all state changes in this sub-call)
        bytes memory payload = abi.encodeWithSelector(SIM_RESULT_SELECTOR, wethReceived);
        assembly { revert(add(payload, 0x20), mload(payload)) }
    }

    // ── V3 simulation ─────────────────────────────────────────────────────────

    function _simulateSellV3(
        address pool,
        address token,
        uint256 tokenAmount,
        bool zeroForOne,
        uint24  fee
    ) internal returns (uint256 taxBps) {
        bytes memory callData = abi.encodeWithSelector(
            this._testSellV3AndRevert.selector,
            pool, token, tokenAmount, zeroForOne, fee
        );

        (bool success, bytes memory result) = address(this).call(callData);
        if (success) revert SimulationFailed();

        if (result.length >= 36 && bytes4(result) == SIM_RESULT_SELECTOR) {
            uint256 wethReceived;
            assembly { wethReceived := mload(add(result, 0x24)) }

            // Use current sqrtPrice to estimate theoretical output
            // For simplicity in MVP we compare wethReceived to amountIn stored in pool
            // A more precise approach would use the V3 quoter off-chain.
            // Here: if wethReceived > 0 → derive tax from ratio.
            // We need amountIn; pass it through by re-deriving from pool state.
            (uint160 sqrtPriceX96,,,,,,) = IUniswapV3Pool(pool).slot0();
            if (sqrtPriceX96 == 0) return BPS;

            // theoretical wethOut ≈ tokenAmount * (WETH reserve / token reserve)
            // For V3, best approximation: tokenAmount * P where P = (sqrtPrice/2^96)^2
            // We use uint256 arithmetic carefully to avoid overflow.
            uint256 priceX192 = uint256(sqrtPriceX96) * uint256(sqrtPriceX96);
            uint256 theoreticalWeth;
            if (zeroForOne) {
                // selling token1 for token0 (WETH): out ≈ in * (2^192 / priceX192)
                theoreticalWeth = (tokenAmount << 96) / (priceX192 >> 96);
            } else {
                // selling token0 for token1 (WETH): out ≈ in * priceX192 / 2^192
                theoreticalWeth = (tokenAmount * (priceX192 >> 96)) >> 96;
            }

            if (theoreticalWeth == 0) return BPS;
            taxBps = theoreticalWeth > wethReceived
                ? ((theoreticalWeth - wethReceived) * BPS) / theoreticalWeth
                : 0;
        } else {
            taxBps = BPS;
        }
    }

    function _testSellV3AndRevert(
        address pool,
        address token,
        uint256 tokenAmount,
        bool zeroForOne,
        uint24  fee
    ) external {
        if (msg.sender != address(this)) revert Unauthorized();

        uint256 wethBefore = IWETH(WETH).balanceOf(address(this));

        IUniswapV3Pool(pool).swap(
            address(this),
            zeroForOne,
            int256(tokenAmount),
            zeroForOne ? MIN_SQRT_RATIO + 1 : MAX_SQRT_RATIO - 1,
            abi.encode(V3CallbackData({tokenIn: token, fee: fee, payer: address(this)}))
        );

        uint256 wethReceived = IWETH(WETH).balanceOf(address(this)) - wethBefore;

        bytes memory payload = abi.encodeWithSelector(SIM_RESULT_SELECTOR, wethReceived);
        assembly { revert(add(payload, 0x20), mload(payload)) }
    }

    // ═════════════════════════════════════════════════════════════════════════
    //                          PURE  HELPERS
    // ═════════════════════════════════════════════════════════════════════════

    /// @dev Standard Uniswap V2 output calculation (0.3% fee).
    function _getAmountOut(uint256 amountIn, uint256 reserveIn, uint256 reserveOut)
        internal
        pure
        returns (uint256 amountOut)
    {
        unchecked {
            uint256 amountInWithFee = amountIn * 997;
            amountOut = (amountInWithFee * reserveOut) / (reserveIn * 1000 + amountInWithFee);
        }
    }

    // ─── Gas-optimised ERC-20 transfer ────────────────────────────────────────
    function _safeTransfer(address token, address to, uint256 amount) internal {
        assembly {
            let ptr := mload(0x40)
            // transfer(address,uint256)  selector = 0xa9059cbb
            mstore(ptr,         0xa9059cbb00000000000000000000000000000000000000000000000000000000)
            mstore(add(ptr, 4), and(to, 0xffffffffffffffffffffffffffffffffffffffff))
            mstore(add(ptr, 36), amount)
            let ok := call(gas(), token, 0, ptr, 68, 0x00, 0x20)
            // Revert if call failed OR token returned explicit false
            if iszero(ok) {
                mstore(0x00, 0x90b8ec18) // TransferFailed()
                revert(0x1c, 0x04)
            }
            if and(gt(returndatasize(), 0x1f), iszero(mload(0x00))) {
                mstore(0x00, 0x90b8ec18)
                revert(0x1c, 0x04)
            }
        }
    }

    // ─── Validator bribe ──────────────────────────────────────────────────────
    function _payBribe(uint256 amount) internal {
        assembly {
            if iszero(call(gas(), coinbase(), amount, 0, 0, 0, 0)) {
                mstore(0x00, 0x90b8ec18) // TransferFailed()
                revert(0x1c, 0x04)
            }
        }
    }
}
