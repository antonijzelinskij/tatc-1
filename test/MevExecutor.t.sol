// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "forge-std/Test.sol";
import "forge-std/console2.sol";

import {MevExecutor}        from "../src/MevExecutor.sol";
import {MockERC20}           from "./mocks/MockERC20.sol";
import {IUniswapV2Factory}   from "../src/interfaces/IUniswapV2Factory.sol";
import {IUniswapV2Pair}      from "../src/interfaces/IUniswapV2Pair.sol";
import {IUniswapV2Router02}  from "../src/interfaces/IUniswapV2Router02.sol";
import {IWETH}               from "../src/interfaces/IWETH.sol";
import {IERC20}              from "../src/interfaces/IERC20.sol";

// ─── Constants ────────────────────────────────────────────────────────────────
address constant WETH_ADDR       = 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2;
address constant V2_FACTORY_ADDR = 0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f;
address constant V2_ROUTER_ADDR  = 0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D;

// PEPE token (mainnet)
address constant PEPE_ADDR       = 0x6982508145454Ce325dDBe47a25d4ec3d2311933;
// PEPE/WETH V2 pair (mainnet)
address constant PEPE_WETH_PAIR  = 0xA43fe16908251ee70EF74718545e4FE6C5cCEc9f;
// Block just before PEPE pair had significant liquidity (initial LP event)
uint256 constant PEPE_FORK_BLOCK = 17_046_105;

// ─── Base fixture ─────────────────────────────────────────────────────────────
abstract contract BaseTest is Test {
    MevExecutor internal executor;
    address     internal owner = address(this);

    function setUp() public virtual {
        executor = new MevExecutor{value: 10 ether}();
        vm.deal(address(executor), 10 ether);
    }

    receive() external payable {}
}

// =============================================================================
//  SUITE 1 — Unit tests with mock tokens (no fork required)
// =============================================================================
contract MevExecutorUnitTest is BaseTest {
    IUniswapV2Factory factory;
    IUniswapV2Router02 router;
    MockERC20 token;
    address pair;

    function setUp() public override {
        // Fork mainnet so we have real V2 factory/router/WETH
        vm.createSelectFork(vm.envString("MAINNET_RPC_URL"));
        super.setUp();

        factory = IUniswapV2Factory(V2_FACTORY_ADDR);
        router  = IUniswapV2Router02(V2_ROUTER_ADDR);

        // Deploy a fresh mock token
        token = new MockERC20("TestToken", "TEST", 1_000_000_000e18);

        // Create V2 pair
        pair = factory.createPair(address(token), WETH_ADDR);

        // Add initial liquidity: 100M tokens + 10 ETH
        uint256 tokenLiq = 100_000_000e18;
        uint256 ethLiq   = 10 ether;

        token.approve(address(router), tokenLiq);
        router.addLiquidityETH{value: ethLiq}(
            address(token),
            tokenLiq,
            0, 0,
            address(this),
            block.timestamp + 60
        );
    }

    // ─── Test: happy path snipe ───────────────────────────────────────────────
    function test_SnipeV2_HappyPath() public {
        uint256 amountIn    = 0.1 ether;
        uint256 bribeAmount = 0.001 ether;

        address coinbase = address(0xCB);
        vm.coinbase(coinbase);

        uint256 coinbaseBefore  = coinbase.balance;
        uint256 contractEthBefore = address(executor).balance;

        // Expected minimum tokens
        (uint112 r0, uint112 r1,) = IUniswapV2Pair(pair).getReserves();
        bool wethIsToken0 = WETH_ADDR < address(token);
        uint256 rIn  = wethIsToken0 ? uint256(r0) : uint256(r1);
        uint256 rOut = wethIsToken0 ? uint256(r1) : uint256(r0);
        uint256 expectedOut = (amountIn * 997 * rOut) / (rIn * 1000 + amountIn * 997);
        uint256 minOut = expectedOut * 95 / 100; // 5% slippage

        executor.snipeV2(
            pair,
            address(token),
            amountIn,
            minOut,
            500,        // 5% max tax
            bribeAmount
        );

        uint256 tokenBalance = token.balanceOf(address(executor));
        assertGe(tokenBalance, minOut, "should have received tokens");
        assertEq(coinbase.balance - coinbaseBefore, bribeAmount, "bribe must be paid");
        assertEq(
            contractEthBefore - address(executor).balance,
            amountIn + bribeAmount,
            "ETH spend mismatch"
        );

        console2.log("Tokens bought:", tokenBalance / 1e18, "TEST");
        console2.log("ETH spent:    ", amountIn / 1e15, "mETH");
        console2.log("Bribe paid:   ", bribeAmount / 1e15, "mETH");
    }

    // ─── Test: honeypot — 100% sell tax ──────────────────────────────────────
    function test_HoneypotDetection_100PercentTax() public {
        // Set 100% sell tax targeting the pair
        token.configureTax(pair, 10_000);

        uint256 ethBefore = address(executor).balance;

        vm.expectRevert(
            abi.encodeWithSelector(MevExecutor.HoneypotDetected.selector, address(token), uint256(10_000))
        );
        executor.snipeV2(
            pair,
            address(token),
            0.1 ether,
            0,
            500,   // max 5% — 100% should trigger revert
            0
        );

        // Verify ETH is fully refunded (only gas cost changes, not balance)
        assertEq(address(executor).balance, ethBefore, "ETH must not be spent on honeypot");
    }

    // ─── Test: honeypot — high tax (30%) ─────────────────────────────────────
    function test_HoneypotDetection_HighTax_Reverts() public {
        // 30% sell tax
        token.configureTax(pair, 3_000);

        vm.expectRevert(abi.encodeWithSelector(MevExecutor.HoneypotDetected.selector, address(token)));
        executor.snipeV2(pair, address(token), 0.1 ether, 0, 500, 0);
    }

    // ─── Test: borderline tax — just within allowed ───────────────────────────
    function test_HoneypotDetection_AcceptableTax_Passes() public {
        // 3% sell tax, we allow up to 5%
        token.configureTax(pair, 300);

        executor.snipeV2(pair, address(token), 0.1 ether, 0, 500, 0);

        uint256 bal = token.balanceOf(address(executor));
        assertGt(bal, 0, "should have tokens");
    }

    // ─── Test: blacklist causes revert ────────────────────────────────────────
    function test_HoneypotDetection_Blacklist_Reverts() public {
        // Blacklist the executor contract
        token.setBlacklist(address(executor), true);

        vm.expectRevert(); // any revert is fine (blacklisted transfer fails)
        executor.snipeV2(pair, address(token), 0.1 ether, 0, 500, 0);

        // ETH must be untouched
        assertEq(address(executor).balance, 10 ether);
    }

    // ─── Test: bribe payment ─────────────────────────────────────────────────
    function test_BribeIsPaidToValidator() public {
        address coinbase = makeAddr("validator");
        vm.coinbase(coinbase);

        uint256 bribe = 0.05 ether;
        executor.snipeV2(pair, address(token), 0.1 ether, 0, 10_000, bribe);

        assertEq(coinbase.balance, bribe, "validator must receive exact bribe");
    }

    // ─── Test: slippage protection ────────────────────────────────────────────
    function test_SlippageProtection_Reverts() public {
        vm.expectRevert(abi.encodeWithSelector(MevExecutor.InsufficientOutput.selector));
        executor.snipeV2(
            pair,
            address(token),
            0.1 ether,
            type(uint256).max, // impossible min
            10_000,
            0
        );
    }

    // ─── Test: insufficient ETH ───────────────────────────────────────────────
    function test_InsufficientFunds_Reverts() public {
        vm.expectRevert(MevExecutor.InsufficientFunds.selector);
        executor.snipeV2(pair, address(token), 100 ether, 0, 10_000, 0);
    }

    // ─── Test: access control ────────────────────────────────────────────────
    function test_OnlyOwner_Reverts() public {
        vm.prank(address(0xBEEF));
        vm.expectRevert(MevExecutor.Unauthorized.selector);
        executor.snipeV2(pair, address(token), 0.1 ether, 0, 10_000, 0);
    }

    // ─── Test: sell after snipe ───────────────────────────────────────────────
    function test_SellAfterSnipe() public {
        // Buy first
        executor.snipeV2(pair, address(token), 0.1 ether, 0, 500, 0);
        uint256 tokensBought = token.balanceOf(address(executor));
        assertGt(tokensBought, 0);

        uint256 ethBefore = address(executor).balance;

        // Sell all tokens back
        executor.sellV2(pair, address(token), tokensBought, 0, 0);

        uint256 ethAfter = address(executor).balance;
        assertGt(ethAfter, ethBefore, "should receive ETH from sell");

        console2.log("ETH after sell:", ethAfter / 1e15, "mETH");
        console2.log("Tokens sold:   ", tokensBought / 1e18, "TEST");
    }

    // ─── Gas benchmark ────────────────────────────────────────────────────────
    function test_GasBenchmark_SnipeV2() public {
        uint256 gasBefore = gasleft();

        executor.snipeV2(pair, address(token), 0.1 ether, 0, 500, 0.001 ether);

        uint256 gasUsed = gasBefore - gasleft();
        console2.log("Gas used for snipeV2:", gasUsed);

        // Rough upper bound — should be well under 300k
        assertLt(gasUsed, 300_000, "gas should be under 300k");
    }

    // ─── PnL calculation ─────────────────────────────────────────────────────
    function test_PnL_Calculation() public {
        uint256 amountIn = 0.5 ether;
        uint256 bribe    = 0.01 ether;
        uint256 gasPrice = 20 gwei;

        vm.txGasPrice(gasPrice);

        uint256 ethBefore = address(executor).balance;
        uint256 gasBefore = gasleft();

        executor.snipeV2(pair, address(token), amountIn, 0, 500, bribe);

        uint256 gasUsed  = gasBefore - gasleft();
        uint256 tokensBought = token.balanceOf(address(executor));

        // Simulate price impact: after our buy, next buyer pays more
        // Mine 3 blocks, then sell
        vm.roll(block.number + 3);

        uint256 ethMid = address(executor).balance; // after snipe, before sell

        executor.sellV2(pair, address(token), tokensBought, 0, 0);

        uint256 ethAfter   = address(executor).balance;
        uint256 ethReceived = ethAfter - ethMid;
        uint256 gasCost     = gasUsed * gasPrice;

        int256 grossPnL = int256(ethReceived) - int256(amountIn);
        int256 netPnL   = grossPnL - int256(gasCost) - int256(bribe);

        console2.log("=== PnL Report ===");
        console2.log("ETH in:        ", amountIn / 1e15,    "mETH");
        console2.log("ETH out (sell):", ethReceived / 1e15, "mETH");
        console2.log("Gas cost:      ", gasCost / 1e9,      "gwei");
        console2.log("Bribe:         ", bribe / 1e15,       "mETH");

        if (netPnL >= 0) {
            console2.log("Net PnL (+):   ", uint256(netPnL) / 1e15, "mETH");
        } else {
            console2.log("Net PnL (-):  -", uint256(-netPnL) / 1e15, "mETH");
        }

        // In a freshly seeded pool with immediate round-trip, we expect a small
        // loss due to 2x AMM fees (0.3% buy + 0.3% sell).
        // The gross PnL should be approximately -(2 * 0.3%) = -0.6% of amountIn.
        console2.log("ETH before snipe:", ethBefore);
        console2.log("Final ETH balance:", address(executor).balance);
    }
}

// =============================================================================
//  SUITE 2 — Fork backtest on historical mainnet block
//  Demonstrates the REAL backtesting workflow against on-chain state.
//  Run with: forge test --match-contract ForkBacktest --fork-url $MAINNET_RPC_URL
// =============================================================================
contract ForkBacktestTest is Test {
    MevExecutor executor;

    // Known mainnet addresses
    address constant WETH          = WETH_ADDR;
    address constant V2_FACTORY    = V2_FACTORY_ADDR;
    address constant V2_ROUTER     = V2_ROUTER_ADDR;

    // We test against SHIB/WETH pair which has been live since 2021
    // and is a well-known non-honeypot token with large liquidity
    address constant SHIB_TOKEN    = 0x95aD61b0a150d79219dCF64E1E6Cc01f0B64C4cE;
    address constant SHIB_WETH_PAIR = 0x811beEd0119b4AfCE20D2583EB608C6F7AF1954f;
    uint256 constant TEST_BLOCK    = 15_000_000; // stable historical block

    function setUp() public {
        // Fork at a stable historical block
        vm.createSelectFork(vm.envString("MAINNET_RPC_URL"), TEST_BLOCK);

        executor = new MevExecutor{value: 5 ether}();
    }

    receive() external payable {}

    // ─── Fork test: snipe an established pool ────────────────────────────────
    function test_Fork_SnipeEstablishedPool() public {
        uint256 amountIn = 0.1 ether;

        (uint112 r0, uint112 r1,) = IUniswapV2Pair(SHIB_WETH_PAIR).getReserves();
        bool wethIsToken0 = WETH < SHIB_TOKEN;
        uint256 rIn  = wethIsToken0 ? uint256(r0) : uint256(r1);
        uint256 rOut = wethIsToken0 ? uint256(r1) : uint256(r0);

        uint256 expectedOut = (amountIn * 997 * rOut) / (rIn * 1000 + amountIn * 997);
        uint256 minOut = expectedOut * 95 / 100;

        console2.log("Fork block:   ", TEST_BLOCK);
        console2.log("ETH in:       ", amountIn / 1e15, "mETH");
        console2.log("Expected SHIB:", expectedOut / 1e18, "SHIB");

        address coinbase = makeAddr("validator");
        vm.coinbase(coinbase);
        uint256 bribe = 0.002 ether;

        executor.snipeV2(SHIB_WETH_PAIR, SHIB_TOKEN, amountIn, minOut, 500, bribe);

        uint256 shibBalance = IERC20(SHIB_TOKEN).balanceOf(address(executor));
        assertGe(shibBalance, minOut, "received SHIB");
        assertEq(coinbase.balance, bribe, "bribe paid");

        console2.log("SHIB received:", shibBalance / 1e18, "SHIB");
        console2.log("Bribe sent:   ", bribe / 1e15, "mETH");
    }

    // ─── Fork test: simulate the snipe BEFORE the block is mined ─────────────
    /// @notice Demonstrates the Flashbots bundle workflow:
    ///         1. Impersonate the LP provider
    ///         2. Add liquidity (victim tx)
    ///         3. Our snipe follows in the same block
    function test_Fork_BundleSimulation_NewLiquidity() public {
        // Deploy a fresh token and seed it with liquidity
        // This simulates finding a new pool in the pending mempool
        MockERC20 freshToken = new MockERC20("FreshToken", "FRSH", 1_000_000_000e18);

        IUniswapV2Factory factory = IUniswapV2Factory(V2_FACTORY);
        address newPair = factory.createPair(address(freshToken), WETH);

        // ── TX 1 (victim): Developer adds initial liquidity ───────────────────
        address developer = makeAddr("developer");
        vm.deal(developer, 20 ether);
        freshToken.mint(developer, 500_000_000e18);

        vm.startPrank(developer);
        freshToken.approve(V2_ROUTER, 500_000_000e18);
        IUniswapV2Router02(V2_ROUTER).addLiquidityETH{value: 10 ether}(
            address(freshToken),
            500_000_000e18,
            0, 0,
            developer,
            block.timestamp + 60
        );
        vm.stopPrank();

        console2.log("=== Bundle Simulation ===");
        console2.log("New pair deployed:", newPair);
        console2.log("Initial liquidity: 10 ETH + 500M FRSH");

        // ── TX 2 (ours): Snipe immediately in same block ──────────────────────
        uint256 snipeAmount = 0.5 ether;
        uint256 bribe       = 0.01 ether;

        address coinbase = makeAddr("coinbase");
        vm.coinbase(coinbase);

        uint256 executorEthBefore = address(executor).balance;

        (uint112 r0, uint112 r1,) = IUniswapV2Pair(newPair).getReserves();
        bool wethIsToken0 = WETH < address(freshToken);
        uint256 rIn  = wethIsToken0 ? uint256(r0) : uint256(r1);
        uint256 rOut = wethIsToken0 ? uint256(r1) : uint256(r0);
        uint256 expectedOut = (snipeAmount * 997 * rOut) / (rIn * 1000 + snipeAmount * 997);

        executor.snipeV2(newPair, address(freshToken), snipeAmount, expectedOut * 90 / 100, 500, bribe);

        uint256 tokensBought = freshToken.balanceOf(address(executor));
        uint256 ethSpent     = executorEthBefore - address(executor).balance;

        console2.log("Tokens bought:  ", tokensBought / 1e18, "FRSH");
        console2.log("ETH spent:      ", ethSpent / 1e15, "mETH (incl bribe)");
        console2.log("Coinbase bribe: ", coinbase.balance / 1e15, "mETH");

        assertGt(tokensBought, 0);
        assertEq(coinbase.balance, bribe);

        // ── Simulate price appreciation (others buy after us) ─────────────────
        address buyer2 = makeAddr("buyer2");
        vm.deal(buyer2, 5 ether);
        vm.startPrank(buyer2);
        address[] memory path = new address[](2);
        path[0] = WETH;
        path[1] = address(freshToken);
        IUniswapV2Router02(V2_ROUTER).swapExactETHForTokens{value: 2 ether}(
            0, path, buyer2, block.timestamp + 60
        );
        vm.stopPrank();

        // ── Sell our position for profit ──────────────────────────────────────
        uint256 ethBeforeSell = address(executor).balance;
        executor.sellV2(newPair, address(freshToken), tokensBought, 0, 0);
        uint256 ethFromSell = address(executor).balance - ethBeforeSell;

        int256 grossPnL = int256(ethFromSell) - int256(snipeAmount);
        console2.log("=== Final PnL Report ===");
        console2.log("ETH from sell:", ethFromSell / 1e15, "mETH");
        if (grossPnL >= 0) {
            console2.log("Gross PnL (+):", uint256(grossPnL) / 1e15, "mETH");
        } else {
            console2.log("Gross PnL (-):", uint256(-grossPnL) / 1e15, "mETH");
        }
        // After buying by buyer2, the price is higher, so we expect profit.
        assertGt(ethFromSell, snipeAmount, "should profit after price appreciation");
    }

    // ─── Fork test: honeypot detection on-chain ───────────────────────────────
    function test_Fork_HoneypotDetectedOnFork() public {
        // Create a honeypot token on the fork
        MockERC20 honeypot = new MockERC20("HoneypotToken", "HONEY", 1_000_000_000e18);

        IUniswapV2Factory factory = IUniswapV2Factory(V2_FACTORY);
        address hpPair = factory.createPair(address(honeypot), WETH);

        // Add liquidity
        honeypot.approve(V2_ROUTER, 100_000_000e18);
        IUniswapV2Router02(V2_ROUTER).addLiquidityETH{value: 5 ether}(
            address(honeypot),
            100_000_000e18,
            0, 0,
            address(this),
            block.timestamp + 60
        );

        // Activate 100% sell tax
        honeypot.configureTax(hpPair, 10_000);

        uint256 ethBefore = address(executor).balance;

        // Attempt to snipe — should revert with HoneypotDetected
        vm.expectRevert(abi.encodeWithSelector(MevExecutor.HoneypotDetected.selector, address(honeypot)));
        executor.snipeV2(hpPair, address(honeypot), 0.1 ether, 0, 500, 0);

        // ETH must be completely safe
        assertEq(address(executor).balance, ethBefore, "ETH balance unchanged after honeypot detection");
        assertEq(honeypot.balanceOf(address(executor)), 0, "no honeypot tokens held");

        console2.log("[OK] Honeypot detected and reverted. No capital lost.");
    }

    // ─── Fork test: compete on bribe (simulate competing bots) ──────────────
    function test_Fork_BribeCompetition() public {
        MockERC20 newToken = new MockERC20("CompeteToken", "COMP", 1_000_000_000e18);
        IUniswapV2Factory factory = IUniswapV2Factory(V2_FACTORY);
        address compPair = factory.createPair(address(newToken), WETH);

        // Add initial liquidity
        newToken.approve(V2_ROUTER, 500_000_000e18);
        IUniswapV2Router02(V2_ROUTER).addLiquidityETH{value: 10 ether}(
            address(newToken), 500_000_000e18, 0, 0, address(this), block.timestamp + 60
        );

        address validator = makeAddr("validator");
        vm.coinbase(validator);

        // Our bribe is competitive
        uint256 ourBribe = 0.05 ether;

        executor.snipeV2(compPair, address(newToken), 0.3 ether, 0, 500, ourBribe);

        assertEq(validator.balance, ourBribe, "validator received our bribe");

        uint256 tokens = newToken.balanceOf(address(executor));
        assertGt(tokens, 0, "tokens acquired");

        console2.log("Bribe paid to validator:", ourBribe / 1e15, "mETH");
        console2.log("Tokens acquired:", tokens / 1e18, "COMP");
    }
}
