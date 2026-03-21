// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @dev Minimal ERC-20 used in tests.  Configurable sell-tax and blacklist.
contract MockERC20 {
    string public name;
    string public symbol;
    uint8  public constant decimals = 18;

    uint256 public totalSupply;
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    // ─── Honeypot controls ────────────────────────────────────────────────────
    /// @dev Tax applied to transfers TO `taxedRecipient` (simulates sell tax).
    ///      Expressed in BPS (10000 = 100%).
    uint256 public sellTaxBps;
    address public taxedRecipient; // usually the LP pair
    bool    public blacklistActive;
    mapping(address => bool) public blacklisted;

    address public immutable deployer;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    constructor(string memory _name, string memory _symbol, uint256 _totalSupply) {
        name        = _name;
        symbol      = _symbol;
        totalSupply = _totalSupply;
        balanceOf[msg.sender] = _totalSupply;
        deployer = msg.sender;
        emit Transfer(address(0), msg.sender, _totalSupply);
    }

    // ─── Admin ────────────────────────────────────────────────────────────────
    function configureTax(address recipient, uint256 bps) external {
        require(msg.sender == deployer, "only deployer");
        taxedRecipient = recipient;
        sellTaxBps     = bps;
    }

    function setBlacklist(address account, bool status) external {
        require(msg.sender == deployer, "only deployer");
        blacklisted[account] = status;
        blacklistActive = true;
    }

    // ─── ERC-20 ───────────────────────────────────────────────────────────────
    function approve(address spender, uint256 amount) external returns (bool) {
        allowance[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }

    function transfer(address to, uint256 amount) external returns (bool) {
        return _transfer(msg.sender, to, amount);
    }

    function transferFrom(address from, address to, uint256 amount) external returns (bool) {
        if (allowance[from][msg.sender] != type(uint256).max) {
            allowance[from][msg.sender] -= amount;
        }
        return _transfer(from, to, amount);
    }

    function _transfer(address from, address to, uint256 amount) internal returns (bool) {
        require(balanceOf[from] >= amount, "insufficient balance");
        if (blacklistActive) {
            require(!blacklisted[from] && !blacklisted[to], "blacklisted");
        }

        uint256 taxAmount = 0;
        if (to == taxedRecipient && sellTaxBps > 0) {
            taxAmount = (amount * sellTaxBps) / 10_000;
        }

        balanceOf[from]    -= amount;
        balanceOf[to]      += amount - taxAmount;
        // taxed tokens are burned (simulate fee-on-transfer)
        if (taxAmount > 0) {
            totalSupply -= taxAmount;
            emit Transfer(from, address(0), taxAmount);
        }

        emit Transfer(from, to, amount - taxAmount);
        return true;
    }

    function mint(address to, uint256 amount) external {
        require(msg.sender == deployer, "only deployer");
        balanceOf[to] += amount;
        totalSupply   += amount;
        emit Transfer(address(0), to, amount);
    }
}
