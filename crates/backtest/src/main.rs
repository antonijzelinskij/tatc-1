//! EVM-based backtester for the MEV sniper.
//!
//! Workflow:
//!   1. Spawn an Anvil node forked at the target historical block.
//!   2. Deploy MevExecutor onto the fork.
//!   3. Submit our snipe transaction.
//!   4. Mine several blocks and sell.
//!   5. Calculate and report net PnL.
//!
//! Usage:
//!   cargo run -p backtest -- \
//!     --block    17046105 \
//!     --pair     0xA43fe16908251ee70EF74718545e4FE6C5cCEc9f \
//!     --token    0x6982508145454Ce325dDbE47a25d4ec3d2311933 \
//!     --eth-in   0.5

use alloy::{
    node_bindings::Anvil,
    primitives::{address, Address, Bytes, U256, I256},
    providers::{Provider, ProviderBuilder},
    rpc::types::TransactionRequest,
    sol,
    sol_types::SolCall,
    signers::local::PrivateKeySigner,
    network::EthereumWallet,
};
use clap::Parser;
use eyre::{Context, Result};
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

// ─── CLI ──────────────────────────────────────────────────────────────────────
#[derive(Parser, Debug)]
#[command(name = "mev-backtest", about = "EVM-based MEV sniper backtester")]
struct Args {
    /// Mainnet RPC URL for the Anvil fork (overrides .env)
    #[arg(long, env = "MAINNET_RPC_URL")]
    rpc_url: String,

    /// Fork block number (e.g. 17046105 for PEPE launch)
    #[arg(long)]
    block: u64,

    /// V2 pair address to snipe
    #[arg(long)]
    pair: Address,

    /// Token address to buy
    #[arg(long)]
    token: Address,

    /// ETH to invest (in ether, e.g. "0.5")
    #[arg(long, default_value = "0.1")]
    eth_in: f64,

    /// Max acceptable sell tax in BPS
    #[arg(long, default_value = "500")]
    max_tax_bps: u64,

    /// Number of blocks to mine before selling (simulate price appreciation)
    #[arg(long, default_value = "5")]
    hold_blocks: u64,
}

// ─── Contract ABIs ────────────────────────────────────────────────────────────
sol! {
    function snipeV2(
        address pair,
        address tokenOut,
        uint256 amountIn,
        uint256 minAmountOut,
        uint256 maxTaxBps,
        uint256 bribeAmount
    ) external payable;

    function sellV2(
        address pair,
        address token,
        uint256 amount,
        uint256 minEthOut,
        uint256 bribeAmount
    ) external;

    function withdrawETH() external;

    function balanceOf(address account) external view returns (uint256);

    function getReserves() external view returns (uint112 reserve0, uint112 reserve1, uint32 blockTimestampLast);

    function token0() external view returns (address);
}

// ─── Result ───────────────────────────────────────────────────────────────────
#[derive(Debug)]
pub struct BacktestResult {
    pub fork_block:          u64,
    pub token_address:       Address,
    pub pair_address:        Address,
    pub eth_invested:        U256,
    pub tokens_bought:       U256,
    pub gas_cost_eth:        U256,
    pub bribe_paid:          U256,
    pub sell_eth_received:   U256,
    pub gross_pnl:           I256,
    pub net_pnl:             I256,
    pub honeypot_detected:   bool,
}

// ─── Main ─────────────────────────────────────────────────────────────────────
#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,backtest=debug")),
        )
        .init();

    dotenvy::dotenv().ok();
    let args = Args::parse();

    info!("=== MEV Sniper Backtest ===");
    info!("Fork block:  {}", args.block);
    info!("Pair:        {}", args.pair);
    info!("Token:       {}", args.token);
    info!("ETH in:      {} ETH", args.eth_in);
    info!("Hold blocks: {}", args.hold_blocks);

    let result = run_backtest(&args).await?;
    print_result(&result);

    Ok(())
}

async fn run_backtest(args: &Args) -> Result<BacktestResult> {
    // ── 1. Launch Anvil forked at the target block ────────────────────────────
    info!("Spawning Anvil forked at block {} ...", args.block);
    let anvil = Anvil::new()
        .fork(&args.rpc_url)
        .fork_block_number(args.block)
        .block_time(12u64)
        .spawn();

    let endpoint = anvil.endpoint();
    info!("Anvil at {endpoint}");

    // ── 2. Connect provider ───────────────────────────────────────────────────
    let signer: PrivateKeySigner = anvil.keys()[0].clone().into();
    let owner  = signer.address();
    let wallet = EthereumWallet::from(signer);

    let provider = ProviderBuilder::new()
        .wallet(wallet)
        .connect_http(endpoint.parse()?);

    let owner_bal = provider.get_balance(owner).await?;
    info!("Owner: {owner}  balance: {}", format_eth(owner_bal));

    // ── 3. Deploy MevExecutor ─────────────────────────────────────────────────
    info!("Deploying MevExecutor ...");
    let executor_address = deploy_executor(&provider, owner).await
        .context("deploy MevExecutor")?;
    info!("MevExecutor → {executor_address}");

    // Fund the executor
    let eth_in_wei  = eth_to_wei(args.eth_in);
    let extra       = eth_to_wei(0.1);
    let fund_tx = TransactionRequest::default()
        .from(owner)
        .to(executor_address)
        .value(eth_in_wei + extra);
    provider.send_transaction(fund_tx).await?.get_receipt().await?;
    info!("Executor funded: {}", format_eth(provider.get_balance(executor_address).await?));

    // ── 4. Check liquidity ────────────────────────────────────────────────────
    let (r0, r1) = get_reserves(&provider, args.pair).await?;
    if r0 == 0 && r1 == 0 {
        eyre::bail!(
            "Pair {} has no liquidity at block {}. Try a later block.",
            args.pair, args.block
        );
    }
    info!("Reserves: r0={r0}  r1={r1}");

    // Determine token ordering
    let t0_addr   = call_token0(&provider, args.pair).await?;
    let weth_addr = address!("C02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2");
    let weth_is_t0 = t0_addr == weth_addr;

    let (reserve_weth, reserve_token) = if weth_is_t0 {
        (U256::from(r0), U256::from(r1))
    } else {
        (U256::from(r1), U256::from(r0))
    };

    let expected_out = get_amount_out(eth_in_wei, reserve_weth, reserve_token);
    let min_out      = expected_out * U256::from(90u64) / U256::from(100u64);
    info!("Expected out: {expected_out}  min accepted: {min_out}");

    // ── 5. Execute snipe ──────────────────────────────────────────────────────
    let bribe_amount = eth_to_wei(0.001);
    let gas_price    = provider.get_gas_price().await?;

    let snipe_start_balance = provider.get_balance(executor_address).await?;

    let calldata = snipeV2Call {
        pair:         args.pair,
        tokenOut:     args.token,
        amountIn:     eth_in_wei,
        minAmountOut: min_out,
        maxTaxBps:    U256::from(args.max_tax_bps),
        bribeAmount:  bribe_amount,
    }.abi_encode();

    let snipe_tx = TransactionRequest::default()
        .from(owner)
        .to(executor_address)
        .input(Bytes::from(calldata).into())
        .gas_limit(400_000);

    // Simulate first to get the revert reason if any
    let sim_call = TransactionRequest::default()
        .from(owner)
        .to(executor_address)
        .input(Bytes::from(snipeV2Call {
            pair: args.pair, tokenOut: args.token, amountIn: eth_in_wei,
            minAmountOut: min_out, maxTaxBps: U256::from(args.max_tax_bps),
            bribeAmount: bribe_amount,
        }.abi_encode()).into())
        .gas_limit(400_000);
    match provider.call(sim_call).await {
        Ok(_)  => info!("eth_call simulation: OK"),
        Err(e) => warn!("eth_call simulation REVERTED: {e}"),
    }

    info!("Executing snipeV2 ...");
    let (honeypot_detected, tokens_bought, gas_used_eth) =
        match provider.send_transaction(snipe_tx).await {
            Ok(pending) => {
                let receipt = pending.get_receipt().await?;
                let gas_eth = U256::from(receipt.gas_used) * U256::from(gas_price);
                if !receipt.status() {
                    warn!("Snipe reverted — honeypot or insufficient liquidity");
                    (true, U256::ZERO, gas_eth)
                } else {
                    let bal = call_balance_of(&provider, args.token, executor_address).await?;
                    info!("Snipe OK — tokens: {bal}  gas: {}", format_eth(gas_eth));
                    (false, bal, gas_eth)
                }
            }
            Err(e) => {
                warn!("Snipe submission failed: {e}");
                (true, U256::ZERO, U256::ZERO)
            }
        };

    if honeypot_detected || tokens_bought.is_zero() {
        return Ok(BacktestResult {
            fork_block:        args.block,
            token_address:     args.token,
            pair_address:      args.pair,
            eth_invested:      eth_in_wei,
            tokens_bought:     U256::ZERO,
            gas_cost_eth:      gas_used_eth,
            bribe_paid:        bribe_amount,
            sell_eth_received: U256::ZERO,
            gross_pnl:         I256::ZERO,
            net_pnl:           I256::ZERO,
            honeypot_detected: true,
        });
    }

    // ── 6. Mine blocks ────────────────────────────────────────────────────────
    info!("Mining {} blocks ...", args.hold_blocks);
    mine_blocks(&provider, args.hold_blocks).await?;
    info!("Now at block {}", provider.get_block_number().await?);

    // ── 7. Sell ───────────────────────────────────────────────────────────────
    info!("Executing sellV2 ...");
    let sell_calldata = sellV2Call {
        pair:       args.pair,
        token:      args.token,
        amount:     tokens_bought,
        minEthOut:  U256::ZERO,
        bribeAmount: U256::ZERO,
    }.abi_encode();

    let sell_tx = TransactionRequest::default()
        .from(owner)
        .to(executor_address)
        .input(Bytes::from(sell_calldata).into())
        .gas_limit(300_000);

    // Record executor ETH balance before sell (after buy, WETH was used, ETH balance is lower)
    let eth_before_sell = provider.get_balance(executor_address).await?;

    let sell_receipt = provider.send_transaction(sell_tx).await?.get_receipt().await?;
    let sell_gas_eth = U256::from(sell_receipt.gas_used) * U256::from(gas_price);

    // sellV2 calls WETH.withdraw internally — executor now has ETH from the sell
    let eth_after_sell = provider.get_balance(executor_address).await?;
    let sell_proceeds = if eth_after_sell > eth_before_sell {
        eth_after_sell - eth_before_sell
    } else {
        U256::ZERO
    };

    let total_gas = gas_used_eth + sell_gas_eth;
    info!("Sell proceeds: {}  total gas: {}", format_eth(sell_proceeds), format_eth(total_gas));

    let gross_pnl = I256::try_from(sell_proceeds).unwrap_or(I256::ZERO)
        - I256::try_from(eth_in_wei).unwrap_or(I256::ZERO);
    let net_pnl = gross_pnl
        - I256::try_from(total_gas).unwrap_or(I256::ZERO)
        - I256::try_from(bribe_amount).unwrap_or(I256::ZERO);

    Ok(BacktestResult {
        fork_block:        args.block,
        token_address:     args.token,
        pair_address:      args.pair,
        eth_invested:      eth_in_wei,
        tokens_bought,
        gas_cost_eth:      total_gas,
        bribe_paid:        bribe_amount,
        sell_eth_received: sell_proceeds,
        gross_pnl,
        net_pnl,
        honeypot_detected: false,
    })
}

// ─── Deploy ───────────────────────────────────────────────────────────────────
async fn deploy_executor<P: Provider>(provider: &P, owner: Address) -> Result<Address> {
    let artifact_path = std::env::var("FOUNDRY_OUT").unwrap_or_else(|_| "out".to_string());
    let artifact_file = format!("{artifact_path}/MevExecutor.sol/MevExecutor.json");

    let bytecode: Vec<u8> = {
        let json_str = std::fs::read_to_string(&artifact_file)
            .with_context(|| format!("artifact not found: {artifact_file}. Run `forge build` first."))?;
        let json: serde_json::Value = serde_json::from_str(&json_str)?;
        let hex_str = json["bytecode"]["object"]
            .as_str()
            .ok_or_else(|| eyre::eyre!("bytecode.object missing in {artifact_file}"))?
            .trim_start_matches("0x");
        hex::decode(hex_str)?
    };

    // Use eth_sendTransaction directly — Anvil's accounts are unlocked,
    // so we don't need wallet signing for the deploy. This bypasses the
    // WalletFiller which rejects deployment txs (no `to` field).
    let bytecode_hex = format!("0x{}", hex::encode(&bytecode));
    let value_hex   = format!("0x{:x}", 1_000_000_000_000_000_000u128); // 1 ETH
    let gas_hex     = "0x2DC6C0"; // 3_000_000
    let tx_hash: alloy::primitives::B256 = provider
        .raw_request(
            "eth_sendTransaction".into(),
            (serde_json::json!({
                "from":  format!("{owner:?}"),
                "data":  bytecode_hex,
                "value": value_hex,
                "gas":   gas_hex,
            }),),
        )
        .await
        .context("eth_sendTransaction for deploy failed")?;

    // Wait for the receipt
    let mut receipt = None;
    for _ in 0..30 {
        if let Ok(Some(r)) = provider.get_transaction_receipt(tx_hash).await {
            receipt = Some(r);
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
    let receipt = receipt.ok_or_else(|| eyre::eyre!("deploy receipt not found"))?;

    receipt.contract_address
        .ok_or_else(|| eyre::eyre!("no contract_address in deploy receipt"))
}

// ─── Anvil ────────────────────────────────────────────────────────────────────
async fn mine_blocks<P: Provider>(provider: &P, count: u64) -> Result<()> {
    provider
        .raw_request::<(u64,), serde_json::Value>("anvil_mine".into(), (count,))
        .await
        .context("anvil_mine failed")?;
    Ok(())
}

// ─── On-chain reads ───────────────────────────────────────────────────────────
async fn get_reserves<P: Provider>(provider: &P, pair: Address) -> Result<(u128, u128)> {
    let result = provider
        .call(
            TransactionRequest::default()
                .to(pair)
                .input(Bytes::from(getReservesCall {}.abi_encode()).into()),
        )
        .await?;
    if result.len() < 96 {
        // Pair has no code or empty reserves — return zeros
        return Ok((0, 0));
    }
    let decoded = getReservesCall::abi_decode_returns(&result)?;
    Ok((decoded.reserve0.to::<u128>(), decoded.reserve1.to::<u128>()))
}

async fn call_token0<P: Provider>(provider: &P, pair: Address) -> Result<Address> {
    let result = provider
        .call(
            TransactionRequest::default()
                .to(pair)
                .input(Bytes::from(token0Call {}.abi_encode()).into()),
        )
        .await?;
    // alloy 1.x: single-return functions decode to the primitive directly
    let decoded = token0Call::abi_decode_returns(&result)?;
    Ok(decoded)
}

async fn call_balance_of<P: Provider>(
    provider: &P,
    token: Address,
    account: Address,
) -> Result<U256> {
    let result = provider
        .call(
            TransactionRequest::default()
                .to(token)
                .input(Bytes::from(balanceOfCall { account }.abi_encode()).into()),
        )
        .await?;
    let decoded = balanceOfCall::abi_decode_returns(&result)?;
    Ok(decoded)
}

// ─── Math ─────────────────────────────────────────────────────────────────────
fn get_amount_out(amount_in: U256, reserve_in: U256, reserve_out: U256) -> U256 {
    let in_with_fee = amount_in * U256::from(997u64);
    let numerator   = in_with_fee * reserve_out;
    let denominator = reserve_in  * U256::from(1000u64) + in_with_fee;
    numerator / denominator
}

fn eth_to_wei(eth: f64) -> U256 {
    U256::from((eth * 1e18) as u128)
}

fn format_eth(wei: U256) -> String {
    let meth = wei / U256::from(10u128.pow(15));
    let full = meth / U256::from(1000u64);
    let frac = meth % U256::from(1000u64);
    format!("{}.{:03} ETH", full, frac)
}

// ─── Display ──────────────────────────────────────────────────────────────────
fn print_result(r: &BacktestResult) {
    println!("\n╔══════════════════════════════════════════════════╗");
    println!("║          MEV Sniper Backtest Results             ║");
    println!("╠══════════════════════════════════════════════════╣");
    println!("║ Fork block:   {:>35} ║", r.fork_block);
    println!("║ Token:  {}… ║", &format!("{}", r.token_address)[..20]);
    println!("║ Pair:   {}… ║", &format!("{}", r.pair_address)[..20]);
    println!("╠══════════════════════════════════════════════════╣");

    if r.honeypot_detected {
        println!("║  HONEYPOT DETECTED — tx reverted                 ║");
        println!("║  Capital loss: ZERO (protected)                  ║");
        println!("║  Gas cost: {:>39} ║", format_eth(r.gas_cost_eth));
    } else {
        println!("║ ETH in:       {:>35} ║", format_eth(r.eth_invested));
        println!("║ Tokens bought:{:>35} ║", r.tokens_bought);
        println!("║ ETH from sell:{:>35} ║", format_eth(r.sell_eth_received));
        println!("║ Gas cost:     {:>35} ║", format_eth(r.gas_cost_eth));
        println!("║ Bribe paid:   {:>35} ║", format_eth(r.bribe_paid));
        println!("╠══════════════════════════════════════════════════╣");

        let (gs, gv) = if r.gross_pnl >= I256::ZERO {
            ("+", format_eth(U256::try_from(r.gross_pnl).unwrap_or_default()))
        } else {
            ("-", format_eth(U256::try_from(-r.gross_pnl).unwrap_or_default()))
        };
        let (ns, nv) = if r.net_pnl >= I256::ZERO {
            ("+", format_eth(U256::try_from(r.net_pnl).unwrap_or_default()))
        } else {
            ("-", format_eth(U256::try_from(-r.net_pnl).unwrap_or_default()))
        };
        println!("║ Gross PnL:  {}{:>37} ║", gs, gv);
        println!("║ Net PnL:    {}{:>37} ║", ns, nv);
    }

    println!("╚══════════════════════════════════════════════════╝\n");
}
