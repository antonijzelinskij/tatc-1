//! EVM-based backtester for the MEV sniper.
//!
//! Workflow:
//!   1. Fork Anvil at target block.
//!   2. Deploy MevExecutor.
//!   3. Execute our snipe.
//!   4. Replay real transactions from the next N blocks (actual buyers).
//!   5. Sell our position into the moved market.
//!   6. Report PnL.
//!
//! Usage:
//!   cargo run -p backtest -- \
//!     --rpc-url $MAINNET_RPC_URL \
//!     --block   17100000 \
//!     --pair    0xA43fe16908251ee70EF74718545e4FE6C5cCEc9f \
//!     --token   0x6982508145454Ce325dDbE47a25d4ec3d2311933 \
//!     --eth-in  0.5 \
//!     --replay-blocks 20

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
use tracing::{info, warn, debug};
use tracing_subscriber::EnvFilter;

// ─── CLI ──────────────────────────────────────────────────────────────────────
#[derive(Parser, Debug)]
#[command(name = "mev-backtest", about = "EVM-based MEV sniper backtester with real tx replay")]
struct Args {
    /// Mainnet RPC URL for the Anvil fork (overrides .env)
    #[arg(long, env = "MAINNET_RPC_URL")]
    rpc_url: String,

    /// Fork block number
    #[arg(long)]
    block: u64,

    /// V2 pair address to snipe
    #[arg(long)]
    pair: Address,

    /// Token address to buy
    #[arg(long)]
    token: Address,

    /// ETH to invest (in ether)
    #[arg(long, default_value = "0.1")]
    eth_in: f64,

    /// Max acceptable sell tax in BPS
    #[arg(long, default_value = "500")]
    max_tax_bps: u64,

    /// Replay this many real blocks after our snipe (0 = mine empty blocks)
    #[arg(long, default_value = "0")]
    replay_blocks: u64,

    /// Mine this many empty blocks before selling (used when replay-blocks = 0)
    #[arg(long, default_value = "5")]
    hold_blocks: u64,

    /// [Sandwich mode] Replay the SAME block as the fork block.
    /// Forks at (block-1), runs our snipe first, then replays all real txs
    /// of `block` — so we front-run every buyer in that block.
    #[arg(long, default_value = "false")]
    replay_same_block: bool,

    /// [Position mode] Gas price in Gwei for our snipe tx.
    /// The target block is split: txs with HIGHER gasPrice run before us,
    /// txs with LOWER gasPrice run after us. Shows real queue position PnL.
    /// Requires --replay-same-block to be set.
    #[arg(long, default_value = "0")]
    gas_price_gwei: u64,
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
    pub replayed_txs:        u64,
    pub failed_txs:          u64,
    pub price_impact_pct:    f64,  // % price move from replay
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

    info!("=== MEV Sniper Backtest (real tx replay) ===");
    info!("Fork block:     {}", args.block);
    info!("Pair:           {}", args.pair);
    info!("Token:          {}", args.token);
    info!("ETH in:         {} ETH", args.eth_in);
    if args.replay_blocks > 0 {
        info!("Replay blocks:  {} (real mainnet txs)", args.replay_blocks);
    } else {
        info!("Hold blocks:    {} (empty)", args.hold_blocks);
    }

    let result = run_backtest(&args).await?;
    print_result(&result);
    Ok(())
}

// ─── Core ─────────────────────────────────────────────────────────────────────
async fn run_backtest(args: &Args) -> Result<BacktestResult> {
    // ── 1. Spawn Anvil fork ───────────────────────────────────────────────────
    // In sandwich mode: fork at block-1 so we can front-run the target block
    let fork_block = if args.replay_same_block {
        args.block.saturating_sub(1)
    } else {
        args.block
    };
    info!("Spawning Anvil forked at block {} ...", fork_block);
    // Resolve anvil binary — support both PATH and ~/.foundry/bin
    let anvil_bin = if std::process::Command::new("anvil").arg("--version").output().is_ok() {
        "anvil".to_string()
    } else {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
        format!("{}/.foundry/bin/anvil", home)
    };
    let anvil = Anvil::new()
        .path(&anvil_bin)
        .fork(&args.rpc_url)
        .fork_block_number(fork_block)
        .block_time(12u64)
        .spawn();

    let endpoint = anvil.endpoint();
    info!("Anvil at {endpoint}");

    // ── 2. Provider (wallet for signing non-deploy txs) ───────────────────────
    let signer: PrivateKeySigner = anvil.keys()[0].clone().into();
    let owner  = signer.address();
    let wallet = EthereumWallet::from(signer);

    let provider = ProviderBuilder::new()
        .wallet(wallet)
        .connect_http(endpoint.parse()?);

    info!("Owner: {owner}  balance: {}", format_eth(provider.get_balance(owner).await?));

    // ── 3. Deploy MevExecutor ─────────────────────────────────────────────────
    info!("Deploying MevExecutor ...");
    let executor_address = deploy_executor(&provider, owner).await
        .context("deploy MevExecutor")?;
    info!("MevExecutor → {executor_address}");

    // Fund executor
    let eth_in_wei = eth_to_wei(args.eth_in);
    let fund_tx = TransactionRequest::default()
        .from(owner)
        .to(executor_address)
        .value(eth_in_wei + eth_to_wei(0.1));
    provider.send_transaction(fund_tx).await?.get_receipt().await?;
    info!("Executor funded: {}", format_eth(provider.get_balance(executor_address).await?));

    // ── 4. Read pair reserves ─────────────────────────────────────────────────
    let (r0, r1) = get_reserves(&provider, args.pair).await?;
    if r0 == 0 && r1 == 0 {
        eyre::bail!("Pair {} has no liquidity at block {}", args.pair, args.block);
    }
    info!("Reserves: r0={r0}  r1={r1}");

    let t0_addr    = call_token0(&provider, args.pair).await?;
    let weth_addr  = address!("C02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2");
    let weth_is_t0 = t0_addr == weth_addr;

    let (reserve_weth, reserve_token) = if weth_is_t0 {
        (U256::from(r0), U256::from(r1))
    } else {
        (U256::from(r1), U256::from(r0))
    };

    let expected_out = get_amount_out(eth_in_wei, reserve_weth, reserve_token);
    let min_out      = expected_out * U256::from(90u64) / U256::from(100u64);
    info!("Expected PEPE out: {expected_out}");

    // ── 5. Snipe ──────────────────────────────────────────────────────────────
    let bribe_amount = eth_to_wei(0.001);
    let gas_price    = provider.get_gas_price().await?;

    let calldata = snipeV2Call {
        pair: args.pair, tokenOut: args.token, amountIn: eth_in_wei,
        minAmountOut: min_out, maxTaxBps: U256::from(args.max_tax_bps),
        bribeAmount: bribe_amount,
    }.abi_encode();

    // Pre-flight simulation
    match provider.call(TransactionRequest::default()
        .from(owner).to(executor_address)
        .input(Bytes::from(calldata.clone()).into()).gas_limit(400_000)
    ).await {
        Ok(_)  => info!("Simulation: OK"),
        Err(e) => warn!("Simulation REVERTED: {e}"),
    }

    let snipe_tx = TransactionRequest::default()
        .from(owner).to(executor_address)
        .input(Bytes::from(calldata).into()).gas_limit(400_000);

    info!("Executing snipeV2 ...");
    let (honeypot_detected, tokens_bought, gas_used_eth) =
        match provider.send_transaction(snipe_tx).await {
            Ok(pending) => {
                let receipt = pending.get_receipt().await?;
                let gas_eth = U256::from(receipt.gas_used) * U256::from(gas_price);
                if !receipt.status() {
                    warn!("Snipe reverted — honeypot!");
                    (true, U256::ZERO, gas_eth)
                } else {
                    let bal = call_balance_of(&provider, args.token, executor_address).await?;
                    info!("Snipe OK — tokens: {bal}  gas: {}", format_eth(gas_eth));
                    (false, bal, gas_eth)
                }
            }
            Err(e) => { warn!("Snipe failed: {e}"); (true, U256::ZERO, U256::ZERO) }
        };

    if honeypot_detected || tokens_bought.is_zero() {
        return Ok(BacktestResult {
            fork_block: args.block, token_address: args.token, pair_address: args.pair,
            eth_invested: eth_in_wei, tokens_bought: U256::ZERO,
            gas_cost_eth: gas_used_eth, bribe_paid: bribe_amount,
            sell_eth_received: U256::ZERO, gross_pnl: I256::ZERO, net_pnl: I256::ZERO,
            honeypot_detected: true, replayed_txs: 0, failed_txs: 0, price_impact_pct: 0.0,
        });
    }

    // Record WETH reserve after our buy (baseline for price impact)
    let (r0_post_buy, r1_post_buy) = get_reserves(&provider, args.pair).await?;
    let weth_reserve_post_buy = if weth_is_t0 {
        U256::from(r0_post_buy)
    } else {
        U256::from(r1_post_buy)
    };

    // ── 6. Replay real blocks OR mine empty ───────────────────────────────────
    let (replayed_txs, failed_txs) = if args.replay_same_block && args.gas_price_gwei > 0 {
        // Position mode: split the target block by gas price
        // Txs with gasPrice > ours → ran before us
        // Our snipe → already executed above
        // Txs with gasPrice < ours → ran after us (these pump our price)
        info!("=== POSITION MODE: gas price {} Gwei, block {} ===", args.gas_price_gwei, args.block);
        let our_gas_price_wei = args.gas_price_gwei * 1_000_000_000;
        replay_block_after_us(&provider, &args.rpc_url, args.block, our_gas_price_wei).await?
    } else if args.replay_same_block {
        // Sandwich mode: all txs of target block run AFTER our snipe
        info!("=== SANDWICH MODE: replaying real txs of block {} after our snipe ===", args.block);
        replay_real_blocks(&provider, &args.rpc_url, args.block, 1).await?
    } else if args.replay_blocks > 0 {
        replay_real_blocks(&provider, &args.rpc_url, args.block + 1, args.replay_blocks).await?
    } else {
        info!("Mining {} empty blocks ...", args.hold_blocks);
        mine_blocks(&provider, args.hold_blocks).await?;
        (0u64, 0u64)
    };

    let current_block = provider.get_block_number().await?;
    info!("Now at block {current_block}");

    // Measure price impact from replay
    let (r0_pre_sell, r1_pre_sell) = get_reserves(&provider, args.pair).await?;
    let weth_reserve_pre_sell = if weth_is_t0 {
        U256::from(r0_pre_sell)
    } else {
        U256::from(r1_pre_sell)
    };

    let price_impact_pct = if weth_reserve_post_buy > U256::ZERO {
        let diff = if weth_reserve_pre_sell > weth_reserve_post_buy {
            weth_reserve_pre_sell - weth_reserve_post_buy
        } else {
            U256::ZERO
        };
        // WETH reserve goes UP means buyers sent WETH in → price of token went up
        let num: u128 = diff.try_into().unwrap_or(u128::MAX);
        let den: u128 = weth_reserve_post_buy.try_into().unwrap_or(1);
        (num as f64 / den as f64) * 100.0
    } else {
        0.0
    };
    info!("WETH reserve change: {} → {}  ({:+.2}% price move)",
          format_eth(weth_reserve_post_buy), format_eth(weth_reserve_pre_sell), price_impact_pct);

    // ── 7. Sell ───────────────────────────────────────────────────────────────
    info!("Executing sellV2 ...");
    let sell_calldata = sellV2Call {
        pair: args.pair, token: args.token, amount: tokens_bought,
        minEthOut: U256::ZERO, bribeAmount: U256::ZERO,
    }.abi_encode();

    let eth_before_sell = provider.get_balance(executor_address).await?;

    let sell_tx = TransactionRequest::default()
        .from(owner).to(executor_address)
        .input(Bytes::from(sell_calldata).into()).gas_limit(300_000);

    let sell_receipt = provider.send_transaction(sell_tx).await?.get_receipt().await?;
    let sell_gas_eth = U256::from(sell_receipt.gas_used) * U256::from(gas_price);

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
        fork_block: args.block, token_address: args.token, pair_address: args.pair,
        eth_invested: eth_in_wei, tokens_bought,
        gas_cost_eth: total_gas, bribe_paid: bribe_amount,
        sell_eth_received: sell_proceeds, gross_pnl, net_pnl,
        honeypot_detected: false, replayed_txs, failed_txs, price_impact_pct,
    })
}

// ─── Real tx replay ───────────────────────────────────────────────────────────

/// Fetch full transaction data from mainnet and replay via anvil_impersonateAccount.
///
/// Strategy:
///   1. Fetch blocks with full tx objects (eth_getBlockByNumber, fullTx=true).
///   2. Zero out the fork's base fee so any gas price is accepted.
///   3. For each tx: impersonate sender, set balance if needed, send via eth_sendTransaction.
///
/// This works with any RPC provider (no eth_getRawTransactionByHash needed).
async fn replay_real_blocks<P: Provider>(
    provider: &P,
    rpc_url: &str,
    start_block: u64,
    count: u64,
) -> Result<(u64, u64)> {
    let http = reqwest::ClientBuilder::new()
        .danger_accept_invalid_certs(true)
        .timeout(std::time::Duration::from_secs(30))
        .build()?;

    // Zero out base fee on the fork so any historical gasPrice passes
    let _ = provider.raw_request::<_, serde_json::Value>(
        "anvil_setNextBlockBaseFeePerGas".into(),
        (serde_json::json!("0x0"),),
    ).await;

    let mut ok = 0u64;
    let mut fail = 0u64;

    for block_num in start_block..start_block + count {
        info!("── Replaying block {} ──", block_num);

        // Fetch full block (fullTransactions = true)
        let block_resp: serde_json::Value = http
            .post(rpc_url)
            .json(&serde_json::json!({
                "jsonrpc": "2.0", "id": 1,
                "method": "eth_getBlockByNumber",
                "params": [format!("0x{:x}", block_num), true]
            }))
            .send().await.context("eth_getBlockByNumber send")?
            .json().await.context("eth_getBlockByNumber parse")?;

        let txs = match block_resp["result"]["transactions"].as_array() {
            Some(arr) => arr.clone(),
            None => {
                warn!("Block {block_num}: no transactions");
                continue;
            }
        };

        info!("Block {block_num}: {} txs", txs.len());

        // Impersonate all unique senders upfront (batch)
        let senders: std::collections::HashSet<String> = txs.iter()
            .filter_map(|tx| tx["from"].as_str().map(|s| s.to_string()))
            .collect();
        for sender in &senders {
            let _ = provider.raw_request::<_, serde_json::Value>(
                "anvil_impersonateAccount".into(), (sender.clone(),),
            ).await;
            let _ = provider.raw_request::<_, serde_json::Value>(
                "anvil_setBalance".into(),
                (sender.clone(), "0x56BC75E2D63100000"), // 100 ETH
            ).await;
        }

        for (i, tx) in txs.iter().enumerate() {
            let from  = match tx["from"].as_str() { Some(s) => s.to_string(), None => continue };
            let to    = tx["to"].as_str().map(|s| s.to_string());
            let value = tx["value"].as_str().unwrap_or("0x0").to_string();
            let input = tx["input"].as_str().unwrap_or("0x").to_string();
            let gas   = tx["gas"].as_str().unwrap_or("0x5208").to_string();
            let gas_price = tx["gasPrice"].as_str()
                .or(tx["maxFeePerGas"].as_str())
                .unwrap_or("0x1").to_string();
            let nonce = tx["nonce"].as_str().unwrap_or("0x0").to_string();

            let mut tx_obj = serde_json::json!({
                "from": from, "value": value, "gas": gas,
                "gasPrice": gas_price, "data": input, "nonce": nonce,
            });
            if let Some(t) = &to { tx_obj["to"] = serde_json::json!(t); }

            let send_result: Result<alloy::primitives::B256, _> = provider
                .raw_request::<_, alloy::primitives::B256>(
                    "eth_sendTransaction".into(), (tx_obj,),
                ).await;

            match send_result {
                Ok(_) => { ok += 1; debug!("  [{i}] OK"); }
                Err(e) => { fail += 1; debug!("  [{i}] skip: {e}"); }
            }
        }

        // Stop impersonating all senders
        for sender in &senders {
            let _ = provider.raw_request::<_, serde_json::Value>(
                "anvil_stopImpersonatingAccount".into(), (sender.clone(),),
            ).await;
        }

        // Mine one block to include submitted txs; zero base fee again for next block
        mine_blocks(provider, 1).await?;
        let _ = provider.raw_request::<_, serde_json::Value>(
            "anvil_setNextBlockBaseFeePerGas".into(),
            (serde_json::json!("0x0"),),
        ).await;

        info!("Block {block_num} done: cumulative {ok} ok / {fail} skipped");
    }

    info!("Replay complete: {ok} applied, {fail} skipped");
    Ok((ok, fail))
}

/// Position-aware replay: only execute txs from `block` whose gasPrice < our_gas_price_wei.
/// These are the txs that in the real block ran AFTER us — they push the price in our favour.
async fn replay_block_after_us<P: Provider>(
    provider: &P,
    rpc_url: &str,
    block: u64,
    our_gas_price_wei: u64,
) -> Result<(u64, u64)> {
    let http = reqwest::ClientBuilder::new()
        .danger_accept_invalid_certs(true)
        .timeout(std::time::Duration::from_secs(30))
        .build()?;

    let block_resp: serde_json::Value = http
        .post(rpc_url)
        .json(&serde_json::json!({
            "jsonrpc": "2.0", "id": 1,
            "method": "eth_getBlockByNumber",
            "params": [format!("0x{:x}", block), true]
        }))
        .send().await.context("eth_getBlockByNumber")?
        .json().await.context("parse block")?;

    let all_txs = match block_resp["result"]["transactions"].as_array() {
        Some(arr) => arr.clone(),
        None => { warn!("No txs in block {block}"); return Ok((0, 0)); }
    };

    // Split by gas price
    let before_us: Vec<_> = all_txs.iter().filter(|tx| {
        let gp = tx["gasPrice"].as_str()
            .and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok())
            .unwrap_or(0);
        gp > our_gas_price_wei
    }).collect();

    let after_us: Vec<_> = all_txs.iter().filter(|tx| {
        let gp = tx["gasPrice"].as_str()
            .and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok())
            .unwrap_or(0);
        gp <= our_gas_price_wei
    }).collect();

    info!("Block {block}: {} txs before us (higher gasPrice), {} txs after us",
          before_us.len(), after_us.len());

    // Show PEPE-related txs in context
    let router = "0x7a250d5630b4cf539739df2c5dacb4c659f2488d";
    let pepe_pair = "0xa43fe16908251ee70ef74718545e4fe6c5ccec9f";
    for (label, group) in [("BEFORE US", &before_us), ("AFTER US", &after_us)] {
        for tx in group.iter() {
            let to = tx["to"].as_str().unwrap_or("").to_lowercase();
            if to == router || to == pepe_pair {
                let gp = tx["gasPrice"].as_str()
                    .and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok())
                    .unwrap_or(0);
                info!("  [{label}] Uniswap tx  gasPrice={:.1} Gwei  hash={}…",
                      gp as f64 / 1e9, &tx["hash"].as_str().unwrap_or("?")[..20]);
            }
        }
    }

    // Zero base fee so any historical gasPrice is accepted
    let _ = provider.raw_request::<_, serde_json::Value>(
        "anvil_setNextBlockBaseFeePerGas".into(), (serde_json::json!("0x0"),),
    ).await;

    // Run "txs after us" — these are the buyers that pump our token's price
    let mut ok = 0u64;
    let mut fail = 0u64;

    if after_us.is_empty() {
        info!("No txs after us — our gas price is lowest in the block");
        mine_blocks(provider, 1).await?;
        return Ok((0, 0));
    }

    // Impersonate all senders
    let senders: std::collections::HashSet<String> = after_us.iter()
        .filter_map(|tx| tx["from"].as_str().map(|s| s.to_string()))
        .collect();
    for sender in &senders {
        let _ = provider.raw_request::<_, serde_json::Value>(
            "anvil_impersonateAccount".into(), (sender.clone(),),
        ).await;
        let _ = provider.raw_request::<_, serde_json::Value>(
            "anvil_setBalance".into(), (sender.clone(), "0x56BC75E2D63100000"),
        ).await;
    }

    for (i, tx) in after_us.iter().enumerate() {
        let from  = match tx["from"].as_str() { Some(s) => s.to_string(), None => continue };
        let to    = tx["to"].as_str().map(|s| s.to_string());
        let value = tx["value"].as_str().unwrap_or("0x0").to_string();
        let input = tx["input"].as_str().unwrap_or("0x").to_string();
        let gas   = tx["gas"].as_str().unwrap_or("0x5208").to_string();
        let gas_price = tx["gasPrice"].as_str()
            .or(tx["maxFeePerGas"].as_str()).unwrap_or("0x1").to_string();
        let nonce = tx["nonce"].as_str().unwrap_or("0x0").to_string();

        let mut tx_obj = serde_json::json!({
            "from": from, "value": value, "gas": gas,
            "gasPrice": gas_price, "data": input, "nonce": nonce,
        });
        if let Some(t) = &to { tx_obj["to"] = serde_json::json!(t); }

        let r: Result<alloy::primitives::B256, _> = provider
            .raw_request("eth_sendTransaction".into(), (tx_obj,)).await;
        match r {
            Ok(_) => { ok += 1; debug!("  [{i}] OK"); }
            Err(e) => { fail += 1; debug!("  [{i}] skip: {e}"); }
        }
    }

    for sender in &senders {
        let _ = provider.raw_request::<_, serde_json::Value>(
            "anvil_stopImpersonatingAccount".into(), (sender.clone(),),
        ).await;
    }

    mine_blocks(provider, 1).await?;
    info!("Position replay done: {ok} after-us txs applied, {fail} skipped");
    Ok((ok, fail))
}

// ─── Deploy ───────────────────────────────────────────────────────────────────
async fn deploy_executor<P: Provider>(provider: &P, owner: Address) -> Result<Address> {
    let artifact_path = std::env::var("FOUNDRY_OUT").unwrap_or_else(|_| "out".to_string());
    let artifact_file = format!("{artifact_path}/MevExecutor.sol/MevExecutor.json");

    let bytecode: Vec<u8> = {
        let json_str = std::fs::read_to_string(&artifact_file)
            .with_context(|| format!("artifact not found: {artifact_file}. Run `forge build`."))?;
        let json: serde_json::Value = serde_json::from_str(&json_str)?;
        let hex_str = json["bytecode"]["object"]
            .as_str()
            .ok_or_else(|| eyre::eyre!("bytecode.object missing in {artifact_file}"))?
            .trim_start_matches("0x");
        hex::decode(hex_str)?
    };

    let bytecode_hex = format!("0x{}", hex::encode(&bytecode));
    let tx_hash: alloy::primitives::B256 = provider
        .raw_request(
            "eth_sendTransaction".into(),
            (serde_json::json!({
                "from":  format!("{owner:?}"),
                "data":  bytecode_hex,
                "value": format!("0x{:x}", 1_000_000_000_000_000_000u128),
                "gas":   "0x2DC6C0",
            }),),
        )
        .await
        .context("eth_sendTransaction for deploy failed")?;

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
        .call(TransactionRequest::default()
            .to(pair)
            .input(Bytes::from(getReservesCall {}.abi_encode()).into()))
        .await?;
    if result.len() < 96 {
        return Ok((0, 0));
    }
    let decoded = getReservesCall::abi_decode_returns(&result)?;
    Ok((decoded.reserve0.to::<u128>(), decoded.reserve1.to::<u128>()))
}

async fn call_token0<P: Provider>(provider: &P, pair: Address) -> Result<Address> {
    let result = provider
        .call(TransactionRequest::default()
            .to(pair)
            .input(Bytes::from(token0Call {}.abi_encode()).into()))
        .await?;
    Ok(token0Call::abi_decode_returns(&result)?)
}

async fn call_balance_of<P: Provider>(provider: &P, token: Address, account: Address) -> Result<U256> {
    let result = provider
        .call(TransactionRequest::default()
            .to(token)
            .input(Bytes::from(balanceOfCall { account }.abi_encode()).into()))
        .await?;
    Ok(balanceOfCall::abi_decode_returns(&result)?)
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
    format!("{}.{:03} ETH", meth / U256::from(1000u64), meth % U256::from(1000u64))
}

fn format_pnl(v: I256) -> String {
    if v >= I256::ZERO {
        format!("+{}", format_eth(U256::try_from(v).unwrap_or_default()))
    } else {
        format!("-{}", format_eth(U256::try_from(-v).unwrap_or_default()))
    }
}

// ─── Display ──────────────────────────────────────────────────────────────────
fn print_result(r: &BacktestResult) {
    println!("\n╔══════════════════════════════════════════════════╗");
    println!("║        MEV Sniper Backtest — Final Results       ║");
    println!("╠══════════════════════════════════════════════════╣");
    println!("║ Fork block:  {:>36} ║", r.fork_block);
    println!("║ Token:  {}… ║", &format!("{}", r.token_address)[..20]);
    println!("║ Pair:   {}… ║", &format!("{}", r.pair_address)[..20]);
    println!("╠══════════════════════════════════════════════════╣");

    if r.honeypot_detected {
        println!("║  HONEYPOT — tx reverted, capital protected       ║");
        println!("║  Gas cost: {:>39} ║", format_eth(r.gas_cost_eth));
    } else {
        println!("║ ETH invested:  {:>33} ║", format_eth(r.eth_invested));
        println!("║ Tokens bought: {:>33} ║", r.tokens_bought);
        println!("║ ETH from sell: {:>33} ║", format_eth(r.sell_eth_received));
        println!("║ Gas cost:      {:>33} ║", format_eth(r.gas_cost_eth));
        println!("║ Bribe paid:    {:>33} ║", format_eth(r.bribe_paid));

        if r.replayed_txs > 0 || r.failed_txs > 0 {
            println!("╠══════════════════════════════════════════════════╣");
            println!("║ Real txs applied: {:>30} ║", r.replayed_txs);
            println!("║ Real txs skipped: {:>30} ║", r.failed_txs);
            println!("║ Price move (WETH reserve): {:+>22.2}% ║", r.price_impact_pct);
        }

        println!("╠══════════════════════════════════════════════════╣");
        println!("║ Gross PnL: {:>37} ║", format_pnl(r.gross_pnl));
        println!("║ Net PnL:   {:>37} ║", format_pnl(r.net_pnl));
        println!("╠══════════════════════════════════════════════════╣");

        if r.net_pnl > I256::ZERO {
            println!("║  PROFITABLE ✓                                    ║");
        } else {
            println!("║  LOSS (no real buyers in window)                 ║");
        }
    }

    println!("╚══════════════════════════════════════════════════╝\n");
}
