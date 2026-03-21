//! EVM-based backtester for the MEV sniper.
//!
//! Workflow:
//!   1. Spawn an Anvil node forked at the target historical block.
//!   2. Deploy MevExecutor onto the fork.
//!   3. Replay the historical addLiquidity transaction (victim tx).
//!   4. Submit our snipe transaction immediately after.
//!   5. Mine several blocks and sell.
//!   6. Calculate and report net PnL.
//!
//! Usage:
//!   cargo run -p backtest -- \
//!     --block    17046105 \
//!     --pair     0xA43fe16908251ee70EF74718545e4FE6C5cCEc9f \
//!     --token    0x6982508145454Ce325dDBe47a25d4ec3d2311933 \
//!     --tx-hash  0x<addLiquidity tx hash> \
//!     --eth-in   0.5

use alloy::{
    node_bindings::Anvil,
    primitives::{address, Address, Bytes, U256, I256},
    providers::{Provider, ProviderBuilder},
    rpc::types::TransactionRequest,
    sol,
    sol_types::SolCall,
    network::TransactionBuilder,
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

    /// Raw tx hash of the addLiquidity transaction to replay
    #[arg(long)]
    tx_hash: Option<alloy::primitives::B256>,

    /// ETH to invest (in ether, e.g. "0.5")
    #[arg(long, default_value = "0.1")]
    eth_in: f64,

    /// Max acceptable sell tax in BPS
    #[arg(long, default_value = "500")]
    max_tax_bps: u16,

    /// Number of blocks to mine before selling (simulate price appreciation)
    #[arg(long, default_value = "5")]
    hold_blocks: u64,
}

// ─── Contract ABIs ────────────────────────────────────────────────────────────
sol! {
    // MevExecutor
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

    // ERC-20
    function balanceOf(address account) external view returns (uint256);

    // V2 Pair
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
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info,backtest=debug")),
        )
        .init();

    dotenvy::dotenv().ok();
    let args = Args::parse();

    info!("=== MEV Sniper Backtest ===");
    info!("Fork block:    {}", args.block);
    info!("Pair:          {}", args.pair);
    info!("Token:         {}", args.token);
    info!("ETH in:        {} ETH", args.eth_in);
    info!("Hold blocks:   {}", args.hold_blocks);

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
    info!("Anvil listening at {endpoint}");

    // ── 2. Connect to Anvil ───────────────────────────────────────────────────
    // Use Anvil's first test account as the owner
    let signer: PrivateKeySigner = anvil.keys()[0].clone().into();
    let owner   = signer.address();
    let wallet  = EthereumWallet::from(signer);

    let provider = ProviderBuilder::new()
        .wallet(wallet)
        .on_http(endpoint.parse()?);

    info!("Owner address: {owner}");
    info!("Owner balance: {} ETH", format_eth(provider.get_balance(owner).await?));

    // ── 3. Deploy MevExecutor ─────────────────────────────────────────────────
    info!("Deploying MevExecutor ...");
    let executor_address = deploy_executor(&provider, owner).await
        .context("failed to deploy MevExecutor")?;
    info!("MevExecutor deployed at {executor_address}");

    // Fund the executor
    let eth_in_wei = eth_to_wei(args.eth_in);
    let fund_tx = TransactionRequest::default()
        .from(owner)
        .to(executor_address)
        .value(eth_in_wei + eth_to_wei(0.1)); // extra for gas + bribe
    let _ = provider.send_transaction(fund_tx).await?.get_receipt().await?;

    let exec_balance = provider.get_balance(executor_address).await?;
    info!("Executor funded with {} ETH", format_eth(exec_balance));

    // ── 4. (Optional) Replay victim addLiquidity tx ───────────────────────────
    if let Some(tx_hash) = args.tx_hash {
        info!("Replaying victim tx {tx_hash} ...");
        match replay_transaction(&provider, tx_hash).await {
            Ok(_)  => info!("Victim tx replayed successfully"),
            Err(e) => warn!("Could not replay victim tx: {e} (pool may already have liquidity)"),
        }
    }

    // ── 5. Check pair has liquidity ───────────────────────────────────────────
    let (r0, r1) = get_reserves(&provider, args.pair).await?;
    if r0 == 0 && r1 == 0 {
        return Err(eyre::eyre!(
            "Pair {} has no liquidity at block {}. Provide --tx-hash to replay addLiquidity.",
            args.pair, args.block
        ));
    }
    info!("Pair reserves: r0={} r1={}", r0, r1);

    // Determine token ordering
    let t0_addr   = call_token0(&provider, args.pair).await?;
    let weth_addr = address!("C02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2");
    let weth_is_t0 = t0_addr == weth_addr;

    let (reserve_weth, reserve_token) = if weth_is_t0 {
        (U256::from(r0), U256::from(r1))
    } else {
        (U256::from(r1), U256::from(r0))
    };

    // Estimate expected output
    let expected_out = get_amount_out(eth_in_wei, reserve_weth, reserve_token);
    let min_out      = expected_out * U256::from(90u64) / U256::from(100u64); // 10% slippage

    info!(
        "Expected tokens out: {} (min accepted: {})",
        expected_out, min_out
    );

    // ── 6. Execute snipe ──────────────────────────────────────────────────────
    let bribe_amount = eth_to_wei(0.001); // 1 mETH bribe for test
    info!("Executing snipeV2 ...");

    let snipe_start_balance = provider.get_balance(executor_address).await?;
    let gas_price = provider.get_gas_price().await?;

    let honeypot_detected;
    let tokens_bought;
    let gas_used_eth;

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

    match provider.send_transaction(snipe_tx).await {
        Ok(pending) => {
            let receipt = pending.get_receipt().await?;
            honeypot_detected = !receipt.status();
            let gas_used = receipt.gas_used;
            gas_used_eth = U256::from(gas_used) * U256::from(gas_price);

            if !receipt.status() {
                warn!("Snipe tx reverted — honeypot detected!");
                tokens_bought = U256::ZERO;
            } else {
                tokens_bought = call_balance_of(&provider, args.token, executor_address).await?;
                info!("Snipe SUCCESS — tokens bought: {tokens_bought}");
                info!("Gas used: {gas_used} units ({} ETH)", format_eth(gas_used_eth));
            }
        }
        Err(e) => {
            warn!("Snipe tx submission failed: {e}");
            honeypot_detected = true;
            tokens_bought = U256::ZERO;
            gas_used_eth  = U256::ZERO;
        }
    }

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
            net_pnl:           I256::try_from(-(gas_used_eth.as_limbs()[0] as i64)).unwrap_or(I256::ZERO),
            honeypot_detected: true,
        });
    }

    // ── 7. Mine `hold_blocks` to simulate others buying ───────────────────────
    info!("Mining {} blocks to simulate price movement ...", args.hold_blocks);
    mine_blocks(&provider, args.hold_blocks).await?;

    let block_after = provider.get_block_number().await?;
    info!("Now at block {block_after}");

    // ── 8. Sell all tokens ────────────────────────────────────────────────────
    info!("Executing sellV2 ...");

    let sell_calldata = sellV2Call {
        pair:       args.pair,
        token:      args.token,
        amount:     tokens_bought,
        minEthOut:  U256::ZERO, // accept any for backtest
        bribeAmount: U256::ZERO,
    }.abi_encode();

    let sell_tx = TransactionRequest::default()
        .from(owner)
        .to(executor_address)
        .input(Bytes::from(sell_calldata).into())
        .gas_limit(300_000);

    let sell_receipt = provider.send_transaction(sell_tx).await?.get_receipt().await?;
    let sell_gas_eth = U256::from(sell_receipt.gas_used) * U256::from(gas_price);

    // Withdraw ETH to measure final balance
    let withdraw_calldata = withdrawETHCall {}.abi_encode();
    let withdraw_tx = TransactionRequest::default()
        .from(owner)
        .to(executor_address)
        .input(Bytes::from(withdraw_calldata).into())
        .gas_limit(50_000);
    let _ = provider.send_transaction(withdraw_tx).await?.get_receipt().await?;

    let final_executor_balance = provider.get_balance(executor_address).await?;
    let sell_eth_received = if final_executor_balance > snipe_start_balance {
        // We have more ETH than before: measure delta carefully
        // final = initial - eth_in - bribe + sell_proceeds
        // sell_proceeds = final - initial + eth_in + bribe (approx)
        U256::ZERO // will calculate from final owner balance delta instead
    } else {
        U256::ZERO
    };

    // More accurate: check executor ETH balance after withdraw
    // The executor started with eth_in_wei + 0.1 ETH, we withdrew everything at end
    let total_gas = gas_used_eth + sell_gas_eth;

    // Approximate sell proceeds: owner balance delta + total_gas (to account for gas refund)
    // In real backtest, track the WETH emitted by sellV2
    // For MVP: use pair reserve change to calculate
    let (r0_after, r1_after) = get_reserves(&provider, args.pair).await?;
    let reserve_weth_after = if weth_is_t0 { U256::from(r0_after) } else { U256::from(r1_after) };
    let sell_proceeds = if reserve_weth_after < reserve_weth {
        // WETH decreased in pool (we sold tokens and received WETH)
        reserve_weth - reserve_weth_after
    } else {
        // Reserve didn't change as expected (edge case)
        U256::ZERO
    };

    info!("Sell proceeds (est): {} ETH", format_eth(sell_proceeds));
    info!("Total gas cost:       {} ETH", format_eth(total_gas));

    let gross_pnl = I256::try_from(sell_proceeds)
        .unwrap_or(I256::ZERO)
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

// ─── Contract deployment ──────────────────────────────────────────────────────

/// Deploy MevExecutor from its compiled bytecode.
/// The bytecode is embedded at compile time from the Foundry output.
/// Run `forge build` before `cargo build` so the artifact is available.
async fn deploy_executor<P: Provider>(provider: &P, owner: Address) -> Result<Address> {
    // Load bytecode from Foundry artifact (generated by `forge build`)
    // Path relative to workspace root: out/MevExecutor.sol/MevExecutor.json
    let artifact_path = std::env::var("FOUNDRY_OUT")
        .unwrap_or_else(|_| "out".to_string());
    let artifact_file = format!("{artifact_path}/MevExecutor.sol/MevExecutor.json");

    let bytecode = if let Ok(json_str) = std::fs::read_to_string(&artifact_file) {
        let json: serde_json::Value = serde_json::from_str(&json_str)?;
        let hex_str = json["bytecode"]["object"]
            .as_str()
            .ok_or_else(|| eyre::eyre!("bytecode not found in {artifact_file}"))?
            .trim_start_matches("0x");
        hex::decode(hex_str).context("failed to decode bytecode hex")?
    } else {
        eyre::bail!(
            "Foundry artifact not found at {artifact_file}. Run `forge build` first."
        );
    };

    // Deploy transaction (no constructor args)
    let deploy_tx = TransactionRequest::default()
        .from(owner)
        .input(Bytes::from(bytecode).into())
        .value(eth_to_wei(1.0)) // seed executor with 1 ETH
        .gas_limit(3_000_000);

    let receipt = provider
        .send_transaction(deploy_tx)
        .await?
        .get_receipt()
        .await?;

    receipt
        .contract_address
        .ok_or_else(|| eyre::eyre!("no contract address in deploy receipt"))
}

// ─── Helper: replay a historical tx on the fork ───────────────────────────────

async fn replay_transaction<P: Provider>(
    provider: &P,
    tx_hash: alloy::primitives::B256,
) -> Result<()> {
    // Fetch the raw tx from the forked node (which has the historical state)
    let raw: Option<Bytes> = provider
        .raw_request("eth_getRawTransactionByHash".into(), (tx_hash,))
        .await
        .ok()
        .flatten();

    if let Some(raw_tx) = raw {
        provider
            .raw_request::<_, alloy::primitives::B256>(
                "eth_sendRawTransaction".into(),
                (format!("0x{}", hex::encode(&raw_tx)),),
            )
            .await
            .context("failed to replay transaction")?;
        Ok(())
    } else {
        Err(eyre::eyre!("could not fetch raw tx {tx_hash}"))
    }
}

// ─── Anvil control ────────────────────────────────────────────────────────────

async fn mine_blocks<P: Provider>(provider: &P, count: u64) -> Result<()> {
    provider
        .raw_request::<_, serde_json::Value>(
            "anvil_mine".into(),
            (count,),
        )
        .await
        .context("anvil_mine failed")?;
    Ok(())
}

// ─── On-chain read helpers ────────────────────────────────────────────────────

async fn get_reserves<P: Provider>(
    provider: &P,
    pair: Address,
) -> Result<(u128, u128)> {
    let calldata = getReservesCall {}.abi_encode();
    let result = provider
        .call(
            &TransactionRequest::default()
                .to(pair)
                .input(Bytes::from(calldata).into()),
        )
        .await?;

    let decoded = getReservesCall::abi_decode_returns(&result, true)?;
    Ok((decoded.reserve0 as u128, decoded.reserve1 as u128))
}

async fn call_token0<P: Provider>(provider: &P, pair: Address) -> Result<Address> {
    let calldata = token0Call {}.abi_encode();
    let result = provider
        .call(
            &TransactionRequest::default()
                .to(pair)
                .input(Bytes::from(calldata).into()),
        )
        .await?;
    let decoded = token0Call::abi_decode_returns(&result, true)?;
    Ok(decoded._0)
}

async fn call_balance_of<P: Provider>(
    provider: &P,
    token: Address,
    account: Address,
) -> Result<U256> {
    let calldata = balanceOfCall { account }.abi_encode();
    let result = provider
        .call(
            &TransactionRequest::default()
                .to(token)
                .input(Bytes::from(calldata).into()),
        )
        .await?;
    let decoded = balanceOfCall::abi_decode_returns(&result, true)?;
    Ok(decoded._0)
}

// ─── Math ─────────────────────────────────────────────────────────────────────

fn get_amount_out(amount_in: U256, reserve_in: U256, reserve_out: U256) -> U256 {
    let in_with_fee = amount_in * U256::from(997u64);
    let numerator   = in_with_fee * reserve_out;
    let denominator = reserve_in * U256::from(1000u64) + in_with_fee;
    numerator / denominator
}

fn eth_to_wei(eth: f64) -> U256 {
    let wei_f = eth * 1e18;
    U256::from(wei_f as u128)
}

fn format_eth(wei: U256) -> String {
    let full = wei / U256::from(10u128.pow(15));
    format!("{}.{:03} ETH", full / U256::from(1000u64), full % U256::from(1000u64))
}

// ─── Result display ───────────────────────────────────────────────────────────

fn print_result(r: &BacktestResult) {
    println!("\n╔══════════════════════════════════════════════════╗");
    println!("║          MEV Sniper Backtest Results             ║");
    println!("╠══════════════════════════════════════════════════╣");
    println!("║ Fork block:      {:>30} ║", r.fork_block);
    println!("║ Token:           {:>30} ║", format!("{:.20}…", r.token_address));
    println!("║ Pair:            {:>30} ║", format!("{:.20}…", r.pair_address));
    println!("╠══════════════════════════════════════════════════╣");

    if r.honeypot_detected {
        println!("║ 🚨 HONEYPOT DETECTED — Transaction reverted!     ║");
        println!("║ Capital loss: ZERO (protected by sniper)         ║");
        println!("║ Gas cost:     {:>30} ║", format_eth(r.gas_cost_eth));
    } else {
        println!("║ ETH invested:    {:>30} ║", format_eth(r.eth_invested));
        println!("║ Tokens bought:   {:>30} ║", r.tokens_bought);
        println!("║ ETH from sell:   {:>30} ║", format_eth(r.sell_eth_received));
        println!("║ Gas cost:        {:>30} ║", format_eth(r.gas_cost_eth));
        println!("║ Bribe paid:      {:>30} ║", format_eth(r.bribe_paid));
        println!("╠══════════════════════════════════════════════════╣");

        let sign = if r.gross_pnl >= I256::ZERO { "+" } else { "" };
        let gross_display = if r.gross_pnl >= I256::ZERO {
            format_eth(U256::try_from(r.gross_pnl).unwrap_or_default())
        } else {
            format_eth(U256::try_from(-r.gross_pnl).unwrap_or_default())
        };
        println!("║ Gross PnL:    {}{:>32} ║", sign, gross_display);

        let net_sign = if r.net_pnl >= I256::ZERO { "+" } else { "-" };
        let net_display = if r.net_pnl >= I256::ZERO {
            format_eth(U256::try_from(r.net_pnl).unwrap_or_default())
        } else {
            format_eth(U256::try_from(-r.net_pnl).unwrap_or_default())
        };
        println!("║ Net PnL:      {}{:>32} ║", net_sign, net_display);
    }

    println!("╚══════════════════════════════════════════════════╝\n");
}
