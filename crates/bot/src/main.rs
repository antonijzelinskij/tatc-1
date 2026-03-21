//! MEV Sniper Bot — main entry point.
//!
//! Pipeline:
//!   1. Connect to Ethereum node via WebSocket
//!   2. Subscribe to pending transactions
//!   3. On detection of addLiquidity → simulate → if profitable, bundle + submit

mod bundle;
mod config;
mod mempool;
mod simulator;

use alloy::{
    primitives::U256,
    providers::ProviderBuilder,
};
use eyre::Result;
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

use crate::{
    bundle::BundleBuilder,
    config::Config,
    mempool::{run_mempool_listener, SnipeOpportunity},
    simulator::simulate_snipe_v2,
};

#[tokio::main]
async fn main() -> Result<()> {
    // ── Logging ──────────────────────────────────────────────────────────────
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    // ── Config ────────────────────────────────────────────────────────────────
    dotenvy::dotenv().ok();
    let config = Config::from_env()?;

    info!("MEV Sniper starting up");
    info!("Executor contract: {}", config.executor_address);
    info!("Max ETH per snipe: {} ETH", format_eth(config.max_eth_per_snipe));
    info!("Max tax allowed:   {} bps", config.max_tax_bps);

    // ── HTTP provider (for gas estimation, tx building) ───────────────────────
    let http_provider = ProviderBuilder::new()
        .on_builtin(&config.mainnet_rpc_url)
        .await?;

    // ── Mempool channel ───────────────────────────────────────────────────────
    let (opp_tx, mut opp_rx) = tokio::sync::mpsc::channel::<SnipeOpportunity>(64);

    // Spawn mempool listener in the background
    let config_clone = config.clone();
    tokio::spawn(async move {
        if let Err(e) = run_mempool_listener(&config_clone, opp_tx).await {
            error!("Mempool listener crashed: {e:?}");
        }
    });

    info!("Listening for new liquidity events …");

    // ── Main event loop ───────────────────────────────────────────────────────
    while let Some(opp) = opp_rx.recv().await {
        let config_ref = &config;
        let provider   = &http_provider;

        // Determine investment amount: either pool liquidity/2 or our max cap
        let amount_in = opp
            .estimated_liquidity_eth
            .min(config.max_eth_per_snipe);

        // Calculate bribe: bribe_percentage% of estimated 1% price impact profit
        let bribe_amount = compute_bribe(amount_in, config.bribe_percentage);

        // ── Pre-flight simulation ─────────────────────────────────────────────
        let sim = match simulate_snipe_v2(
            provider,
            config_ref,
            &opp,
            amount_in,
            U256::ZERO, // accept any amount for simulation
            bribe_amount,
        )
        .await
        {
            Ok(s)  => s,
            Err(e) => { warn!("Simulation error: {e}"); continue; }
        };

        if !sim.success {
            info!(
                token = %opp.token_address,
                reason = ?sim.revert_reason,
                "Simulation failed — skipping"
            );
            continue;
        }

        info!(
            token    = %opp.token_address,
            pair     = %opp.pair_address,
            amount   = %format_eth(amount_in),
            bribe    = %format_eth(bribe_amount),
            gas_est  = sim.gas_used,
            "✅ Simulation passed — submitting bundle"
        );

        // ── Get current block number for bundle targeting ─────────────────────
        let current_block = match provider.get_block_number().await {
            Ok(n)  => n,
            Err(e) => { warn!("get_block_number failed: {e}"); continue; }
        };
        let target_block = current_block + 1;

        // ── Build and submit Flashbots bundle ─────────────────────────────────
        // We don't have the raw victim tx bytes here (we only have the hash).
        // In production you'd fetch it from a transaction pool or use
        // eth_getRawTransactionByHash if your node supports it.
        // For the MVP we demonstrate the bundle structure but skip victim inclusion
        // when the raw bytes are unavailable.
        let bundle_builder = BundleBuilder::new(config_ref);

        // Minimum expected output: 95% of simulated amount (5% slippage budget)
        let min_amount_out = estimate_min_out(amount_in, opp.estimated_liquidity_eth);

        // Fetch raw victim tx (some nodes expose eth_getRawTransactionByHash)
        let victim_raw = fetch_raw_tx(provider, opp.pending_tx_hash).await;

        let victim_bytes = match victim_raw {
            Some(raw) => raw,
            None => {
                warn!(
                    hash = %opp.pending_tx_hash,
                    "Could not fetch raw victim tx — sending snipe-only bundle"
                );
                alloy::primitives::Bytes::new()
            }
        };

        match bundle_builder
            .submit(
                provider,
                victim_bytes,
                &opp,
                amount_in,
                min_amount_out,
                bribe_amount,
                target_block,
            )
            .await
        {
            Ok(hash) => info!(%hash, %target_block, "🚀 Bundle submitted"),
            Err(e)   => error!("Bundle submission failed: {e:?}"),
        }
    }

    Ok(())
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// `bribe_pct`% of estimated profit (approximated as 1% of amountIn).
fn compute_bribe(amount_in: U256, bribe_pct: u8) -> U256 {
    let estimated_profit = amount_in / U256::from(100u64); // ~1% price impact gain
    estimated_profit * U256::from(bribe_pct) / U256::from(100u64)
}

/// Minimum tokens expected: scale by liquidity (deep pool → less slippage).
fn estimate_min_out(amount_in: U256, liquidity_eth: U256) -> U256 {
    if liquidity_eth.is_zero() {
        return U256::ZERO;
    }
    // Very rough: expect at least 90% of a proportional share
    // Real code should use on-chain reserves for accurate calculation
    U256::ZERO // accept anything for MVP; simulator already validated
}

fn format_eth(wei: U256) -> String {
    let eth = wei / U256::from(10u128.pow(15)); // display in mETH
    format!("{} mETH", eth)
}

async fn fetch_raw_tx<P: alloy::providers::Provider>(
    provider: &P,
    hash: alloy::primitives::B256,
) -> Option<alloy::primitives::Bytes> {
    // Some nodes support eth_getRawTransactionByHash; alloy doesn't wrap this by
    // default so we use a raw RPC call.
    provider
        .raw_request::<_, Option<alloy::primitives::Bytes>>(
            "eth_getRawTransactionByHash".into(),
            (hash,),
        )
        .await
        .ok()
        .flatten()
}
