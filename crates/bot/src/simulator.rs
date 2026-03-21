//! Pre-flight simulation: call `eth_call` against our executor contract to
//! verify the snipe will succeed and measure expected gas usage.

use alloy::{
    network::TransactionBuilder,
    primitives::{Address, Bytes, U256},
    providers::Provider,
    rpc::types::{TransactionRequest, state::StateOverride},
    sol,
    sol_types::SolCall,
};
use eyre::Result;
use tracing::{debug, warn};

use crate::{config::Config, mempool::SnipeOpportunity};

// ─── ABI binding for MevExecutor.snipeV2 ─────────────────────────────────────
sol! {
    function snipeV2(
        address pair,
        address tokenOut,
        uint256 amountIn,
        uint256 minAmountOut,
        uint256 maxTaxBps,
        uint256 bribeAmount
    ) external payable;
}

// ─── Result types ─────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct SimResult {
    /// Whether the call would succeed.
    pub success: bool,
    /// Estimated gas units consumed.
    pub gas_used: u64,
    /// If failed, the decoded revert reason (best-effort).
    pub revert_reason: Option<String>,
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Simulate `snipeV2` via `eth_call` and return the result.
/// Uses a state override to ensure the executor has enough ETH balance.
pub async fn simulate_snipe_v2<P: Provider>(
    provider: &P,
    config: &Config,
    opportunity: &SnipeOpportunity,
    amount_in: U256,
    min_amount_out: U256,
    bribe_amount: U256,
) -> Result<SimResult> {
    let calldata = snipeV2Call {
        pair:         opportunity.pair_address,
        tokenOut:     opportunity.token_address,
        amountIn:     amount_in,
        minAmountOut: min_amount_out,
        maxTaxBps:    U256::from(config.max_tax_bps),
        bribeAmount:  bribe_amount,
    }
    .abi_encode();

    // Override: pretend the executor has 100 ETH so simulation never fails on balance
    let mut state_override = StateOverride::default();
    state_override.insert(
        config.executor_address,
        alloy::rpc::types::state::AccountOverride {
            balance: Some(U256::from(100u128) * U256::from(10u128).pow(U256::from(18u32))),
            ..Default::default()
        },
    );

    let tx = TransactionRequest::default()
        .to(config.executor_address)
        .input(Bytes::from(calldata).into())
        .value(amount_in + bribe_amount);

    debug!("Simulating snipeV2 for pair {}", opportunity.pair_address);

    match provider
        .call(&tx)
        .state(state_override)
        .await
    {
        Ok(_) => {
            // Estimate gas for the real submission
            let gas = provider
                .estimate_gas(&tx)
                .await
                .unwrap_or(300_000);

            Ok(SimResult { success: true, gas_used: gas, revert_reason: None })
        }
        Err(e) => {
            let reason = decode_revert_reason(e.to_string());
            warn!("Simulation failed: {:?}", reason);
            Ok(SimResult {
                success: false,
                gas_used: 0,
                revert_reason: Some(reason),
            })
        }
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn decode_revert_reason(err: String) -> String {
    // Try to extract a human-readable reason from the alloy error string.
    // Typically looks like: "execution reverted: <reason>" or hex-encoded data.
    if let Some(pos) = err.find("execution reverted") {
        return err[pos..].to_string();
    }
    err
}
