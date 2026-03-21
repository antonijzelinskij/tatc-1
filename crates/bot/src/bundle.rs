//! Flashbots bundle builder and submitter.
//!
//! Bundle structure:
//!   [0] victim tx  (addLiquidity — included so our tx is guaranteed to follow it)
//!   [1] our snipe tx
//!
//! Reference: https://docs.flashbots.net/flashbots-auction/advanced/rpc-endpoint

use alloy::{
    network::{EthereumWallet, TransactionBuilder},
    primitives::{Address, Bytes, B256, U256},
    providers::Provider,
    rpc::types::{TransactionRequest, TransactionReceipt},
    signers::local::PrivateKeySigner,
    sol,
    sol_types::SolCall,
};
use eyre::{bail, Context, Result};
use k256::ecdsa::SigningKey;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sha3::{Digest, Keccak256};
use tracing::{debug, info, warn};

use crate::{config::Config, mempool::SnipeOpportunity};

// ─── ABI binding ─────────────────────────────────────────────────────────────
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

// ─── Flashbots RPC types ──────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct JsonRpcRequest<T: Serialize> {
    jsonrpc: &'static str,
    id:      u64,
    method:  &'static str,
    params:  T,
}

#[derive(Debug, Serialize)]
struct SendBundleParams {
    txs:          Vec<String>, // hex-encoded signed raw transactions
    #[serde(rename = "blockNumber")]
    block_number: String,      // hex block number e.g. "0x112a5c0"
    #[serde(rename = "minTimestamp", skip_serializing_if = "Option::is_none")]
    min_timestamp: Option<u64>,
    #[serde(rename = "maxTimestamp", skip_serializing_if = "Option::is_none")]
    max_timestamp: Option<u64>,
    #[serde(rename = "revertingTxHashes", skip_serializing_if = "Vec::is_empty")]
    reverting_tx_hashes: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct JsonRpcResponse {
    #[allow(dead_code)]
    id: u64,
    #[serde(default)]
    result: Option<serde_json::Value>,
    #[serde(default)]
    error: Option<serde_json::Value>,
}

// ─── Bundle builder ───────────────────────────────────────────────────────────

pub struct BundleBuilder<'a> {
    config:        &'a Config,
    http_client:   Client,
}

impl<'a> BundleBuilder<'a> {
    pub fn new(config: &'a Config) -> Self {
        Self {
            config,
            http_client: Client::new(),
        }
    }

    /// Build and submit a two-tx bundle:
    ///   [victim_raw_tx, our_snipe_tx]
    ///
    /// Returns the bundle hash if submission was accepted.
    pub async fn submit<P: Provider>(
        &self,
        provider: &P,
        victim_raw_tx: Bytes,
        opportunity: &SnipeOpportunity,
        amount_in: U256,
        min_amount_out: U256,
        bribe_amount: U256,
        target_block: u64,
    ) -> Result<B256> {
        // ── Build and sign our snipe tx ───────────────────────────────────────
        let our_raw_tx = self
            .build_snipe_tx(provider, opportunity, amount_in, min_amount_out, bribe_amount)
            .await
            .context("failed to build snipe tx")?;

        // ── Assemble bundle ───────────────────────────────────────────────────
        let txs = vec![
            format!("0x{}", hex::encode(&victim_raw_tx)),
            format!("0x{}", hex::encode(&our_raw_tx)),
        ];

        let block_hex = format!("0x{:x}", target_block);

        let params = SendBundleParams {
            txs,
            block_number: block_hex,
            min_timestamp: None,
            max_timestamp: Some(
                // Give a 12-second window (one block)
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
                    + 12,
            ),
            reverting_tx_hashes: vec![],
        };

        // ── Sign and send ─────────────────────────────────────────────────────
        let bundle_hash = self
            .send_to_relay(params)
            .await
            .context("flashbots relay submission failed")?;

        info!(%target_block, %bundle_hash, "Bundle submitted");
        Ok(bundle_hash)
    }

    // ─── Private helpers ──────────────────────────────────────────────────────

    async fn build_snipe_tx<P: Provider>(
        &self,
        provider: &P,
        opp: &SnipeOpportunity,
        amount_in: U256,
        min_amount_out: U256,
        bribe_amount: U256,
    ) -> Result<Bytes> {
        let signer = PrivateKeySigner::from_signing_key(self.config.signing_key()?);
        let wallet = EthereumWallet::from(signer.clone());
        let from   = signer.address();

        let calldata: Vec<u8> = snipeV2Call {
            pair:         opp.pair_address,
            tokenOut:     opp.token_address,
            amountIn:     amount_in,
            minAmountOut: min_amount_out,
            maxTaxBps:    U256::from(self.config.max_tax_bps),
            bribeAmount:  bribe_amount,
        }
        .abi_encode();

        let nonce    = provider.get_transaction_count(from).await?;
        let gas_price = provider.get_gas_price().await?;

        // Build EIP-1559 transaction
        let tx = TransactionRequest::default()
            .from(from)
            .to(self.config.executor_address)
            .input(Bytes::from(calldata).into())
            .value(amount_in + bribe_amount)
            .nonce(nonce)
            .gas_limit(350_000)
            // Priority fee = gas_price * 110% to beat typical txs
            .max_fee_per_gas(gas_price * 2)
            .max_priority_fee_per_gas(gas_price / 10);

        // Sign and encode
        let tx_envelope = tx.build(&wallet).await?;
        let mut encoded = Vec::new();
        alloy::rlp::Encodable::encode(&tx_envelope, &mut encoded);
        Ok(Bytes::from(encoded))
    }

    async fn send_to_relay(&self, params: SendBundleParams) -> Result<B256> {
        let body = serde_json::to_string(&JsonRpcRequest {
            jsonrpc: "2.0",
            id:      1,
            method:  "eth_sendBundle",
            params:  vec![serde_json::to_value(params)?],
        })?;

        // Compute X-Flashbots-Signature
        let signature = self.flashbots_sign(&body)?;

        debug!("POSTing bundle to {}", self.config.flashbots_relay_url);

        let resp = self
            .http_client
            .post(&self.config.flashbots_relay_url)
            .header("Content-Type", "application/json")
            .header("X-Flashbots-Signature", signature)
            .body(body)
            .send()
            .await
            .context("HTTP request to Flashbots relay failed")?;

        let status = resp.status();
        let text   = resp.text().await?;

        if !status.is_success() {
            bail!("Flashbots relay HTTP {status}: {text}");
        }

        let rpc_resp: JsonRpcResponse = serde_json::from_str(&text)
            .context("failed to parse Flashbots response")?;

        if let Some(err) = rpc_resp.error {
            bail!("Flashbots RPC error: {err}");
        }

        // Response contains { bundleHash: "0x..." }
        let hash_str = rpc_resp
            .result
            .and_then(|v| v.get("bundleHash").cloned())
            .and_then(|v| v.as_str().map(|s| s.to_string()))
            .unwrap_or_default();

        let bundle_hash = hash_str
            .trim_start_matches("0x")
            .parse::<B256>()
            .unwrap_or_default();

        Ok(bundle_hash)
    }

    /// Sign the body with the Flashbots signer key.
    /// Header format: `<EOA_address>:<signature_of_keccak256(body)>`
    fn flashbots_sign(&self, body: &str) -> Result<String> {
        use k256::ecdsa::{signature::Signer, Signature};

        let signing_key = self.config.flashbots_signing_key()?;
        let verifying_key = signing_key.verifying_key();

        // Derive the Ethereum address from the verifying key
        let pubkey_bytes = verifying_key.to_encoded_point(false);
        let pubkey_hash  = Keccak256::digest(&pubkey_bytes.as_bytes()[1..]);
        let address_hex  = hex::encode(&pubkey_hash[12..]);

        // Sign keccak256 of body (Ethereum personal_sign style)
        let body_hash    = Keccak256::digest(body.as_bytes());
        let message      = format!("\x19Ethereum Signed Message:\n32");
        let mut full_msg = Vec::new();
        full_msg.extend_from_slice(message.as_bytes());
        full_msg.extend_from_slice(&body_hash);
        let final_hash = Keccak256::digest(&full_msg);

        let (sig, recovery_id): (Signature, _) = signing_key
            .sign_prehash_recoverable(&final_hash)
            .context("signing failed")?;

        let mut sig_bytes = sig.to_bytes().to_vec();
        sig_bytes.push(recovery_id.to_byte() + 27);

        Ok(format!("0x{}:0x{}", address_hex, hex::encode(sig_bytes)))
    }
}
