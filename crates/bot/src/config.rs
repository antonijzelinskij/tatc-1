use alloy::primitives::{Address, U256};
use eyre::{Context, Result};
use std::str::FromStr;

/// All configuration loaded from environment variables.
#[derive(Debug, Clone)]
pub struct Config {
    // ─── Node connectivity ────────────────────────────────────────────────────
    /// HTTP RPC endpoint (used for one-off calls and simulation).
    pub mainnet_rpc_url: String,
    /// WebSocket endpoint (used for mempool subscription).
    pub mainnet_ws_url: String,

    // ─── Credentials ─────────────────────────────────────────────────────────
    /// EOA private key (hex, with or without 0x prefix).
    pub private_key: String,

    // ─── Deployed contract ────────────────────────────────────────────────────
    /// Address of the deployed MevExecutor.
    pub executor_address: Address,

    // ─── Strategy parameters ──────────────────────────────────────────────────
    /// Maximum ETH to spend per snipe (in wei).
    pub max_eth_per_snipe: U256,
    /// Minimum ETH liquidity a pool must have before we care.
    pub min_liquidity_eth: U256,
    /// Maximum sell-tax we accept (basis points, e.g. 500 = 5%).
    pub max_tax_bps: u16,
    /// Percentage of estimated gross profit to offer as validator bribe (0-100).
    pub bribe_percentage: u8,

    // ─── Flashbots ───────────────────────────────────────────────────────────
    /// Flashbots relay URL.
    pub flashbots_relay_url: String,
    /// Separate private key used only for Flashbots signing (reputation key).
    pub flashbots_signer_key: String,
}

impl Config {
    /// Load configuration from the process environment.
    /// Call `dotenvy::dotenv().ok()` before this if you use a `.env` file.
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            mainnet_rpc_url: require_env("MAINNET_RPC_URL")?,
            mainnet_ws_url:  require_env("MAINNET_WS_URL")?,
            private_key:     require_env("PRIVATE_KEY")?,

            executor_address: Address::from_str(&require_env("EXECUTOR_ADDRESS")?)
                .context("EXECUTOR_ADDRESS is not a valid address")?,

            max_eth_per_snipe: U256::from_str(&require_env("MAX_ETH_PER_SNIPE")?)
                .context("MAX_ETH_PER_SNIPE must be a decimal wei value")?,

            min_liquidity_eth: U256::from_str(&require_env("MIN_LIQUIDITY_ETH")?)
                .context("MIN_LIQUIDITY_ETH must be a decimal wei value")?,

            max_tax_bps: require_env("MAX_TAX_BPS")?
                .parse::<u16>()
                .context("MAX_TAX_BPS must be 0-10000")?,

            bribe_percentage: require_env("BRIBE_PERCENTAGE")?
                .parse::<u8>()
                .context("BRIBE_PERCENTAGE must be 0-100")?,

            flashbots_relay_url: require_env("FLASHBOTS_RELAY_URL")
                .unwrap_or_else(|_| "https://relay.flashbots.net".to_string()),

            flashbots_signer_key: require_env("FLASHBOTS_SIGNER_KEY")?,
        })
    }

    /// Convenience: parse the private key into a k256 signing key.
    pub fn signing_key(&self) -> Result<k256::ecdsa::SigningKey> {
        let hex_str = self.private_key.trim_start_matches("0x");
        let bytes   = hex::decode(hex_str).context("PRIVATE_KEY is not valid hex")?;
        k256::ecdsa::SigningKey::from_bytes(bytes.as_slice().into())
            .context("PRIVATE_KEY is not a valid secp256k1 key")
    }

    pub fn flashbots_signing_key(&self) -> Result<k256::ecdsa::SigningKey> {
        let hex_str = self.flashbots_signer_key.trim_start_matches("0x");
        let bytes   = hex::decode(hex_str).context("FLASHBOTS_SIGNER_KEY is not valid hex")?;
        k256::ecdsa::SigningKey::from_bytes(bytes.as_slice().into())
            .context("FLASHBOTS_SIGNER_KEY is not a valid secp256k1 key")
    }
}

fn require_env(key: &str) -> Result<String> {
    std::env::var(key).with_context(|| format!("Missing environment variable: {key}"))
}
