//! Mempool listener: subscribe to pending transactions and identify new
//! Uniswap V2 / V3 liquidity-addition events.

use alloy::{
    primitives::{address, Address, Bytes, B256, U256},
    providers::{Provider, WsConnect},
    pubsub::PubSubFrontend,
    rpc::types::Transaction,
};
use eyre::Result;
use futures_util::StreamExt;
use tracing::{debug, info, warn};

use crate::config::Config;

// ─── Known router / manager addresses ────────────────────────────────────────
const UNISWAP_V2_ROUTER: Address = address!("7a250d5630B4cF539739dF2C5dAcb4c659F2488D");
const UNISWAP_V2_FACTORY: Address = address!("5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f");
// V3 NonfungiblePositionManager
const V3_POSITION_MANAGER: Address = address!("C36442b4a4522E871399CD717aBDD847Ab11FE88");
// V3 Router
const V3_ROUTER: Address = address!("E592427A0AEce92De3Edee1F18E0157C05861564");
const WETH: Address = address!("C02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2");

// ─── Function selectors ───────────────────────────────────────────────────────
/// `addLiquidity(address,address,uint256,uint256,uint256,uint256,address,uint256)`
const ADD_LIQUIDITY_SEL: [u8; 4]     = [0xe8, 0xe3, 0x37, 0x00];
/// `addLiquidityETH(address,uint256,uint256,uint256,address,uint256)`
const ADD_LIQUIDITY_ETH_SEL: [u8; 4] = [0xf3, 0x05, 0xd7, 0x19];
/// `mint((address,address,uint24,int24,int24,uint256,uint256,uint256,uint256,address,uint256))`
const V3_MINT_SEL: [u8; 4]           = [0x88, 0x31, 0x64, 0x56];

// ─── Data types ───────────────────────────────────────────────────────────────

/// Describes an opportunity detected in the mempool.
#[derive(Debug, Clone)]
pub struct SnipeOpportunity {
    /// Target pair address (V2) or pool address (V3).
    pub pair_address: Address,
    /// Token we want to buy.
    pub token_address: Address,
    /// Estimated ETH value of liquidity being added.
    pub estimated_liquidity_eth: U256,
    /// Hash of the pending victim transaction.
    pub pending_tx_hash: B256,
    /// Protocol version.
    pub protocol: Protocol,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Protocol {
    UniswapV2,
    UniswapV3 { fee: u32 },
}

// ─── Mempool listener ─────────────────────────────────────────────────────────

/// Subscribe to pending transactions and yield snipe opportunities.
///
/// The caller receives opportunities through the returned `tokio::sync::mpsc`
/// channel and decides whether to act on them.
pub async fn run_mempool_listener(
    config: &Config,
    tx: tokio::sync::mpsc::Sender<SnipeOpportunity>,
) -> Result<()> {
    info!("Connecting to WebSocket: {}", config.mainnet_ws_url);

    let ws = WsConnect::new(config.mainnet_ws_url.clone());
    let provider = alloy::providers::ProviderBuilder::new()
        .on_ws(ws)
        .await?;

    info!("Subscribing to pending transactions …");
    let mut stream = provider.subscribe_pending_transactions().await?.into_stream();

    while let Some(tx_hash) = stream.next().await {
        // Fetch full transaction
        let Some(Ok(Some(tx))) = Some(provider.get_transaction_by_hash(tx_hash).await) else {
            continue;
        };

        if let Some(opp) = decode_opportunity(&tx, config).await {
            info!(
                hash   = %tx_hash,
                token  = %opp.token_address,
                pair   = %opp.pair_address,
                liq_eth = %opp.estimated_liquidity_eth,
                "🎯 Opportunity found"
            );
            let _ = tx.send(opp).await;
        }
    }

    Ok(())
}

/// Inspect a single pending transaction and return an opportunity if it's a
/// liquidity-addition call we care about.
async fn decode_opportunity(
    tx: &Transaction,
    config: &Config,
) -> Option<SnipeOpportunity> {
    let to = tx.to?;
    let input = tx.input.as_ref();

    if input.len() < 4 {
        return None;
    }

    let sel: [u8; 4] = input[..4].try_into().ok()?;

    // ── Uniswap V2: addLiquidityETH ──────────────────────────────────────────
    if to == UNISWAP_V2_ROUTER && sel == ADD_LIQUIDITY_ETH_SEL {
        let eth_value = tx.value;
        if eth_value < config.min_liquidity_eth {
            debug!("Skipping: liquidity below threshold ({eth_value})");
            return None;
        }

        // ABI decode: addLiquidityETH(token, amountTokenDesired, amountTokenMin,
        //              amountETHMin, to, deadline)
        // Offset 4 bytes selector + 12 bytes padding + 20 bytes address
        let token_addr = decode_address_from_calldata(input, 4)?;

        // Derive pair address via CREATE2 (cheaper than an RPC call)
        let pair_address = v2_pair_address(UNISWAP_V2_FACTORY, token_addr, WETH);

        return Some(SnipeOpportunity {
            pair_address,
            token_address: token_addr,
            estimated_liquidity_eth: eth_value,
            pending_tx_hash: tx.hash,
            protocol: Protocol::UniswapV2,
        });
    }

    // ── Uniswap V2: addLiquidity (token/token — less common, only if one is WETH)
    if to == UNISWAP_V2_ROUTER && sel == ADD_LIQUIDITY_SEL {
        let token_a = decode_address_from_calldata(input, 4)?;
        let token_b = decode_address_from_calldata(input, 36)?;

        let token_out = if token_a == WETH {
            token_b
        } else if token_b == WETH {
            token_a
        } else {
            return None; // not a WETH pair
        };

        let pair = v2_pair_address(UNISWAP_V2_FACTORY, token_out, WETH);
        let liq_eth = tx.value; // may be 0; estimate from amountBDesired if token_b == WETH
        if liq_eth < config.min_liquidity_eth {
            return None;
        }

        return Some(SnipeOpportunity {
            pair_address: pair,
            token_address: token_out,
            estimated_liquidity_eth: liq_eth,
            pending_tx_hash: tx.hash,
            protocol: Protocol::UniswapV2,
        });
    }

    // ── Uniswap V3: mint ─────────────────────────────────────────────────────
    if to == V3_POSITION_MANAGER && sel == V3_MINT_SEL {
        // MintParams struct; token0 at offset 4, token1 at 36, fee at 68
        let token0  = decode_address_from_calldata(input, 4)?;
        let token1  = decode_address_from_calldata(input, 36)?;
        let fee_raw = decode_u32_from_calldata(input, 68)?;

        let token_out = if token0 == WETH { token1 } else if token1 == WETH { token0 } else { return None; };

        // Pool address via CREATE2
        let pool = v3_pool_address(token0, token1, fee_raw);

        let eth_value = tx.value;
        if eth_value < config.min_liquidity_eth {
            return None;
        }

        return Some(SnipeOpportunity {
            pair_address: pool,
            token_address: token_out,
            estimated_liquidity_eth: eth_value,
            pending_tx_hash: tx.hash,
            protocol: Protocol::UniswapV3 { fee: fee_raw },
        });
    }

    None
}

// ─── ABI decode helpers ───────────────────────────────────────────────────────

fn decode_address_from_calldata(input: &[u8], offset: usize) -> Option<Address> {
    if input.len() < offset + 32 {
        return None;
    }
    // Address is right-padded in the last 20 bytes of the 32-byte word
    let word = &input[offset..offset + 32];
    Some(Address::from_slice(&word[12..]))
}

fn decode_u32_from_calldata(input: &[u8], offset: usize) -> Option<u32> {
    if input.len() < offset + 32 {
        return None;
    }
    let word = &input[offset..offset + 32];
    // u32 is in the last 4 bytes of the 32-byte word (big-endian)
    let bytes: [u8; 4] = word[28..32].try_into().ok()?;
    Some(u32::from_be_bytes(bytes))
}

// ─── CREATE2 pair/pool address computation ────────────────────────────────────

/// Compute the V2 pair address deterministically (avoids an RPC call).
fn v2_pair_address(factory: Address, token_a: Address, token_b: Address) -> Address {
    use sha3::{Digest, Keccak256};

    // Sort tokens (V2 always has token0 < token1)
    let (t0, t1) = if token_a < token_b { (token_a, token_b) } else { (token_b, token_a) };

    // keccak256(abi.encodePacked(token0, token1))
    let mut salt_input = [0u8; 40];
    salt_input[..20].copy_from_slice(t0.as_slice());
    salt_input[20..].copy_from_slice(t1.as_slice());
    let salt = Keccak256::digest(&salt_input);

    // V2 init code hash (mainnet)
    let init_code_hash =
        hex::decode("96e8ac4277198ff8b6f785478aa9a39f403cb768dd02cbee326c3e7da348845f")
            .expect("valid hex");

    // CREATE2: keccak256(0xff ++ factory ++ salt ++ init_code_hash)[12..]
    let mut preimage = Vec::with_capacity(85);
    preimage.push(0xff);
    preimage.extend_from_slice(factory.as_slice());
    preimage.extend_from_slice(&salt);
    preimage.extend_from_slice(&init_code_hash);

    let hash = Keccak256::digest(&preimage);
    Address::from_slice(&hash[12..])
}

/// Compute the V3 pool address deterministically.
fn v3_pool_address(token_a: Address, token_b: Address, fee: u32) -> Address {
    use sha3::{Digest, Keccak256};

    let (t0, t1) = if token_a < token_b { (token_a, token_b) } else { (token_b, token_a) };

    // abi.encode(token0, token1, fee) → 96 bytes
    let mut key = [0u8; 96];
    key[12..32].copy_from_slice(t0.as_slice());
    key[44..64].copy_from_slice(t1.as_slice());
    key[92..96].copy_from_slice(&fee.to_be_bytes());
    let salt = Keccak256::digest(&key);

    // V3 init code hash (mainnet)
    let init_code_hash =
        hex::decode("e34f199b19b2b4f47f68442619d555527d244f78a3297ea89325f843f87b8b54")
            .expect("valid hex");

    let factory = address!("1F98431c8aD98523631AE4a59f267346ea31F984");
    let mut preimage = Vec::with_capacity(85);
    preimage.push(0xff);
    preimage.extend_from_slice(factory.as_slice());
    preimage.extend_from_slice(&salt);
    preimage.extend_from_slice(&init_code_hash);

    let hash = Keccak256::digest(&preimage);
    Address::from_slice(&hash[12..])
}
