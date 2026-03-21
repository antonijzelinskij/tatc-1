// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "forge-std/Script.sol";
import {MevExecutor} from "../src/MevExecutor.sol";

/// @notice Deployment script for MevExecutor.
///
/// Usage:
///   forge script script/Deploy.s.sol \
///     --rpc-url $MAINNET_RPC_URL     \
///     --private-key $PRIVATE_KEY     \
///     --broadcast                    \
///     --verify                       \
///     --etherscan-api-key $ETHERSCAN_API_KEY
contract DeployMevExecutor is Script {
    function run() external returns (MevExecutor executor) {
        uint256 initialBalance = 0.1 ether; // seed the contract with ETH

        vm.startBroadcast();
        executor = new MevExecutor{value: initialBalance}();
        vm.stopBroadcast();

        console.log("MevExecutor deployed at:", address(executor));
        console.log("Owner:                  ", executor.owner());
        console.log("Initial ETH balance:    ", address(executor).balance);
    }
}
