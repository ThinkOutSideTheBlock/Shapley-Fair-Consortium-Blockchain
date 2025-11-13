#!/usr/bin/env python3
"""
Smart Contract Deployment Script for Shapley Allocation System

Deploys and configures the complete on-chain infrastructure.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any
from web3 import Web3
from eth_account import Account
from dotenv import load_dotenv
import click

# Load environment variables
load_dotenv()

# Contract ABIs and Bytecodes (simplified - normally loaded from compiled artifacts)
CONTRACTS_DIR = Path(__file__).parent.parent / "artifacts"


class ShapleyDeployer:
    """Deploy and configure Shapley allocation contracts"""

    def __init__(
        self,
        web3: Web3,
        account: Account,
        network: str = "localhost"
    ):
        self.w3 = web3
        self.account = account
        self.network = network
        self.deployed_contracts = {}
        self.quorum_threshold = 3  # Default quorum threshold

    def load_contract(self, name: str) -> Dict[str, Any]:
        """Load contract ABI and bytecode"""
        # Look for the artifact file in the subdirectories
        for root, dirs, files in os.walk(CONTRACTS_DIR):
            for file in files:
                if file == f"{name}.json" and "dbg.json" not in file:
                    artifact_path = os.path.join(root, file)
                    with open(artifact_path, 'r') as f:
                        return json.load(f)

        # If not found, raise an error
        raise FileNotFoundError(
            f"Contract artifact for {name} not found in {CONTRACTS_DIR}")

    def deploy_contract(
        self,
        contract_name: str,
        constructor_args: list = None
    ) -> str:
        """Deploy a single contract"""
        click.echo(f"📦 Deploying {contract_name}...")

        # Load artifact
        artifact = self.load_contract(contract_name)

        # Create contract instance
        Contract = self.w3.eth.contract(
            abi=artifact['abi'],
            bytecode=artifact['bytecode']
        )

        # Build constructor transaction
        constructor_args = constructor_args or []
        construct_txn = Contract.constructor(*constructor_args).build_transaction({
            'from': self.account.address,
            'nonce': self.w3.eth.get_transaction_count(self.account.address),
            'gas': 3000000,
            'gasPrice': self.w3.eth.gas_price
        })

        # Sign and send transaction
        signed_txn = self.account.sign_transaction(construct_txn)
        tx_hash = self.w3.eth.send_raw_transaction(signed_txn.raw_transaction)

        # Wait for receipt
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        contract_address = tx_receipt.contractAddress

        click.echo(f"✅ {contract_name} deployed at: {contract_address}")

        # Store deployment info
        self.deployed_contracts[contract_name] = {
            'address': contract_address,
            'abi': artifact['abi'],
            'tx_hash': tx_hash.hex(),
            'block': tx_receipt.blockNumber
        }

        return contract_address

    def deploy_system(self, token_address: str = None) -> Dict[str, str]:
        """Deploy complete Shapley system"""
        click.echo("🚀 Starting Shapley System Deployment\n")

        # 1. Deploy ContributionRegistry
        registry_address = self.deploy_contract("ContributionRegistry")

        # 2. Deploy ShapleyOracle with quorum threshold
        # ✅ FIX: Add required constructor argument
        oracle_address = self.deploy_contract(
            "ShapleyOracle",
            [self.quorum_threshold]  # quorum_threshold parameter
        )

        # 3. Deploy token if not provided
        if not token_address:
            # Deploy mock token for testing
            token_address = self.deploy_contract(
                "MockERC20",
                [
                    "Shapley Reward Token",  # name
                    "SRT",                   # symbol
                    18,                      # decimals
                    Web3.to_wei(1000000, 'ether')  # total supply
                ]
            )

        # 4. Deploy AllocationSettlement
        settlement_address = self.deploy_contract(
            "AllocationSettlement",
            [registry_address, oracle_address, token_address]
        )

        click.echo("\n📋 Deployment Summary:")
        click.echo(f"  ContributionRegistry: {registry_address}")
        click.echo(
            f"  ShapleyOracle: {oracle_address} (quorum: {self.quorum_threshold})")
        click.echo(f"  AllocationSettlement: {settlement_address}")
        click.echo(f"  RewardToken: {token_address}")

        return self.deployed_contracts

    def configure_system(
        self,
        consortium_members: list = None,
        oracle_addresses: list = None
    ):
        """Configure the deployed system"""
        click.echo("\n⚙️ Configuring System...")

        # Get contract instances
        registry = self.w3.eth.contract(
            address=self.deployed_contracts['ContributionRegistry']['address'],
            abi=self.deployed_contracts['ContributionRegistry']['abi']
        )

        oracle = self.w3.eth.contract(
            address=self.deployed_contracts['ShapleyOracle']['address'],
            abi=self.deployed_contracts['ShapleyOracle']['abi']
        )

        settlement = self.w3.eth.contract(
            address=self.deployed_contracts['AllocationSettlement']['address'],
            abi=self.deployed_contracts['AllocationSettlement']['abi']
        )

        # Add consortium members
        if consortium_members:
            for member in consortium_members:
                try:
                    tx_hash = registry.functions.addConsortiumMember(member).transact({
                        'from': self.account.address,
                        'gas': 200000
                    })
                    self.w3.eth.wait_for_transaction_receipt(tx_hash)
                    click.echo(f"  ✓ Added consortium member: {member}")
                except Exception as e:
                    click.echo(
                        f"  ⚠️ Failed to add member {member}: {str(e)[:50]}")

        # Add oracle addresses
        if oracle_addresses:
            for oracle_addr in oracle_addresses:
                try:
                    tx_hash = oracle.functions.addOracle(oracle_addr).transact({
                        'from': self.account.address,
                        'gas': 200000
                    })
                    self.w3.eth.wait_for_transaction_receipt(tx_hash)
                    click.echo(f"  ✓ Added oracle: {oracle_addr}")
                except Exception as e:
                    click.echo(
                        f"  ⚠️ Failed to add oracle {oracle_addr}: {str(e)[:50]}")

        click.echo("\n✨ System configured successfully!")

    def save_deployment(self, output_dir: Path):
        """Save deployment information"""
        output_dir.mkdir(parents=True, exist_ok=True)

        deployment_file = output_dir / f"{self.network}_latest.json"

        deployment_data = {
            'network': self.network,
            'deployer': self.account.address,
            'block_number': self.w3.eth.block_number,
            'timestamp': self.w3.eth.get_block('latest')['timestamp'],
            'quorum_threshold': self.quorum_threshold,
            'contracts': self.deployed_contracts
        }

        with open(deployment_file, 'w') as f:
            json.dump(deployment_data, f, indent=2)

        click.echo(f"\n💾 Deployment saved to: {deployment_file}")

        # Also save as timestamped backup
        backup_file = output_dir / \
            f"deployment_{self.network}_{self.w3.eth.block_number}.json"
        with open(backup_file, 'w') as f:
            json.dump(deployment_data, f, indent=2)


@click.command()
@click.option('--network', default='localhost', help='Network to deploy to')
@click.option('--private-key', envvar='PRIVATE_KEY', help='Deployer private key')
@click.option('--rpc-url', envvar='RPC_URL', default='http://127.0.0.1:8545', help='RPC URL')
@click.option('--token-address', help='Existing token address (optional)')
@click.option('--quorum', type=int, default=3, help='Oracle quorum threshold')
def deploy(network, private_key, rpc_url, token_address, quorum):
    """Deploy Shapley allocation contracts"""

    # Use default private key if not provided (for localhost testing)
    if not private_key and network == 'localhost':
        # Hardhat default account #0
        private_key = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
        click.echo("🔧 Using default localhost private key")

    if not private_key:
        click.echo("❌ Private key required")
        return

    # Connect to network
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        click.echo(f"❌ Failed to connect to {rpc_url}")
        return

    # Setup account
    account = Account.from_key(private_key)
    balance_wei = w3.eth.get_balance(account.address)
    balance_eth = w3.from_wei(balance_wei, 'ether')

    click.echo(f"🔑 Deploying from: {account.address}")
    click.echo(f"💰 Balance: {balance_eth:.4f} ETH")
    click.echo(f"🌐 Network: {w3.eth.chain_id}")
    click.echo(f"📦 Block: {w3.eth.block_number}")

    if balance_eth < 0.1:
        click.echo("⚠️ Low balance - deployment may fail")

    # Deploy contracts
    deployer = ShapleyDeployer(w3, account, network)

    # Update quorum in deployer before deployment
    deployer.quorum_threshold = quorum

    deployer.deploy_system(token_address)

    # Configure with test addresses if localhost
    if network == 'localhost':
        # Use Hardhat test accounts
        test_members = [
            "0x70997970C51812dc3A010C7d01b50e0d17dc79C8",  # Account #1
            "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC",  # Account #2
            "0x90F79bf6EB2c4f870365E785982E1f101E93b906",  # Account #3
        ]

        # Use first few accounts as oracles
        test_oracles = [
            account.address,  # Deployer
            "0x70997970C51812dc3A010C7d01b50e0d17dc79C8",  # Account #1
            "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC",  # Account #2
        ]

        click.echo(
            f"\n🧪 Configuring for localhost with {len(test_oracles)} oracles...")
        deployer.configure_system(test_members, test_oracles)

    # Save deployment
    deployer.save_deployment(Path("deployments"))

    click.echo(f"\n🎉 Deployment complete!")
    click.echo(f"📝 Use this deployment file with integrate.py:")
    click.echo(f"   --deployment deployments/{network}_latest.json")


if __name__ == "__main__":
    deploy()
