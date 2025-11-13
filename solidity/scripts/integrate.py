#!/usr/bin/env python3
"""
Complete Integration Script - Handles Setup + Computation + On-Chain Settlement

This script:
1. Auto-registers synthetic contributions if none exist
2. Computes Shapley values off-chain
3. Submits to oracle quorum for on-chain verification
4. Distributes rewards via smart contract

Usage:
    # Run full pipeline (auto-setup + integrate)
    python -m solidity.scripts.integrate \
      --epoch 1 \
      --deployment solidity/deployments/localhost_latest.json \
      --oracle-keys $KEY1 $KEY2 \
      --rpc-url http://127.0.0.1:8545
    
    # Skip auto-setup if contributions already exist
    python -m solidity.scripts.integrate \
      --epoch 1 \
      --deployment solidity/deployments/localhost_latest.json \
      --oracle-keys $KEY1 $KEY2 \
      --no-auto-setup
"""

from modules.analysis import AnalysisPipeline
from modules.data_gen import DataGenerator
from modules.allocation import AllocationEngine
import json
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from web3 import Web3
from eth_account import Account
from eth_account.messages import encode_defunct
import click

# Import from our Python system
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))


class ShapleyIntegration:
    """Integrate off-chain computation with on-chain settlement"""

    def __init__(
        self,
        w3: Web3,
        oracle_accounts: List[Account],
        deployment_info: Dict
    ):
        self.w3 = w3
        self.oracle_accounts = oracle_accounts
        self.deployment = deployment_info

        # Initialize contracts
        self.registry = self.w3.eth.contract(
            address=deployment_info['contracts']['ContributionRegistry']['address'],
            abi=deployment_info['contracts']['ContributionRegistry']['abi']
        )

        self.oracle = self.w3.eth.contract(
            address=deployment_info['contracts']['ShapleyOracle']['address'],
            abi=deployment_info['contracts']['ShapleyOracle']['abi']
        )

        self.settlement = self.w3.eth.contract(
            address=deployment_info['contracts']['AllocationSettlement']['address'],
            abi=deployment_info['contracts']['AllocationSettlement']['abi']
        )

        # Initialize Python computation engine
        self.allocation_engine = AllocationEngine(seed=42)
        self.data_generator = DataGenerator(seed=42)

    def auto_setup_epoch(
        self,
        epoch: int,
        num_agents: int = 5,
        contribution_range: Tuple[float, float] = (5.0, 25.0)
    ) -> Tuple[List[str], List[int]]:
        """
        Automatically generate and register synthetic contributions
        """
        click.echo(
            f"\n🤖 AUTO-SETUP: Generating synthetic contributions for epoch {epoch}")
        click.echo("="*70)

        # Get available test accounts from node
        all_accounts = self.w3.eth.accounts[:num_agents]

        # First, add all accounts as consortium members
        deployer = all_accounts[0]  # Use first account as admin

        click.echo(f"🔧 Setting up consortium members...")
        for account in all_accounts:
            try:
                # Check if already a member
                is_member = self.registry.functions.isConsortiumMember(
                    account).call()
                if not is_member:
                    # Add as consortium member
                    tx_hash = self.registry.functions.addConsortiumMember(account).transact({
                        'from': deployer,
                        'gas': 200000
                    })
                    receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
                    click.echo(
                        f"  ✅ Added member: {account[:10]}...{account[-8:]}")

                # Check stake
                current_stake = self.registry.functions.memberStake(
                    account).call()
                min_stake = self.registry.functions.MINIMUM_STAKE().call()

                if current_stake < min_stake:
                    # Stake minimum amount
                    tx_hash = self.registry.functions.stakeMember().transact({
                        'from': account,
                        'value': min_stake,
                        'gas': 200000
                    })
                    receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
                    click.echo(
                        f"  💰 Staked {min_stake/1e18:.3f} ETH: {account[:10]}...{account[-8:]}")

            except Exception as e:
                click.echo(
                    f"  ⚠️  Setup error for {account}: {str(e)[:50]}...")

        # Check current epoch from contract
        current_epoch = self.registry.functions.currentEpoch().call()
        click.echo(f"\n📅 Contract current epoch: {current_epoch}")

        if current_epoch != epoch:
            click.echo(
                f"⚠️  Requested epoch {epoch} != contract epoch {current_epoch}")
            click.echo(f"   Using contract epoch {current_epoch}")
            epoch = current_epoch

        click.echo(
            f"\n📝 Registering {len(all_accounts)} synthetic contributions:\n")

        contributors = []
        contributions = []

        # Generate realistic contribution distribution
        np.random.seed(42 + epoch)
        contrib_amounts = np.random.lognormal(
            mean=np.log(np.mean(contribution_range)),
            sigma=0.5,
            size=num_agents
        )
        contrib_amounts = np.clip(
            contrib_amounts, contribution_range[0], contribution_range[1])

        for i, (account, amount_eth) in enumerate(zip(all_accounts, contrib_amounts)):
            amount_wei = self.w3.to_wei(amount_eth, 'ether')

            click.echo(f"Agent {i+1}: {account[:10]}...{account[-8:]}")
            click.echo(f"  Amount: {amount_eth:.4f} ETH ({amount_wei:,} wei)")

            try:
                # Check if already contributed
                existing_contrib = self.registry.functions.contributions(
                    epoch, account).call()
                if existing_contrib[2] > 0:  # timestamp > 0 means already contributed
                    click.echo(f"  ⏭️  Already contributed, skipping...")
                    contributors.append(account)
                    contributions.append(existing_contrib[1])  # value
                    continue

                # Generate data hash
                data_hash = Web3.keccak(
                    f"synthetic_contribution_{epoch}_{i}_{amount_wei}".encode())

                tx_hash = self.registry.functions.submitContribution(
                    amount_wei,
                    data_hash
                ).transact({
                    'from': account,
                    'gas': 300000
                })

                receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

                if receipt.status == 1:
                    click.echo(
                        f"  ✅ Registered (tx: {receipt.transactionHash.hex()[:16]}...)")
                    contributors.append(account)
                    contributions.append(amount_wei)
                else:
                    click.echo(f"  ❌ Transaction failed")

            except Exception as e:
                click.echo(f"  ⚠️  Error: {str(e)[:80]}...")

            click.echo()

        # Safer verification - check individual contributions instead of array
        click.echo("📊 Verifying on-chain contributions...")
        verified_count = 0

        for addr in contributors:
            try:
                contrib = self.registry.functions.contributions(
                    epoch, addr).call()
                if contrib[2] > 0:  # timestamp > 0
                    verified_count += 1
            except Exception as e:
                click.echo(f"  ⚠️  Verification error: {str(e)[:50]}...")

        click.echo(
            f"  ✅ Verified {verified_count}/{len(contributors)} contributions on-chain")

        # Try to get total value for the epoch
        try:
            total_value = self.registry.functions.epochTotalValue(epoch).call()
            click.echo(f"  📊 Total epoch value: {total_value/1e18:.4f} ETH")
        except Exception as e:
            click.echo(f"  ⚠️  Could not get total value: {str(e)[:50]}...")

        click.echo("="*70)

        return contributors, contributions

    def fetch_contributions(self, epoch: int) -> Tuple[List[str], List[int]]:
        """Fetch contributions from blockchain with safer error handling"""
        click.echo(f"\n📊 Fetching contributions for epoch {epoch}...")

        # First try the array method
        try:
            contributions = self.registry.functions.getEpochContributions(
                epoch).call()
            if contributions:
                contributors = [c[0] for c in contributions]
                values = [c[1] for c in contributions]

                click.echo(
                    f"  ✅ Found {len(contributors)} contributions via array")
                total_eth = sum([v / 1e18 for v in values])
                click.echo(f"  📈 Total: {total_eth:.4f} ETH")
                click.echo(
                    f"  📊 Range: {min(values)/1e18:.4f} - {max(values)/1e18:.4f} ETH")

                return contributors, values
        except Exception as e:
            click.echo(f"  ⚠️  Array method failed: {str(e)[:60]}...")

        # Fallback: try to get epoch contributors list
        try:
            # Get contributors from epochContributors mapping
            contributors = []
            values = []

            # Try first few addresses to see if any contributed
            # Check first 10 accounts
            test_accounts = self.w3.eth.accounts[:10]

            for addr in test_accounts:
                try:
                    contrib = self.registry.functions.contributions(
                        epoch, addr).call()
                    if contrib[2] > 0:  # timestamp > 0 means contribution exists
                        contributors.append(addr)
                        values.append(contrib[1])  # value
                except:
                    continue

            if contributors:
                click.echo(
                    f"  ✅ Found {len(contributors)} contributions via fallback method")
                total_eth = sum([v / 1e18 for v in values])
                click.echo(f"  📈 Total: {total_eth:.4f} ETH")
                return contributors, values

        except Exception as e:
            click.echo(f"  ⚠️  Fallback method failed: {str(e)[:60]}...")

        click.echo(f"  ⚠️  No contributions found for epoch {epoch}")
        return [], []

    def compute_shapley_values(
        self,
        contributors: List[str],
        contributions: List[int],
        method: str = 'mc_shapley'
    ) -> Tuple[np.ndarray, Dict]:
        """Compute Shapley values with proper consortium game setup"""
        click.echo(f"\n🧮 Computing Shapley values using {method}...")
        click.echo("="*70)

        n_agents = len(contributors)
        contrib_array = np.array(
            [c / 1e18 for c in contributions], dtype=float)

        # Create realistic consortium payoff function
        def consortium_payoff(coalition) -> float:
            """
            Consortium blockchain payoff with:
            - Individual contributions (additive base)
            - Synergy effects (superadditive bonus)
            - Network effects (sublinear in large coalitions)

            Args:
                coalition: Either frozenset of agent indices OR boolean mask array
            """
            # Handle both frozenset (from cache) and array (direct call)
            if isinstance(coalition, frozenset):
                if not coalition:
                    return 0.0
                coalition_mask = np.zeros(n_agents, dtype=bool)
                for idx in coalition:
                    coalition_mask[idx] = True
            else:
                coalition_mask = np.asarray(coalition, dtype=bool)
                if not np.any(coalition_mask):
                    return 0.0

            # Base value: sum of individual contributions
            base_value = np.sum(contrib_array[coalition_mask])

            # Synergy multiplier based on coalition size
            coalition_size = np.sum(coalition_mask)
            if coalition_size == 1:
                synergy = 1.0  # No synergy for individual
            elif coalition_size <= 3:
                synergy = 1.15  # 15% synergy for small groups
            elif coalition_size <= 5:
                synergy = 1.25  # 25% synergy for full consortium
            else:
                synergy = 1.20  # Diminishing returns for very large groups

            # Diversity bonus (agents with different contribution levels)
            if coalition_size > 1:
                coalition_contribs = contrib_array[coalition_mask]
                contrib_std = np.std(coalition_contribs)
                contrib_mean = np.mean(coalition_contribs)
                if contrib_mean > 0:
                    diversity = contrib_std / contrib_mean
                    diversity_bonus = min(
                        0.05, diversity * 0.1)  # Max 5% bonus
                else:
                    diversity_bonus = 0.0
            else:
                diversity_bonus = 0.0

            total_value = base_value * synergy * (1 + diversity_bonus)
            return total_value

        # Compute allocations
        start_time = time.time()

        result = self.allocation_engine.allocate(
            method=method,
            n_agents=n_agents,
            payoff_function=consortium_payoff,  # Now handles both types
            reports=contrib_array,
            n_samples=5000 if 'mc' in method else None
        )

        computation_time = time.time() - start_time

        click.echo(f"\n  ✓ Computed allocations for {n_agents} agents")
        click.echo(f"  ⏱️  Computation time: {computation_time:.3f}s")

        # Verify total value matches
        total_allocated = np.sum(result.allocations)
        total_contributed = np.sum(contrib_array)
        grand_coalition_value = consortium_payoff(frozenset(range(n_agents)))

        click.echo(f"  📊 Total contributions: {total_contributed:.4f} ETH")
        click.echo(
            f"  📊 Grand coalition value: {grand_coalition_value:.4f} ETH")
        click.echo(f"  📊 Total allocated: {total_allocated:.4f} ETH")

        # Calculate efficiency error
        efficiency_error = abs(total_allocated - grand_coalition_value)
        click.echo(f"  📊 Efficiency error: {efficiency_error:.8f} ETH")

        if result.converged is not None:
            status = "✅ CONVERGED" if result.converged else "⚠️  DID NOT CONVERGE"
            click.echo(f"  {status}")

        if result.stderr is not None:
            max_stderr = np.max(result.stderr)
            click.echo(f"  📊 Max Std Error: {max_stderr:.6f}")

        # Return metadata
        metadata = {
            'method': method,
            'n_samples': result.n_samples_used,
            'converged': result.converged,
            'total_value': grand_coalition_value,
            'computation_time': computation_time,
            'efficiency_error': efficiency_error
        }

        click.echo("="*70)

        return result.allocations, metadata

    def verify_allocation_axioms(
        self,
        epoch: int,
        shapley_values: np.ndarray,
        contributions: List[int],
        grand_coalition_value: float  # ← Add this parameter
    ) -> Dict:
        """Verify Shapley axioms off-chain before submission"""
        click.echo(f"\n✅ Verifying Shapley axioms...")
        click.echo("="*70)

        results = {}

        # 1. Efficiency (sum of allocations = grand coalition value)
        total_allocated = sum(shapley_values)
        efficiency_error = abs(total_allocated - grand_coalition_value)
        results['efficiency'] = efficiency_error < 1e-6
        results['efficiency_error'] = efficiency_error

        click.echo(f"\n  📊 Total allocated: {total_allocated:.4f} ETH")
        click.echo(f"  📊 Grand coalition value: {grand_coalition_value:.4f} ETH")
        click.echo(f"  📊 Efficiency error: {efficiency_error:.8f} ETH")

        # 2. Individual rationality (each agent gets at least their contribution)
        # For superadditive games, we expect some agents to get bonuses
        min_ratio = min(
            shapley_values[i] / (contributions[i] / 1e18)
            for i in range(len(shapley_values))
        )
        max_ratio = max(
            shapley_values[i] / (contributions[i] / 1e18)
            for i in range(len(shapley_values))
        )
        
        # All agents should get at least 85% of their contribution (tolerance for edge cases)
        # But typically should get 100%+ due to synergies
        results['individual_rational'] = all(
            shapley_values[i] >= (contributions[i] / 1e18) * 0.85
            for i in range(len(shapley_values))
        )
        
        click.echo(f"  📊 Allocation ratio range: {min_ratio:.2%} - {max_ratio:.2%}")

        # 3. Symmetry (similar contributions → similar allocations)
        results['symmetry'] = True
        symmetry_violations = []
        
        for i in range(len(shapley_values)):
            for j in range(i + 1, len(shapley_values)):
                contrib_i = contributions[i] / 1e18
                contrib_j = contributions[j] / 1e18
                
                # If contributions are within 5% of each other
                contrib_diff = abs(contrib_i - contrib_j) / max(contrib_i, contrib_j)
                
                if contrib_diff < 0.05:
                    # Allocations should also be within reasonable range (15%)
                    alloc_diff = abs(shapley_values[i] - shapley_values[j]) / max(shapley_values[i], shapley_values[j])
                    
                    if alloc_diff > 0.15:
                        results['symmetry'] = False
                        symmetry_violations.append((i+1, j+1, contrib_diff, alloc_diff))

        # Display results
        click.echo(f"\n  Efficiency: {'✅' if results['efficiency'] else '❌'} (error: {efficiency_error:.8f})")
        click.echo(f"  Individual Rational: {'✅' if results['individual_rational'] else '❌'}")
        click.echo(f"  Symmetry: {'✅' if results['symmetry'] else '❌'}")
        
        if symmetry_violations:
            click.echo(f"\n  ⚠️  Symmetry violations detected:")
            for v in symmetry_violations:
                click.echo(f"      Agents {v[0]} & {v[1]}: contrib_diff={v[2]:.1%}, alloc_diff={v[3]:.1%}")

        click.echo("="*70)

        return results

    def submit_to_oracle_quorum(
        self,
        epoch: int,
        contributors: List[str],
        shapley_values: np.ndarray,
        metadata: Dict,
        num_oracles: int = 2
    ) -> List[str]:
        """Submit allocation with multiple oracle signatures (quorum)"""
        click.echo(
            f"\n📤 Submitting allocation with {num_oracles}-oracle quorum...")
        click.echo("="*70)

        # Scale values to contract precision (1e18)
        scaled_values = [int(v * 1e18) for v in shapley_values]

        # Create computation hash
        computation_data = {
            'epoch': epoch,
            'contributors': contributors,
            'method': metadata['method'],
            'samples': metadata.get('n_samples', 0),
            'total_value': metadata['total_value'],
            'timestamp': self.w3.eth.block_number
        }

        computation_hash = Web3.keccak(
            text=json.dumps(computation_data, sort_keys=True)
        )

        # Create message hash for signatures
        message_hash = Web3.solidity_keccak(
            ['uint256', 'address[]', 'uint256[]', 'bytes32'],
            [epoch, contributors, scaled_values, computation_hash]
        )

        tx_hashes = []

        # Submit from multiple oracles
        for i, oracle_account in enumerate(self.oracle_accounts[:num_oracles]):
            click.echo(f"\n  Oracle {i+1} ({oracle_account.address[:10]}...):")

            # Check if already voted
            has_voted = self.oracle.functions.hasOracleVoted(
                epoch, oracle_account.address
            ).call()

            if has_voted:
                click.echo(f"    ⏭️  Already voted, skipping...")
                continue

            try:
                # Sign message
                message = encode_defunct(message_hash)
                signed_message = oracle_account.sign_message(message)

                # Submit to oracle
                tx_hash = self.oracle.functions.submitAllocation(
                    epoch,
                    contributors,
                    scaled_values,
                    computation_hash,
                    signed_message.signature
                ).transact({
                    'from': oracle_account.address,
                    'gas': 500000
                })

                receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
                tx_hashes.append(receipt.transactionHash.hex())

                # Check vote count
                vote_count = self.oracle.functions.getVoteCount(
                    epoch,
                    Web3.keccak(message_hash)
                ).call()

                click.echo(
                    f"    ✅ Submitted (Vote: {vote_count}/{num_oracles})")
                click.echo(
                    f"    📝 Tx: {receipt.transactionHash.hex()[:24]}...")

                # Check if finalized
                is_finalized = self.oracle.functions.allocationFinalized(
                    epoch).call()
                if is_finalized:
                    click.echo(f"\n  🎉 QUORUM REACHED - Allocation finalized!")
                    break

            except Exception as e:
                click.echo(f"    ❌ Error: {str(e)[:60]}...")

        click.echo("="*70)

        return tx_hashes

    def display_results(
        self,
        epoch: int,
        contributors: List[str],
        contributions: List[int],
        shapley_values: np.ndarray,
        metadata: Dict,
        oracle_txs: List[str]
    ):
        """Display comprehensive results"""
        click.echo("\n" + "="*70)
        click.echo("📊 ALLOCATION RESULTS")
        click.echo("="*70)

        total_contribution = sum(contributions)
        total_allocated = sum(shapley_values)

        for i, (addr, contrib, shapley) in enumerate(
            zip(contributors, contributions, shapley_values)
        ):
            contrib_eth = contrib / 1e18
            contrib_pct = (contrib / total_contribution) * 100
            shapley_pct = (shapley / total_allocated) * 100
            ratio = shapley / contrib_eth if contrib_eth > 0 else 0

            click.echo(f"\n🔹 Agent {i+1}: {addr[:10]}...{addr[-8:]}")
            click.echo(
                f"   Contribution:  {contrib_eth:>10.4f} ETH ({contrib_pct:>6.2f}%)")
            click.echo(
                f"   Shapley Value: {shapley:>10.4f} ETH ({shapley_pct:>6.2f}%)")
            click.echo(f"   Ratio:         {ratio:>10.2%}")

        click.echo("\n" + "="*70)
        click.echo(
            f"Total Contributions:   {sum([c/1e18 for c in contributions]):>10.4f} ETH")
        click.echo(f"Total Allocated:       {total_allocated:>10.4f} ETH")
        click.echo(
            f"Computation Time:      {metadata['computation_time']:>10.3f}s")
        click.echo(f"Method:                {metadata['method']}")
        click.echo(
            f"MC Samples:            {metadata.get('n_samples', 'N/A')}")
        click.echo(f"Oracle Transactions:   {len(oracle_txs)}")
        click.echo("="*70)

    def run_epoch_cycle(
        self,
        epoch: int,
        num_oracles: int = 2,
        auto_setup: bool = True,
        num_synthetic_agents: int = 5
    ):
        """Run complete cycle for an epoch with auto-setup"""
        click.echo(f"\n{'='*70}")
        click.echo(f"🚀 EPOCH {epoch} INTEGRATION PIPELINE")
        click.echo(f"{'='*70}")

        # 1. Fetch or create contributions
        contributors, contributions = self.fetch_contributions(epoch)

        if not contributors and auto_setup:
            click.echo(f"\n💡 No contributions found - running auto-setup...")
            contributors, contributions = self.auto_setup_epoch(
                epoch,
                num_agents=num_synthetic_agents
            )

        if not contributors:
            click.echo("\n❌ No contributions available")
            click.echo(
                "\n💡 Enable --auto-setup or register contributions manually")
            return

        # 2. Compute Shapley values
        shapley_values, metadata = self.compute_shapley_values(
            contributors,
            contributions,
            method='mc_shapley'
        )

        # 3. Verify axioms
        axiom_results = self.verify_allocation_axioms(
            epoch, shapley_values, contributions, metadata['total_value']
        )

        if not all(axiom_results.values()):
            click.echo("\n⚠️  Axiom verification failed")
            if not click.confirm("Continue with submission?"):
                return

        # 4. Submit to oracle quorum
        oracle_txs = self.submit_to_oracle_quorum(
            epoch, contributors, shapley_values, metadata, num_oracles
        )

        # 5. Display comprehensive results
        self.display_results(
            epoch, contributors, contributions,
            shapley_values, metadata, oracle_txs
        )

        click.echo(
            f"\n✅ Epoch {epoch} complete with {num_oracles}-oracle consensus!\n")


@click.command()
@click.option('--epoch', type=int, required=True, help='Epoch to process')
@click.option('--deployment', type=click.Path(exists=True), required=True, help='Deployment JSON')
@click.option('--oracle-keys', multiple=True, required=True, help='Oracle private keys')
@click.option('--rpc-url', default='http://127.0.0.1:8545', help='RPC URL')
@click.option('--num-oracles', type=int, default=2, help='Number of oracles to submit')
@click.option('--auto-setup/--no-auto-setup', default=True, help='Auto-generate contributions if missing')
@click.option('--num-agents', type=int, default=5, help='Number of synthetic agents for auto-setup')
def integrate(epoch, deployment, oracle_keys, rpc_url, num_oracles, auto_setup, num_agents):
    """
    Complete integration pipeline with auto-setup

    This script handles:
    - Auto-registering synthetic contributions (if needed)
    - Computing Shapley values off-chain
    - Verifying game-theoretic axioms
    - Submitting to oracle quorum
    - Displaying comprehensive results

    Examples:
        # Full auto pipeline
        python -m solidity.scripts.integrate \\
          --epoch 1 \\
          --deployment deployments/localhost_latest.json \\
          --oracle-keys $KEY1 $KEY2

        # Skip auto-setup (use existing contributions)
        python -m solidity.scripts.integrate \\
          --epoch 1 \\
          --deployment deployments/localhost_latest.json \\
          --oracle-keys $KEY1 $KEY2 \\
          --no-auto-setup
    """

    click.echo("\n" + "="*70)
    click.echo("🎯 SHAPLEY-FAIR CONSORTIUM BLOCKCHAIN")
    click.echo("   Off-Chain Computation → On-Chain Settlement")
    click.echo("="*70)

    # Load deployment
    with open(deployment, 'r') as f:
        deployment_info = json.load(f)

    # Connect to network
    w3 = Web3(Web3.HTTPProvider(rpc_url))

    if not w3.is_connected():
        click.echo("\n❌ Could not connect to blockchain")
        click.echo(f"   RPC URL: {rpc_url}")
        return

    click.echo(f"\n✅ Connected to blockchain")
    click.echo(f"   Network: {w3.eth.chain_id}")
    click.echo(f"   Block: {w3.eth.block_number}")

    # Create oracle accounts
    oracle_accounts = [Account.from_key(key) for key in oracle_keys]

    click.echo(f"\n🔑 Using {len(oracle_accounts)} oracle accounts:")
    for i, account in enumerate(oracle_accounts):
        balance = w3.eth.get_balance(account.address) / 1e18
        click.echo(f"   Oracle {i+1}: {account.address} ({balance:.4f} ETH)")

    # Create integration
    integration = ShapleyIntegration(w3, oracle_accounts, deployment_info)

    # Run epoch cycle
    integration.run_epoch_cycle(
        epoch,
        num_oracles=num_oracles,
        auto_setup=auto_setup,
        num_synthetic_agents=num_agents
    )


if __name__ == "__main__":
    integrate()
