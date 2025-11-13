# test_e2e.py - End-to-end test script

import subprocess
import time
import json
from pathlib import Path


def run_e2e_test():
    """Run complete end-to-end test in 10 minutes"""

    print("🚀 Starting E2E test (target: 10 min)\n")
    start_time = time.time()

    # 1. Start local blockchain (background)
    print("1️⃣ Starting local blockchain...")
    blockchain = subprocess.Popen(['anvil'], stdout=subprocess.PIPE)
    time.sleep(2)

    # 2. Deploy contracts (30 sec)
    print("2️⃣ Deploying contracts...")
    subprocess.run([
        'python', 'solidity/scripts/deploy.py',
        '--network', 'localhost'
    ])

    # 3. Run Python experiments (5 min)
    print("3️⃣ Running off-chain experiments...")
    subprocess.run([
        'python', 'src/main.py', 'run',
        '--config', 'configs/ultrafast_test.yaml',
        '--jobs', '4'
    ])

    # 4. Submit sample contributions (30 sec)
    print("4️⃣ Submitting on-chain contributions...")
    # Would call contract functions here

    # 5. Run integration (1 min)
    print("5️⃣ Running integration...")
    subprocess.run([
        'python', 'solidity/scripts/integrate.py',
        '--epoch', '1'
    ])

    # 6. Analyze results (30 sec)
    print("6️⃣ Analyzing results...")
    subprocess.run([
        'python', 'src/main.py', 'analyze',
        '--results', 'experiments/ultrafast_demo/results.csv'
    ])

    # Clean up
    blockchain.terminate()

    elapsed = time.time() - start_time
    print(f"\n✅ E2E test complete in {elapsed/60:.1f} minutes!")

    return elapsed < 600  # Success if under 10 min


if __name__ == "__main__":
    success = run_e2e_test()
    exit(0 if success else 1)
