# Gas Benchmark Results

**Generated:** 2025-10-10T16:38:39.101Z

| n Agents | Submit Direct (gas) | Submit Merkle (gas) | Verify (gas) | Claim (gas) | Cost (USD @ 50 gwei) |
|----------|---------------------|---------------------|--------------|-------------|----------------------|
| 5 | 445,724 | --- | 50,000 | 50,000 | 66.86 |
| 10 | 673,099 | --- | 50,000 | 50,000 | 100.96 |
| 20 | 1,127,914 | --- | 50,000 | 50,000 | 169.19 |
| 50 | --- | 224,398 | 50,000 | 50,000 | 33.66 |
| 100 | --- | 224,398 | 50,000 | 50,000 | 33.66 |

**Notes:**
- ETH price: $3000, Gas price: 50 gwei
- Direct submission used for n ≤ 20, Merkle submission for n > 20
- Merkle method reduces gas from O(n) to O(log n) for large consortiums
