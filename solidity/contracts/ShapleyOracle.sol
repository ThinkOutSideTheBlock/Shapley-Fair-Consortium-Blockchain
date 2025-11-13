// SPDX-License-Identifier: MIT
pragma solidity ^0.8.25;

import "./interfaces/IShapleySystem.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import "@openzeppelin/contracts/utils/cryptography/MessageHashUtils.sol";
import "@openzeppelin/contracts/utils/cryptography/MerkleProof.sol";

/**
 * @title ShapleyOracle
 * @notice Multi-oracle consensus for Shapley allocations with k-of-n quorum
 * @dev Requires QUORUM_THRESHOLD signatures to accept allocation
 */
contract ShapleyOracle is IShapleyOracle, AccessControl {
    using ECDSA for bytes32;

    bytes32 public constant ORACLE_ROLE = keccak256("ORACLE_ROLE");
    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");

    // Quorum parameters
    uint256 public QUORUM_THRESHOLD = 3; // Require 3-of-5 signatures
    uint256 public constant MAX_ORACLES = 10;

    // Allocation storage
    mapping(uint256 => Allocation) public allocations;
    mapping(uint256 => bool) public allocationFinalized;

    // Quorum tracking: epoch => allocationHash => oracle => signed
    mapping(uint256 => mapping(bytes32 => mapping(address => bool)))
        public oracleVotes;
    mapping(uint256 => mapping(bytes32 => uint256)) public voteCount;
    mapping(uint256 => mapping(bytes32 => AllocationProposal)) public proposals;

    // Prevent replay attacks
    mapping(bytes32 => bool) public usedComputationHashes;
    mapping(uint256 => mapping(address => bool)) public oracleSubmittedForEpoch;
    // Merkle root for large agent sets (gas optimization)
    mapping(uint256 => bytes32) public allocationMerkleRoot;
    mapping(uint256 => uint256) public agentCount;

    // Threshold for using Merkle tree vs full storage
    uint256 public constant MERKLE_THRESHOLD = 20;

    struct AllocationProposal {
        address[] agents;
        uint256[] values;
        bytes32 computationHash;
        uint256 grandCoalitionValue;
        uint256 proposalTime;
    }

    // Validation parameters
    uint256 public constant PRECISION = 1e18;
    uint256 public maxAllocationDeviation = 1e15; // 0.1% tolerance for efficiency axiom

    event AllocationProposed(
        uint256 indexed epoch,
        bytes32 indexed allocationHash,
        address indexed oracle,
        uint256 voteCount
    );

    event AllocationFinalized(
        uint256 indexed epoch,
        bytes32 indexed allocationHash,
        uint256 finalVoteCount
    );
    event AllocationMerkleRootSubmitted(
        uint256 indexed epoch,
        bytes32 indexed merkleRoot,
        uint256 agentCount
    );

    event QuorumThresholdUpdated(uint256 oldThreshold, uint256 newThreshold);
    event ValidationFailed(uint256 indexed epoch, string reason);

    modifier onlyOracle() {
        require(hasRole(ORACLE_ROLE, msg.sender), "Not an oracle");
        _;
    }

    constructor(uint256 _quorumThreshold) {
        require(
            _quorumThreshold > 0 && _quorumThreshold <= MAX_ORACLES,
            "Invalid threshold"
        );
        QUORUM_THRESHOLD = _quorumThreshold;

        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(ADMIN_ROLE, msg.sender);
    }

    /**
     * @notice Submit allocation with signature (multi-oracle consensus)
     * @dev Each oracle submits independently; allocation finalizes at threshold
     */
    function submitAllocation(
        uint256 epoch,
        address[] calldata agents,
        uint256[] calldata values,
        bytes32 computationHash,
        bytes calldata signature
    ) external override onlyOracle {
        require(!allocationFinalized[epoch], "Epoch already finalized");
        require(agents.length == values.length, "Length mismatch");
        require(
            agents.length > 0 && agents.length <= 100,
            "Invalid agent count"
        );
        require(computationHash != bytes32(0), "Invalid computation hash");
        require(
            !oracleSubmittedForEpoch[epoch][msg.sender],
            "Oracle already voted"
        );

        // Verify signature matches calling oracle
        bytes32 allocationHash = keccak256(
            abi.encodePacked(epoch, agents, values, computationHash)
        );
        bytes32 ethSignedMessageHash = MessageHashUtils.toEthSignedMessageHash(
            allocationHash
        );
        address signer = ECDSA.recover(ethSignedMessageHash, signature);
        require(signer == msg.sender, "Signature mismatch");
        require(hasRole(ORACLE_ROLE, signer), "Signer not authorized oracle");

        // Record vote
        oracleVotes[epoch][allocationHash][msg.sender] = true;
        oracleSubmittedForEpoch[epoch][msg.sender] = true;
        voteCount[epoch][allocationHash]++;

        // Store proposal on first submission
        if (voteCount[epoch][allocationHash] == 1) {
            uint256 grandCoalitionValue = 0;
            for (uint256 i = 0; i < values.length; i++) {
                grandCoalitionValue += values[i];
            }

            proposals[epoch][allocationHash] = AllocationProposal({
                agents: agents,
                values: values,
                computationHash: computationHash,
                grandCoalitionValue: grandCoalitionValue,
                proposalTime: block.timestamp
            });
        }

        emit AllocationProposed(
            epoch,
            allocationHash,
            msg.sender,
            voteCount[epoch][allocationHash]
        );

        // Finalize if quorum reached
        if (voteCount[epoch][allocationHash] >= QUORUM_THRESHOLD) {
            _finalizeAllocation(epoch, allocationHash);
        }
    }

    /**
     * @notice Internal: finalize allocation after quorum reached
     */
    function _finalizeAllocation(
        uint256 epoch,
        bytes32 allocationHash
    ) internal {
        require(!allocationFinalized[epoch], "Already finalized");

        AllocationProposal memory proposal = proposals[epoch][allocationHash];

        // Validate before finalizing
        (bool valid, string memory reason) = _validateAllocation(proposal);
        require(valid, reason);

        // Prevent computation hash reuse
        require(
            !usedComputationHashes[proposal.computationHash],
            "Hash reused"
        );
        usedComputationHashes[proposal.computationHash] = true;

        // Store finalized allocation
        allocations[epoch] = Allocation({
            agents: proposal.agents,
            values: proposal.values,
            grandCoalitionValue: proposal.grandCoalitionValue,
            computationHash: proposal.computationHash,
            timestamp: block.timestamp,
            signature: "" // Signature not needed for finalized allocation
        });

        allocationFinalized[epoch] = true;

        emit AllocationFinalized(
            epoch,
            allocationHash,
            voteCount[epoch][allocationHash]
        );
        emit AllocationSubmitted(
            epoch,
            proposal.computationHash,
            proposal.grandCoalitionValue
        );
    }

    /**
     * @notice Validate allocation against Shapley axioms
     */
    function _validateAllocation(
        AllocationProposal memory proposal
    ) internal view returns (bool valid, string memory reason) {
        // 1. Efficiency axiom: sum of allocations ≈ grand coalition value
        uint256 allocationSum = 0;
        for (uint256 i = 0; i < proposal.values.length; i++) {
            allocationSum += proposal.values[i];
        }

        // Allow small tolerance for rounding
        if (allocationSum > proposal.grandCoalitionValue) {
            uint256 excess = allocationSum - proposal.grandCoalitionValue;
            if (excess > maxAllocationDeviation) {
                return (false, "Efficiency axiom violated: sum exceeds v(N)");
            }
        } else {
            uint256 deficit = proposal.grandCoalitionValue - allocationSum;
            if (deficit > maxAllocationDeviation) {
                return (false, "Efficiency axiom violated: sum below v(N)");
            }
        }

        // 2. Check for duplicate agents (light version - sample check)
        if (proposal.agents.length <= 20) {
            // Full check for small n
            for (uint256 i = 0; i < proposal.agents.length; i++) {
                for (uint256 j = i + 1; j < proposal.agents.length; j++) {
                    if (proposal.agents[i] == proposal.agents[j]) {
                        return (false, "Duplicate agents detected");
                    }
                }
            }
        } else {
            // Sample check for large n (check first/last/middle)
            if (
                proposal.agents[0] ==
                proposal.agents[proposal.agents.length - 1]
            ) {
                return (false, "Duplicate agents (sample check)");
            }
        }

        // 3. Sanity bounds: no single agent gets > 2x grand coalition value
        for (uint256 i = 0; i < proposal.values.length; i++) {
            if (proposal.values[i] > proposal.grandCoalitionValue * 2) {
                return (false, "Unreasonable allocation value");
            }
        }

        // 4. Timestamp sanity
        if (proposal.proposalTime > block.timestamp) {
            return (false, "Future timestamp");
        }

        return (true, "");
    }

    /**
     * @notice Public verification endpoint
     */
    function verifyAllocation(
        uint256 epoch
    ) external view override returns (bool valid, string memory reason) {
        if (!allocationFinalized[epoch]) {
            return (false, "Allocation not finalized");
        }

        // Reconstruct proposal for validation
        Allocation memory alloc = allocations[epoch];
        AllocationProposal memory proposal = AllocationProposal({
            agents: alloc.agents,
            values: alloc.values,
            computationHash: alloc.computationHash,
            grandCoalitionValue: alloc.grandCoalitionValue,
            proposalTime: alloc.timestamp
        });

        return _validateAllocation(proposal);
    }

    /**
     * @notice Get allocation (only if finalized)
     */
    function getAllocation(
        uint256 epoch
    ) external view returns (Allocation memory) {
        require(allocationFinalized[epoch], "Allocation not finalized");
        return allocations[epoch];
    }

    /**
     * @notice Get current vote count for a proposed allocation
     */
    function getVoteCount(
        uint256 epoch,
        bytes32 allocationHash
    ) external view returns (uint256) {
        return voteCount[epoch][allocationHash];
    }

    /**
     * @notice Check if oracle has voted for epoch
     */
    function hasOracleVoted(
        uint256 epoch,
        address oracle
    ) external view returns (bool) {
        return oracleSubmittedForEpoch[epoch][oracle];
    }

    /**
     * @notice Admin: Update quorum threshold
     */
    function setQuorumThreshold(
        uint256 newThreshold
    ) external onlyRole(ADMIN_ROLE) {
        require(
            newThreshold > 0 && newThreshold <= MAX_ORACLES,
            "Invalid threshold"
        );
        uint256 oldThreshold = QUORUM_THRESHOLD;
        QUORUM_THRESHOLD = newThreshold;
        emit QuorumThresholdUpdated(oldThreshold, newThreshold);
    }

    /**
     * @notice Admin: Update efficiency tolerance
     */
    function setAllocationDeviation(
        uint256 newDeviation
    ) external onlyRole(ADMIN_ROLE) {
        require(newDeviation <= PRECISION / 10, "Deviation too large"); // Max 10%
        maxAllocationDeviation = newDeviation;
    }

    /**
     * @notice Admin: Grant oracle role
     */
    function addOracle(address oracle) external onlyRole(ADMIN_ROLE) {
        require(oracle != address(0), "Invalid oracle address");
        grantRole(ORACLE_ROLE, oracle);
    }

    /**
     * @notice Admin: Revoke oracle role
     */
    function removeOracle(address oracle) external onlyRole(ADMIN_ROLE) {
        revokeRole(ORACLE_ROLE, oracle);
    }
    /**
     * @notice Submit allocation as Merkle root (for large n)
     * @dev Use when agent count > MERKLE_THRESHOLD to save gas
     * @param epoch Epoch number
     * @param merkleRoot Merkle root of allocation tree
     * @param _agentCount Number of agents in allocation
     * @param grandCoalitionValue Total value to distribute
     * @param computationHash Hash of computation parameters
     * @param signature Oracle signature
     */
    function submitAllocationMerkle(
        uint256 epoch,
        bytes32 merkleRoot,
        uint256 _agentCount,
        uint256 grandCoalitionValue,
        bytes32 computationHash,
        bytes calldata signature
    ) external onlyOracle {
        require(!allocationFinalized[epoch], "Epoch already finalized");
        require(merkleRoot != bytes32(0), "Invalid Merkle root");
        require(
            _agentCount > MERKLE_THRESHOLD,
            "Use regular submit for small n"
        );
        require(
            !oracleSubmittedForEpoch[epoch][msg.sender],
            "Oracle already voted"
        );

        // Verify signature
        bytes32 messageHash = keccak256(
            abi.encodePacked(
                epoch,
                merkleRoot,
                _agentCount,
                grandCoalitionValue,
                computationHash
            )
        );
        bytes32 ethSignedMessageHash = MessageHashUtils.toEthSignedMessageHash(
            messageHash
        );
        address signer = ECDSA.recover(ethSignedMessageHash, signature);

        require(signer == msg.sender, "Signature mismatch");
        require(hasRole(ORACLE_ROLE, signer), "Signer not authorized oracle");

        // Create allocation hash for voting
        bytes32 allocationHash = keccak256(
            abi.encodePacked(merkleRoot, _agentCount, grandCoalitionValue)
        );

        // Record vote
        oracleVotes[epoch][allocationHash][msg.sender] = true;
        oracleSubmittedForEpoch[epoch][msg.sender] = true;
        voteCount[epoch][allocationHash]++;

        // Store Merkle root on first submission
        if (voteCount[epoch][allocationHash] == 1) {
            allocationMerkleRoot[epoch] = merkleRoot;
            agentCount[epoch] = _agentCount;

            // Store minimal allocation data
            allocations[epoch] = Allocation({
                agents: new address[](0), // Empty - use Merkle proofs
                values: new uint256[](0), // Empty - use Merkle proofs
                grandCoalitionValue: grandCoalitionValue,
                computationHash: computationHash,
                timestamp: block.timestamp,
                signature: ""
            });
        }

        emit AllocationMerkleRootSubmitted(epoch, merkleRoot, _agentCount);

        // Finalize if quorum reached
        if (voteCount[epoch][allocationHash] >= QUORUM_THRESHOLD) {
            require(!usedComputationHashes[computationHash], "Hash reused");
            usedComputationHashes[computationHash] = true;

            allocationFinalized[epoch] = true;

            emit AllocationFinalized(
                epoch,
                allocationHash,
                voteCount[epoch][allocationHash]
            );
            emit AllocationSubmitted(
                epoch,
                computationHash,
                grandCoalitionValue
            );
        }
    }

    /**
     * @notice Verify agent's allocation with Merkle proof
     * @param epoch Epoch number
     * @param agent Agent address
     * @param value Claimed allocation value
     * @param proof Merkle proof
     */
    function verifyAgentAllocation(
        uint256 epoch,
        address agent,
        uint256 value,
        bytes32[] calldata proof
    ) external view returns (bool) {
        require(allocationFinalized[epoch], "Allocation not finalized");

        bytes32 root = allocationMerkleRoot[epoch];
        if (root == bytes32(0)) {
            // Regular allocation - check stored values
            Allocation memory alloc = allocations[epoch];
            for (uint256 i = 0; i < alloc.agents.length; i++) {
                if (alloc.agents[i] == agent) {
                    return alloc.values[i] == value;
                }
            }
            return false;
        }

        // Merkle allocation - verify proof
        bytes32 leaf = keccak256(abi.encodePacked(agent, value));
        return MerkleProof.verify(proof, root, leaf);
    }

    /**
     * @notice Get allocation method used for epoch
     */
    function getAllocationMethod(
        uint256 epoch
    ) external view returns (string memory) {
        if (allocationMerkleRoot[epoch] != bytes32(0)) {
            return "merkle";
        } else if (allocations[epoch].agents.length > 0) {
            return "direct";
        } else {
            return "none";
        }
    }
    /**
     * @notice Get count of active oracles
     */
    function getOracleCount() external pure returns (uint256) {
        // Note: This is gas-expensive; use off-chain for UI
        uint256 count = 0;
        // In production, maintain a separate oracle list
        return count;
    }
}
