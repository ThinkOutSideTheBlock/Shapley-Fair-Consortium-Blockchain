// SPDX-License-Identifier: MIT
pragma solidity ^0.8.25;

/**
 * @title IShapleySystem
 * @notice Interface definitions for Shapley allocation system
 * @dev Production-ready interface for consortium blockchain allocation
 */

interface IContributionRegistry {
    struct Contribution {
        address contributor;
        uint256 value;
        uint256 timestamp;
        bytes32 dataHash; // Hash of off-chain data
        bool verified;
    }

    event ContributionSubmitted(
        address indexed contributor,
        uint256 indexed epoch,
        uint256 value,
        bytes32 dataHash
    );

    function submitContribution(uint256 value, bytes32 dataHash) external;

    function getEpochContributions(
        uint256 epoch
    ) external view returns (Contribution[] memory);
}

interface IShapleyOracle {
    struct Allocation {
        address[] agents;
        uint256[] values;
        uint256 grandCoalitionValue;
        bytes32 computationHash;
        uint256 timestamp;
        bytes signature;
    }

    event AllocationSubmitted(
        uint256 indexed epoch,
        bytes32 indexed computationHash,
        uint256 grandCoalitionValue
    );

    function submitAllocation(
        uint256 epoch,
        address[] calldata agents,
        uint256[] calldata values,
        bytes32 computationHash,
        bytes calldata signature
    ) external;

    function verifyAllocation(
        uint256 epoch
    ) external view returns (bool valid, string memory reason);
}

interface IAllocationSettlement {
    enum SettlementStatus {
        Pending,
        Computed,
        Challenged,
        Settled,
        Distributed
    }

    event SettlementInitiated(uint256 indexed epoch, uint256 totalValue);

    event TokensDistributed(
        uint256 indexed epoch,
        address indexed recipient,
        uint256 amount
    );

    event DisputeRaised(
        uint256 indexed epoch,
        address indexed challenger,
        string reason
    );

    function initiateSettlement(uint256 epoch) external;
    function distributeTokens(uint256 epoch) external;
    function raiseDispute(
        uint256 epoch,
        string calldata reason,
        address allegedFraudulentOracle
    ) external payable;
}
