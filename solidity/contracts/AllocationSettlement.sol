// SPDX-License-Identifier: MIT
pragma solidity ^0.8.25;

import "./interfaces/IShapleySystem.sol";
import "./ContributionRegistry.sol";
import "./ShapleyOracle.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";

/**
 * @title AllocationSettlement
 * @notice Settles Shapley allocations and distributes rewards
 * @dev Handles token distribution and dispute resolution
 */
contract AllocationSettlement is
    IAllocationSettlement,
    AccessControl,
    ReentrancyGuard
{
    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");
    bytes32 public constant DISPUTE_RESOLVER_ROLE =
        keccak256("DISPUTE_RESOLVER_ROLE");

    ContributionRegistry public immutable contributionRegistry;
    ShapleyOracle public immutable shapleyOracle;
    IERC20 public immutable rewardToken;

    struct Settlement {
        SettlementStatus status;
        uint256 totalRewards;
        uint256 distributedRewards;
        uint256 challengeDeadline;
        mapping(address => uint256) rewards;
        mapping(address => bool) claimed;
    }

    mapping(uint256 => Settlement) public settlements;
    mapping(uint256 => address[]) public epochRecipients;
    // Pull payment pattern for dispute rewards
    mapping(address => uint256) public pendingWithdrawals;

    // Link to OracleRegistry for slashing
    address public oracleRegistry;
    uint256 public challengePeriod = 6 hours;
    uint256 public minDisputeStake = 1e18; // 1 token

    // Dispute tracking
    struct Dispute {
        address challenger;
        string reason;
        uint256 stake;
        bool resolved;
        bool successful;
    }

    mapping(uint256 => Dispute[]) public epochDisputes;

    event RewardsClaimed(
        address indexed recipient,
        uint256 indexed epoch,
        uint256 amount
    );

    event DisputeResolved(
        uint256 indexed epoch,
        uint256 indexed disputeId,
        bool successful
    );
    event WithdrawalProcessed(address indexed account, uint256 amount);

    modifier settlementExists(uint256 epoch) {
        require(
            settlements[epoch].status != SettlementStatus.Pending,
            "Settlement not initiated"
        );
        _;
    }

    constructor(
        address _contributionRegistry,
        address _shapleyOracle,
        address _rewardToken
    ) {
        contributionRegistry = ContributionRegistry(_contributionRegistry);
        shapleyOracle = ShapleyOracle(_shapleyOracle);
        rewardToken = IERC20(_rewardToken);

        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(ADMIN_ROLE, msg.sender);
    }

    /**
     * @notice Initiate settlement for an epoch
     * @param epoch Epoch to settle
     */
    function initiateSettlement(
        uint256 epoch
    ) external override onlyRole(ADMIN_ROLE) {
        require(
            settlements[epoch].status == SettlementStatus.Pending,
            "Already initiated"
        );

        // Verify oracle has submitted allocation
        (bool valid, string memory reason) = shapleyOracle.verifyAllocation(
            epoch
        );
        require(valid, reason);

        // Get allocation from oracle
        IShapleyOracle.Allocation memory allocation = shapleyOracle
            .getAllocation(epoch);

        // Calculate total rewards for epoch
        uint256 totalRewards = rewardToken.balanceOf(address(this));
        require(totalRewards > 0, "No rewards available");

        // Store settlement
        Settlement storage settlement = settlements[epoch];
        settlement.status = SettlementStatus.Computed;
        settlement.totalRewards = totalRewards;
        settlement.challengeDeadline = block.timestamp + challengePeriod;

        // Store individual rewards
        for (uint256 i = 0; i < allocation.agents.length; i++) {
            address agent = allocation.agents[i];
            uint256 shapleyValue = allocation.values[i];

            // Calculate reward proportional to Shapley value
            uint256 reward = (shapleyValue * totalRewards) /
                allocation.grandCoalitionValue;

            settlement.rewards[agent] = reward;
            epochRecipients[epoch].push(agent);
        }

        emit SettlementInitiated(epoch, totalRewards);
    }

    /**
     * @notice Distribute tokens after challenge period
     * @param epoch Epoch to distribute
     */
    function distributeTokens(
        uint256 epoch
    ) external override nonReentrant settlementExists(epoch) {
        Settlement storage settlement = settlements[epoch];

        require(
            settlement.status == SettlementStatus.Computed,
            "Not ready for distribution"
        );
        require(
            block.timestamp > settlement.challengeDeadline,
            "Challenge period active"
        );

        settlement.status = SettlementStatus.Settled;
    }

    /**
     * @notice Claim rewards for an address
     * @param epoch Epoch to claim from
     */
    function claimRewards(uint256 epoch) external nonReentrant {
        Settlement storage settlement = settlements[epoch];

        require(settlement.status == SettlementStatus.Settled, "Not settled");
        require(!settlement.claimed[msg.sender], "Already claimed");
        require(settlement.rewards[msg.sender] > 0, "No rewards");

        uint256 reward = settlement.rewards[msg.sender];
        settlement.claimed[msg.sender] = true;
        settlement.distributedRewards += reward;

        require(rewardToken.transfer(msg.sender, reward), "Transfer failed");

        emit TokensDistributed(epoch, msg.sender, reward);
        emit RewardsClaimed(msg.sender, epoch, reward);

        // Update status if fully distributed
        if (settlement.distributedRewards == settlement.totalRewards) {
            settlement.status = SettlementStatus.Distributed;
        }
    }

    /**
     * @notice Raise dispute for an epoch
     * @param epoch Epoch to dispute
     * @param reason Dispute reason
     * @param allegedFraudulentOracle Oracle accused of fraud
     */
    function raiseDispute(
        uint256 epoch,
        string calldata reason,
        address allegedFraudulentOracle
    ) external payable override settlementExists(epoch) {
        Settlement storage settlement = settlements[epoch];

        require(
            settlement.status == SettlementStatus.Computed,
            "Cannot dispute"
        );
        require(
            block.timestamp <= settlement.challengeDeadline,
            "Challenge period ended"
        );
        require(msg.value >= minDisputeStake, "Insufficient stake");
        require(
            allegedFraudulentOracle != address(0),
            "Invalid oracle address"
        );

        epochDisputes[epoch].push(
            Dispute({
                challenger: msg.sender,
                reason: reason,
                stake: msg.value,
                resolved: false,
                successful: false
            })
        );

        settlement.status = SettlementStatus.Challenged;

        emit DisputeRaised(epoch, msg.sender, reason);
    }

    /**
     * @notice Resolve a dispute with fraud proof
     * @param epoch Epoch of dispute
     * @param disputeId Dispute ID
     * @param successful Whether dispute was valid
     * @param fraudulentOracle Oracle to slash (if successful)
     * @param fraudProof Evidence hash for fraud
     */
    function resolveDispute(
        uint256 epoch,
        uint256 disputeId,
        bool successful,
        address fraudulentOracle,
        bytes32 fraudProof
    ) external onlyRole(DISPUTE_RESOLVER_ROLE) {
        require(disputeId < epochDisputes[epoch].length, "Invalid dispute ID");
        Dispute storage dispute = epochDisputes[epoch][disputeId];

        require(!dispute.resolved, "Already resolved");
        require(fraudProof != bytes32(0), "Fraud proof required");

        dispute.resolved = true;
        dispute.successful = successful;

        if (successful) {
            // Slash oracle through OracleRegistry
            if (
                oracleRegistry != address(0) && fraudulentOracle != address(0)
            ) {
                // Call slash function on OracleRegistry
                // This will transfer slashed funds to challenger automatically
                (bool success, bytes memory data) = oracleRegistry.call(
                    abi.encodeWithSignature(
                        "slashOracle(address,string,address)",
                        fraudulentOracle,
                        dispute.reason,
                        dispute.challenger
                    )
                );

                if (success) {
                    // Slashing successful - oracle bond used for reward
                    // Return challenger's dispute stake
                    pendingWithdrawals[dispute.challenger] += dispute.stake;
                } else {
                    // Fallback: use dispute stake pool
                    // Return 2× stake to challenger
                    pendingWithdrawals[dispute.challenger] += dispute.stake * 2;
                }
            } else {
                // No oracle registry - use dispute stake
                pendingWithdrawals[dispute.challenger] += dispute.stake * 2;
            }

            // Reset settlement to allow recomputation
            settlements[epoch].status = SettlementStatus.Pending;
        } else {
            // Dispute unsuccessful - challenger loses stake
            // Stake goes to treasury (keep in contract)
            // Settlement continues
            settlements[epoch].status = SettlementStatus.Computed;
        }

        emit DisputeResolved(epoch, disputeId, successful);
    }

    /**
     * @notice Withdraw pending rewards/dispute winnings (Pull pattern)
     */
    function withdraw() external nonReentrant {
        uint256 amount = pendingWithdrawals[msg.sender];
        require(amount > 0, "No pending withdrawal");

        pendingWithdrawals[msg.sender] = 0;

        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Withdrawal failed");

        emit WithdrawalProcessed(msg.sender, amount);
    }

    /**
     * @notice Get pending withdrawal amount
     */
    function getPendingWithdrawal(
        address account
    ) external view returns (uint256) {
        return pendingWithdrawals[account];
    }

    /**
     * @notice Admin: Set oracle registry address
     */
    function setOracleRegistry(
        address _oracleRegistry
    ) external onlyRole(ADMIN_ROLE) {
        require(_oracleRegistry != address(0), "Invalid address");
        oracleRegistry = _oracleRegistry;
    }

    /**
     * @notice Get dispute count for epoch
     */
    function getDisputeCount(uint256 epoch) external view returns (uint256) {
        return epochDisputes[epoch].length;
    }

    /**
     * @notice Get dispute details
     */
    function getDispute(
        uint256 epoch,
        uint256 disputeId
    )
        external
        view
        returns (
            address challenger,
            string memory reason,
            uint256 stake,
            bool resolved,
            bool successful
        )
    {
        require(disputeId < epochDisputes[epoch].length, "Invalid dispute ID");
        Dispute memory dispute = epochDisputes[epoch][disputeId];

        return (
            dispute.challenger,
            dispute.reason,
            dispute.stake,
            dispute.resolved,
            dispute.successful
        );
    }

    /**
     * @notice Get pending rewards for an address
     * @param account Address to check
     * @param epoch Epoch to check
     * @return Pending reward amount
     */
    function getPendingRewards(
        address account,
        uint256 epoch
    ) external view returns (uint256) {
        if (settlements[epoch].claimed[account]) {
            return 0;
        }
        return settlements[epoch].rewards[account];
    }

    /**
     * @notice Get all recipients for an epoch
     * @param epoch Epoch number
     * @return Array of recipient addresses
     */
    function getEpochRecipients(
        uint256 epoch
    ) external view returns (address[] memory) {
        return epochRecipients[epoch];
    }

    /**
     * @notice Update challenge period
     * @param newPeriod New period in seconds
     */
    function setChallengePeriod(
        uint256 newPeriod
    ) external onlyRole(ADMIN_ROLE) {
        require(newPeriod >= 1 hours, "Too short");
        require(newPeriod <= 7 days, "Too long");
        challengePeriod = newPeriod;
    }
}
