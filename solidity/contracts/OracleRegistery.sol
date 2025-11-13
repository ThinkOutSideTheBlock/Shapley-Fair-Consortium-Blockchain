// SPDX-License-Identifier: MIT
pragma solidity ^0.8.25;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/**
 * @title OracleRegistry
 * @notice Manages oracle bonding, slashing, and economic security
 * @dev Oracles must bond ETH to participate; bonds slashed for fraud
 */
contract OracleRegistry is AccessControl, ReentrancyGuard {
    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");
    bytes32 public constant SLASHER_ROLE = keccak256("SLASHER_ROLE");

    uint256 public constant MINIMUM_BOND = 1 ether;
    uint256 public constant MAXIMUM_BOND = 100 ether;
    uint256 public slashPercentage = 50; // 50% of bond slashed for fraud

    struct OracleInfo {
        uint256 bondAmount;
        uint256 totalSlashed;
        uint256 successfulSubmissions;
        uint256 registrationTime;
        bool active;
    }

    mapping(address => OracleInfo) public oracles;
    address[] public oracleList;

    // Slash history for transparency
    struct SlashEvent {
        address oracle;
        uint256 amount;
        string reason;
        uint256 timestamp;
        address beneficiary;
    }
    SlashEvent[] public slashHistory;

    // Treasury for slashed funds
    address public treasury;
    uint256 public totalSlashedFunds;

    event OracleBonded(
        address indexed oracle,
        uint256 amount,
        uint256 totalBond
    );
    event OracleUnbonded(
        address indexed oracle,
        uint256 amount,
        uint256 remainingBond
    );
    event OracleSlashed(
        address indexed oracle,
        uint256 amount,
        string reason,
        address indexed beneficiary
    );
    event OracleActivated(address indexed oracle);
    event OracleDeactivated(address indexed oracle);

    modifier onlyActiveOracle(address oracle) {
        require(oracles[oracle].active, "Oracle not active");
        require(
            oracles[oracle].bondAmount >= MINIMUM_BOND,
            "Insufficient bond"
        );
        _;
    }

    constructor(address _treasury) {
        require(_treasury != address(0), "Invalid treasury");
        treasury = _treasury;

        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(ADMIN_ROLE, msg.sender);
        _grantRole(SLASHER_ROLE, msg.sender);
    }

    /**
     * @notice Bond ETH to become oracle
     * @dev Must meet minimum bond requirement
     */
    function bondOracle() external payable nonReentrant {
        require(msg.value > 0, "Must bond ETH");

        OracleInfo storage oracle = oracles[msg.sender];

        // First time bonding
        if (oracle.registrationTime == 0) {
            oracle.registrationTime = block.timestamp;
            oracleList.push(msg.sender);
        }

        oracle.bondAmount += msg.value;
        require(oracle.bondAmount <= MAXIMUM_BOND, "Exceeds maximum bond");

        // Activate if meets minimum
        if (oracle.bondAmount >= MINIMUM_BOND && !oracle.active) {
            oracle.active = true;
            emit OracleActivated(msg.sender);
        }

        emit OracleBonded(msg.sender, msg.value, oracle.bondAmount);
    }

    /**
     * @notice Unbond ETH (only if not slashed below minimum)
     * @param amount Amount to unbond
     */
    function unbondOracle(uint256 amount) external nonReentrant {
        OracleInfo storage oracle = oracles[msg.sender];
        require(oracle.bondAmount >= amount, "Insufficient bond");

        oracle.bondAmount -= amount;

        // Deactivate if below minimum
        if (oracle.bondAmount < MINIMUM_BOND && oracle.active) {
            oracle.active = false;
            emit OracleDeactivated(msg.sender);
        }

        // Transfer unbonded amount
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");

        emit OracleUnbonded(msg.sender, amount, oracle.bondAmount);
    }

    /**
     * @notice Slash oracle for fraudulent behavior
     * @param oracle Oracle to slash
     * @param reason Reason for slashing
     * @param beneficiary Address to receive slashed funds (e.g., challenger)
     */
    function slashOracle(
        address oracle,
        string calldata reason,
        address beneficiary
    )
        external
        onlyRole(SLASHER_ROLE)
        nonReentrant
        returns (uint256 slashedAmount)
    {
        OracleInfo storage oracleInfo = oracles[oracle];
        require(oracleInfo.bondAmount > 0, "No bond to slash");

        // Calculate slash amount
        slashedAmount = (oracleInfo.bondAmount * slashPercentage) / 100;
        require(slashedAmount > 0, "Slash amount too small");

        // Update oracle state
        oracleInfo.bondAmount -= slashedAmount;
        oracleInfo.totalSlashed += slashedAmount;
        oracleInfo.active = false; // Deactivate immediately

        totalSlashedFunds += slashedAmount;

        // Record slash event
        slashHistory.push(
            SlashEvent({
                oracle: oracle,
                amount: slashedAmount,
                reason: reason,
                timestamp: block.timestamp,
                beneficiary: beneficiary
            })
        );

        // Distribute slashed funds
        uint256 beneficiaryShare = slashedAmount / 2; // 50% to beneficiary
        uint256 treasuryShare = slashedAmount - beneficiaryShare; // 50% to treasury

        if (beneficiary != address(0)) {
            (bool success1, ) = beneficiary.call{value: beneficiaryShare}("");
            require(success1, "Beneficiary transfer failed");
        } else {
            treasuryShare += beneficiaryShare; // All to treasury if no beneficiary
        }

        (bool success2, ) = treasury.call{value: treasuryShare}("");
        require(success2, "Treasury transfer failed");

        emit OracleSlashed(oracle, slashedAmount, reason, beneficiary);
        emit OracleDeactivated(oracle);

        return slashedAmount;
    }

    /**
     * @notice Record successful submission (used by ShapleyOracle)
     */
    function recordSuccessfulSubmission(
        address oracle
    ) external onlyRole(SLASHER_ROLE) {
        require(oracles[oracle].active, "Oracle not active");
        oracles[oracle].successfulSubmissions++;
    }

    /**
     * @notice Check if oracle is eligible to submit
     */
    function isOracleEligible(address oracle) external view returns (bool) {
        OracleInfo memory info = oracles[oracle];
        return info.active && info.bondAmount >= MINIMUM_BOND;
    }

    /**
     * @notice Get oracle bond amount
     */
    function getOracleBond(address oracle) external view returns (uint256) {
        return oracles[oracle].bondAmount;
    }

    /**
     * @notice Get oracle info
     */
    function getOracleInfo(
        address oracle
    )
        external
        view
        returns (
            uint256 bondAmount,
            uint256 totalSlashed,
            uint256 successfulSubmissions,
            uint256 registrationTime,
            bool active
        )
    {
        OracleInfo memory info = oracles[oracle];
        return (
            info.bondAmount,
            info.totalSlashed,
            info.successfulSubmissions,
            info.registrationTime,
            info.active
        );
    }

    /**
     * @notice Get total number of oracles
     */
    function getOracleCount() external view returns (uint256) {
        return oracleList.length;
    }

    /**
     * @notice Get active oracle count
     */
    function getActiveOracleCount() external view returns (uint256) {
        uint256 count = 0;
        for (uint256 i = 0; i < oracleList.length; i++) {
            if (oracles[oracleList[i]].active) {
                count++;
            }
        }
        return count;
    }

    /**
     * @notice Get slash history count
     */
    function getSlashHistoryCount() external view returns (uint256) {
        return slashHistory.length;
    }

    /**
     * @notice Admin: Update slash percentage
     */
    function setSlashPercentage(
        uint256 newPercentage
    ) external onlyRole(ADMIN_ROLE) {
        require(
            newPercentage > 0 && newPercentage <= 100,
            "Invalid percentage"
        );
        slashPercentage = newPercentage;
    }

    /**
     * @notice Admin: Update treasury address
     */
    function setTreasury(address newTreasury) external onlyRole(ADMIN_ROLE) {
        require(newTreasury != address(0), "Invalid treasury");
        treasury = newTreasury;
    }

    /**
     * @notice Emergency: Deactivate oracle
     */
    function emergencyDeactivateOracle(
        address oracle
    ) external onlyRole(ADMIN_ROLE) {
        oracles[oracle].active = false;
        emit OracleDeactivated(oracle);
    }
}
