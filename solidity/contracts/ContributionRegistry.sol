// SPDX-License-Identifier: MIT
pragma solidity ^0.8.25;

import "./interfaces/IShapleySystem.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/Pausable.sol";
import "@openzeppelin/contracts/utils/cryptography/MerkleProof.sol";

/**
 * @title ContributionRegistry
 * @notice Commit-reveal contribution registry with Merkle proof verification
 * @dev Prevents front-running and ensures verifiable contributions
 */
contract ContributionRegistry is
    IContributionRegistry,
    AccessControl,
    ReentrancyGuard,
    Pausable
{
    bytes32 public constant VALIDATOR_ROLE = keccak256("VALIDATOR_ROLE");
    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");

    uint256 public currentEpoch;
    uint256 public epochDuration = 1 days;
    uint256 public epochStartTime;

    enum ContributionPhase {
        Commit,
        Reveal,
        Finalized
    }

    // Phase management
    mapping(uint256 => ContributionPhase) public epochPhase;
    mapping(uint256 => uint256) public commitDeadline;
    mapping(uint256 => uint256) public revealDeadline;

    uint256 public commitPhaseDuration = 6 hours;
    uint256 public revealPhaseDuration = 6 hours;

    // Commit-reveal storage
    mapping(uint256 => mapping(address => bytes32)) public commitments;
    mapping(uint256 => mapping(address => bool)) public hasCommitted;
    mapping(uint256 => mapping(address => bool)) public hasRevealed;

    // Contributions (after reveal)
    mapping(uint256 => mapping(address => Contribution)) public contributions;
    mapping(uint256 => address[]) public epochContributors;
    mapping(uint256 => uint256) public epochTotalValue;

    // Merkle root for off-chain data verification
    mapping(uint256 => bytes32) public epochDataMerkleRoot;

    // Consortium member whitelist
    mapping(address => bool) public isConsortiumMember;
    mapping(address => uint256) public memberStake;

    uint256 public constant MINIMUM_STAKE = 0.01 ether;
    uint256 public minContribution = 1e15;
    uint256 public maxContribution = 1e21;

    event EpochAdvanced(uint256 indexed newEpoch, uint256 timestamp);
    event CommitPhaseStarted(uint256 indexed epoch, uint256 deadline);
    event RevealPhaseStarted(uint256 indexed epoch, uint256 deadline);
    event EpochFinalized(uint256 indexed epoch);

    event ContributionCommitted(
        address indexed contributor,
        uint256 indexed epoch,
        bytes32 commitHash
    );

    event ContributionRevealed(
        address indexed contributor,
        uint256 indexed epoch,
        uint256 value
    );

    event MemberAdded(address indexed member);
    event MemberRemoved(address indexed member);
    event MemberStaked(address indexed member, uint256 amount);

    modifier onlyConsortiumMember() {
        require(isConsortiumMember[msg.sender], "Not a consortium member");
        require(memberStake[msg.sender] >= MINIMUM_STAKE, "Insufficient stake");
        _;
    }

    modifier inCommitPhase(uint256 epoch) {
        require(
            epochPhase[epoch] == ContributionPhase.Commit,
            "Not in commit phase"
        );
        require(block.timestamp <= commitDeadline[epoch], "Commit phase ended");
        _;
    }

    modifier inRevealPhase(uint256 epoch) {
        require(
            epochPhase[epoch] == ContributionPhase.Reveal,
            "Not in reveal phase"
        );
        require(block.timestamp <= revealDeadline[epoch], "Reveal phase ended");
        _;
    }

    constructor() {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(ADMIN_ROLE, msg.sender);
        epochStartTime = block.timestamp;
        currentEpoch = 1;

        // Initialize first epoch
        _startCommitPhase(currentEpoch);
    }

    /**
     * @notice Stake ETH to become consortium member
     */
    function stakeMember() external payable {
        require(isConsortiumMember[msg.sender], "Not a member");
        require(msg.value > 0, "Must stake ETH");

        memberStake[msg.sender] += msg.value;
        emit MemberStaked(msg.sender, msg.value);
    }

    /**
     * @notice Commit to contribution (Phase 1)
     * @param commitHash keccak256(abi.encodePacked(value, salt, dataHash))
     */
    function commitContribution(
        bytes32 commitHash
    ) external onlyConsortiumMember whenNotPaused inCommitPhase(currentEpoch) {
        require(!hasCommitted[currentEpoch][msg.sender], "Already committed");
        require(commitHash != bytes32(0), "Invalid commit");

        commitments[currentEpoch][msg.sender] = commitHash;
        hasCommitted[currentEpoch][msg.sender] = true;

        emit ContributionCommitted(msg.sender, currentEpoch, commitHash);
    }

    /**
     * @notice Reveal contribution (Phase 2)
     * @param value Contribution value
     * @param salt Random salt used in commit
     * @param dataHash Hash of off-chain contribution data
     * @param merkleProof Proof that dataHash is in epoch Merkle tree
     */
    function revealContribution(
        uint256 value,
        bytes32 salt,
        bytes32 dataHash,
        bytes32[] calldata merkleProof
    ) external onlyConsortiumMember nonReentrant inRevealPhase(currentEpoch) {
        require(hasCommitted[currentEpoch][msg.sender], "No commitment found");
        require(!hasRevealed[currentEpoch][msg.sender], "Already revealed");
        require(
            value >= minContribution && value <= maxContribution,
            "Invalid value"
        );

        // Verify commit matches reveal
        bytes32 computedCommit = keccak256(
            abi.encodePacked(value, salt, dataHash)
        );
        require(
            computedCommit == commitments[currentEpoch][msg.sender],
            "Commit mismatch"
        );

        // Verify Merkle proof (if root is set)
        if (epochDataMerkleRoot[currentEpoch] != bytes32(0)) {
            bytes32 leaf = keccak256(abi.encodePacked(msg.sender, dataHash));
            require(
                MerkleProof.verify(
                    merkleProof,
                    epochDataMerkleRoot[currentEpoch],
                    leaf
                ),
                "Invalid Merkle proof"
            );
        }

        // Store contribution
        contributions[currentEpoch][msg.sender] = Contribution({
            contributor: msg.sender,
            value: value,
            timestamp: block.timestamp,
            dataHash: dataHash,
            verified: false
        });

        hasRevealed[currentEpoch][msg.sender] = true;
        epochContributors[currentEpoch].push(msg.sender);
        epochTotalValue[currentEpoch] += value;

        emit ContributionSubmitted(msg.sender, currentEpoch, value, dataHash);
        emit ContributionRevealed(msg.sender, currentEpoch, value);
    }

    /**
     * @notice Legacy submit (direct, no commit-reveal) - for testing
     * @dev Only use in development; production should use commit-reveal
     */
    function submitContribution(
        uint256 value,
        bytes32 dataHash
    ) external override onlyConsortiumMember whenNotPaused nonReentrant {
        require(
            value >= minContribution && value <= maxContribution,
            "Invalid value"
        );
        require(dataHash != bytes32(0), "Invalid data hash");
        require(
            contributions[currentEpoch][msg.sender].timestamp == 0,
            "Already contributed"
        );

        // Store contribution
        contributions[currentEpoch][msg.sender] = Contribution({
            contributor: msg.sender,
            value: value,
            timestamp: block.timestamp,
            dataHash: dataHash,
            verified: false
        });

        epochContributors[currentEpoch].push(msg.sender);
        epochTotalValue[currentEpoch] += value;

        emit ContributionSubmitted(msg.sender, currentEpoch, value, dataHash);
    }

    /**
     * @notice Advance epoch phases automatically
     */
    function advancePhase() external {
        uint256 epoch = currentEpoch;

        if (epochPhase[epoch] == ContributionPhase.Commit) {
            require(
                block.timestamp > commitDeadline[epoch],
                "Commit phase active"
            );
            _startRevealPhase(epoch);
        } else if (epochPhase[epoch] == ContributionPhase.Reveal) {
            require(
                block.timestamp > revealDeadline[epoch],
                "Reveal phase active"
            );
            _finalizeEpoch(epoch);
            _advanceEpoch();
        }
    }
    /**
     * @notice Internal: Start commit phase
     */
    function _startCommitPhase(uint256 epoch) internal {
        epochPhase[epoch] = ContributionPhase.Commit;
        commitDeadline[epoch] = block.timestamp + commitPhaseDuration;

        emit CommitPhaseStarted(epoch, commitDeadline[epoch]);
    }

    /**
     * @notice Internal: Start reveal phase
     */
    function _startRevealPhase(uint256 epoch) internal {
        epochPhase[epoch] = ContributionPhase.Reveal;
        revealDeadline[epoch] = block.timestamp + revealPhaseDuration;

        emit RevealPhaseStarted(epoch, revealDeadline[epoch]);
    }

    /**
     * @notice Internal: Finalize epoch
     */
    function _finalizeEpoch(uint256 epoch) internal {
        epochPhase[epoch] = ContributionPhase.Finalized;
        emit EpochFinalized(epoch);
    }

    /**
     * @notice Internal: Advance to next epoch
     */
    function _advanceEpoch() internal {
        currentEpoch++;
        epochStartTime = block.timestamp;
        _startCommitPhase(currentEpoch);

        emit EpochAdvanced(currentEpoch, block.timestamp);
    }

    /**
     * @notice Admin: Set Merkle root for epoch data verification
     */
    function setEpochDataMerkleRoot(
        uint256 epoch,
        bytes32 merkleRoot
    ) external onlyRole(VALIDATOR_ROLE) {
        require(
            epochPhase[epoch] == ContributionPhase.Commit,
            "Can only set during commit"
        );
        epochDataMerkleRoot[epoch] = merkleRoot;
    }

    /**
     * @notice Get all contributions for an epoch
     */
    function getEpochContributions(
        uint256 epoch
    ) external view override returns (Contribution[] memory) {
        address[] memory contributors = epochContributors[epoch];
        Contribution[] memory epochContribs = new Contribution[](
            contributors.length
        );

        for (uint256 i = 0; i < contributors.length; i++) {
            epochContribs[i] = contributions[epoch][contributors[i]];
        }

        return epochContribs;
    }

    /**
     * @notice Verify a contribution
     */
    function verifyContribution(
        uint256 epoch,
        address contributor
    ) external onlyRole(VALIDATOR_ROLE) {
        contributions[epoch][contributor].verified = true;
    }

    /**
     * @notice Add consortium member
     */
    function addConsortiumMember(address member) external onlyRole(ADMIN_ROLE) {
        require(member != address(0), "Invalid address");
        require(!isConsortiumMember[member], "Already a member");

        isConsortiumMember[member] = true;
        emit MemberAdded(member);
    }

    /**
     * @notice Remove consortium member
     */
    function removeConsortiumMember(
        address member
    ) external onlyRole(ADMIN_ROLE) {
        require(isConsortiumMember[member], "Not a member");

        isConsortiumMember[member] = false;
        emit MemberRemoved(member);
    }

    /**
     * @notice Update phase durations
     */
    function setPhaseDurations(
        uint256 _commitDuration,
        uint256 _revealDuration
    ) external onlyRole(ADMIN_ROLE) {
        require(
            _commitDuration >= 1 hours && _commitDuration <= 24 hours,
            "Invalid commit duration"
        );
        require(
            _revealDuration >= 1 hours && _revealDuration <= 24 hours,
            "Invalid reveal duration"
        );

        commitPhaseDuration = _commitDuration;
        revealPhaseDuration = _revealDuration;
    }

    /**
     * @notice Get current phase for epoch
     */
    function getCurrentPhase(
        uint256 epoch
    ) external view returns (ContributionPhase) {
        return epochPhase[epoch];
    }

    /**
     * @notice Check if address has committed for epoch
     */
    function hasAddressCommitted(
        uint256 epoch,
        address addr
    ) external view returns (bool) {
        return hasCommitted[epoch][addr];
    }

    /**
     * @notice Check if address has revealed for epoch
     */
    function hasAddressRevealed(
        uint256 epoch,
        address addr
    ) external view returns (bool) {
        return hasRevealed[epoch][addr];
    }
}
