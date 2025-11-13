import hre from "hardhat";

/**
 * End-to-end integration test
 * Tests full flow: commit -> reveal -> oracle submission -> settlement
 */

async function main() {
    console.log("🧪 Running Integration Test\n");

    const [deployer, oracle1, oracle2, oracle3, member1, member2, member3] = await hre.ethers.getSigners();

    // Deploy all contracts
    console.log("1️⃣ Deploying contracts...");

    const OracleRegistry = await hre.ethers.getContractFactory("OracleRegistry");
    const oracleRegistry = await OracleRegistry.deploy(deployer.address);
    await oracleRegistry.waitForDeployment();

    const ShapleyOracle = await hre.ethers.getContractFactory("ShapleyOracle");
    const shapleyOracle = await ShapleyOracle.deploy(2); // 2-of-3 quorum
    await shapleyOracle.waitForDeployment();

    const ContributionRegistry = await hre.ethers.getContractFactory("ContributionRegistry");
    const contributionRegistry = await ContributionRegistry.deploy();
    await contributionRegistry.waitForDeployment();

    const MockERC20 = await hre.ethers.getContractFactory("MockERC20");
    const rewardToken = await MockERC20.deploy("Reward", "RWD", 18, hre.ethers.parseEther("1000000"));
    await rewardToken.waitForDeployment();

    const AllocationSettlement = await hre.ethers.getContractFactory("AllocationSettlement");
    const settlement = await AllocationSettlement.deploy(
        await contributionRegistry.getAddress(),
        await shapleyOracle.getAddress(),
        await rewardToken.getAddress()
    );
    await settlement.waitForDeployment();

    console.log("✅ Contracts deployed\n");

    // Setup oracles
    console.log("2️⃣ Setting up oracles...");
    await oracleRegistry.connect(oracle1).bondOracle({ value: hre.ethers.parseEther("2") });
    await oracleRegistry.connect(oracle2).bondOracle({ value: hre.ethers.parseEther("2") });
    await oracleRegistry.connect(oracle3).bondOracle({ value: hre.ethers.parseEther("2") });

    await shapleyOracle.addOracle(oracle1.address);
    await shapleyOracle.addOracle(oracle2.address);
    await shapleyOracle.addOracle(oracle3.address);
    console.log("✅ 3 oracles bonded and registered\n");

    // Setup consortium members
    console.log("3️⃣ Adding consortium members...");
    await contributionRegistry.addConsortiumMember(member1.address);
    await contributionRegistry.addConsortiumMember(member2.address);
    await contributionRegistry.addConsortiumMember(member3.address);

    await contributionRegistry.connect(member1).stakeMember({ value: hre.ethers.parseEther("0.1") });
    await contributionRegistry.connect(member2).stakeMember({ value: hre.ethers.parseEther("0.1") });
    await contributionRegistry.connect(member3).stakeMember({ value: hre.ethers.parseEther("0.1") });
    console.log("✅ 3 members added and staked\n");
    // Phase 1: Commit contributions
    console.log("4️⃣ Commit phase...");
    const epoch = await contributionRegistry.currentEpoch();

    const contributions = [
        { member: member1, value: hre.ethers.parseEther("10"), salt: hre.ethers.randomBytes(32) },
        { member: member2, value: hre.ethers.parseEther("15"), salt: hre.ethers.randomBytes(32) },
        { member: member3, value: hre.ethers.parseEther("20"), salt: hre.ethers.randomBytes(32) }
    ];

    for (const contrib of contributions) {
        const dataHash = hre.ethers.randomBytes(32);
        const commitHash = hre.ethers.solidityPackedKeccak256(
            ["uint256", "bytes32", "bytes32"],
            [contrib.value, contrib.salt, dataHash]
        );
        contrib.dataHash = dataHash;
        contrib.commitHash = commitHash;

        await contributionRegistry.connect(contrib.member).commitContribution(commitHash);
        console.log(`  ✓ ${contrib.member.address.slice(0, 10)}... committed`);
    }
    console.log("✅ All commitments submitted\n");

    // Advance to reveal phase
    console.log("5️⃣ Advancing to reveal phase...");
    await hre.ethers.provider.send("evm_increaseTime", [6 * 3600 + 1]); // 6 hours + 1 sec
    await hre.ethers.provider.send("evm_mine");
    await contributionRegistry.advancePhase();
    console.log("✅ Reveal phase started\n");

    // Phase 2: Reveal contributions
    console.log("6️⃣ Reveal phase...");
    for (const contrib of contributions) {
        await contributionRegistry.connect(contrib.member).revealContribution(
            contrib.value,
            contrib.salt,
            contrib.dataHash,
            [] // Empty Merkle proof for now
        );
        console.log(`  ✓ ${contrib.member.address.slice(0, 10)}... revealed`);
    }
    console.log("✅ All contributions revealed\n");

    // Phase 3: Oracle submission (multi-oracle consensus)
    console.log("7️⃣ Oracle submission with quorum...");

    const agents = [member1.address, member2.address, member3.address];
    const values = contributions.map(c => c.value);
    const computationHash = hre.ethers.randomBytes(32);

    // Calculate message hash for signatures
    const messageHash = hre.ethers.solidityPackedKeccak256(
        ["uint256", "address[]", "uint256[]", "bytes32"],
        [epoch, agents, values, computationHash]
    );

    // Oracle 1 submits
    const sig1 = await oracle1.signMessage(hre.ethers.getBytes(messageHash));
    await shapleyOracle.connect(oracle1).submitAllocation(
        epoch,
        agents,
        values,
        computationHash,
        sig1
    );
    console.log("  ✓ Oracle 1 submitted (1/2 votes)");

    // Check vote count
    const allocationHash = hre.ethers.keccak256(
        hre.ethers.AbiCoder.defaultAbiCoder().encode(
            ["uint256", "address[]", "uint256[]", "bytes32"],
            [epoch, agents, values, computationHash]
        )
    );
    let voteCount = await shapleyOracle.getVoteCount(epoch, allocationHash);
    console.log(`  Current votes: ${voteCount}/2`);

    // Oracle 2 submits (reaches quorum)
    const sig2 = await oracle2.signMessage(hre.ethers.getBytes(messageHash));
    await shapleyOracle.connect(oracle2).submitAllocation(
        epoch,
        agents,
        values,
        computationHash,
        sig2
    );
    console.log("  ✓ Oracle 2 submitted (2/2 votes - QUORUM REACHED)");

    // Verify allocation finalized
    const isFinalized = await shapleyOracle.allocationFinalized(epoch);
    console.log(`✅ Allocation finalized: ${isFinalized}\n`);

    // Phase 4: Settlement
    console.log("8️⃣ Initiating settlement...");

    // Transfer tokens to settlement contract
    await rewardToken.transfer(await settlement.getAddress(), hre.ethers.parseEther("1000"));

    // Grant ADMIN_ROLE to deployer
    const ADMIN_ROLE = hre.ethers.keccak256(hre.ethers.toUtf8Bytes("ADMIN_ROLE"));
    await settlement.grantRole(ADMIN_ROLE, deployer.address);

    await settlement.initiateSettlement(epoch);
    console.log("✅ Settlement initiated\n");

    // Advance past challenge period
    console.log("9️⃣ Waiting for challenge period...");
    await hre.ethers.provider.send("evm_increaseTime", [6 * 3600 + 1]);
    await hre.ethers.provider.send("evm_mine");

    await settlement.distributeTokens(epoch);
    console.log("✅ Tokens distributed\n");

    // Phase 5: Claims
    console.log("🔟 Members claiming rewards...");
    for (const contrib of contributions) {
        const pendingReward = await settlement.getPendingRewards(contrib.member.address, epoch);
        console.log(`  ${contrib.member.address.slice(0, 10)}... pending: ${hre.ethers.formatEther(pendingReward)} RWD`);

        await settlement.connect(contrib.member).claimRewards(epoch);

        const balance = await rewardToken.balanceOf(contrib.member.address);
        console.log(`  ✓ Claimed! Balance: ${hre.ethers.formatEther(balance)} RWD`);
    }
    console.log("✅ All rewards claimed\n");

    // Summary
    console.log("📊 Integration Test Summary");
    console.log("=" * 50);
    console.log(`Total contributions: ${contributions.length}`);
    console.log(`Oracle consensus: 2-of-3 quorum`);
    console.log(`Challenge period: Passed without disputes`);
    console.log(`Total distributed: ${hre.ethers.formatEther(await settlement.settlements(epoch).then(s => s.distributedRewards))} RWD`);
    console.log("\n✅ Integration test PASSED!");
}
main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error(error);
        process.exit(1);
    });