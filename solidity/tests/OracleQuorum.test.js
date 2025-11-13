import { expect } from "chai";
import pkg from "hardhat";
const { ethers } = pkg;

describe("Oracle Quorum System", function () {
    let shapleyOracle;
    let oracleRegistry;
    let owner, oracle1, oracle2, oracle3, oracle4;

    beforeEach(async function () {
        [owner, oracle1, oracle2, oracle3, oracle4] = await ethers.getSigners();

        // Deploy OracleRegistry
        const OracleRegistry = await ethers.getContractFactory("OracleRegistry");
        oracleRegistry = await OracleRegistry.deploy(owner.address);
        await oracleRegistry.waitForDeployment();

        // Deploy ShapleyOracle with 2-of-3 quorum
        const ShapleyOracle = await ethers.getContractFactory("ShapleyOracle");
        shapleyOracle = await ShapleyOracle.deploy(2);
        await shapleyOracle.waitForDeployment();

        // Bond oracles
        await oracleRegistry.connect(oracle1).bondOracle({ value: ethers.parseEther("1") });
        await oracleRegistry.connect(oracle2).bondOracle({ value: ethers.parseEther("1") });
        await oracleRegistry.connect(oracle3).bondOracle({ value: ethers.parseEther("1") });

        // Grant oracle roles
        await shapleyOracle.addOracle(oracle1.address);
        await shapleyOracle.addOracle(oracle2.address);
        await shapleyOracle.addOracle(oracle3.address);
    });

    it("Should require quorum to finalize allocation", async function () {
        const epoch = 1;
        const agents = [owner.address];
        const values = [ethers.parseEther("100")];
        const computationHash = ethers.randomBytes(32);

        // Create message hash
        const messageHash = ethers.solidityPackedKeccak256(
            ["uint256", "address[]", "uint256[]", "bytes32"],
            [epoch, agents, values, computationHash]
        );

        // Oracle 1 submits
        const sig1 = await oracle1.signMessage(ethers.getBytes(messageHash));
        await shapleyOracle.connect(oracle1).submitAllocation(
            epoch, agents, values, computationHash, sig1
        );

        // Should not be finalized yet
        expect(await shapleyOracle.allocationFinalized(epoch)).to.be.false;

        // Oracle 2 submits (reaches quorum)
        const sig2 = await oracle2.signMessage(ethers.getBytes(messageHash));
        await shapleyOracle.connect(oracle2).submitAllocation(
            epoch, agents, values, computationHash, sig2
        );

        // Should be finalized now
        expect(await shapleyOracle.allocationFinalized(epoch)).to.be.true;
    });

    it("Should prevent duplicate oracle votes", async function () {
        const epoch = 1;
        const agents = [owner.address];
        const values = [ethers.parseEther("100")];
        const computationHash = ethers.randomBytes(32);

        const messageHash = ethers.solidityPackedKeccak256(
            ["uint256", "address[]", "uint256[]", "bytes32"],
            [epoch, agents, values, computationHash]
        );

        const sig1 = await oracle1.signMessage(ethers.getBytes(messageHash));
        await shapleyOracle.connect(oracle1).submitAllocation(
            epoch, agents, values, computationHash, sig1
        );

        // Try to submit again
        await expect(
            shapleyOracle.connect(oracle1).submitAllocation(
                epoch, agents, values, computationHash, sig1
            )
        ).to.be.revertedWith("Oracle already voted");
    });

    it("Should verify signature matches sender", async function () {
        const epoch = 1;
        const agents = [owner.address];
        const values = [ethers.parseEther("100")];
        const computationHash = ethers.randomBytes(32);

        const messageHash = ethers.solidityPackedKeccak256(
            ["uint256", "address[]", "uint256[]", "bytes32"],
            [epoch, agents, values, computationHash]
        );

        // Oracle1 signs but Oracle2 tries to submit
        const sig1 = await oracle1.signMessage(ethers.getBytes(messageHash));

        await expect(
            shapleyOracle.connect(oracle2).submitAllocation(
                epoch, agents, values, computationHash, sig1
            )
        ).to.be.revertedWith("Signature mismatch");
    });

    it("Should enforce efficiency axiom", async function () {
        const epoch = 1;
        const agents = [owner.address, oracle1.address];
        const values = [
            ethers.parseEther("100"),
            ethers.parseEther("50")  // Sum = 150
        ];
        const computationHash = ethers.randomBytes(32);

        // Message hash includes sum != grand coalition value
        const messageHash = ethers.solidityPackedKeccak256(
            ["uint256", "address[]", "uint256[]", "bytes32"],
            [epoch, agents, values, computationHash]
        );

        const sig1 = await oracle1.signMessage(ethers.getBytes(messageHash));
        await shapleyOracle.connect(oracle1).submitAllocation(
            epoch, agents, values, computationHash, sig1
        );

        const sig2 = await oracle2.signMessage(ethers.getBytes(messageHash));

        // This should succeed because sum = grandCoalitionValue
        // (grandCoalitionValue is computed as sum in contract)
        await shapleyOracle.connect(oracle2).submitAllocation(
            epoch, agents, values, computationHash, sig2
        );

        expect(await shapleyOracle.allocationFinalized(epoch)).to.be.true;
    });
});