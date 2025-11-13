import "@nomicfoundation/hardhat-toolbox";

/** @type import('hardhat/config').HardhatUserConfig */
export default {
  solidity: {
    version: "0.8.25",
    settings: {
      optimizer: {
        enabled: true,
        runs: 200
      },
      viaIR: true
    }
  }, gasReporter: {
    enabled: true,
    currency: "USD",
    gasPrice: 50,
    coinmarketcap: process.env.COINMARKETCAP_API_KEY,
    outputFile: "benchmarks/gas-report.txt",
    noColors: true
  },
  networks: {
    hardhat: {
      chainId: 31337
    }
  },
  paths: {
    sources: "./contracts",
    tests: "./test",
    cache: "./cache",
    artifacts: "./artifacts"
  }
};