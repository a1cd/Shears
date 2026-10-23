// swift-tools-version: 6.0
import PackageDescription

// CryptoKit is Apple-only; elsewhere SHA-256 comes from swift-crypto, which has the same API.
#if canImport(Darwin)
let cryptoDependencies: [Package.Dependency] = []
let cryptoProducts: [Target.Dependency] = []
#else
let cryptoDependencies: [Package.Dependency] = [
    .package(url: "https://github.com/apple/swift-crypto.git", "3.0.0"..<"5.0.0"),
]
let cryptoProducts: [Target.Dependency] = [.product(name: "Crypto", package: "swift-crypto")]
#endif

let package = Package(
    name: "Shears",
    platforms: [.macOS(.v14), .iOS(.v17), .visionOS(.v1)],
    products: [
        .library(name: "Shears", targets: ["Shears"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.0"),
        .package(url: "https://github.com/huggingface/swift-jinja.git", from: "2.5.2"),
    ] + cryptoDependencies,
    targets: [
        // A std::terminate handler that prints what() and a backtrace (Sources/CShearsDiag).
        .target(name: "CShearsDiag", path: "Sources/CShearsDiag"),
        .target(
            name: "Shears",
            dependencies: [
                "CShearsDiag",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Jinja", package: "swift-jinja"),
            ] + cryptoProducts,
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "ShearsTests",
            dependencies: ["Shears"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
