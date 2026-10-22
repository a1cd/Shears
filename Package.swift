// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Shears",
    platforms: [.macOS(.v14), .iOS(.v17), .visionOS(.v1)],
    products: [
        .library(name: "Shears", targets: ["Shears"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.0"),
        .package(url: "https://github.com/huggingface/swift-jinja.git", from: "2.5.2"),
    ],
    targets: [
        .target(
            name: "Shears",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Jinja", package: "swift-jinja"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "ShearsTests",
            dependencies: ["Shears"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
