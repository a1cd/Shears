// swift-tools-version: 6.0
import PackageDescription

// The `shears-serve` command-line server and benchmark: a shim over `Shears.ServeCommand`. It lives in
// its own package so the library package keeps a single product (and so a single Xcode scheme,
// `Shears`, that builds and tests it). macOS only.
let package = Package(
    name: "shears-serve",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(path: ".."),
    ],
    targets: [
        .executableTarget(
            name: "shears-serve",
            dependencies: [.product(name: "Shears", package: "Shears")],
            path: "Sources/shears-serve",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
