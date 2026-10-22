// swift-tools-version: 6.0
import PackageDescription

// The `shears-cli` command-line runner: a shim over `Shears.ShearsRunner`. It lives in its own
// package so the library package keeps a single product (and so a single Xcode scheme, `Shears`,
// that builds and tests it). macOS only.
let package = Package(
    name: "shears-cli",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(path: ".."),
    ],
    targets: [
        .executableTarget(
            name: "shears-cli",
            dependencies: [.product(name: "Shears", package: "Shears")],
            path: "Sources/shears-cli",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
