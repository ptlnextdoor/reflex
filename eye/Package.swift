// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "eye",
    platforms: [.macOS(.v14)],
    // ponytail: v5 language mode. Ceiling: strict Swift 6 concurrency over
    // SCStream delegate callbacks is a lot of annotation churn for a bench CLI.
    targets: [
        .executableTarget(
            name: "eye",
            path: "Sources/eye",
            swiftSettings: [.swiftLanguageMode(.v5)]
        )
    ]
)
