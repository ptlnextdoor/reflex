import Foundation
import AppKit

// eye: SCK capture + CGEvent keys + latency benches for the reflex loop.
// Plain CommandLine dispatch, no ArgumentParser dependency.

func die(_ msg: String, _ code: Int32 = 1) -> Never {
    FileHandle.standardError.write((msg + "\n").data(using: .utf8)!)
    exit(code)
}

// --flag value parsing over CommandLine.arguments[2...].
struct Args {
    private var map: [String: String] = [:]
    init(_ argv: ArraySlice<String>) {
        var it = argv.makeIterator()
        while let a = it.next() {
            if a.hasPrefix("--") {
                let key = String(a.dropFirst(2))
                if let v = it.next() { map[key] = v } else { map[key] = "" }
            }
        }
    }
    func string(_ k: String) -> String? { map[k] }
    func int(_ k: String) -> Int? { map[k].flatMap { Int($0) } }
    func double(_ k: String) -> Double? { map[k].flatMap { Double($0) } }
    func rect(_ k: String) -> Capture.RectPts? {
        guard let s = map[k] else { return nil }
        let parts = s.split(separator: ",").compactMap { Double($0) }
        guard parts.count == 4 else { return nil }
        return Capture.RectPts(x: parts[0], y: parts[1], w: parts[2], h: parts[3])
    }
    func size(_ k: String, _ def: (Int, Int)) -> (Int, Int) {
        guard let s = map[k] else { return def }
        let parts = s.split(separator: "x").compactMap { Int($0) }
        guard parts.count == 2 else { return def }
        return (parts[0], parts[1])
    }
}

let argv = CommandLine.arguments
guard argv.count >= 2 else {
    die("""
    usage: eye <command> [flags]
      bench-capture --rect x,y,w,h [--size 256x144] [--seconds 10]
      bench-m2p     --app <bundleid> --rect x,y,w,h [--trials 200] [--mode frontmost|topid]
      postid-test   --app <bundleid> --rect x,y,w,h   (tests postToPid vs Chrome)
      list-windows  [--app <bundleid>]
      selftest
    """)
}

let cmd = argv[1]
let args = Args(argv.dropFirst(2))

switch cmd {
case "selftest":
    runSelftest()
case "list-windows":
    runListWindows(args)
case "bench-capture":
    runBenchCapture(args)
case "postid-test":
    runPostIDTest(args)
case "bench-m2p":
    runBenchM2P(args)
default:
    die("unknown command: \(cmd)")
}
