import Foundation
import AppKit

// ---- signal-safe key release -----------------------------------------------
// A global so the signal handler can reach the live Hand. ponytail: a single
// process-wide injector is fine for a CLI; ceiling is one target at a time.
var liveHand: Hand?
private var signalsInstalled = false

func installSignalRelease() {
    if signalsInstalled { return }
    signalsInstalled = true
    for sig in [SIGINT, SIGTERM] {
        signal(sig) { _ in
            liveHand?.releaseAll()
            // Re-raise default so the exit status reflects the signal.
            _exit(130)
        }
    }
    // atexit covers normal returns and die().
    atexit { liveHand?.releaseAll() }
}

// ---- selftest --------------------------------------------------------------

func runSelftest() {
    var failures = 0
    func check(_ name: String, _ cond: Bool) {
        if cond { print("ok   \(name)") }
        else { print("FAIL \(name)"); failures += 1 }
    }

    // Luminance threshold crossing: 50% of 0..255 is 127.5.
    let threshold = 127.5
    check("black below threshold", lumaCrossed(10.0, threshold) == false)
    check("white above threshold", lumaCrossed(240.0, threshold) == true)
    check("exact mid not yet crossed", lumaCrossed(127.0, threshold) == false)
    check("just over crosses", lumaCrossed(128.0, threshold) == true)

    // Rec.601 integer luma weights sum to 256 (keeps 0..255 range).
    check("luma weights sum 256", 77 + 150 + 29 == 256)
    // Pure white -> ~255, pure black -> 0 with those weights.
    let white = (255 * 77 + 255 * 150 + 255 * 29) >> 8
    check("white luma == 255", white == 255)
    let black = (0 * 77 + 0 * 150 + 0 * 29) >> 8
    check("black luma == 0", black == 0)

    // Percentiles on a known ramp 0..100.
    let ramp = (0...100).map { Double($0) }
    check("p50 of 0..100 == 50", percentile(ramp, 50) == 50)
    check("p99 of 0..100 == 99", percentile(ramp, 99) == 99)
    check("p0 == 0", percentile(ramp, 0) == 0)
    check("p100 == 100", percentile(ramp, 100) == 100)

    // Timebase: round-trip a known interval is positive and monotone.
    let a = Clock.now(); let b = a &+ 1_000_000
    check("deltaMillis positive", Clock.deltaMillis(b, a) > 0)
    check("deltaMillis sign flips", Clock.deltaMillis(a, b) < 0)

    if failures == 0 { print("\nselftest PASS") ; exit(0) }
    else { print("\nselftest FAIL (\(failures))"); exit(1) }
}

// Threshold crossing used by bench-m2p. Pulled out so selftest can assert it.
@inline(__always)
func lumaCrossed(_ luma: Double, _ threshold: Double) -> Bool { luma > threshold }

// ---- capture target --------------------------------------------------------
// --window <id> captures one window (works on another Desktop, no focus needed).
// Otherwise --rect captures a display region. ponytail: two modes, one helper.
func startCapture(_ cap: Capture, _ args: Args, outW: Int, outH: Int) async throws {
    if let wid = args.int("window") {
        try await cap.start(windowID: CGWindowID(wid), rect: args.rect("rect"), outW: outW, outH: outH)
    } else if let rect = args.rect("rect") {
        try await cap.start(rect: rect, outW: outW, outH: outH)
    } else {
        die("need --window <id> or --rect x,y,w,h")
    }
}

// ---- list-windows ----------------------------------------------------------

func runListWindows(_ args: Args) {
    let filterApp = args.string("app")
    for w in Windows.list(all: args.string("all") != nil) where w.layer == 0 {
        let app = NSRunningApplication(processIdentifier: w.pid)
        let bid = app?.bundleIdentifier ?? "?"
        if let f = filterApp, f != bid { continue }
        let r = w.rect
        print("window \(w.id)  pid \(w.pid)  \(bid)  \"\(w.title)\"  rect \(Int(r.minX)),\(Int(r.minY)),\(Int(r.width)),\(Int(r.height))  \(w.onScreen ? "on-screen" : "other-desktop")")
    }
}

// ---- bench-capture ---------------------------------------------------------

func runBenchCapture(_ args: Args) {
    let (w, h) = args.size("size", (256, 144))
    let seconds = args.double("seconds") ?? 10.0

    // Pre-size ring buffers so the callback never allocates.
    let cap = Int(seconds * 130) + 256
    var intervalsMs = [Double](); intervalsMs.reserveCapacity(cap)
    var captureMs = [Double](); captureMs.reserveCapacity(cap)
    var completeCount = 0
    var idleCount = 0
    var lastHost: UInt64 = 0

    let sem = DispatchSemaphore(value: 0)
    let cap0 = Capture(wantLuma: false) { f in
        if !f.complete { idleCount += 1; return }
        completeCount += 1
        if lastHost != 0 { intervalsMs.append(Clock.deltaMillis(f.hostTicks, lastHost)) }
        lastHost = f.hostTicks
        if f.displayTicks != 0 {
            captureMs.append(Clock.deltaMillis(f.hostTicks, f.displayTicks))
        }
    }

    Task {
        do { try await startCapture(cap0, args, outW: w, outH: h) }
        catch { die("capture start failed: \(error)") }
        try? await Task.sleep(nanoseconds: UInt64(seconds * 1e9))
        await cap0.stop()
        sem.signal()
    }
    sem.wait()

    print("bench-capture  target=\(args.string("window").map { "window \($0)" } ?? "rect \(args.string("rect") ?? "")")  size=\(w)x\(h)  seconds=\(fmt(seconds,1))")
    print("complete frames: \(completeCount)   idle frames: \(idleCount)   fps: \(fmt(Double(completeCount)/seconds,1))")
    print("inter-frame interval ms:  p50 \(fmt(percentile(intervalsMs,50)))  p99 \(fmt(percentile(intervalsMs,99)))  mean \(fmt(mean(intervalsMs)))")
    print("capture latency ms (host-displayTime):  p50 \(fmt(percentile(captureMs,50)))  p99 \(fmt(percentile(captureMs,99)))  mean \(fmt(mean(captureMs)))")
    exit(0)
}

// ---- postToPid test --------------------------------------------------------
// Opens by watching the probe region: posts a key via postToPid to the target
// pid WITHOUT focusing it, and checks whether the captured region turns white.

func runPostIDTest(_ args: Args) {
    guard let bundleID = args.string("app") else { die("postid-test needs --app <bundleid>") }
    guard let (pid, bid) = Hand.resolve(bundleID: bundleID, pid: nil) else {
        die("could not find running app \(bundleID)")
    }

    let hand = Hand(bundleID: bid, pid: pid, mode: .toPid(pid))
    liveHand = hand
    installSignalRelease()

    // Report who is frontmost so the evidence is unambiguous.
    let front = NSWorkspace.shared.frontmostApplication
    print("postid-test target=\(bid ?? "?") pid=\(pid)  frontmost=\(front?.bundleIdentifier ?? "?") (pid \(front?.processIdentifier ?? -1))")

    var luma = LumaWatcher()
    let sem = DispatchSemaphore(value: 0)
    let cap = Capture(wantLuma: true) { f in luma.update(f) }

    Task {
        do { try await startCapture(cap, args, outW: 64, outH: 64) }
        catch { die("capture start failed: \(error)") }
        try? await Task.sleep(nanoseconds: 400_000_000)

        let baseline = luma.latest
        // Press and hold up arrow for 250 ms via postToPid, then release.
        hand.down(.up)
        try? await Task.sleep(nanoseconds: 250_000_000)
        let whileDown = luma.maxSince(baselineReset: false)
        hand.up(.up)
        try? await Task.sleep(nanoseconds: 250_000_000)

        let delta = whileDown - baseline
        print("baseline luma \(fmt(baseline)) -> peak-while-down \(fmt(whileDown))  (delta \(fmt(delta)))")
        let worked = delta > 40 && whileDown > 127.5
        print("postToPid delivers keydown to \(bid ?? "?") without focus: \(worked ? "YES" : "NO")")
        await cap.stop()
        sem.signal()
    }
    sem.wait()
    exit(0)
}

// Small luma tracker, updated from the capture callback.
struct LumaWatcher {
    var latest: Double = 0
    private var maxSeen: Double = 0
    mutating func update(_ f: FrameSample) {
        if f.lumaMean < 0 { return }
        latest = f.lumaMean
        if f.lumaMean > maxSeen { maxSeen = f.lumaMean }
    }
    mutating func maxSince(baselineReset: Bool) -> Double {
        let v = maxSeen
        if baselineReset { maxSeen = 0 }
        return v
    }
}
