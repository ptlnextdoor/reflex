import Foundation
import AppKit
import os.lock

// Motion-to-photon: post a key at host time t0, watch the captured probe region
// until mean luma crosses 50%, record (frame displayTime - t0). The capture
// callback runs the state machine; the driver thread only posts keys and sleeps.

private enum Phase {
    case armBlack   // waiting for the region to settle black
    case armWhite   // key is down, waiting for luma to cross up
    case done
}

private final class M2PState {
    private var lock = os_unfair_lock()
    var phase: Phase = .armBlack
    var t0: UInt64 = 0                 // keydown host time
    var blackRun = 0                   // consecutive black frames seen
    // last trial result (mach ticks)
    var photonTicks: UInt64 = 0        // displayTime of first white frame
    var hostTicks: UInt64 = 0          // callback host time of that frame
    var displayAtPhoton: UInt64 = 0
    let blackSem = DispatchSemaphore(value: 0)
    let whiteSem = DispatchSemaphore(value: 0)
    let lowThresh = 60.0               // luma below this counts as black
    let hiThresh = 127.5               // 50% crossing
    let stableNeeded = 3

    // Synchronous critical section. Safe to call from async code since it never
    // awaits while the lock is held.
    @inline(__always) func withLock<T>(_ body: () -> T) -> T {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        return body()
    }
    // For the capture callback, which manages its own early unlock paths.
    @inline(__always) func lockRaw() { os_unfair_lock_lock(&lock) }
    @inline(__always) func unlockRaw() { os_unfair_lock_unlock(&lock) }
}

func runBenchM2P(_ args: Args) {
    guard let bundleID = args.string("app") else { die("bench-m2p needs --app <bundleid>") }
    guard let rect = args.rect("rect") else { die("bench-m2p needs --rect x,y,w,h") }
    let trials = args.int("trials") ?? 200
    let modeStr = args.string("mode") ?? "frontmost"

    guard let (pid, bid) = Hand.resolve(bundleID: bundleID, pid: nil) else {
        die("could not find running app \(bundleID)")
    }
    let mode: HandMode = (modeStr == "topid") ? .toPid(pid) : .frontmost
    let hand = Hand(bundleID: bid, pid: pid, mode: mode)
    liveHand = hand
    installSignalRelease()

    if case .frontmost = mode, !hand.targetIsFrontmost() {
        die("SAFETY: \(bid ?? bundleID) is not frontmost. Focus the probe window and retry (or use --mode topid).")
    }

    let st = M2PState()
    // Per-frame scratch is all on the stack; measurements appended to pre-sized arrays.
    var m2pMs = [Double](); m2pMs.reserveCapacity(trials)        // displayTime - t0  (true motion-to-photon)
    var detectMs = [Double](); detectMs.reserveCapacity(trials)  // host - t0 (what our pipeline sees)
    var captureMs = [Double](); captureMs.reserveCapacity(trials)// host - displayTime (capture overhead)

    let cap = Capture(wantLuma: true) { f in
        if !f.complete || f.lumaMean < 0 { return }
        st.lockRaw()
        switch st.phase {
        case .armBlack:
            if f.lumaMean < st.lowThresh {
                st.blackRun += 1
                if st.blackRun >= st.stableNeeded {
                    st.blackRun = 0
                    st.phase = .done // park until driver re-arms
                    st.unlockRaw()
                    st.blackSem.signal()
                    return
                }
            } else {
                st.blackRun = 0
            }
        case .armWhite:
            if f.lumaMean > st.hiThresh {
                st.photonTicks = f.displayTicks != 0 ? f.displayTicks : f.hostTicks
                st.hostTicks = f.hostTicks
                st.displayAtPhoton = f.displayTicks
                st.phase = .done
                st.unlockRaw()
                st.whiteSem.signal()
                return
            }
        case .done:
            break
        }
        st.unlockRaw()
    }

    let sem = DispatchSemaphore(value: 0)
    Task {
        do { try await cap.start(rect: rect, outW: 64, outH: 64) }
        catch { die("capture start failed: \(error)") }
        try? await Task.sleep(nanoseconds: 500_000_000) // settle

        var completed = 0
        for t in 0..<trials {
            // Arm for black and wait until settled.
            st.withLock { st.phase = .armBlack; st.blackRun = 0 }
            if st.blackSem.wait(timeout: .now() + 2.0) == .timedOut {
                FileHandle.standardError.write("trial \(t): timed out waiting for black\n".data(using: .utf8)!)
                continue
            }
            // Random 50..150 ms gap.
            let gap = UInt64((50.0 + Double.random(in: 0...100)) * 1e6)
            try? await Task.sleep(nanoseconds: gap)

            // Arm for white, then post keydown and stamp t0.
            st.withLock { st.phase = .armWhite }
            let t0 = mach_absolute_time()
            st.withLock { st.t0 = t0 }
            if !hand.down(.up) { die("key post aborted by safety gate") }

            if st.whiteSem.wait(timeout: .now() + 2.0) == .timedOut {
                FileHandle.standardError.write("trial \(t): timed out waiting for white\n".data(using: .utf8)!)
                hand.up(.up)
                continue
            }
            let (photon, host, disp) = st.withLock { (st.photonTicks, st.hostTicks, st.displayAtPhoton) }

            m2pMs.append(Clock.deltaMillis(photon, t0))
            detectMs.append(Clock.deltaMillis(host, t0))
            if disp != 0 { captureMs.append(Clock.deltaMillis(host, disp)) }
            completed += 1

            // Release and let it fall back to black.
            hand.up(.up)
        }

        await cap.stop()

        print("bench-m2p  app=\(bid ?? bundleID)  mode=\(modeStr)  trials=\(trials)  completed=\(completed)")
        print("rect=\(fmt(rect.x,0)),\(fmt(rect.y,0)),\(fmt(rect.w,0)),\(fmt(rect.h,0))")
        print("")
        print("motion-to-photon (displayTime - t0), ms:")
        printDist(m2pMs)
        print("")
        print("split:")
        print("  capture latency (host - displayTime), ms:  p50 \(fmt(percentile(captureMs,50)))  p99 \(fmt(percentile(captureMs,99)))")
        print("  pipeline-observed detect (host - t0), ms:  p50 \(fmt(percentile(detectMs,50)))  p99 \(fmt(percentile(detectMs,99)))")
        print("  -> everything-else (keydown to photon) = motion-to-photon above; detect = m2p + capture overhead")
        sem.signal()
    }
    sem.wait()
    exit(0)
}

private func printDist(_ v: [Double]) {
    print("  p50 \(fmt(percentile(v,50)))  p90 \(fmt(percentile(v,90)))  p99 \(fmt(percentile(v,99)))  max \(fmt(v.max() ?? .nan))  mean \(fmt(mean(v)))  n=\(v.count)")
}
