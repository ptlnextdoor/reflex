import Foundation
import CoreGraphics
import AppKit

// The keys the driver is allowed to press. Virtual keycodes (ANSI / layout
// independent arrows) from HIToolbox Events.h.
enum Key: String, CaseIterable {
    case up, down, left, right
    case w, a, s, d
    case r, enter, escape

    var code: CGKeyCode {
        switch self {
        case .up: return 126
        case .down: return 125
        case .left: return 123
        case .right: return 124
        case .w: return 13
        case .a: return 0
        case .s: return 1
        case .d: return 2
        case .r: return 15
        case .enter: return 36
        case .escape: return 53
        }
    }
}

// Delivery mode for key events.
enum HandMode {
    case frontmost          // CGEvent.post(.cghidEventTap), requires target to be frontmost
    case toPid(pid_t)       // CGEvent.postToPid, background delivery
}

// Injects keys to a specific target app and guarantees held keys are released.
final class Hand {
    let targetBundleID: String?
    let targetPID: pid_t
    let mode: HandMode
    private var held = Set<CGKeyCode>()
    private let source = CGEventSource(stateID: .hidSystemState)

    init(bundleID: String?, pid: pid_t, mode: HandMode) {
        self.targetBundleID = bundleID
        self.targetPID = pid
        self.mode = mode
    }

    // Resolve a target app from a bundle id or pid. Returns (pid, bundleID).
    static func resolve(bundleID: String?, pid: pid_t?) -> (pid_t, String?)? {
        if let pid = pid {
            if let app = NSRunningApplication(processIdentifier: pid) {
                return (pid, app.bundleIdentifier)
            }
            return (pid, nil)
        }
        if let bid = bundleID {
            let apps = NSRunningApplication.runningApplications(withBundleIdentifier: bid)
            if let app = apps.first { return (app.processIdentifier, bid) }
        }
        return nil
    }

    // True if our target is the frontmost app right now.
    func targetIsFrontmost() -> Bool {
        guard let front = NSWorkspace.shared.frontmostApplication else { return false }
        if front.processIdentifier == targetPID { return true }
        if let bid = targetBundleID, front.bundleIdentifier == bid { return true }
        return false
    }

    // Safety gate for .frontmost mode. For .toPid we deliver regardless of focus
    // (that is the whole point), but still only to the target pid.
    @discardableResult
    private func guardFocus() -> Bool {
        if case .frontmost = mode {
            if !targetIsFrontmost() {
                releaseAll()
                let name = targetBundleID ?? "pid \(targetPID)"
                FileHandle.standardError.write(
                    "SAFETY: target \(name) is not frontmost; releasing keys and aborting.\n".data(using: .utf8)!)
                return false
            }
        }
        return true
    }

    @discardableResult
    func down(_ key: Key) -> Bool {
        guard guardFocus() else { return false }
        post(key.code, down: true)
        held.insert(key.code)
        return true
    }

    @discardableResult
    func up(_ key: Key) -> Bool {
        // Key-up is always safe to deliver; it only releases.
        post(key.code, down: false)
        held.remove(key.code)
        return true
    }

    @inline(__always)
    private func post(_ code: CGKeyCode, down: Bool) {
        guard let ev = CGEvent(keyboardEventSource: source, virtualKey: code, keyDown: down) else { return }
        switch mode {
        case .frontmost:
            ev.post(tap: .cghidEventTap)
        case .toPid(let pid):
            ev.postToPid(pid)
        }
    }

    func releaseAll() {
        for code in held {
            if let ev = CGEvent(keyboardEventSource: source, virtualKey: code, keyDown: false) {
                switch mode {
                case .frontmost: ev.post(tap: .cghidEventTap)
                case .toPid(let pid): ev.postToPid(pid)
                }
            }
        }
        held.removeAll()
    }

    var heldCount: Int { held.count }
}
