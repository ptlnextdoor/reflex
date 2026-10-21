import Foundation
import CoreGraphics

// A visible window's on-screen bounds in points (top-left origin), plus owner.
struct WindowInfo {
    var pid: pid_t
    var owner: String
    var title: String
    var rect: CGRect   // points, global, top-left origin
    var layer: Int
}

enum Windows {
    // All on-screen windows, front to back.
    static func onScreen() -> [WindowInfo] {
        let opts: CGWindowListOption = [.optionOnScreenOnly, .excludeDesktopElements]
        guard let arr = CGWindowListCopyWindowInfo(opts, kCGNullWindowID) as? [[String: Any]] else { return [] }
        var out: [WindowInfo] = []
        for w in arr {
            guard let boundsDict = w[kCGWindowBounds as String] as? [String: Any],
                  let rect = CGRect(dictionaryRepresentation: boundsDict as CFDictionary) else { continue }
            let pid = (w[kCGWindowOwnerPID as String] as? pid_t) ?? 0
            let owner = (w[kCGWindowOwnerName as String] as? String) ?? ""
            let title = (w[kCGWindowName as String] as? String) ?? ""
            let layer = (w[kCGWindowLayer as String] as? Int) ?? 0
            out.append(WindowInfo(pid: pid, owner: owner, title: title, rect: rect, layer: layer))
        }
        return out
    }

    // Largest normal-layer window owned by `pid` (the content window, not a panel).
    static func largestWindow(pid: pid_t) -> WindowInfo? {
        onScreen()
            .filter { $0.pid == pid && $0.layer == 0 }
            .max { $0.rect.width * $0.rect.height < $1.rect.width * $1.rect.height }
    }
}
