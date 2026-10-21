import Foundation
import CoreGraphics

// A window's bounds in points (top-left origin), plus owner and id.
struct WindowInfo {
    var id: CGWindowID
    var pid: pid_t
    var owner: String
    var title: String
    var rect: CGRect   // points, global, top-left origin
    var layer: Int
    var onScreen: Bool // false = minimized or on another Desktop
}

enum Windows {
    // Front to back. all=true includes windows on other Desktops.
    static func list(all: Bool = false) -> [WindowInfo] {
        let opts: CGWindowListOption = all ? [.excludeDesktopElements] : [.optionOnScreenOnly, .excludeDesktopElements]
        guard let arr = CGWindowListCopyWindowInfo(opts, kCGNullWindowID) as? [[String: Any]] else { return [] }
        var out: [WindowInfo] = []
        for w in arr {
            guard let boundsDict = w[kCGWindowBounds as String] as? [String: Any],
                  let rect = CGRect(dictionaryRepresentation: boundsDict as CFDictionary) else { continue }
            out.append(WindowInfo(
                id: (w[kCGWindowNumber as String] as? CGWindowID) ?? 0,
                pid: (w[kCGWindowOwnerPID as String] as? pid_t) ?? 0,
                owner: (w[kCGWindowOwnerName as String] as? String) ?? "",
                title: (w[kCGWindowName as String] as? String) ?? "",
                rect: rect,
                layer: (w[kCGWindowLayer as String] as? Int) ?? 0,
                onScreen: (w[kCGWindowIsOnscreen as String] as? Bool) ?? false))
        }
        return out
    }

    static func onScreen() -> [WindowInfo] { list(all: false) }

    // Largest normal-layer window owned by `pid` (the content window, not a panel).
    static func largestWindow(pid: pid_t) -> WindowInfo? {
        onScreen()
            .filter { $0.pid == pid && $0.layer == 0 }
            .max { $0.rect.width * $0.rect.height < $1.rect.width * $1.rect.height }
    }
}
