import Foundation
import ScreenCaptureKit
import CoreMedia
import CoreVideo

// One captured frame, reduced to scalars in the callback. No buffers escape.
struct FrameSample {
    var hostTicks: UInt64      // mach_absolute_time at callback entry
    var displayTicks: UInt64   // SCStreamFrameInfo.displayTime (mach ticks)
    var lumaMean: Double       // 0..255 mean over a sampled grid, or -1 if skipped
    var complete: Bool         // status == .complete
}

final class Capture: NSObject, SCStreamOutput, SCStreamDelegate {
    private var stream: SCStream?
    private let queue = DispatchQueue(label: "eye.capture", qos: .userInteractive)
    private let onFrame: (FrameSample) -> Void
    private let wantLuma: Bool

    // Pulled out of the hot path so the callback allocates nothing.
    private let lumaStride: Int

    // Attachment keys resolved once, held as retained CFStrings.
    static let statusKey = SCStreamFrameInfo.status.rawValue as CFString
    static let displayTimeKey = SCStreamFrameInfo.displayTime.rawValue as CFString

    init(wantLuma: Bool, lumaStride: Int = 4, onFrame: @escaping (FrameSample) -> Void) {
        self.wantLuma = wantLuma
        self.lumaStride = max(1, lumaStride)
        self.onFrame = onFrame
    }

    struct RectPts { var x: Double; var y: Double; var w: Double; var h: Double }

    // Capture ONE window by id, wherever it is (another Desktop/Space, behind other
    // windows). SCK window filters render the window's own surface, so the user's
    // screen and focus are untouched. rect is in window-local points.
    func start(windowID: CGWindowID, rect: RectPts?, outW: Int, outH: Int) async throws {
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
        guard let win = content.windows.first(where: { $0.windowID == windowID }) else {
            throw NSError(domain: "eye", code: 2, userInfo: [NSLocalizedDescriptionKey: "window \(windowID) not found"])
        }
        let filter = SCContentFilter(desktopIndependentWindow: win)
        let cfg = SCStreamConfiguration()
        if let r = rect { cfg.sourceRect = CGRect(x: r.x, y: r.y, width: r.w, height: r.h) }
        try await startStream(filter: filter, cfg: cfg, outW: outW, outH: outH)
    }

    func start(rect: RectPts, outW: Int, outH: Int) async throws {
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
        guard let display = content.displays.first else {
            throw NSError(domain: "eye", code: 1, userInfo: [NSLocalizedDescriptionKey: "no display"])
        }

        let filter = SCContentFilter(display: display, excludingWindows: [])

        let cfg = SCStreamConfiguration()
        cfg.sourceRect = CGRect(x: rect.x, y: rect.y, width: rect.w, height: rect.h)
        try await startStream(filter: filter, cfg: cfg, outW: outW, outH: outH)
    }

    private func startStream(filter: SCContentFilter, cfg: SCStreamConfiguration, outW: Int, outH: Int) async throws {
        cfg.width = outW
        cfg.height = outH
        cfg.minimumFrameInterval = CMTime(value: 1, timescale: 120)
        cfg.queueDepth = 3
        cfg.pixelFormat = kCVPixelFormatType_32BGRA
        cfg.showsCursor = false
        cfg.scalesToFit = true
        // Keep color handling simple and consistent for luma math.
        cfg.colorSpaceName = CGColorSpace.sRGB

        let s = SCStream(filter: filter, configuration: cfg, delegate: self)
        try s.addStreamOutput(self, type: .screen, sampleHandlerQueue: queue)
        try await s.startCapture()
        self.stream = s
    }

    func stop() async {
        if let s = stream { try? await s.stopCapture() }
        stream = nil
    }

    // Hot path. Runs on `queue`. No allocation here.
    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        let host = mach_absolute_time()
        guard type == .screen else { return }
        guard let arr = CMSampleBufferGetSampleAttachmentsArray(sampleBuffer, createIfNecessary: false) else { return }
        let n = CFArrayGetCount(arr)
        if n == 0 { return }
        let dictPtr = CFArrayGetValueAtIndex(arr, 0)
        let info = unsafeBitCast(dictPtr, to: CFDictionary.self)

        // status (CFNumber -> Int). No bridging, no alloc.
        var statusRaw: Int = -1
        _ = cfNumber(info, Capture.statusKey, &statusRaw, .intType)
        let complete = (statusRaw == SCFrameStatus.complete.rawValue)

        // displayTime in mach ticks (CFNumber -> UInt64).
        var displayTicks: UInt64 = 0
        _ = cfNumber(info, Capture.displayTimeKey, &displayTicks, .sInt64Type)

        var luma: Double = -1
        if wantLuma && complete, let pb = CMSampleBufferGetImageBuffer(sampleBuffer) {
            luma = meanLuma(pb)
        }

        onFrame(FrameSample(hostTicks: host, displayTicks: displayTicks, lumaMean: luma, complete: complete))
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        FileHandle.standardError.write("capture stopped with error: \(error)\n".data(using: .utf8)!)
    }

    // sRGB BGRA mean luminance over a sampled grid. Reads IOSurface in place.
    @inline(__always)
    private func meanLuma(_ pb: CVPixelBuffer) -> Double {
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return -1 }
        let w = CVPixelBufferGetWidth(pb)
        let h = CVPixelBufferGetHeight(pb)
        let rowBytes = CVPixelBufferGetBytesPerRow(pb)
        let p = base.assumingMemoryBound(to: UInt8.self)
        var sum: UInt64 = 0
        var count: UInt64 = 0
        let step = lumaStride
        var y = 0
        while y < h {
            let row = p + y * rowBytes
            var x = 0
            while x < w {
                let o = x * 4
                let b = UInt64(row[o])
                let g = UInt64(row[o + 1])
                let r = UInt64(row[o + 2])
                // Rec.601 luma, integer weights (sum = 256). Cheap and monotone.
                sum += (r * 77 + g * 150 + b * 29) >> 8
                count += 1
                x += step
            }
            y += step
        }
        if count == 0 { return -1 }
        return Double(sum) / Double(count)
    }
}

// Read a CFNumber out of a CFDictionary into `out` with no Swift bridging.
@inline(__always)
private func cfNumber<T>(_ dict: CFDictionary, _ key: CFString, _ out: inout T, _ type: CFNumberType) -> Bool {
    let keyPtr = Unmanaged.passUnretained(key).toOpaque()
    guard let valuePtr = CFDictionaryGetValue(dict, keyPtr) else { return false }
    let num = unsafeBitCast(valuePtr, to: CFNumber.self)
    return withUnsafeMutablePointer(to: &out) { CFNumberGetValue(num, type, $0) }
}
