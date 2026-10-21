import Foundation

// mach_absolute_time ticks -> nanoseconds, using the timebase once.
enum Clock {
    static let timebase: mach_timebase_info_data_t = {
        var tb = mach_timebase_info_data_t()
        mach_timebase_info(&tb)
        return tb
    }()

    @inline(__always) static func now() -> UInt64 { mach_absolute_time() }

    @inline(__always) static func ticksToNanos(_ ticks: UInt64) -> Double {
        Double(ticks) * Double(timebase.numer) / Double(timebase.denom)
    }

    @inline(__always) static func ticksToMillis(_ ticks: UInt64) -> Double {
        ticksToNanos(ticks) / 1_000_000.0
    }

    // Signed delta in ms (a - b), handling the usual UInt64 ordering.
    @inline(__always) static func deltaMillis(_ a: UInt64, _ b: UInt64) -> Double {
        if a >= b { return ticksToMillis(a - b) }
        return -ticksToMillis(b - a)
    }
}

// Percentile over a copy sorted in place. Nearest-rank, clamped.
func percentile(_ values: [Double], _ p: Double) -> Double {
    if values.isEmpty { return .nan }
    let s = values.sorted()
    let idx = Int((p / 100.0) * Double(s.count - 1)).clamped(0, s.count - 1)
    return s[idx]
}

func mean(_ values: [Double]) -> Double {
    if values.isEmpty { return .nan }
    return values.reduce(0, +) / Double(values.count)
}

extension Int {
    func clamped(_ lo: Int, _ hi: Int) -> Int { Swift.min(Swift.max(self, lo), hi) }
}

func fmt(_ x: Double, _ places: Int = 2) -> String {
    if x.isNaN { return "nan" }
    return String(format: "%.\(places)f", x)
}
