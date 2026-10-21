import Foundation

/// What kind of JSON value a `JSONSerialization` number came from. CoreFoundation's type checks
/// are Apple-only; elsewhere Foundation hands back the same two boolean singletons, so identity
/// tells a `true`/`false` apart from `1`/`0`, and the stored C type tells a float from an integer.
extension NSNumber {
    var isBoolean: Bool {
        #if canImport(Darwin)
        return CFGetTypeID(self) == CFBooleanGetTypeID()
        #else
        return self === (true as NSNumber) || self === (false as NSNumber)
        #endif
    }

    var isFloatingPoint: Bool {
        #if canImport(Darwin)
        return CFNumberIsFloatType(self)
        #else
        let t = String(cString: objCType)
        return t == "f" || t == "d"
        #endif
    }
}

#if !canImport(ObjectiveC)
/// Apple's `autoreleasepool` drains Objective-C temporaries; without the Objective-C runtime there
/// are none, so the body just runs.
@inline(__always)
func autoreleasepool<Result>(invoking body: () throws -> Result) rethrows -> Result {
    try body()
}
#endif
