import Foundation
#if canImport(Darwin)
import Darwin

private let streamSocket = SOCK_STREAM
/// Darwin turns SIGPIPE off per socket (`SO_NOSIGPIPE`, set on accept).
private let sendFlags: Int32 = 0
private let socketSend = Darwin.send
#else
import Glibc

private let streamSocket = Int32(SOCK_STREAM.rawValue)
/// Linux has no `SO_NOSIGPIPE`; each send asks not to raise SIGPIPE instead.
private let sendFlags = Int32(MSG_NOSIGNAL)
private let socketSend = Glibc.send
#endif

/// A minimal HTTP/1.1 server over POSIX sockets, on a Unix socket path or a TCP port.
///
/// It reads one request at a time per connection (keep-alive is supported), with a `Content-Length` or
/// chunked body, and answers with either a complete response or a chunked stream (server-sent events).
/// Each connection gets its own thread; what the handler does with concurrency is its business.
public final class HTTPServer: @unchecked Sendable {
    public enum Endpoint {
        case unixSocket(path: String)
        case tcp(host: String, port: UInt16)
    }

    public struct Request {
        public var method: String
        /// The path without the query string.
        public var path: String
        public var query: String
        /// Header names lowercased.
        public var headers: [String: String]
        public var body: Data
    }

    /// Writes one response, whole or streamed. A write that fails (the client went away) returns false.
    public final class ResponseWriter {
        let fd: Int32
        private(set) var streaming = false
        private(set) var finished = false

        init(fd: Int32) { self.fd = fd }

        static func reason(_ status: Int) -> String {
            switch status {
            case 200: "OK"
            case 204: "No Content"
            case 400: "Bad Request"
            case 404: "Not Found"
            case 405: "Method Not Allowed"
            case 408: "Request Timeout"
            case 413: "Payload Too Large"
            case 500: "Internal Server Error"
            case 501: "Not Implemented"
            case 503: "Service Unavailable"
            default: "Status"
            }
        }

        @discardableResult
        public func send(status: Int, contentType: String = "application/json; charset=utf-8", body: Data) -> Bool {
            precondition(!streaming && !finished)
            finished = true
            let head = "HTTP/1.1 \(status) \(Self.reason(status))\r\nContent-Type: \(contentType)\r\nContent-Length: \(body.count)\r\n\r\n"
            return write(Data(head.utf8) + body)
        }

        /// Starts a chunked response; follow with ``chunk(_:)`` and ``end()``.
        @discardableResult
        public func beginStream(status: Int = 200, contentType: String = "text/event-stream") -> Bool {
            precondition(!streaming && !finished)
            streaming = true
            let head = "HTTP/1.1 \(status) \(Self.reason(status))\r\nContent-Type: \(contentType)\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\n\r\n"
            return write(Data(head.utf8))
        }

        @discardableResult
        public func chunk(_ data: Data) -> Bool {
            precondition(streaming && !finished)
            guard !data.isEmpty else { return true }
            return write(Data(String(data.count, radix: 16).utf8) + Data("\r\n".utf8) + data + Data("\r\n".utf8))
        }

        @discardableResult
        public func end() -> Bool {
            precondition(streaming && !finished)
            finished = true
            return write(Data("0\r\n\r\n".utf8))
        }

        fileprivate func write(_ data: Data) -> Bool {
            data.withUnsafeBytes { raw -> Bool in
                var offset = 0
                while offset < raw.count {
                    let n = socketSend(fd, raw.baseAddress! + offset, raw.count - offset, sendFlags)
                    if n < 0 {
                        if errno == EINTR { continue }
                        return false
                    }
                    offset += n
                }
                return true
            }
        }
    }

    public typealias Handler = @Sendable (Request, ResponseWriter) -> Void

    public struct Failure: Error, CustomStringConvertible {
        public var description: String
    }

    private let handler: Handler
    private let maxBodyBytes: Int
    private var listenFD: Int32 = -1
    private var running = false
    private var boundPath: String?
    private let lock = NSLock()
    private var connections = Set<Int32>()
    /// The TCP port actually bound (useful when 0 was asked for).
    public private(set) var port: UInt16 = 0

    public init(maxBodyBytes: Int = 64 << 20, handler: @escaping Handler) {
        self.handler = handler
        self.maxBodyBytes = maxBodyBytes
    }

    /// The longest Unix socket path `bind` accepts here.
    public static let maxSocketPathBytes = 103

    public func start(_ endpoint: Endpoint) throws {
        precondition(listenFD < 0, "already started")
        switch endpoint {
        case .unixSocket(let path):
            guard path.utf8.count <= Self.maxSocketPathBytes else {
                throw Failure(description: "socket path is \(path.utf8.count) bytes; the limit is \(Self.maxSocketPathBytes)")
            }
            var st = stat()
            if lstat(path, &st) == 0 {
                guard (st.st_mode & S_IFMT) == S_IFSOCK else {
                    throw Failure(description: "\(path) exists and is not a socket")
                }
                unlink(path)
            }
            let fd = socket(AF_UNIX, streamSocket, 0)
            guard fd >= 0 else { throw Failure(description: "socket: \(String(cString: strerror(errno)))") }
            var addr = sockaddr_un()
            addr.sun_family = sa_family_t(AF_UNIX)
            withUnsafeMutableBytes(of: &addr.sun_path) { buf in
                for (i, b) in path.utf8.enumerated() { buf[i] = b }
                buf[path.utf8.count] = 0
            }
            let size = socklen_t(MemoryLayout<sockaddr_un>.size)
            let rc = withUnsafePointer(to: &addr) {
                $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(fd, $0, size) }
            }
            guard rc == 0 else {
                let e = String(cString: strerror(errno))
                close(fd)
                throw Failure(description: "bind \(path): \(e)")
            }
            chmod(path, 0o600)
            boundPath = path
            listenFD = fd
        case .tcp(let host, let port):
            let fd = socket(AF_INET, streamSocket, 0)
            guard fd >= 0 else { throw Failure(description: "socket: \(String(cString: strerror(errno)))") }
            var one: Int32 = 1
            setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, socklen_t(MemoryLayout<Int32>.size))
            var addr = sockaddr_in()
            addr.sin_family = sa_family_t(AF_INET)
            addr.sin_port = port.bigEndian
            guard inet_pton(AF_INET, host, &addr.sin_addr) == 1 else {
                close(fd)
                throw Failure(description: "\(host) is not an IPv4 address")
            }
            let size = socklen_t(MemoryLayout<sockaddr_in>.size)
            let rc = withUnsafePointer(to: &addr) {
                $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(fd, $0, size) }
            }
            guard rc == 0 else {
                let e = String(cString: strerror(errno))
                close(fd)
                throw Failure(description: "bind \(host):\(port): \(e)")
            }
            var bound = sockaddr_in()
            var len = size
            _ = withUnsafeMutablePointer(to: &bound) {
                $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(fd, $0, &len) }
            }
            self.port = UInt16(bigEndian: bound.sin_port)
            listenFD = fd
        }
        guard listen(listenFD, 32) == 0 else {
            let e = String(cString: strerror(errno))
            stop()
            throw Failure(description: "listen: \(e)")
        }
        running = true
        let fd = listenFD
        let thread = Thread { [weak self] in self?.acceptLoop(fd) }
        thread.name = "shears-http-accept"
        thread.start()
    }

    public func stop() {
        lock.lock()
        running = false
        let open = connections
        lock.unlock()
        for c in open { shutdown(c, Int32(SHUT_RDWR)) }
        // The accept loop notices `running` within its poll interval and closes the descriptor.
        if let p = boundPath {
            unlink(p)
            boundPath = nil
        }
    }

    private func acceptLoop(_ fd: Int32) {
        while true {
            lock.lock()
            let go = running
            lock.unlock()
            if !go { break }
            var p = pollfd(fd: fd, events: Int16(POLLIN), revents: 0)
            let r = poll(&p, 1, 100)
            if r <= 0 { continue }
            let c = accept(fd, nil, nil)
            if c < 0 { continue }
            #if canImport(Darwin)
            var on: Int32 = 1
            setsockopt(c, SOL_SOCKET, SO_NOSIGPIPE, &on, socklen_t(MemoryLayout<Int32>.size))
            #endif
            lock.lock()
            connections.insert(c)
            lock.unlock()
            let t = Thread { [weak self] in
                self?.serve(c)
                close(c)
                self?.lock.lock()
                self?.connections.remove(c)
                self?.lock.unlock()
            }
            t.name = "shears-http-connection"
            t.start()
        }
        close(fd)
        lock.lock()
        listenFD = -1
        lock.unlock()
    }

    // MARK: Connection

    private func serve(_ fd: Int32) {
        var buffer = [UInt8]()
        while true {
            guard let request = readRequest(fd, &buffer) else { return }
            let writer = ResponseWriter(fd: fd)
            switch request {
            case .failure(let status, let message):
                let body = try? JSONSerialization.data(withJSONObject: ["error": ["code": status, "message": message, "type": "invalid_request_error"]])
                writer.send(status: status, body: body ?? Data())
                return
            case .request(let r):
                handler(r, writer)
                if !writer.finished {
                    // The handler wrote nothing complete: close rather than leave the client waiting.
                    return
                }
                if r.headers["connection"]?.lowercased() == "close" { return }
            }
        }
    }

    private enum Parsed {
        case request(Request)
        case failure(Int, String)
    }

    private func fill(_ fd: Int32, _ buffer: inout [UInt8]) -> Bool {
        var tmp = [UInt8](repeating: 0, count: 64 * 1024)
        while true {
            let n = read(fd, &tmp, tmp.count)
            if n > 0 {
                buffer.append(contentsOf: tmp[0 ..< n])
                return true
            }
            if n < 0, errno == EINTR { continue }
            return false
        }
    }

    private func readRequest(_ fd: Int32, _ buffer: inout [UInt8]) -> Parsed? {
        let separator: [UInt8] = [13, 10, 13, 10]
        var headEnd: Int?
        while headEnd == nil {
            if let r = buffer.firstRange(of: separator) { headEnd = r.lowerBound; break }
            if buffer.count > 64 * 1024 { return .failure(400, "request head too large") }
            if !fill(fd, &buffer) { return nil }
        }
        let head = String(decoding: buffer[0 ..< headEnd!], as: UTF8.self)
        buffer.removeFirst(headEnd! + 4)
        var lines = head.components(separatedBy: "\r\n")
        let first = lines.removeFirst().split(separator: " ")
        guard first.count == 3 else { return .failure(400, "malformed request line") }
        var headers: [String: String] = [:]
        for l in lines {
            guard let c = l.firstIndex(of: ":") else { continue }
            headers[l[..<c].lowercased()] = l[l.index(after: c)...].trimmingCharacters(in: .whitespaces)
        }
        let target = String(first[1])
        let q = target.firstIndex(of: "?")
        let path = q.map { String(target[..<$0]) } ?? target
        let query = q.map { String(target[target.index(after: $0)...]) } ?? ""

        var body = Data()
        if headers["expect"]?.lowercased() == "100-continue" {
            _ = ResponseWriter(fd: fd).write(Data("HTTP/1.1 100 Continue\r\n\r\n".utf8))
        }
        if headers["transfer-encoding"]?.lowercased().contains("chunked") == true {
            while true {
                while buffer.firstRange(of: [13, 10]) == nil { if !fill(fd, &buffer) { return nil } }
                let r = buffer.firstRange(of: [13, 10])!
                let sizeText = String(decoding: buffer[0 ..< r.lowerBound], as: UTF8.self).split(separator: ";").first.map(String.init) ?? ""
                guard let size = Int(sizeText.trimmingCharacters(in: .whitespaces), radix: 16) else {
                    return .failure(400, "bad chunk size")
                }
                buffer.removeFirst(r.upperBound)
                if size == 0 {
                    // Trailer, up to a blank line.
                    while true {
                        while buffer.firstRange(of: [13, 10]) == nil { if !fill(fd, &buffer) { return nil } }
                        let t = buffer.firstRange(of: [13, 10])!
                        let empty = t.lowerBound == 0
                        buffer.removeFirst(t.upperBound)
                        if empty { break }
                    }
                    break
                }
                if body.count + size > maxBodyBytes { return .failure(413, "request body too large") }
                while buffer.count < size + 2 { if !fill(fd, &buffer) { return nil } }
                body.append(contentsOf: buffer[0 ..< size])
                buffer.removeFirst(size + 2)
            }
        } else if let len = headers["content-length"] {
            guard let n = Int(len), n >= 0 else { return .failure(400, "bad Content-Length") }
            if n > maxBodyBytes { return .failure(413, "request body too large") }
            while buffer.count < n { if !fill(fd, &buffer) { return nil } }
            body = Data(buffer[0 ..< n])
            buffer.removeFirst(n)
        }
        return .request(Request(method: String(first[0]), path: path, query: query, headers: headers, body: body))
    }
}
