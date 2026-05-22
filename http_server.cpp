/*
 * http_server.cpp  -  BigBoyAgent TQC Brain | Taha Iqbal
 *
 * ── BUGS FIXED IN THIS VERSION ──────────────────────────────────────────────
 *
 * BUG-HS1  start(): server_fd not closed before throw on bind()/listen() fail.
 *
 *          socket() returns a valid fd.  If bind() or listen() subsequently
 *          fails and we throw, the fd is never closed — a file-descriptor leak.
 *          On HuggingFace Spaces (limited to 1024 fds per process), repeated
 *          restarts (e.g. from a mis-typed port) exhaust the fd table, making
 *          all subsequent socket() calls return -1 with EMFILE.  The Brain
 *          then refuses to bind even after the underlying problem is fixed.
 *
 *          FIX: RAII FdGuard struct closes the fd when it goes out of scope.
 *          Wrapped around server_fd in start() — the guard releases ownership
 *          only after listen() succeeds.  On any throw, the guard's destructor
 *          fires and closes the fd cleanly.
 *
 * BUG-HS2  workerThread(): ci_find searches the full recv_buf (header + body).
 *
 *          ci_find returns the first occurrence of "content-length:" anywhere
 *          in the buffer.  A POST body containing that string (e.g. a nested
 *          JSON from the executor's /predict payload that has a "content-length"
 *          key, or any base64-encoded data that happens to decode to those
 *          bytes) causes ci_find to return a body offset instead of the header
 *          offset.  The subsequent:
 *            std::atoi(recv_buf.c_str() + cl_pos + 15_or_16)
 *          parses JSON body content as an integer, producing cl = 0 or a
 *          garbage value.  With cl = 0:
 *            body_received >= cl → true immediately
 *          The recv loop exits before all body bytes have arrived.  The partial
 *          body is passed to the JSON parser in the handler, which throws or
 *          returns a 400.  The executor retries, hitting the same issue
 *          indefinitely — effectively blocking all /predict traffic whenever
 *          the body contains that string.
 *
 *          FIX: bound ci_find to [0, header_end] — the header section only.
 *          The body starts at header_end + 4 ("\r\n\r\n"); ci_find now stops
 *          before that boundary.
 *
 * BUG-HS3  workerThread(): recv_buf[cl_pos + 15] unchecked bounds access.
 *
 *          After ci_find locates "content-length:" at cl_pos, the code reads:
 *            recv_buf[cl_pos + 15]
 *          to check whether the next character is a space (for the value
 *          start offset).  If the header line is exactly "content-length:"
 *          with nothing after it (malformed but legal to receive), then
 *          cl_pos + 15 == recv_buf.size() — an out-of-bounds read.
 *          On most platforms this reads one byte past the string's null
 *          terminator — typically '\0' — producing cl_pos + 15 as the
 *          value start offset, which makes atoi parse the next header line
 *          or body content as the Content-Length value.
 *
 *          FIX: guard with an explicit bounds check before accessing
 *          recv_buf[cl_pos + 15].  If out of bounds, skip this header
 *          (treat as cl = 0, meaning no body).
 *
 * Pre-existing fixes (preserved from original):
 *   FIX 1: strncasecmp is POSIX, not std:: — correct call is strncasecmp(a,b,n).
 *   FIX 2: MPSCRingBuffer::push takes T&& — std::move() on lvalue client_fd.
 *   FIX 3: parseRequest raw IS req.raw_buf — no move; all string_views valid.
 *   FIX (ISSUE-11): "Connection: close" — not keep-alive.
 *   FIX (THREAD SAFETY): gmtime_r() instead of gmtime() — thread-safe.
 *   FIX (ISSUE-content-length): ci_find uses strncasecmp for case-insensitive
 *     Content-Length header matching (all casing variants, per RFC 7230).
 *
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include "http_server.hpp"
#include "config.hpp"

#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <strings.h>    // strncasecmp (POSIX, not std::)

#include <cstring>
#include <cstdio>
#include <ctime>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <thread>

namespace tqc {

// ── RAII fd guard (BUG-HS1 FIX) ──────────────────────────────────────────────
// Ensures server_fd is closed on any exception path in start().
// release() is called after listen() succeeds to transfer ownership to the
// acceptorThread() logic which closes the fd explicitly on shutdown.
struct FdGuard {
    int fd;
    explicit FdGuard(int f) noexcept : fd(f) {}
    ~FdGuard() noexcept { if (fd >= 0) ::close(fd); }
    int  release() noexcept { int f = fd; fd = -1; return f; }
    FdGuard(const FdGuard&) = delete;
    FdGuard& operator=(const FdGuard&) = delete;
};

// ── Route registration ────────────────────────────────────────────────────────
void HttpServer::addRoute(const char* method, const char* path,
                          bool require_auth, HandlerFn handler) {
    routes_.push_back({method, path, require_auth, std::move(handler)});
}

// ── Auth check ────────────────────────────────────────────────────────────────
bool HttpServer::checkAuth(const HttpRequest& req) const noexcept {
    const char*       secret = globalConfig().secret_key;
    const std::size_t slen   = std::strlen(secret);
    if (!req.api_key.empty() &&
        req.api_key.size() == slen &&
        std::strncmp(req.api_key.data(), secret, slen) == 0)
        return true;
    if (!req.auth_bearer.empty() &&
        req.auth_bearer.size() == slen &&
        std::strncmp(req.auth_bearer.data(), secret, slen) == 0)
        return true;
    return false;
}

// ── Request parser ────────────────────────────────────────────────────────────
// raw IS req.raw_buf — no move needed; all string_views into raw remain valid.
bool HttpServer::parseRequest(std::string& raw, HttpRequest& req) const noexcept {
    const std::size_t method_end = raw.find(' ');
    if (method_end == std::string::npos) return false;

    const std::size_t path_start = method_end + 1;
    const std::size_t path_end   = raw.find(' ', path_start);
    if (path_end == std::string::npos) return false;

    req.method = std::string_view(raw.data(),              method_end);
    req.path   = std::string_view(raw.data() + path_start, path_end - path_start);

    std::size_t pos = raw.find("\r\n");
    if (pos == std::string::npos) return false;
    pos += 2;

    int content_length = 0;
    while (pos < raw.size()) {
        const std::size_t end = raw.find("\r\n", pos);
        if (end == std::string::npos) break;
        if (end == pos) { pos += 2; break; }

        std::string_view line(raw.data() + pos, end - pos);
        const std::size_t colon = line.find(':');
        if (colon != std::string_view::npos) {
            std::string_view name  = line.substr(0, colon);
            std::string_view value = line.substr(colon + 1);
            while (!value.empty() && value[0] == ' ') value.remove_prefix(1);

            if (name.size() == 9 && strncasecmp(name.data(), "x-api-key", 9) == 0)
                req.api_key = value;

            if (name.size() == 13 &&
                strncasecmp(name.data(), "authorization", 13) == 0) {
                req.auth_bearer = (value.size() > 7 &&
                                   value.substr(0, 7) == "Bearer ")
                                  ? value.substr(7) : value;
            }

            if (name.size() == 14 &&
                strncasecmp(name.data(), "content-length", 14) == 0) {
                // Use strtol directly on string_view data; no heap allocation.
                // value.data() is guaranteed non-null (points into raw_buf).
                content_length = static_cast<int>(
                    std::strtol(value.data(), nullptr, 10));
            }
        }
        pos = end + 2;
    }

    if (content_length > 0 && pos < raw.size())
        req.body = std::string_view(raw.data() + pos,
                       std::min(static_cast<std::size_t>(content_length),
                                raw.size() - pos));
    return true;
}

// ── Response formatter ────────────────────────────────────────────────────────
std::string HttpServer::formatResponse(const HttpResponse& resp) const {
    const char* status_text = "OK";
    switch (resp.status) {
        case 200: status_text = "OK";           break;
        case 400: status_text = "Bad Request";  break;
        case 403: status_text = "Forbidden";    break;
        case 404: status_text = "Not Found";    break;
        case 500: status_text = "Server Error"; break;
        default:  status_text = "Unknown";      break;
    }

    // FIX (THREAD SAFETY): gmtime_r writes into a stack-local struct.
    // std::gmtime() returns a pointer to a shared static — data race under
    // concurrent worker threads calling formatResponse() simultaneously.
    char     date_buf[64];
    time_t   now_t = std::time(nullptr);
    struct tm tm_buf{};
    gmtime_r(&now_t, &tm_buf);
    std::strftime(date_buf, sizeof(date_buf),
                  "%a, %d %b %Y %H:%M:%S GMT", &tm_buf);

    std::string out;
    out.reserve(resp.body.size() + 256);
    char header[256];
    // FIX (ISSUE-11): "Connection: close" — server closes socket after every
    // response.  "Connection: keep-alive" was a protocol lie that caused
    // Python requests.Session to attempt reuse on a closed fd → ConnectionResetError.
    std::snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Date: %s\r\n"
        "Content-Type: %s; charset=utf-8\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n"
        "\r\n",
        resp.status, status_text,
        date_buf,
        resp.content_type.c_str(),
        resp.body.size());
    out += header;
    out += resp.body;
    return out;
}

// ── Dispatcher ────────────────────────────────────────────────────────────────
HttpResponse HttpServer::dispatch(const HttpRequest& req) const {
    for (const auto& route : routes_) {
        if (req.method == route.method && req.path == route.path) {
            if (route.auth && !checkAuth(req)) [[unlikely]]
                return {403, R"({"detail":"Unauthorized"})"};
            return route.handler(req);
        }
    }
    return {404, R"({"detail":"Not Found"})"};
}

// ── Worker thread ─────────────────────────────────────────────────────────────
void HttpServer::workerThread() {
    std::string recv_buf;
    recv_buf.reserve(64 * 1024);  // allocated ONCE at thread startup

    while (running_.load(std::memory_order_acquire)) {
        auto opt_fd = fd_queue_.pop();
        if (!opt_fd.has_value()) {
            std::this_thread::yield();
            continue;
        }
        const int client_fd = *opt_fd;

        // Set recv timeout on the client socket.
        // Without this, a client that connects then sends nothing holds a worker
        // thread permanently — effectively a single-connection DoS.
        {
            struct timeval ctv{5, 0};
            setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &ctv, sizeof(ctv));
        }

        recv_buf.clear();
        char    chunk[4096];
        ssize_t n;
        while ((n = recv(client_fd, chunk, sizeof(chunk), 0)) > 0) {
            recv_buf.append(chunk, static_cast<std::size_t>(n));

            const std::size_t header_end = recv_buf.find("\r\n\r\n");
            if (header_end != std::string::npos) {

                // BUG-HS2 FIX: search only the header section [0, header_end].
                //
                // BEFORE: ci_find searched recv_buf from index 0 to size()-nlen,
                //   including the request body.  A body containing the string
                //   "content-length:" (e.g. nested JSON from executor, or base64
                //   data) caused ci_find to return a body offset.  atoi then
                //   parsed JSON content as an integer → cl=0 or garbage → recv
                //   loop exited immediately with partial body → handler received
                //   truncated JSON → 400 errors on every affected request.
                //
                // AFTER: search_end = header_end ensures ci_find only inspects
                //   the header section of the buffer, never the body.
                auto ci_find = [&](const char* needle) -> std::size_t {
                    const std::size_t nlen       = std::strlen(needle);
                    const std::size_t search_end = header_end;  // BUG-HS2 FIX
                    if (search_end < nlen || recv_buf.size() < nlen)
                        return std::string::npos;
                    const std::size_t limit = std::min(search_end, recv_buf.size() - nlen);
                    for (std::size_t i = 0; i <= limit; ++i)
                        if (strncasecmp(recv_buf.c_str() + i, needle, nlen) == 0)
                            return i;
                    return std::string::npos;
                };

                const std::size_t cl_pos = ci_find("content-length:");
                if (cl_pos == std::string::npos) break;  // GET or no body

                // BUG-HS3 FIX: bounds-check before accessing recv_buf[cl_pos+15].
                //
                // BEFORE: recv_buf[cl_pos + 15] was accessed unconditionally.
                //   A malformed header "content-length:" with no value (15 chars
                //   then CRLF) means cl_pos + 15 == recv_buf.size() — the
                //   subscript operator reads one byte past the buffer end (UB).
                //   std::string guarantees a null terminator at data()[size()],
                //   so this read always returns '\0' on compliant implementations,
                //   but it is still formally undefined behaviour per the standard.
                //
                // AFTER: explicit check that cl_pos + 15 < recv_buf.size() before
                //   the subscript.  If the header has no value, cl = 0 and no body
                //   is expected — the recv loop exits naturally.
                int cl = 0;
                if (cl_pos + 15 < recv_buf.size()) {
                    const std::size_t val_start =
                        (recv_buf[cl_pos + 15] == ' ') ? cl_pos + 16 : cl_pos + 15;
                    if (val_start < recv_buf.size())
                        cl = std::atoi(recv_buf.c_str() + val_start);
                }

                const int body_received =
                    static_cast<int>(recv_buf.size()) -
                    static_cast<int>(header_end) - 4;
                if (body_received >= cl) break;
            }
        }

        if (recv_buf.empty()) { close(client_fd); continue; }

        HttpRequest req;
        req.raw_buf = recv_buf;
        const bool ok = parseRequest(req.raw_buf, req);

        HttpResponse resp = ok ? dispatch(req)
                               : HttpResponse{400, R"({"error":"malformed_request"})"};

        const std::string wire = formatResponse(resp);
        const char*  ptr = wire.data();
        std::size_t  rem = wire.size();
        while (rem > 0) {
            const ssize_t sent = send(client_fd, ptr, rem, MSG_NOSIGNAL);
            if (sent <= 0) break;
            ptr += sent;
            rem -= static_cast<std::size_t>(sent);
        }

        close(client_fd);
    }
}

// ── Acceptor thread ───────────────────────────────────────────────────────────
void HttpServer::acceptorThread(int server_fd) {
    while (running_.load(std::memory_order_acquire)) {
        struct sockaddr_in client_addr{};
        socklen_t addr_len = sizeof(client_addr);
        const int client_fd = accept(server_fd,
                                     reinterpret_cast<struct sockaddr*>(&client_addr),
                                     &addr_len);
        if (client_fd < 0) {
            if (errno == EINTR || errno == EWOULDBLOCK || errno == EAGAIN) continue;
            break;
        }

        bool pushed = false;
        while (!pushed && running_.load(std::memory_order_relaxed)) {
            // std::move on int is equivalent to copy; written for type-correctness
            // with MPSCRingBuffer::push(T&&).
            pushed = fd_queue_.push(std::move(client_fd));
            if (!pushed) std::this_thread::yield();
        }
        if (!pushed) close(client_fd);
    }
}

// ── Start (blocks until stop()) ───────────────────────────────────────────────
void HttpServer::start() {
    const int raw_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (raw_fd < 0) throw std::runtime_error("socket() failed");

    // BUG-HS1 FIX: FdGuard ensures server_fd is closed on any exception.
    // BEFORE: throw from bind() or listen() leaked the fd.  On HuggingFace
    //   Spaces (fd limit ~1024), repeated failed starts exhaust the fd table,
    //   making all future socket() calls fail with EMFILE.
    // AFTER: FdGuard destructor calls close(fd) if we throw before release().
    //   After listen() succeeds, release() transfers ownership to the caller
    //   scope — the fd is then closed explicitly at the end of start().
    FdGuard guard(raw_fd);

    int opt = 1;
    setsockopt(raw_fd, SOL_SOCKET,  SO_REUSEADDR, &opt, sizeof(opt));
#ifdef SO_REUSEPORT
    setsockopt(raw_fd, SOL_SOCKET,  SO_REUSEPORT, &opt, sizeof(opt));
#endif
    setsockopt(raw_fd, IPPROTO_TCP, TCP_NODELAY,  &opt, sizeof(opt));

    struct timeval tv{5, 0};
    setsockopt(raw_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    struct sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(static_cast<uint16_t>(port_));

    if (bind(raw_fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0)
        throw std::runtime_error("bind() failed on port " + std::to_string(port_));
        // FdGuard destructor closes raw_fd here if bind() throws or we re-throw.

    if (listen(raw_fd, 1024) < 0)
        throw std::runtime_error("listen() failed");
        // FdGuard destructor closes raw_fd here if listen() throws or we re-throw.

    // listen() succeeded — release ownership from the guard.
    // The fd is now managed by this function scope and closed at the end.
    const int server_fd = guard.release();

    std::fprintf(stderr, "[HTTP] Listening on 0.0.0.0:%d  workers=%d\n",
                 port_, n_workers_);

    std::vector<std::thread> workers;
    workers.reserve(n_workers_);
    for (int i = 0; i < n_workers_; ++i)
        workers.emplace_back([this]{ workerThread(); });

    acceptorThread(server_fd);  // blocks until stop()

    close(server_fd);
    for (auto& t : workers) if (t.joinable()) t.join();
}

} // namespace tqc
