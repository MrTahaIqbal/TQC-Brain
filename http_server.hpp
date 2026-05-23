#pragma once
/*
 * http_server.hpp  -  TQC Brain | Taha Iqbal
 *
 * Production-grade POSIX HTTP/1.1 server — no external libraries.
 *
 * Architecture:
 *   - One acceptor thread: calls accept(), pushes fd into MPSC ring buffer.
 *   - N worker threads: pop fd, recv request, dispatch handler, send response.
 *   - MPSCRingBuffer<int,256> between acceptor and workers (lock-free IPC).
 *
 * Response memory:
 *   Per-thread recv_buf allocated ONCE at thread startup (64 KiB).
 *   Zero heap allocation per request on the recv/parse path.
 *
 * FIX (ISSUE-11): "Connection: close" — server always closes the socket
 *   after responding.  "Connection: keep-alive" was a protocol lie that caused
 *   Python requests.Session to attempt reuse on a closed fd (ConnectionResetError).
 *
 * BUG-HS1 FIX: start() now closes server_fd before throwing on bind()/listen()
 *   failure — no fd leak on startup errors.
 *
 * BUG-HS2 FIX: ci_find in workerThread() now searches only the header section
 *   (bytes 0..header_end), not the full recv_buf including the body.
 *   A POST body containing "content-length:" (e.g. nested JSON from executor)
 *   previously caused ci_find to return the wrong offset, mis-parsing the
 *   Content-Length value and either dropping body bytes or blocking forever.
 *
 * BUG-HS3 FIX: recv_buf[cl_pos + 15] bounds-checked before access.
 *   Without the check, a malformed header line exactly 15 chars long (no value)
 *   caused UB via out-of-bounds read.
 */

#include "ring_buffer.hpp"
#include <functional>
#include <string>
#include <string_view>
#include <vector>
#include <thread>
#include <atomic>
#include <cstddef>

namespace tqc {

struct HttpRequest {
    std::string_view method;
    std::string_view path;
    std::string_view api_key;
    std::string_view auth_bearer;
    std::string_view body;
    std::string      raw_buf;  // backing store — all string_views point into this
};

struct HttpResponse {
    int         status       = 200;
    std::string body;
    std::string content_type = "application/json";
};

using HandlerFn = std::function<HttpResponse(const HttpRequest&)>;

struct Route {
    const char* method;
    const char* path;
    bool        auth;
    HandlerFn   handler;
};

class HttpServer {
public:
    explicit HttpServer(int port = 7860, int n_workers = 4) noexcept
        : port_(port), n_workers_(n_workers) {}

    void addRoute(const char* method, const char* path,
                  bool require_auth, HandlerFn handler);

    void start();   // blocks until stop()
    void stop() noexcept { running_.store(false, std::memory_order_release); }

private:
    int                      port_;
    int                      n_workers_;
    std::atomic<bool>        running_{true};
    std::vector<Route>       routes_;
    MPSCRingBuffer<int, 256> fd_queue_{};

    void acceptorThread(int server_fd);
    void workerThread();

    [[nodiscard]] bool        parseRequest  (std::string& raw,
                                             HttpRequest& req) const noexcept;
    [[nodiscard]] std::string formatResponse(const HttpResponse& resp) const;
    [[nodiscard]] HttpResponse dispatch     (const HttpRequest& req) const;
    [[nodiscard]] bool        checkAuth     (const HttpRequest& req) const noexcept;
};

} // namespace tqc
