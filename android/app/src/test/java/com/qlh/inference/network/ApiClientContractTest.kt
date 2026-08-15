package com.qlh.inference.network

import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.io.BufferedInputStream
import java.io.IOException
import java.net.ServerSocket
import java.net.Socket

/**
 * ApiClient HTTP 契约测试（JVM，本地 ServerSocket 桩，无需设备/网络/新依赖）。
 *
 * 验证：请求方法/路径/JSON body、响应解析（数组与包装两种格式）、
 * 非 2xx fail-closed（Result.failure 而非抛异常）、404 删除语义、网络不可达
 * 返回 Result.failure 而不是崩溃。
 */
class ApiClientContractTest {

    private lateinit var server: HttpStub
    private lateinit var client: ApiClient

    @Before
    fun setUp() {
        server = HttpStub()
        server.start()
        client = ApiClient(baseUrl = "http://127.0.0.1:${server.port}")
    }

    @After
    fun tearDown() {
        server.stop()
    }

    private fun route(path: String, handler: (Request, (Int, String) -> Unit) -> Unit) {
        server.routes[path] = handler
    }

    private class Request(val method: String, val path: String, val body: String)

    /** 迷你 HTTP 桩：解析请求行/头/体，回调后写固定响应。 */
    private class HttpStub {
        private val serverSocket = ServerSocket(0)
        val port: Int get() = serverSocket.localPort
        val routes = mutableMapOf<String, (Request, (Int, String) -> Unit) -> Unit>()
        private var running = true
        private val thread = Thread {
            while (running) {
                val socket = try {
                    serverSocket.accept()
                } catch (_: IOException) {
                    break
                }
                Thread {
                    try {
                        handle(socket)
                    } catch (_: IOException) {
                        // 客户端主动断开等，忽略
                    } finally {
                        try {
                            socket.close()
                        } catch (_: IOException) {
                        }
                    }
                }.start()
            }
        }

        fun start() = thread.start()

        fun stop() {
            running = false
            try {
                serverSocket.close()
            } catch (_: IOException) {
            }
        }

        private fun handle(socket: Socket) {
            val input = BufferedInputStream(socket.getInputStream())
            fun readLine(): String? {
                val buf = java.io.ByteArrayOutputStream()
                while (true) {
                    val b = input.read()
                    if (b < 0) return null
                    if (b == '\n'.code) break
                    if (b != '\r'.code) buf.write(b)
                }
                return buf.toString(Charsets.UTF_8.name())
            }

            val requestLine = readLine() ?: return
            val parts = requestLine.split(" ")
            if (parts.size < 2) return
            val method = parts[0]
            val path = parts[1]

            var contentLength = 0
            while (true) {
                val line = readLine() ?: break
                if (line.isEmpty()) break
                val lower = line.lowercase()
                if (lower.startsWith("content-length:")) {
                    contentLength = line.substringAfter(':').trim().toIntOrNull() ?: 0
                }
            }
            val body = if (contentLength > 0) {
                val bytes = ByteArray(contentLength)
                var off = 0
                while (off < contentLength) {
                    val n = input.read(bytes, off, contentLength - off)
                    if (n < 0) break
                    off += n
                }
                bytes.toString(Charsets.UTF_8)
            } else {
                ""
            }

            val handler = routes[path]
            val (code, responseBody) = if (handler != null) {
                var code = 200
                var response = "{}"
                handler(Request(method, path, body)) { c, b ->
                    code = c
                    response = b
                }
                code to response
            } else {
                // 未知路径：记录但不匹配时返回 404
                404 to """{"error":"not found"}"""
            }

            val bytes = responseBody.toByteArray(Charsets.UTF_8)
            val out = socket.getOutputStream()
            out.write(
                ("HTTP/1.1 $code OK\r\n" +
                    "Content-Type: application/json\r\n" +
                    "Content-Length: ${bytes.size}\r\n" +
                    "Connection: close\r\n" +
                    "\r\n").toByteArray(Charsets.UTF_8)
            )
            out.write(bytes)
            out.flush()
        }
    }

    // ---- chat ----

    @Test
    fun `chat posts json body with message and parses response`() {
        var seen: Request? = null
        route("/api/chat") { req, reply ->
            seen = req
            reply(200, """{"content":"hello back"}""")
        }

        val result = runBlocking { client.chat(ChatRequest(message = "hello")) }
        assertTrue(result.isSuccess)
        assertEquals("hello back", result.getOrNull()?.content)
        assertEquals("POST", seen?.method)
        assertEquals("/api/chat", seen?.path)
        assertTrue(seen!!.body.contains("\"message\":\"hello\""))
        assertTrue(seen!!.body.contains("\"max_new_tokens\":1024"))
    }

    @Test
    fun `chat non-2xx returns failure not exception`() {
        route("/api/chat") { _, reply -> reply(500, """{"error":"boom"}""") }
        val result = runBlocking { client.chat(ChatRequest(message = "x")) }
        assertTrue(result.isFailure)
        assertTrue(result.exceptionOrNull() is IOException)
        assertTrue(result.exceptionOrNull()!!.message!!.contains("500"))
    }

    // ---- sessions ----

    @Test
    fun `getSessions parses plain array format`() {
        route("/api/sessions") { _, reply ->
            reply(200, """[{"id":"s1","title":"T1","message_count":2}]""")
        }
        val sessions = runBlocking { client.getSessions() }.getOrNull()!!
        assertEquals(1, sessions.size)
        assertEquals("s1", sessions[0].id)
        assertEquals("T1", sessions[0].title)
        assertEquals(2, sessions[0].messageCount)
    }

    @Test
    fun `getSessions parses wrapped object format`() {
        route("/api/sessions") { _, reply -> reply(200, """{"sessions":[{"id":"w1"}]}""") }
        val sessions = runBlocking { client.getSessions() }.getOrNull()!!
        assertEquals(1, sessions.size)
        assertEquals("w1", sessions[0].id)
    }

    @Test
    fun `createSession posts title and parses session`() {
        var seen: Request? = null
        route("/api/sessions") { req, reply ->
            seen = req
            reply(200, """{"id":"new-1","title":"我的对话"}""")
        }

        val result = runBlocking { client.createSession(title = "我的对话") }
        assertTrue("createSession failed: ${result.exceptionOrNull()}", result.isSuccess)
        val session = result.getOrNull()!!
        assertEquals("new-1", session.id)
        assertEquals("我的对话", session.title)
        assertEquals("POST", seen?.method)
        assertTrue(seen!!.body.contains("\"title\":\"我的对话\""))
    }

    @Test
    fun `deleteSession uses DELETE and treats 404 as success`() {
        var seen: String? = null
        route("/api/sessions/del-1") { req, reply ->
            seen = req.method
            reply(404, "")
        }
        val result = runBlocking { client.deleteSession("del-1") }
        assertTrue(result.isSuccess)
        assertEquals("DELETE", seen)
    }

    @Test
    fun `deleteSession other errors return failure`() {
        route("/api/sessions/del-2") { _, reply -> reply(500, "") }
        val result = runBlocking { client.deleteSession("del-2") }
        assertTrue(result.isFailure)
    }

    @Test
    fun `getSessionMessages parses array and rejects non-2xx`() {
        route("/api/sessions/s1/messages") { _, reply ->
            reply(200, """[{"role":"user","content":"hi"}]""")
        }
        val messages = runBlocking { client.getSessionMessages("s1") }.getOrNull()!!
        assertEquals(1, messages.size)
        assertEquals("user", messages[0].role)

        route("/api/sessions/s1/messages") { _, reply -> reply(403, "") }
        assertTrue(runBlocking { client.getSessionMessages("s1") }.isFailure)
    }

    // ---- cluster / connection ----

    @Test
    fun `getClusterStatus parses master and counters`() {
        route("/api/cluster/status") { _, reply ->
            reply(200, """{"online_count":2,"total_count":3,"distributed_enabled":true}""")
        }
        val status = runBlocking { client.getClusterStatus() }.getOrNull()!!
        assertEquals(2, status.onlineCount)
        assertEquals(3, status.totalCount)
        assertTrue(status.distributedEnabled)
    }

    @Test
    fun `testConnection returns success boolean on any http response`() {
        route("/api/cluster/status") { _, reply -> reply(200, "{}") }
        assertEquals(true, runBlocking { client.testConnection() }.getOrNull())

        route("/api/cluster/status") { _, reply -> reply(503, "") }
        assertEquals(false, runBlocking { client.testConnection() }.getOrNull())
    }

    @Test
    fun `registerAndroidNode posts to android register endpoint`() {
        var seen: Request? = null
        route("/api/cluster/android/register") { req, reply ->
            seen = req
            reply(200, """{"node_id":"n1","status":"registered"}""")
        }
        val response = runBlocking { client.registerAndroidNode(
            RegisterNodeRequest(nodeId = "n1", hostname = "node-1", address = "100.1.2.3")
        ) }.getOrNull()!!
        assertEquals("n1", response.nodeId)
        assertEquals("POST", seen?.method)
        assertEquals("/api/cluster/android/register", seen?.path)
        assertTrue("body was: ${seen!!.body}", seen!!.body.contains("\"node_id\":\"n1\""))
    }

    @Test
    fun `firstConnectBootstrap posts to bootstrap endpoint`() {
        var seen: Request? = null
        route("/api/bootstrap/first-connect") { req, reply ->
            seen = req
            reply(200, """{"cluster":{"cluster_id":"c1","master_api_host":"100.1.2.3"}}""")
        }
        val response = runBlocking { client.firstConnectBootstrap(
            BootstrapRequest(nodeId = "n1", hostname = "100.1.2.3")
        ) }.getOrNull()!!
        assertEquals("c1", response.cluster.clusterId)
        assertEquals("/api/bootstrap/first-connect", seen?.path)
    }

    @Test
    fun `network unreachable returns failure`() {
        val unreachable = ApiClient(baseUrl = "http://127.0.0.1:1")
        val result = runBlocking { unreachable.chat(ChatRequest(message = "x")) }
        assertTrue(result.isFailure)
    }
}
