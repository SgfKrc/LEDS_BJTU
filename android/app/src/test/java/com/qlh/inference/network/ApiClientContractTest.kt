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
    fun `chat serializes remote image data urls using PC field name`() {
        var seen: Request? = null
        route("/api/chat") { req, reply ->
            seen = req
            reply(200, """{"content":"vision reply"}""")
        }

        val image = "data:image/png;base64,iVBORw0KGgo="
        val result = runBlocking {
            client.chat(
                ChatRequest(
                    message = "describe",
                    imageDataUrls = listOf(image),
                    allowExternal = true,
                    preferExternal = true,
                )
            )
        }

        assertTrue(result.isSuccess)
        // Gson escapes '=' as \\u003d, so assert the wire field and stable data URL prefix.
        assertTrue(seen!!.body.contains("\"image_data_urls\":[\"data:image/png;base64,"))
        assertTrue(seen!!.body.contains("iVBORw0KGgo"))
        assertTrue(seen!!.body.contains("\"allow_external\":true"))
        assertTrue(seen!!.body.contains("\"prefer_external\":true"))
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

    // ---- remote diffusion ----

    @Test
    fun `diffusion generation serializes PC request and parses job`() {
        var seen: Request? = null
        route("/api/diffusion/generate") { req, reply ->
            seen = req
            reply(202, """{"job_id":"sdjob_1","state":"queued","progress":{"step":0,"total":28}}""")
        }

        val result = runBlocking {
            client.submitDiffusionGeneration(
                DiffusionGenerateRequest(
                    presetId = "sd15_original_v1",
                    prompt = "a lighthouse",
                    negativePrompt = "blurry",
                    seed = 7,
                    steps = 28,
                )
            )
        }
        assertTrue("generation failed: ${result.exceptionOrNull()}", result.isSuccess)
        assertEquals("sdjob_1", result.getOrNull()?.jobId)
        assertEquals("queued", result.getOrNull()?.state)
        assertEquals("POST", seen?.method)
        assertTrue(seen!!.body.contains("\"preset_id\":\"sd15_original_v1\""))
        assertTrue(seen!!.body.contains("\"steps\":28"))
    }

    @Test
    fun `diffusion blob upload uses multipart purpose and file`() {
        var seen: Request? = null
        route("/api/diffusion/blobs") { req, reply ->
            seen = req
            reply(201, """{"blob_id":"img_1","content_type":"image/png","size_bytes":8,"sha256":"abc"}""")
        }

        val result = runBlocking {
            client.uploadDiffusionBlob(
                DiffusionBlobUpload(
                    data = "png-data".toByteArray(),
                    fileName = "reference.png",
                    contentType = "image/png",
                )
            )
        }
        assertTrue("upload failed: ${result.exceptionOrNull()}", result.isSuccess)
        assertEquals("img_1", result.getOrNull()?.blobId)
        assertEquals("POST", seen?.method)
        assertTrue(seen!!.body.contains("name=\"purpose\""))
        assertTrue(seen!!.body.contains("input_image"))
        assertTrue(seen!!.body.contains("filename=\"reference.png\""))
        assertTrue(seen!!.body.contains("png-data"))
    }

    @Test
    fun `diffusion edit and cancellation use job endpoints`() {
        var editBody = ""
        route("/api/diffusion/edit") { req, reply ->
            editBody = req.body
            reply(202, """{"job_id":"sdedit_1","kind":"edit","state":"queued"}""")
        }
        route("/api/diffusion/jobs/sdedit_1/cancel") { req, reply ->
            assertEquals("POST", req.method)
            reply(200, """{"accepted":true,"job":{"job_id":"sdedit_1","state":"queued","cancel_requested":true}}""")
        }

        val edit = runBlocking {
            client.submitDiffusionEdit(
                DiffusionEditRequest(
                    mode = "reference",
                    sourceBlobId = "img_1",
                    prompt = "same subject",
                    editAdapterId = "ip-adapter",
                    ipAdapterScale = 0.65f,
                )
            )
        }
        assertTrue("edit failed: ${edit.exceptionOrNull()}", edit.isSuccess)
        assertEquals("sdedit_1", edit.getOrNull()?.jobId)
        assertTrue(editBody.contains("\"source_blob_id\":\"img_1\""))
        assertTrue(editBody.contains("\"ip_adapter_scale\":0.65"))

        val cancelled = runBlocking { client.cancelDiffusionJob("sdedit_1") }
        assertTrue("cancel failed: ${cancelled.exceptionOrNull()}", cancelled.isSuccess)
        assertTrue(cancelled.getOrNull()?.accepted == true)
        assertEquals("sdedit_1", cancelled.getOrNull()?.job?.jobId)
    }

    @Test
    fun `diffusion polling stops at terminal state and rejects invalid job id`() {
        var calls = 0
        route("/api/diffusion/jobs/sdjob_poll") { _, reply ->
            calls += 1
            if (calls == 1) {
                reply(200, """{"job_id":"sdjob_poll","state":"running"}""")
            } else {
                reply(200, """{"job_id":"sdjob_poll","state":"completed","output_blob_id":"out_1"}""")
            }
        }

        val polled = runBlocking {
            client.pollDiffusionJob("sdjob_poll", intervalMillis = 0L, maxPolls = 3)
        }
        assertTrue("poll failed: ${polled.exceptionOrNull()}", polled.isSuccess)
        assertEquals("completed", polled.getOrNull()?.state)
        assertEquals(2, calls)

        val invalid = runBlocking { client.getDiffusionJob("bad/id") }
        assertTrue(invalid.isFailure)
    }

    @Test
    fun `diffusion result download reads blob bytes and rejects missing blob`() {
        route("/api/diffusion/blobs/out_1") { req, reply ->
            assertEquals("GET", req.method)
            reply(200, "png-result")
        }
        val downloaded = runBlocking { client.downloadDiffusionBlob("out_1") }
        assertTrue("download failed: ${downloaded.exceptionOrNull()}", downloaded.isSuccess)
        assertEquals("png-result", downloaded.getOrThrow().data.toString(Charsets.UTF_8))

        route("/api/diffusion/blobs/missing") { _, reply -> reply(404, "not found") }
        assertTrue(runBlocking { client.downloadDiffusionBlob("missing") }.isFailure)
    }

    @Test
    fun `gguf catalog parses verified models and drops incomplete entries`() {
        route("/api/models/gguf") { req, reply ->
            assertEquals("GET", req.method)
            reply(
                200,
                """{"models":[{"filename":"qwen.gguf","size_bytes":1024,"sha256":"${"a".repeat(64)}","download_url":"/api/models/download/qwen.gguf"},{"filename":"broken.gguf","size_bytes":0,"sha256":""}],"exists":true,"count":2}""",
            )
        }
        val models = runBlocking { client.getGgufModels() }.getOrThrow()
        assertEquals(1, models.size)
        assertEquals("qwen.gguf", models[0].filename)
        assertEquals(1024L, models[0].sizeBytes)
    }

    @Test
    fun `network unreachable returns failure`() {
        val unreachable = ApiClient(baseUrl = "http://127.0.0.1:1")
        val result = runBlocking { unreachable.chat(ChatRequest(message = "x")) }
        assertTrue(result.isFailure)
    }

    // ---- T10：上传前置校验 + poll 边界（测试修复票排期） ----

    @Test
    fun `T10 upload rejects empty data without hitting network`() {
        val result = runBlocking {
            client.uploadDiffusionBlob(
                DiffusionBlobUpload(data = ByteArray(0)),
            )
        }
        assertTrue(result.isFailure)
    }

    @Test
    fun `T10 upload rejects oversize payload`() {
        val result = runBlocking {
            client.uploadDiffusionBlob(
                DiffusionBlobUpload(data = ByteArray(DIFFUSION_MAX_UPLOAD_BYTES + 1)),
            )
        }
        assertTrue(result.isFailure)
    }

    @Test
    fun `T10 upload rejects illegal purpose`() {
        val result = runBlocking {
            client.uploadDiffusionBlob(
                DiffusionBlobUpload(
                    data = ByteArray(8),
                    purpose = "not_input_image",
                ),
            )
        }
        assertTrue(result.isFailure)
    }

    @Test
    fun `T10 poll rejects negative interval and zero polls`() {
        val negInterval = runBlocking { client.pollDiffusionJob("j", intervalMillis = -1) }
        assertTrue(negInterval.isFailure)
        assertTrue(negInterval.exceptionOrNull() is IllegalArgumentException)
        val zeroPolls = runBlocking { client.pollDiffusionJob("j", maxPolls = 0) }
        assertTrue(zeroPolls.isFailure)
        assertTrue(zeroPolls.exceptionOrNull() is IllegalArgumentException)
    }

    @Test
    fun `T10 poll timeout surfaces TimeoutException`() {
        route("/api/diffusion/jobs/j") { _, respond ->
            respond(
                200,
                """{"job_id":"j","state":"queued","progress":{"step":0,"total":1}}""",
            )
        }
        val result = runBlocking {
            client.pollDiffusionJob("j", intervalMillis = 1, maxPolls = 1)
        }
        assertTrue(result.isFailure)
        assertTrue(result.exceptionOrNull() is java.util.concurrent.TimeoutException)
    }
}
