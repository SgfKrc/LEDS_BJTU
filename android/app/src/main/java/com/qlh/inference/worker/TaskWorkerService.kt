package com.qlh.inference.worker

import android.app.Notification
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Binder
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import com.qlh.inference.BuildConfig
import com.qlh.inference.MainActivity
import com.qlh.inference.QlhApplication
import com.qlh.inference.R

/** Foreground lifecycle shell for the Android Full Worker client. */
class TaskWorkerService : Service() {
    private val binder = LocalBinder()
    private var client: TaskWorkerClient? = null

    inner class LocalBinder : Binder() {
        fun getService(): TaskWorkerService = this@TaskWorkerService
        fun snapshot(): TaskWorkerSnapshot = this@TaskWorkerService.snapshot()
    }

    override fun onCreate() {
        super.onCreate()
        startForegroundNotification()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START -> startWorker(intent)
            ACTION_STOP -> {
                stopWorker()
                stopSelf(startId)
            }
            ACTION_CANCEL -> client?.cancelActive(intent.getStringExtra(EXTRA_REASON) ?: "user_cancelled")
        }
        return if (client != null) START_STICKY else START_NOT_STICKY
    }

    override fun onBind(intent: Intent?): IBinder = binder

    override fun onDestroy() {
        stopWorker()
        super.onDestroy()
    }

    fun snapshot(): TaskWorkerSnapshot = client?.snapshot?.value ?: TaskWorkerSnapshot()

    fun cancelActive(reasonCode: String = "user_cancelled"): Boolean = client?.cancelActive(reasonCode) == true

    private fun startWorker(intent: Intent) {
        if (BuildConfig.IS_LITE) {
            stopSelf()
            return
        }
        val host = intent.getStringExtra(EXTRA_COORDINATOR_HOST).orEmpty().trim()
        val port = intent.getIntExtra(EXTRA_COORDINATOR_PORT, 0)
        val nodeId = intent.getStringExtra(EXTRA_NODE_ID).orEmpty().trim()
        if (host.isEmpty() || port !in 1..65535 || nodeId.isEmpty()) {
            stopSelf()
            return
        }
        stopWorker()
        client = TaskWorkerClient(
            host = host,
            port = port,
            nodeId = nodeId,
            capabilities = {
                mapOf(
                    "stage_types" to listOf("full_inference"),
                    "engines" to listOf("llama_cpp"),
                    "models" to emptyList<Map<String, Any?>>(),
                    "max_concurrency" to 1,
                )
            },
        ).also { it.start() }
    }

    private fun stopWorker() {
        val old = client ?: return
        client = null
        old.stop()
    }

    private fun startForegroundNotification() {
        val pendingIntent = PendingIntent.getActivity(
            this,
            1,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val notification: Notification = NotificationCompat.Builder(
            this,
            QlhApplication.NOTIFICATION_CHANNEL_INFERENCE,
        )
            .setContentTitle("QLH Worker")
            .setContentText("Android Worker client is running")
            .setSmallIcon(android.R.drawable.ic_media_play)
            .setOngoing(true)
            .setContentIntent(pendingIntent)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startForeground(
                QlhApplication.NOTIFICATION_ID_WORKER,
                notification,
                android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE,
            )
        } else {
            startForeground(QlhApplication.NOTIFICATION_ID_WORKER, notification)
        }
    }

    companion object {
        const val ACTION_START = "com.qlh.inference.worker.START"
        const val ACTION_STOP = "com.qlh.inference.worker.STOP"
        const val ACTION_CANCEL = "com.qlh.inference.worker.CANCEL"
        const val EXTRA_COORDINATOR_HOST = "coordinator_host"
        const val EXTRA_COORDINATOR_PORT = "coordinator_port"
        const val EXTRA_NODE_ID = "node_id"
        const val EXTRA_REASON = "reason"

        fun startIntent(context: Context, host: String, port: Int, nodeId: String): Intent = Intent(
            context,
            TaskWorkerService::class.java,
        ).setAction(ACTION_START)
            .putExtra(EXTRA_COORDINATOR_HOST, host)
            .putExtra(EXTRA_COORDINATOR_PORT, port)
            .putExtra(EXTRA_NODE_ID, nodeId)

        fun stopIntent(context: Context): Intent = Intent(context, TaskWorkerService::class.java)
            .setAction(ACTION_STOP)

        fun cancelIntent(context: Context, reasonCode: String = "user_cancelled"): Intent = Intent(
            context,
            TaskWorkerService::class.java,
        ).setAction(ACTION_CANCEL).putExtra(EXTRA_REASON, reasonCode)
    }
}
