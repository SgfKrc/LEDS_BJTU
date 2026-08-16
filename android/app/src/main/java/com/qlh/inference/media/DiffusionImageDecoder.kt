package com.qlh.inference.media

import android.graphics.Bitmap
import android.graphics.BitmapFactory

/** Decode remote SD output into a bounded preview bitmap for Compose. */
object DiffusionImageDecoder {
    const val MAX_PREVIEW_DIMENSION = 1024

    fun decodeThumbnail(bytes: ByteArray): Bitmap? {
        if (bytes.isEmpty()) return null
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeByteArray(bytes, 0, bytes.size, bounds)
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null
        var sample = 1
        while (maxOf(bounds.outWidth, bounds.outHeight) / sample > MAX_PREVIEW_DIMENSION) {
            sample *= 2
        }
        return BitmapFactory.decodeByteArray(
            bytes,
            0,
            bytes.size,
            BitmapFactory.Options().apply { inSampleSize = sample },
        )
    }
}
