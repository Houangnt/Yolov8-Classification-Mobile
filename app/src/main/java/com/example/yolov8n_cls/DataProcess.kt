package com.example.yolov8n_cls

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.camera.core.ImageProxy
import java.io.BufferedReader
import java.io.File
import java.io.FileOutputStream
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

class DataProcess {

    private lateinit var classes: Array<String>

    companion object {
        const val BATCH_SIZE = 1
        const val INPUT_SIZE = 224
        const val PIXEL_SIZE = 3
        const val FILE_NAME = "yolov8n.onnx"
        const val LABEL_NAME = "yolov8n.txt"
    }

    fun getClassName(i: Int?): String? {
        return if (i != null) {
            classes[i]
        } else null
    }

    fun getHighConf(output: FloatArray): Int? {
        val confThreshold = 0.6f
        return output.withIndex().filter { it.value >= confThreshold }
            .maxByOrNull { it.value }?.index
    }



    fun bitmapToFloatBuffer(bitmap: Bitmap): FloatBuffer {
        val imageSTD = 255f

        val cap = BATCH_SIZE * PIXEL_SIZE * INPUT_SIZE * INPUT_SIZE
        val order = ByteOrder.nativeOrder()
        val buffer = ByteBuffer.allocateDirect(cap * Float.SIZE_BYTES).order(order).asFloatBuffer()

        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)

        val bitmapData = IntArray(INPUT_SIZE * INPUT_SIZE)
        scaledBitmap.getPixels(bitmapData, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        for (i in 0 until INPUT_SIZE) {
            for (j in 0 until INPUT_SIZE) {
                val idx = INPUT_SIZE * i + j
                val pixelValue = bitmapData[idx]
                buffer.put(idx, ((pixelValue shr 16 and 0xff) / imageSTD)) // Red
                buffer.put(idx + INPUT_SIZE * INPUT_SIZE, ((pixelValue shr 8 and 0xff) / imageSTD)) // Green
                buffer.put(idx + INPUT_SIZE * INPUT_SIZE * 2, ((pixelValue and 0xff) / imageSTD)) // Blue
            }
        }
        buffer.rewind()
        return buffer
    }


    fun loadLabel(context: Context) {
        BufferedReader(InputStreamReader(context.assets.open(LABEL_NAME))).use { reader ->
            var line: String?
            val classList = ArrayList<String>()
            while (reader.readLine().also { line = it } != null) {
                classList.add(line!!)
            }
            classes = classList.toTypedArray()
        }
    }

    fun loadModel(context: Context) {
        val assetManager = context.assets
        val outputFile = File(context.filesDir.toString() + "/" + FILE_NAME)

        assetManager.open(FILE_NAME).use { inputStream ->
            FileOutputStream(outputFile).use { outputStream ->
                val buffer = ByteArray(1024)
                var read: Int
                while (inputStream.read(buffer).also { read = it } != -1) {
                    outputStream.write(buffer, 0, read)
                }
            }
        }
    }
}
