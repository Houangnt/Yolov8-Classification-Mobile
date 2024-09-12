package com.example.yolov8n_cls
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.view.WindowManager
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import java.io.File
import java.io.InputStream
import java.util.Collections // <-- Add this import

class MainActivity : AppCompatActivity() {

    private lateinit var imageView: ImageView
    private lateinit var textView: TextView
    private lateinit var selectButton: Button
    private lateinit var predictButton: Button
    private lateinit var dataProcess: DataProcess
    private lateinit var ortEnvironment: OrtEnvironment
    private lateinit var session: OrtSession
    private var selectedImageUri: Uri? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.imageView)
        textView = findViewById(R.id.textView)
        selectButton = findViewById(R.id.selectButton)
        predictButton = findViewById(R.id.predictButton)

        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        dataProcess = DataProcess()
        loadModel()

        selectButton.setOnClickListener {
            selectImage()
        }

        predictButton.setOnClickListener {
            selectedImageUri?.let { uri ->
                predictImage(uri)
            } ?: Toast.makeText(this, "No image selected!", Toast.LENGTH_SHORT).show()
        }
    }

    private fun loadModel() {
        dataProcess.loadModel(this)
        dataProcess.loadLabel(this)

        ortEnvironment = OrtEnvironment.getEnvironment()
        session = ortEnvironment.createSession(
            this.filesDir.absolutePath.toString() + "/" + DataProcess.FILE_NAME,
            OrtSession.SessionOptions()
        )
    }

    private fun selectImage() {
        val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
        imagePickerLauncher.launch(intent)
    }

    private val imagePickerLauncher =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode == RESULT_OK) {
                val uri = result.data?.data
                uri?.let {
                    selectedImageUri = it
                    val inputStream: InputStream? = contentResolver.openInputStream(it)
                    val bitmap = BitmapFactory.decodeStream(inputStream)
                    imageView.setImageBitmap(bitmap)
                }
            }
        }

    private fun predictImage(uri: Uri) {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val inputStream = contentResolver.openInputStream(uri)
                val bitmap = BitmapFactory.decodeStream(inputStream)

                val scaledBitmap = Bitmap.createScaledBitmap(
                    bitmap,
                    DataProcess.INPUT_SIZE,
                    DataProcess.INPUT_SIZE,
                    true
                )
                val floatBuffer = dataProcess.bitmapToFloatBuffer(scaledBitmap)

                val inputName = session.inputNames.iterator().next()
                val shape = longArrayOf(
                    DataProcess.BATCH_SIZE.toLong(),
                    DataProcess.PIXEL_SIZE.toLong(),
                    DataProcess.INPUT_SIZE.toLong(),
                    DataProcess.INPUT_SIZE.toLong()
                )
                val inputTensor = OnnxTensor.createTensor(ortEnvironment, floatBuffer, shape)
                val resultTensor = session.run(Collections.singletonMap(inputName, inputTensor))

                // Handle 2D array output
                val output = resultTensor[0].value as Array<FloatArray>
                val flattenedOutput =
                    output.flatMap { it.asIterable() }.toFloatArray() // Convert to 1D array

                // Log output for debugging
                println("Raw Output Values: ${flattenedOutput.joinToString()}")

                // Create a formatted string with class names and confidence values
                val outputText = StringBuilder()
                flattenedOutput.forEachIndexed { index, value ->
                    val className = dataProcess.getClassName(index)
                    if (className != null) {
                        outputText.append("${className}: ${String.format("%.8f", value)}\n")
                    }
                }

                withContext(Dispatchers.Main) {
                    textView.text = outputText.toString()
                }
            } catch (e: Exception) {
                e.printStackTrace()
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, "Error: ${e.message}", Toast.LENGTH_SHORT)
                        .show()
                }
            }
        }
    }
}