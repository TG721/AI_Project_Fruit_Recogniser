package com.tengizmkcorp.aiproject_fruitrecogniser

import android.annotation.SuppressLint
import android.app.Activity
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Toast
import androidx.activity.result.ActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.github.dhaval2404.imagepicker.ImagePicker
import com.tengizmkcorp.aiproject_fruitrecogniser.databinding.ActivityMainBinding
import com.tengizmkcorp.aiproject_fruitrecogniser.ml.Model
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : AppCompatActivity() {
    private var imageURI: Uri? = null
    private val imageSize = 32
    private lateinit var binding: ActivityMainBinding
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        listeners()
    }

    private fun listeners() {
        binding.apply {
            val selectButton = buttonSelectPicture
            val resultTV = textView
            val image = imageView
            selectButton.setOnClickListener {
                ImagePicker.with(this@MainActivity)
                    .crop(150f,150f)
                    .createIntent { intent ->
                        startForProfileImageResult.launch(intent)
                    }
            }
        }
    }
    private val startForProfileImageResult =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result: ActivityResult ->
            val resultCode = result.resultCode
            val data = result.data

            when (resultCode) {
                Activity.RESULT_OK -> {
                    //Image Uri will not be null for RESULT_OK
                    imageURI = data?.data


                   val bitmapImage = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                       ImageDecoder.decodeBitmap(ImageDecoder.createSource(this@MainActivity.contentResolver,
                           imageURI!!))
                   } else {
                       MediaStore.Images.Media.getBitmap(this@MainActivity.contentResolver, imageURI)
                   }
                    val scaledBitmapImage = Bitmap.createScaledBitmap(bitmapImage, imageSize, imageSize, false)
                    binding.imageView.setImageBitmap(bitmapImage)
                    classifyImage(scaledBitmapImage)
                }
                ImagePicker.RESULT_ERROR -> {
                    Toast.makeText(this, ImagePicker.getError(data), Toast.LENGTH_SHORT).show()
                }
                else -> {
                    Toast.makeText(this, getString(R.string.task_cancelled), Toast.LENGTH_SHORT).show()
                }
            }
        }

    @SuppressLint("SetTextI18n")
    private fun classifyImage(bitmapImage: Bitmap) {
        val model = Model.newInstance(this@MainActivity)

        // Create inputs for reference.
        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 32, 32, 3), DataType.FLOAT32)
        val byteBuffer = ByteBuffer.allocateDirect(4*imageSize*imageSize*3)
        byteBuffer.order(ByteOrder.nativeOrder())
        val intValues = IntArray(imageSize * imageSize)
        val bmpCopy: Bitmap = bitmapImage.copy(Bitmap.Config.ARGB_8888, true)
        bmpCopy.getPixels(intValues, 0, bmpCopy.width, 0, 0, bmpCopy.width, bmpCopy.height)
        var pixel = 0
        //iterate over each pixel and extract R, G, and B values. Add the values individually to the byte buffer.
        for (i in 0 until imageSize) {
            for (j in 0 until imageSize) {
                val `val` = intValues[pixel++] // RGB
                byteBuffer.putFloat((`val` shr 16 and 0xFF) * (1f / 1))
                byteBuffer.putFloat((`val` shr 8 and 0xFF) * (1f / 1))
                byteBuffer.putFloat((`val` and 0xFF) * (1f / 1))
            }
        }
        inputFeature0.loadBuffer(byteBuffer)

        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer
        val confidences = outputFeature0.floatArray
        // find the index of the class with the biggest confidence.
        var maxPos = 0
        var maxConfidence = 0f
        for (i in confidences.indices) {
            if (confidences[i] > maxConfidence) {
                maxConfidence = confidences[i]
                maxPos = i
            }
        }
        val classes = arrayOf(getString(R.string.apple), getString(R.string.banana), getString(R.string.orange))
        binding.textView.text=getString(R.string.classified_as) + " " + classes[maxPos]
        // Release model resources if no longer used.
        model.close()
    }
}