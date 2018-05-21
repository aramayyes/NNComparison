package info.ayyes.nncomparison

import android.graphics.Bitmap
import android.support.v7.app.AppCompatActivity
import android.os.Bundle
import android.util.Base64
import android.widget.TextView
import com.williamww.silkysignature.views.SignaturePad
import java.io.ByteArrayOutputStream

import io.socket.client.IO
import io.socket.client.Socket
import kotlinx.android.synthetic.main.activity_digits_painting.*
import java.net.URISyntaxException

class DigitsPaintingActivity : AppCompatActivity() {

    private val serverUrl = "http://f853f411.ngrok.io"
    private var socket: Socket? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_digits_painting)

        setUpSignaturePad()
        setUpPredictButton()
        setUpClearButton()

        openSocket()
    }

    private fun setUpSignaturePad() {
        signaturePad.setOnSignedListener(object : SignaturePad.OnSignedListener {
            override fun onStartSigning() {
                print("StartSigning")
            }

            override fun onClear() {
                print("Cleared")
            }

            override fun onSigned() {
                print("Signed")
            }
        })
    }

    private fun setUpPredictButton() {
        predictBtn.setOnClickListener {
            var img = signaturePad.signatureBitmap
            img = Bitmap.createScaledBitmap(img, 28, 28, false)

            val byteArrayOutputStream = ByteArrayOutputStream()
            img.compress(Bitmap.CompressFormat.PNG, 100, byteArrayOutputStream)

            val byteArray = byteArrayOutputStream.toByteArray()
            val encodedString = Base64.encodeToString(byteArray, Base64.DEFAULT)

            socket!!.emit("predict", encodedString)
        }
    }

    private fun setUpClearButton() {
        clearBtn.setOnClickListener {
            signaturePad.clear()
        }
    }

    private fun openSocket() {
        try {
            socket = IO.socket(serverUrl)
        } catch (e: URISyntaxException) {
            print("=========Error")
            e.printStackTrace()
            return
        }

        socket!!.on(Socket.EVENT_CONNECT) {
            socket!!.emit("accuracy")
            println("Connected")

            runOnUiThread {
                predictBtn.isEnabled = true
            }

        }.on("calculated_accuracy") { args ->

            println("Accuracy: " + args[0].toString() + "%")

            val accuracyInString = args[0].toString() + "%"
            textViewSetText(accValueTextView, accuracyInString)

        }.on("predicted") { args ->

            val answerInString = args[0] as String
            textViewSetText(ansValueTextView, answerInString)

        }.on(Socket.EVENT_DISCONNECT) { println("Disconnected") }

        socket!!.connect()
    }

    private fun textViewSetText(textView: TextView, value: String) {
        runOnUiThread { textView.text = value }
    }
}
