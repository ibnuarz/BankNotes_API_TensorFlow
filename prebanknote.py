import io
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)
CORS(app, supports_credentials=True)

model_banknote_path = "model2.h5"
model_banknote = tf.keras.models.load_model(model_banknote_path)

nominal_labels = ['Seribu', 'Sepuluh Ribu', 'Seratus Ribu', 'Dua Ribu', 'Dua Puluh Ribu', 'Lima Ribu', 'Lima Puluh Ribu']

def transform_image(pillow_image):
    if pillow_image.mode == 'RGBA':
        pillow_image = pillow_image.convert('RGB')
    pillow_image = pillow_image.resize((150, 150))
    data = np.asarray(pillow_image)
    data = data.astype(np.float32)  # Convert to FLOAT32
    data /= 255.0  # Normalize to the range [0, 1]
    data = np.expand_dims(data, axis=0)
    return data

def predict_nominal(x):
    predictions = model_banknote.predict(x)
    predicted_class = np.argmax(predictions)
    predicted_nominal_label = nominal_labels[predicted_class]
    max_prob = np.max(predictions)
    return predicted_class, predicted_nominal_label, max_prob

@app.route("/", methods=["GET","POST"])
def ismprebanknote():
    try:
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({"error": "no file"})

        # Read and process the image using PIL.Image
        pillow_img = Image.open(io.BytesIO(file.read()))
        pillow_img = pillow_img.resize((224, 224))
        img_array = transform_image(pillow_img)

        # Perform nominal prediction for banknote
        predicted_nominal_class, predicted_nominal_label, max_nominal_prob = predict_nominal(img_array)

        # Create a dictionary for the output
        output_dict = {
            "feature_nominal": {
                "fitur": "Prediksi Nominal Banknote",
                "gambar": file.filename,
                "prediksi_kelas": int(predicted_nominal_class),
                "prediksi_nominal": predicted_nominal_label,
                "prediksi_akurasi": f"{round(max_nominal_prob * 100, 2)}%"
            },
        }

        return jsonify(output_dict)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
