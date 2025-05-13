from flask import Flask, render_template, request, jsonify
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)

# Load your trained model
model = load_model("m32.h5")

# Your class labels
class_labels = ['Cell', 'Cell-Multi', 'Cracking', 'Diode', 'Diode-Multi',
                'Hot-Spot', 'Hot-Spot-Multi', 'No-Anomaly', 'Offline-Module',
                'Shadowing', 'Soiling', 'Vegetation']

def model_predict(img_path, model):
    # Load and preprocess image
    img = Image.open(img_path).convert("RGB")
    img = img.resize((24, 40))  # width, height (24, 40) as per your model
    img_array = np.array(img)

    # Reshape to match model input
    img_input = img_array.reshape(1, 40, 24, 3)
    
    # Predict
    result = model.predict(img_input)
    return result

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')  # Ensure index.html exists

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(os.path.realpath('__file__'))
        upload_path = os.path.join(basepath, 'uploads')
        os.makedirs(upload_path, exist_ok=True)
        file_path = os.path.join(upload_path, secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        result = model_predict(file_path, model)
        max_prob = np.max(result[0])
        class_ind = np.argmax(result[0])
        predicted_class = class_labels[class_ind]

        # Return the result in a safe way
        response = {
            "class": str(predicted_class),  # Ensure it's a string to avoid Unicode issues
            "probability": round(float(max_prob), 4)
        }

        return jsonify(response)

    return jsonify({"message": "Send a POST request with an image."})

if __name__ == '__main__':
    app.run(debug=False, port=5926)
