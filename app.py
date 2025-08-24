from flask import Flask, render_template, request
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXT = {"png", "jpg", "jpeg", "gif"}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

# Load model (ensure path is correct)
model = load_model("plant_disease_model.h5")

# Class labels (adjust to your model)
class_names = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites_Two_spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___Tomato_YellowLeaf__Curl_Virus',
    'Tomato___healthy'
]

# Professional, non-prescriptive suggestions (placeholders)
treatment_suggestions = {
    'Pepper__bell___Bacterial_spot': "Recommend copper-based sprays and sanitation measures. Consult an agronomist for exact dosage and timing.",
    'Pepper__bell___healthy': "No treatment required. Continue good cultural practices and monitoring.",
    'Potato___Early_blight': "Consider contact fungicides (e.g., chlorothalonil) as part of an integrated program; verify local label recommendations.",
    'Potato___Late_blight': "Consider systemic fungicides (e.g., metalaxyl) and remove infected material; consult local extension services.",
    'Potato___healthy': "No treatment required. Maintain crop rotation and monitoring.",
    'Tomato___Bacterial_spot': "Implement copper-containing sprays and remove symptomatic foliage; seek expert advice for product choice.",
    'Tomato___Early_blight': "Use recommended fungicides and cultural controls; follow local guidelines for application.",
    'Tomato___Late_blight': "Emergency treatment may be required; consult local plant health authorities for active ingredients and timing.",
    'Tomato___Leaf_Mold': "Improve ventilation, reduce leaf wetness, and consider registered fungicides; consult a crop specialist.",
    'Tomato___Septoria_leaf_spot': "Remove affected leaves and consider fungicide rotations; follow label instructions.",
    'Tomato___Spider_mites_Two_spotted_spider_mite': "Consider miticides or horticultural oil, and encourage natural predators; consult integrated pest management resources.",
    'Tomato___Target_Spot': "Use copper-based or appropriate fungicides; maintain field hygiene and consult extension resources.",
    'Tomato___Tomato_mosaic_virus': "No chemical cure; remove infected plants and disinfect equipment. Seek pest management guidance.",
    'Tomato___Tomato_YellowLeaf__Curl_Virus': "Control vector populations (e.g., whiteflies) and remove infected plants; consult IPM specialists.",
    'Tomato___healthy': "No treatment required. Continue monitoring and good cultural practices."
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("index.html", error="No file part in request.")
    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", error="No file selected.")
    if not allowed_file(file.filename):
        return render_template("index.html", error="Unsupported file format.")
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Preprocess image for model
    img = image.load_img(filepath, target_size=(224, 224))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    # Predict
    preds = model.predict(x)
    pred_idx = int(np.argmax(preds, axis=1)[0])
    predicted_label = class_names[pred_idx]
    confidence = float(np.max(preds)) * 100.0
    confidence = round(confidence, 2)

    # Fetch a responsible suggestion (non-prescriptive)
    suggestion = treatment_suggestions.get(predicted_label, "No suggestion available. Consult a specialist.")

    # Professional phrasing string (you requested)
    professional_sentence = "I aim to provide accurate and appropriate medication or treatment recommendations for diagnosed conditions."

    # Pass everything to template (use relative URL for image)
    image_url = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    return render_template(
        "result.html",
        prediction=predicted_label,
        confidence=confidence,
        suggestion=suggestion,
        image_url=image_url,
        professional_sentence=professional_sentence
    )

if __name__ == "__main__":
    app.run(debug=True)
