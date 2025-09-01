from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
import base64
import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import webbrowser
import threading

app = Flask(__name__)

device = "cpu"  # Force CPU usage to avoid GPU memory issues
processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl").to(device)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    img_data = data.get("image")
    question = data.get("instruction", "").strip()

    if not img_data or not question:
        return jsonify({"error": "Image or question missing."}), 400

    header, encoded = img_data.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    inputs = processor(images=img, text=question, return_tensors="pt").to(device)
    output_ids = model.generate(**inputs, max_new_tokens=50)
    answer = processor.decode(output_ids[0], skip_special_tokens=True)

    return jsonify({"response": answer})

def open_browser():
    webbrowser.open_new("http://localhost:7860")

if __name__ == "__main__":
    print("Opening http://localhost:7860 in your default web browser...")
    threading.Timer(1.5, open_browser).start()
    app.run(host="localhost", port=7860)

