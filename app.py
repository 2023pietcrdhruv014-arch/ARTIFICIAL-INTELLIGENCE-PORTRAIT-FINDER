import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask, request, render_template, send_from_directory
from PIL import Image, ImageDraw, ImageFont

# Load model
print("Loading model...")
model_url = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
detector = hub.load(model_url).signatures['default']

# Flask setup
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads_web'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index_v2.html')

@app.route('/predict_v2', methods=['POST'])
def predict():
    file = request.files['file']
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    # Run model
    img_raw = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
    input_tensor = tf.image.convert_image_dtype(img_raw, tf.float32)[tf.newaxis, ...]
    result = detector(input_tensor)
    result = {k: v.numpy() for k, v in result.items()}

    # Draw boxes
    image_pil = Image.fromarray(np.uint8(img_raw.numpy()))
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.load_default()

    boxes = result["detection_boxes"]
    entities = result["detection_class_entities"]
    scores = result["detection_scores"]

    for i in range(min(boxes.shape[0], 15)):
        if scores[i] >= 0.1:
            ymin, xmin, ymax, xmax = boxes[i]
            w, h = image_pil.size
            left, right = xmin * w, xmax * w
            top, bottom = ymin * h, ymax * h

            draw.rectangle([left, top, right, bottom], outline="lime", width=3)
            label = f"{entities[i].decode('ascii')}: {int(scores[i]*100)}%"
            draw.text((left, top-10), label, fill="lime", font=font)

    result_path = os.path.join(UPLOAD_FOLDER, "result_" + file.filename)
    image_pil.save(result_path)

    return render_template('index_v2.html', image_path=result_path)

@app.route('/uploads_web/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
