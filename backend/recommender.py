import pickle
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from io import BytesIO
import base64

app = Flask(__name__)

# Загружаем сохранённые данные
with open("features_clip.pkl", "rb") as f:
    features = pickle.load(f)
with open("similarity_matrix_clip.pkl", "rb") as f:
    similarity_matrix = pickle.load(f)

# Загружаем CLIP модель
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_image_features(img_data):
    inputs = processor(images=img_data, return_tensors="pt")
    image_features = model.get_image_features(**inputs)
    return image_features.detach().numpy()

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    img_b64 = data["image"]
    img_bytes = base64.b64decode(img_b64.split(",")[1])
    img = Image.open(BytesIO(img_bytes)).convert("RGB")

    # Получаем фичи изображения
    query_features = get_image_features(img).flatten()

    # Считаем похожесть
    similarities = np.dot(features, query_features) / (
        np.linalg.norm(features, axis=1) * np.linalg.norm(query_features)
    )

    # Берём индексы самых похожих
    top_indices = np.argsort(similarities)[::-1][:5]
    result = {"recommendations": top_indices.tolist()}
    return jsonify(result)