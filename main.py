import torch
from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import requests

# 載入中文 CLIP 模型與處理器
model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

# 定義圖片清單
image_paths = ["蔥.jpg", "薑.jpg", "辣椒.jpg"]  # 替換為實際圖片路徑
images = [Image.open(path).convert("RGB") for path in image_paths]

# 定義中文關鍵字
text = [""]

# 編碼圖片與文字
inputs = processor(text=text, images=images, return_tensors="pt", padding=True)
outputs = model(**inputs)

# 提取特徵向量
image_features = outputs.image_embeds  # 圖片特徵向量
text_features = outputs.text_embeds    # 文字特徵向量

# 計算相似度
similarities = torch.matmul(text_features, image_features.T)

# 找出最高相似度的圖片
most_similar_index = similarities.argmax().item()
most_similar_image_path = image_paths[most_similar_index]

print(f"與{text}最匹配的圖片是：{most_similar_image_path}")
