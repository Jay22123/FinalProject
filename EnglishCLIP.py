import pandas as pd
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

if __name__ == '__main__':
    # 載入模型
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # 從 CSV 文件中讀取候選名單
    csv_path = "食譜.csv"  # 替換為您的 CSV 文件路徑
    data = pd.read_csv(csv_path)
    class_list = data['品名'].dropna().tolist()  # 確保去除空值

    # 載入圖片
    image_path = "images/雞肉.jpg"  # 替換為圖片路徑
    image = Image.open(image_path)

    # 跑模型
    inputs = processor(text=class_list, images=image,
                       return_tensors="pt", padding=True)
    outputs = model(**inputs)

    # 計算相似度
    logits_per_image = outputs.logits_per_image  # 圖片與文字的相似度分數
    probs = logits_per_image.softmax(dim=1)  # 將分數轉為機率

   # 找出前五名
    top_k = 5  # 前五名
    top_probs, top_indices = torch.topk(probs, top_k, dim=1)

    # 輸出前五名結果
    print("候選名單  : ", class_list)
    print("前五名預測 :")
    for i in range(top_k):
        label = class_list[top_indices[0, i]]  # 獲取對應的標籤
        probability = top_probs[0, i].item()  # 獲取對應的機率
        print(f"{i + 1}. {label} (機率: {probability:.4f})")
    # 預測
    # max_index = torch.argmax(probs)
    # print("候選名單  : ", class_list)
    # print("預測機率分布 : ", probs.detach())
    # print("預測結果 : " + class_list[max_index])
