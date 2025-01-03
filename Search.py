import torch
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from PIL import Image
import os
from tqdm import tqdm
import math
import numpy as np
import matplotlib.pyplot as plt

#透過一段話來找到對應的食譜
def Search_by_Word(recipes, text_query, model, processor, tokenizer, top_k=5):
    with torch.no_grad():
        tokenized_text = tokenizer(text_query, return_tensors="pt", padding=True)
        text_embedding = model.get_text_features(tokenized_text["input_ids"].to("cpu"))
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        text_embeddings = text_embedding.cpu().numpy()

     #取得品名列表
    candidate_list = list(recipes.keys())

    images_files=[]

    for name in candidate_list:
        image_file = f"{name}.jpg"
        image_path = os.path.join("./images/", image_file)
        if os.path.exists(image_path):
            images_files.append(image_path) 
        else:
            print(f"警告: 找不到 {name}.jpg 文件")
    
    image_embeddings= get_image_embedding(model, processor, images_files)

    similarities = (image_embeddings @ text_embeddings.T).squeeze(1)
    best_match_image_idx = (-similarities).argsort()
    best_image_ids, best_similarities = [images_files[i] for i in best_match_image_idx[:top_k]], similarities[best_match_image_idx]

    # 將找到的圖片繪製出來
    plt.figure(figsize = (20, 10))
    for i, path in enumerate(best_image_ids):
        plt.subplot(1, len(best_image_ids), i+1)
        plt.imshow(Image.open(path))
        plt.title(f"similarity : {best_similarities[i]}")
    plt.tight_layout()
    plt.show()


#透過食材圖片來找到食譜
def Search_by_Image(recipes, image , model,processor,top_k=5):

    #取得品名列表
    candidate_list = list(recipes.keys())

    # 跑模型
    inputs = processor(text=candidate_list, images=image,
                       return_tensors="pt", padding=True)
    outputs = model(**inputs)

    # 計算相似度
    logits_per_image = outputs.logits_per_image  # 圖片與文字的相似度分數
    probs = logits_per_image.softmax(dim=1)  # 將分數轉為機率

    #找出前 k 名
    top_probs, top_indices = torch.topk(probs, top_k, dim=1)

    results = []
    # 輸出前五名結果
   
    for i in range(top_k):
        label = candidate_list[top_indices[0, i]]  # 獲取對應的品名
        probability = top_probs[0, i].item()  # 獲取對應的機率
        # print(f"{i + 1}. {label} (機率: {probability:.4f})")
        # if label  in recipes:
        #     print(f"食材: {recipes[label]['食材']}")
        #     print(f"步驟: {recipes[label]['步驟']}")
        prediction = {
            "品名": label,
            "機率": round(probability, 4),
            "食材": recipes[label]["食材"],
            "步驟": recipes[label]["步驟"]
        }
        # 添加到結果列表
        results.append(prediction)

    return results


def get_image_embedding(model, processor, images_files):
   
    batch_size = 32
    image_embeddings = []

 
    for i in tqdm(range(math.ceil(len(images_files) / batch_size)), desc="Processing Images"):
        batch_files = images_files[batch_size * i : batch_size * (i + 1)]


       # 加载图片
        batch_images = []
        for path in batch_files:
            if path is not None:
                try:
                    batch_images.append(Image.open(path))
                except Exception as e:
                    print(f"警告: 无法加载图片 {path}, 错误: {e}")

        if not batch_images:
            continue  # 如果没有有效图片，跳过

        # 使用 processor 进行批量预处理
        image_preproc = processor(images=batch_images, return_tensors="pt").to("cpu")

        with torch.no_grad():
       
            batch_embeddings = model.get_image_features(image_preproc.pixel_values)
            batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)
            batch_embeddings = batch_embeddings.cpu().numpy()

        image_embeddings.append(batch_embeddings)

    return np.vstack(image_embeddings)

