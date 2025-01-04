from transformers import ChineseCLIPProcessor, ChineseCLIPModel,AutoTokenizer
import pandas as pd
import torch

def Read_Model():
    # 模型與處理器
    model_name = "OFA-Sys/chinese-clip-vit-base-patch16"
    model = ChineseCLIPModel.from_pretrained(model_name)
    processor = ChineseCLIPProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, processor,tokenizer

def Read_Data():
    # 從 CSV 文件中讀取候選名單
    csv_path = "食譜.csv"  # 替換為您的 CSV 文件路徑
    data = pd.read_csv(csv_path)

    # 將品名作為鍵，食材與步驟結合作為值
    recipes = {
        row["品名"].strip(): {"食材": row["食材"], "步驟": row["步驟"]}
        for _, row in data.iterrows()
    }

    return recipes


