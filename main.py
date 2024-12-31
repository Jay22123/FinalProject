from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from PIL import Image
import os

# 圖片轉換器：不含 Normalize
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()  # 範圍保持在 [0, 1]
])

# 自定義數據集
class ImageTextDataset(Dataset):
    def __init__(self, captions_file, image_dir):
        self.image_dir = image_dir
        self.data = []
        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                image_name, caption = line.strip().split(",")
                self.data.append((image_name, caption))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name, caption = self.data[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        image = transform(image)  # 轉換為張量
        return {"image": image, "text": caption}

# 模型與處理器
model_name = "OFA-Sys/chinese-clip-vit-base-patch16"
model = ChineseCLIPModel.from_pretrained(model_name)
processor = ChineseCLIPProcessor.from_pretrained(model_name)

# 數據加載
dataset = ImageTextDataset("captions.txt", "images")
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 訓練迴圈
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(3):
    for batch in dataloader:
        # 使用 processor 處理批量數據
        inputs = processor(
            text=batch["text"],
            images=batch["image"],
            return_tensors="pt",
            padding=True
        ).to(device)

        outputs = model(**inputs)
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds

        # 計算損失
        logits_per_image = torch.matmul(image_features, text_features.T)
        labels = torch.arange(len(batch["image"])).to(device)
        loss = torch.nn.CrossEntropyLoss()(logits_per_image, labels)

        # 更新權重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1} - Loss: {loss.item()}")

# 測試模型
model.eval()
text = ["蔥"]  # 測試的關鍵字
test_image_paths = ["images/蔥.jpg", "images/薑.jpg", "images/辣椒.jpg"]
test_images = [Image.open(path).convert("RGB") for path in test_image_paths]

inputs = processor(text=text, images=test_images, return_tensors="pt", padding=True).to(device)
outputs = model(**inputs)

# 提取特徵並計算相似度
image_features = outputs.image_embeds
text_features = outputs.text_embeds
similarities = torch.matmul(text_features, image_features.T)

most_similar_index = similarities.argmax().item()
print(f"與『{text[0]}』最匹配的圖片是：{test_image_paths[most_similar_index]}")