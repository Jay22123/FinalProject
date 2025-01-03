import pandas as pd
import torch
from PIL import Image
from Read import Read_Model, Read_Data
from Search import Search_by_Image, Search_by_Word


if __name__ == '__main__':

    # 模型與處理器
    model, processor, tokenizer  = Read_Model()


    recipes = Read_Data()
    

    # 以食材圖片找出食譜
    image_path = "images/豬肉.jpg"  # 替換為圖片路徑
    image = Image.open(image_path)
    results = Search_by_Image(recipes , image , model, processor,top_k=5)

    # 打印結果
    for result in results:
        print(result)



    #以文字敘述找出食譜
    search_query = "今天想吃一個清淡的食物，最好是有青菜"
    Search_by_Word(recipes,search_query, model, processor,tokenizer, top_k=3)

    