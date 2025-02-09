# main.py
import os 
import random
from glob import glob
import model as md
import preprocess as pre
import tensorflow as tf


def main():
    path = "./dataset/"
    # 獲取所有的wav文件，recursive 表示是否進行遞迴搜尋
    wavs = [w.replace("\\", "/") for w in glob(path + "wavs/*.wav", recursive=True)]
    
    id2text = {}
    with open(os.path.join(path, "metadata.csv"), encoding="utf-8") as f:
        for line in f:
            id = line.strip().split("|")[0]
            text = line.strip().split("|")[2]
            id2text[id] = text
            
    #print(wavs[0])
    #print(id2text["LJ001-0001"])
    
    # 資料中的所有轉錄本均小於 200 個字符
    max_target_len = 200  
    data = pre.get_data(wavs, id2text, maxlen=max_target_len)
    # 初始化 vectorizer
    vectorizer = pre.VectorizeChar(max_target_len)  
    print("Vocab size", len(vectorizer.get_vocabulary()))
    #print("\n", vectorizer.get_vocabulary())
    #print("\n", len(vectorizer('t')))
    
    split = int(len(data) * 0.99)
    train_data = data[:split]
    test_data = data[split:]
    ds = pre.create_tf_dataset(train_data, vectorizer, bs=16)
    val_ds = pre.create_tf_dataset(test_data, vectorizer, bs=4)
    print(train_data[0].keys())
    
if __name__ == "__main__":
    main()    
