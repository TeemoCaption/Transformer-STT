# main.py
import os 
import random
from glob import glob
import model as md
import preprocess as pre


def main():
    path = "./dataset/"
    # 獲取所有的wav文件，recursive 表示是否進行遞迴搜尋
    wavs = glob(path + "wavs/*.wav", recursive=True)
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
    vectorizer = pre.VectorizeChar(max_target_len)
    print("Vocab size", len(vectorizer.get_vocabulary()))
    print("WAV files:", wavs)
    print("\n", data[0])
    
if __name__ == "__main__":
    main()    
