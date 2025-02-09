# main.py
import os 
import random
from glob import glob
import model as md
import preprocess as pre
import tensorflow as tf


def create_text_ds(data):
    """
    創建文本數據集
    """
    texts = ["text" for _ in data]
    text_ds = [vectorizer(t) for t in texts]
    # tf.data.Dataset.from_tensor_slices 創建數據集
    text_ds = tf.data.Dataset.from_tensor_slices(text_ds)
    return text_ds

def path_to_audio(path):
    """
    將音頻文件轉換為 stft 頻譜圖\n
    參數：\n
    path: 音頻文件路徑
    """
    # tf.io.read_file 讀取音頻文件
    audio = tf.io.read_file(path)
    # tf.audio.decode_wav 解碼音頻文件，1 表示只讀取單聲道
    audio, _ = tf.audio.decode_wav(audio, 1)
    # tf.squeeze 用來移除指定維度大小為 1 的維度，axis=-1 代表對最後一個維度進行操作
    audio = tf.squeeze(audio, axis=-1)
    # tf.signal.stft 使用短時傅立葉變換將音頻轉換為頻譜圖
    # frame_length 表示每一幀的長度，frame_step 表示每一幀之間的間隔，fft_length 表示傅立葉變換的點數
    stfts = tf.signal.stft(audio, frame_length=200, frame_step=80, fft_length=256)
    # tf.math.pow 對 stfts 的絕對值進行 0.5 次方運算
    x = tf.math.pow(tf.abs(stfts), 0.5)

    # tf.math.reduce_mean 求平均值，axis=1 表示對第二個維度進行操作，keepdims=True 表示保持維度
    means = tf.math.reduce_mean(x, 1, keepdims=True)
    # tf.math.reduce_std 求標準差
    stddevs = tf.math.reduce_std(x, 1, keepdims=True)
    # 標準化
    x = (x - means) / stddevs

    audio_len = tf.shape(x)[0]
    # 填充至 10 秒
    pad_len = 2754
    # tf.constant 是用來創建一個常數張量的函數
    # 第一行 [0, pad_len] 表示對第一個維度（時間或長度維度）進行填充，填充的方式是 在起始位置不進行填充（0），在結尾位置填充 pad_len 長度的空間
    # 第二行 [0, 0] 表示對第二個維度（頻率維度）不進行填充
    paddings = tf.constant([[0, pad_len], [0, 0]])
    # tf.pad 用來將張量 x 進行填充操作
    # "CONSTANT" 表示填充的方式是用常數填充
    x = tf.pad(x, paddings, "CONSTANT")[:pad_len, :]

    return x

def create_audio_ds(data):
    """
    創建音頻數據集\n
    """
    flist = ["audio" for _ in data]
    # tf.data.Dataset.from_tensor_slices 創建數據集
    audio_ds = tf.data.Dataset.from_tensor_slices(flist)
    # 使用 map 函數對數據集中的每一個元素進行 path_to_audio 函數的操作
    audio_ds = audio_ds.map(
        path_to_audio, num_parallel_calls=tf.data.AUTOTUNE # 使用多線程加速
    )
    return audio_ds

def create_tf_dataset(data, bs=4):
    """
    創建數據集\n
    參數：\n
    data: 數據\n
    bs: 批次大小
    """
    audio_ds = create_audio_ds(data)
    text_ds = create_text_ds(data)
    # tf.data.Dataset.zip 將兩個數據集合併
    ds = tf.data.Dataset.zip((audio_ds, text_ds))
    # source: 音頻，target: 文本
    ds = ds.map(lambda x, y: {"source": x, "target": y})
    # batch 是將數據集分成多個批次，每個批次的大小是 bs
    ds = ds.batch(bs)
    # prefetch 是用來加速數據集的方法，tf.data.AUTOTUNE 表示自動選擇最佳的參數
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

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
    vectorizer = pre.VectorizeChar(max_target_len)
    print("Vocab size", len(vectorizer.get_vocabulary()))
    #print("\n", vectorizer.get_vocabulary())
    print("\n", len(vectorizer('t')))
    
if __name__ == "__main__":
    main()    
