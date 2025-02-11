# speech2txt.py
import os 
import random
from glob import glob
import model as md
import preprocess as pre
import tensorflow as tf
from tensorflow.keras import mixed_precision

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
    #print(train_data[0].keys())
    #print(train_data[0])
    #print(type(ds))
    
    batch = next(iter(ds))
    id2text  = vectorizer.get_vocabulary()
    display_cb = md.DisplayOutputs(batch, id2text, target_start_token_idx=2, target_end_token_idx=3)    
    
    model = md.Transformer(
        num_hid=200, 
        num_head=2, 
        num_feed_forward=400, 
        target_maxlen=max_target_len, 
        num_layers_enc=4, 
        num_layers_dec=1,
        num_classes=34
    )
   
    loss_fn = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True,
        label_smoothing=0.1,
    )
    
    checkpoint_path = "./checkpoints/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        save_freq=1000
    )
    
    learning_rate = md.CustomSchedule(
        init_lr=0.00001,
        lr_after_warmup=0.001,
        final_lr=0.00001,
        warmup_epochs=15,
        decay_epochs=40,
        steps_per_epoch=len(ds)
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    
    model.compile(optimizer=optimizer, loss=loss_fn)
    
    model.save_weights(checkpoint_path.format(epoch=10))
    
    # 設定 Mixed Precision 計算
    mixed_precision.set_global_policy("mixed_float16")
    
    model.fit(ds, validation_data=val_ds, callbacks=[display_cb], epochs=100)
    
    print("\n", model.val_loss.numpy())
    
    idx_to_char = vectorizer.get_vocabulary()
    for i in range(10,15):
        preds = model.generate(tf.expand_dims(pre.path_to_audio(test_data[i]['audio']), axis=0), target_start_token_idx=2)
        prediction = ""

        preds = preds.numpy()

        prediction = ""
        for idx in preds[0]:
            prediction += idx_to_char[idx]
            if idx == 3:
                break

        print(f"\nactual = {test_data[i]['text']}")
        print(f"\npredicted = {prediction}")
        print("\n-----" * 10)

    ROOT = os.getcwd()
    # 模型權重路徑
    weights_path = os.path.join(ROOT, '/checkpoints/')

    # os.makedirs(weights_path, exist_ok=True)
    # model.save_weights(weights_path)

    print(os.listdir(weights_path))

    model2 = md.Transformer(
        num_hid=200,
        num_head=2,
        num_feed_forward=400,
        target_maxlen=max_target_len,
        num_layers_enc=4,
        num_layers_dec=1,
        num_classes=34,
    )
    model2.load_weights(weights_path)
    print(model2.val_loss.numpy())
    
    idx_to_char = vectorizer.get_vocabulary()
    for i in range(5):
        preds = model2.generate(tf.expand_dims(pre.path_to_audio(test_data[i]['audio']), axis=0), target_start_token_idx=2)

        preds = preds.numpy()

        prediction = ""
        for idx in preds[0]:
            prediction += idx_to_char[idx]
            if idx_to_char[idx] == ">":
                break

        print(f"actual = {test_data[i]['text']}")
        print(f"predicted = {prediction}")
        print("-----" * 50)

if __name__ == "__main__":
    main()    
