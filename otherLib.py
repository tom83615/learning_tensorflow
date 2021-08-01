import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 限制 GPU memory (1G)
def solve_cudnn_error():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
    return

# 顯示影像 function
# images: 圖集; labels: 數字標籤; prediction :預測結果;
# idx: 起始點; num: 顯示筆數;
# dict: label 的額外註記
def plot_prediction(images, labels, prediction, idx=0 ,num=25, dict={}):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    row = int(math.ceil(math.pow(num, 0.5))) #高
    col = int(math.pow(num, 0.5))            #寬
    for i in range(0, num):
        ax = plt.subplot(row, col, 1+i) # 
        ax.imshow(images[idx], cmap='binary')
        ## 標題
        title = "r:" + print_labels(labels[idx], dict)
        ## 預測值
        if len(prediction) > 0:
            title += ",p:" + print_labels(prediction[idx], dict)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx+=1
    plt.show()
    return

# 遞迴處理文字標籤 function
# labels: 數字標籤;
# dict: label 的額外註記
def print_labels(labels, dict={}):
    re = ""
    if(isinstance(labels, list) or isinstance(labels,np.ndarray)):
        re += "["
        for i in range(0, len(labels)):
            re += print_labels(labels[i], dict) + ","
        re =  re[:-1] + "]"
    else:
        re = str(labels)
        if labels in dict.keys(): 
            re += "("+ dict[labels] +")"
    return re

# 顯示模式預測後結果
# train_history: 訓練紀錄;
def show_train_history(train_history):
    plt.plot(train_history.history['accuracy'])
    plt.plot(train_history.history['val_accuracy'])
    plt.title('Train History')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

# 顯示圖片以及機率
# y: 真實值; prediction: 預測值;
# x_img: 圖像;
# predicted_probability: 該組預測機率;
# i: 目標數據點;
# dict: label 的額外註記
def show_Predicted_Probability(y, prediction, x_img, predicted_probability, i, dict={}):
    if y[i][0] in dict.keys(): 
        print('label:', dict[y[i][0]])
    if prediction[i] in dict.keys(): 
        print('prediction:', dict[prediction[i]])
    
    plt.figure(figsize=(2,2))
    plt.imshow(np.reshape(x_img[i], x_img[i].shape))
    plt.show()
    for j in range(len(dict)):
        if j in dict.keys(): 
            print(dict[j] + ' Probability:%1.9f'%(predicted_probability[i][j]))
        else:
            print(j + ' Probability:%1.9f'%(predicted_probability[i][j]))
    
# tensorflow 建立神經網路
def tf_layer(output_dim, input_dim, inputs, activation=None):
    W = tf.Variable(tf.random.normal([input_dim, output_dim]))
    b = tf.Variable(tf.random.normal([1, output_dim]))
    XWb = tf.matmul(inputs, W) + b
    if activation is None:
        outputs = XWb
    else:
        outputs = activation(XWb)
    return outputs

# tensorflow 建立神經網路 (多帶 weight 與 bias)
def tf_layer_debug(output_dim, input_dim, inputs, activation=None):
    W = tf.Variable(tf.random.normal([input_dim, output_dim]))
    b = tf.Variable(tf.random.normal([1, output_dim]))
    XWb = tf.matmul(inputs, W) + b
    if activation is None:
        outputs = XWb
    else:
        outputs = activation(XWb)
    return outputs, W, b