# src/model.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam # 导入 Adam 优化器

# 从 config 文件中导入配置参数
try:
    from . import config
except ImportError:
    import config # 如果直接运行此脚本，则使用此导入方式

def create_model():
    """
    创建并编译CNN模型。
    """
    model = Sequential()

    # --- 卷积基 ---
    # 第一个卷积块
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=config.INPUT_SHAPE, padding='same'))
    model.add(BatchNormalization()) # 批量归一化
    model.add(MaxPooling2D((2, 2)))
    # model.add(Dropout(0.25)) # 可选的 Dropout 层

    # 第二个卷积块
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    # model.add(Dropout(0.25))

    # 第三个卷积块
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    # model.add(Dropout(0.25))

    # 第四个卷积块 (可选，可以根据需要增加更多层)
    # model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Dropout(0.25))

    # --- 分类器 ---
    model.add(Flatten()) # 将卷积层的输出展平

    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5)) # Dropout 层，防止过拟合

    # 输出层
    # 因为是二分类问题 (猫/狗)，所以输出层只有一个神经元，并使用 sigmoid 激活函数
    # sigmoid 函数的输出范围是 0 到 1，可以解释为属于正类（比如 'dog'）的概率
    model.add(Dense(1, activation='sigmoid'))

    # 编译模型
    # 我们需要指定优化器、损失函数和评估指标
    optimizer = Adam(learning_rate=config.LEARNING_RATE)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',  # 二分类交叉熵损失函数
                  metrics=['accuracy'])       # 评估指标为准确率

    print("模型结构概览：")
    model.summary() # 打印模型结构

    return model

# (可选) 测试函数，用于直接运行此脚本时检查模型是否能成功创建
if __name__ == '__main__':
    print("正在创建和编译模型...")
    # 确保 TensorFlow 使用 GPU (如果可用)
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #         print(f"检测到并已配置 {len(gpus)} 个GPU。")
    #     except RuntimeError as e:
    #         print(e)
    # else:
    #     print("未检测到GPU，将使用CPU。")

    cnn_model = create_model()
    if cnn_model:
        print("模型成功创建和编译。")
    else:
        print("模型创建失败。")
