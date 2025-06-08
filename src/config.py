# src/config.py

import os

# --- 路径配置 ---
# 获取当前 config.py 文件所在目录的绝对路径 (src/)
CONFIG_FILE_DIR = os.path.dirname(os.path.abspath(__file__))

# 通过从 CONFIG_FILE_DIR 向上一级获取项目根目录 (cat_dog/)
PROJECT_ROOT_DIR = os.path.dirname(CONFIG_FILE_DIR)

# 相对于项目根目录定义数据和模型目录
DATA_DIR = os.path.join(PROJECT_ROOT_DIR, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALIDATION_DIR = os.path.join(DATA_DIR, 'validation')
TEST_DIR = os.path.join(DATA_DIR, 'test')  # 如果准备了测试集

MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT_DIR, 'saved_models')
MODEL_NAME = 'cat_dog_classifier_v1.h5'  # 可以更改版本/名称
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)


# --- 图像处理配置 ---
IMG_WIDTH = 250  # 目标图像宽度
IMG_HEIGHT = 250  # 目标图像高度
IMAGE_SIZE = (IMG_WIDTH, IMG_HEIGHT)
CHANNELS = 3  # 图像通道数 (3 表示 RGB，1 表示灰度)
INPUT_SHAPE = (IMG_WIDTH, IMG_HEIGHT, CHANNELS)

# --- 训练配置 ---
BATCH_SIZE = 32  # 每次梯度更新的样本数
EPOCHS = 17  # 遍历整个训练数据集的次数
LEARNING_RATE = 0.001  # 优化器的初始学习率

# --- 类别信息 ---
# Keras ImageDataGenerator 从子文件夹名称推断类别。
# 显式定义它们在一致性方面很有用，尤其是在使用 predict.py 时。
CLASSES = ['cats', 'dogs']  # 确保此顺序与文件夹结构匹配（如果显式使用）
NUM_CLASSES = len(CLASSES)

# --- 数据增强参数（可选）---
DATA_AUGMENTATION_PARAMS = {
    'rotation_range': 20,       # 随机旋转的角度范围（度）
    'width_shift_range': 0.2,   # 水平方向随机平移的比例（相对于总宽度）
    'height_shift_range': 0.2,  # 垂直方向随机平移的比例（相对于总高度）
    'shear_range': 0.2,         # 逆时针方向的剪切角度（度）
    'zoom_range': 0.2,          # 随机缩放范围
    'horizontal_flip': True,    # 随机水平翻转输入
    'fill_mode': 'nearest'      # 填充新创建像素的策略
}

# --- 杂项 ---
RANDOM_SEED = 42  # 随机种子（可选，但有助于一致的结果）

# 确保模型保存目录存在（也可以在 train.py 中保存之前创建）
# 通常最好在实际保存脚本中完成，但如果需要，也可以放在这里。
# if not os.path.exists(MODEL_SAVE_DIR):
#     os.makedirs(MODEL_SAVE_DIR)