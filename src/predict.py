# src/predict.py

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import argparse # 用于命令行参数解析

# 从当前目录的模块中导入
try:
    from . import config
except ImportError:
    # 如果直接运行此脚本 (例如 python src/predict.py --image_path path/to/image.jpg)
    import config

def load_and_preprocess_image(image_path):
    """
    加载并预处理单张图片以适应模型输入。
    :param image_path: 要预测的图片的路径
    :return: 预处理后的图片数组
    """
    if not os.path.exists(image_path):
        print(f"错误：图片路径 '{image_path}' 不存在。")
        return None

    try:
        # 加载图片，目标尺寸与训练时一致
        img = image.load_img(image_path, target_size=config.IMAGE_SIZE)
        
        # 将图片转换为numpy数组
        img_array = image.img_to_array(img)
        
        # 归一化像素值 (与训练时一致)
        img_array /= 255.0
        
        # 模型期望的输入是 (batch_size, height, width, channels)
        # 因此我们需要在最前面增加一个维度
        img_array_expanded = np.expand_dims(img_array, axis=0)
        
        return img_array_expanded
    except Exception as e:
        print(f"加载或预处理图片 '{image_path}' 时发生错误: {e}")
        return None

def predict_image(model_path, image_path):
    """
    加载模型并对单张图片进行预测。
    :param model_path: 已训练模型的路径 (.h5 文件)
    :param image_path: 要预测的图片的路径
    """
    if not os.path.exists(model_path):
        print(f"错误：模型文件路径 '{model_path}' 不存在。请确保模型已训练并保存在此路径。")
        return

    # 1. 加载模型
    print(f"正在从 '{model_path}' 加载模型...")
    try:
        model = tf.keras.models.load_model(model_path)
        print("模型加载成功。")
    except Exception as e:
        print(f"加载模型时发生错误: {e}")
        return

    # 2. 加载并预处理图片
    print(f"正在加载并预处理图片: '{image_path}'...")
    processed_image = load_and_preprocess_image(image_path)

    if processed_image is None:
        return

    # 3. 进行预测
    print("正在进行预测...")
    try:
        prediction = model.predict(processed_image)
        # prediction 是一个包含单个值的数组，例如 [[0.03]] 或 [[0.98]]
        # 这个值是sigmoid的输出，表示图片是 'dog' (类别1) 的概率
        # (假设 'cats' 是类别0, 'dogs' 是类别1，这取决于ImageDataGenerator的class_indices)
        
        print(f"原始预测输出 (概率值): {prediction[0][0]:.4f}")

        # 确定类别
        # 我们需要知道哪个索引对应哪个类别。
        # 通常 ImageDataGenerator 会按字母顺序分配，或者我们可以从 config.CLASSES 获取
        # config.CLASSES = ['cats', 'dogs'] -> cats=0, dogs=1
        
        predicted_class_index = int(prediction[0][0] > 0.5) # 大于0.5认为是类别1，否则是类别0
        predicted_class_name = config.CLASSES[predicted_class_index]
        confidence = prediction[0][0] if predicted_class_index == 1 else 1 - prediction[0][0]

        print(f"预测结果: 这是一张 **{predicted_class_name}** (置信度: {confidence:.2%})")

    except Exception as e:
        print(f"进行预测时发生错误: {e}")


if __name__ == '__main__':
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="使用训练好的模型预测图片是猫还是狗。")
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="需要预测的图片的路径。"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=config.MODEL_PATH, # 默认使用config中定义的模型路径
        help=f"已训练模型的路径 (.h5文件)。默认为: {config.MODEL_PATH}"
    )
    
    args = parser.parse_args()

    # 确保 TensorFlow 使用 GPU (如果可用)，但对于预测来说通常CPU也足够快
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

    predict_image(args.model_path, args.image_path)
