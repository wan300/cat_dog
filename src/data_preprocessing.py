# src/data_preprocessing.py

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os # 导入 os 模块以检查目录是否存在


try:
    from . import config
except ImportError:
    import config # 如果直接运行此脚本，则使用此导入方式

def create_data_generators():
    """
    创建训练和验证数据生成器。
    训练数据生成器会进行数据增强和归一化。
    验证数据生成器仅进行归一化。
    """

    # 检查训练和验证目录是否存在
    if not os.path.exists(config.TRAIN_DIR):
        print(f"错误：训练数据目录 {config.TRAIN_DIR} 不存在。请检查 config.py 中的路径设置和你的数据存放位置。")
        return None, None
    if not os.path.exists(config.VALIDATION_DIR):
        print(f"错误：验证数据目录 {config.VALIDATION_DIR} 不存在。请检查 config.py 中的路径设置和你的数据存放位置。")
        return None, None

    # 训练数据生成器，包含数据增强
    # 像素值归一化到 0-1 范围
    train_datagen = ImageDataGenerator(
        rescale=1./255,  # 将像素值从 0-255 缩放到 0-1
        rotation_range=config.DATA_AUGMENTATION_PARAMS.get('rotation_range', 0),
        width_shift_range=config.DATA_AUGMENTATION_PARAMS.get('width_shift_range', 0.0),
        height_shift_range=config.DATA_AUGMENTATION_PARAMS.get('height_shift_range', 0.0),
        shear_range=config.DATA_AUGMENTATION_PARAMS.get('shear_range', 0.0),
        zoom_range=config.DATA_AUGMENTATION_PARAMS.get('zoom_range', 0.0),
        horizontal_flip=config.DATA_AUGMENTATION_PARAMS.get('horizontal_flip', False),
        fill_mode=config.DATA_AUGMENTATION_PARAMS.get('fill_mode', 'nearest')
    )

    # 验证数据生成器，仅进行像素值归一化，不进行数据增强
    validation_datagen = ImageDataGenerator(rescale=1./255)

    print(f"从目录 {config.TRAIN_DIR} 加载训练数据...")
    train_generator = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=config.IMAGE_SIZE,  # 将所有图像调整为此大小
        batch_size=config.BATCH_SIZE,
        class_mode='binary',  # 因为是二分类（猫/狗），所以使用 'binary'
        classes=config.CLASSES, # 明确指定类别顺序
        shuffle=True, # 打乱数据
        seed=config.RANDOM_SEED # 可选，用于复现
    )

    print(f"从目录 {config.VALIDATION_DIR} 加载验证数据...")
    validation_generator = validation_datagen.flow_from_directory(
        config.VALIDATION_DIR,
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
        classes=config.CLASSES,
        shuffle=False, # 验证集通常不需要打乱
        seed=config.RANDOM_SEED
    )
    
    # 打印类别索引，确保与预期一致
    print(f"训练数据类别索引: {train_generator.class_indices}")
    print(f"验证数据类别索引: {validation_generator.class_indices}")

    # 检查是否成功加载了图片
    if train_generator.samples == 0:
        print(f"警告：在训练目录 {config.TRAIN_DIR} 中没有找到图片。请确保你的图片已正确放置在 'cats' 和 'dogs' 子目录下。")
    if validation_generator.samples == 0:
        print(f"警告：在验证目录 {config.VALIDATION_DIR} 中没有找到图片。请确保你的图片已正确放置在 'cats' 和 'dogs' 子目录下。")


    return train_generator, validation_generator

# (可选) 测试函数，用于直接运行此脚本时检查数据生成器是否工作正常
if __name__ == '__main__':
    print("正在测试数据生成器...")
    
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

    train_gen, val_gen = create_data_generators()

    if train_gen and val_gen:
        print(f"成功创建训练数据生成器，共找到 {train_gen.samples} 张图片，分为 {train_gen.num_classes} 类。")
        print(f"成功创建验证数据生成器，共找到 {val_gen.samples} 张图片，分为 {val_gen.num_classes} 类。")

        # (可选) 显示一批增强后的图像，以验证数据增强是否按预期工作
        # import matplotlib.pyplot as plt
        # x_batch, y_batch = next(train_gen)
        # plt.figure(figsize=(12, 12))
        # for i in range(min(9, config.BATCH_SIZE)): # 最多显示9张图
        #     plt.subplot(3, 3, i + 1)
        #     plt.imshow(x_batch[i])
        #     plt.title(f"Class: {config.CLASSES[int(y_batch[i])]}")
        #     plt.axis('off')
        # plt.tight_layout()
        # plt.show()
    else:
        print("创建数据生成器失败。请检查错误信息。")

