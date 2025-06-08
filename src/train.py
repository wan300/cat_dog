# src/train.py

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
import numpy as np # Added for numerical operations
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc # Added for metrics
import seaborn as sns # Added for confusion matrix heatmap

# 从当前目录的模块中导入
try:
    from . import config
    from .data_preprocessing import create_data_generators
    from .model import create_model
except ImportError:
    # 如果直接运行此脚本 (例如 python src/train.py)
    import config
    from data_preprocessing import create_data_generators
    from model import create_model

def plot_training_history(history, save_path_base):
    """
    绘制并保存训练过程中的准确率和损失曲线。
    :param history: Keras训练历史对象
    :param save_path_base: 保存图片的基础路径和文件名 (不含扩展名)
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc)) # 或者 history.epoch

    plt.figure(figsize=(12, 6))

    # 绘制准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='训练准确率 (Training Accuracy)')
    plt.plot(epochs_range, val_acc, label='验证准确率 (Validation Accuracy)')
    plt.legend(loc='lower right')
    plt.title('训练和验证准确率 (Training and Validation Accuracy)')
    plt.xlabel('轮次 (Epoch)')
    plt.ylabel('准确率 (Accuracy)')

    # 绘制损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='训练损失 (Training Loss)')
    plt.plot(epochs_range, val_loss, label='验证损失 (Validation Loss)')
    plt.legend(loc='upper right')
    plt.title('训练和验证损失 (Training and Validation Loss)')
    plt.xlabel('轮次 (Epoch)')
    plt.ylabel('损失 (Loss)')

    plt.tight_layout()
    # 保存图像
    plot_save_path = f"{save_path_base}_history.png"
    plt.savefig(plot_save_path)
    print(f"训练历史曲线图已保存至: {plot_save_path}")
    plt.close() # 关闭图像以释放内存

# --- 新增的可视化函数 ---
def plot_confusion_matrix_and_report(y_true, y_pred_probs, class_names, save_path_base):
    """
    计算、绘制并保存混淆矩阵，同时打印并保存分类报告。
    :param y_true: 真实标签
    :param y_pred_probs: 模型预测的概率 (sigmoid 输出)
    :param class_names: 类别名称列表 (例如 ['cats', 'dogs'])
    :param save_path_base: 保存图片和报告的基础路径和文件名 (不含扩展名)
    """
    # 将概率转换为类别标签 (0 或 1)
    y_pred_classes = (y_pred_probs > 0.5).astype(int).flatten()

    # 1. 分类报告
    print("\n--- 分类报告 (Validation Set) ---")
    report = classification_report(y_true, y_pred_classes, target_names=class_names)
    print(report)
    report_save_path = f"{save_path_base}_classification_report.txt"
    try:
        with open(report_save_path, 'w') as f:
            f.write("Classification Report (Validation Set):\n")
            f.write(report)
        print(f"分类报告已保存至: {report_save_path}")
    except IOError as e:
        print(f"错误：无法保存分类报告至 {report_save_path}: {e}")


    # 2. 混淆矩阵
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 16}) # 增大 annot 字体大小
    plt.title('混淆矩阵 (Validation Set)', fontsize=15)
    plt.ylabel('真实类别 (Actual Class)', fontsize=12)
    plt.xlabel('预测类别 (Predicted Class)', fontsize=12)
    cm_save_path = f"{save_path_base}_confusion_matrix.png"
    try:
        plt.savefig(cm_save_path)
        print(f"混淆矩阵图已保存至: {cm_save_path}")
    except IOError as e:
        print(f"错误：无法保存混淆矩阵图至 {cm_save_path}: {e}")
    plt.close() # 关闭图像以释放内存

def plot_roc_curve(y_true, y_pred_probs, save_path_base):
    """
    计算、绘制并保存ROC曲线及AUC值。
    :param y_true: 真实标签
    :param y_pred_probs: 模型预测的概率 (sigmoid 输出)
    :param save_path_base: 保存图片的基础路径和文件名 (不含扩展名)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs.flatten())
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC 曲线 (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正例率 (False Positive Rate)', fontsize=12)
    plt.ylabel('真正例率 (True Positive Rate)', fontsize=12)
    plt.title('ROC 曲线 (Validation Set)', fontsize=15)
    plt.legend(loc="lower right", fontsize=10)
    roc_save_path = f"{save_path_base}_roc_curve.png"
    try:
        plt.savefig(roc_save_path)
        print(f"ROC曲线图已保存至: {roc_save_path}")
    except IOError as e:
        print(f"错误：无法保存ROC曲线图至 {roc_save_path}: {e}")
    plt.close() # 关闭图像以释放内存
# --- 结束新增的可视化函数 ---

def train():
    """
    执行模型训练过程。
    """
    print("开始训练过程...")

    # 1. 创建数据生成器
    print("步骤 1/6: 创建数据生成器...")
    train_generator, validation_generator = create_data_generators()
    if not train_generator or not validation_generator:
        print("错误：未能创建数据生成器。请检查 data_preprocessing.py 和数据路径。")
        return
    
    if train_generator.samples == 0:
        print(f"错误：训练数据生成器未找到任何图片。请检查 {config.TRAIN_DIR} 目录及其子目录。")
        return
    if validation_generator.samples == 0:
        print(f"错误：验证数据生成器未找到任何图片。请检查 {config.VALIDATION_DIR} 目录及其子目录。")
        return

    print(f"训练集样本数: {train_generator.samples}, 批大小: {config.BATCH_SIZE}")
    print(f"验证集样本数: {validation_generator.samples}, 批大小: {config.BATCH_SIZE}")

    # 2. 创建模型
    print("\n步骤 2/6: 创建并编译模型...")
    model = create_model()
    if not model:
        print("错误：未能创建模型。请检查 model.py。")
        return

    # 3. 定义回调函数 (Callbacks)
    print("\n步骤 3/6: 定义回调函数...")
    if not os.path.exists(config.MODEL_SAVE_DIR):
        try:
            os.makedirs(config.MODEL_SAVE_DIR)
            print(f"已创建模型保存目录: {config.MODEL_SAVE_DIR}")
        except OSError as e:
            print(f"错误: 无法创建目录 {config.MODEL_SAVE_DIR}: {e}")
            return


    checkpoint_filepath = config.MODEL_PATH
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=10, # 增加了耐心值，因为有时学习率调整后会有改善
        verbose=1,
        restore_best_weights=True
    )
    
    reduce_lr_callback = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5, # 增加了耐心值
        min_lr=0.00001,
        verbose=1
    )

    callbacks_list = [model_checkpoint_callback, early_stopping_callback, reduce_lr_callback]

    # 4. 训练模型
    print("\n步骤 4/6: 开始训练模型...")
    print(f"训练轮次 (Epochs): {config.EPOCHS}")
    print(f"批大小 (Batch Size): {config.BATCH_SIZE}")
    print(f"初始学习率 (Initial Learning Rate): {config.LEARNING_RATE}")

    steps_per_epoch = train_generator.samples // config.BATCH_SIZE
    if train_generator.samples % config.BATCH_SIZE > 0:
        steps_per_epoch += 1
    
    validation_steps = validation_generator.samples // config.BATCH_SIZE
    if validation_generator.samples % config.BATCH_SIZE > 0:
        validation_steps += 1

    if steps_per_epoch == 0:
        print(f"错误：steps_per_epoch 为 0。可能是因为训练样本数 ({train_generator.samples}) 小于批大小 ({config.BATCH_SIZE})。请增加训练样本或减小批大小。")
        return
    if validation_steps == 0:
        print(f"错误：validation_steps 为 0。可能是因为验证样本数 ({validation_generator.samples}) 小于批大小 ({config.BATCH_SIZE})。请增加验证样本或减小批大小。")
        return

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=config.EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks_list,
        verbose=1
    )

    print(f"最佳模型已通过 ModelCheckpoint 保存在: {checkpoint_filepath}")

    # 5. 绘制并保存训练历史
    print("\n步骤 5/6: 绘制并保存训练历史曲线图...")
    plot_save_base = os.path.join(config.MODEL_SAVE_DIR, os.path.splitext(config.MODEL_NAME)[0])
    plot_training_history(history, plot_save_base)

    # --- 新增：评估模型并生成更多可视化 ---
    print("\n步骤 6/6: 在验证集上评估最佳模型并生成可视化...")
    
    # 加载在训练过程中保存的最佳模型
    # 注意：如果 EarlyStopping 的 restore_best_weights=True 有效，
    # 当前的 'model' 对象可能已经是最佳模型。但为了确保，我们加载保存的文件。
    try:
        print(f"正在从 {checkpoint_filepath} 加载最佳模型进行评估...")
        best_model = tf.keras.models.load_model(checkpoint_filepath)
    except Exception as e:
        print(f"错误：无法加载最佳模型 {checkpoint_filepath} 进行评估: {e}")
        print("将使用训练结束时的模型进行评估（可能不是最佳模型）。")
        best_model = model # Fallback to the model in memory

    # 确保验证数据生成器从头开始，并且不打乱数据
    validation_generator.reset() 
    
    # 获取真实标签
    # validation_generator.classes 包含了所有验证样本的标签，顺序与 flow_from_directory 读取文件时的顺序一致
    # 当 shuffle=False (在 data_preprocessing.py 中为验证集设置) 时，此顺序是固定的。
    y_true = validation_generator.classes
    
    # 获取预测概率
    # predict 方法会按顺序处理来自生成器的数据批次
    print(f"在 {validation_generator.samples} 个验证样本上进行预测...")
    y_pred_probs = best_model.predict(
        validation_generator,
        steps=validation_steps, # 确保处理所有验证样本
        verbose=1
    )
    
    # 确保预测数量与真实标签数量一致
    if len(y_pred_probs) != len(y_true):
        print(f"警告：预测数量 ({len(y_pred_probs)}) 与真实标签数量 ({len(y_true)}) 不匹配。")
        # 调整 y_true 或 y_pred_probs 的大小以匹配较小者，或者截断到验证样本总数
        # 理想情况下，validation_steps 应该确保它们匹配 validation_generator.samples
        num_samples_to_evaluate = min(len(y_pred_probs), len(y_true), validation_generator.samples)
        y_true = y_true[:num_samples_to_evaluate]
        y_pred_probs = y_pred_probs[:num_samples_to_evaluate]
        print(f"将评估前 {num_samples_to_evaluate} 个样本。")


    if len(y_true) > 0:
         # 绘制混淆矩阵并打印/保存分类报告
        plot_confusion_matrix_and_report(y_true, y_pred_probs, config.CLASSES, plot_save_base)
        # 绘制ROC曲线
        plot_roc_curve(y_true, y_pred_probs, plot_save_base)
    else:
        print("错误：没有可用于评估的真实标签或预测结果。")
    # --- 结束新增评估部分 ---

    print("\n训练和评估完成！")

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"检测到并已配置 {len(gpus)} 个GPU。")
        except RuntimeError as e:
            print(e)
    else:
        print("未检测到GPU，将使用CPU。")

    if config.RANDOM_SEED is not None:
        tf.random.set_seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED) # 为 scikit-learn 和 numpy 添加随机种子
        print(f"已设置随机种子: {config.RANDOM_SEED}")

    train()
