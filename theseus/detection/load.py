import os
import torch


def load_yolo(weight):
    """Load YOLOv5 model, temporarily change directory for loading, then restore."""

    # 保存当前工作目录
    original_dir = os.getcwd()
    print("当前工作目录是：", original_dir)


    os.chdir("theseus/detection")
    print("更改后的工作目录是：", os.getcwd())

    # 加载模型权重
    model = torch.load(weight)
    print("模型加载成功")

    # 无论发生什么情况，都还原工作目录
    os.chdir(original_dir)
    print("工作目录还原为：", os.getcwd())

    return model
