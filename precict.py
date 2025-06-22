import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from torch import nn
import torch.serialization as serialization
# 从model模块导入DogClassifier
from model_02 import DogClassifier
#from model import  DogClassifier
# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# 定义犬种名称
BREED_NAMES = [
    "beagle", "dachshund", "dalmatian", "jindo", "maltese",
    "pomeranian", "retriever", "ShihTzu", "toypoodle", "Yorkshirerrier"
]


def predict_single_image(image_path, model_path, class_names=BREED_NAMES):
    """预测单张图片的犬种类别"""
    # 检查文件是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 加载并预处理图片
    image = Image.open(image_path).convert('RGB')

    # 图像预处理
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    ])

    processed_image = transform(image).unsqueeze(0)

    with serialization.safe_globals([DogClassifier, nn.Sequential]):
        model = torch.load(model_path, map_location=torch.device('cpu'),weights_only=False)
    model.eval()


    # 执行预测
    with torch.no_grad():
        output = model(processed_image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    # 准备预测结果
    result = {
        '原始图片': image,
        '预测类别索引': predicted_class.item(),
        '预测类别名称': class_names[predicted_class.item()],
        '置信度': confidence.item() * 100,
        '各类别概率': probabilities.numpy()[0]
    }

    return result


def visualize_prediction(result, class_names=BREED_NAMES):
    """可视化预测结果"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 显示原始图片
    ax1.imshow(np.array(result['原始图片']))
    ax1.set_title(f"预测结果: {result['预测类别名称']} ({result['置信度']:.2f}%)", fontsize=14)
    ax1.axis('off')

    # 显示各类别概率分布
    y_pos = np.arange(len(class_names))
    ax2.barh(y_pos, result['各类别概率'], align='center', color='skyblue')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(class_names)
    ax2.invert_yaxis()  # 从上到下显示
    ax2.set_xlabel('概率', fontsize=12)
    ax2.set_title('各类别预测概率分布', fontsize=14)

    # 在每个条形上显示具体概率值
    for i, v in enumerate(result['各类别概率']):
        ax2.text(v + 0.01, i, f'{v * 100:.1f}%', color='blue', fontweight='bold')

    plt.tight_layout()
    plt.show()


def main():
    print("===== 犬种识别系统 =====")

    image_path = "archive/dachshund/547.Screenshot_17.jpg"
    model_path = "model2_max_0.79.pth"

    try:
        # 执行预测
        prediction = predict_single_image(image_path, model_path)

        # 打印文本结果
        print("\n===== 预测结果 =====")
        print(f"预测类别: {prediction['预测类别名称']}")
        print(f"置信度: {prediction['置信度']:.2f}%")

        # 可视化结果
        visualize_prediction(prediction)

    except Exception as e:
        print(f"预测过程中出错: {e}")

if __name__ == "__main__":
    main()