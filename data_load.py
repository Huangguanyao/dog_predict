import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader, ConcatDataset, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


class BreedDataset(Dataset):
    """自定义数据集类，处理多品种标签映射"""

    def __init__(self, dataset, label_offset=0):
        self.dataset = dataset
        self.label_offset = label_offset
        self.classes = dataset.classes if hasattr(dataset, 'classes') else [str(label_offset)]

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return img, label + self.label_offset

    def __len__(self):
        return len(self.dataset)


def date_load():
    # 设置随机种子确保可重复性
    random.seed(42)
    torch.manual_seed(42)

    # 初始化TensorBoard
    writer = SummaryWriter('runs/dog_breed_experiment_1')

    # 数据集配置
    data_root = r"D:\pycharm hgy\dog_predict\archive"
    breed_folders = [
        "beagle", "dachshund", "dalmatian", "jindo", "maltese",
        "pomeranian", "retriever", "ShihTzu", "toypoodle", "Yorkshirerrier"
    ]

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载并预处理数据
    all_datasets = []
    for breed_idx, breed in enumerate(breed_folders):
        breed_path = os.path.join(data_root, breed)
        if not os.path.exists(breed_path):
            print(f"警告：跳过不存在的品种目录 {breed_path}")
            continue

        try:
            # 原始数据集（标签从0开始）
            raw_dataset = datasets.ImageFolder(root=breed_path, transform=transform)
            # 添加品种偏移量（0-9）
            dataset = BreedDataset(raw_dataset, label_offset=breed_idx)
            all_datasets.append(dataset)
            print(f"加载 {breed} 成功，样本数: {len(dataset)}")
        except Exception as e:
            print(f"加载 {breed} 失败: {str(e)}")
            continue

    # 划分训练测试集
    train_datasets = []
    test_datasets = []
    for dataset in all_datasets:
        half_size = len(dataset)
        half_dataset, _ = random_split(dataset, [half_size, len(dataset) - half_size])

        train_size = int(0.9 * len(half_dataset))
        test_size = len(half_dataset) - train_size
        train_part, test_part = random_split(half_dataset, [train_size, test_size])

        train_datasets.append(train_part)
        test_datasets.append(test_part)

    # 合并数据集
    train_dataset = ConcatDataset(train_datasets)
    test_dataset = ConcatDataset(test_datasets)

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 打印数据集信息
    print(f"\n训练集总样本数: {len(train_dataset)}")
    print(f"测试集总样本数: {len(test_dataset)}")
    writer.add_text('Dataset Info',
                    f'Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}')

    # 可视化函数
    def imshow(img, title=None):
        """显示反归一化后的图片"""
        img = img.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        if title is not None:
            plt.title(title)
        plt.axis('off')

    # 可视化训练样本
    images, labels = next(iter(train_loader))
    plt.figure(figsize=(12, 12))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        breed_name = breed_folders[labels[i]] if labels[i] < len(breed_folders) else "unknown"
        imshow(images[i], title=f"{breed_name}\nLabel: {labels[i]}")
    plt.tight_layout()
    plt.show()

    # 记录到TensorBoard
    img_grid = make_grid(images[:16], nrow=4, normalize=True, scale_each=True)
    writer.add_image('Train Samples', img_grid)

    # 保存样本到本地
    def save_samples(loader, save_dir="train_samples"):
        os.makedirs(save_dir, exist_ok=True)
        images, labels = next(iter(loader))
        for i in range(min(20, len(images))):
            img = images[i].numpy().transpose((1, 2, 0))
            img = np.clip(img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)
            img = (img * 255).astype(np.uint8)
            breed_name = breed_folders[labels[i]] if labels[i] < len(breed_folders) else "unknown"
            Image.fromarray(img).save(
                os.path.join(save_dir, f"sample_{i}_label_{labels[i]}_{breed_name}.png")
            )

    train_len=len(train_dataset)
    test_len=len(test_dataset)

    save_samples(train_loader)

    return train_loader, test_loader,train_len,test_len


# 使用示例
if __name__ == "__main__":
    train_loader, test_loader = date_load()

    # 验证标签范围
    sample_labels = next(iter(train_loader))[1]
    print("\n验证标签范围:")
    print(f"最小标签: {torch.min(sample_labels).item()}")
    print(f"最大标签: {torch.max(sample_labels).item()}")
   # print(f"品种数量: {len(breed_folders)}")

    # 训练完成后关闭writer
    # writer.close()