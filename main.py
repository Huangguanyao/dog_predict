import torchvision
from torch.utils.tensorboard import SummaryWriter

#from model import *  # 从model.py文件导入自定义模型
from model_02 import *
# 准备数据集
from torch import nn
from torch.utils.data import DataLoader
from data_load import *
from data_load2 import *

train_dataloader,test_dataloader,train_data_size,test_data_size=date_load2()


# 获取数据集长度
#print("训练数据集的长度为：{}".format(train_data_size))
#print("测试数据集的长度为：{}".format(test_data_size))

# 使用DataLoader加载数据集（设置批量大小）
#train_dataloader = DataLoader(train_data, batch_size=64)
#test_dataloader = DataLoader(test_data, batch_size=64)

# 创建自定义模型实例（假设Model类在model.py中定义）
model = DogClassifier()

# 定义损失函数（交叉熵损失，适用于分类任务）
loss_function = nn.CrossEntropyLoss()

# 定义优化器（随机梯度下降，学习率0.01）
#learning_rate = 2e-2  # 0.02
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,               # 通常更小的学习率
    betas=(0.9, 0.999),    # 动量参数
    weight_decay=0.01      # 解耦权重衰减
)

# 训练参数设置
total_train_steps = 0  # 记录总训练步数
total_test_steps = 0  # 记录总测试步数
epochs = 50  # 训练轮数

# 初始化TensorBoard日志写入器
writer = SummaryWriter("../logs_train")

for epoch in range(epochs):
    print(f"-------第 {epoch + 1} 轮训练开始-------")

    # --------------------- 训练阶段 ---------------------
    model.train()  # 设置模型为训练模式（启用dropout、BatchNorm等）
    for batch_data in train_dataloader:
        images, labels = batch_data  # 解包数据（图像和真实标签）
        outputs = model(images)  # 前向传播：模型预测
        loss = loss_function(outputs, labels)  # 计算损失

        # 优化步骤：梯度清零 -> 反向传播 -> 更新参数
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 根据梯度更新参数

        total_train_steps += 1  # 步数累加
        # 每100步打印训练信息并写入TensorBoard
        if total_train_steps % 100 == 0:
            print(f"训练次数：{total_train_steps}, 损失值：{loss.item()}")
            writer.add_scalar("train_loss", loss.item(), total_train_steps)  # 记录训练损失

    # --------------------- 测试阶段 ---------------------
    model.eval()  # 设置模型为评估模式（关闭dropout、BatchNorm等）
    total_test_loss = 0.0  # 测试集总损失
    total_accuracy = 0.0  # 测试集总正确数

    class_names = ["beagle", "dachshund", "dalmatian", "jindo", "maltese",
        "pomeranian", "retriever", "ShihTzu", "toypoodle", "Yorkshirerrier"]

    # 新增：初始化每类统计
    num_classes = len(class_names)
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():  # 测试阶段不计算梯度
        for batch_data in test_dataloader:
            images, labels = batch_data
            outputs = model(images)
            loss = loss_function(outputs, labels)
            total_test_loss += loss.item()  # 累加损失值

            # 原始整体准确率计算（保持不动）
            _, predicted = torch.max(outputs, 1)
            batch_correct = (predicted == labels).sum().item()
            total_accuracy += batch_correct

            # 新增的类别级别统计
            for label, pred in zip(labels, predicted):
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1

    # 打印测试结果
    print(f"整体测试集上的损失：{total_test_loss}")
    print(f"整体测试集上的正确率：{total_accuracy / test_data_size}")
    print("各类别准确率：")
    for i in range(num_classes):
        if class_total[i] > 0:
            print(f"类别{i}: {class_correct[i] / class_total[i]:.4f}")

    # 写入TensorBoard日志
    writer.add_scalar("test_loss", total_test_loss, total_test_steps)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_steps)
    total_test_steps += 1  # 测试步数累加

    # 保存当前轮次的模型
    torch.save(model, f"model_{epoch}.pth") 
    print("模型已保存")

# 关闭TensorBoard写入器
writer.close()