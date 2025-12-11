from tensorboard import summary
from load import loadData, kinds, pre_sample,past_sample
from model import *
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary
import json

model_type = 3
output_model = f'./output_file/ecg_model_{model_type}.pth'
output_json = f'./output_file/training_history_{model_type}.json'

# 训练循环
num_epochs = 30
test_ratio = 0.2
valid_ratio = 0.1

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device:{device}")

    X_train, Y_train, X_test, Y_test = loadData(test_ratio=test_ratio, dataset_type="MIT-BIH")
    X_train, Y_train, X_test, Y_test = X_train.to(device), Y_train.to(device), X_test.to(device), Y_test.to(device)
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    # exit()

    # 创建模型
    model = Model(model_type=model_type, output_dim=len(kinds)).to(device)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    # 模型信息
    summary(model, input_size=(1, pre_sample + past_sample))
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # 创建数据加载器
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # 训练历史记录
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }

    # 划分训练集和验证集
    total_train_samples = len(X_train)
    val_size = int(valid_ratio * total_train_samples)

    print(f"开始训练，共 {num_epochs} 个epoch...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # 训练阶段
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        # 计算训练指标
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_inputs, val_targets = X_train[:val_size], Y_train[:val_size]
            val_outputs = model(val_inputs)
            loss = criterion(val_outputs, val_targets)
            val_loss = loss.item()
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_total = val_targets.size(0)
            val_correct = (val_predicted == val_targets).sum().item()

        val_accuracy = 100 * val_correct / val_total
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

    # 保存模型
    torch.save(model.state_dict(), output_model)
    print(f"模型已保存为 {output_model.split('/')[-1]}")

    # 保存训练历史
    with open(output_json, 'w') as f:
        json.dump(history, f)
    print(f"训练历史已保存为 {output_json.split('/')[-1]}")

    # 测试集预测
    print(f"\n开始预测测试数据 ({len(X_test)} 个样本)...")
    print("=" * 60)

    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for i in tqdm(range(len(X_test))):
            # 获取单个样本进行预测
            ecg_data = X_test[i:i+1].to(device)
            outputs = model(ecg_data)

            # 使用softmax获取概率
            probabilities = torch.softmax(outputs, dim=1)

            # 获取预测结果
            predicted_class = torch.argmax(probabilities, dim=1).item()
            true_class = Y_test[i].item()

            # 判断预测是否正确
            is_correct = predicted_class == true_class
            if is_correct:
                correct_predictions += 1
            total_predictions += 1

    # 计算并输出最终准确率
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print("=" * 60)
        print(f"预测完成!")
        print(f"预测准确率: {accuracy:.2%} ({correct_predictions}/{total_predictions})")

if __name__ == '__main__':
    main()