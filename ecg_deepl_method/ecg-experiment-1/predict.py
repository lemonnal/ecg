from tensorboard import summary
from load import loadData, kinds, pre_sample, past_sample
from model import *
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary

model_type = 3
load_model = f"./output_file/ecg_model_{model_type}.pth"

# 设置matplotlib字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device:{device}")

    X_train, Y_train, X_test, Y_test = loadData(dataset_type="MIT-BIH")
    X_train, Y_train, X_test, Y_test = X_train.to(device), Y_train.to(device), X_test.to(device), Y_test.to(device)
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

    # # 合并训练和测试数据
    # X_all = torch.cat([X_train, X_test], dim=0)
    # Y_all = torch.cat([Y_train, Y_test], dim=0)
    # print(f"合并后数据: {X_all.shape}, {Y_all.shape}")

    # 合并训练和测试数据
    X_all = X_test
    Y_all = Y_test
    print(f"合并后数据: {X_all.shape}, {Y_all.shape}")

    # 创建模型
    model = Model(model_type=model_type, output_dim=len(kinds)).to(device)
    model.load_state_dict(torch.load(load_model, map_location='cuda'))
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    # 模型信息
    summary(model, input_size=(1, pre_sample + past_sample))

    # 合并数据预测
    print(f"\n开始预测所有合并数据 ({len(X_all)} 个样本)...")
    print("=" * 60)

    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        search_result = 0
        for i in tqdm(range(len(X_all))):
            # 获取单个样本进行预测
            ecg_data = X_all[i:i + 1]
            outputs = model(ecg_data)

            # 使用softmax获取概率
            probabilities = torch.softmax(outputs, dim=1)
            # 获取预测结果
            predicted_class = torch.argmax(probabilities, dim=1).item()
            true_class = Y_all[i].item()

            # 绘制前10个样本的波形图和概率
            # if 302 <= i < 308:
            # if probabilities.max() < 0.8 and search_result <= 10:
            if true_class != predicted_class and search_result <= 20:
                search_result += 1
                # 创建图像
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                fig.suptitle(f'ECG Sample #{i} Prediction Results', fontsize=16, fontweight='bold')

                # 提取ECG数据
                ecg_signal = ecg_data.squeeze().cpu().numpy()

                # 绘制ECG波形
                ax1.plot(ecg_signal, 'b-', linewidth=1.5)
                ax1.set_title('ECG Waveform', fontsize=14, fontweight='bold')
                ax1.set_xlabel('Sample Points', fontsize=12)
                ax1.set_ylabel('Amplitude (mV)', fontsize=12)
                ax1.grid(True, alpha=0.3)
                ax1.set_xlim(0, pre_sample + past_sample)

                # # 添加R峰标记（简单检测）
                # r_peaks = []
                # threshold = np.mean(np.abs(ecg_signal)) + 2 * np.std(ecg_signal)
                # for j in range(50, 250):
                #     if ecg_signal[j] > threshold and ecg_signal[j] > ecg_signal[j-1] and ecg_signal[j] > ecg_signal[j+1]:
                #         r_peaks.append(j)
                #
                # for peak in r_peaks[:3]:
                #     ax1.axvline(x=peak, color='r', linestyle='--', alpha=0.7)
                #     ax1.text(peak, max(ecg_signal)*0.8, 'R', fontsize=10, color='red')

                # 绘制概率柱状图
                num_classes = len(kinds)
                colors = ['green' if j == predicted_class else 'lightblue' for j in range(num_classes)]
                bars = ax2.bar(range(num_classes), probabilities.squeeze().cpu().numpy(), color=colors)

                # 设置柱状图标签
                ax2.set_title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Arrhythmia Type', fontsize=12)
                ax2.set_ylabel('Probability', fontsize=12)
                ax2.set_xticks(range(num_classes))
                ax2.set_xticklabels(kinds, rotation=45, ha='right')  # 旋转标签以避免重叠
                ax2.set_ylim(0, 1)
                ax2.grid(True, alpha=0.3, axis='y')

                # 在柱状图上显示概率值
                for j, (bar, prob) in enumerate(zip(bars, probabilities.squeeze())):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{prob:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold', rotation=45)

                # 添加预测结果文本
                pred_text = f"Prediction: {kinds[predicted_class]} ({probabilities.squeeze()[predicted_class]:.3f})\n"
                pred_text += f"True Label: {kinds[true_class]}\n"
                pred_text += f"Result: {'Correct' if predicted_class == true_class else 'Wrong'}"

                ax2.text(0.02, 0.98, pred_text, transform=ax2.transAxes,
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

                plt.tight_layout()
                plt.savefig(f'./output_file/ecg_prediction_{i}.png', dpi=300, bbox_inches='tight')
                plt.show()
                plt.close()

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

        print(f"\n已保存前10个样本的波形图和预测结果")

if __name__ == "__main__":
    main()