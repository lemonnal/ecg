import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, model_type=1, output_dim=5):
        """
        ECG分类模型，通过model_type参数选择使用哪个子模型

        Args:
            model_type (int): 模型类型，1为Model_1，2为Model_2，默认为1
            output_dim (int): 输出维度，分类数量，默认为5
        """
        super(Model, self).__init__()
        self.model_type = model_type
        self.output_dim = output_dim

        # 根据model_type创建对应的子模型
        if model_type == 1:
            self.model = self._create_model1()
        elif model_type == 2:
            self.model = self._create_model2()
        elif model_type == 3:
            self.model = self._create_model3()
        else:
            raise ValueError("model_type必须是1、2或3")

    def _create_model1(self):
        """创建模型1"""
        class SubModel1(nn.Module):
            def __init__(self, output_dim):
                super(SubModel1, self).__init__()
                # 第一个卷积层, 4 个 21x1 卷积核
                self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=21, stride=1, padding=10)
                self.tanh1 = nn.Tanh()
                # 第一个池化层, 最大池化,4 个 3x1 卷积核, 步长为 2
                self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

                # 第二个卷积层, 16 个 23x1 卷积核
                self.conv2 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=23, stride=1, padding=11)
                self.relu2 = nn.ReLU()
                # 第二个池化层, 最大池化,4 个 3x1 卷积核, 步长为 2
                self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

                # 第三个卷积层, 32 个 25x1 卷积核
                self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=25, stride=1, padding=12)
                self.tanh3 = nn.Tanh()
                # 第三个池化层, 平均池化,4 个 3x1 卷积核, 步长为 2
                self.avgpool3 = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)

                # 第四个卷积层, 64 个 27x1 卷积核
                self.conv4 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=27, stride=1, padding=13)
                self.relu4 = nn.ReLU()

                # 全连接层
                self.flatten = nn.Flatten()
                # 计算展平后的特征数
                # 经过三次下采样：350 -> 176 -> 89 -> 45
                # 最终通道数为64
                self.fc1 = nn.Linear(64 * 45, 128)
                self.dropout = nn.Dropout(0.2)
                self.fc2 = nn.Linear(128, output_dim)

            def forward(self, x):
                x = self.conv1(x)
                x = self.tanh1(x)
                x = self.maxpool1(x)

                x = self.conv2(x)
                x = self.relu2(x)
                x = self.maxpool2(x)

                x = self.conv3(x)
                x = self.tanh3(x)
                x = self.avgpool3(x)

                x = self.conv4(x)
                x = self.relu4(x)

                x = self.flatten(x)
                x = self.fc1(x)
                x = self.dropout(x)
                x = self.fc2(x)

                return x

        return SubModel1(self.output_dim)

    def _create_model2(self):
        """创建模型2"""
        class SubModel2(nn.Module):
            def __init__(self, output_dim):
                super(SubModel2, self).__init__()
                # 第一部分：input->conv->bn_relu
                self.conv = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3)
                self.bn = nn.BatchNorm1d(64)
                self.relu = nn.ReLU()

                # 第二部分：特殊的残差块
                self.conv1 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
                self.bn1 = nn.BatchNorm1d(64)
                self.relu1 = nn.ReLU()
                self.dropout1 = nn.Dropout(0.2)
                self.conv2 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
                self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)

                # 第三部分：15个相同的残差块（循环使用）
                self.conv_block1 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
                self.bn_block1 = nn.BatchNorm1d(64)
                self.relu_block1 = nn.ReLU()
                self.dropout_block1 = nn.Dropout(0.2)
                self.conv_block2 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
                self.bn_block2 = nn.BatchNorm1d(64)
                self.relu_block2 = nn.ReLU()
                self.dropout_block2 = nn.Dropout(0.2)
                self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)

                # 第四部分：bn->relu->dense->softmax
                self.final_bn = nn.BatchNorm1d(64)
                self.final_relu = nn.ReLU()
                self.flatten = nn.Flatten()
                self.fc1 = nn.Linear(64, 128)
                self.dropout = nn.Dropout(0.2)
                self.fc2 = nn.Linear(128, output_dim)

            def forward(self, x):
                # 第一部分：input->conv->bn_relu
                x = self.conv(x)
                x = self.bn(x)
                x = self.relu(x)

                # 第二部分：进行一个残差连接（conv->bn->relu-dropout->conv）+(maxpool）
                shortcut = x
                y = self.conv1(x)
                y = self.bn1(y)
                y = self.relu1(y)
                y = self.dropout1(y)
                y = self.conv2(y)
                x = y + shortcut
                x = self.maxpool1(x)

                # 第三部分：进行残差连接((bn->relu->dropout->conv->bn->relu->dropout->conv)+(maxpool))*15次
                for i in range(15):
                    shortcut = x
                    # 第一个卷积块
                    y = self.bn_block1(x)
                    y = self.relu_block1(y)
                    y = self.dropout_block1(y)
                    y = self.conv_block1(y)

                    # 第二个卷积块
                    y = self.bn_block2(y)
                    y = self.relu_block2(y)
                    y = self.dropout_block2(y)
                    y = self.conv_block2(y)

                    # 残差连接
                    x = y + shortcut

                    # 检查序列长度，如果太小就不进行池化
                    if x.size(-1) > 1:
                        x = self.maxpool(x)

                # 第四部分：bn->relu->dense->softmax
                x = self.final_bn(x)
                x = self.final_relu(x)
                x = self.flatten(x)
                x = x.view(x.size(0), -1)  # 确保展平

                # 自适应处理特征维度
                if x.size(1) > 64:
                    x = x.view(x.size(0), 64, -1).mean(dim=2)

                x = self.fc1(x)
                x = self.dropout(x)
                x = self.fc2(x)

                return x

        return SubModel2(self.output_dim)

    def _create_model3(self):
        """创建模型3 - 基于Keras ResNet架构的ECG分类模型"""

        class SubModel3(nn.Module):
            def __init__(self, output_dim):
                super(SubModel3, self).__init__()

                # 初始卷积层
                self.initial_conv = nn.Conv1d(in_channels=1, out_channels=32,
                                            kernel_size=16, stride=1, padding=8)
                self.initial_bn = nn.BatchNorm1d(32)
                self.initial_relu = nn.ReLU()

                # 残差块配置
                self.conv_subsample_lengths = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
                self.conv_num_skip = 2
                self.conv_increase_channels_at = 4
                self.conv_num_filters_start = 32
                self.conv_filter_length = 16
                self.conv_dropout = 0.2

                # 创建16个残差块的所有层，直接集成到SubModel3中
                self._create_residual_blocks()

                # 最终层
                self.final_bn = nn.BatchNorm1d(256)
                self.final_relu = nn.ReLU()

                # 自适应平均池化处理可变长度输入
                self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

                # 全连接层
                self.fc1 = nn.Linear(256, 128)
                self.dropout = nn.Dropout(0.2)
                self.fc2 = nn.Linear(128, output_dim)

            def _create_residual_blocks(self):
                """创建16个残差块的所有层，集成到SubModel3中"""
                # 使用ModuleDict来组织残差块的层
                self.residual_layers = nn.ModuleDict()

                for i, subsample_length in enumerate(self.conv_subsample_lengths):
                    num_filters = self._get_num_filters_at_index(i)
                    in_channels = self._get_input_channels(i)
                    use_padding = (i > 0 and i % self.conv_increase_channels_at == 0)

                    # 为每个残差块创建层
                    block_prefix = f'block_{i}_'

                    # 快捷连接路径
                    if subsample_length > 1:
                        self.residual_layers[f'{block_prefix}shortcut'] = nn.MaxPool1d(
                            kernel_size=subsample_length, stride=subsample_length)
                    else:
                        self.residual_layers[f'{block_prefix}shortcut'] = nn.Identity()

                    # 通道扩展
                    if use_padding and num_filters > in_channels:
                        self.residual_layers[f'{block_prefix}expand'] = nn.Conv1d(
                            in_channels, num_filters, kernel_size=1)
                    else:
                        self.residual_layers[f'{block_prefix}expand'] = nn.Identity()

                    # 第一个卷积层
                    self.residual_layers[f'{block_prefix}conv1'] = nn.Conv1d(
                        in_channels, num_filters, kernel_size=self.conv_filter_length,
                        stride=subsample_length, padding=self.conv_filter_length//2)
                    self.residual_layers[f'{block_prefix}bn1'] = nn.BatchNorm1d(num_filters)

                    # 第二个卷积层
                    self.residual_layers[f'{block_prefix}conv2'] = nn.Conv1d(
                        num_filters, num_filters, kernel_size=self.conv_filter_length,
                        stride=1, padding=self.conv_filter_length//2)
                    self.residual_layers[f'{block_prefix}bn2'] = nn.BatchNorm1d(num_filters)

                    # 激活和dropout
                    self.residual_layers[f'{block_prefix}relu'] = nn.ReLU()
                    self.residual_layers[f'{block_prefix}dropout'] = nn.Dropout(self.conv_dropout)

            def _residual_block_forward(self, x, block_index):
                """单个残差块的前向传播"""
                block_prefix = f'block_{block_index}_'

                # 快捷连接
                shortcut = self.residual_layers[f'{block_prefix}shortcut'](x)

                # 通道扩展（如果需要）
                num_filters = self._get_num_filters_at_index(block_index)
                in_channels = self._get_input_channels(block_index)
                use_padding = (block_index > 0 and block_index % self.conv_increase_channels_at == 0)

                if use_padding and num_filters > in_channels:
                    # 使用1x1卷积进行通道扩展，而不是零填充
                    shortcut = self.residual_layers[f'{block_prefix}expand'](shortcut)

                # 主分支
                out = self.residual_layers[f'{block_prefix}conv1'](x)
                out = self.residual_layers[f'{block_prefix}bn1'](out)
                out = self.residual_layers[f'{block_prefix}relu'](out)
                out = self.residual_layers[f'{block_prefix}dropout'](out)

                out = self.residual_layers[f'{block_prefix}conv2'](out)
                out = self.residual_layers[f'{block_prefix}bn2'](out)

                # 确保尺寸匹配后再进行残差连接
                if out.shape != shortcut.shape:
                    # 如果长度不匹配，调整快捷连接的长度
                    min_length = min(out.shape[2], shortcut.shape[2])
                    out = out[:, :, :min_length]
                    shortcut = shortcut[:, :, :min_length]

                # 残差连接
                out = out + shortcut
                out = self.residual_layers[f'{block_prefix}relu'](out)

                return out

            def _get_num_filters_at_index(self, index):
                """计算指定索引处的滤波器数量"""
                return 2**(index // self.conv_increase_channels_at) * self.conv_num_filters_start

            def _get_input_channels(self, index):
                """获取残差块的输入通道数"""
                if index == 0:
                    return self.conv_num_filters_start
                else:
                    return self._get_num_filters_at_index(index - 1)

            def forward(self, x):
                # 初始卷积
                x = self.initial_conv(x)
                x = self.initial_bn(x)
                x = self.initial_relu(x)

                # 通过所有残差块
                for i in range(len(self.conv_subsample_lengths)):
                    x = self._residual_block_forward(x, i)

                # 最终批归一化和激活
                x = self.final_bn(x)
                x = self.final_relu(x)

                # 全局平均池化
                x = self.adaptive_pool(x)
                x = x.squeeze(-1)  # 移除最后一维

                # 全连接层
                x = self.fc1(x)
                x = self.dropout(x)
                x = self.fc2(x)

                return x

        return SubModel3(self.output_dim)

    def forward(self, x):
        """前向传播，直接调用选中的子模型"""
        return self.model(x)


# 向后兼容
class Model_1(Model):
    def __init__(self):
        super(Model_1, self).__init__(model_type=1)


class Model_2(Model):
    def __init__(self):
        super(Model_2, self).__init__(model_type=2)


class Model_3(Model):
    def __init__(self):
        super(Model_3, self).__init__(model_type=3)