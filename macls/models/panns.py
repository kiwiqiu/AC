import torch.nn as nn
import torch.nn.functional as F
from macls.models.pooling import SelfAttentivePooling

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, pool_size=(2, 2), pool_type='avg'):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x = F.avg_pool2d(x, kernel_size=pool_size) + F.max_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception(
                f'Pooling type of {pool_type} is not supported. It must be one of "max", "avg" and "avg+max".')
        return x


class ConvBlock5x5(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock5x5, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(5, 5),
                               stride=(1, 1),
                               padding=(2, 2))
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x, pool_size=(2, 2), pool_type='avg'):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x = F.avg_pool2d(x, kernel_size=pool_size) + F.max_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception(
                f'Pooling type of {pool_type} is not supported. It must be one of "max", "avg" and "avg+max".')
        return x


class PANNS_CNN6(nn.Module):
    """
    ---------------------改池化层SAP--------------------------------------------
    The CNN14(14-layer CNNs) mainly consist of 4 convolutional blocks while each convolutional
    block consists of 1 convolutional layers with a kernel size of 5 × 5.

    Reference:
        PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition
        https://arxiv.org/pdf/1912.10211.pdf
    """
    emb_size = 512

    def __init__(self, num_class, input_size, dropout=0.1, extract_embedding: bool = True):

        super(PANNS_CNN6, self).__init__()
        self.bn0 = nn.BatchNorm2d(input_size)
        self.conv_block1 = ConvBlock5x5(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock5x5(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock5x5(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock5x5(in_channels=256, out_channels=512)
        # 修改2：初始化注意力池化层
        self.attentive_pool = SelfAttentivePooling(in_dim=2560)
        # 修改1：调整全连接层输入维度
        self.fc1 = nn.Linear(2560, self.emb_size) # 输入维度512→2560

        self.fc_audioset = nn.Linear(self.emb_size, 527)
        self.extract_embedding = extract_embedding

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.emb_size, num_class)

    def forward(self, x):
        x = x.unsqueeze(1) # x:(1,98,80)-->(1,1,98,80)
        x = x.permute([0, 3, 2, 1]) #x:(1,80,98,1)
        x = self.bn0(x)
        x = x.permute([0, 3, 2, 1]) # (1,1,98,80)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg') # (1,64,49,40)
        x = F.dropout(x, p=0.2, training=self.training) # (1,64,49,40)

        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg') # (1,128,24,40)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg') # (1,256,12,10)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg') # (1,512,6,5)
        x = F.dropout(x, p=0.2, training=self.training)

        # 修改3：维度调整流程
        x = x.permute(0, 1, 3, 2)  # 交换最后两个维度 → (B,512,5,6)
        x = x.reshape(x.size(0), x.size(1) * x.size(2), x.size(3))  # (B,2560,6)

        # 应用注意力池化
        x = self.attentive_pool(x)  # (B,2560)

        # 删除原有池化操作
        # x = x.mean(dim=3) # (1,512,6) dim=3是音频频率特征，dim=2是音频长度
        # x = x.max(dim=2)[0] + x.mean(dim=2) #(1,512)

        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc1(x)) #(1,512)

        if self.extract_embedding:
            output = F.dropout(x, p=0.5, training=self.training)
        else:
            output = F.sigmoid(self.fc_audioset(x))

        x = self.dropout(output)
        logits = self.fc(x)

        return logits

# class PANNS_CNN6(nn.Module):
#     """
#     原来版本
#     The CNN14(14-layer CNNs) mainly consist of 4 convolutional blocks while each convolutional
#     block consists of 1 convolutional layers with a kernel size of 5 × 5.
#
#     Reference:
#         PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition
#         https://arxiv.org/pdf/1912.10211.pdf
#     """
#     emb_size = 512
#
#     def __init__(self, num_class, input_size, dropout=0.1, extract_embedding: bool = True):
#
#         super(PANNS_CNN6, self).__init__()
#         self.bn0 = nn.BatchNorm2d(input_size)
#         self.conv_block1 = ConvBlock5x5(in_channels=1, out_channels=64)
#         self.conv_block2 = ConvBlock5x5(in_channels=64, out_channels=128)
#         self.conv_block3 = ConvBlock5x5(in_channels=128, out_channels=256)
#         self.conv_block4 = ConvBlock5x5(in_channels=256, out_channels=512)
#
#         self.fc1 = nn.Linear(512, self.emb_size)
#         self.fc_audioset = nn.Linear(self.emb_size, 527)
#         self.extract_embedding = extract_embedding
#
#         self.dropout = nn.Dropout(dropout)
#         self.fc = nn.Linear(self.emb_size, num_class)
#
#     def forward(self, x):
#         x = x.unsqueeze(1) # x:(1,98,80)-->(1,1,98,80)
#         x = x.permute([0, 3, 2, 1]) #x:(1,80,98,1)
#         x = self.bn0(x)
#         x = x.permute([0, 3, 2, 1]) # (1,1,98,80)
#
#         x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg') # (1,64,49,40)
#         x = F.dropout(x, p=0.2, training=self.training) # (1,64,49,40)
#
#         x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg') # (1,128,24,40)
#         x = F.dropout(x, p=0.2, training=self.training)
#
#         x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg') # (1,256,12,10)
#         x = F.dropout(x, p=0.2, training=self.training)
#
#         x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg') # (1,512,6,5)
#         x = F.dropout(x, p=0.2, training=self.training)
#
#         x = x.mean(dim=3) # (1,512,6) dim=3是音频频率特征，dim=2是音频长度
#         x = x.max(dim=2)[0] + x.mean(dim=2) #(1,512)
#
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = F.relu(self.fc1(x)) #(1,512)
#
#         if self.extract_embedding:
#             output = F.dropout(x, p=0.5, training=self.training)
#         else:
#             output = F.sigmoid(self.fc_audioset(x))
#
#         x = self.dropout(output)
#         logits = self.fc(x)
#
#         return logits


class PANNS_CNN10(nn.Module):
    """
    The CNN10(14-layer CNNs) mainly consist of 4 convolutional blocks while each convolutional
    block consists of 2 convolutional layers with a kernel size of 3 × 3.

    Reference:
        PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition
        https://arxiv.org/pdf/1912.10211.pdf
    """
    emb_size = 512

    def __init__(self, num_class, input_size, dropout=0.1, extract_embedding: bool = True):

        super(PANNS_CNN10, self).__init__()
        self.bn0 = nn.BatchNorm2d(input_size)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, self.emb_size)
        self.fc_audioset = nn.Linear(self.emb_size, 527)
        self.extract_embedding = extract_embedding

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.emb_size, num_class)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.permute([0, 3, 2, 1])
        x = self.bn0(x)
        x = x.permute([0, 3, 2, 1])

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = x.mean(dim=3)
        x = x.max(dim=2)[0] + x.mean(dim=2)

        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc1(x))

        if self.extract_embedding:
            output = F.dropout(x, p=0.5, training=self.training)
        else:
            output = F.sigmoid(self.fc_audioset(x))

        x = self.dropout(output)
        logits = self.fc(x)

        return logits


class PANNS_CNN14(nn.Module):
    """
    The CNN14(14-layer CNNs) mainly consist of 6 convolutional blocks while each convolutional
    block consists of 2 convolutional layers with a kernel size of 3 × 3.

    Reference:
        PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition
        https://arxiv.org/pdf/1912.10211.pdf
    """
    emb_size = 2048

    def __init__(self, num_class, input_size, dropout=0.1, extract_embedding: bool = True):

        super(PANNS_CNN14, self).__init__()
        self.bn0 = nn.BatchNorm2d(input_size)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, self.emb_size)
        self.fc_audioset = nn.Linear(self.emb_size, 527)
        self.extract_embedding = extract_embedding

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.emb_size, num_class)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.permute([0, 3, 2, 1])
        x = self.bn0(x)
        x = x.permute([0, 3, 2, 1])

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = x.mean(dim=3)
        x = x.max(dim=2)[0] + x.mean(dim=2)

        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc1(x))

        if self.extract_embedding:
            output = F.dropout(x, p=0.5, training=self.training)
        else:
            output = F.sigmoid(self.fc_audioset(x))

        x = self.dropout(output)
        logits = self.fc(x)

        return logits
