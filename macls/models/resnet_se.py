import torch.nn as nn

from macls.models.pooling import AttentiveStatisticsPooling, TemporalAveragePooling, MQMHASP, GlobalMultiHeadAttentionPooling
from macls.models.pooling import SelfAttentivePooling, TemporalStatisticsPooling, LDEPooling, MultiHeadAttentionPooling


class SEBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * self.expansion, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResNetSE(nn.Module):
    def __init__(self, num_class, input_size, layers=[3, 4, 6, 3], num_filters=[32, 64, 128, 256], embd_dim=192,
                 pooling_type="SAP"):
        super(ResNetSE, self).__init__()
        self.inplanes = num_filters[0]
        self.emb_size = embd_dim
        self.conv1 = nn.Conv2d(1, num_filters[0], kernel_size=3, stride=(1, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(SEBottleneck, num_filters[0], layers[0])
        self.layer2 = self._make_layer(SEBottleneck, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(SEBottleneck, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(SEBottleneck, num_filters[3], layers[3], stride=(2, 2))

        cat_channels = num_filters[3] * SEBottleneck.expansion * (input_size // 8) # cat_channel:5120
        if pooling_type == "ASP":
            self.pooling = AttentiveStatisticsPooling(cat_channels, 128)
            self.bn2 = nn.BatchNorm1d(cat_channels * 2)
            self.linear = nn.Linear(cat_channels * 2, embd_dim)
            self.bn3 = nn.BatchNorm1d(embd_dim)
        elif pooling_type == "SAP":
            self.pooling = SelfAttentivePooling(cat_channels, 128)
            self.bn2 = nn.BatchNorm1d(cat_channels)
            self.linear = nn.Linear(cat_channels, embd_dim)
            self.bn3 = nn.BatchNorm1d(embd_dim)
        elif pooling_type == "TAP":
            self.pooling = TemporalAveragePooling()
            self.bn2 = nn.BatchNorm1d(cat_channels)
            self.linear = nn.Linear(cat_channels, embd_dim)
            self.bn3 = nn.BatchNorm1d(embd_dim)
        elif pooling_type == "TSP":
            self.pooling = TemporalStatisticsPooling()
            self.bn2 = nn.BatchNorm1d(cat_channels * 2)
            self.linear = nn.Linear(cat_channels * 2, embd_dim)
            self.bn3 = nn.BatchNorm1d(embd_dim)
        elif pooling_type =="MQMHASP":
            self.pooling = MQMHASP(in_dim=5120, num_q=1, stddev=True)
            self.bn2 = nn.BatchNorm1d(cat_channels * 2) # stddev=True时，池化输出维度为（1，10240）；stddev=False时，池化输出维度为（1，5120）
            self.linear = nn.Linear(cat_channels * 2, embd_dim)
            self.bn3 = nn.BatchNorm1d(embd_dim)
        elif pooling_type =="LDEPooling":
            self.pooling = LDEPooling(input_dim=5120, c_num=1)
            self.bn2 = nn.BatchNorm1d(cat_channels)
            self.linear = nn.Linear(cat_channels, embd_dim)
            self.bn3 = nn.BatchNorm1d(embd_dim)
        elif pooling_type == "MultiHeadAttention":
            self.pooling = MultiHeadAttentionPooling(input_dim=5120, stddev=False, num_head=4)
            self.bn2 = nn.BatchNorm1d(cat_channels) # stddev=True时，池化输出维度为（1，10240）；stddev=False时，池化输出维度为（1，5120）
            self.linear = nn.Linear(cat_channels, embd_dim)
            self.bn3 = nn.BatchNorm1d(embd_dim)
        elif pooling_type == "GlobalMultiHeadAttentionPooling":
            self.pooling = GlobalMultiHeadAttentionPooling(input_dim=5120, num_head=4, stddev=False)
            self.bn2 = nn.BatchNorm1d(20480)
            self.linear = nn.Linear(20480, embd_dim)
            self.bn3 = nn.BatchNorm1d(embd_dim)
        else:
            raise Exception(f'没有{pooling_type}池化层！')

        self.fc = nn.Linear(embd_dim, num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(2, 1) # x:(1,98,80) -->(1,80,98) 80:n_mels ,98：音频长度
        x = x.unsqueeze(1)  # (1,1,80,98)
        x = self.conv1(x) # (1,32,80,98)
        x = self.bn1(x) # (1,32,80,98)
        x = self.relu(x)

        x = self.layer1(x) # (1,64,80,98)
        x = self.layer2(x) # (1,128,40,49)
        x = self.layer3(x) # (1,256,20,25)
        x = self.layer4(x) # (1,512,10,13)

        x = x.reshape(x.shape[0], -1, x.shape[-1]) # (1,5120,13)

        x = self.pooling(x) # (1,5120)
        # x = x.reshape(x.shape[0], -1)
        x = self.bn2(x)
        x = self.linear(x)
        x = self.bn3(x) # (1,192)
        out = self.fc(x)
        return out
