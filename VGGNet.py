import torch
import torch.nn as nn
import math

class VGG(nn.Module):
    # 构造函数
    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__() # 等价于nn.Module.__init__(self)
        # 网络结构（仅包含卷积层和池化层，不包含分类器）
        self.features = features
        # 分类器结构
        self.classifier = nn.Sequential(
            # fc6
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            # fc7
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            # fc8
            nn.Linear(4096, num_classes),
        )
        # 初始化权重
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

# 生成网络每层的信息
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # 设定卷积层的输出数量
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
                # 通道数设为v
            in_channels = v
    return nn.Sequential(*layers) # 返回一个包含了网络结构的时序容器


cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

def vgg16(model_path, **kwargs):
    """VGG 16-layer model (configuration "D")
        load a model pre-trained on ImageNet
    Args:
        model_path: the model path
    """
    model = VGG(make_layers(cfg), **kwargs) # 首先确定模型结构
    model.load_state_dict(torch.load(model_path)) # 再加载pth文件（参数？）

    return model