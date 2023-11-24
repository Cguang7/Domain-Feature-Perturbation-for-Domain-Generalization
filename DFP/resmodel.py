import torch
import torch.nn as nn

grads = {'cuda0': [], 'cuda1': [], 'cuda2': [], 'cuda3': []}


def save_grad(name):
    def hook(grad):
        grads[name].append(grad)
    return hook


def ini_noise(x):
    keyname = str(x.device).replace(":", "")
    x.register_hook(save_grad(keyname))


def gen_noise(grad_eps, grad_n):
    grad_abs = torch.abs(grad_n)
    g_max = torch.norm(grad_abs,p=2)
    std = (grad_eps/g_max)*grad_abs

    device = grad_n.device
    size = grad_n.shape
    noise = torch.cuda.FloatTensor(size, device=device) if torch.cuda.is_available() else torch.FloatTensor(size)
    mean = 0.0

    torch.normal(mean, std, out=noise)

    return noise
    

class Backbone(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self):
        pass

    @property
    def out_features(self):
        """Output feature dimension."""
        if self.__dict__.get("_out_features") is None:
            return None
        return self._out_features


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.noiseset = None
        self.grad_eps = 1.0

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def set_blocknoise(self, noiseset='initial'):
        self.noiseset = noiseset

    def set_blockeps(self, grad_eps=1.0):
        self.grad_eps = grad_eps

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        if self.training and self.noiseset == 'initial':
            ini_noise(out)

        elif self.training and self.noiseset == 'addnoise':
            keyname = str(out.device).replace(":", "")
            grad_n = grads[keyname].pop()
            noise = gen_noise(self.grad_eps, grad_n)
            out = out+noise

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.training and self.noiseset == 'initial':
            ini_noise(out)

        elif self.training and self.noiseset == 'addnoise':
            keyname = str(out.device).replace(":", "")
            grad_n = grads[keyname].pop()
            noise = gen_noise(self.grad_eps, grad_n)
            out = out+noise

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(Backbone):

    def __init__(
            self,
            block,
            layers,
            **kwargs
    ):
        self.inplanes = 64
        super().__init__()

        self.noiseset = None
        self.grad_eps = 1.0
        
        # backbone network
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.n_outputs = 512

        self._init_params()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def set_gradeps(self, grad_eps=1.0):
        self.grad_eps = grad_eps

        for i in self.layer1:
            i.set_blockeps(grad_eps)

        for i in self.layer2:
            i.set_blockeps(grad_eps)

        for i in self.layer3:
            i.set_blockeps(grad_eps)

        for i in self.layer4:
            i.set_blockeps(grad_eps)

    def set_noise(self, noiseset='initial'):
        self.noiseset = noiseset

        for i in self.layer1:
            i.set_blocknoise(noiseset)

        for i in self.layer2:
            i.set_blocknoise(noiseset)

        for i in self.layer3:
            i.set_blocknoise(noiseset)

        for i in self.layer4:
            i.set_blocknoise(noiseset)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        if self.training and self.noiseset == 'initial':
            ini_noise(x)
        elif self.training and self.noiseset == 'addnoise':
            keyname = str(x.device).replace(":", "")
            grad_n = grads[keyname].pop()
            noise = gen_noise(self.grad_eps, grad_n)
            x = x+noise

        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward(self, x):
        f = self.featuremaps(x)
        v = self.global_avgpool(f)
        out = v.view(v.size(0), -1)

        return out


def init_pretrained_weights(model, model_url):
    pretrain_dict = torch.load(model_url)
    model_dict = model.state_dict()
    pretrain_dict.pop('fc.weight')
    pretrain_dict.pop('fc.bias')  # 筛除不加载的层结构
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}  # 更新当前网络的结构字典
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)


def dfpresnet18(pretrained=True, pre_dir="pretrained_weights/resnet18-5c106cde.pth"):
    model = ResNet(block=BasicBlock, layers=[2, 2, 2, 2])
    model_dir = pre_dir

    if pretrained:
        init_pretrained_weights(model, model_dir)

    return model

