"""
Multimodal Resnet-ish Deep Conv Net proposed in
'Robotic Grasp Detection using Deep Convolutional Neural Networks'
by Sulabh Kumra and Christopher Kanan. 
We use this model for both classification and grasp predicastion.

Weights available @ https://download.pytorch.org/models/resnet50-11ad3fa6.pth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride 
        
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class BottleneckT(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, upsample=None, stride=1):
        super().__init__()
        
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.ConvTranspose2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride 

        # xavier initialization of weights
        nn.init.xavier_uniform_(self.conv1.weight)
        #nn.init.xavier_uniform_(self.bn1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        #nn.init.xavier_uniform_(self.bn2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        #nn.init.xavier_uniform_(self.bn3.weight)
        if self.upsample is not None:
            for m in self.upsample:
                nn.init.xavier_uniform_(m.weight)
                break
        
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out

class BasicBlockT(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, upsample=None, stride=1):
        super().__init__()
       
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.upsample = upsample
        self.stride = stride

        # xavier initialization of weights
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.bn1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.bn2.weight)
        if self.upsample is not None:
            for m in self.upsample:
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
        
class ResNet(nn.Module):
    def __init__(self, block, layer_list, num_channels):
        super().__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(num_channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, layer_list[0], planes=64)
        self.layer2 = self._make_layer(block, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(block, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(block, layer_list[3], planes=512, stride=2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
        
    def _make_layer(self, block, blocks, planes, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.in_channels, planes, downsample=downsample, stride=stride
            )
        )
        self.in_channels = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_channels, planes
                )
            )

        return nn.Sequential(*layers)
        
def resnet50(channels=3, weights_path="/Users/grok0n/Workbench/consens/consens-lab/models/resnet50_weights.pth"):
    model = ResNet(block=Bottleneck, layer_list=[3,4,6,3], num_channels=channels)
    weights = torch.load(weights_path)
    # Remove saved weights of last fully connected layer of ResNet50
    weights.popitem()
    weights.popitem()
    model.load_state_dict(weights)

    return model

class ResNetT(nn.Module):
    def __init__(self, block, layer_list, num_channels):
        super().__init__()
        self.in_channels = 64
        
        self.layer1 = self._make_layer(block, layer_list[3], planes=512, stride=2)
        self.layer2 = self._make_layer(block, layer_list[2], planes=256, stride=2)
        self.layer3 = self._make_layer(block, layer_list[1], planes=128, stride=2)
        self.layer4 = self._make_layer(block, layer_list[0], planes=64)

        self.conv1 = nn.ConvTranspose2d(self.in_channels, num_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # xavier initialization of weights
        nn.init.xavier_uniform_(self.conv1.weight)
        #nn.init.xavier_uniform_(self.bn1.weight)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
        
    def _make_layer(self, block, blocks, planes, stride=1):
        upsample = None
        if stride != 1 or self.in_channels != planes * block.expansion:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(planes * block.expansion, self.in_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.in_channels),
            )

        layers = []
        in_channels = self.in_channels
        self.in_channels = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(
                block(
                    planes, self.in_channels
                )
            )
        layers.append(
            block(
                planes, in_channels, upsample=upsample, stride=stride
            )
        )
        return nn.Sequential(*layers)

def resnet50(channels=3, weights_path="/Users/grok0n/Workbench/consens/consens-lab/models/resnet50_weights.pth"):
    model = ResNet(block=Bottleneck, layer_list=[3,4,6,3], num_channels=channels)
    weights = torch.load(weights_path)
    # Remove saved weights of last fully connected layer of ResNet50
    weights.popitem()
    weights.popitem()
    model.load_state_dict(weights)

    return model

def resnet50T(channels=3):
    model = ResNetT(block=BottleneckT, layer_list=[3,4,6,3], num_channels=channels)
    return model

class DeepConvNetKumra(nn.Module):
    def __init__(self, dropout_p=0.2):
        super().__init__()
        self.dropout_p = dropout_p
        self.rgb_features = resnet50()
        self.d_features = resnet50()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.features = nn.Sequential(
			nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
			nn.Dropout(p=self.dropout_p),
			nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
			nn.Dropout(p=self.dropout_p),
            nn.Linear(512, 5),
            nn.Tanh()
        )

        # freeze resnet50 weights
        for param in self.rgb_features.parameters():
            param.requires_grad = False
        for param in self.d_features.parameters():
            param.requires_grad = False

        # xavier initialization of weights
        for m in self.features.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        rgb = x[:, :3, :, :]
        d = torch.unsqueeze(x[:, 3, :, :], dim=1)
        d = torch.cat((d, d, d), dim=1)

        rgb = F.normalize(self.rgb_features(rgb), p=2.0, dim=1)
        d = F.normalize(self.rgb_features(d), p=2.0, dim=1)
        x = torch.cat((rgb, d), dim=1)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        out = self.features(x)

        return out

    # Unfreeze pretrained layers
    def unfreeze_depth_backbone(self):
        for param in self.rgb_features.parameters():
            param.requires_grad = True

        for param in self.d_features.parameters():
            param.requires_grad = True

class DeepConvNetKumraMap(nn.Module):
    def __init__(self, dropout_p=0.2):
        super().__init__()
        self.dropout_p = dropout_p
        self.rgb_features = resnet50()
        self.d_features = resnet50()
        self.conv = nn.Conv2d(in_channels=4096, out_channels=2048, kernel_size=1)
        self.map_features = resnet50T()

        # freeze resnet50 weights
        for param in self.rgb_features.parameters():
            param.requires_grad = False
        for param in self.d_features.parameters():
            param.requires_grad = False

    def forward(self, x):
        rgb = x[:, :3, :, :]
        d = torch.unsqueeze(x[:, 3, :, :], dim=1)
        d = torch.cat((d, d, d), dim=1)

        rgb = F.normalize(self.rgb_features(rgb), p=2.0, dim=1)
        d = F.normalize(self.rgb_features(d), p=2.0, dim=1)
        x = torch.cat((rgb, d), dim=1)
        
        #x = torch.flatten(x, 1)
        x = self.conv(x)

        out = self.map_features(x)

        return out

    # Unfreeze pretrained layers
    def unfreeze_depth_backbone(self):
        for param in self.rgb_features.parameters():
            param.requires_grad = True

        for param in self.d_features.parameters():
            param.requires_grad = True

x = torch.randn(1, 4, 224, 224)
