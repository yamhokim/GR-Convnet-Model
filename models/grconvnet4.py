import torch
from torch import nn
import torch.nn.functional as F

# Code taken from https://github.com/skumra/robotic-grasping

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x_in):
        x = self.bn1(self.conv1(x_in))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        return x + x_in


class GRConvNet4(nn.Module):

    def __init__(self, input_channels=4, output_channels=1, channel_size=32, dropout=False, prob=0.0, clip=False):
        super(GRConvNet4, self).__init__()
        self.clip = clip
        self.conv1 = nn.Conv2d(input_channels, channel_size, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(channel_size)

        self.conv2 = nn.Conv2d(channel_size, channel_size // 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_size // 2)

        self.conv3 = nn.Conv2d(channel_size // 2, channel_size // 4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channel_size // 4)

        self.res1 = ResidualBlock(channel_size // 4, channel_size // 4)
        self.res2 = ResidualBlock(channel_size // 4, channel_size // 4)
        self.res3 = ResidualBlock(channel_size // 4, channel_size // 4)
        self.res4 = ResidualBlock(channel_size // 4, channel_size // 4)
        self.res5 = ResidualBlock(channel_size // 4, channel_size // 4)

        self.conv4 = nn.ConvTranspose2d(channel_size // 4, channel_size // 2, kernel_size=4, stride=2, padding=1,
                                        output_padding=1)
        self.bn4 = nn.BatchNorm2d(channel_size // 2)

        self.conv5 = nn.ConvTranspose2d(channel_size // 2, channel_size, kernel_size=4, stride=2, padding=2,
                                        output_padding=1)
        self.bn5 = nn.BatchNorm2d(channel_size)

        self.conv6 = nn.ConvTranspose2d(channel_size, channel_size, kernel_size=9, stride=1, padding=4)

        self.grasp_outputs = nn.Conv2d(in_channels=channel_size, out_channels=output_channels*4, kernel_size=2)
        self.confidence_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)

        self.dropout = dropout
        self.dropout_conf = nn.Dropout(p=prob)
        self.dropout_grasp = nn.Dropout(p=prob)


        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x_in):
        x = F.relu(self.bn1(self.conv1(x_in)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)

        if self.dropout:
            grasp_output = self.grasp_outputs(self.dropout_grasp(x))
            confidence_output = self.confidence_output(self.dropout_conf(x))
        else:
            grasp_output = self.grasp_outputs(x)
            confidence_output = self.confidence_output(x)

        confidence_output = F.sigmoid(confidence_output)      
        grasp_output = F.tanh(grasp_output)  

        output = torch.cat([confidence_output, grasp_output], dim=1)
        if self.clip:
            output = output.clip(-1, 1)
        return output


if __name__ == "__main__":
    model = GRConvNet4()
    random_input = torch.randn(8, 4, 224, 224)
    output = model(random_input)
    print(random_input.shape)
    print(output.shape)
