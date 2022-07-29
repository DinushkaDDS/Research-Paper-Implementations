import torch
from torch import nn

class BaselineBlock(torch.nn.Module):

    def __init__(self, input_channels):
        super().__init__()
        n_filters = input_channels

        self.conv1 = nn.Conv2d(input_channels, n_filters, kernel_size=1)
        self.deconv = nn.Conv2d(n_filters, n_filters * n_filters, kernel_size=3, groups=n_filters) # Depthwise convolution
        
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(n_filters, n_filters, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(n_filters, n_filters, kernel_size=1, stride=1)

        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=1, stride=1),
            nn.Sigmoid()
        )


    def forward(self, input_data):

        lnorm1_out = nn.functional.layer_norm(input_data, input_data.size())
        conv1_out = self.conv1(lnorm1_out)
        deconved_out = self.deconv(conv1_out)
        relu1_out = nn.functional.gelu(deconved_out)

        ca_out = relu1_out * self.ca(relu1_out)
        
        conv2_out = self.conv2(ca_out)
        
        skip1_out = torch.add(input_data, conv2_out)

        lnorm2_out = nn.functional.layer_norm(skip1_out, skip1_out.size())
        conv3_out = self.conv3(lnorm2_out)
        relu2_out = nn.functional.gelu(conv3_out)
        conv4_out = self.conv4(relu2_out)
        
        out = torch.add(skip1_out, conv4_out)

        return out