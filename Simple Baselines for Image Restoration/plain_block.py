import torch
from torch import nn

class PlainBlock(torch.nn.Module):
    '''Note the dimensions are not tested to work'''
    
    def __init__(self, input_channels):
        super().__init__()
        n_filters = input_channels

        self.conv1 = nn.Conv2d(input_channels, n_filters, kernel_size=1)
        self.deconv = nn.Conv2d(n_filters, n_filters * n_filters, kernel_size=3, groups=n_filters) # Depthwise convolution
        
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(n_filters, n_filters, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(n_filters, n_filters, kernel_size=1, stride=1)
        
    def forward(self, input_data):

        conv1_out = self.conv1(input_data)
        deconved_out = self.deconv(conv1_out)
        relu1_out = nn.functional.relu(deconved_out)
        conv2_out = self.conv2(relu1_out)
        
        skip1_out = torch.add(input_data, conv2_out)

        conv3_out = self.conv3(skip1_out)
        relu2_out = nn.functional.relu(conv3_out)
        conv4_out = self.conv4(relu2_out)
        
        out = torch.add(skip1_out, conv4_out)

        return out