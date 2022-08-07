import torch
from torch import nn

class NAFNetBlock(torch.nn.Module):

    def __init__(self, input_channels, num_filters):
        super().__init__()
        n_filters = num_filters

        self.conv1 = nn.Conv2d(input_channels, n_filters, kernel_size=1, padding=0, stride=1)
        self.deconv = nn.Conv2d(n_filters, n_filters * n_filters, kernel_size=3,\
                                         padding=1, groups=n_filters) # Depthwise convolution
        
        self.conv2 = nn.Conv2d((n_filters*n_filters)//2, input_channels, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(input_channels, n_filters, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(n_filters//2, input_channels, kernel_size=1, stride=1)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=( n_filters*n_filters)//2, out_channels=(n_filters*n_filters)//2,\
                                    kernel_size=1, stride=1)
        )


    def forward(self, input_data):

        lnorm1_out = nn.functional.layer_norm(input_data, input_data.size())
        conv1_out = self.conv1(lnorm1_out)
        deconved_out = self.deconv(conv1_out)
        
        # Simple gate function
        p1, p2 = torch.tensor_split(deconved_out, 2, dim=1)
        sg1_out = p1*p2
        sca_out = sg1_out * self.sca(sg1_out)

        conv2_out = self.conv2(sca_out)
        skip1_out = torch.add(input_data, conv2_out)

        lnorm2_out = nn.functional.layer_norm(skip1_out, skip1_out.size())
        conv3_out = self.conv3(lnorm2_out)
        
        # Simple gate function
        p3, p4 = torch.tensor_split(conv3_out, 2, dim=1)
        sg2_out = p3*p4
        
        conv4_out = self.conv4(sg2_out)
        out = torch.add(skip1_out, conv4_out)

        return out