# Font classifier model V1

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.utils as util

def visualize(x):
    y = x[:,:,:,:]
    y = y.clone().detach().cpu()
    y = util.make_grid(y, normalize=True, nrow=3)
    return y

class CBR(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, norm='bn', negative_slope=0.2):
        super().__init__()
        layers = []

        layers += [torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
        
        if not norm is None:
            if norm == 'bn':
                layers += [torch.nn.BatchNorm2d(out_channels)]
            elif norm == 'in':
                layers += [torch.nn.InstanceNorm2d(out_channels)]
        
        layers += [torch.nn.LeakyReLU(negative_slope=negative_slope)]

        self._net = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        return self._net(x)


class bottleneck(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self._bottleneck = torch.nn.Sequential(
            CBR(in_channels=in_channels, out_channels=in_channels//4, kernel_size=1, stride=1, padding=0, norm=None),
            CBR(in_channels=in_channels//4, out_channels=in_channels//4, kernel_size=3, stride=1, padding=1, norm=None),
            torch.nn.Conv2d(in_channels=in_channels//4, out_channels=in_channels, kernel_size=1, stride=1, padding=0),
            torch.nn.BatchNorm2d(in_channels)
        )
    
    def forward(self, x):
        y = self._bottleneck(x)
        y = torch.add(x, y)
        y = F.relu(y)
        return y


class resBlock(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self._resblock = torch.nn.Sequential(
            CBR(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0),
            torch.nn.BatchNorm2d(in_channels)
        )
    
    def forward(self, x):
        y = self._resblock(x)
        y = torch.add(x, y)
        y = F.relu(y)
        return y


class colorClassifier(torch.nn.Module):
    def __init__(self, in_channels, chanNum=32):
        super().__init__()
        
        self._l1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=chanNum, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            resBlock(in_channels=chanNum),
            resBlock(in_channels=chanNum),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self._l2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=chanNum, out_channels=2*chanNum, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            resBlock(in_channels=2*chanNum),
            resBlock(in_channels=2*chanNum),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        '''
        self._out3 = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=24320, out_features=chanNum*8),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=chanNum*8, out_features=chanNum*4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=chanNum*4, out_features=1),
        )
        self._out4 = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=24320, out_features=chanNum*8),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=chanNum*8, out_features=chanNum*4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=chanNum*4, out_features=2),
        )
        '''
        self._out5 = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=4608, out_features=chanNum*8),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=chanNum*8, out_features=chanNum*4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=chanNum*4, out_features=3),
            torch.nn.Tanh()
        )
        '''
        self._out6 = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=24320, out_features=chanNum*8),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=chanNum*8, out_features=chanNum*4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=chanNum*4, out_features=1),
            torch.nn.Sigmoid()
        )
        '''
    def forward(self, x):
        w = x.shape[3]
        wpad = (288 - w) // 2
        x = F.pad(x, (wpad, wpad), 'constant', 0)

        x = self._l1(x)
        x = self._l2(x)
        # Need change.

        font_color = self._out5(x)

        return font_color
        #[font, font_size, rotate, shear, font_color, font_underline, border_color]

class fontClassifier(torch.nn.Module):
    def __init__(self, in_channels, chanNum=32):
        super().__init__()
        
        self._l1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=chanNum, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            resBlock(in_channels=chanNum),
            resBlock(in_channels=chanNum),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self._l2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=chanNum, out_channels=2*chanNum, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            bottleneck(in_channels=2*chanNum),
            bottleneck(in_channels=2*chanNum),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self._fontOut = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=4608, out_features=chanNum*8),
            torch.nn.BatchNorm1d(chanNum*8),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=chanNum*8, out_features=chanNum*4),
            torch.nn.BatchNorm1d(chanNum*4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=chanNum*4, out_features=7),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        w = x.shape[3]
        wpad = (288 - w) // 2
        x = F.pad(x, (wpad, wpad), 'constant', 0)

        x = self._l1(x)
        x = self._l2(x)
        font = self._fontOut(x)

        return font


class sizeClassifier(torch.nn.Module):
    def __init__(self, in_channels, chanNum=32):
        super().__init__()
        
        self._l1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=2*chanNum, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            resBlock(in_channels=2*chanNum),
            resBlock(in_channels=2*chanNum),
        )
        self._l2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=2*chanNum, out_channels=4*chanNum, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            bottleneck(in_channels=4*chanNum),
            bottleneck(in_channels=4*chanNum),
            bottleneck(in_channels=4*chanNum),
            torch.nn.Conv2d(in_channels=4*chanNum, out_channels=chanNum//4, kernel_size=1, stride=1, padding=0)
        )
        self._fontSize = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=9216, out_features=chanNum*8), #9216
            torch.nn.BatchNorm1d(chanNum*8),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=chanNum*8, out_features=chanNum*4),
            torch.nn.BatchNorm1d(chanNum*4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=chanNum*4, out_features=1),
            torch.nn.ReLU()
        )
        self._shear = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=9216, out_features=chanNum*8), #9216
            torch.nn.BatchNorm1d(chanNum*8),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=chanNum*8, out_features=chanNum*4),
            torch.nn.BatchNorm1d(chanNum*4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=chanNum*4, out_features=2)
        )

    def forward(self, x):
        w = x.shape[3]
        wpad = (288 - w) // 2
        x = F.pad(x, (wpad, wpad), 'constant', 0)

        x = self._l1(x)
        x = self._l2(x)
        size = self._fontSize(x)
        shear = self._shear(x)

        return size, shear

class visual_net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self._l1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            resBlock(in_channels=3),
            resBlock(in_channels=3),
        )
        self._l2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=3, padding=1),
            torch.nn.ReLU(),
            bottleneck(in_channels=32),
            bottleneck(in_channels=32),
            torch.nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1, stride=1, padding=0)
        )
        self._fontOut = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=6336, out_features=1024), #9216
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=1024, out_features=512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=512, out_features=7),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        w = x.shape[3]
        wpad = (288 - w) // 2
        x = F.pad(x, (wpad, wpad), 'constant', 0)

        x = self._l1(x)
        y = visualize(x)
        x = self._l2(x)
        y = visualize(x)
        font = self._fontOut(x)

        return font
