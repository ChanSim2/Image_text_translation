"""
Copyright (c) 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch.nn as nn
import torch
import torch.nn.functional as F

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention


class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.stages = {'Trans': opt.Transformation, 'Feat': opt.FeatureExtraction,
                       'Seq': opt.SequenceModeling, 'Pred': opt.Prediction}

        """ Transformation """
        if opt.Transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW), I_channel_num=opt.input_channel)
        else:
            print('No Transformation module specified')

        """ FeatureExtraction """
        if opt.FeatureExtraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        if opt.SequenceModeling == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
                BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
            self.SequenceModeling_output = opt.hidden_size
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if opt.Prediction == 'CTC':
            self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)
        elif opt.Prediction == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class)
        else:
            raise Exception('Prediction is neither CTC or Attn')

    def forward(self, input, text, is_train=True):
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.opt.batch_max_length)

        return prediction


class CBR(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, norm='bn', negative_slope=0.01):
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

class DECBR(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, norm='bn', negative_slope=0.01):
        super().__init__()
        layers = []

        layers += [torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]

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
        y = F.leaky_relu(y, negative_slope=0.01)
        return y

class resBlock(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self._resblock = torch.nn.Sequential(
            CBR(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(in_channels)
        )
    
    def forward(self, x):
        y = self._resblock(x)
        y = torch.add(x, y)
        y = F.leaky_relu(y, negative_slope=0.1)
        return y


#Need consideration
class encoderNet(torch.nn.Module):
    def __init__(self, in_channels, feat=False, chanNum=32):
        super().__init__()
        
        self._featureOut = feat

        self._layer0 = torch.nn.Sequential(
            CBR(in_channels=in_channels, out_channels=chanNum, kernel_size=3, stride=1, padding=1, norm=None),
            CBR(in_channels=chanNum, out_channels=chanNum, kernel_size=3, stride=1, padding=1),
            CBR(in_channels=chanNum, out_channels=chanNum, kernel_size=3, stride=1, padding=1)
        )

        self._pool1 = CBR(in_channels=chanNum, out_channels=2*chanNum, kernel_size=4, stride=2, padding=1, norm=None)
        self._layer1 = torch.nn.Sequential(
            CBR(in_channels=2*chanNum, out_channels=2*chanNum, kernel_size=3, stride=1, padding=1),
            CBR(in_channels=2*chanNum, out_channels=2*chanNum, kernel_size=3, stride=1, padding=1)
        )

        self._pool2 = CBR(in_channels=2*chanNum, out_channels=4*chanNum, kernel_size=4, stride=2, padding=1, norm=None)
        self._layer2 = torch.nn.Sequential(
            CBR(in_channels=4*chanNum, out_channels=4*chanNum, kernel_size=3, stride=1, padding=1),
            CBR(in_channels=4*chanNum, out_channels=4*chanNum, kernel_size=3, stride=1, padding=1)
        )

        self._pool3 = CBR(in_channels=4*chanNum, out_channels=8*chanNum, kernel_size=4, stride=2, padding=1, norm=None)
        self._layer3 = torch.nn.Sequential(
            CBR(in_channels=8*chanNum, out_channels=8*chanNum, kernel_size=3, stride=1, padding=1),
            CBR(in_channels=8*chanNum, out_channels=8*chanNum, kernel_size=3, stride=1, padding=1)
        )

        
    def forward(self, x):

        x0 = self._layer0(x)

        x1 = self._pool1(x0)
        x1 = self._layer1(x1)

        x2 = self._pool2(x1)
        x2 = self._layer2(x2)
        
        x3 = self._pool3(x2)
        x3 = self._layer3(x3)

        if self._featureOut:
            y = [x3, x2, x1, x0]
        else:
            y = x3

        return y

class decoderNet(torch.nn.Module):
    def __init__(self, out_channels, feat=False, chanNum=32):
        super().__init__()

        self._featIn = feat
        if feat:
            fn = 2
        else:
            fn = 1

        self._layer3 = torch.nn.Sequential(
            CBR(in_channels=8*chanNum, out_channels=8*chanNum, kernel_size=3, stride=1, padding=1),
            CBR(in_channels=8*chanNum, out_channels=8*chanNum, kernel_size=3, stride=1, padding=1)
        )
        self._deconv3 = DECBR(in_channels=8*chanNum, out_channels=4*chanNum, kernel_size=4, stride=2, padding=1,
                        norm=None)

        #skip connection connected
        self._layer2 = torch.nn.Sequential(
            CBR(in_channels=4*fn*chanNum, out_channels=4*chanNum, kernel_size=3, stride=1, padding=1),
            CBR(in_channels=4*chanNum, out_channels=4*chanNum, kernel_size=3, stride=1, padding=1)
        )
        self._deconv2 = DECBR(in_channels=4*chanNum, out_channels=2*chanNum, kernel_size=4, stride=2, padding=1,
                        norm=None)

        #skip connection connected
        self._layer1 = torch.nn.Sequential(
            CBR(in_channels=2*fn*chanNum, out_channels=2*chanNum, kernel_size=3, stride=1, padding=1),
            CBR(in_channels=2*chanNum, out_channels=2*chanNum, kernel_size=3, stride=1, padding=1)
        )
        self._deconv1 = DECBR(in_channels=2*chanNum, out_channels=1*chanNum, kernel_size=4, stride=2, padding=1,
                        norm=None)

        #skip connection connected
        self._layer0 = torch.nn.Sequential(
            CBR(in_channels=1*fn*chanNum, out_channels=1*chanNum, kernel_size=3, stride=1, padding=1),
            CBR(in_channels=1*chanNum, out_channels=1*chanNum, kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(in_channels=1*chanNum, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        )
        
        
    def forward(self, x):
        if self._featIn:
            x, features = x[0], x[1:]

        x = self._layer3(x)
        x = self._deconv3(x)
        
        if self._featIn:
            x = torch.cat((x, features[0]), dim=1)

        x = self._layer2(x)
        x = self._deconv2(x)
        
        if self._featIn:
            x = torch.cat((x, features[1]), dim=1)

        x = self._layer1(x)
        x = self._deconv1(x)
        
        if self._featIn:
            x = torch.cat((x, features[2]), dim=1)

        x = self._layer0(x)

        return torch.tanh(x)


class discriminator(torch.nn.Module):
    def __init__(self, in_channels, out_channels, channels=32):
        super().__init__()

        chanNum = channels

        self._l1 = CBR(in_channels=in_channels, out_channels=1*chanNum, kernel_size=4, stride=2, padding=2, bias=False, norm=None, negative_slope=0.1)
        self._l2 = resBlock(1*chanNum)
        self._l3 = CBR(in_channels=1*chanNum, out_channels=2*chanNum, kernel_size=4, stride=2, padding=2, bias=False, negative_slope=0.1)
        self._l4 = resBlock(2*chanNum)
        self._l5 = CBR(in_channels=2*chanNum, out_channels=4*chanNum, kernel_size=4, stride=2, padding=2, bias=False, negative_slope=0.1)
        self._l6 = resBlock(4*chanNum)
        self._l7 = CBR(in_channels=4*chanNum, out_channels=8*chanNum, kernel_size=4, stride=2, padding=2, bias=False, negative_slope=0.1)
        self._l8 = torch.nn.Conv2d(in_channels=8*chanNum, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):

        x = self._l1(x)
        x = self._l2(x)
        x = self._l3(x)
        x = self._l4(x)
        x = self._l5(x)
        x = self._l6(x)
        x = self._l7(x)
        x = self._l8(x)
        return x #BCEWithLogitLoss


class Unet(torch.nn.Module):
    def __init__(self, in_channels, chanNum=32):
        super().__init__()

        self._encoder = encoderNet(in_channels=in_channels, chanNum=chanNum, feat=True)
        self._bottleneck = torch.nn.Sequential(
            bottleneck(in_channels=chanNum*8),
            bottleneck(in_channels=chanNum*8),
            bottleneck(in_channels=chanNum*8),
            bottleneck(in_channels=chanNum*8)
        )
        self._decoder = decoderNet(out_channels=3, chanNum=chanNum, feat=True)

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        h_pad, w_pad = to_multiple(h, w, 8)
        x = F.pad(x, (0, w_pad, 0, h_pad), 'replicate')

        en_out = self._encoder(x)
        en_out[0] = self._bottleneck(en_out[0])
        y = self._decoder(en_out)

        return y[:, :, 0:h, 0:w]

def to_multiple(h, w, num):
    h_diff = h % num
    w_diff = w % num

    if h_diff != 0:
        h_pad = num - h_diff
    else:
        h_pad = 0
    if w_diff != 0:
        w_pad = num - w_diff
    else:
        w_pad = 0

    return h_pad, w_pad