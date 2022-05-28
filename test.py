import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from utils import CTCLabelConverter, AttnLabelConverter
from model import Unet

from PIL import Image
from model import Model
import model_c

from text_gen import text_generation
from render_standard_text import make_standard_text
from translate import get_translate

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile
torch.cuda.set_device(1)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
from craft import CRAFT
from collections import OrderedDict
from skimage.transform import resize

# vidcap = cv2.VideoCapture('./video/videosample.mp4')
# fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
# count = 0

# while True:
#     success,capt = vidcap.read()
#     fgmask = fgbg.apply(capt)
#     if (fgmask is None) == False:
#         if vidcap.get(1) % 2 == 0:
#             cv2.imwrite("./data/capt%d.jpg" % count, capt)
#             count += 1
#     else:
#         break
# vidcap.release()


# pathOut = './result/result_video.mp4'
# fps = 15
# frame_array = []
# for i in range(0, len(os.listdir('./data/'))) : 
#     pathIn= './data/capt%d.jpg'% i
#     # if (idx % 2 == 0) | (idx % 5 == 0) :
#     #     continue
#     frame_img = cv2.imread(pathIn)
#     height, width, layers = frame_img.shape
#     size = (width,height)
#     frame_array.append(frame_img)
# out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
# for i in range(len(frame_array)):
#     # writing to a image array
#     out.write(frame_array[i])
# out.release()


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description="CRAFT Text Detection")
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='./data/', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')
args = parser.parse_args()

parser_re = argparse.ArgumentParser()
parser_re.add_argument('--image_folder', #required=True, 
                        help='path to image_folder which contains text images', default='demo_image/')
parser_re.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser_re.add_argument('--batch_size', type=int, default=1, help='input batch size')
parser_re.add_argument('--saved_model', #required=True, 
                    help="path to saved_model to evaluation", default='TPS-ResNet-BiLSTM-Attn.pth')
""" Data processing """
parser_re.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
parser_re.add_argument('--imgH', type=int, default=32, help='the height of the input image')
parser_re.add_argument('--imgW', type=int, default=100, help='the width of the input image')
parser_re.add_argument('--rgb', action='store_true', help='use rgb input')
parser_re.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
parser_re.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
parser_re.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
""" Model Architecture """
parser_re.add_argument('--Transformation', type=str, #required=True, 
                    help='Transformation stage. None|TPS', default='TPS')
parser_re.add_argument('--FeatureExtraction', type=str, #required=True, 
                    help='FeatureExtraction stage. VGG|RCNN|ResNet', default='ResNet')
parser_re.add_argument('--SequenceModeling', type=str, #required=True, 
                    help='SequenceModeling stage. None|BiLSTM', default='BiLSTM')
parser_re.add_argument('--Prediction', type=str, #required=True, 
                    help='Prediction stage. CTC|Attn', default='Attn')
parser_re.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
parser_re.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
parser_re.add_argument('--output_channel', type=int, default=512,
                    help='the number of output channel of Feature extractor')
parser_re.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
opt = parser_re.parse_args()


"""For test images in a folder"""
image_list, _, _ = file_utils.get_files(args.test_folder)

result_folder = './demo_image/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)


def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    #resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    #preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))
    if cuda:
        x = x.cuda()

    #forward pass
    with torch.no_grad():
        y, feature = net(x)

    #make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    #refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy() 

    t0 = time.time() - t0
    t1 = time.time()

    #Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    #coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    #render results(optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text



if __name__ == '__main__':
    #load net
    net = CRAFT()

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net, device_ids = [1, 0])
        cudnn.benchmark = False

    net.eval()

    #LinkRefiner
    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True

    #t = time.time()

    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt).to(device)
    #model = torch.nn.DataParallel(model).to(device)
    state_dict = torch.load('TPS-ResNet-BiLSTM-Attn.pth', map_location='cpu')
    n_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] #'module.' 제거
        n_state_dict[name] = v

    model.load_state_dict(n_state_dict)
    model.eval()

    inp_model = Unet(in_channels=3, chanNum=48).to(device)
    txt_model = Unet(in_channels=6, chanNum=48).to(device)
    fus_model = Unet(in_channels=6).to(device)

    models_load = torch.load('./module/step-1.model', map_location='cpu')
    inp_model.load_state_dict(models_load['inp'])
    txt_model.load_state_dict(models_load['text_gen'])
    fus_model.load_state_dict(models_load['fus'])

    inp_model.eval()
    txt_model.eval()
    fus_model.eval()
    
    #load data
    for k, image_path in enumerate(image_list):
        #print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)

        #save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = result_folder + "/res_" + filename + '_mask.jpg'
        cv2.imwrite(mask_file, score_text)

        inpaint_img, inpaint_point, pred = file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder, model=model, converter=converter, opt=opt)

        with torch.no_grad():
            image = image[...,::-1]
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            for i in range(len(inpaint_img)):
                img = inpaint_img[i]
                img_shape = img.shape
                img = file_utils.img_transform(img)

                
                bg = inp_model(img) #inpainting된 background
                bg_dbg = file_utils.img_inv_transform(bg)
                cv2.imwrite("inpainting/inpaint_{}.jpg".format(i), bg_dbg * 255.)

                txt = get_translate(pred[i])
                i_t = text_generation(txt, img.shape[2:])
                i_t = file_utils.img_transform(i_t)
                o_t = txt_model(torch.cat((img, i_t), dim=1))
                t_dbg = file_utils.img_inv_transform(o_t)

                

                fus = fus_model(torch.cat((bg, o_t), dim=1))
                fus = file_utils.img_inv_transform(fus) ##### chagned
                cv2.imwrite("inpainting/fusion_{}.jpg".format(i), fus * 255.)
                
                # color_o = color_net(img)
                # font_o = font_net(img)
                # size_o, shear_o = size_net(img)

                point = inpaint_point[i]
                fus_resize = file_utils.img_centering(fus, (point[3]-point[2], point[1]-point[0]))
                image[point[2]:point[3], point[0]:point[1]] = fus_resize * 255.

            #cv2.imwrite('./result_capture/capt%d.jpg' % k, image)
            cv2.imwrite('./result_capture/' + filename + 'fusion.jpg', image)

    # pathOut = './result/result_video.mp4'
    # fps = 15
    # frame_array = []
    # for i in range(0, len(os.listdir('./result_capture/'))) : 
    #     pathIn= './result_capture/capt%d.jpg'% i
    #     # if (idx % 2 == 0) | (idx % 5 == 0) :
    #     #     continue
    #     frame_img = cv2.imread(pathIn)
    #     height, width, layers = frame_img.shape
    #     size = (width,height)
    #     frame_array.append(frame_img)
    # out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    # for i in range(len(frame_array)):
    #     # writing to a image array
    #     out.write(frame_array[i])
    # out.release()

    #print("elapsed time : {}s".format(time.time() - t))


