import os
import numpy as np
import cv2
import imgproc
from math import ceil
import argparse
import torch
import torch.utils.data
import torchvision.transforms as trfm
from dataset import RawDataset, AlignCollate
from skimage.transform import resize

import torch.nn.functional as F
from translate import get_translate
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--image_folder', #required=True, 
                        help='path to image_folder which contains text images', default='demo_image/')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--saved_model', #required=True, 
                    help="path to saved_model to evaluation", default='TPS-ResNet-BiLSTM-Attn.pth')
""" Data processing """
parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
parser.add_argument('--rgb', action='store_true', help='use rgb input')
parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz,.?!', help='character label')
parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
""" Model Architecture """
parser.add_argument('--Transformation', type=str, #required=True, 
                    help='Transformation stage. None|TPS', default='TPS')
parser.add_argument('--FeatureExtraction', type=str, #required=True, 
                    help='FeatureExtraction stage. VGG|RCNN|ResNet', default='ResNet')
parser.add_argument('--SequenceModeling', type=str, #required=True, 
                    help='SequenceModeling stage. None|BiLSTM', default='BiLSTM')
parser.add_argument('--Prediction', type=str, #required=True, 
                    help='Prediction stage. CTC|Attn', default='Attn')
parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
parser.add_argument('--output_channel', type=int, default=512,
                    help='the number of output channel of Feature extractor')
parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

#opt = parser.parse_args()

def img_transform(img, rescale=True):
    tf = trfm.ToTensor()
    img_new = tf(img)
    img_new = img_new.type(torch.FloatTensor).unsqueeze(dim=0)
    if rescale:
        img_new = (img_new.to(device) * 2.) - 1.
    return img_new

def img_inv_transform(img):
    out_img = img.squeeze(0).detach().to('cpu')
    out_img = out_img.numpy().transpose((1,2,0))
    out_img = ((out_img + 1.) * .5)
    return out_img

def img_centering(img, imgsize):
    h, w = imgsize
    i_h, i_w = img.shape[:2]

    h_diff = i_h - h
    w_diff = i_w - w

    out_img = img[h_diff//2 : i_h - ceil(h_diff/2), w_diff//2 : i_w - ceil(w_diff/2)]
    return out_img

def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls

def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    
    return img_files, mask_files, gt_files

def saveResult(img_file, img, boxes, model, converter, opt, dirname='./demo_image/', verticals=None, texts=None):
    img = np.array(img)
    img_copy = img.copy()

    # make result file list
    filename, file_ext = os.path.splitext(os.path.basename(img_file))

    # result directory
    res_file = dirname + "res_" + filename + '.txt'
    res_img_file = dirname + "res_" + filename + '.jpg'
    
    list_inpaint_img = []
    list_pred = []
    list_inpaint_point = []
    list_all = []

    tf = trfm.ToTensor()
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
#여기서부터
    with open(res_file, 'w') as f:
        for i, box in enumerate(boxes):
            poly = np.array(box).astype(np.int32).reshape((-1))
            strResult = ','.join([str(p) for p in poly]) + '\r\n'
            f.write(strResult)

            min_x = min(poly[0], poly[2], poly[4], poly[6])
            max_x = max(poly[0], poly[2], poly[4], poly[6])
            min_y = min(poly[1], poly[3], poly[5], poly[7])
            max_y = max(poly[1], poly[3], poly[5], poly[7])
            cut_img = img_copy[min_y:max_y, min_x:max_x]

            # temp_list = []
            # for j in range(0, 8):
            #     temp_list.append(poly[j])

            #AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
            #demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
            #demo_loader = torch.utils.data.DataLoader(
            #    demo_data, batch_size=opt.batch_size,
            #    shuffle=False,
            #    num_workers=int(opt.workers),
            #    collate_fn=AlignCollate_demo, pin_memory=True)

            with torch.no_grad():
                #for image_tensors, labels in demo_loader:
                #    batch_size = image_tensors.size(0)
                #    cut_img = image_tensors.to(device)
                batch_size = 1
                length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device) # 1= batch_size
                text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
                
                #image = imgproc.normalizeMeanVariance(cut_img)
                #image = torch.from_numpy(image).permute(2, 0, 1)
                
                # transform = trfm.Compose([
                #     trfm.ToTensor(),
                #     trfm.ConvertImageDtype(torch.float),
                #     trfm.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                # ])
                
                #image = cv2.cvtColor(cut_img, cv2.COLOR_BGR2RGB)
                
                #image = torch.tensor(cut_img)
                image = tf(cut_img)
                image = image.unsqueeze(dim=1).to(device)

                if 'CTC' in opt.Prediction:
                    preds = model(image, text_for_pred)

                    # Select max probabilty (greedy decoding) then decode index to character
                    preds_size = torch.IntTensor([preds.size(1)] * 1)
                    _, preds_index = preds.max(2)
                    # preds_index = preds_index.view(-1)
                    preds_str = converter.decode(preds_index, preds_size)

                else:
                    preds = model(image, text_for_pred, is_train=False)

                    # select max probabilty (greedy decoding) then decode index to character
                    _, preds_index = preds.max(2)
                    preds_str = converter.decode(preds_index, length_for_pred)

                preds_prob = F.softmax(preds, dim=2)
                preds_max_prob, _ = preds_prob.max(dim=2)
                for pred, pred_max_prob in zip(preds_str, preds_max_prob):
                    if 'Attn' in opt.Prediction:
                        pred_EOS = pred.find('[s]')
                        pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                        pred_max_prob = pred_max_prob[:pred_EOS]
                        pred_trans = get_translate(pred)

                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                print(f'{pred:25s}\t{pred_trans:25s}\t{confidence_score:0.4f}')
                
            res_cut_file = dirname + "res_" + filename + '_cut{}.jpg'.format(i)
            cv2.imwrite(res_cut_file, cut_img)
            if confidence_score >= 0.9:
                points = [min_y, max_y, min_x, max_x]
                list_inpaint_point.append(points)
                max_x += round(min((max_x-min_x),(max_y-min_y))/5)
                max_y += round(min((max_x-min_x),(max_y-min_y))/5)
                min_x -= round(min((max_x-min_x),(max_y-min_y))/5)
                min_y -= round(min((max_x-min_x),(max_y-min_y))/5)

                if max_x >= img.shape[1]: max_x = img.shape[1]
                if max_y >= img.shape[0]: max_y = img.shape[0]
                if min_x <= 0: min_x = 0
                if min_y <= 0: min_y = 0

                #res_inpaint_file = "./result_inpainting/" + "res_" + filename + '_cut{}.jpg'.format(i)
                cut_inpaint_img = img_copy[min_y:max_y, min_x:max_x]
                
                
                #cut_inpaint_img = cv2.cvtColor(cut_inpaint_img, cv2.COLOR_BGR2RGB)
                
                #cut_inpaint_img = torch.tensor(cut_inpaint_img)
                #test = trfm.functional.to_numpy(cut_inpaint_img)
                
                list_inpaint_img.append(cut_inpaint_img)
                list_pred.append(pred_trans)
                

            poly = poly.reshape(-1, 2)
            #cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
            ptColor = (0, 255, 255)
            if verticals is not None:
                if verticals[i]:
                    ptColor = (255, 0, 0)

            if texts is not None:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                cv2.putText(img, "{}".format(texts[i]), (poly[0][0]+1, poly[0][1]+1), font, font_scale, (0, 0, 0), thickness=1)
                cv2.putText(img, "{}".format(texts[i]), tuple(poly[0]), font, font_scale, (0, 255, 255), thickness=1)
    cv2.imwrite(res_img_file, img)
    return list_inpaint_img, list_inpaint_point, list_pred
    # num = 0
    # with open(res_file, 'w') as f:
    #     for i, box in enumerate(boxes):
    #         poly = np.array(box).astype(np.int32).reshape((-1))
    #         strResult = ','.join([str(p) for p in poly]) + '\r\n'
    #         f.write(strResult)

    #         min_x = min(poly[0], poly[2], poly[4], poly[6])
    #         max_x = max(poly[0], poly[2], poly[4], poly[6])
    #         min_y = min(poly[1], poly[3], poly[5], poly[7])
    #         max_y = max(poly[1], poly[3], poly[5], poly[7])
    #         cut_img = img_copy[min_y:max_y, min_x:max_x]

    #         with torch.no_grad():
    #             batch_size = 1
    #             length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device) # 1= batch_size
    #             text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
                
    #             image = tf(cut_img)
    #             image = image.unsqueeze(dim=1).to(device)

    #             if 'CTC' in opt.Prediction:
    #                 preds = model(image, text_for_pred)

    #                 # Select max probabilty (greedy decoding) then decode index to character
    #                 preds_size = torch.IntTensor([preds.size(1)] * 1)
    #                 _, preds_index = preds.max(2)
    #                 # preds_index = preds_index.view(-1)
    #                 preds_str = converter.decode(preds_index, preds_size)

    #             else:
    #                 preds = model(image, text_for_pred, is_train=False)

    #                 # select max probabilty (greedy decoding) then decode index to character
    #                 _, preds_index = preds.max(2)
    #                 preds_str = converter.decode(preds_index, length_for_pred)

    #             preds_prob = F.softmax(preds, dim=2)
    #             preds_max_prob, _ = preds_prob.max(dim=2)
    #             for pred, pred_max_prob in zip(preds_str, preds_max_prob):
    #                 if 'Attn' in opt.Prediction:
    #                     pred_EOS = pred.find('[s]')
    #                     pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
    #                     pred_max_prob = pred_max_prob[:pred_EOS]
                        

    #             confidence_score = pred_max_prob.cumprod(dim=0)[-1]
    #             print(f'{pred:25s}\t{confidence_score:0.4f}')
                
    #         res_cut_file = dirname + "res_" + filename + '_cut{}.jpg'.format(i)
    #         cv2.imwrite(res_cut_file, cut_img)
    #         if confidence_score >= 0.85:
    #             points = [min_x, max_x, min_y, max_y]
    #             #list_inpaint_point.append(points)
    #             num += 1
                
    #             list_inpaint_img.append(cut_img)
    #             #list_pred.append(pred)
    #             list_ip = [points, pred]
    #             list_all.append(list_ip)

                
    #         poly = poly.reshape(-1, 2)
    #         cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
    #         ptColor = (0, 255, 255)
    #         if verticals is not None:
    #             if verticals[i]:
    #                 ptColor = (255, 0, 0)

    #         if texts is not None:
    #             font = cv2.FONT_HERSHEY_SIMPLEX
    #             font_scale = 0.5
    #             cv2.putText(img, "{}".format(texts[i]), (poly[0][0]+1, poly[0][1]+1), font, font_scale, (0, 0, 0), thickness=1)
    #             cv2.putText(img, "{}".format(texts[i]), tuple(poly[0]), font, font_scale, (0, 255, 255), thickness=1)

    # nlist_inpaint_img = []
    # nlist_all = []
    # nlist_inpaint_point = []
    # nlist_pred = []
    # realnum = 0

    # #if texts is not None:  
    # temp_list = [list_all[0].copy()]
    # cutnum = 0

    # for i in range(0, num-1):
    #     y_0_min = list_all[i][0][2]
    #     y_0_max = list_all[i][0][3]
    #     x_0_max = list_all[i][0][1]
    #     y_1_min = list_all[i+1][0][2]
    #     y_1_max = list_all[i+1][0][3]
    #     x_1_min = list_all[i+1][0][0]
    #     x_1_max = list_all[i+1][0][1]

    #     y_0_cen = (y_0_min+y_0_max)/2
    #     y_1_cen = (y_1_min+y_1_max)/2
    #     y_0 = y_0_max-y_0_min
    #     y_1 = y_1_max-y_1_min

    #     #x = list_inpaint_point[i+1][0] - list_inpaint_point[i][1]

    #     if ((y_0_min<y_1_cen<y_0_max) and (y_1_min<y_0_cen<y_1_max)):
    #         temp_list.append(list_all[i+1])
    #     else:
    #         nlist_all.append(temp_list)
    #         temp_list = [list_all[i+1].copy()]

    #         realnum += 1

    # nlist_all.append(temp_list)
    
    # for i in range(0, realnum+1):
    #     nlist_all[i].sort()
    #     n_temp_list = nlist_all[i][0][0].copy()
    #     n_temp_pred = nlist_all[i][0][1]

    #     for j in range(0, len(nlist_all[i])-1):
    #         y_0_min = nlist_all[i][j][0][2]
    #         y_0_max = nlist_all[i][j][0][3]
    #         x_0_max = nlist_all[i][j][0][1]
    #         y_1_min = nlist_all[i][j+1][0][2]
    #         y_1_max = nlist_all[i][j+1][0][3]
    #         x_1_min = nlist_all[i][j+1][0][0]
    #         x_1_max = nlist_all[i][j+1][0][1]

    #         y_0_cen = (y_0_min+y_0_max)/2
    #         y_1_cen = (y_1_min+y_1_max)/2
    #         y_0 = y_0_max-y_0_min
    #         y_1 = y_1_max-y_1_min

    #         x = nlist_all[i][j+1][0][0] - nlist_all[i][j][0][1]

    #         if ((y_0_min<y_1_cen<y_0_max) and (y_1_min<y_0_cen<y_1_max) and (x<y_0/2) and (x<y_1/2) and (x_0_max<x_1_min)):
    #             n_temp_list = [n_temp_list[0], x_1_max, min(n_temp_list[2],y_1_min), max(n_temp_list[3],y_1_max)]
    #             n_temp_pred += ' ' + nlist_all[i][j+1][1]
    #         else:
    #             nlist_inpaint_point.append(n_temp_list)
    #             nlist_pred.append(n_temp_pred)

    #             ncut_img = img_copy[n_temp_list[2]:n_temp_list[3], n_temp_list[0]:n_temp_list[1]]
    #             nlist_inpaint_img.append(ncut_img)
    #             new_res_cut_file = dirname + "res_" + filename + 'new_cut{}.jpg'.format(cutnum)
    #             cv2.imwrite(new_res_cut_file, ncut_img)

    #             n_temp_list = [x_1_min, x_1_max, y_1_min, y_1_max]
    #             n_temp_pred = nlist_all[i][j+1][1]
    #             cutnum += 1

    #     nlist_inpaint_point.append(n_temp_list)
    #     nlist_pred.append(n_temp_pred)
    #     ncut_img = img_copy[n_temp_list[2]:n_temp_list[3], n_temp_list[0]:n_temp_list[1]]
    #     nlist_inpaint_img.append(ncut_img)
    #     new_res_cut_file = dirname + "res_" + filename + 'new_cut{}.jpg'.format(cutnum)
    #     cv2.imwrite(new_res_cut_file, ncut_img)

    # cv2.imwrite(res_img_file, img)
    # return nlist_inpaint_img, nlist_inpaint_point, nlist_pred

