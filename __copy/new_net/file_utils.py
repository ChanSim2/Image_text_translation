import os
import numpy as np
import cv2
import imgproc
from demo import demo

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
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == 'pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    
    return img_files, mask_files, gt_files

def saveResult(img_file, img, boxes, dirname='./result/', verticals=None, texts=None):
    img = np.array(img)
    img_copy = img.copy()

    # make result file list
    filename, file_ext = os.path.splitext(os.path.basename(img_file))

    # result directory
    res_file = dirname + "res_" + filename + '.txt'
    res_img_file = dirname + "res_" + filename + '.jpg'
    

    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    with open(res_file, 'w') as f:
        for i, box in enumerate(boxes):
            poly = np.array(box).astype(np.int32).reshape((-1))
            strResult = ','.join([str(p) for p in poly]) + '\r\n'
            f.write(strResult)

            res_cut_file = dirname + "res_" + filename + '_cut{}.jpg'.format(i)
            min_x = min(poly[0], poly[2], poly[4], poly[6])
            max_x = max(poly[0], poly[2], poly[4], poly[6])
            min_y = min(poly[1], poly[3], poly[5], poly[7])
            max_y = max(poly[1], poly[3], poly[5], poly[7])
            cut_img = img_copy[min_y:max_y, min_x:max_x]
            cv2.imwrite(res_cut_file, cut_img)

            max_x += round(min((max_x-min_x),(max_y-min_y))/2)
            max_y += round(min((max_x-min_x),(max_y-min_y))/2)
            min_x -= round(min((max_x-min_x),(max_y-min_y))/2)
            min_y -= round(min((max_x-min_x),(max_y-min_y))/2)

            if max_x >= img.shape[1]: max_x = img.shape[1]
            if max_y >= img.shape[0]: max_y = img.shape[0]
            if min_x <= 0: min_x = 0
            if min_y <= 0: min_y = 0

            res_inpaint_file = "./result_inpainting/" + "res_" + filename + '_cut{}.jpg'.format(i)
            cut_img = img_copy[min_y:max_y, min_x:max_x]
            #gen_input_img = cut_img
            cv2.imwrite(res_inpaint_file, cut_img)

            poly = poly.reshape(-1, 2)
            #print(poly[0], type(poly))
            cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
            ptColor = (0, 255, 255)
            if verticals is not None:
                if verticals[i]:
                    ptColor = (255, 0, 0)

            if texts is not None:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                cv2.putText(img, "{}".format(texts[i]), (poly[0][0]+1, poly[0][1]+1), font, font_scale, (0, 0, 0), thickness=1)
                cv2.putText(img, "{}".format(texts[i]), tuple(poly[0]), font, font_scale, (0, 255, 255), thickness=1)
    
    # Save result image
    cv2.imwrite(res_img_file, img)

