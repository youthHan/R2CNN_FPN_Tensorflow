import math
import os
import sys
from help_utils.tools import mkdir

detpath = r'/home/ai-i-hanmingfei/datasets/ODAI-ICPR/split_origin/val/s9_labelTxt/oriented/Task1_{:s}.txt'
classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']

def remove_empty_annos():
    with open('/home/ai-i-hanmingfei/proj/R-DFPN_FPN_Tensorflow/data/io/empty_slice_train_so.txt', 'r') as handle:
        empty_lines = handle.readlines()

    for el in empty_lines:
        imgid = el.strip().split('.')[0]
        command = "rm /home/ai-i-hanmingfei/datasets/ODAI-ICPR/split_origin/train/images/{:s}.png".format(imgid)
        os.system(command)
        command = "rm /home/ai-i-hanmingfei/datasets/ODAI-ICPR/split_origin/train/labelTxt/{:s}.txt".format(imgid)
        os.system(command)

def anno_convert():
    o_txt_dir = '/home/ai-i-hanmingfei/datasets/ODAI-ICPR/split_origin/val/s8_inference_oriented/labelTxt'
    out_dir = \
        '/home/ai-i-hanmingfei/datasets/ODAI-ICPR/split_origin/val/s8_inference_oriented/labelTxt_transferred'
    txt_paths = os.listdir(o_txt_dir)
    mkdir(out_dir)

    # print(txt_paths)
    for txt in txt_paths:
        with open(os.path.join(o_txt_dir, txt), 'r') as h_in:
            print(txt)
            lines = h_in.readlines()
            print(lines)

        # if not os.path.isdir(os.path.join(o_txt_dir, 'tmp')):
        #     os.makedirs(os.path.join(o_txt_dir, 'tmp'))

        with open(os.path.join(out_dir, txt), 'w') as h_out:
            for line in lines:
                anno = line.strip().split(' ')
                print(anno)
                line_tmp = [str(math.ceil(int(anno[i]) * 1024.0 / 600.0)) for i in range(8)]
                line_tmp.append(anno[8])
                print(line_tmp)
                h_out.write(' '.join(line_tmp) + '\n')

if __name__ == '__main__':
    remove_empty_annos()