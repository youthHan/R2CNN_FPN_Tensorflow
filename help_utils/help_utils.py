# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import cv2
import os
from libs.label_name_dict.label_dict import LABEl_NAME_MAP
from libs.configs import cfgs

def show_boxes_in_img(img, boxes_and_label):
    '''

    :param img:
    :param boxes: must be int
    :return:
    '''
    boxes_and_label = boxes_and_label.astype(np.int64)
    img = np.array(img, np.float32)
    img = np.array(img*255/np.max(img), np.uint8)
    for box in boxes_and_label:
        ymin, xmin, ymax, xmax, label = box[0], box[1], box[2], box[3], box[4]

        category = LABEl_NAME_MAP[label]

        color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
        cv2.rectangle(img,
                      pt1=(xmin, ymin),
                      pt2=(xmax, ymax),
                      color=color)
        cv2.putText(img,
                    text=category,
                    org=((xmin+xmax)//2, (ymin+ymax)//2),
                    fontFace=1,
                    fontScale=1,
                    color=(0, 0, 255))

    cv2.imshow('img_', img)
    cv2.waitKey(0)


def draw_box_cv(img, boxes, labels, scores, ori_shape):
    if ori_shape is None:
        scale = 1.0
    else:
        scale = min(ori_shape[:2]) * 1.0 / 600.0
    # print("scales", scale)
    img = img + np.array([103.939, 116.779, 123.68])
    boxes = boxes.astype(np.int64)
    labels = labels.astype(np.int32)
    img = np.array(img, np.float32)
    img = np.array(img*255/np.max(img), np.uint8)

    boxes = boxes*scale
    boxes = boxes.astype(np.int64)
    num_of_object = 0
    for i, box in enumerate(boxes):
        ymin, xmin, ymax, xmax = box[0], box[1], box[2], box[3]
        # print(box)
        label = labels[i]
        if label != 0:
            num_of_object += 1
            color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
            cv2.rectangle(img,
                          pt1=(xmin, ymin),
                          pt2=(xmax, ymax),
                          color=color,
                          thickness=2)

            category = LABEl_NAME_MAP[label]

            if scores is not None:
                cv2.rectangle(img,
                              pt1=(xmin, ymin),
                              pt2=(xmin+150, ymin+15),
                              color=color,
                              thickness=-1)
                cv2.putText(img,
                            text=category+": "+str(scores[i]),
                            org=(xmin, ymin+10),
                            fontFace=1,
                            fontScale=1,
                            thickness=2,
                            color=(color[1], color[2], color[0]))
            else:
                cv2.rectangle(img,
                              pt1=(xmin, ymin),
                              pt2=(xmin + 40, ymin + 15),
                              color=color,
                              thickness=-1)
                cv2.putText(img,
                            text=category,
                            org=(xmin, ymin + 10),
                            fontFace=1,
                            fontScale=1,
                            thickness=2,
                            color=(color[1], color[2], color[0]))
    cv2.putText(img,
                text=str(num_of_object),
                org=((img.shape[1]) // 2, (img.shape[0]) // 2),
                fontFace=3,
                fontScale=1,
                color=(255, 0, 0))
    return img


def draw_rotate_box_cv(img, boxes, labels, scores, ori_shape):
    if ori_shape is None:
        scale = 1.0
    else:
        scale = min(ori_shape[:2]) * 1.0 / 600.0

    img = img + np.array([103.939, 116.779, 123.68])
    boxes = boxes.astype(np.int64)
    labels = labels.astype(np.int32)
    img = np.array(img, np.float32)
    img = np.array(img*255/np.max(img), np.uint8)

    num_of_object = 0
    for i, box in enumerate(boxes):
        box[:4] = np.ceil(box[:4]*scale)
        box = box.astype(np.int64)
        y_c, x_c, h, w, theta = box[0], box[1], box[2], box[3], box[4]

        # print(box)

        label = labels[i]
        if label != 0:
            num_of_object += 1
            color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
            rect = ((x_c, y_c), (w, h), theta)
            rect = cv2.boxPoints(rect)
            rect = np.int0(rect)
            cv2.drawContours(img, [rect], -1, color, 3)

            category = LABEl_NAME_MAP[label]

            if scores is not None:
                cv2.rectangle(img,
                              pt1=(x_c, y_c),
                              pt2=(x_c + 120, y_c + 15),
                              color=color,
                              thickness=-1)
                cv2.putText(img,
                            text=category+": "+str(scores[i]),
                            org=(x_c, y_c+10),
                            fontFace=1,
                            fontScale=1,
                            thickness=2,
                            color=(color[1], color[2], color[0]))
            else:
                cv2.rectangle(img,
                              pt1=(x_c, y_c),
                              pt2=(x_c + 40, y_c + 15),
                              color=color,
                              thickness=-1)
                cv2.putText(img,
                            text=category,
                            org=(x_c, y_c + 10),
                            fontFace=1,
                            fontScale=1,
                            thickness=2,
                            color=(color[1], color[2], color[0]))
    cv2.putText(img,
                text=str(num_of_object),
                org=((img.shape[1]) // 2, (img.shape[0]) // 2),
                fontFace=3,
                fontScale=1,
                color=(255, 0, 0))
    return img


def convert_rotate_box_to_points(ori_shape, boxes, scores, labels):
    scale = min(ori_shape[:2]) * 1.0 / 600.0
    boxes = boxes.astype(np.int64)
    labels = labels.astype(np.int32)

    converted_boxes = []
    converted_label = []
    converted_scores = []
    num_of_object = 0
    for i, box in enumerate(boxes):
        box[:4] = np.ceil(box[:4]*scale)
        box = box.astype(np.int64)
        y_c, x_c, h, w, theta = box[0], box[1], box[2], box[3], box[4]

        label = labels[i]
        if label != 0:
            num_of_object += 1
            rect = ((x_c, y_c), (w, h), theta)
            rect = cv2.boxPoints(rect)
            rect = np.int0(rect)

            # print(rect)

            converted_boxes.append(rect)
            converted_label.append(label)
            converted_scores.append(scores[i])

    return np.asarray(converted_boxes), np.asarray(converted_label), np.asarray(converted_scores)


def convert_h_box_to_points(ori_shape, boxes, scores, labels):
    scale = min(ori_shape[:2]) * 1.0 / 600.0
    boxes = boxes.astype(np.int64)
    labels = labels.astype(np.int32)

    converted_boxes = []
    converted_label = []
    converted_scores = []
    num_of_object = 0
    boxes = boxes*scale
    boxes = boxes.astype(np.int64)
    for i, box in enumerate(boxes):
        ymin, xmin, ymax, xmax = box[0], box[1], box[2], box[3]

        label = labels[i]
        if label != 0:
            num_of_object += 1
            rect = ((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax))
            rect = np.int0(rect)
            # print(rect)

            converted_boxes.append(rect)
            converted_label.append(label)
            converted_scores.append(scores[i])

    return np.asarray(converted_boxes), np.asarray(converted_label), np.asarray(converted_scores)

def print_tensors(tensor, tensor_name):

    def np_print(ary):
        ary = ary + np.zeros_like(ary)
        # print(tensor_name + ':', ary)

        print('shape is: ',ary.shape)
        print(10*"%%%%%")
        return ary
    result = tf.py_func(np_print,
                        [tensor],
                        [tensor.dtype])
    result = tf.reshape(result, tf.shape(tensor))
    result = tf.cast(result, tf.float32)
    sum_ = tf.reduce_sum(result)
    tf.summary.scalar('print_s/{}'.format(tensor_name), sum_)


def save_prediction(boxes_dict, type="oriented"):
    for image_name, image_inf in boxes_dict.items():
        print(image_name)
        split_name = image_name.split('.')
        lens = len(image_name.split('.'))
        if lens == 2:
            txt_name = split_name[0]+'.txt'
        elif lens == 3:
            txt_name = '.'.join(split_name[:-1])+'.txt'
        else:
            raise IOError("No such type of image_name as {}".format(image_name))

        txt_dir = os.path.join(cfgs.INFERENCE_SAVE_PATH.format(type),'labelTxt')

        if not os.path.isdir(txt_dir):
            os.makedirs(txt_dir)

        txt_path = os.path.join(txt_dir, txt_name)
        with open(txt_path, 'w') as handle:
            boxes = image_inf["boxes"]
            labels = image_inf["labels"]
            scores = image_inf["scores"]

            num = labels.shape[0]

            for i, box in enumerate(boxes):
                for point in box:
                    handle.write(str(point[0])+' '+str(point[1])+' ')
                handle.write(LABEl_NAME_MAP[labels[i]]+' '+str(scores[i])+'\n')