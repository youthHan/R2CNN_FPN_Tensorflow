# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
sys.path.append('../')

import time
from data.io import image_preprocess
from libs.networks.network_factory import get_network_byname
from libs.rpn import build_rpn
from help_utils.help_utils import *
from help_utils.tools import *
from libs.configs import cfgs
from tools import restore_model
from libs.fast_rcnn import build_fast_rcnn1

import threadpool

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_imgs(id_section=None):
    mkdir(cfgs.INFERENCE_IMAGE_PATH)
    root_dir = cfgs.INFERENCE_IMAGE_PATH

    if id_section is None:
        img_name_list = os.listdir(root_dir)
        # img_name_list = ['P1089__1__512___0.png']
        img_list = [cv2.imread(os.path.join(root_dir, img_name))
                    for img_name in img_name_list]
    else:
        img_name_list = os.listdir(root_dir)[id_section[0]:id_section[1]]
        # img_name_list = ['P1089__1__512___0.png']
        img_list = [cv2.imread(os.path.join(root_dir, img_name))
                    for img_name in img_name_list]
    if len(img_name_list) == 0:
        assert 'no test image in {}!'.format(cfgs.INFERENCE_IMAGE_PATH)

    return img_list, img_name_list


def inference(id_section=None):
    with tf.Graph().as_default():

        img_plac = tf.placeholder(shape=[None, None, 3], dtype=tf.uint8)

        img_tensor = tf.cast(img_plac, tf.float32) - tf.constant([103.939, 116.779, 123.68])
        img_batch = image_preprocess.short_side_resize_for_inference_data(img_tensor,
                                                                          target_shortside_len=cfgs.SHORT_SIDE_LEN)

        # ***********************************************************************************************
        # *                                         share net                                           *
        # ***********************************************************************************************
        _, share_net = get_network_byname(net_name=cfgs.NET_NAME,
                                          inputs=img_batch,
                                          num_classes=None,
                                          is_training=True,
                                          output_stride=None,
                                          global_pool=False,
                                          spatial_squeeze=False)
        # ***********************************************************************************************
        # *                                            RPN                                              *
        # ***********************************************************************************************
        rpn = build_rpn.RPN(net_name=cfgs.NET_NAME,
                            inputs=img_batch,
                            gtboxes_and_label=None,
                            is_training=False,
                            share_head=cfgs.SHARE_HEAD,
                            share_net=share_net,
                            stride=cfgs.STRIDE,
                            anchor_ratios=cfgs.ANCHOR_RATIOS,
                            anchor_scales=cfgs.ANCHOR_SCALES,
                            scale_factors=cfgs.SCALE_FACTORS,
                            base_anchor_size_list=cfgs.BASE_ANCHOR_SIZE_LIST,  # P2, P3, P4, P5, P6
                            level=cfgs.LEVEL,
                            top_k_nms=cfgs.RPN_TOP_K_NMS,
                            rpn_nms_iou_threshold=cfgs.RPN_NMS_IOU_THRESHOLD,
                            max_proposals_num=cfgs.MAX_PROPOSAL_NUM,
                            rpn_iou_positive_threshold=cfgs.RPN_IOU_POSITIVE_THRESHOLD,
                            rpn_iou_negative_threshold=cfgs.RPN_IOU_NEGATIVE_THRESHOLD,
                            rpn_mini_batch_size=cfgs.RPN_MINIBATCH_SIZE,
                            rpn_positives_ratio=cfgs.RPN_POSITIVE_RATE,
                            remove_outside_anchors=False,  # whether remove anchors outside
                            rpn_weight_decay=cfgs.WEIGHT_DECAY[cfgs.NET_NAME])

        # rpn predict proposals
        rpn_proposals_boxes, rpn_proposals_scores = rpn.rpn_proposals()  # rpn_score shape: [300, ]

        # ***********************************************************************************************
        # *                                         Fast RCNN                                           *
        # ***********************************************************************************************
        fast_rcnn = build_fast_rcnn1.FastRCNN(feature_pyramid=rpn.feature_pyramid,
                                              rpn_proposals_boxes=rpn_proposals_boxes,
                                              rpn_proposals_scores=rpn_proposals_scores,
                                              img_shape=tf.shape(img_batch),
                                              roi_size=cfgs.ROI_SIZE,
                                              roi_pool_kernel_size=cfgs.ROI_POOL_KERNEL_SIZE,
                                              scale_factors=cfgs.SCALE_FACTORS,
                                              gtboxes_and_label=None,
                                              gtboxes_and_label_minAreaRectangle=None,
                                              fast_rcnn_nms_iou_threshold=cfgs.FAST_RCNN_NMS_IOU_THRESHOLD,
                                              fast_rcnn_maximum_boxes_per_img=cfgs.FAST_RCNN_MAXIMUM_BOXES_PER_IMAGE,
                                              fast_rcnn_nms_max_boxes_per_class=cfgs.FAST_RCNN_NMS_MAX_BOXES_PER_CLASS,
                                              show_detections_score_threshold=cfgs.FINAL_SCORE_THRESHOLD,
                                              # show detections which score >= 0.6
                                              num_classes=cfgs.CLASS_NUM,
                                              fast_rcnn_minibatch_size=cfgs.FAST_RCNN_MINIBATCH_SIZE,
                                              fast_rcnn_positives_ratio=cfgs.FAST_RCNN_POSITIVE_RATE,
                                              fast_rcnn_positives_iou_threshold=cfgs.FAST_RCNN_IOU_POSITIVE_THRESHOLD,
                                              # iou>0.5 is positive, iou<0.5 is negative
                                              use_dropout=cfgs.USE_DROPOUT,
                                              weight_decay=cfgs.WEIGHT_DECAY[cfgs.NET_NAME],
                                              is_training=False,
                                              level=cfgs.LEVEL)

        fast_rcnn_decode_boxes, fast_rcnn_score, num_of_objects, detection_category, \
        fast_rcnn_decode_boxes_rotate, fast_rcnn_score_rotate, num_of_objects_rotate, detection_category_rotate = \
            fast_rcnn.fast_rcnn_predict()

        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )

        restorer, restore_ckpt = restore_model.get_restorer()

        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(init_op)
            if not restorer is None:
                restorer.restore(sess, restore_ckpt)
                print('restore model')

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)

            imgs, img_names = get_imgs(id_section)
            rotate_boxes_dict = {}
            h_boxes_dict = {}
            for i, img in enumerate(imgs):

                start = time.time()

                _img_batch, _fast_rcnn_decode_boxes, _fast_rcnn_score, _detection_category, \
                _fast_rcnn_decode_boxes_rotate,  _fast_rcnn_score_rotate, _detection_category_rotate = \
                    sess.run([img_batch, fast_rcnn_decode_boxes, fast_rcnn_score, detection_category,
                              fast_rcnn_decode_boxes_rotate, fast_rcnn_score_rotate, detection_category_rotate],
                             feed_dict={img_plac: img})
                end = time.time()

                img_np = np.squeeze(_img_batch, axis=0)
                # print(img_np.shape)
                img_horizontal_np = draw_box_cv(img_np,
                                                boxes=_fast_rcnn_decode_boxes,
                                                labels=_detection_category,
                                                scores=_fast_rcnn_score,
                                                # ori_shape=list(img.shape))
                                                ori_shape=None)
                img_oriented_np = draw_rotate_box_cv(img_np,
                                                boxes=_fast_rcnn_decode_boxes_rotate,
                                                labels=_detection_category_rotate,
                                                scores=_fast_rcnn_score_rotate,
                                                # ori_shape=list(img.shape))
                                                ori_shape=None)
                # print("shape of img: ", img.shape)
                # print("shape of img_np: ", img_np.shape)

                mkdir(os.path.join(cfgs.INFERENCE_SAVE_PATH.format('horizontal'), 'images_inference'))
                mkdir(os.path.join(cfgs.INFERENCE_SAVE_PATH.format('oriented'), 'images_inference'))
                h_points_boxes, h_labels, h_scores = convert_h_box_to_points(
                                                    ori_shape=list(img.shape),
                                                    boxes=_fast_rcnn_decode_boxes,
                                                    scores=_fast_rcnn_score,
                                                    labels=_detection_category)

                cv2.imwrite(os.path.join(cfgs.INFERENCE_SAVE_PATH.format('horizontal'), 'images_inference') + '/{}.jpg'.format(img_names[i]), img_horizontal_np)
                cv2.imwrite(os.path.join(cfgs.INFERENCE_SAVE_PATH.format('oriented'), 'images_inference') + '/{}.jpg'.format(img_names[i]), img_oriented_np)
                rotate_points_boxes, rotate_labels, rotate_scores = convert_rotate_box_to_points(
                                                    ori_shape=list(img.shape),
                                                    boxes=_fast_rcnn_decode_boxes_rotate,
                                                    scores=_fast_rcnn_score_rotate,
                                                    labels=_detection_category_rotate)

                h_boxes_dict[img_names[i].split('/')[-1]] = {"boxes": h_points_boxes, "labels":h_labels, "scores":h_scores}
                rotate_boxes_dict[img_names[i].split('/')[-1]] = {"boxes": rotate_points_boxes, "labels":rotate_labels, "scores":rotate_scores}
                view_bar('{} cost {}s'.format(img_names[i], (end - start)), i + 1, len(imgs))
            coord.request_stop()
            coord.join(threads)

    return h_boxes_dict, rotate_boxes_dict


def multi_threads_inference(max_workers):
    lens = len(os.listdir(cfgs.INFERENCE_IMAGE_PATH)[:10])

    args = []
    stride = int(lens / max_workers)
    for i in range(max_workers):
        begin = i * stride
        end = lens if i == max_workers - 1 else (i + 1) * stride
        print(i, " ", begin, " ", end)
        args.append([begin, end])

    pool = threadpool.ThreadPool(max_workers)
    requests = threadpool.makeRequests(inference, args)
    [pool.putRequest(req) for req in requests]
    pool.wait()


if __name__ == '__main__':
    h_boxes_dict, rotate_boxes_dict = inference()

    save_prediction(h_boxes_dict, "horizontal")
    save_prediction(rotate_boxes_dict, "oriented")

    # multi_threads_inference(2)