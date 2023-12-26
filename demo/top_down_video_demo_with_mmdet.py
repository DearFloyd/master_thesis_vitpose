# Copyright (c) OpenMMLab. All rights reserved.
import os
import time
import warnings
import numpy as np
from argparse import ArgumentParser

import cv2

from ultralytics import YOLO
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, process_yolo_result, vis_pose_result)
from mmpose.datasets import DatasetInfo
from mmpose.pose_estimation import PoseEstimator

# try:
#     from mmdet.apis import inference_detector, init_detector
#     has_mmdet = True
# except (ImportError, ModuleNotFoundError):
#     has_mmdet = False


def transformer_opt(opt):
    opt = vars(opt)
    del opt['source']
    del opt['weight']
    del opt['save_verbose']
    return opt


def parse_opt_yolov8():
    parser = ArgumentParser()
    parser.add_argument('--weight', type=str, default='/workspace/cv-docker/joey04.li/datasets/master_thesis_yolov8/runs/train/yolov8s-C2f-EMSCP-164bs-500ep/weights/best.pt', help='training model path')
    parser.add_argument('--source', type=str, default='/workspace/cv-docker/joey04.li/datasets/video_data/10.22-raw-split1.mp4', help='source directory for images or videos')
    parser.add_argument('--conf', type=float, default=0.25, help='object confidence threshold for detection')
    parser.add_argument('--iou', type=float, default=0.7, help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--mode', type=str, default='predict', choices=['predict', 'track'], help='predict mode or track mode')
    parser.add_argument('--project', type=str, default='runs/detect', help='project name')
    parser.add_argument('--name', type=str, default='test_11_28_bytetrack_5fps', help='experiment name (project/name)')
    parser.add_argument('--show', action="store_true", help='show results if possible')
    parser.add_argument('--save_verbose', type=str, default='/workspace/cv-docker/joey04.li/datasets/master_thesis_yolov8/runs/detect/test_11_28_bytetrack_5fps/verbose.txt', help='save detail predict verbose results as .txt file')
    parser.add_argument('--save_txt', action="store_true", help='save results as .txt file')
    parser.add_argument('--save_conf', action="store_true", help='save results with confidence scores')
    parser.add_argument('--show_labels', action="store_true", default=False, help='show object labels in plots')
    parser.add_argument('--show_conf', action="store_true", default=False, help='show object confidence scores in plots')
    parser.add_argument('--vid_stride', type=int, default=25, help='video frame-rate stride')
    parser.add_argument('--line_width', type=int, default=1, help='line width of the bounding boxes')
    parser.add_argument('--visualize', action="store_true", help='visualize model features')
    parser.add_argument('--augment', action="store_true", help='apply image augmentation to prediction sources')
    parser.add_argument('--agnostic_nms', action="store_true", help='class-agnostic NMS')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--retina_masks', action="store_true", help='use high-resolution segmentation masks')
    parser.add_argument('--boxes', action="store_true", default=False, help='Show boxes in segmentation predictions')
    parser.add_argument('--save', action="store_true", default=False, help='save result')
    parser.add_argument('--tracker', type=str, default='bytetrack.yaml', choices=['botsort.yaml', 'bytetrack.yaml', 'deepocsort.yaml', 'hybirdsort.yaml', 'ocsort.yaml'], help='tracker type, [botsort.yaml, bytetrack.yaml, deepocsort.yaml, hybirdsort.yaml, ocsort.yaml]')
    parser.add_argument('--reid_weight', type=str, default='/workspace/cv-docker/joey04.li/datasets/yolo_tracking/examples/weights/osnet_x1_0_imagenet.pt', help='if tracker have reid, add reid model path')

    return parser.parse_known_args()[0]


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('--det_config', 
                        default='/workspace/cv-docker/joey04.li/datasets/ViTPose/demo/mmdetection_cfg/yolov3_d53_320_273e_coco.py',
                        help='Config file for detection')
    parser.add_argument('--det_checkpoint', 
                        default='/workspace/cv-docker/joey04.li/datasets/ViTPose/weights/yolov3_d53_320_273e_coco-421362b6.pth',
                        help='Checkpoint file for detection')
    parser.add_argument('--pose_config', 
                        # default='/workspace/cv-docker/joey04.li/datasets/ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitPose+_base_coco+aic+mpii+ap10k+apt36k+wholebody_256x192_udp.py',
                        default='/workspace/cv-docker/joey04.li/datasets/ViTPose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/halpe/ViTPose_base_halpe136_256x192.py',
                        # default='',
                        help='Config file for pose')
    parser.add_argument('--pose_checkpoint', 
                        # default='/workspace/cv-docker/joey04.li/datasets/ViTPose/weights/vitpose+_base_coco+aic+mpii+ap10k+apt36k+wholebody_256x192_udp.pth',
                        # default='/workspace/cv-docker/joey04.li/datasets/ViTPose/runs/train/vit_halpe136_384bs_250ep/best_AP_epoch_250.pth',
                        default='/workspace/cv-docker/joey04.li/datasets/ViTPose/runs/train/vit_halpe136/best_AP_epoch_140.pt',
                        help='Checkpoint file for pose')
    parser.add_argument('--video-path', type=str, 
                        default='/workspace/cv-docker/joey04.li/datasets/video_data/10.22-raw-split1.mp4',
                        help='Video path')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--out-video-root',
        default='/workspace/cv-docker/joey04.li/datasets/ViTPose/runs/detect',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--device', default='cuda:6', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=2,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    # assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()
    opt_yolov8 = parse_opt_yolov8()
    assert args.show or (args.out_video_root != '')
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    det_model_yolov8 = YOLO(model='/workspace/cv-docker/joey04.li/datasets/master_thesis_yolov8/runs/train/yolov8s-C2f-EMSCP-164bs-500ep/weights/best.pt')
    # det_model = init_detector(
    #     args.det_config, args.det_checkpoint, device=args.device.lower())
    pose_estimator = PoseEstimator()
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    cap = cv2.VideoCapture(args.video_path)
    assert cap.isOpened(), f'Faild to load video file {args.video_path}'
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    if args.out_video_root == '':
        save_out_video = False
    else:
        os.makedirs(args.out_video_root, exist_ok=True)
        save_out_video = True

    if save_out_video:
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(
            os.path.join(args.out_video_root,
                         f'vis_{os.path.basename(args.video_path)}'), fourcc,
            fps, size)

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None
    idx = 0
    time_start = time.time()
    opt_yolov8 = transformer_opt(opt_yolov8)
    vid_stride = opt_yolov8['vid_stride']
    while (cap.isOpened()):
        for _ in range(vid_stride):  # 抽帧
            cap.grab()
        success, img = cap.retrieve()
        if not success:
            break
        # flag, img = cap.read()
        print(f'{idx} / {frame_count // vid_stride}')
        idx += 1
        # if not flag:
        #     break
        # test a single image, the resulting box is (x1, y1, x2, y2)
        # mmdet_results = inference_detector(det_model, img)
        yolo_results = det_model_yolov8.predict(source=img, **opt_yolov8)
        yolo_results = yolo_results[0].boxes.data
        yolo_results = np.asanyarray(yolo_results.cpu())
        yolo_results= yolo_results[:, :-1]
        # keep the person class bounding boxes.
        # person_results = process_mmdet_results(mmdet_results, args.det_cat_id)
        person_results = process_yolo_result(yolo_results)

        # test a single image, with a list of bboxes.
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            img,
            person_results,
            bbox_thr=args.bbox_thr,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        for pose in pose_results:
            bbox = pose['bbox'].tolist()  # 将bbox转为scale格式
            xmin, ymin, xmax, ymax = bbox[0:4]
            crop_img = img[int(ymin):int(ymax), int(xmin):int(xmax)]
            dw = 1. / img.shape[0]
            dh = 1. / img.shape[1]
            w = xmax - xmin
            h = ymax - ymin
            center = np.array([xmin + w * 0.5, ymin + h * 0.5])
            scale = np.array([w, h])
            face_pose = pose['keypoints'][26:94]  # 面部关键点索引26-93
            face_pose = np.float64(face_pose[:, 0:2])
            # face_pose = np.float32(face_pose[:, 0:2] - center)
            head_rotation_vector, head_translation_vector = pose_estimator.solve_pose(face_pose)
            angles_pitch = head_rotation_vector[0][0] * 57.3  # 47.3 -160多的为背对着
            # pose_estimator.draw_axis(img, head_rotation_vector, head_translation_vector)

        # show the results
        vis_img = vis_pose_result(
            pose_model,
            img,
            pose_results,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=args.kpt_thr,
            radius=args.radius,
            thickness=args.thickness,
            show=False)

        if args.show:
            cv2.imshow('Image', vis_img)

        if save_out_video:
            videoWriter.write(vis_img)

        if args.show and cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if save_out_video:
        videoWriter.release()
    if args.show:
        cv2.destroyAllWindows()
    
    time_end = time.time()
    print("time use: ", time_end - time_start)


if __name__ == '__main__':
    main()
