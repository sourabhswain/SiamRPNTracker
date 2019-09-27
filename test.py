from __future__ import absolute_import

import cv2
import numpy as np
import os

from got10k.trackers import Tracker
from got10k.experiments import ExperimentGOT10k
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.model_load import load_pretrain
from pysot.models.model_builder import ModelBuilder
from pysot.core.config import cfg
from pysot.utils.bbox import get_axis_aligned_bbox
from toolkit.datasets import DatasetFactory
from matplotlib.image import imread

from PIL import  Image
from got10k.datasets import GOT10k
from got10k.utils.viz import show_frame

if __name__ == '__main__':
    cfg.merge_from_file('/home/sourabhswain/Documents/SiamRPN/config.yaml')
    net_path = '/home/sourabhswain/Documents/SiamRPN/model.pth'
    model = ModelBuilder()
    model = load_pretrain(model, net_path).cuda().eval()
    tracker = build_tracker(model)

    dataset = GOT10k(root_dir='/home/sourabhswain/Documents/SiamRPN/dataset', subset='val')

    #dataset = DatasetFactory.create_dataset(name='GOT-10k',
    #                                        dataset_root='/home/sourabhswain/Documents/SiamRPN/dataset',
    #                                        load_img=False)
    """
    for v_idx, (video, anno) in enumerate(dataset):


        toc = 0
        pred_bboxes = []
        scores = []
        track_times = []


        #Access all frames
        for idx, img_file in enumerate(video):
            tic = cv2.getTickCount()
            video_name = img_file.split('/')[-2]
            img = imread(img_file)
            if idx == 0:

                cx, cy, w, h = get_axis_aligned_bbox(np.array(anno[idx]))
                gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                tracker.init(img, gt_bbox_)
                pred_bbox = gt_bbox_
                scores.append(None)
                pred_bboxes.append(pred_bbox)
            else :
                outputs = tracker.track(img)
                pred_bbox = outputs['bbox']
                pred_bboxes.append(pred_bbox)
                scores.append(outputs['best_score'])
                toc += cv2.getTickCount() - tic
                track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())
                if idx == 0:
                    cv2.destroyAllWindows()

                vis = True
                if vis and idx > 0:
                    gt_bbox = list(map(int, anno[idx]))
                    pred_bbox = list(map(int, pred_bbox))
                    #cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                     #             (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]), (0, 255, 0), 3)
                    #cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                     #             (pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3]), (0, 255, 255), 3)
                    #cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    #cv2.imshow(video_name, img)
                    #cv2.waitKey(1)
                    print(video_name, idx)


                toc /= cv2.getTickFrequency()
                model_name = "SiamRPN++"
                dataset_path = '/home/sourabhswain/Documents/SiamRPN/GOT/results'

                video_path = os.path.join(dataset_path, model_name, video_name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path, '{}_001.txt'.format(video_name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x]) + '\n')
                result_path = os.path.join(video_path,
                                           '{}_time.txt'.format(video_name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))


    """
    #experiment

    experiments = ExperimentGOT10k('/home/sourabhswain/Documents/SiamRPN/dataset', subset='val',
                                   result_dir='/home/sourabhswain/Documents/SiamRPN/compare/GOT_res',
                                   report_dir='/home/sourabhswain/Documents/SiamRPN/GOT/reports_org')


    #experiments.run(tracker)
    experiments.report([tracker.name])



