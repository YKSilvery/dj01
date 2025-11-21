from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
from pathlib import Path

import numpy as np
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import torch.utils.data as data


class RMRoboDataset(data.Dataset):
  num_classes = 14
  default_resolution = [512, 512]
  mean = np.array([0.40789654, 0.44719302, 0.47026115],
                   dtype=np.float32).reshape(1, 1, 3)
  std = np.array([0.28863828, 0.27408164, 0.27809835],
                  dtype=np.float32).reshape(1, 1, 3)
  class_name = [
    '__background__',
    'B1', 'B2', 'B3', 'B4', 'B5', 'BO', 'BS',
    'R1', 'R2', 'R3', 'R4', 'R5', 'RO', 'RS',
  ]
  _valid_ids = list(range(1, num_classes + 1))

  def __init__(self, opt, split):
    super(RMRoboDataset, self).__init__()
    self.split = split
    self.opt = opt

    data_root = Path(opt.data_dir)
    self.data_dir = data_root.as_posix()
    self.img_dir = data_root.joinpath('images', split).as_posix()
    annot_file = data_root.joinpath('annotations', f'annotations_{split}.json')
    if not annot_file.exists():
      raise FileNotFoundError(f'Missing annotations file: {annot_file}')
    self.annot_path = annot_file.as_posix()
    self.max_objs = 128
    self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
    self.voc_color = [
      (v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32)
      for v in range(1, self.num_classes + 1)
    ]
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)

    print('==> Initializing RM Robo {} data.'.format(split))
    self.coco = coco.COCO(self.annot_path)
    self.images = self.coco.getImgIds()
    self.num_samples = len(self.images)
    print('Loaded {} {} samples'.format(split, self.num_samples))

  def _to_float(self, x):
    return float('{:.2f}'.format(x))

  def __len__(self):
    return self.num_samples

  def convert_eval_format(self, all_bboxes):
    detections = []
    for image_id in all_bboxes:
      for cls_ind in all_bboxes[image_id]:
        category_id = self._valid_ids[cls_ind - 1]
        for bbox in all_bboxes[image_id][cls_ind]:
          bbox[2] -= bbox[0]
          bbox[3] -= bbox[1]
          score = bbox[4]
          bbox_out = list(map(self._to_float, bbox[0:4]))
          detection = {
              'image_id': int(image_id),
              'category_id': int(category_id),
              'bbox': bbox_out,
              'score': float('{:.2f}'.format(score))
          }
          if len(bbox) > 5:
            extreme_points = list(map(self._to_float, bbox[5:13]))
            detection['extreme_points'] = extreme_points
          detections.append(detection)
    return detections

  def save_results(self, results, save_dir):
    json.dump(self.convert_eval_format(results),
              open('{}/results.json'.format(save_dir), 'w'))

  def run_eval(self, results, save_dir):
    self.save_results(results, save_dir)
    coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
    coco_eval = COCOeval(self.coco, coco_dets, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
