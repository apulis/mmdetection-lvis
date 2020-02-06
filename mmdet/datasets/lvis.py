import os.path as osp

import mmcv
import numpy as np
import pycocotools.mask as maskUtils

from mmdet.core.evaluation import get_classes
from mmdet.lvis import LVIS
from .custom import CustomDataset
from .registry import DATASETS


@DATASETS.register_module
class LVISDataset(CustomDataset):

    CLASSES = get_classes('lvis')

    def load_annotations(self, ann_file):
        self.lvis = LVIS(ann_file)
        self.cat_ids = self.lvis.get_cat_ids()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.lvis.get_img_ids()
        img_infos = []
        for i in self.img_ids:
            info = self.lvis.load_imgs([i])[0]
            fname = info['file_name']
            if fname.find('COCO_val2014_') > -1:
                fname = fname.split('COCO_val2014_')[-1]
            info['filename'] = fname
            img_infos.append(info)
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.lvis.get_ann_ids(img_ids=[img_id])
        ann_info = self.lvis.load_anns(ids=ann_ids)
        return self._parse_ann_info(self.img_infos[idx], ann_info)

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.lvis.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def show_annotations(self, img_idx, show=False, out_file=None, **kwargs):
        img_info = self.img_infos[img_idx]
        img_id = img_info['id']
        img_path = osp.join(self.img_prefix, img_info['file_name'])
        img = mmcv.imread(img_path)
        annotations = self.lvis.load_anns(
            ids=self.lvis.get_ann_ids(img_ids=[img_id]))

        bboxes = []
        labels = []
        class_names = ['bg'] + list(self.CLASSES)
        for ann in annotations:
            if len(ann['segmentation']) > 0:
                rle = maskUtils.frPyObjects(ann['segmentation'],
                                            img_info['height'],
                                            img_info['width'])
                ann_mask = np.sum(
                    maskUtils.decode(rle), axis=2).astype(np.bool)
                color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                img[ann_mask] = img[ann_mask] * 0.5 + color_mask * 0.5
            bbox = ann['bbox']
            x, y, w, h = bbox
            bboxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
        bboxes = np.stack(bboxes)
        labels = np.stack(labels)
        mmcv.imshow_det_bboxes(
            img,
            bboxes,
            labels,
            class_names=class_names,
            show=show,
            out_file=out_file,
            **kwargs)
        if not (show or out_file):
            return img
