import os

import numpy as np
import torch.utils.data as data
from PIL import Image


import lib.utils.kitti.calibration as calibration
import lib.utils.kitti.kitti_utils as kitti_utils


class KittiDataset(data.Dataset):
    def __init__(self, root_dir, split='train'):
        assert split in ['train', 'val', 'trainval', 'test']
        self.split = split
        is_test = self.split == 'test'
        self.dataset_dir = os.path.join(root_dir, 'KITTI', 'object', 'testing' if is_test else 'training')

        split_dir = os.path.join(root_dir, 'KITTI', 'ImageSets', split + '.txt')
        self.idx_list = [x.strip() for x in open(split_dir).readlines()]
        self.num_sample = self.idx_list.__len__()

        self.image_dir = os.path.join(self.dataset_dir, 'image_2')
        self.depth_dir = os.path.join(self.dataset_dir, 'depth')
        self.lidar_dir = os.path.join(self.dataset_dir, 'velodyne')
        self.calib_dir = os.path.join(self.dataset_dir, 'calib')
        self.label_dir = os.path.join(self.dataset_dir, 'label_2')
        self.plane_dir = os.path.join(self.dataset_dir, 'planes')
        self.pseudo_lidar_dir = os.path.join(self.dataset_dir, 'pseudo_lidar')

    def get_image(self, idx):
        #assert False, 'DO NOT USE cv2 NOW, AVOID DEADLOCK'
        import cv2
        # cv2.setNumThreads(0)  # for solving deadlock when switching epoch
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        return Image.open(img_file)    # (H, W, 3) RGB mode

    def get_depth(self, idx):
        img_file = os.path.join(self.depth_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        return Image.open(img_file)

    def get_image_shape(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        im = Image.open(img_file)
        width, height = im.size
        return height, width, 3

    def get_lidar(self, idx):
        lidar_file = os.path.join(self.lidar_dir, '%06d.bin' % idx)
        assert os.path.exists(lidar_file)
        return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

    def get_pseudo_lidar(self, idx, channels):
        pseudo_lidar_file = os.path.join(self.pseudo_lidar_dir, '%06d.bin' % idx)
        assert os.path.exists(pseudo_lidar_file)
        return np.fromfile(pseudo_lidar_file, dtype=np.float32).reshape(-1, channels)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        assert os.path.exists(calib_file)
        return calibration.Calibration(calib_file)

    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
        assert os.path.exists(label_file)
        return kitti_utils.get_objects_from_label(label_file)

    def get_road_plane(self, idx):
        plane_file = os.path.join(self.plane_dir, '%06d.txt' % idx)
        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

if __name__ == '__main__':
    print('just for debug: kitti_dataset')
    kitti_dataset = KittiDataset(root_dir='../../data', split='val')
    print(kitti_dataset.image_dir)
    print(kitti_dataset.get_image_shape(1))
    print(kitti_dataset.idx_list.__len__())
    points = kitti_dataset.get_lidar(0)
    print (points[0:10])
    # calib = kitti_dataset.get_calib(2)
    # uvdepth = np.zeros((1, 3))
    # uvdepth[0, 0:2] = np.array([(280.38 + 344.9)/2.0, (185.1 + 215.59)/2.0])
    # uvdepth[0, 2] = 20  # some random depth
    # box2d_center_rect = calib.img_to_rect(uvdepth[:, 0], uvdepth[:, 1], uvdepth[:, 2])
    # frustum_angle = -1 * np.arctan2(box2d_center_rect[0, 2], box2d_center_rect[0, 0])
    # print(frustum_angle)