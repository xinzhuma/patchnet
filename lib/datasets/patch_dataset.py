import os
import pickle

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

# TODO: replace this func
from lib.utils.kitti.kitti_utils import rotate_pc_along_y


class PatchDataset(data.Dataset):
    def __init__(self, dataset_helper, pickle_file, patch_size,
                 rotate_to_center=False, random_flip=False, random_shift=False,
                 add_rgb=False, from_rgb_detection=False, logger=None):

        self.dataset_helper = dataset_helper
        self.pickle_file = pickle_file
        self.patch_size = patch_size
        self.random_flip = random_flip
        self.random_shift = random_shift
        self.add_rgb = add_rgb
        self.from_rgb_detection = from_rgb_detection  # if true, return 2d box, type, and score, else training label
        self.logger = logger
        self.rotate_to_center = rotate_to_center

        if logger is not None:
            logger.info('Load data from %s' %(self.pickle_file))

        if self.from_rgb_detection:
            with open(self.pickle_file,'rb') as fp:
                self.id_list = pickle.load(fp)
                self.box2d_list = pickle.load(fp)
                self.patch_xyz_list = pickle.load(fp)
                self.patch_rgb_list = pickle.load(fp)
                self.type_list = pickle.load(fp)
                self.frustum_angle_list = pickle.load(fp)
                self.prob_list = pickle.load(fp)
        else:
            with open(self.pickle_file, 'rb') as fp:
                self.patch_xyz_list = pickle.load(fp)
                self.patch_rgb_list = pickle.load(fp)
                self.type_list = pickle.load(fp)
                self.heading_list = pickle.load(fp)
                self.box3d_center_list = pickle.load(fp)
                self.box3d_size_list = pickle.load(fp)
                self.frustum_angle_list = pickle.load(fp)


    def __len__(self):
        return len(self.patch_xyz_list)


    def __getitem__(self, index):
        ''' Get index-th element from the picked file dataset. '''
        # ------------------------------ INPUTS ----------------------------
        rot_angle = self.get_center_view_rot_angle(index)

        patch = self.patch_xyz_list[index]
        if self.add_rgb:
            rgb = self.patch_rgb_list[index]
            patch = np.concatenate((patch, rgb), axis=-1)

        self.rotate_to_center = True
        if self.rotate_to_center:
            patch[:, :, 0:3] = self.rotato_patch_to_center(patch[:, :, 0:3], rot_angle)

        # transport patch to fixed size
        # and change shape to C * H * W from H * W * C
        patch = torch.from_numpy(patch)                 # H * W * C
        patch = patch.unsqueeze(0)                      # 1 * H * W * C
        patch = patch.transpose(2, 3).transpose(1, 2)   # 1 * H * W * C -> 1 * H * C * W  ->  1 * C * H * W
        patch = F.interpolate(patch, self.patch_size, mode='bilinear', align_corners=True).squeeze(0).numpy()

        cls_type = self.type_list[index]
        assert(cls_type in ['Car', 'Pedestrian', 'Cyclist'])
        one_hot_vec = np.zeros((3), dtype=np.float32)
        one_hot_vec[self.dataset_helper.type2onehot[cls_type]] = 1

        if self.from_rgb_detection:
            return patch, rot_angle, self.prob_list[index], self.id_list[index], \
                   self.type_list[index], self.box2d_list[index], one_hot_vec

        # ------------------------------ LABELS ----------------------------
        # size labels
        size_class, size_residual = self.dataset_helper.size2class(self.box3d_size_list[index], self.type_list[index])

        # center labels
        center = self.box3d_center_list[index]
        if self.rotate_to_center:
            center = rotate_pc_along_y(np.expand_dims(center,0), self.get_center_view_rot_angle(index)).squeeze()

        # heading labels
        heading_angle = self.heading_list[index]
        if self.rotate_to_center:
            heading_angle = heading_angle - rot_angle

        if self.random_flip:
            if np.random.random() > 0.5: # 50% chance flipping
                patch[0, :, :] *= -1
                center[0] *= -1
                heading_angle = np.pi - heading_angle

        if self.random_shift:  # random shift object center
            dist = np.sqrt(np.sum(center[0]**2 + center[2]**2))
            shift = np.clip(np.random.randn() * dist * 0.2, -dist * 0.2, dist * 0.2)
            patch[2, :, :] += shift
            center[2] += shift

        angle_class, angle_residual = self.dataset_helper.angle2class(heading_angle)

        return patch, center, angle_class, angle_residual, size_class, size_residual, rot_angle, one_hot_vec


    def get_center_view_rot_angle(self, index):
        ''' Get the frustum rotation angle, it is shifted by pi/2 so that it
        can be directly used to adjust GT heading angle '''
        return np.pi/2.0 + self.frustum_angle_list[index]


    def rotato_patch_to_center(self, patch, angle):
        # Use np.copy to avoid corrupting original data
        w, h, c = patch.shape
        xyz = np.copy(patch).reshape(-1, 3)
        xyz_updated = rotate_pc_along_y(xyz, angle)
        patch_updated = xyz_updated.reshape(w, h, c)
        return patch_updated


if __name__ == '__main__':
    import yaml
    import matplotlib.pyplot as plt
    from lib.helpers.kitti_helper import Kitti_Config
    from torch.utils.data import DataLoader

    dataset_config = Kitti_Config()
    config = '../../cfgs/config_patchnet.yaml'
    assert (os.path.exists(config))
    cfg = yaml.load(open(config, 'r'), Loader=yaml.Loader)
    cfg = cfg['dataset']['train']
    cfg['pickle_file'] = '/Users/maxinzhu/Documents/GitHub/3DDetection/data/KITTI/pickle_files/patch_carpedcyc_train.pickle'
    dataset = PatchDataset(dataset_helper=dataset_config,
                           pickle_file=cfg['pickle_file'],
                           patch_size=[64, 64],
                           add_rgb = False,
                           random_flip=cfg['random_flip'],
                           random_shift=cfg['random_shift'])

    dataloader = DataLoader(dataset=dataset, batch_size=3)
    for i, data in enumerate(dataloader):

        patches = data[0].numpy()
        x_maps = patches[:, 0, :, :]
        y_maps = patches[:, 1, :, :]
        depth_maps = patches[:, 2, :, :]

        plt.subplot(3, 3, 1)
        plt.imshow(depth_maps[0])
        plt.subplot(3, 3, 2)
        plt.imshow(depth_maps[1])
        plt.subplot(3, 3, 3)
        plt.imshow(depth_maps[2])

        plt.subplot(3, 3, 4)
        plt.imshow(x_maps[0])
        plt.subplot(3, 3, 5)
        plt.imshow(x_maps[1])
        plt.subplot(3, 3, 6)
        plt.imshow(x_maps[2])

        plt.subplot(3, 3, 7)
        plt.imshow(y_maps[0])
        plt.subplot(3, 3, 8)
        plt.imshow(y_maps[1])
        plt.subplot(3, 3, 9)
        plt.imshow(y_maps[2])

        print('center:', data[1])
        print('angle_class:', data[2])
        print('angle_residual:', data[3])
        print('size_class:', data[4])
        print('size_residual:', data[5])
        print('rot_angle:', data[6])
        print('one_hot_vec:', data[7])

        plt.show()
        break


