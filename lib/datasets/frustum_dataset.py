import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_DIR = os.path.dirname(BASE_DIR)
ROOT_DIR = os.path.dirname(LIB_DIR)
import pickle
import numpy as np
import torch.utils.data as data

from lib.utils.kitti.kitti_utils import rotate_pc_along_y



class FrustumDataset(data.Dataset):
    ''' Dataset class for Frustum PointNets training/evaluation.
    Load prepared KITTI data from pickled files, return individual data element
    [optional] along with its annotations.
    '''
    def __init__(self, dataset_helper, npoints,
                 pickle_file, random_flip=False,
                 random_shift=False, rotate_to_center=False,
                 from_rgb_detection=False, logger=None):
        '''
        Input:
            npoints: int scalar, number of points for frustum point cloud.
            random_flip: bool, in 50% randomly flip the point cloud
                in left and right (after the frustum rotation if any)
            random_shift: bool, if True randomly shift the point cloud
                back and forth by a random distance
            rotate_to_center: bool, whether to do frustum rotation
            from_rgb_detection: bool, if True we assume we do not have
                groundtruth, just return data elements.
            logger: logger
        '''

        self.npoints = npoints
        self.random_flip = random_flip
        self.random_shift = random_shift
        self.rotate_to_center = rotate_to_center
        self.from_rgb_detection = from_rgb_detection
        self.logger = logger
        self.pickle_file = pickle_file
        self.dataset_helper = dataset_helper


        if logger is not None:
            logger.info('Load data from %s' %(self.pickle_file))

        if self.from_rgb_detection:
            with open(self.pickle_file,'rb') as fp:
                self.id_list = pickle.load(fp)
                self.box2d_list = pickle.load(fp)
                self.input_list = pickle.load(fp)
                self.type_list = pickle.load(fp)
                self.frustum_angle_list = pickle.load(fp)
                self.prob_list = pickle.load(fp)
        else:
            with open(self.pickle_file, 'rb') as fp:
                self.id_list = pickle.load(fp)
                self.box2d_list = pickle.load(fp)
                self.box3d_list = pickle.load(fp)
                self.input_list = pickle.load(fp)
                self.label_list = pickle.load(fp)
                self.type_list = pickle.load(fp)
                self.heading_list = pickle.load(fp)
                self.size_list = pickle.load(fp)
                self.frustum_angle_list = pickle.load(fp)


    def __len__(self):
            return len(self.input_list)


    def __getitem__(self, index):
        ''' Get index-th element from the picked file dataset. '''
        # ------------------------------ INPUTS ----------------------------
        rot_angle = self.get_center_view_rot_angle(index)

        # Compute one hot vector
        cls_type = self.type_list[index]
        assert(cls_type in ['Car', 'Pedestrian', 'Cyclist'])
        one_hot_vec = np.zeros((3), dtype=np.float32)
        one_hot_vec[self.dataset_helper.type2onehot[cls_type]] = 1

        # Get point cloud
        if self.rotate_to_center:
            point_set = self.get_center_view_point_set(index)
        else:
            point_set = self.input_list[index]

        # Resample
        choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        point_set = point_set[choice, :]

        if self.from_rgb_detection:
            point_set = point_set.transpose(1, 0)  # N * C -> C * N
            return point_set, rot_angle, self.prob_list[index], self.id_list[index], \
                   self.type_list[index], self.box2d_list[index], one_hot_vec


        # ------------------------------ LABELS ----------------------------
        # segmentation labels
        seg = self.label_list[index]
        seg = seg[choice]

        # Get center point of 3D box
        if self.rotate_to_center:
            box3d_center = self.get_center_view_box3d_center(index)
        else:
            box3d_center = self.get_box3d_center(index)

        # Heading
        if self.rotate_to_center:
            heading_angle = self.heading_list[index] - rot_angle
        else:
            heading_angle = self.heading_list[index]

        # Size
        size_class, size_residual = self.dataset_helper.size2class(self.size_list[index], self.type_list[index])

        # Data Augmentation
        if self.random_flip:
            # note: rot_angle won't be correct if we have random_flip
            # so do not use it in case of random flipping.
            if np.random.random() > 0.5: # 50% chance flipping
                point_set[:, 0] *= -1
                box3d_center[0] *= -1
                heading_angle = np.pi - heading_angle

        if self.random_shift:  # random shift object center
            dist = np.sqrt(np.sum(box3d_center[0]**2+box3d_center[2]**2))
            shift = np.clip(np.random.randn()*dist*0.2, -dist*0.2, dist*0.2)
            point_set[:,2] += shift
            box3d_center[2] += shift

        angle_class, angle_residual = self.dataset_helper.angle2class(heading_angle)

        point_set = point_set.transpose(1, 0)  # N * C -> C * N
        return point_set, seg, box3d_center, angle_class, angle_residual,\
            size_class, size_residual, rot_angle, one_hot_vec



    def get_center_view_rot_angle(self, index):
        ''' Get the frustum rotation angle, it is shifted by pi/2 so that it
        can be directly used to adjust GT heading angle '''
        return np.pi/2.0 + self.frustum_angle_list[index]


    def get_center_view_point_set(self, index):
        ''' Frustum rotation of point clouds.
        NxC points with first 3 channels as XYZ
        z is facing forward, x is left ward, y is downward
        '''
        # Use np.copy to avoid corrupting original data
        point_set = np.copy(self.input_list[index])
        return rotate_pc_along_y(point_set, self.get_center_view_rot_angle(index))

    def get_box3d_center(self, index):
        ''' Get the center (XYZ) of 3D bounding box. '''
        box3d_center = (self.box3d_list[index][0, :] + self.box3d_list[index][6, :]) / 2.0
        return box3d_center

    def get_center_view_box3d_center(self, index):
        ''' Frustum rotation of 3D bounding box center. '''
        box3d_center = (self.box3d_list[index][0, :] + self.box3d_list[index][6, :]) / 2.0
        return rotate_pc_along_y(np.expand_dims(box3d_center,0), self.get_center_view_rot_angle(index)).squeeze()



if __name__ == '__main__':
    from lib.helpers.kitti_helper import Kitti_Config

    dataset = FrustumDataset(Kitti_Config(), 1000, '//data/KITTI/pickle_files/frustum_caronly_val.pickle',
                             True, True, True)
    data = dataset.__getitem__(10)




