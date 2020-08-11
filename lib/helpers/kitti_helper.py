import numpy as np


class Kitti_Config(object):
    def __init__(self):
        self.num_class = 8
        self.num_heading_bin = 12
        self.num_size_cluster = 8

        self.type_list = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']
        self.type2class={'Car':0, 'Van':1, 'Truck':2, 'Pedestrian':3, 'Person_sitting':4, 'Cyclist':5, 'Tram':6, 'Misc':7}
        self.type2onehot = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
        self.class2type = {self.type2class[t]:t for t in self.type2class}

        self.type_mean_size = {'Car': np.array([3.88311640418,1.62856739989,1.52563191462]),
                               'Van': np.array([5.06763659,1.9007158,2.20532825]),
                               'Truck': np.array([10.13586957,2.58549199,3.2520595]),
                               'Pedestrian': np.array([0.84422524,0.66068622,1.76255119]),
                               'Person_sitting': np.array([0.80057803,0.5983815,1.27450867]),
                               'Cyclist': np.array([1.76282397,0.59706367,1.73698127]),
                               'Tram': np.array([16.17150617,2.53246914,3.53079012]),
                               'Misc': np.array([3.64300781,1.54298177,1.92320313])}  # note this is dict
        self.mean_size_arr = np.zeros((8, 3), dtype=np.float32) # size clustrs
        for i in range(self.num_size_cluster):
            self.mean_size_arr[i] = self.type_mean_size[self.class2type[i]]


    def size2class(self, size, type_name):
        ''' Convert 3D bounding box size to template class and residuals '''
        size_class = self.type2class[type_name]
        size_residual = size - self.type_mean_size[type_name]
        return size_class, size_residual

    def class2size(self, cls, residual):
        ''' Inverse function to size2class '''
        # mean_size = self.type_mean_size[self.class2type[cls]]
        mean_size = self.mean_size_arr[cls]
        return mean_size + residual

    def angle2class(self, angle):
        ''' Convert continuous angle to discrete class and residual. '''
        angle = angle % (2 * np.pi)
        assert (angle >= 0 and angle <= 2*np.pi)
        angle_per_class = 2*np.pi / float(self.num_heading_bin)
        shifted_angle = (angle + angle_per_class/2) % (2 * np.pi)
        class_id = int(shifted_angle / angle_per_class)
        residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
        return class_id, residual_angle

    def class2angle(self, cls, residual, to_label_format=True):
        ''' Inverse function to angle2class. '''
        angle_per_class = 2 * np.pi / float(self.num_heading_bin)
        angle_center = cls * angle_per_class
        angle = angle_center + residual
        if to_label_format and angle > np.pi:
            angle = angle - 2 * np.pi
        return angle





if __name__ == '__main__':
    test = Kitti_Config()
    print(test.type_mean_size_arr)
    print(test.class2size(0, np.array([1.0, 1.0, 1.0])))
    print(test.size2class([5.8831164, 3.6285674, 3.52563191], 'Van'))
