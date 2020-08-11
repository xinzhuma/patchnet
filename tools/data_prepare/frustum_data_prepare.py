'''
data generation script for FPointNet, pseudo-LiDAR and AM3D
'''

import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOOL_DIR = os.path.dirname(BASE_DIR)
ROOT_DIR = os.path.dirname(TOOL_DIR)
sys.path.append(ROOT_DIR)
import argparse
import tqdm
import pickle
import numpy as np
from lib.datasets.kitti_dataset import KittiDataset
import lib.utils.kitti.kitti_utils as kitti_utils


data_root = os.path.join(ROOT_DIR, 'data/KITTI/object')
parser = argparse.ArgumentParser()
parser.add_argument('--gen_train', action='store_true', help='Generate train split frustum data [from GT]')
parser.add_argument('--gen_val', action='store_true', help='Generate val split frustum data [from GT]')
parser.add_argument('--gen_val_rgb_detection', action='store_true', help='Generate frustum val split frustum data [from 2d detections]')
parser.add_argument('--model', default='fpointnet', help='fpointnet, pseudo_lidar or am3d')
parser.add_argument('--car_only', action='store_true', help='Only generate frustum data from car instances')
parser.add_argument('--pseudo_lidar', action='store_true', help='Generate frustum data from pseudo lidar')
args = parser.parse_args()


def random_shift_box2d(box2d, shift_ratio=0.1):
    ''' Randomly shift box center, randomly scale width and height
    '''
    r = shift_ratio
    xmin,ymin,xmax,ymax = box2d
    h = ymax-ymin
    w = xmax-xmin
    cx = (xmin+xmax)/2.0
    cy = (ymin+ymax)/2.0
    cx2 = cx + w*r*(np.random.random()*2-1)
    cy2 = cy + h*r*(np.random.random()*2-1)
    h2 = h*(1+np.random.random()*2*r-r) # 0.9 to 1.1
    w2 = w*(1+np.random.random()*2*r-r) # 0.9 to 1.1
    return np.array([cx2-w2/2.0, cy2-h2/2.0, cx2+w2/2.0, cy2+h2/2.0])


def extract_frustum_data(split,
                         output_filename,
                         perturb_box2d=False,
                         augment_times=1,
                         whitelist=['Car']):
    ''' Extract point clouds and corresponding annotations in frustums
        defined generated from 2D bounding boxes
        (Pseudo) LiDAR points and 3d boxes are in *rect camera* coord system

    Input:
        split: string, either trianing or testing
        output_filename: string, the name for output .pickle file
        perturb_box2d: bool, whether to perturb the box2d (used for data augmentation in train set)
        augmentX: scalar, how many augmentations to have for each 2D box.
        whitelist: a list of strings, object types we are interested in.
    Output:
        None (will write a .pickle file to the disk)
    '''
    data_dir = os.path.join(ROOT_DIR, 'data')
    dataset = KittiDataset(root_dir=data_dir, split=split)

    id_list = [] # int number
    box2d_list = [] # [xmin, ymin, xmax, ymax]
    box3d_list = [] # (8, 3) array in rect camera coord
    input_list = [] # channel number = 3/4/6, xyz/xyzi/xyzrgb
    label_list = [] # 1 for roi object, 0 for clutter
    type_list  = [] # string e.g. Car
    heading_list = [] # ry (along y-axis in rect camera coord) radius of
                      # (cont.) clockwise angle from positive x axis in velo coord.
    box3d_size_list = [] # array of l, w, h
    frustum_angle_list = [] # angle of 2d box center from pos x-axis

    pos_cnt = 0
    all_cnt = 0

    progress_bar = tqdm.tqdm(total=len(dataset.idx_list), leave=True, desc='%s split frustum data gen' % split)
    #dataset.idx_list = dataset.idx_list[0:20] # for code testing
    for data_idx in dataset.idx_list:
        data_idx = int(data_idx)
        calib = dataset.get_calib(data_idx)
        objects = dataset.get_label(data_idx)
        if args.model == 'pseudo_lidar':
            pc_lidar = dataset.get_pseudo_lidar(data_idx, 6) # six channels (x, y, z, r, g, b)
            pc_lidar = pc_lidar[:, 0:3] # only extract xyz channels for 'pseudo_lidar'
        else: # fpointnet
            pc_lidar = dataset.get_lidar(data_idx)
        pc_rect = np.zeros_like(pc_lidar)
        pc_rect[:, 0:3] = calib.lidar_to_rect(pc_lidar[:, 0:3])
        if pc_lidar.shape[1] > 2:
            pc_rect[:, 3:] = pc_lidar[:, 3:]
        heigth, width, channel = dataset.get_image_shape(data_idx)
        pc_img, pc_rect_depth = calib.rect_to_img(pc_rect[:, 0:3])

        gt_boxes3d = kitti_utils.objs_to_boxes3d(objects)
        gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes3d, rotate=True)

        for obj_idx in range(len(objects)):
            if objects[obj_idx].cls_type not in whitelist:
                continue

            # 2D BOX
            box2d = objects[obj_idx].box2d
            for _ in range(augment_times):
                # Augment data by box2d perturbation
                if perturb_box2d:
                    xmin, ymin, xmax, ymax = random_shift_box2d(box2d)
                    # print(box2d)
                    # print(xmin,ymin,xmax,ymax)
                else:
                    xmin, ymin, xmax, ymax = box2d

                # get points in box fov
                valid_flag_1 = np.logical_and(pc_img[:, 0] >= xmin, pc_img[:, 0] < xmax)
                valid_flag_2 = np.logical_and(pc_img[:, 1] >= ymin, pc_img[:, 1] < ymax)
                box_fov_inds = np.logical_and(valid_flag_1, valid_flag_2)
                # get points in image fov
                valid_flag_3 = np.logical_and(pc_img[:, 0] >= 0, pc_img[:, 0] < width)
                valid_flag_4 = np.logical_and(pc_img[:, 1] >= 0, pc_img[:, 1] < heigth)
                img_fov_indx = np.logical_and(valid_flag_3, valid_flag_4)
                img_fov_indx = np.logical_and(img_fov_indx, pc_rect_depth >= 0)

                box_fov_inds = np.logical_and(box_fov_inds, img_fov_indx)
                pc_in_box_fov = pc_rect[box_fov_inds, :]

                # Get frustum angle (according to center pixel in 2D BOX)
                box2d_center = np.array([(xmin+xmax)/2.0, (ymin+ymax)/2.0])
                uvdepth = np.zeros((1, 3))
                uvdepth[0, 0:2] = box2d_center
                uvdepth[0, 2] = 20  # some random depth
                box2d_center_rect = calib.img_to_rect(uvdepth[:, 0], uvdepth[:, 1], uvdepth[:, 2])
                frustum_angle = -1 * np.arctan2(box2d_center_rect[0,2], box2d_center_rect[0,0])

                # 3D box: get pts velo in 3d box
                obj = objects[obj_idx]
                fg_pt_flag = kitti_utils.in_hull(pc_in_box_fov[:, 0:3], gt_corners[obj_idx])
                label = np.zeros((pc_in_box_fov.shape[0]), dtype=np.float32)
                label[fg_pt_flag] = 1
                # get 3D box heading
                heading_angle = obj.ry
                # get 3D box size
                box3d_size = np.array([obj.l, obj.w, obj.h])

                # Reject too far away object or object without points
                if obj.level > 3 or np.sum(label)==0:
                    continue

                id_list.append(data_idx)
                box2d_list.append(np.array([xmin,ymin,xmax,ymax]))
                box3d_list.append(gt_corners[obj_idx])
                input_list.append(pc_in_box_fov)
                label_list.append(label)
                type_list.append(objects[obj_idx].cls_type)
                heading_list.append(heading_angle)
                box3d_size_list.append(box3d_size)
                frustum_angle_list.append(frustum_angle) #

                # collect statistics
                pos_cnt += np.sum(label)
                all_cnt += pc_in_box_fov.shape[0]

        progress_bar.update()
    progress_bar.close()

    print('Average pos ratio: %f' % (pos_cnt/float(all_cnt)))
    print('Average npoints: %f' % (float(all_cnt)/len(id_list)))

    with open(output_filename,'wb') as fp:
        pickle.dump(id_list, fp)
        pickle.dump(box2d_list,fp)
        pickle.dump(box3d_list,fp)
        pickle.dump(input_list, fp)
        pickle.dump(label_list, fp)
        pickle.dump(type_list, fp)
        pickle.dump(heading_list, fp)
        pickle.dump(box3d_size_list, fp)
        pickle.dump(frustum_angle_list, fp)


def read_det_file(det_filename):
    ''' Parse lines in 2D detection output files '''
    det_id2str = {1: 'Pedestrian', 2: 'Car', 3: 'Cyclist'}
    id_list = []
    type_list = []
    prob_list = []
    box2d_list = []
    for line in open(det_filename, 'r'):
        t = line.rstrip().split(' ')
        id_list.append(int(os.path.basename(t[0]).rstrip('.png')))
        type_list.append(det_id2str[int(t[1])])
        prob_list.append(float(t[2]))
        box2d_list.append(np.array([float(t[i]) for i in range(3, 7)]))
    return id_list, type_list, box2d_list, prob_list


def extract_frustum_data_rgb_detection(det_filename,
                                       split,
                                       output_filename,
                                       whitelist=['Car'],
                                       img_height_threshold=25,
                                       lidar_point_threshold=5):
    ''' Extract point clouds in frustums extruded from 2D detection boxes.
        Update: Lidar points and 3d boxes are in *rect camera* coord system
            (as that in 3d box label files)

    Input:
        det_filename: string, each line is img_path typeid confidence xmin ymin xmax ymax
        split: string, either trianing or testing
        output_filename: string, the name for output .pickle file
        whitelist: a list of strings, object types we are interested in.
        img_height_threshold: int, neglect image with height lower than that.
        lidar_point_threshold: int, neglect frustum with too few points.
    Output:
        None (will write a .pickle file to the disk)
    '''
    data_dir = os.path.join(ROOT_DIR, 'data')
    dataset = KittiDataset(root_dir=data_dir, split=split)
    det_id_list, det_type_list, det_box2d_list, det_prob_list = read_det_file(det_filename)
    cache_id = -1
    cache = None

    id_list = []
    type_list = []
    box2d_list = []
    prob_list = []
    input_list = []
    frustum_angle_list = []  # angle of 2d box center from pos x-axis

    progress_bar = tqdm.tqdm(total=len(det_id_list), leave=True, desc='%s split frustum data gen (from 2d detections)' % split)
    det_id_list = det_id_list[0:20]
    for det_idx in range(len(det_id_list)):
        # ignore object belong to other categories
        if det_type_list[det_idx] not in whitelist:
            progress_bar.update()
            continue

        data_idx = det_id_list[det_idx]
        heigth, width, channel = dataset.get_image_shape(data_idx)  # update image shape

        if cache_id != data_idx:
            calib = dataset.get_calib(data_idx)
            if args.model == 'pseudo_lidar':
                pc_lidar = dataset.get_pseudo_lidar(data_idx, 6)
                pc_lidar = pc_lidar[:, 0:3]
            else:
                pc_lidar = dataset.get_lidar(data_idx)

            pc_rect = np.zeros_like(pc_lidar)
            pc_rect[:, 0:3] = calib.lidar_to_rect(pc_lidar[:, 0:3])
            if pc_lidar.shape[1] > 2:
                pc_rect[:, 3:] = pc_lidar[:, 3:]
            pc_img, pc_rect_depth = calib.rect_to_img(pc_rect[:, 0:3])

            cache = [calib, pc_rect, pc_img]
            cache_id = data_idx
        else:
            calib, pc_rect, pc_img = cache


        # 2D BOX: Get pts rect backprojected
        xmin, ymin, xmax, ymax = det_box2d_list[det_idx]

        # get frustum point index
        valid_flag_1 = np.logical_and(pc_img[:, 0] >= xmin, pc_img[:, 0] < xmax)
        valid_flag_2 = np.logical_and(pc_img[:, 1] >= ymin, pc_img[:, 1] < ymax)
        box_fov_inds = np.logical_and(valid_flag_1, valid_flag_2)

        valid_flag_3 = np.logical_and(pc_img[:, 0] >= 0, pc_img[:, 0] < width)
        valid_flag_4 = np.logical_and(pc_img[:, 1] >= 0, pc_img[:, 1] < heigth)
        img_fov_indx = np.logical_and(valid_flag_3, valid_flag_4)
        img_fov_indx = np.logical_and(img_fov_indx, pc_rect_depth >= 0)

        box_fov_inds = np.logical_and(box_fov_inds, img_fov_indx)
        pc_in_box_fov = pc_rect[box_fov_inds, :]

        # Get frustum angle (according to center pixel in 2D BOX)
        box2d_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
        uvdepth = np.zeros((1, 3))
        uvdepth[0, 0:2] = box2d_center
        uvdepth[0, 2] = 20  # some random depth
        box2d_center_rect = calib.img_to_rect(uvdepth[:, 0], uvdepth[:, 1], uvdepth[:, 2])
        frustum_angle = -1 * np.arctan2(box2d_center_rect[0, 2], box2d_center_rect[0, 0])

        # Pass objects that are too small
        if ymax - ymin < img_height_threshold or len(pc_in_box_fov) < lidar_point_threshold:
            progress_bar.update()
            continue

        id_list.append(data_idx)
        type_list.append(det_type_list[det_idx])
        box2d_list.append(det_box2d_list[det_idx])
        prob_list.append(det_prob_list[det_idx])
        input_list.append(pc_in_box_fov)
        frustum_angle_list.append(frustum_angle)

        progress_bar.update()
    progress_bar.close()

    with open(output_filename, 'wb') as fp:
        pickle.dump(id_list, fp)
        pickle.dump(box2d_list, fp)
        pickle.dump(input_list, fp)
        pickle.dump(type_list, fp)
        pickle.dump(frustum_angle_list, fp)
        pickle.dump(prob_list, fp)


if __name__ == '__main__':
    # setting
    whitelist = ['Car'] if args.car_only else ['Car', 'Pedestrian', 'Cyclist']
    assert args.model in ['fpointnet', 'pseudo_lidar']
    output_prefix = args.model
    output_prefix += '_caronly' if args.car_only else '_carpedcyc'

    train_filename = output_prefix + '_train.pickle'
    val_filename = output_prefix + '_val.pickle'
    val_detections_filename = output_prefix + '_val_rgb_detection.pickle'
    rgb_detections = os.path.join(ROOT_DIR, 'data/KITTI/2d_detections/fpointnet/rgb_detection_val.txt')

    if args.gen_train:
        extract_frustum_data(split = 'train',
                             output_filename = train_filename,
                             perturb_box2d = True,
                             augment_times = 5,
                             whitelist = whitelist)
    if args.gen_val:
        extract_frustum_data(split = 'val',
                             output_filename = val_filename,
                             perturb_box2d = False,
                             augment_times = 1,
                             whitelist = whitelist)

    if args.gen_val_rgb_detection:
        extract_frustum_data_rgb_detection(det_filename = rgb_detections,
                                           split = 'val',
                                           output_filename = val_detections_filename,
                                           whitelist = whitelist)
