import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOOL_DIR = os.path.dirname(BASE_DIR)
ROOT_DIR = os.path.dirname(TOOL_DIR)
sys.path.append(ROOT_DIR)
import numpy as np
import argparse
import tqdm
from PIL import Image
from lib.utils.kitti.calibration import Calibration

data_root = os.path.join(ROOT_DIR, 'data/KITTI/object')
parser = argparse.ArgumentParser()
parser.add_argument('--gen_train', action='store_true', help='Generate train split pseudo lidar [train and val]')
parser.add_argument('--gen_test', action='store_true', help='Generate test split pseudo lidar')
parser.add_argument('--sampling', action='store_true', help='sample dense points or not')
parser.add_argument('--sampling_rate', type=float, default=0.5, help='number of samples')
parser.add_argument('--vis_test', action='store_true',
                    help='visulize pesudo lidar [default case: 000001.bin in training set]')
args = parser.parse_args()


def depth2points(tag, total_files):
    img_path_prefix = os.path.join(data_root, tag, 'image_2')
    depth_path_prefix = os.path.join(data_root, tag, 'depth')
    calib_path_prefix = os.path.join(data_root, tag, 'calib')
    output_path_prefix = os.path.join(data_root, tag, 'pseudo_lidar')

    progress_bar = tqdm.tqdm(total=total_files, leave=True, desc='%s data generation' % tag)

    for i in range(total_files):
        depth_path = os.path.join(depth_path_prefix, '%06d.png' % i)
        calib_path = os.path.join(calib_path_prefix, '%06d.txt' % i)
        output_path = os.path.join(output_path_prefix, '%06d.bin' % i)

        depth = np.array(Image.open(depth_path)).astype(np.float32)
        depth = depth / 256
        calib = Calibration(calib_path)
        height, width = depth.shape
        uvdepth = np.zeros((height, width, 3), dtype=np.float32)

        # add RGB values
        img_path = os.path.join(img_path_prefix, '%06d.png' % i)
        img = np.array(Image.open(img_path)).astype(np.float32)
        assert img.shape == uvdepth.shape, 'RGB image and depth map should have the same shape'
        uvdepth = np.concatenate((uvdepth, img), axis=2)

        for v in range(height):
            for u in range(width):
                uvdepth[v, u, 0] = u
                uvdepth[v, u, 1] = v
        uvdepth[:, :, 2] = depth

        uvdepth = uvdepth.reshape(-1, 6)

        # sampling, to reduce the number of pseudo lidar points
        if args.sampling:
            num_points = uvdepth.shape[0]
            choice = np.random.choice(num_points, int(num_points * args.sampling_rate), replace=True)
            uvdepth = uvdepth[choice]

        points = calib.img_to_rect(uvdepth[:, 0], uvdepth[:, 1], uvdepth[:, 2])
        points = calib.rect_to_lidar(points)
        points = np.concatenate((points, uvdepth[:, 3:6]), -1)

        # remove points with heights larger than 1.0 meter
        idx = np.argwhere(points[:, 2] <= 1.0)
        points = points[idx, :].squeeze(1)

        points.tofile(output_path)

        progress_bar.update()
    progress_bar.close()


def vis_demo():
    import mayavi.mlab as mlab
    pseudo_lidar_path = os.path.join(data_root, 'testing', 'pseudo_lidar', '000001.bin')
    assert os.path.exists(pseudo_lidar_path)
    channels = 6 if args.add_rgb else 3
    pesudo_lidar = np.fromfile(pseudo_lidar_path, dtype=np.float32).reshape(-1, channels)

    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500))
    color = pesudo_lidar[:, 2]
    mlab.points3d(pesudo_lidar[:, 0], pesudo_lidar[:, 1], pesudo_lidar[:, 2], color, color=None,
                  mode='point', colormap='plasma', scale_factor=1, figure=fig)

    # draw origin
    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2)

    # draw axis
    # axes = np.array([
    #     [5., 0., 0., 0.],
    #     [0., 5., 0., 0.],
    #     [0., 0., 5., 0.],
    # ], dtype=np.float64)
    # mlab.plot3d([0, axes[0, 0]], [0, axes[0, 1]], [0, axes[0, 2]], color=(1, 0, 0), tube_radius=None, figure=fig)
    # mlab.plot3d([0, axes[1, 0]], [0, axes[1, 1]], [0, axes[1, 2]], color=(0, 1, 0), tube_radius=None, figure=fig)
    # mlab.plot3d([0, axes[2, 0]], [0, axes[2, 1]], [0, axes[2, 2]], color=(0, 0, 1), tube_radius=None, figure=fig)

    # draw fov
    # fov = np.array([  # 45 degree
    #     [10., 10., 0., 0.],
    #     [10., -10., 0., 0.],
    # ], dtype=np.float64)
    # mlab.plot3d([0, fov[0,0]], [0, fov[0,1]], [0, fov[0,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)
    # mlab.plot3d([0, fov[1,0]], [0, fov[1,1]], [0, fov[1,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)

    mlab.view(azimuth=180, elevation=75, focalpoint=[ 20.0909996 , -1.04700089, -2.03249991], distance=120.0, figure=fig)
    # mlab.savefig('pc_view.jpg', figure=fig)
    input()


if __name__ == '__main__':
    if args.gen_train:
        depth2points(tag='training',
                     total_files=7481)

    if args.gen_test:
        depth2points(tag='testing',
                     total_files=7518)

    if args.vis_test:
        vis_demo()

    vis_demo()

