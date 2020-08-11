import os
import torch
import numpy as np
from torch.nn import functional as F
import tqdm

from lib.helpers.save_helper import load_checkpoint

# TODO: update the following lib
from lib.utils.fpointnet_utils import write_detection_results
from lib.utils.fpointnet_utils import fill_files


class Tester(object):
    def __init__(self, cfg_tester, model, test_dataloader, logger):
        self.cfg = cfg_tester
        self.model = model
        self.dataloader = test_dataloader
        self.logger = logger
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def test(self):
        load_checkpoint(model=self.model,
                        optimizer=None,
                        filename=self.cfg['resume_model'],
                        logger=self.logger)
        self.test_one_epoch()


    def test_one_epoch(self):
        torch.set_grad_enabled(False)
        self.model.eval()

        center_list = []
        heading_cls_list = []
        heading_res_list = []
        size_cls_list = []
        size_res_list = []
        rot_angle_list = []
        score_list = []
        id_list = []
        type_list = []
        box2d_list = []

        progress_bar = tqdm.tqdm(total=len(self.dataloader), leave=True, desc='eval')
        for _, data in enumerate(self.dataloader):
            assert self.cfg['format'] in ['fpointnet', 'patchnet']
            if self.cfg['format'] == 'fpointnet':
                input, rot_angle, rgb_prob, id, type, box2d, one_hot_vec = data

            if self.cfg['format'] == 'patchnet':
                input, rot_angle, rgb_prob, id, type, box2d, one_hot_vec = data
            # TODO: if rotate_to_center is False, uncomment the following code
            #rot_angle = torch.zeros_like(rgb_prob)

            # model inference
            outputs = self.model(input.cuda(), one_hot_vec.cuda())
            batch_size = input.shape[0]

            outputs['center'] = outputs['center'].cpu().numpy()
            outputs['heading_scores'] = outputs['heading_scores'].cpu().numpy()
            outputs['heading_residuals'] = outputs['heading_residuals'].cpu().numpy()
            outputs['size_scores'] = outputs['size_scores'].cpu().numpy()
            outputs['size_residuals'] = outputs['size_residuals'].cpu().numpy()

            rot_angle = rot_angle.numpy()
            rgb_prob = rgb_prob.numpy()
            id = id.numpy()
            box2d = box2d.numpy()

            for i in range(batch_size):
                center_list.append(outputs['center'][i, :])
                heading_cls = np.argmax(outputs['heading_scores'][i, :])
                heading_cls_list.append(heading_cls)
                heading_res = outputs['heading_residuals'][i, heading_cls]
                heading_res_list.append(heading_res)
                size_cls = np.argmax(outputs['size_scores'][i, :])
                size_cls_list.append(size_cls)
                size_res = outputs['size_residuals'][i][size_cls]
                size_res_list.append(size_res)
                rot_angle_list.append(rot_angle[i])
                score_list.append(rgb_prob[i])  # 2D RGB detection score
                id_list.append(id[i])
                type_list.append(type[i])
                box2d_list.append(box2d[i])

            progress_bar.update()
        progress_bar.close()

        self.logger.info('Write detection results for KITTI evaluation')

        result_dir = './output'
        os.makedirs('./output', exist_ok=True)
        write_detection_results(result_dir=result_dir,
                                id_list=id_list,
                                type_list=type_list,
                                box2d_list=box2d_list,
                                center_list=center_list,
                                heading_cls_list=heading_cls_list,
                                heading_res_list=heading_res_list,
                                size_cls_list=size_cls_list,
                                size_res_list=size_res_list,
                                rot_angle_list=rot_angle_list,
                                score_list=score_list)

        # Make sure for each frame (no matter if we have measurment for that frame),
        # there is a TXT file
        output_dir = os.path.join(result_dir, 'data')
        split_idx_path = self.cfg['files_fill_set']
        to_fill_filename_list = [line.rstrip() + '.txt' for line in open(split_idx_path)]
        fill_files(output_dir, to_fill_filename_list)


