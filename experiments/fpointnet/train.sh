python ../../tools/train_val.py --config config_fpointnet.yaml
python ../../tools/train_val.py --config config_fpointnet.yaml --evaluate
../../tools/kitti_eval/evaluate_object_3d_offline_ap11 ../../data/KITTI/object/training/label_2 ./output
../../tools/kitti_eval/evaluate_object_3d_offline_ap40 ../../data/KITTI/object/training/label_2 ./output