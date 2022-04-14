import torch 
from torch.utils.data import Dataset
from kitti_uitls import Calibration
import os


class KittiDataset(Dataset):
    def __init__(self, root_dir, split='train', mode='TRAIN'):
        self.split = split
        self.mode = mode 
        self.classes = ['Car']
        is_test = self.split == 'test'
        self.imageset_dir = os.path.join(root_dir, 'kitti', 'testing' if is_test else 'training')

        split_dir = os.path.join(root_dir, 'kitti', 'ImageSets', split + '.txt')
        self.image_idx_list = [x.strip() for x in open(split_dir).readlines()]
        self.sample_idx_list = [int(x) for x in self.image_idx_list]
        self.num_sample = self.sample_idx_list.__len__()

        self.npoint = 16384
        self.calib_dir = os.path.join(self.imageset_dir, 'calib')
        self.lidar_dir = os.path.join(self.imageset_dir, 'velodyne')
        self.image_dir = os.path.join(self.imageset_dir, 'image_2')
        self.label_dir = os.path.join(self.imageset_dir, 'label_2')

    def get_calib(self, index):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' %index)
        assert os.path.exists(calib_file)
        return Calibration(calib_file)
        
        

    def __len__(self):
        return len(self.sample_idx_list)

    def __getitem__(self, index):
        sample_id = int(self.sample_idx_list[index])
        calib = self.get_calib(sample_id)
        img_shape = self.get_image_shape(sample_id)
        pts_lidar = self.get_lidar(sample_id)
        pts_rect = calib.lidar_to_rect(pts_lidar[:, :3])
        pts_indensity = pts_lidar[:, 3]

        pts_img, pts_rect_depth = calib.rect_to_img[pts_rect]
        pts_valid_flag = self.get_valid_flag(pts_rect, pts_img, pts_rect_depth, img_shape)

        pts_rect = pts_rect[pts_valid_flag][:, 0:3]
        pts_intensity = pts_intensity[pts_valid_flag]


