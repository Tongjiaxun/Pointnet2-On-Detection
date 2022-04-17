from codecs import ignore_errors
import torch 
from torch.utils.data import Dataset
import kitti_uitls 
import cv2
import os
from PIL import Image
import numpy as np

USE_INTENSITY = False

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

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' %idx)
        assert os.path.exists(calib_file)
        return kitti_uitls.Calibration(calib_file)
    
    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        return cv2.imread(img_file)  # (H, W, 3) BGR mode
    
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

    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
        assert os.path.exists(label_file)
        return kitti_uitls.get_objects_from_label(label_file)

    @staticmethod
    def get_valid_flag(pts_rect, pts_img, pts_rect_depth, img_shape):
        val_flag_1 = np.logical_and(pts_rect[:, 0] >= 0, pts_rect[:, 0] <= img_shape[1])
        val_flag_2 = np.logical_and(pts_rect[:, 1] >= 0, pts_rect[:, 1] <= img_shape[0])    
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
        return pts_valid_flag

    def filtrate_objects(self, obj_list):
        type = self.classes
        if self.mode == 'TRAIN':
            type_list = list(type)
            if 'Car' in type_list:
                type_list.append('Van')
        
        valid_obj_list = []
        for obj in obj_list:
            if obj.cls_type not in type_list:
                continue            
            valid_obj_list.append(obj)
        return valid_obj_list
    
    @staticmethod
    def generate_training_labels(pts_rect, gt_boxes3d):
        cls_label = np.zeros((pts_rect.shape[0]), dtype=np.int32)
        gt_corners = kitti_uitls.boxes3d_to_corners3d(gt_boxes3d, rotate=True)
        extend_gt_boxes3d = kitti_uitls.enlarge_boxes3d(gt_boxes3d, extra_width=0.2)
        extend_gt_corners = kitti_uitls.boxes3d_to_corners3d(extend_gt_boxes3d, rotate=True)
        for k in range(gt_boxes3d.shape[0]):
            boxes_corner = gt_corners[k]
            inside_flag = kitti_uitls.in_boxes(pts_rect, boxes_corner)
            cls_label[inside_flag] = 1

            enlarge_boxes_corner = extend_gt_corners[k]
            enlarge_inside_flag =kitti_uitls.in_boxes(pts_rect, enlarge_boxes_corner)
            ignore_flag = np.logical_xor(inside_flag, enlarge_inside_flag)
            cls_label[ignore_flag] = -1
        return cls_label  
    def __len__(self):
        return len(self.sample_idx_list)

    def __getitem__(self, index):
        sample_id = int(self.sample_idx_list[index])
        calib = self.get_calib(sample_id)
        img_shape = self.get_image_shape(sample_id)
        pts_lidar = self.get_lidar(sample_id)
        pts_rect = calib.lidar_to_rect(pts_lidar[:, :3])
        pts_indensity = pts_lidar[:, 3]

        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        pts_valid_flag = self.get_valid_flag(pts_rect, pts_img, pts_rect_depth, img_shape)

        pts_rect = pts_rect[pts_valid_flag][:, 0:3]
        pts_indensity = pts_indensity[pts_valid_flag]

        if self.npoint < len(pts_rect):
            pts_dept = pts_rect[:, 2]
            pts_near_flag = pts_dept < 40
            far_idx_choice = np.where(pts_near_flag == 0)[0]
            near_idx = np.where(pts_near_flag == 1)[0]
            near_idx_choice = np.random.choice(near_idx, self.npoint - len(far_idx_choice), replace=False)
            choice = np.concatenate((near_idx_choice, far_idx_choice), axis=0)\
            if len(far_idx_choice) > 0 else near_idx_choice
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(pts_rect), dtype=np.int32)
            if self.npoint > len(pts_rect):
                extra_choice = np.random.choice(choice, self.npoint - len(pts_rect))
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        ret_pts_rect = pts_rect[choice, :]                                                                        ##最终获得点云
        ret_pts_indensity = pts_indensity[choice] - 0.5
        pts_feature = [ret_pts_indensity.reshape(-1, 1)]
        ret_pts_feature = np.concatenate(pts_feature, axis=1) if len(pts_feature) > 1 else pts_feature[0]      ##点云强度特征
        sample_info = {'sample_id' : sample_id}
        

        gt_obj_list = self.filtrate_objects(self.get_label(sample_id))                                         ##提取有效目标
        gt_boxes3d = kitti_uitls.objs_to_boxes3d(gt_obj_list)                                                  ##将目标转为3D框

  ######    准备输入数据   ######
        if USE_INTENSITY:
            pts_input = np.concatenate((ret_pts_rect, ret_pts_indensity), axis=1)
        else:
            pts_input = ret_pts_rect
        
  ######    生成训练标签   ######       
        
        cls_labels = self.generate_training_labels(ret_pts_rect, gt_boxes3d)       
        sample_info['pts_input'] = pts_input
        sample_info['pts_rect'] = ret_pts_rect
        sample_info['cls_labels'] = cls_labels
        return sample_info

        




if __name__ == '__main__':
    root_dir = '/home/bai/Documents/TJX/Deeplearning/Pointnet2-On-Detection/dataset'
    dataset = KittiDataset(root_dir=root_dir, mode='TRAIN', split='train')
    dataset.__getitem__(3)
