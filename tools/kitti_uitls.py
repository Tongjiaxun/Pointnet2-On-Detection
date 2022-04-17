import numpy as np
from scipy.spatial import Delaunay
import scipy

def cls_type_to_id(cls_type):
    type_to_id = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Van': 4}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]


class Object3d(object):
    def __init__(self, line):
        label = line.strip().split(' ')
        self.src = line
        self.cls_type = label[0]
        self.cls_id = cls_type_to_id(self.cls_type)
        self.trucation = float(label[1])
        self.occlusion = float(label[2])  # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown
        self.alpha = float(label[3])
        self.box2d = np.array((float(label[4]), float(label[5]), float(label[6]), float(label[7])), dtype=np.float32)
        self.h = float(label[8])
        self.w = float(label[9])
        self.l = float(label[10])
        self.pos = np.array((float(label[11]), float(label[12]), float(label[13])), dtype=np.float32)
        self.dis_to_cam = np.linalg.norm(self.pos)
        self.ry = float(label[14])
        self.score = float(label[15]) if label.__len__() == 16 else -1.0
        self.level_str = None
        self.level = self.get_obj_level()

    def get_obj_level(self):
        height = float(self.box2d[3]) - float(self.box2d[1]) + 1

        if height >= 40 and self.trucation <= 0.15 and self.occlusion <= 0:
            self.level_str = 'Easy'
            return 1  # Easy
        elif height >= 25 and self.trucation <= 0.3 and self.occlusion <= 1:
            self.level_str = 'Moderate'
            return 2  # Moderate
        elif height >= 25 and self.trucation <= 0.5 and self.occlusion <= 2:
            self.level_str = 'Hard'
            return 3  # Hard
        else:
            self.level_str = 'UnKnown'
            return 4

def get_calib_from_file(calib_file):
    with open(calib_file) as f:
        lines = f.readlines()
    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype='float32')
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype='float32')
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype='float32')
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype='float32')

    return {'P2': P2.reshape(3, 4),
            'P3': P3.reshape(3, 4),
            'R0': R0.reshape(3, 3),
            'Tr_velo_to_cam': Tr_velo_to_cam.reshape(3, 4)}

def get_objects_from_label(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    objects = [Object3d(line) for line in lines]
    return objects

class Calibration(object):
    def __init__(self, calib_file):
        if isinstance(calib_file, str):
            calib = get_calib_from_file(calib_file)
        else:
            calib = calib_file
        self.P2 = calib['P2']
        self.P3 = calib['P3']
        self.R0 = calib['R0']
        self.V2C = calib['Tr_velo_to_cam']


    def cart_to_hom(self, pts):    ####  为矩阵扩充维度
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def lidar_to_rect(self, pts_lidar):
        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        pts_rect = np.dot(pts_lidar_hom, np.dot(self.V2C.T, self.R0.T))
        return pts_rect
    
    def rect_to_img(self, pts_rect):
        pts_rect_hom = self.cart_to_hom(pts_rect)
        pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)
        pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
        pts_rect_depth = pts_2d_hom[:, 2] - self.P2.T[3, 2]  # depth in rect camera coord
        return pts_img, pts_rect_depth

    def lidar_to_img(self, pts_lidar):

        pts_rect = self.lidar_to_rect(pts_lidar)
        pts_img, pts_depth = self.rect_to_img(pts_rect)
        return pts_img, pts_depth    
    
def objs_to_boxes3d(obj_list):
    box3d = np.zeros((obj_list.__len__(), 7), dtype=np.float32)
    for k, obj in enumerate(obj_list):
        box3d[k, 0:3], box3d[k, 3], box3d[k, 4], box3d[k, 5], box3d[k, 6] = \
        obj.pos, obj.l, obj.w, obj.h, obj.ry    
    return box3d 

def boxes3d_to_corners3d(boxes3d, rotate=True):                                 ##############   相机坐标系   ############   x : right  y : down  z : forward
    boxes_num = len(boxes3d)
    h, w, l = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]
    x_corners = np.array([l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2.], dtype=np.float32).T
    z_corners = np.array([w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.], dtype=np.float32).T

    y_corners = np.zeros((boxes_num, 8), dtype=np.float32)
    y_corners[:, 4:8] = -h.reshape(boxes_num, 1).repeat(4, axis=1)

    if rotate:
        ry = boxes3d[:, 6]
        zeros, ones = np.zeros(ry.size, dtype=np.float32), np.ones(ry.size, dtype=np.float32)
        metric = np.array([[np.cos(ry), zeros, -np.sin(ry)],
                           [zeros,       ones,  zeros],
                           [np.sin(ry), zeros, np.cos(ry)]])  ##(3, 3, N)
        rotate_metric = np.transpose(metric, (2, 0, 1))
        tmp_corners = np.concatenate((x_corners.reshape(-1, 8, 1), y_corners.reshape(-1, 8, 1), z_corners.reshape(-1, 8, 1)), axis=2)
        rotate_corners = np.matmul(tmp_corners, rotate_metric)
        x_corners = rotate_corners[:, :, 0]
        y_corners = rotate_corners[:, :, 1]
        z_corners = rotate_corners[:, :, 2]
    
    x_center, y_center, z_center = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]
    x = x_center.reshape(-1, 1) + x_corners.reshape(-1, 8)
    y = y_center.reshape(-1, 1) + y_corners.reshape(-1, 8)
    z = z_center.reshape(-1, 1) + z_corners.reshape(-1, 8)

    corner = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1), z.reshape(-1, 8, 1)), axis=2)
    return corner.astype(np.float32)

def enlarge_boxes3d(boxes3d, extra_width=0.2):
    """
    Args:
        boxes3d (x, y, z, h, w, l, ry)
    """
    if isinstance(boxes3d, np.ndarray):
        large_boxes = boxes3d.copy()
    else:
        large_boxes = boxes3d.clone()

    large_boxes[:, 3:6] += extra_width * 2
    large_boxes[:, 1] += extra_width

    return large_boxes

def in_boxes(pts_rect, boxes_corner):

    boxes_num = 1

    x_min = np.min(boxes_corner[:, 0])
    x_max = np.max(boxes_corner[:, 0])
    y_min = np.min(boxes_corner[:, 1])
    y_max = np.max(boxes_corner[:, 1])
    z_min = np.min(boxes_corner[:, 2])
    z_max = np.max(boxes_corner[:, 2])
    ones = np.ones((pts_rect.shape[0], boxes_num), dtype=np.int32)
    temp_flag = np.zeros((pts_rect.shape[0], boxes_num), dtype=np.int32)
    for index, pts in enumerate(pts_rect):
        if x_min < pts[0] < x_max and y_min < pts[1] < y_max and z_min < pts[2] < z_max:
            temp_flag[index] = 1
        else:
            temp_flag[index] = 0
    return np.concatenate(np.logical_and(list(temp_flag), list(ones)))
# def in_boxes(p, hull):
#     """
#     :param p: (N, K) test points
#     :param hull: (M, K) M corners of a box
#     :return (N) bool
#     """
#     try:
#         if not isinstance(hull, Delaunay):
#             hull = Delaunay(hull)
#         flag = hull.find_simplex(p) >= 0
#     except scipy.spatial.qhull.QhullError:
#         print('Warning: not a hull %s' % str(hull))
#         flag = np.zeros(p.shape[0], dtype=np.bool)

#     return flag

