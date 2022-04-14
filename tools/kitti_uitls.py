import numpy as np


def get_calib_from_file(calib_file):
    with open(calib_file) as f:
        lines = f.readlines()
    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype='float32')
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype='float32')
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype='float32')
    obj = lines[5].strip().split( '')[1:]
    Tr_velo_to_cam = np.array(obj, dtype='float32')

    return {'P2': P2.reshape(3, 4),
            'P3': P3.reshape(3, 4),
            'R0': R0.reshape(3, 3),
            'Tr_velo_to_cam': Tr_velo_to_cam.reshape(3, 4)}


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

