import _init_path
import argparse
import importlib
import sys
from dataset import KittiDataset

sys.path.append("..")

def parse_config():
    parser = argparse.ArgumentParser(description="Arg parser") 
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--ckpt", type=None)

    parser.add_argument("--net", type=str, default='pointnet2_msg')
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--lr_decay", type=float, default=0.2)
    parser.add_argument("--lr_clip", type=float, default=0.000001)
    parser.add_argument('--decay_step_list', type=list, default=[50, 70, 80, 90])
    parser.add_argument('--weight_decay', type=float, default=0.001)

    parser.add_argument("--output_dir", type=str, default='output')
    parser.add_argument("--extra_tag", type=str, default='default')

    args = parser.parse_args()

    return args





def main():
    args = parse_config()
    MODEL = importlib.import_module(args.net)
    model = MODEL.create_model(input_channels=0)
    eval_set = KittiDataset(root_dir='/home/bai/Documents/TJX/Deeplearning/Pointnet2-On-Detection/dataset', mode='EVAL', split='val')







if __name__ == '__main__':
    main()
