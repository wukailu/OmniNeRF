import argparse
import sys, os

sys.path = [os.getcwd()] + sys.path

from utils.tools import gather_tensorboard_to

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('project_name')
    parser.add_argument('--path', default='../tb')
    parser.add_argument('--recent', default=0, type=int)
    opt = parser.parse_args()
    gather_tensorboard_to(path=opt.path, project_name=opt.project_name, recent=opt.recent)
