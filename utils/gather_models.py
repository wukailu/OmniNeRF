import argparse
import random
import sys, os

sys.path = [os.getcwd()] + sys.path

from utils.tools import batch_result_extract


def standford_name(train_param, artifact_name):
    if 'rgb_path' in train_param:
        return train_param['rgb_path'].split("/")[-1][:-len("_rgb.png")] + "." + artifact_name.split(".")[-1]
    else:
        return f"RAND{random.randint(0, 99999)}_" + artifact_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('project_name')
    parser.add_argument('--path', default='../results')
    parser.add_argument('--artifact', default="metric.pkl", type=str)
    opt = parser.parse_args()
    batch_result_extract(path=opt.path, project_name=opt.project_name,
                         artifact_name=opt.artifact, name_gen=standford_name)
