import sys, os

sys.path = [os.getcwd()] + sys.path

from utils.tools import submit_jobs, random_params
from copy import deepcopy

templates = {
    'standford': {
    },
}


def StandFordArea3():
    files = os.listdir('/data/Standford/area_3/pano/rgb/')
    file = random_params(files)
    params = templates['standford']
    iters = 30000
    params.update({
        'project_name': "Area3",
        'iters': iters,
        'i_weights': iters,
        'i_testset': iters,
        'i_video': iters,
        "rgb_path": '/data/Standford/area_3/pano/rgb/' + file,
        "depth_path": '/data/Standford/area_3/pano/depth/' + file.replace("_rgb", "_depth"),
    })
    return random_params(params)


def params_for_Nerf():
    params = StandFordArea3()
    return params


if __name__ == "__main__":
    submit_jobs(params_for_Nerf, 'run_nerf.py', number_jobs=88, job_directory='.')
