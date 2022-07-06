import sys, os

sys.path = [os.getcwd()] + sys.path

from utils.tools import submit_jobs, random_params
from copy import deepcopy

templates = {
    'standford': {
    },
}


def StandFordArea():
    area_id = '2'
    files = os.listdir(f'/data/Standford/area_{area_id}/pano/rgb/')
    file = random_params(files)
    params = templates['standford']
    iters = 50000
    params.update({
        'project_name': f"Area{area_id}",
        'iters': iters,
        'i_weights': iters,
        'i_testset': iters,
        'i_video': iters,
        "rgb_path": f'/data/Standford/area_{area_id}/pano/rgb/' + file,
        "depth_path": f'/data/Standford/area_{area_id}/pano/depth/' + file.replace("_rgb", "_depth"),
    })
    return random_params(params)


def params_for_Nerf():
    params = StandFordArea()
    return params


if __name__ == "__main__":
    submit_jobs(params_for_Nerf, 'run_nerf.py', number_jobs=300, job_directory='.')
