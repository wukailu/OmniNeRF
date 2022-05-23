import random

import numpy
import numpy as np
import os
import pwd
import json
import pickle

base_dir = os.path.expanduser('~') + "/.foundations/job_data/archive/"
if os.path.exists(base_dir):
    dirs = os.listdir(base_dir)
else:
    dirs = []
hparams_cache = {}


def seed_name(train_param, artifact_name):
    if 'seed' in train_param:
        return artifact_name.split('.')[0] + f"_{train_param['seed']}." + artifact_name.split('.')[-1]
    else:
        return artifact_name.split('.')[0] + f"_RAND{random.randint(0, 99999)}." + artifact_name.split('.')[-1]


def batch_result_extract(path="../results", project_name="default", artifact_name='metric.pkl', name_gen=seed_name):
    os.makedirs(path, exist_ok=True)
    targets = get_targets(dict_filter({'project_name': project_name}))
    for folder in targets:
        import shutil
        src = os.path.join(folder, "user_artifacts/", artifact_name)
        hp = get_hparams(folder)
        if os.path.exists(src):
            try:
                dst = os.path.join(path, name_gen(hp, artifact_name))
                print(f"copy from {src} to {dst}")
                shutil.copy(src, dst)
            except:
                print("Warning: Copy failed!")


def gather_tensorboard_to(path="./tb", project_name="default", recent=0):
    os.makedirs(path, exist_ok=True)
    targets = get_targets(dict_filter({'project_name': project_name}))
    targets = sorted(targets, key=lambda x: os.path.getctime(x))
    if recent != 0:
        targets = targets[-recent:]
    for folder in targets:
        try:
            run_id = folder.split('/')[-1].split('-')[0]
            dst = os.path.join(path, run_id)
            if os.path.exists(dst):
                os.rmdir(dst)
            import shutil
            src = os.path.join(folder, "synced_directories/__tensorboard__")
            shutil.copytree(src, dst)
            print(f"copy from {src} to {dst}")
        except:
            print(f"copying {folder} failed!")


# TODO: add support for torch_stat

def update_dirs(base="/home/kailu/.foundations/job_data/archive/"):
    global dirs, hparams_cache, base_dir
    base_dir = base
    if os.path.exists(base_dir):
        dirs = os.listdir(base_dir)
    else:
        dirs = []
    hparams_cache = {}


def get_hparams(folder):
    if folder in hparams_cache:
        return hparams_cache[folder]

    jpath = os.path.join(folder, "artifacts", 'foundations_job_parameters.json')
    if os.path.exists(jpath):
        user = pwd.getpwuid(os.stat(jpath).st_uid).pw_name
        if user == 'root':
            with open(jpath, 'r') as f:
                params = json.load(f)
            if 'project_name' in params:
                hparams_cache[folder] = params
                return params
    hparams_cache[folder] = {'project_name': 'no_project'}
    return {'project_name': 'no_project'}


def get_artifacts(folder, name):
    from fnmatch import fnmatch
    prefix = os.path.join(folder, "user_artifacts")
    items = [i for i in os.listdir(prefix) if fnmatch(i, name)]
    if len(items) > 1:
        print("Warning! Multiple matched artifacts, the first is selected: ", ";".join(items))
    return os.path.join(prefix, items[0])


def pkl_load_artifacts(folder, name="test_result.pkl", sub_item='test/result'):
    with open(get_artifacts(folder, name), 'rb') as f:
        ret = pickle.load(f)
        if sub_item is not None:
            return ret[sub_item]
        return ret


def get_targets(param_filter, hole_range=None):
    if hole_range is None:
        hole_range = dirs
    hole_range = [d[len(base_dir):] if d.startswith(base_dir) else d for d in hole_range]
    return [base_dir + d for d in hole_range if param_filter(get_hparams(base_dir + d))]


def mean_results(param_filter):
    return np.mean([pkl_load_artifacts(t) for t in get_targets(param_filter)], axis=0)


def get_model_weight_hash(model):
    import hashlib
    d = frozenset({k: v.cpu().numpy() for k, v in model.state_dict().items()})
    return hashlib.sha256(str(d).encode()).hexdigest()


def all_list_to_tuple(my_dict):
    if isinstance(my_dict, dict):
        return {key: all_list_to_tuple(my_dict[key]) for key in my_dict}
    elif isinstance(my_dict, list) or isinstance(my_dict, tuple):
        return tuple(all_list_to_tuple(v) for v in my_dict)
    else:
        return my_dict


def dict_filter(filter_dict, net_name=None):
    def myfilter(params):
        params = all_list_to_tuple(params)
        for k in filter_dict:
            if k not in params or params[k] != filter_dict[k]:
                return False
        if net_name is not None and net_name not in params['pretrain_paths'][0]:
            return False
        return True

    return myfilter


def parse_params(params: dict):
    # Process trainer
    defaults = {
        'precision': 32,
        'deterministic': True,
        'benchmark': True,
        'gpus': 1,
        'num_epochs': 1,
        "progress_bar_refresh_rate": 100,
        'auto_select_gpus': False,
    }
    params = {**defaults, **params}
    if "backend" not in params:
        if params["gpus"] == 1:
            params["backend"] = None
        else:
            params["backend"] = "ddp"
            # params["backend"] = "ddp_spawn"  # Incompatible with atlas

    # Process backbone
    backbone_list = ['vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn', 'resnet18',
                     'resnet34', 'resnet50', 'resnet101', 'resnet152', 'densenet121',
                     'densenet161', 'densenet169', 'mobilenet_v2', 'googlenet', 'inception_v3',
                     'Rep_ResNet50', 'resnet20']

    if 'backbone' in params and isinstance(params['backbone'], int):
        params['backbone'] = backbone_list[params['backbone']]

    # Process dataset
    if isinstance(params['dataset'], str):
        params['dataset'] = {'name': params['dataset']}
    default_dataset_params = {
        'workers': 4,
    }
    params['dataset'] = {**default_dataset_params, **params['dataset']}
    if 'total_batch_size' in params['dataset'] and 'batch_size' not in params['dataset']:
        params['dataset']["batch_size"] = params['dataset']["total_batch_size"] // params["gpus"]
    if 'total_batch_size' not in params['dataset'] and 'batch_size' in params['dataset']:
        params['dataset']["total_batch_size"] = params['dataset']["batch_size"] * params["gpus"]

    # Process Training Settings
    optimizer_list = ['SGD', 'Adam']
    scheduler_list = ['ExpLR', 'CosLR', 'StepLR', 'OneCycLR', 'MultiStepLR', 'MultiStepLR_CRD']
    if 'optimizer' in params and isinstance(params['optimizer'], int):
        params['optimizer'] = optimizer_list[params['optimizer']]
    if 'lr_scheduler' in params and isinstance(params['lr_scheduler'], int):
        params['lr_scheduler'] = scheduler_list[params['lr_scheduler']]

    equivalent_keys = [('learning_rate', 'lr', 'max_lr')]
    for groups in equivalent_keys:
        for key in groups:
            if key in params:
                val = params[key]
                for key2 in groups:
                    params[key2] = val
                break

    return params


def get_trainer_params(params) -> dict:
    name_mapping = {
        "gpus": "gpus",
        "backend": "accelerator",
        "plugins": "plugins",
        "accumulate": "accumulate_grad_batches",
        "auto_scale_batch_size": "auto_scale_batch_size",
        "auto_select_gpus": "auto_select_gpus",
        "num_epochs": "max_epochs",
        "benchmark": "benchmark",
        "deterministic": "deterministic",
        "progress_bar_refresh_rate": "progress_bar_refresh_rate",
        "gradient_clip_val": "gradient_clip_val",
        "track_grad_norm": "track_grad_norm",
    }
    ret = {}
    for key in params:
        if key in name_mapping:
            ret[name_mapping[key]] = params[key]

    if ret['gpus'] != 0 and isinstance(ret['gpus'], int):
        ret['gpus'] = find_best_gpus(ret['gpus'])
        print('using gpu ', ret['gpus'])
    return ret


def submit_jobs(param_generator, command: str, number_jobs=1, project_name=None, job_directory='.',
                global_seed=23336666, ignore_exist=False):
    import time
    time.sleep(0.5)
    update_dirs()
    numpy.random.seed(global_seed)
    submitted_jobs = [{}]
    for idx in range(number_jobs):
        while True:
            ignore = ignore_exist
            hyper_params = param_generator()
            if 'ignore_exist' in hyper_params:
                ignore = hyper_params['ignore_exist']
                hyper_params.pop('ignore_exist')
            if (hyper_params not in submitted_jobs) and (
                    (not ignore) or len(get_targets(dict_filter(hyper_params))) == 0):
                break
        submitted_jobs.append(hyper_params.copy())

        if 'seed' not in hyper_params:
            hyper_params['seed'] = int(2018011328)
        if 'gpus' not in hyper_params:
            hyper_params['gpus'] = 1

        name = project_name if 'project_name' not in hyper_params else hyper_params['project_name']
        print('ready to submit')
        import utils.backend as backend
        backend.submit(scheduler_config='scheduler', job_directory=job_directory, command=command, params=hyper_params,
                       stream_job_logs=False, num_gpus=hyper_params["gpus"], project_name=name)

        print(f"Submit to {backend.name}, task {idx}, {hyper_params}")


def random_params(val):
    """
        use [x, y, z, ...] as the value of dict to use random select in the list.
        use (x, y, z, ...) to avoid random select or add '_no_choice' suffix to the key to avoid random for a list
        the function will recursively find [x,y,z,...] and select one element to replace it.
        :param params: dict for params
        :return: params after random choice
    """
    if isinstance(val, list):
        idx = np.random.randint(len(val))  # np.random.choice can't random rows
        ret = random_params(val[idx])
    elif isinstance(val, tuple):
        ret = tuple([random_params(i) for i in val])
    elif isinstance(val, dict):
        ret = {}
        for key, values in val.items():
            if isinstance(values, list) and key.endswith("_no_choice"):
                ret[key[:-10]] = values  # please use tuple to avoid be random selected
            else:
                ret[key] = random_params(values)
    elif isinstance(val, np.int64):
        ret = int(val)
    elif isinstance(val, np.float64):
        ret = float(val)
    else:
        ret = val
    return ret


def tuples_to_lists(val):
    if isinstance(val, list):
        ret = [tuples_to_lists(v) for v in val]
    elif isinstance(val, tuple):
        ret = [tuples_to_lists(i) for i in val]
    elif isinstance(val, dict):
        ret = {}
        for key, values in val.items():
            ret[key] = tuples_to_lists(values)
    elif isinstance(val, np.int64):
        ret = int(val)
    elif isinstance(val, np.float64):
        ret = float(val)
    else:
        ret = val
    return ret


def lists_to_tuples(val):
    if isinstance(val, list):
        ret = tuple([lists_to_tuples(v) for v in val])
    elif isinstance(val, tuple):
        ret = tuple([lists_to_tuples(i) for i in val])
    elif isinstance(val, dict):
        ret = {}
        for key, values in val.items():
            ret[key] = lists_to_tuples(values)
    elif isinstance(val, np.int64):
        ret = int(val)
    elif isinstance(val, np.float64):
        ret = float(val)
    else:
        ret = val
    return ret


def cnt_all_combinations(obj):
    comb = 1
    if isinstance(obj, list):
        comb = sum([cnt_all_combinations(i) for i in obj])
    elif isinstance(obj, tuple):
        for i in obj:
            comb *= cnt_all_combinations(i)
    elif isinstance(obj, dict):
        for key, values in obj.items():
            if isinstance(values, list) and key.endswith("_no_choice"):
                continue
            else:
                comb *= cnt_all_combinations(values)
    return comb


def find_best_gpus(num_gpu_needs=1):
    import subprocess as sp
    gpu_ids = []
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [(int(x.split()[0]), i) for i, x in enumerate(memory_free_info) if i not in gpu_ids]
    print('memories left ', memory_free_values)
    memory_free_values = sorted(memory_free_values)[::-1]
    gpu_ids = [k for m, k in memory_free_values[:num_gpu_needs]]
    return gpu_ids

# def summarize_result(exp_filter):
#     targets = get_targets(exp_filter)
#     assert len(targets) > 0
#     params = {t: get_hparams(t) for t in targets}
#     example = params[targets[0]]
#     for key, value in example.items():
#         all_same = True
#         for t in targets:
#             if params[t][key] != value:
#                 all_same = False
#                 break
#         if all_same:
#             for t in targets:
#                 params[t].pop(key)
#
