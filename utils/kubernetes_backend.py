job_info = {'params': {}, 'results': {}, 'tensorboard_path': '', 'artifacts': {}}
name = 'kube_backend'


def save_job_info():
    import pickle
    with open('/job/job_source/job_info.pkl', 'wb') as f:
        pickle.dump(job_info, f)
    print('saving job info!')


def log_metric(key, value):
    job_info['results'][key] = value
    save_job_info()


def log_param(key, value):
    log_params({key: value})


def log_params(parameters):
    job_info['params'] = {**job_info['params'], **parameters}
    save_job_info()


def set_tensorboard_logdir(path):
    job_info['tensorboard_path'] = path
    save_job_info()


def save_artifact(filepath: str, key=None):
    import random
    if key is None:
        key = str(random.randint(0, 9999)) + "_" + filepath.split('/')[-1].split('.')[0]
    if filepath.startswith("/job/job_source/"):
        filepath = filepath[len("/job/job_source/"):]
    job_info['artifacts'][key] = filepath
    save_job_info()


def log(*info):
    print(*info)


def submit(job_directory, command, params, num_gpus, **kwargs):
    # default_params = {
    #     'auto_select_gpus': True,
    # }
    # params = {**default_params, **params}

    runner_params = {
        'job_directory': job_directory,
        'command': 'python -W ignore ' + command,
        'params': params,
        'num_gpus': num_gpus
    }
    from utils.tools import tuples_to_lists
    runner_params = tuples_to_lists(runner_params)
    import yaml
    with open('kube_runner_param.yaml', 'w') as f:
        yaml.dump(runner_params, f)

    with open('kube_runner_param.yaml', 'r') as f:
        verify = yaml.safe_load(f)
        assert verify == runner_params

    if 'num_gpus' in kwargs:
        kwargs.pop('num_gpus')

    import os
    os.system(f'cp ~/.kube/config {job_directory}/kube.config')
    from utils.atlas_backend import submit as atlas_submit
    atlas_submit(job_directory='.', command='utils/kubernetes_runner.py', params=params, **kwargs)


def load_parameters(log_parameters=True):
    print('loading parameters from kube backend!')
    import yaml
    with open('kube_job_parameters.yaml', 'r') as f:
        param = yaml.safe_load(f)
    if log_parameters:
        log_params(param)
    return param


"""
kubectl exec --stdin --tty <pod-name> -- /bin/bash
kubectl cp /tmp/foo <some-namespace>/<some-pod>:/tmp/bar
pod 到本地
kubectl cp -c <container> <some-namespace>/<some-pod>:/tmp/bar /temp/foo
kubectl delete --all deployments --namespace=wuvin
"""