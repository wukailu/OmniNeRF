import sys
import os

sys.path.append(os.getcwd())

import utils.atlas_backend as atlas_backend

namespace = 'wuvin'


def fetch(relative_file_path, pod_name):
    os.system(
        f'kubectl -n {namespace} cp {namespace}/{pod_name}:/job/job_source/{relative_file_path} {relative_file_path}')


# now, we are inside the docker of atlas
def run_job(kube_job_id, yaml_data):
    # start the pod
    from kubernetes import client, config

    config.kube_config.load_kube_config(config_file="kube.config")
    v1 = client.CoreV1Api()
    k8s_apps_v1 = client.AppsV1Api()

    # 部署 Deployment
    resp = k8s_apps_v1.create_namespaced_deployment(body=yaml_data, namespace=namespace)
    deployment_name = resp.metadata.name
    cuda_mem_error = False
    cudnn_error = False

    try:
        import time

        while True:
            resp = k8s_apps_v1.read_namespaced_deployment_status(name=deployment_name, namespace=namespace)
            if resp.status.available_replicas == 1 and resp.status.ready_replicas == 1:
                atlas_backend.log('deployment created! available_replicas = ', resp.status.available_replicas)
                break
            time.sleep(1)  # sleep 1 second

        # 列出所有的pod
        pod_name = None
        resp = v1.list_namespaced_pod(namespace=namespace)
        for i in resp.items:
            if i.metadata.name.startswith(kube_job_id):
                pod_name = i.metadata.name
                atlas_backend.log('ip=', i.status.pod_ip)
                break
        if pod_name is None:
            raise Exception("Pod not found! job_id = ", kube_job_id)
        atlas_backend.log('pod found with pod name = ', pod_name)

        # Calling exec and waiting for response
        from kubernetes.stream import stream

        exec_command = ['/bin/sh', '-c', 'cp -r /data/cache/* ~/.cache']
        resp = stream(v1.connect_get_namespaced_pod_exec,
                      pod_name,
                      namespace,
                      command=exec_command,
                      stderr=True, stdin=False,
                      stdout=True, tty=False)
        print(resp)

        exec_command = ['/bin/sh', '-c', 'mkdir /job']
        resp = stream(v1.connect_get_namespaced_pod_exec,
                      pod_name,
                      namespace,
                      command=exec_command,
                      stderr=True, stdin=False,
                      stdout=True, tty=False)
        print(resp)

        # 拷贝可执行程序
        os.system(
            f'kubectl -n {namespace} cp {job_directory} {namespace}/{pod_name}:/job')  # 会在job 下创建文件夹 job 然后把东西放进去，太离谱了

        # exec_command = ['/bin/sh', '-c', "echo \"This message goes to stderr\" >&2; sleep 5; echo This message is late."]
        exec_command = ['/bin/sh', '-c', 'mv /job/job /job/job_source && cd /job/job_source/ &&' + command]
        resp = stream(v1.connect_get_namespaced_pod_exec,
                      pod_name,
                      namespace,
                      command=exec_command,
                      stderr=True, stdin=True,
                      stdout=True, tty=False,
                      _preload_content=False)

        while resp.is_open():
            resp.update(timeout=10000)
            if resp.peek_stdout():
                atlas_backend.log("%s" % resp.read_stdout())
            if resp.peek_stderr():
                ret = resp.read_stderr()
                if "out of memory" in ret or 'No GPUs available.' in ret:
                    cuda_mem_error = True
                if "CUDNN_STATUS_INTERNAL_ERROR" in ret:
                    cudnn_error = True
                if not ret.startswith("Global seed set to"):
                    atlas_backend.log("STDERR: %s" % ret)  # 总有一些奇怪的信息走这里出来，明明该走上面的
            time.sleep(1)
        resp.close()
        print('connection close...')

        if not (cuda_mem_error or cudnn_error):
            print('program running finished! copying back results...')
            fetch('job_info.pkl', pod_name)
            import pickle

            with open('job_info.pkl', 'rb') as fx:
                job_info = pickle.load(fx)
            # job_info = {'params': {}, 'results': {}, 'tensorboard_path': '', 'artifacts': {}}
            if job_info['params'] != {}:
                atlas_backend.log_params(job_info['params'])
            if job_info['tensorboard_path'] != '':
                atlas_backend.set_tensorboard_logdir(job_info['tensorboard_path'])
                fetch(job_info["tensorboard_path"], pod_name)
            for key, path in job_info['artifacts'].items():
                fetch(path, pod_name)
                atlas_backend.save_artifact(path, key=key)
            for key, value in job_info['results'].items():
                atlas_backend.log_metric(key, value)

    finally:
        # shutdown pod
        os.system(f'kubectl -n {namespace} delete deployment {deployment_name}')
        return cuda_mem_error or cudnn_error


if __name__ == "__main__":
    import random
    import string

    job_id = ''.join(random.choice(string.ascii_lowercase) for i in range(12))
    atlas_backend.log('job_id = ', job_id)

    # load the params and info
    import yaml

    with open('kube_runner_param.yaml', 'r') as f:
        params = yaml.safe_load(f)
    job_directory = params['job_directory']
    command = params['command']
    job_params = params['params']
    num_gpus = params['num_gpus']
    atlas_backend.log_params(job_params)

    with open('kube_job_parameters.yaml', 'w') as f:
        yaml.dump(job_params, f)

    # 初始化本地 kubectl
    os.system('mkdir ~/.kube')
    os.system('cp kube.config ~/.kube/config')
    os.system(f'kubectl -n {namespace} cluster-info')

    # modify the kubernetes config
    import yaml

    with open('kubernetes.config.yaml', 'r') as f:
        data = yaml.safe_load(f)
    data['metadata']['name'] = job_id + '-deployment'  # deployment_name
    data['spec']['selector']['matchLabels']['name'] = job_id + '-pod'  # pod name
    data['spec']['template']['metadata']['labels']['name'] = job_id + '-pod'
    data['spec']['template']['spec']['containers'][0]['resources']['limits']['virtaitech.com/gpu'] = num_gpus
    data['spec']['template']['spec']['containers'][0]['resources']['requests']['virtaitech.com/gpu'] = num_gpus
    data['spec']['template']['spec']['containers'][0]['env'][2]['value'] = str(num_gpus)
    data['spec']['replicas'] = 1  # could only creat 1 pod a time

    retry = True
    limits = 10
    while retry and limits > 0:
        limits -= 1
        retry = run_job(job_id, data)
        if retry:
            import time
            time.sleep(60)
    if retry:
        raise Exception("No available node can be found!")

