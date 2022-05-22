# Atlas 的 Kubernetes 集群使用插件

* 本份小插件代码目前仅支持 [dessa/atlas](https://github.com/dessa-oss/atlas) 框架，如果要用于 NNI 可能还得改一小下。

## 使用方法

* 根据 https://github.com/dessa-oss/atlas 的 User guide 安装 Atlas 框架。
* 在 `backend.py` 中，一共有三种 `backend` 选项，分别为 本地运行、基于 Atlas 运行、基于调用 Kubernetes 的 Atlas 运行。
三种接口完全一致。
* [这里](https://docs.atlas.dessa.com/en/latest/) 有如何使用这套接口的教程。
* 将你需要的训练数据（这部分对应于 Atlas 中 Docker 挂载的内容）上传到集群 PVC 中，并在对应的 pod 挂载 PVC 到正确位置。

## 创建镜像

* 由于 Atlas 和集群都是基于 Docker 运行程序，所以各自需要一个镜像。
* `custom_docker_image` 中 `torch1_8_1` 目录下为一个我写好的带有常用 python 库的镜像文件，自己 build 即可，此镜像用于 Atlas。
* `custom_docker_image` 中 `kuber` 目录下为一个带有常用 python 库的适用于集群的镜像文件，可以自己 build，也可也直接从 `172.16.112.220:30006/wuvin/pylight:v0` 获取。

## 样例

* `job.config.yaml` 为一个 Atlas 的配置样例，该样例挂载了 `/data`, `/home/kailu/.cache` 到 docker 中，并使用了上述 build 得到的镜像。
* `kubernetes.config.yaml` 为插件需要的一个默认配置文件，文件中部分属性复写后会被提交到集群，这里你可以修改你在集群中一个 pod 需要的内存、CPU、显存容量等。

## 已知bug

* 如果 Atlas 框架运行在GPU模式下，那么你使用的 Kubernetes 集群的 pods 数量不能超过当前机器的 GPU 数量。
* 如果 Atlas 框架运行在CPU模式下，那么你使用 Submit 提交的 Job 将全部提交给集群，超过集群承受容量部分的 Job 会由于资源不足而直接失败。
也就是不再会有排队的机制。
* 要解决上述问题，需要等下一位好心人写一个 Scheduler 来代替 Atlas 原本的 Scheduler。
