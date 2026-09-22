# 环境部署

在使用PyPTO开发或运行之前，请您先参考下面步骤完成基础环境搭建和编译安装，确保已安装NPU驱动、固件和CANN软件（`Ascend-cann-toolkit`和`Ascend-cann-ops`）等。

## 环境安装

本项目提供多种搭建昇腾环境的方式，请按需选择。

> **说明**：本文提到的编译态和运行态含义如下，请根据实际情况选择。
>
> - 编译态：针对仅编译PyPTO不运行的场景，只需安装CANN toolkit包及PyPTO编译依赖。
> - 运行态：针对运行PyPTO的场景（编译运行或纯运行），需安装驱动与固件、CANN toolkit包、CANN ops包。

|  安装方式  |  使用说明  |  使用场景  |
| ----- | ------ | ------ |
|  CANNLab  | 一站式开发平台，提供在线直接运行的昇腾环境，无需手动安装。<br>当前可提供单机算力，**默认安装最新商发版CANN包**。 | 适用于没有昇腾设备的开发者。 |
|  Docker  | CANN镜像已预集成CANN及PyPTO运行所需依赖，开箱即用。<br>环境默认安装最新商发版CANN包，源码下载时注意与软件配套。 | 适用有昇腾设备，**需要快速搭建环境的开发者**。 |
|  手动安装  | 手动安装CANN包和PyPTO基础依赖，灵活性高。 | 适用有昇腾设备，**想体验PyPTO最新master分支能力、已发布版本能力，或基于源码进行PyPTO框架开发的开发者**。 |

### 方式1：CANNLab

对于无昇腾设备的开发者，可直接使用CANNLab云开发环境，即"**一站式开发平台**"，该平台为您提供在线可直接运行的昇腾环境，环境中已安装必备的驱动固件、软件包和依赖，无需手动安装。

> **说明**：环境默认安装最新商发版CANN包，源码下载时注意与软件配套。更多关于开发平台的介绍和使用操作请参考[CANNLab指导](https://gitcode.com/org/cann/cannlab/docs?code=quick-start-guide)。

### 方式2：Docker部署

对于有昇腾设备的开发者，若您想快速搭建昇腾环境，可使用Docker镜像部署。

> **说明**：
>
> - 镜像文件比较大，下载需要一定时间，请您耐心等待。关于docker命令的选项介绍可通过`docker --help`查询。
> - 环境默认安装最新商发版CANN包，源码下载时注意与软件配套。

1. **安装驱动与固件（运行态依赖）**

    驱动与固件是运行态依赖，若仅编译PyPTO，可不安装。使用`npu-smi info`检查是否有NPU相关信息，若没有，请参考《[CANN快速安装](https://www.hiascend.com/cann/download)》完成驱动与固件安装。

2. **下载镜像**

    - 步骤1：以root用户登录宿主机。确保宿主机已安装Docker引擎（版本1.11.2及以上），使用`docker --version`检查Docker版本，若没有，请参考[Docker官方安装指南](https://docs.docker.com/engine/install/)。
    - 步骤2：从[昇腾镜像仓库](https://www.hiascend.com/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884)拉取已预集成CANN软件包及PyPTO运行所需依赖的镜像。

        示例如下，请自行替换CANN版本号、芯片系列、操作系统、python版本信息。

        ```bash
        # 以cann:9.1.0-beta.1版本为例
        docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-beta.1-910b-openeuler24.03-py3.12-devel
        ```

3. **运行Docker**

    拉取镜像后，需要以特定参数启动容器，以便容器内能访问宿主的昇腾设备。

    ```bash
    docker run --name cann_container --device /dev/davinci0 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info -v /etc/ascend_install.info:/etc/ascend_install.info -it swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-beta.1-910b-openeuler24.03-py3.12-devel bash
    ```

    | 参数 | 说明 | 注意事项 |
    | :--- | :--- | :--- |
    | `--name cann_container` | 为容器指定名称，便于管理。 | 可自定义。 |
    | `--device /dev/davinci0` | 核心：将宿主机的NPU设备卡映射到容器内，可指定映射多张NPU设备卡。 | 必须根据实际情况调整：`davinci0`对应系统中的第0张NPU卡。请先在宿主机执行`npu-smi info`命令，根据输出显示的设备号（如`NPU 0`, `NPU 1`）来修改此编号。|
    | `--device /dev/davinci_manager` | 映射NPU设备管理接口。 | - |
    | `--device /dev/devmm_svm` | 映射设备内存管理接口。 | - |
    | `--device /dev/hisi_hdc` | 映射主机与设备间的通信接口。 | - |
    | `-v /usr/local/dcmi:/usr/local/dcmi` | 挂载设备容器管理接口（DCMI）相关工具和库。 | - |
    | `-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi` | 挂载`npu-smi`工具。 | 使容器内可以直接运行此命令来查询NPU状态和性能信息。|
    | `-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/` | 关键挂载：将宿主机的NPU驱动库映射到容器内。 | - |
    | `-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info` | 挂载驱动版本信息文件。 | - |
    | `-v /etc/ascend_install.info:/etc/ascend_install.info` | 挂载CANN软件安装信息文件。 | - |
    | `-it` | `-i`（交互式）和`-t`（分配伪终端）的组合参数。 | - |
    | `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-beta.1-910b-openeuler24.03-py3.12-devel` | 指定要运行的Docker镜像。 | 请确保此镜像名和标签（tag）与你通过`docker pull`拉取的镜像完全一致。 |
    | `bash` | 容器启动后立即执行的命令。 | - |

4. **安装PyPTO依赖**

    进入容器后，请参考[手动安装 - PyPTO依赖](#pypto依赖)完成Python依赖及其它编译依赖的安装。

### 方式3：手动安装

对于有昇腾设备的开发者，若您想手动搭建昇腾环境，请参考下述步骤。

#### 安装软件

- **场景1：体验master版本能力或基于master版本进行开发**

    1. **安装驱动与固件（运行态依赖）**

        驱动与固件是运行态依赖，若仅编译PyPTO，可不安装。使用`npu-smi info`检查是否有NPU相关信息，若没有，请参考《[CANN快速安装](https://www.hiascend.com/cann/download)》完成驱动与固件安装。

        > **重要**：
        >
        > - 支持版本：Ascend HDK 25.5.1及以上。
        > - 低于支持版本的HDK环境不在PyPTO验证和支持范围内，运行异常PyPTO程序时可能导致NPU状态异常，进而影响后续任务执行，出现AIC超时等问题；严重情况下可能需要重启设备或主机后恢复。

    2. **安装CANN包**

        请单击[下载链接](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/)，选择最新时间版本，并根据产品型号和环境架构下载对应包。安装命令如下，更多指导参考《[CANN快速安装](https://www.hiascend.com/cann/download)》。

        - 安装CANN toolkit包

            ```bash
            bash ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
            ```

        - 安装CANN ops包（运行态依赖）

            ops包是运行态依赖，若仅编译PyPTO，可不安装此包。

            ```bash
            bash ./Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
            ```

        变量含义说明：

        - \$\{cann\_version\}：表示CANN包版本号。
        - \$\{arch\}：表示CPU架构，可通过`uname -m`查询，例如aarch64、x86_64。
        - \$\{soc\_name\}：表示NPU型号名称。
        - \$\{install\_path\}：表示指定安装路径，ops包需与toolkit包安装在相同路径，root用户默认安装在`/usr/local/Ascend`目录。

- **场景2：体验PyPTO已发布版本能力或基于已发布版本进行开发**

    请访问[CANN官网下载中心](https://www.hiascend.com/cann/download)，选择与PyPTO版本配套的CANN发布版本，并根据产品型号和环境架构下载对应包，最后参考网页提供的命令完成安装。

#### PyPTO依赖

1. **安装Python依赖**

    - Python：版本 >= 3.9
        - **重要**：需要安装Python的Development组件（常称为`python3-dev`）。

    - 安装Python依赖包：

        依赖的pip包及对应版本在`python/requirements.txt`中描述，可以使用如下命令完成安装：

        ```bash
        # 进入PyPTO项目源码根目录
        cd pypto

        # 安装相关pip包依赖
        python3 -m pip install -r python/requirements.txt
        ```

    - PyTorch及TorchNPU：
        - **顺序说明**：请务必先完成上文"安装CANN包"章节中的toolkit包安装后，再安装`TorchNPU`。
        - 请根据实际环境的Python版本单独安装，请参考《[TorchNPU软件安装](https://www.hiascend.com/document/detail/zh/Pytorch/latest/installguide/swinstall/docs/zh/installation_guide/installation_description.md)》。
        - **重要**：需确保`PyTorch`、`TorchNPU`与`PyPTO`三者的Python版本一致。

2. **安装其它依赖**

    - cmake >= 3.16.3
    - make
    - g++ >= 7.3.1
    - gcc >= 7.3.1
    - pybind11 >= 2.13.6（pip包，可通过`python3 -m pip install pybind11`安装）
    - patch

## 环境验证

安装完CANN包后，需验证环境和驱动是否正常。

- **检查NPU设备**

    ```bash
    # 运行npu-smi，若能正常显示设备信息，则驱动正常
    npu-smi info
    ```

- **检查CANN版本**

    ```bash
    # 查看CANN toolkit包及ops包版本信息（默认路径安装），CANNLab场景下将/usr/local替换为/home/developer
    cat /usr/local/Ascend/cann/${arch}-linux/ascend*install.info
    ```

    其中\${arch}可通过`uname -m`查询当前架构，如aarch64、x86_64。

环境准备完成后，请参考[PyPTO安装](./build_and_install.md)文档完成PyPTO的安装。

## 可选安装

### PyPTO Toolkit插件

如需体验计算图和泳道图的查看能力，请安装PyPTO Toolkit插件：
具体使用文档参考 [PyPTO Toolkit 文档](https://pypto-tools.gitcode.com/index.html)

### MPI依赖

PyPTO的分布式用例依赖MPI，推荐版本 >= 3.2.1。

**方式一：通过系统包管理器安装**

```bash
# Ubuntu/Debian系统
apt-get update && apt-get install -y mpich

# CentOS/RHEL系统
yum install -y mpich
```

**方式二：通过源码编译安装**

```bash
# 以3.2.1版本为例
version='3.2.1'
wget https://www.mpich.org/static/downloads/${version}/mpich-${version}.tar.gz
tar -xzf mpich-${version}.tar.gz
cd mpich-${version}
./configure --prefix=/usr/local/mpich --disable-fortran
make && make install
```

安装完成后设置环境变量：

```bash
export MPI_HOME=/usr/local/mpich
export PATH=${MPI_HOME}/bin:${PATH}
```

## 环境变量配置

按需选择合适的命令使CANN环境变量生效。上述环境变量配置只在当前窗口生效，用户可以按需将以上命令写入环境变量配置文件（如`.bashrc`文件）。

```bash
# 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
source /usr/local/Ascend/cann/set_env.sh

# 指定路径安装
source ${install_path}/cann/set_env.sh
```
