# PTO_TILE_LIB_CODE_PATH

## 功能描述

pto-isa源码路径，用于编译和运行PyPTO算子。CANN toolkit安装后自带pto-isa，通常无需手动设置。当内置版本不满足需求时，可通过该环境变量指向独立的pto-isa源码目录。

- 类型：字符串（绝对路径）

## 配置示例

```bash
# 使用CANN内置pto-isa（推荐）
export PTO_TILE_LIB_CODE_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}/$(uname -m)-linux

# 使用独立pto-isa源码
git clone https://gitcode.com/cann/pto-isa.git
export PTO_TILE_LIB_CODE_PATH="$PWD/pto-isa"
```

## 使用约束

- 路径下必须包含`include/pto/`目录。
- 使用CANN包内置的PyPTO时，pto-isa已随CANN包安装，无需单独设置。

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A3系列产品：支持
<!-- end id3 -->
