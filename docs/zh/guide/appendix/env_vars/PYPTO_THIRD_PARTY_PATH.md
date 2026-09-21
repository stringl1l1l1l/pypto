# PYPTO_THIRD_PARTY_PATH

## 功能描述

指定PyPTO编译所需的第三方开源软件源码包路径。当编译环境无法访问`cann-src-third-party`仓库自动下载时，需手动准备源码包并通过该变量指定路径。

- 类型：字符串（绝对路径）

## 配置示例

```bash
# 手动准备第三方源码包后设置
export PYPTO_THIRD_PARTY_PATH=<path-to-thirdparty>
```

设置完成后，再执行编译和安装。

## 使用约束

- 路径下需包含`json-3.11.3`和`libboundscheck-v1.1.16`的源码包。
- 仅在源码编译安装时生效，PyPI安装无需设置。

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
