#!/bin/bash
set -euo pipefail
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

rm -rf /home/jenkins/opensource/json

BASE_DIR=${WORKSPACE}
PYPTO_GOLDEN_PATH="$BASE_DIR/pypto_golden"
mkdir -p $PYPTO_GOLDEN_PATH
PYPTO_3RD_LIB_PATH="/home/opensource"
CHANGED_FILES_PARAM="$BASE_DIR/pr_filelist.txt"

# ===== slog 日志打包：注册为 EXIT trap，无论脚本从哪里退出（成功/失败）都会执行 =====
pack_slog() {
    local slog_name="slog.tar.gz"
    if [ -n "${soc_version:-}" ]; then
        slog_name="slog_${soc_version}.tar.gz"
    fi
    mkdir -p /root/ascend
    if tar -zcf "${BASE_DIR}/${slog_name}" -C /root/ascend log 2>/dev/null; then
        echo "[INFO] slog packaged: ${BASE_DIR}/${slog_name}"
    else
        echo "[WARN] no slog log dir to package, skip"
    fi
}
trap pack_slog EXIT

echo "run branch \${GIT_TARGET_BRANCH} smoke test"

SHM_SIZE_GB=$(df -BG /dev/shm 2>/dev/null | awk 'NR==2 {print $2}' | sed 's/G//')
if [ "$SHM_SIZE_GB" -lt 32 ]; then
    echo "[FIX] Expanding /dev/shm from ${SHM_SIZE_GB}GB to 32GB for distributed tests..."
else
    echo "[CHECK] /dev/shm already ${SHM_SIZE_GB}GB (sufficient)"
fi

CANN_PATH="/usr/local/Ascend"
source /opt/conda/bin/activate python39
source $CANN_PATH/cann/bin/setenv.bash

# C++ stest_distributed 需要 mpirun 命令和 MPI_HOME 环境变量（用于动态加载 libmpi.so）
MPI_BIN_DIR="/opt/conda/bin"
MPI_LIB_DIR="/opt/conda/lib"
PYTHON39_BIN_DIR="/opt/conda/envs/python39/bin"
if [ -f "$MPI_BIN_DIR/mpirun" ]; then
    # 软链 mpirun 到 python39 的 bin，避免污染 PATH 中的 python 版本
    for tool in mpirun mpiexec hydra_pmi_proxy; do
        [ -f "$MPI_BIN_DIR/$tool" ] && ln -sf "$MPI_BIN_DIR/$tool" "$PYTHON39_BIN_DIR/$tool" 2>/dev/null
    done
    export MPI_HOME="/opt/conda"
    export LD_LIBRARY_PATH="$MPI_LIB_DIR:${LD_LIBRARY_PATH:-}"
    echo "[FIX] Configured MPI environment: MPI_HOME=$MPI_HOME"
else
    echo "[WARN] mpirun not found at $MPI_BIN_DIR, C++ distributed tests may fail"
fi

cd "$BASE_DIR"

set +e

# ===== 安装 pto-isa =====
case "${GIT_TARGET_BRANCH}" in
    br_0.1.1_20260313_beta)
        PTO_ISA_URL="https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/pypto/cann/br_0313/package/cann-pto-isa_9.0.0_linux-aarch64.run"
        ;;
    0.2.0)
        PTO_ISA_URL="https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/package/pto/9.0.0/20260330/aarch64/cann-pto-isa_9.0.0_linux-aarch64.run"
        ;;
    9.1.0)
        PTO_ISA_URL="https://opencann-obs.obs.cn-north-4.myhuaweicloud.com/pto-isa/9.1.0/cann-pto-isa_linux-aarch64.run"
        ;;
    *)
        PTO_ISA_URL="https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/pto-isa/daily/cann-pto-isa_linux-aarch64.run"
        ;;
esac
PTO_ISA_PACKAGE_NAME=$(basename "${PTO_ISA_URL}")
if ! wget -q -O "${PTO_ISA_PACKAGE_NAME}" "${PTO_ISA_URL}"; then
    echo "[ERROR] pto-isa package download failed!URL:${PTO_ISA_URL}"
    exit 1
fi
echo "pto-isa package downloaded successfully: ./${PTO_ISA_PACKAGE_NAME}"
echo "Adding execute permission: chmod +x ${PTO_ISA_PACKAGE_NAME}"
chmod +x "${PTO_ISA_PACKAGE_NAME}" || echo "Failed to add execute permission to the package"
bash "${PTO_ISA_PACKAGE_NAME}" --full --quiet --install-path=/usr/local/Ascend

RUN_PACKAGE_NAME="cann-pypto_linux-aarch64_ubuntu24.run"
RUN_PACKAGE_URL="https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/${obs_path}/${RUN_PACKAGE_NAME}"
if ! wget -q -O "${RUN_PACKAGE_NAME}" "${RUN_PACKAGE_URL}"; then
    echo "Package download failed!URL:${RUN_PACKAGE_URL}"
else
    echo "Package downloaded successfully: ./${RUN_PACKAGE_NAME}"
fi
# Add execute permission to the downloaded package
echo "Adding execute permission: chmod +x ${RUN_PACKAGE_NAME}"
chmod +x "${RUN_PACKAGE_NAME}" || echo "Failed to add execute permission to the package"
bash "${RUN_PACKAGE_NAME}" --full -q --pylocal --install-path=/usr/local/Ascend
source /usr/local/Ascend/cann/set_env.sh

# ===== 测试阶段 =====
timeout --signal=INT 600 python3 examples/validate_examples.py -t examples -d 0,1,2,3
ret=$?
if [ $ret -ne 0 ]; then
    echo "[ERROR] Python Examples failed"
    exit $ret
fi
echo "[INFO] Python Examples succeeded"
# ===== gym 测试阶段 =====
# PYTEST_AVAILABLE_DEVICES 用于 conftest 按 worker 分卡
GYM_DEVICES=(0 1 2 3)
GYM_DEVICES_STR=$(IFS=,; echo "${GYM_DEVICES[*]}")
GYM_N_WORKERS=${#GYM_DEVICES[@]}
GYM_TEST_TIMEOUT=${GYM_TEST_TIMEOUT:-600}

echo "[BGN] Clone pypto-gym & run tests (${GYM_N_WORKERS} cards)"
rm -rf "$BASE_DIR/pypto-gym"
git clone --depth 1 https://gitcode.com/cann/pypto-gym.git "$BASE_DIR/pypto-gym"
if [ $? -ne 0 ]; then
    echo "[ERROR] Clone pypto-gym failed"
    exit 1
fi

pushd "$BASE_DIR/pypto-gym" > /dev/null || exit 1
if PYTEST_AVAILABLE_DEVICES="$GYM_DEVICES_STR" ASCEND_VISIBLE_DEVICES="$GYM_DEVICES_STR" \
    timeout --signal=INT "$GYM_TEST_TIMEOUT" \
    python3 -m pytest \
    tests/ops/deepseek_v32_exp/test_sparse_attention_antiquant.py::test_sfa_bf16_b4_s2_seq64k_per_int8_d \
    tests/ops/deepseek_v32_exp/test_mla_prolog_quant.py::test_b64_s64k2_pa_nd_bf16_quantb_d \
    tests/ops/experimental/ops_transformer/flash_attention_mha/test_flash_attention_mha.py::test_00_910_1b \
    tests/ops/experimental/ops_transformer/flash_attention_mha_grad/test_flash_attention_mha_grad_a3.py::test_02 \
    -v -n "$GYM_N_WORKERS" --dist=loadscope -p no:skipping --durations=0; then
    gym_ret=0
else
    gym_ret=$?
fi
popd > /dev/null || true

if [ $gym_ret -ne 0 ]; then
    echo "[ERROR] pypto-gym tests failed, keep repo for debug"
    exit $gym_ret
fi
rm -rf "$BASE_DIR/pypto-gym"
echo "[INFO] pypto-gym tests succeeded"

source /opt/conda/bin/deactivate
