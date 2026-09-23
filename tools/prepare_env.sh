#!/bin/bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 说明，此脚本支持快速部署仿真和NPU上板环境
# 如果是仿真，则安装运行依赖
# 如果是上板环境，则根据输入选项，下载CANN包、安装CANN包，并决定是否要安装1~2包 默认不下载安装
# 此脚本不下载安装torch_npu
set -euo pipefail

TYPE=""
DOWNLOAD_HDK=false
DEVICE_TYPE=""
INSTALL_PATH="/usr/local/Ascend"
DOWNLOAD_DIR=$(dirname "$(dirname "$(dirname "$(readlink -f "$0")")")")/pypto_download
QUIET=false
ONLY_DOWNLOAD=false

DOWNLOADED_CANN_FILES=()
INSTALL_CANN_FILES=()
BASIC_MISSING_PKGS=()
BASIC_OUTDATED_PKGS=()

SCRIPT_PATH=$(readlink -f "$0")

CANN_DOWNLOAD_PATH="$DOWNLOAD_DIR/cann_packages"
THIRD_PARTY_DOWNLOAD_PATH="$DOWNLOAD_DIR/third_party_packages"
CANN_VERSION_LATEST="9.1.1"
OS=""
ARCH=""

readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly MAGENTA='\033[0;35m'
readonly NC='\033[0m'

BASIC_DEPENDENCIES=(
    "cmake:3.16.3"
    "gcc:7.3.1"
    "python3:3.9.5"
    "pip3:"
    "make:"
    "g++:"
    "ninja:"
)

THIRD_PARTY_DEPENDENCIES=(
    "json"
    "securec"
)

readonly CANN_VERSION="9.1.1"

JSON_URL="https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/json-3.11.3.tar.gz"
SECUREC_URL="https://gitcode.com/cann-src-third-party/libboundscheck/releases/download/v1.1.16/libboundscheck-v1.1.16.tar.gz"

CANN_TOOLKIT_URL_X86="https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN 9.1.1/Ascend-cann-toolkit_9.1.1_linux-x86_64.run?response-content-type=application/octet-stream"
CANN_TOOLKIT_URL_ARM="https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN 9.1.1/Ascend-cann-toolkit_9.1.1_linux-aarch64.run?response-content-type=application/octet-stream"

CANN_PTO_ISA_URL_X86="https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/pto-isa/daily/cann-pto-isa_linux-x86_64.run"
CANN_PTO_ISA_URL_ARM="https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/pto-isa/daily/cann-pto-isa_linux-aarch64.run"

CANN_DRIVER_URL_X86_910b="https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Ascend%20HDK/Ascend%20HDK%2025.3.RC1/Ascend-hdk-910b-npu-driver_25.3.rc1_linux-x86-64.run?response-content-type=application/octet-stream"
CANN_DRIVER_URL_ARM_910b="https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Ascend%20HDK/Ascend%20HDK%2025.3.RC1/Ascend-hdk-910b-npu-driver_25.3.rc1_linux-aarch64.run?response-content-type=application/octet-stream"
CANN_DRIVER_URL_X86_910c="https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Ascend%20HDK/Ascend%20HDK%2025.3.RC1/Atlas-A3-hdk-npu-driver_25.3.rc1_linux-x86-64.run?response-content-type=application/octet-stream"
CANN_DRIVER_URL_ARM_910c="https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Ascend%20HDK/Ascend%20HDK%2025.3.RC1/Atlas-A3-hdk-npu-driver_25.3.rc1_linux-aarch64.run?response-content-type=application/octet-stream"

CANN_OPS_URL_X86_910b="https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN 9.1.1/Ascend-cann-910b-ops_9.1.1_linux-x86_64.run?response-content-type=application/octet-stream"
CANN_OPS_URL_ARM_910b="https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN 9.1.1/Ascend-cann-910b-ops_9.1.1_linux-aarch64.run?response-content-type=application/octet-stream"
CANN_OPS_URL_X86_910c="https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN 9.1.1/Ascend-cann-A3-ops_9.1.1_linux-x86_64.run?response-content-type=application/octet-stream"
CANN_OPS_URL_ARM_910c="https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN 9.1.1/Ascend-cann-A3-ops_9.1.1_linux-aarch64.run?response-content-type=application/octet-stream"

CANN_FIRMWARE_URL_910b="https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Ascend%20HDK/Ascend%20HDK%2025.3.RC1/Ascend-hdk-910b-npu-firmware_7.8.0.2.212.run?response-content-type=application/octet-stream"
CANN_FIRMWARE_URL_910c="https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Ascend%20HDK/Ascend%20HDK%2025.3.RC1/Atlas-A3-hdk-npu-firmware_7.8.0.2.212.run?response-content-type=application/octet-stream"

print_header() {
    local title="$1"
    local separator="══════════════════════════════════════════════════════"
    local separator_length=${#separator}
    local title_length=${#title}
    local padding=$(( (separator_length - title_length) / 2 ))
    local spaces=$(printf "%${padding:-0}s" "")

    echo -e "${BLUE}${separator}${NC}"
    echo -e "${CYAN}${spaces}${title}${NC}"
    echo -e "${BLUE}${separator}${NC}"
}

parse_arguments() {
    local help_flag=false
    while [[ $# -gt 0 ]]; do
        case $1 in
            --type=*)
                TYPE="${1#*=}"
                shift
                ;;
            --with-install-driver=*)
                local hdk_value="${1#*=}"
                if [[ "$hdk_value" =~ ^(true|false)$ ]]; then
                    DOWNLOAD_HDK="$hdk_value"
                else
                    log_print "error" "Invalid value for --with-install-driver: $hdk_value (must be true or false)"
                    help_flag=true
                fi
                shift
                ;;
            --device-type=*)
                local device_type_value="${1#*=}"
                device_type_value=$(echo "$device_type_value" | tr '[:upper:]' '[:lower:]')
                if [[ "$device_type_value" == "a2" || "$device_type_value" == "a3" ]]; then
                    DEVICE_TYPE="$device_type_value"
                else
                    log_print "error" "Invalid value for --device-type: $device_type_value (must be a2 or a3)"
                    help_flag=true
                fi
                shift
                ;;

            --install-path=*)
                INSTALL_PATH="${1#*=}"
                INSTALL_PATH=$(echo "$INSTALL_PATH" | sed 's:/*$::')
                if [[ "$INSTALL_PATH" != /* ]]; then
                    log_print "error" "Install path must be an absolute path: $INSTALL_PATH"
                    return 1
                fi
                shift
                ;;
            --download-path=*)
                DOWNLOAD_DIR="${1#*=}"
                DOWNLOAD_DIR=$(echo "$DOWNLOAD_DIR" | sed 's:/*$::')
                CANN_DOWNLOAD_PATH="$DOWNLOAD_DIR/cann_packages"
                THIRD_PARTY_DOWNLOAD_PATH="$DOWNLOAD_DIR/third_party_packages"
                if [[ "$DOWNLOAD_DIR" != /* ]]; then
                    log_print "error" "Download path must be an absolute path: $DOWNLOAD_DIR"
                    return 1
                fi
                shift
                ;;
            --only-download)
                ONLY_DOWNLOAD=true
                shift
                ;;
            --quiet)
                QUIET=true
                shift
                ;;

            -h|--help)
                help_flag=true
                shift
                ;;
            *)
                log_print "error" "Unknown option: $1"
                help_flag=true
                shift
                ;;
        esac
    done

    if [ "$help_flag" = true ]; then
        show_usage
        return 1
    fi

    # Validate required parameters
    local missing_params=()

    if [ -z "$TYPE" ]; then
        missing_params+=("--type")
    elif [[ ! "$TYPE" =~ ^("deps"|"cann"|"third_party"|"all")$ ]]; then
        log_print "error" "Invalid value for --type: $TYPE (must be 'deps' or 'cann' or third_party or 'all')"
        return 1
    fi

    if [[ ( "$TYPE" == "all" || "$TYPE" == "cann" ) && -z "$DEVICE_TYPE" ]]; then
        log_print "error" "The --device-type parameter must be specified when --type is $TYPE!"
        exit 1
    fi

    if [ ${#missing_params[@]} -gt 0 ]; then
        log_print "error" "Missing required parameters: ${missing_params[*]}"
        show_usage
        return 1
    fi
    return 0
}

validate_and_display_config() {
    print_header "Input Parameters"
    log_print "info" "Install Type: $TYPE"
    log_print "info" "Download HDK: $DOWNLOAD_HDK"
    log_print "info" "Packages Download Path: $DOWNLOAD_DIR"
    log_print "info" "Cann Packages Install Path: $INSTALL_PATH"
}

show_usage() {
cat << EOF
Usage: $0 [REQUIRED_OPTIONS] [OPTIONAL_OPTIONS]
Required Options:
    --type=<type>                   Installation mode (cann, deps, third_party, all)
    --device-type=<type>            Device type (a2 or a3)

Optional Options:
    --with-install-driver=<bool>    Download driver and firmware packages (true or false, default: false)
    --download-path=<path>          Download cann packages to specific dir path
    --install-path=<path>           Install cann packages to specific dir path
    --quiet                         Run in quiet mode, automatically answer yes to all prompts
    --only-download                 Download cann packages and third_party packages
    -h | --help                     Show this help message
EOF
}

log_print() {
    local type="$1"
    local message="$2"
    case "$type" in
        info)
            echo -e "${CYAN}INFO${NC}: $message"
            ;;
        warning)
            echo -e "${YELLOW}WARNING${NC}: $message"
            ;;
        error)
            echo -e "${RED}ERROR${NC}: $message" >&2
            ;;
        success)
            echo -e "${GREEN}SUCCESS${NC}: $message"
            ;;
        *)
            echo "$message"
            ;;
    esac
}

detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        echo "$ID"
    elif command -v lsb_release >/dev/null 2>&1; then
        lsb_release -si | tr '[:upper:]' '[:lower:]'
    else
        echo "unknown"
    fi
}

detect_architecture() {
    local arch
    arch=$(uname -m)
    case "$arch" in
        x86_64|i386|i686)
            ARCH="x86"
            ;;
        aarch64|arm64|armv8l|armv7l)
            ARCH="arm"
            ;;
        *)
            log_print "error" "Unsupported CPU architecture"
            exit 1
            ;;
    esac
}

get_os_version() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        echo "$VERSION_ID"
    else
        echo ""
    fi
}

version_ge() {
    local ver1="$1"
    local ver2="$2"
    if [ "$(printf '%s\n' "$ver1" "$ver2" | sort -V | head -n1)" = "$ver2" ]; then
        return 0
    else
        return 1
    fi
}

clean_version() {
    local version="$1"
    echo "$version" | grep -oE '^[0-9]+\.[0-9]+(\.[0-9]+)*' | head -n1
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

get_version_by_cmd() {
    local pkg="$1"
    $pkg --version 2>/dev/null | head -n1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -n1 || true
}

get_installed_version() {
    local package="$1"
    local version=""
    version=$(get_version_by_cmd "$package")
    version=$(clean_version "$version")
    echo "$version"
}

check_basic_dependency() {
    local package="$1"
    local required_version="$2"

    if ! command_exists "$package"; then
        if [ -z "$required_version" ]; then
            echo -e "${RED}$package [NOT INSTALLED]${NC}"
        else
            echo -e "${RED}$package (requires: $required_version) [NOT INSTALLED]${NC}"
        fi
        return 1
    fi
    local installed_version
    installed_version=$(get_installed_version "$package")

    if [ -z "$required_version" ]; then
        echo -e "Requirement already satisfied: $package ${NC}"
        return 0
    fi

    if [ -z "$installed_version" ]; then
        echo -e "${YELLOW}$package (requires: $required_version) [VERSION DETECTION FAILED]${NC}"
        return 2
    fi

    if version_ge "$installed_version" "$required_version"; then
        echo -e "Requirement already satisfied: $package (==$installed_version)"
        return 0
    else
        echo -e "${YELLOW}$package $installed_version (requires: $required_version) [VERSION TOO LOW]${NC}"
        return 3
    fi
}

check_all_basic_dependencies() {
    local missing_pkgs=()
    local outdated_pkgs=()
    local all_satisfied=true
    log_print "info" "Checking basic system dependencies..."
    log_print "info" "Operating System: $(detect_os)"
    echo
    for dep in "${BASIC_DEPENDENCIES[@]}"; do
        local package="${dep%%:*}"
        local required_version="${dep##*:}"
        check_basic_dependency "$package" "$required_version"
        local result=$?
        case $result in
            1)
                missing_pkgs+=("$package:$required_version")
                all_satisfied=false
                ;;
            3)
                outdated_pkgs+=("$package:$required_version")
                all_satisfied=false
                ;;
        esac
    done
    echo "=================================================="

    if [ "$all_satisfied" = true ]; then
        echo
        return 0
    fi
    BASIC_MISSING_PKGS=("${missing_pkgs[@]}")
    BASIC_OUTDATED_PKGS=("${outdated_pkgs[@]}")
    return 1
}

prompt_yes_no() {
    local prompt="$1"
    local default="${2:-y}"

    # quiet mode:
    if [ "$QUIET" = true ]; then
        if [ "$default" = "y" ]; then
            log_print "info" "$prompt [Y/n]: Y (auto-selected in quiet mode)"
            return 0
        else
            log_print "info" "$prompt [y/N]: N (auto-selected in quiet mode)"
            return 1
        fi
    fi

    # non-quiet mode：
    while true; do
        if [ "$default" = "y" ]; then
            read -p "$prompt [Y/n]: " -n 1 -r
        else
            read -p "$prompt [y/N]: " -n 1 -r
        fi
        echo
        case "$REPLY" in
            y|Y|"")
                if [ "$default" = "y" ] || [ -n "$REPLY" ]; then
                    return 0
                fi
                ;;
            n|N)
                return 1
                ;;
            *)
                echo -e "${RED}Please enter 'y' or 'n'${NC}"
                ;;
        esac
    done
}

install_basic_dependencies() {
    local os_type=$(detect_os)
    local os_version=$(get_os_version)
    local installed_count=0
    local failed_count=0

    log_print "info" "Using package manager: $os_type $os_version"
    echo

    for pkg_info in "${BASIC_MISSING_PKGS[@]}" "${BASIC_OUTDATED_PKGS[@]}"; do
        local package="${pkg_info%%:*}"
        local required_version="${pkg_info##*:}"

        package=${package/pip3/python3-pip}
        package=${package/ninja/ninja-build}

        if [ -z "$required_version" ]; then
            log_print "installing_version" "$package"
        else
            log_print "installing_version" "$package==$required_version"
        fi

        local install_success=false
        local actual_version=""

        case "$os_type" in
            ubuntu|debian)
                sudo apt-get update >/dev/null 2>&1
                if sudo apt-get install -y "$package"; then
                    install_success=true
                    actual_version=$(get_installed_version "$package")
                fi
                ;;
            centos|rhel)
                package=${package/g++/gcc-c++}
                if command -v dnf >/dev/null 2>&1; then
                    if sudo dnf install -y "$package"; then
                        install_success=true
                        actual_version=$(get_installed_version "$package")
                    fi
                else
                    if sudo yum install -y "$package"; then
                        install_success=true
                        actual_version=$(get_installed_version "$package")
                    fi
                fi
                ;;
            *)
                log_print "error" "Unsupported operating system: $os_type"
                return 1
                ;;
        esac

        if [ "$install_success" = true ]; then
            if [ -z "$required_version" ]; then
                log_print "success" "Successfully installed $package $actual_version"
            elif [ -n "$actual_version" ] && version_ge "$actual_version" "$required_version"; then
                log_print "success" "Successfully installed $package $actual_version"
                ((installed_count++))
            else
                if [ -n "$actual_version" ]; then
                    log_print "error" "Installed version $actual_version does not meet reqSuirement $required_version"
                else
                    log_print "error" "Failed to verify installed version of $package"
                fi
                ((failed_count++))
            fi
        else
            log_print "error" "Failed to install $package"
            ((failed_count++))
        fi
    done

    echo
    echo "=================================================="
    log_print "info" "Basic dependencies: $installed_count successful, $failed_count failed"

    return $failed_count
}

show_summary() {
    print_header "Dependency Check Summary"
    echo -e "${CYAN}Basic Dependencies:${NC}"
    for dep in "${BASIC_DEPENDENCIES[@]}"; do
        local package="${dep%%:*}"
        local required_version="${dep##*:}"

        if command_exists "$package"; then
            local installed_version=$(get_installed_version "$package")

            if [ -z "$required_version" ]; then
                echo -e "  ${GREEN} $package $installed_version [OK]${NC}"
            elif [ -n "$installed_version" ] && version_ge "$installed_version" "$required_version"; then
                echo -e "  ${GREEN} $package $installed_version [OK]${NC}"
            else
                echo -e "  ${YELLOW} $package $installed_version (requires: $required_version)${NC}"
            fi
        else
            if [ -z "$required_version" ]; then
                echo -e "  ${RED} $package [NOT INSTALLED] ${NC}"
            else
                echo -e "  ${RED} $package [NOT INSTALLED] (requires: $required_version)${NC}"
            fi
        fi
    done
}

get_package_url() {
    local resource_type="$1"
    local cann_version="$CANN_VERSION_LATEST";
    case "$resource_type" in
        toolkit)
            case "$ARCH" in
                x86) echo "$CANN_TOOLKIT_URL_X86" ;;
                arm) echo "$CANN_TOOLKIT_URL_ARM" ;;
                *) echo "" ;;
            esac
            ;;
        driver)
            case "$ARCH" in
                x86)
                    [ "$DEVICE_TYPE" = "a2" ] && echo "$CANN_DRIVER_URL_X86_910b" || echo "$CANN_DRIVER_URL_X86_910c"
                    ;;
                arm)
                    [ "$DEVICE_TYPE" = "a2" ] && echo "$CANN_DRIVER_URL_ARM_910b" || echo "$CANN_DRIVER_URL_ARM_910c"
                    ;;
                *) echo "" ;;
            esac
            ;;
        firmware)
            case "$DEVICE_TYPE" in
                a2) echo "$CANN_FIRMWARE_URL_910b" ;;
                a3) echo "$CANN_FIRMWARE_URL_910c" ;;
                *) echo "" ;;
            esac
            ;;
        ops)
            case "$ARCH" in
                x86)
                    [ "$DEVICE_TYPE" = "a2" ] && echo "$CANN_OPS_URL_X86_910b" || echo "$CANN_OPS_URL_X86_910c"
                    ;;
                arm)
                    [ "$DEVICE_TYPE" = "a2" ] && echo "$CANN_OPS_URL_ARM_910b" || echo "$CANN_OPS_URL_ARM_910c"
                    ;;
                *) echo "" ;;
            esac
            ;;
        pto-isa)
            case "$ARCH" in
                x86) echo "$CANN_PTO_ISA_URL_X86" ;;
                arm) echo "$CANN_PTO_ISA_URL_ARM" ;;
                *) echo "" ;;
            esac
            ;;
        *)
            echo ""
            ;;
    esac
}

get_package_name_from_url() {
    local resource_url="$1"
    local resource_type="$2"
    local target_filename=""
    if [[ "$resource_type" = "driver" || "$resource_type" = "firmware" || "$resource_type" = "toolkit" || "$resource_type" = "ops" ]]; then
        target_filename=$(echo "$resource_url" | grep -o '[^/]*\.run' | head -1)
    else
        target_filename=$(basename "$resource_url")
    fi
    echo "$target_filename"
}

download_single_cann_package() {
    local resource_type="$1"
    local resource_url="$2"

    cd "$CANN_DOWNLOAD_PATH" || {
        log_print "error" "Failed to enter download directory: $CANN_DOWNLOAD_PATH"
        return 1
    }
    local package_file=$(get_package_name_from_url "$resource_url" "$resource_type")
    local target_file="$CANN_DOWNLOAD_PATH/$package_file"
    local resource_name=$(echo "$resource_type" | awk '{print toupper(substr($0,1,1)) substr($0,2)}')
    local download_log_msg="Downloading Huawei Ascend CANN $resource_name $CANN_VERSION_LATEST"

    if [ -f "$target_file" ]; then
        log_print "info" "$resource_name file already exists: $target_file"
        log_print "info" "Using existing $resource_name file"
        DOWNLOADED_CANN_FILES+=("$target_file")
        [[ "$resource_type" != "driver" && "$resource_type" != "firmware" ]] && INSTALL_CANN_FILES+=("$target_file")
        cd - >/dev/null
        return 0
    fi

    log_print "info" "$download_log_msg"

    local download_success=false
    if command_exists wget; then
        if wget --progress=bar:force -O "$target_file" "$resource_url"; then
            log_print "info" "$resource_name download completed"
            download_success=true
        fi
    elif command_exists curl; then
        if curl -L -# -o "$target_file" "$resource_url"; then
            log_print "info" "$resource_name download completed"
            download_success=true
        fi
    else
        log_print "error" "Neither wget nor curl is available for download"
    fi

    if [ "$download_success" = false ]; then
        log_print "error" "Failed to download CANN $resource_name from $resource_url"
        cd - >/dev/null
        return 1
    fi

    DOWNLOADED_CANN_FILES+=("$target_file")
    [[ "$resource_type" != "driver" && "$resource_type" != "firmware" ]] && INSTALL_CANN_FILES+=("$target_file")
    if chmod +x "$target_file"; then
        log_print "success" "CANN $resource_name downloaded and made executable: $target_file"
    else
        log_print "warning" "CANN $resource_name downloaded but could not make it executable: $target_file"
    fi

    cd - >/dev/null
    return 0
}

install_downloaded_packages() {
    if [ "${#INSTALL_CANN_FILES[@]}" -le 0 ]; then
        log_print "warning" "No downloaded files to install"
        return 0
    fi
    cd "$CANN_DOWNLOAD_PATH" || {
        log_print "error" "Failed to enter download directory: $CANN_DOWNLOAD_PATH"
        return 1
    }
    log_print "info" "================================"
    for pkg_file in "${INSTALL_CANN_FILES[@]}"; do
        if ! install_single_package "$pkg_file"; then
            log_print "error" "Failed to install ${pkg_file}"
            return 1
        fi
    done
    cd - >/dev/null
    log_print "success" "Successfully installed CANN packages."
    return 0
}

install_single_package() {
    local filename="$1"

    if [ ! -f "$filename" ]; then
        log_print "error" "File not found: $filename"
        return 1
    fi

    if [ ! -x "$filename" ]; then
        log_print "info" "Making file executable: $filename"
        chmod +x "$filename" || {
            log_print "error" "Failed to make file executable: $filename"
            return 1
        }
    fi

    local install_cmd=""

    if [[ "$filename" =~ "ops" ]]; then
        install_cmd="$filename --quiet --install --force --install-path=$INSTALL_PATH "
    elif [[ "$filename" =~ "toolkit" ]]; then
        install_cmd="$filename --quiet --install --force --install-path=$INSTALL_PATH "
    elif [[ "$filename" =~ "pto-isa" ]]; then
        install_cmd="$filename --quiet --full --install-path=$INSTALL_PATH "
    else
        install_cmd="$filename --full --install-path=$INSTALL_PATH "
    fi

    log_print "info" "Running: $install_cmd"
    if eval "$install_cmd"; then
        return 0
    else
        return 1
    fi
}

show_installation_plan() {
    log_print "info" "Installation List"
    log_print "info" "================="
    for files in "${INSTALL_CANN_FILES[@]}"; do
        log_print "info" "$files"
    done
    log_print "warn" "The driver and firmware packages need to be installed manually according to the installation guide."
}

download_third_party_packages() {
    print_header "Download Third-Party Package"

    mkdir -p "$THIRD_PARTY_DOWNLOAD_PATH" || {
        log_print "error" "Failed to create download directory: $THIRD_PARTY_DOWNLOAD_PATH"
        return 1
    }

    cd "$THIRD_PARTY_DOWNLOAD_PATH" || {
        log_print "error" "Failed to enter download directory: $THIRD_PARTY_DOWNLOAD_PATH"
        exit 1
    }

    local download_success=false
    for dep in "${THIRD_PARTY_DEPENDENCIES[@]}"; do
        local dep_upper=${dep^^}
        local dep_url="${dep_upper}_URL"
        var_value="${!dep_url}"
        if command_exists wget; then
            if wget "$var_value"; then
                log_print "info" "$dep download completed"
                download_success=true
            fi
        elif command_exists curl; then
            if curl -L -O "$var_value"; then
                log_print "info" "$dep download completed"
                download_success=true
            fi
        else
            log_print "error" "Neither wget nor curl is available for download"
        fi
    done

    cd - >/dev/null

    if [ "$download_success" = true ]; then
        log_print "success" "Download third-party packages successfully!"
        log_print "success" "path: ${THIRD_PARTY_DOWNLOAD_PATH}"
        return 0
    else
        log_print "error" "Some packages downloads failed"
        return 1
    fi
}

perform_installation() {
    show_installation_plan

    if ! prompt_yes_no "Proceed with installation as shown above?"; then
        log_print "warn" "Installation cancelled by user"
        return 0
    fi

    if [ -n "$INSTALL_PATH" ]; then
        local parent_dir=$(dirname "$INSTALL_PATH")
        if [ ! -d "$parent_dir" ]; then
            log_print "info" "Creating parent directory: $parent_dir"
            sudo mkdir -p "$parent_dir" || {
                log_print "error" "Failed to create parent directory: $parent_dir"
                return 1
            }
        fi
    fi
    install_downloaded_packages
    return $?
}

check_and_install_dependencies() {
    print_header "System Dependency Check & Installer"

    set +e
    check_all_basic_dependencies
    local basic_check_result=$?
    set -e

    if [ $basic_check_result -ne 0 ]; then
        echo
        if prompt_yes_no "Some basic dependencies are missing or outdated. Install/Update?"; then
            echo "Installing basic dependencies..."
            set +e
            install_basic_dependencies
            local basic_install_result=$?
            set -e

            if [ $basic_install_result -gt 0 ]; then
                log_print "warning" "Some basic dependencies failed to install properly"
            fi
            echo "Basic dependency installation completed."
        else
            log_print "warning" "Skipping basic dependency installation"
        fi
    else
        log_print "success" "All basic dependencies are satisfied!"
    fi

    show_summary
    local basic_ok=true
    for dep in "${BASIC_DEPENDENCIES[@]}"; do
        local package="${dep%%:*}"
        local required_version="${dep##*:}"

        if ! command_exists "$package"; then
            basic_ok=false
        else
            local installed_version=$(get_installed_version "$package")
            if [ -z "$required_version" ]; then
                :
            elif [ -n "$installed_version" ] && version_ge "$installed_version" "$required_version"; then
                :
            else
                basic_ok=false
            fi
        fi
    done

    if [ "$basic_ok" = true ]; then
        log_print "success" "All required dependencies are satisfied!"
    else
        log_print "warning" "Some dependencies may still need attention"
    fi
}

download_cann_packages() {
    print_header "Download CANN Package"
    log_print "info" "Starting Huawei Ascend CANN packages download..."
    log_print "info" "======================================="
    log_print "info" "Detected architecture: $ARCH"

    if [ "$DOWNLOAD_HDK" = true ]; then
        log_print "info" "Download: CANN-Toolkit + CANN-ops + CANN-pto-isa + CANN-deiver + CANN-firmware"
    else
        log_print "info" "Download: CANN-Toolkit + CANN-ops + CANN-pto-isa"
    fi
    echo

    mkdir -p "$CANN_DOWNLOAD_PATH" || {
        log_print "error" "Failed to create download directory: $CANN_DOWNLOAD_PATH"
        return 1
    }

    log_print "info" "Starting downloads for version $CANN_VERSION_LATEST..."
    log_print "info" "=============================================="

    local download_success=true
    local download_target=("toolkit" "ops" "pto-isa")
    local pkg_count=1
    if [ "$DOWNLOAD_HDK" = true ]; then
        download_target+=("driver" "firmware")
    fi

    local target_num=${#download_target[@]}
    for pkg_name in "${download_target[@]}"; do
        local pkg_url=$(get_package_url "$pkg_name")
        log_print "info" ""
        log_print "info" "Download ${pkg_count}/${target_num}: CANN-${pkg_name}"
        log_print "info" "----------------------------"
        download_single_cann_package "$pkg_name" "$pkg_url"
        ((pkg_count++))
    done

    # Show summary
    log_print "info" ""
    log_print "info" "Download Summary"
    log_print "info" "================"

    if [ "${#DOWNLOADED_CANN_FILES[@]}" -gt 0 ]; then
        log_print "success" "Successfully downloaded ${#DOWNLOADED_CANN_FILES[@]} package(s):"
        for file_info in "${DOWNLOADED_CANN_FILES[@]}"; do
            log_print "info" "  - $file_info"
        done
        log_print "info" "All files are saved in: $CANN_DOWNLOAD_PATH"
    else
        log_print "warning" "No packages were downloaded"
    fi

    if [ "$download_success" = true ]; then
        log_print "success" "Download process completed successfully!"
        return 0
    else
        log_print "error" "Some packages downloads failed"
        return 1
    fi
}

install_cann_packages() {
    print_header "Install CANN Packages"
    perform_installation
    return $?
}

precheck_system_info() {
    detect_os
    detect_architecture
}

main() {
    if ! parse_arguments "$@"; then
        exit 1
    fi

    precheck_system_info
    validate_and_display_config

    if [[ "$TYPE" == "deps" || "$TYPE" == "all" ]]; then
        check_and_install_dependencies
    fi

    if [[ "$TYPE" == "third_party" || "$TYPE" == "all" ]]; then
        download_third_party_packages
    fi

    if [[ "$TYPE" == "cann" || "$TYPE" == "all" ]]; then
        download_cann_packages
        if [[ "$ONLY_DOWNLOAD" != true ]]; then
            install_cann_packages
        fi
    fi
}

main "$@"
