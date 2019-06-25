#!/bin/bash
set -ex

TESTS_FILE="./lite_tests.txt"
LIBS_FILE="./lite_libs.txt"

readonly common_flags="-DWITH_LITE=ON -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=OFF -DWITH_PYTHON=OFF -DWITH_TESTING=ON -DLITE_WITH_ARM=OFF"

NUM_CORES_FOR_COMPILE=8

# for code gen, a source file is generated after a test, but is dependended by some targets in cmake.
# here we fake an empty file to make cmake works.
function prepare_workspace {
    # in build directory
    # 1. Prepare gen_code file
    GEN_CODE_PATH_PREFIX=paddle/fluid/lite/gen_code
    mkdir -p ./${GEN_CODE_PATH_PREFIX}
    touch ./${GEN_CODE_PATH_PREFIX}/__generated_code__.cc

    # 2.Prepare debug tool
    DEBUG_TOOL_PATH_PREFIX=paddle/fluid/lite/tools/debug
    mkdir -p ./${DEBUG_TOOL_PATH_PREFIX}
    cp ../${DEBUG_TOOL_PATH_PREFIX}/analysis_tool.py ./${DEBUG_TOOL_PATH_PREFIX}/
}

function check_need_ci {
    git log -1 --oneline | grep "test=develop" || exit -1
}

function cmake_x86 {
    prepare_workspace
    cmake ..  -DWITH_GPU=OFF -DWITH_MKLDNN=OFF -DLITE_WITH_X86=ON ${common_flags}
}

function cmake_opencl {
    # $1: ARM_TARGET_OS in "android" , "armlinux"
    # $2: ARM_TARGET_ARCH_ABI in "arm64-v8a", "armeabi-v7a" ,"armeabi-v7a-hf"
    cmake .. \
        -DLITE_WITH_OPENCL=ON \
        -DWITH_GPU=OFF \
        -DWITH_MKL=OFF \
        -DWITH_LITE=ON \
        -DLITE_WITH_CUDA=OFF \
        -DLITE_WITH_X86=OFF \
        -DLITE_WITH_ARM=ON \
        -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=ON \
        -DWITH_TESTING=ON \
        -DARM_TARGET_OS=$1 -DARM_TARGET_ARCH_ABI=$2
}


# This method is only called in CI.
function cmake_x86_for_CI {
    prepare_workspace # fake an empty __generated_code__.cc to pass cmake.
    cmake ..  -DWITH_GPU=OFF -DWITH_MKLDNN=OFF -DLITE_WITH_X86=ON ${common_flags} -DLITE_WITH_PROFILE=ON

    # Compile and execute the gen_code related test, so it will generate some code, and make the compilation reasonable.
    make test_gen_code_lite -j$NUM_CORES_FOR_COMPILE
    make test_cxx_api_lite -j$NUM_CORES_FOR_COMPILE
    ctest -R test_cxx_api_lite
    ctest -R test_gen_code_lite
    make test_generated_code -j$NUM_CORES_FOR_COMPILE
}

function cmake_gpu {
    prepare_workspace
    cmake .. " -DWITH_GPU=ON {common_flags} -DLITE_WITH_GPU=ON"
}

function check_style {
    export PATH=/usr/bin:$PATH
    #pre-commit install
    clang-format --version

    if ! pre-commit run -a ; then
        git diff
        exit 1
    fi
}

function build_single {
    #make $1 -j$(expr $(nproc) - 2)
    make $1 -j$NUM_CORES_FOR_COMPILE
}

function build {
    make lite_compile_deps -j$NUM_CORES_FOR_COMPILE

    # test publish inference lib
    make publish_inference_lite
}

# It will eagerly test all lite related unittests.
function test_lite {
    local file=$1
    echo "file: ${file}"

    for _test in $(cat $file); do
        ctest -R $_test -V
    done
}

# Build the code and run lite server tests. This is executed in the CI system.
function build_test_server {
    mkdir -p ./build
    cd ./build
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/paddle/build/third_party/install/mklml/lib"
    cmake_x86_for_CI
    build

    test_lite $TESTS_FILE
}

function build_test_train {
    mkdir -p ./build
    cd ./build
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/paddle/build/third_party/install/mklml/lib"
    prepare_workspace # fake an empty __generated_code__.cc to pass cmake.
    cmake .. -DWITH_LITE=ON -DWITH_GPU=OFF -DWITH_PYTHON=ON -DLITE_WITH_X86=ON -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=OFF -DWITH_TESTING=ON -DWITH_MKL=OFF 
    #make lite_compile_deps -j$NUM_CORES_FOR_COMPILE
    make -j$NUM_CORES_FOR_COMPILE

    find -name "*.whl" | xargs pip2 install 
    python ../paddle/fluid/lite/python/lite_test.py

}

# test_arm_android <some_test_name> <adb_port_number>
function test_arm_android {
    local test_name=$1
    local port=$2
    if [[ "${test_name}x" == "x" ]]; then
        echo "test_name can not be empty"
        exit 1
    fi
    if [[ "${port}x" == "x" ]]; then
        echo "Port can not be empty"
        exit 1
    fi

    echo "test name: ${test_name}"
    adb_work_dir="/data/local/tmp"

    skip_list=("test_model_parser_lite" "test_mobilenetv1_lite" "test_mobilenetv2_lite" "test_resnet50_lite" "test_inceptionv4_lite" "test_light_api_lite" "test_apis_lite" "test_paddle_api_lite")
    for skip_name in ${skip_list[@]} ; do
        [[ $skip_name =~ (^|[[:space:]])$test_name($|[[:space:]]) ]] && echo "skip $test_name" && return
    done

    local testpath=$(find ./paddle/fluid -name ${test_name})

    # if [[ "$test_name" == "test_light_api" ]]; then
    #     local model_path=$(find . -name "lite_naive_model")
    #     arm_push_necessary_file $port $model_path $adb_work_dir
    # fi

    adb -s emulator-${port} push ${testpath} ${adb_work_dir}
    adb -s emulator-${port} shell chmod +x "${adb_work_dir}/${test_name}"
    adb -s emulator-${port} shell "./${adb_work_dir}/${test_name}"
}

function test_arm_model {
    local test_name=$1
    local port=$2
    local model_dir=$3

    if [[ "${test_name}x" == "x" ]]; then
        echo "test_name can not be empty"
        exit 1
    fi
    if [[ "${port}x" == "x" ]]; then
        echo "Port can not be empty"
        exit 1
    fi
    if [[ "${model_dir}x" == "x" ]]; then
        echo "Model dir can not be empty"
        exit 1
    fi

    echo "test name: ${test_name}"
    adb_work_dir="/data/local/tmp"

    testpath=$(find ./paddle/fluid -name ${test_name})
    adb -s emulator-${port} push ${model_dir} ${adb_work_dir}
    adb -s emulator-${port} push ${testpath} ${adb_work_dir}
    adb -s emulator-${port} shell chmod +x "${adb_work_dir}/${test_name}"
    local adb_model_path="${adb_work_dir}/`basename ${model_dir}`"
    adb -s emulator-${port} shell "${adb_work_dir}/${test_name} --model_dir=$adb_model_path"

}

function cmake_arm {
    prepare_workspace
    # $1: ARM_TARGET_OS in "android" , "armlinux"
    # $2: ARM_TARGET_ARCH_ABI in "armv8", "armv7" ,"armv7hf"
    # $3: ARM_TARGET_LANG in "gcc" "clang"
    cmake .. \
        -DWITH_GPU=OFF \
        -DWITH_MKL=OFF \
        -DWITH_LITE=ON \
        -DLITE_WITH_CUDA=OFF \
        -DLITE_WITH_X86=OFF \
        -DLITE_WITH_ARM=ON \
        -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=ON \
        -DWITH_TESTING=ON \
        -DARM_TARGET_OS=$1 -DARM_TARGET_ARCH_ABI=$2 -DARM_TARGET_LANG=$3
}

# $1: ARM_TARGET_OS in "android" , "armlinux"
# $2: ARM_TARGET_ARCH_ABI in "armv8", "armv7" ,"armv7hf"
# $3: ARM_TARGET_LANG in "gcc" "clang"
function build_arm {
    os=$1
    abi=$2
    lang=$3

    cur_dir=$(pwd)
    if [[ ${os} == "armlinux" ]]; then
        # TODO(hongming): enable compile armv7 and armv7hf on armlinux, and clang compile
        if [[ ${lang} == "clang" ]]; then
            echo "clang is not enabled on armlinux yet"
            return 0
        fi
        if [[ ${abi} == "armv7hf" ]]; then
            echo "armv7hf is not supported on armlinux yet"
            return 0
        fi
        if [[ ${abi} == "armv7" ]]; then
            echo "armv7 is not supported on armlinux yet"
            return 0
        fi
    fi

    if [[ ${os} == "android" && ${abi} == "armv7hf" ]]; then
        echo "android do not need armv7hf"
        return 0
    fi

    build_dir=$cur_dir/build.lite.${os}.${abi}.${lang}
    mkdir -p $build_dir
    cd $build_dir

    cmake_arm ${os} ${abi} ${lang}
    build $TESTS_FILE

    # test publish inference lib
    make publish_inference_lite
}

# $1: ARM_TARGET_OS in "android" , "armlinux"
# $2: ARM_TARGET_ARCH_ABI in "armv8", "armv7" ,"armv7hf"
# $3: ARM_TARGET_LANG in "gcc" "clang"
# $4: android test port
# Note: test must be in build dir
function test_arm {
    os=$1
    abi=$2
    lang=$3
    port=$4

    if [[ ${os} == "armlinux" ]]; then
        # TODO(hongming): enable test armlinux on armv8, armv7 and armv7hf
        echo "Skip test arm linux yet. armlinux must in another docker"
        return 0
    fi

    if [[ ${os} == "android" && ${abi} == "armv7hf" ]]; then
        echo "android do not need armv7hf"
        return 0
    fi
   
    echo "test file: ${TESTS_FILE}"
    for _test in $(cat $TESTS_FILE); do
        test_arm_android $_test $port
    done
}

function prepare_emulator {
    local port_armv8=$1
    local port_armv7=$2

    adb kill-server
    adb devices | grep emulator | cut -f1 | while read line; do adb -s $line emu kill; done
    # start android armv8 and armv7 emulators first
    echo n | avdmanager create avd -f -n paddle-armv8 -k "system-images;android-24;google_apis;arm64-v8a"
    echo -ne '\n' | ${ANDROID_HOME}/emulator/emulator -avd paddle-armv8 -noaudio -no-window -gpu off -port ${port_armv8} &
    sleep 1m
    echo n | avdmanager create avd -f -n paddle-armv7 -k "system-images;android-24;google_apis;armeabi-v7a"
    echo -ne '\n' | ${ANDROID_HOME}/emulator/emulator -avd paddle-armv7 -noaudio -no-window -gpu off -port ${port_armv7} &
    sleep 1m
}

function arm_push_necessary_file {
    local port=$1
    local testpath=$2
    local adb_work_dir=$3

    adb -s emulator-${port} push ${testpath} ${adb_work_dir}
}


# We split the arm unittest into several sub-tasks to parallel and reduce the overall CI timetime.
# sub-task1
function build_test_arm_subtask_android {
    ########################################################################
    # job 1-4 must be in one runner
    port_armv8=5554
    port_armv7=5556

    prepare_emulator $port_armv8 $port_armv7

    # job 1
    build_arm "android" "armv8" "gcc"
    test_arm "android" "armv8" "gcc" ${port_armv8}
    cd -

    # job 2
    #build_arm "android" "armv8" "clang"
    #test_arm "android" "armv8" "clang" ${port_armv8}
    #cd -

    # job 3
    build_arm "android" "armv7" "gcc"
    test_arm "android" "armv7" "gcc" ${port_armv7}
    cd -

    # job 4
    #build_arm "android" "armv7" "clang"
    #test_arm "android" "armv7" "clang" ${port_armv7}
    #cd -

    adb devices | grep emulator | cut -f1 | while read line; do adb -s $line emu kill; done
    echo "Done"
}

# sub-task2
function build_test_arm_subtask_armlinux {
    ########################################################################
    # job 1-4 must be in one runner
    port_armv8=5554
    port_armv7=5556

    prepare_emulator $port_armv8 $port_armv7

    cur=$PWD

    # job 5
    build_arm "armlinux" "armv8" "gcc" $port_armv8
    test_arm "armlinux" "armv8" "gcc" $port_armv8
    cd $cur

    # job 6
    build_arm "armlinux" "armv7" "gcc" $port_armv8
    test_arm "armlinux" "armv7" "gcc" $port_armv8
    cd $cur

    # job 7
    build_arm "armlinux" "armv7hf" "gcc" $port_armv8
    test_arm "armlinux" "armv7hf" "gcc" $port_armv8
    cd $cur

    adb devices | grep emulator | cut -f1 | while read line; do adb -s $line emu kill; done
    echo "Done"
}

# sub-task-model
function build_test_arm_subtask_model {
    local port_armv8=5554
    local port_armv7=5556
    # We just test following single one environment to limit the CI time.
    local os=android
    local abi=armv8
    local lang=gcc

    local test_name=$1
    local model_name=$2

    cur_dir=$(pwd)
    build_dir=$cur_dir/build.lite.${os}.${abi}.${lang}
    mkdir -p $build_dir
    cd $build_dir
    cmake_arm $os $abi $lang
    make $test_name -j$NUM_CORES_FOR_COMPILE

    prepare_emulator $port_armv8 $port_armv7

    # just test the model on armv8
    test_arm_model $test_name $port_armv8 "./third_party/install/$model_name"

    adb devices | grep emulator | cut -f1 | while read line; do adb -s $line emu kill; done
    echo "Done"
}


# this test load a model, optimize it and check the prediction result of both cxx and light APIS.
function test_arm_predict_apis {
    local port=$1
    local workspace=$2
    local naive_model_path=$3
    local api_test_path=$(find . -name "test_apis_lite")
    # the model is pushed to ./lite_naive_model
    adb -s emulator-${port} push ${naive_model_path} ${workspace}
    adb -s emulator-${port} push $api_test_path ${workspace}

    # test cxx_api first to store the optimized model.
    adb -s emulator-${port} shell ./test_apis_lite --model_dir ./lite_naive_model --optimized_model ./lite_naive_model_opt
}


# Build the code and run lite arm tests. This is executed in the CI system.
function build_test_arm {
    ########################################################################
    # job 1-4 must be in one runner
    port_armv8=5554
    port_armv7=5556

    build_test_arm_subtask_android
    build_test_arm_subtask_armlinux
}


############################# MAIN #################################
function print_usage {
    echo -e "\nUSAGE:"
    echo
    echo "----------------------------------------"
    echo -e "cmake_x86: run cmake with X86 mode"
    echo -e "cmake_cuda: run cmake with CUDA mode"
    echo -e "--arm_os=<os> --arm_abi=<abi> cmake_arm: run cmake with ARM mode"
    echo
    echo -e "build: compile the tests"
    echo -e "--test_name=<test_name> build_single: compile single test"
    echo
    echo -e "test_server: run server tests"
    echo -e "--test_name=<test_name> --adb_port_number=<adb_port_number> test_arm_android: run arm test"
    echo "----------------------------------------"
    echo
}

function main {
    # Parse command line.
    for i in "$@"; do
        case $i in
            --tests=*)
                TESTS_FILE="${i#*=}"
                shift
                ;;
            --test_name=*)
                TEST_NAME="${i#*=}"
                shift
                ;;
            --arm_os=*)
                ARM_OS="${i#*=}"
                shift
                ;;
            --arm_abi=*)
                ARM_ABI="${i#*=}"
                shift
                ;;
            --arm_lang=*)
                ARM_LANG="${i#*=}"
                shift
                ;;
            --arm_port=*)
                ARM_PORT="${i#*=}"
                shift
                ;;
            build)
                build $TESTS_FILE
                build $LIBS_FILE
                shift
                ;;
            build_single)
                build_single $TEST_NAME
                shift
                ;;
            cmake_x86)
                cmake_x86
                shift
                ;;
            cmake_opencl)
                cmake_opencl $ARM_OS $ARM_ABI
                shift
                ;;
            cmake_cuda)
                cmake_cuda
                shift
                ;;
            cmake_arm)
                cmake_arm $ARM_OS $ARM_ABI $ARM_LANG
                shift
                ;;
            build_arm)
                build_arm $ARM_OS $ARM_ABI $ARM_LANG
                shift
                ;;
            test_server)
                test_lite $TESTS_FILE
                shift
                ;;
            test_arm)
                build_arm $ARM_OS $ARM_ABI $ARM_LANG $ARM_PORT
                shift
                ;;
            test_arm_android)
                test_arm_android $TEST_NAME $ARM_PORT
                shift
                ;;
            build_test_server)
                build_test_server
                shift
                ;;
            build_test_train)
                build_test_train
                shift
                ;;
            build_test_arm)
                build_test_arm
                shift
                ;;
            build_test_arm_subtask_android)
                build_test_arm_subtask_android
                shift
                ;;
            build_test_arm_subtask_armlinux)
                build_test_arm_subtask_armlinux
                shift
                ;;
            build_test_arm_model_mobilenetv1)
                build_test_arm_subtask_model test_mobilenetv1_lite mobilenet_v1
                shift
                ;;
            build_test_arm_model_mobilenetv2)
                build_test_arm_subtask_model test_mobilenetv2_lite mobilenet_v2_relu
                shift
                ;;
            build_test_arm_model_resnet50)
                build_test_arm_subtask_model test_resnet50_lite resnet50
                shift
                ;;
            build_test_arm_model_inceptionv4)
                build_test_arm_subtask_model test_inceptionv4_lite inception_v4_simple
                shift
                ;;
            check_style)
                check_style
                shift
                ;;
            check_need_ci)
                check_need_ci
                shift
                ;;
            *)
                # unknown option
                print_usage
                exit 1
                ;;
        esac
    done
}

main $@
