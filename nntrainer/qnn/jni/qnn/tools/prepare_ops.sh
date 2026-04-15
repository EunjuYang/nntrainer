#!/bin/bash

if [ "x${HEXAGON_SDK_ROOT}" = "x" ]; then
    echo "HEXAGON_SDK_ROOT is not set, we will set evn using /local/mnt/workspace/Qualcomm/Hexagon_SDK/5.5.2.0/setup_sdk_env.source"
    # ln -s /local/mnt/workspace/Qualcomm/Hexagon_SDK/5.5.2.0/ HexagonSDK
    if [ ! -d "./HexagonSDK" ]; then
	ln -s /local/mnt/workspace/Qualcomm/Hexagon_SDK/5.5.2.0/ HexagonSDK	
	# ln -s /local/mnt/workspace/Qualcomm/Hexagon_SDK/6.1.0.1/ HexagonSDK
    fi
    source HexagonSDK/setup_sdk_env.source
fi

# echo "QNN_SDK_ROOT is not set, we will set /opt/qcom/aistack/qnn/2.20.0.240223"
# QNN_SDK_ROOT=/opt/qcom/aistack/qairt/2.29.0.241129
# QNN_SDK_ROOT=/opt/qcom/aistack/qnn/2.20.0.240223
# QNN_SDK_ROOT=/opt/qcom/aistack/qairt/2.22.0.240425/
QNN_SDK_ROOT=/opt/qcom/aistack/qairt/2.31.0.250130
QNN_SDK_LIB=$QNN_SDK_ROOT/lib/aarch64-android
ln -s $QNN_HOME qairt
# ln -s /opt/qcom/aistack/qnn/2.20.0.240223/ qairt
# export QNN_SDK_ROOT=/opt/qcom/aistack/qnn/2.20.0.240223
export QNN_SDK_ROOT=$QNN_SDK_ROOT
source ${QNN_SDK_ROOT}/bin/envsetup.sh

echo "QNN_SDK_ROOT=./qairt"
echo "HEXAGON_SDK_ROOT=./HexagonSDK"

echo "ANDROID_ROOT_DIR=${ANDROID_ROOT_DIR}"
echo "ANDROID_NDK_ROOT=${ANDROID_NDK_ROOT}"
echo "QNX_BIN_DIR=${QNX_BIN_DIR}"
echo "LV_TOOLS_DIR=${LV_TOOLS_DIR}"
echo "LRH_TOOLS_DIR=${LRH_TOOLS_DIR}"

echo "DEFAULT_HEXAGON_TOOLS_ROOT=${DEFAULT_HEXAGON_TOOLS_ROOT}"
echo "DEFAULT_DSP_ARCH=${DEFAULT_DSP_ARCH}"
echo "DEFAULT_BUILD=${DEFAULT_BUILD}"
echo "DEFAULT_HLOS_ARCH=${DEFAULT_HLOS_ARCH}"
echo "DEFAULT_TOOLS_VARIANT=${DEFAULT_TOOLS_VARIANT}"
echo "DEFAULT_NO_OURT_INC=${DEFAULT_NO_QURT_INC}"
echo "DEFAULT_TREE=${DEFAULT_TREE}"
echo "CMAKE_ROOT_PATH=${CMAKE_ROOT_PATH}"
echo "DEBUGGER_UTILS=${DEBUGGER_UTILS}"
echo "HEXAGONSDK_TELEMATICS_ROOT=$HEXAGONSDK_TELEMATICS_ROOT}"

echo "AISW_SDK_ROOT=${AISW_SDK_ROOT}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "PATH=${PATH}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
echo "HEXAGON_TOOLS_DIR=${HEXAGON_TOOLS_DIR}"
echo "SNPE_ROOT=${SNPE_ROOT}"

cd LLaMAPackage

if [ ! -d "./build" ]; then
    # make htp_v75 && make htp_aarch64 && make htp_v79
    make htp_aarch64 && make htp_v75
fi

QNN_LIB_DIR=/data/local/tmp/QNN/qnn-lib/

adb shell mkdir -p $QNN_LIB_DIR


adb push $QNN_SDK_LIB/libQnnHtp.so $QNN_LIB_DIR
adb push $QNN_SDK_LIB/libQnnHtpV75Stub.so $QNN_LIB_DIR
#adb push $QNN_SDK_LIB/libQnnHtpV79Stub.so $QNN_LIB_DIR
adb push $QNN_SDK_LIB/libQnnHtpPrepare.so $QNN_LIB_DIR
adb push $QNN_SDK_LIB/libQnnHtpProfilingReader.so $QNN_LIB_DIR
adb push $QNN_SDK_LIB/libQnnHtpOptraceProfilingReader.so $QNN_LIB_DIR
adb push $QNN_SDK_LIB/libQnnHtpV75CalculatorStub.so $QNN_LIB_DIR
adb push $QNN_SDK_LIB/../hexagon-v75/unsigned/libQnnHtpV75Skel.so $QNN_LIB_DIR
#adb push $QNN_SDK_LIB/../hexagon-v79/unsigned/libQnnHtpV79Skel.so $QNN_LIB_DIR
adb push $QNN_SDK_LIB/libQnnSystem.so $QNN_LIB_DIR
adb push $QNN_SDK_LIB/libQnnSaver.so $QNN_LIB_DIR

adb push ../LLaMAPackage/build/aarch64-android/libQnnLLaMAPackage.so $QNN_LIB_DIR/libQnnLLaMAPackage_CPU.so
adb push ../LLaMAPackage/build/hexagon-v75/libQnnLLaMAPackage.so $QNN_LIB_DIR/libQnnLLaMAPackage_HTP.so
