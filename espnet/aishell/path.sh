MAIN_ROOT=<YOUR_DIR>/espnet
FAIRSEQ_ROOT=<YOUR_DIR>/fairseq
S3PRL_ROOT=<YOUR_DIR>/s3prl

export PATH=$PWD/utils/:$PATH
export PATH=$MAIN_ROOT/utils:$MAIN_ROOT/tools/sctk-2.4.10/bin:$MAIN_ROOT/espnet/bin:$PATH
export LC_ALL=C

# Set PYTHONPATH
export PYTHONPATH=${MAIN_ROOT}:${FAIRSEQ_ROOT}:${S3PRL_ROOT}:${PYTHONPATH:-}

export OMP_NUM_THREADS=1

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8

# You need to change or unset NCCL_SOCKET_IFNAME according to your network environment
# https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/env.html#nccl-socket-ifname
export NCCL_SOCKET_IFNAME="^lo,docker,virbr,vmnet,vboxnet"
