PORT=0
#判断当前端口是否被占用，没被占用返回0，反之1
function Listening {
    TCPListeningnum=`netstat -an | grep ":$1 " | awk '$1 == "tcp" && $NF == "LISTEN" {print $0}' | wc -l`
    UDPListeningnum=`netstat -an | grep ":$1 " | awk '$1 == "udp" && $NF == "0.0.0.0:*" {print $0}' | wc -l`
    (( Listeningnum = TCPListeningnum + UDPListeningnum ))
    if [ $Listeningnum == 0 ]; then
        echo "0"
    else
        echo "1"
    fi
}

while [ $PORT == 0 ]; do
    temp1=`shuf -i 10000-50000 -n1`
    if [ `Listening $temp1` == 0 ] ; then
        PORT=$temp1
    fi
done

# -----需要自行配置-----
DEVICE='0' # -1表示使用CPU，其他表示使用多少号的GPU卡
CONFIG_FILE=config/$1 # 配置文件
MASTER_ADDR='127.0.0.1' # Master物理机器的IP地址
MASTER_PORT=${PORT} # Master物理机器的开放端口号
NNODES=1 # 总共几台物理机器
NODE_RANK=0 # 当前是第几台物理机器
SOCKET=eth0 # 需要绑定的网卡名称

# -----自动配置-----
TIMESTAMP=$(date +%Y_%m_%d_%H_%M_%S) # 时间戳
# TIMESTAMP="2024_04_19_08_13_32"
DEVICE_ARR=(${DEVICE//,/ })
DEVICE_LEN=${#DEVICE_ARR[@]}
echo $TIMESTAMP

# -----执行命令-----
# export NCCL_DEBUG=info
export NCCL_SOCKET_IFNAME=${SOCKET}
export NCCL_IB_DISABLE=1

CUDA_VISIBLE_DEVICES=$DEVICE torchrun \
    --nnodes=${NNODES} \
    --nproc_per_node=${DEVICE_LEN} \
	--master-addr=${MASTER_ADDR} \
	--master-port=${MASTER_PORT} \
	--node-rank=${NODE_RANK} \
	$2 \
	--gpu ${DEVICE} \
	--datetime ${TIMESTAMP} \
	--config_file ${CONFIG_FILE}