# 基于预训练模型的语音识别


1. 安装 ESPnet 工具包：https://github.com/espnet/espnet；
2. 安装 Fairseq 工具包：https://github.com/facebookresearch/fairseq
3. 安装 S3PRL 工具包：https://github.com/s3prl/s3prl；
4. 将提供的配置文件拷贝至 `espnet/egs2/aishell/asr1/conf` 目录，将提供的 `path.sh` 文件替换 `espnet/egs2/aishell/asr1` 目录下的 `path.sh` 文件，将提供的 `run_ssl.sh` 文件拷贝至 `espnet/egs2/aishell/asr1` 目录下；
5. 进入 `espnet/egs2/aishell/asr1` 目录，进行模型训练 `./run_ssl.sh`.
