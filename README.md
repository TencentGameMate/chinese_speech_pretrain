# chinese_speech_pretrain

我们使用 WenetSpeech [1] train_l 集的 1 万小时中文数据作为无监督预训练数据。数据主要来源于 YouTube 和 Podcast，覆盖了各种类型录制场景、背景噪声、说话方式等，其领域主要包括有声书、解说、纪录片、电视剧、访谈、新闻、朗读、演讲、综艺和其他等10大场景。我们基于 Fairseq 工具包 [2] 分别训练了 wav2vec 2.0 [3] 和 HuBERT [4] 模型，遵循 [3,4] 中模型配置，每个预训练模型模型包括 BASE 和 LARGE 两种大小。对于 BASE 模型，我们使用 8 张 A100 显卡，梯度累计为 8，模拟 64 张显卡进行训练。对于 LARGE 模型，我们使用 16 张 A100 显卡，梯度累计为 8，模拟 128 张显卡进行训练。

### 模型下载

| 模型                   | 预训练数据          | fairseq模型下载                                                                    | huggingface模型下载 |
| ---------------------- | ------------------- | ---------------------------------------------------------------------------------- | ------------------- |
| chinese-wav2vec2-base  | WenetSpeech train L | [chinese-wav2vec2-base](https://pan.baidu.com/s/1TwlSNDmihs_mjjPpNLhzoA) 提取码: d2hq |  [chinese-wav2vec2-base](https://huggingface.co/TencentGameMate/chinese-wav2vec2-base)  |
| chinese-wav2vec2-large | WenetSpeech train L | [chinese-wav2vec2-large](https://pan.baidu.com/s/1WbAv3PUqRWmHwwp6GsmLnw) 提取码: 7p8r | [chinese-wav2vec2-large](https://huggingface.co/TencentGameMate/chinese-wav2vec2-large)  |
| chinese-hubert-base    | WenetSpeech train L | [chinese-hubert-base](https://pan.baidu.com/s/1F3i1u27szmLtBnbMufEv0w) 提取码: xjiy | [chinese-hubert-base](https://huggingface.co/TencentGameMate/chinese-hubert-base)  |
| chinese-hubert-large   | WenetSpeech train L | [chinese-hubert-large](https://pan.baidu.com/s/1ReagTulgkESGpGJhB5DWRQ) 提取码: hhn7 | [chinese-hubert-large](https://huggingface.co/TencentGameMate/chinese-hubert-large)  |

## 下游任务：中文语音识别

为了验证预训练模型在下游 ASR 任务的效果，我们遵循 ESPnet [5,6,7] 工具包中的 Conformer [8] 模型实验配置，即将预训练模型作为特征提取器，对于输入语音提取预训练模型各隐层表征进行加权求和，得到的语音表征将替换传统 FBank 特征作为 Conformer ASR 模型的输入。

### Aishell 数据集 实验结果
我们使用 Aishell 178 小时训练集作为有监督数据进行训练，分别对比了使用 FBank 特征、wav2vec 2.0 BASE/LARGE 模型特征和 HuBERT BASE/LARGE 模型特征的字错误率 (Character Error Rate, CER) 结果。同时，我们额外对比了使用 WenetSpeech train_l 集 1 万小时中文数据进行训练时，其在 Aishell 测试集上的效果。训练数据使用了变速（0.9、1.0、1.1 倍）和 SpecAugment 数据增广技术，解码方式为 beam search，使用了基于 Transformer 的语言模型进行 rescoring。具体实验结果见下表：

| 输入特征      | 训练数据 | Dev | Test |
| ----------------- | -------- | --- | ---- |
| FBank [6]         | 178h     | 4.4 | 4.7  |
| FBank [1]         | 1wh      | /   | 3.9  |
| Wav2vec 2.0 BASE  | 178h     | 4.2 | 4.7  |
| Wav2vec 2.0 LARGE | 178h     | 3.8 | 4.1  |
| HuBERT Base       | 178h     | 4.1 | 4.3  |
| HuBERT LARGE      | 178h     | 3.1 | 3.3  |

### WenetSpeech 实验结果

我们使用 WenetSpeech train_s 100h 数据集作为有监督数据进行训练，分别对比了使用 FBank 特征、wav2vec 2.0 模型特征和 HuBERT 模型特征的字错误率 (Character Error Rate, CER) 结果。同时，额外对比了使用 train_m 集 1000h 和 train_l 集 1wh 中文数据 FBank 特征训练的模型结果。训练数据没有使用变速或 SpecAugment 数据增广技术，解码方式为 beam search，没有使用语言模型 rescoring。具体实验结果见下表：

| 输入特征          | 训练数据 | Dev 集 | Test_Net 集 | Test_Meeting 集 |
| ----------------- | -------- | ------ | ----------- | --------------- |
| FBank             | 100h     | 17.4   | 22.6        | 32.7            |
| FBank             | 1000h    | 11.6   | 14.6        | 22.4            |
| FBank             | 1wh      | 9.7    | 8.9         | 15.9            |
| wav2vec 2.0 BASE  | 100h     | 13.1   | 16.1        | 25.5            |
| wav2vec 2.0 LARGE | 100h     | 11.7   | 13.8        | 25.5            |
| HuBERT BASE       | 100h     | 12.6   | 14.7        | 21.3            |
| HuBERT LARGE      | 100h     | 10.0   | 10.2        | 14.5            |


### 模型使用

```python
# This model does not have a tokenizer as it was pretrained on audio alone. 
# In order to use this model speech recognition, a tokenizer should be created and the model should be fine-tuned on labeled text data.

# python package
# transformers==4.16.2

# fairseq 使用
import torch
import torch.nn.functional as F
import soundfile as sf
from fairseq import checkpoint_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path=""
wav_path=""

def postprocess(feats, normalize=False):
    if feats.dim() == 2:
        feats = feats.mean(-1)

    assert feats.dim() == 1, feats.dim()

    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    return feats

print("loading model(s) from {}".format(model_path))
models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
    [model_path],
    suffix="",
)
print("loaded model(s) from {}".format(model_path))
print(f"normalize: {saved_cfg.task.normalize}")


model = models[0]
model = model.to(device)
model = model.half()
model.eval()

wav, sr = sf.read(wav_path)
feat = torch.from_numpy(wav).float()
feat = postprocess(feat, normalize=saved_cfg.task.normalize)
feats = feat.view(1, -1)
padding_mask = (
    torch.BoolTensor(feats.shape).fill_(False)
)
inputs = {
    "source": feats.half().to(device),
    "padding_mask": padding_mask.to(device),
}

with torch.no_grad():
    logits = model.extract_features(**inputs)


# huggingface 使用

import torch
import torch.nn.functional as F
import soundfile as sf
from fairseq import checkpoint_utils

from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForPreTraining,
    Wav2Vec2Model,
)
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices

model_path=""
wav_path=""
mask_prob=0.0
mask_length=10

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
model = Wav2Vec2Model.from_pretrained(model_path)

# for pretrain: Wav2Vec2ForPreTraining
# model = Wav2Vec2ForPreTraining.from_pretrained(model_path)

model = model.to(device)
model = model.half()
model.eval()

wav, sr = sf.read(wav_path)
input_values = feature_extractor(wav, return_tensors="pt").input_values
input_values = input_values.half()
input_values = input_values.to(device)

# for Wav2Vec2ForPreTraining
# batch_size, raw_sequence_length = input_values.shape
# sequence_length = model._get_feat_extract_output_lengths(raw_sequence_length)
# mask_time_indices = _compute_mask_indices((batch_size, sequence_length), mask_prob=0.0, mask_length=2)
# mask_time_indices = torch.tensor(mask_time_indices, device=input_values.device, dtype=torch.long)

with torch.no_grad():
    outputs = model(input_values)
    last_hidden_state = outputs.last_hidden_state

    # for Wav2Vec2ForPreTraining
    # outputs = model(input_values, mask_time_indices=mask_time_indices, output_hidden_states=True)
    # last_hidden_state = outputs.hidden_states[-1]

```

欢迎大家使用我们提供的中文语音预训练模型开展研究工作，一起探索语音预训练模型在中文和相关众多场景下的应用。

<p align="center">
    <br>
    <img src="./figs/tencent.png" width="300"/><img src="./figs/zhijiyouxi.png" width="300"/>
    <br>
</p>

<p align="center">
    <br>
    <img src="./figs/npu@aslp.jpeg" width="500"/><img src="./figs/npu_wechat.jpeg" width="200"/>
    <br>
</p>



## 参考文献

[1] Binbin Zhang, Hang Lv, Pengcheng Guo, Qijie Shao, Chao Yang, Lei Xie, Xin Xu, Hui Bu, Xiaoyu Chen, Chenhen Zeng, Di Wu, and Zhendong Peng, "WenetSpeech: A 10000+ hours multi-domain Mandarin corpus for speech recognition," in Proc. ICASSP, 2021

[2] Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng, David Grangier, and Michael Auli, "fairseq: A fast, extensible toolkit for sequence modeling," in Proc. NAACL, 2019.

[3] Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, and Michael Auli, "wav2vec 2.0: A framework for self-supervised learning of speech representations," in Proc. NeurIPS, 2020.

[4] Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, and Abdelrahman Mohamed, "HuBERT: Self-supervised speech representation learning by masked prediction of hidden units," IEEE/ACM Transactions of Audio, Speech, and Language Processing, vol. 29, pp. 3451-3460, 2021

[5] Shinji Watanabe, Takaaki Hori, Shigeki Karita, Tomoki Hayashi, Jiro Nishitoba, Yuya Unno, Nelson Enrique Yalta Soplin, Jahn Heymann, Matthew Wiesner, Nanxin Chen, Adithya Renduchintala, and Tsubasa Ochiai, "ESPnet: End-to-end speech processing toolkit," in Proc. Interspeech, 2018, pp. 2207–2211

[6] Pengcheng Guo, Florian Boyer, Xuankai Chang, Tomoki Hayashi, Yosuke Higuchi, Hirofumi Inaguma, Naoyuki Kamo, Chenda Li, Daniel Garcia-Romero, Jiatong Shi, Jing Shi, Shinji Watanabe, Kun Wei, Wangyou Zhang and Yuekai Zhang, "Recent development on ESPnet tookit boosted by Conformer," in Proc. ICASSP, 2021

[7] Xuankai Chang, Takashi Maekaku, Pengcheng Guo, Jing Shi, Yen-Ju Lu, Aswin Shanmugam Subramanian, Tianzi Wang, Shu-wen Yang, Yu Tsao, Hung-yi Lee, and Shinji Watanabe, "An exploratino of self-supervised pretrained representations for end-to-end speech recognition," in Proc. ASRU, 2021

[8] Anmol Gulati, James Qin, Chung-Cheng Chiu, Niki Parmar, Yu Zhang, Jiahui Yu, Wei Han, Shibo Wang, Zhengdong Zhang, Yonghui Wu, and Ruoming Pan, "Conformer: Convolution-augmented Transformer for speech recognition," in Proc. Interspeech, 2020, pp.5036–5040

