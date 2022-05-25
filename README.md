# chinese_speech_pretrain
chinese speech pretrained models


## 下游任务：中文语音识别

### WenetSpeech 实验结果
实验遵循 ESPnet [6,7,8] 工具包默认配置，即将预训练模型作为特征提取器，对于输入语音提取预训练模型各隐层表征进行加权求和，得到的最终语音表征将替换传统 FBank 特征作为 Conformer ASR 模型的输入。解码方法为 Beamsearch，没有使用语言模型 rescoring。

下游 ASR 任务训练中，使用 WenetSpeech train_s 100h 数据集作为有监督数据进行训练，分别对比了使用 FBank 特征、wav2vec 2.0 模型特征和 HuBERT 模型特征的字错误率 (Character Error Rate, CER) 结果。同时，额外对比了使用 train_m 集 1000h 和 train_l 集 1wh 中文数据 FBank 特征训练的模型结果。具体实验结果见下表：


| 输入特征          | 训练数据 | Dev 集 | Test_Net 集 | Test_Meeting 集 |
| ---               | ---      | ---    | ---         | ---             |
| FBank             | 100h     | 17.4   | 22.6        | 32.7            |
| FBank             | 1000h    | 11.6   | 14.6        | 22.4            |
| FBank             | 1wh      | 9.7    | 8.9         | 15.9            |
| wav2vec 2.0 BASE  | 100h     | 13.1   | 16.1        | 25.5            |
| wav2vec 2.0 LARGE | 100h     | 11.7   | 13.8        | 25.5            |
| HuBERT BASE       | 100h     | 12.6   | 14.7        | 21.3            |
| HuBERT LARGE      | 100h     | 10.0   | 10.2        | 14.5            |


## 参考文献

[1] Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, and Michael Auli, "wav2vec 2.0: A framework for self-supervised learning of speech representations," in Proc. NeurIPS, 2020.  
[2] Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, and Abdelrahman Mohamed, "HuBERT: Self-supervised speech representation learning by masked prediction of hidden units," IEEE/ACM Transactions of Audio, Speech, and Language Processing, vol. 29, pp. 3451-3460, 2021  
[3] Sanyuan Chen, Chengyi Wang, Zhengyang Chen, Yu Wu, Shujie Liu, Zhuo Chen, Jinyu Li, Naoyuki Kanda, Takuya Yoshioka, Xiong Xiao, Jian Wu, Long Zhou, Shuo Ren, Yanmin Qian, Yao Qian, Jian Wu, Michael Zeng, Xiangzhan Yu, and Furu Wei, "WavLM: Large-scale self-supervised pre-training for full stack speech processing," arXiv preprint arXiv:2110.13900, 2021  
[4] Binbin Zhang, Hang Lv, Pengcheng Guo, Qijie Shao, Chao Yang, Lei Xie, Xin Xu, Hui Bu, Xiaoyu Chen, Chenhen Zeng, Di Wu, and Zhendong Peng, "WenetSpeech: A 10000+ hours multi-domain Mandarin corpus for speech recognition," in Proc. ICASSP, 2021  
[5] Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng, David Grangier, and Michael Auli, "fairseq: A fast, extensible toolkit for sequence modeling," in Proc. NAACL, 2019.  
[6] Shinji Watanabe, Takaaki Hori, Shigeki Karita, Tomoki Hayashi, Jiro Nishitoba, Yuya Unno, Nelson Enrique Yalta Soplin, Jahn Heymann, Matthew Wiesner, Nanxin Chen, Adithya Renduchintala, and Tsubasa Ochiai, "ESPnet: End-to-end speech processing toolkit," in Proc. Interspeech, 2018, pp. 2207–2211  
[7] Pengcheng Guo, Florian Boyer, Xuankai Chang, Tomoki Hayashi, Yosuke Higuchi, Hirofumi Inaguma, Naoyuki Kamo, Chenda Li, Daniel Garcia-Romero, Jiatong Shi, Jing Shi, Shinji Watanabe, Kun Wei, Wangyou Zhang and Yuekai Zhang, "Recent development on ESPnet tookit boosted by Conformer," in Proc. ICASSP, 2021  
[8] Xuankai Chang, Takashi Maekaku, Pengcheng Guo, Jing Shi, Yen-Ju Lu, Aswin Shanmugam Subramanian, Tianzi Wang, Shu-wen Yang, Yu Tsao, Hung-yi Lee, and Shinji Watanabe, "An exploratino of self-supervised pretrained representations for end-to-end speech recognition," in Proc. ASRU, 2021  
