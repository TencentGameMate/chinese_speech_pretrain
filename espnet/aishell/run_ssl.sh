#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

ngpu=4
stage=1
stop_stage=13

train_set=train
valid_set=dev
test_sets="dev test"

# ssl realted config
expdir=exp/hubert-base; asr_config=conf/train_asr_conformer_hubert-base.yaml
# expdir=exp/hubert-large; asr_config=conf/train_asr_conformer_hubert-large.yaml
# expdir=exp/w2v2-base; asr_config=conf/train_asr_conformer_w2v2-base.yaml
# expdir=exp/w2v2-large; asr_config=conf/train_asr_conformer_w2v2-large.yaml

inference_config=conf/decode_asr_rnn.yaml

lm_config=conf/train_lm_transformer.yaml
use_lm=true
use_wordlm=false

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors="0.9 1.0 1.1"

./asr.sh                                               \
    --ngpu ${ngpu}                                     \
    --stage ${stage}                                   \
    --stop_stage ${stop_stage}                         \
    --expdir ${expdir}                                 \
    --lang zh                                          \
    --audio_format wav                                 \
    --feats_type raw                                   \
    --token_type char                                  \
    --use_lm ${use_lm}                                 \
    --use_word_lm ${use_wordlm}                        \
    --lm_config "${lm_config}"                         \
    --asr_config "${asr_config}"                       \
    --inference_config "${inference_config}"           \
    --gpu_inference true                               \
    --train_set "${train_set}"                         \
    --valid_set "${valid_set}"                         \
    --test_sets "${test_sets}"                         \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --lm_fold_length 150 \
    --lm_train_text "data/${train_set}/text" "$@"
