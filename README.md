## Readme

This repository contains the code for the paper [**Non-autoregressive Machine Translation with Probabilistic Context-free Grammar**](https://neurips.cc/virtual/2023/poster/71942).



This project is based on [fairseq](https://github.com/facebookresearch/fairseq) and [DA-Transformer](https://github.com/thu-coai/DA-Transformer).


#### PCFG-NAT files

We provide the fairseq plugins in the directory fs_plugins/, some of them (custom_ops/, utilities.py, translation_lev_moditied.py) are copied from the original [DA-Transformer](https://github.com/thu-coai/DA-Transformer).


```
DASpeech
├── __init__.py
├── criterions
│   ├── __init__.py
│   ├── nat_pcfg_loss.py                     ## PCFG-NAT loss
│   └── utilities.py
├── custom_ops                              ## CUDA implementations
│   ├── __init__.py
│   ├── pcfg_best_tree.cu                   ## best alignment for glat training
│   ├── pcfg_loss.cpp                       ## cpp wrapper of PCFG-NAT loss
│   ├── pcfg_loss.cu                        ## forward of PCFG-NAT loss
│   ├── pcfg_loss_backward.cu               ## backward of PCFG-NAT loss
│   ├── pcfg_viterbi.cu                     ## viterbi algorithm of PCFG-NAT inference
│   ├── pcfg_loss.py                        ## python wrapper of PCFG-NAT loss
│   ├── logsoftmax_gather.cu                ## logsoftmax gather
│   └── utilities.h
├── models
│   ├── __init__.py
│   └── glat_decomposed_with_link_two_hands_tri_pcfg.py ## PCFG-NAT model
│   └── lemon_tree.py   ## support tree structure of PCFG-NAT
└── tasks
    ├── __init__.py
    ├── translation_lev_modified.py   ## PCFG-NAT translation task
```

#### Requirements and Installation

* Python >= 3.7
* Pytorch == 1.10.1 (tested with cuda == 11.3)
* gcc >= 7.0.0
* Install fairseq via `pip install -e fairseq/.`

#### Preparing Data
Fairseq provides the preprocessed raw datasets here. Please build the binarized dataset by the following script:

```bash
input_dir=path/to/raw_data        # directory of raw text data
data_dir=path/to/binarized_data   # directory of the generated binarized data
src=en                            # source language id
tgt=de                            # target language id
fairseq-preprocess --source-lang ${src} --target-lang ${tgt} \
    --trainpref ${input_dir}/train.${src}-${tgt} --validpref ${input_dir}/valid.${src}-${tgt} --testpref ${input_dir}/test.${src}-${tgt} \
    --src-dict ${input_dir}/dict.${src}.txt --tgt-dict ${input_dir}/dict.${tgt}.txt \
    --destdir ${data_dir} --workers 32
```

#### Training

Here we provide the training script of PCFG-NAT on WMT-14 En-De, and the training scripts of PCFG-NAT on WMT17 En-Zh and WMT-16 En-Ro are in `train_scripts/`.
```bash
exp=exp_name
root=fairseq
data_dir=data_dir
checkpoint_dir=checkpoint_dir
user_dir=fs_plugins
fairseq-train ${data_dir} \
    --user-dir $user_dir \
    --task translation_lev_modified  --noise full_mask \
    --arch glat_decomposed_with_link_two_hands_tri_pcfg_base \
    --decoder-learned-pos --encoder-learned-pos \
    --share-all-embeddings --activation-fn gelu \
    --apply-bert-init \
    --links-feature feature:position --decode-strategy lookahead \
    --max-source-positions 128 --max-target-positions 1030 --src-upsample-scale 4.0 \
    --left-tree-layer 1 \
    --criterion nat_pcfg_loss \
    --length-loss-factor 0 --max-transition-length 99999 \
    --glat-p 0.5:0.1@200k --glance-strategy number-random \
    --no-force-emit \
    --optimizer adam --adam-betas '(0.9,0.999)' \
    --label-smoothing 0.0 --weight-decay 0.01 --dropout 0.1 \
    --lr-scheduler inverse_sqrt --warmup-updates 10000 \
    --clip-norm 0.1 --lr 0.0005 --warmup-init-lr '1e-07' --stop-min-lr '1e-09' \
    --min-loss-scale 0 --ddp-backend c10d \
    --max-tokens 2730 --update-freq 3 --grouped-shuffling \
    --max-update 300000 --max-tokens-valid 1024 \
    --save-interval 1  --save-interval-updates 10000 \
    --seed 0 --fp16 \
    --validate-interval 1       --validate-interval-updates 10000 \
    --skip-invalid-size-inputs-valid-test \
    --fixed-validation-seed 7 \
    --best-checkpoint-metric loss \
    --keep-last-epochs 32 \
    --keep-best-checkpoints 10 --save-dir ${checkpoint_dir} \
    --log-format 'simple' --log-interval 100
```
Most the command line arguments are the same as [fairseq](https://github.com/facebookresearch/fairseq) and [DA-Transformer](https://github.com/thu-coai/DA-Transformer).
`--left-tree-layer 1 \` means the local prefix tree in support tree only has one layer.


#### Evaluation

* Average the best 5 checkpoints.
* Here we provide the decoding script of PCFG-NAT on WMT-14 En-De, and the evaluation scripts of PCFG-NAT on WMT17 En-Zh and WMT-16 En-Ro are in `test_scripts/`.

```bash
exp=exp_name
root=fairseq
data_dir=data_dir
checkpoint_dir=checkpoint_dir
user_dir=fs_plugins

fairseq-generate ${data_dir} \
    --gen-subset test --user-dir $user_dir --task translation_lev_modified \
    --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 --beam 1 \
    --remove-bpe  --batch-size 1 --seed 0 \
    --model-overrides "{\"decode_strategy\":\"viterbi\", \"decode_viterbibeta\":1.0}" \
    --path $checkpoint_dir/average_best_5.pt
```

#### Citation

If this repository is useful for you, please cite as:
```
@inproceedings{
    gui2023pcfg,
    title={Non-autoregressive Machine Translation with Probabilistic Context-free Grammar},
    author={Gui, Shangtong and Shao, Chenze and Ma, Zhengrui and  Zhang, Xishan and Chen, Yunji and Feng, Yang},
    booktitle={Advances in Neural Information Processing Systems},
    year={2023},
}
```

