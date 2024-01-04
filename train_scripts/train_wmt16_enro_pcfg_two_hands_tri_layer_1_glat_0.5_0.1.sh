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
    --max-source-positions 256 --max-target-positions 2048 --src-upsample-scale 4.0 \
    --left-tree-layer 1 \
    --criterion nat_pcfg_loss \
    --length-loss-factor 0 --max-transition-length 99999 \
    --glat-p 0.5:0.1@30k --glance-strategy number-random \
    --optimizer adam --adam-betas '(0.9,0.999)' \
    --label-smoothing 0.0 --weight-decay 0.01 --dropout 0.3 \
    --lr-scheduler inverse_sqrt --warmup-updates 10000 \
    --clip-norm 0.1 --lr 0.0007 --warmup-init-lr '1e-07' --stop-min-lr '1e-09' \
    --min-loss-scale 0 --ddp-backend c10d \
    --max-tokens 4096 --update-freq 4 --grouped-shuffling \
    --max-update 300000 --max-tokens-valid 1024 \
    --save-interval 1  --save-interval-updates 10000 \
    --patience 32 \
    --seed 0 --fp16 \
    --validate-interval 1 --validate-interval-updates 10000 \
    --fixed-validation-seed 7 \
    --best-checkpoint-metric loss \
    --keep-last-epochs 32 \
    --keep-best-checkpoints 5 --save-dir ${checkpoint_dir} \
    --log-format 'simple' --log-interval 100