exp=exp_name
root=fairseq
data_dir=data_dir
checkpoint_dir=checkpoint_dir
user_dir=fs_plugins

fairseq-generate ${data_dir} \
    --gen-subset test --user-dir $user_dir --task translation_lev_modified \
    --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 --beam 1 \
    --remove-bpe --max-tokens 1024 --seed 0 \
    --model-overrides "{\"decode_strategy\":\"viterbi\", \"decode_viterbibeta\":1.0}" \
    --path ${checkpoint_path}

