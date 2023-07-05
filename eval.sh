python -m training.main \
    --dataset-type "webdataset" \
    --precision amp \
    --val-data="/zhangpai21/webdataset/laion-aes/train/laion-aes_part_004_00000079.tar"  \
    --batch-size=32 \
    --customized-config "/zhangpai21/workspace/yzy/open_clip/open_clip_config_evalnobitfit.json"