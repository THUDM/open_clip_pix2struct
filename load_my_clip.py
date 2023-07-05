import torch

import open_clip
from transformers import AutoTokenizer

'''
model.visual.input_patchnorm=False

'''
def load_my_clip():
    config_path = "open_clip_config.json"
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('CLIP-ViT-L-14-DataComp.XL-s13B-b90K', customized_config=config_path, cache_dir='/zhangpai21/checkpoints/clip')

    return model
if __name__ == "__main__":
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K', cache_dir='/zhangpai21/checkpoints/clip')
    tokenizer = AutoTokenizer.from_pretrained('laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K', cache_dir='/zhangpai21/checkpoints/clip', local_files_only=True)
    # conv1_weight = model.visual.conv1.weight
    # new_weight = conv1_weight.permute(0, 2, 3, 1).contiguous().view(1024, -1)
    # model_weight = model.state_dict()
    # model_weight['visual.patch_embedding.weight'] = new_weight
    # model_weight['visual.class_embedding'] += model_weight['visual.positional_embedding'].data[0]
    # model_weight.pop('visual.positional_embedding')
    # model_weight.pop('visual.conv1.weight')
    # torch.save(model_weight, '/zhangpai21/checkpoints/clip/clip_vit_l_14_adapted.pth')

    new_model = load_my_clip()

    # new_weight2 = torch.load('/zhangpai21/checkpoints/clip/clip_vit_l_14_adapted.pth')['visual.patch_embedding.weight']
    # breakpoint()
    image_size = (224, 224)
    test_x = torch.zeros(1, 3, 224, 224)
    # test_y = model(test_x)

    npatch = 400
    lpatch = 14
    rows = image_size[0] // lpatch
    cols = image_size[1] // lpatch
    test_x = test_x.view(1, 3, 16, lpatch, 16, lpatch).permute(0, 2, 4, 3, 5, 1).contiguous()
    test_x = test_x.view(1, -1, 14 ** 2 * 3)
    test_x = torch.cat([test_x, torch.zeros(1, npatch - test_x.size(1), lpatch ** 2 * 3)], dim=1)  # [seqlen, patch^2 * 3]
    position_ids = torch.zeros(npatch, 2, dtype=torch.long) - 1
    position_ids[:rows*cols, 0] = torch.arange(rows*cols) // cols
    position_ids[:rows*cols, 1] = torch.arange(rows*cols) % cols
    size = torch.zeros([1, 2])
    size[0, 0] = rows
    size[0, 1] = cols
    size = size.long()
    position_ids = position_ids.unsqueeze(0)
    test_y2 = new_model(test_x, position_ids=position_ids, image_size=size)
    # breakpoint()
    # deviation = torch.sum(torch.abs(test_y[0] - test_y2[0]))




'''
python -m training.main \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --train-data="/path/to/train_data.csv"  \
    --val-data="/path/to/validation_data.csv"  \
    --csv-img-key filepath \
    --csv-caption-key title \
    --imagenet-val=/path/to/imagenet/root/val/ \
    --warmup 10000 \
    --batch-size=128 \
    --lr=1e-3 \
    --wd=0.1 \
    --epochs=30 \
    --workers=8 \
    --model ViT-L-14

'''