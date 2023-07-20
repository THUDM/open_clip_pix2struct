import io
import math

import datasets
import numpy as np
import torch
from PIL import Image
from sat.data_utils.datasets import SimpleDistributedWebDataset
# from sat.helpers import print_all
from torchvision.transforms import ToTensor, Normalize
import json
import random
import torchvision.datasets


def parse_resize(img_bytes, h, w, method='fixed', arlist=None, input_image=False):
    if not input_image:
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        except Exception as e:
            print_all(e)
            raise e
    else:
        img = img_bytes

    if method == 'fixed':  # fixed size
        img = img.resize((h, w))  # Pillow-SIMD is needed
    elif method == 'patch-resize':
        # fixed number of patches, resize image to fit patch size
        totalpatch, lpatch = h, w  # different meaning in this setting
        # example arlist: np.array [105/4, ..., 4/105] width/height
        imgar = img.size[0] * 1. / img.size[1]
        # find the closest aspect ratio
        ar, npatch = arlist[np.argmin(np.abs(arlist[:, 0] - imgar))]
        target_width = (npatch * ar) ** 0.5 * lpatch
        target_height = target_width / ar
        img = img.resize((round(target_width), round(target_height)))
    elif method == 'patch-crop':
        npatch, lpatch = h, w
        imgar = img.size[0] * 1. / img.size[1]
        raise NotImplementedError
    elif method == 'patch-resize-2':
        # variable number of patches to find the maximize rectangle
        totalpatch, lpatch = h, w
        # maximize scale s.t.
        scale = math.sqrt(totalpatch * (lpatch / img.size[1]) * (lpatch / img.size[0]))
        num_feasible_rows = max(min(math.floor(scale * img.size[1] / lpatch), totalpatch), 1)
        num_feasible_cols = max(min(math.floor(scale * img.size[0] / lpatch), totalpatch), 1)
        target_height = max(num_feasible_rows * lpatch, 1)
        target_width = max(num_feasible_cols * lpatch, 1)
        img = img.resize((round(target_width), round(target_height)))
    elif method == 'patch-crop-2':
        totalpatch, lpatch = h, w
        # maximize scale s.t.
        scale = math.sqrt(totalpatch * (lpatch / img.size[1]) * (lpatch / img.size[0]))
        num_feasible_rows = max(min(math.floor(scale * img.size[1] / lpatch), totalpatch), 1)
        num_feasible_cols = max(min(math.floor(scale * img.size[0] / lpatch), totalpatch), 1)
        target_height = max(num_feasible_rows * lpatch, 1)
        target_width = max(num_feasible_cols * lpatch, 1)
        img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)))
        img = img.crop((0, 0, target_width, target_height))
    return img


def image_transform(sample, size=(224, 224), resize_method='fixed', arlist=None, input_image=False):
    # image
    if ('png' in sample or 'jpg' in sample):
        img_bytes = sample['png'] if 'png' in sample else sample['jpg']
        img = parse_resize(img_bytes, size[0], size[1], method=resize_method, arlist=arlist, input_image=input_image)
        img = ToTensor()(img)
        OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
        OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
        normalize = Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)
        img = normalize(img)

        if resize_method in ['patch-resize', 'patch-crop', 'patch-resize-2', 'patch-crop-2']:
            # split image into patches
            npatch, lpatch = size
            rows, cols = img.size(1) // lpatch, img.size(2) // lpatch
            img = img.view(3, rows, lpatch, cols, lpatch).permute(1, 3, 2, 4, 0).contiguous()
            img = img.view(-1, lpatch ** 2 * 3)
            # pad to npatch
            img = torch.cat([img, torch.zeros(npatch - img.size(0), lpatch ** 2 * 3)],
                            dim=0)  # [seqlen, patch^2 * 3]
            position_ids = torch.zeros(npatch, 2, dtype=torch.long) - 1
            # 2d position [seqlen, 2]
            position_ids[:rows * cols, 0] = torch.arange(rows * cols) // cols
            position_ids[:rows * cols, 1] = torch.arange(rows * cols) % cols
            size = (rows, cols)
            return img, position_ids, size
        else:
            raise Exception
    else:
        raise Exception
    # TODO other data key


def resize_fn(src, size=(224, 224), resize_method='fixed', tokenizer=None):
    ''' Use as a middleware in SimpleDistributedWebDataset
        If resize_method == 'fixed': resize image to (height, width).
        If resize_method == 'patch-resize':
            Resize image to fit (num_patch, patch_size) by finding the closest aspect ratio by varying number of patches in [0.75num_patch, num_patch].
            The returned tensor is padded to (num_patch, 3*patch_size^2).
            Will also return a 2D position_ids tensor of shape (num_patch, 2).
        If resize_method == 'patch-crop':
            similar to patch-resize, but crop the image to target HW instead of resize.
        If resize_method == 'patch-resize-2':
            similar to patch-resize, but find the max rectangle instead of the closest aspect ratio. used in Pix2Struct.
        If resize_method == 'patch-crop-2':
            similar to patch-crop, but find the max rectangle instead of the closest aspect ratio.
    Args:
        src: Iterable, each sample contains a 'png' or 'jpg' key and a 'txt' key
        size: (height, width) for fixed resize method, (num_patch, patch_size) for patch resize method.
        resize_method: fixed, patch-resize or patch-crop.
    '''
    if resize_method == 'patch-resize':
        npatch, lpatch = size
        # factorize npatch
        res = []
        for patch in range(npatch // 4 * 3, npatch + 1):
            res.extend([[patch // i * 1. / i, patch] for i in range(1, patch + 1) if patch % i == 0])
        arlist = np.array(res)
    else:
        arlist = None
    for r in src:
        ret = {}
        # text
        if 'txt' in r:
            if isinstance(r['txt'], list):
                # multiple text, randomly choose one
                txt0 = r['txt'].sample()
            else:
                txt0 = r['txt']
            if isinstance(txt0, str):
                txt = txt0
            elif isinstance(txt0, bytes):
                txt = txt0.decode('utf-8')
        elif 'json' in r:
            txt = json.loads(r['json'].decode('utf-8'))['txt']
            if isinstance(txt, list):
                txt = random.choice(txt)
        else:
            print("NO text")
            txt = ""

        if tokenizer is not None:
            ret['txt'] = tokenizer(txt)[0]
        else:
            ret['txt'] = txt

        img, position_ids, image_size = image_transform(r, size=size, resize_method=resize_method, arlist=arlist)
        ret["jpg"] = img
        ret["position_ids"] = position_ids
        ret["size"] = image_size
        yield ret


class MyImageNet(torchvision.datasets.ImageFolder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = "patch-crop2"
        self.size = (400, 14)
        if self.method == 'patch-crop2':
            npatch, lpatch = self.size
            # factorize npatch
            res = []
            for patch in range(npatch // 4 * 3, npatch + 1):
                res.extend([[patch // i * 1. / i, patch] for i in range(1, patch + 1) if patch % i == 0])
            self.arlist = np.array(res)
        else:
            self.arlist = None

    def __getitem__(self, index: int):
        sample, target = super().__getitem__(index)

        sample, position_ids, image_size = image_transform({"jpg": sample}, size=self.size, resize_method=self.method,
                                                  arlist=self.arlist, input_image=True)
        return sample, position_ids, image_size, target
