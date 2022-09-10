"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""

from carvekit.api.high import HiInterface


# def test_init():
#     HiInterface(batch_size_seg=1, batch_size_matting=4,
#                 device='cpu',
#                 seg_mask_size=160, matting_mask_size=1024)
#     HiInterface(batch_size_seg=0, batch_size_matting=0,
#                 device='cpu',
#                 seg_mask_size=0, matting_mask_size=0)
import torch
from carvekit.api.high import HiInterface
def test_init():
    interface = HiInterface(batch_size_seg=5, batch_size_matting=1,
                                   device='cuda' if torch.cuda.is_available() else 'cpu',
                                   seg_mask_size=320, matting_mask_size=2048)
    images_without_background = interface(['./data/cat.jpg'])
    cat_wo_bg = images_without_background[0]
    cat_wo_bg.save('2.png')