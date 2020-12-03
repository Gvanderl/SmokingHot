# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
from human_models.apps.recon import reconWrapper


def simple_test(input_path='./sample_images',
                out_path='./results',
                ckpt_path='./checkpoints/pifuhd.pt',
                resolution=512,
                use_rect=True):
    resolution = str(resolution)

    start_id = -1
    end_id = -1
    cmd = ['--dataroot', input_path, '--results_path', out_path, \
           '--loadSize', '1024', '--resolution', resolution, '--load_netMR_checkpoint_path', \
           ckpt_path, \
           '--start_id', '%d' % start_id, '--end_id', '%d' % end_id]
    reconWrapper(cmd, use_rect)
