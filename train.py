import argparse
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
from utils.util import set_seed, load_specific_dict
from utils.logger import create_run_directories
from data_loader.loader import HMEDataset
import torch
from trainer.trainer import Trainer
from models.unet import VARG
from torch import optim
import torch.nn as nn
from models.diffusion import Diffusion, EMA
import copy
from diffusers import AutoencoderKL
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from models.loss import SupConLoss
from utils.checkpoint import load_varg_checkpoint
import os


def main(opt):
    """ load config file into cfg"""
    cfg_from_file(opt.cfg_file)
    assert_and_infer_cfg()
    """fix the random seed"""
    set_seed(cfg.TRAIN.SEED)
    """ prepare log file """
    logs = create_run_directories(cfg.OUTPUT_DIR, opt.cfg_file, opt.log_name)

    """ set mulit-gpu """
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get('LOCAL_RANK', opt.local_rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(opt.device, local_rank)
    

    """ set dataset"""
    train_dataset = HMEDataset(
        cfg.DATA_LOADER.IMAGE_PATH,
        cfg.DATA_LOADER.STYLE_PATH,
        cfg.DATA_LOADER.LAPLACE_PATH,
        cfg.DATA_LOADER.CONTENT_PATH,
        cfg.TRAIN.TYPE)

    print('number of training images: ', len(train_dataset))
    train_sampler = DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg.TRAIN.IMS_PER_BATCH,
                                               drop_last=False,
                                               collate_fn=train_dataset.collate_batch,
                                               num_workers=cfg.DATA_LOADER.NUM_THREADS,
                                               pin_memory=True,
                                               sampler=train_sampler)
    
    
    test_dataset = HMEDataset(
        cfg.DATA_LOADER.IMAGE_PATH,
        cfg.DATA_LOADER.STYLE_PATH,
        cfg.DATA_LOADER.LAPLACE_PATH,
        cfg.DATA_LOADER.CONTENT_PATH,
        cfg.TEST.TYPE)

    test_sampler = DistributedSampler(test_dataset)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=cfg.TEST.IMS_PER_BATCH,
                                              drop_last=False,
                                              collate_fn=test_dataset.collate_batch,
                                              pin_memory=True,
                                              num_workers=cfg.DATA_LOADER.NUM_THREADS,
                                              sampler=test_sampler)
    
    """build model architecture"""
    unet = VARG(in_channels=cfg.MODEL.IN_CHANNELS, model_channels=cfg.MODEL.EMB_DIM,
                     out_channels=cfg.MODEL.OUT_CHANNELS, num_res_blocks=cfg.MODEL.NUM_RES_BLOCKS, 
                     attention_resolutions=(1,1), channel_mult=(1, 1), num_heads=cfg.MODEL.NUM_HEADS, 
                     context_dim=cfg.MODEL.EMB_DIM).to(device)
    
    """load pretrained  model"""
    if opt.checkpoint:
        load_varg_checkpoint(unet, opt.checkpoint)
        print('load pretrained VARG model from {}'.format(opt.checkpoint))

    """load pretrained resnet18 model"""
    if len(opt.feat_model) > 0:
        checkpoint = torch.load(opt.feat_model, map_location=torch.device('cpu'))
        checkpoint['conv1.weight'] = checkpoint['conv1.weight'].mean(1).unsqueeze(1)
        miss, unexp = unet.conditioner.style_encoder.load_state_dict(checkpoint, strict=False)
        assert len(unexp) <= 32, "faile to load the pretrained model"
        print('load pretrained model from {}'.format(opt.feat_model))
    
    """Initialize the U-Net model for parallel training on multiple GPUs"""
    unet = DDP(unet, device_ids=[local_rank])
    """build criterion and optimizer"""
    criterion = dict(nce=SupConLoss(contrast_mode='all'), recon=nn.MSELoss())
    optimizer = optim.AdamW(unet.parameters(), lr=cfg.SOLVER.BASE_LR)

    diffusion = Diffusion(device=device, noise_offset=opt.noise_offset)

    vae = AutoencoderKL.from_pretrained(opt.stable_dif_path, subfolder="vae")
    """Freeze vae and text_encoder"""
    vae.requires_grad_(False)
    vae = vae.to(device)

    """build trainer"""
    trainer = Trainer(diffusion, unet, vae, criterion, optimizer, train_loader, logs, test_loader, device)
    trainer.train()

if __name__ == '__main__':
    """Parse input arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument('--vae-path', dest='stable_dif_path', type=str, required=True,
                        help='Local Stable Diffusion v1.5 directory containing the frozen vae subfolder')

    parser.add_argument('--config', dest='cfg_file', default='configs/crohme.yaml',
                        help='Config file for training (and optionally testing)')
    parser.add_argument('--encoder-checkpoint', dest='feat_model', default='', help='pre-trained ResNet-18 weights')
    parser.add_argument('--checkpoint', default='', help='pre-trained VARG tensor checkpoint')
    parser.add_argument('--run-name', default='varg',
                        dest='log_name', required=False, help='the filename of log')
    parser.add_argument('--noise-offset', dest='noise_offset', default=0, type=float, help='control the strength of noise')
    parser.add_argument('--device', type=str, default='cuda', help='device for training')
    parser.add_argument('--local-rank', '--local_rank', dest='local_rank', type=int, default=0,
                        help='Local CUDA rank; torchrun supplies LOCAL_RANK')
    opt = parser.parse_args()
    main(opt)
