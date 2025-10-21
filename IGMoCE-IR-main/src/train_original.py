import random
from typing import List

import os
import pathlib
import numpy as np
import torch

from tqdm import tqdm
from datetime import datetime

import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger

from net.igmoce_ir import IGMoCEIR

from options import train_options
from utils.schedulers import LinearWarmupCosineAnnealingLR
from data.dataset_utils import AIOTrainDataset, CDD11
from utils.loss_utils import FFTLoss, PSNRLoss, SSIMloss, PerceptualLoss,EdgeLoss



class PLTrainModel(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        
        self.opt = opt
        self.balance_loss_weight = opt.balance_loss_weight

        self.net = IGMoCEIR(
            dim=opt.dim, 
            num_blocks=opt.num_blocks, 
            num_dec_blocks=opt.num_dec_blocks, 
            levels=len(opt.num_blocks),
            heads=opt.heads,
            topk=opt.topk, 
            num_experts=opt.num_exp_blocks,
            rank=opt.latent_dim,
            with_complexity=opt.with_complexity, 
            depth_type=opt.depth_type, 
            stage_depth=opt.stage_depth, 
            rank_type=opt.rank_type, 
            complexity_scale=opt.complexity_scale,
            dilations=opt.dilations,
            fusion_weight=opt.fusion_weight,
        )

        # 初始化多个损失函数
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMloss(loss_weight=opt.ssim_loss_weight)
        self.edge_loss = EdgeLoss(loss_weight=opt.edge_loss_weight)
        self.fft_loss = FFTLoss(loss_weight=opt.fft_loss_weight)
        self.perceptual_loss = PerceptualLoss(loss_weight=opt.perceptual_loss_weight)

        # 损失权重从 opt 中读取
        self.weights = {
            'l1': opt.l1_loss_weight,
            'ssim': opt.ssim_loss_weight,
            'edge': opt.edge_loss_weight,
            'fft': opt.fft_loss_weight,
            'perceptual': opt.perceptual_loss_weight,
            'balance': opt.balance_loss_weight
        }


    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch, de_id)
        balance_loss = self.net.total_loss

        # 多个损失计算
        loss_l1 = self.l1_loss(restored, clean_patch)
        loss_ssim = self.ssim_loss(restored, clean_patch)
        loss_edge = self.edge_loss(restored, clean_patch)
        loss_fft = self.fft_loss(restored, clean_patch)
        loss_perceptual = self.perceptual_loss(restored, clean_patch)

        # 加权总损失
        loss = (
                self.weights['l1'] * loss_l1 +
                self.weights['ssim'] * loss_ssim +
                self.weights['edge'] * loss_edge +
                self.weights['fft'] * loss_fft +
                self.weights['perceptual'] * loss_perceptual+
                self.balance_loss_weight * balance_loss
        )

        # 日志记录
        self.log("Train_Loss", loss, sync_dist=True)
        self.log_dict({
            "L1": loss_l1,
            "SSIM": loss_ssim,
            "Edge": loss_edge,
            "FFT": loss_fft,
            "Perceptual": loss_perceptual,
            "Balance":balance_loss
        }, sync_dist=True)
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("LR Schedule", lr, sync_dist=True)

        return loss


        
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=150)
        
        if self.opt.fine_tune_from:
            scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=1,max_epochs=self.opt.epochs)      
        return [optimizer],[scheduler]
                        


def main(opt):
    print("Options")
    print(opt)
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    time_stamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        
    log_dir = os.path.join("logs/", time_stamp)
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    if opt.wblogger:
        name = opt.model + "_" + time_stamp
        logger = WandbLogger(name=name, save_dir=log_dir, config=opt)
        
    else:
        logger = TensorBoardLogger(save_dir=log_dir)

    # Create model
    if opt.fine_tune_from:
        model = PLTrainModel.load_from_checkpoint(
            os.path.join(opt.ckpt_dir, opt.fine_tune_from, "last.ckpt"), opt=opt)
    else:
        model = PLTrainModel(opt)

    print(model)
    checkpoint_path = os.path.join(opt.ckpt_dir, time_stamp)
    pathlib.Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path, every_n_epochs=5, save_top_k=-1, save_last=True)
    
    # Create datasets and dataloaders
    if "CDD11" in opt.trainset:
        _, subset = opt.trainset.split("_")
        trainset = CDD11(opt, split="train", subset=subset)
    else:
        trainset = AIOTrainDataset(opt)
        
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True, drop_last=True, num_workers=opt.num_workers)
    
    # Create trainer
    trainer = pl.Trainer(max_epochs=opt.epochs,
                         accelerator="gpu",
                         devices=opt.num_gpus,
                         strategy="ddp_find_unused_parameters_true",
                         logger=logger,
                         callbacks=[checkpoint_callback],
                         accumulate_grad_batches=opt.accum_grad,
                         deterministic=True)
    
    # Optionally resume from a checkpoint
    if opt.resume_from:
        checkpoint_path = os.path.join(opt.ckpt_dir, opt.resume_from, "last.ckpt")
    else:
        checkpoint_path = None

    # Train model
    trainer.fit(
        model=model, 
        train_dataloaders=trainloader, 
        ckpt_path=checkpoint_path  # Specify the checkpoint path to resume from
    )
    


if __name__ == '__main__':
    train_opt = train_options()
    main(train_opt)


#  nohup python src/train_original_mulloss.py --model IGMoCE_IR --batch_size 16 --trainset CDD11_all --num_gpus 4 --de_type denoise_15 denoise_25 denoise_50 dehaze derain --data_file_dir /home/chujielie/hl/image_restoration/Data/ > train.log 2>&1
# python src/train.py --model [MoCE_IR_S/MoCE_IR] --batch_size 8 --de_type denoise_15 denoise_25 denoise_50 dehaze derain deblur synllie --trainset standard --num_gpus 4 --loss_type FFT --fft_loss_weight 0.1 --balance_loss_weight 0.01