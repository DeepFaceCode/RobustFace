import argparse
import logging
import os


import torch
from torch import distributed, softmax
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Criterions import CrossEntropy
from backbones import get_model
from dataset import get_dataloader
from losses import CurricularFace, BoundaryMargin, AdaFace, RobustFace, ArcFace
from lr_scheduler import PolyScheduler
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_logging import AverageMeter, init_logging
from utils.utils_distributed_sampler import setup_seed

assert torch.__version__ >= "1.9.0", "In order to enjoy the features of the new torch, \
we have upgraded the torch to 1.9.0. torch before than 1.9.0 may not work in the future."

try:
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    distributed.init_process_group("nccl")
except KeyError:
    world_size = 1
    rank = 0
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )

def main(args):
    # get config
    cfg = get_config(args.config)
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    torch.cuda.set_device(args.local_rank)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "RobustFace"))
        if rank == 0
        else None
    )

    train_loader = get_dataloader(
        cfg.rec,
        args.local_rank,
        cfg.batch_size,
        cfg.dali,
        cfg.seed,
        cfg.num_workers
    )

    train_loader_noise = get_dataloader(
        cfg.rec_noise,
        args.local_rank,
        int(cfg.batch_size * cfg.error_rate_open)+2,
        cfg.dali,
        cfg.seed,
        cfg.num_workers
    )

    backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()

    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[args.local_rank], bucket_cap_mb=16,
        find_unused_parameters=False)

    backbone.train()

    #  Margin Function
    if cfg.loss == 'robustface':
        margin_loss = RobustFace(
            cfg.num_classes,
            cfg.embedding_size,
            64,
            cfg.margin_list[0],  # for m
            cfg.margin_list[1],  # for m1
            cfg.t,
            cfg.error_rate_open+cfg.error_rate_close, # mu
            cfg.interclass_filtering_threshold
        ).train().cuda()
    elif cfg.loss == 'curricularface':
        margin_loss = CurricularFace(
            cfg.num_classes,
            cfg.embedding_size,
            s = 64,
            margin = 0.5
        ).train().cuda()
    elif cfg.loss == 'boundarymargin':
        margin_loss = BoundaryMargin(
            cfg.num_classes,
            cfg.embedding_size,
            s=64,
            marigin=0.5
        ).train().cuda()
    elif cfg.loss == 'adaface':
        margin_loss = AdaFace(
            cfg.num_classes,
            embedding_size=512
        ).train().cuda()
    elif cfg.loss == 'arcface':
        # For ArcFace
        margin_loss = ArcFace(
            cfg.num_classes,
            cfg.embedding_size,
            64,
            cfg.margin_list[0],
            cfg.interclass_filtering_threshold
        ).train().cuda()
    else:
        raise

    criterion = CrossEntropy()

    optimizer = torch.optim.SGD(
        params=[{"params": backbone.parameters()}, {"params": margin_loss.parameters()}],
        lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    cfg.total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.total_batch_size * cfg.num_epoch

    lr_scheduler = PolyScheduler(
        optimizer=optimizer,
        base_lr=cfg.lr,
        max_steps=cfg.total_step,
        warmup_steps=cfg.warmup_step,
        last_epoch=-1
    )

    start_epoch = 0
    global_step = 0
    if cfg.resume:
        dict_checkpoint = torch.load(os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
        margin_loss.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
        optimizer.load_state_dict(dict_checkpoint["state_optimizer"])
        lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        del dict_checkpoint

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec, summary_writer=summary_writer
    )
    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step = global_step,
        writer=summary_writer
    )

    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    for epoch in range(start_epoch, cfg.num_epoch):

        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)

        for (_, (img, local_labels)), (_, (img_noise, local_labels_noise)) in zip(enumerate(train_loader),# ms1mv2
                                                                                      enumerate(train_loader_noise)): # glint360k
            optimizer.zero_grad()
            global_step += 1

            # 确定要引入错误标签的样本数量
            num_samples_close = int(len(local_labels) * cfg.error_rate_close)
            num_samples_open = int(len(local_labels) * cfg.error_rate_open)

            # open noise
            img[:num_samples_open] = img_noise[:num_samples_open]

            for idx in range(num_samples_open, num_samples_open + num_samples_close):
                # close noise
                local_labels[idx] = (local_labels[idx] + 1) % 85742

            local_embeddings = backbone(img)

            if cfg.loss == 'boundarymargin':
                output, easy_num, noise_num, hard_num, m_ = margin_loss(local_embeddings, local_labels, epoch)
            elif cfg.loss == 'adaface':
                norms = torch.norm(local_embeddings, 2, -1, keepdim=True)
                output, easy_num, noise_num, hard_num, m_ = margin_loss(local_embeddings, local_labels, norms)
            else:
                output, easy_num, noise_num, hard_num, m_ = margin_loss(local_embeddings, local_labels)#XY,and other

            loss = criterion(output,local_labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            with torch.no_grad():
                callback_logging(global_step, loss, epoch, cfg.fp16, lr_scheduler.get_last_lr()[0], amp, easy_num, noise_num, hard_num, m_)
                if global_step % cfg.verbose == 0 and global_step > 0:
                    callback_verification(global_step, backbone)

        if cfg.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.module.state_dict(),
                "state_dict_softmax_fc": margin_loss.state_dict(),
                "state_optimizer": optimizer.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))

        if rank == 0:
            path_module = os.path.join(cfg.output, "model.pt")
            torch.save(backbone.module.state_dict(), path_module)

        if cfg.dali:
            train_loader.reset()

    if rank == 0:

        path_module = os.path.join(cfg.output, "model.pt")
        torch.save(backbone.module.state_dict(), path_module)

    distributed.destroy_process_group()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    parser.add_argument("--local_rank", type=int, default=0, help="local_rank")
    main(parser.parse_args())
