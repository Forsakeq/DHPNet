# train_dhp_net_backbone_warmup.py
# ------------------------------------------------------------
# Warmup 方案A：冻 backbone（RGB/FS Swin + MDFE encoder），训练其它所有模块
#   - 包含：decoder/预测头 + 所有 gating（slice_gating / omega / g_pos/g_neg 等）
#   - 目的：保证预测头能产生有效监督信号，gating 才“有梯度可学”。
# 其它部分保持不变（loss / dataloader / eval / 优化器分组 / 训练流程）。
# ------------------------------------------------------------


import os
import sys
import time
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

from DHP_Net import DHPNet, init_weights
from lib.utils import LFDataset


# -----------------------------
# Logger（控制台 + 文件）
# -----------------------------
class Logger(object):
    def __init__(self, filename="exp.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()


def prepare_dir(path: str, name: str):
    if os.path.exists(path) and (not os.path.isdir(path)):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_path = f"{path}_{name}_{ts}"
        print(f"[WARN] {name} path exists but is NOT a directory: {path}")
        print(f"[WARN] Use new {name} path: {new_path}")
        path = new_path
    os.makedirs(path, exist_ok=True)
    return path


# -----------------------------
# Losses
# -----------------------------
def focal_loss(pred, mask, gamma=2.0, alpha=0.25):
    pred_sig = torch.sigmoid(pred)
    pt = (1 - pred_sig) * mask + pred_sig * (1 - mask)
    focal_weight = (alpha * mask + (1 - alpha) * (1 - mask)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(pred, mask, reduction='none') * focal_weight
    return loss.mean()


def hybrid_e_loss(pred, mask):
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')

    pred_sig = torch.sigmoid(pred)
    mpred = pred_sig.mean(dim=(2, 3), keepdim=True)
    phiFM = pred_sig - mpred

    mmask = mask.mean(dim=(2, 3), keepdim=True)
    phiGT = mask - mmask

    EFM = (2.0 * phiFM * phiGT + 1e-8) / (phiFM * phiFM + phiGT * phiGT + 1e-8)
    QFM = (1 + EFM) * (1 + EFM) / 4.0
    eloss = 1.0 - QFM.mean(dim=(2, 3))

    inter = (pred_sig * mask).sum(dim=(2, 3))
    union = (pred_sig + mask).sum(dim=(2, 3))
    wiou = 1.0 - (inter + 1 + 1e-8) / (union - inter + 1 + 1e-8)

    return (bce + eloss + wiou).mean()


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is not None:
                p.grad.data.clamp_(-grad_clip, grad_clip)


# -----------------------------
# Focal-stack reshape
# fs: (B, 3*S, H, W) -> (B, S, 3, H, W)
# 支持 (S,3,H,W) / (B,S,3,H,W)
# -----------------------------
def fs_to_bs3hw(fs: torch.Tensor, num_slices: int = None):
    if fs.dim() == 5:
        return fs  # (B,S,3,H,W)

    if fs.dim() == 4:
        # (S,3,H,W)
        if fs.size(1) == 3:
            S, C, H, W = fs.shape
            assert C == 3
            fs = fs.unsqueeze(0)  # (1,S,3,H,W)
            return fs

        # (B,3*S,H,W)
        B, C, H, W = fs.shape
        assert C % 3 == 0, f"fs channel must be 3*S, got C={C}"
        S = C // 3
        if num_slices is not None:
            assert S == num_slices, f"S={S} != num_slices={num_slices}"
        return fs.contiguous().view(B, S, 3, H, W)

    raise ValueError(f"Unsupported fs shape: {fs.shape}")


# -----------------------------
# AdamW param grouping (no weight decay on bias/norm/1D/gate scalars)
# -----------------------------
def build_adamw_param_groups(model: torch.nn.Module, weight_decay: float):
    decay = []
    no_decay = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lname = name.lower()
        if (p.ndim == 1) or name.endswith('.bias') or ('norm' in lname) or ('bn' in lname):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {'params': decay, 'weight_decay': float(weight_decay)},
        {'params': no_decay, 'weight_decay': 0.0},
    ]




# -----------------------------
# Warmup 方案A：冻 backbone（但训练其它所有模块）
#   - 前 gating_warm_epochs：冻结 backbone_rgb/backbone_fs/mdfe_encoder
#   - 之后：恢复全量端到端训练
# 说明：为了尽量少改动，仍保留 gating_train_keywords 参数，但 warmup 逻辑不再依赖它。
# -----------------------------
def _split_keywords(s: str):
    if s is None:
        return []
    ks = []
    for k in str(s).split(","):
        k = k.strip().lower()
        if k:
            ks.append(k)
    return ks


def apply_gating_warmup_requires_grad(model: torch.nn.Module,
                                     epoch: int,
                                     warm_epochs: int,
                                     keywords,
                                     freeze_bn: bool = True):
    """
    方案A：warmup 期间只冻结 backbone，其它全部训练。
    Returns:
      warm_on (bool), trainable_params (int), total_params (int)
    """
    total_params = 0
    trainable_params = 0

    warm_on = (warm_epochs is not None) and (int(warm_epochs) > 0) and (epoch < int(warm_epochs))
    if warm_on:
        # 1) default: enable all
        for p in model.parameters():
            p.requires_grad = True

        # 2) freeze backbones by name prefix
        backbone_prefixes = ("backbone_rgb.", "backbone_fs.", "mdfe_encoder.")
        for name, p in model.named_parameters():
            total_params += p.numel()
            if name.startswith(backbone_prefixes):
                p.requires_grad = False
            if p.requires_grad:
                trainable_params += p.numel()

        # 3) put frozen backbones to eval to disable dropout/stochastic depth
        if hasattr(model, "backbone_rgb"):
            model.backbone_rgb.eval()
        if hasattr(model, "backbone_fs"):
            model.backbone_fs.eval()
        if hasattr(model, "mdfe_encoder"):
            model.mdfe_encoder.eval()

        # BN: keep in eval if you have BN layers (your current net基本没有BN，但保留兼容)
        if freeze_bn:
            for m in model.modules():
                if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm)):
                    m.eval()
    else:
        for p in model.parameters():
            total_params += p.numel()
            p.requires_grad = True
            trainable_params += p.numel()

    return warm_on, trainable_params, total_params


# -----------------------------
# Eval: 输出 MAE
# -----------------------------
@torch.no_grad()
def evaluate(args, model, datasets, device):
    model.eval()
    maes = []

    for dataset in datasets:
        test_loader = DataLoader(
            LFDataset(location=os.path.join(args.eval_data_location, dataset) + '/',
                      crop=False, train=False, image_size=args.image_size),
            batch_size=1, shuffle=False, num_workers=args.num_worker
        )

        mae_sum = 0.0
        for allfocus, fs, depth, gt, names in tqdm(test_loader, desc=f"Eval {dataset}", leave=True):
            rgb = allfocus.to(device)
            depth = depth.to(device)
            gt = gt.to(device)

            fs = fs.to(device)
            fs = fs_to_bs3hw(fs, num_slices=args.num_slices)  # (1,S,3,H,W)

            pred, contour_pred, coarse = model(fs, rgb, depth, return_aux=False)
            pred = torch.sigmoid(pred)
            mae_sum += torch.abs(pred - gt).mean().item()

        maes.append(mae_sum / max(len(test_loader), 1))

    return float(np.mean(maes))


# -----------------------------
# Train
# -----------------------------
def train(args, model, train_loader, device, optimizer, scheduler, writer):
    best_mae = 1e9
    best_epoch = -1

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    global_step = 0

    for epoch in range(args.epochs):
        model.train()

        # ----------------- Warmup(A): freeze backbone only -----------------
        warm_kw = _split_keywords(args.gating_train_keywords)  # 保留原参数（warmup逻辑不再依赖它）
        warm_on, n_tr, n_all = apply_gating_warmup_requires_grad(
            model, epoch=epoch, warm_epochs=args.gating_warm_epochs,
            keywords=warm_kw, freeze_bn=args.gating_warm_freeze_bn
        )
        if epoch == 0 or epoch == args.gating_warm_epochs:
            print(f"[BackboneWarmup] warm_on={warm_on} (epoch {epoch}/{args.epochs}), "
                  f"trainable_params={n_tr}/{n_all}, frozen_prefixes=['backbone_rgb','backbone_fs','mdfe_encoder']")
        if writer is not None:
            writer.add_scalar("train/gating_warm_on", 1.0 if warm_on else 0.0, epoch)
            writer.add_scalar("train/gating_trainable_ratio", float(n_tr) / float(max(n_all, 1)), epoch)

        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        t0 = time.time()

        # --------- omega: learnable scale (no omega_max warmup) ---------
        cur_om = args.omega_max

        # edge loss 权重调度（可选，但建议保留这个杠杆）
        edge_w = args.edge_loss_weight_after if epoch >= args.edge_weight_after_epoch else args.edge_loss_weight_before

        for it, (allfocus, fs, depth, gt, contour, names) in enumerate(train_loader):
            rgb = allfocus.to(device)
            depth = depth.to(device)
            gt = gt.to(device)
            contour = contour.to(device)

            fs = fs.to(device)
            fs = fs_to_bs3hw(fs, num_slices=args.num_slices)  # (B,S,3,H,W)

            with torch.cuda.amp.autocast(enabled=args.use_amp):
                if args.entropy_lambda > 0:
                    pred, edge_pred, coarse, aux_all = model(fs, rgb, depth, return_aux=True)
                else:
                    pred, edge_pred, coarse = model(fs, rgb, depth, return_aux=False)
                    aux_all = None

                loss = hybrid_e_loss(pred, gt) + edge_w * focal_loss(
                    edge_pred, contour, gamma=args.focal_gamma, alpha=args.focal_alpha
                ) + hybrid_e_loss(coarse, gt)

                if args.entropy_lambda > 0 and aux_all is not None:
                    u_list = []
                    for aux in aux_all:
                        if isinstance(aux, dict) and ("u_f" in aux) and (aux["u_f"] is not None):
                            u_list.append(aux["u_f"].mean())
                    if len(u_list) > 0:
                        u_mean = torch.stack(u_list).mean()
                        loss = loss - args.entropy_lambda * u_mean

                loss = loss / args.accum_steps

            scaler.scale(loss).backward()
            running += loss.item()

            if (it + 1) % args.accum_steps == 0:
                scaler.unscale_(optimizer)
                clip_gradient(optimizer, args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if (it + 1) % args.print_freq == 0:
                lr = optimizer.param_groups[0]["lr"]
                loss_show = running * args.accum_steps / args.print_freq
                print(f"epoch {epoch}, {it+1}/{len(train_loader)}, lr={lr:.6e}, "
                      f"loss={loss_show:.4f}, edge_w={edge_w:.2f}, omega_max={cur_om:.3f}")
                if writer is not None:
                    writer.add_scalar("train/loss", loss_show, global_step)
                    writer.add_scalar("train/lr", lr, global_step)
                    writer.add_scalar("train/edge_w", edge_w, global_step)
                    writer.add_scalar("train/omega_max", cur_om, global_step)
                running = 0.0

            global_step += 1

        if scheduler is not None:
            scheduler.step()

        dt = time.time() - t0
        print(f"[Epoch {epoch}] time={dt/60:.2f} min, lr={optimizer.param_groups[0]['lr']:.6e}")

        mae = evaluate(args, model, args.eval_dataset, device)
        print(f"[Epoch {epoch}] Val MAE = {mae:.6f}")
        if writer is not None:
            writer.add_scalar("val/mae", mae, epoch)

        if mae < best_mae:
            best_mae = mae
            best_epoch = epoch
            save_path = os.path.join(args.model_path, "best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"[BEST] epoch={epoch} MAE={mae:.6f} saved -> {save_path}")

        if epoch >= args.save_after and (epoch % args.save_every == 0):
            p = os.path.join(args.model_path, f"epoch_{epoch}.pth")
            torch.save(model.state_dict(), p)
            print(f"[Save] {p}")

    print(f"Training done. Best epoch={best_epoch}, best MAE={best_mae:.6f}")


# -----------------------------
# Args
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser("Train DHP-Net (MHFF + ERA) with backbone warmup")

    # paths
    p.add_argument("--model_path", type=str, default="models/DHP-Net")
    p.add_argument("--log_path", type=str, default="log/DHP-Net")
    p.add_argument("--pretrained_model", type=str, default="./pre_trained/swin_tiny_patch4_window7_224.pth")
    p.add_argument("--cuda", type=str, default="0")

    # data
    p.add_argument("--train_data_location", type=str, default="./data/train/DUTLF-FS/")
    p.add_argument("--eval_data_location", type=str, default="./data/test/")
    p.add_argument("--eval_dataset", nargs="+", default=["DUTLF-FS"])
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_worker", type=int, default=0)

    # model (H2)
    p.add_argument("--num_slices", type=int, default=12)
    p.add_argument("--ffn_expansion", type=float, default=2.0)
    p.add_argument("--enable_parallel_when_reso_ge", type=int, default=28)
    p.add_argument("--omega_init_bias", type=float, default=-2.0)
    p.add_argument("--omega_max", type=float, default=0.3)
    p.add_argument("--omega_warmup_epochs", type=int, default=15)

    # Warmup(A): freeze backbone only
    p.add_argument("--gating_warm_epochs", type=int, default=10,
                   help=">0 时启用：前 N 个 epoch 冻结 backbone（RGB/FS/Depth），训练其它所有模块")
    p.add_argument("--gating_train_keywords", type=str, default="slice_gating",
                   help="(兼容保留) warmup A 不依赖该参数；端到端阶段也不会用到。")
    p.add_argument("--gating_warm_freeze_bn", action="store_false", default=True,
                   help="关闭 warmup 期间 BN=eval（你的模型基本无BN，可忽略）")

    # optim
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--min_lr", type=float, default=1e-7)
    p.add_argument("--grad_clip", type=float, default=0.5)

    # misc
    p.add_argument("--accum_steps", type=int, default=4)
    p.add_argument("--print_freq", type=int, default=20)
    p.add_argument("--use_amp", action="store_true", default=False)

    # edge loss schedule
    p.add_argument("--edge_weight_after_epoch", type=int, default=120)
    p.add_argument("--edge_loss_weight_before", type=float, default=1.0)
    p.add_argument("--edge_loss_weight_after", type=float, default=1.3)

    # focal params for contour
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--focal_alpha", type=float, default=0.25)

    # optional regularization
    p.add_argument("--entropy_lambda", type=float, default=0.0,
                   help=">0 时启用：对 aux.u_f 做最大化（loss -= lam*u_f_mean）")

    # saving
    p.add_argument("--save_after", type=int, default=0)
    p.add_argument("--save_every", type=int, default=2)

    return p.parse_args()


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    writer = SummaryWriter(args.log_path)
    args.model_path = prepare_dir(args.model_path, "model")
    args.log_path = prepare_dir(args.log_path, "tb")

    log_file = os.path.join(args.model_path, "exp.txt")
    sys.stdout = Logger(log_file)
    print("Logging to:", log_file)
    print("Args:", args)

    train_set = LFDataset(location=args.train_data_location, image_size=args.image_size)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_worker,
        pin_memory=True,
        drop_last=True
    )

    model = DHPNet(
        backbone_type="swin",
        num_slices=args.num_slices,
        ffn_expansion=args.ffn_expansion,
        enable_parallel_when_reso_ge=args.enable_parallel_when_reso_ge,
        omega_init_bias=args.omega_init_bias,
        omega_scale_init=args.omega_max, learnable_omega_scale=True
    )

    model.apply(init_weights)
    model.load_pretrained(args.pretrained_model)
    model.to(device)

    optimizer = torch.optim.AdamW(build_adamw_param_groups(model, args.weight_decay), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    train(args, model, train_loader, device, optimizer, scheduler, writer)
