import os
import sys
import time
import glob
import time
import torch
import random
import socket
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.utils.data
from common.utils import *
import torch.optim as optim
from common.camera import *
import common.loss as eval_loss
from common.arguments import parse_args
from common.load_data_hm36 import Fusion
from common.h36m_dataset import Human36mDataset

args = parse_args()

# Optional: select GPUs via command line, e.g. --gpu 0,1
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# Dynamically import the selected model
exec('from model.' + args.model + ' import Model')


class ActionHead(nn.Module):
    """
    Simple action recognition head on top of 3D poses.

    Input:  output_3D of shape [B, T, J, 3]
    Output: logits [B, num_actions]
    """
    def __init__(self, num_joints=17, num_actions=15, hidden_dim=256):
        super().__init__()
        self.num_joints = num_joints
        self.fc = nn.Sequential(
            nn.Linear(num_joints * 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, x):
        # x: [B, T, J, 3]
        B, T, J, C = x.shape
        x = x.view(B, T, J * C)   # [B, T, J*3]
        x = x.mean(dim=1)         # temporal average → [B, J*3]
        logits = self.fc(x)       # [B, num_actions]
        return logits


def normalize_action_name(a: str) -> str:
    """
    Human3.6M often has names like 'Photo 2', 'Directions 1', 'Walking 1'.
    But define_actions() typically uses base names like 'Photo', 'Directions', 'Walking'.

    This function maps:
        'Photo 2'       -> 'Photo'
        'SittingDown 1' -> 'SittingDown'
        'Walking'       -> 'Walking'  (no change)
    """
    a = str(a)
    base = a.split(' ')[0]
    return base


def train(dataloader, model, action_head, optimizer, epoch,
          action_to_idx, criterion_action):
    """
    Train only the action head (backbone is frozen and used as feature extractor).
    Works with single GPU or DataParallel-wrapped backbone/head.
    """
    model.eval()   # backbone frozen; keep it in eval mode
    action_head.train()

    loss_all = {'loss': AccumLoss()}

    for i, data in enumerate(tqdm(dataloader)):
        batch_cam, gt_3D, input_2D, input_2D_GT, action, subject, cam_ind = data
        input_2D, input_2D_GT, gt_3D, batch_cam = (
            input_2D.cuda(),
            input_2D_GT.cuda(),
            gt_3D.cuda(),
            batch_cam.cuda(),
        )

        # ----- Backbone forward (no grad, frozen) -----
        with torch.no_grad():
            output_3D = model(input_2D)   # [B, T, J, 3]

        # ----- Action classification loss -----
        # 'action' is a list/tuple of action names, one per sample in the batch
        label_indices = []
        for a in action:
            base = normalize_action_name(a)  # e.g. 'Photo 2' -> 'Photo'
            idx = action_to_idx.get(base, 0)  # fallback to 0 if unseen
            label_indices.append(idx)

        labels = torch.tensor(
            label_indices,
            device=output_3D.device,
            dtype=torch.long,
        )  # [B]

        logits = action_head(output_3D)   # [B, num_actions]
        loss = criterion_action(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        N = input_2D.shape[0]
        loss_all['loss'].update(loss.detach().cpu().numpy() * N, N)

    return loss_all['loss'].avg


def test(actions, dataloader, model, action_head, action_to_idx):
    """
    Test 3D pose (P1/P2) using the backbone, and action accuracy using the action head.
    Works with single GPU or DataParallel-wrapped models.
    """
    model.eval()
    action_head.eval()

    action_error = define_error_list(actions)

    joints_left = [4, 5, 6, 11, 12, 13]
    joints_right = [1, 2, 3, 14, 15, 16]

    action_correct = 0
    action_total = 0

    for i, data in enumerate(tqdm(dataloader, dynamic_ncols=True)):
        batch_cam, gt_3D, input_2D, input_2D_GT, action, subject, cam_ind = data
        input_2D, input_2D_GT, gt_3D, batch_cam = (
            input_2D.cuda(),
            input_2D_GT.cuda(),
            gt_3D.cuda(),
            batch_cam.cuda(),
        )

        # ----- 3D pose evaluation (original HoT logic) -----
        # input_2D is [B, 2, T, J, 2] (non-flip, flip)
        output_3D_non_flip = model(input_2D[:, 0])
        output_3D_flip = model(input_2D[:, 1])

        output_3D_flip[:, :, :, 0] *= -1
        output_3D_flip[:, :, joints_left + joints_right, :] = \
            output_3D_flip[:, :, joints_right + joints_left, :]

        output_3D = (output_3D_non_flip + output_3D_flip) / 2

        out_target = gt_3D.clone()
        if args.stride == 1:
            out_target = out_target[:, args.pad].unsqueeze(1)
            output_3D = output_3D[:, args.pad].unsqueeze(1)

        output_3D[:, :, args.root_joint] = 0
        out_target[:, :, args.root_joint] = 0

        action_error = test_calculation(
            output_3D, out_target, action, action_error, args.dataset, subject
        )

        # ----- Action accuracy (use non-flip output_3D_non_flip as features) -----
        with torch.no_grad():
            label_indices = []
            for a in action:
                base = normalize_action_name(a)
                idx = action_to_idx.get(base, 0)
                label_indices.append(idx)

            labels = torch.tensor(
                label_indices,
                device=output_3D_non_flip.device,
                dtype=torch.long,
            )  # [B]

            logits = action_head(output_3D_non_flip)  # [B, num_actions]
            preds = torch.argmax(logits, dim=1)
            correct = (preds == labels).sum().item()
            total = labels.numel()

            action_correct += correct
            action_total += total

    p1, p2 = print_error(args.dataset, action_error, 1)

    if action_total > 0:
        action_acc = 100.0 * action_correct / action_total
    else:
        action_acc = 0.0

    return p1, p2, action_acc


if __name__ == '__main__':
    seed = 1126

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    dataset_path = args.root_path + 'data_3d_' + args.dataset + '.npz'
    dataset = Human36mDataset(dataset_path, args)
    actions = define_actions(args.actions)

    # Map action name → index for classification.
    # actions here are the base names, e.g. ['Directions', 'Discussion', 'Eating', ...]
    action_to_idx = {name: i for i, name in enumerate(actions)}
    num_actions = len(actions)

    if args.train:
        train_data = Fusion(args, dataset, args.root_path, train=True)
        train_dataloader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=int(args.workers),
            pin_memory=True,
        )

    test_data = Fusion(args, dataset, args.root_path, train=False)
    test_dataloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.workers),
        pin_memory=True,
    )

    # ----- 3D backbone (HoT / MixSTE) -----
    # Build on CPU first so we can wrap with DataParallel later cleanly
    model = Model(args)

    # Load pretrained weights if provided (same as original code)
    if args.previous_dir != '':
        Load_model(args, model)

    # Freeze backbone: use pretrained HoT as fixed feature extractor
    for p in model.parameters():
        p.requires_grad = False

    # Wrap backbone with DataParallel if multiple GPUs are visible
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs for backbone with DataParallel")
        model = nn.DataParallel(model)

    # Move (possibly wrapped) backbone to GPU
    model = model.cuda()

    # ----- Action head (trainable) -----
    # Human3.6M uses 17 joints; change if you use a different skeleton
    action_head = ActionHead(num_joints=17, num_actions=num_actions)

    # Wrap action head with DataParallel too (optional but uses both GPUs for the head)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs for action head with DataParallel")
        action_head = nn.DataParallel(action_head)

    action_head = action_head.cuda()

    lr = args.lr
    optimizer = optim.AdamW(action_head.parameters(), lr=lr, weight_decay=0.1)
    criterion_action = nn.CrossEntropyLoss().cuda()

    best_epoch = 0
    loss_epochs = []
    mpjpes = []

    for epoch in range(1, args.nepoch + 1):
        if args.train:
            loss = train(
                train_dataloader,
                model,
                action_head,
                optimizer,
                epoch,
                action_to_idx,
                criterion_action,
            )
            loss_epochs.append(loss * 1000)

        with torch.no_grad():
            p1, p2, action_acc = test(
                actions,
                test_dataloader,
                model,
                action_head,
                action_to_idx,
            )
            mpjpes.append(p1)

        # NOTE: p1 (pose error) will stay basically constant, since backbone is frozen.
        # You still get action training via 'loss' above and can track action_acc.

        if args.train and p1 < args.previous_best:
            best_epoch = epoch
            args.previous_name = save_model(args, epoch, p1, model, 'model')
            args.previous_best = p1

        if args.train:
            logging.info(
                'epoch: %d, lr: %.6f, l: %.4f, p1: %.2f, p2: %.2f, acc: %.2f, %d: %.2f'
                % (epoch, lr, loss, p1, p2, action_acc, best_epoch, args.previous_best)
            )
            print(
                '%d, lr: %.6f, l: %.4f, p1: %.2f, p2: %.2f, acc: %.2f, %d: %.2f'
                % (epoch, lr, loss, p1, p2, action_acc, best_epoch, args.previous_best)
            )

            if epoch % args.lr_decay_epoch == 0:
                lr *= args.lr_decay_large
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= args.lr_decay_large
            else:
                lr *= args.lr_decay
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= args.lr_decay
        else:
            print('p1: %.2f, p2: %.2f, action_acc: %.2f' % (p1, p2, action_acc))
            break
