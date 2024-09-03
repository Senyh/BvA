import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
import time
import argparse
import copy
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from easydict import EasyDict
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset, random_split
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToPILImage
from exp_la.data.dataset import LADataset
from models import deeplabv3
from wheels.loss_functions import DSCLossH
from wheels.logger import logger as logging
from utils import ensure_dir, frequency_mixup
from wheels.mask_generator3d import BoxMaskGenerator, AddMaskParamsToBatch, SegCollate
from wheels.torch_utils import seed_torch
from wheels.model_init import init_weight
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args(known=False):
    parser = argparse.ArgumentParser(description='PyTorch Implementation')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--project', type=str, default=os.path.dirname(os.path.realpath(__file__)) + '/runs/BvA', help='project path for saving results')
    parser.add_argument('--backbone', type=str, default='VNet', choices=['VNet'], help='segmentation backbone')
    parser.add_argument('--ssl', type=bool, default=False, help='ssl or bsl')
    parser.add_argument('--data_path', type=str, default='YOUR_DATA_PATH', help='path to the data')
    parser.add_argument('--image_size', type=int, default=[80, 112, 112], help='the size of images for training and testing')
    parser.add_argument('--labeled_percentage', type=float, default=0.1, help='the percentage of labeled data')
    parser.add_argument('--num_epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='number of inputs per batch')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers to use for dataloader')
    parser.add_argument('--in_channels', type=int, default=1, help='input channels')
    parser.add_argument('--num_classes', type=int, default=2, help='number of target categories')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--cutmix_prob', type=float, default=0.5, help='probability for performing cut mix')
    parser.add_argument('--log_freq', type=float, default=1, help='logging frequency of metrics accord to the current iteration')
    parser.add_argument('--save_freq', type=float, default=10, help='saving frequency of model weights accord to the current epoch')
    args = parser.parse_known_args()[0] if known else parser.parse_args()
    return args


def get_data(args):
    val_set = LADataset(image_path=args.data_path, stage='val', image_size=args.image_size, is_augmentation=False)
    labeled_train_set = LADataset(image_path=args.data_path, stage='train', image_size=args.image_size, is_augmentation=True, labeled=True, percentage=args.labeled_percentage, ssl=args.ssl)
    plabeled_train_set = LADataset(image_path=args.data_path, stage='train', image_size=args.image_size, is_augmentation=True, labeled=True, percentage=args.labeled_percentage, nfc=True)
    unlabeled_train_set = LADataset(image_path=args.data_path, stage='train', image_size=args.image_size, is_augmentation=True, labeled=False, percentage=args.labeled_percentage, nfc=True)
    train_set = ConcatDataset([labeled_train_set, unlabeled_train_set])

    # repeat the labeled set to have a equal length with the unlabeled set (dataset)
    print('before: ', len(train_set), len(labeled_train_set), len(plabeled_train_set), len(val_set))
    labeled_ratio = len(train_set) // len(labeled_train_set)
    labeled_train_set = ConcatDataset([labeled_train_set for i in range(labeled_ratio)])
    labeled_train_set = ConcatDataset([labeled_train_set,
                                       Subset(labeled_train_set, range(len(train_set) - len(labeled_train_set)))])
    plabeled_ratio = len(train_set) // len(plabeled_train_set)
    plabeled_train_set = ConcatDataset([plabeled_train_set for i in range(plabeled_ratio)])
    plabeled_train_set = ConcatDataset([plabeled_train_set,
                                       Subset(plabeled_train_set, range(len(train_set) - len(plabeled_train_set)))])
    print('after: ', len(train_set), len(labeled_train_set), len(plabeled_train_set), len(val_set))
    assert len(labeled_train_set) == len(plabeled_train_set) == len(train_set)
    train_labeled_dataloder = DataLoader(dataset=labeled_train_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    train_plabeled_dataloder = DataLoader(dataset=plabeled_train_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    train_unlabeled_dataloder = DataLoader(dataset=train_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_dataloder = DataLoader(dataset=val_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    mask_generator = BoxMaskGenerator(prop_range=(0.25, 0.5),
                                        n_boxes=3,
                                        random_aspect_ratio=True,
                                        prop_by_area=True,
                                        within_bounds=True,
                                        invert=True)

    add_mask_params_to_batch = AddMaskParamsToBatch(mask_generator)
    mask_collate_fn = SegCollate(batch_aug_fn=add_mask_params_to_batch)
    aux_dataloder = DataLoader(dataset=train_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, pin_memory=True, collate_fn=mask_collate_fn)
    return train_labeled_dataloder, train_plabeled_dataloder, train_unlabeled_dataloder, val_dataloder, aux_dataloder


def main(is_debug=False):
    args = get_args()
    seed_torch(args.seed)
    # Project Saving Path
    if args.ssl:
        project_path = args.project + '_{}_label_{}_SSL/'.format(args.backbone, args.labeled_percentage)
    else:
        project_path = args.project + '_{}_label_{}/'.format(args.backbone, args.labeled_percentage)
    ensure_dir(project_path)
    save_path = project_path + 'weights/'
    ensure_dir(save_path)

    # Tensorboard & Statistics Results & Logger
    tb_dir = project_path + '/tensorboard{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
    writer = SummaryWriter(tb_dir)
    metrics = EasyDict()
    metrics.train_loss = []
    metrics.train_s_loss = []
    metrics.train_u_loss = []
    metrics.val_loss = []
    logger = logging(project_path + 'train_val.log')
    logger.info('PyTorch Version {}\n Experiment{}'.format(torch.__version__, project_path))

    # Load Data
    train_labeled_dataloader, train_plabeled_dataloader, train_unlabeled_dataloader, val_dataloader, aux_dataloader = get_data(args=args)
    iters = len(train_labeled_dataloader)
    val_iters = len(val_dataloader)

    # Load Model & EMA
    student = deeplabv3.__dict__[args.backbone](in_channels=args.in_channels, out_channels=args.num_classes).to(device)
    init_weight(student.net.classifier, nn.init.kaiming_normal_,
                nn.BatchNorm3d, 1e-5, 0.1,
                mode='fan_in', nonlinearity='relu')
    teacher = deeplabv3.__dict__[args.backbone](in_channels=args.in_channels, out_channels=args.num_classes).to(device)
    init_weight(teacher.net.classifier, nn.init.kaiming_normal_,
                nn.BatchNorm3d, 1e-5, 0.1,
                mode='fan_in', nonlinearity='relu')
    teacher.detach_model()
    best_model_wts = copy.deepcopy(student.state_dict())
    logger.info('#parameters: {}'.format(sum(param.numel() for param in student.parameters())))
    best_epoch = 0
    best_loss = 100
    conf_threshold = 0.95  # the confidence threshold for pixel-level pseudo label selection

    # Criterion & Optimizer & LR Schedule
    criterion = DSCLossH(num_classes=args.num_classes, device=device, is_3d=True)
    criterion_u = DSCLossH(num_classes=args.num_classes, device=device, is_3d=True)
    optimizer = optim.AdamW(student.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))

    # Train
    since = time.time()
    logger.info('start training')
    for epoch in range(1, args.num_epochs + 1):
        epoch_metrics = EasyDict()
        epoch_metrics.train_loss = []
        epoch_metrics.train_s_loss = []
        epoch_metrics.train_u_loss = []
        if is_debug:
            pbar = range(10)
        else:
            pbar = range(iters)
        iter_train_labeled_dataloader = iter(train_labeled_dataloader)
        iter_train_plabeled_dataloader = iter(train_plabeled_dataloader)
        iter_train_unlabeled_dataloader = iter(train_unlabeled_dataloader)
        iter_aux_loader = iter(aux_dataloader)

        ############################
        # Train
        ############################
        student.train()
        for idx in pbar:
            # label data
            image, label, imageA1, imageA2 = iter_train_labeled_dataloader.next()
            image, label = image.to(device), label.to(device)
            imageA1, imageA2 = imageA1.to(device), imageA2.to(device)
            if args.ssl:
                barely_mask = torch.ones_like(label.squeeze(1))
            else:
                barely_mask = label.squeeze(1) >= 0
                label[label < 0] = 0
            # plabel data (NFC synthesized data)
            pimage, plabel, pimageA1, pimageA2 = iter_train_plabeled_dataloader.next()
            pimage, plabel = pimage.to(device), plabel.to(device)
            pimageA1, pimageA2 = pimageA1.to(device), pimageA2.to(device)
            # unlabel data
            uimage, _, uimageA1, uimageA2 = next(iter_train_unlabeled_dataloader)
            uimage, uimageA1, uimageA2 = uimage.to(device), uimageA1.to(device), uimageA2.to(device)
            # auxiliary data
            aimage, _, aimageA1, aimageA2, amask = next(iter_aux_loader)
            aimage = aimage.to(device)
            aimageA1, aimageA2, amask = aimageA1.to(device), aimageA2.to(device), amask.to(device).long()
            if torch.rand(1) > args.cutmix_prob:
                amask[:] = 0

            # frequency mixup with a probability of 0.5
            if torch.rand(1) < 0.5:
                imageA1 = frequency_mixup(imageA1, aimageA1)
            if torch.rand(1) < 0.5:
                pimageA1 = frequency_mixup(pimageA1, aimageA1)
            if torch.rand(1) < 0.5:
                uimageA1 = frequency_mixup(uimageA1, aimageA1) 

            optimizer.zero_grad()

            # the supervised learning path #
            pred = student(torch.cat([imageA1, pimageA1]))
            pred_o_logits, pred_p_logits = pred['out'].chunk(2)

            # the unsupervised learning path #
            with torch.no_grad():
                # the forward pass of weak-augmented images
                pred_u = teacher(uimage)
                pred_u_logits = pred_u['out']
                pred_u_probs = torch.softmax(pred_u_logits, dim=1) # 8 4 256 256
                pred_u_pseudo = torch.argmax(pred_u_probs, dim=1) # 8 256 256
                pred_u_conf = pred_u_probs.max(dim=1)[0].clone()
                if torch.rand(1) < 0.5:
                    pred_a = teacher(aimage)
                    pred_a_logits = pred_a['out']
                    pred_a_probs = torch.softmax(pred_a_logits, dim=1)
                    pred_a_pseudo = torch.argmax(pred_a_probs, dim=1).clone()
                    pred_a_conf = pred_a_probs.max(dim=1)[0].clone()
                else:
                    pred_a = teacher(pimage)
                    pred_a_logits = pred_a['out']
                    pred_a_probs = torch.softmax(pred_a_logits, dim=1)
                    pred_a_pseudo = torch.argmax(pred_a_probs, dim=1).clone()
                    pred_a_conf = pred_a_probs.max(dim=1)[0].clone()
                # spatial mixup
                uimageA1_cutmixed, pred_u_pseudo_cutmixed, pred_u_conf_cutmixed = uimageA1.clone(), pred_u_pseudo.clone(), pred_u_conf.clone()
                uimageA1_cutmixed[amask.expand_as(uimageA1) == 1] = aimageA1[amask.expand_as(uimageA1) == 1]
                pred_u_pseudo_cutmixed[amask.squeeze(1) == 1] = pred_a_pseudo[amask.squeeze(1) == 1]
                pred_u_conf_cutmixed[amask.squeeze(1) == 1] = pred_a_conf[amask.squeeze(1) == 1]
                pred_u_conf_cutmixed = pred_u_conf_cutmixed.detach()

            # the forward pass of strong-augmented images
            pred_uA1_cutmixed = student(uimageA1_cutmixed)
            pred_uA1_cutmixed_logits = pred_uA1_cutmixed['out']

            # the supervised loss
            loss_s = (criterion(pred_o_logits, label.squeeze(1).long(), pixel_mask=barely_mask.float()) + criterion(pred_p_logits, plabel.squeeze(1).long())) / 2.
            # the unsupervised loss
            loss_u = criterion_u(pred_uA1_cutmixed_logits, pred_u_pseudo_cutmixed.detach(), pixel_mask=(pred_u_conf_cutmixed >= conf_threshold).float())

            loss = (loss_s + loss_u) / 2.
            
            loss.backward()
            optimizer.step()
            teacher.ema_update(student=student, ema_decay=0.99, cur_step=idx + len(pbar) * (epoch-1))

            include_rate = (pred_u_conf_cutmixed >= conf_threshold).sum() / pred_u_conf_cutmixed.numel()
            writer.add_scalar('train_loss_s', loss_s.item(), idx + len(pbar) * (epoch-1))
            writer.add_scalar('train_loss_u', loss_u.item(), idx + len(pbar) * (epoch-1))
            writer.add_scalar('train_loss', loss.item(), idx + len(pbar) * (epoch-1))
            writer.add_scalar('include_rate', include_rate, idx + len(pbar) * (epoch-1))
            if idx % args.log_freq == 0:
                logger.info("Train: Epoch/Epochs {}/{}, "
                            "iter/iters {}/{}, "
                            "loss {:.3f}, loss_s {:.3f}, loss_u {:.3f}, included_rate {:.3f}".format(epoch, args.num_epochs, idx, len(pbar),
                                                                                  loss.item(), loss_s.item(), loss_u.item(), include_rate))
            epoch_metrics.train_loss.append(loss.item())
            epoch_metrics.train_s_loss.append(loss_s.item())
            epoch_metrics.train_u_loss.append(loss_u.item())
        metrics.train_loss.append(np.mean(epoch_metrics.train_loss))
        metrics.train_s_loss.append(np.mean(epoch_metrics.train_s_loss))
        metrics.train_u_loss.append(np.mean(epoch_metrics.train_u_loss))

        ############################
        # Validation
        ############################
        epoch_metrics.val_loss = []
        iter_val_dataloader = iter(val_dataloader)
        if is_debug:
            val_pbar = range(10)
        else:
            val_pbar = range(val_iters)
        student.eval()
        with torch.no_grad():
            for idx in val_pbar:
                image, label = iter_val_dataloader.next()
                image, label = image.to(device), label.to(device)
                pred = student(image)['out']
                loss = criterion(pred, label.squeeze(1).long())
                writer.add_scalar('val_loss', loss.item(), idx + len(val_pbar) * (epoch-1))
                if idx % args.log_freq == 0:
                    logger.info("Val: Epoch/Epochs {}/{}\t"
                                "iter/iters {}/{}\t"
                                "loss {:.3f}".format(epoch, args.num_epochs, idx, len(val_pbar),
                                                     loss.item()))
                epoch_metrics.val_loss.append(loss.item())
        metrics.val_loss.append(np.mean(epoch_metrics.val_loss))

        # Save Model
        if np.mean(epoch_metrics.val_loss) <= best_loss:
            best_model_wts = copy.deepcopy(student.state_dict())
            best_epoch = epoch
            best_loss = np.mean(epoch_metrics.val_loss)
            torch.save(best_model_wts, save_path + 'best.pth'.format(best_epoch))
        torch.save(student.state_dict(), save_path + 'last.pth'.format(best_epoch))
        logger.info("Average: Epoch/Epoches {}/{}, "
                    "train epoch loss {:.3f}, "
                    "val epoch loss {:.3f}, "
                    "best loss {:.3f} at {}\n".format(epoch, args.num_epochs, np.mean(epoch_metrics.train_loss),
                                                     np.mean(epoch_metrics.val_loss), best_loss, best_epoch))
    ############################
    # Save Metrics
    ############################
    data_frame = pd.DataFrame(
        data={'loss': metrics.train_loss,
              'loss_s': metrics.train_s_loss,
              'loss_u': metrics.train_u_loss,
              'val_loss': metrics.val_loss},
        index=range(1, args.num_epochs + 1))
    data_frame.to_csv(project_path + 'train_val_loss.csv', index_label='Epoch')
    plt.figure()
    plt.title("Loss During Training and Validating")
    plt.plot(metrics.train_loss, label="Train")
    plt.plot(metrics.val_loss, label="Val")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(project_path + 'train_val_loss.png')

    print(project_path)
    time_elapsed = time.time() - since
    logger.info('Training completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logger.info('TRAINING FINISHED!')


if __name__ == '__main__':
    main()

