import argparse
import random
import shutil
import os
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import transforms
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from albumentations import RandomRotate90, Resize
import src.utils.losses as losses
from src.utils.util import AverageMeter
from src.utils.metrics import iou_score
from src.utils import ramps
from src.dataloader.dataset import (SemiDataSets, TwoStreamBatchSampler)
from src.network.DFCG import DFCG
from thop import profile
import csv
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF


def seed_torch(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--semi_percent', type=float, default=0.5)
parser.add_argument('--base_dir', type=str, default="./data/BUSI", help='dir')
parser.add_argument('--train_file_dir', type=str, default="BUSI_train.txt", help='dir')
parser.add_argument('--val_file_dir', type=str, default="BUSI_val.txt", help='dir')
parser.add_argument('--max_iterations', type=int,
                    default=40000, help='maximum epoch number to train')
parser.add_argument('--total_batch_size', type=int, default=6,
                    help='batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=41, help='random seed')

parser.add_argument('--labeled_bs', type=int, default=3,
                    help='labeled_batch_size per gpu')

parser.add_argument('--consistency', type=float,
                    default=7, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')

parser.add_argument('--kernel_size', type=int,
                    default=7, help='FCMxierBlock kernel size')
parser.add_argument('--length', type=tuple,
                    default=(3, 3, 3), help='length of FCMxierBlock')
args = parser.parse_args()

seed_torch(args.seed)


def getDataloader(args):
    train_transform = Compose([
        RandomRotate90(),
        transforms.Flip(),
        Resize(256, 256),
        transforms.Normalize(),
    ])
    val_transform = Compose([
        Resize(256, 256),

        transforms.Normalize(),
    ])
    labeled_slice = args.semi_percent
    db_train = SemiDataSets(base_dir=args.base_dir, split="train", transform=train_transform,
                            train_file_dir=args.train_file_dir, val_file_dir=args.val_file_dir,
                            )
    db_val = SemiDataSets(base_dir=args.base_dir, split="val", transform=val_transform,
                          train_file_dir=args.train_file_dir, val_file_dir=args.val_file_dir
                          )

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    total_slices = len(db_train)
    labeled_idxs = list(range(0, int(labeled_slice * total_slices)))
    unlabeled_idxs = list(range(int(labeled_slice * total_slices), total_slices))
    print("label num:{}, unlabel num:{} percent:{}".format(len(labeled_idxs), len(unlabeled_idxs), labeled_slice))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.total_batch_size, args.labeled_bs)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=8, pin_memory=False, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    return trainloader, valloader


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def getModel(args):
    print("FCMxierBlock1:{}, FCMxierBlock2:{}, FCMxierBlock3:{}, kernal:{}".format(args.length[0], args.length[1],
                                                                          args.length[2], args.kernel_size))

    model = DFCG(length=args.length, k=args.kernel_size).cuda()

    # Calculate FLOPs and Params
    input_size = (3, 256, 256)  # Replace with your actual input size
    input_data = torch.randn(1, *input_size).cuda()
    flops, params = profile(model, inputs=(input_data,))

    print(f"FLOPs: {flops}, Params: {params}")

    return model


def save_images(original, prediction, true_label, output_dir, index):
    os.makedirs(output_dir, exist_ok=True)

    original_path = os.path.join(output_dir, f"original_{index:04d}.png")
    prediction_path = os.path.join(output_dir, f"prediction_{index:04d}.png")
    true_label_path = os.path.join(output_dir, f"true_label_{index:04d}.png")

    # Convert tensors to PIL images
    original_pil = TF.to_pil_image(original)
    prediction_pil = TF.to_pil_image(prediction)
    true_label_pil = TF.to_pil_image(true_label)

    # Save the images
    original_pil.save(original_path)
    prediction_pil.save(prediction_path)
    true_label_pil.save(true_label_path)

def train(args):
    base_lr = args.base_lr
    max_iterations = int(args.max_iterations * args.semi_percent)
    trainloader, valloader = getDataloader(args)

    model = getModel(args)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    print("lr", base_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iterations)

    criterion = losses.__dict__['BCEDiceLoss']().cuda()

    print("{} iterations per epoch".format(len(trainloader)))
    best_iou = 0

    best_val_iou = 0
    best_epoch = 0

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    #    max_epoch = max_epoch + 100
    results_file = os.path.join('checkpoint', 'results.csv')
    with open(results_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Supervised Loss', 'Consistency Loss', 'Train IOU',
                         'Val Loss', 'Val IOU', 'Val DICE', 'Val SE', 'Val PC', 'Val F1', 'Val SP', 'Val ACC'])
    train_losses = []
    val_losses = []

    for epoch_num in range(max_epoch):
        avg_meters = {'total_loss': AverageMeter(),
                      'train_iou': AverageMeter(),
                      'consistency_loss': AverageMeter(),
                      'supervised_loss': AverageMeter(),
                      'val_loss': AverageMeter(),
                      'val_iou': AverageMeter(),
                      'val_dice': AverageMeter(),
                      'val_se': AverageMeter(),
                      'val_pc': AverageMeter(),
                      'val_f1': AverageMeter(),
                      'val_sp': AverageMeter(),
                      'val_acc': AverageMeter()
                      }
        model.train()
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs, outputs_aux1, outputs_aux2, outputs_aux3 = model(volume_batch)


            outputs_soft = torch.sigmoid(outputs)
            outputs_aux1_soft = torch.sigmoid(outputs_aux1)
            outputs_aux2_soft = torch.sigmoid(outputs_aux2)
            outputs_aux3_soft = torch.sigmoid(outputs_aux3)

            loss_ce = criterion(outputs[:args.labeled_bs],
                                label_batch[:args.labeled_bs][:])
            loss_ce_aux1 = criterion(outputs_aux1[:args.labeled_bs],
                                     label_batch[:args.labeled_bs][:])
            loss_ce_aux2 = criterion(outputs_aux2[:args.labeled_bs],
                                     label_batch[:args.labeled_bs][:])
            loss_ce_aux3 = criterion(outputs_aux3[:args.labeled_bs],
                                     label_batch[:args.labeled_bs][:])

            supervised_loss = (loss_ce + loss_ce_aux1 + loss_ce_aux2 + loss_ce_aux3) / 4


            consistency_weight = get_current_consistency_weight(iter_num // 150)
            consistency_loss_aux1 = torch.mean(
                (outputs_soft[args.labeled_bs:] - outputs_aux1_soft[args.labeled_bs:]) ** 2)
            consistency_loss_aux2 = torch.mean(
                (outputs_soft[args.labeled_bs:] - outputs_aux2_soft[args.labeled_bs:]) ** 2)
            consistency_loss_aux3 = torch.mean(
                (outputs_soft[args.labeled_bs:] - outputs_aux3_soft[args.labeled_bs:]) ** 2)

            consistency_loss = (consistency_loss_aux1 + consistency_loss_aux2 + consistency_loss_aux3) / 3

            loss = supervised_loss + consistency_weight * consistency_loss
            iou, dice, _, _, _, _, _ = iou_score(outputs[:args.labeled_bs], label_batch[:args.labeled_bs])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step()

            iter_num = iter_num + 1

            avg_meters['total_loss'].update(loss.item(), volume_batch[:args.labeled_bs].size(0))
            avg_meters['supervised_loss'].update(supervised_loss.item(), volume_batch[:args.labeled_bs].size(0))
            avg_meters['consistency_loss'].update(consistency_loss.item(), volume_batch[args.labeled_bs:].size(0))
            avg_meters['train_iou'].update(iou, volume_batch[:args.labeled_bs].size(0))

        model.eval()
        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(valloader):
                input, target = sampled_batch['image'], sampled_batch['label']
                input = input.cuda()
                target = target.cuda()
                output = model(input)
                loss = criterion(output, target)
                iou, dice, SE, PC, F1, SP, ACC = iou_score(output, target)
                avg_meters['val_loss'].update(loss.item(), input.size(0))
                avg_meters['val_iou'].update(iou, input.size(0))
                avg_meters['val_dice'].update(dice, input.size(0))
                avg_meters['val_se'].update(SE, input.size(0))
                avg_meters['val_pc'].update(PC, input.size(0))
                avg_meters['val_f1'].update(F1, input.size(0))
                avg_meters['val_sp'].update(SP, input.size(0))
                avg_meters['val_acc'].update(ACC, input.size(0))

                input_normalized = input.cpu().detach().numpy()[0][0]
                input_normalized = (input_normalized - np.min(input_normalized)) / (np.max(input_normalized) - np.min(input_normalized))
                ori = input_normalized * 255
                pre = output.cpu().detach().numpy()[0][0] * 255
                gt = target.cpu().detach().numpy()[0][0] * 255
                file_name1 = "original_{}.png".format(i_batch)
                file_name2 = "prediction_{}.jpg".format(i_batch)
                file_name3 = "true_label_{}.jpg".format(i_batch)
                folder_path = "./checkpoint/temp_mask"
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                file_path1 = os.path.join(folder_path, file_name1)
                file_path2 = os.path.join(folder_path, file_name2)
                file_path3 = os.path.join(folder_path, file_name3)
                cv2.imencode('.png', ori)[1].tofile(file_path1)
                cv2.imencode('.jpg', pre)[1].tofile(file_path2)
                cv2.imencode('.jpg', gt)[1].tofile(file_path3)

            # Here check if this epoch's average is better than the best avg iou score
            if avg_meters['val_iou'].avg > best_val_iou:
                best_val_iou = avg_meters['val_iou'].avg
                best_epoch = epoch_num

                # If the current score is better, delete the old directory and rename the new one
                mask_dir = "./checkpoint/mask"
                if os.path.exists(mask_dir):
                    shutil.rmtree(mask_dir)
                os.rename(folder_path, mask_dir)
            else:
                # Clear temp data if current is not best
                if os.path.exists(folder_path):
                    shutil.rmtree(folder_path)


        avg_train_loss = avg_meters['total_loss'].avg
        avg_val_loss = avg_meters['val_loss'].avg
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)


        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train and Validation Loss')
        plt.legend()


        checkpoint_dir = './checkpoint'
        os.makedirs(checkpoint_dir, exist_ok=True)
        loss_curve_path = os.path.join(checkpoint_dir, 'loss_curve.png')
        plt.savefig(loss_curve_path)
        plt.close()

        # print('epoch [%3d/%d]'% (epoch_num, max_epoch))

        print(
            'epoch [%3d/%d]  train_loss %.4f supervised_loss %.4f consistency_loss %.4f train_iou: %.4f '
            '- val_loss %.4f - val_iou %.4f - val_Dice %.4f - val_SE %.4f - val_PC %.4f - val_F1 %.4f - val_SP %.4f - val_ACC %.4f'
            % (epoch_num, max_epoch, avg_meters['total_loss'].avg,
               avg_meters['supervised_loss'].avg, avg_meters['consistency_loss'].avg, avg_meters['train_iou'].avg,
               avg_meters['val_loss'].avg, avg_meters['val_iou'].avg, avg_meters['val_dice'].avg,
               avg_meters['val_se'].avg, avg_meters['val_pc'].avg, avg_meters['val_f1'].avg,
               avg_meters['val_sp'].avg, avg_meters['val_acc'].avg))

        with open(results_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch_num, avg_meters['total_loss'].avg, avg_meters['supervised_loss'].avg,
                             avg_meters['consistency_loss'].avg, avg_meters['train_iou'].avg,
                             avg_meters['val_loss'].avg, avg_meters['val_iou'].avg, avg_meters['val_dice'].avg,
                             avg_meters['val_se'].avg, avg_meters['val_pc'].avg, avg_meters['val_f1'].avg,
                             avg_meters['val_sp'].avg, avg_meters['val_acc'].avg, ])

        if avg_meters['val_iou'].avg > best_iou:
            torch.save(model.state_dict(), 'checkpoint/model.pth')
            best_iou = avg_meters['val_iou'].avg
            print("=> saved best model")

    return "Training Finished! Results saved in: {}".format(results_file)


if __name__ == "__main__":
    train(args)
