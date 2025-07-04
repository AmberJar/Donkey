import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import time
import random
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from utils.metrics import evaluate_metrics,calculate_nme,calculate_iou,calculate_auc,calculate_precision_recall
from config import get_config
from models import RetinaFace
from layers import PriorBox, MultiBoxLoss

from utils.dataset import WiderFaceDetection
from utils.transform import Augmentation
from utils.general import draw_detections
from utils.box_utils import decode, decode_landmarks, nms
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='Training Arguments for RetinaFace')
    parser.add_argument(
        '--train-data',
        type=str,
        default='/scratch/pf2m24/data/Donkey_xiabao_face',
        help='Path to the training dataset directory.'
    )
    parser.add_argument(
        '--network',
        type=str,
        default='resnet50',
        choices=[
            'mobilenetv1', 'mobilenetv1_0.25', 'mobilenetv1_0.50',
            'mobilenetv2', 'resnet50', 'resnet34', 'resnet18'
        ],
        help='Backbone network architecture to use'
    )
    parser.add_argument('--num-workers', default=4, type=int, help='Number of workers to use for data loading.')

    # Traning arguments
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes in the dataset.')
    parser.add_argument('--batch-size', default=64, type=int, help='Number of samples in each batch during training.')
    parser.add_argument('--print-freq', type=int, default=10, help='Print frequency during training.')

    # Optimizer and scheduler arguments
    parser.add_argument('--learning-rate', default=1e-4, type=float, help='Initial learning rate.')
    parser.add_argument('--lr-warmup-epochs', type=int, default=10, help='Number of warmup epochs.')
    parser.add_argument('--power', type=float, default=0.9, help='Power for learning rate policy.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum factor in SGD optimizer.')
    parser.add_argument('--weight-decay', default=5e-4, type=float, help='Weight decay (L2 penalty) for the optimizer.')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD.')

    parser.add_argument(
        '--save-dir',
        default='./weights',
        type=str,
        help='Directory where trained model checkpoints will be saved.'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from checkpoint.ckpt from weights folder')

    args = parser.parse_args()

    return args


rgb_mean = (124, 135, 133)  # bgr order

def visualize_model_output_for_tensorboard(image_tensor, loc, conf, landmarks, priors, cfg, epoch, writer):
    device = image_tensor.device
    image_tensor = image_tensor.detach().cpu()
    loc = loc.detach().cpu()
    conf = conf.detach().cpu()
    landmarks = landmarks.detach().cpu()
    priors = priors.detach().cpu()

    # 随机选3张图像可视化
    batch_size = image_tensor.shape[0]
    indices = random.sample(range(batch_size), min(3, batch_size))

    for idx in indices:
        img = image_tensor[idx].cpu().numpy().transpose(1, 2, 0)
        img = np.clip(img + np.array(rgb_mean), 0, 255).astype(np.uint8).copy()
        # img = img[..., ::-1]  # BGR to RGB

        loc_i = loc[idx]
        conf_i = conf[idx]
        landms_i = landmarks[idx]

        boxes = decode(loc_i, priors, cfg['variance'])
        landms = decode_landmarks(landms_i, priors, cfg['variance'])
        scores = conf_i[:, 1]

        # 选出最大score对应的框和点
        best_idx = torch.argmax(scores)
        box = boxes[best_idx].numpy()
        landm = landms[best_idx].numpy().reshape(5, 2)

        h, w = img.shape[:2]
        box = (box * np.array([w, h, w, h])).astype(int)
        landm = (landm * np.array([w, h])).astype(int)


        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for (x, y) in landm:
            cv2.circle(img, (x, y), 3, (255, 0, 0), -1)

        # 写入 TensorBoard
        img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0) / 255.0
        writer.add_images("Output_Preview/Epoch_{:02d}_Idx_{:02d}".format(epoch, idx), img_tensor, epoch)


def random_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train_one_epoch(
    model,
    criterion,
    optimizer,
    data_loader,
    epoch,
    device,
    writer,
    priors,
    print_freq=10,
    scaler=None
) -> None:
    model.train()
    batch_loss = []

    for batch_idx, (images, targets) in enumerate(data_loader):
        start_time = time.time()
        images = images.to(device)
        targets = [target.to(device) for target in targets]

        with torch.amp.autocast("cuda", enabled=scaler is not None):
            outputs = model(images)



            loss_loc, loss_conf, loss_land,nme = criterion(outputs, targets)
            loss = cfg['loc_weight'] * loss_loc + loss_conf + loss_land

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        # Print training status
        if (batch_idx + 1) % print_freq == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch: {epoch + 1}/{cfg['epochs']} | Batch: {batch_idx + 1}/{len(data_loader)} | "
                f"Loss Localization : {loss_loc.item():.4f} | Classification: {loss_conf.item():.4f} | "
                f"Landmarks: {loss_land.item():.4f} | "
                f"LR: {lr:.8f} | Time: {(time.time() - start_time):.4f} s |",
                f"Mean NME : {nme :.4f}"
            )

        batch_loss.append(loss.item())
        if writer:
            global_step = epoch * len(data_loader) + batch_idx
            writer.add_scalar("Loss/total", loss.item(), global_step)
            writer.add_scalar("Loss/loc", loss_loc.item(), global_step)
            writer.add_scalar("Loss/conf", loss_conf.item(), global_step)
            writer.add_scalar("Loss/landmark", loss_land.item(), global_step)
            writer.add_scalar("Loss/nme", nme, global_step)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], global_step)

    print(f"Average batch loss: {np.mean(batch_loss):.7f}")

    #
    # with torch.no_grad():
    #     if epoch==0 or epoch % 3 ==0:
    #         val_batch = next(iter(data_loader))
    #         images, _ = val_batch
    #         images = images.to(device)
    #         loc, conf, landms = model(images)
    #         visualize_model_output_for_tensorboard(images, loc, conf, landms, priors, cfg,
    #                                              epoch, writer)
    #



def main(params):
    random_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create folder to save weights if not exists
    os.makedirs(params.save_dir, exist_ok=True)
    from datetime import datetime

    # 在 main 函数中设置 writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=os.path.join(params.save_dir, "logs", timestamp))
    # Prepare dataset and data loaders
    dataset = WiderFaceDetection(params.train_data, Augmentation(cfg['image_size'], rgb_mean))
    print('batch_size: ', params.batch_size)
    data_loader = DataLoader(
        dataset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=params.num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
        drop_last=True
    )
    print("Data successfully loaded!")

    # Generate prior boxes
    priorbox = PriorBox(cfg, image_size=(cfg['image_size'], cfg['image_size']))
    priors = priorbox.generate_anchors()
    priors = priors.to(device)

    # Multi Box Loss
    criterion = MultiBoxLoss(priors=priors, threshold=0.35, neg_pos_ratio=7, variance=cfg['variance'], device=device)

    # Initialize model
    print(cfg)
    model = RetinaFace(cfg=cfg)
    # 统计参数量（不影响模型位置）
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Model Params: {num_params / 1e6:.2f}M")

    model.to(device)

    # Optimizer
    parameters = model.parameters()
    # optimizer = torch.optim.SGD(
    #     parameters,
    #     lr=params.learning_rate,
    #     momentum=params.momentum,
    #     weight_decay=params.weight_decay
    # )

    optimizer = torch.optim.AdamW(
        parameters,
        lr=params.learning_rate,
        weight_decay=params.weight_decay,
        betas=(0.9, 0.99),
        eps=1e-8
    )

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['milestones'], gamma=params.gamma)

    start_epoch = 0
    if params.resume:
        try:
            checkpoint = torch.load(f"{params.save_dir}/{params.network}_checkpoint.ckpt", map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            start_epoch = checkpoint["epoch"] + 1
            print(f"Checkpoint successfully loaded from {params.save_dir}/{params.network}_checkpoint.ckpt")
        except Exception as e:
            print(f"Exception occurred while loading checkpoint, exception message: {e}")

    print("Training started!")
    for epoch in range(start_epoch, cfg['epochs']):
        train_one_epoch(
            model,
            criterion,
            optimizer,
            data_loader,
            epoch,
            device,
            writer,
            priors,
            params.print_freq,
            scaler=None
        )

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
        }

        lr_scheduler.step()

        # torch.save(ckpt, f'{params.save_dir}/{params.network}_{epoch}_checkpoint.ckpt')
        torch.save(model.state_dict(), f'{params.save_dir}/{params.network}_{epoch}.pth')

    # save final model
    state = model.state_dict()
    torch.save(state, f'{params.save_dir}/{params.network}_final.pth')
    writer.close()


if __name__ == '__main__':
    args = parse_args()
    cfg = get_config(args.network)
    main(args)
