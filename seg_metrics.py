from utils import AverageMeter
import torch
import torchvision
from pytorch_toolbelt import losses
import numpy as np  
from sklearn.metrics import roc_auc_score, confusion_matrix
import wandb
import gc

def get_avg_dice(predictions:torch.tensor, targets:torch.tensor): 
    if len(predictions) == 0:
          return 0, (0, 0), (0, 0)

    """Dice coeff for batches"""
    tot_dice = 0.0
    mx_dice, worst_dice = -1, 1e9
    best_idx, worst_idx = 0, 0

    for i, (pred, gt) in enumerate(zip(predictions, targets)):
        assert pred.shape == gt.shape
        assert pred.requires_grad == False
        
        d = losses.functional.soft_dice_score(pred, gt, smooth=1e-7).item()
        tot_dice += d
        if (d >= 0 and d <= 1.0) and d > mx_dice:
            mx_dice = d
            best_idx = i
        if (d >= 0 and d <= 1.0) and d < worst_dice:
            worst_dice = d
            worst_idx = i 
    
    return tot_dice / (predictions.shape[0]), (mx_dice, best_idx), (worst_dice, worst_idx)


def get_avg_jaccard(predictions:torch.tensor, targets:torch.tensor): 
    if len(predictions) == 0:
        return 0, (0, 0), (0, 0)

    """Jaccard coeff for batches"""
    tot_iou = 0.0
    mx_iou, worst_iou = -1, 1e9
    best_idx, worst_idx = 0, 0

    for i, (pred, gt) in enumerate(zip(predictions, targets)):
        assert pred.shape == gt.shape
        assert pred.requires_grad == False
        
        d = losses.functional.soft_jaccard_score(pred, gt, smooth=1e-7).item()
        tot_iou += d
        if (d >= 0 and d <= 1.0) and d > mx_iou:
            mx_iou = d
            best_idx = i 
        if (d >= 0 and d <= 1.0) and d < worst_iou:
            worst_iou = d
            worst_idx = i 
    
    return tot_iou / (predictions.shape[0]), (mx_iou, best_idx), (worst_iou, worst_idx)


def get_pixel_auc(predictions:torch.tensor, targets:torch.tensor, paths):
    if len(predictions) == 0:
          return 0, 0
          
    sum, i = 0, 0
    for pred, truth, path in zip(predictions, targets, paths):
        try:
            roc = roc_auc_score(truth.numpy().ravel(), pred.numpy().ravel())
            sum += roc
            i += 1
        except ValueError as e:
            print(e)
            print(path)
        
    return (sum / i), i


def get_fpr(predictions:torch.tensor, targets:torch.tensor, thr=0.5):
    if len(predictions) == 0:
        return 0, (0, 0), (0, 0)

    sum, i = 0, 0
    min_fpr, max_fpr = 1e9, -1e12
    best_idx, worst_idx = 0, 0

    for truth, pred in zip(targets, predictions):
        truth = (truth.numpy().ravel() >= thr).astype('uint8')
        pred = (pred.numpy().ravel() >= thr).astype('uint8')

        if np.count_nonzero(pred) == 0:
            min_fpr, best_idx = 0, i
            i += 1
            continue

        tn, fp, fn, tp = confusion_matrix(truth, pred).ravel()
        fpr = fp/(fp+tn)

        if fpr < min_fpr: 
          min_fpr = fpr
          best_idx = i
        if fpr > max_fpr: 
          max_fpr = fpr
          worst_idx = i

        sum += fpr
        i += 1
    
    return (sum/i), (min_fpr, best_idx), (max_fpr, worst_idx)


def sensitivity(predictions, targets):
    true_positives = torch.sum(torch.round(torch.clamp(targets * predictions, 0, 1)))
    possible_positives = torch.sum(torch.round(torch.clamp(targets, 0, 1)))
    return true_positives / (possible_positives + 1e-7)

def specificity(predictions, targets):
    true_negatives = torch.sum(torch.round(torch.clamp((1 - targets) * (1 - predictions), 0, 1)))
    possible_negatives = torch.sum(torch.round(torch.clamp(1 - targets, 0, 1)))
    return true_negatives / (possible_negatives + 1e-7)



#region DICE TEST
####################################################################################
def predict(X, threshold):
    X_p = np.copy(X)
    preds = (X_p > threshold).astype('uint8')
    return preds

def metric(probability, truth, threshold=0.5, reduction='none'):
    '''Calculates dice of positive and negative images seperately'''
    '''probability and truth must be torch tensors'''
    batch_size = len(truth)
    with torch.no_grad():
        probability = probability.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert(probability.shape == truth.shape)
        
        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p*t).sum(-1)/((p+t).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])
        
        # dice_neg = np.nan_to_num(dice_neg.mean().item(), 0)
        # dice_pos = np.nan_to_num(dice_pos.mean().item(), 0)
        # dice = dice.mean().item()

        num_neg = len(neg_index)
        num_pos = len(pos_index)
    return dice, dice_neg, dice_pos, num_neg, num_pos

def compute_ious(pred, label, classes, ignore_index=255, only_present=True):
    '''computes iou for one ground truth mask and predicted mask'''
    pred[label == ignore_index] = 0
    ious = []
    for c in classes:
        label_c = label == c
        if only_present and np.sum(label_c) == 0:
            ious.append(np.nan)
            continue
        pred_c = pred == c
        intersection = np.logical_and(pred_c, label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()
        if union != 0:
            ious.append(intersection / union)
    return ious if ious else [1]


def compute_iou_batch(predictions, labels, classes=None):
    '''computes mean iou for a batch of ground truth masks and predicted masks'''
    ious = []
    preds = np.copy(predictions) # copy is imp
    labels = np.array(labels) # tensor to np
    for pred, label in zip(preds, labels):
        ious.append(np.nanmean(compute_ious(pred, label, classes)))
    iou = np.nanmean(ious)
    return iou

class SegMeter:
    '''A meter to keep track of iou and dice scores throughout an epoch'''
    def __init__(self):
        self.base_threshold = 0.5 # <<<<<<<<<<< here's the threshold
        self.base_dice_scores = []
        self.dice_neg_scores = []
        self.dice_pos_scores = []
        self.iou_scores = []

    def update(self, predictions, targets):
        # probs = torch.sigmoid(predictions)
        probs = predictions
        dice, dice_neg, dice_pos, _, _ = metric(probs, targets, self.base_threshold)
        self.base_dice_scores.extend(dice)
        self.dice_pos_scores.extend(dice_pos)
        self.dice_neg_scores.extend(dice_neg)
        preds = predict(probs, self.base_threshold)
        iou = compute_iou_batch(preds, targets, classes=[1])
        self.iou_scores.append(iou)

    def get_metrics(self):
        dice = np.nanmean(self.base_dice_scores)
        dice_neg = np.nanmean(self.dice_neg_scores)
        dice_pos = np.nanmean(self.dice_pos_scores)
        dices = [dice, dice_neg, dice_pos]
        iou = np.nanmean(self.iou_scores)
        return dices, iou
    
def epoch_score_log(phase, meter):
    '''logging the metrics at the end of an epoch'''
    dices, iou = meter.get_metrics()
    dice, dice_neg, dice_pos = dices
    print("%s -> dice: %0.4f | dice_neg: %0.4f | dice_pos: %0.4f | IoU: %0.4f" % (phase, dice, dice_neg, dice_pos, iou))
    return dice, dice_neg, dice_pos, iou
#######################################################################################
#endregion


class MetricMeter:
    '''A meter to keep track of scores throughout an epoch'''
    def __init__(self, mode):
        self.mode = mode
        self.fake_dice = AverageMeter()
        self.fake_jaccard = AverageMeter()
        self.fake_pixel_auc = AverageMeter()
        self.fake_fpr = AverageMeter()
        self.real_fpr = AverageMeter()
        self.total_fpr = AverageMeter()

        self.example_images = []

    def update(self, predictions, targets, batch=None):
        target_labels = batch["label"].cpu().detach()
        images = batch["image"].cpu().detach()
        paths = batch["image_path"]

        targets = (targets >= 0.5).to(dtype=torch.uint8)

        real_indices = torch.where(target_labels < 0.5)[0]
        real_images = images[real_indices]
        real_pred_mask = predictions[real_indices]
        real_gt = targets[real_indices]
        real_image_paths = [paths[i] for i in real_indices]

        fake_indices = torch.where(target_labels > 0.5)[0]
        fake_images = images[fake_indices]
        fake_pred_mask = predictions[fake_indices]
        fake_gt = targets[fake_indices]
        fake_image_paths = [paths[i] for i in fake_indices]

        batch_fake_dice, (mx_fake_dice, best_fake_dice_idx), (worst_fake_dice, worst_fake_dice_idx) = get_avg_dice(fake_pred_mask, fake_gt)
        self.fake_dice.update(batch_fake_dice, fake_pred_mask.shape[0])

        batch_fake_jaccard, _, _ = get_avg_jaccard(fake_pred_mask, fake_gt)
        self.fake_jaccard.update(batch_fake_jaccard, fake_pred_mask.shape[0])

        batch_fake_fpr, _, _ = get_fpr(fake_pred_mask, fake_gt)
        self.fake_fpr.update(batch_fake_fpr, fake_pred_mask.shape[0])

        batch_fake_pixel_auc, num = get_pixel_auc(fake_pred_mask, fake_gt, fake_image_paths)
        self.fake_pixel_auc.update(batch_fake_pixel_auc, num)

        batch_real_fpr, (min_fpr, best_real_fpr_idx), (max_fpr, worst_real_fpr_idx) = get_fpr(real_pred_mask, real_gt)
        self.real_fpr.update(batch_real_fpr, real_pred_mask.shape[0])

        batch_total_fpr, _, _ = get_fpr(predictions, targets)
        self.total_fpr.update(batch_total_fpr, predictions.shape[0])

        _, (mx_real_dice, best_real_dice_idx), (worst_real_dice, worst_real_dice_idx) = get_avg_dice(real_pred_mask, real_gt)
        _, (mx_avg_dice, best_avg_dice_idx), (worst_avg_dice, worst_avg_dice_idx) = get_avg_dice(predictions, targets)

        if self.mode != "TRAIN":
            if(np.random.rand() < 0.5 and len(self.example_images) < 50):
                self.example_images.append(
                    wandb.Image(
                        get_image_plot(fake_images[best_fake_dice_idx],
                            fake_pred_mask[best_fake_dice_idx], 
                            fake_gt[best_fake_dice_idx],
                            f"BestFakeDice : {mx_fake_dice}"
                        ),
                        caption= fake_image_paths[best_fake_dice_idx].split('/', 2)[-1]
                    )
                )
                self.example_images.append(
                    wandb.Image(   
                        get_image_plot(fake_images[worst_fake_dice_idx],
                            fake_pred_mask[worst_fake_dice_idx],
                            fake_gt[worst_fake_dice_idx],
                            f"WorstFakeDice : {worst_fake_dice}"
                        ),
                        caption= fake_image_paths[worst_fake_dice_idx].split('/', 2)[-1]
                    )
                )
                # self.example_images.append(
                #     wandb.Image(
                #         get_image_plot(real_images[best_real_fpr_idx],
                #             real_pred_mask[best_real_fpr_idx], 
                #             real_gt[best_real_fpr_idx],
                #             f"BestRealFPR : {min_fpr}"
                #         ),
                #         caption= real_image_paths[best_real_fpr_idx].split('/', 2)[-1]
                #     )
                # )
                if max_fpr > 0:
                    self.example_images.append(
                        wandb.Image(
                            get_image_plot(real_images[worst_real_fpr_idx],
                                real_pred_mask[worst_real_fpr_idx], 
                                real_gt[worst_real_fpr_idx],
                                f"WorstRealFPR : {max_fpr}"
                            ),
                            caption= real_image_paths[worst_real_fpr_idx].split('/', 2)[-1]
                        )
                    )
                self.example_images.append(
                        wandb.Image(
                            get_image_plot(real_images[best_real_dice_idx],
                                real_pred_mask[best_real_dice_idx], 
                                real_gt[best_real_dice_idx],
                                f"BestRealDice : {mx_real_dice}"
                            ),
                            caption= real_image_paths[best_real_dice_idx].split('/', 2)[-1]
                        )
                    )
                # self.example_images.append(
                #         wandb.Image(
                #             get_image_plot(real_images[worst_real_dice_idx],
                #                 real_pred_mask[worst_real_dice_idx], 
                #                 real_gt[worst_real_dice_idx],
                #                 f"WorstRealDice : {worst_real_dice}"
                #             ),
                #             caption= real_image_paths[worst_real_dice_idx].split('/', 2)[-1]
                #         )
                #     )
                # self.example_images.append(
                #         wandb.Image(
                #             get_image_plot(images[worst_avg_dice_idx],
                #                 predictions[worst_avg_dice_idx], 
                #                 targets[worst_avg_dice_idx],
                #                 f"WorstAvgDice : {worst_avg_dice}"
                #             ),
                #             caption= paths[worst_avg_dice_idx].split('/', 2)[-1]
                #         )
                #     )


from matplotlib import gridspec
import matplotlib.pyplot as plt
import io
import PIL

def get_image_plot(image, pred, gt, title):
    fig = plt.figure(figsize=(10,5))
    gs = gridspec.GridSpec(1,3)
    gs.update(wspace=0., hspace=0.)

    ax1 = plt.subplot(gs[0,0])
    ax1.imshow(tensor2image(image))
    ax1.axis('off')
    ax1.set_aspect('equal')
    ax2 = plt.subplot(gs[0,1])
    ax2.imshow(pred.squeeze(), cmap='gray')
    ax2.axis('off')
    ax2.set_aspect('equal')
    ax3 = plt.subplot(gs[0,2])
    ax3.imshow(gt.squeeze(), cmap='gray')
    ax3.axis('off')
    ax3.set_aspect('equal')
    fig.tight_layout()
    fig.suptitle(title, fontsize=18)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches = 'tight', pad_inches = 0)
    buf.seek(0)
    im = PIL.Image.open(buf)
    
    del buf
    gc.collect()

    return im

def tensor2image(tensor):
    image = torchvision.utils.make_grid(tensor, normalize=True)
    image = PIL.Image.fromarray(image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy())
    return image