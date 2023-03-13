import numpy as np
import torch
import torch_optimizer
from torch import optim
import time   

import albumentations
from albumentations import augmentations as A
import imgaug.augmenters as iaa
import albumentations.pytorch


class EarlyStopping:
    """
    EarlyStopping taken from avishekthakur/wtfml
    """

    def __init__(self, patience=7, mode="max", delta=0.0001, tpu=False):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.tpu = tpu
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path):
        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                "EarlyStopping counter: {} out of {}".format(
                    self.counter, self.patience
                )
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print(
                "Validation score improved ({} --> {}). Saving model!".format(
                    self.val_score, epoch_score
                )
            )

            torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score


class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_train_transforms():
    train_imgaug  = iaa.Sequential(
        [
            iaa.SomeOf((0, 2),
                [   
                    iaa.OneOf([
                        iaa.JpegCompression(compression=(10, 40)),
                        iaa.GaussianBlur((0, 1.75)), # blur images with a sigma between 0 and 3.0
                        iaa.AverageBlur(k=(1, 3)), # blur image using local means with kernel sizes between 2 and 7
                        iaa.MedianBlur(k=(1, 3)), # blur image using local medians with kernel sizes between 2 and 7
                    ]),
                    # iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                    # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                    # # iaa.Sometimes(0.3, iaa.Invert(0.05, per_channel=True)), # invert color channels
                    # # iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                    # iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                    iaa.LinearContrast((0.5, 1.5)), # improve or worsen the contrast
                    # # # either change the brightness of the whole image (sometimes
                    # # # per channel) or change the brightness of subareas
                    iaa.Sometimes(0.4,
                        iaa.OneOf([
                            iaa.Multiply((0.5, 1.5)),
                            iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-20,20)),
                            iaa.MultiplyHueAndSaturation(),
                            # iaa.BlendAlphaFrequencyNoise(
                            #     exponent=(-4, 0),
                            #     foreground=iaa.Multiply((0.5, 1.5), per_channel=True),
                            #     background=iaa.LinearContrast((0.5, 2.0))
                            # )
                        ])
                    ),
                ], random_order=True
            )
        ], random_order=True
    )
    train_geo_aug = albumentations.Compose(
        [
            A.transforms.HorizontalFlip(p=0.5),
            A.transforms.VerticalFlip(p=0.5),
            albumentations.RandomRotate90(p=0.5),
            # A.geometric.transforms.Perspective(p=0.3),
            # albumentations.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=35, p=0.25),
            # albumentations.OneOf([
            #     albumentations.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            #     albumentations.GridDistortion(p=0.5),
            #     albumentations.OpticalDistortion(p=0.5, distort_limit=2, shift_limit=0.5)                  
            # ], p=0.7),
        ],
        additional_targets={'ela':'image'}
    )
    return train_imgaug, train_geo_aug

def get_transforms_normalize():
    normalize = {
        "mean": [0.4535408213875562, 0.42862278450748387, 0.41780105499276865],
        "std": [0.2672804038612597, 0.2550410416463668, 0.29475415579144293],
    }

    transforms_normalize = albumentations.Compose(
        [
            albumentations.Normalize(mean=normalize['mean'], std=normalize['std'], always_apply=True, p=1),
            albumentations.pytorch.transforms.ToTensorV2()
        ],
        additional_targets={'ela':'image'}
    )
    return transforms_normalize


def get_optimizer(model, optimizer, learning_rate, weight_decay):
    if optimizer == "radam":
        optimizer = torch_optimizer.RAdam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer == "adamP":
        optimizer = torch_optimizer.AdamP(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer == "qhadam":
        optimizer = torch_optimizer.QHAdam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    return optimizer


def image2np(image: torch.Tensor) -> np.ndarray:
    "Convert from torch style `image` to numpy/matplotlib style."
    res = image.cpu().permute(1, 2, 0).numpy()
    return res[..., 0] if res.shape[2] == 1 else res                                            


import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

def get_dataframe(csv, folds=None, frac=1.0):
    df = pd.read_csv(csv, keep_default_na=False).sample(frac=frac, random_state=123).reset_index(drop=True)
    
    if folds is not None:
        df['fold'] = -1
        
        if folds > 0:
            y = df.label.values
            kf = StratifiedKFold(n_splits=folds)

            for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
                df.loc[v_, 'fold'] = f
        
    return df

def stratified_train_val_test_split(df_input, stratify_colname='y',
                                    frac_train=0.8, frac_val=0.2, frac_test=0.2,
                                    random_state=123):
    '''
    Splits a Pandas dataframe into three subsets (train, val, and test)
    following fractional ratios provided by the user, where each subset is
    stratified by the values in a specific column (that is, each subset has
    the same relative frequency of the values in the column). It performs this
    splitting by running train_test_split() twice.

    Parameters
    ----------
    df_input : Pandas dataframe
        Input dataframe to be split.
    stratify_colname : str
        The name of the column that will be used for stratification. Usually
        this column would be for the label.
    frac_train : float
    frac_val   : float
    frac_test  : float
        The ratios with which the dataframe will be split into train, val, and
        test data. The values should be expressed as float fractions and should
        sum to 1.0.
    random_state : int, None, or RandomStateInstance
        Value to be passed to train_test_split().

    Returns
    -------
    df_train, df_val, df_test :
        Dataframes containing the three splits.
    '''
    if isinstance(df_input, str):
        df_input = pd.read_csv(df_input)
        
    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError('fractions %f, %f, %f do not add up to 1.0' % \
                         (frac_train, frac_val, frac_test))

    if stratify_colname not in df_input.columns:
        raise ValueError('%s is not a column in the dataframe' % (stratify_colname))

    X = df_input # Contains all columns.
    y = df_input[[stratify_colname]] # Dataframe of just the column on which to stratify.

    # Split original dataframe into train and temp dataframes.
    df_train, df_temp, y_train, y_temp = train_test_split(X,
                                                          y,
                                                          stratify=y,
                                                          test_size=(1.0 - frac_train),
                                                          random_state=random_state)

    # Split the temp dataframe into val and test dataframes.
    relative_frac_test = frac_test / (frac_val + frac_test)
    df_val, df_test, y_val, y_test = train_test_split(df_temp,
                                                      y_temp,
                                                      stratify=y_temp,
                                                      test_size=relative_frac_test,
                                                      random_state=random_state)

    assert len(df_input) == len(df_train) + len(df_val) + len(df_test)
    
    return df_train, df_val, df_test
    

def measure_time(f):
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        print('%r, %2.2f sec' % (f.__name__, te-ts))
        return result
    return timed


from io import BytesIO
from PIL import Image, ImageChops

# @measure_time
def get_ela(image, scale):
    
    pil_ori_image = Image.fromarray(image)
    output = BytesIO()
    pil_ori_image.save(output, format="JPEG", quality=90)
    pil_compressed_image = Image.open(output)
    diff = ImageChops.difference(pil_ori_image, pil_compressed_image)
    d = diff.load()
    WIDTH, HEIGHT = diff.size
    for x in range(WIDTH):
        for y in range(HEIGHT):
            d[x, y] = tuple(k * scale for k in d[x, y])

    ela = np.array(diff).astype('uint8')

    return ela


def topk_accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> "list[torch.FloatTensor]":
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.

    ref:
    - https://pytorch.org/docs/stable/generated/torch.topk.html
    - https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
    - https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    - https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
    - https://stackoverflow.com/questions/59474987/how-to-get-top-k-accuracy-in-semantic-segmentation-using-pytorch

    :param output: output is the prediction of the model e.g. scores, logits, raw y_pred before normalization or getting classes
    :param target: target is the truth
    :param topk: tuple of topk's to compute e.g. (1, 2, 5) computes top 1, top 2 and top 5.
    e.g. in top 2 it means you get a +1 if your models's top 2 predictions are in the right label.
    So if your model predicts cat, dog (0, 1) and the true label was bird (3) you get zero
    but if it were either cat or dog you'd accumulate +1 for that example.
    :return: list of topk accuracy [top1st, top2nd, ...] depending on your topk input
    """
    with torch.no_grad():
        # ---- get the topk most likely labels according to your model
        # get the largest k \in [n_classes] (i.e. the number of most likely probabilities we will use)
        maxk = max(topk)  # max number labels we will consider in the right choices for out model
        batch_size = target.size(0)

        # get top maxk indicies that correspond to the most likely probability scores
        # (note _ means we don't care about the actual top maxk scores just their corresponding indicies/labels)
        _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
        y_pred = y_pred.t()  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

        # - get the credit for each example if the models predictions is in maxk values (main crux of code)
        # for any example, the model will get credit if it's prediction matches the ground truth
        # for each example we compare if the model's best prediction matches the truth. If yes we get an entry of 1.
        # if the k'th top answer of the model matches the truth we get 1.
        # Note: this for any example in batch we can only ever get 1 match (so we never overestimate accuracy <1)
        target_reshaped = target.view(1, -1).expand_as(y_pred)  # [B] -> [B, 1] -> [maxk, B]
        # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
        correct = (y_pred == target_reshaped)  # [maxk, B] were for each example we know which topk prediction matched truth
        # original: correct = pred.eq(target.view(1, -1).expand_as(pred))

        # -- get topk accuracy
        list_topk_accs = []  # idx is topk1, topk2, ... etc
        for k in topk:
            # get tensor of which topk answer was right
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            # flatten it to help compute if we got it correct for each example in batch
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()  # [k, B] -> [kB]
            # get if we got it right for any of our top k prediction for each example in batch
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)  # [kB] -> [1]
            # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
            topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
            list_topk_accs.append(topk_acc)
        return list_topk_accs  # list of topk accuracies for entire batch [topk1, topk2, ... etc]