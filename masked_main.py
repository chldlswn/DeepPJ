import models  
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import xray_data
import matplotlib.pyplot as plt
import random
from sklearn import metrics
from tqdm import tqdm
from argparse import ArgumentParser
from tensorboardX import SummaryWriter
from torchvision.utils import save_image

BATCH_SIZE = 64
WORKERS = 4
IMG_SIZE = 64
DATASET = ['rsna', 'pedia']
torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
np.random.seed(0)

#  초기 마스크 불러오기 (64x64 pneumonia heatmap)
MASK_PATH = 'pneumonia_all.npy'
try:
    INIT_MASK = torch.tensor(np.load(MASK_PATH)).float().unsqueeze(0).unsqueeze(0).cuda()  # (1, 1, 64, 64)
except:
    INIT_MASK = None


def train():
    os.makedirs('./models', exist_ok=True)
    model = models.AE(opt.ls, opt.mp, opt.u, img_size=IMG_SIZE, init_mask=INIT_MASK)  # 초기 마스크 전달
    model.to(device)

    EPOCHS = 250
    loader = xray_data.get_xray_dataloader(BATCH_SIZE, WORKERS, 'train', img_size=IMG_SIZE, dataset=DATASET)
    test_loader = xray_data.get_xray_dataloader(BATCH_SIZE, WORKERS, 'test', img_size=IMG_SIZE, dataset=DATASET)

    opt.epochs = EPOCHS
    train_loop(model, loader, test_loader, opt)


def train_loop(model, loader, test_loader, opt):
    device = torch.device('cuda:{}'.format(opt.cuda))
    print(opt.exp)
    optim = torch.optim.Adam(model.parameters(), 5e-4, betas=(0.5, 0.999))
    writer = SummaryWriter('log/%s' % opt.exp)
    for e in tqdm(range(opt.epochs)):
        l1s, l2s = [], []
        model.train()
        for (x, _) in tqdm(loader):
            x = x.to(device)
            x.requires_grad = False
            if not opt.u:
                out = model(x)
                rec_err = (out - x) ** 2
                masked_rec_err = model.apply_mask(rec_err)  # 마스크 적용
                loss = masked_rec_err.mean()  # 마스크가 적용된 손실
                l1s.append(loss.item())
            else:
                mean, logvar = model(x)
                rec_err = (mean - x) ** 2
                masked_rec_err = model.apply_mask(rec_err)  # 마스크 적용
                loss1 = torch.mean(torch.exp(-logvar) * masked_rec_err)
                loss2 = torch.mean(logvar)
                loss = loss1 + loss2
                l1s.append(masked_rec_err.mean().item())
                l2s.append(loss2.item())

            optim.zero_grad()
            loss.backward()
            optim.step()

        auc = test_for_xray(opt, model, test_loader)

        if not opt.u:
            l1s = np.mean(l1s)
            writer.add_scalar('auc', auc, e)
            writer.add_scalar('rec_err', l1s, e)
            writer.add_images('recons', torch.cat((x, out)).cpu() * 0.5 + 0.5, e)
            print(f'epochs:{e}, recon error:{l1s}')
        else:
            l1s = np.mean(l1s)
            l2s = np.mean(l2s)
            writer.add_scalar('auc', auc, e)
            writer.add_scalar('rec_err', l1s, e)
            writer.add_scalar('logvars', l2s, e)
            writer.add_images('recons', torch.cat((x, mean)).cpu() * 0.5 + 0.5, e)
            writer.add_images('vars', torch.cat((x * 0.5 + 0.5, logvar.exp())).cpu(), e)
            print(f'epochs:{e}, recon error:{l1s}, logvars:{l2s}')

    torch.save(model.state_dict(), f'./models/{opt.exp}.pth')


def test_for_xray(opt, model=None, loader=None, plot=False, vae=False):
    if model is None:
        model = models.AE(opt.ls, opt.mp, opt.u, img_size=IMG_SIZE, vae=vae).to(device)
        model.load_state_dict(torch.load(f'./models/{opt.exp}.pth'))
    if loader is None:
        loader = xray_data.get_xray_dataloader(1, WORKERS, opt.testset, dataset=DATASET, img_size=IMG_SIZE)

    model.eval()
    with torch.no_grad():
        y_score, y_true = [], []
        for bid, (x, label) in tqdm(enumerate(loader)):
            x = x.to(device)
            if opt.u:
                out, logvar = model(x)
                rec_err = (out - x) ** 2
                res = torch.exp(-logvar) * rec_err
            else:
                out = model(x)
                rec_err = (out - x) ** 2
                res = rec_err

            # ✅ 마스크 적용 (optional)
            if INIT_MASK is not None:
                res = res * INIT_MASK

            res = res.mean(dim=(1, 2, 3))

            y_true.append(label.cpu())
            y_score.append(res.cpu().view(-1))

        y_true = np.concatenate(y_true)
        y_score = np.concatenate(y_score)
        auc = metrics.roc_auc_score(y_true, y_score)
        print('AUC', auc)

        if plot:
            metrics_at_eer(y_score, y_true)
            plt.figure()
            plt.hist(y_score[y_true == 0], bins=100, density=True, color='blue', alpha=0.5, label="Normal")
            plt.hist(y_score[y_true == 1], bins=100, density=True, color='red', alpha=0.5, label="Abnormal")
            plt.legend()
            plt.figure()
            fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
            plt.plot(fpr, tpr)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.show()

        return auc


def metrics_at_eer(y_score, y_true):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    idx = None
    for i in range(len(fpr)):
        fnr = 1 - tpr[i]
        if abs(fpr[i] - fnr) <= 5e-3:
            idx = i
            break
    assert idx is not None

    t = thresholds[idx]
    y_pred = np.zeros_like(y_true)
    y_pred[y_score < t] = 0
    y_pred[y_score >= t] = 1

    pres = metrics.precision_score(y_true, y_pred)
    sens = metrics.recall_score(y_true, y_pred, pos_label=1)
    spec = metrics.recall_score(y_true, y_pred, pos_label=0)
    f1 = metrics.f1_score(y_true, y_pred)

    print('Error rate:', fpr[idx])
    print(f'Precision: {pres:.4f}, Sensitivity: {sens:.4f}, Specificity: {spec:.4f}, F1-score: {f1:.4f}\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--u', dest='u', action='store_true')  # use uncertainty
    parser.add_argument('--gpu', dest='cuda', type=int, default=0)  # cuda id
    parser.add_argument('--exp', dest='exp', type=str, default='ae')  # experiment name
    parser.add_argument('--eval', dest='eval', action='store_true')  # test model
    parser.add_argument('--ls', dest='ls', type=int, default=16)  # latent size
    parser.add_argument('--mp', dest='mp', type=float, default=1)  # model capacity multiplier
    parser.add_argument('--dataset', dest='dataset', type=int, default=0)  # 0: rsna, 1: pedia
    parser.add_argument('--testset', dest='testset', type=str, default='test', help='Test set folder name (e.g., test1, test2, test3)')
    opt = parser.parse_args()

    device = torch.device(f'cuda:{opt.cuda}')
    torch.cuda.set_device(device)
    opt.exp += 'u' if opt.u else ''
    DATASET = DATASET[opt.dataset]

    if not opt.eval:
        print('start ae training...')
        train()
    else:
        for testset_name in ['test1', 'test2', 'test3']:
            print(f'▶ Evaluating on {testset_name}...')
            opt.testset = testset_name
            test_loader = xray_data.get_xray_dataloader(
                1, WORKERS, dtype=testset_name, dataset=DATASET, img_size=IMG_SIZE)
            test_for_xray(opt, loader=test_loader, plot=True)
