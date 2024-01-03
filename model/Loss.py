import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.fft
from torch.autograd import Variable
import math

from torch.autograd import Function
from numba import jit

# %% Negative Pearson's correlation
# Traken from https://github.com/ZitongYu/PhysNet/blob/master/NegPearsonLoss.py
'''
Code of 'Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks' 
By Zitong Yu, 2019/05/05
If you use the code, please cite:
@inproceedings{yu2019remote,
    title={Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks},
    author={Yu, Zitong and Li, Xiaobai and Zhao, Guoying},
    booktitle= {British Machine Vision Conference (BMVC)},
    year = {2019}
}
Only for research purpose, and commercial use is not allowed.
MIT License
Copyright (c) 2019 
'''


class Neg_Pearson(nn.Module):  # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(Neg_Pearson, self).__init__()
        return

    def forward(self, preds, labels):  # tensor [Batch, Temporal]
        loss = 0
        for i in range(preds.shape[0]):
            sum_x = torch.sum(preds[i])  # x
            sum_y = torch.sum(labels[i])  # y
            sum_xy = torch.sum(preds[i] * labels[i])  # xy
            sum_x2 = torch.sum(torch.pow(preds[i], 2))  # x^2
            sum_y2 = torch.sum(torch.pow(labels[i], 2))  # y^2
            N = preds.shape[1]
            pearson = (N * sum_xy - sum_x * sum_y) / (
                torch.sqrt((N * sum_x2 - torch.pow(sum_x, 2)) * (N * sum_y2 - torch.pow(sum_y, 2))))

            # if (pearson>=0).data.cpu().numpy():    # torch.cuda.ByteTensor -->  numpy
            #    loss += 1 - pearson
            # else:
            #    loss += 1 - torch.abs(pearson)

            loss += 1 - pearson

        loss = loss / preds.shape[0]
        return loss


# %% Negative Pearson's correlation + Signal-to-Noise-Ratio (NPSNR)

class NPSNR(nn.Module):
    def __init__(self, Lambda=1.32, LowF=0.7, upF=3.5, width=0.4, Fs=30):
        super(NPSNR, self).__init__()
        self.Lambda = Lambda
        self.LowF = LowF
        self.upF = upF
        self.width = width
        self.frame_rate = Fs
        self.NormaliceK = 1 / 10.9  # Constant to normalize SNR between -1 and 1
        return

    def forward(self, rppg, gt):
        device = rppg.device
        loss = 0
        for i in range(rppg.shape[0]):
            ##############################
            # PEARSON'S CORRELATION
            ##############################
            sum_x = torch.sum(rppg[i])  # x
            sum_y = torch.sum(gt[i])  # y
            sum_xy = torch.sum(rppg[i] * gt[i])  # xy
            sum_x2 = torch.sum(torch.pow(rppg[i], 2))  # x^2
            sum_y2 = torch.sum(torch.pow(gt[i], 2))  # y^2
            N = rppg.shape[1]
            pearson = (N * sum_xy - sum_x * sum_y) / (
                torch.sqrt((N * sum_x2 - torch.pow(sum_x, 2)) * (N * sum_y2 - torch.pow(sum_y, 2))))
            ##############################
            # SNR
            ##############################
            N = rppg.shape[-1] * 3
            # Fs = 1 / time[i].diff().mean()
            freq = torch.arange(0, N, 1, device=device) * self.frame_rate / N
            fft = torch.abs(torch.fft.fft(rppg[i], dim=-1, n=N)) ** 2
            gt_fft = torch.abs(torch.fft.fft(gt[i], dim=-1, n=N)) ** 2
            fft = fft.masked_fill(torch.logical_or(freq > self.upF, freq < self.LowF).to(device), 0)
            gt_fft = gt_fft.masked_fill(torch.logical_or(freq > self.upF, freq < self.LowF).to(device), 0)
            PPG_peaksLoc = freq[gt_fft.argmax()]
            mask = torch.zeros(fft.shape[-1], dtype=torch.bool, device=device)
            mask = mask.masked_fill(
                torch.logical_and(freq < PPG_peaksLoc + (self.width / 2), PPG_peaksLoc - (self.width / 2) < freq).to(
                    device), 1)  # Main signal
            mask = mask.masked_fill(torch.logical_and(freq < PPG_peaksLoc * 2 + (self.width / 2),
                                                      PPG_peaksLoc * 2 - (self.width / 2) < freq).to(device),
                                    1)  # Armonic
            power = fft * mask
            noise = fft * mask.logical_not().to(device)
            SNR = (10 * torch.log10(power.sum() / noise.sum())) * self.NormaliceK
            ##############################
            # JOIN BOTH LOSS FUNCTION
            ##############################
            loss += 1 - (pearson + (self.Lambda * SNR))

        loss = loss / rppg.shape[0]
        return loss


'''
Credits: https://github.com/ToyotaResearchInstitute/RemotePPG
'''


class NegativeMaxCrossCov(nn.Module):
    def __init__(self, Fs=30, high_pass=250, low_pass=40):
        super(NegativeMaxCrossCov, self).__init__()
        self.Fs = Fs
        self.high_pass = high_pass
        self.low_pass = low_pass

    def forward(self, preds, labels):
        # Normalize
        preds_norm = preds - torch.mean(preds, dim=-1, keepdim=True)
        labels_norm = labels - torch.mean(labels, dim=-1, keepdim=True)

        # Zero-pad signals to prevent circular cross-correlation
        # Also allows for signals of different length
        # https://dsp.stackexchange.com/questions/736/how-do-i-implement-cross-correlation-to-prove-two-audio-files-are-similar
        min_N = min(preds.shape[-1], labels.shape[-1])
        padded_N = max(preds.shape[-1], labels.shape[-1]) * 2
        preds_pad = F.pad(preds_norm, (0, padded_N - preds.shape[-1]))
        labels_pad = F.pad(labels_norm, (0, padded_N - labels.shape[-1]))

        # FFT
        preds_fft = torch.fft.rfft(preds_pad, dim=-1)
        labels_fft = torch.fft.rfft(labels_pad, dim=-1)

        # Cross-correlation in frequency space
        X = preds_fft * torch.conj(labels_fft)
        X_real = torch.view_as_real(X)

        # Determine ratio of energy between relevant and non-relevant regions
        Fn = self.Fs / 2
        freqs = torch.linspace(0, Fn, X.shape[-1])
        use_freqs = torch.logical_and(freqs <= self.high_pass / 60, freqs >= self.low_pass / 60)
        zero_freqs = torch.logical_not(use_freqs)
        use_energy = torch.sum(torch.linalg.norm(X_real[:, use_freqs], dim=-1), dim=-1)
        zero_energy = torch.sum(torch.linalg.norm(X_real[:, zero_freqs], dim=-1), dim=-1)
        denom = use_energy + zero_energy
        energy_ratio = torch.ones_like(denom)
        for ii in range(len(denom)):
            if denom[ii] > 0:
                energy_ratio[ii] = use_energy[ii] / denom[ii]

        # Zero out irrelevant freqs
        X[:, zero_freqs] = 0.

        # Inverse FFT and normalization
        cc = torch.fft.irfft(X, dim=-1) / (min_N - 1)

        # Max of cross correlation, adjusted for relevant energy
        max_cc = torch.max(cc, dim=-1)[0] / energy_ratio

        # return -max_cc
        return max_cc


class NegativeMaxCrossCorr(nn.Module):
    def __init__(self, Fs=30, high_pass=250, low_pass=40):
        super(NegativeMaxCrossCorr, self).__init__()
        self.cross_cov = NegativeMaxCrossCov(Fs, high_pass, low_pass)

    def forward(self, preds, labels):
        denom = torch.std(preds, dim=-1) * torch.std(labels, dim=-1)
        cov = self.cross_cov(preds, labels)
        output = torch.zeros_like(cov)
        for ii in range(len(denom)):
            if denom[ii] > 0:
                output[ii] = cov[ii] / denom[ii]
        loss = torch.mean(output)
        return loss


class MCC_NP(nn.Module):
    def __init__(self, Fs=30):
        super(MCC_NP, self).__init__()
        self.mcc = NegativeMaxCrossCorr(Fs=Fs)
        self.np = Neg_Pearson()

    def forward(self, preds, labels):
        return self.mcc(preds, labels) + self.np(preds, labels)


class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean')

    def forward(self, preds, labels):
        labels = F.one_hot(labels, num_classes=140).float()
        # labels = F.softmax(labels, dim=1)
        return self.kl(preds, labels)


class MultiFocalLoss(nn.Module):
    """
    Focal_Loss= -1*alpha*((1-pt)**gamma)*log(pt)
    Args:
        num_class: number of classes
        alpha: class balance factor shape=[num_class, ]
        gamma: hyper-parameter
        reduction: reduction type
    """

    def __init__(self, num_class, alpha=None, gamma=2, reduction='mean'):
        super(MultiFocalLoss, self).__init__()
        self.num_class = num_class
        self.gamma = gamma
        self.reduction = reduction
        self.smooth = 1e-4
        self.alpha = alpha
        if alpha is None:
            self.alpha = torch.ones(num_class, ) - 0.5
        elif isinstance(alpha, (int, float)):
            self.alpha = torch.as_tensor([alpha] * num_class)
        elif isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.as_tensor(alpha)
        if self.alpha.shape[0] != num_class:
            raise RuntimeError('the length not equal to number of class')

    def forward(self, output, target):
        # assert isinstance(self.alpha,torch.Tensor)\
        alpha = self.alpha.to(output.device)
        prob = F.softmax(output, dim=1)

        if prob.dim() > 2:
            # used for 3d-conv:  N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            N, C = output.shape[:2]
            prob = prob.view(N, C, -1)
            prob = prob.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
            prob = prob.view(-1, prob.size(-1))  # [N,d1*d2..,C]-> [N*d1*d2..,C]

        ori_shp = target.shape
        target = target.view(-1, 1)

        prob = prob.gather(1, target).view(-1) + self.smooth  # avoid nan
        logpt = torch.log(prob)
        # alpha_class = alpha.gather(0, target.squeeze(-1))
        alpha_weight = alpha[target.squeeze().long()]
        loss = -alpha_weight * torch.pow(torch.sub(1.0, prob), self.gamma) * logpt

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'none':
            loss = loss.view(ori_shp)

        return loss


class MixDILATELoss(nn.Module):
    def __init__(self):
        super(MixDILATELoss, self).__init__()
        self.np = Neg_Pearson()
        self.softdtw_batch = SoftDTWBatch.apply
        self.path_dtw = PathDTWBatch.apply
        self.gamma = 0.01
        self.alpha = 0.5

    def forward(self, preds, labels):
        loss_np = self.np(preds, labels)
        preds = preds.unsqueeze(2)
        labels = labels.unsqueeze(2)
        batch_size, N_output = preds.shape[0:2]
        D = torch.zeros((batch_size, N_output, N_output)).to(preds.device)
        for k in range(batch_size):
            Dk = pairwise_distances(labels[k, :, :].view(-1, 1), preds[k, :, :].view(-1, 1))
            D[k:k + 1, :, :] = Dk
        loss_shape = self.softdtw_batch(D, self.gamma)
        path = self.path_dtw(D, self.gamma)
        Omega = pairwise_distances(torch.arange(1, N_output + 1).view(N_output, 1)).to(preds.device)
        loss_temporal = torch.sum(path * Omega) / (N_output * N_output)
        loss = self.alpha * loss_shape + (1 - self.alpha) * loss_temporal
        return loss_np + 0.1 * loss


class PathDTWBatch(Function):
    @staticmethod
    def forward(ctx, D, gamma):  # D.shape: [batch_size, N , N]
        batch_size, N, N = D.shape
        device = D.device
        D_cpu = D.detach().cpu().numpy()
        gamma_gpu = torch.FloatTensor([gamma]).to(device)

        grad_gpu = torch.zeros((batch_size, N, N)).to(device)
        Q_gpu = torch.zeros((batch_size, N + 2, N + 2, 3)).to(device)
        E_gpu = torch.zeros((batch_size, N + 2, N + 2)).to(device)

        for k in range(0, batch_size):  # loop over all D in the batch
            _, grad_cpu_k, Q_cpu_k, E_cpu_k = dtw_grad(D_cpu[k, :, :], gamma)
            grad_gpu[k, :, :] = torch.FloatTensor(grad_cpu_k).to(device)
            Q_gpu[k, :, :, :] = torch.FloatTensor(Q_cpu_k).to(device)
            E_gpu[k, :, :] = torch.FloatTensor(E_cpu_k).to(device)
        ctx.save_for_backward(grad_gpu, D, Q_gpu, E_gpu, gamma_gpu)
        return torch.mean(grad_gpu, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        device = grad_output.device
        grad_gpu, D_gpu, Q_gpu, E_gpu, gamma = ctx.saved_tensors
        D_cpu = D_gpu.detach().cpu().numpy()
        Q_cpu = Q_gpu.detach().cpu().numpy()
        E_cpu = E_gpu.detach().cpu().numpy()
        gamma = gamma.detach().cpu().numpy()[0]
        Z = grad_output.detach().cpu().numpy()

        batch_size, N, N = D_cpu.shape
        Hessian = torch.zeros((batch_size, N, N)).to(device)
        for k in range(0, batch_size):
            _, hess_k = dtw_hessian_prod(D_cpu[k, :, :], Z, Q_cpu[k, :, :, :], E_cpu[k, :, :], gamma)
            Hessian[k:k + 1, :, :] = torch.FloatTensor(hess_k).to(device)

        return Hessian, None


class SoftDTWBatch(Function):
    @staticmethod
    def forward(ctx, D, gamma=1.0):  # D.shape: [batch_size, N , N]
        dev = D.device
        batch_size, N, N = D.shape
        gamma = torch.FloatTensor([gamma]).to(dev)
        D_ = D.detach().cpu().numpy()
        g_ = gamma.item()

        total_loss = 0
        R = torch.zeros((batch_size, N + 2, N + 2)).to(dev)
        for k in range(0, batch_size):  # loop over all D in the batch
            Rk = torch.FloatTensor(compute_softdtw(D_[k, :, :], g_)).to(dev)
            R[k:k + 1, :, :] = Rk
            total_loss = total_loss + Rk[-2, -2]
        ctx.save_for_backward(D, R, gamma)
        return total_loss / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        D, R, gamma = ctx.saved_tensors
        batch_size, N, N = D.shape
        D_ = D.detach().cpu().numpy()
        R_ = R.detach().cpu().numpy()
        g_ = gamma.item()

        E = torch.zeros((batch_size, N, N)).to(dev)
        for k in range(batch_size):
            Ek = torch.FloatTensor(compute_softdtw_backward(D_[k, :, :], R_[k, :, :], g_)).to(dev)
            E[k:k + 1, :, :] = Ek

        return grad_output * E, None


@jit(nopython=True)
def my_max(x, gamma):
    # use the log-sum-exp trick
    max_x = np.max(x)
    exp_x = np.exp((x - max_x) / gamma)
    Z = np.sum(exp_x)
    return gamma * np.log(Z) + max_x, exp_x / Z


@jit(nopython=True)
def my_min(x, gamma):
    min_x, argmax_x = my_max(-x, gamma)
    return - min_x, argmax_x


@jit(nopython=True)
def my_max_hessian_product(p, z, gamma):
    return (p * z - p * np.sum(p * z)) / gamma


@jit(nopython=True)
def my_min_hessian_product(p, z, gamma):
    return - my_max_hessian_product(p, z, gamma)


@jit(nopython=True)
def dtw_grad(theta, gamma):
    m = theta.shape[0]
    n = theta.shape[1]
    V = np.zeros((m + 1, n + 1))
    V[:, 0] = 1e10
    V[0, :] = 1e10
    V[0, 0] = 0

    Q = np.zeros((m + 2, n + 2, 3))

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # theta is indexed starting from 0.
            v, Q[i, j] = my_min(np.array([V[i, j - 1],
                                          V[i - 1, j - 1],
                                          V[i - 1, j]]), gamma)
            V[i, j] = theta[i - 1, j - 1] + v

    E = np.zeros((m + 2, n + 2))
    E[m + 1, :] = 0
    E[:, n + 1] = 0
    E[m + 1, n + 1] = 1
    Q[m + 1, n + 1] = 1

    for i in range(m, 0, -1):
        for j in range(n, 0, -1):
            E[i, j] = Q[i, j + 1, 0] * E[i, j + 1] + \
                      Q[i + 1, j + 1, 1] * E[i + 1, j + 1] + \
                      Q[i + 1, j, 2] * E[i + 1, j]

    return V[m, n], E[1:m + 1, 1:n + 1], Q, E


@jit(nopython=True)
def dtw_hessian_prod(theta, Z, Q, E, gamma):
    m = Z.shape[0]
    n = Z.shape[1]

    V_dot = np.zeros((m + 1, n + 1))
    V_dot[0, 0] = 0

    Q_dot = np.zeros((m + 2, n + 2, 3))
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # theta is indexed starting from 0.
            V_dot[i, j] = Z[i - 1, j - 1] + \
                          Q[i, j, 0] * V_dot[i, j - 1] + \
                          Q[i, j, 1] * V_dot[i - 1, j - 1] + \
                          Q[i, j, 2] * V_dot[i - 1, j]

            v = np.array([V_dot[i, j - 1], V_dot[i - 1, j - 1], V_dot[i - 1, j]])
            Q_dot[i, j] = my_min_hessian_product(Q[i, j], v, gamma)
    E_dot = np.zeros((m + 2, n + 2))

    for j in range(n, 0, -1):
        for i in range(m, 0, -1):
            E_dot[i, j] = Q_dot[i, j + 1, 0] * E[i, j + 1] + \
                          Q[i, j + 1, 0] * E_dot[i, j + 1] + \
                          Q_dot[i + 1, j + 1, 1] * E[i + 1, j + 1] + \
                          Q[i + 1, j + 1, 1] * E_dot[i + 1, j + 1] + \
                          Q_dot[i + 1, j, 2] * E[i + 1, j] + \
                          Q[i + 1, j, 2] * E_dot[i + 1, j]

    return V_dot[m, n], E_dot[1:m + 1, 1:n + 1]


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, float('inf'))


@jit(nopython=True)
def compute_softdtw(D, gamma):
    N = D.shape[0]
    M = D.shape[1]
    R = np.zeros((N + 2, M + 2)) + 1e8
    R[0, 0] = 0
    for j in range(1, M + 1):
        for i in range(1, N + 1):
            r0 = -R[i - 1, j - 1] / gamma
            r1 = -R[i - 1, j] / gamma
            r2 = -R[i, j - 1] / gamma
            rmax = max(max(r0, r1), r2)
            rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
            softmin = - gamma * (np.log(rsum) + rmax)
            R[i, j] = D[i - 1, j - 1] + softmin
    return R


@jit(nopython=True)
def compute_softdtw_backward(D_, R, gamma):
    N = D_.shape[0]
    M = D_.shape[1]
    D = np.zeros((N + 2, M + 2))
    E = np.zeros((N + 2, M + 2))
    D[1:N + 1, 1:M + 1] = D_
    E[-1, -1] = 1
    R[:, -1] = -1e8
    R[-1, :] = -1e8
    R[-1, -1] = R[-2, -2]
    for j in range(M, 0, -1):
        for i in range(N, 0, -1):
            a0 = (R[i + 1, j] - R[i, j] - D[i + 1, j]) / gamma
            b0 = (R[i, j + 1] - R[i, j] - D[i, j + 1]) / gamma
            c0 = (R[i + 1, j + 1] - R[i, j] - D[i + 1, j + 1]) / gamma
            a = np.exp(a0)
            b = np.exp(b0)
            c = np.exp(c0)
            E[i, j] = E[i + 1, j] * a + E[i, j + 1] * b + E[i + 1, j + 1] * c
    return E[1:N + 1, 1:M + 1]


class MultiStepClassLoss(nn.Module):

    def __init__(self, num_class, rough_num_class, alpha=None, gamma=2, reduction='mean',
                 rough_alpha=None, rough_gamma=2, rough_reduction='mean'):
        super(MultiStepClassLoss, self).__init__()
        self.subdivision = MultiFocalLoss(num_class, alpha, gamma, reduction)
        self.rough = MultiFocalLoss(rough_num_class, rough_alpha, rough_gamma, rough_reduction)
        self.rate = int(num_class / rough_num_class)
        self.rough_num_class = rough_num_class

    def forward(self, output, target):
        batch_size = output.shape[0]
        loss_subdivision = self.subdivision(output, target)
        target_rough = (target / 10).long()
        output_rough = torch.zeros((batch_size, self.rough_num_class)).to(output.device)
        for b in range(batch_size):
            for id, item in enumerate(output[b]):
                output_rough[b][int(id / 10)] += item
        loss_rough = self.rough(output_rough, target_rough)
        loss = loss_subdivision + loss_rough
        return loss


class MixWaveLoss(nn.Module):
    def __init__(self, Fs=25):
        super(MixWaveLoss, self).__init__()
        self.np = Neg_Pearson()
        self.l1loss = nn.L1Loss()
        self.psdloss = PSDLoss(Fs)

    def forward(self, preds, labels):
        loss_np = self.np(preds, labels)
        loss_l1 = self.l1loss(preds, labels)
        loss_psd = self.psdloss(preds, labels) * 50
        return loss_np + loss_l1 + loss_psd


class PSDLoss(nn.Module):
    def __init__(self, Fs, high_pass=40, low_pass=180):
        super(PSDLoss, self).__init__()
        self.norm_psd = CalculateNormPSD(Fs, high_pass, low_pass)
        self.distance_func = nn.MSELoss(reduction='mean')

    def forward(self, preds, labels):
        loss_mse = 0
        len = 0
        for id, item in enumerate(preds):
            pred = self.norm_psd(item)
            label = self.norm_psd(labels[id])
            loss_mse += self.distance_func(pred, label)
            len += 1
        return loss_mse / len


class CalculateNormPSD(nn.Module):
    # we reuse the code in Gideon2021 to get the normalized power spectral density
    # Gideon, John, and Simon Stent. "The way to my heart is through contrastive learning: Remote photoplethysmography from unlabelled video." Proceedings of the IEEE/CVF international conference on computer vision. 2021.

    def __init__(self, Fs, high_pass, low_pass):
        super().__init__()
        self.Fs = Fs
        self.high_pass = high_pass
        self.low_pass = low_pass

    def forward(self, x, zero_pad=0):
        x = x - torch.mean(x, dim=-1, keepdim=True)
        if zero_pad > 0:
            L = x.shape[-1]
            x = F.pad(x, (int(zero_pad / 2 * L), int(zero_pad / 2 * L)), 'constant', 0)

        # Get PSD
        x = torch.view_as_real(torch.fft.rfft(x, dim=-1, norm='forward'))
        x = torch.add(x[:, 0] ** 2, x[:, 1] ** 2)

        # Filter PSD for relevant parts
        Fn = self.Fs / 2
        freqs = torch.linspace(0, Fn, x.shape[0])
        use_freqs = torch.logical_and(freqs >= self.high_pass / 60, freqs <= self.low_pass / 60)
        x = x[use_freqs]

        # Normalize PSD
        x = x / torch.sum(x, dim=-1, keepdim=True)
        return x


class SelfWaveLoss(nn.Module):
    def __init__(self, Fs, low_hz=0.66666667, high_hz=3, device=None):
        super(SelfWaveLoss, self).__init__()
        self.np = Neg_Pearson()
        self.fps = Fs
        self.low_hz = low_hz
        self.high_hz = high_hz
        self.EPSILON = 1e-10
        self.l1loss = nn.L1Loss()

    def forward(self, preds, labels):
        device = preds.device
        l1loss = self.l1loss(preds, labels)
        loss_np = self.np(preds, labels)
        freqs, psd = self.power_spectral_density(preds, normalize=False, bandpass=False)
        bandwidth_loss = self.bandwidth_loss(freqs, psd)
        sparsity_loss = self.sparsity_loss(freqs, psd, device=device)
        return [l1loss + loss_np + bandwidth_loss + sparsity_loss, l1loss, loss_np, bandwidth_loss, sparsity_loss]

    def power_spectral_density(self, x, nfft=5400, return_angle=False, radians=True, normalize=True,
                               bandpass=True):
        centered = x - torch.mean(x, keepdim=True, dim=1)
        rfft_out = torch.fft.rfft(centered, n=nfft, dim=1)
        psd = torch.abs(rfft_out) ** 2
        N = psd.shape[1]
        freqs = torch.fft.rfftfreq(2 * N - 1, 1 / self.fps)
        if return_angle:
            angle = torch.angle(rfft_out)
            if not radians:
                angle = torch.rad2deg(angle)
            if bandpass:
                freqs, psd, angle = self.ideal_bandpass(freqs, psd, self.low_hz, self.high_hz, angle=angle)
            if normalize:
                psd = self.normalize_psd(psd)
            return freqs, psd, angle
        else:
            if bandpass:
                freqs, psd = self.ideal_bandpass(freqs, psd, self.low_hz, self.high_hz)
            if normalize:
                psd = self.normalize_psd(psd)
            return freqs, psd

    def ideal_bandpass(self, freqs, psd, low_hz, high_hz):
        freq_idcs = torch.logical_and(freqs >= low_hz, freqs <= high_hz)
        freqs = freqs[freq_idcs]
        psd = psd[:, freq_idcs]
        return freqs, psd

    def normalize_psd(self, psd):
        return psd / torch.sum(psd, keepdim=True, dim=1)  ## treat as probabilities

    def bandwidth_loss(self, freqs, psd):
        use_freqs = torch.logical_and(freqs >= self.low_hz, freqs <= self.high_hz)
        zero_freqs = torch.logical_not(use_freqs)
        use_energy = torch.sum(psd[:, use_freqs], dim=1)
        zero_energy = torch.sum(psd[:, zero_freqs], dim=1)
        denom = use_energy + zero_energy + self.EPSILON
        bandwidth_loss = torch.mean(zero_energy / denom)
        return bandwidth_loss

    def sparsity_loss(self, freqs, psd, freq_delta=0.1, device=None):
        ''' We treat this as a dynamic IPR dependent on the maximum predicted frequency.
            Arguments:
                freq_delta (float): pad for maximum frequency window we integrate over in Hertz
        '''
        signal_freq_idx = torch.argmax(psd, dim=1)
        signal_freq = freqs[signal_freq_idx].view(-1, 1)
        freqs = freqs.repeat(psd.shape[0], 1)
        low_cut = signal_freq - freq_delta
        high_cut = signal_freq + freq_delta
        band_idcs = torch.logical_and(freqs >= low_cut, freqs <= high_cut).to(device)
        signal_band = torch.sum(psd * band_idcs, dim=1)
        noise_band = torch.sum(psd * torch.logical_not(band_idcs), dim=1)
        denom = signal_band + noise_band + self.EPSILON
        sparsity_loss = torch.mean(noise_band / denom)
        return sparsity_loss
