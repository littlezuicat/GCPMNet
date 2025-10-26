import torch
from torch.nn import functional as FF


def loss_fn(pred_img, label_img):

    label_img2 = FF.interpolate(label_img, scale_factor=0.5, mode='bilinear')
    label_img4 = FF.interpolate(label_img, scale_factor=0.25, mode='bilinear')
    l1 = FF.l1_loss(pred_img[0], label_img4)
    l2 = FF.l1_loss(pred_img[1], label_img2)
    l3 = FF.l1_loss(pred_img[2], label_img)
    loss_content = l1+l2+l3

    label_fft1 = torch.fft.fft2(label_img4, dim=(-2,-1))
    label_fft1 = torch.stack((label_fft1.real, label_fft1.imag), -1)

    pred_fft1 = torch.fft.fft2(pred_img[0], dim=(-2,-1))
    pred_fft1 = torch.stack((pred_fft1.real, pred_fft1.imag), -1)

    label_fft2 = torch.fft.fft2(label_img2, dim=(-2,-1))
    label_fft2 = torch.stack((label_fft2.real, label_fft2.imag), -1)

    pred_fft2 = torch.fft.fft2(pred_img[1], dim=(-2,-1))
    pred_fft2 = torch.stack((pred_fft2.real, pred_fft2.imag), -1)

    label_fft3 = torch.fft.fft2(label_img, dim=(-2,-1))
    label_fft3 = torch.stack((label_fft3.real, label_fft3.imag), -1)

    pred_fft3 = torch.fft.fft2(pred_img[2], dim=(-2,-1))
    pred_fft3 = torch.stack((pred_fft3.real, pred_fft3.imag), -1)

    f1 = FF.l1_loss(pred_fft1, label_fft1)
    f2 = FF.l1_loss(pred_fft2, label_fft2)
    f3 = FF.l1_loss(pred_fft3, label_fft3)
    loss_fft = f1+f2+f3

    loss = loss_content + 0.1 * loss_fft
    return loss