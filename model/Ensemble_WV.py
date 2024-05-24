import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import clamp, scale_pixels_neg11_to_01, scale_pixels_01_to_neg11
from generate_wm import generate_watermark,generate_watermark_ori
from model.WDNet import generator
from model.BVMR import load_globals, init_nets
from model.SplitNet import models

class Ensemble(nn.Module):
    def __init__(self):
        super().__init__()
        print("Loading WDNet...")
        model = generator(3, 3)
        model.load_state_dict(torch.load('./checkpoints/WDNet_G.pkl', map_location='cpu'))
        self.WDNet = model.cuda().eval()

        print("Loading BVMR...")
        _opt = load_globals(nets_path='./checkpoints/BVMR', globals_dict={}, override=False)
        _device = torch.device('cuda')
        model = init_nets(opt=_opt, net_path='./checkpoints/BVMR', device=_device, tag=11).eval()
        self.BVMR = model

        print("Loading SplitNet...")
        model = models.__dict__['vvv4n']()
        model.load_state_dict(torch.load('./checkpoints/SplitNet/splitnet.pth.tar')['state_dict'])
        self.SplitNet = model.cuda().eval()

    def forward(self, img_source, delta, type):
        loss = 0
        if type == 'DWV':
            # WDNet
            start_pred_target, start_mask, start_alpha, start_w, start_I_watermark = self.WDNet(img_source + delta)
            loss += F.mse_loss(img_source.data, start_pred_target.float())

            # SplitNet
            mid_output, mid_mask, _ = self.SplitNet(img_source + delta)
            mid_refine = mid_output[0] * mid_mask + img_source * (1 - mid_mask)
            loss += 2*F.mse_loss(img_source.data, mid_refine.float())

            # BVMR
            mid_adv1 = scale_pixels_01_to_neg11(img_source + delta) # [-1, 1]
            mid_output = self.BVMR(mid_adv1)
            # mid_guess_images: [-1, 1], mid_guess_mask: [0, 1]
            mid_guess_images, mid_guess_mask = mid_output[0], mid_output[1]
            mid_expanded_guess_mask = mid_guess_mask.repeat(1, 3, 1, 1)
            mid_reconstructed_pixels = mid_guess_images * mid_expanded_guess_mask
            mid_reconstructed_images = mid_adv1 * (1 - mid_expanded_guess_mask) + mid_reconstructed_pixels
            mid_reconstructed_images = scale_pixels_neg11_to_01(mid_reconstructed_images) # [-1, 1] -> [0, 1]
            loss += F.mse_loss(img_source.data, mid_reconstructed_images.float())

            return loss

        elif type == 'IWV':
            mask_black = torch.zeros((1, 256, 256)).cuda()
            mask_black = torch.unsqueeze(mask_black, 0)

            # WDNet
            start_pred_target, start_mask, start_alpha, start_w, start_I_watermark = self.WDNet(img_source + delta)
            loss += 2 * F.mse_loss(img_source.data, start_pred_target.float()) + F.mse_loss(mask_black.data,
                                                                                start_mask.float())
            
            # SplitNet
            mid_output, mid_mask, _ = self.SplitNet(img_source + delta)
            mid_refine = mid_output[0] * mid_mask + img_source * (1 - mid_mask)
            loss += 2 * F.mse_loss(img_source.data, mid_refine.float()) + F.mse_loss(mask_black.data, mid_mask.float())

            # BVMR
            mid_adv2 = scale_pixels_01_to_neg11(img_source + delta) # [-1, 1]
            mid_output = self.BVMR(mid_adv2)
            mid_guess_images, mid_guess_mask = mid_output[0], mid_output[1]
            mid_expanded_guess_mask = mid_guess_mask.repeat(1, 3, 1, 1)
            mid_reconstructed_pixels = mid_guess_images * mid_expanded_guess_mask
            mid_reconstructed_images = mid_adv2 * (1 - mid_expanded_guess_mask) + mid_reconstructed_pixels
            mid_reconstructed_images = scale_pixels_neg11_to_01(mid_reconstructed_images) # [-1, 1] -> [0, 1]
            mask_black = torch.zeros_like(mid_expanded_guess_mask).cuda()
            loss += F.mse_loss(img_source.data, mid_reconstructed_images.float()) + 2*F.mse_loss((mask_black).data,
                                                                                                     mid_expanded_guess_mask.float())
            return loss
            
class Ensemble_WV(object):
    def __init__(self,model,args,epsilon,start_epsilon,step_alpha,upper_limit,lower_limit):
        self.model_loss = model
        self.epsilon = epsilon
        self.start_epsilon = start_epsilon
        self.step_alpha = step_alpha
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit
        self.args = args

    def DWV(self,img_source,logo_path,seed,attack_type='Original'):
        delta = torch.zeros_like(img_source).cuda()
        delta.requires_grad = True
        if attack_type == 'MIFGSM':
            decay = 1.0
            momentum = torch.zeros_like(img_source).cuda()
        for i in range(self.args.attack_iter):
            loss = self.model_loss(img_source, delta, type='DWV')
            loss.backward()
            grad = delta.grad.detach()
            d = delta

            if attack_type == 'MIFGSM':
                grad_norm = grad / torch.norm(torch.abs(grad), p=1, dim=(1, 2, 3), keepdim=True)
                grad = grad_norm + momentum * decay
                momentum = grad            

            d = clamp(d + self.step_alpha * torch.sign(grad), -self.epsilon, self.epsilon)
            delta.data = d
            delta.grad.zero_()
        adv1 = clamp(img_source + delta, self.lower_limit, self.upper_limit)
        adv1, _ = generate_watermark_ori(adv1, logo_path, seed)
        adv1 = torch.unsqueeze(adv1.cuda(), 0)
        # adv1_pred, adv1_mask, alpha, w, _ = self.model_loss(adv1)
        return adv1 # ,adv1_pred,adv1_mask

    def IWV(self,img_source,logo_path,seed,attack_type='Original'):
        delta = torch.zeros_like(img_source).cuda()
        delta.requires_grad = True
        if attack_type == 'MIFGSM':
            decay = 1.0
            momentum = torch.zeros_like(img_source).cuda()

        for i in range(self.args.attack_iter):
            loss = self.model_loss(img_source, delta, type='IWV')
            loss.backward()
            grad = -delta.grad.detach()
            d = delta

            if attack_type == 'MIFGSM':
                grad_norm = grad / torch.norm(torch.abs(grad), p=1, dim=(1, 2, 3), keepdim=True)
                grad = grad_norm + momentum * decay
                momentum = grad            

            d = clamp(d + self.step_alpha * torch.sign(grad), -self.epsilon, self.epsilon)
            delta.data = d
            delta.grad.zero_()
        adv2 = clamp(img_source + delta, self.lower_limit, self.upper_limit)
        adv2, _ = generate_watermark_ori(adv2, logo_path, seed)
        adv2 = torch.unsqueeze(adv2.cuda(), 0)
        # adv2_pred, adv2_mask, alpha, w, _ = self.model_loss(adv2)
        return adv2 # ,adv2_pred, adv2_mask