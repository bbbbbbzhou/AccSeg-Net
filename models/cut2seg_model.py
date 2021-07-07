import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import itertools
import util.util as util
import math
import pdb


def dice_loss_norm(input, target):
    """
    input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
    target is a 1-hot representation of the groundtruth, should have same size as the input
    """
    assert input.size() == target.size(), "Input sizes must be equal."
    assert input.dim() == 4, "Input must be a 4D Tensor."
    # uniques = np.unique(target.numpy())
    # assert set(list(uniques)) <= set([0, 1]), "target must only contain zeros and ones"

    probs = input
    num = probs * target  # b,c,h,w--p*g
    num = torch.sum(num, dim=3)
    num = torch.sum(num, dim=2)  #
    num = torch.sum(num, dim=0)# b,c

    den1 = probs * probs  # --p^2
    den1 = torch.sum(den1, dim=3)
    den1 = torch.sum(den1, dim=2)  # b,c,1,1
    den1 = torch.sum(den1, dim=0)

    den2 = target * target  # --g^2
    den2 = torch.sum(den2, dim=3)
    den2 = torch.sum(den2, dim=2)  # b,c,1,1
    den2 = torch.sum(den2, dim=0)

    dice = 2 * ((num+0.0000001) / (den1 + den2+0.0000001))
    # dice_eso = dice[1:]  # we ignore bg dice val, and take the fg
    dice_eso = dice
    dice_total = -1 * torch.sum(dice_eso) / dice_eso.size(0)  # divide by batch_sz
    return dice_total


def Cor_CoeLoss(y_pred, y_target):
    x = y_pred
    y = y_target
    x_var = x - torch.mean(x)
    y_var = y - torch.mean(y)
    r_num = torch.sum(x_var * y_var)
    r_den = torch.sqrt(torch.sum(x_var ** 2)) * torch.sqrt(torch.sum(y_var ** 2))
    r = r_num / r_den

    # return 1 - r  # best are 0
    return 1 - r**2    # abslute constrain


class MIND(torch.nn.Module):
    def __init__(self, non_local_region_size=9, patch_size=7, neighbor_size=3, gaussian_patch_sigma=3.0):
        super(MIND, self).__init__()
        self.nl_size = non_local_region_size
        self.p_size = patch_size
        self.n_size = neighbor_size
        self.sigma2 = gaussian_patch_sigma * gaussian_patch_sigma

        # calc shifted images in non local region
        self.image_shifter = torch.nn.Conv2d(in_channels=1, out_channels=self.nl_size * self.nl_size,
                                             kernel_size=(self.nl_size, self.nl_size),
                                             stride=1, padding=((self.nl_size-1)//2, (self.nl_size-1)//2),
                                             dilation=1, groups=1, bias=False, padding_mode='zeros')

        for i in range(self.nl_size*self.nl_size):
            t = torch.zeros((1, self.nl_size, self.nl_size))
            t[0, i % self.nl_size, i//self.nl_size] = 1
            self.image_shifter.weight.data[i] = t

        # patch summation
        self.summation_patcher = torch.nn.Conv2d(in_channels=self.nl_size*self.nl_size, out_channels=self.nl_size*self.nl_size,
                                                 kernel_size=(self.p_size, self.p_size),
                                                 stride=1, padding=((self.p_size-1)//2, (self.p_size-1)//2),
                                                 dilation=1, groups=self.nl_size*self.nl_size, bias=False, padding_mode='zeros')

        for i in range(self.nl_size*self.nl_size):
            # gaussian kernel
            t = torch.zeros((1, self.p_size, self.p_size))
            cx = (self.p_size-1)//2
            cy = (self.p_size-1)//2
            for j in range(self.p_size * self.p_size):
                x = j % self.p_size
                y = j//self.p_size
                d2 = torch.norm(torch.tensor([x-cx, y-cy]).float(), 2)
                t[0, x, y] = math.exp(-d2 / self.sigma2)

            self.summation_patcher.weight.data[i] = t

        # neighbor images
        self.neighbors = torch.nn.Conv2d(in_channels=1, out_channels=self.n_size*self.n_size,
                                         kernel_size=(self.n_size, self.n_size),
                                         stride=1, padding=((self.n_size-1)//2, (self.n_size-1)//2),
                                         dilation=1, groups=1, bias=False, padding_mode='zeros')

        for i in range(self.n_size*self.n_size):
            t = torch.zeros((1, self.n_size, self.n_size))
            t[0, i % self.n_size, i//self.n_size] = 1
            self.neighbors.weight.data[i] = t


        # neighbor patcher
        self.neighbor_summation_patcher = torch.nn.Conv2d(in_channels=self.n_size*self.n_size, out_channels=self.n_size * self.n_size,
                                                          kernel_size=(self.p_size, self.p_size),
                                                          stride=1, padding=((self.p_size-1)//2, (self.p_size-1)//2),
                                                          dilation=1, groups=self.n_size*self.n_size, bias=False, padding_mode='zeros')

        for i in range(self.n_size*self.n_size):
            t = torch.ones((1, self.p_size, self.p_size))
            self.neighbor_summation_patcher.weight.data[i] = t

    def forward(self, orig):
        assert(len(orig.shape) == 4)
        assert(orig.shape[1] == 1)

        # get original image channel stack
        orig_stack = torch.stack([orig.squeeze(dim=1) for i in range(self.nl_size*self.nl_size)], dim=1)

        # get shifted images
        shifted = self.image_shifter(orig)

        # get image diff
        diff_images = shifted - orig_stack

        # diff's L2 norm
        Dx_alpha = self.summation_patcher(torch.pow(diff_images, 2.0))

        # calc neighbor's variance
        neighbor_images = self.neighbor_summation_patcher(self.neighbors(orig))
        Vx = neighbor_images.var(dim=1).unsqueeze(dim=1)

        # output mind
        nume = torch.exp(-Dx_alpha / (Vx + 1e-8))
        denomi = nume.sum(dim=1).unsqueeze(dim=1)
        mind = nume / denomi
        return mind


class MINDLoss(torch.nn.Module):
    def __init__(self, non_local_region_size=9, patch_size=7, neighbor_size=3, gaussian_patch_sigma=3.0):
        super(MINDLoss, self).__init__()
        self.nl_size = non_local_region_size
        self.MIND = MIND(non_local_region_size=non_local_region_size,
                         patch_size=patch_size,
                         neighbor_size=neighbor_size,
                         gaussian_patch_sigma=gaussian_patch_sigma)

    def forward(self, input, target):
        in_mind = self.MIND(input)
        tar_mind = self.MIND(target)
        mind_diff = in_mind - tar_mind
        l1 =torch.norm(mind_diff, 1)
        return l1/(input.shape[2] * input.shape[3] * self.nl_size * self.nl_size)


class CUT2SEGModel_TRAIN(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE', 'CC', 'MIND', 'DICE']
        self.visual_names = ['real_A', 'fake_B', 'real_B', 'seg_real_A', 'seg_fake_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D', 'S']
        else:  # during test time, only load S
            self.model_names = ['S']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        self.netS = networks.define_S(opt.input_nc, opt.output_nc_seg, opt.ngf, opt.netS, opt.normS, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []
            self.criterionMIND = MINDLoss(non_local_region_size=9, patch_size=7, neighbor_size=3, gaussian_patch_sigma=3.0).cuda()

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters(), self.netS.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                  # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.seg_real_A = input['Seg'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

        self.seg_fake_B = self.netS(self.fake_B)

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        if self.opt.lambda_MIND > 0.0:
            self.loss_MIND = self.criterionMIND(self.real_A, self.fake_B) * self.opt.lambda_MIND
        else:
            self.loss_MIND = 0.0

        if self.opt.lambda_CC > 0.0:
            self.loss_CC = Cor_CoeLoss(self.real_A, self.fake_B) * self.opt.lambda_CC
        else:
            self.loss_CC = 0.0

        if self.opt.lambda_DICE > 0.0:
            self.loss_DICE = dice_loss_norm(self.seg_real_A, self.seg_fake_B) * self.opt.lambda_DICE
        else:
            self.loss_DICE = 0.0

        self.loss_G = self.loss_G_GAN + loss_NCE_both + self.loss_MIND + self.loss_CC + self.loss_DICE
        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers


class CUT2SEGModel_TEST(BaseModel):
    def __init__(self, opt):
        assert (not opt.isTrain)
        BaseModel.__init__(self, opt)
        self.netS = networks.define_S(opt.input_nc, opt.output_nc_seg, opt.ngf, opt.netS, opt.normS, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.load_network_S(self.netS, 'S', opt.which_epoch_S)

    def forward(self):
        pass

    def optimize_parameters(self):
        pass

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        self.real_B = input['B'].to(self.device)
        self.seg_real_B_gt = input['Seg'].to(self.device)
        self.image_paths = input['B_paths']

    def test(self):
        self.real_B = Variable(self.real_B)
        self.seg_real_B = self.netS.forward(self.real_B)

    def get_image_paths(self):
        return self.image_paths

    def get_current_visuals(self):
        real_B = util.tensor2im_test(self.real_B.data)

        seg_real_B = self.seg_real_B.data[:, -1:, :, :]
        seg_real_B[seg_real_B >= 0.5] = 1
        seg_real_B[seg_real_B < 0.5] = 0
        seg_real_B = util.tensor2seg_test(seg_real_B)

        seg_real_B_gt = self.seg_real_B_gt.data[:, -1:, :, :]
        seg_real_B_gt[seg_real_B_gt >= 0.5] = 1
        seg_real_B_gt[seg_real_B_gt < 0.5] = 0
        seg_real_B_gt = util.tensor2seg_test(seg_real_B_gt)
        # seg_real_B_gt = util.tensor2seg_test(torch.max(self.seg_real_B_gt.data, dim=1, keepdim=True)[1])

        return OrderedDict([('real_B', real_B), ('real_B_seg', seg_real_B), ('gt_real_B_seg', seg_real_B_gt)])
