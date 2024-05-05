import matplotlib.pyplot as plt
import torch
import torch as th
import torch.distributions as td
import torch.nn.functional as F
from torch import nn  # , einsum, rearrange
import numpy as np

from model.set_diffusion.gaussian_diffusion import GaussianDiffusion
from model.set_diffusion.nn import SiLU, timestep_embedding
from model.set_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from model.set_diffusion.unet import EncoderUNetModel, UNetModel
from model.set_diffusion.nn import mean_flat

from memory_conv_patch import *
def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)
class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)

class DDPM(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.bs = args.batch_size
        self.ch = args.in_channels
        self.image_size = args.image_size

        self.generative_model, self.diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )

        print(
            "generative model parameters:", self.count_parameters(self.generative_model)
        )

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def forward(self, x_set, t):
        x = x_set.view(-1, self.ch, self.image_size, self.image_size)
        loss = self.diffusion.training_losses(self.generative_model, x, t, None)
        return loss


def log(t, eps=1e-20):
    return torch.log(t + eps)


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1., dim=-1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim=dim)


def top_k(logits, thres=0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs


class VFSDDPM(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.ns = args.sample_size
        self.bs = args.batch_size
        self.patch_size = args.patch_size
        self.image_size = args.image_size
        self.encoder_mode = args.encoder_mode
        self.hdim = args.hdim
        self.mode_conditioning = args.mode_conditioning

        self.mode_context = args.mode_context
        self.memory = Memory(args.hdim)
        self.postpool = nn.Sequential(
            SiLU(),
            linear(
                self.hdim*2,
                self.hdim,
            ),
        )
        if self.encoder_mode == "unet":
            # load classifier
            self.encoder = EncoderUNetModel(
                image_size=args.image_size,
                in_channels=args.in_channels,
                model_channels=args.num_channels,
                out_channels=args.hdim,
                num_res_blocks=2,
                dropout=args.dropout,
                num_head_channels=64,
            )

        
        self.generative_model, self.diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )

        print("encoder parameters:", self.count_parameters(self.encoder))
        print(
            "generative model parameters:", self.count_parameters(self.generative_model)
        )

        # assert self.mode_context == "variational" and self.mode_conditioning != "lag", 'Use mean, cls, sum with variational.'
        # assert self.mode_context == "variational_discrete" and self.mode_conditioning == "lag", 'Use lag with variational_discrete.'
        # assert self.mode_conditioning == "lag" and self.encoder_mode == "vit", "Use vit with lag."

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def forward_c(self, x_set, label=None, t=None, tag='test'):
        """
        Process input set X and return aggregated representation c for the set.
        c can be deterministic or stochastic. c can be obtained using patches (vit) or images (unet)
        """
        bs, ns, ch, h, w = x_set.size()
        # straightforward conditional DDPM
        # we encode the images in the sets
        # and use the mean as conditioning for DDPM
        # each image here is an element in a set
        if self.encoder_mode == "unet":
            # image level aggregation
            x = x_set.view(-1, ch, h, w)
            out = self.encoder.forward(x, t, self.bs)
            out = out.view(bs, ns, -1)
            hc = out.mean(1)
            ##structure_memory

            if tag == 'train':
                new_memory = self.memory.memory_update(hc, hc, label, 0.3)
                self.memory.memory_add(label, new_memory.to(torch.float32), 0.3)
                # self.memory.update(hc)
            
            mc = self.memory(hc, label, tag)
            hc = torch.cat((mc, hc), dim=-1)
            hc = self.postpool(hc)

       
        return {"c": hc}

    def normal(self, loc: torch.Tensor, log_var: torch.Tensor, temperature=None):
        log_std = log_var / 2
        # if temperature:
        #     log_std = log_std * temperature
        scale = torch.exp(log_std)
        distro = td.Normal(loc=loc, scale=scale)
        return distro

    def forward(self, batch, label, t, tag='test'):
        """
        forward input set X, compute c and condition ddpm on c.
        """
        bs, ns, ch, h, w = batch.size()

        c_list = []
        for i in range(batch.shape[1]):
            ix = torch.LongTensor([k for k in range(batch.shape[1]) if k != i])
            x_set_tmp = batch[:, ix]

            out = self.forward_c(x_set_tmp, label, t, tag)
            c_set_tmp = out["c"]
            c_list.append(c_set_tmp.unsqueeze(1))

        c_set = torch.cat(c_list, dim=1)

       
        x = batch.view(-1, ch, self.image_size, self.image_size)

        if self.mode_conditioning == "lag":
            # (b*ns, np, dim)
            c = c_set.view(-1, c_set.size(-2), c_set.size(-1))
        else:
            # (b*ns, dim)
            c = c_set.view(-1, c_set.size(-1))
        # forward and denoising process
        losses = self.diffusion.training_losses(self.generative_model, x, t, c)
        # losses["klm"] = self.loss_m(out)

        

    def sample_conditional(self, x_set, sample_size, k=1):
        out = self.forward_c(x_set, None)  # improve with per-layer conditioning using t

        if self.mode_context == "deterministic":
            c_set = out["c"]
        

        # if we want more than sample_size samples, increase here
        # c_set = c_set.unsqueeze(1) # attention here
        # c_set = torch.repeat_interleave(c_set, k * sample_size, dim=1)

        if self.mode_conditioning == "lag":
            # (b*ns, np, dim)
            c = c_set.view(-1, c_set.size(-2), c_set.size(-1))
        else:
            # (b*ns, dim)
            c = c_set.view(-1, c_set.size(-1))

        

        return {"c": c}


if __name__ == "__main__":
    # attention, adaptive, spatial
    # model = EncoderUNetModel(image_size=64, pool='adaptive')
    # x = torch.randn(12, 5, 3, 64, 64)
    # x = x.view(-1, 3, 64, 64)

    # out = model.forward(x)
    # print(out.size())
    # out = out.view(12, 5, -1).mean(1)
    # print(out.size())

    model = UNetModel(
        image_size=64,
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions="16,8",
    )

    x = torch.randn(12, 5, 3, 64, 64)
    x = x.view(-1, 3, 64, 64)

    out = model.forward(x)
    print(out.size())