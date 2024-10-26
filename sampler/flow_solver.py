import numpy as np
import torch

from .random_util import get_generator

def dynamic_thresholding_fn(x0_, t, ratio=0.995):
           
            def expand_dims(v, dims):
                return v[(...,) + (None,)*(dims - 1)]
            
            dims = x0_.dim()
            p = ratio
            s = torch.quantile(torch.abs(x0_).reshape((x0_.shape[0], -1)), p, dim=1)
            s = expand_dims(torch.maximum(s, 1.0 * torch.ones_like(s).to(s.device)), dims)
            x0_ = torch.clamp(x0_, -s, s) / s
            return x0_

def ot_sampler_3(new_net, x_start, model_kwargs, steps, beta, order, XT=None, is_latent=False):
        new_net.eval()
        traj = []
        traj.append(x_start)
        t0 = 0.01
        T = 1 - t0
        t_span = torch.linspace(T, t0, steps + 1).to(new_net.device)
        Qi_2 = Qi_1 = new_net(t_span[0], x_start, y=None, xT=XT)
        
        for i in range(1, steps+1):
            t_t = t_span[i]
            t_s = t_span[i-1]
            co1 = (1 - t_t) / (1 - t_s)
            if i == 1:
                x1 = Qi_2
                
                if not is_latent:
                    x1 = dynamic_thresholding_fn(x1, t_s)
                
                co1 = (1 - t_t) / (1 - t_s)
                x_start = co1 * x_start + x1 * (1. - co1)
            else:
                x1_1 = Qi_1
                x1_2 = Qi_2
                
                if not is_latent:
                    x1_1 = dynamic_thresholding_fn(x1_1, t_s)
                    x1_2 = dynamic_thresholding_fn(x1_2, t_s)
                
                Di = x1_1 + (x1_1 - x1_2) / (2. * 1)
                x_start = co1 * x_start + Di * (1. - co1)
                Qi_2 = Qi_1
            if i != steps:
                Qi_1 = new_net(t_t, x_start, y=None, xT=XT)
            traj.append(x_start)
        return torch.stack(traj)

def plms_b_mixer(old_eps, order=1, b=1):
    cur_order = min(order, len(old_eps))
    if cur_order == 1:
        eps_prime = b * old_eps[-1]
    elif cur_order == 2:
        eps_prime = ((2+b) * old_eps[-1] - (2-b)*old_eps[-2]) / 2
    elif cur_order == 3:
        eps_prime = ((18+5*b) * old_eps[-1] - (24-8*b) * old_eps[-2] + (6-1*b) * old_eps[-3]) / 12
    elif cur_order == 4:
        eps_prime = ((46+9*b) * old_eps[-1] - (78-19*b) * old_eps[-2] + (42-5*b) * old_eps[-3] - (10-b) * old_eps[-4]) / 24
    elif cur_order == 5:
        eps_prime = ((1650+251*b) * old_eps[-1] - (3420-646*b) * old_eps[-2] 
                     + (2880-264*b) * old_eps[-3] - (1380-106*b) * old_eps[-4]
                     + (270-19*b)* old_eps[-5]) / 720

    eps_prime = eps_prime / b
    if len(old_eps) >= order+1:
        old_eps.pop(0)
    return eps_prime



def linear_multistep_euler(new_net, x_start, model_kwargs, steps, beta=1, order=2, XT=None):

    new_net.eval()
    traj = []
    v_buf = []
    t0=0.01
    # t_span = torch.linspace(t0, 1-t0, steps+1).to(new_net.device)
    t_span = torch.linspace(1-t0, t0, steps+1).to(new_net.device) # reverse
    traj.append(x_start)
    vel = None
 
    for i in range(1, steps+1):

        Xn = traj[i - 1]
        t_n1 = t_span[i]
        t_n = t_span[i - 1]
        delta = t_n1 - t_n
        model_s = new_net(t_n, Xn, y=None, xT=XT)

        if vel is None:
            vel = model_s
        else:
            vel = (1 - beta) * vel + beta * model_s

        v_buf.append(vel)

        v_prime = plms_b_mixer(v_buf, order=order, b=beta)
    
        Xn1 = Xn + delta * v_prime
        traj.append(Xn1)

    return torch.stack(traj)