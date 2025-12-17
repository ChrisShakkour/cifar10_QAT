import torch as t
import copy
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import math

from .quantizer import Quantizer
from torch.autograd.function import InplaceFunction
from torch.autograd.variable import Variable
#from ...main import *
import sys
#sys.path.append('/home/gild/Lsq_with_gSTE')
#import .main

# This file contains the implementiation of diffrent quantization methods under LsqQuan the name LSQ is legacy and could be 
# described as the class containing the implementation of the Quantization methods and gSTE methods


class PWLSingleSTE(t.autograd.Function):
    @staticmethod
    def forward(ctx, v, s, a, Qn, Qp):
        ctx.save_for_backward(v, s, a)
        ctx.Qn, ctx.Qp = Qn, Qp
        return v

    @staticmethod
    def backward(ctx, grad_output):
        #print("hewrwr")
        v, s, a = ctx.saved_tensors
        Qn, Qp = ctx.Qn, ctx.Qp

        # u = v/s
        u = v.div(s)

        # ensure `a` can broadcast to `u`'s shape
        a_b = a
        while a_b.dim() < u.dim():
            a_b = a_b.unsqueeze(-1)

        # build the piecewise-linear slope φₐ(u)
        coeff = t.zeros_like(u)
        mask_n = (u >= -Qn / a_b) & (u <= 0)
        mask_p = (u >  0      ) & (u <=  Qp / a_b)

        coeff[mask_n] = 2 * a_b[mask_n] * (1 + (a_b[mask_n] / Qn) * u[mask_n].detach())
        coeff[mask_p] = 2 * a_b[mask_p] * (1 - (a_b[mask_p] / Qp) * u[mask_p].detach())

        # ∂L/∂v = grad_output * φₐ(u)
        grad_v = grad_output.detach() * coeff

        # no gradient w.r.t. s, a, Qn, or Qp
        return grad_v, None, None, None, None



class PWLSingleSTE_new(t.autograd.Function):
    @staticmethod
    def forward(ctx, v, s, a, Qn, Qp):
        ctx.save_for_backward(v, s, a)
        ctx.Qn, ctx.Qp = Qn, Qp
        return v

    @staticmethod
    def backward(ctx, grad_output):
        v, s, a = ctx.saved_tensors
        # print(">>> v.shape:", v.shape)
        # print(">>> u.shape:", (v/s).shape)
        # print(">>> a.shape:", a.shape)
        Qn, Qp = ctx.Qn, ctx.Qp
        u = v.div(s)
        # a is scalar tensor
        coeff = t.zeros_like(u)
        mask_n = (u >= -Qn/a) & (u <= 0)
        mask_p = (u > 0) & (u <= Qp/a)


        dc_da = t.zeros_like(u)

        # print(">>> (a/Qn).shape:", (a/Qn).shape)
        # print(">>> (v/s).shape:", (v/s).shape)
        # print(">>> mask_n.shape:", mask_n.shape)
        # print(">>> u.shape:", u.shape)
        # print(">>> (2*(1 + (a/Qn)*u)[mask_n]).shape.shape:", (2*(1 + (a/Qn)*u)[mask_n]).shape)
        # print(">>> (u/Qn).shape.shape:", (u/Qn).shape)
        # print(">>> (2*a*(u/Qn)).shape.shape:", (2*a*(u/Qn)).shape)
        # print(">>> mask_n.sum().item().shape.shape:", (mask_n.sum()).shape)
        #print(">>> (2*a*(u/Qn)[mask_n]).shape.shape:", (2*a*(u/Qn)[mask_n]).shape)
        full_term = 2 * a * (u / Qn)       # → shape [10,64]
        # now pick out only the masked entries, which is 1-D [377]
        dc_da[mask_n] = 2*(1 + (a/Qn)*u.detach())[mask_n] + full_term[mask_n.detach()]

        full_term = 2 * a * (u / Qp)       # → shape [10,64]

        dc_da[mask_p] = 2*(1 - (a/Qp)*u.detach())[mask_p.detach()] - full_term[mask_p.detach()]
        grad_a = (grad_output.detach() * dc_da)
        grad_v = grad_output * coeff
        # print(">>> maede it out:", grad_a.shape)
        return grad_v.detach(), None, grad_a, None, None




class PWLDualSTE(t.autograd.Function):
    @staticmethod
    def forward(ctx, v, s, a_n, a_p, Qn, Qp):
        ctx.save_for_backward(v, s, a_n, a_p)
        ctx.Qn, ctx.Qp = Qn, Qp
        return v

    @staticmethod
    def backward(ctx, grad_output):
        v, s, a_n, a_p = ctx.saved_tensors
        Qn, Qp = ctx.Qn, ctx.Qp
        u = v.div(s)

        # broadcast a_n, a_p to match v's rank
        a_n_b, a_p_b = a_n, a_p
        while a_n_b.dim() < u.dim(): a_n_b = a_n_b.unsqueeze(-1)
        while a_p_b.dim() < u.dim(): a_p_b = a_p_b.unsqueeze(-1)

        mask_n = (u >= -Qn/a_n_b) & (u <= 0)
        mask_p = (u >  0     ) & (u <=  Qp/a_p_b)
        
        # compute the STE slope φ(u) separately on each side
        coeff = t.zeros_like(u)
        coeff[mask_n] = 2 * a_n_b[mask_n] * (1 + (a_n_b[mask_n]/Qn) * u[mask_n].detach())
        coeff[mask_p] = 2 * a_p_b[mask_p] * (1 - (a_p_b[mask_p]/Qp) * u[mask_p].detach())

        # ∂L/∂v = grad_output · φ(u)
        grad_v = grad_output * coeff

        # we do NOT return gradients for a_n or a_p—they’re updated only by GDTUO
        return grad_v, None, None, None, None, None



class DoReFaPWLSTE(t.autograd.Function):
    @staticmethod
    def forward(ctx, x, a, Qn, Qp):
        """
        1-bit DoReFa forward → w_bin, plus stash v_for_phi for the PWL slope in backward.

        Args:
          w          (Tensor): the raw, full-precision weights (shape [⋯]).
          v_for_phi  (Tensor): = 2*u − 1, where u = tanh(w)/(2*alpha)+0.5  ∈ [−1,+1].
          a          (Tensor): the current PWL slope-parameter (same shape as w or broadcastable).
          Qn, Qp     (float):   left/right thresholds (for PWL), typically = −self.thd_neg, self.thd_pos.

        Returns:
          w_bin      (Tensor): the binarized weights ∈ {−1,+1}.  Grad hook is φₐ(v_for_phi).
        """
        # 1) compute scale = E[|w|]  (mean of absolute values over all elements of w)
        scale = x.abs().mean()
        # 2) w_sign = sign(w)  ∈ {-1, +1}
        w_sign = x.sign()
        # 3) w_bin = w_sign * scale
        w_bin = w_sign * scale

        ctx.save_for_backward(x, scale,a)
        ctx.Qn, ctx.Qp = 1, 1
        return w_bin 

    @staticmethod
    def backward(ctx, grad_output):
        v, scale, a = ctx.saved_tensors
        # print(">>> v.shape:", v.shape)
        # print(">>> u.shape:", (v/s).shape)
        # print(">>> a.shape:", a.shape)
        Qn, Qp = ctx.Qn, ctx.Qp#should be -1 and 1
        u = v.div(scale)
        # ensure `a` can broadcast to `u`'s shape
        a_b = a
        while a_b.dim() < u.dim():
            a_b = a_b.unsqueeze(-1)

        # build the piecewise-linear slope φₐ(u)
        coeff = t.zeros_like(u)
        mask_n = (u >= -Qn / a_b) & (u <= 0)
        mask_p = (u >  0      ) & (u <=  Qp / a_b)

        coeff[mask_n] = 2 * a_b[mask_n] * (1 + (a_b[mask_n] / Qn) * u[mask_n].detach())
        coeff[mask_p] = 2 * a_b[mask_p] * (1 - (a_b[mask_p] / Qp) * u[mask_p].detach())

        # ∂L/∂v = grad_output * φₐ(u)
        grad_v = grad_output.detach() * coeff

        # no gradient w.r.t. s, a, Qn, or Qp
        return grad_v, None, None, None, None


class DoReFaSTE(t.autograd.Function):
    @staticmethod
    def forward(ctx, x, a, Qn, Qp):
        """
        1-bit DoReFa forward → w_bin, plus stash v_for_phi for the PWL slope in backward.

        Args:
          w          (Tensor): the raw, full-precision weights (shape [⋯]).
          v_for_phi  (Tensor): = 2*u − 1, where u = tanh(w)/(2*alpha)+0.5  ∈ [−1,+1].
          a          (Tensor): the current PWL slope-parameter (same shape as w or broadcastable).
          Qn, Qp     (float):   left/right thresholds (for PWL), typically = −self.thd_neg, self.thd_pos.

        Returns:
          w_bin      (Tensor): the binarized weights ∈ {−1,+1}.  Grad hook is φₐ(v_for_phi).
        """
        # 1) compute scale = E[|w|]  (mean of absolute values over all elements of w)
        scale = x.abs().mean()
        #print("scale : ",scale)
        # 2) w_sign = sign(w)  ∈ {-1, +1}
        w_sign = x.sign()
        # 3) w_bin = w_sign * scale
        w_bin = w_sign * scale

        ctx.save_for_backward(x, scale,a)
        ctx.Qn, ctx.Qp = -1, 1
        return w_bin 

    @staticmethod
    def backward(ctx, grad_output):
        """
        Straight‐through: ∂L/∂x = ∂L/∂w_bin (identity),
        no gradients to a, Qn or Qp.
        """
        v, scale, a = ctx.saved_tensors
        
        Qn, Qp = ctx.Qn, ctx.Qp#should be -1 and 1
        # pass the incoming gradient unchanged back to x
        clip_varient=True
        if clip_varient==True:
            # build mask of inputs inside the quant range
            mask = (v >= -scale/a) & (v <= scale/a)
            # pass gradient through only where mask is True
            grad_x = grad_output.detach() * mask.to(grad_output.dtype).detach()*a
        else:
            grad_x = grad_output.detach()*a
        # we drop grads for a, Qn, Qp
        return grad_x, None, None, None, None

class DoReFaPWLDual(t.autograd.Function):
    @staticmethod
    def forward(ctx, x, a_n, a_p, Qn, Qp):
        # 1) compute scale = E[|w|]  (mean of absolute values over all elements of w)
        scale = x.abs().mean()
        # 2) w_sign = sign(w)  ∈ {-1, +1}
        w_sign = x.sign()
        # 3) w_bin = w_sign * scale
        w_bin = w_sign * scale

        ctx.save_for_backward(x, scale, a_n, a_p)
        ctx.Qn, ctx.Qp = 1, 1
        return w_bin 
        
    @staticmethod
    def backward(ctx, grad_output):
        x, scale, a_n, a_p = ctx.saved_tensors
        Qn, Qp = ctx.Qn, ctx.Qp
        u = (x.div(scale)).detach()
        #print(" a_p is ",a_p)
        # broadcast a_n, a_p to match v's rank
        a_n_b, a_p_b = a_n, a_p
        while a_n_b.dim() < u.dim(): a_n_b = a_n_b.unsqueeze(-1)
        while a_p_b.dim() < u.dim(): a_p_b = a_p_b.unsqueeze(-1)

        mask_n = ((u >= -Qn/a_n_b) & (u <= 0)).detach()
        mask_p = ((u >  0     ) & (u <=  Qp/a_p_b)).detach()
        
        # compute the STE slope φ(u) separately on each side
        coeff = t.zeros_like(u)
        coeff[mask_n] = 2 * a_n_b[mask_n] * (1 + (a_n_b[mask_n]/Qn) * u[mask_n].detach())
        coeff[mask_p] = 2 * a_p_b[mask_p] * (1 - (a_p_b[mask_p]/Qp) * u[mask_p].detach())
        #coeff[mask_p] = -2 * a_p_b[mask_p].squared()*u[mask_p].detach() + 2*a_p_b[mask_p]

        # ∂L/∂v = grad_output · φ(u)
        grad_v = grad_output.detach() * coeff

        temp1 = t.zeros_like(u)
        #print("mask_n.sum()",mask_n.sum())
        #print("mask_p.sum()",mask_p.sum())

        temp1[mask_n]=2 * a_n_b[mask_n] * (1 + (a_n_b[mask_n]/Qn) * u[mask_n].detach())
        temp2 = t.zeros_like(u)
        temp2[mask_p] = 2 * a_p_b[mask_p] * (1 - (a_p_b[mask_p]/Qp) * u[mask_p].detach())
        # we do NOT return gradients for a_n or a_p—they’re updated only by GDTUO
        return grad_v, None, None, None, None, None
        #return temp1[mask_n]*grad_output.detach()+temp2[mask_p]*grad_output.detach()
        #return (grad_v.detach()-grad_output.detach()*temp2.detach()).detach()+grad_output.detach()*temp2, None, None, None, None, None


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad


count_inst=0


class split_grad(InplaceFunction):

    @staticmethod
    def forward(ctx, x , x_prev,v_hat):

        return x

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output == None:
            print (" None here")
        return grad_output ,grad_output,grad_output



class Calc_grad_a_STE(InplaceFunction):

    @staticmethod
    def forward(ctx, x , a,name):
        ctx.save_for_backward(a)
        ctx.name=name
        #ctx.name=name
        return x

    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        return a*grad_output.detach(), None, None


class Calc_grad_a_STE_dupl(InplaceFunction):

    @staticmethod
    def forward(ctx, x ,x_hat, a,name):
        ctx.save_for_backward(a)
        ctx.name=name
        #ctx.name=name
        return x

    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        return a*grad_output.detach(),a*grad_output, None, None


class Clamp_STE(InplaceFunction):

    @staticmethod
    def forward(ctx, i,x_parallel, min_val, max_val,a):
        #ctx._mask1 = (i.ge(min_val/a) * i.le(max_val/a))
        ctx.shapea=a.shape
        maxdiva = max_val / a
        mindiva = min_val / a
        ctx._mask1 = i.ge(mindiva) * i.le(maxdiva)
        ctx._mask2 = (x_parallel.ge(min_val) * x_parallel.le(max_val))
        return i.clamp(min_val, max_val)

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output == None:
            print (" None here")
        mask1 = Variable(ctx._mask1.type_as(grad_output.data))

        mask2 = Variable(ctx._mask2.type_as(grad_output.data))
        return grad_output * mask1,grad_output * mask2, None, None,t.zeros(ctx.shapea).cuda()


class LsqQuan(Quantizer):#Name is legacy look at the comment in the beggining of the file
    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=True):
        super().__init__(bit)
        global count_inst
        self.name = count_inst
        count_inst += 1
        self.is_weight=False
        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            #self.is_weight =True
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1
        isBinary=False
        if isBinary:
            self.thd_neg = 0
            self.thd_pos = 1

        self.per_channel = per_channel
        self.x_hat=t.nn.Parameter(t.ones(1))

        self.s = t.nn.Parameter(t.ones(1))
        self.a = t.nn.Parameter(t.ones(1))
        self.a_p = nn.Parameter(t.ones(1))   # placeholder for positive-side slope
        self.v_hat = t.nn.Parameter(t.ones(1))
        self.num_solution=0
        self.T=0
        self.counter=0
        self.meta_modules_STE_const = t.nn.Parameter(t.zeros(1))
        self.set_use_last_a_trained=False
    def update_strname(self,strname):
        self.strname=strname
    def use_last_a_trained(self):
        self.set_use_last_a_trained=True
    def update_list_for_lsq(self,num_solution,list_for_lsq):
        self.num_solution=num_solution
        self.T=list_for_lsq[0]
        self.a_per=list_for_lsq[1]
        self.num_share_params=list_for_lsq[2]

    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            self.s = t.nn.Parameter(
                x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / (self.thd_pos ** 0.5))
            #print("check dim : ",x.size()," check size : ",x.size())
            self.is_weight=True
            self.a=t.nn.Parameter(t.ones(x.size()))
            self.x_hat=t.nn.Parameter(t.ones(x.size()))
            self.v_hat = t.nn.Parameter(t.ones(x.size()))
            self.num_share_params=1
            if self.num_solution == 2 or self.num_solution==8 or self.num_solution==9 or self.num_solution==10 or self.num_solution==11 or self.num_solution == 12:
                
                if self.a_per == 0:#a per element
                    my_list = [i for i in x.size()]
                if self.a_per == 1:#a per layer
                    my_list = [1]
                if self.a_per == 2:#a per channel
                    my_list = [i for i in x.size()]
                    my_list = [my_list[0]] + [1] * (len(my_list) - 1)

                my_list=[math.ceil(self.T/self.num_share_params)]+my_list
                my_tupel=tuple(my_list)
                #self.a=t.nn.Parameter(t.ones(my_tupel,dtype=t.float16))
                self.a=t.nn.Parameter(t.ones(my_tupel))
                print("shape self.a : ",self.a.shape,"shape self.x_hat : ",self.x_hat.shape)

                self.a_p = nn.Parameter(t.ones(my_tupel, dtype=x.dtype, device=x.device))

                self.num_segments = self.T // self.num_share_params


            if self.num_solution == 7:
                meta_copy=copy.deepcopy(self.meta_network)
                
                a_list = t.nn.ModuleList()
                for i in range(self.T):
                    a_list.append(meta_copy)
                    meta_copy=copy.deepcopy(self.meta_network)
                    
                self.meta_modules = a_list

                self.meta_modules_STE_const = t.nn.Parameter(t.zeros(self.T))

        else:
            self.s = t.nn.Parameter(x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5))
            self.a = t.nn.Parameter(t.ones(1))
            self.v_hat = t.nn.Parameter(t.ones(1))

    def forward_original(self, x):#original forward from lsq
        if self.per_channel:
            if self.thd_pos !=0:
                s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
            else:
                s_grad_scale=1.0
        else:
            if self.thd_pos !=0:
                s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
            else:
                s_grad_scale=1.0
        s_scale = grad_scale(self.s, s_grad_scale)
        if self.is_weight:
            x = x / s_scale
            x = t.clamp(x, self.thd_neg, self.thd_pos)
            x = round_pass(x)
            x = x * s_scale
        return x

    def forward_no_quantization(self, x):#no quantization
        return x


    def forward_all_times_original(self, x):#original forward from lsq
        if self.per_channel:
            if self.thd_pos !=0:
                s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
            else:
                s_grad_scale=1.0
        else:
            if self.thd_pos !=0:
                s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
            else:
                s_grad_scale=1.0

        s_scale = grad_scale(self.s, s_grad_scale)
        
        if self.is_weight:
            x_parallel = x.detach()
            x_parallel = x_parallel / s_scale
            xdivs_save=x_parallel.detach()
            x_prev=x.detach()
            use_ste_end=False
            
            if self.set_use_last_a_trained:
                x = Calc_grad_a_STE.apply(x,self.a[-2],self.name)
            else:
                if int(math.floor(self.counter/self.num_share_params)%(self.T/self.num_share_params)) == int((self.T/self.num_share_params)-1) and use_ste_end==False:
                    x = Calc_grad_a_STE.apply(x,self.a[int(math.floor(self.counter/self.num_share_params-1)%(self.T/self.num_share_params))],self.name)
                else:
                    x = Calc_grad_a_STE.apply(x,self.a[int(math.floor(self.counter/self.num_share_params)%(self.T/self.num_share_params))],self.name)
            

            x = x / s_scale.detach()
            if self.num_solution == 7:
                x = t.clamp(x, self.thd_neg, self.thd_pos)
            else:
                x = Clamp_STE.apply(x,x_parallel, self.thd_neg, self.thd_pos,self.a[int(math.floor(self.counter/self.num_share_params)%(self.T/self.num_share_params))])
            x = round_pass(x)

            #print("bef",x)
            x = x * s_scale
            x = split_grad.apply(x,x_prev,self.v_hat)
            
            self.counter+=1
            #print("aft",x)
       
        return x
    
    def forward_all_times_wrong(self, x, use_pwl=False):

        use_pwl = True
        self.ste_type = 'Single'

        # 1) compute grad-scale factor & wrap s
        if self.per_channel:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5) if self.thd_pos != 0 else 1.0
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5) if self.thd_pos != 0 else 1.0
        s_scale = grad_scale(self.s, s_grad_scale)

        if self.is_weight:
            x_parallel = x.detach() / s_scale
            x_prev     = x.detach()

            # pick which segment a[idx] to use
            idx = (self.counter // self.num_share_params) % self.num_segments

            # 2) PWL STE forward with raw self.s
            if use_pwl:
                if self.ste_type == 'single':
                    x = PWLSingleSTE.apply(x, self.s,    self.a[idx],
                                           -self.thd_neg, self.thd_pos)
                else:
                    x = PWLDualSTE.apply(x,   self.s,
                                         self.a[idx], self.a_p[idx],
                                        -self.thd_neg, self.thd_pos)
            else:
                # your original gSTE logic unchanged
                if self.set_use_last_a_trained:
                    a_val = self.a[-1]
                else:
                    a_val = self.a[idx]
                x = Calc_grad_a_STE.apply(x, a_val, self.name)

            # 3) quantize & clamp via plain torch.clamp
            x = x / s_scale.detach()
            x = t.clamp(x, -self.thd_neg, self.thd_pos)
            x = round_pass(x)

            # 4) restore scale & split grad
            x = x * s_scale
            x = split_grad.apply(x, x_prev, self.v_hat)

            self.counter += 1

        return x


    def forward_all_times(self, x, use_pwl=False):
        use_pwl=True
        # don't force PWL—use the caller's flag
        self.ste_type = 'double'

        # 1) grad-scale factor & wrap s
        s_grad_scale = 1.0 / ((self.thd_pos * x.numel())**0.5) if self.thd_pos != 0 else 1.0
        s_scale = grad_scale(self.s, s_grad_scale)

        if not self.is_weight:
            return x

        # detach a parallel copy for Clamp_STE later
        x_parallel = x.detach() / s_scale
        x_prev     = x.detach()

        # pick segment index
        idx = (self.counter // self.num_share_params) % self.num_segments

        # 2) choose STE
        if use_pwl:
            # PWL‐STE (your fixed version)
            x = PWLSingleSTE.apply(
                    x, self.s, self.a[idx],
                -self.thd_neg, self.thd_pos
                )
        else:
            # original GSTE logic
            if self.set_use_last_a_trained:
                # use next-to-last a
                pick = -2
            else:
                # if we're at the very last segment, use the previous one
                last_idx = self.num_segments - 1
                if idx == last_idx:
                    pick = last_idx - 1
                else:
                    pick = idx

            x = Calc_grad_a_STE.apply(x, self.a[pick], self.name)

        # 3) quantize & clamp
        x = x / s_scale.detach()
        if self.num_solution == 7:
            x = t.clamp(x, self.thd_neg, self.thd_pos)
        else:
            x = Clamp_STE.apply(
                x, x_parallel,
                self.thd_neg, self.thd_pos,
                self.a[idx]
            )
        x = round_pass(x)

        # 4) restore scale & split grad
        x = x * s_scale
        x = split_grad.apply(x, x_prev, self.v_hat)

        self.counter += 1
        return x

    def forward_dorefa_1bit_PWL(self, x):#Not working properly
        """
        1) Perform DoReFa 1-bit forward quantization on the raw weights `x`.
        2) Then hand the binarized result into DoReFaPWLSTE for backward, passing along
        the helper tensor v_for_phi = 2*u − 1 so that PWL is applied to that in backward.
        """
        if not self.is_weight:
            return x
        
        idx = (self.counter // self.num_share_params) % self.num_segments

            # Use “sign(w) * mean(abs(w))” forward, then PWL-STE backward:
        x = DoReFaPWLSTE.apply(
                x,             # raw weights w
                self.a[idx],   # PWL slope parameter a for this segment
            -self.thd_neg,  # Qn  (we only use −1 internally)
                self.thd_pos   # Qp  (we only use +1 internally)
            )
        self.counter += 1
        return x


    def forward_dorefa_1bit_STE(self, x):#Not working properly
        """
        1) Perform DoReFa 1-bit forward quantization on the raw weights `x`.
        2) Then hand the binarized result into DoReFaPWLSTE for backward, passing along
        the helper tensor v_for_phi = 2*u − 1 so that PWL is applied to that in backward.
        """
        if not self.is_weight:
            return x
        
        idx = (self.counter // self.num_share_params) % self.num_segments

            # Use “sign(w) * mean(abs(w))” forward, then PWL-STE backward:
        x = DoReFaSTE.apply(
                x,             # raw weights w
                self.a[idx],   # PWL slope parameter a for this segment
            -self.thd_neg,  # Qn  (we only use −1 internally)
                self.thd_pos   # Qp  (we only use +1 internally)
            )
        self.counter += 1
        return x

    def forward_dorefa_1bit_DualPWL(self, x):#Not working properly
        """
        1) Perform DoReFa 1-bit forward quantization on the raw weights `x`.
        2) Then hand the binarized result into DoReFaPWLSTE for backward, passing along
        the helper tensor v_for_phi = 2*u − 1 so that PWL is applied to that in backward.
        """
        if not self.is_weight:
            return x
        
        idx = (self.counter // self.num_share_params) % self.num_segments
        
        # x = DoReFaPWLSTE.apply(
        #         x,             # raw weights w
        #         self.a[idx],   # PWL slope parameter a for this segment
        #     -self.thd_neg,  # Qn  (we only use −1 internally)
        #         self.thd_pos   # Qp  (we only use +1 internally)
        #     )

            # Use “sign(w) * mean(abs(w))” forward, then PWL-STE backward:
        x = DoReFaPWLDual.apply(
                x,             # raw weights w
                self.a[idx],   # PWL slope parameter a for this segment
                self.a_p[idx],
            -self.thd_neg,  # Qn  (we only use −1 internally)
                self.thd_pos   # Qp  (we only use +1 internally)
            )
        self.counter += 1
        return x


    def forward_less_greedy_Updates(self, x):#original forward from lsq
        if self.per_channel:
            if self.thd_pos !=0:
                s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
            else:
                s_grad_scale=1.0
        else:
            if self.thd_pos !=0:
                s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
            else:
                s_grad_scale=1.0

        s_scale = grad_scale(self.s, s_grad_scale)
        
        if self.is_weight:
            x_parallel = x.detach()
            x_parallel = x_parallel / s_scale
            xdivs_save=x_parallel.detach()
            x_prev=x.detach()
            use_ste_end=True
            
            if self.set_use_last_a_trained:
                x = Calc_grad_a_STE_dupl.apply(x,self.a[-2],self.name)
            else:
                if int(math.floor(self.counter/self.num_share_params)%(self.T/self.num_share_params)) == int((self.T/self.num_share_params)-1) and use_ste_end==False:
                    x = Calc_grad_a_STE_dupl.apply(x,self.x_hat,self.a[int(math.floor(self.counter/self.num_share_params-1)%(self.T/self.num_share_params))],self.name)
                else:
                    x = Calc_grad_a_STE_dupl.apply(x,self.x_hat,self.a[int(math.floor(self.counter/self.num_share_params)%(self.T/self.num_share_params))],self.name)
            

            x = x / s_scale.detach()
            if self.num_solution == 7:
                x = t.clamp(x, self.thd_neg, self.thd_pos)
            else:
                x = Clamp_STE.apply(x,x_parallel, self.thd_neg, self.thd_pos,self.a[int(math.floor(self.counter/self.num_share_params)%(self.T/self.num_share_params))])
            x = round_pass(x)

            #print("bef",x)
            x = x * s_scale
            x = split_grad.apply(x,x_prev,self.v_hat)
            
            self.counter+=1
            #print("aft",x)

        return x
    

    def forward_baseline_no_quant(self, x):#original forward from lsq
        if self.per_channel:
            if self.thd_pos !=0:
                s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
            else:
                s_grad_scale=1.0
        else:
            if self.thd_pos !=0:
                s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
            else:
                s_grad_scale=1.0
        s_scale = grad_scale(self.s, s_grad_scale)
        if t.equal(t.ones(x.size()).to(device='cuda'),self.x_hat):
            x_original=x
            x = x / s_scale
            x = t.clamp(x, self.thd_neg, self.thd_pos)
            x = round_pass(x)
            
            x = x * s_scale
            self.x_hat.data =(x.detach())
            if self.is_weight:
                return self.x_hat
            else:
                return x_original
        else:
            if self.is_weight:
                return self.x_hat
            else:
                return x

        

    def forward(self, x):
        if self.num_solution == -1:
            return self.forward_original( x)
        
        #elif self.num_solution == 0 or self.num_solution == 5:
        #    return self.forward_analytical_gSTE( x)
        #elif self.num_solution == 1:
        #    return self.forward_delayed_updates( x)
        #elif self.num_solution == 1.5 or self.num_solution == 6:
        #    return self.forward_delayed_updates_meta_quant(x)
        elif self.num_solution == 2 or self.num_solution == 7 or self.num_solution == 10 or self.num_solution == 11:
            #print("herererer")
            regular_all_times_together_lsq_sgste=False
            if regular_all_times_together_lsq_sgste:
                return self.forward_all_times_original(x)
                
            # DoReFa with PWL/Dual PWL/STE
            dorefa_alltimes =True
            if dorefa_alltimes==True:
                dual_dorefa=False
                if dual_dorefa == True:
                    return self.forward_dorefa_1bit_DualPWL(x)
                else:
                    return self.forward_dorefa_1bit_PWL(x) # Piece-wise linear STE on dorefa binary weights
            else:
                ste_binary_dorefa=True
                if ste_binary_dorefa==True:
                    return self.forward_dorefa_1bit_STE(x) #regular STE on dorefa binary weights
                else:
                    return self.forward_all_times( x)
        elif self.num_solution == 8:
            return self.forward_baseline_no_quant( x)
        #elif self.num_solution == 9:
        #    return self.forward_all_times_for_MAD( x)
        elif self.num_solution == 12:
            return self.forward_less_greedy_Updates( x)
        else:
            print("Solution not defined")
        




    