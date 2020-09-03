from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

class MFGSM():
    def __init__(self, model, max_val, min_val,loss,device,epsilon,margin_loss=1.2):
        self.model = model
        # Minimum value of the pixels
        self.min_val = min_val
        # Maximum value of the pixels
        self.max_val = max_val
        # loss function
        self.loss = loss
        # device | cpu or gpu
        self.device = device
        self.epsilon = epsilon
        self.margin_loss = margin_loss

    def perturb(self,X,y):
        X_adv = Variable(X.clone().detach().data, requires_grad=True)
        opt = optim.SGD([X_adv], lr=1e-3)
        opt.zero_grad()
        with torch.enable_grad():
            base_loss = self.loss(self.model(X_adv), y)
            print("loss:{}".format(base_loss))
            loss = torch.abs(base_loss - self.margin_loss) + self.margin_loss

        loss.backward()
        eta = self.epsilon * X_adv.grad.data.sign()

        X_adv = Variable(X_adv.data + eta, requires_grad=True)
        eta = torch.clamp(X_adv.data - X.data, -self.epsilon, self.epsilon)
        X_adv = Variable(X.data + eta, requires_grad=True)
        X_adv = Variable(torch.clamp(X_adv, self.min_val, self.max_val), requires_grad=True)
        return X_adv.clone().detach()
