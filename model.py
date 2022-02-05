import math
import logging
import numpy as np
import math

import torch
from torch import nn
import torch.utils.data
from torch.nn import functional as F
import torchvision.models as models

from VAE.layers import GatedDense
from utils import Reshape

def Xavier(m):
    if m.__class__.__name__ == 'Linear':
        fan_in, fan_out = m.weight.data.size(1), m.weight.data.size(0)
        std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        m.weight.data.uniform_(-a, a)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

# Generative Models
# -----------------------------------------------------------------------------------

# We will use it as an autoencoder for now
class CVAE(nn.Module):
    def __init__(self, d, args, **kwargs):
        super(CVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),

            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),

            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),

            #nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1, bias=False),
            #nn.BatchNorm2d(d),
            #nn.ReLU(inplace=True),

            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
        )
        self.decoder = nn.Sequential(
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),

            nn.ConvTranspose2d(d, d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),

            #nn.ConvTranspose2d(d, d, kernel_size=4, stride=2, padding=1, bias=False),
            #nn.BatchNorm2d(d),
            #nn.ReLU(inplace=True),

            nn.ConvTranspose2d(d, d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d, 3, kernel_size=4, stride=2, padding=1, bias=False),
        )
        self.f = 4 #8
        self.d = d

    def encode(self, x):
        h1 = self.encoder(x)
        h1 = h1.view(-1, self.d * self.f ** 2)
        return h1
        #return self.fc11(h1), self.fc12(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        z = z.view(-1, self.d, self.f, self.f)
        h3 = self.decoder(z)
        return torch.tanh(h3)

    def forward(self, x):
        #mu, logvar = self.encode(x)
        #z = self.reparameterize(mu, logvar)
        #return self.decode(z), mu, logvar
        hid = self.encode(x)
        return self.decode(hid), hid

    def sample(self, size):
        sample = Variable(torch.randn(size, self.d * self.f ** 2), requires_grad=False)
        if self.cuda():
            sample = sample.cuda()
        return self.decode(sample).cpu()

    def loss_function(self, x, recon_x, mu, logvar):
        self.mse = F.mse_loss(recon_x, x)
        batch_size = x.size(0)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        self.kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Normalise by same number of elements as in reconstruction
        self.kl_loss /= batch_size * 3 * 1024

        # return mse
        return self.mse + self.kl_coef * self.kl_loss

class ResBlock(nn.Module):
    def __init__(self, in_channels, channels, bn=False):
        super(ResBlock, self).__init__()

        layers = [
            nn.ReLU(),
            nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0)]
        if bn:
            layers.insert(2, nn.BatchNorm2d(channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)



# Classifiers
# -----------------------------------------------------------------------------------

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CategoricalConditionalBatchNorm(torch.nn.Module):
    # as in the chainer SN-GAN implementation, we keep per-cat weight and bias
    def __init__(self, num_features, num_cats, eps=2e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.num_cats = num_cats
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = torch.nn.Parameter(torch.Tensor(num_cats, num_features))
            self.bias = torch.nn.Parameter(torch.Tensor(num_cats, num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.fill_(1.0)
            self.bias.data.zero_()

    def forward(self, input, cats):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        out = torch.nn.functional.batch_norm(
            input, self.running_mean, self.running_var, None, None,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)
        if self.affine:
            shape = [input.size(0), self.num_features] + (input.dim() - 2) * [1]
            #weight = self.weight.index_select(0, cats).view(shape)
            weight  = self.weight[cats].view(1, -1, 1, 1)#.expand_as(shape)
            bias    = self.bias[cats].view(1, -1, 1, 1)
            #bias = self.bias.index_select(0, cats).view(shape)
            out = out * weight + bias
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf, input_size):
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.input_size = input_size

        self.conv1 = conv3x3(input_size[0], nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        #self.bn1  = CategoricalConditionalBatchNorm(nf, 2)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

        # hardcoded for now
        last_hid = nf * 8 * block.expansion if input_size[1] in [8,16,21,32,42] else 640
        self.linear = nn.Linear(last_hid, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def return_hidden(self, x):
        bsz = x.size(0)
        #pre_bn = self.conv1(x.view(bsz, 3, 32, 32))
        #post_bn = self.bn1(pre_bn, 1 if is_real else 0)
        #out = F.relu(post_bn)
        out = F.relu(self.bn1(self.conv1(x.view(bsz, *self.input_size))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def forward(self, x):
        out = self.return_hidden(x)
        out = self.linear(out)
        return out

def ResNet18(nclasses, nf=20, input_size=(3, 32, 32)):
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, input_size)

class MLP(nn.Module):
    def __init__(self, args, num_classes=10, nf=100):
        super(MLP, self).__init__()

        self.input_size = np.prod(args.input_size)
        self.hidden = nn.Sequential(nn.Linear(self.input_size, nf),
                                    nn.ReLU(True),
                                    nn.Dropout(args.dropout),
                                    nn.Linear(nf, nf),
                                    nn.ReLU(True),
                                    nn.Dropout(args.dropout))

        self.linear = nn.Linear(nf, num_classes)

    def return_hidden(self,x):
        x = x.view(-1, self.input_size)
        return self.hidden(x)

    def forward(self, x):
        out = self.return_hidden(x)
        return self.linear(out)

# class MLP(nn.Module):
#   def __init__(self, n_input, n_feature, n_output, args, bias=True):
#     super(MLP, self).__init__()
#     self.device = None

#     self.hidden = nn.Sequential(nn.Linear(n_input, 100),
#                                 nn.ReLU(True),
#                                 nn.Dropout(args.dropout),
#                                 nn.Linear(100, n_feature),
#                                 nn.ReLU(True),
#                                 nn.Dropout(args.dropout))
#     self.linear = nn.Linear(n_feature, n_output, bias=bias)
#     self.hidden.apply(Xavier)
  
#   def forward(self, samples):
#     x = samples.view(samples.size(0), -1)
#     features = self.hidden(x)
#     outputs = self.linear(features)
#     return outputs

class Conv_4(nn.Module):
    def __init__(self, args):
        super(Conv_4, self).__init__()
        
        img_channels = 1	  	# 1
        self.last_layer = 1 	# 3 for 3-layers - 1 for 4-layers
        # if args.dataset in ['mnist', 'rmnist', 'fmnist', 'pfmnist', 'rfmnist']:
        #     img_channels = 1	  	# 1
        #     self.last_layer = 1 	# 3 for 3-layers - 1 for 4-layers
        # elif args.dataset in ['cifar10', 'cifar100']:
        #     img_channels = 3	  	# 3 
        #     self.last_layer = 2 	# 4 for 3-layers - 2 for 4-layers

        

        self.filters_length = 256    # 128 for 3-layers - 256 for 4-layers

        self.layer1 = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=5, padding=2), #input: 28 * 28 * 3, output: 28 * 28 * 32
            # nn.ReLU(),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=5, padding=2), #input: 28 * 28 * 3, output: 28 * 28 * 32
            nn.BatchNorm2d(32),
            nn.PReLU(),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),   #input: 28 * 28 * 32, output: 14 * 14 * 32
            nn.Dropout(args.dropout)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2), #input: 14 * 14 * 32, output: 14 * 14 * 64
            nn.PReLU(),
            # nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding=2), #input: 14 * 14 * 64, output: 14 * 14 * 64
            nn.BatchNorm2d(64),
            nn.PReLU(),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),   #input: 14 * 14 * 64, output: 7* 7 * 64
            nn.Dropout(args.dropout)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), #input: 7 * 7 * 64, output: 7 * 7 * 128
            nn.PReLU(),
            # nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), #input: 7 * 7 * 128, output: 7 * 7 * 128
            nn.BatchNorm2d(128),
            nn.PReLU(),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),   #input: 7 * 7 * 128, output: 3* 3 * 128
            nn.Dropout(args.dropout)
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), #input: 3 * 3 * 128, output: 3 * 3 * 256
            nn.PReLU(),
            # nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), #input: 3*3*256, output: 3*3*256
            nn.BatchNorm2d(256),
            nn.PReLU(),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),   #input: 3*3*256, output: 1*1*256
            nn.Dropout(args.dropout)
        )

        self.ip1 = nn.Linear(self.filters_length*self.last_layer*self.last_layer, args.hidden_dims)
        self.preluip1 = nn.PReLU()
        self.dropoutip1 = nn.Dropout(args.dropout)
        self.linear = nn.Linear(args.hidden_dims, 10)

    def return_hidden(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(-1, self.filters_length*self.last_layer*self.last_layer)
        features = self.preluip1(self.ip1(x))
        return features

    def forward(self, x):
        features = self.return_hidden(x)
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        # x = x.view(-1, self.filters_length*self.last_layer*self.last_layer)
        # features = self.preluip1(self.ip1(x))
        
        x = self.dropoutip1(features)
        logits = self.linear(x)
        return logits

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.device = args[0] # store device
        
        self.layer1 = self.layer1.to(*args, **kwargs)
        self.layer2 = self.layer2.to(*args, **kwargs)
        self.layer3 = self.layer3.to(*args, **kwargs)
        self.layer4 = self.layer4.to(*args, **kwargs)

        self.ip1 = self.ip1.to(*args, **kwargs)
        self.preluip1 = self.preluip1.to(*args, **kwargs)
        self.dropoutip1 = self.dropoutip1.to(*args, **kwargs)
        self.linear = self.linear.to(*args, **kwargs)
        return self

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)


class Resnet50(nn.Module):
    def __init__(self, args):
        super(Resnet50, self).__init__()

        self.pretrained = models.resnet50(pretrained=False)
        self.fc1 = nn.Linear(1000, 160)
        self.dp1 = nn.Dropout(args.dropout)
        self.linear = nn.Linear(160, 100)
        self.dp2 = nn.Dropout(args.dropout)

        # init the fc layers
        self.pretrained.fc.weight.data.normal_(mean=0.0, std=0.01)
        self.pretrained.fc.bias.data.zero_()
        self.fc1.apply(Xavier)
        self.linear.apply(Xavier)

    def return_hidden(self,x):
        # x = x.view(x.size(0), -1)
        x = x.view(x.size(0), 3, 32, 32)
        x = self.pretrained(x)
        x = self.dp1(torch.relu(x))
        features = torch.relu(self.fc1(x))
        return features

    def forward(self, x):
        features = self.return_hidden(x)
        out = self.linear(self.dp2(features))
        return out
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)



''' classifier for GEN and GEN-MIR'''
class classifier(nn.Module):
    def __init__(self, args):
        super(classifier, self).__init__()

        K = args.cls_hiddens
        L = np.prod(args.input_size)
        n_classes = args.n_classes
        self.args = args

        activation = nn.ReLU()
        self.layer = nn.Sequential(
            Reshape([-1]),
            GatedDense(L, K, activation=activation),
            nn.Dropout(p=0.2),
            GatedDense(K, n_classes, activation=None)
        )

        # get gradient dimension:
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())

    def forward(self, x):
        out = self.layer(x)
        return out
