from torch.nn import functional as F
from torch import Tensor
from torch import nn
import torch as th
import numpy as np

# custom pytorch linear layer
class SlimLinear(nn.Linear):
    def __init__(self, max_in_features: int, max_out_features: int, bias: bool = True,
                 device=None, dtype=None, slim_in=True, slim_out=True) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(max_in_features, max_out_features, bias, device, dtype)
        self.max_in_features = self.in_features = max_in_features
        self.max_out_features = self.out_features = max_out_features
        self.slim_in = slim_in
        self.slim_out = slim_out
        self.rho = 1

    def forward(self, input: Tensor) -> Tensor:
        if self.slim_in:
            self.in_features = max(1, int(self.rho * self.max_in_features))
        if self.slim_out:
            self.out_features = max(1,int(self.rho * self.max_out_features))
        #print(f'B4-shape:{self.weight.shape}')
        weight = self.weight[:self.out_features, :self.in_features]
        if self.bias is not None:
            bias = self.bias[:self.out_features]
        else:
            bias = self.bias
        y = F.linear(input, weight, bias)
        #utils.speak(f'RHO:{self.rho} IN:{weight.shape} OUT:{y.shape}')
        return y
        
# custom pytorch conv2d
class SlimConv2d(nn.Conv2d):
    def __init__(self, max_in_channels: int, max_out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = True, device=None, dtype=None, slim_in=True, slim_out=True) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(max_in_channels, max_out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, device=device, dtype=dtype)
        self.max_in_channels = self.in_channels = max_in_channels
        self.max_out_channels = self.out_channels = max_out_channels
        self.slim_in = slim_in
        self.slim_out = slim_out
        self.rho = 1

    def forward(self, input: Tensor) -> Tensor:
        if self.slim_in:
            self.in_channels = max(1,int(self.rho * self.max_in_channels))
        if self.slim_out:
            self.out_channels = max(1,int(self.rho * self.max_out_channels))
        #print(f'conv2d B4-shape:{self.weight.shape}')
        weight = self.weight[:self.out_channels,:self.in_channels,:,:]
        #print(f'conv2d A4-shape:{weight.shape}')
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        y = F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        #utils.speak(f'RHO:{self.rho} IN:{weight.shape} OUT:{y.shape}')
        return y
        
# custom pytorch convTrans2d
class SlimConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, max_in_channels: int, max_out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = True, device=None, dtype=None, slim_in=True, slim_out=True) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(max_in_channels, max_out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, device=device, dtype=dtype)
        self.max_in_channels = self.in_channels = max_in_channels
        self.max_out_channels = self.out_channels = max_out_channels
        self.slim_in = slim_in
        self.slim_out = slim_out
        self.rho = 1

    def forward(self, input: Tensor, output_size = None) -> Tensor:
        if self.slim_in:
            self.in_channels = max(1,int(self.rho * self.max_in_channels))
        if self.slim_out:
            self.out_channels = max(1,int(self.rho * self.max_out_channels))
        #print(f'trans2d B4-shape:{self.weight.shape}')
        weight = self.weight[:self.in_channels,:self.out_channels,:,:]
        #print(f'trans2d A4-shape:{weight.shape}')
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        num_spatial_dims = 2
        output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size,
        num_spatial_dims, self.dilation)
        y = F.conv_transpose2d(input, weight, bias, self.stride, self.padding, output_padding, self.groups, self.dilation)
        #utils.speak(f'RHO:{self.rho} IN:{weight.shape} OUT:{y.shape}')
        return y
        
# custom pytorch switch group norm
class SlimGroupNorm(nn.Module):
    def __init__(self, num_groups, max_channels, rhos):
        super().__init__()
        self.max_channels = max_channels
        self.idx_map = {}
        bns = []
        for idx, rho in enumerate(rhos):
            self.idx_map[rho] = idx
            n_channels = max(1,int(max_channels*rho))
            bns.append(nn.GroupNorm(num_groups, n_channels))
        self.bn = nn.ModuleList(bns)
        self.rho = 1
    def forward(self, input):
        idx = self.idx_map[self.rho]
        y = self.bn[idx](input)
        return y
        
# custom pytorch switch batch norm
class SlimBatchNorm2d(nn.Module):
    # num_groups is dummy value 
    def __init__(self, max_channels, rhos, num_groups=0):
        super().__init__()
        self.max_channels = max_channels
        self.idx_map = {}
        bns = []
        for idx, rho in enumerate(rhos):
            self.idx_map[rho] = idx
            n_channels = max(1,int(max_channels*rho))
            bns.append(nn.BatchNorm2d(n_channels))
        self.bn = nn.ModuleList(bns)
        self.rho = 1
    def forward(self, input):
        idx = self.idx_map[self.rho]
        y = self.bn[idx](input)
        return y

# change the slimming factor, rho, for all modules in model which contain a slim layer
def set_slim(model, rho):
    for module in model.modules():
        if 'Slim' in str(type(module)):
            module.rho = rho

# forward pass through model neural network
    # can return preditions from sampled rho values or explicitly set rho values
def slim_forward(model, data_loader, device, n_iterations=0,
            criterion=None, with_grad=False, memory_saver=True, return_predictions=False, return_losses=False, 
            x_preproc_funcs=None, x_preproc_paramss=None, y_preproc_funcs=None, y_preproc_paramss=None,
            rhos=None, low_rho=0.25, high_rho=0.75, nRhos=2):
    predictions = []
    losses = []
    
    # mini-batch iterations
    for iteration, data in enumerate(data_loader):
        x, y = data

        # process x
        if x_preproc_funcs is not None:
            for i in range(len(x_preproc_funcs)):
                x_preproc_func = x_preproc_funcs[i]
                x_preproc_params = x_preproc_paramss[i]
                x = x_preproc_func(x, **x_preproc_params)

        # process y
        if y_preproc_funcs is not None:
            for i in range(len(y_preproc_funcs)):
                y_preproc_func = y_preproc_funcs[i]
                y_preproc_params = y_preproc_paramss[i]
                y = y_preproc_func(y, **y_preproc_params)

        # sample rhos using sandwich rule or use preset list of rhos
        if rhos is None:
            use_rhos = np.random.uniform(low=low_rho, high=high_rho, size=nRhos)
            use_rhos = [low_rho] + list(use_rhos)
        else:
            use_rhos = rhos

        # forward pass
        model.optimizer.zero_grad()
        slim_predictions = []
        for rho in rhos:
            set_slim(model, rho)
            if with_grad:
                p = model(x.to(device=device))
                loss = criterion(p, y.to(device=device))
                loss.backward(retain_graph=True) # accumalate gradient over each rho
                model.optimizer.step()
                realized_loss = float(loss.detach().cpu())
            else:
                with th.no_grad():
                    p = model(x.to(device=device))
                    if return_losses:
                        loss = criterion(p, y.to(device=device))
                        realized_loss = float(loss.detach().cpu())
            if return_predictions:
                slim_predictions.append(p.detach().cpu().numpy())
        if return_predictions:
            predictions.append(np.stack(slim_predictions, axis=1))
        if return_losses:
            losses.append(realized_loss)
        if memory_saver:
            del x, y, p # clear mem from gpu

    # aggregate outputs as requested
    if return_predictions and return_losses:
        return np.vstack(predictions), losses
    if return_predictions:
        return np.vstack(predictions)
    if return_losses:
        return losses