import torch
import numpy as np
import torch.nn.functional as F


def coral(source, target):

    d = source.size(1)  # dim vector

    source_c, source_mu = compute_covariance(source)
    target_c, target_mu = compute_covariance(target)

    loss_c = torch.sum(torch.mul((source_c - target_c), (source_c - target_c)))

    #loss_c = loss_c / (4 * d * d)
    loss_mu = torch.sum(torch.mul((source_mu - target_mu), (source_mu - target_mu)))
    loss_mu = loss_mu / (2*d)
    #print(loss_c, loss_mu)
    return loss_c # loss_c #+ loss_mu


def compute_covariance(input_data):
    """
    Compute Covariance matrix of the input data
    """
    n = input_data.size(0)  # batch_size

    # Check if using gpu or cpu
    if input_data.is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    id_row = torch.ones(n).resize(1, n).to(device=device)
    sum_column = torch.mm(id_row, input_data)
    mean_column = torch.div(sum_column, n)
    term_mul_2 = torch.mm(mean_column.t(), mean_column)
    d_t_d = torch.mm(input_data.t(), input_data)
    c = torch.add(d_t_d, (-1 * term_mul_2)) * 1 / (n - 1)

    return c, mean_column

######

def log_var(input_data):
    n = input_data.size(0)  # batch_size
    d = input_data.size(1)

    # Check if using gpu or cpu
    if input_data.is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    var = torch.var(input_data, dim=0)
    # var = var -1 
    # var = var ** 2
    # log_var = sum(torch.log(var+1))

    input_data = F.normalize(input_data, p=2, dim=1)
    covar = torch.cov(input_data.T)
    det_covar = torch.linalg.det(covar)
    log_det_covar = torch.log10(det_covar / d**2 + 1)
    return log_det_covar






if __name__ == '__main__':
    a = torch.randn((128, 3))
    b = torch.randn((123, 3))
    loss = coral(a,b)
    c, _ = compute_covariance(a)
    d = log_var(a)
    print('log_var:', c.shape, c)
    print('variance:', d)
    print(loss)
