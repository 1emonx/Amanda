
import numpy as np
import torch
#from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


##################################
#https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/

class ClassEmbedding():
    def __init__(self, dim, n):
        super(ClassEmbedding, self).__init__()
        self.dim = dim 
        self.n = n 
    
    def get_emd(self, class_label):
        num_class = class_label.shape[0]
        emd = np.zeros((num_class, self.dim))
        for k, label in enumerate(class_label):
            for i in np.arange(int(self.dim/2)):
                denominator = np.power(self.n, 2*i/self.dim)
                emd[k, 2*i] = np.sin(label/denominator)
                emd[k, 2*i+1] = np.cos(label/denominator)
        emd = torch.tensor(emd)
        emd = emd.to(torch.float32)
        emd = F.normalize(emd, p=2, dim=1)
        return emd

############################################
#https://github.com/ckarouzos/slp_daptmlm/blob/master/slp/util/embeddings.py

class PositionalEncoding(nn.Module):
    """
    PositionalEncoding
    PE(pos,2i)=sin(pos/10000^(2i/dmodel))
    PE(pos,2i+1)=cos(pos/10000^(2i/dmodel))
    """
    def __init__(self, max_length, embedding_dim=512, device='cpu'):
        super(PositionalEncoding, self).__init__()
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        pe = torch.zeros(max_length, embedding_dim,
                         dtype=torch.float, device=device)
        embedding_indices = torch.arange(0, embedding_dim,
                                         dtype=torch.float, device=device)
        position_indices = (torch
                            .arange(0, max_length,
                                    dtype=torch.float, device=device)
                            .unsqueeze(-1))
        # freq => (E,)
        freq_term = 10000 ** (2 * embedding_indices / embedding_dim)
        pe[:, 0::2] = torch.sin(position_indices / freq_term[0::2])
        pe[:, 1::2] = torch.cos(position_indices / freq_term[1::2])
        # pe => (1, max_length, E)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x => (B, L, E) sequence of embedded tokens
        """
        # (B, L, E)
        F.normalize(x, p=2, dim=2) + F.normalize(self.pe[:, :x.size(1)], p=2, dim=2)
        return x + 0.1*self.pe[:, :x.size(1)]

###############################################
"""
https://github.com/HobbitLong/SupContrast/blob/master/losses.py
"""



class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        #print('LLLLLLLLLLL:', labels)
        #print('MMMMMMMMMMMMMMMM:', mask, mask.shape, sum(mask))
        mask = mask * logits_mask
        #print('AAAAAAAAAAA:', mask, sum(mask))
        #print('BBBBBBBBBBBBBB:', logits_mask)

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



if __name__ == '__main__':      
    cls_emb = ClassEmbedding(dim=4, n=100)
    label = torch.tensor([1,2,2,3,0, 0])

    print(cls_emb.n)
    a1 = cls_emb.get_emd(label)
    print(a1)




# def classembedding(class_label, d, n=10000):
#     num_class = class_label.shape[0]
#     emd = np.zeros((num_class, d))
#     for k, label in enumerate(class_label):
#         for i in np.arange(int(d/2)):
#             denominator = np.power(n, 2*i/d)
#             emd[k, 2*i] = np.sin(label/denominator)
#             emd[k, 2*i+1] = np.cos(label/denominator)
#     emd = torch.tensor(emd)
#     return emd

# label = torch.tensor([1,2,2,3,0, 0])
# P = classembedding(class_label=label, d=4, n=100)
# print(P)

