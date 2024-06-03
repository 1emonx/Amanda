import torch
import os
import json
import torch.nn.functional as F
from models.base_model import BaseModel
from models.networks.fc import FcEncoder
from models.networks.lstm import LSTMEncoder, LSTMEncoder2, TextSubNet
from models.networks.textcnn import TextCNN, TextCNN2, TextCNN3, RCNN, BERTCNN
from models.networks.classifier import FcClassifier, OnelayerClassifier
from models.networks.discriminator import DomainDiscriminator, DomainDiscriminatorS, DomainDiscriminatorSig
from models.func import ClassEmbedding, SupConLoss, PositionalEncoding
from models.opt import pte
from models.coralutils.coral import coral, log_var
import numpy as np
from scipy.stats import entropy
import math
import torch.nn.functional as F
from torch.distributions import Categorical


class AmandaModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--input_dim_a', type=int, default=130, help='acoustic input dim')
        parser.add_argument('--input_dim_l', type=int, default=1024, help='lexical input dim')
        parser.add_argument('--input_dim_v', type=int, default=384, help='visual input dim')
        parser.add_argument('--embd_size_a', default=128, type=int, help='audio model embedding size')
        parser.add_argument('--embd_size_l', default=128, type=int, help='text model embedding size')
        parser.add_argument('--embd_size_v', default=128, type=int, help='visual model embedding size')
        parser.add_argument('--embd_method_a', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'], \
            help='audio embedding method,last,mean or atten')
        parser.add_argument('--embd_method_v', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'], \
            help='visual embedding method,last,mean or atten')
        parser.add_argument('--cls_layers', type=str, default='128,128', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--dropout_rate', type=float, default=0.3, help='rate of dropout')
        parser.add_argument('--bn', action='store_true', help='if specified, use bn layers in FC')
        parser.add_argument('--modality', type=str, help='which modality to use for model')
        return parser

    def __init__(self, opt):
        """Initialize the LSTM autoencoder class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        # our expriment is on 10 fold setting, teacher is on 5 fold setting, the train set should match
        self.weight = opt.weight
        self.error_pte = []
        self.error_pte2 = []
        self.disc_test = opt.disc_test
        self.loss_names = ['CE', 'CEave_t', 'coral']
        self.modality = opt.modality
        self.model_names = ['C']
        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))
        cls_input_size = opt.embd_size_a * int("A" in self.modality) + \
                         opt.embd_size_v * int("V" in self.modality) + \
                         opt.embd_size_l * int("L" in self.modality)
        self.netC = FcClassifier(cls_input_size, cls_layers, output_dim=opt.output_dim, dropout=opt.dropout_rate, use_bn=opt.bn)
        self.netembedding = torch.nn.Embedding(num_embeddings=4, embedding_dim=128).to(self.device)

        self.clsemddim = 32
        self.netembedding2 = ClassEmbedding(dim=self.clsemddim, n=100)
        
        # acoustic model
        if 'A' in self.modality:
            self.model_names.append('A')
            self.model_names.append('CA')
            self.loss_names.append('CE_A')
            self.loss_names.append('coral_a')
            self.loss_names.append('CEave_A_t')
            self.loss_names.append('lv_a')

            self.netA = TextCNN3(opt.input_dim_a, opt.embd_size_a)
            self.netCA = OnelayerClassifier(opt.embd_size_a, opt.output_dim, dropout=opt.dropout_rate)
            
            
        # lexical model
        if 'L' in self.modality:
            self.model_names.append('L')
            self.model_names.append('CL')
            self.loss_names.append('CE_L')
            self.loss_names.append('coral_l')
            self.loss_names.append('CEave_L_t')
           
            self.netL = TextCNN3(opt.input_dim_l, opt.embd_size_l)
            self.netCL = OnelayerClassifier(opt.embd_size_l, opt.output_dim, dropout=opt.dropout_rate)
            
        # visual model
        if 'V' in self.modality:
            self.model_names.append('V')
            self.model_names.append('CV')
            self.loss_names.append('CE_V')
            self.loss_names.append('coral_v')
            self.loss_names.append('CEave_V_t')

            self.netV = LSTMEncoder(opt.input_dim_v, opt.embd_size_v, opt.embd_method_v)
            self.netCV = OnelayerClassifier(opt.embd_size_v, opt.output_dim, dropout=opt.dropout_rate)
            
            
        if self.isTrain:

            self.criterion_ce = torch.nn.CrossEntropyLoss()
            self.criterion_domdisc = torch.nn.CrossEntropyLoss() #torch.nn.BCELoss()   #
            self.criterion_supcon = SupConLoss()
            self.criterion_nll = torch.nn.NLLLoss()
            self.criterion_mse = torch.nn.MSELoss()

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            parameters = [{'params': getattr(self, 'net'+net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(parameters, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
            self.optimizers.append(self.optimizer)
            self.output_dim = opt.output_dim

            self.output_dim = opt.output_dim

        # modify save_dir
        self.save_dir = os.path.join(self.save_dir, str(opt.cvNo))
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
    
    def set_input(self, input, input_t=None, alpha=None):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """

        if self.isTrain or (self.disc_test and input_t != None):
            if 'A' in self.modality:
                self.acoustic = input['A_feat'].float().to(self.device)
                self.acoustic_t = input_t['A_feat'].float().to(self.device)

            if 'L' in self.modality:
                self.lexical = input['L_feat'].float().to(self.device)
                self.lexical_t = input_t['L_feat'].float().to(self.device)

            if 'V' in self.modality:
                self.visual = input['V_feat'].float().to(self.device)
                self.visual_t = input_t['V_feat'].float().to(self.device)
            
            self.label = input['label'].to(self.device)
            self.label_t = input_t['label'].to(self.device)
            self.alpha = alpha


        else:
            if 'A' in self.modality:
                self.acoustic = input['A_feat'].float().to(self.device)
            if 'L' in self.modality:
                self.lexical = input['L_feat'].float().to(self.device)
            if 'V' in self.modality:
                self.visual = input['V_feat'].float().to(self.device)
            
            self.label = input['label'].to(self.device)




    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.final_embd = []
        self.final_embd_target = []
        if self.isTrain:
            if 'A' in self.modality:
                self.feat_A = self.netA(self.acoustic)
                self.final_embd.append(self.feat_A)

                self.feat_A_t = self.netA(self.acoustic_t)
                self.final_embd_target.append(self.feat_A_t)

                self.logits_A, _ = self.netCA(self.feat_A)
                self.pred_A = F.softmax(self.logits_A, dim=-1)

                self.logits_A_t, _ = self.netCA(self.feat_A_t)
                self.pred_A_t = F.softmax(self.logits_A_t, dim=-1)

                self.pred_A_t_ave = self.pred_A_t.mean(dim=0)
            
            if 'V' in self.modality:
                self.feat_V = self.netV(self.visual)
                self.final_embd.append(self.feat_V)

                self.feat_V_t = self.netV(self.visual_t)
                self.final_embd_target.append(self.feat_V_t)

                self.logits_V, _ = self.netCV(self.feat_V)
                self.pred_V = F.softmax(self.logits_V, dim=-1)

                self.logits_V_t, _ = self.netCV(self.feat_V_t)
                self.pred_V_t = F.softmax(self.logits_V_t, dim=-1)

                self.pred_V_t_ave = self.pred_V_t.mean(dim=0)

            if 'L' in self.modality:
                self.feat_L = self.netL(self.lexical)
                self.final_embd.append(self.feat_L)
                self.feat_L_t = self.netL(self.lexical_t)
                self.final_embd_target.append(self.feat_L_t)

                self.logits_L, _ = self.netCL(self.feat_L)
                self.pred_L = F.softmax(self.logits_L, dim=-1)

                self.logits_L_t, _ = self.netCL(self.feat_L_t)
                self.pred_L_t = F.softmax(self.logits_L_t, dim=-1)

                self.pred_L_t_ave = self.pred_L_t.mean(dim=0)
            

        else:
            if 'A' in self.modality:
                self.feat_A = self.netA(self.acoustic)
                self.final_embd.append(self.feat_A)
            
            if 'V' in self.modality:

                self.feat_V = self.netV(self.visual)
                self.final_embd.append(self.feat_V)

            if 'L' in self.modality:
                self.feat_L = self.netL(self.lexical)
                self.final_embd.append(self.feat_L)

        
        # get model outputs
        self.feat = torch.cat(self.final_embd, dim=-1)
        self.logits, self.fusion_feat = self.netC(self.feat)
        self.pred = F.softmax(self.logits, dim=-1)

        if self.isTrain:

            self.feat_target = torch.cat(self.final_embd_target, dim=-1)
            self.logits_target, self.fusion_feat_target = self.netC(self.feat_target)
            self.pred_target = F.softmax(self.logits_target, dim=-1)  
            self.entropy_target = entropy(self.pred_target.T.cpu().detach().numpy())

            self.pred_t_ave = self.pred_target.mean(dim=0)
  
        
    def backward(self):
        """Calculate the loss for back propagation"""
        ref = torch.tensor([2/9, 3/9, 2/9, 2/9]).to(self.device)
        self.loss_CE = self.criterion_ce(self.logits, self.label)
        self.loss_CEave_t = sum([-ref[i]*torch.log(self.pred_t_ave[i]) for i in range(0, self.output_dim)])
        self.loss_lv_a = log_var(self.feat_A)
        self.loss_lv_v = log_var(self.feat_V)
        self.loss_lv_l = log_var(self.feat_L)
        loss_lv = (self.loss_lv_a + self.loss_lv_v + self.loss_lv_l)
        

        self.loss_CE_A = 0
        self.loss_CE_V = 0
        self.loss_CE_L = 0
        self.loss_coral_a = 0
        self.loss_coral_v = 0
        self.loss_coral_l = 0

        self.loss_CEave_A_t = 0
        self.loss_CEave_V_t = 0
        self.loss_CEave_L_t = 0
        if 'A' in self.modality:
            self.loss_CE_A = self.criterion_ce(self.logits_A, self.label)
            self.loss_coral_a = coral(self.pred_A, self.pred_A_t)  
            self.loss_CEave_A_t = sum([-ref[i]*torch.log(self.pred_A_t_ave[i]) for i in range(0, self.output_dim)])
        if 'V' in self.modality:
            self.loss_CE_V = self.criterion_ce(self.logits_V, self.label)
            self.loss_coral_v = coral(self.pred_V, self.pred_V_t) 
            self.loss_CEave_V_t = sum([-ref[i]*torch.log(self.pred_V_t_ave[i]) for i in range(0, self.output_dim)])
        if 'L' in self.modality:
            self.loss_CE_L = self.criterion_ce(self.logits_L, self.label)
            self.loss_coral_l = coral(self.pred_L, self.pred_L_t) 
            self.loss_CEave_L_t = sum([-ref[i]*torch.log(self.pred_L_t_ave[i]) for i in range(0, self.output_dim)])

        self.loss_coral = coral(self.pred, self.pred_target) 
        loss_CEave = self.loss_CEave_t #+ self.loss_CEave_A_t + self.loss_CEave_V_t + self.loss_CEave_L_t 

        weight_cls = 1
        weight_disc_a = self.weight
        weight_disc_v = self.weight
        weight_disc_l = self.weight
        p=2
        logexp = 0
        if logexp == 1:
            loss_coral_sum = torch.log(torch.exp(self.loss_coral) + torch.exp(self.loss_coral_a) + torch.exp(self.loss_coral_v) + torch.exp(self.loss_coral_l))
        elif p < 100 and p>=1:
            loss_coral_sum = (self.loss_coral**p + self.loss_coral_a**p + self.loss_coral_v**p + self.loss_coral_l**p)**(1/p)
        elif p == 100:
            loss_coral_sum = max([self.loss_coral, self.loss_coral_a, self.loss_coral_v, self.loss_coral_l])
        elif p < 1:
            loss_coral_sum = torch.log(torch.exp(p * self.loss_coral) + torch.exp(p * self.loss_coral_a) + torch.exp(p * self.loss_coral_v) + torch.exp(p * self.loss_coral_l))
        elif p>100:
            p = p-100
            loss_coral_sum = torch.log(torch.exp(p * self.loss_coral) + torch.exp(p * self.loss_coral_a) + torch.exp(p * self.loss_coral_v) + torch.exp(p * self.loss_coral_l))
        wei = 0.01
        if self.weight == 0:
            loss = weight_cls * self.loss_CE 
        
        elif self.weight == 1e-6:
            loss = weight_cls * (0.0005*2 * loss_lv + self.loss_CE) + 0.008 * (self.loss_CE_A + self.loss_CE_V + self.loss_CE_L) + 0.08 * loss_CEave #

        else:
            loss = weight_cls * (0.0005*2 * loss_lv + self.loss_CE) + 0.008 * (self.loss_CE_A + self.loss_CE_V + self.loss_CE_L) + 0.08 * loss_CEave + self.weight * loss_coral_sum  #


        loss.backward()
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net'+model).parameters(), 0.5)
    

    def test_loss(self):

        final_embd = []
        if self.isTrain or self.disc_test:
            if 'A' in self.modality:
                self.feat_A = self.netA(self.acoustic)
                final_embd.append(self.feat_A)

                self.feat_A_t = self.netA(self.acoustic_t)
                self.feat_A_c = torch.cat((self.feat_A, self.feat_A_t), 0)
                self.domainA = self.netdiscA(self.feat_A_c, self.alpha)
            
            if 'V' in self.modality:
                self.feat_V = self.netV(self.visual)
                final_embd.append(self.feat_V)

                self.feat_V_t = self.netV(self.visual_t)
                self.feat_V_c = torch.cat((self.feat_V, self.feat_V_t), 0)
                self.domainV = self.netdiscV(self.feat_V_c, self.alpha)

            if 'L' in self.modality:
                self.feat_L = self.netL(self.lexical)
                final_embd.append(self.feat_L)

                self.feat_L_t = self.netL(self.lexical_t)
                self.feat_L_c = torch.cat((self.feat_L, self.feat_L_t), 0)
                self.domainL = self.netdiscL(self.feat_L_c, self.alpha)
                # get model outputs
        self.feat = torch.cat(final_embd, dim=-1)
        self.logits, self.ef_fusion_feat = self.netC(self.feat)
        self.pred = F.softmax(self.logits, dim=-1)




    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()   
        # backward
        self.optimizer.zero_grad() 
        self.backward()  
        self.optimizer.step()
