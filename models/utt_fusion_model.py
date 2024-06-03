
import torch
import os
import json
import torch.nn.functional as F
from models.base_model import BaseModel
from models.networks.fc import FcEncoder
from models.networks.lstm import LSTMEncoder
from models.networks.textcnn import TextCNN
from models.networks.transformer import Transformer2
from models.networks.classifier import FcClassifier
from models.networks.discriminator import DomainDiscriminator


class UttFusionModel(BaseModel):
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
        #self.weight = 0
        self.disc_test = opt.disc_test
        self.loss_names = ['CE']
        self.modality = opt.modality
        self.model_names = ['C']
        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))
        cls_input_size = opt.embd_size_a * int("A" in self.modality) + \
                         opt.embd_size_v * int("V" in self.modality) + \
                         opt.embd_size_l * int("L" in self.modality)
        self.netC = FcClassifier(cls_input_size, cls_layers, output_dim=opt.output_dim, dropout=opt.dropout_rate, use_bn=opt.bn)
        
        # acoustic model
        if 'A' in self.modality:
            self.model_names.append('A')
            self.model_names.append('discA')
            self.loss_names.append('domdisc_a')
            self.netA = LSTMEncoder(opt.input_dim_a, opt.embd_size_a, embd_method=opt.embd_method_a)

            #self.netA = TextCNN(opt.input_dim_a, opt.embd_size_a)
            #self.netA = Transformer2(opt.input_dim_a, 2, 8, opt.embd_size_a)


            self.netdiscA = DomainDiscriminator(opt.embd_size_a)
            
            
        # lexical model
        if 'L' in self.modality:
            self.model_names.append('L')
            self.model_names.append('discL')
            self.loss_names.append('domdisc_l')
            self.netL = TextCNN(opt.input_dim_l, opt.embd_size_l)
            #self.netL = Transformer2(opt.input_dim_l, 2, 16, opt.embd_size_l)


            self.netdiscL = DomainDiscriminator(opt.embd_size_l)
            
        # visual model
        if 'V' in self.modality:
            self.model_names.append('V')
            self.model_names.append('discV')
            self.loss_names.append('domdisc_v')
            self.netV = LSTMEncoder(opt.input_dim_v, opt.embd_size_v, opt.embd_method_v)
            #self.netV = TextCNN(opt.input_dim_v, opt.embd_size_v)
            #self.netV = Transformer2(opt.input_dim_v, 1, 8, opt.embd_size_v)


            self.netdiscV = DomainDiscriminator(opt.embd_size_v)
            
        if self.isTrain:
            # ce_weight = torch.tensor([0.01, 0.33, 0.33, 0.33]).to(self.device)
            # self.criterion_ce = torch.nn.CrossEntropyLoss(weight=ce_weight)

            self.criterion_ce = torch.nn.CrossEntropyLoss()
            self.criterion_domdisc = torch.nn.CrossEntropyLoss()

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            parameters = [{'params': getattr(self, 'net'+net).parameters()} for net in self.model_names]
            #parameters += [{'params': getattr(self, 'disc'+net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(parameters, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
            
            # parameters = [{'params': getattr(self, 'net'+net).parameters(), 'lr': opt.lr} for net in ['C', 'A', 'L']]
            # parameters += [{'params': self.netV.parameters(), 'lr': 2e-4}]
            # self.optimizer = torch.optim.Adam(parameters, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
            #self.optimizer = torch.optim.SGD(parameters, lr=0.1)
            self.optimizers.append(self.optimizer)
            self.output_dim = opt.output_dim


            
            # parameters_C = [{'params': getattr(self, 'net'+net).parameters()} for net in self.model_names if net == 'C']
            # self.optimizerC = torch.optim.Adam(parameters_C, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
            # self.optimizers.append(self.optimizerC)
            # parameters_D = [{'params': getattr(self, 'net'+net).parameters()} for net in self.model_names if net != 'C']
            # self.optimizerD = torch.optim.RMSprop(parameters_D, lr=0.001, alpha=0.9)
            # self.optimizers.append(self.optimizerD)

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

            domain_source_labels = torch.zeros(self.label.shape[0]).type(torch.LongTensor)
            domain_target_labels = torch.ones(self.label_t.shape[0]).type(torch.LongTensor)
            self.domain_combined_label = torch.cat((domain_source_labels, domain_target_labels), 0).cuda()

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
        final_embd = []
        if self.isTrain:
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
            

        else:
            if 'A' in self.modality:
                self.feat_A = self.netA(self.acoustic)
                final_embd.append(self.feat_A)
            
            if 'V' in self.modality:
                #self.visual[self.visual > 10] = 10
                #self.visual[self.visual < -10] = -10
                self.feat_V = self.netV(self.visual)
                final_embd.append(self.feat_V)

            if 'L' in self.modality:
                self.feat_L = self.netL(self.lexical)
                final_embd.append(self.feat_L)

        
        # get model outputs
        self.feat = torch.cat(final_embd, dim=-1)
        self.logits, self.ef_fusion_feat = self.netC(self.feat)
        self.pred = F.softmax(self.logits, dim=-1)
        
    def backward(self):
        """Calculate the loss for back propagation"""
        self.loss_CE = self.criterion_ce(self.logits, self.label)
        self.loss_domdisc_a = 0
        self.loss_domdisc_v = 0
        self.loss_domdisc_l = 0 
        if 'A' in self.modality:
            self.loss_domdisc_a = self.criterion_domdisc(self.domainA, self.domain_combined_label)
        if 'V' in self.modality:
            self.loss_domdisc_v = self.criterion_domdisc(self.domainV, self.domain_combined_label)
        if 'L' in self.modality:
            self.loss_domdisc_l = self.criterion_domdisc(self.domainL, self.domain_combined_label)
        weight_cls = 1
        
        weight_disc_a = self.weight
        weight_disc_v = self.weight
        weight_disc_l = self.weight
        # loss = weight_cls * self.loss_CE + weight_disc_a * self.loss_domdisc_a + weight_disc_v * self.loss_domdisc_v + weight_disc_l * self.loss_domdisc_l

        p = 2
        loss_lp = (self.loss_domdisc_a**p + self.loss_domdisc_v**p + self.loss_domdisc_l**p)**(1/p)
        loss = weight_cls * self.loss_CE + weight_disc_a * loss_lp

        # loss_max = max(self.loss_domdisc_a, self.loss_domdisc_v, self.loss_domdisc_l)
        # loss = weight_cls * self.loss_CE + weight_disc_a * loss_max

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


        """Calculate the loss"""
        self.loss_domdisc_a = 0
        self.floss_domdisc_v = 0
        self.loss_domdisc_l = 0 
        if 'A' in self.modality:
            self.loss_domdisc_a = self.criterion_domdisc(self.domainA, self.domain_combined_label).item()  #.detach().cpu().numpy()
        if 'V' in self.modality:
            self.loss_domdisc_v = self.criterion_domdisc(self.domainV, self.domain_combined_label).item() # .detach().cpu().numpy()
        if 'L' in self.modality:
            self.loss_domdisc_l = self.criterion_domdisc(self.domainL, self.domain_combined_label).item() #.detach().cpu().numpy()



    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()   
        # backward
        self.optimizer.zero_grad() 
        self.backward()  
        self.optimizer.step()

        # self.optimizerC.zero_grad()  
        # self.optimizerD.zero_grad()  
        # self.backward()            
        # self.optimizerC.step() 
        # self.optimizerD.step() 

        # for model_name in self.model_names:
        #     if model_name.startswith('disc'):
        #         # 获取对应模型的参数并进行 clamp 操作
        #         for param in getattr(self, 'net' + model_name).parameters():
        #             param.data.clamp_(-0.01, 0.01)
