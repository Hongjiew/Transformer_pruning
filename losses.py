# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Implements the knowledge distillation loss
"""
import torch
from torch.nn import functional as F


class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                #We provide the teacher's targets in log probability because we use log_target=True 
                #(as recommended in pytorch https://github.com/pytorch/pytorch/blob/9324181d0ac7b4f7949a574dbc3e8be30abe7041/torch/nn/functional.py#L2719)
                #but it is possible to give just the probabilities and set log_target=False. In our experiments we tried both.
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
            #We divide by outputs_kd.numel() to have the legacy PyTorch behavior. 
            #But we also experiments output_kd.size(0) 
            #see issue 61(https://github.com/facebookresearch/deit/issues/61) for more details
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss

class TokenRankLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, base_criterion: torch.nn.Module, prune_list, alpha: float, beta:float):
        super().__init__()
        self.base_criterion = base_criterion
        #self.teacher_model = teacher_model
        #assert distillation_type in ['none', 'soft', 'hard']
        #self.distillation_type = distillation_type
        self.alpha = alpha
        self.beta = beta
        self.prune_list = prune_list
        self.count = 0
        self.clf_loss = 0
        self.prune_loss = 0
        #self.tau = tau

    def forward(self, inputs, outputs, labels, eps=1e-6):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs, all_features, all_importance, all_tokens = outputs
        base_loss = self.base_criterion(outputs, labels)
        token_mask = torch.cat(all_tokens)
        imp = torch.cat(all_importance) + eps
        imp = imp[token_mask.bool()].view(outputs.shape[0],-1)
        # prune_loss = torch.mean(torch.sum(-imp*torch.log(imp),dim=-1))
        loss = base_loss*self.alpha 
        self.count+=1
        self.clf_loss += base_loss
        # self.prune_loss += prune_loss
        if self.count%100 == 0:
            print(f'Base loss: {self.clf_loss/100}')
            self.clf_loss = 0
            # self.prune_loss = 0

        return loss

class TokenRankLossWithDistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self,teacher_model:torch.nn.Module,  prune_list, base_criterion: torch.nn.Module,
                 alpha: float, beta: float, gamma:float, tau:float):
        super().__init__()
        self.base_criterion = base_criterion
        # #self.teacher_model = teacher_model
        # assert distillation_type in ['none', 'soft', 'hard']
        # self.distillation_type = distillation_type
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prune_list = prune_list
        self.teacher = teacher_model
        self.tau = tau
        self.count = 0
        self.clf_loss = 0
        self.distillation_loss = 0
        # self.prune_loss = 0
        self.kl_loss = 0

    def forward(self, inputs, outputs, labels, eps=1e-6):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs, all_features, all_tokens = outputs
        base_loss = self.base_criterion(outputs, labels)
        # prune_loss = torch.mean(torch.sum(-imp*torch.log(imp),dim=-1))
        with torch.no_grad():
            teacher_outputs, teacher_all_features = self.teacher(inputs)
        distillation_loss = 0
        for loc in self.prune_list:
            distillation_loss += ((teacher_all_features[loc][all_tokens[loc].bool()] - all_features[loc][all_tokens[loc].bool()])**2).mean()
        T = self.tau
        # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
        # with slight modifications
        kl_loss = F.kl_div(
            F.log_softmax(outputs / T, dim=1),
            #We provide the teacher's targets in log probability because we use log_target=True 
            #(as recommended in pytorch https://github.com/pytorch/pytorch/blob/9324181d0ac7b4f7949a574dbc3e8be30abe7041/torch/nn/functional.py#L2719)
            #but it is possible to give just the probabilities and set log_target=False. In our experiments we tried both.
            F.log_softmax(teacher_outputs / T, dim=1),
            reduction='batchmean',
            log_target=True
        ) 
        loss = base_loss*self.alpha + kl_loss*self.gamma + self.beta*distillation_loss/len(self.prune_list)
        self.count+=1
        self.clf_loss += base_loss.item()
        self.distillation_loss += distillation_loss.item()
        self.kl_loss+=kl_loss.item()
        if self.count%100 == 0:
            print(f'Base loss: {self.clf_loss/100}, KL Loss: {self.kl_loss/100}, Distillation Loss: {self.distillation_loss/(100*len(self.prune_list))}')
            self.clf_loss = 0
            self.distillation_loss = 0
            self.kl_loss = 0

        return loss



