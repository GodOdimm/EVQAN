import torch
import torch.nn as nn
from attention import Attention, NewAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier, SimpleClassifier1
from fc import FCNet
import numpy as np
from torch.nn import functional as F


def mask_softmax(x,mask):
    mask=mask.unsqueeze(2).float()
    x2=torch.exp(x-torch.max(x))
    x3=x2*mask
    epsilon=1e-5
    x3_sum=torch.sum(x3,dim=1,keepdim=True)+epsilon
    x4=x3/x3_sum.expand_as(x3)
    return x4


class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net1, v_net1, classifier1,q_net2, v_net2, classifier2, mask_classifier):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net1 = q_net1
        self.v_net1 = v_net1
        self.classifier1 = classifier1
        self.q_net2 = q_net2
        self.v_net2 = v_net2
        self.classifier2 = classifier2
        self.debias_loss_fn1 = None
        self.debias_loss_fn2 = None


    def forward(self, v, q, labels, bias, v_mask, alpha, q_mask=None, loss_type=None):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb, q_hidden = self.q_emb(w_emb)  # [batch, q_dim]

        att = self.v_att(v, q_emb)

        if v_mask is None:
            att = nn.functional.softmax(att, 1)
        else:
            att= mask_softmax(att,v_mask)

        v_emb = (att * v).sum(1)  # [batch, v_dim]

        q_repr1 = self.q_net1(q_emb)
        v_repr1 = self.v_net1(v_emb)
        joint_repr1 = v_repr1 * q_repr1
        # print(q_emb.size(),v_emb.size(),joint_repr1.size())
        # #[512, 1024]) torch.Size([512, 2048]) torch.Size([512, 1024]
        # joint_repr1 = q_emb * v_repr1

        q_repr2 = self.q_net2(q_emb)
        v_repr2 = self.v_net2(v_emb)
        joint_repr2 = v_repr2 * q_repr2
        # joint_repr2 = q_emb * v_repr2

        joint_repr = alpha*joint_repr1+(1-alpha)*joint_repr2

        # logits1 = self.classifier1(joint_repr1)
        #
        # logits2 = self.classifier2(joint_repr2)
        logits1 = self.classifier1(joint_repr1)

        logits2 = self.classifier2(joint_repr2)

        logits = alpha*logits1+(1-alpha)*logits2

        if labels is not None:
            loss1 = self.debias_loss_fn1(joint_repr1, logits1, bias, labels).mean(0)
            loss2 = self.debias_loss_fn2(joint_repr2, logits2, bias, labels).mean(0)
            # loss1 = self.debias_loss_fn1(joint_repr, logits1, bias, labels).mean(0)
            # loss2 = self.debias_loss_fn2(joint_repr, logits2, bias, labels).mean(0)
            loss = alpha*loss1+(1-alpha)*loss2

        else:
            loss = None
        return logits, loss, att

def build_baseline0(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier)


def build_baseline0_newatt(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net1 = FCNet([q_emb.num_hid, num_hid])
    v_net1 = FCNet([dataset.v_dim, num_hid])
    q_net2 = FCNet([q_emb.num_hid, num_hid])
    v_net2 = FCNet([dataset.v_dim, num_hid])
    print('v_att = ({},{},{})'.format(dataset.v_dim, q_emb.num_hid, num_hid))
    print('q_net=({},{})'.format(q_emb.num_hid, num_hid))
    print('v_net=({},{})'.format(dataset.v_dim, num_hid))
    mask_classifier = FCNet([36, dataset.num_ans_candidates])
    classifier1 = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    classifier2 = SimpleClassifier1(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net1, v_net1, classifier1,q_net2, v_net2, classifier2, mask_classifier)