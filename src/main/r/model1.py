import math
import os
import sys

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.clip_grad import clip_grad_norm
import torch.nn.functional as F

def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


def EncoderImage(img_dim, embed_size, use_abs=False, no_imgnorm=False):
    """A wrapper to image encoders. Chooses between an encoder that uses
    precomputed image features, `EncoderImagePrecomp`, or an encoder that
    computes image features on the fly `EncoderImageFull`.
    """
    img_enc = EncoderImagePrecomp(img_dim, embed_size, use_abs, no_imgnorm)
    return img_enc


class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, use_abs=False, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs

        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)

        # take the absolute value of embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecomp, self).load_state_dict(new_state)


# tutorials/08 - Language Model
# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 use_abs=False):
        super(EncoderText, self).__init__()
        self.use_abs = use_abs
        self.embed_size = embed_size

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        # caption embedding
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True)

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # **** sort
        lengths = torch.LongTensor(lengths).to(x.device)

        lengths, index = torch.sort(lengths, descending=True)
        x = torch.index_select(x, dim=0, index=index)

        # Embed word ids to vectors
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        I = lengths.view(-1, 1, 1)
        I = Variable(I.expand(x.size(0), 1, self.embed_size)-1).cuda()
        out = torch.gather(padded[0], 1, I).squeeze(1)

        # normalization in the joint embedding space
        out = l2norm(out)

        # take absolute value, used by order embeddings
        if self.use_abs:
            out = torch.abs(out)

        # **** unsort
        reverse_index = torch.sort(index)[1]  # argsort
        out = torch.index_select(out, dim=0, index=reverse_index)

        return out


def cosine_sim(x, y):
    """Cosine similarity between all the image and sentence pairs
    """
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        return x.mm(y.t())
    elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        return np.matmul(x, y.T)


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, sim_func, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.sim = sim_func
        self.max_violation = max_violation

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()


class VSE(nn.Module):
    def __init__(self, img_dim, embed_size, vocab_size, word_dim, num_layers, use_abs=False, no_imgnorm=False):
        super().__init__()
        self.img_enc = EncoderImage(img_dim, embed_size,
                                    use_abs=use_abs,
                                    no_imgnorm=no_imgnorm)
        self.txt_enc = EncoderText(vocab_size, word_dim,
                                   embed_size, num_layers,
                                   use_abs=use_abs)

    def state_dict(self, **kwargs):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict, **kwargs):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

    def forward_emb(self, images, captions, lengths):
        return self.forward(images, captions, lengths)

    def forward(self, images, captions, lengths):
        """Compute the image and caption embeddings
        """
        # Forward
        img_emb = self.img_enc(images)
        cap_emb = self.txt_enc(captions, lengths)
        return img_emb, cap_emb


class Dictionary(nn.Module):
    def __init__(self, feat_dim, dict_size):
        super().__init__()
        self.feat_dim = feat_dim
        self.dict_size = dict_size
        self.w = nn.Parameter(torch.Tensor(feat_dim, dict_size))
        self.reset_parameters()

    def reset_parameters(self):
        truncated_normal_(self.w, mean=0, std=1)
        # memory_init = np.random.rand(self.feat_dim, self.dict_size) / 100
        # self.w.data = torch.from_numpy(memory_init).to(self.w.data.dtype).cuda().requires_grad_()

    def forward(self, x):
        """
        :param x: (batch_size, feat_dim)
        :return:
        """
        weight = torch.mm(x, self.w)        # (batch_size, dict_size)
        return F.softmax(weight, dim=1)
        # return weight

    def reconstruct(self, x):
        """
        :param x:  (batch_size, feat_dim)
        :return:
        """
        batch_size = x.shape[0]
        _weight = self.forward(x)            # (batch_size, dict_size)
        weight = _weight.unsqueeze(2)        # (batch_size, dict_size, 1)
        _w = self.w.unsqueeze(0).expand(batch_size, self.feat_dim, self.dict_size)
        _x = torch.bmm(_w, weight)          # (batch_size, feat_dim, 1)
        _x = _x.squeeze(2)                  # (batch_size, feat_dim)
        return _x, _weight


class VSE0(nn.Module):
    def __init__(self, img_dim, sent_dim, use_abs=False, no_imgnorm=False):
        super().__init__()

        self.img_enc = EncoderImage(img_dim, 512, use_abs=use_abs, no_imgnorm=no_imgnorm)
        self.txt_enc = EncoderImage(sent_dim, 512, use_abs=use_abs, no_imgnorm=no_imgnorm)
        # self.A = nn.Parameter(torch.Tensor(512, 512))

        self.reset_parameters()

    def reset_parameters(self):
        pass
        # stdv = 1. / math.sqrt(self.weight.size(0))
        # self.A.data.uniform_(-stdv, stdv)
        # r = np.sqrt(6.) / np.sqrt(self.A.shape[0] +
        #                           self.A.shape[1])
        # self.A.data.uniform_(-r, r)

    def get_sim_func(self):
        def sim_func(x, y):
            return cosine_sim(x, y)
        return sim_func

    def forward_emb(self, images, captions):
        return self.forward(images, captions)

    def forward(self, images, captions):
        image_emb = self.img_enc(images)
        # image_emb = torch.matmul(image_emb, self.A)
        sent_emb = self.txt_enc(captions)
        return image_emb, sent_emb

class VSE1(nn.Module):
    def __init__(self, img_dim, embed_size, vocab_size, word_dim, num_layers, use_abs=False, no_imgnorm=False,
                 use_dict=False):
        super().__init__()
        self.use_dict = use_dict
        self.img_enc = EncoderImage(img_dim, embed_size,
                                    use_abs=use_abs,
                                    no_imgnorm=no_imgnorm)
        self.txt_enc = EncoderText(vocab_size, word_dim,
                                   embed_size, num_layers,
                                   use_abs=use_abs)
        self.A = nn.Parameter(torch.Tensor(embed_size, embed_size))

        if self.use_dict:
            self.img_dict = Dictionary(dict_size=512, feat_dim=embed_size)
            # self.txt_dict = Dictionary(dict_size=512, feat_dim=embed_size)
            self.txt_dict = self.img_dict

        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.weight.size(0))
        # self.A.data.uniform_(-stdv, stdv)
        r = np.sqrt(6.) / np.sqrt(self.A.shape[0] +
                                      self.A.shape[1])
        self.A.data.uniform_(-r, r)

    def forward_emb(self, images, captions, lengths, **kwargs):
        return self.forward(images, captions, lengths)

    def get_sim_func(self):
        def sim_func(x, y):
            return cosine_sim(x, y)
        return sim_func

    def forward(self, images, captions, lengths, **kwargs):
        img_emb = self.img_enc(images)
        if self.use_dict:
            # img_emb, _img_weight = self.img_dict.reconstruct(img_emb)
            _, img_emb = self.img_dict.reconstruct(img_emb)
        img_emb = torch.matmul(img_emb, self.A)

        cap_emb = self.txt_enc(captions, lengths)
        if self.use_dict:
            # cap_emb, _img_weight = self.txt_dict.reconstruct(cap_emb)
            _, cap_emb = self.txt_dict.reconstruct(cap_emb)

        return img_emb, cap_emb


class VSE2(nn.Module):
    def __init__(self, img_dim, embed_size, vocab_size, word_dim, num_layers, use_abs=False, no_imgnorm=False,
                 use_dict=False):
        super().__init__()
        self.use_dict = use_dict
        self.img_enc = EncoderImage(img_dim, embed_size,
                                    use_abs=use_abs,
                                    no_imgnorm=no_imgnorm)
        self.txt_enc = EncoderText(vocab_size, word_dim,
                                   embed_size, num_layers,
                                   use_abs=use_abs)
        self.A = nn.Parameter(torch.Tensor(embed_size, embed_size))
        self.A_t = nn.Parameter(torch.Tensor(embed_size, embed_size))

        if self.use_dict:
            self.img_dict = Dictionary(dict_size=512, feat_dim=embed_size)
            # self.txt_dict = Dictionary(dict_size=512, feat_dim=embed_size)
            self.txt_dict = self.img_dict

            self.img_dict_t = Dictionary(dict_size=512, feat_dim=embed_size)
            self.txt_dict_t = self.img_dict_t

        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.weight.size(0))
        # self.A.data.uniform_(-stdv, stdv)
        r = np.sqrt(6.) / np.sqrt(self.A.shape[0] +
                                      self.A.shape[1])
        self.A.data.uniform_(-r, r)
        self.A_t.data.uniform_(-r, r)

    def get_params(self, mode='source'):
        modules = {}
        if mode == 'source':
            modules = {self.img_enc, self.txt_enc, self.A}
            if self.use_dict:
                modules.add(self.img_dict)
                modules.add(self.txt_dict)
        elif mode == 'target':
            if self.use_dict:
                modules = {self.img_dict_t, self.txt_dict_t, self.A_t}
            else:
                modules = {self.img_enc, self.txt_enc, self.A}
        params = []
        for m in modules:
            if isinstance(m, nn.Module):
                params.extend(m.parameters())
            elif isinstance(m, nn.Parameter):
                params.append(m)
        return params

    def forward_emb(self, images, captions, lengths, mode='source'):
        return self.forward(images, captions, lengths, mode)

    def get_sim_func(self):
        def sim_func(x, y):
            return cosine_sim(x, y)
        return sim_func

    def forward(self, images, captions, lengths, mode='source'):
        img_emb_input = self.img_enc(images)
        if self.use_dict:
            img_emb, _ = self.img_dict.reconstruct(img_emb_input)
            img_emb = torch.matmul(img_emb, self.A)
            if mode == 'target':
                img_emb_err = img_emb_input - img_emb
                _img_emb_err, _ = self.img_dict_t.reconstruct(img_emb_err)
                _img_emb_err = torch.matmul(_img_emb_err, self.A_t)
                img_emb = img_emb + _img_emb_err
        else:
            img_emb = img_emb_input

        cap_emb_input = self.txt_enc(captions, lengths)
        if self.use_dict:
            cap_emb, _ = self.txt_dict.reconstruct(cap_emb_input)
            if mode == 'target':
                cap_emb_err = cap_emb_input - cap_emb
                _cap_emb_err, _ = self.txt_dict_t.reconstruct(cap_emb_err)
                cap_emb = cap_emb + _cap_emb_err
        else:
            cap_emb = cap_emb_input

        return img_emb, cap_emb

