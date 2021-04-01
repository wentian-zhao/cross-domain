import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from util.model import LanguageModel


class FCModel(nn.Module):
    def __init__(self, input_size, hidden_size, drop_prob_lm=0.5):
        print('init FCModel')
        super(FCModel, self).__init__()
        self.input_encoding_size = input_size
        self.rnn_size = hidden_size
        self.drop_prob_lm = drop_prob_lm
        self.hidden_size = hidden_size

        # Build a LSTM
        self.i2h = nn.Linear(self.input_encoding_size, 5 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.dropout = nn.Dropout(self.drop_prob_lm)

    def forward(self, xt, state):
        all_input_sums = self.i2h(xt) + self.h2h(state[0])
        sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
        sigmoid_chunk = F.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)

        in_transform = torch.max(all_input_sums.narrow(1, 3 * self.rnn_size, self.rnn_size),
            all_input_sums.narrow(1, 4 * self.rnn_size, self.rnn_size))
        next_c = forget_gate * state[1] + in_gate * in_transform
        next_h = out_gate * F.tanh(next_c)

        # output = self.dropout(next_h)
        state = next_h, next_c
        return state


class LSTMLanguageModel(LanguageModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        default_args = {
            'feat_dim': 2048, 'embedding_dim': 300, 'hidden_dim': 512, 'dropout_prob': 0.5
        }
        default_args.update(kwargs)
        kwargs = default_args

        feat_dim = kwargs['feat_dim']
        embedding_dim = kwargs['embedding_dim']
        hidden_dim = kwargs['hidden_dim']
        dropout_prob = kwargs['dropout_prob']

        embedding = kwargs.get('pretrained_embedding', None)
        self.use_pretrained_embedding = embedding is not None

        self.image_embedding = nn.Linear(in_features=feat_dim, out_features=embedding_dim)
        self.input_embedding = nn.Embedding(num_embeddings=len(self.vocab), embedding_dim=embedding_dim, padding_idx=0, _weight=embedding)
        # self.lstm = nn.LSTMCell(input_size=300, hidden_size=512)
        self.lstm = FCModel(input_size=embedding_dim, hidden_size=hidden_dim)
        self.output_embedding = nn.Linear(in_features=hidden_dim, out_features=len(self.vocab))
        self.dropout = nn.Dropout(dropout_prob)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        if not self.use_pretrained_embedding:
            self.input_embedding.weight.data.uniform_(-initrange, initrange)
        self.output_embedding.bias.data.fill_(0)
        self.output_embedding.weight.data.uniform_(-initrange, initrange)

    def prepare_feat(self, input_feature, **kwargs):
        batch_size = len(input_feature)
        prepared_feat = self.image_embedding(input_feature)
        return batch_size, prepared_feat

    def init_state(self, input_feature, **kwargs):
        device = input_feature.device
        batch_size = input_feature.shape[0]
        h_0 = torch.zeros((batch_size, self.lstm.hidden_size)).to(device)
        return self.lstm(input_feature, (h_0, h_0))

    def step(self, input_feature, last_word_id_batch, last_state, **kwargs):
        device = input_feature.device
        last_word_id_batch = torch.LongTensor(np.array(last_word_id_batch).astype(np.int64)).to(device)
        emb = self.input_embedding(last_word_id_batch)
        h, c = self.lstm(emb, last_state)
        output = self.dropout(h)
        output = self.output_embedding(output)
        return output, (h, c), None


class Attention(nn.Module):
    def __init__(self, rnn_size, att_hid_size):
        super(Attention, self).__init__()
        self.rnn_size = rnn_size
        self.att_hid_size = att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    # img_feats_attn ==> p_att_feats
    # h -> h_attn_1, att_feats -> img_feats_attn, p_att_feats -> p_att_feats
    def forward(self, h, att_feats, p_att_feats):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // self.rnn_size
        att = p_att_feats.view(-1, att_size, self.att_hid_size)

        att_h = self.h2att(h)  # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)  # batch * att_size * att_hid_size
        dot = att + att_h  # batch * att_size * att_hid_size
        # dot = F.tanh(dot)  # batch * att_size * att_hid_size
        dot = torch.tanh(dot)
        dot = dot.view(-1, self.att_hid_size)  # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)  # (batch * att_size) * 1
        dot = dot.view(-1, att_size)  # batch * att_size

        # weight = F.softmax(dot)  # batch * att_size
        weight = F.softmax(dot, dim=-1)  # batch * att_size
        att_feats_ = att_feats.view(-1, att_size, self.rnn_size)  # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)  # batch * att_feat_size

        return att_res


class TopDownAttnModel(LanguageModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        default_args = {
            'feat_dim': 2048, 'embedding_dim': 300, 'hidden_dim': 512, 'dropout_prob': 0.5,
            'attn_hidden_dim': 512,
        }
        default_args.update(kwargs)
        kwargs = default_args

        feat_dim = kwargs['feat_dim']
        embedding_dim = kwargs['embedding_dim']
        hidden_dim = kwargs['hidden_dim']
        image_embedding_dim = hidden_dim
        dropout_prob = kwargs['dropout_prob']
        attn_hidden_dim = kwargs['attn_hidden_dim']

        self.dropout_prob = dropout_prob
        self.hidden_dim = hidden_dim

        self.input_embedding = nn.Embedding(num_embeddings=len(self.vocab), embedding_dim=embedding_dim)
        self.image_embedding = nn.Sequential(nn.Linear(feat_dim, image_embedding_dim),
                                             nn.ReLU(),
                                             nn.Dropout(dropout_prob))
        self.image_embedding_avg = nn.Sequential(nn.Linear(feat_dim, image_embedding_dim),
                                                 nn.ReLU(),
                                                 nn.Dropout(dropout_prob))

        self.lstm_0 = nn.LSTMCell(input_size=hidden_dim+image_embedding_dim+embedding_dim,
                                  hidden_size=hidden_dim)
        self.ctx2att = nn.Linear(image_embedding_dim, attn_hidden_dim)
        self.att = Attention(rnn_size=hidden_dim, att_hid_size=attn_hidden_dim)
        self.lstm_1 = nn.LSTMCell(input_size=image_embedding_dim+hidden_dim, hidden_size=hidden_dim)
        self.output_embedding = nn.Linear(in_features=hidden_dim, out_features=len(self.vocab))

    def prepare_feat(self, input_feature, **kwargs):
        img_feat_avg, img_feat_attn = input_feature
        assert len(img_feat_avg) == len(img_feat_attn), 'batch size not consistent: {}, {}'.format(img_feat_avg.shape, img_feat_attn.shape)
        batch_size, attn_size, _ = img_feat_attn.shape
        img_feat_avg = self.image_embedding_avg(img_feat_avg)
        img_feat_attn = self.image_embedding(img_feat_attn)
        p_att_feats = self.ctx2att(img_feat_attn)
        return batch_size, (img_feat_avg, img_feat_attn, p_att_feats)

    def init_state(self, input_feature, **kwargs):
        (img_feat_avg, img_feat_attn, p_att_feats) = input_feature
        device = img_feat_avg.device
        batch_size = len(img_feat_avg)

        h_0 = torch.zeros((batch_size, self.hidden_dim)).to(device)
        c_0 = h_0
        return (h_0, c_0), (h_0, c_0)

    def reorder_feat(self, prepared_feature, new_order):
        feats = prepared_feature
        return [torch.index_select(f, dim=0, index=new_order) for f in feats]

    def reorder_state(self, state, new_order):
        (h_attn, c_attn), (h_lang, c_lang) = state
        states = (h_attn, c_attn, h_lang, c_lang)
        states = (torch.index_select(s, dim=0, index=new_order) for s in states)
        (h_attn, c_attn, h_lang, c_lang) = states
        return (h_attn, c_attn), (h_lang, c_lang)

    def step(self, input_feature, last_word_id_batch, last_state, **kwargs):
        (img_feat_avg, img_feat_attn, p_att_feats) = input_feature
        device = img_feat_avg.device
        batch_size, attn_size, dim_feat = img_feat_attn.shape
        if not torch.is_tensor(last_word_id_batch):
            last_word_id_batch = torch.LongTensor(last_word_id_batch).to(device)

        (h_attn_0, c_attn_0), (h_lang_0, c_lang_0) = last_state     # h_lang_0: (batch_size, hidden_dim)

        last_word_embedding = self.input_embedding(last_word_id_batch)  # (batch_size, embedding_dim)
        x_attn = torch.cat([h_lang_0, img_feat_avg, last_word_embedding], dim=1)
        h_attn_1, c_attn_1 = self.lstm_0(x_attn, (h_attn_0, c_attn_0))

        att = self.att(h_attn_1, img_feat_attn, p_att_feats)

        x_lang = torch.cat([att, h_attn_1], dim=1)
        h_lang_1, c_lang_1 = self.lstm_1(x_lang, (h_lang_0, c_lang_0))

        _output = F.dropout(h_lang_1, self.dropout_prob, self.training)
        output = self.output_embedding(_output)

        current_state = ((h_attn_1, c_attn_1), (h_lang_1, c_lang_1))

        return output, current_state, None  # output: (batch_size, vocab_size) not normalized


