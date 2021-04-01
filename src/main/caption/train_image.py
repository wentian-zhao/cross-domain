import os
import sys
sys.path.append(os.getcwd())
sys.path.append('coco-caption')

import torch
import numpy as np

import util
from util import *
import util.reward
from config import *
from main.data import *
from main.caption.model import LSTMLanguageModel, TopDownAttnModel

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class ImageCaptionPipeline(SupervisedPipeline):
    def __init__(self):
        super().__init__()

    def add_arguments(self, parser):
        super().add_arguments(parser)
        parser.add_argument('-vocab', default='vocab_coco.json', type=str)
        parser.add_argument('-train_mode', default='source', type=str)      # source | target
        parser.add_argument('-source', default='coco', type=str)
        parser.add_argument('-target', default='', type=str)

        parser.add_argument('-retrieve_result', default='', type=str)
        parser.add_argument('-k', default=5, type=int)

        parser.add_argument('-batch_size', default=128, type=int)
        parser.add_argument('-num_workers', default=1, type=int)

        parser.add_argument('-lr', default=5e-4, type=float)
        parser.add_argument('-grad_clip', default=0.1, type=float)
        parser.add_argument('-scheduler_step_size', default=3, type=int)
        parser.add_argument('-scheduler_gamma', default=0.8, type=float)

        parser.add_argument('-ss_start', default=0, type=int)
        parser.add_argument('-ss_increase_every', default=5, type=int)
        parser.add_argument('-ss_prob_increase', default=0.05, type=float)
        parser.add_argument('-ss_prob_max', default=0.25, type=float)

        parser.add_argument('-sc_after', default=10, type=int)
        parser.add_argument('-reward_weight_bleu', default=0., type=float)
        parser.add_argument('-reward_weight_cider', default=1., type=float)

        parser.add_argument('-beam_size', default=5, type=int)

        parser.add_argument('-pretrained_model', default='', type=str, help='path to pretrained model (does not load scheduler)')

    def init_data(self):
        v = os.path.join(vocab_path, self.args.vocab) if not os.path.exists(self.args.vocab) else self.args.vocab
        print('loading vocabulary from {}'.format(v))
        vocab = load_custom(v)
        self.vocab = vocab

        if self.args.train_mode == 'source':
            dataset_name = self.args.source
        elif self.args.train_mode == 'target':
            dataset_name = self.args.target

        batch_size = self.args.batch_size
        num_workers = self.args.num_workers

        max_sent_length = {'coco': 18, 'flickr30k': 22, 'oxford102': 20, 'cub200': 20,}[dataset_name]
        self.max_sent_length = max_sent_length

        if self.args.action == 'train':
            if self.args.train_mode == 'source':
                self.train_dataloader = get_dataloader(dataset_name=dataset_name, vocab=vocab, split='train', image_mode='att',
                                                       iter_mode='single', max_sent_length=max_sent_length, sort=False,
                                                       batch_size=batch_size, num_workers=num_workers, shuffle=False)
            elif self.args.train_mode == 'target':
                retrieve_result = load_custom(self.args.retrieve_result)
                assert retrieve_result['dataset'] == dataset_name, \
                    'retrieve result is {}, target is {}'.format(retrieve_result['dataset'], dataset_name)
                print('using retrieve result from {}, k={}'.format(self.args.retrieve_result, self.args.k))
                self.train_dataloader = get_dataloader(dataset_name=dataset_name, vocab=vocab, split='train',
                                                       image_mode='att',
                                                       iter_mode='retrieved', retrieve_result=retrieve_result, k=5,
                                                       max_sent_length=max_sent_length, sort=False,
                                                       batch_size=batch_size, num_workers=num_workers, shuffle=False)

        if self.args.action == 'train' or self.args.action == 'test':
            self.test_dataloader = get_dataloader(dataset_name=dataset_name, vocab=vocab, split='test', image_mode='att',
                                                  iter_mode='single', max_sent_length=max_sent_length, sort=False,
                                                  batch_size=batch_size, num_workers=0, shuffle=False)

    def init_model(self, state_dict=None):
        saved_args = state_dict.get('args', None) if state_dict is not None else None
        if saved_args is None or saved_args.train_mode != self.args.train_mode:
            self.global_step = 0
            self.epoch = 1

        model = TopDownAttnModel(feat_dim=2048, vocab=self.vocab)
        model.to(device)
        optimizer = torch.optim.Adam([
            {'params': model.parameters(), 'lr': self.args.lr}
        ])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.scheduler_step_size,
                                                    gamma=self.args.scheduler_gamma)

        if state_dict is not None:
            print('loading saved model')
            model.load_state_dict(state_dict['model'])
            optimizer.load_state_dict(state_dict['optimizer'])
            scheduler.load_state_dict(state_dict['scheduler'])

        if len(self.args.pretrained_model) > 0:
            print('loading pretrained model from ', self.args.pretrained_model)
            state_dict = torch.load(self.args.pretrained_model)
            model.load_state_dict(state_dict['model'])
            optimizer.load_state_dict(state_dict['optimizer'])

        for param_group in optimizer.param_groups:
            param_group['lr'] = self.args.lr

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler


    def train_epoch(self):
        if self.args.train_mode == 'target' and self.epoch == 1:
            self.epoch = 0
            # self.test()
            self.epoch = 1

        for param_group in self.optimizer.param_groups:
            print('lr:', param_group['lr'])

        scheduler, model, optimizer = self.scheduler, self.model, self.optimizer

        scheduler.step()

        ss_start, ss_increase_every, ss_prob_increase, ss_prob_max = \
            self.args.ss_start, self.args.ss_increase_every, self.args.ss_prob_increase, self.args.ss_prob_max
        ss_prob = 0
        if self.args.ss_start >= 0:
            ss_prob = min(ss_prob_max, (max(0, self.epoch - ss_start) // ss_increase_every) * ss_prob_increase )

        sc_after = self.args.sc_after
        self_critical_flag = self.epoch >= sc_after and sc_after >= 0
        max_sample_seq_len = self.max_sent_length
        max_sent_length = self.max_sent_length

        # shuffle dataset
        if not self_critical_flag:
            self.train_dataloader.dataset.shuffle(group=True)
        else:
            self.train_dataloader.dataset.shuffle(group=False)

        if self_critical_flag:
            dataset_name = self.train_dataloader.dataset.dataset_name
            df = '../data/preprocessed/ngram_{}_train_words.p'.format(dataset_name)
            util.reward.init_scorer(df=df)

        model.train(True)

        timer = Timer()
        timer.tick('step')

        for i, batch_data in tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), ncols=64):
            timer.tick('loop')
            timer.tick('prep')

            self.global_step += 1

            image_id, feat_fc, feat_att, sent_ids, tokens_fixedlen, sent_length, raw = \
                [batch_data[key] for key in ['image_id', 'feat_fc', 'feat_att', 'sent_id', 'token', 'length', 'raw']]

            # if self.args.train_mode == 'target':
            #     sim = batch_data['sim']

            feat_fc, feat_att = (torch.Tensor(np.array(f)).to(device) for f in (feat_fc, feat_att))
            tokens = torch.LongTensor(np.array(tokens_fixedlen)).to(device)
            sent_length = torch.LongTensor(sent_length)

            timer.tock('prep')

            optimizer.zero_grad()

            log_dict = {'ss_prob': ss_prob}
            log_flag = i % 20 == 0

            timer.tick('train')
            if not self_critical_flag:
                token_input = tokens[:, :-1].contiguous()
                token_target = tokens[:, 1:].contiguous()

                outputs = model.forward(input_feature=(feat_fc, feat_att), input_sentence=token_input, ss_prob=ss_prob)

                loss = util.masked_cross_entropy(outputs, token_target, sent_length - 1)

                loss.backward()
                util.clip_gradient(optimizer, self.args.grad_clip)
                optimizer.step()

                if log_flag:
                    log_dict['loss/loss_crossentropy'] = loss.detach().cpu().numpy()
            else:
                sample_logprob, sample_seq, _ = model.sample(input_feature=(feat_fc, feat_att), max_length=max_sample_seq_len + 1,
                                                             sample_max=False)
                model.train(False)
                with torch.no_grad():
                    greedy_logprob, greedy_seq, _ = model.sample(input_feature=(feat_fc, feat_att), max_length=max_sample_seq_len + 1,
                                                                 sample_max=True)
                model.train(True)

                _greedy_sents = [' '.join(self.vocab.get_word(i) for i in sent if i > 0) for sent in greedy_seq]

                train_dataloader = self.train_dataloader
                gts_raw = []
                for _i, id in enumerate(image_id):
                    g = []
                    # for s in train_dataloader.dataset.get_caption_item_by_image_id(id).sentences:     # use all retrieved sentences
                    for s in [train_dataloader.dataset.get_sentence_item(sent_ids[_i])]:                # use current sentence
                        words = s.words + [util.Vocabulary.end_token]
                        g.append(' '.join(words[:max_sent_length]))
                        # if len(s.words) < max_sent_length:
                        #     g.append(' '.join(s.words + [util.Vocabulary.end_token]))
                        # else:
                        #     g.append(' '.join(s.words[:max_sent_length]))
                    gts_raw.append(g)

                timer.tick('reward')
                reward = util.reward.get_self_critical_reward(sample_seq, greedy_seq, gts_raw,
                                                              weights={'bleu': self.args.reward_weight_bleu,
                                                                       'cider': self.args.reward_weight_cider},
                                                              vocab=self.vocab)
                timer.tock('reward')
                loss = util.reward.rl_criterion(log_prob=sample_logprob, generated_seq=sample_seq, reward=reward)

                loss.backward()
                util.clip_gradient(optimizer, self.args.grad_clip)
                optimizer.step()

                if log_flag:
                    avg_reward = np.mean(reward[:, 0])
                    sc_loss = loss.detach().cpu().numpy()
                    log_dict['loss/self_critical'] = sc_loss
                    log_dict['avg_reward'] = avg_reward

            timer.tock('train')

            if log_flag:
                for key, value in log_dict.items():
                    self.writer.add_scalar(key, value, self.global_step)

                for i, param_group in enumerate(optimizer.param_groups):
                    self.writer.add_scalar('lr_{}'.format(i), param_group['lr'], self.global_step)

            timer.tock('loop')
            timer.tock('step')
            self.writer.add_scalars(main_tag='time', tag_scalar_dict=timer.get_time(), global_step=self.global_step)
            timer.clear(); timer.tick('step')

    def test_epoch(self):
        # if self.epoch % 3 != 0:     # test every 3 epoch
        #     return

        model = self.model
        test_dataloader = self.test_dataloader
        vocab = self.vocab

        result_generator = util.COCOResultGenerator()
        model.train(False)
        beam_size = self.args.beam_size

        for i, batch_data in tqdm(enumerate(test_dataloader), total=len(test_dataloader), ncols=64):
            image_ids, feat_fc, feat_att, sent_ids, tokens_fixedlen, sent_length, raw = \
                [batch_data[key] for key in ['image_id', 'feat_fc', 'feat_att', 'sent_id', 'token', 'length', 'raw']]

            batch_size = len(image_ids)
            # feats = torch.Tensor(feats).to(device)

            for batch_index in range(batch_size):
                image_id = image_ids[batch_index]
                result_generator.add_annotation(image_id, raw[batch_index])

                if result_generator.has_output(image_id):
                    continue

                _feat_fc = torch.Tensor(feat_fc[batch_index]).unsqueeze(0).to(device)
                _feat_att = torch.Tensor(feat_att[batch_index]).unsqueeze(0).to(device)

                log_prob_seq, word_id_seq, _ = model.sample_beam(input_feature=(_feat_fc, _feat_att),
                                                                 max_length=20, beam_size=beam_size)

                words = util.trim_generated_tokens(word_id_seq)
                words = [vocab.get_word(i) for i in words]
                sent = ' '.join(words)
                result_generator.add_output(image_id, sent)

        ann_file = os.path.join(self.save_folder, 'annotation.json')
        result_file = os.path.join(self.save_folder, 'result_{}.json'.format(self.epoch))
        metric_file = os.path.join(self.save_folder, 'metrics.csv')
        result_generator.dump_annotation_and_output(ann_file, result_file)

        metrics, img_scores = util.eval(ann_file, result_file, return_imgscores=True)
        self.writer.add_scalars(main_tag='metric/', tag_scalar_dict=metrics, global_step=self.global_step)
        # with open(metric_file, 'a') as f:
        #     line = ['\"epoch {} step {}\"'.format(self.epoch, self.global_step)]
        #     for metric, value in metrics.items():
        #         line.append('\"{}:{:.6f}\"'.format(metric, value))
        #     f.write(', '.join(line) + '\n')
        self.save_results(metric_file, metrics)
        result_generator.add_img_scores(img_scores)
        result_generator.dump_output(result_file)

    def save_results(self, metric_file, metrics):
        lines = []
        if not os.path.exists(metric_file):
            first_line = ['epoch', 'step']
            for metric in metrics:
                first_line.append(metric)
            lines.append(','.join('{:<10}'.format(i) for i in first_line))
        else:
            with open(metric_file, 'r') as f:
                first_line = [i.strip() for i in f.readline().split(',')]
        strs = []
        for i in first_line:
            if i == 'epoch':
                strs.append('{:<10}'.format(self.epoch))
            elif i == 'step':
                strs.append('{:<10}'.format(self.global_step))
            else:
                strs.append('{:<10.6f}'.format(metrics[i]))
        lines.append(','.join(strs))
        with open(metric_file, 'a') as f:
            f.writelines([i + '\n' for i in lines])

    def get_state_dict(self):
        return {'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'vocab': self.vocab,
                'args': self.args}


p = ImageCaptionPipeline()
p.run()

