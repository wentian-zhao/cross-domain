# used for retrieval

import os
import sys
sys.path.append('.')

import torch
torch.backends.cudnn.enabled = False

import util
from util.pipeline import BasePipeline, SupervisedPipeline
from util.loss import *
from config import *
from main.data import *
from main.r.model1 import *
from main.r.evaluation import *
from tqdm import tqdm
from torch.nn.utils.clip_grad import clip_grad_norm_


device = torch.device('cuda')


def _dist(x):
    return torch.pow(x, 2).sum(dim=1, keepdim=True).sqrt().sum()

def _l1(x):
    return torch.abs(x).sum(dim=1, keepdims=False).sum()


class RPipeline(SupervisedPipeline):
    def __init__(self):
        super().__init__()

    def add_arguments(self, parser):
        super().add_arguments(parser)
        parser.add_argument('-train_mode', type=str, default='source')  # source | target
        parser.add_argument('-source', type=str, default='coco')
        parser.add_argument('-target', type=str, default='')    # flickr30k | oxford102 | cub200 | tgif
        parser.add_argument('-saved_model', type=str, default='')

        parser.add_argument('-batch_size', type=int, default=256)

        parser.add_argument('-lr', default=2e-4, type=int)
        parser.add_argument('-scheduler_step_size', default=15, type=int)
        parser.add_argument('-scheduler_gamma', default=0.1, type=int)

        parser.add_argument('-vocab', type=str, default='vocab_merged_all.json')

    def init_data(self):
        v = os.path.join(vocab_path, self.args.vocab)
        print('loading vocab from {}'.format(v))
        vocab = util.load_custom(v)
        self.vocab = vocab

        batch_size = self.args.batch_size

        if self.args.action == 'train':
            if self.args.train_mode == 'source' or self.args.train_mode == 'target':
                dataset_name = self.args.source

                if dataset_name in {'coco', 'flickr30k', 'cub200', 'oxford102'}:
                    image_mode = 'fc'
                sent_feat_mode = 'none'

                train_dataloader = get_dataloader(dataset_name=dataset_name, vocab=vocab, split='train',
                                                  image_mode=image_mode, sent_feat_mode=sent_feat_mode,
                                                  iter_mode='single', max_sent_length=20,
                                                  sort=True, batch_size=batch_size, num_workers=1, shuffle=True)
                train_dataloader_1 = get_dataloader(dataset_name=dataset_name, vocab=vocab, split='train',
                                                  image_mode=image_mode, sent_feat_mode=sent_feat_mode,
                                                  iter_mode='single', max_sent_length=20,
                                                  sort=True, batch_size=batch_size, num_workers=1, shuffle=True)
                test_dataloader = get_dataloader(dataset_name=dataset_name, vocab=vocab, split='test',
                                                 image_mode=image_mode, sent_feat_mode=sent_feat_mode,
                                                 iter_mode='single', max_sent_length=20,
                                                  sort=True, batch_size=batch_size, num_workers=0)
            self.train_dataloader = train_dataloader
            self.train_dataloader_1 = train_dataloader_1
            self.test_dataloader = test_dataloader
        elif self.args.action == 'test':
            target_dataset = self.args.target
            test_dataloader = get_dataloader(dataset_name=target_dataset, vocab=vocab, split='train', image_mode='fc',
                                              iter_mode='single', max_sent_length=20,
                                              sort=True, batch_size=batch_size, num_workers=1, shuffle=True)
            self.test_dataloader = test_dataloader

    def init_model(self, state_dict=None):
        batch_size = self.args.batch_size

        vocab = self.vocab
        train_mode = self.args.train_mode

        if self.args.source in {'coco', 'flickr30k', 'cub200', 'oxford102'}:
            image_feat_size = 2048

        # model = VSE1(img_dim=image_feat_size, sent_dim=4096)
        model = VSE2(img_dim=image_feat_size, embed_size=512, vocab_size=len(self.vocab), word_dim=300, num_layers=1,
                     use_dict=True)
        model.to(device)

        # optimizer = torch.optim.Adam(lr=self.args.lr, params=model.get_params(train_mode))
        optimizer = torch.optim.Adam(lr=self.args.lr, params=model.get_params(train_mode))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.scheduler_step_size,
                                                    gamma=self.args.scheduler_gamma)

        criterion_rank = ContrastiveLoss(sim_func=model.get_sim_func(), margin=0.2, max_violation=True)
        criterion_rank.cuda()
        self.criterion_rank = criterion_rank

        if state_dict is not None:
            model.load_state_dict(state_dict['model'])
            scheduler.load_state_dict(state_dict['scheduler'])
            # try:
            #     optimizer.load_state_dict(state_dict['optimizer'])
            # except:
            #     print('optimizer load failed')

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train_epoch(self):
        train_dataloader = self.train_dataloader
        model = self.model
        optimizer, scheduler = self.optimizer, self.scheduler
        criterion_rank = self.criterion_rank
        train_mode = self.args.train_mode

        self.scheduler.step()

        if (not hasattr(self, 'initial_test')):
            self.test_epoch()
            self.initial_test = True

        # if self.args.train_mode == 'target':
        metrics, result = self.evaluate(self.model, self.train_dataloader_1, save_result=True)
        self.train_dataloader.dataset.set_retrieve_result(result, k=5)

        for i, batch_data in tqdm(enumerate(train_dataloader), ncols=64, total=len(train_dataloader)):
            model.train()
            optimizer.zero_grad()

            image_ids, image_feats, sent_ids, all_tokens, all_lengths, all_raw = \
                [batch_data[key] for key in ['image_id', 'feat_fc', 'sent_id', 'token', 'length', 'raw']]
            images = torch.Tensor(image_feats).to(device)
            tokens = torch.LongTensor(all_tokens).to(device)
            lengths = all_lengths
            img_emb, cap_emb = model.forward(images, tokens, lengths, mode=train_mode)

            loss_dict = {}
            loss_rank = criterion_rank.forward(img_emb, cap_emb)
            loss_dict['loss/rank'] = loss_rank

            loss = 0.
            for l in loss_dict.values():
                loss += l
            loss_dict['loss/total'] = loss

            loss.backward()
            clip_grad_norm_(model.parameters(), 2.)
            optimizer.step()

            self.global_step += 1

            if self.global_step % 2 == 0:
                for name, l in loss_dict.items():
                    scalar = l.detach().cpu().numpy()
                    self.writer.add_scalar(name, scalar, global_step=self.global_step)
                    # print(name, scalar)

    def test_epoch(self):
        print('{ -------- test --------')
        metrics, result = self.evaluate(self.model, self.test_dataloader, save_result=True)
        print('} -------- test -------- done')
        for tag, value in metrics.items():
            self.writer.add_scalar('result/' + tag, value, global_step=self.global_step)

    def evaluate(self, model, test_dataloader, save_result=False):
        model.eval()

        dataset_name = test_dataloader.dataset.dataset_name
        split = test_dataloader.dataset.split

        sim_func = model.get_sim_func()

        metrics = {}
        result = {}
        all_image_emb, all_sent_emb, all_image_ids, all_sent_ids = self.encode_data(model, test_dataloader)
        score = 0.
        (r1, r5, r10, medr, meanr), (_, top_k, sim_i2t) = i2t(all_image_emb, all_sent_emb, all_image_ids, all_sent_ids,
                                                     sim_func=sim_func, k=10, return_ranks=True, return_sim=True)
        print('i2t', (r1, r5, r10, medr, meanr))
        score += sum((r1, r5, r10))
        metrics.update(dict(zip(['i2t_r1', 'i2t_r5', 'i2t_r10', 'i2t_medr', 'i2t_meanr'], (r1, r5, r10, medr, meanr))))
        result['i2t'] = top_k

        (r1, r5, r10, medr, meanr), (_, top_k, sim_t2i) = t2i(all_image_emb, all_sent_emb, all_image_ids, all_sent_ids,
                                                     sim_func=sim_func, k=10, return_ranks=True, return_sim=True)
        print('t2i', (r1, r5, r10, medr, meanr))
        score += sum((r1, r5, r10))
        metrics.update(dict(zip(['t2i_r1', 't2i_r5', 't2i_r10', 't2i_medr', 't2i_meanr'], (r1, r5, r10, medr, meanr))))
        result['t2i'] = top_k

        print('score:', score)
        metrics['score'] = score

        result_i2t = []
        for image_id, sent_ids in result['i2t'].items():
            # gt_sents = [s.raw for s in dataset.get_caption_item_by_image_id(image_id).sentences][:5]
            # r_sents = [dataset.get_sentence_item(sent_id).raw for sent_id in sent_ids][:5]
            result_i2t.append({
                'image_id': image_id, 'sent_ids': sent_ids, 'sim': sim_i2t[image_id]
            })
        result_t2i = []
        for sent_id, image_ids in result['t2i'].items():
            result_t2i.append({
                'sent_id': sent_id, 'image_ids': image_ids, 'sim': sim_t2i[sent_id]
            })

        result = {'dataset': dataset_name, 'split': split, 'metrics': metrics,
                              't2i': result_t2i, 'i2t': result_i2t}

        if save_result:
            print('saving result...')
            dataset = test_dataloader.dataset

            result_path = os.path.join(self.save_folder, 'retrieve_result')
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            util.dump_custom(result,
                             os.path.join(result_path, 'r_result_{}_{}_{}.json'.format(dataset_name, split, self.epoch)), indent=4)
            with open(os.path.join(self.save_folder, 'metrics_{}_{}'.format(split, self.epoch)), 'w') as f:
                json.dump(metrics, f)
        return metrics, result

    def encode_data(self, model, test_dataloader):
        all_image_emb = []
        all_sent_emb = []
        all_image_ids = []
        all_sent_ids = []
        print('encode data...')
        for i, batch_data in enumerate(test_dataloader):
            image_ids, image_feats, sent_ids, all_tokens, all_lengths, all_raw = \
                [batch_data[key] for key in ['image_id', 'feat_fc', 'sent_id', 'token', 'length', 'raw']]

            # image_ids, sent_ids = batch_data['image_id'], batch_data['sent_id']
            # image_feat, sent_feat = (batch_data[key] for key in ['feat_fc', 'feat_sent'])
            # image_feat = torch.Tensor(image_feat).to(device)
            # sent_feat = torch.Tensor(sent_feat).to(device)
            # img_emb, cap_emb = model.forward(image_feat, sent_feat)

            images = torch.Tensor(image_feats).to(device)
            tokens = torch.LongTensor(all_tokens).to(device)
            lengths = all_lengths
            img_emb, cap_emb = model.forward(images, tokens, lengths, mode=self.args.train_mode)

            image_emb = img_emb.detach().cpu().numpy()
            sent_emb = cap_emb.detach().cpu().numpy()

            all_image_emb.extend(image_emb)
            all_sent_emb.extend(sent_emb)
            all_image_ids.extend(image_ids)
            all_sent_ids.extend(sent_ids)
        all_image_emb = np.array(all_image_emb)
        all_sent_emb = np.array(all_sent_emb)
        all_image_ids = np.array(all_image_ids)
        all_sent_ids = np.array(all_sent_ids)
        print('encode data done')
        return all_image_emb, all_sent_emb, all_image_ids, all_sent_ids

    def get_state_dict(self):
        return {'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(), 'scheduler': self.scheduler.state_dict(),
                'epoch': self.epoch, 'global_step': self.global_step}


p = RPipeline()
p.run()
