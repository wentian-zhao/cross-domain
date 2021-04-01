import os
import random
import re
import sys
import traceback

import h5py
import numpy as np
from functools import lru_cache
from collections import namedtuple
import scipy
import scipy.sparse

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataloader import default_collate

import util
from config import *
from util import *

# image_item.image_id, feat, sent_ids, all_tokens, sent_lengths, [sent.raw for sent in sents]
BatchData = namedtuple('BatchData', ['image_id', 'image_feat', 'sent_id', 'token', 'sent_length', 'raw'])

open_files = dict()


class CaptionDataset(util.BaseCaptionDataset):
    def __init__(self, **kwargs):
        self.iter_mode = kwargs.get('iter_mode', 'single')

        # this does not include <start> and <end> token
        self.max_sent_length = kwargs['max_sent_length']
        self.image_mode = kwargs.get('image_mode', 'fc')
        self.use_restval = kwargs.get('use_restval', True)

        self.sent_feat_mode = kwargs.get('sent_feat_mode', None)

        assert self.iter_mode in ['single', 'group', 'retrieved', 'random']
        #         assert self.image_mode in ['none', 'fc', 'att']

        super().__init__(**kwargs)

        self.feat_file_fc = os.path.join(feat_path, self.dataset_name, '{}_fc.h5'.format(self.dataset_name))
        self.feat_file_att = os.path.join(feat_path, self.dataset_name, '{}_att.h5'.format(self.dataset_name))
        self.feat_file_sent = os.path.join(data_path, 'sentence_embedding', 'sent_emb_{}.h5'.format(self.dataset_name))
        self.feat_folder_att_sparse = os.path.join(feat_path, self.dataset_name, '{}_att_sparse'.format(self.dataset_name))

        for sent in self.sentence_list:
            sent.token_ids = [self.vocab.get_index(w) for w in sent.words]

        if self.iter_mode == 'retrieved':
            self.set_retrieve_result(kwargs['retrieve_result'], kwargs['k'])

    @staticmethod
    @lru_cache(maxsize=5000)
    def read_h5(file, id):
        try:
            # print('read', id, 'pid:', os.getpid())
            if file not in open_files:
                open_files[file] = h5py.File(file, 'r', libver='latest', swmr=True)
            f = open_files[file]
            arr = np.array(f[id]).astype(np.float32)
            return arr
        except:
            for _ in range(5):
                print('retry {}'.format(_))
                try:
                    open_files[file] = h5py.File(file, 'r', libver='latest', swmr=True)
                    f = open_files[file]
                    arr = np.array(f[id]).astype(np.float32)
                    return arr
                except:
                    if _ == 4:
                        traceback.print_exc()
                        raise Exception()

    @staticmethod
    @lru_cache(maxsize=5000)
    def read_sparse(feat_folder_att_sparse, id):
        f_sparse = os.path.join(feat_folder_att_sparse, '{}.npz'.format(id))
        a = np.array(scipy.sparse.load_npz(f_sparse).todense())
        return a

    @staticmethod
    def read_np(filename):
        return np.load(filename)

    def read_image_feat(self, image_id):
        """
        :param image_id:
        :param n_repeat:
        :return: (n_repeat, 2048), or (n_repeat, 196, 2048)
        """
        data = {}

        if self.image_mode == 'fc' or self.image_mode == 'att':
            feat = self.read_h5(self.feat_file_fc, str(image_id))
            data['feat_fc'] = feat
        if self.image_mode == 'att':
            feat = self.read_h5(self.feat_file_att, str(image_id))
            feat = feat.reshape(-1, feat.shape[-1])
            data['feat_att'] = feat
        return data

    def read_sent_feat(self, sent_id):
        data = {}
        if self.sent_feat_mode == 'infersent':
            feat = self.read_h5(self.feat_file_sent, str(sent_id))
            data['feat_sent'] = feat
        return data

    @staticmethod
    @lru_cache(maxsize=5)
    def load_json(filename):
        return util.load_custom(filename)

    def load(self):
        dataset_file = os.path.join(annotation_path, 'dataset_{}.json'.format(self.dataset_name))
        obj = self.load_json(dataset_file)
        caption_item_list = obj['caption_item']
        if self.use_restval:
            for caption_item in caption_item_list:
                if caption_item.split == 'restval':
                    caption_item.split = 'train'
        # if self.dataset_name == 'tgif':
        # 	print('before: {}', len(obj['caption_item']))
        # 	for caption_item in obj['caption_item']:
        # 		caption_item.image.image_filename += '_0.jpg'
        # 	feat_file_fc = os.path.join(feat_path, self.dataset_name, '{}_fc.h5'.format(self.dataset_name))
        # 	f = h5py.File(feat_file_fc, 'r')
        # 	keys = set(f.keys())
        # 	obj['caption_item'] = list(filter(lambda x: x.image.image_filename in keys, obj['caption_item']))
        # 	print('after: {}', len(obj['caption_item']))
        self.caption_item_list = obj['caption_item']

    def __len__(self):
        if self.iter_mode == 'single':
            return len(self.image_sentence_pair_split[self.split])
        elif self.iter_mode == 'retrieved':
            return len(self.image_sentence_pair_split[self.split])
        elif self.iter_mode == 'random':
            return len(self.image_sentence_pair_split[self.split])

    def __getitem__(self, index):
        if self.iter_mode == 'single':
            pair = self.image_sentence_pair_split[self.split][index]
            image_item = pair.image
            sent = pair.sentence
        elif self.iter_mode == 'retrieved':  # 1 image -> 1 sentence
            pair = self.image_sentence_pair_split[self.split][index]
            image_item = pair.image
            sent = pair.sentence
        elif self.iter_mode == 'random':
            pair1, pair2 = random.sample(self.image_sentence_pair_split[self.split], k=2)
            image_item = pair1.image
            sent = pair2.sentence

        fixed_length = self.max_sent_length + 2  # with <start> and <end>

        sent_id, raw = sent.sentence_id, sent.raw
        tokens = [Vocabulary.start_token_id] + sent.token_ids + [Vocabulary.end_token_id]
        sent_length = min(len(tokens), fixed_length)
        fixed_length_tokens = np.zeros(fixed_length)
        fixed_length_tokens[:sent_length] = tokens[:sent_length]

        # faster, use np.ndarray
        data = {
            'image_id': image_item.image_id,
            'sent_id': sent_id, 'token': fixed_length_tokens, 'length': sent_length, 'raw': raw
        }

        if self.dataset_name == 'tgif':
            feat = self.read_image_feat(image_item.image_filename + '_0.jpg')
        else:
            feat = self.read_image_feat(image_item.image_filename)
        data.update(feat)
        feat = self.read_sent_feat(sent_id)
        data.update(feat)

        if self.iter_mode == 'retrieved':
            sim = self.similarity_dict[(image_item.image_id, sent_id)]
            data['sim'] = sim

        return data

        # to avoid memory leak, use torch.Tensor
        # return image_item.image_id, torch.Tensor(feat), torch.LongTensor(tokens_fixedlen), sent_length, sent.raw

    def set_retrieve_result(self, retrieve_result, k=5):
        """
        :param retrieve_result: {'i2t': {1: [2, 3, 4, 5]}, 't2i': {10: [4, 5, 6, 7]}}
        :return:
        """

        # if isinstance(retrieve_result['i2t'], list):
        #     d = dict((i['image_id'], i['sent_ids']) for i in retrieve_result['i2t'])
        #     retrieve_result['i2t'] = d
        #     d = dict((i['sent_id'], i['image_ids']) for i in retrieve_result['t2i'])
        #     retrieve_result['t2i'] = d
        #
        # r_result = []
        # d = defaultdict(list)
        #
        # for image_id, sent_ids in retrieve_result['i2t'].items():
        #     assert len(sent_ids) >= k
        #     for sent_id in sent_ids[:k]:
        #         r_result.append(ImageSentencePair(self.image_id_map[image_id], self.sentence_id_map[sent_id]))
        #         d[image_id].append(sent_id)
        # for sent_id, image_ids in retrieve_result['t2i'].items():
        #     assert len(image_ids) >= k
        #     for image_id in image_ids[:k]:
        #         r_result.append(ImageSentencePair(self.image_id_map[image_id], self.sentence_id_map[sent_id]))
        #         d[image_id].append(sent_id)
        # print('total {} pair in retrieve results'.format(len(r_result)))
        #
        # for image_id, sent_ids in d.items():
        #     item = CaptionItem(image=self.image_id_map[image_id], sentences=[self.sentence_id_map[i] for i in sent_ids],
        #                        split=self.split)
        #     self.image_id_map_2[image_id] = item
        #
        # self.image_sentence_pair_split[self.split] = r_result

        similarity_dict = {}    # (image_id, sent_id) -> sim

        _r_result = defaultdict(set)
        for item in retrieve_result['i2t']:
            image_id = item['image_id']
            for i, sent_id in enumerate(item['sent_ids']):
                sim = item['sim'][i]
                _r_result[image_id].add((sent_id, sim))
        for item in retrieve_result['t2i']:
            sent_id = item['sent_id']
            for i, image_id in enumerate(item['image_ids']):
                sim = item['sim'][i]
                _r_result[image_id].add((sent_id, sim))

        r_result = []
        for image_id, value in _r_result.items():
            sent_list = list(value)
            sent_list.sort(key=lambda x: x[1], reverse=True)
            sent_list = sent_list[:k]
            for sent_id, sim in sent_list:
                r_result.append(ImageSentencePair(self.image_id_map[image_id], self.sentence_id_map[sent_id]))
                similarity_dict[(image_id, sent_id)] = sim
            self.image_id_map_2[image_id] = CaptionItem(image=self.image_id_map[image_id],
                                                        sentences=[self.sentence_id_map[i] for i, _ in sent_list],
                                                        split=self.split)

        print('total {} pair in retrieve results'.format(len(r_result)))

        self.image_sentence_pair_split[self.split] = r_result
        self.similarity_dict = similarity_dict

    def shuffle(self, group=True):
        print('shuffle {}, group = {}'.format(self.dataset_name, group))
        if group:
            all_image_sentence_pair = self.image_sentence_pair_split[self.split]
            all_image_sentence_pair.sort(key=lambda x: x.image.image_id)
            groups = []
            last_image_id = None
            for pair in all_image_sentence_pair:
                if pair.image.image_id != last_image_id:
                    groups.append([pair])
                else:
                    groups[-1].append(pair)
                last_image_id = pair.image.image_id
            random.shuffle(groups)
            new_list = []
            for group in groups:
                new_list.extend(group)
            self.image_sentence_pair_split[self.split] = new_list
        else:
            random.shuffle(self.image_sentence_pair_split[self.split])

    def _get_sub_collate_fn(self):
        def collate_ndarray(data):
            return np.array(data)

        d = {'image_id': collate_ndarray,
             'sent_id': collate_ndarray, 'token': collate_ndarray, 'length': collate_ndarray, 'raw': collate_ndarray}

        if self.image_mode == 'fc':
            d['feat_fc'] = collate_ndarray
        if self.image_mode == 'att':
            d.update({'feat_fc': collate_ndarray, 'feat_att': collate_ndarray})

        if self.sent_feat_mode == 'infersent':
            d['feat_sent'] = collate_ndarray

        if self.iter_mode == 'retrieved':
            d['sim'] = collate_ndarray

        return d


def get_collate_fn(sub_collate_fn, sort=False):
    def _collate_fn(batch):
        _ = time.time()
        collected = defaultdict(list)
        for data_dict in batch:
            assert data_dict.keys() == sub_collate_fn.keys(), '{}, {}'.format(data_dict.keys(), sub_collate_fn.keys())
            for field in sub_collate_fn:
                collected[field].append(data_dict[field])
        for field in collected.keys():
            collected_field_data = collected[field]
            collected[field] = sub_collate_fn[field](collected_field_data)

        if sort:
            all_lengths = collected['length']
            sorted_index = np.argsort(all_lengths)[::-1]
            for field in collected.keys():
                collected[field] = collected[field][sorted_index]
        return collected  # dict

    return _collate_fn


def get_dataloader(**kwargs):
    dataset = CaptionDataset(**kwargs)
    collate_fn = get_collate_fn(dataset._get_sub_collate_fn(), sort=kwargs.get('sort', False))

    dataloader_args = {'collate_fn': collate_fn}
    for key, value in kwargs.items():
        if key in ['batch_size', 'shuffle', 'num_workers', 'pin_memory', 'sampler']:
            dataloader_args[key] = value
    dataloader = DataLoader(dataset, **dataloader_args)
    return dataloader

