from __future__ import print_function
import os
import pickle

import time
import numpy as np
from collections import OrderedDict, defaultdict

#
# class AverageMeter(object):
#     """Computes and stores the average and current value"""
#
#     def __init__(self):
#         self.reset()
#
#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0
#
#     def update(self, val, n=0):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / (.0001 + self.count)
#
#     def __str__(self):
#         """String representation for logging
#         """
#         # for values that should be recorded exactly e.g. iteration number
#         if self.count == 0:
#             return str(self.val)
#         # for stats
#         return '%.4f (%.4f)' % (self.val, self.avg)
#
#
# class LogCollector(object):
#     """A collection of logging objects that can change from train to val"""
#
#     def __init__(self):
#         # to keep the order of logged variables deterministic
#         self.meters = OrderedDict()
#
#     def update(self, k, v, n=0):
#         # create a new meter if previously not recorded
#         if k not in self.meters:
#             self.meters[k] = AverageMeter()
#         self.meters[k].update(v, n)
#
#     def __str__(self):
#         """Concatenate the meters in one log line
#         """
#         s = ''
#         for i, (k, v) in enumerate(self.meters.iteritems()):
#             if i > 0:
#                 s += '  '
#             s += k + ' ' + str(v)
#         return s
#
#     def tb_log(self, tb_logger, prefix='', step=None):
#         """Log using tensorboard
#         """
#         for k, v in self.meters.iteritems():
#             tb_logger.log_value(prefix + k, v.val, step=step)
#
#
# def encode_data(model, data_loader, log_step=10, logging=print):
#     """Encode all images and captions loadable by `data_loader`
#     """
#     batch_time = AverageMeter()
#     val_logger = LogCollector()
#
#     # switch to evaluate mode
#     model.val_start()
#
#     end = time.time()
#
#     # numpy array to keep all the embeddings
#     img_embs = None
#     cap_embs = None
#     for i, (images, captions, lengths, ids) in enumerate(data_loader):
#         # make sure val logger is used
#         model.logger = val_logger
#
#         # compute the embeddings
#         img_emb, cap_emb = model.forward_emb(images, captions, lengths,
#                                              volatile=True)
#
#         # initialize the numpy arrays given the size of the embeddings
#         if img_embs is None:
#             img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
#             cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))
#
#         # preserve the embeddings by copying from gpu and converting to numpy
#         img_embs[ids] = img_emb.data.cpu().numpy().copy()
#         cap_embs[ids] = cap_emb.data.cpu().numpy().copy()
#
#         # measure accuracy and record loss
#         model.forward_loss(img_emb, cap_emb)
#
#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()
#
#         if i % log_step == 0:
#             logging('Test: [{0}/{1}]\t'
#                     '{e_log}\t'
#                     'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                     .format(
#                         i, len(data_loader), batch_time=batch_time,
#                         e_log=str(model.logger)))
#         del images, captions
#
#     return img_embs, cap_embs
#
#
#

from tqdm import tqdm


def i2t(images, captions, image_ids, caption_ids, sim_func, k=5, return_ranks=False, return_sim=False):
    """

    :param images: (N, dim_emb)
    :param captions: (N, dim_emb)
    :param image_ids: list of length N
    :param caption_ids: list of length N
    :param sim_func: callable, sim_func(x, y), returns similarity
    :param npts:
    :param return_ranks:
    :return:
    """
    assert len(images) == len(image_ids)
    assert len(captions) == len(caption_ids)
    assert len(images) == len(captions)

    unique_image_ids = defaultdict(list)    # image_id: [index1, index2, ...]
    for i, image_id in enumerate(image_ids):
        unique_image_ids[image_id].append(i)

    ranks = {}
    retrieve_result = {}
    all_sim = {}

    for _, (image_id, image_index_list) in tqdm(enumerate(unique_image_ids.items()), total=len(unique_image_ids), ncols=64):
        index = image_index_list[0]
        im_emb = images[index]      # (2048,)

        sim = sim_func(im_emb, captions)      # shape of (N,)
        retrieved_sent_index = np.argsort(sim)[::-1]
        retrieved_sent_id = caption_ids[retrieved_sent_index]

        correct_sent_id = [caption_ids[_index] for _index in image_index_list]
        sentence_ranks = []
        for sentence_id in correct_sent_id:
            rank = np.where(sentence_id == retrieved_sent_id)
            sentence_ranks.append(rank)
        min_rank = np.min(sentence_ranks)

        ranks[image_id] = min_rank
        retrieve_result[image_id] = retrieved_sent_id[:k].tolist()
        all_sim[image_id] = sim[retrieved_sent_index[:k]]
        del retrieved_sent_id

    ranks = np.array(list(ranks.values()))
    top_k = retrieve_result
    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        if return_sim:
            return (r1, r5, r10, medr, meanr), (ranks, top_k, all_sim)
        else:
            return (r1, r5, r10, medr, meanr), (ranks, top_k)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, captions, image_ids, caption_ids, sim_func, k=5, return_ranks=False, return_sim=False):
    """

    :param images:
    :param captions:
    :param image_ids:
    :param caption_ids:
    :param sim_func:
    :param k:
    :param return_ranks:
    :return:
    """
    assert len(images) == len(image_ids)
    assert len(captions) == len(caption_ids)
    # assert len(images) == len(captions)

    # remove duplicate
    original_image_id = image_ids
    _unique_image_id = set()
    _image_id = []
    _images = []
    for i, image_id in enumerate(image_ids):
        if image_id not in _unique_image_id:
            _image_id.append(image_id)
            _images.append(images[i])
            _unique_image_id.add(image_id)
    image_ids = np.array(_image_id)
    images = np.array(_images)

    ranks = {}
    retrieve_result = {}
    all_sim = {}

    for i, sentence_id in tqdm(enumerate(caption_ids), total=len(caption_ids), ncols=64):
        sent_emb = captions[i]

        sim = sim_func(sent_emb, images)
        retrieved_image_index = np.argsort(sim)[::-1]
        retrieved_image_id = image_ids[retrieved_image_index]

        correct_image_id = original_image_id[i]
        rank = np.where(correct_image_id == retrieved_image_id)
        ranks[sentence_id] = min(rank)
        retrieve_result[sentence_id] = retrieved_image_id[:k].tolist()
        all_sim[sentence_id] = sim[retrieved_image_index[:k]]
        del retrieved_image_id

    ranks = np.array(list(ranks.values()))
    top_k = retrieve_result
    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        if return_sim:
            return (r1, r5, r10, medr, meanr), (ranks, top_k, all_sim)
        else:
            return (r1, r5, r10, medr, meanr), (ranks, top_k)
    else:
        return (r1, r5, r10, medr, meanr)


