from torch import nn
from collections import OrderedDict
import numpy as np


class SCLoss(nn.Module):
    def __init__(self, loss_type='seqnll'):
        self.loss_type = loss_type

    def get_scores(self, data_gts, gen_result, opt):
        batch_size = gen_result.size(0)
        seq_per_img = batch_size // len(data_gts)

        res = OrderedDict()

        gen_result = gen_result.data.cpu().numpy()
        for i in range(batch_size):
            res[i] = [array_to_str(gen_result[i])]

        gts = OrderedDict()
        for i in range(len(data_gts)):
            gts[i] = [array_to_str(data_gts[i][j]) for j in range(len(data_gts[i]))]

        res_ = [{'image_id':i, 'caption': res[i]} for i in range(batch_size)]
        res__ = {i: res[i] for i in range(batch_size)}
        gts = {i: gts[i // seq_per_img] for i in range(batch_size)}
        if opt.cider_reward_weight > 0:
            _, cider_scores = CiderD_scorer.compute_score(gts, res_)
            print('Cider scores:', _)
        else:
            cider_scores = 0
        if opt.bleu_reward_weight > 0:
            _, bleu_scores = Bleu_scorer.compute_score(gts, res__)
            bleu_scores = np.array(bleu_scores[3])
            print('Bleu scores:', _[3])
        else:
            bleu_scores = 0

        scores = opt.cider_reward_weight * cider_scores + opt.bleu_reward_weight * bleu_scores

        return scores
