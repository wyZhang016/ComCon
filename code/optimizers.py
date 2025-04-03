import tqdm
import torch
from torch import nn
from torch import optim
import numpy as np
from models import KBCModel
from regularizers import Regularizer


class KBCOptimizer(object):
    def __init__(
            self, model: KBCModel, model_exp: KBCModel, model_imp: KBCModel, regularizer: list, optimizer: optim.Optimizer, batch_size: int = 256,
            verbose: bool = True
    ):
        self.model = model
        self.model_exp = model_exp
        self.model_imp = model_imp
        self.regularizer = regularizer[0]
        self.regularizer2 = regularizer[1]
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose

    def epoch(self, examples: torch.LongTensor, e=0, weight=None):
        self.model.train()
        self.model_exp.train()
        self.model_imp.train()
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        loss = nn.CrossEntropyLoss(reduction='mean', weight=weight)

        fb_ling_f = r'../pre_train/MKG-W-textual.pth'
        fb_visual_f = r'../pre_train/MKG-W-visual.pth'
        wn_ling_f = r"../pre_train/DB15K-textual.pth"
        wn_visual_f = r"../pre_train/DB15K-visual.pth"
        # fb_ling, fb_visual, wn_ling, wn_visual = torch.tensor(np.load(fb_ling_f)), torch.tensor(
        #     np.load(fb_visual_f)), torch.tensor(np.load(wn_ling_f)), torch.tensor(np.load(wn_visual_f))
        fb_ling, fb_visual, wn_ling, wn_visual = (torch.load(fb_ling_f), torch.load(fb_visual_f), torch.load(wn_ling_f),
                                                  torch.load(wn_visual_f))
        multimodal_embeddings = [wn_ling, wn_visual]
        multimodal_embeddings1 = [fb_ling, fb_visual]

        with tqdm.tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            while b_begin < examples.shape[0]:

                input_batch = actual_examples[ b_begin: (b_begin+self.batch_size) ].cuda()  # shape = [bs,3]

                h, r, t = self.model.get_emb(input_batch)
                h_exp, r_exp, t_exp = self.model_exp.get_emb(input_batch)
                h_imp, r_imp, t_imp = self.model_imp.get_emb(input_batch)
                # h_imp[0], h_imp[1], h_imp[2], h_imp[3] = h[0]-h_exp[0], h[1]-h_exp[1],h[2]-h_exp[2], h[3]-h_exp[3]
                # r_imp[0], r_imp[1],r_imp[2], r_imp[3] = r[0]-r_exp[0], r[1]-r_exp[1],r[2]-r_exp[2], r[3]-r_exp[3]
                # t_imp[0], t_imp[1], t_imp[2], t_imp[3] = t[0]-t_exp[0] , t[1]-t_exp[1],t[2]-t_exp[2], t[3]-t_exp[3]
                h_imp[0], h_imp[1] = h[0] - h_exp[0], h[1] - h_exp[1]
                r_imp[0], r_imp[1] = r[0] - r_exp[0], r[1] - r_exp[1]
                t_imp[0], t_imp[1] = t[0] - t_exp[0], t[1] - t_exp[1]

                # factors[0].shape = [bs,rank]
                predictions, factors = self.model.forward_cal(input_batch, h, r, t)
                predictions_exp, factors_exp = self.model_exp.forward_cal(input_batch, h_exp, r_exp, t_exp)# predictions.shape=[bs, N of E]
                predictions_imp, factors_imp = self.model_imp.forward_cal(input_batch, h_imp, r_imp, t_imp)

                post = torch.gather(predictions, dim=1, index=input_batch[:,-1].view(-1, 1))
                post_exp = torch.gather(predictions_exp, dim=1, index=input_batch[:,-1].view(-1, 1))
                post_imp = torch.gather(predictions_imp, dim=1, index=input_batch[:,-1].view(-1, 1))


                # predictions, factors = self.model.forward(input_batch, multimodal_embeddings)
                truth = input_batch[:, 2]

                l_fit = loss(predictions, truth) + loss(predictions_exp, truth) + loss(predictions_imp, truth)

                l_auxiliary = -torch.nn.functional.logsigmoid(post)-torch.nn.functional.logsigmoid(-post_exp.detach()) - torch.nn.functional.logsigmoid(post_exp) - torch.nn.functional.logsigmoid(-post_imp.detach())
                l_reg = self.regularizer.forward(factors) + self.regularizer.forward(factors_exp) + self.regularizer.forward(factors_imp)
                l_auxiliary = l_auxiliary.mean()
                l = l_fit + l_reg + l_auxiliary

                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
                bar.set_postfix(loss=f'{l.item():.1f}', reg=f'{l_reg.item():.1f}')

        return l