import torch
import random

def slice2d(x, start, end):
    return x[:, :, start:end, ...]


def slice3d(x, start, end):
    return x[:, :, :, start:end, ...]


def slice1d(x, start, end):
    return x[:, start:end, ...]

import torch.nn.functional as F
DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}

class ElasticCache:
    def __init__(
        self,
        start_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        ratio=0.,
        distance=-25,
        layer_num=40,
    ):
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]

        self.score_sum = torch.zeros(layer_num, self.cache_size + 1).cuda()
        self.ratio = ratio
        self.protect_size = 1
        self.lazy_size = 0
        self.flag = True
        self.distance = distance
        self.layer_num = layer_num

        self.selected_idx = 0

    def __call__(self, past_key_values, num_of_token=None, attentions=None):
        if past_key_values is None:
            return None
        attn_score = [attention for attention in attentions]
        seq_len = past_key_values[0][0].size(self.k_seq_dim)

        # update attn score
        attn_score = torch.cat(attn_score, dim=0)
        attn_score = attn_score.mean(dim=1, keepdim=False)
        
        if attn_score.shape[-2] > 1:
            assert self.flag is True # only use for the first time
            for idx in range(attn_score.shape[-1]):
                cur_score = attn_score[:, idx, :idx+1]
                self.score_sum[:, :(cur_score.shape[-1])] += cur_score
        else:
            pass

        forget_num = int(seq_len - num_of_token * (1 - self.ratio))
        if forget_num <= 0:
            return past_key_values
        else:
            if forget_num > 1:
                assert self.flag is True
                self.flag = False

                selected_idx_all = []
                merge_idx_all = []
                throw_idx_all = []
                for idx in range(self.layer_num):
                    selected_idx = torch.where(torch.argsort(self.score_sum[idx, self.start_size:(seq_len - self.protect_size)]) > forget_num)[0] + self.start_size
                    throw_idx = torch.where(torch.argsort(self.score_sum[idx, self.start_size:(seq_len - self.protect_size)]) <= forget_num)[0]
                    merge_idx = []
                    for i in range(len(throw_idx)):
                        merge_idx.append(selected_idx[torch.abs((selected_idx - throw_idx[i])).argmin()].unsqueeze(0))
                    merge_idx = torch.cat(merge_idx)

                    selected_idx = torch.cat([torch.arange(self.start_size).cuda(), selected_idx, torch.tensor([seq_len - self.protect_size]).cuda()], dim=0) # the last token is always kept

                    selected_idx_all.append(selected_idx)
                    merge_idx_all.append(merge_idx)
                    throw_idx_all.append(throw_idx)

                if self.distance > 0:
                    self.selected_idx = self.distance
                else:
                    self.selected_idx = seq_len - forget_num + self.distance

                past_key_values_return = []
                for idx, (k, v) in enumerate(past_key_values):
                    selected_idx = selected_idx_all[idx]
                    merge_idx = merge_idx_all[idx]
                    throw_idx = throw_idx_all[idx]

                    k_forget = k.gather(dim=1, index=throw_idx.view(1,-1,1,1).expand(k.shape[0], -1, k.shape[2], k.shape[-1]))
                    v_forget = v.gather(dim=1, index=throw_idx.view(1,-1,1,1).expand(v.shape[0], -1, v.shape[2], v.shape[-1]))

                    k = k.scatter_reduce(1, merge_idx.view(1,-1,1,1).expand(k.shape[0], -1, k.shape[2], k.shape[-1]), k_forget, 'mean')
                    v = v.scatter_reduce(1, merge_idx.view(1,-1,1,1).expand(v.shape[0], -1, v.shape[2], v.shape[-1]), v_forget, 'mean')

                    k_new = k.gather(dim=1, index=selected_idx.view(1,-1,1,1).expand(k.shape[0], -1, k.shape[2] ,k.shape[-1]))
                    v_new = v.gather(dim=1, index=selected_idx.view(1,-1,1,1).expand(v.shape[0], -1, v.shape[2] ,v.shape[-1]))

                    past_key_values_return.append([k_new, v_new])
                return past_key_values_return
            else:
                selected_idx = self.selected_idx

                return [[torch.cat([self.k_slice(k, 0, selected_idx), self.k_slice(k, (selected_idx+1), seq_len),],
                            dim=self.k_seq_dim,),
                        torch.cat([self.v_slice(v, 0, selected_idx), self.v_slice(v, (selected_idx+1), seq_len),],
                            dim=self.v_seq_dim,)]
                    for k, v in past_key_values]


class LocalCache:
    def __init__(
        self,
        start_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        ratio=0.
    ):
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]
        self.ratio = ratio

    def __call__(self, past_key_values, num_of_token=None, attentions=None):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)

        forget_num = int(seq_len - num_of_token * (1 - self.ratio))
        if forget_num <= 0:
            return past_key_values
        else:
            return [[torch.cat([self.k_slice(k, 0, self.start_size), self.k_slice(k, forget_num + self.start_size, seq_len),],
                        dim=self.k_seq_dim,),
                    torch.cat([self.v_slice(v, 0, self.start_size), self.v_slice(v, forget_num + self.start_size, seq_len),],
                        dim=self.v_seq_dim,),]
                for k, v in past_key_values]
        

class H2OCache:
    def __init__(
        self,
        start_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        ratio=0.
    ):
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]

        self.score_sum = torch.zeros(self.cache_size + 1).cuda()
        self.ratio = ratio
        self.protect_size = 1
        self.flag = True

    def __call__(self, past_key_values, num_of_token=None, attentions=None):
        if past_key_values is None:
            return None
        attn_score = [attention for attention in attentions]
        past_key_values_new = tuple(x for x in past_key_values)
        seq_len = past_key_values_new[0][0].size(self.k_seq_dim)
        # update attn score
        attn_score = torch.cat(attn_score, dim=0)
        attn_score = attn_score.mean(dim=1, keepdim=False).mean(dim=0, keepdim=False)

        if attn_score.shape[-2] > 1:
            assert self.flag is True # only use for the first time
            for idx in range(attn_score.shape[-1]):
                cur_score = attn_score[idx][:idx+1]
                self.score_sum[:len(cur_score)] += cur_score
        else:
            attn_score = attn_score.squeeze(0)
            self.score_sum[:seq_len] += attn_score

        forget_num = int(seq_len - num_of_token * (1 - self.ratio))
        if forget_num <= 0:
            return past_key_values_new
        else:
            if forget_num > 1:
                assert self.flag is True
                self.flag = False
                selected_idx = torch.where(torch.argsort(self.score_sum[:(seq_len - self.protect_size)]) > forget_num)[0]
                selected_idx = torch.cat([selected_idx, torch.arange(seq_len - self.protect_size, seq_len).cuda()], dim=0)
                past_key_values_return = []
                for k, v in past_key_values_new:
                    k_new = k.gather(dim=1, index=selected_idx.view(1,-1,1,1).expand(k.shape[0], -1, k.shape[2], k.shape[-1]))
                    v_new = v.gather(dim=1, index=selected_idx.view(1,-1,1,1).expand(v.shape[0], -1, v.shape[2], v.shape[-1]))
                    past_key_values_return.append([k_new, v_new])
                
                return past_key_values_return
            else:
                selected_idx = self.score_sum[self.start_size:(seq_len - self.protect_size)].argmin() + self.start_size
                self.score_sum[(selected_idx):-1] = self.score_sum[(selected_idx+1):].clone()
                
                return [[torch.cat([self.k_slice(k, 0, selected_idx), self.k_slice(k, (selected_idx+1), seq_len),],
                            dim=self.k_seq_dim,),
                        torch.cat([self.v_slice(v, 0, selected_idx), self.v_slice(v, (selected_idx+1), seq_len),],
                            dim=self.v_seq_dim,)]
                    for k, v in past_key_values_new]