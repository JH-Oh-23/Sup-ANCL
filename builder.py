import torch
import torch.nn as nn
from copy import deepcopy

class SupSiam(nn.Module):
    def __init__(self, base_encoder, dim=2048, pred_dim=512
                K=8192, num_positive=0, alpha=0.5):
        super(SupSiam, self).__init__()

        self.K = K
        self.num_positive = num_positive 
        self.alpha = alpha

        # create the encoder
        self.dim = dim
        self.pred_dim = pred_dim
        self.in_dim = base_encoder().fc.in_features
        self.encoder = base_encoder()
        self.encoder.fc = nn.Identity()

        # build a 3-layer projector
        self.projector = nn.Sequential(nn.Linear(self.in_dim, self.dim, bias=False), nn.BatchNorm1d(self.dim),
                                    nn.ReLU(inplace=True), nn.Linear(self.dim, self.dim, bias=False), nn.BatchNorm1d(self.dim),
                                    nn.ReLU(inplace=True), nn.Linear(self.dim, self.dim, bias=False), nn.BatchNorm1d(self.dim, affine=False))

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(self.dim, self.pred_dim, bias=False),
                                        nn.BatchNorm1d(self.pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(self.pred_dim, self.dim)) # output layer

        # create queue for keeping neighbor
        self.register_buffer("queue", torch.randn(K, dim))
        self.queue = nn.functional.normalize(self.queue, dim=-1)
        self.register_buffer("queue_labels", -torch.ones(self.K, dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def dequeue_and_enqueue(self, features, labels):
        # gather features and labels before updating queue
        features = concat_all_gather(features)
        labels = concat_all_gather(labels)

        batch_size = features.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0 # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr : ptr+batch_size, :] = features
        self.queue_labels[ptr : ptr+batch_size] = labels
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def sample_target(self, features, labels):
        neighbor = []
        for i, label in enumerate(labels):
            pos = torch.where(self.queue_labels == label)[0]
            if len(pos) != 0:
                if self.num_positive > 0:
                    choice = torch.multinomial(torch.ones_like(pos).type(torch.FloatTensor), self.num_positive, replacement=True)
                    idx = pos[choice]
                    neighbor.append(self.queue[idx].mean(0))
                else:
                    neighbor.append(self.queue[pos].mean(0))
            else:
                neighbor.append(features[i])
        neighbor = torch.stack(neighbor, dim=0)

        return neighbor

    def forward(self, x1, x2, labels):

        # compute features for each view
        z1, z2 = self.projector(self.encoder(x1)), self.projector(self.encoder(x2))
        p1, p2 = self.predictor(z1), self.predictor(z2) 
        z1, z2 = nn.functional.normalize(z1, dim=-1), nn.functional.normalize(z2, dim=-1)
        p1, p2 = nn.functional.normalize(p1, dim=-1), nn.functional.normalize(p2, dim=-1)

        # sample supervised targets
        z1_sup = self.sample_target(z1.detach(), labels)
        z2_sup = self.sample_target(z2.detach(), labels)

        # calculate loss
        SSL_loss = -0.5 * (torch.sum(p1 * z2.detach(), dim=-1) + torch.sum(p2 * z1.detach(), dim=-1))
        Sup_loss = -0.5 * (torch.sum(p1 * z2_sup, dim=-1) + torch.sum(p2 * z1_sup, dim=-1))
        loss = self.alpha * SSL_loss + (1-self.alpha) * Sup_loss

        # dequeue and enqueue
        self.dequeue_and_enqueue(z1.detach(), labels)

        return SSL_loss.mean(), Sup_loss.mean(), loss.mean()
    
class SupBYOL(nn.Module):
    def __init__(self, base_encoder, dim=256, hid_dim=4096, m=0.996,
                 K=8192,  M=0, alpha = 0.5):
        super(SupBYOL, self).__init__()

        self.m = m
        self.K = K
        self.num_positive = num_positive
        self.alpha = alpha

        # create the encoder
        self.dim = dim
        self.in_dim = base_encoder().fc.in_features
        self.encoder = base_encoder()
        self.encoder.fc = nn.Identity()

        # build a 2-layer projector
        self.projector = nn.Sequential(nn.Linear(self.in_dim, hid_dim, bias=False), nn.BatchNorm1d(hid_dim),
                                    nn.ReLU(inplace=True), nn.Linear(hid_dim, dim, bias=False))

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, hid_dim, bias=False),
                                        nn.BatchNorm1d(hid_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hid_dim, dim))


        # cretae target network
        self.target_encoder = deepcopy(self.encoder)
        self.target_projector = deepcopy(self.projector)

        for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
            
        for param_q, param_k in zip(self.projector.parameters(), self.target_projector.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
            
        # create queue for keeping neighbor
        self.register_buffer("queue", torch.randn(K, dim))
        self.queue = nn.functional.normalize(self.queue, dim=-1)
        self.register_buffer("queue_labels", -torch.ones(self.K, dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def dequeue_and_enqueue(self, features, labels):

        # gather features and labels before updating queue
        features = concat_all_gather(features)
        labels = concat_all_gather(labels)

        batch_size = features.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0 # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr : ptr+batch_size, :] = features
        self.queue_labels[ptr : ptr+batch_size] = labels
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _momentum_update_key_encoder(self):

        for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
            
        for param_q, param_k in zip(self.projector.parameters(), self.target_projector.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def sample_target(self, features, labels):

        neighbor = []
        for i, label in enumerate(labels):
            pos = torch.where(self.queue_labels == label)[0]
            if len(pos) != 0:
                if self.num_positive > 0:
                    choice = torch.multinomial(torch.ones_like(pos).type(torch.FloatTensor), self.num_positive, replacement=True)
                    idx = pos[choice]
                    neighbor.append(self.queue[idx].mean(0))
                else:
                    neighbor.append(self.queue[pos].mean(0))
            else:
                neighbor.append(features[i])
        neighbor = torch.stack(neighbor, dim=0)

        return neighbor

    def forward(self, x1, x2, labels):

        # compute predictions
        p1, p2 = self.predictor(self.projector(self.encoder(x1))), self.predictor(self.projector(self.encoder(x2)))
        p1, p2 = nn.functional.normalize(p1, dim=-1), nn.functional.normalize(p2, dim=-1)

        with torch.no_grad():
            # update target encoder
            self._momentum_update_key_encoder()

            # compute targets
            z1, z2 = self.target_projector(self.target_encoder(x1)), self.target_projector(self.target_encoder(x2))
            z1, z2 = nn.functional.normalize(z1, dim=-1), nn.functional.normalize(z2, dim=-1)

            # sample supervised targets
            z1_sup = self.sample_target(z1.detach(), labels)
            z2_sup = self.sample_target(z2.detach(), labels)

        # compute loss
        SSL_loss = -0.5 * (torch.sum(p1 * z2, dim=-1) + torch.sum(p2 * z1, dim=-1))
        Sup_loss = -0.5 * (torch.sum(p1 * z2_sup, dim=-1) + torch.sum(p2 * z1_sup, dim=-1))
        loss = self.alpha * SSL_loss + (1-self.alpha) * Sup_loss

        # dequeue and enqueue
        self.dequeue_and_enqueue(z1, labels)

        return SSL_loss.mean(), Sup_loss.mean(), loss.mean()
  
class SupMoCo(nn.Module):
    def __init__(self, base_encoder, dim=128, K=8192, m=0.999, T=0.07):
        super(SupMoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoder
        self.dim = dim
        self.in_dim = base_encoder().fc.in_features
        self.encoder = base_encoder()
        self.encoder.fc = nn.Identity()

        # build a 2-layer projector
        self.projector = nn.Sequential(nn.Linear(self.in_dim, self.in_dim, bias=False), nn.BatchNorm1d(self.in_dim),
                                    nn.ReLU(inplace=True), nn.Linear(self.in_dim, self.dim))

        # cretae target network
        self.target_encoder = deepcopy(self.encoder)
        self.target_projector = deepcopy(self.projector)

        for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
            
        for param_q, param_k in zip(self.projector.parameters(), self.target_projector.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_labels", -torch.ones(1, K).long())
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):

        for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
            
        for param_q, param_k in zip(self.projector.parameters(), self.target_projector.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):

        # gather keys before updating queue
        keys = concat_all_gather(keys)
        labels = concat_all_gather(labels)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_labels[:, ptr:ptr + batch_size] = labels.contiguous().view(1,-1)
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):

        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):

        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def contrastive_loss(self, q, k, im_labels):

        # gather all targets
        k = concat_all_gather(k)

        # logits from batch 
        l_batch = torch.einsum('nc,ck->nk', [q, k.T.detach()])
        # logits from queue N x K
        l_queue = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        
        # logits
        logits = torch.cat([l_batch, l_queue], dim=1)
        
        # mask
        im_labels = im_labels.contiguous().view(-1, 1)
        labels_all = concat_all_gather(im_labels)

        batch_mask = torch.eq(im_labels, labels_all.T).float()
        queue_mask = torch.eq(im_labels, self.queue_labels.clone().detach()).float()
        mask = torch.cat([batch_mask, queue_mask], dim=1)
                    
        N = logits.shape[0]
        SSL_mask = torch.zeros_like(mask)
        SSL_mask[torch.arange(N),(torch.arange(N) + N * torch.distributed.get_rank())] = 1
        Sup_mask = mask.clone()
        Sup_mask[torch.arange(N),(torch.arange(N) + N * torch.distributed.get_rank())] = 0
        
        # apply temperature
        logits /= self.T
        
        log_prob = nn.functional.normalize(logits.exp(), dim=1, p=1).log()

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)

        SSL_loss = - torch.sum((SSL_mask * log_prob).sum(1) / mask_pos_pairs) / mask.shape[0]
        Sup_loss = - torch.sum((Sup_mask * log_prob).sum(1) / mask_pos_pairs) / mask.shape[0]
        loss = - torch.sum((mask * log_prob).sum(1) / mask_pos_pairs) / mask.shape[0]

        return SSL_loss, Sup_loss, loss


    def forward(self, x1, x2, labels):

        q1, q2 = self.projector(self.encoder(x1)), self.projector(self.encoder(x2))
        q1, q2 = nn.functional.normalize(q1, dim=1), nn.functional.normalize(q2, dim=1)

        with torch.no_grad():   # no gradient
            self._momentum_update_key_encoder()
            
            # shuffle for making use of BN
            im_k1, idx_unshuffle1 = self._batch_shuffle_ddp(x1)
            im_k2, idx_unshuffle2 = self._batch_shuffle_ddp(x2)
            
            k1, k2 = self.target_projector(self.target_encoder(im_k1)), self.target_projector(self.target_encoder(im_k2)) 
            k1, k2 = nn.functional.normalize(k1, dim=1), nn.functional.normalize(k2, dim=1)
            
            # undo shuffle
            k1 = self._batch_unshuffle_ddp(k1, idx_unshuffle1)
            k2 = self._batch_unshuffle_ddp(k2, idx_unshuffle2)

        SSL_loss_1, Sup_loss_1, loss_1 = self.contrastive_loss(q1, k2, labels)
        SSL_loss_2, Sup_loss_2, loss_2 = self.contrastive_loss(q1, k2, labels)
                
        SSL_loss, Sup_loss = 0.5 * (SSL_loss_1 + SSL_loss_2), 0.5 * (Sup_loss_1 + Sup_loss_2)
        loss = 0.5 * (loss_1 + loss_2)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k1, labels)

        return SSL_loss.mean(), Sup_loss.mean(), loss.mean()

class SupCon(nn.Module):
    def __init__(self, base_encoder, dim=128, T=0.1):
        super(SupCon, self).__init__()
 
       self.T = T

        # create the encoder
        self.dim = dim
        self.in_dim = base_encoder().fc.in_features
        self.encoder = base_encoder()
        self.encoder.fc = nn.Identity()

        # build a 2-layer projector
        self.projector = nn.Sequential(nn.Linear(self.in_dim, self.in_dim, bias=False), nn.BatchNorm1d(self.in_dim),
                                    nn.ReLU(inplace=True), nn.Linear(self.in_dim, self.dim))

    def forward(self, x1, x2, labels):

        z1, z2 = self.projector(self.encoder(x1)), self.projector(self.encoder(x2))
        z1, z2 = nn.functional.normalize(z1, dim=1), nn.functional.normalize(z2, dim=1)

        z = torch.cat([z1.unsqueeze(1), z2.unsqueeze(1)], dim=1)
        
        batch_size = len(labels)

        # mask for equal instance
        I = torch.eye(batch_size).float().to(labels.device)

        # mask for equal label
        labels = labels.contiguous().view(-1,1)
        mask = torch.eq(labels, labels.T).float().to(labels.device)

        # compute logits
        contrast_count = z.shape[1]
        contrast_feature = torch.cat(torch.unbind(z, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count
        score = torch.mm(anchor_feature, contrast_feature.T) / self.T

        # for numerical stability
        logits_max, _ = torch.max(score, dim=1, keepdim=True)
        logits = score - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        I = I.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(mask.device), 0)
        mask = mask * logits_mask
        I = I * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        SSL_loss = - (I * log_prob).sum(1) / mask_pos_pairs
        Sup_loss = - ((mask - I) * log_prob).sum(1) / mask_pos_pairs
        loss = -(mask * log_prob).sum(1) / mask_pos_pairs

        SSL_loss = SSL_loss.view(anchor_count, batch_size).mean(0)
        Sup_loss = Sup_loss.view(anchor_count, batch_size).mean(0)
        loss = loss.view(anchor_count, batch_size).mean(0)
        
        return SSL_loss.mean(), Sup_loss.mean(), loss.mean()

class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.dim = dim
        self.pred_dim = pred_dim
        self.in_dim = base_encoder().fc.in_features
        self.encoder = base_encoder()
        self.encoder.fc = nn.Identity()

        # build a 3-layer projector
        self.projector = nn.Sequential(nn.Linear(self.in_dim, self.dim, bias=False), nn.BatchNorm1d(self.dim),
                                    nn.ReLU(inplace=True), nn.Linear(self.dim, self.dim, bias=False), nn.BatchNorm1d(self.dim),
                                    nn.ReLU(inplace=True), nn.Linear(self.dim, self.dim, bias=False), nn.BatchNorm1d(self.dim, affine=False))

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(self.dim, self.pred_dim, bias=False),
                                        nn.BatchNorm1d(self.pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(self.pred_dim, self.dim)) # output layer

    def forward(self, x1, x2, label):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        z1, z2 = self.projector(self.encoder(x1)), self.projector(self.encoder(x2))
        p1, p2 = self.predictor(z1), self.predictor(z2) 

        z1, z2 = nn.functional.normalize(z1, dim=-1), nn.functional.normalize(z2, dim=-1)
        p1, p2 = nn.functional.normalize(p1, dim=-1), nn.functional.normalize(p2, dim=-1)


        SSL_loss = -0.5 * (torch.sum(p1*z2.detach(), dim=-1) + torch.sum(p2*z1.detach(), dim=-1))
        Sup_loss = torch.zeros_like(SSL_loss)
        loss = SSL_loss

        return SSL_loss.mean(), Sup_loss.mean(), loss.mean()

class BYOL(nn.Module):
    def __init__(self, base_encoder, dim=256, hid_dim=4096, m=0.996):
        """
        hidden_dim : hidden dimension of the projector (default: 4096)
        dim: feature dimension (default: 256)
        """
        super(BYOL, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.m = m

        self.dim = dim
        self.in_dim = base_encoder().fc.in_features
        self.encoder = base_encoder()
        self.encoder.fc = nn.Identity()

        # build a 2-layer projector
        self.projector = nn.Sequential(nn.Linear(self.in_dim, hid_dim, bias=False), nn.BatchNorm1d(hid_dim),
                                    nn.ReLU(inplace=True), nn.Linear(hid_dim, dim, bias=False))


        # cretae target network
        self.target_encoder = deepcopy(self.encoder)
        self.target_projector = deepcopy(self.projector)

        for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
            
        for param_q, param_k in zip(self.projector.parameters(), self.target_projector.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, hid_dim, bias=False),
                                        nn.BatchNorm1d(hid_dim),
                                        nn.ReLU(inplace=True), 
                                        nn.Linear(hid_dim, dim))
            
    @torch.no_grad()
    def _momentum_update_key_encoder(self):

        for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
            
        for param_q, param_k in zip(self.projector.parameters(), self.target_projector.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, x1, x2, labels):

        p1, p2 = self.predictor(self.projector(self.encoder(x1))), self.predictor(self.projector(self.encoder(x2)))
        p1, p2 = nn.functional.normalize(p1, dim=-1), nn.functional.normalize(p2, dim=-1)

        with torch.no_grad():
            # update target encoder
            self._momentum_update_key_encoder()

            # compute targets
            z1, z2 = self.target_projector(self.target_encoder(x1)), self.target_projector(self.target_encoder(x2))
            z1, z2 = nn.functional.normalize(z1, dim=-1), nn.functional.normalize(z2, dim=-1)

        # compute loss
        SSL_loss = -0.5 * (torch.sum(p1 * z2, dim=-1) + torch.sum(p2 * z1, dim=-1))
        Sup_loss = torch.zeros_like(SSL_loss)
        loss = SSL_loss

        return SSL_loss.mean(), Sup_loss.mean(), loss.mean()
        
class MoCo(nn.Module):
    def __init__(self, base_encoder, dim=128, K=8192, m=0.999, T=0.2):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 8192)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.2)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoder
        self.dim = dim
        self.in_dim = base_encoder().fc.in_features
        self.encoder = base_encoder()
        self.encoder.fc = nn.Identity()

        # build a 2-layer projector
        self.projector = nn.Sequential(nn.Linear(self.in_dim, self.in_dim, bias=False), nn.BatchNorm1d(self.in_dim),
                                    nn.ReLU(inplace=True), nn.Linear(self.in_dim, self.dim))

        # cretae target network
        self.target_encoder = base_encoder()
        self.target_encoder.fc = nn.Identity()
        self.target_projector = nn.Sequential(nn.Linear(self.in_dim, self.in_dim, bias=False), nn.BatchNorm1d(self.in_dim),
                                    nn.ReLU(inplace=True), nn.Linear(self.in_dim, self.dim))

        for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
            
        for param_q, param_k in zip(self.projector.parameters(), self.target_projector.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
            
        for param_q, param_k in zip(self.projector.parameters(), self.target_projector.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def contrastive_loss(self, q, k):
        """
        Input:
            q: embedding of query images
            k: embedding of key images
        """
        # compute logits
        # Einstein sum is more intuitive
        # positive logits from augmentation: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        
        # apply temperature
        logits /= self.T

        # labels : positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        
        loss = nn.functional.cross_entropy(logits, labels)
        
        return loss


    def forward(self, x1, x2, labels):

        # compute query features
        q1, q2 = self.projector(self.encoder(x1)), self.projector(self.encoder(x2))
        q1, q2 = nn.functional.normalize(q1, dim=1), nn.functional.normalize(q2, dim=1)

        # compute key features
        with torch.no_grad():   # no gradient
            self._momentum_update_key_encoder()
            
            # shuffle for making use of BN
            im_k1, idx_unshuffle_1 = self._batch_shuffle_ddp(x1)
            im_k2, idx_unshuffle_2 = self._batch_shuffle_ddp(x2)
            
            k1, k2 = self.target_projector(self.target_encoder(im_k1)), self.target_projector(self.target_encoder(im_k2))  # keys: NxC
            k1, k2 = nn.functional.normalize(k1, dim=1), nn.functional.normalize(k2, dim=1)
            
            # undo shuffle
            k1 = self._batch_unshuffle_ddp(k1, idx_unshuffle_1)
            k2 = self._batch_unshuffle_ddp(k2, idx_unshuffle_2)


        SSL_loss = 0.5 * ( self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1) )
        Sup_loss = torch.zeros_like(SSL_loss)
        loss = SSL_loss
        
        # dequeue and enqueue
        self._dequeue_and_enqueue(k1)

        return SSL_loss, Sup_loss, loss

class SimCLR(nn.Module):

    def __init__(self, base_encoder, dim=128, T=0.1):
        super(SimCLR, self).__init__()

        self.T = T

        # encoder
        self.dim = dim
        self.in_dim = base_encoder().fc.in_features
        self.encoder = base_encoder()
        self.encoder.fc = nn.Identity()

        # build a 2-layer projector
        self.projector = nn.Sequential(nn.Linear(self.in_dim, self.in_dim, bias=False), nn.BatchNorm1d(self.in_dim),
                        nn.ReLU(inplace=True), nn.Linear(self.in_dim, self.dim))
    
    def forward(self, x1, x2, label):

        z1, z2 = self.projector(self.encoder(x1)), self.projector(self.encoder(x2))
        z1, z2 = nn.functional.normalize(z1, dim=-1), nn.functional.normalize(z2, dim=-1)
        z = torch.cat([z1.unsqueeze(1), z2.unsqueeze(1)], dim=1)

        # mask for equal instance
        labels = label.contiguous().view(-1,1)
        batch_size = len(label)
        mask = torch.eye(batch_size).float().to(label.device)

        # compute logits
        contrast_count = z.shape[1]
        contrast_feature = torch.cat(torch.unbind(z, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count
        score = torch.mm(anchor_feature, contrast_feature.T) / self.T

        # for numerical stability
        logits_max, _ = torch.max(score, dim=1, keepdim=True)
        logits = score - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        
        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(mask.device), 0)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        SSL_loss = - mean_log_prob_pos
        SSL_loss = SSL_loss (anchor_count, batch_size).mean(0)
        Sup_loss = torch.zeros_like(SSL_loss)
        loss= SSL_loss 
 
        return SSL_loss.mean(), Sup_loss.mean(), loss.mean()


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

