import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from utils import save_tensor


class PQLayer(nn.Module):
    def __init__(self, feat_dim, M, K, alpha=1, flag=False):
        super(PQLayer, self).__init__()
        self.feat_dim, self.M, self.K, self.D = feat_dim, M, K, feat_dim//M
        self.alpha = alpha
        self._C = nn.Parameter(torch.empty(
            (self.M, self.K, self.D)), requires_grad=flag)
        nn.init.xavier_uniform_(self._C.data)

    @torch.no_grad()
    def _codebook_normalization(self):
        # normalize the codewords
        codewords = self._C.data.clone()
        codewords = F.normalize(codewords, dim=-1)
        self._C.copy_(codewords)

    def reconstruct(self, codes, hard_quant=False):
        # self._codebook_normalization()
        if hard_quant:
            # codesT:[Mxb]
            codesT = codes.T
            # x_hat:[bxMxD]: self._C:[MxKxD].lookup(codesT:[Mxb])
            x_hat_ = []
            for i in range(self.M):
                # x_hat_[i]:[bxD]: _C[i]:[KxD].lookup(codesT[i]:[b])
                x_hat_.append(self._C[i][codesT[i]])
            # x_hat_:[MxbxD]=>[bxMxD]
            x_hat_ = torch.transpose(torch.stack(x_hat_), 0, 1)
            # print("x_hat_:\n", x_hat_, "\nshape[bxMxD]=", x_hat_.shape, end='\n\n')
            # x_hat:[bxMxD]=>[bxfeat_dim]
            x_hat = x_hat_.reshape(x_hat_.shape[0], -1)
        else: # soft assignment
            # x_hat_:[bxMxD]
            # _C:[MxKxD], codes:[bxMxK] => x_hat_:[bxMxD]
            x_hat_ = torch.einsum('mkd,bmk->bmd', self._C, codes)
            # print("x_hat_:\n", x_hat_, "\nshape[bxMxD]=", x_hat_.shape, end='\n\n')
            # x_hat:[bxMxD]=>[bxfeat_dim]
            x_hat = x_hat_.view(x_hat_.shape[0], -1)

        return x_hat

    def forward(self, x, hard_quant=False, compute_err=False):
        # self._codebook_normalization()
        # print("x:\n", x, "\nshape[bxfeat_dim]=", x.shape, end='\n\n')
        # x:[bxd]=>[bxMxD]
        x_ = F.normalize(x.view(x.shape[0], self.M, self.D), dim=-1)
        # print("x_:\n", x_, "\nshape[bxMxD]=", x_.shape, end='\n\n')
        # print("_C:\n", _C, "\nshape[MxKxD]=", _C.shape, end='\n\n')
        # x_:[bxMxD], _C:[MxKxD] => ips:[bxMxK]
        ips = torch.einsum('bmd,mkd->bmk', x_, self._C)
        # print("ips:\n", ips, "\nshape[bxMxK]=", ips.shape, end='\n\n')
        if hard_quant:
            # codes:[bxM]
            codes = ips.argmax(dim=-1)
            # print("codes:\n", codes, "\nshape[bxM]=", codes.shape, end='\n\n')
            # codesT:[Mxb]
            codesT = codes.T
            # x_hat:[bxMxD]: _C:[MxKxD].lookup(codesT:[Mxb])
            x_hat_ = []
            for i in range(self.M):
                # x_hat_[i]:[bxD]: _C[i]:[KxD].lookup(codesT[i]:[b])
                x_hat_.append(self._C[i][codesT[i]])
            # x_hat_:[MxbxD]=>[bxMxD]
            x_hat_ = torch.transpose(torch.stack(x_hat_), 0, 1)
            # print("x_hat_:\n", x_hat_, "\nshape[bxMxD]=", x_hat_.shape, end='\n\n')
            # x_hat:[bxMxD]=>[bxfeat_dim]
            x_hat = x_hat_.reshape(x_hat_.shape[0], -1)
        else: # soft assignment
            # codes:[bxMxK]
            codes = F.softmax(ips * self.alpha, dim=-1)
            # print("codes:\n", codes, "\nshape[bxMxK]=", codes.shape, end='\n\n')
            # x_hat_:[bxMxD]
            # _C:[MxKxD], codes:[bxMxK] => x_hat_:[bxMxD]
            x_hat_ = torch.einsum('mkd,bmk->bmd', self._C, codes)
            # print("x_hat_:\n", x_hat_, "\nshape[bxMxD]=", x_hat_.shape, end='\n\n')
            # x_hat:[bxMxD]=>[bxfeat_dim]
            x_hat = x_hat_.view(x_hat_.shape[0], -1)
        
        # print("x_hat:\n", x_hat, "\nshape=[bxfeat_dim]", x_hat.shape, end='\n\n')
        if compute_err:
            with torch.no_grad():
                err = F.mse_loss(x_, x_hat_)
            # print("err:\n", err, end='\n\n')
            return x_hat, codes, err
        else:
            return x_hat, codes

    @property
    def codebooks(self):
        return F.normalize(self._C.data, dim=-1)

    def save_codebooks(self, path):
        save_tensor(self.codebooks, path)


class MeCoQ(nn.Module):
    def __init__(self, feat_dim, M, K, alpha=1, trainable_layer_num=0, queue_size=384, CNN_model_path=None):
        super(MeCoQ, self).__init__()
        if CNN_model_path:
            self.vgg = torchvision.models.vgg16()
            state_dict = torch.load(CNN_model_path)
            self.vgg.load_state_dict(state_dict)
        else:
            self.vgg = torchvision.models.vgg16(pretrained=True)
        
        self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier.children())[:6])

        assert trainable_layer_num <= 2
        for i, param in enumerate(self.vgg.parameters()):
            if (i + trainable_layer_num * 2) < 30:
                param.requires_grad = False

        self.projection = nn.Linear(4096, feat_dim)
        self.feat_dim, self.M, self.K, self.D = feat_dim, M, K, feat_dim//M
        self.pq_layer = PQLayer(feat_dim, M, K, alpha)
        self.queue_size = queue_size

    def codebook_normalization(self):
        self.pq_layer._codebook_normalization()

    def register_queue(self, enqueue_size, device='cpu'):
        self.register_buffer('soft_code_queue',
                             torch.zeros((self.queue_size, self.M, self.K), device=device))
        self.start_ptr = 0
        self.enqueue_size = enqueue_size

    def dequeue_enqueue(self, enqueue_batch):
        # the size of last batch of epoch may be less than enqueue_size
        enqueue_size = min(self.enqueue_size, len(enqueue_batch))
        self.soft_code_queue[self.start_ptr: self.start_ptr+enqueue_size] = enqueue_batch
        self.start_ptr = (self.start_ptr  + self.enqueue_size) % self.queue_size

    def release_queue(self):
        del self.soft_code_queue
        del self.start_ptr
        del self.queue_size

    def get_queue_feats(self):
        return self.pq_layer.reconstruct(self.soft_code_queue)

    def forward(self, x, only_feats=False, norm_feats=True, 
                hard_quant=False, compute_err=False):
        x = self.vgg.features(x)
        x = x.view(x.shape[0], -1)
        x = self.vgg.classifier(x)
        x = self.projection(x)

        if only_feats:
            if norm_feats:
                # Intra-normalization
                x = F.normalize(x.view(x.shape[0], self.M, self.D), dim=-1)
                x = x.view(x.shape[0], -1)
            return x
        else:
            if compute_err:
                x_hat, codes, err = self.pq_layer(x, hard_quant=hard_quant, compute_err=True)
                if norm_feats:
                    # Intra-normalization
                    x = F.normalize(x.view(x.shape[0], self.M, self.D), dim=-1)
                    x = x.view(x.shape[0], -1)
                return x, x_hat, codes, err
            else:
                x_hat, codes = self.pq_layer(x, hard_quant=hard_quant)
                if norm_feats:
                    # Intra-normalization
                    x = F.normalize(x.view(x.shape[0], self.M, self.D), dim=-1)
                    x = x.view(x.shape[0], -1)
                return x, x_hat, codes

    @property
    def codebooks(self):
        return self.pq_layer.codebooks

    def save_codebooks(self, path):
        self.pq_layer.save_codebooks(path)


if __name__ == '__main__':
    b, M, K, D, feat_dim = 5, 2, 4, 3, 6
    # print("b=%d, M=%d, K=%d, D=%d, feat_dim=%d" % (b, M, K, D, feat_dim))
    with torch.no_grad():
        pq_layer = PQLayer(feat_dim, M, K)
        x = torch.rand(b, feat_dim)
        # print("/***************** HARD *****************/")
        pq_layer(x, hard_quant=True, compute_err=True)
        # print("/***************** SOFT *****************/")
        pq_layer(x, compute_err=True)
