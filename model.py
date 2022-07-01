import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, RelaxedOneHotCategorical
import math
import numpy as np

class VQEmbeddingEMA(nn.Module):
    def __init__(self, latent_dim, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.999, epsilon=1e-5):
        super(VQEmbeddingEMA, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        embedding = torch.zeros(latent_dim, num_embeddings, embedding_dim)
        embedding.uniform_(-1/num_embeddings, 1/num_embeddings)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(latent_dim, num_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())

    def forward(self, x):
        B, C, H, W = x.size()
        N, M, D = self.embedding.size()
        assert C == N * D

        x = x.view(B, N, D, H, W).permute(1, 0, 3, 4, 2)
        x_flat = x.detach().reshape(N, -1, D)

        distances = torch.baddbmm(torch.sum(self.embedding ** 2, dim=2).unsqueeze(1) +
                                  torch.sum(x_flat ** 2, dim=2, keepdim=True),
                                  x_flat, self.embedding.transpose(1, 2),
                                  alpha=-2.0, beta=1.0)

        indices = torch.argmin(distances, dim=-1)
        encodings = F.one_hot(indices, M).float()
        quantized = torch.gather(self.embedding, 1, indices.unsqueeze(-1).expand(-1, -1, D))
        quantized = quantized.view_as(x)

        if self.training:
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=1)

            n = torch.sum(self.ema_count, dim=-1, keepdim=True)
            self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n

            dw = torch.bmm(encodings.transpose(1, 2), x_flat)
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw

            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()

        avg_probs = torch.mean(encodings, dim=1)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10), dim=-1))

        return quantized.permute(1, 0, 4, 2, 3).reshape(B, C, H, W), loss, perplexity.sum()

class MultiCodebookSoftVQ(nn.Module):
    def __init__(self,m,k,d,hard=True,rand_cb="False",scale_mode="per_cluster",
                 eps=1e-5,prior_mode="uniform"):
        """
        m: codebook number
        rand_cb=False, scale_mode="hard" is equvailent to VQVAE
        rand_cb=False, scale_mode="per_cluster" is soft cluster VQVAE
        rand_cb=True, scale_mode="per_cluster"/"per_dimension" is stochastic cb VQVAE
        """
        super().__init__()
        self.m=m
        self.k=k
        self.d=d  # dim per codeword
        self.c=self.m*self.d
        self.st_gumbel=hard
        if rand_cb=="False":
            self.rand_cb=False
        elif rand_cb=="True":
            self.rand_cb=True
        else:
            raise NotImplementedError
        self.scale_mode=scale_mode
        self.prior_mode=prior_mode
        self.eps=eps
        self.mus=nn.Parameter(torch.Tensor(self.m,self.k,self.d))
        self.rcn=8
        nn.init.uniform_(self.mus,-1/self.k,1/self.k)
        if self.scale_mode=="per_cluster":
            self.scales=nn.Parameter(torch.ones([self.m, self.k, 1], requires_grad=True)/math.sqrt(2))
        elif self.scale_mode=="hard":
            self.scales=nn.Parameter(torch.ones([1,1,1],requires_grad=False)/math.sqrt(2))
            self.scales.requires_grad=False
        elif self.scale_mode=="uniform":
            self.scales=nn.Parameter(torch.ones([1,1,1],requires_grad=True)/math.sqrt(2))
        elif self.scale_mode=="per_dimension":
            self.scales=nn.Parameter(torch.ones([self.m,self.k,self.d],requires_grad=True)/math.sqrt(2))
        else:
            raise NotImplementedError
        if self.prior_mode=="uniform":
            self.log_py_raw=nn.Parameter(torch.ones([self.m,self.k],requires_grad=False))
            self.log_py_raw.requires_grad=False
        elif self.prior_mode=="categorical":
            self.log_py_raw=nn.Parameter(torch.ones([self.m,self.k],requires_grad=True))
        else:
            raise NotImplementedError

    def forward(self, input, tau=0.5):
        # input: B,C,H,W
        b, c, h, w = input.shape
        assert(c == self.c)
        m = self.m
        k = self.k
        d = self.d
        input=input.permute(0,2,3,1).reshape(b * h * w, m, d)  # bhw m d
        dist=torch.distributions.normal.Normal(self.mus, torch.clamp(self.scales,min=self.eps))  # m k d
        log_probs_raw = torch.sum(dist.log_prob(input[..., None, :]), dim=-1)  # bhw m k
        log_probs_raw += self.log_py_raw
        probs = F.softmax(log_probs_raw, dim=-1)  # bhw m k
        log_probs=F.log_softmax(log_probs_raw, dim=-1) # bhw m k
        if self.training:
            onehot=F.gumbel_softmax(log_probs_raw,tau=tau,hard=self.st_gumbel,eps=self.eps,dim=-1)
        else:
            # MLE
            onehot=torch.argmax(probs,dim=-1,keepdim=True)
            onehot=torch.zeros_like(probs).scatter_(-1,onehot,1)
        if self.rand_cb and self.training:
            onehot=onehot.reshape(b,h,w,m,k)
            onehot=torch.repeat_interleave(onehot,repeats=self.rcn,dim=0).reshape(b,self.rcn,h,w,m,k)
            reparam_mu=torch.repeat_interleave(torch.zeros_like(self.mus),repeats=self.rcn,dim=0).reshape(self.rcn,m,k,d)
            reparam_sigma=torch.ones_like(reparam_mu)
            dist_eps = torch.distributions.normal.Normal(reparam_mu,reparam_sigma)
            reparam_sample = dist_eps.sample() * self.scales[None] + self.mus[None]
            # bmk, mkd -> bmd 
            # bnmk, nmkd -> bnmd 
            sample = torch.einsum('bnhwmk, nmkd -> bnhwmd', onehot, reparam_sample)
            sample = sample.reshape(b*self.rcn, h, w, m*d).permute(0, 3, 1, 2)
        else:
            sample = torch.einsum('bmk, mkd -> bmd', onehot, self.mus)
            sample = sample.reshape(b, h, w, m*d).permute(0, 3, 1, 2)
        # klde = probs*(log_probs+np.log(self.k))
        klde = probs*(log_probs-(self.log_py_raw-torch.logsumexp(self.log_py_raw,dim=-1,keepdim=True))[None])
        klde[(probs == 0).expand_as(klde)] = 0 # force 0 log 0 = 0
        klde = klde.reshape(b,-1)
        kldesum = klde.sum(-1).mean()
        return sample, kldesum, torch.zeros_like(kldesum)


class VQEmbeddingGSSoft(nn.Module):
    def __init__(self, latent_dim, num_embeddings, embedding_dim):
        super(VQEmbeddingGSSoft, self).__init__()
        # latent_dim: code book number
        # num_embeddings: categorical size
        # embedding dim: latent dim
        self.embedding = nn.Parameter(torch.Tensor(latent_dim, num_embeddings, embedding_dim))
        nn.init.uniform_(self.embedding, -1/num_embeddings, 1/num_embeddings)

    def forward(self, x):
        B, C, H, W = x.size()
        N, M, D = self.embedding.size()
        assert C == N * D

        x = x.view(B, N, D, H, W).permute(1, 0, 3, 4, 2)
        x_flat = x.reshape(N, -1, D)

        distances = torch.baddbmm(torch.sum(self.embedding ** 2, dim=2).unsqueeze(1) +
                                  torch.sum(x_flat ** 2, dim=2, keepdim=True),
                                  x_flat, self.embedding.transpose(1, 2),
                                  alpha=-2.0, beta=1.0)
        distances = distances.view(N, B, H, W, M)

        dist = RelaxedOneHotCategorical(0.5, logits=-distances)
        if self.training:
            samples = dist.rsample().view(N, -1, M)
        else:
            samples = torch.argmax(dist.probs, dim=-1)
            samples = F.one_hot(samples, M).float()
            samples = samples.view(N, -1, M)

        quantized = torch.bmm(samples, self.embedding)
        quantized = quantized.view_as(x)

        KL = dist.probs * (dist.logits + math.log(M))
        KL[(dist.probs == 0).expand_as(KL)] = 0
        KL = KL.sum(dim=(0, 2, 3, 4)).mean()

        avg_probs = torch.mean(samples, dim=1)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10), dim=-1))

        return quantized.permute(1, 0, 4, 2, 3).reshape(B, C, H, W), KL, perplexity.sum()


class Residual(nn.Module):
    def __init__(self, channels):
        super(Residual, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)


class Encoder(nn.Module):
    def __init__(self, channels, latent_dim, embedding_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels),
            Residual(channels),
            Residual(channels),
            nn.Conv2d(channels, latent_dim * embedding_dim, 1)
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, channels, latent_dim, embedding_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim * embedding_dim, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            Residual(channels),
            Residual(channels),
            nn.ConvTranspose2d(channels, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(channels, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, 3 * 256, 1)
        )

    def forward(self, x):
        x = self.decoder(x)
        B, _, H, W = x.size()
        x = x.view(B, 3, 256, H, W).permute(0, 1, 3, 4, 2)
        dist = Categorical(logits=x)
        return dist


class VQVAE(nn.Module):
    def __init__(self, channels, latent_dim, num_embeddings, embedding_dim):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(channels, latent_dim, embedding_dim)
        self.codebook = VQEmbeddingEMA(latent_dim, num_embeddings, embedding_dim)
        self.decoder = Decoder(channels, latent_dim, embedding_dim)

    def forward(self, x):
        x = self.encoder(x)
        x, loss, perplexity = self.codebook(x)
        dist = self.decoder(x)
        return dist, loss, perplexity


class GSSOFT(nn.Module):
    def __init__(self, channels, latent_dim, num_embeddings, embedding_dim, sigma_tag, rand_cb, prior_mode):
        super(GSSOFT, self).__init__()
        self.encoder = Encoder(channels, latent_dim, embedding_dim)
        if sigma_tag=="default":
            self.codebook = VQEmbeddingGSSoft(latent_dim, num_embeddings, embedding_dim)
        else:
            self.codebook = MultiCodebookSoftVQ(latent_dim,num_embeddings,embedding_dim,hard=False,rand_cb=rand_cb,scale_mode=sigma_tag,prior_mode=prior_mode)
        self.decoder = Decoder(channels, latent_dim, embedding_dim)

    def forward(self, x):
        x = self.encoder(x)
        x, KL, perplexity = self.codebook(x)
        dist = self.decoder(x)
        return dist, KL, perplexity

'''
class GSSOFTHyper(nn.Module):
    def __init__(self, channels, latent_dim, num_embeddings, embedding_dim, sigma_tag, rand_cb, prior_mode):
        super(GSSOFTHyper, self).__init__()
        # 32x32x3 -> 
        self.encoder1 = Encoder(channels, latent_dim, embedding_dim)
        self.encoder2 = Encoder(latent_dim * embedding_dim, latent_dim, embedding_dim)
        if sigma_tag=="default":
            self.codebook1 = VQEmbeddingGSSoft(latent_dim, num_embeddings, embedding_dim)
            self.codebook2 = VQEmbeddingGSSoft(latent_dim, num_embeddings, embedding_dim)
        else:
            self.codebook1 = MultiCodebookSoftVQ(latent_dim,num_embeddings,embedding_dim,hard=False,rand_cb=rand_cb,scale_mode=sigma_tag,prior_mode=prior_mode)
            self.codebook2 = MultiCodebookSoftVQ(latent_dim,num_embeddings,embedding_dim,hard=False,rand_cb=rand_cb,scale_mode=sigma_tag,prior_mode=prior_mode)
        self.decoder2 = Decoder(latent_dim * embedding_dim, latent_dim, embedding_dim)
        self.decoder1 = Decoder(channels, latent_dim, embedding_dim)

    def forward(self, x):
        x = self.encoder(x)
        x, KL, perplexity = self.codebook(x)
        dist = self.decoder(x)
        return dist, KL, perplexity
'''