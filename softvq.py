
class Quantizer_SoftVQ(nn.Module):
    def __init__(self,k,c,st_gumbel=True,rand_cb=False,scale_mode="per_cluster",eps=1e-5):
        super(Quantizer_SoftVQ,self).__init__()
        # rand_cb=False, scale_mode="hard" is equvailent to VQVAE
        # rand_cb=False, scale_mode="per_cluster" is soft cluster VQVAE
        # rand_cb=True, scale_mode="per_cluster"/"per_dimension" is stochastic cb VQVAE
        self.k=k
        self.c=c
        self.st_gumbel=st_gumbel
        self.rand_cb=rand_cb
        self.scale_mode=scale_mode
        self.eps=eps
        self.mus=nn.Parameter(torch.randn([self.k,self.c],requires_grad=True))
        if self.scale_mode=="hard":
            self.scales=nn.Parameter(torch.ones([1,1],requires_grad=False))
            self.scales.requires_grad=False
        if self.scale_mode=="uniform":
            self.scales=nn.Parameter(torch.ones([1,1],requires_grad=True))
        elif self.scale_mode=="per_cluster":
            self.scales=nn.Parameter(torch.ones([self.k,1],requires_grad=True))
        elif self.scale_mode=="per_dimension":
            self.scales=nn.Parameter(torch.ones([self.k,self.c],requires_grad=True))
        else:
            raise NotImplementedError

    def forward(self,input,tau=1.0):
        # input: B,C,H,W
        b,c,h,w=input.shape
        assert(c==self.c)
        input=input.permute(0,2,3,1).reshape(-1,c)
        dist=torch.distributions.normal.Normal(self.mus,torch.clamp(self.scales,min=self.eps)) # k*c
        log_probs_raw=torch.sum(dist.log_prob(input[:,None,:]),dim=2)
        probs=F.softmax(log_probs_raw,dim=1)
        log_probs=F.log_softmax(log_probs_raw,dim=1) # b*h*w,k
        if self.training:
            onehot=F.gumbel_softmax(log_probs,tau=tau,hard=self.st_gumbel,eps=self.eps,dim=1)
            if self.rand_cb:
                dist_eps=torch.distributions.normal.Normal(torch.zeros_like(self.mus),torch.ones_like(self.scales))
                reparam_sample=dist_eps.sample()*self.scales+self.mus
                sample=torch.matmul(onehot,reparam_sample)
            else:
                sample=torch.matmul(onehot,self.mus)
            if self.st_gumbel==True:
                sample_logeqzCx=torch.sum(onehot*log_probs,dim=1).reshape(b,1,h,w)
            else: # UD log q
                sample_logeqzCx=None
        else:
            # MLE
            onehot=torch.argmax(probs,dim=1,keepdim=True)
            onehot=torch.zeros_like(probs).scatter_(1,onehot,1)
            sample=torch.matmul(onehot,self.mus)
            sample_logeqzCx=torch.ones([b,1,h,w])
        sample=sample.reshape(b,h,w,c).permute(0,3,1,2)
        klde=torch.sum(probs*(log_probs-np.log(1.0/self.k)),dim=1)
        klde=klde.reshape(b,1,h,w)
        return sample,sample_logeqzCx,klde

class Quantizer_SoftVQWrapper(nn.Module):
    def __init__(self,k,c,st_gumbel=True,rand_cb=False,scale_mode="per_cluster",eps=1e-5):
        super(Quantizer_SoftVQWrapper,self).__init__()
        self.impl=Quantizer_SoftVQ(k,c,st_gumbel,rand_cb,scale_mode,eps)

    def forward(self,input,tau=1.0):
        sample,_,klde=self.impl(input,tau)
        kld_per_pix = klde.flatten(1).sum(-1).mean() / np.log(2) / (input.shape[-1] * input.shape[-2])
        return sample, kld_per_pix


class MultiCodebookSoftVQ(nn.Module):
    def __init__(self, m, k, n, hard=True, rand_cb=False, scale_mode="per_cluster", eps=1e-5, resolution_scale=256):
        """
        rand_cb=False, scale_mode="hard" is equvailent to VQVAE
        rand_cb=False, scale_mode="per_cluster" is soft cluster VQVAE
        rand_cb=True, scale_mode="per_cluster"/"per_dimension" is stochastic cb VQVAE
        """

        super().__init__()
        self.m = m
        self.k = k
        self.d = d = n // m  # dim per codeword
        self.c = n
        self.st_gumbel = hard
        self.rand_cb = rand_cb
        self.scale_mode = scale_mode
        self.eps = eps
        self.mus = nn.Parameter(torch.randn([m, k, d], requires_grad=True))
        self.resolution_scale = resolution_scale

        if self.scale_mode == "per_cluster":
            self.scales = nn.Parameter(torch.ones([m, k, 1], requires_grad=True))
        # elif self.scale_mode=="hard":
        #     self.scales=nn.Parameter(torch.ones([1,1],requires_grad=False))
        #     self.scales.requires_grad=False
        # elif self.scale_mode=="uniform":
        #     self.scales=nn.Parameter(torch.ones([1,1],requires_grad=True))
        # elif self.scale_mode=="per_dimension":
        #     self.scales=nn.Parameter(torch.ones([self.k,self.c],requires_grad=True))
        else:
            raise NotImplementedError

    def forward(self, input, tau=1.0):
        # input: B,C,H,W
        b, c, h, w = input.shape
        assert(c == self.c)
        m = self.m
        d = self.d
        input=input.permute(0,2,3,1).reshape(b * h * w, m, d)  # bhw m d

        # generate gaussian distributions N(mu, sigma)
        dist=torch.distributions.normal.Normal(self.mus, torch.clamp(self.scales,min=self.eps))  # m k d

        log_probs_raw = torch.sum(dist.log_prob(input[..., None, :]), dim=-1)  # bhw m k
        probs = F.softmax(log_probs_raw, dim=-1)  # bhw m k
        log_probs=F.log_softmax(log_probs_raw, dim=-1) # bhw m k
        if self.training:
            onehot=F.gumbel_softmax(log_probs_raw,tau=tau,hard=self.st_gumbel,eps=self.eps,dim=-1)
        else:
            # MLE
            onehot=torch.argmax(probs,dim=-1,keepdim=True)
            onehot=torch.zeros_like(probs).scatter_(-1,onehot,1)

        if self.rand_cb and self.training:
            dist_eps = torch.distributions.normal.Normal(torch.zeros_like(self.mus), torch.ones_like(self.scales))
            reparam_sample = dist_eps.sample() * self.scales + self.mus
            sample = torch.einsum('bmk, mkd -> bmd', onehot, reparam_sample)
        else:
            sample = torch.einsum('bmk, mkd -> bmd', onehot, self.mus)

        sample = sample.reshape(b, h, w, m*d).permute(0, 3, 1, 2)

        if self.training:
            klde = torch.sum(probs*(log_probs+np.log(self.k)), dim=-1)  # bhw m
        else:
            klde = torch.sum(probs*torch.full_like(log_probs, np.log(self.k)), dim=-1)  # bhw m

        klde = klde.reshape(b, h * w * m)
        kld_per_pix = klde.sum(-1).mean() / np.log(2) / (h * w * self.resolution_scale)

        return sample, kld_per_pix


class HyperSoftVQ(nn.Module):
    def __init__(self, m, k, n, hard=True, rand_cb=False, scale_mode="per_cluster", prior_kernel='l2', eps=1e-5, resolution_scale=256):
        """
        rand_cb=False, scale_mode="hard" is equvailent to VQVAE
        rand_cb=False, scale_mode="per_cluster" is soft cluster VQVAE
        rand_cb=True, scale_mode="per_cluster"/"per_dimension" is stochastic cb VQVAE
        """

        super().__init__()
        self.m = m
        self.k = k
        self.d = d = n // m  # dim per codeword
        self.c = n
        self.st_gumbel = hard
        self.rand_cb = rand_cb
        self.scale_mode = scale_mode
        self.eps = eps
        self.mus = nn.Parameter(torch.randn([m, k, d], requires_grad=True))
        self.resolution_scale = resolution_scale
        self.prior_kernel=prior_kernel

        if self.scale_mode == "per_cluster":
            self.scales = nn.Parameter(torch.ones([m, k, 1], requires_grad=True))
        # elif self.scale_mode=="hard":
        #     self.scales=nn.Parameter(torch.ones([1,1],requires_grad=False))
        #     self.scales.requires_grad=False
        # elif self.scale_mode=="uniform":
        #     self.scales=nn.Parameter(torch.ones([1,1],requires_grad=True))
        # elif self.scale_mode=="per_dimension":
        #     self.scales=nn.Parameter(torch.ones([self.k,self.c],requires_grad=True))
        else:
            raise NotImplementedError

    def forward(self, input, prior, tau=1.0):
        # input: B,C,H,W
        b, c, h, w = input.shape
        assert(c == self.c)
        m = self.m
        k = self.k
        d = self.d
        input=input.permute(0, 2, 3, 1).reshape(b * h * w, m, d)  # bhw m d

        # generate gaussian distributions N(mu, sigma)
        dist=torch.distributions.normal.Normal(self.mus, torch.clamp(self.scales,min=self.eps))  # m k d

        log_probs_raw = torch.sum(dist.log_prob(input[..., None, :]), dim=-1)  # bhw m k
        probs = F.softmax(log_probs_raw, dim=-1)  # bhw m k
        log_probs=F.log_softmax(log_probs_raw, dim=-1) # bhw m k
        if self.training:
            onehot=F.gumbel_softmax(log_probs_raw,tau=tau,hard=self.st_gumbel,eps=self.eps,dim=-1)
        else:
            # MLE
            onehot=torch.argmax(probs,dim=-1,keepdim=True)
            onehot=torch.zeros_like(probs).scatter_(-1,onehot,1)

        if self.prior_kernel == 'l2':
            prior = prior.permute(0, 2, 3, 1).reshape(b * h * w, m, 1, d)
            prior_logits = torch.sum(dist.log_prob(prior), dim=-1)  # bhw m k
        elif self.prior_kernel == 'identity':
            prior_logits = prior.permute(0, 2, 3, 1).reshape(b * h * w, m, k)
        else:
            raise ValueError(self.prior_kernel)
        logp = F.log_softmax(prior_logits, dim=-1)

        if self.rand_cb and self.training:
            dist_eps = torch.distributions.normal.Normal(torch.zeros_like(self.mus), torch.ones_like(self.scales))
            reparam_sample = dist_eps.sample() * self.scales + self.mus
            sample = torch.einsum('bmk, mkd -> bmd', onehot, reparam_sample)
        else:
            sample = torch.einsum('bmk, mkd -> bmd', onehot, self.mus)

        sample = sample.reshape(b, h, w, m*d).permute(0, 3, 1, 2)

        if self.training:
            klde = torch.sum(probs*(log_probs-logp), dim=-1)  # bhw m
        else:
            klde = torch.sum(-probs*logp, dim=-1)  # bhw m

        klde = klde.reshape(b, h * w * m)
        kld_per_pix = klde.sum(-1).mean() / np.log(2) / (h * w * self.resolution_scale)

        return sample, kld_per_pix