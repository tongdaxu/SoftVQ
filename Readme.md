salloc --time=60 --gres=gpu:a100:1

srun --gres=gpu:a100:1 --time 10 --pty python train.py --model=GSSOFT --latent-dim=8 --num-embeddings=128 --tag=default

python train.py --model=VQVAE --latent-dim=8 --num-embeddings=128

python train.py --model=GSSOFT --latent-dim=8 --num-embeddings=128 --tag=default

srun --gres=gpu:a100:1 --time 30 --job-name "test" python train.py --model=VQVAE --latent-dim=8 --num-embeddings=128 

srun --gres=gpu:a100:1 --time 10 --job-name "test" python train.py --model=GSSOFT --latent-dim=8 --num-embeddings=128 --tag=per_dimension --randcb=True

# CIFAR10 VQVAE-EMA
* 15142_4294967294.out
# CIFAR10 VQVAE-Gumbel
* 15143_4294967294.out
# CIFAR10 SCAE-HardSigma
* 15442_4294967294.out
# CIFAR10 SCAE-PerCSigma
* 15443_4294967294.out
* 15807
# CIFAR10 SCAE-PerDSigma
* 15499_4294967294.out
* 15809_4294967294.out
* 15857_4294967294.out
# CIFAR10 SCAE-PerDSigma-Randcb-1
* 15515_4294967294.out
# CIFAR10 SCAE-PerDSigma-Randcb-8
* 15595_4294967294.out not long enough, die
* 15779
# CIFAR10 SCAE-PerDSigma-Uniprior (sanity check, should be same as CIFAR10 SCAE-PerDSigma)
* 15856_4294967294.out
# CIFAR10 SCAE-PerDSigma-Catprior
* 15859_4294967294.out
