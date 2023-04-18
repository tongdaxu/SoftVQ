import argparse
from pathlib import Path

import numpy as np

from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, utils

from model import VQVAE, GSSOFT


def save_checkpoint(model, optimizer, step, checkpoint_dir):
    checkpoint_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step}
    checkpoint_path = checkpoint_dir / "model.ckpt-{}.pt".format(step)
    torch.save(checkpoint_state, checkpoint_path)
    print("Saved checkpoint: {}".format(checkpoint_path))


def shift(x):
    return x - 0.5


def train_gssoft(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GSSOFT(args.channels, args.latent_dim, args.num_embeddings,
                   args.embedding_dim, args.tag, args.randcb, args.priormode)
    model.to(device)

    model_name = "{}_C_{}_N_{}_M_{}_D_{}_sigma_{}_rcb_{}_prior_{}_train_prior".format(args.model, args.channels, args.latent_dim,
                                                 args.num_embeddings, args.embedding_dim, args.tag, args.randcb, args.priormode)
    print("saving: {}",model_name)
    checkpoint_dir = Path(model_name)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=Path("runs") / model_name)
    model.codebook.log_py_raw.requires_grad=True
    optimizer = optim.Adam([model.codebook.log_py_raw], lr=args.learning_rate)
    assert args.resume is not None
    print("Resume checkpoint from: {}:".format(args.resume))
    checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["model"])
    global_step = 0

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(shift)
    ])
    training_dataset = datasets.CIFAR10("./CIFAR10", train=True, download=True,
                                        transform=transform)

    test_dataset = datasets.CIFAR10("./CIFAR10", train=False, download=True,
                                    transform=transform)

    training_dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.num_workers, pin_memory=True)

    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=True,
                                 num_workers=args.num_workers, pin_memory=True)

    num_epochs = args.num_training_steps // len(training_dataloader) + 1
    start_epoch = global_step // len(training_dataloader) + 1

    N = 3 * 32 * 32
    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        average_logp = average_KL = average_elbo = average_bpd = average_perplexity = 0
        for i, (images, _) in enumerate(tqdm(training_dataloader), 1):
            images = images.to(device)
            dist, KL, perplexity = model(images)
            b,c,h,w=images.shape
            if args.tag != "default" and model.codebook.rand_cb:
                images=torch.repeat_interleave(images,repeats=model.codebook.rcn,dim=0).reshape(b*model.codebook.rcn,c,h,w)
                targets = (images + 0.5) * 255
                targets = targets.long()
                logp_raw=dist.log_prob(targets).reshape(b,model.codebook.rcn,c,h,w)
                logp_raw=torch.logsumexp(logp_raw, dim=1)-torch.log(torch.tensor(model.codebook.rcn+0.0,device=device))
                logp = logp_raw.sum((1, 2, 3)).mean()
            else:
                targets = (images + 0.5) * 255
                targets = targets.long()
                logp = dist.log_prob(targets).sum((1, 2, 3)).mean()
            loss = (KL - logp) / N
            elbo = (KL - logp) / N
            bpd = elbo / np.log(2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

            if global_step % 25000 == 0:
                save_checkpoint(model, optimizer, global_step, checkpoint_dir)

            average_logp += (logp.item() - average_logp) / i
            average_KL += (KL.item() - average_KL) / i
            average_elbo += (elbo.item() - average_elbo) / i
            average_bpd += (bpd.item() - average_bpd) / i
            average_perplexity += (perplexity.item() - average_perplexity) / i

        writer.add_scalar("logp/train", average_logp, epoch)
        writer.add_scalar("kl/train", average_KL, epoch)
        writer.add_scalar("elbo/train", average_elbo, epoch)
        writer.add_scalar("bpd/train", average_bpd, epoch)
        writer.add_scalar("perplexity/train", average_perplexity, epoch)

        model.eval()
        average_logp = average_KL = average_elbo = average_bpd = average_perplexity = 0
        for i, (images, _) in enumerate(test_dataloader, 1):
            images = images.to(device)

            with torch.no_grad():
                dist, KL, perplexity = model(images)

            targets = (images + 0.5) * 255
            targets = targets.long()
            logp = dist.log_prob(targets).sum((1, 2, 3)).mean()
            elbo = (KL - logp) / N
            bpd = elbo / np.log(2)

            average_logp += (logp.item() - average_logp) / i
            average_KL += (KL.item() - average_KL) / i
            average_elbo += (elbo.item() - average_elbo) / i
            average_bpd += (bpd.item() - average_bpd) / i
            average_perplexity += (perplexity.item() - average_perplexity) / i

        writer.add_scalar("logp/test", average_logp, epoch)
        writer.add_scalar("kl/test", average_KL, epoch)
        writer.add_scalar("elbo/test", average_elbo, epoch)
        writer.add_scalar("bpd/test", average_bpd, epoch)
        writer.add_scalar("perplexity/test", average_perplexity, epoch)

        samples = torch.argmax(dist.logits, dim=-1)
        grid = utils.make_grid(samples.float() / 255)
        writer.add_image("reconstructions", grid, epoch)

        print("epoch:{}, logp:{:.3E}, KL:{:.3E}, elbo:{:.3f}, bpd:{:.3f}, perplexity:{:.3f}"
              .format(epoch, average_logp, average_KL, average_elbo, average_bpd, average_perplexity))
        
        if epoch==21:
            return


if __name__ == "__main__":
    import random
    import numpy as np
    SEED=3470
    random.seed(SEED)
    np.random.seed(SEED) 
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic=True

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume.")
    parser.add_argument("--model", choices=["VQVAE", "GSSOFT"], help="Select model to train (either VQVAE or GSSOFT)")
    parser.add_argument("--channels", type=int, default=256, help="Number of channels in conv layers.")
    parser.add_argument("--latent-dim", type=int, default=8, help="Dimension of categorical latents.")
    parser.add_argument("--num-embeddings", type=int, default=128, help="Number of codebook embeddings size.")
    parser.add_argument("--embedding-dim", type=int, default=32, help="Dimension of codebook embeddings.")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--num-training-steps", type=int, default=250000, help="Number of training steps.")
    parser.add_argument("--tag", type=str, default="default", help="tag name of model.")
    parser.add_argument("--randcb", type=str, default="False", help="tag name of model.")
    parser.add_argument("--priormode", type=str, default="uniform", help="tag name of model.")

    args = parser.parse_args()
    if args.model == "VQVAE":
        train_vqvae(args)
    if args.model == "GSSOFT":
        train_gssoft(args)