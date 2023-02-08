from torch import optim, nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import pytorch_lightning as pl
import numpy as np
from utils.visualization import imshow_unnorm
import torchvision
import torch.nn.init as init


def reparametrize(mu, logvar):
    std = torch.exp(0.5 * logvar)  # e^(1/2 * log(std^2))
    eps = torch.randn_like(std)  # random ~ N(0, 1)
    return mu + std * eps


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class FoVAE(pl.LightningModule):
    def __init__(
        self,
        patch_dim=5,
        patch_channels=3,
        z_dim=10,
        n_steps: int = 1,
        lr=1e-3,
        beta=1,
        # grad_clip=100,
        grad_skip_threshold=1000,
        do_add_pos_encoding=True,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.z_dim = z_dim
        self.patch_dim = patch_dim
        if do_add_pos_encoding:
            self.num_channels = patch_channels + 2
        else:
            self.num_channels = patch_channels
        self.lr = lr
        self.beta = beta
        self.encoder = nn.Sequential(
            View((-1, self.patch_dim * self.patch_dim * self.num_channels)),
            nn.GELU(),
            nn.Linear(self.patch_dim * self.patch_dim * self.num_channels, 1024),
            nn.GELU(),
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Linear(256, z_dim * 2),
        )
        self.decoder = nn.Sequential(
            nn.GELU(),
            nn.Linear(z_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1024),
            nn.GELU(),
            nn.Linear(1024, self.patch_dim * self.patch_dim * self.num_channels),
            View((-1, self.num_channels, self.patch_dim, self.patch_dim)),
        )
        # self.encoder = nn.Sequential(
        #     # nn.Conv2d(3, 32, 4, 2, 1),          # B,  32, 32, 32
        #     # nn.ReLU(True),
        #     nn.Conv2d(3, 32, 4, 2, 1),          # B,  32, 16, 16
        #     nn.ReLU(True),
        #     nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
        #     nn.ReLU(True),
        #     nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  4,  4
        #     nn.ReLU(True),
        #     nn.Conv2d(64, 256, 4, 1),            # B, 256,  1,  1
        #     nn.ReLU(True),
        #     View((-1, 256*1*1)),                 # B, 256
        #     nn.Linear(256, z_dim*2),             # B, z_dim*2
        # )
        # self.decoder = nn.Sequential(
        #     nn.Linear(z_dim, 256),               # B, 256
        #     View((-1, 256, 1, 1)),               # B, 256,  1,  1
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(256, 64, 4),      # B,  64,  4,  4
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  64,  8,  8
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 16, 16
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(32, 3, 4, 2, 1), # B,  nc, 32, 32
        #     # nn.ReLU(True),
        #     # nn.ConvTranspose2d(32, 3, 4, 2, 1),  # B, nc, 64, 64
        # )
        # for block in self._modules:
        #     for m in self._modules[block]:
        #         kaiming_init(m)

        # self.grad_clip = grad_clip
        self.grad_skip_threshold = grad_skip_threshold
        self.do_add_pos_encoding = do_add_pos_encoding

        # Disable automatic optimization!
        # self.automatic_optimization = False

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        b, c, h, w = x.size()

        if self.do_add_pos_encoding:
            x_full = self._add_pos_encodings_to_img_batch(x)
        else:
            x_full = x

        patch_loss, patch_rec_loss, patch_kl_div = (
            0.0,
            0.0,
            0.0,
        )
        for step in range(self.n_steps):
            patch = x_full[:, :, : self.patch_dim, : self.patch_dim]
            mu, logvar = self._encode_patch(patch)
            z = reparametrize(mu, logvar)
            patch_recon = self._decode_patch(z)
            assert torch.is_same_size(patch_recon, patch)
            loss, rec_loss, kl_div = self._loss(patch, patch_recon, mu, logvar)
            patch_loss += loss
            patch_rec_loss += rec_loss
            patch_kl_div += kl_div

        return (loss, rec_loss, kl_div), patch_recon, mu, logvar

    def _loss(self, x: torch.Tensor, x_recon: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor):
        # reconstruction losses are summed over all elements and batch
        # recon loss is MSE ONLY FOR DECODERS PRODUCING GAUSSIAN DISTRIBUTIONS
        recon_loss = F.mse_loss(x_recon, x, reduction="sum")

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # divergence from standard-normal
        kl_diverge = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp().pow(2))

        # divide losses by batch size
        recon_loss /= x.shape[0]
        kl_diverge /= x.shape[0]

        return (recon_loss + self.beta * kl_diverge), recon_loss, kl_diverge

    def _encode_patch(self, x: torch.Tensor):
        distributions = self.encoder(x)
        mu = distributions[:, : self.z_dim]
        logvar = distributions[:, self.z_dim :]
        return mu, logvar

    def _decode_patch(self, z):
        return self.decoder(z)

    def _add_pos_encodings_to_img_batch(self, x: torch.Tensor):
        b, c, h, w = x.size()
        # add position encoding as in wattersSpatialBroadcastDecoder2019
        width_pos = torch.linspace(-1, 1, w)
        height_pos = torch.linspace(-1, 1, w)
        xb, yb = torch.meshgrid(width_pos, height_pos)
        # match dimensions of x except for channels
        xb = xb.expand(b, 1, -1, -1).to(x.device)
        yb = yb.expand(b, 1, -1, -1).to(x.device)
        x_full = torch.concat((x, xb, yb), dim=1)
        assert x_full.size() == torch.Size([b, c + 2, h, w])
        return x_full

    # generate n=num images using the model
    def generate(self, num: int):
        self.eval()
        z = torch.randn(num, self.z_dim)
        with torch.no_grad():
            return self._decode_patch(z).cpu()

    # returns z for position-augmented patch
    def get_patch_zs(self, patch_with_pos: torch.Tensor):
        assert patch_with_pos.ndim == 4
        self.eval()

        with torch.no_grad():
            mu, logvar = self._encode_patch(patch_with_pos)
            z = reparametrize(mu, logvar)

        return z

    # def linear_interpolate(self, im1, im2):
    #     self.eval()
    #     z1 = self.get_z(im1)
    #     z2 = self.get_z(im2)

    #     factors = np.linspace(1, 0, num=10)
    #     result = []

    #     with torch.no_grad():
    #         for f in factors:
    #             z = f * z1 + (1 - f) * z2
    #             im = torch.squeeze(self._decode_patch(z).cpu())
    #             result.append(im)

    #     return result

    def latent_traverse(self, z, range_limit=3, step=0.5, around_z=False):
        self.eval()
        interpolation = torch.arange(-range_limit, range_limit + 0.1, step)
        samples = []
        with torch.no_grad():
            for row in range(self.z_dim):
                row_samples = []
                # copy to CPU to bypass https://github.com/pytorch/pytorch/issues/94390
                interp_z = z.clone().to("cpu")
                for val in interpolation:
                    if around_z:
                        interp_z[:, row] += val
                    else:
                        interp_z[:, row] = val
                    sample = self._decode_patch(interp_z.to(self.device)).data.cpu()
                    row_samples.append(sample)
                samples.append(row_samples)
        return samples

    def training_step(self, batch, batch_idx):
        x, y = batch
        (loss, rec_loss, kl_div), x_recon, _, _ = self.forward(x, y)
        self.log("train_loss", loss)
        self.log("train_rec_loss", rec_loss, prog_bar=True)
        self.log("train_kl_div", kl_div, prog_bar=True)

        # self._optimizer_step(loss)
        skip_update = float(torch.isnan(loss))  # TODO: skip on grad norm
        if skip_update:
            print(f"Skipping update! {loss=}, {rec_loss=}, {kl_div=}")

        self.log("n_skipped_steps", skip_update, on_epoch=True, logger=True, reduce_fx=torch.sum)
        # self.log(grad_norm, skip_update, on_epoch=True, logger=True)

        return None if skip_update else loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        (loss, rec_loss, kl_div), x_recon, _, _ = self.forward(x, y)
        self.log("val_loss", loss)
        self.log("val_rec_loss", rec_loss)
        self.log("val_kl_div", kl_div)
        if batch_idx == 0:
            tensorboard = self.logger.experiment
            # real = torchvision.utils.make_grid(x).cpu()
            # recon = torchvision.utils.make_grid(x_recon).cpu()
            # img = torch.concat((real, recon), dim=1)

            # step constant bc real images don't change
            def remove_pos_channels_from_batch(g):
                n_pos_channels = 2 if self.do_add_pos_encoding else 0
                return g[:, :-n_pos_channels, :, :]

            tensorboard.add_images("Real Images", remove_pos_channels_from_batch(x[:32]), global_step=0)
            tensorboard.add_images(
                "Reconstructed Images", remove_pos_channels_from_batch(x_recon[:32]), global_step=self.global_step
            )


            # TODO: this traversal stuff makes no sense on sub-image patches!

            def stack_traversal_output(g):
                # stack by interp image, then squeeze out the singular batch dimension and index out the 2 position channels
                return [remove_pos_channels_from_batch(torch.stack(dt).squeeze(1)) for dt in traversal_abs]

            img = self._add_pos_encodings_to_img_batch(x[[0]])
            traversal_abs = self.latent_traverse(self.get_patch_zs(img), range_limit=3, step=0.5)
            images_by_row_and_interp = stack_traversal_output(traversal_abs)

            tensorboard.add_image(
                "Absolute Latent Traversal",
                torchvision.utils.make_grid(
                    torch.concat(images_by_row_and_interp), nrow=self.z_dim
                ),
                global_step=self.global_step,
            )
            traversal_around = self.latent_traverse(
                self.get_patch_zs(img), range_limit=3, step=0.5, around_z=True
            )
            images_by_row_and_interp = stack_traversal_output(traversal_around)

            tensorboard.add_image(
                "Latent Traversal Around Z",
                torchvision.utils.make_grid(
                    torch.concat(images_by_row_and_interp), nrow=self.z_dim
                ),
                global_step=self.global_step,
            )

        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    # def _get_grad_norm(self):
    #     total_norm = 0
    #     parameters = [p for p in self.parameters() if p.grad is not None and p.requires_grad]
    #     for p in parameters:
    #         param_norm = p.grad.detach().data.norm(2)
    #         total_norm += param_norm.item() ** 2
    #     total_norm = total_norm ** 0.5
    #     return total_norm

    # def _optimizer_step(self, loss):
    #     opt = self.optimizers()
    #     opt.zero_grad()
    #     self.manual_backward(loss)

    #     grad_norm = self.clip_gradients(opt, gradient_clip_val=self.grad_clip, gradient_clip_algorithm="norm")

    #     # only update if loss is not NaN and if the grad norm is below a specific threshold
    #     skipped_update = 1
    #     if not torch.isnan(loss) and (
    #         self.grad_skip_threshold == -1 or grad_norm < self.grad_skip_threshold
    #     ):
    #         skipped_update = 0
    #         opt.step()
    #         # TODO: EMA updating
    #         # TODO: factor out loss NaNs by what produced them (kl or reconstruction)
    #         # update_ema(vae, ema_vae, H.ema_rate)

    #     self.log("n_skipped_steps", skipped_update, on_epoch=True, logger=True)

    # def optimizer_step(
    #     self,
    #     epoch,
    #     batch_idx,
    #     optimizer,
    #     optimizer_idx,
    #     optimizer_closure,
    #     on_tpu=False,
    #     using_lbfgs=False,
    # ):
    #     # skip updates with nans
    #     if True:
    #         # the closure (which includes the `training_step`) will be executed by `optimizer.step`
    #         optimizer.step(closure=optimizer_closure)
    #     else:
    #         # call the closure by itself to run `training_step` + `backward` without an optimizer step
    #         optimizer_closure()


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()
