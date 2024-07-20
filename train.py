"""
Training for CycleGAN

"""

import torch
from dataset import FaceSketchDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator


def train_fn(
    disc_F, disc_S, gen_S, gen_F, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler
):
    F_reals = 0
    F_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (sketch, face) in enumerate(loop):
        sketch = sketch.to(config.DEVICE)
        face = face.to(config.DEVICE)

        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_face = gen_F(sketch)
            D_F_real = disc_F(face)
            D_F_fake = disc_F(fake_face.detach())
            F_reals += D_F_real.mean().item()
            F_fakes += D_F_fake.mean().item()
            D_F_real_loss = mse(D_F_real, torch.ones_like(D_F_real))
            D_F_fake_loss = mse(D_F_fake, torch.zeros_like(D_F_fake))
            D_F_loss = D_F_real_loss + D_F_fake_loss

            fake_sketch = gen_S(face)
            D_S_real = disc_S(sketch)
            D_S_fake = disc_S(fake_sketch.detach())
            D_S_real_loss = mse(D_S_real, torch.ones_like(D_S_real))
            D_S_fake_loss = mse(D_S_fake, torch.zeros_like(D_S_fake))
            D_S_loss = D_S_real_loss + D_S_fake_loss

            # put it togethor
            D_loss = (D_F_loss + D_S_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_F_fake = disc_F(fake_face)
            D_S_fake = disc_S(fake_sketch)
            loss_G_F = mse(D_F_fake, torch.ones_like(D_F_fake))
            loss_G_S = mse(D_S_fake, torch.ones_like(D_S_fake))

            # cycle loss
            cycle_sketch = gen_S(fake_face)
            cycle_face = gen_F(fake_sketch)
            cycle_sketch_loss = l1(sketch, cycle_sketch)
            cycle_face_loss = l1(face, cycle_face)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_sketch = gen_S(sketch)
            identity_face = gen_F(face)
            identity_sketch_loss = l1(sketch, identity_sketch)
            identity_face_loss = l1(face, identity_face)

            # add all togethor
            G_loss = (
                loss_G_S
                + loss_G_F
                + cycle_sketch_loss * config.LAMBDA_CYCLE
                + cycle_face_loss * config.LAMBDA_CYCLE
                + identity_face_loss * config.LAMBDA_IDENTITY
                + identity_sketch_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(face*0.5+0.5, f"saved_images/real_photo_{idx}.png")
            save_image(fake_face*0.5+0.5, f"saved_images/sketch_to_photo_{idx}.png")
            save_image(sketch*0.5+0.5, f"saved_images/real_sketch_{idx}.png")
            save_image(fake_sketch*0.5+0.5, f"saved_images/photo_to_sketch_{idx}.png")

        loop.set_postfix(H_real=F_reals / (idx + 1), H_fake=F_fakes / (idx + 1))


def main():
    disc_F = Discriminator(in_channels=3).to(config.DEVICE)
    disc_S = Discriminator(in_channels=3).to(config.DEVICE)
    gen_F = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_S = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_F.parameters()) + list(disc_S.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_S.parameters()) + list(gen_F.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_F,
            gen_F,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_S,
            gen_S,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_F,
            disc_F,
            opt_disc,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_S,
            disc_S,
            opt_disc,
            config.LEARNING_RATE,
        )

    dataset = FaceSketchDataset(
        root_face=config.TRAIN_DIR + "/faces",
        root_sketch=config.TRAIN_DIR + "/sketches",
        transform=config.transforms,
    )
    val_dataset = FaceSketchDataset(
        root_face=config.VAL_DIR+"/faces",
        root_sketch=config.VAL_DIR+"/sketches",
        transform=config.transforms,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        print("EPOCH == ",epoch)
        train_fn(
            disc_F,
            disc_S,
            gen_S,
            gen_F,
            loader,
            opt_disc,
            opt_gen,
            L1,
            mse,
            d_scaler,
            g_scaler,
        )

        if config.SAVE_MODEL:
            save_checkpoint(gen_F, opt_gen, filename=config.CHECKPOINT_GEN_F)
            save_checkpoint(gen_S, opt_gen, filename=config.CHECKPOINT_GEN_S)
            save_checkpoint(disc_F, opt_disc, filename=config.CHECKPOINT_CRITIC_F)
            save_checkpoint(disc_S, opt_disc, filename=config.CHECKPOINT_CRITIC_S)


if __name__ == "__main__":
    main()
