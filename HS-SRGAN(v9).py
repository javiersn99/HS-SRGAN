import numpy as np
import argparse
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import scipy.io
import mat73
from torchmetrics.image import SpectralAngleMapper
from collections import OrderedDict
from pytorch_msssim import ssim
import shutil

# LR HS Cube size = 128 x  64 x 128  (bands x X x Y)
# HR HS Cube size = 128 x 128 x 256 (bands x X x Y)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.batchnorm(x)
        x = self.prelu(x)
        x = self.conv2(x)
        x = self.batchnorm(x)
        x += residual
        return x

class Generator(nn.Module):
    def __init__(self, args, num_residual_blocks=16, num_bands=256, num_features=256):
        super(Generator, self).__init__()
        self.bands = num_bands

        self.conv1 = nn.Conv2d(self.bands, num_features, 9, padding='same', stride=1)
        self.prelu = nn.PReLU()
        self.residual_blocks = nn.ModuleList([ResidualBlock(num_features, num_features, 3, 1, 1) for _ in range(num_residual_blocks)])
        self.conv2 = nn.Conv2d(num_features, num_features, 3, padding='same', stride=1)
        self.batchnorm = nn.BatchNorm2d(num_features)
        self.conv3 = nn.Conv2d(num_features, num_bands*4, 3, padding='same', stride=1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv4 = nn.Conv2d(self.bands, self.bands, 9, padding='same', stride=1)

    def forward(self, z):
        x = self.conv1(z)
        x = self.prelu(x)
        
        residual = x
        for block in self.residual_blocks:
            x = block(x)
        
        x = self.conv2(x)
        x = self.batchnorm(x)
        x += residual

        x = self.conv3(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)

        x = self.conv4(x)
        return x
    
class Discriminator_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Discriminator_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=1, padding='same')
        self.batchnorm1 = nn.BatchNorm2d(in_channels)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=2)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.leaky_relu(self.batchnorm1(self.conv1(x)))
        x = self.leaky_relu(self.batchnorm2(self.conv2(x)))
        return x

class Discriminator(nn.Module):
    def __init__(self, args, num_blocks=3, num_bands=256, X=128, Y=256, num_features=256):
        super(Discriminator, self).__init__()
        self.bands = num_bands
        self.X = X; self.Y = Y

        self.conv1 = nn.Conv2d(self.bands, num_features, 3, padding='same', stride=1)
        size = [self.X, self.Y]
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(num_features, num_features, 3, stride=2)
        size = [math.floor((size[0]-3)/2 + 1), math.floor((size[1]-3)/2 + 1)] 
        self.batchnorm = nn.BatchNorm2d(num_features)
        
        self.blocks = nn.ModuleList()
        input_channels = num_features

        for _ in range(num_blocks):
            size = [math.floor((size[0]-3)/2 + 1), math.floor((size[1]-3)/2 + 1)] 
            out_channels = input_channels*2
            self.blocks.append(Discriminator_block(input_channels, out_channels, 3))
            input_channels = out_channels
            
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(out_channels * size[0] * size[1], 1024)
        self.linear2 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.conv1(x)
        y = self.leaky_relu(y)
        
        y = self.leaky_relu(self.batchnorm(self.conv2(y)))
        
        for block in self.blocks:
            y = block(y)

        y = self.flatten(y)
        y = self.leaky_relu(self.linear1(y))
        y = self.sigmoid(self.linear2(y))
        return y

class HS_Dataset(Dataset):
    def __init__(self, data_dir):
        self.HR_dir = data_dir+"/HR/"
        self.LR_dir = data_dir+"/LR/"
        self.HR_data = os.listdir(self.HR_dir)
        self.LR_data = os.listdir(self.LR_dir)

        if len(self.HR_data) != len(self.LR_data):
            raise ValueError("Number of HR and LR images do not match")

    def __len__(self):
        return len(self.HR_data)
    
    def __getitem__(self, idx):
        LR = mat73.loadmat(os.path.join(self.LR_dir, self.LR_data[idx]))
        LR = torch.from_numpy(LR[list(LR.keys())[-1]]).float()
        
        HR = mat73.loadmat(os.path.join(self.HR_dir, self.HR_data[idx]))
        HR = torch.from_numpy(HR[list(HR.keys())[-1]]).float()

        return LR, HR

class GAN(nn.Module):
    def __init__(self, args):
        super(GAN, self).__init__()
        # Parameters
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.eval_batch_size = args.eval_batch_size
        self.epochs = args.epochs
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.log_dir = args.log_dir
        self.patience = args.patience if args.patience != -1 else self.epochs
        self.model_path = args.path_model

        # Nets
        self.generator = Generator(args)
        self.discriminator = Discriminator(args)
        
        # Data
        self.train_loader = DataLoader(HS_Dataset('../data/train'), batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        self.eval_loader = DataLoader(HS_Dataset('../data/eval'), batch_size=self.eval_batch_size, shuffle=False, num_workers=4, pin_memory=True)
        self.test_loader = DataLoader(HS_Dataset('../data/test'), batch_size=self.test_batch_size, shuffle=False, num_workers=4, pin_memory=True)
                
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        # Losses
        self.criterion_bce = nn.BCELoss();                          self.lambda_bce = 0.001
        self.criterion_mse = nn.MSELoss();                          self.lambda_mse = 0.5
        self.criterion_sam = SpectralAngleMapper().to(self.device); self.lambda_sam = 0.5
        self.eval_losses = []

        # noise params
        self.noise_mean = 0; self.noise_std = 0.001

        self.current_log_path = os.path.join("..","results",self.log_dir, datetime.now().strftime('%b%d_%H-%M-%S')) + "_GAN"
        os.makedirs(self.current_log_path, exist_ok=True)

        # copy code file 
        file_name = os.path.basename(__file__)
        os.system(f'cp {file_name} {self.current_log_path}/{file_name}')

        self.LR_fixed, self.HR_fixed = next(iter(self.eval_loader))
        self.LR_fixed = self.LR_fixed.to(self.device)
        self.HR_fixed = self.HR_fixed.to(self.device)
        get_model_memory_usage(self.generator)
        get_model_memory_usage(self.discriminator)

        self.printl(f'v7; losses: {self.lambda_bce}*bce_loss, {self.lambda_mse}*mse_loss, {self.lambda_tv}*tv_loss, {self.lambda_mae}*mae_loss, {self.lambda_sam}*sam_loss; \
optimizer: Adam, lr G:{self.lr} D:{self.lr}, betas: (0.5, 0.999); \
num_residual_blocks: 16; num_blocks: 3; batch_size: {self.batch_size}; \
epochs: {self.epochs}; log_interval: {self.log_interval}; device: {self.device}; patiente: {self.patience}; noise: N({self.noise_mean},{self.noise_std})')
        
    def train_GAN(self):
        self.printl('Training GAN...')
        for epoch in range(self.epochs):
            for batch_idx, (LR, HR) in enumerate(self.train_loader):
                LR = LR.to(self.device, non_blocking=True)
                # add low energy noise to the LR image
                LR = LR + torch.normal(self.noise_mean, self.noise_std, LR.size(), device=self.device)
                
                HR = HR.to(self.device, non_blocking=True)
                
                real_label = torch.ones(HR.size(0), 1).to(self.device, non_blocking=True)
                SR_label = torch.zeros(LR.size(0), 1).to(self.device, non_blocking=True)

                
                self.discriminator.zero_grad()
                output = self.discriminator(HR)
                loss_real = self.criterion_bce(output, real_label)
                loss_real.backward()
                
                SR = self.generator(LR)
                
                output = self.discriminator(SR.detach().to(self.device, non_blocking=True))
                loss_fake = self.criterion_bce(output, SR_label)
                loss_fake.backward()
                self.optimizer_D.step()
                
                # Train the generator
                self.generator.zero_grad()
                output = self.discriminator(SR)
                loss_G_bce = self.criterion_bce(output, real_label)
                loss_G_mse = self.criterion_mse(SR, HR)
                loss_G_mae = self.criterion_mae(SR, HR)
                loss_G_tv = self.criterion_tv(SR)
                loss_G_sam = self.criterion_sam(SR, HR)

                loss_G = self.lambda_bce * loss_G_bce + self.lambda_mse * loss_G_mse + self.lambda_sam * loss_G_sam
                loss_G.backward()
                
                self.optimizer_G.step()

                if batch_idx % self.log_interval == 0:
                    self.printl(f'Train Epoch: {epoch} [{batch_idx * len(LR)}/{len(self.train_loader.dataset)} ({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss_D: {loss_real.item() + loss_fake.item():.6f}\tLoss_G: {loss_G.item():.6f}\tTime: {datetime.now().strftime("%d-%m-%Y, %H:%M:%S")}')
                    
            with torch.no_grad():
                if self.eval(epoch):
                    break # Early stopping
                self.generate_images(epoch)
                self.printl('Generated image saved...')
        
        scipy.io.savemat(os.path.join(self.current_log_path, 'eval_losses.mat'), {'eval_losses': self.eval_losses})
    
    def getGeneratorLoss(self, HR, LR):
        SR = self.generator(LR)
        output = self.discriminator(SR)
        loss_G_bce = self.criterion_bce(output, torch.ones(HR.size(0), 1).to(self.device))
        loss_G_mse = self.criterion_mse(SR, HR)
        loss_G_sam = self.criterion_sam(SR, HR)
        return [loss_G_bce, loss_G_mse, loss_G_sam]

    def eval(self, epoch=-1, save=False):
        lambdas = [self.lambda_bce, self.lambda_mse, self.lambda_sam]
        with torch.no_grad():
            losses = []
            for idx, (LR, HR) in enumerate(self.eval_loader):
                print(f'Eval {idx/len(self.eval_loader)*100:.2f}%')
                
                loss = self.getGeneratorLoss(HR.to(self.device), LR.to(self.device))
                losses.append([lambdas[i]*loss[i].item() for i in range(len(loss))])
            
            self.eval_losses.append(np.mean(losses))
            self.printl(f'Eval Epoch: {epoch}\tLoss: {np.mean(losses)}')

            if len(self.eval_losses) > self.patience:
                if all([self.eval_losses[-1] > self.eval_losses[-i] for i in range(1, self.patience)]):
                    self.printl(f'Early stopping at epoch {epoch}...')
                    self.save_model(epoch)
                    # self.writer.close()
                    return True
                
            if np.mean(losses) <= min(self.eval_losses):
                self.save_model(epoch)
                self.generate_images(epoch)
                self.printl('Best model found...')
                
    def test(self, dataloader, save_classic=False):
        self.printl('Testing GAN...')
        nearest_PSNR = []; bilinear_PSNR = []; bicubic_PSNR = []; SR_PSNR = []
        nearest_SAM = []; bilinear_SAM = []; bicubic_SAM = []; SR_SAM = []
        nearest_SSIM = []; bilinear_SSIM = []; bicubic_SSIM = []; SR_SSIM = []
        nearest_MSE = []; bilinear_MSE = []; bicubic_MSE = []; SR_MSE = []

        nearest_path = os.path.join(self.current_log_path, 'nearest'); os.makedirs(nearest_path, exist_ok=True)
        bilinear_path = os.path.join(self.current_log_path, 'bilinear'); os.makedirs(bilinear_path, exist_ok=True)
        bicubic_path = os.path.join(self.current_log_path, 'bicubic'); os.makedirs(bicubic_path, exist_ok=True)
        SR_path = os.path.join(self.current_log_path, 'SR'); os.makedirs(SR_path, exist_ok=True)
        HR_path = os.path.join(self.current_log_path, 'HR'); os.makedirs(HR_path, exist_ok=True)
        LR_path = os.path.join(self.current_log_path, 'LR'); os.makedirs(LR_path, exist_ok=True)

        with torch.no_grad():
            for idx, (LR, HR) in enumerate(dataloader):
                self.printl(f'Testing {idx/len(dataloader)*100:.2f}%')

                SR = self.generator(LR.to(self.device))
                nearest_SR = torch.nn.functional.interpolate(LR, scale_factor=2, mode='nearest')
                bilinear_SR = torch.nn.functional.interpolate(LR, scale_factor=2, mode='bilinear', align_corners=False)
                bicubic_SR = torch.nn.functional.interpolate(LR, scale_factor=2, mode='bicubic', align_corners=False)

                scipy.io.savemat(os.path.join(SR_path, f'{idx}.mat'), {'SR': SR.squeeze(0).permute(1,2,0).numpy()})
                
                if save_classic:
                    scipy.io.savemat(os.path.join(HR_path, f'{idx}.mat'), {'HR': HR.squeeze(0).permute(1,2,0).numpy()})
                    scipy.io.savemat(os.path.join(LR_path, f'{idx}.mat'), {'LR': LR.squeeze(0).permute(1,2,0).numpy()})
                    scipy.io.savemat(os.path.join(nearest_path, f'{idx}.mat'), {'SR': nearest_SR.squeeze(0).permute(1,2,0).numpy()})
                    scipy.io.savemat(os.path.join(bilinear_path, f'{idx}.mat'), {'SR': bilinear_SR.squeeze(0).permute(1,2,0).numpy()})
                    scipy.io.savemat(os.path.join(bicubic_path, f'{idx}.mat'), {'SR': bicubic_SR.squeeze(0).permute(1,2,0).numpy()})

                nearest_PSNR.append(10 * math.log10(1 / F.mse_loss(nearest_SR, HR).item()))
                bilinear_PSNR.append(10 * math.log10(1 / F.mse_loss(bilinear_SR, HR).item()))
                bicubic_PSNR.append(10 * math.log10(1 / F.mse_loss(bicubic_SR, HR).item()))
                SR_PSNR.append(10 * math.log10(1 / F.mse_loss(SR, HR).item()))

                nearest_SAM.append(self.criterion_sam(nearest_SR, HR).item())
                bilinear_SAM.append(self.criterion_sam(bilinear_SR, HR).item())
                bicubic_SAM.append(self.criterion_sam(bicubic_SR, HR).item())
                SR_SAM.append(self.criterion_sam(SR, HR).item())

                max_pixel_value = torch.max(nearest_SR.max(), HR.max()).item()
                nearest_SSIM.append(ssim(nearest_SR, HR, data_range=max_pixel_value).item())
                bilinear_SSIM.append(ssim(bilinear_SR, HR, data_range=max_pixel_value).item())
                bicubic_SSIM.append(ssim(bicubic_SR, HR, data_range=max_pixel_value).item())
                SR_SSIM.append(ssim(SR, HR, data_range=max_pixel_value).item())

                nearest_MSE.append(F.mse_loss(nearest_SR, HR).item())
                bilinear_MSE.append(F.mse_loss(bilinear_SR, HR).item())
                bicubic_MSE.append(F.mse_loss(bicubic_SR, HR).item())
                SR_MSE.append(F.mse_loss(SR, HR).item())

        scipy.io.savemat(os.path.join(self.current_log_path, 'test_results.mat'), \
                            {'nearest_PSNR': nearest_PSNR, 'bilinear_PSNR': bilinear_PSNR, 'bicubic_PSNR': bicubic_PSNR, 'SR_PSNR': SR_PSNR, \
                                'nearest_SAM': nearest_SAM, 'bilinear_SAM': bilinear_SAM, 'bicubic_SAM': bicubic_SAM, 'SR_SAM': SR_SAM, \
                                'nearest_SSIM': nearest_SSIM, 'bilinear_SSIM': bilinear_SSIM, 'bicubic_SSIM': bicubic_SSIM, 'SR_SSIM': SR_SSIM, \
                                'nearest_MSE': nearest_MSE, 'bilinear_MSE': bilinear_MSE, 'bicubic_MSE': bicubic_MSE, 'SR_MSE': SR_MSE})


            
    def save_model(self, epoch=-1):
        torch.save(self.generator.state_dict(), os.path.join(self.current_log_path,f'SR_generator_{epoch}.pt'))
        torch.save(self.discriminator.state_dict(), os.path.join(self.current_log_path,f'SR_discriminator_{epoch}.pt'))

    def generate_images(self, epoch=-1):
        self.generator.eval()
        
        with torch.no_grad():
            SR = self.generator(self.LR_fixed.to(self.device))
            SR = SR.to('cpu')
            SR = SR[0].detach().numpy()
            LR = self.LR_fixed[0].to('cpu').numpy()
            HR = self.HR_fixed[0].to('cpu').numpy()
            # save the image as a .mat file
            scipy.io.savemat(os.path.join(self.current_log_path, f'generated_image_{epoch}.mat'), \
                             {'SR_cube': SR, 'LR_cube': LR, 'HR_cube': HR})


    
    def load_model(self, path='', change_folder = False):
        for root, _, files in os.walk(self.model_path):
            if len(files) == 1:
                model = torch.load(os.path.join(root, files[0]), map_location=torch.device('cpu'))
            else:
                raise ValueError("Multiple files in model directory")
        
        # copy model to current log path
        shutil.copy(os.path.join(root, files[0]), self.current_log_path)
        
        new_state_dict = OrderedDict()
        for k, v in model.module.state_dict().items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v

        self.generator.load_state_dict(new_state_dict)

        if change_folder: self.current_log_path = path


    def printl(self,string):
        print(string)
        with open(self.current_log_path + '/log.txt', 'a') as f:
            print(string, file=f)
    
def get_args(parser):
    parser.add_argument('--batch-size', 
                        type=int, 
                        default=12, 
                        metavar='N',
                        help='input batch size for training (default: 12)')
    parser.add_argument('--test-batch-size', 
                        type=int, 
                        default=1, 
                        metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--eval-batch-size',
                        type=int,
                        default=12,
                        metavar='N',
                        help='input batch size for evaluation (default: 12)')
    parser.add_argument('--epochs', 
                        type=int, 
                        default=100, 
                        metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', 
                        type=float, 
                        default=0.0001, 
                        metavar='LR',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--log-interval', 
                        type=int, 
                        default=10, 
                        metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--log-dir', 
                        type=str, 
                        default='logs', 
                        metavar='N',
                        help='directory to save logs')
    parser.add_argument('--patience',
                        type=int,
                        default=25,
                        metavar='N',
                        help='number of epochs to wait before early stopping')
    parser.add_argument('--path_model',
                        type=str,
                        default='../model/last',
                        help='Path to model')

def check_GPU_memory():
    current_device = torch.cuda.current_device()
    print("Current memory allocated:", round(torch.cuda.memory_allocated(current_device)/1024**3,1), "GB")
    print("Total memory:", round(torch.cuda.get_device_properties(current_device).total_memory/1024**3,1), "GB")

def get_model_memory_usage(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_params_bytes = total_params * 4  # each parameter is a float32, which is 4 bytes
    total_params_megabytes = total_params_bytes / (1024 ** 2)  # convert bytes to megabytes
    print(f'Model memory usage: {total_params_megabytes} MB')

if __name__ == '__main__':
    # Get arguments
    parser = argparse.ArgumentParser(description='PyTorch GAN Example')
    get_args(parser)
    args = parser.parse_args()
    # Define the GAN
    gan = GAN(args)
    # Train the GAN
    gan.train_GAN()
    input("Press Enter to continue...")
    # Load the best model
    gan.load_model()
    input("Press Enter to continue...")
    # Test the model on the evaluation dataset
    gan.test(gan.eval_loader, save_classic=True)
    # Test the model on the test dataset
    if input("Test the model on the test dataset? (y/n): ") == 'y':
        gan.test(gan.test_loader, save_classic=False)
