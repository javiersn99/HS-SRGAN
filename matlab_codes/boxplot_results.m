%% EVALUATION
% load results no sam
cd Jul14_11-25-27_GAN_HS-SRGAN_1(EVAL)\
load test_results.mat
SR_PSNR_1 = SR_PSNR; SR_SAM_1 = SR_SAM; SR_SSIM_1 = SR_SSIM; SR_MSE_1 = SR_MSE;
cd ..\

% load results with sam
cd Jul14_12-48-50_GAN_HS-SRGAN_0.5(EVAL)\
load test_results.mat
SR_PSNR_05 = SR_PSNR; SR_SAM_05 = SR_SAM; SR_SSIM_05 = SR_SSIM; SR_MSE_05 = SR_MSE;
cd ..

%%
% boxplots
figure
boxplot([SR_PSNR_1',SR_PSNR_05'], 'Labels',{'HS-SRGAN_1','HS-SRGAN_0.5'}, 'Notch','on');setFigureProperties()

% boxplot([nearest_PSNR',bilinear_PSNR',bicubic_PSNR',SR_PSNR_1',SR_PSNR_05'], 'Labels',{'NN','Bilinear','Bicubic','HS-SRGAN_1','HS-SRGAN_0.5'})
mean([nearest_PSNR',bilinear_PSNR',bicubic_PSNR',SR_PSNR_1',SR_PSNR_05'])
std([nearest_PSNR',bilinear_PSNR',bicubic_PSNR',SR_PSNR_1',SR_PSNR_05'])

ylabel("PSNR")
set(gca,'LooseInset',get(gca,'TightInset'));
saveFigures("C:\Users\jsnunez\OneDrive - Universidad de Las Palmas de Gran Canaria (1)\Master\TFM\Memoria\Imagenes\","PSNR(EVAL)")

figure
boxplot([SR_SAM_1',SR_SAM_05'], 'Labels',{'HS-SRGAN_1','HS-SRGAN_0.5'}, 'Notch','on');setFigureProperties()

% boxplot([nearest_SAM',bilinear_SAM',bicubic_SAM',SR_SAM_1',SR_SAM_05'], 'Labels',{'NN','Bilinear','Bicubic','HS-SRGAN_1','HS-SRGAN_0.5'})
mean([nearest_SAM',bilinear_SAM',bicubic_SAM',SR_SAM_1',SR_SAM_05'])
std([nearest_SAM',bilinear_SAM',bicubic_SAM',SR_SAM_1',SR_SAM_05'])

ylabel("SAM")
set(gca,'LooseInset',get(gca,'TightInset'));
saveFigures("C:\Users\jsnunez\OneDrive - Universidad de Las Palmas de Gran Canaria (1)\Master\TFM\Memoria\Imagenes\","SAM(EVAL)")

figure
boxplot([SR_SSIM_1',SR_SSIM_05'], 'Labels',{'HS-SRGAN_1','HS-SRGAN_0.5'}, 'Notch','on');setFigureProperties()

% boxplot([nearest_SSIM',bilinear_SSIM',bicubic_SSIM',SR_SSIM_1',SR_SSIM_05'], 'Labels',{'NN','Bilinear','Bicubic','HS-SRGAN_1','HS-SRGAN_0.5'})
mean([nearest_SSIM',bilinear_SSIM',bicubic_SSIM',SR_SSIM_1',SR_SSIM_05'])
std([nearest_SSIM',bilinear_SSIM',bicubic_SSIM',SR_SSIM_1',SR_SSIM_05'])

ylabel("SSIM")
set(gca,'LooseInset',get(gca,'TightInset'));
saveFigures("C:\Users\jsnunez\OneDrive - Universidad de Las Palmas de Gran Canaria (1)\Master\TFM\Memoria\Imagenes\","SSIM(EVAL)")

figure
boxplot([SR_MSE_1',SR_MSE_05'], 'Labels',{'HS-SRGAN_1','HS-SRGAN_0.5'}, 'Notch','on');setFigureProperties()

% boxplot([nearest_MSE',bilinear_MSE',bicubic_MSE',SR_MSE_1',SR_MSE_05'], 'Labels',{'NN','Bilinear','Bicubic','HS-SRGAN_1','HS-SRGAN_0.5'})
mean([nearest_MSE',bilinear_MSE',bicubic_MSE',SR_MSE_1',SR_MSE_05'])
std([nearest_MSE',bilinear_MSE',bicubic_MSE',SR_MSE_1',SR_MSE_05'])

ylabel("MSE")
set(gca,'LooseInset',get(gca,'TightInset'));
saveFigures("C:\Users\jsnunez\OneDrive - Universidad de Las Palmas de Gran Canaria (1)\Master\TFM\Memoria\Imagenes\","MSE(EVAL)")


%%
% load results no sam
cd Jul14_18-24-34_GAN_HS-SRGAN_1(TEST)\
load test_results.mat
SR_PSNR_1 = SR_PSNR; SR_SAM_1 = SR_SAM; SR_SSIM_1 = SR_SSIM; SR_MSE_1 = SR_MSE;
cd ..\

% load results with sam
cd Jul14_17-12-58_GAN_HS-SRGAN_05(TEST)\
load test_results.mat
SR_PSNR_05 = SR_PSNR; SR_SAM_05 = SR_SAM; SR_SSIM_05 = SR_SSIM; SR_MSE_05 = SR_MSE;
cd ..

% boxplots
figure
% boxplot([nearest_PSNR',bilinear_PSNR',bicubic_PSNR',SR_PSNR_05'], 'Labels',{'NN','Bilinear','Bicubic','HS-SRGAN_0.5'})
% mean([nearest_PSNR',bilinear_PSNR',bicubic_PSNR',SR_PSNR_05'])
boxplot([nearest_PSNR',bilinear_PSNR',bicubic_PSNR',SR_PSNR_1',SR_PSNR_05'], 'Labels',{'NN','Bilinear','Bicubic','HS-SRGAN_1','HS-SRGAN_0.5'}, 'Notch','on');setFigureProperties()
mean([nearest_PSNR',bilinear_PSNR',bicubic_PSNR',SR_PSNR_1',SR_PSNR_05'])
std([nearest_PSNR',bilinear_PSNR',bicubic_PSNR',SR_PSNR_1',SR_PSNR_05'])

ylabel("PSNR")
set(gca,'LooseInset',get(gca,'TightInset'));
saveFigures("C:\Users\jsnunez\OneDrive - Universidad de Las Palmas de Gran Canaria (1)\Master\TFM\Memoria\Imagenes\","PSNR(TEST)")

figure
% boxplot([nearest_SAM',bilinear_SAM',bicubic_SAM',SR_SAM_05'], 'Labels',{'NN','Bilinear','Bicubic','HS-SRGAN_0.5'})
% mean([nearest_SAM',bilinear_SAM',bicubic_SAM',SR_SAM_05'])
boxplot([nearest_SAM',bilinear_SAM',bicubic_SAM',SR_SAM_1',SR_SAM_05'], 'Labels',{'NN','Bilinear','Bicubic','HS-SRGAN_1','HS-SRGAN_0.5'}, 'Notch','on');setFigureProperties()
mean([nearest_SAM',bilinear_SAM',bicubic_SAM',SR_SAM_1',SR_SAM_05'])
std([nearest_SAM',bilinear_SAM',bicubic_SAM',SR_SAM_1',SR_SAM_05'])

ylabel("SAM")
set(gca,'LooseInset',get(gca,'TightInset'));
saveFigures("C:\Users\jsnunez\OneDrive - Universidad de Las Palmas de Gran Canaria (1)\Master\TFM\Memoria\Imagenes\","SAM(TEST)")

figure
% boxplot([nearest_SSIM',bilinear_SSIM',bicubic_SSIM',SR_SSIM_05'], 'Labels',{'NN','Bilinear','Bicubic','HS-SRGAN_0.5'})
% mean([nearest_SSIM',bilinear_SSIM',bicubic_SSIM',SR_SSIM_05'])
boxplot([nearest_SSIM',bilinear_SSIM',bicubic_SSIM',SR_SSIM_1',SR_SSIM_05'], 'Labels',{'NN','Bilinear','Bicubic','HS-SRGAN_1','HS-SRGAN_0.5'}, 'Notch','on');setFigureProperties()
mean([nearest_SSIM',bilinear_SSIM',bicubic_SSIM',SR_SSIM_1',SR_SSIM_05'])
std([nearest_SSIM',bilinear_SSIM',bicubic_SSIM',SR_SSIM_1',SR_SSIM_05'])

ylabel("SSIM")
set(gca,'LooseInset',get(gca,'TightInset'));
saveFigures("C:\Users\jsnunez\OneDrive - Universidad de Las Palmas de Gran Canaria (1)\Master\TFM\Memoria\Imagenes\","SSIM(TEST)")

figure
% boxplot([nearest_MSE',bilinear_MSE',bicubic_MSE',SR_MSE_05'], 'Labels',{'NN','Bilinear','Bicubic','HS-SRGAN_0.5'})
% mean([nearest_MSE',bilinear_MSE',bicubic_MSE',SR_MSE_05'])
boxplot([nearest_MSE',bilinear_MSE',bicubic_MSE',SR_MSE_1',SR_MSE_05'], 'Labels',{'NN','Bilinear','Bicubic','HS-SRGAN_1','HS-SRGAN_0.5'}, 'Notch','on');setFigureProperties()
mean([nearest_MSE',bilinear_MSE',bicubic_MSE',SR_MSE_1',SR_MSE_05'])
std([nearest_MSE',bilinear_MSE',bicubic_MSE',SR_MSE_1',SR_MSE_05'])

ylabel("MSE")
set(gca,'LooseInset',get(gca,'TightInset'));
saveFigures("C:\Users\jsnunez\OneDrive - Universidad de Las Palmas de Gran Canaria (1)\Master\TFM\Memoria\Imagenes\","MSE(TEST)")
%%
cd Jul11_10-24-16_GAN\
load("eval_losses.mat")
eval_losses_1 = eval_losses;
cd ..\

cd Jul04_11-48-29_GAN\
load("eval_losses.mat")
eval_losses_05 = eval_losses;
cd ..\

figure
semilogy(eval_losses_1,'Marker','+','LineWidth',2,'MarkerSize',10,'LineStyle','-')
hold on
semilogy(eval_losses_05,'Marker','+','LineWidth',2,'MarkerSize',10,'LineStyle','-')
legend("HS-SRGAN_1","HS-SRGAN_{0.5}"); legend("Location","southeast")
xlabel("Epoch")
ylabel("Loss")
grid on
grid minor
fontname(gcf,"Times New Roman")
axis tight
fontsize(gcf,15,"points")
saveFigures(".\","Loss_Evolution")

function setFigureProperties()
grid on
grid minor
fontname(gcf,"Times New Roman")
axis tight
fontsize(gcf,15,"points")
end
