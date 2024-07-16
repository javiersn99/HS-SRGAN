% close all; clear all; clc
disp(pwd)
load('wavelengthsVNIR.mat'); wavelengthsVNIR = wavelengthsVNIR(285:540);
folder = dir("*.mat");

SAMS = zeros(1,length(folder));
f = figure; hold on; axis tight; grid on; grid minor; xlabel("Wavelength [nm]") % Pure signature
% g = figure; hold on; axis tight; grid on; grid minor; xlabel("Wavelength [nm]") % Normalized Image
for i = 1:1:length(folder)
%     load(folder(i).name)
    load("generated_image_"+(i-1)+".mat")
    SR_cube = permute(SR_cube,[2 3 1]);
    HR_cube = permute(HR_cube,[2 3 1]);
    
    SR_sig = squeeze(mean(mean(SR_cube)));
    HR_sig = squeeze(mean(mean(HR_cube)));
    
    figure(f)
    plot(wavelengthsVNIR,SR_sig,'LineStyle','-.','LineWidth',1,'Marker','+')


%     figure(g)
%     plot(wavelengthsVNIR,SR_sig/max(SR_cube,[],"all"),'LineStyle','-.','LineWidth',1,'Marker','+')
    
%     cube_size = size(SR_cube);
%     aux = zeros(1,cube_size(1)*cube_size(2));
%     for x = 1:cube_size(1)
%         for y = 1:cube_size(2)
%             aux((x-1)*cube_size(2)+y) = sam(squeeze(SR_cube(x,y,:)),squeeze(HR_cube(x,y,:)));
%         end
%     end
%     SAMS(i) = mean(aux);

    SAMS(i) = sam(SR_sig,HR_sig);
end
plot(wavelengthsVNIR, squeeze(mean(mean(HR_cube))), 'Marker','+')

[~,i] = min(SAMS);
load("generated_image_"+(i-1)+".mat")
SR_sig = squeeze(mean(mean(permute(SR_cube,[2 3 1]))));
figure; hold on; axis tight; grid on; grid minor; xlabel("Wavelength [nm]")
% plot(wavelengthsVNIR,SR_sig/max(SR_sig,[],"all"),'LineStyle','-.','LineWidth',1,'Marker','+')
plot(wavelengthsVNIR,SR_sig,'LineStyle','-.','LineWidth',1,'Marker','+')
plot(wavelengthsVNIR,squeeze(mean(mean(permute(HR_cube,[2 3 1])))),'Marker','+')

figure; hold on; axis tight; grid on; grid minor; xlabel("Epoch"); ylabel("Sam")
plot(SAMS,'Marker','+')
% aux=axis; axis([aux(1:2) 0 1])

SR_cube = permute(SR_cube,[2 3 1]);
% implay(SR_cube)
%%
HR_cube = permute(HR_cube,[2 3 1]);
figure
% imshow(squeeze(HR_cube(:,:,1)))