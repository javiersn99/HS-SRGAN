cd("..\data\raw_cubes\")
folder = dir; folder = folder(3:end);
%%
for i = 1:3
    clc; fprintf("%.2f%%\n",i*100/length(folder))
    % Calibrate
    cube = hypercube(folder(i).name+"\raw.hdr");
    dark = hypercube(folder(i).name+"\darkReference.hdr");
    white = hypercube(folder(i).name+"\whiteReference.hdr");

    calibrated = (double(cube.DataCube)-mean(double(dark.DataCube)))./mean(double(white.DataCube)-double(dark.DataCube));
    
    % Smooth
    cube_size = size(calibrated);
    calibrated = reshape(calibrated,[prod(cube_size(1:2)),cube_size(3)]);
    aux = calibrated;
    for j = 1:prod(cube_size(1:2))
        aux(j,:)=smooth(calibrated(j,:));
    end
    calibrated = reshape(aux,cube_size);
    
    % Band Select (256 bands)
    calibrated = calibrated(:,:,285:540);

    mkdir("..\cubes\"+folder(i).name(1:end-4))
    save("..\cubes\"+folder(i).name(1:end-4)+"\"+folder(i).name+"_calibrated","calibrated",'-v7.3')
%     figure
%     imshow(generateRGB(calibrated,cube.Wavelength'))
end
