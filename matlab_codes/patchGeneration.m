clear all; close all; clc
%%
load("wavelengthsVNIR.mat")
wavelengthsVNIR = wavelengthsVNIR(285:540);
cd ("..\data\cubes\")
%%
folder = dir; folder = folder(3:end);

patch_size = [2^7,2^8];
patch_path = "..\patchs\";
mkdir(patch_path)
patch_path = "..\" + patch_path;

ID = 3611;
%%
patches_not_done = {};
%%
for i = 17:length(folder)
    fprintf("Paciente %s (%d / %d [%.2f%%])\n",folder(i).name,i,length(folder),i/length(folder)*100)
    cd(folder(i).name)
    cubes = dir("*.mat");
    [~,info,~] = mkdir(patch_path + "HR\"+ folder(i).name);
    mkdir(patch_path + "LR\"+ folder(i).name)

    for j = 1:length(cubes)
        fprintf("\tCaptura %s (%d / %d [%.2f%%])\n",cubes(j).name,j,length(cubes),j/length(cubes)*100)
        cube = load(cubes(j).name); cube = struct2cell(cube(1)); cube = cube{1};
        cube_size = size(cube);
        numPatches = floor(cube_size(1:2)./patch_size);
        
        if sum(cube,"all") == prod(cube_size)
            continue
        end

        for x = 0:numPatches(2)-1
            for y = 0:numPatches(1)-1
                HR_patch = cube(y*patch_size(1)+1:(y+1)*patch_size(1),x*patch_size(2)+1:(x+1)*patch_size(2),:);            
                
                samValue = sam(squeeze(mean(mean(HR_patch))),ones(1,size(HR_patch,3)));
                if samValue < 0.02
                    imshow(HR_patch(:,:,end/2))
                    if input("Do you want to use this patch? [Y/N]: ","s") ~= "Y"
                        patches_not_done{end+1} = HR_patch;
                        continue
                    end
                end
                LR_patch = degenerate(HR_patch,2);
                
                % las dimensiones deben ser lambda x spatial porque pytorch
                % coge la primera dimension como los canales 
                HR_patch = permute(HR_patch,[3,1,2]);
                LR_patch = permute(LR_patch,[3,1,2]);
                
                save(patch_path + "HR\"+ folder(i).name + "\patch_" + ID,"HR_patch","-v7.3")
                save(patch_path + "LR\"+ folder(i).name + "\patch_" + ID,"LR_patch","-v7.3")
                ID = ID + 1;
            end
        end
    end

    cd ..
end

cd("..\patchs\")
%%
data_split(".","..\train","..\test","..\eval")
