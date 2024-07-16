function patch_degenerated = degenerate(patch, degeneration_factor)
% cd(path)
% folder = dir; folder = folder(3:end);
% 
% for i = 1:length(folder)
%    load(path + folder(i).name,'patch');
% 
    patch_size = size(patch);
    degenerated_size = floor(patch_size / degeneration_factor);
    patch_degenerated=zeros(degenerated_size);
    
    for j = 1:patch_size(3)
        aux = conv2(squeeze(patch(:,:,j)),ones(degeneration_factor),"valid");
        patch_degenerated(:,:,j) = aux(1:degeneration_factor:end,1:degeneration_factor:end)/degeneration_factor^2;
    end

%     save("../"+result_path+folder(i).name,'patch_degenerated')

end
