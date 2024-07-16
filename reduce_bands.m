function reduce_bands(folder)
for i = 1:length(folder)
    patch = load(folder(i).name); 
    patch = struct2cell(patch(1)); 
    patch = patch{1};
    patch = patch(:,:,285:end-286);
    patch = permute(patch,[3 2 1]);
    save("..\..\patch_reduced\HR\"+folder(i).name,"patch")
end