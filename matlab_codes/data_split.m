function data_split(data_path,train_path,test_path,eval_path)
patients = dir(data_path+"\HR"); patients = patients(3:end);

num_patches = zeros(1,length(patients));
for i = 1:length(patients)
    num_patches(i) = length(dir(data_path+"\HR\"+patients(i).name)) - 2;
end

% cum_num_patches = cumsum(num_patches);

randIdx = randperm(length(patients));
trainIdx = []; evalIdx = []; testIdx = [];
for idx=randIdx
    if sum(num_patches(trainIdx)) + num_patches(idx) < floor(sum(num_patches)*.8)
        trainIdx(end+1) = idx;
    elseif sum(num_patches(evalIdx)) + num_patches(idx) < floor(sum(num_patches)*.1)
        evalIdx(end+1) = idx;
    else
        testIdx(end+1) = idx;
    end
end

train_patients = patients(sort(trainIdx)); 
eval_patients  = patients(sort(evalIdx)); 
test_patients  = patients(sort(testIdx));

% train_patients = patients(cum_num_patches<floor(sum(num_patches)*.8));
% eval_patients = patients(cum_num_patches>=floor(sum(num_patches)*.8) & cum_num_patches<floor(sum(num_patches)*.9));
% test_patients = patients(cum_num_patches>=floor(sum(num_patches)*.9));

ID = fopen("split.txt","w");
fprintf(ID,"Train Patients:\n");
for i = 1:length(train_patients)
    fprintf(ID,"\t"+train_patients(i).name+"\n");
end
fprintf(ID,"Eval Patients:\n");
for i = 1:length(eval_patients)
    fprintf(ID,"\t"+eval_patients(i).name+"\n");
end
fprintf(ID,"Test Patients:\n");
for i = 1:length(test_patients)
    fprintf(ID,"\t"+test_patients(i).name+"\n");
end

mkdir(train_path+"\HR");mkdir(train_path+"\LR");
for i = 1:length(train_patients)
    copyfile("HR\"+train_patients(i).name,train_path+"\HR");
    copyfile("LR\"+train_patients(i).name,train_path+"\LR");
end

mkdir(test_path+"\HR");mkdir(test_path+"\LR");
for i = 1:length(test_patients)
    copyfile("HR\"+test_patients(i).name,test_path+"\HR");
    copyfile("LR\"+test_patients(i).name,test_path+"\LR");
end

mkdir(eval_path+"\HR");mkdir(eval_path+"\LR");
for i = 1:length(eval_patients)
    copyfile("HR\"+eval_patients(i).name,eval_path+"\HR");
    copyfile("LR\"+eval_patients(i).name,eval_path+"\LR");
end

fclose(ID);
return























% num_patches = 0;
% for i = 1:length(patients)
%     num_patches = num_patches + length(dir(data_path+"\HR\"+patients(i).name)) - 2;
% end
% 
% num_patients = [floor(length(patients)*.8) floor(length(patients)*.1)]; % Train - Eval - Test
% num_patients(3) = length(patients)-sum(num_patients);
% 
% idx = 1;
% idx_train = {[],0};
% idx_test = {[],0};
% idx_eval = {[],0};
% while length(idx_train{1}) <= num_patients(1) || ...
%       length(idx_test{1}) <= num_patients(2) || ...
%       length(idx_eval{1}) <= num_patients(3)
%     
%     
% 
%     if length(idx_train{1}) <= num_patients(1)
%         
%     end
% 
% end
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% folder = dir(data_path + "\*.mat");
% mkdir(train_path);mkdir(test_path);mkdir(eval_path);
% 
% idx_tests = unique(randi(69,[1,70])); idx_tests = idx_tests(1:14);
% idx_train = 1:70;
% cd ..\data\patch_reduced\
% mkdir ..\test\HR; mkdir ..\test\LR;
% mkdir ..\train\HR; mkdir ..\train\LR; 
% 
% for i = idx_tests
%     idx_train(i) = 0;
%     
%     name = "patch_"+(i-1)+".mat";
%     patch = load("HR\"+name); patch = struct2cell(patch(1)); patch = patch{1}; save("..\test\HR\"+name,"patch");
%     patch = load("LR\"+name); patch = struct2cell(patch(1)); patch = patch{1}; save("..\test\LR\"+name,"patch");
% end
% 
% idx_train = unique(idx_train); idx_train = idx_train(2:end);
% for i = idx_train
%     name = "patch_"+(i-1)+".mat";
%     patch = load("HR\"+name); patch = struct2cell(patch(1)); patch = patch{1}; save("..\train\HR\"+name,"patch");
%     patch = load("LR\"+name); patch = struct2cell(patch(1)); patch = patch{1}; save("..\train\LR\"+name,"patch");
% end
% 
% for test = idx_tests
%     for train = idx_train
%         if test == train
%             error("!!")
%         end
%     end
% end
