function im = load_suite2p_mean_image(foldname)
%%
old_fold = pwd;
cd(foldname)
%%
load('Fall.mat','ops')
im = ops.meanImg;
%%
cd(old_fold)