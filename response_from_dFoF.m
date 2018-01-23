function [response,orilist,raw_dFoF,baseline] = response_from_dFoF(dFoF,result,stim_frames,bl_frame_no,addon)
% given a matrix of dFoF traces for an entire experiment (row: ROI no.,
% col: frame), return a 4d matrix: ROI no. x time point after stim delivery
% x grating orientation no. x trial no. bl_frame_no is the number of frames
% immediately before stim delivery used in calculating the baseline.

ori = result.stimParams(1,:);
orilist = unique(ori);
% [~,ori_index] = ismember(ori,orilist);
Nori = numel(orilist);
len = numel(stim_frames)/2;
repno = len/Nori;
sigon = sigon_mat_addon(stim_frames,addon)';
sigoff = vec_colon(sigon(1,:)'-bl_frame_no,bl_frame_no)';
on_frame_no = size(sigon,1);
stim_no = size(sigon,2);
ROIno = size(dFoF,1);
raw_dFoF = zeros(ROIno,on_frame_no,stim_no);
baseline = zeros(ROIno,bl_frame_no,stim_no);
raw_dFoF(:) = dFoF(:,sigon);
baseline(:) = dFoF(:,sigoff);
bl_corr = raw_dFoF - repmat(mean(baseline,2),1,on_frame_no,1);
response = zeros(ROIno,on_frame_no,repno,Nori);
response(:) = bl_corr(:,:,sort_by(ori));
response = permute(response,[1 2 4 3]);
end
