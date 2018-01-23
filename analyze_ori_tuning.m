function [response,orilist,ori_response,tavg_response,mean_tavg_response,...
    mean_tavg_response_hi,mean_tavg_response_lo,reliable] = ...
    analyze_ori_tuning(ROIfile,stimfile,maxframe,bl_time,rel_thresh,hi,lo,...
    nboot,framerate,addon)
if nargin < 3 || isempty(maxframe)
    maxframe = Inf;
end
if nargin < 4 || isempty(bl_time)
    bl_time = 2;
end
if nargin < 5 || isempty(rel_thresh)
    rel_thresh = 5;
end
if nargin < 6 || isempty(hi)
    hi = 97.5;
end
if nargin < 7 || isempty(lo)
    lo = 2.5;
end
if nargin < 8 || isempty(nboot)
    nboot = 10000;
end
if nargin < 9 || isempty(framerate)
    framerate = 15.5;
end
if nargin < 10 || isempty(addon)
    addon = 32;
end
bl_frame_no = floor(bl_time*framerate);
load([ROIfile '.rois'],'-mat')
load([ROIfile '.mat'],'info')
try
    info.frame = info.frame(info.frame<=maxframe & info.event_id==1);
catch
    info.frame = info.frame(info.frame<=maxframe);
end
load([stimfile '.mat'],'result')
try
    [NeuropilWeight, ROIdata] = determineNeuropilWeight(ROIdata);
    Data = Data-repmat(NeuropilWeight,1,size(Neuropil,2)).*Neuropil;
    Data = Data(:,1:end-1);
catch
    Data = Data(:,1:end-1);
end
dFoF = bsxfun(@rdivide, bsxfun(@minus, Data, ...
    prctile(Data, 30, 2)), prctile(Data, 30, 2));
ROIno = size(dFoF,1);

[response,orilist,evoked,baseline] = response_from_dFoF(dFoF,result,info.frame,bl_frame_no,addon);

Nori = numel(orilist);

ori_response = mean(response,4);
tavg_response = squeeze(mean(response,2));
mean_tavg_response = mean(tavg_response,3);
mean_tavg_response_bs = zeros(ROIno,Nori,nboot);
[mean_tavg_response_hi,mean_tavg_response_lo] = deal(zeros(ROIno,Nori));

for j=1:Nori
    % calculating bootstrapped error bars
    mean_tavg_response_bs(:,j,:) = bootstrp(nboot,@mean,squeeze(tavg_response(:,j,:))')';
    mean_tavg_response_hi(:,j) = prctile(mean_tavg_response_bs(:,j,:),hi,3);
    mean_tavg_response_lo(:,j) = prctile(mean_tavg_response_bs(:,j,:),lo,3);
end
% a stimulus driven response is called "reliable" if the 95 pct. confidence
% intervals do not include 0, and upper bound < rel_thresh * lower bound for
% some orientation (default rel_thresh = 5).
reliable = find(min(mean_tavg_response_hi./max(mean_tavg_response_lo,0),[],2) < rel_thresh);
figure;
plot_tuning_cvs(orilist,mean_tavg_response,mean_tavg_response_hi,...
    mean_tavg_response_lo,1:ROIno);
save([ROIfile '_tuning'],'response','orilist','ori_response','tavg_response','mean_tavg_response',...
    'mean_tavg_response_hi','mean_tavg_response_lo','reliable','mean_tavg_response_bs')
end
