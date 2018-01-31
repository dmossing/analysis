function opto_tuning = analyze_vis_opto_tuning(ROIfile,stimfile,options)
% maxframe,bl_time,rel_thresh,hi,lo,...
%     nboot,framerate,addon)
if nargin < 3 || isempty(options)
    options = [];
end
if ~isfield(options,'maxframe')
    options.maxframe = Inf;
end
if ~isfield(options,'bl_time')
    options.bl_time = 2;
end
if ~isfield(options,'rel_thresh')
    options.rel_thresh = 5;
end
if ~isfield(options,'hi')
    options.hi = 97.5;
end
if ~isfield(options,'lo')
    options.hi = 2.5;
end
if ~isfield(options,'nboot')
    options.hi = 10000;
end
if ~isfield(options,'framerate')
    options.framerate = 15.5;
end
if ~isfield(options,'addon')
    options.addon = 32;
end
bl_frame_no = floor(options.bl_time*options.framerate);
load([ROIfile '.rois'],'-mat')
load([ROIfile '.mat'],'info')
try
    info.frame = info.frame(info.frame<=options.maxframe & info.event_id==1);
catch
    info.frame = info.frame(info.frame<=options.maxframe);
end
load([stimfile '.mat'],'result')
% try
NeuropilWeight = determineNeuropilWeight(ROIdata); % , ROIdata]
Data = Data-repmat(NeuropilWeight,1,size(Neuropil,2)).*Neuropil;
Data = Data(:,1:end-1);
% catch
%     Data = Data(:,1:end-1);
% end
dFoF = bsxfun(@rdivide, bsxfun(@minus, Data, ...
    prctile(Data, 30, 2)), prctile(Data, 30, 2));
ROIno = size(dFoF,1);
opto_tuning.trialwise = trialize(dFoF,info.frame,round(options.addon/2),options.addon);
opto_tuning.sorted = opto_tuning.trialwise(:,sort_by(result.stimParams'),:);