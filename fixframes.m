%%
% fnbase = 'M7307_185_000';
fnbase = 'M7478_100_002';
fns = {[fnbase '_ot_000'],[fnbase '_ot_001'],[fnbase '_ot_002'],...
    [fnbase '_ot_003']};
depthno = numel(fns);
%% first pass: trigger is assumed to be synchronized across all planes, but
% may mess up causality
for i=4%1:depthno 
    load([fns{i} '.rois'],'-mat','ROIdata')
    load([fns{i} '.mat'],'info')
    if ~isfield(info,'frameCorrected') || ~info.frameCorrected
        info.frame = ceil(info.frame/depthno);
        info.frameCorrecte = true;
        save([fns{i} '.mat'],'info','-append')
    end
end
%%
fns = {'M7307_185_001_ot_000','M7307_185_001_ot_001','M7307_185_001_ot_002',...
    'M7307_185_001_ot_003'};
depthno = numel(fns);
%% first pass: trigger is assumed to be synchronized across all planes, but
% may mess up causality
for i=1:depthno 
    load([fns{i} '.rois'],'-mat','ROIdata')
    load([fns{i} '.mat'],'info')
    if ~isfield(info,'frameCorrected') || ~info.frameCorrected
        info.frame = ceil(info.frame/depthno);
        info.frameCorrected = true;
        save([fns{i} '.mat'],'info','-append')
    end
end
%%
% load('../M7307_185_000','info')
% frm = info.frame;
% for i=1:depthno 
%     load([fns{i} '.rois'],'-mat','ROIdata')
%     load([fns{i} '.mat'],'info')
%     info.frame = frm;
%     info.frameCorrected = false;
%     if ~isfield(info,'frameCorrected') || ~info.frameCorrected
%         info.frame = ceil(info.frame/depthno);
%         save([fns{i} '.mat'],'info','-append')
%         info.frameCorrected = true;
%     end
%     save([fns{i} '.mat'],'info','-append')
% end