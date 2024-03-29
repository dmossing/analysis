function function_run_1p_opto_correction(foldname,varargin)
p = inputParser;

p.addParameter('datafold','/media/mossing/backup_0/data/2P/');
p.addParameter('stimfold','/home/mossing/modulation/visual_stim/');
p.addParameter('sbxfold','/home/mossing/modulation/2P/');

p.parse(varargin{:});

datafold = p.Results.datafold;
stimfold = p.Results.stimfold;
sbxfold = p.Results.sbxfold;

%%

d = dir(sprintf('%s/%s/ot/M*.mat',datafold,foldname));
filenames = {d(:).name};
has_opto_stim = zeros(size(filenames));
for i=1:numel(filenames)
    load(sprintf('%s/%s/%s',stimfold,foldname,filenames{i}),'result')
    has_opto_stim(i) = (isfield(result,'lights_on') && numel(result.lights_on)>1);
end
has_opto_stim = has_opto_stim>0;
filenames = filenames(has_opto_stim);

nplanes = 4;

% offset = 1; % the first these many triggers are fake
% toffset = 1; % static, related to the timing properties of the triggers and LED
% loffset1 = 1;
% loffset2 = 1;
noffset = 200;

%%

for ff=1:numel(filenames)
    filename = filenames{ff}(1:end-4);
    
    roifile = cell(nplanes,1);
    for i=1:nplanes
        roifile{i} = load(sprintf('%s/%s/ot/%s_ot_00%d.rois',datafold,foldname,filename,i-1),'-mat');
    end
    
    load(sprintf('%s/%s/%s.mat',stimfold,foldname,filename),'result')
    load(sprintf('%s/%s/ot/%s.mat',datafold,foldname,filename),'info')
    
    lights_on = result.gratingInfo.lightsOn(end,:);
    
    while find(diff(info.frame)<0,1)
        seam = find(diff(info.frame)<0,1);
        info.frame(seam+1:end) = info.frame(seam+1:end)+65536;
    end
    
    sbxbase = sprintf('%s/%s/',sbxfold,foldname);
    filebase = filename;
    pad_nans = true;
    levels = unique(lights_on);
    levels = levels(levels>0);
    nlevels = numel(levels);
    %%
%     from_raw = true; %true; %false;
%     %
%     %%
%     artifact_levels_cell = cell(nplanes,nlevels);
%     gain_artifact_levels_cell = cell(nplanes,nlevels);
%     for ilevel=1:nlevels
%         [artifact_levels_cell(:,ilevel),gain_artifact_levels_cell(:,ilevel)] = compute_artifact_optimizing_offsets(info,roifile,lights_on==levels(ilevel),noffset,sbxbase,filebase,from_raw,pad_nans);
%     end
%     artifact_cell = artifact_levels_cell(:,1);
%     gain_artifact_cell = gain_artifact_levels_cell(:,1);
%     for iplane=1:nplanes
%         for ilevel=2:nlevels
%             artifact_cell{iplane} = artifact_cell{iplane} + artifact_levels_cell{iplane,ilevel};
%             gain_artifact_cell{iplane} = gain_artifact_cell{iplane}.*gain_artifact_levels_cell{iplane,ilevel};
%         end
%     end
% %%
%     for i=1:nplanes
%         if ~isfield(roifile{i},'opto_stim_corrected')
%             maxind = size(artifact_cell{i},2);
% %             roifile{i}.Data(:,1:maxind) = roifile{i}.Data(:,1:maxind)-artifact_cell{i}; %repmat(artifact',size(roifile{i}.Data,1),1);
% %             roifile{i}.Neuropil(:,1:maxind) = roifile{i}.Neuropil(:,1:maxind)-artifact_cell{i}; %repmat(artifact',size(roifile{i}.Neuropil,1),1);
%             roifile{i}.Data(:,1:maxind) = apply_artifact_correction(roifile{i}.Data(:,1:maxind),0*artifact_cell{i},gain_artifact_cell{i}); %repmat(artifact',size(roifile{i}.Data,1),1);
%             roifile{i}.Neuropil(:,1:maxind) = apply_artifact_correction(roifile{i}.Neuropil(:,1:maxind),0*artifact_cell{i},gain_artifact_cell{i});
%         end
%     end
    %%
    from_raw = false;
        %%
    artifact_levels_cell = cell(nplanes,nlevels);
    gain_artifact_levels_cell = cell(nplanes,nlevels);
    for ilevel=1:nlevels
        [artifact_levels_cell(:,ilevel),gain_artifact_levels_cell(:,ilevel)] = compute_artifact_optimizing_offsets(info,roifile,lights_on==levels(ilevel),noffset,sbxbase,filebase,from_raw,pad_nans);
    end
    artifact_cell = artifact_levels_cell(:,1);
    gain_artifact_cell = gain_artifact_levels_cell(:,1);
    for iplane=1:nplanes
        for ilevel=2:nlevels
            artifact_cell{iplane} = artifact_cell{iplane} + artifact_levels_cell{iplane,ilevel};
            gain_artifact_cell{iplane} = gain_artifact_cell{iplane}.*gain_artifact_levels_cell{iplane,ilevel};
        end
    end
    %%
    for i=1:nplanes
        if ~isfield(roifile{i},'opto_stim_corrected')
            maxind = size(artifact_cell{i},2);
%             roifile{i}.Data(:,1:maxind) = roifile{i}.Data(:,1:maxind)-artifact_cell{i}; %repmat(artifact',size(roifile{i}.Data,1),1);
%             roifile{i}.Neuropil(:,1:maxind) = roifile{i}.Neuropil(:,1:maxind)-artifact_cell{i}; %repmat(artifact',size(roifile{i}.Neuropil,1),1);
            roifile{i}.Data(:,1:maxind) = apply_artifact_correction(roifile{i}.Data(:,1:maxind),artifact_cell{i},gain_artifact_cell{i}); %repmat(artifact',size(roifile{i}.Data,1),1);
            roifile{i}.Neuropil(:,1:maxind) = apply_artifact_correction(roifile{i}.Neuropil(:,1:maxind),artifact_cell{i},gain_artifact_cell{i});
            roifile{i}.opto_stim_corrected = 1;
        end
    end
    %%
    for i=1:nplanes
        temp = roifile{i};
        save(sprintf('%s/%s/ot/%s_ot_00%d.rois',datafold,foldname,filename,i-1),'-mat','-v7.3','-struct','temp');
    end
    
end
%%
% lkat = 1;
% data = roifile{i}.Data;
% neuropil = roifile{i}.Neuropil;
% d = data(lkat,:);
% n = neuropil(lkat,:);
% plot(d-n)

%%
% for i=1:nplanes
%     temp = roifile{i};
%     temp.redratio = [roifile{i}.ROIdata.rois(:).redratio];
%     save(sprintf('%s/%s/%s_ot_00%d.rois',datafold,foldname,filename,i-1),'-mat','-v7.3','-struct','temp');
% end
%%
run_preprocessing_fold(sprintf('%s/%s/ot/',datafold,foldname),'/home/mossing/modulation/running/');

function artifact_corrected = apply_artifact_correction(arr,artifact,gain_artifact)
if nargin < 2 || isempty(artifact)
    artifact = zeros(size(arr));
end
if nargin < 3
    gain_artifact = ones(size(artifact));
end
artifact_corrected = gain_artifact.*(arr-artifact);

% function artifact_corrected = apply_gain_correction(arr,gain_artifact)
% if nargin < 2
%     gain_artifact = ones(size(artifact));
% end
% artifact_corrected = gain_artifact.*arr;

%         roiline = round(roifile{i}.ctr(1,:));
%         if isfield(info,'rect')
%             roiline = roiline + info.rect(1);
%         end
%         affected = false(size(roifile{i}.Data));
%         control = false(size(roifile{i}.Data));
%         
%         for j=1:numel(lights_on)
%             frames = 1+floor((-toffset+info.frame(offset+(j-1)*4+1))/nplanes):1+floor((-toffset+info.frame(offset+j*4))/nplanes);
%             lines = [info.line(offset+(j-1)*4+1) info.line(offset+j*4)];
%             lines(1) = lines(1) + mod(-toffset+info.frame(offset+(j-1)*4+1),nplanes)*512 + loffset1;
%             lines(end) = lines(end) + mod(-toffset+info.frame(offset+j*4),nplanes)*512 + loffset2;
%             if lights_on(j)
%                 affected(:,frames(2:end-1)) = 1;
%                 affected(:,frames(1)) = ((i-1)*512+roiline)>lines(1);
%                 affected(:,frames(end)) = ((i-1)*512+roiline)<lines(end);
%             else
%                 control(:,frames(2:end-1)) = 1;
%                 control(:,frames(1)) = ((i-1)*512+roiline)>lines(1);
%                 control(:,frames(end)) = ((i-1)*512+roiline)<lines(end);
%             end
%         end
%         af_off = [diff(affected,[],2)>0 zeros(size(affected,1),1)]>0;
%         af_off = af_off | [zeros(size(affected,1),1) diff(affected,[],2)<0]>0;
%         af_on = [zeros(size(affected,1),1) diff(affected,[],2)>0]>0;
%         af_on = af_on | [diff(affected,[],2)<0 zeros(size(affected,1),1)]>0;
%         
%         caf_off = [diff(control,[],2)>0 zeros(size(control,1),1)]>0;
%         caf_off = caf_off | [zeros(size(control,1),1) diff(control,[],2)<0]>0;
%         caf_on = [zeros(size(control,1),1) diff(control,[],2)>0]>0;
%         caf_on = caf_on | [diff(control,[],2)<0 zeros(size(control,1),1)]>0;
%         
%         artifact_size = mean(roifile{i}.Neuropil(af_on)-roifile{i}.Neuropil(af_off));
%         control_size = mean(roifile{i}.Neuropil(caf_on)-roifile{i}.Neuropil(caf_off));
        
        
%         artifact = (artifact_size-control_size)*affected;