function trigaligned = visualize_opto_artifact_fold(foldname,varargin)
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
trigaligned = cell(size(filenames));
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
        
    %%
    [neuropil,~] = append_roifile_parts(roifile,'Neuropil',false);
    
    trigaligned{ff} = visualize_opto_artifact(sbxbase,filebase,neuropil,info,lights_on);
end