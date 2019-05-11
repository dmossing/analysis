function opto_correct_tiffs
nplanes = 4;
datafold = '/media/mossing/backup_1/data/2P/';
stimfold = '/home/mossing/modulation/visual_stim/';
foldname = '190407/M10368';
filename = 'M10368_070_003';
% foldname = '190408/M0002';
% filename = 'M10368_080_003';

offset = 1; % the first these many triggers are fake
toffset = 1; % static, related to the timing properties of the triggers and LED
loffset1 = 7;
loffset2 = 7;

roifile = cell(nplanes,1);
for i=1:nplanes
    roifile{i} = load(sprintf('%s/%s/ot/%s_ot_00%d.rois',datafold,foldname,filename,i-1),'-mat');
end

load(sprintf('%s/%s/%s.mat',stimfold,foldname,filename),'result')
load(sprintf('%s/%s/ot/%s.mat',datafold,foldname,filename),'info')

lights_on = result.gratingInfo.lightsOn(end,:);

    for j=1:numel(lights_on)
        if lights_on(j)
            frames = 1+floor((-toffset+info.frame(offset+(j-1)*4+1))/nplanes):1+floor((-toffset+info.frame(offset+j*4))/nplanes);
            lines = [info.line(offset+(j-1)*4+1) info.line(offset+j*4)];
            lines(1) = lines(1) + mod(-toffset+info.frame(offset+(j-1)*4+1),nplanes)*512 + loffset1;
            lines(end) = lines(end) + mod(-toffset+info.frame(offset+j*4),nplanes)*512 + loffset2;
            affected(:,frames(2:end-1)) = 1;
            affected(:,frames(1)) = ((i-1)*512+roiline)>lines(1);
            affected(:,frames(end)) = ((i-1)*512+roiline)<lines(end);
        end
    end
    
    mn = mean(roifile{i}.Neuropil)';
    [idx,C] = kmeans(mn,2);
    [~,bigger] = max(C);
    artifact = abs(diff(C))*affected; %(idx==bigger)
    if ~isfield(roifile{i},'opto_stim_corrected')
        roifile{i}.Data = roifile{i}.Data-artifact; %repmat(artifact',size(roifile{i}.Data,1),1);
        roifile{i}.Neuropil = roifile{i}.Neuropil-artifact; %repmat(artifact',size(roifile{i}.Neuropil,1),1);
        roifile{i}.opto_stim_corrected = 1;
    end
end