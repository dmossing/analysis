function trigaligned = visualize_opto_artifact(sbxbase,filebase,neuropil,info,lights_on)

% stimbase = '/home/mossing/modulation/visual_stim/190807/M0153/';
% foldbase = '/home/mossing/modulation/matfiles/190807/M0153/ot/';
% sbxbase = '/home/mossing/modulation/2P/190807/M0153/';
% filebase = 'M0153_090_004';
filename = sprintf('%s%s',sbxbase,filebase);
% for i=1:4
%     filenames{i} = sprintf('%s%s_ot_%03d.rois',foldbase,filebase,i-1);
% end

%%
offset = 1;
toffset = 1;
ts = 4*(find(lights_on)-1)+1+offset;
naround = 3;
nbefore = 100;
nafter = 200;
nlines = info.sz(1);
sz = size(neuropil);
signal = zeros(numel(ts),naround*nlines);
for i=1:numel(ts)
    im = sbxreadpacked(filename,info.frame(ts(i))-1,naround);
    signal(i,:) = reshape(mean(im(:,101:end,:),2),1,[]);
end

trigaligned = zeros(numel(ts),nbefore+nafter+1);
for i=1:numel(ts)
    trigaligned(i,:) = signal(i,nlines+info.line(ts(i))-nbefore:nlines+info.line(ts(i))+nafter);
end

%%
figure
hold on
p = plot(trigaligned','color',[0 0 1 1e-2]); %,'alpha',0.01)
plot(mean(trigaligned),'color','r')
hold off
% p.Color = [0 0 0 0.01]; %alpha(p,0.01)