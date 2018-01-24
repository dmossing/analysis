function gen2Pframes(fname)
fname = strsplit(fname,'.');
fname = fname{1};
Images = load2P([fname '.sbx']);
sbxDistribute(fname);
load([fname '.align'],'-mat','MCdata')
Images = applyMotionCorrection(Images, MCdata);
folder = strsplit(fname,{'/','\'},'CollapseDelimiters',true);
folder = [strjoin(folder(1:end-1),'/') '/'];
load([fname '.mat'],'info')
dlmwrite([folder 'stims.txt'],info.frame-1,'\n');
framefolder = [folder '2Pframes/'];
mkdir(framefolder);
L = size(Images,5);
k = length(num2str(L-1));
for t=1:L
    t
    imwrite(squeeze(Images(:,:,1,1,t)),[framefolder ddigit(t-1,k) '.tif']);
end