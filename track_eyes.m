% function [ctr,area] = track_eyes(foldname,rad_range)
% cutoff = 10;
% softmax_scale = 5e-3;
% d = dir([foldname '/*.tiff']);
% d = d(1:1000);
% frameno = numel(d);
% idxframe = zeros(frameno,1);
% for i=1:numel(d)
%     s = strsplit(d(1).name,'_');
%     s = strsplit(s{end},'.tiff');
%     s = s{1};
%     idxframe(i) = str2num(s);
% end
% ctr = zeros(frameno,2);
% area = zeros(frameno,1);
% for i=1:frameno
%     i
%     if i==76
%         disp('here')
%     end
%     data = imread([foldname '/' d(i).name]);
%     [center,radii,metric] = imfindcircles(squeeze(data),rad_range,'Sensitivity',1);
%     if(isempty(center))
%         ctr(i,:) = [NaN NaN]; % could not find anything...
%         area(i) = NaN;
%     else
%         wt = softmax(metric(1:cutoff),softmax_scale); % pick the circle with best score
%         ctr(i,:) = sum(repmat(wt,1,2).*center(1:cutoff,:));
%         area(i) = 4*pi*sum(wt.*radii(1:cutoff))^2;
%     end
% end
% ctr(idxframe,:) = ctr;
% area(idxframe) = area;
%
% function xstar = softmax(x,const)
% wt = exp(x/const);
% xstar = wt/sum(wt);

function [ctr,area] = track_eyes(foldname,rad_range)
d = dir([foldname '/*.tiff']);
% cutoff = 100;
frameno = numel(d);
idxframe = zeros(frameno,1);
for i=1:numel(d)
    s = strsplit(d(i).name,'_');
    s = strsplit(s{end},'.tiff');
    s = s{1};
    idxframe(i) = str2num(s);
end
ctr = zeros(frameno,2);
area = zeros(frameno,1);
if exist([foldname '/msk.mat'])
    load([foldname '/msk.mat'],'msk')
else
    msk = draw_msk([foldname '/' d(1).name]);
%     img = imread([foldname '/' d(1).name]);
%     figure;
%     imshow(img)
%     h = impoly;
%     msk = h.createMask;
end

% hg = hist(img(msk),0:255);
% figure;
% plot(hg)
% cutoff_brightness = input('where to cut?')
% parfor i=1:frameno
%     i
%     data = imread([foldname '/' d(i).name]);
%     data = imbinarize(data);
%     data(~msk) = 0;
% %     data(data<cutoff_brightness) = 0;
%     [center,radii,metric] = imfindcircles(squeeze(data),rad_range,'Sensitivity',1,'ObjectPolarity','bright');
%     if(isempty(center))
%         ctr(i,:) = [NaN NaN]; % could not find anything...
%         area(i) = NaN;
%     else
% %         metric = metric(1:cutoff);
% %         center = center(1:cutoff,:);
% %         radius = radius(1:cutoff);
%         gd = msk(sub2ind(size(msk),round(center(:,2)),round(center(:,1))));
%         idx = find(gd,1);
% %         [~,idx] = max(metric(gd)); % pick the circle with best score
%         ctr(i,:) = center(idx,:);
%         area(i) = 4*pi*radii(idx)^2;
%     end
% end
% ctr(idxframe,:) = ctr;
% area(idxframe) = area;

for i=1:frameno
    i
    data = imread([foldname '/' d(i).name]);
    data = imbinarize(data);
    data(~msk) = 0;
    [xx,yy] = meshgrid(1:size(data,2),1:size(data,1));
    distfun = @(x,y,x0,y0)((x-x0).^2+(y-y0).^2);
    circfun = @(x0,y0,rad)(distfun(xx,yy,x0,y0)<rad^2);
    L = @(x)(sum(sum(abs(data-circfun(x(1),x(2),x(3))))));
    Nwhite = sum(data(:));
    x0 = sum((xx(:).*data(:)))/Nwhite;
    y0 = sum((yy(:).*data(:)))/Nwhite;
    ctr(i,:) = [x0 y0];
    area(i) = Nwhite;
end
ctr(idxframe,:) = ctr;
area(idxframe) = area;

function msk = draw_msk(filename)
img = imread(filename);
figure;
imshow(img)
h = impoly;
msk = h.createMask;