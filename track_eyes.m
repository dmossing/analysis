function [ctr,area] = track_eyes(foldname,rad_range)
d = dir([foldname '/*.tiff']);
frameno = numel(d);
idxframe = zeros(frameno,1);
for i=1:numel(d)
    s = strsplit(d(1).name,'_');
    s = strsplit(s{end},'.tiff');
    s = s{1};
    idxframe(i) = str2num(s);
end
ctr = zeros(frameno,2);
area = zeros(frameno,1);
parfor i=1:frameno
    i
    data = imread([foldname '/' d(i).name]);
    [center,radii,metric] = imfindcircles(squeeze(data),rad_range,'Sensitivity',1);
    if(isempty(center))
        ctr(i,:) = [NaN NaN]; % could not find anything...
        area(i) = NaN;
    else
        [~,idx] = max(metric); % pick the circle with best score
        ctr(i,:) = center(idx,:);
        area(i) = 4*pi*radii(idx)^2;
    end
end
ctr(idxframe,:) = ctr;
area(idxframe) = area;

% %%
% img = zeros(256,320,numel(d),'uint8');
% for i=1:size(img,3)
%     img(:,:,i) = imread(d(i).name);
%     i
% end
% 
% %%
% data = img;
% rad_range = [30 50];
% 
% data = squeeze(data); % the raw images...
% xc = size(data,2)/2; % image center
% yc = size(data,1)/2;
% 
% warning off;
% 
% for(n=1:size(data,3))
%     n
%     [center,radii,metric] = imfindcircles(squeeze(data(:,:,n)),rad_range,'Sensitivity',1);
%     if(isempty(center))
%         eye(n).Centroid = [NaN NaN]; % could not find anything...
%         eye(n).Area = NaN;
%     else
%         [~,idx] = max(metric); % pick the circle with best score
%         eye(n).Centroid = center(idx,:);
%         eye(n).Area = 4*pi*radii(idx)^2;
%     end
% end
% 
% %%
% ctr = zeros(numel(eye),2);
% area = zeros(numel(eye),1);
% for i=1:size(ctr,1)
%     ctr(i,:) = eye(i).Centroid;
%     area(i) = eye(i).Area;
% end
% 
% %%
% for i=1:size(data,3)
%     imshow(data(:,:,i))
%     hold on
%     plot(eye(i).Centroid(1),eye(i).Centroid(2),'m+')
%     hold off
%     pause
% end