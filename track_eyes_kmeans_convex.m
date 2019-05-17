function [ctr,area,props,ctr2,area2,props2] = track_eyes_kmeans_convex(foldname)

% global old_ctr iframe

d = dir([foldname '/*.tiff']);
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
ctr2 = zeros(frameno,2);
area2 = zeros(frameno,1);
if exist([foldname '/msk.mat'])
    load([foldname '/msk.mat'],'msk')
else
    msk = draw_msk([foldname '/' d(1).name]);
    save([foldname '/msk.mat'],'msk')
end
% if exist([foldname '/pupil_template.mat'])
%     load([foldname '/pupil_template.mat'],'template')
% else
%     template = circle_pupil([foldname '/' d(1).name]);
%     save([foldname '/pupil_template.mat'],'template')
% end
msk = impyramid(msk,'reduce')>0.5;
% pupil_template = impyramid(template,'reduce')>0.5;
% ideal_props = extract_properties(pupil_template);
ideal_props = zeros(1,5);
ideal_props(1) = 1; % ideally solid
ideal_props(2) = 0; % ideally circular
ideal_props(3) = 250; % ideally around 250 pixels
msk_props = extract_properties(msk);
ideal_props(4:5) = msk_props(4:5);
props = zeros(frameno,numel(ideal_props));
props2 = zeros(frameno,numel(ideal_props));
parfor iframe=1:frameno % iframe=idxframe(2068) %  %  % % 
    iframe
    data = double(imread([foldname '/' d(iframe).name]));
    data = impyramid(data,'reduce');
    [ctr(iframe,:),area(iframe),props(iframe,:)] = extract_ctr_area(data,msk,ideal_props);
    old_ctr = ctr(iframe,:);
    [ctr2(iframe,:),area2(iframe),props2(iframe,:)] = clean_ctr_area(data,msk,ctr(iframe,:),area(iframe));
end
ctr(idxframe,:) = ctr;
area(idxframe) = area;
props(idxframe,:) = props;
ctr2(idxframe,:) = ctr2;
area2(idxframe) = area2;
props2(idxframe,:) = props2;
end

function msk = draw_msk(filename)
img = imread(filename);
figure;
imshow(img)
h = impoly;
msk = h.createMask;
end

function template = circle_pupil(filename)
img = imread(filename);
figure;
imshow(img)
h = imellipse;
template = h.createMask;
end

%%
function [ctr,area,props] = extract_ctr_area(data,msk,pupil_props)

% global old_ctr iframe
min_size = 100;

is_bright = zeros(size(data))==1;
sortsize = [];
while sum(sortsize > min_size)<1
    data(is_bright) = nan;
    cvals = data(msk);
    rng(1)
    [cluster_labels,centroids] = kmeans(cvals,2);
    [~,brightest_idx] = max(centroids);
    is_bright(msk) = cluster_labels==brightest_idx;
    connected_components = bwconncomp(is_bright);
    compsize = cellfun(@numel,connected_components.PixelIdxList);
    [sortsize,sortind] = sort(compsize);
end
if sum(sortsize > min_size)==1
    [area,pupil_ind] = max(compsize);
    thismsk = false(size(data));
    thismsk(connected_components.PixelIdxList{pupil_ind}) = 1;
    props = extract_properties(thismsk);
    area = props(1,end-2);
    ctr = props(1,end-1:end);
else
    candidate_inds = sortind(sortsize > min_size);
    candidate_convexities = zeros(numel(candidate_inds),1);
    candidate_props = zeros(numel(candidate_inds),numel(pupil_props));
    for i=1:numel(candidate_inds)
        thisind = candidate_inds(i);
        thismsk = false(size(data));
        thismsk(connected_components.PixelIdxList{thisind}) = 1;
        candidate_props(i,:) = extract_properties(thismsk);
        %         [thisy,thisx] = ind2sub(size(data),connected_components.PixelIdxList{thisind});
        %         thissize = compsize(thisind);
        %         thisp = convhull(thisy,thisx);
        %         thishull = polyarea(thisx(thisp),thisy(thisp));
        %         candidate_convexities(i) = thissize/thishull;
        %         hold on;
        %         fill(thisx(thisp),thisy(thisp),'m')
        %         hold off;
    end
    %     [~,selected_ind] = max(candidate_convexities);
    selected_ind = nn_props(pupil_props,candidate_props);
    props = candidate_props(selected_ind,:);
    pupil_ind = candidate_inds(selected_ind);
    area = candidate_props(selected_ind,end-2);
    ctr = candidate_props(selected_ind,end-1:end);
end

% [y,x] = ind2sub(size(data),connected_components.PixelIdxList{pupil_ind});
% ctr = [mean(x) mean(y)];

% imagesc(data)
% hold on;
% scatter(ctr(1),ctr(2),'m+')
% hold off;
% if iframe > 1
%     assert(sum(abs(ctr-old_ctr))<20)
% end
% pause(1e-3)
end

function prop_vals = extract_properties(mask)
fieldnames = {'Solidity','Eccentricity','Area','Centroid'};
rp = regionprops(mask,fieldnames{:});
prop_vals = zeros(1,numel(fieldnames)+1);
for i=1:numel(fieldnames)-1 % non-centroid fields
    prop_vals(i) = getfield(rp,fieldnames{i});
end
prop_vals(end-1:end) = getfield(rp,fieldnames{end}); % Centroid
end

function closest_ind = nn_props(ground_truth,candidate_props)
% vote based on region props specified above, which of the ROIs is closest.
nprops = numel(ground_truth);
ncands = size(candidate_props,1);
ranked_property_nearness = zeros(ncands,nprops);
for i=1:nprops
    %     ground_truth_val = getfield(ground_truth,propnames{i});
    %     for j=1:ncands
    %         candidate_vals(j) = getfield(candidate_props(j),propnames{i});
    %     end
    [~,how_to_rank] = sort(abs(candidate_props(:,i)-ground_truth(i)),'ascend');
    [~,ranked_property_nearness(:,i)] = sort(how_to_rank);
end
[~,closest_ind] = min(mean(ranked_property_nearness,2));
end

function [ctr_new,area_new,props] = clean_ctr_area(data,msk,ctr,area)
edge_image = edge(data,'canny');
edge_image(~msk) = 0;
square = strel('square',3);
edge_image = imerode(imdilate(edge_image,square),square);
fill_points = repmat(round(ctr(end:-1:1)),25,1);
[fillx,filly] = meshgrid(-2:2,-2:2);
fill_points = fill_points + [fillx(:) filly(:)];
filled_image = imfill(edge_image,fill_points); %'holes');
conncomp_labeled = bwlabel(filled_image,4);
in_pupil = conncomp_labeled == conncomp_labeled(round(ctr(2)),round(ctr(1)));
props = extract_properties(in_pupil);
ctr_new = props(end-1:end);
area_new = props(end-2);

[thisy,thisx] = ind2sub(size(data),find(in_pupil));
thisp = convhull(thisy,thisx);
imagesc(data)
hold on;
% fill(thisx(thisp),thisy(thisp),'c') %,'alpha',0.5)
scatter(ctr(1),ctr(2),'m+')
hold off;
pause(1e-3)
end