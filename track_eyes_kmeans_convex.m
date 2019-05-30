function [ctr,area,props,ctr2,area2,props2,hulls] = track_eyes_kmeans_convex(foldname)

% global old_ctr iframe

d = dir([foldname '/*.tiff']);
frameno = numel(d);
if frameno>0
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
%     hulls = sparse(frameno,prod(size(msk)));
%     hulls = cell(frameno,1);
    max_hull_size = 300;
    hulls = zeros(frameno,max_hull_size);
    parfor iframe=1:frameno % find(idxframe==11203) 
        iframe
        try
            data = double(imread([foldname '/' d(iframe).name]));
            data = impyramid(data,'reduce');
            [ctr(iframe,:),area(iframe),props(iframe,:)] = extract_ctr_area(data,msk,ideal_props);
            old_ctr = ctr(iframe,:);
            [ctr2(iframe,:),area2(iframe),props2(iframe,:),hulls(iframe,:)] = clean_ctr_area(data,msk,ctr(iframe,:),area(iframe),max_hull_size);
%             hulls(iframe,pts) = 1;
%             hulls{iframe} = pts;
%             hulls(iframe,1:numel(pts)) = pts;
        catch
            [ctr(iframe,:),area(iframe),props(iframe,:),...
                ctr2(iframe,:),area2(iframe),props2(iframe,:)] = deal(nan);
        end
    end
    ctr(idxframe,:) = ctr;
    area(idxframe) = area;
    props(idxframe,:) = props;
    ctr2(idxframe,:) = ctr2;
    area2(idxframe) = area2;
    props2(idxframe,:) = props2;
    hulls(idxframe,:) = hulls;
    hulls = hulls(:,1:find(sum(hulls>0)==0,1));
else
    [ctr,area,props,ctr2,area2,props2] = deal([]);
end
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
min_size = 30;

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
% pause
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

function [ctr_new,area_new,props,hull] = clean_ctr_area(data,msk,ctr,area,max_hull_size)
edge_image = edge(data,'canny');
edge_image(~msk) = 0;
data(~msk) = 0;

if area < sum(msk(:))
    sz = floor(sqrt(area)/2);
else
    sz = 5;
end
blur_by = 3;
area_new = inf;
while isempty(area_new) || area_new > sum(msk(:))
    square = strel('square',blur_by);
    edge_image = imerode(imdilate(edge_image,square),square);
    [fillx,filly] = meshgrid(-sz:sz,-sz:sz);
    in_radius = fillx.^2 + filly.^2 <= sz^2;
    fill_points = repmat(round(ctr(end:-1:1)),sum(in_radius(:)),1);
    fill_points = fill_points + [fillx(in_radius) filly(in_radius)];
    filled_image = imfill(edge_image,fill_points); %'holes');
    conncomp_labeled = bwlabel(filled_image,4);
    in_pupil = conncomp_labeled == conncomp_labeled(round(ctr(2)),round(ctr(1)));
    props = extract_properties(in_pupil);
    ctr_new = props(end-1:end);
    area_new = props(end-2);
    if area_new > sum(msk(:))*0.5
        [ctr_new,radius_new] = imfindcircles(data,[floor(sz/2.5) floor(sz)]);
        [ctr_newb,radius_newb] = imfindcircles(data,[floor(sz)+1 floor(sz*2.5)]);
        ctr_new = [ctr_new; ctr_newb];
        radius_new = [radius_new; radius_newb];
        [~,mxind] = max(radius_new);
        ctr_new = ctr_new(mxind,:);
        radius_new = radius_new(mxind);
        area_new = pi*radius_new^2;
        props = [1 0 area_new ctr_new];
        [xx,yy] = meshgrid(1:size(data,2),1:size(data,1));
        if ~isempty(ctr_new)
            in_pupil = (xx-ctr_new(1)).^2 + (yy-ctr_new(2)).^2 < radius_new^2;
        end
    end
    sz = sz-1;
%     blur_by = blur_by+2;
end

pupil_inds = find(in_pupil);
[thisy,thisx] = ind2sub(size(data),pupil_inds);
thisp = convhull(thisy,thisx);
hull = zeros(1,max_hull_size);
hull(1:numel(thisp)) = pupil_inds(thisp);
% imagesc(data)
% hold on;
% h = fill(thisx(thisp),thisy(thisp),'m'); %,'alpha',0.5)
% set(h,'facealpha',0.25)
% scatter(ctr(1),ctr(2),'m+')
% scatter(ctr_new(1),ctr_new(2),'g+')
% hold off;
% pause(1e-3)
% disp('got here')
end

% %%
% rgb = zeros(size(data));
% rgb(:,:,1) = data/max(data(:));
% rgb(:,:,2) = is_bright;
% rgb(:,:,3) = zeros(size(data));
% imshow(rgb)