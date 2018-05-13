function fig = gen_roi_targeting_image(filename,roi_indices,img,texton,ax)

figure

if isstr(filename)
    load([filename '.rois'],'-mat','ROIdata');
    msk = make_msk(ROIdata.rois);
else
    msk = filename;
end
if nargin < 2 || isempty(roi_indices)
    roi_indices = 1:size(msk,3);
end
if nargin < 3 || isempty(img)
    load([filename '.align'],'-mat','m')
    img = m;
end
if nargin < 4 || isempty(texton)
    texton = true;
end
if nargin < 5 || isempty(ax)
    ax = gca;
end

roino = numel(roi_indices);
mskbd = cell(roino,1);
for i=1:roino
    b = bwboundaries(msk(:,:,roi_indices(i)));
    mskbd(i) = b(1);
end

imagesc(ax,img)
hold on
for i=1:roino
    plot(mskbd{i}(:,2),mskbd{i}(:,1))
    if texton
        text(mean(mskbd{i}(:,2)),mean(mskbd{i}(:,1)),num2str(roi_indices(i)),'Color','w')
    end
end
hold off
fig = gcf;