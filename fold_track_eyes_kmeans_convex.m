function fold_track_eyes_kmeans_convex(foldname)
d = dir(foldname);
forbidden = {'.','..'};
for i=1:numel(d)
    msk = [];
    if ~ismember(d(i).name,forbidden)
        thisfold = [foldname '/' d(i).name];
        d2 = dir([thisfold '/*.tiff']);
        msk = draw_msk([thisfold '/' d2(1).name]);
        save([thisfold '/msk.mat'],'msk')
    end
end
for i=1:numel(d)
    if ~ismember(d(i).name,forbidden)
        i
        thisfold = [foldname '/' d(i).name];
        tic
        [ctr,area,props,ctr2,area2,props2] = track_eyes_kmeans_convex(thisfold)
        toc
        save([foldname '/eye_tracking_' d(i).name '.mat'],'ctr','area','props','ctr2','area2','props2')
    end
end

function msk = draw_msk(filename,msk)
img = imread(filename);
if nargin < 2 || isempty(msk)
    msk = zeros(size(img));
    msk_available = false;
else
    msk_available = true;
end
figure;
rgb = zeros([size(img) 3]);
rgb(:,:,1) = img;
rgb(:,:,2) = msk;
imshow(rgb)
if msk_available
    its_good = strcmp(input('this good? (y/n)\n'),'y');
end
if ~its_good
    h = impoly;
    msk = h.createMask;
end
