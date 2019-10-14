function fold_track_eyes_kmeans_convex(foldname,dry_run)
if nargin < 2
    dry_run = false;
end
d = dir(foldname);
forbidden = {'.','..'};
msk = [];
for i=1:numel(d)
    if ~ismember(d(i).name,forbidden) && ~contains(d(i).name,'eye_tracking')
        thisfold = [foldname '/' d(i).name];
        if ~exist([thisfold '/msk.mat'])
            d2 = dir([thisfold '/*.tiff']);
            if numel(d2)>0
                msk = draw_msk([thisfold '/' d2(1).name],msk);
                save([thisfold '/msk.mat'],'msk')
            end
        end
    end
end
if ~dry_run
    for i=1:numel(d)
        if ~ismember(d(i).name,forbidden) && ~contains(d(i).name,'eye_tracking')
            if ~exist([foldname '/eye_tracking_' d(i).name '.mat'])
                i
                thisfold = [foldname '/' d(i).name];
                tic
                [ctr,area,props,ctr2,area2,props2,hulls] = track_eyes_kmeans_convex(thisfold);
                toc
                save([foldname '/eye_tracking_' d(i).name '.mat'],'ctr','area','props','ctr2','area2','props2','hulls')
            end
        end
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
figure(1);
rgb = zeros([size(img) 3]); % uint8(zeros([size(img) 3]));
rgb(:,:,1) = double(img)/255;
rgb(:,:,2) = msk*0.25; %uint8(msk*255);

if msk_available
    imshow(rgb)
    its_good = strcmp(input('this good? (y/n)\n','s'),'y');
else
    imshow(img)
    its_good = false;
end
if ~its_good
    imshow(img)
    h = impoly;
    msk = h.createMask;
end
