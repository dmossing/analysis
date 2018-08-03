function fold_track_eyes(foldname,rad_range)
d = dir(foldname);
forbidden = {'.','..'};
for i=1:numel(d)
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
        [ctr,area] = track_eyes(thisfold,rad_range)
        toc
        save([foldname '/eye_tracking_' d(i).name '.mat'],'ctr','area')
    end
end

function msk = draw_msk(filename)
img = imread(filename);
figure;
imshow(img)
h = impoly;
msk = h.createMask;