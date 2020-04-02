function fold_fold_save_msk(basefold,rg)
d = dir(basefold);
forbidden = {'.','..','Duplicate','.DS_Store'};
for i=1:numel(d)
    foldname = d(i).name;
    if ~ismember(foldname,forbidden)
        in_bounds = (str2num(foldname) >= rg(1)) && (str2num(foldname) <= rg(2));
        if in_bounds
            fold_save_msk([basefold '/' foldname])
        end
    end
end

function fold_save_msk(foldname)
d = dir(foldname);
forbidden = {'.','..'};
for i=1:numel(d)
    if ~ismember(d(i).name,forbidden)
        thisfold = [foldname '/' d(i).name];
        save_msk(thisfold);
    end
end

function save_msk(foldname)
d = dir(foldname);
forbidden = {'.','..'};
msk = [];
for i=1:numel(d)
    if ~ismember(d(i).name,forbidden)
        thisfold = [foldname '/' d(i).name];
        if ~exist([thisfold '/msk.mat'])
            thisfold
            d2 = dir([thisfold '/*.tiff']);
            if numel(d2)>0
                nframes = 1; %50;
                foldnames = cell(nframes,1);
                for i=1:nframes
                    foldnames{i} = [thisfold '/' d2(i).name];
                end
                msk = draw_msk(foldnames,msk);
                save([thisfold '/msk.mat'],'msk')
            end
        end
    end
end

function msk = draw_msk(filename,msk)
if ~iscell(filename)
    img = imread(filename);
else
    im0 = imread(filename{1});
    ims = uint8(zeros([size(im0) numel(filename)]));
    for i=1:numel(filename)
        ims(:,:,i) = imread(filename{i});
    end
    img = mean(double(ims),3);
end
if nargin < 2 || isempty(msk)
    msk = zeros(size(img));
    msk_available = false;
else
    msk_available = true;
end
figure(1);
rgb = zeros([size(img) 3]); % uint8(zeros([size(img) 3]));
rgb(:,:,1) = msk*0.5;
rgb(:,:,2) = 0.75*double(img)/255 + msk*0.25; %uint8(msk*255);
rgb(:,:,3) = double(img)/255;

if msk_available
    imshow(rgb)
    its_good = strcmp(input('this good? (y/n)\n','s'),'y');
else
    imshow(double(img)/255)
    its_good = false;
end
if ~its_good
    imshow(double(img)/255)
    h = impoly;
    msk = h.createMask;
end