function fold_smooth_eye_tracks(foldname,dry_run)
if nargin < 2
    dry_run = false;
end
d = dir(foldname);
forbidden = {'.','..','Duplicate','.DS_Store'};
if ~dry_run
    for i=1:numel(d)
        if ~ismember(d(i).name,forbidden) && ~contains(d(i).name,'eye_tracking')
            if exist([foldname '/eye_tracking_' d(i).name '.mat'])
                i
                thisfold = [foldname '/' d(i).name];
                tic
                load([foldname '/eye_tracking_' d(i).name '.mat'],'ctr','area','props','ctr2','area2','props2','hulls');
                toc
                [ctr_sm,area_sm] = smooth_eye_tracks(thisfold,props2);
                save([foldname '/eye_tracking_' d(i).name '.mat'],'ctr_sm','area_sm','-append')
            end
        end
    end
end

function [ctr_sm,area_sm] = smooth_eye_tracks(foldname,props2)
area = props2(:,3);
ctr = props2(:,4:5);
load([foldname '/msk.mat'],'msk');
msk = impyramid(msk,'reduce')>0.5;
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
    d(idxframe) = d; % now correcting the order here
end
iframe = 1;
data = double(imread([foldname '/' d(iframe).name]));
data = impyramid(data,'reduce');
[xx,yy] = meshgrid(1:size(data,2),1:size(data,1));
ctr_sm = ctr;
filt_area = 25;
filt_ctr = 5;
for i=1:2
    ctr_sm(:,i) = medfilt1(ctr(:,i),filt_ctr,'omitnan');
end
area_sm = medfilt1(area,filt_area,'omitnan');
for iframe=1:numel(d)
    if rem(iframe,100)==0
        disp(num2str(iframe))
    end
    data = double(imread([foldname '/' d(iframe).name]));
    data = impyramid(data,'reduce');
    z = compute_zscore(data,xx,yy,msk,ctr(iframe,:),area(iframe));
    zsm = compute_zscore(data,xx,yy,msk,ctr_sm(iframe,:),area_sm(iframe));
    if zsm < z
        ctr_sm(iframe,:) = ctr(iframe,:);
        area_sm(iframe) = area(iframe);
    end
end


function z = compute_zscore(data,xx,yy,msk,ctr,area)
rad = sqrt(area/pi);
in_pupil = (xx-ctr(1)).^2 + (yy-ctr(2)).^2 <= rad^2;
not_in_pupil = msk & ~in_pupil;
data_in = data(msk & in_pupil);
data_out = data(msk & ~in_pupil);
mean_in = mean(data_in);
std_in = std(data_in);
mean_out = mean(data_out);
z = (mean_in - mean_out)/std_in;