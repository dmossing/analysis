function sig = sbxpullsignalspacked(fname)

%       sig = sbxpullsignalspacked(fname);

load([fname '.segment'],'-mat'); % load segmentation

mask = squeeze(max(bsxfun(@times, permute(mask,[1,3,2]), 1:size(mask,3)),[],2)); % /Evan/

%sbxgetinfo(fname);

sbxreadpacked(fname,0,1);

global info;



ncell = max(mask(:))



for(i=1:ncell)

    idx{i} = find(mask==i);

end



g = exp(-(-10:10).^2/2/2^2);

maskb = conv2(g,g,double(mask>0),'same')>.02; % dilation for border region around ROIs



centroids = regionprops(mask,'Centroid');

[xi,yi] = meshgrid(1:796,1:512);



idx_pil = cell(ncell,1);

for nn = 1:ncell

    for neuropilrad = 40:5:100

        M = (xi-centroids(nn).Centroid(1)).^2+(yi-centroids(nn).Centroid(2)).^2 < neuropilrad^2;

        neuropilmask = M.*~maskb;

        if nnz(neuropilmask) > 4000

            break

        end

    end

    idx_pil{nn} = find(neuropilmask);

end

info.max_idx

sig = zeros(info.max_idx, ncell);

pil = zeros(info.max_idx, ncell);



%h = waitbar(0,sprintf('Pulling %d signals out...',ncell));

% T = info.aligned.T;
load([fname,'.align'], 'T', '-mat');


parfor(i=0:info.max_idx-1)

    %waitbar(i/(info.max_idx-1),h);          % update waitbar...

    z = sbxreadpacked(fname,i,1);

    z = circshift(z,T(i+1,:)); % align the image

    for(j=1:ncell)                          % for each cell

        sig(i+1,j) = mean(z(idx{j}));       % pull the mean signal out...

        pil(i+1,j) = trimmean(z(idx_pil{j}),10);

    end

    

end

%delete(h);



save([fname '.signals'],'sig','pil');     % append the motion estimate data...





