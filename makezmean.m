function zmean = makezmean(fname,zno,framesperz,varargin)
arglist = {{'ignore_movement',false},{'prctile_cutoff',90}};
[ignore_movement,prctile_cutoff] = parse_args(varargin,arglist);
try
    % single channel case
    ex = load2P(fname,'frames',1);
    zmean = zeros(zno,size(ex,1),size(ex,2));
    for i=1:zno
        i
        Images = squeeze(load2P(fname,'frames',(i-1)*framesperz+(1:framesperz)));
        if ~ignore_movement
            zmean(i,:,:) = mean(Images,3);
        else
            m = mean(Images,3);
            sqdiff = zeros(size(Images,3),1);
            for j=1:size(Images,3)
                sqdiff(j) = sum(sum((double(Images(:,:,j))-m).^2));
            end
            gd = sqdiff<prctile(sqdiff,prctile_cutoff);
            zmean(i,:,:) = mean(Images(:,:,gd),3);
        end
    end
catch
    % two channel case
    ex = load2P(fname,'frames',1,'channels',[1 2]);
    zmean = zeros(2,zno,size(ex,1),size(ex,2));
    for i=1:zno
        i
        Images = squeeze(load2P(fname,'frames',(i-1)*framesperz+(1:framesperz),'channels',[1 2]));
        if ~ignore_movement
            zmean(:,i,:,:) = permute(mean(Images,4),[3 1 2]);
        else
            m = permute(mean(Images,4),[3 1 2]); % channel goes first
            sqdiff = zeros(size(Images,3),1);
            for j=1:size(Images,3)
                sqdiff(j) = sum(sum(sum((double(Images(:,:,:,j))-m).^2)));
            end
            gd = sqdiff<prctile(sqdiff,prctile_cutoff);
            zmean(i,:,:) = permute(mean(Images(:,:,:,gd),4),[3 1 2]);
        end
    end
end
% sz = size(Images);
% Images = reshape(Images,sz(1),sz(2),framesperz,zno);
% zmean = permute(Images,[4 1 2 3]);