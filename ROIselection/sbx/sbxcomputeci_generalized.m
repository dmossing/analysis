function [c3,xray] = sbxcomputeci_generalized(data,Q,s,thestd)

%     vals = load([fname '.align'],'-mat','T','v','Q','thestd');
% 
%     T = vals.T;
% 
%     Q = vals.Q;
% 
%     s = sqrt(vals.v);
% 
%     thestd = vals.thestd;
% 
%     
% 
%     sbxread(fname,0,1);
% 
%     global info

    

%     %Compute sum, sum of squares
% 
%     
% 
%     imsize = size(vals.v); %[info.recordsPerBuffer,size(info.S,2)];   
% 
%     nframes = info.max_idx;

imsize = size(s);

nframes = size(data,3);

    %mean, standard deviation    

    %compute correlation coefficient with respect to 3x3 window

%     winsize = 35;
% for LF, choose smaller window
    winsize = 7;

%     res = .5;
    res = 1;

    



    p = gcp();

    nblocks = p.NumWorkers;

    

    xray = zeros([imsize*res,winsize,winsize,nblocks],'double');

    c = zeros(imsize(1),imsize(2),2,nblocks);

        

    maxidx = size(data,3);

    datablocks = cell(nblocks,1);
    for ii = 1:nblocks
        rg = floor((ii-1)*maxidx/nblocks)+1:floor(ii*maxidx/nblocks);
        datablocks{ii} = data(:,:,rg);
    end
        
    
    for ii = 1:nblocks % split frames evenly across cores

%         rg = floor((ii-1)*maxidx/nblocks)+1:floor(ii*maxidx/nblocks);

        [c(:,:,:,ii),xray(:,:,:,:,ii)] = doOneBlock(datablocks{ii},imsize,res,winsize,Q,s,nframes,thestd);

    end

    c = sum(c,4);

%     xray = sum(xray,5);
    xray = nansum(xray,5)/nframes;




    c3 = (c(:,:,2)-c(:,:,1))/8/nframes;

%     xray = single(xray/nframes/2);

%     xray = int16(xray*2^15);

    

%     save([fname '.align'],'-mat','c3','xray','-append');

    

end



function [c,xray] = doOneBlock(datablock,imsize,res,winsize,Q,s,nframes,thestd)

    xray = zeros([imsize*res,winsize,winsize],'double');

    c = zeros(imsize(1),imsize(2),2);



    Am = zeros([imsize*res,winsize,winsize],'double');

    Ar2 = 0;

    

    

    for nn = 1:size(datablock,3)

%         A = double(sbxchan0(sbxreadpacked(fname,nn-1,1)));        % load frame
        Ar = datablock(:,:,nn)./thestd;

%         A = A./thestd;                                  % normalize by ? (whiten image?)

%         Ar = circshift(A,T(nn,:));                      % apply motion correction

        Ar = reshape(Ar(:) - Q*(Q'*Ar(:)),size(Ar));

        Ar = Ar./s;                                     % normalize by ?



        c(:,:,1) = c(:,:,1) + Ar.^2;

        c(:,:,2) = c(:,:,2) + conv2(ones(3,1),ones(1,3),Ar,'same').*Ar;

        

        %Ar = imresize(Ar,res);

        Ar = conv2([.5,1,.5],[.5,1,.5],Ar,'same')/4;    % blur

%         Ar = Ar(2:2:end,2:2:end);                       % subsample

        Ar2 = Ar2 + Ar;

        if mod(nn,2)==0 % every other frame

            %At low temporal res

            Ar = Ar2;

            for ii = 1:winsize

                for jj = 1:winsize

                    Am(:,:,ii,jj) = Ar.*circshift(Ar,[-(ii-ceil(winsize/2)),-(jj-ceil(winsize/2))]);

                end

            end



            xray = xray + Am;

            Ar2 = 0;

        end



        if mod(nn,100)==0

            fprintf('Pass 2, #%d/%d\n',nn,nframes);

        end

    end

end