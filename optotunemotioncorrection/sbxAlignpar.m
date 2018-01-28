function [m,v,T] = sbxAlignpar(fname,thestd,gl,l,Frames,numDepths,rect)



% Aligns images in fname for all indices in idx

% 

% m - mean image after the alignment

% T - optimal translation for each frame

    if ~exist('rect','var') || isempty(rect)
        rect = [];
    end
    
    A = sbxreadpacked(fname,1,1);
    
    global info

    
    numFrames = size(Frames,2);
    

    p = gcp();

    nblocks = 2^floor(log2(p.NumWorkers));

    

    rg = [Frames(1,:)-1;Frames(2,:)];

    rgs = spliteven(rg,log2(nblocks));    

    
    if isempty(rect)
        rg1 = 33:size(A,1);
        rg2 = 46:size(A,2)-45;
    else
        rg1 = rect(1):rect(2);
        rg2 = rect(3):rect(4);
    end
    
    
    size(thestd)

    thestd = thestd(rg1,rg2);


    

    A = A(rg1,rg2);

    l = l(rg1,rg2);

    

    ms = zeros([size(A),nblocks]);

    vs = zeros([size(A),nblocks]);

    Ts = cell(nblocks,1);

    

    parfor ii = 1:nblocks %

        subrg = rgs{ii};

        [ms(:,:,ii),vs(:,:,ii),Ts{ii}] = sbxalignsub(fname,subrg,rg1,rg2,thestd,gl,l,numDepths);   

    end

    

    %non-recursive

    for nn = 1:log2(nblocks)

        nblocksafter = 2^(log2(nblocks)-nn);

        msnew = zeros([size(A),nblocksafter]);

        vsnew = zeros([size(A),nblocksafter]);

        Tsnew = cell(nblocksafter,1);

        

        for ii = 1:nblocksafter

            [u,v] = fftAlign(ms(:,:,ii*2-1)/length(Ts{ii*2-1}), ms(:,:,ii*2  )/length(Ts{ii*2  }));



            Tsnew{ii} = [bsxfun(@plus,Ts{ii*2-1},[u,v]);

                         Ts{ii*2}];

            Ar = circshift(ms(:,:,ii*2-1),[u, v]);

            msnew(:,:,ii) = (Ar+ms(:,:,ii*2  ));

            

            Ar = circshift(vs(:,:,ii*2-1),[u, v]);

            vsnew(:,:,ii) = (Ar+vs(:,:,ii*2  ));



        end

        

        ms = msnew;

        vs = vsnew;

        Ts = Tsnew;

    end

    

    m = ms/numFrames;

    v = vs/numFrames;

    T = Ts{1};

end



function A = spliteven(idx,ns)

    if ns > 0

        idx0 = idx(:,1:floor(end/2));

        idx1 = idx(:,floor(end/2)+1 : end);

        idx0s = spliteven(idx0,ns-1);

        idx1s = spliteven(idx1,ns-1);

        A = {idx0s{:},idx1s{:}};

    else

        A = {idx};

    end

    

end



function [m,v,T] = sbxalignsub(fname,idx,rg1,rg2,thestd,gl,l,numDepths)

    if(size(idx,2)==1)



        A = double(sbxreadpacked(fname,idx(1,1),1));
        
        A = A(rg1,rg2);

        A = A - gl(idx(2,1))*l;

        A = A./thestd;

        

        m = A;

        v = A.^2;

        T = [0 0];



    elseif (size(idx,2)==2)



        A = double(sbxreadpacked(fname,idx(1,1),1));
        
        A = A(rg1,rg2);

        A = A - gl(idx(2,1))*l;

        A = A./thestd;

        

        B = double(sbxreadpacked(fname,idx(1,2),1));

        B = B(rg1,rg2);

        B = B - gl(idx(2,2))*l;

        B = B./thestd;

        

        [u,v] = fftAlign(A,B);



        Ar = circshift(A,[u,v]);

        m = Ar+B;

        T = [[u v] ; [0 0]];



        Ar = circshift(A.^2,[u,v]);

        v = (Ar+B.^2);



    else



        idx0 = idx(:,1:floor(end/2));

        idx1 = idx(:,floor(end/2)+1 : end);

        [A,v1,T0] = sbxalignsub(fname,idx0,rg1,rg2,thestd,gl,l,numDepths);

        [B,v2,T1] = sbxalignsub(fname,idx1,rg1,rg2,thestd,gl,l,numDepths);



        [u,v] = fftAlign(A/size(idx0,2), B/size(idx1,2));



        Ar = circshift(A,[u, v]);

        m = (Ar+B);

        T = [(ones(size(T0,1),1)*[u v] + T0) ; T1];



        v1 = circshift(v1,[u, v]);

        v = v1+v2;



    end

end
