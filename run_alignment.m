function run_alignment(foldname,lookfor,c)
    if ~exist('lookfor','var') || isempty(lookfor)
        lookfor = '';
    end
    if ~exist('c','var') || isempty(c)
        c = 1;
    end
    lookfor
    d = dir([foldname '/*.sbx'])
    for i=1:numel(d)
        fname = d(i).name
        filebase = strsplit(fname,'.sbx');
        filebase = filebase{1};
        load([foldname '/' filebase],'info')
            which sbxAlignmaster
            %sbxAlignmaster([foldname '/' filebase],[],info.rect);
            load([foldname '/' filebase],'info')
            try
                numDepths = info.otparam(3);
            catch
                numDepths = 1;
            end
            for depth=1:numDepths
                try
                    sbxAlignmaster([foldname '/' filebase],depth,info.rect,c);
                catch
                    rect = [1 info.sz(1) 1 info.sz(2)];
                    sbxAlignmaster([foldname '/' filebase],depth,rect,c)
                end
            end
    end
end
