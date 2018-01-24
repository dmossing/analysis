function run_alignment(foldname,lookfor)
    if ~exist('lookfor','var') || isempty(lookfor)
        lookfor = '';
    end
    lookfor
    d = dir([foldname '/*' lookfor '*.sbx'])
    for i=1:numel(d)
        fname = d(i).name
        filebase = strsplit(fname,'.sbx');
        filebase = filebase{1};
        load([foldname '/' filebase],'info')
        %if info.scanmode==0
        %    info.recordsPerBuffer = info.recordsPerBuffer*2;
        %end
%
%        try
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
                    sbxAlignmaster([foldname '/' filebase],depth,info.rect);
                catch
                    rect = [1 info.sz(1) 1 info.sz(2)];
                    sbxAlignmaster([foldname '/' filebase],depth,rect);
                end
            end
%        catch
%            sbxalignmaster([foldname '/' filebase])
%
%        which sbxAlignmaster
%        if isfield(info,'rect')
%            sbxAlignmaster([foldname '/' filebase],[],info.rect);
%        else
%            sbxAlignmaster([foldname '/' filebase]);
%            which sbxComputeci
%        	sbxComputeci([foldname '/' filebase]);
%
%        end
        %which sbxComputeci
        %sbxComputeci([foldname '/' filebase],[],info.rect);
    end
end
