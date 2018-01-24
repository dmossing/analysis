function ftotal = sumwholefov(filename)
segfilename = strrep(filename,'.sbx','.segment');
if exist(segfilename)
    subtract_rois = true;
    load(segfilename,'-mat','mask','dim')
    masks_ignored = uint16(~full(reshape(sum(mask,2),dim)));
else
    subtract_rois = false;
end
chunksize = 100;
ftotal = [];
currentframe = 1;
while true
    currentframe
    try
        theseframes = load2P(filename,'frames',currentframe:currentframe+chunksize-1);
        if subtract_rois
            for i=1:size(theseframes,5)
                theseframes(:,:,1,1,i) = theseframes(:,:,1,1,i).*masks_ignored;
            end
        end
        ftotal = [ftotal; squeeze(sum(sum(theseframes)))];
        currentframe = currentframe+chunksize;
    catch
%         theseframes = load2P(filename,'frames',[currentframe inf]);
        break
    end
end
ftotalfilename = strrep(filename,'.sbx','_ftotal.mat');
save(ftotalfilename,'ftotal')