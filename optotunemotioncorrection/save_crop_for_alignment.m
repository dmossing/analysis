function save_crop_for_alignment(fns,sbxlocn,rect)
% fns: list of .sbx/.mat filenames (assumes both are in the same folder)
% sbxlocn: folder containing .sbx files
% rect: desired corners of cropped image: imcrop =
% im(rect(1):rect(2),rect(3):rect(4)). If not specified, it is manually set
% using the GUI by the user
% 
if nargin < 2
    sbxlocn = '';
end
try
    if isdir(fns)
        d = dir([fns '/*.sbx']);
        fns = {d(:).name};
    end
catch
end
for i=1:numel(fns)
    fname = fns{i};
    if fname(end-3:end) == '.sbx'
        fname = fname(1:end-4);
    end
    if sbxlocn
        fileparts = strsplit(fname,'/');
        fname_itself = fileparts{end};
        sbxfile = [sbxlocn '/' fname_itself];
    else
        sbxfile = fname;
    end
    load(fname,'info')
    if nargin < 3
        imgtocrop = sbxreadpacked(sbxfile,1,1);
        if numel(size(imgtocrop))==3
            imgtocrop = squeeze(imgtocrop(1,:,:));
        end
        [~, ~, rect] = crop(imgtocrop, true);
        %     [~, ~, rect] = crop(load2P([fname '.sbx'],'Frames',2), true);
        %     [~, ~, rect] = crop(sbxreadpacked(fname,1,1), true);
        %         rect = round([rect(3),rect(3)+rect(4),rect(1),rect(1)+rect(2)]);
        rect = [rect(2),rect(2)+rect(4),rect(1),rect(1)+rect(3)];
        rect(1) = max(rect(1),1);
        rect(3) = max(rect(3),1);
        rect(2) = min(rect(2),info.sz(1));
        rect(4) = min(rect(4),info.sz(2));
        if ~mod(rect(2)-rect(1),2)
            rect(2) = rect(2)-1;
        end
        if ~mod(rect(4)-rect(3),2)
            rect(4) = rect(4)-1;
        end
    else
        info.rect = rect;
        save([fname '.mat'],'info','-append')
    end
end
