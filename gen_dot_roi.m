function gen_dot_roi(sourcefold,targetfold)
% convert suite2P output to .rois files like I've been using
% sourcefold = './';
% targetfold = '/home/mossing/scratch/2Pdata/180802/M9053/ot/';
vars_of_interest = {'meanImg','meanImg_chan2','meanImg_chan2_corrected','meanImgE'};
d = dir([sourcefold '/*_proc.mat']);
targetfiles = matchfiles(sourcefold,targetfold)
for i=1:numel(d)
    nonproc = strsplit(d(i).name,'_proc.mat');
    nonproc = [nonproc{1}];
    sourcefile = [sourcefold '/' nonproc '.mat'];
    load(sourcefile,'ops');
%     var_list = {};
%     val_list = {};
    to_append = struct;
    for ivar=1:numel(vars_of_interest)
        this_var = vars_of_interest{ivar};
        if isfield(ops,this_var)
            this_val = getfield(ops,this_var);
            to_append = setfield(to_append,this_var,this_val);
%             var_list = {var_list; this_var};
%             val_list = {val_list; this_val};
        end
    end
%     nonproc = d(i).name;
    allROIdata = suite2P2ROIdata(sourcefile,'ROIindex',true);
    redratio = load(sourcefile,'redratio');
    red_saved = isfield(redratio,'redratio');
    if red_saved
        redratio = redratio.redratio;
    end
    for j=1:numel(allROIdata)
        nroi = numel(allROIdata{1}.rois);
        nt = numel(allROIdata{j}.rois(1).rawdata);
        Data = zeros(nroi,nt);
        Neuropil = zeros(nroi,nt);
        for k=1:nroi
            Data(k,:) = double(allROIdata{j}.rois(k).rawdata);
            Neuropil(k,:) = double(allROIdata{j}.rois(k).rawneuropil);
        end
        ROIdata = allROIdata{j};
        planeno = ddigit(str2num(nonproc(end))-1,3);
        targetfile = [targetfiles{j}(1:end-4) '_ot_' planeno '.rois'];
%         expno = ddigit(str2num(expts{j}),3);
%         targetfile = [nonproc(1:end-10) expno '_ot_' planeno '.rois'];
        save([targetfold '/' targetfile],'ROIdata','Data','Neuropil','sourcefile','-v7.3')
        save([targetfold '/' targetfile],'-struct','to_append','-append')
    end
end

function targetfiles = matchfiles(sourcefold,targetfold)
s = what(sourcefold);
p = s.path;
expts = strsplit(p,'/');
expts = expts{end};
expts = strsplit(expts,'_');
targetfiles = cell(size(expts));
for j=1:numel(expts)
    expno = ddigit(str2num(expts{j}),3);
    d = dir([targetfold '/M*' expts{j} '.mat']);
    if numel(d)~=1
        d = dir([targetfold '/../M*' expts{j} '.mat']);
        if numel(d)==1
            copyfile([targetfold '/../' d(1).name],[targetfold '/'])
        end
    end
    targetfiles{j} = d(1).name;
end
    
