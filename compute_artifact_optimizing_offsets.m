function artifact_cell = compute_artifact_optimizing_offsets(info,roifile,lights_on,noffset,sbxbase,filebase,from_raw,pad_nans)


if nargin < 4
    noffset = 100;
end

%%

[mskcdf,iplane] = compute_mskcdf(roifile,info);
[ctr,iplane] = append_roifile_parts(roifile,'ctr',true);
[data,~] = append_roifile_parts(roifile,'Data',false);
[neuropil,~] = append_roifile_parts(roifile,'Neuropil',false);
roiline = round(ctr(1,:));
roiline = roiline(:);
if isfield(info,'rect')
    roiline = roiline + info.rect(1);
end
[yoff,~] = append_roifile_parts(roifile,'yoff',false);
% for i=1:numel(roifile)
%     if i==1
%         yoff = roifile{i}.yoff;
%     else
%         yoff = [yoff; roifile{i}.yoff];
%     end
% end

%%
evaluation_fn = @(loffset1) evaluate_tv_loffset1(loffset1,roiline,mskcdf,iplane,neuropil,info,yoff,lights_on,data);
% evaluation_fn = @(loffset1) evaluate_tv_loffset1(loffset1,roiline,iplane,neuropil,info,lights_on,data);
ys = [evaluation_fn(-noffset) evaluation_fn(noffset)];
loffset1 = binary_search_offset([-noffset noffset],ys,evaluation_fn);

evaluation_fn = @(loffset2) evaluate_tv_loffset2(loffset2,roiline,mskcdf,iplane,neuropil,info,yoff,lights_on,data);
% evaluation_fn = @(loffset2) evaluate_tv_loffset2(loffset2,roiline,iplane,neuropil,info,lights_on,data);
ys = [evaluation_fn(-noffset) evaluation_fn(noffset)];
loffset2 = binary_search_offset([-noffset noffset],ys,evaluation_fn);

if from_raw
    artifact = compute_opto_artifact_decaying_exponential(sbxbase,filebase,mskcdf,iplane,neuropil,info,yoff,lights_on,loffset1,loffset2);
    costfun = @(x) evaluate_tv_multiplier(x,artifact,neuropil);
    multiplier = fminunc(costfun,1);

    artifact_cell = cell(size(roifile));
    for i=1:numel(roifile)
        artifact_cell{i} = multiplier*artifact(iplane==i,:);
    end
else
    artifact = compute_artifact_mskcdf(roiline,mskcdf,iplane,neuropil,info,yoff,lights_on,loffset1,loffset2,pad_nans);
    % artifact = compute_artifact_mskcdf(roiline,iplane,neuropil,info,lights_on,loffset1,loffset2);
    artifact_cell = cell(size(roifile));
    for i=1:numel(roifile)
        artifact_cell{i} = artifact(iplane==i,:);
    end
end

%%


function [output,iplane] = append_roifile_parts(roifile,fieldname,transpose)
if nargin < 3
    transpose = false;
end
output = cell(size(roifile));
iplane = cell(size(roifile));
minsize = inf;
for i=1:numel(roifile)
    output{i} = getfield(roifile{i},fieldname);
    sz = size(output{i},2);
    if sz<minsize
        minsize = sz;
    end
end
for i=1:numel(roifile)
    output{i} = output{i}(:,1:minsize);
    if transpose
        output{i} = output{i}';
    end
end
for i=1:numel(roifile)
    iplane{i} = i*ones(size(output{i},1),1);
end
output = cell2mat(output);
iplane = cell2mat(iplane);
if transpose
    output = output';
end

function xmin = binary_search_offset(xs,ys,evaluation_fn)
[~,minind] = min(ys);
if abs(diff(xs))==1
    xmin = xs(minind);
else
    xnew = floor(mean(xs));
    ynew = evaluation_fn(xnew);
    xmin = binary_search_offset([xs(minind) xnew],[ys(minind) ynew],evaluation_fn);
end

function tv = evaluate_tv_loffset1(loffset1,roiline,mskcdf,iplane,neuropil,info,yoff,lights_on,data)
artifact = compute_artifact_mskcdf(roiline,mskcdf,iplane,neuropil,info,yoff,lights_on,loffset1,0,0);
tv = sum(sum(abs(diff(data-artifact,[],2))));
% function tv = evaluate_tv_loffset1(loffset1,roiline,iplane,neuropil,info,lights_on,data)
% artifact = compute_artifact(roiline,iplane,neuropil,info,lights_on,loffset1,0);
% tv = sum(sum(abs(diff(data-artifact,[],2))));

function tv = evaluate_tv_loffset2(loffset2,roiline,mskcdf,iplane,neuropil,info,yoff,lights_on,data)
artifact = compute_artifact_mskcdf(roiline,mskcdf,iplane,neuropil,info,yoff,lights_on,0,loffset2,0);
tv = sum(sum(abs(diff(data-artifact,[],2))));
% function tv = evaluate_tv_loffset2(loffset2,roiline,iplane,neuropil,info,lights_on,data)
% artifact = compute_artifact(roiline,iplane,neuropil,info,lights_on,0,loffset2);
% tv = sum(sum(abs(diff(data-artifact,[],2))));

function tv = evaluate_tv_multiplier(multiplier,artifact,neuropil)
tv = sum(sum(abs(diff(neuropil-artifact*multiplier,[],2))));