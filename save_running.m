function save_running(runfoldname,roifoldname,nbefore,nafter)
if nargin < 3
    nbefore = 4;
    nafter = 8;
end
fnames = dir([runfoldname '/*.bin']);
fnames = {fnames(:).name};
for i=1:numel(fnames)
    try
        name = fnames{i};
        [dx_dt,stim_trigger] = process_run_data([runfoldname '/' name]);
        othername = strrep(name,'.bin','.rois');
        try
            load([roifoldname '/' othername],'-mat','Data')
            load([roifoldname '/' strrep(othername,'.rois','.mat')],'info')
            othernames = {othername};
            ndepths = 1;
        catch
            d = dir(strrep(othername,'.rois','*.rois'));
            ndepths = numel(d);
            othernames = cell(ndepths,1);
            for j=1:ndepths
                othernames{j} = d(j).name;
            end
            load([roifoldname '/' othernames{1}],'-mat','Data')
            try
                load([roifoldname '/' strrep(othernames{1},'.rois','.mat')],'info')
            catch
                load([roifoldname '/' strrep(othernames{1},'_ot_000.rois','.mat')],'info')
            end
        end
        frm = info.frame(info.event_id==1);
        frm_run = find(stim_trigger);
        [ufrm,uidx] = unique(frm);
        if numel(ufrm)~=numel(frm)
            frm = frm(uidx);
            frm_run = frm_run(uidx);
        end
        dxdt = resamplebytrigs(dx_dt,size(Data,2),frm_run,frm); %find(stim_trigger),info.frame(info.event_id==1));
        %     save([runfoldname '/' strrep(name,'.bin','_running.mat')],'dxdt')
        for j=1:ndepths
            save([roifoldname '/' othernames{j}],'-mat','-append','dxdt')
            try
                trialrun = trialize(dxdt,info.frame(info.event_id==1),nbefore,nafter);
                save([roifoldname '/' othernames{j}],'-mat','-append','trialrun')
            catch
                disp(['could not trialize depth #' num2str(j)])
            end
        end
    catch
        warning(['unable to save running data for ' fnames{i}])
    end
end

% function save_running(runfoldname,roifoldname,nbefore,nafter)
% if nargin < 3
%     nbefore = 15;
%     nafter = 30;
% end
% fnames = dir([runfoldname '/*.bin']);
% fnames = {fnames(:).name};
% for i=1:numel(fnames)
%     try
%         name = fnames{i};
%         [dx_dt,stim_trigger] = process_run_data([runfoldname '/' name]);
%         othername = name;
%         try
%             load([roifoldname '/' strrep(name,'.bin','.rois')],'-mat','Data')
%             load([roifoldname '/' strrep(name,'.bin','.mat')],'info')
%         catch
%             othername = strrep(name,'.bin','_ot_001.bin');
%             load([roifoldname '/' strrep(othername,'.bin','.rois')],'-mat','Data')
%             load([roifoldname '/' strrep(othername,'.bin','.mat')],'info')
%         end
%         frm = info.frame(info.event_id==1);
%         frm_run = find(stim_trigger);
%         [ufrm,uidx] = unique(frm);
%         if numel(ufrm)~=numel(frm)
%             frm = frm(uidx);
%             frm_run = frm_run(uidx);
%         end
%         dxdt = resamplebytrigs(dx_dt,size(Data,2),frm_run,frm); %find(stim_trigger),info.frame(info.event_id==1));
%         %     save([runfoldname '/' strrep(name,'.bin','_running.mat')],'dxdt')
%         save([roifoldname '/' strrep(othername,'.bin','.rois')],'-mat','-append','dxdt')
%         trialrun = trialize(dxdt,info.frame(info.event_id==1),nbefore,nafter);
%         save([roifoldname '/' strrep(othername,'.bin','.rois')],'-mat','-append','trialrun')
%     catch
%         warning(['unable to save running data for ' fnames{i}])
%     end
% end