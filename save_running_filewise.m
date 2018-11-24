function save_running_filewise(runfiles,roifiles,nbefore,nafter)
if nargin < 3
    nbefore = 15;
    nafter = 30;
end
% fnames = runfiles;
for i=1:numel(fnames)
    try
        runname = runfiles{i};
        roiname = roifiles{i};
        [dx_dt,stim_trigger] = process_run_data(runname);
        othername = name;
        try
            load([roiname '.rois'],'-mat','Data')
            load([roiname '.mat'],'info')
        catch
            othername = strrep(name,'.bin','_ot_001.bin');
            load([roifoldname '/' strrep(othername,'.bin','.rois')],'-mat','Data')
            load([roifoldname '/' strrep(othername,'.bin','.mat')],'info')
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
        save([roifoldname '/' strrep(othername,'.bin','.rois')],'-mat','-append','dxdt')
        trialrun = trialize(dxdt,info.frame(info.event_id==1),nbefore,nafter);
        save([roifoldname '/' strrep(othername,'.bin','.rois')],'-mat','-append','trialrun')
    catch
        warning(['unable to save running data for ' fnames{i}])
    end
end