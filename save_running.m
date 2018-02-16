function save_running(runfoldname,roifoldname)
fnames = dir([runfoldname '/*.bin']);
fnames = {fnames(:).name};
for i=1:numel(fnames)
    name = fnames{i};
    [dx_dt,stim_trigger] = process_run_data([runfoldname '/' name]);
    othername = name;
    try
        load([roifoldname '/' strrep(name,'.bin','.rois')],'-mat','Data')
        load([roifoldname '/' strrep(name,'.bin','.mat')],'info')
    catch
        othername = strrep(name,'.bin','_ot_000.bin');
        load([roifoldname '/' strrep(othername,'.bin','.rois')],'-mat','Data')
        load([roifoldname '/' strrep(othername,'.bin','.mat')],'info')
    end
    dxdt = resamplebytrigs(dx_dt,size(Data,2),find(stim_trigger),info.frame(info.event_id==1));
%     save([runfoldname '/' strrep(name,'.bin','_running.mat')],'dxdt')
    trialrun = trialize(dxdt,info.frame(info.event_id==1),15,30);
    save([roifoldname '/' strrep(othername,'.bin','.rois')],'-mat','-append','dxdt','trialrun')
end