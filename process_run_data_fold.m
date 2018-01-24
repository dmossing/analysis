function process_run_data_fold(foldname)
files = dir([foldname '/*.bin']);
for i=1:numel(files)
    [dx_dt,stim_trigger] = process_run_data([foldname '/' files(i).name]);
    save(strrep([foldname '/' files(i).name],'.bin','_running.mat'),'dx_dt','stim_trigger')
end