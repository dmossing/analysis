function [dx_dt,stim_trigger] = process_run_data(fname)

% read in data
fid = fopen(fname);
Data = fread(fid,[4 Inf],'uint8')';
fclose(fid);

% set filtering options
samplingFrequency = 30000;
dsamp = 20;
dsamp_Fs = samplingFrequency / dsamp;
smooth_win = gausswin(dsamp_Fs, 23.5/2);
smooth_win = smooth_win/sum(smooth_win);
sw_len = length(smooth_win);
d_smooth_win = [0;diff(smooth_win)]/(1/dsamp_Fs);

% compute x_t, dx_dt
RunChannelIndices = [2 1];
DataIn = Data(:,RunChannelIndices);
stim_trigger = Data(:,4);
Data = [0;diff(DataIn(:,1))>0];             % gather pulses' front edges
Data(all([Data,DataIn(:,2)],2)) = -1;       % set backwards steps to be backwards
x_t = downsample(cumsum(Data), dsamp);      % convert pulses to counter data & downsample data to speed up computation
x_t = padarray(x_t, sw_len, 'replicate');   % pad for convolution
dx_dt = conv(x_t, d_smooth_win, 'same');    % perform convolution
dx_dt([1:sw_len,end-sw_len+1:end]) = [];

stim_trigger = [0;diff(stim_trigger)>0];
stim_trigger = downsample(cumsum(stim_trigger), dsamp);
stim_trigger = [0;diff(stim_trigger)>0];