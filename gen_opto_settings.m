function opto_settings = gen_opto_settings(info,result,frame_offset)
% delay between when ttl trigger is received, and when LED seems to come on
ttl_delay = 7;
opto_settings = [];
opto_settings.lights_on = result.gratingInfo.lightsOn(end,:);
opto_settings.frame = info.frame(frame_offset+1:end);
opto_settings.line = info.line(frame_offset+1:end);
opto_settings.line(1:4:end) = opto_settings.line(1:4:end) - ttl_delay;
opto_settings.line(4:4:end) = opto_settings.line(4:4:end) + ttl_delay;
% correct situations where light actually comes on in the next frame
undershot = opto_settings.line < 1;
opto_settings.line(undershot) = 512 + opto_settings.line(undershot);
opto_settings.frame(undershot) = opto_settings.frame(undershot) - 1;
overshot = opto_settings.line > 512;
opto_settings.line(overshot) = -512 + opto_settings.line(overshot);
opto_settings.frame(overshot) = opto_settings.frame(overshot) + 1;