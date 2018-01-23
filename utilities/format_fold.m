function x = format_fold(x)
	x = strrep(x,'\','/');
	if x(end)~='/'
		x = [x '/'];
	end
