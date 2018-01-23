function z = ddigit(n,d)
	q = num2str(n);
	z = [repmat('0',1,d-length(q)) q];
end
