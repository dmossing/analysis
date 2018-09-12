function varargout = parse_args(varargin,arglist)
% arglist a cell array of argument-default pairs
if ~iscell(arglist{1})
    arglist = {arglist};
end
p = inputParser;
for i=1:numel(arglist)
    p.addParameter(arglist{i}{1},arglist{i}{2});
end
p.parse(varargin{:})
result = p.Results;
assert(nargout==numel(fieldnames(result)));
for i=1:nargout
    varargout{i} = result.(arglist{i}{1});
end