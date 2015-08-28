function varargout = multi_output( x, varargin )
%MULTI_OUTPUT function that returns a variable number of output as
%  [a1, ..., an] = multi_output(x, f1, ..., fn);
%    where ai = fi(x)
% TODO: easier way to do this?
for i=1:nargout
    varargout{i} = varargin{i}(x);
end

end

