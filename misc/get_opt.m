function opt = get_opt(opts, name, varargin)
    default = [];
    if ~isempty(varargin)
        default = varargin{1};
    end
    if isfield(opts,name)
        opt = opts.(name);
    else
        opt = default;
    end
end