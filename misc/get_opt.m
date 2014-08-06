function opt = get_opt(opts, name, default)
if isfield(opts,name)
    opt = opts.(name);
else
    opt = default;
end
end