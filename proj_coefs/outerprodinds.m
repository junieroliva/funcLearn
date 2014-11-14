function inds = outerprodinds( list, d, varargin )
list = list(:);
T = length(list);

if isempty(varargin)
    max_norm = inf;
else
    max_norm = varargin{1};
end

inds = list;
for i=2:d
    inds = [repmat(inds, T, 1), kron(list,ones(size(inds,1),1))];
    if ~isinf(max_norm)
        inds = inds(sum(inds.^2,2)<=max_norm^2,:);
    end
end


end

