function [pcs, inds, SM] = osfe(x, y, varargin)
[N,n,p] = size(y);
y = permute(y,[2 3 1]);

if isempty(varargin)
    opts = struct;
else
    opts = varargin{1};
end

N_rot = get_opt(opts, 'N_rot', min(N,20));
p_rot = get_opt(opts, 'p_rot', min(p,10));

if ~iscell(x) % TODO: implement
    cv_norms = nan(N_rot, p_rot);
    irprm = randperm(N,N_rot);
    jrprm = randperm(p,p_rot);
    for ii=1:N_rot
        for jj=1:p_rot
            [~,~,inds] = cv_os(x,squeeze(y(:,jrprm(jj),irprm(ii))),opts);
            cv_norms(ii,jj) = max(sqrt(sum(inds.^2,2)));
            end
            end
            end

phix = evalbasis(x,inds, 
