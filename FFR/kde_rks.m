function [rks, p] = kde_rks( x, varargin )

if ~isempty(varargin) && ~isstruct(varargin{1})
    xe = varargin{1};
else
    xe = x;
end
if ~isempty(varargin) && isstruct(varargin{1})
    opts = varargin{1};
elseif length(varargin)>1
    opts = varargin{2};
else
    opts = struct;
end

p = [];
if iscell(x) % bulk mode, cv sigma2 on a subset if needed
    N = length(x);
    d = size(x{1},2);
    rks = get_opt(opts, 'rks', struct);
    rks.trunc = get_opt(rks, 'trunc', true);
    % check if given a bandwidth parameter, if not CV on a subset of sets
    if ~isfield(rks,'sigma2')
        N_rot = get_opt(opts, 'N_rot', min(N,20));
        rprm = randperm(N,N_rot);
        sigma2s = nan(N_rot,1);
        for ii=1:N_rot
            i = rprm(ii);
            param = kde_gauss(x{i}, opts);
            sigma2s(ii) = param.sigma2;
        end
        sigma2 = mean(sigma2s);
    end
    % get the random features (basis functions)
    rks.sigma2 = sigma2;
    rks.D = get_opt(rks, 'D', 1000);
    rks.W = sqrt(1./rks.sigma2)*randn(d,rks.D);
    rks.b = 2*pi*rand(1,rks.D);
    % compute the "projection coefficients"
    opts.rks = rks;
    pc = nan(N,D);
    if ~isempty(xe)
        p = cell(N,1);
        for i=1:N
            [rksp, p{i}] = kde_rks(x{i}, xe, opts);
            pc(i,:) = rksp.pc;
        end
    else
        for i=1:N
            rksp = kde_rks(x{i}, opts);
            pc(i,:) = rksp.pc;
        end
    end
    rks.pc = pc;
else
    [~,d] = size(x);
    rks = get_opt(opts, 'rks', struct);
    
    % check if we don't have all parameters neccesary for density estimation
    if ~isfield(rks,'pc') || ~isfield(rks,'W') || ~isfield(rks,'b') 
        rks.trunc = get_opt(rks, 'trunc', true);
        % check if we only need to compute projection coefficients
        if isfield(rks,'W') && isfield(rks,'b') && isfield(rks,'sigma2')
            rks.pc = sqrt(2/size(rks.W,2))*cos(bsxfun(@plus,x*rks.W,rks.b))...
                        *gauss_norma(x,rks.sigma2,rks.trunc);
        else % need to compute everything
            if ~isfield(rks,'sigma2')
                kde = kde_gauss( x, rks );
                rks.sigma2 = kde.sigma2;
                rks.norma = kde.norma;
            else
                rks.norma = gauss_norma(x,rks.sigma2,rks.trunc);
            end
            
            rks.D = get_opt(rks, 'D', 1000);
            rks.W = sqrt(1./rks.sigma2)*randn(d,rks.D);
            rks.b = 2*pi*rand(1,rks.D);
            rks.pc = (rks.norma'*sqrt(2/rks.D)*cos(bsxfun(@plus,x*rks.W,rks.b)))';
        end
    end

    if ~isempty(xe)
        p = sqrt(2/size(rks.W,2))*cos(bsxfun(@plus,xe*rks.W,rks.b))*rks.pc;
        p(p<=eps) = min(p(p>eps));
    end
end

end