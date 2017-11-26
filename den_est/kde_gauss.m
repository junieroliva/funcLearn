function [kde, p] = kde_gauss( x, varargin )
% KDE based on points x, evaluated at points xe.
%   Inputs -
%   x: Points KDE is based on
%   xe (optional): Points KDE will be evaluated at (xe = x if xe not
%       provided)
%   opts (optional): a struct of options with the following possible fields
%       sigma2: bandwidth to use
%       trunc: boolean indicating whether to use kernels of truncated
%           normals (true) or untruncated normals (false)
%       CV_ISE: CV bandwidth using integrated squared error if sigma2 not
%           provided and CV_ISE is true, or CV using negative log
%           likelihood if sigma2 not provided and CV_ISE is false

if ~isempty(varargin) && ~isstruct(varargin{1})
    xe = varargin{1};
else
    xe = [];
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
    sigma2 = get_opt(opts, 'sigma2', nan);
    if isnan(sigma2)
        N_rot = get_opt(opts, 'N_rot', min(N,20));
        rprm = randperm(N,N_rot);
        sigma2s = nan(N_rot,1);
        for ii=1:N_rot
            i = rprm(ii);
            param = kde_gauss(x{i}, opts);
            sigma2s(ii) = param.sigma2;
        end
        sigma2 = min(median(sigma2s),mean(sigma2s));
    end
    opts.sigma2 = sigma2;
    kde(N).sigma2 = [];
    kde(N).norma = [];
    if ~isempty(xe)
        p = cell(N,1);
        for i=1:N
            [kde(i), p{i}] = kde_gauss(x{i}, xe, opts);
        end
    else
        for i=1:N
            kde(i) = kde_gauss(x{i}, opts);
        end
    end
else
    [n,~] = size(x);
    trunc = get_opt(opts, 'trunc', true);
    sigma2 = get_opt(opts, 'sigma2', nan);
    CV_ISE = get_opt(opts, 'CV_ISE', true);

    cv = [];
    if isnan(sigma2) % cross-validate bandwidth 
        D2xx = slmetric_pw(x',x','sqdist');
        sigma2_rot = mean(mean(D2xx));
        mult = get_opt(opts, 'mult', 2.^(-20:10));
        scores = nan(length(mult),1);
        PX = nan(n,length(mult));
        for mi=1:length(mult)
            sigma2 = sigma2_rot*mult(mi);
            % compute leave one out density estimates
            norma = gauss_norma(x,sigma2,trunc);
            px = (n/(n-1))*(exp(-D2xx/(2*sigma2))*norma-norma);
            PX(:,mi) = px;
            % if CVing ISE, compute the L2 norm squared
            if CV_ISE
                C = gauss_prod_C(x, sigma2, x, sigma2, trunc);
                p2 = sum(sum(exp(-D2xx/(4*sigma2)).*C.*(norma*norma')));
                scores(mi) = p2 - 2*mean(px);
            else % use negative LL score
                scores(mi) = -sum(log(px));
            end
        end
        [~, mi] = min(scores);
        sigma2 = sigma2_rot*mult(mi);
        % record results of CVing
        cv.PX = PX;
        cv.scores = scores;
        cv.sigma2s = sigma2_rot*mult';
        cv.CV_ISE = CV_ISE;
    end
    norma = gauss_norma(x,sigma2,trunc);
    kde.sigma2 = sigma2;
    kde.norma = norma;
    if ~isempty(cv)
        kde.cv = cv;
    end
    if ~isempty(xe)
        D = n;
        D2 = get_opt(opts,'D2');
        if ~isempty(D2)
            p = exp(-D2/(2*sigma2))*norma;
        else
            maxmem = get_opt(opts,'maxmem',2^30); % use no more than this to eval
            nstep = ceil(maxmem/(8*D));
            ne = size(xe,1);
            p = nan(ne,1);
            for ci = 1:ceil(ne/nstep)
                n1 = (ci-1)*nstep+1;
                n2 = min(ne,ci*nstep);
                p(n1:n2) = exp(-slmetric_pw(xe(n1:n2,:)',x','sqdist')/(2*sigma2))*norma;
            end
        end
        
%         mp = min(p(p>eps));
%         if ~isempty(mp)
%             p(p<=eps) = mp;
%         else
%             p(p<=eps) = eps;
%         end
        
%         trunc_val = get_opt(opts,'trunc_val',eps);
%         p = trunc_density(p,trunc_val);
    end
end

end

