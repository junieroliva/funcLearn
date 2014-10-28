function [p, norma, sigma2] = kde_gauss( x, varargin )
%kde_gauss  KDE based on points x, evaluated at points xe.
%   Inputs -
%   xe: Points KDE will be evaluated at
%   x: Points KDE is based on
%   opts (optional): a struct of options with the following possible fields
%       sigma2: bandwidth to use
%       trunc: boolean indicating whether to use kernels of truncated
%           normals (true) or untruncated normals (false)

if ~isempty(varargin)
    xe = varargin{1};
else
    xe = x;
end
if length(varargin)<=1;
    opts = struct;
else
    opts = varargin{2};
end

[n,d] = size(x);
trunc = get_opt(opts, 'trunc', true);
sigma2 = get_opt(opts, 'sigma2', nan);

if isnan(sigma2) % cross-validate bandwidth based on LOO-LL
                 % TODO: cv based on IMSE
    D2xx = slmetric_pw(x',x','sqdist');
    sigma2_rot = mean(mean(D2xx));
    mult = get_opt(opts, 'mult', 2.^(-10:10));
    ll = nan(length(mult),1);
    for mi=1:length(mult)
        sigma2 = sigma2_rot*mult(mi);
        norma = gauss_norma(x,sigma2,trunc);
        px = (n/(n-1))*(exp(-D2xx/(2*sigma2))*norma-norma);
        ll(mi) = sum(log(px));
    end
    [~, mi] = max(ll);
    sigma2 = sigma2_rot*mult(mi);
end

norma = gauss_norma(x,sigma2,trunc);
p = exp(-slmetric_pw(xe',x','sqdist')/(2*sigma2))*norma;
p(p<=eps) = min(p(p>eps));
end

