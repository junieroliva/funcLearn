function [p, norma, sigma2] = kde_gauss( x, varargin )
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
CV_ISE = get_opt(opts, 'CV_ISE', true);

if isnan(sigma2) % cross-validate bandwidth 
    D2xx = slmetric_pw(x',x','sqdist');
    sigma2_rot = mean(mean(D2xx));
    mult = get_opt(opts, 'mult', 2.^(-20:10));
    scores = nan(length(mult),1);
    for mi=1:length(mult)
        sigma2 = sigma2_rot*mult(mi);
        % compute leave one out density estimates
        norma = gauss_norma(x,sigma2,trunc);
        px = (n/(n-1))*(exp(-D2xx/(2*sigma2))*norma-norma);
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
end

norma = gauss_norma(x,sigma2,trunc);
p = exp(-slmetric_pw(xe',x','sqdist')/(2*sigma2))*norma;
p(p<=eps) = min(p(p>eps));
end

