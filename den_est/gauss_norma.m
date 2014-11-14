function norma = gauss_norma(x, sigma2, trunc, varargin)
% Computes normalizing constants to exponential bumps located at points x
% for KDE
%   Inputs -
%   x: Points KDE is based on (n x d)
%   sigma2: bandwidth for KDE
%   trunc: boolean indicating whether to use kernels of truncated
%           normals (true) or untruncated normals (false)
%   a (optional), b (optional): [a,b]^d cube containing support [0,1]^d by
%       default

[n,d] = size(x);

if length(varargin)>=2
    a = varargin{1};
    b = varargin{2};
else
    a = 0;
    b = 1;
end

if trunc
    norma = 1./((2*pi*sigma2)^(d/2)*prod(normcdf((b-x)/sqrt(sigma2))-normcdf((a-x)/sqrt(sigma2)),2)*n);
else
    norma = 1./((2*pi*sigma2)^(d/2)*ones(n,1)*n);
end

end