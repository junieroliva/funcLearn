function C = gauss_prod_C(x, sigma2, y, tau2, trunc, varargin)
% Computes normalizing constants to pairwise functional inner products of
% exponential bumps located at points x and y respectively.
% This is primeraly a helper function for CVing bandwidths but may be of
% use in computing L2 distances of KDEs for example.
%   Inputs -
%   x: Points for first KDE (n x d)
%   sigma2: bandwidth for first KDE
%   y: Points for second KDE (m x d)
%   tau2: bandwidth for second KDE
%   trunc: boolean indicating whether to use kernels of truncated
%           normals (true) or untruncated normals (false)
%   a (optional), b (optional): [a,b]^d cube containing support [0,1]^d by
%       default

[n,d] = size(x);
m = size(y,1);

if length(varargin)>=2
    a = varargin{1};
    b = varargin{2};
else
    a = 0;
    b = 1;
end

if ~trunc
    C = (sqrt(2*pi*sigma2*tau2)/sqrt(sigma2+tau2))^d * ones(n,m);
else
    c = sqrt(2*sigma2*tau2)*sqrt(sigma2+tau2);
    pAdd = @(A,B)bsxfun(@plus,permute(B,[3,2,1]),A); % n x d x m
    C = -erf(pAdd((a-y)*sigma2/c,(a-x)*tau2/c))+erf(pAdd((b-y)*sigma2/c,(b-x)*tau2/c));
    C = (sqrt(pi*sigma2*tau2/2)/sqrt(sigma2+tau2))^d*squeeze(prod(C,2));
    
%    % with for-loops
%     C = nan(n,m);
%     for i=1:n
%         for j=1:m
%             C(i,j) = (sqrt(pi*sigma2*tau2/2)/sqrt(sigma2+tau2))^d *...
%                         prod(-erf(((a-y(j,:))*sigma2+(a-x(i,:))*tau2)/c)+erf(((b-y(j,:))*sigma2+(b-x(i,:))*tau2)/c));
%         end
%     end
end

end