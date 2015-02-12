function samp = sample_kde( x, ne, varargin )
%sample_kde 	Sample ne points from a kde on points x
%   Detailed explanation goes here

if isempty(varargin)
    opts = struct;
else
    opts = varargin{1};
end
sigma2 = get_opt(opts, 'sigma2', nan);
if isnan(sigma2)
    sigma2 = mean(mean(slmetric_pw(x',x','sqdist')));
end

a = get_opt(opts, 'a', -inf);
b = get_opt(opts, 'b', inf);
[n,d] = size(x);
samp = x(randi(n,ne,1),:) + sqrt(sigma2)*randn(ne,d);
oob = max(samp,[],2)>b | min(samp,[],2)<a;
while any(oob)
    noob = sum(oob);
    samp(oob,:) = x(randi(n,noob,1),:)+sqrt(sigma2)*randn(noob,d);
    oob = max(samp,[],2)>1 | min(samp,[],2)<0;
end

end

