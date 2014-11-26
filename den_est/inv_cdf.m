function [Finv] = inv_cdf(x,cdfv,xgrid)
d = size(x,2);
Finv = nan(size(x));
for i=1:d
    cdf = cdfv(:,i);
    [~,IA] = unique(cdf);
    Finv(:,d) = interp1(cdf(IA), xgrid(IA), x(:,d), 'linear', 0);
end
end
