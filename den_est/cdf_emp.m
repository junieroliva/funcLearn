function [ cdfv ] = cdf_emp( x, varargin )
%CDF_EMP Summary of this function goes here
%   Detailed explanation goes here
if ~isempty(varargin)
    opts = varargin{1};
else
    opts = struct;
end

n = size(x,1);

L = get_opt(opts,'L',1000);
xgrid = get_opt(opts,'xgrid'); 
if isempty(xgrid)
    xgrid = 1/L:1/L:1;
end

cdfv = bsxfun(@le, permute(x,[3 1 2]), xgrid');
cdfv = squeeze(mean(cdfv,2));

winz = get_opt(opts,'winz',1/( 4*n^(.25)*sqrt(pi*log(n)) ));
cdfv(cdfv<winz) = winz;
cdfv(cdfv>1-winz) = 1-winz;

end

