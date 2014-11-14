function [ figh ] = view_density( X, p, varargin )
if ~isempty(varargin)
    opts = varargin{1};
else
    opts = struct;
end

cscale = get_opt(opts, 'cscale', max(p));
figh = get_opt(opts, 'figh');
if isempty(figh)
    figh = figure;
end
layout = get_opt(opts, 'layout', [5 5]);

if length(X)==1
    plot(X{1},p);
elseif length(X)==2
    contourf(X{1}, X{2}, reshape(p,size(X{1})), 100, 'LineStyle','none');
    caxis([0 cscale]);
else
    z = squeeze(X{3}(1,1,:));
    nplots = prod(layout);
    %zstep = round((1:nplots)*length(z)/nplots);
    zstep = round(linspace(1,length(z),nplots));
    pslices = reshape(p,size(X{1}));
    for i = 1:nplots
        subplot(layout(1),layout(2),i);
        zi = zstep(i);
        contourf(squeeze(X{1}(:,:,zi)),squeeze(X{2}(:,:,zi)), squeeze(pslices(:,:,zi)), 100, 'LineStyle','none');
        caxis([0 cscale]);
        title(sprintf('z = %.3f',z(zi)));
    end
end

end