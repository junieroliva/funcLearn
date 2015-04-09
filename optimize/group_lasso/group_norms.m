function gn = group_norms(v, gsize, ginds)

if isempty(ginds)
    gn = sqrt(sum(reshape(v,gsize,[]).^2,1));
else
    gn = cumsum(v(:).^2);
    gn = sqrt(gn(ginds)-[0; gn(ginds(1:end-1))]);
end
    
end