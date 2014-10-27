function inds = outerprodinds( T, d )
listT = (1:T)';
inds = listT;
for i=2:d
    inds = [repmat(inds, T, 1), kron(listT,ones(size(inds,1),1))];
end


end

