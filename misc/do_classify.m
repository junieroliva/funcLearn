function c = do_classify( Y )
% Are responses in {0,1}?
uv = unique(Y(:));
c = (length(uv)==2 && uv(1)==0 && uv(2)==1) || (length(uv)==1 && (uv(1)==0 || uv(1)==1));
end

