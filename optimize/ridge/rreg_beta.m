function beta = rreg_beta(design, lambda)
if isfield(design,'S')
    if design.isfat
        [S, XtU, XtY, UtXXtY] = deal(design.S, design.XtU, design.XtY, design.UtXXtY);       
        beta = (1/lambda)*(XtY-XtU*bsxfun(@times,UtXXtY,1./(S+lambda)));
    else
        [U, S, UtXtY] = deal(design.U, design.S, design.UtXtY);
        beta = U*bsxfun(@times,UtXtY,1./(S+lambda));
    end
else
    if design.isfat
        % TODO: implement
        [X,Y,XXt,XXtY] = deal(design.X, design.Y, design.XXt, design.XXtY);
        beta = (1/lambda)*( X'* (Y-(XXt+lambda*eye(length(XXt)))\XXtY) );
    else
        [XtX, XtY] = deal(design.XtX, design.XtY);
        beta = (XtX+lambda*eye(length(XtX))) \ XtY;
    end
end
end