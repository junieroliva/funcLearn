function SM = rreg_SM(X, design, lambda)
if isfield(design,'S')
    if design.isfat
        [S, XtU] = deal(design.S, design.XtU);   
        UtXXt = XtU'*X';
        SM = (1/lambda)*(X*X'-X*XtU*bsxfun(@times,UtXXt,1./(S+lambda)));
    else
        [U, S] = deal(design.U, design.S);
        UtXt = U'*X';
        SM = X*U*bsxfun(@times,UtXt,1./(S+lambda));
    end
else
    if design.isfat
        % TODO: implement
    else
        XtX = design.XtX;
        SM = X*((XtX+lambda*eye(length(XtX))) \ X');
    end
end
end