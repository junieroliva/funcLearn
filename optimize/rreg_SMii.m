function SMii = rreg_SMii(X, design, lambda)
if isfield(design,'S')
    if design.isfat
        [S, XtU] = deal(design.S, design.XtU);   
        UtXXt = XtU'*X';
        XtXinvXt = (1/lambda)*(X'-XtU*bsxfun(@times,UtXXt,1./(S+lambda)));
    else
        [U, S] = deal(design.U, design.S);
        UtXt = U'*X';
        XtXinvXt = U*bsxfun(@times,UtXt,1./(S+lambda));
    end
else
    if design.isfat
        % TODO: implement
    else
        XtX = design.XtX;
        XtXinvXt = (XtX+lambda*eye(length(XtX))) \ X';
    end
end
    SMii = sum(X.*XtXinvXt',2);
end