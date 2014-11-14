function beta = rreg_beta(design, lambda)
    if design.isfat
        [S, XtU, XtY, UtXXtY] = deal(design.S, design.XtU, design.XtY, design.UtXXtY);       
        beta = (1/lambda)*(XtY-XtU*bsxfun(@times,UtXXtY,1./(S+lambda)));
    else
        [U, S, UtXtY] = deal(design.U, design.S, design.UtXtY);
        beta = U*bsxfun(@times,UtXtY,1./(S+lambda));
    end
end