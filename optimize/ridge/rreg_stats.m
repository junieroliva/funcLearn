function rstats = rreg_stats(X,Y,varargin)
    if isempty(varargin)
        eigen_decomp = true;
    else
        eigen_decomp = varargin{1};
    end
    
    if eigen_decomp
        if size(X,1)< size(X,2)
            [U,S] = eig(X*X');
            rstats.S = diag(S);
            rstats.XtU = X'*U;
            rstats.XtY = X'*Y;
            rstats.UtXXtY = rstats.XtU'*rstats.XtY;
            rstats.isfat = true;
        else
            [U,S] = eig(X'*X);
            rstats.S = diag(S);
            rstats.U = U;
            rstats.UtXtY = U'*X'*Y;
            rstats.isfat = false;
        end
    else
        if size(X,1)< size(X,2)
            % TODO: implement
            rstats.isfat = true;
            rstats.X = X; 
            rstats.Y = Y; 
            rstats.XXt = X*X'; 
            rstats.XXtY = rstats.XXt*Y;
        else
            rstats.XtX = X'*X;
            rstats.XtY = X'*Y;
            rstats.isfat = false;
        end
    end
end