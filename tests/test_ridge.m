tol = sqrt(eps);

n = randi(1000);
d = randi(1000);
p = randi(10);

lambda = abs(randn)*10;
Y = randn(n,p);
X = randn(n,d);
design = rreg_stats(X,Y);

beta_simple = (X'*X+lambda*eye(d))\(X'*Y);
beta_code = rreg_beta(design, lambda);
same_beta = max(abs(beta_simple(:)-beta_code(:)))<tol;

SM_simple = X*((X'*X+lambda*eye(d))\X');
SM_code = rreg_SM(X, design, lambda);
same_SM = max(abs(SM_simple(:)-SM_code(:)))<tol;

SMii_simple = diag(SM_simple);
SMii_code = rreg_SMii(X, design, lambda);
same_SMii = max(abs(SMii_simple(:)-SMii_code(:)))<tol;

i = randi(n);
trn_set = true(n,1);
trn_set(i) = false;
beta_trn_simple = (X(trn_set,:)'*X(trn_set,:)+lambda*eye(d))\(X(trn_set,:)'*Y(trn_set,:));
ressid_simple = Y(i,:)-X(i,:)*beta_trn_simple;
ressid_code = (Y(i,:)-X(i,:)*beta_code)./(1-SMii_code(i));
same_ressid = max(abs(ressid_simple(:)-ressid_code(:)))<tol;