%% MPM_L2Lp: Iterative Algorithm for \( \ell_p \)-Norm in Diagonal Form
% ============================================================================
% This function solves the following optimization problem iteratively:
%
% Objective:
% Minimize:
%   0.5 * ||w||^2 + C1 * ||w||_p^p + C2 / (1 + N1)
%
% Subject to:
%   (1/2) * (N1 * t1 + w' * Sigma1 * w / t1) <= w' * Mu1 + b
%   (1/2) * (N1 * t2 + w' * Sigma2 * w / t2) <= -w' * Mu2 - b
%   w' * Mu1 + b >= 1
%   -w' * Mu2 - b >= 1
%   N1 >= 0
%
% Key Features:
% --------------
% 1. **Regularization Terms**:
%    - \( 0.5 \|w\|^2 \): l2-norm to control the magnitude of the weights \( w \).
%    - \( C1 \|w\|_p^p \): lp-norm regularization to promote sparsity in \( w \).
%    - \( C2 / (1 + N1) \): Auxiliary term to control slack variables \( N1 \).
% 2. **Diagonal Iterative Method**:
%    - Solves the optimization problem through a three-stage iterative process:
%      a) Update weights \( w \), bias \( b \), and auxiliary variable \( k1 \) by solving a convex optimization problem using CVX.
%      b) Recalculate auxiliary terms \( t1 \) and \( t2 \).
%      c) Update regularization weights \( \Phi \) for the ℓp-norm.
%    - Iterations continue until the solution converges or a maximum number of iterations is reached.
% 3. **Robustness**:
%    - Handles feature covariance matrices \( \Sigma1 \) and \( \Sigma2 \) for positive and negative classes.
%
% Inputs:
% -------
%   - X: Data matrix (size m x n), where m = number of samples, n = features.
%   - Y: Label vector (size m x 1) with class values (+1, -1).
%   - C1, C2: Regularization parameters controlling sparsity and auxiliary terms.
%   - p: Exponent for the ℓp-norm (e.g., \( p < 1 \) promotes sparsity).
%
% Outputs:
% --------
%   - w: Weight vector (size n x 1), representing feature importance.
%   - b: Bias term for the decision boundary.
%   - k1, k2: Auxiliary variables for regularization terms.
%   - Tf: Total CPU time required for optimization.
%
% Algorithm Workflow:
% -------------------
% 1. **Initialization**:
%    - Compute class means \( \mu1, \mu2 \) and covariance matrices \( \Sigma1, \Sigma2 \).
%    - Initialize weights \( w \), bias \( b \), auxiliary variables \( t1, t2 \), and regularization weights \( \Phi \).
% 2. **Iterative Optimization**:
%    - At each iteration:
%      a) Solve a convex optimization problem using CVX.
%      b) Update auxiliary terms \( t1 \), \( t2 \) based on the updated weights \( w \).
%      c) Adjust regularization weights \( \Phi \) for the ℓp-norm.
%    - Check for convergence based on changes in \( w, b, t1, t2 \).
% 3. **Output**:
%    - Return the optimized weights \( w \), bias \( b \), and auxiliary variables \( k1, k2 \).
%
% Example Usage:
% --------------
%   [w, b, k1, k2, Tf] = MPM_L2Lp(X, Y, C1, C2, 0.5);
%   This solves the ℓp-norm optimization problem with p = 0.5 and regularization 
%   parameters C1 and C2.
%
% Notes:
% ------
% - Requires the CVX toolbox for convex optimization.
% - The algorithm is well-suited for high-dimensional data and imbalanced classes.
%
% ============================================================================

function [w,b,k1,k2,Tf]=MPM_L2Lp(X,Y,C1,C2,p)

Tinic=cputime;
epsi=1e-7;% the threshold value below which we consider an element to be zero


Min_label=min(Y);
if Min_label<0
    find1=find(Y==1);
    find2=find(Y==-1);
else
    find1=find(Y==1);
    find2=find(Y==2);
end
n=size(X,2);
x=X(find1,:);
xx=X(find2,:);
mu(1,:)=mean(x);
mu(2,:)=mean(xx);
Sigma1=cov(x);
Sigma2=cov(xx);

clear x xx

Phi=ones(n,1);
wold=100*ones(n,1);
bold=100;
t1=6;
t2=10;
Tol=10^(-3);
% See: http://ask.cvxr.com/t/minimize-log-1-1-x-where-0-x-inf/4039/11


for k=1:15

        %% 1st Step
        cvx_begin quiet
        cvx_expert true
        cvx_precision low
        variables w(n) b k1
        minimize (0.5*sum_square(w)+C1*sum(Phi.*abs(w))+C2*inv_pos(k1))
        subject to
        t1*k1+quad_form(w,Sigma1)/t1<=2*(w'*mu(1,:)'+b);
        t2*k1+ quad_form(w,Sigma2)/t2<=2*(-w'*mu(2,:)'-b);
        0<=w'*mu(1,:)'+b-1;
        0<=-w'*mu(2,:)'-b-1;
        k1>=0.0001;
        cvx_end

        %% 2nd Step %%
        t1n=sqrt(w'*Sigma1*w/k1);
        t2n=sqrt(w'*Sigma2*w/k1);
        %% 3th step: adjust the weights and re-iterate
        Phin=p./((abs(w)+epsi).^(1-p));

        Norm_t(k)=norm([t1;t2]-[t1n;t2n]);
        Norm_w(k)=norm(wold-w);
        Norm_b(k)=abs(bold-b);
        if max([Norm_t(k),Norm_w(k),Norm_b(k)])<=Tol
            break
        end

        %% 4th Step: Update
        t1=t1n;
        t2=t2n;
        wold=w;
        bold=b;

end
k2=k1;
Tf=cputime-Tinic;