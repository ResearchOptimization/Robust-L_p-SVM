%% MEMPM_Lp: Iterative Algorithm for \( \ell_p \)-Norm in Diagonal Form
% ============================================================================
% This function solves the following optimization problem iteratively:
%
% Objective:
% Minimize:
%   |w|_p^p + C1 / (1 + N1) + C2 / (1 + N2)
%
% Subject to:
%   (1/2) * (N1 * t1 + w' * Sigma1 * w / t1) <= w' * Mu1 + b
%   (1/2) * (N2 * t2 + w' * Sigma2 * w / t2) <= -w' * Mu2 - b
%   w' * Mu1 + b >= 1
%   -w' * Mu2 - b >= 1
%   N1, N2 >= 0
%
% Key Features:
% 1. **lp-Norm Regularization**:
%    - Promotes sparsity in the weight vector \( w \) to enhance interpretability.
% 2. **Diagonal Iterative Method**:
%    - Alternates between three main stages:
%      a) Solve a convex optimization problem using CVX to update \( w \), \( b \), \( k1 \), and \( k2 \).
%      b) Adjust auxiliary terms \( t1 \) and \( t2 \) based on the updated variables.
%      c) Update regularization weights for the ℓp-norm, \( \Phi \).
%    - The process repeats until convergence (tolerance threshold) or a maximum number of iterations is reached.
% 3. **Robustness**:
%    - Handles feature covariance matrices \( \Sigma1 \) and \( \Sigma2 \) for positive and negative classes, respectively.
%
% Inputs:
% -------
%   - X: Data matrix (size m x n), where m = number of samples and n = number of features.
%   - Y: Label vector (size m x 1) with values in {+1, -1}.
%   - C1, C2: Regularization parameters controlling the influence of auxiliary terms \( N1, N2 \).
%   - p: Exponent for the ℓp-norm (e.g., \( p < 1 \) promotes sparsity).
%
% Outputs:
% --------
%   - w: Weight vector (size n x 1), representing feature importance.
%   - b: Bias term for the decision boundary.
%   - k1, k2: Auxiliary variables used for regularization terms.
%   - Tf: Total CPU time required for optimization.
%
% Algorithm Workflow:
% -------------------
% 1. **Initialization**:
%    - Compute means \( \mu1, \mu2 \) and covariance matrices \( \Sigma1, \Sigma2 \) for 
%      the positive and negative classes, respectively.
%    - Set initial values for \( w, b, t1, t2 \), and regularization weights \( \Phi \).
% 2. **Iterative Optimization**:
%    - At each iteration:
%      a) Solve a convex optimization problem using the CVX toolbox.
%      b) Update auxiliary terms \( t1, t2 \) based on the current weights \( w \).
%      c) Adjust the regularization weights \( \Phi \) based on \( w \).
%    - Check for convergence based on changes in \( w, b, t1, t2 \).
% 3. **Output**:
%    - Return the optimized weights \( w \), bias \( b \), and auxiliary terms \( k1, k2 \).
%
% Example Usage:
% --------------
%   [w, b, k1, k2, Tf] = itera_MEMPM_Lp_Diagonal(X, Y, C1, C2, 0.5);
%   This solves the ℓp-norm optimization problem with p = 0.5 and the provided 
%   regularization parameters C1 and C2.
%
% Notes:
% ------
% - Requires the CVX toolbox for convex optimization.
% - The algorithm is robust for datasets with high dimensionality and imbalanced classes.
%
% ============================================================================


function [w,b,k1,k2,Tf]=MEMPM_Lp(X,Y,C1,C2,p)

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

        %% 1er Paso
        cvx_begin quiet
        cvx_expert true
        cvx_precision low
        variables w(n)  b k1 k2
        minimize (sum(Phi.*abs(w))+C1*inv_pos(k1) +C2*inv_pos(k2))
        subject to
        t1*k1+quad_form(w,Sigma1)/t1<=2*(w'*mu(1,:)'+b);
        t2*k2+ quad_form(w,Sigma2)/t2<=2*(-w'*mu(2,:)'-b);
        0<=w'*mu(1,:)'+b-1;
        0<=-w'*mu(2,:)'-b-1;
        k1>=0.0001;
        k2>=0.0001;
        cvx_end

        %% 2nd Step %%
        t1n=sqrt(w'*Sigma1*w/k1);
        t2n=sqrt(w'*Sigma2*w/k2);
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
Tf=cputime-Tinic;