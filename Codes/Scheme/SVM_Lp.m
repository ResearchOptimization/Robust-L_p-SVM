% ============================================================
% CCCP-method for solving the following nonlinear, nonconvex SVM problem:
% Minimize:
%   ||w||_p^p + C * sum(Xi)
% Subject to:
%   D(X * w + b * e) >= e - Xi,
%   Xi >= 0
%
% This implementation uses the Concave-Convex Procedure (CCCP) to iteratively
% solve the nonconvex optimization problem by linearizing the concave terms.
%
% Objective Function:
%   - ||w||_p^p: ℓp-quasi-norm regularization term, promoting sparsity in w.
%   - C * sum(Xi): Penalty for misclassified samples using slack variables.
%
% Inputs:
%   - X: Data matrix (size m x n), where m = number of samples and n = features.
%   - Y: Label vector (size m x 1) with values in {+1, -1}.
%   - C: Regularization parameter, balancing sparsity and misclassification.
%   - C1: Dummy parameter for compatibility (not used in this function).
%   - p: Exponent for the lp-quasi-norm (e.g., p < 1 promotes sparsity).
%
% Outputs:
%   - w: Weight vector (size n x 1), representing feature importance.
%   - b: Bias term for the decision boundary.
%   - k1, k2: Dummy outputs for compatibility (set to 0 in this implementation).
%   - Tf: Total CPU time required for optimization.
%
% Algorithm:
% 1. Initialize variables (w, b, Xi, v).
% 2. Iteratively solve a linear programming (LP) problem using the CCCP:
%    - Compute the gradient of the concave term (lp-quasi-norm).
%    - Solve the resulting convex problem with `linprog`.
%    - Update variables until convergence (error < tolerance or max iterations).
%
% Constraints:
%   - Linear constraints are formulated to enforce the SVM margin with slack.
%
% Key Features:
%   - lp-Norm Regularization: Promotes sparsity in w, enhancing model robustness
%     and interpretability by selecting the most relevant features.
%   - Soft-Margin SVM: Handles non-linearly separable data with slack variables.
%   - CCCP Optimization: Efficiently handles nonconvexity through iterative 
%     linearization.
%
% Example:
%   [w, b, ~, ~, Tf] = SVM_Lp(X, Y, 0, 1, 0.5);
%   This solves the SVM problem with ℓp-quasi-norm (p=0.5) and C=1.
%
% Author: [Your Name]
% ============================================================

function [w,b,k1,k2,Tf]=SVM_Lp(X,Y,C1,C,p)

k1=0;
k2=0;

find1=find(Y==1);
find2=find(Y==-1);
[m,n]=size(X);
e=ones(m,1);

%% Tolerance
Error = 10;
Tol=1e-5;

%% Linear constraints
A=[-Y.*X, -Y, -eye(m), zeros(m,n);zeros(m,n+1), -eye(m), zeros(m,n);...
    -eye(n), zeros(n,m+1), -eye(n);eye(n), zeros(n,m+1), -eye(n)];

bi=[-e; zeros(m+2*n,1)];

%% Initial point of CCCP method
w0=zeros(n,1);
b0=1;
v0=ones(n,1);
Xi0=zeros(m,1);

iter=0;
Tinic=cputime;
while Error > Tol && iter<200
    iter=iter+1;
    gradG=p*(v0+1e-6).^(p-1); 
    f=[zeros(n+1,1); C*e; gradG];

    xs=linprog(f,A,bi);
    w=xs(1:n);
    b=xs(n+1);
    Xi=xs(n+2:n+1+m);
    v=xs(n+m+2:end);
    Error=abs(gradG'*(v-v0)+C*sum(Xi-Xi0));
    w0=w;
    v0=v;
    b0=b;
    Xi0=Xi;
end
Tf=cputime-Tinic;
Sol.w=w;
Sol.b=b;

