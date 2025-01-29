%% SVM_L2Lp: CCCP Method for Nonlinear Nonconvex SVM Problem
% ============================================================================
% This function solves the following nonlinear, nonconvex SVM optimization problem:
%
% Objective:
% Minimize:
%   0.5 * ||w||^2 + C1 * ||w||_p^p + C2 * (sum(Xi1) + sum(Xi2))
%
% Subject to:
%   A * w + b * e >= e - Xi1,
%   -B * w - b * e >= e - Xi2,
%   Xi1, Xi2 >= 0.
%
% Key Features:
% --------------
% 1. **Regularization Terms**:
%    - \( 0.5 ||w||^2 \): l2-norm to control the magnitude of the weight vector \( w \).
%    - \( C1 ||w||_p^p \): lp-norm regularization to promote sparsity in \( w \).
%    - \( C2 * (sum(Xi1) + sum(Xi2)) \): Penalty for misclassified samples using slack variables \( \Xi1 \) and \( \Xi2 \).
% 2. **Concave-Convex Procedure (CCCP)**:
%    - Decomposes the nonconvex objective into concave and convex parts.
%    - Iteratively solves a convex optimization problem until convergence.
% 3. **Soft-Margin SVM**:
%    - Incorporates slack variables \( \Xi1 \) and \( \Xi2 \) for soft margin classification.
%
% Inputs:
% -------
%   - X: Data matrix (size m x n), where m = number of samples, n = features.
%   - Y: Label vector (size m x 1) with values in {+1, -1}.
%   - C1: Regularization parameter for the ℓp-norm term.
%   - C2: Regularization parameter for slack variables \( \Xi1, \Xi2 \).
%   - p: Exponent for the lp-norm (e.g., \( p < 1 \) promotes sparsity).
%
% Outputs:
% --------
%   - w: Weight vector (size n x 1), representing feature importance.
%   - b: Bias term for the decision boundary.
%   - k1, k2: Auxiliary variables for compatibility (not used in this implementation).
%   - Tf: Total CPU time required for optimization.
%
% Algorithm Workflow:
% -------------------
% 1. **Initialization**:
%    - Extract positive (\( A \)) and negative (\( B \)) class samples from \( X \).
%    - Initialize weights \( w \), bias \( b \), auxiliary variables \( v \), and slack variables \( \Xi1, \Xi2 \).
% 2. **Iterative CCCP Optimization**:
%    - Compute the gradient of the concave part \( \nabla G(v) = C1 * p * v^{p-1} \).
%    - Solve the convex optimization problem using the `cvx` toolbox.
%    - Update \( w, b, v, \Xi1, \Xi2 \) based on the solution.
%    - Repeat until the solution converges or the maximum number of iterations is reached.
% 3. **Output**:
%    - Return the optimized weights \( w \), bias \( b \), and total CPU time \( Tf \).
%
% Example Usage:
% --------------
%   [w, b, ~, ~, Tf] = SVM_L2Lp(X, Y, 1, 0.5, 0.8);
%   This solves the nonlinear SVM problem with C1 = 1, C2 = 0.5, and p = 0.8.
%
% Notes:
% ------
% - Requires the CVX toolbox for convex optimization (e.g., `sedumi` solver).
% - Suitable for high-dimensional datasets and imbalanced classes.
%
% ============================================================================



function [w,b,k1,k2,Tf]=SVM_L2Lp(X,Y,C1,C2,p)

k1=0;
k2=0;

find1=find(Y==1);
find2=find(Y==-1);
A=X(find1,:); 
B=X(find2,:); 
[m,n]=size(X);
m1=length(find1);
m2=length(find2);
e1=ones(m1,1);
e2=ones(m2,1);


%% Tolerance
Error = 10;
Epsilon=1e-5;

%% Initial point of CCCP method
w0=zeros(n,1);
b0=1;
v0=ones(n,1);
Xi0=zeros(m,1);

iter=0;
Tinic=cputime;
while Error > Epsilon && iter<150
    iter=iter+1;
    gradG=C1*p*(v0+1e-7).^(p-1); 

    cvx_begin quiet
    cvx_solver sedumi

    variables w(n) b Xi1(m1) Xi2(m2) v(n)
    minimize (0.5*sum_square(w)+C2*(sum(Xi1)+sum(Xi2))+v'*gradG)
    subject to
    A*w+b*e1>=e1-Xi1,
    -B*w-b*e2>=e2-Xi2,
    w<=v;
    -v<=w;
    Xi1>=0;
    Xi2>=0;
    cvx_end

    Xi=[Xi1;Xi2];
    Error= abs(gradG'*(v-v0)+0.5*norm(w-w0)^2+C2*sum(Xi-Xi0));
    w0=w;
    v0=v;
    b0=b;
    Xi0=Xi;
end
Tf=cputime-Tinic;
Sol.w=w;
Sol.b=b;
