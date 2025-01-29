%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Robust Feature Selection with SVM (FSRSVM)
% Version: 22.01.25
% Authors: Miguel Carrasco, Benjamin Ivorra, Julio López, Matthieu Marechal, and Angel R. Ramos
% Related work: PREPRINT
% Requirements: CVX toolbox (https://cvxr.com/cvx/download/)
%
% DESCRIPTION:
% This script implements a robust classification framework with embedded
% feature selection, using Support Vector Machines (SVMs). It evaluates
% the performance of feature selection and classification models through
% Leave-One-Out (LOO) and Cross-Validation (CV) schemes.
%
% USER PARAMETERS:
% - model: The classification model can be changed by using
%   any available implementation in the 'Scheme' directory.
%   Ensure that the chosen model is compatible with the script and properly
%   referenced in the optimization section (e.g., 'MPM_L2Lp').
%    * SVM_Lp: Solves a nonlinear, nonconvex SVM problem with lp-norm regularization using CCCP.
%    * CoDo_Lp: Iteratively optimizes an SVM problem with lp-norm regularization and auxiliary variables (Diagonal form).
%    * MEMPM_Lp: Solves an lp-norm optimization problem using a three-stage diagonal iterative method.
%    * MPM_L2Lp: Minimizes a combined l2- and lp-norm with regularization, using a diagonal iterative algorithm.
%    * SVM_L2Lp: Uses CCCP to solve a nonlinear, nonconvex SVM problem with lp- and l2-norm regularization.
%
% - pcoef: Coefficient for the considered \( \ell_p \)-quasi-norm regularization.
%   Default: 0.25
% - CVin: Number of folds for Cross-Validation (CV).
%   Default: 10
% - numloo: Number of elements for Leave-One-Out (LOO).
%   Default: 50
% - case: Dataset name (corresponding to a .mat file in the 'Data' directory).
%   Example: 'colorectal'
% - NFredloo: List of feature selection reduction percentages for Recursive Feature 
%   Elimination (RFE) during the Leave-One-Out (LOO) step. Each value represents 
%   the proportion of features retained at each iteration.
%   Default: [1, 0.5, 0.25, 0.01]
% - NFredCV: List of feature selection reduction percentages for Recursive Feature 
%   Elimination (RFE) during the Cross-Validation (CV) step. The process 
%   iteratively reduces the number of features based on these proportions.
%   Default: [1, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01]
%
% OUTPUTS:
% - Results are stored in the 'results' folder.
% - Figures and logs are saved to the specified output directory.
% - The final solution is saved with the timestamp of the run.
%
% STEPS OF THE PROGRAM:
% 1. **Initialization**:
%    - Sets user-defined parameters.
%    - Configures output directories and logs.
%    - Loads the specified dataset.
%
% 2. **Simulation Loop**:
%    - Runs the Leave-One-Out (LOO) procedure for model evaluation.
%    - Performs feature ranking and selection iteratively.
%    - Optimizes the robust SVM model using the specified \( \ell_p \)-norm.
%
% 3. **Cross-Validation**:
%    - Evaluates the model using k-fold Cross-Validation.
%    - Calculates AUC (Area Under the Curve) for each fold and feature subset.
%
% 4. **Output Generation**:
%    - Saves results to files.
%    - Generates performance plots showing the relationship between the number
%      of features and model performance (Mean AUC).
%
% HOW TO USE:
% - Set the user parameters at the beginning of the script (pcoef, CVin, numloo, etc.).
% - Ensure the dataset file (e.g., 'colorectal.mat') is in the current directory.
% - Run the script in MATLAB. The progress and timing information will be
%   displayed in the command window and logged in a file.
%
% NOTE:
% - The CVX toolbox must be installed and accessible.
%
% CONTACT:
% For questions or support, contact macarrasco@miuandes.cl, julio.lopez@udp.cl or ivorra@ucm.es.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% ---------------------- Initial Setup ----------------------
close all; clear all; clc;
warning off; diary off;

% ---------------------- User Parameters ----------------------
model=@MPM_L2Lp;                     % Classification model
pcoef = 0.25;                                       % Coefficient for the considered norm
CVin = 10;                                          % Number of Cross-Validation Folds
numloo = 50;                                        % Number of Leave-One-Out (LOO) elements
NFredloo = [1, 0.5, 0.25, 0.01];                    % List of feature selection reduction percentages for Recursive Feature Elimination (RFE) during the Leave-One-Out (LOO) step. Each value represents the proportion of features retained at each iteration.
NFredCV  = [1, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01];   % List of feature selection reduction percentages for Recursive Feature Elimination (RFE) during the Cross-Validation (CV) step. The process iteratively reduces the number of features based on these proportions.
caso = 'colorectal';                                % Dataset name

% ---------------------- Start Processes ----------------------
% Display initial information
disp([' '])
disp(['**************'])
disp(['*   FSRSVM   *'])
disp(['*            *'])
disp(['* V 22.01.25 *'])
disp(['**************'])
disp([' '])
disp(['Feature Selection Robust SVM Model'])
disp(['Starting Simulation'])
disp([' '])

% Initialize paths
cpath = cd;                  % Save current path
tcomp = cputime;             % Start CPU timer
tic;                         % Start real-time timer
path(path,'Scheme')
path(path,'Data')
dirout = fullfile(cd, ['results/outputs-', caso, '-', num2str(pcoef)]); % output path


% Create output directory if it doesn't exist
if ~exist(dirout, 'dir')
    mkdir(dirout);
end

cd(dirout)
diary([caso, '_log.txt']);   % Start logging output to a file
cd(cpath)
clear wm bm k1m k2m Pred AUCin wmcv bmcv k1mcv k2mcv CVPred mmeanLOOAUC mmaxLOOAUC CVLOOAUCred Predictioncv LOOAUC LOOAUCred Pred Prediction

% ---------------------- Load and Preprocess Data ----------------------

disp([' '])
disp(['Problem description:'])
disp(['Considered database: ' caso])
load([caso, '.mat']);         % Load dataset
[numel, nfeat] = size(X);     % Get the number of samples and features
numloop = min(numel, numloo); % Determine the number of LOO elements

disp(['Number of indiviudals: ' num2str(numel) ' | Number of features: ' num2str(nfeat)])
disp(['Number of elements for LOO: ' num2str(numloop)])
disp(['Directory for outputs: ' dirout])
addpath .


disp([' '])

% ---------------------- Initialize Variables ----------------------

listt=[]; % List to estimate remaining time
NatV2=ceil(length(X)*NFredloo);

% Randomly select LOO samples
LLOlistpo=[];
LLOlistne=[];
for tt=1:size(Y,1)
    if Y(tt)==1
        LLOlistpo=[LLOlistpo,tt];
    else
        LLOlistne=[LLOlistne,tt];
    end
end
numepo=round(numloop*length(LLOlistpo)/size(Y,1));
numene=numloop-numepo;
inelpo=randperm(length(LLOlistpo));
nelpo=LLOlistpo(inelpo);
nelpo=nelpo(1:numepo);
inelne=randperm(length(LLOlistne));
nelne=LLOlistne(inelne);
nelne=nelne(1:numene);
LLOlist=[nelpo,nelne];
iLLOlist=randperm(length(LLOlist));
LLOlist=LLOlist(iLLOlist);

% Parameters for optimization
listcoef=[-7,-2,2,7];
listcoef2=[-7,-2,2,7];

% ---------------------- Main Simulation Loop ----------------------
for intento=1:2
    disp([' '])
    disp(['Step ' num2str(intento) '- list of coef. 1: ' num2str(listcoef) '- list of coef. 2: ' num2str(listcoef2)])
    disp([' '])

    cvtotev=CVin*length(NatV2);
    totev=length(NatV2)*length(listcoef)*length(listcoef)*numloop;
    acev=0;

    for ec1=1:length(listcoef)
        for ec2=1:length(listcoef2)
            C1=2^listcoef(ec1);
            C2=2^listcoef2(ec2);
            nu=[0.8;0.8];
            kapa=sqrt(nu./(1-nu)); % Compute the initial value of ki

            % Leave-One-Out (LOO) loop
            for k=1:numloop;
                [m,n]=size(X);
                Xa=X;
                Ya=Y;
                % Split data into training and test sets
                Xt = X(LLOlist(k), :);    % Test sample
                Yt = Y(LLOlist(k));       % Test label
                Xa = X; Xa(LLOlist(k), :) = []; % Training samples
                Ya = Y; Ya(LLOlist(k)) = [];    % Training labels

                for i=1:length(NatV2) % Feature removal
                    act=tic;
                    acev=acev+1;
                    disp(['Step Total: ' num2str(round(100*acev/totev)) ' (%) | C1: ' num2str(ec1) '/' num2str(max(length(listcoef))) ' | C2: ' num2str(ec2) '/' num2str(max(length(listcoef)))  '  | LOO: ' num2str(k) '/' num2str(numloop) ' | Nfeat Remov.: ' num2str(i) '/' num2str(length(NatV2))])

                    % Feature ranking and reduction
                    if i>1
                        [wnew,rank]=sort(abs(wm(ec1,ec2,k,(i-1),:)),'descend');
                        Xa=Xa(:,rank(1:NatV2(i)));
                        Xt=Xt(:,rank(1:NatV2(i)));
                    end

                    % Optimization model
                    [w,b,k1,k2,Tf]=model(Xa,Ya,C1,C2,pcoef); % Use the considered optimization model

                    % Store results
                    wm(ec1, ec2, k, i, 1:NatV2(i)) = w;  % Solution \( w \)
                    bm(ec1, ec2, k, i) = b;              % Solution \( b \)
                    k1m(ec1, ec2, k, i) = k1;           % Parameter \( k1 \)
                    k2m(ec1, ec2, k, i) = k2;           % Parameter \( k2 \)

                    clear wpred
                    wpred(:)=wm(ec1,ec2,k,i,1:NatV2(i));
                    Prediction(ec1,ec2,k,i)=sign(Xt*wpred'+bm(ec1,ec2,k,i));

                    if i==1
                        [wnew,rank]=sort(abs(wpred),'descend');
                        for ii=1:length(wnew)
                            list=rank(1:length(wnew)-(ii-1));
                            wred=wpred(list);
                            Xtred=Xt(:,list);
                            Pred(ec1,ec2,k,ii)=sign(Xtred*wred'+bm(ec1,ec2,k,i));
                        end


                    end
                    listt=[listt,toc(act)];
                    disp([' '])
                end

            end

            for ii=1:NatV2(1)
                clear listP
                listP(:)=Pred(ec1,ec2,:,ii);
                LOOAUCred(ec1,ec2,ii)=AUCcalc(listP',Y(LLOlist));
            end

            for i=1:length(NatV2)
                clear listP
                listP(:)=Prediction(ec1,ec2,:,i);
                LOOAUC(ec1,ec2,i)=AUCcalc(listP',Y(LLOlist));
            end
        end
    end
    disp([' '])
    disp(['Simulation ended with success'])
    disp([' '])
    disp(['Simulation CPU Time: ' num2str(cputime-tcomp)])
    disp(['Simulation Time: ' num2str(toc) ' (s)'])

    disp([' '])
    disp(['Creating Simulation Outputs'])
    disp([' '])

    cd(dirout)
    for ec1=1:length(listcoef)
        for ec2=1:length(listcoef2)
            mmeanLOOAUC(ec1,ec2)=mean(LOOAUC(ec1,ec2,:));
        end
    end
    [a,sec1]=max(max(mmeanLOOAUC'));
    [a,sec2]=max(max(mmeanLOOAUC));
    disp(['Best Coefficients: C1=' num2str(2^listcoef(sec1)) ' | C2=' num2str(2^listcoef2(sec2))])
    disp([' '])

    for ec1=1:length(listcoef)
        for ec2=1:length(listcoef2)
            mmaxLOOAUC(ec1,ec2)=max(LOOAUC(ec1,ec2,:));
        end
    end


    clear meanLOOAUC maxLOOAUC maxLOOAUCred meanLOOAUCred
    meanLOOAUC(1:length(NatV2))=[mean(mean(LOOAUC))];
    maxLOOAUC(1:length(NatV2))=[max(max(LOOAUC))];
    maxLOOAUCred(1:NatV2)=[max(max(LOOAUCred))];
    meanLOOAUCred(1:NatV2)=[mean(mean(LOOAUCred))];


    cd(cpath)
    if intento==1
        if sec1==1
            listcoef=[-7,-5,-4];
        elseif sec1==2
            listcoef=[-3,-2,-1];
        elseif sec1==3
            listcoef=[0,2,3];
        elseif sec1==4
            listcoef=[4,5,7];
        end

        if sec2==1
            listcoef2=[-7,-5,-4];
        elseif sec2==2
            listcoef2=[-3,-2,-1];
        elseif sec2==3
            listcoef2=[0,2,3];
        elseif sec2==4
            listcoef2=[4,5,7];
        end
    end
end

% ---------------------- Cross-Validation Process ----------------------
disp([' '])
disp(['Beginning Cross Validation with ' num2str(CVin) '-folds'])
disp([' '])


% Initialize timers and variables
cvtim=tic;
cvcput=cputime;
cvtlist=[];
cvacev=0;

% Define feature removal levels
NatV2=ceil(length(X)*NFredCV);
cvtotev=CVin*length(NatV2);

% Optimal coefficients based on prior simulation
C1=2^listcoef(sec1);
C2=2^listcoef2(sec2);


% Cross-validation loop
for k=1:CVin
    [m,n]=size(X);
    tst=perm2(k:CVin:m);
    trn=setdiff(1:m,tst);
    Ya=Y(trn,:);
    Xa=X(trn,1:n);
    Yt=Y(tst',:);
    Xt=X(tst',1:n);

    % Feature removal loop
    for i=1:length(NatV2);
        cvacev=cvacev+1;
        ctrt=tic;

        disp(['CV Total: ' num2str(round(100*cvacev/cvtotev)) ' (%) | Kfold: ' num2str(k) '/' num2str(max(CVin)) ' | Nfeat Remov.: ' num2str(i) '/' num2str(length(NatV2))])
        if i>1
            [wnew,rank]=sort(abs(wmcv(k,(i-1),:)),'descend');
            Xa=Xa(:,rank(1:NatV2(i)));
            Xt=Xt(:,rank(1:NatV2(i)));
        end

        % Optimization using the robust SVM model
        [w,b,k1,k2,Tf]=model(Xa,Ya,C1,C2,pcoef);

        % Store results
        wmcv(k,i,1:NatV2(i))=w;  % Solution w
        bmcv(k,i)=b;  % Solution b
        k1mcv(k,i)=k1.^2/(1+k1.^2);
        k2mcv(k,i)=k2.^2/(1+k2.^2);

        % Make predictions on test set
        clear wpred
        wpred(:)=wmcv(k,i,1:NatV2(i));
        Predictioncv(k,i,1:length(Yt))=sign(Xt*wpred'+bmcv(k,i));
        cvtlist=[cvtlist,toc(ctrt)];
        disp([' '])
        clear vPred
        vPred(:)=[Predictioncv(k,i,1:length(Yt))];
        AUCin(k,i)=AUCcalc(vPred',Yt);

        % Full model reduction analysis
        if i==1
            [wnew,rank]=sort(abs(wpred),'descend');
            for ii=1:length(wnew)
                list=rank(1:length(wnew)-(ii-1));
                wred=wpred(list);
                Xtred=Xt(:,list);
                CVPred(k,ii,1:length(Yt))=sign(Xtred*wred'+bmcv(k,i));
            end

            for ii=1:NatV2(1)
                clear listP
                listP(1:length(Yt))=CVPred(k,ii,1:length(Yt));
                CVLOOAUCred(k,ii)=AUCcalc(listP',Yt);
            end
        end


    end

end

% ---------------------- Generate Outputs ----------------------

cvmeanAUCin=mean(AUCin);

disp([' '])
disp(['Creating Cross Validation Outputs'])
disp([' '])

cd(dirout)
figure(1);clf;
hold on
plot(NatV2(1):-1:1,mean(CVLOOAUCred),':rx')
plot(NatV2,cvmeanAUCin,'--sb','linewidth',2)
legend({'DFE','RFE'},'location','southeast')
ylabel('Mean_{AUC}')
xlabel('Number of features (N_f)')
title(['N_f vs Mean_{AUC}'])
set(gca, 'XScale', 'log')
box on
grid on
axis tight
ylim([0.5,1])
saveas(gcf,'CV-Mean','fig')
saveas(gcf,'CV-Mean','eps')
saveas(gcf,'CV-Mean','jpg')
cd(cpath)

% ---------------------- Display Completion Message ----------------------

disp([' '])
disp(['Cross Validation ended with success'])
disp(['Cross Validation CPU Time: ' num2str(cputime-cvcput)])
disp(['Cross Validation Real Time: ' num2str(toc(cvtim)) ' (s)'])
disp([' '])


disp([' '])
disp(['Final CPU Time: ' num2str(cputime-tcomp)])
disp(['Final Real Time: ' num2str(toc) ' (s)'])
clear X Xt Xa
cd(dirout)
save(['Solution-' caso '-' datestr(today,'dd-mm-yy')])
cd(cpath)

disp([' '])
disp(['All processes suceed'])
disp([' '])
diary off


% AUCcalc - Calculate the Area Under the ROC Curve (AUC)
%
% Purpose:
% This function computes the Area Under the ROC Curve (AUC) for binary
% classification problems. It evaluates the performance of a classifier
% by comparing the distribution of scores for positive and negative
% samples. The AUC represents the probability that a randomly chosen
% positive instance will be ranked higher than a randomly chosen
% negative instance.
%
% Inputs:
% - X: A matrix where each column corresponds to a set of predicted scores
%      or outputs from a classifier.
% - y: A vector of true binary labels corresponding to the rows of X, where
%      positive samples are labeled as >0 and negative samples as <0.
%
% Output:
% - aucfin: A vector containing the AUC values for each column of X.
%
function aucfin = AUCcalc(X,y)
for i=1:size(X,2)
    x=X(:,i);
    posidx=find(y>0);
    negidx=find(y<0);
    [p1,p2]=size(posidx);
    [n1,n2]=size(negidx);
    posout=repmat(x(posidx),n2,n1);
    negout=repmat(x(negidx)',p1,p2);
    rocmat=2*(negout<posout);
    rocmat(negout==posout)=1;
    aucfin(i)=sum(sum(rocmat))/(2*max(n1,n2)*max(p1,p2));
end
end

