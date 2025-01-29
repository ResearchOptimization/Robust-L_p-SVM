# Robust-L_p-SVM
This study presents a robust classification framework with embedded feature selection to tackle challenges in high-dimensional datasets.  The proposed methods are validated on benchmark datasets using four classification models and two feature elimination techniques: Direct Feature Elimination and Recursive Feature Elimination.

Robust Feature Selection with SVM (FSRSVM)
===========================================

Overview
--------
This project implements a robust classification framework with embedded 
feature selection using Support Vector Machines (SVMs). The model integrates 
optimization techniques to perform feature selection and classification, 
evaluated through Leave-One-Out (LOO) and Cross-Validation (CV) schemes.

Structure
---------
Main.m                - Main script to execute the program
Scheme/               - Folder containing classification and optimization models
    SVM_Lp.m
    CoDo_Lp_Diagonal.m
    MEMPM_Lp_Diagonal.m
    MPM_L2Lp.m
    SVM_L2Lp.m
Data/                 - Folder containing datasets in .mat format
    colorectal.mat
    lymphoma_XY.mat
    gravier.mat
    west.mat
    shipp.mat
    pomeroy.mat
results/              - Folder where simulation outputs will be saved

Main Components
---------------
1. Main Script (Main.m): 
   Entry point of the program. Users set parameters and execute the simulation from this script.
2. Models (Scheme/): 
   Contains optimization functions implementing various SVM-based classification techniques.
   Default model: `itera_MPM_L2Lp_Diagonal`.
3. Datasets (Data/): 
   Contains .mat files with sample datasets used for classification and feature selection.
4. Results (results/): 
   Stores outputs, such as logs, plots, and performance metrics, organized by simulation run.

How to Use
----------
Step 1: Set User Parameters
    Modify the `Main.m` script to configure the parameters for your simulation. Key parameters include:
    - model: Classification model from the `Scheme` directory (default: MPM_L2Lp).
    - pcoef: Coefficient for the \( \ell_p \)-quasi-norm regularization (default: 0.25).
    - CVin: Number of folds for Cross-Validation (default: 10).
    - numloo: Number of elements for Leave-One-Out (default: 50).
    - caso: Dataset name in the `Data` folder (e.g., 'colorectal').
    - NFredloo: Percentages of feature reduction during LOO (default: [1, 0.5, 0.25, 0.01]).
    - NFredCV: Percentages of feature reduction during CV (default: [1, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01]).

Step 2: Run the Script
    Run `Main.m` in MATLAB. The script will:
    1. Load the specified dataset from the `Data` folder.
    2. Perform feature selection and classification using the configured SVM model.
    3. Evaluate the model with LOO and CV schemes.
    4. Save results (metrics, plots, and logs) in the `results` folder.

Step 3: Check Outputs
    Outputs are stored in a subdirectory inside `results`, named according to the dataset and parameters used. Outputs include:
    - Logs: Progress and results of the simulation.
    - Plots: Performance metrics, such as Mean AUC vs. Number of Features.
    - Saved Models: Parameters and performance metrics for the best models.

Available Methods in the Scheme Directory
------------------------------------------

The following methods are included in the `Scheme` directory, each implementing a specific SVM optimization approach:

- SVM_Lp  : Solves a nonlinear, nonconvex SVM problem with \( \ell_p \)-norm regularization using the Concave-Convex Procedure (CCCP).
- CoDo_Lp : Iteratively optimizes an SVM problem with \( \ell_p \)-norm regularization and auxiliary variables in diagonal form.
- MEMPM_Lp: Solves an \( \ell_p \)-norm optimization problem using a three-stage diagonal iterative method.
- MPM_L2Lp: Minimizes a combined \( \ell_2 \)- and \( \ell_p \)-norm with regularization, using a diagonal iterative algorithm.
- SVM_L2Lp: Uses CCCP to solve a nonlinear, nonconvex SVM problem with both \( \ell_p \)- and \( \ell_2 \)-norm regularization.

Each method is tailored to handle specific optimization challenges, such as sparsity, feature selection, and robustness in high-dimensional datasets.


Requirements
------------
- MATLAB (Tested with R2021b and later)
- CVX toolbox for convex optimization (Download: https://cvxr.com/cvx/download/)

Examples
--------
Example 1: Running with the 'colorectal.mat' dataset
    1. Set `caso = 'colorectal';` in `Main.m`.
    2. Run the script in MATLAB.
    3. Check the results in the `results` folder under `outputs-colorectal-0.25`.

Example 2: Modifying the Model
    1. Set `model = @itera_CoDo_Lp_Diagonal;` in `Main.m`.
    2. Run the script.

Contact
-------
For questions or support, contact:
- Miguel Carrasco: macarrasco@miuandes.cl
- Julio López: julio.lopez@udp.cl
- Benjamin Ivorra: ivorra@ucm.es
