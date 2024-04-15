% Clear workspace and command window
clear all;
clc;
rng(42); % Ensure reproducibility

%% ================== Variable and Path Declarations ====================
wd='/Users/e410377/Desktop/Ludo/KOApred';
dataFilePath = [wd '/FINALWOMAC.xlsx'];
utilityPath = [wd '/utility'];
addpath(genpath(utilityPath));

threshold = 0.21;

%% ======================== Data Preprocessing ==========================
% Import and preprocess data
[data, titles] = importExcelData(dataFilePath);
variables = extractVariables(data, titles);
variables = extractROIs(variables);
variables = computeMetrics(variables);

ALL_DATA = normalizeData([variables.ratio_Pre_TKA_WOMAC_pain_, variables.Pre_TKA_WOMAC_pain_, ...
    variables.Genotype0, variables.Genotype1, variables.Sex0, variables.Sex1, variables.ROIs]);

pNamesBase = {'WpainImpr', 'PreWpain', 'HAB', 'MAB', 'F', 'M'};
variables.modelname = 'PrePain + Genotype + Sex + ROIs';
variables.targetname = pNamesBase{1};

roiFeatureNames = variables.ROIstitles;
allpNames = [pNamesBase, roiFeatureNames];

%% ==================== Data Preparation =======================
[target, input, featureNames] = prepareData(ALL_DATA, allpNames);

%% =================== Leave-One-Out Cross-Validation ===================
[selectedFeaturesSTORE, predictedTarget, STORE, WEIGHTS] = leaveOneOutCV(input, target, featureNames, 'SVM', threshold);

%% ======================== Plotting Results ==========================
plotResults(variables, target, predictedTarget, STORE, WEIGHTS, featureNames);

%% ================== Function Definitions ===================

function [data, titles] = importExcelData(filePath)
    % Import data from Excel file
    importedData = importdata(filePath);
    
    % Extract numeric data and titles
    data = importedData.data;
    titles = importedData.textdata(1, 2:end);
end


function variables = extractVariables(data, titles)
    variables.numericData = data;
    variables.titles = titles;
    % Extract specific variables
    extract = @(name) data(:, strcmp(titles, name));
    varNames = {'Age (pre)', 'Genotype (1=GG)', 'Sex (1=F)', 'TKA pain pre', 'TKA pain post', ...
                'Pre-TKA,WOMAC (pain)', 'Pre-TKA,WOMAC (physical function)', 'Pre-TKA,WOMAC (stiffness)', ...
                '1yr POST-TKA, Womac (pain)', '1yr POST-TKA,Womac (phys func)', '1yr POST-TKA,Womac (stiffness)', ...
                'PRE-TKA, Promis (pain intensity)'};
    for varName = varNames
        cleanedName = matlab.lang.makeValidName(varName{1});
        variables.(cleanedName) = extract(varName{1});
    end
    % Assuming genotype is your original arrays
    variables.Genotype0 = variables.Genotype_1_GG_ == 1; % This will be 0 when genotype is GG
    variables.Genotype1 = variables.Genotype_1_GG_ == 2; % This will be 1 when genotype is GA
    
    
    % Assuming genotype is your original array
    variables.Sex0 = variables.Sex_1_F_ == 1; % This will be 0 when sex is F
    variables.Sex1 = variables.Sex_1_F_ == 2; % This will be 1 when sex is M
end


function variables = computeMetrics(variables)
    eps = 1e-10; % For numerical stability
    computeRatio = @(pre, post) (pre - post) ./ (pre + post + eps);
    
    % Define variable pairs for ratio computation
    pairs = {'TKAPainPre', 'TKAPainPost'; ...
             'Pre_TKA_WOMAC_pain_', 'x1yrPOST_TKA_Womac_pain_'; ...
             'Pre_TKA_WOMAC_physicalFunction_', 'x1yrPOST_TKA_Womac_physFunc_'; ...
             'Pre_TKA_WOMAC_stiffness_', 'x1yrPOST_TKA_Womac_stiffness_'};
    
    for i = 1:size(pairs, 1)
        preVar = variables.(pairs{i, 1});
        postVar = variables.(pairs{i, 2});
        ratioName = ['ratio_' pairs{i, 1}];
        variables.(ratioName) = computeRatio(preVar, postVar);
    end
end


function variables = extractROIs(variables)
    % Define pattern to match ROI titles
    pattern = '^(SC|CC|)';
    
    % Find indices of titles matching the pattern
    roiIndices = find(~cellfun('isempty', regexp(variables.titles, pattern)));
    
    % Extract ROIs and their titles
    variables.ROIs = variables.numericData(:, roiIndices);
    variables.ROIstitles = variables.titles(:, roiIndices);
end


function out = normalizeData(input)
    % Get the size of the input data
    [rows, cols] = size(input);
    
    % Initialize output matrix
    out = zeros(rows, cols);
    
    % Normalize each column separately
    for col = 1:cols
        colData = input(:, col);
        out(:, col) = (colData - min(colData)) ./ (max(colData) - min(colData));
    end
end


function [target, input, featureNames] = prepareData(ALL_DATA, titles)
    % Assuming the first column is the target and the rest are features
    target = ALL_DATA(:, 1);
    input = ALL_DATA(:, 2:end);
    
    % Adjust featureNames based on your actual data structure
    featureNames = titles(2:end);
end


function [selectedFeaturesSTORE, predictedTarget, STORE, WEIGHTS] = leaveOneOutCV(input, target, featureNames, method, threshold)
    numFolds = size(input, 1);
    STORE = zeros(length(featureNames), numFolds); % Store selected features for each fold
    WEIGHTS = zeros(length(featureNames), numFolds); % Store selected features for each fold

    predictedTarget = zeros(size(target));
    numSelectedFeatures = round(threshold * size(input, 2)); % Adjust as needed

    for fold = 1:numFolds
        fprintf('Processing fold %d/%d...\n', fold, numFolds);
        testIndex = fold;
        trainIndex = setdiff(1:numFolds, testIndex);

        % Split data into training and testing sets
        inputTrain = input(trainIndex, :);
        targetTrain = target(trainIndex);
        inputTest = input(testIndex, :);

        % Perform feature selection and model training
        [selectedFeatures, ~, coeff, ~] = performFeatureSelection(inputTrain, targetTrain, method, numSelectedFeatures);

        % Perform predictions based on selected features
        if strcmp(method, 'SVM')
            selectedInputTrain = inputTrain(:, selectedFeatures == 1);
            selectedInputTest = inputTest(:, selectedFeatures == 1);
            combinedModel = fitrsvm(selectedInputTrain, targetTrain, 'KernelFunction', 'linear', 'Standardize', true);
            predictedTarget(testIndex) = predict(combinedModel, selectedInputTest);
            WEIGHTS(:, fold) = coeff;
            STORE(:, fold) = selectedFeatures;
        end
    end

    selectedFeaturesSTORE = sum(STORE, 2); % Sum selected features across folds

     % plot one example of training
     figure(1121)
     [rho2, p2] = PlotSimpleCorrelationWithRegression(targetTrain,  predict(combinedModel, selectedInputTrain), 30, 'b');
     title({sprintf("TRAINING Rho: %.2f; p: %.2f", rho2, p2)},'FontSize',18);
     ylabel('Pred. Improvement','FontSize',18);
     xlabel('True Improvement','FontSize',18);
     hold off
end

function [selectedFeatures, model, coeff, intercept] = performFeatureSelection(inputTrain, targetTrain, method, numSelectedFeatures)
    if strcmp(method, 'SVM')
        % Placeholder for SVM feature selection, adjust as needed
        % This example uses all features and fits an SVM, real feature selection for SVM might require a different approach
        model = fitrsvm(inputTrain, targetTrain, 'KernelFunction', 'linear', 'Standardize', true);
        featureWeights = abs(model.Beta);
        [~, featureIdx] = sort(featureWeights, 'descend');
        selectedFeaturesIdx = featureIdx(1:numSelectedFeatures);
        selectedFeatures = zeros(size(inputTrain, 2), 1);
        selectedFeatures(selectedFeaturesIdx) = 1;
        coeff = model.Beta; % Coefficients (weights) of the SVM model
        intercept = [];
    else
        error('Unsupported method.');
    end
end


function plotResults(variables, target, predictedTarget, STORE, WEIGHTS, featureNames)
    % Plotting correlation between true and predicted values
    figure(1)
    subplot(2,3,2)
    [rho2, p2] = PlotSimpleCorrelationWithRegression(target, predictedTarget, 30, 'b');
    title({sprintf('Model: %s', variables.modelname), sprintf("Rho: %.2f; p: %.2f", rho2, p2)}, 'FontSize', 18);
    ylabel('Pred. Improvement', 'FontSize', 18);
    xlabel('True Improvement', 'FontSize', 18);
    xlim([0, 1.3])
    ylim([0, 1.3])
    hold off

    % Sort feature frequencies and weights
    [sortedFreq, sortedTitleStrings, sortedWeights] = sortfreq(STORE, WEIGHTS, featureNames);

    % Normalize weights for transparency
    maxWeight = max(abs(sortedWeights));
    normalizedIntensity = abs(sortedWeights) / maxWeight;

    % Plot histogram of feature frequencies with colored bars
    subplot(2,1,2);
    for i = 1:length(sortedFreq)
        barColor = [1, 0, 0]; % Default to red
        if sortedWeights(i) < 0
            barColor = [0, 0, 1]; % Adjust to blue for negative weights
        end
        bar(i, sortedFreq(i), 'FaceColor', barColor, 'EdgeColor', 'none', 'FaceAlpha', normalizedIntensity(i));
        hold on;
    end
    hold off;

    % Set x-axis labels to sorted feature names
    xticks(1:length(sortedFreq));
    xticklabels(sortedTitleStrings);
    xtickangle(45);

    % Set axis labels and title
    xlabel('Features');
    ylabel('Frequency');
    set(gcf, 'Color', 'w');
    set(gca, 'FontSize', 15);
    grid on;
end


function [sortedFreq, sortedTitleStrings, sortedWeights] = sortfreq(STORE, WEIGHTS, featureNames)
    % Sum the frequencies across folds for selected features
    freq = sum(STORE, 2);

    % Normalize and sum weights across folds
    normalizedWeights = normalizeWeights(WEIGHTS);
    summedNormalizedWeights = sum(normalizedWeights, 2);

    % Combine frequencies, feature names, and weights
    combinedData = [num2cell(freq), featureNames', num2cell(summedNormalizedWeights)];

    % Sort based on frequency in descending order
    sortedData = sortrows(combinedData, -1);

    % Extract sorted frequencies, feature names, and weights
    sortedFreq = cell2mat(sortedData(:, 1));
    sortedWeights = cell2mat(sortedData(:, 3));
    sortedTitle = sortedData(:, 2);
    sortedTitleStrings = cellfun(@str2mat, sortedTitle, 'UniformOutput', false);
end


function normalizedWeights = normalizeWeights(WEIGHTS)
    % Initialize scaledWeights with the same size as WEIGHTS for column-wise scaling
    scaledWeights = zeros(size(WEIGHTS));
    
    % Scale down weights column-wise
    for col = 1:size(WEIGHTS, 2)
        maxWeight = max(abs(WEIGHTS(:, col)));
        if maxWeight == 0
            scaledWeights(:, col) = WEIGHTS(:, col);
        else
            scaledWeights(:, col) = WEIGHTS(:, col) / maxWeight;
        end
    end
    
    % Initialize normalizedWeights with the same size as scaledWeights for row-wise normalization
    normalizedWeights = zeros(size(scaledWeights));
    
    % Normalize weights row-wise
    for row = 1:size(scaledWeights, 1)
        rowWeights = scaledWeights(row, :);
        maxAbsWeight = max(abs(rowWeights));
        if maxAbsWeight == 0
            normalizedWeights(row, :) = rowWeights;
        else
            normalizedWeights(row, :) = rowWeights / maxAbsWeight;
        end
    end
end

