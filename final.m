%% ========================================================================
%  Full模型三参数搜索主程序
%
%  搜索参数：
%    1. rho：粒球数量比例，M = rho * c
%    2. tau_o：组间核族权重的熵正则化参数
%    3. tau_i：组内基础核权重的熵正则化参数
%
%  说明：
%    epsilon_out = outerSufficientIncrease 是外层充分上升判据的
%    数值容差，不属于需要搜索的模型参数。
%% ========================================================================

clearvars;
clc;

%% ===================== 1. 添加代码路径 =====================

projectDir = fileparts(mfilename('fullpath'));

if isempty(projectDir)
    projectDir = pwd;
end

codeRoot = fileparts(projectDir);

sharedCoreDir = fullfile(codeRoot, ...
    'UDGAMKC');

naturalNeighborDir = fullfile(sharedCoreDir, ...
    'UDGAMKC_Entropy_NaturalNeighbor');

addpath(projectDir, '-begin');

if isfolder(sharedCoreDir)
    addpath(sharedCoreDir, '-end');
end

if isfolder(naturalNeighborDir)
    addpath(naturalNeighborDir, '-end');
end

fprintf('代码路径添加完成。\n');


%% ===================== 2. 设置数据文件 =====================

dataFile = ...
    'D:\代码\需要构建核的多核聚类数据集\R15.mat';

kernelFile = ...
    ['D:\第三篇论文\代码\需要构建核的多核聚类数据集\', ...
     '新核矩阵\R15_KH.mat'];

assert(isfile(dataFile), ...
    '没有找到数据标签文件：%s', dataFile);

assert(isfile(kernelFile), ...
    '没有找到核矩阵文件：%s', kernelFile);


%% ===================== 3. 设置参数搜索范围 =====================

% 粒球数量比例：
% M = rho * c
search.rhoList = [2, 3, 5, 8, 10];

% 如果希望直接搜索粒球数量M，可以填写numBallsList，
% 此时程序将优先采用numBallsList，而不使用rhoList。
search.numBallsList = [];

% 组间核族权重熵正则参数 tau_o
search.tauOuterList = [0.1, 0.5, 1, 2, 5, 10];

% 组内基础核权重熵正则参数 tau_i
search.tauInnerList = [0.1, 0.5, 1, 2, 5, 10];

% 最优参数选择指标：
% 可选 'ACC'、'NMI'、'ARI' 或 'Purity'
search.selectionMetric = 'ACC';


%% ===================== 4. 设置算法固定参数 =====================

% 外层充分上升阈值epsilon_out
% 这是算法容差，不参与参数搜索
search.outerSufficientIncrease = 1e-6;

% 随机种子
search.randomSeedBase = 42;

% 最大外层迭代次数
search.maxOuterIter = 5;

% 固定粒球结构下的最大内层交替迭代次数
search.maxInnerIter = 100;

% 核二均值最大迭代次数
search.maxKernelKmeansIter = 100;

% 内层目标函数收敛阈值
search.innerTolerance = 1e-7;

% 核权重收敛阈值
search.innerWeightTolerance = 1e-7;

% 核矩阵数值稳定常数
search.kernelEpsilon = 1e-12;

% 距离计算数值稳定常数
search.distanceTolerance = 1e-12;

% 是否输出每一个Full模型的详细迭代信息
search.verboseModel = false;

% 是否保存各基础核对应的锚图B和Z
search.storeGraphs = false;

% 是否保存全部参数组合对应的模型
search.storeModels = false;

% 结果保存路径
search.outputDir = fullfile(projectDir, ...
    'results', 'parameter_search_hierarchical_sa');


%% ===================== 5. 加载数据与核矩阵 =====================

fprintf('\n============================================================\n');
fprintf('开始加载数据和核矩阵\n');
fprintf('============================================================\n');

loadOptions = struct();
loadOptions.labelFile = dataFile;

dataset = load_udgamkc_dataset( ...
    kernelFile, fileparts(dataFile), loadOptions);

assert(~isempty(dataset.y), ...
    '没有在数据文件中读取到真实标签。');

% 保证真实标签为列向量
trueLabels = dataset.y(:);

% 样本数
n = dataset.numSamples;

% 基础核数量
L = dataset.numKernels;

% 聚类类别数
c = numel(unique(trueLabels));

fprintf('数据集名称：%s\n', dataset.name);
fprintf('样本数量：%d\n', n);
fprintf('基础核数量：%d\n', L);
fprintf('真实类别数量：%d\n', c);


%% ===================== 6. 预处理基础核矩阵 =====================

fprintf('\n============================================================\n');
fprintf('开始预处理基础核矩阵\n');
fprintf('============================================================\n');

preprocessTimer = tic;

[Kbar, preprocessInfo] = preprocess_udgamkc_kernels( ...
    dataset.KH, search.kernelEpsilon);

preprocessSeconds = toc(preprocessTimer);

% 释放原始核矩阵，减少内存占用
dataset.KH = [];

fprintf('基础核矩阵预处理完成。\n');
fprintf('预处理时间：%.4f 秒\n', preprocessSeconds);


%% ===================== 7. 确定候选粒球数量 =====================

if ~isempty(search.numBallsList)

    % 用户直接指定粒球数量M
    numBallsList = unique( ...
        search.numBallsList(:)', 'stable');

    rhoList = numBallsList / c;

else

    % 根据M = rho*c计算粒球数量
    rhoList = unique( ...
        search.rhoList(:)', 'stable');

    numBallsList = round(rhoList * c);

end

% 保存未筛选前的参数
requestedNumBallsList = numBallsList;
requestedRhoList = rhoList;

% 每个粒球至少包含两个样本，因此：
% c <= M <= floor(n/2)
upperBallLimit = floor(n / 2);

validBallCandidate = ...
    isfinite(numBallsList) & ...
    numBallsList == floor(numBallsList) & ...
    numBallsList >= c & ...
    numBallsList <= upperBallLimit;

% 显示并删除不合法的粒球数量
if any(~validBallCandidate)

    excludedM = requestedNumBallsList( ...
        ~validBallCandidate);

    excludedRho = requestedRhoList( ...
        ~validBallCandidate);

    fprintf('\n以下粒球数量候选不满足约束，将被删除：\n');
    fprintf('粒球数量约束：%d <= M <= %d\n', ...
        c, upperBallLimit);
    fprintf('被删除的M：%s\n', mat2str(excludedM));
    fprintf('对应的rho：%s\n', mat2str(excludedRho));

end

numBallsList = numBallsList(validBallCandidate);
rhoList = rhoList(validBallCandidate);

% 删除可能产生的重复粒球数量
[numBallsList, uniqueBallIndex] = unique( ...
    numBallsList, 'stable');

rhoList = rhoList(uniqueBallIndex);

assert(~isempty(numBallsList), ...
    ['不存在合法的粒球数量候选。', ...
     '当前数据集要求 %d <= M <= %d。'], ...
    c, upperBallLimit);

fprintf('\n有效的rho候选：%s\n', mat2str(rhoList));
fprintf('对应的粒球数量M：%s\n', mat2str(numBallsList));


%% ===================== 8. 整理正则化参数候选 =====================

tauOuterList = unique( ...
    search.tauOuterList(:)', 'stable');

tauInnerList = unique( ...
    search.tauInnerList(:)', 'stable');

assert(all(isfinite(tauOuterList)) && ...
       all(tauOuterList > 0), ...
    '所有tau_o候选值都必须为正数。');

assert(all(isfinite(tauInnerList)) && ...
       all(tauInnerList > 0), ...
    '所有tau_i候选值都必须为正数。');

fprintf('tau_o候选：%s\n', mat2str(tauOuterList));
fprintf('tau_i候选：%s\n', mat2str(tauInnerList));


%% ===================== 9. 初始化参数搜索结果 =====================

combinationCount = ...
    numel(numBallsList) * ...
    numel(tauOuterList) * ...
    numel(tauInnerList);

rhoColumn = nan(combinationCount, 1);
ballColumn = nan(combinationCount, 1);
tauOuterColumn = nan(combinationCount, 1);
tauInnerColumn = nan(combinationCount, 1);

% 自然邻居自动确定的锚点邻居数量
kaColumn = nan(combinationCount, 1);

% 聚类指标的均值和标准差
% 顺序为：ACC、NMI、ARI、Purity
metricMean = nan(combinationCount, 4);
metricStd = nan(combinationCount, 4);

% 模型目标函数值
objectiveColumn = nan(combinationCount, 1);

% 被接受的外层候选结构数量
acceptedOuterColumn = nan(combinationCount, 1);

% 每个参数组合的运行时间
trainingSeconds = nan(combinationCount, 1);

% 最大基础核权重
maxKernelWeight = nan(combinationCount, 1);

% 有效基础核数量：1/sum(beta.^2)
effectiveKernelCount = nan(combinationCount, 1);

% 保存每个参数组合对应的基础核权重
kernelWeights = nan(combinationCount, L);

% 根据设置决定是否保留全部模型
models = cell(combinationCount, 1);

fprintf('\n============================================================\n');
fprintf('Full模型三参数搜索\n');
fprintf('数据集：%s\n', dataset.name);
fprintf('样本数：%d，基础核数：%d，类别数：%d\n', ...
    n, L, c);
fprintf('搜索参数：M或rho、tau_o、tau_i\n');
fprintf('自然邻居数量k_a由数据自适应确定，不参与搜索\n');
fprintf('外层充分上升阈值：%.3e\n', ...
    search.outerSufficientIncrease);
fprintf('参数组合总数：%d\n', combinationCount);
fprintf('============================================================\n');


%% ===================== 10. 遍历全部参数组合 =====================

row = 0;

for mIndex = 1:numel(numBallsList)

    currentM = numBallsList(mIndex);
    currentRho = rhoList(mIndex);

    for outerIndex = 1:numel(tauOuterList)

        currentTauOuter = tauOuterList(outerIndex);

        for innerIndex = 1:numel(tauInnerList)

            currentTauInner = tauInnerList(innerIndex);
            row = row + 1;

            fprintf('\n------------------------------------------------------------\n');
            fprintf('正在运行参数组合 %d/%d\n', ...
                row, combinationCount);
            fprintf('rho = %g\n', currentRho);
            fprintf('M = %d\n', currentM);
            fprintf('tau_o = %g\n', currentTauOuter);
            fprintf('tau_i = %g\n', currentTauInner);
            fprintf('------------------------------------------------------------\n');


            %% ---------- 10.1 设置Full模型参数 ----------

            modelOptions = struct();

            % 当前搜索的三个模型参数
            modelOptions.numBalls = currentM;
            modelOptions.rho = currentRho;
            modelOptions.tauOuter = currentTauOuter;
            modelOptions.tauInner = currentTauInner;

            % 外层充分上升参数
            modelOptions.outerSufficientIncrease = ...
                search.outerSufficientIncrease;

            modelOptions.outerAcceptanceMode = 'sufficient';

            % 最大迭代次数
            modelOptions.maxOuterIter = ...
                search.maxOuterIter;

            modelOptions.maxInnerIter = ...
                search.maxInnerIter;

            modelOptions.maxKernelKmeansIter = ...
                search.maxKernelKmeansIter;

            % 收敛阈值
            modelOptions.innerTolerance = ...
                search.innerTolerance;

            modelOptions.innerWeightTolerance = ...
                search.innerWeightTolerance;

            modelOptions.kernelEpsilon = ...
                search.kernelEpsilon;

            modelOptions.distanceTolerance = ...
                search.distanceTolerance;

            % 随机种子
            modelOptions.randomSeed = ...
                search.randomSeedBase;

            % Kbar已经完成预处理
            modelOptions.kernelsPreprocessed = true;

            % 是否保存图矩阵
            modelOptions.storeGraphs = ...
                search.storeGraphs;

            % 是否输出模型内部迭代过程
            modelOptions.verbose = ...
                search.verboseModel;


            %% ---------- 10.2 明确设置Full模型结构 ----------

            % 使用期望平方分布距离
            modelOptions.ballDistanceMode = ...
                'distribution';

            % 保留粒球内部方差项
            modelOptions.anchorVarianceWeight = 1;

            % 粒球分裂使用规模平衡因子
            modelOptions.useBalanceFactor = true;

            % 使用两层核权重：
            % alpha_g为组间权重，theta_{p|g}为组内权重
            modelOptions.kernelWeightMode = ...
                'hierarchical';

            % tau_o和tau_i分别控制两层权重
            modelOptions.sharedTemperature = false;

            % 使用动态粒球—锚图结构更新
            modelOptions.fixedStructure = false;

            % 组内熵由对应组权重alpha_g加权
            modelOptions.withinEntropyWeighted = true;

            % 三个核族的划分
            modelOptions.familyNames = ...
                {'polynomial', 'cosine', 'rbf'};

            modelOptions.familyIndices = ...
                {8:11, 12, 1:7};


            %% ---------- 10.3 运行当前Full模型 ----------

            runTimer = tic;

            model = udgamkc_hierarchical_sa( ...
                Kbar, c, modelOptions);

            trainingSeconds(row) = toc(runTimer);


            %% ---------- 10.4 评估当前聚类结果 ----------

            [meanOriginal, stdOriginal] = ...
                myNMIACCV2(model.U, trueLabels, c);

            % myNMIACCV2原始指标顺序：
            % ACC、NMI、Purity、ARI
            %
            % 这里统一整理为：
            % ACC、NMI、ARI、Purity

            metricMean(row, :) = [ ...
                meanOriginal(1), ...
                meanOriginal(2), ...
                meanOriginal(4), ...
                meanOriginal(3)];

            metricStd(row, :) = [ ...
                stdOriginal(1), ...
                stdOriginal(2), ...
                stdOriginal(4), ...
                stdOriginal(3)];


            %% ---------- 10.5 保存当前参数和结果 ----------

            rhoColumn(row) = currentRho;
            ballColumn(row) = currentM;
            tauOuterColumn(row) = currentTauOuter;
            tauInnerColumn(row) = currentTauInner;

            kaColumn(row) = ...
                model.numAnchorNeighbors;

            objectiveColumn(row) = ...
                model.objective;

            acceptedOuterColumn(row) = ...
                model.history.acceptedCandidateCount;

            kernelWeights(row, :) = ...
                model.beta(:)';

            maxKernelWeight(row) = ...
                max(model.beta);

            effectiveKernelCount(row) = ...
                1 / sum(model.beta .^ 2);

            if search.storeModels
                models{row} = model;
            end


            %% ---------- 10.6 输出当前运行结果 ----------

            fprintf('当前参数组合运行完成：\n');
            fprintf('  自然邻居数k_a：%d\n', ...
                model.numAnchorNeighbors);
            fprintf('  ACC：%.4f +/- %.4f\n', ...
                metricMean(row,1), metricStd(row,1));
            fprintf('  NMI：%.4f +/- %.4f\n', ...
                metricMean(row,2), metricStd(row,2));
            fprintf('  ARI：%.4f +/- %.4f\n', ...
                metricMean(row,3), metricStd(row,3));
            fprintf('  Purity：%.4f +/- %.4f\n', ...
                metricMean(row,4), metricStd(row,4));
            fprintf('  目标函数值：%.12g\n', ...
                objectiveColumn(row));
            fprintf('  接受的外层候选结构数：%d\n', ...
                acceptedOuterColumn(row));
            fprintf('  运行时间：%.2f 秒\n', ...
                trainingSeconds(row));

        end
    end
end


%% ===================== 11. 构造参数搜索结果表 =====================

summaryTable = table( ...
    rhoColumn, ...
    ballColumn, ...
    tauOuterColumn, ...
    tauInnerColumn, ...
    kaColumn, ...
    metricMean(:,1), ...
    metricStd(:,1), ...
    metricMean(:,2), ...
    metricStd(:,2), ...
    metricMean(:,3), ...
    metricStd(:,3), ...
    metricMean(:,4), ...
    metricStd(:,4), ...
    objectiveColumn, ...
    acceptedOuterColumn, ...
    trainingSeconds, ...
    maxKernelWeight, ...
    effectiveKernelCount, ...
    'VariableNames', { ...
        'Rho', ...
        'NumBalls', ...
        'TauOuter', ...
        'TauInner', ...
        'Ka', ...
        'ACCMean', ...
        'ACCStd', ...
        'NMIMean', ...
        'NMIStd', ...
        'ARIMean', ...
        'ARIStd', ...
        'PurityMean', ...
        'PurityStd', ...
        'Objective', ...
        'AcceptedOuterCandidates', ...
        'TrainingSeconds', ...
        'MaxKernelWeight', ...
        'EffectiveKernelCount'});

fprintf('\n============================================================\n');
fprintf('所有参数组合运行完成\n');
fprintf('============================================================\n');

disp(summaryTable);


%% ===================== 12. 根据指定指标选择最优参数 =====================

switch upper(search.selectionMetric)

    case 'ACC'
        selectionMeanName = 'ACCMean';
        selectionStdName = 'ACCStd';

    case 'NMI'
        selectionMeanName = 'NMIMean';
        selectionStdName = 'NMIStd';

    case 'ARI'
        selectionMeanName = 'ARIMean';
        selectionStdName = 'ARIStd';

    case 'PURITY'
        selectionMeanName = 'PurityMean';
        selectionStdName = 'PurityStd';

    otherwise
        error(['selectionMetric必须为ACC、NMI、', ...
               'ARI或Purity。']);

end

% 排序优先级：
% 1. 指定指标的均值越大越好；
% 2. 指定指标的标准差越小越好；
% 3. NMI均值越大越好；
% 4. 训练时间越短越好；
% 5. 若仍相同，选择最先运行的参数组合。

rankingMatrix = [ ...
    -summaryTable.(selectionMeanName), ...
     summaryTable.(selectionStdName), ...
    -summaryTable.NMIMean, ...
     summaryTable.TrainingSeconds, ...
    (1:height(summaryTable))'];

[~, rankingOrder] = sortrows( ...
    rankingMatrix, 1:5);

selectedIndex = rankingOrder(1);

selectedRow = summaryTable( ...
    selectedIndex, :);


%% ===================== 13. 获得最优参数对应的Full模型 =====================

selectedModel = models{selectedIndex};

% 如果参数搜索期间没有保存全部模型，
% 使用最优参数重新运行一次Full模型。
if isempty(selectedModel)

    fprintf('\n============================================================\n');
    fprintf('使用最优参数重新训练Full模型\n');
    fprintf('============================================================\n');

    bestOptions = struct();

    bestOptions.numBalls = ...
        selectedRow.NumBalls;

    bestOptions.rho = ...
        selectedRow.Rho;

    bestOptions.tauOuter = ...
        selectedRow.TauOuter;

    bestOptions.tauInner = ...
        selectedRow.TauInner;

    bestOptions.outerSufficientIncrease = ...
        search.outerSufficientIncrease;

    bestOptions.outerAcceptanceMode = ...
        'sufficient';

    bestOptions.maxOuterIter = ...
        search.maxOuterIter;

    bestOptions.maxInnerIter = ...
        search.maxInnerIter;

    bestOptions.maxKernelKmeansIter = ...
        search.maxKernelKmeansIter;

    bestOptions.innerTolerance = ...
        search.innerTolerance;

    bestOptions.innerWeightTolerance = ...
        search.innerWeightTolerance;

    bestOptions.kernelEpsilon = ...
        search.kernelEpsilon;

    bestOptions.distanceTolerance = ...
        search.distanceTolerance;

    bestOptions.randomSeed = ...
        search.randomSeedBase;

    bestOptions.kernelsPreprocessed = true;

    bestOptions.verbose = ...
        search.verboseModel;

    bestOptions.storeGraphs = ...
        search.storeGraphs;

    % Full模型设置
    bestOptions.ballDistanceMode = ...
        'distribution';

    bestOptions.anchorVarianceWeight = 1;

    bestOptions.useBalanceFactor = true;

    bestOptions.kernelWeightMode = ...
        'hierarchical';

    bestOptions.sharedTemperature = false;

    bestOptions.fixedStructure = false;

    bestOptions.withinEntropyWeighted = true;

    bestOptions.familyNames = ...
        {'polynomial', 'cosine', 'rbf'};

    bestOptions.familyIndices = ...
        {8:11, 12, 1:7};

    selectedModel = udgamkc_hierarchical_sa( ...
        Kbar, c, bestOptions);

end


%% ===================== 14. 输出最优参数和聚类结果 =====================

fprintf('\n============================================================\n');
fprintf('最优参数搜索结果\n');
fprintf('============================================================\n');

fprintf('最优参数选择指标：%s\n', ...
    upper(search.selectionMetric));

fprintf('最优rho：%g\n', ...
    selectedRow.Rho);

fprintf('最优粒球数量M：%d\n', ...
    selectedRow.NumBalls);

fprintf('最优tau_o：%g\n', ...
    selectedRow.TauOuter);

fprintf('最优tau_i：%g\n', ...
    selectedRow.TauInner);

fprintf('自适应自然邻居数k_a：%d\n', ...
    selectedRow.Ka);

fprintf('\n最优参数对应的聚类性能：\n');

fprintf('ACC：%.4f +/- %.4f\n', ...
    selectedRow.ACCMean, selectedRow.ACCStd);

fprintf('NMI：%.4f +/- %.4f\n', ...
    selectedRow.NMIMean, selectedRow.NMIStd);

fprintf('ARI：%.4f +/- %.4f\n', ...
    selectedRow.ARIMean, selectedRow.ARIStd);

fprintf('Purity：%.4f +/- %.4f\n', ...
    selectedRow.PurityMean, selectedRow.PurityStd);

fprintf('目标函数值：%.12g\n', ...
    selectedRow.Objective);

fprintf('运行时间：%.2f 秒\n', ...
    selectedRow.TrainingSeconds);


%% ===================== 15. 输出最优层次核权重 =====================

fprintf('\n============================================================\n');
fprintf('最优Full模型的层次核权重\n');
fprintf('============================================================\n');

fprintf('组间核族权重alpha：\n');

for g = 1:numel(selectedModel.alpha)

    fprintf('  alpha_%d (%s) = %.8f\n', ...
        g, ...
        selectedModel.familyNames{g}, ...
        selectedModel.alpha(g));

end

fprintf('\n组内基础核权重theta：\n');

for g = 1:numel(selectedModel.theta)

    currentKernelIndices = ...
        selectedModel.options.familyIndices{g};

    currentTheta = ...
        selectedModel.theta{g};

    fprintf('核族%d（%s）：\n', ...
        g, selectedModel.familyNames{g});

    for localIndex = 1:numel(currentKernelIndices)

        kernelIndex = ...
            currentKernelIndices(localIndex);

        fprintf('  theta_{%d|%d} = %.8f\n', ...
            kernelIndex, g, currentTheta(localIndex));

    end
end

fprintf('\n最终基础核组合权重beta：\n');

for p = 1:numel(selectedModel.beta)

    fprintf('  beta_%d = %.8f\n', ...
        p, selectedModel.beta(p));

end


%% ===================== 16. 保存全部搜索结果 =====================

if ~isfolder(search.outputDir)
    mkdir(search.outputDir);
end

resultPrefix = fullfile( ...
    search.outputDir, ...
    [dataset.name, '_hierarchical_SA_search']);

allCsvFile = ...
    [resultPrefix, '_all_combinations.csv'];

bestCsvFile = ...
    [resultPrefix, '_best_parameters.csv'];

weightsCsvFile = ...
    [resultPrefix, '_selected_kernel_weights.csv'];

matFile = ...
    [resultPrefix, '_complete.mat'];

% 保存全部参数组合
writetable(summaryTable, allCsvFile);

% 保存最优参数
writetable(selectedRow, bestCsvFile);

% 保存最优基础核权重
kernelWeightNames = compose( ...
    'Kernel%02d', 1:L);

kernelWeightTable = array2table( ...
    selectedModel.beta(:)', ...
    'VariableNames', cellstr(kernelWeightNames));

writetable(kernelWeightTable, weightsCsvFile);


%% ===================== 17. 构造完整结果结构体 =====================

searchResult = struct();

searchResult.dataset = dataset;
searchResult.options = search;

searchResult.requestedNumBallsList = ...
    requestedNumBallsList;

searchResult.requestedRhoList = ...
    requestedRhoList;

searchResult.validNumBallsList = ...
    numBallsList;

searchResult.validRhoList = ...
    rhoList;

searchResult.excludedNumBallsList = ...
    requestedNumBallsList(~validBallCandidate);

searchResult.excludedRhoList = ...
    requestedRhoList(~validBallCandidate);

searchResult.summary = summaryTable;
searchResult.summaryTable = summaryTable;

searchResult.selectedIndex = ...
    selectedIndex;

searchResult.selected = ...
    selectedRow;

searchResult.bestRow = ...
    selectedRow;

searchResult.selectedModel = ...
    selectedModel;

searchResult.kernelWeights = ...
    kernelWeights;

searchResult.models = ...
    models;

searchResult.preprocessInfo = ...
    preprocessInfo;

searchResult.preprocessSeconds = ...
    preprocessSeconds;

searchResult.allCsvFile = ...
    allCsvFile;

searchResult.bestCsvFile = ...
    bestCsvFile;

searchResult.weightsCsvFile = ...
    weightsCsvFile;

searchResult.matFile = ...
    matFile;

save(matFile, 'searchResult', '-v7.3');


%% ===================== 18. 输出结果保存位置 =====================

fprintf('\n============================================================\n');
fprintf('参数搜索实验全部完成\n');
fprintf('============================================================\n');

fprintf('全部参数组合结果：\n%s\n', ...
    allCsvFile);

fprintf('最优参数结果：\n%s\n', ...
    bestCsvFile);

fprintf('最优基础核权重：\n%s\n', ...
    weightsCsvFile);

fprintf('完整MAT结果：\n%s\n', ...
    matFile);