function [res_mean,res_std,res1,res2,res3,res4,rawLabels,mappedLabels]= myNMIACCV2(U,Y,numclass)
% Local compatibility copy from SimpleMKKM ClusteringEvaluation.
%
% The first six outputs retain the original behavior and ordering:
%   res1=ACC, res2=NMI, res3=Purity, res4=ARI.
% Two optional outputs were added so the UDGAMKC experiment can save the
% actual discrete labels from all 20 litekmeans evaluations.

stream = RandStream.getGlobalStream;
reset(stream);
U_normalized = U ./ repmat(sqrt(sum(U.^2, 2)), 1,numclass);
maxIter = 20;
% tmp1 = zeros(maxIter,1);
% tmp2 = zeros(maxIter,1);
% tmp3 = zeros(maxIter,1);
% for iter = 1:maxIter
%     indx = litekmeans(U_normalized,numclass,'MaxIter',100, 'Replicates',1);
%     indx = indx(:);
%     [newIndx] = bestMap(Y,indx);
%     tmp1(iter) = mean(Y==newIndx);
%     tmp2(iter) = MutualInfo(Y,newIndx);
%     tmp3(iter) = purFuc(Y,newIndx);
% end
% res = [max(tmp1);max(tmp2);max(tmp3)];
res1 = zeros(maxIter,1);
res2 = zeros(maxIter,1);
res3 = zeros(maxIter,1);
res4 = zeros(maxIter,1);
rawLabels = zeros(size(U,1),maxIter);
mappedLabels = zeros(size(U,1),maxIter);
for it = 1 : maxIter
    indx = litekmeans(U_normalized,numclass, 'MaxIter',100, 'Replicates',20);
    %% indx = kmeans(U_normalized,numclass, 'MaxIter',100, 'Replicates',maxIter);
    indx = indx(:);
    [newIndx] = bestMap(Y,indx);
    rawLabels(:,it) = indx;
    mappedLabels(:,it) = newIndx;
    res1(it) = mean(Y==newIndx);
    res2(it) = MutualInfo(Y,newIndx);
    res3(it) = purFuc(Y,newIndx);
    res4(it) = adjrandindex(Y,newIndx);
end
res_mean(1) = mean(res1);
res_mean(2) = mean(res2);
res_mean(3) = mean(res3);
res_mean(4) = mean(res4);
res_std(1) = std(res1);
res_std(2) = std(res2);
res_std(3) = std(res3);
res_std(4) = std(res4);
end

