clc
close all 
clear all

%Learning Geometriucally Active Subspace (FAS)
GAS_traningSamples = trainingDataset(:,1:end);
[eigenVector,eigenValue,samplesOnGAS,reducedDim_DesignSpace,mu] = PIFFL_GAS(GAS_traningSamples);

%Learning Functionally Active Subspace (FAS)
pValue = 100;
mBoots = 10000; 
localModelForGradApprox = 'GPR';
activeSubspaceDim = 4;
FAS_trainingDataset = cat(2, samplesOnGAS, trainingDataset(:,end));
[eigenValues, FAF, samplesOnFAS] = PIFFL_FAS(FAS_trainingDataset, pValue, mBoots, localModelForGradApprox, activeSubspaceDim);

%Training of the Surrogate Model
trainingDatasetOnFAS = cat(1, samplesOnFAS, trainingDataset(:,end));
surrogateModel = PIFFL_surrogateModelling(trainingDatasetOnFAS);

%Running Optimisation
populationSize = 60;
numOfIterations = 1500;
[OverallBestDesign, BestCostValue, cost_iterations] = PIFFL_optimisation(samplesOnFAS,samplesOnActiveSubspace, populationSize, numOfIterations);

%Projecting the optimal deisgn on GAS
projectOnGAS = OverallBestDesign*activeFeatures';
%Denormalisation
for i =1:size(samplesOnFAS,2)
    projectOnGAS(i) = (projectOnGAS(i)*std(samplesOnFAS(:,i))) + mean(samplesOnFAS(:,i));
end

%Projecting the optimal deisgn On orignal Design space 
OptimalDesignInFullSpace = bsxfun(@plus, projectOnGAS * eigenVector.', mu);


