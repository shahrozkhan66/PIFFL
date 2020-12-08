function [overallBestDesign, bestCostValue, cost_iterations] = PIFFL_optimisation(trainingSamples,samplesOnActiveSubspace, populationSize, numOfIterations)
mins = zeros(1,size(samplesOnActiveSubspace,2));
maxs = mins ;
for i=1:size(samplesOnActiveSubspace,2)
    mins(i) = min(samplesOnActiveSubspace(:,i));
    maxs(i) = max(samplesOnActiveSubspace(:,i));
end
DesignsSpace = [mins; maxs];
DesignsSpace= DesignsSpace';
initialPop = create_initPop(DesignsSpace,populationSize);
[overallBestDesign, bestCostValue, cost_iterations] = JayaAlgorithm(surrogateModel,initialPop, DesignsSpace, numOfIterations);
figure
%Denormalisation
plot((cost_iterations*std(trainingSamples)) + mean(trainingSamples))
grid on
ylabel('QoI')
xlabel('Iterations')
disp(['Best Design - normalized:' num2str(bestCostValue)]);
%Denormalisation
denorm_bestCostValue = (bestCostValue*std(output)) + mean(output);
disp(['Best Design -denormalized:' num2str(denorm_bestCostValue)]);
end