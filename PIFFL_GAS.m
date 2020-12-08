function [eigenVector,eigenValue,samplesOnGAS,reducedDim_DesignSpace,mu] = PIFFL_GAS(traningSamples)
%PCA
[eigenVector, samplesOnGAS, eigenValue, ~, explained, mu] = pca(traningSamples);

%checking how many components explain a given variance threshhold
[~,n_components] = max(cumsum(explained) >= 95);
 samplesOnGAS = samplesOnGAS(:,1:n_components);
eigenVector = eigenVector(:, 1:n_components);
%projecting orignal DesignSpace into Lower Dimension
designSpace_temp = DesignSpace';
normDesignSpace = bsxfun(@minus,designSpace_temp,mean(designSpace_temp));
reducedDim_DesignSpace = normDesignSpace * eigenVector;
disp(reducedDim_DesignSpace);

figure(1);
percentVariance = zeros(1,size(explained,1)); 
for i = 1:size(explained,1)
    percentVariance(i) = sum(explained(1:i));
end
subplot(2,1,1)
plot (1:size(explained,1), percentVariance,'k-');
title('(a)')
xlabel('Principal Components');
ylabel('Cumulative Sum of Variance');
grid on
subplot(2,1,2)
bar(1:size(explained,1), explained, 0.3);
title('(b)')
xlabel('Principal Components');
ylabel('Variance');
grid on
end
