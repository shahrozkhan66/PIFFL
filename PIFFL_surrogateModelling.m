%%%%%%%%%% FINAL GPR MODEL TRAINING %%%%%%%%%%%%%%%%%%5
% GPR on Train on Functionally Active subspace
function surrogateModel = PIFFL_surrogateModelling(traningDataset)
tic
X = traningDataset(:,1:end-1);
Y = traningDataset(:,end);
rng default
surrogateModel = fitrgp(array2table(X),Y,...
    'KernelFunction', 'squaredexponential',...
    'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',...
    struct('AcquisitionFunctionName','expected-improvement-plus'));
partitionedModel = crossval(surrogateModel, 'KFold', 5);
valid_MSE = kfoldLoss(partitionedModel, 'LossFun', 'mse');
valid_RMSE = sqrt(kfoldLoss(partitionedModel, 'LossFun', 'mse'));
ypred_GPR = predict(surrogateModel,X);
%---- Cal GPR errors
Rsq2 = 1 - sum((Y - ypred_GPR).^2)/sum((Y - mean(Y)).^2); % R Square Ordinary
Samp_MSE = loss(surrogateModel,X,Y,'lossfun','mse');
Samp_RMSE = sqrt(loss(surrogateModel,X,Y,'lossfun','mse'));
disp(['GPR Sampling MSE:' num2str(Samp_MSE)]);
disp(['GPR Cross-Validation MSE:' num2str(valid_MSE)]);
disp(['GPR Sampling RMSE:' num2str(Samp_RMSE)]);
disp(['GPR Cross-Validation RMSE:' num2str(valid_RMSE)]);
disp(['GPR R^2:' num2str(Rsq2)]);
figure
plot(Y,ypred_GPR,'r.');
toc
end