function [eigenValues, activeFeatures, samplesOnActiveSubspace] = PIFFL_FAS(trainingDataset, pValue, mBoots, localModelForGradApprox, activeSubspaceDim)

output = trainingDataset(:,end);
trainingDataset = normalize(trainingDataset);
%Samples(:,1:end-1) = normalize(Samples(:,1:end-1), 'range', [-1 1]);

DesignParas = size(trainingDataset(1,:),2)-1;
Dataset_size = size(trainingDataset(:,1),1);
Samples_withNoOutput = trainingDataset(:,1:DesignParas);

Reg_Weights_gpr = zeros(Dataset_size, DesignParas);
RegressionDataset = zeros(pValue,DesignParas+1);
Reg_Weights = zeros(Dataset_size, DesignParas);
Row_indx = 1;
for i = 1:Dataset_size
    Dataset_M = zeros(Dataset_size, 1);
    for j = 1:Dataset_size
        if (i ~= j)
            Dataset_M(j,:) = norm(Samples_withNoOutput(i,:)-Samples_withNoOutput(j,:), 2);
        end
    end
    [~, Index] = mink(Dataset_M, pValue);
    RegressionDataset(1:pValue,:) =  trainingDataset(Index(1:pValue),:);
    %%% Gaussian REgression %%%%
    if (localModelForGradApprox == 'GPR')
        Input_GPR = RegressionDataset(:,1:end-1);
        Output_GPR = RegressionDataset(:,end);
        mdl_GPR = fitrgp(array2table(Input_GPR),Output_GPR, 'Basis','linear',...
            'FitMethod','exact','PredictMethod','exact','KernelFunction','ardsquaredexponential');
        mdl_Coef_GPR = (mdl_GPR.Beta);
        ypred_GPR = predict(mdl_GPR,Input_GPR);
        %---- Cal GPR errors
        Rsq1 = 1 - sum((Output_GPR - ypred_GPR).^2)/sum((Output_GPR - mean(Output_GPR)).^2); % R Square Ordinary
        L = sqrt(loss(mdl_GPR,Input_GPR,Output_GPR));
        disp(['GPR RMSE:' num2str(L)]);
        disp(['GPR R^2:' num2str(Rsq1)]);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%  Linear REgression %%
    if (localModelForGradApprox == 'MLR')
        mdl = fitlm(array2table(RegressionDataset));
        mdl_Coef = table2array(mdl.Coefficients);
        ypred = predict(mdl,Input_GPR);
        disp(['MLR RMSE: ' num2str(mdl.RMSE)]);
        disp(['MLR R^2: ' num2str(mdl.Rsquared.Ordinary)]);
    end
    Reg_Weights_gpr(Row_indx,:) = transpose(mdl_Coef_GPR(2:DesignParas+1,1));
    Reg_Weights(Row_indx,:) = transpose(mdl_Coef(2:DesignParas+1,1));
    Row_indx = Row_indx + 1;
end

CovMatrix = (transpose(Reg_Weights)* Reg_Weights);
[~, eigen, CovMatrix_sort] = eig(CovMatrix);
eigen = diag(abs(eigen));

[~, Ind] = maxk(eigen, DesignParas);
eigenValues = transpose(sort(eigen, 'descend'));

CovMatrix_sort(:,1:DesignParas) = CovMatrix_sort(:,Ind(1:DesignParas));

activeFeatures = zeros(DesignParas, activeSubspaceDim);
activeFeatures(:,1:activeSubspaceDim) = CovMatrix_sort(:,1:activeSubspaceDim);
activeFeatures = transpose(activeFeatures);

activeSubError = zeros(DesignParas-1,1);
% Subspace error
for i = 1:DesignParas-1
    activeSubError(i,1) = norm((transpose(CovMatrix_sort(:,1:i)) * CovMatrix_sort(:,i+1:end)), 2);
end

% Sensitive Score
eigenValues = eigenValues';
sensScore_m = zeros(1,DesignParas);
for i=1:activeSubspaceDim
    sensScore_m(:,i) = CovMatrix_sort(i,:).^2*eigenValues;
end
sensScore_m = sensScore_m./sum(sensScore_m);


% Output from Active Subspace
output_actSubspace_1 = Samples_withNoOutput*CovMatrix_sort(:,1);
output_actSubspace_2 = Samples_withNoOutput*CovMatrix_sort(:,2);
samplesOnActiveSubspace =  [Samples_withNoOutput*CovMatrix_sort(:,1:activeSubspaceDim) trainingDataset(:,end)];

% Boot Strap Replicates
eigenVectors_boots = zeros(mBoots, DesignParas, activeSubspaceDim);
eigenValues_boots = zeros(mBoots, DesignParas);
dist_boots = zeros(DesignParas-1, mBoots);

for i = 1:mBoots
    Reg_Weights_boots = datasample(Reg_Weights,Dataset_size);
    CovMatrix_boots = transpose(Reg_Weights_boots) * Reg_Weights_boots;
    [~, eigen_boots, CovMatrix_sort_boots] = eig(CovMatrix_boots);
    eigen_boots = diag(abs(eigen_boots));
    [~, Ind_boots] = maxk(eigen_boots, DesignParas);
    eigenValues_boots(i,:) = sort(eigen_boots, 'descend');
    CovMatrix_sort_boots(:,1:DesignParas) = CovMatrix_sort_boots(:,Ind_boots(1:DesignParas));
    % eigen vetors
    for j = 1:activeSubspaceDim
        eigenVectors_boots(i,:,j) = CovMatrix_sort_boots(j,:);
    end
    % Subspace error (boot Strap)
    for k = 1:DesignParas-1
        dist_boots(k,i) = norm((transpose(CovMatrix_sort(:,1:k)) * CovMatrix_sort_boots(:,k+1:end)));
    end
end

min_dist_boots = sort(min(dist_boots,[],2));
mean_dist_boots = sort(mean(dist_boots,2));
max_dist_boots = sort(max(dist_boots,[],2));
min_eigenValues_boots = min(eigenValues_boots);
max_eigenValues_boots = max(eigenValues_boots);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         Figures            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Num = 1:DesignParas;
fig_1 = figure('Name','Eigenvalues');
hold on;
grid on;
plot(Num, min_eigenValues_boots, 'c', Num, max_eigenValues_boots, 'c')
fill([Num, fliplr(Num)], [min_eigenValues_boots, fliplr(max_eigenValues_boots)], 'c', 'EdgeColor', 'none')
plot(Num, eigenValues, '-o', 'Color', 'k', 'MarkerSize', 4, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'k')
xlim([1 DesignParas]);
xticks(1:1:DesignParas);
ax = gca;
set(gca, 'YScale', 'log')
ax.GridColor = [0.5 .5 .5];
ax.GridLineStyle = '-';
ax.GridAlpha = 0.5;
ax.Layer = 'top';
hold off;
set(fig_1,'Units','Inches');
pos = get(fig_1,'Position');
set(fig_1,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
ylabel('Eigenvalue');
xlabel('Index');


for j = 1:2
    fig_2 = figure('Name','Eigenvector');
    hold on;
    grid on;
    bar(Num, activeFeatures(j,:), 0.3, 'FaceColor',[0 .7 .7],'EdgeColor',[0 .5 .5])
    ylim([-1 1]);
    xticks(1:1:DesignParas);
    ax = gca;
    ax.GridColor = [0.5 .5 .5];
    ax.GridLineStyle = '-';
    ax.GridAlpha = 0.5;
    ax.Layer = 'top';
    hold off;
    set(fig_2,'Units','Inches');
    pos = get(fig_2,'Position');
    set(fig_2,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
    ylabel('Weights');
    xlabel('Parameter index');
    print(fig_2,figName,'-dpng')
end

Num = 1:DesignParas-1;
min_dist_boots = min_dist_boots';
max_dist_boots = max_dist_boots';
fig_3 = figure('Name','Subspace Error');
hold on;
grid on;
plot(Num, min_dist_boots, 'c', Num, max_dist_boots, 'c')
fill([Num, fliplr(Num)], [min_dist_boots, fliplr(max_dist_boots)], 'c', 'EdgeColor', 'none')
plot(Num, mean_dist_boots, '-o', 'Color', 'k', 'MarkerSize', 4, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'k')
xlim([1 DesignParas-1]);
xticks(1:1:DesignParas-1);
ax = gca;
set(gca, 'YScale', 'log')
ax.GridColor = [0.5 .5 .5];
ax.GridLineStyle = '-';
ax.GridAlpha = 0.5;
ax.Layer = 'top';
hold off;
set(fig_3,'Units','Inches');
pos = get(fig_3,'Position');
set(fig_3,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
ylabel('Distance');
xlabel('Subspace Dimension');

fig_4 = figure('Name','Sufficient Summary Plots 1D');
hold on;
grid on;
scatter(output_actSubspace_1, output, 40,'MarkerEdgeColor',[0 .5 .5],'MarkerFaceColor',[0 .7 .7],'LineWidth',1.5)
ax = gca;
ax.GridColor = [0.5 .5 .5];
ax.GridLineStyle = '-';
ax.GridAlpha = 0.5;
ax.Layer = 'top';
hold off;
set(fig_4,'Units','Inches');
pos = get(fig_4,'Position');
set(fig_4,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
ylabel('Quantity of Interest');
xlabel('Active Variable 1');

if (activeSubspaceDim > 1)
    fig_5 = figure('Name','Sufficient Summary Plots 2D'); % between two active parameters
    colorbar
    hold on;
    grid on;
    scatter(output_actSubspace_1, output_actSubspace_2, 40, output, 'filled')
    ax = gca;
    ax.GridColor = [0.5 .5 .5];
    ax.GridLineStyle = '-';
    ax.GridAlpha = 0.5;
    ax.Layer = 'top';
    hold off;
    set(fig_5,'Units','Inches');
    pos = get(fig_5,'Position');
    set(fig_5,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
    ylabel('Active Variable 2');
    xlabel('Active Variable 1');
    print(fig_5,figName,'-dpng')
end

fig_6 = figure('Name','Sensitivity Score');
hold on;
grid on;
bar(1:DesignParas,  sensScore_m)
xticks(1:DesignParas);
ax = gca;
ax.GridColor = [0.5 .5 .5];
ax.GridLineStyle = '-';
ax.GridAlpha = 0.5;
ax.Layer = 'top';
hold off;
set(fig_6,'Units','Inches');
pos = get(fig_6,'Position');
set(fig_6,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
xlabel('Parameters Index');
ylabel({'[$\alpha_i(n)$]'},'Interpreter','latex')
labels = {['m = ' num2str(activeSubspaceDim)]};
legend(labels, 'Location', 'Bestoutside');
end