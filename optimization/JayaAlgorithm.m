function [OverallBestDesign, BestCostValue, cost_iterations] = JayaAlgorithm(model, inp_population, designSpace, numberOfIternations)

cost_iterations = zeros(1,numberOfIternations);
for n=1:numberOfIternations
    train_population = zeros(size(inp_population,1),size(inp_population,2));
    rand_1 = rand(1,size(inp_population,2));
    rand_2 = rand(1,size(inp_population,2));
    Costs = zeros(1,size(inp_population,1));
    for k=1:size(inp_population,1)
        designTemp = array2table(inp_population(k,:));
        if (size(designTemp,2)==1)
            designTemp.Properties.VariableNames{'Var1'} = 'Var';
        end
        Costs(k) = predict(model,designTemp);
    end
    [~, minIndx] = min(Costs);
     bestDesign = inp_population(minIndx,:);
    [~, maxIndx] = max(Costs);
    worstDesign = inp_population(maxIndx,:);
    
    for i = 1:size(inp_population,1)
        train_population(i,:) = inp_population(i,:) + (rand_1.*(bestDesign - abs(inp_population(i,:)))) ...
            - (rand_2.*(worstDesign - abs(inp_population(i,:))));
        for j = 1:size(inp_population,2)% this loop checks if any of the variable is outside the design space
            if (train_population(i,j) < designSpace(j,1) || train_population(i,j) > designSpace(j,2))
                train_population(i,j) =  inp_population(i,j);
            end
        end
        designTemp = array2table(train_population(i,:));
        if (size(designTemp,2)==1)
            designTemp.Properties.VariableNames{'Var1'} = 'Var';
        end
        newCost = predict(model,designTemp);
        
        designTemp = array2table(inp_population(i,:));
        if (size(designTemp,2)==1)
            designTemp.Properties.VariableNames{'Var1'} = 'Var';
        end
        oldCost = predict(model,designTemp);
        
        if (oldCost <= newCost)
            train_population(i,:) = inp_population(i,:);
        end
    end
    inp_population = train_population;
    
    %%% storing cost in each iteration %%%%%%%%%%%%%%%%%5
    Costs_temp = zeros(1,size(inp_population,1));
    for k=1:size(inp_population,1)
        designTemp = array2table(inp_population(k,:));
        if (size(designTemp,2)==1)
            designTemp.Properties.VariableNames{'Var1'} = 'Var';
        end
        Costs_temp(k) = predict(model,designTemp);
    end
    [cost_iterations(n), ~] = min(Costs_temp);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55
    
end
Costs_temp = zeros(1,size(inp_population,1));
for k=1:size(inp_population,1)
    designTemp = array2table(inp_population(k,:));
    if (size(designTemp,2)==1)
       designTemp.Properties.VariableNames{'Var1'} = 'Var';
    end
    Costs_temp(k) = predict(model,designTemp);
end
[BestCostValue, minIndx] = min(Costs_temp);
OverallBestDesign = inp_population(minIndx,:);
end