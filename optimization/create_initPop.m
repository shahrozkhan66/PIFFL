function [initialPop] = create_initPop(designSpace,popSize)
initialPop = zeros(popSize, size(designSpace,1));
    for j = 1:size(designSpace,1)
        initialPop(:,j) = designSpace(j,1) + (designSpace(j,2)-designSpace(j,1)).*rand(popSize,1);
    end
end