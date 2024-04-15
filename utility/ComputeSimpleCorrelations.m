function [rho,pvalue] = ComputeSimpleCorrelations(var1,var2)
    % This function computes simple correlations with no covariate
    %   Detailed explanation goes here
    
    [rho,pvalue] = corr(var1,var2,'rows','complete'); 
end