function [rho,pvalue,slope,intercept] = PlotSimpleCorrelationWithRegression(var1,var2,markersize,color)

    set(groot,'defaulttextinterpreter','latex');  
    set(groot, 'defaultAxesTickLabelInterpreter','latex');  
    set(groot, 'defaultLegendInterpreter','latex'); 
    set(gcf,'color','w')
    
    [rho,pvalue] = ComputeSimpleCorrelations(var1,var2);
    
    glm = fitglm(var1,var2); intercept=glm.Coefficients.Estimate(1); slope=glm.Coefficients.Estimate(2);
    %%
    regression_line=slope*var1+intercept;
    s=scatter(var1,var2,markersize,'fill'); s.MarkerFaceColor=color;
    hold on
    plot(var1,regression_line,'Color','k','LineWidth',1)
    set(gca,'FontSize',25)
end