load('PGExtraC.mat')

figure
hold on
plot(losses(1,:),'k')
plot(losses(2,:),'b')
plot(losses(3,:),'r')

legend(legend1,legend2,legend3,'ItemHitFcn',@cb_legend)

xlabel('# Iterations (k)')
ylabel('Relative err ( (f(xk) - f*)/f* )')

ylim([10e-15,10e0])
Ax = gca;
Ax.YScale = 'log';

%%
load('PGExtraC_SeriesPar_Damping.mat')

figure
hold on
plot([losses{1}; losses{2}].')
plot(losses{3}.','r','LineWidth',1)

lgdName0 = who('legend*');
lgdNames = lgdName0 + ",";
lgdNames{end} = strrep(lgdNames{end},',','');
eval(['legend(' [lgdNames{:}] ',"ItemHitFcn",@cb_legend)'])

xlabel('# Iterations (k)')
ylabel('Relative err ( (f(xk) - f*)/f* )')

ylim([10e-15,10e0])
Ax = gca;
Ax.YScale ='log';

