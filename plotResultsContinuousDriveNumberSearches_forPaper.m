%Use figure 4!!!

% Number of MCTS searches

plotResultNumberSearches;


close all


%%

figWidth = 600;
figHeight = 230;

plotMaxMin = false;

xmin = 0;
xmax = 10000;
ticksX = [0 1 2 3 4 5]*1e3;
ticksNorm = [1.0 1.05 1.1];
% axisVal = [0 3000000 0 1];
% axisValActions1 = [0 2350000 0 0.02];
% axisValActions2 = [0 2350000 0 1];
% axisValScores = [0 3000000 0 1.2];

% lgdPos1 = [0.7042    0.2301    0.1967    0.3739];
% lgdPos1 = [0.6961    0.3812    0.1967    0.2304];
lgdPos1 = [0.6928    0.4377    0.1967    0.2304];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Norm with IDM/MOBIL performance
figure(4)
clf(4)
hold on
% plot(totalSteps,ones(1,length(totalSteps))*mean(normIdleDistance),'r-.')
% plot(totalSteps,mean(normDistance,2,'omitnan'),'bx-')
% errorbar(totalSteps,mean(normDistance,2,'omitnan'),std(normDistance','omitnan')/sqrt(size(normDistance,2)),'b')
selection = [1,2,4,6,7,8];
totalStepsSelection = totalSteps(selection);
normDistanceSelection = normDistance(selection,:);
plot(totalStepsSelection,mean(normDistanceSelection,2,'omitnan'),'b')
errorbar(totalStepsSelection,mean(normDistanceSelection,2,'omitnan'),std(normDistanceSelection','omitnan')/sqrt(size(normDistanceSelection,2)),'b','HandleVisibility','Off')
plot(totalSteps,ones(1,length(totalSteps))*mean(normDpwDistance),'m-.')
plot(totalSteps,ones(1,length(totalSteps)),'g--')
if plotMaxMin
    plot(totalSteps,min(normDistance,[],2),'b--')
    plot(totalSteps,max(normDistance,[],2),'b--')
end
% errorbar(totalSteps,mean(normDistance,2,'omitnan'),std(normDistance','omitnan')/sqrt(length(uniqueSteps)),'b')
% plot(totalStepsVec,ones(1,length(step))*mean(normWrtRef25),'k-.')

xlabel('Iteration, $$n$$','Interpreter','Latex')
ylabel('Normalized mean distance')
ylabel('$$\bar{v}/\bar{v}_\mathrm{IDM/MOBIL}$$','Interpreter','Latex')

% xticks(ticksX)
yticks([0.8 0.85 0.9 0.98 1.0 1.02 1.04 1.06 1.1])
hAxes = gca;
set(hAxes, 'XScale','log')
% hAxes.XAxis.Exponent = 5;
set(gca,'FontName','Times','FontSize',12)
% lgd4 = legend('IDM','IDM/MOBIL','MCTS','MCTS/RL','Empty road');
% lgd1.FontSize = 10;
% set(lgd4,'Position',lgdPos1)
lgd4 = legend('MCTS/NN','MCTS','IDM/MOBIL');
set(lgd4,'Position',lgdPos1)

axis([xmin xmax 0.98 1.06])


tix=get(gca,'ytick')';
set(gca,'yticklabel',num2str(tix,'%.2f'))



% set(figure(1), 'Position', [100, 1000, figWidth, figHeight])
% set(figure(2), 'Position', [1000, 1000, figWidth, figHeight])
% set(figure(3), 'Position', [100, 100, figWidth, figHeight])
set(figure(4), 'Position', [1000, 100, figWidth, figHeight])
% set(figure(5), 'Position', [2000, 100, figWidth, figHeight])
% set(figure(6), 'Position', [2000, 1000, figWidth, figHeight])



