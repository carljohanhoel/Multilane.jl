%Use figure 1 and 2!!!

plotResultsExitLane;
solvedScenarios = squeeze(sortedData(:,11,:));

close all

%% Temp to test code
solvedScenarios2 = ones(100,16);
solvedScenarios2(5:end,1) = 0;
solvedScenarios2(75:end,2) = 0;
solvedScenarios2(89:end,3) = 0;
solvedScenarios2(96:end,4) = 0;
solvedScenarios2(99:end,5) = 0;
solvedScenarios2(99:end,6) = 0;
solvedScenariosTemp = solvedScenarios;
solvedScenarios = solvedScenarios2;


%%

figWidth = 600;
figHeight = 230;

plotMaxMin = false;

xmin = 0;
xmax = 2.6e5;
ticksX = [0 1 2 3]*1e5;
ticksNorm = [0.9 1.0 1.1];
% axisVal = [0 3000000 0 1];
% axisValActions1 = [0 2350000 0 0.02];
% axisValActions2 = [0 2350000 0 1];
% axisValScores = [0 3000000 0 1.2];

lgdPos1 = [0.7042    0.2301    0.1967    0.3739];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Success proportion

figure(1)
clf(1)
hold on
plot(totalSteps,mean(solvedScenarios,1),'b-')
% errorbar(totalSteps,mean(solvedScenarios,1,'omitnan'),std(solvedScenarios,'omitnan')/sqrt(size(solvedScenarios,1)),'b')
plot(totalSteps,mean(dpwSuccess)*ones(1,length(totalSteps)),'m-.')
plot(totalSteps,mean(refSuccess)*ones(1,length(totalSteps)),'g--')

xlabel('Training step')
% ylabel({'Proportion successful' ; 'episodes'})
ylabel('Success')


xticks(ticksX)
yticks([0.0 0.5 1.0])
hAxes = gca;
hAxes.XAxis.Exponent = 5;
set(gca,'FontName','Times','FontSize',12)
lgd1 = legend('MCTS/NN','MCTS','IDM/MOBIL');
% lgd1.FontSize = 10;
set(lgd1,'Position',[0.6861    0.2681    0.1967    0.2304])

axis([xmin xmax 0 1])

tix=get(gca,'ytick')';
set(gca,'yticklabel',num2str(tix,'%.1f'))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Mean time
figure(2)
clf(2)
hold on
plot(totalSteps,mean(normOnlySuccess,2,'omitnan'),'b')
errorbar(totalSteps,mean(normOnlySuccess,2,'omitnan'),std(normOnlySuccess','omitnan')/sqrt(size(normOnlySuccess,2)),'b','HandleVisibility','off')
plot(totalStepsVec,ones(1,length(step))*mean(normDpwOnlySuccess),'m-.')
plot(totalStepsVec,ones(1,length(step)),'g--')

xlabel('Training step')
ylabel('$$\bar{T}/\bar{T}_\mathrm{IDM/MOBIL}$$','Interpreter','Latex')

xticks(ticksX)
yticks([0.95 1.0])
% yticks([0.94 0.97 1.0 1.03])
hAxes = gca;
hAxes.XAxis.Exponent = 5;

set(gca,'FontName','Times','FontSize',12)
lgd2 = legend('MCTS/NN','MCTS','IDM/MOBIL');
% lgd1.FontSize = 10;
set(lgd2,'Position',[0.6928    0.6851    0.1967    0.2304])

axis([xmin xmax 0.95 1.03])

tix=get(gca,'ytick')';
set(gca,'yticklabel',num2str(tix,'%.2f'))

set(figure(1), 'Position', [100, 1000, figWidth, figHeight])
set(figure(2), 'Position', [1000, 1000, figWidth, figHeight])




