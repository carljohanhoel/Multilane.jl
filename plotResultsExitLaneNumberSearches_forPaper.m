% Number of MCTS searches

plotResultExitLaneNumberSearches;
solvedScenarios = squeeze(sortedData(:,11,:));

close all

%% Temp to test code
solvedScenarios2 = ones(100,6);
solvedScenarios2(15:end,1) = 0;
solvedScenarios2(70:end,2) = 0;
solvedScenarios2(84:end,3) = 0;
solvedScenarios2(98:end,4) = 0;
solvedScenariosTemp = solvedScenarios;
solvedScenarios = solvedScenarios2;

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

lgdPos1 = [0.7042    0.2301    0.1967    0.3739];



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Success proportion

figure(1)
clf(1)
hold on
plot(totalSteps,mean(solvedScenarios,1),'bx-')
% errorbar(totalSteps,mean(solvedScenarios,1,'omitnan'),std(solvedScenarios,'omitnan')/sqrt(size(solvedScenarios,1)),'b')
plot(totalSteps,mean(dpwSuccess)*ones(1,length(totalSteps)),'m-.')
plot(totalSteps,mean(refSuccess)*ones(1,length(totalSteps)),'g--')
xlabel('Iteration, $$n$$','Interpreter','Latex')
% ylabel({'Proportion successful' ; 'episodes'})
ylabel('Success')

% xticks(ticksX)
yticks([0.0 0.5 1.0])
set(gca,'FontName','Times','FontSize',12)
lgd1 = legend('MCTS/NN','MCTS','IDM/MOBIL');
% lgd1.FontSize = 10;
set(lgd1,'Position',[0.6861    0.2681    0.1967    0.2304])


hAxes = gca;
set(hAxes, 'XScale','log')
set(gca,'FontName','Times','FontSize',12)


axis([xmin xmax 0 1])

tix=get(gca,'ytick')';
set(gca,'yticklabel',num2str(tix,'%.1f'))


set(figure(1), 'Position', [100, 1000, figWidth, figHeight])


