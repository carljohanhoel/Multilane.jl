plotResultsContinuousDrive;

close all


%%

figWidth = 600;
figHeight = 230;

plotMaxMin = false;

ticksX = [0 1 2 3]*1e5;
ticksNorm = [0.9 1.0 1.1];
% axisVal = [0 3000000 0 1];
% axisValActions1 = [0 2350000 0 0.02];
% axisValActions2 = [0 2350000 0 1];
% axisValScores = [0 3000000 0 1.2];

lgdPos1 = [0.7042    0.2301    0.1967    0.3739];



figure(1)
hold on
sumReward = squeeze(sortedData(:,2,:));
sumReward = sumReward(~isnan(sumReward(:)));
plot(totalSteps,meanValue/nSimSteps,'b')
xlabel('Training step')
ylabel('Mean reward per step')
ylabel('$$\bar{r}$$','Interpreter','Latex')

xticks(ticksX)
yticks([0.8 0.9 1.0])
hAxes = gca;
hAxes.XAxis.Exponent = 5;
set(gca,'FontName','Times','FontSize',12)
% lgd1 = legend('Agent1_{CNN}','Agent2_{CNN}','Agent1_{FCNN}','Agent2_{FCNN}');
% % lgd1.FontSize = 10;
% set(lgd1,'Position',[0.69 0.27 0.1933 0.4239])


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Mean speed
figure(2)
clf(2)
hold on
plot(totalStepsVec,mean(idleDistance)/nSimSteps/dt*ones(1,length(step)),'r-.')
plot(totalStepsVec,mean(refDistance)/nSimSteps/dt*ones(1,length(step)),'g-.')
plot(totalStepsVec,mean(dpwDistance)/nSimSteps/dt*ones(1,length(step)),'m-.')
plot(totalSteps,squeeze(mean(sortedData(1:20,3,:),'omitnan'))/nSimSteps/dt,'b')  
plot(totalStepsVec,ones(1,length(step))*25,'k-.')

xlabel('Training step')
ylabel('Normalized mean distance')
ylabel('$$\bar{v}$$','Interpreter','Latex')

xticks(ticksX)
yticks([19 21 23 25])
hAxes = gca;
hAxes.XAxis.Exponent = 5;

set(gca,'FontName','Times','FontSize',12)
lgd2 = legend('IDM','IDM/MOBIL','MCTS','MCTS/RL','Empty road');
% lgd1.FontSize = 10;
set(lgd2,'Position',lgdPos1)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Norm with idle performance
figure(3)
clf(3)
hold on
plot(totalStepsVec,ones(1,length(step)),'r-.')
plot(totalStepsVec,ones(1,length(step))*mean(normWrtIdleRef),'g-.')
plot(totalStepsVec,ones(1,length(step))*mean(normWrtIdleDpw),'m-.')
plot(totalSteps,mean(normWrtIdleAz,2,'omitnan'),'b')
if plotMaxMin
    plot(totalSteps,min(normWrtIdleAz,[],2),'b--')
    plot(totalSteps,max(normWrtIdleAz,[],2),'b--')
end
% errorbar(totalSteps,mean(normWrtIdleAz,2,'omitnan'),std(normWrtIdleAz','omitnan')/sqrt(length(uniqueSteps)),'b')
plot(totalStepsVec,ones(1,length(step))*mean(normWrtIdle25),'k-.')

xlabel('Training step')
ylabel('Normalized mean distance')
ylabel('$$\bar{d}/\bar{d}_\mathrm{IDM}$$','Interpreter','Latex')

xticks(ticksX)
yticks(ticksNorm)
hAxes = gca;
hAxes.XAxis.Exponent = 5;
set(gca,'FontName','Times','FontSize',12)
lgd3 = legend('IDM','IDM/MOBIL','MCTS','MCTS/RL','Empty road');
% lgd1.FontSize = 10;
set(lgd3,'Position',lgdPos1)



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Norm with IDM/MOBIL performance
figure(4)
clf(4)
hold on
plot(totalStepsVec,ones(1,length(step))*mean(normIdleDistance),'r-.')
plot(totalStepsVec,ones(1,length(step)),'g-.')
plot(totalStepsVec,ones(1,length(step))*mean(normDpwDistance),'m-.')
plot(totalSteps,mean(normDistance,2,'omitnan'),'b')
if plotMaxMin
    plot(totalSteps,min(normDistance,[],2),'b--')
    plot(totalSteps,max(normDistance,[],2),'b--')
end
% errorbar(totalSteps,mean(normDistance,2,'omitnan'),std(normDistance','omitnan')/sqrt(length(uniqueSteps)),'b')
plot(totalStepsVec,ones(1,length(step))*mean(normWrtRef25),'k-.')

xlabel('Training step')
ylabel('Normalized mean distance')
ylabel('$$\bar{d}/\bar{d}_\mathrm{IDM/MOBIL}$$','Interpreter','Latex')

xticks(ticksX)
yticks(ticksNorm)
hAxes = gca;
hAxes.XAxis.Exponent = 5;
set(gca,'FontName','Times','FontSize',12)
lgd4 = legend('IDM','IDM/MOBIL','MCTS','MCTS/RL','Empty road');
% lgd1.FontSize = 10;
set(lgd4,'Position',lgdPos1)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Norm with MCTS performance
figure(5)
clf(5)
hold on
plot(totalStepsVec,ones(1,length(step))*mean(normWrtDpwIdle),'r-.')
plot(totalStepsVec,ones(1,length(step))*mean(normWrtDpwRef),'g-.')
plot(totalStepsVec,ones(1,length(step)),'m-.')
plot(totalSteps,mean(normWrtDpwAz,2,'omitnan'),'b')
if plotMaxMin
    plot(totalSteps,min(normWrtDpwAz,[],2),'b--')
    plot(totalSteps,max(normWrtDpwAz,[],2),'b--')
end
% errorbar(totalSteps,mean(normWrtDpwAz,2,'omitnan'),std(normWrtDpwAz','omitnan')/sqrt(length(uniqueSteps)),'b')
plot(totalStepsVec,ones(1,length(step))*mean(normWrtDpw25),'k-.')

xlabel('Training step')
ylabel('Normalized mean distance')
ylabel('$$\bar{d}/\bar{d}_\mathrm{MCTS}$$','Interpreter','Latex')

xticks(ticksX)
yticks(ticksNorm)
hAxes = gca;
hAxes.XAxis.Exponent = 5;
set(gca,'FontName','Times','FontSize',12)
lgd5 = legend('IDM','IDM/MOBIL','MCTS','MCTS/RL','Empty road');
% lgd4.FontSize = 10;
set(lgd5,'Position',lgdPos1)



set(figure(1), 'Position', [100, 1000, figWidth, figHeight])
set(figure(2), 'Position', [1000, 1000, figWidth, figHeight])
set(figure(3), 'Position', [100, 100, figWidth, figHeight])
set(figure(4), 'Position', [1000, 100, figWidth, figHeight])
set(figure(5), 'Position', [2000, 100, figWidth, figHeight])
set(figure(6), 'Position', [2000, 1000, figWidth, figHeight])



%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Histograms
%Use the 100 reruns for this!!


figWidth = 1.5*300;
figHeight = 1.5*200;
% setAxis = [0.8 1.5 0 0.28];
% ticksY = [0 0.05 0.1 0.15 0.2 0.25];
ticksY = [0 0.1 0.2];
nbBins = 10;

figure(11)
histogram(normDistance(end,:),nbBins,'Normalization','probability');
xlabel('$$\bar{d}/\bar{d}_\mathrm{IDM/MOBIL}$$','Interpreter','Latex')
% axis(setAxis);
yticks(ticksY);
mu = mean(normDistance(end,:));
t2 = text(mu+0.04,0.23,['Mean: ',num2str(mu,'%1.2f')])
line([mu, mu], ylim, 'LineWidth', 1.5, 'Color', 'k','LineStyle','--');
set(t2,'FontName','Times','FontSize',12)
set(gca,'FontName','Times','FontSize',12)


figure(12)
histogram(normWrtDpwAz(end,:),nbBins,'Normalization','probability');
xlabel('$$\bar{d}/\bar{d}_\mathrm{IDM/MOBIL}$$','Interpreter','Latex')
% axis(setAxis);
yticks(ticksY);
mu2 = mean(normWrtDpwAz(end,:));
t2 = text(mu2+0.04,0.23,['Mean: ',num2str(mu2,'%1.2f')])
line([mu2, mu2], ylim, 'LineWidth', 1.5, 'Color', 'k','LineStyle','--');
set(t2,'FontName','Times','FontSize',12)
set(gca,'FontName','Times','FontSize',12)


set(figure(11), 'Position', [100, 100, figWidth, figHeight])
set(gcf,'renderer','Painters')
set(figure(12), 'Position', [800, 100, figWidth, figHeight])
set(gcf,'renderer','Painters')

