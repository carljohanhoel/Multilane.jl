clear all
clc


%%

logfile = './Logs/190128_142353_driving_exit_lane_Cpuct_0p1_Bigger_net_New_action_space_No_batchnorm';
x = load([logfile '/Reruns/x_empty_road.txt']);
v = load([logfile '/Reruns/values_empty_road.txt']);
a = load([logfile '/Reruns/a_empty_road.txt']);
xa = load([logfile '/Reruns/x_action_map.txt']);

v = v(:,[1,4,7,10]);
v(:,5) = v(:,4);

a(:,5) = a(:,4);

y = 0.5:1:4.5;
ya = 1:1:4;

[X,Y] = meshgrid(x,y);
[XA,YA] = meshgrid(xa,ya);


%%
figure(1)
clf(1)
s = surf(X,Y,v')
s.EdgeColor = 'none';
colorbar


%%
figure(2)
clf(2)
p = pcolor(X,Y,v');
p.EdgeColor = 'none';
colorbar
colormap summer(10000)
% colormap autumn
% colormap cool
caxis([0 20])

%%
figure(3)
clf(3)
p = pcolor(X,Y,v');
p.EdgeColor = 'none';
cb = colorbar;
colormap summer(10000)
caxis([0 20])


% set(figure(3), 'Position', [10, 100, 3000, 200])
% a_length_pixels = 20;
% a_length_horisontal = 5;
% a_length_vertical = 0.32;

%For paper
set(figure(3), 'Position', [10, 100, 6000, 400])
a_length_pixels = 40;
a_length_horisontal = 5;
a_length_vertical = 0.32;

set(gca,'ytick',[])
set(gca,'ycolor',[1 1 1])
xticks([0 500 1000])
set(gca,'FontSize',50)

xa = xa*0.995 + 0.0025*max(xa);

for i=1:size(xa,1)
    for j=1:size(ya,2)
        if a(i,j) == 1
            angle_arrow = 0;
            a_length = a_length_horisontal;
            c = 'black';
        elseif a(i,j) == 2
            angle_arrow = 0;
            a_length = a_length_horisontal;
            c = 'red';
        elseif a(i,j) == 3
            angle_arrow = 0;
            a_length = a_length_horisontal;
            c = 'red';
        elseif a(i,j) == 4
            angle_arrow = -90;
            a_length = a_length_vertical;
            c = 'black';
        elseif a(i,j) == 5
            angle_arrow = 90;
            a_length = a_length_vertical;
            c = 'black';
        else
            disp('ERROR')
        end
        if xa(i) == max(xa) && ya(j) == 1
            angle_arrow = -90;
            a_length = a_length_vertical;
            c = 'black';
        end
           
        p1_arrow = [xa(i),ya(j)];
        p2_arrow = [p1_arrow(1)+a_length*cos(angle_arrow), p1_arrow(2)+a_length*sin(angle_arrow)];
        arrow(p1_arrow,p2_arrow,'Length',a_length_pixels,'BaseAngle',70,'Color',c);
    end
end


cx1=get(gca,'position');
cx=get(cb,'Position');
cx(3)=0.02;
set(cb,'Position',cx)
set(gca,'position',cx1)
