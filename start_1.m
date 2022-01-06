clear all;
close all;
clc;

circle(0,0,0.5);
hold on;
circle(0,0,0.2);
hold on;
plot(0,0,'d','LineWidth',2);  %control station placement
hold on;
r=0.5; %radius of disk
xx0=0; yy0=0; %centre of disk
areaTotal=pi*r^2; %area of disk
NRB_tot=275;
numbPoints=50; % Number of sensor nodes in the area

it=100;
for j=1:it
% Reading the CQI-MCS table from 3GPP specification
num = xlsread('D:\Semester7\Mini Project\Matlab_code\CQI_MCS_mm.xlsx'); % Change the path on your computer
save('savenum.mat','num');
example = matfile('savenum.mat');
CQI_MCS_NR= example.num;

num1 = xlsread('D:\Semester7\Mini Project\Matlab_code\Data.csv'); % Change the path on your computer
save('savenum.mat','num1');
example = matfile('savenum.mat');
Data = example.num1;
 
% Creating the user information (Location, Data rate requirements)
a = 0.1;
b = 0.5;
radius1 = (b-a).*rand(numbPoints,1) + a;

a = 0;
b = 360;
theta1 = (b-a).*rand(numbPoints,1) + a;

sector_1 = [];
sector_2 = [];
sector_3 = [];
sector_4 = [];
sector_5 = [];
sector_6 = [];
sector_7 = [];
sector_8 = [];

total_1 = 0;
total_2 = 0;
total_3 = 0;
total_4 = 0;
total_5 = 0;
total_6 = 0;
total_7 = 0;
total_8 = 0;


for i=1:numbPoints
xx1(i)=radius1(i)*cosd(theta1(i))+xx0;
yy1(i)=radius1(i)*sind(theta1(i))+yy0;
T = i;

if (((theta1(i) >= 0)&& (theta1(i) < 45)) && (radius1(i)>=0.2) )
sector_1 = [sector_1, i];
V = num1(num1(:,1) == T,:);
a2 = V(:,2);
total_1 = total_1 + a2;

elseif (((theta1(i) >= 45)&& (theta1(i) < 90)) && (radius1(i)>=0.2) )
sector_2 = [sector_2, i];
V = num1(num1(:,1) == T,:);
a2 = V(:,2);
total_2 = total_2 + a2;

elseif (((theta1(i) >= 90)&& (theta1(i) < 135)) && (radius1(i)>=0.2) )
sector_3 = [sector_3, i];
V = num1(num1(:,1) == T,:);
a2 = V(:,2);
total_3 = total_3 + a2;

elseif (((theta1(i) >= 135)&& (theta1(i) < 180)) && (radius1(i)>=0.2) )
sector_4 = [sector_4, i];
V = num1(num1(:,1) == T,:);
a2 = V(:,2);
total_4 = total_4 + a2;

elseif (((theta1(i) >= 180)&& (theta1(i) < 225)) && (radius1(i)>=0.2) ) 
sector_5 = [sector_5, i];
V = num1(num1(:,1) == T,:);
a2 = V(:,2);
total_5 = total_5 + a2;

elseif (((theta1(i) >= 225)&& (theta1(i) < 270)) && (radius1(i)>=0.2) )  
sector_6 = [sector_6, i];
V = num1(num1(:,1) == T,:);
a2 = V(:,2);
total_6 = total_6 + a2;

elseif (((theta1(i) >= 270)&& (theta1(i) < 315)) && (radius1(i)>=0.2) )    
sector_7 = [sector_7, i];
V = num1(num1(:,1) == T,:);
a2 = V(:,2);
total_7 = total_7 + a2;

elseif(((theta1(i) >= 315)&& (theta1(i) < 360)) && (radius1(i)>=0.2) )
sector_8 = [sector_8, i];
V = num1(num1(:,1) == T,:);
a2 = V(:,2);
total_8 = total_8 + a2;
end
end

%Shift centre of disk to (xx0,yy0)

scatter(xx1,yy1,'r');
labels = cellstr( num2str([1:numbPoints]') );
text(xx1,yy1,labels);

%voronoi(xx1,yy1);
xlabel('x');ylabel('y');
axis square;

user_info(1,:)=xx1;
user_info(2,:)=yy1;

% disp("Sector_1")
% disp(sector_1)
% disp("Sector_2")
% disp(sector_2)
% disp("Sector_3")
% disp(sector_3)
% disp("Sector_4")
% disp(sector_4)
% disp("Sector_5")
% disp(sector_5)
% disp("Sector_6")
% disp(sector_6)
% disp("Sector_7")
% disp(sector_7)
% disp("Sector_8")
% disp(sector_8)


vectorarray = {sector_1, sector_2, sector_3, sector_4, sector_5, sector_6, sector_7, sector_8}; 
len=0;
for k = 1:numel(vectorarray)
c = length(vectorarray{k});
if c > len
len = c;
end    
end
disp("The maximum sensor nodes among the obtained sectors is:")
disp(len)

for k = 1:numel(vectorarray)
c = length(vectorarray{k});
    if length(vectorarray{k}) < len
    vectorarray{k} = [vectorarray{k}, zeros(1, len - length(vectorarray{k}))];
    vac(k,:)=vectorarray{k};
    disp(vectorarray{k});    
end
end

required_data_rate1 = (total_1 * 8) / 60000;
required_data_rate2 = (total_2 * 8) / 60000;
required_data_rate3 = (total_3 * 8) / 60000;
required_data_rate4 = (total_4* 8) / 60000;
required_data_rate5 = (total_5 * 8) / 60000;
required_data_rate6 = (total_6 * 8) / 60000;
required_data_rate7 = (total_7 * 8) / 60000;
required_data_rate8 = (total_8 * 8) / 60000;


N = 8;                                                         % Number Of Segments
a = linspace(0, 2*pi, N*10);
x = r*cos(a);
y = r*sin(a);
figure(1);
plot(x, y)
hold on
plot([zeros(1,N); x(1:10:end)], [zeros(1,N); y(1:10:end)])
hold off
axis equal

datarate_array(j,:)=[required_data_rate1,required_data_rate2,required_data_rate3,required_data_rate4,required_data_rate5,required_data_rate6,required_data_rate7,required_data_rate8];
clearvars -except datarate_array j it r xx0 yy0 areaTotal NRB_tot numbPoints CQI_MCS_NR
end

value=sum(datarate_array)/it
sum(value)

for i=1:8
       % Find required data from the sensor nodes insiode the sector i
   
    ra = randi([8 15],1);
    rs = randi([8 15],1);
    
    % Findind achievable data rate using CQI MCS mapping
    rate_per_RB(i)=(CQI_MCS_NR(ra+1,5)*log2(CQI_MCS_NR(ra+1,4))*114)/(17.6*14);
    
    % Calculating required RBs
    req_RB_a(i)=ceil(datarate_array(i)/rate_per_RB(i));
    
    % Repear it for anchor drone to the control station
    
        % Findind achievable data rate using CQI MCS mapping
    rate_per_RB(i)=(CQI_MCS_NR(rs+1,5)*log2(CQI_MCS_NR(rs+1,4))*114)/(17.6*14);
    
    % Calculating required RBs
    req_RB_s(i)=ceil(datarate_array(i)/rate_per_RB(i));
    N_total(i)= req_RB_a(i)+ req_RB_s(i);
end
disp(max(N_total))
figure(2)
bar(1:1:8,value)
p = [7,12,14,17,20]
q = [10,20,30,40,50]
figure(3)
bar(q,p)

