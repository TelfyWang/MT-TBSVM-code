%function[accuracy,time]=MTNPSVM(X,Y,xTrain,yTrain,xTest,yTest,mu,c1,c2,p,ep)
% MTNPSVM 此处显示有关此函数的摘要
% Multi-Task Nonparallel Support Vector Machine
% clear all;
spHeart = load('...\data\isolet-ab.mat');
X = spHeart.X1; Y = spHeart.Y1;xTrain=spHeart.x1Train; yTrain=spHeart.y1Train;xTest = spHeart.x1Test; yTest = spHeart.y1Test;
mu=0.5;
c1=5;c2=5;
ep=0.7;
p=1;
c3=c1;c4=c2;
Tasknum=length(xTrain);
A=X(Y==1,:);
B=X(Y==-1,:);
[m1,~]=size(A);
[m2,~]=size(B);
A=[A,ones(m1,1)];
B=[B,ones(m2,1)];

Kaa=Kernel(A,A,p);
Kab=Kernel(A,B,p);
Kbb=Kernel(B,B,p);
Kba=Kernel(B,A,p);


%按任务分出每个任务的正类和负类，
N=zeros(2,Tasknum);
for t=1:Tasknum
    N(1,t)=sum(yTrain{t}==1);
    N(2,t)=sum(yTrain{t}==-1);   
end

T1=[];%为kaa变化准备
T2=[];%为kbb变化准备
for t = 1 : Tasknum
        m = N(1,t);
        n = N(2,t);
        Tt1 = t*ones(m, 1);
        Tt2 = t*ones(n, 1);
        T1 = cat(1, T1, Tt1);%%%%按列连接
        T2 = cat(1, T2, Tt2);
end
PP= cell(Tasknum, 1);
QQ= cell(Tasknum, 1);
PQ= cell(Tasknum, 1);
QP= cell(Tasknum, 1);
for t = 1 : Tasknum
    Tt1 = T1==t;
    Tt2 = T2==t;
    PP{t} = Kaa(Tt1,Tt1);
    QQ{t} = Kbb(Tt2,Tt2);
    PQ{t} = Kab(Tt1,Tt2);
    QP{t} = Kba(Tt2,Tt1);
end
%生成正超平面的二次矩阵
aa=Cond(Kaa/mu+Tasknum*spblkdiag(PP{:}));
bb=Cond(Kbb/mu+Tasknum*spblkdiag(QQ{:}));
ab=Cond(Kab/mu+Tasknum*spblkdiag(PQ{:}));
ba=Cond(Kba/mu+Tasknum*spblkdiag(QP{:}));
H1=[aa,-aa;-aa,aa];
H2=[ab;-ab];
H3=bb;
large_H=[H1,-H2;-H2',H3];  %-------------修改过负号
%%生成负超平面的二次矩阵
Q1=[bb,-bb;-bb,bb];
Q2=[ba;-ba];
Q3=aa;
large_G=[Q1,Q2;Q2',Q3];
%参数转换
e1= ones(m1,1);
e2= ones(m2,1);
C1=[c1*e1;c1*e1;c2*e2];
C2=[c3*e2;c3*e2;c4*e1];
kk1=[ep*e1;ep*e1;-e2];
kk2=[ep*e2;ep*e2;-e1];
%二次规划
large_H=(large_H+large_H')/2;
large_G=(large_G+large_G')/2;
tic;
pai1=quadprog(large_H,kk1,[],[],[],[],zeros(2*m1+m2,1),C1);
pai2=quadprog(large_G,kk2,[],[],[],[],zeros(2*m2+m1,1),C2);
time=toc;
% 计算支持向量的值
% m11=findsv(pai1,m1);
% error_1=length(find(pai1(2*m1+1:length(pai1),:)<1e-5));
% m22=findsv(pai2,m2);
% error_2=length(find(pai2(2*m2+1:length(pai2),:)<1e-5));
% perti1=m11/m1;
% perti2=m22/m2;
% perti12=(m2-error_1)/m2;
% perti21=(m1-error_2)/m1;

precision=zeros(Tasknum,1);
for t = 1 : Tasknum
    Tt1 = T1==t;
    Tt2 = T2==t;
    Xt=xTest{t};
    [mtt,~]=size(Xt);
    Xt=[Xt,ones(mtt,1)];
    y_predict=zeros(mtt,1);
%     Ht1=exp(-(repmat(sum(Xt.*Xt,2)',m1,1)+repmat(sum(A.*A,2),1,mtt)-2*A*Xt')/(2*p^2));
%     Ht2=exp(-(repmat(sum(Xt.*Xt,2)',m2,1)+repmat(sum(B.*B,2),1,mtt)-2*B*Xt')/(2*p^2));
%       Ht1=Ht1';
%       Ht2=Ht2';
%       Ht1=(A*Xt').^p;
%       Ht2=(B*Xt').^p;
    Ht1=Kernel(Xt,A,p);
    Ht2=Kernel(Xt,B,p);
    y01=predict(Ht1, Ht2,pai1);
    y02=predict(Ht2,-Ht1,pai2);
    alpha1=pai1(1:m1);
    alpha2=pai1(m1+1:2*m1);
    alpha3=pai1(2*m1+1:end);
    pai1t=[alpha1(Tt1,:);alpha2(Tt1,:);alpha3(Tt2,:)];
    beta1=pai2(1:m2);
    beta2=pai2(m2+1:2*m2);
    beta3=pai2(2*m2+1:end);
    pai2t=[beta1(Tt2,:);beta2(Tt2,:);beta3(Tt1,:)];
    y1 =predict(Ht1(:,Tt1), Ht2(:,Tt2), pai1t);
    y2 =predict(Ht2(:,Tt2),-Ht1(:,Tt1), pai2t);
    yt1 = abs(mu\y01 + Tasknum*y1);
    yt2 = abs(mu\y02 + Tasknum*y2);
    y_predict(yt1<=yt2)=1;
    y_predict(yt1>yt2)=-1;
    err=sum(y_predict~=yTest{t});
    err=err/length(yTest{t});
    precision(t,1)=1-err;
end
accuracy = mean(precision);
function [ y ] = predict(E, F, Alpha)
        svi = Alpha~=0;
        hh=[E,-E,-F];    
        y = hh(:,svi)*Alpha(svi,:);

end
%end