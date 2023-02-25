%  function [precision,time]=MTBSVM(X,Y,xTrain,yTrain,xTest,yTest,c,C3,rho,p1) 
% MT-TBSVM 此处显示有关此函数的摘要
% Multi-Task Twin Bounded Support Vector Machine
spHeart = load('...\data\isolet-ab.mat');
X = spHeart.X1; Y = spHeart.Y1;xTrain=spHeart.x1Train; yTrain=spHeart.y1Train;xTest = spHeart.x1Test; yTest = spHeart.y1Test;
c=0.01;rho=1;p1=2;C3=1e-5;
epsilon = 1e-10;
%%构造矩阵A,B
[m,n] = size(X);
L1=find(Y==1);
L2=find(Y==-1);
p=length(L1);
q=length(L2);
A=[];B=[];
A=X(L1,:);
B=X(L2,:);
[m1, ~] = size(A);
[m2, ~] = size(B);
C=[A;B];
%%
%核函数
E=exp(-(repmat(sum(A.*A,2)',m,1)+repmat(sum(C.*C,2),1,p)-2*C*A')/(2*p1^2));
E=E';
E=[E ones(m1,1)];
F=exp(-(repmat(sum(B.*B,2)',m,1)+repmat(sum(C.*C,2),1,q)-2*C*B')/(2*p1^2));
F=F';
F=[F ones(m2,1)];
%得到Q,R矩阵
 I = speye(size(E, 2));
  EEF = (E'*E+C3*I)\F';
   FFE = (F'*F+C3*I)\E';
   Q = F*EEF;
   R = E*FFE;
%得到P,S矩阵
TaskNum = length(xTrain);
N = zeros(2, TaskNum);
for t = 1 : TaskNum
        N(1, t) = sum(yTrain{t}==1);
        N(2, t) = sum(yTrain{t}==-1);
end
Ec=mat2cell(E,N(1,:));
Fc=mat2cell(F,N(2,:));
EEFc=cell(TaskNum, 1);
FFEc=cell(TaskNum, 1);
% P = sparse(0, 0); 
% S = sparse(0, 0);
P=cell(TaskNum,1);%%%%空元胞，所有任务的个性组成，对角矩阵的构建
S=cell(TaskNum,1);
 for t = 1 : TaskNum
    Et = Ec{t}; Ft = Fc{t}; It = speye(size(Et, 2));
    EEFt{t} = (rho/TaskNum*(Et'*Et)+C3/TaskNum*It)\(Ft');
    FFEt{t} = (rho/TaskNum*(Ft'*Ft)+C3/TaskNum*It)\(Et');
%     P = blkdiag(P, Ft*EEFt{t});
%     S = blkdiag(S, Et*FFEt{t});
  P{t} = Ft*EEFt{t};%%%正类
    S{t} = Et*FFEt{t};%%%负类
 end
P = spblkdiag(P{:});
S = spblkdiag(S{:});
 %%
%计算 U & V & ut &vt
e1=ones(m1,1);
e2=ones(m2,1);
Sym = @(H) (H+H')/2+epsilon*speye(size(H));
solver = struct('Display', 'off');
tic
 Alpha = quadprog(Sym(Q + P),-e2,[],[],[],[],zeros(m2, 1),c*e2,[],solver);
 Gamma = quadprog(Sym(R + S),-e1,[],[],[],[],zeros(m1, 1),c*e1,[],solver);
 time=toc;
 CAlpha = mat2cell(Alpha, N(2,:));
 CGamma = mat2cell(Gamma, N(1,:));
   u = -EEF*Alpha;
  v = FFE*Gamma;
  U = cell(TaskNum, 1);
  V = cell(TaskNum, 1);
  for t = 1 : TaskNum
        U{t} = u - EEFt{t}*CAlpha{t};
        V{t} = v + FFEt{t}*CGamma{t};
 end

%%
% 预测的程序
Precision=zeros(TaskNum,1);
for t = 1 :TaskNum
    Xt=xTest{t};
    [mtt,~]=size(Xt);
    KAt=exp(-(repmat(sum(Xt.*Xt,2)',m,1)+repmat(sum(C.*C,2),1,mtt)-2*C*Xt')/(2*p1^2));
    KAt=[KAt' ones(mtt,1)];
    d1=abs(KAt*U{t})/norm(U{t}(1:end-1));
    d2=abs(KAt*V{t})/norm(V{t}(1:end-1));
    yt=sign(d2-d1);
    yt(yt==0)=1;
    err=sum(yt~=yTest{t});
    err=err/length(yTest{t});
    Precision(t,1)=1-err;
end
precision=mean(Precision);
% end