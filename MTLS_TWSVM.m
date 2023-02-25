%  function [precision,time]=myMTLS_TWSVM(X,Y,xTrain,yTrain,xTest,yTest,c,rho,p1) 
%MTLS_TWSVM 此处显示有关此函数的摘要
% Multi-Task Least Square Twin Support Vector Machine
%   此处显示详细说明
 c=0.5;rho=0.5;p1=0.5;

epsilon = 1e-5;
%%
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
%e1=ones(m1,1);
%e2=ones(m2.1);
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
Q=F/(E'*E+1e-10*eye(m+1))*F';
R=E/(F'*F+1e-10*eye(m+1))*E';
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
P=cell(TaskNum,1);
S=cell(TaskNum,1);
for t = 1 : TaskNum
    Et = Ec{t};
    Ft = Fc{t};
    [mt, ~] = size(Et'*Et);
    EEFc{t} =(Et'*Et+epsilon*eye(mt))\Ft';
    FFEc{t} =(Ft'*Ft+epsilon*eye(mt))\Et';
    P{t} = Ft*EEFc{t};
    S{t} = Et*FFEc{t};
end
P = spblkdiag(P{:});%P=spblkdiag(P{:});
S = spblkdiag(S{:});%S=spblkdiag(S{:});
%%
%计算 U & V & ut &vt
[mq,~]=size(Q);
[mr,~]=size(R);
tic
Alpha=(Q+(TaskNum/rho)*P+(1/c)*speye(mq))\ones(m2,1);
Gamma=(R+(TaskNum/rho)*S+(1/c)*speye(mr))\ones(m1,1);
time=toc;
CAlpha=mat2cell(Alpha,N(2,:));
CGamma=mat2cell(Gamma,N(1,:));
u=-(E'*E+epsilon*speye(m+1))\(F'*Alpha);
v=(F'*F+epsilon*speye(m+1))\(E'*Gamma);
U=cell(TaskNum,1);
V=cell(TaskNum,1);
for t = 1 : TaskNum
    U{t}=u+(-(TaskNum/rho)*EEFc{t}*CAlpha{t});
    V{t}=v+((TaskNum/rho)*FFEc{t}*CGamma{t});
end

%%
% 预测的程序
Precision=zeros(TaskNum,1);
for t = 1 :TaskNum
    Et=xTest{t};
    [mtt,~]=size(Et);
    KAt=exp(-(repmat(sum(Et.*Et,2)',m,1)+repmat(sum(C.*C,2),1,mtt)-2*C*Et')/(2*p1^2));
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

