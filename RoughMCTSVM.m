%function [precision,time]=RoughMCTSVM(X,Y,xTrain,yTrain,xTest,yTest,nu,deta,rho,p1)

% clc
% clear
 c=8;rho=8;p1=8;nu=0.4;deta=2;

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
C=[A;B];
%%
%核函数
for i=1:m1
    for j=1:m
        H(i,j)=Kerfun(A(i,:),C(j,:),p1);
    end
end
H=[H ones(m1,1)];
for i=1:m2
    for j=1:m
        G(i,j)=Kerfun(B(i,:),C(j,:),p1);
    end
end
G=[G ones(m2,1)];
%得到Q,R矩阵
Q=G/(H'*H+1e-5*eye(m+1))*G';
R=H/(G'*G+1e-5*eye(m+1))*H';
%得到P,S矩阵
TaskNum = length(xTrain);
N = zeros(2, TaskNum);
for t = 1 : TaskNum
        N(1, t) = sum(yTrain{t}==1);
        N(2, t) = sum(yTrain{t}==-1);
end
Ec=mat2cell(H,N(1,:));
Fc=mat2cell(G,N(2,:));
EEFc=cell(TaskNum, 1);%%%%空元胞，每个任务，个性
FFEc=cell(TaskNum, 1);
P=cell(TaskNum,1);%%%%空元胞，所有任务的个性组成，对角矩阵的构建
S=cell(TaskNum,1);
for t = 1 : TaskNum
    Et = Ec{t};
    Ft = Fc{t};
    [mt, ~] = size(Et'*Et);
    EEFc{t} =(Et'*Et+1e-5*eye(mt))\Ft';
    FFEc{t} =(Ft'*Ft+1e-5*eye(mt))\Et';%%%%Pt=Bt*inv(At'*At)Bt';个性矩阵的构造
    P{t} = Ft*EEFc{t};%%%正类
    S{t} = Et*FFEc{t};%%%负类
end
P = spblkdiag(P{:});
S = spblkdiag(S{:});%%%%分块对角矩阵
%%
%计算 U & V & ut &vt
Sym = @(H) (H+H')/2 + 1e-5*speye(size(H));
H1 = Sym(Q+(TaskNum/rho)*P);
H2 = Sym(R+(TaskNum/rho)*S);
% f1=zeros(m2,1);
% A1=-ones(1,m2);
b=-2*nu;
e2=ones(m2,1);
e1=ones(m1,1);
% f2=zeros(m1,1);
% A2=-ones(1,m1);
solver = struct('Display', 'off');
tic
Alpha = quadprog(H1,[],-e2',b,[],[],zeros(m2, 1),(deta/m2)*e2,[],solver);
Gamma = quadprog(H2,[],-e1',b,[],[],zeros(m1, 1),(deta/m1)*e1,[],solver);
time=toc;
CAlpha=mat2cell(Alpha,N(2,:));%%%%将矩阵转换为元胞数组，按任务量来划分。3个任务就是每个任务的维度，用来求解个性
CGamma=mat2cell(Gamma,N(1,:));
u=-(H'*H+1e-5*speye(m+1))\(G'*Alpha);%%%共性的值
v=(G'*G+1e-5*speye(m+1))\(H'*Gamma);
U=cell(TaskNum,1);
V=cell(TaskNum,1);
for t = 1 : TaskNum
    U{t}=u+(-(TaskNum/rho)*EEFc{t}*CAlpha{t});%%%%U=u+ut
    V{t}=v+((TaskNum/rho)*FFEc{t}*CGamma{t});
end
%%
% 预测的程序
Precision=zeros(TaskNum,1);
for t = 1 :TaskNum
    Xt=xTest{t};
    [mtt,~]=size(Xt);
    KAt = [];
    for i=1:mtt
        for j=1:m
            KAt(i,j)=Kerfun(Xt(i,:),C(j,:),p1);
        end
    end
    KAt=[KAt ones(mtt,1)];
    d1=abs(KAt*U{t})/norm(U{t}(1:end-1));
    d2=abs(KAt*V{t})/norm(V{t}(1:end-1));
    yt=sign(d2-d1);
    yt(yt==0)=1;
    Precision(t,1)=1-sum(yt~=yTest{t})/length(yTest{t});
end
precision=mean(Precision);