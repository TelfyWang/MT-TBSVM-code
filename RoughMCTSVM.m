%function [precision,time]=RoughMCTSVM(X,Y,xTrain,yTrain,xTest,yTest,nu,deta,rho,p1)

% clc
% clear
 c=8;rho=8;p1=8;nu=0.4;deta=2;

%%
%%�������A,B
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
%�˺���
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
%�õ�Q,R����
Q=G/(H'*H+1e-5*eye(m+1))*G';
R=H/(G'*G+1e-5*eye(m+1))*H';
%�õ�P,S����
TaskNum = length(xTrain);
N = zeros(2, TaskNum);
for t = 1 : TaskNum
        N(1, t) = sum(yTrain{t}==1);
        N(2, t) = sum(yTrain{t}==-1);
end
Ec=mat2cell(H,N(1,:));
Fc=mat2cell(G,N(2,:));
EEFc=cell(TaskNum, 1);%%%%��Ԫ����ÿ�����񣬸���
FFEc=cell(TaskNum, 1);
P=cell(TaskNum,1);%%%%��Ԫ������������ĸ�����ɣ��ԽǾ���Ĺ���
S=cell(TaskNum,1);
for t = 1 : TaskNum
    Et = Ec{t};
    Ft = Fc{t};
    [mt, ~] = size(Et'*Et);
    EEFc{t} =(Et'*Et+1e-5*eye(mt))\Ft';
    FFEc{t} =(Ft'*Ft+1e-5*eye(mt))\Et';%%%%Pt=Bt*inv(At'*At)Bt';���Ծ���Ĺ���
    P{t} = Ft*EEFc{t};%%%����
    S{t} = Et*FFEc{t};%%%����
end
P = spblkdiag(P{:});
S = spblkdiag(S{:});%%%%�ֿ�ԽǾ���
%%
%���� U & V & ut &vt
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
CAlpha=mat2cell(Alpha,N(2,:));%%%%������ת��ΪԪ�����飬�������������֡�3���������ÿ�������ά�ȣ�����������
CGamma=mat2cell(Gamma,N(1,:));
u=-(H'*H+1e-5*speye(m+1))\(G'*Alpha);%%%���Ե�ֵ
v=(G'*G+1e-5*speye(m+1))\(H'*Gamma);
U=cell(TaskNum,1);
V=cell(TaskNum,1);
for t = 1 : TaskNum
    U{t}=u+(-(TaskNum/rho)*EEFc{t}*CAlpha{t});%%%%U=u+ut
    V{t}=v+((TaskNum/rho)*FFEc{t}*CGamma{t});
end
%%
% Ԥ��ĳ���
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