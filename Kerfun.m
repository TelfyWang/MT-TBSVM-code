function k=Kerfun(u,v,p1)
k=exp(-norm((u-v),1)/p1);
% k=exp(-(u-v)*(u-v)'/(2*p1^2));
end