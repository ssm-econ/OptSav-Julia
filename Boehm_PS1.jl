using CompEcon;
using Distributions;
using LinearAlgebra;
using Plots;
tol=0.0001;
ny=3;
ymin=2;
ymax=6;
na=100;
amin=-20;
amax=60;
sigma=0.75;
beta=0.9;

function u(x)
    if x<=0
        return -Inf
    else 
        return (1/(1-1/sigma))*x^(1-1/sigma)
    end
end

ygrid=LinRange(ymin,ymax,ny)
agrid=LinRange(amin,amax,na)
U=ones(na,1)
itmax=1000;
R=1/beta-0.00215;

Vnew=ones(na,ny)
ap_indpol=ones(na,ny)
Ptrans=fill(1/3,ny,1)

it=0;
dif=1;

function iterate_V(Vnew)
    it=0;
    dif=1;
    while dif>=tol && it <=itmax
    it=it+1;
    V=copy(Vnew);


        for i=1:na
            for j=1:ny

                U=u.(ygrid[j] + agrid[i] .- agrid/R)
                Vnew[i,j]=findmax(U+beta*V*Ptrans)[1]
                ap_indpol[i,j]=findmax(U+beta*V*Ptrans)[2][1]

            end
        end

        dif=maximum(maximum(abs.(V-Vnew)))
        print([it dif])    
    end
    return Vnew,ap_indpol
end

Val_f,pol_f=iterate_V(Vnew)

plot(agrid,agrid[Int.(pol_f)][:,1],label="y=2")
plot!(agrid,agrid[Int.(pol_f)][:,2],label="y=4")
plot!(agrid,agrid[Int.(pol_f)][:,3],label="y=6")

plot(agrid,Val_f[:,1],label="y=2")
plot!(agrid,Val_f[:,2],label="y=4")
plot!(agrid,Val_f[:,3],label="y=6")



function F(a,a_prime,y)
    u(y+a-a_prime/R)
end

fspace=fundefn(:cheb,[na,ny],[amin ymin],[amax ymax])
Phi=funbase(fspace);


fspace2=fundefn(:cheb,na,amin,amax)
Phi2=funbase(fspace2)
inv_Phi=inv(Matrix(Phi))
inv_Phi2=inv(Matrix(Phi2))
P_kron_I=LinearAlgebra.kron(1/3*ones(ny,ny),I(na))

c1_new=ones(na*ny)
dif=10;
it=1;
while dif>tol && it<itmax
    it=it+1;
    c1=copy(c1_new);
    s=[funeval(c1,fspace,[agrid[i],ygrid[j]])[1][1] for i=1:na,j=1:ny]
    Ev=s*Ptrans
    M=[findmax(F.(agrid[i],agrid,ygrid[j]) + beta*Ev)[1] for i=1:na,j=1:ny]
    M_argmax=[findmax(F.(agrid[i],agrid,ygrid[j]) + beta*Ev)[2] for i=1:na,j=1:ny]
    c1_new=inv_Phi*[M[:,1];M[:,2];M[:,3]]
    dif=findmax(abs.(c1_new-c1))[1]
end

function get_val_f(a,i)
    return funeval(c1,fspace,[a,ygrid[i]])[1][1]
end

plot(agrid,get_val_f.(agrid,1))
plot!(agrid,get_val_f.(agrid,2))
plot!(agrid,get_val_f.(agrid,3))

function get_pol_f(i,j)
    return M_argmax[i,j]
end

plot(agrid,get_pol_f())
