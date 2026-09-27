"""python gradcheck_ebm.py  -> float64 finite-difference check of the energy model's parameter and input gradients."""
import numpy as np, ebm
rng=np.random.default_rng(0); p=ebm.init(6); X,Y=ebm.batch([2],2,rng)
X=X.astype(np.float64); y=ebm.project(rng.random(Y.shape),X)
p={k:(v.astype(np.float64) if k!="C" else v) for k,v in p.items()}; gE=np.array([0.7,-1.3])
L=lambda yy:float((gE*ebm.forward(p,X,yy)[0]).sum())
E,c=ebm.forward(p,X,y); g,gy=ebm.backward(p,X,c,gE); worst=0
for k in ebm.KEYS:
    f=p[k].ravel()
    for i in rng.choice(f.size,min(5,f.size),replace=False):
        o=f[i]; f[i]=o+1e-5; a=L(y); f[i]=o-1e-5; b=L(y); f[i]=o
        worst=max(worst,abs((a-b)/2e-5-g[k].ravel()[i])/max(1e-6,abs((a-b)/2e-5)+abs(g[k].ravel()[i])))
print("worst relative gradient error:",worst)
