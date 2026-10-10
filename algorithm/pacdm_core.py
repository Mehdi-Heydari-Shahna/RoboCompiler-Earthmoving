"""Path-Assembly Closure Differential Mapping with angular-first SE(3) twists.

Point cuts use three auxiliary orientation coordinates that carry no inertia
or actuator effort. The solver uses adjoint transport, logarithmic derivatives,
fixed-row defect homotopy and full physical closure/rank acceptance gates.
"""
import numpy as np
from scipy.linalg import qr, expm
from scipy.spatial.transform import Rotation


def skew(a):
    x,y,z=a
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def inv(T):
    out=np.eye(4);out[:3,:3]=T[:3,:3].T
    out[:3,3]=-out[:3,:3]@T[:3,3]
    return out


def adj(T):
    R=T[:3,:3]
    return np.block([[R,np.zeros((3,3))],[skew(T[:3,3])@R,R]])


def exp(x):
    w,v=x[:3],x[3:];t=np.linalg.norm(w);K=skew(w)
    if t<1e-4:
        a=1-t*t/6+t**4/120;b=.5-t*t/24+t**4/720;c=1/6-t*t/120+t**4/5040
    else:
        a=np.sin(t)/t;b=(1-np.cos(t))/t**2;c=(t-np.sin(t))/t**3
    T=np.eye(4);T[:3,:3]+=a*K+b*K@K;T[:3,3]=(np.eye(3)+b*K+c*K@K)@v
    return T


def log(T):
    R=T[:3,:3];u=np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])/2
    s=np.linalg.norm(u);theta=np.arctan2(s,np.clip((np.trace(R)-1)/2,-1,1))
    if np.pi-theta<1e-6:raise ValueError('SE3 logarithm near pi branch')
    w=u if s<1e-14 else theta/s*u;K=skew(w)
    beta=1/12+theta**2/720+theta**4/30240 if theta<1e-4 else (1-theta/2/np.tan(theta/2))/theta**2
    return np.r_[w,(np.eye(3)-K/2+beta*K@K)@T[:3,3]]


def inv_left(x):
    W,V=skew(x[:3]),skew(x[3:]);Z=np.block([[W,np.zeros((3,3))],[V,W]])
    if np.linalg.norm(Z)<=1e-3:
        Z2=Z@Z
        return np.eye(6)-Z/2+Z2/12-Z2@Z2/720
    A=np.zeros((12,12));A[:6,:6]=Z;A[:6,6:]=np.eye(6)
    return np.linalg.solve(expm(A)[:6,6:],np.eye(6))


def rank(A):
    s=np.linalg.svd(A,compute_uv=False)
    return int(np.sum(s>1e-10*s[0])) if len(s) and s[0] else 0


def select(J):
    n=J.shape[1];r=rank(J)
    if n == 0:return np.array([],int),1.,r
    if r<n:return np.array([],int),0.,r
    rows=qr(J.T,pivoting=True,mode='economic')[2][:n]
    return rows,1/np.linalg.cond(J[rows],p=1),r


class PointGraph:
    """Physical tree plus spherical cut paths derived from point constraints.

    Rotation chart is initialized from the caller's seed, without changing
    any body frame, anchor, or physical coordinate. This keeps the free
    orientation chart local and nonsingular. Full physical SE3 residuals
    (including redundant rows) remain mandatory at acceptance.
    """
    def __init__(self,cmg,seed):
        self.cmg=cmg;self.ids=list(cmg['coordinate_ids']);self.nt=len(self.ids)
        self.nc=len(cmg['closures']);self.n=self.nt+3*self.nc
        self.root=cmg['root_body'];self.joints=cmg['joints'];self.cuts=cmg['closures']
        self.active=np.array([self.ids.index(x) for x in cmg['independent_ids']],int)
        self.passive=np.setdiff1d(np.arange(self.n),self.active)
        self.lower=np.full(self.n,-np.pi);self.upper=np.full(self.n,np.pi)
        records={j['id']:j for j in self.joints}
        for i,k in enumerate(self.ids):
            self.lower[i]=records[k]['limits']['lower'];self.upper[i]=records[k]['limits']['upper']
        self.edges=[];done={self.root}
        while len(self.edges)<len(self.joints):
            before=len(self.edges)
            for j in self.joints:
                if j['base_body'] in done and j['follower_body'] not in done:
                    self.edges.append((j,np.asarray(j['T_BJ']),inv(np.asarray(j['T_FJ'])),
                                       self.ids.index(j['id']) if j['type']!='fixed' else None))
                    done.add(j['follower_body'])
            if len(self.edges)==before:raise ValueError('Not a directed connected tree')
        p,_=self.poses(np.r_[seed,np.zeros(3*self.nc)])
        self.references=[p[c['body1']][:3,:3].T@p[c['body2']][:3,:3] for c in self.cuts]
        self.paths=[]
        for c in self.cuts:
            def chain(body):
                result=[]
                while body!=self.root:
                    j=next(j for j in self.joints if j['follower_body']==body)
                    result.append(j['id']);body=j['base_body']
                return result[::-1]
            self.paths.append({'cut':c['id'],'root_to_body1':chain(c['body1']),
                               'root_to_body2':chain(c['body2']),
                               'cut_type':'point with three free orientation chart coordinates'})

    def augment(self,q):
        return np.r_[q,np.zeros(3*self.nc)]

    def lift(self,q):
        """Set only free cut orientations from current physical body poses."""
        out=self.augment(q);P,_=self.poses(out)
        for i,c in enumerate(self.cuts):
            R=self.references[i].T@P[c['body1']][:3,:3].T@P[c['body2']][:3,:3]
            out[self.nt+3*i:self.nt+3*i+3]=Rotation.from_matrix(R).as_euler('XYZ')
        return out

    def poses(self,q):
        P={self.root:np.eye(4)};E={self.root:np.zeros((6,self.n))}
        for j,enter,leave,k in self.edges:
            prefix=P[j['base_body']]@enter;eta=E[j['base_body']].copy();motion=np.eye(4)
            if k is not None:
                twist=np.r_[j['axis'],[0,0,0]] if j['type']=='revolute' else np.r_[[0,0,0],j['axis']]
                motion=exp(twist*q[k]);eta[:,k]+=adj(prefix)@twist
            P[j['follower_body']]=prefix@motion@leave;E[j['follower_body']]=eta
        return P,E

    def residual(self,q,defects=None):
        P,E=self.poses(q);rs=[];js=[];deltas=[]
        for i,c in enumerate(self.cuts):
            A=np.eye(4);A[:3,3]=c['point1_m'];A[:3,:3]=self.references[i]
            Tminus=P[c['body1']]@A;eminus=E[c['body1']].copy()
            for k,axis in enumerate(np.eye(3)):
                col=self.nt+3*i+k;twist=np.r_[axis,[0,0,0]]
                eminus[:,col]+=adj(Tminus)@twist;Tminus=Tminus@exp(twist*q[col])
            B=np.eye(4);B[:3,3]=c['point2_m']
            Tplus=P[c['body2']]@B;eplus=E[c['body2']]
            D=np.eye(4) if defects is None else defects[i]
            reverse=inv(Tminus@D);delta=reverse@Tplus;r=log(delta)
            rs.append(r);js.append(inv_left(r)@adj(reverse)@(eplus-eminus));deltas.append(delta)
        if not self.cuts:
            return np.zeros(0),np.zeros((0,self.n)),np.zeros((0,4,4))
        return np.concatenate(rs),np.vstack(js),np.asarray(deltas)


class PACDM:
    def __init__(self,graph):self.g=graph

    @staticmethod
    def physical_ok(r,D):
        if r.size == 0:return bool(D.size == 0)
        return bool(np.all(np.isfinite(r)) and np.max(abs(r))<=1e-8 and
                    np.max(np.linalg.norm(D[:,:3,3],axis=1))<=1e-8 and
                    np.max(np.linalg.norm(r.reshape(-1,6)[:,:3],axis=1))<=1e-8)

    def mapping(self,q):
        g=self.g;r,J,D=g.residual(q);rows,rc,rp=select(J[:,g.passive]);rf=rank(J)
        info={'success':False,'rank_full':rf,'rank_passive':rp,'expected_rank':len(g.passive),
              'rcond':rc,'rows':rows.tolist(),'residual_inf':float(np.max(abs(r))) if r.size else 0.}
        if not self.physical_ok(r,D):info['message']='Full physical closure failed';return None,info
        if rf!=len(g.passive) or rp!=len(g.passive) or rc<1e-10:
            info['message']='Physical rank/conditioning does not support independent coordinates';return None,info
        N=np.zeros((g.n,len(g.active)));N[g.active]=np.eye(len(g.active))
        N[g.passive]=-np.linalg.solve(J[np.ix_(rows,g.passive)],J[np.ix_(rows,g.active)])
        tangent=J@N
        info.update(success=True,message='Physical closure and rank passed',tangent_residual=float(np.max(abs(tangent))) if tangent.size else 0.)
        return N,info

    def correct(self,qa,p,D,rows,maxiter=100,full=True):
        g=self.g;q=np.zeros(g.n);q[g.active]=qa;q[g.passive]=p;mu=1e-6
        info={'success':False,'message':'Iteration limit','iterations':0}
        if np.any(q<g.lower) or np.any(q>g.upper):info['message']='Outside branch bounds';return q,info
        if len(g.passive) == 0:
            r,_,Delta=g.residual(q,D)
            info.update(success=self.physical_ok(r,Delta),residual_inf=float(np.max(abs(r))) if r.size else 0.)
            info['message']='Converged' if info['success'] else 'No dependent coordinates can close the residual'
            return q,info
        for iteration in range(maxiter+1):
            r,J,Delta=g.residual(q,D);A=J[np.ix_(rows,g.passive)];f=r[rows]
            info.update(iterations=iteration,residual_inf=float(np.max(abs(r))) if r.size else 0.)
            if 1/np.linalg.cond(A,p=1)<1e-10:info['message']='Ill-conditioned selected block';return q,info
            if np.max(abs(f))<=1e-9:
                info['success']=not full or self.physical_ok(r,Delta)
                info['message']='Converged' if info['success'] else 'Selected rows closed but full physical closure failed'
                return q,info
            dp=-np.linalg.solve(A.T@A+mu*np.eye(len(p)),A.T@f)
            if np.max(abs(dp))<1e-12:info['message']='Stagnation';return q,info
            trial=q.copy();trial[g.passive]+=dp
            if np.all(trial>=g.lower) and np.all(trial<=g.upper) and np.linalg.norm(g.residual(trial,D)[0][rows])<np.linalg.norm(f):
                q=trial;mu=max(1e-12,mu/10)
            else:
                mu=min(1e12,mu*10)
                if mu==1e12:info['message']='No admissible descent';return q,info
        return q,info

    def acquire(self,qa,seed):
        g=self.g;q=seed.copy();q[g.active]=qa
        d0=g.residual(q)[0].reshape(-1,6)
        def defects(lam):return np.asarray([exp((1-lam)*x) for x in d0])
        rows,rc,rp=select(g.residual(q,defects(0))[1][:,g.passive])
        info={'success':False,'lambda':0.,'accepted_steps':0,'rejected_steps':0,'rows':rows.tolist()}
        if rp!=len(g.passive) or rc<1e-10:info['message']='Singular artificial start';return q,info
        step=.1
        while info['lambda']<1:
            lam=min(1.,info['lambda']+step)
            if 1-lam<1e-14:lam=1.
            candidate,ci=self.correct(qa,q[g.passive],defects(lam),rows,full=lam==1.)
            if ci['success']:
                q=candidate;info['lambda']=lam;info['accepted_steps']+=1;step=min(.1,step*2)
            else:
                info['rejected_steps']+=1;step/=2
                if step<1e-5:info['message']=ci['message'];return q,info
        _,mi=self.mapping(q);info.update(success=mi['success'],message=mi['message'],mapping=mi)
        return q,info

    def track(self,independent,seed):
        q,ai=self.acquire(independent[0],seed)
        if not ai['success']:raise ValueError(str(ai))
        N,mi=self.mapping(q);out=[q.copy()];fallback=0
        for qa in independent[1:]:
            predicted=q[self.g.passive]+N[self.g.passive]@(qa-q[self.g.active])
            candidate,ci=self.correct(qa,predicted,None,np.array(mi['rows']),maxiter=8)
            if not ci['success']:
                fallback+=1;candidate,ci=self.acquire(qa,q)
            if not ci['success']:raise ValueError(str(ci))
            q=candidate;N,mi=self.mapping(q)
            if not mi['success']:raise ValueError(str(mi))
            out.append(q.copy())
        return np.asarray(out),{'initial_acquisition':ai,'fallback_count':fallback,'samples':len(out)}
