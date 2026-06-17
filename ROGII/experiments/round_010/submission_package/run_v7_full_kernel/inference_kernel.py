"""ROGII inference-only kernel — loads pre-trained models and runs prediction.

Does NOT train any models! Loads from dataset:
- lgb_model.txt (LightGBM native format)
- cat_model.cbm (CatBoost native format)
- feat_cols.pkl (pickled list of feature columns)
"""
# %%
import os, time, warnings, pickle, json
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from numba import njit
import joblib
import lightgbm as lgb
from catboost import CatBoostRegressor

# Discover Kaggle input path (varies by competition: some at /kaggle/input/<slug>,
# others at /kaggle/input/competitions/<slug>).
def _find_kaggle_input(slug="rogii-wellbore-geology-prediction"):
    base = f"/kaggle/input/competitions/{slug}"
    if os.path.isdir(base): return base
    base = f"/kaggle/input/{slug}"
    if os.path.isdir(base): return base
    return None  # not on Kaggle

INPUT_DIR = _find_kaggle_input() or "/Users/liucong/code/kaggle/ROGII/rogii-wellbore-geology-prediction"
TRAIN_DIR = f"{INPUT_DIR}/train"
TEST_DIR  = f"{INPUT_DIR}/test"
OUT_PATH  = "/kaggle/working/submission.csv" if _find_kaggle_input() else "/Users/liucong/code/kaggle/ROGII/results/round_010/submission_inference.csv"

def _find_dataset_input(*slugs, local=None):
    for slug in slugs:
        for base in (
            f"/kaggle/input/{slug}",
            f"/kaggle/input/datasets/{slug}",
        ):
            if os.path.isdir(base):
                return base
    if local and os.path.isdir(local):
        return local
    return None

RAVAGHI_DIR = _find_dataset_input(
    "wellbore-geology-prediction-artifacts",
    "ravaghi/wellbore-geology-prediction-artifacts",
    local="/Users/liucong/code/kaggle/ROGII/experiments/public_resources/datasets_raw/ravaghi-artifacts",
)
V10_DIR = _find_dataset_input(
    "rogii-v10-fresh-artifacts",
    "thbdh5765/rogii-v10-fresh-artifacts",
    local="/Users/liucong/code/kaggle/ROGII/experiments/public_resources/datasets_raw/thbdh5765_rogii-v10-fresh-artifacts",
)

print(f"INPUT_DIR = {INPUT_DIR}")
print(f"RAVAGHI_DIR = {RAVAGHI_DIR}")
print(f"V10_DIR = {V10_DIR}")
assert os.path.isdir(TRAIN_DIR), f"TRAIN_DIR missing: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR),  f"TEST_DIR  missing: {TEST_DIR}"
assert RAVAGHI_DIR and os.path.isdir(RAVAGHI_DIR), "RAVAGHI_DIR missing"
assert V10_DIR and os.path.isdir(V10_DIR), "V10_DIR missing"

ROLLING_WINS = [5, 21, 51, 101]

RUN_V7_WEIGHTS = {
    "c20_r9_pf128_full": 0.5028569013483001,
    "v10_tabicl_A": 0.19441129734773116,
    "c54_ravaghi_cat2": 0.09492358689965985,
    "c50_ravaghi_lgb1": 0.060044610924396336,
    "c52_ravaghi_lgb3": 0.04821063418364803,
    "c51_ravaghi_lgb2": 0.04564549095942647,
    "c53_ravaghi_cat1": 0.016237222462336894,
    "v10_tabicl_B": 0.01279830211826902,
}
RAVAGHI_MODELS = {
    "c50_ravaghi_lgb1": "models/lightgbm-1/lgbmregressor_trainer_20260526182612.pkl",
    "c51_ravaghi_lgb2": "models/lightgbm-2/lgbmregressor_trainer_20260526190415.pkl",
    "c52_ravaghi_lgb3": "models/lightgbm-3/lgbmregressor_trainer_20260526192806.pkl",
    "c53_ravaghi_cat1": "models/catboost-1/catboostregressor_trainer_20260526193740.pkl",
    "c54_ravaghi_cat2": "models/catboost-2/catboostregressor_trainer_20260526194838.pkl",
}
V10_KEYS = ["lgb123", "lgb42", "lgb7", "cb42", "cb7", "cb123", "tabicl_A", "tabicl_B"]


# ── PF (numba JIT, verbatim from LB-7.776 kernel) ─────────────────────────────
PF_N=600; ANCC_N=600
PF_MOM=0.993; PF_VN=0.005; PF_PN=0.01
PF_GR_SIG_MIN=10.; PF_GR_SIG_MAX=60.; PF_GR_SIG_DEF=30.
PF_INIT_V_STD=0.02; PF_INIT_SPR=0.5; PF_RESAMP=0.5
PF_ROUGH_P=0.2; PF_ROUGH_V=0.003; PF_GR_WIN=5; PF_GR_WT=0.3
ANCC_ALPHA=0.998; ANCC_RN=0.002; ANCC_PN=0.005
ANCC_IR=0.01; ANCC_IS=4.5; ANCC_RP=0.1; ANCC_RR=0.001

# Ensemble PF tracks total log-likelihood + uses sp45 init spread (kernel pattern)
ENS_N_SEEDS = 128   # hill climb v3: PF 128-seed is the dominant ensemble member
ENS_SCALES  = [3.0, 5.0, 8.0, 12.0]
ENS_ANCC_IS = 4.5   # sp45 patch (kernel sel15-vb-best)


@njit(cache=True)
def _interp1(grid, v, vmin, step):
    i = int((v - vmin) / step)
    if i < 0: return grid[0]
    n = len(grid) - 1
    if i >= n: return grid[n]
    t = (v - vmin) / step - i
    return grid[i]*(1.-t) + grid[i+1]*t


@njit(cache=True)
def _resamp(pos, aux, w, N, rp, rv):
    cum = np.zeros(N+1)
    for j in range(N): cum[j+1]=cum[j]+w[j]
    u0=np.random.uniform(0.,1./N)
    np2=np.empty(N); na=np.empty(N); ci=0
    for j in range(N):
        u=u0+j/N
        while ci<N-1 and cum[ci+1]<u: ci+=1
        np2[j]=pos[ci]+rp*np.random.randn()
        na[j] =aux[ci]+rv*np.random.randn()
    return np2,na


@njit(cache=True)
def _pf_ancc(md_v,z_v,gr_v,gg,vmin,step,gs,ls,ir,N,ALPHA,RN,PN,IS,RP,RR,RESAMP):
    pos=np.empty(N); rate=np.empty(N); w=np.ones(N)/N
    for j in range(N):
        pos[j]=ls+IS*np.random.randn()
        rate[j]=ir+0.01*np.random.randn()
    pts=np.empty(len(md_v)); std_=np.empty(len(md_v)); pm=md_v[0]-1.
    for i in range(len(md_v)):
        dm=md_v[i]-pm; dm=max(dm,1.)
        for j in range(N):
            rate[j]=ALPHA*rate[j]+RN*np.random.randn()
            pos[j]+=rate[j]*dm+PN*np.random.randn()
            tvt_j=pos[j]-z_v[i]
            tvt_j=max(tvt_j,vmin-50.); tvt_j=min(tvt_j,vmin+len(gg)*step+50.)
            pos[j]=tvt_j+z_v[i]
        if not np.isnan(gr_v[i]):
            ws=0.
            for j in range(N):
                eg=_interp1(gg,pos[j]-z_v[i],vmin,step)
                d=(gr_v[i]-eg)/gs
                lk=max(np.exp(-0.5*d*d) if d*d<600. else 0.,1e-300)
                w[j]*=lk; ws+=w[j]
            if ws>0.:
                for j in range(N): w[j]/=ws
            else:
                for j in range(N): w[j]=1./N
        ne=0.
        for j in range(N): ne+=w[j]*w[j]
        if 1./ne<RESAMP*N:
            pos,rate=_resamp(pos,rate,w,N,RP,RR)
            for j in range(N): w[j]=1./N
        tv=0.
        for j in range(N): tv+=w[j]*(pos[j]-z_v[i])
        pts[i]=tv; va=0.
        for j in range(N): va+=w[j]*(pos[j]-z_v[i]-tv)**2
        std_[i]=va**0.5; pm=md_v[i]
    return pts,std_


@njit(cache=True)
def _pf_z(md_v,z_v,gr_v,gr_sm_v,gg_p,gg_s,vmin,step,gs,ip,iv,beta,icpt,zsig,N,
          MOM,VN,PN,GR_WT,RP,RV,RESAMP):
    pos=np.empty(N); vel=np.empty(N); w=np.ones(N)/N
    for j in range(N):
        pos[j]=ip+0.5*np.random.randn()
        vel[j]=iv+0.02*np.random.randn()
    pts=np.empty(len(md_v)); std_=np.empty(len(md_v)); pm=md_v[0]-1.; pz=z_v[0]-1.
    for i in range(len(md_v)):
        dm=md_v[i]-pm; dm=max(dm,1.)
        dzd=(z_v[i]-pz)/dm; ve=beta*dzd+icpt
        for j in range(N):
            vel[j]=MOM*vel[j]+VN*np.random.randn()
            pos[j]+=vel[j]*dm+PN*np.random.randn()
            pos[j]=max(pos[j],vmin-50.); pos[j]=min(pos[j],vmin+len(gg_p)*step+50.)
        if not np.isnan(gr_v[i]):
            ws=0.
            for j in range(N):
                ep=_interp1(gg_p,pos[j],vmin,step)
                dp=(gr_v[i]-ep)/gs
                lp=max(np.exp(-0.5*dp*dp) if dp*dp<600. else 0.,1e-300)
                if not np.isnan(gr_sm_v[i]):
                    es=_interp1(gg_s,pos[j],vmin,step)
                    ds=(gr_sm_v[i]-es)/(gs*1.5)
                    ls=max(np.exp(-0.5*ds*ds) if ds*ds<600. else 0.,1e-300)
                    lk=(1.-GR_WT)*lp+GR_WT*ls
                else: lk=lp
                lk=max(lk,1e-300); w[j]*=lk; ws+=w[j]
            if ws>0.:
                for j in range(N): w[j]/=ws
            else:
                for j in range(N): w[j]=1./N
        ws2=0.
        for j in range(N):
            dv=(vel[j]-ve)/max(zsig*2.,0.005)
            lz=max(np.exp(-0.5*dv*dv) if dv*dv<600. else 0.,1e-300)
            w[j]*=lz; ws2+=w[j]
        if ws2>0.:
            for j in range(N): w[j]/=ws2
        else:
            for j in range(N): w[j]=1./N
        ne=0.
        for j in range(N): ne+=w[j]*w[j]
        if 1./ne<RESAMP*N:
            pos,vel=_resamp(pos,vel,w,N,RP,RV)
            for j in range(N): w[j]=1./N
        wm=0.
        for j in range(N): wm+=w[j]*pos[j]
        pts[i]=wm; va=0.
        for j in range(N): va+=w[j]*(pos[j]-wm)**2
        std_[i]=va**0.5; pm=md_v[i]; pz=z_v[i]
    return pts,std_


def _grid(tw_tvt,tw_gr,step=0.2):
    tmin=float(tw_tvt.min()); tmax=float(tw_tvt.max())
    tvt_g=np.arange(tmin,tmax+step,step)
    return np.interp(tvt_g,tw_tvt,tw_gr).astype(np.float64),float(tmin),float(step)


def _gr_sig(hw,tw_tvt,tw_gr):
    kn=hw[hw['TVT_input'].notna()&hw['GR'].notna()]
    if len(kn)<20: return float(PF_GR_SIG_DEF)
    return float(np.clip(np.std(kn['GR'].values-np.interp(kn['TVT_input'].values,tw_tvt,tw_gr)),
                          PF_GR_SIG_MIN,PF_GR_SIG_MAX))


def run_pf_ancc(hw,tw_tvt,tw_gr,N=ANCC_N):
    gs=_gr_sig(hw,tw_tvt,tw_gr)
    kn=hw[hw['TVT_input'].notna()]; ev=hw[hw['TVT_input'].isna()]
    if len(ev)==0: return np.array([]),np.array([])
    ls=float(kn['TVT_input'].iloc[-1]+kn['Z'].iloc[-1])
    tail=kn.tail(30); dt=np.diff(tail['TVT_input'].values)
    dz=np.diff(tail['Z'].values); dm=np.diff(tail['MD'].values); m=dm>0
    ir=float(np.median((dt+dz)[m]/dm[m])) if m.sum()>=3 else 0.
    gg,gmin,gst=_grid(tw_tvt,tw_gr)
    # Pre-interpolate GR over the FULL well so lateral NaN gaps are filled
    # before slicing (matches training pattern; raw GR is ~32% NaN on lateral).
    gr_full = hw['GR'].interpolate(limit_direction='both').fillna(float(np.nanmean(tw_gr)))
    gr_v = gr_full.loc[ev.index].values.astype(np.float64)
    pts,std=_pf_ancc(ev['MD'].values.astype(np.float64),ev['Z'].values.astype(np.float64),
                      gr_v,gg,gmin,gst,
                      gs,ls,ir,N,ANCC_ALPHA,ANCC_RN,ANCC_PN,ANCC_IS,ANCC_RP,ANCC_RR,PF_RESAMP)
    return pts.astype(np.float32),std.astype(np.float32)


def run_pf_z(hw,tw_tvt,tw_gr,N=PF_N,PN=PF_PN):
    gs=_gr_sig(hw,tw_tvt,tw_gr)
    tw_s=pd.Series(tw_gr).rolling(PF_GR_WIN,center=True,min_periods=1).mean().values.astype(np.float32)
    kna=hw[hw['TVT_input'].notna()]; ev=hw[hw['TVT_input'].isna()]
    if len(ev)==0: return np.array([]),np.array([])
    dz_k=np.diff(kna['Z'].values); dvt=np.diff(kna['TVT_input'].values)
    dmd_k=np.diff(kna['MD'].values); m2=dmd_k>0
    if m2.sum()>=10:
        vz=dz_k[m2]/dmd_k[m2]; vt=dvt[m2]/dmd_k[m2]
        A=np.column_stack([vz,np.ones_like(vz)]); c,_,_,_=np.linalg.lstsq(A,vt,rcond=None)
        beta,icpt,zsig=float(c[0]),float(c[1]),max(float(np.std(vt-(c[0]*vz+c[1]))),0.001)
    else: beta,icpt,zsig=-1.,0.,0.1
    t2=kna.tail(20); dvt2=np.diff(t2['TVT_input'].values); dmd2=np.diff(t2['MD'].values); m3=dmd2>0
    iv=float(np.median(dvt2[m3]/dmd2[m3])) if m3.sum()>=3 else 0.
    gg,gmin,gst=_grid(tw_tvt,tw_gr)
    gs2,_,_=_grid(tw_tvt,tw_s)
    gr_sm=hw['GR'].rolling(PF_GR_WIN,center=True,min_periods=1).mean()
    pts,std=_pf_z(ev['MD'].values.astype(np.float64),ev['Z'].values.astype(np.float64),
                   ev['GR'].values.astype(np.float64),
                   gr_sm.loc[ev.index].values.astype(np.float64),
                   gg,gs2,gmin,gst,gs,float(kna['TVT_input'].iloc[-1]),iv,
                   beta,icpt,zsig,N,
                   PF_MOM,PF_VN,PN,PF_GR_WT,PF_ROUGH_P,PF_ROUGH_V,PF_RESAMP)
    return pts.astype(np.float32),std.astype(np.float32)



# ── PF ensemble JIT (sp45 init + GR preinterp + cumulative log-lik) ───────────
@njit(cache=True)
def _pf_ancc_ll(md_v,z_v,gr_v,gg,vmin,step,gs,ls,ir,N,ALPHA,RN,PN,IS,RP,RR,RESAMP):
    """Same as _pf_ancc but with sp45 init spread and returning total log-lik."""
    pos=np.empty(N); rate=np.empty(N); w=np.ones(N)/N
    for j in range(N):
        pos[j]=ls+IS*np.random.randn()
        rate[j]=ir+0.01*np.random.randn()
    pts=np.empty(len(md_v)); pm=md_v[0]-1.; cum_ll=0.
    for i in range(len(md_v)):
        dm=md_v[i]-pm; dm=max(dm,1.)
        for j in range(N):
            rate[j]=ALPHA*rate[j]+RN*np.random.randn()
            pos[j]+=rate[j]*dm+PN*np.random.randn()
            tvt_j=pos[j]-z_v[i]
            tvt_j=max(tvt_j,vmin-50.); tvt_j=min(tvt_j,vmin+len(gg)*step+50.)
            pos[j]=tvt_j+z_v[i]
        if not np.isnan(gr_v[i]):
            ws=0.; avg_lk=0.
            for j in range(N):
                eg=_interp1(gg,pos[j]-z_v[i],vmin,step)
                d=(gr_v[i]-eg)/gs
                lk=max(np.exp(-0.5*d*d) if d*d<600. else 0.,1e-300)
                avg_lk+=w[j]*lk
                w[j]*=lk; ws+=w[j]
            cum_ll += np.log(max(avg_lk,1e-300))
            if ws>0.:
                for j in range(N): w[j]/=ws
            else:
                for j in range(N): w[j]=1./N
        ne=0.
        for j in range(N): ne+=w[j]*w[j]
        if 1./ne<RESAMP*N:
            pos,rate=_resamp(pos,rate,w,N,RP,RR)
            for j in range(N): w[j]=1./N
        tv=0.
        for j in range(N): tv+=w[j]*(pos[j]-z_v[i])
        pts[i]=tv; pm=md_v[i]
    return pts, cum_ll


def run_pf_ensemble(hw, tw_tvt, tw_gr, n_seeds=ENS_N_SEEDS, N=ANCC_N):
    """16-seed log-lik-weighted PF ensemble, kernel pattern.

    Returns dict with one TVT array per scale + the uniform mean,
    each shaped [n_ev]. Returns None if no lateral rows.
    """
    kn = hw[hw['TVT_input'].notna()]; ev = hw[hw['TVT_input'].isna()]
    if len(ev)==0 or len(kn)==0: return None
    gs = _gr_sig(hw, tw_tvt, tw_gr)
    ls = float(kn['TVT_input'].iloc[-1] + kn['Z'].iloc[-1])
    tail = kn.tail(30); dt=np.diff(tail['TVT_input'].values)
    dz=np.diff(tail['Z'].values); dm=np.diff(tail['MD'].values); m=dm>0
    ir = float(np.median((dt+dz)[m]/dm[m])) if m.sum()>=3 else 0.
    gg, gmin, gst = _grid(tw_tvt, tw_gr)
    # Pre-interpolate GR over the FULL well (kernel pattern)
    gr_full = hw['GR'].interpolate(limit_direction='both').fillna(float(np.nanmean(tw_gr)))
    gr_v = gr_full.loc[ev.index].values.astype(np.float64)
    md_v = ev['MD'].values.astype(np.float64)
    z_v  = ev['Z'].values.astype(np.float64)

    preds = np.empty((n_seeds, len(ev)), dtype=np.float64)
    lls   = np.empty(n_seeds, dtype=np.float64)
    for s in range(n_seeds):
        np.random.seed(s)
        pts, ll = _pf_ancc_ll(md_v, z_v, gr_v, gg, gmin, gst,
                               gs, ls, ir, N,
                               ANCC_ALPHA, ANCC_RN, ANCC_PN, ENS_ANCC_IS, ANCC_RP, ANCC_RR, PF_RESAMP)
        # _pf_ancc_ll returns (pos - z) i.e. TVT directly when interpreted as ANCC anchor
        # (legacy contract from kernel). pts here = TVT prediction.
        preds[s] = pts
        lls[s]   = ll

    lls_n = lls - lls.max()
    out = {}
    for sc in ENS_SCALES:
        wts = np.exp(lls_n / float(sc)); wts /= wts.sum()
        out[f'pf_ens_s{int(sc)}'] = (wts[:, None] * preds).sum(0).astype(np.float32)
    out['pf_ens_mean'] = preds.mean(0).astype(np.float32)
    return out



# ── Beam search (numba JIT, 14 configs) ───────────────────────────────────────
BEAM_CONFIGS = [
    (10, 20.0, 144.0, 2),(10,  8.0,  64.0, 2),( 8, 35.0, 220.0, 1),
    (10, 14.0,  90.0, 5),(20,  4.0,  36.0, 3),(12, 12.0, 100.0, 3),
    (15, 25.0, 180.0, 2),(20, 30.0, 200.0, 2),(15, 10.0,  80.0, 4),
    (25,  6.0,  50.0, 3),(10, 40.0, 300.0, 1),(12, 18.0, 120.0, 5),
    (30,  8.0,  70.0, 2),(10, 50.0, 400.0, 0),
]


@njit(cache=True)
def _beam_jit(sgr, tw_gr, si, BS, mc, es):
    n=len(sgr); nt=len(tw_gr); MAX=BS*6
    bidx=np.zeros(BS,np.int64); bidx[0]=si
    bcost=np.full(BS,1e30);     bcost[0]=0.; bn=np.int64(1)
    hI=np.zeros((n,BS),np.int64); hP=np.zeros((n,BS),np.int64)
    cI=np.zeros(MAX,np.int64); cC=np.full(MAX,1e30); cP=np.zeros(MAX,np.int64)
    for step in range(n):
        gv=sgr[step]; nc=np.int64(0)
        for bi in range(bn):
            idx=bidx[bi]; cost=bcost[bi]
            for d in range(-2,3):
                ni=idx+d
                if ni<0 or ni>=nt: continue
                tot=cost+(gv-tw_gr[ni])**2/es+mc*(d if d>=0 else -d)
                fnd=np.int64(-1)
                for ci in range(nc):
                    if cI[ci]==ni: fnd=ci; break
                if fnd>=0:
                    if tot<cC[fnd]: cC[fnd]=tot; cP[fnd]=bi
                else:
                    if nc<MAX: cI[nc]=ni; cC[nc]=tot; cP[nc]=bi; nc+=1
        kept=min(BS,nc)
        for i in range(kept):
            mi=i
            for j in range(i+1,nc):
                if cC[j]<cC[mi]: mi=j
            if mi!=i:
                cI[i],cI[mi]=cI[mi],cI[i]
                cC[i],cC[mi]=cC[mi],cC[i]
                cP[i],cP[mi]=cP[mi],cP[i]
        hI[step,:kept]=cI[:kept]; hP[step,:kept]=cP[:kept]
        bidx[:kept]=cI[:kept]; bcost[:kept]=cC[:kept]; bn=kept
    best=np.int64(0)
    for b in range(1,bn):
        if bcost[b]<bcost[best]: best=b
    path=np.zeros(n,np.int64); b=best
    for s in range(n-1,-1,-1): path[s]=hI[s,b]; b=hP[s,b]
    return path


def _nn(arr,v):
    i=int(np.searchsorted(arr,v,'left'))
    if i>=len(arr): return len(arr)-1
    if i>0 and abs(arr[i-1]-v)<=abs(arr[i]-v): return i-1
    return i


def _smooth(vals,fb,r):
    s=pd.Series(vals,dtype='float32').interpolate(limit_direction='both').fillna(fb)
    return (s.rolling(r*2+1,center=True,min_periods=1).mean() if r>0 else s).to_numpy(np.float32)


def beam_search(gr_h,tw_tvt,tw_gr,start_tvt,bs,mc,es,r):
    si=_nn(tw_tvt,start_tvt)
    sgr=_smooth(gr_h,float(np.nanmean(tw_gr)),r).astype(np.float64)
    path=_beam_jit(sgr,tw_gr.astype(np.float64),si,bs,float(mc),float(es))
    return tw_tvt[path].astype(np.float32)


def run_beam_all(hw, tw):
    kn = hw[hw['TVT_input'].notna()]; ev = hw[hw['TVT_input'].isna()]
    if len(ev) == 0 or len(kn) == 0: return None, None
    last_tvt = float(kn.iloc[-1]['TVT_input'])
    tw_s = tw.sort_values('TVT')
    tw_tvt = tw_s['TVT'].values.astype(np.float64)
    tw_gr  = tw_s['GR'].fillna(tw_s['GR'].mean()).values.astype(np.float64)
    gr_all = hw['GR'].interpolate(limit_direction='both').fillna(tw_gr.mean()).values.astype(np.float64)
    hgr = gr_all[ev.index]
    paths = np.zeros((len(ev), len(BEAM_CONFIGS)), dtype=np.float32)
    for k, (bs, mc, es, r) in enumerate(BEAM_CONFIGS):
        paths[:, k] = beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r)
    return paths, ev.index.values


# ── Base features (no formation columns) ──────────────────────────────────────
def safe_savgol(x, win=51, order=2):
    if len(x) <= win: return x.copy()
    return savgol_filter(x, win, order)


def extract_well_features(wid, data_dir):
    """Combined extractor: base + PF + Beam → joined per-lateral-row DF."""
    try:
        hw = pd.read_csv(f"{data_dir}/{wid}__horizontal_well.csv")
        tw = pd.read_csv(f"{data_dir}/{wid}__typewell.csv")
    except Exception as e:
        print(f"⚠ 井 {wid} 跳过：CSV 读取失败 ({type(e).__name__}: {e})")
        return None
    if len(hw) < 50:
        print(f"⚠ 井 {wid} 跳过：horizontal_well 行数 < 50 (={len(hw)})")
        return None
    md=hw["MD"].values; x=hw["X"].values; y=hw["Y"].values; z=hw["Z"].values
    tvt_inp = hw["TVT_input"].values
    gr_raw  = hw["GR"].values

    mask_lat = np.isnan(tvt_inp)
    if mask_lat.sum() == 0:
        print(f"⚠ 井 {wid} 跳过：无 lateral 行 (TVT_input 全部 non-NaN)")
        return None
    known = ~mask_lat
    if known.sum() < 10:
        print(f"⚠ 井 {wid} 跳过：known 段 < 10 行 (={int(known.sum())})")
        return None
    last_idx = np.flatnonzero(known)[-1]
    last_tvt = float(tvt_inp[last_idx])
    last_z = float(z[last_idx]); last_md=float(md[last_idx])
    last_x = float(x[last_idx]); last_y=float(y[last_idx])

    gr_clean = pd.Series(gr_raw).interpolate(limit_direction="both").bfill().ffill().values
    if np.all(np.isnan(gr_clean)): gr_clean = np.zeros_like(z)
    gr_smooth51 = safe_savgol(gr_clean, 51, 2)
    last_gr = float(gr_smooth51[last_idx])

    gr_s = pd.Series(gr_clean)
    rolls = {}
    for w in ROLLING_WINS:
        r = gr_s.rolling(w, center=True, min_periods=1)
        rolls[f"gr_mean_{w}"] = r.mean().values
        rolls[f"gr_std_{w}"]  = r.std().fillna(0).values

    dmd=np.gradient(md); dz=np.gradient(z); dx=np.gradient(x); dy=np.gradient(y)
    nmz = np.sqrt(dmd**2+dz**2)+1e-8; nxy=np.sqrt(dx**2+dy**2)+1e-8
    sin_dmd_dz=dz/nmz; cos_dmd_dz=dmd/nmz
    sin_dx_dy =dy/nxy; cos_dx_dy =dx/nxy

    neg_dz = np.zeros_like(z)
    for i in range(last_idx+1, len(z)):
        neg_dz[i] = neg_dz[i-1] + (-(z[i]-z[i-1]))

    lat_idx = np.flatnonzero(mask_lat)
    n_known = int(known.sum()); n_lat = int(mask_lat.sum())

    # PF (single, used as the dominant features)
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].values.astype(np.float64)
    tw_gr  = tw_s["GR"].fillna(tw_s["GR"].mean()).values.astype(np.float64)
    np.random.seed(42)
    try:
        pf_a, pf_a_std = run_pf_ancc(hw, tw_tvt, tw_gr)
        pf_z_, pf_z_std= run_pf_z(hw,  tw_tvt, tw_gr)
    except Exception as e:
        print(f"⚠ 井 {wid} 跳过：PF 单 seed 异常 ({type(e).__name__}: {e})")
        return None
    if len(pf_a) != len(lat_idx):
        print(f"⚠ 井 {wid} 跳过：PF 单 seed 长度不匹配 (pf_a={len(pf_a)} vs lat_idx={len(lat_idx)})")
        return None

    # PF ensemble (16 seeds × 4 scales). Failure here is non-fatal — we fall
    # back to single-seed only (rare; would require a numerical blowup).
    try:
        ens = run_pf_ensemble(hw, tw_tvt, tw_gr)
    except Exception as e:
        print(f"⚠ 井 {wid} PF ensemble 异常 ({type(e).__name__}: {e})")
        ens = None
    if ens is None or len(ens['pf_ens_s12']) != len(lat_idx):
        print(f"⚠ 井 {wid} PF ensemble 回退到 single-seed")
        extract_well_features._ens_fallback_count = getattr(extract_well_features, "_ens_fallback_count", 0) + 1
        # graceful fallback: copy single-seed PF into all ensemble slots
        ens = {f'pf_ens_s{int(s)}': pf_a.astype(np.float32) for s in ENS_SCALES}
        ens['pf_ens_mean'] = pf_a.astype(np.float32)

    # Beam
    try:
        paths, _ = run_beam_all(hw, tw)
    except Exception as e:
        print(f"⚠ 井 {wid} 跳过：beam search 异常 ({type(e).__name__}: {e})")
        return None
    if paths is None or len(paths) != len(lat_idx):
        n_paths = "None" if paths is None else len(paths)
        print(f"⚠ 井 {wid} 跳过：beam search 长度不匹配 (paths={n_paths} vs lat_idx={len(lat_idx)})")
        return None

    beam_mean = paths.mean(1); beam_std = paths.std(1); beam_med=np.median(paths,1)
    beam_rng  = paths.max(1) - paths.min(1)
    beam_cons = paths[:, 0]; beam_sm5 = paths[:, 3]

    rows = []
    for k, r in enumerate(lat_idx):
        rec = {
            "well": wid, "row_idx": int(r),
            "md_offset": float(md[r]-last_md),
            "z_rel": float(z[r]-last_z),
            "x_rel": float(x[r]-last_x),
            "y_rel": float(y[r]-last_y),
            "cumsum_neg_dz": float(neg_dz[r]),
            "sin_dmd_dz": float(sin_dmd_dz[r]),
            "cos_dmd_dz": float(cos_dmd_dz[r]),
            "sin_dx_dy":  float(sin_dx_dy[r]),
            "cos_dx_dy":  float(cos_dx_dy[r]),
            "gr_smooth": float(gr_smooth51[r]),
            "gr_diff_from_last": float(gr_smooth51[r]-last_gr),
            "last_known_tvt": last_tvt,
            "last_known_z":   last_z,
            "last_known_gr":  last_gr,
            "n_known_rows":  n_known,
            "n_lateral_rows": n_lat,
            "row_position_norm": float((r-last_idx)/max(n_lat,1)),
            # PF
            "pf_ancc_std": float(pf_a_std[k]),
            "pf_z_std":    float(pf_z_std[k]),
            "pf_ancc_offset": float(pf_a[k]  - last_tvt),
            "pf_z_offset":    float(pf_z_[k] - last_tvt),
            "pf_disagreement": float(pf_a[k] - pf_z_[k]),
            "pf_mean_offset":  float(0.5*((pf_a[k]-last_tvt)+(pf_z_[k]-last_tvt))),
            # Beam
            "beam_mean_offset": float(beam_mean[k] - last_tvt),
            "beam_med_offset":  float(beam_med[k]  - last_tvt),
            "beam_cons_offset": float(beam_cons[k] - last_tvt),
            "beam_sm5_offset":  float(beam_sm5[k]  - last_tvt),
            "beam_vs_pf": float((beam_mean[k]-last_tvt) - 0.5*((pf_a[k]-last_tvt)+(pf_z_[k]-last_tvt))),
            # PF ensemble (16 seeds, 4 scales — Phase 14B)
            "pf_ens_s3_offset":   float(ens['pf_ens_s3'][k]   - last_tvt),
            "pf_ens_s5_offset":   float(ens['pf_ens_s5'][k]   - last_tvt),
            "pf_ens_s8_offset":   float(ens['pf_ens_s8'][k]   - last_tvt),
            "pf_ens_s12_offset":  float(ens['pf_ens_s12'][k]  - last_tvt),
            "pf_ens_mean_offset": float(ens['pf_ens_mean'][k] - last_tvt),
            "pf_ens_vs_ancc":      float((ens['pf_ens_s12'][k] - last_tvt) - (pf_a[k] - last_tvt)),
            "pf_ens_scale_disag":  float((ens['pf_ens_s3'][k]  - last_tvt) - (ens['pf_ens_s12'][k] - last_tvt)),
            # Keep absolutes for v9 heuristic blend (not used as model features)
            "pf_ancc_abs": float(pf_a[k]),
            "pf_ens_s12_abs": float(ens['pf_ens_s12'][k]),
        }
        for kk, arr in rolls.items(): rec[kk] = float(arr[r])
        # Target only available in train
        if not np.isnan(hw["TVT"].iloc[r] if "TVT" in hw.columns else np.nan):
            rec["target"] = float(hw["TVT"].iloc[r] - last_tvt)
        else:
            rec["target"] = np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def _warmup_numba():
    md=np.linspace(1,50,20,np.float64); z=np.zeros(20,np.float64)
    gr=np.full(20,50.,np.float64); gg=np.linspace(45,55,100,np.float64)
    _pf_ancc(md,z,gr,gg,45.,0.1,20.,50.,0.,8,0.998,0.002,0.005,0.3,0.1,0.001,0.5)
    _pf_z(md,z,gr,gr,gg,gg,45.,0.1,20.,50.,0.,-1.,0.,0.1,8,0.993,0.005,0.01,0.3,0.2,0.003,0.5)
    _pf_ancc_ll(md,z,gr,gg,45.,0.1,20.,50.,0.,8,0.998,0.002,0.005,4.5,0.1,0.001,0.5)
    _beam_jit(np.random.randn(30), np.random.randn(50), 25, 8, 15., 100.)


# %%
def load_v10_offsets(test):
    diag = Path(V10_DIR) / "diagnostics"
    csv_path = diag / "test_base_predictions.csv"
    if not csv_path.exists():
        csv_path = diag / "test_base_predictions.csv.gz"
    if not csv_path.exists():
        raise FileNotFoundError(f"v10 test predictions missing under {diag}")
    v10 = pd.read_csv(csv_path)
    expected = ["id"] + V10_KEYS
    if list(v10.columns) != expected:
        raise ValueError(f"unexpected v10 columns: {v10.columns.tolist()}")
    v10["well"] = v10["id"].str.rsplit("_", n=1).str[0]
    v10["row_idx"] = v10["id"].str.rsplit("_", n=1).str[1].astype("int32")

    base = test[["well", "row_idx"]].copy()
    base["id"] = base["well"] + "_" + base["row_idx"].astype(str)
    aligned = base.merge(v10, on=["well", "row_idx", "id"], how="left", validate="one_to_one")
    missing = int(aligned[V10_KEYS].isna().any(axis=1).sum())
    print(f"  v10 alignment: rows={len(aligned):,} missing={missing}")
    if missing:
        raise ValueError(f"missing v10 predictions for {missing} rows")
    return {
        "v10_tabicl_A": aligned["tabicl_A"].to_numpy(np.float32),
        "v10_tabicl_B": aligned["tabicl_B"].to_numpy(np.float32),
    }


def load_ravaghi_offsets(test):
    import ravaghi_features

    print("  building ravaghi-schema test features")
    t0 = time.time()
    rav = ravaghi_features.build_test_features(INPUT_DIR)
    if len(rav) == 0:
        raise ValueError("ravaghi feature builder returned no rows")
    rav["well"] = rav["well"].astype(str)
    rav["row_idx"] = rav["id"].str.rsplit("_", n=1).str[1].astype("int32")

    base = test[["well", "row_idx"]].copy()
    base["_order"] = np.arange(len(base), dtype=np.int32)
    rav = base.merge(rav, on=["well", "row_idx"], how="left", validate="one_to_one").sort_values("_order")
    if len(rav) != len(test):
        raise ValueError(f"ravaghi alignment row count mismatch: {len(rav)} vs {len(test)}")
    missing_cols = [c for c in ["id", "last_known_tvt"] if c not in rav.columns or rav[c].isna().any()]
    if missing_cols:
        raise ValueError(f"ravaghi alignment missing columns/values: {missing_cols}")
    print(f"  ravaghi features: rows={len(rav):,} cols={rav.shape[1]} built in {time.time()-t0:.0f}s")

    out = {}
    for cid, rel in RAVAGHI_MODELS.items():
        pkl_path = Path(RAVAGHI_DIR) / rel
        if not pkl_path.exists():
            raise FileNotFoundError(pkl_path)
        trainer = joblib.load(pkl_path)
        feature_names = None
        for est in getattr(trainer, "estimators", []):
            if hasattr(est, "feature_names_in_"):
                feature_names = list(est.feature_names_in_)
                break
            if hasattr(est, "feature_names_"):
                feature_names = list(est.feature_names_)
                break
            if hasattr(est, "booster_"):
                feature_names = list(est.booster_.feature_name())
                break
        if not feature_names:
            feature_names = [c for c in rav.columns if c not in {"well", "id", "target", "row_idx", "_order"}]
        missing = [c for c in feature_names if c not in rav.columns]
        if missing:
            raise ValueError(f"{cid} missing {len(missing)} ravaghi features, first={missing[:10]}")
        pred = trainer.predict(rav[feature_names]).astype(np.float32)
        out[cid] = pred
        print(f"    {cid:18s} mean={pred.mean():.3f} std={pred.std():.3f} range=[{pred.min():.1f},{pred.max():.1f}]")
    return out


def main():
    t0 = time.time()
    print("=== ROGII v9: Inference-only with pre-trained models ===\n")

    # Debug: Check input directory structure
    print("[DEBUG] Checking Kaggle input directory structure:")
    if os.path.exists('/kaggle/input'):
        for item in os.listdir('/kaggle/input'):
            item_path = os.path.join('/kaggle/input', item)
            print(f"[DEBUG]   {item} -> {item_path}")
            if os.path.isdir(item_path):
                try:
                    subitems = os.listdir(item_path)
                    for subitem in subitems[:20]:
                        subitem_path = os.path.join(item_path, subitem)
                        size_str = f" ({os.path.getsize(subitem_path)} bytes)" if os.path.isfile(subitem_path) else ""
                        print(f"[DEBUG]     - {subitem}{size_str}")
                except Exception as e:
                    print(f"[DEBUG]     [ERROR listing {item_path}: {e}]")

    for label, path in [("RAVAGHI_DIR", RAVAGHI_DIR), ("V10_DIR", V10_DIR)]:
        print(f"\n[DEBUG] Checking {label}: {path}")
        try:
            for f in sorted(os.listdir(path))[:30]:
                fpath = os.path.join(path, f)
                size = os.path.getsize(fpath) if os.path.isfile(fpath) else 0
                print(f"[DEBUG]     - {f} ({size} bytes)")
        except Exception as e:
            print(f"[DEBUG]   [ERROR listing {path}: {e}]")

    print("\n[DEBUG] === End of debug info ===\n")

    print("[1/4] Numba warmup")
    np.random.seed(0); _warmup_numba()

    print(f"\n[2/4] Checking mounted artifacts")
    t1 = time.time()
    print(f"  ✓ PF feature code embedded in notebook")
    print(f"  ✓ ravaghi artifacts: {RAVAGHI_DIR}")
    print(f"  ✓ v10 artifacts: {V10_DIR}")
    print(f"  Artifact check time: {time.time()-t1:.0f}s")

    print(f"\n[3/4] Building PF128 TEST features")
    t2 = time.time()
    test_wells = sorted({f.replace("__horizontal_well.csv","")
                         for f in os.listdir(TEST_DIR)
                         if f.endswith("__horizontal_well.csv")})
    print(f"  test wells: {len(test_wells)}")
    dft = []
    for i, wid in enumerate(test_wells):
        df_w = extract_well_features(wid, TEST_DIR)
        if df_w is None:
            print(f"  ! {wid}: feature build failed"); continue
        dft.append(df_w)
        if (i+1) % 100 == 0:
            print(f"  {i+1}/{len(test_wells)} | rows so far: {sum(len(d) for d in dft):,} | "
                  f"{time.time()-t2:.0f}s", flush=True)
    test = pd.concat(dft, ignore_index=True)
    print(f"  test rows: {len(test):,} | feat build {time.time()-t2:.0f}s")
    ens_fb_test = getattr(extract_well_features, "_ens_fallback_count", 0)
    print(f"  test PF ensemble fallbacks: {ens_fb_test} wells")

    print(f"\n[4/4] Generating full run_v7 predictions")
    t3 = time.time()

    # Every component is in offset space (target = TVT - last_known_tvt), matching
    # the OOF candidates used by run_v7_with_v10_tabicl. Add last_known_tvt once.
    components = {
        "c20_r9_pf128_full": (test["pf_ens_s12_abs"].values - test["last_known_tvt"].values).astype(np.float32),
    }
    components.update(load_v10_offsets(test))
    components.update(load_ravaghi_offsets(test))

    wsum = float(sum(RUN_V7_WEIGHTS.values()))
    print(f"  raw run_v7 weight sum: {wsum:.6f}")
    blend_off = np.zeros(len(test), dtype=np.float64)
    for cid, w in RUN_V7_WEIGHTS.items():
        if cid not in components:
            raise KeyError(f"missing component: {cid}")
        arr = components[cid].astype(np.float64)
        blend_off += (w / wsum) * arr
        print(f"    {cid:20s} w={w/wsum:.4f} mean={arr.mean():.3f} std={arr.std():.3f}")

    pred_tvt = test["last_known_tvt"].values.astype(np.float64) + blend_off
    print(f"  blend offset mean: {blend_off.mean():.3f}  std: {blend_off.std():.3f}")
    print(f"  blend tvt mean: {pred_tvt.mean():.0f}  range: [{pred_tvt.min():.0f}, {pred_tvt.max():.0f}]")
    print(f"  Prediction time: {time.time()-t3:.0f}s")

    sub = pd.DataFrame({
        "id":  test["well"] + "_" + test["row_idx"].astype(str),
        "tvt": pred_tvt.astype(np.float32),
    })

    # Align to sample_submission
    sample_path = f"{INPUT_DIR}/sample_submission.csv"
    if not os.path.exists(sample_path):
        # Kaggle may mount at different depths
        for candidate in [f"/kaggle/input/rogii-wellbore-geology-prediction/sample_submission.csv",
                          f"/kaggle/input/competitions/rogii-wellbore-geology-prediction/sample_submission.csv"]:
            if os.path.exists(candidate):
                sample_path = candidate
                break
    if os.path.exists(sample_path):
        sample = pd.read_csv(sample_path)
        sub = sample[["id"]].merge(sub, on="id", how="left")
        n_missing = int(sub["tvt"].isna().sum())
        if n_missing:
            missing_ids = sub.loc[sub["tvt"].isna(), "id"].head(10).tolist()
            raise ValueError(f"missing predictions for {n_missing} sample rows, first={missing_ids}")
    sub.to_csv(OUT_PATH, index=False)
    print(f"  → {OUT_PATH}  ({len(sub)} rows)")
    print(sub.head().to_string(index=False))
    print(f"\n  pred stats: min={pred_tvt.min():.0f}, max={pred_tvt.max():.0f}, "
          f"mean={pred_tvt.mean():.0f}, median={np.median(pred_tvt):.0f}")
    print(f"\nTotal wall time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
