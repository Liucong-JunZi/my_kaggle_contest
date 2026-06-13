"""ROGII Kaggle submission notebook (self-contained).

Clean LGB stack: base geometry/GR + PF (ancc + z) + Beam ensemble.
Test-safe features only — no formation columns (`hw[ANCC/EGFDU/...]` is
labeled in train but absent at test).

CV per-well RMSE: 10.26, flat 13.21 (GroupKFold-5 on 723 train wells).

Kaggle paths:
  /kaggle/input/rogii-wellbore-geology-prediction/{train,test}/
  /kaggle/working/submission.csv

Wall time: ~10-15 min on Kaggle CPU.
"""
# %%
import os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from numba import njit
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
OUT_PATH  = "/kaggle/working/submission.csv" if _find_kaggle_input() else "/Users/liucong/code/kaggle/ROGII/results/round_008/submission_local.csv"

print(f"INPUT_DIR = {INPUT_DIR}")
assert os.path.isdir(TRAIN_DIR), f"TRAIN_DIR missing: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR),  f"TEST_DIR  missing: {TEST_DIR}"

ROLLING_WINS = [5, 21, 51, 101]


# ── PF (numba JIT, verbatim from LB-7.776 kernel) ─────────────────────────────
PF_N=600; ANCC_N=600
PF_MOM=0.993; PF_VN=0.005; PF_PN=0.01
PF_GR_SIG_MIN=10.; PF_GR_SIG_MAX=60.; PF_GR_SIG_DEF=30.
PF_INIT_V_STD=0.02; PF_INIT_SPR=0.5; PF_RESAMP=0.5
PF_ROUGH_P=0.2; PF_ROUGH_V=0.003; PF_GR_WIN=5; PF_GR_WT=0.3
ANCC_ALPHA=0.998; ANCC_RN=0.002; ANCC_PN=0.005
ANCC_IR=0.01; ANCC_IS=0.3; ANCC_RP=0.1; ANCC_RR=0.001


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
    pts,std=_pf_ancc(ev['MD'].values.astype(np.float64),ev['Z'].values.astype(np.float64),
                      ev['GR'].values.astype(np.float64),gg,gmin,gst,
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


# Multi-PN scales for spatial-rich features (Phase 9 / Phase 11)
PN_SCALES = [0.005, 0.010, 0.030, 0.080]
PN_LABELS = ["pn005","pn010","pn030","pn080"]


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
    except Exception:
        return None
    if len(hw) < 50: return None
    md=hw["MD"].values; x=hw["X"].values; y=hw["Y"].values; z=hw["Z"].values
    tvt_inp = hw["TVT_input"].values
    gr_raw  = hw["GR"].values

    mask_lat = np.isnan(tvt_inp)
    if mask_lat.sum() == 0: return None
    known = ~mask_lat
    if known.sum() < 10: return None
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
    except Exception:
        return None
    if len(pf_a) != len(lat_idx): return None

    # Multi-PN PF (4 process-noise scales for high-dip well rescue)
    pf_z_multi = {}     # label -> array
    pf_z_multi_std = {}
    for label, PN in zip(PN_LABELS, PN_SCALES):
        np.random.seed(42)
        try:
            pts, std_ = run_pf_z(hw, tw_tvt, tw_gr, PN=PN)
            if len(pts) != len(lat_idx):
                return None
            pf_z_multi[label] = pts
            pf_z_multi_std[label] = std_
        except Exception:
            return None

    # Beam
    try:
        paths, _ = run_beam_all(hw, tw)
    except Exception:
        return None
    if paths is None or len(paths) != len(lat_idx): return None

    beam_mean = paths.mean(1); beam_std = paths.std(1); beam_med=np.median(paths,1)
    beam_rng  = paths.max(1) - paths.min(1)
    beam_cons = paths[:, 0]; beam_sm5 = paths[:, 3]

    rows = []
    for k, r in enumerate(lat_idx):
        rec = {
            "well": wid, "row_idx": int(r),
            "x_abs": float(x[r]),  # for spatial lookup; dropped before training
            "y_abs": float(y[r]),  # for spatial lookup; dropped before training
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
        }
        # Multi-PN PF features
        pn_offsets = []
        for label in PN_LABELS:
            off = float(pf_z_multi[label][k] - last_tvt)
            rec[f"pf_z_{label}_offset"] = off
            rec[f"pf_z_std_{label}"]    = float(pf_z_multi_std[label][k])
            pn_offsets.append(off)
        rec["pf_pn_span"] = float(max(pn_offsets) - min(pn_offsets))
        rec["pf_pn_mean"] = float(np.mean(pn_offsets))
        rec["pf_pn_std"]  = float(np.std(pn_offsets))
        rec["pf_high_vs_default"] = float(rec["pf_z_pn080_offset"] - rec["pf_z_pn010_offset"])
        rec["pf_low_vs_default"]  = float(rec["pf_z_pn005_offset"] - rec["pf_z_pn010_offset"])
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
    _beam_jit(np.random.randn(30), np.random.randn(50), 25, 8, 15., 100.)


# ── Spatial-neighbor features (cKDTree over all train known segments) ────
from scipy.spatial import cKDTree

K_SP_NEIGHBORS = 5
K_SP_QUERY = 200       # need >= max same-well prefix length to break out of self-cluster
K_SP_CHUNK = 100_000   # chunked query keeps peak RAM ~1.5GB on Kaggle 16GB


def build_spatial_pool(train_wells, train_dir):
    """Pool all (X, Y, TVT_input) from known segments of all train wells."""
    rows = []
    for wid in train_wells:
        try:
            h = pd.read_csv(f"{train_dir}/{wid}__horizontal_well.csv")
        except Exception:
            continue
        mask = h["TVT_input"].notna()
        if mask.sum() < 10:
            continue
        rows.append(pd.DataFrame({
            "x":   h["X"].values[mask],
            "y":   h["Y"].values[mask],
            "tvt": h["TVT_input"].values[mask],
            "well": wid,
        }))
    pool = pd.concat(rows, ignore_index=True)
    return pool


def add_spatial_features(df, tree, pool_well_arr, pool_tvt):
    """Augment a feature DF (x_abs, y_abs, well present) with neighbor stats.

    For each row, query K_SP_QUERY nearest from the cKDTree, mask self-well
    entries, take K_SP_NEIGHBORS best, compute median/mean/std/count/distance.
    Chunked to bound peak RAM (Kaggle 16GB).
    Returns DF with appended columns.
    """
    xy = df[["x_abs", "y_abs"]].values
    own_well = np.asarray(df["well"].astype(object).to_numpy(), dtype=object)
    pool_well_arr = np.asarray(pool_well_arr, dtype=object)
    n = len(df)
    print(f"  spatial: querying tree for {n:,} rows in chunks of {K_SP_CHUNK:,} ...", flush=True)

    medians = np.full(n, np.nan, dtype=np.float32)
    means   = np.full(n, np.nan, dtype=np.float32)
    stds    = np.full(n, np.nan, dtype=np.float32)
    n_valid_arr = np.zeros(n, dtype=np.float32)
    d_min_arr   = np.full(n, np.nan, dtype=np.float32)

    for start in range(0, n, K_SP_CHUNK):
        end = min(start + K_SP_CHUNK, n)
        dist, idx = tree.query(xy[start:end], k=K_SP_QUERY, workers=-1)
        is_self = (pool_well_arr[idx] == own_well[start:end, None])
        dist_m = np.where(is_self, np.inf, dist)
        order  = np.argsort(dist_m, axis=1)
        idx_s  = np.take_along_axis(idx, order, axis=1)
        dist_s = np.take_along_axis(dist_m, order, axis=1)
        idx_top  = idx_s[:, :K_SP_NEIGHBORS]
        dist_top = dist_s[:, :K_SP_NEIGHBORS]
        valid = np.isfinite(dist_top)
        tvt_top = pool_tvt[idx_top]
        n_valid_arr[start:end] = valid.sum(axis=1).astype(np.float32)
        d_min_arr[start:end]   = dist_top[:, 0].astype(np.float32)
        for i in range(end - start):
            v = valid[i]
            if v.sum() == 0: continue
            t = tvt_top[i, v]
            medians[start + i] = float(np.median(t))
            means[start + i]   = float(np.mean(t))
            stds[start + i]    = float(np.std(t)) if v.sum() > 1 else 0.0
        if (start // K_SP_CHUNK) % 5 == 0:
            print(f"    spatial chunk {end:,}/{n:,}", flush=True)
        del dist, idx, is_self, dist_m, order, idx_s, dist_s, idx_top, dist_top, valid, tvt_top

    df = df.copy()
    df["neighbor_tvt_median"] = medians
    df["neighbor_tvt_mean"]   = means
    df["neighbor_tvt_std"]    = stds
    df["neighbor_count"]      = n_valid_arr
    df["neighbor_dist_min"]   = d_min_arr
    df["neighbor_tvt_offset"] = df["neighbor_tvt_median"] - df["last_known_tvt"]
    df["neighbor_dist_log"]   = np.log1p(df["neighbor_dist_min"].clip(0, 1e6).fillna(1e6))
    return df


# %%
def main():
    t0 = time.time()
    print("=== ROGII clean LGB+CAT submission (Phase 11: + multi-PN PF + spatial) ===\n")

    print("[1/5] Numba warmup")
    np.random.seed(0); _warmup_numba()

    print(f"\n[2/5] Building TRAIN features (full corpus)")
    train_wells = sorted({f.replace("__horizontal_well.csv","")
                          for f in os.listdir(TRAIN_DIR)
                          if f.endswith("__horizontal_well.csv")})
    print(f"  train wells: {len(train_wells)}")
    # Optional cache for local dry-runs only — set ROGII_TRAIN_CACHE=/path/x.parquet
    cache_path = os.environ.get("ROGII_TRAIN_CACHE", "")
    if cache_path and os.path.exists(cache_path):
        print(f"  loading cached train from {cache_path}")
        train = pd.read_parquet(cache_path)
        print(f"  train rows: {len(train):,}, wells: {train['well'].nunique()}")
    else:
        dfs = []; fail = 0
        for i, wid in enumerate(train_wells):
            df_w = extract_well_features(wid, TRAIN_DIR)
            if df_w is None: fail += 1; continue
            dfs.append(df_w)
            if (i+1) % 100 == 0:
                print(f"  {i+1}/{len(train_wells)} | rows so far: {sum(len(d) for d in dfs):,} | "
                      f"fails: {fail} | {time.time()-t0:.0f}s", flush=True)
        train = pd.concat(dfs, ignore_index=True)
        train = train.dropna(subset=["target"]).reset_index(drop=True)
        print(f"  train rows: {len(train):,}, wells: {train['well'].nunique()}, fails: {fail}")
        print(f"  train build: {time.time()-t0:.0f}s")
        if cache_path:
            print(f"  saving train cache to {cache_path}")
            train.to_parquet(cache_path)

    print(f"\n[3/5] Building spatial cKDTree from train known segments + augmenting train")
    t_sp = time.time()
    pool = build_spatial_pool(train_wells, TRAIN_DIR)
    print(f"  pool: {len(pool):,} rows from {pool['well'].nunique()} wells")
    pool_xy   = pool[["x", "y"]].values
    pool_tvt  = pool["tvt"].values.astype(np.float64)
    pool_well = pool["well"].to_numpy(dtype=object)
    tree = cKDTree(pool_xy)
    train = add_spatial_features(train, tree, pool_well, pool_tvt)
    print(f"  train + spatial: {len(train):,} rows × {train.shape[1]} cols | {time.time()-t_sp:.0f}s")

    print(f"\n[4/5] Fitting LightGBM + CatBoost on full corpus")
    feat_cols = [c for c in train.columns if c not in {
        "well","row_idx","target","x_abs","y_abs",
        "neighbor_tvt_median","neighbor_tvt_mean","neighbor_dist_min",
    }]
    print(f"  features ({len(feat_cols)})")
    t1 = time.time()
    model_lgb = lgb.LGBMRegressor(
        n_estimators=900, learning_rate=0.03, num_leaves=63,
        min_child_samples=50, reg_alpha=0.1, reg_lambda=0.1,
        colsample_bytree=0.8, subsample=0.85, subsample_freq=5,
        verbose=-1, n_jobs=-1,
    )
    model_lgb.fit(train[feat_cols], train["target"])
    print(f"  LGB fit: {time.time()-t1:.0f}s")

    t1b = time.time()
    # Phase 11 CV best_iter mean ~400, max 1352; cap at 600 for full-corpus refit
    model_cat = CatBoostRegressor(
        iterations=600, learning_rate=0.05, depth=8,
        l2_leaf_reg=3.0, subsample=0.85, rsm=0.8,
        loss_function="RMSE", eval_metric="RMSE",
        verbose=False, thread_count=-1, random_seed=42,
        bootstrap_type="Bernoulli",
    )
    model_cat.fit(train[feat_cols], train["target"])
    print(f"  CAT fit: {time.time()-t1b:.0f}s")

    print(f"\n[5/5] Building TEST features + spatial + predict")
    t2 = time.time()
    test_wells = sorted({f.replace("__horizontal_well.csv","")
                         for f in os.listdir(TEST_DIR)
                         if f.endswith("__horizontal_well.csv")})
    print(f"  test wells: {len(test_wells)}")
    dft = []
    for wid in test_wells:
        df_w = extract_well_features(wid, TEST_DIR)
        if df_w is None:
            print(f"  ! {wid}: feature build failed"); continue
        dft.append(df_w)
    test = pd.concat(dft, ignore_index=True)
    print(f"  test rows: {len(test):,} | feat build {time.time()-t2:.0f}s")
    # Spatial against same train pool (well_id pool_well_arr will not match
    # any test wid — so all 200 returned neighbors are valid candidates)
    test = add_spatial_features(test, tree, pool_well, pool_tvt)

    pred_lgb = model_lgb.predict(test[feat_cols])
    pred_cat = model_cat.predict(test[feat_cols])
    # Phase 11 best blend: 0.5 · LGB + 0.5 · CAT (per-well 10.03, flat 12.96)
    pred_off = 0.5 * pred_lgb + 0.5 * pred_cat
    pred_tvt = test["last_known_tvt"].values + pred_off
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
        n_missing = sub["tvt"].isna().sum()
        if n_missing:
            print(f"  ⚠ {n_missing} missing preds → filling with median of submitted")
            sub["tvt"] = sub["tvt"].fillna(sub["tvt"].median())
    sub.to_csv(OUT_PATH, index=False)
    print(f"  → {OUT_PATH}  ({len(sub)} rows)")
    print(sub.head().to_string(index=False))
    print(f"\n  pred stats: min={pred_tvt.min():.0f}, max={pred_tvt.max():.0f}, "
          f"mean={pred_tvt.mean():.0f}, median={np.median(pred_tvt):.0f}")
    print(f"\nTotal wall time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
