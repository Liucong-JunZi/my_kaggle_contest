"""R8 Phase 2A: compute PF features for all 723 train wells (one pass).

Extracts the Particle Filter (PF) code from the LB-7.776 kernel
(docs/lb-references/lb-7-776-rogii-ridge-sp.py) and runs it once per well
to produce two per-lateral-row signals:

  pf_ancc, pf_ancc_std   — ANCC-anchored PF (uses TVT+Z position + GR)
  pf_z,    pf_z_std      — Z-velocity PF (TVT-velocity-tracking + GR)

Output: results/round_008/pf_features.parquet
   columns: well, row_idx, pf_ancc, pf_ancc_std, pf_z, pf_z_std

These will be joined with Phase-1 features (features_full.parquet) and
re-fed to LightGBM in r8_lgb_phase2.py.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

DATA_DIR = "/Users/liucong/code/kaggle/ROGII/rogii-wellbore-geology-prediction/train"
OUT_DIR  = Path("/Users/liucong/code/kaggle/ROGII/results/round_008")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── PF hyperparameters (verbatim from kernel) ─────────────────────────────────
PF_N=600; ANCC_N=600
PF_MOM=0.993; PF_VN=0.005; PF_PN=0.01
PF_GR_SIG_MIN=10.; PF_GR_SIG_MAX=60.; PF_GR_SIG_DEF=30.
PF_INIT_V_STD=0.02; PF_INIT_SPR=0.5; PF_RESAMP=0.5
PF_ROUGH_P=0.2; PF_ROUGH_V=0.003; PF_GR_WIN=5; PF_GR_WT=0.3
ANCC_ALPHA=0.998; ANCC_RN=0.002; ANCC_PN=0.005
ANCC_IR=0.01; ANCC_IS=4.5; ANCC_RP=0.1; ANCC_RR=0.001  # IS 0.3 → 4.5 (sp45 patch from kernel)


# ── JIT PF kernels (verbatim from kernel) ─────────────────────────────────────
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
def _pf_ancc(md_v,z_v,gr_v,gg,vmin,step,gs,ls,ir,N,
              ALPHA,RN,PN,IS,RP,RR,RESAMP):
    pos=np.empty(N); rate=np.empty(N); w=np.ones(N)/N
    for j in range(N):
        pos[j]=ls+IS*np.random.randn()
        rate[j]=ir+0.01*np.random.randn()
    pts=np.empty(len(md_v)); std_=np.empty(len(md_v))
    loglk=np.empty(len(md_v))   # cumulative log-evidence up to row i
    pm=md_v[0]-1.; cum_ll=0.
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
                avg_lk+=w[j]*lk      # observation evidence under prior
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
        pts[i]=tv; va=0.
        for j in range(N): va+=w[j]*(pos[j]-z_v[i]-tv)**2
        std_[i]=va**0.5; loglk[i]=cum_ll; pm=md_v[i]
    return pts,std_,loglk


@njit(cache=True)
def _pf_z(md_v,z_v,gr_v,gr_sm_v,gg_p,gg_s,vmin,step,
          gs,ip,iv,beta,icpt,zsig,N,
          MOM,VN,PN,GR_WT,RP,RV,RESAMP):
    pos=np.empty(N); vel=np.empty(N); w=np.ones(N)/N
    for j in range(N):
        pos[j]=ip+0.5*np.random.randn()
        vel[j]=iv+0.02*np.random.randn()
    pts=np.empty(len(md_v)); std_=np.empty(len(md_v))
    loglk=np.empty(len(md_v))
    pm=md_v[0]-1.; pz=z_v[0]-1.; cum_ll=0.
    for i in range(len(md_v)):
        dm=md_v[i]-pm; dm=max(dm,1.)
        dzd=(z_v[i]-pz)/dm; ve=beta*dzd+icpt
        for j in range(N):
            vel[j]=MOM*vel[j]+VN*np.random.randn()
            pos[j]+=vel[j]*dm+PN*np.random.randn()
            pos[j]=max(pos[j],vmin-50.); pos[j]=min(pos[j],vmin+len(gg_p)*step+50.)
        if not np.isnan(gr_v[i]):
            ws=0.; avg_lk=0.
            for j in range(N):
                ep=_interp1(gg_p,pos[j],vmin,step)
                dp=(gr_v[i]-ep)/gs
                lp=max(np.exp(-0.5*dp*dp) if dp*dp<600. else 0.,1e-300)
                if not np.isnan(gr_sm_v[i]):
                    es=_interp1(gg_s,pos[j],vmin,step)
                    ds=(gr_sm_v[i]-es)/(gs*1.5)
                    lsm=max(np.exp(-0.5*ds*ds) if ds*ds<600. else 0.,1e-300)
                    lk=(1.-GR_WT)*lp+GR_WT*lsm
                else: lk=lp
                lk=max(lk,1e-300); avg_lk+=w[j]*lk
                w[j]*=lk; ws+=w[j]
            cum_ll += np.log(max(avg_lk,1e-300))
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
        std_[i]=va**0.5; loglk[i]=cum_ll; pm=md_v[i]; pz=z_v[i]
    return pts,std_,loglk


# ── Helpers ────────────────────────────────────────────────────────────────────
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
    if len(ev)==0: return np.array([]),np.array([]),np.array([])
    ls=float(kn['TVT_input'].iloc[-1]+kn['Z'].iloc[-1])
    tail=kn.tail(30); dt=np.diff(tail['TVT_input'].values)
    dz=np.diff(tail['Z'].values); dm=np.diff(tail['MD'].values); m=dm>0
    ir=float(np.median((dt+dz)[m]/dm[m])) if m.sum()>=3 else 0.
    gg,gmin,gst=_grid(tw_tvt,tw_gr)
    # Pre-interpolate GR over the FULL well, then slice ev rows (kernel pattern).
    gr_full = hw['GR'].interpolate(limit_direction='both').fillna(float(np.nanmean(tw_gr)))
    gr_v = gr_full.loc[ev.index].values.astype(np.float64)
    pts,std,loglk=_pf_ancc(ev['MD'].values.astype(np.float64),ev['Z'].values.astype(np.float64),
                      gr_v,gg,gmin,gst,
                      gs,ls,ir,N,ANCC_ALPHA,ANCC_RN,ANCC_PN,ANCC_IS,ANCC_RP,ANCC_RR,PF_RESAMP)
    return pts.astype(np.float32),std.astype(np.float32),loglk.astype(np.float32)


def run_pf_z(hw,tw_tvt,tw_gr,N=PF_N):
    gs=_gr_sig(hw,tw_tvt,tw_gr)
    tw_s=pd.Series(tw_gr).rolling(PF_GR_WIN,center=True,min_periods=1).mean().values.astype(np.float32)
    kna=hw[hw['TVT_input'].notna()]; ev=hw[hw['TVT_input'].isna()]
    if len(ev)==0: return np.array([]),np.array([]),np.array([])
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
    # Pre-interpolate raw + smoothed GR over the FULL well (kernel pattern).
    gr_full   = hw['GR'].interpolate(limit_direction='both').fillna(float(np.nanmean(tw_gr)))
    gr_sm_ful = gr_full.rolling(PF_GR_WIN,center=True,min_periods=1).mean()
    gr_v   = gr_full.loc[ev.index].values.astype(np.float64)
    gr_smv = gr_sm_ful.loc[ev.index].values.astype(np.float64)
    pts,std,loglk=_pf_z(ev['MD'].values.astype(np.float64),ev['Z'].values.astype(np.float64),
                   gr_v,gr_smv,
                   gg,gs2,gmin,gst,gs,float(kna['TVT_input'].iloc[-1]),iv,
                   beta,icpt,zsig,N,
                   PF_MOM,PF_VN,PF_PN,PF_GR_WT,PF_ROUGH_P,PF_ROUGH_V,PF_RESAMP)
    return pts.astype(np.float32),std.astype(np.float32),loglk.astype(np.float32)


# ── JIT warmup (tiny inputs, just to compile) ─────────────────────────────────
def _warmup():
    md=np.linspace(1,50,20,np.float64); z=np.zeros(20,np.float64)
    gr=np.full(20,50.,np.float64); gg=np.linspace(45,55,100,np.float64)
    _pf_ancc(md,z,gr,gg,45.,0.1,20.,50.,0.,8,0.998,0.002,0.005,4.5,0.1,0.001,0.5)
    _pf_z(md,z,gr,gr,gg,gg,45.,0.1,20.,50.,0.,-1.,0.,0.1,8,0.993,0.005,0.01,0.3,0.2,0.003,0.5)


def main():
    print(f"=== R8 Phase 2A: PF features for full corpus ===\n")
    print("[warmup] compiling numba kernels...")
    t0 = time.time()
    np.random.seed(0)
    _warmup()
    print(f"  done in {time.time()-t0:.1f}s\n")

    all_wells = sorted({f.replace("__horizontal_well.csv","")
                        for f in os.listdir(DATA_DIR)
                        if f.endswith("__horizontal_well.csv")})
    print(f"Wells: {len(all_wells)}\n")

    records = []
    n_ok = n_fail = 0
    t0 = time.time()
    for i, wid in enumerate(all_wells):
        try:
            hw = pd.read_csv(f"{DATA_DIR}/{wid}__horizontal_well.csv")
            tw = pd.read_csv(f"{DATA_DIR}/{wid}__typewell.csv")
        except Exception:
            n_fail += 1; continue

        # Typewell sort + GR fill
        tw_s = tw.sort_values('TVT')
        tw_tvt = tw_s['TVT'].values.astype(np.float64)
        tw_gr = tw_s['GR'].fillna(tw_s['GR'].mean()).values.astype(np.float64)
        if len(tw_tvt) < 10 or np.isnan(tw_gr).all():
            n_fail += 1; continue

        # Need a known segment + lateral
        if hw['TVT_input'].notna().sum() < 10 or hw['TVT_input'].isna().sum() < 10:
            n_fail += 1; continue

        np.random.seed(42)  # match kernel seed for reproducibility
        try:
            ancc_pts, ancc_std, ancc_ll = run_pf_ancc(hw, tw_tvt, tw_gr)
            z_pts,    z_std,    z_ll    = run_pf_z(hw, tw_tvt, tw_gr)
        except Exception as e:
            print(f"  ! {wid}: {e}")
            n_fail += 1; continue

        ev_idx = hw.index[hw['TVT_input'].isna()].values
        if len(ev_idx) != len(ancc_pts):
            n_fail += 1; continue

        df_pf = pd.DataFrame({
            "well": wid,
            "row_idx": ev_idx.astype(np.int32),
            "pf_ancc":     ancc_pts,
            "pf_ancc_std": ancc_std,
            "pf_ancc_ll":  ancc_ll,
            "pf_z":        z_pts,
            "pf_z_std":    z_std,
            "pf_z_ll":     z_ll,
        })
        records.append(df_pf)
        n_ok += 1

        if (i+1) % 25 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i+1) * (len(all_wells) - i - 1)
            print(f"  {i+1}/{len(all_wells)} | ok={n_ok} fail={n_fail} | "
                  f"elapsed={elapsed:.0f}s eta={eta:.0f}s")

    print(f"\nDone in {time.time()-t0:.0f}s | ok={n_ok} fail={n_fail}")
    df = pd.concat(records, ignore_index=True)
    print(f"Rows: {len(df):,}  Wells: {df['well'].nunique()}")
    print(f"Per-well log-lik stats: ancc med={df['pf_ancc_ll'].median():.1f}  "
          f"z med={df['pf_z_ll'].median():.1f}")
    out_path = OUT_DIR / "pf_features_v9.parquet"   # new file; keep v8 baseline intact
    df.to_parquet(out_path)
    print(f"→ {out_path}")


if __name__ == "__main__":
    main()
