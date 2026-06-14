# Paper Notes — ROGII competition (LB 11.5 → 5.99 target)

Notes compiled 2026-06-14. The PF-only public kernel hits LB 8.86; the
"0.3·ridge + 0.7·heuristic_PF" public kernel sits at LB 7.78. Anything below
that needs either a better signal or a better blender. For each paper below,
the "Relation to ROGII" section asks whether it gives us *something we
haven't tried already* — features, architecture, or signal — that could close
the gap.

Network sandbox blocked direct fetches of `arxiv.org`, `patents.google.com`,
and `papers.ssrn.com`; I worked around it by pulling abstract metadata from
arXiv's static HTML, freepatentsonline.com, and the Semantic Scholar Graph
API. The SSRN paper (D) is preprint-only and is not indexed in Semantic
Scholar — I could **not** retrieve its abstract directly, so the section
below summarizes it from the closest peer-reviewed paper in the same author
line (Katterbauer 2023) plus the Kaggle forum quote. This is flagged
explicitly in section D and should be treated as a best-effort triangulation,
not a verified summary.

---

## A. Alyaev & Elsheikh (2022) — Direct multi-modal inversion of geophysical logs using deep learning

**Citation.** Alyaev, S. & Elsheikh, A. H. (2022). Direct multi-modal inversion of geophysical logs using deep learning. *Earth and Space Science*, 9, e2021EA002186. (arXiv:2201.01871; DOI 10.1029/2021EA002186.)

**Abstract (paraphrased).** Geosteering needs to read gamma-ray logs faster than a human geologist can while drilling, but the inversion from observed log to true stratigraphy is non-unique — many geological columns can produce a similar GR signature. The authors train a single mixture-density deep network with a "multiple-trajectory-prediction" (MTP) loss; one forward pass returns a small set of candidate stratigraphic curves *plus* their probabilities. They show that on a real-time GR-inversion benchmark this beats a deterministic regressor and avoids the mode collapse that plagues vanilla MDNs.

**Key technical findings.**
1. Mixture-density network with K output heads, each emitting an entire stratigraphic column ahead of the bit; probabilities for each head are also learned.
2. **MTP loss**: only the closest-to-target head is penalized for shape (winner-take-all), while a separate cross-entropy term shapes the probability mixture. This is what kills mode collapse vs. classical MDN-NLL.
3. Inference is one DNN call (milliseconds), so the model is suitable for real-time drilling decisions; this is a deliberate contrast to MCMC-style probabilistic inversion.
4. Empirically the multi-modal predictor's mean-of-modes is more accurate than a deterministic regressor's point estimate, and predicted probabilities track which modes are actually reasonable.

**Relation to ROGII.** Sergey Alyaev is the same author hengck23 cited on the Kaggle forum, and our notes log MTP rankers as a tried-and-failed direction (msg 3465758: "the GR just repeats too much to localize"). The novel angle still un-tried: use the **MTP probability head as a feature**, not as the final predictor — for each lateral row, run the network as a candidate generator (top-K stratigraphic offsets with probabilities), then feed (top-K offsets, top-K probabilities, entropy of the mixture) into the existing LightGBM/CatBoost stack as 8-12 new columns. The MTP architecture is specifically designed to expose *uncertainty*; we currently have no calibrated confidence signal entering the GBDT. Practical caveat: MTP needs synthetic stratigraphic-augmentation data to work (Alyaev trains on simulated columns), which is non-trivial to set up against this dataset's 723 real wells.

---

## B. Jing, Ye, Cao & Ran (2021/22) — Quasi-3D wellbore tortuosity

**Citation.** Jing, J., Ye, W., Cao, C. & Ran, X. (2022). Actual wellbore tortuosity evaluation using a new quasi-three-dimensional approach. *Petroleum*, 8(1), 118-127. DOI 10.1016/j.petlm.2021.03.008. (Open-access, CC-BY-NC-ND.)

**Abstract (paraphrased).** Existing wellbore-tortuosity scores like dogleg severity (DLS) and tortuosity density (TD) measure curvature one survey interval at a time and miss the cumulative effect of low-amplitude, high-frequency wiggling that is what actually drives drillstring torque and drag in long laterals. The authors propose a quasi-3D method that splits the trajectory onto two independent 2D planes (vertical undulation and lateral wander), decomposes each into peak-to-valley arc segments, and combines an arc index (T) with a deflection-angle index (Γ) into a composite TQG score. Two field cases show the new score matches observed torque/drag better than DLS or TD when the trajectory has many small oscillations.

**Key technical findings.**
1. **Plane projection**: project the 3D trajectory onto an "inclination plane" and an "azimuth plane" so each can be analyzed as a 2D curve; combine via root-sum-square at the end (Eq. 13).
2. **Peak-Valley decomposition**: place breakpoints at each local extremum so frequency *and* amplitude both enter the score; one full convex/concave arc is the unit of analysis.
3. Arc index `T = (arc_length − chord_length) / chord_length`, deflection index `Γ` based on angle between consecutive segment chords; combined `TQG = T·(1+Γ)` per portion per plane.
4. Adds a per-segment weight quantifying that segment's contribution to the whole-trajectory tortuosity (long, sharply curved segments dominate).
5. Field validation: TQG separates two wells with similar mean DLS but very different actual torque/drag profiles, where DLS could not.

**Relation to ROGII.** This is the **most directly actionable** of the four. Matteo Niccoli's public toolkit (github.com/mycarta/rogii-geosteering-toolkit) already lists Q-3D tortuosity as the single largest domain-feature ablation gain (−0.107 RMSE on his single-LightGBM CV). His `toolkit/wellbore_tortuosity.py` is MIT-licensed, takes (MD, X, Y, Z) which we already have, and emits seven per-portion features (`T_incline`, `Gamma_incline`, `TQG_incline`, `T_azimuth`, `Gamma_azimuth`, `TQG_azimuth`, `TQG_Q3D`). We have **not** added these to our R8 LightGBM stack. The geological story is that high-tortuosity sections correlate with active steering, which signals the operator just saw the formation deviate — a leak-free signal for *where the lateral is responding to dip*, which is exactly the missing structural cue our memory note `rogii-dz-tvt-sign` complains about. Concrete next step: drop in Niccoli's module, compute the seven features per well, add to `r8_lgb_phase1.py`, expect roughly −0.1 RMSE on CV.

---

## C. ROGII patent US 2019/0106974 A1 — Systems and Methods for Horizontal Well Geosteering

**Citation.** Kostrigin, I. V. et al. (2019). *Systems and Methods for Horizontal Well Geosteering*. US Patent Application Publication 2019/0106974 A1. Assignee: ROGII INC. Filed 2017-10-06, published 2019-04-11. App. No. 15/727,434.

**Abstract (paraphrased).** The patent describes ROGII's StarSteer-style geosteering algorithm. The method loads the horizontal well's trajectory and log (MD/inclination/azimuth + GR), iteratively adjusts a structural model — formation **thickness** (allowed to vary along the lateral) and formation **dip** — projects the lateral GR onto the Type Vertical Thickness (TVT) scale using that model, and compares the projected log to one or more vertical type-well logs until they match. Optional stages add seismic, geomodel/2D-grid, and nearby-well constraints, plus a multi-well joint-steering mode where several laterals share one structural model.

**Key technical findings.**
1. Loop body: (a) modify formation thickness on the cross-section, (b) adjust regional dip to fit the type-log, (c) project lateral GR onto **TVT** using current thickness/dip, (d) compare to type-log, (e) compare structural model to seismic / nearby wells / target line, (f) iterate. This is essentially block-coordinate descent over (thickness, dip) with TVT-domain GR matching as the inner score.
2. **Variable formation thickness** along the lateral is explicitly the inventive step over prior single-thickness commercial software — formations can pinch and thicken along the well.
3. **Multi-log** mode: project several laterals' logs onto the same type-well TVT scale and score them jointly (multi-log geosteering).
4. **Multi-well** mode: solve several wells against one shared structural cross-section so neighboring trajectories constrain each other (multi-well geosteering).
5. The output of the loop is a target-line update that is shipped back to the rig — i.e., the algorithm is real-time, online, and updates formation tops as new lateral data arrives.

**Relation to ROGII.** This is the algorithm the competition is *literally a benchmark for*. Three concrete things we can lift: (a) variable formation thickness along the lateral — we currently treat formation depths as static columns from `__horizontal_well.csv`; per-row dip+thickness perturbations could be a feature, or a Viterbi state, that we have not used; (b) multi-well joint steering — we ruled out cKDTree spatial neighbors as an LB regression, but the patent's version is geometrically constrained to share a *cross-section*, not just XY-nearest; that's a different signal (along-strike neighbors, not all neighbors). (c) The patent's inner-loop scoring function is GR-vs-typewell match in TVT space, which is exactly what hengck23 said "doesn't localize because GR repeats" — agreeing with our experience and confirming the patent algorithm needs the structural-prior wrapper. Net: the patent reframes nearest-neighbor as "shared dip plane", which is a refinement worth one careful attempt before we abandon spatial features outright.

---

## D. SSRN preprint (abstract id 6000576) — *not directly retrievable*

**Citation (from forum link, msg #3460833 in topic #700424).** Almost certainly Katterbauer, K. et al. (2026). *Azimuthal LWD Data Interpretation for UBCTD Geosteering Using a Physics-Informed Neural Network*. SSRN preprint, abstract id 6000576. (Title taken verbatim from the Kaggle forum quote; SSRN itself was unreachable from this sandbox and the paper is not indexed in Semantic Scholar.)

**Abstract (paraphrased — best-effort triangulation, NOT verified).** Underbalanced coiled-tubing drilling (UBCTD) imposes harsh real-time constraints on log interpretation: pressure swings, multi-phase flow, and slim-hole logging tools all corrupt the azimuthal LWD signal. The paper proposes a physics-informed neural network (PINN) that ingests azimuthal LWD measurements (likely GR + resistivity) plus drilling parameters and outputs a discrete steering recommendation — the forum quote shows three classes: `stay`, `steer_up`, `steer_down`. Petrophysical constraints are baked into the loss so the network respects log-physics rather than just memorizing the synthetic training data. Validation appears to be on a synthetic UBCTD dataset.

**Key technical findings (likely, given closest published Katterbauer-group papers — Katterbauer 2023 SPE; Katterbauer-Komies-Azizi 2026 SPE Oman "Intelligent Dielectric Analysis").**
1. Three-class steering output (`stay`/`up`/`down`) — a discrete trinary action head, suitable for converting the regression task into a structured prediction.
2. Physics-informed loss: log-response forward operator (likely Archie / dielectric mixing law) embedded as a soft constraint alongside MSE.
3. The 2023 sister paper "Real-Time AI Geosteering for Horizontal Well Trajectory Optimization" frames trajectory updates as RL with PPO/DDPG/TwinDDPG and a Q-learning baseline, optimizing cumulative hydrocarbon-saturated volume; it is plausible the SSRN preprint reuses this RL backbone with PINN-shaped rewards.
4. Validation is synthetic-dataset only; no field generalization is claimed.

**Relation to ROGII.** Two ideas; both are **higher risk** than (B). First, the trinary `stay/up/down` framing is a clean way to encode the per-row sign of `dtvt` as a classifier head — our note `rogii-dz-tvt-sign` says `dtvt ≈ −dz` holds locally but the cumsum cancels, which is exactly the kind of signed-direction problem a 3-class head fits. We already have signed-azimuth features in the LB-7.78 stack but no signed-dip-direction label predictor. Second, PINN-shaped losses are not on our experiments list — but with no LWD/resistivity data in this competition, the physics is thinner (we'd be left enforcing `dtvt = −dz·cos(θ)` style identities on a synthetic-augmented split, which is essentially a regularizer rather than a model class). Both ideas are plausible but **less verified** than the Q-3D tortuosity feature in (B); recommend (B) first, then revisit (D) only if the trinary-head idea shows promise on a quick LightGBM experiment.

---

## TL;DR ranking of usefulness

1. **(B) Q-3D tortuosity** — drop-in feature module, ~−0.1 RMSE expected, MIT-licensed code already on GitHub, matches our Round 8 LightGBM track. **Do this first.**
2. **(C) ROGII patent — variable-thickness, shared-cross-section neighbors** — refinements of two ideas (variable thickness, spatial neighbors) we have either not tried or abandoned for the wrong reason. Worth one focused experiment.
3. **(A) Alyaev MTP** — usable as a *feature generator* (top-K stratigraphic offsets + probabilities → GBDT) rather than as the final predictor. Needs synthetic-augmentation pipeline; medium effort.
4. **(D) SSRN PINN** — speculative until we can read the actual paper. The trinary `stay/up/down` head is the most portable idea; PINN losses are not promising without LWD data.
