theorem hasFPowerSeriesOnBall_const {c : F} {e : E} :
    HasFPowerSeriesOnBall (fun _ => c) (constFormalMultilinearSeries 𝕜 E c) e ⊤ := by
  /-
    𝕜 : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_3
    F : Type u_4
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    c : F
    e : E
    ⊢ HasFPowerSeriesOnBall (fun x => c) (constFormalMultilinearSeries 𝕜 E c) e To …
  -/
  refine ⟨by simp, WithTop.top_pos, fun _ => hasSum_single 0 fun n hn => ?_⟩
  /-
    𝕜 : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_3
    F : Type u_4
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    c : F
    e y✝ : E
    x✝ : Membership.mem (EMetric.ball 0 Top.top) y✝
    n : Nat
    hn : Ne n 0
    ⊢ Eq ((constFormalMultilinearSeries 𝕜 E c n) fun x => y✝) 0
  -/
  simp [constFormalMultilinearSeries_apply hn]
  /-
    🎉 no goals
  -/


theorem hasFPowerSeriesAt_const {c : F} {e : E} :
    HasFPowerSeriesAt (fun _ => c) (constFormalMultilinearSeries 𝕜 E c) e :=
  ⟨⊤, hasFPowerSeriesOnBall_const⟩


theorem analyticAt_const {v : F} {x : E} : AnalyticAt 𝕜 (fun _ => v) x :=
  ⟨constFormalMultilinearSeries 𝕜 E v, hasFPowerSeriesAt_const⟩


theorem analyticOnNhd_const {v : F} {s : Set E} : AnalyticOnNhd 𝕜 (fun _ => v) s :=
  fun _ _ => analyticAt_const


theorem analyticWithinAt_const {v : F} {s : Set E} {x : E} : AnalyticWithinAt 𝕜 (fun _ => v) s x :=
  analyticAt_const.analyticWithinAt


theorem analyticOn_const {v : F} {s : Set E} : AnalyticOn 𝕜 (fun _ => v) s :=
  analyticOnNhd_const.analyticOn


@[deprecated (since := "2024-09-26")]
alias analyticWithinOn_const := analyticOn_const


theorem HasFPowerSeriesWithinOnBall.add (hf : HasFPowerSeriesWithinOnBall f pf s x r)
    (hg : HasFPowerSeriesWithinOnBall g pg s x r) :
    HasFPowerSeriesWithinOnBall (f + g) (pf + pg) s x r :=
  { r_le := le_trans (le_min_iff.2 ⟨hf.r_le, hg.r_le⟩) (pf.min_radius_le_radius_add pg)
    r_pos := hf.r_pos
    hasSum := fun hy h'y => (hf.hasSum hy h'y).add (hg.hasSum hy h'y) }


theorem HasFPowerSeriesOnBall.add (hf : HasFPowerSeriesOnBall f pf x r)
    (hg : HasFPowerSeriesOnBall g pg x r) : HasFPowerSeriesOnBall (f + g) (pf + pg) x r :=
  { r_le := le_trans (le_min_iff.2 ⟨hf.r_le, hg.r_le⟩) (pf.min_radius_le_radius_add pg)
    r_pos := hf.r_pos
    hasSum := fun hy => (hf.hasSum hy).add (hg.hasSum hy) }


theorem HasFPowerSeriesWithinAt.add
    (hf : HasFPowerSeriesWithinAt f pf s x) (hg : HasFPowerSeriesWithinAt g pg s x) :
    HasFPowerSeriesWithinAt (f + g) (pf + pg) s x := by
  /-
    𝕜 : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_3
    F : Type u_4
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f g : E → F
    pf pg : FormalMultilinearSeries 𝕜 E F
    s : Set E
    x : E
    hf : HasFPowerSeriesWithinAt f pf s x
    hg : HasFPowerSeriesWithinAt g pg s x
    ⊢ HasFPowerSeriesWithinAt (HAdd.hAdd f g) (HAdd.hAdd pf pg) s x
  -/
  rcases (hf.eventually.and hg.eventually).exists with ⟨r, hr⟩
  /-
    case intro
    𝕜 : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_3
    F : Type u_4
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f g : E → F
    pf pg : FormalMultilinearSeries 𝕜 E F
    s : Set E
    x : E
    hf : HasFPowerSeriesWithinAt f pf s x
    hg : HasFPowerSeriesWithinAt g pg s x
    r : ENNReal
    hr : And (HasFPowerSeriesWithinOnBall f pf s x r) (HasFPowerSeriesWithinOnBall …
    ⊢ HasFPowerSeriesWithinAt (HAdd.hAdd f g) (HAdd.hAdd pf pg) s x
  -/
  exact ⟨r, hr.1.add hr.2⟩
  /-
    🎉 no goals
  -/


theorem HasFPowerSeriesAt.add (hf : HasFPowerSeriesAt f pf x) (hg : HasFPowerSeriesAt g pg x) :
    HasFPowerSeriesAt (f + g) (pf + pg) x := by
  /-
    𝕜 : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_3
    F : Type u_4
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f g : E → F
    pf pg : FormalMultilinearSeries 𝕜 E F
    x : E
    hf : HasFPowerSeriesAt f pf x
    hg : HasFPowerSeriesAt g pg x
    ⊢ HasFPowerSeriesAt (HAdd.hAdd f g) (HAdd.hAdd pf pg) x
  -/
  rcases (hf.eventually.and hg.eventually).exists with ⟨r, hr⟩
  /-
    case intro
    𝕜 : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_3
    F : Type u_4
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f g : E → F
    pf pg : FormalMultilinearSeries 𝕜 E F
    x : E
    hf : HasFPowerSeriesAt f pf x
    hg : HasFPowerSeriesAt g pg x
    r : ENNReal
    hr : And (HasFPowerSeriesOnBall f pf x r) (HasFPowerSeriesOnBall g pg x r)
    ⊢ HasFPowerSeriesAt (HAdd.hAdd f g) (HAdd.hAdd pf pg) x
  -/
  exact ⟨r, hr.1.add hr.2⟩
  /-
    🎉 no goals
  -/


theorem AnalyticWithinAt.add (hf : AnalyticWithinAt 𝕜 f s x) (hg : AnalyticWithinAt 𝕜 g s x) :
    AnalyticWithinAt 𝕜 (f + g) s x :=
  let ⟨_, hpf⟩ := hf
  let ⟨_, hqf⟩ := hg
  (hpf.add hqf).analyticWithinAt


theorem AnalyticAt.add (hf : AnalyticAt 𝕜 f x) (hg : AnalyticAt 𝕜 g x) : AnalyticAt 𝕜 (f + g) x :=
  let ⟨_, hpf⟩ := hf
  let ⟨_, hqf⟩ := hg
  (hpf.add hqf).analyticAt


theorem HasFPowerSeriesWithinOnBall.neg (hf : HasFPowerSeriesWithinOnBall f pf s x r) :
    HasFPowerSeriesWithinOnBall (-f) (-pf) s x r :=
  { r_le := by
      /-
        𝕜 : Type u_2
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type u_3
        F : Type u_4
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        f : E → F
        pf : FormalMultilinearSeries 𝕜 E F
        s : Set E
        x : E
        r : ENNReal
        hf : HasFPowerSeriesWithinOnBall f pf s x r
        ⊢ LE.le r (Neg.neg pf).radius
      -/
      rw [pf.radius_neg]
      /-
        𝕜 : Type u_2
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type u_3
        F : Type u_4
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        f : E → F
        pf : FormalMultilinearSeries 𝕜 E F
        s : Set E
        x : E
        r : ENNReal
        hf : HasFPowerSeriesWithinOnBall f pf s x r
        ⊢ LE.le r pf.radius
      -/
      exact hf.r_le
      /-
        🎉 no goals
      -/
    r_pos := hf.r_pos
    hasSum := fun hy h'y => (hf.hasSum hy h'y).neg }


theorem HasFPowerSeriesOnBall.neg (hf : HasFPowerSeriesOnBall f pf x r) :
    HasFPowerSeriesOnBall (-f) (-pf) x r :=
  { r_le := by
      /-
        𝕜 : Type u_2
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type u_3
        F : Type u_4
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        f : E → F
        pf : FormalMultilinearSeries 𝕜 E F
        x : E
        r : ENNReal
        hf : HasFPowerSeriesOnBall f pf x r
        ⊢ LE.le r (Neg.neg pf).radius
      -/
      rw [pf.radius_neg]
      /-
        𝕜 : Type u_2
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type u_3
        F : Type u_4
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        f : E → F
        pf : FormalMultilinearSeries 𝕜 E F
        x : E
        r : ENNReal
        hf : HasFPowerSeriesOnBall f pf x r
        ⊢ LE.le r pf.radius
      -/
      exact hf.r_le
      /-
        🎉 no goals
      -/
    r_pos := hf.r_pos
    hasSum := fun hy => (hf.hasSum hy).neg }


theorem HasFPowerSeriesWithinAt.neg (hf : HasFPowerSeriesWithinAt f pf s x) :
    HasFPowerSeriesWithinAt (-f) (-pf) s x :=
  let ⟨_, hrf⟩ := hf
  hrf.neg.hasFPowerSeriesWithinAt


theorem HasFPowerSeriesAt.neg (hf : HasFPowerSeriesAt f pf x) : HasFPowerSeriesAt (-f) (-pf) x :=
  let ⟨_, hrf⟩ := hf
  hrf.neg.hasFPowerSeriesAt


theorem AnalyticWithinAt.neg (hf : AnalyticWithinAt 𝕜 f s x) : AnalyticWithinAt 𝕜 (-f) s x :=
  let ⟨_, hpf⟩ := hf
  hpf.neg.analyticWithinAt


theorem AnalyticAt.neg (hf : AnalyticAt 𝕜 f x) : AnalyticAt 𝕜 (-f) x :=
  let ⟨_, hpf⟩ := hf
  hpf.neg.analyticAt


theorem HasFPowerSeriesWithinOnBall.sub (hf : HasFPowerSeriesWithinOnBall f pf s x r)
    (hg : HasFPowerSeriesWithinOnBall g pg s x r) :
    HasFPowerSeriesWithinOnBall (f - g) (pf - pg) s x r := by
  /-
    𝕜 : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_3
    F : Type u_4
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f g : E → F
    pf pg : FormalMultilinearSeries 𝕜 E F
    s : Set E
    x : E
    r : ENNReal
    hf : HasFPowerSeriesWithinOnBall f pf s x r
    hg : HasFPowerSeriesWithinOnBall g pg s x r
    ⊢ HasFPowerSeriesWithinOnBall (HSub.hSub f g) (HSub.hSub pf pg) s x r
  -/
  simpa only [sub_eq_add_neg] using hf.add hg.neg
  /-
    🎉 no goals
  -/


theorem HasFPowerSeriesOnBall.sub (hf : HasFPowerSeriesOnBall f pf x r)
    (hg : HasFPowerSeriesOnBall g pg x r) : HasFPowerSeriesOnBall (f - g) (pf - pg) x r := by
  /-
    𝕜 : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_3
    F : Type u_4
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f g : E → F
    pf pg : FormalMultilinearSeries 𝕜 E F
    x : E
    r : ENNReal
    hf : HasFPowerSeriesOnBall f pf x r
    hg : HasFPowerSeriesOnBall g pg x r
    ⊢ HasFPowerSeriesOnBall (HSub.hSub f g) (HSub.hSub pf pg) x r
  -/
  simpa only [sub_eq_add_neg] using hf.add hg.neg
  /-
    🎉 no goals
  -/


theorem HasFPowerSeriesWithinAt.sub
    (hf : HasFPowerSeriesWithinAt f pf s x) (hg : HasFPowerSeriesWithinAt g pg s x) :
    HasFPowerSeriesWithinAt (f - g) (pf - pg) s x := by
  /-
    𝕜 : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_3
    F : Type u_4
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f g : E → F
    pf pg : FormalMultilinearSeries 𝕜 E F
    s : Set E
    x : E
    hf : HasFPowerSeriesWithinAt f pf s x
    hg : HasFPowerSeriesWithinAt g pg s x
    ⊢ HasFPowerSeriesWithinAt (HSub.hSub f g) (HSub.hSub pf pg) s x
  -/
  simpa only [sub_eq_add_neg] using hf.add hg.neg
  /-
    🎉 no goals
  -/


theorem HasFPowerSeriesAt.sub (hf : HasFPowerSeriesAt f pf x) (hg : HasFPowerSeriesAt g pg x) :
    HasFPowerSeriesAt (f - g) (pf - pg) x := by
  /-
    𝕜 : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_3
    F : Type u_4
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f g : E → F
    pf pg : FormalMultilinearSeries 𝕜 E F
    x : E
    hf : HasFPowerSeriesAt f pf x
    hg : HasFPowerSeriesAt g pg x
    ⊢ HasFPowerSeriesAt (HSub.hSub f g) (HSub.hSub pf pg) x
  -/
  simpa only [sub_eq_add_neg] using hf.add hg.neg
  /-
    🎉 no goals
  -/


theorem AnalyticWithinAt.sub (hf : AnalyticWithinAt 𝕜 f s x) (hg : AnalyticWithinAt 𝕜 g s x) :
    AnalyticWithinAt 𝕜 (f - g) s x := by
  /-
    𝕜 : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_3
    F : Type u_4
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f g : E → F
    s : Set E
    x : E
    hf : AnalyticWithinAt 𝕜 f s x
    hg : AnalyticWithinAt 𝕜 g s x
    ⊢ AnalyticWithinAt 𝕜 (HSub.hSub f g) s x
  -/
  simpa only [sub_eq_add_neg] using hf.add hg.neg
  /-
    🎉 no goals
  -/


theorem AnalyticAt.sub (hf : AnalyticAt 𝕜 f x) (hg : AnalyticAt 𝕜 g x) :
    AnalyticAt 𝕜 (f - g) x := by
  /-
    𝕜 : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_3
    F : Type u_4
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f g : E → F
    x : E
    hf : AnalyticAt 𝕜 f x
    hg : AnalyticAt 𝕜 g x
    ⊢ AnalyticAt 𝕜 (HSub.hSub f g) x
  -/
  simpa only [sub_eq_add_neg] using hf.add hg.neg
  /-
    🎉 no goals
  -/


theorem AnalyticOn.add (hf : AnalyticOn 𝕜 f s) (hg : AnalyticOn 𝕜 g s) :
    AnalyticOn 𝕜 (f + g) s :=
  fun z hz => (hf z hz).add (hg z hz)


@[deprecated (since := "2024-09-26")]
alias AnalyticWithinOn.add := AnalyticOn.add


theorem AnalyticOnNhd.add (hf : AnalyticOnNhd 𝕜 f s) (hg : AnalyticOnNhd 𝕜 g s) :
    AnalyticOnNhd 𝕜 (f + g) s :=
  fun z hz => (hf z hz).add (hg z hz)


theorem AnalyticOn.neg (hf : AnalyticOn 𝕜 f s) : AnalyticOn 𝕜 (-f) s :=
  fun z hz ↦ (hf z hz).neg


@[deprecated (since := "2024-09-26")]
alias AnalyticWithinOn.neg := AnalyticOn.neg


theorem AnalyticOnNhd.neg (hf : AnalyticOnNhd 𝕜 f s) : AnalyticOnNhd 𝕜 (-f) s :=
  fun z hz ↦ (hf z hz).neg


theorem AnalyticOn.sub (hf : AnalyticOn 𝕜 f s) (hg : AnalyticOn 𝕜 g s) :
    AnalyticOn 𝕜 (f - g) s :=
  fun z hz => (hf z hz).sub (hg z hz)


@[deprecated (since := "2024-09-26")]
alias AnalyticWithinOn.sub := AnalyticOn.sub


theorem AnalyticOnNhd.sub (hf : AnalyticOnNhd 𝕜 f s) (hg : AnalyticOnNhd 𝕜 g s) :
    AnalyticOnNhd 𝕜 (f - g) s :=
  fun z hz => (hf z hz).sub (hg z hz)


/-- The radius of the Cartesian product of two formal series is the minimum of their radii. -/
lemma FormalMultilinearSeries.radius_prod_eq_min
    (p : FormalMultilinearSeries 𝕜 E F) (q : FormalMultilinearSeries 𝕜 E G) :
    (p.prod q).radius = min p.radius q.radius := by
  /-
    𝕜 : Type u_2
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_3
    F : Type u_4
    G : Type u_5
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    p : FormalMultilinearSeries 𝕜 E F
    q : FormalMultilinearSeries 𝕜 E G
    ⊢ Eq (p.prod q).radius (Min.min p.radius q.radius)
  -/
  apply le_antisymm
    /-
      case a
      𝕜 : Type u_2
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_3
      F : Type u_4
      G : Type u_5
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      p : FormalMultilinearSeries 𝕜 E F
      q : FormalMultilinearSeries 𝕜 E G
      ⊢ LE.le (p.prod q).radius (Min.min p.radius q.radius)
    -/
  · refine ENNReal.le_of_forall_nnreal_lt fun r hr => ?_
    /-
      case a
      𝕜 : Type u_2
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_3
      F : Type u_4
      G : Type u_5
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      p : FormalMultilinearSeries 𝕜 E F
      q : FormalMultilinearSeries 𝕜 E G
      r : NNReal
      hr : LT.lt (↑r) (p.prod q).radius
      ⊢ LE.le (↑r) (Min.min p.radius q.radius)
    -/
    rw [le_min_iff]
    /-
      case a
      𝕜 : Type u_2
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_3
      F : Type u_4
      G : Type u_5
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      p : FormalMultilinearSeries 𝕜 E F
      q : FormalMultilinearSeries 𝕜 E G
      r : NNReal
      hr : LT.lt (↑r) (p.prod q).radius
      ⊢ And (LE.le (↑r) p.radius) (LE.le (↑r) q.radius)
    -/
    have := (p.prod q).isLittleO_one_of_lt_radius hr
    /-
      case a
      𝕜 : Type u_2
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_3
      F : Type u_4
      G : Type u_5
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      p : FormalMultilinearSeries 𝕜 E F
      q : FormalMultilinearSeries 𝕜 E G
      r : NNReal
      hr : LT.lt (↑r) (p.prod q).radius
      this : Asymptotics.IsLittleO Filter.atTop (fun n => HMul.hMul (Norm.norm (p.pr …
      ⊢ And (LE.le (↑r) p.radius) (LE.le (↑r) q.radius)
    -/
    constructor
    all_goals
      apply FormalMultilinearSeries.le_radius_of_isBigO
      refine (isBigO_of_le _ fun n ↦ ?_).trans this.isBigO
      rw [norm_mul, norm_norm, norm_mul, norm_norm]
      refine mul_le_mul_of_nonneg_right ?_ (norm_nonneg _)
      rw [FormalMultilinearSeries.prod, ContinuousMultilinearMap.opNorm_prod]
      /-
        case a.left.h
        𝕜 : Type u_2
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_3
        F : Type u_4
        G : Type u_5
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        p : FormalMultilinearSeries 𝕜 E F
        q : FormalMultilinearSeries 𝕜 E G
        r : NNReal
        hr : LT.lt (↑r) (p.prod q).radius
        this : Asymptotics.IsLittleO Filter.atTop (fun n => HMul.hMul (Norm.norm (p.pr …
        n : Nat
        ⊢ LE.le (Norm.norm (p n)) (Max.max (Norm.norm (p n)) (Norm.norm (q n)))
      -/
    · apply le_max_left
      /-
        🎉 no goals
      -/
      /-
        case a.right.h
        𝕜 : Type u_2
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_3
        F : Type u_4
        G : Type u_5
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        p : FormalMultilinearSeries 𝕜 E F
        q : FormalMultilinearSeries 𝕜 E G
        r : NNReal
        hr : LT.lt (↑r) (p.prod q).radius
        this : Asymptotics.IsLittleO Filter.atTop (fun n => HMul.hMul (Norm.norm (p.pr …
        n : Nat
        ⊢ LE.le (Norm.norm (q n)) (Max.max (Norm.norm (p n)) (Norm.norm (q n)))
      -/
    · apply le_max_right
      /-
        🎉 no goals
      -/
    /-
      case a
      𝕜 : Type u_2
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_3
      F : Type u_4
      G : Type u_5
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      p : FormalMultilinearSeries 𝕜 E F
      q : FormalMultilinearSeries 𝕜 E G
      ⊢ LE.le (Min.min p.radius q.radius) (p.prod q).radius
    -/
  · refine ENNReal.le_of_forall_nnreal_lt fun r hr => ?_
    /-
      case a
      𝕜 : Type u_2
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_3
      F : Type u_4
      G : Type u_5
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      p : FormalMultilinearSeries 𝕜 E F
      q : FormalMultilinearSeries 𝕜 E G
      r : NNReal
      hr : LT.lt (↑r) (Min.min p.radius q.radius)
      ⊢ LE.le (↑r) (p.prod q).radius
    -/
    rw [lt_min_iff] at hr
    have := ((p.isLittleO_one_of_lt_radius hr.1).add
      (q.isLittleO_one_of_lt_radius hr.2)).isBigO
    /-
      case a
      𝕜 : Type u_2
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_3
      F : Type u_4
      G : Type u_5
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      p : FormalMultilinearSeries 𝕜 E F
      q : FormalMultilinearSeries 𝕜 E G
      r : NNReal
      hr : And (LT.lt (↑r) p.radius) (LT.lt (↑r) q.radius)
      this : Asymptotics.IsBigO Filter.atTop (fun x => HAdd.hAdd (HMul.hMul (Norm.no …
      ⊢ LE.le (↑r) (p.prod q).radius
    -/
    refine (p.prod q).le_radius_of_isBigO ((isBigO_of_le _ fun n ↦ ?_).trans this)
    /-
      case a
      𝕜 : Type u_2
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_3
      F : Type u_4
      G : Type u_5
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      p : FormalMultilinearSeries 𝕜 E F
      q : FormalMultilinearSeries 𝕜 E G
      r : NNReal
      hr : And (LT.lt (↑r) p.radius) (LT.lt (↑r) q.radius)
      this : Asymptotics.IsBigO Filter.atTop (fun x => HAdd.hAdd (HMul.hMul (Norm.no …
      n : Nat
      ⊢ LE.le (Norm.norm (HMul.hMul (Norm.norm (p.prod q n)) (HPow.hPow (↑r) n))) (N …
    -/
    rw [norm_mul, norm_norm, ← add_mul, norm_mul]
    /-
      case a
      𝕜 : Type u_2
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_3
      F : Type u_4
      G : Type u_5
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      p : FormalMultilinearSeries 𝕜 E F
      q : FormalMultilinearSeries 𝕜 E G
      r : NNReal
      hr : And (LT.lt (↑r) p.radius) (LT.lt (↑r) q.radius)
      this : Asymptotics.IsBigO Filter.atTop (fun x => HAdd.hAdd (HMul.hMul (Norm.no …
      n : Nat
      ⊢ LE.le (HMul.hMul (Norm.norm (p.prod q n)) (Norm.norm (HPow.hPow (↑r) n))) (H …
    -/
    refine mul_le_mul_of_nonneg_right ?_ (norm_nonneg _)
    /-
      case a
      𝕜 : Type u_2
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_3
      F : Type u_4
      G : Type u_5
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      p : FormalMultilinearSeries 𝕜 E F
      q : FormalMultilinearSeries 𝕜 E G
      r : NNReal
      hr : And (LT.lt (↑r) p.radius) (LT.lt (↑r) q.radius)
      this : Asymptotics.IsBigO Filter.atTop (fun x => HAdd.hAdd (HMul.hMul (Norm.no …
      n : Nat
      ⊢ LE.le (Norm.norm (p.prod q n)) (Norm.norm (HAdd.hAdd (Norm.norm (p n)) (Norm …
    -/
    rw [FormalMultilinearSeries.prod, ContinuousMultilinearMap.opNorm_prod]
    /-
      case a
      𝕜 : Type u_2
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_3
      F : Type u_4
      G : Type u_5
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      p : FormalMultilinearSeries 𝕜 E F
      q : FormalMultilinearSeries 𝕜 E G
      r : NNReal
      hr : And (LT.lt (↑r) p.radius) (LT.lt (↑r) q.radius)
      this : Asymptotics.IsBigO Filter.atTop (fun x => HAdd.hAdd (HMul.hMul (Norm.no …
      n : Nat
      ⊢ LE.le (Max.max (Norm.norm (p n)) (Norm.norm (q n))) (Norm.norm (HAdd.hAdd (N …
    -/
    refine (max_le_add_of_nonneg (norm_nonneg _) (norm_nonneg _)).trans ?_
    /-
      case a
      𝕜 : Type u_2
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_3
      F : Type u_4
      G : Type u_5
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      p : FormalMultilinearSeries 𝕜 E F
      q : FormalMultilinearSeries 𝕜 E G
      r : NNReal
      hr : And (LT.lt (↑r) p.radius) (LT.lt (↑r) q.radius)
      this : Asymptotics.IsBigO Filter.atTop (fun x => HAdd.hAdd (HMul.hMul (Norm.no …
      n : Nat
      ⊢ LE.le (HAdd.hAdd (Norm.norm (p n)) (Norm.norm (q n))) (Norm.norm (HAdd.hAdd  …
    -/
    apply Real.le_norm_self
    /-
      🎉 no goals
    -/


lemma HasFPowerSeriesWithinOnBall.prod {e : E} {f : E → F} {g : E → G} {r s : ℝ≥0∞} {t : Set E}
    {p : FormalMultilinearSeries 𝕜 E F} {q : FormalMultilinearSeries 𝕜 E G}
    (hf : HasFPowerSeriesWithinOnBall f p t e r) (hg : HasFPowerSeriesWithinOnBall g q t e s) :
    HasFPowerSeriesWithinOnBall (fun x ↦ (f x, g x)) (p.prod q) t e (min r s) where
  r_le := by
    /-
      𝕜 : Type u_2
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_3
      F : Type u_4
      G : Type u_5
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      e : E
      f : E → F
      g : E → G
      r s : ENNReal
      t : Set E
      p : FormalMultilinearSeries 𝕜 E F
      q : FormalMultilinearSeries 𝕜 E G
      hf : HasFPowerSeriesWithinOnBall f p t e r
      hg : HasFPowerSeriesWithinOnBall g q t e s
      ⊢ LE.le (Min.min r s) (p.prod q).radius
    -/
    rw [p.radius_prod_eq_min]
    /-
      𝕜 : Type u_2
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_3
      F : Type u_4
      G : Type u_5
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      e : E
      f : E → F
      g : E → G
      r s : ENNReal
      t : Set E
      p : FormalMultilinearSeries 𝕜 E F
      q : FormalMultilinearSeries 𝕜 E G
      hf : HasFPowerSeriesWithinOnBall f p t e r
      hg : HasFPowerSeriesWithinOnBall g q t e s
      ⊢ LE.le (Min.min r s) (Min.min p.radius q.radius)
    -/
    exact min_le_min hf.r_le hg.r_le
    /-
      🎉 no goals
    -/
  r_pos := lt_min hf.r_pos hg.r_pos
  hasSum := by
    /-
      𝕜 : Type u_2
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_3
      F : Type u_4
      G : Type u_5
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      e : E
      f : E → F
      g : E → G
      r s : ENNReal
      t : Set E
      p : FormalMultilinearSeries 𝕜 E F
      q : FormalMultilinearSeries 𝕜 E G
      hf : HasFPowerSeriesWithinOnBall f p t e r
      hg : HasFPowerSeriesWithinOnBall g q t e s
      ⊢ ∀ {y : E}, Membership.mem (Insert.insert e t) (HAdd.hAdd e y) → Membership.m …
    -/
    intro y h'y hy
    /-
      𝕜 : Type u_2
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_3
      F : Type u_4
      G : Type u_5
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      e : E
      f : E → F
      g : E → G
      r s : ENNReal
      t : Set E
      p : FormalMultilinearSeries 𝕜 E F
      q : FormalMultilinearSeries 𝕜 E G
      hf : HasFPowerSeriesWithinOnBall f p t e r
      hg : HasFPowerSeriesWithinOnBall g q t e s
      y : E
      h'y : Membership.mem (Insert.insert e t) (HAdd.hAdd e y)
      hy : Membership.mem (EMetric.ball 0 (Min.min r s)) y
      ⊢ HasSum (fun n => (p.prod q n) fun x => y) { fst := f (HAdd.hAdd e y), snd := …
    -/
    simp_rw [FormalMultilinearSeries.prod, ContinuousMultilinearMap.prod_apply]
    /-
      𝕜 : Type u_2
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type u_3
      F : Type u_4
      G : Type u_5
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      e : E
      f : E → F
      g : E → G
      r s : ENNReal
      t : Set E
      p : FormalMultilinearSeries 𝕜 E F
      q : FormalMultilinearSeries 𝕜 E G
      hf : HasFPowerSeriesWithinOnBall f p t e r
      hg : HasFPowerSeriesWithinOnBall g q t e s
      y : E
      h'y : Membership.mem (Insert.insert e t) (HAdd.hAdd e y)
      hy : Membership.mem (EMetric.ball 0 (Min.min r s)) y
      ⊢ HasSum (fun n => { fst := (p n) fun x => y, snd := (q n) fun x => y }) { fst …
    -/
    refine (hf.hasSum h'y ?_).prod_mk (hg.hasSum h'y ?_)
      /-
        case refine_1
        𝕜 : Type u_2
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_3
        F : Type u_4
        G : Type u_5
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        e : E
        f : E → F
        g : E → G
        r s : ENNReal
        t : Set E
        p : FormalMultilinearSeries 𝕜 E F
        q : FormalMultilinearSeries 𝕜 E G
        hf : HasFPowerSeriesWithinOnBall f p t e r
        hg : HasFPowerSeriesWithinOnBall g q t e s
        y : E
        h'y : Membership.mem (Insert.insert e t) (HAdd.hAdd e y)
        hy : Membership.mem (EMetric.ball 0 (Min.min r s)) y
        ⊢ Membership.mem (EMetric.ball 0 r) y
      -/
    · exact EMetric.mem_ball.mpr (lt_of_lt_of_le hy (min_le_left _ _))
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        𝕜 : Type u_2
        inst✝⁶ : NontriviallyNormedField 𝕜
        E : Type u_3
        F : Type u_4
        G : Type u_5
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        inst✝¹ : NormedAddCommGroup G
        inst✝ : NormedSpace 𝕜 G
        e : E
        f : E → F
        g : E → G
        r s : ENNReal
        t : Set E
        p : FormalMultilinearSeries 𝕜 E F
        q : FormalMultilinearSeries 𝕜 E G
        hf : HasFPowerSeriesWithinOnBall f p t e r
        hg : HasFPowerSeriesWithinOnBall g q t e s
        y : E
        h'y : Membership.mem (Insert.insert e t) (HAdd.hAdd e y)
        hy : Membership.mem (EMetric.ball 0 (Min.min r s)) y
        ⊢ Membership.mem (EMetric.ball 0 s) y
      -/
    · exact EMetric.mem_ball.mpr (lt_of_lt_of_le hy (min_le_right _ _))
      /-
        🎉 no goals
      -/


lemma HasFPowerSeriesOnBall.prod {e : E} {f : E → F} {g : E → G} {r s : ℝ≥0∞}
    {p : FormalMultilinearSeries 𝕜 E F} {q : FormalMultilinearSeries 𝕜 E G}
    (hf : HasFPowerSeriesOnBall f p e r) (hg : HasFPowerSeriesOnBall g q e s) :
    HasFPowerSeriesOnBall (fun x ↦ (f x, g x)) (p.prod q) e (min r s) := by
  /-
    𝕜 : Type u_2
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_3
    F : Type u_4
    G : Type u_5
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    e : E
    f : E → F
    g : E → G
    r s : ENNReal
    p : FormalMultilinearSeries 𝕜 E F
    q : FormalMultilinearSeries 𝕜 E G
    hf : HasFPowerSeriesOnBall f p e r
    hg : HasFPowerSeriesOnBall g q e s
    ⊢ HasFPowerSeriesOnBall (fun x => { fst := f x, snd := g x }) (p.prod q) e (Mi …
  -/
  rw [← hasFPowerSeriesWithinOnBall_univ] at hf hg ⊢
  /-
    𝕜 : Type u_2
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_3
    F : Type u_4
    G : Type u_5
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    e : E
    f : E → F
    g : E → G
    r s : ENNReal
    p : FormalMultilinearSeries 𝕜 E F
    q : FormalMultilinearSeries 𝕜 E G
    hf : HasFPowerSeriesWithinOnBall f p Set.univ e r
    hg : HasFPowerSeriesWithinOnBall g q Set.univ e s
    ⊢ HasFPowerSeriesWithinOnBall (fun x => { fst := f x, snd := g x }) (p.prod q) …
  -/
  exact hf.prod hg
  /-
    🎉 no goals
  -/


lemma HasFPowerSeriesWithinAt.prod {e : E} {f : E → F} {g : E → G} {s : Set E}
    {p : FormalMultilinearSeries 𝕜 E F} {q : FormalMultilinearSeries 𝕜 E G}
    (hf : HasFPowerSeriesWithinAt f p s e) (hg : HasFPowerSeriesWithinAt g q s e) :
    HasFPowerSeriesWithinAt (fun x ↦ (f x, g x)) (p.prod q) s e := by
  /-
    𝕜 : Type u_2
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_3
    F : Type u_4
    G : Type u_5
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    e : E
    f : E → F
    g : E → G
    s : Set E
    p : FormalMultilinearSeries 𝕜 E F
    q : FormalMultilinearSeries 𝕜 E G
    hf : HasFPowerSeriesWithinAt f p s e
    hg : HasFPowerSeriesWithinAt g q s e
    ⊢ HasFPowerSeriesWithinAt (fun x => { fst := f x, snd := g x }) (p.prod q) s e
  -/
  rcases hf with ⟨_, hf⟩
  /-
    case intro
    𝕜 : Type u_2
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_3
    F : Type u_4
    G : Type u_5
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    e : E
    f : E → F
    g : E → G
    s : Set E
    p : FormalMultilinearSeries 𝕜 E F
    q : FormalMultilinearSeries 𝕜 E G
    hg : HasFPowerSeriesWithinAt g q s e
    w✝ : ENNReal
    hf : HasFPowerSeriesWithinOnBall f p s e w✝
    ⊢ HasFPowerSeriesWithinAt (fun x => { fst := f x, snd := g x }) (p.prod q) s e
  -/
  rcases hg with ⟨_, hg⟩
  /-
    case intro.intro
    𝕜 : Type u_2
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_3
    F : Type u_4
    G : Type u_5
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    e : E
    f : E → F
    g : E → G
    s : Set E
    p : FormalMultilinearSeries 𝕜 E F
    q : FormalMultilinearSeries 𝕜 E G
    w✝¹ : ENNReal
    hf : HasFPowerSeriesWithinOnBall f p s e w✝¹
    w✝ : ENNReal
    hg : HasFPowerSeriesWithinOnBall g q s e w✝
    ⊢ HasFPowerSeriesWithinAt (fun x => { fst := f x, snd := g x }) (p.prod q) s e
  -/
  exact ⟨_, hf.prod hg⟩
  /-
    🎉 no goals
  -/


lemma HasFPowerSeriesAt.prod {e : E} {f : E → F} {g : E → G}
    {p : FormalMultilinearSeries 𝕜 E F} {q : FormalMultilinearSeries 𝕜 E G}
    (hf : HasFPowerSeriesAt f p e) (hg : HasFPowerSeriesAt g q e) :
    HasFPowerSeriesAt (fun x ↦ (f x, g x)) (p.prod q) e := by
  /-
    𝕜 : Type u_2
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_3
    F : Type u_4
    G : Type u_5
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    e : E
    f : E → F
    g : E → G
    p : FormalMultilinearSeries 𝕜 E F
    q : FormalMultilinearSeries 𝕜 E G
    hf : HasFPowerSeriesAt f p e
    hg : HasFPowerSeriesAt g q e
    ⊢ HasFPowerSeriesAt (fun x => { fst := f x, snd := g x }) (p.prod q) e
  -/
  rcases hf with ⟨_, hf⟩
  /-
    case intro
    𝕜 : Type u_2
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_3
    F : Type u_4
    G : Type u_5
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    e : E
    f : E → F
    g : E → G
    p : FormalMultilinearSeries 𝕜 E F
    q : FormalMultilinearSeries 𝕜 E G
    hg : HasFPowerSeriesAt g q e
    w✝ : ENNReal
    hf : HasFPowerSeriesOnBall f p e w✝
    ⊢ HasFPowerSeriesAt (fun x => { fst := f x, snd := g x }) (p.prod q) e
  -/
  rcases hg with ⟨_, hg⟩
  /-
    case intro.intro
    𝕜 : Type u_2
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_3
    F : Type u_4
    G : Type u_5
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    e : E
    f : E → F
    g : E → G
    p : FormalMultilinearSeries 𝕜 E F
    q : FormalMultilinearSeries 𝕜 E G
    w✝¹ : ENNReal
    hf : HasFPowerSeriesOnBall f p e w✝¹
    w✝ : ENNReal
    hg : HasFPowerSeriesOnBall g q e w✝
    ⊢ HasFPowerSeriesAt (fun x => { fst := f x, snd := g x }) (p.prod q) e
  -/
  exact ⟨_, hf.prod hg⟩
  /-
    🎉 no goals
  -/


/-- The Cartesian product of analytic functions is analytic. -/
lemma AnalyticWithinAt.prod {e : E} {f : E → F} {g : E → G} {s : Set E}
    (hf : AnalyticWithinAt 𝕜 f s e) (hg : AnalyticWithinAt 𝕜 g s e) :
    AnalyticWithinAt 𝕜 (fun x ↦ (f x, g x)) s e := by
  /-
    𝕜 : Type u_2
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_3
    F : Type u_4
    G : Type u_5
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    e : E
    f : E → F
    g : E → G
    s : Set E
    hf : AnalyticWithinAt 𝕜 f s e
    hg : AnalyticWithinAt 𝕜 g s e
    ⊢ AnalyticWithinAt 𝕜 (fun x => { fst := f x, snd := g x }) s e
  -/
  rcases hf with ⟨_, hf⟩
  /-
    case intro
    𝕜 : Type u_2
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_3
    F : Type u_4
    G : Type u_5
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    e : E
    f : E → F
    g : E → G
    s : Set E
    hg : AnalyticWithinAt 𝕜 g s e
    w✝ : FormalMultilinearSeries 𝕜 E F
    hf : HasFPowerSeriesWithinAt f w✝ s e
    ⊢ AnalyticWithinAt 𝕜 (fun x => { fst := f x, snd := g x }) s e
  -/
  rcases hg with ⟨_, hg⟩
  /-
    case intro.intro
    𝕜 : Type u_2
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_3
    F : Type u_4
    G : Type u_5
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    e : E
    f : E → F
    g : E → G
    s : Set E
    w✝¹ : FormalMultilinearSeries 𝕜 E F
    hf : HasFPowerSeriesWithinAt f w✝¹ s e
    w✝ : FormalMultilinearSeries 𝕜 E G
    hg : HasFPowerSeriesWithinAt g w✝ s e
    ⊢ AnalyticWithinAt 𝕜 (fun x => { fst := f x, snd := g x }) s e
  -/
  exact ⟨_, hf.prod hg⟩
  /-
    🎉 no goals
  -/


/-- The Cartesian product of analytic functions is analytic. -/
lemma AnalyticAt.prod {e : E} {f : E → F} {g : E → G}
    (hf : AnalyticAt 𝕜 f e) (hg : AnalyticAt 𝕜 g e) :
    AnalyticAt 𝕜 (fun x ↦ (f x, g x)) e := by
  /-
    𝕜 : Type u_2
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_3
    F : Type u_4
    G : Type u_5
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    e : E
    f : E → F
    g : E → G
    hf : AnalyticAt 𝕜 f e
    hg : AnalyticAt 𝕜 g e
    ⊢ AnalyticAt 𝕜 (fun x => { fst := f x, snd := g x }) e
  -/
  rcases hf with ⟨_, hf⟩
  /-
    case intro
    𝕜 : Type u_2
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_3
    F : Type u_4
    G : Type u_5
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    e : E
    f : E → F
    g : E → G
    hg : AnalyticAt 𝕜 g e
    w✝ : FormalMultilinearSeries 𝕜 E F
    hf : HasFPowerSeriesAt f w✝ e
    ⊢ AnalyticAt 𝕜 (fun x => { fst := f x, snd := g x }) e
  -/
  rcases hg with ⟨_, hg⟩
  /-
    case intro.intro
    𝕜 : Type u_2
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_3
    F : Type u_4
    G : Type u_5
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    e : E
    f : E → F
    g : E → G
    w✝¹ : FormalMultilinearSeries 𝕜 E F
    hf : HasFPowerSeriesAt f w✝¹ e
    w✝ : FormalMultilinearSeries 𝕜 E G
    hg : HasFPowerSeriesAt g w✝ e
    ⊢ AnalyticAt 𝕜 (fun x => { fst := f x, snd := g x }) e
  -/
  exact ⟨_, hf.prod hg⟩
  /-
    🎉 no goals
  -/


/-- The Cartesian product of analytic functions within a set is analytic. -/
lemma AnalyticOn.prod {f : E → F} {g : E → G} {s : Set E}
    (hf : AnalyticOn 𝕜 f s) (hg : AnalyticOn 𝕜 g s) :
    AnalyticOn 𝕜 (fun x ↦ (f x, g x)) s :=
  fun x hx ↦ (hf x hx).prod (hg x hx)


@[deprecated (since := "2024-09-26")]
alias AnalyticWithinOn.prod := AnalyticOn.prod


/-- The Cartesian product of analytic functions is analytic. -/
lemma AnalyticOnNhd.prod {f : E → F} {g : E → G} {s : Set E}
    (hf : AnalyticOnNhd 𝕜 f s) (hg : AnalyticOnNhd 𝕜 g s) :
    AnalyticOnNhd 𝕜 (fun x ↦ (f x, g x)) s :=
  fun x hx ↦ (hf x hx).prod (hg x hx)


/-- `AnalyticAt.comp` for functions on product spaces -/
theorem AnalyticAt.comp₂ {h : F × G → H} {f : E → F} {g : E → G} {x : E}
    (ha : AnalyticAt 𝕜 h (f x, g x)) (fa : AnalyticAt 𝕜 f x)
    (ga : AnalyticAt 𝕜 g x) :
    AnalyticAt 𝕜 (fun x ↦ h (f x, g x)) x :=
  AnalyticAt.comp ha (fa.prod ga)


/-- `AnalyticWithinAt.comp` for functions on product spaces -/
theorem AnalyticWithinAt.comp₂ {h : F × G → H} {f : E → F} {g : E → G} {s : Set (F × G)}
    {t : Set E} {x : E}
    (ha : AnalyticWithinAt 𝕜 h s (f x, g x)) (fa : AnalyticWithinAt 𝕜 f t x)
    (ga : AnalyticWithinAt 𝕜 g t x) (hf : Set.MapsTo (fun y ↦ (f y, g y)) t s) :
    AnalyticWithinAt 𝕜 (fun x ↦ h (f x, g x)) t x :=
  AnalyticWithinAt.comp ha (fa.prod ga) hf


/-- `AnalyticAt.comp_analyticWithinAt` for functions on product spaces -/
theorem AnalyticAt.comp₂_analyticWithinAt
    {h : F × G → H} {f : E → F} {g : E → G} {x : E} {s : Set E}
    (ha : AnalyticAt 𝕜 h (f x, g x)) (fa : AnalyticWithinAt 𝕜 f s x)
    (ga : AnalyticWithinAt 𝕜 g s x) :
    AnalyticWithinAt 𝕜 (fun x ↦ h (f x, g x)) s x :=
  AnalyticAt.comp_analyticWithinAt ha (fa.prod ga)


/-- `AnalyticOnNhd.comp` for functions on product spaces -/
theorem AnalyticOnNhd.comp₂ {h : F × G → H} {f : E → F} {g : E → G} {s : Set (F × G)} {t : Set E}
    (ha : AnalyticOnNhd 𝕜 h s) (fa : AnalyticOnNhd 𝕜 f t) (ga : AnalyticOnNhd 𝕜 g t)
    (m : ∀ x, x ∈ t → (f x, g x) ∈ s) : AnalyticOnNhd 𝕜 (fun x ↦ h (f x, g x)) t :=
  fun _ xt ↦ (ha _ (m _ xt)).comp₂ (fa _ xt) (ga _ xt)


/-- `AnalyticOn.comp` for functions on product spaces -/
theorem AnalyticOn.comp₂ {h : F × G → H} {f : E → F} {g : E → G} {s : Set (F × G)}
    {t : Set E}
    (ha : AnalyticOn 𝕜 h s) (fa : AnalyticOn 𝕜 f t)
    (ga : AnalyticOn 𝕜 g t) (m : Set.MapsTo (fun y ↦ (f y, g y)) t s) :
    AnalyticOn 𝕜 (fun x ↦ h (f x, g x)) t :=
  fun x hx ↦ (ha _ (m hx)).comp₂ (fa x hx) (ga x hx) m


@[deprecated (since := "2024-09-26")]
alias AnalyticWithinOn.comp₂ := AnalyticOn.comp₂


/-- Analytic functions on products are analytic in the first coordinate -/
theorem AnalyticAt.curry_left {f : E × F → G} {p : E × F} (fa : AnalyticAt 𝕜 f p) :
    AnalyticAt 𝕜 (fun x ↦ f (x, p.2)) p.1 :=
  AnalyticAt.comp₂ fa analyticAt_id analyticAt_const

alias AnalyticAt.along_fst := AnalyticAt.curry_left


theorem AnalyticWithinAt.curry_left
    {f : E × F → G} {s : Set (E × F)} {p : E × F} (fa : AnalyticWithinAt 𝕜 f s p) :
    AnalyticWithinAt 𝕜 (fun x ↦ f (x, p.2)) {x | (x, p.2) ∈ s} p.1 :=
  AnalyticWithinAt.comp₂ fa analyticWithinAt_id analyticWithinAt_const (fun _ hx ↦ hx)


/-- Analytic functions on products are analytic in the second coordinate -/
theorem AnalyticAt.curry_right {f : E × F → G} {p : E × F} (fa : AnalyticAt 𝕜 f p) :
    AnalyticAt 𝕜 (fun y ↦ f (p.1, y)) p.2 :=
  AnalyticAt.comp₂ fa analyticAt_const analyticAt_id

alias AnalyticAt.along_snd := AnalyticAt.curry_right


theorem AnalyticWithinAt.curry_right
    {f : E × F → G} {s : Set (E × F)} {p : E × F} (fa : AnalyticWithinAt 𝕜 f s p) :
    AnalyticWithinAt 𝕜 (fun y ↦ f (p.1, y)) {y | (p.1, y) ∈ s} p.2 :=
  AnalyticWithinAt.comp₂ fa  analyticWithinAt_const analyticWithinAt_id (fun _ hx ↦ hx)


/-- Analytic functions on products are analytic in the first coordinate -/
theorem AnalyticOnNhd.curry_left {f : E × F → G} {s : Set (E × F)} {y : F}
    (fa : AnalyticOnNhd 𝕜 f s) :
    AnalyticOnNhd 𝕜 (fun x ↦ f (x, y)) {x | (x, y) ∈ s} :=
  fun x m ↦ (fa (x, y) m).curry_left

alias AnalyticOnNhd.along_fst := AnalyticOnNhd.curry_left


@[deprecated (since := "2024-09-26")]
alias AnalyticOn.along_fst := AnalyticOnNhd.curry_left


theorem AnalyticOn.curry_left
    {f : E × F → G} {s : Set (E × F)} {y : F} (fa : AnalyticOn 𝕜 f s) :
    AnalyticOn 𝕜 (fun x ↦ f (x, y)) {x | (x, y) ∈ s} :=
  fun x m ↦ (fa (x, y) m).curry_left


@[deprecated (since := "2024-09-26")]
alias AnalyticWithinOn.curry_left := AnalyticOn.curry_left


/-- Analytic functions on products are analytic in the second coordinate -/
theorem AnalyticOnNhd.curry_right {f : E × F → G} {x : E} {s : Set (E × F)}
    (fa : AnalyticOnNhd 𝕜 f s) :
    AnalyticOnNhd 𝕜 (fun y ↦ f (x, y)) {y | (x, y) ∈ s} :=
  fun y m ↦ (fa (x, y) m).curry_right

alias AnalyticOnNhd.along_snd := AnalyticOnNhd.curry_right


@[deprecated (since := "2024-09-26")]
alias AnalyticOn.along_snd := AnalyticOnNhd.curry_right


theorem AnalyticOn.curry_right
    {f : E × F → G} {x : E} {s : Set (E × F)} (fa : AnalyticOn 𝕜 f s) :
    AnalyticOn 𝕜 (fun y ↦ f (x, y)) {y | (x, y) ∈ s} :=
  fun y m ↦ (fa (x, y) m).curry_right


@[deprecated (since := "2024-09-26")]
alias AnalyticWithinOn.curry_right := AnalyticOn.curry_right


lemma FormalMultilinearSeries.radius_pi_le (p : Π i, FormalMultilinearSeries 𝕜 E (Fm i)) (i : ι) :
    (FormalMultilinearSeries.pi p).radius ≤ (p i).radius := by
  /-
    𝕜 : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    ι : Type u_9
    inst✝² : Fintype ι
    Fm : ι → Type u_10
    inst✝¹ : (i : ι) → NormedAddCommGroup (Fm i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (Fm i)
    p : (i : ι) → FormalMultilinearSeries 𝕜 E (Fm i)
    i : ι
    ⊢ LE.le (FormalMultilinearSeries.pi p).radius (p i).radius
  -/
  apply le_of_forall_nnreal_lt (fun r' hr' ↦ ?_)
  /-
    𝕜 : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    ι : Type u_9
    inst✝² : Fintype ι
    Fm : ι → Type u_10
    inst✝¹ : (i : ι) → NormedAddCommGroup (Fm i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (Fm i)
    p : (i : ι) → FormalMultilinearSeries 𝕜 E (Fm i)
    i : ι
    r' : NNReal
    hr' : LT.lt (↑r') (FormalMultilinearSeries.pi p).radius
    ⊢ LE.le (↑r') (p i).radius
  -/
  obtain ⟨C, -, hC⟩ : ∃ C > 0, ∀ n, ‖pi p n‖ * ↑r' ^ n ≤ C := norm_mul_pow_le_of_lt_radius _ hr'
  /-
    case intro.intro
    𝕜 : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    ι : Type u_9
    inst✝² : Fintype ι
    Fm : ι → Type u_10
    inst✝¹ : (i : ι) → NormedAddCommGroup (Fm i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (Fm i)
    p : (i : ι) → FormalMultilinearSeries 𝕜 E (Fm i)
    i : ι
    r' : NNReal
    hr' : LT.lt (↑r') (FormalMultilinearSeries.pi p).radius
    C : Real
    hC : ∀ (n : Nat), LE.le (HMul.hMul (Norm.norm (FormalMultilinearSeries.pi p n) …
    ⊢ LE.le (↑r') (p i).radius
  -/
  apply le_radius_of_bound _ C (fun n ↦ ?_)
  /-
    𝕜 : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    ι : Type u_9
    inst✝² : Fintype ι
    Fm : ι → Type u_10
    inst✝¹ : (i : ι) → NormedAddCommGroup (Fm i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (Fm i)
    p : (i : ι) → FormalMultilinearSeries 𝕜 E (Fm i)
    i : ι
    r' : NNReal
    hr' : LT.lt (↑r') (FormalMultilinearSeries.pi p).radius
    C : Real
    hC : ∀ (n : Nat), LE.le (HMul.hMul (Norm.norm (FormalMultilinearSeries.pi p n) …
    n : Nat
    ⊢ LE.le (HMul.hMul (Norm.norm (p i n)) (HPow.hPow (↑r') n)) C
  -/
  apply le_trans _ (hC n)
  /-
    𝕜 : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    ι : Type u_9
    inst✝² : Fintype ι
    Fm : ι → Type u_10
    inst✝¹ : (i : ι) → NormedAddCommGroup (Fm i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (Fm i)
    p : (i : ι) → FormalMultilinearSeries 𝕜 E (Fm i)
    i : ι
    r' : NNReal
    hr' : LT.lt (↑r') (FormalMultilinearSeries.pi p).radius
    C : Real
    hC : ∀ (n : Nat), LE.le (HMul.hMul (Norm.norm (FormalMultilinearSeries.pi p n) …
    n : Nat
    ⊢ LE.le (HMul.hMul (Norm.norm (p i n)) (HPow.hPow (↑r') n)) (HMul.hMul (Norm.n …
  -/
  gcongr
  /-
    case h
    𝕜 : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    ι : Type u_9
    inst✝² : Fintype ι
    Fm : ι → Type u_10
    inst✝¹ : (i : ι) → NormedAddCommGroup (Fm i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (Fm i)
    p : (i : ι) → FormalMultilinearSeries 𝕜 E (Fm i)
    i : ι
    r' : NNReal
    hr' : LT.lt (↑r') (FormalMultilinearSeries.pi p).radius
    C : Real
    hC : ∀ (n : Nat), LE.le (HMul.hMul (Norm.norm (FormalMultilinearSeries.pi p n) …
    n : Nat
    ⊢ LE.le (Norm.norm (p i n)) (Norm.norm (FormalMultilinearSeries.pi p n))
  -/
  rw [pi, ContinuousMultilinearMap.opNorm_pi]
  /-
    case h
    𝕜 : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    ι : Type u_9
    inst✝² : Fintype ι
    Fm : ι → Type u_10
    inst✝¹ : (i : ι) → NormedAddCommGroup (Fm i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (Fm i)
    p : (i : ι) → FormalMultilinearSeries 𝕜 E (Fm i)
    i : ι
    r' : NNReal
    hr' : LT.lt (↑r') (FormalMultilinearSeries.pi p).radius
    C : Real
    hC : ∀ (n : Nat), LE.le (HMul.hMul (Norm.norm (FormalMultilinearSeries.pi p n) …
    n : Nat
    ⊢ LE.le (Norm.norm (p i n)) (Norm.norm fun i => p i n)
  -/
  exact norm_le_pi_norm (fun i ↦ p i n) i
  /-
    🎉 no goals
  -/


lemma FormalMultilinearSeries.le_radius_pi (h : ∀ i, r ≤ (p i).radius) :
    r ≤ (FormalMultilinearSeries.pi p).radius := by
  /-
    𝕜 : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    ι : Type u_9
    inst✝² : Fintype ι
    Fm : ι → Type u_10
    inst✝¹ : (i : ι) → NormedAddCommGroup (Fm i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (Fm i)
    r : ENNReal
    p : (i : ι) → FormalMultilinearSeries 𝕜 E (Fm i)
    h : ∀ (i : ι), LE.le r (p i).radius
    ⊢ LE.le r (FormalMultilinearSeries.pi p).radius
  -/
  apply le_of_forall_nnreal_lt (fun r' hr' ↦ ?_)
  have I i : ∃ C > 0, ∀ n, ‖p i n‖ * (r' : ℝ) ^ n ≤ C :=
    norm_mul_pow_le_of_lt_radius _ (hr'.trans_le (h i))
  /-
    𝕜 : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    ι : Type u_9
    inst✝² : Fintype ι
    Fm : ι → Type u_10
    inst✝¹ : (i : ι) → NormedAddCommGroup (Fm i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (Fm i)
    r : ENNReal
    p : (i : ι) → FormalMultilinearSeries 𝕜 E (Fm i)
    h : ∀ (i : ι), LE.le r (p i).radius
    r' : NNReal
    hr' : LT.lt (↑r') r
    I : ∀ (i : ι), Exists fun C => And (GT.gt C 0) (∀ (n : Nat), LE.le (HMul.hMul  …
    ⊢ LE.le (↑r') (FormalMultilinearSeries.pi p).radius
  -/
  choose C C_pos hC using I
  obtain ⟨D, D_nonneg, hD⟩ : ∃ D ≥ 0, ∀ i, C i ≤ D :=
    ⟨∑ i, C i, Finset.sum_nonneg (fun i _ ↦ (C_pos i).le),
      fun i ↦ Finset.single_le_sum (fun j _ ↦ (C_pos j).le) (Finset.mem_univ _)⟩
  /-
    case intro.intro
    𝕜 : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    ι : Type u_9
    inst✝² : Fintype ι
    Fm : ι → Type u_10
    inst✝¹ : (i : ι) → NormedAddCommGroup (Fm i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (Fm i)
    r : ENNReal
    p : (i : ι) → FormalMultilinearSeries 𝕜 E (Fm i)
    h : ∀ (i : ι), LE.le r (p i).radius
    r' : NNReal
    hr' : LT.lt (↑r') r
    C : ι → Real
    C_pos : ∀ (i : ι), GT.gt (C i) 0
    hC : ∀ (i : ι) (n : Nat), LE.le (HMul.hMul (Norm.norm (p i n)) (HPow.hPow (↑r' …
    D : Real
    D_nonneg : GE.ge D 0
    hD : ∀ (i : ι), LE.le (C i) D
    ⊢ LE.le (↑r') (FormalMultilinearSeries.pi p).radius
  -/
  apply le_radius_of_bound _ D (fun n ↦ ?_)
  /-
    𝕜 : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    ι : Type u_9
    inst✝² : Fintype ι
    Fm : ι → Type u_10
    inst✝¹ : (i : ι) → NormedAddCommGroup (Fm i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (Fm i)
    r : ENNReal
    p : (i : ι) → FormalMultilinearSeries 𝕜 E (Fm i)
    h : ∀ (i : ι), LE.le r (p i).radius
    r' : NNReal
    hr' : LT.lt (↑r') r
    C : ι → Real
    C_pos : ∀ (i : ι), GT.gt (C i) 0
    hC : ∀ (i : ι) (n : Nat), LE.le (HMul.hMul (Norm.norm (p i n)) (HPow.hPow (↑r' …
    D : Real
    D_nonneg : GE.ge D 0
    hD : ∀ (i : ι), LE.le (C i) D
    n : Nat
    ⊢ LE.le (HMul.hMul (Norm.norm (FormalMultilinearSeries.pi p n)) (HPow.hPow (↑r …
  -/
  rcases le_or_lt ((r' : ℝ)^n) 0 with hr' | hr'
    /-
      case inl
      𝕜 : Type u_2
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      ι : Type u_9
      inst✝² : Fintype ι
      Fm : ι → Type u_10
      inst✝¹ : (i : ι) → NormedAddCommGroup (Fm i)
      inst✝ : (i : ι) → NormedSpace 𝕜 (Fm i)
      r : ENNReal
      p : (i : ι) → FormalMultilinearSeries 𝕜 E (Fm i)
      h : ∀ (i : ι), LE.le r (p i).radius
      r' : NNReal
      hr'✝ : LT.lt (↑r') r
      C : ι → Real
      C_pos : ∀ (i : ι), GT.gt (C i) 0
      hC : ∀ (i : ι) (n : Nat), LE.le (HMul.hMul (Norm.norm (p i n)) (HPow.hPow (↑r' …
      D : Real
      D_nonneg : GE.ge D 0
      hD : ∀ (i : ι), LE.le (C i) D
      n : Nat
      hr' : LE.le (HPow.hPow (↑r') n) 0
      ⊢ LE.le (HMul.hMul (Norm.norm (FormalMultilinearSeries.pi p n)) (HPow.hPow (↑r …
    -/
  · exact le_trans (mul_nonpos_of_nonneg_of_nonpos (by positivity) hr') D_nonneg
    /-
      🎉 no goals
    -/
    /-
      case inr
      𝕜 : Type u_2
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      ι : Type u_9
      inst✝² : Fintype ι
      Fm : ι → Type u_10
      inst✝¹ : (i : ι) → NormedAddCommGroup (Fm i)
      inst✝ : (i : ι) → NormedSpace 𝕜 (Fm i)
      r : ENNReal
      p : (i : ι) → FormalMultilinearSeries 𝕜 E (Fm i)
      h : ∀ (i : ι), LE.le r (p i).radius
      r' : NNReal
      hr'✝ : LT.lt (↑r') r
      C : ι → Real
      C_pos : ∀ (i : ι), GT.gt (C i) 0
      hC : ∀ (i : ι) (n : Nat), LE.le (HMul.hMul (Norm.norm (p i n)) (HPow.hPow (↑r' …
      D : Real
      D_nonneg : GE.ge D 0
      hD : ∀ (i : ι), LE.le (C i) D
      n : Nat
      hr' : LT.lt 0 (HPow.hPow (↑r') n)
      ⊢ LE.le (HMul.hMul (Norm.norm (FormalMultilinearSeries.pi p n)) (HPow.hPow (↑r …
    -/
  · simp only [pi]
    rw [← le_div_iff₀ hr', ContinuousMultilinearMap.opNorm_pi,
      pi_norm_le_iff_of_nonneg (by positivity)]
    /-
      case inr
      𝕜 : Type u_2
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      ι : Type u_9
      inst✝² : Fintype ι
      Fm : ι → Type u_10
      inst✝¹ : (i : ι) → NormedAddCommGroup (Fm i)
      inst✝ : (i : ι) → NormedSpace 𝕜 (Fm i)
      r : ENNReal
      p : (i : ι) → FormalMultilinearSeries 𝕜 E (Fm i)
      h : ∀ (i : ι), LE.le r (p i).radius
      r' : NNReal
      hr'✝ : LT.lt (↑r') r
      C : ι → Real
      C_pos : ∀ (i : ι), GT.gt (C i) 0
      hC : ∀ (i : ι) (n : Nat), LE.le (HMul.hMul (Norm.norm (p i n)) (HPow.hPow (↑r' …
      D : Real
      D_nonneg : GE.ge D 0
      hD : ∀ (i : ι), LE.le (C i) D
      n : Nat
      hr' : LT.lt 0 (HPow.hPow (↑r') n)
      ⊢ ∀ (i : ι), LE.le (Norm.norm (p i n)) (HDiv.hDiv D (HPow.hPow (↑r') n))
    -/
    intro i
    /-
      case inr
      𝕜 : Type u_2
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      ι : Type u_9
      inst✝² : Fintype ι
      Fm : ι → Type u_10
      inst✝¹ : (i : ι) → NormedAddCommGroup (Fm i)
      inst✝ : (i : ι) → NormedSpace 𝕜 (Fm i)
      r : ENNReal
      p : (i : ι) → FormalMultilinearSeries 𝕜 E (Fm i)
      h : ∀ (i : ι), LE.le r (p i).radius
      r' : NNReal
      hr'✝ : LT.lt (↑r') r
      C : ι → Real
      C_pos : ∀ (i : ι), GT.gt (C i) 0
      hC : ∀ (i : ι) (n : Nat), LE.le (HMul.hMul (Norm.norm (p i n)) (HPow.hPow (↑r' …
      D : Real
      D_nonneg : GE.ge D 0
      hD : ∀ (i : ι), LE.le (C i) D
      n : Nat
      hr' : LT.lt 0 (HPow.hPow (↑r') n)
      i : ι
      ⊢ LE.le (Norm.norm (p i n)) (HDiv.hDiv D (HPow.hPow (↑r') n))
    -/
    exact (le_div_iff₀ hr').2 ((hC i n).trans (hD i))
    /-
      🎉 no goals
    -/


lemma FormalMultilinearSeries.radius_pi_eq_iInf :
    (FormalMultilinearSeries.pi p).radius = ⨅ i, (p i).radius := by
  /-
    𝕜 : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    ι : Type u_9
    inst✝² : Fintype ι
    Fm : ι → Type u_10
    inst✝¹ : (i : ι) → NormedAddCommGroup (Fm i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (Fm i)
    p : (i : ι) → FormalMultilinearSeries 𝕜 E (Fm i)
    ⊢ Eq (FormalMultilinearSeries.pi p).radius (iInf fun i => (p i).radius)
  -/
  refine le_antisymm (by simp [radius_pi_le]) ?_
  /-
    𝕜 : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    ι : Type u_9
    inst✝² : Fintype ι
    Fm : ι → Type u_10
    inst✝¹ : (i : ι) → NormedAddCommGroup (Fm i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (Fm i)
    p : (i : ι) → FormalMultilinearSeries 𝕜 E (Fm i)
    ⊢ LE.le (iInf fun i => (p i).radius) (FormalMultilinearSeries.pi p).radius
  -/
  apply le_of_forall_nnreal_lt (fun r' hr' ↦ ?_)
  /-
    𝕜 : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    ι : Type u_9
    inst✝² : Fintype ι
    Fm : ι → Type u_10
    inst✝¹ : (i : ι) → NormedAddCommGroup (Fm i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (Fm i)
    p : (i : ι) → FormalMultilinearSeries 𝕜 E (Fm i)
    r' : NNReal
    hr' : LT.lt (↑r') (iInf fun i => (p i).radius)
    ⊢ LE.le (↑r') (FormalMultilinearSeries.pi p).radius
  -/
  exact le_radius_pi (fun i ↦ le_iInf_iff.1 hr'.le i)
  /-
    🎉 no goals
  -/


/-- If each function in a finite family has a power series within a ball, then so does the
family as a whole. Note that the positivity assumption on the radius is only needed when
the family is empty. -/
lemma HasFPowerSeriesWithinOnBall.pi
    (hf : ∀ i, HasFPowerSeriesWithinOnBall (f i) (p i) s e r) (hr : 0 < r) :
    HasFPowerSeriesWithinOnBall (fun x ↦ (f · x)) (FormalMultilinearSeries.pi p) s e r where
  r_le := by
    /-
      𝕜 : Type u_2
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      ι : Type u_9
      inst✝² : Fintype ι
      e : E
      Fm : ι → Type u_10
      inst✝¹ : (i : ι) → NormedAddCommGroup (Fm i)
      inst✝ : (i : ι) → NormedSpace 𝕜 (Fm i)
      f : (i : ι) → E → Fm i
      s : Set E
      r : ENNReal
      p : (i : ι) → FormalMultilinearSeries 𝕜 E (Fm i)
      hf : ∀ (i : ι), HasFPowerSeriesWithinOnBall (f i) (p i) s e r
      hr : LT.lt 0 r
      ⊢ LE.le r (FormalMultilinearSeries.pi p).radius
    -/
    apply FormalMultilinearSeries.le_radius_pi (fun i ↦ ?_)
    /-
      𝕜 : Type u_2
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      ι : Type u_9
      inst✝² : Fintype ι
      e : E
      Fm : ι → Type u_10
      inst✝¹ : (i : ι) → NormedAddCommGroup (Fm i)
      inst✝ : (i : ι) → NormedSpace 𝕜 (Fm i)
      f : (i : ι) → E → Fm i
      s : Set E
      r : ENNReal
      p : (i : ι) → FormalMultilinearSeries 𝕜 E (Fm i)
      hf : ∀ (i : ι), HasFPowerSeriesWithinOnBall (f i) (p i) s e r
      hr : LT.lt 0 r
      i : ι
      ⊢ LE.le r (p i).radius
    -/
    exact (hf i).r_le
    /-
      🎉 no goals
    -/
  r_pos := hr
  hasSum {_} m hy := Pi.hasSum.2 (fun i ↦ (hf i).hasSum m hy)


lemma hasFPowerSeriesWithinOnBall_pi_iff (hr : 0 < r) :
    HasFPowerSeriesWithinOnBall (fun x ↦ (f · x)) (FormalMultilinearSeries.pi p) s e r ↔
      ∀ i, HasFPowerSeriesWithinOnBall (f i) (p i) s e r where
  mp h i :=
    ⟨h.r_le.trans (FormalMultilinearSeries.radius_pi_le _ _), hr,
      fun m hy ↦ Pi.hasSum.1 (h.hasSum m hy) i⟩
  mpr h := .pi h hr


lemma HasFPowerSeriesOnBall.pi
    (hf : ∀ i, HasFPowerSeriesOnBall (f i) (p i) e r) (hr : 0 < r) :
    HasFPowerSeriesOnBall (fun x ↦ (f · x)) (FormalMultilinearSeries.pi p) e r := by
  /-
    𝕜 : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    ι : Type u_9
    inst✝² : Fintype ι
    e : E
    Fm : ι → Type u_10
    inst✝¹ : (i : ι) → NormedAddCommGroup (Fm i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (Fm i)
    f : (i : ι) → E → Fm i
    r : ENNReal
    p : (i : ι) → FormalMultilinearSeries 𝕜 E (Fm i)
    hf : ∀ (i : ι), HasFPowerSeriesOnBall (f i) (p i) e r
    hr : LT.lt 0 r
    ⊢ HasFPowerSeriesOnBall (fun x x_1 => f x_1 x) (FormalMultilinearSeries.pi p)  …
  -/
  simp_rw [← hasFPowerSeriesWithinOnBall_univ] at hf ⊢
  /-
    𝕜 : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    ι : Type u_9
    inst✝² : Fintype ι
    e : E
    Fm : ι → Type u_10
    inst✝¹ : (i : ι) → NormedAddCommGroup (Fm i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (Fm i)
    f : (i : ι) → E → Fm i
    r : ENNReal
    p : (i : ι) → FormalMultilinearSeries 𝕜 E (Fm i)
    hr : LT.lt 0 r
    hf : ∀ (i : ι), HasFPowerSeriesWithinOnBall (f i) (p i) Set.univ e r
    ⊢ HasFPowerSeriesWithinOnBall (fun x x_1 => f x_1 x) (FormalMultilinearSeries. …
  -/
  exact HasFPowerSeriesWithinOnBall.pi hf hr
  /-
    🎉 no goals
  -/


lemma hasFPowerSeriesOnBall_pi_iff (hr : 0 < r) :
    HasFPowerSeriesOnBall (fun x ↦ (f · x)) (FormalMultilinearSeries.pi p) e r ↔
      ∀ i, HasFPowerSeriesOnBall (f i) (p i) e r := by
  /-
    𝕜 : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    ι : Type u_9
    inst✝² : Fintype ι
    e : E
    Fm : ι → Type u_10
    inst✝¹ : (i : ι) → NormedAddCommGroup (Fm i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (Fm i)
    f : (i : ι) → E → Fm i
    r : ENNReal
    p : (i : ι) → FormalMultilinearSeries 𝕜 E (Fm i)
    hr : LT.lt 0 r
    ⊢ Iff (HasFPowerSeriesOnBall (fun x x_1 => f x_1 x) (FormalMultilinearSeries.p …
  -/
  simp_rw [← hasFPowerSeriesWithinOnBall_univ]
  /-
    𝕜 : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    ι : Type u_9
    inst✝² : Fintype ι
    e : E
    Fm : ι → Type u_10
    inst✝¹ : (i : ι) → NormedAddCommGroup (Fm i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (Fm i)
    f : (i : ι) → E → Fm i
    r : ENNReal
    p : (i : ι) → FormalMultilinearSeries 𝕜 E (Fm i)
    hr : LT.lt 0 r
    ⊢ Iff (HasFPowerSeriesWithinOnBall (fun x x_1 => f x_1 x) (FormalMultilinearSe …
  -/
  exact hasFPowerSeriesWithinOnBall_pi_iff hr
  /-
    🎉 no goals
  -/


lemma HasFPowerSeriesWithinAt.pi
    (hf : ∀ i, HasFPowerSeriesWithinAt (f i) (p i) s e) :
    HasFPowerSeriesWithinAt (fun x ↦ (f · x)) (FormalMultilinearSeries.pi p) s e := by
  have : ∀ᶠ r in 𝓝[>] 0, ∀ i, HasFPowerSeriesWithinOnBall (f i) (p i) s e r :=
    eventually_all.mpr (fun i ↦ (hf i).eventually)
  /-
    𝕜 : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    ι : Type u_9
    inst✝² : Fintype ι
    e : E
    Fm : ι → Type u_10
    inst✝¹ : (i : ι) → NormedAddCommGroup (Fm i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (Fm i)
    f : (i : ι) → E → Fm i
    s : Set E
    p : (i : ι) → FormalMultilinearSeries 𝕜 E (Fm i)
    hf : ∀ (i : ι), HasFPowerSeriesWithinAt (f i) (p i) s e
    this : Filter.Eventually (fun r => ∀ (i : ι), HasFPowerSeriesWithinOnBall (f i …
    ⊢ HasFPowerSeriesWithinAt (fun x x_1 => f x_1 x) (FormalMultilinearSeries.pi p …
  -/
  obtain ⟨r, hr, r_pos⟩ := (this.and self_mem_nhdsWithin).exists
  /-
    case intro.intro
    𝕜 : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    ι : Type u_9
    inst✝² : Fintype ι
    e : E
    Fm : ι → Type u_10
    inst✝¹ : (i : ι) → NormedAddCommGroup (Fm i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (Fm i)
    f : (i : ι) → E → Fm i
    s : Set E
    p : (i : ι) → FormalMultilinearSeries 𝕜 E (Fm i)
    hf : ∀ (i : ι), HasFPowerSeriesWithinAt (f i) (p i) s e
    this : Filter.Eventually (fun r => ∀ (i : ι), HasFPowerSeriesWithinOnBall (f i …
    r : ENNReal
    hr : ∀ (i : ι), HasFPowerSeriesWithinOnBall (f i) (p i) s e r
    r_pos : LT.lt 0 r
    ⊢ HasFPowerSeriesWithinAt (fun x x_1 => f x_1 x) (FormalMultilinearSeries.pi p …
  -/
  exact ⟨r, HasFPowerSeriesWithinOnBall.pi hr r_pos⟩
  /-
    🎉 no goals
  -/


lemma hasFPowerSeriesWithinAt_pi_iff :
    HasFPowerSeriesWithinAt (fun x ↦ (f · x)) (FormalMultilinearSeries.pi p) s e ↔
      ∀ i, HasFPowerSeriesWithinAt (f i) (p i) s e := by
  /-
    𝕜 : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    ι : Type u_9
    inst✝² : Fintype ι
    e : E
    Fm : ι → Type u_10
    inst✝¹ : (i : ι) → NormedAddCommGroup (Fm i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (Fm i)
    f : (i : ι) → E → Fm i
    s : Set E
    p : (i : ι) → FormalMultilinearSeries 𝕜 E (Fm i)
    ⊢ Iff (HasFPowerSeriesWithinAt (fun x x_1 => f x_1 x) (FormalMultilinearSeries …
  -/
  refine ⟨fun h i ↦ ?_, fun h ↦ .pi h⟩
  /-
    𝕜 : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    ι : Type u_9
    inst✝² : Fintype ι
    e : E
    Fm : ι → Type u_10
    inst✝¹ : (i : ι) → NormedAddCommGroup (Fm i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (Fm i)
    f : (i : ι) → E → Fm i
    s : Set E
    p : (i : ι) → FormalMultilinearSeries 𝕜 E (Fm i)
    h : HasFPowerSeriesWithinAt (fun x x_1 => f x_1 x) (FormalMultilinearSeries.pi …
    i : ι
    ⊢ HasFPowerSeriesWithinAt (f i) (p i) s e
  -/
  obtain ⟨r, hr⟩ := h
  /-
    case intro
    𝕜 : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    ι : Type u_9
    inst✝² : Fintype ι
    e : E
    Fm : ι → Type u_10
    inst✝¹ : (i : ι) → NormedAddCommGroup (Fm i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (Fm i)
    f : (i : ι) → E → Fm i
    s : Set E
    p : (i : ι) → FormalMultilinearSeries 𝕜 E (Fm i)
    i : ι
    r : ENNReal
    hr : HasFPowerSeriesWithinOnBall (fun x x_1 => f x_1 x) (FormalMultilinearSeri …
    ⊢ HasFPowerSeriesWithinAt (f i) (p i) s e
  -/
  exact ⟨r, (hasFPowerSeriesWithinOnBall_pi_iff hr.r_pos).1 hr i⟩
  /-
    🎉 no goals
  -/


lemma HasFPowerSeriesAt.pi
    (hf : ∀ i, HasFPowerSeriesAt (f i) (p i) e) :
    HasFPowerSeriesAt (fun x ↦ (f · x)) (FormalMultilinearSeries.pi p) e := by
  /-
    𝕜 : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    ι : Type u_9
    inst✝² : Fintype ι
    e : E
    Fm : ι → Type u_10
    inst✝¹ : (i : ι) → NormedAddCommGroup (Fm i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (Fm i)
    f : (i : ι) → E → Fm i
    p : (i : ι) → FormalMultilinearSeries 𝕜 E (Fm i)
    hf : ∀ (i : ι), HasFPowerSeriesAt (f i) (p i) e
    ⊢ HasFPowerSeriesAt (fun x x_1 => f x_1 x) (FormalMultilinearSeries.pi p) e
  -/
  simp_rw [← hasFPowerSeriesWithinAt_univ] at hf ⊢
  /-
    𝕜 : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    ι : Type u_9
    inst✝² : Fintype ι
    e : E
    Fm : ι → Type u_10
    inst✝¹ : (i : ι) → NormedAddCommGroup (Fm i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (Fm i)
    f : (i : ι) → E → Fm i
    p : (i : ι) → FormalMultilinearSeries 𝕜 E (Fm i)
    hf : ∀ (i : ι), HasFPowerSeriesWithinAt (f i) (p i) Set.univ e
    ⊢ HasFPowerSeriesWithinAt (fun x x_1 => f x_1 x) (FormalMultilinearSeries.pi p …
  -/
  exact HasFPowerSeriesWithinAt.pi hf
  /-
    🎉 no goals
  -/


lemma hasFPowerSeriesAt_pi_iff :
    HasFPowerSeriesAt (fun x ↦ (f · x)) (FormalMultilinearSeries.pi p) e ↔
      ∀ i, HasFPowerSeriesAt (f i) (p i) e := by
  /-
    𝕜 : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    ι : Type u_9
    inst✝² : Fintype ι
    e : E
    Fm : ι → Type u_10
    inst✝¹ : (i : ι) → NormedAddCommGroup (Fm i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (Fm i)
    f : (i : ι) → E → Fm i
    p : (i : ι) → FormalMultilinearSeries 𝕜 E (Fm i)
    ⊢ Iff (HasFPowerSeriesAt (fun x x_1 => f x_1 x) (FormalMultilinearSeries.pi p) …
  -/
  simp_rw [← hasFPowerSeriesWithinAt_univ]
  /-
    𝕜 : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    ι : Type u_9
    inst✝² : Fintype ι
    e : E
    Fm : ι → Type u_10
    inst✝¹ : (i : ι) → NormedAddCommGroup (Fm i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (Fm i)
    f : (i : ι) → E → Fm i
    p : (i : ι) → FormalMultilinearSeries 𝕜 E (Fm i)
    ⊢ Iff (HasFPowerSeriesWithinAt (fun x x_1 => f x_1 x) (FormalMultilinearSeries …
  -/
  exact hasFPowerSeriesWithinAt_pi_iff
  /-
    🎉 no goals
  -/


lemma AnalyticWithinAt.pi (hf : ∀ i, AnalyticWithinAt 𝕜 (f i) s e) :
    AnalyticWithinAt 𝕜 (fun x ↦ (f · x)) s e := by
  /-
    𝕜 : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    ι : Type u_9
    inst✝² : Fintype ι
    e : E
    Fm : ι → Type u_10
    inst✝¹ : (i : ι) → NormedAddCommGroup (Fm i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (Fm i)
    f : (i : ι) → E → Fm i
    s : Set E
    hf : ∀ (i : ι), AnalyticWithinAt 𝕜 (f i) s e
    ⊢ AnalyticWithinAt 𝕜 (fun x x_1 => f x_1 x) s e
  -/
  choose p hp using hf
  /-
    𝕜 : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    ι : Type u_9
    inst✝² : Fintype ι
    e : E
    Fm : ι → Type u_10
    inst✝¹ : (i : ι) → NormedAddCommGroup (Fm i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (Fm i)
    f : (i : ι) → E → Fm i
    s : Set E
    p : (i : ι) → FormalMultilinearSeries 𝕜 E (Fm i)
    hp : ∀ (i : ι), HasFPowerSeriesWithinAt (f i) (p i) s e
    ⊢ AnalyticWithinAt 𝕜 (fun x x_1 => f x_1 x) s e
  -/
  exact ⟨FormalMultilinearSeries.pi p, HasFPowerSeriesWithinAt.pi hp⟩
  /-
    🎉 no goals
  -/


lemma analyticWithinAt_pi_iff :
    AnalyticWithinAt 𝕜 (fun x ↦ (f · x)) s e ↔ ∀ i, AnalyticWithinAt 𝕜 (f i) s e := by
  /-
    𝕜 : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    ι : Type u_9
    inst✝² : Fintype ι
    e : E
    Fm : ι → Type u_10
    inst✝¹ : (i : ι) → NormedAddCommGroup (Fm i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (Fm i)
    f : (i : ι) → E → Fm i
    s : Set E
    ⊢ Iff (AnalyticWithinAt 𝕜 (fun x x_1 => f x_1 x) s e) (∀ (i : ι), AnalyticWith …
  -/
  refine ⟨fun h i ↦ ?_, fun h ↦ .pi h⟩
  /-
    𝕜 : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    ι : Type u_9
    inst✝² : Fintype ι
    e : E
    Fm : ι → Type u_10
    inst✝¹ : (i : ι) → NormedAddCommGroup (Fm i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (Fm i)
    f : (i : ι) → E → Fm i
    s : Set E
    h : AnalyticWithinAt 𝕜 (fun x x_1 => f x_1 x) s e
    i : ι
    ⊢ AnalyticWithinAt 𝕜 (f i) s e
  -/
  exact ((ContinuousLinearMap.proj (R := 𝕜) i).analyticAt _).comp_analyticWithinAt h
  /-
    🎉 no goals
  -/


lemma AnalyticAt.pi (hf : ∀ i, AnalyticAt 𝕜 (f i) e) :
    AnalyticAt 𝕜 (fun x ↦ (f · x)) e := by
  /-
    𝕜 : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    ι : Type u_9
    inst✝² : Fintype ι
    e : E
    Fm : ι → Type u_10
    inst✝¹ : (i : ι) → NormedAddCommGroup (Fm i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (Fm i)
    f : (i : ι) → E → Fm i
    hf : ∀ (i : ι), AnalyticAt 𝕜 (f i) e
    ⊢ AnalyticAt 𝕜 (fun x x_1 => f x_1 x) e
  -/
  simp_rw [← analyticWithinAt_univ] at hf ⊢
  /-
    𝕜 : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    ι : Type u_9
    inst✝² : Fintype ι
    e : E
    Fm : ι → Type u_10
    inst✝¹ : (i : ι) → NormedAddCommGroup (Fm i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (Fm i)
    f : (i : ι) → E → Fm i
    hf : ∀ (i : ι), AnalyticWithinAt 𝕜 (f i) Set.univ e
    ⊢ AnalyticWithinAt 𝕜 (fun x x_1 => f x_1 x) Set.univ e
  -/
  exact AnalyticWithinAt.pi hf
  /-
    🎉 no goals
  -/


lemma analyticAt_pi_iff :
    AnalyticAt 𝕜 (fun x ↦ (f · x)) e ↔ ∀ i, AnalyticAt 𝕜 (f i) e := by
  /-
    𝕜 : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    ι : Type u_9
    inst✝² : Fintype ι
    e : E
    Fm : ι → Type u_10
    inst✝¹ : (i : ι) → NormedAddCommGroup (Fm i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (Fm i)
    f : (i : ι) → E → Fm i
    ⊢ Iff (AnalyticAt 𝕜 (fun x x_1 => f x_1 x) e) (∀ (i : ι), AnalyticAt 𝕜 (f i) e)
  -/
  simp_rw [← analyticWithinAt_univ]
  /-
    𝕜 : Type u_2
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    ι : Type u_9
    inst✝² : Fintype ι
    e : E
    Fm : ι → Type u_10
    inst✝¹ : (i : ι) → NormedAddCommGroup (Fm i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (Fm i)
    f : (i : ι) → E → Fm i
    ⊢ Iff (AnalyticWithinAt 𝕜 (fun x x_1 => f x_1 x) Set.univ e) (∀ (i : ι), Analy …
  -/
  exact analyticWithinAt_pi_iff
  /-
    🎉 no goals
  -/


lemma AnalyticOn.pi (hf : ∀ i, AnalyticOn 𝕜 (f i) s) :
    AnalyticOn 𝕜 (fun x ↦ (f · x)) s :=
  fun x hx ↦ AnalyticWithinAt.pi (fun i ↦ hf i x hx)


@[deprecated (since := "2024-09-26")]
alias AnalyticWithinOn.pi := AnalyticOn.pi


lemma analyticOn_pi_iff :
    AnalyticOn 𝕜 (fun x ↦ (f · x)) s ↔ ∀ i, AnalyticOn 𝕜 (f i) s :=
  ⟨fun h i x hx ↦ analyticWithinAt_pi_iff.1 (h x hx) i, fun h ↦ .pi h⟩


@[deprecated (since := "2024-09-26")]
alias analyticWithinOn_pi_iff := analyticOn_pi_iff


lemma AnalyticOnNhd.pi (hf : ∀ i, AnalyticOnNhd 𝕜 (f i) s) :
    AnalyticOnNhd 𝕜 (fun x ↦ (f · x)) s :=
  fun x hx ↦ AnalyticAt.pi (fun i ↦ hf i x hx)


lemma analyticOnNhd_pi_iff :
    AnalyticOnNhd 𝕜 (fun x ↦ (f · x)) s ↔ ∀ i, AnalyticOnNhd 𝕜 (f i) s :=
  ⟨fun h i x hx ↦ analyticAt_pi_iff.1 (h x hx) i, fun h ↦ .pi h⟩


/-- Scalar multiplication is analytic (jointly in both variables). The statement is a little
pedantic to allow towers of field extensions.

TODO: can we replace `𝕜'` with a "normed module" in such a way that `analyticAt_mul` is a special
case of this? -/
lemma analyticAt_smul [NormedSpace 𝕝 E] [IsScalarTower 𝕜 𝕝 E] (z : 𝕝 × E) :
    AnalyticAt 𝕜 (fun x : 𝕝 × E ↦ x.1 • x.2) z :=
  (ContinuousLinearMap.lsmul 𝕜 𝕝).analyticAt_bilinear z


/-- Multiplication in a normed algebra over `𝕜` is analytic. -/
lemma analyticAt_mul (z : A × A) : AnalyticAt 𝕜 (fun x : A × A ↦ x.1 * x.2) z :=
  (ContinuousLinearMap.mul 𝕜 A).analyticAt_bilinear z


/-- Scalar multiplication of one analytic function by another. -/
lemma AnalyticWithinAt.smul [NormedSpace 𝕝 F] [IsScalarTower 𝕜 𝕝 F]
    {f : E → 𝕝} {g : E → F} {s : Set E} {z : E}
    (hf : AnalyticWithinAt 𝕜 f s z) (hg : AnalyticWithinAt 𝕜 g s z) :
    AnalyticWithinAt 𝕜 (fun x ↦ f x • g x) s z :=
  (analyticAt_smul _).comp₂_analyticWithinAt hf hg


/-- Scalar multiplication of one analytic function by another. -/
lemma AnalyticAt.smul [NormedSpace 𝕝 F] [IsScalarTower 𝕜 𝕝 F] {f : E → 𝕝} {g : E → F} {z : E}
    (hf : AnalyticAt 𝕜 f z) (hg : AnalyticAt 𝕜 g z) :
    AnalyticAt 𝕜 (fun x ↦ f x • g x) z :=
  (analyticAt_smul _).comp₂ hf hg


/-- Scalar multiplication of one analytic function by another. -/
lemma AnalyticOn.smul [NormedSpace 𝕝 F] [IsScalarTower 𝕜 𝕝 F]
    {f : E → 𝕝} {g : E → F} {s : Set E}
    (hf : AnalyticOn 𝕜 f s) (hg : AnalyticOn 𝕜 g s) :
    AnalyticOn 𝕜 (fun x ↦ f x • g x) s :=
  fun _ m ↦ (hf _ m).smul (hg _ m)


@[deprecated (since := "2024-09-26")]
alias AnalyticWithinOn.smul := AnalyticOn.smul


/-- Scalar multiplication of one analytic function by another. -/
lemma AnalyticOnNhd.smul [NormedSpace 𝕝 F] [IsScalarTower 𝕜 𝕝 F] {f : E → 𝕝} {g : E → F} {s : Set E}
    (hf : AnalyticOnNhd 𝕜 f s) (hg : AnalyticOnNhd 𝕜 g s) :
    AnalyticOnNhd 𝕜 (fun x ↦ f x • g x) s :=
  fun _ m ↦ (hf _ m).smul (hg _ m)


/-- Multiplication of analytic functions (valued in a normed `𝕜`-algebra) is analytic. -/
lemma AnalyticWithinAt.mul {f g : E → A} {s : Set E} {z : E}
    (hf : AnalyticWithinAt 𝕜 f s z) (hg : AnalyticWithinAt 𝕜 g s z) :
    AnalyticWithinAt 𝕜 (fun x ↦ f x * g x) s z :=
  (analyticAt_mul _).comp₂_analyticWithinAt hf hg


/-- Multiplication of analytic functions (valued in a normed `𝕜`-algebra) is analytic. -/
lemma AnalyticAt.mul {f g : E → A} {z : E} (hf : AnalyticAt 𝕜 f z) (hg : AnalyticAt 𝕜 g z) :
    AnalyticAt 𝕜 (fun x ↦ f x * g x) z :=
  (analyticAt_mul _).comp₂ hf hg


/-- Multiplication of analytic functions (valued in a normed `𝕜`-algebra) is analytic. -/
lemma AnalyticOn.mul {f g : E → A} {s : Set E}
    (hf : AnalyticOn 𝕜 f s) (hg : AnalyticOn 𝕜 g s) :
    AnalyticOn 𝕜 (fun x ↦ f x * g x) s :=
  fun _ m ↦ (hf _ m).mul (hg _ m)


@[deprecated (since := "2024-09-26")]
alias AnalyticWithinOn.mul := AnalyticOn.mul


/-- Multiplication of analytic functions (valued in a normed `𝕜`-algebra) is analytic. -/
lemma AnalyticOnNhd.mul {f g : E → A} {s : Set E}
    (hf : AnalyticOnNhd 𝕜 f s) (hg : AnalyticOnNhd 𝕜 g s) :
    AnalyticOnNhd 𝕜 (fun x ↦ f x * g x) s :=
  fun _ m ↦ (hf _ m).mul (hg _ m)


/-- Powers of analytic functions (into a normed `𝕜`-algebra) are analytic. -/
lemma AnalyticWithinAt.pow {f : E → A} {z : E} {s : Set E} (hf : AnalyticWithinAt 𝕜 f s z) (n : ℕ) :
    AnalyticWithinAt 𝕜 (fun x ↦ f x ^ n) s z := by
  induction n with
  | zero =>
    simp only [pow_zero]
    apply analyticWithinAt_const
  | succ m hm =>
    simp only [pow_succ]
    exact hm.mul hf


/-- Powers of analytic functions (into a normed `𝕜`-algebra) are analytic. -/
lemma AnalyticAt.pow {f : E → A} {z : E} (hf : AnalyticAt 𝕜 f z) (n : ℕ) :
    AnalyticAt 𝕜 (fun x ↦ f x ^ n) z := by
  /-
    𝕜 : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    A : Type u_8
    inst✝¹ : NormedRing A
    inst✝ : NormedAlgebra 𝕜 A
    f : E → A
    z : E
    hf : AnalyticAt 𝕜 f z
    n : Nat
    ⊢ AnalyticAt 𝕜 (fun x => HPow.hPow (f x) n) z
  -/
  rw [← analyticWithinAt_univ] at hf ⊢
  /-
    𝕜 : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    A : Type u_8
    inst✝¹ : NormedRing A
    inst✝ : NormedAlgebra 𝕜 A
    f : E → A
    z : E
    hf : AnalyticWithinAt 𝕜 f Set.univ z
    n : Nat
    ⊢ AnalyticWithinAt 𝕜 (fun x => HPow.hPow (f x) n) Set.univ z
  -/
  exact hf.pow n
  /-
    🎉 no goals
  -/


/-- Powers of analytic functions (into a normed `𝕜`-algebra) are analytic. -/
lemma AnalyticOn.pow {f : E → A} {s : Set E} (hf : AnalyticOn 𝕜 f s) (n : ℕ) :
    AnalyticOn 𝕜 (fun x ↦ f x ^ n) s :=
  fun _ m ↦ (hf _ m).pow n


@[deprecated (since := "2024-09-26")]
alias AnalyticWithinOn.pow := AnalyticOn.pow


/-- Powers of analytic functions (into a normed `𝕜`-algebra) are analytic. -/
lemma AnalyticOnNhd.pow {f : E → A} {s : Set E} (hf : AnalyticOnNhd 𝕜 f s) (n : ℕ) :
    AnalyticOnNhd 𝕜 (fun x ↦ f x ^ n) s :=
  fun _ m ↦ (hf _ m).pow n



lemma HasFPowerSeriesWithinOnBall.restrictScalars (hf : HasFPowerSeriesWithinOnBall f p s x r) :
    HasFPowerSeriesWithinOnBall f (p.restrictScalars 𝕜) s x r :=
                                                                      /-
                                                                        𝕜 : Type u_2
                                                                        inst✝¹⁰ : NontriviallyNormedField 𝕜
                                                                        E : Type u_3
                                                                        F : Type u_4
                                                                        inst✝⁹ : NormedAddCommGroup E
                                                                        inst✝⁸ : NormedSpace 𝕜 E
                                                                        inst✝⁷ : NormedAddCommGroup F
                                                                        inst✝⁶ : NormedSpace 𝕜 F
                                                                        𝕜' : Type u_9
                                                                        inst✝⁵ : NontriviallyNormedField 𝕜'
                                                                        inst✝⁴ : NormedAlgebra 𝕜 𝕜'
                                                                        inst✝³ : NormedSpace 𝕜' E
                                                                        inst✝² : IsScalarTower 𝕜 𝕜' E
                                                                        inst✝¹ : NormedSpace 𝕜' F
                                                                        inst✝ : IsScalarTower 𝕜 𝕜' F
                                                                        f : E → F
                                                                        p : FormalMultilinearSeries 𝕜' E F
                                                                        x : E
                                                                        s : Set E
                                                                        r : ENNReal
                                                                        hf : HasFPowerSeriesWithinOnBall f p s x r
                                                                        n : Nat
                                                                        ⊢ LE.le (Norm.norm (FormalMultilinearSeries.restrictScalars 𝕜 p n)) (Norm.norm …
                                                                      -/
  ⟨hf.r_le.trans (FormalMultilinearSeries.radius_le_of_le (fun n ↦ by simp)), hf.r_pos, hf.hasSum⟩
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


lemma HasFPowerSeriesOnBall.restrictScalars (hf : HasFPowerSeriesOnBall f p x r) :
    HasFPowerSeriesOnBall f (p.restrictScalars 𝕜) x r :=
                                                                      /-
                                                                        𝕜 : Type u_2
                                                                        inst✝¹⁰ : NontriviallyNormedField 𝕜
                                                                        E : Type u_3
                                                                        F : Type u_4
                                                                        inst✝⁹ : NormedAddCommGroup E
                                                                        inst✝⁸ : NormedSpace 𝕜 E
                                                                        inst✝⁷ : NormedAddCommGroup F
                                                                        inst✝⁶ : NormedSpace 𝕜 F
                                                                        𝕜' : Type u_9
                                                                        inst✝⁵ : NontriviallyNormedField 𝕜'
                                                                        inst✝⁴ : NormedAlgebra 𝕜 𝕜'
                                                                        inst✝³ : NormedSpace 𝕜' E
                                                                        inst✝² : IsScalarTower 𝕜 𝕜' E
                                                                        inst✝¹ : NormedSpace 𝕜' F
                                                                        inst✝ : IsScalarTower 𝕜 𝕜' F
                                                                        f : E → F
                                                                        p : FormalMultilinearSeries 𝕜' E F
                                                                        x : E
                                                                        r : ENNReal
                                                                        hf : HasFPowerSeriesOnBall f p x r
                                                                        n : Nat
                                                                        ⊢ LE.le (Norm.norm (FormalMultilinearSeries.restrictScalars 𝕜 p n)) (Norm.norm …
                                                                      -/
  ⟨hf.r_le.trans (FormalMultilinearSeries.radius_le_of_le (fun n ↦ by simp)), hf.r_pos, hf.hasSum⟩
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


lemma HasFPowerSeriesWithinAt.restrictScalars (hf : HasFPowerSeriesWithinAt f p s x) :
    HasFPowerSeriesWithinAt f (p.restrictScalars 𝕜) s x := by
  /-
    𝕜 : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_3
    F : Type u_4
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    𝕜' : Type u_9
    inst✝⁵ : NontriviallyNormedField 𝕜'
    inst✝⁴ : NormedAlgebra 𝕜 𝕜'
    inst✝³ : NormedSpace 𝕜' E
    inst✝² : IsScalarTower 𝕜 𝕜' E
    inst✝¹ : NormedSpace 𝕜' F
    inst✝ : IsScalarTower 𝕜 𝕜' F
    f : E → F
    p : FormalMultilinearSeries 𝕜' E F
    x : E
    s : Set E
    hf : HasFPowerSeriesWithinAt f p s x
    ⊢ HasFPowerSeriesWithinAt f (FormalMultilinearSeries.restrictScalars 𝕜 p) s x
  -/
  rcases hf with ⟨r, hr⟩
  /-
    case intro
    𝕜 : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_3
    F : Type u_4
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    𝕜' : Type u_9
    inst✝⁵ : NontriviallyNormedField 𝕜'
    inst✝⁴ : NormedAlgebra 𝕜 𝕜'
    inst✝³ : NormedSpace 𝕜' E
    inst✝² : IsScalarTower 𝕜 𝕜' E
    inst✝¹ : NormedSpace 𝕜' F
    inst✝ : IsScalarTower 𝕜 𝕜' F
    f : E → F
    p : FormalMultilinearSeries 𝕜' E F
    x : E
    s : Set E
    r : ENNReal
    hr : HasFPowerSeriesWithinOnBall f p s x r
    ⊢ HasFPowerSeriesWithinAt f (FormalMultilinearSeries.restrictScalars 𝕜 p) s x
  -/
  exact ⟨r, hr.restrictScalars⟩
  /-
    🎉 no goals
  -/


lemma HasFPowerSeriesAt.restrictScalars (hf : HasFPowerSeriesAt f p x) :
    HasFPowerSeriesAt f (p.restrictScalars 𝕜) x := by
  /-
    𝕜 : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_3
    F : Type u_4
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    𝕜' : Type u_9
    inst✝⁵ : NontriviallyNormedField 𝕜'
    inst✝⁴ : NormedAlgebra 𝕜 𝕜'
    inst✝³ : NormedSpace 𝕜' E
    inst✝² : IsScalarTower 𝕜 𝕜' E
    inst✝¹ : NormedSpace 𝕜' F
    inst✝ : IsScalarTower 𝕜 𝕜' F
    f : E → F
    p : FormalMultilinearSeries 𝕜' E F
    x : E
    hf : HasFPowerSeriesAt f p x
    ⊢ HasFPowerSeriesAt f (FormalMultilinearSeries.restrictScalars 𝕜 p) x
  -/
  rcases hf with ⟨r, hr⟩
  /-
    case intro
    𝕜 : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_3
    F : Type u_4
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    𝕜' : Type u_9
    inst✝⁵ : NontriviallyNormedField 𝕜'
    inst✝⁴ : NormedAlgebra 𝕜 𝕜'
    inst✝³ : NormedSpace 𝕜' E
    inst✝² : IsScalarTower 𝕜 𝕜' E
    inst✝¹ : NormedSpace 𝕜' F
    inst✝ : IsScalarTower 𝕜 𝕜' F
    f : E → F
    p : FormalMultilinearSeries 𝕜' E F
    x : E
    r : ENNReal
    hr : HasFPowerSeriesOnBall f p x r
    ⊢ HasFPowerSeriesAt f (FormalMultilinearSeries.restrictScalars 𝕜 p) x
  -/
  exact ⟨r, hr.restrictScalars⟩
  /-
    🎉 no goals
  -/


lemma AnalyticWithinAt.restrictScalars (hf : AnalyticWithinAt 𝕜' f s x) :
    AnalyticWithinAt 𝕜 f s x := by
  /-
    𝕜 : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_3
    F : Type u_4
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    𝕜' : Type u_9
    inst✝⁵ : NontriviallyNormedField 𝕜'
    inst✝⁴ : NormedAlgebra 𝕜 𝕜'
    inst✝³ : NormedSpace 𝕜' E
    inst✝² : IsScalarTower 𝕜 𝕜' E
    inst✝¹ : NormedSpace 𝕜' F
    inst✝ : IsScalarTower 𝕜 𝕜' F
    f : E → F
    x : E
    s : Set E
    hf : AnalyticWithinAt 𝕜' f s x
    ⊢ AnalyticWithinAt 𝕜 f s x
  -/
  rcases hf with ⟨p, hp⟩
  /-
    case intro
    𝕜 : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_3
    F : Type u_4
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    𝕜' : Type u_9
    inst✝⁵ : NontriviallyNormedField 𝕜'
    inst✝⁴ : NormedAlgebra 𝕜 𝕜'
    inst✝³ : NormedSpace 𝕜' E
    inst✝² : IsScalarTower 𝕜 𝕜' E
    inst✝¹ : NormedSpace 𝕜' F
    inst✝ : IsScalarTower 𝕜 𝕜' F
    f : E → F
    x : E
    s : Set E
    p : FormalMultilinearSeries 𝕜' E F
    hp : HasFPowerSeriesWithinAt f p s x
    ⊢ AnalyticWithinAt 𝕜 f s x
  -/
  exact ⟨p.restrictScalars 𝕜, hp.restrictScalars⟩
  /-
    🎉 no goals
  -/


lemma AnalyticAt.restrictScalars (hf : AnalyticAt 𝕜' f x) :
    AnalyticAt 𝕜 f x := by
  /-
    𝕜 : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_3
    F : Type u_4
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    𝕜' : Type u_9
    inst✝⁵ : NontriviallyNormedField 𝕜'
    inst✝⁴ : NormedAlgebra 𝕜 𝕜'
    inst✝³ : NormedSpace 𝕜' E
    inst✝² : IsScalarTower 𝕜 𝕜' E
    inst✝¹ : NormedSpace 𝕜' F
    inst✝ : IsScalarTower 𝕜 𝕜' F
    f : E → F
    x : E
    hf : AnalyticAt 𝕜' f x
    ⊢ AnalyticAt 𝕜 f x
  -/
  rcases hf with ⟨p, hp⟩
  /-
    case intro
    𝕜 : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_3
    F : Type u_4
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    𝕜' : Type u_9
    inst✝⁵ : NontriviallyNormedField 𝕜'
    inst✝⁴ : NormedAlgebra 𝕜 𝕜'
    inst✝³ : NormedSpace 𝕜' E
    inst✝² : IsScalarTower 𝕜 𝕜' E
    inst✝¹ : NormedSpace 𝕜' F
    inst✝ : IsScalarTower 𝕜 𝕜' F
    f : E → F
    x : E
    p : FormalMultilinearSeries 𝕜' E F
    hp : HasFPowerSeriesAt f p x
    ⊢ AnalyticAt 𝕜 f x
  -/
  exact ⟨p.restrictScalars 𝕜, hp.restrictScalars⟩
  /-
    🎉 no goals
  -/


lemma AnalyticOn.restrictScalars (hf : AnalyticOn 𝕜' f s) :
    AnalyticOn 𝕜 f s :=
  fun x hx ↦ (hf x hx).restrictScalars


lemma AnalyticOnNhd.restrictScalars (hf : AnalyticOnNhd 𝕜' f s) :
    AnalyticOnNhd 𝕜 f s :=
  fun x hx ↦ (hf x hx).restrictScalars


/-- The geometric series `1 + x + x ^ 2 + ...` as a `FormalMultilinearSeries`. -/
def formalMultilinearSeries_geometric : FormalMultilinearSeries 𝕜 A A :=
  fun n ↦ ContinuousMultilinearMap.mkPiAlgebraFin 𝕜 n A


/-- The geometric series as an `ofScalars` series. -/
theorem formalMultilinearSeries_geometric_eq_ofScalars :
    formalMultilinearSeries_geometric 𝕜 A =
      FormalMultilinearSeries.ofScalars A fun _ ↦ (1 : 𝕜) := by
  simp_rw [FormalMultilinearSeries.ext_iff, FormalMultilinearSeries.ofScalars,
    formalMultilinearSeries_geometric, one_smul, implies_true]


lemma formalMultilinearSeries_geometric_apply_norm_le (n : ℕ) :
    ‖formalMultilinearSeries_geometric 𝕜 A n‖ ≤ max 1 ‖(1 : A)‖ :=
  ContinuousMultilinearMap.norm_mkPiAlgebraFin_le


lemma formalMultilinearSeries_geometric_apply_norm [NormOneClass A] (n : ℕ) :
    ‖formalMultilinearSeries_geometric 𝕜 A n‖ = 1 :=
  ContinuousMultilinearMap.norm_mkPiAlgebraFin


lemma one_le_formalMultilinearSeries_geometric_radius (𝕜 : Type*) [NontriviallyNormedField 𝕜]
    (A : Type*) [NormedRing A] [NormedAlgebra 𝕜 A] :
    1 ≤ (formalMultilinearSeries_geometric 𝕜 A).radius := by
  convert formalMultilinearSeries_geometric_eq_ofScalars 𝕜 A ▸
    FormalMultilinearSeries.ofScalars_radius_ge_inv_of_tendsto A _ one_ne_zero (by simp) |>.le
  /-
    case h.e'_3
    𝕜 : Type u_9
    inst✝² : NontriviallyNormedField 𝕜
    A : Type u_10
    inst✝¹ : NormedRing A
    inst✝ : NormedAlgebra 𝕜 A
    ⊢ Eq 1 ↑(Inv.inv 1)
  -/
  simp
  /-
    🎉 no goals
  -/


lemma formalMultilinearSeries_geometric_radius (𝕜 : Type*) [NontriviallyNormedField 𝕜]
    (A : Type*) [NormedRing A] [NormOneClass A] [NormedAlgebra 𝕜 A] :
    (formalMultilinearSeries_geometric 𝕜 A).radius = 1 :=
  formalMultilinearSeries_geometric_eq_ofScalars 𝕜 A ▸
                                                                               /-
                                                                                 𝕜 : Type u_9
                                                                                 inst✝³ : NontriviallyNormedField 𝕜
                                                                                 A : Type u_10
                                                                                 inst✝² : NormedRing A
                                                                                 inst✝¹ : NormOneClass A
                                                                                 inst✝ : NormedAlgebra 𝕜 A
                                                                                 ⊢ Filter.Tendsto (fun n => HDiv.hDiv (Norm.norm 1) (Norm.norm 1)) Filter.atTop …
                                                                               -/
    FormalMultilinearSeries.ofScalars_radius_eq_of_tendsto A _ one_ne_zero (by simp)
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


lemma hasFPowerSeriesOnBall_inverse_one_sub
    (𝕜 : Type*) [NontriviallyNormedField 𝕜]
    (A : Type*) [NormedRing A] [NormedAlgebra 𝕜 A] [HasSummableGeomSeries A] :
    HasFPowerSeriesOnBall (fun x : A ↦ Ring.inverse (1 - x))
      (formalMultilinearSeries_geometric 𝕜 A) 0 1 := by
  /-
    𝕜 : Type u_9
    inst✝³ : NontriviallyNormedField 𝕜
    A : Type u_10
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : HasSummableGeomSeries A
    ⊢ HasFPowerSeriesOnBall (fun x => Ring.inverse (HSub.hSub 1 x)) (formalMultili …
  -/
  constructor
    /-
      case r_le
      𝕜 : Type u_9
      inst✝³ : NontriviallyNormedField 𝕜
      A : Type u_10
      inst✝² : NormedRing A
      inst✝¹ : NormedAlgebra 𝕜 A
      inst✝ : HasSummableGeomSeries A
      ⊢ LE.le 1 (formalMultilinearSeries_geometric 𝕜 A).radius
    -/
  · exact one_le_formalMultilinearSeries_geometric_radius 𝕜 A
    /-
      🎉 no goals
    -/
    /-
      case r_pos
      𝕜 : Type u_9
      inst✝³ : NontriviallyNormedField 𝕜
      A : Type u_10
      inst✝² : NormedRing A
      inst✝¹ : NormedAlgebra 𝕜 A
      inst✝ : HasSummableGeomSeries A
      ⊢ LT.lt 0 1
    -/
  · exact one_pos
    /-
      🎉 no goals
    -/
    /-
      case hasSum
      𝕜 : Type u_9
      inst✝³ : NontriviallyNormedField 𝕜
      A : Type u_10
      inst✝² : NormedRing A
      inst✝¹ : NormedAlgebra 𝕜 A
      inst✝ : HasSummableGeomSeries A
      ⊢ ∀ {y : A}, Membership.mem (EMetric.ball 0 1) y → HasSum (fun n => (formalMul …
    -/
  · intro y hy
    /-
      case hasSum
      𝕜 : Type u_9
      inst✝³ : NontriviallyNormedField 𝕜
      A : Type u_10
      inst✝² : NormedRing A
      inst✝¹ : NormedAlgebra 𝕜 A
      inst✝ : HasSummableGeomSeries A
      y : A
      hy : Membership.mem (EMetric.ball 0 1) y
      ⊢ HasSum (fun n => (formalMultilinearSeries_geometric 𝕜 A n) fun x => y) (Ring …
    -/
    simp only [EMetric.mem_ball, edist_dist, dist_zero_right, ofReal_lt_one] at hy
    simp only [zero_add, NormedRing.inverse_one_sub _ hy, Units.oneSub, Units.inv_mk,
      formalMultilinearSeries_geometric, ContinuousMultilinearMap.mkPiAlgebraFin_apply,
      List.ofFn_const, List.prod_replicate]
    /-
      case hasSum
      𝕜 : Type u_9
      inst✝³ : NontriviallyNormedField 𝕜
      A : Type u_10
      inst✝² : NormedRing A
      inst✝¹ : NormedAlgebra 𝕜 A
      inst✝ : HasSummableGeomSeries A
      y : A
      hy : LT.lt (Norm.norm y) 1
      ⊢ HasSum (fun n => HPow.hPow y n) (tsum fun n => HPow.hPow y n)
    -/
    exact (summable_geometric_of_norm_lt_one hy).hasSum
    /-
      🎉 no goals
    -/


lemma analyticAt_inverse_one_sub (𝕜 : Type*) [NontriviallyNormedField 𝕜]
    (A : Type*) [NormedRing A] [NormedAlgebra 𝕜 A] [HasSummableGeomSeries A] :
    AnalyticAt 𝕜 (fun x : A ↦ Ring.inverse (1 - x)) 0 :=
  ⟨_, ⟨_, hasFPowerSeriesOnBall_inverse_one_sub 𝕜 A⟩⟩


/-- If `A` is a normed algebra over `𝕜` with summable geometric series, then inversion on `A` is
analytic at any unit. -/
lemma analyticAt_inverse {𝕜 : Type*} [NontriviallyNormedField 𝕜]
    {A : Type*} [NormedRing A] [NormedAlgebra 𝕜 A] [HasSummableGeomSeries A] (z : Aˣ) :
    AnalyticAt 𝕜 Ring.inverse (z : A) := by
  /-
    𝕜 : Type u_9
    inst✝³ : NontriviallyNormedField 𝕜
    A : Type u_10
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : HasSummableGeomSeries A
    z : Units A
    ⊢ AnalyticAt 𝕜 Ring.inverse ↑z
  -/
  rcases subsingleton_or_nontrivial A with hA|hA
    /-
      case inl
      𝕜 : Type u_9
      inst✝³ : NontriviallyNormedField 𝕜
      A : Type u_10
      inst✝² : NormedRing A
      inst✝¹ : NormedAlgebra 𝕜 A
      inst✝ : HasSummableGeomSeries A
      z : Units A
      hA : Subsingleton A
      ⊢ AnalyticAt 𝕜 Ring.inverse ↑z
    -/
  · convert analyticAt_const (v := (0 : A))
    /-
      🎉 no goals
    -/
    /-
      case inr
      𝕜 : Type u_9
      inst✝³ : NontriviallyNormedField 𝕜
      A : Type u_10
      inst✝² : NormedRing A
      inst✝¹ : NormedAlgebra 𝕜 A
      inst✝ : HasSummableGeomSeries A
      z : Units A
      hA : Nontrivial A
      ⊢ AnalyticAt 𝕜 Ring.inverse ↑z
    -/
  · let f1 : A → A := fun a ↦ a * z.inv
    /-
      case inr
      𝕜 : Type u_9
      inst✝³ : NontriviallyNormedField 𝕜
      A : Type u_10
      inst✝² : NormedRing A
      inst✝¹ : NormedAlgebra 𝕜 A
      inst✝ : HasSummableGeomSeries A
      z : Units A
      hA : Nontrivial A
      f1 : A → A := fun a => HMul.hMul a z.inv
      ⊢ AnalyticAt 𝕜 Ring.inverse ↑z
    -/
    let f2 : A → A := fun b ↦ Ring.inverse (1 - b)
    /-
      case inr
      𝕜 : Type u_9
      inst✝³ : NontriviallyNormedField 𝕜
      A : Type u_10
      inst✝² : NormedRing A
      inst✝¹ : NormedAlgebra 𝕜 A
      inst✝ : HasSummableGeomSeries A
      z : Units A
      hA : Nontrivial A
      f1 : A → A := fun a => HMul.hMul a z.inv
      f2 : A → A := fun b => Ring.inverse (HSub.hSub 1 b)
      ⊢ AnalyticAt 𝕜 Ring.inverse ↑z
    -/
    let f3 : A → A := fun c ↦ 1 - z.inv * c
    have feq : ∀ᶠ y in 𝓝 (z : A), (f1 ∘ f2 ∘ f3) y = Ring.inverse y := by
      have : Metric.ball (z : A) (‖(↑z⁻¹ : A)‖⁻¹) ∈ 𝓝 (z : A) := by
        apply Metric.ball_mem_nhds
        simp
      filter_upwards [this] with y hy
      simp only [Metric.mem_ball, dist_eq_norm] at hy
      have : y = Units.ofNearby z y hy := rfl
      rw [this, Eq.comm]
      simp only [Ring.inverse_unit, Function.comp_apply]
      simp [Units.ofNearby, f1, f2, f3, Units.add, _root_.mul_sub]
      rw [← Ring.inverse_unit]
      congr
      simp
    /-
      case inr
      𝕜 : Type u_9
      inst✝³ : NontriviallyNormedField 𝕜
      A : Type u_10
      inst✝² : NormedRing A
      inst✝¹ : NormedAlgebra 𝕜 A
      inst✝ : HasSummableGeomSeries A
      z : Units A
      hA : Nontrivial A
      f1 : A → A := fun a => HMul.hMul a z.inv
      f2 : A → A := fun b => Ring.inverse (HSub.hSub 1 b)
      f3 : A → A := fun c => HSub.hSub 1 (HMul.hMul z.inv c)
      feq : Filter.Eventually (fun y => Eq (Function.comp f1 (Function.comp f2 f3) y …
      ⊢ AnalyticAt 𝕜 Ring.inverse ↑z
    -/
    apply AnalyticAt.congr _ feq
    /-
      𝕜 : Type u_9
      inst✝³ : NontriviallyNormedField 𝕜
      A : Type u_10
      inst✝² : NormedRing A
      inst✝¹ : NormedAlgebra 𝕜 A
      inst✝ : HasSummableGeomSeries A
      z : Units A
      hA : Nontrivial A
      f1 : A → A := fun a => HMul.hMul a z.inv
      f2 : A → A := fun b => Ring.inverse (HSub.hSub 1 b)
      f3 : A → A := fun c => HSub.hSub 1 (HMul.hMul z.inv c)
      feq : Filter.Eventually (fun y => Eq (Function.comp f1 (Function.comp f2 f3) y …
      ⊢ AnalyticAt 𝕜 (Function.comp f1 (Function.comp f2 f3)) ↑z
    -/
    apply (analyticAt_id.mul analyticAt_const).comp
    /-
      𝕜 : Type u_9
      inst✝³ : NontriviallyNormedField 𝕜
      A : Type u_10
      inst✝² : NormedRing A
      inst✝¹ : NormedAlgebra 𝕜 A
      inst✝ : HasSummableGeomSeries A
      z : Units A
      hA : Nontrivial A
      f1 : A → A := fun a => HMul.hMul a z.inv
      f2 : A → A := fun b => Ring.inverse (HSub.hSub 1 b)
      f3 : A → A := fun c => HSub.hSub 1 (HMul.hMul z.inv c)
      feq : Filter.Eventually (fun y => Eq (Function.comp f1 (Function.comp f2 f3) y …
      ⊢ AnalyticAt 𝕜 (Function.comp f2 f3) ↑z
    -/
    apply AnalyticAt.comp
      /-
        case hg
        𝕜 : Type u_9
        inst✝³ : NontriviallyNormedField 𝕜
        A : Type u_10
        inst✝² : NormedRing A
        inst✝¹ : NormedAlgebra 𝕜 A
        inst✝ : HasSummableGeomSeries A
        z : Units A
        hA : Nontrivial A
        f1 : A → A := fun a => HMul.hMul a z.inv
        f2 : A → A := fun b => Ring.inverse (HSub.hSub 1 b)
        f3 : A → A := fun c => HSub.hSub 1 (HMul.hMul z.inv c)
        feq : Filter.Eventually (fun y => Eq (Function.comp f1 (Function.comp f2 f3) y …
        ⊢ AnalyticAt 𝕜 f2 (f3 ↑z)
      -/
    · simp only [Units.inv_eq_val_inv, Units.inv_mul, sub_self, f2, f3]
      /-
        case hg
        𝕜 : Type u_9
        inst✝³ : NontriviallyNormedField 𝕜
        A : Type u_10
        inst✝² : NormedRing A
        inst✝¹ : NormedAlgebra 𝕜 A
        inst✝ : HasSummableGeomSeries A
        z : Units A
        hA : Nontrivial A
        f1 : A → A := fun a => HMul.hMul a z.inv
        f2 : A → A := fun b => Ring.inverse (HSub.hSub 1 b)
        f3 : A → A := fun c => HSub.hSub 1 (HMul.hMul z.inv c)
        feq : Filter.Eventually (fun y => Eq (Function.comp f1 (Function.comp f2 f3) y …
        ⊢ AnalyticAt 𝕜 (fun b => Ring.inverse (HSub.hSub 1 b)) 0
      -/
      exact analyticAt_inverse_one_sub 𝕜 A
      /-
        🎉 no goals
      -/
      /-
        case hf
        𝕜 : Type u_9
        inst✝³ : NontriviallyNormedField 𝕜
        A : Type u_10
        inst✝² : NormedRing A
        inst✝¹ : NormedAlgebra 𝕜 A
        inst✝ : HasSummableGeomSeries A
        z : Units A
        hA : Nontrivial A
        f1 : A → A := fun a => HMul.hMul a z.inv
        f2 : A → A := fun b => Ring.inverse (HSub.hSub 1 b)
        f3 : A → A := fun c => HSub.hSub 1 (HMul.hMul z.inv c)
        feq : Filter.Eventually (fun y => Eq (Function.comp f1 (Function.comp f2 f3) y …
        ⊢ AnalyticAt 𝕜 f3 ↑z
      -/
    · exact analyticAt_const.sub (analyticAt_const.mul analyticAt_id)
      /-
        🎉 no goals
      -/


lemma analyticOnNhd_inverse {𝕜 : Type*} [NontriviallyNormedField 𝕜]
    {A : Type*} [NormedRing A] [NormedAlgebra 𝕜 A] [HasSummableGeomSeries A] :
    AnalyticOnNhd 𝕜 Ring.inverse {x : A | IsUnit x} :=
  fun _ hx ↦ analyticAt_inverse (IsUnit.unit hx)


lemma hasFPowerSeriesOnBall_inv_one_sub
    (𝕜 𝕝 : Type*) [NontriviallyNormedField 𝕜] [NontriviallyNormedField 𝕝] [NormedAlgebra 𝕜 𝕝] :
    HasFPowerSeriesOnBall (fun x : 𝕝 ↦ (1 - x)⁻¹) (formalMultilinearSeries_geometric 𝕜 𝕝) 0 1 := by
  /-
    𝕜 : Type u_9
    𝕝 : Type u_10
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NontriviallyNormedField 𝕝
    inst✝ : NormedAlgebra 𝕜 𝕝
    ⊢ HasFPowerSeriesOnBall (fun x => Inv.inv (HSub.hSub 1 x)) (formalMultilinearS …
  -/
  convert hasFPowerSeriesOnBall_inverse_one_sub 𝕜 𝕝
  /-
    case h.e'_9.h.h.e
    𝕜 : Type u_9
    𝕝 : Type u_10
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : NontriviallyNormedField 𝕝
    inst✝ : NormedAlgebra 𝕜 𝕝
    x✝ : 𝕝
    ⊢ Eq Inv.inv Ring.inverse
  -/
  exact Ring.inverse_eq_inv'.symm
  /-
    🎉 no goals
  -/


lemma analyticAt_inv_one_sub (𝕝 : Type*) [NontriviallyNormedField 𝕝] [NormedAlgebra 𝕜 𝕝] :
    AnalyticAt 𝕜 (fun x : 𝕝 ↦ (1 - x)⁻¹) 0 :=
  ⟨_, ⟨_, hasFPowerSeriesOnBall_inv_one_sub 𝕜 𝕝⟩⟩


/-- If `𝕝` is a normed field extension of `𝕜`, then the inverse map `𝕝 → 𝕝` is `𝕜`-analytic
away from 0. -/
lemma analyticAt_inv {z : 𝕝} (hz : z ≠ 0) : AnalyticAt 𝕜 Inv.inv z := by
  /-
    𝕜 : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    𝕝 : Type u_7
    inst✝¹ : NontriviallyNormedField 𝕝
    inst✝ : NormedAlgebra 𝕜 𝕝
    z : 𝕝
    hz : Ne z 0
    ⊢ AnalyticAt 𝕜 Inv.inv z
  -/
  convert analyticAt_inverse (𝕜 := 𝕜) (Units.mk0 _ hz)
  /-
    case h.e'_9
    𝕜 : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    𝕝 : Type u_7
    inst✝¹ : NontriviallyNormedField 𝕝
    inst✝ : NormedAlgebra 𝕜 𝕝
    z : 𝕝
    hz : Ne z 0
    ⊢ Eq Inv.inv Ring.inverse
  -/
  exact Ring.inverse_eq_inv'.symm
  /-
    🎉 no goals
  -/


/-- `x⁻¹` is analytic away from zero -/
lemma analyticOnNhd_inv : AnalyticOnNhd 𝕜 (fun z ↦ z⁻¹) {z : 𝕝 | z ≠ 0} := by
  /-
    𝕜 : Type u_2
    inst✝² : NontriviallyNormedField 𝕜
    𝕝 : Type u_7
    inst✝¹ : NontriviallyNormedField 𝕝
    inst✝ : NormedAlgebra 𝕜 𝕝
    ⊢ AnalyticOnNhd 𝕜 (fun z => Inv.inv z) (setOf fun z => Ne z 0)
  -/
  intro z m; exact analyticAt_inv m
             /-
               🎉 no goals
             -/


lemma analyticOn_inv : AnalyticOn 𝕜 (fun z ↦ z⁻¹) {z : 𝕝 | z ≠ 0} :=
  analyticOnNhd_inv.analyticOn


/-- `(f x)⁻¹` is analytic away from `f x = 0` -/
theorem AnalyticWithinAt.inv {f : E → 𝕝} {x : E} {s : Set E}
    (fa : AnalyticWithinAt 𝕜 f s x) (f0 : f x ≠ 0) :
    AnalyticWithinAt 𝕜 (fun x ↦ (f x)⁻¹) s x :=
  (analyticAt_inv f0).comp_analyticWithinAt fa


/-- `(f x)⁻¹` is analytic away from `f x = 0` -/
theorem AnalyticAt.inv {f : E → 𝕝} {x : E} (fa : AnalyticAt 𝕜 f x) (f0 : f x ≠ 0) :
    AnalyticAt 𝕜 (fun x ↦ (f x)⁻¹) x :=
  (analyticAt_inv f0).comp fa


/-- `(f x)⁻¹` is analytic away from `f x = 0` -/
theorem AnalyticOn.inv {f : E → 𝕝} {s : Set E}
    (fa : AnalyticOn 𝕜 f s) (f0 : ∀ x ∈ s, f x ≠ 0) :
    AnalyticOn 𝕜 (fun x ↦ (f x)⁻¹) s :=
  fun x m ↦ (fa x m).inv (f0 x m)


@[deprecated (since := "2024-09-26")]
alias AnalyticWithinOn.inv := AnalyticOn.inv


/-- `(f x)⁻¹` is analytic away from `f x = 0` -/
theorem AnalyticOnNhd.inv {f : E → 𝕝} {s : Set E}
    (fa : AnalyticOnNhd 𝕜 f s) (f0 : ∀ x ∈ s, f x ≠ 0) :
    AnalyticOnNhd 𝕜 (fun x ↦ (f x)⁻¹) s :=
  fun x m ↦ (fa x m).inv (f0 x m)


/-- `f x / g x` is analytic away from `g x = 0` -/
theorem AnalyticWithinAt.div {f g : E → 𝕝} {s : Set E} {x : E}
    (fa : AnalyticWithinAt 𝕜 f s x) (ga : AnalyticWithinAt 𝕜 g s x) (g0 : g x ≠ 0) :
    AnalyticWithinAt 𝕜 (fun x ↦ f x / g x) s x := by
  /-
    𝕜 : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    𝕝 : Type u_7
    inst✝¹ : NontriviallyNormedField 𝕝
    inst✝ : NormedAlgebra 𝕜 𝕝
    f g : E → 𝕝
    s : Set E
    x : E
    fa : AnalyticWithinAt 𝕜 f s x
    ga : AnalyticWithinAt 𝕜 g s x
    g0 : Ne (g x) 0
    ⊢ AnalyticWithinAt 𝕜 (fun x => HDiv.hDiv (f x) (g x)) s x
  -/
  simp_rw [div_eq_mul_inv]; exact fa.mul (ga.inv g0)
                            /-
                              🎉 no goals
                            -/


/-- `f x / g x` is analytic away from `g x = 0` -/
theorem AnalyticAt.div {f g : E → 𝕝} {x : E}
    (fa : AnalyticAt 𝕜 f x) (ga : AnalyticAt 𝕜 g x) (g0 : g x ≠ 0) :
    AnalyticAt 𝕜 (fun x ↦ f x / g x) x := by
  /-
    𝕜 : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    𝕝 : Type u_7
    inst✝¹ : NontriviallyNormedField 𝕝
    inst✝ : NormedAlgebra 𝕜 𝕝
    f g : E → 𝕝
    x : E
    fa : AnalyticAt 𝕜 f x
    ga : AnalyticAt 𝕜 g x
    g0 : Ne (g x) 0
    ⊢ AnalyticAt 𝕜 (fun x => HDiv.hDiv (f x) (g x)) x
  -/
  simp_rw [div_eq_mul_inv]; exact fa.mul (ga.inv g0)
                            /-
                              🎉 no goals
                            -/


/-- `f x / g x` is analytic away from `g x = 0` -/
theorem AnalyticOn.div {f g : E → 𝕝} {s : Set E}
    (fa : AnalyticOn 𝕜 f s) (ga : AnalyticOn 𝕜 g s) (g0 : ∀ x ∈ s, g x ≠ 0) :
    AnalyticOn 𝕜 (fun x ↦ f x / g x) s := fun x m ↦
  (fa x m).div (ga x m) (g0 x m)


@[deprecated (since := "2024-09-26")]
alias AnalyticWithinOn.div := AnalyticOn.div


/-- `f x / g x` is analytic away from `g x = 0` -/
theorem AnalyticOnNhd.div {f g : E → 𝕝} {s : Set E}
    (fa : AnalyticOnNhd 𝕜 f s) (ga : AnalyticOnNhd 𝕜 g s) (g0 : ∀ x ∈ s, g x ≠ 0) :
    AnalyticOnNhd 𝕜 (fun x ↦ f x / g x) s := fun x m ↦
  (fa x m).div (ga x m) (g0 x m)


/-- Finite sums of analytic functions are analytic -/
theorem Finset.analyticWithinAt_sum {f : α → E → F} {c : E} {s : Set E}
    (N : Finset α) (h : ∀ n ∈ N, AnalyticWithinAt 𝕜 (f n) s c) :
    AnalyticWithinAt 𝕜 (fun z ↦ ∑ n ∈ N, f n z) s c := by
  classical
  induction' N using Finset.induction with a B aB hB
  · simp only [Finset.sum_empty]
    exact analyticWithinAt_const
  · simp_rw [Finset.sum_insert aB]
    simp only [Finset.mem_insert] at h
    exact (h a (Or.inl rfl)).add (hB fun b m ↦ h b (Or.inr m))


/-- Finite sums of analytic functions are analytic -/
theorem Finset.analyticAt_sum {f : α → E → F} {c : E}
    (N : Finset α) (h : ∀ n ∈ N, AnalyticAt 𝕜 (f n) c) :
    AnalyticAt 𝕜 (fun z ↦ ∑ n ∈ N, f n z) c := by
  /-
    α : Type u_1
    𝕜 : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_3
    F : Type u_4
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : α → E → F
    c : E
    N : Finset α
    h : ∀ (n : α), Membership.mem N n → AnalyticAt 𝕜 (f n) c
    ⊢ AnalyticAt 𝕜 (fun z => N.sum fun n => f n z) c
  -/
  simp_rw [← analyticWithinAt_univ] at h ⊢
  /-
    α : Type u_1
    𝕜 : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_3
    F : Type u_4
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : α → E → F
    c : E
    N : Finset α
    h : ∀ (n : α), Membership.mem N n → AnalyticWithinAt 𝕜 (f n) Set.univ c
    ⊢ AnalyticWithinAt 𝕜 (fun z => N.sum fun n => f n z) Set.univ c
  -/
  exact N.analyticWithinAt_sum h
  /-
    🎉 no goals
  -/


/-- Finite sums of analytic functions are analytic -/
theorem Finset.analyticOn_sum {f : α → E → F} {s : Set E}
    (N : Finset α) (h : ∀ n ∈ N, AnalyticOn 𝕜 (f n) s) :
    AnalyticOn 𝕜 (fun z ↦ ∑ n ∈ N, f n z) s :=
  fun z zs ↦ N.analyticWithinAt_sum (fun n m ↦ h n m z zs)


@[deprecated (since := "2024-09-26")]
alias Finset.analyticWithinOn_sum := Finset.analyticOn_sum


/-- Finite sums of analytic functions are analytic -/
theorem Finset.analyticOnNhd_sum {f : α → E → F} {s : Set E}
    (N : Finset α) (h : ∀ n ∈ N, AnalyticOnNhd 𝕜 (f n) s) :
    AnalyticOnNhd 𝕜 (fun z ↦ ∑ n ∈ N, f n z) s :=
  fun z zs ↦ N.analyticAt_sum (fun n m ↦ h n m z zs)


/-- Finite products of analytic functions are analytic -/
theorem Finset.analyticWithinAt_prod {A : Type*} [NormedCommRing A] [NormedAlgebra 𝕜 A]
    {f : α → E → A} {c : E} {s : Set E} (N : Finset α) (h : ∀ n ∈ N, AnalyticWithinAt 𝕜 (f n) s c) :
    AnalyticWithinAt 𝕜 (fun z ↦ ∏ n ∈ N, f n z) s c := by
  classical
  induction' N using Finset.induction with a B aB hB
  · simp only [Finset.prod_empty]
    exact analyticWithinAt_const
  · simp_rw [Finset.prod_insert aB]
    simp only [Finset.mem_insert] at h
    exact (h a (Or.inl rfl)).mul (hB fun b m ↦ h b (Or.inr m))


/-- Finite products of analytic functions are analytic -/
theorem Finset.analyticAt_prod {A : Type*} [NormedCommRing A] [NormedAlgebra 𝕜 A]
    {f : α → E → A} {c : E} (N : Finset α) (h : ∀ n ∈ N, AnalyticAt 𝕜 (f n) c) :
    AnalyticAt 𝕜 (fun z ↦ ∏ n ∈ N, f n z) c := by
  /-
    α : Type u_1
    𝕜 : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    A : Type u_9
    inst✝¹ : NormedCommRing A
    inst✝ : NormedAlgebra 𝕜 A
    f : α → E → A
    c : E
    N : Finset α
    h : ∀ (n : α), Membership.mem N n → AnalyticAt 𝕜 (f n) c
    ⊢ AnalyticAt 𝕜 (fun z => N.prod fun n => f n z) c
  -/
  simp_rw [← analyticWithinAt_univ] at h ⊢
  /-
    α : Type u_1
    𝕜 : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    A : Type u_9
    inst✝¹ : NormedCommRing A
    inst✝ : NormedAlgebra 𝕜 A
    f : α → E → A
    c : E
    N : Finset α
    h : ∀ (n : α), Membership.mem N n → AnalyticWithinAt 𝕜 (f n) Set.univ c
    ⊢ AnalyticWithinAt 𝕜 (fun z => N.prod fun n => f n z) Set.univ c
  -/
  exact N.analyticWithinAt_prod h
  /-
    🎉 no goals
  -/


/-- Finite products of analytic functions are analytic -/
theorem Finset.analyticOn_prod {A : Type*} [NormedCommRing A] [NormedAlgebra 𝕜 A]
    {f : α → E → A} {s : Set E} (N : Finset α) (h : ∀ n ∈ N, AnalyticOn 𝕜 (f n) s) :
    AnalyticOn 𝕜 (fun z ↦ ∏ n ∈ N, f n z) s :=
  fun z zs ↦ N.analyticWithinAt_prod (fun n m ↦ h n m z zs)


@[deprecated (since := "2024-09-26")]
alias Finset.analyticWithinOn_prod := Finset.analyticOn_prod


/-- Finite products of analytic functions are analytic -/
theorem Finset.analyticOnNhd_prod {A : Type*} [NormedCommRing A] [NormedAlgebra 𝕜 A]
    {f : α → E → A} {s : Set E} (N : Finset α) (h : ∀ n ∈ N, AnalyticOnNhd 𝕜 (f n) s) :
    AnalyticOnNhd 𝕜 (fun z ↦ ∏ n ∈ N, f n z) s :=
  fun z zs ↦ N.analyticAt_prod (fun n m ↦ h n m z zs)


theorem HasFPowerSeriesWithinOnBall.unshift (hf : HasFPowerSeriesWithinOnBall f pf s x r) :
    HasFPowerSeriesWithinOnBall (fun y ↦ z + f y (y - x)) (pf.unshift z) s x r where
  r_le := by
    /-
      𝕜 : Type u_2
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_3
      F : Type u_4
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → ContinuousLinearMap (RingHom.id 𝕜) E F
      pf : FormalMultilinearSeries 𝕜 E (ContinuousLinearMap (RingHom.id 𝕜) E F)
      s : Set E
      x : E
      r : ENNReal
      z : F
      hf : HasFPowerSeriesWithinOnBall f pf s x r
      ⊢ LE.le r (pf.unshift z).radius
    -/
    rw [FormalMultilinearSeries.radius_unshift]
    /-
      𝕜 : Type u_2
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_3
      F : Type u_4
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → ContinuousLinearMap (RingHom.id 𝕜) E F
      pf : FormalMultilinearSeries 𝕜 E (ContinuousLinearMap (RingHom.id 𝕜) E F)
      s : Set E
      x : E
      r : ENNReal
      z : F
      hf : HasFPowerSeriesWithinOnBall f pf s x r
      ⊢ LE.le r pf.radius
    -/
    exact hf.r_le
    /-
      🎉 no goals
    -/
  r_pos := hf.r_pos
  hasSum := by
    /-
      𝕜 : Type u_2
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_3
      F : Type u_4
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → ContinuousLinearMap (RingHom.id 𝕜) E F
      pf : FormalMultilinearSeries 𝕜 E (ContinuousLinearMap (RingHom.id 𝕜) E F)
      s : Set E
      x : E
      r : ENNReal
      z : F
      hf : HasFPowerSeriesWithinOnBall f pf s x r
      ⊢ ∀ {y : E}, Membership.mem (Insert.insert x s) (HAdd.hAdd x y) → Membership.m …
    -/
    intro y hy h'y
    /-
      𝕜 : Type u_2
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_3
      F : Type u_4
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → ContinuousLinearMap (RingHom.id 𝕜) E F
      pf : FormalMultilinearSeries 𝕜 E (ContinuousLinearMap (RingHom.id 𝕜) E F)
      s : Set E
      x : E
      r : ENNReal
      z : F
      hf : HasFPowerSeriesWithinOnBall f pf s x r
      y : E
      hy : Membership.mem (Insert.insert x s) (HAdd.hAdd x y)
      h'y : Membership.mem (EMetric.ball 0 r) y
      ⊢ HasSum (fun n => (pf.unshift z n) fun x => y) (HAdd.hAdd z ((f (HAdd.hAdd x  …
    -/
    apply HasSum.zero_add
    simp only [FormalMultilinearSeries.unshift, Nat.succ_eq_add_one,
      continuousMultilinearCurryRightEquiv_symm_apply', add_sub_cancel_left]
    /-
      case h
      𝕜 : Type u_2
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_3
      F : Type u_4
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → ContinuousLinearMap (RingHom.id 𝕜) E F
      pf : FormalMultilinearSeries 𝕜 E (ContinuousLinearMap (RingHom.id 𝕜) E F)
      s : Set E
      x : E
      r : ENNReal
      z : F
      hf : HasFPowerSeriesWithinOnBall f pf s x r
      y : E
      hy : Membership.mem (Insert.insert x s) (HAdd.hAdd x y)
      h'y : Membership.mem (EMetric.ball 0 r) y
      ⊢ HasSum (fun n => ((pf n) (Fin.init fun x => y)) y) ((f (HAdd.hAdd x y)) y)
    -/
    exact (ContinuousLinearMap.apply 𝕜 F y).hasSum (hf.hasSum hy h'y)
    /-
      🎉 no goals
    -/


theorem HasFPowerSeriesOnBall.unshift (hf : HasFPowerSeriesOnBall f pf x r) :
    HasFPowerSeriesOnBall (fun y ↦ z + f y (y - x)) (pf.unshift z) x r where
  r_le := by
    /-
      𝕜 : Type u_2
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_3
      F : Type u_4
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → ContinuousLinearMap (RingHom.id 𝕜) E F
      pf : FormalMultilinearSeries 𝕜 E (ContinuousLinearMap (RingHom.id 𝕜) E F)
      x : E
      r : ENNReal
      z : F
      hf : HasFPowerSeriesOnBall f pf x r
      ⊢ LE.le r (pf.unshift z).radius
    -/
    rw [FormalMultilinearSeries.radius_unshift]
    /-
      𝕜 : Type u_2
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_3
      F : Type u_4
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → ContinuousLinearMap (RingHom.id 𝕜) E F
      pf : FormalMultilinearSeries 𝕜 E (ContinuousLinearMap (RingHom.id 𝕜) E F)
      x : E
      r : ENNReal
      z : F
      hf : HasFPowerSeriesOnBall f pf x r
      ⊢ LE.le r pf.radius
    -/
    exact hf.r_le
    /-
      🎉 no goals
    -/
  r_pos := hf.r_pos
  hasSum := by
    /-
      𝕜 : Type u_2
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_3
      F : Type u_4
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → ContinuousLinearMap (RingHom.id 𝕜) E F
      pf : FormalMultilinearSeries 𝕜 E (ContinuousLinearMap (RingHom.id 𝕜) E F)
      x : E
      r : ENNReal
      z : F
      hf : HasFPowerSeriesOnBall f pf x r
      ⊢ ∀ {y : E}, Membership.mem (EMetric.ball 0 r) y → HasSum (fun n => (pf.unshif …
    -/
    intro y hy
    /-
      𝕜 : Type u_2
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_3
      F : Type u_4
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → ContinuousLinearMap (RingHom.id 𝕜) E F
      pf : FormalMultilinearSeries 𝕜 E (ContinuousLinearMap (RingHom.id 𝕜) E F)
      x : E
      r : ENNReal
      z : F
      hf : HasFPowerSeriesOnBall f pf x r
      y : E
      hy : Membership.mem (EMetric.ball 0 r) y
      ⊢ HasSum (fun n => (pf.unshift z n) fun x => y) (HAdd.hAdd z ((f (HAdd.hAdd x  …
    -/
    apply HasSum.zero_add
    simp only [FormalMultilinearSeries.unshift, Nat.succ_eq_add_one,
      continuousMultilinearCurryRightEquiv_symm_apply', add_sub_cancel_left]
    /-
      case h
      𝕜 : Type u_2
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_3
      F : Type u_4
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → ContinuousLinearMap (RingHom.id 𝕜) E F
      pf : FormalMultilinearSeries 𝕜 E (ContinuousLinearMap (RingHom.id 𝕜) E F)
      x : E
      r : ENNReal
      z : F
      hf : HasFPowerSeriesOnBall f pf x r
      y : E
      hy : Membership.mem (EMetric.ball 0 r) y
      ⊢ HasSum (fun n => ((pf n) (Fin.init fun x => y)) y) ((f (HAdd.hAdd x y)) y)
    -/
    exact (ContinuousLinearMap.apply 𝕜 F y).hasSum (hf.hasSum hy)
    /-
      🎉 no goals
    -/


theorem HasFPowerSeriesWithinAt.unshift (hf : HasFPowerSeriesWithinAt f pf s x) :
    HasFPowerSeriesWithinAt (fun y ↦ z + f y (y - x)) (pf.unshift z) s x :=
  let ⟨_, hrf⟩ := hf
  hrf.unshift.hasFPowerSeriesWithinAt


