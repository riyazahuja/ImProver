/-- `AnalyticWithinAt` is trivial if `{x} ∈ 𝓝[s] x` -/
lemma analyticWithinAt_of_singleton_mem {f : E → F} {s : Set E} {x : E} (h : {x} ∈ 𝓝[s] x) :
    AnalyticWithinAt 𝕜 f s x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    s : Set E
    x : E
    h : Membership.mem (nhdsWithin x s) (Singleton.singleton x)
    ⊢ AnalyticWithinAt 𝕜 f s x
  -/
  rcases mem_nhdsWithin.mp h with ⟨t, ot, xt, st⟩
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    s : Set E
    x : E
    h : Membership.mem (nhdsWithin x s) (Singleton.singleton x)
    t : Set E
    ot : IsOpen t
    xt : Membership.mem t x
    st : HasSubset.Subset (Inter.inter t s) (Singleton.singleton x)
    ⊢ AnalyticWithinAt 𝕜 f s x
  -/
  rcases Metric.mem_nhds_iff.mp (ot.mem_nhds xt) with ⟨r, r0, rt⟩
  exact ⟨constFormalMultilinearSeries 𝕜 E (f x), .ofReal r,
  { r_le := by simp only [FormalMultilinearSeries.constFormalMultilinearSeries_radius, le_top]
    r_pos := by positivity
    hasSum := by
      intro y ys yr
      simp only [subset_singleton_iff, mem_inter_iff, and_imp] at st
      simp only [mem_insert_iff, add_right_eq_self] at ys
      have : x + y = x := by
        rcases ys with rfl | ys
        · simp
        · exact st (x + y) (rt (by simpa using yr)) ys
      simp only [this]
      apply (hasFPowerSeriesOnBall_const (e := 0)).hasSum
      simp only [Metric.emetric_ball_top, mem_univ] }⟩


/-- If `f` is `AnalyticOn` near each point in a set, it is `AnalyticOn` the set -/
lemma analyticOn_of_locally_analyticOn {f : E → F} {s : Set E}
    (h : ∀ x ∈ s, ∃ u, IsOpen u ∧ x ∈ u ∧ AnalyticOn 𝕜 f (s ∩ u)) :
    AnalyticOn 𝕜 f s := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    s : Set E
    h : ∀ (x : E), Membership.mem s x → Exists fun u => And (IsOpen u) (And (Membe …
    ⊢ AnalyticOn 𝕜 f s
  -/
  intro x m
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    s : Set E
    h : ∀ (x : E), Membership.mem s x → Exists fun u => And (IsOpen u) (And (Membe …
    x : E
    m : Membership.mem s x
    ⊢ AnalyticWithinAt 𝕜 f s x
  -/
  rcases h x m with ⟨u, ou, xu, fu⟩
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    s : Set E
    h : ∀ (x : E), Membership.mem s x → Exists fun u => And (IsOpen u) (And (Membe …
    x : E
    m : Membership.mem s x
    u : Set E
    ou : IsOpen u
    xu : Membership.mem u x
    fu : AnalyticOn 𝕜 f (Inter.inter s u)
    ⊢ AnalyticWithinAt 𝕜 f s x
  -/
  rcases Metric.mem_nhds_iff.mp (ou.mem_nhds xu) with ⟨r, r0, ru⟩
  /-
    case intro.intro.intro.intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    s : Set E
    h : ∀ (x : E), Membership.mem s x → Exists fun u => And (IsOpen u) (And (Membe …
    x : E
    m : Membership.mem s x
    u : Set E
    ou : IsOpen u
    xu : Membership.mem u x
    fu : AnalyticOn 𝕜 f (Inter.inter s u)
    r : Real
    r0 : GT.gt r 0
    ru : HasSubset.Subset (Metric.ball x r) u
    ⊢ AnalyticWithinAt 𝕜 f s x
  -/
  rcases fu x ⟨m, xu⟩ with ⟨p, t, fp⟩
  exact ⟨p, min (.ofReal r) t,
    { r_pos := lt_min (by positivity) fp.r_pos
      r_le := min_le_of_right_le fp.r_le
      hasSum := by
        intro y ys yr
        simp only [EMetric.mem_ball, lt_min_iff, edist_lt_ofReal, dist_zero_right] at yr
        apply fp.hasSum
        · simp only [mem_insert_iff, add_right_eq_self] at ys
          rcases ys with rfl | ys
          · simp
          · simp only [mem_insert_iff, add_right_eq_self, mem_inter_iff, ys, true_and]
            apply Or.inr (ru ?_)
            simp only [Metric.mem_ball, dist_self_add_left, yr]
        · simp only [EMetric.mem_ball, yr] }⟩


@[deprecated (since := "2024-09-26")]
alias analyticWithinOn_of_locally_analyticWithinOn := analyticOn_of_locally_analyticOn


/-- On open sets, `AnalyticOnNhd` and `AnalyticOn` coincide -/
lemma IsOpen.analyticOn_iff_analyticOnNhd {f : E → F} {s : Set E} (hs : IsOpen s) :
    AnalyticOn 𝕜 f s ↔ AnalyticOnNhd 𝕜 f s := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    s : Set E
    hs : IsOpen s
    ⊢ Iff (AnalyticOn 𝕜 f s) (AnalyticOnNhd 𝕜 f s)
  -/
  refine ⟨?_, AnalyticOnNhd.analyticOn⟩
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    s : Set E
    hs : IsOpen s
    ⊢ AnalyticOn 𝕜 f s → AnalyticOnNhd 𝕜 f s
  -/
  intro hf x m
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    s : Set E
    hs : IsOpen s
    hf : AnalyticOn 𝕜 f s
    x : E
    m : Membership.mem s x
    ⊢ AnalyticAt 𝕜 f x
  -/
  rcases Metric.mem_nhds_iff.mp (hs.mem_nhds m) with ⟨r, r0, rs⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    s : Set E
    hs : IsOpen s
    hf : AnalyticOn 𝕜 f s
    x : E
    m : Membership.mem s x
    r : Real
    r0 : GT.gt r 0
    rs : HasSubset.Subset (Metric.ball x r) s
    ⊢ AnalyticAt 𝕜 f x
  -/
  rcases hf x m with ⟨p, t, fp⟩
  exact ⟨p, min (.ofReal r) t,
  { r_pos := lt_min (by positivity) fp.r_pos
    r_le := min_le_of_right_le fp.r_le
    hasSum := by
      intro y ym
      simp only [EMetric.mem_ball, lt_min_iff, edist_lt_ofReal, dist_zero_right] at ym
      refine fp.hasSum ?_ ym.2
      apply mem_insert_of_mem
      apply rs
      simp only [Metric.mem_ball, dist_self_add_left, ym.1] }⟩


@[deprecated (since := "2024-09-26")]
alias IsOpen.analyticWithinOn_iff_analyticOn := IsOpen.analyticOn_iff_analyticOnNhd



set_option linter.style.multiGoal false in
/-- `f` has power series `p` at `x` iff some local extension of `f` has that series -/
lemma hasFPowerSeriesWithinOnBall_iff_exists_hasFPowerSeriesOnBall [CompleteSpace F] {f : E → F}
    {p : FormalMultilinearSeries 𝕜 E F} {s : Set E} {x : E} {r : ℝ≥0∞} :
    HasFPowerSeriesWithinOnBall f p s x r ↔
      ∃ g, EqOn f g (insert x s ∩ EMetric.ball x r) ∧
        HasFPowerSeriesOnBall g p x r := by
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : E → F
    p : FormalMultilinearSeries 𝕜 E F
    s : Set E
    x : E
    r : ENNReal
    ⊢ Iff (HasFPowerSeriesWithinOnBall f p s x r) (Exists fun g => And (Set.EqOn f …
  -/
  constructor
    /-
      case mp
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : CompleteSpace F
      f : E → F
      p : FormalMultilinearSeries 𝕜 E F
      s : Set E
      x : E
      r : ENNReal
      ⊢ HasFPowerSeriesWithinOnBall f p s x r → Exists fun g => And (Set.EqOn f g (I …
    -/
  · intro h
    /-
      case mp
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : CompleteSpace F
      f : E → F
      p : FormalMultilinearSeries 𝕜 E F
      s : Set E
      x : E
      r : ENNReal
      h : HasFPowerSeriesWithinOnBall f p s x r
      ⊢ Exists fun g => And (Set.EqOn f g (Inter.inter (Insert.insert x s) (EMetric. …
    -/
    refine ⟨fun y ↦ p.sum (y - x), ?_, ?_⟩
      /-
        case mp.refine_1
        𝕜 : Type u_1
        inst✝⁵ : NontriviallyNormedField 𝕜
        E : Type u_2
        F : Type u_3
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace 𝕜 E
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace 𝕜 F
        inst✝ : CompleteSpace F
        f : E → F
        p : FormalMultilinearSeries 𝕜 E F
        s : Set E
        x : E
        r : ENNReal
        h : HasFPowerSeriesWithinOnBall f p s x r
        ⊢ Set.EqOn f (fun y => p.sum (HSub.hSub y x)) (Inter.inter (Insert.insert x s) …
      -/
    · intro y ⟨ys,yb⟩
      /-
        case mp.refine_1
        𝕜 : Type u_1
        inst✝⁵ : NontriviallyNormedField 𝕜
        E : Type u_2
        F : Type u_3
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace 𝕜 E
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace 𝕜 F
        inst✝ : CompleteSpace F
        f : E → F
        p : FormalMultilinearSeries 𝕜 E F
        s : Set E
        x : E
        r : ENNReal
        h : HasFPowerSeriesWithinOnBall f p s x r
        y : E
        ys : Membership.mem (Insert.insert x s) y
        yb : Membership.mem (EMetric.ball x r) y
        ⊢ Eq (f y) ((fun y => p.sum (HSub.hSub y x)) y)
      -/
      simp only [EMetric.mem_ball, edist_eq_coe_nnnorm_sub] at yb
      /-
        case mp.refine_1
        𝕜 : Type u_1
        inst✝⁵ : NontriviallyNormedField 𝕜
        E : Type u_2
        F : Type u_3
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace 𝕜 E
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace 𝕜 F
        inst✝ : CompleteSpace F
        f : E → F
        p : FormalMultilinearSeries 𝕜 E F
        s : Set E
        x : E
        r : ENNReal
        h : HasFPowerSeriesWithinOnBall f p s x r
        y : E
        ys : Membership.mem (Insert.insert x s) y
        yb : LT.lt (↑(NNNorm.nnnorm (HSub.hSub y x))) r
        ⊢ Eq (f y) ((fun y => p.sum (HSub.hSub y x)) y)
      -/
      have e0 := p.hasSum (x := y - x) ?_
      /-
        case mp.refine_1.refine_2
        𝕜 : Type u_1
        inst✝⁵ : NontriviallyNormedField 𝕜
        E : Type u_2
        F : Type u_3
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace 𝕜 E
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace 𝕜 F
        inst✝ : CompleteSpace F
        f : E → F
        p : FormalMultilinearSeries 𝕜 E F
        s : Set E
        x : E
        r : ENNReal
        h : HasFPowerSeriesWithinOnBall f p s x r
        y : E
        ys : Membership.mem (Insert.insert x s) y
        yb : LT.lt (↑(NNNorm.nnnorm (HSub.hSub y x))) r
        e0 : HasSum (fun n => (p n) fun x_1 => HSub.hSub y x) (p.sum (HSub.hSub y x))
        ⊢ Eq (f y) ((fun y => p.sum (HSub.hSub y x)) y)
      -/
      have e1 := (h.hasSum (y := y - x) ?_ ?_)
        /-
          case mp.refine_1.refine_2.refine_3
          𝕜 : Type u_1
          inst✝⁵ : NontriviallyNormedField 𝕜
          E : Type u_2
          F : Type u_3
          inst✝⁴ : NormedAddCommGroup E
          inst✝³ : NormedSpace 𝕜 E
          inst✝² : NormedAddCommGroup F
          inst✝¹ : NormedSpace 𝕜 F
          inst✝ : CompleteSpace F
          f : E → F
          p : FormalMultilinearSeries 𝕜 E F
          s : Set E
          x : E
          r : ENNReal
          h : HasFPowerSeriesWithinOnBall f p s x r
          y : E
          ys : Membership.mem (Insert.insert x s) y
          yb : LT.lt (↑(NNNorm.nnnorm (HSub.hSub y x))) r
          e0 : HasSum (fun n => (p n) fun x_1 => HSub.hSub y x) (p.sum (HSub.hSub y x))
          e1 : HasSum (fun n => (p n) fun x_1 => HSub.hSub y x) (f (HAdd.hAdd x (HSub.hS …
          ⊢ Eq (f y) ((fun y => p.sum (HSub.hSub y x)) y)
        -/
      · simp only [add_sub_cancel] at e1
        /-
          case mp.refine_1.refine_2.refine_3
          𝕜 : Type u_1
          inst✝⁵ : NontriviallyNormedField 𝕜
          E : Type u_2
          F : Type u_3
          inst✝⁴ : NormedAddCommGroup E
          inst✝³ : NormedSpace 𝕜 E
          inst✝² : NormedAddCommGroup F
          inst✝¹ : NormedSpace 𝕜 F
          inst✝ : CompleteSpace F
          f : E → F
          p : FormalMultilinearSeries 𝕜 E F
          s : Set E
          x : E
          r : ENNReal
          h : HasFPowerSeriesWithinOnBall f p s x r
          y : E
          ys : Membership.mem (Insert.insert x s) y
          yb : LT.lt (↑(NNNorm.nnnorm (HSub.hSub y x))) r
          e0 : HasSum (fun n => (p n) fun x_1 => HSub.hSub y x) (p.sum (HSub.hSub y x))
          e1 : HasSum (fun n => (p n) fun x_1 => HSub.hSub y x) (f y)
          ⊢ Eq (f y) ((fun y => p.sum (HSub.hSub y x)) y)
        -/
        exact e1.unique e0
        /-
          🎉 no goals
        -/
        /-
          case mp.refine_1.refine_2.refine_1
          𝕜 : Type u_1
          inst✝⁵ : NontriviallyNormedField 𝕜
          E : Type u_2
          F : Type u_3
          inst✝⁴ : NormedAddCommGroup E
          inst✝³ : NormedSpace 𝕜 E
          inst✝² : NormedAddCommGroup F
          inst✝¹ : NormedSpace 𝕜 F
          inst✝ : CompleteSpace F
          f : E → F
          p : FormalMultilinearSeries 𝕜 E F
          s : Set E
          x : E
          r : ENNReal
          h : HasFPowerSeriesWithinOnBall f p s x r
          y : E
          ys : Membership.mem (Insert.insert x s) y
          yb : LT.lt (↑(NNNorm.nnnorm (HSub.hSub y x))) r
          e0 : HasSum (fun n => (p n) fun x_1 => HSub.hSub y x) (p.sum (HSub.hSub y x))
          ⊢ Membership.mem (Insert.insert x s) (HAdd.hAdd x (HSub.hSub y x))
        -/
      · simpa only [add_sub_cancel]
        /-
          🎉 no goals
        -/
        /-
          case mp.refine_1.refine_2.refine_2
          𝕜 : Type u_1
          inst✝⁵ : NontriviallyNormedField 𝕜
          E : Type u_2
          F : Type u_3
          inst✝⁴ : NormedAddCommGroup E
          inst✝³ : NormedSpace 𝕜 E
          inst✝² : NormedAddCommGroup F
          inst✝¹ : NormedSpace 𝕜 F
          inst✝ : CompleteSpace F
          f : E → F
          p : FormalMultilinearSeries 𝕜 E F
          s : Set E
          x : E
          r : ENNReal
          h : HasFPowerSeriesWithinOnBall f p s x r
          y : E
          ys : Membership.mem (Insert.insert x s) y
          yb : LT.lt (↑(NNNorm.nnnorm (HSub.hSub y x))) r
          e0 : HasSum (fun n => (p n) fun x_1 => HSub.hSub y x) (p.sum (HSub.hSub y x))
          ⊢ Membership.mem (EMetric.ball 0 r) (HSub.hSub y x)
        -/
      · simpa only [EMetric.mem_ball, edist_eq_coe_nnnorm]
        /-
          🎉 no goals
        -/
        /-
          case mp.refine_1.refine_1
          𝕜 : Type u_1
          inst✝⁵ : NontriviallyNormedField 𝕜
          E : Type u_2
          F : Type u_3
          inst✝⁴ : NormedAddCommGroup E
          inst✝³ : NormedSpace 𝕜 E
          inst✝² : NormedAddCommGroup F
          inst✝¹ : NormedSpace 𝕜 F
          inst✝ : CompleteSpace F
          f : E → F
          p : FormalMultilinearSeries 𝕜 E F
          s : Set E
          x : E
          r : ENNReal
          h : HasFPowerSeriesWithinOnBall f p s x r
          y : E
          ys : Membership.mem (Insert.insert x s) y
          yb : LT.lt (↑(NNNorm.nnnorm (HSub.hSub y x))) r
          ⊢ Membership.mem (EMetric.ball 0 p.radius) (HSub.hSub y x)
        -/
      · simp only [EMetric.mem_ball, edist_eq_coe_nnnorm]
        /-
          case mp.refine_1.refine_1
          𝕜 : Type u_1
          inst✝⁵ : NontriviallyNormedField 𝕜
          E : Type u_2
          F : Type u_3
          inst✝⁴ : NormedAddCommGroup E
          inst✝³ : NormedSpace 𝕜 E
          inst✝² : NormedAddCommGroup F
          inst✝¹ : NormedSpace 𝕜 F
          inst✝ : CompleteSpace F
          f : E → F
          p : FormalMultilinearSeries 𝕜 E F
          s : Set E
          x : E
          r : ENNReal
          h : HasFPowerSeriesWithinOnBall f p s x r
          y : E
          ys : Membership.mem (Insert.insert x s) y
          yb : LT.lt (↑(NNNorm.nnnorm (HSub.hSub y x))) r
          ⊢ LT.lt (↑(NNNorm.nnnorm (HSub.hSub y x))) p.radius
        -/
        exact lt_of_lt_of_le yb h.r_le
        /-
          🎉 no goals
        -/
      /-
        case mp.refine_2
        𝕜 : Type u_1
        inst✝⁵ : NontriviallyNormedField 𝕜
        E : Type u_2
        F : Type u_3
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace 𝕜 E
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace 𝕜 F
        inst✝ : CompleteSpace F
        f : E → F
        p : FormalMultilinearSeries 𝕜 E F
        s : Set E
        x : E
        r : ENNReal
        h : HasFPowerSeriesWithinOnBall f p s x r
        ⊢ HasFPowerSeriesOnBall (fun y => p.sum (HSub.hSub y x)) p x r
      -/
    · refine ⟨h.r_le, h.r_pos, ?_⟩
      /-
        case mp.refine_2
        𝕜 : Type u_1
        inst✝⁵ : NontriviallyNormedField 𝕜
        E : Type u_2
        F : Type u_3
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace 𝕜 E
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace 𝕜 F
        inst✝ : CompleteSpace F
        f : E → F
        p : FormalMultilinearSeries 𝕜 E F
        s : Set E
        x : E
        r : ENNReal
        h : HasFPowerSeriesWithinOnBall f p s x r
        ⊢ ∀ {y : E}, Membership.mem (EMetric.ball 0 r) y → HasSum (fun n => (p n) fun  …
      -/
      intro y lt
      /-
        case mp.refine_2
        𝕜 : Type u_1
        inst✝⁵ : NontriviallyNormedField 𝕜
        E : Type u_2
        F : Type u_3
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace 𝕜 E
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace 𝕜 F
        inst✝ : CompleteSpace F
        f : E → F
        p : FormalMultilinearSeries 𝕜 E F
        s : Set E
        x : E
        r : ENNReal
        h : HasFPowerSeriesWithinOnBall f p s x r
        y : E
        lt : Membership.mem (EMetric.ball 0 r) y
        ⊢ HasSum (fun n => (p n) fun x => y) (p.sum (HSub.hSub (HAdd.hAdd x y) x))
      -/
      simp only [add_sub_cancel_left]
      /-
        case mp.refine_2
        𝕜 : Type u_1
        inst✝⁵ : NontriviallyNormedField 𝕜
        E : Type u_2
        F : Type u_3
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace 𝕜 E
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace 𝕜 F
        inst✝ : CompleteSpace F
        f : E → F
        p : FormalMultilinearSeries 𝕜 E F
        s : Set E
        x : E
        r : ENNReal
        h : HasFPowerSeriesWithinOnBall f p s x r
        y : E
        lt : Membership.mem (EMetric.ball 0 r) y
        ⊢ HasSum (fun n => (p n) fun x => y) (p.sum y)
      -/
      apply p.hasSum
      /-
        case mp.refine_2
        𝕜 : Type u_1
        inst✝⁵ : NontriviallyNormedField 𝕜
        E : Type u_2
        F : Type u_3
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace 𝕜 E
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace 𝕜 F
        inst✝ : CompleteSpace F
        f : E → F
        p : FormalMultilinearSeries 𝕜 E F
        s : Set E
        x : E
        r : ENNReal
        h : HasFPowerSeriesWithinOnBall f p s x r
        y : E
        lt : Membership.mem (EMetric.ball 0 r) y
        ⊢ Membership.mem (EMetric.ball 0 p.radius) y
      -/
      simp only [EMetric.mem_ball] at lt ⊢
      /-
        case mp.refine_2
        𝕜 : Type u_1
        inst✝⁵ : NontriviallyNormedField 𝕜
        E : Type u_2
        F : Type u_3
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace 𝕜 E
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace 𝕜 F
        inst✝ : CompleteSpace F
        f : E → F
        p : FormalMultilinearSeries 𝕜 E F
        s : Set E
        x : E
        r : ENNReal
        h : HasFPowerSeriesWithinOnBall f p s x r
        y : E
        lt : LT.lt (EDist.edist y 0) r
        ⊢ LT.lt (EDist.edist y 0) p.radius
      -/
      exact lt_of_lt_of_le lt h.r_le
      /-
        🎉 no goals
      -/
    /-
      case mpr
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : CompleteSpace F
      f : E → F
      p : FormalMultilinearSeries 𝕜 E F
      s : Set E
      x : E
      r : ENNReal
      ⊢ (Exists fun g => And (Set.EqOn f g (Inter.inter (Insert.insert x s) (EMetric …
    -/
  · intro ⟨g, hfg, hg⟩
    /-
      case mpr
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : CompleteSpace F
      f : E → F
      p : FormalMultilinearSeries 𝕜 E F
      s : Set E
      x : E
      r : ENNReal
      g : E → F
      hfg : Set.EqOn f g (Inter.inter (Insert.insert x s) (EMetric.ball x r))
      hg : HasFPowerSeriesOnBall g p x r
      ⊢ HasFPowerSeriesWithinOnBall f p s x r
    -/
    refine ⟨hg.r_le, hg.r_pos, ?_⟩
    /-
      case mpr
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : CompleteSpace F
      f : E → F
      p : FormalMultilinearSeries 𝕜 E F
      s : Set E
      x : E
      r : ENNReal
      g : E → F
      hfg : Set.EqOn f g (Inter.inter (Insert.insert x s) (EMetric.ball x r))
      hg : HasFPowerSeriesOnBall g p x r
      ⊢ ∀ {y : E}, Membership.mem (Insert.insert x s) (HAdd.hAdd x y) → Membership.m …
    -/
    intro y ys lt
    /-
      case mpr
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : CompleteSpace F
      f : E → F
      p : FormalMultilinearSeries 𝕜 E F
      s : Set E
      x : E
      r : ENNReal
      g : E → F
      hfg : Set.EqOn f g (Inter.inter (Insert.insert x s) (EMetric.ball x r))
      hg : HasFPowerSeriesOnBall g p x r
      y : E
      ys : Membership.mem (Insert.insert x s) (HAdd.hAdd x y)
      lt : Membership.mem (EMetric.ball 0 r) y
      ⊢ HasSum (fun n => (p n) fun x => y) (f (HAdd.hAdd x y))
    -/
    rw [hfg]
      /-
        case mpr
        𝕜 : Type u_1
        inst✝⁵ : NontriviallyNormedField 𝕜
        E : Type u_2
        F : Type u_3
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace 𝕜 E
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace 𝕜 F
        inst✝ : CompleteSpace F
        f : E → F
        p : FormalMultilinearSeries 𝕜 E F
        s : Set E
        x : E
        r : ENNReal
        g : E → F
        hfg : Set.EqOn f g (Inter.inter (Insert.insert x s) (EMetric.ball x r))
        hg : HasFPowerSeriesOnBall g p x r
        y : E
        ys : Membership.mem (Insert.insert x s) (HAdd.hAdd x y)
        lt : Membership.mem (EMetric.ball 0 r) y
        ⊢ HasSum (fun n => (p n) fun x => y) (g (HAdd.hAdd x y))
      -/
    · exact hg.hasSum lt
      /-
        🎉 no goals
      -/
      /-
        case mpr.a
        𝕜 : Type u_1
        inst✝⁵ : NontriviallyNormedField 𝕜
        E : Type u_2
        F : Type u_3
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace 𝕜 E
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace 𝕜 F
        inst✝ : CompleteSpace F
        f : E → F
        p : FormalMultilinearSeries 𝕜 E F
        s : Set E
        x : E
        r : ENNReal
        g : E → F
        hfg : Set.EqOn f g (Inter.inter (Insert.insert x s) (EMetric.ball x r))
        hg : HasFPowerSeriesOnBall g p x r
        y : E
        ys : Membership.mem (Insert.insert x s) (HAdd.hAdd x y)
        lt : Membership.mem (EMetric.ball 0 r) y
        ⊢ Membership.mem (Inter.inter (Insert.insert x s) (EMetric.ball x r)) (HAdd.hA …
      -/
    · refine ⟨ys, ?_⟩
      /-
        case mpr.a
        𝕜 : Type u_1
        inst✝⁵ : NontriviallyNormedField 𝕜
        E : Type u_2
        F : Type u_3
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace 𝕜 E
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace 𝕜 F
        inst✝ : CompleteSpace F
        f : E → F
        p : FormalMultilinearSeries 𝕜 E F
        s : Set E
        x : E
        r : ENNReal
        g : E → F
        hfg : Set.EqOn f g (Inter.inter (Insert.insert x s) (EMetric.ball x r))
        hg : HasFPowerSeriesOnBall g p x r
        y : E
        ys : Membership.mem (Insert.insert x s) (HAdd.hAdd x y)
        lt : Membership.mem (EMetric.ball 0 r) y
        ⊢ Membership.mem (EMetric.ball x r) (HAdd.hAdd x y)
      -/
      simpa only [EMetric.mem_ball, edist_eq_coe_nnnorm_sub, add_sub_cancel_left, sub_zero] using lt
      /-
        🎉 no goals
      -/


/-- `f` has power series `p` at `x` iff some local extension of `f` has that series -/
lemma hasFPowerSeriesWithinAt_iff_exists_hasFPowerSeriesAt [CompleteSpace F] {f : E → F}
    {p : FormalMultilinearSeries 𝕜 E F} {s : Set E} {x : E} :
    HasFPowerSeriesWithinAt f p s x ↔
      ∃ g, f =ᶠ[𝓝[insert x s] x] g ∧ HasFPowerSeriesAt g p x := by
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : E → F
    p : FormalMultilinearSeries 𝕜 E F
    s : Set E
    x : E
    ⊢ Iff (HasFPowerSeriesWithinAt f p s x) (Exists fun g => And ((nhdsWithin x (I …
  -/
  constructor
    /-
      case mp
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : CompleteSpace F
      f : E → F
      p : FormalMultilinearSeries 𝕜 E F
      s : Set E
      x : E
      ⊢ HasFPowerSeriesWithinAt f p s x → Exists fun g => And ((nhdsWithin x (Insert …
    -/
  · intro ⟨r, h⟩
    /-
      case mp
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : CompleteSpace F
      f : E → F
      p : FormalMultilinearSeries 𝕜 E F
      s : Set E
      x : E
      r : ENNReal
      h : HasFPowerSeriesWithinOnBall f p s x r
      ⊢ Exists fun g => And ((nhdsWithin x (Insert.insert x s)).EventuallyEq f g) (H …
    -/
    rcases hasFPowerSeriesWithinOnBall_iff_exists_hasFPowerSeriesOnBall.mp h with ⟨g, e, h⟩
    /-
      case mp.intro.intro
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : CompleteSpace F
      f : E → F
      p : FormalMultilinearSeries 𝕜 E F
      s : Set E
      x : E
      r : ENNReal
      h✝ : HasFPowerSeriesWithinOnBall f p s x r
      g : E → F
      e : Set.EqOn f g (Inter.inter (Insert.insert x s) (EMetric.ball x r))
      h : HasFPowerSeriesOnBall g p x r
      ⊢ Exists fun g => And ((nhdsWithin x (Insert.insert x s)).EventuallyEq f g) (H …
    -/
    refine ⟨g, ?_, ⟨r, h⟩⟩
    /-
      case mp.intro.intro
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : CompleteSpace F
      f : E → F
      p : FormalMultilinearSeries 𝕜 E F
      s : Set E
      x : E
      r : ENNReal
      h✝ : HasFPowerSeriesWithinOnBall f p s x r
      g : E → F
      e : Set.EqOn f g (Inter.inter (Insert.insert x s) (EMetric.ball x r))
      h : HasFPowerSeriesOnBall g p x r
      ⊢ (nhdsWithin x (Insert.insert x s)).EventuallyEq f g
    -/
    refine Filter.eventuallyEq_iff_exists_mem.mpr ⟨_, ?_, e⟩
    /-
      case mp.intro.intro
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : CompleteSpace F
      f : E → F
      p : FormalMultilinearSeries 𝕜 E F
      s : Set E
      x : E
      r : ENNReal
      h✝ : HasFPowerSeriesWithinOnBall f p s x r
      g : E → F
      e : Set.EqOn f g (Inter.inter (Insert.insert x s) (EMetric.ball x r))
      h : HasFPowerSeriesOnBall g p x r
      ⊢ Membership.mem (nhdsWithin x (Insert.insert x s)) (Inter.inter (Insert.inser …
    -/
    exact inter_mem_nhdsWithin _ (EMetric.ball_mem_nhds _ h.r_pos)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : CompleteSpace F
      f : E → F
      p : FormalMultilinearSeries 𝕜 E F
      s : Set E
      x : E
      ⊢ (Exists fun g => And ((nhdsWithin x (Insert.insert x s)).EventuallyEq f g) ( …
    -/
  · intro ⟨g, hfg, ⟨r, hg⟩⟩
    /-
      case mpr
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : CompleteSpace F
      f : E → F
      p : FormalMultilinearSeries 𝕜 E F
      s : Set E
      x : E
      g : E → F
      hfg : (nhdsWithin x (Insert.insert x s)).EventuallyEq f g
      r : ENNReal
      hg : HasFPowerSeriesOnBall g p x r
      ⊢ HasFPowerSeriesWithinAt f p s x
    -/
    simp only [eventuallyEq_nhdsWithin_iff, Metric.eventually_nhds_iff] at hfg
    /-
      case mpr
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : CompleteSpace F
      f : E → F
      p : FormalMultilinearSeries 𝕜 E F
      s : Set E
      x : E
      g : E → F
      r : ENNReal
      hg : HasFPowerSeriesOnBall g p x r
      hfg : Exists fun ε => And (GT.gt ε 0) (∀ ⦃y : E⦄, LT.lt (Dist.dist y x) ε → Me …
      ⊢ HasFPowerSeriesWithinAt f p s x
    -/
    rcases hfg with ⟨e, e0, hfg⟩
    /-
      case mpr.intro.intro
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : CompleteSpace F
      f : E → F
      p : FormalMultilinearSeries 𝕜 E F
      s : Set E
      x : E
      g : E → F
      r : ENNReal
      hg : HasFPowerSeriesOnBall g p x r
      e : Real
      e0 : GT.gt e 0
      hfg : ∀ ⦃y : E⦄, LT.lt (Dist.dist y x) e → Membership.mem (Insert.insert x s)  …
      ⊢ HasFPowerSeriesWithinAt f p s x
    -/
    refine ⟨min r (.ofReal e), ?_⟩
    /-
      case mpr.intro.intro
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : CompleteSpace F
      f : E → F
      p : FormalMultilinearSeries 𝕜 E F
      s : Set E
      x : E
      g : E → F
      r : ENNReal
      hg : HasFPowerSeriesOnBall g p x r
      e : Real
      e0 : GT.gt e 0
      hfg : ∀ ⦃y : E⦄, LT.lt (Dist.dist y x) e → Membership.mem (Insert.insert x s)  …
      ⊢ HasFPowerSeriesWithinOnBall f p s x (Min.min r (ENNReal.ofReal e))
    -/
    refine hasFPowerSeriesWithinOnBall_iff_exists_hasFPowerSeriesOnBall.mpr ⟨g, ?_, ?_⟩
      /-
        case mpr.intro.intro.refine_1
        𝕜 : Type u_1
        inst✝⁵ : NontriviallyNormedField 𝕜
        E : Type u_2
        F : Type u_3
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace 𝕜 E
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace 𝕜 F
        inst✝ : CompleteSpace F
        f : E → F
        p : FormalMultilinearSeries 𝕜 E F
        s : Set E
        x : E
        g : E → F
        r : ENNReal
        hg : HasFPowerSeriesOnBall g p x r
        e : Real
        e0 : GT.gt e 0
        hfg : ∀ ⦃y : E⦄, LT.lt (Dist.dist y x) e → Membership.mem (Insert.insert x s)  …
        ⊢ Set.EqOn f g (Inter.inter (Insert.insert x s) (EMetric.ball x (Min.min r (EN …
      -/
    · intro y ⟨ys, xy⟩
      /-
        case mpr.intro.intro.refine_1
        𝕜 : Type u_1
        inst✝⁵ : NontriviallyNormedField 𝕜
        E : Type u_2
        F : Type u_3
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace 𝕜 E
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace 𝕜 F
        inst✝ : CompleteSpace F
        f : E → F
        p : FormalMultilinearSeries 𝕜 E F
        s : Set E
        x : E
        g : E → F
        r : ENNReal
        hg : HasFPowerSeriesOnBall g p x r
        e : Real
        e0 : GT.gt e 0
        hfg : ∀ ⦃y : E⦄, LT.lt (Dist.dist y x) e → Membership.mem (Insert.insert x s)  …
        y : E
        ys : Membership.mem (Insert.insert x s) y
        xy : Membership.mem (EMetric.ball x (Min.min r (ENNReal.ofReal e))) y
        ⊢ Eq (f y) (g y)
      -/
      refine hfg ?_ ys
      /-
        case mpr.intro.intro.refine_1
        𝕜 : Type u_1
        inst✝⁵ : NontriviallyNormedField 𝕜
        E : Type u_2
        F : Type u_3
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace 𝕜 E
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace 𝕜 F
        inst✝ : CompleteSpace F
        f : E → F
        p : FormalMultilinearSeries 𝕜 E F
        s : Set E
        x : E
        g : E → F
        r : ENNReal
        hg : HasFPowerSeriesOnBall g p x r
        e : Real
        e0 : GT.gt e 0
        hfg : ∀ ⦃y : E⦄, LT.lt (Dist.dist y x) e → Membership.mem (Insert.insert x s)  …
        y : E
        ys : Membership.mem (Insert.insert x s) y
        xy : Membership.mem (EMetric.ball x (Min.min r (ENNReal.ofReal e))) y
        ⊢ LT.lt (Dist.dist y x) e
      -/
      simp only [EMetric.mem_ball, lt_min_iff, edist_lt_ofReal] at xy
      /-
        case mpr.intro.intro.refine_1
        𝕜 : Type u_1
        inst✝⁵ : NontriviallyNormedField 𝕜
        E : Type u_2
        F : Type u_3
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace 𝕜 E
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace 𝕜 F
        inst✝ : CompleteSpace F
        f : E → F
        p : FormalMultilinearSeries 𝕜 E F
        s : Set E
        x : E
        g : E → F
        r : ENNReal
        hg : HasFPowerSeriesOnBall g p x r
        e : Real
        e0 : GT.gt e 0
        hfg : ∀ ⦃y : E⦄, LT.lt (Dist.dist y x) e → Membership.mem (Insert.insert x s)  …
        y : E
        ys : Membership.mem (Insert.insert x s) y
        xy : And (LT.lt (EDist.edist y x) r) (LT.lt (Dist.dist y x) e)
        ⊢ LT.lt (Dist.dist y x) e
      -/
      exact xy.2
      /-
        🎉 no goals
      -/
      /-
        case mpr.intro.intro.refine_2
        𝕜 : Type u_1
        inst✝⁵ : NontriviallyNormedField 𝕜
        E : Type u_2
        F : Type u_3
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace 𝕜 E
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace 𝕜 F
        inst✝ : CompleteSpace F
        f : E → F
        p : FormalMultilinearSeries 𝕜 E F
        s : Set E
        x : E
        g : E → F
        r : ENNReal
        hg : HasFPowerSeriesOnBall g p x r
        e : Real
        e0 : GT.gt e 0
        hfg : ∀ ⦃y : E⦄, LT.lt (Dist.dist y x) e → Membership.mem (Insert.insert x s)  …
        ⊢ HasFPowerSeriesOnBall g p x (Min.min r (ENNReal.ofReal e))
      -/
    · exact hg.mono (lt_min hg.r_pos (by positivity)) (min_le_left _ _)
      /-
        🎉 no goals
      -/


/-- `f` is analytic within `s` at `x` iff some local extension of `f` is analytic at `x` -/
lemma analyticWithinAt_iff_exists_analyticAt [CompleteSpace F] {f : E → F} {s : Set E} {x : E} :
    AnalyticWithinAt 𝕜 f s x ↔
      ∃ g, f =ᶠ[𝓝[insert x s] x] g ∧ AnalyticAt 𝕜 g x := by
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : E → F
    s : Set E
    x : E
    ⊢ Iff (AnalyticWithinAt 𝕜 f s x) (Exists fun g => And ((nhdsWithin x (Insert.i …
  -/
  simp only [AnalyticWithinAt, AnalyticAt, hasFPowerSeriesWithinAt_iff_exists_hasFPowerSeriesAt]
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : E → F
    s : Set E
    x : E
    ⊢ Iff (Exists fun p => Exists fun g => And ((nhdsWithin x (Insert.insert x s)) …
  -/
  tauto
  /-
    🎉 no goals
  -/


/-- `f` is analytic within `s` at `x` iff some local extension of `f` is analytic at `x`. In this
version, we make sure that the extension coincides with `f` on all of `insert x s`. -/
lemma analyticWithinAt_iff_exists_analyticAt' [CompleteSpace F] {f : E → F} {s : Set E} {x : E} :
    AnalyticWithinAt 𝕜 f s x ↔
      ∃ g, f x = g x ∧ EqOn f g (insert x s) ∧ AnalyticAt 𝕜 g x := by
  classical
  simp only [analyticWithinAt_iff_exists_analyticAt]
  refine ⟨?_, ?_⟩
  · rintro ⟨g, hf, hg⟩
    rcases mem_nhdsWithin.1 hf with ⟨u, u_open, xu, hu⟩
    let g' := Set.piecewise u g f
    refine ⟨g', ?_, ?_, ?_⟩
    · have : x ∈ u ∩ insert x s := ⟨xu, by simp⟩
      simpa [g', xu, this] using hu this
    · intro y hy
      by_cases h'y : y ∈ u
      · have : y ∈ u ∩ insert x s := ⟨h'y, hy⟩
        simpa [g', h'y, this] using hu this
      · simp [g', h'y]
    · apply hg.congr
      filter_upwards [u_open.mem_nhds xu] with y hy using by simp [g', hy]
  · rintro ⟨g, -, hf, hg⟩
    exact ⟨g, by filter_upwards [self_mem_nhdsWithin] using hf, hg⟩


alias ⟨AnalyticWithinAt.exists_analyticAt, _⟩ := analyticWithinAt_iff_exists_analyticAt'


lemma AnalyticWithinAt.exists_mem_nhdsWithin_analyticOn
    [CompleteSpace F] {f : E → F} {s : Set E} {x : E} (h : AnalyticWithinAt 𝕜 f s x) :
    ∃ u ∈ 𝓝[insert x s] x, AnalyticOn 𝕜 f u := by
  obtain ⟨g, -, h'g, hg⟩ : ∃ g, f x = g x ∧ EqOn f g (insert x s) ∧ AnalyticAt 𝕜 g x :=
    h.exists_analyticAt
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : E → F
    s : Set E
    x : E
    h : AnalyticWithinAt 𝕜 f s x
    g : E → F
    h'g : Set.EqOn f g (Insert.insert x s)
    hg : AnalyticAt 𝕜 g x
    ⊢ Exists fun u => And (Membership.mem (nhdsWithin x (Insert.insert x s)) u) (A …
  -/
  let u := insert x s ∩ {y | AnalyticAt 𝕜 g y}
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : E → F
    s : Set E
    x : E
    h : AnalyticWithinAt 𝕜 f s x
    g : E → F
    h'g : Set.EqOn f g (Insert.insert x s)
    hg : AnalyticAt 𝕜 g x
    u : Set E := Inter.inter (Insert.insert x s) (setOf fun y => AnalyticAt 𝕜 g y)
    ⊢ Exists fun u => And (Membership.mem (nhdsWithin x (Insert.insert x s)) u) (A …
  -/
  refine ⟨u, ?_, ?_⟩
    /-
      case intro.intro.intro.refine_1
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : CompleteSpace F
      f : E → F
      s : Set E
      x : E
      h : AnalyticWithinAt 𝕜 f s x
      g : E → F
      h'g : Set.EqOn f g (Insert.insert x s)
      hg : AnalyticAt 𝕜 g x
      u : Set E := Inter.inter (Insert.insert x s) (setOf fun y => AnalyticAt 𝕜 g y)
      ⊢ Membership.mem (nhdsWithin x (Insert.insert x s)) u
    -/
  · exact inter_mem_nhdsWithin _ ((isOpen_analyticAt 𝕜 g).mem_nhds hg)
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.refine_2
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : CompleteSpace F
      f : E → F
      s : Set E
      x : E
      h : AnalyticWithinAt 𝕜 f s x
      g : E → F
      h'g : Set.EqOn f g (Insert.insert x s)
      hg : AnalyticAt 𝕜 g x
      u : Set E := Inter.inter (Insert.insert x s) (setOf fun y => AnalyticAt 𝕜 g y)
      ⊢ AnalyticOn 𝕜 f u
    -/
  · intro y hy
    /-
      case intro.intro.intro.refine_2
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : CompleteSpace F
      f : E → F
      s : Set E
      x : E
      h : AnalyticWithinAt 𝕜 f s x
      g : E → F
      h'g : Set.EqOn f g (Insert.insert x s)
      hg : AnalyticAt 𝕜 g x
      u : Set E := Inter.inter (Insert.insert x s) (setOf fun y => AnalyticAt 𝕜 g y)
      y : E
      hy : Membership.mem u y
      ⊢ AnalyticWithinAt 𝕜 f u y
    -/
    have : AnalyticWithinAt 𝕜 g u y := hy.2.analyticWithinAt
    /-
      case intro.intro.intro.refine_2
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : CompleteSpace F
      f : E → F
      s : Set E
      x : E
      h : AnalyticWithinAt 𝕜 f s x
      g : E → F
      h'g : Set.EqOn f g (Insert.insert x s)
      hg : AnalyticAt 𝕜 g x
      u : Set E := Inter.inter (Insert.insert x s) (setOf fun y => AnalyticAt 𝕜 g y)
      y : E
      hy : Membership.mem u y
      this : AnalyticWithinAt 𝕜 g u y
      ⊢ AnalyticWithinAt 𝕜 f u y
    -/
    exact this.congr (h'g.mono (inter_subset_left)) (h'g (inter_subset_left hy))
    /-
      🎉 no goals
    -/

