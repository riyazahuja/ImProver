/-- A function which is analytic within a set is strictly differentiable there. Since we
don't have a predicate `HasStrictFDerivWithinAt`, we spell out what it would mean. -/
theorem HasFPowerSeriesWithinAt.hasStrictFDerivWithinAt (h : HasFPowerSeriesWithinAt f p s x) :
    (fun y ↦ f y.1 - f y.2 - (continuousMultilinearCurryFin1 𝕜 E F (p 1)) (y.1 - y.2))
      =o[𝓝[insert x s ×ˢ insert x s] (x, x)] fun y ↦ y.1 - y.2 := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    f : E → F
    x : E
    s : Set E
    h : HasFPowerSeriesWithinAt f p s x
    ⊢ Asymptotics.IsLittleO (nhdsWithin { fst := x, snd := x } (SProd.sprod (Inser …
  -/
  refine h.isBigO_image_sub_norm_mul_norm_sub.trans_isLittleO (IsLittleO.of_norm_right ?_)
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    f : E → F
    x : E
    s : Set E
    h : HasFPowerSeriesWithinAt f p s x
    ⊢ Asymptotics.IsLittleO (nhdsWithin { fst := x, snd := x } (SProd.sprod (Inser …
  -/
  refine isLittleO_iff_exists_eq_mul.2 ⟨fun y => ‖y - (x, x)‖, ?_, EventuallyEq.rfl⟩
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    f : E → F
    x : E
    s : Set E
    h : HasFPowerSeriesWithinAt f p s x
    ⊢ Filter.Tendsto (fun y => Norm.norm (HSub.hSub y { fst := x, snd := x })) (nh …
  -/
  apply Tendsto.mono_left _ nhdsWithin_le_nhds
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    f : E → F
    x : E
    s : Set E
    h : HasFPowerSeriesWithinAt f p s x
    ⊢ Filter.Tendsto (fun y => Norm.norm (HSub.hSub y { fst := x, snd := x })) (nh …
  -/
  refine (continuous_id.sub continuous_const).norm.tendsto' _ _ ?_
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    f : E → F
    x : E
    s : Set E
    h : HasFPowerSeriesWithinAt f p s x
    ⊢ Eq (Norm.norm (HSub.hSub (id { fst := x, snd := x }) { fst := x, snd := x }) …
  -/
  rw [_root_.id, sub_self, norm_zero]
  /-
    🎉 no goals
  -/


theorem HasFPowerSeriesAt.hasStrictFDerivAt (h : HasFPowerSeriesAt f p x) :
    HasStrictFDerivAt f (continuousMultilinearCurryFin1 𝕜 E F (p 1)) x := by
  simpa only [hasStrictFDerivAt_iff_isLittleO, Set.insert_eq_of_mem, Set.mem_univ,
      Set.univ_prod_univ, nhdsWithin_univ]
    using (h.hasFPowerSeriesWithinAt (s := Set.univ)).hasStrictFDerivWithinAt


theorem HasFPowerSeriesWithinAt.hasFDerivWithinAt (h : HasFPowerSeriesWithinAt f p s x) :
    HasFDerivWithinAt f (continuousMultilinearCurryFin1 𝕜 E F (p 1)) (insert x s) x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    f : E → F
    x : E
    s : Set E
    h : HasFPowerSeriesWithinAt f p s x
    ⊢ HasFDerivWithinAt f ((continuousMultilinearCurryFin1 𝕜 E F) (p 1)) (Insert.i …
  -/
  rw [HasFDerivWithinAt, hasFDerivAtFilter_iff_isLittleO, isLittleO_iff]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    f : E → F
    x : E
    s : Set E
    h : HasFPowerSeriesWithinAt f p s x
    ⊢ ∀ ⦃c : Real⦄, LT.lt 0 c → Filter.Eventually (fun x_1 => LE.le (Norm.norm (HS …
  -/
  intro c hc
  have : Tendsto (fun y ↦ (y, x)) (𝓝[insert x s] x) (𝓝[insert x s ×ˢ insert x s] (x, x)) := by
    rw [nhdsWithin_prod_eq]
    exact Tendsto.prod_mk tendsto_id (tendsto_const_nhdsWithin (by simp))
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    f : E → F
    x : E
    s : Set E
    h : HasFPowerSeriesWithinAt f p s x
    c : Real
    hc : LT.lt 0 c
    this : Filter.Tendsto (fun y => { fst := y, snd := x }) (nhdsWithin x (Insert. …
    ⊢ Filter.Eventually (fun x_1 => LE.le (Norm.norm (HSub.hSub (HSub.hSub (f x_1) …
  -/
  exact this (isLittleO_iff.1 h.hasStrictFDerivWithinAt hc)
  /-
    🎉 no goals
  -/


theorem HasFPowerSeriesAt.hasFDerivAt (h : HasFPowerSeriesAt f p x) :
    HasFDerivAt f (continuousMultilinearCurryFin1 𝕜 E F (p 1)) x :=
  h.hasStrictFDerivAt.hasFDerivAt


theorem HasFPowerSeriesWithinAt.differentiableWithinAt (h : HasFPowerSeriesWithinAt f p s x) :
    DifferentiableWithinAt 𝕜 f (insert x s) x :=
  h.hasFDerivWithinAt.differentiableWithinAt


theorem HasFPowerSeriesAt.differentiableAt (h : HasFPowerSeriesAt f p x) : DifferentiableAt 𝕜 f x :=
  h.hasFDerivAt.differentiableAt


theorem AnalyticWithinAt.differentiableWithinAt (h : AnalyticWithinAt 𝕜 f s x) :
    DifferentiableWithinAt 𝕜 f (insert x s) x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    s : Set E
    h : AnalyticWithinAt 𝕜 f s x
    ⊢ DifferentiableWithinAt 𝕜 f (Insert.insert x s) x
  -/
  obtain ⟨p, hp⟩ := h
  /-
    case intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    s : Set E
    p : FormalMultilinearSeries 𝕜 E F
    hp : HasFPowerSeriesWithinAt f p s x
    ⊢ DifferentiableWithinAt 𝕜 f (Insert.insert x s) x
  -/
  exact hp.differentiableWithinAt
  /-
    🎉 no goals
  -/


theorem AnalyticAt.differentiableAt : AnalyticAt 𝕜 f x → DifferentiableAt 𝕜 f x
  | ⟨_, hp⟩ => hp.differentiableAt


theorem AnalyticAt.differentiableWithinAt (h : AnalyticAt 𝕜 f x) : DifferentiableWithinAt 𝕜 f s x :=
  h.differentiableAt.differentiableWithinAt


theorem HasFPowerSeriesWithinAt.fderivWithin_eq
    (h : HasFPowerSeriesWithinAt f p s x) (hu : UniqueDiffWithinAt 𝕜 (insert x s) x) :
    fderivWithin 𝕜 f (insert x s) x = continuousMultilinearCurryFin1 𝕜 E F (p 1) :=
  h.hasFDerivWithinAt.fderivWithin hu


theorem HasFPowerSeriesAt.fderiv_eq (h : HasFPowerSeriesAt f p x) :
    fderiv 𝕜 f x = continuousMultilinearCurryFin1 𝕜 E F (p 1) :=
  h.hasFDerivAt.fderiv


theorem AnalyticAt.hasStrictFDerivAt (h : AnalyticAt 𝕜 f x) :
    HasStrictFDerivAt f (fderiv 𝕜 f x) x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    h : AnalyticAt 𝕜 f x
    ⊢ HasStrictFDerivAt f (fderiv 𝕜 f x) x
  -/
  rcases h with ⟨p, hp⟩
  /-
    case intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    p : FormalMultilinearSeries 𝕜 E F
    hp : HasFPowerSeriesAt f p x
    ⊢ HasStrictFDerivAt f (fderiv 𝕜 f x) x
  -/
  rw [hp.fderiv_eq]
  /-
    case intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    p : FormalMultilinearSeries 𝕜 E F
    hp : HasFPowerSeriesAt f p x
    ⊢ HasStrictFDerivAt f ((continuousMultilinearCurryFin1 𝕜 E F) (p 1)) x
  -/
  exact hp.hasStrictFDerivAt
  /-
    🎉 no goals
  -/


theorem HasFPowerSeriesWithinOnBall.differentiableOn [CompleteSpace F]
    (h : HasFPowerSeriesWithinOnBall f p s x r) :
    DifferentiableOn 𝕜 f (insert x s ∩ EMetric.ball x r) := by
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    r : ENNReal
    f : E → F
    x : E
    s : Set E
    inst✝ : CompleteSpace F
    h : HasFPowerSeriesWithinOnBall f p s x r
    ⊢ DifferentiableOn 𝕜 f (Inter.inter (Insert.insert x s) (EMetric.ball x r))
  -/
  intro y hy
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    r : ENNReal
    f : E → F
    x : E
    s : Set E
    inst✝ : CompleteSpace F
    h : HasFPowerSeriesWithinOnBall f p s x r
    y : E
    hy : Membership.mem (Inter.inter (Insert.insert x s) (EMetric.ball x r)) y
    ⊢ DifferentiableWithinAt 𝕜 f (Inter.inter (Insert.insert x s) (EMetric.ball x  …
  -/
  have Z := (h.analyticWithinAt_of_mem hy).differentiableWithinAt
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    r : ENNReal
    f : E → F
    x : E
    s : Set E
    inst✝ : CompleteSpace F
    h : HasFPowerSeriesWithinOnBall f p s x r
    y : E
    hy : Membership.mem (Inter.inter (Insert.insert x s) (EMetric.ball x r)) y
    Z : DifferentiableWithinAt 𝕜 f (Insert.insert y s) y
    ⊢ DifferentiableWithinAt 𝕜 f (Inter.inter (Insert.insert x s) (EMetric.ball x  …
  -/
  rcases eq_or_ne y x with rfl | hy
    /-
      case inl
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type v
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      r : ENNReal
      f : E → F
      s : Set E
      inst✝ : CompleteSpace F
      y : E
      Z : DifferentiableWithinAt 𝕜 f (Insert.insert y s) y
      h : HasFPowerSeriesWithinOnBall f p s y r
      hy : Membership.mem (Inter.inter (Insert.insert y s) (EMetric.ball y r)) y
      ⊢ DifferentiableWithinAt 𝕜 f (Inter.inter (Insert.insert y s) (EMetric.ball y  …
    -/
  · exact Z.mono inter_subset_left
    /-
      🎉 no goals
    -/
    /-
      case inr
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type v
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      r : ENNReal
      f : E → F
      x : E
      s : Set E
      inst✝ : CompleteSpace F
      h : HasFPowerSeriesWithinOnBall f p s x r
      y : E
      hy✝ : Membership.mem (Inter.inter (Insert.insert x s) (EMetric.ball x r)) y
      Z : DifferentiableWithinAt 𝕜 f (Insert.insert y s) y
      hy : Ne y x
      ⊢ DifferentiableWithinAt 𝕜 f (Inter.inter (Insert.insert x s) (EMetric.ball x  …
    -/
  · apply (Z.mono (subset_insert _ _)).mono_of_mem_nhdsWithin
    /-
      case inr
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type v
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      r : ENNReal
      f : E → F
      x : E
      s : Set E
      inst✝ : CompleteSpace F
      h : HasFPowerSeriesWithinOnBall f p s x r
      y : E
      hy✝ : Membership.mem (Inter.inter (Insert.insert x s) (EMetric.ball x r)) y
      Z : DifferentiableWithinAt 𝕜 f (Insert.insert y s) y
      hy : Ne y x
      ⊢ Membership.mem (nhdsWithin y (Inter.inter (Insert.insert x s) (EMetric.ball  …
    -/
    suffices s ∈ 𝓝[insert x s] y from nhdsWithin_mono _ inter_subset_left this
    /-
      case inr
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type v
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      r : ENNReal
      f : E → F
      x : E
      s : Set E
      inst✝ : CompleteSpace F
      h : HasFPowerSeriesWithinOnBall f p s x r
      y : E
      hy✝ : Membership.mem (Inter.inter (Insert.insert x s) (EMetric.ball x r)) y
      Z : DifferentiableWithinAt 𝕜 f (Insert.insert y s) y
      hy : Ne y x
      ⊢ Membership.mem (nhdsWithin y (Insert.insert x s)) s
    -/
    rw [nhdsWithin_insert_of_ne hy]
    /-
      case inr
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type v
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      r : ENNReal
      f : E → F
      x : E
      s : Set E
      inst✝ : CompleteSpace F
      h : HasFPowerSeriesWithinOnBall f p s x r
      y : E
      hy✝ : Membership.mem (Inter.inter (Insert.insert x s) (EMetric.ball x r)) y
      Z : DifferentiableWithinAt 𝕜 f (Insert.insert y s) y
      hy : Ne y x
      ⊢ Membership.mem (nhdsWithin y s) s
    -/
    exact self_mem_nhdsWithin
    /-
      🎉 no goals
    -/


theorem HasFPowerSeriesOnBall.differentiableOn [CompleteSpace F]
    (h : HasFPowerSeriesOnBall f p x r) : DifferentiableOn 𝕜 f (EMetric.ball x r) := fun _ hy =>
  (h.analyticAt_of_mem hy).differentiableWithinAt


theorem AnalyticOn.differentiableOn (h : AnalyticOn 𝕜 f s) : DifferentiableOn 𝕜 f s :=
                                                      /-
                                                        𝕜 : Type u_1
                                                        inst✝⁴ : NontriviallyNormedField 𝕜
                                                        E : Type u
                                                        inst✝³ : NormedAddCommGroup E
                                                        inst✝² : NormedSpace 𝕜 E
                                                        F : Type v
                                                        inst✝¹ : NormedAddCommGroup F
                                                        inst✝ : NormedSpace 𝕜 F
                                                        f : E → F
                                                        s : Set E
                                                        h : AnalyticOn 𝕜 f s
                                                        y : E
                                                        hy : Membership.mem s y
                                                        ⊢ HasSubset.Subset s (Insert.insert y s)
                                                      -/
  fun y hy ↦ (h y hy).differentiableWithinAt.mono (by simp)
                                                      /-
                                                        🎉 no goals
                                                      -/


theorem AnalyticOnNhd.differentiableOn (h : AnalyticOnNhd 𝕜 f s) : DifferentiableOn 𝕜 f s :=
  fun y hy ↦ (h y hy).differentiableWithinAt


theorem HasFPowerSeriesWithinOnBall.hasFDerivWithinAt [CompleteSpace F]
    (h : HasFPowerSeriesWithinOnBall f p s x r)
    {y : E} (hy : (‖y‖₊ : ℝ≥0∞) < r) (h'y : x + y ∈ insert x s) :
    HasFDerivWithinAt f (continuousMultilinearCurryFin1 𝕜 E F (p.changeOrigin y 1))
      (insert x s) (x + y) := by
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    r : ENNReal
    f : E → F
    x : E
    s : Set E
    inst✝ : CompleteSpace F
    h : HasFPowerSeriesWithinOnBall f p s x r
    y : E
    hy : LT.lt (↑(NNNorm.nnnorm y)) r
    h'y : Membership.mem (Insert.insert x s) (HAdd.hAdd x y)
    ⊢ HasFDerivWithinAt f ((continuousMultilinearCurryFin1 𝕜 E F) (p.changeOrigin  …
  -/
  rcases eq_or_ne y 0 with rfl | h''y
    /-
      case inl
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type v
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      r : ENNReal
      f : E → F
      x : E
      s : Set E
      inst✝ : CompleteSpace F
      h : HasFPowerSeriesWithinOnBall f p s x r
      hy : LT.lt (↑(NNNorm.nnnorm 0)) r
      h'y : Membership.mem (Insert.insert x s) (HAdd.hAdd x 0)
      ⊢ HasFDerivWithinAt f ((continuousMultilinearCurryFin1 𝕜 E F) (p.changeOrigin  …
    -/
  · convert (h.changeOrigin hy h'y).hasFPowerSeriesWithinAt.hasFDerivWithinAt
    /-
      case h.e'_13.h.e'_4
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type v
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      r : ENNReal
      f : E → F
      x : E
      s : Set E
      inst✝ : CompleteSpace F
      h : HasFPowerSeriesWithinOnBall f p s x r
      hy : LT.lt (↑(NNNorm.nnnorm 0)) r
      h'y : Membership.mem (Insert.insert x s) (HAdd.hAdd x 0)
      ⊢ Eq x (HAdd.hAdd x 0)
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type v
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      r : ENNReal
      f : E → F
      x : E
      s : Set E
      inst✝ : CompleteSpace F
      h : HasFPowerSeriesWithinOnBall f p s x r
      y : E
      hy : LT.lt (↑(NNNorm.nnnorm y)) r
      h'y : Membership.mem (Insert.insert x s) (HAdd.hAdd x y)
      h''y : Ne y 0
      ⊢ HasFDerivWithinAt f ((continuousMultilinearCurryFin1 𝕜 E F) (p.changeOrigin  …
    -/
  · have Z := (h.changeOrigin hy h'y).hasFPowerSeriesWithinAt.hasFDerivWithinAt
    /-
      case inr
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type v
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      r : ENNReal
      f : E → F
      x : E
      s : Set E
      inst✝ : CompleteSpace F
      h : HasFPowerSeriesWithinOnBall f p s x r
      y : E
      hy : LT.lt (↑(NNNorm.nnnorm y)) r
      h'y : Membership.mem (Insert.insert x s) (HAdd.hAdd x y)
      h''y : Ne y 0
      Z : HasFDerivWithinAt f ((continuousMultilinearCurryFin1 𝕜 E F) (p.changeOrigi …
      ⊢ HasFDerivWithinAt f ((continuousMultilinearCurryFin1 𝕜 E F) (p.changeOrigin  …
    -/
    apply (Z.mono (subset_insert _ _)).mono_of_mem_nhdsWithin
    /-
      case inr
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type v
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      r : ENNReal
      f : E → F
      x : E
      s : Set E
      inst✝ : CompleteSpace F
      h : HasFPowerSeriesWithinOnBall f p s x r
      y : E
      hy : LT.lt (↑(NNNorm.nnnorm y)) r
      h'y : Membership.mem (Insert.insert x s) (HAdd.hAdd x y)
      h''y : Ne y 0
      Z : HasFDerivWithinAt f ((continuousMultilinearCurryFin1 𝕜 E F) (p.changeOrigi …
      ⊢ Membership.mem (nhdsWithin (HAdd.hAdd x y) (Insert.insert x s)) s
    -/
    rw [nhdsWithin_insert_of_ne]
      /-
        case inr
        𝕜 : Type u_1
        inst✝⁵ : NontriviallyNormedField 𝕜
        E : Type u
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace 𝕜 E
        F : Type v
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace 𝕜 F
        p : FormalMultilinearSeries 𝕜 E F
        r : ENNReal
        f : E → F
        x : E
        s : Set E
        inst✝ : CompleteSpace F
        h : HasFPowerSeriesWithinOnBall f p s x r
        y : E
        hy : LT.lt (↑(NNNorm.nnnorm y)) r
        h'y : Membership.mem (Insert.insert x s) (HAdd.hAdd x y)
        h''y : Ne y 0
        Z : HasFDerivWithinAt f ((continuousMultilinearCurryFin1 𝕜 E F) (p.changeOrigi …
        ⊢ Membership.mem (nhdsWithin (HAdd.hAdd x y) s) s
      -/
    · exact self_mem_nhdsWithin
      /-
        🎉 no goals
      -/
      /-
        case inr
        𝕜 : Type u_1
        inst✝⁵ : NontriviallyNormedField 𝕜
        E : Type u
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace 𝕜 E
        F : Type v
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace 𝕜 F
        p : FormalMultilinearSeries 𝕜 E F
        r : ENNReal
        f : E → F
        x : E
        s : Set E
        inst✝ : CompleteSpace F
        h : HasFPowerSeriesWithinOnBall f p s x r
        y : E
        hy : LT.lt (↑(NNNorm.nnnorm y)) r
        h'y : Membership.mem (Insert.insert x s) (HAdd.hAdd x y)
        h''y : Ne y 0
        Z : HasFDerivWithinAt f ((continuousMultilinearCurryFin1 𝕜 E F) (p.changeOrigi …
        ⊢ Ne (HAdd.hAdd x y) x
      -/
    · simpa using h''y
      /-
        🎉 no goals
      -/


theorem HasFPowerSeriesOnBall.hasFDerivAt [CompleteSpace F] (h : HasFPowerSeriesOnBall f p x r)
    {y : E} (hy : (‖y‖₊ : ℝ≥0∞) < r) :
    HasFDerivAt f (continuousMultilinearCurryFin1 𝕜 E F (p.changeOrigin y 1)) (x + y) :=
  (h.changeOrigin hy).hasFPowerSeriesAt.hasFDerivAt


theorem HasFPowerSeriesWithinOnBall.fderivWithin_eq [CompleteSpace F]
    (h : HasFPowerSeriesWithinOnBall f p s x r)
    {y : E} (hy : (‖y‖₊ : ℝ≥0∞) < r) (h'y : x + y ∈ insert x s) (hu : UniqueDiffOn 𝕜 (insert x s)) :
    fderivWithin 𝕜 f (insert x s) (x + y) =
      continuousMultilinearCurryFin1 𝕜 E F (p.changeOrigin y 1) :=
  (h.hasFDerivWithinAt hy h'y).fderivWithin (hu _ h'y)


theorem HasFPowerSeriesOnBall.fderiv_eq [CompleteSpace F] (h : HasFPowerSeriesOnBall f p x r)
    {y : E} (hy : (‖y‖₊ : ℝ≥0∞) < r) :
    fderiv 𝕜 f (x + y) = continuousMultilinearCurryFin1 𝕜 E F (p.changeOrigin y 1) :=
  (h.hasFDerivAt hy).fderiv


/-- If a function has a power series on a ball, then so does its derivative. -/
protected theorem HasFPowerSeriesOnBall.fderiv [CompleteSpace F]
    (h : HasFPowerSeriesOnBall f p x r) :
    HasFPowerSeriesOnBall (fderiv 𝕜 f) p.derivSeries x r := by
  refine .congr (f := fun z ↦ continuousMultilinearCurryFin1 𝕜 E F (p.changeOrigin (z - x) 1)) ?_
    fun z hz ↦ ?_
  · refine continuousMultilinearCurryFin1 𝕜 E F
      |>.toContinuousLinearEquiv.toContinuousLinearMap.comp_hasFPowerSeriesOnBall ?_
    simpa using ((p.hasFPowerSeriesOnBall_changeOrigin 1
      (h.r_pos.trans_le h.r_le)).mono h.r_pos h.r_le).comp_sub x
  /-
    case refine_2
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    r : ENNReal
    f : E → F
    x : E
    inst✝ : CompleteSpace F
    h : HasFPowerSeriesOnBall f p x r
    z : E
    hz : Membership.mem (EMetric.ball x r) z
    ⊢ Eq ((fun z => (continuousMultilinearCurryFin1 𝕜 E F) (p.changeOrigin (HSub.h …
  -/
  dsimp only
  /-
    case refine_2
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    r : ENNReal
    f : E → F
    x : E
    inst✝ : CompleteSpace F
    h : HasFPowerSeriesOnBall f p x r
    z : E
    hz : Membership.mem (EMetric.ball x r) z
    ⊢ Eq ((continuousMultilinearCurryFin1 𝕜 E F) (p.changeOrigin (HSub.hSub z x) 1 …
  -/
  rw [← h.fderiv_eq, add_sub_cancel]
  /-
    case refine_2
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    r : ENNReal
    f : E → F
    x : E
    inst✝ : CompleteSpace F
    h : HasFPowerSeriesOnBall f p x r
    z : E
    hz : Membership.mem (EMetric.ball x r) z
    ⊢ LT.lt (↑(NNNorm.nnnorm (HSub.hSub z x))) r
  -/
  simpa only [edist_eq_coe_nnnorm_sub, EMetric.mem_ball] using hz
  /-
    🎉 no goals
  -/


/-- If a function has a power series within a set on a ball, then so does its derivative. -/
protected theorem HasFPowerSeriesWithinOnBall.fderivWithin [CompleteSpace F]
    (h : HasFPowerSeriesWithinOnBall f p s x r) (hu : UniqueDiffOn 𝕜 (insert x s)) :
    HasFPowerSeriesWithinOnBall (fderivWithin 𝕜 f (insert x s)) p.derivSeries s x r := by
  refine .congr' (f := fun z ↦ continuousMultilinearCurryFin1 𝕜 E F (p.changeOrigin (z - x) 1)) ?_
    (fun z hz ↦ ?_)
  · refine continuousMultilinearCurryFin1 𝕜 E F
      |>.toContinuousLinearEquiv.toContinuousLinearMap.comp_hasFPowerSeriesWithinOnBall ?_
    /-
      case refine_1
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type v
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      r : ENNReal
      f : E → F
      x : E
      s : Set E
      inst✝ : CompleteSpace F
      h : HasFPowerSeriesWithinOnBall f p s x r
      hu : UniqueDiffOn 𝕜 (Insert.insert x s)
      ⊢ HasFPowerSeriesWithinOnBall (fun z => p.changeOrigin (HSub.hSub z x) 1) (p.c …
    -/
    apply HasFPowerSeriesOnBall.hasFPowerSeriesWithinOnBall
    simpa using ((p.hasFPowerSeriesOnBall_changeOrigin 1
      (h.r_pos.trans_le h.r_le)).mono h.r_pos h.r_le).comp_sub x
    /-
      case refine_2
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type v
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      r : ENNReal
      f : E → F
      x : E
      s : Set E
      inst✝ : CompleteSpace F
      h : HasFPowerSeriesWithinOnBall f p s x r
      hu : UniqueDiffOn 𝕜 (Insert.insert x s)
      z : E
      hz : Membership.mem (Inter.inter (Insert.insert x s) (EMetric.ball x r)) z
      ⊢ Eq (fderivWithin 𝕜 f (Insert.insert x s) z) ((fun z => (continuousMultilinea …
    -/
  · dsimp only
    /-
      case refine_2
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type v
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      r : ENNReal
      f : E → F
      x : E
      s : Set E
      inst✝ : CompleteSpace F
      h : HasFPowerSeriesWithinOnBall f p s x r
      hu : UniqueDiffOn 𝕜 (Insert.insert x s)
      z : E
      hz : Membership.mem (Inter.inter (Insert.insert x s) (EMetric.ball x r)) z
      ⊢ Eq (fderivWithin 𝕜 f (Insert.insert x s) z) ((continuousMultilinearCurryFin1 …
    -/
    rw [← h.fderivWithin_eq _ _ hu, add_sub_cancel]
      /-
        𝕜 : Type u_1
        inst✝⁵ : NontriviallyNormedField 𝕜
        E : Type u
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace 𝕜 E
        F : Type v
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace 𝕜 F
        p : FormalMultilinearSeries 𝕜 E F
        r : ENNReal
        f : E → F
        x : E
        s : Set E
        inst✝ : CompleteSpace F
        h : HasFPowerSeriesWithinOnBall f p s x r
        hu : UniqueDiffOn 𝕜 (Insert.insert x s)
        z : E
        hz : Membership.mem (Inter.inter (Insert.insert x s) (EMetric.ball x r)) z
        ⊢ LT.lt (↑(NNNorm.nnnorm (HSub.hSub z x))) r
      -/
    · simpa only [edist_eq_coe_nnnorm_sub, EMetric.mem_ball] using hz.2
      /-
        🎉 no goals
      -/
      /-
        𝕜 : Type u_1
        inst✝⁵ : NontriviallyNormedField 𝕜
        E : Type u
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace 𝕜 E
        F : Type v
        inst✝² : NormedAddCommGroup F
        inst✝¹ : NormedSpace 𝕜 F
        p : FormalMultilinearSeries 𝕜 E F
        r : ENNReal
        f : E → F
        x : E
        s : Set E
        inst✝ : CompleteSpace F
        h : HasFPowerSeriesWithinOnBall f p s x r
        hu : UniqueDiffOn 𝕜 (Insert.insert x s)
        z : E
        hz : Membership.mem (Inter.inter (Insert.insert x s) (EMetric.ball x r)) z
        ⊢ Membership.mem (Insert.insert x s) (HAdd.hAdd x (HSub.hSub z x))
      -/
    · simpa using hz.1
      /-
        🎉 no goals
      -/


/-- If a function has a power series within a set on a ball, then so does its derivative. For a
version without completeness, but assuming that the function is analytic on the set `s`, see
`HasFPowerSeriesWithinOnBall.fderivWithin_of_mem_of_analyticOn`. -/
protected theorem HasFPowerSeriesWithinOnBall.fderivWithin_of_mem [CompleteSpace F]
    (h : HasFPowerSeriesWithinOnBall f p s x r) (hu : UniqueDiffOn 𝕜 s) (hx : x ∈ s) :
    HasFPowerSeriesWithinOnBall (fderivWithin 𝕜 f s) p.derivSeries s x r := by
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    r : ENNReal
    f : E → F
    x : E
    s : Set E
    inst✝ : CompleteSpace F
    h : HasFPowerSeriesWithinOnBall f p s x r
    hu : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    ⊢ HasFPowerSeriesWithinOnBall (fderivWithin 𝕜 f s) p.derivSeries s x r
  -/
  have : insert x s = s := insert_eq_of_mem hx
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    r : ENNReal
    f : E → F
    x : E
    s : Set E
    inst✝ : CompleteSpace F
    h : HasFPowerSeriesWithinOnBall f p s x r
    hu : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    this : Eq (Insert.insert x s) s
    ⊢ HasFPowerSeriesWithinOnBall (fderivWithin 𝕜 f s) p.derivSeries s x r
  -/
  rw [← this] at hu
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    r : ENNReal
    f : E → F
    x : E
    s : Set E
    inst✝ : CompleteSpace F
    h : HasFPowerSeriesWithinOnBall f p s x r
    hu : UniqueDiffOn 𝕜 (Insert.insert x s)
    hx : Membership.mem s x
    this : Eq (Insert.insert x s) s
    ⊢ HasFPowerSeriesWithinOnBall (fderivWithin 𝕜 f s) p.derivSeries s x r
  -/
  convert h.fderivWithin hu
  /-
    case h.e'_9.h.e'_12
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    r : ENNReal
    f : E → F
    x : E
    s : Set E
    inst✝ : CompleteSpace F
    h : HasFPowerSeriesWithinOnBall f p s x r
    hu : UniqueDiffOn 𝕜 (Insert.insert x s)
    hx : Membership.mem s x
    this : Eq (Insert.insert x s) s
    ⊢ Eq s (Insert.insert x s)
  -/
  exact this.symm
  /-
    🎉 no goals
  -/


/-- If a function is analytic on a set `s`, so is its Fréchet derivative. -/
protected theorem AnalyticAt.fderiv [CompleteSpace F] (h : AnalyticAt 𝕜 f x) :
    AnalyticAt 𝕜 (fderiv 𝕜 f) x := by
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    f : E → F
    x : E
    inst✝ : CompleteSpace F
    h : AnalyticAt 𝕜 f x
    ⊢ AnalyticAt 𝕜 (fderiv 𝕜 f) x
  -/
  rcases h with ⟨p, r, hp⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    f : E → F
    x : E
    inst✝ : CompleteSpace F
    p : FormalMultilinearSeries 𝕜 E F
    r : ENNReal
    hp : HasFPowerSeriesOnBall f p x r
    ⊢ AnalyticAt 𝕜 (fderiv 𝕜 f) x
  -/
  exact hp.fderiv.analyticAt
  /-
    🎉 no goals
  -/


/-- If a function is analytic on a set `s`, so is its Fréchet derivative. See also
`AnalyticOnNhd.fderiv_of_isOpen`, removing the completeness assumption but requiring the set
to be open. -/
protected theorem AnalyticOnNhd.fderiv [CompleteSpace F] (h : AnalyticOnNhd 𝕜 f s) :
    AnalyticOnNhd 𝕜 (fderiv 𝕜 f) s :=
  fun y hy ↦ AnalyticAt.fderiv (h y hy)


@[deprecated (since := "2024-09-26")]
alias AnalyticOn.fderiv := AnalyticOnNhd.fderiv


/-- If a function is analytic on a set `s`, so are its successive Fréchet derivative. See also
`AnalyticOnNhd.iteratedFDeriv_of_isOpen`, removing the completeness assumption but requiring the set
to be open.-/
protected theorem AnalyticOnNhd.iteratedFDeriv [CompleteSpace F] (h : AnalyticOnNhd 𝕜 f s) (n : ℕ) :
    AnalyticOnNhd 𝕜 (iteratedFDeriv 𝕜 n f) s := by
  induction n with
  | zero =>
    rw [iteratedFDeriv_zero_eq_comp]
    exact ((continuousMultilinearCurryFin0 𝕜 E F).symm : F →L[𝕜] E[×0]→L[𝕜] F).comp_analyticOnNhd h
  | succ n IH =>
    rw [iteratedFDeriv_succ_eq_comp_left]
    -- Porting note: for reasons that I do not understand at all, `?g` cannot be inlined.
    convert ContinuousLinearMap.comp_analyticOnNhd ?g IH.fderiv
    case g => exact ↑(continuousMultilinearCurryLeftEquiv 𝕜 (fun _ : Fin (n + 1) ↦ E) F).symm
    simp


@[deprecated (since := "2024-09-26")]
protected alias AnalyticOn.iteratedFDeriv := AnalyticOnNhd.iteratedFDeriv


/-- If a function is analytic on a neighborhood of a set `s`, then it has a Taylor series given
by the sequence of its derivatives. Note that, if the function were just analytic on `s`, then
one would have to use instead the sequence of derivatives inside the set, as in
`AnalyticOn.hasFTaylorSeriesUpToOn`. -/
lemma AnalyticOnNhd.hasFTaylorSeriesUpToOn [CompleteSpace F]
    (n : WithTop ℕ∞) (h : AnalyticOnNhd 𝕜 f s) :
    HasFTaylorSeriesUpToOn n f (ftaylorSeries 𝕜 f) s := by
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    f : E → F
    s : Set E
    inst✝ : CompleteSpace F
    n : WithTop ENat
    h : AnalyticOnNhd 𝕜 f s
    ⊢ HasFTaylorSeriesUpToOn n f (ftaylorSeries 𝕜 f) s
  -/
  refine ⟨fun x _hx ↦ rfl, fun m _hm x hx ↦ ?_, fun m _hm x hx ↦ ?_⟩
    /-
      case refine_1
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type v
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      f : E → F
      s : Set E
      inst✝ : CompleteSpace F
      n : WithTop ENat
      h : AnalyticOnNhd 𝕜 f s
      m : Nat
      _hm : LT.lt (↑m) n
      x : E
      hx : Membership.mem s x
      ⊢ HasFDerivWithinAt (fun x => ftaylorSeries 𝕜 f x m) (ftaylorSeries 𝕜 f x m.su …
    -/
  · apply HasFDerivAt.hasFDerivWithinAt
    /-
      case refine_1.h
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type v
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      f : E → F
      s : Set E
      inst✝ : CompleteSpace F
      n : WithTop ENat
      h : AnalyticOnNhd 𝕜 f s
      m : Nat
      _hm : LT.lt (↑m) n
      x : E
      hx : Membership.mem s x
      ⊢ HasFDerivAt (fun x => ftaylorSeries 𝕜 f x m) (ftaylorSeries 𝕜 f x m.succ).cu …
    -/
    exact ((h.iteratedFDeriv m x hx).differentiableAt).hasFDerivAt
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type v
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      f : E → F
      s : Set E
      inst✝ : CompleteSpace F
      n : WithTop ENat
      h : AnalyticOnNhd 𝕜 f s
      m : Nat
      _hm : LE.le (↑m) n
      x : E
      hx : Membership.mem s x
      ⊢ ContinuousWithinAt (fun x => ftaylorSeries 𝕜 f x m) s x
    -/
  · apply (DifferentiableAt.continuousAt (𝕜 := 𝕜) ?_).continuousWithinAt
    /-
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type v
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      f : E → F
      s : Set E
      inst✝ : CompleteSpace F
      n : WithTop ENat
      h : AnalyticOnNhd 𝕜 f s
      m : Nat
      _hm : LE.le (↑m) n
      x : E
      hx : Membership.mem s x
      ⊢ DifferentiableAt 𝕜 (fun x => ftaylorSeries 𝕜 f x m) x
    -/
    exact (h.iteratedFDeriv m x hx).differentiableAt
    /-
      🎉 no goals
    -/


lemma AnalyticWithinAt.exists_hasFTaylorSeriesUpToOn [CompleteSpace F]
    (n : WithTop ℕ∞) (h : AnalyticWithinAt 𝕜 f s x) :
    ∃ u ∈ 𝓝[insert x s] x, ∃ (p : E → FormalMultilinearSeries 𝕜 E F),
    HasFTaylorSeriesUpToOn n f p u ∧ ∀ i, AnalyticOn 𝕜 (fun x ↦ p x i) u := by
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    f : E → F
    x : E
    s : Set E
    inst✝ : CompleteSpace F
    n : WithTop ENat
    h : AnalyticWithinAt 𝕜 f s x
    ⊢ Exists fun u => And (Membership.mem (nhdsWithin x (Insert.insert x s)) u) (E …
  -/
  rcases h.exists_analyticAt with ⟨g, -, fg, hg⟩
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    f : E → F
    x : E
    s : Set E
    inst✝ : CompleteSpace F
    n : WithTop ENat
    h : AnalyticWithinAt 𝕜 f s x
    g : E → F
    fg : Set.EqOn f g (Insert.insert x s)
    hg : AnalyticAt 𝕜 g x
    ⊢ Exists fun u => And (Membership.mem (nhdsWithin x (Insert.insert x s)) u) (E …
  -/
  rcases hg.exists_mem_nhds_analyticOnNhd with ⟨v, vx, hv⟩
  /-
    case intro.intro.intro.intro.intro
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    f : E → F
    x : E
    s : Set E
    inst✝ : CompleteSpace F
    n : WithTop ENat
    h : AnalyticWithinAt 𝕜 f s x
    g : E → F
    fg : Set.EqOn f g (Insert.insert x s)
    hg : AnalyticAt 𝕜 g x
    v : Set E
    vx : Membership.mem (nhds x) v
    hv : AnalyticOnNhd 𝕜 g v
    ⊢ Exists fun u => And (Membership.mem (nhdsWithin x (Insert.insert x s)) u) (E …
  -/
  refine ⟨insert x s ∩ v, inter_mem_nhdsWithin _ vx, ftaylorSeries 𝕜 g, ?_, fun i ↦ ?_⟩
  · suffices HasFTaylorSeriesUpToOn n g (ftaylorSeries 𝕜 g) (insert x s ∩ v) from
      this.congr (fun y hy ↦ fg hy.1)
    /-
      case intro.intro.intro.intro.intro.refine_1
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type v
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      f : E → F
      x : E
      s : Set E
      inst✝ : CompleteSpace F
      n : WithTop ENat
      h : AnalyticWithinAt 𝕜 f s x
      g : E → F
      fg : Set.EqOn f g (Insert.insert x s)
      hg : AnalyticAt 𝕜 g x
      v : Set E
      vx : Membership.mem (nhds x) v
      hv : AnalyticOnNhd 𝕜 g v
      ⊢ HasFTaylorSeriesUpToOn n g (ftaylorSeries 𝕜 g) (Inter.inter (Insert.insert x …
    -/
    exact AnalyticOnNhd.hasFTaylorSeriesUpToOn _ (hv.mono Set.inter_subset_right)
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.refine_2
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type v
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      f : E → F
      x : E
      s : Set E
      inst✝ : CompleteSpace F
      n : WithTop ENat
      h : AnalyticWithinAt 𝕜 f s x
      g : E → F
      fg : Set.EqOn f g (Insert.insert x s)
      hg : AnalyticAt 𝕜 g x
      v : Set E
      vx : Membership.mem (nhds x) v
      hv : AnalyticOnNhd 𝕜 g v
      i : Nat
      ⊢ AnalyticOn 𝕜 (fun x => ftaylorSeries 𝕜 g x i) (Inter.inter (Insert.insert x  …
    -/
  · exact (hv.iteratedFDeriv i).analyticOn.mono Set.inter_subset_right
    /-
      🎉 no goals
    -/


/-- If a function has a power series `p` within a set of unique differentiability, inside a ball,
and is differentiable at a point, then the derivative series of `p` is summable at a point, with
sum the given differential. Note that this theorem does not require completeness of the space.-/
theorem HasFPowerSeriesWithinOnBall.hasSum_derivSeries_of_hasFDerivWithinAt
    (h : HasFPowerSeriesWithinOnBall f p s x r)
    {f' : E →L[𝕜] F}
    {y : E} (hy : (‖y‖₊ : ℝ≥0∞) < r) (h'y : x + y ∈ insert x s)
    (hf' : HasFDerivWithinAt f f' (insert x s) (x + y))
    (hu : UniqueDiffOn 𝕜 (insert x s)) :
    HasSum (fun n ↦ p.derivSeries n (fun _ ↦ y)) f' := by
  /- In the completion of the space, the derivative series is summable, and its sum is a derivative
  of the function. Therefore, by uniqueness of derivatives, its sum is the image of `f'` under
  the canonical embedding. As this is an embedding, it means that there was also convergence in
  the original space, to `f'`. -/
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    r : ENNReal
    f : E → F
    x : E
    s : Set E
    h : HasFPowerSeriesWithinOnBall f p s x r
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    y : E
    hy : LT.lt (↑(NNNorm.nnnorm y)) r
    h'y : Membership.mem (Insert.insert x s) (HAdd.hAdd x y)
    hf' : HasFDerivWithinAt f f' (Insert.insert x s) (HAdd.hAdd x y)
    hu : UniqueDiffOn 𝕜 (Insert.insert x s)
    ⊢ HasSum (fun n => (p.derivSeries n) fun x => y) f'
  -/
  let F' := UniformSpace.Completion F
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    r : ENNReal
    f : E → F
    x : E
    s : Set E
    h : HasFPowerSeriesWithinOnBall f p s x r
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    y : E
    hy : LT.lt (↑(NNNorm.nnnorm y)) r
    h'y : Membership.mem (Insert.insert x s) (HAdd.hAdd x y)
    hf' : HasFDerivWithinAt f f' (Insert.insert x s) (HAdd.hAdd x y)
    hu : UniqueDiffOn 𝕜 (Insert.insert x s)
    F' : Type v := UniformSpace.Completion F
    ⊢ HasSum (fun n => (p.derivSeries n) fun x => y) f'
  -/
  let a : F →L[𝕜] F' := UniformSpace.Completion.toComplL
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    r : ENNReal
    f : E → F
    x : E
    s : Set E
    h : HasFPowerSeriesWithinOnBall f p s x r
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    y : E
    hy : LT.lt (↑(NNNorm.nnnorm y)) r
    h'y : Membership.mem (Insert.insert x s) (HAdd.hAdd x y)
    hf' : HasFDerivWithinAt f f' (Insert.insert x s) (HAdd.hAdd x y)
    hu : UniqueDiffOn 𝕜 (Insert.insert x s)
    F' : Type v := UniformSpace.Completion F
    a : ContinuousLinearMap (RingHom.id 𝕜) F F' := UniformSpace.Completion.toComplL
    ⊢ HasSum (fun n => (p.derivSeries n) fun x => y) f'
  -/
  let b : (E →L[𝕜] F) →ₗᵢ[𝕜] (E →L[𝕜] F') := UniformSpace.Completion.toComplₗᵢ.postcomp
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    r : ENNReal
    f : E → F
    x : E
    s : Set E
    h : HasFPowerSeriesWithinOnBall f p s x r
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    y : E
    hy : LT.lt (↑(NNNorm.nnnorm y)) r
    h'y : Membership.mem (Insert.insert x s) (HAdd.hAdd x y)
    hf' : HasFDerivWithinAt f f' (Insert.insert x s) (HAdd.hAdd x y)
    hu : UniqueDiffOn 𝕜 (Insert.insert x s)
    F' : Type v := UniformSpace.Completion F
    a : ContinuousLinearMap (RingHom.id 𝕜) F F' := UniformSpace.Completion.toComplL
    b : LinearIsometry (RingHom.id 𝕜) (ContinuousLinearMap (RingHom.id 𝕜) E F) (Co …
    ⊢ HasSum (fun n => (p.derivSeries n) fun x => y) f'
  -/
  rw [← b.isEmbedding.hasSum_iff]
  have : HasFPowerSeriesWithinOnBall (a ∘ f) (a.compFormalMultilinearSeries p) s x r :=
    a.comp_hasFPowerSeriesWithinOnBall h
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    r : ENNReal
    f : E → F
    x : E
    s : Set E
    h : HasFPowerSeriesWithinOnBall f p s x r
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    y : E
    hy : LT.lt (↑(NNNorm.nnnorm y)) r
    h'y : Membership.mem (Insert.insert x s) (HAdd.hAdd x y)
    hf' : HasFDerivWithinAt f f' (Insert.insert x s) (HAdd.hAdd x y)
    hu : UniqueDiffOn 𝕜 (Insert.insert x s)
    F' : Type v := UniformSpace.Completion F
    a : ContinuousLinearMap (RingHom.id 𝕜) F F' := UniformSpace.Completion.toComplL
    b : LinearIsometry (RingHom.id 𝕜) (ContinuousLinearMap (RingHom.id 𝕜) E F) (Co …
    this : HasFPowerSeriesWithinOnBall (Function.comp (⇑a) f) (a.compFormalMultili …
    ⊢ HasSum (Function.comp ⇑b fun n => (p.derivSeries n) fun x => y) (b f')
  -/
  have Z := (this.fderivWithin hu).hasSum h'y (by simpa [edist_eq_coe_nnnorm] using hy)
  have : fderivWithin 𝕜 (a ∘ f) (insert x s) (x + y) = a ∘L f' := by
    apply HasFDerivWithinAt.fderivWithin _ (hu _ h'y)
    exact a.hasFDerivAt.comp_hasFDerivWithinAt (x + y) hf'
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    r : ENNReal
    f : E → F
    x : E
    s : Set E
    h : HasFPowerSeriesWithinOnBall f p s x r
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    y : E
    hy : LT.lt (↑(NNNorm.nnnorm y)) r
    h'y : Membership.mem (Insert.insert x s) (HAdd.hAdd x y)
    hf' : HasFDerivWithinAt f f' (Insert.insert x s) (HAdd.hAdd x y)
    hu : UniqueDiffOn 𝕜 (Insert.insert x s)
    F' : Type v := UniformSpace.Completion F
    a : ContinuousLinearMap (RingHom.id 𝕜) F F' := UniformSpace.Completion.toComplL
    b : LinearIsometry (RingHom.id 𝕜) (ContinuousLinearMap (RingHom.id 𝕜) E F) (Co …
    this✝ : HasFPowerSeriesWithinOnBall (Function.comp (⇑a) f) (a.compFormalMultil …
    Z : HasSum (fun n => ((a.compFormalMultilinearSeries p).derivSeries n) fun x = …
    this : Eq (fderivWithin 𝕜 (Function.comp (⇑a) f) (Insert.insert x s) (HAdd.hAd …
    ⊢ HasSum (Function.comp ⇑b fun n => (p.derivSeries n) fun x => y) (b f')
  -/
  rw [this] at Z
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    r : ENNReal
    f : E → F
    x : E
    s : Set E
    h : HasFPowerSeriesWithinOnBall f p s x r
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    y : E
    hy : LT.lt (↑(NNNorm.nnnorm y)) r
    h'y : Membership.mem (Insert.insert x s) (HAdd.hAdd x y)
    hf' : HasFDerivWithinAt f f' (Insert.insert x s) (HAdd.hAdd x y)
    hu : UniqueDiffOn 𝕜 (Insert.insert x s)
    F' : Type v := UniformSpace.Completion F
    a : ContinuousLinearMap (RingHom.id 𝕜) F F' := UniformSpace.Completion.toComplL
    b : LinearIsometry (RingHom.id 𝕜) (ContinuousLinearMap (RingHom.id 𝕜) E F) (Co …
    this✝ : HasFPowerSeriesWithinOnBall (Function.comp (⇑a) f) (a.compFormalMultil …
    Z : HasSum (fun n => ((a.compFormalMultilinearSeries p).derivSeries n) fun x = …
    this : Eq (fderivWithin 𝕜 (Function.comp (⇑a) f) (Insert.insert x s) (HAdd.hAd …
    ⊢ HasSum (Function.comp ⇑b fun n => (p.derivSeries n) fun x => y) (b f')
  -/
  convert Z with n
  /-
    case h.e'_5.h
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    r : ENNReal
    f : E → F
    x : E
    s : Set E
    h : HasFPowerSeriesWithinOnBall f p s x r
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    y : E
    hy : LT.lt (↑(NNNorm.nnnorm y)) r
    h'y : Membership.mem (Insert.insert x s) (HAdd.hAdd x y)
    hf' : HasFDerivWithinAt f f' (Insert.insert x s) (HAdd.hAdd x y)
    hu : UniqueDiffOn 𝕜 (Insert.insert x s)
    F' : Type v := UniformSpace.Completion F
    a : ContinuousLinearMap (RingHom.id 𝕜) F F' := UniformSpace.Completion.toComplL
    b : LinearIsometry (RingHom.id 𝕜) (ContinuousLinearMap (RingHom.id 𝕜) E F) (Co …
    this✝ : HasFPowerSeriesWithinOnBall (Function.comp (⇑a) f) (a.compFormalMultil …
    Z : HasSum (fun n => ((a.compFormalMultilinearSeries p).derivSeries n) fun x = …
    this : Eq (fderivWithin 𝕜 (Function.comp (⇑a) f) (Insert.insert x s) (HAdd.hAd …
    n : Nat
    ⊢ Eq (Function.comp (⇑b) (fun n => (p.derivSeries n) fun x => y) n) (((a.compF …
  -/
  ext v
  simp only [FormalMultilinearSeries.derivSeries,
    ContinuousLinearMap.compFormalMultilinearSeries_apply,
    FormalMultilinearSeries.changeOriginSeries,
    ContinuousLinearMap.compContinuousMultilinearMap_coe, ContinuousLinearEquiv.coe_coe,
    LinearIsometryEquiv.coe_coe, Function.comp_apply, ContinuousMultilinearMap.sum_apply, map_sum,
    ContinuousLinearMap.coe_sum', Finset.sum_apply,
    Matrix.zero_empty]
  /-
    case h.e'_5.h.h
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    r : ENNReal
    f : E → F
    x : E
    s : Set E
    h : HasFPowerSeriesWithinOnBall f p s x r
    f' : ContinuousLinearMap (RingHom.id 𝕜) E F
    y : E
    hy : LT.lt (↑(NNNorm.nnnorm y)) r
    h'y : Membership.mem (Insert.insert x s) (HAdd.hAdd x y)
    hf' : HasFDerivWithinAt f f' (Insert.insert x s) (HAdd.hAdd x y)
    hu : UniqueDiffOn 𝕜 (Insert.insert x s)
    F' : Type v := UniformSpace.Completion F
    a : ContinuousLinearMap (RingHom.id 𝕜) F F' := UniformSpace.Completion.toComplL
    b : LinearIsometry (RingHom.id 𝕜) (ContinuousLinearMap (RingHom.id 𝕜) E F) (Co …
    this✝ : HasFPowerSeriesWithinOnBall (Function.comp (⇑a) f) (a.compFormalMultil …
    Z : HasSum (fun n => ((a.compFormalMultilinearSeries p).derivSeries n) fun x = …
    this : Eq (fderivWithin 𝕜 (Function.comp (⇑a) f) (Insert.insert x s) (HAdd.hAd …
    n : Nat
    v : E
    ⊢ Eq (Finset.univ.sum fun c => (b ((continuousMultilinearCurryFin1 𝕜 E F) ((p. …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- If a function has a power series within a set on a ball, then so does its derivative. Version
assuming that the function is analytic on `s`. For a version without this assumption but requiring
that `F` is complete, see `HasFPowerSeriesWithinOnBall.fderivWithin_of_mem`. -/
protected theorem HasFPowerSeriesWithinOnBall.fderivWithin_of_mem_of_analyticOn
    (hr : HasFPowerSeriesWithinOnBall f p s x r)
    (h : AnalyticOn 𝕜 f s) (hs : UniqueDiffOn 𝕜 s) (hx : x ∈ s) :
    HasFPowerSeriesWithinOnBall (fderivWithin 𝕜 f s) p.derivSeries s x r := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    r : ENNReal
    f : E → F
    x : E
    s : Set E
    hr : HasFPowerSeriesWithinOnBall f p s x r
    h : AnalyticOn 𝕜 f s
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    ⊢ HasFPowerSeriesWithinOnBall (fderivWithin 𝕜 f s) p.derivSeries s x r
  -/
  refine ⟨hr.r_le.trans p.radius_le_radius_derivSeries, hr.r_pos, fun {y} hy h'y ↦ ?_⟩
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    r : ENNReal
    f : E → F
    x : E
    s : Set E
    hr : HasFPowerSeriesWithinOnBall f p s x r
    h : AnalyticOn 𝕜 f s
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    y : E
    hy : Membership.mem (Insert.insert x s) (HAdd.hAdd x y)
    h'y : Membership.mem (EMetric.ball 0 r) y
    ⊢ HasSum (fun n => (p.derivSeries n) fun x => y) (fderivWithin 𝕜 f s (HAdd.hAd …
  -/
  apply hr.hasSum_derivSeries_of_hasFDerivWithinAt (by simpa [edist_eq_coe_nnnorm] using h'y) hy
    /-
      case hf'
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type v
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      r : ENNReal
      f : E → F
      x : E
      s : Set E
      hr : HasFPowerSeriesWithinOnBall f p s x r
      h : AnalyticOn 𝕜 f s
      hs : UniqueDiffOn 𝕜 s
      hx : Membership.mem s x
      y : E
      hy : Membership.mem (Insert.insert x s) (HAdd.hAdd x y)
      h'y : Membership.mem (EMetric.ball 0 r) y
      ⊢ HasFDerivWithinAt f (fderivWithin 𝕜 f s (HAdd.hAdd x y)) (Insert.insert x s) …
    -/
  · rw [insert_eq_of_mem hx] at hy ⊢
    /-
      case hf'
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type v
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      r : ENNReal
      f : E → F
      x : E
      s : Set E
      hr : HasFPowerSeriesWithinOnBall f p s x r
      h : AnalyticOn 𝕜 f s
      hs : UniqueDiffOn 𝕜 s
      hx : Membership.mem s x
      y : E
      hy : Membership.mem s (HAdd.hAdd x y)
      h'y : Membership.mem (EMetric.ball 0 r) y
      ⊢ HasFDerivWithinAt f (fderivWithin 𝕜 f s (HAdd.hAdd x y)) s (HAdd.hAdd x y)
    -/
    apply DifferentiableWithinAt.hasFDerivWithinAt
    /-
      case hf'.h
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type v
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      r : ENNReal
      f : E → F
      x : E
      s : Set E
      hr : HasFPowerSeriesWithinOnBall f p s x r
      h : AnalyticOn 𝕜 f s
      hs : UniqueDiffOn 𝕜 s
      hx : Membership.mem s x
      y : E
      hy : Membership.mem s (HAdd.hAdd x y)
      h'y : Membership.mem (EMetric.ball 0 r) y
      ⊢ DifferentiableWithinAt 𝕜 f s (HAdd.hAdd x y)
    -/
    exact h.differentiableOn _ hy
    /-
      🎉 no goals
    -/
    /-
      case hu
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type v
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      r : ENNReal
      f : E → F
      x : E
      s : Set E
      hr : HasFPowerSeriesWithinOnBall f p s x r
      h : AnalyticOn 𝕜 f s
      hs : UniqueDiffOn 𝕜 s
      hx : Membership.mem s x
      y : E
      hy : Membership.mem (Insert.insert x s) (HAdd.hAdd x y)
      h'y : Membership.mem (EMetric.ball 0 r) y
      ⊢ UniqueDiffOn 𝕜 (Insert.insert x s)
    -/
  · rwa [insert_eq_of_mem hx]
    /-
      🎉 no goals
    -/


/-- If a function is analytic within a set with unique differentials, then so is its derivative.
Note that this theorem does not require completeness of the space. -/
protected theorem AnalyticOn.fderivWithin (h : AnalyticOn 𝕜 f s) (hu : UniqueDiffOn 𝕜 s) :
    AnalyticOn 𝕜 (fderivWithin 𝕜 f s) s := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    s : Set E
    h : AnalyticOn 𝕜 f s
    hu : UniqueDiffOn 𝕜 s
    ⊢ AnalyticOn 𝕜 (fderivWithin 𝕜 f s) s
  -/
  intro x hx
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    s : Set E
    h : AnalyticOn 𝕜 f s
    hu : UniqueDiffOn 𝕜 s
    x : E
    hx : Membership.mem s x
    ⊢ AnalyticWithinAt 𝕜 (fderivWithin 𝕜 f s) s x
  -/
  rcases h x hx with ⟨p, r, hr⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    s : Set E
    h : AnalyticOn 𝕜 f s
    hu : UniqueDiffOn 𝕜 s
    x : E
    hx : Membership.mem s x
    p : FormalMultilinearSeries 𝕜 E F
    r : ENNReal
    hr : HasFPowerSeriesWithinOnBall f p s x r
    ⊢ AnalyticWithinAt 𝕜 (fderivWithin 𝕜 f s) s x
  -/
  refine ⟨p.derivSeries, r, hr.fderivWithin_of_mem_of_analyticOn h hu hx⟩
  /-
    🎉 no goals
  -/


/-- If a function is analytic on a set `s`, so are its successive Fréchet derivative within this
set. Note that this theorem does not require completeness of the space. -/
protected theorem AnalyticOn.iteratedFDerivWithin (h : AnalyticOn 𝕜 f s)
    (hu : UniqueDiffOn 𝕜 s) (n : ℕ) :
    AnalyticOn 𝕜 (iteratedFDerivWithin 𝕜 n f s) s := by
  induction n with
  | zero =>
    rw [iteratedFDerivWithin_zero_eq_comp]
    exact ((continuousMultilinearCurryFin0 𝕜 E F).symm : F →L[𝕜] E[×0]→L[𝕜] F)
      |>.comp_analyticOn h
  | succ n IH =>
    rw [iteratedFDerivWithin_succ_eq_comp_left]
    apply AnalyticOnNhd.comp_analyticOn _ (IH.fderivWithin hu) (mapsTo_univ _ _)
    apply LinearIsometryEquiv.analyticOnNhd


protected lemma AnalyticOn.hasFTaylorSeriesUpToOn {n : WithTop ℕ∞}
    (h : AnalyticOn 𝕜 f s) (hu : UniqueDiffOn 𝕜 s) :
    HasFTaylorSeriesUpToOn n f (ftaylorSeriesWithin 𝕜 f s) s := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    s : Set E
    n : WithTop ENat
    h : AnalyticOn 𝕜 f s
    hu : UniqueDiffOn 𝕜 s
    ⊢ HasFTaylorSeriesUpToOn n f (ftaylorSeriesWithin 𝕜 f s) s
  -/
  refine ⟨fun x _hx ↦ rfl, fun m _hm x hx ↦ ?_, fun m _hm x hx ↦ ?_⟩
    /-
      case refine_1
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type v
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → F
      s : Set E
      n : WithTop ENat
      h : AnalyticOn 𝕜 f s
      hu : UniqueDiffOn 𝕜 s
      m : Nat
      _hm : LT.lt (↑m) n
      x : E
      hx : Membership.mem s x
      ⊢ HasFDerivWithinAt (fun x => ftaylorSeriesWithin 𝕜 f s x m) (ftaylorSeriesWit …
    -/
  · have := (h.iteratedFDerivWithin hu m x hx).differentiableWithinAt.hasFDerivWithinAt
    /-
      case refine_1
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type v
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → F
      s : Set E
      n : WithTop ENat
      h : AnalyticOn 𝕜 f s
      hu : UniqueDiffOn 𝕜 s
      m : Nat
      _hm : LT.lt (↑m) n
      x : E
      hx : Membership.mem s x
      this : HasFDerivWithinAt (iteratedFDerivWithin 𝕜 m f s) (fderivWithin 𝕜 (itera …
      ⊢ HasFDerivWithinAt (fun x => ftaylorSeriesWithin 𝕜 f s x m) (ftaylorSeriesWit …
    -/
    rwa [insert_eq_of_mem hx] at this
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type v
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → F
      s : Set E
      n : WithTop ENat
      h : AnalyticOn 𝕜 f s
      hu : UniqueDiffOn 𝕜 s
      m : Nat
      _hm : LE.le (↑m) n
      x : E
      hx : Membership.mem s x
      ⊢ ContinuousWithinAt (fun x => ftaylorSeriesWithin 𝕜 f s x m) s x
    -/
  · exact (h.iteratedFDerivWithin hu m x hx).continuousWithinAt
    /-
      🎉 no goals
    -/


lemma AnalyticOn.exists_hasFTaylorSeriesUpToOn
    (h : AnalyticOn 𝕜 f s) (hu : UniqueDiffOn 𝕜 s) :
    ∃ p : E → FormalMultilinearSeries 𝕜 E F,
      HasFTaylorSeriesUpToOn ⊤ f p s ∧ ∀ i, AnalyticOn 𝕜 (fun x ↦ p x i) s :=
  ⟨ftaylorSeriesWithin 𝕜 f s, h.hasFTaylorSeriesUpToOn hu, h.iteratedFDerivWithin hu⟩


theorem AnalyticOnNhd.fderiv_of_isOpen (h : AnalyticOnNhd 𝕜 f s) (hs : IsOpen s) :
    AnalyticOnNhd 𝕜 (fderiv 𝕜 f) s := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    s : Set E
    h : AnalyticOnNhd 𝕜 f s
    hs : IsOpen s
    ⊢ AnalyticOnNhd 𝕜 (fderiv 𝕜 f) s
  -/
  rw [← hs.analyticOn_iff_analyticOnNhd] at h ⊢
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    s : Set E
    h : AnalyticOn 𝕜 f s
    hs : IsOpen s
    ⊢ AnalyticOn 𝕜 (fderiv 𝕜 f) s
  -/
  exact (h.fderivWithin hs.uniqueDiffOn).congr (fun x hx ↦ (fderivWithin_of_isOpen hs hx).symm)
  /-
    🎉 no goals
  -/


theorem AnalyticOnNhd.iteratedFDeriv_of_isOpen (h : AnalyticOnNhd 𝕜 f s) (hs : IsOpen s) (n : ℕ) :
    AnalyticOnNhd 𝕜 (iteratedFDeriv 𝕜 n f) s := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    s : Set E
    h : AnalyticOnNhd 𝕜 f s
    hs : IsOpen s
    n : Nat
    ⊢ AnalyticOnNhd 𝕜 (iteratedFDeriv 𝕜 n f) s
  -/
  rw [← hs.analyticOn_iff_analyticOnNhd] at h ⊢
  exact (h.iteratedFDerivWithin hs.uniqueDiffOn n).congr
    (fun x hx ↦ (iteratedFDerivWithin_of_isOpen n hs hx).symm)


/-- If a partial homeomorphism `f` is analytic at a point `a`, with invertible derivative, then
its inverse is analytic at `f a`. -/
theorem PartialHomeomorph.analyticAt_symm' (f : PartialHomeomorph E F) {a : E}
    {i : E ≃L[𝕜] F} (h0 : a ∈ f.source) (h : AnalyticAt 𝕜 f a) (h' : fderiv 𝕜 f a = i) :
    AnalyticAt 𝕜 f.symm (f a) := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : PartialHomeomorph E F
    a : E
    i : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    h0 : Membership.mem f.source a
    h : AnalyticAt 𝕜 (↑f) a
    h' : Eq (fderiv 𝕜 (↑f) a) ↑i
    ⊢ AnalyticAt 𝕜 (↑f.symm) (↑f a)
  -/
  rcases h with ⟨p, hp⟩
  /-
    case intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : PartialHomeomorph E F
    a : E
    i : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    h0 : Membership.mem f.source a
    h' : Eq (fderiv 𝕜 (↑f) a) ↑i
    p : FormalMultilinearSeries 𝕜 E F
    hp : HasFPowerSeriesAt (↑f) p a
    ⊢ AnalyticAt 𝕜 (↑f.symm) (↑f a)
  -/
  have : p 1 = (continuousMultilinearCurryFin1 𝕜 E F).symm i := by simp [← h', hp.fderiv_eq]
  /-
    case intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : PartialHomeomorph E F
    a : E
    i : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    h0 : Membership.mem f.source a
    h' : Eq (fderiv 𝕜 (↑f) a) ↑i
    p : FormalMultilinearSeries 𝕜 E F
    hp : HasFPowerSeriesAt (↑f) p a
    this : Eq (p 1) ((continuousMultilinearCurryFin1 𝕜 E F).symm ↑i)
    ⊢ AnalyticAt 𝕜 (↑f.symm) (↑f a)
  -/
  exact (f.hasFPowerSeriesAt_symm h0 hp this).analyticAt
  /-
    🎉 no goals
  -/


/-- If a partial homeomorphism `f` is analytic at a point `f.symm a`, with invertible derivative,
then its inverse is analytic at `a`. -/
theorem PartialHomeomorph.analyticAt_symm (f : PartialHomeomorph E F) {a : F}
    {i : E ≃L[𝕜] F} (h0 : a ∈ f.target) (h : AnalyticAt 𝕜 f (f.symm a))
    (h' : fderiv 𝕜 f (f.symm a) = i) :
    AnalyticAt 𝕜 f.symm a := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : PartialHomeomorph E F
    a : F
    i : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    h0 : Membership.mem f.target a
    h : AnalyticAt 𝕜 (↑f) (↑f.symm a)
    h' : Eq (fderiv 𝕜 (↑f) (↑f.symm a)) ↑i
    ⊢ AnalyticAt 𝕜 (↑f.symm) a
  -/
  have : a = f (f.symm a) := by simp [h0]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : PartialHomeomorph E F
    a : F
    i : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    h0 : Membership.mem f.target a
    h : AnalyticAt 𝕜 (↑f) (↑f.symm a)
    h' : Eq (fderiv 𝕜 (↑f) (↑f.symm a)) ↑i
    this : Eq a (↑f (↑f.symm a))
    ⊢ AnalyticAt 𝕜 (↑f.symm) a
  -/
  rw [this]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : PartialHomeomorph E F
    a : F
    i : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    h0 : Membership.mem f.target a
    h : AnalyticAt 𝕜 (↑f) (↑f.symm a)
    h' : Eq (fderiv 𝕜 (↑f) (↑f.symm a)) ↑i
    this : Eq a (↑f (↑f.symm a))
    ⊢ AnalyticAt 𝕜 (↑f.symm) (↑f (↑f.symm a))
  -/
  exact f.analyticAt_symm' (by simp [h0]) h h'
  /-
    🎉 no goals
  -/


protected theorem HasFPowerSeriesAt.hasStrictDerivAt (h : HasFPowerSeriesAt f p x) :
    HasStrictDerivAt f (p 1 fun _ => 1) x :=
  h.hasStrictFDerivAt.hasStrictDerivAt


protected theorem HasFPowerSeriesAt.hasDerivAt (h : HasFPowerSeriesAt f p x) :
    HasDerivAt f (p 1 fun _ => 1) x :=
  h.hasStrictDerivAt.hasDerivAt


protected theorem HasFPowerSeriesAt.deriv (h : HasFPowerSeriesAt f p x) :
    deriv f x = p 1 fun _ => 1 :=
  h.hasDerivAt.deriv


/-- If a function is analytic on a set `s` in a complete space, so is its derivative. -/
protected theorem AnalyticOnNhd.deriv [CompleteSpace F] (h : AnalyticOnNhd 𝕜 f s) :
    AnalyticOnNhd 𝕜 (deriv f) s :=
  (ContinuousLinearMap.apply 𝕜 F (1 : 𝕜)).comp_analyticOnNhd h.fderiv


@[deprecated (since := "2024-09-26")]
alias AnalyticOn.deriv := AnalyticOnNhd.deriv


/-- If a function is analytic on an open set `s`, so is its derivative. -/
theorem AnalyticOnNhd.deriv_of_isOpen (h : AnalyticOnNhd 𝕜 f s) (hs : IsOpen s) :
    AnalyticOnNhd 𝕜 (deriv f) s :=
  (ContinuousLinearMap.apply 𝕜 F (1 : 𝕜)).comp_analyticOnNhd (h.fderiv_of_isOpen hs)


/-- If a function is analytic on a set `s`, so are its successive derivatives. -/
theorem AnalyticOnNhd.iterated_deriv [CompleteSpace F] (h : AnalyticOnNhd 𝕜 f s) (n : ℕ) :
    AnalyticOnNhd 𝕜 (_root_.deriv^[n] f) s := by
  induction n with
  | zero => exact h
  | succ n IH => simpa only [Function.iterate_succ', Function.comp_apply] using IH.deriv


@[deprecated (since := "2024-09-26")]
alias AnalyticOn.iterated_deriv := AnalyticOnNhd.iterated_deriv


theorem HasFiniteFPowerSeriesOnBall.differentiableOn
    (h : HasFiniteFPowerSeriesOnBall f p x n r) : DifferentiableOn 𝕜 f (EMetric.ball x r) :=
  fun _ hy ↦ (h.cPolynomialAt_of_mem hy).analyticAt.differentiableWithinAt


theorem HasFiniteFPowerSeriesOnBall.hasFDerivAt (h : HasFiniteFPowerSeriesOnBall f p x n r)
    {y : E} (hy : (‖y‖₊ : ℝ≥0∞) < r) :
    HasFDerivAt f (continuousMultilinearCurryFin1 𝕜 E F (p.changeOrigin y 1)) (x + y) :=
  (h.changeOrigin hy).toHasFPowerSeriesOnBall.hasFPowerSeriesAt.hasFDerivAt


theorem HasFiniteFPowerSeriesOnBall.fderiv_eq (h : HasFiniteFPowerSeriesOnBall f p x n r)
    {y : E} (hy : (‖y‖₊ : ℝ≥0∞) < r) :
    fderiv 𝕜 f (x + y) = continuousMultilinearCurryFin1 𝕜 E F (p.changeOrigin y 1) :=
  (h.hasFDerivAt hy).fderiv


/-- If a function has a finite power series on a ball, then so does its derivative. -/
protected theorem HasFiniteFPowerSeriesOnBall.fderiv
    (h : HasFiniteFPowerSeriesOnBall f p x (n + 1) r) :
    HasFiniteFPowerSeriesOnBall (fderiv 𝕜 f) p.derivSeries x n r := by
  refine .congr (f := fun z ↦ continuousMultilinearCurryFin1 𝕜 E F (p.changeOrigin (z - x) 1)) ?_
    fun z hz ↦ ?_
  · refine continuousMultilinearCurryFin1 𝕜 E F
      |>.toContinuousLinearEquiv.toContinuousLinearMap.comp_hasFiniteFPowerSeriesOnBall ?_
    simpa using
      ((p.hasFiniteFPowerSeriesOnBall_changeOrigin 1 h.finite).mono h.r_pos le_top).comp_sub x
  /-
    case refine_2
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    r : ENNReal
    n : Nat
    f : E → F
    x : E
    h : HasFiniteFPowerSeriesOnBall f p x (HAdd.hAdd n 1) r
    z : E
    hz : Membership.mem (EMetric.ball x r) z
    ⊢ Eq ((fun z => (continuousMultilinearCurryFin1 𝕜 E F) (p.changeOrigin (HSub.h …
  -/
  dsimp only
  /-
    case refine_2
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    r : ENNReal
    n : Nat
    f : E → F
    x : E
    h : HasFiniteFPowerSeriesOnBall f p x (HAdd.hAdd n 1) r
    z : E
    hz : Membership.mem (EMetric.ball x r) z
    ⊢ Eq ((continuousMultilinearCurryFin1 𝕜 E F) (p.changeOrigin (HSub.hSub z x) 1 …
  -/
  rw [← h.fderiv_eq, add_sub_cancel]
  /-
    case refine_2
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    r : ENNReal
    n : Nat
    f : E → F
    x : E
    h : HasFiniteFPowerSeriesOnBall f p x (HAdd.hAdd n 1) r
    z : E
    hz : Membership.mem (EMetric.ball x r) z
    ⊢ LT.lt (↑(NNNorm.nnnorm (HSub.hSub z x))) r
  -/
  simpa only [edist_eq_coe_nnnorm_sub, EMetric.mem_ball] using hz
  /-
    🎉 no goals
  -/


/-- If a function has a finite power series on a ball, then so does its derivative.
This is a variant of `HasFiniteFPowerSeriesOnBall.fderiv` where the degree of `f` is `< n`
and not `< n + 1`. -/
theorem HasFiniteFPowerSeriesOnBall.fderiv' (h : HasFiniteFPowerSeriesOnBall f p x n r) :
    HasFiniteFPowerSeriesOnBall (fderiv 𝕜 f) p.derivSeries x (n - 1) r := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    r : ENNReal
    n : Nat
    f : E → F
    x : E
    h : HasFiniteFPowerSeriesOnBall f p x n r
    ⊢ HasFiniteFPowerSeriesOnBall (fderiv 𝕜 f) p.derivSeries x (HSub.hSub n 1) r
  -/
  obtain rfl | hn := eq_or_ne n 0
    /-
      case inl
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type v
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      r : ENNReal
      f : E → F
      x : E
      h : HasFiniteFPowerSeriesOnBall f p x 0 r
      ⊢ HasFiniteFPowerSeriesOnBall (fderiv 𝕜 f) p.derivSeries x (HSub.hSub 0 1) r
    -/
  · rw [zero_tsub]
    /-
      case inl
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type v
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      r : ENNReal
      f : E → F
      x : E
      h : HasFiniteFPowerSeriesOnBall f p x 0 r
      ⊢ HasFiniteFPowerSeriesOnBall (fderiv 𝕜 f) p.derivSeries x 0 r
    -/
    refine HasFiniteFPowerSeriesOnBall.bound_zero_of_eq_zero (fun y hy ↦ ?_) h.r_pos fun n ↦ ?_
      /-
        case inl.refine_1
        𝕜 : Type u_1
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type u
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type v
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        p : FormalMultilinearSeries 𝕜 E F
        r : ENNReal
        f : E → F
        x : E
        h : HasFiniteFPowerSeriesOnBall f p x 0 r
        y : E
        hy : Membership.mem (EMetric.ball x r) y
        ⊢ Eq (fderiv 𝕜 f y) 0
      -/
    · rw [Filter.EventuallyEq.fderiv_eq (f := fun _ ↦ 0)]
        /-
          case inl.refine_1
          𝕜 : Type u_1
          inst✝⁴ : NontriviallyNormedField 𝕜
          E : Type u
          inst✝³ : NormedAddCommGroup E
          inst✝² : NormedSpace 𝕜 E
          F : Type v
          inst✝¹ : NormedAddCommGroup F
          inst✝ : NormedSpace 𝕜 F
          p : FormalMultilinearSeries 𝕜 E F
          r : ENNReal
          f : E → F
          x : E
          h : HasFiniteFPowerSeriesOnBall f p x 0 r
          y : E
          hy : Membership.mem (EMetric.ball x r) y
          ⊢ Eq (fderiv 𝕜 (fun x => 0) y) 0
        -/
      · rw [fderiv_const, Pi.zero_apply]
        /-
          🎉 no goals
        -/
      · exact Filter.eventuallyEq_iff_exists_mem.mpr ⟨EMetric.ball x r,
          EMetric.isOpen_ball.mem_nhds hy, fun z hz ↦ by rw [h.eq_zero_of_bound_zero z hz]⟩
      /-
        case inl.refine_2
        𝕜 : Type u_1
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type u
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type v
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        p : FormalMultilinearSeries 𝕜 E F
        r : ENNReal
        f : E → F
        x : E
        h : HasFiniteFPowerSeriesOnBall f p x 0 r
        n : Nat
        ⊢ Eq (p.derivSeries n) 0
      -/
    · apply ContinuousMultilinearMap.ext; intro a
      /-
        case inl.refine_2.H
        𝕜 : Type u_1
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type u
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type v
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        p : FormalMultilinearSeries 𝕜 E F
        r : ENNReal
        f : E → F
        x : E
        h : HasFiniteFPowerSeriesOnBall f p x 0 r
        n : Nat
        a : Fin n → E
        ⊢ Eq ((p.derivSeries n) a) (0 a)
      -/
      change (continuousMultilinearCurryFin1 𝕜 E F) (p.changeOriginSeries 1 n a) = 0
      /-
        case inl.refine_2.H
        𝕜 : Type u_1
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type u
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type v
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        p : FormalMultilinearSeries 𝕜 E F
        r : ENNReal
        f : E → F
        x : E
        h : HasFiniteFPowerSeriesOnBall f p x 0 r
        n : Nat
        a : Fin n → E
        ⊢ Eq ((continuousMultilinearCurryFin1 𝕜 E F) ((p.changeOriginSeries 1 n) a)) 0
      -/
      rw [p.changeOriginSeries_finite_of_finite h.finite 1 (Nat.zero_le _)]
      /-
        case inl.refine_2.H
        𝕜 : Type u_1
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type u
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type v
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        p : FormalMultilinearSeries 𝕜 E F
        r : ENNReal
        f : E → F
        x : E
        h : HasFiniteFPowerSeriesOnBall f p x 0 r
        n : Nat
        a : Fin n → E
        ⊢ Eq ((continuousMultilinearCurryFin1 𝕜 E F) (0 a)) 0
      -/
      exact map_zero _
      /-
        🎉 no goals
      -/
    /-
      case inr
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type v
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      r : ENNReal
      n : Nat
      f : E → F
      x : E
      h : HasFiniteFPowerSeriesOnBall f p x n r
      hn : Ne n 0
      ⊢ HasFiniteFPowerSeriesOnBall (fderiv 𝕜 f) p.derivSeries x (HSub.hSub n 1) r
    -/
  · rw [← Nat.succ_pred hn] at h
    /-
      case inr
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type v
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      r : ENNReal
      n : Nat
      f : E → F
      x : E
      h : HasFiniteFPowerSeriesOnBall f p x n.pred.succ r
      hn : Ne n 0
      ⊢ HasFiniteFPowerSeriesOnBall (fderiv 𝕜 f) p.derivSeries x (HSub.hSub n 1) r
    -/
    exact h.fderiv
    /-
      🎉 no goals
    -/


/-- If a function is polynomial on a set `s`, so is its Fréchet derivative. -/
theorem CPolynomialOn.fderiv (h : CPolynomialOn 𝕜 f s) :
    CPolynomialOn 𝕜 (fderiv 𝕜 f) s := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    s : Set E
    h : CPolynomialOn 𝕜 f s
    ⊢ CPolynomialOn 𝕜 (_root_.fderiv 𝕜 f) s
  -/
  intro y hy
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    s : Set E
    h : CPolynomialOn 𝕜 f s
    y : E
    hy : Membership.mem s y
    ⊢ CPolynomialAt 𝕜 (_root_.fderiv 𝕜 f) y
  -/
  rcases h y hy with ⟨p, r, n, hp⟩
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    s : Set E
    h : CPolynomialOn 𝕜 f s
    y : E
    hy : Membership.mem s y
    p : FormalMultilinearSeries 𝕜 E F
    r : Nat
    n : ENNReal
    hp : HasFiniteFPowerSeriesOnBall f p y r n
    ⊢ CPolynomialAt 𝕜 (_root_.fderiv 𝕜 f) y
  -/
  exact hp.fderiv'.cPolynomialAt
  /-
    🎉 no goals
  -/


/-- If a function is polynomial on a set `s`, so are its successive Fréchet derivative. -/
theorem CPolynomialOn.iteratedFDeriv (h : CPolynomialOn 𝕜 f s) (n : ℕ) :
    CPolynomialOn 𝕜 (iteratedFDeriv 𝕜 n f) s := by
  induction n with
  | zero =>
    rw [iteratedFDeriv_zero_eq_comp]
    exact ((continuousMultilinearCurryFin0 𝕜 E F).symm : F →L[𝕜] E[×0]→L[𝕜] F).comp_cPolynomialOn h
  | succ n IH =>
    rw [iteratedFDeriv_succ_eq_comp_left]
    convert ContinuousLinearMap.comp_cPolynomialOn ?g IH.fderiv
    case g => exact ↑(continuousMultilinearCurryLeftEquiv 𝕜 (fun _ : Fin (n + 1) ↦ E) F).symm
    simp


/-- If a function is polynomial on a set `s`, so is its derivative. -/
protected theorem CPolynomialOn.deriv (h : CPolynomialOn 𝕜 f s) : CPolynomialOn 𝕜 (deriv f) s :=
  (ContinuousLinearMap.apply 𝕜 F (1 : 𝕜)).comp_cPolynomialOn h.fderiv


/-- If a function is polynomial on a set `s`, so are its successive derivatives. -/
theorem CPolynomialOn.iterated_deriv (h : CPolynomialOn 𝕜 f s) (n : ℕ) :
    CPolynomialOn 𝕜 (deriv^[n] f) s := by
  induction n with
  | zero => exact h
  | succ n IH => simpa only [Function.iterate_succ', Function.comp_apply] using IH.deriv


theorem changeOriginSeries_support {k l : ℕ} (h : k + l ≠ Fintype.card ι) :
    f.toFormalMultilinearSeries.changeOriginSeries k l = 0 :=
  Finset.sum_eq_zero fun _ _ ↦ by
    simp_rw [FormalMultilinearSeries.changeOriginSeriesTerm,
      toFormalMultilinearSeries, dif_neg h.symm, LinearIsometryEquiv.map_zero]


open Finset in
theorem changeOrigin_toFormalMultilinearSeries [DecidableEq ι] :
    continuousMultilinearCurryFin1 𝕜 (∀ i, E i) F (f.toFormalMultilinearSeries.changeOrigin x 1) =
    f.linearDeriv x := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    ι : Type u_2
    E : ι → Type u_3
    inst✝³ : (i : ι) → NormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝¹ : Fintype ι
    f : ContinuousMultilinearMap 𝕜 E F
    x : (i : ι) → E i
    inst✝ : DecidableEq ι
    ⊢ Eq ((continuousMultilinearCurryFin1 𝕜 ((i : ι) → E i) F) (f.toFormalMultilin …
  -/
  ext y
  rw [continuousMultilinearCurryFin1_apply, linearDeriv_apply,
      changeOrigin, FormalMultilinearSeries.sum]
  /-
    case h
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    ι : Type u_2
    E : ι → Type u_3
    inst✝³ : (i : ι) → NormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝¹ : Fintype ι
    f : ContinuousMultilinearMap 𝕜 E F
    x : (i : ι) → E i
    inst✝ : DecidableEq ι
    y : (i : ι) → E i
    ⊢ Eq ((tsum fun n => (f.toFormalMultilinearSeries.changeOriginSeries 1 n) fun  …
  -/
  cases isEmpty_or_nonempty ι
  · have (l) : 1 + l ≠ Fintype.card ι := by
      rw [add_comm, Fintype.card_eq_zero]; exact Nat.succ_ne_zero _
    /-
      case h.inl
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      F : Type v
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      ι : Type u_2
      E : ι → Type u_3
      inst✝³ : (i : ι) → NormedAddCommGroup (E i)
      inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝¹ : Fintype ι
      f : ContinuousMultilinearMap 𝕜 E F
      x : (i : ι) → E i
      inst✝ : DecidableEq ι
      y : (i : ι) → E i
      h✝ : IsEmpty ι
      this : ∀ (l : Nat), Ne (HAdd.hAdd 1 l) (Fintype.card ι)
      ⊢ Eq ((tsum fun n => (f.toFormalMultilinearSeries.changeOriginSeries 1 n) fun  …
    -/
    simp_rw [Fintype.sum_empty, changeOriginSeries_support _ (this _), zero_apply _, tsum_zero]; rfl
                                                                                                 /-
                                                                                                   🎉 no goals
                                                                                                 -/
  /-
    case h.inr
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    ι : Type u_2
    E : ι → Type u_3
    inst✝³ : (i : ι) → NormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝¹ : Fintype ι
    f : ContinuousMultilinearMap 𝕜 E F
    x : (i : ι) → E i
    inst✝ : DecidableEq ι
    y : (i : ι) → E i
    h✝ : Nonempty ι
    ⊢ Eq ((tsum fun n => (f.toFormalMultilinearSeries.changeOriginSeries 1 n) fun  …
  -/
  rw [tsum_eq_single (Fintype.card ι - 1), changeOriginSeries]; swap
    /-
      case h.inr
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      F : Type v
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      ι : Type u_2
      E : ι → Type u_3
      inst✝³ : (i : ι) → NormedAddCommGroup (E i)
      inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝¹ : Fintype ι
      f : ContinuousMultilinearMap 𝕜 E F
      x : (i : ι) → E i
      inst✝ : DecidableEq ι
      y : (i : ι) → E i
      h✝ : Nonempty ι
      ⊢ ∀ (b' : Nat), Ne b' (HSub.hSub (Fintype.card ι) 1) → Eq ((f.toFormalMultilin …
    -/
  · intro m hm
    /-
      case h.inr
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      F : Type v
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      ι : Type u_2
      E : ι → Type u_3
      inst✝³ : (i : ι) → NormedAddCommGroup (E i)
      inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝¹ : Fintype ι
      f : ContinuousMultilinearMap 𝕜 E F
      x : (i : ι) → E i
      inst✝ : DecidableEq ι
      y : (i : ι) → E i
      h✝ : Nonempty ι
      m : Nat
      hm : Ne m (HSub.hSub (Fintype.card ι) 1)
      ⊢ Eq ((f.toFormalMultilinearSeries.changeOriginSeries 1 m) fun x_1 => x) 0
    -/
    rw [Ne, eq_tsub_iff_add_eq_of_le (by exact Fintype.card_pos), add_comm] at hm
    /-
      case h.inr
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      F : Type v
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      ι : Type u_2
      E : ι → Type u_3
      inst✝³ : (i : ι) → NormedAddCommGroup (E i)
      inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝¹ : Fintype ι
      f : ContinuousMultilinearMap 𝕜 E F
      x : (i : ι) → E i
      inst✝ : DecidableEq ι
      y : (i : ι) → E i
      h✝ : Nonempty ι
      m : Nat
      hm : Not (Eq (HAdd.hAdd 1 m) (Fintype.card ι))
      ⊢ Eq ((f.toFormalMultilinearSeries.changeOriginSeries 1 m) fun x_1 => x) 0
    -/
    rw [f.changeOriginSeries_support hm, zero_apply]
    /-
      🎉 no goals
    -/
  /-
    case h.inr
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    ι : Type u_2
    E : ι → Type u_3
    inst✝³ : (i : ι) → NormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝¹ : Fintype ι
    f : ContinuousMultilinearMap 𝕜 E F
    x : (i : ι) → E i
    inst✝ : DecidableEq ι
    y : (i : ι) → E i
    h✝ : Nonempty ι
    ⊢ Eq (((Finset.univ.sum fun s => f.toFormalMultilinearSeries.changeOriginSerie …
  -/
  rw [sum_apply, ContinuousMultilinearMap.sum_apply, Fin.snoc_zero]
  /-
    case h.inr
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    ι : Type u_2
    E : ι → Type u_3
    inst✝³ : (i : ι) → NormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝¹ : Fintype ι
    f : ContinuousMultilinearMap 𝕜 E F
    x : (i : ι) → E i
    inst✝ : DecidableEq ι
    y : (i : ι) → E i
    h✝ : Nonempty ι
    ⊢ Eq (Finset.univ.sum fun a => ((f.toFormalMultilinearSeries.changeOriginSerie …
  -/
  simp_rw [changeOriginSeriesTerm_apply]
  refine (Fintype.sum_bijective (?_ ∘ Fintype.equivFinOfCardEq (Nat.add_sub_of_le
    Fintype.card_pos).symm) (.comp ?_ <| Equiv.bijective _) _ _ fun i ↦ ?_).symm
  · exact (⟨{·}ᶜ, by
      rw [card_compl, Fintype.card_fin, Finset.card_singleton, Nat.add_sub_cancel_left]⟩)
    /-
      case h.inr.refine_2
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      F : Type v
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      ι : Type u_2
      E : ι → Type u_3
      inst✝³ : (i : ι) → NormedAddCommGroup (E i)
      inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝¹ : Fintype ι
      f : ContinuousMultilinearMap 𝕜 E F
      x : (i : ι) → E i
      inst✝ : DecidableEq ι
      y : (i : ι) → E i
      h✝ : Nonempty ι
      ⊢ Function.Bijective fun x => ⟨HasCompl.compl (Singleton.singleton x), ⋯⟩
    -/
  · use fun _ _ ↦ (singleton_injective <| compl_injective <| Subtype.ext_iff.mp ·)
    /-
      case right
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      F : Type v
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      ι : Type u_2
      E : ι → Type u_3
      inst✝³ : (i : ι) → NormedAddCommGroup (E i)
      inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝¹ : Fintype ι
      f : ContinuousMultilinearMap 𝕜 E F
      x : (i : ι) → E i
      inst✝ : DecidableEq ι
      y : (i : ι) → E i
      h✝ : Nonempty ι
      ⊢ Function.Surjective fun x => ⟨HasCompl.compl (Singleton.singleton x), ⋯⟩
    -/
    intro ⟨s, hs⟩
    /-
      case right
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      F : Type v
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      ι : Type u_2
      E : ι → Type u_3
      inst✝³ : (i : ι) → NormedAddCommGroup (E i)
      inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝¹ : Fintype ι
      f : ContinuousMultilinearMap 𝕜 E F
      x : (i : ι) → E i
      inst✝ : DecidableEq ι
      y : (i : ι) → E i
      h✝ : Nonempty ι
      s : Finset (Fin (HAdd.hAdd 1 (HSub.hSub (Fintype.card ι) 1)))
      hs : Eq s.card (HSub.hSub (Fintype.card ι) 1)
      ⊢ Exists fun a => Eq ((fun x => ⟨HasCompl.compl (Singleton.singleton x), ⋯⟩) a …
    -/
    have h : #sᶜ = 1 := by rw [card_compl, hs, Fintype.card_fin, Nat.add_sub_cancel]
    /-
      case right
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      F : Type v
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      ι : Type u_2
      E : ι → Type u_3
      inst✝³ : (i : ι) → NormedAddCommGroup (E i)
      inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝¹ : Fintype ι
      f : ContinuousMultilinearMap 𝕜 E F
      x : (i : ι) → E i
      inst✝ : DecidableEq ι
      y : (i : ι) → E i
      h✝ : Nonempty ι
      s : Finset (Fin (HAdd.hAdd 1 (HSub.hSub (Fintype.card ι) 1)))
      hs : Eq s.card (HSub.hSub (Fintype.card ι) 1)
      h : Eq (HasCompl.compl s).card 1
      ⊢ Exists fun a => Eq ((fun x => ⟨HasCompl.compl (Singleton.singleton x), ⋯⟩) a …
    -/
    obtain ⟨a, ha⟩ := card_eq_one.mp h
    /-
      case right.intro
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      F : Type v
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      ι : Type u_2
      E : ι → Type u_3
      inst✝³ : (i : ι) → NormedAddCommGroup (E i)
      inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝¹ : Fintype ι
      f : ContinuousMultilinearMap 𝕜 E F
      x : (i : ι) → E i
      inst✝ : DecidableEq ι
      y : (i : ι) → E i
      h✝ : Nonempty ι
      s : Finset (Fin (HAdd.hAdd 1 (HSub.hSub (Fintype.card ι) 1)))
      hs : Eq s.card (HSub.hSub (Fintype.card ι) 1)
      h : Eq (HasCompl.compl s).card 1
      a : Fin (HAdd.hAdd 1 (HSub.hSub (Fintype.card ι) 1))
      ha : Eq (HasCompl.compl s) (Singleton.singleton a)
      ⊢ Exists fun a => Eq ((fun x => ⟨HasCompl.compl (Singleton.singleton x), ⋯⟩) a …
    -/
    exact ⟨a, Subtype.ext (compl_eq_comm.mp ha)⟩
    /-
      🎉 no goals
    -/
  rw [Function.comp_apply, Subtype.coe_mk, compl_singleton, piecewise_erase_univ,
    toFormalMultilinearSeries, dif_pos (Nat.add_sub_of_le Fintype.card_pos).symm]
  simp_rw [domDomCongr_apply, compContinuousLinearMap_apply, ContinuousLinearMap.proj_apply,
    Function.update_apply, (Equiv.injective _).eq_iff, ite_apply]
  /-
    case h.inr.refine_3
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    ι : Type u_2
    E : ι → Type u_3
    inst✝³ : (i : ι) → NormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝¹ : Fintype ι
    f : ContinuousMultilinearMap 𝕜 E F
    x : (i : ι) → E i
    inst✝ : DecidableEq ι
    y : (i : ι) → E i
    h✝ : Nonempty ι
    i : ι
    ⊢ Eq (f (Function.update x i (y i))) (f fun i_1 => ite (Eq i_1 i) (y i_1) (x i …
  -/
  congr; ext j
  /-
    case h.inr.refine_3.h.e_6.h.h
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    ι : Type u_2
    E : ι → Type u_3
    inst✝³ : (i : ι) → NormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝¹ : Fintype ι
    f : ContinuousMultilinearMap 𝕜 E F
    x : (i : ι) → E i
    inst✝ : DecidableEq ι
    y : (i : ι) → E i
    h✝ : Nonempty ι
    i j : ι
    ⊢ Eq (Function.update x i (y i) j) (ite (Eq j i) (y j) (x j))
  -/
  obtain rfl | hj := eq_or_ne j i
    /-
      case h.inr.refine_3.h.e_6.h.h.inl
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      F : Type v
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      ι : Type u_2
      E : ι → Type u_3
      inst✝³ : (i : ι) → NormedAddCommGroup (E i)
      inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝¹ : Fintype ι
      f : ContinuousMultilinearMap 𝕜 E F
      x : (i : ι) → E i
      inst✝ : DecidableEq ι
      y : (i : ι) → E i
      h✝ : Nonempty ι
      j : ι
      ⊢ Eq (Function.update x j (y j) j) (ite (Eq j j) (y j) (x j))
    -/
  · rw [Function.update_self, if_pos rfl]
    /-
      🎉 no goals
    -/
    /-
      case h.inr.refine_3.h.e_6.h.h.inr
      𝕜 : Type u_1
      inst✝⁶ : NontriviallyNormedField 𝕜
      F : Type v
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      ι : Type u_2
      E : ι → Type u_3
      inst✝³ : (i : ι) → NormedAddCommGroup (E i)
      inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝¹ : Fintype ι
      f : ContinuousMultilinearMap 𝕜 E F
      x : (i : ι) → E i
      inst✝ : DecidableEq ι
      y : (i : ι) → E i
      h✝ : Nonempty ι
      i j : ι
      hj : Ne j i
      ⊢ Eq (Function.update x i (y i) j) (ite (Eq j i) (y j) (x j))
    -/
  · rw [Function.update_of_ne hj, if_neg hj]
    /-
      🎉 no goals
    -/


protected theorem hasFDerivAt [DecidableEq ι] : HasFDerivAt f (f.linearDeriv x) x := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    ι : Type u_2
    E : ι → Type u_3
    inst✝³ : (i : ι) → NormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝¹ : Fintype ι
    f : ContinuousMultilinearMap 𝕜 E F
    x : (i : ι) → E i
    inst✝ : DecidableEq ι
    ⊢ HasFDerivAt (⇑f) (f.linearDeriv x) x
  -/
  rw [← changeOrigin_toFormalMultilinearSeries]
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    ι : Type u_2
    E : ι → Type u_3
    inst✝³ : (i : ι) → NormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝¹ : Fintype ι
    f : ContinuousMultilinearMap 𝕜 E F
    x : (i : ι) → E i
    inst✝ : DecidableEq ι
    ⊢ HasFDerivAt (⇑f) ((continuousMultilinearCurryFin1 𝕜 ((i : ι) → E i) F) (f.to …
  -/
  convert f.hasFiniteFPowerSeriesOnBall.hasFDerivAt (y := x) ENNReal.coe_lt_top
  /-
    case h.e'_13
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    ι : Type u_2
    E : ι → Type u_3
    inst✝³ : (i : ι) → NormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝¹ : Fintype ι
    f : ContinuousMultilinearMap 𝕜 E F
    x : (i : ι) → E i
    inst✝ : DecidableEq ι
    ⊢ Eq x (HAdd.hAdd 0 x)
  -/
  rw [zero_add]
  /-
    🎉 no goals
  -/


/-- Given `f` a multilinear map, then the derivative of `x ↦ f (g₁ x, ..., gₙ x)` at `x` applied
to a vector `v` is given by `∑ i, f (g₁ x, ..., g'ᵢ v, ..., gₙ x)`. Version inside a set. -/
theorem _root_.HasFDerivWithinAt.multilinear_comp
    [DecidableEq ι] {G : Type*} [NormedAddCommGroup G] [NormedSpace 𝕜 G]
    {g : ∀ i, G → E i} {g' : ∀ i, G →L[𝕜] E i} {s : Set G} {x : G}
    (hg : ∀ i, HasFDerivWithinAt (g i) (g' i) s x) :
    HasFDerivWithinAt (fun x ↦ f (fun i ↦ g i x))
      ((∑ i : ι, (f.toContinuousLinearMap (fun j ↦ g j x) i) ∘L (g' i))) s x := by
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    ι : Type u_2
    E : ι → Type u_3
    inst✝⁵ : (i : ι) → NormedAddCommGroup (E i)
    inst✝⁴ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝³ : Fintype ι
    f : ContinuousMultilinearMap 𝕜 E F
    inst✝² : DecidableEq ι
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    g : (i : ι) → G → E i
    g' : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) G (E i)
    s : Set G
    x : G
    hg : ∀ (i : ι), HasFDerivWithinAt (g i) (g' i) s x
    ⊢ HasFDerivWithinAt (fun x => f fun i => g i x) (Finset.univ.sum fun i => (f.t …
  -/
  convert (f.hasFDerivAt (fun j ↦ g j x)).comp_hasFDerivWithinAt x (hasFDerivWithinAt_pi.2 hg)
  /-
    case h.e'_12
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    ι : Type u_2
    E : ι → Type u_3
    inst✝⁵ : (i : ι) → NormedAddCommGroup (E i)
    inst✝⁴ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝³ : Fintype ι
    f : ContinuousMultilinearMap 𝕜 E F
    inst✝² : DecidableEq ι
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    g : (i : ι) → G → E i
    g' : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) G (E i)
    s : Set G
    x : G
    hg : ∀ (i : ι), HasFDerivWithinAt (g i) (g' i) s x
    ⊢ Eq (Finset.univ.sum fun i => (f.toContinuousLinearMap (fun j => g j x) i).co …
  -/
  ext v
  /-
    case h.e'_12.h
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    ι : Type u_2
    E : ι → Type u_3
    inst✝⁵ : (i : ι) → NormedAddCommGroup (E i)
    inst✝⁴ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝³ : Fintype ι
    f : ContinuousMultilinearMap 𝕜 E F
    inst✝² : DecidableEq ι
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    g : (i : ι) → G → E i
    g' : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) G (E i)
    s : Set G
    x : G
    hg : ∀ (i : ι), HasFDerivWithinAt (g i) (g' i) s x
    v : G
    ⊢ Eq ((Finset.univ.sum fun i => (f.toContinuousLinearMap (fun j => g j x) i).c …
  -/
  simp [linearDeriv]
  /-
    🎉 no goals
  -/


/-- Given `f` a multilinear map, then the derivative of `x ↦ f (g₁ x, ..., gₙ x)` at `x` applied
to a vector `v` is given by `∑ i, f (g₁ x, ..., g'ᵢ v, ..., gₙ x)`. -/
theorem _root_.HasFDerivAt.multilinear_comp
    [DecidableEq ι] {G : Type*} [NormedAddCommGroup G] [NormedSpace 𝕜 G]
    {g : ∀ i, G → E i} {g' : ∀ i, G →L[𝕜] E i} {x : G}
    (hg : ∀ i, HasFDerivAt (g i) (g' i) x) :
    HasFDerivAt (fun x ↦ f (fun i ↦ g i x))
      ((∑ i : ι, (f.toContinuousLinearMap (fun j ↦ g j x) i) ∘L (g' i))) x := by
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    ι : Type u_2
    E : ι → Type u_3
    inst✝⁵ : (i : ι) → NormedAddCommGroup (E i)
    inst✝⁴ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝³ : Fintype ι
    f : ContinuousMultilinearMap 𝕜 E F
    inst✝² : DecidableEq ι
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    g : (i : ι) → G → E i
    g' : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) G (E i)
    x : G
    hg : ∀ (i : ι), HasFDerivAt (g i) (g' i) x
    ⊢ HasFDerivAt (fun x => f fun i => g i x) (Finset.univ.sum fun i => (f.toConti …
  -/
  convert (f.hasFDerivAt (fun j ↦ g j x)).comp x (hasFDerivAt_pi.2 hg)
  /-
    case h.e'_12
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    ι : Type u_2
    E : ι → Type u_3
    inst✝⁵ : (i : ι) → NormedAddCommGroup (E i)
    inst✝⁴ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝³ : Fintype ι
    f : ContinuousMultilinearMap 𝕜 E F
    inst✝² : DecidableEq ι
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    g : (i : ι) → G → E i
    g' : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) G (E i)
    x : G
    hg : ∀ (i : ι), HasFDerivAt (g i) (g' i) x
    ⊢ Eq (Finset.univ.sum fun i => (f.toContinuousLinearMap (fun j => g j x) i).co …
  -/
  ext v
  /-
    case h.e'_12.h
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    ι : Type u_2
    E : ι → Type u_3
    inst✝⁵ : (i : ι) → NormedAddCommGroup (E i)
    inst✝⁴ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝³ : Fintype ι
    f : ContinuousMultilinearMap 𝕜 E F
    inst✝² : DecidableEq ι
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    g : (i : ι) → G → E i
    g' : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) G (E i)
    x : G
    hg : ∀ (i : ι), HasFDerivAt (g i) (g' i) x
    v : G
    ⊢ Eq ((Finset.univ.sum fun i => (f.toContinuousLinearMap (fun j => g j x) i).c …
  -/
  simp [linearDeriv]
  /-
    🎉 no goals
  -/


/-- Technical lemma used in the proof of `hasFTaylorSeriesUpTo_iteratedFDeriv`, to compare sums
over embedding of `Fin k` and `Fin (k + 1)`. -/
private lemma _root_.Equiv.succ_embeddingFinSucc_fst_symm_apply {ι : Type*} [DecidableEq ι]
    {n : ℕ} (e : Fin (n+1) ↪ ι) {k : ι}
    (h'k : k ∈ Set.range (Equiv.embeddingFinSucc n ι e).1) (hk : k ∈ Set.range e) :
    Fin.succ ((Equiv.embeddingFinSucc n ι e).1.toEquivRange.symm ⟨k, h'k⟩)
      = e.toEquivRange.symm ⟨k, hk⟩ := by
  /-
    ι : Type u_4
    inst✝ : DecidableEq ι
    n : Nat
    e : Function.Embedding (Fin (HAdd.hAdd n 1)) ι
    k : ι
    h'k : Membership.mem (Set.range ⇑((Equiv.embeddingFinSucc n ι) e).fst) k
    hk : Membership.mem (Set.range ⇑e) k
    ⊢ Eq (((Equiv.embeddingFinSucc n ι) e).fst.toEquivRange.symm ⟨k, h'k⟩).succ (e …
  -/
  rcases hk with ⟨j, rfl⟩
  have hj : j ≠ 0 := by
    rintro rfl
    simp at h'k
  /-
    case intro
    ι : Type u_4
    inst✝ : DecidableEq ι
    n : Nat
    e : Function.Embedding (Fin (HAdd.hAdd n 1)) ι
    j : Fin (HAdd.hAdd n 1)
    h'k : Membership.mem (Set.range ⇑((Equiv.embeddingFinSucc n ι) e).fst) (e j)
    hj : Ne j 0
    ⊢ Eq (((Equiv.embeddingFinSucc n ι) e).fst.toEquivRange.symm ⟨e j, h'k⟩).succ  …
  -/
  simp only [Function.Embedding.toEquivRange_symm_apply_self]
  /-
    case intro
    ι : Type u_4
    inst✝ : DecidableEq ι
    n : Nat
    e : Function.Embedding (Fin (HAdd.hAdd n 1)) ι
    j : Fin (HAdd.hAdd n 1)
    h'k : Membership.mem (Set.range ⇑((Equiv.embeddingFinSucc n ι) e).fst) (e j)
    hj : Ne j 0
    ⊢ Eq (((Equiv.embeddingFinSucc n ι) e).fst.toEquivRange.symm ⟨e j, h'k⟩).succ j
  -/
  have : e j = (Equiv.embeddingFinSucc n ι e).1 (Fin.pred j hj) := by simp
  /-
    case intro
    ι : Type u_4
    inst✝ : DecidableEq ι
    n : Nat
    e : Function.Embedding (Fin (HAdd.hAdd n 1)) ι
    j : Fin (HAdd.hAdd n 1)
    h'k : Membership.mem (Set.range ⇑((Equiv.embeddingFinSucc n ι) e).fst) (e j)
    hj : Ne j 0
    this : Eq (e j) (((Equiv.embeddingFinSucc n ι) e).fst (j.pred hj))
    ⊢ Eq (((Equiv.embeddingFinSucc n ι) e).fst.toEquivRange.symm ⟨e j, h'k⟩).succ j
  -/
  simp_rw [this]
  /-
    case intro
    ι : Type u_4
    inst✝ : DecidableEq ι
    n : Nat
    e : Function.Embedding (Fin (HAdd.hAdd n 1)) ι
    j : Fin (HAdd.hAdd n 1)
    h'k : Membership.mem (Set.range ⇑((Equiv.embeddingFinSucc n ι) e).fst) (e j)
    hj : Ne j 0
    this : Eq (e j) (((Equiv.embeddingFinSucc n ι) e).fst (j.pred hj))
    ⊢ Eq (((Equiv.embeddingFinSucc n ι) e).fst.toEquivRange.symm ⟨((Equiv.embeddin …
  -/
  simp [-Equiv.embeddingFinSucc_fst]
  /-
    🎉 no goals
  -/


/-- A continuous multilinear function `f` admits a Taylor series, whose successive terms are given
by `f.iteratedFDeriv n`. This is the point of the definition of `f.iteratedFDeriv`. -/
theorem hasFTaylorSeriesUpTo_iteratedFDeriv :
    HasFTaylorSeriesUpTo ⊤ f (fun v n ↦ f.iteratedFDeriv n v) := by
  classical
  constructor
  · simp [ContinuousMultilinearMap.iteratedFDeriv]
  · rintro n - x
    suffices H : curryLeft (f.iteratedFDeriv (Nat.succ n) x) = (∑ e : Fin n ↪ ι,
          ((iteratedFDerivComponent f e.toEquivRange).linearDeriv
            (Pi.compRightL 𝕜 _ Subtype.val x)) ∘L (Pi.compRightL 𝕜 _ Subtype.val)) by
      have A : HasFDerivAt (f.iteratedFDeriv n) (∑ e : Fin n ↪ ι,
          ((iteratedFDerivComponent f e.toEquivRange).linearDeriv (Pi.compRightL 𝕜 _ Subtype.val x))
            ∘L (Pi.compRightL 𝕜 _ Subtype.val)) x := by
        apply HasFDerivAt.sum (fun s _hs ↦ ?_)
        exact (ContinuousMultilinearMap.hasFDerivAt _ _).comp x (ContinuousLinearMap.hasFDerivAt _)
      rwa [← H] at A
    ext v m
    simp only [ContinuousMultilinearMap.iteratedFDeriv, curryLeft_apply, sum_apply,
      iteratedFDerivComponent_apply, Finset.univ_sigma_univ,
      Pi.compRightL_apply, ContinuousLinearMap.coe_sum', ContinuousLinearMap.coe_comp',
      Finset.sum_apply, Function.comp_apply, linearDeriv_apply, Finset.sum_sigma']
    rw [← (Equiv.embeddingFinSucc n ι).sum_comp]
    congr with e
    congr with k
    by_cases hke : k ∈ Set.range e
    · simp only [hke, ↓reduceDIte]
      split_ifs with hkf
      · simp only [← Equiv.succ_embeddingFinSucc_fst_symm_apply e hkf hke, Fin.cons_succ]
      · obtain rfl : k = e 0 := by
          rcases hke with ⟨j, rfl⟩
          simpa using hkf
        simp only [Function.Embedding.toEquivRange_symm_apply_self, Fin.cons_zero, Function.update,
          Pi.compRightL_apply]
        split_ifs with h
        · congr!
        · exfalso
          apply h
          simp_rw [← Equiv.embeddingFinSucc_snd e]
    · have hkf : k ∉ Set.range (Equiv.embeddingFinSucc n ι e).1 := by
        contrapose! hke
        rw [Equiv.embeddingFinSucc_fst] at hke
        exact Set.range_comp_subset_range _ _ hke
      simp only [hke, hkf, ↓reduceDIte, Pi.compRightL,
        ContinuousLinearMap.coe_mk', LinearMap.coe_mk, AddHom.coe_mk]
      rw [Function.update_of_ne]
      contrapose! hke
      rw [show k = _ from Subtype.ext_iff_val.1 hke, Equiv.embeddingFinSucc_snd e]
      exact Set.mem_range_self _
  · rintro n -
    apply continuous_finset_sum _ (fun e _ ↦ ?_)
    exact (ContinuousMultilinearMap.coe_continuous _).comp (ContinuousLinearMap.continuous _)


theorem iteratedFDeriv_eq (n : ℕ) :
    iteratedFDeriv 𝕜 n f = f.iteratedFDeriv n :=
  funext fun x ↦ (f.hasFTaylorSeriesUpTo_iteratedFDeriv.eq_iteratedFDeriv (m := n) le_top x).symm


theorem norm_iteratedFDeriv_le (n : ℕ) (x : (i : ι) → E i) :
    ‖iteratedFDeriv 𝕜 n f x‖
      ≤ Nat.descFactorial (Fintype.card ι) n * ‖f‖ * ‖x‖ ^ (Fintype.card ι - n) := by
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    ι : Type u_2
    E : ι → Type u_3
    inst✝² : (i : ι) → NormedAddCommGroup (E i)
    inst✝¹ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝ : Fintype ι
    f : ContinuousMultilinearMap 𝕜 E F
    n : Nat
    x : (i : ι) → E i
    ⊢ LE.le (Norm.norm (iteratedFDeriv 𝕜 n (⇑f) x)) (HMul.hMul (HMul.hMul (↑((Fint …
  -/
  rw [f.iteratedFDeriv_eq]
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    F : Type v
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    ι : Type u_2
    E : ι → Type u_3
    inst✝² : (i : ι) → NormedAddCommGroup (E i)
    inst✝¹ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝ : Fintype ι
    f : ContinuousMultilinearMap 𝕜 E F
    n : Nat
    x : (i : ι) → E i
    ⊢ LE.le (Norm.norm (f.iteratedFDeriv n x)) (HMul.hMul (HMul.hMul (↑((Fintype.c …
  -/
  exact f.norm_iteratedFDeriv_le' n x
  /-
    🎉 no goals
  -/


lemma cPolynomialAt : CPolynomialAt 𝕜 f x :=
  f.hasFiniteFPowerSeriesOnBall.cPolynomialAt_of_mem
        /-
          𝕜 : Type u_1
          inst✝⁵ : NontriviallyNormedField 𝕜
          F : Type v
          inst✝⁴ : NormedAddCommGroup F
          inst✝³ : NormedSpace 𝕜 F
          ι : Type u_2
          E : ι → Type u_3
          inst✝² : (i : ι) → NormedAddCommGroup (E i)
          inst✝¹ : (i : ι) → NormedSpace 𝕜 (E i)
          inst✝ : Fintype ι
          f : ContinuousMultilinearMap 𝕜 E F
          x : (i : ι) → E i
          ⊢ Membership.mem (EMetric.ball 0 Top.top) x
        -/
    (by simp only [Metric.emetric_ball_top, Set.mem_univ])
        /-
          🎉 no goals
        -/


lemma cPolyomialOn : CPolynomialOn 𝕜 f ⊤ := fun x _ ↦ f.cPolynomialAt x


open Fintype ContinuousLinearMap in
theorem derivSeries_apply_diag (n : ℕ) (x : E) :
    derivSeries p n (fun _ ↦ x) x = (n + 1) • p (n + 1) fun _ ↦ x := by
  simp only [derivSeries, compFormalMultilinearSeries_apply, changeOriginSeries,
    compContinuousMultilinearMap_coe, ContinuousLinearEquiv.coe_coe, LinearIsometryEquiv.coe_coe,
    Function.comp_apply, ContinuousMultilinearMap.sum_apply, map_sum, coe_sum', Finset.sum_apply,
    continuousMultilinearCurryFin1_apply, Matrix.zero_empty]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    x : E
    ⊢ Eq (Finset.univ.sum fun x_1 => ((p.changeOriginSeriesTerm 1 n ↑x_1 ⋯) fun x_ …
  -/
  convert Finset.sum_const _
    /-
      case h.e'_2.a
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type v
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      n : Nat
      x : E
      x✝ : Subtype fun s => Eq s.card n
      a✝ : Membership.mem Finset.univ x✝
      ⊢ Eq (((p.changeOriginSeriesTerm 1 n ↑x✝ ⋯) fun x_1 => x) (Fin.snoc Matrix.vec …
    -/
  · rw [Fin.snoc_zero, changeOriginSeriesTerm_apply, Finset.piecewise_same, add_comm]
    /-
      🎉 no goals
    -/
  · rw [← card, card_subtype, ← Finset.powerset_univ, ← Finset.powersetCard_eq_filter,
      Finset.card_powersetCard, ← card, card_fin, eq_comm, add_comm, Nat.choose_succ_self_right]


include h in
theorem iteratedFDeriv_zero_apply_diag : iteratedFDeriv 𝕜 0 f x = p 0 := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    f : E → F
    x : E
    r : ENNReal
    h : HasFPowerSeriesOnBall f p x r
    ⊢ Eq (iteratedFDeriv 𝕜 0 f x) (p 0)
  -/
  ext
  /-
    case H
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    f : E → F
    x : E
    r : ENNReal
    h : HasFPowerSeriesOnBall f p x r
    x✝ : Fin 0 → E
    ⊢ Eq ((iteratedFDeriv 𝕜 0 f x) x✝) ((p 0) x✝)
  -/
  convert (h.hasSum <| EMetric.mem_ball_self h.r_pos).tsum_eq.symm
    /-
      case h.e'_2
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type v
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      f : E → F
      x : E
      r : ENNReal
      h : HasFPowerSeriesOnBall f p x r
      x✝ : Fin 0 → E
      ⊢ Eq ((iteratedFDeriv 𝕜 0 f x) x✝) (f (HAdd.hAdd x 0))
    -/
  · rw [iteratedFDeriv_zero_apply, add_zero]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type v
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      f : E → F
      x : E
      r : ENNReal
      h : HasFPowerSeriesOnBall f p x r
      x✝ : Fin 0 → E
      ⊢ Eq ((p 0) x✝) (tsum fun b => (p b) fun x => 0)
    -/
  · rw [tsum_eq_single 0 fun n hn ↦ by haveI := NeZero.mk hn; exact (p n).map_zero]
    /-
      case h.e'_3
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type v
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      f : E → F
      x : E
      r : ENNReal
      h : HasFPowerSeriesOnBall f p x r
      x✝ : Fin 0 → E
      ⊢ Eq ((p 0) x✝) ((p 0) fun x => 0)
    -/
    exact congr(p 0 $(Subsingleton.elim _ _))
    /-
      🎉 no goals
    -/


private theorem factorial_smul' {n : ℕ} : ∀ {F : Type max u v} [NormedAddCommGroup F]
    [NormedSpace 𝕜 F] [CompleteSpace F] {p : FormalMultilinearSeries 𝕜 E F}
    {f : E → F}, HasFPowerSeriesOnBall f p x r →
    n ! • p n (fun _ ↦ y) = iteratedFDeriv 𝕜 n f x (fun _ ↦ y) := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    r : ENNReal
    y : E
    n : Nat
    ⊢ ∀ {F : Type (max u v)} [inst : NormedAddCommGroup F] [inst_1 : NormedSpace 𝕜 …
  -/
  induction n with | zero => _ | succ n ih => _ <;> intro F _ _ _ p f h
    /-
      case zero
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      x : E
      r : ENNReal
      y : E
      F : Type (max u v)
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : CompleteSpace F
      p : FormalMultilinearSeries 𝕜 E F
      f : E → F
      h : HasFPowerSeriesOnBall f p x r
      ⊢ Eq (HSMul.hSMul (Nat.factorial 0) ((p 0) fun x => y)) ((iteratedFDeriv 𝕜 0 f …
    -/
  · rw [factorial_zero, one_smul, h.iteratedFDeriv_zero_apply_diag]
    /-
      🎉 no goals
    -/
  · rw [factorial_succ, mul_comm, mul_smul, ← derivSeries_apply_diag,
      ← ContinuousLinearMap.smul_apply, ih h.fderiv, iteratedFDeriv_succ_apply_right]
    /-
      case succ
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      x : E
      r : ENNReal
      y : E
      n : Nat
      ih : ∀ {F : Type (max u v)} [inst : NormedAddCommGroup F] [inst_1 : NormedSpac …
      F : Type (max u v)
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      inst✝ : CompleteSpace F
      p : FormalMultilinearSeries 𝕜 E F
      f : E → F
      h : HasFPowerSeriesOnBall f p x r
      ⊢ Eq (((iteratedFDeriv 𝕜 n (fderiv 𝕜 f) x) fun x => y) y) (((iteratedFDeriv 𝕜  …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- The iterated derivative of an analytic function, on vectors `(y, ..., y)`, is given by `n!`
times the `n`-th term in the power series. For a more general result giving the full iterated
derivative as a sum over the permutations of `Fin n`, see
`HasFPowerSeriesOnBall.iteratedFDeriv_eq_sum`. -/
theorem factorial_smul (n : ℕ) :
    n ! • p n (fun _ ↦ y) = iteratedFDeriv 𝕜 n f x (fun _ ↦ y) := by
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type v
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    f : E → F
    x : E
    r : ENNReal
    h : HasFPowerSeriesOnBall f p x r
    y : E
    inst✝ : CompleteSpace F
    n : Nat
    ⊢ Eq (HSMul.hSMul n.factorial ((p n) fun x => y)) ((iteratedFDeriv 𝕜 n f x) fu …
  -/
  cases n
    /-
      case zero
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type v
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      f : E → F
      x : E
      r : ENNReal
      h : HasFPowerSeriesOnBall f p x r
      y : E
      inst✝ : CompleteSpace F
      ⊢ Eq (HSMul.hSMul (Nat.factorial 0) ((p 0) fun x => y)) ((iteratedFDeriv 𝕜 0 f …
    -/
  · rw [factorial_zero, one_smul, h.iteratedFDeriv_zero_apply_diag]
    /-
      🎉 no goals
    -/
  · rw [factorial_succ, mul_comm, mul_smul, ← derivSeries_apply_diag,
      ← ContinuousLinearMap.smul_apply, factorial_smul' _ h.fderiv, iteratedFDeriv_succ_apply_right]
    /-
      case succ
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type v
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      p : FormalMultilinearSeries 𝕜 E F
      f : E → F
      x : E
      r : ENNReal
      h : HasFPowerSeriesOnBall f p x r
      y : E
      inst✝ : CompleteSpace F
      n✝ : Nat
      ⊢ Eq (((iteratedFDeriv 𝕜 n✝ (fderiv 𝕜 f) x) fun x => y) y) (((iteratedFDeriv 𝕜 …
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem hasSum_iteratedFDeriv [CharZero 𝕜] {y : E} (hy : y ∈ EMetric.ball 0 r) :
    HasSum (fun n ↦ (n ! : 𝕜)⁻¹ • iteratedFDeriv 𝕜 n f x fun _ ↦ y) (f (x + y)) := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type v
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    f : E → F
    x : E
    r : ENNReal
    h : HasFPowerSeriesOnBall f p x r
    inst✝¹ : CompleteSpace F
    inst✝ : CharZero 𝕜
    y : E
    hy : Membership.mem (EMetric.ball 0 r) y
    ⊢ HasSum (fun n => HSMul.hSMul (Inv.inv ↑n.factorial) ((iteratedFDeriv 𝕜 n f x …
  -/
  convert h.hasSum hy with n
  rw [← h.factorial_smul y n, smul_comm, ← smul_assoc, nsmul_eq_mul,
    mul_inv_cancel₀ <| cast_ne_zero.mpr n.factorial_ne_zero, one_smul]


theorem hasFDerivAt_uncurry_of_multilinear [DecidableEq ι]
    (f : E →L[𝕜] ContinuousMultilinearMap 𝕜 G F) (v : E × Π i, G i) :
    HasFDerivAt (fun (p : E × Π i, G i) ↦ f p.1 p.2)
      ((f.flipMultilinear v.2) ∘L (.fst _ _ _) +
        ∑ i : ι, ((f v.1).toContinuousLinearMap v.2 i) ∘L (.proj _) ∘L (.snd _ _ _)) v := by
  convert HasFDerivAt.multilinear_comp (f.continuousMultilinearMapOption)
    (g := fun (_ : Option ι) p ↦ p) (g' := fun _ ↦ ContinuousLinearMap.id _ _) (x := v)
    (fun _ ↦ hasFDerivAt_id _)
  have I : f.continuousMultilinearMapOption.toContinuousLinearMap (fun _ ↦ v) none =
      (f.flipMultilinear v.2) ∘L (.fst _ _ _) := by
    simp [ContinuousMultilinearMap.toContinuousLinearMap, continuousMultilinearMapOption]
    apply ContinuousLinearMap.ext (fun w ↦ ?_)
    simp
  have J : ∀ (i : ι), f.continuousMultilinearMapOption.toContinuousLinearMap (fun _ ↦ v) (some i)
      = ((f v.1).toContinuousLinearMap v.2 i) ∘L (.proj _) ∘L (.snd _ _ _) := by
    intro i
    apply ContinuousLinearMap.ext (fun w ↦ ?_)
    simp only [ContinuousMultilinearMap.toContinuousLinearMap, continuousMultilinearMapOption,
      coe_mk', MultilinearMap.toLinearMap_apply, ContinuousMultilinearMap.coe_coe,
      MultilinearMap.coe_mkContinuous, MultilinearMap.coe_mk, ne_eq, reduceCtorEq,
      not_false_eq_true, Function.update_of_ne, coe_comp', coe_snd', Function.comp_apply,
      proj_apply]
    congr
    ext j
    rcases eq_or_ne j i with rfl | hij
    · simp
    · simp [hij]
  /-
    case h.e'_12.h.h.h
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    F : Type v
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    ι : Type u_2
    G : ι → Type u_3
    inst✝³ : (i : ι) → NormedAddCommGroup (G i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (G i)
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    f : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousMultilinearMap 𝕜 G F)
    v : Prod E ((i : ι) → G i)
    e_4✝ : Eq Prod.instAddCommGroup SeminormedAddCommGroup.toAddCommGroup
    he✝ : Eq Prod.instModule NormedSpace.toModule
    e_6✝ : Eq instTopologicalSpaceProd UniformSpace.toTopologicalSpace
    e_8✝ : Eq NormedAddCommGroup.toAddCommGroup SeminormedAddCommGroup.toAddCommGr …
    I : Eq (f.continuousMultilinearMapOption.toContinuousLinearMap (fun x => v) Op …
    J : ∀ (i : ι), Eq (f.continuousMultilinearMapOption.toContinuousLinearMap (fun …
    ⊢ Eq (HAdd.hAdd ((f.flipMultilinear v.2).comp (ContinuousLinearMap.fst 𝕜 E ((i …
  -/
  simp [I, J]
  /-
    🎉 no goals
  -/


/-- Given `f` a linear map into multilinear maps, then the derivative
of `x ↦ f (a x) (b₁ x, ..., bₙ x)` at `x` applied to a vector `v` is given by
`f (a' v) (b₁ x, ...., bₙ x) + ∑ i, f a (b₁ x, ..., b'ᵢ v, ..., bₙ x)`. Version inside a set. -/
theorem _root_.HasFDerivWithinAt.linear_multilinear_comp
    [DecidableEq ι] {a : H → E} {a' : H →L[𝕜] E}
    {b : ∀ i, H → G i} {b' : ∀ i, H →L[𝕜] G i} {s : Set H} {x : H}
    (ha : HasFDerivWithinAt a a' s x) (hb : ∀ i, HasFDerivWithinAt (b i) (b' i) s x)
    (f : E →L[𝕜] ContinuousMultilinearMap 𝕜 G F) :
    HasFDerivWithinAt (fun y ↦ f (a y) (fun i ↦ b i y))
      ((f.flipMultilinear (fun i ↦ b i x)) ∘L a' +
        ∑ i, ((f (a x)).toContinuousLinearMap (fun j ↦ b j x) i) ∘L (b' i)) s x := by
  convert (hasFDerivAt_uncurry_of_multilinear f (a x, fun i ↦ b i x)).comp_hasFDerivWithinAt x
    (ha.prod (hasFDerivWithinAt_pi.mpr hb))
  /-
    case h.e'_12
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    F : Type v
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    ι : Type u_2
    G : ι → Type u_3
    inst✝⁵ : (i : ι) → NormedAddCommGroup (G i)
    inst✝⁴ : (i : ι) → NormedSpace 𝕜 (G i)
    inst✝³ : Fintype ι
    H : Type u_4
    inst✝² : NormedAddCommGroup H
    inst✝¹ : NormedSpace 𝕜 H
    inst✝ : DecidableEq ι
    a : H → E
    a' : ContinuousLinearMap (RingHom.id 𝕜) H E
    b : (i : ι) → H → G i
    b' : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) H (G i)
    s : Set H
    x : H
    ha : HasFDerivWithinAt a a' s x
    hb : ∀ (i : ι), HasFDerivWithinAt (b i) (b' i) s x
    f : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousMultilinearMap 𝕜 G F)
    ⊢ Eq (HAdd.hAdd ((f.flipMultilinear fun i => b i x).comp a') (Finset.univ.sum  …
  -/
  ext v
  /-
    case h.e'_12.h
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    F : Type v
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    ι : Type u_2
    G : ι → Type u_3
    inst✝⁵ : (i : ι) → NormedAddCommGroup (G i)
    inst✝⁴ : (i : ι) → NormedSpace 𝕜 (G i)
    inst✝³ : Fintype ι
    H : Type u_4
    inst✝² : NormedAddCommGroup H
    inst✝¹ : NormedSpace 𝕜 H
    inst✝ : DecidableEq ι
    a : H → E
    a' : ContinuousLinearMap (RingHom.id 𝕜) H E
    b : (i : ι) → H → G i
    b' : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) H (G i)
    s : Set H
    x : H
    ha : HasFDerivWithinAt a a' s x
    hb : ∀ (i : ι), HasFDerivWithinAt (b i) (b' i) s x
    f : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousMultilinearMap 𝕜 G F)
    v : H
    ⊢ Eq ((HAdd.hAdd ((f.flipMultilinear fun i => b i x).comp a') (Finset.univ.sum …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Given `f` a linear map into multilinear maps, then the derivative
of `x ↦ f (a x) (b₁ x, ..., bₙ x)` at `x` applied to a vector `v` is given by
`f (a' v) (b₁ x, ...., bₙ x) + ∑ i, f a (b₁ x, ..., b'ᵢ v, ..., bₙ x)`. -/
theorem _root_.HasFDerivAt.linear_multilinear_comp [DecidableEq ι] {a : H → E} {a' : H →L[𝕜] E}
    {b : ∀ i, H → G i} {b' : ∀ i, H →L[𝕜] G i} {x : H}
    (ha : HasFDerivAt a a' x) (hb : ∀ i, HasFDerivAt (b i) (b' i) x)
    (f : E →L[𝕜] ContinuousMultilinearMap 𝕜 G F) :
    HasFDerivAt (fun y ↦ f (a y) (fun i ↦ b i y))
      ((f.flipMultilinear (fun i ↦ b i x)) ∘L a' +
        ∑ i, ((f (a x)).toContinuousLinearMap (fun j ↦ b j x) i) ∘L (b' i)) x := by
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    F : Type v
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    ι : Type u_2
    G : ι → Type u_3
    inst✝⁵ : (i : ι) → NormedAddCommGroup (G i)
    inst✝⁴ : (i : ι) → NormedSpace 𝕜 (G i)
    inst✝³ : Fintype ι
    H : Type u_4
    inst✝² : NormedAddCommGroup H
    inst✝¹ : NormedSpace 𝕜 H
    inst✝ : DecidableEq ι
    a : H → E
    a' : ContinuousLinearMap (RingHom.id 𝕜) H E
    b : (i : ι) → H → G i
    b' : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) H (G i)
    x : H
    ha : HasFDerivAt a a' x
    hb : ∀ (i : ι), HasFDerivAt (b i) (b' i) x
    f : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousMultilinearMap 𝕜 G F)
    ⊢ HasFDerivAt (fun y => (f (a y)) fun i => b i y) (HAdd.hAdd ((f.flipMultiline …
  -/
  simp_rw [← hasFDerivWithinAt_univ] at ha hb ⊢
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    F : Type v
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    ι : Type u_2
    G : ι → Type u_3
    inst✝⁵ : (i : ι) → NormedAddCommGroup (G i)
    inst✝⁴ : (i : ι) → NormedSpace 𝕜 (G i)
    inst✝³ : Fintype ι
    H : Type u_4
    inst✝² : NormedAddCommGroup H
    inst✝¹ : NormedSpace 𝕜 H
    inst✝ : DecidableEq ι
    a : H → E
    a' : ContinuousLinearMap (RingHom.id 𝕜) H E
    b : (i : ι) → H → G i
    b' : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) H (G i)
    x : H
    f : ContinuousLinearMap (RingHom.id 𝕜) E (ContinuousMultilinearMap 𝕜 G F)
    ha : HasFDerivWithinAt a a' Set.univ x
    hb : ∀ (i : ι), HasFDerivWithinAt (b i) (b' i) Set.univ x
    ⊢ HasFDerivWithinAt (fun y => (f (a y)) fun i => b i y) (HAdd.hAdd ((f.flipMul …
  -/
  exact HasFDerivWithinAt.linear_multilinear_comp ha hb f
  /-
    🎉 no goals
  -/


