/-- The norm on a real normed space is convex on any convex set. See also `Seminorm.convexOn`
and `convexOn_univ_norm`. -/
theorem convexOn_norm (hs : Convex ℝ s) : ConvexOn ℝ s norm :=
  ⟨hs, fun x _ y _ a b ha hb _ =>
    calc
      ‖a • x + b • y‖ ≤ ‖a • x‖ + ‖b • y‖ := norm_add_le _ _
      _ = a * ‖x‖ + b * ‖y‖ := by
        /-
          E : Type u_1
          inst✝¹ : SeminormedAddCommGroup E
          inst✝ : NormedSpace Real E
          s : Set E
          hs : Convex Real s
          x : E
          x✝² : Membership.mem s x
          y : E
          x✝¹ : Membership.mem s y
          a b : Real
          ha : LE.le 0 a
          hb : LE.le 0 b
          x✝ : Eq (HAdd.hAdd a b) 1
          ⊢ Eq (HAdd.hAdd (Norm.norm (HSMul.hSMul a x)) (Norm.norm (HSMul.hSMul b y))) ( …
        -/
        rw [norm_smul, norm_smul, Real.norm_of_nonneg ha, Real.norm_of_nonneg hb]⟩
        /-
          🎉 no goals
        -/


/-- The norm on a real normed space is convex on the whole space. See also `Seminorm.convexOn`
and `convexOn_norm`. -/
theorem convexOn_univ_norm : ConvexOn ℝ univ (norm : E → ℝ) :=
  convexOn_norm convex_univ


theorem convexOn_dist (z : E) (hs : Convex ℝ s) : ConvexOn ℝ s fun z' => dist z' z := by
  simpa [dist_eq_norm, preimage_preimage] using
    (convexOn_norm (hs.translate (-z))).comp_affineMap (AffineMap.id ℝ E - AffineMap.const ℝ E z)


theorem convexOn_univ_dist (z : E) : ConvexOn ℝ univ fun z' => dist z' z :=
  convexOn_dist z convex_univ


theorem convex_ball (a : E) (r : ℝ) : Convex ℝ (Metric.ball a r) := by
  /-
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a : E
    r : Real
    ⊢ Convex Real (Metric.ball a r)
  -/
  simpa only [Metric.ball, sep_univ] using (convexOn_univ_dist a).convex_lt r
  /-
    🎉 no goals
  -/


theorem convex_closedBall (a : E) (r : ℝ) : Convex ℝ (Metric.closedBall a r) := by
  /-
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    a : E
    r : Real
    ⊢ Convex Real (Metric.closedBall a r)
  -/
  simpa only [Metric.closedBall, sep_univ] using (convexOn_univ_dist a).convex_le r
  /-
    🎉 no goals
  -/


theorem Convex.thickening (hs : Convex ℝ s) (δ : ℝ) : Convex ℝ (thickening δ s) := by
  /-
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set E
    hs : Convex Real s
    δ : Real
    ⊢ Convex Real (Metric.thickening δ s)
  -/
  rw [← add_ball_zero]
  /-
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set E
    hs : Convex Real s
    δ : Real
    ⊢ Convex Real (HAdd.hAdd s (Metric.ball 0 δ))
  -/
  exact hs.add (convex_ball 0 _)
  /-
    🎉 no goals
  -/


theorem Convex.cthickening (hs : Convex ℝ s) (δ : ℝ) : Convex ℝ (cthickening δ s) := by
  /-
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set E
    hs : Convex Real s
    δ : Real
    ⊢ Convex Real (Metric.cthickening δ s)
  -/
  obtain hδ | hδ := le_total 0 δ
    /-
      case inl
      E : Type u_1
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace Real E
      s : Set E
      hs : Convex Real s
      δ : Real
      hδ : LE.le 0 δ
      ⊢ Convex Real (Metric.cthickening δ s)
    -/
  · rw [cthickening_eq_iInter_thickening hδ]
    /-
      case inl
      E : Type u_1
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace Real E
      s : Set E
      hs : Convex Real s
      δ : Real
      hδ : LE.le 0 δ
      ⊢ Convex Real (Set.iInter fun ε => Set.iInter fun x => Metric.thickening ε s)
    -/
    exact convex_iInter₂ fun _ _ => hs.thickening _
    /-
      🎉 no goals
    -/
    /-
      case inr
      E : Type u_1
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace Real E
      s : Set E
      hs : Convex Real s
      δ : Real
      hδ : LE.le δ 0
      ⊢ Convex Real (Metric.cthickening δ s)
    -/
  · rw [cthickening_of_nonpos hδ]
    /-
      case inr
      E : Type u_1
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace Real E
      s : Set E
      hs : Convex Real s
      δ : Real
      hδ : LE.le δ 0
      ⊢ Convex Real (closure s)
    -/
    exact hs.closure
    /-
      🎉 no goals
    -/


/-- Given a point `x` in the convex hull of `s` and a point `y`, there exists a point
of `s` at distance at least `dist x y` from `y`. -/
theorem convexHull_exists_dist_ge {s : Set E} {x : E} (hx : x ∈ convexHull ℝ s) (y : E) :
    ∃ x' ∈ s, dist x y ≤ dist x' y :=
  (convexOn_dist y (convex_convexHull ℝ _)).exists_ge_of_mem_convexHull (subset_convexHull ..) hx


/-- Given a point `x` in the convex hull of `s` and a point `y` in the convex hull of `t`,
there exist points `x' ∈ s` and `y' ∈ t` at distance at least `dist x y`. -/
theorem convexHull_exists_dist_ge2 {s t : Set E} {x y : E} (hx : x ∈ convexHull ℝ s)
    (hy : y ∈ convexHull ℝ t) : ∃ x' ∈ s, ∃ y' ∈ t, dist x y ≤ dist x' y' := by
  /-
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s t : Set E
    x y : E
    hx : Membership.mem ((convexHull Real) s) x
    hy : Membership.mem ((convexHull Real) t) y
    ⊢ Exists fun x' => And (Membership.mem s x') (Exists fun y' => And (Membership …
  -/
  rcases convexHull_exists_dist_ge hx y with ⟨x', hx', Hx'⟩
  /-
    case intro.intro
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s t : Set E
    x y : E
    hx : Membership.mem ((convexHull Real) s) x
    hy : Membership.mem ((convexHull Real) t) y
    x' : E
    hx' : Membership.mem s x'
    Hx' : LE.le (Dist.dist x y) (Dist.dist x' y)
    ⊢ Exists fun x' => And (Membership.mem s x') (Exists fun y' => And (Membership …
  -/
  rcases convexHull_exists_dist_ge hy x' with ⟨y', hy', Hy'⟩
  /-
    case intro.intro.intro.intro
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s t : Set E
    x y : E
    hx : Membership.mem ((convexHull Real) s) x
    hy : Membership.mem ((convexHull Real) t) y
    x' : E
    hx' : Membership.mem s x'
    Hx' : LE.le (Dist.dist x y) (Dist.dist x' y)
    y' : E
    hy' : Membership.mem t y'
    Hy' : LE.le (Dist.dist y x') (Dist.dist y' x')
    ⊢ Exists fun x' => And (Membership.mem s x') (Exists fun y' => And (Membership …
  -/
  use x', hx', y', hy'
  /-
    case right
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s t : Set E
    x y : E
    hx : Membership.mem ((convexHull Real) s) x
    hy : Membership.mem ((convexHull Real) t) y
    x' : E
    hx' : Membership.mem s x'
    Hx' : LE.le (Dist.dist x y) (Dist.dist x' y)
    y' : E
    hy' : Membership.mem t y'
    Hy' : LE.le (Dist.dist y x') (Dist.dist y' x')
    ⊢ LE.le (Dist.dist x y) (Dist.dist x' y')
  -/
  exact le_trans Hx' (dist_comm y x' ▸ dist_comm y' x' ▸ Hy')
  /-
    🎉 no goals
  -/


/-- Emetric diameter of the convex hull of a set `s` equals the emetric diameter of `s`. -/
@[simp]
theorem convexHull_ediam (s : Set E) : EMetric.diam (convexHull ℝ s) = EMetric.diam s := by
  /-
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set E
    ⊢ Eq (EMetric.diam ((convexHull Real) s)) (EMetric.diam s)
  -/
  refine (EMetric.diam_le fun x hx y hy => ?_).antisymm (EMetric.diam_mono <| subset_convexHull ℝ s)
  /-
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set E
    x : E
    hx : Membership.mem ((convexHull Real) s) x
    y : E
    hy : Membership.mem ((convexHull Real) s) y
    ⊢ LE.le (EDist.edist x y) (EMetric.diam s)
  -/
  rcases convexHull_exists_dist_ge2 hx hy with ⟨x', hx', y', hy', H⟩
  /-
    case intro.intro.intro.intro
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set E
    x : E
    hx : Membership.mem ((convexHull Real) s) x
    y : E
    hy : Membership.mem ((convexHull Real) s) y
    x' : E
    hx' : Membership.mem s x'
    y' : E
    hy' : Membership.mem s y'
    H : LE.le (Dist.dist x y) (Dist.dist x' y')
    ⊢ LE.le (EDist.edist x y) (EMetric.diam s)
  -/
  rw [edist_dist]
  /-
    case intro.intro.intro.intro
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set E
    x : E
    hx : Membership.mem ((convexHull Real) s) x
    y : E
    hy : Membership.mem ((convexHull Real) s) y
    x' : E
    hx' : Membership.mem s x'
    y' : E
    hy' : Membership.mem s y'
    H : LE.le (Dist.dist x y) (Dist.dist x' y')
    ⊢ LE.le (ENNReal.ofReal (Dist.dist x y)) (EMetric.diam s)
  -/
  apply le_trans (ENNReal.ofReal_le_ofReal H)
  /-
    case intro.intro.intro.intro
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set E
    x : E
    hx : Membership.mem ((convexHull Real) s) x
    y : E
    hy : Membership.mem ((convexHull Real) s) y
    x' : E
    hx' : Membership.mem s x'
    y' : E
    hy' : Membership.mem s y'
    H : LE.le (Dist.dist x y) (Dist.dist x' y')
    ⊢ LE.le (ENNReal.ofReal (Dist.dist x' y')) (EMetric.diam s)
  -/
  rw [← edist_dist]
  /-
    case intro.intro.intro.intro
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set E
    x : E
    hx : Membership.mem ((convexHull Real) s) x
    y : E
    hy : Membership.mem ((convexHull Real) s) y
    x' : E
    hx' : Membership.mem s x'
    y' : E
    hy' : Membership.mem s y'
    H : LE.le (Dist.dist x y) (Dist.dist x' y')
    ⊢ LE.le (EDist.edist x' y') (EMetric.diam s)
  -/
  exact EMetric.edist_le_diam_of_mem hx' hy'
  /-
    🎉 no goals
  -/


/-- Diameter of the convex hull of a set `s` equals the emetric diameter of `s`. -/
@[simp]
theorem convexHull_diam (s : Set E) : Metric.diam (convexHull ℝ s) = Metric.diam s := by
  /-
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set E
    ⊢ Eq (Metric.diam ((convexHull Real) s)) (Metric.diam s)
  -/
  simp only [Metric.diam, convexHull_ediam]
  /-
    🎉 no goals
  -/


/-- Convex hull of `s` is bounded if and only if `s` is bounded. -/
@[simp]
theorem isBounded_convexHull {s : Set E} :
    Bornology.IsBounded (convexHull ℝ s) ↔ Bornology.IsBounded s := by
  /-
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set E
    ⊢ Iff (Bornology.IsBounded ((convexHull Real) s)) (Bornology.IsBounded s)
  -/
  simp only [Metric.isBounded_iff_ediam_ne_top, convexHull_ediam]
  /-
    🎉 no goals
  -/


instance (priority := 100) NormedSpace.instPathConnectedSpace : PathConnectedSpace E :=
  TopologicalAddGroup.pathConnectedSpace


theorem Wbtw.dist_add_dist {x y z : P} (h : Wbtw ℝ x y z) :
    dist x y + dist y z = dist x z := by
  /-
    E : Type u_1
    P : Type u_2
    inst✝³ : SeminormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor E P
    x y z : P
    h : Wbtw Real x y z
    ⊢ Eq (HAdd.hAdd (Dist.dist x y) (Dist.dist y z)) (Dist.dist x z)
  -/
  obtain ⟨a, ⟨ha₀, ha₁⟩, rfl⟩ := h
  /-
    case intro.intro.intro
    E : Type u_1
    P : Type u_2
    inst✝³ : SeminormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor E P
    x z : P
    a : Real
    ha₀ : LE.le 0 a
    ha₁ : LE.le a 1
    ⊢ Eq (HAdd.hAdd (Dist.dist x ((AffineMap.lineMap x z) a)) (Dist.dist ((AffineM …
  -/
  simp [abs_of_nonneg, ha₀, ha₁, sub_mul]
  /-
    🎉 no goals
  -/


theorem dist_add_dist_of_mem_segment {x y z : E} (h : y ∈ [x -[ℝ] z]) :
    dist x y + dist y z = dist x z :=
  (mem_segment_iff_wbtw.1 h).dist_add_dist


/-- The set of vectors in the same ray as `x` is connected. -/
theorem isConnected_setOf_sameRay (x : E) : IsConnected { y | SameRay ℝ x y } := by
  /-
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x : E
    ⊢ IsConnected (setOf fun y => SameRay Real x y)
  -/
  by_cases hx : x = 0; · simpa [hx] using isConnected_univ (α := E)
                         /-
                           🎉 no goals
                         -/
  /-
    case neg
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x : E
    hx : Not (Eq x 0)
    ⊢ IsConnected (setOf fun y => SameRay Real x y)
  -/
  simp_rw [← exists_nonneg_left_iff_sameRay hx]
  /-
    case neg
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x : E
    hx : Not (Eq x 0)
    ⊢ IsConnected (setOf fun y => Exists fun r => And (LE.le 0 r) (Eq (HSMul.hSMul …
  -/
  exact isConnected_Ici.image _ (continuous_id.smul continuous_const).continuousOn
  /-
    🎉 no goals
  -/


/-- The set of nonzero vectors in the same ray as the nonzero vector `x` is connected. -/
theorem isConnected_setOf_sameRay_and_ne_zero {x : E} (hx : x ≠ 0) :
    IsConnected { y | SameRay ℝ x y ∧ y ≠ 0 } := by
  /-
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x : E
    hx : Ne x 0
    ⊢ IsConnected (setOf fun y => And (SameRay Real x y) (Ne y 0))
  -/
  simp_rw [← exists_pos_left_iff_sameRay_and_ne_zero hx]
  /-
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x : E
    hx : Ne x 0
    ⊢ IsConnected (setOf fun y => Exists fun r => And (LT.lt 0 r) (Eq (HSMul.hSMul …
  -/
  exact isConnected_Ioi.image _ (continuous_id.smul continuous_const).continuousOn
  /-
    🎉 no goals
  -/


/-- We can intercalate a simplex between a point and one of its neighborhoods. -/
lemma exists_mem_interior_convexHull_affineBasis (hs : s ∈ 𝓝 x) :
    ∃ b : AffineBasis (Fin (finrank ℝ E + 1)) ℝ E,
      x ∈ interior (convexHull ℝ (range b)) ∧ convexHull ℝ (range b) ⊆ s := by
  classical
  -- By translating, WLOG `x` is the origin.
  wlog hx : x = 0
  · obtain ⟨b, hb⟩ := this (s := -x +ᵥ s) (by simpa using vadd_mem_nhds_vadd (-x) hs) rfl
    use x +ᵥ b
    simpa [subset_set_vadd_iff, mem_vadd_set_iff_neg_vadd_mem, convexHull_vadd, interior_vadd,
      Pi.vadd_def, -vadd_eq_add, vadd_eq_add (a := -x), ← Set.vadd_set_range] using hb
  subst hx
  -- The strategy is now to find an arbitrary maximal spanning simplex (aka an affine basis)...
  obtain ⟨b⟩ := exists_affineBasis_of_finiteDimensional
    (ι := Fin (finrank ℝ E + 1)) (k := ℝ) (P := E) (by simp)
  -- ... translate it to contain the origin...
  set c : AffineBasis (Fin (finrank ℝ E + 1)) ℝ E := -Finset.univ.centroid ℝ b +ᵥ b
  have hc₀ : 0 ∈ interior (convexHull ℝ (range c) : Set E) := by
    simpa [c, convexHull_vadd, interior_vadd, range_add, Pi.vadd_def, mem_vadd_set_iff_neg_vadd_mem]
      using b.centroid_mem_interior_convexHull
  set cnorm := Finset.univ.sup' Finset.univ_nonempty (fun i ↦ ‖c i‖)
  have hcnorm : range c ⊆ closedBall 0 (cnorm + 1) := by
    simpa only [cnorm, subset_def, Finset.mem_coe, mem_closedBall, dist_zero_right,
      ← sub_le_iff_le_add, Finset.le_sup'_iff, forall_mem_range] using fun i ↦ ⟨i, by simp⟩
  -- ... and finally scale it to fit inside the neighborhood `s`.
  obtain ⟨ε, hε, hεs⟩ := Metric.mem_nhds_iff.1 hs
  set ε' : ℝ := ε / 2 / (cnorm + 1)
  have hc' : 0 < cnorm + 1 := by
    have : 0 ≤ cnorm := Finset.le_sup'_of_le _ (Finset.mem_univ 0) (norm_nonneg _)
    positivity
  have hε' : 0 < ε' := by positivity
  set d : AffineBasis (Fin (finrank ℝ E + 1)) ℝ E := Units.mk0 ε' hε'.ne' • c
  have hε₀ : 0 < ε / 2 := by positivity
  have hdnorm : (range d : Set E) ⊆ closedBall 0 (ε / 2) := by
    simp [d, Set.set_smul_subset_iff₀ hε'.ne', hε₀.le, _root_.smul_closedBall, abs_of_nonneg hε'.le,
      range_subset_iff, norm_smul]
    simpa [ε', hε₀.ne', range_subset_iff, ← mul_div_right_comm (ε / 2), div_le_iff₀ hc',
      mul_le_mul_left hε₀] using hcnorm
  refine ⟨d, ?_, ?_⟩
  · simpa [d, Pi.smul_def, range_smul, interior_smul₀, convexHull_smul, zero_mem_smul_set_iff,
      hε'.ne']
  · calc
      convexHull ℝ (range d) ⊆ closedBall 0 (ε / 2) := convexHull_min hdnorm (convex_closedBall ..)
      _ ⊆ ball 0 ε := closedBall_subset_ball (by linarith)
      _ ⊆ s := hεs


