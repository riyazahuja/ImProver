lemma closedBall_eq_segment (hε : 0 ≤ ε) : closedBall r ε = segment ℝ (r - ε) (r + ε) := by
  /-
    r ε : Real
    hε : LE.le 0 ε
    ⊢ Eq (Metric.closedBall r ε) (segment Real (HSub.hSub r ε) (HAdd.hAdd r ε))
  -/
  rw [closedBall_eq_Icc, segment_eq_Icc ((sub_le_self _ hε).trans <| le_add_of_nonneg_right hε)]
  /-
    🎉 no goals
  -/


lemma ball_eq_openSegment (hε : 0 < ε) : ball r ε = openSegment ℝ (r - ε) (r + ε) := by
  /-
    r ε : Real
    hε : LT.lt 0 ε
    ⊢ Eq (Metric.ball r ε) (openSegment Real (HSub.hSub r ε) (HAdd.hAdd r ε))
  -/
  rw [ball_eq_Ioo, openSegment_eq_Ioo ((sub_lt_self _ hε).trans <| lt_add_of_pos_right _ hε)]
  /-
    🎉 no goals
  -/


theorem convex_iff_isPreconnected : Convex ℝ s ↔ IsPreconnected s :=
  convex_iff_ordConnected.trans isPreconnected_iff_ordConnected.symm


alias ⟨_, IsPreconnected.convex⟩ := Real.convex_iff_isPreconnected


/-- Every vector in `stdSimplex 𝕜 ι` has `max`-norm at most `1`. -/
theorem stdSimplex_subset_closedBall : stdSimplex ℝ ι ⊆ Metric.closedBall 0 1 := fun f hf ↦ by
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    f : ι → Real
    hf : Membership.mem (stdSimplex Real ι) f
    ⊢ Membership.mem (Metric.closedBall 0 1) f
  -/
  rw [Metric.mem_closedBall, dist_pi_le_iff zero_le_one]
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    f : ι → Real
    hf : Membership.mem (stdSimplex Real ι) f
    ⊢ ∀ (b : ι), LE.le (Dist.dist (f b) (0 b)) 1
  -/
  intro x
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    f : ι → Real
    hf : Membership.mem (stdSimplex Real ι) f
    x : ι
    ⊢ LE.le (Dist.dist (f x) (0 x)) 1
  -/
  rw [Pi.zero_apply, Real.dist_0_eq_abs, abs_of_nonneg <| hf.1 x]
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    f : ι → Real
    hf : Membership.mem (stdSimplex Real ι) f
    x : ι
    ⊢ LE.le (f x) 1
  -/
  exact (mem_Icc_of_mem_stdSimplex hf x).2
  /-
    🎉 no goals
  -/


/-- `stdSimplex ℝ ι` is bounded. -/
theorem bounded_stdSimplex : IsBounded (stdSimplex ℝ ι) :=
  (Metric.isBounded_iff_subset_closedBall 0).2 ⟨1, stdSimplex_subset_closedBall⟩


/-- `stdSimplex ℝ ι` is closed. -/
theorem isClosed_stdSimplex : IsClosed (stdSimplex ℝ ι) :=
  (stdSimplex_eq_inter ℝ ι).symm ▸
    IsClosed.inter (isClosed_iInter fun i => isClosed_le continuous_const (continuous_apply i))
      (isClosed_eq (continuous_finset_sum _ fun x _ => continuous_apply x) continuous_const)


/-- `stdSimplex ℝ ι` is compact. -/
theorem isCompact_stdSimplex : IsCompact (stdSimplex ℝ ι) :=
  Metric.isCompact_iff_isClosed_bounded.2 ⟨isClosed_stdSimplex ι, bounded_stdSimplex ι⟩


instance stdSimplex.instCompactSpace_coe : CompactSpace ↥(stdSimplex ℝ ι) :=
  isCompact_iff_compactSpace.mp <| isCompact_stdSimplex _


/-- The standard one-dimensional simplex in `ℝ² = Fin 2 → ℝ`
is homeomorphic to the unit interval. -/
@[simps! (config := .asFn)]
def stdSimplexHomeomorphUnitInterval : stdSimplex ℝ (Fin 2) ≃ₜ unitInterval where
  toEquiv := stdSimplexEquivIcc ℝ
  continuous_toFun := .subtype_mk ((continuous_apply 0).comp continuous_subtype_val) _
  continuous_invFun := by
    /-
      ι : Type u_1
      𝕜 : Type u_2
      E : Type u_3
      inst✝ : Fintype ι
      ⊢ Continuous (stdSimplexEquivIcc Real).invFun
    -/
    apply Continuous.subtype_mk
    exact (continuous_pi <| Fin.forall_fin_two.2
      ⟨continuous_subtype_val, continuous_const.sub continuous_subtype_val⟩)


theorem segment_subset_closure_openSegment : [x -[𝕜] y] ⊆ closure (openSegment 𝕜 x y) := by
  /-
    𝕜 : Type u_2
    E : Type u_3
    inst✝⁸ : LinearOrderedRing 𝕜
    inst✝⁷ : DenselyOrdered 𝕜
    inst✝⁶ : TopologicalSpace 𝕜
    inst✝⁵ : OrderTopology 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : TopologicalSpace E
    inst✝² : ContinuousAdd E
    inst✝¹ : Module 𝕜 E
    inst✝ : ContinuousSMul 𝕜 E
    x y : E
    ⊢ HasSubset.Subset (segment 𝕜 x y) (closure (openSegment 𝕜 x y))
  -/
  rw [segment_eq_image, openSegment_eq_image, ← closure_Ioo (zero_ne_one' 𝕜)]
  /-
    𝕜 : Type u_2
    E : Type u_3
    inst✝⁸ : LinearOrderedRing 𝕜
    inst✝⁷ : DenselyOrdered 𝕜
    inst✝⁶ : TopologicalSpace 𝕜
    inst✝⁵ : OrderTopology 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : TopologicalSpace E
    inst✝² : ContinuousAdd E
    inst✝¹ : Module 𝕜 E
    inst✝ : ContinuousSMul 𝕜 E
    x y : E
    ⊢ HasSubset.Subset (Set.image (fun θ => HAdd.hAdd (HSMul.hSMul (HSub.hSub 1 θ) …
  -/
  exact image_closure_subset_closure_image (by fun_prop)
  /-
    🎉 no goals
  -/


@[simp]
theorem closure_openSegment (x y : E) : closure (openSegment 𝕜 x y) = [x -[𝕜] y] := by
  /-
    𝕜 : Type u_2
    E : Type u_3
    inst✝¹¹ : LinearOrderedRing 𝕜
    inst✝¹⁰ : DenselyOrdered 𝕜
    inst✝⁹ : PseudoMetricSpace 𝕜
    inst✝⁸ : OrderTopology 𝕜
    inst✝⁷ : ProperSpace 𝕜
    inst✝⁶ : CompactIccSpace 𝕜
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : T2Space E
    inst✝² : ContinuousAdd E
    inst✝¹ : Module 𝕜 E
    inst✝ : ContinuousSMul 𝕜 E
    x y : E
    ⊢ Eq (closure (openSegment 𝕜 x y)) (segment 𝕜 x y)
  -/
  rw [segment_eq_image, openSegment_eq_image, ← closure_Ioo (zero_ne_one' 𝕜)]
  exact (image_closure_of_isCompact (isBounded_Ioo _ _).isCompact_closure <|
    Continuous.continuousOn <| by fun_prop).symm


/-- If `s` is a convex set, then `a • interior s + b • closure s ⊆ interior s` for all `0 < a`,
`0 ≤ b`, `a + b = 1`. See also `Convex.combo_interior_self_subset_interior` for a weaker version. -/
theorem Convex.combo_interior_closure_subset_interior {s : Set E} (hs : Convex 𝕜 s) {a b : 𝕜}
    (ha : 0 < a) (hb : 0 ≤ b) (hab : a + b = 1) : a • interior s + b • closure s ⊆ interior s :=
  interior_smul₀ ha.ne' s ▸
    calc
      interior (a • s) + b • closure s ⊆ interior (a • s) + closure (b • s) :=
        add_subset_add Subset.rfl (smul_closure_subset b s)
                                         /-
                                           𝕜 : Type u_2
                                           E : Type u_3
                                           inst✝⁵ : LinearOrderedField 𝕜
                                           inst✝⁴ : AddCommGroup E
                                           inst✝³ : Module 𝕜 E
                                           inst✝² : TopologicalSpace E
                                           inst✝¹ : TopologicalAddGroup E
                                           inst✝ : ContinuousConstSMul 𝕜 E
                                           s : Set E
                                           hs : Convex 𝕜 s
                                           a b : 𝕜
                                           ha : LT.lt 0 a
                                           hb : LE.le 0 b
                                           hab : Eq (HAdd.hAdd a b) 1
                                           ⊢ Eq (HAdd.hAdd (interior (HSMul.hSMul a s)) (closure (HSMul.hSMul b s))) (HAd …
                                         -/
      _ = interior (a • s) + b • s := by rw [isOpen_interior.add_closure (b • s)]
                                         /-
                                           🎉 no goals
                                         -/
      _ ⊆ interior (a • s + b • s) := subset_interior_add_left
      _ ⊆ interior s := interior_mono <| hs.set_combo_subset ha.le hb hab


/-- If `s` is a convex set, then `a • interior s + b • s ⊆ interior s` for all `0 < a`, `0 ≤ b`,
`a + b = 1`. See also `Convex.combo_interior_closure_subset_interior` for a stronger version. -/
theorem Convex.combo_interior_self_subset_interior {s : Set E} (hs : Convex 𝕜 s) {a b : 𝕜}
    (ha : 0 < a) (hb : 0 ≤ b) (hab : a + b = 1) : a • interior s + b • s ⊆ interior s :=
  calc
    a • interior s + b • s ⊆ a • interior s + b • closure s :=
      add_subset_add Subset.rfl <| image_subset _ subset_closure
    _ ⊆ interior s := hs.combo_interior_closure_subset_interior ha hb hab


/-- If `s` is a convex set, then `a • closure s + b • interior s ⊆ interior s` for all `0 ≤ a`,
`0 < b`, `a + b = 1`. See also `Convex.combo_self_interior_subset_interior` for a weaker version. -/
theorem Convex.combo_closure_interior_subset_interior {s : Set E} (hs : Convex 𝕜 s) {a b : 𝕜}
    (ha : 0 ≤ a) (hb : 0 < b) (hab : a + b = 1) : a • closure s + b • interior s ⊆ interior s := by
  /-
    𝕜 : Type u_2
    E : Type u_3
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousConstSMul 𝕜 E
    s : Set E
    hs : Convex 𝕜 s
    a b : 𝕜
    ha : LE.le 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ HasSubset.Subset (HAdd.hAdd (HSMul.hSMul a (closure s)) (HSMul.hSMul b (inte …
  -/
  rw [add_comm]
  /-
    𝕜 : Type u_2
    E : Type u_3
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousConstSMul 𝕜 E
    s : Set E
    hs : Convex 𝕜 s
    a b : 𝕜
    ha : LE.le 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ HasSubset.Subset (HAdd.hAdd (HSMul.hSMul b (interior s)) (HSMul.hSMul a (clo …
  -/
  exact hs.combo_interior_closure_subset_interior hb ha (add_comm a b ▸ hab)
  /-
    🎉 no goals
  -/


/-- If `s` is a convex set, then `a • s + b • interior s ⊆ interior s` for all `0 ≤ a`, `0 < b`,
`a + b = 1`. See also `Convex.combo_closure_interior_subset_interior` for a stronger version. -/
theorem Convex.combo_self_interior_subset_interior {s : Set E} (hs : Convex 𝕜 s) {a b : 𝕜}
    (ha : 0 ≤ a) (hb : 0 < b) (hab : a + b = 1) : a • s + b • interior s ⊆ interior s := by
  /-
    𝕜 : Type u_2
    E : Type u_3
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousConstSMul 𝕜 E
    s : Set E
    hs : Convex 𝕜 s
    a b : 𝕜
    ha : LE.le 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ HasSubset.Subset (HAdd.hAdd (HSMul.hSMul a s) (HSMul.hSMul b (interior s)))  …
  -/
  rw [add_comm]
  /-
    𝕜 : Type u_2
    E : Type u_3
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousConstSMul 𝕜 E
    s : Set E
    hs : Convex 𝕜 s
    a b : 𝕜
    ha : LE.le 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ HasSubset.Subset (HAdd.hAdd (HSMul.hSMul b (interior s)) (HSMul.hSMul a s))  …
  -/
  exact hs.combo_interior_self_subset_interior hb ha (add_comm a b ▸ hab)
  /-
    🎉 no goals
  -/


theorem Convex.combo_interior_closure_mem_interior {s : Set E} (hs : Convex 𝕜 s) {x y : E}
    (hx : x ∈ interior s) (hy : y ∈ closure s) {a b : 𝕜} (ha : 0 < a) (hb : 0 ≤ b)
    (hab : a + b = 1) : a • x + b • y ∈ interior s :=
  hs.combo_interior_closure_subset_interior ha hb hab <|
    add_mem_add (smul_mem_smul_set hx) (smul_mem_smul_set hy)


theorem Convex.combo_interior_self_mem_interior {s : Set E} (hs : Convex 𝕜 s) {x y : E}
    (hx : x ∈ interior s) (hy : y ∈ s) {a b : 𝕜} (ha : 0 < a) (hb : 0 ≤ b) (hab : a + b = 1) :
    a • x + b • y ∈ interior s :=
  hs.combo_interior_closure_mem_interior hx (subset_closure hy) ha hb hab


theorem Convex.combo_closure_interior_mem_interior {s : Set E} (hs : Convex 𝕜 s) {x y : E}
    (hx : x ∈ closure s) (hy : y ∈ interior s) {a b : 𝕜} (ha : 0 ≤ a) (hb : 0 < b)
    (hab : a + b = 1) : a • x + b • y ∈ interior s :=
  hs.combo_closure_interior_subset_interior ha hb hab <|
    add_mem_add (smul_mem_smul_set hx) (smul_mem_smul_set hy)


theorem Convex.combo_self_interior_mem_interior {s : Set E} (hs : Convex 𝕜 s) {x y : E} (hx : x ∈ s)
    (hy : y ∈ interior s) {a b : 𝕜} (ha : 0 ≤ a) (hb : 0 < b) (hab : a + b = 1) :
    a • x + b • y ∈ interior s :=
  hs.combo_closure_interior_mem_interior (subset_closure hx) hy ha hb hab


theorem Convex.openSegment_interior_closure_subset_interior {s : Set E} (hs : Convex 𝕜 s) {x y : E}
    (hx : x ∈ interior s) (hy : y ∈ closure s) : openSegment 𝕜 x y ⊆ interior s := by
  /-
    𝕜 : Type u_2
    E : Type u_3
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousConstSMul 𝕜 E
    s : Set E
    hs : Convex 𝕜 s
    x y : E
    hx : Membership.mem (interior s) x
    hy : Membership.mem (closure s) y
    ⊢ HasSubset.Subset (openSegment 𝕜 x y) (interior s)
  -/
  rintro _ ⟨a, b, ha, hb, hab, rfl⟩
  /-
    case intro.intro.intro.intro.intro
    𝕜 : Type u_2
    E : Type u_3
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousConstSMul 𝕜 E
    s : Set E
    hs : Convex 𝕜 s
    x y : E
    hx : Membership.mem (interior s) x
    hy : Membership.mem (closure s) y
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem (interior s) (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))
  -/
  exact hs.combo_interior_closure_mem_interior hx hy ha hb.le hab
  /-
    🎉 no goals
  -/


theorem Convex.openSegment_interior_self_subset_interior {s : Set E} (hs : Convex 𝕜 s) {x y : E}
    (hx : x ∈ interior s) (hy : y ∈ s) : openSegment 𝕜 x y ⊆ interior s :=
  hs.openSegment_interior_closure_subset_interior hx (subset_closure hy)


theorem Convex.openSegment_closure_interior_subset_interior {s : Set E} (hs : Convex 𝕜 s) {x y : E}
    (hx : x ∈ closure s) (hy : y ∈ interior s) : openSegment 𝕜 x y ⊆ interior s := by
  /-
    𝕜 : Type u_2
    E : Type u_3
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousConstSMul 𝕜 E
    s : Set E
    hs : Convex 𝕜 s
    x y : E
    hx : Membership.mem (closure s) x
    hy : Membership.mem (interior s) y
    ⊢ HasSubset.Subset (openSegment 𝕜 x y) (interior s)
  -/
  rintro _ ⟨a, b, ha, hb, hab, rfl⟩
  /-
    case intro.intro.intro.intro.intro
    𝕜 : Type u_2
    E : Type u_3
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousConstSMul 𝕜 E
    s : Set E
    hs : Convex 𝕜 s
    x y : E
    hx : Membership.mem (closure s) x
    hy : Membership.mem (interior s) y
    a b : 𝕜
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem (interior s) (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))
  -/
  exact hs.combo_closure_interior_mem_interior hx hy ha.le hb hab
  /-
    🎉 no goals
  -/


theorem Convex.openSegment_self_interior_subset_interior {s : Set E} (hs : Convex 𝕜 s) {x y : E}
    (hx : x ∈ s) (hy : y ∈ interior s) : openSegment 𝕜 x y ⊆ interior s :=
  hs.openSegment_closure_interior_subset_interior (subset_closure hx) hy


/-- If `x ∈ closure s` and `y ∈ interior s`, then the segment `(x, y]` is included in `interior s`.
-/
theorem Convex.add_smul_sub_mem_interior' {s : Set E} (hs : Convex 𝕜 s) {x y : E}
    (hx : x ∈ closure s) (hy : y ∈ interior s) {t : 𝕜} (ht : t ∈ Ioc (0 : 𝕜) 1) :
    x + t • (y - x) ∈ interior s := by
  simpa only [sub_smul, smul_sub, one_smul, add_sub, add_comm] using
    hs.combo_interior_closure_mem_interior hy hx ht.1 (sub_nonneg.mpr ht.2)
      (add_sub_cancel _ _)


/-- If `x ∈ s` and `y ∈ interior s`, then the segment `(x, y]` is included in `interior s`. -/
theorem Convex.add_smul_sub_mem_interior {s : Set E} (hs : Convex 𝕜 s) {x y : E} (hx : x ∈ s)
    (hy : y ∈ interior s) {t : 𝕜} (ht : t ∈ Ioc (0 : 𝕜) 1) : x + t • (y - x) ∈ interior s :=
  hs.add_smul_sub_mem_interior' (subset_closure hx) hy ht


/-- If `x ∈ closure s` and `x + y ∈ interior s`, then `x + t y ∈ interior s` for `t ∈ (0, 1]`. -/
theorem Convex.add_smul_mem_interior' {s : Set E} (hs : Convex 𝕜 s) {x y : E} (hx : x ∈ closure s)
    (hy : x + y ∈ interior s) {t : 𝕜} (ht : t ∈ Ioc (0 : 𝕜) 1) : x + t • y ∈ interior s := by
  /-
    𝕜 : Type u_2
    E : Type u_3
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousConstSMul 𝕜 E
    s : Set E
    hs : Convex 𝕜 s
    x y : E
    hx : Membership.mem (closure s) x
    hy : Membership.mem (interior s) (HAdd.hAdd x y)
    t : 𝕜
    ht : Membership.mem (Set.Ioc 0 1) t
    ⊢ Membership.mem (interior s) (HAdd.hAdd x (HSMul.hSMul t y))
  -/
  simpa only [add_sub_cancel_left] using hs.add_smul_sub_mem_interior' hx hy ht
  /-
    🎉 no goals
  -/


/-- If `x ∈ s` and `x + y ∈ interior s`, then `x + t y ∈ interior s` for `t ∈ (0, 1]`. -/
theorem Convex.add_smul_mem_interior {s : Set E} (hs : Convex 𝕜 s) {x y : E} (hx : x ∈ s)
    (hy : x + y ∈ interior s) {t : 𝕜} (ht : t ∈ Ioc (0 : 𝕜) 1) : x + t • y ∈ interior s :=
  hs.add_smul_mem_interior' (subset_closure hx) hy ht


/-- In a topological vector space, the interior of a convex set is convex. -/
protected theorem Convex.interior {s : Set E} (hs : Convex 𝕜 s) : Convex 𝕜 (interior s) :=
  convex_iff_openSegment_subset.mpr fun _ hx _ hy =>
    hs.openSegment_closure_interior_subset_interior (interior_subset_closure hx) hy


/-- In a topological vector space, the closure of a convex set is convex. -/
protected theorem Convex.closure {s : Set E} (hs : Convex 𝕜 s) : Convex 𝕜 (closure s) :=
  fun x hx y hy a b ha hb hab =>
  let f : E → E → E := fun x' y' => a • x' + b • y'
  have hf : Continuous (Function.uncurry f) :=
    (continuous_fst.const_smul _).add (continuous_snd.const_smul _)
  show f x y ∈ closure s from map_mem_closure₂ hf hx hy fun _ hx' _ hy' => hs hx' hy' ha hb hab


/-- A convex set `s` is strictly convex provided that for any two distinct points of
`s \ interior s`, the line passing through these points has nonempty intersection with
`interior s`. -/
protected theorem Convex.strictConvex' {s : Set E} (hs : Convex 𝕜 s)
    (h : (s \ interior s).Pairwise fun x y => ∃ c : 𝕜, lineMap x y c ∈ interior s) :
    StrictConvex 𝕜 s := by
  /-
    𝕜 : Type u_2
    E : Type u_3
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousConstSMul 𝕜 E
    s : Set E
    hs : Convex 𝕜 s
    h : (SDiff.sdiff s (interior s)).Pairwise fun x y => Exists fun c => Membershi …
    ⊢ StrictConvex 𝕜 s
  -/
  refine strictConvex_iff_openSegment_subset.2 ?_
  /-
    𝕜 : Type u_2
    E : Type u_3
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousConstSMul 𝕜 E
    s : Set E
    hs : Convex 𝕜 s
    h : (SDiff.sdiff s (interior s)).Pairwise fun x y => Exists fun c => Membershi …
    ⊢ s.Pairwise fun x y => HasSubset.Subset (openSegment 𝕜 x y) (interior s)
  -/
  intro x hx y hy hne
  /-
    𝕜 : Type u_2
    E : Type u_3
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousConstSMul 𝕜 E
    s : Set E
    hs : Convex 𝕜 s
    h : (SDiff.sdiff s (interior s)).Pairwise fun x y => Exists fun c => Membershi …
    x : E
    hx : Membership.mem s x
    y : E
    hy : Membership.mem s y
    hne : Ne x y
    ⊢ HasSubset.Subset (openSegment 𝕜 x y) (interior s)
  -/
  by_cases hx' : x ∈ interior s
    /-
      case pos
      𝕜 : Type u_2
      E : Type u_3
      inst✝⁵ : LinearOrderedField 𝕜
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module 𝕜 E
      inst✝² : TopologicalSpace E
      inst✝¹ : TopologicalAddGroup E
      inst✝ : ContinuousConstSMul 𝕜 E
      s : Set E
      hs : Convex 𝕜 s
      h : (SDiff.sdiff s (interior s)).Pairwise fun x y => Exists fun c => Membershi …
      x : E
      hx : Membership.mem s x
      y : E
      hy : Membership.mem s y
      hne : Ne x y
      hx' : Membership.mem (interior s) x
      ⊢ HasSubset.Subset (openSegment 𝕜 x y) (interior s)
    -/
  · exact hs.openSegment_interior_self_subset_interior hx' hy
    /-
      🎉 no goals
    -/
  /-
    case neg
    𝕜 : Type u_2
    E : Type u_3
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousConstSMul 𝕜 E
    s : Set E
    hs : Convex 𝕜 s
    h : (SDiff.sdiff s (interior s)).Pairwise fun x y => Exists fun c => Membershi …
    x : E
    hx : Membership.mem s x
    y : E
    hy : Membership.mem s y
    hne : Ne x y
    hx' : Not (Membership.mem (interior s) x)
    ⊢ HasSubset.Subset (openSegment 𝕜 x y) (interior s)
  -/
  by_cases hy' : y ∈ interior s
    /-
      case pos
      𝕜 : Type u_2
      E : Type u_3
      inst✝⁵ : LinearOrderedField 𝕜
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module 𝕜 E
      inst✝² : TopologicalSpace E
      inst✝¹ : TopologicalAddGroup E
      inst✝ : ContinuousConstSMul 𝕜 E
      s : Set E
      hs : Convex 𝕜 s
      h : (SDiff.sdiff s (interior s)).Pairwise fun x y => Exists fun c => Membershi …
      x : E
      hx : Membership.mem s x
      y : E
      hy : Membership.mem s y
      hne : Ne x y
      hx' : Not (Membership.mem (interior s) x)
      hy' : Membership.mem (interior s) y
      ⊢ HasSubset.Subset (openSegment 𝕜 x y) (interior s)
    -/
  · exact hs.openSegment_self_interior_subset_interior hx hy'
    /-
      🎉 no goals
    -/
  /-
    case neg
    𝕜 : Type u_2
    E : Type u_3
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousConstSMul 𝕜 E
    s : Set E
    hs : Convex 𝕜 s
    h : (SDiff.sdiff s (interior s)).Pairwise fun x y => Exists fun c => Membershi …
    x : E
    hx : Membership.mem s x
    y : E
    hy : Membership.mem s y
    hne : Ne x y
    hx' : Not (Membership.mem (interior s) x)
    hy' : Not (Membership.mem (interior s) y)
    ⊢ HasSubset.Subset (openSegment 𝕜 x y) (interior s)
  -/
  rcases h ⟨hx, hx'⟩ ⟨hy, hy'⟩ hne with ⟨c, hc⟩
  refine (openSegment_subset_union x y ⟨c, rfl⟩).trans
    (insert_subset_iff.2 ⟨hc, union_subset ?_ ?_⟩)
  exacts [hs.openSegment_self_interior_subset_interior hx hc,
    hs.openSegment_interior_self_subset_interior hc hy]


/-- A convex set `s` is strictly convex provided that for any two distinct points `x`, `y` of
`s \ interior s`, the segment with endpoints `x`, `y` has nonempty intersection with
`interior s`. -/
protected theorem Convex.strictConvex {s : Set E} (hs : Convex 𝕜 s)
    (h : (s \ interior s).Pairwise fun x y => ([x -[𝕜] y] \ frontier s).Nonempty) :
    StrictConvex 𝕜 s := by
  /-
    𝕜 : Type u_2
    E : Type u_3
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousConstSMul 𝕜 E
    s : Set E
    hs : Convex 𝕜 s
    h : (SDiff.sdiff s (interior s)).Pairwise fun x y => (SDiff.sdiff (segment 𝕜 x …
    ⊢ StrictConvex 𝕜 s
  -/
  refine hs.strictConvex' <| h.imp_on fun x hx y hy _ => ?_
  /-
    𝕜 : Type u_2
    E : Type u_3
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousConstSMul 𝕜 E
    s : Set E
    hs : Convex 𝕜 s
    h : (SDiff.sdiff s (interior s)).Pairwise fun x y => (SDiff.sdiff (segment 𝕜 x …
    x : E
    hx : Membership.mem (SDiff.sdiff s (interior s)) x
    y : E
    hy : Membership.mem (SDiff.sdiff s (interior s)) y
    x✝ : Ne x y
    ⊢ fun ⦃a b⦄ => (SDiff.sdiff (segment 𝕜 a b) (frontier s)).Nonempty → Exists fu …
  -/
  simp only [segment_eq_image_lineMap, ← self_diff_frontier]
  /-
    𝕜 : Type u_2
    E : Type u_3
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousConstSMul 𝕜 E
    s : Set E
    hs : Convex 𝕜 s
    h : (SDiff.sdiff s (interior s)).Pairwise fun x y => (SDiff.sdiff (segment 𝕜 x …
    x : E
    hx : Membership.mem (SDiff.sdiff s (interior s)) x
    y : E
    hy : Membership.mem (SDiff.sdiff s (interior s)) y
    x✝ : Ne x y
    ⊢ (SDiff.sdiff (Set.image (⇑(AffineMap.lineMap x y)) (Set.Icc 0 1)) (frontier  …
  -/
  rintro ⟨_, ⟨⟨c, hc, rfl⟩, hcs⟩⟩
  /-
    case intro.intro.intro.intro
    𝕜 : Type u_2
    E : Type u_3
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousConstSMul 𝕜 E
    s : Set E
    hs : Convex 𝕜 s
    h : (SDiff.sdiff s (interior s)).Pairwise fun x y => (SDiff.sdiff (segment 𝕜 x …
    x : E
    hx : Membership.mem (SDiff.sdiff s (interior s)) x
    y : E
    hy : Membership.mem (SDiff.sdiff s (interior s)) y
    x✝ : Ne x y
    c : 𝕜
    hc : Membership.mem (Set.Icc 0 1) c
    hcs : Not (Membership.mem (frontier s) ((AffineMap.lineMap x y) c))
    ⊢ Exists fun c => Membership.mem (SDiff.sdiff s (frontier s)) ((AffineMap.line …
  -/
  refine ⟨c, hs.segment_subset hx.1 hy.1 ?_, hcs⟩
  /-
    case intro.intro.intro.intro
    𝕜 : Type u_2
    E : Type u_3
    inst✝⁵ : LinearOrderedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousConstSMul 𝕜 E
    s : Set E
    hs : Convex 𝕜 s
    h : (SDiff.sdiff s (interior s)).Pairwise fun x y => (SDiff.sdiff (segment 𝕜 x …
    x : E
    hx : Membership.mem (SDiff.sdiff s (interior s)) x
    y : E
    hy : Membership.mem (SDiff.sdiff s (interior s)) y
    x✝ : Ne x y
    c : 𝕜
    hc : Membership.mem (Set.Icc 0 1) c
    hcs : Not (Membership.mem (frontier s) ((AffineMap.lineMap x y) c))
    ⊢ Membership.mem (segment 𝕜 x y) ((AffineMap.lineMap x y) c)
  -/
  exact (segment_eq_image_lineMap 𝕜 x y).symm ▸ mem_image_of_mem _ hc
  /-
    🎉 no goals
  -/


theorem Convex.closure_interior_eq_closure_of_nonempty_interior {s : Set E} (hs : Convex 𝕜 s)
    (hs' : (interior s).Nonempty) : closure (interior s) = closure s :=
  subset_antisymm (closure_mono interior_subset)
    fun _ h ↦ closure_mono (hs.openSegment_interior_closure_subset_interior hs'.choose_spec h)
      (segment_subset_closure_openSegment (right_mem_segment ..))


theorem Convex.interior_closure_eq_interior_of_nonempty_interior {s : Set E} (hs : Convex 𝕜 s)
    (hs' : (interior s).Nonempty) : interior (closure s) = interior s := by
  /-
    𝕜 : Type u_2
    E : Type u_3
    inst✝⁷ : LinearOrderedField 𝕜
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜 E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : TopologicalAddGroup E
    inst✝² : TopologicalSpace 𝕜
    inst✝¹ : OrderTopology 𝕜
    inst✝ : ContinuousSMul 𝕜 E
    s : Set E
    hs : Convex 𝕜 s
    hs' : (interior s).Nonempty
    ⊢ Eq (interior (closure s)) (interior s)
  -/
  refine subset_antisymm ?_ (interior_mono subset_closure)
  /-
    𝕜 : Type u_2
    E : Type u_3
    inst✝⁷ : LinearOrderedField 𝕜
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜 E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : TopologicalAddGroup E
    inst✝² : TopologicalSpace 𝕜
    inst✝¹ : OrderTopology 𝕜
    inst✝ : ContinuousSMul 𝕜 E
    s : Set E
    hs : Convex 𝕜 s
    hs' : (interior s).Nonempty
    ⊢ HasSubset.Subset (interior (closure s)) (interior s)
  -/
  intro y hy
  /-
    𝕜 : Type u_2
    E : Type u_3
    inst✝⁷ : LinearOrderedField 𝕜
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜 E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : TopologicalAddGroup E
    inst✝² : TopologicalSpace 𝕜
    inst✝¹ : OrderTopology 𝕜
    inst✝ : ContinuousSMul 𝕜 E
    s : Set E
    hs : Convex 𝕜 s
    hs' : (interior s).Nonempty
    y : E
    hy : Membership.mem (interior (closure s)) y
    ⊢ Membership.mem (interior s) y
  -/
  rcases hs' with ⟨x, hx⟩
  /-
    case intro
    𝕜 : Type u_2
    E : Type u_3
    inst✝⁷ : LinearOrderedField 𝕜
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜 E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : TopologicalAddGroup E
    inst✝² : TopologicalSpace 𝕜
    inst✝¹ : OrderTopology 𝕜
    inst✝ : ContinuousSMul 𝕜 E
    s : Set E
    hs : Convex 𝕜 s
    y : E
    hy : Membership.mem (interior (closure s)) y
    x : E
    hx : Membership.mem (interior s) x
    ⊢ Membership.mem (interior s) y
  -/
  have h := AffineMap.lineMap_apply_one (k := 𝕜) x y
  obtain ⟨t, ht1, ht⟩ := AffineMap.lineMap_continuous.tendsto' _ _ h |>.eventually_mem
    (mem_interior_iff_mem_nhds.1 hy) |>.exists_gt
  /-
    case intro.intro.intro
    𝕜 : Type u_2
    E : Type u_3
    inst✝⁷ : LinearOrderedField 𝕜
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜 E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : TopologicalAddGroup E
    inst✝² : TopologicalSpace 𝕜
    inst✝¹ : OrderTopology 𝕜
    inst✝ : ContinuousSMul 𝕜 E
    s : Set E
    hs : Convex 𝕜 s
    y : E
    hy : Membership.mem (interior (closure s)) y
    x : E
    hx : Membership.mem (interior s) x
    h : Eq ((AffineMap.lineMap x y) 1) y
    t : 𝕜
    ht1 : GT.gt t 1
    ht : Membership.mem (closure s) ((AffineMap.lineMap x y) t)
    ⊢ Membership.mem (interior s) y
  -/
  apply hs.openSegment_interior_closure_subset_interior hx ht
  /-
    case intro.intro.intro.a
    𝕜 : Type u_2
    E : Type u_3
    inst✝⁷ : LinearOrderedField 𝕜
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜 E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : TopologicalAddGroup E
    inst✝² : TopologicalSpace 𝕜
    inst✝¹ : OrderTopology 𝕜
    inst✝ : ContinuousSMul 𝕜 E
    s : Set E
    hs : Convex 𝕜 s
    y : E
    hy : Membership.mem (interior (closure s)) y
    x : E
    hx : Membership.mem (interior s) x
    h : Eq ((AffineMap.lineMap x y) 1) y
    t : 𝕜
    ht1 : GT.gt t 1
    ht : Membership.mem (closure s) ((AffineMap.lineMap x y) t)
    ⊢ Membership.mem (openSegment 𝕜 x ((AffineMap.lineMap x y) t)) y
  -/
  nth_rw 1 [← AffineMap.lineMap_apply_zero (k := 𝕜) x y, ← image_openSegment]
  /-
    case intro.intro.intro.a
    𝕜 : Type u_2
    E : Type u_3
    inst✝⁷ : LinearOrderedField 𝕜
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜 E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : TopologicalAddGroup E
    inst✝² : TopologicalSpace 𝕜
    inst✝¹ : OrderTopology 𝕜
    inst✝ : ContinuousSMul 𝕜 E
    s : Set E
    hs : Convex 𝕜 s
    y : E
    hy : Membership.mem (interior (closure s)) y
    x : E
    hx : Membership.mem (interior s) x
    h : Eq ((AffineMap.lineMap x y) 1) y
    t : 𝕜
    ht1 : GT.gt t 1
    ht : Membership.mem (closure s) ((AffineMap.lineMap x y) t)
    ⊢ Membership.mem (Set.image (⇑(AffineMap.lineMap x y)) (openSegment 𝕜 0 t)) y
  -/
  exact ⟨1, Ioo_subset_openSegment ⟨zero_lt_one, ht1⟩, h⟩
  /-
    🎉 no goals
  -/


theorem convex_closed_sInter {S : Set (Set E)} (h : ∀ s ∈ S, Convex 𝕜 s ∧ IsClosed s) :
    Convex 𝕜 (⋂₀ S) ∧ IsClosed (⋂₀ S) :=
  ⟨fun _ hx => starConvex_sInter fun _ hs => (h _ hs).1 <| hx _ hs,
    isClosed_sInter fun _ hs => (h _ hs).2⟩


/-- The convex closed hull of a set `s` is the minimal convex closed set that includes `s`. -/
@[simps! isClosed]
def closedConvexHull : ClosureOperator (Set E) := .ofCompletePred (fun s => Convex 𝕜 s ∧ IsClosed s)
  fun _ ↦ convex_closed_sInter


theorem convex_closedConvexHull {s : Set E} :
    Convex 𝕜 (closedConvexHull 𝕜 s) := ((closedConvexHull 𝕜).isClosed_closure s).1


theorem isClosed_closedConvexHull {s : Set E} :
    IsClosed (closedConvexHull 𝕜 s) := ((closedConvexHull 𝕜).isClosed_closure s).2


theorem subset_closedConvexHull {s : Set E} : s ⊆ closedConvexHull 𝕜 s :=
  (closedConvexHull 𝕜).le_closure s


theorem closure_subset_closedConvexHull {s : Set E} : closure s ⊆ closedConvexHull 𝕜 s :=
  closure_minimal subset_closedConvexHull isClosed_closedConvexHull


theorem closedConvexHull_min {s t : Set E} (hst : s ⊆ t) (h_conv : Convex 𝕜 t)
    (h_closed : IsClosed t) : closedConvexHull 𝕜 s ⊆ t :=
  (closedConvexHull 𝕜).closure_min hst ⟨h_conv, h_closed⟩


theorem convexHull_subset_closedConvexHull {s : Set E} :
    (convexHull 𝕜) s ⊆ (closedConvexHull 𝕜) s :=
  convexHull_min subset_closedConvexHull convex_closedConvexHull


@[simp]
theorem closedConvexHull_closure_eq_closedConvexHull {s : Set E} :
    closedConvexHull 𝕜 (closure s) = closedConvexHull 𝕜 s :=
  subset_antisymm (by
    /-
      𝕜 : Type u_2
      E : Type u_3
      inst✝³ : OrderedSemiring 𝕜
      inst✝² : AddCommGroup E
      inst✝¹ : Module 𝕜 E
      inst✝ : TopologicalSpace E
      s : Set E
      ⊢ HasSubset.Subset ((closedConvexHull 𝕜) (closure s)) ((closedConvexHull 𝕜) s)
    -/
    simpa using ((closedConvexHull 𝕜).monotone (closure_subset_closedConvexHull (𝕜 := 𝕜) (E := E))))
    /-
      🎉 no goals
    -/
    ((closedConvexHull 𝕜).monotone subset_closure)


theorem closedConvexHull_eq_closure_convexHull {s : Set E} :
    closedConvexHull 𝕜 s = closure (convexHull 𝕜 s) := subset_antisymm
  (closedConvexHull_min (subset_trans (subset_convexHull 𝕜 s) subset_closure)
    (Convex.closure (convex_convexHull 𝕜 s)) isClosed_closure)
  (closure_minimal convexHull_subset_closedConvexHull isClosed_closedConvexHull)


/-- Convex hull of a finite set is compact. -/
theorem Set.Finite.isCompact_convexHull {s : Set E} (hs : s.Finite) :
    IsCompact (convexHull ℝ s) := by
  /-
    E : Type u_3
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul Real E
    s : Set E
    hs : s.Finite
    ⊢ IsCompact ((convexHull Real) s)
  -/
  rw [hs.convexHull_eq_image]
  /-
    E : Type u_3
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul Real E
    s : Set E
    hs : s.Finite
    ⊢ IsCompact (Set.image (⇑(Finset.univ.sum fun x => (LinearMap.proj x).smulRigh …
  -/
  apply (@isCompact_stdSimplex _ hs.fintype).image
  /-
    E : Type u_3
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul Real E
    s : Set E
    hs : s.Finite
    ⊢ Continuous ⇑(Finset.univ.sum fun x => (LinearMap.proj x).smulRight ↑x)
  -/
  haveI := hs.fintype
  /-
    E : Type u_3
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul Real E
    s : Set E
    hs : s.Finite
    this : Fintype ↑s
    ⊢ Continuous ⇑(Finset.univ.sum fun x => (LinearMap.proj x).smulRight ↑x)
  -/
  apply LinearMap.continuous_on_pi
  /-
    🎉 no goals
  -/


/-- Convex hull of a finite set is closed. -/
theorem Set.Finite.isClosed_convexHull [T2Space E] {s : Set E} (hs : s.Finite) :
    IsClosed (convexHull ℝ s) :=
  hs.isCompact_convexHull.isClosed


/-- If we dilate the interior of a convex set about a point in its interior by a scale `t > 1`,
the result includes the closure of the original set.

TODO Generalise this from convex sets to sets that are balanced / star-shaped about `x`. -/
theorem Convex.closure_subset_image_homothety_interior_of_one_lt {s : Set E} (hs : Convex ℝ s)
    {x : E} (hx : x ∈ interior s) (t : ℝ) (ht : 1 < t) :
    closure s ⊆ homothety x t '' interior s := by
  /-
    E : Type u_3
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul Real E
    s : Set E
    hs : Convex Real s
    x : E
    hx : Membership.mem (interior s) x
    t : Real
    ht : LT.lt 1 t
    ⊢ HasSubset.Subset (closure s) (Set.image (⇑(AffineMap.homothety x t)) (interi …
  -/
  intro y hy
  /-
    E : Type u_3
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul Real E
    s : Set E
    hs : Convex Real s
    x : E
    hx : Membership.mem (interior s) x
    t : Real
    ht : LT.lt 1 t
    y : E
    hy : Membership.mem (closure s) y
    ⊢ Membership.mem (Set.image (⇑(AffineMap.homothety x t)) (interior s)) y
  -/
  have hne : t ≠ 0 := (one_pos.trans ht).ne'
  refine
    ⟨homothety x t⁻¹ y, hs.openSegment_interior_closure_subset_interior hx hy ?_,
      (AffineEquiv.homothetyUnitsMulHom x (Units.mk0 t hne)).apply_symm_apply y⟩
  rw [openSegment_eq_image_lineMap, ← inv_one, ← inv_Ioi₀ (zero_lt_one' ℝ), ← image_inv_eq_inv,
    image_image, homothety_eq_lineMap]
  /-
    E : Type u_3
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul Real E
    s : Set E
    hs : Convex Real s
    x : E
    hx : Membership.mem (interior s) x
    t : Real
    ht : LT.lt 1 t
    y : E
    hy : Membership.mem (closure s) y
    hne : Ne t 0
    ⊢ Membership.mem (Set.image (fun x_1 => (AffineMap.lineMap x y) (Inv.inv x_1)) …
  -/
  exact mem_image_of_mem _ ht
  /-
    🎉 no goals
  -/


/-- If we dilate a convex set about a point in its interior by a scale `t > 1`, the interior of
the result includes the closure of the original set.

TODO Generalise this from convex sets to sets that are balanced / star-shaped about `x`. -/
theorem Convex.closure_subset_interior_image_homothety_of_one_lt {s : Set E} (hs : Convex ℝ s)
    {x : E} (hx : x ∈ interior s) (t : ℝ) (ht : 1 < t) :
    closure s ⊆ interior (homothety x t '' s) :=
  (hs.closure_subset_image_homothety_interior_of_one_lt hx t ht).trans <|
    (homothety_isOpenMap x t (one_pos.trans ht).ne').image_interior_subset _


/-- If we dilate a convex set about a point in its interior by a scale `t > 1`, the interior of
the result includes the closure of the original set.

TODO Generalise this from convex sets to sets that are balanced / star-shaped about `x`. -/
theorem Convex.subset_interior_image_homothety_of_one_lt {s : Set E} (hs : Convex ℝ s) {x : E}
    (hx : x ∈ interior s) (t : ℝ) (ht : 1 < t) : s ⊆ interior (homothety x t '' s) :=
  subset_closure.trans <| hs.closure_subset_interior_image_homothety_of_one_lt hx t ht


theorem JoinedIn.of_segment_subset {E : Type*} [AddCommGroup E] [Module ℝ E]
    [TopologicalSpace E] [ContinuousAdd E] [ContinuousSMul ℝ E]
    {x y : E} {s : Set E} (h : [x -[ℝ] y] ⊆ s) : JoinedIn s x y := by
  /-
    E : Type u_4
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousAdd E
    inst✝ : ContinuousSMul Real E
    x y : E
    s : Set E
    h : HasSubset.Subset (segment Real x y) s
    ⊢ JoinedIn s x y
  -/
  have A : Continuous (fun t ↦ (1 - t) • x + t • y : ℝ → E) := by fun_prop
  /-
    E : Type u_4
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousAdd E
    inst✝ : ContinuousSMul Real E
    x y : E
    s : Set E
    h : HasSubset.Subset (segment Real x y) s
    A : Continuous fun t => HAdd.hAdd (HSMul.hSMul (HSub.hSub 1 t) x) (HSMul.hSMul …
    ⊢ JoinedIn s x y
  -/
  apply JoinedIn.ofLine A.continuousOn (by simp) (by simp)
  /-
    E : Type u_4
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousAdd E
    inst✝ : ContinuousSMul Real E
    x y : E
    s : Set E
    h : HasSubset.Subset (segment Real x y) s
    A : Continuous fun t => HAdd.hAdd (HSMul.hSMul (HSub.hSub 1 t) x) (HSMul.hSMul …
    ⊢ HasSubset.Subset (Set.image (fun t => HAdd.hAdd (HSMul.hSMul (HSub.hSub 1 t) …
  -/
  convert h
  /-
    case h.e'_3
    E : Type u_4
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousAdd E
    inst✝ : ContinuousSMul Real E
    x y : E
    s : Set E
    h : HasSubset.Subset (segment Real x y) s
    A : Continuous fun t => HAdd.hAdd (HSMul.hSMul (HSub.hSub 1 t) x) (HSMul.hSMul …
    ⊢ Eq (Set.image (fun t => HAdd.hAdd (HSMul.hSMul (HSub.hSub 1 t) x) (HSMul.hSM …
  -/
  rw [segment_eq_image ℝ x y]
  /-
    🎉 no goals
  -/


/-- A nonempty convex set is path connected. -/
protected theorem Convex.isPathConnected {s : Set E} (hconv : Convex ℝ s) (hne : s.Nonempty) :
    IsPathConnected s := by
  /-
    E : Type u_3
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul Real E
    s : Set E
    hconv : Convex Real s
    hne : s.Nonempty
    ⊢ IsPathConnected s
  -/
  refine isPathConnected_iff.mpr ⟨hne, ?_⟩
  /-
    E : Type u_3
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul Real E
    s : Set E
    hconv : Convex Real s
    hne : s.Nonempty
    ⊢ ∀ (x : E), Membership.mem s x → ∀ (y : E), Membership.mem s y → JoinedIn s x y
  -/
  intro x x_in y y_in
  /-
    E : Type u_3
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul Real E
    s : Set E
    hconv : Convex Real s
    hne : s.Nonempty
    x : E
    x_in : Membership.mem s x
    y : E
    y_in : Membership.mem s y
    ⊢ JoinedIn s x y
  -/
  exact JoinedIn.of_segment_subset ((segment_subset_iff ℝ).2 (hconv x_in y_in))
  /-
    🎉 no goals
  -/


/-- A nonempty convex set is connected. -/
protected theorem Convex.isConnected {s : Set E} (h : Convex ℝ s) (hne : s.Nonempty) :
    IsConnected s :=
  (h.isPathConnected hne).isConnected


/-- A convex set is preconnected. -/
protected theorem Convex.isPreconnected {s : Set E} (h : Convex ℝ s) : IsPreconnected s :=
  s.eq_empty_or_nonempty.elim (fun h => h.symm ▸ isPreconnected_empty) fun hne =>
    (h.isConnected hne).isPreconnected


/-- Every topological vector space over ℝ is path connected.

Not an instance, because it creates enormous TC subproblems (turn on `pp.all`).
-/
protected theorem TopologicalAddGroup.pathConnectedSpace : PathConnectedSpace E :=
  pathConnectedSpace_iff_univ.mpr <| convex_univ.isPathConnected ⟨(0 : E), trivial⟩


local notation "π" => Submodule.linearProjOfIsCompl _ _


/-- Given two complementary subspaces `p` and `q` in `E`, if the complement of `{0}`
is path connected in `p` then the complement of `q` is path connected in `E`. -/
theorem isPathConnected_compl_of_isPathConnected_compl_zero [ContinuousSMul ℝ E]
    {p q : Submodule ℝ E} (hpq : IsCompl p q) (hpc : IsPathConnected ({0}ᶜ : Set p)) :
    IsPathConnected (qᶜ : Set E) := by
  /-
    E : Type u_4
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul Real E
    p q : Submodule Real E
    hpq : IsCompl p q
    hpc : IsPathConnected (HasCompl.compl (Singleton.singleton 0))
    ⊢ IsPathConnected (HasCompl.compl ↑q)
  -/
  rw [isPathConnected_iff] at hpc ⊢
  /-
    E : Type u_4
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul Real E
    p q : Submodule Real E
    hpq : IsCompl p q
    hpc : And (HasCompl.compl (Singleton.singleton 0)).Nonempty (∀ (x : Subtype fu …
    ⊢ And (HasCompl.compl ↑q).Nonempty (∀ (x : E), Membership.mem (HasCompl.compl  …
  -/
  constructor
    /-
      case left
      E : Type u_4
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module Real E
      inst✝² : TopologicalSpace E
      inst✝¹ : TopologicalAddGroup E
      inst✝ : ContinuousSMul Real E
      p q : Submodule Real E
      hpq : IsCompl p q
      hpc : And (HasCompl.compl (Singleton.singleton 0)).Nonempty (∀ (x : Subtype fu …
      ⊢ (HasCompl.compl ↑q).Nonempty
    -/
  · rcases hpc.1 with ⟨a, ha⟩
    /-
      case left.intro
      E : Type u_4
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module Real E
      inst✝² : TopologicalSpace E
      inst✝¹ : TopologicalAddGroup E
      inst✝ : ContinuousSMul Real E
      p q : Submodule Real E
      hpq : IsCompl p q
      hpc : And (HasCompl.compl (Singleton.singleton 0)).Nonempty (∀ (x : Subtype fu …
      a : Subtype fun x => Membership.mem p x
      ha : Membership.mem (HasCompl.compl (Singleton.singleton 0)) a
      ⊢ (HasCompl.compl ↑q).Nonempty
    -/
    exact ⟨a, mt (Submodule.eq_zero_of_coe_mem_of_disjoint hpq.disjoint) ha⟩
    /-
      🎉 no goals
    -/
    /-
      case right
      E : Type u_4
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module Real E
      inst✝² : TopologicalSpace E
      inst✝¹ : TopologicalAddGroup E
      inst✝ : ContinuousSMul Real E
      p q : Submodule Real E
      hpq : IsCompl p q
      hpc : And (HasCompl.compl (Singleton.singleton 0)).Nonempty (∀ (x : Subtype fu …
      ⊢ ∀ (x : E), Membership.mem (HasCompl.compl ↑q) x → ∀ (y : E), Membership.mem  …
    -/
  · intro x hx y hy
    have : π hpq x ≠ 0 ∧ π hpq y ≠ 0 := by
      constructor <;> intro h <;> rw [Submodule.linearProjOfIsCompl_apply_eq_zero_iff hpq] at h <;>
        [exact hx h; exact hy h]
    /-
      case right
      E : Type u_4
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module Real E
      inst✝² : TopologicalSpace E
      inst✝¹ : TopologicalAddGroup E
      inst✝ : ContinuousSMul Real E
      p q : Submodule Real E
      hpq : IsCompl p q
      hpc : And (HasCompl.compl (Singleton.singleton 0)).Nonempty (∀ (x : Subtype fu …
      x : E
      hx : Membership.mem (HasCompl.compl ↑q) x
      y : E
      hy : Membership.mem (HasCompl.compl ↑q) y
      this : And (Ne ((p.linearProjOfIsCompl q hpq) x) 0) (Ne ((p.linearProjOfIsComp …
      ⊢ JoinedIn (HasCompl.compl ↑q) x y
    -/
    rcases hpc.2 (π hpq x) this.1 (π hpq y) this.2 with ⟨γ₁, hγ₁⟩
    /-
      case right.intro
      E : Type u_4
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module Real E
      inst✝² : TopologicalSpace E
      inst✝¹ : TopologicalAddGroup E
      inst✝ : ContinuousSMul Real E
      p q : Submodule Real E
      hpq : IsCompl p q
      hpc : And (HasCompl.compl (Singleton.singleton 0)).Nonempty (∀ (x : Subtype fu …
      x : E
      hx : Membership.mem (HasCompl.compl ↑q) x
      y : E
      hy : Membership.mem (HasCompl.compl ↑q) y
      this : And (Ne ((p.linearProjOfIsCompl q hpq) x) 0) (Ne ((p.linearProjOfIsComp …
      γ₁ : Path ((p.linearProjOfIsCompl q hpq) x) ((p.linearProjOfIsCompl q hpq) y)
      hγ₁ : ∀ (t : ↑unitInterval), Membership.mem (HasCompl.compl (Singleton.singlet …
      ⊢ JoinedIn (HasCompl.compl ↑q) x y
    -/
    let γ₂ := PathConnectedSpace.somePath (π hpq.symm x) (π hpq.symm y)
    /-
      case right.intro
      E : Type u_4
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module Real E
      inst✝² : TopologicalSpace E
      inst✝¹ : TopologicalAddGroup E
      inst✝ : ContinuousSMul Real E
      p q : Submodule Real E
      hpq : IsCompl p q
      hpc : And (HasCompl.compl (Singleton.singleton 0)).Nonempty (∀ (x : Subtype fu …
      x : E
      hx : Membership.mem (HasCompl.compl ↑q) x
      y : E
      hy : Membership.mem (HasCompl.compl ↑q) y
      this : And (Ne ((p.linearProjOfIsCompl q hpq) x) 0) (Ne ((p.linearProjOfIsComp …
      γ₁ : Path ((p.linearProjOfIsCompl q hpq) x) ((p.linearProjOfIsCompl q hpq) y)
      hγ₁ : ∀ (t : ↑unitInterval), Membership.mem (HasCompl.compl (Singleton.singlet …
      γ₂ : Path ((q.linearProjOfIsCompl p ⋯) x) ((q.linearProjOfIsCompl p ⋯) y) := P …
      ⊢ JoinedIn (HasCompl.compl ↑q) x y
    -/
    let γ₁' : Path (_ : E) _ := γ₁.map continuous_subtype_val
    /-
      case right.intro
      E : Type u_4
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module Real E
      inst✝² : TopologicalSpace E
      inst✝¹ : TopologicalAddGroup E
      inst✝ : ContinuousSMul Real E
      p q : Submodule Real E
      hpq : IsCompl p q
      hpc : And (HasCompl.compl (Singleton.singleton 0)).Nonempty (∀ (x : Subtype fu …
      x : E
      hx : Membership.mem (HasCompl.compl ↑q) x
      y : E
      hy : Membership.mem (HasCompl.compl ↑q) y
      this : And (Ne ((p.linearProjOfIsCompl q hpq) x) 0) (Ne ((p.linearProjOfIsComp …
      γ₁ : Path ((p.linearProjOfIsCompl q hpq) x) ((p.linearProjOfIsCompl q hpq) y)
      hγ₁ : ∀ (t : ↑unitInterval), Membership.mem (HasCompl.compl (Singleton.singlet …
      γ₂ : Path ((q.linearProjOfIsCompl p ⋯) x) ((q.linearProjOfIsCompl p ⋯) y) := P …
      γ₁' : Path ↑((p.linearProjOfIsCompl q hpq) x) ↑((p.linearProjOfIsCompl q hpq)  …
      ⊢ JoinedIn (HasCompl.compl ↑q) x y
    -/
    let γ₂' : Path (_ : E) _ := γ₂.map continuous_subtype_val
    refine ⟨(γ₁'.add γ₂').cast (Submodule.linear_proj_add_linearProjOfIsCompl_eq_self hpq x).symm
      (Submodule.linear_proj_add_linearProjOfIsCompl_eq_self hpq y).symm, fun t ↦ ?_⟩
    /-
      case right.intro
      E : Type u_4
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module Real E
      inst✝² : TopologicalSpace E
      inst✝¹ : TopologicalAddGroup E
      inst✝ : ContinuousSMul Real E
      p q : Submodule Real E
      hpq : IsCompl p q
      hpc : And (HasCompl.compl (Singleton.singleton 0)).Nonempty (∀ (x : Subtype fu …
      x : E
      hx : Membership.mem (HasCompl.compl ↑q) x
      y : E
      hy : Membership.mem (HasCompl.compl ↑q) y
      this : And (Ne ((p.linearProjOfIsCompl q hpq) x) 0) (Ne ((p.linearProjOfIsComp …
      γ₁ : Path ((p.linearProjOfIsCompl q hpq) x) ((p.linearProjOfIsCompl q hpq) y)
      hγ₁ : ∀ (t : ↑unitInterval), Membership.mem (HasCompl.compl (Singleton.singlet …
      γ₂ : Path ((q.linearProjOfIsCompl p ⋯) x) ((q.linearProjOfIsCompl p ⋯) y) := P …
      γ₁' : Path ↑((p.linearProjOfIsCompl q hpq) x) ↑((p.linearProjOfIsCompl q hpq)  …
      γ₂' : Path ↑((q.linearProjOfIsCompl p ⋯) x) ↑((q.linearProjOfIsCompl p ⋯) y) : …
      t : ↑unitInterval
      ⊢ Membership.mem (HasCompl.compl ↑q) (((γ₁'.add γ₂').cast ⋯ ⋯) t)
    -/
    rw [Path.cast_coe, Path.add_apply]
    /-
      case right.intro
      E : Type u_4
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module Real E
      inst✝² : TopologicalSpace E
      inst✝¹ : TopologicalAddGroup E
      inst✝ : ContinuousSMul Real E
      p q : Submodule Real E
      hpq : IsCompl p q
      hpc : And (HasCompl.compl (Singleton.singleton 0)).Nonempty (∀ (x : Subtype fu …
      x : E
      hx : Membership.mem (HasCompl.compl ↑q) x
      y : E
      hy : Membership.mem (HasCompl.compl ↑q) y
      this : And (Ne ((p.linearProjOfIsCompl q hpq) x) 0) (Ne ((p.linearProjOfIsComp …
      γ₁ : Path ((p.linearProjOfIsCompl q hpq) x) ((p.linearProjOfIsCompl q hpq) y)
      hγ₁ : ∀ (t : ↑unitInterval), Membership.mem (HasCompl.compl (Singleton.singlet …
      γ₂ : Path ((q.linearProjOfIsCompl p ⋯) x) ((q.linearProjOfIsCompl p ⋯) y) := P …
      γ₁' : Path ↑((p.linearProjOfIsCompl q hpq) x) ↑((p.linearProjOfIsCompl q hpq)  …
      γ₂' : Path ↑((q.linearProjOfIsCompl p ⋯) x) ↑((q.linearProjOfIsCompl p ⋯) y) : …
      t : ↑unitInterval
      ⊢ Membership.mem (HasCompl.compl ↑q) (HAdd.hAdd (γ₁' t) (γ₂' t))
    -/
    change γ₁ t + (γ₂ t : E) ∉ q
    rw [← Submodule.linearProjOfIsCompl_apply_eq_zero_iff hpq, LinearMap.map_add,
      Submodule.linearProjOfIsCompl_apply_right, add_zero,
      Submodule.linearProjOfIsCompl_apply_eq_zero_iff]
    /-
      case right.intro
      E : Type u_4
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module Real E
      inst✝² : TopologicalSpace E
      inst✝¹ : TopologicalAddGroup E
      inst✝ : ContinuousSMul Real E
      p q : Submodule Real E
      hpq : IsCompl p q
      hpc : And (HasCompl.compl (Singleton.singleton 0)).Nonempty (∀ (x : Subtype fu …
      x : E
      hx : Membership.mem (HasCompl.compl ↑q) x
      y : E
      hy : Membership.mem (HasCompl.compl ↑q) y
      this : And (Ne ((p.linearProjOfIsCompl q hpq) x) 0) (Ne ((p.linearProjOfIsComp …
      γ₁ : Path ((p.linearProjOfIsCompl q hpq) x) ((p.linearProjOfIsCompl q hpq) y)
      hγ₁ : ∀ (t : ↑unitInterval), Membership.mem (HasCompl.compl (Singleton.singlet …
      γ₂ : Path ((q.linearProjOfIsCompl p ⋯) x) ((q.linearProjOfIsCompl p ⋯) y) := P …
      γ₁' : Path ↑((p.linearProjOfIsCompl q hpq) x) ↑((p.linearProjOfIsCompl q hpq)  …
      γ₂' : Path ↑((q.linearProjOfIsCompl p ⋯) x) ↑((q.linearProjOfIsCompl p ⋯) y) : …
      t : ↑unitInterval
      ⊢ Not (Membership.mem q ↑(γ₁ t))
    -/
    exact mt (Submodule.eq_zero_of_coe_mem_of_disjoint hpq.disjoint) (hγ₁ t)
    /-
      🎉 no goals
    -/


