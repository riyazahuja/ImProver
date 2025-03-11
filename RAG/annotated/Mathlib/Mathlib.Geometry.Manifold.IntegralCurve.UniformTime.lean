/-- This is the uniqueness theorem of integral curves applied to a real-indexed family of integral
  curves with the same starting point. -/
lemma eqOn_of_isIntegralCurveOn_Ioo [BoundarylessManifold I M]
    (hv : ContMDiff I I.tangent 1 (fun x ↦ (⟨x, v x⟩ : TangentBundle I M))) {x : M}
    (γ : ℝ → ℝ → M) (hγx : ∀ a, γ a 0 = x) (hγ : ∀ a > 0, IsIntegralCurveOn (γ a) v (Ioo (-a) a))
    {a a' : ℝ} (hpos : 0 < a') (hle : a' ≤ a) :
    EqOn (γ a') (γ a) (Ioo (-a') a') := by
  apply isIntegralCurveOn_Ioo_eqOn_of_contMDiff_boundaryless _ hv
    (hγ a' (by positivity)) ((hγ a (gt_of_ge_of_gt hle hpos)).mono _)
    (by rw [hγx a, hγx a'])
    /-
      E : Type u_1
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      H : Type u_2
      inst✝⁵ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : SmoothManifoldWithCorners I M
      inst✝¹ : T2Space M
      v : (x : M) → TangentSpace I x
      inst✝ : BoundarylessManifold I M
      hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
      x : M
      γ : Real → Real → M
      hγx : ∀ (a : Real), Eq (γ a 0) x
      hγ : ∀ (a : Real), GT.gt a 0 → IsIntegralCurveOn (γ a) v (Set.Ioo (Neg.neg a) a)
      a a' : Real
      hpos : LT.lt 0 a'
      hle : LE.le a' a
      ⊢ Membership.mem (Set.Ioo (Neg.neg a') a') 0
    -/
  · rw [mem_Ioo]
    /-
      E : Type u_1
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      H : Type u_2
      inst✝⁵ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : SmoothManifoldWithCorners I M
      inst✝¹ : T2Space M
      v : (x : M) → TangentSpace I x
      inst✝ : BoundarylessManifold I M
      hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
      x : M
      γ : Real → Real → M
      hγx : ∀ (a : Real), Eq (γ a 0) x
      hγ : ∀ (a : Real), GT.gt a 0 → IsIntegralCurveOn (γ a) v (Set.Ioo (Neg.neg a) a)
      a a' : Real
      hpos : LT.lt 0 a'
      hle : LE.le a' a
      ⊢ And (LT.lt (Neg.neg a') 0) (LT.lt 0 a')
    -/
    exact ⟨neg_lt_zero.mpr hpos, by positivity⟩
    /-
      🎉 no goals
    -/
    /-
      E : Type u_1
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      H : Type u_2
      inst✝⁵ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : SmoothManifoldWithCorners I M
      inst✝¹ : T2Space M
      v : (x : M) → TangentSpace I x
      inst✝ : BoundarylessManifold I M
      hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
      x : M
      γ : Real → Real → M
      hγx : ∀ (a : Real), Eq (γ a 0) x
      hγ : ∀ (a : Real), GT.gt a 0 → IsIntegralCurveOn (γ a) v (Set.Ioo (Neg.neg a) a)
      a a' : Real
      hpos : LT.lt 0 a'
      hle : LE.le a' a
      ⊢ HasSubset.Subset (Set.Ioo (Neg.neg a') a') (Set.Ioo (Neg.neg a) a)
    -/
                             /-
                               🎉 no goals
                             -/
  · apply Ioo_subset_Ioo <;> linarith
                             /-
                               🎉 no goals
                             -/


/-- For a family of integral curves `γ : ℝ → ℝ → M` with the same starting point `γ 0 = x` such that
  each `γ a` is defined on `Ioo (-a) a`, the global curve `γ_ext := fun t ↦ γ (|t| + 1) t` agrees
  with each `γ a` on `Ioo (-a) a`. This will help us show that `γ_ext` is a global integral
  curve. -/
lemma eqOn_abs_add_one_of_isIntegralCurveOn_Ioo [BoundarylessManifold I M]
    (hv : ContMDiff I I.tangent 1 (fun x ↦ (⟨x, v x⟩ : TangentBundle I M))) {x : M}
    (γ : ℝ → ℝ → M) (hγx : ∀ a, γ a 0 = x) (hγ : ∀ a > 0, IsIntegralCurveOn (γ a) v (Ioo (-a) a))
    {a : ℝ} : EqOn (fun t ↦ γ (|t| + 1) t) (γ a) (Ioo (-a) a) := by
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type u_2
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    v : (x : M) → TangentSpace I x
    inst✝ : BoundarylessManifold I M
    hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
    x : M
    γ : Real → Real → M
    hγx : ∀ (a : Real), Eq (γ a 0) x
    hγ : ∀ (a : Real), GT.gt a 0 → IsIntegralCurveOn (γ a) v (Set.Ioo (Neg.neg a) a)
    a : Real
    ⊢ Set.EqOn (fun t => γ (HAdd.hAdd (abs t) 1) t) (γ a) (Set.Ioo (Neg.neg a) a)
  -/
  intros t ht
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type u_2
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    v : (x : M) → TangentSpace I x
    inst✝ : BoundarylessManifold I M
    hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
    x : M
    γ : Real → Real → M
    hγx : ∀ (a : Real), Eq (γ a 0) x
    hγ : ∀ (a : Real), GT.gt a 0 → IsIntegralCurveOn (γ a) v (Set.Ioo (Neg.neg a) a)
    a t : Real
    ht : Membership.mem (Set.Ioo (Neg.neg a) a) t
    ⊢ Eq ((fun t => γ (HAdd.hAdd (abs t) 1) t) t) (γ a t)
  -/
  by_cases hlt : |t| + 1 < a
  · exact eqOn_of_isIntegralCurveOn_Ioo hv γ hγx hγ
      (by positivity) hlt.le (abs_lt.mp <| lt_add_one _)
  · exact eqOn_of_isIntegralCurveOn_Ioo hv γ hγx hγ
      (neg_lt_self_iff.mp <| lt_trans ht.1 ht.2) (not_lt.mp hlt) ht |>.symm


/-- For a family of integral curves `γ : ℝ → ℝ → M` with the same starting point `γ 0 = x` such that
  each `γ a` is defined on `Ioo (-a) a`, the function `γ_ext := fun t ↦ γ (|t| + 1) t` is a global
  integral curve. -/
lemma isIntegralCurve_abs_add_one_of_isIntegralCurveOn_Ioo [BoundarylessManifold I M]
    (hv : ContMDiff I I.tangent 1 (fun x ↦ (⟨x, v x⟩ : TangentBundle I M))) {x : M}
    (γ : ℝ → ℝ → M) (hγx : ∀ a, γ a 0 = x) (hγ : ∀ a > 0, IsIntegralCurveOn (γ a) v (Ioo (-a) a)) :
    IsIntegralCurve (fun t ↦ γ (|t| + 1) t) v := by
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type u_2
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    v : (x : M) → TangentSpace I x
    inst✝ : BoundarylessManifold I M
    hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
    x : M
    γ : Real → Real → M
    hγx : ∀ (a : Real), Eq (γ a 0) x
    hγ : ∀ (a : Real), GT.gt a 0 → IsIntegralCurveOn (γ a) v (Set.Ioo (Neg.neg a) a)
    ⊢ IsIntegralCurve (fun t => γ (HAdd.hAdd (abs t) 1) t) v
  -/
  intro t
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type u_2
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    v : (x : M) → TangentSpace I x
    inst✝ : BoundarylessManifold I M
    hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
    x : M
    γ : Real → Real → M
    hγx : ∀ (a : Real), Eq (γ a 0) x
    hγ : ∀ (a : Real), GT.gt a 0 → IsIntegralCurveOn (γ a) v (Set.Ioo (Neg.neg a) a)
    t : Real
    ⊢ HasMFDerivAt (modelWithCornersSelf Real Real) I (fun t => γ (HAdd.hAdd (abs  …
  -/
  apply HasMFDerivAt.congr_of_eventuallyEq (f := γ (|t| + 1))
    /-
      case h
      E : Type u_1
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      H : Type u_2
      inst✝⁵ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : SmoothManifoldWithCorners I M
      inst✝¹ : T2Space M
      v : (x : M) → TangentSpace I x
      inst✝ : BoundarylessManifold I M
      hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
      x : M
      γ : Real → Real → M
      hγx : ∀ (a : Real), Eq (γ a 0) x
      hγ : ∀ (a : Real), GT.gt a 0 → IsIntegralCurveOn (γ a) v (Set.Ioo (Neg.neg a) a)
      t : Real
      ⊢ HasMFDerivAt (modelWithCornersSelf Real Real) I (γ (HAdd.hAdd (abs t) 1)) t  …
    -/
  · apply hγ (|t| + 1) (by positivity)
    /-
      case h.a
      E : Type u_1
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      H : Type u_2
      inst✝⁵ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : SmoothManifoldWithCorners I M
      inst✝¹ : T2Space M
      v : (x : M) → TangentSpace I x
      inst✝ : BoundarylessManifold I M
      hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
      x : M
      γ : Real → Real → M
      hγx : ∀ (a : Real), Eq (γ a 0) x
      hγ : ∀ (a : Real), GT.gt a 0 → IsIntegralCurveOn (γ a) v (Set.Ioo (Neg.neg a) a)
      t : Real
      ⊢ Membership.mem (Set.Ioo (Neg.neg (HAdd.hAdd (abs t) 1)) (HAdd.hAdd (abs t) 1 …
    -/
    rw [mem_Ioo, ← abs_lt]
    /-
      case h.a
      E : Type u_1
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      H : Type u_2
      inst✝⁵ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : SmoothManifoldWithCorners I M
      inst✝¹ : T2Space M
      v : (x : M) → TangentSpace I x
      inst✝ : BoundarylessManifold I M
      hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
      x : M
      γ : Real → Real → M
      hγx : ∀ (a : Real), Eq (γ a 0) x
      hγ : ∀ (a : Real), GT.gt a 0 → IsIntegralCurveOn (γ a) v (Set.Ioo (Neg.neg a) a)
      t : Real
      ⊢ LT.lt (abs t) (HAdd.hAdd (abs t) 1)
    -/
    exact lt_add_one _
    /-
      🎉 no goals
    -/
    /-
      case h₁
      E : Type u_1
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      H : Type u_2
      inst✝⁵ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : SmoothManifoldWithCorners I M
      inst✝¹ : T2Space M
      v : (x : M) → TangentSpace I x
      inst✝ : BoundarylessManifold I M
      hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
      x : M
      γ : Real → Real → M
      hγx : ∀ (a : Real), Eq (γ a 0) x
      hγ : ∀ (a : Real), GT.gt a 0 → IsIntegralCurveOn (γ a) v (Set.Ioo (Neg.neg a) a)
      t : Real
      ⊢ (nhds t).EventuallyEq (fun t => γ (HAdd.hAdd (abs t) 1) t) (γ (HAdd.hAdd (ab …
    -/
  · rw [Filter.eventuallyEq_iff_exists_mem]
    refine ⟨Ioo (-(|t| + 1)) (|t| + 1), ?_,
      eqOn_abs_add_one_of_isIntegralCurveOn_Ioo hv γ hγx hγ⟩
    /-
      case h₁
      E : Type u_1
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      H : Type u_2
      inst✝⁵ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : SmoothManifoldWithCorners I M
      inst✝¹ : T2Space M
      v : (x : M) → TangentSpace I x
      inst✝ : BoundarylessManifold I M
      hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
      x : M
      γ : Real → Real → M
      hγx : ∀ (a : Real), Eq (γ a 0) x
      hγ : ∀ (a : Real), GT.gt a 0 → IsIntegralCurveOn (γ a) v (Set.Ioo (Neg.neg a) a)
      t : Real
      ⊢ Membership.mem (nhds t) (Set.Ioo (Neg.neg (HAdd.hAdd (abs t) 1)) (HAdd.hAdd  …
    -/
    have : |t| < |t| + 1 := lt_add_of_pos_right |t| zero_lt_one
    /-
      case h₁
      E : Type u_1
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      H : Type u_2
      inst✝⁵ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : SmoothManifoldWithCorners I M
      inst✝¹ : T2Space M
      v : (x : M) → TangentSpace I x
      inst✝ : BoundarylessManifold I M
      hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
      x : M
      γ : Real → Real → M
      hγx : ∀ (a : Real), Eq (γ a 0) x
      hγ : ∀ (a : Real), GT.gt a 0 → IsIntegralCurveOn (γ a) v (Set.Ioo (Neg.neg a) a)
      t : Real
      this : LT.lt (abs t) (HAdd.hAdd (abs t) 1)
      ⊢ Membership.mem (nhds t) (Set.Ioo (Neg.neg (HAdd.hAdd (abs t) 1)) (HAdd.hAdd  …
    -/
    rw [abs_lt] at this
    /-
      case h₁
      E : Type u_1
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      H : Type u_2
      inst✝⁵ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : SmoothManifoldWithCorners I M
      inst✝¹ : T2Space M
      v : (x : M) → TangentSpace I x
      inst✝ : BoundarylessManifold I M
      hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
      x : M
      γ : Real → Real → M
      hγx : ∀ (a : Real), Eq (γ a 0) x
      hγ : ∀ (a : Real), GT.gt a 0 → IsIntegralCurveOn (γ a) v (Set.Ioo (Neg.neg a) a)
      t : Real
      this : And (LT.lt (Neg.neg (HAdd.hAdd (abs t) 1)) t) (LT.lt t (HAdd.hAdd (abs  …
      ⊢ Membership.mem (nhds t) (Set.Ioo (Neg.neg (HAdd.hAdd (abs t) 1)) (HAdd.hAdd  …
    -/
    exact Ioo_mem_nhds this.1 this.2
    /-
      🎉 no goals
    -/


/-- The existence of a global integral curve is equivalent to the existence of a family of local
  integral curves `γ : ℝ → ℝ → M` with the same starting point `γ 0 = x` such that each `γ a` is
  defined on `Ioo (-a) a`. -/
lemma exists_isIntegralCurve_iff_exists_isIntegralCurveOn_Ioo [BoundarylessManifold I M]
    (hv : ContMDiff I I.tangent 1 (fun x ↦ (⟨x, v x⟩ : TangentBundle I M))) (x : M) :
    (∃ γ, γ 0 = x ∧ IsIntegralCurve γ v) ↔
      ∀ a, ∃ γ, γ 0 = x ∧ IsIntegralCurveOn γ v (Ioo (-a) a) := by
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type u_2
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    v : (x : M) → TangentSpace I x
    inst✝ : BoundarylessManifold I M
    hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
    x : M
    ⊢ Iff (Exists fun γ => And (Eq (γ 0) x) (IsIntegralCurve γ v)) (∀ (a : Real),  …
  -/
  refine ⟨fun ⟨γ, h1, h2⟩ _ ↦ ⟨γ, h1, h2.isIntegralCurveOn _⟩, fun h ↦ ?_⟩
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type u_2
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    v : (x : M) → TangentSpace I x
    inst✝ : BoundarylessManifold I M
    hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
    x : M
    h : ∀ (a : Real), Exists fun γ => And (Eq (γ 0) x) (IsIntegralCurveOn γ v (Set …
    ⊢ Exists fun γ => And (Eq (γ 0) x) (IsIntegralCurve γ v)
  -/
  choose γ hγx hγ using h
  exact ⟨fun t ↦ γ (|t| + 1) t, hγx (|0| + 1),
    isIntegralCurve_abs_add_one_of_isIntegralCurveOn_Ioo hv γ hγx (fun a _ ↦  hγ a)⟩


/-- Let `γ` and `γ'` be integral curves defined on `Ioo a b` and `Ioo a' b'`, respectively. Then,
  `piecewise (Ioo a b) γ γ'` is equal to `γ` and `γ'` in their respective domains.
  `Set.piecewise_eqOn` shows the equality for `γ` by definition, while this lemma shows the equality
  for `γ'` by the uniqueness of integral curves. -/
lemma eqOn_piecewise_of_isIntegralCurveOn_Ioo [BoundarylessManifold I M]
    (hv : ContMDiff I I.tangent 1 (fun x ↦ (⟨x, v x⟩ : TangentBundle I M)))
    {a b a' b' : ℝ} (hγ : IsIntegralCurveOn γ v (Ioo a b))
    (hγ' : IsIntegralCurveOn γ' v (Ioo a' b'))
    (ht₀ : t₀ ∈ Ioo a b ∩ Ioo a' b') (h : γ t₀ = γ' t₀) :
    EqOn (piecewise (Ioo a b) γ γ') γ' (Ioo a' b') := by
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type u_2
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    γ γ' : Real → M
    v : (x : M) → TangentSpace I x
    t₀ : Real
    inst✝ : BoundarylessManifold I M
    hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
    a b a' b' : Real
    hγ : IsIntegralCurveOn γ v (Set.Ioo a b)
    hγ' : IsIntegralCurveOn γ' v (Set.Ioo a' b')
    ht₀ : Membership.mem (Inter.inter (Set.Ioo a b) (Set.Ioo a' b')) t₀
    h : Eq (γ t₀) (γ' t₀)
    ⊢ Set.EqOn ((Set.Ioo a b).piecewise γ γ') γ' (Set.Ioo a' b')
  -/
  intros t ht
  suffices H : EqOn γ γ' (Ioo (max a a') (min b b')) by
    by_cases hmem : t ∈ Ioo a b
    · rw [piecewise, if_pos hmem]
      apply H
      simp [ht.1, ht.2, hmem.1, hmem.2]
    · rw [piecewise, if_neg hmem]
  apply isIntegralCurveOn_Ioo_eqOn_of_contMDiff_boundaryless _ hv
    (hγ.mono (Ioo_subset_Ioo (le_max_left ..) (min_le_left ..)))
    (hγ'.mono (Ioo_subset_Ioo (le_max_right ..) (min_le_right ..))) h
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type u_2
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    γ γ' : Real → M
    v : (x : M) → TangentSpace I x
    t₀ : Real
    inst✝ : BoundarylessManifold I M
    hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
    a b a' b' : Real
    hγ : IsIntegralCurveOn γ v (Set.Ioo a b)
    hγ' : IsIntegralCurveOn γ' v (Set.Ioo a' b')
    ht₀ : Membership.mem (Inter.inter (Set.Ioo a b) (Set.Ioo a' b')) t₀
    h : Eq (γ t₀) (γ' t₀)
    t : Real
    ht : Membership.mem (Set.Ioo a' b') t
    ⊢ Membership.mem (Set.Ioo (Max.max a a') (Min.min b b')) t₀
  -/
  exact ⟨max_lt ht₀.1.1 ht₀.2.1, lt_min ht₀.1.2 ht₀.2.2⟩
  /-
    🎉 no goals
  -/


/-- The extension of an integral curve by another integral curve is an integral curve.

  If two integral curves are defined on overlapping open intervals, and they agree at a point in
  their common domain, then they can be patched together to form a longer integral curve.

  This is stated for manifolds without boundary for simplicity. We actually only need to assume that
  the images of `γ` and `γ'` lie in the interior of the manifold. TODO: Generalise to manifolds with
  boundary. -/
lemma isIntegralCurveOn_piecewise [BoundarylessManifold I M]
    (hv : ContMDiff I I.tangent 1 (fun x ↦ (⟨x, v x⟩ : TangentBundle I M)))
    {a b a' b' : ℝ} (hγ : IsIntegralCurveOn γ v (Ioo a b))
    (hγ' : IsIntegralCurveOn γ' v (Ioo a' b')) {t₀ : ℝ}
    (ht₀ : t₀ ∈ Ioo a b ∩ Ioo a' b') (h : γ t₀ = γ' t₀) :
    IsIntegralCurveOn (piecewise (Ioo a b) γ γ') v (Ioo a b ∪ Ioo a' b') := by
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type u_2
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    γ γ' : Real → M
    v : (x : M) → TangentSpace I x
    inst✝ : BoundarylessManifold I M
    hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
    a b a' b' : Real
    hγ : IsIntegralCurveOn γ v (Set.Ioo a b)
    hγ' : IsIntegralCurveOn γ' v (Set.Ioo a' b')
    t₀ : Real
    ht₀ : Membership.mem (Inter.inter (Set.Ioo a b) (Set.Ioo a' b')) t₀
    h : Eq (γ t₀) (γ' t₀)
    ⊢ IsIntegralCurveOn ((Set.Ioo a b).piecewise γ γ') v (Union.union (Set.Ioo a b …
  -/
  intros t ht
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type u_2
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    γ γ' : Real → M
    v : (x : M) → TangentSpace I x
    inst✝ : BoundarylessManifold I M
    hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
    a b a' b' : Real
    hγ : IsIntegralCurveOn γ v (Set.Ioo a b)
    hγ' : IsIntegralCurveOn γ' v (Set.Ioo a' b')
    t₀ : Real
    ht₀ : Membership.mem (Inter.inter (Set.Ioo a b) (Set.Ioo a' b')) t₀
    h : Eq (γ t₀) (γ' t₀)
    t : Real
    ht : Membership.mem (Union.union (Set.Ioo a b) (Set.Ioo a' b')) t
    ⊢ HasMFDerivAt (modelWithCornersSelf Real Real) I ((Set.Ioo a b).piecewise γ γ …
  -/
  by_cases hmem : t ∈ Ioo a b
    /-
      case pos
      E : Type u_1
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      H : Type u_2
      inst✝⁵ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : SmoothManifoldWithCorners I M
      inst✝¹ : T2Space M
      γ γ' : Real → M
      v : (x : M) → TangentSpace I x
      inst✝ : BoundarylessManifold I M
      hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
      a b a' b' : Real
      hγ : IsIntegralCurveOn γ v (Set.Ioo a b)
      hγ' : IsIntegralCurveOn γ' v (Set.Ioo a' b')
      t₀ : Real
      ht₀ : Membership.mem (Inter.inter (Set.Ioo a b) (Set.Ioo a' b')) t₀
      h : Eq (γ t₀) (γ' t₀)
      t : Real
      ht : Membership.mem (Union.union (Set.Ioo a b) (Set.Ioo a' b')) t
      hmem : Membership.mem (Set.Ioo a b) t
      ⊢ HasMFDerivAt (modelWithCornersSelf Real Real) I ((Set.Ioo a b).piecewise γ γ …
    -/
  · rw [piecewise, if_pos hmem]
    /-
      case pos
      E : Type u_1
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      H : Type u_2
      inst✝⁵ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : SmoothManifoldWithCorners I M
      inst✝¹ : T2Space M
      γ γ' : Real → M
      v : (x : M) → TangentSpace I x
      inst✝ : BoundarylessManifold I M
      hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
      a b a' b' : Real
      hγ : IsIntegralCurveOn γ v (Set.Ioo a b)
      hγ' : IsIntegralCurveOn γ' v (Set.Ioo a' b')
      t₀ : Real
      ht₀ : Membership.mem (Inter.inter (Set.Ioo a b) (Set.Ioo a' b')) t₀
      h : Eq (γ t₀) (γ' t₀)
      t : Real
      ht : Membership.mem (Union.union (Set.Ioo a b) (Set.Ioo a' b')) t
      hmem : Membership.mem (Set.Ioo a b) t
      ⊢ HasMFDerivAt (modelWithCornersSelf Real Real) I ((Set.Ioo a b).piecewise γ γ …
    -/
    apply (hγ t hmem).congr_of_eventuallyEq
    /-
      case pos
      E : Type u_1
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      H : Type u_2
      inst✝⁵ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : SmoothManifoldWithCorners I M
      inst✝¹ : T2Space M
      γ γ' : Real → M
      v : (x : M) → TangentSpace I x
      inst✝ : BoundarylessManifold I M
      hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
      a b a' b' : Real
      hγ : IsIntegralCurveOn γ v (Set.Ioo a b)
      hγ' : IsIntegralCurveOn γ' v (Set.Ioo a' b')
      t₀ : Real
      ht₀ : Membership.mem (Inter.inter (Set.Ioo a b) (Set.Ioo a' b')) t₀
      h : Eq (γ t₀) (γ' t₀)
      t : Real
      ht : Membership.mem (Union.union (Set.Ioo a b) (Set.Ioo a' b')) t
      hmem : Membership.mem (Set.Ioo a b) t
      ⊢ (nhds t).EventuallyEq ((Set.Ioo a b).piecewise γ γ') γ
    -/
    rw [Filter.eventuallyEq_iff_exists_mem]
    /-
      case pos
      E : Type u_1
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      H : Type u_2
      inst✝⁵ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : SmoothManifoldWithCorners I M
      inst✝¹ : T2Space M
      γ γ' : Real → M
      v : (x : M) → TangentSpace I x
      inst✝ : BoundarylessManifold I M
      hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
      a b a' b' : Real
      hγ : IsIntegralCurveOn γ v (Set.Ioo a b)
      hγ' : IsIntegralCurveOn γ' v (Set.Ioo a' b')
      t₀ : Real
      ht₀ : Membership.mem (Inter.inter (Set.Ioo a b) (Set.Ioo a' b')) t₀
      h : Eq (γ t₀) (γ' t₀)
      t : Real
      ht : Membership.mem (Union.union (Set.Ioo a b) (Set.Ioo a' b')) t
      hmem : Membership.mem (Set.Ioo a b) t
      ⊢ Exists fun s => And (Membership.mem (nhds t) s) (Set.EqOn ((Set.Ioo a b).pie …
    -/
    refine ⟨Ioo a b, isOpen_Ioo.mem_nhds hmem, ?_⟩
    /-
      case pos
      E : Type u_1
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      H : Type u_2
      inst✝⁵ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : SmoothManifoldWithCorners I M
      inst✝¹ : T2Space M
      γ γ' : Real → M
      v : (x : M) → TangentSpace I x
      inst✝ : BoundarylessManifold I M
      hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
      a b a' b' : Real
      hγ : IsIntegralCurveOn γ v (Set.Ioo a b)
      hγ' : IsIntegralCurveOn γ' v (Set.Ioo a' b')
      t₀ : Real
      ht₀ : Membership.mem (Inter.inter (Set.Ioo a b) (Set.Ioo a' b')) t₀
      h : Eq (γ t₀) (γ' t₀)
      t : Real
      ht : Membership.mem (Union.union (Set.Ioo a b) (Set.Ioo a' b')) t
      hmem : Membership.mem (Set.Ioo a b) t
      ⊢ Set.EqOn ((Set.Ioo a b).piecewise γ γ') γ (Set.Ioo a b)
    -/
    intros t' ht'
    /-
      case pos
      E : Type u_1
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      H : Type u_2
      inst✝⁵ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : SmoothManifoldWithCorners I M
      inst✝¹ : T2Space M
      γ γ' : Real → M
      v : (x : M) → TangentSpace I x
      inst✝ : BoundarylessManifold I M
      hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
      a b a' b' : Real
      hγ : IsIntegralCurveOn γ v (Set.Ioo a b)
      hγ' : IsIntegralCurveOn γ' v (Set.Ioo a' b')
      t₀ : Real
      ht₀ : Membership.mem (Inter.inter (Set.Ioo a b) (Set.Ioo a' b')) t₀
      h : Eq (γ t₀) (γ' t₀)
      t : Real
      ht : Membership.mem (Union.union (Set.Ioo a b) (Set.Ioo a' b')) t
      hmem : Membership.mem (Set.Ioo a b) t
      t' : Real
      ht' : Membership.mem (Set.Ioo a b) t'
      ⊢ Eq ((Set.Ioo a b).piecewise γ γ' t') (γ t')
    -/
    rw [piecewise, if_pos ht']
    /-
      🎉 no goals
    -/
    /-
      case neg
      E : Type u_1
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      H : Type u_2
      inst✝⁵ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : SmoothManifoldWithCorners I M
      inst✝¹ : T2Space M
      γ γ' : Real → M
      v : (x : M) → TangentSpace I x
      inst✝ : BoundarylessManifold I M
      hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
      a b a' b' : Real
      hγ : IsIntegralCurveOn γ v (Set.Ioo a b)
      hγ' : IsIntegralCurveOn γ' v (Set.Ioo a' b')
      t₀ : Real
      ht₀ : Membership.mem (Inter.inter (Set.Ioo a b) (Set.Ioo a' b')) t₀
      h : Eq (γ t₀) (γ' t₀)
      t : Real
      ht : Membership.mem (Union.union (Set.Ioo a b) (Set.Ioo a' b')) t
      hmem : Not (Membership.mem (Set.Ioo a b) t)
      ⊢ HasMFDerivAt (modelWithCornersSelf Real Real) I ((Set.Ioo a b).piecewise γ γ …
    -/
  · rw [mem_union, or_iff_not_imp_left] at ht
    /-
      case neg
      E : Type u_1
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      H : Type u_2
      inst✝⁵ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : SmoothManifoldWithCorners I M
      inst✝¹ : T2Space M
      γ γ' : Real → M
      v : (x : M) → TangentSpace I x
      inst✝ : BoundarylessManifold I M
      hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
      a b a' b' : Real
      hγ : IsIntegralCurveOn γ v (Set.Ioo a b)
      hγ' : IsIntegralCurveOn γ' v (Set.Ioo a' b')
      t₀ : Real
      ht₀ : Membership.mem (Inter.inter (Set.Ioo a b) (Set.Ioo a' b')) t₀
      h : Eq (γ t₀) (γ' t₀)
      t : Real
      ht : Not (Membership.mem (Set.Ioo a b) t) → Membership.mem (Set.Ioo a' b') t
      hmem : Not (Membership.mem (Set.Ioo a b) t)
      ⊢ HasMFDerivAt (modelWithCornersSelf Real Real) I ((Set.Ioo a b).piecewise γ γ …
    -/
    rw [piecewise, if_neg hmem]
    /-
      case neg
      E : Type u_1
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      H : Type u_2
      inst✝⁵ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : SmoothManifoldWithCorners I M
      inst✝¹ : T2Space M
      γ γ' : Real → M
      v : (x : M) → TangentSpace I x
      inst✝ : BoundarylessManifold I M
      hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
      a b a' b' : Real
      hγ : IsIntegralCurveOn γ v (Set.Ioo a b)
      hγ' : IsIntegralCurveOn γ' v (Set.Ioo a' b')
      t₀ : Real
      ht₀ : Membership.mem (Inter.inter (Set.Ioo a b) (Set.Ioo a' b')) t₀
      h : Eq (γ t₀) (γ' t₀)
      t : Real
      ht : Not (Membership.mem (Set.Ioo a b) t) → Membership.mem (Set.Ioo a' b') t
      hmem : Not (Membership.mem (Set.Ioo a b) t)
      ⊢ HasMFDerivAt (modelWithCornersSelf Real Real) I ((Set.Ioo a b).piecewise γ γ …
    -/
    apply (hγ' t <| ht hmem).congr_of_eventuallyEq
    /-
      case neg
      E : Type u_1
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Real E
      H : Type u_2
      inst✝⁵ : TopologicalSpace H
      I : ModelWithCorners Real E H
      M : Type u_3
      inst✝⁴ : TopologicalSpace M
      inst✝³ : ChartedSpace H M
      inst✝² : SmoothManifoldWithCorners I M
      inst✝¹ : T2Space M
      γ γ' : Real → M
      v : (x : M) → TangentSpace I x
      inst✝ : BoundarylessManifold I M
      hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
      a b a' b' : Real
      hγ : IsIntegralCurveOn γ v (Set.Ioo a b)
      hγ' : IsIntegralCurveOn γ' v (Set.Ioo a' b')
      t₀ : Real
      ht₀ : Membership.mem (Inter.inter (Set.Ioo a b) (Set.Ioo a' b')) t₀
      h : Eq (γ t₀) (γ' t₀)
      t : Real
      ht : Not (Membership.mem (Set.Ioo a b) t) → Membership.mem (Set.Ioo a' b') t
      hmem : Not (Membership.mem (Set.Ioo a b) t)
      ⊢ (nhds t).EventuallyEq ((Set.Ioo a b).piecewise γ γ') γ'
    -/
    rw [Filter.eventuallyEq_iff_exists_mem]
    exact ⟨Ioo a' b', isOpen_Ioo.mem_nhds <| ht hmem,
      eqOn_piecewise_of_isIntegralCurveOn_Ioo hv hγ hγ' ht₀ h⟩


/-- If there exists `ε > 0` such that the local integral curve at each point `x : M` is defined at
  least on an open interval `Ioo (-ε) ε`, then every point on `M` has a global integral
  curve passing through it.

  See Lemma 9.15, [J.M. Lee (2012)][lee2012]. -/
lemma exists_isIntegralCurve_of_isIntegralCurveOn [BoundarylessManifold I M]
    {v : (x : M) → TangentSpace I x}
    (hv : ContMDiff I I.tangent 1 (fun x ↦ (⟨x, v x⟩ : TangentBundle I M)))
    {ε : ℝ} (hε : 0 < ε) (h : ∀ x : M, ∃ γ : ℝ → M, γ 0 = x ∧ IsIntegralCurveOn γ v (Ioo (-ε) ε))
    (x : M) : ∃ γ : ℝ → M, γ 0 = x ∧ IsIntegralCurve γ v := by
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type u_2
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    inst✝ : BoundarylessManifold I M
    v : (x : M) → TangentSpace I x
    hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
    ε : Real
    hε : LT.lt 0 ε
    h : ∀ (x : M), Exists fun γ => And (Eq (γ 0) x) (IsIntegralCurveOn γ v (Set.Io …
    x : M
    ⊢ Exists fun γ => And (Eq (γ 0) x) (IsIntegralCurve γ v)
  -/
  let s := { a | ∃ γ, γ 0 = x ∧ IsIntegralCurveOn γ v (Ioo (-a) a) }
  suffices hbdd : ¬BddAbove s by
    rw [not_bddAbove_iff] at hbdd
    rw [exists_isIntegralCurve_iff_exists_isIntegralCurveOn_Ioo hv]
    intro a
    obtain ⟨y, ⟨γ, hγ1, hγ2⟩, hlt⟩ := hbdd a
    exact ⟨γ, hγ1, hγ2.mono <| Ioo_subset_Ioo (neg_le_neg hlt.le) hlt.le⟩
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type u_2
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    inst✝ : BoundarylessManifold I M
    v : (x : M) → TangentSpace I x
    hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
    ε : Real
    hε : LT.lt 0 ε
    h : ∀ (x : M), Exists fun γ => And (Eq (γ 0) x) (IsIntegralCurveOn γ v (Set.Io …
    x : M
    s : Set Real := setOf fun a => Exists fun γ => And (Eq (γ 0) x) (IsIntegralCur …
    ⊢ Not (BddAbove s)
  -/
  intro hbdd
  /-
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type u_2
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    inst✝ : BoundarylessManifold I M
    v : (x : M) → TangentSpace I x
    hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
    ε : Real
    hε : LT.lt 0 ε
    h : ∀ (x : M), Exists fun γ => And (Eq (γ 0) x) (IsIntegralCurveOn γ v (Set.Io …
    x : M
    s : Set Real := setOf fun a => Exists fun γ => And (Eq (γ 0) x) (IsIntegralCur …
    hbdd : BddAbove s
    ⊢ False
  -/
  set asup := sSup s with hasup
  -- we will obtain two integral curves, one centred at some `t₀ > 0` with
  -- `0 ≤ asup - ε < t₀ < asup`; let `t₀ = asup - ε / 2`
  -- another centred at 0 with domain up to `a ∈ S` with `t₀ < a < asup`
  obtain ⟨a, ha, hlt⟩ := Real.add_neg_lt_sSup (⟨ε, h x⟩ : Set.Nonempty s) (ε := - (ε / 2))
    (by rw [neg_lt, neg_zero]; exact half_pos hε)
  /-
    case intro.intro
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type u_2
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    inst✝ : BoundarylessManifold I M
    v : (x : M) → TangentSpace I x
    hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
    ε : Real
    hε : LT.lt 0 ε
    h : ∀ (x : M), Exists fun γ => And (Eq (γ 0) x) (IsIntegralCurveOn γ v (Set.Io …
    x : M
    s : Set Real := setOf fun a => Exists fun γ => And (Eq (γ 0) x) (IsIntegralCur …
    hbdd : BddAbove s
    asup : Real := SupSet.sSup s
    hasup : Eq asup (SupSet.sSup s)
    a : Real
    ha : Membership.mem s a
    hlt : LT.lt (HAdd.hAdd (SupSet.sSup s) (Neg.neg (HDiv.hDiv ε 2))) a
    ⊢ False
  -/
  rw [mem_setOf] at ha
  /-
    case intro.intro
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type u_2
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    inst✝ : BoundarylessManifold I M
    v : (x : M) → TangentSpace I x
    hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
    ε : Real
    hε : LT.lt 0 ε
    h : ∀ (x : M), Exists fun γ => And (Eq (γ 0) x) (IsIntegralCurveOn γ v (Set.Io …
    x : M
    s : Set Real := setOf fun a => Exists fun γ => And (Eq (γ 0) x) (IsIntegralCur …
    hbdd : BddAbove s
    asup : Real := SupSet.sSup s
    hasup : Eq asup (SupSet.sSup s)
    a : Real
    ha : Exists fun γ => And (Eq (γ 0) x) (IsIntegralCurveOn γ v (Set.Ioo (Neg.neg …
    hlt : LT.lt (HAdd.hAdd (SupSet.sSup s) (Neg.neg (HDiv.hDiv ε 2))) a
    ⊢ False
  -/
  rw [← hasup, ← sub_eq_add_neg] at hlt

  -- integral curve defined on `Ioo (-a) a`
  /-
    case intro.intro
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type u_2
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    inst✝ : BoundarylessManifold I M
    v : (x : M) → TangentSpace I x
    hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
    ε : Real
    hε : LT.lt 0 ε
    h : ∀ (x : M), Exists fun γ => And (Eq (γ 0) x) (IsIntegralCurveOn γ v (Set.Io …
    x : M
    s : Set Real := setOf fun a => Exists fun γ => And (Eq (γ 0) x) (IsIntegralCur …
    hbdd : BddAbove s
    asup : Real := SupSet.sSup s
    hasup : Eq asup (SupSet.sSup s)
    a : Real
    ha : Exists fun γ => And (Eq (γ 0) x) (IsIntegralCurveOn γ v (Set.Ioo (Neg.neg …
    hlt : LT.lt (HSub.hSub asup (HDiv.hDiv ε 2)) a
    ⊢ False
  -/
  obtain ⟨γ, h0, hγ⟩ := ha
  -- integral curve starting at `-(asup - ε / 2)` with radius `ε`
  /-
    case intro.intro.intro.intro
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type u_2
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    inst✝ : BoundarylessManifold I M
    v : (x : M) → TangentSpace I x
    hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
    ε : Real
    hε : LT.lt 0 ε
    h : ∀ (x : M), Exists fun γ => And (Eq (γ 0) x) (IsIntegralCurveOn γ v (Set.Io …
    x : M
    s : Set Real := setOf fun a => Exists fun γ => And (Eq (γ 0) x) (IsIntegralCur …
    hbdd : BddAbove s
    asup : Real := SupSet.sSup s
    hasup : Eq asup (SupSet.sSup s)
    a : Real
    hlt : LT.lt (HSub.hSub asup (HDiv.hDiv ε 2)) a
    γ : Real → M
    h0 : Eq (γ 0) x
    hγ : IsIntegralCurveOn γ v (Set.Ioo (Neg.neg a) a)
    ⊢ False
  -/
  obtain ⟨γ1_aux, h1_aux, hγ1⟩ := h (γ (-(asup - ε / 2)))
  /-
    case intro.intro.intro.intro.intro.intro
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type u_2
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    inst✝ : BoundarylessManifold I M
    v : (x : M) → TangentSpace I x
    hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
    ε : Real
    hε : LT.lt 0 ε
    h : ∀ (x : M), Exists fun γ => And (Eq (γ 0) x) (IsIntegralCurveOn γ v (Set.Io …
    x : M
    s : Set Real := setOf fun a => Exists fun γ => And (Eq (γ 0) x) (IsIntegralCur …
    hbdd : BddAbove s
    asup : Real := SupSet.sSup s
    hasup : Eq asup (SupSet.sSup s)
    a : Real
    hlt : LT.lt (HSub.hSub asup (HDiv.hDiv ε 2)) a
    γ : Real → M
    h0 : Eq (γ 0) x
    hγ : IsIntegralCurveOn γ v (Set.Ioo (Neg.neg a) a)
    γ1_aux : Real → M
    h1_aux : Eq (γ1_aux 0) (γ (Neg.neg (HSub.hSub asup (HDiv.hDiv ε 2))))
    hγ1 : IsIntegralCurveOn γ1_aux v (Set.Ioo (Neg.neg ε) ε)
    ⊢ False
  -/
  rw [isIntegralCurveOn_comp_add (dt := asup - ε / 2)] at hγ1
  /-
    case intro.intro.intro.intro.intro.intro
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type u_2
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    inst✝ : BoundarylessManifold I M
    v : (x : M) → TangentSpace I x
    hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
    ε : Real
    hε : LT.lt 0 ε
    h : ∀ (x : M), Exists fun γ => And (Eq (γ 0) x) (IsIntegralCurveOn γ v (Set.Io …
    x : M
    s : Set Real := setOf fun a => Exists fun γ => And (Eq (γ 0) x) (IsIntegralCur …
    hbdd : BddAbove s
    asup : Real := SupSet.sSup s
    hasup : Eq asup (SupSet.sSup s)
    a : Real
    hlt : LT.lt (HSub.hSub asup (HDiv.hDiv ε 2)) a
    γ : Real → M
    h0 : Eq (γ 0) x
    hγ : IsIntegralCurveOn γ v (Set.Ioo (Neg.neg a) a)
    γ1_aux : Real → M
    h1_aux : Eq (γ1_aux 0) (γ (Neg.neg (HSub.hSub asup (HDiv.hDiv ε 2))))
    hγ1 : IsIntegralCurveOn (Function.comp γ1_aux fun x => HAdd.hAdd x (HSub.hSub  …
    ⊢ False
  -/
  set γ1 := γ1_aux ∘ (· + (asup - ε / 2)) with γ1_def
  /-
    case intro.intro.intro.intro.intro.intro
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type u_2
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    inst✝ : BoundarylessManifold I M
    v : (x : M) → TangentSpace I x
    hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
    ε : Real
    hε : LT.lt 0 ε
    h : ∀ (x : M), Exists fun γ => And (Eq (γ 0) x) (IsIntegralCurveOn γ v (Set.Io …
    x : M
    s : Set Real := setOf fun a => Exists fun γ => And (Eq (γ 0) x) (IsIntegralCur …
    hbdd : BddAbove s
    asup : Real := SupSet.sSup s
    hasup : Eq asup (SupSet.sSup s)
    a : Real
    hlt : LT.lt (HSub.hSub asup (HDiv.hDiv ε 2)) a
    γ : Real → M
    h0 : Eq (γ 0) x
    hγ : IsIntegralCurveOn γ v (Set.Ioo (Neg.neg a) a)
    γ1_aux : Real → M
    h1_aux : Eq (γ1_aux 0) (γ (Neg.neg (HSub.hSub asup (HDiv.hDiv ε 2))))
    γ1 : Real → M := Function.comp γ1_aux fun x => HAdd.hAdd x (HSub.hSub asup (HD …
    hγ1 : IsIntegralCurveOn γ1 v (HVAdd.hVAdd (Neg.neg (HSub.hSub asup (HDiv.hDiv  …
    γ1_def : Eq γ1 (Function.comp γ1_aux fun x => HAdd.hAdd x (HSub.hSub asup (HDi …
    ⊢ False
  -/
  have heq1 : γ1 (-(asup - ε / 2)) = γ (-(asup - ε / 2)) := by simp [γ1_def, h1_aux]
  -- integral curve starting at `asup - ε / 2` with radius `ε`
  /-
    case intro.intro.intro.intro.intro.intro
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type u_2
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    inst✝ : BoundarylessManifold I M
    v : (x : M) → TangentSpace I x
    hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
    ε : Real
    hε : LT.lt 0 ε
    h : ∀ (x : M), Exists fun γ => And (Eq (γ 0) x) (IsIntegralCurveOn γ v (Set.Io …
    x : M
    s : Set Real := setOf fun a => Exists fun γ => And (Eq (γ 0) x) (IsIntegralCur …
    hbdd : BddAbove s
    asup : Real := SupSet.sSup s
    hasup : Eq asup (SupSet.sSup s)
    a : Real
    hlt : LT.lt (HSub.hSub asup (HDiv.hDiv ε 2)) a
    γ : Real → M
    h0 : Eq (γ 0) x
    hγ : IsIntegralCurveOn γ v (Set.Ioo (Neg.neg a) a)
    γ1_aux : Real → M
    h1_aux : Eq (γ1_aux 0) (γ (Neg.neg (HSub.hSub asup (HDiv.hDiv ε 2))))
    γ1 : Real → M := Function.comp γ1_aux fun x => HAdd.hAdd x (HSub.hSub asup (HD …
    hγ1 : IsIntegralCurveOn γ1 v (HVAdd.hVAdd (Neg.neg (HSub.hSub asup (HDiv.hDiv  …
    γ1_def : Eq γ1 (Function.comp γ1_aux fun x => HAdd.hAdd x (HSub.hSub asup (HDi …
    heq1 : Eq (γ1 (Neg.neg (HSub.hSub asup (HDiv.hDiv ε 2)))) (γ (Neg.neg (HSub.hS …
    ⊢ False
  -/
  obtain ⟨γ2_aux, h2_aux, hγ2⟩ := h (γ (asup - ε / 2))
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type u_2
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    inst✝ : BoundarylessManifold I M
    v : (x : M) → TangentSpace I x
    hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
    ε : Real
    hε : LT.lt 0 ε
    h : ∀ (x : M), Exists fun γ => And (Eq (γ 0) x) (IsIntegralCurveOn γ v (Set.Io …
    x : M
    s : Set Real := setOf fun a => Exists fun γ => And (Eq (γ 0) x) (IsIntegralCur …
    hbdd : BddAbove s
    asup : Real := SupSet.sSup s
    hasup : Eq asup (SupSet.sSup s)
    a : Real
    hlt : LT.lt (HSub.hSub asup (HDiv.hDiv ε 2)) a
    γ : Real → M
    h0 : Eq (γ 0) x
    hγ : IsIntegralCurveOn γ v (Set.Ioo (Neg.neg a) a)
    γ1_aux : Real → M
    h1_aux : Eq (γ1_aux 0) (γ (Neg.neg (HSub.hSub asup (HDiv.hDiv ε 2))))
    γ1 : Real → M := Function.comp γ1_aux fun x => HAdd.hAdd x (HSub.hSub asup (HD …
    hγ1 : IsIntegralCurveOn γ1 v (HVAdd.hVAdd (Neg.neg (HSub.hSub asup (HDiv.hDiv  …
    γ1_def : Eq γ1 (Function.comp γ1_aux fun x => HAdd.hAdd x (HSub.hSub asup (HDi …
    heq1 : Eq (γ1 (Neg.neg (HSub.hSub asup (HDiv.hDiv ε 2)))) (γ (Neg.neg (HSub.hS …
    γ2_aux : Real → M
    h2_aux : Eq (γ2_aux 0) (γ (HSub.hSub asup (HDiv.hDiv ε 2)))
    hγ2 : IsIntegralCurveOn γ2_aux v (Set.Ioo (Neg.neg ε) ε)
    ⊢ False
  -/
  rw [isIntegralCurveOn_comp_sub (dt := asup - ε / 2)] at hγ2
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type u_2
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    inst✝ : BoundarylessManifold I M
    v : (x : M) → TangentSpace I x
    hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
    ε : Real
    hε : LT.lt 0 ε
    h : ∀ (x : M), Exists fun γ => And (Eq (γ 0) x) (IsIntegralCurveOn γ v (Set.Io …
    x : M
    s : Set Real := setOf fun a => Exists fun γ => And (Eq (γ 0) x) (IsIntegralCur …
    hbdd : BddAbove s
    asup : Real := SupSet.sSup s
    hasup : Eq asup (SupSet.sSup s)
    a : Real
    hlt : LT.lt (HSub.hSub asup (HDiv.hDiv ε 2)) a
    γ : Real → M
    h0 : Eq (γ 0) x
    hγ : IsIntegralCurveOn γ v (Set.Ioo (Neg.neg a) a)
    γ1_aux : Real → M
    h1_aux : Eq (γ1_aux 0) (γ (Neg.neg (HSub.hSub asup (HDiv.hDiv ε 2))))
    γ1 : Real → M := Function.comp γ1_aux fun x => HAdd.hAdd x (HSub.hSub asup (HD …
    hγ1 : IsIntegralCurveOn γ1 v (HVAdd.hVAdd (Neg.neg (HSub.hSub asup (HDiv.hDiv  …
    γ1_def : Eq γ1 (Function.comp γ1_aux fun x => HAdd.hAdd x (HSub.hSub asup (HDi …
    heq1 : Eq (γ1 (Neg.neg (HSub.hSub asup (HDiv.hDiv ε 2)))) (γ (Neg.neg (HSub.hS …
    γ2_aux : Real → M
    h2_aux : Eq (γ2_aux 0) (γ (HSub.hSub asup (HDiv.hDiv ε 2)))
    hγ2 : IsIntegralCurveOn (Function.comp γ2_aux fun x => HSub.hSub x (HSub.hSub  …
    ⊢ False
  -/
  set γ2 := γ2_aux ∘ (· - (asup - ε / 2)) with γ2_def
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type u_2
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    inst✝ : BoundarylessManifold I M
    v : (x : M) → TangentSpace I x
    hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
    ε : Real
    hε : LT.lt 0 ε
    h : ∀ (x : M), Exists fun γ => And (Eq (γ 0) x) (IsIntegralCurveOn γ v (Set.Io …
    x : M
    s : Set Real := setOf fun a => Exists fun γ => And (Eq (γ 0) x) (IsIntegralCur …
    hbdd : BddAbove s
    asup : Real := SupSet.sSup s
    hasup : Eq asup (SupSet.sSup s)
    a : Real
    hlt : LT.lt (HSub.hSub asup (HDiv.hDiv ε 2)) a
    γ : Real → M
    h0 : Eq (γ 0) x
    hγ : IsIntegralCurveOn γ v (Set.Ioo (Neg.neg a) a)
    γ1_aux : Real → M
    h1_aux : Eq (γ1_aux 0) (γ (Neg.neg (HSub.hSub asup (HDiv.hDiv ε 2))))
    γ1 : Real → M := Function.comp γ1_aux fun x => HAdd.hAdd x (HSub.hSub asup (HD …
    hγ1 : IsIntegralCurveOn γ1 v (HVAdd.hVAdd (Neg.neg (HSub.hSub asup (HDiv.hDiv  …
    γ1_def : Eq γ1 (Function.comp γ1_aux fun x => HAdd.hAdd x (HSub.hSub asup (HDi …
    heq1 : Eq (γ1 (Neg.neg (HSub.hSub asup (HDiv.hDiv ε 2)))) (γ (Neg.neg (HSub.hS …
    γ2_aux : Real → M
    h2_aux : Eq (γ2_aux 0) (γ (HSub.hSub asup (HDiv.hDiv ε 2)))
    γ2 : Real → M := Function.comp γ2_aux fun x => HSub.hSub x (HSub.hSub asup (HD …
    hγ2 : IsIntegralCurveOn γ2 v (HVAdd.hVAdd (HSub.hSub asup (HDiv.hDiv ε 2)) (Se …
    γ2_def : Eq γ2 (Function.comp γ2_aux fun x => HSub.hSub x (HSub.hSub asup (HDi …
    ⊢ False
  -/
  have heq2 : γ2 (asup - ε / 2) = γ (asup - ε / 2) := by simp [γ2_def, h2_aux]

  -- rewrite shifted Ioo as Ioo
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type u_2
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    inst✝ : BoundarylessManifold I M
    v : (x : M) → TangentSpace I x
    hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
    ε : Real
    hε : LT.lt 0 ε
    h : ∀ (x : M), Exists fun γ => And (Eq (γ 0) x) (IsIntegralCurveOn γ v (Set.Io …
    x : M
    s : Set Real := setOf fun a => Exists fun γ => And (Eq (γ 0) x) (IsIntegralCur …
    hbdd : BddAbove s
    asup : Real := SupSet.sSup s
    hasup : Eq asup (SupSet.sSup s)
    a : Real
    hlt : LT.lt (HSub.hSub asup (HDiv.hDiv ε 2)) a
    γ : Real → M
    h0 : Eq (γ 0) x
    hγ : IsIntegralCurveOn γ v (Set.Ioo (Neg.neg a) a)
    γ1_aux : Real → M
    h1_aux : Eq (γ1_aux 0) (γ (Neg.neg (HSub.hSub asup (HDiv.hDiv ε 2))))
    γ1 : Real → M := Function.comp γ1_aux fun x => HAdd.hAdd x (HSub.hSub asup (HD …
    hγ1 : IsIntegralCurveOn γ1 v (HVAdd.hVAdd (Neg.neg (HSub.hSub asup (HDiv.hDiv  …
    γ1_def : Eq γ1 (Function.comp γ1_aux fun x => HAdd.hAdd x (HSub.hSub asup (HDi …
    heq1 : Eq (γ1 (Neg.neg (HSub.hSub asup (HDiv.hDiv ε 2)))) (γ (Neg.neg (HSub.hS …
    γ2_aux : Real → M
    h2_aux : Eq (γ2_aux 0) (γ (HSub.hSub asup (HDiv.hDiv ε 2)))
    γ2 : Real → M := Function.comp γ2_aux fun x => HSub.hSub x (HSub.hSub asup (HD …
    hγ2 : IsIntegralCurveOn γ2 v (HVAdd.hVAdd (HSub.hSub asup (HDiv.hDiv ε 2)) (Se …
    γ2_def : Eq γ2 (Function.comp γ2_aux fun x => HSub.hSub x (HSub.hSub asup (HDi …
    heq2 : Eq (γ2 (HSub.hSub asup (HDiv.hDiv ε 2))) (γ (HSub.hSub asup (HDiv.hDiv  …
    ⊢ False
  -/
  rw [neg_sub] at hγ1
  rw [Real.Ioo_eq_ball, neg_add_cancel, zero_div, sub_neg_eq_add, add_self_div_two,
    Metric.vadd_ball, vadd_eq_add, add_zero, Real.ball_eq_Ioo] at hγ1 hγ2

  -- to help `linarith`
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    E : Type u_1
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Real E
    H : Type u_2
    inst✝⁵ : TopologicalSpace H
    I : ModelWithCorners Real E H
    M : Type u_3
    inst✝⁴ : TopologicalSpace M
    inst✝³ : ChartedSpace H M
    inst✝² : SmoothManifoldWithCorners I M
    inst✝¹ : T2Space M
    inst✝ : BoundarylessManifold I M
    v : (x : M) → TangentSpace I x
    hv : ContMDiff I I.tangent 1 fun x => { proj := x, snd := v x }
    ε : Real
    hε : LT.lt 0 ε
    h : ∀ (x : M), Exists fun γ => And (Eq (γ 0) x) (IsIntegralCurveOn γ v (Set.Io …
    x : M
    s : Set Real := setOf fun a => Exists fun γ => And (Eq (γ 0) x) (IsIntegralCur …
    hbdd : BddAbove s
    asup : Real := SupSet.sSup s
    hasup : Eq asup (SupSet.sSup s)
    a : Real
    hlt : LT.lt (HSub.hSub asup (HDiv.hDiv ε 2)) a
    γ : Real → M
    h0 : Eq (γ 0) x
    hγ : IsIntegralCurveOn γ v (Set.Ioo (Neg.neg a) a)
    γ1_aux : Real → M
    h1_aux : Eq (γ1_aux 0) (γ (Neg.neg (HSub.hSub asup (HDiv.hDiv ε 2))))
    γ1 : Real → M := Function.comp γ1_aux fun x => HAdd.hAdd x (HSub.hSub asup (HD …
    hγ1 : IsIntegralCurveOn γ1 v (Set.Ioo (HSub.hSub (HSub.hSub (HDiv.hDiv ε 2) as …
    γ1_def : Eq γ1 (Function.comp γ1_aux fun x => HAdd.hAdd x (HSub.hSub asup (HDi …
    heq1 : Eq (γ1 (Neg.neg (HSub.hSub asup (HDiv.hDiv ε 2)))) (γ (Neg.neg (HSub.hS …
    γ2_aux : Real → M
    h2_aux : Eq (γ2_aux 0) (γ (HSub.hSub asup (HDiv.hDiv ε 2)))
    γ2 : Real → M := Function.comp γ2_aux fun x => HSub.hSub x (HSub.hSub asup (HD …
    hγ2 : IsIntegralCurveOn γ2 v (Set.Ioo (HSub.hSub (HSub.hSub asup (HDiv.hDiv ε  …
    γ2_def : Eq γ2 (Function.comp γ2_aux fun x => HSub.hSub x (HSub.hSub asup (HDi …
    heq2 : Eq (γ2 (HSub.hSub asup (HDiv.hDiv ε 2))) (γ (HSub.hSub asup (HDiv.hDiv  …
    ⊢ False
  -/
  have hεle : ε ≤ asup := le_csSup hbdd (h x)

  -- extend `γ` on the left by `γ1` and on the right by `γ2`
  set γ_ext : ℝ → M := piecewise (Ioo (-(asup + ε / 2)) a)
    (piecewise (Ioo (-a) a) γ γ1) γ2 with γ_ext_def
  have heq_ext : γ_ext 0 = x := by
    rw [γ_ext_def, piecewise, if_pos ⟨by linarith, by linarith⟩, piecewise,
      if_pos ⟨by linarith, by linarith⟩, h0]
  -- `asup + ε / 2` is an element of `s` greater than `asup`, a contradiction
  suffices hext : IsIntegralCurveOn γ_ext v (Ioo (-(asup + ε / 2)) (asup + ε / 2)) from
    (not_lt.mpr <| le_csSup hbdd ⟨γ_ext, heq_ext, hext⟩) <| lt_add_of_pos_right asup (half_pos hε)
  apply (isIntegralCurveOn_piecewise (t₀ := asup - ε / 2) hv _ hγ2
      ⟨⟨by linarith, hlt⟩, ⟨by linarith, by linarith⟩⟩
      (by rw [piecewise, if_pos ⟨by linarith, hlt⟩, ← heq2])).mono
    (Ioo_subset_Ioo_union_Ioo le_rfl (by linarith) (by linarith))
  exact (isIntegralCurveOn_piecewise (t₀ := -(asup - ε / 2)) hv hγ hγ1
      ⟨⟨neg_lt_neg hlt, by linarith⟩, ⟨by linarith, by linarith⟩⟩ heq1.symm).mono
    (union_comm _ _ ▸ Ioo_subset_Ioo_union_Ioo (by linarith) (by linarith) le_rfl)

