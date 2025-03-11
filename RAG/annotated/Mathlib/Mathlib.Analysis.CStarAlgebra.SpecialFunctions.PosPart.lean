/-- A C⋆-algebra is spanned by nonnegative elements of norm at most `r` -/
lemma span_nonneg_inter_closedBall {r : ℝ} (hr : 0 < r) :
    span ℂ ({x : A | 0 ≤ x} ∩ Metric.closedBall 0 r) = ⊤ := by
  /-
    A : Type u_1
    inst✝² : NonUnitalCStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    r : Real
    hr : LT.lt 0 r
    ⊢ Eq (Submodule.span Complex (Inter.inter (setOf fun x => LE.le 0 x) (Metric.c …
  -/
  rw [eq_top_iff, ← span_nonneg, span_le]
  /-
    A : Type u_1
    inst✝² : NonUnitalCStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    r : Real
    hr : LT.lt 0 r
    ⊢ HasSubset.Subset (setOf fun a => LE.le 0 a) ↑(Submodule.span Complex (Inter. …
  -/
  intro x hx
  /-
    A : Type u_1
    inst✝² : NonUnitalCStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    r : Real
    hr : LT.lt 0 r
    x : A
    hx : Membership.mem (setOf fun a => LE.le 0 a) x
    ⊢ Membership.mem (↑(Submodule.span Complex (Inter.inter (setOf fun x => LE.le  …
  -/
  obtain (rfl | hx_pos) := eq_zero_or_norm_pos x
    /-
      case inl
      A : Type u_1
      inst✝² : NonUnitalCStarAlgebra A
      inst✝¹ : PartialOrder A
      inst✝ : StarOrderedRing A
      r : Real
      hr : LT.lt 0 r
      hx : Membership.mem (setOf fun a => LE.le 0 a) 0
      ⊢ Membership.mem (↑(Submodule.span Complex (Inter.inter (setOf fun x => LE.le  …
    -/
  · exact zero_mem _
    /-
      🎉 no goals
    -/
  · suffices (r * ‖x‖⁻¹ : ℂ)⁻¹ • ((r * ‖x‖⁻¹ : ℂ) • x) = x by
      rw [← this]
      refine smul_mem _ _ (subset_span <| Set.mem_inter ?_ ?_)
      · norm_cast
        exact smul_nonneg (by positivity) hx
      · simp [mul_smul, norm_smul, abs_of_pos hr, inv_mul_cancel₀ hx_pos.ne']
    /-
      case inr
      A : Type u_1
      inst✝² : NonUnitalCStarAlgebra A
      inst✝¹ : PartialOrder A
      inst✝ : StarOrderedRing A
      r : Real
      hr : LT.lt 0 r
      x : A
      hx : Membership.mem (setOf fun a => LE.le 0 a) x
      hx_pos : LT.lt 0 (Norm.norm x)
      ⊢ Eq (HSMul.hSMul (Inv.inv (HMul.hMul ↑r ↑(Inv.inv (Norm.norm x)))) (HSMul.hSM …
    -/
    apply inv_smul_smul₀
    /-
      case inr.ha
      A : Type u_1
      inst✝² : NonUnitalCStarAlgebra A
      inst✝¹ : PartialOrder A
      inst✝ : StarOrderedRing A
      r : Real
      hr : LT.lt 0 r
      x : A
      hx : Membership.mem (setOf fun a => LE.le 0 a) x
      hx_pos : LT.lt 0 (Norm.norm x)
      ⊢ Ne (HMul.hMul ↑r ↑(Inv.inv (Norm.norm x))) 0
    -/
    norm_cast
    /-
      case inr.ha
      A : Type u_1
      inst✝² : NonUnitalCStarAlgebra A
      inst✝¹ : PartialOrder A
      inst✝ : StarOrderedRing A
      r : Real
      hr : LT.lt 0 r
      x : A
      hx : Membership.mem (setOf fun a => LE.le 0 a) x
      hx_pos : LT.lt 0 (Norm.norm x)
      ⊢ Not (Eq (HMul.hMul r (Inv.inv (Norm.norm x))) 0)
    -/
    positivity
    /-
      🎉 no goals
    -/


/-- A C⋆-algebra is spanned by nonnegative elements of norm less than `r`. -/
lemma span_nonneg_inter_ball {r : ℝ} (hr : 0 < r) :
    span ℂ ({x : A | 0 ≤ x} ∩ Metric.ball 0 r) = ⊤ := by
  /-
    A : Type u_1
    inst✝² : NonUnitalCStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    r : Real
    hr : LT.lt 0 r
    ⊢ Eq (Submodule.span Complex (Inter.inter (setOf fun x => LE.le 0 x) (Metric.b …
  -/
  rw [eq_top_iff, ← span_nonneg_inter_closedBall (half_pos hr)]
  /-
    A : Type u_1
    inst✝² : NonUnitalCStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    r : Real
    hr : LT.lt 0 r
    ⊢ LE.le (Submodule.span Complex (Inter.inter (setOf fun x => LE.le 0 x) (Metri …
  -/
  gcongr
  /-
    case h.H
    A : Type u_1
    inst✝² : NonUnitalCStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    r : Real
    hr : LT.lt 0 r
    ⊢ HasSubset.Subset (Metric.closedBall 0 (HDiv.hDiv r 2)) (Metric.ball 0 r)
  -/
  exact Metric.closedBall_subset_ball <| half_lt_self hr
  /-
    🎉 no goals
  -/


/-- A C⋆-algebra is spanned by nonnegative contractions. -/
lemma span_nonneg_inter_unitClosedBall :
    span ℂ ({x : A | 0 ≤ x} ∩ Metric.closedBall 0 1) = ⊤ :=
  span_nonneg_inter_closedBall zero_lt_one


/-- A C⋆-algebra is spanned by nonnegative strict contractions. -/
lemma span_nonneg_inter_unitBall :
    span ℂ ({x : A | 0 ≤ x} ∩ Metric.ball 0 1) = ⊤ :=
  span_nonneg_inter_ball zero_lt_one


