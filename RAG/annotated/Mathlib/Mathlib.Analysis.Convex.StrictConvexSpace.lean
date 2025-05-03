/-- A *strictly convex space* is a normed space where the closed balls are strictly convex. We only
require balls of positive radius with center at the origin to be strictly convex in the definition,
then prove that any closed ball is strictly convex in `strictConvex_closedBall` below.

See also `StrictConvexSpace.of_strictConvex_unitClosedBall`. -/
class StrictConvexSpace (𝕜 E : Type*) [NormedLinearOrderedField 𝕜] [NormedAddCommGroup E]
  [NormedSpace 𝕜 E] : Prop where
  strictConvex_closedBall : ∀ r : ℝ, 0 < r → StrictConvex 𝕜 (closedBall (0 : E) r)


/-- A closed ball in a strictly convex space is strictly convex. -/
theorem strictConvex_closedBall [StrictConvexSpace 𝕜 E] (x : E) (r : ℝ) :
    StrictConvex 𝕜 (closedBall x r) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : NormedLinearOrderedField 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : StrictConvexSpace 𝕜 E
    x : E
    r : Real
    ⊢ StrictConvex 𝕜 (Metric.closedBall x r)
  -/
  rcases le_or_lt r 0 with hr | hr
    /-
      case inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : NormedLinearOrderedField 𝕜
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : StrictConvexSpace 𝕜 E
      x : E
      r : Real
      hr : LE.le r 0
      ⊢ StrictConvex 𝕜 (Metric.closedBall x r)
    -/
  · exact (subsingleton_closedBall x hr).strictConvex
    /-
      🎉 no goals
    -/
  /-
    case inr
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : NormedLinearOrderedField 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : StrictConvexSpace 𝕜 E
    x : E
    r : Real
    hr : LT.lt 0 r
    ⊢ StrictConvex 𝕜 (Metric.closedBall x r)
  -/
  rw [← vadd_closedBall_zero]
  /-
    case inr
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : NormedLinearOrderedField 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : StrictConvexSpace 𝕜 E
    x : E
    r : Real
    hr : LT.lt 0 r
    ⊢ StrictConvex 𝕜 (HVAdd.hVAdd x (Metric.closedBall 0 r))
  -/
  exact (StrictConvexSpace.strictConvex_closedBall r hr).vadd _
  /-
    🎉 no goals
  -/


/-- A real normed vector space is strictly convex provided that the unit ball is strictly convex. -/
theorem StrictConvexSpace.of_strictConvex_unitClosedBall [LinearMap.CompatibleSMul E E 𝕜 ℝ]
    (h : StrictConvex 𝕜 (closedBall (0 : E) 1)) : StrictConvexSpace 𝕜 E :=
                  /-
                    𝕜 : Type u_1
                    E : Type u_2
                    inst✝⁴ : NormedLinearOrderedField 𝕜
                    inst✝³ : NormedAddCommGroup E
                    inst✝² : NormedSpace 𝕜 E
                    inst✝¹ : NormedSpace Real E
                    inst✝ : LinearMap.CompatibleSMul E E 𝕜 Real
                    h : StrictConvex 𝕜 (Metric.closedBall 0 1)
                    r : Real
                    hr : LT.lt 0 r
                    ⊢ StrictConvex 𝕜 (Metric.closedBall 0 r)
                  -/
  ⟨fun r hr => by simpa only [smul_unitClosedBall_of_nonneg hr.le] using h.smul r⟩
                  /-
                    🎉 no goals
                  -/


@[deprecated (since := "2024-12-01")]
alias StrictConvexSpace.of_strictConvex_closed_unit_ball :=
  StrictConvexSpace.of_strictConvex_unitClosedBall


/-- Strict convexity is equivalent to `‖a • x + b • y‖ < 1` for all `x` and `y` of norm at most `1`
and all strictly positive `a` and `b` such that `a + b = 1`. This lemma shows that it suffices to
check this for points of norm one and some `a`, `b` such that `a + b = 1`. -/
theorem StrictConvexSpace.of_norm_combo_lt_one
    (h : ∀ x y : E, ‖x‖ = 1 → ‖y‖ = 1 → x ≠ y → ∃ a b : ℝ, a + b = 1 ∧ ‖a • x + b • y‖ < 1) :
    StrictConvexSpace ℝ E := by
  refine
    StrictConvexSpace.of_strictConvex_unitClosedBall ℝ
      ((convex_closedBall _ _).strictConvex' fun x hx y hy hne => ?_)
  rw [interior_closedBall (0 : E) one_ne_zero, closedBall_diff_ball,
    mem_sphere_zero_iff_norm] at hx hy
  /-
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    h : ∀ (x y : E), Eq (Norm.norm x) 1 → Eq (Norm.norm y) 1 → Ne x y → Exists fun …
    x : E
    hx : Eq (Norm.norm x) 1
    y : E
    hy : Eq (Norm.norm y) 1
    hne : Ne x y
    ⊢ Exists fun c => Membership.mem (interior (Metric.closedBall 0 1)) ((AffineMa …
  -/
  rcases h x y hx hy hne with ⟨a, b, hab, hlt⟩
  /-
    case intro.intro.intro
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    h : ∀ (x y : E), Eq (Norm.norm x) 1 → Eq (Norm.norm y) 1 → Ne x y → Exists fun …
    x : E
    hx : Eq (Norm.norm x) 1
    y : E
    hy : Eq (Norm.norm y) 1
    hne : Ne x y
    a b : Real
    hab : Eq (HAdd.hAdd a b) 1
    hlt : LT.lt (Norm.norm (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))) 1
    ⊢ Exists fun c => Membership.mem (interior (Metric.closedBall 0 1)) ((AffineMa …
  -/
  use b
  rwa [AffineMap.lineMap_apply_module, interior_closedBall (0 : E) one_ne_zero, mem_ball_zero_iff,
    sub_eq_iff_eq_add.2 hab.symm]


theorem StrictConvexSpace.of_norm_combo_ne_one
    (h :
      ∀ x y : E,
        ‖x‖ = 1 → ‖y‖ = 1 → x ≠ y → ∃ a b : ℝ, 0 ≤ a ∧ 0 ≤ b ∧ a + b = 1 ∧ ‖a • x + b • y‖ ≠ 1) :
    StrictConvexSpace ℝ E := by
  refine StrictConvexSpace.of_strictConvex_unitClosedBall ℝ
    ((convex_closedBall _ _).strictConvex ?_)
  simp only [interior_closedBall _ one_ne_zero, closedBall_diff_ball, Set.Pairwise,
    frontier_closedBall _ one_ne_zero, mem_sphere_zero_iff_norm]
  /-
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    h : ∀ (x y : E), Eq (Norm.norm x) 1 → Eq (Norm.norm y) 1 → Ne x y → Exists fun …
    ⊢ ∀ ⦃x : E⦄, Eq (Norm.norm x) 1 → ∀ ⦃y : E⦄, Eq (Norm.norm y) 1 → Ne x y → (SD …
  -/
  intro x hx y hy hne
  /-
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    h : ∀ (x y : E), Eq (Norm.norm x) 1 → Eq (Norm.norm y) 1 → Ne x y → Exists fun …
    x : E
    hx : Eq (Norm.norm x) 1
    y : E
    hy : Eq (Norm.norm y) 1
    hne : Ne x y
    ⊢ (SDiff.sdiff (segment Real x y) (Metric.sphere 0 1)).Nonempty
  -/
  rcases h x y hx hy hne with ⟨a, b, ha, hb, hab, hne'⟩
  /-
    case intro.intro.intro.intro.intro
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    h : ∀ (x y : E), Eq (Norm.norm x) 1 → Eq (Norm.norm y) 1 → Ne x y → Exists fun …
    x : E
    hx : Eq (Norm.norm x) 1
    y : E
    hy : Eq (Norm.norm y) 1
    hne : Ne x y
    a b : Real
    ha : LE.le 0 a
    hb : LE.le 0 b
    hab : Eq (HAdd.hAdd a b) 1
    hne' : Ne (Norm.norm (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))) 1
    ⊢ (SDiff.sdiff (segment Real x y) (Metric.sphere 0 1)).Nonempty
  -/
  exact ⟨_, ⟨a, b, ha, hb, hab, rfl⟩, mt mem_sphere_zero_iff_norm.1 hne'⟩
  /-
    🎉 no goals
  -/


theorem StrictConvexSpace.of_norm_add_ne_two
    (h : ∀ ⦃x y : E⦄, ‖x‖ = 1 → ‖y‖ = 1 → x ≠ y → ‖x + y‖ ≠ 2) : StrictConvexSpace ℝ E := by
  refine
    StrictConvexSpace.of_norm_combo_ne_one fun x y hx hy hne =>
      ⟨1 / 2, 1 / 2, one_half_pos.le, one_half_pos.le, add_halves _, ?_⟩
  rw [← smul_add, norm_smul, Real.norm_of_nonneg one_half_pos.le, one_div, ← div_eq_inv_mul, Ne,
    div_eq_one_iff_eq (two_ne_zero' ℝ)]
  /-
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    h : ∀ ⦃x y : E⦄, Eq (Norm.norm x) 1 → Eq (Norm.norm y) 1 → Ne x y → Ne (Norm.n …
    x y : E
    hx : Eq (Norm.norm x) 1
    hy : Eq (Norm.norm y) 1
    hne : Ne x y
    ⊢ Not (Eq (Norm.norm (HAdd.hAdd x y)) 2)
  -/
  exact h hx hy hne
  /-
    🎉 no goals
  -/


theorem StrictConvexSpace.of_pairwise_sphere_norm_ne_two
    (h : (sphere (0 : E) 1).Pairwise fun x y => ‖x + y‖ ≠ 2) : StrictConvexSpace ℝ E :=
  StrictConvexSpace.of_norm_add_ne_two fun _ _ hx hy =>
    h (mem_sphere_zero_iff_norm.2 hx) (mem_sphere_zero_iff_norm.2 hy)


/-- If `‖x + y‖ = ‖x‖ + ‖y‖` implies that `x y : E` are in the same ray, then `E` is a strictly
convex space. See also a more -/
theorem StrictConvexSpace.of_norm_add
    (h : ∀ x y : E, ‖x‖ = 1 → ‖y‖ = 1 → ‖x + y‖ = 2 → SameRay ℝ x y) : StrictConvexSpace ℝ E := by
  /-
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    h : ∀ (x y : E), Eq (Norm.norm x) 1 → Eq (Norm.norm y) 1 → Eq (Norm.norm (HAdd …
    ⊢ StrictConvexSpace Real E
  -/
  refine StrictConvexSpace.of_pairwise_sphere_norm_ne_two fun x hx y hy => mt fun h₂ => ?_
  /-
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    h : ∀ (x y : E), Eq (Norm.norm x) 1 → Eq (Norm.norm y) 1 → Eq (Norm.norm (HAdd …
    x : E
    hx : Membership.mem (Metric.sphere 0 1) x
    y : E
    hy : Membership.mem (Metric.sphere 0 1) y
    h₂ : Eq (Norm.norm (HAdd.hAdd x y)) 2
    ⊢ Eq x y
  -/
  rw [mem_sphere_zero_iff_norm] at hx hy
  /-
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    h : ∀ (x y : E), Eq (Norm.norm x) 1 → Eq (Norm.norm y) 1 → Eq (Norm.norm (HAdd …
    x : E
    hx : Eq (Norm.norm x) 1
    y : E
    hy : Eq (Norm.norm y) 1
    h₂ : Eq (Norm.norm (HAdd.hAdd x y)) 2
    ⊢ Eq x y
  -/
  exact (sameRay_iff_of_norm_eq (hx.trans hy.symm)).1 (h x y hx hy h₂)
  /-
    🎉 no goals
  -/


/-- If `x ≠ y` belong to the same closed ball, then a convex combination of `x` and `y` with
positive coefficients belongs to the corresponding open ball. -/
theorem combo_mem_ball_of_ne (hx : x ∈ closedBall z r) (hy : y ∈ closedBall z r) (hne : x ≠ y)
    (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) : a • x + b • y ∈ ball z r := by
  /-
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : StrictConvexSpace Real E
    x y z : E
    a b r : Real
    hx : Membership.mem (Metric.closedBall z r) x
    hy : Membership.mem (Metric.closedBall z r) y
    hne : Ne x y
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ Membership.mem (Metric.ball z r) (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b …
  -/
  rcases eq_or_ne r 0 with (rfl | hr)
    /-
      case inl
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : StrictConvexSpace Real E
      x y z : E
      a b : Real
      hne : Ne x y
      ha : LT.lt 0 a
      hb : LT.lt 0 b
      hab : Eq (HAdd.hAdd a b) 1
      hx : Membership.mem (Metric.closedBall z 0) x
      hy : Membership.mem (Metric.closedBall z 0) y
      ⊢ Membership.mem (Metric.ball z 0) (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b …
    -/
  · rw [closedBall_zero, mem_singleton_iff] at hx hy
    /-
      case inl
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : StrictConvexSpace Real E
      x y z : E
      a b : Real
      hne : Ne x y
      ha : LT.lt 0 a
      hb : LT.lt 0 b
      hab : Eq (HAdd.hAdd a b) 1
      hx : Eq x z
      hy : Eq y z
      ⊢ Membership.mem (Metric.ball z 0) (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b …
    -/
    exact (hne (hx.trans hy.symm)).elim
    /-
      🎉 no goals
    -/
    /-
      case inr
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : StrictConvexSpace Real E
      x y z : E
      a b r : Real
      hx : Membership.mem (Metric.closedBall z r) x
      hy : Membership.mem (Metric.closedBall z r) y
      hne : Ne x y
      ha : LT.lt 0 a
      hb : LT.lt 0 b
      hab : Eq (HAdd.hAdd a b) 1
      hr : Ne r 0
      ⊢ Membership.mem (Metric.ball z r) (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b …
    -/
  · simp only [← interior_closedBall _ hr] at hx hy ⊢
    /-
      case inr
      E : Type u_2
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : StrictConvexSpace Real E
      x y z : E
      a b r : Real
      hx : Membership.mem (Metric.closedBall z r) x
      hy : Membership.mem (Metric.closedBall z r) y
      hne : Ne x y
      ha : LT.lt 0 a
      hb : LT.lt 0 b
      hab : Eq (HAdd.hAdd a b) 1
      hr : Ne r 0
      ⊢ Membership.mem (interior (Metric.closedBall z r)) (HAdd.hAdd (HSMul.hSMul a  …
    -/
    exact strictConvex_closedBall ℝ z r hx hy hne ha hb hab
    /-
      🎉 no goals
    -/


/-- If `x ≠ y` belong to the same closed ball, then the open segment with endpoints `x` and `y` is
included in the corresponding open ball. -/
theorem openSegment_subset_ball_of_ne (hx : x ∈ closedBall z r) (hy : y ∈ closedBall z r)
    (hne : x ≠ y) : openSegment ℝ x y ⊆ ball z r :=
  (openSegment_subset_iff _).2 fun _ _ => combo_mem_ball_of_ne hx hy hne


/-- If `x` and `y` are two distinct vectors of norm at most `r`, then a convex combination of `x`
and `y` with positive coefficients has norm strictly less than `r`. -/
theorem norm_combo_lt_of_ne (hx : ‖x‖ ≤ r) (hy : ‖y‖ ≤ r) (hne : x ≠ y) (ha : 0 < a) (hb : 0 < b)
    (hab : a + b = 1) : ‖a • x + b • y‖ < r := by
  /-
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : StrictConvexSpace Real E
    x y : E
    a b r : Real
    hx : LE.le (Norm.norm x) r
    hy : LE.le (Norm.norm y) r
    hne : Ne x y
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ LT.lt (Norm.norm (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b y))) r
  -/
  simp only [← mem_ball_zero_iff, ← mem_closedBall_zero_iff] at hx hy ⊢
  /-
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : StrictConvexSpace Real E
    x y : E
    a b r : Real
    hne : Ne x y
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    hx : Membership.mem (Metric.closedBall 0 r) x
    hy : Membership.mem (Metric.closedBall 0 r) y
    ⊢ Membership.mem (Metric.ball 0 r) (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul b …
  -/
  exact combo_mem_ball_of_ne hx hy hne ha hb hab
  /-
    🎉 no goals
  -/


/-- In a strictly convex space, if `x` and `y` are not in the same ray, then `‖x + y‖ < ‖x‖ + ‖y‖`.
-/
theorem norm_add_lt_of_not_sameRay (h : ¬SameRay ℝ x y) : ‖x + y‖ < ‖x‖ + ‖y‖ := by
  /-
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : StrictConvexSpace Real E
    x y : E
    h : Not (SameRay Real x y)
    ⊢ LT.lt (Norm.norm (HAdd.hAdd x y)) (HAdd.hAdd (Norm.norm x) (Norm.norm y))
  -/
  simp only [sameRay_iff_inv_norm_smul_eq, not_or, ← Ne.eq_def] at h
  /-
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : StrictConvexSpace Real E
    x y : E
    h : And (Ne x 0) (And (Ne y 0) (Ne (HSMul.hSMul (Inv.inv (Norm.norm x)) x) (HS …
    ⊢ LT.lt (Norm.norm (HAdd.hAdd x y)) (HAdd.hAdd (Norm.norm x) (Norm.norm y))
  -/
  rcases h with ⟨hx, hy, hne⟩
  /-
    case intro.intro
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : StrictConvexSpace Real E
    x y : E
    hx : Ne x 0
    hy : Ne y 0
    hne : Ne (HSMul.hSMul (Inv.inv (Norm.norm x)) x) (HSMul.hSMul (Inv.inv (Norm.n …
    ⊢ LT.lt (Norm.norm (HAdd.hAdd x y)) (HAdd.hAdd (Norm.norm x) (Norm.norm y))
  -/
  rw [← norm_pos_iff] at hx hy
  /-
    case intro.intro
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : StrictConvexSpace Real E
    x y : E
    hx : LT.lt 0 (Norm.norm x)
    hy : LT.lt 0 (Norm.norm y)
    hne : Ne (HSMul.hSMul (Inv.inv (Norm.norm x)) x) (HSMul.hSMul (Inv.inv (Norm.n …
    ⊢ LT.lt (Norm.norm (HAdd.hAdd x y)) (HAdd.hAdd (Norm.norm x) (Norm.norm y))
  -/
  have hxy : 0 < ‖x‖ + ‖y‖ := add_pos hx hy
  have :=
    combo_mem_ball_of_ne (inv_norm_smul_mem_unitClosedBall x)
      (inv_norm_smul_mem_unitClosedBall y) hne (div_pos hx hxy) (div_pos hy hxy)
      (by rw [← add_div, div_self hxy.ne'])
  rwa [mem_ball_zero_iff, div_eq_inv_mul, div_eq_inv_mul, mul_smul, mul_smul, smul_inv_smul₀ hx.ne',
    smul_inv_smul₀ hy.ne', ← smul_add, norm_smul, Real.norm_of_nonneg (inv_pos.2 hxy).le, ←
    div_eq_inv_mul, div_lt_one hxy] at this


theorem lt_norm_sub_of_not_sameRay (h : ¬SameRay ℝ x y) : ‖x‖ - ‖y‖ < ‖x - y‖ := by
  /-
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : StrictConvexSpace Real E
    x y : E
    h : Not (SameRay Real x y)
    ⊢ LT.lt (HSub.hSub (Norm.norm x) (Norm.norm y)) (Norm.norm (HSub.hSub x y))
  -/
  nth_rw 1 [← sub_add_cancel x y] at h ⊢
  /-
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : StrictConvexSpace Real E
    x y : E
    h : Not (SameRay Real (HAdd.hAdd (HSub.hSub x y) y) y)
    ⊢ LT.lt (HSub.hSub (Norm.norm (HAdd.hAdd (HSub.hSub x y) y)) (Norm.norm y)) (N …
  -/
  exact sub_lt_iff_lt_add.2 (norm_add_lt_of_not_sameRay fun H' => h <| H'.add_left SameRay.rfl)
  /-
    🎉 no goals
  -/


theorem abs_lt_norm_sub_of_not_sameRay (h : ¬SameRay ℝ x y) : |‖x‖ - ‖y‖| < ‖x - y‖ := by
  /-
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : StrictConvexSpace Real E
    x y : E
    h : Not (SameRay Real x y)
    ⊢ LT.lt (abs (HSub.hSub (Norm.norm x) (Norm.norm y))) (Norm.norm (HSub.hSub x  …
  -/
  refine abs_sub_lt_iff.2 ⟨lt_norm_sub_of_not_sameRay h, ?_⟩
  /-
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : StrictConvexSpace Real E
    x y : E
    h : Not (SameRay Real x y)
    ⊢ LT.lt (HSub.hSub (Norm.norm y) (Norm.norm x)) (Norm.norm (HSub.hSub x y))
  -/
  rw [norm_sub_rev]
  /-
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : StrictConvexSpace Real E
    x y : E
    h : Not (SameRay Real x y)
    ⊢ LT.lt (HSub.hSub (Norm.norm y) (Norm.norm x)) (Norm.norm (HSub.hSub y x))
  -/
  exact lt_norm_sub_of_not_sameRay (mt SameRay.symm h)
  /-
    🎉 no goals
  -/


/-- In a strictly convex space, two vectors `x`, `y` are in the same ray if and only if the triangle
inequality for `x` and `y` becomes an equality. -/
theorem sameRay_iff_norm_add : SameRay ℝ x y ↔ ‖x + y‖ = ‖x‖ + ‖y‖ :=
  ⟨SameRay.norm_add, fun h => Classical.not_not.1 fun h' => (norm_add_lt_of_not_sameRay h').ne h⟩


/-- If `x` and `y` are two vectors in a strictly convex space have the same norm and the norm of
their sum is equal to the sum of their norms, then they are equal. -/
theorem eq_of_norm_eq_of_norm_add_eq (h₁ : ‖x‖ = ‖y‖) (h₂ : ‖x + y‖ = ‖x‖ + ‖y‖) : x = y :=
  (sameRay_iff_norm_add.mpr h₂).eq_of_norm_eq h₁


/-- In a strictly convex space, two vectors `x`, `y` are not in the same ray if and only if the
triangle inequality for `x` and `y` is strict. -/
theorem not_sameRay_iff_norm_add_lt : ¬SameRay ℝ x y ↔ ‖x + y‖ < ‖x‖ + ‖y‖ :=
  sameRay_iff_norm_add.not.trans (norm_add_le _ _).lt_iff_ne.symm


theorem sameRay_iff_norm_sub : SameRay ℝ x y ↔ ‖x - y‖ = |‖x‖ - ‖y‖| :=
  ⟨SameRay.norm_sub, fun h =>
    Classical.not_not.1 fun h' => (abs_lt_norm_sub_of_not_sameRay h').ne' h⟩


theorem not_sameRay_iff_abs_lt_norm_sub : ¬SameRay ℝ x y ↔ |‖x‖ - ‖y‖| < ‖x - y‖ :=
  sameRay_iff_norm_sub.not.trans <| ne_comm.trans (abs_norm_sub_norm_le _ _).lt_iff_ne.symm


theorem norm_midpoint_lt_iff (h : ‖x‖ = ‖y‖) : ‖(1 / 2 : ℝ) • (x + y)‖ < ‖x‖ ↔ x ≠ y := by
  rw [norm_smul, Real.norm_of_nonneg (one_div_nonneg.2 zero_le_two), ← inv_eq_one_div, ←
    div_eq_inv_mul, div_lt_iff₀ (zero_lt_two' ℝ), mul_two, ← not_sameRay_iff_of_norm_eq h,
    not_sameRay_iff_norm_add_lt, h]

