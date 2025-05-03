/-- A *uniformly convex space* is a real normed space where the triangle inequality is strict with a
uniform bound. Namely, over the `x` and `y` of norm `1`, `‖x + y‖` is uniformly bounded above
by a constant `< 2` when `‖x - y‖` is uniformly bounded below by a positive constant. -/
class UniformConvexSpace (E : Type*) [SeminormedAddCommGroup E] : Prop where
  uniform_convex : ∀ ⦃ε : ℝ⦄,
    0 < ε → ∃ δ, 0 < δ ∧ ∀ ⦃x : E⦄, ‖x‖ = 1 → ∀ ⦃y⦄, ‖y‖ = 1 → ε ≤ ‖x - y‖ → ‖x + y‖ ≤ 2 - δ


theorem exists_forall_sphere_dist_add_le_two_sub (hε : 0 < ε) :
    ∃ δ, 0 < δ ∧ ∀ ⦃x : E⦄, ‖x‖ = 1 → ∀ ⦃y⦄, ‖y‖ = 1 → ε ≤ ‖x - y‖ → ‖x + y‖ ≤ 2 - δ :=
  UniformConvexSpace.uniform_convex hε


theorem exists_forall_closed_ball_dist_add_le_two_sub (hε : 0 < ε) :
    ∃ δ, 0 < δ ∧ ∀ ⦃x : E⦄, ‖x‖ ≤ 1 → ∀ ⦃y⦄, ‖y‖ ≤ 1 → ε ≤ ‖x - y‖ → ‖x + y‖ ≤ 2 - δ := by
  /-
    E : Type u_1
    inst✝² : SeminormedAddCommGroup E
    inst✝¹ : UniformConvexSpace E
    ε : Real
    inst✝ : NormedSpace Real E
    hε : LT.lt 0 ε
    ⊢ Exists fun δ => And (LT.lt 0 δ) (∀ ⦃x : E⦄, LE.le (Norm.norm x) 1 → ∀ ⦃y : E …
  -/
  have hε' : 0 < ε / 3 := div_pos hε zero_lt_three
  /-
    E : Type u_1
    inst✝² : SeminormedAddCommGroup E
    inst✝¹ : UniformConvexSpace E
    ε : Real
    inst✝ : NormedSpace Real E
    hε : LT.lt 0 ε
    hε' : LT.lt 0 (HDiv.hDiv ε 3)
    ⊢ Exists fun δ => And (LT.lt 0 δ) (∀ ⦃x : E⦄, LE.le (Norm.norm x) 1 → ∀ ⦃y : E …
  -/
  obtain ⟨δ, hδ, h⟩ := exists_forall_sphere_dist_add_le_two_sub E hε'
  /-
    case intro.intro
    E : Type u_1
    inst✝² : SeminormedAddCommGroup E
    inst✝¹ : UniformConvexSpace E
    ε : Real
    inst✝ : NormedSpace Real E
    hε : LT.lt 0 ε
    hε' : LT.lt 0 (HDiv.hDiv ε 3)
    δ : Real
    hδ : LT.lt 0 δ
    h : ∀ ⦃x : E⦄, Eq (Norm.norm x) 1 → ∀ ⦃y : E⦄, Eq (Norm.norm y) 1 → LE.le (HDi …
    ⊢ Exists fun δ => And (LT.lt 0 δ) (∀ ⦃x : E⦄, LE.le (Norm.norm x) 1 → ∀ ⦃y : E …
  -/
  set δ' := min (1 / 2) (min (ε / 3) <| δ / 3)
  /-
    case intro.intro
    E : Type u_1
    inst✝² : SeminormedAddCommGroup E
    inst✝¹ : UniformConvexSpace E
    ε : Real
    inst✝ : NormedSpace Real E
    hε : LT.lt 0 ε
    hε' : LT.lt 0 (HDiv.hDiv ε 3)
    δ : Real
    hδ : LT.lt 0 δ
    h : ∀ ⦃x : E⦄, Eq (Norm.norm x) 1 → ∀ ⦃y : E⦄, Eq (Norm.norm y) 1 → LE.le (HDi …
    δ' : Real := Min.min (1 / 2) (Min.min (HDiv.hDiv ε 3) (HDiv.hDiv δ 3))
    ⊢ Exists fun δ => And (LT.lt 0 δ) (∀ ⦃x : E⦄, LE.le (Norm.norm x) 1 → ∀ ⦃y : E …
  -/
  refine ⟨δ', lt_min one_half_pos <| lt_min hε' (div_pos hδ zero_lt_three), fun x hx y hy hxy => ?_⟩
  /-
    case intro.intro
    E : Type u_1
    inst✝² : SeminormedAddCommGroup E
    inst✝¹ : UniformConvexSpace E
    ε : Real
    inst✝ : NormedSpace Real E
    hε : LT.lt 0 ε
    hε' : LT.lt 0 (HDiv.hDiv ε 3)
    δ : Real
    hδ : LT.lt 0 δ
    h : ∀ ⦃x : E⦄, Eq (Norm.norm x) 1 → ∀ ⦃y : E⦄, Eq (Norm.norm y) 1 → LE.le (HDi …
    δ' : Real := Min.min (1 / 2) (Min.min (HDiv.hDiv ε 3) (HDiv.hDiv δ 3))
    x : E
    hx : LE.le (Norm.norm x) 1
    y : E
    hy : LE.le (Norm.norm y) 1
    hxy : LE.le ε (Norm.norm (HSub.hSub x y))
    ⊢ LE.le (Norm.norm (HAdd.hAdd x y)) (HSub.hSub 2 δ')
  -/
  obtain hx' | hx' := le_or_lt ‖x‖ (1 - δ')
    /-
      case intro.intro.inl
      E : Type u_1
      inst✝² : SeminormedAddCommGroup E
      inst✝¹ : UniformConvexSpace E
      ε : Real
      inst✝ : NormedSpace Real E
      hε : LT.lt 0 ε
      hε' : LT.lt 0 (HDiv.hDiv ε 3)
      δ : Real
      hδ : LT.lt 0 δ
      h : ∀ ⦃x : E⦄, Eq (Norm.norm x) 1 → ∀ ⦃y : E⦄, Eq (Norm.norm y) 1 → LE.le (HDi …
      δ' : Real := Min.min (1 / 2) (Min.min (HDiv.hDiv ε 3) (HDiv.hDiv δ 3))
      x : E
      hx : LE.le (Norm.norm x) 1
      y : E
      hy : LE.le (Norm.norm y) 1
      hxy : LE.le ε (Norm.norm (HSub.hSub x y))
      hx' : LE.le (Norm.norm x) (HSub.hSub 1 δ')
      ⊢ LE.le (Norm.norm (HAdd.hAdd x y)) (HSub.hSub 2 δ')
    -/
  · rw [← one_add_one_eq_two]
    /-
      case intro.intro.inl
      E : Type u_1
      inst✝² : SeminormedAddCommGroup E
      inst✝¹ : UniformConvexSpace E
      ε : Real
      inst✝ : NormedSpace Real E
      hε : LT.lt 0 ε
      hε' : LT.lt 0 (HDiv.hDiv ε 3)
      δ : Real
      hδ : LT.lt 0 δ
      h : ∀ ⦃x : E⦄, Eq (Norm.norm x) 1 → ∀ ⦃y : E⦄, Eq (Norm.norm y) 1 → LE.le (HDi …
      δ' : Real := Min.min (1 / 2) (Min.min (HDiv.hDiv ε 3) (HDiv.hDiv δ 3))
      x : E
      hx : LE.le (Norm.norm x) 1
      y : E
      hy : LE.le (Norm.norm y) 1
      hxy : LE.le ε (Norm.norm (HSub.hSub x y))
      hx' : LE.le (Norm.norm x) (HSub.hSub 1 δ')
      ⊢ LE.le (Norm.norm (HAdd.hAdd x y)) (HSub.hSub (HAdd.hAdd 1 1) δ')
    -/
    exact (norm_add_le_of_le hx' hy).trans (sub_add_eq_add_sub _ _ _).le
    /-
      🎉 no goals
    -/
  /-
    case intro.intro.inr
    E : Type u_1
    inst✝² : SeminormedAddCommGroup E
    inst✝¹ : UniformConvexSpace E
    ε : Real
    inst✝ : NormedSpace Real E
    hε : LT.lt 0 ε
    hε' : LT.lt 0 (HDiv.hDiv ε 3)
    δ : Real
    hδ : LT.lt 0 δ
    h : ∀ ⦃x : E⦄, Eq (Norm.norm x) 1 → ∀ ⦃y : E⦄, Eq (Norm.norm y) 1 → LE.le (HDi …
    δ' : Real := Min.min (1 / 2) (Min.min (HDiv.hDiv ε 3) (HDiv.hDiv δ 3))
    x : E
    hx : LE.le (Norm.norm x) 1
    y : E
    hy : LE.le (Norm.norm y) 1
    hxy : LE.le ε (Norm.norm (HSub.hSub x y))
    hx' : LT.lt (HSub.hSub 1 δ') (Norm.norm x)
    ⊢ LE.le (Norm.norm (HAdd.hAdd x y)) (HSub.hSub 2 δ')
  -/
  obtain hy' | hy' := le_or_lt ‖y‖ (1 - δ')
    /-
      case intro.intro.inr.inl
      E : Type u_1
      inst✝² : SeminormedAddCommGroup E
      inst✝¹ : UniformConvexSpace E
      ε : Real
      inst✝ : NormedSpace Real E
      hε : LT.lt 0 ε
      hε' : LT.lt 0 (HDiv.hDiv ε 3)
      δ : Real
      hδ : LT.lt 0 δ
      h : ∀ ⦃x : E⦄, Eq (Norm.norm x) 1 → ∀ ⦃y : E⦄, Eq (Norm.norm y) 1 → LE.le (HDi …
      δ' : Real := Min.min (1 / 2) (Min.min (HDiv.hDiv ε 3) (HDiv.hDiv δ 3))
      x : E
      hx : LE.le (Norm.norm x) 1
      y : E
      hy : LE.le (Norm.norm y) 1
      hxy : LE.le ε (Norm.norm (HSub.hSub x y))
      hx' : LT.lt (HSub.hSub 1 δ') (Norm.norm x)
      hy' : LE.le (Norm.norm y) (HSub.hSub 1 δ')
      ⊢ LE.le (Norm.norm (HAdd.hAdd x y)) (HSub.hSub 2 δ')
    -/
  · rw [← one_add_one_eq_two]
    /-
      case intro.intro.inr.inl
      E : Type u_1
      inst✝² : SeminormedAddCommGroup E
      inst✝¹ : UniformConvexSpace E
      ε : Real
      inst✝ : NormedSpace Real E
      hε : LT.lt 0 ε
      hε' : LT.lt 0 (HDiv.hDiv ε 3)
      δ : Real
      hδ : LT.lt 0 δ
      h : ∀ ⦃x : E⦄, Eq (Norm.norm x) 1 → ∀ ⦃y : E⦄, Eq (Norm.norm y) 1 → LE.le (HDi …
      δ' : Real := Min.min (1 / 2) (Min.min (HDiv.hDiv ε 3) (HDiv.hDiv δ 3))
      x : E
      hx : LE.le (Norm.norm x) 1
      y : E
      hy : LE.le (Norm.norm y) 1
      hxy : LE.le ε (Norm.norm (HSub.hSub x y))
      hx' : LT.lt (HSub.hSub 1 δ') (Norm.norm x)
      hy' : LE.le (Norm.norm y) (HSub.hSub 1 δ')
      ⊢ LE.le (Norm.norm (HAdd.hAdd x y)) (HSub.hSub (HAdd.hAdd 1 1) δ')
    -/
    exact (norm_add_le_of_le hx hy').trans (add_sub_assoc _ _ _).ge
    /-
      🎉 no goals
    -/
  /-
    case intro.intro.inr.inr
    E : Type u_1
    inst✝² : SeminormedAddCommGroup E
    inst✝¹ : UniformConvexSpace E
    ε : Real
    inst✝ : NormedSpace Real E
    hε : LT.lt 0 ε
    hε' : LT.lt 0 (HDiv.hDiv ε 3)
    δ : Real
    hδ : LT.lt 0 δ
    h : ∀ ⦃x : E⦄, Eq (Norm.norm x) 1 → ∀ ⦃y : E⦄, Eq (Norm.norm y) 1 → LE.le (HDi …
    δ' : Real := Min.min (1 / 2) (Min.min (HDiv.hDiv ε 3) (HDiv.hDiv δ 3))
    x : E
    hx : LE.le (Norm.norm x) 1
    y : E
    hy : LE.le (Norm.norm y) 1
    hxy : LE.le ε (Norm.norm (HSub.hSub x y))
    hx' : LT.lt (HSub.hSub 1 δ') (Norm.norm x)
    hy' : LT.lt (HSub.hSub 1 δ') (Norm.norm y)
    ⊢ LE.le (Norm.norm (HAdd.hAdd x y)) (HSub.hSub 2 δ')
  -/
  have hδ' : 0 < 1 - δ' := sub_pos_of_lt (min_lt_of_left_lt one_half_lt_one)
  have h₁ : ∀ z : E, 1 - δ' < ‖z‖ → ‖‖z‖⁻¹ • z‖ = 1 := by
    rintro z hz
    rw [norm_smul_of_nonneg (inv_nonneg.2 <| norm_nonneg _), inv_mul_cancel₀ (hδ'.trans hz).ne']
  have h₂ : ∀ z : E, ‖z‖ ≤ 1 → 1 - δ' ≤ ‖z‖ → ‖‖z‖⁻¹ • z - z‖ ≤ δ' := by
    rintro z hz hδz
    nth_rw 3 [← one_smul ℝ z]
    rwa [← sub_smul,
      norm_smul_of_nonneg (sub_nonneg_of_le <| (one_le_inv₀ (hδ'.trans_le hδz)).2 hz),
      sub_mul, inv_mul_cancel₀ (hδ'.trans_le hδz).ne', one_mul, sub_le_comm]
  /-
    case intro.intro.inr.inr
    E : Type u_1
    inst✝² : SeminormedAddCommGroup E
    inst✝¹ : UniformConvexSpace E
    ε : Real
    inst✝ : NormedSpace Real E
    hε : LT.lt 0 ε
    hε' : LT.lt 0 (HDiv.hDiv ε 3)
    δ : Real
    hδ : LT.lt 0 δ
    h : ∀ ⦃x : E⦄, Eq (Norm.norm x) 1 → ∀ ⦃y : E⦄, Eq (Norm.norm y) 1 → LE.le (HDi …
    δ' : Real := Min.min (1 / 2) (Min.min (HDiv.hDiv ε 3) (HDiv.hDiv δ 3))
    x : E
    hx : LE.le (Norm.norm x) 1
    y : E
    hy : LE.le (Norm.norm y) 1
    hxy : LE.le ε (Norm.norm (HSub.hSub x y))
    hx' : LT.lt (HSub.hSub 1 δ') (Norm.norm x)
    hy' : LT.lt (HSub.hSub 1 δ') (Norm.norm y)
    hδ' : LT.lt 0 (HSub.hSub 1 δ')
    h₁ : ∀ (z : E), LT.lt (HSub.hSub 1 δ') (Norm.norm z) → Eq (Norm.norm (HSMul.hS …
    h₂ : ∀ (z : E), LE.le (Norm.norm z) 1 → LE.le (HSub.hSub 1 δ') (Norm.norm z) → …
    ⊢ LE.le (Norm.norm (HAdd.hAdd x y)) (HSub.hSub 2 δ')
  -/
  set x' := ‖x‖⁻¹ • x
  /-
    case intro.intro.inr.inr
    E : Type u_1
    inst✝² : SeminormedAddCommGroup E
    inst✝¹ : UniformConvexSpace E
    ε : Real
    inst✝ : NormedSpace Real E
    hε : LT.lt 0 ε
    hε' : LT.lt 0 (HDiv.hDiv ε 3)
    δ : Real
    hδ : LT.lt 0 δ
    h : ∀ ⦃x : E⦄, Eq (Norm.norm x) 1 → ∀ ⦃y : E⦄, Eq (Norm.norm y) 1 → LE.le (HDi …
    δ' : Real := Min.min (1 / 2) (Min.min (HDiv.hDiv ε 3) (HDiv.hDiv δ 3))
    x : E
    hx : LE.le (Norm.norm x) 1
    y : E
    hy : LE.le (Norm.norm y) 1
    hxy : LE.le ε (Norm.norm (HSub.hSub x y))
    hx' : LT.lt (HSub.hSub 1 δ') (Norm.norm x)
    hy' : LT.lt (HSub.hSub 1 δ') (Norm.norm y)
    hδ' : LT.lt 0 (HSub.hSub 1 δ')
    h₁ : ∀ (z : E), LT.lt (HSub.hSub 1 δ') (Norm.norm z) → Eq (Norm.norm (HSMul.hS …
    h₂ : ∀ (z : E), LE.le (Norm.norm z) 1 → LE.le (HSub.hSub 1 δ') (Norm.norm z) → …
    x' : E := HSMul.hSMul (Inv.inv (Norm.norm x)) x
    ⊢ LE.le (Norm.norm (HAdd.hAdd x y)) (HSub.hSub 2 δ')
  -/
  set y' := ‖y‖⁻¹ • y
  have hxy' : ε / 3 ≤ ‖x' - y'‖ :=
    calc
      ε / 3 = ε - (ε / 3 + ε / 3) := by ring
      _ ≤ ‖x - y‖ - (‖x' - x‖ + ‖y' - y‖) := by
        gcongr
        · exact (h₂ _ hx hx'.le).trans <| min_le_of_right_le <| min_le_left _ _
        · exact (h₂ _ hy hy'.le).trans <| min_le_of_right_le <| min_le_left _ _
      _ ≤ _ := by
        have : ∀ x' y', x - y = x' - y' + (x - x') + (y' - y) := fun _ _ => by abel
        rw [sub_le_iff_le_add, norm_sub_rev _ x, ← add_assoc, this]
        exact norm_add₃_le
  calc
    ‖x + y‖ ≤ ‖x' + y'‖ + ‖x' - x‖ + ‖y' - y‖ := by
      have : ∀ x' y', x + y = x' + y' + (x - x') + (y - y') := fun _ _ => by abel
      rw [norm_sub_rev, norm_sub_rev y', this]
      exact norm_add₃_le
    _ ≤ 2 - δ + δ' + δ' :=
      (add_le_add_three (h (h₁ _ hx') (h₁ _ hy') hxy') (h₂ _ hx hx'.le) (h₂ _ hy hy'.le))
    _ ≤ 2 - δ' := by
      suffices δ' ≤ δ / 3 by linarith
      exact min_le_of_right_le <| min_le_right _ _


theorem exists_forall_closed_ball_dist_add_le_two_mul_sub (hε : 0 < ε) (r : ℝ) :
    ∃ δ, 0 < δ ∧ ∀ ⦃x : E⦄, ‖x‖ ≤ r → ∀ ⦃y⦄, ‖y‖ ≤ r → ε ≤ ‖x - y‖ → ‖x + y‖ ≤ 2 * r - δ := by
  /-
    E : Type u_1
    inst✝² : SeminormedAddCommGroup E
    inst✝¹ : UniformConvexSpace E
    ε : Real
    inst✝ : NormedSpace Real E
    hε : LT.lt 0 ε
    r : Real
    ⊢ Exists fun δ => And (LT.lt 0 δ) (∀ ⦃x : E⦄, LE.le (Norm.norm x) r → ∀ ⦃y : E …
  -/
  obtain hr | hr := le_or_lt r 0
  · exact ⟨1, one_pos, fun x hx y hy h => (hε.not_le <|
      h.trans <| (norm_sub_le _ _).trans <| add_nonpos (hx.trans hr) (hy.trans hr)).elim⟩
  /-
    case inr
    E : Type u_1
    inst✝² : SeminormedAddCommGroup E
    inst✝¹ : UniformConvexSpace E
    ε : Real
    inst✝ : NormedSpace Real E
    hε : LT.lt 0 ε
    r : Real
    hr : LT.lt 0 r
    ⊢ Exists fun δ => And (LT.lt 0 δ) (∀ ⦃x : E⦄, LE.le (Norm.norm x) r → ∀ ⦃y : E …
  -/
  obtain ⟨δ, hδ, h⟩ := exists_forall_closed_ball_dist_add_le_two_sub E (div_pos hε hr)
  /-
    case inr.intro.intro
    E : Type u_1
    inst✝² : SeminormedAddCommGroup E
    inst✝¹ : UniformConvexSpace E
    ε : Real
    inst✝ : NormedSpace Real E
    hε : LT.lt 0 ε
    r : Real
    hr : LT.lt 0 r
    δ : Real
    hδ : LT.lt 0 δ
    h : ∀ ⦃x : E⦄, LE.le (Norm.norm x) 1 → ∀ ⦃y : E⦄, LE.le (Norm.norm y) 1 → LE.l …
    ⊢ Exists fun δ => And (LT.lt 0 δ) (∀ ⦃x : E⦄, LE.le (Norm.norm x) r → ∀ ⦃y : E …
  -/
  refine ⟨δ * r, mul_pos hδ hr, fun x hx y hy hxy => ?_⟩
  /-
    case inr.intro.intro
    E : Type u_1
    inst✝² : SeminormedAddCommGroup E
    inst✝¹ : UniformConvexSpace E
    ε : Real
    inst✝ : NormedSpace Real E
    hε : LT.lt 0 ε
    r : Real
    hr : LT.lt 0 r
    δ : Real
    hδ : LT.lt 0 δ
    h : ∀ ⦃x : E⦄, LE.le (Norm.norm x) 1 → ∀ ⦃y : E⦄, LE.le (Norm.norm y) 1 → LE.l …
    x : E
    hx : LE.le (Norm.norm x) r
    y : E
    hy : LE.le (Norm.norm y) r
    hxy : LE.le ε (Norm.norm (HSub.hSub x y))
    ⊢ LE.le (Norm.norm (HAdd.hAdd x y)) (HSub.hSub (HMul.hMul 2 r) (HMul.hMul δ r))
  -/
  rw [← div_le_one hr, div_eq_inv_mul, ← norm_smul_of_nonneg (inv_nonneg.2 hr.le)] at hx hy
  /-
    case inr.intro.intro
    E : Type u_1
    inst✝² : SeminormedAddCommGroup E
    inst✝¹ : UniformConvexSpace E
    ε : Real
    inst✝ : NormedSpace Real E
    hε : LT.lt 0 ε
    r : Real
    hr : LT.lt 0 r
    δ : Real
    hδ : LT.lt 0 δ
    h : ∀ ⦃x : E⦄, LE.le (Norm.norm x) 1 → ∀ ⦃y : E⦄, LE.le (Norm.norm y) 1 → LE.l …
    x : E
    hx : LE.le (Norm.norm (HSMul.hSMul (Inv.inv r) x)) 1
    y : E
    hy : LE.le (Norm.norm (HSMul.hSMul (Inv.inv r) y)) 1
    hxy : LE.le ε (Norm.norm (HSub.hSub x y))
    ⊢ LE.le (Norm.norm (HAdd.hAdd x y)) (HSub.hSub (HMul.hMul 2 r) (HMul.hMul δ r))
  -/
  have := h hx hy
  simp_rw [← smul_add, ← smul_sub, norm_smul_of_nonneg (inv_nonneg.2 hr.le), ← div_eq_inv_mul,
    div_le_div_iff_of_pos_right hr, div_le_iff₀ hr, sub_mul] at this
  /-
    case inr.intro.intro
    E : Type u_1
    inst✝² : SeminormedAddCommGroup E
    inst✝¹ : UniformConvexSpace E
    ε : Real
    inst✝ : NormedSpace Real E
    hε : LT.lt 0 ε
    r : Real
    hr : LT.lt 0 r
    δ : Real
    hδ : LT.lt 0 δ
    h : ∀ ⦃x : E⦄, LE.le (Norm.norm x) 1 → ∀ ⦃y : E⦄, LE.le (Norm.norm y) 1 → LE.l …
    x : E
    hx : LE.le (Norm.norm (HSMul.hSMul (Inv.inv r) x)) 1
    y : E
    hy : LE.le (Norm.norm (HSMul.hSMul (Inv.inv r) y)) 1
    hxy : LE.le ε (Norm.norm (HSub.hSub x y))
    this : LE.le ε (Norm.norm (HSub.hSub x y)) → LE.le (Norm.norm (HAdd.hAdd x y)) …
    ⊢ LE.le (Norm.norm (HAdd.hAdd x y)) (HSub.hSub (HMul.hMul 2 r) (HMul.hMul δ r))
  -/
  exact this hxy
  /-
    🎉 no goals
  -/


instance (priority := 100) UniformConvexSpace.toStrictConvexSpace : StrictConvexSpace ℝ E :=
  StrictConvexSpace.of_norm_add_ne_two fun _ _ hx hy hxy =>
    let ⟨_, hδ, h⟩ := exists_forall_closed_ball_dist_add_le_two_sub E (norm_sub_pos_iff.2 hxy)
    ((h hx.le hy.le le_rfl).trans_lt <| sub_lt_self _ hδ).ne

