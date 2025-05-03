/-- If `x` and `y` are on the same ray, then the triangle inequality becomes the equality: the norm
of `x + y` is the sum of the norms of `x` and `y`. The converse is true for a strictly convex
space. -/
theorem norm_add (h : SameRay ℝ x y) : ‖x + y‖ = ‖x‖ + ‖y‖ := by
  /-
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x y : E
    h : SameRay Real x y
    ⊢ Eq (Norm.norm (HAdd.hAdd x y)) (HAdd.hAdd (Norm.norm x) (Norm.norm y))
  -/
  rcases h.exists_eq_smul with ⟨u, a, b, ha, hb, -, rfl, rfl⟩
  rw [← add_smul, norm_smul_of_nonneg (add_nonneg ha hb), norm_smul_of_nonneg ha,
    norm_smul_of_nonneg hb, add_mul]


theorem norm_sub (h : SameRay ℝ x y) : ‖x - y‖ = |‖x‖ - ‖y‖| := by
  /-
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x y : E
    h : SameRay Real x y
    ⊢ Eq (Norm.norm (HSub.hSub x y)) (abs (HSub.hSub (Norm.norm x) (Norm.norm y)))
  -/
  rcases h.exists_eq_smul with ⟨u, a, b, ha, hb, -, rfl, rfl⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    u : E
    a b : Real
    ha : LE.le 0 a
    hb : LE.le 0 b
    h : SameRay Real (HSMul.hSMul a u) (HSMul.hSMul b u)
    ⊢ Eq (Norm.norm (HSub.hSub (HSMul.hSMul a u) (HSMul.hSMul b u))) (abs (HSub.hS …
  -/
  wlog hab : b ≤ a generalizing a b with H
    /-
      case intro.intro.intro.intro.intro.intro.intro.inr
      E : Type u_1
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace Real E
      u : E
      a b : Real
      ha : LE.le 0 a
      hb : LE.le 0 b
      h : SameRay Real (HSMul.hSMul a u) (HSMul.hSMul b u)
      H : ∀ (a b : Real), LE.le 0 a → LE.le 0 b → SameRay Real (HSMul.hSMul a u) (HS …
      hab : Not (LE.le b a)
      ⊢ Eq (Norm.norm (HSub.hSub (HSMul.hSMul a u) (HSMul.hSMul b u))) (abs (HSub.hS …
    -/
  · rw [SameRay.sameRay_comm] at h
    /-
      case intro.intro.intro.intro.intro.intro.intro.inr
      E : Type u_1
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace Real E
      u : E
      a b : Real
      ha : LE.le 0 a
      hb : LE.le 0 b
      h : SameRay Real (HSMul.hSMul b u) (HSMul.hSMul a u)
      H : ∀ (a b : Real), LE.le 0 a → LE.le 0 b → SameRay Real (HSMul.hSMul a u) (HS …
      hab : Not (LE.le b a)
      ⊢ Eq (Norm.norm (HSub.hSub (HSMul.hSMul a u) (HSMul.hSMul b u))) (abs (HSub.hS …
    -/
    rw [norm_sub_rev, abs_sub_comm]
    /-
      case intro.intro.intro.intro.intro.intro.intro.inr
      E : Type u_1
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace Real E
      u : E
      a b : Real
      ha : LE.le 0 a
      hb : LE.le 0 b
      h : SameRay Real (HSMul.hSMul b u) (HSMul.hSMul a u)
      H : ∀ (a b : Real), LE.le 0 a → LE.le 0 b → SameRay Real (HSMul.hSMul a u) (HS …
      hab : Not (LE.le b a)
      ⊢ Eq (Norm.norm (HSub.hSub (HSMul.hSMul b u) (HSMul.hSMul a u))) (abs (HSub.hS …
    -/
    exact H b a hb ha h (le_of_not_le hab)
    /-
      🎉 no goals
    -/
  /-
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    u : E
    a b : Real
    ha : LE.le 0 a
    hb : LE.le 0 b
    h : SameRay Real (HSMul.hSMul a u) (HSMul.hSMul b u)
    hab : LE.le b a
    ⊢ Eq (Norm.norm (HSub.hSub (HSMul.hSMul a u) (HSMul.hSMul b u))) (abs (HSub.hS …
  -/
  rw [← sub_nonneg] at hab
  rw [← sub_smul, norm_smul_of_nonneg hab, norm_smul_of_nonneg ha, norm_smul_of_nonneg hb, ←
    sub_mul, abs_of_nonneg (mul_nonneg hab (norm_nonneg _))]


theorem norm_smul_eq (h : SameRay ℝ x y) : ‖x‖ • y = ‖y‖ • x := by
  /-
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x y : E
    h : SameRay Real x y
    ⊢ Eq (HSMul.hSMul (Norm.norm x) y) (HSMul.hSMul (Norm.norm y) x)
  -/
  rcases h.exists_eq_smul with ⟨u, a, b, ha, hb, -, rfl, rfl⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    u : E
    a b : Real
    ha : LE.le 0 a
    hb : LE.le 0 b
    h : SameRay Real (HSMul.hSMul a u) (HSMul.hSMul b u)
    ⊢ Eq (HSMul.hSMul (Norm.norm (HSMul.hSMul a u)) (HSMul.hSMul b u)) (HSMul.hSMu …
  -/
  simp only [norm_smul_of_nonneg, *, mul_smul]
  /-
    case intro.intro.intro.intro.intro.intro.intro
    E : Type u_1
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    u : E
    a b : Real
    ha : LE.le 0 a
    hb : LE.le 0 b
    h : SameRay Real (HSMul.hSMul a u) (HSMul.hSMul b u)
    ⊢ Eq (HSMul.hSMul a (HSMul.hSMul (Norm.norm u) (HSMul.hSMul b u))) (HSMul.hSMu …
  -/
  rw [smul_comm, smul_comm b, smul_comm a b u]
  /-
    🎉 no goals
  -/


theorem norm_injOn_ray_left (hx : x ≠ 0) : { y | SameRay ℝ x y }.InjOn norm := by
  /-
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    x : F
    hx : Ne x 0
    ⊢ Set.InjOn Norm.norm (setOf fun y => SameRay Real x y)
  -/
  rintro y hy z hz h
  /-
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    x : F
    hx : Ne x 0
    y : F
    hy : Membership.mem (setOf fun y => SameRay Real x y) y
    z : F
    hz : Membership.mem (setOf fun y => SameRay Real x y) z
    h : Eq (Norm.norm y) (Norm.norm z)
    ⊢ Eq y z
  -/
  rcases hy.exists_nonneg_left hx with ⟨r, hr, rfl⟩
  /-
    case intro.intro
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    x : F
    hx : Ne x 0
    z : F
    hz : Membership.mem (setOf fun y => SameRay Real x y) z
    r : Real
    hr : LE.le 0 r
    hy : Membership.mem (setOf fun y => SameRay Real x y) (HSMul.hSMul r x)
    h : Eq (Norm.norm (HSMul.hSMul r x)) (Norm.norm z)
    ⊢ Eq (HSMul.hSMul r x) z
  -/
  rcases hz.exists_nonneg_left hx with ⟨s, hs, rfl⟩
  rw [norm_smul, norm_smul, mul_left_inj' (norm_ne_zero_iff.2 hx), norm_of_nonneg hr,
    norm_of_nonneg hs] at h
  /-
    case intro.intro.intro.intro
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    x : F
    hx : Ne x 0
    r : Real
    hr : LE.le 0 r
    hy : Membership.mem (setOf fun y => SameRay Real x y) (HSMul.hSMul r x)
    s : Real
    hs : LE.le 0 s
    hz : Membership.mem (setOf fun y => SameRay Real x y) (HSMul.hSMul s x)
    h : Eq r s
    ⊢ Eq (HSMul.hSMul r x) (HSMul.hSMul s x)
  -/
  rw [h]
  /-
    🎉 no goals
  -/


theorem norm_injOn_ray_right (hy : y ≠ 0) : { x | SameRay ℝ x y }.InjOn norm := by
  /-
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    y : F
    hy : Ne y 0
    ⊢ Set.InjOn Norm.norm (setOf fun x => SameRay Real x y)
  -/
  simpa only [SameRay.sameRay_comm] using norm_injOn_ray_left hy
  /-
    🎉 no goals
  -/


theorem sameRay_iff_norm_smul_eq : SameRay ℝ x y ↔ ‖x‖ • y = ‖y‖ • x :=
  ⟨SameRay.norm_smul_eq, fun h =>
    or_iff_not_imp_left.2 fun hx =>
      or_iff_not_imp_left.2 fun hy => ⟨‖y‖, ‖x‖, norm_pos_iff.2 hy, norm_pos_iff.2 hx, h.symm⟩⟩


/-- Two nonzero vectors `x y` in a real normed space are on the same ray if and only if the unit
vectors `‖x‖⁻¹ • x` and `‖y‖⁻¹ • y` are equal. -/
theorem sameRay_iff_inv_norm_smul_eq_of_ne (hx : x ≠ 0) (hy : y ≠ 0) :
    SameRay ℝ x y ↔ ‖x‖⁻¹ • x = ‖y‖⁻¹ • y := by
  /-
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    x y : F
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Iff (SameRay Real x y) (Eq (HSMul.hSMul (Inv.inv (Norm.norm x)) x) (HSMul.hS …
  -/
  rw [inv_smul_eq_iff₀, smul_comm, eq_comm, inv_smul_eq_iff₀, sameRay_iff_norm_smul_eq] <;>
    /-
      case ha
      F : Type u_2
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace Real F
      x y : F
      hx : Ne x 0
      hy : Ne y 0
      ⊢ Ne (Norm.norm y) 0
    -/
    /-
      🎉 no goals
    -/
    rwa [norm_ne_zero_iff]
    /-
      🎉 no goals
    -/


alias ⟨SameRay.inv_norm_smul_eq, _⟩ := sameRay_iff_inv_norm_smul_eq_of_ne


/-- Two vectors `x y` in a real normed space are on the ray if and only if one of them is zero or
the unit vectors `‖x‖⁻¹ • x` and `‖y‖⁻¹ • y` are equal. -/
theorem sameRay_iff_inv_norm_smul_eq : SameRay ℝ x y ↔ x = 0 ∨ y = 0 ∨ ‖x‖⁻¹ • x = ‖y‖⁻¹ • y := by
  /-
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    x y : F
    ⊢ Iff (SameRay Real x y) (Or (Eq x 0) (Or (Eq y 0) (Eq (HSMul.hSMul (Inv.inv ( …
  -/
  rcases eq_or_ne x 0 with (rfl | hx); · simp [SameRay.zero_left]
                                         /-
                                           🎉 no goals
                                         -/
  /-
    case inr
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    x y : F
    hx : Ne x 0
    ⊢ Iff (SameRay Real x y) (Or (Eq x 0) (Or (Eq y 0) (Eq (HSMul.hSMul (Inv.inv ( …
  -/
  rcases eq_or_ne y 0 with (rfl | hy); · simp [SameRay.zero_right]
                                         /-
                                           🎉 no goals
                                         -/
  /-
    case inr.inr
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    x y : F
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Iff (SameRay Real x y) (Or (Eq x 0) (Or (Eq y 0) (Eq (HSMul.hSMul (Inv.inv ( …
  -/
  simp only [sameRay_iff_inv_norm_smul_eq_of_ne hx hy, *, false_or]
  /-
    🎉 no goals
  -/


/-- Two vectors of the same norm are on the same ray if and only if they are equal. -/
theorem sameRay_iff_of_norm_eq (h : ‖x‖ = ‖y‖) : SameRay ℝ x y ↔ x = y := by
  /-
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace Real F
    x y : F
    h : Eq (Norm.norm x) (Norm.norm y)
    ⊢ Iff (SameRay Real x y) (Eq x y)
  -/
  obtain rfl | hy := eq_or_ne y 0
    /-
      case inl
      F : Type u_2
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace Real F
      x : F
      h : Eq (Norm.norm x) (Norm.norm 0)
      ⊢ Iff (SameRay Real x 0) (Eq x 0)
    -/
  · rw [norm_zero, norm_eq_zero] at h
    /-
      case inl
      F : Type u_2
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace Real F
      x : F
      h : Eq x 0
      ⊢ Iff (SameRay Real x 0) (Eq x 0)
    -/
    exact iff_of_true (SameRay.zero_right _) h
    /-
      🎉 no goals
    -/
    /-
      case inr
      F : Type u_2
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace Real F
      x y : F
      h : Eq (Norm.norm x) (Norm.norm y)
      hy : Ne y 0
      ⊢ Iff (SameRay Real x y) (Eq x y)
    -/
  · exact ⟨fun hxy => norm_injOn_ray_right hy hxy SameRay.rfl h, fun hxy => hxy ▸ SameRay.rfl⟩
    /-
      🎉 no goals
    -/


theorem not_sameRay_iff_of_norm_eq (h : ‖x‖ = ‖y‖) : ¬SameRay ℝ x y ↔ x ≠ y :=
  (sameRay_iff_of_norm_eq h).not


/-- If two points on the same ray have the same norm, then they are equal. -/
theorem SameRay.eq_of_norm_eq (h : SameRay ℝ x y) (hn : ‖x‖ = ‖y‖) : x = y :=
  (sameRay_iff_of_norm_eq hn).mp h


/-- The norms of two vectors on the same ray are equal if and only if they are equal. -/
theorem SameRay.norm_eq_iff (h : SameRay ℝ x y) : ‖x‖ = ‖y‖ ↔ x = y :=
  ⟨h.eq_of_norm_eq, fun h => h ▸ rfl⟩

