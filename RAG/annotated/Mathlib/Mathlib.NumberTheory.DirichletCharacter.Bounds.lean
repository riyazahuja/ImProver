/-- The value at a unit of a Dirichlet character with target a normed field has norm `1`. -/
@[simp] lemma unit_norm_eq_one (a : (ZMod n)ˣ) : ‖χ a‖ = 1 := by
  /-
    F : Type u_1
    inst✝ : NormedField F
    n : Nat
    χ : DirichletCharacter F n
    a : Units (ZMod n)
    ⊢ Eq (Norm.norm (χ ↑a)) 1
  -/
  refine (pow_eq_one_iff_of_nonneg (norm_nonneg _) (Nat.card_pos (α := (ZMod n)ˣ)).ne').mp ?_
  rw [← norm_pow, ← map_pow, ← Units.val_pow_eq_pow_val, pow_card_eq_one', Units.val_one, map_one,
    norm_one]


/-- The values of a Dirichlet character with target a normed field have norm bounded by `1`. -/
lemma norm_le_one (a : ZMod n) : ‖χ a‖ ≤ 1 := by
  /-
    F : Type u_1
    inst✝ : NormedField F
    n : Nat
    χ : DirichletCharacter F n
    a : ZMod n
    ⊢ LE.le (Norm.norm (χ a)) 1
  -/
  by_cases h : IsUnit a
    /-
      case pos
      F : Type u_1
      inst✝ : NormedField F
      n : Nat
      χ : DirichletCharacter F n
      a : ZMod n
      h : IsUnit a
      ⊢ LE.le (Norm.norm (χ a)) 1
    -/
  · exact (χ.unit_norm_eq_one h.unit).le
    /-
      🎉 no goals
    -/
    /-
      case neg
      F : Type u_1
      inst✝ : NormedField F
      n : Nat
      χ : DirichletCharacter F n
      a : ZMod n
      h : Not (IsUnit a)
      ⊢ LE.le (Norm.norm (χ a)) 1
    -/
  · rw [χ.map_nonunit h, norm_zero]
    /-
      case neg
      F : Type u_1
      inst✝ : NormedField F
      n : Nat
      χ : DirichletCharacter F n
      a : ZMod n
      h : Not (IsUnit a)
      ⊢ LE.le 0 1
    -/
    exact zero_le_one
    /-
      🎉 no goals
    -/



