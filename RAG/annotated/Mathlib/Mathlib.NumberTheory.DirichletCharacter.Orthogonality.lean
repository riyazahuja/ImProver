noncomputable instance fintype {R : Type*} [CommRing R] [IsDomain R] {n : ℕ} :
    Fintype (DirichletCharacter R n) := .ofFinite _


/-- The group of Dirichlet characters mod `n` with values in a ring `R` that has enough
roots of unity is (noncanonically) isomorphic to `(ZMod n)ˣ`. -/
lemma mulEquiv_units : Nonempty (DirichletCharacter R n ≃* (ZMod n)ˣ) :=
  MulChar.mulEquiv_units ..


/-- There are `n.totient` Dirichlet characters mod `n` with values in a ring that has enough
roots of unity. -/
lemma card_eq_totient_of_hasEnoughRootsOfUnity :
    Nat.card (DirichletCharacter R n) = n.totient := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    n : Nat
    inst✝¹ : NeZero n
    inst✝ : HasEnoughRootsOfUnity R (Monoid.exponent (Units (ZMod n)))
    ⊢ Eq (Nat.card (DirichletCharacter R n)) n.totient
  -/
  rw [← ZMod.card_units_eq_totient n, ← Nat.card_eq_fintype_card]
  /-
    R : Type u_1
    inst✝² : CommRing R
    n : Nat
    inst✝¹ : NeZero n
    inst✝ : HasEnoughRootsOfUnity R (Monoid.exponent (Units (ZMod n)))
    ⊢ Eq (Nat.card (DirichletCharacter R n)) (Nat.card (Units (ZMod n)))
  -/
  exact Nat.card_congr (mulEquiv_units R n).some.toEquiv
  /-
    🎉 no goals
  -/


/-- If `R` is a ring that has enough roots of unity and `n ≠ 0`, then for each
`a ≠ 1` in `ZMod n`, there exists a Dirichlet character `χ` mod `n` with values in `R`
such that `χ a ≠ 1`. -/
theorem exists_apply_ne_one_of_hasEnoughRootsOfUnity [Nontrivial R] ⦃a : ZMod n⦄ (ha : a ≠ 1) :
    ∃ χ : DirichletCharacter R n, χ a ≠ 1 :=
  MulChar.exists_apply_ne_one_of_hasEnoughRootsOfUnity (ZMod n) R ha


/-- If `R` is an integral domain that has enough roots of unity and `n ≠ 0`, then
for each `a ≠ 1` in `ZMod n`, the sum of `χ a` over all Dirichlet characters mod `n`
with values in `R` vanishes. -/
theorem sum_characters_eq_zero ⦃a : ZMod n⦄ (ha : a ≠ 1) :
    ∑ χ : DirichletCharacter R n, χ a = 0 := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    n : Nat
    inst✝² : NeZero n
    inst✝¹ : HasEnoughRootsOfUnity R (Monoid.exponent (Units (ZMod n)))
    inst✝ : IsDomain R
    a : ZMod n
    ha : Ne a 1
    ⊢ Eq (Finset.univ.sum fun χ => χ a) 0
  -/
  obtain ⟨χ, hχ⟩ := exists_apply_ne_one_of_hasEnoughRootsOfUnity R ha
  /-
    case intro
    R : Type u_1
    inst✝³ : CommRing R
    n : Nat
    inst✝² : NeZero n
    inst✝¹ : HasEnoughRootsOfUnity R (Monoid.exponent (Units (ZMod n)))
    inst✝ : IsDomain R
    a : ZMod n
    ha : Ne a 1
    χ : DirichletCharacter R n
    hχ : Ne (χ a) 1
    ⊢ Eq (Finset.univ.sum fun χ => χ a) 0
  -/
  refine eq_zero_of_mul_eq_self_left hχ ?_
  /-
    case intro
    R : Type u_1
    inst✝³ : CommRing R
    n : Nat
    inst✝² : NeZero n
    inst✝¹ : HasEnoughRootsOfUnity R (Monoid.exponent (Units (ZMod n)))
    inst✝ : IsDomain R
    a : ZMod n
    ha : Ne a 1
    χ : DirichletCharacter R n
    hχ : Ne (χ a) 1
    ⊢ Eq (HMul.hMul (χ a) (Finset.univ.sum fun χ => χ a)) (Finset.univ.sum fun χ = …
  -/
  simp only [Finset.mul_sum, ← MulChar.mul_apply]
  /-
    case intro
    R : Type u_1
    inst✝³ : CommRing R
    n : Nat
    inst✝² : NeZero n
    inst✝¹ : HasEnoughRootsOfUnity R (Monoid.exponent (Units (ZMod n)))
    inst✝ : IsDomain R
    a : ZMod n
    ha : Ne a 1
    χ : DirichletCharacter R n
    hχ : Ne (χ a) 1
    ⊢ Eq (Finset.univ.sum fun x => (HMul.hMul χ x) a) (Finset.univ.sum fun χ => χ a)
  -/
  exact Fintype.sum_bijective _ (Group.mulLeft_bijective χ) _ _ fun χ' ↦ rfl
  /-
    🎉 no goals
  -/


/-- If `R` is an integral domain that has enough roots of unity and `n ≠ 0`, then
for `a` in `ZMod n`, the sum of `χ a` over all Dirichlet characters mod `n`
with values in `R` vanishes if `a ≠ 1` and has the value `n.totient` if `a = 1`. -/
theorem sum_characters_eq (a : ZMod n) :
    ∑ χ : DirichletCharacter R n, χ a = if a = 1 then (n.totient : R) else 0 := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    n : Nat
    inst✝² : NeZero n
    inst✝¹ : HasEnoughRootsOfUnity R (Monoid.exponent (Units (ZMod n)))
    inst✝ : IsDomain R
    a : ZMod n
    ⊢ Eq (Finset.univ.sum fun χ => χ a) (ite (Eq a 1) (↑n.totient) 0)
  -/
  split_ifs with ha
  · simpa only [ha, map_one, Finset.sum_const, Finset.card_univ, nsmul_eq_mul, mul_one,
      ← Nat.card_eq_fintype_card]
      using congrArg Nat.cast <| card_eq_totient_of_hasEnoughRootsOfUnity R n
    /-
      case neg
      R : Type u_1
      inst✝³ : CommRing R
      n : Nat
      inst✝² : NeZero n
      inst✝¹ : HasEnoughRootsOfUnity R (Monoid.exponent (Units (ZMod n)))
      inst✝ : IsDomain R
      a : ZMod n
      ha : Not (Eq a 1)
      ⊢ Eq (Finset.univ.sum fun χ => χ a) 0
    -/
  · exact sum_characters_eq_zero R ha
    /-
      🎉 no goals
    -/


/-- If `R` is an integral domain that has enough roots of unity and `n ≠ 0`, then for `a` and `b`
in `ZMod n` with `a` a unit, the sum of `χ a⁻¹ * χ b` over all Dirichlet characters
mod `n` with values in `R` vanishes if `a ≠ b` and has the value `n.totient` if `a = b`. -/
theorem sum_char_inv_mul_char_eq {a : ZMod n} (ha : IsUnit a) (b : ZMod n) :
    ∑ χ : DirichletCharacter R n, χ a⁻¹ * χ b = if a = b then (n.totient : R) else 0 := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    n : Nat
    inst✝² : NeZero n
    inst✝¹ : HasEnoughRootsOfUnity R (Monoid.exponent (Units (ZMod n)))
    inst✝ : IsDomain R
    a : ZMod n
    ha : IsUnit a
    b : ZMod n
    ⊢ Eq (Finset.univ.sum fun χ => HMul.hMul (χ (Inv.inv a)) (χ b)) (ite (Eq a b)  …
  -/
  simp only [← map_mul, sum_characters_eq, ZMod.inv_mul_eq_one_of_isUnit ha]
  /-
    🎉 no goals
  -/


