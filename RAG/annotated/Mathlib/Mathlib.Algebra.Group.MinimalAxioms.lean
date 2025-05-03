/-- Define a `Group` structure on a Type by proving `∀ a, 1 * a = a` and
`∀ a, a⁻¹ * a = 1`.
Note that this uses the default definitions for `npow`, `zpow` and `div`.
See note [reducible non-instances]. -/
@[to_additive
"Define an `AddGroup` structure on a Type by proving `∀ a, 0 + a = a` and
`∀ a, -a + a = 0`.
Note that this uses the default definitions for `nsmul`, `zsmul` and `sub`.
See note [reducible non-instances]."]
abbrev Group.ofLeftAxioms {G : Type u} [Mul G] [Inv G] [One G]
    (assoc : ∀ a b c : G, (a * b) * c = a * (b * c))
    (one_mul : ∀ a : G, 1 * a = a)
    (inv_mul_cancel : ∀ a : G, a⁻¹ * a = 1) : Group G :=
  { mul_assoc := assoc,
    one_mul := one_mul,
    inv_mul_cancel := inv_mul_cancel,
    mul_one := fun a => by
      have mul_inv_cancel : ∀ a : G, a * a⁻¹ = 1 := fun a =>
        calc a * a⁻¹ = 1 * (a * a⁻¹) := (one_mul _).symm
          _ = ((a * a⁻¹)⁻¹ * (a * a⁻¹)) * (a * a⁻¹) := by
            rw [inv_mul_cancel]
          _ = (a * a⁻¹)⁻¹ * (a * ((a⁻¹ * a) * a⁻¹)) := by
            simp only [assoc]
          _ = 1 := by
            rw [inv_mul_cancel, one_mul, inv_mul_cancel]
      /-
        G : Type u
        inst✝² : Mul G
        inst✝¹ : Inv G
        inst✝ : One G
        assoc : ∀ (a b c : G), Eq (HMul.hMul (HMul.hMul a b) c) (HMul.hMul a (HMul.hMu …
        one_mul : ∀ (a : G), Eq (HMul.hMul 1 a) a
        inv_mul_cancel : ∀ (a : G), Eq (HMul.hMul (Inv.inv a) a) 1
        a : G
        mul_inv_cancel : ∀ (a : G), Eq (HMul.hMul a (Inv.inv a)) 1
        ⊢ Eq (HMul.hMul a 1) a
      -/
      rw [← inv_mul_cancel a, ← assoc, mul_inv_cancel a, one_mul] }
      /-
        🎉 no goals
      -/


/-- Define a `Group` structure on a Type by proving `∀ a, a * 1 = a` and
`∀ a, a * a⁻¹ = 1`.
Note that this uses the default definitions for `npow`, `zpow` and `div`.
See note [reducible non-instances]. -/
@[to_additive
"Define an `AddGroup` structure on a Type by proving `∀ a, a + 0 = a` and
`∀ a, a + -a = 0`.
Note that this uses the default definitions for `nsmul`, `zsmul` and `sub`.
See note [reducible non-instances]."]
abbrev Group.ofRightAxioms {G : Type u} [Mul G] [Inv G] [One G]
    (assoc : ∀ a b c : G, (a * b) * c = a * (b * c))
    (mul_one : ∀ a : G, a * 1 = a)
    (mul_inv_cancel : ∀ a : G, a * a⁻¹ = 1) : Group G :=
  have inv_mul_cancel : ∀ a : G, a⁻¹ * a = 1 := fun a =>
    calc a⁻¹ * a = (a⁻¹ * a) * 1 := (mul_one _).symm
      _ = (a⁻¹ * a) * ((a⁻¹ * a) * (a⁻¹ * a)⁻¹) := by
        /-
          G : Type u
          inst✝² : Mul G
          inst✝¹ : Inv G
          inst✝ : One G
          assoc : ∀ (a b c : G), Eq (HMul.hMul (HMul.hMul a b) c) (HMul.hMul a (HMul.hMu …
          mul_one : ∀ (a : G), Eq (HMul.hMul a 1) a
          mul_inv_cancel : ∀ (a : G), Eq (HMul.hMul a (Inv.inv a)) 1
          a : G
          ⊢ Eq (HMul.hMul (HMul.hMul (Inv.inv a) a) 1) (HMul.hMul (HMul.hMul (Inv.inv a) …
        -/
        rw [mul_inv_cancel]
        /-
          🎉 no goals
        -/
      _ = ((a⁻¹ * (a * a⁻¹)) * a) * (a⁻¹ * a)⁻¹ := by
        /-
          G : Type u
          inst✝² : Mul G
          inst✝¹ : Inv G
          inst✝ : One G
          assoc : ∀ (a b c : G), Eq (HMul.hMul (HMul.hMul a b) c) (HMul.hMul a (HMul.hMu …
          mul_one : ∀ (a : G), Eq (HMul.hMul a 1) a
          mul_inv_cancel : ∀ (a : G), Eq (HMul.hMul a (Inv.inv a)) 1
          a : G
          ⊢ Eq (HMul.hMul (HMul.hMul (Inv.inv a) a) (HMul.hMul (HMul.hMul (Inv.inv a) a) …
        -/
        simp only [assoc]
        /-
          🎉 no goals
        -/
      _ = 1 := by
        /-
          G : Type u
          inst✝² : Mul G
          inst✝¹ : Inv G
          inst✝ : One G
          assoc : ∀ (a b c : G), Eq (HMul.hMul (HMul.hMul a b) c) (HMul.hMul a (HMul.hMu …
          mul_one : ∀ (a : G), Eq (HMul.hMul a 1) a
          mul_inv_cancel : ∀ (a : G), Eq (HMul.hMul a (Inv.inv a)) 1
          a : G
          ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (Inv.inv a) (HMul.hMul a (Inv.inv a))) a …
        -/
        rw [mul_inv_cancel, mul_one, mul_inv_cancel]
        /-
          🎉 no goals
        -/
  { mul_assoc := assoc,
    mul_one := mul_one,
    inv_mul_cancel := inv_mul_cancel,
    one_mul := fun a => by
      /-
        G : Type u
        inst✝² : Mul G
        inst✝¹ : Inv G
        inst✝ : One G
        assoc : ∀ (a b c : G), Eq (HMul.hMul (HMul.hMul a b) c) (HMul.hMul a (HMul.hMu …
        mul_one : ∀ (a : G), Eq (HMul.hMul a 1) a
        mul_inv_cancel : ∀ (a : G), Eq (HMul.hMul a (Inv.inv a)) 1
        inv_mul_cancel : ∀ (a : G), Eq (HMul.hMul (Inv.inv a) a) 1
        a : G
        ⊢ Eq (HMul.hMul 1 a) a
      -/
      rw [← mul_inv_cancel a, assoc, inv_mul_cancel, mul_one] }
      /-
        🎉 no goals
      -/

