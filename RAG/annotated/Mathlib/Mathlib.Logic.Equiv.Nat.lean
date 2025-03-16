/-- An equivalence between `Bool × ℕ` and `ℕ`, by mapping `(true, x)` to `2 * x + 1` and
`(false, x)` to `2 * x`. -/
@[simps]
def boolProdNatEquivNat : Bool × ℕ ≃ ℕ where
  toFun := uncurry bit
  invFun := boddDiv2
                               /-
                                 α : Type u_1
                                 x✝ : Prod Bool Nat
                                 b : Bool
                                 n : Nat
                                 ⊢ Eq (Function.uncurry Nat.bit { fst := b, snd := n }).boddDiv2 { fst := b, sn …
                               -/
  left_inv := fun ⟨b, n⟩ => by simp only [bodd_bit, div2_bit, uncurry_apply_pair, boddDiv2_eq]
                               /-
                                 🎉 no goals
                               -/
                    /-
                      α : Type u_1
                      n : Nat
                      ⊢ Eq (Function.uncurry Nat.bit n.boddDiv2) n
                    -/
  right_inv n := by simp only [bit_decomp, boddDiv2_eq, uncurry_apply_pair]
                    /-
                      🎉 no goals
                    -/


/-- An equivalence between `ℕ ⊕ ℕ` and `ℕ`, by mapping `(Sum.inl x)` to `2 * x` and `(Sum.inr x)` to
`2 * x + 1`.
-/
@[simps! symm_apply]
def natSumNatEquivNat : ℕ ⊕ ℕ ≃ ℕ :=
  (boolProdEquivSum ℕ).symm.trans boolProdNatEquivNat


@[simp]
theorem natSumNatEquivNat_apply : ⇑natSumNatEquivNat = Sum.elim (2 * ·) (2 * · + 1) := by
  /-
    ⊢ Eq (⇑Equiv.natSumNatEquivNat) (Sum.elim (fun x => HMul.hMul 2 x) fun x => HA …
  -/
                  /-
                    🎉 no goals
                  -/
  ext (x | x) <;> rfl
                  /-
                    🎉 no goals
                  -/


/-- An equivalence between `ℤ` and `ℕ`, through `ℤ ≃ ℕ ⊕ ℕ` and `ℕ ⊕ ℕ ≃ ℕ`.
-/
def intEquivNat : ℤ ≃ ℕ :=
  intEquivNatSumNat.trans natSumNatEquivNat


/-- An equivalence between `α × α` and `α`, given that there is an equivalence between `α` and `ℕ`.
-/
def prodEquivOfEquivNat (e : α ≃ ℕ) : α × α ≃ α :=
  calc
    α × α ≃ ℕ × ℕ := prodCongr e e
    _ ≃ ℕ := pairEquiv
    _ ≃ α := e.symm


