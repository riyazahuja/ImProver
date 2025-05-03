/-- A `MulAction` over `ℚ` restricts to a `MulAction` over `ℚ≥0`. -/
instance [MulAction ℚ α] : MulAction ℚ≥0 α :=
  MulAction.compHom α coeHom.toMonoidHom


/-- A `DistribMulAction` over `ℚ` restricts to a `DistribMulAction` over `ℚ≥0`. -/
instance [AddCommMonoid α] [DistribMulAction ℚ α] : DistribMulAction ℚ≥0 α :=
  DistribMulAction.compHom α coeHom.toMonoidHom


@[simp, norm_cast]
lemma coe_indicator (s : Set α) (f : α → ℚ≥0) (a : α) :
    ((s.indicator f a : ℚ≥0) : ℚ) = s.indicator (fun x ↦ ↑(f x)) a :=
  (coeHom : ℚ≥0 →+ ℚ).map_indicator _ _ _


lemma toNNRat_inv (q : ℚ) : toNNRat q⁻¹ = (toNNRat q)⁻¹ := by
  /-
    q : Rat
    ⊢ Eq (Inv.inv q).toNNRat (Inv.inv q.toNNRat)
  -/
  obtain hq | hq := le_total q 0
    /-
      case inl
      q : Rat
      hq : LE.le q 0
      ⊢ Eq (Inv.inv q).toNNRat (Inv.inv q.toNNRat)
    -/
  · rw [toNNRat_eq_zero.mpr hq, inv_zero, toNNRat_eq_zero.mpr (inv_nonpos.mpr hq)]
    /-
      🎉 no goals
    -/
    /-
      case inr
      q : Rat
      hq : LE.le 0 q
      ⊢ Eq (Inv.inv q).toNNRat (Inv.inv q.toNNRat)
    -/
  · nth_rw 1 [← Rat.coe_toNNRat q hq]
    /-
      case inr
      q : Rat
      hq : LE.le 0 q
      ⊢ Eq (Inv.inv ↑q.toNNRat).toNNRat (Inv.inv q.toNNRat)
    -/
    rw [← coe_inv, toNNRat_coe]
    /-
      🎉 no goals
    -/


lemma toNNRat_div (hp : 0 ≤ p) : toNNRat (p / q) = toNNRat p / toNNRat q := by
  /-
    p q : Rat
    hp : LE.le 0 p
    ⊢ Eq (HDiv.hDiv p q).toNNRat (HDiv.hDiv p.toNNRat q.toNNRat)
  -/
  rw [div_eq_mul_inv, div_eq_mul_inv, ← toNNRat_inv, ← toNNRat_mul hp]
  /-
    🎉 no goals
  -/


lemma toNNRat_div' (hq : 0 ≤ q) : toNNRat (p / q) = toNNRat p / toNNRat q := by
  /-
    p q : Rat
    hq : LE.le 0 q
    ⊢ Eq (HDiv.hDiv p q).toNNRat (HDiv.hDiv p.toNNRat q.toNNRat)
  -/
  rw [div_eq_inv_mul, div_eq_inv_mul, toNNRat_mul (inv_nonneg.2 hq), toNNRat_inv]
  /-
    🎉 no goals
  -/


/-- A recursor for nonnegative rationals in terms of numerators and denominators. -/
protected def rec {α : ℚ≥0 → Sort*} (h : ∀ m n : ℕ, α (m / n)) (q : ℚ≥0) : α q := by
  /-
    q✝ : NNRat
    α : NNRat → Sort u_1
    h : (m n : Nat) → α (HDiv.hDiv ↑m ↑n)
    q : NNRat
    ⊢ α q
  -/
  rw [← num_div_den q]; apply h
                        /-
                          🎉 no goals
                        -/


