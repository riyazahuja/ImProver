@[simp]
lemma tsum_set_one_eq : ∑' (_ : s), (1 : ℝ≥0∞) = s.encard := by
  /-
    α : Type u_1
    s : Set α
    ⊢ Eq (tsum fun x => 1) ↑s.encard
  -/
  obtain (hfin | hinf) := Set.finite_or_infinite s
    /-
      case inl
      α : Type u_1
      s : Set α
      hfin : s.Finite
      ⊢ Eq (tsum fun x => 1) ↑s.encard
    -/
  · lift s to Finset α using hfin
    /-
      case inl.intro
      α : Type u_1
      s : Finset α
      ⊢ Eq (tsum fun x => 1) ↑(↑s).encard
    -/
    simp [tsum_fintype]
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      s : Set α
      hinf : s.Infinite
      ⊢ Eq (tsum fun x => 1) ↑s.encard
    -/
  · have : Infinite s := infinite_coe_iff.mpr hinf
    /-
      case inr
      α : Type u_1
      s : Set α
      hinf : s.Infinite
      this : Infinite ↑s
      ⊢ Eq (tsum fun x => 1) ↑s.encard
    -/
    rw [tsum_const_eq_top_of_ne_zero one_ne_zero, encard_eq_top hinf, ENat.toENNReal_top]
    /-
      🎉 no goals
    -/


@[simp]
lemma tsum_set_const_eq (c : ℝ≥0∞) : ∑' (_:s), (c : ℝ≥0∞) = s.encard * c := by
  /-
    α : Type u_1
    s : Set α
    c : ENNReal
    ⊢ Eq (tsum fun x => c) (HMul.hMul (↑s.encard) c)
  -/
  nth_rw 1 [← one_mul c]
  /-
    α : Type u_1
    s : Set α
    c : ENNReal
    ⊢ Eq (tsum fun x => HMul.hMul 1 c) (HMul.hMul (↑s.encard) c)
  -/
  rw [ENNReal.tsum_mul_right,tsum_set_one_eq]
  /-
    🎉 no goals
  -/


