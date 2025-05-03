@[simp] lemma isConj_iff₀ : IsConj a b ↔ ∃ c : α, c ≠ 0 ∧ c * a * c⁻¹ = b := by
  /-
    α : Type u_1
    inst✝ : GroupWithZero α
    a b : α
    ⊢ Iff (IsConj a b) (Exists fun c => And (Ne c 0) (Eq (HMul.hMul (HMul.hMul c a …
  -/
  rw [IsConj, Units.exists_iff_ne_zero (p := (SemiconjBy · a b))]
  /-
    α : Type u_1
    inst✝ : GroupWithZero α
    a b : α
    ⊢ Iff (Exists fun x => And (Ne x 0) (SemiconjBy x a b)) (Exists fun c => And ( …
  -/
  congr! 2 with c
  /-
    case a.h.e'_2.h.a
    α : Type u_1
    inst✝ : GroupWithZero α
    a b c : α
    ⊢ Iff (And (Ne c 0) (SemiconjBy c a b)) (And (Ne c 0) (Eq (HMul.hMul (HMul.hMu …
  -/
  exact and_congr_right (mul_inv_eq_iff_eq_mul₀ · |>.symm)
  /-
    🎉 no goals
  -/


lemma conj_pow₀ {s : ℕ} {a d : α} (ha : a ≠ 0) : (a⁻¹ * d * a) ^ s = a⁻¹ * d ^ s * a :=
  let u : αˣ := ⟨a, a⁻¹, mul_inv_cancel₀ ha, inv_mul_cancel₀ ha⟩
  Units.conj_pow' u d s


