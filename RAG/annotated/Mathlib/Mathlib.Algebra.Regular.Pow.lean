/-- Any power of a left-regular element is left-regular. -/
theorem IsLeftRegular.pow (n : ℕ) (rla : IsLeftRegular a) : IsLeftRegular (a ^ n) := by
  /-
    R : Type u_1
    a : R
    inst✝ : Monoid R
    n : Nat
    rla : IsLeftRegular a
    ⊢ IsLeftRegular (HPow.hPow a n)
  -/
  simp only [IsLeftRegular, ← mul_left_iterate, rla.iterate n]
  /-
    🎉 no goals
  -/


/-- Any power of a right-regular element is right-regular. -/
theorem IsRightRegular.pow (n : ℕ) (rra : IsRightRegular a) : IsRightRegular (a ^ n) := by
  /-
    R : Type u_1
    a : R
    inst✝ : Monoid R
    n : Nat
    rra : IsRightRegular a
    ⊢ IsRightRegular (HPow.hPow a n)
  -/
  rw [IsRightRegular, ← mul_right_iterate]
  /-
    R : Type u_1
    a : R
    inst✝ : Monoid R
    n : Nat
    rra : IsRightRegular a
    ⊢ Function.Injective (Nat.iterate (fun x => HMul.hMul x a) n)
  -/
  exact rra.iterate n
  /-
    🎉 no goals
  -/


/-- Any power of a regular element is regular. -/
theorem IsRegular.pow (n : ℕ) (ra : IsRegular a) : IsRegular (a ^ n) :=
  ⟨IsLeftRegular.pow n ra.left, IsRightRegular.pow n ra.right⟩


/-- An element `a` is left-regular if and only if a positive power of `a` is left-regular. -/
theorem IsLeftRegular.pow_iff {n : ℕ} (n0 : 0 < n) : IsLeftRegular (a ^ n) ↔ IsLeftRegular a := by
  /-
    R : Type u_1
    a : R
    inst✝ : Monoid R
    n : Nat
    n0 : LT.lt 0 n
    ⊢ Iff (IsLeftRegular (HPow.hPow a n)) (IsLeftRegular a)
  -/
  refine ⟨?_, IsLeftRegular.pow n⟩
  /-
    R : Type u_1
    a : R
    inst✝ : Monoid R
    n : Nat
    n0 : LT.lt 0 n
    ⊢ IsLeftRegular (HPow.hPow a n) → IsLeftRegular a
  -/
  rw [← Nat.succ_pred_eq_of_pos n0, pow_succ]
  /-
    R : Type u_1
    a : R
    inst✝ : Monoid R
    n : Nat
    n0 : LT.lt 0 n
    ⊢ IsLeftRegular (HMul.hMul (HPow.hPow a n.pred) a) → IsLeftRegular a
  -/
  exact IsLeftRegular.of_mul
  /-
    🎉 no goals
  -/


/-- An element `a` is right-regular if and only if a positive power of `a` is right-regular. -/
theorem IsRightRegular.pow_iff {n : ℕ} (n0 : 0 < n) :
    IsRightRegular (a ^ n) ↔ IsRightRegular a := by
  /-
    R : Type u_1
    a : R
    inst✝ : Monoid R
    n : Nat
    n0 : LT.lt 0 n
    ⊢ Iff (IsRightRegular (HPow.hPow a n)) (IsRightRegular a)
  -/
  refine ⟨?_, IsRightRegular.pow n⟩
  /-
    R : Type u_1
    a : R
    inst✝ : Monoid R
    n : Nat
    n0 : LT.lt 0 n
    ⊢ IsRightRegular (HPow.hPow a n) → IsRightRegular a
  -/
  rw [← Nat.succ_pred_eq_of_pos n0, pow_succ']
  /-
    R : Type u_1
    a : R
    inst✝ : Monoid R
    n : Nat
    n0 : LT.lt 0 n
    ⊢ IsRightRegular (HMul.hMul a (HPow.hPow a n.pred)) → IsRightRegular a
  -/
  exact IsRightRegular.of_mul
  /-
    🎉 no goals
  -/


/-- An element `a` is regular if and only if a positive power of `a` is regular. -/
theorem IsRegular.pow_iff {n : ℕ} (n0 : 0 < n) : IsRegular (a ^ n) ↔ IsRegular a :=
  ⟨fun h => ⟨(IsLeftRegular.pow_iff n0).mp h.left, (IsRightRegular.pow_iff n0).mp h.right⟩, fun h =>
    ⟨IsLeftRegular.pow n h.left, IsRightRegular.pow n h.right⟩⟩


lemma IsLeftRegular.prod (h : ∀ i ∈ s, IsLeftRegular (f i)) :
    IsLeftRegular (∏ i ∈ s, f i) :=
  s.prod_induction _ _ (@IsLeftRegular.mul R _) isRegular_one.left h


lemma IsRightRegular.prod (h : ∀ i ∈ s, IsRightRegular (f i)) :
    IsRightRegular (∏ i ∈ s, f i) :=
  s.prod_induction _ _ (@IsRightRegular.mul R _) isRegular_one.right h


lemma IsRegular.prod (h : ∀ i ∈ s, IsRegular (f i)) :
    IsRegular (∏ i ∈ s, f i) :=
  ⟨IsLeftRegular.prod fun a ha ↦ (h a ha).left,
   IsRightRegular.prod fun a ha ↦ (h a ha).right⟩


