@[deprecated (since := "2024-08-23")] alias ⟨eq_zero_of_mul_eq_zero, _⟩ := mul_eq_zero

-- TODO: there are four variations, depending on which variables we assume to be nonneg


@[deprecated "No deprecation message was provided." (since := "2024-08-23")]
def discriminate {B : Sort u} {n : ℕ} (H1 : n = 0 → B) (H2 : ∀ m, n = succ m → B) : B := by
  induction n with
  | zero => exact H1 rfl
  | succ n _ => apply H2 _ rfl

-- Unused in Mathlib;
-- if downstream projects find this essential please copy it or remove the deprecation.

@[deprecated "No deprecation message was provided." (since := "2024-07-27")]
theorem one_eq_succ_zero : 1 = succ 0 :=
  rfl


@[deprecated "No deprecation message was provided." (since := "2024-07-27")]
def subInduction {P : ℕ → ℕ → Sort u} (H1 : ∀ m, P 0 m) (H2 : ∀ n, P (succ n) 0)
    (H3 : ∀ n m, P n m → P (succ n) (succ m)) : ∀ n m : ℕ, P n m
  | 0, _m => H1 _
  | succ _n, 0 => H2 _
  | succ n, succ m => H3 _ _ (subInduction H1 H2 H3 n m)


@[deprecated "No deprecation message was provided." (since := "2024-07-27")]
theorem cond_decide_mod_two (x : ℕ) [d : Decidable (x % 2 = 1)] :
    cond (@decide (x % 2 = 1) d) 1 0 = x % 2 := by
  /-
    x : Nat
    d : Decidable (Eq (HMod.hMod x 2) 1)
    ⊢ Eq (cond (Decidable.decide (Eq (HMod.hMod x 2) 1)) 1 0) (HMod.hMod x 2)
  -/
  simp only [cond_eq_if, decide_eq_true_eq]
  /-
    x : Nat
    d : Decidable (Eq (HMod.hMod x 2) 1)
    ⊢ Eq (ite (Eq (HMod.hMod x 2) 1) 1 0) (HMod.hMod x 2)
  -/
            /-
              🎉 no goals
            -/
  split <;> omega
            /-
              🎉 no goals
            -/


