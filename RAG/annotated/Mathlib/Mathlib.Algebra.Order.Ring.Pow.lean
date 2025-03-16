/-- **Bernoulli's inequality**. This version works for semirings but requires
additional hypotheses `0 ≤ a * a` and `0 ≤ (1 + a) * (1 + a)`. -/
lemma one_add_mul_le_pow' (Hsq : 0 ≤ a * a) (Hsq' : 0 ≤ (1 + a) * (1 + a)) (H : 0 ≤ 2 + a) :
    ∀ n : ℕ, 1 + n * a ≤ (1 + a) ^ n
            /-
              R : Type u_1
              inst✝ : OrderedSemiring R
              a : R
              Hsq : LE.le 0 (HMul.hMul a a)
              Hsq' : LE.le 0 (HMul.hMul (HAdd.hAdd 1 a) (HAdd.hAdd 1 a))
              H : LE.le 0 (HAdd.hAdd 2 a)
              ⊢ LE.le (HAdd.hAdd 1 (HMul.hMul (↑0) a)) (HPow.hPow (HAdd.hAdd 1 a) 0)
            -/
  | 0 => by simp
            /-
              🎉 no goals
            -/
            /-
              R : Type u_1
              inst✝ : OrderedSemiring R
              a : R
              Hsq : LE.le 0 (HMul.hMul a a)
              Hsq' : LE.le 0 (HMul.hMul (HAdd.hAdd 1 a) (HAdd.hAdd 1 a))
              H : LE.le 0 (HAdd.hAdd 2 a)
              ⊢ LE.le (HAdd.hAdd 1 (HMul.hMul (↑1) a)) (HPow.hPow (HAdd.hAdd 1 a) 1)
            -/
  | 1 => by simp
            /-
              🎉 no goals
            -/
  | n + 2 =>
    have : 0 ≤ n * (a * a * (2 + a)) + a * a :=
      add_nonneg (mul_nonneg n.cast_nonneg (mul_nonneg Hsq H)) Hsq
    calc
      _ ≤ 1 + ↑(n + 2) * a + (n * (a * a * (2 + a)) + a * a) := le_add_of_nonneg_right this
      _ = (1 + a) * (1 + a) * (1 + n * a) := by
          simp only [Nat.cast_add, add_mul, mul_add, one_mul, mul_one, ← one_add_one_eq_two,
            Nat.cast_one, add_assoc, add_right_inj]
          /-
            R : Type u_1
            inst✝ : OrderedSemiring R
            a : R
            Hsq : LE.le 0 (HMul.hMul a a)
            Hsq' : LE.le 0 (HMul.hMul (HAdd.hAdd 1 a) (HAdd.hAdd 1 a))
            H : LE.le 0 (HAdd.hAdd 2 a)
            n : Nat
            this : LE.le 0 (HAdd.hAdd (HMul.hMul (↑n) (HMul.hMul (HMul.hMul a a) (HAdd.hAd …
            ⊢ Eq (HAdd.hAdd 1 (HAdd.hAdd (HMul.hMul (↑n) a) (HAdd.hAdd a (HAdd.hAdd a (HAd …
          -/
          simp only [← add_assoc, add_comm _ (↑n * a)]
          /-
            R : Type u_1
            inst✝ : OrderedSemiring R
            a : R
            Hsq : LE.le 0 (HMul.hMul a a)
            Hsq' : LE.le 0 (HMul.hMul (HAdd.hAdd 1 a) (HAdd.hAdd 1 a))
            H : LE.le 0 (HAdd.hAdd 2 a)
            n : Nat
            this : LE.le 0 (HAdd.hAdd (HMul.hMul (↑n) (HMul.hMul (HMul.hMul a a) (HAdd.hAd …
            ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd (HAdd.h …
          -/
          simp only [add_assoc, (n.cast_commute (_ : R)).left_comm]
          /-
            R : Type u_1
            inst✝ : OrderedSemiring R
            a : R
            Hsq : LE.le 0 (HMul.hMul a a)
            Hsq' : LE.le 0 (HMul.hMul (HAdd.hAdd 1 a) (HAdd.hAdd 1 a))
            H : LE.le 0 (HAdd.hAdd 2 a)
            n : Nat
            this : LE.le 0 (HAdd.hAdd (HMul.hMul (↑n) (HMul.hMul (HMul.hMul a a) (HAdd.hAd …
            ⊢ Eq (HAdd.hAdd (HMul.hMul (↑n) a) (HAdd.hAdd 1 (HAdd.hAdd a (HAdd.hAdd a (HAd …
          -/
          simp only [add_comm, add_left_comm]
          /-
            🎉 no goals
          -/
      _ ≤ (1 + a) * (1 + a) * (1 + a) ^ n :=
        mul_le_mul_of_nonneg_left (one_add_mul_le_pow' Hsq Hsq' H _) Hsq'
                                  /-
                                    R : Type u_1
                                    inst✝ : OrderedSemiring R
                                    a : R
                                    Hsq : LE.le 0 (HMul.hMul a a)
                                    Hsq' : LE.le 0 (HMul.hMul (HAdd.hAdd 1 a) (HAdd.hAdd 1 a))
                                    H : LE.le 0 (HAdd.hAdd 2 a)
                                    n : Nat
                                    this : LE.le 0 (HAdd.hAdd (HMul.hMul (↑n) (HMul.hMul (HMul.hMul a a) (HAdd.hAd …
                                    ⊢ Eq (HMul.hMul (HMul.hMul (HAdd.hAdd 1 a) (HAdd.hAdd 1 a)) (HPow.hPow (HAdd.h …
                                  -/
      _ = (1 + a) ^ (n + 2) := by simp only [pow_succ', mul_assoc]
                                  /-
                                    🎉 no goals
                                  -/


/-- **Bernoulli's inequality** for `n : ℕ`, `-2 ≤ a`. -/
lemma one_add_mul_le_pow (H : -2 ≤ a) (n : ℕ) : 1 + n * a ≤ (1 + a) ^ n :=
  one_add_mul_le_pow' (mul_self_nonneg _) (mul_self_nonneg _) (neg_le_iff_add_nonneg'.1 H) _


/-- **Bernoulli's inequality** reformulated to estimate `a^n`. -/
lemma one_add_mul_sub_le_pow (H : -1 ≤ a) (n : ℕ) : 1 + n * (a - 1) ≤ a ^ n := by
  have : -2 ≤ a - 1 := by
    rwa [← one_add_one_eq_two, neg_add, ← sub_eq_add_neg, sub_le_sub_iff_right]
  /-
    R : Type u_1
    inst✝ : LinearOrderedRing R
    a : R
    H : LE.le (-1) a
    n : Nat
    this : LE.le (-2) (HSub.hSub a 1)
    ⊢ LE.le (HAdd.hAdd 1 (HMul.hMul (↑n) (HSub.hSub a 1))) (HPow.hPow a n)
  -/
  simpa only [add_sub_cancel] using one_add_mul_le_pow this n
  /-
    🎉 no goals
  -/


