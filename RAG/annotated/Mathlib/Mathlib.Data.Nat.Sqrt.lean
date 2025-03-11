private lemma iter_fp_bound (n k : ℕ) :
    let iter_next (n guess : ℕ) := (guess + n / guess) / 2;
    sqrt.iter n k ≤ iter_next n (sqrt.iter n k) := by
  /-
    n k : Nat
    ⊢ let iter_next := fun n guess => HDiv.hDiv (HAdd.hAdd guess (HDiv.hDiv n gues …
      LE.le (Nat.sqrt.iter n k) (iter_next n (Nat.sqrt.iter n k))
  -/
  intro iter_next
  /-
    n k : Nat
    iter_next : Nat → Nat → Nat := fun n guess => HDiv.hDiv (HAdd.hAdd guess (HDiv …
    ⊢ LE.le (Nat.sqrt.iter n k) (iter_next n (Nat.sqrt.iter n k))
  -/
  unfold sqrt.iter
  if h : (k + n / k) / 2 < k then
    simpa [if_pos h] using iter_fp_bound _ _
  else
    simpa [if_neg h] using Nat.le_of_not_lt h


private lemma AM_GM : {a b : ℕ} → (4 * a * b ≤ (a + b) * (a + b))
               /-
                 x✝ : Nat
                 ⊢ LE.le (HMul.hMul (HMul.hMul 4 0) x✝) (HMul.hMul (HAdd.hAdd 0 x✝) (HAdd.hAdd  …
               -/
  | 0, _ => by rw [Nat.mul_zero, Nat.zero_mul]; exact zero_le _
                                                /-
                                                  🎉 no goals
                                                -/
               /-
                 x✝ : Nat
                 ⊢ LE.le (HMul.hMul (HMul.hMul 4 x✝) 0) (HMul.hMul (HAdd.hAdd x✝ 0) (HAdd.hAdd  …
               -/
  | _, 0 => by rw [Nat.mul_zero]; exact zero_le _
                                  /-
                                    🎉 no goals
                                  -/
  | a + 1, b + 1 => by
    simpa only [Nat.mul_add, Nat.add_mul, show (4 : ℕ) = 1 + 1 + 1 + 1 from rfl, Nat.one_mul,
      Nat.mul_one, Nat.add_assoc, Nat.add_left_comm, Nat.add_le_add_iff_left]
      using Nat.add_le_add_right (@AM_GM a b) 4

-- These two lemmas seem like they belong to `Batteries.Data.Nat.Basic`.


lemma sqrt.iter_sq_le (n guess : ℕ) : sqrt.iter n guess * sqrt.iter n guess ≤ n := by
  /-
    n guess : Nat
    ⊢ LE.le (HMul.hMul (Nat.sqrt.iter n guess) (Nat.sqrt.iter n guess)) n
  -/
  unfold sqrt.iter
  /-
    n guess : Nat
    ⊢ LE.le
        (HMul.hMul
          (let next := HDiv.hDiv (HAdd.hAdd guess (HDiv.hDiv n guess)) 2;
          dite (LT.lt next guess) (fun _h => Nat.sqrt.iter n next) fun _h => guess)
          (let next := HDiv.hDiv (HAdd.hAdd guess (HDiv.hDiv n guess)) 2;
          dite (LT.lt next guess) (fun _h => Nat.sqrt.iter n next) fun _h => guess))
        n
  -/
  let next := (guess + n / guess) / 2
  if h : next < guess then
    simpa only [next, dif_pos h] using sqrt.iter_sq_le n next
  else
    simp only [next, dif_neg h]
    apply Nat.mul_le_of_le_div
    apply Nat.le_of_add_le_add_left (a := guess)
    rw [← Nat.mul_two, ← le_div_iff_mul_le]
    · exact Nat.le_of_not_lt h
    · exact Nat.zero_lt_two


lemma sqrt.lt_iter_succ_sq (n guess : ℕ) (hn : n < (guess + 1) * (guess + 1)) :
    n < (sqrt.iter n guess + 1) * (sqrt.iter n guess + 1) := by
  /-
    n guess : Nat
    hn : LT.lt n (HMul.hMul (HAdd.hAdd guess 1) (HAdd.hAdd guess 1))
    ⊢ LT.lt n (HMul.hMul (HAdd.hAdd (Nat.sqrt.iter n guess) 1) (HAdd.hAdd (Nat.sqr …
  -/
  unfold sqrt.iter
  -- m was `next`
  /-
    n guess : Nat
    hn : LT.lt n (HMul.hMul (HAdd.hAdd guess 1) (HAdd.hAdd guess 1))
    ⊢ LT.lt n
        (HMul.hMul
          (HAdd.hAdd
            (let next := HDiv.hDiv (HAdd.hAdd guess (HDiv.hDiv n guess)) 2;
            dite (LT.lt next guess) (fun _h => Nat.sqrt.iter n next) fun _h => gue …
            1)
          (HAdd.hAdd
            (let next := HDiv.hDiv (HAdd.hAdd guess (HDiv.hDiv n guess)) 2;
            dite (LT.lt next guess) (fun _h => Nat.sqrt.iter n next) fun _h => gue …
            1))
  -/
  let m := (guess + n / guess) / 2
  /-
    n guess : Nat
    hn : LT.lt n (HMul.hMul (HAdd.hAdd guess 1) (HAdd.hAdd guess 1))
    m : Nat := HDiv.hDiv (HAdd.hAdd guess (HDiv.hDiv n guess)) 2
    ⊢ LT.lt n
        (HMul.hMul
          (HAdd.hAdd
            (let next := HDiv.hDiv (HAdd.hAdd guess (HDiv.hDiv n guess)) 2;
            dite (LT.lt next guess) (fun _h => Nat.sqrt.iter n next) fun _h => gue …
            1)
          (HAdd.hAdd
            (let next := HDiv.hDiv (HAdd.hAdd guess (HDiv.hDiv n guess)) 2;
            dite (LT.lt next guess) (fun _h => Nat.sqrt.iter n next) fun _h => gue …
            1))
  -/
  dsimp
  /-
    n guess : Nat
    hn : LT.lt n (HMul.hMul (HAdd.hAdd guess 1) (HAdd.hAdd guess 1))
    m : Nat := HDiv.hDiv (HAdd.hAdd guess (HDiv.hDiv n guess)) 2
    ⊢ LT.lt n (HMul.hMul (HAdd.hAdd (ite (LT.lt (HDiv.hDiv (HAdd.hAdd guess (HDiv. …
  -/
  split_ifs with h
  · suffices n < (m + 1) * (m + 1) by
      simpa only [dif_pos h] using sqrt.lt_iter_succ_sq n m this
    /-
      case pos
      n guess : Nat
      hn : LT.lt n (HMul.hMul (HAdd.hAdd guess 1) (HAdd.hAdd guess 1))
      m : Nat := HDiv.hDiv (HAdd.hAdd guess (HDiv.hDiv n guess)) 2
      h : LT.lt (HDiv.hDiv (HAdd.hAdd guess (HDiv.hDiv n guess)) 2) guess
      ⊢ LT.lt n (HMul.hMul (HAdd.hAdd m 1) (HAdd.hAdd m 1))
    -/
    refine Nat.lt_of_mul_lt_mul_left ?_ (a := 4 * (guess * guess))
    /-
      case pos
      n guess : Nat
      hn : LT.lt n (HMul.hMul (HAdd.hAdd guess 1) (HAdd.hAdd guess 1))
      m : Nat := HDiv.hDiv (HAdd.hAdd guess (HDiv.hDiv n guess)) 2
      h : LT.lt (HDiv.hDiv (HAdd.hAdd guess (HDiv.hDiv n guess)) 2) guess
      ⊢ LT.lt (HMul.hMul (HMul.hMul 4 (HMul.hMul guess guess)) n) (HMul.hMul (HMul.h …
    -/
    apply Nat.lt_of_le_of_lt AM_GM
    /-
      case pos
      n guess : Nat
      hn : LT.lt n (HMul.hMul (HAdd.hAdd guess 1) (HAdd.hAdd guess 1))
      m : Nat := HDiv.hDiv (HAdd.hAdd guess (HDiv.hDiv n guess)) 2
      h : LT.lt (HDiv.hDiv (HAdd.hAdd guess (HDiv.hDiv n guess)) 2) guess
      ⊢ LT.lt (HMul.hMul (HAdd.hAdd (HMul.hMul guess guess) n) (HAdd.hAdd (HMul.hMul …
    -/
    rw [show (4 : ℕ) = 2 * 2 from rfl]
    /-
      case pos
      n guess : Nat
      hn : LT.lt n (HMul.hMul (HAdd.hAdd guess 1) (HAdd.hAdd guess 1))
      m : Nat := HDiv.hDiv (HAdd.hAdd guess (HDiv.hDiv n guess)) 2
      h : LT.lt (HDiv.hDiv (HAdd.hAdd guess (HDiv.hDiv n guess)) 2) guess
      ⊢ LT.lt (HMul.hMul (HAdd.hAdd (HMul.hMul guess guess) n) (HAdd.hAdd (HMul.hMul …
    -/
    rw [Nat.mul_mul_mul_comm 2, Nat.mul_mul_mul_comm (2 * guess)]
    /-
      case pos
      n guess : Nat
      hn : LT.lt n (HMul.hMul (HAdd.hAdd guess 1) (HAdd.hAdd guess 1))
      m : Nat := HDiv.hDiv (HAdd.hAdd guess (HDiv.hDiv n guess)) 2
      h : LT.lt (HDiv.hDiv (HAdd.hAdd guess (HDiv.hDiv n guess)) 2) guess
      ⊢ LT.lt (HMul.hMul (HAdd.hAdd (HMul.hMul guess guess) n) (HAdd.hAdd (HMul.hMul …
    -/
    refine Nat.mul_self_lt_mul_self (?_ : _ < _ * ((_ / 2) + 1))
    rw [← add_div_right _ (by decide), Nat.mul_comm 2, Nat.mul_assoc,
      show guess + n / guess + 2 = (guess + n / guess + 1) + 1 from rfl]
    /-
      case pos
      n guess : Nat
      hn : LT.lt n (HMul.hMul (HAdd.hAdd guess 1) (HAdd.hAdd guess 1))
      m : Nat := HDiv.hDiv (HAdd.hAdd guess (HDiv.hDiv n guess)) 2
      h : LT.lt (HDiv.hDiv (HAdd.hAdd guess (HDiv.hDiv n guess)) 2) guess
      ⊢ LT.lt (HAdd.hAdd (HMul.hMul guess guess) n) (HMul.hMul guess (HMul.hMul 2 (H …
    -/
    have aux_lemma {a : ℕ} : a ≤ 2 * ((a + 1) / 2) := by omega
    /-
      case pos
      n guess : Nat
      hn : LT.lt n (HMul.hMul (HAdd.hAdd guess 1) (HAdd.hAdd guess 1))
      m : Nat := HDiv.hDiv (HAdd.hAdd guess (HDiv.hDiv n guess)) 2
      h : LT.lt (HDiv.hDiv (HAdd.hAdd guess (HDiv.hDiv n guess)) 2) guess
      aux_lemma : ∀ {a : Nat}, LE.le a (HMul.hMul 2 (HDiv.hDiv (HAdd.hAdd a 1) 2))
      ⊢ LT.lt (HAdd.hAdd (HMul.hMul guess guess) n) (HMul.hMul guess (HMul.hMul 2 (H …
    -/
    refine lt_of_lt_of_le ?_ (Nat.mul_le_mul_left _ aux_lemma)
    /-
      case pos
      n guess : Nat
      hn : LT.lt n (HMul.hMul (HAdd.hAdd guess 1) (HAdd.hAdd guess 1))
      m : Nat := HDiv.hDiv (HAdd.hAdd guess (HDiv.hDiv n guess)) 2
      h : LT.lt (HDiv.hDiv (HAdd.hAdd guess (HDiv.hDiv n guess)) 2) guess
      aux_lemma : ∀ {a : Nat}, LE.le a (HMul.hMul 2 (HDiv.hDiv (HAdd.hAdd a 1) 2))
      ⊢ LT.lt (HAdd.hAdd (HMul.hMul guess guess) n) (HMul.hMul guess (HAdd.hAdd (HAd …
    -/
    rw [Nat.add_assoc, Nat.mul_add]
    /-
      case pos
      n guess : Nat
      hn : LT.lt n (HMul.hMul (HAdd.hAdd guess 1) (HAdd.hAdd guess 1))
      m : Nat := HDiv.hDiv (HAdd.hAdd guess (HDiv.hDiv n guess)) 2
      h : LT.lt (HDiv.hDiv (HAdd.hAdd guess (HDiv.hDiv n guess)) 2) guess
      aux_lemma : ∀ {a : Nat}, LE.le a (HMul.hMul 2 (HDiv.hDiv (HAdd.hAdd a 1) 2))
      ⊢ LT.lt (HAdd.hAdd (HMul.hMul guess guess) n) (HAdd.hAdd (HMul.hMul guess gues …
    -/
    exact Nat.add_lt_add_left (lt_mul_div_succ _ (lt_of_le_of_lt (Nat.zero_le m) h)) _
    /-
      🎉 no goals
    -/
    /-
      case neg
      n guess : Nat
      hn : LT.lt n (HMul.hMul (HAdd.hAdd guess 1) (HAdd.hAdd guess 1))
      m : Nat := HDiv.hDiv (HAdd.hAdd guess (HDiv.hDiv n guess)) 2
      h : Not (LT.lt (HDiv.hDiv (HAdd.hAdd guess (HDiv.hDiv n guess)) 2) guess)
      ⊢ LT.lt n (HMul.hMul (HAdd.hAdd guess 1) (HAdd.hAdd guess 1))
    -/
  · simpa only [dif_neg h] using hn
    /-
      🎉 no goals
    -/
-- Porting note: the implementation of `Nat.sqrt` in `Batteries` no longer needs `sqrt_aux`.

private def IsSqrt (n q : ℕ) : Prop :=
  q * q ≤ n ∧ n < (q + 1) * (q + 1)
-- Porting note: as the definition of square root has changed,
-- the proof of `sqrt_isSqrt` is attempted from scratch.
/-
Sketch of proof:
Up to rounding, in terms of the definition of `sqrt.iter`,

* By AM-GM inequality, `next² ≥ n` giving one of the bounds.
* When we terminated, we have `guess ≥ next` from which we deduce the other bound `n ≥ next²`.

To turn this into a lean proof we need to manipulate, use properties of natural number division etc.
-/

private lemma sqrt_isSqrt (n : ℕ) : IsSqrt n (sqrt n) := by
  match n with
  | 0 => simp [IsSqrt, sqrt]
  | 1 => simp [IsSqrt, sqrt]
  | n + 2 =>
    have h : ¬ (n + 2) ≤ 1 := by simp
    simp only [IsSqrt, sqrt, h, ite_false]
    refine ⟨sqrt.iter_sq_le _ _, sqrt.lt_iter_succ_sq _ _ ?_⟩
    simp only [Nat.mul_add, Nat.add_mul, Nat.one_mul, Nat.mul_one, ← Nat.add_assoc]
    rw [Nat.lt_add_one_iff, Nat.add_assoc, ← Nat.mul_two]
    refine le_trans (Nat.le_of_eq (div_add_mod' (n + 2) 2).symm) ?_
    rw [Nat.add_comm, Nat.add_le_add_iff_right, add_mod_right]
    simp only [Nat.zero_lt_two, add_div_right, succ_mul_succ]
    refine le_trans (b := 1) ?_ ?_
    · exact (lt_succ.1 <| mod_lt n Nat.zero_lt_two)
    · exact Nat.le_add_left _ _


lemma sqrt_le (n : ℕ) : sqrt n * sqrt n ≤ n := (sqrt_isSqrt n).left


                                              /-
                                                n : Nat
                                                ⊢ LE.le (HPow.hPow n.sqrt 2) n
                                              -/
lemma sqrt_le' (n : ℕ) : sqrt n ^ 2 ≤ n := by simpa [Nat.pow_two] using sqrt_le n
                                              /-
                                                🎉 no goals
                                              -/


lemma lt_succ_sqrt (n : ℕ) : n < succ (sqrt n) * succ (sqrt n) := (sqrt_isSqrt n).right


                                                          /-
                                                            n : Nat
                                                            ⊢ LT.lt n (HPow.hPow n.sqrt.succ 2)
                                                          -/
lemma lt_succ_sqrt' (n : ℕ) : n < succ (sqrt n) ^ 2 := by simpa [Nat.pow_two] using lt_succ_sqrt n
                                                          /-
                                                            🎉 no goals
                                                          -/


lemma sqrt_le_add (n : ℕ) : n ≤ sqrt n * sqrt n + sqrt n + sqrt n := by
  /-
    n : Nat
    ⊢ LE.le n (HAdd.hAdd (HAdd.hAdd (HMul.hMul n.sqrt n.sqrt) n.sqrt) n.sqrt)
  -/
  rw [← succ_mul]; exact le_of_lt_succ (lt_succ_sqrt n)
                   /-
                     🎉 no goals
                   -/


lemma le_sqrt : m ≤ sqrt n ↔ m * m ≤ n :=
  ⟨fun h ↦ le_trans (mul_self_le_mul_self h) (sqrt_le n),
    fun h ↦ le_of_lt_succ <| Nat.mul_self_lt_mul_self_iff.1 <| lt_of_le_of_lt h (lt_succ_sqrt n)⟩


                                              /-
                                                m n : Nat
                                                ⊢ Iff (LE.le m n.sqrt) (LE.le (HPow.hPow m 2) n)
                                              -/
lemma le_sqrt' : m ≤ sqrt n ↔ m ^ 2 ≤ n := by simpa only [Nat.pow_two] using le_sqrt
                                              /-
                                                🎉 no goals
                                              -/


                                             /-
                                               m n : Nat
                                               ⊢ Iff (LT.lt m.sqrt n) (LT.lt m (HMul.hMul n n))
                                             -/
lemma sqrt_lt : sqrt m < n ↔ m < n * n := by simp only [← not_le, le_sqrt]
                                             /-
                                               🎉 no goals
                                             -/


                                              /-
                                                m n : Nat
                                                ⊢ Iff (LT.lt m.sqrt n) (LT.lt m (HPow.hPow n 2))
                                              -/
lemma sqrt_lt' : sqrt m < n ↔ m < n ^ 2 := by simp only [← not_le, le_sqrt']
                                              /-
                                                🎉 no goals
                                              -/


lemma sqrt_le_self (n : ℕ) : sqrt n ≤ n := le_trans (le_mul_self _) (sqrt_le n)


lemma sqrt_le_sqrt (h : m ≤ n) : sqrt m ≤ sqrt n := le_sqrt.2 (le_trans (sqrt_le _) h)


@[simp] lemma sqrt_zero : sqrt 0 = 0 := rfl


@[simp] lemma sqrt_one : sqrt 1 = 1 := rfl


lemma sqrt_eq_zero : sqrt n = 0 ↔ n = 0 :=
  ⟨fun h ↦
                                                                        /-
                                                                          n : Nat
                                                                          h : Eq n.sqrt 0
                                                                          ⊢ LT.lt n.sqrt 1
                                                                        -/
      Nat.eq_zero_of_le_zero <| le_of_lt_succ <| (@sqrt_lt n 1).1 <| by rw [h]; decide,
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
       /-
         n : Nat
         ⊢ Eq n 0 → Eq n.sqrt 0
       -/
    by rintro rfl; simp⟩
                   /-
                     🎉 no goals
                   -/


lemma eq_sqrt : a = sqrt n ↔ a * a ≤ n ∧ n < (a + 1) * (a + 1) :=
  ⟨fun e ↦ e.symm ▸ sqrt_isSqrt n,
   fun ⟨h₁, h₂⟩ ↦ le_antisymm (le_sqrt.2 h₁) (le_of_lt_succ <| sqrt_lt.2 h₂)⟩


lemma eq_sqrt' : a = sqrt n ↔ a ^ 2 ≤ n ∧ n < (a + 1) ^ 2 := by
  /-
    n a : Nat
    ⊢ Iff (Eq a n.sqrt) (And (LE.le (HPow.hPow a 2) n) (LT.lt n (HPow.hPow (HAdd.h …
  -/
  simpa only [Nat.pow_two] using eq_sqrt
  /-
    🎉 no goals
  -/


lemma le_three_of_sqrt_eq_one (h : sqrt n = 1) : n ≤ 3 :=
                                          /-
                                            n : Nat
                                            h : Eq n.sqrt 1
                                            ⊢ LT.lt n.sqrt 2
                                          -/
  le_of_lt_succ <| (@sqrt_lt n 2).1 <| by rw [h]; decide
                                                  /-
                                                    🎉 no goals
                                                  -/


lemma sqrt_lt_self (h : 1 < n) : sqrt n < n :=
                  /-
                    n : Nat
                    h : LT.lt 1 n
                    ⊢ LT.lt n (HMul.hMul n n)
                  -/
  sqrt_lt.2 <| by have := Nat.mul_lt_mul_of_pos_left h (lt_of_succ_lt h); rwa [Nat.mul_one] at this
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


lemma sqrt_pos : 0 < sqrt n ↔ 0 < n :=
  le_sqrt


lemma sqrt_add_eq (n : ℕ) (h : a ≤ n + n) : sqrt (n * n + a) = n :=
  le_antisymm
    (le_of_lt_succ <|
      sqrt_lt.2 <| by
        /-
          a n : Nat
          h : LE.le a (HAdd.hAdd n n)
          ⊢ LT.lt (HAdd.hAdd (HMul.hMul n n) a) (HMul.hMul n.succ n.succ)
        -/
        rw [succ_mul, mul_succ, add_succ, Nat.add_assoc]
        /-
          a n : Nat
          h : LE.le a (HAdd.hAdd n n)
          ⊢ LT.lt (HAdd.hAdd (HMul.hMul n n) a) (HAdd.hAdd (HMul.hMul n n) (HAdd.hAdd n  …
        -/
        exact lt_succ_of_le (Nat.add_le_add_left h _))
        /-
          🎉 no goals
        -/
    (le_sqrt.2 <| Nat.le_add_right _ _)


lemma sqrt_add_eq' (n : ℕ) (h : a ≤ n + n) : sqrt (n ^ 2 + a) = n := by
  /-
    a n : Nat
    h : LE.le a (HAdd.hAdd n n)
    ⊢ Eq (HAdd.hAdd (HPow.hPow n 2) a).sqrt n
  -/
  simpa [Nat.pow_two] using sqrt_add_eq n h
  /-
    🎉 no goals
  -/


lemma sqrt_eq (n : ℕ) : sqrt (n * n) = n := sqrt_add_eq n (zero_le _)


lemma sqrt_eq' (n : ℕ) : sqrt (n ^ 2) = n := sqrt_add_eq' n (zero_le _)


lemma sqrt_succ_le_succ_sqrt (n : ℕ) : sqrt n.succ ≤ n.sqrt.succ :=
  le_of_lt_succ <| sqrt_lt.2 <| lt_succ_of_le <|
  succ_le_succ <| le_trans (sqrt_le_add n) <| Nat.add_le_add_right
        /-
          n : Nat
          ⊢ LE.le (HAdd.hAdd (HMul.hMul n.sqrt n.sqrt) n.sqrt) (n.sqrt.succ.succ.mul (HA …
        -/
                                                             /-
                                                               🎉 no goals
                                                             -/
    (by refine add_le_add (Nat.mul_le_mul_right _ ?_) ?_ <;> exact Nat.le_add_right _ 2) _
                                                             /-
                                                               🎉 no goals
                                                             -/


lemma exists_mul_self (x : ℕ) : (∃ n, n * n = x) ↔ sqrt x * sqrt x = x :=
                    /-
                      x : Nat
                      x✝ : Exists fun n => Eq (HMul.hMul n n) x
                      n : Nat
                      hn : Eq (HMul.hMul n n) x
                      ⊢ Eq (HMul.hMul x.sqrt x.sqrt) x
                    -/
  ⟨fun ⟨n, hn⟩ ↦ by rw [← hn, sqrt_eq], fun h ↦ ⟨sqrt x, h⟩⟩
                    /-
                      🎉 no goals
                    -/


lemma exists_mul_self' (x : ℕ) : (∃ n, n ^ 2 = x) ↔ sqrt x ^ 2 = x := by
  /-
    x : Nat
    ⊢ Iff (Exists fun n => Eq (HPow.hPow n 2) x) (Eq (HPow.hPow x.sqrt 2) x)
  -/
  simpa only [Nat.pow_two] using exists_mul_self x
  /-
    🎉 no goals
  -/


lemma sqrt_mul_sqrt_lt_succ (n : ℕ) : sqrt n * sqrt n < n + 1 :=
  Nat.lt_succ_iff.mpr (sqrt_le _)


lemma sqrt_mul_sqrt_lt_succ' (n : ℕ) : sqrt n ^ 2 < n + 1 :=
  Nat.lt_succ_iff.mpr (sqrt_le' _)


lemma succ_le_succ_sqrt (n : ℕ) : n + 1 ≤ (sqrt n + 1) * (sqrt n + 1) :=
  le_of_pred_lt (lt_succ_sqrt _)


lemma succ_le_succ_sqrt' (n : ℕ) : n + 1 ≤ (sqrt n + 1) ^ 2 :=
  le_of_pred_lt (lt_succ_sqrt' _)


/-- There are no perfect squares strictly between m² and (m+1)² -/
lemma not_exists_sq (hl : m * m < n) (hr : n < (m + 1) * (m + 1)) : ¬∃ t, t * t = n := by
  /-
    m n : Nat
    hl : LT.lt (HMul.hMul m m) n
    hr : LT.lt n (HMul.hMul (HAdd.hAdd m 1) (HAdd.hAdd m 1))
    ⊢ Not (Exists fun t => Eq (HMul.hMul t t) n)
  -/
  rintro ⟨t, rfl⟩
  /-
    case intro
    m t : Nat
    hl : LT.lt (HMul.hMul m m) (HMul.hMul t t)
    hr : LT.lt (HMul.hMul t t) (HMul.hMul (HAdd.hAdd m 1) (HAdd.hAdd m 1))
    ⊢ False
  -/
  have h1 : m < t := Nat.mul_self_lt_mul_self_iff.1 hl
  /-
    case intro
    m t : Nat
    hl : LT.lt (HMul.hMul m m) (HMul.hMul t t)
    hr : LT.lt (HMul.hMul t t) (HMul.hMul (HAdd.hAdd m 1) (HAdd.hAdd m 1))
    h1 : LT.lt m t
    ⊢ False
  -/
  have h2 : t < m + 1 := Nat.mul_self_lt_mul_self_iff.1 hr
  /-
    case intro
    m t : Nat
    hl : LT.lt (HMul.hMul m m) (HMul.hMul t t)
    hr : LT.lt (HMul.hMul t t) (HMul.hMul (HAdd.hAdd m 1) (HAdd.hAdd m 1))
    h1 : LT.lt m t
    h2 : LT.lt t (HAdd.hAdd m 1)
    ⊢ False
  -/
  exact (not_lt_of_ge <| le_of_lt_succ h2) h1
  /-
    🎉 no goals
  -/


lemma not_exists_sq' : m ^ 2 < n → n < (m + 1) ^ 2 → ¬∃ t, t ^ 2 = n := by
  /-
    m n : Nat
    ⊢ LT.lt (HPow.hPow m 2) n → LT.lt n (HPow.hPow (HAdd.hAdd m 1) 2) → Not (Exist …
  -/
  simpa only [Nat.pow_two] using not_exists_sq
  /-
    🎉 no goals
  -/


