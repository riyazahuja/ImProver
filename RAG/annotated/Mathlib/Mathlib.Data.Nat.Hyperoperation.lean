/-- Implementation of the hyperoperation sequence
where `hyperoperation n m k` is the `n`th hyperoperation between `m` and `k`.
-/
def hyperoperation : ℕ → ℕ → ℕ → ℕ
  | 0, _, k => k + 1
  | 1, m, 0 => m
  | 2, _, 0 => 0
  | _ + 3, _, 0 => 1
  | n + 1, m, k + 1 => hyperoperation n m (hyperoperation (n + 1) m k)

-- Basic hyperoperation lemmas

@[simp]
theorem hyperoperation_zero (m : ℕ) : hyperoperation 0 m = Nat.succ :=
                     /-
                       m k : Nat
                       ⊢ Eq (hyperoperation 0 m k) k.succ
                     -/
  funext fun k => by rw [hyperoperation, Nat.succ_eq_add_one]
                     /-
                       🎉 no goals
                     -/


theorem hyperoperation_ge_three_eq_one (n m : ℕ) : hyperoperation (n + 3) m 0 = 1 := by
  /-
    n m : Nat
    ⊢ Eq (hyperoperation (HAdd.hAdd n 3) m 0) 1
  -/
  rw [hyperoperation]
  /-
    🎉 no goals
  -/


theorem hyperoperation_recursion (n m k : ℕ) :
    hyperoperation (n + 1) m (k + 1) = hyperoperation n m (hyperoperation (n + 1) m k) := by
  /-
    n m k : Nat
    ⊢ Eq (hyperoperation (HAdd.hAdd n 1) m (HAdd.hAdd k 1)) (hyperoperation n m (h …
  -/
  rw [hyperoperation]
  /-
    🎉 no goals
  -/

-- Interesting hyperoperation lemmas

@[simp]
theorem hyperoperation_one : hyperoperation 1 = (· + ·) := by
  /-
    ⊢ Eq (hyperoperation 1) fun x1 x2 => HAdd.hAdd x1 x2
  -/
  ext m k
  /-
    case h.h
    m k : Nat
    ⊢ Eq (hyperoperation 1 m k) (HAdd.hAdd m k)
  -/
  induction' k with bn bih
    /-
      case h.h.zero
      m : Nat
      ⊢ Eq (hyperoperation 1 m 0) (HAdd.hAdd m 0)
    -/
  · rw [Nat.add_zero m, hyperoperation]
    /-
      🎉 no goals
    -/
    /-
      case h.h.succ
      m bn : Nat
      bih : Eq (hyperoperation 1 m bn) (HAdd.hAdd m bn)
      ⊢ Eq (hyperoperation 1 m (HAdd.hAdd bn 1)) (HAdd.hAdd m (HAdd.hAdd bn 1))
    -/
  · rw [hyperoperation_recursion, bih, hyperoperation_zero]
    /-
      case h.h.succ
      m bn : Nat
      bih : Eq (hyperoperation 1 m bn) (HAdd.hAdd m bn)
      ⊢ Eq (HAdd.hAdd m bn).succ (HAdd.hAdd m (HAdd.hAdd bn 1))
    -/
    exact Nat.add_assoc m bn 1
    /-
      🎉 no goals
    -/


@[simp]
theorem hyperoperation_two : hyperoperation 2 = (· * ·) := by
  /-
    ⊢ Eq (hyperoperation 2) fun x1 x2 => HMul.hMul x1 x2
  -/
  ext m k
  /-
    case h.h
    m k : Nat
    ⊢ Eq (hyperoperation 2 m k) (HMul.hMul m k)
  -/
  induction' k with bn bih
    /-
      case h.h.zero
      m : Nat
      ⊢ Eq (hyperoperation 2 m 0) (HMul.hMul m 0)
    -/
  · rw [hyperoperation]
    /-
      case h.h.zero
      m : Nat
      ⊢ Eq 0 (HMul.hMul m 0)
    -/
    exact (Nat.mul_zero m).symm
    /-
      🎉 no goals
    -/
    /-
      case h.h.succ
      m bn : Nat
      bih : Eq (hyperoperation 2 m bn) (HMul.hMul m bn)
      ⊢ Eq (hyperoperation 2 m (HAdd.hAdd bn 1)) (HMul.hMul m (HAdd.hAdd bn 1))
    -/
  · rw [hyperoperation_recursion, hyperoperation_one, bih]
    -- Porting note: was `ring`
    /-
      case h.h.succ
      m bn : Nat
      bih : Eq (hyperoperation 2 m bn) (HMul.hMul m bn)
      ⊢ Eq ((fun x1 x2 => HAdd.hAdd x1 x2) m (HMul.hMul m bn)) (HMul.hMul m (HAdd.hA …
    -/
    dsimp only
    /-
      case h.h.succ
      m bn : Nat
      bih : Eq (hyperoperation 2 m bn) (HMul.hMul m bn)
      ⊢ Eq (HAdd.hAdd m (HMul.hMul m bn)) (HMul.hMul m (HAdd.hAdd bn 1))
    -/
    nth_rewrite 1 [← mul_one m]
    /-
      case h.h.succ
      m bn : Nat
      bih : Eq (hyperoperation 2 m bn) (HMul.hMul m bn)
      ⊢ Eq (HAdd.hAdd (HMul.hMul m 1) (HMul.hMul m bn)) (HMul.hMul m (HAdd.hAdd bn 1))
    -/
    rw [← mul_add, add_comm]
    /-
      🎉 no goals
    -/


@[simp]
theorem hyperoperation_three : hyperoperation 3 = (· ^ ·) := by
  /-
    ⊢ Eq (hyperoperation 3) fun x1 x2 => HPow.hPow x1 x2
  -/
  ext m k
  /-
    case h.h
    m k : Nat
    ⊢ Eq (hyperoperation 3 m k) (HPow.hPow m k)
  -/
  induction' k with bn bih
    /-
      case h.h.zero
      m : Nat
      ⊢ Eq (hyperoperation 3 m 0) (HPow.hPow m 0)
    -/
  · rw [hyperoperation_ge_three_eq_one]
    /-
      case h.h.zero
      m : Nat
      ⊢ Eq 1 (HPow.hPow m 0)
    -/
    exact (pow_zero m).symm
    /-
      🎉 no goals
    -/
    /-
      case h.h.succ
      m bn : Nat
      bih : Eq (hyperoperation 3 m bn) (HPow.hPow m bn)
      ⊢ Eq (hyperoperation 3 m (HAdd.hAdd bn 1)) (HPow.hPow m (HAdd.hAdd bn 1))
    -/
  · rw [hyperoperation_recursion, hyperoperation_two, bih]
    /-
      case h.h.succ
      m bn : Nat
      bih : Eq (hyperoperation 3 m bn) (HPow.hPow m bn)
      ⊢ Eq ((fun x1 x2 => HMul.hMul x1 x2) m (HPow.hPow m bn)) (HPow.hPow m (HAdd.hA …
    -/
    exact (pow_succ' m bn).symm
    /-
      🎉 no goals
    -/


theorem hyperoperation_ge_two_eq_self (n m : ℕ) : hyperoperation (n + 2) m 1 = m := by
  /-
    n m : Nat
    ⊢ Eq (hyperoperation (HAdd.hAdd n 2) m 1) m
  -/
  induction' n with nn nih
    /-
      case zero
      m : Nat
      ⊢ Eq (hyperoperation (HAdd.hAdd 0 2) m 1) m
    -/
  · rw [hyperoperation_two]
    /-
      case zero
      m : Nat
      ⊢ Eq ((fun x1 x2 => HMul.hMul x1 x2) m 1) m
    -/
    ring
    /-
      🎉 no goals
    -/
    /-
      case succ
      m nn : Nat
      nih : Eq (hyperoperation (HAdd.hAdd nn 2) m 1) m
      ⊢ Eq (hyperoperation (HAdd.hAdd (HAdd.hAdd nn 1) 2) m 1) m
    -/
  · rw [hyperoperation_recursion, hyperoperation_ge_three_eq_one, nih]
    /-
      🎉 no goals
    -/


theorem hyperoperation_two_two_eq_four (n : ℕ) : hyperoperation (n + 1) 2 2 = 4 := by
  /-
    n : Nat
    ⊢ Eq (hyperoperation (HAdd.hAdd n 1) 2 2) 4
  -/
  induction' n with nn nih
    /-
      case zero
      ⊢ Eq (hyperoperation (HAdd.hAdd 0 1) 2 2) 4
    -/
  · rw [hyperoperation_one]
    /-
      🎉 no goals
    -/
    /-
      case succ
      nn : Nat
      nih : Eq (hyperoperation (HAdd.hAdd nn 1) 2 2) 4
      ⊢ Eq (hyperoperation (HAdd.hAdd (HAdd.hAdd nn 1) 1) 2 2) 4
    -/
  · rw [hyperoperation_recursion, hyperoperation_ge_two_eq_self, nih]
    /-
      🎉 no goals
    -/


theorem hyperoperation_ge_three_one (n : ℕ) : ∀ k : ℕ, hyperoperation (n + 3) 1 k = 1 := by
  /-
    n : Nat
    ⊢ ∀ (k : Nat), Eq (hyperoperation (HAdd.hAdd n 3) 1 k) 1
  -/
  induction' n with nn nih
    /-
      case zero
      ⊢ ∀ (k : Nat), Eq (hyperoperation (HAdd.hAdd 0 3) 1 k) 1
    -/
  · intro k
    /-
      case zero
      k : Nat
      ⊢ Eq (hyperoperation (HAdd.hAdd 0 3) 1 k) 1
    -/
    rw [hyperoperation_three]
    /-
      case zero
      k : Nat
      ⊢ Eq ((fun x1 x2 => HPow.hPow x1 x2) 1 k) 1
    -/
    dsimp
    /-
      case zero
      k : Nat
      ⊢ Eq (HPow.hPow 1 k) 1
    -/
    rw [one_pow]
    /-
      🎉 no goals
    -/
    /-
      case succ
      nn : Nat
      nih : ∀ (k : Nat), Eq (hyperoperation (HAdd.hAdd nn 3) 1 k) 1
      ⊢ ∀ (k : Nat), Eq (hyperoperation (HAdd.hAdd (HAdd.hAdd nn 1) 3) 1 k) 1
    -/
  · intro k
    /-
      case succ
      nn : Nat
      nih : ∀ (k : Nat), Eq (hyperoperation (HAdd.hAdd nn 3) 1 k) 1
      k : Nat
      ⊢ Eq (hyperoperation (HAdd.hAdd (HAdd.hAdd nn 1) 3) 1 k) 1
    -/
    cases k
      /-
        case succ.zero
        nn : Nat
        nih : ∀ (k : Nat), Eq (hyperoperation (HAdd.hAdd nn 3) 1 k) 1
        ⊢ Eq (hyperoperation (HAdd.hAdd (HAdd.hAdd nn 1) 3) 1 0) 1
      -/
    · rw [hyperoperation_ge_three_eq_one]
      /-
        🎉 no goals
      -/
      /-
        case succ.succ
        nn : Nat
        nih : ∀ (k : Nat), Eq (hyperoperation (HAdd.hAdd nn 3) 1 k) 1
        n✝ : Nat
        ⊢ Eq (hyperoperation (HAdd.hAdd (HAdd.hAdd nn 1) 3) 1 (HAdd.hAdd n✝ 1)) 1
      -/
    · rw [hyperoperation_recursion, nih]
      /-
        🎉 no goals
      -/


theorem hyperoperation_ge_four_zero (n k : ℕ) :
    hyperoperation (n + 4) 0 k = if Even k then 1 else 0 := by
  /-
    n k : Nat
    ⊢ Eq (hyperoperation (HAdd.hAdd n 4) 0 k) (ite (Even k) 1 0)
  -/
  induction' k with kk kih
    /-
      case zero
      n : Nat
      ⊢ Eq (hyperoperation (HAdd.hAdd n 4) 0 0) (ite (Even 0) 1 0)
    -/
  · rw [hyperoperation_ge_three_eq_one]
    /-
      case zero
      n : Nat
      ⊢ Eq 1 (ite (Even 0) 1 0)
    -/
    simp only [even_zero, if_true]
    /-
      🎉 no goals
    -/
    /-
      case succ
      n kk : Nat
      kih : Eq (hyperoperation (HAdd.hAdd n 4) 0 kk) (ite (Even kk) 1 0)
      ⊢ Eq (hyperoperation (HAdd.hAdd n 4) 0 (HAdd.hAdd kk 1)) (ite (Even (HAdd.hAdd …
    -/
  · rw [hyperoperation_recursion]
    /-
      case succ
      n kk : Nat
      kih : Eq (hyperoperation (HAdd.hAdd n 4) 0 kk) (ite (Even kk) 1 0)
      ⊢ Eq (hyperoperation (HAdd.hAdd n 3) 0 (hyperoperation (HAdd.hAdd (HAdd.hAdd n …
    -/
    rw [kih]
    /-
      case succ
      n kk : Nat
      kih : Eq (hyperoperation (HAdd.hAdd n 4) 0 kk) (ite (Even kk) 1 0)
      ⊢ Eq (hyperoperation (HAdd.hAdd n 3) 0 (ite (Even kk) 1 0)) (ite (Even (HAdd.h …
    -/
    simp_rw [Nat.even_add_one]
    /-
      case succ
      n kk : Nat
      kih : Eq (hyperoperation (HAdd.hAdd n 4) 0 kk) (ite (Even kk) 1 0)
      ⊢ Eq (hyperoperation (HAdd.hAdd n 3) 0 (ite (Even kk) 1 0)) (ite (Not (Even kk …
    -/
    split_ifs
      /-
        case pos
        n kk : Nat
        kih : Eq (hyperoperation (HAdd.hAdd n 4) 0 kk) (ite (Even kk) 1 0)
        h✝ : Even kk
        ⊢ Eq (hyperoperation (HAdd.hAdd n 3) 0 1) 0
      -/
    · exact hyperoperation_ge_two_eq_self (n + 1) 0
      /-
        🎉 no goals
      -/
      /-
        case neg
        n kk : Nat
        kih : Eq (hyperoperation (HAdd.hAdd n 4) 0 kk) (ite (Even kk) 1 0)
        h✝ : Not (Even kk)
        ⊢ Eq (hyperoperation (HAdd.hAdd n 3) 0 0) 1
      -/
    · exact hyperoperation_ge_three_eq_one n 0
      /-
        🎉 no goals
      -/

