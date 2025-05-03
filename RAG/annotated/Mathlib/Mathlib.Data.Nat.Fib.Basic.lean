/-- Implementation of the fibonacci sequence satisfying
`fib 0 = 0, fib 1 = 1, fib (n + 2) = fib n + fib (n + 1)`.

*Note:* We use a stream iterator for better performance when compared to the naive recursive
implementation.
-/

@[pp_nodot]
def fib (n : ℕ) : ℕ :=
  ((fun p : ℕ × ℕ => (p.snd, p.fst + p.snd))^[n] (0, 1)).fst


@[simp]
theorem fib_zero : fib 0 = 0 :=
  rfl


@[simp]
theorem fib_one : fib 1 = 1 :=
  rfl


@[simp]
theorem fib_two : fib 2 = 1 :=
  rfl


/-- Shows that `fib` indeed satisfies the Fibonacci recurrence `Fₙ₊₂ = Fₙ + Fₙ₊₁.` -/
theorem fib_add_two {n : ℕ} : fib (n + 2) = fib n + fib (n + 1) := by
  /-
    n : Nat
    ⊢ Eq (Nat.fib (HAdd.hAdd n 2)) (HAdd.hAdd (Nat.fib n) (Nat.fib (HAdd.hAdd n 1)))
  -/
  simp [fib, Function.iterate_succ_apply']
  /-
    🎉 no goals
  -/


lemma fib_add_one : ∀ {n}, n ≠ 0 → fib (n + 1) = fib (n - 1) + fib n
  | _n + 1, _ => fib_add_two


                                                            /-
                                                              n : Nat
                                                              ⊢ LE.le (Nat.fib n) (Nat.fib (HAdd.hAdd n 1))
                                                            -/
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
theorem fib_le_fib_succ {n : ℕ} : fib n ≤ fib (n + 1) := by cases n <;> simp [fib_add_two]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


@[mono]
theorem fib_mono : Monotone fib :=
  monotone_nat_of_le_succ fun _ => fib_le_fib_succ


@[simp] lemma fib_eq_zero : ∀ {n}, fib n = 0 ↔ n = 0
| 0 => Iff.rfl
| 1 => Iff.rfl
              /-
                n : Nat
                ⊢ Iff (Eq (Nat.fib (HAdd.hAdd n 2)) 0) (Eq (HAdd.hAdd n 2) 0)
              -/
| n + 2 => by simp [fib_add_two, fib_eq_zero]
              /-
                🎉 no goals
              -/


                                                        /-
                                                          n : Nat
                                                          ⊢ Iff (LT.lt 0 (Nat.fib n)) (LT.lt 0 n)
                                                        -/
@[simp] lemma fib_pos {n : ℕ} : 0 < fib n ↔ 0 < n := by simp [pos_iff_ne_zero]
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem fib_add_two_sub_fib_add_one {n : ℕ} : fib (n + 2) - fib (n + 1) = fib n := by
  /-
    n : Nat
    ⊢ Eq (HSub.hSub (Nat.fib (HAdd.hAdd n 2)) (Nat.fib (HAdd.hAdd n 1))) (Nat.fib n)
  -/
  rw [fib_add_two, add_tsub_cancel_right]
  /-
    🎉 no goals
  -/


theorem fib_lt_fib_succ {n : ℕ} (hn : 2 ≤ n) : fib n < fib (n + 1) := by
  /-
    n : Nat
    hn : LE.le 2 n
    ⊢ LT.lt (Nat.fib n) (Nat.fib (HAdd.hAdd n 1))
  -/
  rcases exists_add_of_le hn with ⟨n, rfl⟩
  /-
    case intro
    n : Nat
    hn : LE.le 2 (HAdd.hAdd 2 n)
    ⊢ LT.lt (Nat.fib (HAdd.hAdd 2 n)) (Nat.fib (HAdd.hAdd (HAdd.hAdd 2 n) 1))
  -/
  rw [← tsub_pos_iff_lt, add_comm 2, add_right_comm, fib_add_two, add_tsub_cancel_right, fib_pos]
  /-
    case intro
    n : Nat
    hn : LE.le 2 (HAdd.hAdd 2 n)
    ⊢ LT.lt 0 (HAdd.hAdd n 1)
  -/
  exact succ_pos n
  /-
    🎉 no goals
  -/


/-- `fib (n + 2)` is strictly monotone. -/
theorem fib_add_two_strictMono : StrictMono fun n => fib (n + 2) := by
  /-
    ⊢ StrictMono fun n => Nat.fib (HAdd.hAdd n 2)
  -/
  refine strictMono_nat_of_lt_succ fun n => ?_
  /-
    n : Nat
    ⊢ LT.lt (Nat.fib (HAdd.hAdd n 2)) (Nat.fib (HAdd.hAdd (HAdd.hAdd n 1) 2))
  -/
  rw [add_right_comm]
  /-
    n : Nat
    ⊢ LT.lt (Nat.fib (HAdd.hAdd n 2)) (Nat.fib (HAdd.hAdd (HAdd.hAdd n 2) 1))
  -/
  exact fib_lt_fib_succ (self_le_add_left _ _)
  /-
    🎉 no goals
  -/


lemma fib_strictMonoOn : StrictMonoOn fib (Set.Ici 2)
  | _m + 2, _, _n + 2, _, hmn => fib_add_two_strictMono <| lt_of_add_lt_add_right hmn


lemma fib_lt_fib {m : ℕ} (hm : 2 ≤ m) : ∀ {n}, fib m < fib n ↔ m < n
            /-
              m : Nat
              hm : LE.le 2 m
              ⊢ Iff (LT.lt (Nat.fib m) (Nat.fib 0)) (LT.lt m 0)
            -/
  | 0 => by simp [hm]
            /-
              🎉 no goals
            -/
            /-
              m : Nat
              hm : LE.le 2 m
              ⊢ Iff (LT.lt (Nat.fib m) (Nat.fib 1)) (LT.lt m 1)
            -/
  | 1 => by simp [hm]
            /-
              🎉 no goals
            -/
                                                 /-
                                                   m : Nat
                                                   hm : LE.le 2 m
                                                   n : Nat
                                                   ⊢ Membership.mem (Set.Ici 2) (HAdd.hAdd n 2)
                                                 -/
  | n + 2 => fib_strictMonoOn.lt_iff_lt hm <| by simp
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem le_fib_self {n : ℕ} (five_le_n : 5 ≤ n) : n ≤ fib n := by
  /-
    n : Nat
    five_le_n : LE.le 5 n
    ⊢ LE.le n (Nat.fib n)
  -/
  induction' five_le_n with n five_le_n IH
  · -- 5 ≤ fib 5
    /-
      case refl
      n : Nat
      ⊢ LE.le 5 (Nat.fib 5)
    -/
    rfl
    /-
      🎉 no goals
    -/
  · -- n + 1 ≤ fib (n + 1) for 5 ≤ n
    /-
      case step
      n✝ n : Nat
      five_le_n : Nat.le 5 n
      IH : LE.le n (Nat.fib n)
      ⊢ LE.le n.succ (Nat.fib n.succ)
    -/
    rw [succ_le_iff]
    calc
      n ≤ fib n := IH
      _ < fib (n + 1) := fib_lt_fib_succ (le_trans (by decide) five_le_n)


lemma le_fib_add_one : ∀ n, n ≤ fib n + 1
  | 0 => zero_le_one
  | 1 => one_le_two
  | 2 => le_rfl
  | 3 => le_rfl
  | 4 => le_rfl
  | _n + 5 => (le_fib_self le_add_self).trans <| le_succ _


/-- Subsequent Fibonacci numbers are coprime,
  see https://proofwiki.org/wiki/Consecutive_Fibonacci_Numbers_are_Coprime -/
theorem fib_coprime_fib_succ (n : ℕ) : Nat.Coprime (fib n) (fib (n + 1)) := by
  /-
    n : Nat
    ⊢ (Nat.fib n).Coprime (Nat.fib (HAdd.hAdd n 1))
  -/
  induction' n with n ih
    /-
      case zero
      ⊢ (Nat.fib 0).Coprime (Nat.fib (HAdd.hAdd 0 1))
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case succ
      n : Nat
      ih : (Nat.fib n).Coprime (Nat.fib (HAdd.hAdd n 1))
      ⊢ (Nat.fib (HAdd.hAdd n 1)).Coprime (Nat.fib (HAdd.hAdd (HAdd.hAdd n 1) 1))
    -/
  · simp only [fib_add_two, coprime_add_self_right, Coprime, ih.symm]
    /-
      🎉 no goals
    -/


/-- See https://proofwiki.org/wiki/Fibonacci_Number_in_terms_of_Smaller_Fibonacci_Numbers -/
theorem fib_add (m n : ℕ) : fib (m + n + 1) = fib m * fib n + fib (m + 1) * fib (n + 1) := by
  /-
    m n : Nat
    ⊢ Eq (Nat.fib (HAdd.hAdd (HAdd.hAdd m n) 1)) (HAdd.hAdd (HMul.hMul (Nat.fib m) …
  -/
  induction' n with n ih generalizing m
    /-
      case zero
      m : Nat
      ⊢ Eq (Nat.fib (HAdd.hAdd (HAdd.hAdd m 0) 1)) (HAdd.hAdd (HMul.hMul (Nat.fib m) …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case succ
      n : Nat
      ih : ∀ (m : Nat), Eq (Nat.fib (HAdd.hAdd (HAdd.hAdd m n) 1)) (HAdd.hAdd (HMul. …
      m : Nat
      ⊢ Eq (Nat.fib (HAdd.hAdd (HAdd.hAdd m (HAdd.hAdd n 1)) 1)) (HAdd.hAdd (HMul.hM …
    -/
  · specialize ih (m + 1)
    /-
      case succ
      n m : Nat
      ih : Eq (Nat.fib (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd m 1) n) 1)) (HAdd.hAdd (HMul …
      ⊢ Eq (Nat.fib (HAdd.hAdd (HAdd.hAdd m (HAdd.hAdd n 1)) 1)) (HAdd.hAdd (HMul.hM …
    -/
    rw [add_assoc m 1 n, add_comm 1 n] at ih
    /-
      case succ
      n m : Nat
      ih : Eq (Nat.fib (HAdd.hAdd (HAdd.hAdd m (HAdd.hAdd n 1)) 1)) (HAdd.hAdd (HMul …
      ⊢ Eq (Nat.fib (HAdd.hAdd (HAdd.hAdd m (HAdd.hAdd n 1)) 1)) (HAdd.hAdd (HMul.hM …
    -/
    simp only [fib_add_two, succ_eq_add_one, ih]
    /-
      case succ
      n m : Nat
      ih : Eq (Nat.fib (HAdd.hAdd (HAdd.hAdd m (HAdd.hAdd n 1)) 1)) (HAdd.hAdd (HMul …
      ⊢ Eq (HAdd.hAdd (HMul.hMul (Nat.fib (HAdd.hAdd m 1)) (Nat.fib n)) (HMul.hMul ( …
    -/
    ring
    /-
      🎉 no goals
    -/


theorem fib_two_mul (n : ℕ) : fib (2 * n) = fib n * (2 * fib (n + 1) - fib n) := by
  /-
    n : Nat
    ⊢ Eq (Nat.fib (HMul.hMul 2 n)) (HMul.hMul (Nat.fib n) (HSub.hSub (HMul.hMul 2  …
  -/
  cases n
    /-
      case zero
      ⊢ Eq (Nat.fib (HMul.hMul 2 0)) (HMul.hMul (Nat.fib 0) (HSub.hSub (HMul.hMul 2  …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case succ
      n✝ : Nat
      ⊢ Eq (Nat.fib (HMul.hMul 2 (HAdd.hAdd n✝ 1))) (HMul.hMul (Nat.fib (HAdd.hAdd n …
    -/
  · rw [two_mul, ← add_assoc, fib_add, fib_add_two, two_mul]
    /-
      case succ
      n✝ : Nat
      ⊢ Eq (HAdd.hAdd (HMul.hMul (Nat.fib (HAdd.hAdd n✝ 1)) (Nat.fib n✝)) (HMul.hMul …
    -/
    simp only [← add_assoc, add_tsub_cancel_right]
    /-
      case succ
      n✝ : Nat
      ⊢ Eq (HAdd.hAdd (HMul.hMul (Nat.fib (HAdd.hAdd n✝ 1)) (Nat.fib n✝)) (HMul.hMul …
    -/
    ring
    /-
      🎉 no goals
    -/


theorem fib_two_mul_add_one (n : ℕ) : fib (2 * n + 1) = fib (n + 1) ^ 2 + fib n ^ 2 := by
  /-
    n : Nat
    ⊢ Eq (Nat.fib (HAdd.hAdd (HMul.hMul 2 n) 1)) (HAdd.hAdd (HPow.hPow (Nat.fib (H …
  -/
  rw [two_mul, fib_add]
  /-
    n : Nat
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Nat.fib n) (Nat.fib n)) (HMul.hMul (Nat.fib (HAdd. …
  -/
  ring
  /-
    🎉 no goals
  -/


theorem fib_two_mul_add_two (n : ℕ) :
    fib (2 * n + 2) = fib (n + 1) * (2 * fib n + fib (n + 1)) := by
  /-
    n : Nat
    ⊢ Eq (Nat.fib (HAdd.hAdd (HMul.hMul 2 n) 2)) (HMul.hMul (Nat.fib (HAdd.hAdd n  …
  -/
  rw [fib_add_two, fib_two_mul, fib_two_mul_add_one]
  -- Porting note: A bunch of issues similar to [this zulip thread](https://github.com/leanprover-community/mathlib4/pull/1576) with `zify`
  have : fib n ≤ 2 * fib (n + 1) :=
    le_trans fib_le_fib_succ (mul_comm 2 _ ▸ Nat.le_mul_of_pos_right _ two_pos)
  /-
    n : Nat
    this : LE.le (Nat.fib n) (HMul.hMul 2 (Nat.fib (HAdd.hAdd n 1)))
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Nat.fib n) (HSub.hSub (HMul.hMul 2 (Nat.fib (HAdd. …
  -/
  zify [this]
  /-
    n : Nat
    this : LE.le (Nat.fib n) (HMul.hMul 2 (Nat.fib (HAdd.hAdd n 1)))
    ⊢ Eq (HAdd.hAdd (HMul.hMul (↑(Nat.fib n)) (HSub.hSub (HMul.hMul 2 ↑(Nat.fib (H …
  -/
  ring
  /-
    🎉 no goals
  -/


/-- Computes `(Nat.fib n, Nat.fib (n + 1))` using the binary representation of `n`.
Supports `Nat.fastFib`. -/
def fastFibAux : ℕ → ℕ × ℕ :=
  Nat.binaryRec (fib 0, fib 1) fun b _ p =>
    if b then (p.2 ^ 2 + p.1 ^ 2, p.2 * (2 * p.1 + p.2))
    else (p.1 * (2 * p.2 - p.1), p.2 ^ 2 + p.1 ^ 2)


/-- Computes `Nat.fib n` using the binary representation of `n`.
Proved to be equal to `Nat.fib` in `Nat.fast_fib_eq`. -/
def fastFib (n : ℕ) : ℕ :=
  (fastFibAux n).1


theorem fast_fib_aux_bit_ff (n : ℕ) :
    fastFibAux (bit false n) =
      let p := fastFibAux n
      (p.1 * (2 * p.2 - p.1), p.2 ^ 2 + p.1 ^ 2) := by
  /-
    n : Nat
    ⊢ Eq (Nat.bit Bool.false n).fastFibAux
        (let p := n.fastFibAux;
        { fst := HMul.hMul p.1 (HSub.hSub (HMul.hMul 2 p.2) p.1), snd := HAdd.hAdd …
  -/
  rw [fastFibAux, binaryRec_eq]
    /-
      n : Nat
      ⊢ Eq (ite (Eq Bool.false Bool.true) { fst := HAdd.hAdd (HPow.hPow (Nat.binaryR …
          (let p := n.fastFibAux;
          { fst := HMul.hMul p.1 (HSub.hSub (HMul.hMul 2 p.2) p.1), snd := HAdd.hAdd …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case h
      n : Nat
      ⊢ Or (Eq (ite (Eq Bool.false Bool.true) { fst := HAdd.hAdd (HPow.hPow { fst := …
    -/
  · simp
    /-
      🎉 no goals
    -/


theorem fast_fib_aux_bit_tt (n : ℕ) :
    fastFibAux (bit true n) =
      let p := fastFibAux n
      (p.2 ^ 2 + p.1 ^ 2, p.2 * (2 * p.1 + p.2)) := by
  /-
    n : Nat
    ⊢ Eq (Nat.bit Bool.true n).fastFibAux
        (let p := n.fastFibAux;
        { fst := HAdd.hAdd (HPow.hPow p.2 2) (HPow.hPow p.1 2), snd := HMul.hMul p …
  -/
  rw [fastFibAux, binaryRec_eq]
    /-
      n : Nat
      ⊢ Eq (ite (Eq Bool.true Bool.true) { fst := HAdd.hAdd (HPow.hPow (Nat.binaryRe …
          (let p := n.fastFibAux;
          { fst := HAdd.hAdd (HPow.hPow p.2 2) (HPow.hPow p.1 2), snd := HMul.hMul p …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case h
      n : Nat
      ⊢ Or (Eq (ite (Eq Bool.false Bool.true) { fst := HAdd.hAdd (HPow.hPow { fst := …
    -/
  · simp
    /-
      🎉 no goals
    -/


theorem fast_fib_aux_eq (n : ℕ) : fastFibAux n = (fib n, fib (n + 1)) := by
  /-
    n : Nat
    ⊢ Eq n.fastFibAux { fst := Nat.fib n, snd := Nat.fib (HAdd.hAdd n 1) }
  -/
  refine Nat.binaryRec ?_ ?_ n
    /-
      case refine_1
      n : Nat
      ⊢ Eq (Nat.fastFibAux 0) { fst := Nat.fib 0, snd := Nat.fib (HAdd.hAdd 0 1) }
    -/
  · simp [fastFibAux]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      n : Nat
      ⊢ ∀ (b : Bool) (n : Nat), Eq n.fastFibAux { fst := Nat.fib n, snd := Nat.fib ( …
    -/
  · rintro (_|_) n' ih <;>
      simp only [fast_fib_aux_bit_ff, fast_fib_aux_bit_tt, congr_arg Prod.fst ih,
        congr_arg Prod.snd ih, Prod.mk.inj_iff] <;>
      /-
        case refine_2.false
        n n' : Nat
        ih : Eq n'.fastFibAux { fst := Nat.fib n', snd := Nat.fib (HAdd.hAdd n' 1) }
        ⊢ And (Eq (HMul.hMul (Nat.fib n') (HSub.hSub (HMul.hMul 2 (Nat.fib (HAdd.hAdd  …
      -/
      /-
        🎉 no goals
      -/
      simp [bit, fib_two_mul, fib_two_mul_add_one, fib_two_mul_add_two]
      /-
        🎉 no goals
      -/


                                                      /-
                                                        n : Nat
                                                        ⊢ Eq n.fastFib (Nat.fib n)
                                                      -/
theorem fast_fib_eq (n : ℕ) : fastFib n = fib n := by rw [fastFib, fast_fib_aux_eq]
                                                      /-
                                                        🎉 no goals
                                                      -/


theorem gcd_fib_add_self (m n : ℕ) : gcd (fib m) (fib (n + m)) = gcd (fib m) (fib n) := by
  /-
    m n : Nat
    ⊢ Eq ((Nat.fib m).gcd (Nat.fib (HAdd.hAdd n m))) ((Nat.fib m).gcd (Nat.fib n))
  -/
  rcases Nat.eq_zero_or_pos n with rfl | h
    /-
      case inl
      m : Nat
      ⊢ Eq ((Nat.fib m).gcd (Nat.fib (HAdd.hAdd 0 m))) ((Nat.fib m).gcd (Nat.fib 0))
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    m n : Nat
    h : GT.gt n 0
    ⊢ Eq ((Nat.fib m).gcd (Nat.fib (HAdd.hAdd n m))) ((Nat.fib m).gcd (Nat.fib n))
  -/
  replace h := Nat.succ_pred_eq_of_pos h; rw [← h, succ_eq_add_one]
  calc
    gcd (fib m) (fib (n.pred + 1 + m)) =
        gcd (fib m) (fib n.pred * fib m + fib (n.pred + 1) * fib (m + 1)) := by
        rw [← fib_add n.pred _]
        ring_nf
    _ = gcd (fib m) (fib (n.pred + 1) * fib (m + 1)) := by
        rw [add_comm, gcd_add_mul_right_right (fib m) _ (fib n.pred)]
    _ = gcd (fib m) (fib (n.pred + 1)) :=
      Coprime.gcd_mul_right_cancel_right (fib (n.pred + 1)) (Coprime.symm (fib_coprime_fib_succ m))


theorem gcd_fib_add_mul_self (m n : ℕ) : ∀ k, gcd (fib m) (fib (n + k * m)) = gcd (fib m) (fib n)
            /-
              m n : Nat
              ⊢ Eq ((Nat.fib m).gcd (Nat.fib (HAdd.hAdd n (HMul.hMul 0 m)))) ((Nat.fib m).gc …
            -/
  | 0 => by simp
            /-
              🎉 no goals
            -/
  | k + 1 => by
    /-
      m n k : Nat
      ⊢ Eq ((Nat.fib m).gcd (Nat.fib (HAdd.hAdd n (HMul.hMul (HAdd.hAdd k 1) m)))) ( …
    -/
    rw [← gcd_fib_add_mul_self m n k, add_mul, ← add_assoc, one_mul, gcd_fib_add_self _ _]
    /-
      🎉 no goals
    -/


/-- `fib n` is a strong divisibility sequence,
  see https://proofwiki.org/wiki/GCD_of_Fibonacci_Numbers -/
theorem fib_gcd (m n : ℕ) : fib (gcd m n) = gcd (fib m) (fib n) := by
  induction m, n using Nat.gcd.induction with
  | H0 => simp
  | H1 m n _ h' =>
    rw [← gcd_rec m n] at h'
    conv_rhs => rw [← mod_add_div' n m]
    rwa [gcd_fib_add_mul_self m (n % m) (n / m), gcd_comm (fib m) _]


theorem fib_dvd (m n : ℕ) (h : m ∣ n) : fib m ∣ fib n := by
  /-
    m n : Nat
    h : Dvd.dvd m n
    ⊢ Dvd.dvd (Nat.fib m) (Nat.fib n)
  -/
  rwa [gcd_eq_left_iff_dvd, ← fib_gcd, gcd_eq_left_iff_dvd.mp]
  /-
    🎉 no goals
  -/


theorem fib_succ_eq_sum_choose :
    ∀ n : ℕ, fib (n + 1) = ∑ p ∈ Finset.antidiagonal n, choose p.1 p.2 :=
  twoStepInduction rfl rfl fun n h1 h2 => by
    /-
      n : Nat
      h1 : Eq (Nat.fib (HAdd.hAdd n 1)) ((Finset.HasAntidiagonal.antidiagonal n).sum …
      h2 : Eq (Nat.fib (HAdd.hAdd (HAdd.hAdd n 1) 1)) ((Finset.HasAntidiagonal.antid …
      ⊢ Eq (Nat.fib (HAdd.hAdd (HAdd.hAdd n 2) 1)) ((Finset.HasAntidiagonal.antidiag …
    -/
    rw [fib_add_two, h1, h2, Finset.Nat.antidiagonal_succ_succ', Finset.Nat.antidiagonal_succ']
    /-
      n : Nat
      h1 : Eq (Nat.fib (HAdd.hAdd n 1)) ((Finset.HasAntidiagonal.antidiagonal n).sum …
      h2 : Eq (Nat.fib (HAdd.hAdd (HAdd.hAdd n 1) 1)) ((Finset.HasAntidiagonal.antid …
      ⊢ Eq (HAdd.hAdd ((Finset.HasAntidiagonal.antidiagonal n).sum fun p => p.1.choo …
    -/
    simp [choose_succ_succ, Finset.sum_add_distrib, add_left_comm]
    /-
      🎉 no goals
    -/


theorem fib_succ_eq_succ_sum (n : ℕ) : fib (n + 1) = (∑ k ∈ Finset.range n, fib k) + 1 := by
  /-
    n : Nat
    ⊢ Eq (Nat.fib (HAdd.hAdd n 1)) (HAdd.hAdd ((Finset.range n).sum fun k => Nat.f …
  -/
  induction' n with n ih
    /-
      case zero
      ⊢ Eq (Nat.fib (HAdd.hAdd 0 1)) (HAdd.hAdd ((Finset.range 0).sum fun k => Nat.f …
    -/
  · simp
    /-
      🎉 no goals
    -/
  · calc
      fib (n + 2) = fib n + fib (n + 1) := fib_add_two
      _ = (fib n + ∑ k ∈ Finset.range n, fib k) + 1 := by rw [ih, add_assoc]
      _ = (∑ k ∈ Finset.range (n + 1), fib k) + 1 := by simp [Finset.range_add_one]


