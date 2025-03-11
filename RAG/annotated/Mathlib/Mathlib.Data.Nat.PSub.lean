/-- Partial predecessor operation. Returns `ppred n = some m`
  if `n = m + 1`, otherwise `none`. -/
def ppred : ℕ → Option ℕ
  | 0 => none
  | n + 1 => some n


@[simp]
theorem ppred_zero : ppred 0 = none := rfl


@[simp]
theorem ppred_succ {n : ℕ} : ppred (succ n) = some n := rfl


/-- Partial subtraction operation. Returns `psub m n = some k`
  if `m = n + k`, otherwise `none`. -/
def psub (m : ℕ) : ℕ → Option ℕ
  | 0 => some m
  | n + 1 => psub m n >>= ppred


@[simp]
theorem psub_zero {m : ℕ} : psub m 0 = some m := rfl


@[simp]
theorem psub_succ {m n : ℕ} : psub m (succ n) = psub m n >>= ppred := rfl


                                                                /-
                                                                  n : Nat
                                                                  ⊢ Eq n.pred (n.ppred.getD 0)
                                                                -/
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
theorem pred_eq_ppred (n : ℕ) : pred n = (ppred n).getD 0 := by cases n <;> rfl
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


theorem sub_eq_psub (m : ℕ) : ∀ n, m - n = (psub m n).getD 0
  | 0 => rfl
                                                 /-
                                                   m n : Nat
                                                   ⊢ Eq ((HSub.hSub m n).ppred.getD 0) ((m.psub (HAdd.hAdd n 1)).getD 0)
                                                 -/
                                                                                                /-
                                                                                                  🎉 no goals
                                                                                                -/
  | n + 1 => (pred_eq_ppred (m - n)).trans <| by rw [sub_eq_psub m n, psub]; cases psub m n <;> rfl
                                                                                                /-
                                                                                                  🎉 no goals
                                                                                                -/


@[simp]
theorem ppred_eq_some {m : ℕ} : ∀ {n}, ppred n = some m ↔ succ m = n
            /-
              m : Nat
              ⊢ Iff (Eq (Nat.ppred 0) (Option.some m)) (Eq m.succ 0)
            -/
                                        /-
                                          🎉 no goals
                                        -/
  | 0 => by constructor <;> intro h <;> contradiction
                                        /-
                                          🎉 no goals
                                        -/
                /-
                  m n : Nat
                  ⊢ Iff (Eq (HAdd.hAdd n 1).ppred (Option.some m)) (Eq m.succ (HAdd.hAdd n 1))
                -/
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
  | n + 1 => by constructor <;> intro h <;> injection h <;> subst m <;> rfl
                                                                        /-
                                                                          🎉 no goals
                                                                        -/

-- Porting note: `contradiction` required an `intro` for the goals
-- `ppred (n + 1) = none → n + 1 = 0` and `n + 1 = 0 → ppred (n + 1) = none`


@[simp]
theorem ppred_eq_none : ∀ {n : ℕ}, ppred n = none ↔ n = 0
            /-
              ⊢ Iff (Eq (Nat.ppred 0) Option.none) (Eq 0 0)
            -/
  | 0 => by simp
            /-
              🎉 no goals
            -/
                /-
                  n : Nat
                  ⊢ Iff (Eq (HAdd.hAdd n 1).ppred Option.none) (Eq (HAdd.hAdd n 1) 0)
                -/
                                          /-
                                            🎉 no goals
                                          -/
  | n + 1 => by constructor <;> intro <;> contradiction
                                          /-
                                            🎉 no goals
                                          -/


theorem psub_eq_some {m : ℕ} : ∀ {n k}, psub m n = some k ↔ k + n = m
               /-
                 m k : Nat
                 ⊢ Iff (Eq (m.psub 0) (Option.some k)) (Eq (HAdd.hAdd k 0) m)
               -/
  | 0, k => by simp [eq_comm]
               /-
                 🎉 no goals
               -/
  | n + 1, k => by
    /-
      m n k : Nat
      ⊢ Iff (Eq (m.psub (HAdd.hAdd n 1)) (Option.some k)) (Eq (HAdd.hAdd k (HAdd.hAd …
    -/
    apply Option.bind_eq_some.trans
    /-
      m n k : Nat
      ⊢ Iff (Exists fun a => And (Eq (m.psub n) (Option.some a)) (Eq a.ppred (Option …
    -/
    simp only [psub_eq_some, ppred_eq_some]
    /-
      m n k : Nat
      ⊢ Iff (Exists fun a => And (Eq (HAdd.hAdd a n) m) (Eq k.succ a)) (Eq (HAdd.hAd …
    -/
    simp [add_comm, add_left_comm]
    /-
      🎉 no goals
    -/


theorem psub_eq_none {m n : ℕ} : psub m n = none ↔ m < n := by
  /-
    m n : Nat
    ⊢ Iff (Eq (m.psub n) Option.none) (LT.lt m n)
  -/
  rcases s : psub m n <;> simp [eq_comm]
    /-
      case none
      m n : Nat
      s : Eq (m.psub n) Option.none
      ⊢ LT.lt m n
    -/
  · refine lt_of_not_ge fun h => ?_
    /-
      case none
      m n : Nat
      s : Eq (m.psub n) Option.none
      h : GE.ge m n
      ⊢ False
    -/
    obtain ⟨k, e⟩ := le.dest h
    /-
      case none.intro
      m n : Nat
      s : Eq (m.psub n) Option.none
      h : GE.ge m n
      k : Nat
      e : Eq (HAdd.hAdd n k) m
      ⊢ False
    -/
    injection s.symm.trans (psub_eq_some.2 <| (add_comm _ _).trans e)
    /-
      🎉 no goals
    -/
    /-
      case some
      m n val✝ : Nat
      s : Eq (m.psub n) (Option.some val✝)
      ⊢ LE.le n m
    -/
  · rw [← psub_eq_some.1 s]
    /-
      case some
      m n val✝ : Nat
      s : Eq (m.psub n) (Option.some val✝)
      ⊢ LE.le n (HAdd.hAdd val✝ n)
    -/
    apply Nat.le_add_left
    /-
      🎉 no goals
    -/


theorem ppred_eq_pred {n} (h : 0 < n) : ppred n = some (pred n) :=
  ppred_eq_some.2 <| succ_pred_eq_of_pos h


theorem psub_eq_sub {m n} (h : n ≤ m) : psub m n = some (m - n) :=
  psub_eq_some.2 <| Nat.sub_add_cancel h

-- Porting note: we only have the simp lemma `Option.bind_some` which uses `Option.bind` not `>>=`

theorem psub_add (m n k) :
    psub m (n + k) = (do psub (← psub m n) k) := by
    induction k with
    | zero => simp only [zero_eq, add_zero, psub_zero, Option.bind_eq_bind, Option.bind_some]
    | succ n ih => simp only [ih, add_succ, psub_succ, bind_assoc]


/-- Same as `psub`, but with a more efficient implementation. -/
@[inline]
def psub' (m n : ℕ) : Option ℕ :=
  if n ≤ m then some (m - n) else none


theorem psub'_eq_psub (m n) : psub' m n = psub m n := by
  /-
    m n : Nat
    ⊢ Eq (m.psub' n) (m.psub n)
  -/
  rw [psub']
  /-
    m n : Nat
    ⊢ Eq (ite (LE.le n m) (Option.some (HSub.hSub m n)) Option.none) (m.psub n)
  -/
  split_ifs with h
    /-
      case pos
      m n : Nat
      h : LE.le n m
      ⊢ Eq (Option.some (HSub.hSub m n)) (m.psub n)
    -/
  · exact (psub_eq_sub h).symm
    /-
      🎉 no goals
    -/
    /-
      case neg
      m n : Nat
      h : Not (LE.le n m)
      ⊢ Eq Option.none (m.psub n)
    -/
  · exact (psub_eq_none.2 (not_le.1 h)).symm
    /-
      🎉 no goals
    -/


