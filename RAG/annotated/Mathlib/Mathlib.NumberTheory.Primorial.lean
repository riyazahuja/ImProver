/-- The primorial `n#` of `n` is the product of the primes less than or equal to `n`.
-/
def primorial (n : ℕ) : ℕ := ∏ p ∈ range (n + 1) with p.Prime, p


local notation x "#" => primorial x


theorem primorial_pos (n : ℕ) : 0 < n# :=
  prod_pos fun _p hp ↦ (mem_filter.1 hp).2.pos


theorem primorial_succ {n : ℕ} (hn1 : n ≠ 1) (hn : Odd n) : (n + 1)# = n# := by
  /-
    n : Nat
    hn1 : Ne n 1
    hn : Odd n
    ⊢ Eq (primorial (HAdd.hAdd n 1)) (primorial n)
  -/
  refine prod_congr ?_ fun _ _ ↦ rfl
  /-
    n : Nat
    hn1 : Ne n 1
    hn : Odd n
    ⊢ Eq (Finset.filter (fun p => Nat.Prime p) (Finset.range (HAdd.hAdd (HAdd.hAdd …
  -/
  rw [range_succ, filter_insert, if_neg fun h ↦ not_even_iff_odd.2 hn _]
  /-
    n : Nat
    hn1 : Ne n 1
    hn : Odd n
    ⊢ Nat.Prime (HAdd.hAdd n 1) → Even n
  -/
  exact fun h ↦ h.even_sub_one <| mt succ.inj hn1
  /-
    🎉 no goals
  -/


theorem primorial_add (m n : ℕ) :
    (m + n)# = m# * ∏ p ∈ Ico (m + 1) (m + n + 1) with p.Prime, p := by
  /-
    m n : Nat
    ⊢ Eq (primorial (HAdd.hAdd m n)) (HMul.hMul (primorial m) ((Finset.filter (fun …
  -/
  rw [primorial, primorial, ← Ico_zero_eq_range, ← prod_union, ← filter_union, Ico_union_Ico_eq_Ico]
  exacts [Nat.zero_le _, add_le_add_right (Nat.le_add_right _ _) _,
    disjoint_filter_filter <| Ico_disjoint_Ico_consecutive _ _ _]


theorem primorial_add_dvd {m n : ℕ} (h : n ≤ m) : (m + n)# ∣ m# * choose (m + n) m :=
  calc
    (m + n)# = m# * ∏ p ∈ Ico (m + 1) (m + n + 1) with p.Prime, p := primorial_add _ _
    _ ∣ m# * choose (m + n) m :=
      mul_dvd_mul_left _ <|
        prod_primes_dvd _ (fun _ hk ↦ (mem_filter.1 hk).2.prime) fun p hp ↦ by
          /-
            m n : Nat
            h : LE.le n m
            p : Nat
            hp : Membership.mem (Finset.filter (fun p => Nat.Prime p) (Finset.Ico (HAdd.hA …
            ⊢ Dvd.dvd p ((HAdd.hAdd m n).choose m)
          -/
          rw [mem_filter, mem_Ico] at hp
          exact hp.2.dvd_choose_add hp.1.1 (h.trans_lt (m.lt_succ_self.trans_le hp.1.1))
              (Nat.lt_succ_iff.1 hp.1.2)


theorem primorial_add_le {m n : ℕ} (h : n ≤ m) : (m + n)# ≤ m# * choose (m + n) m :=
  le_of_dvd (mul_pos (primorial_pos _) (choose_pos <| Nat.le_add_right _ _)) (primorial_add_dvd h)


theorem primorial_le_4_pow (n : ℕ) : n# ≤ 4 ^ n := by
  /-
    n : Nat
    ⊢ LE.le (primorial n) (HPow.hPow 4 n)
  -/
  induction' n using Nat.strong_induction_on with n ihn
  /-
    case h
    n : Nat
    ihn : ∀ (m : Nat), LT.lt m n → LE.le (primorial m) (HPow.hPow 4 m)
    ⊢ LE.le (primorial n) (HPow.hPow 4 n)
  -/
  cases' n with n; · rfl
                     /-
                       🎉 no goals
                     -/
  /-
    case h.succ
    n : Nat
    ihn : ∀ (m : Nat), LT.lt m (HAdd.hAdd n 1) → LE.le (primorial m) (HPow.hPow 4 m)
    ⊢ LE.le (primorial (HAdd.hAdd n 1)) (HPow.hPow 4 (HAdd.hAdd n 1))
  -/
  rcases n.even_or_odd with (⟨m, rfl⟩ | ho)
    /-
      case h.succ.inl.intro
      m : Nat
      ihn : ∀ (m_1 : Nat), LT.lt m_1 (HAdd.hAdd (HAdd.hAdd m m) 1) → LE.le (primoria …
      ⊢ LE.le (primorial (HAdd.hAdd (HAdd.hAdd m m) 1)) (HPow.hPow 4 (HAdd.hAdd (HAd …
    -/
  · rcases m.eq_zero_or_pos with (rfl | hm)
      /-
        case h.succ.inl.intro.inl
        ihn : ∀ (m : Nat), LT.lt m (HAdd.hAdd (HAdd.hAdd 0 0) 1) → LE.le (primorial m) …
        ⊢ LE.le (primorial (HAdd.hAdd (HAdd.hAdd 0 0) 1)) (HPow.hPow 4 (HAdd.hAdd (HAd …
      -/
    · decide
      /-
        🎉 no goals
      -/
    calc
      (m + m + 1)# = (m + 1 + m)# := by rw [add_right_comm]
      _ ≤ (m + 1)# * choose (m + 1 + m) (m + 1) := primorial_add_le m.le_succ
      _ = (m + 1)# * choose (2 * m + 1) m := by rw [choose_symm_add, two_mul, add_right_comm]
      _ ≤ 4 ^ (m + 1) * 4 ^ m :=
        mul_le_mul' (ihn _ <| succ_lt_succ <| (lt_add_iff_pos_left _).2 hm) (choose_middle_le_pow _)
      _ ≤ 4 ^ (m + m + 1) := by rw [← pow_add, add_right_comm]
    /-
      case h.succ.inr
      n : Nat
      ihn : ∀ (m : Nat), LT.lt m (HAdd.hAdd n 1) → LE.le (primorial m) (HPow.hPow 4 m)
      ho : Odd n
      ⊢ LE.le (primorial (HAdd.hAdd n 1)) (HPow.hPow 4 (HAdd.hAdd n 1))
    -/
  · rcases Decidable.eq_or_ne n 1 with (rfl | hn)
      /-
        case h.succ.inr.inl
        ihn : ∀ (m : Nat), LT.lt m (HAdd.hAdd 1 1) → LE.le (primorial m) (HPow.hPow 4 m)
        ho : Odd 1
        ⊢ LE.le (primorial (HAdd.hAdd 1 1)) (HPow.hPow 4 (HAdd.hAdd 1 1))
      -/
    · decide
      /-
        🎉 no goals
      -/
    · calc
        (n + 1)# = n# := primorial_succ hn ho
        _ ≤ 4 ^ n := ihn n n.lt_succ_self
        _ ≤ 4 ^ (n + 1) := pow_le_pow_of_le_right four_pos n.le_succ

