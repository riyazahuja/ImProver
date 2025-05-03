lemma odd_iff : Odd n ↔ n % 2 = 1 where
                         /-
                           n : Int
                           x✝ : Odd n
                           m : Int
                           hm : Eq n (HAdd.hAdd (HMul.hMul 2 m) 1)
                           ⊢ Eq (HMod.hMod n 2) 1
                         -/
  mp := fun ⟨m, hm⟩ ↦ by simp [hm, add_emod]
                         /-
                           🎉 no goals
                         -/
                      /-
                        n : Int
                        h : Eq (HMod.hMod n 2) 1
                        ⊢ Eq n (HAdd.hAdd (HMul.hMul 2 (HDiv.hDiv n 2)) 1)
                      -/
  mpr h := ⟨n / 2, by rw [← h, add_comm, emod_add_ediv n 2]⟩
                      /-
                        🎉 no goals
                      -/


                                             /-
                                               n : Int
                                               ⊢ Iff (Not (Odd n)) (Eq (HMod.hMod n 2) 0)
                                             -/
lemma not_odd_iff : ¬Odd n ↔ n % 2 = 0 := by rw [odd_iff, emod_two_ne_one]
                                             /-
                                               🎉 no goals
                                             -/


@[simp] lemma not_odd_zero : ¬Odd (0 : ℤ) := not_odd_iff.mpr rfl


                                                       /-
                                                         n : Int
                                                         ⊢ Iff (Not (Odd n)) (Even n)
                                                       -/
@[simp] lemma not_odd_iff_even : ¬Odd n ↔ Even n := by rw [not_odd_iff, even_iff]
                                                       /-
                                                         🎉 no goals
                                                       -/

                                                       /-
                                                         n : Int
                                                         ⊢ Iff (Not (Even n)) (Odd n)
                                                       -/
@[simp] lemma not_even_iff_odd : ¬Even n ↔ Odd n := by rw [not_even_iff, odd_iff]
                                                       /-
                                                         🎉 no goals
                                                       -/


@[deprecated not_odd_iff_even (since := "2024-08-21")]
                                               /-
                                                 n : Int
                                                 ⊢ Iff (Even n) (Not (Odd n))
                                               -/
lemma even_iff_not_odd : Even n ↔ ¬Odd n := by rw [not_odd_iff, even_iff]
                                               /-
                                                 🎉 no goals
                                               -/


@[deprecated not_even_iff_odd (since := "2024-08-21")]
                                               /-
                                                 n : Int
                                                 ⊢ Iff (Odd n) (Not (Even n))
                                               -/
lemma odd_iff_not_even : Odd n ↔ ¬Even n := by rw [not_even_iff, odd_iff]
                                               /-
                                                 🎉 no goals
                                               -/


lemma even_or_odd (n : ℤ) : Even n ∨ Odd n := Or.imp_right not_even_iff_odd.1 <| em <| Even n


lemma even_or_odd' (n : ℤ) : ∃ k, n = 2 * k ∨ n = 2 * k + 1 := by
  /-
    n : Int
    ⊢ Exists fun k => Or (Eq n (HMul.hMul 2 k)) (Eq n (HAdd.hAdd (HMul.hMul 2 k) 1))
  -/
  simpa only [two_mul, exists_or, Odd, Even] using even_or_odd n
  /-
    🎉 no goals
  -/


lemma even_xor'_odd (n : ℤ) : Xor' (Even n) (Odd n) := by
  cases even_or_odd n with
  | inl h => exact Or.inl ⟨h, not_odd_iff_even.2 h⟩
  | inr h => exact Or.inr ⟨h, not_even_iff_odd.2 h⟩


lemma even_xor'_odd' (n : ℤ) : ∃ k, Xor' (n = 2 * k) (n = 2 * k + 1) := by
  /-
    n : Int
    ⊢ Exists fun k => Xor' (Eq n (HMul.hMul 2 k)) (Eq n (HAdd.hAdd (HMul.hMul 2 k) …
  -/
  rcases even_or_odd n with (⟨k, rfl⟩ | ⟨k, rfl⟩) <;> use k
  · simpa only [← two_mul, Xor', true_and, eq_self_iff_true, not_true, or_false,
      and_false] using (succ_ne_self (2 * k)).symm
  · simp only [Xor', add_right_eq_self, false_or, eq_self_iff_true, not_true, not_false_iff,
      one_ne_zero, and_self_iff]


instance : DecidablePred (Odd : ℤ → Prop) := fun _ => decidable_of_iff _ not_even_iff_odd


lemma even_add' : Even (m + n) ↔ (Odd m ↔ Odd n) := by
  /-
    m n : Int
    ⊢ Iff (Even (HAdd.hAdd m n)) (Iff (Odd m) (Odd n))
  -/
  rw [even_add, ← not_odd_iff_even, ← not_odd_iff_even, not_iff_not]
  /-
    🎉 no goals
  -/


lemma not_even_two_mul_add_one (n : ℤ) : ¬ Even (2 * n + 1) :=
  not_even_iff_odd.2 <| odd_two_mul_add_one n


lemma even_sub' : Even (m - n) ↔ (Odd m ↔ Odd n) := by
  /-
    m n : Int
    ⊢ Iff (Even (HSub.hSub m n)) (Iff (Odd m) (Odd n))
  -/
  rw [even_sub, ← not_odd_iff_even, ← not_odd_iff_even, not_iff_not]
  /-
    🎉 no goals
  -/


                                                  /-
                                                    m n : Int
                                                    ⊢ Iff (Odd (HMul.hMul m n)) (And (Odd m) (Odd n))
                                                  -/
lemma odd_mul : Odd (m * n) ↔ Odd m ∧ Odd n := by simp [← not_even_iff_odd, not_or, parity_simps]
                                                  /-
                                                    🎉 no goals
                                                  -/


lemma Odd.of_mul_left (h : Odd (m * n)) : Odd m := (odd_mul.mp h).1


lemma Odd.of_mul_right (h : Odd (m * n)) : Odd n := (odd_mul.mp h).2


@[parity_simps] lemma odd_pow {n : ℕ} : Odd (m ^ n) ↔ Odd m ∨ n = 0 := by
  /-
    m : Int
    n : Nat
    ⊢ Iff (Odd (HPow.hPow m n)) (Or (Odd m) (Eq n 0))
  -/
  rw [← not_iff_not, not_odd_iff_even, not_or, not_odd_iff_even, even_pow]
  /-
    🎉 no goals
  -/


lemma odd_pow' {n : ℕ} (h : n ≠ 0) : Odd (m ^ n) ↔ Odd m := odd_pow.trans <| or_iff_left h


@[parity_simps] lemma odd_add : Odd (m + n) ↔ (Odd m ↔ Even n) := by
  /-
    m n : Int
    ⊢ Iff (Odd (HAdd.hAdd m n)) (Iff (Odd m) (Even n))
  -/
  rw [← not_even_iff_odd, even_add, not_iff, ← not_even_iff_odd]
  /-
    🎉 no goals
  -/


                                                      /-
                                                        m n : Int
                                                        ⊢ Iff (Odd (HAdd.hAdd m n)) (Iff (Odd n) (Even m))
                                                      -/
lemma odd_add' : Odd (m + n) ↔ (Odd n ↔ Even m) := by rw [add_comm, odd_add]
                                                      /-
                                                        🎉 no goals
                                                      -/


                                                    /-
                                                      m n : Int
                                                      h : Odd (HAdd.hAdd m n)
                                                      ⊢ Ne m n
                                                    -/
lemma ne_of_odd_add (h : Odd (m + n)) : m ≠ n := by rintro rfl; simp [← not_even_iff_odd] at h
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[parity_simps] lemma odd_sub : Odd (m - n) ↔ (Odd m ↔ Even n) := by
  /-
    m n : Int
    ⊢ Iff (Odd (HSub.hSub m n)) (Iff (Odd m) (Even n))
  -/
  rw [← not_even_iff_odd, even_sub, not_iff, ← not_even_iff_odd]
  /-
    🎉 no goals
  -/


lemma odd_sub' : Odd (m - n) ↔ (Odd n ↔ Even m) := by
  /-
    m n : Int
    ⊢ Iff (Odd (HSub.hSub m n)) (Iff (Odd n) (Even m))
  -/
  rw [← not_even_iff_odd, even_sub, not_iff, not_iff_comm, ← not_even_iff_odd]
  /-
    🎉 no goals
  -/


lemma even_mul_succ_self (n : ℤ) : Even (n * (n + 1)) := by
  /-
    n : Int
    ⊢ Even (HMul.hMul n (HAdd.hAdd n 1))
  -/
  simpa [even_mul, parity_simps] using n.even_or_odd
  /-
    🎉 no goals
  -/


lemma even_mul_pred_self (n : ℤ) : Even (n * (n - 1)) := by
  /-
    n : Int
    ⊢ Even (HMul.hMul n (HSub.hSub n 1))
  -/
  simpa [even_mul, parity_simps] using n.even_or_odd
  /-
    🎉 no goals
  -/


@[simp, norm_cast] lemma odd_coe_nat (n : ℕ) : Odd (n : ℤ) ↔ Odd n := by
  /-
    n : Nat
    ⊢ Iff (Odd ↑n) (Odd n)
  -/
  rw [← not_even_iff_odd, ← Nat.not_even_iff_odd, even_coe_nat]
  /-
    🎉 no goals
  -/


@[simp] lemma natAbs_even : Even n.natAbs ↔ Even n := by
  /-
    n : Int
    ⊢ Iff (Even n.natAbs) (Even n)
  -/
  simp [even_iff_two_dvd, dvd_natAbs, natCast_dvd.symm]
  /-
    🎉 no goals
  -/


@[simp]
lemma natAbs_odd : Odd n.natAbs ↔ Odd n := by
  /-
    n : Int
    ⊢ Iff (Odd n.natAbs) (Odd n)
  -/
  rw [← not_even_iff_odd, ← Nat.not_even_iff_odd, natAbs_even]
  /-
    🎉 no goals
  -/


protected alias ⟨_, _root_.Even.natAbs⟩ := natAbs_even

protected alias ⟨_, _root_.Odd.natAbs⟩ := natAbs_odd


lemma four_dvd_add_or_sub_of_odd {a b : ℤ} (ha : Odd a) (hb : Odd b) :
    4 ∣ a + b ∨ 4 ∣ a - b := by
  /-
    a b : Int
    ha : Odd a
    hb : Odd b
    ⊢ Or (Dvd.dvd 4 (HAdd.hAdd a b)) (Dvd.dvd 4 (HSub.hSub a b))
  -/
  obtain ⟨m, rfl⟩ := ha
  /-
    case intro
    b : Int
    hb : Odd b
    m : Int
    ⊢ Or (Dvd.dvd 4 (HAdd.hAdd (HAdd.hAdd (HMul.hMul 2 m) 1) b)) (Dvd.dvd 4 (HSub. …
  -/
  obtain ⟨n, rfl⟩ := hb
  /-
    case intro.intro
    m n : Int
    ⊢ Or (Dvd.dvd 4 (HAdd.hAdd (HAdd.hAdd (HMul.hMul 2 m) 1) (HAdd.hAdd (HMul.hMul …
  -/
  obtain h | h := Int.even_or_odd (m + n)
    /-
      case intro.intro.inl
      m n : Int
      h : Even (HAdd.hAdd m n)
      ⊢ Or (Dvd.dvd 4 (HAdd.hAdd (HAdd.hAdd (HMul.hMul 2 m) 1) (HAdd.hAdd (HMul.hMul …
    -/
  · right
    /-
      case intro.intro.inl.h
      m n : Int
      h : Even (HAdd.hAdd m n)
      ⊢ Dvd.dvd 4 (HSub.hSub (HAdd.hAdd (HMul.hMul 2 m) 1) (HAdd.hAdd (HMul.hMul 2 n …
    -/
    rw [Int.even_add, ← Int.even_sub] at h
    /-
      case intro.intro.inl.h
      m n : Int
      h : Even (HSub.hSub m n)
      ⊢ Dvd.dvd 4 (HSub.hSub (HAdd.hAdd (HMul.hMul 2 m) 1) (HAdd.hAdd (HMul.hMul 2 n …
    -/
    obtain ⟨k, hk⟩ := h
    /-
      case intro.intro.inl.h.intro
      m n k : Int
      hk : Eq (HSub.hSub m n) (HAdd.hAdd k k)
      ⊢ Dvd.dvd 4 (HSub.hSub (HAdd.hAdd (HMul.hMul 2 m) 1) (HAdd.hAdd (HMul.hMul 2 n …
    -/
    convert dvd_mul_right 4 k using 1
    /-
      case h.e'_4
      m n k : Int
      hk : Eq (HSub.hSub m n) (HAdd.hAdd k k)
      ⊢ Eq (HSub.hSub (HAdd.hAdd (HMul.hMul 2 m) 1) (HAdd.hAdd (HMul.hMul 2 n) 1)) ( …
    -/
    rw [eq_add_of_sub_eq hk, mul_add, add_assoc, add_sub_cancel_right, ← two_mul, ← mul_assoc]
    /-
      case h.e'_4
      m n k : Int
      hk : Eq (HSub.hSub m n) (HAdd.hAdd k k)
      ⊢ Eq (HMul.hMul (HMul.hMul 2 2) k) (HMul.hMul 4 k)
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.inr
      m n : Int
      h : Odd (HAdd.hAdd m n)
      ⊢ Or (Dvd.dvd 4 (HAdd.hAdd (HAdd.hAdd (HMul.hMul 2 m) 1) (HAdd.hAdd (HMul.hMul …
    -/
  · left
    /-
      case intro.intro.inr.h
      m n : Int
      h : Odd (HAdd.hAdd m n)
      ⊢ Dvd.dvd 4 (HAdd.hAdd (HAdd.hAdd (HMul.hMul 2 m) 1) (HAdd.hAdd (HMul.hMul 2 n …
    -/
    obtain ⟨k, hk⟩ := h
    /-
      case intro.intro.inr.h.intro
      m n k : Int
      hk : Eq (HAdd.hAdd m n) (HAdd.hAdd (HMul.hMul 2 k) 1)
      ⊢ Dvd.dvd 4 (HAdd.hAdd (HAdd.hAdd (HMul.hMul 2 m) 1) (HAdd.hAdd (HMul.hMul 2 n …
    -/
    convert dvd_mul_right 4 (k + 1) using 1
    rw [eq_sub_of_add_eq hk, add_right_comm, ← add_sub, mul_add, mul_sub, add_assoc, add_assoc,
      sub_add, add_assoc, ← sub_sub (2 * n), sub_self, zero_sub, sub_neg_eq_add, ← mul_assoc,
      mul_add]
    /-
      case h.e'_4
      m n k : Int
      hk : Eq (HAdd.hAdd m n) (HAdd.hAdd (HMul.hMul 2 k) 1)
      ⊢ Eq (HAdd.hAdd (HMul.hMul (HMul.hMul 2 2) k) (HAdd.hAdd (HMul.hMul 2 1) (HAdd …
    -/
    rfl
    /-
      🎉 no goals
    -/


lemma two_mul_ediv_two_add_one_of_odd : Odd n → 2 * (n / 2) + 1 = n := by
  /-
    n : Int
    ⊢ Odd n → Eq (HAdd.hAdd (HMul.hMul 2 (HDiv.hDiv n 2)) 1) n
  -/
  rintro ⟨c, rfl⟩
  /-
    case intro
    c : Int
    ⊢ Eq (HAdd.hAdd (HMul.hMul 2 (HDiv.hDiv (HAdd.hAdd (HMul.hMul 2 c) 1) 2)) 1) ( …
  -/
  rw [mul_comm]
  /-
    case intro
    c : Int
    ⊢ Eq (HAdd.hAdd (HMul.hMul (HDiv.hDiv (HAdd.hAdd (HMul.hMul 2 c) 1) 2) 2) 1) ( …
  -/
  convert Int.ediv_add_emod' (2 * c + 1) 2
  /-
    case h.e'_2.h.e'_6
    c : Int
    ⊢ Eq 1 (HMod.hMod (HAdd.hAdd (HMul.hMul 2 c) 1) 2)
  -/
  simp [Int.add_emod]
  /-
    🎉 no goals
  -/


lemma ediv_two_mul_two_add_one_of_odd : Odd n → n / 2 * 2 + 1 = n := by
  /-
    n : Int
    ⊢ Odd n → Eq (HAdd.hAdd (HMul.hMul (HDiv.hDiv n 2) 2) 1) n
  -/
  rintro ⟨c, rfl⟩
  /-
    case intro
    c : Int
    ⊢ Eq (HAdd.hAdd (HMul.hMul (HDiv.hDiv (HAdd.hAdd (HMul.hMul 2 c) 1) 2) 2) 1) ( …
  -/
  convert Int.ediv_add_emod' (2 * c + 1) 2
  /-
    case h.e'_2.h.e'_6
    c : Int
    ⊢ Eq 1 (HMod.hMod (HAdd.hAdd (HMul.hMul 2 c) 1) 2)
  -/
  simp [Int.add_emod]
  /-
    🎉 no goals
  -/


lemma add_one_ediv_two_mul_two_of_odd : Odd n → 1 + n / 2 * 2 = n := by
  /-
    n : Int
    ⊢ Odd n → Eq (HAdd.hAdd 1 (HMul.hMul (HDiv.hDiv n 2) 2)) n
  -/
  rintro ⟨c, rfl⟩
  /-
    case intro
    c : Int
    ⊢ Eq (HAdd.hAdd 1 (HMul.hMul (HDiv.hDiv (HAdd.hAdd (HMul.hMul 2 c) 1) 2) 2)) ( …
  -/
  rw [add_comm]
  /-
    case intro
    c : Int
    ⊢ Eq (HAdd.hAdd (HMul.hMul (HDiv.hDiv (HAdd.hAdd (HMul.hMul 2 c) 1) 2) 2) 1) ( …
  -/
  convert Int.ediv_add_emod' (2 * c + 1) 2
  /-
    case h.e'_2.h.e'_6
    c : Int
    ⊢ Eq 1 (HMod.hMod (HAdd.hAdd (HMul.hMul 2 c) 1) 2)
  -/
  simp [Int.add_emod]
  /-
    🎉 no goals
  -/


lemma two_mul_ediv_two_of_odd (h : Odd n) : 2 * (n / 2) = n - 1 :=
  eq_sub_of_add_eq (two_mul_ediv_two_add_one_of_odd h)


@[norm_cast, simp]
theorem isSquare_natCast_iff {n : ℕ} : IsSquare (n : ℤ) ↔ IsSquare n := by
  /-
    n : Nat
    ⊢ Iff (IsSquare ↑n) (IsSquare n)
  -/
  constructor <;> rintro ⟨x, h⟩
    /-
      case mp.intro
      n : Nat
      x : Int
      h : Eq (↑n) (HMul.hMul x x)
      ⊢ IsSquare n
    -/
  · exact ⟨x.natAbs, (natAbs_mul_natAbs_eq h.symm).symm⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr.intro
      n x : Nat
      h : Eq n (HMul.hMul x x)
      ⊢ IsSquare ↑n
    -/
  · exact ⟨x, mod_cast h⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem isSquare_ofNat_iff {n : ℕ} :
    IsSquare (ofNat(n) : ℤ) ↔ IsSquare (ofNat(n) : ℕ) :=
  isSquare_natCast_iff


