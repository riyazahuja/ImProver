@[to_additive] lemma mabs_zpow (n : ℤ) (a : α) : |a ^ n|ₘ = |a|ₘ ^ |n| := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedCommGroup α
    n : Int
    a : α
    ⊢ Eq (mabs (HPow.hPow a n)) (HPow.hPow (mabs a) (abs n))
  -/
  obtain n0 | n0 := le_total 0 n
    /-
      case inl
      α : Type u_1
      inst✝ : LinearOrderedCommGroup α
      n : Int
      a : α
      n0 : LE.le 0 n
      ⊢ Eq (mabs (HPow.hPow a n)) (HPow.hPow (mabs a) (abs n))
    -/
  · obtain ⟨n, rfl⟩ := Int.eq_ofNat_of_zero_le n0
    /-
      case inl.intro
      α : Type u_1
      inst✝ : LinearOrderedCommGroup α
      a : α
      n : Nat
      n0 : LE.le 0 ↑n
      ⊢ Eq (mabs (HPow.hPow a ↑n)) (HPow.hPow (mabs a) (abs ↑n))
    -/
    simp only [mabs_pow, zpow_natCast, Nat.abs_cast]
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝ : LinearOrderedCommGroup α
      n : Int
      a : α
      n0 : LE.le n 0
      ⊢ Eq (mabs (HPow.hPow a n)) (HPow.hPow (mabs a) (abs n))
    -/
  · obtain ⟨m, h⟩ := Int.eq_ofNat_of_zero_le (neg_nonneg.2 n0)
    /-
      case inr.intro
      α : Type u_1
      inst✝ : LinearOrderedCommGroup α
      n : Int
      a : α
      n0 : LE.le n 0
      m : Nat
      h : Eq (Neg.neg n) ↑m
      ⊢ Eq (mabs (HPow.hPow a n)) (HPow.hPow (mabs a) (abs n))
    -/
    rw [← mabs_inv, ← zpow_neg, ← abs_neg, h, zpow_natCast, Nat.abs_cast, zpow_natCast]
    /-
      case inr.intro
      α : Type u_1
      inst✝ : LinearOrderedCommGroup α
      n : Int
      a : α
      n0 : LE.le n 0
      m : Nat
      h : Eq (Neg.neg n) ↑m
      ⊢ Eq (mabs (HPow.hPow a m)) (HPow.hPow (mabs a) m)
    -/
    exact mabs_pow m _
    /-
      🎉 no goals
    -/


lemma odd_abs [LinearOrder α] [Ring α] {a : α} : Odd (abs a) ↔ Odd a := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    inst✝ : Ring α
    a : α
    ⊢ Iff (Odd (abs a)) (Odd a)
  -/
                                   /-
                                     🎉 no goals
                                   -/
  cases' abs_choice a with h h <;> simp only [h, odd_neg]
                                   /-
                                     🎉 no goals
                                   -/


@[simp] lemma abs_one : |(1 : α)| = 1 := abs_of_pos zero_lt_one


lemma abs_two : |(2 : α)| = 2 := abs_of_pos zero_lt_two


lemma abs_mul (a b : α) : |a * b| = |a| * |b| := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedRing α
    a b : α
    ⊢ Eq (abs (HMul.hMul a b)) (HMul.hMul (abs a) (abs b))
  -/
  rw [abs_eq (mul_nonneg (abs_nonneg a) (abs_nonneg b))]
  /-
    α : Type u_1
    inst✝ : LinearOrderedRing α
    a b : α
    ⊢ Or (Eq (HMul.hMul a b) (HMul.hMul (abs a) (abs b))) (Eq (HMul.hMul a b) (Neg …
  -/
  rcases le_total a 0 with ha | ha <;> rcases le_total b 0 with hb | hb <;>
    simp only [abs_of_nonpos, abs_of_nonneg, true_or, or_true, eq_self_iff_true, neg_mul,
      mul_neg, neg_neg, *]


/-- `abs` as a `MonoidWithZeroHom`. -/
def absHom : α →*₀ α where
  toFun := abs
  map_zero' := abs_zero
  map_one' := abs_one
  map_mul' := abs_mul


@[simp]
lemma abs_pow (a : α) (n : ℕ) : |a ^ n| = |a| ^ n := (absHom.toMonoidHom : α →* α).map_pow _ _


lemma pow_abs (a : α) (n : ℕ) : |a| ^ n = |a ^ n| := (abs_pow a n).symm


lemma Even.pow_abs (hn : Even n) (a : α) : |a| ^ n = a ^ n := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedRing α
    n : Nat
    hn : Even n
    a : α
    ⊢ Eq (HPow.hPow (abs a) n) (HPow.hPow a n)
  -/
  rw [← abs_pow, abs_eq_self]; exact hn.pow_nonneg _
                               /-
                                 🎉 no goals
                               -/


                                                         /-
                                                           α : Type u_1
                                                           inst✝ : LinearOrderedRing α
                                                           n : Nat
                                                           ⊢ Eq (abs (HPow.hPow (-1) n)) 1
                                                         -/
lemma abs_neg_one_pow (n : ℕ) : |(-1 : α) ^ n| = 1 := by rw [← pow_abs, abs_neg, abs_one, one_pow]
                                                         /-
                                                           🎉 no goals
                                                         -/


lemma abs_pow_eq_one (a : α) (h : n ≠ 0) : |a ^ n| = 1 ↔ |a| = 1 := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedRing α
    n : Nat
    a : α
    h : Ne n 0
    ⊢ Iff (Eq (abs (HPow.hPow a n)) 1) (Eq (abs a) 1)
  -/
  convert pow_left_inj₀ (abs_nonneg a) zero_le_one h
  /-
    case h.e'_1.h.e'_2
    α : Type u_1
    inst✝ : LinearOrderedRing α
    n : Nat
    a : α
    h : Ne n 0
    ⊢ Eq (abs (HPow.hPow a n)) (HPow.hPow (abs a) n)
  -/
  exacts [(pow_abs _ _).symm, (one_pow _).symm]
  /-
    🎉 no goals
  -/


@[simp] lemma abs_mul_abs_self (a : α) : |a| * |a| = a * a :=
  abs_by_cases (fun x => x * x = a * a) rfl (neg_mul_neg a a)


@[simp]
                                                   /-
                                                     α : Type u_1
                                                     inst✝ : LinearOrderedRing α
                                                     a : α
                                                     ⊢ Eq (abs (HMul.hMul a a)) (HMul.hMul a a)
                                                   -/
lemma abs_mul_self (a : α) : |a * a| = a * a := by rw [abs_mul, abs_mul_abs_self]
                                                   /-
                                                     🎉 no goals
                                                   -/


lemma abs_eq_iff_mul_self_eq : |a| = |b| ↔ a * a = b * b := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedRing α
    a b : α
    ⊢ Iff (Eq (abs a) (abs b)) (Eq (HMul.hMul a a) (HMul.hMul b b))
  -/
  rw [← abs_mul_abs_self, ← abs_mul_abs_self b]
  /-
    α : Type u_1
    inst✝ : LinearOrderedRing α
    a b : α
    ⊢ Iff (Eq (abs a) (abs b)) (Eq (HMul.hMul (abs a) (abs a)) (HMul.hMul (abs b)  …
  -/
  exact (mul_self_inj (abs_nonneg a) (abs_nonneg b)).symm
  /-
    🎉 no goals
  -/


lemma abs_lt_iff_mul_self_lt : |a| < |b| ↔ a * a < b * b := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedRing α
    a b : α
    ⊢ Iff (LT.lt (abs a) (abs b)) (LT.lt (HMul.hMul a a) (HMul.hMul b b))
  -/
  rw [← abs_mul_abs_self, ← abs_mul_abs_self b]
  /-
    α : Type u_1
    inst✝ : LinearOrderedRing α
    a b : α
    ⊢ Iff (LT.lt (abs a) (abs b)) (LT.lt (HMul.hMul (abs a) (abs a)) (HMul.hMul (a …
  -/
  exact mul_self_lt_mul_self_iff (abs_nonneg a) (abs_nonneg b)
  /-
    🎉 no goals
  -/


lemma abs_le_iff_mul_self_le : |a| ≤ |b| ↔ a * a ≤ b * b := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedRing α
    a b : α
    ⊢ Iff (LE.le (abs a) (abs b)) (LE.le (HMul.hMul a a) (HMul.hMul b b))
  -/
  rw [← abs_mul_abs_self, ← abs_mul_abs_self b]
  /-
    α : Type u_1
    inst✝ : LinearOrderedRing α
    a b : α
    ⊢ Iff (LE.le (abs a) (abs b)) (LE.le (HMul.hMul (abs a) (abs a)) (HMul.hMul (a …
  -/
  exact mul_self_le_mul_self_iff (abs_nonneg a) (abs_nonneg b)
  /-
    🎉 no goals
  -/


lemma abs_le_one_iff_mul_self_le_one : |a| ≤ 1 ↔ a * a ≤ 1 := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedRing α
    a : α
    ⊢ Iff (LE.le (abs a) 1) (LE.le (HMul.hMul a a) 1)
  -/
  simpa only [abs_one, one_mul] using @abs_le_iff_mul_self_le α _ a 1
  /-
    🎉 no goals
  -/

-- Porting note: added `simp` to replace `pow_bit0_abs`

                                                     /-
                                                       α : Type u_1
                                                       inst✝ : LinearOrderedRing α
                                                       a : α
                                                       ⊢ Eq (HPow.hPow (abs a) 2) (HPow.hPow a 2)
                                                     -/
@[simp] lemma sq_abs (a : α) : |a| ^ 2 = a ^ 2 := by simpa only [sq] using abs_mul_abs_self a
                                                     /-
                                                       🎉 no goals
                                                     -/


                                             /-
                                               α : Type u_1
                                               inst✝ : LinearOrderedRing α
                                               x : α
                                               ⊢ Eq (abs (HPow.hPow x 2)) (HPow.hPow x 2)
                                             -/
lemma abs_sq (x : α) : |x ^ 2| = x ^ 2 := by simpa only [sq] using abs_mul_self x
                                             /-
                                               🎉 no goals
                                             -/


lemma sq_lt_sq : a ^ 2 < b ^ 2 ↔ |a| < |b| := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedRing α
    a b : α
    ⊢ Iff (LT.lt (HPow.hPow a 2) (HPow.hPow b 2)) (LT.lt (abs a) (abs b))
  -/
  simpa only [sq_abs] using sq_lt_sq₀ (abs_nonneg a) (abs_nonneg b)
  /-
    🎉 no goals
  -/


lemma sq_lt_sq' (h1 : -b < a) (h2 : a < b) : a ^ 2 < b ^ 2 :=
  sq_lt_sq.2 (lt_of_lt_of_le (abs_lt.2 ⟨h1, h2⟩) (le_abs_self _))


lemma sq_le_sq : a ^ 2 ≤ b ^ 2 ↔ |a| ≤ |b| := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedRing α
    a b : α
    ⊢ Iff (LE.le (HPow.hPow a 2) (HPow.hPow b 2)) (LE.le (abs a) (abs b))
  -/
  simpa only [sq_abs] using sq_le_sq₀ (abs_nonneg a) (abs_nonneg b)
  /-
    🎉 no goals
  -/


lemma sq_le_sq' (h1 : -b ≤ a) (h2 : a ≤ b) : a ^ 2 ≤ b ^ 2 :=
  sq_le_sq.2 (le_trans (abs_le.mpr ⟨h1, h2⟩) (le_abs_self _))


lemma abs_lt_of_sq_lt_sq (h : a ^ 2 < b ^ 2) (hb : 0 ≤ b) : |a| < b := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedRing α
    a b : α
    h : LT.lt (HPow.hPow a 2) (HPow.hPow b 2)
    hb : LE.le 0 b
    ⊢ LT.lt (abs a) b
  -/
  rwa [← abs_of_nonneg hb, ← sq_lt_sq]
  /-
    🎉 no goals
  -/


lemma abs_lt_of_sq_lt_sq' (h : a ^ 2 < b ^ 2) (hb : 0 ≤ b) : -b < a ∧ a < b :=
  abs_lt.1 <| abs_lt_of_sq_lt_sq h hb


lemma abs_le_of_sq_le_sq (h : a ^ 2 ≤ b ^ 2) (hb : 0 ≤ b) : |a| ≤ b := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedRing α
    a b : α
    h : LE.le (HPow.hPow a 2) (HPow.hPow b 2)
    hb : LE.le 0 b
    ⊢ LE.le (abs a) b
  -/
  rwa [← abs_of_nonneg hb, ← sq_le_sq]
  /-
    🎉 no goals
  -/


theorem le_of_sq_le_sq (h : a ^ 2 ≤ b ^ 2) (hb : 0 ≤ b) : a ≤ b :=
  le_abs_self a |>.trans <| abs_le_of_sq_le_sq h hb


lemma abs_le_of_sq_le_sq' (h : a ^ 2 ≤ b ^ 2) (hb : 0 ≤ b) : -b ≤ a ∧ a ≤ b :=
  abs_le.1 <| abs_le_of_sq_le_sq h hb


lemma sq_eq_sq_iff_abs_eq_abs (a b : α) : a ^ 2 = b ^ 2 ↔ |a| = |b| := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedRing α
    a b : α
    ⊢ Iff (Eq (HPow.hPow a 2) (HPow.hPow b 2)) (Eq (abs a) (abs b))
  -/
  simp only [le_antisymm_iff, sq_le_sq]
  /-
    🎉 no goals
  -/


@[simp] lemma sq_le_one_iff_abs_le_one (a : α) : a ^ 2 ≤ 1 ↔ |a| ≤ 1 := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedRing α
    a : α
    ⊢ Iff (LE.le (HPow.hPow a 2) 1) (LE.le (abs a) 1)
  -/
  simpa only [one_pow, abs_one] using @sq_le_sq _ _ a 1
  /-
    🎉 no goals
  -/


@[simp] lemma sq_lt_one_iff_abs_lt_one (a : α) : a ^ 2 < 1 ↔ |a| < 1 := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedRing α
    a : α
    ⊢ Iff (LT.lt (HPow.hPow a 2) 1) (LT.lt (abs a) 1)
  -/
  simpa only [one_pow, abs_one] using @sq_lt_sq _ _ a 1
  /-
    🎉 no goals
  -/


@[simp] lemma one_le_sq_iff_one_le_abs (a : α) : 1 ≤ a ^ 2 ↔ 1 ≤ |a| := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedRing α
    a : α
    ⊢ Iff (LE.le 1 (HPow.hPow a 2)) (LE.le 1 (abs a))
  -/
  simpa only [one_pow, abs_one] using @sq_le_sq _ _ 1 a
  /-
    🎉 no goals
  -/


@[simp] lemma one_lt_sq_iff_one_lt_abs (a : α) : 1 < a ^ 2 ↔ 1 < |a| := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedRing α
    a : α
    ⊢ Iff (LT.lt 1 (HPow.hPow a 2)) (LT.lt 1 (abs a))
  -/
  simpa only [one_pow, abs_one] using @sq_lt_sq _ _ 1 a
  /-
    🎉 no goals
  -/


lemma exists_abs_lt {α : Type*} [LinearOrderedRing α] (a : α) : ∃ b > 0, |a| < b :=
                                             /-
                                               α : Type u_2
                                               inst✝ : LinearOrderedRing α
                                               a : α
                                               ⊢ LE.le 1 (HAdd.hAdd (abs a) 1)
                                             -/
  ⟨|a| + 1, lt_of_lt_of_le zero_lt_one <| by simp, lt_add_one |a|⟩
                                             /-
                                               🎉 no goals
                                             -/


theorem abs_sub_sq (a b : α) : |a - b| * |a - b| = a * a + b * b - (1 + 1) * a * b := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedCommRing α
    a b : α
    ⊢ Eq (HMul.hMul (abs (HSub.hSub a b)) (abs (HSub.hSub a b))) (HSub.hSub (HAdd. …
  -/
  rw [abs_mul_abs_self]
  simp only [mul_add, add_comm, add_left_comm, mul_comm, sub_eq_add_neg, mul_one, mul_neg,
    neg_add_rev, neg_neg, add_assoc]


lemma abs_unit_intCast (a : ℤˣ) : |((a : ℤ) : α)| = 1 := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedCommRing α
    a : Units Int
    ⊢ Eq (abs ↑↑a) 1
  -/
                                  /-
                                    🎉 no goals
                                  -/
  cases Int.units_eq_one_or a <;> simp_all
                                  /-
                                    🎉 no goals
                                  -/


private def geomSum : ℕ → α
  | 0 => 1
  | n + 1 => a * geomSum n + b ^ (n + 1)


private theorem abs_geomSum_le : |geomSum a b n| ≤ (n + 1) * max |a| |b| ^ n := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedCommRing α
    a b : α
    n : Nat
    ⊢ LE.le (abs (geomSum a b n)) (HMul.hMul (HAdd.hAdd (↑n) 1) (HPow.hPow (Max.ma …
  -/
  induction' n with n ih; · simp [geomSum]
                            /-
                              🎉 no goals
                            -/
  /-
    case succ
    α : Type u_1
    inst✝ : LinearOrderedCommRing α
    a b : α
    n✝ n : Nat
    ih : LE.le (abs (geomSum a b n)) (HMul.hMul (HAdd.hAdd (↑n) 1) (HPow.hPow (Max …
    ⊢ LE.le (abs (geomSum a b (HAdd.hAdd n 1))) (HMul.hMul (HAdd.hAdd (↑(HAdd.hAdd …
  -/
  refine (abs_add_le ..).trans ?_
  /-
    case succ
    α : Type u_1
    inst✝ : LinearOrderedCommRing α
    a b : α
    n✝ n : Nat
    ih : LE.le (abs (geomSum a b n)) (HMul.hMul (HAdd.hAdd (↑n) 1) (HPow.hPow (Max …
    ⊢ LE.le (HAdd.hAdd (abs (HMul.hMul a (geomSum a b n))) (abs (HPow.hPow b (HAdd …
  -/
  rw [abs_mul, abs_pow, Nat.cast_succ, add_one_mul]
  /-
    case succ
    α : Type u_1
    inst✝ : LinearOrderedCommRing α
    a b : α
    n✝ n : Nat
    ih : LE.le (abs (geomSum a b n)) (HMul.hMul (HAdd.hAdd (↑n) 1) (HPow.hPow (Max …
    ⊢ LE.le (HAdd.hAdd (HMul.hMul (abs a) (abs (geomSum a b n))) (HPow.hPow (abs b …
  -/
  refine add_le_add ?_ (pow_le_pow_left₀ (abs_nonneg _) le_sup_right _)
  /-
    case succ
    α : Type u_1
    inst✝ : LinearOrderedCommRing α
    a b : α
    n✝ n : Nat
    ih : LE.le (abs (geomSum a b n)) (HMul.hMul (HAdd.hAdd (↑n) 1) (HPow.hPow (Max …
    ⊢ LE.le (HMul.hMul (abs a) (abs (geomSum a b n))) (HMul.hMul (HAdd.hAdd (↑n) 1 …
  -/
  rw [pow_succ, ← mul_assoc, mul_comm |a|]
  exact mul_le_mul ih le_sup_left (abs_nonneg _) (mul_nonneg
    (@Nat.cast_succ α .. ▸ Nat.cast_nonneg _) <| pow_nonneg ((abs_nonneg _).trans le_sup_left) _)


private theorem pow_sub_pow_eq_sub_mul_geomSum :
    a ^ (n + 1) - b ^ (n + 1) = (a - b) * geomSum a b n := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedCommRing α
    a b : α
    n : Nat
    ⊢ Eq (HSub.hSub (HPow.hPow a (HAdd.hAdd n 1)) (HPow.hPow b (HAdd.hAdd n 1))) ( …
  -/
  induction' n with n ih; · simp [geomSum]
                            /-
                              🎉 no goals
                            -/
  rw [geomSum, mul_add, mul_comm a, ← mul_assoc, ← ih,
    sub_mul, sub_mul, ← pow_succ, ← pow_succ', mul_comm, sub_add_sub_cancel]


theorem abs_pow_sub_pow_le : |a ^ n - b ^ n| ≤ |a - b| * n * max |a| |b| ^ (n - 1) := by
  /-
    α : Type u_1
    inst✝ : LinearOrderedCommRing α
    a b : α
    n : Nat
    ⊢ LE.le (abs (HSub.hSub (HPow.hPow a n) (HPow.hPow b n))) (HMul.hMul (HMul.hMu …
  -/
  obtain _ | n := n; · simp
                       /-
                         🎉 no goals
                       -/
  /-
    case succ
    α : Type u_1
    inst✝ : LinearOrderedCommRing α
    a b : α
    n : Nat
    ⊢ LE.le (abs (HSub.hSub (HPow.hPow a (HAdd.hAdd n 1)) (HPow.hPow b (HAdd.hAdd  …
  -/
  rw [Nat.add_sub_cancel, pow_sub_pow_eq_sub_mul_geomSum, abs_mul, mul_assoc, Nat.cast_succ]
  /-
    case succ
    α : Type u_1
    inst✝ : LinearOrderedCommRing α
    a b : α
    n : Nat
    ⊢ LE.le (HMul.hMul (abs (HSub.hSub a b)) (abs (geomSum a b n))) (HMul.hMul (ab …
  -/
  exact mul_le_mul_of_nonneg_left (abs_geomSum_le ..) (abs_nonneg _)
  /-
    🎉 no goals
  -/


@[simp]
theorem abs_dvd (a b : α) : |a| ∣ b ↔ a ∣ b := by
  /-
    α : Type u_1
    inst✝¹ : Ring α
    inst✝ : LinearOrder α
    a b : α
    ⊢ Iff (Dvd.dvd (abs a) b) (Dvd.dvd a b)
  -/
                                   /-
                                     🎉 no goals
                                   -/
  cases' abs_choice a with h h <;> simp only [h, neg_dvd]
                                   /-
                                     🎉 no goals
                                   -/


theorem abs_dvd_self (a : α) : |a| ∣ a :=
  (abs_dvd a a).mpr (dvd_refl a)


@[simp]
theorem dvd_abs (a b : α) : a ∣ |b| ↔ a ∣ b := by
  /-
    α : Type u_1
    inst✝¹ : Ring α
    inst✝ : LinearOrder α
    a b : α
    ⊢ Iff (Dvd.dvd a (abs b)) (Dvd.dvd a b)
  -/
                                   /-
                                     🎉 no goals
                                   -/
  cases' abs_choice b with h h <;> simp only [h, dvd_neg]
                                   /-
                                     🎉 no goals
                                   -/


theorem self_dvd_abs (a : α) : a ∣ |a| :=
  (dvd_abs a a).mpr (dvd_refl a)


theorem abs_dvd_abs (a b : α) : |a| ∣ |b| ↔ a ∣ b :=
  (abs_dvd _ _).trans (dvd_abs _ _)


lemma pow_eq_pow_iff_of_ne_zero (hn : n ≠ 0) : a ^ n = b ^ n ↔ a = b ∨ a = -b ∧ Even n :=
  match n.even_xor_odd with
  | .inl hne => by simp only [*, and_true, ← abs_eq_abs,
    ← pow_left_inj₀ (abs_nonneg a) (abs_nonneg b) hn, hne.1.pow_abs]
                  /-
                    R : Type u_2
                    inst✝ : LinearOrderedRing R
                    a b : R
                    n : Nat
                    hn✝ : Ne n 0
                    hn : And (Odd n) (Not (Even n))
                    ⊢ Iff (Eq (HPow.hPow a n) (HPow.hPow b n)) (Or (Eq a b) (And (Eq a (Neg.neg b) …
                  -/
  | .inr hn => by simp [hn, (hn.1.strictMono_pow (R := R)).injective.eq_iff]
                  /-
                    🎉 no goals
                  -/


lemma pow_eq_pow_iff_cases : a ^ n = b ^ n ↔ n = 0 ∨ a = b ∨ a = -b ∧ Even n := by
  /-
    R : Type u_2
    inst✝ : LinearOrderedRing R
    a b : R
    n : Nat
    ⊢ Iff (Eq (HPow.hPow a n) (HPow.hPow b n)) (Or (Eq n 0) (Or (Eq a b) (And (Eq  …
  -/
                                        /-
                                          🎉 no goals
                                        -/
  rcases eq_or_ne n 0 with rfl | hn <;> simp [pow_eq_pow_iff_of_ne_zero, *]
                                        /-
                                          🎉 no goals
                                        -/


lemma pow_eq_one_iff_of_ne_zero (hn : n ≠ 0) : a ^ n = 1 ↔ a = 1 ∨ a = -1 ∧ Even n := by
  /-
    R : Type u_2
    inst✝ : LinearOrderedRing R
    a : R
    n : Nat
    hn : Ne n 0
    ⊢ Iff (Eq (HPow.hPow a n) 1) (Or (Eq a 1) (And (Eq a (-1)) (Even n)))
  -/
  simp [← pow_eq_pow_iff_of_ne_zero hn]
  /-
    🎉 no goals
  -/


lemma pow_eq_one_iff_cases : a ^ n = 1 ↔ n = 0 ∨ a = 1 ∨ a = -1 ∧ Even n := by
  /-
    R : Type u_2
    inst✝ : LinearOrderedRing R
    a : R
    n : Nat
    ⊢ Iff (Eq (HPow.hPow a n) 1) (Or (Eq n 0) (Or (Eq a 1) (And (Eq a (-1)) (Even  …
  -/
  simp [← pow_eq_pow_iff_cases]
  /-
    🎉 no goals
  -/


lemma pow_eq_neg_pow_iff (hb : b ≠ 0) : a ^ n = -b ^ n ↔ a = -b ∧ Odd n :=
  match n.even_or_odd with
  | .inl he =>
                               /-
                                 R : Type u_2
                                 inst✝ : LinearOrderedRing R
                                 a b : R
                                 n : Nat
                                 hb : Ne b 0
                                 he : Even n
                                 this : GT.gt (HPow.hPow a n) (Neg.neg (HPow.hPow b n))
                                 ⊢ Iff (Eq (HPow.hPow a n) (Neg.neg (HPow.hPow b n))) (And (Eq a (Neg.neg b)) ( …
                               -/
                       /-
                         R : Type u_2
                         inst✝ : LinearOrderedRing R
                         a b : R
                         n : Nat
                         hb : Ne b 0
                         he : Even n
                         ⊢ LT.lt (Neg.neg (HPow.hPow b n)) 0
                       -/
    suffices a ^ n > -b ^ n by simpa [he, not_odd_iff_even.2 he] using this.ne'
                       /-
                         🎉 no goals
                       -/
                               /-
                                 🎉 no goals
                               -/
    lt_of_lt_of_le (by simp [he.pow_pos hb]) (he.pow_nonneg _)
  | .inr ho => by
    /-
      R : Type u_2
      inst✝ : LinearOrderedRing R
      a b : R
      n : Nat
      hb : Ne b 0
      ho : Odd n
      ⊢ Iff (Eq (HPow.hPow a n) (Neg.neg (HPow.hPow b n))) (And (Eq a (Neg.neg b)) ( …
    -/
    simp only [ho, and_true, ← ho.neg_pow, (ho.strictMono_pow (R := R)).injective.eq_iff]
    /-
      🎉 no goals
    -/


lemma pow_eq_neg_one_iff : a ^ n = -1 ↔ a = -1 ∧ Odd n := by
  /-
    R : Type u_2
    inst✝ : LinearOrderedRing R
    a : R
    n : Nat
    ⊢ Iff (Eq (HPow.hPow a n) (-1)) (And (Eq a (-1)) (Odd n))
  -/
  simpa using pow_eq_neg_pow_iff (R := R) one_ne_zero
  /-
    🎉 no goals
  -/


/-- If `a` is even, then `n` is odd iff `n % a` is odd. -/
lemma Odd.mod_even_iff (ha : Even a) : Odd (n % a) ↔ Odd n :=
  ((even_sub' <| mod_le n a).mp <|
      even_iff_two_dvd.mpr <| (even_iff_two_dvd.mp ha).trans <| dvd_sub_mod n).symm


/-- If `a` is even, then `n` is even iff `n % a` is even. -/
lemma Even.mod_even_iff (ha : Even a) : Even (n % a) ↔ Even n :=
  ((even_sub <| mod_le n a).mp <|
      even_iff_two_dvd.mpr <| (even_iff_two_dvd.mp ha).trans <| dvd_sub_mod n).symm


/-- If `n` is odd and `a` is even, then `n % a` is odd. -/
lemma Odd.mod_even (hn : Odd n) (ha : Even a) : Odd (n % a) := (Odd.mod_even_iff ha).mpr hn


/-- If `n` is even and `a` is even, then `n % a` is even. -/
lemma Even.mod_even (hn : Even n) (ha : Even a) : Even (n % a) :=
  (Even.mod_even_iff ha).mpr hn


lemma Odd.of_dvd_nat (hn : Odd n) (hm : m ∣ n) : Odd m :=
  not_even_iff_odd.1 <| mt hm.even (not_even_iff_odd.2 hn)


/-- `2` is not a factor of an odd natural number. -/
lemma Odd.ne_two_of_dvd_nat {m n : ℕ} (hn : Odd n) (hm : m ∣ n) : m ≠ 2 := by
  /-
    m n : Nat
    hn : Odd n
    hm : Dvd.dvd m n
    ⊢ Ne m 2
  -/
  rintro rfl
  /-
    n : Nat
    hn : Odd n
    hm : Dvd.dvd 2 n
    ⊢ False
  -/
  exact absurd (hn.of_dvd_nat hm) (by decide)
  /-
    🎉 no goals
  -/

