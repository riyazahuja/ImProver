theorem natCast_strictMono : StrictMono (· : ℕ → ℤ) := fun _ _ ↦ Int.ofNat_lt.2


@[deprecated (since := "2024-05-25")] alias coe_nat_strictMono := natCast_strictMono


theorem abs_eq_natAbs : ∀ a : ℤ, |a| = natAbs a
  | (n : ℕ) => abs_of_nonneg <| ofNat_zero_le _
  | -[_+1] => abs_of_nonpos <| le_of_lt <| negSucc_lt_zero _


@[simp, norm_cast] lemma natCast_natAbs (n : ℤ) : (n.natAbs : ℤ) = |n| := n.abs_eq_natAbs.symm


                                                         /-
                                                           a : Int
                                                           ⊢ Eq (abs a).natAbs a.natAbs
                                                         -/
theorem natAbs_abs (a : ℤ) : natAbs |a| = natAbs a := by rw [abs_eq_natAbs]; rfl
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


theorem sign_mul_abs (a : ℤ) : sign a * |a| = a := by
  /-
    a : Int
    ⊢ Eq (HMul.hMul a.sign (abs a)) a
  -/
  rw [abs_eq_natAbs, sign_mul_natAbs a]
  /-
    🎉 no goals
  -/


theorem sign_mul_self_eq_abs (a : ℤ) : sign a * a = |a| := by
  /-
    a : Int
    ⊢ Eq (HMul.hMul a.sign a) (abs a)
  -/
  rw [abs_eq_natAbs, sign_mul_self_eq_natAbs]
  /-
    🎉 no goals
  -/


lemma natAbs_le_self_sq (a : ℤ) : (Int.natAbs a : ℤ) ≤ a ^ 2 := by
  /-
    a : Int
    ⊢ LE.le (↑a.natAbs) (HPow.hPow a 2)
  -/
  rw [← Int.natAbs_sq a, sq]
  /-
    a : Int
    ⊢ LE.le (↑a.natAbs) (HMul.hMul ↑a.natAbs ↑a.natAbs)
  -/
  norm_cast
  /-
    a : Int
    ⊢ LE.le a.natAbs (HMul.hMul a.natAbs a.natAbs)
  -/
  apply Nat.le_mul_self
  /-
    🎉 no goals
  -/


alias natAbs_le_self_pow_two := natAbs_le_self_sq


lemma le_self_sq (b : ℤ) : b ≤ b ^ 2 := le_trans le_natAbs (natAbs_le_self_sq _)


alias le_self_pow_two := le_self_sq


@[norm_cast] lemma abs_natCast (n : ℕ) : |(n : ℤ)| = n := abs_of_nonneg (natCast_nonneg n)


theorem natAbs_sub_pos_iff {i j : ℤ} : 0 < natAbs (i - j) ↔ i ≠ j := by
  /-
    i j : Int
    ⊢ Iff (LT.lt 0 (HSub.hSub i j).natAbs) (Ne i j)
  -/
  rw [natAbs_pos, ne_eq, sub_eq_zero]
  /-
    🎉 no goals
  -/


theorem natAbs_sub_ne_zero_iff {i j : ℤ} : natAbs (i - j) ≠ 0 ↔ i ≠ j :=
  Nat.ne_zero_iff_zero_lt.trans natAbs_sub_pos_iff


@[simp]
theorem abs_lt_one_iff {a : ℤ} : |a| < 1 ↔ a = 0 := by
  /-
    a : Int
    ⊢ Iff (LT.lt (abs a) 1) (Eq a 0)
  -/
  rw [← zero_add 1, lt_add_one_iff, abs_nonpos_iff]
  /-
    🎉 no goals
  -/


theorem abs_le_one_iff {a : ℤ} : |a| ≤ 1 ↔ a = 0 ∨ a = 1 ∨ a = -1 := by
  /-
    a : Int
    ⊢ Iff (LE.le (abs a) 1) (Or (Eq a 0) (Or (Eq a 1) (Eq a (-1))))
  -/
  rw [le_iff_lt_or_eq, abs_lt_one_iff]
  match a with
  | (n : ℕ) => simp [abs_eq_natAbs]
  | -[n+1] =>
      simp only [negSucc_ne_zero, abs_eq_natAbs, natAbs_negSucc, succ_eq_add_one,
        natCast_add, Nat.cast_ofNat_Int, add_left_eq_self, natCast_eq_zero, false_or, reduceNeg]
      rw [negSucc_eq']
      omega


theorem one_le_abs {z : ℤ} (h₀ : z ≠ 0) : 1 ≤ |z| :=
  add_one_le_iff.mpr (abs_pos.mpr h₀)


lemma eq_zero_of_abs_lt_dvd {m x : ℤ} (h1 : m ∣ x) (h2 : |x| < m) : x = 0 := by
  /-
    m x : Int
    h1 : Dvd.dvd m x
    h2 : LT.lt (abs x) m
    ⊢ Eq x 0
  -/
  by_contra h
  /-
    m x : Int
    h1 : Dvd.dvd m x
    h2 : LT.lt (abs x) m
    h : Not (Eq x 0)
    ⊢ False
  -/
  have := Int.natAbs_le_of_dvd_ne_zero h1 h
  /-
    m x : Int
    h1 : Dvd.dvd m x
    h2 : LT.lt (abs x) m
    h : Not (Eq x 0)
    this : LE.le m.natAbs x.natAbs
    ⊢ False
  -/
  rw [Int.abs_eq_natAbs] at h2
  /-
    m x : Int
    h1 : Dvd.dvd m x
    h2 : LT.lt (↑x.natAbs) m
    h : Not (Eq x 0)
    this : LE.le m.natAbs x.natAbs
    ⊢ False
  -/
  omega
  /-
    🎉 no goals
  -/


lemma abs_sub_lt_of_lt_lt {m a b : ℕ} (ha : a < m) (hb : b < m) : |(b : ℤ) - a| < m := by
  /-
    m a b : Nat
    ha : LT.lt a m
    hb : LT.lt b m
    ⊢ LT.lt (abs (HSub.hSub ↑b ↑a)) ↑m
  -/
  rw [abs_lt]; omega
               /-
                 🎉 no goals
               -/


theorem ediv_eq_zero_of_lt_abs {a b : ℤ} (H1 : 0 ≤ a) (H2 : a < |b|) : a / b = 0 :=
  match b, |b|, abs_eq_natAbs b, H2 with
  | (n : ℕ), _, rfl, H2 => ediv_eq_zero_of_lt H1 H2
                                              /-
                                                a b : Int
                                                H1 : LE.le 0 a
                                                H2✝ : LT.lt a (abs b)
                                                n : Nat
                                                H2 : LT.lt a ↑(Int.negSucc n).natAbs
                                                ⊢ Eq (Neg.neg (HDiv.hDiv a (Int.negSucc n))) (-0)
                                              -/
  | -[n+1], _, rfl, H2 => neg_injective <| by rw [← Int.ediv_neg]; exact ediv_eq_zero_of_lt H1 H2
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[simp]
theorem emod_abs (a b : ℤ) : a % |b| = a % b :=
  abs_by_cases (fun i => a % i = a % b) rfl (emod_neg _ _)


theorem emod_lt (a : ℤ) {b : ℤ} (H : b ≠ 0) : a % b < |b| := by
  /-
    a b : Int
    H : Ne b 0
    ⊢ LT.lt (HMod.hMod a b) (abs b)
  -/
  rw [← emod_abs]; exact emod_lt_of_pos _ (abs_pos.2 H)
                   /-
                     🎉 no goals
                   -/


theorem abs_ediv_le_abs : ∀ a b : ℤ, |a / b| ≤ |a| :=
  suffices ∀ (a : ℤ) (n : ℕ), |a / n| ≤ |a| from fun a b =>
    match b, eq_nat_or_neg b with
    | _, ⟨n, Or.inl rfl⟩ => this _ _
                               /-
                                 this : ∀ (a : Int) (n : Nat), LE.le (abs (HDiv.hDiv a ↑n)) (abs a)
                                 a b : Int
                                 n : Nat
                                 ⊢ LE.le (abs (HDiv.hDiv a (Neg.neg ↑n))) (abs a)
                               -/
    | _, ⟨n, Or.inr rfl⟩ => by rw [Int.ediv_neg, abs_neg]; apply this
  /-
    a : Int
    n : Nat
    ⊢ LE.le (abs (HDiv.hDiv a ↑n)) (abs a)
  -/
                                                           /-
                                                             🎉 no goals
                                                           -/
  fun a n => by
  rw [abs_eq_natAbs, abs_eq_natAbs];
  exact ofNat_le_ofNat_of_le
    (match a, n with
      | (m : ℕ), n => Nat.div_le_self _ _
      | -[m+1], 0 => Nat.zero_le _
      | -[m+1], n + 1 => Nat.succ_le_succ (Nat.div_le_self _ _))


theorem abs_sign_of_nonzero {z : ℤ} (hz : z ≠ 0) : |z.sign| = 1 := by
  /-
    z : Int
    hz : Ne z 0
    ⊢ Eq (abs z.sign) 1
  -/
  rw [abs_eq_natAbs, natAbs_sign_of_nonzero hz, Int.ofNat_one]
  /-
    🎉 no goals
  -/


protected theorem sign_eq_ediv_abs (a : ℤ) : sign a = a / |a| :=
                        /-
                          a : Int
                          az : Eq a 0
                          ⊢ Eq a.sign (HDiv.hDiv a (abs a))
                        -/
  if az : a = 0 then by simp [az]
                        /-
                          🎉 no goals
                        -/
  else (Int.ediv_eq_of_eq_mul_left (mt abs_eq_zero.1 az) (sign_mul_abs _).symm).symm


protected theorem sign_eq_abs_ediv (a : ℤ) : sign a = |a| / a :=
                        /-
                          a : Int
                          az : Eq a 0
                          ⊢ Eq a.sign (HDiv.hDiv (abs a) a)
                        -/
  if az : a = 0 then by simp [az]
                        /-
                          🎉 no goals
                        -/
  else (Int.ediv_eq_of_eq_mul_left az (sign_mul_self_eq_abs _).symm).symm


@[to_additive (attr := simp) abs_zsmul_eq_zero]
lemma zpow_abs_eq_one (a : G) (n : ℤ) : a ^ |n| = 1 ↔ a ^ n = 1 := by
  /-
    G : Type u_1
    inst✝ : Group G
    a : G
    n : Int
    ⊢ Iff (Eq (HPow.hPow a (abs n)) 1) (Eq (HPow.hPow a n) 1)
  -/
  rw [← Int.natCast_natAbs, zpow_natCast, pow_natAbs_eq_one]
  /-
    🎉 no goals
  -/


