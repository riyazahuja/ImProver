@[simp, norm_cast]
theorem cast_sub {m n} (h : m ≤ n) : ((n - m : ℕ) : R) = n - m :=
                         /-
                           R : Type u
                           inst✝ : AddGroupWithOne R
                           m n : Nat
                           h : LE.le m n
                           ⊢ Eq (HAdd.hAdd ↑(HSub.hSub n m) ↑m) ↑n
                         -/
  eq_sub_of_add_eq <| by rw [← cast_add, Nat.sub_add_cancel h]
                         /-
                           🎉 no goals
                         -/
-- `HasLiftT` appeared in the type signature


@[simp, norm_cast]
theorem cast_pred : ∀ {n}, 0 < n → ((n - 1 : ℕ) : R) = n - 1
               /-
                 R : Type u
                 inst✝ : AddGroupWithOne R
                 h : LT.lt 0 0
                 ⊢ Eq (↑(HSub.hSub 0 1)) (HSub.hSub (↑0) 1)
               -/
  | 0, h => by cases h
               /-
                 🎉 no goals
               -/
                   /-
                     R : Type u
                     inst✝ : AddGroupWithOne R
                     n : Nat
                     x✝ : LT.lt 0 (HAdd.hAdd n 1)
                     ⊢ Eq (↑(HSub.hSub (HAdd.hAdd n 1) 1)) (HSub.hSub (↑(HAdd.hAdd n 1)) 1)
                   -/
  | n + 1, _ => by rw [cast_succ, add_sub_cancel_right]; rfl
                                                         /-
                                                           🎉 no goals
                                                         -/


@[simp, norm_cast squash]
theorem cast_negSucc (n : ℕ) : (-[n+1] : R) = -(n + 1 : ℕ) :=
  AddGroupWithOne.intCast_negSucc n
-- expected `n` to be implicit, and `HasLiftT`


@[simp, norm_cast]
theorem cast_zero : ((0 : ℤ) : R) = 0 :=
  (AddGroupWithOne.intCast_ofNat 0).trans Nat.cast_zero
-- type had `HasLiftT`

-- This lemma competes with `Int.ofNat_eq_natCast` to come later

@[simp high, nolint simpNF, norm_cast]
theorem cast_natCast (n : ℕ) : ((n : ℤ) : R) = n :=
  AddGroupWithOne.intCast_ofNat _
-- expected `n` to be implicit, and `HasLiftT`


@[simp, norm_cast]
theorem cast_ofNat (n : ℕ) [n.AtLeastTwo] :
    ((ofNat(n) : ℤ) : R) = ofNat(n) := by
  /-
    R : Type u
    inst✝¹ : AddGroupWithOne R
    n : Nat
    inst✝ : n.AtLeastTwo
    ⊢ Eq (↑(OfNat.ofNat n)) (OfNat.ofNat n)
  -/
  simpa only [OfNat.ofNat] using AddGroupWithOne.intCast_ofNat (R := R) n
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem cast_one : ((1 : ℤ) : R) = 1 := by
  /-
    R : Type u
    inst✝ : AddGroupWithOne R
    ⊢ Eq (↑1) 1
  -/
  erw [cast_natCast, Nat.cast_one]
  /-
    🎉 no goals
  -/
-- type had `HasLiftT`


@[simp, norm_cast]
theorem cast_neg : ∀ n, ((-n : ℤ) : R) = -n
                  /-
                    R : Type u
                    inst✝ : AddGroupWithOne R
                    ⊢ Eq (↑(Neg.neg ↑0)) (Neg.neg ↑↑0)
                  -/
  | (0 : ℕ) => by erw [cast_zero, neg_zero]
                  /-
                    🎉 no goals
                  -/
                      /-
                        R : Type u
                        inst✝ : AddGroupWithOne R
                        n : Nat
                        ⊢ Eq (↑(Neg.neg ↑(HAdd.hAdd n 1))) (Neg.neg ↑↑(HAdd.hAdd n 1))
                      -/
  | (n + 1 : ℕ) => by erw [cast_natCast, cast_negSucc]
                      /-
                        🎉 no goals
                      -/
                 /-
                   R : Type u
                   inst✝ : AddGroupWithOne R
                   n : Nat
                   ⊢ Eq (↑(Neg.neg (Int.negSucc n))) (Neg.neg ↑(Int.negSucc n))
                 -/
  | -[n+1] => by erw [cast_natCast, cast_negSucc, neg_neg]
                 /-
                   🎉 no goals
                 -/
-- type had `HasLiftT`


@[simp, norm_cast]
theorem cast_subNatNat (m n) : ((Int.subNatNat m n : ℤ) : R) = m - n := by
  /-
    R : Type u
    inst✝ : AddGroupWithOne R
    m n : Nat
    ⊢ Eq (↑(Int.subNatNat m n)) (HSub.hSub ↑m ↑n)
  -/
  unfold subNatNat
  /-
    R : Type u
    inst✝ : AddGroupWithOne R
    m n : Nat
    ⊢ Eq (↑(Int.negOfNat.match_1 (fun x => Int) (HSub.hSub n m) (fun _ => Int.ofNa …
  -/
  cases e : n - m
    /-
      case zero
      R : Type u
      inst✝ : AddGroupWithOne R
      m n : Nat
      e : Eq (HSub.hSub n m) 0
      ⊢ Eq (↑(Int.negOfNat.match_1 (fun x => Int) 0 (fun _ => Int.ofNat (HSub.hSub m …
    -/
  · simp only [ofNat_eq_coe]
    /-
      case zero
      R : Type u
      inst✝ : AddGroupWithOne R
      m n : Nat
      e : Eq (HSub.hSub n m) 0
      ⊢ Eq (↑↑(HSub.hSub m n)) (HSub.hSub ↑m ↑n)
    -/
    simp [e, Nat.le_of_sub_eq_zero e]
    /-
      🎉 no goals
    -/
    /-
      case succ
      R : Type u
      inst✝ : AddGroupWithOne R
      m n n✝ : Nat
      e : Eq (HSub.hSub n m) (HAdd.hAdd n✝ 1)
      ⊢ Eq (↑(Int.negOfNat.match_1 (fun x => Int) (HAdd.hAdd n✝ 1) (fun _ => Int.ofN …
    -/
  · rw [cast_negSucc, ← e, Nat.cast_sub <| _root_.le_of_lt <| Nat.lt_of_sub_eq_succ e, neg_sub]
    /-
      🎉 no goals
    -/
-- type had `HasLiftT`


@[simp]
                                                                  /-
                                                                    R : Type u
                                                                    inst✝ : AddGroupWithOne R
                                                                    n : Nat
                                                                    ⊢ Eq (↑(Int.negOfNat n)) (Neg.neg ↑n)
                                                                  -/
theorem cast_negOfNat (n : ℕ) : ((negOfNat n : ℤ) : R) = -n := by simp [Int.cast_neg, negOfNat_eq]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[simp, norm_cast]
theorem cast_add : ∀ m n, ((m + n : ℤ) : R) = m + n
                           /-
                             R : Type u
                             inst✝ : AddGroupWithOne R
                             m n : Nat
                             ⊢ Eq (↑(HAdd.hAdd ↑m ↑n)) (HAdd.hAdd ↑↑m ↑↑n)
                           -/
  | (m : ℕ), (n : ℕ) => by simp [-Int.natCast_add, ← Int.ofNat_add]
                           /-
                             🎉 no goals
                           -/
                          /-
                            R : Type u
                            inst✝ : AddGroupWithOne R
                            m n : Nat
                            ⊢ Eq (↑(HAdd.hAdd (↑m) (Int.negSucc n))) (HAdd.hAdd ↑↑m ↑(Int.negSucc n))
                          -/
  | (m : ℕ), -[n+1] => by erw [cast_subNatNat, cast_natCast, cast_negSucc, sub_eq_add_neg]
                          /-
                            🎉 no goals
                          -/
  | -[m+1], (n : ℕ) => by
    erw [cast_subNatNat, cast_natCast, cast_negSucc, sub_eq_iff_eq_add, add_assoc,
      eq_neg_add_iff_add_eq, ← Nat.cast_add, ← Nat.cast_add, Nat.add_comm]
  | -[m+1], -[n+1] =>
    show (-[m + n + 1+1] : R) = _ by
      rw [cast_negSucc, cast_negSucc, cast_negSucc, ← neg_add_rev, ← Nat.cast_add,
        Nat.add_right_comm m n 1, Nat.add_assoc, Nat.add_comm]
-- type had `HasLiftT`


@[simp, norm_cast]
theorem cast_sub (m n) : ((m - n : ℤ) : R) = m - n := by
  /-
    R : Type u
    inst✝ : AddGroupWithOne R
    m n : Int
    ⊢ Eq (↑(HSub.hSub m n)) (HSub.hSub ↑m ↑n)
  -/
  simp [Int.sub_eq_add_neg, sub_eq_add_neg, Int.cast_neg, Int.cast_add]
  /-
    🎉 no goals
  -/
-- type had `HasLiftT`


theorem cast_two : ((2 : ℤ) : R) = 2 := cast_ofNat _


theorem cast_three : ((3 : ℤ) : R) = 3 := cast_ofNat _


theorem cast_four : ((4 : ℤ) : R) = 4 := cast_ofNat _


                                                                            /-
                                                                              R : Type u_1
                                                                              inst✝ : AddGroupWithOne R
                                                                              n : Int
                                                                              ⊢ Eq (HSMul.hSMul n 1) ↑n
                                                                            -/
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/
@[simp] lemma zsmul_one [AddGroupWithOne R] (n : ℤ) : n • (1 : R) = n := by cases n <;> simp
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/


