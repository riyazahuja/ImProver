theorem le_natCast_sub (m n : ℕ) : (m - n : ℤ) ≤ ↑(m - n : ℕ) := by
  /-
    m n : Nat
    ⊢ LE.le (HSub.hSub ↑m ↑n) ↑(HSub.hSub m n)
  -/
  by_cases h : m ≥ n
    /-
      case pos
      m n : Nat
      h : GE.ge m n
      ⊢ LE.le (HSub.hSub ↑m ↑n) ↑(HSub.hSub m n)
    -/
  · exact le_of_eq (Int.ofNat_sub h).symm
    /-
      🎉 no goals
    -/
    /-
      case neg
      m n : Nat
      h : Not (GE.ge m n)
      ⊢ LE.le (HSub.hSub ↑m ↑n) ↑(HSub.hSub m n)
    -/
  · simp [le_of_not_ge h, ofNat_le]
    /-
      🎉 no goals
    -/


theorem succ_natCast_pos (n : ℕ) : 0 < (n : ℤ) + 1 :=
                         /-
                           n : Nat
                           ⊢ LE.le 0 ↑n
                         -/
  lt_add_one_iff.mpr (by simp)
                         /-
                           🎉 no goals
                         -/


theorem natAbs_eq_iff_sq_eq {a b : ℤ} : a.natAbs = b.natAbs ↔ a ^ 2 = b ^ 2 := by
  /-
    a b : Int
    ⊢ Iff (Eq a.natAbs b.natAbs) (Eq (HPow.hPow a 2) (HPow.hPow b 2))
  -/
  rw [sq, sq]
  /-
    a b : Int
    ⊢ Iff (Eq a.natAbs b.natAbs) (Eq (HMul.hMul a a) (HMul.hMul b b))
  -/
  exact natAbs_eq_iff_mul_self_eq
  /-
    🎉 no goals
  -/


theorem natAbs_lt_iff_sq_lt {a b : ℤ} : a.natAbs < b.natAbs ↔ a ^ 2 < b ^ 2 := by
  /-
    a b : Int
    ⊢ Iff (LT.lt a.natAbs b.natAbs) (LT.lt (HPow.hPow a 2) (HPow.hPow b 2))
  -/
  rw [sq, sq]
  /-
    a b : Int
    ⊢ Iff (LT.lt a.natAbs b.natAbs) (LT.lt (HMul.hMul a a) (HMul.hMul b b))
  -/
  exact natAbs_lt_iff_mul_self_lt
  /-
    🎉 no goals
  -/


theorem natAbs_le_iff_sq_le {a b : ℤ} : a.natAbs ≤ b.natAbs ↔ a ^ 2 ≤ b ^ 2 := by
  /-
    a b : Int
    ⊢ Iff (LE.le a.natAbs b.natAbs) (LE.le (HPow.hPow a 2) (HPow.hPow b 2))
  -/
  rw [sq, sq]
  /-
    a b : Int
    ⊢ Iff (LE.le a.natAbs b.natAbs) (LE.le (HMul.hMul a a) (HMul.hMul b b))
  -/
  exact natAbs_le_iff_mul_self_le
  /-
    🎉 no goals
  -/


theorem natAbs_inj_of_nonneg_of_nonneg {a b : ℤ} (ha : 0 ≤ a) (hb : 0 ≤ b) :
                                      /-
                                        a b : Int
                                        ha : LE.le 0 a
                                        hb : LE.le 0 b
                                        ⊢ Iff (Eq a.natAbs b.natAbs) (Eq a b)
                                      -/
    natAbs a = natAbs b ↔ a = b := by rw [← sq_eq_sq₀ ha hb, ← natAbs_eq_iff_sq_eq]
                                      /-
                                        🎉 no goals
                                      -/


theorem natAbs_inj_of_nonpos_of_nonpos {a b : ℤ} (ha : a ≤ 0) (hb : b ≤ 0) :
    natAbs a = natAbs b ↔ a = b := by
  simpa only [Int.natAbs_neg, neg_inj] using
    natAbs_inj_of_nonneg_of_nonneg (neg_nonneg_of_nonpos ha) (neg_nonneg_of_nonpos hb)


theorem natAbs_inj_of_nonneg_of_nonpos {a b : ℤ} (ha : 0 ≤ a) (hb : b ≤ 0) :
    natAbs a = natAbs b ↔ a = -b := by
  /-
    a b : Int
    ha : LE.le 0 a
    hb : LE.le b 0
    ⊢ Iff (Eq a.natAbs b.natAbs) (Eq a (Neg.neg b))
  -/
  simpa only [Int.natAbs_neg] using natAbs_inj_of_nonneg_of_nonneg ha (neg_nonneg_of_nonpos hb)
  /-
    🎉 no goals
  -/


theorem natAbs_inj_of_nonpos_of_nonneg {a b : ℤ} (ha : a ≤ 0) (hb : 0 ≤ b) :
    natAbs a = natAbs b ↔ -a = b := by
  /-
    a b : Int
    ha : LE.le a 0
    hb : LE.le 0 b
    ⊢ Iff (Eq a.natAbs b.natAbs) (Eq (Neg.neg a) b)
  -/
  simpa only [Int.natAbs_neg] using natAbs_inj_of_nonneg_of_nonneg (neg_nonneg_of_nonpos ha) hb
  /-
    🎉 no goals
  -/


/-- A specialization of `abs_sub_le_of_nonneg_of_le` for working with the signed subtraction
  of natural numbers. -/
theorem natAbs_coe_sub_coe_le_of_le {a b n : ℕ} (a_le_n : a ≤ n) (b_le_n : b ≤ n) :
    natAbs (a - b : ℤ) ≤ n := by
  /-
    a b n : Nat
    a_le_n : LE.le a n
    b_le_n : LE.le b n
    ⊢ LE.le (HSub.hSub ↑a ↑b).natAbs n
  -/
  rw [← Nat.cast_le (α := ℤ), natCast_natAbs]
  exact abs_sub_le_of_nonneg_of_le (ofNat_nonneg a) (ofNat_le.mpr a_le_n)
    (ofNat_nonneg b) (ofNat_le.mpr b_le_n)


/-- A specialization of `abs_sub_lt_of_nonneg_of_lt` for working with the signed subtraction
  of natural numbers. -/
theorem natAbs_coe_sub_coe_lt_of_lt {a b n : ℕ} (a_lt_n : a < n) (b_lt_n : b < n) :
    natAbs (a - b : ℤ) < n := by
  /-
    a b n : Nat
    a_lt_n : LT.lt a n
    b_lt_n : LT.lt b n
    ⊢ LT.lt (HSub.hSub ↑a ↑b).natAbs n
  -/
  rw [← Nat.cast_lt (α := ℤ), natCast_natAbs]
  exact abs_sub_lt_of_nonneg_of_lt (ofNat_nonneg a) (ofNat_lt.mpr a_lt_n)
    (ofNat_nonneg b) (ofNat_lt.mpr b_lt_n)


theorem strictMonoOn_natAbs : StrictMonoOn natAbs (Ici 0) := fun _ ha _ _ hab =>
  natAbs_lt_natAbs_of_nonneg_of_lt ha hab


theorem strictAntiOn_natAbs : StrictAntiOn natAbs (Iic 0) := fun a _ b hb hab => by
  simpa [Int.natAbs_neg] using
    natAbs_lt_natAbs_of_nonneg_of_lt (Right.nonneg_neg_iff.mpr hb) (neg_lt_neg_iff.mpr hab)


theorem injOn_natAbs_Ici : InjOn natAbs (Ici 0) :=
  strictMonoOn_natAbs.injOn


theorem injOn_natAbs_Iic : InjOn natAbs (Iic 0) :=
  strictAntiOn_natAbs.injOn


theorem toNat_of_nonpos : ∀ {z : ℤ}, z ≤ 0 → z.toNat = 0
  | 0, _ => rfl
                                    /-
                                      n : Nat
                                      h : LE.le (↑(HAdd.hAdd n 1)) 0
                                      ⊢ LT.lt 0 ↑(HAdd.hAdd n 1)
                                    -/
  | (n + 1 : ℕ), h => (h.not_lt (by simp)).elim
                                    /-
                                      🎉 no goals
                                    -/
  | -[_+1], _ => rfl


@[simp]
theorem div2_bit (b n) : div2 (bit b n) = n := by
  /-
    b : Bool
    n : Int
    ⊢ Eq (Int.bit b n).div2 n
  -/
  rw [bit_val, div2_val, add_comm, Int.add_mul_ediv_left, (_ : (_ / 2 : ℤ) = 0), zero_add]
  /-
    b : Bool
    n : Int
    ⊢ Eq (HDiv.hDiv (cond b 1 0) 2) 0
  -/
  cases b
    /-
      case false
      n : Int
      ⊢ Eq (HDiv.hDiv (cond Bool.false 1 0) 2) 0
    -/
  · decide
    /-
      🎉 no goals
    -/
    /-
      case true
      n : Int
      ⊢ Eq (HDiv.hDiv (cond Bool.true 1 0) 2) 0
    -/
  · show ofNat _ = _
    /-
      case true
      n : Int
      ⊢ Eq (Int.ofNat (HDiv.hDiv 1 2)) 0
    -/
                              /-
                                🎉 no goals
                              -/
    rw [Nat.div_eq_of_lt] <;> simp
                              /-
                                🎉 no goals
                              -/
    /-
      case H
      b : Bool
      n : Int
      ⊢ Ne 2 0
    -/
  · decide
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-04-02")] alias le_coe_nat_sub := le_natCast_sub

@[deprecated (since := "2024-04-02")] alias succ_coe_nat_pos := succ_natCast_pos

@[deprecated (since := "2024-04-02")] alias coe_natAbs := natCast_natAbs

@[deprecated (since := "2024-04-02")] alias coe_nat_eq_zero := natCast_eq_zero

@[deprecated (since := "2024-04-02")] alias coe_nat_ne_zero := natCast_ne_zero

@[deprecated (since := "2024-04-02")] alias coe_nat_ne_zero_iff_pos := natCast_ne_zero_iff_pos

@[deprecated (since := "2024-04-02")] alias abs_coe_nat := abs_natCast

@[deprecated (since := "2024-04-02")] alias coe_nat_nonpos_iff := natCast_nonpos_iff


/-- Like `Int.ediv_emod_unique`, but permitting negative `b`. -/
theorem ediv_emod_unique' {a b r q : Int} (h : b ≠ 0) :
    a / b = q ∧ a % b = r ↔ r + b * q = a ∧ 0 ≤ r ∧ r < |b| := by
  /-
    a b r q : Int
    h : Ne b 0
    ⊢ Iff (And (Eq (HDiv.hDiv a b) q) (Eq (HMod.hMod a b) r)) (And (Eq (HAdd.hAdd  …
  -/
  constructor
    /-
      case mp
      a b r q : Int
      h : Ne b 0
      ⊢ And (Eq (HDiv.hDiv a b) q) (Eq (HMod.hMod a b) r) → And (Eq (HAdd.hAdd r (HM …
    -/
  · intro ⟨rfl, rfl⟩
    /-
      case mp
      a b r q : Int
      h : Ne b 0
      ⊢ And (Eq (HAdd.hAdd (HMod.hMod a b) (HMul.hMul b (HDiv.hDiv a b))) a) (And (L …
    -/
    exact ⟨emod_add_ediv a b, emod_nonneg _ h, emod_lt _ h⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      a b r q : Int
      h : Ne b 0
      ⊢ And (Eq (HAdd.hAdd r (HMul.hMul b q)) a) (And (LE.le 0 r) (LT.lt r (abs b))) …
    -/
  · intro ⟨rfl, hz, hb⟩
    /-
      case mpr
      a b r q : Int
      h : Ne b 0
      hz : LE.le 0 r
      hb : LT.lt r (abs b)
      ⊢ And (Eq (HDiv.hDiv (HAdd.hAdd r (HMul.hMul b q)) b) q) (Eq (HMod.hMod (HAdd. …
    -/
    constructor
      /-
        case mpr.left
        a b r q : Int
        h : Ne b 0
        hz : LE.le 0 r
        hb : LT.lt r (abs b)
        ⊢ Eq (HDiv.hDiv (HAdd.hAdd r (HMul.hMul b q)) b) q
      -/
    · rw [Int.add_mul_ediv_left r q h, ediv_eq_zero_of_lt_abs hz hb]
      /-
        case mpr.left
        a b r q : Int
        h : Ne b 0
        hz : LE.le 0 r
        hb : LT.lt r (abs b)
        ⊢ Eq (HAdd.hAdd 0 q) q
      -/
      simp [Int.zero_add]
      /-
        🎉 no goals
      -/
      /-
        case mpr.right
        a b r q : Int
        h : Ne b 0
        hz : LE.le 0 r
        hb : LT.lt r (abs b)
        ⊢ Eq (HMod.hMod (HAdd.hAdd r (HMul.hMul b q)) b) r
      -/
    · rw [add_mul_emod_self_left, ← emod_abs, emod_eq_of_lt hz hb]
      /-
        🎉 no goals
      -/


