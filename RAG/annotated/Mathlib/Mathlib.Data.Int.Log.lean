/-- The greatest power of `b` such that `b ^ log b r ≤ r`. -/
def log (b : ℕ) (r : R) : ℤ :=
  if 1 ≤ r then Nat.log b ⌊r⌋₊ else -Nat.clog b ⌈r⁻¹⌉₊


theorem log_of_one_le_right (b : ℕ) {r : R} (hr : 1 ≤ r) : log b r = Nat.log b ⌊r⌋₊ :=
  if_pos hr


theorem log_of_right_le_one (b : ℕ) {r : R} (hr : r ≤ 1) : log b r = -Nat.clog b ⌈r⁻¹⌉₊ := by
  /-
    R : Type u_1
    inst✝¹ : LinearOrderedSemifield R
    inst✝ : FloorSemiring R
    b : Nat
    r : R
    hr : LE.le r 1
    ⊢ Eq (Int.log b r) (Neg.neg ↑(Nat.clog b (Nat.ceil (Inv.inv r))))
  -/
  obtain rfl | hr := hr.eq_or_lt
  · rw [log, if_pos hr, inv_one, Nat.ceil_one, Nat.floor_one, Nat.log_one_right, Nat.clog_one_right,
      Int.ofNat_zero, neg_zero]
    /-
      case inr
      R : Type u_1
      inst✝¹ : LinearOrderedSemifield R
      inst✝ : FloorSemiring R
      b : Nat
      r : R
      hr✝ : LE.le r 1
      hr : LT.lt r 1
      ⊢ Eq (Int.log b r) (Neg.neg ↑(Nat.clog b (Nat.ceil (Inv.inv r))))
    -/
  · exact if_neg hr.not_le
    /-
      🎉 no goals
    -/


@[simp, norm_cast]
theorem log_natCast (b : ℕ) (n : ℕ) : log b (n : R) = Nat.log b n := by
  /-
    R : Type u_1
    inst✝¹ : LinearOrderedSemifield R
    inst✝ : FloorSemiring R
    b n : Nat
    ⊢ Eq (Int.log b ↑n) ↑(Nat.log b n)
  -/
  cases n
    /-
      case zero
      R : Type u_1
      inst✝¹ : LinearOrderedSemifield R
      inst✝ : FloorSemiring R
      b : Nat
      ⊢ Eq (Int.log b ↑0) ↑(Nat.log b 0)
    -/
  · simp [log_of_right_le_one]
    /-
      🎉 no goals
    -/
    /-
      case succ
      R : Type u_1
      inst✝¹ : LinearOrderedSemifield R
      inst✝ : FloorSemiring R
      b n✝ : Nat
      ⊢ Eq (Int.log b ↑(HAdd.hAdd n✝ 1)) ↑(Nat.log b (HAdd.hAdd n✝ 1))
    -/
  · rw [log_of_one_le_right, Nat.floor_natCast]
    /-
      case succ.hr
      R : Type u_1
      inst✝¹ : LinearOrderedSemifield R
      inst✝ : FloorSemiring R
      b n✝ : Nat
      ⊢ LE.le 1 ↑(HAdd.hAdd n✝ 1)
    -/
    simp
    /-
      🎉 no goals
    -/


@[simp]
theorem log_ofNat (b : ℕ) (n : ℕ) [n.AtLeastTwo] :
    log b (ofNat(n) : R) = Nat.log b ofNat(n) :=
  log_natCast b n


theorem log_of_left_le_one {b : ℕ} (hb : b ≤ 1) (r : R) : log b r = 0 := by
  /-
    R : Type u_1
    inst✝¹ : LinearOrderedSemifield R
    inst✝ : FloorSemiring R
    b : Nat
    hb : LE.le b 1
    r : R
    ⊢ Eq (Int.log b r) 0
  -/
  rcases le_total 1 r with h | h
    /-
      case inl
      R : Type u_1
      inst✝¹ : LinearOrderedSemifield R
      inst✝ : FloorSemiring R
      b : Nat
      hb : LE.le b 1
      r : R
      h : LE.le 1 r
      ⊢ Eq (Int.log b r) 0
    -/
  · rw [log_of_one_le_right _ h, Nat.log_of_left_le_one hb, Int.ofNat_zero]
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u_1
      inst✝¹ : LinearOrderedSemifield R
      inst✝ : FloorSemiring R
      b : Nat
      hb : LE.le b 1
      r : R
      h : LE.le r 1
      ⊢ Eq (Int.log b r) 0
    -/
  · rw [log_of_right_le_one _ h, Nat.clog_of_left_le_one hb, Int.ofNat_zero, neg_zero]
    /-
      🎉 no goals
    -/


theorem log_of_right_le_zero (b : ℕ) {r : R} (hr : r ≤ 0) : log b r = 0 := by
  rw [log_of_right_le_one _ (hr.trans zero_le_one),
    Nat.clog_of_right_le_one ((Nat.ceil_eq_zero.mpr <| inv_nonpos.2 hr).trans_le zero_le_one),
    Int.ofNat_zero, neg_zero]


theorem zpow_log_le_self {b : ℕ} {r : R} (hb : 1 < b) (hr : 0 < r) : (b : R) ^ log b r ≤ r := by
  /-
    R : Type u_1
    inst✝¹ : LinearOrderedSemifield R
    inst✝ : FloorSemiring R
    b : Nat
    r : R
    hb : LT.lt 1 b
    hr : LT.lt 0 r
    ⊢ LE.le (HPow.hPow (↑b) (Int.log b r)) r
  -/
  rcases le_total 1 r with hr1 | hr1
    /-
      case inl
      R : Type u_1
      inst✝¹ : LinearOrderedSemifield R
      inst✝ : FloorSemiring R
      b : Nat
      r : R
      hb : LT.lt 1 b
      hr : LT.lt 0 r
      hr1 : LE.le 1 r
      ⊢ LE.le (HPow.hPow (↑b) (Int.log b r)) r
    -/
  · rw [log_of_one_le_right _ hr1]
    /-
      case inl
      R : Type u_1
      inst✝¹ : LinearOrderedSemifield R
      inst✝ : FloorSemiring R
      b : Nat
      r : R
      hb : LT.lt 1 b
      hr : LT.lt 0 r
      hr1 : LE.le 1 r
      ⊢ LE.le (HPow.hPow ↑b ↑(Nat.log b (Nat.floor r))) r
    -/
    rw [zpow_natCast, ← Nat.cast_pow, ← Nat.le_floor_iff hr.le]
    /-
      case inl
      R : Type u_1
      inst✝¹ : LinearOrderedSemifield R
      inst✝ : FloorSemiring R
      b : Nat
      r : R
      hb : LT.lt 1 b
      hr : LT.lt 0 r
      hr1 : LE.le 1 r
      ⊢ LE.le (HPow.hPow b (Nat.log b (Nat.floor r))) (Nat.floor r)
    -/
    exact Nat.pow_log_le_self b (Nat.floor_pos.mpr hr1).ne'
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u_1
      inst✝¹ : LinearOrderedSemifield R
      inst✝ : FloorSemiring R
      b : Nat
      r : R
      hb : LT.lt 1 b
      hr : LT.lt 0 r
      hr1 : LE.le r 1
      ⊢ LE.le (HPow.hPow (↑b) (Int.log b r)) r
    -/
  · rw [log_of_right_le_one _ hr1, zpow_neg, zpow_natCast, ← Nat.cast_pow]
    /-
      case inr
      R : Type u_1
      inst✝¹ : LinearOrderedSemifield R
      inst✝ : FloorSemiring R
      b : Nat
      r : R
      hb : LT.lt 1 b
      hr : LT.lt 0 r
      hr1 : LE.le r 1
      ⊢ LE.le (Inv.inv ↑(HPow.hPow b (Nat.clog b (Nat.ceil (Inv.inv r))))) r
    -/
    exact inv_le_of_inv_le₀ hr (Nat.ceil_le.1 <| Nat.le_pow_clog hb _)
    /-
      🎉 no goals
    -/


theorem lt_zpow_succ_log_self {b : ℕ} (hb : 1 < b) (r : R) : r < (b : R) ^ (log b r + 1) := by
  /-
    R : Type u_1
    inst✝¹ : LinearOrderedSemifield R
    inst✝ : FloorSemiring R
    b : Nat
    hb : LT.lt 1 b
    r : R
    ⊢ LT.lt r (HPow.hPow (↑b) (HAdd.hAdd (Int.log b r) 1))
  -/
  rcases le_or_lt r 0 with hr | hr
    /-
      case inl
      R : Type u_1
      inst✝¹ : LinearOrderedSemifield R
      inst✝ : FloorSemiring R
      b : Nat
      hb : LT.lt 1 b
      r : R
      hr : LE.le r 0
      ⊢ LT.lt r (HPow.hPow (↑b) (HAdd.hAdd (Int.log b r) 1))
    -/
  · rw [log_of_right_le_zero _ hr, zero_add, zpow_one]
    /-
      case inl
      R : Type u_1
      inst✝¹ : LinearOrderedSemifield R
      inst✝ : FloorSemiring R
      b : Nat
      hb : LT.lt 1 b
      r : R
      hr : LE.le r 0
      ⊢ LT.lt r ↑b
    -/
    exact hr.trans_lt (zero_lt_one.trans_le <| mod_cast hb.le)
    /-
      🎉 no goals
    -/
  /-
    case inr
    R : Type u_1
    inst✝¹ : LinearOrderedSemifield R
    inst✝ : FloorSemiring R
    b : Nat
    hb : LT.lt 1 b
    r : R
    hr : LT.lt 0 r
    ⊢ LT.lt r (HPow.hPow (↑b) (HAdd.hAdd (Int.log b r) 1))
  -/
  rcases le_or_lt 1 r with hr1 | hr1
    /-
      case inr.inl
      R : Type u_1
      inst✝¹ : LinearOrderedSemifield R
      inst✝ : FloorSemiring R
      b : Nat
      hb : LT.lt 1 b
      r : R
      hr : LT.lt 0 r
      hr1 : LE.le 1 r
      ⊢ LT.lt r (HPow.hPow (↑b) (HAdd.hAdd (Int.log b r) 1))
    -/
  · rw [log_of_one_le_right _ hr1]
    /-
      case inr.inl
      R : Type u_1
      inst✝¹ : LinearOrderedSemifield R
      inst✝ : FloorSemiring R
      b : Nat
      hb : LT.lt 1 b
      r : R
      hr : LT.lt 0 r
      hr1 : LE.le 1 r
      ⊢ LT.lt r (HPow.hPow (↑b) (HAdd.hAdd (↑(Nat.log b (Nat.floor r))) 1))
    -/
    rw [Int.ofNat_add_one_out, zpow_natCast, ← Nat.cast_pow]
    /-
      case inr.inl
      R : Type u_1
      inst✝¹ : LinearOrderedSemifield R
      inst✝ : FloorSemiring R
      b : Nat
      hb : LT.lt 1 b
      r : R
      hr : LT.lt 0 r
      hr1 : LE.le 1 r
      ⊢ LT.lt r ↑(HPow.hPow b (Nat.log b (Nat.floor r)).succ)
    -/
    apply Nat.lt_of_floor_lt
    /-
      case inr.inl.h
      R : Type u_1
      inst✝¹ : LinearOrderedSemifield R
      inst✝ : FloorSemiring R
      b : Nat
      hb : LT.lt 1 b
      r : R
      hr : LT.lt 0 r
      hr1 : LE.le 1 r
      ⊢ LT.lt (Nat.floor r) (HPow.hPow b (Nat.log b (Nat.floor r)).succ)
    -/
    exact Nat.lt_pow_succ_log_self hb _
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      R : Type u_1
      inst✝¹ : LinearOrderedSemifield R
      inst✝ : FloorSemiring R
      b : Nat
      hb : LT.lt 1 b
      r : R
      hr : LT.lt 0 r
      hr1 : LT.lt r 1
      ⊢ LT.lt r (HPow.hPow (↑b) (HAdd.hAdd (Int.log b r) 1))
    -/
  · rw [log_of_right_le_one _ hr1.le]
    /-
      case inr.inr
      R : Type u_1
      inst✝¹ : LinearOrderedSemifield R
      inst✝ : FloorSemiring R
      b : Nat
      hb : LT.lt 1 b
      r : R
      hr : LT.lt 0 r
      hr1 : LT.lt r 1
      ⊢ LT.lt r (HPow.hPow (↑b) (HAdd.hAdd (Neg.neg ↑(Nat.clog b (Nat.ceil (Inv.inv  …
    -/
    have hcri : 1 < r⁻¹ := (one_lt_inv₀ hr).2 hr1
    have : 1 ≤ Nat.clog b ⌈r⁻¹⌉₊ :=
      Nat.succ_le_of_lt (Nat.clog_pos hb <| Nat.one_lt_cast.1 <| hcri.trans_le (Nat.le_ceil _))
    rw [neg_add_eq_sub, ← neg_sub, ← Int.ofNat_one, ← Int.ofNat_sub this, zpow_neg, zpow_natCast,
      lt_inv_comm₀ hr (pow_pos (Nat.cast_pos.mpr <| zero_lt_one.trans hb) _), ← Nat.cast_pow]
    /-
      case inr.inr
      R : Type u_1
      inst✝¹ : LinearOrderedSemifield R
      inst✝ : FloorSemiring R
      b : Nat
      hb : LT.lt 1 b
      r : R
      hr : LT.lt 0 r
      hr1 : LT.lt r 1
      hcri : LT.lt 1 (Inv.inv r)
      this : LE.le 1 (Nat.clog b (Nat.ceil (Inv.inv r)))
      ⊢ LT.lt (↑(HPow.hPow b (HSub.hSub (Nat.clog b (Nat.ceil (Inv.inv r))) 1))) (In …
    -/
    refine Nat.lt_ceil.1 ?_
    /-
      case inr.inr
      R : Type u_1
      inst✝¹ : LinearOrderedSemifield R
      inst✝ : FloorSemiring R
      b : Nat
      hb : LT.lt 1 b
      r : R
      hr : LT.lt 0 r
      hr1 : LT.lt r 1
      hcri : LT.lt 1 (Inv.inv r)
      this : LE.le 1 (Nat.clog b (Nat.ceil (Inv.inv r)))
      ⊢ LT.lt (HPow.hPow b (HSub.hSub (Nat.clog b (Nat.ceil (Inv.inv r))) 1)) (Nat.c …
    -/
    exact Nat.pow_pred_clog_lt_self hb <| Nat.one_lt_cast.1 <| hcri.trans_le <| Nat.le_ceil _
    /-
      🎉 no goals
    -/


@[simp]
theorem log_zero_right (b : ℕ) : log b (0 : R) = 0 :=
  log_of_right_le_zero b le_rfl


@[simp]
theorem log_one_right (b : ℕ) : log b (1 : R) = 0 := by
  /-
    R : Type u_1
    inst✝¹ : LinearOrderedSemifield R
    inst✝ : FloorSemiring R
    b : Nat
    ⊢ Eq (Int.log b 1) 0
  -/
  rw [log_of_one_le_right _ le_rfl, Nat.floor_one, Nat.log_one_right, Int.ofNat_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem log_zero_left (r : R) : log 0 r = 0 := by
  /-
    R : Type u_1
    inst✝¹ : LinearOrderedSemifield R
    inst✝ : FloorSemiring R
    r : R
    ⊢ Eq (Int.log 0 r) 0
  -/
  simp only [log, Nat.log_zero_left, Nat.cast_zero, Nat.clog_zero_left, neg_zero, ite_self]
  /-
    🎉 no goals
  -/


@[simp]
theorem log_one_left (r : R) : log 1 r = 0 := by
  /-
    R : Type u_1
    inst✝¹ : LinearOrderedSemifield R
    inst✝ : FloorSemiring R
    r : R
    ⊢ Eq (Int.log 1 r) 0
  -/
  by_cases hr : 1 ≤ r
    /-
      case pos
      R : Type u_1
      inst✝¹ : LinearOrderedSemifield R
      inst✝ : FloorSemiring R
      r : R
      hr : LE.le 1 r
      ⊢ Eq (Int.log 1 r) 0
    -/
  · simp_all only [log, ↓reduceIte, Nat.log_one_left, Nat.cast_zero]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝¹ : LinearOrderedSemifield R
      inst✝ : FloorSemiring R
      r : R
      hr : Not (LE.le 1 r)
      ⊢ Eq (Int.log 1 r) 0
    -/
  · simp only [log, Nat.log_one_left, Nat.cast_zero, Nat.clog_one_left, neg_zero, ite_self]
    /-
      🎉 no goals
    -/

-- Porting note: needed to replace b ^ z with (b : R) ^ z in the below

theorem log_zpow {b : ℕ} (hb : 1 < b) (z : ℤ) : log b ((b : R) ^ z : R) = z := by
  /-
    R : Type u_1
    inst✝¹ : LinearOrderedSemifield R
    inst✝ : FloorSemiring R
    b : Nat
    hb : LT.lt 1 b
    z : Int
    ⊢ Eq (Int.log b (HPow.hPow (↑b) z)) z
  -/
  obtain ⟨n, rfl | rfl⟩ := Int.eq_nat_or_neg z
  · rw [log_of_one_le_right _ (one_le_zpow₀ (mod_cast hb.le) <| Int.natCast_nonneg _), zpow_natCast,
      ← Nat.cast_pow, Nat.floor_natCast, Nat.log_pow hb]
  · rw [log_of_right_le_one _ (zpow_le_one_of_nonpos₀ (mod_cast hb.le) <|
      neg_nonpos.2 (Int.natCast_nonneg _)),
      zpow_neg, inv_inv, zpow_natCast, ← Nat.cast_pow, Nat.ceil_natCast, Nat.clog_pow _ _ hb]


@[mono]
theorem log_mono_right {b : ℕ} {r₁ r₂ : R} (h₀ : 0 < r₁) (h : r₁ ≤ r₂) : log b r₁ ≤ log b r₂ := by
  /-
    R : Type u_1
    inst✝¹ : LinearOrderedSemifield R
    inst✝ : FloorSemiring R
    b : Nat
    r₁ r₂ : R
    h₀ : LT.lt 0 r₁
    h : LE.le r₁ r₂
    ⊢ LE.le (Int.log b r₁) (Int.log b r₂)
  -/
  rcases le_total r₁ 1 with h₁ | h₁ <;> rcases le_total r₂ 1 with h₂ | h₂
    /-
      case inl.inl
      R : Type u_1
      inst✝¹ : LinearOrderedSemifield R
      inst✝ : FloorSemiring R
      b : Nat
      r₁ r₂ : R
      h₀ : LT.lt 0 r₁
      h : LE.le r₁ r₂
      h₁ : LE.le r₁ 1
      h₂ : LE.le r₂ 1
      ⊢ LE.le (Int.log b r₁) (Int.log b r₂)
    -/
  · rw [log_of_right_le_one _ h₁, log_of_right_le_one _ h₂, neg_le_neg_iff, Int.ofNat_le]
    /-
      case inl.inl
      R : Type u_1
      inst✝¹ : LinearOrderedSemifield R
      inst✝ : FloorSemiring R
      b : Nat
      r₁ r₂ : R
      h₀ : LT.lt 0 r₁
      h : LE.le r₁ r₂
      h₁ : LE.le r₁ 1
      h₂ : LE.le r₂ 1
      ⊢ LE.le (Nat.clog b (Nat.ceil (Inv.inv r₂))) (Nat.clog b (Nat.ceil (Inv.inv r₁ …
    -/
    exact Nat.clog_mono_right _ (Nat.ceil_mono <| inv_anti₀ h₀ h)
    /-
      🎉 no goals
    -/
    /-
      case inl.inr
      R : Type u_1
      inst✝¹ : LinearOrderedSemifield R
      inst✝ : FloorSemiring R
      b : Nat
      r₁ r₂ : R
      h₀ : LT.lt 0 r₁
      h : LE.le r₁ r₂
      h₁ : LE.le r₁ 1
      h₂ : LE.le 1 r₂
      ⊢ LE.le (Int.log b r₁) (Int.log b r₂)
    -/
  · rw [log_of_right_le_one _ h₁, log_of_one_le_right _ h₂]
    /-
      case inl.inr
      R : Type u_1
      inst✝¹ : LinearOrderedSemifield R
      inst✝ : FloorSemiring R
      b : Nat
      r₁ r₂ : R
      h₀ : LT.lt 0 r₁
      h : LE.le r₁ r₂
      h₁ : LE.le r₁ 1
      h₂ : LE.le 1 r₂
      ⊢ LE.le (Neg.neg ↑(Nat.clog b (Nat.ceil (Inv.inv r₁)))) ↑(Nat.log b (Nat.floor …
    -/
    exact (neg_nonpos.mpr (Int.natCast_nonneg _)).trans (Int.natCast_nonneg _)
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      R : Type u_1
      inst✝¹ : LinearOrderedSemifield R
      inst✝ : FloorSemiring R
      b : Nat
      r₁ r₂ : R
      h₀ : LT.lt 0 r₁
      h : LE.le r₁ r₂
      h₁ : LE.le 1 r₁
      h₂ : LE.le r₂ 1
      ⊢ LE.le (Int.log b r₁) (Int.log b r₂)
    -/
  · obtain rfl := le_antisymm h (h₂.trans h₁)
    /-
      case inr.inl
      R : Type u_1
      inst✝¹ : LinearOrderedSemifield R
      inst✝ : FloorSemiring R
      b : Nat
      r₁ : R
      h₀ : LT.lt 0 r₁
      h₁ : LE.le 1 r₁
      h : LE.le r₁ r₁
      h₂ : LE.le r₁ 1
      ⊢ LE.le (Int.log b r₁) (Int.log b r₁)
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      R : Type u_1
      inst✝¹ : LinearOrderedSemifield R
      inst✝ : FloorSemiring R
      b : Nat
      r₁ r₂ : R
      h₀ : LT.lt 0 r₁
      h : LE.le r₁ r₂
      h₁ : LE.le 1 r₁
      h₂ : LE.le 1 r₂
      ⊢ LE.le (Int.log b r₁) (Int.log b r₂)
    -/
  · rw [log_of_one_le_right _ h₁, log_of_one_le_right _ h₂, Int.ofNat_le]
    /-
      case inr.inr
      R : Type u_1
      inst✝¹ : LinearOrderedSemifield R
      inst✝ : FloorSemiring R
      b : Nat
      r₁ r₂ : R
      h₀ : LT.lt 0 r₁
      h : LE.le r₁ r₂
      h₁ : LE.le 1 r₁
      h₂ : LE.le 1 r₂
      ⊢ LE.le (Nat.log b (Nat.floor r₁)) (Nat.log b (Nat.floor r₂))
    -/
    exact Nat.log_mono_right (Nat.floor_mono h)
    /-
      🎉 no goals
    -/


/-- Over suitable subtypes, `zpow` and `Int.log` form a galois coinsertion -/
def zpowLogGi {b : ℕ} (hb : 1 < b) :
    GaloisCoinsertion
      (fun z : ℤ =>
        Subtype.mk ((b : R) ^ z) <| zpow_pos (mod_cast zero_lt_one.trans hb) z)
      fun r : Set.Ioi (0 : R) => Int.log b (r : R) :=
  GaloisCoinsertion.monotoneIntro (fun r₁ _ => log_mono_right r₁.2)
    (fun _ _ hz => Subtype.coe_le_coe.mp <| (zpow_right_strictMono₀ <| mod_cast hb).monotone hz)
    (fun r => Subtype.coe_le_coe.mp <| zpow_log_le_self hb r.2) fun _ => log_zpow (R := R) hb _


/-- `zpow b` and `Int.log b` (almost) form a Galois connection. -/
theorem lt_zpow_iff_log_lt {b : ℕ} (hb : 1 < b) {x : ℤ} {r : R} (hr : 0 < r) :
    r < (b : R) ^ x ↔ log b r < x :=
  @GaloisConnection.lt_iff_lt _ _ _ _ _ _ (zpowLogGi R hb).gc x ⟨r, hr⟩


/-- `zpow b` and `Int.log b` (almost) form a Galois connection. -/
theorem zpow_le_iff_le_log {b : ℕ} (hb : 1 < b) {x : ℤ} {r : R} (hr : 0 < r) :
    (b : R) ^ x ≤ r ↔ x ≤ log b r :=
  @GaloisConnection.le_iff_le _ _ _ _ _ _ (zpowLogGi R hb).gc x ⟨r, hr⟩


/-- The least power of `b` such that `r ≤ b ^ log b r`. -/
def clog (b : ℕ) (r : R) : ℤ :=
  if 1 ≤ r then Nat.clog b ⌈r⌉₊ else -Nat.log b ⌊r⁻¹⌋₊


theorem clog_of_one_le_right (b : ℕ) {r : R} (hr : 1 ≤ r) : clog b r = Nat.clog b ⌈r⌉₊ :=
  if_pos hr


theorem clog_of_right_le_one (b : ℕ) {r : R} (hr : r ≤ 1) : clog b r = -Nat.log b ⌊r⁻¹⌋₊ := by
  /-
    R : Type u_1
    inst✝¹ : LinearOrderedSemifield R
    inst✝ : FloorSemiring R
    b : Nat
    r : R
    hr : LE.le r 1
    ⊢ Eq (Int.clog b r) (Neg.neg ↑(Nat.log b (Nat.floor (Inv.inv r))))
  -/
  obtain rfl | hr := hr.eq_or_lt
  · rw [clog, if_pos hr, inv_one, Nat.ceil_one, Nat.floor_one, Nat.log_one_right,
      Nat.clog_one_right, Int.ofNat_zero, neg_zero]
    /-
      case inr
      R : Type u_1
      inst✝¹ : LinearOrderedSemifield R
      inst✝ : FloorSemiring R
      b : Nat
      r : R
      hr✝ : LE.le r 1
      hr : LT.lt r 1
      ⊢ Eq (Int.clog b r) (Neg.neg ↑(Nat.log b (Nat.floor (Inv.inv r))))
    -/
  · exact if_neg hr.not_le
    /-
      🎉 no goals
    -/


theorem clog_of_right_le_zero (b : ℕ) {r : R} (hr : r ≤ 0) : clog b r = 0 := by
  rw [clog, if_neg (hr.trans_lt zero_lt_one).not_le, neg_eq_zero, Int.natCast_eq_zero,
    Nat.log_eq_zero_iff]
  /-
    R : Type u_1
    inst✝¹ : LinearOrderedSemifield R
    inst✝ : FloorSemiring R
    b : Nat
    r : R
    hr : LE.le r 0
    ⊢ Or (LT.lt (Nat.floor (Inv.inv r)) b) (LE.le b 1)
  -/
  rcases le_or_lt b 1 with hb | hb
    /-
      case inl
      R : Type u_1
      inst✝¹ : LinearOrderedSemifield R
      inst✝ : FloorSemiring R
      b : Nat
      r : R
      hr : LE.le r 0
      hb : LE.le b 1
      ⊢ Or (LT.lt (Nat.floor (Inv.inv r)) b) (LE.le b 1)
    -/
  · exact Or.inr hb
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u_1
      inst✝¹ : LinearOrderedSemifield R
      inst✝ : FloorSemiring R
      b : Nat
      r : R
      hr : LE.le r 0
      hb : LT.lt 1 b
      ⊢ Or (LT.lt (Nat.floor (Inv.inv r)) b) (LE.le b 1)
    -/
  · refine Or.inl (lt_of_le_of_lt ?_ hb)
    /-
      case inr
      R : Type u_1
      inst✝¹ : LinearOrderedSemifield R
      inst✝ : FloorSemiring R
      b : Nat
      r : R
      hr : LE.le r 0
      hb : LT.lt 1 b
      ⊢ LE.le (Nat.floor (Inv.inv r)) 1
    -/
    exact Nat.floor_le_one_of_le_one ((inv_nonpos.2 hr).trans zero_le_one)
    /-
      🎉 no goals
    -/


@[simp]
theorem clog_inv (b : ℕ) (r : R) : clog b r⁻¹ = -log b r := by
  /-
    R : Type u_1
    inst✝¹ : LinearOrderedSemifield R
    inst✝ : FloorSemiring R
    b : Nat
    r : R
    ⊢ Eq (Int.clog b (Inv.inv r)) (Neg.neg (Int.log b r))
  -/
  cases' lt_or_le 0 r with hrp hrp
    /-
      case inl
      R : Type u_1
      inst✝¹ : LinearOrderedSemifield R
      inst✝ : FloorSemiring R
      b : Nat
      r : R
      hrp : LT.lt 0 r
      ⊢ Eq (Int.clog b (Inv.inv r)) (Neg.neg (Int.log b r))
    -/
  · obtain hr | hr := le_total 1 r
      /-
        case inl.inl
        R : Type u_1
        inst✝¹ : LinearOrderedSemifield R
        inst✝ : FloorSemiring R
        b : Nat
        r : R
        hrp : LT.lt 0 r
        hr : LE.le 1 r
        ⊢ Eq (Int.clog b (Inv.inv r)) (Neg.neg (Int.log b r))
      -/
    · rw [clog_of_right_le_one _ (inv_le_one_of_one_le₀ hr), log_of_one_le_right _ hr, inv_inv]
      /-
        🎉 no goals
      -/
      /-
        case inl.inr
        R : Type u_1
        inst✝¹ : LinearOrderedSemifield R
        inst✝ : FloorSemiring R
        b : Nat
        r : R
        hrp : LT.lt 0 r
        hr : LE.le r 1
        ⊢ Eq (Int.clog b (Inv.inv r)) (Neg.neg (Int.log b r))
      -/
    · rw [clog_of_one_le_right _ ((one_le_inv₀ hrp).2 hr), log_of_right_le_one _ hr, neg_neg]
      /-
        🎉 no goals
      -/
    /-
      case inr
      R : Type u_1
      inst✝¹ : LinearOrderedSemifield R
      inst✝ : FloorSemiring R
      b : Nat
      r : R
      hrp : LE.le r 0
      ⊢ Eq (Int.clog b (Inv.inv r)) (Neg.neg (Int.log b r))
    -/
  · rw [clog_of_right_le_zero _ (inv_nonpos.mpr hrp), log_of_right_le_zero _ hrp, neg_zero]
    /-
      🎉 no goals
    -/


@[simp]
theorem log_inv (b : ℕ) (r : R) : log b r⁻¹ = -clog b r := by
  /-
    R : Type u_1
    inst✝¹ : LinearOrderedSemifield R
    inst✝ : FloorSemiring R
    b : Nat
    r : R
    ⊢ Eq (Int.log b (Inv.inv r)) (Neg.neg (Int.clog b r))
  -/
  rw [← inv_inv r, clog_inv, neg_neg, inv_inv]
  /-
    🎉 no goals
  -/

-- note this is useful for writing in reverse

                                                                          /-
                                                                            R : Type u_1
                                                                            inst✝¹ : LinearOrderedSemifield R
                                                                            inst✝ : FloorSemiring R
                                                                            b : Nat
                                                                            r : R
                                                                            ⊢ Eq (Neg.neg (Int.log b (Inv.inv r))) (Int.clog b r)
                                                                          -/
theorem neg_log_inv_eq_clog (b : ℕ) (r : R) : -log b r⁻¹ = clog b r := by rw [log_inv, neg_neg]
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


                                                                          /-
                                                                            R : Type u_1
                                                                            inst✝¹ : LinearOrderedSemifield R
                                                                            inst✝ : FloorSemiring R
                                                                            b : Nat
                                                                            r : R
                                                                            ⊢ Eq (Neg.neg (Int.clog b (Inv.inv r))) (Int.log b r)
                                                                          -/
theorem neg_clog_inv_eq_log (b : ℕ) (r : R) : -clog b r⁻¹ = log b r := by rw [clog_inv, neg_neg]
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


@[simp, norm_cast]
theorem clog_natCast (b : ℕ) (n : ℕ) : clog b (n : R) = Nat.clog b n := by
  /-
    R : Type u_1
    inst✝¹ : LinearOrderedSemifield R
    inst✝ : FloorSemiring R
    b n : Nat
    ⊢ Eq (Int.clog b ↑n) ↑(Nat.clog b n)
  -/
  cases' n with n
    /-
      case zero
      R : Type u_1
      inst✝¹ : LinearOrderedSemifield R
      inst✝ : FloorSemiring R
      b : Nat
      ⊢ Eq (Int.clog b ↑0) ↑(Nat.clog b 0)
    -/
  · simp [clog_of_right_le_one]
    /-
      🎉 no goals
    -/
    /-
      case succ
      R : Type u_1
      inst✝¹ : LinearOrderedSemifield R
      inst✝ : FloorSemiring R
      b n : Nat
      ⊢ Eq (Int.clog b ↑(HAdd.hAdd n 1)) ↑(Nat.clog b (HAdd.hAdd n 1))
    -/
                                                                              /-
                                                                                🎉 no goals
                                                                              -/
  · rw [clog_of_one_le_right, (Nat.ceil_eq_iff (Nat.succ_ne_zero n)).mpr] <;> simp
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


@[simp]
theorem clog_ofNat (b : ℕ) (n : ℕ) [n.AtLeastTwo] :
    clog b (ofNat(n) : R) = Nat.clog b ofNat(n) :=
  clog_natCast b n


theorem clog_of_left_le_one {b : ℕ} (hb : b ≤ 1) (r : R) : clog b r = 0 := by
  /-
    R : Type u_1
    inst✝¹ : LinearOrderedSemifield R
    inst✝ : FloorSemiring R
    b : Nat
    hb : LE.le b 1
    r : R
    ⊢ Eq (Int.clog b r) 0
  -/
  rw [← neg_log_inv_eq_clog, log_of_left_le_one hb, neg_zero]
  /-
    🎉 no goals
  -/


theorem self_le_zpow_clog {b : ℕ} (hb : 1 < b) (r : R) : r ≤ (b : R) ^ clog b r := by
  /-
    R : Type u_1
    inst✝¹ : LinearOrderedSemifield R
    inst✝ : FloorSemiring R
    b : Nat
    hb : LT.lt 1 b
    r : R
    ⊢ LE.le r (HPow.hPow (↑b) (Int.clog b r))
  -/
  rcases le_or_lt r 0 with hr | hr
    /-
      case inl
      R : Type u_1
      inst✝¹ : LinearOrderedSemifield R
      inst✝ : FloorSemiring R
      b : Nat
      hb : LT.lt 1 b
      r : R
      hr : LE.le r 0
      ⊢ LE.le r (HPow.hPow (↑b) (Int.clog b r))
    -/
  · rw [clog_of_right_le_zero _ hr, zpow_zero]
    /-
      case inl
      R : Type u_1
      inst✝¹ : LinearOrderedSemifield R
      inst✝ : FloorSemiring R
      b : Nat
      hb : LT.lt 1 b
      r : R
      hr : LE.le r 0
      ⊢ LE.le r 1
    -/
    exact hr.trans zero_le_one
    /-
      🎉 no goals
    -/
  /-
    case inr
    R : Type u_1
    inst✝¹ : LinearOrderedSemifield R
    inst✝ : FloorSemiring R
    b : Nat
    hb : LT.lt 1 b
    r : R
    hr : LT.lt 0 r
    ⊢ LE.le r (HPow.hPow (↑b) (Int.clog b r))
  -/
  rw [← neg_log_inv_eq_clog, zpow_neg, le_inv_comm₀ hr (zpow_pos ..)]
    /-
      case inr
      R : Type u_1
      inst✝¹ : LinearOrderedSemifield R
      inst✝ : FloorSemiring R
      b : Nat
      hb : LT.lt 1 b
      r : R
      hr : LT.lt 0 r
      ⊢ LE.le (HPow.hPow (↑b) (Int.log b (Inv.inv r))) (Inv.inv r)
    -/
  · exact zpow_log_le_self hb (inv_pos.mpr hr)
    /-
      🎉 no goals
    -/
    /-
      R : Type u_1
      inst✝¹ : LinearOrderedSemifield R
      inst✝ : FloorSemiring R
      b : Nat
      hb : LT.lt 1 b
      r : R
      hr : LT.lt 0 r
      ⊢ LT.lt 0 ↑b
    -/
  · exact Nat.cast_pos.mpr (zero_le_one.trans_lt hb)
    /-
      🎉 no goals
    -/


theorem zpow_pred_clog_lt_self {b : ℕ} {r : R} (hb : 1 < b) (hr : 0 < r) :
    (b : R) ^ (clog b r - 1) < r := by
  /-
    R : Type u_1
    inst✝¹ : LinearOrderedSemifield R
    inst✝ : FloorSemiring R
    b : Nat
    r : R
    hb : LT.lt 1 b
    hr : LT.lt 0 r
    ⊢ LT.lt (HPow.hPow (↑b) (HSub.hSub (Int.clog b r) 1)) r
  -/
  rw [← neg_log_inv_eq_clog, ← neg_add', zpow_neg, inv_lt_comm₀ _ hr]
    /-
      R : Type u_1
      inst✝¹ : LinearOrderedSemifield R
      inst✝ : FloorSemiring R
      b : Nat
      r : R
      hb : LT.lt 1 b
      hr : LT.lt 0 r
      ⊢ LT.lt (Inv.inv r) (HPow.hPow (↑b) (HAdd.hAdd (Int.log b (Inv.inv r)) 1))
    -/
  · exact lt_zpow_succ_log_self hb _
    /-
      🎉 no goals
    -/
    /-
      R : Type u_1
      inst✝¹ : LinearOrderedSemifield R
      inst✝ : FloorSemiring R
      b : Nat
      r : R
      hb : LT.lt 1 b
      hr : LT.lt 0 r
      ⊢ LT.lt 0 (HPow.hPow (↑b) (HAdd.hAdd (Int.log b (Inv.inv r)) 1))
    -/
  · exact zpow_pos (Nat.cast_pos.mpr <| zero_le_one.trans_lt hb) _
    /-
      🎉 no goals
    -/


@[simp]
theorem clog_zero_right (b : ℕ) : clog b (0 : R) = 0 :=
  clog_of_right_le_zero _ le_rfl


@[simp]
theorem clog_one_right (b : ℕ) : clog b (1 : R) = 0 := by
  /-
    R : Type u_1
    inst✝¹ : LinearOrderedSemifield R
    inst✝ : FloorSemiring R
    b : Nat
    ⊢ Eq (Int.clog b 1) 0
  -/
  rw [clog_of_one_le_right _ le_rfl, Nat.ceil_one, Nat.clog_one_right, Int.ofNat_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem clog_zero_left (r : R) : clog 0 r = 0 := by
  /-
    R : Type u_1
    inst✝¹ : LinearOrderedSemifield R
    inst✝ : FloorSemiring R
    r : R
    ⊢ Eq (Int.clog 0 r) 0
  -/
  by_cases hr : 1 ≤ r
    /-
      case pos
      R : Type u_1
      inst✝¹ : LinearOrderedSemifield R
      inst✝ : FloorSemiring R
      r : R
      hr : LE.le 1 r
      ⊢ Eq (Int.clog 0 r) 0
    -/
  · simp only [clog, Nat.clog_zero_left, Nat.cast_zero, Nat.log_zero_left, neg_zero, ite_self]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝¹ : LinearOrderedSemifield R
      inst✝ : FloorSemiring R
      r : R
      hr : Not (LE.le 1 r)
      ⊢ Eq (Int.clog 0 r) 0
    -/
  · simp only [clog, hr, ite_cond_eq_false, Nat.log_zero_left, Nat.cast_zero, neg_zero]
    /-
      🎉 no goals
    -/


@[simp]
theorem clog_one_left (r : R) : clog 1 r = 0 := by
  /-
    R : Type u_1
    inst✝¹ : LinearOrderedSemifield R
    inst✝ : FloorSemiring R
    r : R
    ⊢ Eq (Int.clog 1 r) 0
  -/
  simp only [clog, Nat.log_one_left, Nat.cast_zero, Nat.clog_one_left, neg_zero, ite_self]
  /-
    🎉 no goals
  -/

-- Porting note: needed to replace b ^ z with (b : R) ^ z in the below

theorem clog_zpow {b : ℕ} (hb : 1 < b) (z : ℤ) : clog b ((b : R) ^ z : R) = z := by
  /-
    R : Type u_1
    inst✝¹ : LinearOrderedSemifield R
    inst✝ : FloorSemiring R
    b : Nat
    hb : LT.lt 1 b
    z : Int
    ⊢ Eq (Int.clog b (HPow.hPow (↑b) z)) z
  -/
  rw [← neg_log_inv_eq_clog, ← zpow_neg, log_zpow hb, neg_neg]
  /-
    🎉 no goals
  -/


@[mono]
theorem clog_mono_right {b : ℕ} {r₁ r₂ : R} (h₀ : 0 < r₁) (h : r₁ ≤ r₂) :
    clog b r₁ ≤ clog b r₂ := by
  /-
    R : Type u_1
    inst✝¹ : LinearOrderedSemifield R
    inst✝ : FloorSemiring R
    b : Nat
    r₁ r₂ : R
    h₀ : LT.lt 0 r₁
    h : LE.le r₁ r₂
    ⊢ LE.le (Int.clog b r₁) (Int.clog b r₂)
  -/
  rw [← neg_log_inv_eq_clog, ← neg_log_inv_eq_clog, neg_le_neg_iff]
  /-
    R : Type u_1
    inst✝¹ : LinearOrderedSemifield R
    inst✝ : FloorSemiring R
    b : Nat
    r₁ r₂ : R
    h₀ : LT.lt 0 r₁
    h : LE.le r₁ r₂
    ⊢ LE.le (Int.log b (Inv.inv r₂)) (Int.log b (Inv.inv r₁))
  -/
  exact log_mono_right (inv_pos.mpr <| h₀.trans_le h) (inv_anti₀ h₀ h)
  /-
    🎉 no goals
  -/


/-- Over suitable subtypes, `Int.clog` and `zpow` form a galois insertion -/
def clogZPowGi {b : ℕ} (hb : 1 < b) :
    GaloisInsertion (fun r : Set.Ioi (0 : R) => Int.clog b (r : R)) fun z : ℤ =>
      ⟨(b : R) ^ z, zpow_pos (mod_cast zero_lt_one.trans hb) z⟩ :=
  GaloisInsertion.monotoneIntro
    (fun _ _ hz => Subtype.coe_le_coe.mp <| (zpow_right_strictMono₀ <| mod_cast hb).monotone hz)
    (fun r₁ _ => clog_mono_right r₁.2)
    (fun _ => Subtype.coe_le_coe.mp <| self_le_zpow_clog hb _) fun _ => clog_zpow (R := R) hb _


/-- `Int.clog b` and `zpow b` (almost) form a Galois connection. -/
theorem zpow_lt_iff_lt_clog {b : ℕ} (hb : 1 < b) {x : ℤ} {r : R} (hr : 0 < r) :
    (b : R) ^ x < r ↔ x < clog b r :=
  (@GaloisConnection.lt_iff_lt _ _ _ _ _ _ (clogZPowGi R hb).gc ⟨r, hr⟩ x).symm


/-- `Int.clog b` and `zpow b` (almost) form a Galois connection. -/
theorem le_zpow_iff_clog_le {b : ℕ} (hb : 1 < b) {x : ℤ} {r : R} (hr : 0 < r) :
    r ≤ (b : R) ^ x ↔ clog b r ≤ x :=
  (@GaloisConnection.le_iff_le _ _ _ _ _ _ (clogZPowGi R hb).gc ⟨r, hr⟩ x).symm


