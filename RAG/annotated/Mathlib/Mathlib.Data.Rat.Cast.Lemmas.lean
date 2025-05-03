@[simp, norm_cast]
lemma cast_pow (p : ℚ) (n : ℕ) : ↑(p ^ n) = (p ^ n : α) := by
  rw [cast_def, cast_def, den_pow, num_pow, Nat.cast_pow, Int.cast_pow, div_eq_mul_inv, ← inv_pow,
    ← (Int.cast_commute _ _).mul_pow, ← div_eq_mul_inv]

-- Porting note: rewrote proof

@[simp]
theorem cast_inv_nat (n : ℕ) : ((n⁻¹ : ℚ) : α) = (n : α)⁻¹ := by
  /-
    α : Type u_1
    inst✝ : DivisionRing α
    n : Nat
    ⊢ Eq (↑(Inv.inv ↑n)) (Inv.inv ↑n)
  -/
  cases' n with n
    /-
      case zero
      α : Type u_1
      inst✝ : DivisionRing α
      ⊢ Eq (↑(Inv.inv ↑0)) (Inv.inv ↑0)
    -/
  · simp
    /-
      🎉 no goals
    -/
  rw [cast_def, inv_natCast_num, inv_natCast_den, if_neg n.succ_ne_zero,
    Int.sign_eq_one_of_pos (Int.ofNat_succ_pos n), Int.cast_one, one_div]

-- Porting note: proof got a lot easier - is this still the intended statement?

@[simp]
theorem cast_inv_int (n : ℤ) : ((n⁻¹ : ℚ) : α) = (n : α)⁻¹ := by
  /-
    α : Type u_1
    inst✝ : DivisionRing α
    n : Int
    ⊢ Eq (↑(Inv.inv ↑n)) (Inv.inv ↑n)
  -/
  cases' n with n n
    /-
      case ofNat
      α : Type u_1
      inst✝ : DivisionRing α
      n : Nat
      ⊢ Eq (↑(Inv.inv ↑(Int.ofNat n))) (Inv.inv ↑(Int.ofNat n))
    -/
  · simp [ofInt_eq_cast, cast_inv_nat]
    /-
      🎉 no goals
    -/
    /-
      case negSucc
      α : Type u_1
      inst✝ : DivisionRing α
      n : Nat
      ⊢ Eq (↑(Inv.inv ↑(Int.negSucc n))) (Inv.inv ↑(Int.negSucc n))
    -/
  · simp only [ofInt_eq_cast, Int.cast_negSucc, ← Nat.cast_succ, cast_neg, inv_neg, cast_inv_nat]
    /-
      🎉 no goals
    -/


@[simp, norm_cast]
theorem cast_nnratCast {K} [DivisionRing K] (q : ℚ≥0) :
    ((q : ℚ) : K) = (q : K) := by
  /-
    K : Type u_2
    inst✝ : DivisionRing K
    q : NNRat
    ⊢ Eq ↑↑q ↑q
  -/
  rw [Rat.cast_def, NNRat.cast_def, NNRat.cast_def]
  /-
    K : Type u_2
    inst✝ : DivisionRing K
    q : NNRat
    ⊢ Eq (HDiv.hDiv ↑(HDiv.hDiv ↑q.num ↑q.den).num ↑(HDiv.hDiv ↑q.num ↑q.den).den) …
  -/
  have hn := @num_div_eq_of_coprime q.num q.den ?hdp q.coprime_num_den
  /-
    K : Type u_2
    inst✝ : DivisionRing K
    q : NNRat
    hn : Eq (HDiv.hDiv ↑↑q.num ↑↑q.den).num ↑q.num
    ⊢ Eq (HDiv.hDiv ↑(HDiv.hDiv ↑q.num ↑q.den).num ↑(HDiv.hDiv ↑q.num ↑q.den).den) …
  -/
  on_goal 1 => have hd := @den_div_eq_of_coprime q.num q.den ?hdp q.coprime_num_den
  /-
    K : Type u_2
    inst✝ : DivisionRing K
    q : NNRat
    hn : Eq (HDiv.hDiv ↑↑q.num ↑↑q.den).num ↑q.num
    hd : Eq ↑(HDiv.hDiv ↑↑q.num ↑↑q.den).den ↑q.den
    ⊢ Eq (HDiv.hDiv ↑(HDiv.hDiv ↑q.num ↑q.den).num ↑(HDiv.hDiv ↑q.num ↑q.den).den) …
  -/
  case hdp => simpa only [Int.ofNat_pos] using q.den_pos
  /-
    K : Type u_2
    inst✝ : DivisionRing K
    q : NNRat
    hn : Eq (HDiv.hDiv ↑↑q.num ↑↑q.den).num ↑q.num
    hd : Eq ↑(HDiv.hDiv ↑↑q.num ↑↑q.den).den ↑q.den
    ⊢ Eq (HDiv.hDiv ↑(HDiv.hDiv ↑q.num ↑q.den).num ↑(HDiv.hDiv ↑q.num ↑q.den).den) …
  -/
  simp only [Int.cast_natCast, Nat.cast_inj] at hn hd
  /-
    K : Type u_2
    inst✝ : DivisionRing K
    q : NNRat
    hn : Eq (HDiv.hDiv ↑q.num ↑q.den).num ↑q.num
    hd : Eq (HDiv.hDiv ↑q.num ↑q.den).den q.den
    ⊢ Eq (HDiv.hDiv ↑(HDiv.hDiv ↑q.num ↑q.den).num ↑(HDiv.hDiv ↑q.num ↑q.den).den) …
  -/
  rw [hn, hd, Int.cast_natCast]
  /-
    🎉 no goals
  -/


/-- Casting a scientific literal via `ℚ` is the same as casting directly. -/
@[simp, norm_cast]
theorem cast_ofScientific {K} [DivisionRing K] (m : ℕ) (s : Bool) (e : ℕ) :
    (OfScientific.ofScientific m s e : ℚ) = (OfScientific.ofScientific m s e : K) := by
  /-
    K : Type u_2
    inst✝ : DivisionRing K
    m : Nat
    s : Bool
    e : Nat
    ⊢ Eq (↑(OfScientific.ofScientific m s e)) (OfScientific.ofScientific m s e)
  -/
  rw [← NNRat.cast_ofScientific (K := K), ← NNRat.cast_ofScientific, cast_nnratCast]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem cast_pow {K} [DivisionSemiring K] (q : ℚ≥0) (n : ℕ) :
    NNRat.cast (q ^ n) = (NNRat.cast q : K) ^ n := by
  rw [cast_def, cast_def, den_pow, num_pow, Nat.cast_pow, Nat.cast_pow, div_eq_mul_inv, ← inv_pow,
    ← (Nat.cast_commute _ _).mul_pow, ← div_eq_mul_inv]


theorem cast_zpow_of_ne_zero {K} [DivisionSemiring K] (q : ℚ≥0) (z : ℤ) (hq : (q.num : K) ≠ 0) :
    NNRat.cast (q ^ z) = (NNRat.cast q : K) ^ z := by
  /-
    K : Type u_1
    inst✝ : DivisionSemiring K
    q : NNRat
    z : Int
    hq : Ne (↑q.num) 0
    ⊢ Eq (↑(HPow.hPow q z)) (HPow.hPow (↑q) z)
  -/
  obtain ⟨n, rfl | rfl⟩ := z.eq_nat_or_neg
    /-
      case intro.inl
      K : Type u_1
      inst✝ : DivisionSemiring K
      q : NNRat
      hq : Ne (↑q.num) 0
      n : Nat
      ⊢ Eq (↑(HPow.hPow q ↑n)) (HPow.hPow ↑q ↑n)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case intro.inr
      K : Type u_1
      inst✝ : DivisionSemiring K
      q : NNRat
      hq : Ne (↑q.num) 0
      n : Nat
      ⊢ Eq (↑(HPow.hPow q (Neg.neg ↑n))) (HPow.hPow (↑q) (Neg.neg ↑n))
    -/
  · simp_rw [zpow_neg, zpow_natCast, ← inv_pow, NNRat.cast_pow]
    /-
      case intro.inr
      K : Type u_1
      inst✝ : DivisionSemiring K
      q : NNRat
      hq : Ne (↑q.num) 0
      n : Nat
      ⊢ Eq (HPow.hPow (↑(Inv.inv q)) n) (HPow.hPow (Inv.inv ↑q) n)
    -/
    congr
    /-
      case intro.inr.e_a
      K : Type u_1
      inst✝ : DivisionSemiring K
      q : NNRat
      hq : Ne (↑q.num) 0
      n : Nat
      ⊢ Eq (↑(Inv.inv q)) (Inv.inv ↑q)
    -/
    rw [cast_inv_of_ne_zero hq]
    /-
      🎉 no goals
    -/


open OfScientific in
theorem Nonneg.coe_ofScientific {K} [LinearOrderedField K] (m : ℕ) (s : Bool) (e : ℕ) :
    (ofScientific m s e : {x : K // 0 ≤ x}).val = ofScientific m s e := rfl


