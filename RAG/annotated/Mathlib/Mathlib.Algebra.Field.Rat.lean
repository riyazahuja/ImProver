instance instField : Field ℚ where
  __ := commRing
  __ := commGroupWithZero
  nnqsmul := _
  nnqsmul_def := fun _ _ => rfl
  qsmul := _
  qsmul_def := fun _ _ => rfl
  nnratCast_def q := by
    /-
      q : NNRat
      ⊢ Eq (↑q) (HDiv.hDiv ↑q.num ↑q.den)
    -/
    rw [← NNRat.den_coe, ← Int.cast_natCast q.num, ← NNRat.num_coe]; exact(num_div_den _).symm
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
  ratCast_def _ := (num_div_den _).symm


instance instDivisionRing : DivisionRing ℚ := inferInstance


protected lemma inv_nonneg {a : ℚ} (ha : 0 ≤ a) : 0 ≤ a⁻¹ := by
  /-
    a : Rat
    ha : LE.le 0 a
    ⊢ LE.le 0 (Inv.inv a)
  -/
  rw [inv_def']
  /-
    a : Rat
    ha : LE.le 0 a
    ⊢ LE.le 0 (Rat.divInt (↑a.den) a.num)
  -/
  exact divInt_nonneg (Int.ofNat_nonneg a.den) (num_nonneg.mpr ha)
  /-
    🎉 no goals
  -/


protected lemma div_nonneg {a b : ℚ} (ha : 0 ≤ a) (hb : 0 ≤ b) : 0 ≤ a / b :=
  mul_nonneg ha (Rat.inv_nonneg hb)


protected lemma zpow_nonneg {a : ℚ} (ha : 0 ≤ a) : ∀ n : ℤ, 0 ≤ a ^ n
                      /-
                        a : Rat
                        ha : LE.le 0 a
                        n : Nat
                        ⊢ LE.le 0 (HPow.hPow a (Int.ofNat n))
                      -/
  | Int.ofNat n => by simp [ha]
                      /-
                        🎉 no goals
                      -/
                        /-
                          a : Rat
                          ha : LE.le 0 a
                          n : Nat
                          ⊢ LE.le 0 (HPow.hPow a (Int.negSucc n))
                        -/
  | Int.negSucc n => by simpa using Rat.inv_nonneg (pow_nonneg ha (n + 1))
                        /-
                          🎉 no goals
                        -/


instance instInv : Inv ℚ≥0 where
  inv x := ⟨x⁻¹, Rat.inv_nonneg x.2⟩


instance instDiv : Div ℚ≥0 where
  div x y := ⟨x / y, Rat.div_nonneg x.2 y.2⟩


instance instZPow : Pow ℚ≥0 ℤ where
  pow x n := ⟨x ^ n, Rat.zpow_nonneg x.2 n⟩


@[simp, norm_cast] lemma coe_inv (q : ℚ≥0) : ((q⁻¹ : ℚ≥0) : ℚ) = (q : ℚ)⁻¹ := rfl

@[simp, norm_cast] lemma coe_div (p q : ℚ≥0) : ((p / q : ℚ≥0) : ℚ) = p / q := rfl

@[simp, norm_cast] lemma coe_zpow (p : ℚ≥0) (n : ℤ) : ((p ^ n : ℚ≥0) : ℚ) = p ^ n := rfl


                                                         /-
                                                           q : NNRat
                                                           ⊢ Eq (Inv.inv q) (NNRat.divNat q.den q.num)
                                                         -/
lemma inv_def (q : ℚ≥0) : q⁻¹ = divNat q.den q.num := by ext; simp [Rat.inv_def', num_coe, den_coe]
                                                              /-
                                                                🎉 no goals
                                                              -/

lemma div_def (p q : ℚ≥0) : p / q = divNat (p.num * q.den) (p.den * q.num) := by
  /-
    p q : NNRat
    ⊢ Eq (HDiv.hDiv p q) (NNRat.divNat (HMul.hMul p.num q.den) (HMul.hMul p.den q. …
  -/
  ext; simp [Rat.div_def', num_coe, den_coe]
       /-
         🎉 no goals
       -/


lemma num_inv_of_ne_zero {q : ℚ≥0} (hq : q ≠ 0) : q⁻¹.num = q.den := by
  rw [inv_def, divNat, num, coe_mk, Rat.divInt_ofNat, ← Rat.mk_eq_mkRat _ _ (num_ne_zero.mpr hq),
    Int.natAbs_ofNat]
  /-
    q : NNRat
    hq : Ne q 0
    ⊢ (↑q.den).natAbs.Coprime q.num
  -/
  simpa using q.coprime_num_den.symm
  /-
    🎉 no goals
  -/


lemma den_inv_of_ne_zero {q : ℚ≥0} (hq : q ≠ 0) : q⁻¹.den = q.num := by
  /-
    q : NNRat
    hq : Ne q 0
    ⊢ Eq (Inv.inv q).den q.num
  -/
  rw [inv_def, divNat, den, coe_mk, Rat.divInt_ofNat, ← Rat.mk_eq_mkRat _ _ (num_ne_zero.mpr hq)]
  /-
    q : NNRat
    hq : Ne q 0
    ⊢ (↑q.den).natAbs.Coprime q.num
  -/
  simpa using q.coprime_num_den.symm
  /-
    🎉 no goals
  -/


@[simp]
lemma num_div_den (q : ℚ≥0) : (q.num : ℚ≥0) / q.den = q := by
  /-
    q : NNRat
    ⊢ Eq (HDiv.hDiv ↑q.num ↑q.den) q
  -/
  ext1
  /-
    case a
    q : NNRat
    ⊢ Eq ↑(HDiv.hDiv ↑q.num ↑q.den) ↑q
  -/
  rw [coe_div, coe_natCast, coe_natCast, num, ← Int.cast_natCast]
  /-
    case a
    q : NNRat
    ⊢ Eq (HDiv.hDiv ↑↑(↑q).num.natAbs ↑q.den) ↑q
  -/
  exact (cast_def _).symm
  /-
    🎉 no goals
  -/


instance instSemifield : Semifield ℚ≥0 where
  __ := instNNRatCommSemiring
                 /-
                   ⊢ Eq (Inv.inv 0) 0
                 -/
  inv_zero := by ext; simp
                      /-
                        🎉 no goals
                      -/
                           /-
                             q : NNRat
                             h : Ne q 0
                             ⊢ Eq (HMul.hMul q (Inv.inv q)) 1
                           -/
  mul_inv_cancel q h := by ext; simp [h]
                                /-
                                  🎉 no goals
                                -/
                     /-
                       a : NNRat
                       ⊢ Eq ((fun n a => HPow.hPow a n) 0 a) 1
                     -/
  nnratCast_def q := q.num_div_den.symm
                          /-
                            🎉 no goals
                          -/
                       /-
                         n : Nat
                         a : NNRat
                         ⊢ Eq ((fun n a => HPow.hPow a n) (↑n.succ) a) (HMul.hMul ((fun n a => HPow.hPo …
                       -/
  nnqsmul q a := q * a
                            /-
                              🎉 no goals
                            -/
                      /-
                        n : Nat
                        a : NNRat
                        ⊢ Eq ((fun n a => HPow.hPow a n) (Int.negSucc n) a) (Inv.inv ((fun n a => HPow …
                      -/
  nnqsmul_def q a := rfl
                           /-
                             🎉 no goals
                           -/
  zpow n a := a ^ n
  zpow_zero' a := by ext; norm_cast
  zpow_succ' n a := by ext; norm_cast
  zpow_neg' n a := by ext; norm_cast


