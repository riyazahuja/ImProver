instance commRing : CommRing ℚ where
  __ := addCommGroup
  __ := commMonoid
  zero_mul := Rat.zero_mul
  mul_zero := Rat.mul_zero
  left_distrib := Rat.mul_add
  right_distrib := Rat.add_mul
  intCast := fun n => n
  natCast n := Int.cast n
  natCast_zero := rfl
  natCast_succ n := by
    simp only [intCast_eq_divInt, divInt_add_divInt _ _ Int.one_ne_zero Int.one_ne_zero,
      ← divInt_one_one, Int.natCast_add, Int.natCast_one, mul_one]


instance commGroupWithZero : CommGroupWithZero ℚ :=
  { exists_pair_ne := ⟨0, 1, Rat.zero_ne_one⟩
    inv_zero := by
      /-
        ⊢ Eq (Inv.inv 0) 0
      -/
      change Rat.inv 0 = 0
      /-
        ⊢ Eq (Rat.inv 0) 0
      -/
      rw [Rat.inv_def]
      /-
        ⊢ Eq (Rat.divInt (↑(Rat.den 0)) (Rat.num 0)) 0
      -/
      rfl
      /-
        🎉 no goals
      -/
    mul_inv_cancel := Rat.mul_inv_cancel
    mul_zero := mul_zero
    zero_mul := zero_mul }


instance isDomain : IsDomain ℚ := NoZeroDivisors.to_isDomain _

/-- The characteristic of `ℚ` is 0. -/
@[stacks 09FS "Second part."]
                                                                      /-
                                                                        a b : Nat
                                                                        hab : Eq ↑a ↑b
                                                                        ⊢ Eq a b
                                                                      -/
instance instCharZero : CharZero ℚ where cast_injective a b hab := by simpa using congr_arg num hab
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


                                             /-
                                               ⊢ CommSemiring Rat
                                             -/
instance commSemiring : CommSemiring ℚ := by infer_instance
                                             /-
                                               🎉 no goals
                                             -/

                                             /-
                                               ⊢ Semiring Rat
                                             -/
instance semiring     : Semiring ℚ     := by infer_instance
                                             /-
                                               🎉 no goals
                                             -/


lemma mkRat_eq_div (n : ℤ) (d : ℕ) : mkRat n d = n / d := by
  /-
    n : Int
    d : Nat
    ⊢ Eq (mkRat n d) (HDiv.hDiv ↑n ↑d)
  -/
  simp only [mkRat_eq_divInt, divInt_eq_div, Int.cast_natCast]
  /-
    🎉 no goals
  -/


lemma divInt_div_divInt_cancel_left {x : ℤ} (hx : x ≠ 0) (n d : ℤ) :
    n /. x / (d /. x) = n /. d := by
  /-
    x : Int
    hx : Ne x 0
    n d : Int
    ⊢ Eq (HDiv.hDiv (Rat.divInt n x) (Rat.divInt d x)) (Rat.divInt n d)
  -/
  rw [div_eq_mul_inv, inv_divInt', divInt_mul_divInt_cancel hx]
  /-
    🎉 no goals
  -/


lemma divInt_div_divInt_cancel_right {x : ℤ} (hx : x ≠ 0) (n d : ℤ) :
    x /. n / (x /. d) = d /. n := by
  /-
    x : Int
    hx : Ne x 0
    n d : Int
    ⊢ Eq (HDiv.hDiv (Rat.divInt x n) (Rat.divInt x d)) (Rat.divInt d n)
  -/
  rw [div_eq_mul_inv, inv_divInt', mul_comm, divInt_mul_divInt_cancel hx]
  /-
    🎉 no goals
  -/


lemma num_div_den (r : ℚ) : (r.num : ℚ) / (r.den : ℚ) = r := by
  /-
    r : Rat
    ⊢ Eq (HDiv.hDiv ↑r.num ↑r.den) r
  -/
  rw [← Int.cast_natCast, ← divInt_eq_div, num_divInt_den]
  /-
    🎉 no goals
  -/


@[simp] lemma divInt_pow (num : ℕ) (den : ℤ) (n : ℕ) : (num /. den) ^ n = num ^ n /. den ^ n := by
  /-
    num : Nat
    den : Int
    n : Nat
    ⊢ Eq (HPow.hPow (Rat.divInt (↑num) den) n) (Rat.divInt (HPow.hPow (↑num) n) (H …
  -/
  simp [divInt_eq_div, div_pow, Int.natCast_pow]
  /-
    🎉 no goals
  -/


@[simp] lemma mkRat_pow (num den : ℕ) (n : ℕ) : mkRat num den ^ n = mkRat (num ^ n) (den ^ n) := by
  /-
    num den n : Nat
    ⊢ Eq (HPow.hPow (mkRat (↑num) den) n) (mkRat (HPow.hPow (↑num) n) (HPow.hPow d …
  -/
  rw [mkRat_eq_divInt, mkRat_eq_divInt, divInt_pow, Int.natCast_pow]
  /-
    🎉 no goals
  -/


                                                    /-
                                                      n : Nat
                                                      ⊢ Eq (↑n) (Rat.divInt (↑n) 1)
                                                    -/
lemma natCast_eq_divInt (n : ℕ) : ↑n = n /. 1 := by rw [← Int.cast_natCast, intCast_eq_divInt]
                                                    /-
                                                      🎉 no goals
                                                    -/


@[simp] lemma mul_den_eq_num (q : ℚ) : q * q.den = q.num := by
  suffices (q.num /. ↑q.den) * (↑q.den /. 1) = q.num /. 1 by
    conv => pattern (occs := 1) q; (rw [← num_divInt_den q])
    simp only [intCast_eq_divInt, natCast_eq_divInt, num_divInt_den] at this ⊢; assumption
  /-
    q : Rat
    ⊢ Eq (HMul.hMul (Rat.divInt q.num ↑q.den) (Rat.divInt (↑q.den) 1)) (Rat.divInt …
  -/
  have : (q.den : ℤ) ≠ 0 := mod_cast q.den_ne_zero
  /-
    q : Rat
    this : Ne (↑q.den) 0
    ⊢ Eq (HMul.hMul (Rat.divInt q.num ↑q.den) (Rat.divInt (↑q.den) 1)) (Rat.divInt …
  -/
  rw [divInt_mul_divInt _ _ this Int.one_ne_zero, mul_comm (q.den : ℤ) 1, divInt_mul_right this]
  /-
    🎉 no goals
  -/


                                                               /-
                                                                 q : Rat
                                                                 ⊢ Eq (HMul.hMul (↑q.den) q) ↑q.num
                                                               -/
@[simp] lemma den_mul_eq_num (q : ℚ) : q.den * q = q.num := by rw [mul_comm, mul_den_eq_num]
                                                               /-
                                                                 🎉 no goals
                                                               -/


@[deprecated (since := "2024-04-07")] alias coe_nat_eq_divInt := natCast_eq_divInt


