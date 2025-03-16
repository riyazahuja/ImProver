theorem mem_of_integral {x : R} (hx : IsIntegral O x) : x ∈ v.integer :=
  let ⟨p, hpm, hpx⟩ := hx
  le_of_not_lt fun hvx : 1 < v x => by
    /-
      R : Type u
      Γ₀ : Type v
      inst✝³ : CommRing R
      inst✝² : LinearOrderedCommGroupWithZero Γ₀
      v : Valuation R Γ₀
      O : Type w
      inst✝¹ : CommRing O
      inst✝ : Algebra O R
      hv : v.Integers O
      x : R
      hx : IsIntegral O x
      p : Polynomial O
      hpm : p.Monic
      hpx : Eq (Polynomial.eval₂ (algebraMap O R) x p) 0
      hvx : LT.lt 1 (v x)
      ⊢ False
    -/
    rw [hpm.as_sum, eval₂_add, eval₂_pow, eval₂_X, eval₂_finset_sum, add_eq_zero_iff_eq_neg] at hpx
    /-
      R : Type u
      Γ₀ : Type v
      inst✝³ : CommRing R
      inst✝² : LinearOrderedCommGroupWithZero Γ₀
      v : Valuation R Γ₀
      O : Type w
      inst✝¹ : CommRing O
      inst✝ : Algebra O R
      hv : v.Integers O
      x : R
      hx : IsIntegral O x
      p : Polynomial O
      hpm : p.Monic
      hpx : Eq (HPow.hPow x p.natDegree) (Neg.neg ((Finset.range p.natDegree).sum fu …
      hvx : LT.lt 1 (v x)
      ⊢ False
    -/
    replace hpx := congr_arg v hpx; refine ne_of_gt ?_ hpx
    /-
      R : Type u
      Γ₀ : Type v
      inst✝³ : CommRing R
      inst✝² : LinearOrderedCommGroupWithZero Γ₀
      v : Valuation R Γ₀
      O : Type w
      inst✝¹ : CommRing O
      inst✝ : Algebra O R
      hv : v.Integers O
      x : R
      hx : IsIntegral O x
      p : Polynomial O
      hpm : p.Monic
      hvx : LT.lt 1 (v x)
      hpx : Eq (v (HPow.hPow x p.natDegree)) (v (Neg.neg ((Finset.range p.natDegree) …
      ⊢ LT.lt (v (Neg.neg ((Finset.range p.natDegree).sum fun i => Polynomial.eval₂  …
    -/
    rw [v.map_neg, v.map_pow]
    /-
      R : Type u
      Γ₀ : Type v
      inst✝³ : CommRing R
      inst✝² : LinearOrderedCommGroupWithZero Γ₀
      v : Valuation R Γ₀
      O : Type w
      inst✝¹ : CommRing O
      inst✝ : Algebra O R
      hv : v.Integers O
      x : R
      hx : IsIntegral O x
      p : Polynomial O
      hpm : p.Monic
      hvx : LT.lt 1 (v x)
      hpx : Eq (v (HPow.hPow x p.natDegree)) (v (Neg.neg ((Finset.range p.natDegree) …
      ⊢ LT.lt (v ((Finset.range p.natDegree).sum fun i => Polynomial.eval₂ (algebraM …
    -/
    refine v.map_sum_lt' (zero_lt_one.trans_le (one_le_pow_of_one_le' hvx.le _)) fun i hi => ?_
    rw [eval₂_mul, eval₂_pow, eval₂_C, eval₂_X, v.map_mul, v.map_pow, ←
      one_mul (v x ^ p.natDegree)]
    /-
      R : Type u
      Γ₀ : Type v
      inst✝³ : CommRing R
      inst✝² : LinearOrderedCommGroupWithZero Γ₀
      v : Valuation R Γ₀
      O : Type w
      inst✝¹ : CommRing O
      inst✝ : Algebra O R
      hv : v.Integers O
      x : R
      hx : IsIntegral O x
      p : Polynomial O
      hpm : p.Monic
      hvx : LT.lt 1 (v x)
      hpx : Eq (v (HPow.hPow x p.natDegree)) (v (Neg.neg ((Finset.range p.natDegree) …
      i : Nat
      hi : Membership.mem (Finset.range p.natDegree) i
      ⊢ LT.lt (HMul.hMul (v ((algebraMap O R) (p.coeff i))) (HPow.hPow (v x) i)) (HM …
    -/
    cases' (hv.2 <| p.coeff i).lt_or_eq with hvpi hvpi
      /-
        case inl
        R : Type u
        Γ₀ : Type v
        inst✝³ : CommRing R
        inst✝² : LinearOrderedCommGroupWithZero Γ₀
        v : Valuation R Γ₀
        O : Type w
        inst✝¹ : CommRing O
        inst✝ : Algebra O R
        hv : v.Integers O
        x : R
        hx : IsIntegral O x
        p : Polynomial O
        hpm : p.Monic
        hvx : LT.lt 1 (v x)
        hpx : Eq (v (HPow.hPow x p.natDegree)) (v (Neg.neg ((Finset.range p.natDegree) …
        i : Nat
        hi : Membership.mem (Finset.range p.natDegree) i
        hvpi : LT.lt (v ((algebraMap O R) (p.coeff i))) 1
        ⊢ LT.lt (HMul.hMul (v ((algebraMap O R) (p.coeff i))) (HPow.hPow (v x) i)) (HM …
      -/
    · exact mul_lt_mul'' hvpi (pow_lt_pow_right₀ hvx <| Finset.mem_range.1 hi) zero_le' zero_le'
      /-
        🎉 no goals
      -/
      /-
        case inr
        R : Type u
        Γ₀ : Type v
        inst✝³ : CommRing R
        inst✝² : LinearOrderedCommGroupWithZero Γ₀
        v : Valuation R Γ₀
        O : Type w
        inst✝¹ : CommRing O
        inst✝ : Algebra O R
        hv : v.Integers O
        x : R
        hx : IsIntegral O x
        p : Polynomial O
        hpm : p.Monic
        hvx : LT.lt 1 (v x)
        hpx : Eq (v (HPow.hPow x p.natDegree)) (v (Neg.neg ((Finset.range p.natDegree) …
        i : Nat
        hi : Membership.mem (Finset.range p.natDegree) i
        hvpi : Eq (v ((algebraMap O R) (p.coeff i))) 1
        ⊢ LT.lt (HMul.hMul (v ((algebraMap O R) (p.coeff i))) (HPow.hPow (v x) i)) (HM …
      -/
    · rw [hvpi, one_mul, one_mul]; exact pow_lt_pow_right₀ hvx (Finset.mem_range.1 hi)
                                   /-
                                     🎉 no goals
                                   -/


protected theorem integralClosure : integralClosure O R = ⊥ :=
  bot_unique fun _ hr =>
    let ⟨x, hx⟩ := hv.3 (hv.mem_of_integral hr)
    Algebra.mem_bot.2 ⟨x, hx⟩


include hv in
theorem integrallyClosed : IsIntegrallyClosed O :=
  (IsIntegrallyClosed.integralClosure_eq_bot_iff K).mp (Valuation.Integers.integralClosure hv)


