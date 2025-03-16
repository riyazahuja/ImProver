/-- Most basic version of the "Mellin transform = Dirichlet series" argument. -/
lemma hasSum_mellin {a : ι → ℂ} {p : ι → ℝ} {F : ℝ → ℂ} {s : ℂ}
    (hp : ∀ i, a i = 0 ∨ 0 < p i) (hs : 0 < s.re)
    (hF : ∀ t ∈ Ioi 0, HasSum (fun i ↦ a i * rexp (-p i * t)) (F t))
    (h_sum : Summable fun i ↦ ‖a i‖ / (p i) ^ s.re) :
    HasSum (fun i ↦ Gamma s * a i / p i ^ s) (mellin F s) := by
  simp_rw [mellin, smul_eq_mul, ← setIntegral_congr_fun measurableSet_Ioi
    (fun t ht ↦ congr_arg _ (hF t ht).tsum_eq), ← tsum_mul_left]
  convert hasSum_integral_of_summable_integral_norm
    (F := fun i t ↦ t ^ (s - 1) * (a i * rexp (-p i * t))) (fun i ↦ ?_) ?_ using 2 with i
    /-
      case h.e'_5.h
      ι : Type u_1
      inst✝ : Countable ι
      a : ι → Complex
      p : ι → Real
      F : Real → Complex
      s : Complex
      hp : ∀ (i : ι), Or (Eq (a i) 0) (LT.lt 0 (p i))
      hs : LT.lt 0 s.re
      hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => HMul.hMul ( …
      h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (p i) s.re)
      i : ι
      ⊢ Eq (HDiv.hDiv (HMul.hMul (Complex.Gamma s) (a i)) (HPow.hPow (↑(p i)) s)) (M …
    -/
  · simp_rw [← mul_assoc, mul_comm _ (a _), mul_assoc (a _), mul_div_assoc, integral_mul_left]
    /-
      case h.e'_5.h
      ι : Type u_1
      inst✝ : Countable ι
      a : ι → Complex
      p : ι → Real
      F : Real → Complex
      s : Complex
      hp : ∀ (i : ι), Or (Eq (a i) 0) (LT.lt 0 (p i))
      hs : LT.lt 0 s.re
      hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => HMul.hMul ( …
      h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (p i) s.re)
      i : ι
      ⊢ Eq (HMul.hMul (a i) (HDiv.hDiv (Complex.Gamma s) (HPow.hPow (↑(p i)) s))) (H …
    -/
    rcases hp i with hai | hpi
      /-
        case h.e'_5.h.inl
        ι : Type u_1
        inst✝ : Countable ι
        a : ι → Complex
        p : ι → Real
        F : Real → Complex
        s : Complex
        hp : ∀ (i : ι), Or (Eq (a i) 0) (LT.lt 0 (p i))
        hs : LT.lt 0 s.re
        hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => HMul.hMul ( …
        h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (p i) s.re)
        i : ι
        hai : Eq (a i) 0
        ⊢ Eq (HMul.hMul (a i) (HDiv.hDiv (Complex.Gamma s) (HPow.hPow (↑(p i)) s))) (H …
      -/
    · rw [hai, zero_mul, zero_mul]
      /-
        🎉 no goals
      -/
    /-
      case h.e'_5.h.inr
      ι : Type u_1
      inst✝ : Countable ι
      a : ι → Complex
      p : ι → Real
      F : Real → Complex
      s : Complex
      hp : ∀ (i : ι), Or (Eq (a i) 0) (LT.lt 0 (p i))
      hs : LT.lt 0 s.re
      hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => HMul.hMul ( …
      h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (p i) s.re)
      i : ι
      hpi : LT.lt 0 (p i)
      ⊢ Eq (HMul.hMul (a i) (HDiv.hDiv (Complex.Gamma s) (HPow.hPow (↑(p i)) s))) (H …
    -/
    have := integral_cpow_mul_exp_neg_mul_Ioi hs hpi
    /-
      case h.e'_5.h.inr
      ι : Type u_1
      inst✝ : Countable ι
      a : ι → Complex
      p : ι → Real
      F : Real → Complex
      s : Complex
      hp : ∀ (i : ι), Or (Eq (a i) 0) (LT.lt 0 (p i))
      hs : LT.lt 0 s.re
      hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => HMul.hMul ( …
      h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (p i) s.re)
      i : ι
      hpi : LT.lt 0 (p i)
      this : Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict  …
      ⊢ Eq (HMul.hMul (a i) (HDiv.hDiv (Complex.Gamma s) (HPow.hPow (↑(p i)) s))) (H …
    -/
    simp_rw [← ofReal_mul, ← ofReal_neg, ← ofReal_exp, ← neg_mul (p i)] at this
    /-
      case h.e'_5.h.inr
      ι : Type u_1
      inst✝ : Countable ι
      a : ι → Complex
      p : ι → Real
      F : Real → Complex
      s : Complex
      hp : ∀ (i : ι), Or (Eq (a i) 0) (LT.lt 0 (p i))
      hs : LT.lt 0 s.re
      hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => HMul.hMul ( …
      h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (p i) s.re)
      i : ι
      hpi : LT.lt 0 (p i)
      this : Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict  …
      ⊢ Eq (HMul.hMul (a i) (HDiv.hDiv (Complex.Gamma s) (HPow.hPow (↑(p i)) s))) (H …
    -/
    rw [this, one_div, inv_cpow _ _ (arg_ofReal_of_nonneg hpi.le ▸ pi_pos.ne), div_eq_inv_mul]
    /-
      🎉 no goals
    -/
  · -- integrability of terms
    /-
      case convert_3
      ι : Type u_1
      inst✝ : Countable ι
      a : ι → Complex
      p : ι → Real
      F : Real → Complex
      s : Complex
      hp : ∀ (i : ι), Or (Eq (a i) 0) (LT.lt 0 (p i))
      hs : LT.lt 0 s.re
      hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => HMul.hMul ( …
      h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (p i) s.re)
      i : ι
      ⊢ MeasureTheory.Integrable ((fun i t => HMul.hMul (HPow.hPow (↑t) (HSub.hSub s …
    -/
    rcases hp i with hai | hpi
      /-
        case convert_3.inl
        ι : Type u_1
        inst✝ : Countable ι
        a : ι → Complex
        p : ι → Real
        F : Real → Complex
        s : Complex
        hp : ∀ (i : ι), Or (Eq (a i) 0) (LT.lt 0 (p i))
        hs : LT.lt 0 s.re
        hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => HMul.hMul ( …
        h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (p i) s.re)
        i : ι
        hai : Eq (a i) 0
        ⊢ MeasureTheory.Integrable ((fun i t => HMul.hMul (HPow.hPow (↑t) (HSub.hSub s …
      -/
    · simpa only [hai, zero_mul, mul_zero] using integrable_zero _ _ _
      /-
        🎉 no goals
      -/
    /-
      case convert_3.inr
      ι : Type u_1
      inst✝ : Countable ι
      a : ι → Complex
      p : ι → Real
      F : Real → Complex
      s : Complex
      hp : ∀ (i : ι), Or (Eq (a i) 0) (LT.lt 0 (p i))
      hs : LT.lt 0 s.re
      hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => HMul.hMul ( …
      h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (p i) s.re)
      i : ι
      hpi : LT.lt 0 (p i)
      ⊢ MeasureTheory.Integrable ((fun i t => HMul.hMul (HPow.hPow (↑t) (HSub.hSub s …
    -/
    simp_rw [← mul_assoc, mul_comm _ (a i), mul_assoc]
    /-
      case convert_3.inr
      ι : Type u_1
      inst✝ : Countable ι
      a : ι → Complex
      p : ι → Real
      F : Real → Complex
      s : Complex
      hp : ∀ (i : ι), Or (Eq (a i) 0) (LT.lt 0 (p i))
      hs : LT.lt 0 s.re
      hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => HMul.hMul ( …
      h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (p i) s.re)
      i : ι
      hpi : LT.lt 0 (p i)
      ⊢ MeasureTheory.Integrable (fun t => HMul.hMul (a i) (HMul.hMul (HPow.hPow (↑t …
    -/
    have := Complex.GammaIntegral_convergent hs
    /-
      case convert_3.inr
      ι : Type u_1
      inst✝ : Countable ι
      a : ι → Complex
      p : ι → Real
      F : Real → Complex
      s : Complex
      hp : ∀ (i : ι), Or (Eq (a i) 0) (LT.lt 0 (p i))
      hs : LT.lt 0 s.re
      hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => HMul.hMul ( …
      h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (p i) s.re)
      i : ι
      hpi : LT.lt 0 (p i)
      this : MeasureTheory.IntegrableOn (fun x => HMul.hMul (↑(Real.exp (Neg.neg x)) …
      ⊢ MeasureTheory.Integrable (fun t => HMul.hMul (a i) (HMul.hMul (HPow.hPow (↑t …
    -/
    rw [← mul_zero (p i), ← integrableOn_Ioi_comp_mul_left_iff _ _ hpi] at this
    refine (IntegrableOn.congr_fun (this.const_mul (1 / p i ^ (s - 1)))
      (fun t (ht : 0 < t) ↦ ?_) measurableSet_Ioi).const_mul _
    /-
      case convert_3.inr
      ι : Type u_1
      inst✝ : Countable ι
      a : ι → Complex
      p : ι → Real
      F : Real → Complex
      s : Complex
      hp : ∀ (i : ι), Or (Eq (a i) 0) (LT.lt 0 (p i))
      hs : LT.lt 0 s.re
      hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => HMul.hMul ( …
      h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (p i) s.re)
      i : ι
      hpi : LT.lt 0 (p i)
      this : MeasureTheory.IntegrableOn (fun x => HMul.hMul (↑(Real.exp (Neg.neg (HM …
      t : Real
      ht : LT.lt 0 t
      ⊢ Eq (HMul.hMul (HDiv.hDiv 1 (HPow.hPow (↑(p i)) (HSub.hSub s 1))) (HMul.hMul  …
    -/
    simp_rw [mul_comm (↑(rexp _) : ℂ), ← mul_assoc, neg_mul, ofReal_mul]
    /-
      case convert_3.inr
      ι : Type u_1
      inst✝ : Countable ι
      a : ι → Complex
      p : ι → Real
      F : Real → Complex
      s : Complex
      hp : ∀ (i : ι), Or (Eq (a i) 0) (LT.lt 0 (p i))
      hs : LT.lt 0 s.re
      hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => HMul.hMul ( …
      h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (p i) s.re)
      i : ι
      hpi : LT.lt 0 (p i)
      this : MeasureTheory.IntegrableOn (fun x => HMul.hMul (↑(Real.exp (Neg.neg (HM …
      t : Real
      ht : LT.lt 0 t
      ⊢ Eq (HMul.hMul (HMul.hMul (HDiv.hDiv 1 (HPow.hPow (↑(p i)) (HSub.hSub s 1)))  …
    -/
    rw [mul_cpow_ofReal_nonneg hpi.le ht.le, ← mul_assoc, one_div, inv_mul_cancel₀, one_mul]
    /-
      case convert_3.inr
      ι : Type u_1
      inst✝ : Countable ι
      a : ι → Complex
      p : ι → Real
      F : Real → Complex
      s : Complex
      hp : ∀ (i : ι), Or (Eq (a i) 0) (LT.lt 0 (p i))
      hs : LT.lt 0 s.re
      hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => HMul.hMul ( …
      h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (p i) s.re)
      i : ι
      hpi : LT.lt 0 (p i)
      this : MeasureTheory.IntegrableOn (fun x => HMul.hMul (↑(Real.exp (Neg.neg (HM …
      t : Real
      ht : LT.lt 0 t
      ⊢ Ne (HPow.hPow (↑(p i)) (HSub.hSub s 1)) 0
    -/
    rw [Ne, cpow_eq_zero_iff, not_and_or]
    /-
      case convert_3.inr
      ι : Type u_1
      inst✝ : Countable ι
      a : ι → Complex
      p : ι → Real
      F : Real → Complex
      s : Complex
      hp : ∀ (i : ι), Or (Eq (a i) 0) (LT.lt 0 (p i))
      hs : LT.lt 0 s.re
      hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => HMul.hMul ( …
      h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (p i) s.re)
      i : ι
      hpi : LT.lt 0 (p i)
      this : MeasureTheory.IntegrableOn (fun x => HMul.hMul (↑(Real.exp (Neg.neg (HM …
      t : Real
      ht : LT.lt 0 t
      ⊢ Or (Not (Eq (↑(p i)) 0)) (Not (Ne (HSub.hSub s 1) 0))
    -/
    exact Or.inl (ofReal_ne_zero.mpr hpi.ne')
    /-
      🎉 no goals
    -/
  · -- summability of integrals of norms
    /-
      case convert_4
      ι : Type u_1
      inst✝ : Countable ι
      a : ι → Complex
      p : ι → Real
      F : Real → Complex
      s : Complex
      hp : ∀ (i : ι), Or (Eq (a i) 0) (LT.lt 0 (p i))
      hs : LT.lt 0 s.re
      hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => HMul.hMul ( …
      h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (p i) s.re)
      ⊢ Summable fun i => MeasureTheory.integral (MeasureTheory.MeasureSpace.volume. …
    -/
    apply Summable.of_norm
    /-
      case convert_4.hf
      ι : Type u_1
      inst✝ : Countable ι
      a : ι → Complex
      p : ι → Real
      F : Real → Complex
      s : Complex
      hp : ∀ (i : ι), Or (Eq (a i) 0) (LT.lt 0 (p i))
      hs : LT.lt 0 s.re
      hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => HMul.hMul ( …
      h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (p i) s.re)
      ⊢ Summable fun a_1 => Norm.norm (MeasureTheory.integral (MeasureTheory.Measure …
    -/
    convert h_sum.mul_left (Real.Gamma s.re) using 2 with i
    /-
      case h.e'_5.h
      ι : Type u_1
      inst✝ : Countable ι
      a : ι → Complex
      p : ι → Real
      F : Real → Complex
      s : Complex
      hp : ∀ (i : ι), Or (Eq (a i) 0) (LT.lt 0 (p i))
      hs : LT.lt 0 s.re
      hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => HMul.hMul ( …
      h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (p i) s.re)
      i : ι
      ⊢ Eq (Norm.norm (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.res …
    -/
    simp_rw [← mul_assoc, mul_comm _ (a i), mul_assoc, norm_mul (a i), integral_mul_left]
    /-
      case h.e'_5.h
      ι : Type u_1
      inst✝ : Countable ι
      a : ι → Complex
      p : ι → Real
      F : Real → Complex
      s : Complex
      hp : ∀ (i : ι), Or (Eq (a i) 0) (LT.lt 0 (p i))
      hs : LT.lt 0 s.re
      hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => HMul.hMul ( …
      h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (p i) s.re)
      i : ι
      ⊢ Eq (Norm.norm (HMul.hMul (Norm.norm (a i)) (MeasureTheory.integral (MeasureT …
    -/
    rw [← mul_div_assoc, mul_comm (Real.Gamma _), mul_div_assoc, norm_mul ‖a i‖, norm_norm]
    /-
      case h.e'_5.h
      ι : Type u_1
      inst✝ : Countable ι
      a : ι → Complex
      p : ι → Real
      F : Real → Complex
      s : Complex
      hp : ∀ (i : ι), Or (Eq (a i) 0) (LT.lt 0 (p i))
      hs : LT.lt 0 s.re
      hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => HMul.hMul ( …
      h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (p i) s.re)
      i : ι
      ⊢ Eq (HMul.hMul (Norm.norm (a i)) (Norm.norm (MeasureTheory.integral (MeasureT …
    -/
    rcases hp i with hai | hpi
      /-
        case h.e'_5.h.inl
        ι : Type u_1
        inst✝ : Countable ι
        a : ι → Complex
        p : ι → Real
        F : Real → Complex
        s : Complex
        hp : ∀ (i : ι), Or (Eq (a i) 0) (LT.lt 0 (p i))
        hs : LT.lt 0 s.re
        hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => HMul.hMul ( …
        h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (p i) s.re)
        i : ι
        hai : Eq (a i) 0
        ⊢ Eq (HMul.hMul (Norm.norm (a i)) (Norm.norm (MeasureTheory.integral (MeasureT …
      -/
    · simp only [hai, norm_zero, zero_mul]
      /-
        🎉 no goals
      -/
    /-
      case h.e'_5.h.inr
      ι : Type u_1
      inst✝ : Countable ι
      a : ι → Complex
      p : ι → Real
      F : Real → Complex
      s : Complex
      hp : ∀ (i : ι), Or (Eq (a i) 0) (LT.lt 0 (p i))
      hs : LT.lt 0 s.re
      hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => HMul.hMul ( …
      h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (p i) s.re)
      i : ι
      hpi : LT.lt 0 (p i)
      ⊢ Eq (HMul.hMul (Norm.norm (a i)) (Norm.norm (MeasureTheory.integral (MeasureT …
    -/
    congr 1
    /-
      case h.e'_5.h.inr.e_a
      ι : Type u_1
      inst✝ : Countable ι
      a : ι → Complex
      p : ι → Real
      F : Real → Complex
      s : Complex
      hp : ∀ (i : ι), Or (Eq (a i) 0) (LT.lt 0 (p i))
      hs : LT.lt 0 s.re
      hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => HMul.hMul ( …
      h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (p i) s.re)
      i : ι
      hpi : LT.lt 0 (p i)
      ⊢ Eq (Norm.norm (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.res …
    -/
    have := Real.integral_rpow_mul_exp_neg_mul_Ioi hs hpi
    /-
      case h.e'_5.h.inr.e_a
      ι : Type u_1
      inst✝ : Countable ι
      a : ι → Complex
      p : ι → Real
      F : Real → Complex
      s : Complex
      hp : ∀ (i : ι), Or (Eq (a i) 0) (LT.lt 0 (p i))
      hs : LT.lt 0 s.re
      hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => HMul.hMul ( …
      h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (p i) s.re)
      i : ι
      hpi : LT.lt 0 (p i)
      this : Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict  …
      ⊢ Eq (Norm.norm (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.res …
    -/
    simp_rw [← neg_mul (p i), one_div, inv_rpow hpi.le, ← div_eq_inv_mul] at this
    /-
      case h.e'_5.h.inr.e_a
      ι : Type u_1
      inst✝ : Countable ι
      a : ι → Complex
      p : ι → Real
      F : Real → Complex
      s : Complex
      hp : ∀ (i : ι), Or (Eq (a i) 0) (LT.lt 0 (p i))
      hs : LT.lt 0 s.re
      hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => HMul.hMul ( …
      h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (p i) s.re)
      i : ι
      hpi : LT.lt 0 (p i)
      this : Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict  …
      ⊢ Eq (Norm.norm (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.res …
    -/
    rw [norm_of_nonneg (integral_nonneg (fun _ ↦ norm_nonneg _)), ← this]
    /-
      case h.e'_5.h.inr.e_a
      ι : Type u_1
      inst✝ : Countable ι
      a : ι → Complex
      p : ι → Real
      F : Real → Complex
      s : Complex
      hp : ∀ (i : ι), Or (Eq (a i) 0) (LT.lt 0 (p i))
      hs : LT.lt 0 s.re
      hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => HMul.hMul ( …
      h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (p i) s.re)
      i : ι
      hpi : LT.lt 0 (p i)
      this : Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict  …
      ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict (Set. …
    -/
    refine setIntegral_congr_fun measurableSet_Ioi (fun t ht ↦ ?_)
    rw [norm_mul, norm_real, Real.norm_eq_abs, Real.abs_exp, Complex.norm_eq_abs,
      abs_cpow_eq_rpow_re_of_pos ht, sub_re, one_re]


/-- Shortcut version for the commonly arising special case when `p i = π * q i` for some other
sequence `q`. -/
lemma hasSum_mellin_pi_mul {a : ι → ℂ} {q : ι → ℝ} {F : ℝ → ℂ} {s : ℂ}
    (hq : ∀ i, a i = 0 ∨ 0 < q i) (hs : 0 < s.re)
    (hF : ∀ t ∈ Ioi 0, HasSum (fun i ↦ a i * rexp (-π * q i * t)) (F t))
    (h_sum : Summable fun i ↦ ‖a i‖ / (q i) ^ s.re) :
    HasSum (fun i ↦ π ^ (-s) * Gamma s * a i / q i ^ s) (mellin F s) := by
  /-
    ι : Type u_1
    inst✝ : Countable ι
    a : ι → Complex
    q : ι → Real
    F : Real → Complex
    s : Complex
    hq : ∀ (i : ι), Or (Eq (a i) 0) (LT.lt 0 (q i))
    hs : LT.lt 0 s.re
    hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => HMul.hMul ( …
    h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (q i) s.re)
    ⊢ HasSum (fun i => HDiv.hDiv (HMul.hMul (HMul.hMul (HPow.hPow (↑Real.pi) (Neg. …
  -/
  have hp i : a i = 0 ∨ 0 < π * q i := by rcases hq i with h | h <;> simp [h, pi_pos]
  /-
    ι : Type u_1
    inst✝ : Countable ι
    a : ι → Complex
    q : ι → Real
    F : Real → Complex
    s : Complex
    hq : ∀ (i : ι), Or (Eq (a i) 0) (LT.lt 0 (q i))
    hs : LT.lt 0 s.re
    hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => HMul.hMul ( …
    h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (q i) s.re)
    hp : ∀ (i : ι), Or (Eq (a i) 0) (LT.lt 0 (HMul.hMul Real.pi (q i)))
    ⊢ HasSum (fun i => HDiv.hDiv (HMul.hMul (HMul.hMul (HPow.hPow (↑Real.pi) (Neg. …
  -/
  convert hasSum_mellin hp hs (by simpa using hF) ?_ using 2 with i
  · have : a i / ↑(π * q i) ^ s = π ^ (-s) * a i / q i ^ s := by
      rcases hq i with h | h
      · simp [h]
      · rw [ofReal_mul, mul_cpow_ofReal_nonneg pi_pos.le h.le, ← div_div, cpow_neg,
          ← div_eq_inv_mul]
    /-
      case h.e'_5.h
      ι : Type u_1
      inst✝ : Countable ι
      a : ι → Complex
      q : ι → Real
      F : Real → Complex
      s : Complex
      hq : ∀ (i : ι), Or (Eq (a i) 0) (LT.lt 0 (q i))
      hs : LT.lt 0 s.re
      hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => HMul.hMul ( …
      h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (q i) s.re)
      hp : ∀ (i : ι), Or (Eq (a i) 0) (LT.lt 0 (HMul.hMul Real.pi (q i)))
      i : ι
      this : Eq (HDiv.hDiv (a i) (HPow.hPow (↑(HMul.hMul Real.pi (q i))) s)) (HDiv.h …
      ⊢ Eq (HDiv.hDiv (HMul.hMul (HMul.hMul (HPow.hPow (↑Real.pi) (Neg.neg s)) (Comp …
    -/
    simp_rw [mul_div_assoc, this]
    /-
      case h.e'_5.h
      ι : Type u_1
      inst✝ : Countable ι
      a : ι → Complex
      q : ι → Real
      F : Real → Complex
      s : Complex
      hq : ∀ (i : ι), Or (Eq (a i) 0) (LT.lt 0 (q i))
      hs : LT.lt 0 s.re
      hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => HMul.hMul ( …
      h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (q i) s.re)
      hp : ∀ (i : ι), Or (Eq (a i) 0) (LT.lt 0 (HMul.hMul Real.pi (q i)))
      i : ι
      this : Eq (HDiv.hDiv (a i) (HPow.hPow (↑(HMul.hMul Real.pi (q i))) s)) (HDiv.h …
      ⊢ Eq (HMul.hMul (HMul.hMul (HPow.hPow (↑Real.pi) (Neg.neg s)) (Complex.Gamma s …
    -/
    ring_nf
    /-
      🎉 no goals
    -/
  · have (i) : ‖a i‖ / ↑(π * q i) ^ s.re = π ^ (-s.re) * ‖a i‖ / q i ^ s.re := by
      rcases hq i with h | h
      · simp [h]
      · rw [mul_rpow pi_pos.le h.le, ← div_div, rpow_neg pi_pos.le, ← div_eq_inv_mul]
    /-
      ι : Type u_1
      inst✝ : Countable ι
      a : ι → Complex
      q : ι → Real
      F : Real → Complex
      s : Complex
      hq : ∀ (i : ι), Or (Eq (a i) 0) (LT.lt 0 (q i))
      hs : LT.lt 0 s.re
      hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => HMul.hMul ( …
      h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (q i) s.re)
      hp : ∀ (i : ι), Or (Eq (a i) 0) (LT.lt 0 (HMul.hMul Real.pi (q i)))
      this : ∀ (i : ι), Eq (HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (HMul.hMul Real.p …
      ⊢ Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (HMul.hMul Real.pi  …
    -/
    simpa only [this, mul_div_assoc] using h_sum.mul_left _
    /-
      🎉 no goals
    -/


/-- Version allowing some constant terms (which are omitted from the sums). -/
lemma hasSum_mellin_pi_mul₀ {a : ι → ℂ} {p : ι → ℝ} {F : ℝ → ℂ} {s : ℂ}
    (hp : ∀ i, 0 ≤ p i) (hs : 0 < s.re)
    (hF : ∀ t ∈ Ioi 0, HasSum (fun i ↦ if p i = 0 then 0 else a i * rexp (-π * p i * t)) (F t))
    (h_sum : Summable fun i ↦ ‖a i‖ / (p i) ^ s.re) :
    HasSum (fun i ↦ π ^ (-s) * Gamma s * a i / p i ^ s) (mellin F s) := by
  /-
    ι : Type u_1
    inst✝ : Countable ι
    a : ι → Complex
    p : ι → Real
    F : Real → Complex
    s : Complex
    hp : ∀ (i : ι), LE.le 0 (p i)
    hs : LT.lt 0 s.re
    hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => ite (Eq (p  …
    h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (p i) s.re)
    ⊢ HasSum (fun i => HDiv.hDiv (HMul.hMul (HMul.hMul (HPow.hPow (↑Real.pi) (Neg. …
  -/
  have hs' : s ≠ 0 := fun h ↦ lt_irrefl _ (zero_re ▸ h ▸ hs)
  /-
    ι : Type u_1
    inst✝ : Countable ι
    a : ι → Complex
    p : ι → Real
    F : Real → Complex
    s : Complex
    hp : ∀ (i : ι), LE.le 0 (p i)
    hs : LT.lt 0 s.re
    hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => ite (Eq (p  …
    h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (p i) s.re)
    hs' : Ne s 0
    ⊢ HasSum (fun i => HDiv.hDiv (HMul.hMul (HMul.hMul (HPow.hPow (↑Real.pi) (Neg. …
  -/
  let a' i := if p i = 0 then 0 else a i
  have hp' i : a' i = 0 ∨ 0 < p i := by
    simp only [a']
    split_ifs with h <;> try tauto
    exact Or.inr (lt_of_le_of_ne (hp i) (Ne.symm h))
  have (i t) : (if p i = 0 then 0 else a i * rexp (-π * p i * t)) =
      a' i * rexp (-π * p i * t) := by
    simp only [a', ite_mul, zero_mul]
  /-
    ι : Type u_1
    inst✝ : Countable ι
    a : ι → Complex
    p : ι → Real
    F : Real → Complex
    s : Complex
    hp : ∀ (i : ι), LE.le 0 (p i)
    hs : LT.lt 0 s.re
    hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => ite (Eq (p  …
    h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (p i) s.re)
    hs' : Ne s 0
    a' : ι → Complex := fun i => ite (Eq (p i) 0) 0 (a i)
    hp' : ∀ (i : ι), Or (Eq (a' i) 0) (LT.lt 0 (p i))
    this : ∀ (i : ι) (t : Real), Eq (ite (Eq (p i) 0) 0 (HMul.hMul (a i) ↑(Real.ex …
    ⊢ HasSum (fun i => HDiv.hDiv (HMul.hMul (HMul.hMul (HPow.hPow (↑Real.pi) (Neg. …
  -/
  simp_rw [this] at hF
  /-
    ι : Type u_1
    inst✝ : Countable ι
    a : ι → Complex
    p : ι → Real
    F : Real → Complex
    s : Complex
    hp : ∀ (i : ι), LE.le 0 (p i)
    hs : LT.lt 0 s.re
    h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (p i) s.re)
    hs' : Ne s 0
    a' : ι → Complex := fun i => ite (Eq (p i) 0) 0 (a i)
    hp' : ∀ (i : ι), Or (Eq (a' i) 0) (LT.lt 0 (p i))
    this : ∀ (i : ι) (t : Real), Eq (ite (Eq (p i) 0) 0 (HMul.hMul (a i) ↑(Real.ex …
    hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => HMul.hMul ( …
    ⊢ HasSum (fun i => HDiv.hDiv (HMul.hMul (HMul.hMul (HPow.hPow (↑Real.pi) (Neg. …
  -/
  convert hasSum_mellin_pi_mul hp' hs hF ?_ using 2 with i
    /-
      case h.e'_5.h
      ι : Type u_1
      inst✝ : Countable ι
      a : ι → Complex
      p : ι → Real
      F : Real → Complex
      s : Complex
      hp : ∀ (i : ι), LE.le 0 (p i)
      hs : LT.lt 0 s.re
      h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (p i) s.re)
      hs' : Ne s 0
      a' : ι → Complex := fun i => ite (Eq (p i) 0) 0 (a i)
      hp' : ∀ (i : ι), Or (Eq (a' i) 0) (LT.lt 0 (p i))
      this : ∀ (i : ι) (t : Real), Eq (ite (Eq (p i) 0) 0 (HMul.hMul (a i) ↑(Real.ex …
      hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => HMul.hMul ( …
      i : ι
      ⊢ Eq (HDiv.hDiv (HMul.hMul (HMul.hMul (HPow.hPow (↑Real.pi) (Neg.neg s)) (Comp …
    -/
  · rcases eq_or_ne (p i) 0 with h | h <;>
    /-
      case h.e'_5.h.inl
      ι : Type u_1
      inst✝ : Countable ι
      a : ι → Complex
      p : ι → Real
      F : Real → Complex
      s : Complex
      hp : ∀ (i : ι), LE.le 0 (p i)
      hs : LT.lt 0 s.re
      h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (p i) s.re)
      hs' : Ne s 0
      a' : ι → Complex := fun i => ite (Eq (p i) 0) 0 (a i)
      hp' : ∀ (i : ι), Or (Eq (a' i) 0) (LT.lt 0 (p i))
      this : ∀ (i : ι) (t : Real), Eq (ite (Eq (p i) 0) 0 (HMul.hMul (a i) ↑(Real.ex …
      hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => HMul.hMul ( …
      i : ι
      h : Eq (p i) 0
      ⊢ Eq (HDiv.hDiv (HMul.hMul (HMul.hMul (HPow.hPow (↑Real.pi) (Neg.neg s)) (Comp …
    -/
    /-
      🎉 no goals
    -/
    simp [a', h, if_false, ofReal_zero, zero_cpow hs', div_zero]
    /-
      🎉 no goals
    -/
    /-
      ι : Type u_1
      inst✝ : Countable ι
      a : ι → Complex
      p : ι → Real
      F : Real → Complex
      s : Complex
      hp : ∀ (i : ι), LE.le 0 (p i)
      hs : LT.lt 0 s.re
      h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (p i) s.re)
      hs' : Ne s 0
      a' : ι → Complex := fun i => ite (Eq (p i) 0) 0 (a i)
      hp' : ∀ (i : ι), Or (Eq (a' i) 0) (LT.lt 0 (p i))
      this : ∀ (i : ι) (t : Real), Eq (ite (Eq (p i) 0) 0 (HMul.hMul (a i) ↑(Real.ex …
      hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => HMul.hMul ( …
      ⊢ Summable fun i => HDiv.hDiv (Norm.norm (a' i)) (HPow.hPow (p i) s.re)
    -/
  · refine h_sum.of_norm_bounded _ (fun i ↦ ?_)
    /-
      ι : Type u_1
      inst✝ : Countable ι
      a : ι → Complex
      p : ι → Real
      F : Real → Complex
      s : Complex
      hp : ∀ (i : ι), LE.le 0 (p i)
      hs : LT.lt 0 s.re
      h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (p i) s.re)
      hs' : Ne s 0
      a' : ι → Complex := fun i => ite (Eq (p i) 0) 0 (a i)
      hp' : ∀ (i : ι), Or (Eq (a' i) 0) (LT.lt 0 (p i))
      this : ∀ (i : ι) (t : Real), Eq (ite (Eq (p i) 0) 0 (HMul.hMul (a i) ↑(Real.ex …
      hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => HMul.hMul ( …
      i : ι
      ⊢ LE.le (Norm.norm (HDiv.hDiv (Norm.norm (a' i)) (HPow.hPow (p i) s.re))) (HDi …
    -/
    simp only [a']
    /-
      ι : Type u_1
      inst✝ : Countable ι
      a : ι → Complex
      p : ι → Real
      F : Real → Complex
      s : Complex
      hp : ∀ (i : ι), LE.le 0 (p i)
      hs : LT.lt 0 s.re
      h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (p i) s.re)
      hs' : Ne s 0
      a' : ι → Complex := fun i => ite (Eq (p i) 0) 0 (a i)
      hp' : ∀ (i : ι), Or (Eq (a' i) 0) (LT.lt 0 (p i))
      this : ∀ (i : ι) (t : Real), Eq (ite (Eq (p i) 0) 0 (HMul.hMul (a i) ↑(Real.ex …
      hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => HMul.hMul ( …
      i : ι
      ⊢ LE.le (Norm.norm (HDiv.hDiv (Norm.norm (ite (Eq (p i) 0) 0 (a i))) (HPow.hPo …
    -/
    split_ifs
      /-
        case pos
        ι : Type u_1
        inst✝ : Countable ι
        a : ι → Complex
        p : ι → Real
        F : Real → Complex
        s : Complex
        hp : ∀ (i : ι), LE.le 0 (p i)
        hs : LT.lt 0 s.re
        h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (p i) s.re)
        hs' : Ne s 0
        a' : ι → Complex := fun i => ite (Eq (p i) 0) 0 (a i)
        hp' : ∀ (i : ι), Or (Eq (a' i) 0) (LT.lt 0 (p i))
        this : ∀ (i : ι) (t : Real), Eq (ite (Eq (p i) 0) 0 (HMul.hMul (a i) ↑(Real.ex …
        hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => HMul.hMul ( …
        i : ι
        h✝ : Eq (p i) 0
        ⊢ LE.le (Norm.norm (HDiv.hDiv (Norm.norm 0) (HPow.hPow (p i) s.re))) (HDiv.hDi …
      -/
    · simp only [norm_zero, zero_div]
      /-
        case pos
        ι : Type u_1
        inst✝ : Countable ι
        a : ι → Complex
        p : ι → Real
        F : Real → Complex
        s : Complex
        hp : ∀ (i : ι), LE.le 0 (p i)
        hs : LT.lt 0 s.re
        h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (p i) s.re)
        hs' : Ne s 0
        a' : ι → Complex := fun i => ite (Eq (p i) 0) 0 (a i)
        hp' : ∀ (i : ι), Or (Eq (a' i) 0) (LT.lt 0 (p i))
        this : ∀ (i : ι) (t : Real), Eq (ite (Eq (p i) 0) 0 (HMul.hMul (a i) ↑(Real.ex …
        hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => HMul.hMul ( …
        i : ι
        h✝ : Eq (p i) 0
        ⊢ LE.le 0 (HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (p i) s.re))
      -/
      positivity
      /-
        🎉 no goals
      -/
      /-
        case neg
        ι : Type u_1
        inst✝ : Countable ι
        a : ι → Complex
        p : ι → Real
        F : Real → Complex
        s : Complex
        hp : ∀ (i : ι), LE.le 0 (p i)
        hs : LT.lt 0 s.re
        h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (p i) s.re)
        hs' : Ne s 0
        a' : ι → Complex := fun i => ite (Eq (p i) 0) 0 (a i)
        hp' : ∀ (i : ι), Or (Eq (a' i) 0) (LT.lt 0 (p i))
        this : ∀ (i : ι) (t : Real), Eq (ite (Eq (p i) 0) 0 (HMul.hMul (a i) ↑(Real.ex …
        hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => HMul.hMul ( …
        i : ι
        h✝ : Not (Eq (p i) 0)
        ⊢ LE.le (Norm.norm (HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (p i) s.re))) (HDiv …
      -/
    · have := hp i
      /-
        case neg
        ι : Type u_1
        inst✝ : Countable ι
        a : ι → Complex
        p : ι → Real
        F : Real → Complex
        s : Complex
        hp : ∀ (i : ι), LE.le 0 (p i)
        hs : LT.lt 0 s.re
        h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (p i) s.re)
        hs' : Ne s 0
        a' : ι → Complex := fun i => ite (Eq (p i) 0) 0 (a i)
        hp' : ∀ (i : ι), Or (Eq (a' i) 0) (LT.lt 0 (p i))
        this✝ : ∀ (i : ι) (t : Real), Eq (ite (Eq (p i) 0) 0 (HMul.hMul (a i) ↑(Real.e …
        hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => HMul.hMul ( …
        i : ι
        h✝ : Not (Eq (p i) 0)
        this : LE.le 0 (p i)
        ⊢ LE.le (Norm.norm (HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (p i) s.re))) (HDiv …
      -/
      rw [norm_of_nonneg (by positivity)]
      /-
        🎉 no goals
      -/


/-- Tailored version for even Jacobi theta functions. -/
lemma hasSum_mellin_pi_mul_sq {a : ι → ℂ} {r : ι → ℝ} {F : ℝ → ℂ} {s : ℂ} (hs : 0 < s.re)
    (hF : ∀ t ∈ Ioi 0, HasSum (fun i ↦ if r i = 0 then 0 else a i * rexp (-π * r i ^ 2 * t)) (F t))
    (h_sum : Summable fun i ↦ ‖a i‖ / |r i| ^ s.re) :
    HasSum (fun i ↦ Gammaℝ s * a i / |r i| ^ s) (mellin F (s / 2)) := by
  /-
    ι : Type u_1
    inst✝ : Countable ι
    a : ι → Complex
    r : ι → Real
    F : Real → Complex
    s : Complex
    hs : LT.lt 0 s.re
    hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => ite (Eq (r  …
    h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (abs (r i)) s …
    ⊢ HasSum (fun i => HDiv.hDiv (HMul.hMul s.Gammaℝ (a i)) (HPow.hPow (↑(abs (r i …
  -/
  have hs' : 0 < (s / 2).re := by rw [div_ofNat_re]; positivity
  /-
    ι : Type u_1
    inst✝ : Countable ι
    a : ι → Complex
    r : ι → Real
    F : Real → Complex
    s : Complex
    hs : LT.lt 0 s.re
    hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => ite (Eq (r  …
    h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (abs (r i)) s …
    hs' : LT.lt 0 (HDiv.hDiv s 2).re
    ⊢ HasSum (fun i => HDiv.hDiv (HMul.hMul s.Gammaℝ (a i)) (HPow.hPow (↑(abs (r i …
  -/
  simp_rw [← sq_eq_zero_iff (a := r _)] at hF
  /-
    ι : Type u_1
    inst✝ : Countable ι
    a : ι → Complex
    r : ι → Real
    F : Real → Complex
    s : Complex
    hs : LT.lt 0 s.re
    h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (abs (r i)) s …
    hs' : LT.lt 0 (HDiv.hDiv s 2).re
    hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => ite (Eq (HP …
    ⊢ HasSum (fun i => HDiv.hDiv (HMul.hMul s.Gammaℝ (a i)) (HPow.hPow (↑(abs (r i …
  -/
  convert hasSum_mellin_pi_mul₀ (fun i ↦ sq_nonneg (r i)) hs' hF ?_ using 3 with i
    /-
      case h.e'_5.h.h.e'_5
      ι : Type u_1
      inst✝ : Countable ι
      a : ι → Complex
      r : ι → Real
      F : Real → Complex
      s : Complex
      hs : LT.lt 0 s.re
      h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (abs (r i)) s …
      hs' : LT.lt 0 (HDiv.hDiv s 2).re
      hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => ite (Eq (HP …
      i : ι
      ⊢ Eq (HMul.hMul s.Gammaℝ (a i)) (HMul.hMul (HMul.hMul (HPow.hPow (↑Real.pi) (N …
    -/
  · rw [← neg_div, Gammaℝ_def]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_5.h.h.e'_6
      ι : Type u_1
      inst✝ : Countable ι
      a : ι → Complex
      r : ι → Real
      F : Real → Complex
      s : Complex
      hs : LT.lt 0 s.re
      h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (abs (r i)) s …
      hs' : LT.lt 0 (HDiv.hDiv s 2).re
      hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => ite (Eq (HP …
      i : ι
      ⊢ Eq (HPow.hPow (↑(abs (r i))) s) (HPow.hPow (↑(HPow.hPow (r i) 2)) (HDiv.hDiv …
    -/
  · rw [← _root_.sq_abs, ofReal_pow, ← cpow_nat_mul']
      /-
        case h.e'_5.h.h.e'_6
        ι : Type u_1
        inst✝ : Countable ι
        a : ι → Complex
        r : ι → Real
        F : Real → Complex
        s : Complex
        hs : LT.lt 0 s.re
        h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (abs (r i)) s …
        hs' : LT.lt 0 (HDiv.hDiv s 2).re
        hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => ite (Eq (HP …
        i : ι
        ⊢ Eq (HPow.hPow (↑(abs (r i))) s) (HPow.hPow (↑(abs (r i))) (HMul.hMul (↑2) (H …
      -/
    · ring_nf
      /-
        🎉 no goals
      -/
    /-
      case h.e'_5.h.h.e'_6.hlt
      ι : Type u_1
      inst✝ : Countable ι
      a : ι → Complex
      r : ι → Real
      F : Real → Complex
      s : Complex
      hs : LT.lt 0 s.re
      h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (abs (r i)) s …
      hs' : LT.lt 0 (HDiv.hDiv s 2).re
      hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => ite (Eq (HP …
      i : ι
      ⊢ LT.lt (Neg.neg Real.pi) (HMul.hMul (↑2) (↑(abs (r i))).arg)
    -/
    all_goals rw [arg_ofReal_of_nonneg (abs_nonneg _)]; linarith [pi_pos]
    /-
      🎉 no goals
    -/
    /-
      ι : Type u_1
      inst✝ : Countable ι
      a : ι → Complex
      r : ι → Real
      F : Real → Complex
      s : Complex
      hs : LT.lt 0 s.re
      h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (abs (r i)) s …
      hs' : LT.lt 0 (HDiv.hDiv s 2).re
      hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => ite (Eq (HP …
      ⊢ Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (HPow.hPow (r i) 2) …
    -/
  · convert h_sum using 3 with i
    rw [← _root_.sq_abs, ← rpow_natCast_mul (abs_nonneg _), div_ofNat_re, Nat.cast_ofNat,
      mul_div_cancel₀ _ two_pos.ne']


/-- Tailored version for odd Jacobi theta functions. -/
lemma hasSum_mellin_pi_mul_sq' {a : ι → ℂ} {r : ι → ℝ} {F : ℝ → ℂ} {s : ℂ} (hs : 0 < s.re)
    (hF : ∀ t ∈ Ioi 0, HasSum (fun i ↦ a i * r i * rexp (-π * r i ^ 2 * t)) (F t))
    (h_sum : Summable fun i ↦ ‖a i‖ / |r i| ^ s.re) :
    HasSum (fun i ↦ Gammaℝ (s + 1) * a i * SignType.sign (r i) / |r i| ^ s)
    (mellin F ((s + 1) / 2)) := by
  /-
    ι : Type u_1
    inst✝ : Countable ι
    a : ι → Complex
    r : ι → Real
    F : Real → Complex
    s : Complex
    hs : LT.lt 0 s.re
    hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => HMul.hMul ( …
    h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (abs (r i)) s …
    ⊢ HasSum (fun i => HDiv.hDiv (HMul.hMul (HMul.hMul (HAdd.hAdd s 1).Gammaℝ (a i …
  -/
  have hs₁ : s ≠ 0 := fun h ↦ lt_irrefl _ (zero_re ▸ h ▸ hs)
  /-
    ι : Type u_1
    inst✝ : Countable ι
    a : ι → Complex
    r : ι → Real
    F : Real → Complex
    s : Complex
    hs : LT.lt 0 s.re
    hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => HMul.hMul ( …
    h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (abs (r i)) s …
    hs₁ : Ne s 0
    ⊢ HasSum (fun i => HDiv.hDiv (HMul.hMul (HMul.hMul (HAdd.hAdd s 1).Gammaℝ (a i …
  -/
  have hs₂ : 0 < (s + 1).re := by rw [add_re, one_re]; positivity
  /-
    ι : Type u_1
    inst✝ : Countable ι
    a : ι → Complex
    r : ι → Real
    F : Real → Complex
    s : Complex
    hs : LT.lt 0 s.re
    hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => HMul.hMul ( …
    h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (abs (r i)) s …
    hs₁ : Ne s 0
    hs₂ : LT.lt 0 (HAdd.hAdd s 1).re
    ⊢ HasSum (fun i => HDiv.hDiv (HMul.hMul (HMul.hMul (HAdd.hAdd s 1).Gammaℝ (a i …
  -/
  have hs₃ : s + 1 ≠ 0 := fun h ↦ lt_irrefl _ (zero_re ▸ h ▸ hs₂)
  have (i t) : (a i * r i * rexp (-π * r i ^ 2 * t)) =
      if r i = 0 then 0 else (a i * r i * rexp (-π * r i ^ 2 * t)) := by
    split_ifs with h <;> simp [h]
  /-
    ι : Type u_1
    inst✝ : Countable ι
    a : ι → Complex
    r : ι → Real
    F : Real → Complex
    s : Complex
    hs : LT.lt 0 s.re
    hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => HMul.hMul ( …
    h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (abs (r i)) s …
    hs₁ : Ne s 0
    hs₂ : LT.lt 0 (HAdd.hAdd s 1).re
    hs₃ : Ne (HAdd.hAdd s 1) 0
    this : ∀ (i : ι) (t : Real), Eq (HMul.hMul (HMul.hMul (a i) ↑(r i)) ↑(Real.exp …
    ⊢ HasSum (fun i => HDiv.hDiv (HMul.hMul (HMul.hMul (HAdd.hAdd s 1).Gammaℝ (a i …
  -/
  conv at hF => enter [t, ht, 1, i]; rw [this]
  /-
    ι : Type u_1
    inst✝ : Countable ι
    a : ι → Complex
    r : ι → Real
    F : Real → Complex
    s : Complex
    hs : LT.lt 0 s.re
    hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => ite (Eq (r  …
    h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (abs (r i)) s …
    hs₁ : Ne s 0
    hs₂ : LT.lt 0 (HAdd.hAdd s 1).re
    hs₃ : Ne (HAdd.hAdd s 1) 0
    this : ∀ (i : ι) (t : Real), Eq (HMul.hMul (HMul.hMul (a i) ↑(r i)) ↑(Real.exp …
    ⊢ HasSum (fun i => HDiv.hDiv (HMul.hMul (HMul.hMul (HAdd.hAdd s 1).Gammaℝ (a i …
  -/
  convert hasSum_mellin_pi_mul_sq hs₂ hF ?_ using 2 with i
    /-
      case h.e'_5.h
      ι : Type u_1
      inst✝ : Countable ι
      a : ι → Complex
      r : ι → Real
      F : Real → Complex
      s : Complex
      hs : LT.lt 0 s.re
      hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => ite (Eq (r  …
      h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (abs (r i)) s …
      hs₁ : Ne s 0
      hs₂ : LT.lt 0 (HAdd.hAdd s 1).re
      hs₃ : Ne (HAdd.hAdd s 1) 0
      this : ∀ (i : ι) (t : Real), Eq (HMul.hMul (HMul.hMul (a i) ↑(r i)) ↑(Real.exp …
      i : ι
      ⊢ Eq (HDiv.hDiv (HMul.hMul (HMul.hMul (HAdd.hAdd s 1).Gammaℝ (a i)) ↑(SignType …
    -/
  · rcases eq_or_ne (r i) 0 with h | h
      /-
        case h.e'_5.h.inl
        ι : Type u_1
        inst✝ : Countable ι
        a : ι → Complex
        r : ι → Real
        F : Real → Complex
        s : Complex
        hs : LT.lt 0 s.re
        hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => ite (Eq (r  …
        h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (abs (r i)) s …
        hs₁ : Ne s 0
        hs₂ : LT.lt 0 (HAdd.hAdd s 1).re
        hs₃ : Ne (HAdd.hAdd s 1) 0
        this : ∀ (i : ι) (t : Real), Eq (HMul.hMul (HMul.hMul (a i) ↑(r i)) ↑(Real.exp …
        i : ι
        h : Eq (r i) 0
        ⊢ Eq (HDiv.hDiv (HMul.hMul (HMul.hMul (HAdd.hAdd s 1).Gammaℝ (a i)) ↑(SignType …
      -/
    · rw [h, abs_zero, ofReal_zero, zero_cpow hs₁, zero_cpow hs₃, div_zero, div_zero]
      /-
        🎉 no goals
      -/
      /-
        case h.e'_5.h.inr
        ι : Type u_1
        inst✝ : Countable ι
        a : ι → Complex
        r : ι → Real
        F : Real → Complex
        s : Complex
        hs : LT.lt 0 s.re
        hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => ite (Eq (r  …
        h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (abs (r i)) s …
        hs₁ : Ne s 0
        hs₂ : LT.lt 0 (HAdd.hAdd s 1).re
        hs₃ : Ne (HAdd.hAdd s 1) 0
        this : ∀ (i : ι) (t : Real), Eq (HMul.hMul (HMul.hMul (a i) ↑(r i)) ↑(Real.exp …
        i : ι
        h : Ne (r i) 0
        ⊢ Eq (HDiv.hDiv (HMul.hMul (HMul.hMul (HAdd.hAdd s 1).Gammaℝ (a i)) ↑(SignType …
      -/
    · rw [cpow_add _ _ (ofReal_ne_zero.mpr <| abs_ne_zero.mpr h), cpow_one]
      conv_rhs => enter [1]; rw [← sign_mul_abs (r i), ofReal_mul, ← ofRealHom_eq_coe,
        SignType.map_cast]
      /-
        case h.e'_5.h.inr
        ι : Type u_1
        inst✝ : Countable ι
        a : ι → Complex
        r : ι → Real
        F : Real → Complex
        s : Complex
        hs : LT.lt 0 s.re
        hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => ite (Eq (r  …
        h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (abs (r i)) s …
        hs₁ : Ne s 0
        hs₂ : LT.lt 0 (HAdd.hAdd s 1).re
        hs₃ : Ne (HAdd.hAdd s 1) 0
        this : ∀ (i : ι) (t : Real), Eq (HMul.hMul (HMul.hMul (a i) ↑(r i)) ↑(Real.exp …
        i : ι
        h : Ne (r i) 0
        ⊢ Eq (HDiv.hDiv (HMul.hMul (HMul.hMul (HAdd.hAdd s 1).Gammaℝ (a i)) ↑(SignType …
      -/
      field_simp [h]
      /-
        case h.e'_5.h.inr
        ι : Type u_1
        inst✝ : Countable ι
        a : ι → Complex
        r : ι → Real
        F : Real → Complex
        s : Complex
        hs : LT.lt 0 s.re
        hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => ite (Eq (r  …
        h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (abs (r i)) s …
        hs₁ : Ne s 0
        hs₂ : LT.lt 0 (HAdd.hAdd s 1).re
        hs₃ : Ne (HAdd.hAdd s 1) 0
        this : ∀ (i : ι) (t : Real), Eq (HMul.hMul (HMul.hMul (a i) ↑(r i)) ↑(Real.exp …
        i : ι
        h : Ne (r i) 0
        ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (HAdd.hAdd s 1).Gammaℝ (a i)) ↑(SignType …
      -/
      ring_nf
      /-
        🎉 no goals
      -/
    /-
      ι : Type u_1
      inst✝ : Countable ι
      a : ι → Complex
      r : ι → Real
      F : Real → Complex
      s : Complex
      hs : LT.lt 0 s.re
      hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => ite (Eq (r  …
      h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (abs (r i)) s …
      hs₁ : Ne s 0
      hs₂ : LT.lt 0 (HAdd.hAdd s 1).re
      hs₃ : Ne (HAdd.hAdd s 1) 0
      this : ∀ (i : ι) (t : Real), Eq (HMul.hMul (HMul.hMul (a i) ↑(r i)) ↑(Real.exp …
      ⊢ Summable fun i => HDiv.hDiv (Norm.norm (HMul.hMul (a i) ↑(r i))) (HPow.hPow  …
    -/
  · convert h_sum using 2 with i
    /-
      case h.e'_5.h
      ι : Type u_1
      inst✝ : Countable ι
      a : ι → Complex
      r : ι → Real
      F : Real → Complex
      s : Complex
      hs : LT.lt 0 s.re
      hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => ite (Eq (r  …
      h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (abs (r i)) s …
      hs₁ : Ne s 0
      hs₂ : LT.lt 0 (HAdd.hAdd s 1).re
      hs₃ : Ne (HAdd.hAdd s 1) 0
      this : ∀ (i : ι) (t : Real), Eq (HMul.hMul (HMul.hMul (a i) ↑(r i)) ↑(Real.exp …
      i : ι
      ⊢ Eq (HDiv.hDiv (Norm.norm (HMul.hMul (a i) ↑(r i))) (HPow.hPow (abs (r i)) (H …
    -/
    rcases eq_or_ne (r i) 0 with h | h
      /-
        case h.e'_5.h.inl
        ι : Type u_1
        inst✝ : Countable ι
        a : ι → Complex
        r : ι → Real
        F : Real → Complex
        s : Complex
        hs : LT.lt 0 s.re
        hF : ∀ (t : Real), Membership.mem (Set.Ioi 0) t → HasSum (fun i => ite (Eq (r  …
        h_sum : Summable fun i => HDiv.hDiv (Norm.norm (a i)) (HPow.hPow (abs (r i)) s …
        hs₁ : Ne s 0
        hs₂ : LT.lt 0 (HAdd.hAdd s 1).re
        hs₃ : Ne (HAdd.hAdd s 1) 0
        this : ∀ (i : ι) (t : Real), Eq (HMul.hMul (HMul.hMul (a i) ↑(r i)) ↑(Real.exp …
        i : ι
        h : Eq (r i) 0
        ⊢ Eq (HDiv.hDiv (Norm.norm (HMul.hMul (a i) ↑(r i))) (HPow.hPow (abs (r i)) (H …
      -/
    · rw [h, abs_zero, ofReal_zero, zero_rpow hs₂.ne', zero_rpow hs.ne', div_zero, div_zero]
      /-
        🎉 no goals
      -/
    · rw [add_re, one_re, rpow_add (abs_pos.mpr h), rpow_one, norm_mul, norm_real,
        Real.norm_eq_abs, ← div_div, div_right_comm, mul_div_cancel_right₀ _ (abs_ne_zero.mpr h)]

