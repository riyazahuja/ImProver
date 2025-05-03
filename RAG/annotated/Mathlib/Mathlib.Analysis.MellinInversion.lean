private theorem rexp_neg_deriv_aux :
    ∀ x ∈ univ, HasDerivWithinAt (rexp ∘ Neg.neg) (-rexp (-x)) univ x :=
  fun x _ ↦ mul_neg_one (rexp (-x)) ▸
    ((Real.hasDerivAt_exp (-x)).comp x (hasDerivAt_neg x)).hasDerivWithinAt


private theorem rexp_neg_image_aux : rexp ∘ Neg.neg '' univ = Ioi 0 := by
  /-
    ⊢ Eq (Set.image (Function.comp Real.exp Neg.neg) Set.univ) (Set.Ioi 0)
  -/
  rw [Set.image_comp, Set.image_univ_of_surjective neg_surjective, Set.image_univ, Real.range_exp]
  /-
    🎉 no goals
  -/


private theorem rexp_neg_injOn_aux : univ.InjOn (rexp ∘ Neg.neg) :=
  Real.exp_injective.injOn.comp neg_injective.injOn (univ.mapsTo_univ _)


private theorem rexp_cexp_aux (x : ℝ) (s : ℂ) (f : E) :
    rexp (-x) • cexp (-↑x) ^ (s - 1) • f = cexp (-s * ↑x) • f := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    x : Real
    s : Complex
    f : E
    ⊢ Eq (HSMul.hSMul (Real.exp (Neg.neg x)) (HSMul.hSMul (HPow.hPow (Complex.exp  …
  -/
  show (rexp (-x) : ℂ) • _ = _ • f
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    x : Real
    s : Complex
    f : E
    ⊢ Eq (HSMul.hSMul (↑(Real.exp (Neg.neg x))) (HSMul.hSMul (HPow.hPow (Complex.e …
  -/
  rw [← smul_assoc, smul_eq_mul]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    x : Real
    s : Complex
    f : E
    ⊢ Eq (HSMul.hSMul (HMul.hMul (↑(Real.exp (Neg.neg x))) (HPow.hPow (Complex.exp …
  -/
  push_cast
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    x : Real
    s : Complex
    f : E
    ⊢ Eq (HSMul.hSMul (HMul.hMul (Complex.exp (Neg.neg ↑x)) (HPow.hPow (Complex.ex …
  -/
  conv in cexp _ * _ => lhs; rw [← cpow_one (cexp _)]
  rw [← cpow_add _ _ (Complex.exp_ne_zero _), cpow_def_of_ne_zero (Complex.exp_ne_zero _),
    Complex.log_exp (by norm_num; exact pi_pos) (by simpa using pi_nonneg)]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    x : Real
    s : Complex
    f : E
    ⊢ Eq (HSMul.hSMul (Complex.exp (HMul.hMul (Neg.neg ↑x) (HAdd.hAdd 1 (HSub.hSub …
  -/
  ring_nf
  /-
    🎉 no goals
  -/


theorem mellin_eq_fourierIntegral (f : ℝ → E) {s : ℂ} :
    mellin f s = 𝓕 (fun (u : ℝ) ↦ (Real.exp (-s.re * u) • f (Real.exp (-u)))) (s.im / (2 * π)) :=
  calc
    mellin f s
      = ∫ (u : ℝ), Complex.exp (-s * u) • f (Real.exp (-u)) := by
      rw [mellin, ← rexp_neg_image_aux, integral_image_eq_integral_abs_deriv_smul
        MeasurableSet.univ rexp_neg_deriv_aux rexp_neg_injOn_aux]
      /-
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Complex E
        f : Real → E
        s : Complex
        ⊢ Eq (MeasureTheory.integral (MeasureTheory.MeasureSpace.volume.restrict Set.u …
      -/
      simp [rexp_cexp_aux]
      /-
        🎉 no goals
      -/
    _ = ∫ (u : ℝ), Complex.exp (↑(-2 * π * (u * (s.im / (2 * π)))) * I) •
        (Real.exp (-s.re * u) • f (Real.exp (-u))) := by
      /-
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Complex E
        f : Real → E
        s : Complex
        ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun u => HSMul. …
      -/
      congr
      /-
        case e_f
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Complex E
        f : Real → E
        s : Complex
        ⊢ Eq (fun u => HSMul.hSMul (Complex.exp (HMul.hMul (Neg.neg s) ↑u)) (f (Real.e …
      -/
      ext u
      /-
        case e_f.h
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Complex E
        f : Real → E
        s : Complex
        u : Real
        ⊢ Eq (HSMul.hSMul (Complex.exp (HMul.hMul (Neg.neg s) ↑u)) (f (Real.exp (Neg.n …
      -/
      trans Complex.exp (-s.im * u * I) • (Real.exp (-s.re * u) • f (Real.exp (-u)))
        /-
          E : Type u_1
          inst✝¹ : NormedAddCommGroup E
          inst✝ : NormedSpace Complex E
          f : Real → E
          s : Complex
          u : Real
          ⊢ Eq (HSMul.hSMul (Complex.exp (HMul.hMul (Neg.neg s) ↑u)) (f (Real.exp (Neg.n …
        -/
      · conv => lhs; rw [← re_add_im s]
        /-
          E : Type u_1
          inst✝¹ : NormedAddCommGroup E
          inst✝ : NormedSpace Complex E
          f : Real → E
          s : Complex
          u : Real
          ⊢ Eq (HSMul.hSMul (Complex.exp (HMul.hMul (Neg.neg (HAdd.hAdd (↑s.re) (HMul.hM …
        -/
        rw [neg_add, add_mul, Complex.exp_add, mul_comm, ← smul_eq_mul, smul_assoc]
        /-
          E : Type u_1
          inst✝¹ : NormedAddCommGroup E
          inst✝ : NormedSpace Complex E
          f : Real → E
          s : Complex
          u : Real
          ⊢ Eq (HSMul.hSMul (Complex.exp (HMul.hMul (Neg.neg (HMul.hMul (↑s.im) Complex. …
        -/
        norm_cast
        /-
          E : Type u_1
          inst✝¹ : NormedAddCommGroup E
          inst✝ : NormedSpace Complex E
          f : Real → E
          s : Complex
          u : Real
          ⊢ Eq (HSMul.hSMul (Complex.exp (HMul.hMul (Neg.neg (HMul.hMul (↑s.im) Complex. …
        -/
        push_cast
        /-
          E : Type u_1
          inst✝¹ : NormedAddCommGroup E
          inst✝ : NormedSpace Complex E
          f : Real → E
          s : Complex
          u : Real
          ⊢ Eq (HSMul.hSMul (Complex.exp (HMul.hMul (Neg.neg (HMul.hMul (↑s.im) Complex. …
        -/
        ring_nf
        /-
          🎉 no goals
        -/
      /-
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Complex E
        f : Real → E
        s : Complex
        u : Real
        ⊢ Eq (HSMul.hSMul (Complex.exp (HMul.hMul (HMul.hMul (Neg.neg ↑s.im) ↑u) Compl …
      -/
      congr
      /-
        case e_a.e_z.e_a
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Complex E
        f : Real → E
        s : Complex
        u : Real
        ⊢ Eq (HMul.hMul (Neg.neg ↑s.im) ↑u) ↑(HMul.hMul (HMul.hMul (-2) Real.pi) (HMul …
      -/
      rw [mul_comm (-s.im : ℂ) (u : ℂ), mul_comm (-2 * π)]
      /-
        case e_a.e_z.e_a
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Complex E
        f : Real → E
        s : Complex
        u : Real
        ⊢ Eq (HMul.hMul (↑u) (Neg.neg ↑s.im)) ↑(HMul.hMul (HMul.hMul u (HDiv.hDiv s.im …
      -/
      have : 2 * (π : ℂ) ≠ 0 := by norm_num; exact pi_ne_zero
      /-
        case e_a.e_z.e_a
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Complex E
        f : Real → E
        s : Complex
        u : Real
        this : Ne (HMul.hMul 2 ↑Real.pi) 0
        ⊢ Eq (HMul.hMul (↑u) (Neg.neg ↑s.im)) ↑(HMul.hMul (HMul.hMul u (HDiv.hDiv s.im …
      -/
      field_simp
      /-
        🎉 no goals
      -/
    _ = 𝓕 (fun (u : ℝ) ↦ (Real.exp (-s.re * u) • f (Real.exp (-u)))) (s.im / (2 * π)) := by
      /-
        E : Type u_1
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Complex E
        f : Real → E
        s : Complex
        ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun u => HSMul. …
      -/
      simp [fourierIntegral_eq']
      /-
        🎉 no goals
      -/


theorem mellinInv_eq_fourierIntegralInv (σ : ℝ) (f : ℂ → E) {x : ℝ} (hx : 0 < x) :
    mellinInv σ f x =
    (x : ℂ) ^ (-σ : ℂ) • 𝓕⁻ (fun (y : ℝ) ↦ f (σ + 2 * π * y * I)) (-Real.log x) := calc
  mellinInv σ f x
    = (x : ℂ) ^ (-σ : ℂ) •
      (∫ (y : ℝ), Complex.exp (2 * π * (y * (-Real.log x)) * I) • f (σ + 2 * π * y * I)) := by
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      σ : Real
      f : Complex → E
      x : Real
      hx : LT.lt 0 x
      ⊢ Eq (mellinInv σ f x) (HSMul.hSMul (HPow.hPow (↑x) (Neg.neg ↑σ)) (MeasureTheo …
    -/
    rw [mellinInv, one_div, ← abs_of_pos (show 0 < (2 * π)⁻¹ by norm_num; exact pi_pos)]
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      σ : Real
      f : Complex → E
      x : Real
      hx : LT.lt 0 x
      ⊢ Eq (HSMul.hSMul (abs (Inv.inv (HMul.hMul 2 Real.pi))) (MeasureTheory.integra …
    -/
    have hx0 : (x : ℂ) ≠ 0 := ofReal_ne_zero.mpr (ne_of_gt hx)
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      σ : Real
      f : Complex → E
      x : Real
      hx : LT.lt 0 x
      hx0 : Ne (↑x) 0
      ⊢ Eq (HSMul.hSMul (abs (Inv.inv (HMul.hMul 2 Real.pi))) (MeasureTheory.integra …
    -/
    simp_rw [neg_add, cpow_add _ _ hx0, mul_smul, integral_smul]
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      σ : Real
      f : Complex → E
      x : Real
      hx : LT.lt 0 x
      hx0 : Ne (↑x) 0
      ⊢ Eq (HSMul.hSMul (abs (Inv.inv (HMul.hMul 2 Real.pi))) (HSMul.hSMul (HPow.hPo …
    -/
    rw [smul_comm, ← Measure.integral_comp_mul_left]
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      σ : Real
      f : Complex → E
      x : Real
      hx : LT.lt 0 x
      hx0 : Ne (↑x) 0
      ⊢ Eq (HSMul.hSMul (HPow.hPow (↑x) (Neg.neg ↑σ)) (MeasureTheory.integral Measur …
    -/
    congr! 3
    /-
      case h.e'_6.h.e'_7.h
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      σ : Real
      f : Complex → E
      x : Real
      hx : LT.lt 0 x
      hx0 : Ne (↑x) 0
      x✝ : Real
      ⊢ Eq (HSMul.hSMul (HPow.hPow (↑x) (Neg.neg (HMul.hMul (↑(HMul.hMul (HMul.hMul  …
    -/
    rw [cpow_def_of_ne_zero hx0, ← Complex.ofReal_log hx.le]
    /-
      case h.e'_6.h.e'_7.h
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      σ : Real
      f : Complex → E
      x : Real
      hx : LT.lt 0 x
      hx0 : Ne (↑x) 0
      x✝ : Real
      ⊢ Eq (HSMul.hSMul (Complex.exp (HMul.hMul (↑(Real.log x)) (Neg.neg (HMul.hMul  …
    -/
    push_cast
    /-
      case h.e'_6.h.e'_7.h
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      σ : Real
      f : Complex → E
      x : Real
      hx : LT.lt 0 x
      hx0 : Ne (↑x) 0
      x✝ : Real
      ⊢ Eq (HSMul.hSMul (Complex.exp (HMul.hMul (↑(Real.log x)) (Neg.neg (HMul.hMul  …
    -/
    ring_nf
    /-
      🎉 no goals
    -/
  _ = (x : ℂ) ^ (-σ : ℂ) • 𝓕⁻ (fun (y : ℝ) ↦ f (σ + 2 * π * y * I)) (-Real.log x) := by
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      σ : Real
      f : Complex → E
      x : Real
      hx : LT.lt 0 x
      ⊢ Eq (HSMul.hSMul (HPow.hPow (↑x) (Neg.neg ↑σ)) (MeasureTheory.integral Measur …
    -/
    simp [fourierIntegralInv_eq']
    /-
      🎉 no goals
    -/


/-- The inverse Mellin transform of the Mellin transform applied to `x > 0` is x. -/
theorem mellin_inversion (σ : ℝ) (f : ℝ → E) {x : ℝ} (hx : 0 < x) (hf : MellinConvergent f σ)
           /-
             E : Type u_1
             inst✝² : NormedAddCommGroup E
             inst✝¹ : NormedSpace Complex E
             inst✝ : CompleteSpace E
             σ : Real
             f : Real → E
             x : Real
             hx : LT.lt 0 x
             hf : MellinConvergent f ↑σ
             ⊢ MeasureTheory.Measure Real
           -/
    (hFf : VerticalIntegrable (mellin f) σ) (hfx : ContinuousAt f x) :
           /-
             🎉 no goals
           -/
    mellinInv σ (mellin f) x = f x := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    inst✝ : CompleteSpace E
    σ : Real
    f : Real → E
    x : Real
    hx : LT.lt 0 x
    hf : MellinConvergent f ↑σ
    hFf : Complex.VerticalIntegrable (mellin f) σ MeasureTheory.MeasureSpace.volume
    hfx : ContinuousAt f x
    ⊢ Eq (mellinInv σ (mellin f) x) (f x)
  -/
  let g := fun (u : ℝ) => Real.exp (-σ * u) • f (Real.exp (-u))
  replace hf : Integrable g := by
    rw [MellinConvergent, ← rexp_neg_image_aux, integrableOn_image_iff_integrableOn_abs_deriv_smul
      MeasurableSet.univ rexp_neg_deriv_aux rexp_neg_injOn_aux] at hf
    replace hf : Integrable fun (x : ℝ) ↦ cexp (-↑σ * ↑x) • f (rexp (-x)) := by
      simpa [rexp_cexp_aux] using hf
    norm_cast at hf
  replace hFf : Integrable (𝓕 g) := by
    have h2π : 2 * π ≠ 0 := by norm_num; exact pi_ne_zero
    have : Integrable (𝓕 (fun u ↦ rexp (-(σ * u)) • f (rexp (-u)))) := by
      simpa [mellin_eq_fourierIntegral, mul_div_cancel_right₀ _ h2π] using hFf.comp_mul_right' h2π
    simp_rw [neg_mul_eq_neg_mul] at this
    exact this
  replace hfx : ContinuousAt g (-Real.log x) := by
    refine ContinuousAt.smul (by fun_prop) (ContinuousAt.comp ?_ (by fun_prop))
    simpa [Real.exp_log hx] using hfx
  calc
    mellinInv σ (mellin f) x
      = mellinInv σ (fun s ↦ 𝓕 g (s.im / (2 * π))) x := by
      simp [g, mellinInv, mellin_eq_fourierIntegral]
    _ = (x : ℂ) ^ (-σ : ℂ) • g (-Real.log x) := by
      rw [mellinInv_eq_fourierIntegralInv _ _ hx, ← hf.fourier_inversion hFf hfx]
      simp [mul_div_cancel_left₀ _ (show 2 * π ≠ 0 by norm_num; exact pi_ne_zero)]
    _ = (x : ℂ) ^ (-σ : ℂ) • rexp (σ * Real.log x) • f (rexp (Real.log x)) := by simp [g]
    _ = f x := by
      norm_cast
      rw [mul_comm σ, ← rpow_def_of_pos hx, Real.exp_log hx, ← Complex.ofReal_cpow hx.le]
      norm_cast
      rw [← smul_assoc, smul_eq_mul, Real.rpow_neg hx.le,
        inv_mul_cancel₀ (ne_of_gt (rpow_pos_of_pos hx σ)), one_smul]

