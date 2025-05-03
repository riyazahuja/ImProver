                                          /-
                                            V : Type u_1
                                            E : Type u_2
                                            inst✝⁶ : NormedAddCommGroup V
                                            inst✝⁵ : InnerProductSpace Real V
                                            inst✝⁴ : MeasurableSpace V
                                            inst✝³ : BorelSpace V
                                            inst✝² : FiniteDimensional Real V
                                            inst✝¹ : NormedAddCommGroup E
                                            inst✝ : NormedSpace Complex E
                                            f : V → E
                                            ⊢ MeasureTheory.Measure V
                                          -/
lemma tendsto_integral_cexp_sq_smul (hf : Integrable f) :
                                          /-
                                            🎉 no goals
                                          -/
    Tendsto (fun (c : ℝ) ↦ (∫ v : V, cexp (- c⁻¹ * ‖v‖^2) • f v))
      atTop (𝓝 (∫ v : V, f v)) := by
  /-
    V : Type u_1
    E : Type u_2
    inst✝⁶ : NormedAddCommGroup V
    inst✝⁵ : InnerProductSpace Real V
    inst✝⁴ : MeasurableSpace V
    inst✝³ : BorelSpace V
    inst✝² : FiniteDimensional Real V
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : V → E
    hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
    ⊢ Filter.Tendsto (fun c => MeasureTheory.integral MeasureTheory.MeasureSpace.v …
  -/
  apply tendsto_integral_filter_of_dominated_convergence _ _ _ hf.norm
    /-
      V : Type u_1
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup V
      inst✝⁵ : InnerProductSpace Real V
      inst✝⁴ : MeasurableSpace V
      inst✝³ : BorelSpace V
      inst✝² : FiniteDimensional Real V
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : V → E
      hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
      ⊢ Filter.Eventually (fun a => Filter.Tendsto (fun n => HSMul.hSMul (Complex.ex …
    -/
  · filter_upwards with v
    /-
      case h
      V : Type u_1
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup V
      inst✝⁵ : InnerProductSpace Real V
      inst✝⁴ : MeasurableSpace V
      inst✝³ : BorelSpace V
      inst✝² : FiniteDimensional Real V
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : V → E
      hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
      v : V
      ⊢ Filter.Tendsto (fun n => HSMul.hSMul (Complex.exp (HMul.hMul (Neg.neg ↑(Inv. …
    -/
    nth_rewrite 2 [show f v = cexp (- (0 : ℝ) * ‖v‖^2) • f v by simp]
    /-
      case h
      V : Type u_1
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup V
      inst✝⁵ : InnerProductSpace Real V
      inst✝⁴ : MeasurableSpace V
      inst✝³ : BorelSpace V
      inst✝² : FiniteDimensional Real V
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : V → E
      hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
      v : V
      ⊢ Filter.Tendsto (fun n => HSMul.hSMul (Complex.exp (HMul.hMul (Neg.neg ↑(Inv. …
    -/
    apply (Tendsto.cexp _).smul_const
    /-
      V : Type u_1
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup V
      inst✝⁵ : InnerProductSpace Real V
      inst✝⁴ : MeasurableSpace V
      inst✝³ : BorelSpace V
      inst✝² : FiniteDimensional Real V
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : V → E
      hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
      v : V
      ⊢ Filter.Tendsto (fun x => HMul.hMul (Neg.neg ↑(Inv.inv x)) (HPow.hPow (↑(Norm …
    -/
    exact tendsto_inv_atTop_zero.ofReal.neg.mul_const _
    /-
      🎉 no goals
    -/
  · filter_upwards with c using
      AEStronglyMeasurable.smul (Continuous.aestronglyMeasurable (by fun_prop)) hf.1
    /-
      V : Type u_1
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup V
      inst✝⁵ : InnerProductSpace Real V
      inst✝⁴ : MeasurableSpace V
      inst✝³ : BorelSpace V
      inst✝² : FiniteDimensional Real V
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : V → E
      hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
      ⊢ Filter.Eventually (fun n => Filter.Eventually (fun a => LE.le (Norm.norm (HS …
    -/
  · filter_upwards [Ici_mem_atTop (0 : ℝ)] with c (hc : 0 ≤ c)
    /-
      case h
      V : Type u_1
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup V
      inst✝⁵ : InnerProductSpace Real V
      inst✝⁴ : MeasurableSpace V
      inst✝³ : BorelSpace V
      inst✝² : FiniteDimensional Real V
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : V → E
      hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
      c : Real
      hc : LE.le 0 c
      ⊢ Filter.Eventually (fun a => LE.le (Norm.norm (HSMul.hSMul (Complex.exp (HMul …
    -/
    filter_upwards with v
    /-
      case h.h
      V : Type u_1
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup V
      inst✝⁵ : InnerProductSpace Real V
      inst✝⁴ : MeasurableSpace V
      inst✝³ : BorelSpace V
      inst✝² : FiniteDimensional Real V
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : V → E
      hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
      c : Real
      hc : LE.le 0 c
      v : V
      ⊢ LE.le (Norm.norm (HSMul.hSMul (Complex.exp (HMul.hMul (Neg.neg ↑(Inv.inv c)) …
    -/
    simp only [ofReal_inv, neg_mul, norm_smul, Complex.norm_eq_abs]
    /-
      case h.h
      V : Type u_1
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup V
      inst✝⁵ : InnerProductSpace Real V
      inst✝⁴ : MeasurableSpace V
      inst✝³ : BorelSpace V
      inst✝² : FiniteDimensional Real V
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : V → E
      hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
      c : Real
      hc : LE.le 0 c
      v : V
      ⊢ LE.le (HMul.hMul (Complex.abs (Complex.exp (Neg.neg (HMul.hMul (Inv.inv ↑c)  …
    -/
    norm_cast
    /-
      case h.h
      V : Type u_1
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup V
      inst✝⁵ : InnerProductSpace Real V
      inst✝⁴ : MeasurableSpace V
      inst✝³ : BorelSpace V
      inst✝² : FiniteDimensional Real V
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : V → E
      hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
      c : Real
      hc : LE.le 0 c
      v : V
      ⊢ LE.le (HMul.hMul (abs (Real.exp (Neg.neg (HMul.hMul (Inv.inv c) (HPow.hPow ( …
    -/
    conv_rhs => rw [← one_mul (‖f v‖)]
    /-
      case h.h
      V : Type u_1
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup V
      inst✝⁵ : InnerProductSpace Real V
      inst✝⁴ : MeasurableSpace V
      inst✝³ : BorelSpace V
      inst✝² : FiniteDimensional Real V
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : V → E
      hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
      c : Real
      hc : LE.le 0 c
      v : V
      ⊢ LE.le (HMul.hMul (abs (Real.exp (Neg.neg (HMul.hMul (Inv.inv c) (HPow.hPow ( …
    -/
    gcongr
    /-
      case h.h.h
      V : Type u_1
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup V
      inst✝⁵ : InnerProductSpace Real V
      inst✝⁴ : MeasurableSpace V
      inst✝³ : BorelSpace V
      inst✝² : FiniteDimensional Real V
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : V → E
      hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
      c : Real
      hc : LE.le 0 c
      v : V
      ⊢ LE.le (abs (Real.exp (Neg.neg (HMul.hMul (Inv.inv c) (HPow.hPow (Norm.norm v …
    -/
    simp only [abs_exp, exp_le_one_iff, Left.neg_nonpos_iff]
    /-
      case h.h.h
      V : Type u_1
      E : Type u_2
      inst✝⁶ : NormedAddCommGroup V
      inst✝⁵ : InnerProductSpace Real V
      inst✝⁴ : MeasurableSpace V
      inst✝³ : BorelSpace V
      inst✝² : FiniteDimensional Real V
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : V → E
      hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
      c : Real
      hc : LE.le 0 c
      v : V
      ⊢ LE.le 0 (HMul.hMul (Inv.inv c) (HPow.hPow (Norm.norm v) 2))
    -/
    positivity
    /-
      🎉 no goals
    -/


                                           /-
                                             V : Type u_1
                                             E : Type u_2
                                             inst✝⁷ : NormedAddCommGroup V
                                             inst✝⁶ : InnerProductSpace Real V
                                             inst✝⁵ : MeasurableSpace V
                                             inst✝⁴ : BorelSpace V
                                             inst✝³ : FiniteDimensional Real V
                                             inst✝² : NormedAddCommGroup E
                                             inst✝¹ : NormedSpace Complex E
                                             f : V → E
                                             inst✝ : CompleteSpace E
                                             ⊢ MeasureTheory.Measure V
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
lemma tendsto_integral_gaussian_smul (hf : Integrable f) (h'f : Integrable (𝓕 f)) (v : V) :
                                                                /-
                                                                  🎉 no goals
                                                                -/
    Tendsto (fun (c : ℝ) ↦
      ∫ w : V, ((π * c) ^ (finrank ℝ V / 2 : ℂ) * cexp (-π ^ 2 * c * ‖v - w‖ ^ 2)) • f w)
    atTop (𝓝 (𝓕⁻ (𝓕 f) v)) := by
  have A : Tendsto (fun (c : ℝ) ↦ (∫ w : V, cexp (- c⁻¹ * ‖w‖^2 + 2 * π * I * ⟪v, w⟫)
       • (𝓕 f) w)) atTop (𝓝 (𝓕⁻ (𝓕 f) v)) := by
    have : Integrable (fun w ↦ 𝐞 ⟪w, v⟫ • (𝓕 f) w) := by
      have B : Continuous fun p : V × V => (- innerₗ V) p.1 p.2 := continuous_inner.neg
      simpa using
        (VectorFourier.fourierIntegral_convergent_iff Real.continuous_fourierChar B v).2 h'f
    convert tendsto_integral_cexp_sq_smul this using 4 with c w
    · rw [Submonoid.smul_def, Real.fourierChar_apply, smul_smul, ← Complex.exp_add, real_inner_comm]
      congr 3
      simp only [ofReal_mul, ofReal_ofNat]
      ring
    · simp [fourierIntegralInv_eq]
  have B : Tendsto (fun (c : ℝ) ↦ (∫ w : V,
        𝓕 (fun w ↦ cexp (- c⁻¹ * ‖w‖^2 + 2 * π * I * ⟪v, w⟫)) w • f w)) atTop
      (𝓝 (𝓕⁻ (𝓕 f) v)) := by
    apply A.congr'
    filter_upwards [Ioi_mem_atTop 0] with c (hc : 0 < c)
    have J : Integrable (fun w ↦ cexp (- c⁻¹ * ‖w‖^2 + 2 * π * I * ⟪v, w⟫)) :=
      GaussianFourier.integrable_cexp_neg_mul_sq_norm_add (by simpa) _ _
    simpa using (VectorFourier.integral_fourierIntegral_smul_eq_flip (L := innerₗ V)
      Real.continuous_fourierChar continuous_inner J hf).symm
  /-
    V : Type u_1
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup V
    inst✝⁶ : InnerProductSpace Real V
    inst✝⁵ : MeasurableSpace V
    inst✝⁴ : BorelSpace V
    inst✝³ : FiniteDimensional Real V
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    f : V → E
    inst✝ : CompleteSpace E
    hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
    h'f : MeasureTheory.Integrable (Real.fourierIntegral f) MeasureTheory.MeasureS …
    v : V
    A : Filter.Tendsto (fun c => MeasureTheory.integral MeasureTheory.MeasureSpace …
    B : Filter.Tendsto (fun c => MeasureTheory.integral MeasureTheory.MeasureSpace …
    ⊢ Filter.Tendsto (fun c => MeasureTheory.integral MeasureTheory.MeasureSpace.v …
  -/
  apply B.congr'
  /-
    V : Type u_1
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup V
    inst✝⁶ : InnerProductSpace Real V
    inst✝⁵ : MeasurableSpace V
    inst✝⁴ : BorelSpace V
    inst✝³ : FiniteDimensional Real V
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    f : V → E
    inst✝ : CompleteSpace E
    hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
    h'f : MeasureTheory.Integrable (Real.fourierIntegral f) MeasureTheory.MeasureS …
    v : V
    A : Filter.Tendsto (fun c => MeasureTheory.integral MeasureTheory.MeasureSpace …
    B : Filter.Tendsto (fun c => MeasureTheory.integral MeasureTheory.MeasureSpace …
    ⊢ Filter.atTop.EventuallyEq (fun c => MeasureTheory.integral MeasureTheory.Mea …
  -/
  filter_upwards [Ioi_mem_atTop 0] with c (hc : 0 < c)
  /-
    case h
    V : Type u_1
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup V
    inst✝⁶ : InnerProductSpace Real V
    inst✝⁵ : MeasurableSpace V
    inst✝⁴ : BorelSpace V
    inst✝³ : FiniteDimensional Real V
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    f : V → E
    inst✝ : CompleteSpace E
    hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
    h'f : MeasureTheory.Integrable (Real.fourierIntegral f) MeasureTheory.MeasureS …
    v : V
    A : Filter.Tendsto (fun c => MeasureTheory.integral MeasureTheory.MeasureSpace …
    B : Filter.Tendsto (fun c => MeasureTheory.integral MeasureTheory.MeasureSpace …
    c : Real
    hc : LT.lt 0 c
    ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun w => HSMul. …
  -/
  congr with w
  /-
    case h.e_f.h
    V : Type u_1
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup V
    inst✝⁶ : InnerProductSpace Real V
    inst✝⁵ : MeasurableSpace V
    inst✝⁴ : BorelSpace V
    inst✝³ : FiniteDimensional Real V
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    f : V → E
    inst✝ : CompleteSpace E
    hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
    h'f : MeasureTheory.Integrable (Real.fourierIntegral f) MeasureTheory.MeasureS …
    v : V
    A : Filter.Tendsto (fun c => MeasureTheory.integral MeasureTheory.MeasureSpace …
    B : Filter.Tendsto (fun c => MeasureTheory.integral MeasureTheory.MeasureSpace …
    c : Real
    hc : LT.lt 0 c
    w : V
    ⊢ Eq (HSMul.hSMul (Real.fourierIntegral (fun w => Complex.exp (HAdd.hAdd (HMul …
  -/
  rw [fourierIntegral_gaussian_innerProductSpace' (by simpa)]
  /-
    case h.e_f.h
    V : Type u_1
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup V
    inst✝⁶ : InnerProductSpace Real V
    inst✝⁵ : MeasurableSpace V
    inst✝⁴ : BorelSpace V
    inst✝³ : FiniteDimensional Real V
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    f : V → E
    inst✝ : CompleteSpace E
    hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
    h'f : MeasureTheory.Integrable (Real.fourierIntegral f) MeasureTheory.MeasureS …
    v : V
    A : Filter.Tendsto (fun c => MeasureTheory.integral MeasureTheory.MeasureSpace …
    B : Filter.Tendsto (fun c => MeasureTheory.integral MeasureTheory.MeasureSpace …
    c : Real
    hc : LT.lt 0 c
    w : V
    ⊢ Eq (HSMul.hSMul (HMul.hMul (HPow.hPow (HDiv.hDiv ↑Real.pi ↑(Inv.inv c)) (HDi …
  -/
  congr
    /-
      case h.e_f.h.e_a.e_a.e_a
      V : Type u_1
      E : Type u_2
      inst✝⁷ : NormedAddCommGroup V
      inst✝⁶ : InnerProductSpace Real V
      inst✝⁵ : MeasurableSpace V
      inst✝⁴ : BorelSpace V
      inst✝³ : FiniteDimensional Real V
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Complex E
      f : V → E
      inst✝ : CompleteSpace E
      hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
      h'f : MeasureTheory.Integrable (Real.fourierIntegral f) MeasureTheory.MeasureS …
      v : V
      A : Filter.Tendsto (fun c => MeasureTheory.integral MeasureTheory.MeasureSpace …
      B : Filter.Tendsto (fun c => MeasureTheory.integral MeasureTheory.MeasureSpace …
      c : Real
      hc : LT.lt 0 c
      w : V
      ⊢ Eq (HDiv.hDiv ↑Real.pi ↑(Inv.inv c)) (HMul.hMul ↑Real.pi ↑c)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h.e_f.h.e_a.e_a.e_z
      V : Type u_1
      E : Type u_2
      inst✝⁷ : NormedAddCommGroup V
      inst✝⁶ : InnerProductSpace Real V
      inst✝⁵ : MeasurableSpace V
      inst✝⁴ : BorelSpace V
      inst✝³ : FiniteDimensional Real V
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Complex E
      f : V → E
      inst✝ : CompleteSpace E
      hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
      h'f : MeasureTheory.Integrable (Real.fourierIntegral f) MeasureTheory.MeasureS …
      v : V
      A : Filter.Tendsto (fun c => MeasureTheory.integral MeasureTheory.MeasureSpace …
      B : Filter.Tendsto (fun c => MeasureTheory.integral MeasureTheory.MeasureSpace …
      c : Real
      hc : LT.lt 0 c
      w : V
      ⊢ Eq (HDiv.hDiv (HMul.hMul (Neg.neg (HPow.hPow (↑Real.pi) 2)) (HPow.hPow (↑(No …
    -/
  · simp; ring
          /-
            🎉 no goals
          -/


                                            /-
                                              V : Type u_1
                                              E : Type u_2
                                              inst✝⁷ : NormedAddCommGroup V
                                              inst✝⁶ : InnerProductSpace Real V
                                              inst✝⁵ : MeasurableSpace V
                                              inst✝⁴ : BorelSpace V
                                              inst✝³ : FiniteDimensional Real V
                                              inst✝² : NormedAddCommGroup E
                                              inst✝¹ : NormedSpace Complex E
                                              f : V → E
                                              inst✝ : CompleteSpace E
                                              ⊢ MeasureTheory.Measure V
                                            -/
lemma tendsto_integral_gaussian_smul' (hf : Integrable f) {v : V} (h'f : ContinuousAt f v) :
                                            /-
                                              🎉 no goals
                                            -/
    Tendsto (fun (c : ℝ) ↦
      ∫ w : V, ((π * c : ℂ) ^ (finrank ℝ V / 2 : ℂ) * cexp (-π ^ 2 * c * ‖v - w‖ ^ 2)) • f w)
    atTop (𝓝 (f v)) := by
  /-
    V : Type u_1
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup V
    inst✝⁶ : InnerProductSpace Real V
    inst✝⁵ : MeasurableSpace V
    inst✝⁴ : BorelSpace V
    inst✝³ : FiniteDimensional Real V
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    f : V → E
    inst✝ : CompleteSpace E
    hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
    v : V
    h'f : ContinuousAt f v
    ⊢ Filter.Tendsto (fun c => MeasureTheory.integral MeasureTheory.MeasureSpace.v …
  -/
  let φ : V → ℝ := fun w ↦ π ^ (finrank ℝ V / 2 : ℝ) * Real.exp (-π^2 * ‖w‖^2)
  have A : Tendsto (fun (c : ℝ) ↦ ∫ w : V, (c ^ finrank ℝ V * φ (c • (v - w))) • f w)
      atTop (𝓝 (f v)) := by
    apply tendsto_integral_comp_smul_smul_of_integrable'
    · exact fun x ↦ by positivity
    · rw [integral_mul_left, GaussianFourier.integral_rexp_neg_mul_sq_norm (by positivity)]
      nth_rewrite 2 [← pow_one π]
      rw [← rpow_natCast, ← rpow_natCast, ← rpow_sub pi_pos, ← rpow_mul pi_nonneg,
        ← rpow_add pi_pos]
      ring_nf
      exact rpow_zero _
    · have A : Tendsto (fun (w : V) ↦ π^2 * ‖w‖^2) (cobounded V) atTop := by
        rw [tendsto_const_mul_atTop_of_pos (by positivity)]
        apply (tendsto_pow_atTop two_ne_zero).comp tendsto_norm_cobounded_atTop
      have B := tendsto_rpow_mul_exp_neg_mul_atTop_nhds_zero (finrank ℝ V / 2) 1
        zero_lt_one |>.comp A |>.const_mul (π ^ (-finrank ℝ V / 2 : ℝ))
      rw [mul_zero] at B
      convert B using 2 with x
      simp only [neg_mul, one_mul, Function.comp_apply, ← mul_assoc, ← rpow_natCast, φ]
      congr 1
      rw [mul_rpow (by positivity) (by positivity), ← rpow_mul pi_nonneg,
        ← rpow_mul (norm_nonneg _), ← mul_assoc, ← rpow_add pi_pos, mul_comm]
      congr <;> ring
    · exact hf
    · exact h'f
  have B : Tendsto
      (fun (c : ℝ) ↦ ∫ w : V, ((c^(1/2 : ℝ)) ^ finrank ℝ V * φ ((c^(1/2 : ℝ)) • (v - w))) • f w)
      atTop (𝓝 (f v)) :=
    A.comp (tendsto_rpow_atTop (by norm_num))
  /-
    V : Type u_1
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup V
    inst✝⁶ : InnerProductSpace Real V
    inst✝⁵ : MeasurableSpace V
    inst✝⁴ : BorelSpace V
    inst✝³ : FiniteDimensional Real V
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    f : V → E
    inst✝ : CompleteSpace E
    hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
    v : V
    h'f : ContinuousAt f v
    φ : V → Real := fun w => HMul.hMul (HPow.hPow Real.pi (HDiv.hDiv (↑(Module.fin …
    A : Filter.Tendsto (fun c => MeasureTheory.integral MeasureTheory.MeasureSpace …
    B : Filter.Tendsto (fun c => MeasureTheory.integral MeasureTheory.MeasureSpace …
    ⊢ Filter.Tendsto (fun c => MeasureTheory.integral MeasureTheory.MeasureSpace.v …
  -/
  apply B.congr'
  /-
    V : Type u_1
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup V
    inst✝⁶ : InnerProductSpace Real V
    inst✝⁵ : MeasurableSpace V
    inst✝⁴ : BorelSpace V
    inst✝³ : FiniteDimensional Real V
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    f : V → E
    inst✝ : CompleteSpace E
    hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
    v : V
    h'f : ContinuousAt f v
    φ : V → Real := fun w => HMul.hMul (HPow.hPow Real.pi (HDiv.hDiv (↑(Module.fin …
    A : Filter.Tendsto (fun c => MeasureTheory.integral MeasureTheory.MeasureSpace …
    B : Filter.Tendsto (fun c => MeasureTheory.integral MeasureTheory.MeasureSpace …
    ⊢ Filter.atTop.EventuallyEq (fun c => MeasureTheory.integral MeasureTheory.Mea …
  -/
  filter_upwards [Ioi_mem_atTop 0] with c (hc : 0 < c)
  /-
    case h
    V : Type u_1
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup V
    inst✝⁶ : InnerProductSpace Real V
    inst✝⁵ : MeasurableSpace V
    inst✝⁴ : BorelSpace V
    inst✝³ : FiniteDimensional Real V
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    f : V → E
    inst✝ : CompleteSpace E
    hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
    v : V
    h'f : ContinuousAt f v
    φ : V → Real := fun w => HMul.hMul (HPow.hPow Real.pi (HDiv.hDiv (↑(Module.fin …
    A : Filter.Tendsto (fun c => MeasureTheory.integral MeasureTheory.MeasureSpace …
    B : Filter.Tendsto (fun c => MeasureTheory.integral MeasureTheory.MeasureSpace …
    c : Real
    hc : LT.lt 0 c
    ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun w => HSMul. …
  -/
  congr with w
  /-
    case h.e_f.h
    V : Type u_1
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup V
    inst✝⁶ : InnerProductSpace Real V
    inst✝⁵ : MeasurableSpace V
    inst✝⁴ : BorelSpace V
    inst✝³ : FiniteDimensional Real V
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    f : V → E
    inst✝ : CompleteSpace E
    hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
    v : V
    h'f : ContinuousAt f v
    φ : V → Real := fun w => HMul.hMul (HPow.hPow Real.pi (HDiv.hDiv (↑(Module.fin …
    A : Filter.Tendsto (fun c => MeasureTheory.integral MeasureTheory.MeasureSpace …
    B : Filter.Tendsto (fun c => MeasureTheory.integral MeasureTheory.MeasureSpace …
    c : Real
    hc : LT.lt 0 c
    w : V
    ⊢ Eq (HSMul.hSMul (HMul.hMul (HPow.hPow (HPow.hPow c (1 / 2)) (Module.finrank  …
  -/
  rw [← coe_smul]
  /-
    case h.e_f.h
    V : Type u_1
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup V
    inst✝⁶ : InnerProductSpace Real V
    inst✝⁵ : MeasurableSpace V
    inst✝⁴ : BorelSpace V
    inst✝³ : FiniteDimensional Real V
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    f : V → E
    inst✝ : CompleteSpace E
    hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
    v : V
    h'f : ContinuousAt f v
    φ : V → Real := fun w => HMul.hMul (HPow.hPow Real.pi (HDiv.hDiv (↑(Module.fin …
    A : Filter.Tendsto (fun c => MeasureTheory.integral MeasureTheory.MeasureSpace …
    B : Filter.Tendsto (fun c => MeasureTheory.integral MeasureTheory.MeasureSpace …
    c : Real
    hc : LT.lt 0 c
    w : V
    ⊢ Eq (HSMul.hSMul (↑(HMul.hMul (HPow.hPow (HPow.hPow c (1 / 2)) (Module.finran …
  -/
  congr
  /-
    case h.e_f.h.e_a
    V : Type u_1
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup V
    inst✝⁶ : InnerProductSpace Real V
    inst✝⁵ : MeasurableSpace V
    inst✝⁴ : BorelSpace V
    inst✝³ : FiniteDimensional Real V
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    f : V → E
    inst✝ : CompleteSpace E
    hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
    v : V
    h'f : ContinuousAt f v
    φ : V → Real := fun w => HMul.hMul (HPow.hPow Real.pi (HDiv.hDiv (↑(Module.fin …
    A : Filter.Tendsto (fun c => MeasureTheory.integral MeasureTheory.MeasureSpace …
    B : Filter.Tendsto (fun c => MeasureTheory.integral MeasureTheory.MeasureSpace …
    c : Real
    hc : LT.lt 0 c
    w : V
    ⊢ Eq (↑(HMul.hMul (HPow.hPow (HPow.hPow c (1 / 2)) (Module.finrank Real V)) (φ …
  -/
  rw [ofReal_mul, ofReal_mul, ofReal_exp, ← mul_assoc]
  /-
    case h.e_f.h.e_a
    V : Type u_1
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup V
    inst✝⁶ : InnerProductSpace Real V
    inst✝⁵ : MeasurableSpace V
    inst✝⁴ : BorelSpace V
    inst✝³ : FiniteDimensional Real V
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    f : V → E
    inst✝ : CompleteSpace E
    hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
    v : V
    h'f : ContinuousAt f v
    φ : V → Real := fun w => HMul.hMul (HPow.hPow Real.pi (HDiv.hDiv (↑(Module.fin …
    A : Filter.Tendsto (fun c => MeasureTheory.integral MeasureTheory.MeasureSpace …
    B : Filter.Tendsto (fun c => MeasureTheory.integral MeasureTheory.MeasureSpace …
    c : Real
    hc : LT.lt 0 c
    w : V
    ⊢ Eq (HMul.hMul (HMul.hMul ↑(HPow.hPow (HPow.hPow c (1 / 2)) (Module.finrank R …
  -/
  congr
  · rw [mul_cpow_ofReal_nonneg pi_nonneg hc.le, ← rpow_natCast, ← rpow_mul hc.le, mul_comm,
      ofReal_cpow pi_nonneg, ofReal_cpow hc.le]
    /-
      case h.e_f.h.e_a.e_a
      V : Type u_1
      E : Type u_2
      inst✝⁷ : NormedAddCommGroup V
      inst✝⁶ : InnerProductSpace Real V
      inst✝⁵ : MeasurableSpace V
      inst✝⁴ : BorelSpace V
      inst✝³ : FiniteDimensional Real V
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Complex E
      f : V → E
      inst✝ : CompleteSpace E
      hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
      v : V
      h'f : ContinuousAt f v
      φ : V → Real := fun w => HMul.hMul (HPow.hPow Real.pi (HDiv.hDiv (↑(Module.fin …
      A : Filter.Tendsto (fun c => MeasureTheory.integral MeasureTheory.MeasureSpace …
      B : Filter.Tendsto (fun c => MeasureTheory.integral MeasureTheory.MeasureSpace …
      c : Real
      hc : LT.lt 0 c
      w : V
      ⊢ Eq (HMul.hMul (HPow.hPow ↑Real.pi ↑(HDiv.hDiv (↑(Module.finrank Real V)) 2)) …
    -/
    simp [div_eq_inv_mul]
    /-
      🎉 no goals
    -/
    /-
      case h.e_f.h.e_a.e_a.e_z
      V : Type u_1
      E : Type u_2
      inst✝⁷ : NormedAddCommGroup V
      inst✝⁶ : InnerProductSpace Real V
      inst✝⁵ : MeasurableSpace V
      inst✝⁴ : BorelSpace V
      inst✝³ : FiniteDimensional Real V
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Complex E
      f : V → E
      inst✝ : CompleteSpace E
      hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
      v : V
      h'f : ContinuousAt f v
      φ : V → Real := fun w => HMul.hMul (HPow.hPow Real.pi (HDiv.hDiv (↑(Module.fin …
      A : Filter.Tendsto (fun c => MeasureTheory.integral MeasureTheory.MeasureSpace …
      B : Filter.Tendsto (fun c => MeasureTheory.integral MeasureTheory.MeasureSpace …
      c : Real
      hc : LT.lt 0 c
      w : V
      ⊢ Eq (↑(HMul.hMul (Neg.neg (HPow.hPow Real.pi 2)) (HPow.hPow (Norm.norm (HSMul …
    -/
  · norm_cast
    simp only [one_div, norm_smul, Real.norm_eq_abs, mul_pow, _root_.sq_abs, neg_mul, neg_inj,
      ← rpow_natCast, ← rpow_mul hc.le, mul_assoc]
    /-
      case h.e_f.h.e_a.e_a.e_z
      V : Type u_1
      E : Type u_2
      inst✝⁷ : NormedAddCommGroup V
      inst✝⁶ : InnerProductSpace Real V
      inst✝⁵ : MeasurableSpace V
      inst✝⁴ : BorelSpace V
      inst✝³ : FiniteDimensional Real V
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Complex E
      f : V → E
      inst✝ : CompleteSpace E
      hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
      v : V
      h'f : ContinuousAt f v
      φ : V → Real := fun w => HMul.hMul (HPow.hPow Real.pi (HDiv.hDiv (↑(Module.fin …
      A : Filter.Tendsto (fun c => MeasureTheory.integral MeasureTheory.MeasureSpace …
      B : Filter.Tendsto (fun c => MeasureTheory.integral MeasureTheory.MeasureSpace …
      c : Real
      hc : LT.lt 0 c
      w : V
      ⊢ Eq (HMul.hMul (HPow.hPow Real.pi ↑2) (HMul.hMul (HPow.hPow c (HMul.hMul (Inv …
    -/
    norm_num
    /-
      🎉 no goals
    -/


/-- **Fourier inversion formula**: If a function `f` on a finite-dimensional real inner product
space is integrable, and its Fourier transform `𝓕 f` is also integrable, then `𝓕⁻ (𝓕 f) = f` at
continuity points of `f`. -/
theorem MeasureTheory.Integrable.fourier_inversion
          /-
            V : Type u_1
            E : Type u_2
            inst✝⁷ : NormedAddCommGroup V
            inst✝⁶ : InnerProductSpace Real V
            inst✝⁵ : MeasurableSpace V
            inst✝⁴ : BorelSpace V
            inst✝³ : FiniteDimensional Real V
            inst✝² : NormedAddCommGroup E
            inst✝¹ : NormedSpace Complex E
            f : V → E
            inst✝ : CompleteSpace E
            ⊢ MeasureTheory.Measure V
          -/
          /-
            🎉 no goals
          -/
    (hf : Integrable f) (h'f : Integrable (𝓕 f)) {v : V}
                               /-
                                 🎉 no goals
                               -/
    (hv : ContinuousAt f v) : 𝓕⁻ (𝓕 f) v = f v :=
  tendsto_nhds_unique (Real.tendsto_integral_gaussian_smul hf h'f v)
    (Real.tendsto_integral_gaussian_smul' hf hv)


/-- **Fourier inversion formula**: If a function `f` on a finite-dimensional real inner product
space is continuous, integrable, and its Fourier transform `𝓕 f` is also integrable,
then `𝓕⁻ (𝓕 f) = f`. -/
theorem Continuous.fourier_inversion (h : Continuous f)
          /-
            V : Type u_1
            E : Type u_2
            inst✝⁷ : NormedAddCommGroup V
            inst✝⁶ : InnerProductSpace Real V
            inst✝⁵ : MeasurableSpace V
            inst✝⁴ : BorelSpace V
            inst✝³ : FiniteDimensional Real V
            inst✝² : NormedAddCommGroup E
            inst✝¹ : NormedSpace Complex E
            f : V → E
            inst✝ : CompleteSpace E
            h : Continuous f
            ⊢ MeasureTheory.Measure V
          -/
          /-
            🎉 no goals
          -/
    (hf : Integrable f) (h'f : Integrable (𝓕 f)) :
                               /-
                                 🎉 no goals
                               -/
    𝓕⁻ (𝓕 f) = f := by
  /-
    V : Type u_1
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup V
    inst✝⁶ : InnerProductSpace Real V
    inst✝⁵ : MeasurableSpace V
    inst✝⁴ : BorelSpace V
    inst✝³ : FiniteDimensional Real V
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    f : V → E
    inst✝ : CompleteSpace E
    h : Continuous f
    hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
    h'f : MeasureTheory.Integrable (Real.fourierIntegral f) MeasureTheory.MeasureS …
    ⊢ Eq (Real.fourierIntegralInv (Real.fourierIntegral f)) f
  -/
  ext v
  /-
    case h
    V : Type u_1
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup V
    inst✝⁶ : InnerProductSpace Real V
    inst✝⁵ : MeasurableSpace V
    inst✝⁴ : BorelSpace V
    inst✝³ : FiniteDimensional Real V
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    f : V → E
    inst✝ : CompleteSpace E
    h : Continuous f
    hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
    h'f : MeasureTheory.Integrable (Real.fourierIntegral f) MeasureTheory.MeasureS …
    v : V
    ⊢ Eq (Real.fourierIntegralInv (Real.fourierIntegral f) v) (f v)
  -/
  exact hf.fourier_inversion h'f h.continuousAt
  /-
    🎉 no goals
  -/


/-- **Fourier inversion formula**: If a function `f` on a finite-dimensional real inner product
space is integrable, and its Fourier transform `𝓕 f` is also integrable, then `𝓕 (𝓕⁻ f) = f` at
continuity points of `f`. -/
theorem MeasureTheory.Integrable.fourier_inversion_inv
          /-
            V : Type u_1
            E : Type u_2
            inst✝⁷ : NormedAddCommGroup V
            inst✝⁶ : InnerProductSpace Real V
            inst✝⁵ : MeasurableSpace V
            inst✝⁴ : BorelSpace V
            inst✝³ : FiniteDimensional Real V
            inst✝² : NormedAddCommGroup E
            inst✝¹ : NormedSpace Complex E
            f : V → E
            inst✝ : CompleteSpace E
            ⊢ MeasureTheory.Measure V
          -/
          /-
            🎉 no goals
          -/
    (hf : Integrable f) (h'f : Integrable (𝓕 f)) {v : V}
                               /-
                                 🎉 no goals
                               -/
    (hv : ContinuousAt f v) : 𝓕 (𝓕⁻ f) v = f v := by
  /-
    V : Type u_1
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup V
    inst✝⁶ : InnerProductSpace Real V
    inst✝⁵ : MeasurableSpace V
    inst✝⁴ : BorelSpace V
    inst✝³ : FiniteDimensional Real V
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    f : V → E
    inst✝ : CompleteSpace E
    hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
    h'f : MeasureTheory.Integrable (Real.fourierIntegral f) MeasureTheory.MeasureS …
    v : V
    hv : ContinuousAt f v
    ⊢ Eq (Real.fourierIntegral (Real.fourierIntegralInv f) v) (f v)
  -/
  rw [fourierIntegralInv_comm]
  /-
    V : Type u_1
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup V
    inst✝⁶ : InnerProductSpace Real V
    inst✝⁵ : MeasurableSpace V
    inst✝⁴ : BorelSpace V
    inst✝³ : FiniteDimensional Real V
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    f : V → E
    inst✝ : CompleteSpace E
    hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
    h'f : MeasureTheory.Integrable (Real.fourierIntegral f) MeasureTheory.MeasureS …
    v : V
    hv : ContinuousAt f v
    ⊢ Eq (Real.fourierIntegralInv (Real.fourierIntegral f) v) (f v)
  -/
  exact fourier_inversion hf h'f hv
  /-
    🎉 no goals
  -/


/-- **Fourier inversion formula**: If a function `f` on a finite-dimensional real inner product
space is continuous, integrable, and its Fourier transform `𝓕 f` is also integrable,
then `𝓕 (𝓕⁻ f) = f`. -/
theorem Continuous.fourier_inversion_inv (h : Continuous f)
          /-
            V : Type u_1
            E : Type u_2
            inst✝⁷ : NormedAddCommGroup V
            inst✝⁶ : InnerProductSpace Real V
            inst✝⁵ : MeasurableSpace V
            inst✝⁴ : BorelSpace V
            inst✝³ : FiniteDimensional Real V
            inst✝² : NormedAddCommGroup E
            inst✝¹ : NormedSpace Complex E
            f : V → E
            inst✝ : CompleteSpace E
            h : Continuous f
            ⊢ MeasureTheory.Measure V
          -/
          /-
            🎉 no goals
          -/
    (hf : Integrable f) (h'f : Integrable (𝓕 f)) :
                               /-
                                 🎉 no goals
                               -/
    𝓕 (𝓕⁻ f) = f := by
  /-
    V : Type u_1
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup V
    inst✝⁶ : InnerProductSpace Real V
    inst✝⁵ : MeasurableSpace V
    inst✝⁴ : BorelSpace V
    inst✝³ : FiniteDimensional Real V
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    f : V → E
    inst✝ : CompleteSpace E
    h : Continuous f
    hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
    h'f : MeasureTheory.Integrable (Real.fourierIntegral f) MeasureTheory.MeasureS …
    ⊢ Eq (Real.fourierIntegral (Real.fourierIntegralInv f)) f
  -/
  ext v
  /-
    case h
    V : Type u_1
    E : Type u_2
    inst✝⁷ : NormedAddCommGroup V
    inst✝⁶ : InnerProductSpace Real V
    inst✝⁵ : MeasurableSpace V
    inst✝⁴ : BorelSpace V
    inst✝³ : FiniteDimensional Real V
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Complex E
    f : V → E
    inst✝ : CompleteSpace E
    h : Continuous f
    hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
    h'f : MeasureTheory.Integrable (Real.fourierIntegral f) MeasureTheory.MeasureS …
    v : V
    ⊢ Eq (Real.fourierIntegral (Real.fourierIntegralInv f) v) (f v)
  -/
  exact hf.fourier_inversion_inv h'f h.continuousAt
  /-
    🎉 no goals
  -/

