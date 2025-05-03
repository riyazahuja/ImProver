lemma hasDerivAt_fourierChar (x : ℝ) : HasDerivAt (𝐞 · : ℝ → ℂ) (2 * π * I * 𝐞 x) x := by
  have h1 (y : ℝ) : 𝐞 y = fourier 1 (y : UnitAddCircle) := by
    rw [fourierChar_apply, fourier_coe_apply]
    push_cast
    ring_nf
  /-
    x : Real
    h1 : ∀ (y : Real), Eq (↑(Real.fourierChar y)) ((fourier 1) ↑y)
    ⊢ HasDerivAt (fun x => ↑(Real.fourierChar x)) (HMul.hMul (HMul.hMul (HMul.hMul …
  -/
  simpa only [h1, Int.cast_one, ofReal_one, div_one, mul_one] using hasDerivAt_fourier 1 1 x
  /-
    🎉 no goals
  -/


lemma differentiable_fourierChar : Differentiable ℝ (𝐞 · : ℝ → ℂ) :=
  fun x ↦ (Real.hasDerivAt_fourierChar x).differentiableAt


lemma deriv_fourierChar (x : ℝ) : deriv (𝐞 · : ℝ → ℂ) x = 2 * π * I * 𝐞 x :=
  (Real.hasDerivAt_fourierChar x).deriv


lemma hasFDerivAt_fourierChar_neg_bilinear_right (v : V) (w : W) :
    HasFDerivAt (fun w ↦ (𝐞 (-L v w) : ℂ))
      ((-2 * π * I * 𝐞 (-L v w)) • (ofRealCLM ∘L (L v))) w := by
  /-
    V : Type u_1
    W : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : NormedAddCommGroup W
    inst✝ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    v : V
    w : W
    ⊢ HasFDerivAt (fun w => ↑(Real.fourierChar (Neg.neg ((L v) w)))) (HSMul.hSMul  …
  -/
  have ha : HasFDerivAt (fun w' : W ↦ L v w') (L v) w := ContinuousLinearMap.hasFDerivAt (L v)
  /-
    V : Type u_1
    W : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : NormedAddCommGroup W
    inst✝ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    v : V
    w : W
    ha : HasFDerivAt (fun w' => (L v) w') (L v) w
    ⊢ HasFDerivAt (fun w => ↑(Real.fourierChar (Neg.neg ((L v) w)))) (HSMul.hSMul  …
  -/
  convert (hasDerivAt_fourierChar (-L v w)).hasFDerivAt.comp w ha.neg using 1
  /-
    case h.e'_12
    V : Type u_1
    W : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : NormedAddCommGroup W
    inst✝ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    v : V
    w : W
    ha : HasFDerivAt (fun w' => (L v) w') (L v) w
    ⊢ Eq (HSMul.hSMul (HMul.hMul (HMul.hMul (HMul.hMul (-2) ↑Real.pi) Complex.I) ↑ …
  -/
  ext y
  simp only [neg_mul, ContinuousLinearMap.coe_smul', ContinuousLinearMap.coe_comp', Pi.smul_apply,
    Function.comp_apply, ofRealCLM_apply, smul_eq_mul, ContinuousLinearMap.comp_neg,
    ContinuousLinearMap.neg_apply, ContinuousLinearMap.smulRight_apply,
    ContinuousLinearMap.one_apply, real_smul, neg_inj]
  /-
    case h.e'_12.h
    V : Type u_1
    W : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : NormedAddCommGroup W
    inst✝ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    v : V
    w : W
    ha : HasFDerivAt (fun w' => (L v) w') (L v) w
    y : W
    ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul 2 ↑Real.pi) Complex.I) ↑(Real …
  -/
  ring
  /-
    🎉 no goals
  -/


lemma fderiv_fourierChar_neg_bilinear_right_apply (v : V) (w y : W) :
    fderiv ℝ (fun w ↦ (𝐞 (-L v w) : ℂ)) w y = -2 * π * I * L v y * 𝐞 (-L v w) := by
  simp only [(hasFDerivAt_fourierChar_neg_bilinear_right L v w).fderiv, neg_mul,
    ContinuousLinearMap.coe_smul', ContinuousLinearMap.coe_comp', Pi.smul_apply,
    Function.comp_apply, ofRealCLM_apply, smul_eq_mul, neg_inj]
  /-
    V : Type u_1
    W : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : NormedAddCommGroup W
    inst✝ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    v : V
    w y : W
    ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul 2 ↑Real.pi) Complex.I) ↑(Real …
  -/
  ring
  /-
    🎉 no goals
  -/


lemma differentiable_fourierChar_neg_bilinear_right (v : V) :
    Differentiable ℝ (fun w ↦ (𝐞 (-L v w) : ℂ)) :=
  fun w ↦ (hasFDerivAt_fourierChar_neg_bilinear_right L v w).differentiableAt


lemma hasFDerivAt_fourierChar_neg_bilinear_left (v : V) (w : W) :
    HasFDerivAt (fun v ↦ (𝐞 (-L v w) : ℂ))
      ((-2 * π * I * 𝐞 (-L v w)) • (ofRealCLM ∘L (L.flip w))) v :=
  hasFDerivAt_fourierChar_neg_bilinear_right L.flip w v


lemma fderiv_fourierChar_neg_bilinear_left_apply (v y : V) (w : W) :
    fderiv ℝ (fun v ↦ (𝐞 (-L v w) : ℂ)) v y = -2 * π * I * L y w * 𝐞 (-L v w) := by
  simp only [(hasFDerivAt_fourierChar_neg_bilinear_left L v w).fderiv, neg_mul,
    ContinuousLinearMap.coe_smul', ContinuousLinearMap.coe_comp', Pi.smul_apply,
    Function.comp_apply, ContinuousLinearMap.flip_apply, ofRealCLM_apply, smul_eq_mul, neg_inj]
  /-
    V : Type u_1
    W : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : NormedAddCommGroup W
    inst✝ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    v y : V
    w : W
    ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul 2 ↑Real.pi) Complex.I) ↑(Real …
  -/
  ring
  /-
    🎉 no goals
  -/


lemma differentiable_fourierChar_neg_bilinear_left (w : W) :
    Differentiable ℝ (fun v ↦ (𝐞 (-L v w) : ℂ)) :=
  fun v ↦ (hasFDerivAt_fourierChar_neg_bilinear_left L v w).differentiableAt


/-- Send a function `f : V → E` to the function `f : V → Hom (W, E)` given by
`v ↦ (w ↦ -2 * π * I * L (v, w) • f v)`. This is designed so that the Fourier transform of
`fourierSMulRight L f` is the derivative of the Fourier transform of `f`. -/
def fourierSMulRight (v : V) : (W →L[ℝ] E) := -(2 * π * I) • (L v).smulRight (f v)


@[simp] lemma fourierSMulRight_apply (v : V) (w : W) :
    fourierSMulRight L f v w = -(2 * π * I) • L v w • f v := rfl


/-- The `w`-derivative of the Fourier transform integrand. -/
lemma hasFDerivAt_fourierChar_smul (v : V) (w : W) :
    HasFDerivAt (fun w' ↦ 𝐞 (-L v w') • f v) (𝐞 (-L v w) • fourierSMulRight L f v) w := by
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Complex E
    V : Type u_2
    W : Type u_3
    inst✝³ : NormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : NormedAddCommGroup W
    inst✝ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    f : V → E
    v : V
    w : W
    ⊢ HasFDerivAt (fun w' => HSMul.hSMul (Real.fourierChar (Neg.neg ((L v) w'))) ( …
  -/
  have ha : HasFDerivAt (fun w' : W ↦ L v w') (L v) w := ContinuousLinearMap.hasFDerivAt (L v)
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Complex E
    V : Type u_2
    W : Type u_3
    inst✝³ : NormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : NormedAddCommGroup W
    inst✝ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    f : V → E
    v : V
    w : W
    ha : HasFDerivAt (fun w' => (L v) w') (L v) w
    ⊢ HasFDerivAt (fun w' => HSMul.hSMul (Real.fourierChar (Neg.neg ((L v) w'))) ( …
  -/
  convert ((hasDerivAt_fourierChar (-L v w)).hasFDerivAt.comp w ha.neg).smul_const (f v)
  /-
    case h.e'_12.h.h
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Complex E
    V : Type u_2
    W : Type u_3
    inst✝³ : NormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : NormedAddCommGroup W
    inst✝ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    f : V → E
    v : V
    w : W
    ha : HasFDerivAt (fun w' => (L v) w') (L v) w
    e_8✝ : Eq SeminormedAddCommGroup.toAddCommGroup NormedAddCommGroup.toAddCommGr …
    he✝ : Eq NormedSpace.toModule NormedSpace.toModule
    ⊢ Eq (HSMul.hSMul (Real.fourierChar (Neg.neg ((L v) w))) (VectorFourier.fourie …
  -/
  ext w' : 1
  /-
    case h.e'_12.h.h.h
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Complex E
    V : Type u_2
    W : Type u_3
    inst✝³ : NormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : NormedAddCommGroup W
    inst✝ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    f : V → E
    v : V
    w : W
    ha : HasFDerivAt (fun w' => (L v) w') (L v) w
    e_8✝ : Eq SeminormedAddCommGroup.toAddCommGroup NormedAddCommGroup.toAddCommGr …
    he✝ : Eq NormedSpace.toModule NormedSpace.toModule
    w' : W
    ⊢ Eq ((HSMul.hSMul (Real.fourierChar (Neg.neg ((L v) w))) (VectorFourier.fouri …
  -/
  simp_rw [fourierSMulRight, ContinuousLinearMap.smul_apply, ContinuousLinearMap.smulRight_apply]
  rw [ContinuousLinearMap.comp_apply, ContinuousLinearMap.neg_apply,
    ContinuousLinearMap.smulRight_apply, ContinuousLinearMap.one_apply, ← smul_assoc, smul_comm,
    ← smul_assoc, real_smul, real_smul, Submonoid.smul_def, smul_eq_mul]
  /-
    case h.e'_12.h.h.h
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Complex E
    V : Type u_2
    W : Type u_3
    inst✝³ : NormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : NormedAddCommGroup W
    inst✝ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    f : V → E
    v : V
    w : W
    ha : HasFDerivAt (fun w' => (L v) w') (L v) w
    e_8✝ : Eq SeminormedAddCommGroup.toAddCommGroup NormedAddCommGroup.toAddCommGr …
    he✝ : Eq NormedSpace.toModule NormedSpace.toModule
    w' : W
    ⊢ Eq (HSMul.hSMul (HMul.hMul (↑((L v) w')) (HMul.hMul (↑(Real.fourierChar (Neg …
  -/
  push_cast
  /-
    case h.e'_12.h.h.h
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Complex E
    V : Type u_2
    W : Type u_3
    inst✝³ : NormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : NormedAddCommGroup W
    inst✝ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    f : V → E
    v : V
    w : W
    ha : HasFDerivAt (fun w' => (L v) w') (L v) w
    e_8✝ : Eq SeminormedAddCommGroup.toAddCommGroup NormedAddCommGroup.toAddCommGr …
    he✝ : Eq NormedSpace.toModule NormedSpace.toModule
    w' : W
    ⊢ Eq (HSMul.hSMul (HMul.hMul (↑((L v) w')) (HMul.hMul (↑(Real.fourierChar (Neg …
  -/
  ring_nf
  /-
    🎉 no goals
  -/


lemma norm_fourierSMulRight (L : V →L[ℝ] W →L[ℝ] ℝ) (f : V → E) (v : V) :
    ‖fourierSMulRight L f v‖ = (2 * π) * ‖L v‖ * ‖f v‖ := by
  rw [fourierSMulRight, norm_smul _ (ContinuousLinearMap.smulRight (L v) (f v)),
    norm_neg, norm_mul, norm_mul, norm_eq_abs I, abs_I,
    mul_one, norm_eq_abs ((_ : ℝ) : ℂ), Complex.abs_of_nonneg pi_pos.le, norm_eq_abs (2 : ℂ),
    Complex.abs_two, ContinuousLinearMap.norm_smulRight_apply, ← mul_assoc]


lemma norm_fourierSMulRight_le (L : V →L[ℝ] W →L[ℝ] ℝ) (f : V → E) (v : V) :
    ‖fourierSMulRight L f v‖ ≤ 2 * π * ‖L‖ * ‖v‖ * ‖f v‖ := calc
  ‖fourierSMulRight L f v‖ = (2 * π) * ‖L v‖ * ‖f v‖ := norm_fourierSMulRight _ _ _
                                          /-
                                            E : Type u_1
                                            inst✝⁵ : NormedAddCommGroup E
                                            inst✝⁴ : NormedSpace Complex E
                                            V : Type u_2
                                            W : Type u_3
                                            inst✝³ : NormedAddCommGroup V
                                            inst✝² : NormedSpace Real V
                                            inst✝¹ : NormedAddCommGroup W
                                            inst✝ : NormedSpace Real W
                                            L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
                                            f : V → E
                                            v : V
                                            ⊢ LE.le (HMul.hMul (HMul.hMul (HMul.hMul 2 Real.pi) (Norm.norm (L v))) (Norm.n …
                                          -/
  _ ≤ (2 * π) * (‖L‖ * ‖v‖) * ‖f v‖ := by gcongr; exact L.le_opNorm _
                                                  /-
                                                    🎉 no goals
                                                  -/
                                      /-
                                        E : Type u_1
                                        inst✝⁵ : NormedAddCommGroup E
                                        inst✝⁴ : NormedSpace Complex E
                                        V : Type u_2
                                        W : Type u_3
                                        inst✝³ : NormedAddCommGroup V
                                        inst✝² : NormedSpace Real V
                                        inst✝¹ : NormedAddCommGroup W
                                        inst✝ : NormedSpace Real W
                                        L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
                                        f : V → E
                                        v : V
                                        ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul 2 Real.pi) (HMul.hMul (Norm.norm L) (Nor …
                                      -/
  _ = 2 * π * ‖L‖ * ‖v‖ * ‖f v‖ := by ring
                                      /-
                                        🎉 no goals
                                      -/


lemma _root_.MeasureTheory.AEStronglyMeasurable.fourierSMulRight
    [SecondCountableTopologyEither V (W →L[ℝ] ℝ)] [MeasurableSpace V] [BorelSpace V]
    {L : V →L[ℝ] W →L[ℝ] ℝ} {f : V → E} {μ : Measure V}
    (hf : AEStronglyMeasurable f μ) :
    AEStronglyMeasurable (fun v ↦ fourierSMulRight L f v) μ := by
  /-
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Complex E
    V : Type u_2
    W : Type u_3
    inst✝⁶ : NormedAddCommGroup V
    inst✝⁵ : NormedSpace Real V
    inst✝⁴ : NormedAddCommGroup W
    inst✝³ : NormedSpace Real W
    inst✝² : SecondCountableTopologyEither V (ContinuousLinearMap (RingHom.id Real …
    inst✝¹ : MeasurableSpace V
    inst✝ : BorelSpace V
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    f : V → E
    μ : MeasureTheory.Measure V
    hf : MeasureTheory.AEStronglyMeasurable f μ
    ⊢ MeasureTheory.AEStronglyMeasurable (fun v => VectorFourier.fourierSMulRight  …
  -/
  apply AEStronglyMeasurable.const_smul'
  have aux0 : Continuous fun p : (W →L[ℝ] ℝ) × E ↦ p.1.smulRight p.2 :=
    (ContinuousLinearMap.smulRightL ℝ W E).continuous₂
  have aux1 : AEStronglyMeasurable (fun v ↦ (L v, f v)) μ :=
    L.continuous.aestronglyMeasurable.prod_mk hf
  -- Elaboration without the expected type is faster here:
  /-
    case hf
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Complex E
    V : Type u_2
    W : Type u_3
    inst✝⁶ : NormedAddCommGroup V
    inst✝⁵ : NormedSpace Real V
    inst✝⁴ : NormedAddCommGroup W
    inst✝³ : NormedSpace Real W
    inst✝² : SecondCountableTopologyEither V (ContinuousLinearMap (RingHom.id Real …
    inst✝¹ : MeasurableSpace V
    inst✝ : BorelSpace V
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    f : V → E
    μ : MeasureTheory.Measure V
    hf : MeasureTheory.AEStronglyMeasurable f μ
    aux0 : Continuous fun p => p.1.smulRight p.2
    aux1 : MeasureTheory.AEStronglyMeasurable (fun v => { fst := L v, snd := f v } …
    ⊢ MeasureTheory.AEStronglyMeasurable (fun x => (L x).smulRight (f x)) μ
  -/
  exact (aux0.comp_aestronglyMeasurable aux1 : _)
  /-
    🎉 no goals
  -/


/-- Main theorem of this section: if both `f` and `x ↦ ‖x‖ * ‖f x‖` are integrable, then the
Fourier transform of `f` has a Fréchet derivative (everywhere in its domain) and its derivative is
the Fourier transform of `smulRight L f`. -/
theorem hasFDerivAt_fourierIntegral
    [MeasurableSpace V] [BorelSpace V] [SecondCountableTopology V] {μ : Measure V}
    (hf : Integrable f μ) (hf' : Integrable (fun v : V ↦ ‖v‖ * ‖f v‖) μ) (w : W) :
    HasFDerivAt (fourierIntegral 𝐞 μ L.toLinearMap₂ f)
      (fourierIntegral 𝐞 μ L.toLinearMap₂ (fourierSMulRight L f) w) w := by
  /-
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Complex E
    V : Type u_2
    W : Type u_3
    inst✝⁶ : NormedAddCommGroup V
    inst✝⁵ : NormedSpace Real V
    inst✝⁴ : NormedAddCommGroup W
    inst✝³ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    f : V → E
    inst✝² : MeasurableSpace V
    inst✝¹ : BorelSpace V
    inst✝ : SecondCountableTopology V
    μ : MeasureTheory.Measure V
    hf : MeasureTheory.Integrable f μ
    hf' : MeasureTheory.Integrable (fun v => HMul.hMul (Norm.norm v) (Norm.norm (f …
    w : W
    ⊢ HasFDerivAt (VectorFourier.fourierIntegral Real.fourierChar μ L.toLinearMap₂ …
  -/
  let F : W → V → E := fun w' v ↦ 𝐞 (-L v w') • f v
  /-
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Complex E
    V : Type u_2
    W : Type u_3
    inst✝⁶ : NormedAddCommGroup V
    inst✝⁵ : NormedSpace Real V
    inst✝⁴ : NormedAddCommGroup W
    inst✝³ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    f : V → E
    inst✝² : MeasurableSpace V
    inst✝¹ : BorelSpace V
    inst✝ : SecondCountableTopology V
    μ : MeasureTheory.Measure V
    hf : MeasureTheory.Integrable f μ
    hf' : MeasureTheory.Integrable (fun v => HMul.hMul (Norm.norm v) (Norm.norm (f …
    w : W
    F : W → V → E := fun w' v => HSMul.hSMul (Real.fourierChar (Neg.neg ((L v) w') …
    ⊢ HasFDerivAt (VectorFourier.fourierIntegral Real.fourierChar μ L.toLinearMap₂ …
  -/
  let F' : W → V → W →L[ℝ] E := fun w' v ↦ 𝐞 (-L v w') • fourierSMulRight L f v
  /-
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Complex E
    V : Type u_2
    W : Type u_3
    inst✝⁶ : NormedAddCommGroup V
    inst✝⁵ : NormedSpace Real V
    inst✝⁴ : NormedAddCommGroup W
    inst✝³ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    f : V → E
    inst✝² : MeasurableSpace V
    inst✝¹ : BorelSpace V
    inst✝ : SecondCountableTopology V
    μ : MeasureTheory.Measure V
    hf : MeasureTheory.Integrable f μ
    hf' : MeasureTheory.Integrable (fun v => HMul.hMul (Norm.norm v) (Norm.norm (f …
    w : W
    F : W → V → E := fun w' v => HSMul.hSMul (Real.fourierChar (Neg.neg ((L v) w') …
    F' : W → V → ContinuousLinearMap (RingHom.id Real) W E := fun w' v => HSMul.hS …
    ⊢ HasFDerivAt (VectorFourier.fourierIntegral Real.fourierChar μ L.toLinearMap₂ …
  -/
  let B : V → ℝ := fun v ↦ 2 * π * ‖L‖ * ‖v‖ * ‖f v‖
  have h0 (w' : W) : Integrable (F w') μ :=
    (fourierIntegral_convergent_iff continuous_fourierChar
      (by apply L.continuous₂ : Continuous (fun p : V × W ↦ L.toLinearMap₂ p.1 p.2)) w').2 hf
  have h1 : ∀ᶠ w' in 𝓝 w, AEStronglyMeasurable (F w') μ :=
    Eventually.of_forall (fun w' ↦ (h0 w').aestronglyMeasurable)
  have h3 : AEStronglyMeasurable (F' w) μ := by
    refine .smul ?_ hf.1.fourierSMulRight
    refine (continuous_fourierChar.comp ?_).aestronglyMeasurable
    exact (L.continuous₂.comp (Continuous.Prod.mk_left w)).neg
  have h4 : (∀ᵐ v ∂μ, ∀ (w' : W), w' ∈ Metric.ball w 1 → ‖F' w' v‖ ≤ B v) := by
    filter_upwards with v w' _
    rw [Circle.norm_smul _ (fourierSMulRight L f v)]
    exact norm_fourierSMulRight_le L f v
  /-
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Complex E
    V : Type u_2
    W : Type u_3
    inst✝⁶ : NormedAddCommGroup V
    inst✝⁵ : NormedSpace Real V
    inst✝⁴ : NormedAddCommGroup W
    inst✝³ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    f : V → E
    inst✝² : MeasurableSpace V
    inst✝¹ : BorelSpace V
    inst✝ : SecondCountableTopology V
    μ : MeasureTheory.Measure V
    hf : MeasureTheory.Integrable f μ
    hf' : MeasureTheory.Integrable (fun v => HMul.hMul (Norm.norm v) (Norm.norm (f …
    w : W
    F : W → V → E := fun w' v => HSMul.hSMul (Real.fourierChar (Neg.neg ((L v) w') …
    F' : W → V → ContinuousLinearMap (RingHom.id Real) W E := fun w' v => HSMul.hS …
    B : V → Real := fun v => HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul 2 Real.pi) …
    h0 : ∀ (w' : W), MeasureTheory.Integrable (F w') μ
    h1 : Filter.Eventually (fun w' => MeasureTheory.AEStronglyMeasurable (F w') μ) …
    h3 : MeasureTheory.AEStronglyMeasurable (F' w) μ
    h4 : Filter.Eventually (fun v => ∀ (w' : W), Membership.mem (Metric.ball w 1)  …
    ⊢ HasFDerivAt (VectorFourier.fourierIntegral Real.fourierChar μ L.toLinearMap₂ …
  -/
  have h5 : Integrable B μ := by simpa only [← mul_assoc] using hf'.const_mul (2 * π * ‖L‖)
  have h6 : ∀ᵐ v ∂μ, ∀ w', w' ∈ Metric.ball w 1 → HasFDerivAt (fun x ↦ F x v) (F' w' v) w' :=
    ae_of_all _ (fun v w' _ ↦ hasFDerivAt_fourierChar_smul L f v w')
  /-
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Complex E
    V : Type u_2
    W : Type u_3
    inst✝⁶ : NormedAddCommGroup V
    inst✝⁵ : NormedSpace Real V
    inst✝⁴ : NormedAddCommGroup W
    inst✝³ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    f : V → E
    inst✝² : MeasurableSpace V
    inst✝¹ : BorelSpace V
    inst✝ : SecondCountableTopology V
    μ : MeasureTheory.Measure V
    hf : MeasureTheory.Integrable f μ
    hf' : MeasureTheory.Integrable (fun v => HMul.hMul (Norm.norm v) (Norm.norm (f …
    w : W
    F : W → V → E := fun w' v => HSMul.hSMul (Real.fourierChar (Neg.neg ((L v) w') …
    F' : W → V → ContinuousLinearMap (RingHom.id Real) W E := fun w' v => HSMul.hS …
    B : V → Real := fun v => HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul 2 Real.pi) …
    h0 : ∀ (w' : W), MeasureTheory.Integrable (F w') μ
    h1 : Filter.Eventually (fun w' => MeasureTheory.AEStronglyMeasurable (F w') μ) …
    h3 : MeasureTheory.AEStronglyMeasurable (F' w) μ
    h4 : Filter.Eventually (fun v => ∀ (w' : W), Membership.mem (Metric.ball w 1)  …
    h5 : MeasureTheory.Integrable B μ
    h6 : Filter.Eventually (fun v => ∀ (w' : W), Membership.mem (Metric.ball w 1)  …
    ⊢ HasFDerivAt (VectorFourier.fourierIntegral Real.fourierChar μ L.toLinearMap₂ …
  -/
  exact hasFDerivAt_integral_of_dominated_of_fderiv_le one_pos h1 (h0 w) h3 h4 h5 h6
  /-
    🎉 no goals
  -/


lemma fderiv_fourierIntegral
    [MeasurableSpace V] [BorelSpace V] [SecondCountableTopology V] {μ : Measure V}
    (hf : Integrable f μ) (hf' : Integrable (fun v : V ↦ ‖v‖ * ‖f v‖) μ) :
    fderiv ℝ (fourierIntegral 𝐞 μ L.toLinearMap₂ f) =
      fourierIntegral 𝐞 μ L.toLinearMap₂ (fourierSMulRight L f) := by
  /-
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Complex E
    V : Type u_2
    W : Type u_3
    inst✝⁶ : NormedAddCommGroup V
    inst✝⁵ : NormedSpace Real V
    inst✝⁴ : NormedAddCommGroup W
    inst✝³ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    f : V → E
    inst✝² : MeasurableSpace V
    inst✝¹ : BorelSpace V
    inst✝ : SecondCountableTopology V
    μ : MeasureTheory.Measure V
    hf : MeasureTheory.Integrable f μ
    hf' : MeasureTheory.Integrable (fun v => HMul.hMul (Norm.norm v) (Norm.norm (f …
    ⊢ Eq (fderiv Real (VectorFourier.fourierIntegral Real.fourierChar μ L.toLinear …
  -/
  ext w : 1
  /-
    case h
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Complex E
    V : Type u_2
    W : Type u_3
    inst✝⁶ : NormedAddCommGroup V
    inst✝⁵ : NormedSpace Real V
    inst✝⁴ : NormedAddCommGroup W
    inst✝³ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    f : V → E
    inst✝² : MeasurableSpace V
    inst✝¹ : BorelSpace V
    inst✝ : SecondCountableTopology V
    μ : MeasureTheory.Measure V
    hf : MeasureTheory.Integrable f μ
    hf' : MeasureTheory.Integrable (fun v => HMul.hMul (Norm.norm v) (Norm.norm (f …
    w : W
    ⊢ Eq (fderiv Real (VectorFourier.fourierIntegral Real.fourierChar μ L.toLinear …
  -/
  exact (hasFDerivAt_fourierIntegral L hf hf' w).fderiv
  /-
    🎉 no goals
  -/


lemma differentiable_fourierIntegral
    [MeasurableSpace V] [BorelSpace V] [SecondCountableTopology V] {μ : Measure V}
    (hf : Integrable f μ) (hf' : Integrable (fun v : V ↦ ‖v‖ * ‖f v‖) μ) :
    Differentiable ℝ (fourierIntegral 𝐞 μ L.toLinearMap₂ f) :=
  fun w ↦ (hasFDerivAt_fourierIntegral L hf hf' w).differentiableAt


/-- The Fourier integral of the derivative of a function is obtained by multiplying the Fourier
integral of the original function by `-L w v`. -/
theorem fourierIntegral_fderiv [MeasurableSpace V] [BorelSpace V] [FiniteDimensional ℝ V]
    {μ : Measure V} [Measure.IsAddHaarMeasure μ]
    (hf : Integrable f μ) (h'f : Differentiable ℝ f) (hf' : Integrable (fderiv ℝ f) μ) :
    fourierIntegral 𝐞 μ L.toLinearMap₂ (fderiv ℝ f)
      = fourierSMulRight (-L.flip) (fourierIntegral 𝐞 μ L.toLinearMap₂ f) := by
  /-
    E : Type u_1
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace Complex E
    V : Type u_2
    W : Type u_3
    inst✝⁷ : NormedAddCommGroup V
    inst✝⁶ : NormedSpace Real V
    inst✝⁵ : NormedAddCommGroup W
    inst✝⁴ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    f : V → E
    inst✝³ : MeasurableSpace V
    inst✝² : BorelSpace V
    inst✝¹ : FiniteDimensional Real V
    μ : MeasureTheory.Measure V
    inst✝ : μ.IsAddHaarMeasure
    hf : MeasureTheory.Integrable f μ
    h'f : Differentiable Real f
    hf' : MeasureTheory.Integrable (fderiv Real f) μ
    ⊢ Eq (VectorFourier.fourierIntegral Real.fourierChar μ L.toLinearMap₂ (fderiv  …
  -/
  ext w y
  /-
    case h.h
    E : Type u_1
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace Complex E
    V : Type u_2
    W : Type u_3
    inst✝⁷ : NormedAddCommGroup V
    inst✝⁶ : NormedSpace Real V
    inst✝⁵ : NormedAddCommGroup W
    inst✝⁴ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    f : V → E
    inst✝³ : MeasurableSpace V
    inst✝² : BorelSpace V
    inst✝¹ : FiniteDimensional Real V
    μ : MeasureTheory.Measure V
    inst✝ : μ.IsAddHaarMeasure
    hf : MeasureTheory.Integrable f μ
    h'f : Differentiable Real f
    hf' : MeasureTheory.Integrable (fderiv Real f) μ
    w : W
    y : V
    ⊢ Eq ((VectorFourier.fourierIntegral Real.fourierChar μ L.toLinearMap₂ (fderiv …
  -/
  let g (v : V) : ℂ := 𝐞 (-L v w)
  /- First rewrite things in a simplified form, without any real change. -/
  suffices ∫ x, g x • fderiv ℝ f x y ∂μ = ∫ x, (2 * ↑π * I * L y w * g x) • f x ∂μ by
    rw [fourierIntegral_continuousLinearMap_apply' hf']
    simpa only [fourierIntegral, ContinuousLinearMap.toLinearMap₂_apply, fourierSMulRight_apply,
      ContinuousLinearMap.neg_apply, ContinuousLinearMap.flip_apply, ← integral_smul, neg_smul,
      smul_neg, ← smul_smul, coe_smul, neg_neg]
  -- Key step: integrate by parts with respect to `y` to switch the derivative from `f` to `g`.
  have A x : fderiv ℝ g x y = - 2 * ↑π * I * L y w * g x :=
    fderiv_fourierChar_neg_bilinear_left_apply _ _ _ _
  /-
    case h.h
    E : Type u_1
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace Complex E
    V : Type u_2
    W : Type u_3
    inst✝⁷ : NormedAddCommGroup V
    inst✝⁶ : NormedSpace Real V
    inst✝⁵ : NormedAddCommGroup W
    inst✝⁴ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    f : V → E
    inst✝³ : MeasurableSpace V
    inst✝² : BorelSpace V
    inst✝¹ : FiniteDimensional Real V
    μ : MeasureTheory.Measure V
    inst✝ : μ.IsAddHaarMeasure
    hf : MeasureTheory.Integrable f μ
    h'f : Differentiable Real f
    hf' : MeasureTheory.Integrable (fderiv Real f) μ
    w : W
    y : V
    g : V → Complex := fun v => ↑(Real.fourierChar (Neg.neg ((L v) w)))
    A : ∀ (x : V), Eq ((fderiv Real g x) y) (HMul.hMul (HMul.hMul (HMul.hMul (HMul …
    ⊢ Eq (MeasureTheory.integral μ fun x => HSMul.hSMul (g x) ((fderiv Real f x) y …
  -/
  rw [integral_smul_fderiv_eq_neg_fderiv_smul_of_integrable, ← integral_neg]
    /-
      case h.h
      E : Type u_1
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace Complex E
      V : Type u_2
      W : Type u_3
      inst✝⁷ : NormedAddCommGroup V
      inst✝⁶ : NormedSpace Real V
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace Real W
      L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
      f : V → E
      inst✝³ : MeasurableSpace V
      inst✝² : BorelSpace V
      inst✝¹ : FiniteDimensional Real V
      μ : MeasureTheory.Measure V
      inst✝ : μ.IsAddHaarMeasure
      hf : MeasureTheory.Integrable f μ
      h'f : Differentiable Real f
      hf' : MeasureTheory.Integrable (fderiv Real f) μ
      w : W
      y : V
      g : V → Complex := fun v => ↑(Real.fourierChar (Neg.neg ((L v) w)))
      A : ∀ (x : V), Eq ((fderiv Real g x) y) (HMul.hMul (HMul.hMul (HMul.hMul (HMul …
      ⊢ Eq (MeasureTheory.integral μ fun a => Neg.neg (HSMul.hSMul ((fderiv Real g a …
    -/
  · congr with x
    /-
      case h.h.e_f.h
      E : Type u_1
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace Complex E
      V : Type u_2
      W : Type u_3
      inst✝⁷ : NormedAddCommGroup V
      inst✝⁶ : NormedSpace Real V
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace Real W
      L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
      f : V → E
      inst✝³ : MeasurableSpace V
      inst✝² : BorelSpace V
      inst✝¹ : FiniteDimensional Real V
      μ : MeasureTheory.Measure V
      inst✝ : μ.IsAddHaarMeasure
      hf : MeasureTheory.Integrable f μ
      h'f : Differentiable Real f
      hf' : MeasureTheory.Integrable (fderiv Real f) μ
      w : W
      y : V
      g : V → Complex := fun v => ↑(Real.fourierChar (Neg.neg ((L v) w)))
      A : ∀ (x : V), Eq ((fderiv Real g x) y) (HMul.hMul (HMul.hMul (HMul.hMul (HMul …
      x : V
      ⊢ Eq (Neg.neg (HSMul.hSMul ((fderiv Real g x) y) (f x))) (HSMul.hSMul (HMul.hM …
    -/
    simp only [A, neg_mul, neg_smul, neg_neg]
    /-
      🎉 no goals
    -/
  · have : Integrable (fun x ↦ (-(2 * ↑π * I * ↑((L y) w)) • ((g x : ℂ) • f x))) μ :=
      ((fourierIntegral_convergent_iff' _ _).2 hf).smul _
    /-
      case h.h.hf'g
      E : Type u_1
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace Complex E
      V : Type u_2
      W : Type u_3
      inst✝⁷ : NormedAddCommGroup V
      inst✝⁶ : NormedSpace Real V
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace Real W
      L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
      f : V → E
      inst✝³ : MeasurableSpace V
      inst✝² : BorelSpace V
      inst✝¹ : FiniteDimensional Real V
      μ : MeasureTheory.Measure V
      inst✝ : μ.IsAddHaarMeasure
      hf : MeasureTheory.Integrable f μ
      h'f : Differentiable Real f
      hf' : MeasureTheory.Integrable (fderiv Real f) μ
      w : W
      y : V
      g : V → Complex := fun v => ↑(Real.fourierChar (Neg.neg ((L v) w)))
      A : ∀ (x : V), Eq ((fderiv Real g x) y) (HMul.hMul (HMul.hMul (HMul.hMul (HMul …
      this : MeasureTheory.Integrable (fun x => HSMul.hSMul (Neg.neg (HMul.hMul (HMu …
      ⊢ MeasureTheory.Integrable (fun x => HSMul.hSMul ((fderiv Real g x) y) (f x)) μ
    -/
    convert this using 2 with x
    /-
      case h.e'_6.h
      E : Type u_1
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace Complex E
      V : Type u_2
      W : Type u_3
      inst✝⁷ : NormedAddCommGroup V
      inst✝⁶ : NormedSpace Real V
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace Real W
      L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
      f : V → E
      inst✝³ : MeasurableSpace V
      inst✝² : BorelSpace V
      inst✝¹ : FiniteDimensional Real V
      μ : MeasureTheory.Measure V
      inst✝ : μ.IsAddHaarMeasure
      hf : MeasureTheory.Integrable f μ
      h'f : Differentiable Real f
      hf' : MeasureTheory.Integrable (fderiv Real f) μ
      w : W
      y : V
      g : V → Complex := fun v => ↑(Real.fourierChar (Neg.neg ((L v) w)))
      A : ∀ (x : V), Eq ((fderiv Real g x) y) (HMul.hMul (HMul.hMul (HMul.hMul (HMul …
      this : MeasureTheory.Integrable (fun x => HSMul.hSMul (Neg.neg (HMul.hMul (HMu …
      x : V
      ⊢ Eq (HSMul.hSMul ((fderiv Real g x) y) (f x)) (HSMul.hSMul (Neg.neg (HMul.hMu …
    -/
    simp only [A, neg_mul, neg_smul, smul_smul]
    /-
      🎉 no goals
    -/
    /-
      case h.h.hfg'
      E : Type u_1
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace Complex E
      V : Type u_2
      W : Type u_3
      inst✝⁷ : NormedAddCommGroup V
      inst✝⁶ : NormedSpace Real V
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace Real W
      L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
      f : V → E
      inst✝³ : MeasurableSpace V
      inst✝² : BorelSpace V
      inst✝¹ : FiniteDimensional Real V
      μ : MeasureTheory.Measure V
      inst✝ : μ.IsAddHaarMeasure
      hf : MeasureTheory.Integrable f μ
      h'f : Differentiable Real f
      hf' : MeasureTheory.Integrable (fderiv Real f) μ
      w : W
      y : V
      g : V → Complex := fun v => ↑(Real.fourierChar (Neg.neg ((L v) w)))
      A : ∀ (x : V), Eq ((fderiv Real g x) y) (HMul.hMul (HMul.hMul (HMul.hMul (HMul …
      ⊢ MeasureTheory.Integrable (fun x => HSMul.hSMul (g x) ((fderiv Real f x) y)) μ
    -/
  · exact (fourierIntegral_convergent_iff' _ _).2 (hf'.apply_continuousLinearMap _)
    /-
      🎉 no goals
    -/
    /-
      case h.h.hfg
      E : Type u_1
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace Complex E
      V : Type u_2
      W : Type u_3
      inst✝⁷ : NormedAddCommGroup V
      inst✝⁶ : NormedSpace Real V
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace Real W
      L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
      f : V → E
      inst✝³ : MeasurableSpace V
      inst✝² : BorelSpace V
      inst✝¹ : FiniteDimensional Real V
      μ : MeasureTheory.Measure V
      inst✝ : μ.IsAddHaarMeasure
      hf : MeasureTheory.Integrable f μ
      h'f : Differentiable Real f
      hf' : MeasureTheory.Integrable (fderiv Real f) μ
      w : W
      y : V
      g : V → Complex := fun v => ↑(Real.fourierChar (Neg.neg ((L v) w)))
      A : ∀ (x : V), Eq ((fderiv Real g x) y) (HMul.hMul (HMul.hMul (HMul.hMul (HMul …
      ⊢ MeasureTheory.Integrable (fun x => HSMul.hSMul (g x) (f x)) μ
    -/
  · exact (fourierIntegral_convergent_iff' _ _).2 hf
    /-
      🎉 no goals
    -/
    /-
      case h.h.hf
      E : Type u_1
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace Complex E
      V : Type u_2
      W : Type u_3
      inst✝⁷ : NormedAddCommGroup V
      inst✝⁶ : NormedSpace Real V
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace Real W
      L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
      f : V → E
      inst✝³ : MeasurableSpace V
      inst✝² : BorelSpace V
      inst✝¹ : FiniteDimensional Real V
      μ : MeasureTheory.Measure V
      inst✝ : μ.IsAddHaarMeasure
      hf : MeasureTheory.Integrable f μ
      h'f : Differentiable Real f
      hf' : MeasureTheory.Integrable (fderiv Real f) μ
      w : W
      y : V
      g : V → Complex := fun v => ↑(Real.fourierChar (Neg.neg ((L v) w)))
      A : ∀ (x : V), Eq ((fderiv Real g x) y) (HMul.hMul (HMul.hMul (HMul.hMul (HMul …
      ⊢ Differentiable Real g
    -/
  · exact differentiable_fourierChar_neg_bilinear_left _ _
    /-
      🎉 no goals
    -/
    /-
      case h.h.hg
      E : Type u_1
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace Complex E
      V : Type u_2
      W : Type u_3
      inst✝⁷ : NormedAddCommGroup V
      inst✝⁶ : NormedSpace Real V
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace Real W
      L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
      f : V → E
      inst✝³ : MeasurableSpace V
      inst✝² : BorelSpace V
      inst✝¹ : FiniteDimensional Real V
      μ : MeasureTheory.Measure V
      inst✝ : μ.IsAddHaarMeasure
      hf : MeasureTheory.Integrable f μ
      h'f : Differentiable Real f
      hf' : MeasureTheory.Integrable (fderiv Real f) μ
      w : W
      y : V
      g : V → Complex := fun v => ↑(Real.fourierChar (Neg.neg ((L v) w)))
      A : ∀ (x : V), Eq ((fderiv Real g x) y) (HMul.hMul (HMul.hMul (HMul.hMul (HMul …
      ⊢ Differentiable Real f
    -/
  · exact h'f
    /-
      🎉 no goals
    -/


/-- The formal multilinear series whose `n`-th term is
`(w₁, ..., wₙ) ↦ (-2πI)^n * L v w₁ * ... * L v wₙ • f v`, as a continuous multilinear map in
the space `W [×n]→L[ℝ] E`.

This is designed so that the Fourier transform of `v ↦ fourierPowSMulRight L f v n` is the
`n`-th derivative of the Fourier transform of `f`.
-/
def fourierPowSMulRight (f : V → E) (v : V) : FormalMultilinearSeries ℝ W E := fun n ↦
  (- (2 * π * I))^n • ((ContinuousMultilinearMap.mkPiRing ℝ (Fin n) (f v)).compContinuousLinearMap
  (fun _ ↦ L v))

/- Increase the priority to make sure that this lemma is used instead of
`FormalMultilinearSeries.apply_eq_prod_smul_coeff` even in dimension 1. -/

@[simp 1100] lemma fourierPowSMulRight_apply {f : V → E} {v : V} {n : ℕ} {m : Fin n → W} :
    fourierPowSMulRight L f v n m = (- (2 * π * I))^n • (∏ i, L v (m i)) • f v := by
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Complex E
    V : Type u_2
    W : Type u_3
    inst✝³ : NormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : NormedAddCommGroup W
    inst✝ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    f : V → E
    v : V
    n : Nat
    m : Fin n → W
    ⊢ Eq ((VectorFourier.fourierPowSMulRight L f v n) m) (HSMul.hSMul (HPow.hPow ( …
  -/
  simp [fourierPowSMulRight]
  /-
    🎉 no goals
  -/


/-- Decomposing `fourierPowSMulRight L f v n` as a composition of continuous bilinear and
multilinear maps, to deduce easily its continuity and differentiability properties. -/
lemma fourierPowSMulRight_eq_comp {f : V → E} {v : V} {n : ℕ} :
    fourierPowSMulRight L f v n = (- (2 * π * I))^n • smulRightL ℝ (fun (_ : Fin n) ↦ W) E
      (compContinuousLinearMapLRight
        (ContinuousMultilinearMap.mkPiAlgebra ℝ (Fin n) ℝ) (fun _ ↦ L v)) (f v) := rfl


@[continuity, fun_prop]
lemma _root_.Continuous.fourierPowSMulRight {f : V → E} (hf : Continuous f) (n : ℕ) :
    Continuous (fun v ↦ fourierPowSMulRight L f v n) := by
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Complex E
    V : Type u_2
    W : Type u_3
    inst✝³ : NormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : NormedAddCommGroup W
    inst✝ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    f : V → E
    hf : Continuous f
    n : Nat
    ⊢ Continuous fun v => VectorFourier.fourierPowSMulRight L f v n
  -/
  simp_rw [fourierPowSMulRight_eq_comp]
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Complex E
    V : Type u_2
    W : Type u_3
    inst✝³ : NormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : NormedAddCommGroup W
    inst✝ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    f : V → E
    hf : Continuous f
    n : Nat
    ⊢ Continuous fun v => HSMul.hSMul (HPow.hPow (Neg.neg (HMul.hMul (HMul.hMul 2  …
  -/
  apply Continuous.const_smul
  /-
    case hg
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Complex E
    V : Type u_2
    W : Type u_3
    inst✝³ : NormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : NormedAddCommGroup W
    inst✝ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    f : V → E
    hf : Continuous f
    n : Nat
    ⊢ Continuous fun x => ((ContinuousMultilinearMap.smulRightL Real (fun x => W)  …
  -/
  apply (smulRightL ℝ (fun (_ : Fin n) ↦ W) E).continuous₂.comp₂ _ hf
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Complex E
    V : Type u_2
    W : Type u_3
    inst✝³ : NormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : NormedAddCommGroup W
    inst✝ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    f : V → E
    hf : Continuous f
    n : Nat
    ⊢ Continuous fun w => (ContinuousMultilinearMap.mkPiAlgebra Real (Fin n) Real) …
  -/
  exact Continuous.comp (map_continuous _) (continuous_pi (fun _ ↦ L.continuous))
  /-
    🎉 no goals
  -/


lemma _root_.ContDiff.fourierPowSMulRight
    {f : V → E} {k : WithTop ℕ∞} (hf : ContDiff ℝ k f) (n : ℕ) :
    ContDiff ℝ k (fun v ↦ fourierPowSMulRight L f v n) := by
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Complex E
    V : Type u_2
    W : Type u_3
    inst✝³ : NormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : NormedAddCommGroup W
    inst✝ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    f : V → E
    k : WithTop ENat
    hf : ContDiff Real k f
    n : Nat
    ⊢ ContDiff Real k fun v => VectorFourier.fourierPowSMulRight L f v n
  -/
  simp_rw [fourierPowSMulRight_eq_comp]
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Complex E
    V : Type u_2
    W : Type u_3
    inst✝³ : NormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : NormedAddCommGroup W
    inst✝ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    f : V → E
    k : WithTop ENat
    hf : ContDiff Real k f
    n : Nat
    ⊢ ContDiff Real k fun v => HSMul.hSMul (HPow.hPow (Neg.neg (HMul.hMul (HMul.hM …
  -/
  apply ContDiff.const_smul
  /-
    case hf
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Complex E
    V : Type u_2
    W : Type u_3
    inst✝³ : NormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : NormedAddCommGroup W
    inst✝ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    f : V → E
    k : WithTop ENat
    hf : ContDiff Real k f
    n : Nat
    ⊢ ContDiff Real k fun y => ((ContinuousMultilinearMap.smulRightL Real (fun x = …
  -/
  apply (smulRightL ℝ (fun (_ : Fin n) ↦ W) E).isBoundedBilinearMap.contDiff.comp₂ _ hf
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Complex E
    V : Type u_2
    W : Type u_3
    inst✝³ : NormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : NormedAddCommGroup W
    inst✝ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    f : V → E
    k : WithTop ENat
    hf : ContDiff Real k f
    n : Nat
    ⊢ ContDiff Real k fun x => (ContinuousMultilinearMap.mkPiAlgebra Real (Fin n)  …
  -/
  apply (ContinuousMultilinearMap.contDiff _).comp
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Complex E
    V : Type u_2
    W : Type u_3
    inst✝³ : NormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : NormedAddCommGroup W
    inst✝ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    f : V → E
    k : WithTop ENat
    hf : ContDiff Real k f
    n : Nat
    ⊢ ContDiff Real k fun x x_1 => L x
  -/
  exact contDiff_pi.2 (fun _ ↦ L.contDiff)
  /-
    🎉 no goals
  -/


lemma norm_fourierPowSMulRight_le (f : V → E) (v : V) (n : ℕ) :
    ‖fourierPowSMulRight L f v n‖ ≤ (2 * π * ‖L‖) ^ n * ‖v‖ ^ n * ‖f v‖ := by
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Complex E
    V : Type u_2
    W : Type u_3
    inst✝³ : NormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : NormedAddCommGroup W
    inst✝ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    f : V → E
    v : V
    n : Nat
    ⊢ LE.le (Norm.norm (VectorFourier.fourierPowSMulRight L f v n)) (HMul.hMul (HM …
  -/
  apply ContinuousMultilinearMap.opNorm_le_bound (by positivity) (fun m ↦ ?_)
  calc
  ‖fourierPowSMulRight L f v n m‖
    = (2 * π) ^ n * ((∏ x : Fin n, |(L v) (m x)|) * ‖f v‖) := by
      simp [_root_.abs_of_nonneg pi_nonneg, norm_smul]
  _ ≤ (2 * π) ^ n * ((∏ x : Fin n, ‖L‖ * ‖v‖ * ‖m x‖) * ‖f v‖) := by
      gcongr with i _hi
      exact L.le_opNorm₂ v (m i)
  _ = (2 * π * ‖L‖) ^ n * ‖v‖ ^ n * ‖f v‖ * ∏ i : Fin n, ‖m i‖ := by
      simp [Finset.prod_mul_distrib, mul_pow]; ring


set_option maxSynthPendingDepth 2 in
/-- The iterated derivative of a function multiplied by `(L v ⬝) ^ n` can be controlled in terms
of the iterated derivatives of the initial function. -/
lemma norm_iteratedFDeriv_fourierPowSMulRight
    {f : V → E} {K : WithTop ℕ∞} {C : ℝ} (hf : ContDiff ℝ K f) {n : ℕ} {k : ℕ} (hk : k ≤ K)
    {v : V} (hv : ∀ i ≤ k, ∀ j ≤ n, ‖v‖ ^ j * ‖iteratedFDeriv ℝ i f v‖ ≤ C) :
    ‖iteratedFDeriv ℝ k (fun v ↦ fourierPowSMulRight L f v n) v‖ ≤
      (2 * π) ^ n * (2 * n + 2) ^ k * ‖L‖ ^ n * C := by
  /- We write `fourierPowSMulRight L f v n` as a composition of bilinear and multilinear maps,
  thanks to `fourierPowSMulRight_eq_comp`, and then we control the iterated derivatives of these
  thanks to general bounds on derivatives of bilinear and multilinear maps. More precisely,
  `fourierPowSMulRight L f v n m = (- (2 * π * I))^n • (∏ i, L v (m i)) • f v`. Here,
  `(- (2 * π * I))^n` contributes `(2π)^n` to the bound. The second product is bilinear, so the
  iterated derivative is controlled as a weighted sum of those of `v ↦ ∏ i, L v (m i)` and of `f`.

  The harder part is to control the iterated derivatives of `v ↦ ∏ i, L v (m i)`. For this, one
  argues that this is multilinear in `v`, to apply general bounds for iterated derivatives of
  multilinear maps. More precisely, we write it as the composition of a multilinear map `T` (making
  the product operation) and the tuple of linear maps `v ↦ (L v ⬝, ..., L v ⬝)` -/
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Complex E
    V : Type u_2
    W : Type u_3
    inst✝³ : NormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : NormedAddCommGroup W
    inst✝ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    f : V → E
    K : WithTop ENat
    C : Real
    hf : ContDiff Real K f
    n k : Nat
    hk : LE.le (↑k) K
    v : V
    hv : ∀ (i : Nat), LE.le i k → ∀ (j : Nat), LE.le j n → LE.le (HMul.hMul (HPow. …
    ⊢ LE.le (Norm.norm (iteratedFDeriv Real k (fun v => VectorFourier.fourierPowSM …
  -/
  simp_rw [fourierPowSMulRight_eq_comp]
  -- first step: controlling the iterated derivatives of `v ↦ ∏ i, L v (m i)`, written below
  -- as `v ↦ T (fun _ ↦ L v)`, or `T ∘ (ContinuousLinearMap.pi (fun (_ : Fin n) ↦ L))`.
  let T : (W →L[ℝ] ℝ) [×n]→L[ℝ] (W [×n]→L[ℝ] ℝ) :=
    compContinuousLinearMapLRight (ContinuousMultilinearMap.mkPiAlgebra ℝ (Fin n) ℝ)
  have I₁ m : ‖iteratedFDeriv ℝ m T (fun _ ↦ L v)‖ ≤
      n.descFactorial m * 1 * (‖L‖ * ‖v‖) ^ (n - m) := by
    have : ‖T‖ ≤ 1 := by
      apply (norm_compContinuousLinearMapLRight_le _ _).trans
      simp only [norm_mkPiAlgebra, le_refl]
    apply (ContinuousMultilinearMap.norm_iteratedFDeriv_le _ _ _).trans
    simp only [Fintype.card_fin]
    gcongr
    refine (pi_norm_le_iff_of_nonneg (by positivity)).mpr (fun _ ↦ ?_)
    exact ContinuousLinearMap.le_opNorm _ _
  have I₂ m : ‖iteratedFDeriv ℝ m (T ∘ (ContinuousLinearMap.pi (fun (_ : Fin n) ↦ L))) v‖ ≤
      (n.descFactorial m * 1 * (‖L‖ * ‖v‖) ^ (n - m)) * ‖L‖ ^ m := by
    rw [ContinuousLinearMap.iteratedFDeriv_comp_right _ (ContinuousMultilinearMap.contDiff _)
      _ (mod_cast le_top)]
    apply (norm_compContinuousLinearMap_le _ _).trans
    simp only [Finset.prod_const, Finset.card_fin]
    gcongr
    · exact I₁ m
    · exact ContinuousLinearMap.norm_pi_le_of_le (fun _ ↦ le_rfl) (norm_nonneg _)
  have I₃ m : ‖iteratedFDeriv ℝ m (T ∘ (ContinuousLinearMap.pi (fun (_ : Fin n) ↦ L))) v‖ ≤
      n.descFactorial m * ‖L‖ ^ n * ‖v‖ ^ (n - m) := by
    apply (I₂ m).trans (le_of_eq _)
    rcases le_or_lt m n with hm | hm
    · rw [show ‖L‖ ^ n = ‖L‖ ^ (m + (n - m)) by rw [Nat.add_sub_cancel' hm], pow_add]
      ring
    · simp only [Nat.descFactorial_eq_zero_iff_lt.mpr hm, CharP.cast_eq_zero, mul_one, zero_mul]
  -- second step: factor out the `(2 * π) ^ n` factor, and cancel it on both sides.
  have A : ContDiff ℝ K (fun y ↦ T (fun _ ↦ L y)) :=
    (ContinuousMultilinearMap.contDiff _).comp (contDiff_pi.2 fun _ ↦ L.contDiff)
  rw [iteratedFDeriv_const_smul_apply' (hf := (smulRightL ℝ (fun _ ↦ W)
    E).isBoundedBilinearMap.contDiff.comp₂ (A.of_le hk) (hf.of_le hk)),
    norm_smul (β := V [×k]→L[ℝ] (W [×n]→L[ℝ] E))]
  simp only [norm_pow, norm_neg, norm_mul, RCLike.norm_ofNat, Complex.norm_eq_abs, abs_ofReal,
    _root_.abs_of_nonneg pi_nonneg, abs_I, mul_one, mul_assoc]
  /-
    E : Type u_1
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Complex E
    V : Type u_2
    W : Type u_3
    inst✝³ : NormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : NormedAddCommGroup W
    inst✝ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    f : V → E
    K : WithTop ENat
    C : Real
    hf : ContDiff Real K f
    n k : Nat
    hk : LE.le (↑k) K
    v : V
    hv : ∀ (i : Nat), LE.le i k → ∀ (j : Nat), LE.le j n → LE.le (HMul.hMul (HPow. …
    T : ContinuousMultilinearMap Real (fun i => ContinuousLinearMap (RingHom.id Re …
    I₁ : ∀ (m : Nat), LE.le (Norm.norm (iteratedFDeriv Real m ⇑T fun x => L v)) (H …
    I₂ : ∀ (m : Nat), LE.le (Norm.norm (iteratedFDeriv Real m (Function.comp ⇑T ⇑( …
    I₃ : ∀ (m : Nat), LE.le (Norm.norm (iteratedFDeriv Real m (Function.comp ⇑T ⇑( …
    A : ContDiff Real K fun y => T fun x => L y
    ⊢ LE.le (HMul.hMul (HPow.hPow (HMul.hMul 2 Real.pi) n) (Norm.norm (iteratedFDe …
  -/
  gcongr
  -- third step: argue that the scalar multiplication is bilinear to bound the iterated derivatives
  -- of `v ↦ (∏ i, L v (m i)) • f v` in terms of those of `v ↦ (∏ i, L v (m i))` and of `f`.
  -- The former are controlled by the first step, the latter by the assumptions.
  apply (ContinuousLinearMap.norm_iteratedFDeriv_le_of_bilinear_of_le_one _ A hf _
    hk ContinuousMultilinearMap.norm_smulRightL_le).trans
  calc
  ∑ i in Finset.range (k + 1),
    k.choose i * ‖iteratedFDeriv ℝ i (fun (y : V) ↦ T (fun _ ↦ L y)) v‖ *
      ‖iteratedFDeriv ℝ (k - i) f v‖
    ≤ ∑ i in Finset.range (k + 1),
      k.choose i * (n.descFactorial i * ‖L‖ ^ n * ‖v‖ ^ (n - i)) *
        ‖iteratedFDeriv ℝ (k - i) f v‖ := by
    gcongr with i _hi
    exact I₃ i
  _ = ∑ i in Finset.range (k + 1), (k.choose i * n.descFactorial i * ‖L‖ ^ n) *
        (‖v‖ ^ (n - i) * ‖iteratedFDeriv ℝ (k - i) f v‖) := by
    congr with i
    ring
  _ ≤ ∑ i in Finset.range (k + 1), (k.choose i * (n + 1 : ℕ) ^ k * ‖L‖ ^ n) * C := by
    gcongr with i hi
    · rw [← Nat.cast_pow, Nat.cast_le]
      calc n.descFactorial i ≤ n ^ i := Nat.descFactorial_le_pow _ _
      _ ≤ (n + 1) ^ i := by gcongr; omega
      _ ≤ (n + 1) ^ k := by gcongr; exacts [le_add_self, Finset.mem_range_succ_iff.mp hi]
    · exact hv _ (by omega) _ (by omega)
  _ = (2 * n + 2) ^ k * (‖L‖^n * C) := by
    simp only [← Finset.sum_mul, ← Nat.cast_sum, Nat.sum_range_choose, mul_one, ← mul_assoc,
      Nat.cast_pow, Nat.cast_ofNat, Nat.cast_add, Nat.cast_one, ← mul_pow, mul_add]


lemma _root_.MeasureTheory.AEStronglyMeasurable.fourierPowSMulRight
    (hf : AEStronglyMeasurable f μ) (n : ℕ) :
    AEStronglyMeasurable (fun v ↦ fourierPowSMulRight L f v n) μ := by
  /-
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Complex E
    V : Type u_2
    W : Type u_3
    inst✝⁶ : NormedAddCommGroup V
    inst✝⁵ : NormedSpace Real V
    inst✝⁴ : NormedAddCommGroup W
    inst✝³ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    f : V → E
    inst✝² : MeasurableSpace V
    inst✝¹ : BorelSpace V
    μ : MeasureTheory.Measure V
    inst✝ : SecondCountableTopology V
    hf : MeasureTheory.AEStronglyMeasurable f μ
    n : Nat
    ⊢ MeasureTheory.AEStronglyMeasurable (fun v => VectorFourier.fourierPowSMulRig …
  -/
  simp_rw [fourierPowSMulRight_eq_comp]
  /-
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Complex E
    V : Type u_2
    W : Type u_3
    inst✝⁶ : NormedAddCommGroup V
    inst✝⁵ : NormedSpace Real V
    inst✝⁴ : NormedAddCommGroup W
    inst✝³ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    f : V → E
    inst✝² : MeasurableSpace V
    inst✝¹ : BorelSpace V
    μ : MeasureTheory.Measure V
    inst✝ : SecondCountableTopology V
    hf : MeasureTheory.AEStronglyMeasurable f μ
    n : Nat
    ⊢ MeasureTheory.AEStronglyMeasurable (fun v => HSMul.hSMul (HPow.hPow (Neg.neg …
  -/
  apply AEStronglyMeasurable.const_smul'
  /-
    case hf
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Complex E
    V : Type u_2
    W : Type u_3
    inst✝⁶ : NormedAddCommGroup V
    inst✝⁵ : NormedSpace Real V
    inst✝⁴ : NormedAddCommGroup W
    inst✝³ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    f : V → E
    inst✝² : MeasurableSpace V
    inst✝¹ : BorelSpace V
    μ : MeasureTheory.Measure V
    inst✝ : SecondCountableTopology V
    hf : MeasureTheory.AEStronglyMeasurable f μ
    n : Nat
    ⊢ MeasureTheory.AEStronglyMeasurable (fun x => ((ContinuousMultilinearMap.smul …
  -/
  apply (smulRightL ℝ (fun (_ : Fin n) ↦ W) E).continuous₂.comp_aestronglyMeasurable₂ _ hf
  /-
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Complex E
    V : Type u_2
    W : Type u_3
    inst✝⁶ : NormedAddCommGroup V
    inst✝⁵ : NormedSpace Real V
    inst✝⁴ : NormedAddCommGroup W
    inst✝³ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    f : V → E
    inst✝² : MeasurableSpace V
    inst✝¹ : BorelSpace V
    μ : MeasureTheory.Measure V
    inst✝ : SecondCountableTopology V
    hf : MeasureTheory.AEStronglyMeasurable f μ
    n : Nat
    ⊢ MeasureTheory.AEStronglyMeasurable (fun x => (ContinuousMultilinearMap.mkPiA …
  -/
  apply Continuous.aestronglyMeasurable
  /-
    case hf
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Complex E
    V : Type u_2
    W : Type u_3
    inst✝⁶ : NormedAddCommGroup V
    inst✝⁵ : NormedSpace Real V
    inst✝⁴ : NormedAddCommGroup W
    inst✝³ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    f : V → E
    inst✝² : MeasurableSpace V
    inst✝¹ : BorelSpace V
    μ : MeasureTheory.Measure V
    inst✝ : SecondCountableTopology V
    hf : MeasureTheory.AEStronglyMeasurable f μ
    n : Nat
    ⊢ Continuous fun x => (ContinuousMultilinearMap.mkPiAlgebra Real (Fin n) Real) …
  -/
  exact Continuous.comp (map_continuous _) (continuous_pi (fun _ ↦ L.continuous))
  /-
    🎉 no goals
  -/


lemma integrable_fourierPowSMulRight {n : ℕ} (hf : Integrable (fun v ↦ ‖v‖ ^ n * ‖f v‖) μ)
    (h'f : AEStronglyMeasurable f μ) : Integrable (fun v ↦ fourierPowSMulRight L f v n) μ := by
  /-
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Complex E
    V : Type u_2
    W : Type u_3
    inst✝⁶ : NormedAddCommGroup V
    inst✝⁵ : NormedSpace Real V
    inst✝⁴ : NormedAddCommGroup W
    inst✝³ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    f : V → E
    inst✝² : MeasurableSpace V
    inst✝¹ : BorelSpace V
    μ : MeasureTheory.Measure V
    inst✝ : SecondCountableTopology V
    n : Nat
    hf : MeasureTheory.Integrable (fun v => HMul.hMul (HPow.hPow (Norm.norm v) n)  …
    h'f : MeasureTheory.AEStronglyMeasurable f μ
    ⊢ MeasureTheory.Integrable (fun v => VectorFourier.fourierPowSMulRight L f v n …
  -/
  refine (hf.const_mul ((2 * π * ‖L‖) ^ n)).mono' (h'f.fourierPowSMulRight L n) ?_
  /-
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Complex E
    V : Type u_2
    W : Type u_3
    inst✝⁶ : NormedAddCommGroup V
    inst✝⁵ : NormedSpace Real V
    inst✝⁴ : NormedAddCommGroup W
    inst✝³ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    f : V → E
    inst✝² : MeasurableSpace V
    inst✝¹ : BorelSpace V
    μ : MeasureTheory.Measure V
    inst✝ : SecondCountableTopology V
    n : Nat
    hf : MeasureTheory.Integrable (fun v => HMul.hMul (HPow.hPow (Norm.norm v) n)  …
    h'f : MeasureTheory.AEStronglyMeasurable f μ
    ⊢ Filter.Eventually (fun a => LE.le (Norm.norm (VectorFourier.fourierPowSMulRi …
  -/
  filter_upwards with v
  /-
    case h
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Complex E
    V : Type u_2
    W : Type u_3
    inst✝⁶ : NormedAddCommGroup V
    inst✝⁵ : NormedSpace Real V
    inst✝⁴ : NormedAddCommGroup W
    inst✝³ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    f : V → E
    inst✝² : MeasurableSpace V
    inst✝¹ : BorelSpace V
    μ : MeasureTheory.Measure V
    inst✝ : SecondCountableTopology V
    n : Nat
    hf : MeasureTheory.Integrable (fun v => HMul.hMul (HPow.hPow (Norm.norm v) n)  …
    h'f : MeasureTheory.AEStronglyMeasurable f μ
    v : V
    ⊢ LE.le (Norm.norm (VectorFourier.fourierPowSMulRight L f v n)) (HMul.hMul (HP …
  -/
  exact (norm_fourierPowSMulRight_le L f v n).trans (le_of_eq (by ring))
  /-
    🎉 no goals
  -/


lemma hasFTaylorSeriesUpTo_fourierIntegral {N : WithTop ℕ∞}
    (hf : ∀ (n : ℕ), n ≤ N → Integrable (fun v ↦ ‖v‖^n * ‖f v‖) μ)
    (h'f : AEStronglyMeasurable f μ) :
    HasFTaylorSeriesUpTo N (fourierIntegral 𝐞 μ L.toLinearMap₂ f)
      (fun w n ↦ fourierIntegral 𝐞 μ L.toLinearMap₂ (fun v ↦ fourierPowSMulRight L f v n) w) := by
  /-
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Complex E
    V : Type u_2
    W : Type u_3
    inst✝⁶ : NormedAddCommGroup V
    inst✝⁵ : NormedSpace Real V
    inst✝⁴ : NormedAddCommGroup W
    inst✝³ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    f : V → E
    inst✝² : MeasurableSpace V
    inst✝¹ : BorelSpace V
    μ : MeasureTheory.Measure V
    inst✝ : SecondCountableTopology V
    N : WithTop ENat
    hf : ∀ (n : Nat), LE.le (↑n) N → MeasureTheory.Integrable (fun v => HMul.hMul  …
    h'f : MeasureTheory.AEStronglyMeasurable f μ
    ⊢ HasFTaylorSeriesUpTo N (VectorFourier.fourierIntegral Real.fourierChar μ L.t …
  -/
  constructor
    /-
      case zero_eq
      E : Type u_1
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Complex E
      V : Type u_2
      W : Type u_3
      inst✝⁶ : NormedAddCommGroup V
      inst✝⁵ : NormedSpace Real V
      inst✝⁴ : NormedAddCommGroup W
      inst✝³ : NormedSpace Real W
      L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
      f : V → E
      inst✝² : MeasurableSpace V
      inst✝¹ : BorelSpace V
      μ : MeasureTheory.Measure V
      inst✝ : SecondCountableTopology V
      N : WithTop ENat
      hf : ∀ (n : Nat), LE.le (↑n) N → MeasureTheory.Integrable (fun v => HMul.hMul  …
      h'f : MeasureTheory.AEStronglyMeasurable f μ
      ⊢ ∀ (x : W), Eq (VectorFourier.fourierIntegral Real.fourierChar μ L.toLinearMa …
    -/
  · intro w
    rw [curry0_apply, Matrix.zero_empty, fourierIntegral_continuousMultilinearMap_apply'
      (integrable_fourierPowSMulRight L (hf 0 bot_le) h'f)]
    simp only [fourierPowSMulRight_apply, pow_zero, Finset.univ_eq_empty, Finset.prod_empty,
      one_smul]
    /-
      case fderiv
      E : Type u_1
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Complex E
      V : Type u_2
      W : Type u_3
      inst✝⁶ : NormedAddCommGroup V
      inst✝⁵ : NormedSpace Real V
      inst✝⁴ : NormedAddCommGroup W
      inst✝³ : NormedSpace Real W
      L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
      f : V → E
      inst✝² : MeasurableSpace V
      inst✝¹ : BorelSpace V
      μ : MeasureTheory.Measure V
      inst✝ : SecondCountableTopology V
      N : WithTop ENat
      hf : ∀ (n : Nat), LE.le (↑n) N → MeasureTheory.Integrable (fun v => HMul.hMul  …
      h'f : MeasureTheory.AEStronglyMeasurable f μ
      ⊢ ∀ (m : Nat), LT.lt (↑m) N → ∀ (x : W), HasFDerivAt (fun y => VectorFourier.f …
    -/
  · intro n hn w
    have I₁ : Integrable (fun v ↦ fourierPowSMulRight L f v n) μ :=
      integrable_fourierPowSMulRight L (hf n hn.le) h'f
    have I₂ : Integrable (fun v ↦ ‖v‖ * ‖fourierPowSMulRight L f v n‖) μ := by
      apply ((hf (n+1) (ENat.add_one_natCast_le_withTop_of_lt hn)).const_mul
          ((2 * π * ‖L‖) ^ n)).mono'
        (continuous_norm.aestronglyMeasurable.mul (h'f.fourierPowSMulRight L n).norm)
      filter_upwards with v
      simp only [Pi.mul_apply, norm_mul, norm_norm]
      calc
      ‖v‖ * ‖fourierPowSMulRight L f v n‖
        ≤ ‖v‖ * ((2 * π * ‖L‖) ^ n * ‖v‖ ^ n * ‖f v‖) := by
          gcongr; apply norm_fourierPowSMulRight_le
      _ = (2 * π * ‖L‖) ^ n * (‖v‖ ^ (n + 1) * ‖f v‖) := by rw [pow_succ]; ring
    have I₃ : Integrable (fun v ↦ fourierPowSMulRight L f v (n + 1)) μ :=
      integrable_fourierPowSMulRight L (hf (n + 1) (ENat.add_one_natCast_le_withTop_of_lt hn)) h'f
    have I₄ : Integrable
        (fun v ↦ fourierSMulRight L (fun v ↦ fourierPowSMulRight L f v n) v) μ := by
      apply (I₂.const_mul ((2 * π * ‖L‖))).mono' (h'f.fourierPowSMulRight L n).fourierSMulRight
      filter_upwards with v
      exact (norm_fourierSMulRight_le _ _ _).trans (le_of_eq (by ring))
    have E : curryLeft
          (fourierIntegral 𝐞 μ L.toLinearMap₂ (fun v ↦ fourierPowSMulRight L f v (n + 1)) w) =
        fourierIntegral 𝐞 μ L.toLinearMap₂
          (fourierSMulRight L fun v ↦ fourierPowSMulRight L f v n) w := by
      ext w' m
      rw [curryLeft_apply, fourierIntegral_continuousMultilinearMap_apply' I₃,
        fourierIntegral_continuousLinearMap_apply' I₄,
        fourierIntegral_continuousMultilinearMap_apply' (I₄.apply_continuousLinearMap _)]
      congr with v
      simp only [fourierPowSMulRight_apply, mul_comm, pow_succ, neg_mul, Fin.prod_univ_succ,
        Fin.cons_zero, Fin.cons_succ, neg_smul, fourierSMulRight_apply, neg_apply, smul_apply,
        smul_comm (M := ℝ) (N := ℂ) (α := E), smul_smul]
    /-
      case fderiv
      E✝ : Type u_1
      inst✝⁸ : NormedAddCommGroup E✝
      inst✝⁷ : NormedSpace Complex E✝
      V : Type u_2
      W : Type u_3
      inst✝⁶ : NormedAddCommGroup V
      inst✝⁵ : NormedSpace Real V
      inst✝⁴ : NormedAddCommGroup W
      inst✝³ : NormedSpace Real W
      L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
      f : V → E✝
      inst✝² : MeasurableSpace V
      inst✝¹ : BorelSpace V
      μ : MeasureTheory.Measure V
      inst✝ : SecondCountableTopology V
      N : WithTop ENat
      hf : ∀ (n : Nat), LE.le (↑n) N → MeasureTheory.Integrable (fun v => HMul.hMul  …
      h'f : MeasureTheory.AEStronglyMeasurable f μ
      n : Nat
      hn : LT.lt (↑n) N
      w : W
      I₁ : MeasureTheory.Integrable (fun v => VectorFourier.fourierPowSMulRight L f  …
      I₂ : MeasureTheory.Integrable (fun v => HMul.hMul (Norm.norm v) (Norm.norm (Ve …
      I₃ : MeasureTheory.Integrable (fun v => VectorFourier.fourierPowSMulRight L f  …
      I₄ : MeasureTheory.Integrable (fun v => VectorFourier.fourierSMulRight L (fun  …
      E : Eq (VectorFourier.fourierIntegral Real.fourierChar μ L.toLinearMap₂ (fun v …
      ⊢ HasFDerivAt (fun y => VectorFourier.fourierIntegral Real.fourierChar μ L.toL …
    -/
    exact E ▸ hasFDerivAt_fourierIntegral L I₁ I₂ w
    /-
      🎉 no goals
    -/
    /-
      case cont
      E : Type u_1
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Complex E
      V : Type u_2
      W : Type u_3
      inst✝⁶ : NormedAddCommGroup V
      inst✝⁵ : NormedSpace Real V
      inst✝⁴ : NormedAddCommGroup W
      inst✝³ : NormedSpace Real W
      L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
      f : V → E
      inst✝² : MeasurableSpace V
      inst✝¹ : BorelSpace V
      μ : MeasureTheory.Measure V
      inst✝ : SecondCountableTopology V
      N : WithTop ENat
      hf : ∀ (n : Nat), LE.le (↑n) N → MeasureTheory.Integrable (fun v => HMul.hMul  …
      h'f : MeasureTheory.AEStronglyMeasurable f μ
      ⊢ ∀ (m : Nat), LE.le (↑m) N → Continuous fun x => VectorFourier.fourierIntegra …
    -/
  · intro n hn
    /-
      case cont
      E : Type u_1
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Complex E
      V : Type u_2
      W : Type u_3
      inst✝⁶ : NormedAddCommGroup V
      inst✝⁵ : NormedSpace Real V
      inst✝⁴ : NormedAddCommGroup W
      inst✝³ : NormedSpace Real W
      L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
      f : V → E
      inst✝² : MeasurableSpace V
      inst✝¹ : BorelSpace V
      μ : MeasureTheory.Measure V
      inst✝ : SecondCountableTopology V
      N : WithTop ENat
      hf : ∀ (n : Nat), LE.le (↑n) N → MeasureTheory.Integrable (fun v => HMul.hMul  …
      h'f : MeasureTheory.AEStronglyMeasurable f μ
      n : Nat
      hn : LE.le (↑n) N
      ⊢ Continuous fun x => VectorFourier.fourierIntegral Real.fourierChar μ L.toLin …
    -/
    apply fourierIntegral_continuous Real.continuous_fourierChar (by apply L.continuous₂)
    /-
      case cont
      E : Type u_1
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Complex E
      V : Type u_2
      W : Type u_3
      inst✝⁶ : NormedAddCommGroup V
      inst✝⁵ : NormedSpace Real V
      inst✝⁴ : NormedAddCommGroup W
      inst✝³ : NormedSpace Real W
      L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
      f : V → E
      inst✝² : MeasurableSpace V
      inst✝¹ : BorelSpace V
      μ : MeasureTheory.Measure V
      inst✝ : SecondCountableTopology V
      N : WithTop ENat
      hf : ∀ (n : Nat), LE.le (↑n) N → MeasureTheory.Integrable (fun v => HMul.hMul  …
      h'f : MeasureTheory.AEStronglyMeasurable f μ
      n : Nat
      hn : LE.le (↑n) N
      ⊢ MeasureTheory.Integrable (fun v => VectorFourier.fourierPowSMulRight L f v n …
    -/
    exact integrable_fourierPowSMulRight L (hf n hn) h'f
    /-
      🎉 no goals
    -/


/-- Variant of `hasFTaylorSeriesUpTo_fourierIntegral` in which the smoothness index is restricted
to `ℕ∞` (and so are the inequalities in the assumption `hf`). Avoids normcasting in some
applications. -/
lemma hasFTaylorSeriesUpTo_fourierIntegral' {N : ℕ∞}
    (hf : ∀ (n : ℕ), n ≤ N → Integrable (fun v ↦ ‖v‖^n * ‖f v‖) μ)
    (h'f : AEStronglyMeasurable f μ) :
    HasFTaylorSeriesUpTo N (fourierIntegral 𝐞 μ L.toLinearMap₂ f)
      (fun w n ↦ fourierIntegral 𝐞 μ L.toLinearMap₂ (fun v ↦ fourierPowSMulRight L f v n) w) :=
  hasFTaylorSeriesUpTo_fourierIntegral _ (fun n hn ↦ hf n (mod_cast hn)) h'f


/-- If `‖v‖^n * ‖f v‖` is integrable for all `n ≤ N`, then the Fourier transform of `f` is `C^N`. -/
theorem contDiff_fourierIntegral {N : ℕ∞}
    (hf : ∀ (n : ℕ), n ≤ N → Integrable (fun v ↦ ‖v‖^n * ‖f v‖) μ) :
    ContDiff ℝ N (fourierIntegral 𝐞 μ L.toLinearMap₂ f) := by
  /-
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Complex E
    V : Type u_2
    W : Type u_3
    inst✝⁶ : NormedAddCommGroup V
    inst✝⁵ : NormedSpace Real V
    inst✝⁴ : NormedAddCommGroup W
    inst✝³ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    f : V → E
    inst✝² : MeasurableSpace V
    inst✝¹ : BorelSpace V
    μ : MeasureTheory.Measure V
    inst✝ : SecondCountableTopology V
    N : ENat
    hf : ∀ (n : Nat), LE.le (↑n) N → MeasureTheory.Integrable (fun v => HMul.hMul  …
    ⊢ ContDiff Real (↑N) (VectorFourier.fourierIntegral Real.fourierChar μ L.toLin …
  -/
  by_cases h'f : Integrable f μ
    /-
      case pos
      E : Type u_1
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Complex E
      V : Type u_2
      W : Type u_3
      inst✝⁶ : NormedAddCommGroup V
      inst✝⁵ : NormedSpace Real V
      inst✝⁴ : NormedAddCommGroup W
      inst✝³ : NormedSpace Real W
      L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
      f : V → E
      inst✝² : MeasurableSpace V
      inst✝¹ : BorelSpace V
      μ : MeasureTheory.Measure V
      inst✝ : SecondCountableTopology V
      N : ENat
      hf : ∀ (n : Nat), LE.le (↑n) N → MeasureTheory.Integrable (fun v => HMul.hMul  …
      h'f : MeasureTheory.Integrable f μ
      ⊢ ContDiff Real (↑N) (VectorFourier.fourierIntegral Real.fourierChar μ L.toLin …
    -/
  · exact (hasFTaylorSeriesUpTo_fourierIntegral' L hf h'f.1).contDiff
    /-
      🎉 no goals
    -/
  · have : fourierIntegral 𝐞 μ L.toLinearMap₂ f = 0 := by
      ext w; simp [fourierIntegral, integral, h'f]
    /-
      case neg
      E : Type u_1
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Complex E
      V : Type u_2
      W : Type u_3
      inst✝⁶ : NormedAddCommGroup V
      inst✝⁵ : NormedSpace Real V
      inst✝⁴ : NormedAddCommGroup W
      inst✝³ : NormedSpace Real W
      L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
      f : V → E
      inst✝² : MeasurableSpace V
      inst✝¹ : BorelSpace V
      μ : MeasureTheory.Measure V
      inst✝ : SecondCountableTopology V
      N : ENat
      hf : ∀ (n : Nat), LE.le (↑n) N → MeasureTheory.Integrable (fun v => HMul.hMul  …
      h'f : Not (MeasureTheory.Integrable f μ)
      this : Eq (VectorFourier.fourierIntegral Real.fourierChar μ L.toLinearMap₂ f) 0
      ⊢ ContDiff Real (↑N) (VectorFourier.fourierIntegral Real.fourierChar μ L.toLin …
    -/
    simpa [this] using contDiff_const
    /-
      🎉 no goals
    -/


/-- If `‖v‖^n * ‖f v‖` is integrable for all `n ≤ N`, then the `n`-th derivative of the Fourier
transform of `f` is the Fourier transform of `fourierPowSMulRight L f v n`,
i.e., `(L v ⬝) ^ n • f v`. -/
lemma iteratedFDeriv_fourierIntegral {N : ℕ∞}
    (hf : ∀ (n : ℕ), n ≤ N → Integrable (fun v ↦ ‖v‖^n * ‖f v‖) μ)
    (h'f : AEStronglyMeasurable f μ) {n : ℕ} (hn : n ≤ N) :
    iteratedFDeriv ℝ n (fourierIntegral 𝐞 μ L.toLinearMap₂ f) =
      fourierIntegral 𝐞 μ L.toLinearMap₂ (fun v ↦ fourierPowSMulRight L f v n) := by
  /-
    E : Type u_1
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Complex E
    V : Type u_2
    W : Type u_3
    inst✝⁶ : NormedAddCommGroup V
    inst✝⁵ : NormedSpace Real V
    inst✝⁴ : NormedAddCommGroup W
    inst✝³ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    f : V → E
    inst✝² : MeasurableSpace V
    inst✝¹ : BorelSpace V
    μ : MeasureTheory.Measure V
    inst✝ : SecondCountableTopology V
    N : ENat
    hf : ∀ (n : Nat), LE.le (↑n) N → MeasureTheory.Integrable (fun v => HMul.hMul  …
    h'f : MeasureTheory.AEStronglyMeasurable f μ
    n : Nat
    hn : LE.le (↑n) N
    ⊢ Eq (iteratedFDeriv Real n (VectorFourier.fourierIntegral Real.fourierChar μ  …
  -/
  ext w : 1
  exact ((hasFTaylorSeriesUpTo_fourierIntegral' L hf h'f).eq_iteratedFDeriv
    (mod_cast hn) w).symm


/-- The Fourier integral of the `n`-th derivative of a function is obtained by multiplying the
Fourier integral of the original function by `(2πI L w ⬝ )^n`. -/
theorem fourierIntegral_iteratedFDeriv [FiniteDimensional ℝ V]
    {μ : Measure V} [Measure.IsAddHaarMeasure μ] {N : ℕ∞} (hf : ContDiff ℝ N f)
    (h'f : ∀ (n : ℕ), n ≤ N → Integrable (iteratedFDeriv ℝ n f) μ) {n : ℕ} (hn : n ≤ N) :
    fourierIntegral 𝐞 μ L.toLinearMap₂ (iteratedFDeriv ℝ n f)
      = (fun w ↦ fourierPowSMulRight (-L.flip) (fourierIntegral 𝐞 μ L.toLinearMap₂ f) w n) := by
  induction n with
  | zero =>
    ext w m
    simp only [iteratedFDeriv_zero_apply, fourierPowSMulRight_apply, pow_zero,
      Finset.univ_eq_empty, ContinuousLinearMap.neg_apply, ContinuousLinearMap.flip_apply,
      Finset.prod_empty, one_smul, fourierIntegral_continuousMultilinearMap_apply' ((h'f 0 bot_le))]
  | succ n ih =>
    ext w m
    have J : Integrable (fderiv ℝ (iteratedFDeriv ℝ n f)) μ := by
      specialize h'f (n + 1) hn
      rwa [iteratedFDeriv_succ_eq_comp_left, Function.comp_def,
          LinearIsometryEquiv.integrable_comp_iff (𝕜 := ℝ) (φ := fderiv ℝ (iteratedFDeriv ℝ n f))]
        at h'f
    suffices H : (fourierIntegral 𝐞 μ L.toLinearMap₂ (fderiv ℝ (iteratedFDeriv ℝ n f)) w)
          (m 0) (Fin.tail m) =
        (-(2 * π * I)) ^ (n + 1) • (∏ x : Fin (n + 1), -L (m x) w) • ∫ v, 𝐞 (-L v w) • f v ∂μ by
      rw [fourierIntegral_continuousMultilinearMap_apply' (h'f _ hn)]
      simp only [iteratedFDeriv_succ_apply_left, fourierPowSMulRight_apply,
        ContinuousLinearMap.neg_apply, ContinuousLinearMap.flip_apply]
      rw [← fourierIntegral_continuousMultilinearMap_apply' ((J.apply_continuousLinearMap _)),
          ← fourierIntegral_continuousLinearMap_apply' J]
      exact H
    have h'n : n < N := (Nat.cast_lt.mpr n.lt_succ_self).trans_le hn
    rw [fourierIntegral_fderiv _ (h'f n h'n.le)
      (hf.differentiable_iteratedFDeriv (mod_cast h'n)) J]
    simp only [ih h'n.le, fourierSMulRight_apply, ContinuousLinearMap.neg_apply,
      ContinuousLinearMap.flip_apply, neg_smul, smul_neg, neg_neg, smul_apply,
      fourierPowSMulRight_apply, ← coe_smul (E := E), smul_smul]
    congr 1
    simp only [ofReal_prod, ofReal_neg, pow_succ, mul_neg, Fin.prod_univ_succ, neg_mul,
      ofReal_mul, neg_neg, Fin.tail_def]
    ring


/-- The `k`-th derivative of the Fourier integral of `f`, multiplied by `(L v w) ^ n`, is the
Fourier integral of the `n`-th derivative of `(L v w) ^ k * f`. -/
theorem fourierPowSMulRight_iteratedFDeriv_fourierIntegral [FiniteDimensional ℝ V]
    {μ : Measure V} [Measure.IsAddHaarMeasure μ] {K N : ℕ∞} (hf : ContDiff ℝ N f)
    (h'f : ∀ (k n : ℕ), k ≤ K → n ≤ N → Integrable (fun v ↦ ‖v‖^k * ‖iteratedFDeriv ℝ n f v‖) μ)
    {k n : ℕ} (hk : k ≤ K) (hn : n ≤ N) {w : W} :
    fourierPowSMulRight (-L.flip)
      (iteratedFDeriv ℝ k (fourierIntegral 𝐞 μ L.toLinearMap₂ f)) w n =
    fourierIntegral 𝐞 μ L.toLinearMap₂
      (iteratedFDeriv ℝ n (fun v ↦ fourierPowSMulRight L f v k)) w := by
  /-
    E : Type u_1
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace Complex E
    V : Type u_2
    W : Type u_3
    inst✝⁷ : NormedAddCommGroup V
    inst✝⁶ : NormedSpace Real V
    inst✝⁵ : NormedAddCommGroup W
    inst✝⁴ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    f : V → E
    inst✝³ : MeasurableSpace V
    inst✝² : BorelSpace V
    inst✝¹ : FiniteDimensional Real V
    μ : MeasureTheory.Measure V
    inst✝ : μ.IsAddHaarMeasure
    K N : ENat
    hf : ContDiff Real (↑N) f
    h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
    k n : Nat
    hk : LE.le (↑k) K
    hn : LE.le (↑n) N
    w : W
    ⊢ Eq (VectorFourier.fourierPowSMulRight (Neg.neg L.flip) (iteratedFDeriv Real  …
  -/
  rw [fourierIntegral_iteratedFDeriv (N := N) _ (hf.fourierPowSMulRight _ _) _ hn]
    /-
      E : Type u_1
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace Complex E
      V : Type u_2
      W : Type u_3
      inst✝⁷ : NormedAddCommGroup V
      inst✝⁶ : NormedSpace Real V
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace Real W
      L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
      f : V → E
      inst✝³ : MeasurableSpace V
      inst✝² : BorelSpace V
      inst✝¹ : FiniteDimensional Real V
      μ : MeasureTheory.Measure V
      inst✝ : μ.IsAddHaarMeasure
      K N : ENat
      hf : ContDiff Real (↑N) f
      h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
      k n : Nat
      hk : LE.le (↑k) K
      hn : LE.le (↑n) N
      w : W
      ⊢ Eq (VectorFourier.fourierPowSMulRight (Neg.neg L.flip) (iteratedFDeriv Real  …
    -/
  · congr
    /-
      case e_f
      E : Type u_1
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace Complex E
      V : Type u_2
      W : Type u_3
      inst✝⁷ : NormedAddCommGroup V
      inst✝⁶ : NormedSpace Real V
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace Real W
      L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
      f : V → E
      inst✝³ : MeasurableSpace V
      inst✝² : BorelSpace V
      inst✝¹ : FiniteDimensional Real V
      μ : MeasureTheory.Measure V
      inst✝ : μ.IsAddHaarMeasure
      K N : ENat
      hf : ContDiff Real (↑N) f
      h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
      k n : Nat
      hk : LE.le (↑k) K
      hn : LE.le (↑n) N
      w : W
      ⊢ Eq (iteratedFDeriv Real k (VectorFourier.fourierIntegral Real.fourierChar μ  …
    -/
    rw [iteratedFDeriv_fourierIntegral (N := K) _ _ hf.continuous.aestronglyMeasurable hk]
    /-
      E : Type u_1
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace Complex E
      V : Type u_2
      W : Type u_3
      inst✝⁷ : NormedAddCommGroup V
      inst✝⁶ : NormedSpace Real V
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace Real W
      L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
      f : V → E
      inst✝³ : MeasurableSpace V
      inst✝² : BorelSpace V
      inst✝¹ : FiniteDimensional Real V
      μ : MeasureTheory.Measure V
      inst✝ : μ.IsAddHaarMeasure
      K N : ENat
      hf : ContDiff Real (↑N) f
      h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
      k n : Nat
      hk : LE.le (↑k) K
      hn : LE.le (↑n) N
      w : W
      ⊢ ∀ (n : Nat), LE.le (↑n) K → MeasureTheory.Integrable (fun v => HMul.hMul (HP …
    -/
    intro k hk
    /-
      E : Type u_1
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace Complex E
      V : Type u_2
      W : Type u_3
      inst✝⁷ : NormedAddCommGroup V
      inst✝⁶ : NormedSpace Real V
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace Real W
      L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
      f : V → E
      inst✝³ : MeasurableSpace V
      inst✝² : BorelSpace V
      inst✝¹ : FiniteDimensional Real V
      μ : MeasureTheory.Measure V
      inst✝ : μ.IsAddHaarMeasure
      K N : ENat
      hf : ContDiff Real (↑N) f
      h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
      k✝ n : Nat
      hk✝ : LE.le (↑k✝) K
      hn : LE.le (↑n) N
      w : W
      k : Nat
      hk : LE.le (↑k) K
      ⊢ MeasureTheory.Integrable (fun v => HMul.hMul (HPow.hPow (Norm.norm v) k) (No …
    -/
    simpa only [norm_iteratedFDeriv_zero] using h'f k 0 hk bot_le
    /-
      🎉 no goals
    -/
    /-
      E : Type u_1
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace Complex E
      V : Type u_2
      W : Type u_3
      inst✝⁷ : NormedAddCommGroup V
      inst✝⁶ : NormedSpace Real V
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace Real W
      L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
      f : V → E
      inst✝³ : MeasurableSpace V
      inst✝² : BorelSpace V
      inst✝¹ : FiniteDimensional Real V
      μ : MeasureTheory.Measure V
      inst✝ : μ.IsAddHaarMeasure
      K N : ENat
      hf : ContDiff Real (↑N) f
      h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
      k n : Nat
      hk : LE.le (↑k) K
      hn : LE.le (↑n) N
      w : W
      ⊢ ∀ (n : Nat), LE.le (↑n) N → MeasureTheory.Integrable (iteratedFDeriv Real n  …
    -/
  · intro m hm
    have I : Integrable (fun v ↦ ∑ p in Finset.range (k + 1) ×ˢ Finset.range (m + 1),
        ‖v‖ ^ p.1 * ‖iteratedFDeriv ℝ p.2 f v‖) μ := by
      apply integrable_finset_sum _ (fun p hp ↦ ?_)
      simp only [Finset.mem_product, Finset.mem_range_succ_iff] at hp
      exact h'f _ _ ((Nat.cast_le.2 hp.1).trans hk) ((Nat.cast_le.2 hp.2).trans hm)
    apply (I.const_mul ((2 * π) ^ k * (2 * k + 2) ^ m * ‖L‖ ^ k)).mono'
      ((hf.fourierPowSMulRight L k).continuous_iteratedFDeriv (mod_cast hm)).aestronglyMeasurable
    /-
      E : Type u_1
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace Complex E
      V : Type u_2
      W : Type u_3
      inst✝⁷ : NormedAddCommGroup V
      inst✝⁶ : NormedSpace Real V
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace Real W
      L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
      f : V → E
      inst✝³ : MeasurableSpace V
      inst✝² : BorelSpace V
      inst✝¹ : FiniteDimensional Real V
      μ : MeasureTheory.Measure V
      inst✝ : μ.IsAddHaarMeasure
      K N : ENat
      hf : ContDiff Real (↑N) f
      h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
      k n : Nat
      hk : LE.le (↑k) K
      hn : LE.le (↑n) N
      w : W
      m : Nat
      hm : LE.le (↑m) N
      I : MeasureTheory.Integrable (fun v => (SProd.sprod (Finset.range (HAdd.hAdd k …
      ⊢ Filter.Eventually (fun a => LE.le (Norm.norm (iteratedFDeriv Real m (fun v = …
    -/
    filter_upwards with v
    /-
      case h
      E : Type u_1
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace Complex E
      V : Type u_2
      W : Type u_3
      inst✝⁷ : NormedAddCommGroup V
      inst✝⁶ : NormedSpace Real V
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace Real W
      L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
      f : V → E
      inst✝³ : MeasurableSpace V
      inst✝² : BorelSpace V
      inst✝¹ : FiniteDimensional Real V
      μ : MeasureTheory.Measure V
      inst✝ : μ.IsAddHaarMeasure
      K N : ENat
      hf : ContDiff Real (↑N) f
      h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
      k n : Nat
      hk : LE.le (↑k) K
      hn : LE.le (↑n) N
      w : W
      m : Nat
      hm : LE.le (↑m) N
      I : MeasureTheory.Integrable (fun v => (SProd.sprod (Finset.range (HAdd.hAdd k …
      v : V
      ⊢ LE.le (Norm.norm (iteratedFDeriv Real m (fun v => VectorFourier.fourierPowSM …
    -/
    refine norm_iteratedFDeriv_fourierPowSMulRight _ hf (mod_cast hm) (fun i hi j hj ↦ ?_)
    /-
      case h
      E : Type u_1
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace Complex E
      V : Type u_2
      W : Type u_3
      inst✝⁷ : NormedAddCommGroup V
      inst✝⁶ : NormedSpace Real V
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace Real W
      L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
      f : V → E
      inst✝³ : MeasurableSpace V
      inst✝² : BorelSpace V
      inst✝¹ : FiniteDimensional Real V
      μ : MeasureTheory.Measure V
      inst✝ : μ.IsAddHaarMeasure
      K N : ENat
      hf : ContDiff Real (↑N) f
      h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
      k n : Nat
      hk : LE.le (↑k) K
      hn : LE.le (↑n) N
      w : W
      m : Nat
      hm : LE.le (↑m) N
      I : MeasureTheory.Integrable (fun v => (SProd.sprod (Finset.range (HAdd.hAdd k …
      v : V
      i : Nat
      hi : LE.le i m
      j : Nat
      hj : LE.le j k
      ⊢ LE.le (HMul.hMul (HPow.hPow (Norm.norm v) j) (Norm.norm (iteratedFDeriv Real …
    -/
    apply Finset.single_le_sum (f := fun p ↦ ‖v‖ ^ p.1 * ‖iteratedFDeriv ℝ p.2 f v‖) (a := (j, i))
      /-
        case h.hf
        E : Type u_1
        inst✝⁹ : NormedAddCommGroup E
        inst✝⁸ : NormedSpace Complex E
        V : Type u_2
        W : Type u_3
        inst✝⁷ : NormedAddCommGroup V
        inst✝⁶ : NormedSpace Real V
        inst✝⁵ : NormedAddCommGroup W
        inst✝⁴ : NormedSpace Real W
        L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
        f : V → E
        inst✝³ : MeasurableSpace V
        inst✝² : BorelSpace V
        inst✝¹ : FiniteDimensional Real V
        μ : MeasureTheory.Measure V
        inst✝ : μ.IsAddHaarMeasure
        K N : ENat
        hf : ContDiff Real (↑N) f
        h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
        k n : Nat
        hk : LE.le (↑k) K
        hn : LE.le (↑n) N
        w : W
        m : Nat
        hm : LE.le (↑m) N
        I : MeasureTheory.Integrable (fun v => (SProd.sprod (Finset.range (HAdd.hAdd k …
        v : V
        i : Nat
        hi : LE.le i m
        j : Nat
        hj : LE.le j k
        ⊢ ∀ (i : Prod Nat Nat), Membership.mem (SProd.sprod (Finset.range (HAdd.hAdd k …
      -/
    · intro i _hi
      /-
        case h.hf
        E : Type u_1
        inst✝⁹ : NormedAddCommGroup E
        inst✝⁸ : NormedSpace Complex E
        V : Type u_2
        W : Type u_3
        inst✝⁷ : NormedAddCommGroup V
        inst✝⁶ : NormedSpace Real V
        inst✝⁵ : NormedAddCommGroup W
        inst✝⁴ : NormedSpace Real W
        L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
        f : V → E
        inst✝³ : MeasurableSpace V
        inst✝² : BorelSpace V
        inst✝¹ : FiniteDimensional Real V
        μ : MeasureTheory.Measure V
        inst✝ : μ.IsAddHaarMeasure
        K N : ENat
        hf : ContDiff Real (↑N) f
        h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
        k n : Nat
        hk : LE.le (↑k) K
        hn : LE.le (↑n) N
        w : W
        m : Nat
        hm : LE.le (↑m) N
        I : MeasureTheory.Integrable (fun v => (SProd.sprod (Finset.range (HAdd.hAdd k …
        v : V
        i✝ : Nat
        hi : LE.le i✝ m
        j : Nat
        hj : LE.le j k
        i : Prod Nat Nat
        _hi : Membership.mem (SProd.sprod (Finset.range (HAdd.hAdd k 1)) (Finset.range …
        ⊢ LE.le 0 ((fun p => HMul.hMul (HPow.hPow (Norm.norm v) p.1) (Norm.norm (itera …
      -/
      positivity
      /-
        🎉 no goals
      -/
      /-
        case h.h
        E : Type u_1
        inst✝⁹ : NormedAddCommGroup E
        inst✝⁸ : NormedSpace Complex E
        V : Type u_2
        W : Type u_3
        inst✝⁷ : NormedAddCommGroup V
        inst✝⁶ : NormedSpace Real V
        inst✝⁵ : NormedAddCommGroup W
        inst✝⁴ : NormedSpace Real W
        L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
        f : V → E
        inst✝³ : MeasurableSpace V
        inst✝² : BorelSpace V
        inst✝¹ : FiniteDimensional Real V
        μ : MeasureTheory.Measure V
        inst✝ : μ.IsAddHaarMeasure
        K N : ENat
        hf : ContDiff Real (↑N) f
        h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
        k n : Nat
        hk : LE.le (↑k) K
        hn : LE.le (↑n) N
        w : W
        m : Nat
        hm : LE.le (↑m) N
        I : MeasureTheory.Integrable (fun v => (SProd.sprod (Finset.range (HAdd.hAdd k …
        v : V
        i : Nat
        hi : LE.le i m
        j : Nat
        hj : LE.le j k
        ⊢ Membership.mem (SProd.sprod (Finset.range (HAdd.hAdd k 1)) (Finset.range (HA …
      -/
    · simpa only [Finset.mem_product, Finset.mem_range_succ_iff] using ⟨hj, hi⟩
      /-
        🎉 no goals
      -/


/-- One can bound the `k`-th derivative of the Fourier integral of `f`, multiplied by `(L v w) ^ n`,
in terms of integrals of iterated derivatives of `f` (of order up to `n`) multiplied by `‖v‖ ^ i`
(for `i ≤ k`).
Auxiliary version in terms of the operator norm of `fourierPowSMulRight (-L.flip) ⬝`. For a version
in terms of `|L v w| ^ n * ⬝`, see `pow_mul_norm_iteratedFDeriv_fourierIntegral_le`.
-/
theorem norm_fourierPowSMulRight_iteratedFDeriv_fourierIntegral_le [FiniteDimensional ℝ V]
    {μ : Measure V} [Measure.IsAddHaarMeasure μ] {K N : ℕ∞} (hf : ContDiff ℝ N f)
    (h'f : ∀ (k n : ℕ), k ≤ K → n ≤ N → Integrable (fun v ↦ ‖v‖^k * ‖iteratedFDeriv ℝ n f v‖) μ)
    {k n : ℕ} (hk : k ≤ K) (hn : n ≤ N) {w : W} :
    ‖fourierPowSMulRight (-L.flip)
      (iteratedFDeriv ℝ k (fourierIntegral 𝐞 μ L.toLinearMap₂ f)) w n‖ ≤
    (2 * π) ^ k * (2 * k + 2) ^ n * ‖L‖ ^ k * ∑ p in Finset.range (k + 1) ×ˢ Finset.range (n + 1),
      ∫ v, ‖v‖ ^ p.1 * ‖iteratedFDeriv ℝ p.2 f v‖ ∂μ := by
  /-
    E : Type u_1
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace Complex E
    V : Type u_2
    W : Type u_3
    inst✝⁷ : NormedAddCommGroup V
    inst✝⁶ : NormedSpace Real V
    inst✝⁵ : NormedAddCommGroup W
    inst✝⁴ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    f : V → E
    inst✝³ : MeasurableSpace V
    inst✝² : BorelSpace V
    inst✝¹ : FiniteDimensional Real V
    μ : MeasureTheory.Measure V
    inst✝ : μ.IsAddHaarMeasure
    K N : ENat
    hf : ContDiff Real (↑N) f
    h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
    k n : Nat
    hk : LE.le (↑k) K
    hn : LE.le (↑n) N
    w : W
    ⊢ LE.le (Norm.norm (VectorFourier.fourierPowSMulRight (Neg.neg L.flip) (iterat …
  -/
  rw [fourierPowSMulRight_iteratedFDeriv_fourierIntegral L hf h'f hk hn]
  /-
    E : Type u_1
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace Complex E
    V : Type u_2
    W : Type u_3
    inst✝⁷ : NormedAddCommGroup V
    inst✝⁶ : NormedSpace Real V
    inst✝⁵ : NormedAddCommGroup W
    inst✝⁴ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    f : V → E
    inst✝³ : MeasurableSpace V
    inst✝² : BorelSpace V
    inst✝¹ : FiniteDimensional Real V
    μ : MeasureTheory.Measure V
    inst✝ : μ.IsAddHaarMeasure
    K N : ENat
    hf : ContDiff Real (↑N) f
    h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
    k n : Nat
    hk : LE.le (↑k) K
    hn : LE.le (↑n) N
    w : W
    ⊢ LE.le (Norm.norm (VectorFourier.fourierIntegral Real.fourierChar μ L.toLinea …
  -/
  apply (norm_fourierIntegral_le_integral_norm _ _ _ _ _).trans
  have I p (hp : p ∈ Finset.range (k + 1) ×ˢ Finset.range (n + 1)) :
      Integrable (fun v ↦ ‖v‖ ^ p.1 * ‖iteratedFDeriv ℝ p.2 f v‖) μ := by
    simp only [Finset.mem_product, Finset.mem_range_succ_iff] at hp
    exact h'f _ _ (le_trans (by simpa using hp.1) hk) (le_trans (by simpa using hp.2) hn)
  /-
    E : Type u_1
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace Complex E
    V : Type u_2
    W : Type u_3
    inst✝⁷ : NormedAddCommGroup V
    inst✝⁶ : NormedSpace Real V
    inst✝⁵ : NormedAddCommGroup W
    inst✝⁴ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    f : V → E
    inst✝³ : MeasurableSpace V
    inst✝² : BorelSpace V
    inst✝¹ : FiniteDimensional Real V
    μ : MeasureTheory.Measure V
    inst✝ : μ.IsAddHaarMeasure
    K N : ENat
    hf : ContDiff Real (↑N) f
    h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
    k n : Nat
    hk : LE.le (↑k) K
    hn : LE.le (↑n) N
    w : W
    I : ∀ (p : Prod Nat Nat), Membership.mem (SProd.sprod (Finset.range (HAdd.hAdd …
    ⊢ LE.le (MeasureTheory.integral μ fun v => Norm.norm (iteratedFDeriv Real n (f …
  -/
  rw [← integral_finset_sum _ I, ← integral_mul_left]
  /-
    E : Type u_1
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace Complex E
    V : Type u_2
    W : Type u_3
    inst✝⁷ : NormedAddCommGroup V
    inst✝⁶ : NormedSpace Real V
    inst✝⁵ : NormedAddCommGroup W
    inst✝⁴ : NormedSpace Real W
    L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    f : V → E
    inst✝³ : MeasurableSpace V
    inst✝² : BorelSpace V
    inst✝¹ : FiniteDimensional Real V
    μ : MeasureTheory.Measure V
    inst✝ : μ.IsAddHaarMeasure
    K N : ENat
    hf : ContDiff Real (↑N) f
    h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
    k n : Nat
    hk : LE.le (↑k) K
    hn : LE.le (↑n) N
    w : W
    I : ∀ (p : Prod Nat Nat), Membership.mem (SProd.sprod (Finset.range (HAdd.hAdd …
    ⊢ LE.le (MeasureTheory.integral μ fun v => Norm.norm (iteratedFDeriv Real n (f …
  -/
  apply integral_mono_of_nonneg
    /-
      case hf
      E : Type u_1
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace Complex E
      V : Type u_2
      W : Type u_3
      inst✝⁷ : NormedAddCommGroup V
      inst✝⁶ : NormedSpace Real V
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace Real W
      L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
      f : V → E
      inst✝³ : MeasurableSpace V
      inst✝² : BorelSpace V
      inst✝¹ : FiniteDimensional Real V
      μ : MeasureTheory.Measure V
      inst✝ : μ.IsAddHaarMeasure
      K N : ENat
      hf : ContDiff Real (↑N) f
      h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
      k n : Nat
      hk : LE.le (↑k) K
      hn : LE.le (↑n) N
      w : W
      I : ∀ (p : Prod Nat Nat), Membership.mem (SProd.sprod (Finset.range (HAdd.hAdd …
      ⊢ (MeasureTheory.ae μ).EventuallyLE 0 fun a => Norm.norm (iteratedFDeriv Real  …
    -/
  · filter_upwards with v using norm_nonneg _
    /-
      🎉 no goals
    -/
    /-
      case hgi
      E : Type u_1
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace Complex E
      V : Type u_2
      W : Type u_3
      inst✝⁷ : NormedAddCommGroup V
      inst✝⁶ : NormedSpace Real V
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace Real W
      L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
      f : V → E
      inst✝³ : MeasurableSpace V
      inst✝² : BorelSpace V
      inst✝¹ : FiniteDimensional Real V
      μ : MeasureTheory.Measure V
      inst✝ : μ.IsAddHaarMeasure
      K N : ENat
      hf : ContDiff Real (↑N) f
      h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
      k n : Nat
      hk : LE.le (↑k) K
      hn : LE.le (↑n) N
      w : W
      I : ∀ (p : Prod Nat Nat), Membership.mem (SProd.sprod (Finset.range (HAdd.hAdd …
      ⊢ MeasureTheory.Integrable (fun a => HMul.hMul (HMul.hMul (HMul.hMul (HPow.hPo …
    -/
  · exact (integrable_finset_sum _ I).const_mul _
    /-
      🎉 no goals
    -/
    /-
      case h
      E : Type u_1
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace Complex E
      V : Type u_2
      W : Type u_3
      inst✝⁷ : NormedAddCommGroup V
      inst✝⁶ : NormedSpace Real V
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace Real W
      L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
      f : V → E
      inst✝³ : MeasurableSpace V
      inst✝² : BorelSpace V
      inst✝¹ : FiniteDimensional Real V
      μ : MeasureTheory.Measure V
      inst✝ : μ.IsAddHaarMeasure
      K N : ENat
      hf : ContDiff Real (↑N) f
      h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
      k n : Nat
      hk : LE.le (↑k) K
      hn : LE.le (↑n) N
      w : W
      I : ∀ (p : Prod Nat Nat), Membership.mem (SProd.sprod (Finset.range (HAdd.hAdd …
      ⊢ (MeasureTheory.ae μ).EventuallyLE (fun a => Norm.norm (iteratedFDeriv Real n …
    -/
  · filter_upwards with v
    /-
      case h.h
      E : Type u_1
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace Complex E
      V : Type u_2
      W : Type u_3
      inst✝⁷ : NormedAddCommGroup V
      inst✝⁶ : NormedSpace Real V
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace Real W
      L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
      f : V → E
      inst✝³ : MeasurableSpace V
      inst✝² : BorelSpace V
      inst✝¹ : FiniteDimensional Real V
      μ : MeasureTheory.Measure V
      inst✝ : μ.IsAddHaarMeasure
      K N : ENat
      hf : ContDiff Real (↑N) f
      h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
      k n : Nat
      hk : LE.le (↑k) K
      hn : LE.le (↑n) N
      w : W
      I : ∀ (p : Prod Nat Nat), Membership.mem (SProd.sprod (Finset.range (HAdd.hAdd …
      v : V
      ⊢ LE.le (Norm.norm (iteratedFDeriv Real n (fun v => VectorFourier.fourierPowSM …
    -/
    apply norm_iteratedFDeriv_fourierPowSMulRight _ hf (mod_cast hn) _
    /-
      E : Type u_1
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace Complex E
      V : Type u_2
      W : Type u_3
      inst✝⁷ : NormedAddCommGroup V
      inst✝⁶ : NormedSpace Real V
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace Real W
      L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
      f : V → E
      inst✝³ : MeasurableSpace V
      inst✝² : BorelSpace V
      inst✝¹ : FiniteDimensional Real V
      μ : MeasureTheory.Measure V
      inst✝ : μ.IsAddHaarMeasure
      K N : ENat
      hf : ContDiff Real (↑N) f
      h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
      k n : Nat
      hk : LE.le (↑k) K
      hn : LE.le (↑n) N
      w : W
      I : ∀ (p : Prod Nat Nat), Membership.mem (SProd.sprod (Finset.range (HAdd.hAdd …
      v : V
      ⊢ ∀ (i : Nat), LE.le i n → ∀ (j : Nat), LE.le j k → LE.le (HMul.hMul (HPow.hPo …
    -/
    intro i hi j hj
    /-
      E : Type u_1
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace Complex E
      V : Type u_2
      W : Type u_3
      inst✝⁷ : NormedAddCommGroup V
      inst✝⁶ : NormedSpace Real V
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace Real W
      L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
      f : V → E
      inst✝³ : MeasurableSpace V
      inst✝² : BorelSpace V
      inst✝¹ : FiniteDimensional Real V
      μ : MeasureTheory.Measure V
      inst✝ : μ.IsAddHaarMeasure
      K N : ENat
      hf : ContDiff Real (↑N) f
      h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
      k n : Nat
      hk : LE.le (↑k) K
      hn : LE.le (↑n) N
      w : W
      I : ∀ (p : Prod Nat Nat), Membership.mem (SProd.sprod (Finset.range (HAdd.hAdd …
      v : V
      i : Nat
      hi : LE.le i n
      j : Nat
      hj : LE.le j k
      ⊢ LE.le (HMul.hMul (HPow.hPow (Norm.norm v) j) (Norm.norm (iteratedFDeriv Real …
    -/
    apply Finset.single_le_sum (f := fun p ↦ ‖v‖ ^ p.1 * ‖iteratedFDeriv ℝ p.2 f v‖) (a := (j, i))
      /-
        case hf
        E : Type u_1
        inst✝⁹ : NormedAddCommGroup E
        inst✝⁸ : NormedSpace Complex E
        V : Type u_2
        W : Type u_3
        inst✝⁷ : NormedAddCommGroup V
        inst✝⁶ : NormedSpace Real V
        inst✝⁵ : NormedAddCommGroup W
        inst✝⁴ : NormedSpace Real W
        L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
        f : V → E
        inst✝³ : MeasurableSpace V
        inst✝² : BorelSpace V
        inst✝¹ : FiniteDimensional Real V
        μ : MeasureTheory.Measure V
        inst✝ : μ.IsAddHaarMeasure
        K N : ENat
        hf : ContDiff Real (↑N) f
        h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
        k n : Nat
        hk : LE.le (↑k) K
        hn : LE.le (↑n) N
        w : W
        I : ∀ (p : Prod Nat Nat), Membership.mem (SProd.sprod (Finset.range (HAdd.hAdd …
        v : V
        i : Nat
        hi : LE.le i n
        j : Nat
        hj : LE.le j k
        ⊢ ∀ (i : Prod Nat Nat), Membership.mem (SProd.sprod (Finset.range (HAdd.hAdd k …
      -/
    · intro i _hi
      /-
        case hf
        E : Type u_1
        inst✝⁹ : NormedAddCommGroup E
        inst✝⁸ : NormedSpace Complex E
        V : Type u_2
        W : Type u_3
        inst✝⁷ : NormedAddCommGroup V
        inst✝⁶ : NormedSpace Real V
        inst✝⁵ : NormedAddCommGroup W
        inst✝⁴ : NormedSpace Real W
        L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
        f : V → E
        inst✝³ : MeasurableSpace V
        inst✝² : BorelSpace V
        inst✝¹ : FiniteDimensional Real V
        μ : MeasureTheory.Measure V
        inst✝ : μ.IsAddHaarMeasure
        K N : ENat
        hf : ContDiff Real (↑N) f
        h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
        k n : Nat
        hk : LE.le (↑k) K
        hn : LE.le (↑n) N
        w : W
        I : ∀ (p : Prod Nat Nat), Membership.mem (SProd.sprod (Finset.range (HAdd.hAdd …
        v : V
        i✝ : Nat
        hi : LE.le i✝ n
        j : Nat
        hj : LE.le j k
        i : Prod Nat Nat
        _hi : Membership.mem (SProd.sprod (Finset.range (HAdd.hAdd k 1)) (Finset.range …
        ⊢ LE.le 0 ((fun p => HMul.hMul (HPow.hPow (Norm.norm v) p.1) (Norm.norm (itera …
      -/
      positivity
      /-
        🎉 no goals
      -/
      /-
        case h
        E : Type u_1
        inst✝⁹ : NormedAddCommGroup E
        inst✝⁸ : NormedSpace Complex E
        V : Type u_2
        W : Type u_3
        inst✝⁷ : NormedAddCommGroup V
        inst✝⁶ : NormedSpace Real V
        inst✝⁵ : NormedAddCommGroup W
        inst✝⁴ : NormedSpace Real W
        L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
        f : V → E
        inst✝³ : MeasurableSpace V
        inst✝² : BorelSpace V
        inst✝¹ : FiniteDimensional Real V
        μ : MeasureTheory.Measure V
        inst✝ : μ.IsAddHaarMeasure
        K N : ENat
        hf : ContDiff Real (↑N) f
        h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
        k n : Nat
        hk : LE.le (↑k) K
        hn : LE.le (↑n) N
        w : W
        I : ∀ (p : Prod Nat Nat), Membership.mem (SProd.sprod (Finset.range (HAdd.hAdd …
        v : V
        i : Nat
        hi : LE.le i n
        j : Nat
        hj : LE.le j k
        ⊢ Membership.mem (SProd.sprod (Finset.range (HAdd.hAdd k 1)) (Finset.range (HA …
      -/
    · simp only [Finset.mem_product, Finset.mem_range_succ_iff]
      /-
        case h
        E : Type u_1
        inst✝⁹ : NormedAddCommGroup E
        inst✝⁸ : NormedSpace Complex E
        V : Type u_2
        W : Type u_3
        inst✝⁷ : NormedAddCommGroup V
        inst✝⁶ : NormedSpace Real V
        inst✝⁵ : NormedAddCommGroup W
        inst✝⁴ : NormedSpace Real W
        L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
        f : V → E
        inst✝³ : MeasurableSpace V
        inst✝² : BorelSpace V
        inst✝¹ : FiniteDimensional Real V
        μ : MeasureTheory.Measure V
        inst✝ : μ.IsAddHaarMeasure
        K N : ENat
        hf : ContDiff Real (↑N) f
        h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
        k n : Nat
        hk : LE.le (↑k) K
        hn : LE.le (↑n) N
        w : W
        I : ∀ (p : Prod Nat Nat), Membership.mem (SProd.sprod (Finset.range (HAdd.hAdd …
        v : V
        i : Nat
        hi : LE.le i n
        j : Nat
        hj : LE.le j k
        ⊢ And (LE.le j k) (LE.le i n)
      -/
      exact ⟨hj, hi⟩
      /-
        🎉 no goals
      -/


/-- One can bound the `k`-th derivative of the Fourier integral of `f`, multiplied by `(L v w) ^ n`,
in terms of integrals of iterated derivatives of `f` (of order up to `n`) multiplied by `‖v‖ ^ i`
(for `i ≤ k`). -/
lemma pow_mul_norm_iteratedFDeriv_fourierIntegral_le [FiniteDimensional ℝ V]
    {μ : Measure V} [Measure.IsAddHaarMeasure μ] {K N : ℕ∞} (hf : ContDiff ℝ N f)
    (h'f : ∀ (k n : ℕ), k ≤ K → n ≤ N → Integrable (fun v ↦ ‖v‖^k * ‖iteratedFDeriv ℝ n f v‖) μ)
    {k n : ℕ} (hk : k ≤ K) (hn : n ≤ N) (v : V) (w : W) :
    |L v w| ^ n * ‖(iteratedFDeriv ℝ k (fourierIntegral 𝐞 μ L.toLinearMap₂ f)) w‖ ≤
      ‖v‖ ^ n * (2 * π * ‖L‖) ^ k * (2 * k + 2) ^ n *
        ∑ p in Finset.range (k + 1) ×ˢ Finset.range (n + 1),
          ∫ v, ‖v‖ ^ p.1 * ‖iteratedFDeriv ℝ p.2 f v‖ ∂μ := calc
  |L v w| ^ n * ‖(iteratedFDeriv ℝ k (fourierIntegral 𝐞 μ L.toLinearMap₂ f)) w‖
  _ ≤ (2 * π) ^ n
      * (|L v w| ^ n * ‖iteratedFDeriv ℝ k (fourierIntegral 𝐞 μ L.toLinearMap₂ f) w‖) := by
    /-
      E : Type u_1
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace Complex E
      V : Type u_2
      W : Type u_3
      inst✝⁷ : NormedAddCommGroup V
      inst✝⁶ : NormedSpace Real V
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace Real W
      L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
      f : V → E
      inst✝³ : MeasurableSpace V
      inst✝² : BorelSpace V
      inst✝¹ : FiniteDimensional Real V
      μ : MeasureTheory.Measure V
      inst✝ : μ.IsAddHaarMeasure
      K N : ENat
      hf : ContDiff Real (↑N) f
      h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
      k n : Nat
      hk : LE.le (↑k) K
      hn : LE.le (↑n) N
      v : V
      w : W
      ⊢ LE.le (HMul.hMul (HPow.hPow (abs ((L v) w)) n) (Norm.norm (iteratedFDeriv Re …
    -/
    apply le_mul_of_one_le_left (by positivity)
    /-
      E : Type u_1
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace Complex E
      V : Type u_2
      W : Type u_3
      inst✝⁷ : NormedAddCommGroup V
      inst✝⁶ : NormedSpace Real V
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace Real W
      L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
      f : V → E
      inst✝³ : MeasurableSpace V
      inst✝² : BorelSpace V
      inst✝¹ : FiniteDimensional Real V
      μ : MeasureTheory.Measure V
      inst✝ : μ.IsAddHaarMeasure
      K N : ENat
      hf : ContDiff Real (↑N) f
      h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
      k n : Nat
      hk : LE.le (↑k) K
      hn : LE.le (↑n) N
      v : V
      w : W
      ⊢ LE.le 1 (HPow.hPow (HMul.hMul 2 Real.pi) n)
    -/
    apply one_le_pow₀
    /-
      case ha
      E : Type u_1
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace Complex E
      V : Type u_2
      W : Type u_3
      inst✝⁷ : NormedAddCommGroup V
      inst✝⁶ : NormedSpace Real V
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace Real W
      L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
      f : V → E
      inst✝³ : MeasurableSpace V
      inst✝² : BorelSpace V
      inst✝¹ : FiniteDimensional Real V
      μ : MeasureTheory.Measure V
      inst✝ : μ.IsAddHaarMeasure
      K N : ENat
      hf : ContDiff Real (↑N) f
      h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
      k n : Nat
      hk : LE.le (↑k) K
      hn : LE.le (↑n) N
      v : V
      w : W
      ⊢ LE.le 1 (HMul.hMul 2 Real.pi)
    -/
    linarith [one_le_pi_div_two]
    /-
      🎉 no goals
    -/
  _ = ‖fourierPowSMulRight (-L.flip)
        (iteratedFDeriv ℝ k (fourierIntegral 𝐞 μ L.toLinearMap₂ f)) w n (fun _ ↦ v)‖ := by
    /-
      E : Type u_1
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace Complex E
      V : Type u_2
      W : Type u_3
      inst✝⁷ : NormedAddCommGroup V
      inst✝⁶ : NormedSpace Real V
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace Real W
      L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
      f : V → E
      inst✝³ : MeasurableSpace V
      inst✝² : BorelSpace V
      inst✝¹ : FiniteDimensional Real V
      μ : MeasureTheory.Measure V
      inst✝ : μ.IsAddHaarMeasure
      K N : ENat
      hf : ContDiff Real (↑N) f
      h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
      k n : Nat
      hk : LE.le (↑k) K
      hn : LE.le (↑n) N
      v : V
      w : W
      ⊢ Eq (HMul.hMul (HPow.hPow (HMul.hMul 2 Real.pi) n) (HMul.hMul (HPow.hPow (abs …
    -/
    simp [norm_smul, _root_.abs_of_nonneg pi_nonneg]
    /-
      🎉 no goals
    -/
  _ ≤ ‖fourierPowSMulRight (-L.flip)
        (iteratedFDeriv ℝ k (fourierIntegral 𝐞 μ L.toLinearMap₂ f)) w n‖ * ∏ _ : Fin n, ‖v‖ :=
    le_opNorm _ _
  _ ≤ ((2 * π) ^ k * (2 * k + 2) ^ n * ‖L‖ ^ k *
      ∑ p in Finset.range (k + 1) ×ˢ Finset.range (n + 1),
        ∫ v, ‖v‖ ^ p.1 * ‖iteratedFDeriv ℝ p.2 f v‖ ∂μ) * ‖v‖ ^ n := by
    /-
      E : Type u_1
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace Complex E
      V : Type u_2
      W : Type u_3
      inst✝⁷ : NormedAddCommGroup V
      inst✝⁶ : NormedSpace Real V
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace Real W
      L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
      f : V → E
      inst✝³ : MeasurableSpace V
      inst✝² : BorelSpace V
      inst✝¹ : FiniteDimensional Real V
      μ : MeasureTheory.Measure V
      inst✝ : μ.IsAddHaarMeasure
      K N : ENat
      hf : ContDiff Real (↑N) f
      h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
      k n : Nat
      hk : LE.le (↑k) K
      hn : LE.le (↑n) N
      v : V
      w : W
      ⊢ LE.le (HMul.hMul (Norm.norm (VectorFourier.fourierPowSMulRight (Neg.neg L.fl …
    -/
    gcongr
      /-
        case h₁
        E : Type u_1
        inst✝⁹ : NormedAddCommGroup E
        inst✝⁸ : NormedSpace Complex E
        V : Type u_2
        W : Type u_3
        inst✝⁷ : NormedAddCommGroup V
        inst✝⁶ : NormedSpace Real V
        inst✝⁵ : NormedAddCommGroup W
        inst✝⁴ : NormedSpace Real W
        L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
        f : V → E
        inst✝³ : MeasurableSpace V
        inst✝² : BorelSpace V
        inst✝¹ : FiniteDimensional Real V
        μ : MeasureTheory.Measure V
        inst✝ : μ.IsAddHaarMeasure
        K N : ENat
        hf : ContDiff Real (↑N) f
        h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
        k n : Nat
        hk : LE.le (↑k) K
        hn : LE.le (↑n) N
        v : V
        w : W
        ⊢ LE.le (Norm.norm (VectorFourier.fourierPowSMulRight (Neg.neg L.flip) (iterat …
      -/
    · apply norm_fourierPowSMulRight_iteratedFDeriv_fourierIntegral_le _ hf h'f hk hn
      /-
        🎉 no goals
      -/
      /-
        case h₂
        E : Type u_1
        inst✝⁹ : NormedAddCommGroup E
        inst✝⁸ : NormedSpace Complex E
        V : Type u_2
        W : Type u_3
        inst✝⁷ : NormedAddCommGroup V
        inst✝⁶ : NormedSpace Real V
        inst✝⁵ : NormedAddCommGroup W
        inst✝⁴ : NormedSpace Real W
        L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
        f : V → E
        inst✝³ : MeasurableSpace V
        inst✝² : BorelSpace V
        inst✝¹ : FiniteDimensional Real V
        μ : MeasureTheory.Measure V
        inst✝ : μ.IsAddHaarMeasure
        K N : ENat
        hf : ContDiff Real (↑N) f
        h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
        k n : Nat
        hk : LE.le (↑k) K
        hn : LE.le (↑n) N
        v : V
        w : W
        ⊢ LE.le (Finset.univ.prod fun x => Norm.norm v) (HPow.hPow (Norm.norm v) n)
      -/
    · simp
      /-
        🎉 no goals
      -/
  _ = ‖v‖ ^ n * (2 * π * ‖L‖) ^ k * (2 * k + 2) ^ n *
        ∑ p in Finset.range (k + 1) ×ˢ Finset.range (n + 1),
          ∫ v, ‖v‖ ^ p.1 * ‖iteratedFDeriv ℝ p.2 f v‖ ∂μ := by
    /-
      E : Type u_1
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace Complex E
      V : Type u_2
      W : Type u_3
      inst✝⁷ : NormedAddCommGroup V
      inst✝⁶ : NormedSpace Real V
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace Real W
      L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
      f : V → E
      inst✝³ : MeasurableSpace V
      inst✝² : BorelSpace V
      inst✝¹ : FiniteDimensional Real V
      μ : MeasureTheory.Measure V
      inst✝ : μ.IsAddHaarMeasure
      K N : ENat
      hf : ContDiff Real (↑N) f
      h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
      k n : Nat
      hk : LE.le (↑k) K
      hn : LE.le (↑n) N
      v : V
      w : W
      ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul (HPow.hPow (HMul.hMul 2 Real. …
    -/
    simp [mul_pow]
    /-
      E : Type u_1
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace Complex E
      V : Type u_2
      W : Type u_3
      inst✝⁷ : NormedAddCommGroup V
      inst✝⁶ : NormedSpace Real V
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace Real W
      L : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
      f : V → E
      inst✝³ : MeasurableSpace V
      inst✝² : BorelSpace V
      inst✝¹ : FiniteDimensional Real V
      μ : MeasureTheory.Measure V
      inst✝ : μ.IsAddHaarMeasure
      K N : ENat
      hf : ContDiff Real (↑N) f
      h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
      k n : Nat
      hk : LE.le (↑k) K
      hn : LE.le (↑n) N
      v : V
      w : W
      ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul (HPow.hPow 2 k) (H …
    -/
    ring
    /-
      🎉 no goals
    -/


/-- The Fréchet derivative of the Fourier transform of `f` is the Fourier transform of
    `fun v ↦ -2 * π * I ⟪v, ⬝⟫ f v`. -/
theorem hasFDerivAt_fourierIntegral
              /-
                E : Type u_1
                inst✝⁶ : NormedAddCommGroup E
                inst✝⁵ : NormedSpace Complex E
                V : Type u_2
                inst✝⁴ : NormedAddCommGroup V
                inst✝³ : InnerProductSpace Real V
                inst✝² : FiniteDimensional Real V
                inst✝¹ : MeasurableSpace V
                inst✝ : BorelSpace V
                f : V → E
                ⊢ MeasureTheory.Measure V
              -/
              /-
                🎉 no goals
              -/
    (hf_int : Integrable f) (hvf_int : Integrable (fun v ↦ ‖v‖ * ‖f v‖)) (x : V) :
                                       /-
                                         🎉 no goals
                                       -/
    HasFDerivAt (𝓕 f) (𝓕 (fourierSMulRight (innerSL ℝ) f) x) x :=
  VectorFourier.hasFDerivAt_fourierIntegral (innerSL ℝ) hf_int hvf_int x


/-- The Fréchet derivative of the Fourier transform of `f` is the Fourier transform of
    `fun v ↦ -2 * π * I ⟪v, ⬝⟫ f v`. -/
theorem fderiv_fourierIntegral
              /-
                E : Type u_1
                inst✝⁶ : NormedAddCommGroup E
                inst✝⁵ : NormedSpace Complex E
                V : Type u_2
                inst✝⁴ : NormedAddCommGroup V
                inst✝³ : InnerProductSpace Real V
                inst✝² : FiniteDimensional Real V
                inst✝¹ : MeasurableSpace V
                inst✝ : BorelSpace V
                f : V → E
                ⊢ MeasureTheory.Measure V
              -/
              /-
                🎉 no goals
              -/
    (hf_int : Integrable f) (hvf_int : Integrable (fun v ↦ ‖v‖ * ‖f v‖)) :
                                       /-
                                         🎉 no goals
                                       -/
    fderiv ℝ (𝓕 f) = 𝓕 (fourierSMulRight (innerSL ℝ) f) :=
  VectorFourier.fderiv_fourierIntegral (innerSL ℝ) hf_int hvf_int


theorem differentiable_fourierIntegral
              /-
                E : Type u_1
                inst✝⁶ : NormedAddCommGroup E
                inst✝⁵ : NormedSpace Complex E
                V : Type u_2
                inst✝⁴ : NormedAddCommGroup V
                inst✝³ : InnerProductSpace Real V
                inst✝² : FiniteDimensional Real V
                inst✝¹ : MeasurableSpace V
                inst✝ : BorelSpace V
                f : V → E
                ⊢ MeasureTheory.Measure V
              -/
              /-
                🎉 no goals
              -/
    (hf_int : Integrable f) (hvf_int : Integrable (fun v ↦ ‖v‖ * ‖f v‖)) :
                                       /-
                                         🎉 no goals
                                       -/
    Differentiable ℝ (𝓕 f) :=
  VectorFourier.differentiable_fourierIntegral (innerSL ℝ) hf_int hvf_int


/-- The Fourier integral of the Fréchet derivative of a function is obtained by multiplying the
Fourier integral of the original function by `2πI ⟪v, w⟫`. -/
theorem fourierIntegral_fderiv
          /-
            E : Type u_1
            inst✝⁶ : NormedAddCommGroup E
            inst✝⁵ : NormedSpace Complex E
            V : Type u_2
            inst✝⁴ : NormedAddCommGroup V
            inst✝³ : InnerProductSpace Real V
            inst✝² : FiniteDimensional Real V
            inst✝¹ : MeasurableSpace V
            inst✝ : BorelSpace V
            f : V → E
            ⊢ MeasureTheory.Measure V
          -/
          /-
            🎉 no goals
          -/
    (hf : Integrable f) (h'f : Differentiable ℝ f) (hf' : Integrable (fderiv ℝ f)) :
                                                          /-
                                                            🎉 no goals
                                                          -/
    𝓕 (fderiv ℝ f) = fourierSMulRight (-innerSL ℝ) (𝓕 f) := by
  /-
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    V : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : FiniteDimensional Real V
    inst✝¹ : MeasurableSpace V
    inst✝ : BorelSpace V
    f : V → E
    hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
    h'f : Differentiable Real f
    hf' : MeasureTheory.Integrable (fderiv Real f) MeasureTheory.MeasureSpace.volume
    ⊢ Eq (Real.fourierIntegral (fderiv Real f)) (VectorFourier.fourierSMulRight (N …
  -/
  rw [← innerSL_real_flip V]
  /-
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    V : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : FiniteDimensional Real V
    inst✝¹ : MeasurableSpace V
    inst✝ : BorelSpace V
    f : V → E
    hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
    h'f : Differentiable Real f
    hf' : MeasureTheory.Integrable (fderiv Real f) MeasureTheory.MeasureSpace.volume
    ⊢ Eq (Real.fourierIntegral (fderiv Real f)) (VectorFourier.fourierSMulRight (N …
  -/
  exact VectorFourier.fourierIntegral_fderiv (innerSL ℝ) hf h'f hf'
  /-
    🎉 no goals
  -/


/-- If `‖v‖^n * ‖f v‖` is integrable, then the Fourier transform of `f` is `C^n`. -/
theorem contDiff_fourierIntegral {N : ℕ∞}
                             /-
                               E : Type u_1
                               inst✝⁶ : NormedAddCommGroup E
                               inst✝⁵ : NormedSpace Complex E
                               V : Type u_2
                               inst✝⁴ : NormedAddCommGroup V
                               inst✝³ : InnerProductSpace Real V
                               inst✝² : FiniteDimensional Real V
                               inst✝¹ : MeasurableSpace V
                               inst✝ : BorelSpace V
                               f : V → E
                               N : ENat
                               n : Nat
                               ⊢ MeasureTheory.Measure V
                             -/
    (hf : ∀ (n : ℕ), n ≤ N → Integrable (fun v ↦ ‖v‖^n * ‖f v‖)) :
                             /-
                               🎉 no goals
                             -/
    ContDiff ℝ N (𝓕 f) :=
  VectorFourier.contDiff_fourierIntegral (innerSL ℝ) hf


/-- If `‖v‖^n * ‖f v‖` is integrable, then the `n`-th derivative of the Fourier transform of `f` is
  the Fourier transform of `fun v ↦ (-2 * π * I) ^ n ⟪v, ⬝⟫^n f v`. -/
theorem iteratedFDeriv_fourierIntegral {N : ℕ∞}
                             /-
                               E : Type u_1
                               inst✝⁶ : NormedAddCommGroup E
                               inst✝⁵ : NormedSpace Complex E
                               V : Type u_2
                               inst✝⁴ : NormedAddCommGroup V
                               inst✝³ : InnerProductSpace Real V
                               inst✝² : FiniteDimensional Real V
                               inst✝¹ : MeasurableSpace V
                               inst✝ : BorelSpace V
                               f : V → E
                               N : ENat
                               n : Nat
                               ⊢ MeasureTheory.Measure V
                             -/
    (hf : ∀ (n : ℕ), n ≤ N → Integrable (fun v ↦ ‖v‖^n * ‖f v‖))
                             /-
                               🎉 no goals
                             -/
           /-
             E : Type u_1
             inst✝⁶ : NormedAddCommGroup E
             inst✝⁵ : NormedSpace Complex E
             V : Type u_2
             inst✝⁴ : NormedAddCommGroup V
             inst✝³ : InnerProductSpace Real V
             inst✝² : FiniteDimensional Real V
             inst✝¹ : MeasurableSpace V
             inst✝ : BorelSpace V
             f : V → E
             N : ENat
             hf : ∀ (n : Nat), LE.le (↑n) N → MeasureTheory.Integrable (fun v => HMul.hMul  …
             ⊢ MeasureTheory.Measure V
           -/
    (h'f : AEStronglyMeasurable f) {n : ℕ} (hn : n ≤ N) :
           /-
             🎉 no goals
           -/
    iteratedFDeriv ℝ n (𝓕 f) = 𝓕 (fun v ↦ fourierPowSMulRight (innerSL ℝ) f v n) :=
  VectorFourier.iteratedFDeriv_fourierIntegral (innerSL ℝ) hf h'f hn


/-- The Fourier integral of the `n`-th derivative of a function is obtained by multiplying the
Fourier integral of the original function by `(2πI L w ⬝ )^n`. -/
theorem fourierIntegral_iteratedFDeriv {N : ℕ∞} (hf : ContDiff ℝ N f)
                              /-
                                E : Type u_1
                                inst✝⁶ : NormedAddCommGroup E
                                inst✝⁵ : NormedSpace Complex E
                                V : Type u_2
                                inst✝⁴ : NormedAddCommGroup V
                                inst✝³ : InnerProductSpace Real V
                                inst✝² : FiniteDimensional Real V
                                inst✝¹ : MeasurableSpace V
                                inst✝ : BorelSpace V
                                f : V → E
                                N : ENat
                                hf : ContDiff Real (↑N) f
                                n : Nat
                                ⊢ MeasureTheory.Measure V
                              -/
    (h'f : ∀ (n : ℕ), n ≤ N → Integrable (iteratedFDeriv ℝ n f)) {n : ℕ} (hn : n ≤ N) :
                              /-
                                🎉 no goals
                              -/
    𝓕 (iteratedFDeriv ℝ n f)
      = (fun w ↦ fourierPowSMulRight (-innerSL ℝ) (𝓕 f) w n) := by
  /-
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    V : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : FiniteDimensional Real V
    inst✝¹ : MeasurableSpace V
    inst✝ : BorelSpace V
    f : V → E
    N : ENat
    hf : ContDiff Real (↑N) f
    h'f : ∀ (n : Nat), LE.le (↑n) N → MeasureTheory.Integrable (iteratedFDeriv Rea …
    n : Nat
    hn : LE.le (↑n) N
    ⊢ Eq (Real.fourierIntegral (iteratedFDeriv Real n f)) fun w => VectorFourier.f …
  -/
  rw [← innerSL_real_flip V]
  /-
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    V : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : FiniteDimensional Real V
    inst✝¹ : MeasurableSpace V
    inst✝ : BorelSpace V
    f : V → E
    N : ENat
    hf : ContDiff Real (↑N) f
    h'f : ∀ (n : Nat), LE.le (↑n) N → MeasureTheory.Integrable (iteratedFDeriv Rea …
    n : Nat
    hn : LE.le (↑n) N
    ⊢ Eq (Real.fourierIntegral (iteratedFDeriv Real n f)) fun w => VectorFourier.f …
  -/
  exact VectorFourier.fourierIntegral_iteratedFDeriv (innerSL ℝ) hf h'f hn
  /-
    🎉 no goals
  -/


/-- One can bound `‖w‖^n * ‖D^k (𝓕 f) w‖` in terms of integrals of the derivatives of `f` (or order
at most `n`) multiplied by powers of `v` (of order at most `k`). -/
lemma pow_mul_norm_iteratedFDeriv_fourierIntegral_le
    {K N : ℕ∞} (hf : ContDiff ℝ N f)
                                        /-
                                          E : Type u_1
                                          inst✝⁶ : NormedAddCommGroup E
                                          inst✝⁵ : NormedSpace Complex E
                                          V : Type u_2
                                          inst✝⁴ : NormedAddCommGroup V
                                          inst✝³ : InnerProductSpace Real V
                                          inst✝² : FiniteDimensional Real V
                                          inst✝¹ : MeasurableSpace V
                                          inst✝ : BorelSpace V
                                          f : V → E
                                          K N : ENat
                                          hf : ContDiff Real (↑N) f
                                          k n : Nat
                                          ⊢ MeasureTheory.Measure V
                                        -/
    (h'f : ∀ (k n : ℕ), k ≤ K → n ≤ N → Integrable (fun v ↦ ‖v‖^k * ‖iteratedFDeriv ℝ n f v‖))
                                        /-
                                          🎉 no goals
                                        -/
    {k n : ℕ} (hk : k ≤ K) (hn : n ≤ N) (w : V) :
    ‖w‖ ^ n * ‖iteratedFDeriv ℝ k (𝓕 f) w‖ ≤ (2 * π) ^ k * (2 * k + 2) ^ n *
      ∑ p in Finset.range (k + 1) ×ˢ Finset.range (n + 1),
        ∫ v, ‖v‖ ^ p.1 * ‖iteratedFDeriv ℝ p.2 f v‖ := by
  have Z : ‖w‖ ^ n * (‖w‖ ^ n * ‖iteratedFDeriv ℝ k (𝓕 f) w‖) ≤
      ‖w‖ ^ n * ((2 * (π * ‖innerSL (E := V) ℝ‖)) ^ k * ((2 * k + 2) ^ n *
          ∑ p ∈ Finset.range (k + 1) ×ˢ Finset.range (n + 1),
            ∫ (v : V), ‖v‖ ^ p.1 * ‖iteratedFDeriv ℝ p.2 f v‖ ∂volume)) := by
    have := VectorFourier.pow_mul_norm_iteratedFDeriv_fourierIntegral_le (innerSL ℝ) hf h'f hk hn
      w w
    simp only [innerSL_apply _ w w, real_inner_self_eq_norm_sq w, _root_.abs_pow, abs_norm,
      mul_assoc] at this
    rwa [pow_two, mul_pow, mul_assoc] at this
  /-
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    V : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : FiniteDimensional Real V
    inst✝¹ : MeasurableSpace V
    inst✝ : BorelSpace V
    f : V → E
    K N : ENat
    hf : ContDiff Real (↑N) f
    h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
    k n : Nat
    hk : LE.le (↑k) K
    hn : LE.le (↑n) N
    w : V
    Z : LE.le (HMul.hMul (HPow.hPow (Norm.norm w) n) (HMul.hMul (HPow.hPow (Norm.n …
    ⊢ LE.le (HMul.hMul (HPow.hPow (Norm.norm w) n) (Norm.norm (iteratedFDeriv Real …
  -/
  rcases eq_or_ne n 0 with rfl | hn
  · simp only [pow_zero, one_mul, mul_one, zero_add, Finset.range_one, Finset.product_singleton,
      Finset.sum_map, Function.Embedding.coeFn_mk, norm_iteratedFDeriv_zero] at Z ⊢
    /-
      case inl
      E : Type u_1
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace Complex E
      V : Type u_2
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : InnerProductSpace Real V
      inst✝² : FiniteDimensional Real V
      inst✝¹ : MeasurableSpace V
      inst✝ : BorelSpace V
      f : V → E
      K N : ENat
      hf : ContDiff Real (↑N) f
      h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
      k : Nat
      hk : LE.le (↑k) K
      w : V
      hn : LE.le (↑0) N
      Z : LE.le (Norm.norm (iteratedFDeriv Real k (Real.fourierIntegral f) w)) (HMul …
      ⊢ LE.le (Norm.norm (iteratedFDeriv Real k (Real.fourierIntegral f) w)) (HMul.h …
    -/
    apply Z.trans
    /-
      case inl
      E : Type u_1
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace Complex E
      V : Type u_2
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : InnerProductSpace Real V
      inst✝² : FiniteDimensional Real V
      inst✝¹ : MeasurableSpace V
      inst✝ : BorelSpace V
      f : V → E
      K N : ENat
      hf : ContDiff Real (↑N) f
      h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
      k : Nat
      hk : LE.le (↑k) K
      w : V
      hn : LE.le (↑0) N
      Z : LE.le (Norm.norm (iteratedFDeriv Real k (Real.fourierIntegral f) w)) (HMul …
      ⊢ LE.le (HMul.hMul (HPow.hPow (HMul.hMul 2 (HMul.hMul Real.pi (Norm.norm (inne …
    -/
    conv_rhs => rw [← mul_one π]
    /-
      case inl
      E : Type u_1
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace Complex E
      V : Type u_2
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : InnerProductSpace Real V
      inst✝² : FiniteDimensional Real V
      inst✝¹ : MeasurableSpace V
      inst✝ : BorelSpace V
      f : V → E
      K N : ENat
      hf : ContDiff Real (↑N) f
      h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
      k : Nat
      hk : LE.le (↑k) K
      w : V
      hn : LE.le (↑0) N
      Z : LE.le (Norm.norm (iteratedFDeriv Real k (Real.fourierIntegral f) w)) (HMul …
      ⊢ LE.le (HMul.hMul (HPow.hPow (HMul.hMul 2 (HMul.hMul Real.pi (Norm.norm (inne …
    -/
    gcongr
    /-
      case inl.h.hab.h.h
      E : Type u_1
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace Complex E
      V : Type u_2
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : InnerProductSpace Real V
      inst✝² : FiniteDimensional Real V
      inst✝¹ : MeasurableSpace V
      inst✝ : BorelSpace V
      f : V → E
      K N : ENat
      hf : ContDiff Real (↑N) f
      h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
      k : Nat
      hk : LE.le (↑k) K
      w : V
      hn : LE.le (↑0) N
      Z : LE.le (Norm.norm (iteratedFDeriv Real k (Real.fourierIntegral f) w)) (HMul …
      ⊢ LE.le (Norm.norm (innerSL Real)) 1
    -/
    exact norm_innerSL_le _
    /-
      🎉 no goals
    -/
  /-
    case inr
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    V : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : FiniteDimensional Real V
    inst✝¹ : MeasurableSpace V
    inst✝ : BorelSpace V
    f : V → E
    K N : ENat
    hf : ContDiff Real (↑N) f
    h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
    k n : Nat
    hk : LE.le (↑k) K
    hn✝ : LE.le (↑n) N
    w : V
    Z : LE.le (HMul.hMul (HPow.hPow (Norm.norm w) n) (HMul.hMul (HPow.hPow (Norm.n …
    hn : Ne n 0
    ⊢ LE.le (HMul.hMul (HPow.hPow (Norm.norm w) n) (Norm.norm (iteratedFDeriv Real …
  -/
  rcases eq_or_ne w 0 with rfl | hw
    /-
      case inr.inl
      E : Type u_1
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace Complex E
      V : Type u_2
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : InnerProductSpace Real V
      inst✝² : FiniteDimensional Real V
      inst✝¹ : MeasurableSpace V
      inst✝ : BorelSpace V
      f : V → E
      K N : ENat
      hf : ContDiff Real (↑N) f
      h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
      k n : Nat
      hk : LE.le (↑k) K
      hn✝ : LE.le (↑n) N
      hn : Ne n 0
      Z : LE.le (HMul.hMul (HPow.hPow (Norm.norm 0) n) (HMul.hMul (HPow.hPow (Norm.n …
      ⊢ LE.le (HMul.hMul (HPow.hPow (Norm.norm 0) n) (Norm.norm (iteratedFDeriv Real …
    -/
  · simp [hn]
    /-
      case inr.inl
      E : Type u_1
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace Complex E
      V : Type u_2
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : InnerProductSpace Real V
      inst✝² : FiniteDimensional Real V
      inst✝¹ : MeasurableSpace V
      inst✝ : BorelSpace V
      f : V → E
      K N : ENat
      hf : ContDiff Real (↑N) f
      h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
      k n : Nat
      hk : LE.le (↑k) K
      hn✝ : LE.le (↑n) N
      hn : Ne n 0
      Z : LE.le (HMul.hMul (HPow.hPow (Norm.norm 0) n) (HMul.hMul (HPow.hPow (Norm.n …
      ⊢ LE.le 0 (HMul.hMul (HMul.hMul (HPow.hPow (HMul.hMul 2 Real.pi) k) (HPow.hPow …
    -/
    positivity
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    V : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : FiniteDimensional Real V
    inst✝¹ : MeasurableSpace V
    inst✝ : BorelSpace V
    f : V → E
    K N : ENat
    hf : ContDiff Real (↑N) f
    h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
    k n : Nat
    hk : LE.le (↑k) K
    hn✝ : LE.le (↑n) N
    w : V
    Z : LE.le (HMul.hMul (HPow.hPow (Norm.norm w) n) (HMul.hMul (HPow.hPow (Norm.n …
    hn : Ne n 0
    hw : Ne w 0
    ⊢ LE.le (HMul.hMul (HPow.hPow (Norm.norm w) n) (Norm.norm (iteratedFDeriv Real …
  -/
  rw [mul_le_mul_left (pow_pos (by simp [hw]) n)] at Z
  /-
    case inr.inr
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    V : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : FiniteDimensional Real V
    inst✝¹ : MeasurableSpace V
    inst✝ : BorelSpace V
    f : V → E
    K N : ENat
    hf : ContDiff Real (↑N) f
    h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
    k n : Nat
    hk : LE.le (↑k) K
    hn✝ : LE.le (↑n) N
    w : V
    Z : LE.le (HMul.hMul (HPow.hPow (Norm.norm w) n) (Norm.norm (iteratedFDeriv Re …
    hn : Ne n 0
    hw : Ne w 0
    ⊢ LE.le (HMul.hMul (HPow.hPow (Norm.norm w) n) (Norm.norm (iteratedFDeriv Real …
  -/
  apply Z.trans
  /-
    case inr.inr
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    V : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : FiniteDimensional Real V
    inst✝¹ : MeasurableSpace V
    inst✝ : BorelSpace V
    f : V → E
    K N : ENat
    hf : ContDiff Real (↑N) f
    h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
    k n : Nat
    hk : LE.le (↑k) K
    hn✝ : LE.le (↑n) N
    w : V
    Z : LE.le (HMul.hMul (HPow.hPow (Norm.norm w) n) (Norm.norm (iteratedFDeriv Re …
    hn : Ne n 0
    hw : Ne w 0
    ⊢ LE.le (HMul.hMul (HPow.hPow (HMul.hMul 2 (HMul.hMul Real.pi (Norm.norm (inne …
  -/
  conv_rhs => rw [← mul_one π]
  /-
    case inr.inr
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    V : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : FiniteDimensional Real V
    inst✝¹ : MeasurableSpace V
    inst✝ : BorelSpace V
    f : V → E
    K N : ENat
    hf : ContDiff Real (↑N) f
    h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
    k n : Nat
    hk : LE.le (↑k) K
    hn✝ : LE.le (↑n) N
    w : V
    Z : LE.le (HMul.hMul (HPow.hPow (Norm.norm w) n) (Norm.norm (iteratedFDeriv Re …
    hn : Ne n 0
    hw : Ne w 0
    ⊢ LE.le (HMul.hMul (HPow.hPow (HMul.hMul 2 (HMul.hMul Real.pi (Norm.norm (inne …
  -/
  simp only [mul_assoc]
  /-
    case inr.inr
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    V : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : FiniteDimensional Real V
    inst✝¹ : MeasurableSpace V
    inst✝ : BorelSpace V
    f : V → E
    K N : ENat
    hf : ContDiff Real (↑N) f
    h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
    k n : Nat
    hk : LE.le (↑k) K
    hn✝ : LE.le (↑n) N
    w : V
    Z : LE.le (HMul.hMul (HPow.hPow (Norm.norm w) n) (Norm.norm (iteratedFDeriv Re …
    hn : Ne n 0
    hw : Ne w 0
    ⊢ LE.le (HMul.hMul (HPow.hPow (HMul.hMul 2 (HMul.hMul Real.pi (Norm.norm (inne …
  -/
  gcongr
  /-
    case inr.inr.h.hab.h.h
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    V : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : FiniteDimensional Real V
    inst✝¹ : MeasurableSpace V
    inst✝ : BorelSpace V
    f : V → E
    K N : ENat
    hf : ContDiff Real (↑N) f
    h'f : ∀ (k n : Nat), LE.le (↑k) K → LE.le (↑n) N → MeasureTheory.Integrable (f …
    k n : Nat
    hk : LE.le (↑k) K
    hn✝ : LE.le (↑n) N
    w : V
    Z : LE.le (HMul.hMul (HPow.hPow (Norm.norm w) n) (Norm.norm (iteratedFDeriv Re …
    hn : Ne n 0
    hw : Ne w 0
    ⊢ LE.le (Norm.norm (innerSL Real)) 1
  -/
  exact norm_innerSL_le _
  /-
    🎉 no goals
  -/


lemma hasDerivAt_fourierIntegral
                      /-
                        E : Type u_1
                        inst✝⁶ : NormedAddCommGroup E
                        inst✝⁵ : NormedSpace Complex E
                        V : Type u_2
                        inst✝⁴ : NormedAddCommGroup V
                        inst✝³ : InnerProductSpace Real V
                        inst✝² : FiniteDimensional Real V
                        inst✝¹ : MeasurableSpace V
                        inst✝ : BorelSpace V
                        f✝ : V → E
                        f : Real → E
                        ⊢ MeasureTheory.Measure Real
                      -/
                      /-
                        🎉 no goals
                      -/
    {f : ℝ → E} (hf : Integrable f) (hf' : Integrable (fun x : ℝ ↦ x • f x)) (w : ℝ) :
                                           /-
                                             🎉 no goals
                                           -/
    HasDerivAt (𝓕 f) (𝓕 (fun x : ℝ ↦ (-2 * π * I * x) • f x) w) w := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Real → E
    hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
    hf' : MeasureTheory.Integrable (fun x => HSMul.hSMul x (f x)) MeasureTheory.Me …
    w : Real
    ⊢ HasDerivAt (Real.fourierIntegral f) (Real.fourierIntegral (fun x => HSMul.hS …
  -/
  have hf'' : Integrable (fun v : ℝ ↦ ‖v‖ * ‖f v‖) := by simpa only [norm_smul] using hf'.norm
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Real → E
    hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
    hf' : MeasureTheory.Integrable (fun x => HSMul.hSMul x (f x)) MeasureTheory.Me …
    w : Real
    hf'' : MeasureTheory.Integrable (fun v => HMul.hMul (Norm.norm v) (Norm.norm ( …
    ⊢ HasDerivAt (Real.fourierIntegral f) (Real.fourierIntegral (fun x => HSMul.hS …
  -/
  let L := ContinuousLinearMap.mul ℝ ℝ
  have h_int : Integrable fun v ↦ fourierSMulRight L f v := by
    suffices Integrable fun v ↦ ContinuousLinearMap.smulRight (L v) (f v) by
      simpa only [fourierSMulRight, neg_smul, neg_mul, Pi.smul_apply] using this.smul (-2 * π * I)
    convert ((ContinuousLinearMap.ring_lmap_equiv_self ℝ
      E).symm.toContinuousLinearEquiv.toContinuousLinearMap).integrable_comp hf' using 2 with v
    apply ContinuousLinearMap.ext_ring
    rw [ContinuousLinearMap.smulRight_apply, ContinuousLinearMap.mul_apply', mul_one,
      ContinuousLinearMap.map_smul]
    exact congr_arg (fun x ↦ v • x) (one_smul ℝ (f v)).symm
  rw [← VectorFourier.fourierIntegral_convergent_iff continuous_fourierChar L.continuous₂ w]
    at h_int
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Real → E
    hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
    hf' : MeasureTheory.Integrable (fun x => HSMul.hSMul x (f x)) MeasureTheory.Me …
    w : Real
    hf'' : MeasureTheory.Integrable (fun v => HMul.hMul (Norm.norm v) (Norm.norm ( …
    L : ContinuousLinearMap (RingHom.id Real) Real (ContinuousLinearMap (RingHom.i …
    h_int : MeasureTheory.Integrable (fun v => HSMul.hSMul (Real.fourierChar (Neg. …
    ⊢ HasDerivAt (Real.fourierIntegral f) (Real.fourierIntegral (fun x => HSMul.hS …
  -/
  convert (VectorFourier.hasFDerivAt_fourierIntegral L hf hf'' w).hasDerivAt using 1
  /-
    case h.e'_9
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Real → E
    hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
    hf' : MeasureTheory.Integrable (fun x => HSMul.hSMul x (f x)) MeasureTheory.Me …
    w : Real
    hf'' : MeasureTheory.Integrable (fun v => HMul.hMul (Norm.norm v) (Norm.norm ( …
    L : ContinuousLinearMap (RingHom.id Real) Real (ContinuousLinearMap (RingHom.i …
    h_int : MeasureTheory.Integrable (fun v => HSMul.hSMul (Real.fourierChar (Neg. …
    ⊢ Eq (Real.fourierIntegral (fun x => HSMul.hSMul (HMul.hMul (HMul.hMul (HMul.h …
  -/
  erw [ContinuousLinearMap.integral_apply h_int]
  simp_rw [ContinuousLinearMap.smul_apply, fourierSMulRight, ContinuousLinearMap.smul_apply,
    ContinuousLinearMap.smulRight_apply, L, ContinuousLinearMap.mul_apply', mul_one,
    ← neg_mul, mul_smul]
  /-
    case h.e'_9
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Real → E
    hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
    hf' : MeasureTheory.Integrable (fun x => HSMul.hSMul x (f x)) MeasureTheory.Me …
    w : Real
    hf'' : MeasureTheory.Integrable (fun v => HMul.hMul (Norm.norm v) (Norm.norm ( …
    L : ContinuousLinearMap (RingHom.id Real) Real (ContinuousLinearMap (RingHom.i …
    h_int : MeasureTheory.Integrable (fun v => HSMul.hSMul (Real.fourierChar (Neg. …
    ⊢ Eq (Real.fourierIntegral (fun x => HSMul.hSMul (-2) (HSMul.hSMul (↑Real.pi)  …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem deriv_fourierIntegral
                      /-
                        E : Type u_1
                        inst✝⁶ : NormedAddCommGroup E
                        inst✝⁵ : NormedSpace Complex E
                        V : Type u_2
                        inst✝⁴ : NormedAddCommGroup V
                        inst✝³ : InnerProductSpace Real V
                        inst✝² : FiniteDimensional Real V
                        inst✝¹ : MeasurableSpace V
                        inst✝ : BorelSpace V
                        f✝ : V → E
                        f : Real → E
                        ⊢ MeasureTheory.Measure Real
                      -/
                      /-
                        🎉 no goals
                      -/
    {f : ℝ → E} (hf : Integrable f) (hf' : Integrable (fun x : ℝ ↦ x • f x)) :
                                           /-
                                             🎉 no goals
                                           -/
    deriv (𝓕 f) = 𝓕 (fun x : ℝ ↦ (-2 * π * I * x) • f x) := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Real → E
    hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
    hf' : MeasureTheory.Integrable (fun x => HSMul.hSMul x (f x)) MeasureTheory.Me …
    ⊢ Eq (deriv (Real.fourierIntegral f)) (Real.fourierIntegral fun x => HSMul.hSM …
  -/
  ext x
  /-
    case h
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Real → E
    hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
    hf' : MeasureTheory.Integrable (fun x => HSMul.hSMul x (f x)) MeasureTheory.Me …
    x : Real
    ⊢ Eq (deriv (Real.fourierIntegral f) x) (Real.fourierIntegral (fun x => HSMul. …
  -/
  exact (hasDerivAt_fourierIntegral hf hf' x).deriv
  /-
    🎉 no goals
  -/


/-- The Fourier integral of the Fréchet derivative of a function is obtained by multiplying the
Fourier integral of the original function by `2πI x`. -/
theorem fourierIntegral_deriv
                      /-
                        E : Type u_1
                        inst✝⁶ : NormedAddCommGroup E
                        inst✝⁵ : NormedSpace Complex E
                        V : Type u_2
                        inst✝⁴ : NormedAddCommGroup V
                        inst✝³ : InnerProductSpace Real V
                        inst✝² : FiniteDimensional Real V
                        inst✝¹ : MeasurableSpace V
                        inst✝ : BorelSpace V
                        f✝ : V → E
                        f : Real → E
                        ⊢ MeasureTheory.Measure Real
                      -/
                      /-
                        🎉 no goals
                      -/
    {f : ℝ → E} (hf : Integrable f) (h'f : Differentiable ℝ f) (hf' : Integrable (deriv f)) :
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
    𝓕 (deriv f) = fun (x : ℝ) ↦ (2 * π * I * x) • (𝓕 f x) := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Real → E
    hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
    h'f : Differentiable Real f
    hf' : MeasureTheory.Integrable (deriv f) MeasureTheory.MeasureSpace.volume
    ⊢ Eq (Real.fourierIntegral (deriv f)) fun x => HSMul.hSMul (HMul.hMul (HMul.hM …
  -/
  ext x
  have I : Integrable (fun x ↦ fderiv ℝ f x) := by
    simpa only [← deriv_fderiv] using (ContinuousLinearMap.smulRightL ℝ ℝ E 1).integrable_comp hf'
  have : 𝓕 (deriv f) x = 𝓕 (fderiv ℝ f) x 1 := by
    simp only [fourierIntegral_continuousLinearMap_apply I, fderiv_deriv]
  /-
    case h
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Real → E
    hf : MeasureTheory.Integrable f MeasureTheory.MeasureSpace.volume
    h'f : Differentiable Real f
    hf' : MeasureTheory.Integrable (deriv f) MeasureTheory.MeasureSpace.volume
    x : Real
    I : MeasureTheory.Integrable (fun x => fderiv Real f x) MeasureTheory.MeasureS …
    this : Eq (Real.fourierIntegral (deriv f) x) ((Real.fourierIntegral (fderiv Re …
    ⊢ Eq (Real.fourierIntegral (deriv f) x) (HSMul.hSMul (HMul.hMul (HMul.hMul (HM …
  -/
  rw [this, fourierIntegral_fderiv hf h'f I]
  simp only [fourierSMulRight_apply, ContinuousLinearMap.neg_apply, innerSL_apply, smul_smul,
    RCLike.inner_apply, conj_trivial, mul_one, neg_smul, smul_neg, neg_neg, neg_mul, ← coe_smul]


theorem iteratedDeriv_fourierIntegral {f : ℝ → E} {N : ℕ∞} {n : ℕ}
                             /-
                               E : Type u_1
                               inst✝⁶ : NormedAddCommGroup E
                               inst✝⁵ : NormedSpace Complex E
                               V : Type u_2
                               inst✝⁴ : NormedAddCommGroup V
                               inst✝³ : InnerProductSpace Real V
                               inst✝² : FiniteDimensional Real V
                               inst✝¹ : MeasurableSpace V
                               inst✝ : BorelSpace V
                               f✝ : V → E
                               f : Real → E
                               N : ENat
                               n✝ n : Nat
                               ⊢ MeasureTheory.Measure Real
                             -/
    (hf : ∀ (n : ℕ), n ≤ N → Integrable (fun x ↦ x^n • f x)) (hn : n ≤ N) :
                             /-
                               🎉 no goals
                             -/
    iteratedDeriv n (𝓕 f) = 𝓕 (fun x : ℝ ↦ (-2 * π * I * x) ^ n • f x) := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Real → E
    N : ENat
    n : Nat
    hf : ∀ (n : Nat), LE.le (↑n) N → MeasureTheory.Integrable (fun x => HSMul.hSMu …
    hn : LE.le (↑n) N
    ⊢ Eq (iteratedDeriv n (Real.fourierIntegral f)) (Real.fourierIntegral fun x => …
  -/
  ext x : 1
  have A (n : ℕ) (hn : n ≤ N) : Integrable (fun v ↦ ‖v‖^n * ‖f v‖) := by
    convert (hf n hn).norm with x
    simp [norm_smul]
  have B : AEStronglyMeasurable f := by
    convert (hf 0 (zero_le _)).1 with x
    simp
  rw [iteratedDeriv, iteratedFDeriv_fourierIntegral A B hn,
    fourierIntegral_continuousMultilinearMap_apply (integrable_fourierPowSMulRight _ (A n hn) B),
    fourierIntegral_eq, fourierIntegral_eq]
  /-
    case h
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Real → E
    N : ENat
    n : Nat
    hf : ∀ (n : Nat), LE.le (↑n) N → MeasureTheory.Integrable (fun x => HSMul.hSMu …
    hn : LE.le (↑n) N
    x : Real
    A : ∀ (n : Nat), LE.le (↑n) N → MeasureTheory.Integrable (fun v => HMul.hMul ( …
    B : MeasureTheory.AEStronglyMeasurable f MeasureTheory.MeasureSpace.volume
    ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun v => HSMul. …
  -/
  congr with y
  suffices (-(2 * π * I)) ^ n • y ^ n • f y = (-(2 * π * I * y)) ^ n • f y by
    simpa [innerSL_apply _]
  /-
    case h.e_f.h
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Real → E
    N : ENat
    n : Nat
    hf : ∀ (n : Nat), LE.le (↑n) N → MeasureTheory.Integrable (fun x => HSMul.hSMu …
    hn : LE.le (↑n) N
    x : Real
    A : ∀ (n : Nat), LE.le (↑n) N → MeasureTheory.Integrable (fun v => HMul.hMul ( …
    B : MeasureTheory.AEStronglyMeasurable f MeasureTheory.MeasureSpace.volume
    y : Real
    ⊢ Eq (HSMul.hSMul (HPow.hPow (Neg.neg (HMul.hMul (HMul.hMul 2 ↑Real.pi) Comple …
  -/
  simp only [← neg_mul, ← coe_smul, smul_smul, mul_pow, ofReal_pow, mul_assoc]
  /-
    🎉 no goals
  -/


theorem fourierIntegral_iteratedDeriv {f : ℝ → E} {N : ℕ∞} {n : ℕ} (hf : ContDiff ℝ N f)
                              /-
                                E : Type u_1
                                inst✝⁶ : NormedAddCommGroup E
                                inst✝⁵ : NormedSpace Complex E
                                V : Type u_2
                                inst✝⁴ : NormedAddCommGroup V
                                inst✝³ : InnerProductSpace Real V
                                inst✝² : FiniteDimensional Real V
                                inst✝¹ : MeasurableSpace V
                                inst✝ : BorelSpace V
                                f✝ : V → E
                                f : Real → E
                                N : ENat
                                n✝ : Nat
                                hf : ContDiff Real (↑N) f
                                n : Nat
                                ⊢ MeasureTheory.Measure Real
                              -/
    (h'f : ∀ (n : ℕ), n ≤ N → Integrable (iteratedDeriv n f)) (hn : n ≤ N) :
                              /-
                                🎉 no goals
                              -/
    𝓕 (iteratedDeriv n f) = fun (x : ℝ) ↦ (2 * π * I * x) ^ n • (𝓕 f x) := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Real → E
    N : ENat
    n : Nat
    hf : ContDiff Real (↑N) f
    h'f : ∀ (n : Nat), LE.le (↑n) N → MeasureTheory.Integrable (iteratedDeriv n f) …
    hn : LE.le (↑n) N
    ⊢ Eq (Real.fourierIntegral (iteratedDeriv n f)) fun x => HSMul.hSMul (HPow.hPo …
  -/
  ext x : 1
  have A : ∀ (n : ℕ), n ≤ N → Integrable (iteratedFDeriv ℝ n f) := by
    intro n hn
    rw [iteratedFDeriv_eq_equiv_comp]
    exact (LinearIsometryEquiv.integrable_comp_iff _).2 (h'f n hn)
  /-
    case h
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Real → E
    N : ENat
    n : Nat
    hf : ContDiff Real (↑N) f
    h'f : ∀ (n : Nat), LE.le (↑n) N → MeasureTheory.Integrable (iteratedDeriv n f) …
    hn : LE.le (↑n) N
    x : Real
    A : ∀ (n : Nat), LE.le (↑n) N → MeasureTheory.Integrable (iteratedFDeriv Real  …
    ⊢ Eq (Real.fourierIntegral (iteratedDeriv n f) x) (HSMul.hSMul (HPow.hPow (HMu …
  -/
  change 𝓕 (fun x ↦ iteratedDeriv n f x) x = _
  simp_rw [iteratedDeriv, ← fourierIntegral_continuousMultilinearMap_apply (A n hn),
    fourierIntegral_iteratedFDeriv hf A hn]
  /-
    case h
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : Real → E
    N : ENat
    n : Nat
    hf : ContDiff Real (↑N) f
    h'f : ∀ (n : Nat), LE.le (↑n) N → MeasureTheory.Integrable (iteratedDeriv n f) …
    hn : LE.le (↑n) N
    x : Real
    A : ∀ (n : Nat), LE.le (↑n) N → MeasureTheory.Integrable (iteratedFDeriv Real  …
    ⊢ Eq ((VectorFourier.fourierPowSMulRight (Neg.neg (innerSL Real)) (Real.fourie …
  -/
  simp [← coe_smul, smul_smul, ← mul_pow]
  /-
    🎉 no goals
  -/


