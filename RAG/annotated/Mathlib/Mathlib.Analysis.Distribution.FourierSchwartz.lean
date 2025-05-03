/-- The Fourier transform on a real inner product space, as a continuous linear map on the
Schwartz space. -/
noncomputable def fourierTransformCLM : 𝓢(V, E) →L[𝕜] 𝓢(V, E) := by
  /-
    𝕜 : Type u_1
    inst✝⁹ : RCLike 𝕜
    E : Type u_2
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace Complex E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : SMulCommClass Complex 𝕜 E
    V : Type u_3
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : FiniteDimensional Real V
    inst✝¹ : MeasurableSpace V
    inst✝ : BorelSpace V
    ⊢ ContinuousLinearMap (RingHom.id 𝕜) (SchwartzMap V E) (SchwartzMap V E)
  -/
  refine mkCLM (fun (f : V → E) ↦ 𝓕 f) ?_ ?_ ?_ ?_
    /-
      case refine_1
      𝕜 : Type u_1
      inst✝⁹ : RCLike 𝕜
      E : Type u_2
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Complex E
      inst✝⁶ : NormedSpace 𝕜 E
      inst✝⁵ : SMulCommClass Complex 𝕜 E
      V : Type u_3
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : InnerProductSpace Real V
      inst✝² : FiniteDimensional Real V
      inst✝¹ : MeasurableSpace V
      inst✝ : BorelSpace V
      ⊢ ∀ (f g : SchwartzMap V E) (x : V), Eq ((fun f => Real.fourierIntegral f) (HA …
    -/
  · intro f g x
    /-
      case refine_1
      𝕜 : Type u_1
      inst✝⁹ : RCLike 𝕜
      E : Type u_2
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Complex E
      inst✝⁶ : NormedSpace 𝕜 E
      inst✝⁵ : SMulCommClass Complex 𝕜 E
      V : Type u_3
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : InnerProductSpace Real V
      inst✝² : FiniteDimensional Real V
      inst✝¹ : MeasurableSpace V
      inst✝ : BorelSpace V
      f g : SchwartzMap V E
      x : V
      ⊢ Eq ((fun f => Real.fourierIntegral f) (HAdd.hAdd ⇑f ⇑g) x) (HAdd.hAdd ((fun  …
    -/
    simp only [fourierIntegral_eq, Pi.add_apply, smul_add]
    /-
      case refine_1
      𝕜 : Type u_1
      inst✝⁹ : RCLike 𝕜
      E : Type u_2
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Complex E
      inst✝⁶ : NormedSpace 𝕜 E
      inst✝⁵ : SMulCommClass Complex 𝕜 E
      V : Type u_3
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : InnerProductSpace Real V
      inst✝² : FiniteDimensional Real V
      inst✝¹ : MeasurableSpace V
      inst✝ : BorelSpace V
      f g : SchwartzMap V E
      x : V
      ⊢ Eq (MeasureTheory.integral MeasureTheory.MeasureSpace.volume fun v => HAdd.h …
    -/
    rw [integral_add]
      /-
        case refine_1.hf
        𝕜 : Type u_1
        inst✝⁹ : RCLike 𝕜
        E : Type u_2
        inst✝⁸ : NormedAddCommGroup E
        inst✝⁷ : NormedSpace Complex E
        inst✝⁶ : NormedSpace 𝕜 E
        inst✝⁵ : SMulCommClass Complex 𝕜 E
        V : Type u_3
        inst✝⁴ : NormedAddCommGroup V
        inst✝³ : InnerProductSpace Real V
        inst✝² : FiniteDimensional Real V
        inst✝¹ : MeasurableSpace V
        inst✝ : BorelSpace V
        f g : SchwartzMap V E
        x : V
        ⊢ MeasureTheory.Integrable (fun v => HSMul.hSMul (Real.fourierChar (Neg.neg (I …
      -/
    · exact (fourierIntegral_convergent_iff _).2 f.integrable
      /-
        🎉 no goals
      -/
      /-
        case refine_1.hg
        𝕜 : Type u_1
        inst✝⁹ : RCLike 𝕜
        E : Type u_2
        inst✝⁸ : NormedAddCommGroup E
        inst✝⁷ : NormedSpace Complex E
        inst✝⁶ : NormedSpace 𝕜 E
        inst✝⁵ : SMulCommClass Complex 𝕜 E
        V : Type u_3
        inst✝⁴ : NormedAddCommGroup V
        inst✝³ : InnerProductSpace Real V
        inst✝² : FiniteDimensional Real V
        inst✝¹ : MeasurableSpace V
        inst✝ : BorelSpace V
        f g : SchwartzMap V E
        x : V
        ⊢ MeasureTheory.Integrable (fun v => HSMul.hSMul (Real.fourierChar (Neg.neg (I …
      -/
    · exact (fourierIntegral_convergent_iff _).2 g.integrable
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      𝕜 : Type u_1
      inst✝⁹ : RCLike 𝕜
      E : Type u_2
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Complex E
      inst✝⁶ : NormedSpace 𝕜 E
      inst✝⁵ : SMulCommClass Complex 𝕜 E
      V : Type u_3
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : InnerProductSpace Real V
      inst✝² : FiniteDimensional Real V
      inst✝¹ : MeasurableSpace V
      inst✝ : BorelSpace V
      ⊢ ∀ (a : 𝕜) (f : SchwartzMap V E) (x : V), Eq ((fun f => Real.fourierIntegral  …
    -/
  · intro c f x
    /-
      case refine_2
      𝕜 : Type u_1
      inst✝⁹ : RCLike 𝕜
      E : Type u_2
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Complex E
      inst✝⁶ : NormedSpace 𝕜 E
      inst✝⁵ : SMulCommClass Complex 𝕜 E
      V : Type u_3
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : InnerProductSpace Real V
      inst✝² : FiniteDimensional Real V
      inst✝¹ : MeasurableSpace V
      inst✝ : BorelSpace V
      c : 𝕜
      f : SchwartzMap V E
      x : V
      ⊢ Eq ((fun f => Real.fourierIntegral f) (HSMul.hSMul c ⇑f) x) (HSMul.hSMul ((R …
    -/
    simp only [fourierIntegral_eq, Pi.smul_apply, RingHom.id_apply, smul_comm _ c, integral_smul]
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      𝕜 : Type u_1
      inst✝⁹ : RCLike 𝕜
      E : Type u_2
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Complex E
      inst✝⁶ : NormedSpace 𝕜 E
      inst✝⁵ : SMulCommClass Complex 𝕜 E
      V : Type u_3
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : InnerProductSpace Real V
      inst✝² : FiniteDimensional Real V
      inst✝¹ : MeasurableSpace V
      inst✝ : BorelSpace V
      ⊢ ∀ (f : SchwartzMap V E), ContDiff Real (↑Top.top) ((fun f => Real.fourierInt …
    -/
  · intro f
    /-
      case refine_3
      𝕜 : Type u_1
      inst✝⁹ : RCLike 𝕜
      E : Type u_2
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Complex E
      inst✝⁶ : NormedSpace 𝕜 E
      inst✝⁵ : SMulCommClass Complex 𝕜 E
      V : Type u_3
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : InnerProductSpace Real V
      inst✝² : FiniteDimensional Real V
      inst✝¹ : MeasurableSpace V
      inst✝ : BorelSpace V
      f : SchwartzMap V E
      ⊢ ContDiff Real (↑Top.top) ((fun f => Real.fourierIntegral f) ⇑f)
    -/
    exact Real.contDiff_fourierIntegral (fun n _ ↦ integrable_pow_mul volume f n)
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      𝕜 : Type u_1
      inst✝⁹ : RCLike 𝕜
      E : Type u_2
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Complex E
      inst✝⁶ : NormedSpace 𝕜 E
      inst✝⁵ : SMulCommClass Complex 𝕜 E
      V : Type u_3
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : InnerProductSpace Real V
      inst✝² : FiniteDimensional Real V
      inst✝¹ : MeasurableSpace V
      inst✝ : BorelSpace V
      ⊢ ∀ (n : Prod Nat Nat), Exists fun s => Exists fun C => And (LE.le 0 C) (∀ (f  …
    -/
  · rintro ⟨k, n⟩
    refine ⟨Finset.range (n + integrablePower (volume : Measure V) + 1) ×ˢ Finset.range (k + 1),
       (2 * π) ^ n * (2 * ↑n + 2) ^ k * (Finset.range (n + 1) ×ˢ Finset.range (k + 1)).card
         * 2 ^ integrablePower (volume : Measure V) *
         (∫ (x : V), (1 + ‖x‖) ^ (- (integrablePower (volume : Measure V) : ℝ))) * 2,
       ⟨by positivity, fun f x ↦ ?_⟩⟩
    apply (pow_mul_norm_iteratedFDeriv_fourierIntegral_le (f.smooth ⊤)
      (fun k n _hk _hn ↦ integrable_pow_mul_iteratedFDeriv _ f k n) le_top le_top x).trans
    /-
      case refine_4.mk
      𝕜 : Type u_1
      inst✝⁹ : RCLike 𝕜
      E : Type u_2
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Complex E
      inst✝⁶ : NormedSpace 𝕜 E
      inst✝⁵ : SMulCommClass Complex 𝕜 E
      V : Type u_3
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : InnerProductSpace Real V
      inst✝² : FiniteDimensional Real V
      inst✝¹ : MeasurableSpace V
      inst✝ : BorelSpace V
      k n : Nat
      f : SchwartzMap V E
      x : V
      ⊢ LE.le (HMul.hMul (HMul.hMul (HPow.hPow (HMul.hMul 2 Real.pi) { fst := k, snd …
    -/
    simp only [mul_assoc]
    /-
      case refine_4.mk
      𝕜 : Type u_1
      inst✝⁹ : RCLike 𝕜
      E : Type u_2
      inst✝⁸ : NormedAddCommGroup E
      inst✝⁷ : NormedSpace Complex E
      inst✝⁶ : NormedSpace 𝕜 E
      inst✝⁵ : SMulCommClass Complex 𝕜 E
      V : Type u_3
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : InnerProductSpace Real V
      inst✝² : FiniteDimensional Real V
      inst✝¹ : MeasurableSpace V
      inst✝ : BorelSpace V
      k n : Nat
      f : SchwartzMap V E
      x : V
      ⊢ LE.le (HMul.hMul (HPow.hPow (HMul.hMul 2 Real.pi) n) (HMul.hMul (HPow.hPow ( …
    -/
    gcongr
    calc
    ∑ p in Finset.range (n + 1) ×ˢ Finset.range (k + 1),
        ∫ (v : V), ‖v‖ ^ p.1 * ‖iteratedFDeriv ℝ p.2 (⇑f) v‖
      ≤ ∑ p in Finset.range (n + 1) ×ˢ Finset.range (k + 1),
        2 ^ integrablePower (volume : Measure V) *
        (∫ (x : V), (1 + ‖x‖) ^ (- (integrablePower (volume : Measure V) : ℝ))) * 2 *
        ((Finset.range (n + integrablePower (volume : Measure V) + 1) ×ˢ Finset.range (k + 1)).sup
          (schwartzSeminormFamily 𝕜 V E)) f := by
      apply Finset.sum_le_sum (fun p hp ↦ ?_)
      simp only [Finset.mem_product, Finset.mem_range] at hp
      apply (f.integral_pow_mul_iteratedFDeriv_le 𝕜 _ _ _).trans
      simp only [mul_assoc]
      rw [two_mul]
      gcongr
      · apply Seminorm.le_def.1
        have : (0, p.2) ∈ (Finset.range (n + integrablePower (volume : Measure V) + 1)
            ×ˢ Finset.range (k + 1)) := by simp [hp.2]
        apply Finset.le_sup this (f := fun p ↦ SchwartzMap.seminorm 𝕜 p.1 p.2 (E := V) (F := E))
      · apply Seminorm.le_def.1
        have : (p.1 + integrablePower (volume : Measure V), p.2) ∈ (Finset.range
            (n + integrablePower (volume : Measure V) + 1) ×ˢ Finset.range (k + 1)) := by
          simp [hp.2]
          omega
        apply Finset.le_sup this (f := fun p ↦ SchwartzMap.seminorm 𝕜 p.1 p.2 (E := V) (F := E))
    _ = _ := by simp [mul_assoc]


@[simp] lemma fourierTransformCLM_apply (f : 𝓢(V, E)) :
    fourierTransformCLM 𝕜 f = 𝓕 f := rfl


/-- The Fourier transform on a real inner product space, as a continuous linear equiv on the
Schwartz space. -/
noncomputable def fourierTransformCLE : 𝓢(V, E) ≃L[𝕜] 𝓢(V, E) where
  __ := fourierTransformCLM 𝕜
  invFun := (compCLMOfContinuousLinearEquiv 𝕜 (LinearIsometryEquiv.neg ℝ (E := V)))
      ∘L (fourierTransformCLM 𝕜)
  left_inv := by
    /-
      𝕜 : Type u_1
      inst✝¹⁰ : RCLike 𝕜
      E : Type u_2
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace Complex E
      inst✝⁷ : NormedSpace 𝕜 E
      inst✝⁶ : SMulCommClass Complex 𝕜 E
      V : Type u_3
      inst✝⁵ : NormedAddCommGroup V
      inst✝⁴ : InnerProductSpace Real V
      inst✝³ : FiniteDimensional Real V
      inst✝² : MeasurableSpace V
      inst✝¹ : BorelSpace V
      inst✝ : CompleteSpace E
      ⊢ Function.LeftInverse (⇑((SchwartzMap.compCLMOfContinuousLinearEquiv 𝕜 { toLi …
    -/
    intro f
    /-
      𝕜 : Type u_1
      inst✝¹⁰ : RCLike 𝕜
      E : Type u_2
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace Complex E
      inst✝⁷ : NormedSpace 𝕜 E
      inst✝⁶ : SMulCommClass Complex 𝕜 E
      V : Type u_3
      inst✝⁵ : NormedAddCommGroup V
      inst✝⁴ : InnerProductSpace Real V
      inst✝³ : FiniteDimensional Real V
      inst✝² : MeasurableSpace V
      inst✝¹ : BorelSpace V
      inst✝ : CompleteSpace E
      f : SchwartzMap V E
      ⊢ Eq (((SchwartzMap.compCLMOfContinuousLinearEquiv 𝕜 { toLinearEquiv := (Linea …
    -/
    ext x
    /-
      case h
      𝕜 : Type u_1
      inst✝¹⁰ : RCLike 𝕜
      E : Type u_2
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace Complex E
      inst✝⁷ : NormedSpace 𝕜 E
      inst✝⁶ : SMulCommClass Complex 𝕜 E
      V : Type u_3
      inst✝⁵ : NormedAddCommGroup V
      inst✝⁴ : InnerProductSpace Real V
      inst✝³ : FiniteDimensional Real V
      inst✝² : MeasurableSpace V
      inst✝¹ : BorelSpace V
      inst✝ : CompleteSpace E
      f : SchwartzMap V E
      x : V
      ⊢ Eq ((((SchwartzMap.compCLMOfContinuousLinearEquiv 𝕜 { toLinearEquiv := (Line …
    -/
    change 𝓕 (𝓕 f) (-x) = f x
    rw [← fourierIntegralInv_eq_fourierIntegral_neg, Continuous.fourier_inversion f.continuous
      f.integrable (fourierTransformCLM 𝕜 f).integrable]
  right_inv := by
    /-
      𝕜 : Type u_1
      inst✝¹⁰ : RCLike 𝕜
      E : Type u_2
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace Complex E
      inst✝⁷ : NormedSpace 𝕜 E
      inst✝⁶ : SMulCommClass Complex 𝕜 E
      V : Type u_3
      inst✝⁵ : NormedAddCommGroup V
      inst✝⁴ : InnerProductSpace Real V
      inst✝³ : FiniteDimensional Real V
      inst✝² : MeasurableSpace V
      inst✝¹ : BorelSpace V
      inst✝ : CompleteSpace E
      ⊢ Function.RightInverse (⇑((SchwartzMap.compCLMOfContinuousLinearEquiv 𝕜 { toL …
    -/
    intro f
    /-
      𝕜 : Type u_1
      inst✝¹⁰ : RCLike 𝕜
      E : Type u_2
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace Complex E
      inst✝⁷ : NormedSpace 𝕜 E
      inst✝⁶ : SMulCommClass Complex 𝕜 E
      V : Type u_3
      inst✝⁵ : NormedAddCommGroup V
      inst✝⁴ : InnerProductSpace Real V
      inst✝³ : FiniteDimensional Real V
      inst✝² : MeasurableSpace V
      inst✝¹ : BorelSpace V
      inst✝ : CompleteSpace E
      f : SchwartzMap V E
      ⊢ Eq ((↑__spread✝⁻⁰).toFun (((SchwartzMap.compCLMOfContinuousLinearEquiv 𝕜 { t …
    -/
    ext x
    /-
      case h
      𝕜 : Type u_1
      inst✝¹⁰ : RCLike 𝕜
      E : Type u_2
      inst✝⁹ : NormedAddCommGroup E
      inst✝⁸ : NormedSpace Complex E
      inst✝⁷ : NormedSpace 𝕜 E
      inst✝⁶ : SMulCommClass Complex 𝕜 E
      V : Type u_3
      inst✝⁵ : NormedAddCommGroup V
      inst✝⁴ : InnerProductSpace Real V
      inst✝³ : FiniteDimensional Real V
      inst✝² : MeasurableSpace V
      inst✝¹ : BorelSpace V
      inst✝ : CompleteSpace E
      f : SchwartzMap V E
      x : V
      ⊢ Eq (((↑__spread✝⁻⁰).toFun (((SchwartzMap.compCLMOfContinuousLinearEquiv 𝕜 {  …
    -/
    change 𝓕 (fun x ↦ (𝓕 f) (-x)) x = f x
    simp_rw [← fourierIntegralInv_eq_fourierIntegral_neg, Continuous.fourier_inversion_inv
      f.continuous f.integrable (fourierTransformCLM 𝕜 f).integrable]
  continuous_invFun := ContinuousLinearMap.continuous _


@[simp] lemma fourierTransformCLE_apply (f : 𝓢(V, E)) :
    fourierTransformCLE 𝕜 f = 𝓕 f := rfl


@[simp] lemma fourierTransformCLE_symm_apply (f : 𝓢(V, E)) :
    (fourierTransformCLE 𝕜).symm f = 𝓕⁻ f := by
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : RCLike 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace Complex E
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : SMulCommClass Complex 𝕜 E
    V : Type u_3
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : InnerProductSpace Real V
    inst✝³ : FiniteDimensional Real V
    inst✝² : MeasurableSpace V
    inst✝¹ : BorelSpace V
    inst✝ : CompleteSpace E
    f : SchwartzMap V E
    ⊢ Eq (⇑((SchwartzMap.fourierTransformCLE 𝕜).symm f)) (Real.fourierIntegralInv  …
  -/
  ext x
  /-
    case h
    𝕜 : Type u_1
    inst✝¹⁰ : RCLike 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace Complex E
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : SMulCommClass Complex 𝕜 E
    V : Type u_3
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : InnerProductSpace Real V
    inst✝³ : FiniteDimensional Real V
    inst✝² : MeasurableSpace V
    inst✝¹ : BorelSpace V
    inst✝ : CompleteSpace E
    f : SchwartzMap V E
    x : V
    ⊢ Eq (((SchwartzMap.fourierTransformCLE 𝕜).symm f) x) (Real.fourierIntegralInv …
  -/
  exact (fourierIntegralInv_eq_fourierIntegral_neg f x).symm
  /-
    🎉 no goals
  -/


