/-- Kernel with image `(κ a).withDensity (f a)` if `Function.uncurry f` is measurable, and
with image 0 otherwise. If `Function.uncurry f` is measurable, it satisfies
`∫⁻ b, g b ∂(withDensity κ f hf a) = ∫⁻ b, f a b * g b ∂(κ a)`. -/
noncomputable def withDensity (κ : Kernel α β) [IsSFiniteKernel κ] (f : α → β → ℝ≥0∞) :
    Kernel α β :=
  @dite _ (Measurable (Function.uncurry f)) (Classical.dec _) (fun hf =>
    (⟨fun a => (κ a).withDensity (f a),
      by
        /-
          α : Type u_1
          β : Type u_2
          ι : Type u_3
          mα : MeasurableSpace α
          mβ : MeasurableSpace β
          κ✝ : ProbabilityTheory.Kernel α β
          f✝ : α → β → ENNReal
          κ : ProbabilityTheory.Kernel α β
          inst✝ : ProbabilityTheory.IsSFiniteKernel κ
          f : α → β → ENNReal
          hf : Measurable (Function.uncurry f)
          ⊢ Measurable fun a => (κ a).withDensity (f a)
        -/
        refine Measure.measurable_of_measurable_coe _ fun s hs => ?_
        /-
          α : Type u_1
          β : Type u_2
          ι : Type u_3
          mα : MeasurableSpace α
          mβ : MeasurableSpace β
          κ✝ : ProbabilityTheory.Kernel α β
          f✝ : α → β → ENNReal
          κ : ProbabilityTheory.Kernel α β
          inst✝ : ProbabilityTheory.IsSFiniteKernel κ
          f : α → β → ENNReal
          hf : Measurable (Function.uncurry f)
          s : Set β
          hs : MeasurableSet s
          ⊢ Measurable fun b => ((κ b).withDensity (f b)) s
        -/
        simp_rw [withDensity_apply _ hs]
        /-
          α : Type u_1
          β : Type u_2
          ι : Type u_3
          mα : MeasurableSpace α
          mβ : MeasurableSpace β
          κ✝ : ProbabilityTheory.Kernel α β
          f✝ : α → β → ENNReal
          κ : ProbabilityTheory.Kernel α β
          inst✝ : ProbabilityTheory.IsSFiniteKernel κ
          f : α → β → ENNReal
          hf : Measurable (Function.uncurry f)
          s : Set β
          hs : MeasurableSet s
          ⊢ Measurable fun b => MeasureTheory.lintegral ((κ b).restrict s) fun a => f b a
        -/
        exact hf.setLIntegral_kernel_prod_right hs⟩ : Kernel α β)) fun _ => 0
        /-
          🎉 no goals
        -/


theorem withDensity_of_not_measurable (κ : Kernel α β) [IsSFiniteKernel κ]
                                                                        /-
                                                                          α : Type u_1
                                                                          β : Type u_2
                                                                          mα : MeasurableSpace α
                                                                          mβ : MeasurableSpace β
                                                                          f : α → β → ENNReal
                                                                          κ : ProbabilityTheory.Kernel α β
                                                                          inst✝ : ProbabilityTheory.IsSFiniteKernel κ
                                                                          hf : Not (Measurable (Function.uncurry f))
                                                                          ⊢ Eq (κ.withDensity f) 0
                                                                        -/
    (hf : ¬Measurable (Function.uncurry f)) : withDensity κ f = 0 := by classical exact dif_neg hf
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


protected theorem withDensity_apply (κ : Kernel α β) [IsSFiniteKernel κ]
    (hf : Measurable (Function.uncurry f)) (a : α) :
    withDensity κ f a = (κ a).withDensity (f a) := by
  classical
  rw [withDensity, dif_pos hf]
  rfl


protected theorem withDensity_apply' (κ : Kernel α β) [IsSFiniteKernel κ]
    (hf : Measurable (Function.uncurry f)) (a : α) (s : Set β) :
    withDensity κ f a s = ∫⁻ b in s, f a b ∂κ a := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    f : α → β → ENNReal
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    hf : Measurable (Function.uncurry f)
    a : α
    s : Set β
    ⊢ Eq (((κ.withDensity f) a) s) (MeasureTheory.lintegral ((κ a).restrict s) fun …
  -/
  rw [Kernel.withDensity_apply κ hf, withDensity_apply' _ s]
  /-
    🎉 no goals
  -/


nonrec lemma withDensity_congr_ae (κ : Kernel α β) [IsSFiniteKernel κ] {f g : α → β → ℝ≥0∞}
    (hf : Measurable (Function.uncurry f)) (hg : Measurable (Function.uncurry g))
    (hfg : ∀ a, f a =ᵐ[κ a] g a) :
    withDensity κ f = withDensity κ g := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    f g : α → β → ENNReal
    hf : Measurable (Function.uncurry f)
    hg : Measurable (Function.uncurry g)
    hfg : ∀ (a : α), (MeasureTheory.ae (κ a)).EventuallyEq (f a) (g a)
    ⊢ Eq (κ.withDensity f) (κ.withDensity g)
  -/
  ext a
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    f g : α → β → ENNReal
    hf : Measurable (Function.uncurry f)
    hg : Measurable (Function.uncurry g)
    hfg : ∀ (a : α), (MeasureTheory.ae (κ a)).EventuallyEq (f a) (g a)
    a : α
    s✝ : Set β
    a✝ : MeasurableSet s✝
    ⊢ Eq (((κ.withDensity f) a) s✝) (((κ.withDensity g) a) s✝)
  -/
  rw [Kernel.withDensity_apply _ hf,Kernel.withDensity_apply _ hg, withDensity_congr_ae (hfg a)]
  /-
    🎉 no goals
  -/


nonrec lemma withDensity_absolutelyContinuous [IsSFiniteKernel κ]
    (f : α → β → ℝ≥0∞) (a : α) :
    Kernel.withDensity κ f a ≪ κ a := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    f : α → β → ENNReal
    a : α
    ⊢ ((κ.withDensity f) a).AbsolutelyContinuous (κ a)
  -/
  by_cases hf : Measurable (Function.uncurry f)
    /-
      case pos
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α β
      inst✝ : ProbabilityTheory.IsSFiniteKernel κ
      f : α → β → ENNReal
      a : α
      hf : Measurable (Function.uncurry f)
      ⊢ ((κ.withDensity f) a).AbsolutelyContinuous (κ a)
    -/
  · rw [Kernel.withDensity_apply _ hf]
    /-
      case pos
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α β
      inst✝ : ProbabilityTheory.IsSFiniteKernel κ
      f : α → β → ENNReal
      a : α
      hf : Measurable (Function.uncurry f)
      ⊢ ((κ a).withDensity (f a)).AbsolutelyContinuous (κ a)
    -/
    exact withDensity_absolutelyContinuous _ _
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α β
      inst✝ : ProbabilityTheory.IsSFiniteKernel κ
      f : α → β → ENNReal
      a : α
      hf : Not (Measurable (Function.uncurry f))
      ⊢ ((κ.withDensity f) a).AbsolutelyContinuous (κ a)
    -/
  · rw [withDensity_of_not_measurable _ hf]
    /-
      case neg
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α β
      inst✝ : ProbabilityTheory.IsSFiniteKernel κ
      f : α → β → ENNReal
      a : α
      hf : Not (Measurable (Function.uncurry f))
      ⊢ (0 a).AbsolutelyContinuous (κ a)
    -/
    simp [Measure.AbsolutelyContinuous.zero]
    /-
      🎉 no goals
    -/


@[simp]
lemma withDensity_one (κ : Kernel α β) [IsSFiniteKernel κ] :
    Kernel.withDensity κ 1 = κ := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    ⊢ Eq (κ.withDensity 1) κ
  -/
  ext; rw [Kernel.withDensity_apply _ measurable_const]; simp
                                                         /-
                                                           🎉 no goals
                                                         -/


@[simp]
lemma withDensity_one' (κ : Kernel α β) [IsSFiniteKernel κ] :
    Kernel.withDensity κ (fun _ _ ↦ 1) = κ := Kernel.withDensity_one _


@[simp]
lemma withDensity_zero (κ : Kernel α β) [IsSFiniteKernel κ] :
    Kernel.withDensity κ 0 = 0 := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    ⊢ Eq (κ.withDensity 0) 0
  -/
  ext; rw [Kernel.withDensity_apply _ measurable_const]; simp
                                                         /-
                                                           🎉 no goals
                                                         -/


@[simp]
lemma withDensity_zero' (κ : Kernel α β) [IsSFiniteKernel κ] :
    Kernel.withDensity κ (fun _ _ ↦ 0) = 0 := Kernel.withDensity_zero _


theorem lintegral_withDensity (κ : Kernel α β) [IsSFiniteKernel κ]
    (hf : Measurable (Function.uncurry f)) (a : α) {g : β → ℝ≥0∞} (hg : Measurable g) :
    ∫⁻ b, g b ∂withDensity κ f a = ∫⁻ b, f a b * g b ∂κ a := by
  rw [Kernel.withDensity_apply _ hf,
    lintegral_withDensity_eq_lintegral_mul _ (Measurable.of_uncurry_left hf) hg]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    f : α → β → ENNReal
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    hf : Measurable (Function.uncurry f)
    a : α
    g : β → ENNReal
    hg : Measurable g
    ⊢ Eq (MeasureTheory.lintegral (κ a) fun a_1 => HMul.hMul (f a) g a_1) (Measure …
  -/
  simp_rw [Pi.mul_apply]
  /-
    🎉 no goals
  -/


theorem integral_withDensity {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    {f : β → E} [IsSFiniteKernel κ] {a : α} {g : α → β → ℝ≥0}
    (hg : Measurable (Function.uncurry g)) :
    ∫ b, f b ∂withDensity κ (fun a b => g a b) a = ∫ b, g a b • f b ∂κ a := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    E : Type u_4
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    f : β → E
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    a : α
    g : α → β → NNReal
    hg : Measurable (Function.uncurry g)
    ⊢ Eq (MeasureTheory.integral ((κ.withDensity fun a b => ↑(g a b)) a) fun b =>  …
  -/
  rw [Kernel.withDensity_apply, integral_withDensity_eq_integral_smul]
    /-
      case f_meas
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α β
      E : Type u_4
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      f : β → E
      inst✝ : ProbabilityTheory.IsSFiniteKernel κ
      a : α
      g : α → β → NNReal
      hg : Measurable (Function.uncurry g)
      ⊢ Measurable (g a)
    -/
  · exact Measurable.of_uncurry_left hg
    /-
      🎉 no goals
    -/
    /-
      case hf
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α β
      E : Type u_4
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      f : β → E
      inst✝ : ProbabilityTheory.IsSFiniteKernel κ
      a : α
      g : α → β → NNReal
      hg : Measurable (Function.uncurry g)
      ⊢ Measurable (Function.uncurry fun a b => ↑(g a b))
    -/
  · exact measurable_coe_nnreal_ennreal.comp hg
    /-
      🎉 no goals
    -/


theorem withDensity_add_left (κ η : Kernel α β) [IsSFiniteKernel κ] [IsSFiniteKernel η]
    (f : α → β → ℝ≥0∞) : withDensity (κ + η) f = withDensity κ f + withDensity η f := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ η : ProbabilityTheory.Kernel α β
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    f : α → β → ENNReal
    ⊢ Eq ((HAdd.hAdd κ η).withDensity f) (HAdd.hAdd (κ.withDensity f) (η.withDensi …
  -/
  by_cases hf : Measurable (Function.uncurry f)
    /-
      case pos
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ η : ProbabilityTheory.Kernel α β
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      f : α → β → ENNReal
      hf : Measurable (Function.uncurry f)
      ⊢ Eq ((HAdd.hAdd κ η).withDensity f) (HAdd.hAdd (κ.withDensity f) (η.withDensi …
    -/
  · ext a s
    simp only [Kernel.withDensity_apply _ hf, coe_add, Pi.add_apply, withDensity_add_measure,
      Measure.add_apply]
    /-
      case neg
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ η : ProbabilityTheory.Kernel α β
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      f : α → β → ENNReal
      hf : Not (Measurable (Function.uncurry f))
      ⊢ Eq ((HAdd.hAdd κ η).withDensity f) (HAdd.hAdd (κ.withDensity f) (η.withDensi …
    -/
  · simp_rw [withDensity_of_not_measurable _ hf]
    /-
      case neg
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ η : ProbabilityTheory.Kernel α β
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      f : α → β → ENNReal
      hf : Not (Measurable (Function.uncurry f))
      ⊢ Eq 0 (HAdd.hAdd 0 0)
    -/
    rw [zero_add]
    /-
      🎉 no goals
    -/


theorem withDensity_kernel_sum [Countable ι] (κ : ι → Kernel α β) (hκ : ∀ i, IsSFiniteKernel (κ i))
    (f : α → β → ℝ≥0∞) :
    @withDensity _ _ _ _ (Kernel.sum κ) (isSFiniteKernel_sum hκ) f =
      Kernel.sum fun i => withDensity (κ i) f := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝ : Countable ι
    κ : ι → ProbabilityTheory.Kernel α β
    hκ : ∀ (i : ι), ProbabilityTheory.IsSFiniteKernel (κ i)
    f : α → β → ENNReal
    ⊢ Eq ((ProbabilityTheory.Kernel.sum κ).withDensity f) (ProbabilityTheory.Kerne …
  -/
  by_cases hf : Measurable (Function.uncurry f)
    /-
      case pos
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      inst✝ : Countable ι
      κ : ι → ProbabilityTheory.Kernel α β
      hκ : ∀ (i : ι), ProbabilityTheory.IsSFiniteKernel (κ i)
      f : α → β → ENNReal
      hf : Measurable (Function.uncurry f)
      ⊢ Eq ((ProbabilityTheory.Kernel.sum κ).withDensity f) (ProbabilityTheory.Kerne …
    -/
  · ext1 a
    simp_rw [sum_apply, Kernel.withDensity_apply _ hf, sum_apply,
      withDensity_sum (fun n => κ n a) (f a)]
    /-
      case neg
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      inst✝ : Countable ι
      κ : ι → ProbabilityTheory.Kernel α β
      hκ : ∀ (i : ι), ProbabilityTheory.IsSFiniteKernel (κ i)
      f : α → β → ENNReal
      hf : Not (Measurable (Function.uncurry f))
      ⊢ Eq ((ProbabilityTheory.Kernel.sum κ).withDensity f) (ProbabilityTheory.Kerne …
    -/
  · simp_rw [withDensity_of_not_measurable _ hf]
    /-
      case neg
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      inst✝ : Countable ι
      κ : ι → ProbabilityTheory.Kernel α β
      hκ : ∀ (i : ι), ProbabilityTheory.IsSFiniteKernel (κ i)
      f : α → β → ENNReal
      hf : Not (Measurable (Function.uncurry f))
      ⊢ Eq 0 (ProbabilityTheory.Kernel.sum fun i => 0)
    -/
    exact sum_zero.symm
    /-
      🎉 no goals
    -/


lemma withDensity_add_right [IsSFiniteKernel κ] {f g : α → β → ℝ≥0∞}
    (hf : Measurable (Function.uncurry f)) (hg : Measurable (Function.uncurry g)) :
    withDensity κ (f + g) = withDensity κ f + withDensity κ g := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    f g : α → β → ENNReal
    hf : Measurable (Function.uncurry f)
    hg : Measurable (Function.uncurry g)
    ⊢ Eq (κ.withDensity (HAdd.hAdd f g)) (HAdd.hAdd (κ.withDensity f) (κ.withDensi …
  -/
  ext a
  rw [coe_add, Pi.add_apply, Kernel.withDensity_apply _ hf, Kernel.withDensity_apply _ hg,
    Kernel.withDensity_apply, Pi.add_apply, MeasureTheory.withDensity_add_right]
    /-
      case h.h.hg
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α β
      inst✝ : ProbabilityTheory.IsSFiniteKernel κ
      f g : α → β → ENNReal
      hf : Measurable (Function.uncurry f)
      hg : Measurable (Function.uncurry g)
      a : α
      s✝ : Set β
      a✝ : MeasurableSet s✝
      ⊢ Measurable (g a)
    -/
  · exact hg.comp measurable_prod_mk_left
    /-
      🎉 no goals
    -/
    /-
      case h.h.hf
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α β
      inst✝ : ProbabilityTheory.IsSFiniteKernel κ
      f g : α → β → ENNReal
      hf : Measurable (Function.uncurry f)
      hg : Measurable (Function.uncurry g)
      a : α
      s✝ : Set β
      a✝ : MeasurableSet s✝
      ⊢ Measurable (Function.uncurry (HAdd.hAdd f g))
    -/
  · exact hf.add hg
    /-
      🎉 no goals
    -/


lemma withDensity_sub_add_cancel [IsSFiniteKernel κ] {f g : α → β → ℝ≥0∞}
    (hf : Measurable (Function.uncurry f)) (hg : Measurable (Function.uncurry g))
    (hfg : ∀ a, g a ≤ᵐ[κ a] f a) :
    withDensity κ (fun a x ↦ f a x - g a x) + withDensity κ g = withDensity κ f := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    f g : α → β → ENNReal
    hf : Measurable (Function.uncurry f)
    hg : Measurable (Function.uncurry g)
    hfg : ∀ (a : α), (MeasureTheory.ae (κ a)).EventuallyLE (g a) (f a)
    ⊢ Eq (HAdd.hAdd (κ.withDensity fun a x => HSub.hSub (f a x) (g a x)) (κ.withDe …
  -/
  rw [← withDensity_add_right _ hg]
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    f g : α → β → ENNReal
    hf : Measurable (Function.uncurry f)
    hg : Measurable (Function.uncurry g)
    hfg : ∀ (a : α), (MeasureTheory.ae (κ a)).EventuallyLE (g a) (f a)
    ⊢ Eq (κ.withDensity (HAdd.hAdd (fun a x => HSub.hSub (f a x) (g a x)) g)) (κ.w …
  -/
  swap; · exact hf.sub hg
          /-
            🎉 no goals
          -/
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    f g : α → β → ENNReal
    hf : Measurable (Function.uncurry f)
    hg : Measurable (Function.uncurry g)
    hfg : ∀ (a : α), (MeasureTheory.ae (κ a)).EventuallyLE (g a) (f a)
    ⊢ Eq (κ.withDensity (HAdd.hAdd (fun a x => HSub.hSub (f a x) (g a x)) g)) (κ.w …
  -/
  refine withDensity_congr_ae κ ((hf.sub hg).add hg) hf (fun a ↦ ?_)
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    f g : α → β → ENNReal
    hf : Measurable (Function.uncurry f)
    hg : Measurable (Function.uncurry g)
    hfg : ∀ (a : α), (MeasureTheory.ae (κ a)).EventuallyLE (g a) (f a)
    a : α
    ⊢ (MeasureTheory.ae (κ a)).EventuallyEq (HAdd.hAdd (fun a x => HSub.hSub (f a  …
  -/
  filter_upwards [hfg a] with x hx
  /-
    case h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    f g : α → β → ENNReal
    hf : Measurable (Function.uncurry f)
    hg : Measurable (Function.uncurry g)
    hfg : ∀ (a : α), (MeasureTheory.ae (κ a)).EventuallyLE (g a) (f a)
    a : α
    x : β
    hx : LE.le (g a x) (f a x)
    ⊢ Eq (HAdd.hAdd (fun a x => HSub.hSub (f a x) (g a x)) g a x) (f a x)
  -/
  rwa [Pi.add_apply, Pi.add_apply, tsub_add_cancel_iff_le]
  /-
    🎉 no goals
  -/


theorem withDensity_tsum [Countable ι] (κ : Kernel α β) [IsSFiniteKernel κ] {f : ι → α → β → ℝ≥0∞}
    (hf : ∀ i, Measurable (Function.uncurry (f i))) :
    withDensity κ (∑' n, f n) = Kernel.sum fun n => withDensity κ (f n) := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝¹ : Countable ι
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    f : ι → α → β → ENNReal
    hf : ∀ (i : ι), Measurable (Function.uncurry (f i))
    ⊢ Eq (κ.withDensity (tsum fun n => f n)) (ProbabilityTheory.Kernel.sum fun n = …
  -/
  have h_sum_a : ∀ a, Summable fun n => f n a := fun a => Pi.summable.mpr fun b => ENNReal.summable
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝¹ : Countable ι
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    f : ι → α → β → ENNReal
    hf : ∀ (i : ι), Measurable (Function.uncurry (f i))
    h_sum_a : ∀ (a : α), Summable fun n => f n a
    ⊢ Eq (κ.withDensity (tsum fun n => f n)) (ProbabilityTheory.Kernel.sum fun n = …
  -/
  have h_sum : Summable fun n => f n := Pi.summable.mpr h_sum_a
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝¹ : Countable ι
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    f : ι → α → β → ENNReal
    hf : ∀ (i : ι), Measurable (Function.uncurry (f i))
    h_sum_a : ∀ (a : α), Summable fun n => f n a
    h_sum : Summable fun n => f n
    ⊢ Eq (κ.withDensity (tsum fun n => f n)) (ProbabilityTheory.Kernel.sum fun n = …
  -/
  ext a s hs
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝¹ : Countable ι
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    f : ι → α → β → ENNReal
    hf : ∀ (i : ι), Measurable (Function.uncurry (f i))
    h_sum_a : ∀ (a : α), Summable fun n => f n a
    h_sum : Summable fun n => f n
    a : α
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq (((κ.withDensity (tsum fun n => f n)) a) s) (((ProbabilityTheory.Kernel.s …
  -/
  rw [sum_apply' _ a hs, Kernel.withDensity_apply' κ _ a s]
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝¹ : Countable ι
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    f : ι → α → β → ENNReal
    hf : ∀ (i : ι), Measurable (Function.uncurry (f i))
    h_sum_a : ∀ (a : α), Summable fun n => f n a
    h_sum : Summable fun n => f n
    a : α
    s : Set β
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.lintegral ((κ a).restrict s) fun b => tsum (fun n => f n)  …
  -/
  swap
  · have : Function.uncurry (∑' n, f n) = ∑' n, Function.uncurry (f n) := by
      ext1 p
      simp only [Function.uncurry_def]
      rw [tsum_apply h_sum, tsum_apply (h_sum_a _), tsum_apply]
      exact Pi.summable.mpr fun p => ENNReal.summable
    /-
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      inst✝¹ : Countable ι
      κ : ProbabilityTheory.Kernel α β
      inst✝ : ProbabilityTheory.IsSFiniteKernel κ
      f : ι → α → β → ENNReal
      hf : ∀ (i : ι), Measurable (Function.uncurry (f i))
      h_sum_a : ∀ (a : α), Summable fun n => f n a
      h_sum : Summable fun n => f n
      a : α
      s : Set β
      hs : MeasurableSet s
      this : Eq (Function.uncurry (tsum fun n => f n)) (tsum fun n => Function.uncur …
      ⊢ Measurable (Function.uncurry (tsum fun n => f n))
    -/
    rw [this]
    /-
      α : Type u_1
      β : Type u_2
      ι : Type u_3
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      inst✝¹ : Countable ι
      κ : ProbabilityTheory.Kernel α β
      inst✝ : ProbabilityTheory.IsSFiniteKernel κ
      f : ι → α → β → ENNReal
      hf : ∀ (i : ι), Measurable (Function.uncurry (f i))
      h_sum_a : ∀ (a : α), Summable fun n => f n a
      h_sum : Summable fun n => f n
      a : α
      s : Set β
      hs : MeasurableSet s
      this : Eq (Function.uncurry (tsum fun n => f n)) (tsum fun n => Function.uncur …
      ⊢ Measurable (tsum fun n => Function.uncurry (f n))
    -/
    exact Measurable.ennreal_tsum' hf
    /-
      🎉 no goals
    -/
  have : ∫⁻ b in s, (∑' n, f n) a b ∂κ a = ∫⁻ b in s, ∑' n, (fun b => f n a b) b ∂κ a := by
    congr with b
    rw [tsum_apply h_sum, tsum_apply (h_sum_a a)]
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝¹ : Countable ι
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    f : ι → α → β → ENNReal
    hf : ∀ (i : ι), Measurable (Function.uncurry (f i))
    h_sum_a : ∀ (a : α), Summable fun n => f n a
    h_sum : Summable fun n => f n
    a : α
    s : Set β
    hs : MeasurableSet s
    this : Eq (MeasureTheory.lintegral ((κ a).restrict s) fun b => tsum (fun n =>  …
    ⊢ Eq (MeasureTheory.lintegral ((κ a).restrict s) fun b => tsum (fun n => f n)  …
  -/
  rw [this, lintegral_tsum fun n => (Measurable.of_uncurry_left (hf n)).aemeasurable]
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝¹ : Countable ι
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    f : ι → α → β → ENNReal
    hf : ∀ (i : ι), Measurable (Function.uncurry (f i))
    h_sum_a : ∀ (a : α), Summable fun n => f n a
    h_sum : Summable fun n => f n
    a : α
    s : Set β
    hs : MeasurableSet s
    this : Eq (MeasureTheory.lintegral ((κ a).restrict s) fun b => tsum (fun n =>  …
    ⊢ Eq (tsum fun i => MeasureTheory.lintegral ((κ a).restrict s) fun a_1 => f i  …
  -/
  congr with n
  /-
    case h.h.e_f.h
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝¹ : Countable ι
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    f : ι → α → β → ENNReal
    hf : ∀ (i : ι), Measurable (Function.uncurry (f i))
    h_sum_a : ∀ (a : α), Summable fun n => f n a
    h_sum : Summable fun n => f n
    a : α
    s : Set β
    hs : MeasurableSet s
    this : Eq (MeasureTheory.lintegral ((κ a).restrict s) fun b => tsum (fun n =>  …
    n : ι
    ⊢ Eq (MeasureTheory.lintegral ((κ a).restrict s) fun a_1 => f n a a_1) (((κ.wi …
  -/
  rw [Kernel.withDensity_apply' _ (hf n) a s]
  /-
    🎉 no goals
  -/


/-- If a kernel `κ` is finite and a function `f : α → β → ℝ≥0∞` is bounded, then `withDensity κ f`
is finite. -/
theorem isFiniteKernel_withDensity_of_bounded (κ : Kernel α β) [IsFiniteKernel κ] {B : ℝ≥0∞}
    (hB_top : B ≠ ∞) (hf_B : ∀ a b, f a b ≤ B) : IsFiniteKernel (withDensity κ f) := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    f : α → β → ENNReal
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    B : ENNReal
    hB_top : Ne B Top.top
    hf_B : ∀ (a : α) (b : β), LE.le (f a b) B
    ⊢ ProbabilityTheory.IsFiniteKernel (κ.withDensity f)
  -/
  by_cases hf : Measurable (Function.uncurry f)
  · exact ⟨⟨B * IsFiniteKernel.bound κ, ENNReal.mul_lt_top hB_top.lt_top
      (IsFiniteKernel.bound_lt_top κ), fun a => by
        rw [Kernel.withDensity_apply' κ hf a Set.univ]
        calc
          ∫⁻ b in Set.univ, f a b ∂κ a ≤ ∫⁻ _ in Set.univ, B ∂κ a := lintegral_mono (hf_B a)
          _ = B * κ a Set.univ := by
            simp only [Measure.restrict_univ, MeasureTheory.lintegral_const]
          _ ≤ B * IsFiniteKernel.bound κ := mul_le_mul_left' (measure_le_bound κ a Set.univ) _⟩⟩
    /-
      case neg
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      f : α → β → ENNReal
      κ : ProbabilityTheory.Kernel α β
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      B : ENNReal
      hB_top : Ne B Top.top
      hf_B : ∀ (a : α) (b : β), LE.le (f a b) B
      hf : Not (Measurable (Function.uncurry f))
      ⊢ ProbabilityTheory.IsFiniteKernel (κ.withDensity f)
    -/
  · rw [withDensity_of_not_measurable _ hf]
    /-
      case neg
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      f : α → β → ENNReal
      κ : ProbabilityTheory.Kernel α β
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      B : ENNReal
      hB_top : Ne B Top.top
      hf_B : ∀ (a : α) (b : β), LE.le (f a b) B
      hf : Not (Measurable (Function.uncurry f))
      ⊢ ProbabilityTheory.IsFiniteKernel 0
    -/
    infer_instance
    /-
      🎉 no goals
    -/


/-- Auxiliary lemma for `IsSFiniteKernel.withDensity`.
If a kernel `κ` is finite, then `withDensity κ f` is s-finite. -/
theorem isSFiniteKernel_withDensity_of_isFiniteKernel (κ : Kernel α β) [IsFiniteKernel κ]
    (hf_ne_top : ∀ a b, f a b ≠ ∞) : IsSFiniteKernel (withDensity κ f) := by
  -- We already have that for `f` bounded from above and a `κ` a finite kernel,
  -- `withDensity κ f` is finite. We write any function as a countable sum of bounded
  -- functions, and decompose an s-finite kernel as a sum of finite kernels. We then use that
  -- `withDensity` commutes with sums for both arguments and get a sum of finite kernels.
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    f : α → β → ENNReal
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hf_ne_top : ∀ (a : α) (b : β), Ne (f a b) Top.top
    ⊢ ProbabilityTheory.IsSFiniteKernel (κ.withDensity f)
  -/
  by_cases hf : Measurable (Function.uncurry f)
  /-
    case pos
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    f : α → β → ENNReal
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hf_ne_top : ∀ (a : α) (b : β), Ne (f a b) Top.top
    hf : Measurable (Function.uncurry f)
    ⊢ ProbabilityTheory.IsSFiniteKernel (κ.withDensity f)
  -/
  swap; · rw [withDensity_of_not_measurable _ hf]; infer_instance
                                                   /-
                                                     🎉 no goals
                                                   -/
  /-
    case pos
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    f : α → β → ENNReal
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hf_ne_top : ∀ (a : α) (b : β), Ne (f a b) Top.top
    hf : Measurable (Function.uncurry f)
    ⊢ ProbabilityTheory.IsSFiniteKernel (κ.withDensity f)
  -/
  let fs : ℕ → α → β → ℝ≥0∞ := fun n a b => min (f a b) (n + 1) - min (f a b) n
  have h_le : ∀ a b n, ⌈(f a b).toReal⌉₊ ≤ n → f a b ≤ n := by
    intro a b n hn
    have : (f a b).toReal ≤ n := Nat.le_of_ceil_le hn
    rw [← ENNReal.le_ofReal_iff_toReal_le (hf_ne_top a b) _] at this
    · refine this.trans (le_of_eq ?_)
      rw [ENNReal.ofReal_natCast]
    · norm_cast
      exact zero_le _
  have h_zero : ∀ a b n, ⌈(f a b).toReal⌉₊ ≤ n → fs n a b = 0 := by
    intro a b n hn
    suffices min (f a b) (n + 1) = f a b ∧ min (f a b) n = f a b by
      simp_rw [fs, this.1, this.2, tsub_self (f a b)]
    exact ⟨min_eq_left ((h_le a b n hn).trans (le_add_of_nonneg_right zero_le_one)),
      min_eq_left (h_le a b n hn)⟩
  have hf_eq_tsum : f = ∑' n, fs n := by
    have h_sum_a : ∀ a, Summable fun n => fs n a := by
      refine fun a => Pi.summable.mpr fun b => ?_
      suffices ∀ n, n ∉ Finset.range ⌈(f a b).toReal⌉₊ → fs n a b = 0 from
        summable_of_ne_finset_zero this
      intro n hn_not_mem
      rw [Finset.mem_range, not_lt] at hn_not_mem
      exact h_zero a b n hn_not_mem
    ext a b : 2
    rw [tsum_apply (Pi.summable.mpr h_sum_a), tsum_apply (h_sum_a a),
      ENNReal.tsum_eq_liminf_sum_nat]
    have h_finset_sum : ∀ n, ∑ i ∈ Finset.range n, fs i a b = min (f a b) n := by
      intro n
      induction' n with n hn
      · simp
      rw [Finset.sum_range_succ, hn]
      simp [fs]
    simp_rw [h_finset_sum]
    refine (Filter.Tendsto.liminf_eq ?_).symm
    refine Filter.Tendsto.congr' ?_ tendsto_const_nhds
    rw [Filter.EventuallyEq, Filter.eventually_atTop]
    exact ⟨⌈(f a b).toReal⌉₊, fun n hn => (min_eq_left (h_le a b n hn)).symm⟩
  /-
    case pos
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    f : α → β → ENNReal
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hf_ne_top : ∀ (a : α) (b : β), Ne (f a b) Top.top
    hf : Measurable (Function.uncurry f)
    fs : Nat → α → β → ENNReal := fun n a b => HSub.hSub (Min.min (f a b) (HAdd.hA …
    h_le : ∀ (a : α) (b : β) (n : Nat), LE.le (Nat.ceil (f a b).toReal) n → LE.le  …
    h_zero : ∀ (a : α) (b : β) (n : Nat), LE.le (Nat.ceil (f a b).toReal) n → Eq ( …
    hf_eq_tsum : Eq f (tsum fun n => fs n)
    ⊢ ProbabilityTheory.IsSFiniteKernel (κ.withDensity f)
  -/
  rw [hf_eq_tsum, withDensity_tsum _ fun n : ℕ => _]
  /-
    case pos
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    f : α → β → ENNReal
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hf_ne_top : ∀ (a : α) (b : β), Ne (f a b) Top.top
    hf : Measurable (Function.uncurry f)
    fs : Nat → α → β → ENNReal := fun n a b => HSub.hSub (Min.min (f a b) (HAdd.hA …
    h_le : ∀ (a : α) (b : β) (n : Nat), LE.le (Nat.ceil (f a b).toReal) n → LE.le  …
    h_zero : ∀ (a : α) (b : β) (n : Nat), LE.le (Nat.ceil (f a b).toReal) n → Eq ( …
    hf_eq_tsum : Eq f (tsum fun n => fs n)
    ⊢ ProbabilityTheory.IsSFiniteKernel (ProbabilityTheory.Kernel.sum fun n => κ.w …
  -/
  swap; · exact fun _ => (hf.min measurable_const).sub (hf.min measurable_const)
          /-
            🎉 no goals
          -/
  /-
    case pos
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    f : α → β → ENNReal
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hf_ne_top : ∀ (a : α) (b : β), Ne (f a b) Top.top
    hf : Measurable (Function.uncurry f)
    fs : Nat → α → β → ENNReal := fun n a b => HSub.hSub (Min.min (f a b) (HAdd.hA …
    h_le : ∀ (a : α) (b : β) (n : Nat), LE.le (Nat.ceil (f a b).toReal) n → LE.le  …
    h_zero : ∀ (a : α) (b : β) (n : Nat), LE.le (Nat.ceil (f a b).toReal) n → Eq ( …
    hf_eq_tsum : Eq f (tsum fun n => fs n)
    ⊢ ProbabilityTheory.IsSFiniteKernel (ProbabilityTheory.Kernel.sum fun n => κ.w …
  -/
  refine isSFiniteKernel_sum fun n => ?_
  /-
    case pos
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    f : α → β → ENNReal
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hf_ne_top : ∀ (a : α) (b : β), Ne (f a b) Top.top
    hf : Measurable (Function.uncurry f)
    fs : Nat → α → β → ENNReal := fun n a b => HSub.hSub (Min.min (f a b) (HAdd.hA …
    h_le : ∀ (a : α) (b : β) (n : Nat), LE.le (Nat.ceil (f a b).toReal) n → LE.le  …
    h_zero : ∀ (a : α) (b : β) (n : Nat), LE.le (Nat.ceil (f a b).toReal) n → Eq ( …
    hf_eq_tsum : Eq f (tsum fun n => fs n)
    n : Nat
    ⊢ ProbabilityTheory.IsSFiniteKernel (κ.withDensity (fs n))
  -/
  suffices IsFiniteKernel (withDensity κ (fs n)) by haveI := this; infer_instance
  /-
    case pos
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    f : α → β → ENNReal
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hf_ne_top : ∀ (a : α) (b : β), Ne (f a b) Top.top
    hf : Measurable (Function.uncurry f)
    fs : Nat → α → β → ENNReal := fun n a b => HSub.hSub (Min.min (f a b) (HAdd.hA …
    h_le : ∀ (a : α) (b : β) (n : Nat), LE.le (Nat.ceil (f a b).toReal) n → LE.le  …
    h_zero : ∀ (a : α) (b : β) (n : Nat), LE.le (Nat.ceil (f a b).toReal) n → Eq ( …
    hf_eq_tsum : Eq f (tsum fun n => fs n)
    n : Nat
    ⊢ ProbabilityTheory.IsFiniteKernel (κ.withDensity (fs n))
  -/
  refine isFiniteKernel_withDensity_of_bounded _ (ENNReal.coe_ne_top : ↑n + 1 ≠ ∞) fun a b => ?_
  -- After https://github.com/leanprover/lean4/pull/2734, we need to do beta reduction before `norm_cast`
  /-
    case pos
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    f : α → β → ENNReal
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hf_ne_top : ∀ (a : α) (b : β), Ne (f a b) Top.top
    hf : Measurable (Function.uncurry f)
    fs : Nat → α → β → ENNReal := fun n a b => HSub.hSub (Min.min (f a b) (HAdd.hA …
    h_le : ∀ (a : α) (b : β) (n : Nat), LE.le (Nat.ceil (f a b).toReal) n → LE.le  …
    h_zero : ∀ (a : α) (b : β) (n : Nat), LE.le (Nat.ceil (f a b).toReal) n → Eq ( …
    hf_eq_tsum : Eq f (tsum fun n => fs n)
    n : Nat
    a : α
    b : β
    ⊢ LE.le (fs n a b) ↑((fun x1 x2 => HAdd.hAdd x1 x2) (↑n) 1)
  -/
  beta_reduce
  /-
    case pos
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    f : α → β → ENNReal
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    hf_ne_top : ∀ (a : α) (b : β), Ne (f a b) Top.top
    hf : Measurable (Function.uncurry f)
    fs : Nat → α → β → ENNReal := fun n a b => HSub.hSub (Min.min (f a b) (HAdd.hA …
    h_le : ∀ (a : α) (b : β) (n : Nat), LE.le (Nat.ceil (f a b).toReal) n → LE.le  …
    h_zero : ∀ (a : α) (b : β) (n : Nat), LE.le (Nat.ceil (f a b).toReal) n → Eq ( …
    hf_eq_tsum : Eq f (tsum fun n => fs n)
    n : Nat
    a : α
    b : β
    ⊢ LE.le (fs n a b) ↑(HAdd.hAdd (↑n) 1)
  -/
  norm_cast
  calc
    fs n a b ≤ min (f a b) (n + 1) := tsub_le_self
    _ ≤ n + 1 := min_le_right _ _
    _ = ↑(n + 1) := by norm_cast


/-- For an s-finite kernel `κ` and a function `f : α → β → ℝ≥0∞` which is everywhere finite,
`withDensity κ f` is s-finite. -/
nonrec theorem IsSFiniteKernel.withDensity (κ : Kernel α β) [IsSFiniteKernel κ]
    (hf_ne_top : ∀ a b, f a b ≠ ∞) : IsSFiniteKernel (withDensity κ f) := by
  have h_eq_sum : withDensity κ f = Kernel.sum fun i => withDensity (seq κ i) f := by
    rw [← withDensity_kernel_sum _ _]
    congr
    exact (kernel_sum_seq κ).symm
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    f : α → β → ENNReal
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    hf_ne_top : ∀ (a : α) (b : β), Ne (f a b) Top.top
    h_eq_sum : Eq (κ.withDensity f) (ProbabilityTheory.Kernel.sum fun i => (κ.seq  …
    ⊢ ProbabilityTheory.IsSFiniteKernel (κ.withDensity f)
  -/
  rw [h_eq_sum]
  exact isSFiniteKernel_sum fun n =>
    isSFiniteKernel_withDensity_of_isFiniteKernel (seq κ n) hf_ne_top


/-- For an s-finite kernel `κ` and a function `f : α → β → ℝ≥0`, `withDensity κ f` is s-finite. -/
instance (κ : Kernel α β) [IsSFiniteKernel κ] (f : α → β → ℝ≥0) :
    IsSFiniteKernel (withDensity κ fun a b => f a b) :=
  IsSFiniteKernel.withDensity κ fun _ _ => ENNReal.coe_ne_top


nonrec lemma withDensity_mul [IsSFiniteKernel κ] {f : α → β → ℝ≥0} {g : α → β → ℝ≥0∞}
    (hf : Measurable (Function.uncurry f)) (hg : Measurable (Function.uncurry g)) :
    withDensity κ (fun a x ↦ f a x * g a x)
      = withDensity (withDensity κ fun a x ↦ f a x) g := by
  /-
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    f : α → β → NNReal
    g : α → β → ENNReal
    hf : Measurable (Function.uncurry f)
    hg : Measurable (Function.uncurry g)
    ⊢ Eq (κ.withDensity fun a x => HMul.hMul (↑(f a x)) (g a x)) ((κ.withDensity f …
  -/
  ext a : 1
  /-
    case h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    f : α → β → NNReal
    g : α → β → ENNReal
    hf : Measurable (Function.uncurry f)
    hg : Measurable (Function.uncurry g)
    a : α
    ⊢ Eq ((κ.withDensity fun a x => HMul.hMul (↑(f a x)) (g a x)) a) (((κ.withDens …
  -/
  rw [Kernel.withDensity_apply]
  /-
    case h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    f : α → β → NNReal
    g : α → β → ENNReal
    hf : Measurable (Function.uncurry f)
    hg : Measurable (Function.uncurry g)
    a : α
    ⊢ Eq ((κ a).withDensity fun x => HMul.hMul (↑(f a x)) (g a x)) (((κ.withDensit …
  -/
  swap; · exact (measurable_coe_nnreal_ennreal.comp hf).mul hg
          /-
            🎉 no goals
          -/
  change (Measure.withDensity (κ a) ((fun x ↦ (f a x : ℝ≥0∞)) * (fun x ↦ (g a x : ℝ≥0∞)))) =
      (withDensity (withDensity κ fun a x ↦ f a x) g) a
  /-
    case h
    α : Type u_1
    β : Type u_2
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    κ : ProbabilityTheory.Kernel α β
    inst✝ : ProbabilityTheory.IsSFiniteKernel κ
    f : α → β → NNReal
    g : α → β → ENNReal
    hf : Measurable (Function.uncurry f)
    hg : Measurable (Function.uncurry g)
    a : α
    ⊢ Eq ((κ a).withDensity (HMul.hMul (fun x => ↑(f a x)) fun x => g a x)) (((κ.w …
  -/
  rw [withDensity_mul]
    /-
      case h
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α β
      inst✝ : ProbabilityTheory.IsSFiniteKernel κ
      f : α → β → NNReal
      g : α → β → ENNReal
      hf : Measurable (Function.uncurry f)
      hg : Measurable (Function.uncurry g)
      a : α
      ⊢ Eq (((κ a).withDensity fun x => ↑(f a x)).withDensity fun x => g a x) (((κ.w …
    -/
  · rw [Kernel.withDensity_apply _ hg, Kernel.withDensity_apply]
    /-
      case h.hf
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α β
      inst✝ : ProbabilityTheory.IsSFiniteKernel κ
      f : α → β → NNReal
      g : α → β → ENNReal
      hf : Measurable (Function.uncurry f)
      hg : Measurable (Function.uncurry g)
      a : α
      ⊢ Measurable (Function.uncurry fun a x => ↑(f a x))
    -/
    exact measurable_coe_nnreal_ennreal.comp hf
    /-
      🎉 no goals
    -/
    /-
      case h.hf
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α β
      inst✝ : ProbabilityTheory.IsSFiniteKernel κ
      f : α → β → NNReal
      g : α → β → ENNReal
      hf : Measurable (Function.uncurry f)
      hg : Measurable (Function.uncurry g)
      a : α
      ⊢ Measurable fun x => ↑(f a x)
    -/
  · rw [measurable_coe_nnreal_ennreal_iff]
    /-
      case h.hf
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α β
      inst✝ : ProbabilityTheory.IsSFiniteKernel κ
      f : α → β → NNReal
      g : α → β → ENNReal
      hf : Measurable (Function.uncurry f)
      hg : Measurable (Function.uncurry g)
      a : α
      ⊢ Measurable (f a)
    -/
    exact hf.comp measurable_prod_mk_left
    /-
      🎉 no goals
    -/
    /-
      case h.hg
      α : Type u_1
      β : Type u_2
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      κ : ProbabilityTheory.Kernel α β
      inst✝ : ProbabilityTheory.IsSFiniteKernel κ
      f : α → β → NNReal
      g : α → β → ENNReal
      hf : Measurable (Function.uncurry f)
      hg : Measurable (Function.uncurry g)
      a : α
      ⊢ Measurable fun x => g a x
    -/
  · exact hg.comp measurable_prod_mk_left
    /-
      🎉 no goals
    -/


