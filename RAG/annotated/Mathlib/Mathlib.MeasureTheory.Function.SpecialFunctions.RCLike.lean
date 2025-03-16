@[measurability]
theorem measurable_re : Measurable (re : 𝕜 → ℝ) :=
  continuous_re.measurable


@[measurability]
theorem measurable_im : Measurable (im : 𝕜 → ℝ) :=
  continuous_im.measurable


@[measurability]
theorem Measurable.re (hf : Measurable f) : Measurable fun x => RCLike.re (f x) :=
  RCLike.measurable_re.comp hf


@[measurability]
theorem AEMeasurable.re (hf : AEMeasurable f μ) : AEMeasurable (fun x => RCLike.re (f x)) μ :=
  RCLike.measurable_re.comp_aemeasurable hf


@[measurability]
theorem Measurable.im (hf : Measurable f) : Measurable fun x => RCLike.im (f x) :=
  RCLike.measurable_im.comp hf


@[measurability]
theorem AEMeasurable.im (hf : AEMeasurable f μ) : AEMeasurable (fun x => RCLike.im (f x)) μ :=
  RCLike.measurable_im.comp_aemeasurable hf


@[measurability]
theorem RCLike.measurable_ofReal : Measurable ((↑) : ℝ → 𝕜) :=
  RCLike.continuous_ofReal.measurable


theorem measurable_of_re_im (hre : Measurable fun x => RCLike.re (f x))
    (him : Measurable fun x => RCLike.im (f x)) : Measurable f := by
  convert Measurable.add (M := 𝕜) (RCLike.measurable_ofReal.comp hre)
      ((RCLike.measurable_ofReal.comp him).mul_const RCLike.I)
  /-
    case h.e'_5.h
    α : Type u_1
    𝕜 : Type u_2
    inst✝¹ : RCLike 𝕜
    inst✝ : MeasurableSpace α
    f : α → 𝕜
    hre : Measurable fun x => RCLike.re (f x)
    him : Measurable fun x => RCLike.im (f x)
    x✝ : α
    ⊢ Eq (f x✝) (HAdd.hAdd (Function.comp RCLike.ofReal (fun x => RCLike.re (f x)) …
  -/
  exact (RCLike.re_add_im _).symm
  /-
    🎉 no goals
  -/


theorem aemeasurable_of_re_im (hre : AEMeasurable (fun x => RCLike.re (f x)) μ)
    (him : AEMeasurable (fun x => RCLike.im (f x)) μ) : AEMeasurable f μ := by
  convert AEMeasurable.add (M := 𝕜) (RCLike.measurable_ofReal.comp_aemeasurable hre)
      ((RCLike.measurable_ofReal.comp_aemeasurable him).mul_const RCLike.I)
  /-
    case h.e'_5.h
    α : Type u_1
    𝕜 : Type u_2
    inst✝¹ : RCLike 𝕜
    inst✝ : MeasurableSpace α
    f : α → 𝕜
    μ : MeasureTheory.Measure α
    hre : AEMeasurable (fun x => RCLike.re (f x)) μ
    him : AEMeasurable (fun x => RCLike.im (f x)) μ
    x✝ : α
    ⊢ Eq (f x✝) (HAdd.hAdd (Function.comp RCLike.ofReal (fun x => RCLike.re (f x)) …
  -/
  exact (RCLike.re_add_im _).symm
  /-
    🎉 no goals
  -/


