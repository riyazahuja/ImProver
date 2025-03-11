theorem lintegral_nnnorm_eq_lintegral_edist (f : α → β) :
                                                   /-
                                                     α : Type u_1
                                                     β : Type u_2
                                                     m : MeasurableSpace α
                                                     μ : MeasureTheory.Measure α
                                                     inst✝ : NormedAddCommGroup β
                                                     f : α → β
                                                     ⊢ Eq (MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (f a))) (MeasureTheor …
                                                   -/
    ∫⁻ a, ‖f a‖₊ ∂μ = ∫⁻ a, edist (f a) 0 ∂μ := by simp only [edist_eq_coe_nnnorm]
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem lintegral_norm_eq_lintegral_edist (f : α → β) :
    ∫⁻ a, ENNReal.ofReal ‖f a‖ ∂μ = ∫⁻ a, edist (f a) 0 ∂μ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    ⊢ Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (Norm.norm (f a))) (Me …
  -/
  simp only [ofReal_norm_eq_coe_nnnorm, edist_eq_coe_nnnorm]
  /-
    🎉 no goals
  -/


theorem lintegral_edist_triangle {f g h : α → β} (hf : AEStronglyMeasurable f μ)
    (hh : AEStronglyMeasurable h μ) :
    (∫⁻ a, edist (f a) (g a) ∂μ) ≤ (∫⁻ a, edist (f a) (h a) ∂μ) + ∫⁻ a, edist (g a) (h a) ∂μ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f g h : α → β
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hh : MeasureTheory.AEStronglyMeasurable h μ
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => EDist.edist (f a) (g a)) (HAdd.hAd …
  -/
  rw [← lintegral_add_left' (hf.edist hh)]
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f g h : α → β
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hh : MeasureTheory.AEStronglyMeasurable h μ
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => EDist.edist (f a) (g a)) (MeasureT …
  -/
  refine lintegral_mono fun a => ?_
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f g h : α → β
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hh : MeasureTheory.AEStronglyMeasurable h μ
    a : α
    ⊢ LE.le (EDist.edist (f a) (g a)) (HAdd.hAdd (EDist.edist (f a) (h a)) (EDist. …
  -/
  apply edist_triangle_right
  /-
    🎉 no goals
  -/


                                                                    /-
                                                                      α : Type u_1
                                                                      β : Type u_2
                                                                      m : MeasurableSpace α
                                                                      μ : MeasureTheory.Measure α
                                                                      inst✝ : NormedAddCommGroup β
                                                                      ⊢ Eq (MeasureTheory.lintegral μ fun x => ↑(NNNorm.nnnorm 0)) 0
                                                                    -/
theorem lintegral_nnnorm_zero : (∫⁻ _ : α, ‖(0 : β)‖₊ ∂μ) = 0 := by simp
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem lintegral_nnnorm_add_left {f : α → β} (hf : AEStronglyMeasurable f μ) (g : α → γ) :
    ∫⁻ a, ‖f a‖₊ + ‖g a‖₊ ∂μ = (∫⁻ a, ‖f a‖₊ ∂μ) + ∫⁻ a, ‖g a‖₊ ∂μ :=
  lintegral_add_left' hf.ennnorm _


theorem lintegral_nnnorm_add_right (f : α → β) {g : α → γ} (hg : AEStronglyMeasurable g μ) :
    ∫⁻ a, ‖f a‖₊ + ‖g a‖₊ ∂μ = (∫⁻ a, ‖f a‖₊ ∂μ) + ∫⁻ a, ‖g a‖₊ ∂μ :=
  lintegral_add_right' _ hg.ennnorm


theorem lintegral_nnnorm_neg {f : α → β} : (∫⁻ a, ‖(-f) a‖₊ ∂μ) = ∫⁻ a, ‖f a‖₊ ∂μ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    ⊢ Eq (MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (Neg.neg f a))) (Meas …
  -/
  simp only [Pi.neg_apply, nnnorm_neg]
  /-
    🎉 no goals
  -/


/-- `HasFiniteIntegral f μ` means that the integral `∫⁻ a, ‖f a‖ ∂μ` is finite.
  `HasFiniteIntegral f` means `HasFiniteIntegral f volume`. -/
def HasFiniteIntegral {_ : MeasurableSpace α} (f : α → ε)
    (μ : Measure α := by volume_tac) : Prop :=
  (∫⁻ a, ‖f a‖ₑ ∂μ) < ∞


theorem hasFiniteIntegral_def {_ : MeasurableSpace α} (f : α → ε) (μ : Measure α) :
    HasFiniteIntegral f μ ↔ ((∫⁻ a, ‖f a‖ₑ ∂μ) < ∞) :=
  Iff.rfl


theorem hasFiniteIntegral_iff_nnnorm {f : α → β} :
    HasFiniteIntegral f μ ↔ (∫⁻ a, ‖f a‖₊ ∂μ) < ∞ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    ⊢ Iff (MeasureTheory.HasFiniteIntegral f μ) (LT.lt (MeasureTheory.lintegral μ  …
  -/
  simp only [HasFiniteIntegral, ofReal_norm_eq_coe_nnnorm, enorm_eq_nnnorm]
  /-
    🎉 no goals
  -/


theorem hasFiniteIntegral_iff_norm (f : α → β) :
    HasFiniteIntegral f μ ↔ (∫⁻ a, ENNReal.ofReal ‖f a‖ ∂μ) < ∞ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    ⊢ Iff (MeasureTheory.HasFiniteIntegral f μ) (LT.lt (MeasureTheory.lintegral μ  …
  -/
  simp only [hasFiniteIntegral_iff_nnnorm, ofReal_norm_eq_coe_nnnorm]
  /-
    🎉 no goals
  -/


theorem hasFiniteIntegral_iff_edist (f : α → β) :
    HasFiniteIntegral f μ ↔ (∫⁻ a, edist (f a) 0 ∂μ) < ∞ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    ⊢ Iff (MeasureTheory.HasFiniteIntegral f μ) (LT.lt (MeasureTheory.lintegral μ  …
  -/
  simp only [hasFiniteIntegral_iff_norm, edist_dist, dist_zero_right]
  /-
    🎉 no goals
  -/


theorem hasFiniteIntegral_iff_ofReal {f : α → ℝ} (h : 0 ≤ᵐ[μ] f) :
    HasFiniteIntegral f μ ↔ (∫⁻ a, ENNReal.ofReal (f a) ∂μ) < ∞ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    h : (MeasureTheory.ae μ).EventuallyLE 0 f
    ⊢ Iff (MeasureTheory.HasFiniteIntegral f μ) (LT.lt (MeasureTheory.lintegral μ  …
  -/
  rw [hasFiniteIntegral_iff_nnnorm, lintegral_nnnorm_eq_of_ae_nonneg h]
  /-
    🎉 no goals
  -/


theorem hasFiniteIntegral_iff_ofNNReal {f : α → ℝ≥0} :
    HasFiniteIntegral (fun x => (f x : ℝ)) μ ↔ (∫⁻ a, f a ∂μ) < ∞ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → NNReal
    ⊢ Iff (MeasureTheory.HasFiniteIntegral (fun x => ↑(f x)) μ) (LT.lt (MeasureThe …
  -/
  simp [hasFiniteIntegral_iff_norm]
  /-
    🎉 no goals
  -/


theorem HasFiniteIntegral.mono {f : α → β} {g : α → γ} (hg : HasFiniteIntegral g μ)
    (h : ∀ᵐ a ∂μ, ‖f a‖ ≤ ‖g a‖) : HasFiniteIntegral f μ := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    inst✝ : NormedAddCommGroup γ
    f : α → β
    g : α → γ
    hg : MeasureTheory.HasFiniteIntegral g μ
    h : Filter.Eventually (fun a => LE.le (Norm.norm (f a)) (Norm.norm (g a))) (Me …
    ⊢ MeasureTheory.HasFiniteIntegral f μ
  -/
  simp only [hasFiniteIntegral_iff_norm] at *
  calc
    (∫⁻ a, ENNReal.ofReal ‖f a‖ ∂μ) ≤ ∫⁻ a : α, ENNReal.ofReal ‖g a‖ ∂μ :=
      lintegral_mono_ae (h.mono fun a h => ofReal_le_ofReal h)
    _ < ∞ := hg


theorem HasFiniteIntegral.mono' {f : α → β} {g : α → ℝ} (hg : HasFiniteIntegral g μ)
    (h : ∀ᵐ a ∂μ, ‖f a‖ ≤ g a) : HasFiniteIntegral f μ :=
  hg.mono <| h.mono fun _x hx => le_trans hx (le_abs_self _)


theorem HasFiniteIntegral.congr' {f : α → β} {g : α → γ} (hf : HasFiniteIntegral f μ)
    (h : ∀ᵐ a ∂μ, ‖f a‖ = ‖g a‖) : HasFiniteIntegral g μ :=
  hf.mono <| EventuallyEq.le <| EventuallyEq.symm h


theorem hasFiniteIntegral_congr' {f : α → β} {g : α → γ} (h : ∀ᵐ a ∂μ, ‖f a‖ = ‖g a‖) :
    HasFiniteIntegral f μ ↔ HasFiniteIntegral g μ :=
  ⟨fun hf => hf.congr' h, fun hg => hg.congr' <| EventuallyEq.symm h⟩


theorem HasFiniteIntegral.congr {f g : α → β} (hf : HasFiniteIntegral f μ) (h : f =ᵐ[μ] g) :
    HasFiniteIntegral g μ :=
  hf.congr' <| h.fun_comp norm


theorem hasFiniteIntegral_congr {f g : α → β} (h : f =ᵐ[μ] g) :
    HasFiniteIntegral f μ ↔ HasFiniteIntegral g μ :=
  hasFiniteIntegral_congr' <| h.fun_comp norm


theorem hasFiniteIntegral_const_iff {c : β} :
    HasFiniteIntegral (fun _ : α => c) μ ↔ c = 0 ∨ μ univ < ∞ := by
  simp [hasFiniteIntegral_iff_nnnorm, lintegral_const, lt_top_iff_ne_top, ENNReal.mul_eq_top,
    or_iff_not_imp_left]


theorem hasFiniteIntegral_const [IsFiniteMeasure μ] (c : β) :
    HasFiniteIntegral (fun _ : α => c) μ :=
  hasFiniteIntegral_const_iff.2 (Or.inr <| measure_lt_top _ _)


theorem HasFiniteIntegral.of_mem_Icc [IsFiniteMeasure μ] (a b : ℝ) {X : α → ℝ}
    (h : ∀ᵐ ω ∂μ, X ω ∈ Set.Icc a b) :
    HasFiniteIntegral X μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    a b : Real
    X : α → Real
    h : Filter.Eventually (fun ω => Membership.mem (Set.Icc a b) (X ω)) (MeasureTh …
    ⊢ MeasureTheory.HasFiniteIntegral X μ
  -/
  apply (hasFiniteIntegral_const (max ‖a‖ ‖b‖)).mono'
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    a b : Real
    X : α → Real
    h : Filter.Eventually (fun ω => Membership.mem (Set.Icc a b) (X ω)) (MeasureTh …
    ⊢ Filter.Eventually (fun a_1 => LE.le (Norm.norm (X a_1)) (Max.max (Norm.norm  …
  -/
  filter_upwards [h.mono fun ω h ↦ h.1, h.mono fun ω h ↦ h.2] with ω using abs_le_max_abs_abs
  /-
    🎉 no goals
  -/


theorem hasFiniteIntegral_of_bounded [IsFiniteMeasure μ] {f : α → β} {C : ℝ}
    (hC : ∀ᵐ a ∂μ, ‖f a‖ ≤ C) : HasFiniteIntegral f μ :=
  (hasFiniteIntegral_const C).mono' hC


theorem HasFiniteIntegral.of_finite [Finite α] [IsFiniteMeasure μ] {f : α → β} :
    HasFiniteIntegral f μ :=
  let ⟨_⟩ := nonempty_fintype α
  hasFiniteIntegral_of_bounded <| ae_of_all μ <| norm_le_pi_norm f


@[deprecated (since := "2024-02-05")]
alias hasFiniteIntegral_of_fintype := HasFiniteIntegral.of_finite


theorem HasFiniteIntegral.mono_measure {f : α → β} (h : HasFiniteIntegral f ν) (hμ : μ ≤ ν) :
    HasFiniteIntegral f μ :=
  lt_of_le_of_lt (lintegral_mono' hμ le_rfl) h


theorem HasFiniteIntegral.add_measure {f : α → β} (hμ : HasFiniteIntegral f μ)
    (hν : HasFiniteIntegral f ν) : HasFiniteIntegral f (μ + ν) := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    hμ : MeasureTheory.HasFiniteIntegral f μ
    hν : MeasureTheory.HasFiniteIntegral f ν
    ⊢ MeasureTheory.HasFiniteIntegral f (HAdd.hAdd μ ν)
  -/
  simp only [HasFiniteIntegral, lintegral_add_measure] at *
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    hμ : LT.lt (MeasureTheory.lintegral μ fun a => ENorm.enorm (f a)) Top.top
    hν : LT.lt (MeasureTheory.lintegral ν fun a => ENorm.enorm (f a)) Top.top
    ⊢ LT.lt (HAdd.hAdd (MeasureTheory.lintegral μ fun a => ENorm.enorm (f a)) (Mea …
  -/
  exact add_lt_top.2 ⟨hμ, hν⟩
  /-
    🎉 no goals
  -/


theorem HasFiniteIntegral.left_of_add_measure {f : α → β} (h : HasFiniteIntegral f (μ + ν)) :
    HasFiniteIntegral f μ :=
  h.mono_measure <| Measure.le_add_right <| le_rfl


theorem HasFiniteIntegral.right_of_add_measure {f : α → β} (h : HasFiniteIntegral f (μ + ν)) :
    HasFiniteIntegral f ν :=
  h.mono_measure <| Measure.le_add_left <| le_rfl


@[simp]
theorem hasFiniteIntegral_add_measure {f : α → β} :
    HasFiniteIntegral f (μ + ν) ↔ HasFiniteIntegral f μ ∧ HasFiniteIntegral f ν :=
  ⟨fun h => ⟨h.left_of_add_measure, h.right_of_add_measure⟩, fun h => h.1.add_measure h.2⟩


theorem HasFiniteIntegral.smul_measure {f : α → β} (h : HasFiniteIntegral f μ) {c : ℝ≥0∞}
    (hc : c ≠ ∞) : HasFiniteIntegral f (c • μ) := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    h : MeasureTheory.HasFiniteIntegral f μ
    c : ENNReal
    hc : Ne c Top.top
    ⊢ MeasureTheory.HasFiniteIntegral f (HSMul.hSMul c μ)
  -/
  simp only [HasFiniteIntegral, lintegral_smul_measure] at *
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    h : LT.lt (MeasureTheory.lintegral μ fun a => ENorm.enorm (f a)) Top.top
    c : ENNReal
    hc : Ne c Top.top
    ⊢ LT.lt (HMul.hMul c (MeasureTheory.lintegral μ fun a => ENorm.enorm (f a))) T …
  -/
  exact mul_lt_top hc.lt_top h
  /-
    🎉 no goals
  -/


@[simp]
theorem hasFiniteIntegral_zero_measure {m : MeasurableSpace α} (f : α → β) :
    HasFiniteIntegral f (0 : Measure α) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : NormedAddCommGroup β
    m : MeasurableSpace α
    f : α → β
    ⊢ MeasureTheory.HasFiniteIntegral f 0
  -/
  simp only [HasFiniteIntegral, lintegral_zero_measure, zero_lt_top]
  /-
    🎉 no goals
  -/


@[simp]
theorem hasFiniteIntegral_zero : HasFiniteIntegral (fun _ : α => (0 : β)) μ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    ⊢ MeasureTheory.HasFiniteIntegral (fun x => 0) μ
  -/
  simp [hasFiniteIntegral_iff_nnnorm]
  /-
    🎉 no goals
  -/


theorem HasFiniteIntegral.neg {f : α → β} (hfi : HasFiniteIntegral f μ) :
                                   /-
                                     α : Type u_1
                                     β : Type u_2
                                     m : MeasurableSpace α
                                     μ : MeasureTheory.Measure α
                                     inst✝ : NormedAddCommGroup β
                                     f : α → β
                                     hfi : MeasureTheory.HasFiniteIntegral f μ
                                     ⊢ MeasureTheory.HasFiniteIntegral (Neg.neg f) μ
                                   -/
    HasFiniteIntegral (-f) μ := by simpa [hasFiniteIntegral_iff_nnnorm] using hfi
                                   /-
                                     🎉 no goals
                                   -/


@[simp]
theorem hasFiniteIntegral_neg_iff {f : α → β} : HasFiniteIntegral (-f) μ ↔ HasFiniteIntegral f μ :=
  ⟨fun h => neg_neg f ▸ h.neg, HasFiniteIntegral.neg⟩


theorem HasFiniteIntegral.norm {f : α → β} (hfi : HasFiniteIntegral f μ) :
    HasFiniteIntegral (fun a => ‖f a‖) μ := by
  have eq : (fun a => (nnnorm ‖f a‖ : ℝ≥0∞)) = fun a => (‖f a‖₊ : ℝ≥0∞) := by
    funext
    rw [nnnorm_norm]
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    hfi : MeasureTheory.HasFiniteIntegral f μ
    eq : Eq (fun a => ↑(NNNorm.nnnorm (Norm.norm (f a)))) fun a => ↑(NNNorm.nnnorm …
    ⊢ MeasureTheory.HasFiniteIntegral (fun a => Norm.norm (f a)) μ
  -/
  rwa [hasFiniteIntegral_iff_nnnorm, eq]
  /-
    🎉 no goals
  -/


theorem hasFiniteIntegral_norm_iff (f : α → β) :
    HasFiniteIntegral (fun a => ‖f a‖) μ ↔ HasFiniteIntegral f μ :=
  hasFiniteIntegral_congr' <| Eventually.of_forall fun x => norm_norm (f x)


theorem hasFiniteIntegral_toReal_of_lintegral_ne_top {f : α → ℝ≥0∞} (hf : ∫⁻ x, f x ∂μ ≠ ∞) :
    HasFiniteIntegral (fun x ↦ (f x).toReal) μ := by
  have h x : (‖(f x).toReal‖₊ : ℝ≥0∞) = ENNReal.ofNNReal ⟨(f x).toReal, ENNReal.toReal_nonneg⟩ := by
    rw [Real.nnnorm_of_nonneg]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : Ne (MeasureTheory.lintegral μ fun x => f x) Top.top
    h : ∀ (x : α), Eq ↑(NNNorm.nnnorm (f x).toReal) ↑⟨(f x).toReal, ⋯⟩
    ⊢ MeasureTheory.HasFiniteIntegral (fun x => (f x).toReal) μ
  -/
  simp_rw [hasFiniteIntegral_iff_nnnorm, h]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : Ne (MeasureTheory.lintegral μ fun x => f x) Top.top
    h : ∀ (x : α), Eq ↑(NNNorm.nnnorm (f x).toReal) ↑⟨(f x).toReal, ⋯⟩
    ⊢ LT.lt (MeasureTheory.lintegral μ fun a => ↑⟨(f a).toReal, ⋯⟩) Top.top
  -/
  refine lt_of_le_of_lt (lintegral_mono fun x => ?_) (lt_top_iff_ne_top.2 hf)
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : Ne (MeasureTheory.lintegral μ fun x => f x) Top.top
    h : ∀ (x : α), Eq ↑(NNNorm.nnnorm (f x).toReal) ↑⟨(f x).toReal, ⋯⟩
    x : α
    ⊢ LE.le (↑⟨(f x).toReal, ⋯⟩) (f x)
  -/
  by_cases hfx : f x = ∞
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      hf : Ne (MeasureTheory.lintegral μ fun x => f x) Top.top
      h : ∀ (x : α), Eq ↑(NNNorm.nnnorm (f x).toReal) ↑⟨(f x).toReal, ⋯⟩
      x : α
      hfx : Eq (f x) Top.top
      ⊢ LE.le (↑⟨(f x).toReal, ⋯⟩) (f x)
    -/
  · simp [hfx]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      hf : Ne (MeasureTheory.lintegral μ fun x => f x) Top.top
      h : ∀ (x : α), Eq ↑(NNNorm.nnnorm (f x).toReal) ↑⟨(f x).toReal, ⋯⟩
      x : α
      hfx : Not (Eq (f x) Top.top)
      ⊢ LE.le (↑⟨(f x).toReal, ⋯⟩) (f x)
    -/
  · lift f x to ℝ≥0 using hfx with fx h
    /-
      case neg.intro
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → ENNReal
      hf : Ne (MeasureTheory.lintegral μ fun x => f x) Top.top
      h✝ : ∀ (x : α), Eq ↑(NNNorm.nnnorm (f x).toReal) ↑⟨(f x).toReal, ⋯⟩
      x : α
      fx : NNReal
      h : Eq (↑fx) (f x)
      ⊢ LE.le ↑⟨(↑fx).toReal, ⋯⟩ ↑fx
    -/
    simp [← h, ← NNReal.coe_le_coe]
    /-
      🎉 no goals
    -/


lemma hasFiniteIntegral_toReal_iff {f : α → ℝ≥0∞} (hf_ne_top : ∀ᵐ x ∂μ, f x ≠ ∞) :
    HasFiniteIntegral (fun x ↦ (f x).toReal) μ ↔ ∫⁻ x, f x ∂μ ≠ ∞ := by
  have h_eq x : (‖(f x).toReal‖₊ : ℝ≥0∞)
      = ENNReal.ofNNReal ⟨(f x).toReal, ENNReal.toReal_nonneg⟩ := by
    rw [Real.nnnorm_of_nonneg]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf_ne_top : Filter.Eventually (fun x => Ne (f x) Top.top) (MeasureTheory.ae μ)
    h_eq : ∀ (x : α), Eq ↑(NNNorm.nnnorm (f x).toReal) ↑⟨(f x).toReal, ⋯⟩
    ⊢ Iff (MeasureTheory.HasFiniteIntegral (fun x => (f x).toReal) μ) (Ne (Measure …
  -/
  simp_rw [hasFiniteIntegral_iff_nnnorm, h_eq, lt_top_iff_ne_top]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf_ne_top : Filter.Eventually (fun x => Ne (f x) Top.top) (MeasureTheory.ae μ)
    h_eq : ∀ (x : α), Eq ↑(NNNorm.nnnorm (f x).toReal) ↑⟨(f x).toReal, ⋯⟩
    ⊢ Iff (Ne (MeasureTheory.lintegral μ fun a => ↑⟨(f a).toReal, ⋯⟩) Top.top) (Ne …
  -/
  convert Iff.rfl using 2
  /-
    case h.e'_2.h.e'_2
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf_ne_top : Filter.Eventually (fun x => Ne (f x) Top.top) (MeasureTheory.ae μ)
    h_eq : ∀ (x : α), Eq ↑(NNNorm.nnnorm (f x).toReal) ↑⟨(f x).toReal, ⋯⟩
    ⊢ Eq (MeasureTheory.lintegral μ fun x => f x) (MeasureTheory.lintegral μ fun a …
  -/
  refine lintegral_congr_ae ?_
  /-
    case h.e'_2.h.e'_2
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf_ne_top : Filter.Eventually (fun x => Ne (f x) Top.top) (MeasureTheory.ae μ)
    h_eq : ∀ (x : α), Eq ↑(NNNorm.nnnorm (f x).toReal) ↑⟨(f x).toReal, ⋯⟩
    ⊢ (MeasureTheory.ae μ).EventuallyEq f fun a => ↑⟨(f a).toReal, ⋯⟩
  -/
  filter_upwards [hf_ne_top] with x hfx
  /-
    case h
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf_ne_top : Filter.Eventually (fun x => Ne (f x) Top.top) (MeasureTheory.ae μ)
    h_eq : ∀ (x : α), Eq ↑(NNNorm.nnnorm (f x).toReal) ↑⟨(f x).toReal, ⋯⟩
    x : α
    hfx : Ne (f x) Top.top
    ⊢ Eq (f x) ↑⟨(f x).toReal, ⋯⟩
  -/
  lift f x to ℝ≥0 using hfx with fx h
  /-
    case h.intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf_ne_top : Filter.Eventually (fun x => Ne (f x) Top.top) (MeasureTheory.ae μ)
    h_eq : ∀ (x : α), Eq ↑(NNNorm.nnnorm (f x).toReal) ↑⟨(f x).toReal, ⋯⟩
    x : α
    fx : NNReal
    h : Eq (↑fx) (f x)
    ⊢ Eq ↑fx ↑⟨(↑fx).toReal, ⋯⟩
  -/
  simp only [coe_toReal, ENNReal.coe_inj]
  /-
    case h.intro
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf_ne_top : Filter.Eventually (fun x => Ne (f x) Top.top) (MeasureTheory.ae μ)
    h_eq : ∀ (x : α), Eq ↑(NNNorm.nnnorm (f x).toReal) ↑⟨(f x).toReal, ⋯⟩
    x : α
    fx : NNReal
    h : Eq (↑fx) (f x)
    ⊢ Eq fx ⟨↑fx, ⋯⟩
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem isFiniteMeasure_withDensity_ofReal {f : α → ℝ} (hfi : HasFiniteIntegral f μ) :
    IsFiniteMeasure (μ.withDensity fun x => ENNReal.ofReal <| f x) := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hfi : MeasureTheory.HasFiniteIntegral f μ
    ⊢ MeasureTheory.IsFiniteMeasure (μ.withDensity fun x => ENNReal.ofReal (f x))
  -/
  refine isFiniteMeasure_withDensity ((lintegral_mono fun x => ?_).trans_lt hfi).ne
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hfi : MeasureTheory.HasFiniteIntegral f μ
    x : α
    ⊢ LE.le (ENNReal.ofReal (f x)) (ENorm.enorm (f x))
  -/
  exact Real.ofReal_le_ennnorm (f x)
  /-
    🎉 no goals
  -/


theorem all_ae_ofReal_F_le_bound (h : ∀ n, ∀ᵐ a ∂μ, ‖F n a‖ ≤ bound a) :
    ∀ n, ∀ᵐ a ∂μ, ENNReal.ofReal ‖F n a‖ ≤ ENNReal.ofReal (bound a) := fun n =>
  (h n).mono fun _ h => ENNReal.ofReal_le_ofReal h


theorem all_ae_tendsto_ofReal_norm (h : ∀ᵐ a ∂μ, Tendsto (fun n => F n a) atTop <| 𝓝 <| f a) :
    ∀ᵐ a ∂μ, Tendsto (fun n => ENNReal.ofReal ‖F n a‖) atTop <| 𝓝 <| ENNReal.ofReal ‖f a‖ :=
  h.mono fun _ h => tendsto_ofReal <| Tendsto.comp (Continuous.tendsto continuous_norm _) h


theorem all_ae_ofReal_f_le_bound (h_bound : ∀ n, ∀ᵐ a ∂μ, ‖F n a‖ ≤ bound a)
    (h_lim : ∀ᵐ a ∂μ, Tendsto (fun n => F n a) atTop (𝓝 (f a))) :
    ∀ᵐ a ∂μ, ENNReal.ofReal ‖f a‖ ≤ ENNReal.ofReal (bound a) := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    F : Nat → α → β
    f : α → β
    bound : α → Real
    h_bound : ∀ (n : Nat), Filter.Eventually (fun a => LE.le (Norm.norm (F n a)) ( …
    h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => F n a) Filter.atT …
    ⊢ Filter.Eventually (fun a => LE.le (ENNReal.ofReal (Norm.norm (f a))) (ENNRea …
  -/
  have F_le_bound := all_ae_ofReal_F_le_bound h_bound
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    F : Nat → α → β
    f : α → β
    bound : α → Real
    h_bound : ∀ (n : Nat), Filter.Eventually (fun a => LE.le (Norm.norm (F n a)) ( …
    h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => F n a) Filter.atT …
    F_le_bound : ∀ (n : Nat), Filter.Eventually (fun a => LE.le (ENNReal.ofReal (N …
    ⊢ Filter.Eventually (fun a => LE.le (ENNReal.ofReal (Norm.norm (f a))) (ENNRea …
  -/
  rw [← ae_all_iff] at F_le_bound
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    F : Nat → α → β
    f : α → β
    bound : α → Real
    h_bound : ∀ (n : Nat), Filter.Eventually (fun a => LE.le (Norm.norm (F n a)) ( …
    h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => F n a) Filter.atT …
    F_le_bound : Filter.Eventually (fun a => ∀ (i : Nat), LE.le (ENNReal.ofReal (N …
    ⊢ Filter.Eventually (fun a => LE.le (ENNReal.ofReal (Norm.norm (f a))) (ENNRea …
  -/
  apply F_le_bound.mp ((all_ae_tendsto_ofReal_norm h_lim).mono _)
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    F : Nat → α → β
    f : α → β
    bound : α → Real
    h_bound : ∀ (n : Nat), Filter.Eventually (fun a => LE.le (Norm.norm (F n a)) ( …
    h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => F n a) Filter.atT …
    F_le_bound : Filter.Eventually (fun a => ∀ (i : Nat), LE.le (ENNReal.ofReal (N …
    ⊢ ∀ (x : α), Filter.Tendsto (fun n => ENNReal.ofReal (Norm.norm (F n x))) Filt …
  -/
  intro a tendsto_norm F_le_bound
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    F : Nat → α → β
    f : α → β
    bound : α → Real
    h_bound : ∀ (n : Nat), Filter.Eventually (fun a => LE.le (Norm.norm (F n a)) ( …
    h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => F n a) Filter.atT …
    F_le_bound✝ : Filter.Eventually (fun a => ∀ (i : Nat), LE.le (ENNReal.ofReal ( …
    a : α
    tendsto_norm : Filter.Tendsto (fun n => ENNReal.ofReal (Norm.norm (F n a))) Fi …
    F_le_bound : ∀ (i : Nat), LE.le (ENNReal.ofReal (Norm.norm (F i a))) (ENNReal. …
    ⊢ LE.le (ENNReal.ofReal (Norm.norm (f a))) (ENNReal.ofReal (bound a))
  -/
  exact le_of_tendsto' tendsto_norm F_le_bound
  /-
    🎉 no goals
  -/


theorem hasFiniteIntegral_of_dominated_convergence {F : ℕ → α → β} {f : α → β} {bound : α → ℝ}
    (bound_hasFiniteIntegral : HasFiniteIntegral bound μ)
    (h_bound : ∀ n, ∀ᵐ a ∂μ, ‖F n a‖ ≤ bound a)
    (h_lim : ∀ᵐ a ∂μ, Tendsto (fun n => F n a) atTop (𝓝 (f a))) : HasFiniteIntegral f μ := by
  /- `‖F n a‖ ≤ bound a` and `‖F n a‖ --> ‖f a‖` implies `‖f a‖ ≤ bound a`,
    and so `∫ ‖f‖ ≤ ∫ bound < ∞` since `bound` is has_finite_integral -/
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    F : Nat → α → β
    f : α → β
    bound : α → Real
    bound_hasFiniteIntegral : MeasureTheory.HasFiniteIntegral bound μ
    h_bound : ∀ (n : Nat), Filter.Eventually (fun a => LE.le (Norm.norm (F n a)) ( …
    h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => F n a) Filter.atT …
    ⊢ MeasureTheory.HasFiniteIntegral f μ
  -/
  rw [hasFiniteIntegral_iff_norm]
  calc
    (∫⁻ a, ENNReal.ofReal ‖f a‖ ∂μ) ≤ ∫⁻ a, ENNReal.ofReal (bound a) ∂μ :=
      lintegral_mono_ae <| all_ae_ofReal_f_le_bound h_bound h_lim
    _ < ∞ := by
      rw [← hasFiniteIntegral_iff_ofReal]
      · exact bound_hasFiniteIntegral
      exact (h_bound 0).mono fun a h => le_trans (norm_nonneg _) h


theorem tendsto_lintegral_norm_of_dominated_convergence {F : ℕ → α → β} {f : α → β} {bound : α → ℝ}
    (F_measurable : ∀ n, AEStronglyMeasurable (F n) μ)
    (bound_hasFiniteIntegral : HasFiniteIntegral bound μ)
    (h_bound : ∀ n, ∀ᵐ a ∂μ, ‖F n a‖ ≤ bound a)
    (h_lim : ∀ᵐ a ∂μ, Tendsto (fun n => F n a) atTop (𝓝 (f a))) :
    Tendsto (fun n => ∫⁻ a, ENNReal.ofReal ‖F n a - f a‖ ∂μ) atTop (𝓝 0) := by
  have f_measurable : AEStronglyMeasurable f μ :=
    aestronglyMeasurable_of_tendsto_ae _ F_measurable h_lim
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    F : Nat → α → β
    f : α → β
    bound : α → Real
    F_measurable : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (F n) μ
    bound_hasFiniteIntegral : MeasureTheory.HasFiniteIntegral bound μ
    h_bound : ∀ (n : Nat), Filter.Eventually (fun a => LE.le (Norm.norm (F n a)) ( …
    h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => F n a) Filter.atT …
    f_measurable : MeasureTheory.AEStronglyMeasurable f μ
    ⊢ Filter.Tendsto (fun n => MeasureTheory.lintegral μ fun a => ENNReal.ofReal ( …
  -/
  let b a := 2 * ENNReal.ofReal (bound a)
  /- `‖F n a‖ ≤ bound a` and `F n a --> f a` implies `‖f a‖ ≤ bound a`, and thus by the
    triangle inequality, have `‖F n a - f a‖ ≤ 2 * (bound a)`. -/
  have hb : ∀ n, ∀ᵐ a ∂μ, ENNReal.ofReal ‖F n a - f a‖ ≤ b a := by
    intro n
    filter_upwards [all_ae_ofReal_F_le_bound h_bound n,
      all_ae_ofReal_f_le_bound h_bound h_lim] with a h₁ h₂
    calc
      ENNReal.ofReal ‖F n a - f a‖ ≤ ENNReal.ofReal ‖F n a‖ + ENNReal.ofReal ‖f a‖ := by
        rw [← ENNReal.ofReal_add]
        · apply ofReal_le_ofReal
          apply norm_sub_le
        · exact norm_nonneg _
        · exact norm_nonneg _
      _ ≤ ENNReal.ofReal (bound a) + ENNReal.ofReal (bound a) := add_le_add h₁ h₂
      _ = b a := by rw [← two_mul]
  -- On the other hand, `F n a --> f a` implies that `‖F n a - f a‖ --> 0`
  have h : ∀ᵐ a ∂μ, Tendsto (fun n => ENNReal.ofReal ‖F n a - f a‖) atTop (𝓝 0) := by
    rw [← ENNReal.ofReal_zero]
    refine h_lim.mono fun a h => (continuous_ofReal.tendsto _).comp ?_
    rwa [← tendsto_iff_norm_sub_tendsto_zero]
  /- Therefore, by the dominated convergence theorem for nonnegative integration, have
    ` ∫ ‖f a - F n a‖ --> 0 ` -/
  suffices Tendsto (fun n => ∫⁻ a, ENNReal.ofReal ‖F n a - f a‖ ∂μ) atTop (𝓝 (∫⁻ _ : α, 0 ∂μ)) by
    rwa [lintegral_zero] at this
  -- Using the dominated convergence theorem.
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    F : Nat → α → β
    f : α → β
    bound : α → Real
    F_measurable : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (F n) μ
    bound_hasFiniteIntegral : MeasureTheory.HasFiniteIntegral bound μ
    h_bound : ∀ (n : Nat), Filter.Eventually (fun a => LE.le (Norm.norm (F n a)) ( …
    h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => F n a) Filter.atT …
    f_measurable : MeasureTheory.AEStronglyMeasurable f μ
    b : α → ENNReal := fun a => HMul.hMul 2 (ENNReal.ofReal (bound a))
    hb : ∀ (n : Nat), Filter.Eventually (fun a => LE.le (ENNReal.ofReal (Norm.norm …
    h : Filter.Eventually (fun a => Filter.Tendsto (fun n => ENNReal.ofReal (Norm. …
    ⊢ Filter.Tendsto (fun n => MeasureTheory.lintegral μ fun a => ENNReal.ofReal ( …
  -/
  refine tendsto_lintegral_of_dominated_convergence' _ ?_ hb ?_ ?_
  -- Show `fun a => ‖f a - F n a‖` is almost everywhere measurable for all `n`
  · exact fun n =>
      measurable_ofReal.comp_aemeasurable ((F_measurable n).sub f_measurable).norm.aemeasurable
  -- Show `2 * bound` `HasFiniteIntegral`
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      F : Nat → α → β
      f : α → β
      bound : α → Real
      F_measurable : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (F n) μ
      bound_hasFiniteIntegral : MeasureTheory.HasFiniteIntegral bound μ
      h_bound : ∀ (n : Nat), Filter.Eventually (fun a => LE.le (Norm.norm (F n a)) ( …
      h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => F n a) Filter.atT …
      f_measurable : MeasureTheory.AEStronglyMeasurable f μ
      b : α → ENNReal := fun a => HMul.hMul 2 (ENNReal.ofReal (bound a))
      hb : ∀ (n : Nat), Filter.Eventually (fun a => LE.le (ENNReal.ofReal (Norm.norm …
      h : Filter.Eventually (fun a => Filter.Tendsto (fun n => ENNReal.ofReal (Norm. …
      ⊢ Ne (MeasureTheory.lintegral μ fun a => b a) Top.top
    -/
  · rw [hasFiniteIntegral_iff_ofReal] at bound_hasFiniteIntegral
    · calc
        ∫⁻ a, b a ∂μ = 2 * ∫⁻ a, ENNReal.ofReal (bound a) ∂μ := by
          rw [lintegral_const_mul']
          exact coe_ne_top
        _ ≠ ∞ := mul_ne_top coe_ne_top bound_hasFiniteIntegral.ne
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      F : Nat → α → β
      f : α → β
      bound : α → Real
      F_measurable : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (F n) μ
      bound_hasFiniteIntegral : MeasureTheory.HasFiniteIntegral bound μ
      h_bound : ∀ (n : Nat), Filter.Eventually (fun a => LE.le (Norm.norm (F n a)) ( …
      h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => F n a) Filter.atT …
      f_measurable : MeasureTheory.AEStronglyMeasurable f μ
      b : α → ENNReal := fun a => HMul.hMul 2 (ENNReal.ofReal (bound a))
      hb : ∀ (n : Nat), Filter.Eventually (fun a => LE.le (ENNReal.ofReal (Norm.norm …
      h : Filter.Eventually (fun a => Filter.Tendsto (fun n => ENNReal.ofReal (Norm. …
      ⊢ (MeasureTheory.ae μ).EventuallyLE 0 bound
    -/
    filter_upwards [h_bound 0] with _ h using le_trans (norm_nonneg _) h
    /-
      🎉 no goals
    -/
  -- Show `‖f a - F n a‖ --> 0`
    /-
      case refine_3
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      F : Nat → α → β
      f : α → β
      bound : α → Real
      F_measurable : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (F n) μ
      bound_hasFiniteIntegral : MeasureTheory.HasFiniteIntegral bound μ
      h_bound : ∀ (n : Nat), Filter.Eventually (fun a => LE.le (Norm.norm (F n a)) ( …
      h_lim : Filter.Eventually (fun a => Filter.Tendsto (fun n => F n a) Filter.atT …
      f_measurable : MeasureTheory.AEStronglyMeasurable f μ
      b : α → ENNReal := fun a => HMul.hMul 2 (ENNReal.ofReal (bound a))
      hb : ∀ (n : Nat), Filter.Eventually (fun a => LE.le (ENNReal.ofReal (Norm.norm …
      h : Filter.Eventually (fun a => Filter.Tendsto (fun n => ENNReal.ofReal (Norm. …
      ⊢ Filter.Eventually (fun a => Filter.Tendsto (fun n => ENNReal.ofReal (Norm.no …
    -/
  · exact h
    /-
      🎉 no goals
    -/


theorem HasFiniteIntegral.max_zero {f : α → ℝ} (hf : HasFiniteIntegral f μ) :
    HasFiniteIntegral (fun a => max (f a) 0) μ :=
                                              /-
                                                α : Type u_1
                                                m : MeasurableSpace α
                                                μ : MeasureTheory.Measure α
                                                f : α → Real
                                                hf : MeasureTheory.HasFiniteIntegral f μ
                                                x : α
                                                ⊢ LE.le (Norm.norm (Max.max (f x) 0)) (Norm.norm (f x))
                                              -/
  hf.mono <| Eventually.of_forall fun x => by simp [abs_le, le_abs_self]
                                              /-
                                                🎉 no goals
                                              -/


theorem HasFiniteIntegral.min_zero {f : α → ℝ} (hf : HasFiniteIntegral f μ) :
    HasFiniteIntegral (fun a => min (f a) 0) μ :=
                                              /-
                                                α : Type u_1
                                                m : MeasurableSpace α
                                                μ : MeasureTheory.Measure α
                                                f : α → Real
                                                hf : MeasureTheory.HasFiniteIntegral f μ
                                                x : α
                                                ⊢ LE.le (Norm.norm (Min.min (f x) 0)) (Norm.norm (f x))
                                              -/
  hf.mono <| Eventually.of_forall fun x => by simpa [abs_le] using neg_abs_le _
                                              /-
                                                🎉 no goals
                                              -/


theorem HasFiniteIntegral.smul [NormedAddCommGroup 𝕜] [SMulZeroClass 𝕜 β] [BoundedSMul 𝕜 β] (c : 𝕜)
    {f : α → β} : HasFiniteIntegral f μ → HasFiniteIntegral (c • f) μ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : NormedAddCommGroup β
    𝕜 : Type u_6
    inst✝² : NormedAddCommGroup 𝕜
    inst✝¹ : SMulZeroClass 𝕜 β
    inst✝ : BoundedSMul 𝕜 β
    c : 𝕜
    f : α → β
    ⊢ MeasureTheory.HasFiniteIntegral f μ → MeasureTheory.HasFiniteIntegral (HSMul …
  -/
  simp only [HasFiniteIntegral]; intro hfi
  calc
    (∫⁻ a : α, ‖c • f a‖₊ ∂μ) ≤ ∫⁻ a : α, ‖c‖₊ * ‖f a‖₊ ∂μ := by
      refine lintegral_mono ?_
      intro i
      -- After https://github.com/leanprover/lean4/pull/2734, we need to do beta reduction `exact mod_cast`
      beta_reduce
      exact mod_cast (nnnorm_smul_le c (f i))
    _ < ∞ := by
      rw [lintegral_const_mul']
      exacts [mul_lt_top coe_lt_top hfi, coe_ne_top]


theorem hasFiniteIntegral_smul_iff [NormedRing 𝕜] [MulActionWithZero 𝕜 β] [BoundedSMul 𝕜 β] {c : 𝕜}
    (hc : IsUnit c) (f : α → β) : HasFiniteIntegral (c • f) μ ↔ HasFiniteIntegral f μ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : NormedAddCommGroup β
    𝕜 : Type u_6
    inst✝² : NormedRing 𝕜
    inst✝¹ : MulActionWithZero 𝕜 β
    inst✝ : BoundedSMul 𝕜 β
    c : 𝕜
    hc : IsUnit c
    f : α → β
    ⊢ Iff (MeasureTheory.HasFiniteIntegral (HSMul.hSMul c f) μ) (MeasureTheory.Has …
  -/
  obtain ⟨c, rfl⟩ := hc
  /-
    case intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : NormedAddCommGroup β
    𝕜 : Type u_6
    inst✝² : NormedRing 𝕜
    inst✝¹ : MulActionWithZero 𝕜 β
    inst✝ : BoundedSMul 𝕜 β
    f : α → β
    c : Units 𝕜
    ⊢ Iff (MeasureTheory.HasFiniteIntegral (HSMul.hSMul (↑c) f) μ) (MeasureTheory. …
  -/
  constructor
    /-
      case intro.mp
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝³ : NormedAddCommGroup β
      𝕜 : Type u_6
      inst✝² : NormedRing 𝕜
      inst✝¹ : MulActionWithZero 𝕜 β
      inst✝ : BoundedSMul 𝕜 β
      f : α → β
      c : Units 𝕜
      ⊢ MeasureTheory.HasFiniteIntegral (HSMul.hSMul (↑c) f) μ → MeasureTheory.HasFi …
    -/
  · intro h
    /-
      case intro.mp
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝³ : NormedAddCommGroup β
      𝕜 : Type u_6
      inst✝² : NormedRing 𝕜
      inst✝¹ : MulActionWithZero 𝕜 β
      inst✝ : BoundedSMul 𝕜 β
      f : α → β
      c : Units 𝕜
      h : MeasureTheory.HasFiniteIntegral (HSMul.hSMul (↑c) f) μ
      ⊢ MeasureTheory.HasFiniteIntegral f μ
    -/
    simpa only [smul_smul, Units.inv_mul, one_smul] using h.smul ((c⁻¹ : 𝕜ˣ) : 𝕜)
    /-
      🎉 no goals
    -/
  /-
    case intro.mpr
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : NormedAddCommGroup β
    𝕜 : Type u_6
    inst✝² : NormedRing 𝕜
    inst✝¹ : MulActionWithZero 𝕜 β
    inst✝ : BoundedSMul 𝕜 β
    f : α → β
    c : Units 𝕜
    ⊢ MeasureTheory.HasFiniteIntegral f μ → MeasureTheory.HasFiniteIntegral (HSMul …
  -/
  exact HasFiniteIntegral.smul _
  /-
    🎉 no goals
  -/


theorem HasFiniteIntegral.const_mul [NormedRing 𝕜] {f : α → 𝕜} (h : HasFiniteIntegral f μ) (c : 𝕜) :
    HasFiniteIntegral (fun x => c * f x) μ :=
  h.smul c


theorem HasFiniteIntegral.mul_const [NormedRing 𝕜] {f : α → 𝕜} (h : HasFiniteIntegral f μ) (c : 𝕜) :
    HasFiniteIntegral (fun x => f x * c) μ :=
  h.smul (MulOpposite.op c)


/-- `Integrable f μ` means that `f` is measurable and that the integral `∫⁻ a, ‖f a‖ ∂μ` is finite.
  `Integrable f` means `Integrable f volume`. -/
def Integrable {α} {_ : MeasurableSpace α} (f : α → ε)
    (μ : Measure α := by volume_tac) : Prop :=
  AEStronglyMeasurable f μ ∧ HasFiniteIntegral f μ


/-- Notation for `Integrable` with respect to a non-standard σ-algebra. -/
scoped notation "Integrable[" mα "]" => @Integrable _ _ _ _ mα


theorem memℒp_one_iff_integrable {f : α → β} : Memℒp f 1 μ ↔ Integrable f μ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    ⊢ Iff (MeasureTheory.Memℒp f 1 μ) (MeasureTheory.Integrable f μ)
  -/
  simp_rw [Integrable, hasFiniteIntegral_iff_nnnorm, Memℒp, eLpNorm_one_eq_lintegral_nnnorm]
  /-
    🎉 no goals
  -/


theorem Integrable.aestronglyMeasurable {f : α → β} (hf : Integrable f μ) :
    AEStronglyMeasurable f μ :=
  hf.1


theorem Integrable.aemeasurable [MeasurableSpace β] [BorelSpace β] {f : α → β}
    (hf : Integrable f μ) : AEMeasurable f μ :=
  hf.aestronglyMeasurable.aemeasurable


theorem Integrable.hasFiniteIntegral {f : α → β} (hf : Integrable f μ) : HasFiniteIntegral f μ :=
  hf.2


theorem Integrable.mono {f : α → β} {g : α → γ} (hg : Integrable g μ)
    (hf : AEStronglyMeasurable f μ) (h : ∀ᵐ a ∂μ, ‖f a‖ ≤ ‖g a‖) : Integrable f μ :=
  ⟨hf, hg.hasFiniteIntegral.mono h⟩


theorem Integrable.mono' {f : α → β} {g : α → ℝ} (hg : Integrable g μ)
    (hf : AEStronglyMeasurable f μ) (h : ∀ᵐ a ∂μ, ‖f a‖ ≤ g a) : Integrable f μ :=
  ⟨hf, hg.hasFiniteIntegral.mono' h⟩


theorem Integrable.congr' {f : α → β} {g : α → γ} (hf : Integrable f μ)
    (hg : AEStronglyMeasurable g μ) (h : ∀ᵐ a ∂μ, ‖f a‖ = ‖g a‖) : Integrable g μ :=
  ⟨hg, hf.hasFiniteIntegral.congr' h⟩


theorem integrable_congr' {f : α → β} {g : α → γ} (hf : AEStronglyMeasurable f μ)
    (hg : AEStronglyMeasurable g μ) (h : ∀ᵐ a ∂μ, ‖f a‖ = ‖g a‖) :
    Integrable f μ ↔ Integrable g μ :=
  ⟨fun h2f => h2f.congr' hg h, fun h2g => h2g.congr' hf <| EventuallyEq.symm h⟩


theorem Integrable.congr {f g : α → β} (hf : Integrable f μ) (h : f =ᵐ[μ] g) : Integrable g μ :=
  ⟨hf.1.congr h, hf.2.congr h⟩


theorem integrable_congr {f g : α → β} (h : f =ᵐ[μ] g) : Integrable f μ ↔ Integrable g μ :=
  ⟨fun hf => hf.congr h, fun hg => hg.congr h.symm⟩


theorem integrable_const_iff {c : β} : Integrable (fun _ : α => c) μ ↔ c = 0 ∨ μ univ < ∞ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    c : β
    ⊢ Iff (MeasureTheory.Integrable (fun x => c) μ) (Or (Eq c 0) (LT.lt (μ Set.uni …
  -/
  have : AEStronglyMeasurable (fun _ : α => c) μ := aestronglyMeasurable_const
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    c : β
    this : MeasureTheory.AEStronglyMeasurable (fun x => c) μ
    ⊢ Iff (MeasureTheory.Integrable (fun x => c) μ) (Or (Eq c 0) (LT.lt (μ Set.uni …
  -/
  rw [Integrable, and_iff_right this, hasFiniteIntegral_const_iff]
  /-
    🎉 no goals
  -/


theorem Integrable.of_mem_Icc [IsFiniteMeasure μ] (a b : ℝ) {X : α → ℝ} (hX : AEMeasurable X μ)
    (h : ∀ᵐ ω ∂μ, X ω ∈ Set.Icc a b) :
    Integrable X μ :=
  ⟨hX.aestronglyMeasurable, .of_mem_Icc a b h⟩


@[simp]
theorem integrable_const [IsFiniteMeasure μ] (c : β) : Integrable (fun _ : α => c) μ :=
  integrable_const_iff.2 <| Or.inr <| measure_lt_top _ _


@[simp]
lemma Integrable.of_finite [Finite α] [MeasurableSingletonClass α] [IsFiniteMeasure μ] {f : α → β} :
    Integrable f μ := ⟨.of_finite, .of_finite⟩


/-- This lemma is a special case of `Integrable.of_finite`. -/
-- Eternal deprecation for discoverability, don't remove
@[deprecated Integrable.of_finite (since := "2024-10-05"), nolint deprecatedNoSince]
lemma Integrable.of_isEmpty [IsEmpty α] {f : α → β} : Integrable f μ := .of_finite


@[deprecated (since := "2024-02-05")] alias integrable_of_fintype := Integrable.of_finite


theorem Memℒp.integrable_norm_rpow {f : α → β} {p : ℝ≥0∞} (hf : Memℒp f p μ) (hp_ne_zero : p ≠ 0)
    (hp_ne_top : p ≠ ∞) : Integrable (fun x : α => ‖f x‖ ^ p.toReal) μ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    p : ENNReal
    hf : MeasureTheory.Memℒp f p μ
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    ⊢ MeasureTheory.Integrable (fun x => HPow.hPow (Norm.norm (f x)) p.toReal) μ
  -/
  rw [← memℒp_one_iff_integrable]
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    p : ENNReal
    hf : MeasureTheory.Memℒp f p μ
    hp_ne_zero : Ne p 0
    hp_ne_top : Ne p Top.top
    ⊢ MeasureTheory.Memℒp (fun x => HPow.hPow (Norm.norm (f x)) p.toReal) 1 μ
  -/
  exact hf.norm_rpow hp_ne_zero hp_ne_top
  /-
    🎉 no goals
  -/


theorem Memℒp.integrable_norm_rpow' [IsFiniteMeasure μ] {f : α → β} {p : ℝ≥0∞} (hf : Memℒp f p μ) :
    Integrable (fun x : α => ‖f x‖ ^ p.toReal) μ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : α → β
    p : ENNReal
    hf : MeasureTheory.Memℒp f p μ
    ⊢ MeasureTheory.Integrable (fun x => HPow.hPow (Norm.norm (f x)) p.toReal) μ
  -/
  by_cases h_zero : p = 0
    /-
      case pos
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup β
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f : α → β
      p : ENNReal
      hf : MeasureTheory.Memℒp f p μ
      h_zero : Eq p 0
      ⊢ MeasureTheory.Integrable (fun x => HPow.hPow (Norm.norm (f x)) p.toReal) μ
    -/
  · simp [h_zero, integrable_const]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : α → β
    p : ENNReal
    hf : MeasureTheory.Memℒp f p μ
    h_zero : Not (Eq p 0)
    ⊢ MeasureTheory.Integrable (fun x => HPow.hPow (Norm.norm (f x)) p.toReal) μ
  -/
  by_cases h_top : p = ∞
    /-
      case pos
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup β
      inst✝ : MeasureTheory.IsFiniteMeasure μ
      f : α → β
      p : ENNReal
      hf : MeasureTheory.Memℒp f p μ
      h_zero : Not (Eq p 0)
      h_top : Eq p Top.top
      ⊢ MeasureTheory.Integrable (fun x => HPow.hPow (Norm.norm (f x)) p.toReal) μ
    -/
  · simp [h_top, integrable_const]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f : α → β
    p : ENNReal
    hf : MeasureTheory.Memℒp f p μ
    h_zero : Not (Eq p 0)
    h_top : Not (Eq p Top.top)
    ⊢ MeasureTheory.Integrable (fun x => HPow.hPow (Norm.norm (f x)) p.toReal) μ
  -/
  exact hf.integrable_norm_rpow h_zero h_top
  /-
    🎉 no goals
  -/


theorem Integrable.mono_measure {f : α → β} (h : Integrable f ν) (hμ : μ ≤ ν) : Integrable f μ :=
  ⟨h.aestronglyMeasurable.mono_measure hμ, h.hasFiniteIntegral.mono_measure hμ⟩


theorem Integrable.of_measure_le_smul {μ' : Measure α} (c : ℝ≥0∞) (hc : c ≠ ∞) (hμ'_le : μ' ≤ c • μ)
    {f : α → β} (hf : Integrable f μ) : Integrable f μ' := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    μ' : MeasureTheory.Measure α
    c : ENNReal
    hc : Ne c Top.top
    hμ'_le : LE.le μ' (HSMul.hSMul c μ)
    f : α → β
    hf : MeasureTheory.Integrable f μ
    ⊢ MeasureTheory.Integrable f μ'
  -/
  rw [← memℒp_one_iff_integrable] at hf ⊢
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    μ' : MeasureTheory.Measure α
    c : ENNReal
    hc : Ne c Top.top
    hμ'_le : LE.le μ' (HSMul.hSMul c μ)
    f : α → β
    hf : MeasureTheory.Memℒp f 1 μ
    ⊢ MeasureTheory.Memℒp f 1 μ'
  -/
  exact hf.of_measure_le_smul c hc hμ'_le
  /-
    🎉 no goals
  -/


theorem Integrable.add_measure {f : α → β} (hμ : Integrable f μ) (hν : Integrable f ν) :
    Integrable f (μ + ν) := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    hμ : MeasureTheory.Integrable f μ
    hν : MeasureTheory.Integrable f ν
    ⊢ MeasureTheory.Integrable f (HAdd.hAdd μ ν)
  -/
  simp_rw [← memℒp_one_iff_integrable] at hμ hν ⊢
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    hμ : MeasureTheory.Memℒp f 1 μ
    hν : MeasureTheory.Memℒp f 1 ν
    ⊢ MeasureTheory.Memℒp f 1 (HAdd.hAdd μ ν)
  -/
  refine ⟨hμ.aestronglyMeasurable.add_measure hν.aestronglyMeasurable, ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    hμ : MeasureTheory.Memℒp f 1 μ
    hν : MeasureTheory.Memℒp f 1 ν
    ⊢ LT.lt (MeasureTheory.eLpNorm f 1 (HAdd.hAdd μ ν)) Top.top
  -/
  rw [eLpNorm_one_add_measure, ENNReal.add_lt_top]
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    hμ : MeasureTheory.Memℒp f 1 μ
    hν : MeasureTheory.Memℒp f 1 ν
    ⊢ And (LT.lt (MeasureTheory.eLpNorm f 1 μ) Top.top) (LT.lt (MeasureTheory.eLpN …
  -/
  exact ⟨hμ.eLpNorm_lt_top, hν.eLpNorm_lt_top⟩
  /-
    🎉 no goals
  -/


theorem Integrable.left_of_add_measure {f : α → β} (h : Integrable f (μ + ν)) : Integrable f μ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    h : MeasureTheory.Integrable f (HAdd.hAdd μ ν)
    ⊢ MeasureTheory.Integrable f μ
  -/
  rw [← memℒp_one_iff_integrable] at h ⊢
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    h : MeasureTheory.Memℒp f 1 (HAdd.hAdd μ ν)
    ⊢ MeasureTheory.Memℒp f 1 μ
  -/
  exact h.left_of_add_measure
  /-
    🎉 no goals
  -/


theorem Integrable.right_of_add_measure {f : α → β} (h : Integrable f (μ + ν)) :
    Integrable f ν := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    h : MeasureTheory.Integrable f (HAdd.hAdd μ ν)
    ⊢ MeasureTheory.Integrable f ν
  -/
  rw [← memℒp_one_iff_integrable] at h ⊢
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    h : MeasureTheory.Memℒp f 1 (HAdd.hAdd μ ν)
    ⊢ MeasureTheory.Memℒp f 1 ν
  -/
  exact h.right_of_add_measure
  /-
    🎉 no goals
  -/


@[simp]
theorem integrable_add_measure {f : α → β} :
    Integrable f (μ + ν) ↔ Integrable f μ ∧ Integrable f ν :=
  ⟨fun h => ⟨h.left_of_add_measure, h.right_of_add_measure⟩, fun h => h.1.add_measure h.2⟩


@[simp]
theorem integrable_zero_measure {_ : MeasurableSpace α} {f : α → β} :
    Integrable f (0 : Measure α) :=
  ⟨aestronglyMeasurable_zero_measure f, hasFiniteIntegral_zero_measure f⟩


theorem integrable_finset_sum_measure {ι} {m : MeasurableSpace α} {f : α → β} {μ : ι → Measure α}
    {s : Finset ι} : Integrable f (∑ i ∈ s, μ i) ↔ ∀ i ∈ s, Integrable f (μ i) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : NormedAddCommGroup β
    ι : Type u_6
    m : MeasurableSpace α
    f : α → β
    μ : ι → MeasureTheory.Measure α
    s : Finset ι
    ⊢ Iff (MeasureTheory.Integrable f (s.sum fun i => μ i)) (∀ (i : ι), Membership …
  -/
                                            /-
                                              🎉 no goals
                                            -/
  induction s using Finset.induction_on <;> simp [*]
                                            /-
                                              🎉 no goals
                                            -/


theorem Integrable.smul_measure {f : α → β} (h : Integrable f μ) {c : ℝ≥0∞} (hc : c ≠ ∞) :
    Integrable f (c • μ) := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    h : MeasureTheory.Integrable f μ
    c : ENNReal
    hc : Ne c Top.top
    ⊢ MeasureTheory.Integrable f (HSMul.hSMul c μ)
  -/
  rw [← memℒp_one_iff_integrable] at h ⊢
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    h : MeasureTheory.Memℒp f 1 μ
    c : ENNReal
    hc : Ne c Top.top
    ⊢ MeasureTheory.Memℒp f 1 (HSMul.hSMul c μ)
  -/
  exact h.smul_measure hc
  /-
    🎉 no goals
  -/


theorem Integrable.smul_measure_nnreal {f : α → β} (h : Integrable f μ) {c : ℝ≥0} :
    Integrable f (c • μ) := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    h : MeasureTheory.Integrable f μ
    c : NNReal
    ⊢ MeasureTheory.Integrable f (HSMul.hSMul c μ)
  -/
  apply h.smul_measure
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    h : MeasureTheory.Integrable f μ
    c : NNReal
    ⊢ Ne (↑ENNReal.ofNNRealHom.toMonoidWithZeroHom c) Top.top
  -/
  simp
  /-
    🎉 no goals
  -/


theorem integrable_smul_measure {f : α → β} {c : ℝ≥0∞} (h₁ : c ≠ 0) (h₂ : c ≠ ∞) :
    Integrable f (c • μ) ↔ Integrable f μ :=
  ⟨fun h => by
    simpa only [smul_smul, ENNReal.inv_mul_cancel h₁ h₂, one_smul] using
      h.smul_measure (ENNReal.inv_ne_top.2 h₁),
    fun h => h.smul_measure h₂⟩


theorem integrable_inv_smul_measure {f : α → β} {c : ℝ≥0∞} (h₁ : c ≠ 0) (h₂ : c ≠ ∞) :
    Integrable f (c⁻¹ • μ) ↔ Integrable f μ :=
                              /-
                                α : Type u_1
                                β : Type u_2
                                m : MeasurableSpace α
                                μ : MeasureTheory.Measure α
                                inst✝ : NormedAddCommGroup β
                                f : α → β
                                c : ENNReal
                                h₁ : Ne c 0
                                h₂ : Ne c Top.top
                                ⊢ Ne (Inv.inv c) 0
                              -/
                              /-
                                🎉 no goals
                              -/
  integrable_smul_measure (by simpa using h₂) (by simpa using h₁)
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem Integrable.to_average {f : α → β} (h : Integrable f μ) : Integrable f ((μ univ)⁻¹ • μ) := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    h : MeasureTheory.Integrable f μ
    ⊢ MeasureTheory.Integrable f (HSMul.hSMul (Inv.inv (μ Set.univ)) μ)
  -/
  rcases eq_or_ne μ 0 with (rfl | hne)
    /-
      case inl
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      inst✝ : NormedAddCommGroup β
      f : α → β
      h : MeasureTheory.Integrable f 0
      ⊢ MeasureTheory.Integrable f (HSMul.hSMul (Inv.inv (0 Set.univ)) 0)
    -/
  · rwa [smul_zero]
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      f : α → β
      h : MeasureTheory.Integrable f μ
      hne : Ne μ 0
      ⊢ MeasureTheory.Integrable f (HSMul.hSMul (Inv.inv (μ Set.univ)) μ)
    -/
  · apply h.smul_measure
    /-
      case inr
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      f : α → β
      h : MeasureTheory.Integrable f μ
      hne : Ne μ 0
      ⊢ Ne (Inv.inv (μ Set.univ)) Top.top
    -/
    simpa
    /-
      🎉 no goals
    -/


theorem integrable_average [IsFiniteMeasure μ] {f : α → β} :
    Integrable f ((μ univ)⁻¹ • μ) ↔ Integrable f μ :=
                                       /-
                                         α : Type u_1
                                         β : Type u_2
                                         m : MeasurableSpace α
                                         μ : MeasureTheory.Measure α
                                         inst✝¹ : NormedAddCommGroup β
                                         inst✝ : MeasureTheory.IsFiniteMeasure μ
                                         f : α → β
                                         h : Eq μ 0
                                         ⊢ Iff (MeasureTheory.Integrable f (HSMul.hSMul (Inv.inv (μ Set.univ)) μ)) (Mea …
                                       -/
  (eq_or_ne μ 0).by_cases (fun h => by simp [h]) fun h =>
                                       /-
                                         🎉 no goals
                                       -/
    integrable_smul_measure (ENNReal.inv_ne_zero.2 <| measure_ne_top _ _)
      (ENNReal.inv_ne_top.2 <| mt Measure.measure_univ_eq_zero.1 h)


theorem integrable_map_measure {f : α → δ} {g : δ → β}
    (hg : AEStronglyMeasurable g (Measure.map f μ)) (hf : AEMeasurable f μ) :
    Integrable g (Measure.map f μ) ↔ Integrable (g ∘ f) μ := by
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_4
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasurableSpace δ
    inst✝ : NormedAddCommGroup β
    f : α → δ
    g : δ → β
    hg : MeasureTheory.AEStronglyMeasurable g (MeasureTheory.Measure.map f μ)
    hf : AEMeasurable f μ
    ⊢ Iff (MeasureTheory.Integrable g (MeasureTheory.Measure.map f μ)) (MeasureThe …
  -/
  simp_rw [← memℒp_one_iff_integrable]
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_4
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasurableSpace δ
    inst✝ : NormedAddCommGroup β
    f : α → δ
    g : δ → β
    hg : MeasureTheory.AEStronglyMeasurable g (MeasureTheory.Measure.map f μ)
    hf : AEMeasurable f μ
    ⊢ Iff (MeasureTheory.Memℒp g 1 (MeasureTheory.Measure.map f μ)) (MeasureTheory …
  -/
  exact memℒp_map_measure_iff hg hf
  /-
    🎉 no goals
  -/


theorem Integrable.comp_aemeasurable {f : α → δ} {g : δ → β} (hg : Integrable g (Measure.map f μ))
    (hf : AEMeasurable f μ) : Integrable (g ∘ f) μ :=
  (integrable_map_measure hg.aestronglyMeasurable hf).mp hg


theorem Integrable.comp_measurable {f : α → δ} {g : δ → β} (hg : Integrable g (Measure.map f μ))
    (hf : Measurable f) : Integrable (g ∘ f) μ :=
  hg.comp_aemeasurable hf.aemeasurable


theorem _root_.MeasurableEmbedding.integrable_map_iff {f : α → δ} (hf : MeasurableEmbedding f)
    {g : δ → β} : Integrable g (Measure.map f μ) ↔ Integrable (g ∘ f) μ := by
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_4
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasurableSpace δ
    inst✝ : NormedAddCommGroup β
    f : α → δ
    hf : MeasurableEmbedding f
    g : δ → β
    ⊢ Iff (MeasureTheory.Integrable g (MeasureTheory.Measure.map f μ)) (MeasureThe …
  -/
  simp_rw [← memℒp_one_iff_integrable]
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_4
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasurableSpace δ
    inst✝ : NormedAddCommGroup β
    f : α → δ
    hf : MeasurableEmbedding f
    g : δ → β
    ⊢ Iff (MeasureTheory.Memℒp g 1 (MeasureTheory.Measure.map f μ)) (MeasureTheory …
  -/
  exact hf.memℒp_map_measure_iff
  /-
    🎉 no goals
  -/


theorem integrable_map_equiv (f : α ≃ᵐ δ) (g : δ → β) :
    Integrable g (Measure.map f μ) ↔ Integrable (g ∘ f) μ := by
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_4
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasurableSpace δ
    inst✝ : NormedAddCommGroup β
    f : MeasurableEquiv α δ
    g : δ → β
    ⊢ Iff (MeasureTheory.Integrable g (MeasureTheory.Measure.map (⇑f) μ)) (Measure …
  -/
  simp_rw [← memℒp_one_iff_integrable]
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_4
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasurableSpace δ
    inst✝ : NormedAddCommGroup β
    f : MeasurableEquiv α δ
    g : δ → β
    ⊢ Iff (MeasureTheory.Memℒp g 1 (MeasureTheory.Measure.map (⇑f) μ)) (MeasureThe …
  -/
  exact f.memℒp_map_measure_iff
  /-
    🎉 no goals
  -/


theorem MeasurePreserving.integrable_comp {ν : Measure δ} {g : δ → β} {f : α → δ}
    (hf : MeasurePreserving f μ ν) (hg : AEStronglyMeasurable g ν) :
    Integrable (g ∘ f) μ ↔ Integrable g ν := by
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_4
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasurableSpace δ
    inst✝ : NormedAddCommGroup β
    ν : MeasureTheory.Measure δ
    g : δ → β
    f : α → δ
    hf : MeasureTheory.MeasurePreserving f μ ν
    hg : MeasureTheory.AEStronglyMeasurable g ν
    ⊢ Iff (MeasureTheory.Integrable (Function.comp g f) μ) (MeasureTheory.Integrab …
  -/
  rw [← hf.map_eq] at hg ⊢
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_4
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasurableSpace δ
    inst✝ : NormedAddCommGroup β
    ν : MeasureTheory.Measure δ
    g : δ → β
    f : α → δ
    hf : MeasureTheory.MeasurePreserving f μ ν
    hg : MeasureTheory.AEStronglyMeasurable g (MeasureTheory.Measure.map f μ)
    ⊢ Iff (MeasureTheory.Integrable (Function.comp g f) μ) (MeasureTheory.Integrab …
  -/
  exact (integrable_map_measure hg hf.measurable.aemeasurable).symm
  /-
    🎉 no goals
  -/


theorem MeasurePreserving.integrable_comp_emb {f : α → δ} {ν} (h₁ : MeasurePreserving f μ ν)
    (h₂ : MeasurableEmbedding f) {g : δ → β} : Integrable (g ∘ f) μ ↔ Integrable g ν :=
  h₁.map_eq ▸ Iff.symm h₂.integrable_map_iff


theorem lintegral_edist_lt_top {f g : α → β} (hf : Integrable f μ) (hg : Integrable g μ) :
    (∫⁻ a, edist (f a) (g a) ∂μ) < ∞ :=
  lt_of_le_of_lt (lintegral_edist_triangle hf.aestronglyMeasurable aestronglyMeasurable_zero)
    (ENNReal.add_lt_top.2 <| by
      /-
        α : Type u_1
        β : Type u_2
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : NormedAddCommGroup β
        f g : α → β
        hf : MeasureTheory.Integrable f μ
        hg : MeasureTheory.Integrable g μ
        ⊢ And (LT.lt (MeasureTheory.lintegral μ fun a => EDist.edist (f a) (0 a)) Top. …
      -/
      simp_rw [Pi.zero_apply, ← hasFiniteIntegral_iff_edist]
      /-
        α : Type u_1
        β : Type u_2
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        inst✝ : NormedAddCommGroup β
        f g : α → β
        hf : MeasureTheory.Integrable f μ
        hg : MeasureTheory.Integrable g μ
        ⊢ And (MeasureTheory.HasFiniteIntegral f μ) (MeasureTheory.HasFiniteIntegral g …
      -/
      exact ⟨hf.hasFiniteIntegral, hg.hasFiniteIntegral⟩)
      /-
        🎉 no goals
      -/


@[simp]
theorem integrable_zero : Integrable (fun _ => (0 : β)) μ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    ⊢ MeasureTheory.Integrable (fun x => 0) μ
  -/
  simp [Integrable, aestronglyMeasurable_const]
  /-
    🎉 no goals
  -/


theorem Integrable.add' {f g : α → β} (hf : Integrable f μ) (hg : Integrable g μ) :
    HasFiniteIntegral (f + g) μ :=
  calc
    (∫⁻ a, ‖f a + g a‖₊ ∂μ) ≤ ∫⁻ a, ‖f a‖₊ + ‖g a‖₊ ∂μ :=
      lintegral_mono fun a => by
        -- After https://github.com/leanprover/lean4/pull/2734, we need to do beta reduction before `exact mod_cast`
        /-
          α : Type u_1
          β : Type u_2
          m : MeasurableSpace α
          μ : MeasureTheory.Measure α
          inst✝ : NormedAddCommGroup β
          f g : α → β
          hf : MeasureTheory.Integrable f μ
          hg : MeasureTheory.Integrable g μ
          a : α
          ⊢ LE.le (↑(NNNorm.nnnorm (HAdd.hAdd (f a) (g a)))) (HAdd.hAdd ↑(NNNorm.nnnorm  …
        -/
        beta_reduce
        /-
          α : Type u_1
          β : Type u_2
          m : MeasurableSpace α
          μ : MeasureTheory.Measure α
          inst✝ : NormedAddCommGroup β
          f g : α → β
          hf : MeasureTheory.Integrable f μ
          hg : MeasureTheory.Integrable g μ
          a : α
          ⊢ LE.le (↑(NNNorm.nnnorm (HAdd.hAdd (f a) (g a)))) (HAdd.hAdd ↑(NNNorm.nnnorm  …
        -/
        exact mod_cast nnnorm_add_le _ _
        /-
          🎉 no goals
        -/
    _ = _ := lintegral_nnnorm_add_left hf.aestronglyMeasurable _
    _ < ∞ := add_lt_top.2 ⟨hf.hasFiniteIntegral, hg.hasFiniteIntegral⟩


theorem Integrable.add {f g : α → β} (hf : Integrable f μ) (hg : Integrable g μ) :
    Integrable (f + g) μ :=
  ⟨hf.aestronglyMeasurable.add hg.aestronglyMeasurable, hf.add' hg⟩


theorem integrable_finset_sum' {ι} (s : Finset ι) {f : ι → α → β}
    (hf : ∀ i ∈ s, Integrable (f i) μ) : Integrable (∑ i ∈ s, f i) μ :=
  Finset.sum_induction f (fun g => Integrable g μ) (fun _ _ => Integrable.add)
    (integrable_zero _ _ _) hf


theorem integrable_finset_sum {ι} (s : Finset ι) {f : ι → α → β}
    (hf : ∀ i ∈ s, Integrable (f i) μ) : Integrable (fun a => ∑ i ∈ s, f i a) μ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    ι : Type u_6
    s : Finset ι
    f : ι → α → β
    hf : ∀ (i : ι), Membership.mem s i → MeasureTheory.Integrable (f i) μ
    ⊢ MeasureTheory.Integrable (fun a => s.sum fun i => f i a) μ
  -/
  simpa only [← Finset.sum_apply] using integrable_finset_sum' s hf
  /-
    🎉 no goals
  -/


/-- If `f` is integrable, then so is `-f`.
See `Integrable.neg'` for the same statement, but formulated with `x ↦ - f x` instead of `-f`. -/
theorem Integrable.neg {f : α → β} (hf : Integrable f μ) : Integrable (-f) μ :=
  ⟨hf.aestronglyMeasurable.neg, hf.hasFiniteIntegral.neg⟩


/-- If `f` is integrable, then so is `fun x ↦ - f x`.
See `Integrable.neg` for the same statement, but formulated with `-f` instead of `fun x ↦ - f x`. -/
theorem Integrable.neg' {f : α → β} (hf : Integrable f μ) : Integrable (fun x ↦ - f x) μ :=
  ⟨hf.aestronglyMeasurable.neg, hf.hasFiniteIntegral.neg⟩


@[simp]
theorem integrable_neg_iff {f : α → β} : Integrable (-f) μ ↔ Integrable f μ :=
  ⟨fun h => neg_neg f ▸ h.neg, Integrable.neg⟩


/-- if `f` is integrable, then `f + g` is integrable iff `g` is.
See `integrable_add_iff_integrable_right'` for the same statement with `fun x ↦ f x + g x` instead
of `f + g`. -/
@[simp]
lemma integrable_add_iff_integrable_right {f g : α → β} (hf : Integrable f μ) :
    Integrable (f + g) μ ↔ Integrable g μ :=
                                    /-
                                      α : Type u_1
                                      β : Type u_2
                                      m : MeasurableSpace α
                                      μ : MeasureTheory.Measure α
                                      inst✝ : NormedAddCommGroup β
                                      f g : α → β
                                      hf : MeasureTheory.Integrable f μ
                                      h : MeasureTheory.Integrable (HAdd.hAdd f g) μ
                                      ⊢ Eq g (HAdd.hAdd (HAdd.hAdd f g) (Neg.neg f))
                                    -/
  ⟨fun h ↦ show g = f + g + (-f) by simp only [add_neg_cancel_comm] ▸ h.add hf.neg,
                                    /-
                                      🎉 no goals
                                    -/
    fun h ↦ hf.add h⟩


/-- if `f` is integrable, then `fun x ↦ f x + g x` is integrable iff `g` is.
See `integrable_add_iff_integrable_right` for the same statement with `f + g` instead
of `fun x ↦ f x + g x`. -/
@[simp]
lemma integrable_add_iff_integrable_right' {f g : α → β} (hf : Integrable f μ) :
    Integrable (fun x ↦ f x + g x) μ ↔ Integrable g μ :=
  integrable_add_iff_integrable_right hf


/-- if `f` is integrable, then `g + f` is integrable iff `g` is.
See `integrable_add_iff_integrable_left'` for the same statement with `fun x ↦ g x + f x` instead
of `g + f`. -/
@[simp]
lemma integrable_add_iff_integrable_left {f g : α → β} (hf : Integrable f μ) :
    Integrable (g + f) μ ↔ Integrable g μ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f g : α → β
    hf : MeasureTheory.Integrable f μ
    ⊢ Iff (MeasureTheory.Integrable (HAdd.hAdd g f) μ) (MeasureTheory.Integrable g …
  -/
  rw [add_comm, integrable_add_iff_integrable_right hf]
  /-
    🎉 no goals
  -/


/-- if `f` is integrable, then `fun x ↦ g x + f x` is integrable iff `g` is.
See `integrable_add_iff_integrable_left'` for the same statement with `g + f` instead
of `fun x ↦ g x + f x`. -/
@[simp]
lemma integrable_add_iff_integrable_left' {f g : α → β} (hf : Integrable f μ) :
    Integrable (fun x ↦ g x + f x) μ ↔ Integrable g μ :=
  integrable_add_iff_integrable_left hf


lemma integrable_left_of_integrable_add_of_nonneg {f g : α → ℝ}
    (h_meas : AEStronglyMeasurable f μ) (hf : 0 ≤ᵐ[μ] f) (hg : 0 ≤ᵐ[μ] g)
    (h_int : Integrable (f + g) μ) : Integrable f μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → Real
    h_meas : MeasureTheory.AEStronglyMeasurable f μ
    hf : (MeasureTheory.ae μ).EventuallyLE 0 f
    hg : (MeasureTheory.ae μ).EventuallyLE 0 g
    h_int : MeasureTheory.Integrable (HAdd.hAdd f g) μ
    ⊢ MeasureTheory.Integrable f μ
  -/
  refine h_int.mono' h_meas ?_
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → Real
    h_meas : MeasureTheory.AEStronglyMeasurable f μ
    hf : (MeasureTheory.ae μ).EventuallyLE 0 f
    hg : (MeasureTheory.ae μ).EventuallyLE 0 g
    h_int : MeasureTheory.Integrable (HAdd.hAdd f g) μ
    ⊢ Filter.Eventually (fun a => LE.le (Norm.norm (f a)) (HAdd.hAdd f g a)) (Meas …
  -/
  filter_upwards [hf, hg] with a haf hag
  /-
    case h
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → Real
    h_meas : MeasureTheory.AEStronglyMeasurable f μ
    hf : (MeasureTheory.ae μ).EventuallyLE 0 f
    hg : (MeasureTheory.ae μ).EventuallyLE 0 g
    h_int : MeasureTheory.Integrable (HAdd.hAdd f g) μ
    a : α
    haf : LE.le (0 a) (f a)
    hag : LE.le (0 a) (g a)
    ⊢ LE.le (Norm.norm (f a)) (HAdd.hAdd f g a)
  -/
  exact (Real.norm_of_nonneg haf).symm ▸ le_add_of_nonneg_right hag
  /-
    🎉 no goals
  -/


lemma integrable_right_of_integrable_add_of_nonneg {f g : α → ℝ}
    (h_meas : AEStronglyMeasurable f μ) (hf : 0 ≤ᵐ[μ] f) (hg : 0 ≤ᵐ[μ] g)
    (h_int : Integrable (f + g) μ) : Integrable g μ :=
  integrable_left_of_integrable_add_of_nonneg
    ((AEStronglyMeasurable.add_iff_right h_meas).mp h_int.aestronglyMeasurable)
      hg hf (add_comm f g ▸ h_int)


lemma integrable_add_iff_of_nonneg {f g : α → ℝ} (h_meas : AEStronglyMeasurable f μ)
    (hf : 0 ≤ᵐ[μ] f) (hg : 0 ≤ᵐ[μ] g) :
    Integrable (f + g) μ ↔ Integrable f μ ∧ Integrable g μ :=
  ⟨fun h ↦ ⟨integrable_left_of_integrable_add_of_nonneg h_meas hf hg h,
    integrable_right_of_integrable_add_of_nonneg h_meas hf hg h⟩, fun ⟨hf, hg⟩ ↦ hf.add hg⟩


lemma integrable_add_iff_of_nonpos {f g : α → ℝ} (h_meas : AEStronglyMeasurable f μ)
    (hf : f ≤ᵐ[μ] 0) (hg : g ≤ᵐ[μ] 0) :
    Integrable (f + g) μ ↔ Integrable f μ ∧ Integrable g μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → Real
    h_meas : MeasureTheory.AEStronglyMeasurable f μ
    hf : (MeasureTheory.ae μ).EventuallyLE f 0
    hg : (MeasureTheory.ae μ).EventuallyLE g 0
    ⊢ Iff (MeasureTheory.Integrable (HAdd.hAdd f g) μ) (And (MeasureTheory.Integra …
  -/
  rw [← integrable_neg_iff, ← integrable_neg_iff (f := f), ← integrable_neg_iff (f := g), neg_add]
  exact integrable_add_iff_of_nonneg h_meas.neg (hf.mono (fun _ ↦ neg_nonneg_of_nonpos))
    (hg.mono (fun _ ↦ neg_nonneg_of_nonpos))


lemma integrable_add_const_iff [IsFiniteMeasure μ] {f : α → β} {c : β} :
    Integrable (fun x ↦ f x + c) μ ↔ Integrable f μ :=
  integrable_add_iff_integrable_left (integrable_const _)


lemma integrable_const_add_iff [IsFiniteMeasure μ] {f : α → β} {c : β} :
    Integrable (fun x ↦ c + f x) μ ↔ Integrable f μ :=
  integrable_add_iff_integrable_right (integrable_const _)


theorem Integrable.sub {f g : α → β} (hf : Integrable f μ) (hg : Integrable g μ) :
                               /-
                                 α : Type u_1
                                 β : Type u_2
                                 m : MeasurableSpace α
                                 μ : MeasureTheory.Measure α
                                 inst✝ : NormedAddCommGroup β
                                 f g : α → β
                                 hf : MeasureTheory.Integrable f μ
                                 hg : MeasureTheory.Integrable g μ
                                 ⊢ MeasureTheory.Integrable (HSub.hSub f g) μ
                               -/
    Integrable (f - g) μ := by simpa only [sub_eq_add_neg] using hf.add hg.neg
                               /-
                                 🎉 no goals
                               -/


theorem Integrable.norm {f : α → β} (hf : Integrable f μ) : Integrable (fun a => ‖f a‖) μ :=
  ⟨hf.aestronglyMeasurable.norm, hf.hasFiniteIntegral.norm⟩


theorem Integrable.inf {β} [NormedLatticeAddCommGroup β] {f g : α → β} (hf : Integrable f μ)
    (hg : Integrable g μ) : Integrable (f ⊓ g) μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_6
    inst✝ : NormedLatticeAddCommGroup β
    f g : α → β
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g μ
    ⊢ MeasureTheory.Integrable (Min.min f g) μ
  -/
  rw [← memℒp_one_iff_integrable] at hf hg ⊢
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_6
    inst✝ : NormedLatticeAddCommGroup β
    f g : α → β
    hf : MeasureTheory.Memℒp f 1 μ
    hg : MeasureTheory.Memℒp g 1 μ
    ⊢ MeasureTheory.Memℒp (Min.min f g) 1 μ
  -/
  exact hf.inf hg
  /-
    🎉 no goals
  -/


theorem Integrable.sup {β} [NormedLatticeAddCommGroup β] {f g : α → β} (hf : Integrable f μ)
    (hg : Integrable g μ) : Integrable (f ⊔ g) μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_6
    inst✝ : NormedLatticeAddCommGroup β
    f g : α → β
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g μ
    ⊢ MeasureTheory.Integrable (Max.max f g) μ
  -/
  rw [← memℒp_one_iff_integrable] at hf hg ⊢
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_6
    inst✝ : NormedLatticeAddCommGroup β
    f g : α → β
    hf : MeasureTheory.Memℒp f 1 μ
    hg : MeasureTheory.Memℒp g 1 μ
    ⊢ MeasureTheory.Memℒp (Max.max f g) 1 μ
  -/
  exact hf.sup hg
  /-
    🎉 no goals
  -/


theorem Integrable.abs {β} [NormedLatticeAddCommGroup β] {f : α → β} (hf : Integrable f μ) :
    Integrable (fun a => |f a|) μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_6
    inst✝ : NormedLatticeAddCommGroup β
    f : α → β
    hf : MeasureTheory.Integrable f μ
    ⊢ MeasureTheory.Integrable (fun a => _root_.abs (f a)) μ
  -/
  rw [← memℒp_one_iff_integrable] at hf ⊢
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_6
    inst✝ : NormedLatticeAddCommGroup β
    f : α → β
    hf : MeasureTheory.Memℒp f 1 μ
    ⊢ MeasureTheory.Memℒp (fun a => _root_.abs (f a)) 1 μ
  -/
  exact hf.abs
  /-
    🎉 no goals
  -/


theorem Integrable.bdd_mul {F : Type*} [NormedDivisionRing F] {f g : α → F} (hint : Integrable g μ)
    (hm : AEStronglyMeasurable f μ) (hfbdd : ∃ C, ∀ x, ‖f x‖ ≤ C) :
    Integrable (fun x => f x * g x) μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    F : Type u_6
    inst✝ : NormedDivisionRing F
    f g : α → F
    hint : MeasureTheory.Integrable g μ
    hm : MeasureTheory.AEStronglyMeasurable f μ
    hfbdd : Exists fun C => ∀ (x : α), LE.le (Norm.norm (f x)) C
    ⊢ MeasureTheory.Integrable (fun x => HMul.hMul (f x) (g x)) μ
  -/
  cases' isEmpty_or_nonempty α with hα hα
    /-
      case inl
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      F : Type u_6
      inst✝ : NormedDivisionRing F
      f g : α → F
      hint : MeasureTheory.Integrable g μ
      hm : MeasureTheory.AEStronglyMeasurable f μ
      hfbdd : Exists fun C => ∀ (x : α), LE.le (Norm.norm (f x)) C
      hα : IsEmpty α
      ⊢ MeasureTheory.Integrable (fun x => HMul.hMul (f x) (g x)) μ
    -/
  · rw [μ.eq_zero_of_isEmpty]
    /-
      case inl
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      F : Type u_6
      inst✝ : NormedDivisionRing F
      f g : α → F
      hint : MeasureTheory.Integrable g μ
      hm : MeasureTheory.AEStronglyMeasurable f μ
      hfbdd : Exists fun C => ∀ (x : α), LE.le (Norm.norm (f x)) C
      hα : IsEmpty α
      ⊢ MeasureTheory.Integrable (fun x => HMul.hMul (f x) (g x)) 0
    -/
    exact integrable_zero_measure
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      F : Type u_6
      inst✝ : NormedDivisionRing F
      f g : α → F
      hint : MeasureTheory.Integrable g μ
      hm : MeasureTheory.AEStronglyMeasurable f μ
      hfbdd : Exists fun C => ∀ (x : α), LE.le (Norm.norm (f x)) C
      hα : Nonempty α
      ⊢ MeasureTheory.Integrable (fun x => HMul.hMul (f x) (g x)) μ
    -/
  · refine ⟨hm.mul hint.1, ?_⟩
    /-
      case inr
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      F : Type u_6
      inst✝ : NormedDivisionRing F
      f g : α → F
      hint : MeasureTheory.Integrable g μ
      hm : MeasureTheory.AEStronglyMeasurable f μ
      hfbdd : Exists fun C => ∀ (x : α), LE.le (Norm.norm (f x)) C
      hα : Nonempty α
      ⊢ MeasureTheory.HasFiniteIntegral (fun x => HMul.hMul (f x) (g x)) μ
    -/
    obtain ⟨C, hC⟩ := hfbdd
    /-
      case inr.intro
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      F : Type u_6
      inst✝ : NormedDivisionRing F
      f g : α → F
      hint : MeasureTheory.Integrable g μ
      hm : MeasureTheory.AEStronglyMeasurable f μ
      hα : Nonempty α
      C : Real
      hC : ∀ (x : α), LE.le (Norm.norm (f x)) C
      ⊢ MeasureTheory.HasFiniteIntegral (fun x => HMul.hMul (f x) (g x)) μ
    -/
    have hCnonneg : 0 ≤ C := le_trans (norm_nonneg _) (hC hα.some)
    have : (fun x => ‖f x * g x‖₊) ≤ fun x => ⟨C, hCnonneg⟩ * ‖g x‖₊ := by
      intro x
      simp only [nnnorm_mul]
      exact mul_le_mul_of_nonneg_right (hC x) (zero_le _)
    /-
      case inr.intro
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      F : Type u_6
      inst✝ : NormedDivisionRing F
      f g : α → F
      hint : MeasureTheory.Integrable g μ
      hm : MeasureTheory.AEStronglyMeasurable f μ
      hα : Nonempty α
      C : Real
      hC : ∀ (x : α), LE.le (Norm.norm (f x)) C
      hCnonneg : LE.le 0 C
      this : LE.le (fun x => NNNorm.nnnorm (HMul.hMul (f x) (g x))) fun x => HMul.hM …
      ⊢ MeasureTheory.HasFiniteIntegral (fun x => HMul.hMul (f x) (g x)) μ
    -/
    refine lt_of_le_of_lt (lintegral_mono_nnreal this) ?_
    /-
      case inr.intro
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      F : Type u_6
      inst✝ : NormedDivisionRing F
      f g : α → F
      hint : MeasureTheory.Integrable g μ
      hm : MeasureTheory.AEStronglyMeasurable f μ
      hα : Nonempty α
      C : Real
      hC : ∀ (x : α), LE.le (Norm.norm (f x)) C
      hCnonneg : LE.le 0 C
      this : LE.le (fun x => NNNorm.nnnorm (HMul.hMul (f x) (g x))) fun x => HMul.hM …
      ⊢ LT.lt (MeasureTheory.lintegral μ fun a => ↑(HMul.hMul ⟨C, hCnonneg⟩ (NNNorm. …
    -/
    simp only [ENNReal.coe_mul]
    /-
      case inr.intro
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      F : Type u_6
      inst✝ : NormedDivisionRing F
      f g : α → F
      hint : MeasureTheory.Integrable g μ
      hm : MeasureTheory.AEStronglyMeasurable f μ
      hα : Nonempty α
      C : Real
      hC : ∀ (x : α), LE.le (Norm.norm (f x)) C
      hCnonneg : LE.le 0 C
      this : LE.le (fun x => NNNorm.nnnorm (HMul.hMul (f x) (g x))) fun x => HMul.hM …
      ⊢ LT.lt (MeasureTheory.lintegral μ fun a => HMul.hMul ↑⟨C, hCnonneg⟩ ↑(NNNorm. …
    -/
    rw [lintegral_const_mul' _ _ ENNReal.coe_ne_top]
    /-
      case inr.intro
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      F : Type u_6
      inst✝ : NormedDivisionRing F
      f g : α → F
      hint : MeasureTheory.Integrable g μ
      hm : MeasureTheory.AEStronglyMeasurable f μ
      hα : Nonempty α
      C : Real
      hC : ∀ (x : α), LE.le (Norm.norm (f x)) C
      hCnonneg : LE.le 0 C
      this : LE.le (fun x => NNNorm.nnnorm (HMul.hMul (f x) (g x))) fun x => HMul.hM …
      ⊢ LT.lt (HMul.hMul (↑⟨C, hCnonneg⟩) (MeasureTheory.lintegral μ fun a => ↑(NNNo …
    -/
    exact ENNReal.mul_lt_top ENNReal.coe_lt_top hint.2
    /-
      🎉 no goals
    -/


/-- **Hölder's inequality for integrable functions**: the scalar multiplication of an integrable
vector-valued function by a scalar function with finite essential supremum is integrable. -/
theorem Integrable.essSup_smul {𝕜 : Type*} [NormedField 𝕜] [NormedSpace 𝕜 β] {f : α → β}
    (hf : Integrable f μ) {g : α → 𝕜} (g_aestronglyMeasurable : AEStronglyMeasurable g μ)
    (ess_sup_g : essSup (fun x => (‖g x‖₊ : ℝ≥0∞)) μ ≠ ∞) :
    Integrable (fun x : α => g x • f x) μ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup β
    𝕜 : Type u_6
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedSpace 𝕜 β
    f : α → β
    hf : MeasureTheory.Integrable f μ
    g : α → 𝕜
    g_aestronglyMeasurable : MeasureTheory.AEStronglyMeasurable g μ
    ess_sup_g : Ne (essSup (fun x => ↑(NNNorm.nnnorm (g x))) μ) Top.top
    ⊢ MeasureTheory.Integrable (fun x => HSMul.hSMul (g x) (f x)) μ
  -/
  rw [← memℒp_one_iff_integrable] at *
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup β
    𝕜 : Type u_6
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedSpace 𝕜 β
    f : α → β
    hf : MeasureTheory.Memℒp f 1 μ
    g : α → 𝕜
    g_aestronglyMeasurable : MeasureTheory.AEStronglyMeasurable g μ
    ess_sup_g : Ne (essSup (fun x => ↑(NNNorm.nnnorm (g x))) μ) Top.top
    ⊢ MeasureTheory.Memℒp (fun x => HSMul.hSMul (g x) (f x)) 1 μ
  -/
  refine ⟨g_aestronglyMeasurable.smul hf.1, ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup β
    𝕜 : Type u_6
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedSpace 𝕜 β
    f : α → β
    hf : MeasureTheory.Memℒp f 1 μ
    g : α → 𝕜
    g_aestronglyMeasurable : MeasureTheory.AEStronglyMeasurable g μ
    ess_sup_g : Ne (essSup (fun x => ↑(NNNorm.nnnorm (g x))) μ) Top.top
    ⊢ LT.lt (MeasureTheory.eLpNorm (fun x => HSMul.hSMul (g x) (f x)) 1 μ) Top.top
  -/
  have h : (1 : ℝ≥0∞) / 1 = 1 / ∞ + 1 / 1 := by norm_num
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : NormedAddCommGroup β
    𝕜 : Type u_6
    inst✝¹ : NormedField 𝕜
    inst✝ : NormedSpace 𝕜 β
    f : α → β
    hf : MeasureTheory.Memℒp f 1 μ
    g : α → 𝕜
    g_aestronglyMeasurable : MeasureTheory.AEStronglyMeasurable g μ
    ess_sup_g : Ne (essSup (fun x => ↑(NNNorm.nnnorm (g x))) μ) Top.top
    h : Eq (1 / 1) (HAdd.hAdd (HDiv.hDiv 1 Top.top) (1 / 1))
    ⊢ LT.lt (MeasureTheory.eLpNorm (fun x => HSMul.hSMul (g x) (f x)) 1 μ) Top.top
  -/
  have hg' : eLpNorm g ∞ μ ≠ ∞ := by rwa [eLpNorm_exponent_top]
  calc
    eLpNorm (fun x : α => g x • f x) 1 μ ≤ _ := by
      simpa using MeasureTheory.eLpNorm_smul_le_mul_eLpNorm hf.1 g_aestronglyMeasurable h
    _ < ∞ := ENNReal.mul_lt_top hg'.lt_top hf.2


/-- Hölder's inequality for integrable functions: the scalar multiplication of an integrable
scalar-valued function by a vector-value function with finite essential supremum is integrable. -/
theorem Integrable.smul_essSup {𝕜 : Type*} [NormedRing 𝕜] [Module 𝕜 β] [BoundedSMul 𝕜 β]
    {f : α → 𝕜} (hf : Integrable f μ) {g : α → β}
    (g_aestronglyMeasurable : AEStronglyMeasurable g μ)
    (ess_sup_g : essSup (fun x => (‖g x‖₊ : ℝ≥0∞)) μ ≠ ∞) :
    Integrable (fun x : α => f x • g x) μ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : NormedAddCommGroup β
    𝕜 : Type u_6
    inst✝² : NormedRing 𝕜
    inst✝¹ : Module 𝕜 β
    inst✝ : BoundedSMul 𝕜 β
    f : α → 𝕜
    hf : MeasureTheory.Integrable f μ
    g : α → β
    g_aestronglyMeasurable : MeasureTheory.AEStronglyMeasurable g μ
    ess_sup_g : Ne (essSup (fun x => ↑(NNNorm.nnnorm (g x))) μ) Top.top
    ⊢ MeasureTheory.Integrable (fun x => HSMul.hSMul (f x) (g x)) μ
  -/
  rw [← memℒp_one_iff_integrable] at *
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : NormedAddCommGroup β
    𝕜 : Type u_6
    inst✝² : NormedRing 𝕜
    inst✝¹ : Module 𝕜 β
    inst✝ : BoundedSMul 𝕜 β
    f : α → 𝕜
    hf : MeasureTheory.Memℒp f 1 μ
    g : α → β
    g_aestronglyMeasurable : MeasureTheory.AEStronglyMeasurable g μ
    ess_sup_g : Ne (essSup (fun x => ↑(NNNorm.nnnorm (g x))) μ) Top.top
    ⊢ MeasureTheory.Memℒp (fun x => HSMul.hSMul (f x) (g x)) 1 μ
  -/
  refine ⟨hf.1.smul g_aestronglyMeasurable, ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : NormedAddCommGroup β
    𝕜 : Type u_6
    inst✝² : NormedRing 𝕜
    inst✝¹ : Module 𝕜 β
    inst✝ : BoundedSMul 𝕜 β
    f : α → 𝕜
    hf : MeasureTheory.Memℒp f 1 μ
    g : α → β
    g_aestronglyMeasurable : MeasureTheory.AEStronglyMeasurable g μ
    ess_sup_g : Ne (essSup (fun x => ↑(NNNorm.nnnorm (g x))) μ) Top.top
    ⊢ LT.lt (MeasureTheory.eLpNorm (fun x => HSMul.hSMul (f x) (g x)) 1 μ) Top.top
  -/
  have h : (1 : ℝ≥0∞) / 1 = 1 / 1 + 1 / ∞ := by norm_num
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : NormedAddCommGroup β
    𝕜 : Type u_6
    inst✝² : NormedRing 𝕜
    inst✝¹ : Module 𝕜 β
    inst✝ : BoundedSMul 𝕜 β
    f : α → 𝕜
    hf : MeasureTheory.Memℒp f 1 μ
    g : α → β
    g_aestronglyMeasurable : MeasureTheory.AEStronglyMeasurable g μ
    ess_sup_g : Ne (essSup (fun x => ↑(NNNorm.nnnorm (g x))) μ) Top.top
    h : Eq (1 / 1) (HAdd.hAdd (1 / 1) (HDiv.hDiv 1 Top.top))
    ⊢ LT.lt (MeasureTheory.eLpNorm (fun x => HSMul.hSMul (f x) (g x)) 1 μ) Top.top
  -/
  have hg' : eLpNorm g ∞ μ ≠ ∞ := by rwa [eLpNorm_exponent_top]
  calc
    eLpNorm (fun x : α => f x • g x) 1 μ ≤ _ := by
      simpa using MeasureTheory.eLpNorm_smul_le_mul_eLpNorm g_aestronglyMeasurable hf.1 h
    _ < ∞ := ENNReal.mul_lt_top hf.2 hg'.lt_top


theorem integrable_norm_iff {f : α → β} (hf : AEStronglyMeasurable f μ) :
    Integrable (fun a => ‖f a‖) μ ↔ Integrable f μ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    hf : MeasureTheory.AEStronglyMeasurable f μ
    ⊢ Iff (MeasureTheory.Integrable (fun a => Norm.norm (f a)) μ) (MeasureTheory.I …
  -/
  simp_rw [Integrable, and_iff_right hf, and_iff_right hf.norm, hasFiniteIntegral_norm_iff]
  /-
    🎉 no goals
  -/


theorem integrable_of_norm_sub_le {f₀ f₁ : α → β} {g : α → ℝ} (hf₁_m : AEStronglyMeasurable f₁ μ)
    (hf₀_i : Integrable f₀ μ) (hg_i : Integrable g μ) (h : ∀ᵐ a ∂μ, ‖f₀ a - f₁ a‖ ≤ g a) :
    Integrable f₁ μ :=
  haveI : ∀ᵐ a ∂μ, ‖f₁ a‖ ≤ ‖f₀ a‖ + g a := by
    /-
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      f₀ f₁ : α → β
      g : α → Real
      hf₁_m : MeasureTheory.AEStronglyMeasurable f₁ μ
      hf₀_i : MeasureTheory.Integrable f₀ μ
      hg_i : MeasureTheory.Integrable g μ
      h : Filter.Eventually (fun a => LE.le (Norm.norm (HSub.hSub (f₀ a) (f₁ a))) (g …
      ⊢ Filter.Eventually (fun a => LE.le (Norm.norm (f₁ a)) (HAdd.hAdd (Norm.norm ( …
    -/
    apply h.mono
    /-
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      f₀ f₁ : α → β
      g : α → Real
      hf₁_m : MeasureTheory.AEStronglyMeasurable f₁ μ
      hf₀_i : MeasureTheory.Integrable f₀ μ
      hg_i : MeasureTheory.Integrable g μ
      h : Filter.Eventually (fun a => LE.le (Norm.norm (HSub.hSub (f₀ a) (f₁ a))) (g …
      ⊢ ∀ (x : α), LE.le (Norm.norm (HSub.hSub (f₀ x) (f₁ x))) (g x) → LE.le (Norm.n …
    -/
    intro a ha
    calc
      ‖f₁ a‖ ≤ ‖f₀ a‖ + ‖f₀ a - f₁ a‖ := norm_le_insert _ _
      _ ≤ ‖f₀ a‖ + g a := add_le_add_left ha _
  Integrable.mono' (hf₀_i.norm.add hg_i) hf₁_m this


lemma integrable_of_le_of_le {f g₁ g₂ : α → ℝ} (hf : AEStronglyMeasurable f μ)
    (h_le₁ : g₁ ≤ᵐ[μ] f) (h_le₂ : f ≤ᵐ[μ] g₂)
    (h_int₁ : Integrable g₁ μ) (h_int₂ : Integrable g₂ μ) :
    Integrable f μ := by
  have : ∀ᵐ x ∂μ, ‖f x‖ ≤ max ‖g₁ x‖ ‖g₂ x‖ := by
    filter_upwards [h_le₁, h_le₂] with x hx1 hx2
    simp only [Real.norm_eq_abs]
    exact abs_le_max_abs_abs hx1 hx2
  have h_le_add : ∀ᵐ x ∂μ, ‖f x‖ ≤ ‖‖g₁ x‖ + ‖g₂ x‖‖ := by
    filter_upwards [this] with x hx
    refine hx.trans ?_
    conv_rhs => rw [Real.norm_of_nonneg (by positivity)]
    exact max_le_add_of_nonneg (norm_nonneg _) (norm_nonneg _)
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g₁ g₂ : α → Real
    hf : MeasureTheory.AEStronglyMeasurable f μ
    h_le₁ : (MeasureTheory.ae μ).EventuallyLE g₁ f
    h_le₂ : (MeasureTheory.ae μ).EventuallyLE f g₂
    h_int₁ : MeasureTheory.Integrable g₁ μ
    h_int₂ : MeasureTheory.Integrable g₂ μ
    this : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (Max.max (Norm.norm …
    h_le_add : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) (Norm.norm (HAd …
    ⊢ MeasureTheory.Integrable f μ
  -/
  exact Integrable.mono (h_int₁.norm.add h_int₂.norm) hf h_le_add
  /-
    🎉 no goals
  -/


theorem Integrable.prod_mk {f : α → β} {g : α → γ} (hf : Integrable f μ) (hg : Integrable g μ) :
    Integrable (fun x => (f x, g x)) μ :=
  ⟨hf.aestronglyMeasurable.prod_mk hg.aestronglyMeasurable,
    (hf.norm.add' hg.norm).mono <|
      Eventually.of_forall fun x =>
        calc
          max ‖f x‖ ‖g x‖ ≤ ‖f x‖ + ‖g x‖ := max_le_add_of_nonneg (norm_nonneg _) (norm_nonneg _)
          _ ≤ ‖‖f x‖ + ‖g x‖‖ := le_abs_self _⟩


theorem Memℒp.integrable {q : ℝ≥0∞} (hq1 : 1 ≤ q) {f : α → β} [IsFiniteMeasure μ]
    (hfq : Memℒp f q μ) : Integrable f μ :=
  memℒp_one_iff_integrable.mp (hfq.memℒp_of_exponent_le hq1)


/-- A non-quantitative version of Markov inequality for integrable functions: the measure of points
where `‖f x‖ ≥ ε` is finite for all positive `ε`. -/
theorem Integrable.measure_norm_ge_lt_top {f : α → β} (hf : Integrable f μ) {ε : ℝ} (hε : 0 < ε) :
    μ { x | ε ≤ ‖f x‖ } < ∞ := by
  rw [show { x | ε ≤ ‖f x‖ } = { x | ENNReal.ofReal ε ≤ ‖f x‖₊ } by
      simp only [ENNReal.ofReal, Real.toNNReal_le_iff_le_coe, ENNReal.coe_le_coe, coe_nnnorm]]
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    hf : MeasureTheory.Integrable f μ
    ε : Real
    hε : LT.lt 0 ε
    ⊢ LT.lt (μ (setOf fun x => LE.le (ENNReal.ofReal ε) ↑(NNNorm.nnnorm (f x)))) T …
  -/
  refine (meas_ge_le_mul_pow_eLpNorm μ one_ne_zero ENNReal.one_ne_top hf.1 ?_).trans_lt ?_
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup β
      f : α → β
      hf : MeasureTheory.Integrable f μ
      ε : Real
      hε : LT.lt 0 ε
      ⊢ Ne (ENNReal.ofReal ε) 0
    -/
  · simpa only [Ne, ENNReal.ofReal_eq_zero, not_le] using hε
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    hf : MeasureTheory.Integrable f μ
    ε : Real
    hε : LT.lt 0 ε
    ⊢ LT.lt (HMul.hMul (HPow.hPow (Inv.inv (ENNReal.ofReal ε)) (ENNReal.toReal 1)) …
  -/
  apply ENNReal.mul_lt_top
  · simpa only [ENNReal.one_toReal, ENNReal.rpow_one, ENNReal.inv_lt_top, ENNReal.ofReal_pos]
      using hε
  · simpa only [ENNReal.one_toReal, ENNReal.rpow_one] using
      (memℒp_one_iff_integrable.2 hf).eLpNorm_lt_top


/-- A non-quantitative version of Markov inequality for integrable functions: the measure of points
where `‖f x‖ > ε` is finite for all positive `ε`. -/
lemma Integrable.measure_norm_gt_lt_top {f : α → β} (hf : Integrable f μ) {ε : ℝ} (hε : 0 < ε) :
    μ {x | ε < ‖f x‖} < ∞ :=
  lt_of_le_of_lt (measure_mono (fun _ h ↦ (Set.mem_setOf_eq ▸ h).le)) (hf.measure_norm_ge_lt_top hε)


/-- If `f` is `ℝ`-valued and integrable, then for any `c > 0` the set `{x | f x ≥ c}` has finite
measure. -/
lemma Integrable.measure_ge_lt_top {f : α → ℝ} (hf : Integrable f μ) {ε : ℝ} (ε_pos : 0 < ε) :
    μ {a : α | ε ≤ f a} < ∞ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.Integrable f μ
    ε : Real
    ε_pos : LT.lt 0 ε
    ⊢ LT.lt (μ (setOf fun a => LE.le ε (f a))) Top.top
  -/
  refine lt_of_le_of_lt (measure_mono ?_) (hf.measure_norm_ge_lt_top ε_pos)
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.Integrable f μ
    ε : Real
    ε_pos : LT.lt 0 ε
    ⊢ HasSubset.Subset (setOf fun a => LE.le ε (f a)) (setOf fun x => LE.le ε (Nor …
  -/
  intro x hx
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.Integrable f μ
    ε : Real
    ε_pos : LT.lt 0 ε
    x : α
    hx : Membership.mem (setOf fun a => LE.le ε (f a)) x
    ⊢ Membership.mem (setOf fun x => LE.le ε (Norm.norm (f x))) x
  -/
  simp only [Real.norm_eq_abs, Set.mem_setOf_eq] at hx ⊢
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.Integrable f μ
    ε : Real
    ε_pos : LT.lt 0 ε
    x : α
    hx : LE.le ε (f x)
    ⊢ LE.le ε (_root_.abs (f x))
  -/
  exact hx.trans (le_abs_self _)
  /-
    🎉 no goals
  -/


/-- If `f` is `ℝ`-valued and integrable, then for any `c < 0` the set `{x | f x ≤ c}` has finite
measure. -/
lemma Integrable.measure_le_lt_top {f : α → ℝ} (hf : Integrable f μ) {c : ℝ} (c_neg : c < 0) :
    μ {a : α | f a ≤ c} < ∞ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.Integrable f μ
    c : Real
    c_neg : LT.lt c 0
    ⊢ LT.lt (μ (setOf fun a => LE.le (f a) c)) Top.top
  -/
  refine lt_of_le_of_lt (measure_mono ?_) (hf.measure_norm_ge_lt_top (show 0 < -c by linarith))
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.Integrable f μ
    c : Real
    c_neg : LT.lt c 0
    ⊢ HasSubset.Subset (setOf fun a => LE.le (f a) c) (setOf fun x => LE.le (Neg.n …
  -/
  intro x hx
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.Integrable f μ
    c : Real
    c_neg : LT.lt c 0
    x : α
    hx : Membership.mem (setOf fun a => LE.le (f a) c) x
    ⊢ Membership.mem (setOf fun x => LE.le (Neg.neg c) (Norm.norm (f x))) x
  -/
  simp only [Real.norm_eq_abs, Set.mem_setOf_eq] at hx ⊢
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.Integrable f μ
    c : Real
    c_neg : LT.lt c 0
    x : α
    hx : LE.le (f x) c
    ⊢ LE.le (Neg.neg c) (_root_.abs (f x))
  -/
  exact (show -c ≤ - f x by linarith).trans (neg_le_abs _)
  /-
    🎉 no goals
  -/


/-- If `f` is `ℝ`-valued and integrable, then for any `c > 0` the set `{x | f x > c}` has finite
measure. -/
lemma Integrable.measure_gt_lt_top {f : α → ℝ} (hf : Integrable f μ) {ε : ℝ} (ε_pos : 0 < ε) :
    μ {a : α | ε < f a} < ∞ :=
  lt_of_le_of_lt (measure_mono (fun _ hx ↦ (Set.mem_setOf_eq ▸ hx).le))
    (Integrable.measure_ge_lt_top hf ε_pos)


/-- If `f` is `ℝ`-valued and integrable, then for any `c < 0` the set `{x | f x < c}` has finite
measure. -/
lemma Integrable.measure_lt_lt_top {f : α → ℝ} (hf : Integrable f μ) {c : ℝ} (c_neg : c < 0) :
    μ {a : α | f a < c} < ∞ :=
  lt_of_le_of_lt (measure_mono (fun _ hx ↦ (Set.mem_setOf_eq ▸ hx).le))
    (Integrable.measure_le_lt_top hf c_neg)


theorem LipschitzWith.integrable_comp_iff_of_antilipschitz {K K'} {f : α → β} {g : β → γ}
    (hg : LipschitzWith K g) (hg' : AntilipschitzWith K' g) (g0 : g 0 = 0) :
    Integrable (g ∘ f) μ ↔ Integrable f μ := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup β
    inst✝ : NormedAddCommGroup γ
    K K' : NNReal
    f : α → β
    g : β → γ
    hg : LipschitzWith K g
    hg' : AntilipschitzWith K' g
    g0 : Eq (g 0) 0
    ⊢ Iff (MeasureTheory.Integrable (Function.comp g f) μ) (MeasureTheory.Integrab …
  -/
  simp [← memℒp_one_iff_integrable, hg.memℒp_comp_iff_of_antilipschitz hg' g0]
  /-
    🎉 no goals
  -/


theorem Integrable.real_toNNReal {f : α → ℝ} (hf : Integrable f μ) :
    Integrable (fun x => ((f x).toNNReal : ℝ)) μ := by
  refine
    ⟨hf.aestronglyMeasurable.aemeasurable.real_toNNReal.coe_nnreal_real.aestronglyMeasurable, ?_⟩
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.Integrable f μ
    ⊢ MeasureTheory.HasFiniteIntegral (fun x => ↑(f x).toNNReal) μ
  -/
  rw [hasFiniteIntegral_iff_norm]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.Integrable f μ
    ⊢ LT.lt (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (Norm.norm ↑(f a).t …
  -/
  refine lt_of_le_of_lt ?_ ((hasFiniteIntegral_iff_norm _).1 hf.hasFiniteIntegral)
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.Integrable f μ
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (Norm.norm ↑(f a).t …
  -/
  apply lintegral_mono
  /-
    case hfg
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.Integrable f μ
    ⊢ LE.le (fun a => ENNReal.ofReal (Norm.norm ↑(f a).toNNReal)) fun a => ENNReal …
  -/
  intro x
  /-
    case hfg
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.Integrable f μ
    x : α
    ⊢ LE.le ((fun a => ENNReal.ofReal (Norm.norm ↑(f a).toNNReal)) x) ((fun a => E …
  -/
  simp [ENNReal.ofReal_le_ofReal, abs_le, le_abs_self]
  /-
    🎉 no goals
  -/


theorem ofReal_toReal_ae_eq {f : α → ℝ≥0∞} (hf : ∀ᵐ x ∂μ, f x < ∞) :
    (fun x => ENNReal.ofReal (f x).toReal) =ᵐ[μ] f := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : Filter.Eventually (fun x => LT.lt (f x) Top.top) (MeasureTheory.ae μ)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun x => ENNReal.ofReal (f x).toReal) f
  -/
  filter_upwards [hf]
  /-
    case h
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : Filter.Eventually (fun x => LT.lt (f x) Top.top) (MeasureTheory.ae μ)
    ⊢ ∀ (a : α), LT.lt (f a) Top.top → Eq (ENNReal.ofReal (f a).toReal) (f a)
  -/
  intro x hx
  /-
    case h
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : Filter.Eventually (fun x => LT.lt (f x) Top.top) (MeasureTheory.ae μ)
    x : α
    hx : LT.lt (f x) Top.top
    ⊢ Eq (ENNReal.ofReal (f x).toReal) (f x)
  -/
  simp only [hx.ne, ofReal_toReal, Ne, not_false_iff]
  /-
    🎉 no goals
  -/


theorem coe_toNNReal_ae_eq {f : α → ℝ≥0∞} (hf : ∀ᵐ x ∂μ, f x < ∞) :
    (fun x => ((f x).toNNReal : ℝ≥0∞)) =ᵐ[μ] f := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : Filter.Eventually (fun x => LT.lt (f x) Top.top) (MeasureTheory.ae μ)
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun x => ↑(f x).toNNReal) f
  -/
  filter_upwards [hf]
  /-
    case h
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : Filter.Eventually (fun x => LT.lt (f x) Top.top) (MeasureTheory.ae μ)
    ⊢ ∀ (a : α), LT.lt (f a) Top.top → Eq (↑(f a).toNNReal) (f a)
  -/
  intro x hx
  /-
    case h
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : Filter.Eventually (fun x => LT.lt (f x) Top.top) (MeasureTheory.ae μ)
    x : α
    hx : LT.lt (f x) Top.top
    ⊢ Eq (↑(f x).toNNReal) (f x)
  -/
  simp only [hx.ne, Ne, not_false_iff, coe_toNNReal]
  /-
    🎉 no goals
  -/


/-- A function has finite integral for the counting measure iff its norm is summable. -/
lemma hasFiniteIntegral_count_iff :
    HasFiniteIntegral f Measure.count ↔ Summable (‖f ·‖) := by
  simp only [hasFiniteIntegral_iff_nnnorm, lintegral_count, lt_top_iff_ne_top,
    ENNReal.tsum_coe_ne_top_iff_summable,  ← NNReal.summable_coe, coe_nnnorm]


/-- A function is integrable for the counting measure iff its norm is summable. -/
lemma integrable_count_iff :
    Integrable f Measure.count ↔ Summable (‖f ·‖) := by
  -- Note: this proof would be much easier if we assumed `SecondCountableTopology G`. Without
  -- this we have to justify the claim that `f` lands a.e. in a separable subset, which is true
  -- (because summable functions have countable range) but slightly tedious to check.
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup β
    inst✝ : MeasurableSingletonClass α
    f : α → β
    ⊢ Iff (MeasureTheory.Integrable f MeasureTheory.Measure.count) (Summable fun x …
  -/
  rw [Integrable, hasFiniteIntegral_count_iff, and_iff_right_iff_imp]
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup β
    inst✝ : MeasurableSingletonClass α
    f : α → β
    ⊢ (Summable fun x => Norm.norm (f x)) → MeasureTheory.AEStronglyMeasurable f M …
  -/
  intro hs
  have hs' : (Function.support f).Countable := by
    simpa only [Ne, Pi.zero_apply, eq_comm, Function.support, norm_eq_zero]
      using hs.countable_support
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup β
    inst✝ : MeasurableSingletonClass α
    f : α → β
    hs : Summable fun x => Norm.norm (f x)
    hs' : (Function.support f).Countable
    ⊢ MeasureTheory.AEStronglyMeasurable f MeasureTheory.Measure.count
  -/
  letI : MeasurableSpace β := borel β
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup β
    inst✝ : MeasurableSingletonClass α
    f : α → β
    hs : Summable fun x => Norm.norm (f x)
    hs' : (Function.support f).Countable
    this : MeasurableSpace β := borel β
    ⊢ MeasureTheory.AEStronglyMeasurable f MeasureTheory.Measure.count
  -/
  haveI : BorelSpace β := ⟨rfl⟩
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝¹ : NormedAddCommGroup β
    inst✝ : MeasurableSingletonClass α
    f : α → β
    hs : Summable fun x => Norm.norm (f x)
    hs' : (Function.support f).Countable
    this✝ : MeasurableSpace β := borel β
    this : BorelSpace β
    ⊢ MeasureTheory.AEStronglyMeasurable f MeasureTheory.Measure.count
  -/
  refine aestronglyMeasurable_iff_aemeasurable_separable.mpr ⟨?_, ?_⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      inst✝¹ : NormedAddCommGroup β
      inst✝ : MeasurableSingletonClass α
      f : α → β
      hs : Summable fun x => Norm.norm (f x)
      hs' : (Function.support f).Countable
      this✝ : MeasurableSpace β := borel β
      this : BorelSpace β
      ⊢ AEMeasurable f MeasureTheory.Measure.count
    -/
  · refine (measurable_zero.measurable_of_countable_ne ?_).aemeasurable
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      inst✝¹ : NormedAddCommGroup β
      inst✝ : MeasurableSingletonClass α
      f : α → β
      hs : Summable fun x => Norm.norm (f x)
      hs' : (Function.support f).Countable
      this✝ : MeasurableSpace β := borel β
      this : BorelSpace β
      ⊢ (setOf fun x => Ne (0 x) (f x)).Countable
    -/
    simpa only [Ne, Pi.zero_apply, eq_comm, Function.support] using hs'
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      inst✝¹ : NormedAddCommGroup β
      inst✝ : MeasurableSingletonClass α
      f : α → β
      hs : Summable fun x => Norm.norm (f x)
      hs' : (Function.support f).Countable
      this✝ : MeasurableSpace β := borel β
      this : BorelSpace β
      ⊢ Exists fun t => And (TopologicalSpace.IsSeparable t) (Filter.Eventually (fun …
    -/
  · refine ⟨f '' univ, ?_, ae_of_all _ fun a ↦ ⟨a, ⟨mem_univ _, rfl⟩⟩⟩
    suffices f '' univ ⊆ (f '' f.support) ∪ {0} from
      (((hs'.image f).union (countable_singleton 0)).mono this).isSeparable
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      inst✝¹ : NormedAddCommGroup β
      inst✝ : MeasurableSingletonClass α
      f : α → β
      hs : Summable fun x => Norm.norm (f x)
      hs' : (Function.support f).Countable
      this✝ : MeasurableSpace β := borel β
      this : BorelSpace β
      ⊢ HasSubset.Subset (Set.image f Set.univ) (Union.union (Set.image f (Function. …
    -/
    intro g hg
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      inst✝¹ : NormedAddCommGroup β
      inst✝ : MeasurableSingletonClass α
      f : α → β
      hs : Summable fun x => Norm.norm (f x)
      hs' : (Function.support f).Countable
      this✝ : MeasurableSpace β := borel β
      this : BorelSpace β
      g : β
      hg : Membership.mem (Set.image f Set.univ) g
      ⊢ Membership.mem (Union.union (Set.image f (Function.support f)) (Singleton.si …
    -/
    rcases eq_or_ne g 0 with rfl | hg'
      /-
        case refine_2.inl
        α : Type u_1
        β : Type u_2
        m : MeasurableSpace α
        inst✝¹ : NormedAddCommGroup β
        inst✝ : MeasurableSingletonClass α
        f : α → β
        hs : Summable fun x => Norm.norm (f x)
        hs' : (Function.support f).Countable
        this✝ : MeasurableSpace β := borel β
        this : BorelSpace β
        hg : Membership.mem (Set.image f Set.univ) 0
        ⊢ Membership.mem (Union.union (Set.image f (Function.support f)) (Singleton.si …
      -/
    · exact Or.inr (mem_singleton _)
      /-
        🎉 no goals
      -/
      /-
        case refine_2.inr
        α : Type u_1
        β : Type u_2
        m : MeasurableSpace α
        inst✝¹ : NormedAddCommGroup β
        inst✝ : MeasurableSingletonClass α
        f : α → β
        hs : Summable fun x => Norm.norm (f x)
        hs' : (Function.support f).Countable
        this✝ : MeasurableSpace β := borel β
        this : BorelSpace β
        g : β
        hg : Membership.mem (Set.image f Set.univ) g
        hg' : Ne g 0
        ⊢ Membership.mem (Union.union (Set.image f (Function.support f)) (Singleton.si …
      -/
    · obtain ⟨x, -, rfl⟩ := (mem_image ..).mp hg
      /-
        case refine_2.inr.intro.intro
        α : Type u_1
        β : Type u_2
        m : MeasurableSpace α
        inst✝¹ : NormedAddCommGroup β
        inst✝ : MeasurableSingletonClass α
        f : α → β
        hs : Summable fun x => Norm.norm (f x)
        hs' : (Function.support f).Countable
        this✝ : MeasurableSpace β := borel β
        this : BorelSpace β
        x : α
        hg : Membership.mem (Set.image f Set.univ) (f x)
        hg' : Ne (f x) 0
        ⊢ Membership.mem (Union.union (Set.image f (Function.support f)) (Singleton.si …
      -/
      exact Or.inl ⟨x, hg', rfl⟩
      /-
        🎉 no goals
      -/


theorem integrable_withDensity_iff_integrable_coe_smul {f : α → ℝ≥0} (hf : Measurable f)
    {g : α → E} :
    Integrable g (μ.withDensity fun x => f x) ↔ Integrable (fun x => (f x : ℝ) • g x) μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_6
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : α → NNReal
    hf : Measurable f
    g : α → E
    ⊢ Iff (MeasureTheory.Integrable g (μ.withDensity fun x => ↑(f x))) (MeasureThe …
  -/
  by_cases H : AEStronglyMeasurable (fun x : α => (f x : ℝ) • g x) μ
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_6
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : α → NNReal
      hf : Measurable f
      g : α → E
      H : MeasureTheory.AEStronglyMeasurable (fun x => HSMul.hSMul (↑(f x)) (g x)) μ
      ⊢ Iff (MeasureTheory.Integrable g (μ.withDensity fun x => ↑(f x))) (MeasureThe …
    -/
  · simp only [Integrable, aestronglyMeasurable_withDensity_iff hf, hasFiniteIntegral_iff_nnnorm, H]
    /-
      case pos
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_6
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : α → NNReal
      hf : Measurable f
      g : α → E
      H : MeasureTheory.AEStronglyMeasurable (fun x => HSMul.hSMul (↑(f x)) (g x)) μ
      ⊢ Iff (And True (LT.lt (MeasureTheory.lintegral (μ.withDensity fun x => ↑(f x) …
    -/
    rw [lintegral_withDensity_eq_lintegral_mul₀' hf.coe_nnreal_ennreal.aemeasurable]
      /-
        case pos
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        E : Type u_6
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : α → NNReal
        hf : Measurable f
        g : α → E
        H : MeasureTheory.AEStronglyMeasurable (fun x => HSMul.hSMul (↑(f x)) (g x)) μ
        ⊢ Iff (And True (LT.lt (MeasureTheory.lintegral μ fun a => HMul.hMul (fun x => …
      -/
    · rw [iff_iff_eq]
      /-
        case pos
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        E : Type u_6
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : α → NNReal
        hf : Measurable f
        g : α → E
        H : MeasureTheory.AEStronglyMeasurable (fun x => HSMul.hSMul (↑(f x)) (g x)) μ
        ⊢ Eq (And True (LT.lt (MeasureTheory.lintegral μ fun a => HMul.hMul (fun x =>  …
      -/
      congr
      /-
        case pos.e_b.e_a.e_f
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        E : Type u_6
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : α → NNReal
        hf : Measurable f
        g : α → E
        H : MeasureTheory.AEStronglyMeasurable (fun x => HSMul.hSMul (↑(f x)) (g x)) μ
        ⊢ Eq (fun a => HMul.hMul (fun x => ↑(f x)) (fun a => ↑(NNNorm.nnnorm (g a))) a …
      -/
      ext1 x
      /-
        case pos.e_b.e_a.e_f.h
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        E : Type u_6
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : α → NNReal
        hf : Measurable f
        g : α → E
        H : MeasureTheory.AEStronglyMeasurable (fun x => HSMul.hSMul (↑(f x)) (g x)) μ
        x : α
        ⊢ Eq (HMul.hMul (fun x => ↑(f x)) (fun a => ↑(NNNorm.nnnorm (g a))) x) ↑(NNNor …
      -/
      simp only [nnnorm_smul, NNReal.nnnorm_eq, coe_mul, Pi.mul_apply]
      /-
        🎉 no goals
      -/
      /-
        case pos
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        E : Type u_6
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : α → NNReal
        hf : Measurable f
        g : α → E
        H : MeasureTheory.AEStronglyMeasurable (fun x => HSMul.hSMul (↑(f x)) (g x)) μ
        ⊢ AEMeasurable (fun a => ↑(NNNorm.nnnorm (g a))) (μ.withDensity fun x => ↑(f x))
      -/
    · rw [aemeasurable_withDensity_ennreal_iff hf]
      /-
        case pos
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        E : Type u_6
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : α → NNReal
        hf : Measurable f
        g : α → E
        H : MeasureTheory.AEStronglyMeasurable (fun x => HSMul.hSMul (↑(f x)) (g x)) μ
        ⊢ AEMeasurable (fun x => HMul.hMul ↑(f x) ↑(NNNorm.nnnorm (g x))) μ
      -/
      convert H.ennnorm using 1
      /-
        case h.e'_5
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        E : Type u_6
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : α → NNReal
        hf : Measurable f
        g : α → E
        H : MeasureTheory.AEStronglyMeasurable (fun x => HSMul.hSMul (↑(f x)) (g x)) μ
        ⊢ Eq (fun x => HMul.hMul ↑(f x) ↑(NNNorm.nnnorm (g x))) fun a => ↑(NNNorm.nnno …
      -/
      ext1 x
      /-
        case h.e'_5.h
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        E : Type u_6
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : α → NNReal
        hf : Measurable f
        g : α → E
        H : MeasureTheory.AEStronglyMeasurable (fun x => HSMul.hSMul (↑(f x)) (g x)) μ
        x : α
        ⊢ Eq (HMul.hMul ↑(f x) ↑(NNNorm.nnnorm (g x))) ↑(NNNorm.nnnorm (HSMul.hSMul (↑ …
      -/
      simp only [nnnorm_smul, NNReal.nnnorm_eq, coe_mul]
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_6
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : α → NNReal
      hf : Measurable f
      g : α → E
      H : Not (MeasureTheory.AEStronglyMeasurable (fun x => HSMul.hSMul (↑(f x)) (g  …
      ⊢ Iff (MeasureTheory.Integrable g (μ.withDensity fun x => ↑(f x))) (MeasureThe …
    -/
  · simp only [Integrable, aestronglyMeasurable_withDensity_iff hf, H, false_and]
    /-
      🎉 no goals
    -/


theorem integrable_withDensity_iff_integrable_smul {f : α → ℝ≥0} (hf : Measurable f) {g : α → E} :
    Integrable g (μ.withDensity fun x => f x) ↔ Integrable (fun x => f x • g x) μ :=
  integrable_withDensity_iff_integrable_coe_smul hf


theorem integrable_withDensity_iff_integrable_smul' {f : α → ℝ≥0∞} (hf : Measurable f)
    (hflt : ∀ᵐ x ∂μ, f x < ∞) {g : α → E} :
    Integrable g (μ.withDensity f) ↔ Integrable (fun x => (f x).toReal • g x) μ := by
  rw [← withDensity_congr_ae (coe_toNNReal_ae_eq hflt),
    integrable_withDensity_iff_integrable_smul]
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_6
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : α → ENNReal
      hf : Measurable f
      hflt : Filter.Eventually (fun x => LT.lt (f x) Top.top) (MeasureTheory.ae μ)
      g : α → E
      ⊢ Iff (MeasureTheory.Integrable (fun x => HSMul.hSMul (f x).toNNReal (g x)) μ) …
    -/
  · simp_rw [NNReal.smul_def, ENNReal.toReal]
    /-
      🎉 no goals
    -/
    /-
      case hf
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E : Type u_6
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : α → ENNReal
      hf : Measurable f
      hflt : Filter.Eventually (fun x => LT.lt (f x) Top.top) (MeasureTheory.ae μ)
      g : α → E
      ⊢ Measurable fun x => (f x).toNNReal
    -/
  · exact hf.ennreal_toNNReal
    /-
      🎉 no goals
    -/


theorem integrable_withDensity_iff_integrable_coe_smul₀ {f : α → ℝ≥0} (hf : AEMeasurable f μ)
    {g : α → E} :
    Integrable g (μ.withDensity fun x => f x) ↔ Integrable (fun x => (f x : ℝ) • g x) μ :=
  calc
    Integrable g (μ.withDensity fun x => f x) ↔
        Integrable g (μ.withDensity fun x => (hf.mk f x : ℝ≥0)) := by
      suffices (fun x => (f x : ℝ≥0∞)) =ᵐ[μ] (fun x => (hf.mk f x : ℝ≥0)) by
        rw [withDensity_congr_ae this]
      /-
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        E : Type u_6
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : α → NNReal
        hf : AEMeasurable f μ
        g : α → E
        ⊢ (MeasureTheory.ae μ).EventuallyEq (fun x => ↑(f x)) fun x => ↑(AEMeasurable. …
      -/
      filter_upwards [hf.ae_eq_mk] with x hx
      /-
        case h
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        E : Type u_6
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : α → NNReal
        hf : AEMeasurable f μ
        g : α → E
        x : α
        hx : Eq (f x) (AEMeasurable.mk f hf x)
        ⊢ Eq ↑(f x) ↑(AEMeasurable.mk f hf x)
      -/
      simp [hx]
      /-
        🎉 no goals
      -/
    _ ↔ Integrable (fun x => ((hf.mk f x : ℝ≥0) : ℝ) • g x) μ :=
      integrable_withDensity_iff_integrable_coe_smul hf.measurable_mk
    _ ↔ Integrable (fun x => (f x : ℝ) • g x) μ := by
      /-
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        E : Type u_6
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : α → NNReal
        hf : AEMeasurable f μ
        g : α → E
        ⊢ Iff (MeasureTheory.Integrable (fun x => HSMul.hSMul (↑(AEMeasurable.mk f hf  …
      -/
      apply integrable_congr
      /-
        case h
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        E : Type u_6
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : α → NNReal
        hf : AEMeasurable f μ
        g : α → E
        ⊢ (MeasureTheory.ae μ).EventuallyEq (fun x => HSMul.hSMul (↑(AEMeasurable.mk f …
      -/
      filter_upwards [hf.ae_eq_mk] with x hx
      /-
        case h
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        E : Type u_6
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : α → NNReal
        hf : AEMeasurable f μ
        g : α → E
        x : α
        hx : Eq (f x) (AEMeasurable.mk f hf x)
        ⊢ Eq (HSMul.hSMul (↑(AEMeasurable.mk f hf x)) (g x)) (HSMul.hSMul (↑(f x)) (g  …
      -/
      simp [hx]
      /-
        🎉 no goals
      -/


theorem integrable_withDensity_iff_integrable_smul₀ {f : α → ℝ≥0} (hf : AEMeasurable f μ)
    {g : α → E} : Integrable g (μ.withDensity fun x => f x) ↔ Integrable (fun x => f x • g x) μ :=
  integrable_withDensity_iff_integrable_coe_smul₀ hf


theorem integrable_withDensity_iff {f : α → ℝ≥0∞} (hf : Measurable f) (hflt : ∀ᵐ x ∂μ, f x < ∞)
    {g : α → ℝ} : Integrable g (μ.withDensity f) ↔ Integrable (fun x => g x * (f x).toReal) μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : Measurable f
    hflt : Filter.Eventually (fun x => LT.lt (f x) Top.top) (MeasureTheory.ae μ)
    g : α → Real
    ⊢ Iff (MeasureTheory.Integrable g (μ.withDensity f)) (MeasureTheory.Integrable …
  -/
  have : (fun x => g x * (f x).toReal) = fun x => (f x).toReal • g x := by simp [mul_comm]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : Measurable f
    hflt : Filter.Eventually (fun x => LT.lt (f x) Top.top) (MeasureTheory.ae μ)
    g : α → Real
    this : Eq (fun x => HMul.hMul (g x) (f x).toReal) fun x => HSMul.hSMul (f x).t …
    ⊢ Iff (MeasureTheory.Integrable g (μ.withDensity f)) (MeasureTheory.Integrable …
  -/
  rw [this]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : Measurable f
    hflt : Filter.Eventually (fun x => LT.lt (f x) Top.top) (MeasureTheory.ae μ)
    g : α → Real
    this : Eq (fun x => HMul.hMul (g x) (f x).toReal) fun x => HSMul.hSMul (f x).t …
    ⊢ Iff (MeasureTheory.Integrable g (μ.withDensity f)) (MeasureTheory.Integrable …
  -/
  exact integrable_withDensity_iff_integrable_smul' hf hflt
  /-
    🎉 no goals
  -/


theorem memℒ1_smul_of_L1_withDensity {f : α → ℝ≥0} (f_meas : Measurable f)
    (u : Lp E 1 (μ.withDensity fun x => f x)) : Memℒp (fun x => f x • u x) 1 μ :=
  memℒp_one_iff_integrable.2 <|
    (integrable_withDensity_iff_integrable_smul f_meas).1 <| memℒp_one_iff_integrable.1 (Lp.memℒp u)


/-- The map `u ↦ f • u` is an isometry between the `L^1` spaces for `μ.withDensity f` and `μ`. -/
noncomputable def withDensitySMulLI {f : α → ℝ≥0} (f_meas : Measurable f) :
    Lp E 1 (μ.withDensity fun x => f x) →ₗᵢ[ℝ] Lp E 1 μ where
  toFun u := (memℒ1_smul_of_L1_withDensity f_meas u).toLp _
  map_add' := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      ε : Type u_5
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝⁶ : MeasurableSpace δ
      inst✝⁵ : NormedAddCommGroup β
      inst✝⁴ : NormedAddCommGroup γ
      inst✝³ : ENorm ε
      inst✝² : TopologicalSpace ε
      E : Type u_6
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : α → NNReal
      f_meas : Measurable f
      ⊢ ∀ (x y : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 (μ.withDensit …
    -/
    intro u v
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      ε : Type u_5
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝⁶ : MeasurableSpace δ
      inst✝⁵ : NormedAddCommGroup β
      inst✝⁴ : NormedAddCommGroup γ
      inst✝³ : ENorm ε
      inst✝² : TopologicalSpace ε
      E : Type u_6
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : α → NNReal
      f_meas : Measurable f
      u v : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 (μ.withDensity fun …
      ⊢ Eq ((fun u => MeasureTheory.Memℒp.toLp (fun x => HSMul.hSMul (f x) (↑↑u x))  …
    -/
    ext1
    filter_upwards [(memℒ1_smul_of_L1_withDensity f_meas u).coeFn_toLp,
      (memℒ1_smul_of_L1_withDensity f_meas v).coeFn_toLp,
      (memℒ1_smul_of_L1_withDensity f_meas (u + v)).coeFn_toLp,
      Lp.coeFn_add ((memℒ1_smul_of_L1_withDensity f_meas u).toLp _)
        ((memℒ1_smul_of_L1_withDensity f_meas v).toLp _),
      (ae_withDensity_iff f_meas.coe_nnreal_ennreal).1 (Lp.coeFn_add u v)]
    /-
      case h
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      ε : Type u_5
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝⁶ : MeasurableSpace δ
      inst✝⁵ : NormedAddCommGroup β
      inst✝⁴ : NormedAddCommGroup γ
      inst✝³ : ENorm ε
      inst✝² : TopologicalSpace ε
      E : Type u_6
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : α → NNReal
      f_meas : Measurable f
      u v : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 (μ.withDensity fun …
      ⊢ ∀ (a : α), Eq (↑↑(MeasureTheory.Memℒp.toLp (fun x => HSMul.hSMul (f x) (↑↑u  …
    -/
    intro x hu hv huv h' h''
    /-
      case h
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      ε : Type u_5
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝⁶ : MeasurableSpace δ
      inst✝⁵ : NormedAddCommGroup β
      inst✝⁴ : NormedAddCommGroup γ
      inst✝³ : ENorm ε
      inst✝² : TopologicalSpace ε
      E : Type u_6
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : α → NNReal
      f_meas : Measurable f
      u v : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 (μ.withDensity fun …
      x : α
      hu : Eq (↑↑(MeasureTheory.Memℒp.toLp (fun x => HSMul.hSMul (f x) (↑↑u x)) ⋯) x …
      hv : Eq (↑↑(MeasureTheory.Memℒp.toLp (fun x => HSMul.hSMul (f x) (↑↑v x)) ⋯) x …
      huv : Eq (↑↑(MeasureTheory.Memℒp.toLp (fun x => HSMul.hSMul (f x) (↑↑(HAdd.hAd …
      h' : Eq (↑↑(HAdd.hAdd (MeasureTheory.Memℒp.toLp (fun x => HSMul.hSMul (f x) (↑ …
      h'' : Ne (↑(f x)) 0 → Eq (↑↑(HAdd.hAdd u v) x) (HAdd.hAdd (↑↑u) (↑↑v) x)
      ⊢ Eq (↑↑(MeasureTheory.Memℒp.toLp (fun x => HSMul.hSMul (f x) (↑↑(HAdd.hAdd u  …
    -/
    rw [huv, h', Pi.add_apply, hu, hv]
    /-
      case h
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      ε : Type u_5
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝⁶ : MeasurableSpace δ
      inst✝⁵ : NormedAddCommGroup β
      inst✝⁴ : NormedAddCommGroup γ
      inst✝³ : ENorm ε
      inst✝² : TopologicalSpace ε
      E : Type u_6
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : α → NNReal
      f_meas : Measurable f
      u v : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 (μ.withDensity fun …
      x : α
      hu : Eq (↑↑(MeasureTheory.Memℒp.toLp (fun x => HSMul.hSMul (f x) (↑↑u x)) ⋯) x …
      hv : Eq (↑↑(MeasureTheory.Memℒp.toLp (fun x => HSMul.hSMul (f x) (↑↑v x)) ⋯) x …
      huv : Eq (↑↑(MeasureTheory.Memℒp.toLp (fun x => HSMul.hSMul (f x) (↑↑(HAdd.hAd …
      h' : Eq (↑↑(HAdd.hAdd (MeasureTheory.Memℒp.toLp (fun x => HSMul.hSMul (f x) (↑ …
      h'' : Ne (↑(f x)) 0 → Eq (↑↑(HAdd.hAdd u v) x) (HAdd.hAdd (↑↑u) (↑↑v) x)
      ⊢ Eq (HSMul.hSMul (f x) (↑↑(HAdd.hAdd u v) x)) (HAdd.hAdd (HSMul.hSMul (f x) ( …
    -/
    rcases eq_or_ne (f x) 0 with (hx | hx)
      /-
        case h.inl
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        δ : Type u_4
        ε : Type u_5
        m : MeasurableSpace α
        μ ν : MeasureTheory.Measure α
        inst✝⁶ : MeasurableSpace δ
        inst✝⁵ : NormedAddCommGroup β
        inst✝⁴ : NormedAddCommGroup γ
        inst✝³ : ENorm ε
        inst✝² : TopologicalSpace ε
        E : Type u_6
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : α → NNReal
        f_meas : Measurable f
        u v : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 (μ.withDensity fun …
        x : α
        hu : Eq (↑↑(MeasureTheory.Memℒp.toLp (fun x => HSMul.hSMul (f x) (↑↑u x)) ⋯) x …
        hv : Eq (↑↑(MeasureTheory.Memℒp.toLp (fun x => HSMul.hSMul (f x) (↑↑v x)) ⋯) x …
        huv : Eq (↑↑(MeasureTheory.Memℒp.toLp (fun x => HSMul.hSMul (f x) (↑↑(HAdd.hAd …
        h' : Eq (↑↑(HAdd.hAdd (MeasureTheory.Memℒp.toLp (fun x => HSMul.hSMul (f x) (↑ …
        h'' : Ne (↑(f x)) 0 → Eq (↑↑(HAdd.hAdd u v) x) (HAdd.hAdd (↑↑u) (↑↑v) x)
        hx : Eq (f x) 0
        ⊢ Eq (HSMul.hSMul (f x) (↑↑(HAdd.hAdd u v) x)) (HAdd.hAdd (HSMul.hSMul (f x) ( …
      -/
    · simp only [hx, zero_smul, add_zero]
      /-
        🎉 no goals
      -/
      /-
        case h.inr
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        δ : Type u_4
        ε : Type u_5
        m : MeasurableSpace α
        μ ν : MeasureTheory.Measure α
        inst✝⁶ : MeasurableSpace δ
        inst✝⁵ : NormedAddCommGroup β
        inst✝⁴ : NormedAddCommGroup γ
        inst✝³ : ENorm ε
        inst✝² : TopologicalSpace ε
        E : Type u_6
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : α → NNReal
        f_meas : Measurable f
        u v : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 (μ.withDensity fun …
        x : α
        hu : Eq (↑↑(MeasureTheory.Memℒp.toLp (fun x => HSMul.hSMul (f x) (↑↑u x)) ⋯) x …
        hv : Eq (↑↑(MeasureTheory.Memℒp.toLp (fun x => HSMul.hSMul (f x) (↑↑v x)) ⋯) x …
        huv : Eq (↑↑(MeasureTheory.Memℒp.toLp (fun x => HSMul.hSMul (f x) (↑↑(HAdd.hAd …
        h' : Eq (↑↑(HAdd.hAdd (MeasureTheory.Memℒp.toLp (fun x => HSMul.hSMul (f x) (↑ …
        h'' : Ne (↑(f x)) 0 → Eq (↑↑(HAdd.hAdd u v) x) (HAdd.hAdd (↑↑u) (↑↑v) x)
        hx : Ne (f x) 0
        ⊢ Eq (HSMul.hSMul (f x) (↑↑(HAdd.hAdd u v) x)) (HAdd.hAdd (HSMul.hSMul (f x) ( …
      -/
    · rw [h'' _, Pi.add_apply, smul_add]
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        δ : Type u_4
        ε : Type u_5
        m : MeasurableSpace α
        μ ν : MeasureTheory.Measure α
        inst✝⁶ : MeasurableSpace δ
        inst✝⁵ : NormedAddCommGroup β
        inst✝⁴ : NormedAddCommGroup γ
        inst✝³ : ENorm ε
        inst✝² : TopologicalSpace ε
        E : Type u_6
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : α → NNReal
        f_meas : Measurable f
        u v : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 (μ.withDensity fun …
        x : α
        hu : Eq (↑↑(MeasureTheory.Memℒp.toLp (fun x => HSMul.hSMul (f x) (↑↑u x)) ⋯) x …
        hv : Eq (↑↑(MeasureTheory.Memℒp.toLp (fun x => HSMul.hSMul (f x) (↑↑v x)) ⋯) x …
        huv : Eq (↑↑(MeasureTheory.Memℒp.toLp (fun x => HSMul.hSMul (f x) (↑↑(HAdd.hAd …
        h' : Eq (↑↑(HAdd.hAdd (MeasureTheory.Memℒp.toLp (fun x => HSMul.hSMul (f x) (↑ …
        h'' : Ne (↑(f x)) 0 → Eq (↑↑(HAdd.hAdd u v) x) (HAdd.hAdd (↑↑u) (↑↑v) x)
        hx : Ne (f x) 0
        ⊢ Ne (↑(f x)) 0
      -/
      simpa only [Ne, ENNReal.coe_eq_zero] using hx
      /-
        🎉 no goals
      -/
  map_smul' := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      ε : Type u_5
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝⁶ : MeasurableSpace δ
      inst✝⁵ : NormedAddCommGroup β
      inst✝⁴ : NormedAddCommGroup γ
      inst✝³ : ENorm ε
      inst✝² : TopologicalSpace ε
      E : Type u_6
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : α → NNReal
      f_meas : Measurable f
      ⊢ ∀ (m_1 : Real) (x : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 (μ …
    -/
    intro r u
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      ε : Type u_5
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝⁶ : MeasurableSpace δ
      inst✝⁵ : NormedAddCommGroup β
      inst✝⁴ : NormedAddCommGroup γ
      inst✝³ : ENorm ε
      inst✝² : TopologicalSpace ε
      E : Type u_6
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : α → NNReal
      f_meas : Measurable f
      r : Real
      u : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 (μ.withDensity fun x …
      ⊢ Eq ({ toFun := fun u => MeasureTheory.Memℒp.toLp (fun x => HSMul.hSMul (f x) …
    -/
    ext1
    filter_upwards [(ae_withDensity_iff f_meas.coe_nnreal_ennreal).1 (Lp.coeFn_smul r u),
      (memℒ1_smul_of_L1_withDensity f_meas (r • u)).coeFn_toLp,
      Lp.coeFn_smul r ((memℒ1_smul_of_L1_withDensity f_meas u).toLp _),
      (memℒ1_smul_of_L1_withDensity f_meas u).coeFn_toLp]
    /-
      case h
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      ε : Type u_5
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝⁶ : MeasurableSpace δ
      inst✝⁵ : NormedAddCommGroup β
      inst✝⁴ : NormedAddCommGroup γ
      inst✝³ : ENorm ε
      inst✝² : TopologicalSpace ε
      E : Type u_6
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : α → NNReal
      f_meas : Measurable f
      r : Real
      u : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 (μ.withDensity fun x …
      ⊢ ∀ (a : α), (Ne (↑(f a)) 0 → Eq (↑↑(HSMul.hSMul r u) a) (HSMul.hSMul r (↑↑u)  …
    -/
    intro x h h' h'' h'''
    /-
      case h
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      ε : Type u_5
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝⁶ : MeasurableSpace δ
      inst✝⁵ : NormedAddCommGroup β
      inst✝⁴ : NormedAddCommGroup γ
      inst✝³ : ENorm ε
      inst✝² : TopologicalSpace ε
      E : Type u_6
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : α → NNReal
      f_meas : Measurable f
      r : Real
      u : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 (μ.withDensity fun x …
      x : α
      h : Ne (↑(f x)) 0 → Eq (↑↑(HSMul.hSMul r u) x) (HSMul.hSMul r (↑↑u) x)
      h' : Eq (↑↑(MeasureTheory.Memℒp.toLp (fun x => HSMul.hSMul (f x) (↑↑(HSMul.hSM …
      h'' : Eq (↑↑(HSMul.hSMul r (MeasureTheory.Memℒp.toLp (fun x => HSMul.hSMul (f  …
      h''' : Eq (↑↑(MeasureTheory.Memℒp.toLp (fun x => HSMul.hSMul (f x) (↑↑u x)) ⋯) …
      ⊢ Eq (↑↑(MeasureTheory.Memℒp.toLp (fun x => HSMul.hSMul (f x) (↑↑(HSMul.hSMul  …
    -/
    rw [RingHom.id_apply, h', h'', Pi.smul_apply, h''']
    /-
      case h
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      ε : Type u_5
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝⁶ : MeasurableSpace δ
      inst✝⁵ : NormedAddCommGroup β
      inst✝⁴ : NormedAddCommGroup γ
      inst✝³ : ENorm ε
      inst✝² : TopologicalSpace ε
      E : Type u_6
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : α → NNReal
      f_meas : Measurable f
      r : Real
      u : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 (μ.withDensity fun x …
      x : α
      h : Ne (↑(f x)) 0 → Eq (↑↑(HSMul.hSMul r u) x) (HSMul.hSMul r (↑↑u) x)
      h' : Eq (↑↑(MeasureTheory.Memℒp.toLp (fun x => HSMul.hSMul (f x) (↑↑(HSMul.hSM …
      h'' : Eq (↑↑(HSMul.hSMul r (MeasureTheory.Memℒp.toLp (fun x => HSMul.hSMul (f  …
      h''' : Eq (↑↑(MeasureTheory.Memℒp.toLp (fun x => HSMul.hSMul (f x) (↑↑u x)) ⋯) …
      ⊢ Eq (HSMul.hSMul (f x) (↑↑(HSMul.hSMul r u) x)) (HSMul.hSMul r (HSMul.hSMul ( …
    -/
    rcases eq_or_ne (f x) 0 with (hx | hx)
      /-
        case h.inl
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        δ : Type u_4
        ε : Type u_5
        m : MeasurableSpace α
        μ ν : MeasureTheory.Measure α
        inst✝⁶ : MeasurableSpace δ
        inst✝⁵ : NormedAddCommGroup β
        inst✝⁴ : NormedAddCommGroup γ
        inst✝³ : ENorm ε
        inst✝² : TopologicalSpace ε
        E : Type u_6
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : α → NNReal
        f_meas : Measurable f
        r : Real
        u : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 (μ.withDensity fun x …
        x : α
        h : Ne (↑(f x)) 0 → Eq (↑↑(HSMul.hSMul r u) x) (HSMul.hSMul r (↑↑u) x)
        h' : Eq (↑↑(MeasureTheory.Memℒp.toLp (fun x => HSMul.hSMul (f x) (↑↑(HSMul.hSM …
        h'' : Eq (↑↑(HSMul.hSMul r (MeasureTheory.Memℒp.toLp (fun x => HSMul.hSMul (f  …
        h''' : Eq (↑↑(MeasureTheory.Memℒp.toLp (fun x => HSMul.hSMul (f x) (↑↑u x)) ⋯) …
        hx : Eq (f x) 0
        ⊢ Eq (HSMul.hSMul (f x) (↑↑(HSMul.hSMul r u) x)) (HSMul.hSMul r (HSMul.hSMul ( …
      -/
    · simp only [hx, zero_smul, smul_zero]
      /-
        🎉 no goals
      -/
      /-
        case h.inr
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        δ : Type u_4
        ε : Type u_5
        m : MeasurableSpace α
        μ ν : MeasureTheory.Measure α
        inst✝⁶ : MeasurableSpace δ
        inst✝⁵ : NormedAddCommGroup β
        inst✝⁴ : NormedAddCommGroup γ
        inst✝³ : ENorm ε
        inst✝² : TopologicalSpace ε
        E : Type u_6
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : α → NNReal
        f_meas : Measurable f
        r : Real
        u : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 (μ.withDensity fun x …
        x : α
        h : Ne (↑(f x)) 0 → Eq (↑↑(HSMul.hSMul r u) x) (HSMul.hSMul r (↑↑u) x)
        h' : Eq (↑↑(MeasureTheory.Memℒp.toLp (fun x => HSMul.hSMul (f x) (↑↑(HSMul.hSM …
        h'' : Eq (↑↑(HSMul.hSMul r (MeasureTheory.Memℒp.toLp (fun x => HSMul.hSMul (f  …
        h''' : Eq (↑↑(MeasureTheory.Memℒp.toLp (fun x => HSMul.hSMul (f x) (↑↑u x)) ⋯) …
        hx : Ne (f x) 0
        ⊢ Eq (HSMul.hSMul (f x) (↑↑(HSMul.hSMul r u) x)) (HSMul.hSMul r (HSMul.hSMul ( …
      -/
    · rw [h _, smul_comm, Pi.smul_apply]
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        δ : Type u_4
        ε : Type u_5
        m : MeasurableSpace α
        μ ν : MeasureTheory.Measure α
        inst✝⁶ : MeasurableSpace δ
        inst✝⁵ : NormedAddCommGroup β
        inst✝⁴ : NormedAddCommGroup γ
        inst✝³ : ENorm ε
        inst✝² : TopologicalSpace ε
        E : Type u_6
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : α → NNReal
        f_meas : Measurable f
        r : Real
        u : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 (μ.withDensity fun x …
        x : α
        h : Ne (↑(f x)) 0 → Eq (↑↑(HSMul.hSMul r u) x) (HSMul.hSMul r (↑↑u) x)
        h' : Eq (↑↑(MeasureTheory.Memℒp.toLp (fun x => HSMul.hSMul (f x) (↑↑(HSMul.hSM …
        h'' : Eq (↑↑(HSMul.hSMul r (MeasureTheory.Memℒp.toLp (fun x => HSMul.hSMul (f  …
        h''' : Eq (↑↑(MeasureTheory.Memℒp.toLp (fun x => HSMul.hSMul (f x) (↑↑u x)) ⋯) …
        hx : Ne (f x) 0
        ⊢ Ne (↑(f x)) 0
      -/
      simpa only [Ne, ENNReal.coe_eq_zero] using hx
      /-
        🎉 no goals
      -/
  norm_map' := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      ε : Type u_5
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝⁶ : MeasurableSpace δ
      inst✝⁵ : NormedAddCommGroup β
      inst✝⁴ : NormedAddCommGroup γ
      inst✝³ : ENorm ε
      inst✝² : TopologicalSpace ε
      E : Type u_6
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : α → NNReal
      f_meas : Measurable f
      ⊢ ∀ (x : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 (μ.withDensity  …
    -/
    intro u
    -- Porting note: Lean can't infer types of `AddHom.coe_mk`.
    simp only [eLpNorm, LinearMap.coe_mk,
      AddHom.coe_mk (M := Lp E 1 (μ.withDensity fun x => f x)) (N := Lp E 1 μ), Lp.norm_toLp,
      one_ne_zero, ENNReal.one_ne_top, ENNReal.one_toReal, if_false, eLpNorm', ENNReal.rpow_one,
      _root_.div_one, Lp.norm_def]
    rw [lintegral_withDensity_eq_lintegral_mul_non_measurable _ f_meas.coe_nnreal_ennreal
        (Filter.Eventually.of_forall fun x => ENNReal.coe_lt_top)]
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      ε : Type u_5
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝⁶ : MeasurableSpace δ
      inst✝⁵ : NormedAddCommGroup β
      inst✝⁴ : NormedAddCommGroup γ
      inst✝³ : ENorm ε
      inst✝² : TopologicalSpace ε
      E : Type u_6
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : α → NNReal
      f_meas : Measurable f
      u : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 (μ.withDensity fun x …
      ⊢ Eq (MeasureTheory.lintegral μ fun a => ENorm.enorm (↑↑(MeasureTheory.Memℒp.t …
    -/
    congr 1
    /-
      case e_a
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      ε : Type u_5
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝⁶ : MeasurableSpace δ
      inst✝⁵ : NormedAddCommGroup β
      inst✝⁴ : NormedAddCommGroup γ
      inst✝³ : ENorm ε
      inst✝² : TopologicalSpace ε
      E : Type u_6
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : α → NNReal
      f_meas : Measurable f
      u : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 (μ.withDensity fun x …
      ⊢ Eq (MeasureTheory.lintegral μ fun a => ENorm.enorm (↑↑(MeasureTheory.Memℒp.t …
    -/
    apply lintegral_congr_ae
    /-
      case e_a.h
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      ε : Type u_5
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝⁶ : MeasurableSpace δ
      inst✝⁵ : NormedAddCommGroup β
      inst✝⁴ : NormedAddCommGroup γ
      inst✝³ : ENorm ε
      inst✝² : TopologicalSpace ε
      E : Type u_6
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : α → NNReal
      f_meas : Measurable f
      u : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 (μ.withDensity fun x …
      ⊢ (MeasureTheory.ae μ).EventuallyEq (fun a => ENorm.enorm (↑↑(MeasureTheory.Me …
    -/
    filter_upwards [(memℒ1_smul_of_L1_withDensity f_meas u).coeFn_toLp] with x hx
    /-
      case h
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      ε : Type u_5
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝⁶ : MeasurableSpace δ
      inst✝⁵ : NormedAddCommGroup β
      inst✝⁴ : NormedAddCommGroup γ
      inst✝³ : ENorm ε
      inst✝² : TopologicalSpace ε
      E : Type u_6
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : α → NNReal
      f_meas : Measurable f
      u : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 (μ.withDensity fun x …
      x : α
      hx : Eq (↑↑(MeasureTheory.Memℒp.toLp (fun x => HSMul.hSMul (f x) (↑↑u x)) ⋯) x …
      ⊢ Eq (ENorm.enorm (↑↑(MeasureTheory.Memℒp.toLp (fun x => HSMul.hSMul (f x) (↑↑ …
    -/
    rw [hx, Pi.mul_apply]
    /-
      case h
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      ε : Type u_5
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝⁶ : MeasurableSpace δ
      inst✝⁵ : NormedAddCommGroup β
      inst✝⁴ : NormedAddCommGroup γ
      inst✝³ : ENorm ε
      inst✝² : TopologicalSpace ε
      E : Type u_6
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : α → NNReal
      f_meas : Measurable f
      u : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 (μ.withDensity fun x …
      x : α
      hx : Eq (↑↑(MeasureTheory.Memℒp.toLp (fun x => HSMul.hSMul (f x) (↑↑u x)) ⋯) x …
      ⊢ Eq (ENorm.enorm (HSMul.hSMul (f x) (↑↑u x))) (HMul.hMul (↑(f x)) (ENorm.enor …
    -/
    change (‖(f x : ℝ) • u x‖₊ : ℝ≥0∞) = (f x : ℝ≥0∞) * (‖u x‖₊ : ℝ≥0∞)
    /-
      case h
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      ε : Type u_5
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      inst✝⁶ : MeasurableSpace δ
      inst✝⁵ : NormedAddCommGroup β
      inst✝⁴ : NormedAddCommGroup γ
      inst✝³ : ENorm ε
      inst✝² : TopologicalSpace ε
      E : Type u_6
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : α → NNReal
      f_meas : Measurable f
      u : Subtype fun x => Membership.mem (MeasureTheory.Lp E 1 (μ.withDensity fun x …
      x : α
      hx : Eq (↑↑(MeasureTheory.Memℒp.toLp (fun x => HSMul.hSMul (f x) (↑↑u x)) ⋯) x …
      ⊢ Eq (↑(NNNorm.nnnorm (HSMul.hSMul (↑(f x)) (↑↑u x)))) (HMul.hMul ↑(f x) ↑(NNN …
    -/
    simp only [nnnorm_smul, NNReal.nnnorm_eq, ENNReal.coe_mul]
    /-
      🎉 no goals
    -/


@[simp]
theorem withDensitySMulLI_apply {f : α → ℝ≥0} (f_meas : Measurable f)
    (u : Lp E 1 (μ.withDensity fun x => f x)) :
    withDensitySMulLI μ (E := E) f_meas u =
      (memℒ1_smul_of_L1_withDensity f_meas u).toLp fun x => f x • u x :=
  rfl


theorem mem_ℒ1_toReal_of_lintegral_ne_top {f : α → ℝ≥0∞} (hfm : AEMeasurable f μ)
    (hfi : ∫⁻ x, f x ∂μ ≠ ∞) : Memℒp (fun x ↦ (f x).toReal) 1 μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hfm : AEMeasurable f μ
    hfi : Ne (MeasureTheory.lintegral μ fun x => f x) Top.top
    ⊢ MeasureTheory.Memℒp (fun x => (f x).toReal) 1 μ
  -/
  rw [Memℒp, eLpNorm_one_eq_lintegral_nnnorm]
  exact ⟨(AEMeasurable.ennreal_toReal hfm).aestronglyMeasurable,
    hasFiniteIntegral_toReal_of_lintegral_ne_top hfi⟩


theorem integrable_toReal_of_lintegral_ne_top {f : α → ℝ≥0∞} (hfm : AEMeasurable f μ)
    (hfi : ∫⁻ x, f x ∂μ ≠ ∞) : Integrable (fun x ↦ (f x).toReal) μ :=
  memℒp_one_iff_integrable.1 <| mem_ℒ1_toReal_of_lintegral_ne_top hfm hfi


lemma integrable_toReal_iff {f : α → ℝ≥0∞} (hf : AEMeasurable f μ) (hf_ne_top : ∀ᵐ x ∂μ, f x ≠ ∞) :
    Integrable (fun x ↦ (f x).toReal) μ ↔ ∫⁻ x, f x ∂μ ≠ ∞ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : AEMeasurable f μ
    hf_ne_top : Filter.Eventually (fun x => Ne (f x) Top.top) (MeasureTheory.ae μ)
    ⊢ Iff (MeasureTheory.Integrable (fun x => (f x).toReal) μ) (Ne (MeasureTheory. …
  -/
  rw [Integrable, hasFiniteIntegral_toReal_iff hf_ne_top]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → ENNReal
    hf : AEMeasurable f μ
    hf_ne_top : Filter.Eventually (fun x => Ne (f x) Top.top) (MeasureTheory.ae μ)
    ⊢ Iff (And (MeasureTheory.AEStronglyMeasurable (fun x => (f x).toReal) μ) (Ne  …
  -/
  simp only [hf.ennreal_toReal.aestronglyMeasurable, ne_eq, true_and]
  /-
    🎉 no goals
  -/


lemma lintegral_ofReal_ne_top_iff_integrable {f : α → ℝ}
    (hfm : AEStronglyMeasurable f μ) (hf : 0 ≤ᵐ[μ] f) :
    ∫⁻ a, ENNReal.ofReal (f a) ∂μ ≠ ∞ ↔ Integrable f μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hfm : MeasureTheory.AEStronglyMeasurable f μ
    hf : (MeasureTheory.ae μ).EventuallyLE 0 f
    ⊢ Iff (Ne (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (f a)) Top.top) ( …
  -/
  rw [Integrable, hasFiniteIntegral_iff_ofReal hf]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hfm : MeasureTheory.AEStronglyMeasurable f μ
    hf : (MeasureTheory.ae μ).EventuallyLE 0 f
    ⊢ Iff (Ne (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (f a)) Top.top) ( …
  -/
  simp [hfm]
  /-
    🎉 no goals
  -/


theorem Integrable.pos_part {f : α → ℝ} (hf : Integrable f μ) :
    Integrable (fun a => max (f a) 0) μ :=
  ⟨(hf.aestronglyMeasurable.aemeasurable.max aemeasurable_const).aestronglyMeasurable,
    hf.hasFiniteIntegral.max_zero⟩


theorem Integrable.neg_part {f : α → ℝ} (hf : Integrable f μ) :
    Integrable (fun a => max (-f a) 0) μ :=
  hf.neg.pos_part


theorem Integrable.smul [NormedAddCommGroup 𝕜] [SMulZeroClass 𝕜 β] [BoundedSMul 𝕜 β] (c : 𝕜)
    {f : α → β} (hf : Integrable f μ) : Integrable (c • f) μ :=
  ⟨hf.aestronglyMeasurable.const_smul c, hf.hasFiniteIntegral.smul c⟩


theorem _root_.IsUnit.integrable_smul_iff [NormedRing 𝕜] [Module 𝕜 β] [BoundedSMul 𝕜 β] {c : 𝕜}
    (hc : IsUnit c) (f : α → β) : Integrable (c • f) μ ↔ Integrable f μ :=
  and_congr hc.aestronglyMeasurable_const_smul_iff (hasFiniteIntegral_smul_iff hc f)


theorem integrable_smul_iff [NormedDivisionRing 𝕜] [Module 𝕜 β] [BoundedSMul 𝕜 β] {c : 𝕜}
    (hc : c ≠ 0) (f : α → β) : Integrable (c • f) μ ↔ Integrable f μ :=
  (IsUnit.mk0 _ hc).integrable_smul_iff f


theorem Integrable.smul_of_top_right {f : α → β} {φ : α → 𝕜} (hf : Integrable f μ)
    (hφ : Memℒp φ ∞ μ) : Integrable (φ • f) μ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : NormedAddCommGroup β
    𝕜 : Type u_6
    inst✝² : NormedRing 𝕜
    inst✝¹ : Module 𝕜 β
    inst✝ : BoundedSMul 𝕜 β
    f : α → β
    φ : α → 𝕜
    hf : MeasureTheory.Integrable f μ
    hφ : MeasureTheory.Memℒp φ Top.top μ
    ⊢ MeasureTheory.Integrable (HSMul.hSMul φ f) μ
  -/
  rw [← memℒp_one_iff_integrable] at hf ⊢
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : NormedAddCommGroup β
    𝕜 : Type u_6
    inst✝² : NormedRing 𝕜
    inst✝¹ : Module 𝕜 β
    inst✝ : BoundedSMul 𝕜 β
    f : α → β
    φ : α → 𝕜
    hf : MeasureTheory.Memℒp f 1 μ
    hφ : MeasureTheory.Memℒp φ Top.top μ
    ⊢ MeasureTheory.Memℒp (HSMul.hSMul φ f) 1 μ
  -/
  exact Memℒp.smul_of_top_right hf hφ
  /-
    🎉 no goals
  -/


theorem Integrable.smul_of_top_left {f : α → β} {φ : α → 𝕜} (hφ : Integrable φ μ)
    (hf : Memℒp f ∞ μ) : Integrable (φ • f) μ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : NormedAddCommGroup β
    𝕜 : Type u_6
    inst✝² : NormedRing 𝕜
    inst✝¹ : Module 𝕜 β
    inst✝ : BoundedSMul 𝕜 β
    f : α → β
    φ : α → 𝕜
    hφ : MeasureTheory.Integrable φ μ
    hf : MeasureTheory.Memℒp f Top.top μ
    ⊢ MeasureTheory.Integrable (HSMul.hSMul φ f) μ
  -/
  rw [← memℒp_one_iff_integrable] at hφ ⊢
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : NormedAddCommGroup β
    𝕜 : Type u_6
    inst✝² : NormedRing 𝕜
    inst✝¹ : Module 𝕜 β
    inst✝ : BoundedSMul 𝕜 β
    f : α → β
    φ : α → 𝕜
    hφ : MeasureTheory.Memℒp φ 1 μ
    hf : MeasureTheory.Memℒp f Top.top μ
    ⊢ MeasureTheory.Memℒp (HSMul.hSMul φ f) 1 μ
  -/
  exact Memℒp.smul_of_top_left hf hφ
  /-
    🎉 no goals
  -/


theorem Integrable.smul_const {f : α → 𝕜} (hf : Integrable f μ) (c : β) :
    Integrable (fun x => f x • c) μ :=
  hf.smul_of_top_left (memℒp_top_const c)


theorem integrable_smul_const {f : α → 𝕜} {c : E} (hc : c ≠ 0) :
    Integrable (fun x => f x • c) μ ↔ Integrable f μ := by
  simp_rw [Integrable, aestronglyMeasurable_smul_const_iff (f := f) hc, and_congr_right_iff,
    hasFiniteIntegral_iff_nnnorm, nnnorm_smul, ENNReal.coe_mul]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    𝕜 : Type u_6
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : CompleteSpace 𝕜
    E : Type u_7
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : α → 𝕜
    c : E
    hc : Ne c 0
    ⊢ MeasureTheory.AEStronglyMeasurable f μ → Iff (LT.lt (MeasureTheory.lintegral …
  -/
  intro _; rw [lintegral_mul_const' _ _ ENNReal.coe_ne_top, ENNReal.mul_lt_top_iff]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    𝕜 : Type u_6
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : CompleteSpace 𝕜
    E : Type u_7
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : α → 𝕜
    c : E
    hc : Ne c 0
    a✝ : MeasureTheory.AEStronglyMeasurable f μ
    ⊢ Iff (Or (And (LT.lt (MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (f a …
  -/
  have : ∀ x : ℝ≥0∞, x = 0 → x < ∞ := by simp
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    𝕜 : Type u_6
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : CompleteSpace 𝕜
    E : Type u_7
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : α → 𝕜
    c : E
    hc : Ne c 0
    a✝ : MeasureTheory.AEStronglyMeasurable f μ
    this : ∀ (x : ENNReal), Eq x 0 → LT.lt x Top.top
    ⊢ Iff (Or (And (LT.lt (MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (f a …
  -/
  simp [hc, or_iff_left_of_imp (this _)]
  /-
    🎉 no goals
  -/


theorem Integrable.const_mul {f : α → 𝕜} (h : Integrable f μ) (c : 𝕜) :
    Integrable (fun x => c * f x) μ :=
  h.smul c


theorem Integrable.const_mul' {f : α → 𝕜} (h : Integrable f μ) (c : 𝕜) :
    Integrable ((fun _ : α => c) * f) μ :=
  Integrable.const_mul h c


theorem Integrable.mul_const {f : α → 𝕜} (h : Integrable f μ) (c : 𝕜) :
    Integrable (fun x => f x * c) μ :=
  h.smul (MulOpposite.op c)


theorem Integrable.mul_const' {f : α → 𝕜} (h : Integrable f μ) (c : 𝕜) :
    Integrable (f * fun _ : α => c) μ :=
  Integrable.mul_const h c


theorem integrable_const_mul_iff {c : 𝕜} (hc : IsUnit c) (f : α → 𝕜) :
    Integrable (fun x => c * f x) μ ↔ Integrable f μ :=
  hc.integrable_smul_iff f


theorem integrable_mul_const_iff {c : 𝕜} (hc : IsUnit c) (f : α → 𝕜) :
    Integrable (fun x => f x * c) μ ↔ Integrable f μ :=
  hc.op.integrable_smul_iff f


theorem Integrable.bdd_mul' {f g : α → 𝕜} {c : ℝ} (hg : Integrable g μ)
    (hf : AEStronglyMeasurable f μ) (hf_bound : ∀ᵐ x ∂μ, ‖f x‖ ≤ c) :
    Integrable (fun x => f x * g x) μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    𝕜 : Type u_6
    inst✝ : NormedRing 𝕜
    f g : α → 𝕜
    c : Real
    hg : MeasureTheory.Integrable g μ
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hf_bound : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) c) (MeasureTheo …
    ⊢ MeasureTheory.Integrable (fun x => HMul.hMul (f x) (g x)) μ
  -/
  refine Integrable.mono' (hg.norm.smul c) (hf.mul hg.1) ?_
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    𝕜 : Type u_6
    inst✝ : NormedRing 𝕜
    f g : α → 𝕜
    c : Real
    hg : MeasureTheory.Integrable g μ
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hf_bound : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) c) (MeasureTheo …
    ⊢ Filter.Eventually (fun a => LE.le (Norm.norm (HMul.hMul (f a) (g a))) (HSMul …
  -/
  filter_upwards [hf_bound] with x hx
  /-
    case h
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    𝕜 : Type u_6
    inst✝ : NormedRing 𝕜
    f g : α → 𝕜
    c : Real
    hg : MeasureTheory.Integrable g μ
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hf_bound : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) c) (MeasureTheo …
    x : α
    hx : LE.le (Norm.norm (f x)) c
    ⊢ LE.le (Norm.norm (HMul.hMul (f x) (g x))) (HSMul.hSMul c (fun a => Norm.norm …
  -/
  rw [Pi.smul_apply, smul_eq_mul]
  /-
    case h
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    𝕜 : Type u_6
    inst✝ : NormedRing 𝕜
    f g : α → 𝕜
    c : Real
    hg : MeasureTheory.Integrable g μ
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hf_bound : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) c) (MeasureTheo …
    x : α
    hx : LE.le (Norm.norm (f x)) c
    ⊢ LE.le (Norm.norm (HMul.hMul (f x) (g x))) (HMul.hMul c (Norm.norm (g x)))
  -/
  exact (norm_mul_le _ _).trans (mul_le_mul_of_nonneg_right hx (norm_nonneg _))
  /-
    🎉 no goals
  -/


theorem Integrable.div_const {f : α → 𝕜} (h : Integrable f μ) (c : 𝕜) :
                                          /-
                                            α : Type u_1
                                            m : MeasurableSpace α
                                            μ : MeasureTheory.Measure α
                                            𝕜 : Type u_6
                                            inst✝ : NormedDivisionRing 𝕜
                                            f : α → 𝕜
                                            h : MeasureTheory.Integrable f μ
                                            c : 𝕜
                                            ⊢ MeasureTheory.Integrable (fun x => HDiv.hDiv (f x) c) μ
                                          -/
    Integrable (fun x => f x / c) μ := by simp_rw [div_eq_mul_inv, h.mul_const]
                                          /-
                                            🎉 no goals
                                          -/


theorem Integrable.ofReal {f : α → ℝ} (hf : Integrable f μ) :
    Integrable (fun x => (f x : 𝕜)) μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    𝕜 : Type u_6
    inst✝ : RCLike 𝕜
    f : α → Real
    hf : MeasureTheory.Integrable f μ
    ⊢ MeasureTheory.Integrable (fun x => ↑(f x)) μ
  -/
  rw [← memℒp_one_iff_integrable] at hf ⊢
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    𝕜 : Type u_6
    inst✝ : RCLike 𝕜
    f : α → Real
    hf : MeasureTheory.Memℒp f 1 μ
    ⊢ MeasureTheory.Memℒp (fun x => ↑(f x)) 1 μ
  -/
  exact hf.ofReal
  /-
    🎉 no goals
  -/


theorem Integrable.re_im_iff :
    Integrable (fun x => RCLike.re (f x)) μ ∧ Integrable (fun x => RCLike.im (f x)) μ ↔
      Integrable f μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    𝕜 : Type u_6
    inst✝ : RCLike 𝕜
    f : α → 𝕜
    ⊢ Iff (And (MeasureTheory.Integrable (fun x => RCLike.re (f x)) μ) (MeasureThe …
  -/
  simp_rw [← memℒp_one_iff_integrable]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    𝕜 : Type u_6
    inst✝ : RCLike 𝕜
    f : α → 𝕜
    ⊢ Iff (And (MeasureTheory.Memℒp (fun x => RCLike.re (f x)) 1 μ) (MeasureTheory …
  -/
  exact memℒp_re_im_iff
  /-
    🎉 no goals
  -/


theorem Integrable.re (hf : Integrable f μ) : Integrable (fun x => RCLike.re (f x)) μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    𝕜 : Type u_6
    inst✝ : RCLike 𝕜
    f : α → 𝕜
    hf : MeasureTheory.Integrable f μ
    ⊢ MeasureTheory.Integrable (fun x => RCLike.re (f x)) μ
  -/
  rw [← memℒp_one_iff_integrable] at hf ⊢
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    𝕜 : Type u_6
    inst✝ : RCLike 𝕜
    f : α → 𝕜
    hf : MeasureTheory.Memℒp f 1 μ
    ⊢ MeasureTheory.Memℒp (fun x => RCLike.re (f x)) 1 μ
  -/
  exact hf.re
  /-
    🎉 no goals
  -/


theorem Integrable.im (hf : Integrable f μ) : Integrable (fun x => RCLike.im (f x)) μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    𝕜 : Type u_6
    inst✝ : RCLike 𝕜
    f : α → 𝕜
    hf : MeasureTheory.Integrable f μ
    ⊢ MeasureTheory.Integrable (fun x => RCLike.im (f x)) μ
  -/
  rw [← memℒp_one_iff_integrable] at hf ⊢
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    𝕜 : Type u_6
    inst✝ : RCLike 𝕜
    f : α → 𝕜
    hf : MeasureTheory.Memℒp f 1 μ
    ⊢ MeasureTheory.Memℒp (fun x => RCLike.im (f x)) 1 μ
  -/
  exact hf.im
  /-
    🎉 no goals
  -/


theorem Integrable.trim (hm : m ≤ m0) (hf_int : Integrable f μ') (hf : StronglyMeasurable[m] f) :
    Integrable f (μ'.trim hm) := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    H : Type u_6
    inst✝ : NormedAddCommGroup H
    m0 : MeasurableSpace α
    μ' : MeasureTheory.Measure α
    f : α → H
    hm : LE.le m m0
    hf_int : MeasureTheory.Integrable f μ'
    hf : MeasureTheory.StronglyMeasurable f
    ⊢ MeasureTheory.Integrable f (μ'.trim hm)
  -/
  refine ⟨hf.aestronglyMeasurable, ?_⟩
  /-
    α : Type u_1
    m : MeasurableSpace α
    H : Type u_6
    inst✝ : NormedAddCommGroup H
    m0 : MeasurableSpace α
    μ' : MeasureTheory.Measure α
    f : α → H
    hm : LE.le m m0
    hf_int : MeasureTheory.Integrable f μ'
    hf : MeasureTheory.StronglyMeasurable f
    ⊢ MeasureTheory.HasFiniteIntegral f (μ'.trim hm)
  -/
  rw [HasFiniteIntegral, lintegral_trim hm _]
    /-
      α : Type u_1
      m : MeasurableSpace α
      H : Type u_6
      inst✝ : NormedAddCommGroup H
      m0 : MeasurableSpace α
      μ' : MeasureTheory.Measure α
      f : α → H
      hm : LE.le m m0
      hf_int : MeasureTheory.Integrable f μ'
      hf : MeasureTheory.StronglyMeasurable f
      ⊢ LT.lt (MeasureTheory.lintegral μ' fun a => ENorm.enorm (f a)) Top.top
    -/
  · exact hf_int.2
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      m : MeasurableSpace α
      H : Type u_6
      inst✝ : NormedAddCommGroup H
      m0 : MeasurableSpace α
      μ' : MeasureTheory.Measure α
      f : α → H
      hm : LE.le m m0
      hf_int : MeasureTheory.Integrable f μ'
      hf : MeasureTheory.StronglyMeasurable f
      ⊢ Measurable fun a => ENorm.enorm (f a)
    -/
  · exact @StronglyMeasurable.ennnorm _ m _ _ f hf
    /-
      🎉 no goals
    -/


theorem integrable_of_integrable_trim (hm : m ≤ m0) (hf_int : Integrable f (μ'.trim hm)) :
    Integrable f μ' := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    H : Type u_6
    inst✝ : NormedAddCommGroup H
    m0 : MeasurableSpace α
    μ' : MeasureTheory.Measure α
    f : α → H
    hm : LE.le m m0
    hf_int : MeasureTheory.Integrable f (μ'.trim hm)
    ⊢ MeasureTheory.Integrable f μ'
  -/
  obtain ⟨hf_meas_ae, hf⟩ := hf_int
  /-
    case intro
    α : Type u_1
    m : MeasurableSpace α
    H : Type u_6
    inst✝ : NormedAddCommGroup H
    m0 : MeasurableSpace α
    μ' : MeasureTheory.Measure α
    f : α → H
    hm : LE.le m m0
    hf_meas_ae : MeasureTheory.AEStronglyMeasurable f (μ'.trim hm)
    hf : MeasureTheory.HasFiniteIntegral f (μ'.trim hm)
    ⊢ MeasureTheory.Integrable f μ'
  -/
  refine ⟨aestronglyMeasurable_of_aestronglyMeasurable_trim hm hf_meas_ae, ?_⟩
  /-
    case intro
    α : Type u_1
    m : MeasurableSpace α
    H : Type u_6
    inst✝ : NormedAddCommGroup H
    m0 : MeasurableSpace α
    μ' : MeasureTheory.Measure α
    f : α → H
    hm : LE.le m m0
    hf_meas_ae : MeasureTheory.AEStronglyMeasurable f (μ'.trim hm)
    hf : MeasureTheory.HasFiniteIntegral f (μ'.trim hm)
    ⊢ MeasureTheory.HasFiniteIntegral f μ'
  -/
  rw [HasFiniteIntegral] at hf ⊢
  /-
    case intro
    α : Type u_1
    m : MeasurableSpace α
    H : Type u_6
    inst✝ : NormedAddCommGroup H
    m0 : MeasurableSpace α
    μ' : MeasureTheory.Measure α
    f : α → H
    hm : LE.le m m0
    hf_meas_ae : MeasureTheory.AEStronglyMeasurable f (μ'.trim hm)
    hf : LT.lt (MeasureTheory.lintegral (μ'.trim hm) fun a => ENorm.enorm (f a)) T …
    ⊢ LT.lt (MeasureTheory.lintegral μ' fun a => ENorm.enorm (f a)) Top.top
  -/
  rwa [lintegral_trim_ae hm _] at hf
  /-
    α : Type u_1
    m : MeasurableSpace α
    H : Type u_6
    inst✝ : NormedAddCommGroup H
    m0 : MeasurableSpace α
    μ' : MeasureTheory.Measure α
    f : α → H
    hm : LE.le m m0
    hf_meas_ae : MeasureTheory.AEStronglyMeasurable f (μ'.trim hm)
    hf : LT.lt (MeasureTheory.lintegral (μ'.trim hm) fun a => ENorm.enorm (f a)) T …
    ⊢ AEMeasurable (fun a => ENorm.enorm (f a)) (μ'.trim hm)
  -/
  exact AEStronglyMeasurable.ennnorm hf_meas_ae
  /-
    🎉 no goals
  -/


theorem integrable_of_forall_fin_meas_le' {μ : Measure α} (hm : m ≤ m0) [SigmaFinite (μ.trim hm)]
    (C : ℝ≥0∞) (hC : C < ∞) {f : α → E} (hf_meas : AEStronglyMeasurable f μ)
    (hf : ∀ s, MeasurableSet[m] s → μ s ≠ ∞ → (∫⁻ x in s, ‖f x‖₊ ∂μ) ≤ C) : Integrable f μ :=
  ⟨hf_meas, (lintegral_le_of_forall_fin_meas_trim_le hm C hf).trans_lt hC⟩


theorem integrable_of_forall_fin_meas_le [SigmaFinite μ] (C : ℝ≥0∞) (hC : C < ∞) {f : α → E}
    (hf_meas : AEStronglyMeasurable f μ)
    (hf : ∀ s : Set α, MeasurableSet[m] s → μ s ≠ ∞ → (∫⁻ x in s, ‖f x‖₊ ∂μ) ≤ C) :
    Integrable f μ :=
                                           /-
                                             α : Type u_1
                                             m : MeasurableSpace α
                                             μ : MeasureTheory.Measure α
                                             E : Type u_6
                                             inst✝¹ : NormedAddCommGroup E
                                             inst✝ : MeasureTheory.SigmaFinite μ
                                             C : ENNReal
                                             hC : LT.lt C Top.top
                                             f : α → E
                                             hf_meas : MeasureTheory.AEStronglyMeasurable f μ
                                             hf : ∀ (s : Set α), MeasurableSet s → Ne (μ s) Top.top → LE.le (MeasureTheory. …
                                             ⊢ MeasureTheory.SigmaFinite (μ.trim ⋯)
                                           -/
  have : SigmaFinite (μ.trim le_rfl) := by rwa [@trim_eq_self _ m]
                                           /-
                                             🎉 no goals
                                           -/
  integrable_of_forall_fin_meas_le' le_rfl C hC hf_meas hf


/-- A class of almost everywhere equal functions is `Integrable` if its function representative
is integrable. -/
def Integrable (f : α →ₘ[μ] β) : Prop :=
  MeasureTheory.Integrable f μ


theorem integrable_mk {f : α → β} (hf : AEStronglyMeasurable f μ) :
    Integrable (mk f hf : α →ₘ[μ] β) ↔ MeasureTheory.Integrable f μ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    hf : MeasureTheory.AEStronglyMeasurable f μ
    ⊢ Iff (MeasureTheory.AEEqFun.mk f hf).Integrable (MeasureTheory.Integrable f μ)
  -/
  simp only [Integrable]
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    hf : MeasureTheory.AEStronglyMeasurable f μ
    ⊢ Iff (MeasureTheory.Integrable (↑(MeasureTheory.AEEqFun.mk f hf)) μ) (Measure …
  -/
  apply integrable_congr
  /-
    case h
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    hf : MeasureTheory.AEStronglyMeasurable f μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq (↑(MeasureTheory.AEEqFun.mk f hf)) f
  -/
  exact coeFn_mk f hf
  /-
    🎉 no goals
  -/


theorem integrable_coeFn {f : α →ₘ[μ] β} : MeasureTheory.Integrable f μ ↔ Integrable f := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : MeasureTheory.AEEqFun α β μ
    ⊢ Iff (MeasureTheory.Integrable (↑f) μ) f.Integrable
  -/
  rw [← integrable_mk, mk_coeFn]
  /-
    🎉 no goals
  -/


theorem integrable_zero : Integrable (0 : α →ₘ[μ] β) :=
  (MeasureTheory.integrable_zero α β μ).congr (coeFn_mk _ _).symm


theorem Integrable.neg {f : α →ₘ[μ] β} : Integrable f → Integrable (-f) :=
  induction_on f fun _f hfm hfi => (integrable_mk _).2 ((integrable_mk hfm).1 hfi).neg


theorem integrable_iff_mem_L1 {f : α →ₘ[μ] β} : Integrable f ↔ f ∈ (α →₁[μ] β) := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : MeasureTheory.AEEqFun α β μ
    ⊢ Iff f.Integrable (Membership.mem (MeasureTheory.Lp β 1 μ) f)
  -/
  rw [← integrable_coeFn, ← memℒp_one_iff_integrable, Lp.mem_Lp_iff_memℒp]
  /-
    🎉 no goals
  -/


theorem Integrable.add {f g : α →ₘ[μ] β} : Integrable f → Integrable g → Integrable (f + g) := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f g : MeasureTheory.AEEqFun α β μ
    ⊢ f.Integrable → g.Integrable → (HAdd.hAdd f g).Integrable
  -/
  refine induction_on₂ f g fun f hf g hg hfi hgi => ?_
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f✝ g✝ : MeasureTheory.AEEqFun α β μ
    f : α → β
    hf : MeasureTheory.AEStronglyMeasurable f μ
    g : α → β
    hg : MeasureTheory.AEStronglyMeasurable g μ
    hfi : (MeasureTheory.AEEqFun.mk f hf).Integrable
    hgi : (MeasureTheory.AEEqFun.mk g hg).Integrable
    ⊢ (HAdd.hAdd (MeasureTheory.AEEqFun.mk f hf) (MeasureTheory.AEEqFun.mk g hg)). …
  -/
  simp only [integrable_mk, mk_add_mk] at hfi hgi ⊢
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f✝ g✝ : MeasureTheory.AEEqFun α β μ
    f : α → β
    hf : MeasureTheory.AEStronglyMeasurable f μ
    g : α → β
    hg : MeasureTheory.AEStronglyMeasurable g μ
    hfi : MeasureTheory.Integrable f μ
    hgi : MeasureTheory.Integrable g μ
    ⊢ MeasureTheory.Integrable (HAdd.hAdd f g) μ
  -/
  exact hfi.add hgi
  /-
    🎉 no goals
  -/


theorem Integrable.sub {f g : α →ₘ[μ] β} (hf : Integrable f) (hg : Integrable g) :
    Integrable (f - g) :=
  (sub_eq_add_neg f g).symm ▸ hf.add hg.neg


theorem Integrable.smul {c : 𝕜} {f : α →ₘ[μ] β} : Integrable f → Integrable (c • f) :=
  induction_on f fun _f hfm hfi => (integrable_mk _).2 <|
       /-
         α : Type u_1
         β : Type u_2
         m : MeasurableSpace α
         μ : MeasureTheory.Measure α
         inst✝³ : NormedAddCommGroup β
         𝕜 : Type u_6
         inst✝² : NormedRing 𝕜
         inst✝¹ : Module 𝕜 β
         inst✝ : BoundedSMul 𝕜 β
         c : 𝕜
         f : MeasureTheory.AEEqFun α β μ
         _f : α → β
         hfm : MeasureTheory.AEStronglyMeasurable _f μ
         hfi : (MeasureTheory.AEEqFun.mk _f hfm).Integrable
         ⊢ MeasureTheory.Integrable (Function.comp (fun x => HSMul.hSMul c x) ↑⟨_f, hfm …
       -/
    by simpa using ((integrable_mk hfm).1 hfi).smul c
       /-
         🎉 no goals
       -/


theorem integrable_coeFn (f : α →₁[μ] β) : Integrable f μ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp β 1 μ) x
    ⊢ MeasureTheory.Integrable (↑↑f) μ
  -/
  rw [← memℒp_one_iff_integrable]
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp β 1 μ) x
    ⊢ MeasureTheory.Memℒp (↑↑f) 1 μ
  -/
  exact Lp.memℒp f
  /-
    🎉 no goals
  -/


theorem hasFiniteIntegral_coeFn (f : α →₁[μ] β) : HasFiniteIntegral f μ :=
  (integrable_coeFn f).hasFiniteIntegral


theorem stronglyMeasurable_coeFn (f : α →₁[μ] β) : StronglyMeasurable f :=
  Lp.stronglyMeasurable f


theorem measurable_coeFn [MeasurableSpace β] [BorelSpace β] (f : α →₁[μ] β) : Measurable f :=
  (Lp.stronglyMeasurable f).measurable


theorem aestronglyMeasurable_coeFn (f : α →₁[μ] β) : AEStronglyMeasurable f μ :=
  Lp.aestronglyMeasurable f


theorem aemeasurable_coeFn [MeasurableSpace β] [BorelSpace β] (f : α →₁[μ] β) : AEMeasurable f μ :=
  (Lp.stronglyMeasurable f).measurable.aemeasurable


theorem edist_def (f g : α →₁[μ] β) : edist f g = ∫⁻ a, edist (f a) (g a) ∂μ := by
  simp only [Lp.edist_def, eLpNorm, one_ne_zero, eLpNorm'_eq_lintegral_nnnorm, Pi.sub_apply,
    one_toReal, ENNReal.rpow_one, ne_eq, not_false_eq_true, div_self, ite_false]
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp β 1 μ) x
    ⊢ Eq (ite (Eq 1 Top.top) (MeasureTheory.eLpNormEssSup (HSub.hSub ↑↑f ↑↑g) μ) ( …
  -/
  simp [edist_eq_coe_nnnorm_sub]
  /-
    🎉 no goals
  -/


theorem dist_def (f g : α →₁[μ] β) : dist f g = (∫⁻ a, edist (f a) (g a) ∂μ).toReal := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp β 1 μ) x
    ⊢ Eq (Dist.dist f g) (MeasureTheory.lintegral μ fun a => EDist.edist (↑↑f a) ( …
  -/
  simp_rw [dist_edist, edist_def]
  /-
    🎉 no goals
  -/


theorem norm_def (f : α →₁[μ] β) : ‖f‖ = (∫⁻ a, ‖f a‖₊ ∂μ).toReal := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp β 1 μ) x
    ⊢ Eq (Norm.norm f) (MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (↑↑f a) …
  -/
  simp [Lp.norm_def, eLpNorm, eLpNorm'_eq_lintegral_nnnorm]
  /-
    🎉 no goals
  -/


/-- Computing the norm of a difference between two L¹-functions. Note that this is not a
  special case of `norm_def` since `(f - g) x` and `f x - g x` are not equal
  (but only a.e.-equal). -/
theorem norm_sub_eq_lintegral (f g : α →₁[μ] β) :
    ‖f - g‖ = (∫⁻ x, (‖f x - g x‖₊ : ℝ≥0∞) ∂μ).toReal := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp β 1 μ) x
    ⊢ Eq (Norm.norm (HSub.hSub f g)) (MeasureTheory.lintegral μ fun x => ↑(NNNorm. …
  -/
  rw [norm_def]
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp β 1 μ) x
    ⊢ Eq (MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (↑↑(HSub.hSub f g) a) …
  -/
  congr 1
  /-
    case e_a
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp β 1 μ) x
    ⊢ Eq (MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (↑↑(HSub.hSub f g) a) …
  -/
  rw [lintegral_congr_ae]
  /-
    case e_a
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp β 1 μ) x
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun a => ↑(NNNorm.nnnorm (↑↑(HSub.hSub f  …
  -/
  filter_upwards [Lp.coeFn_sub f g] with _ ha
  /-
    case h
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp β 1 μ) x
    a✝ : α
    ha : Eq (↑↑(HSub.hSub f g) a✝) (HSub.hSub (↑↑f) (↑↑g) a✝)
    ⊢ Eq ↑(NNNorm.nnnorm (↑↑(HSub.hSub f g) a✝)) ↑(NNNorm.nnnorm (HSub.hSub (↑↑f a …
  -/
  simp only [ha, Pi.sub_apply]
  /-
    🎉 no goals
  -/


theorem ofReal_norm_eq_lintegral (f : α →₁[μ] β) :
    ENNReal.ofReal ‖f‖ = ∫⁻ x, (‖f x‖₊ : ℝ≥0∞) ∂μ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp β 1 μ) x
    ⊢ Eq (ENNReal.ofReal (Norm.norm f)) (MeasureTheory.lintegral μ fun x => ↑(NNNo …
  -/
  rw [norm_def, ENNReal.ofReal_toReal]
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp β 1 μ) x
    ⊢ Ne (MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (↑↑f a))) Top.top
  -/
  exact ne_of_lt (hasFiniteIntegral_coeFn f)
  /-
    🎉 no goals
  -/


/-- Computing the norm of a difference between two L¹-functions. Note that this is not a
  special case of `ofReal_norm_eq_lintegral` since `(f - g) x` and `f x - g x` are not equal
  (but only a.e.-equal). -/
theorem ofReal_norm_sub_eq_lintegral (f g : α →₁[μ] β) :
    ENNReal.ofReal ‖f - g‖ = ∫⁻ x, (‖f x - g x‖₊ : ℝ≥0∞) ∂μ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp β 1 μ) x
    ⊢ Eq (ENNReal.ofReal (Norm.norm (HSub.hSub f g))) (MeasureTheory.lintegral μ f …
  -/
  simp_rw [ofReal_norm_eq_lintegral, ← edist_eq_coe_nnnorm]
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp β 1 μ) x
    ⊢ Eq (MeasureTheory.lintegral μ fun x => EDist.edist (↑↑(HSub.hSub f g) x) 0)  …
  -/
  apply lintegral_congr_ae
  /-
    case h
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp β 1 μ) x
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun a => EDist.edist (↑↑(HSub.hSub f g) a …
  -/
  filter_upwards [Lp.coeFn_sub f g] with _ ha
  /-
    case h
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp β 1 μ) x
    a✝ : α
    ha : Eq (↑↑(HSub.hSub f g) a✝) (HSub.hSub (↑↑f) (↑↑g) a✝)
    ⊢ Eq (EDist.edist (↑↑(HSub.hSub f g) a✝) 0) (EDist.edist (HSub.hSub (↑↑f a✝) ( …
  -/
  simp only [ha, Pi.sub_apply]
  /-
    🎉 no goals
  -/


/-- Construct the equivalence class `[f]` of an integrable function `f`, as a member of the
space `L1 β 1 μ`. -/
def toL1 (f : α → β) (hf : Integrable f μ) : α →₁[μ] β :=
  (memℒp_one_iff_integrable.2 hf).toLp f


@[simp]
theorem toL1_coeFn (f : α →₁[μ] β) (hf : Integrable f μ) : hf.toL1 f = f := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp β 1 μ) x
    hf : MeasureTheory.Integrable (↑↑f) μ
    ⊢ Eq (MeasureTheory.Integrable.toL1 (↑↑f) hf) f
  -/
  simp [Integrable.toL1]
  /-
    🎉 no goals
  -/


theorem coeFn_toL1 {f : α → β} (hf : Integrable f μ) : hf.toL1 f =ᵐ[μ] f :=
  AEEqFun.coeFn_mk _ _


@[simp]
theorem toL1_zero (h : Integrable (0 : α → β) μ) : h.toL1 0 = 0 :=
  rfl


@[simp]
theorem toL1_eq_mk (f : α → β) (hf : Integrable f μ) :
    (hf.toL1 f : α →ₘ[μ] β) = AEEqFun.mk f hf.aestronglyMeasurable :=
  rfl


@[simp]
theorem toL1_eq_toL1_iff (f g : α → β) (hf : Integrable f μ) (hg : Integrable g μ) :
    toL1 f hf = toL1 g hg ↔ f =ᵐ[μ] g :=
  Memℒp.toLp_eq_toLp_iff _ _


theorem toL1_add (f g : α → β) (hf : Integrable f μ) (hg : Integrable g μ) :
    toL1 (f + g) (hf.add hg) = toL1 f hf + toL1 g hg :=
  rfl


theorem toL1_neg (f : α → β) (hf : Integrable f μ) : toL1 (-f) (Integrable.neg hf) = -toL1 f hf :=
  rfl


theorem toL1_sub (f g : α → β) (hf : Integrable f μ) (hg : Integrable g μ) :
    toL1 (f - g) (hf.sub hg) = toL1 f hf - toL1 g hg :=
  rfl


theorem norm_toL1 (f : α → β) (hf : Integrable f μ) :
    ‖hf.toL1 f‖ = ENNReal.toReal (∫⁻ a, edist (f a) 0 ∂μ) := by
  simp only [toL1, Lp.norm_toLp, eLpNorm, one_ne_zero, eLpNorm'_eq_lintegral_nnnorm, one_toReal,
    ENNReal.rpow_one, ne_eq, not_false_eq_true, div_self, ite_false]
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    hf : MeasureTheory.Integrable f μ
    ⊢ Eq (ite (Eq 1 Top.top) (MeasureTheory.eLpNormEssSup f μ) (MeasureTheory.lint …
  -/
  simp [edist_eq_coe_nnnorm]
  /-
    🎉 no goals
  -/


theorem nnnorm_toL1 {f : α → β} (hf : Integrable f μ) :
    (‖hf.toL1 f‖₊ : ℝ≥0∞) = ∫⁻ a, ‖f a‖₊ ∂μ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    hf : MeasureTheory.Integrable f μ
    ⊢ Eq (↑(NNNorm.nnnorm (MeasureTheory.Integrable.toL1 f hf))) (MeasureTheory.li …
  -/
  simpa [Integrable.toL1, eLpNorm, eLpNorm'] using ENNReal.coe_toNNReal hf.2.ne
  /-
    🎉 no goals
  -/


theorem norm_toL1_eq_lintegral_norm (f : α → β) (hf : Integrable f μ) :
    ‖hf.toL1 f‖ = ENNReal.toReal (∫⁻ a, ENNReal.ofReal ‖f a‖ ∂μ) := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    hf : MeasureTheory.Integrable f μ
    ⊢ Eq (Norm.norm (MeasureTheory.Integrable.toL1 f hf)) (MeasureTheory.lintegral …
  -/
  rw [norm_toL1, lintegral_norm_eq_lintegral_edist]
  /-
    🎉 no goals
  -/


@[simp]
theorem edist_toL1_toL1 (f g : α → β) (hf : Integrable f μ) (hg : Integrable g μ) :
    edist (hf.toL1 f) (hg.toL1 g) = ∫⁻ a, edist (f a) (g a) ∂μ := by
  simp only [toL1, Lp.edist_toLp_toLp, eLpNorm, one_ne_zero, eLpNorm'_eq_lintegral_nnnorm,
    Pi.sub_apply, one_toReal, ENNReal.rpow_one, ne_eq, not_false_eq_true, div_self, ite_false]
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f g : α → β
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g μ
    ⊢ Eq (ite (Eq 1 Top.top) (MeasureTheory.eLpNormEssSup (HSub.hSub f g) μ) (Meas …
  -/
  simp [edist_eq_coe_nnnorm_sub]
  /-
    🎉 no goals
  -/


theorem edist_toL1_zero (f : α → β) (hf : Integrable f μ) :
    edist (hf.toL1 f) 0 = ∫⁻ a, edist (f a) 0 ∂μ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    hf : MeasureTheory.Integrable f μ
    ⊢ Eq (EDist.edist (MeasureTheory.Integrable.toL1 f hf) 0) (MeasureTheory.linte …
  -/
  simp only [edist_zero_right, Lp.nnnorm_coe_ennreal, toL1_eq_mk, eLpNorm_aeeqFun]
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup β
    f : α → β
    hf : MeasureTheory.Integrable f μ
    ⊢ Eq (MeasureTheory.eLpNorm f 1 μ) (MeasureTheory.lintegral μ fun a => ↑(NNNor …
  -/
  apply eLpNorm_one_eq_lintegral_nnnorm
  /-
    🎉 no goals
  -/


theorem toL1_smul (f : α → β) (hf : Integrable f μ) (k : 𝕜) :
    toL1 (fun a => k • f a) (hf.smul k) = k • toL1 f hf :=
  rfl


theorem toL1_smul' (f : α → β) (hf : Integrable f μ) (k : 𝕜) :
    toL1 (k • f) (hf.smul k) = k • toL1 f hf :=
  rfl


lemma HasFiniteIntegral.restrict (h : HasFiniteIntegral f μ) {s : Set α} :
    HasFiniteIntegral f (μ.restrict s) := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_6
    inst✝ : NormedAddCommGroup E
    f : α → E
    h : MeasureTheory.HasFiniteIntegral f μ
    s : Set α
    ⊢ MeasureTheory.HasFiniteIntegral f (μ.restrict s)
  -/
  refine lt_of_le_of_lt ?_ h
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_6
    inst✝ : NormedAddCommGroup E
    f : α → E
    h : MeasureTheory.HasFiniteIntegral f μ
    s : Set α
    ⊢ LE.le (MeasureTheory.lintegral (μ.restrict s) fun a => ENorm.enorm (f a)) (M …
  -/
  convert lintegral_mono_set (μ := μ) (s := s) (t := univ) (f := fun x ↦ ↑‖f x‖₊) (subset_univ s)
  /-
    case h.e'_4.h.e'_3
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_6
    inst✝ : NormedAddCommGroup E
    f : α → E
    h : MeasureTheory.HasFiniteIntegral f μ
    s : Set α
    ⊢ Eq μ (μ.restrict Set.univ)
  -/
  exact Measure.restrict_univ.symm
  /-
    🎉 no goals
  -/


/-- One should usually use `MeasureTheory.Integrable.IntegrableOn` instead. -/
lemma Integrable.restrict (hf : Integrable f μ) {s : Set α} : Integrable f (μ.restrict s) :=
  hf.mono_measure Measure.restrict_le_self


theorem ContinuousLinearMap.integrable_comp {φ : α → H} (L : H →L[𝕜] E) (φ_int : Integrable φ μ) :
    Integrable (fun a : α => L (φ a)) μ :=
  ((Integrable.norm φ_int).const_mul ‖L‖).mono'
    (L.continuous.comp_aestronglyMeasurable φ_int.aestronglyMeasurable)
    (Eventually.of_forall fun a => L.le_opNorm (φ a))


@[simp]
theorem ContinuousLinearEquiv.integrable_comp_iff {φ : α → H} (L : H ≃L[𝕜] E) :
    Integrable (fun a : α ↦ L (φ a)) μ ↔ Integrable φ μ :=
              /-
                α : Type u_1
                m : MeasurableSpace α
                μ : MeasureTheory.Measure α
                E : Type u_6
                inst✝⁴ : NormedAddCommGroup E
                𝕜 : Type u_7
                inst✝³ : NontriviallyNormedField 𝕜
                inst✝² : NormedSpace 𝕜 E
                H : Type u_8
                inst✝¹ : NormedAddCommGroup H
                inst✝ : NormedSpace 𝕜 H
                φ : α → H
                L : ContinuousLinearEquiv (RingHom.id 𝕜) H E
                h : MeasureTheory.Integrable (fun a => L (φ a)) μ
                ⊢ MeasureTheory.Integrable φ μ
              -/
  ⟨fun h ↦ by simpa using ContinuousLinearMap.integrable_comp (L.symm : E →L[𝕜] H) h,
              /-
                🎉 no goals
              -/
  fun h ↦ ContinuousLinearMap.integrable_comp (L : H →L[𝕜] E) h⟩


@[simp]
theorem LinearIsometryEquiv.integrable_comp_iff {φ : α → H} (L : H ≃ₗᵢ[𝕜] E) :
    Integrable (fun a : α ↦ L (φ a)) μ ↔ Integrable φ μ :=
  ContinuousLinearEquiv.integrable_comp_iff (L : H ≃L[𝕜] E)


theorem MeasureTheory.Integrable.apply_continuousLinearMap {φ : α → H →L[𝕜] E}
    (φ_int : Integrable φ μ) (v : H) : Integrable (fun a => φ a v) μ :=
  (ContinuousLinearMap.apply 𝕜 _ v).integrable_comp φ_int


lemma Integrable.fst {f : α → E × F} (hf : Integrable f μ) : Integrable (fun x ↦ (f x).1) μ :=
  (ContinuousLinearMap.fst ℝ E F).integrable_comp hf


lemma Integrable.snd {f : α → E × F} (hf : Integrable f μ) : Integrable (fun x ↦ (f x).2) μ :=
  (ContinuousLinearMap.snd ℝ E F).integrable_comp hf


lemma integrable_prod {f : α → E × F} :
    Integrable f μ ↔ Integrable (fun x ↦ (f x).1) μ ∧ Integrable (fun x ↦ (f x).2) μ :=
  ⟨fun h ↦ ⟨h.fst, h.snd⟩, fun h ↦ h.1.prod_mk h.2⟩


