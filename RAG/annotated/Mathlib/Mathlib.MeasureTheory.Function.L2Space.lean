theorem Memℒp.integrable_sq {f : α → ℝ} (h : Memℒp f 2 μ) : Integrable (fun x => f x ^ 2) μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    h : MeasureTheory.Memℒp f 2 μ
    ⊢ MeasureTheory.Integrable (fun x => HPow.hPow (f x) 2) μ
  -/
  simpa [← memℒp_one_iff_integrable] using h.norm_rpow two_ne_zero ENNReal.two_ne_top
  /-
    🎉 no goals
  -/


theorem memℒp_two_iff_integrable_sq_norm {f : α → F} (hf : AEStronglyMeasurable f μ) :
    Memℒp f 2 μ ↔ Integrable (fun x => ‖f x‖ ^ 2) μ := by
  /-
    α : Type u_1
    F : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    f : α → F
    hf : MeasureTheory.AEStronglyMeasurable f μ
    ⊢ Iff (MeasureTheory.Memℒp f 2 μ) (MeasureTheory.Integrable (fun x => HPow.hPo …
  -/
  rw [← memℒp_one_iff_integrable]
  /-
    α : Type u_1
    F : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    f : α → F
    hf : MeasureTheory.AEStronglyMeasurable f μ
    ⊢ Iff (MeasureTheory.Memℒp f 2 μ) (MeasureTheory.Memℒp (fun x => HPow.hPow (No …
  -/
  convert (memℒp_norm_rpow_iff hf two_ne_zero ENNReal.two_ne_top).symm
    /-
      case h.e'_2.h.e'_6.h
      α : Type u_1
      F : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      f : α → F
      hf : MeasureTheory.AEStronglyMeasurable f μ
      x✝ : α
      ⊢ Eq (HPow.hPow (Norm.norm (f x✝)) 2) (HPow.hPow (Norm.norm (f x✝)) (ENNReal.t …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h.e'_2.h.e'_7
      α : Type u_1
      F : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : NormedAddCommGroup F
      f : α → F
      hf : MeasureTheory.AEStronglyMeasurable f μ
      ⊢ Eq 1 (2 / 2)
    -/
  · rw [div_eq_mul_inv, ENNReal.mul_inv_cancel two_ne_zero ENNReal.two_ne_top]
    /-
      🎉 no goals
    -/


theorem memℒp_two_iff_integrable_sq {f : α → ℝ} (hf : AEStronglyMeasurable f μ) :
    Memℒp f 2 μ ↔ Integrable (fun x => f x ^ 2) μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.AEStronglyMeasurable f μ
    ⊢ Iff (MeasureTheory.Memℒp f 2 μ) (MeasureTheory.Integrable (fun x => HPow.hPo …
  -/
  convert memℒp_two_iff_integrable_sq_norm hf using 3
  /-
    case h.e'_2.h.e'_6.h
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → Real
    hf : MeasureTheory.AEStronglyMeasurable f μ
    x✝ : α
    ⊢ Eq (HPow.hPow (f x✝) 2) (HPow.hPow (Norm.norm (f x✝)) 2)
  -/
  simp
  /-
    🎉 no goals
  -/


local notation "⟪" x ", " y "⟫" => @inner 𝕜 E _ x y


theorem Memℒp.const_inner (c : E) {f : α → E} (hf : Memℒp f p μ) : Memℒp (fun a => ⟪c, f a⟫) p μ :=
  hf.of_le_mul (AEStronglyMeasurable.inner aestronglyMeasurable_const hf.1)
    (Eventually.of_forall fun _ => norm_inner_le_norm _ _)


theorem Memℒp.inner_const {f : α → E} (hf : Memℒp f p μ) (c : E) : Memℒp (fun a => ⟪f a, c⟫) p μ :=
  hf.of_le_mul (c := ‖c‖) (AEStronglyMeasurable.inner hf.1 aestronglyMeasurable_const)
                                      /-
                                        α : Type u_1
                                        m : MeasurableSpace α
                                        p : ENNReal
                                        μ : MeasureTheory.Measure α
                                        E : Type u_2
                                        𝕜 : Type u_3
                                        inst✝² : RCLike 𝕜
                                        inst✝¹ : NormedAddCommGroup E
                                        inst✝ : InnerProductSpace 𝕜 E
                                        f : α → E
                                        hf : MeasureTheory.Memℒp f p μ
                                        c : E
                                        x : α
                                        ⊢ LE.le (Norm.norm (Inner.inner (f x) c)) (HMul.hMul (Norm.norm c) (Norm.norm  …
                                      -/
    (Eventually.of_forall fun x => by rw [mul_comm]; exact norm_inner_le_norm _ _)
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem Integrable.const_inner (c : E) (hf : Integrable f μ) :
    Integrable (fun x => ⟪c, f x⟫) μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_2
    𝕜 : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    f : α → E
    c : E
    hf : MeasureTheory.Integrable f μ
    ⊢ MeasureTheory.Integrable (fun x => Inner.inner c (f x)) μ
  -/
  rw [← memℒp_one_iff_integrable] at hf ⊢; exact hf.const_inner c
                                           /-
                                             🎉 no goals
                                           -/


theorem Integrable.inner_const (hf : Integrable f μ) (c : E) :
    Integrable (fun x => ⟪f x, c⟫) μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_2
    𝕜 : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    f : α → E
    hf : MeasureTheory.Integrable f μ
    c : E
    ⊢ MeasureTheory.Integrable (fun x => Inner.inner (f x) c) μ
  -/
  rw [← memℒp_one_iff_integrable] at hf ⊢; exact hf.inner_const c
                                           /-
                                             🎉 no goals
                                           -/


theorem _root_.integral_inner {f : α → E} (hf : Integrable f μ) (c : E) :
    ∫ x, ⟪c, f x⟫ ∂μ = ⟪c, ∫ x, f x ∂μ⟫ :=
  ((innerSL 𝕜 c).restrictScalars ℝ).integral_comp_comm hf


theorem _root_.integral_eq_zero_of_forall_integral_inner_eq_zero (f : α → E) (hf : Integrable f μ)
    (hf_int : ∀ c : E, ∫ x, ⟪c, f x⟫ ∂μ = 0) : ∫ x, f x ∂μ = 0 := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_2
    𝕜 : Type u_3
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : InnerProductSpace 𝕜 E
    inst✝¹ : CompleteSpace E
    inst✝ : NormedSpace Real E
    f : α → E
    hf : MeasureTheory.Integrable f μ
    hf_int : ∀ (c : E), Eq (MeasureTheory.integral μ fun x => Inner.inner c (f x)) 0
    ⊢ Eq (MeasureTheory.integral μ fun x => f x) 0
  -/
  specialize hf_int (∫ x, f x ∂μ); rwa [integral_inner hf, inner_self_eq_zero] at hf_int
                                   /-
                                     🎉 no goals
                                   -/


local notation "⟪" x ", " y "⟫" => @inner 𝕜 _ _ x y


theorem eLpNorm_rpow_two_norm_lt_top (f : Lp F 2 μ) :
    eLpNorm (fun x => ‖f x‖ ^ (2 : ℝ)) 1 μ < ∞ := by
  /-
    α : Type u_1
    F : Type u_3
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp F 2 μ) x
    ⊢ LT.lt (MeasureTheory.eLpNorm (fun x => HPow.hPow (Norm.norm (↑↑f x)) 2) 1 μ) …
  -/
  have h_two : ENNReal.ofReal (2 : ℝ) = 2 := by simp [zero_le_one]
  /-
    α : Type u_1
    F : Type u_3
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp F 2 μ) x
    h_two : Eq (ENNReal.ofReal 2) 2
    ⊢ LT.lt (MeasureTheory.eLpNorm (fun x => HPow.hPow (Norm.norm (↑↑f x)) 2) 1 μ) …
  -/
  rw [eLpNorm_norm_rpow f zero_lt_two, one_mul, h_two]
  /-
    α : Type u_1
    F : Type u_3
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedAddCommGroup F
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp F 2 μ) x
    h_two : Eq (ENNReal.ofReal 2) 2
    ⊢ LT.lt (HPow.hPow (MeasureTheory.eLpNorm (↑↑f) 2 μ) 2) Top.top
  -/
  exact ENNReal.rpow_lt_top_of_nonneg zero_le_two (Lp.eLpNorm_ne_top f)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm_rpow_two_norm_lt_top := eLpNorm_rpow_two_norm_lt_top


theorem eLpNorm_inner_lt_top (f g : α →₂[μ] E) : eLpNorm (fun x : α => ⟪f x, g x⟫) 1 μ < ∞ := by
  have h : ∀ x, ‖⟪f x, g x⟫‖ ≤ ‖‖f x‖ ^ (2 : ℝ) + ‖g x‖ ^ (2 : ℝ)‖ := by
    intro x
    rw [← @Nat.cast_two ℝ, Real.rpow_natCast, Real.rpow_natCast]
    calc
      ‖⟪f x, g x⟫‖ ≤ ‖f x‖ * ‖g x‖ := norm_inner_le_norm _ _
      _ ≤ 2 * ‖f x‖ * ‖g x‖ :=
        (mul_le_mul_of_nonneg_right (le_mul_of_one_le_left (norm_nonneg _) one_le_two)
          (norm_nonneg _))
      -- TODO(kmill): the type ascription is getting around an elaboration error
      _ ≤ ‖(‖f x‖ ^ 2 + ‖g x‖ ^ 2 : ℝ)‖ := (two_mul_le_add_sq _ _).trans (le_abs_self _)
  /-
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_4
    inst✝³ : RCLike 𝕜
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp E 2 μ) x
    h : ∀ (x : α), LE.le (Norm.norm (Inner.inner (↑↑f x) (↑↑g x))) (Norm.norm (HAd …
    ⊢ LT.lt (MeasureTheory.eLpNorm (fun x => Inner.inner (↑↑f x) (↑↑g x)) 1 μ) Top …
  -/
  refine (eLpNorm_mono_ae (ae_of_all _ h)).trans_lt ((eLpNorm_add_le ?_ ?_ le_rfl).trans_lt ?_)
    /-
      case refine_1
      α : Type u_1
      E : Type u_2
      𝕜 : Type u_4
      inst✝³ : RCLike 𝕜
      inst✝² : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      f g : Subtype fun x => Membership.mem (MeasureTheory.Lp E 2 μ) x
      h : ∀ (x : α), LE.le (Norm.norm (Inner.inner (↑↑f x) (↑↑g x))) (Norm.norm (HAd …
      ⊢ MeasureTheory.AEStronglyMeasurable (fun a => HPow.hPow (Norm.norm (↑↑f a)) 2 …
    -/
  · exact ((Lp.aestronglyMeasurable f).norm.aemeasurable.pow_const _).aestronglyMeasurable
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      E : Type u_2
      𝕜 : Type u_4
      inst✝³ : RCLike 𝕜
      inst✝² : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      f g : Subtype fun x => Membership.mem (MeasureTheory.Lp E 2 μ) x
      h : ∀ (x : α), LE.le (Norm.norm (Inner.inner (↑↑f x) (↑↑g x))) (Norm.norm (HAd …
      ⊢ MeasureTheory.AEStronglyMeasurable (fun a => HPow.hPow (Norm.norm (↑↑g a)) 2 …
    -/
  · exact ((Lp.aestronglyMeasurable g).norm.aemeasurable.pow_const _).aestronglyMeasurable
    /-
      🎉 no goals
    -/
  /-
    case refine_3
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_4
    inst✝³ : RCLike 𝕜
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp E 2 μ) x
    h : ∀ (x : α), LE.le (Norm.norm (Inner.inner (↑↑f x) (↑↑g x))) (Norm.norm (HAd …
    ⊢ LT.lt (HAdd.hAdd (MeasureTheory.eLpNorm (fun a => HPow.hPow (Norm.norm (↑↑f  …
  -/
  rw [ENNReal.add_lt_top]
  /-
    case refine_3
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_4
    inst✝³ : RCLike 𝕜
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp E 2 μ) x
    h : ∀ (x : α), LE.le (Norm.norm (Inner.inner (↑↑f x) (↑↑g x))) (Norm.norm (HAd …
    ⊢ And (LT.lt (MeasureTheory.eLpNorm (fun a => HPow.hPow (Norm.norm (↑↑f a)) 2) …
  -/
  exact ⟨eLpNorm_rpow_two_norm_lt_top f, eLpNorm_rpow_two_norm_lt_top g⟩
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm_inner_lt_top := eLpNorm_inner_lt_top


instance : Inner 𝕜 (α →₂[μ] E) :=
  ⟨fun f g => ∫ a, ⟪f a, g a⟫ ∂μ⟩


theorem inner_def (f g : α →₂[μ] E) : ⟪f, g⟫ = ∫ a : α, ⟪f a, g a⟫ ∂μ :=
  rfl


theorem integral_inner_eq_sq_eLpNorm (f : α →₂[μ] E) :
    ∫ a, ⟪f a, f a⟫ ∂μ = ENNReal.toReal (∫⁻ a, (‖f a‖₊ : ℝ≥0∞) ^ (2 : ℝ) ∂μ) := by
  /-
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_4
    inst✝³ : RCLike 𝕜
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 2 μ) x
    ⊢ Eq (MeasureTheory.integral μ fun a => Inner.inner (↑↑f a) (↑↑f a)) ↑(Measure …
  -/
  simp_rw [inner_self_eq_norm_sq_to_K]
  /-
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_4
    inst✝³ : RCLike 𝕜
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 2 μ) x
    ⊢ Eq (MeasureTheory.integral μ fun a => HPow.hPow (↑(Norm.norm (↑↑f a))) 2) ↑( …
  -/
  norm_cast
  /-
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_4
    inst✝³ : RCLike 𝕜
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 2 μ) x
    ⊢ Eq (MeasureTheory.integral μ fun x => HPow.hPow (Norm.norm (↑↑f x)) 2) (Meas …
  -/
  rw [integral_eq_lintegral_of_nonneg_ae]
  /-
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_4
    inst✝³ : RCLike 𝕜
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 2 μ) x
    ⊢ Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HPow.hPow (Norm.norm  …
  -/
  rotate_left
    /-
      case hf
      α : Type u_1
      E : Type u_2
      𝕜 : Type u_4
      inst✝³ : RCLike 𝕜
      inst✝² : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 2 μ) x
      ⊢ (MeasureTheory.ae μ).EventuallyLE 0 fun x => HPow.hPow (Norm.norm (↑↑f x)) 2
    -/
  · exact Filter.Eventually.of_forall fun x => sq_nonneg _
    /-
      🎉 no goals
    -/
    /-
      case hfm
      α : Type u_1
      E : Type u_2
      𝕜 : Type u_4
      inst✝³ : RCLike 𝕜
      inst✝² : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 2 μ) x
      ⊢ MeasureTheory.AEStronglyMeasurable (fun x => HPow.hPow (Norm.norm (↑↑f x)) 2 …
    -/
  · exact ((Lp.aestronglyMeasurable f).norm.aemeasurable.pow_const _).aestronglyMeasurable
    /-
      🎉 no goals
    -/
  /-
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_4
    inst✝³ : RCLike 𝕜
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 2 μ) x
    ⊢ Eq (MeasureTheory.lintegral μ fun a => ENNReal.ofReal (HPow.hPow (Norm.norm  …
  -/
  congr
  /-
    case e_a.e_f
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_4
    inst✝³ : RCLike 𝕜
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 2 μ) x
    ⊢ Eq (fun a => ENNReal.ofReal (HPow.hPow (Norm.norm (↑↑f a)) 2)) fun a => ↑(HP …
  -/
  ext1 x
  /-
    case e_a.e_f.h
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_4
    inst✝³ : RCLike 𝕜
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 2 μ) x
    x : α
    ⊢ Eq (ENNReal.ofReal (HPow.hPow (Norm.norm (↑↑f x)) 2)) ↑(HPow.hPow (NNNorm.nn …
  -/
  have h_two : (2 : ℝ) = ((2 : ℕ) : ℝ) := by simp
  rw [← Real.rpow_natCast _ 2, ← h_two, ←
    ENNReal.ofReal_rpow_of_nonneg (norm_nonneg _) zero_le_two, ofReal_norm_eq_coe_nnnorm]
  /-
    case e_a.e_f.h
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_4
    inst✝³ : RCLike 𝕜
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 2 μ) x
    x : α
    h_two : Eq 2 ↑2
    ⊢ Eq (HPow.hPow (↑(NNNorm.nnnorm (↑↑f x))) 2) ↑(HPow.hPow (NNNorm.nnnorm (↑↑f  …
  -/
  norm_cast
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias integral_inner_eq_sq_snorm := integral_inner_eq_sq_eLpNorm


private theorem norm_sq_eq_inner' (f : α →₂[μ] E) : ‖f‖ ^ 2 = RCLike.re ⟪f, f⟫ := by
  /-
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_4
    inst✝³ : RCLike 𝕜
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 2 μ) x
    ⊢ Eq (HPow.hPow (Norm.norm f) 2) (RCLike.re (Inner.inner f f))
  -/
  have h_two : (2 : ℝ≥0∞).toReal = 2 := by simp
  rw [inner_def, integral_inner_eq_sq_eLpNorm, norm_def, ← ENNReal.toReal_pow, RCLike.ofReal_re,
    ENNReal.toReal_eq_toReal (ENNReal.pow_ne_top (Lp.eLpNorm_ne_top f)) _]
  · rw [← ENNReal.rpow_natCast, eLpNorm_eq_eLpNorm' two_ne_zero ENNReal.two_ne_top, eLpNorm', ←
      ENNReal.rpow_mul, one_div, h_two]
    /-
      α : Type u_1
      E : Type u_2
      𝕜 : Type u_4
      inst✝³ : RCLike 𝕜
      inst✝² : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 2 μ) x
      h_two : Eq (ENNReal.toReal 2) 2
      ⊢ Eq (HPow.hPow (MeasureTheory.lintegral μ fun a => HPow.hPow (ENorm.enorm (↑↑ …
    -/
    simp [enorm_eq_nnnorm]
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      E : Type u_2
      𝕜 : Type u_4
      inst✝³ : RCLike 𝕜
      inst✝² : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 2 μ) x
      h_two : Eq (ENNReal.toReal 2) 2
      ⊢ Ne (MeasureTheory.lintegral μ fun a => HPow.hPow (↑(NNNorm.nnnorm (↑↑f a)))  …
    -/
  · refine (lintegral_rpow_nnnorm_lt_top_of_eLpNorm'_lt_top zero_lt_two ?_).ne
    /-
      α : Type u_1
      E : Type u_2
      𝕜 : Type u_4
      inst✝³ : RCLike 𝕜
      inst✝² : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 2 μ) x
      h_two : Eq (ENNReal.toReal 2) 2
      ⊢ LT.lt (MeasureTheory.eLpNorm' (↑↑f) 2 μ) Top.top
    -/
    rw [← h_two, ← eLpNorm_eq_eLpNorm' two_ne_zero ENNReal.two_ne_top]
    /-
      α : Type u_1
      E : Type u_2
      𝕜 : Type u_4
      inst✝³ : RCLike 𝕜
      inst✝² : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : NormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 2 μ) x
      h_two : Eq (ENNReal.toReal 2) 2
      ⊢ LT.lt (MeasureTheory.eLpNorm (↑↑f) 2 μ) Top.top
    -/
    exact Lp.eLpNorm_lt_top f
    /-
      🎉 no goals
    -/


theorem mem_L1_inner (f g : α →₂[μ] E) :
    AEEqFun.mk (fun x => ⟪f x, g x⟫)
        ((Lp.aestronglyMeasurable f).inner (Lp.aestronglyMeasurable g)) ∈
      Lp 𝕜 1 μ := by
  /-
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_4
    inst✝³ : RCLike 𝕜
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp E 2 μ) x
    ⊢ Membership.mem (MeasureTheory.Lp 𝕜 1 μ) (MeasureTheory.AEEqFun.mk (fun x =>  …
  -/
  simp_rw [mem_Lp_iff_eLpNorm_lt_top, eLpNorm_aeeqFun]; exact eLpNorm_inner_lt_top f g
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem integrable_inner (f g : α →₂[μ] E) : Integrable (fun x : α => ⟪f x, g x⟫) μ :=
  (integrable_congr
        (AEEqFun.coeFn_mk (fun x => ⟪f x, g x⟫)
          ((Lp.aestronglyMeasurable f).inner (Lp.aestronglyMeasurable g)))).mp
    (AEEqFun.integrable_iff_mem_L1.mpr (mem_L1_inner f g))


private theorem add_left' (f f' g : α →₂[μ] E) : ⟪f + f', g⟫ = inner f g + inner f' g := by
  simp_rw [inner_def, ← integral_add (integrable_inner (𝕜 := 𝕜) f g) (integrable_inner f' g),
    ← inner_add_left]
  /-
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_4
    inst✝³ : RCLike 𝕜
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    f f' g : Subtype fun x => Membership.mem (MeasureTheory.Lp E 2 μ) x
    ⊢ Eq (MeasureTheory.integral μ fun a => Inner.inner (↑↑(HAdd.hAdd f f') a) (↑↑ …
  -/
  refine integral_congr_ae ((coeFn_add f f').mono fun x hx => ?_)
  -- Porting note: was
  -- congr
  -- rwa [Pi.add_apply] at hx
  /-
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_4
    inst✝³ : RCLike 𝕜
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    f f' g : Subtype fun x => Membership.mem (MeasureTheory.Lp E 2 μ) x
    x : α
    hx : Eq (↑↑(HAdd.hAdd f f') x) (HAdd.hAdd (↑↑f) (↑↑f') x)
    ⊢ Eq ((fun a => Inner.inner (↑↑(HAdd.hAdd f f') a) (↑↑g a)) x) ((fun a => Inne …
  -/
  simp only
  /-
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_4
    inst✝³ : RCLike 𝕜
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    f f' g : Subtype fun x => Membership.mem (MeasureTheory.Lp E 2 μ) x
    x : α
    hx : Eq (↑↑(HAdd.hAdd f f') x) (HAdd.hAdd (↑↑f) (↑↑f') x)
    ⊢ Eq (Inner.inner (↑↑(HAdd.hAdd f f') x) (↑↑g x)) (Inner.inner (HAdd.hAdd (↑↑f …
  -/
  congr
  /-
    🎉 no goals
  -/


private theorem smul_left' (f g : α →₂[μ] E) (r : 𝕜) : ⟪r • f, g⟫ = conj r * inner f g := by
  /-
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_4
    inst✝³ : RCLike 𝕜
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp E 2 μ) x
    r : 𝕜
    ⊢ Eq (Inner.inner (HSMul.hSMul r f) g) (HMul.hMul ((starRingEnd 𝕜) r) (Inner.i …
  -/
  rw [inner_def, inner_def, ← smul_eq_mul, ← integral_smul]
  /-
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_4
    inst✝³ : RCLike 𝕜
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp E 2 μ) x
    r : 𝕜
    ⊢ Eq (MeasureTheory.integral μ fun a => Inner.inner (↑↑(HSMul.hSMul r f) a) (↑ …
  -/
  refine integral_congr_ae ((coeFn_smul r f).mono fun x hx => ?_)
  /-
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_4
    inst✝³ : RCLike 𝕜
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp E 2 μ) x
    r : 𝕜
    x : α
    hx : Eq (↑↑(HSMul.hSMul r f) x) (HSMul.hSMul r (↑↑f) x)
    ⊢ Eq ((fun a => Inner.inner (↑↑(HSMul.hSMul r f) a) (↑↑g a)) x) ((fun a => HSM …
  -/
  simp only
  /-
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_4
    inst✝³ : RCLike 𝕜
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp E 2 μ) x
    r : 𝕜
    x : α
    hx : Eq (↑↑(HSMul.hSMul r f) x) (HSMul.hSMul r (↑↑f) x)
    ⊢ Eq (Inner.inner (↑↑(HSMul.hSMul r f) x) (↑↑g x)) (HSMul.hSMul ((starRingEnd  …
  -/
  rw [smul_eq_mul, ← inner_smul_left, hx, Pi.smul_apply]
  /-
    🎉 no goals
  -/
  -- Porting note: was
  -- rw [smul_eq_mul, ← inner_smul_left]
  -- congr
  -- rwa [Pi.smul_apply] at hx


instance innerProductSpace : InnerProductSpace 𝕜 (α →₂[μ] E) where
  norm_sq_eq_inner := norm_sq_eq_inner'
                      /-
                        α : Type u_1
                        E : Type u_2
                        F : Type u_3
                        𝕜 : Type u_4
                        inst✝⁴ : RCLike 𝕜
                        inst✝³ : MeasurableSpace α
                        μ : MeasureTheory.Measure α
                        inst✝² : NormedAddCommGroup E
                        inst✝¹ : InnerProductSpace 𝕜 E
                        inst✝ : NormedAddCommGroup F
                        x✝¹ x✝ : Subtype fun x => Membership.mem (MeasureTheory.Lp E 2 μ) x
                        ⊢ Eq ((starRingEnd 𝕜) (Inner.inner x✝ x✝¹)) (Inner.inner x✝¹ x✝)
                      -/
  conj_symm _ _ := by simp_rw [inner_def, ← integral_conj, inner_conj_symm]
                      /-
                        🎉 no goals
                      -/
  add_left := add_left'
  smul_left := smul_left'


/-- The inner product in `L2` of the indicator of a set `indicatorConstLp 2 hs hμs c` and `f` is
equal to the integral of the inner product over `s`: `∫ x in s, ⟪c, f x⟫ ∂μ`. -/
theorem inner_indicatorConstLp_eq_setIntegral_inner (f : Lp E 2 μ) (hs : MeasurableSet s) (c : E)
    (hμs : μ s ≠ ∞) : (⟪indicatorConstLp 2 hs hμs c, f⟫ : 𝕜) = ∫ x in s, ⟪c, f x⟫ ∂μ := by
  /-
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_4
    inst✝³ : RCLike 𝕜
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    s : Set α
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 2 μ) x
    hs : MeasurableSet s
    c : E
    hμs : Ne (μ s) Top.top
    ⊢ Eq (Inner.inner (MeasureTheory.indicatorConstLp 2 hs hμs c) f) (MeasureTheor …
  -/
  rw [inner_def, ← integral_add_compl hs (L2.integrable_inner _ f)]
  have h_left : (∫ x in s, ⟪(indicatorConstLp 2 hs hμs c) x, f x⟫ ∂μ) = ∫ x in s, ⟪c, f x⟫ ∂μ := by
    suffices h_ae_eq : ∀ᵐ x ∂μ, x ∈ s → ⟪indicatorConstLp 2 hs hμs c x, f x⟫ = ⟪c, f x⟫ from
      setIntegral_congr_ae hs h_ae_eq
    have h_indicator : ∀ᵐ x : α ∂μ, x ∈ s → indicatorConstLp 2 hs hμs c x = c :=
      indicatorConstLp_coeFn_mem
    refine h_indicator.mono fun x hx hxs => ?_
    congr
    exact hx hxs
  have h_right : (∫ x in sᶜ, ⟪(indicatorConstLp 2 hs hμs c) x, f x⟫ ∂μ) = 0 := by
    suffices h_ae_eq : ∀ᵐ x ∂μ, x ∉ s → ⟪indicatorConstLp 2 hs hμs c x, f x⟫ = 0 by
      simp_rw [← Set.mem_compl_iff] at h_ae_eq
      suffices h_int_zero :
          (∫ x in sᶜ, inner (indicatorConstLp 2 hs hμs c x) (f x) ∂μ) = ∫ _ in sᶜ, (0 : 𝕜) ∂μ by
        rw [h_int_zero]
        simp
      exact setIntegral_congr_ae hs.compl h_ae_eq
    have h_indicator : ∀ᵐ x : α ∂μ, x ∉ s → indicatorConstLp 2 hs hμs c x = 0 :=
      indicatorConstLp_coeFn_nmem
    refine h_indicator.mono fun x hx hxs => ?_
    rw [hx hxs]
    exact inner_zero_left _
  /-
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_4
    inst✝³ : RCLike 𝕜
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    s : Set α
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 2 μ) x
    hs : MeasurableSet s
    c : E
    hμs : Ne (μ s) Top.top
    h_left : Eq (MeasureTheory.integral (μ.restrict s) fun x => Inner.inner (↑↑(Me …
    h_right : Eq (MeasureTheory.integral (μ.restrict (HasCompl.compl s)) fun x =>  …
    ⊢ Eq (HAdd.hAdd (MeasureTheory.integral (μ.restrict s) fun x => Inner.inner (↑ …
  -/
  rw [h_left, h_right, add_zero]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias inner_indicatorConstLp_eq_set_integral_inner :=
  inner_indicatorConstLp_eq_setIntegral_inner


/-- The inner product in `L2` of the indicator of a set `indicatorConstLp 2 hs hμs c` and `f` is
equal to the inner product of the constant `c` and the integral of `f` over `s`. -/
theorem inner_indicatorConstLp_eq_inner_setIntegral [CompleteSpace E] [NormedSpace ℝ E]
    (hs : MeasurableSet s) (hμs : μ s ≠ ∞) (c : E) (f : Lp E 2 μ) :
    (⟪indicatorConstLp 2 hs hμs c, f⟫ : 𝕜) = ⟪c, ∫ x in s, f x ∂μ⟫ := by
  rw [← integral_inner (integrableOn_Lp_of_measure_ne_top f fact_one_le_two_ennreal.elim hμs),
    L2.inner_indicatorConstLp_eq_setIntegral_inner]


@[deprecated (since := "2024-04-17")]
alias inner_indicatorConstLp_eq_inner_set_integral :=
  inner_indicatorConstLp_eq_inner_setIntegral


/-- The inner product in `L2` of the indicator of a set `indicatorConstLp 2 hs hμs (1 : 𝕜)` and
a real or complex function `f` is equal to the integral of `f` over `s`. -/
theorem inner_indicatorConstLp_one (hs : MeasurableSet s) (hμs : μ s ≠ ∞) (f : Lp 𝕜 2 μ) :
    ⟪indicatorConstLp 2 hs hμs (1 : 𝕜), f⟫ = ∫ x in s, f x ∂μ := by
  /-
    α : Type u_1
    𝕜 : Type u_4
    inst✝¹ : RCLike 𝕜
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp 𝕜 2 μ) x
    ⊢ Eq (Inner.inner (MeasureTheory.indicatorConstLp 2 hs hμs 1) f) (MeasureTheor …
  -/
  rw [L2.inner_indicatorConstLp_eq_inner_setIntegral 𝕜 hs hμs (1 : 𝕜) f]; simp
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


local notation "⟪" x ", " y "⟫" => @inner 𝕜 (α →₂[μ] 𝕜) _ x y

-- Porting note: added `(E := 𝕜)`

/-- For bounded continuous functions `f`, `g` on a finite-measure topological space `α`, the L^2
inner product is the integral of their pointwise inner product. -/
theorem BoundedContinuousFunction.inner_toLp (f g : α →ᵇ 𝕜) :
    ⟪BoundedContinuousFunction.toLp (E := 𝕜) 2 μ 𝕜 f,
        BoundedContinuousFunction.toLp (E := 𝕜) 2 μ 𝕜 g⟫ =
      ∫ x, conj (f x) * g x ∂μ := by
  /-
    α : Type u_1
    𝕜 : Type u_2
    inst✝⁴ : TopologicalSpace α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    inst✝¹ : RCLike 𝕜
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f g : BoundedContinuousFunction α 𝕜
    ⊢ Eq (Inner.inner ((BoundedContinuousFunction.toLp 2 μ 𝕜) f) ((BoundedContinuo …
  -/
  apply integral_congr_ae
  /-
    case h
    α : Type u_1
    𝕜 : Type u_2
    inst✝⁴ : TopologicalSpace α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    inst✝¹ : RCLike 𝕜
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f g : BoundedContinuousFunction α 𝕜
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun a => Inner.inner (↑↑((BoundedContinuo …
  -/
  have hf_ae := f.coeFn_toLp 2 μ 𝕜
  /-
    case h
    α : Type u_1
    𝕜 : Type u_2
    inst✝⁴ : TopologicalSpace α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    inst✝¹ : RCLike 𝕜
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f g : BoundedContinuousFunction α 𝕜
    hf_ae : (MeasureTheory.ae μ).EventuallyEq ↑↑((BoundedContinuousFunction.toLp 2 …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun a => Inner.inner (↑↑((BoundedContinuo …
  -/
  have hg_ae := g.coeFn_toLp 2 μ 𝕜
  /-
    case h
    α : Type u_1
    𝕜 : Type u_2
    inst✝⁴ : TopologicalSpace α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    inst✝¹ : RCLike 𝕜
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f g : BoundedContinuousFunction α 𝕜
    hf_ae : (MeasureTheory.ae μ).EventuallyEq ↑↑((BoundedContinuousFunction.toLp 2 …
    hg_ae : (MeasureTheory.ae μ).EventuallyEq ↑↑((BoundedContinuousFunction.toLp 2 …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun a => Inner.inner (↑↑((BoundedContinuo …
  -/
  filter_upwards [hf_ae, hg_ae] with _ hf hg
  /-
    case h
    α : Type u_1
    𝕜 : Type u_2
    inst✝⁴ : TopologicalSpace α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    inst✝¹ : RCLike 𝕜
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f g : BoundedContinuousFunction α 𝕜
    hf_ae : (MeasureTheory.ae μ).EventuallyEq ↑↑((BoundedContinuousFunction.toLp 2 …
    hg_ae : (MeasureTheory.ae μ).EventuallyEq ↑↑((BoundedContinuousFunction.toLp 2 …
    a✝ : α
    hf : Eq (↑↑((BoundedContinuousFunction.toLp 2 μ 𝕜) f) a✝) (f a✝)
    hg : Eq (↑↑((BoundedContinuousFunction.toLp 2 μ 𝕜) g) a✝) (g a✝)
    ⊢ Eq (Inner.inner (↑↑((BoundedContinuousFunction.toLp 2 μ 𝕜) f) a✝) (↑↑((Bound …
  -/
  rw [hf, hg]
  /-
    case h
    α : Type u_1
    𝕜 : Type u_2
    inst✝⁴ : TopologicalSpace α
    inst✝³ : MeasurableSpace α
    inst✝² : BorelSpace α
    inst✝¹ : RCLike 𝕜
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure μ
    f g : BoundedContinuousFunction α 𝕜
    hf_ae : (MeasureTheory.ae μ).EventuallyEq ↑↑((BoundedContinuousFunction.toLp 2 …
    hg_ae : (MeasureTheory.ae μ).EventuallyEq ↑↑((BoundedContinuousFunction.toLp 2 …
    a✝ : α
    hf : Eq (↑↑((BoundedContinuousFunction.toLp 2 μ 𝕜) f) a✝) (f a✝)
    hg : Eq (↑↑((BoundedContinuousFunction.toLp 2 μ 𝕜) g) a✝) (g a✝)
    ⊢ Eq (Inner.inner (f a✝) (g a✝)) (HMul.hMul ((starRingEnd 𝕜) (f a✝)) (g a✝))
  -/
  simp
  /-
    🎉 no goals
  -/


/-- For continuous functions `f`, `g` on a compact, finite-measure topological space `α`, the L^2
inner product is the integral of their pointwise inner product. -/
theorem ContinuousMap.inner_toLp (f g : C(α, 𝕜)) :
    ⟪ContinuousMap.toLp (E := 𝕜) 2 μ 𝕜 f, ContinuousMap.toLp (E := 𝕜) 2 μ 𝕜 g⟫ =
      ∫ x, conj (f x) * g x ∂μ := by
  /-
    α : Type u_1
    𝕜 : Type u_2
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : MeasurableSpace α
    inst✝³ : BorelSpace α
    inst✝² : RCLike 𝕜
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : CompactSpace α
    f g : ContinuousMap α 𝕜
    ⊢ Eq (Inner.inner ((ContinuousMap.toLp 2 μ 𝕜) f) ((ContinuousMap.toLp 2 μ 𝕜) g …
  -/
  apply integral_congr_ae
  -- Porting note: added explicitly passed arguments
  /-
    case h
    α : Type u_1
    𝕜 : Type u_2
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : MeasurableSpace α
    inst✝³ : BorelSpace α
    inst✝² : RCLike 𝕜
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : CompactSpace α
    f g : ContinuousMap α 𝕜
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun a => Inner.inner (↑↑((ContinuousMap.t …
  -/
  have hf_ae := f.coeFn_toLp (p := 2) (𝕜 := 𝕜) μ
  /-
    case h
    α : Type u_1
    𝕜 : Type u_2
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : MeasurableSpace α
    inst✝³ : BorelSpace α
    inst✝² : RCLike 𝕜
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : CompactSpace α
    f g : ContinuousMap α 𝕜
    hf_ae : (MeasureTheory.ae μ).EventuallyEq ↑↑((ContinuousMap.toLp 2 μ 𝕜) f) ⇑f
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun a => Inner.inner (↑↑((ContinuousMap.t …
  -/
  have hg_ae := g.coeFn_toLp (p := 2) (𝕜 := 𝕜) μ
  /-
    case h
    α : Type u_1
    𝕜 : Type u_2
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : MeasurableSpace α
    inst✝³ : BorelSpace α
    inst✝² : RCLike 𝕜
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : CompactSpace α
    f g : ContinuousMap α 𝕜
    hf_ae : (MeasureTheory.ae μ).EventuallyEq ↑↑((ContinuousMap.toLp 2 μ 𝕜) f) ⇑f
    hg_ae : (MeasureTheory.ae μ).EventuallyEq ↑↑((ContinuousMap.toLp 2 μ 𝕜) g) ⇑g
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun a => Inner.inner (↑↑((ContinuousMap.t …
  -/
  filter_upwards [hf_ae, hg_ae] with _ hf hg
  /-
    case h
    α : Type u_1
    𝕜 : Type u_2
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : MeasurableSpace α
    inst✝³ : BorelSpace α
    inst✝² : RCLike 𝕜
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : CompactSpace α
    f g : ContinuousMap α 𝕜
    hf_ae : (MeasureTheory.ae μ).EventuallyEq ↑↑((ContinuousMap.toLp 2 μ 𝕜) f) ⇑f
    hg_ae : (MeasureTheory.ae μ).EventuallyEq ↑↑((ContinuousMap.toLp 2 μ 𝕜) g) ⇑g
    a✝ : α
    hf : Eq (↑↑((ContinuousMap.toLp 2 μ 𝕜) f) a✝) (f a✝)
    hg : Eq (↑↑((ContinuousMap.toLp 2 μ 𝕜) g) a✝) (g a✝)
    ⊢ Eq (Inner.inner (↑↑((ContinuousMap.toLp 2 μ 𝕜) f) a✝) (↑↑((ContinuousMap.toL …
  -/
  rw [hf, hg]
  /-
    case h
    α : Type u_1
    𝕜 : Type u_2
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : MeasurableSpace α
    inst✝³ : BorelSpace α
    inst✝² : RCLike 𝕜
    μ : MeasureTheory.Measure α
    inst✝¹ : MeasureTheory.IsFiniteMeasure μ
    inst✝ : CompactSpace α
    f g : ContinuousMap α 𝕜
    hf_ae : (MeasureTheory.ae μ).EventuallyEq ↑↑((ContinuousMap.toLp 2 μ 𝕜) f) ⇑f
    hg_ae : (MeasureTheory.ae μ).EventuallyEq ↑↑((ContinuousMap.toLp 2 μ 𝕜) g) ⇑g
    a✝ : α
    hf : Eq (↑↑((ContinuousMap.toLp 2 μ 𝕜) f) a✝) (f a✝)
    hg : Eq (↑↑((ContinuousMap.toLp 2 μ 𝕜) g) a✝) (g a✝)
    ⊢ Eq (Inner.inner (f a✝) (g a✝)) (HMul.hMul ((starRingEnd 𝕜) (f a✝)) (g a✝))
  -/
  simp
  /-
    🎉 no goals
  -/


