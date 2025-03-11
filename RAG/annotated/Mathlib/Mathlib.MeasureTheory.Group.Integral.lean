@[to_additive]
theorem Integrable.comp_inv [IsInvInvariant μ] {f : G → F} (hf : Integrable f μ) :
    Integrable (fun t => f t⁻¹) μ :=
  (hf.mono_measure (map_inv_eq_self μ).le).comp_measurable measurable_inv


@[to_additive]
theorem integral_inv_eq_self (f : G → E) (μ : Measure G) [IsInvInvariant μ] :
    ∫ x, f x⁻¹ ∂μ = ∫ x, f x ∂μ := by
  /-
    G : Type u_4
    E : Type u_5
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : Group G
    inst✝¹ : MeasurableInv G
    f : G → E
    μ : MeasureTheory.Measure G
    inst✝ : μ.IsInvInvariant
    ⊢ Eq (MeasureTheory.integral μ fun x => f (Inv.inv x)) (MeasureTheory.integral …
  -/
  have h : MeasurableEmbedding fun x : G => x⁻¹ := (MeasurableEquiv.inv G).measurableEmbedding
  /-
    G : Type u_4
    E : Type u_5
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : Group G
    inst✝¹ : MeasurableInv G
    f : G → E
    μ : MeasureTheory.Measure G
    inst✝ : μ.IsInvInvariant
    h : MeasurableEmbedding fun x => Inv.inv x
    ⊢ Eq (MeasureTheory.integral μ fun x => f (Inv.inv x)) (MeasureTheory.integral …
  -/
  rw [← h.integral_map, map_inv_eq_self]
  /-
    🎉 no goals
  -/


/-- Translating a function by left-multiplication does not change its integral with respect to a
left-invariant measure. -/
@[to_additive
      "Translating a function by left-addition does not change its integral with respect to a
      left-invariant measure."] -- Porting note: was `@[simp]`
theorem integral_mul_left_eq_self [IsMulLeftInvariant μ] (f : G → E) (g : G) :
    (∫ x, f (g * x) ∂μ) = ∫ x, f x ∂μ := by
  /-
    G : Type u_4
    E : Type u_5
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    μ : MeasureTheory.Measure G
    inst✝² : Group G
    inst✝¹ : MeasurableMul G
    inst✝ : μ.IsMulLeftInvariant
    f : G → E
    g : G
    ⊢ Eq (MeasureTheory.integral μ fun x => f (HMul.hMul g x)) (MeasureTheory.inte …
  -/
  have h_mul : MeasurableEmbedding fun x => g * x := (MeasurableEquiv.mulLeft g).measurableEmbedding
  /-
    G : Type u_4
    E : Type u_5
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    μ : MeasureTheory.Measure G
    inst✝² : Group G
    inst✝¹ : MeasurableMul G
    inst✝ : μ.IsMulLeftInvariant
    f : G → E
    g : G
    h_mul : MeasurableEmbedding fun x => HMul.hMul g x
    ⊢ Eq (MeasureTheory.integral μ fun x => f (HMul.hMul g x)) (MeasureTheory.inte …
  -/
  rw [← h_mul.integral_map, map_mul_left_eq_self]
  /-
    🎉 no goals
  -/


/-- Translating a function by right-multiplication does not change its integral with respect to a
right-invariant measure. -/
@[to_additive
      "Translating a function by right-addition does not change its integral with respect to a
      right-invariant measure."] -- Porting note: was `@[simp]`
theorem integral_mul_right_eq_self [IsMulRightInvariant μ] (f : G → E) (g : G) :
    (∫ x, f (x * g) ∂μ) = ∫ x, f x ∂μ := by
  have h_mul : MeasurableEmbedding fun x => x * g :=
    (MeasurableEquiv.mulRight g).measurableEmbedding
  /-
    G : Type u_4
    E : Type u_5
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    μ : MeasureTheory.Measure G
    inst✝² : Group G
    inst✝¹ : MeasurableMul G
    inst✝ : μ.IsMulRightInvariant
    f : G → E
    g : G
    h_mul : MeasurableEmbedding fun x => HMul.hMul x g
    ⊢ Eq (MeasureTheory.integral μ fun x => f (HMul.hMul x g)) (MeasureTheory.inte …
  -/
  rw [← h_mul.integral_map, map_mul_right_eq_self]
  /-
    🎉 no goals
  -/


@[to_additive] -- Porting note: was `@[simp]`
theorem integral_div_right_eq_self [IsMulRightInvariant μ] (f : G → E) (g : G) :
    (∫ x, f (x / g) ∂μ) = ∫ x, f x ∂μ := by
  /-
    G : Type u_4
    E : Type u_5
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    μ : MeasureTheory.Measure G
    inst✝² : Group G
    inst✝¹ : MeasurableMul G
    inst✝ : μ.IsMulRightInvariant
    f : G → E
    g : G
    ⊢ Eq (MeasureTheory.integral μ fun x => f (HDiv.hDiv x g)) (MeasureTheory.inte …
  -/
  simp_rw [div_eq_mul_inv, integral_mul_right_eq_self f g⁻¹]
  /-
    🎉 no goals
  -/


/-- If some left-translate of a function negates it, then the integral of the function with respect
to a left-invariant measure is 0. -/
@[to_additive
      "If some left-translate of a function negates it, then the integral of the function with
      respect to a left-invariant measure is 0."]
theorem integral_eq_zero_of_mul_left_eq_neg [IsMulLeftInvariant μ] (hf' : ∀ x, f (g * x) = -f x) :
    ∫ x, f x ∂μ = 0 := by
  /-
    G : Type u_4
    E : Type u_5
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    μ : MeasureTheory.Measure G
    f : G → E
    g : G
    inst✝² : Group G
    inst✝¹ : MeasurableMul G
    inst✝ : μ.IsMulLeftInvariant
    hf' : ∀ (x : G), Eq (f (HMul.hMul g x)) (Neg.neg (f x))
    ⊢ Eq (MeasureTheory.integral μ fun x => f x) 0
  -/
  simp_rw [← self_eq_neg ℝ E, ← integral_neg, ← hf', integral_mul_left_eq_self]
  /-
    🎉 no goals
  -/


/-- If some right-translate of a function negates it, then the integral of the function with respect
to a right-invariant measure is 0. -/
@[to_additive
      "If some right-translate of a function negates it, then the integral of the function with
      respect to a right-invariant measure is 0."]
theorem integral_eq_zero_of_mul_right_eq_neg [IsMulRightInvariant μ] (hf' : ∀ x, f (x * g) = -f x) :
    ∫ x, f x ∂μ = 0 := by
  /-
    G : Type u_4
    E : Type u_5
    inst✝⁵ : MeasurableSpace G
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    μ : MeasureTheory.Measure G
    f : G → E
    g : G
    inst✝² : Group G
    inst✝¹ : MeasurableMul G
    inst✝ : μ.IsMulRightInvariant
    hf' : ∀ (x : G), Eq (f (HMul.hMul x g)) (Neg.neg (f x))
    ⊢ Eq (MeasureTheory.integral μ fun x => f x) 0
  -/
  simp_rw [← self_eq_neg ℝ E, ← integral_neg, ← hf', integral_mul_right_eq_self]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem Integrable.comp_mul_left {f : G → F} [IsMulLeftInvariant μ] (hf : Integrable f μ) (g : G) :
    Integrable (fun t => f (g * t)) μ :=
  (hf.mono_measure (map_mul_left_eq_self μ g).le).comp_measurable <| measurable_const_mul g


@[to_additive]
theorem Integrable.comp_mul_right {f : G → F} [IsMulRightInvariant μ] (hf : Integrable f μ)
    (g : G) : Integrable (fun t => f (t * g)) μ :=
  (hf.mono_measure (map_mul_right_eq_self μ g).le).comp_measurable <| measurable_mul_const g


@[to_additive]
theorem Integrable.comp_div_right {f : G → F} [IsMulRightInvariant μ] (hf : Integrable f μ)
    (g : G) : Integrable (fun t => f (t / g)) μ := by
  /-
    G : Type u_4
    F : Type u_6
    inst✝⁴ : MeasurableSpace G
    inst✝³ : NormedAddCommGroup F
    μ : MeasureTheory.Measure G
    inst✝² : Group G
    inst✝¹ : MeasurableMul G
    f : G → F
    inst✝ : μ.IsMulRightInvariant
    hf : MeasureTheory.Integrable f μ
    g : G
    ⊢ MeasureTheory.Integrable (fun t => f (HDiv.hDiv t g)) μ
  -/
  simp_rw [div_eq_mul_inv]
  /-
    G : Type u_4
    F : Type u_6
    inst✝⁴ : MeasurableSpace G
    inst✝³ : NormedAddCommGroup F
    μ : MeasureTheory.Measure G
    inst✝² : Group G
    inst✝¹ : MeasurableMul G
    f : G → F
    inst✝ : μ.IsMulRightInvariant
    hf : MeasureTheory.Integrable f μ
    g : G
    ⊢ MeasureTheory.Integrable (fun t => f (HMul.hMul t (Inv.inv g))) μ
  -/
  exact hf.comp_mul_right g⁻¹
  /-
    🎉 no goals
  -/


@[to_additive]
theorem Integrable.comp_div_left {f : G → F} [IsInvInvariant μ] [IsMulLeftInvariant μ]
    (hf : Integrable f μ) (g : G) : Integrable (fun t => f (g / t)) μ :=
  ((measurePreserving_div_left μ g).integrable_comp hf.aestronglyMeasurable).mpr hf


@[to_additive] -- Porting note: was `@[simp]`
theorem integrable_comp_div_left (f : G → F) [IsInvInvariant μ] [IsMulLeftInvariant μ] (g : G) :
    Integrable (fun t => f (g / t)) μ ↔ Integrable f μ := by
  /-
    G : Type u_4
    F : Type u_6
    inst✝⁶ : MeasurableSpace G
    inst✝⁵ : NormedAddCommGroup F
    μ : MeasureTheory.Measure G
    inst✝⁴ : Group G
    inst✝³ : MeasurableMul G
    inst✝² : MeasurableInv G
    f : G → F
    inst✝¹ : μ.IsInvInvariant
    inst✝ : μ.IsMulLeftInvariant
    g : G
    ⊢ Iff (MeasureTheory.Integrable (fun t => f (HDiv.hDiv g t)) μ) (MeasureTheory …
  -/
  refine ⟨fun h => ?_, fun h => h.comp_div_left g⟩
  /-
    G : Type u_4
    F : Type u_6
    inst✝⁶ : MeasurableSpace G
    inst✝⁵ : NormedAddCommGroup F
    μ : MeasureTheory.Measure G
    inst✝⁴ : Group G
    inst✝³ : MeasurableMul G
    inst✝² : MeasurableInv G
    f : G → F
    inst✝¹ : μ.IsInvInvariant
    inst✝ : μ.IsMulLeftInvariant
    g : G
    h : MeasureTheory.Integrable (fun t => f (HDiv.hDiv g t)) μ
    ⊢ MeasureTheory.Integrable f μ
  -/
  convert h.comp_inv.comp_mul_left g⁻¹
  /-
    case h.e'_6.h.h.e'_1
    G : Type u_4
    F : Type u_6
    inst✝⁶ : MeasurableSpace G
    inst✝⁵ : NormedAddCommGroup F
    μ : MeasureTheory.Measure G
    inst✝⁴ : Group G
    inst✝³ : MeasurableMul G
    inst✝² : MeasurableInv G
    f : G → F
    inst✝¹ : μ.IsInvInvariant
    inst✝ : μ.IsMulLeftInvariant
    g : G
    h : MeasureTheory.Integrable (fun t => f (HDiv.hDiv g t)) μ
    x✝ : G
    ⊢ Eq x✝ (HDiv.hDiv g (Inv.inv (HMul.hMul (Inv.inv g) x✝)))
  -/
  simp_rw [div_inv_eq_mul, mul_inv_cancel_left]
  /-
    🎉 no goals
  -/


@[to_additive] -- Porting note: was `@[simp]`
theorem integral_div_left_eq_self (f : G → E) (μ : Measure G) [IsInvInvariant μ]
    [IsMulLeftInvariant μ] (x' : G) : (∫ x, f (x' / x) ∂μ) = ∫ x, f x ∂μ := by
  simp_rw [div_eq_mul_inv, integral_inv_eq_self (fun x => f (x' * x)) μ,
    integral_mul_left_eq_self f x']


@[to_additive] -- Porting note: was `@[simp]`
theorem integral_smul_eq_self {μ : Measure α} [SMulInvariantMeasure G α μ] (f : α → E) {g : G} :
    (∫ x, f (g • x) ∂μ) = ∫ x, f x ∂μ := by
  /-
    α : Type u_3
    G : Type u_4
    E : Type u_5
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : Group G
    inst✝³ : MeasurableSpace α
    inst✝² : MulAction G α
    inst✝¹ : MeasurableSMul G α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SMulInvariantMeasure G α μ
    f : α → E
    g : G
    ⊢ Eq (MeasureTheory.integral μ fun x => f (HSMul.hSMul g x)) (MeasureTheory.in …
  -/
  have h : MeasurableEmbedding fun x : α => g • x := (MeasurableEquiv.smul g).measurableEmbedding
  /-
    α : Type u_3
    G : Type u_4
    E : Type u_5
    inst✝⁷ : MeasurableSpace G
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : Group G
    inst✝³ : MeasurableSpace α
    inst✝² : MulAction G α
    inst✝¹ : MeasurableSMul G α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SMulInvariantMeasure G α μ
    f : α → E
    g : G
    h : MeasurableEmbedding fun x => HSMul.hSMul g x
    ⊢ Eq (MeasureTheory.integral μ fun x => f (HSMul.hSMul g x)) (MeasureTheory.in …
  -/
  rw [← h.integral_map, map_smul]
  /-
    🎉 no goals
  -/


