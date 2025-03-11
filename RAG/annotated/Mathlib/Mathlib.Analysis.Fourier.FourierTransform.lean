local notation "𝕊" => Circle


/-- The Fourier transform integral for `f : V → E`, with respect to a bilinear form `L : V × W → 𝕜`
and an additive character `e`. -/
def fourierIntegral (e : AddChar 𝕜 𝕊) (μ : Measure V) (L : V →ₗ[𝕜] W →ₗ[𝕜] 𝕜) (f : V → E)
    (w : W) : E :=
  ∫ v, e (-L v w) • f v ∂μ


theorem fourierIntegral_const_smul (e : AddChar 𝕜 𝕊) (μ : Measure V)
    (L : V →ₗ[𝕜] W →ₗ[𝕜] 𝕜) (f : V → E) (r : ℂ) :
    fourierIntegral e μ L (r • f) = r • fourierIntegral e μ L f := by
  /-
    𝕜 : Type u_1
    inst✝⁷ : CommRing 𝕜
    V : Type u_2
    inst✝⁶ : AddCommGroup V
    inst✝⁵ : Module 𝕜 V
    inst✝⁴ : MeasurableSpace V
    W : Type u_3
    inst✝³ : AddCommGroup W
    inst✝² : Module 𝕜 W
    E : Type u_4
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    e : AddChar 𝕜 Circle
    μ : MeasureTheory.Measure V
    L : LinearMap (RingHom.id 𝕜) V (LinearMap (RingHom.id 𝕜) W 𝕜)
    f : V → E
    r : Complex
    ⊢ Eq (VectorFourier.fourierIntegral e μ L (HSMul.hSMul r f)) (HSMul.hSMul r (V …
  -/
  ext1 w
  /-
    case h
    𝕜 : Type u_1
    inst✝⁷ : CommRing 𝕜
    V : Type u_2
    inst✝⁶ : AddCommGroup V
    inst✝⁵ : Module 𝕜 V
    inst✝⁴ : MeasurableSpace V
    W : Type u_3
    inst✝³ : AddCommGroup W
    inst✝² : Module 𝕜 W
    E : Type u_4
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    e : AddChar 𝕜 Circle
    μ : MeasureTheory.Measure V
    L : LinearMap (RingHom.id 𝕜) V (LinearMap (RingHom.id 𝕜) W 𝕜)
    f : V → E
    r : Complex
    w : W
    ⊢ Eq (VectorFourier.fourierIntegral e μ L (HSMul.hSMul r f) w) (HSMul.hSMul r  …
  -/
  simp only [Pi.smul_apply, fourierIntegral, smul_comm _ r, integral_smul]
  /-
    🎉 no goals
  -/


/-- The uniform norm of the Fourier integral of `f` is bounded by the `L¹` norm of `f`. -/
theorem norm_fourierIntegral_le_integral_norm (e : AddChar 𝕜 𝕊) (μ : Measure V)
    (L : V →ₗ[𝕜] W →ₗ[𝕜] 𝕜) (f : V → E) (w : W) :
    ‖fourierIntegral e μ L f w‖ ≤ ∫ v : V, ‖f v‖ ∂μ := by
  /-
    𝕜 : Type u_1
    inst✝⁷ : CommRing 𝕜
    V : Type u_2
    inst✝⁶ : AddCommGroup V
    inst✝⁵ : Module 𝕜 V
    inst✝⁴ : MeasurableSpace V
    W : Type u_3
    inst✝³ : AddCommGroup W
    inst✝² : Module 𝕜 W
    E : Type u_4
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    e : AddChar 𝕜 Circle
    μ : MeasureTheory.Measure V
    L : LinearMap (RingHom.id 𝕜) V (LinearMap (RingHom.id 𝕜) W 𝕜)
    f : V → E
    w : W
    ⊢ LE.le (Norm.norm (VectorFourier.fourierIntegral e μ L f w)) (MeasureTheory.i …
  -/
  refine (norm_integral_le_integral_norm _).trans (le_of_eq ?_)
  /-
    𝕜 : Type u_1
    inst✝⁷ : CommRing 𝕜
    V : Type u_2
    inst✝⁶ : AddCommGroup V
    inst✝⁵ : Module 𝕜 V
    inst✝⁴ : MeasurableSpace V
    W : Type u_3
    inst✝³ : AddCommGroup W
    inst✝² : Module 𝕜 W
    E : Type u_4
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    e : AddChar 𝕜 Circle
    μ : MeasureTheory.Measure V
    L : LinearMap (RingHom.id 𝕜) V (LinearMap (RingHom.id 𝕜) W 𝕜)
    f : V → E
    w : W
    ⊢ Eq (MeasureTheory.integral μ fun a => Norm.norm (HSMul.hSMul (e (Neg.neg ((L …
  -/
  simp_rw [Circle.norm_smul]
  /-
    🎉 no goals
  -/


/-- The Fourier integral converts right-translation into scalar multiplication by a phase factor. -/
theorem fourierIntegral_comp_add_right [MeasurableAdd V] (e : AddChar 𝕜 𝕊) (μ : Measure V)
    [μ.IsAddRightInvariant] (L : V →ₗ[𝕜] W →ₗ[𝕜] 𝕜) (f : V → E) (v₀ : V) :
    fourierIntegral e μ L (f ∘ fun v ↦ v + v₀) =
      fun w ↦ e (L v₀ w) • fourierIntegral e μ L f w := by
  /-
    𝕜 : Type u_1
    inst✝⁹ : CommRing 𝕜
    V : Type u_2
    inst✝⁸ : AddCommGroup V
    inst✝⁷ : Module 𝕜 V
    inst✝⁶ : MeasurableSpace V
    W : Type u_3
    inst✝⁵ : AddCommGroup W
    inst✝⁴ : Module 𝕜 W
    E : Type u_4
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Complex E
    inst✝¹ : MeasurableAdd V
    e : AddChar 𝕜 Circle
    μ : MeasureTheory.Measure V
    inst✝ : μ.IsAddRightInvariant
    L : LinearMap (RingHom.id 𝕜) V (LinearMap (RingHom.id 𝕜) W 𝕜)
    f : V → E
    v₀ : V
    ⊢ Eq (VectorFourier.fourierIntegral e μ L (Function.comp f fun v => HAdd.hAdd  …
  -/
  ext1 w
  /-
    case h
    𝕜 : Type u_1
    inst✝⁹ : CommRing 𝕜
    V : Type u_2
    inst✝⁸ : AddCommGroup V
    inst✝⁷ : Module 𝕜 V
    inst✝⁶ : MeasurableSpace V
    W : Type u_3
    inst✝⁵ : AddCommGroup W
    inst✝⁴ : Module 𝕜 W
    E : Type u_4
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Complex E
    inst✝¹ : MeasurableAdd V
    e : AddChar 𝕜 Circle
    μ : MeasureTheory.Measure V
    inst✝ : μ.IsAddRightInvariant
    L : LinearMap (RingHom.id 𝕜) V (LinearMap (RingHom.id 𝕜) W 𝕜)
    f : V → E
    v₀ : V
    w : W
    ⊢ Eq (VectorFourier.fourierIntegral e μ L (Function.comp f fun v => HAdd.hAdd  …
  -/
  dsimp only [fourierIntegral, Function.comp_apply, Circle.smul_def]
  /-
    case h
    𝕜 : Type u_1
    inst✝⁹ : CommRing 𝕜
    V : Type u_2
    inst✝⁸ : AddCommGroup V
    inst✝⁷ : Module 𝕜 V
    inst✝⁶ : MeasurableSpace V
    W : Type u_3
    inst✝⁵ : AddCommGroup W
    inst✝⁴ : Module 𝕜 W
    E : Type u_4
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Complex E
    inst✝¹ : MeasurableAdd V
    e : AddChar 𝕜 Circle
    μ : MeasureTheory.Measure V
    inst✝ : μ.IsAddRightInvariant
    L : LinearMap (RingHom.id 𝕜) V (LinearMap (RingHom.id 𝕜) W 𝕜)
    f : V → E
    v₀ : V
    w : W
    ⊢ Eq (MeasureTheory.integral μ fun v => HSMul.hSMul (↑(e (Neg.neg ((L v) w)))) …
  -/
  conv in L _ => rw [← add_sub_cancel_right v v₀]
  /-
    case h
    𝕜 : Type u_1
    inst✝⁹ : CommRing 𝕜
    V : Type u_2
    inst✝⁸ : AddCommGroup V
    inst✝⁷ : Module 𝕜 V
    inst✝⁶ : MeasurableSpace V
    W : Type u_3
    inst✝⁵ : AddCommGroup W
    inst✝⁴ : Module 𝕜 W
    E : Type u_4
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Complex E
    inst✝¹ : MeasurableAdd V
    e : AddChar 𝕜 Circle
    μ : MeasureTheory.Measure V
    inst✝ : μ.IsAddRightInvariant
    L : LinearMap (RingHom.id 𝕜) V (LinearMap (RingHom.id 𝕜) W 𝕜)
    f : V → E
    v₀ : V
    w : W
    ⊢ Eq (MeasureTheory.integral μ fun v => HSMul.hSMul (↑(e (Neg.neg ((L (HSub.hS …
  -/
  rw [integral_add_right_eq_self fun v : V ↦ (e (-L (v - v₀) w) : ℂ) • f v, ← integral_smul]
  /-
    case h
    𝕜 : Type u_1
    inst✝⁹ : CommRing 𝕜
    V : Type u_2
    inst✝⁸ : AddCommGroup V
    inst✝⁷ : Module 𝕜 V
    inst✝⁶ : MeasurableSpace V
    W : Type u_3
    inst✝⁵ : AddCommGroup W
    inst✝⁴ : Module 𝕜 W
    E : Type u_4
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Complex E
    inst✝¹ : MeasurableAdd V
    e : AddChar 𝕜 Circle
    μ : MeasureTheory.Measure V
    inst✝ : μ.IsAddRightInvariant
    L : LinearMap (RingHom.id 𝕜) V (LinearMap (RingHom.id 𝕜) W 𝕜)
    f : V → E
    v₀ : V
    w : W
    ⊢ Eq (MeasureTheory.integral μ fun x => HSMul.hSMul (↑(e (Neg.neg ((L (HSub.hS …
  -/
  congr 1 with v
  rw [← smul_assoc, smul_eq_mul, ← Circle.coe_mul, ← e.map_add_eq_mul, ← LinearMap.neg_apply,
    ← sub_eq_add_neg, ← LinearMap.sub_apply, LinearMap.map_sub, neg_sub]


/-- For any `w`, the Fourier integral is convergent iff `f` is integrable. -/
theorem fourierIntegral_convergent_iff (he : Continuous e)
    (hL : Continuous fun p : V × W ↦ L p.1 p.2) {f : V → E} (w : W) :
    Integrable (fun v : V ↦ e (-L v w) • f v) μ ↔ Integrable f μ := by
  -- first prove one-way implication
  have aux {g : V → E} (hg : Integrable g μ) (x : W) :
      Integrable (fun v : V ↦ e (-L v x) • g v) μ := by
    have c : Continuous fun v ↦ e (-L v x) :=
      he.comp (hL.comp (continuous_prod_mk.mpr ⟨continuous_id, continuous_const⟩)).neg
    simp_rw [← integrable_norm_iff (c.aestronglyMeasurable.smul hg.1), Circle.norm_smul]
    exact hg.norm
  -- then use it for both directions
  /-
    𝕜 : Type u_1
    inst✝¹² : CommRing 𝕜
    V : Type u_2
    inst✝¹¹ : AddCommGroup V
    inst✝¹⁰ : Module 𝕜 V
    inst✝⁹ : MeasurableSpace V
    W : Type u_3
    inst✝⁸ : AddCommGroup W
    inst✝⁷ : Module 𝕜 W
    E : Type u_4
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    inst✝⁴ : TopologicalSpace 𝕜
    inst✝³ : TopologicalRing 𝕜
    inst✝² : TopologicalSpace V
    inst✝¹ : BorelSpace V
    inst✝ : TopologicalSpace W
    e : AddChar 𝕜 Circle
    μ : MeasureTheory.Measure V
    L : LinearMap (RingHom.id 𝕜) V (LinearMap (RingHom.id 𝕜) W 𝕜)
    he : Continuous ⇑e
    hL : Continuous fun p => (L p.1) p.2
    f : V → E
    w : W
    aux : ∀ {g : V → E}, MeasureTheory.Integrable g μ → ∀ (x : W), MeasureTheory.I …
    ⊢ Iff (MeasureTheory.Integrable (fun v => HSMul.hSMul (e (Neg.neg ((L v) w)))  …
  -/
  refine ⟨fun hf ↦ ?_, fun hf ↦ aux hf w⟩
  /-
    𝕜 : Type u_1
    inst✝¹² : CommRing 𝕜
    V : Type u_2
    inst✝¹¹ : AddCommGroup V
    inst✝¹⁰ : Module 𝕜 V
    inst✝⁹ : MeasurableSpace V
    W : Type u_3
    inst✝⁸ : AddCommGroup W
    inst✝⁷ : Module 𝕜 W
    E : Type u_4
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    inst✝⁴ : TopologicalSpace 𝕜
    inst✝³ : TopologicalRing 𝕜
    inst✝² : TopologicalSpace V
    inst✝¹ : BorelSpace V
    inst✝ : TopologicalSpace W
    e : AddChar 𝕜 Circle
    μ : MeasureTheory.Measure V
    L : LinearMap (RingHom.id 𝕜) V (LinearMap (RingHom.id 𝕜) W 𝕜)
    he : Continuous ⇑e
    hL : Continuous fun p => (L p.1) p.2
    f : V → E
    w : W
    aux : ∀ {g : V → E}, MeasureTheory.Integrable g μ → ∀ (x : W), MeasureTheory.I …
    hf : MeasureTheory.Integrable (fun v => HSMul.hSMul (e (Neg.neg ((L v) w))) (f …
    ⊢ MeasureTheory.Integrable f μ
  -/
  have := aux hf (-w)
  simp_rw [← mul_smul (e _) (e _) (f _), ← e.map_add_eq_mul, LinearMap.map_neg, neg_add_cancel,
    e.map_zero_eq_one, one_smul] at this -- the `(e _)` speeds up elaboration considerably
  /-
    𝕜 : Type u_1
    inst✝¹² : CommRing 𝕜
    V : Type u_2
    inst✝¹¹ : AddCommGroup V
    inst✝¹⁰ : Module 𝕜 V
    inst✝⁹ : MeasurableSpace V
    W : Type u_3
    inst✝⁸ : AddCommGroup W
    inst✝⁷ : Module 𝕜 W
    E : Type u_4
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    inst✝⁴ : TopologicalSpace 𝕜
    inst✝³ : TopologicalRing 𝕜
    inst✝² : TopologicalSpace V
    inst✝¹ : BorelSpace V
    inst✝ : TopologicalSpace W
    e : AddChar 𝕜 Circle
    μ : MeasureTheory.Measure V
    L : LinearMap (RingHom.id 𝕜) V (LinearMap (RingHom.id 𝕜) W 𝕜)
    he : Continuous ⇑e
    hL : Continuous fun p => (L p.1) p.2
    f : V → E
    w : W
    aux : ∀ {g : V → E}, MeasureTheory.Integrable g μ → ∀ (x : W), MeasureTheory.I …
    hf : MeasureTheory.Integrable (fun v => HSMul.hSMul (e (Neg.neg ((L v) w))) (f …
    this : MeasureTheory.Integrable (fun v => f v) μ
    ⊢ MeasureTheory.Integrable f μ
  -/
  exact this
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-03-29")]
alias fourier_integral_convergent_iff := VectorFourier.fourierIntegral_convergent_iff


theorem fourierIntegral_add (he : Continuous e) (hL : Continuous fun p : V × W ↦ L p.1 p.2)
    {f g : V → E} (hf : Integrable f μ) (hg : Integrable g μ) :
    fourierIntegral e μ L (f + g) = fourierIntegral e μ L f + fourierIntegral e μ L g := by
  /-
    𝕜 : Type u_1
    inst✝¹² : CommRing 𝕜
    V : Type u_2
    inst✝¹¹ : AddCommGroup V
    inst✝¹⁰ : Module 𝕜 V
    inst✝⁹ : MeasurableSpace V
    W : Type u_3
    inst✝⁸ : AddCommGroup W
    inst✝⁷ : Module 𝕜 W
    E : Type u_4
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    inst✝⁴ : TopologicalSpace 𝕜
    inst✝³ : TopologicalRing 𝕜
    inst✝² : TopologicalSpace V
    inst✝¹ : BorelSpace V
    inst✝ : TopologicalSpace W
    e : AddChar 𝕜 Circle
    μ : MeasureTheory.Measure V
    L : LinearMap (RingHom.id 𝕜) V (LinearMap (RingHom.id 𝕜) W 𝕜)
    he : Continuous ⇑e
    hL : Continuous fun p => (L p.1) p.2
    f g : V → E
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g μ
    ⊢ Eq (VectorFourier.fourierIntegral e μ L (HAdd.hAdd f g)) (HAdd.hAdd (VectorF …
  -/
  ext1 w
  /-
    case h
    𝕜 : Type u_1
    inst✝¹² : CommRing 𝕜
    V : Type u_2
    inst✝¹¹ : AddCommGroup V
    inst✝¹⁰ : Module 𝕜 V
    inst✝⁹ : MeasurableSpace V
    W : Type u_3
    inst✝⁸ : AddCommGroup W
    inst✝⁷ : Module 𝕜 W
    E : Type u_4
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    inst✝⁴ : TopologicalSpace 𝕜
    inst✝³ : TopologicalRing 𝕜
    inst✝² : TopologicalSpace V
    inst✝¹ : BorelSpace V
    inst✝ : TopologicalSpace W
    e : AddChar 𝕜 Circle
    μ : MeasureTheory.Measure V
    L : LinearMap (RingHom.id 𝕜) V (LinearMap (RingHom.id 𝕜) W 𝕜)
    he : Continuous ⇑e
    hL : Continuous fun p => (L p.1) p.2
    f g : V → E
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g μ
    w : W
    ⊢ Eq (VectorFourier.fourierIntegral e μ L (HAdd.hAdd f g) w) (HAdd.hAdd (Vecto …
  -/
  dsimp only [Pi.add_apply, fourierIntegral]
  /-
    case h
    𝕜 : Type u_1
    inst✝¹² : CommRing 𝕜
    V : Type u_2
    inst✝¹¹ : AddCommGroup V
    inst✝¹⁰ : Module 𝕜 V
    inst✝⁹ : MeasurableSpace V
    W : Type u_3
    inst✝⁸ : AddCommGroup W
    inst✝⁷ : Module 𝕜 W
    E : Type u_4
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    inst✝⁴ : TopologicalSpace 𝕜
    inst✝³ : TopologicalRing 𝕜
    inst✝² : TopologicalSpace V
    inst✝¹ : BorelSpace V
    inst✝ : TopologicalSpace W
    e : AddChar 𝕜 Circle
    μ : MeasureTheory.Measure V
    L : LinearMap (RingHom.id 𝕜) V (LinearMap (RingHom.id 𝕜) W 𝕜)
    he : Continuous ⇑e
    hL : Continuous fun p => (L p.1) p.2
    f g : V → E
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g μ
    w : W
    ⊢ Eq (MeasureTheory.integral μ fun v => HSMul.hSMul (e (Neg.neg ((L v) w))) (H …
  -/
  simp_rw [smul_add]
  /-
    case h
    𝕜 : Type u_1
    inst✝¹² : CommRing 𝕜
    V : Type u_2
    inst✝¹¹ : AddCommGroup V
    inst✝¹⁰ : Module 𝕜 V
    inst✝⁹ : MeasurableSpace V
    W : Type u_3
    inst✝⁸ : AddCommGroup W
    inst✝⁷ : Module 𝕜 W
    E : Type u_4
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    inst✝⁴ : TopologicalSpace 𝕜
    inst✝³ : TopologicalRing 𝕜
    inst✝² : TopologicalSpace V
    inst✝¹ : BorelSpace V
    inst✝ : TopologicalSpace W
    e : AddChar 𝕜 Circle
    μ : MeasureTheory.Measure V
    L : LinearMap (RingHom.id 𝕜) V (LinearMap (RingHom.id 𝕜) W 𝕜)
    he : Continuous ⇑e
    hL : Continuous fun p => (L p.1) p.2
    f g : V → E
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g μ
    w : W
    ⊢ Eq (MeasureTheory.integral μ fun v => HAdd.hAdd (HSMul.hSMul (e (Neg.neg ((L …
  -/
  rw [integral_add]
    /-
      case h.hf
      𝕜 : Type u_1
      inst✝¹² : CommRing 𝕜
      V : Type u_2
      inst✝¹¹ : AddCommGroup V
      inst✝¹⁰ : Module 𝕜 V
      inst✝⁹ : MeasurableSpace V
      W : Type u_3
      inst✝⁸ : AddCommGroup W
      inst✝⁷ : Module 𝕜 W
      E : Type u_4
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace Complex E
      inst✝⁴ : TopologicalSpace 𝕜
      inst✝³ : TopologicalRing 𝕜
      inst✝² : TopologicalSpace V
      inst✝¹ : BorelSpace V
      inst✝ : TopologicalSpace W
      e : AddChar 𝕜 Circle
      μ : MeasureTheory.Measure V
      L : LinearMap (RingHom.id 𝕜) V (LinearMap (RingHom.id 𝕜) W 𝕜)
      he : Continuous ⇑e
      hL : Continuous fun p => (L p.1) p.2
      f g : V → E
      hf : MeasureTheory.Integrable f μ
      hg : MeasureTheory.Integrable g μ
      w : W
      ⊢ MeasureTheory.Integrable (fun v => HSMul.hSMul (e (Neg.neg ((L v) w))) (f v) …
    -/
  · exact (fourierIntegral_convergent_iff he hL w).2 hf
    /-
      🎉 no goals
    -/
    /-
      case h.hg
      𝕜 : Type u_1
      inst✝¹² : CommRing 𝕜
      V : Type u_2
      inst✝¹¹ : AddCommGroup V
      inst✝¹⁰ : Module 𝕜 V
      inst✝⁹ : MeasurableSpace V
      W : Type u_3
      inst✝⁸ : AddCommGroup W
      inst✝⁷ : Module 𝕜 W
      E : Type u_4
      inst✝⁶ : NormedAddCommGroup E
      inst✝⁵ : NormedSpace Complex E
      inst✝⁴ : TopologicalSpace 𝕜
      inst✝³ : TopologicalRing 𝕜
      inst✝² : TopologicalSpace V
      inst✝¹ : BorelSpace V
      inst✝ : TopologicalSpace W
      e : AddChar 𝕜 Circle
      μ : MeasureTheory.Measure V
      L : LinearMap (RingHom.id 𝕜) V (LinearMap (RingHom.id 𝕜) W 𝕜)
      he : Continuous ⇑e
      hL : Continuous fun p => (L p.1) p.2
      f g : V → E
      hf : MeasureTheory.Integrable f μ
      hg : MeasureTheory.Integrable g μ
      w : W
      ⊢ MeasureTheory.Integrable (fun v => HSMul.hSMul (e (Neg.neg ((L v) w))) (g v) …
    -/
  · exact (fourierIntegral_convergent_iff he hL w).2 hg
    /-
      🎉 no goals
    -/


/-- The Fourier integral of an `L^1` function is a continuous function. -/
theorem fourierIntegral_continuous [FirstCountableTopology W] (he : Continuous e)
    (hL : Continuous fun p : V × W ↦ L p.1 p.2) {f : V → E} (hf : Integrable f μ) :
    Continuous (fourierIntegral e μ L f) := by
  /-
    𝕜 : Type u_1
    inst✝¹³ : CommRing 𝕜
    V : Type u_2
    inst✝¹² : AddCommGroup V
    inst✝¹¹ : Module 𝕜 V
    inst✝¹⁰ : MeasurableSpace V
    W : Type u_3
    inst✝⁹ : AddCommGroup W
    inst✝⁸ : Module 𝕜 W
    E : Type u_4
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace Complex E
    inst✝⁵ : TopologicalSpace 𝕜
    inst✝⁴ : TopologicalRing 𝕜
    inst✝³ : TopologicalSpace V
    inst✝² : BorelSpace V
    inst✝¹ : TopologicalSpace W
    e : AddChar 𝕜 Circle
    μ : MeasureTheory.Measure V
    L : LinearMap (RingHom.id 𝕜) V (LinearMap (RingHom.id 𝕜) W 𝕜)
    inst✝ : FirstCountableTopology W
    he : Continuous ⇑e
    hL : Continuous fun p => (L p.1) p.2
    f : V → E
    hf : MeasureTheory.Integrable f μ
    ⊢ Continuous (VectorFourier.fourierIntegral e μ L f)
  -/
  apply continuous_of_dominated
    /-
      case hF_meas
      𝕜 : Type u_1
      inst✝¹³ : CommRing 𝕜
      V : Type u_2
      inst✝¹² : AddCommGroup V
      inst✝¹¹ : Module 𝕜 V
      inst✝¹⁰ : MeasurableSpace V
      W : Type u_3
      inst✝⁹ : AddCommGroup W
      inst✝⁸ : Module 𝕜 W
      E : Type u_4
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Complex E
      inst✝⁵ : TopologicalSpace 𝕜
      inst✝⁴ : TopologicalRing 𝕜
      inst✝³ : TopologicalSpace V
      inst✝² : BorelSpace V
      inst✝¹ : TopologicalSpace W
      e : AddChar 𝕜 Circle
      μ : MeasureTheory.Measure V
      L : LinearMap (RingHom.id 𝕜) V (LinearMap (RingHom.id 𝕜) W 𝕜)
      inst✝ : FirstCountableTopology W
      he : Continuous ⇑e
      hL : Continuous fun p => (L p.1) p.2
      f : V → E
      hf : MeasureTheory.Integrable f μ
      ⊢ ∀ (x : W), MeasureTheory.AEStronglyMeasurable (fun a => HSMul.hSMul (e (Neg. …
    -/
  · exact fun w ↦ ((fourierIntegral_convergent_iff he hL w).2 hf).1
    /-
      🎉 no goals
    -/
    /-
      case h_bound
      𝕜 : Type u_1
      inst✝¹³ : CommRing 𝕜
      V : Type u_2
      inst✝¹² : AddCommGroup V
      inst✝¹¹ : Module 𝕜 V
      inst✝¹⁰ : MeasurableSpace V
      W : Type u_3
      inst✝⁹ : AddCommGroup W
      inst✝⁸ : Module 𝕜 W
      E : Type u_4
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Complex E
      inst✝⁵ : TopologicalSpace 𝕜
      inst✝⁴ : TopologicalRing 𝕜
      inst✝³ : TopologicalSpace V
      inst✝² : BorelSpace V
      inst✝¹ : TopologicalSpace W
      e : AddChar 𝕜 Circle
      μ : MeasureTheory.Measure V
      L : LinearMap (RingHom.id 𝕜) V (LinearMap (RingHom.id 𝕜) W 𝕜)
      inst✝ : FirstCountableTopology W
      he : Continuous ⇑e
      hL : Continuous fun p => (L p.1) p.2
      f : V → E
      hf : MeasureTheory.Integrable f μ
      ⊢ ∀ (x : W), Filter.Eventually (fun a => LE.le (Norm.norm (HSMul.hSMul (e (Neg …
    -/
  · exact fun w ↦ ae_of_all _ fun v ↦ le_of_eq (Circle.norm_smul _ _)
    /-
      🎉 no goals
    -/
    /-
      case bound_integrable
      𝕜 : Type u_1
      inst✝¹³ : CommRing 𝕜
      V : Type u_2
      inst✝¹² : AddCommGroup V
      inst✝¹¹ : Module 𝕜 V
      inst✝¹⁰ : MeasurableSpace V
      W : Type u_3
      inst✝⁹ : AddCommGroup W
      inst✝⁸ : Module 𝕜 W
      E : Type u_4
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Complex E
      inst✝⁵ : TopologicalSpace 𝕜
      inst✝⁴ : TopologicalRing 𝕜
      inst✝³ : TopologicalSpace V
      inst✝² : BorelSpace V
      inst✝¹ : TopologicalSpace W
      e : AddChar 𝕜 Circle
      μ : MeasureTheory.Measure V
      L : LinearMap (RingHom.id 𝕜) V (LinearMap (RingHom.id 𝕜) W 𝕜)
      inst✝ : FirstCountableTopology W
      he : Continuous ⇑e
      hL : Continuous fun p => (L p.1) p.2
      f : V → E
      hf : MeasureTheory.Integrable f μ
      ⊢ MeasureTheory.Integrable (fun v => Norm.norm (f v)) μ
    -/
  · exact hf.norm
    /-
      🎉 no goals
    -/
    /-
      case h_cont
      𝕜 : Type u_1
      inst✝¹³ : CommRing 𝕜
      V : Type u_2
      inst✝¹² : AddCommGroup V
      inst✝¹¹ : Module 𝕜 V
      inst✝¹⁰ : MeasurableSpace V
      W : Type u_3
      inst✝⁹ : AddCommGroup W
      inst✝⁸ : Module 𝕜 W
      E : Type u_4
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Complex E
      inst✝⁵ : TopologicalSpace 𝕜
      inst✝⁴ : TopologicalRing 𝕜
      inst✝³ : TopologicalSpace V
      inst✝² : BorelSpace V
      inst✝¹ : TopologicalSpace W
      e : AddChar 𝕜 Circle
      μ : MeasureTheory.Measure V
      L : LinearMap (RingHom.id 𝕜) V (LinearMap (RingHom.id 𝕜) W 𝕜)
      inst✝ : FirstCountableTopology W
      he : Continuous ⇑e
      hL : Continuous fun p => (L p.1) p.2
      f : V → E
      hf : MeasureTheory.Integrable f μ
      ⊢ Filter.Eventually (fun a => Continuous fun x => HSMul.hSMul (e (Neg.neg ((L  …
    -/
  · refine ae_of_all _ fun v ↦ (he.comp ?_).smul continuous_const
    /-
      case h_cont
      𝕜 : Type u_1
      inst✝¹³ : CommRing 𝕜
      V : Type u_2
      inst✝¹² : AddCommGroup V
      inst✝¹¹ : Module 𝕜 V
      inst✝¹⁰ : MeasurableSpace V
      W : Type u_3
      inst✝⁹ : AddCommGroup W
      inst✝⁸ : Module 𝕜 W
      E : Type u_4
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace Complex E
      inst✝⁵ : TopologicalSpace 𝕜
      inst✝⁴ : TopologicalRing 𝕜
      inst✝³ : TopologicalSpace V
      inst✝² : BorelSpace V
      inst✝¹ : TopologicalSpace W
      e : AddChar 𝕜 Circle
      μ : MeasureTheory.Measure V
      L : LinearMap (RingHom.id 𝕜) V (LinearMap (RingHom.id 𝕜) W 𝕜)
      inst✝ : FirstCountableTopology W
      he : Continuous ⇑e
      hL : Continuous fun p => (L p.1) p.2
      f : V → E
      hf : MeasureTheory.Integrable f μ
      v : V
      ⊢ Continuous fun x => Neg.neg ((L v) x)
    -/
    exact (hL.comp (continuous_prod_mk.mpr ⟨continuous_const, continuous_id⟩)).neg
    /-
      🎉 no goals
    -/


/-- The Fourier transform satisfies `∫ 𝓕 f * g = ∫ f * 𝓕 g`, i.e., it is self-adjoint.
Version where the multiplication is replaced by a general bilinear form `M`. -/
theorem integral_bilin_fourierIntegral_eq_flip
    {f : V → E} {g : W → F} (M : E →L[ℂ] F →L[ℂ] G) (he : Continuous e)
    (hL : Continuous fun p : V × W ↦ L p.1 p.2) (hf : Integrable f μ) (hg : Integrable g ν) :
    ∫ ξ, M (fourierIntegral e μ L f ξ) (g ξ) ∂ν =
      ∫ x, M (f x) (fourierIntegral e ν L.flip g x) ∂μ := by
  /-
    𝕜 : Type u_1
    inst✝²³ : CommRing 𝕜
    V : Type u_2
    inst✝²² : AddCommGroup V
    inst✝²¹ : Module 𝕜 V
    inst✝²⁰ : MeasurableSpace V
    W : Type u_3
    inst✝¹⁹ : AddCommGroup W
    inst✝¹⁸ : Module 𝕜 W
    E : Type u_4
    F : Type u_5
    G : Type u_6
    inst✝¹⁷ : NormedAddCommGroup E
    inst✝¹⁶ : NormedSpace Complex E
    inst✝¹⁵ : NormedAddCommGroup F
    inst✝¹⁴ : NormedSpace Complex F
    inst✝¹³ : NormedAddCommGroup G
    inst✝¹² : NormedSpace Complex G
    inst✝¹¹ : TopologicalSpace 𝕜
    inst✝¹⁰ : TopologicalRing 𝕜
    inst✝⁹ : TopologicalSpace V
    inst✝⁸ : BorelSpace V
    inst✝⁷ : TopologicalSpace W
    inst✝⁶ : MeasurableSpace W
    inst✝⁵ : BorelSpace W
    e : AddChar 𝕜 Circle
    μ : MeasureTheory.Measure V
    L : LinearMap (RingHom.id 𝕜) V (LinearMap (RingHom.id 𝕜) W 𝕜)
    ν : MeasureTheory.Measure W
    inst✝⁴ : MeasureTheory.SigmaFinite μ
    inst✝³ : MeasureTheory.SigmaFinite ν
    inst✝² : SecondCountableTopology V
    inst✝¹ : CompleteSpace E
    inst✝ : CompleteSpace F
    f : V → E
    g : W → F
    M : ContinuousLinearMap (RingHom.id Complex) E (ContinuousLinearMap (RingHom.i …
    he : Continuous ⇑e
    hL : Continuous fun p => (L p.1) p.2
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g ν
    ⊢ Eq (MeasureTheory.integral ν fun ξ => (M (VectorFourier.fourierIntegral e μ  …
  -/
  by_cases hG : CompleteSpace G; swap; · simp [integral, hG]
                                         /-
                                           🎉 no goals
                                         -/
  calc
  _ = ∫ ξ, M.flip (g ξ) (∫ x, e (-L x ξ) • f x ∂μ) ∂ν := rfl
  _ = ∫ ξ, (∫ x, M.flip (g ξ) (e (-L x ξ) • f x) ∂μ) ∂ν := by
    congr with ξ
    apply (ContinuousLinearMap.integral_comp_comm _ _).symm
    exact (fourierIntegral_convergent_iff he hL _).2 hf
  _ = ∫ x, (∫ ξ, M.flip (g ξ) (e (-L x ξ) • f x) ∂ν) ∂μ := by
    rw [integral_integral_swap]
    have : Integrable (fun (p : W × V) ↦ ‖M‖ * (‖g p.1‖ * ‖f p.2‖)) (ν.prod μ) :=
      (hg.norm.prod_mul hf.norm).const_mul _
    apply this.mono
    · -- This proof can be golfed but becomes very slow; breaking it up into steps
      -- speeds up compilation.
      change AEStronglyMeasurable (fun p : W × V ↦ (M (e (-(L p.2) p.1) • f p.2) (g p.1))) _
      have A : AEStronglyMeasurable (fun (p : W × V) ↦ e (-L p.2 p.1) • f p.2) (ν.prod μ) := by
        refine (Continuous.aestronglyMeasurable ?_).smul hf.1.snd
        exact he.comp (hL.comp continuous_swap).neg
      have A' : AEStronglyMeasurable (fun p ↦ (g p.1, e (-(L p.2) p.1) • f p.2) : W × V → F × E)
        (Measure.prod ν μ) := hg.1.fst.prod_mk A
      have B : Continuous (fun q ↦ M q.2 q.1 : F × E → G) := M.flip.continuous₂
      apply B.comp_aestronglyMeasurable A' -- `exact` works, but `apply` is 10x faster!
    · filter_upwards with ⟨ξ, x⟩
      rw [Function.uncurry_apply_pair, Submonoid.smul_def, (M.flip (g ξ)).map_smul,
        ← Submonoid.smul_def, Circle.norm_smul, ContinuousLinearMap.flip_apply,
        norm_mul, norm_norm M, norm_mul, norm_norm, norm_norm, mul_comm (‖g _‖), ← mul_assoc]
      exact M.le_opNorm₂ (f x) (g ξ)
  _ = ∫ x, (∫ ξ, M (f x) (e (-L.flip ξ x) • g ξ) ∂ν) ∂μ := by
      simp only [ContinuousLinearMap.flip_apply, ContinuousLinearMap.map_smul_of_tower,
      ContinuousLinearMap.coe_smul', Pi.smul_apply, LinearMap.flip_apply]
  _ = ∫ x, M (f x) (∫ ξ, e (-L.flip ξ x) • g ξ ∂ν) ∂μ := by
    congr with x
    apply ContinuousLinearMap.integral_comp_comm
    apply (fourierIntegral_convergent_iff he _ _).2 hg
    exact hL.comp continuous_swap


/-- The Fourier transform satisfies `∫ 𝓕 f * g = ∫ f * 𝓕 g`, i.e., it is self-adjoint. -/
theorem integral_fourierIntegral_smul_eq_flip
    {f : V → ℂ} {g : W → F} (he : Continuous e)
    (hL : Continuous fun p : V × W ↦ L p.1 p.2) (hf : Integrable f μ) (hg : Integrable g ν) :
    ∫ ξ, (fourierIntegral e μ L f ξ) • (g ξ) ∂ν =
      ∫ x, (f x) • (fourierIntegral e ν L.flip g x) ∂μ :=
  integral_bilin_fourierIntegral_eq_flip (ContinuousLinearMap.lsmul ℂ ℂ) he hL hf hg


theorem fourierIntegral_continuousLinearMap_apply
    {f : V → (F →L[ℝ] E)} {a : F} {w : W} (he : Continuous e) (hf : Integrable f μ) :
    fourierIntegral e μ L.toLinearMap₂ f w a =
      fourierIntegral e μ L.toLinearMap₂ (fun x ↦ f x a) w := by
  /-
    𝕜 : Type u_1
    E : Type u_3
    F : Type u_4
    V : Type u_5
    W : Type u_6
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NormedAddCommGroup V
    inst✝⁸ : NormedSpace 𝕜 V
    inst✝⁷ : MeasurableSpace V
    inst✝⁶ : BorelSpace V
    inst✝⁵ : NormedAddCommGroup W
    inst✝⁴ : NormedSpace 𝕜 W
    e : AddChar 𝕜 Circle
    μ : MeasureTheory.Measure V
    L : ContinuousLinearMap (RingHom.id 𝕜) V (ContinuousLinearMap (RingHom.id 𝕜) W …
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Real F
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Complex E
    f : V → ContinuousLinearMap (RingHom.id Real) F E
    a : F
    w : W
    he : Continuous ⇑e
    hf : MeasureTheory.Integrable f μ
    ⊢ Eq ((VectorFourier.fourierIntegral e μ L.toLinearMap₂ f w) a) (VectorFourier …
  -/
  rw [fourierIntegral, ContinuousLinearMap.integral_apply]
    /-
      𝕜 : Type u_1
      E : Type u_3
      F : Type u_4
      V : Type u_5
      W : Type u_6
      inst✝¹⁰ : NontriviallyNormedField 𝕜
      inst✝⁹ : NormedAddCommGroup V
      inst✝⁸ : NormedSpace 𝕜 V
      inst✝⁷ : MeasurableSpace V
      inst✝⁶ : BorelSpace V
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace 𝕜 W
      e : AddChar 𝕜 Circle
      μ : MeasureTheory.Measure V
      L : ContinuousLinearMap (RingHom.id 𝕜) V (ContinuousLinearMap (RingHom.id 𝕜) W …
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace Real F
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : V → ContinuousLinearMap (RingHom.id Real) F E
      a : F
      w : W
      he : Continuous ⇑e
      hf : MeasureTheory.Integrable f μ
      ⊢ Eq (MeasureTheory.integral μ fun x => (HSMul.hSMul (e (Neg.neg ((L.toLinearM …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case φ_int
      𝕜 : Type u_1
      E : Type u_3
      F : Type u_4
      V : Type u_5
      W : Type u_6
      inst✝¹⁰ : NontriviallyNormedField 𝕜
      inst✝⁹ : NormedAddCommGroup V
      inst✝⁸ : NormedSpace 𝕜 V
      inst✝⁷ : MeasurableSpace V
      inst✝⁶ : BorelSpace V
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace 𝕜 W
      e : AddChar 𝕜 Circle
      μ : MeasureTheory.Measure V
      L : ContinuousLinearMap (RingHom.id 𝕜) V (ContinuousLinearMap (RingHom.id 𝕜) W …
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace Real F
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : V → ContinuousLinearMap (RingHom.id Real) F E
      a : F
      w : W
      he : Continuous ⇑e
      hf : MeasureTheory.Integrable f μ
      ⊢ MeasureTheory.Integrable (fun v => HSMul.hSMul (e (Neg.neg ((L.toLinearMap₂  …
    -/
  · apply (fourierIntegral_convergent_iff he _ _).2 hf
    /-
      𝕜 : Type u_1
      E : Type u_3
      F : Type u_4
      V : Type u_5
      W : Type u_6
      inst✝¹⁰ : NontriviallyNormedField 𝕜
      inst✝⁹ : NormedAddCommGroup V
      inst✝⁸ : NormedSpace 𝕜 V
      inst✝⁷ : MeasurableSpace V
      inst✝⁶ : BorelSpace V
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace 𝕜 W
      e : AddChar 𝕜 Circle
      μ : MeasureTheory.Measure V
      L : ContinuousLinearMap (RingHom.id 𝕜) V (ContinuousLinearMap (RingHom.id 𝕜) W …
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace Real F
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Complex E
      f : V → ContinuousLinearMap (RingHom.id Real) F E
      a : F
      w : W
      he : Continuous ⇑e
      hf : MeasureTheory.Integrable f μ
      ⊢ Continuous fun p => (L.toLinearMap₂ p.1) p.2
    -/
    exact L.continuous₂
    /-
      🎉 no goals
    -/


theorem fourierIntegral_continuousMultilinearMap_apply
    {f : V → (ContinuousMultilinearMap ℝ M E)} {m : (i : ι) → M i} {w : W} (he : Continuous e)
    (hf : Integrable f μ) :
    fourierIntegral e μ L.toLinearMap₂ f w m =
      fourierIntegral e μ L.toLinearMap₂ (fun x ↦ f x m) w := by
  /-
    𝕜 : Type u_1
    ι : Type u_2
    E : Type u_3
    V : Type u_5
    W : Type u_6
    inst✝¹¹ : Fintype ι
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NormedAddCommGroup V
    inst✝⁸ : NormedSpace 𝕜 V
    inst✝⁷ : MeasurableSpace V
    inst✝⁶ : BorelSpace V
    inst✝⁵ : NormedAddCommGroup W
    inst✝⁴ : NormedSpace 𝕜 W
    e : AddChar 𝕜 Circle
    μ : MeasureTheory.Measure V
    L : ContinuousLinearMap (RingHom.id 𝕜) V (ContinuousLinearMap (RingHom.id 𝕜) W …
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Complex E
    M : ι → Type u_7
    inst✝¹ : (i : ι) → NormedAddCommGroup (M i)
    inst✝ : (i : ι) → NormedSpace Real (M i)
    f : V → ContinuousMultilinearMap Real M E
    m : (i : ι) → M i
    w : W
    he : Continuous ⇑e
    hf : MeasureTheory.Integrable f μ
    ⊢ Eq ((VectorFourier.fourierIntegral e μ L.toLinearMap₂ f w) m) (VectorFourier …
  -/
  rw [fourierIntegral, ContinuousMultilinearMap.integral_apply]
    /-
      𝕜 : Type u_1
      ι : Type u_2
      E : Type u_3
      V : Type u_5
      W : Type u_6
      inst✝¹¹ : Fintype ι
      inst✝¹⁰ : NontriviallyNormedField 𝕜
      inst✝⁹ : NormedAddCommGroup V
      inst✝⁸ : NormedSpace 𝕜 V
      inst✝⁷ : MeasurableSpace V
      inst✝⁶ : BorelSpace V
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace 𝕜 W
      e : AddChar 𝕜 Circle
      μ : MeasureTheory.Measure V
      L : ContinuousLinearMap (RingHom.id 𝕜) V (ContinuousLinearMap (RingHom.id 𝕜) W …
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Complex E
      M : ι → Type u_7
      inst✝¹ : (i : ι) → NormedAddCommGroup (M i)
      inst✝ : (i : ι) → NormedSpace Real (M i)
      f : V → ContinuousMultilinearMap Real M E
      m : (i : ι) → M i
      w : W
      he : Continuous ⇑e
      hf : MeasureTheory.Integrable f μ
      ⊢ Eq (MeasureTheory.integral μ fun x => (HSMul.hSMul (e (Neg.neg ((L.toLinearM …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case φ_int
      𝕜 : Type u_1
      ι : Type u_2
      E : Type u_3
      V : Type u_5
      W : Type u_6
      inst✝¹¹ : Fintype ι
      inst✝¹⁰ : NontriviallyNormedField 𝕜
      inst✝⁹ : NormedAddCommGroup V
      inst✝⁸ : NormedSpace 𝕜 V
      inst✝⁷ : MeasurableSpace V
      inst✝⁶ : BorelSpace V
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace 𝕜 W
      e : AddChar 𝕜 Circle
      μ : MeasureTheory.Measure V
      L : ContinuousLinearMap (RingHom.id 𝕜) V (ContinuousLinearMap (RingHom.id 𝕜) W …
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Complex E
      M : ι → Type u_7
      inst✝¹ : (i : ι) → NormedAddCommGroup (M i)
      inst✝ : (i : ι) → NormedSpace Real (M i)
      f : V → ContinuousMultilinearMap Real M E
      m : (i : ι) → M i
      w : W
      he : Continuous ⇑e
      hf : MeasureTheory.Integrable f μ
      ⊢ MeasureTheory.Integrable (fun v => HSMul.hSMul (e (Neg.neg ((L.toLinearMap₂  …
    -/
  · apply (fourierIntegral_convergent_iff he _ _).2 hf
    /-
      𝕜 : Type u_1
      ι : Type u_2
      E : Type u_3
      V : Type u_5
      W : Type u_6
      inst✝¹¹ : Fintype ι
      inst✝¹⁰ : NontriviallyNormedField 𝕜
      inst✝⁹ : NormedAddCommGroup V
      inst✝⁸ : NormedSpace 𝕜 V
      inst✝⁷ : MeasurableSpace V
      inst✝⁶ : BorelSpace V
      inst✝⁵ : NormedAddCommGroup W
      inst✝⁴ : NormedSpace 𝕜 W
      e : AddChar 𝕜 Circle
      μ : MeasureTheory.Measure V
      L : ContinuousLinearMap (RingHom.id 𝕜) V (ContinuousLinearMap (RingHom.id 𝕜) W …
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Complex E
      M : ι → Type u_7
      inst✝¹ : (i : ι) → NormedAddCommGroup (M i)
      inst✝ : (i : ι) → NormedSpace Real (M i)
      f : V → ContinuousMultilinearMap Real M E
      m : (i : ι) → M i
      w : W
      he : Continuous ⇑e
      hf : MeasureTheory.Integrable f μ
      ⊢ Continuous fun p => (L.toLinearMap₂ p.1) p.2
    -/
    exact L.continuous₂
    /-
      🎉 no goals
    -/


/-- The Fourier transform integral for `f : 𝕜 → E`, with respect to the measure `μ` and additive
character `e`. -/
def fourierIntegral (e : AddChar 𝕜 𝕊) (μ : Measure 𝕜) (f : 𝕜 → E) (w : 𝕜) : E :=
  VectorFourier.fourierIntegral e μ (LinearMap.mul 𝕜 𝕜) f w


theorem fourierIntegral_def (e : AddChar 𝕜 𝕊) (μ : Measure 𝕜) (f : 𝕜 → E) (w : 𝕜) :
    fourierIntegral e μ f w = ∫ v : 𝕜, e (-(v * w)) • f v ∂μ :=
  rfl


theorem fourierIntegral_const_smul (e : AddChar 𝕜 𝕊) (μ : Measure 𝕜) (f : 𝕜 → E) (r : ℂ) :
    fourierIntegral e μ (r • f) = r • fourierIntegral e μ f :=
  VectorFourier.fourierIntegral_const_smul _ _ _ _ _


/-- The uniform norm of the Fourier transform of `f` is bounded by the `L¹` norm of `f`. -/
theorem norm_fourierIntegral_le_integral_norm (e : AddChar 𝕜 𝕊) (μ : Measure 𝕜)
    (f : 𝕜 → E) (w : 𝕜) : ‖fourierIntegral e μ f w‖ ≤ ∫ x : 𝕜, ‖f x‖ ∂μ :=
  VectorFourier.norm_fourierIntegral_le_integral_norm _ _ _ _ _


/-- The Fourier transform converts right-translation into scalar multiplication by a phase
factor. -/
theorem fourierIntegral_comp_add_right [MeasurableAdd 𝕜] (e : AddChar 𝕜 𝕊) (μ : Measure 𝕜)
    [μ.IsAddRightInvariant] (f : 𝕜 → E) (v₀ : 𝕜) :
    fourierIntegral e μ (f ∘ fun v ↦ v + v₀) = fun w ↦ e (v₀ * w) • fourierIntegral e μ f w :=
  VectorFourier.fourierIntegral_comp_add_right _ _ _ _ _


/-- The standard additive character of `ℝ`, given by `fun x ↦ exp (2 * π * x * I)`. -/
def fourierChar : AddChar ℝ 𝕊 where
  toFun z := .exp (2 * π * z)
                         /-
                           ⊢ Eq ((fun z => Circle.exp (HMul.hMul (HMul.hMul 2 Real.pi) z)) 0) 1
                         -/
  map_zero_eq_one' := by simp only; rw [mul_zero, Circle.exp_zero]
                                    /-
                                      🎉 no goals
                                    -/
                            /-
                              x y : Real
                              ⊢ Eq ((fun z => Circle.exp (HMul.hMul (HMul.hMul 2 Real.pi) z)) (HAdd.hAdd x y …
                            -/
  map_add_eq_mul' x y := by simp only; rw [mul_add, Circle.exp_add]
                                       /-
                                         🎉 no goals
                                       -/


@[inherit_doc] scoped[FourierTransform] notation "𝐞" => Real.fourierChar


theorem fourierChar_apply (x : ℝ) : 𝐞 x = Complex.exp (↑(2 * π * x) * Complex.I) :=
  rfl


@[continuity]
theorem continuous_fourierChar : Continuous 𝐞 := Circle.exp.continuous.comp (continuous_mul_left _)


theorem vector_fourierIntegral_eq_integral_exp_smul {V : Type*} [AddCommGroup V] [Module ℝ V]
    [MeasurableSpace V] {W : Type*} [AddCommGroup W] [Module ℝ W] (L : V →ₗ[ℝ] W →ₗ[ℝ] ℝ)
    (μ : Measure V) (f : V → E) (w : W) :
    VectorFourier.fourierIntegral fourierChar μ L f w =
      ∫ v : V, Complex.exp (↑(-2 * π * L v w) * Complex.I) • f v ∂μ := by
  simp_rw [VectorFourier.fourierIntegral, Circle.smul_def, Real.fourierChar_apply, mul_neg,
    neg_mul]


/-- The Fourier integral is well defined iff the function is integrable. Version with a general
continuous bilinear function `L`. For the specialization to the inner product in an inner product
space, see `Real.fourierIntegral_convergent_iff`. -/
@[simp]
theorem fourierIntegral_convergent_iff' {V W : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]
    [NormedAddCommGroup W] [NormedSpace ℝ W] [MeasurableSpace V] [BorelSpace V] {μ : Measure V}
    {f : V → E} (L : V →L[ℝ] W →L[ℝ] ℝ) (w : W) :
    Integrable (fun v : V ↦ 𝐞 (- L v w) • f v) μ ↔ Integrable f μ :=
  VectorFourier.fourierIntegral_convergent_iff (E := E) (L := L.toLinearMap₂)
    continuous_fourierChar L.continuous₂ _


theorem fourierIntegral_continuousLinearMap_apply'
    {f : V → (F →L[ℝ] E)} {a : F} {w : W} (hf : Integrable f μ) :
    VectorFourier.fourierIntegral 𝐞 μ L.toLinearMap₂ f w a =
      VectorFourier.fourierIntegral 𝐞 μ L.toLinearMap₂ (fun x ↦ f x a) w :=
  VectorFourier.fourierIntegral_continuousLinearMap_apply continuous_fourierChar hf


theorem fourierIntegral_continuousMultilinearMap_apply'
    {f : V → ContinuousMultilinearMap ℝ M E} {m : (i : ι) → M i} {w : W} (hf : Integrable f μ) :
    VectorFourier.fourierIntegral 𝐞 μ L.toLinearMap₂ f w m =
      VectorFourier.fourierIntegral 𝐞 μ L.toLinearMap₂ (fun x ↦ f x m) w :=
  VectorFourier.fourierIntegral_continuousMultilinearMap_apply continuous_fourierChar hf


@[simp] theorem fourierIntegral_convergent_iff {μ : Measure V} {f : V → E} (w : V) :
    Integrable (fun v : V ↦ 𝐞 (- ⟪v, w⟫) • f v) μ ↔ Integrable f μ :=
  fourierIntegral_convergent_iff' (innerSL ℝ) w


/-- The Fourier transform of a function on an inner product space, with respect to the standard
additive character `ω ↦ exp (2 i π ω)`. -/
def fourierIntegral (f : V → E) (w : V) : E :=
  VectorFourier.fourierIntegral 𝐞 volume (innerₗ V) f w


/-- The inverse Fourier transform of a function on an inner product space, defined as the Fourier
transform but with opposite sign in the exponential. -/
def fourierIntegralInv (f : V → E) (w : V) : E :=
  VectorFourier.fourierIntegral 𝐞 volume (-innerₗ V) f w


@[inherit_doc] scoped[FourierTransform] notation "𝓕" => Real.fourierIntegral

@[inherit_doc] scoped[FourierTransform] notation "𝓕⁻" => Real.fourierIntegralInv


lemma fourierIntegral_eq (f : V → E) (w : V) :
    𝓕 f w = ∫ v, 𝐞 (-⟪v, w⟫) • f v := rfl


lemma fourierIntegral_eq' (f : V → E) (w : V) :
    𝓕 f w = ∫ v, Complex.exp ((↑(-2 * π * ⟪v, w⟫) * Complex.I)) • f v := by
  /-
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    V : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : MeasurableSpace V
    inst✝¹ : BorelSpace V
    inst✝ : FiniteDimensional Real V
    f : V → E
    w : V
    ⊢ Eq (Real.fourierIntegral f w) (MeasureTheory.integral MeasureTheory.MeasureS …
  -/
  simp_rw [fourierIntegral_eq, Circle.smul_def, Real.fourierChar_apply, mul_neg, neg_mul]
  /-
    🎉 no goals
  -/


lemma fourierIntegralInv_eq (f : V → E) (w : V) :
    𝓕⁻ f w = ∫ v, 𝐞 ⟪v, w⟫ • f v := by
  /-
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    V : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : MeasurableSpace V
    inst✝¹ : BorelSpace V
    inst✝ : FiniteDimensional Real V
    f : V → E
    w : V
    ⊢ Eq (Real.fourierIntegralInv f w) (MeasureTheory.integral MeasureTheory.Measu …
  -/
  simp [fourierIntegralInv, VectorFourier.fourierIntegral]
  /-
    🎉 no goals
  -/


lemma fourierIntegralInv_eq' (f : V → E) (w : V) :
    𝓕⁻ f w = ∫ v, Complex.exp ((↑(2 * π * ⟪v, w⟫) * Complex.I)) • f v := by
  /-
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    V : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : MeasurableSpace V
    inst✝¹ : BorelSpace V
    inst✝ : FiniteDimensional Real V
    f : V → E
    w : V
    ⊢ Eq (Real.fourierIntegralInv f w) (MeasureTheory.integral MeasureTheory.Measu …
  -/
  simp_rw [fourierIntegralInv_eq, Circle.smul_def, Real.fourierChar_apply]
  /-
    🎉 no goals
  -/


lemma fourierIntegral_comp_linearIsometry (A : W ≃ₗᵢ[ℝ] V) (f : V → E) (w : W) :
    𝓕 (f ∘ A) w = (𝓕 f) (A w) := by
  simp only [fourierIntegral_eq, ← A.inner_map_map, Function.comp_apply,
    ← MeasurePreserving.integral_comp A.measurePreserving A.toHomeomorph.measurableEmbedding]


lemma fourierIntegralInv_eq_fourierIntegral_neg (f : V → E) (w : V) :
    𝓕⁻ f w = 𝓕 f (-w) := by
  /-
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    V : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : MeasurableSpace V
    inst✝¹ : BorelSpace V
    inst✝ : FiniteDimensional Real V
    f : V → E
    w : V
    ⊢ Eq (Real.fourierIntegralInv f w) (Real.fourierIntegral f (Neg.neg w))
  -/
  simp [fourierIntegral_eq, fourierIntegralInv_eq]
  /-
    🎉 no goals
  -/


lemma fourierIntegralInv_eq_fourierIntegral_comp_neg (f : V → E) :
    𝓕⁻ f = 𝓕 (fun x ↦ f (-x)) := by
  /-
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    V : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : MeasurableSpace V
    inst✝¹ : BorelSpace V
    inst✝ : FiniteDimensional Real V
    f : V → E
    ⊢ Eq (Real.fourierIntegralInv f) (Real.fourierIntegral fun x => f (Neg.neg x))
  -/
  ext y
  /-
    case h
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    V : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : MeasurableSpace V
    inst✝¹ : BorelSpace V
    inst✝ : FiniteDimensional Real V
    f : V → E
    y : V
    ⊢ Eq (Real.fourierIntegralInv f y) (Real.fourierIntegral (fun x => f (Neg.neg  …
  -/
  rw [fourierIntegralInv_eq_fourierIntegral_neg]
  /-
    case h
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    V : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : MeasurableSpace V
    inst✝¹ : BorelSpace V
    inst✝ : FiniteDimensional Real V
    f : V → E
    y : V
    ⊢ Eq (Real.fourierIntegral f (Neg.neg y)) (Real.fourierIntegral (fun x => f (N …
  -/
  change 𝓕 f (LinearIsometryEquiv.neg ℝ y) = 𝓕 (f ∘ LinearIsometryEquiv.neg ℝ) y
  /-
    case h
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    V : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : MeasurableSpace V
    inst✝¹ : BorelSpace V
    inst✝ : FiniteDimensional Real V
    f : V → E
    y : V
    ⊢ Eq (Real.fourierIntegral f ((LinearIsometryEquiv.neg Real) y)) (Real.fourier …
  -/
  exact (fourierIntegral_comp_linearIsometry _ _ _).symm
  /-
    🎉 no goals
  -/


lemma fourierIntegralInv_comm (f : V → E) :
    𝓕 (𝓕⁻ f) = 𝓕⁻ (𝓕 f) := by
  /-
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    V : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : MeasurableSpace V
    inst✝¹ : BorelSpace V
    inst✝ : FiniteDimensional Real V
    f : V → E
    ⊢ Eq (Real.fourierIntegral (Real.fourierIntegralInv f)) (Real.fourierIntegralI …
  -/
  conv_rhs => rw [fourierIntegralInv_eq_fourierIntegral_comp_neg]
  /-
    E : Type u_1
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Complex E
    V : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : MeasurableSpace V
    inst✝¹ : BorelSpace V
    inst✝ : FiniteDimensional Real V
    f : V → E
    ⊢ Eq (Real.fourierIntegral (Real.fourierIntegralInv f)) (Real.fourierIntegral  …
  -/
  simp_rw [← fourierIntegralInv_eq_fourierIntegral_neg]
  /-
    🎉 no goals
  -/


lemma fourierIntegralInv_comp_linearIsometry (A : W ≃ₗᵢ[ℝ] V) (f : V → E) (w : W) :
    𝓕⁻ (f ∘ A) w = (𝓕⁻ f) (A w) := by
  /-
    E : Type u_1
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : NormedSpace Complex E
    V : Type u_2
    inst✝⁹ : NormedAddCommGroup V
    inst✝⁸ : InnerProductSpace Real V
    inst✝⁷ : MeasurableSpace V
    inst✝⁶ : BorelSpace V
    W : Type u_3
    inst✝⁵ : NormedAddCommGroup W
    inst✝⁴ : InnerProductSpace Real W
    inst✝³ : MeasurableSpace W
    inst✝² : BorelSpace W
    inst✝¹ : FiniteDimensional Real W
    inst✝ : FiniteDimensional Real V
    A : LinearIsometryEquiv (RingHom.id Real) W V
    f : V → E
    w : W
    ⊢ Eq (Real.fourierIntegralInv (Function.comp f ⇑A) w) (Real.fourierIntegralInv …
  -/
  simp [fourierIntegralInv_eq_fourierIntegral_neg, fourierIntegral_comp_linearIsometry]
  /-
    🎉 no goals
  -/


theorem fourierIntegral_real_eq (f : ℝ → E) (w : ℝ) :
    fourierIntegral f w = ∫ v : ℝ, 𝐞 (-(v * w)) • f v :=
  rfl


@[deprecated (since := "2024-02-21")] alias fourierIntegral_def := fourierIntegral_real_eq


theorem fourierIntegral_real_eq_integral_exp_smul (f : ℝ → E) (w : ℝ) :
    𝓕 f w = ∫ v : ℝ, Complex.exp (↑(-2 * π * v * w) * Complex.I) • f v := by
  simp_rw [fourierIntegral_real_eq, Circle.smul_def, Real.fourierChar_apply, mul_neg, neg_mul,
    mul_assoc]


@[deprecated (since := "2024-02-21")]
alias fourierIntegral_eq_integral_exp_smul := fourierIntegral_real_eq_integral_exp_smul


theorem fourierIntegral_continuousLinearMap_apply
    {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F]
                                                /-
                                                  E : Type u_1
                                                  inst✝¹³ : NormedAddCommGroup E
                                                  inst✝¹² : NormedSpace Complex E
                                                  V : Type u_2
                                                  inst✝¹¹ : NormedAddCommGroup V
                                                  inst✝¹⁰ : InnerProductSpace Real V
                                                  inst✝⁹ : MeasurableSpace V
                                                  inst✝⁸ : BorelSpace V
                                                  W : Type u_3
                                                  inst✝⁷ : NormedAddCommGroup W
                                                  inst✝⁶ : InnerProductSpace Real W
                                                  inst✝⁵ : MeasurableSpace W
                                                  inst✝⁴ : BorelSpace W
                                                  inst✝³ : FiniteDimensional Real W
                                                  inst✝² : FiniteDimensional Real V
                                                  F : Type u_4
                                                  inst✝¹ : NormedAddCommGroup F
                                                  inst✝ : NormedSpace Real F
                                                  f : V → ContinuousLinearMap (RingHom.id Real) F E
                                                  a : F
                                                  v : V
                                                  ⊢ MeasureTheory.Measure V
                                                -/
    {f : V → (F →L[ℝ] E)} {a : F} {v : V} (hf : Integrable f) :
                                                /-
                                                  🎉 no goals
                                                -/
    𝓕 f v a = 𝓕 (fun x ↦ f x a) v :=
  fourierIntegral_continuousLinearMap_apply' (L := innerSL ℝ) hf


theorem fourierIntegral_continuousMultilinearMap_apply {ι : Type*} [Fintype ι]
    {M : ι → Type*} [∀ i, NormedAddCommGroup (M i)] [∀ i, NormedSpace ℝ (M i)]
                                                                               /-
                                                                                 E : Type u_1
                                                                                 inst✝¹⁴ : NormedAddCommGroup E
                                                                                 inst✝¹³ : NormedSpace Complex E
                                                                                 V : Type u_2
                                                                                 inst✝¹² : NormedAddCommGroup V
                                                                                 inst✝¹¹ : InnerProductSpace Real V
                                                                                 inst✝¹⁰ : MeasurableSpace V
                                                                                 inst✝⁹ : BorelSpace V
                                                                                 W : Type u_3
                                                                                 inst✝⁸ : NormedAddCommGroup W
                                                                                 inst✝⁷ : InnerProductSpace Real W
                                                                                 inst✝⁶ : MeasurableSpace W
                                                                                 inst✝⁵ : BorelSpace W
                                                                                 inst✝⁴ : FiniteDimensional Real W
                                                                                 inst✝³ : FiniteDimensional Real V
                                                                                 ι : Type u_4
                                                                                 inst✝² : Fintype ι
                                                                                 M : ι → Type u_5
                                                                                 inst✝¹ : (i : ι) → NormedAddCommGroup (M i)
                                                                                 inst✝ : (i : ι) → NormedSpace Real (M i)
                                                                                 f : V → ContinuousMultilinearMap Real M E
                                                                                 m : (i : ι) → M i
                                                                                 v : V
                                                                                 ⊢ MeasureTheory.Measure V
                                                                               -/
    {f : V → ContinuousMultilinearMap ℝ M E} {m : (i : ι) → M i} {v : V} (hf : Integrable f) :
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
    𝓕 f v m = 𝓕 (fun x ↦ f x m) v :=
  fourierIntegral_continuousMultilinearMap_apply' (L := innerSL ℝ) hf


