local notation "⟪" x ", " y "⟫" => @inner 𝕜 E _ x y


local notation "⟪" x ", " y "⟫₂" => @inner 𝕜 (α →₂[μ] E) _ x y

-- Porting note: the argument `E` of `condexpL2` is not automatically filled in Lean 4.
-- To avoid typing `(E := _)` every time it is made explicit.

/-- Conditional expectation of a function in L2 with respect to a sigma-algebra -/
noncomputable def condexpL2 (hm : m ≤ m0) : (α →₂[μ] E) →L[𝕜] lpMeas E 𝕜 m 2 μ :=
  @orthogonalProjection 𝕜 (α →₂[μ] E) _ _ _ (lpMeas E 𝕜 m 2 μ)
    haveI : Fact (m ≤ m0) := ⟨hm⟩
    inferInstance


theorem aeStronglyMeasurable'_condexpL2 (hm : m ≤ m0) (f : α →₂[μ] E) :
    AEStronglyMeasurable' (β := E) m (condexpL2 E 𝕜 hm f) μ :=
  lpMeas.aeStronglyMeasurable' _


theorem integrableOn_condexpL2_of_measure_ne_top (hm : m ≤ m0) (hμs : μ s ≠ ∞) (f : α →₂[μ] E) :
    IntegrableOn (ε := E) (condexpL2 E 𝕜 hm f) s μ :=
  integrableOn_Lp_of_measure_ne_top (condexpL2 E 𝕜 hm f : α →₂[μ] E) fact_one_le_two_ennreal.elim
    hμs


theorem integrable_condexpL2_of_isFiniteMeasure (hm : m ≤ m0) [IsFiniteMeasure μ] {f : α →₂[μ] E} :
    Integrable (ε := E) (condexpL2 E 𝕜 hm f) μ :=
  integrableOn_univ.mp <| integrableOn_condexpL2_of_measure_ne_top hm (measure_ne_top _ _) f


theorem norm_condexpL2_le_one (hm : m ≤ m0) : ‖@condexpL2 α E 𝕜 _ _ _ _ _ _ μ hm‖ ≤ 1 :=
  haveI : Fact (m ≤ m0) := ⟨hm⟩
  orthogonalProjection_norm_le _


theorem norm_condexpL2_le (hm : m ≤ m0) (f : α →₂[μ] E) : ‖condexpL2 E 𝕜 hm f‖ ≤ ‖f‖ :=
  ((@condexpL2 _ E 𝕜 _ _ _ _ _ _ μ hm).le_opNorm f).trans
    (mul_le_of_le_one_left (norm_nonneg _) (norm_condexpL2_le_one hm))


theorem eLpNorm_condexpL2_le (hm : m ≤ m0) (f : α →₂[μ] E) :
    eLpNorm (ε := E) (condexpL2 E 𝕜 hm f) 2 μ ≤ eLpNorm f 2 μ := by
  rw [lpMeas_coe, ← ENNReal.toReal_le_toReal (Lp.eLpNorm_ne_top _) (Lp.eLpNorm_ne_top _), ←
    Lp.norm_def, ← Lp.norm_def, Submodule.norm_coe]
  /-
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_7
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 2 μ) x
    ⊢ LE.le (Norm.norm ((MeasureTheory.condexpL2 E 𝕜 hm) f)) (Norm.norm f)
  -/
  exact norm_condexpL2_le hm f
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-27")]
alias snorm_condexpL2_le := eLpNorm_condexpL2_le


theorem norm_condexpL2_coe_le (hm : m ≤ m0) (f : α →₂[μ] E) :
    ‖(condexpL2 E 𝕜 hm f : α →₂[μ] E)‖ ≤ ‖f‖ := by
  /-
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_7
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 2 μ) x
    ⊢ LE.le (Norm.norm ↑((MeasureTheory.condexpL2 E 𝕜 hm) f)) (Norm.norm f)
  -/
  rw [Lp.norm_def, Lp.norm_def, ← lpMeas_coe]
  /-
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_7
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 2 μ) x
    ⊢ LE.le (MeasureTheory.eLpNorm (↑↑↑((MeasureTheory.condexpL2 E 𝕜 hm) f)) 2 μ). …
  -/
  exact ENNReal.toReal_mono (Lp.eLpNorm_ne_top _) (eLpNorm_condexpL2_le hm f)
  /-
    🎉 no goals
  -/


theorem inner_condexpL2_left_eq_right (hm : m ≤ m0) {f g : α →₂[μ] E} :
    ⟪(condexpL2 E 𝕜 hm f : α →₂[μ] E), g⟫₂ = ⟪f, (condexpL2 E 𝕜 hm g : α →₂[μ] E)⟫₂ :=
  haveI : Fact (m ≤ m0) := ⟨hm⟩
  inner_orthogonalProjection_left_eq_right _ f g


theorem condexpL2_indicator_of_measurable (hm : m ≤ m0) (hs : MeasurableSet[m] s) (hμs : μ s ≠ ∞)
    (c : E) :
    (condexpL2 E 𝕜 hm (indicatorConstLp 2 (hm s hs) hμs c) : α →₂[μ] E) =
      indicatorConstLp 2 (hm s hs) hμs c := by
  /-
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_7
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : E
    ⊢ Eq (↑((MeasureTheory.condexpL2 E 𝕜 hm) (MeasureTheory.indicatorConstLp 2 ⋯ h …
  -/
  rw [condexpL2]
  /-
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_7
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : E
    ⊢ Eq (↑((orthogonalProjection (MeasureTheory.lpMeas E 𝕜 m 2 μ)) (MeasureTheory …
  -/
  haveI : Fact (m ≤ m0) := ⟨hm⟩
  have h_mem : indicatorConstLp 2 (hm s hs) hμs c ∈ lpMeas E 𝕜 m 2 μ :=
    mem_lpMeas_indicatorConstLp hm hs hμs
  /-
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_7
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : E
    this : Fact (LE.le m m0)
    h_mem : Membership.mem (MeasureTheory.lpMeas E 𝕜 m 2 μ) (MeasureTheory.indicat …
    ⊢ Eq (↑((orthogonalProjection (MeasureTheory.lpMeas E 𝕜 m 2 μ)) (MeasureTheory …
  -/
  let ind := (⟨indicatorConstLp 2 (hm s hs) hμs c, h_mem⟩ : lpMeas E 𝕜 m 2 μ)
  /-
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_7
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : E
    this : Fact (LE.le m m0)
    h_mem : Membership.mem (MeasureTheory.lpMeas E 𝕜 m 2 μ) (MeasureTheory.indicat …
    ind : Subtype fun x => Membership.mem (MeasureTheory.lpMeas E 𝕜 m 2 μ) x := ⟨M …
    ⊢ Eq (↑((orthogonalProjection (MeasureTheory.lpMeas E 𝕜 m 2 μ)) (MeasureTheory …
  -/
  have h_coe_ind : (ind : α →₂[μ] E) = indicatorConstLp 2 (hm s hs) hμs c := rfl
  /-
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_7
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : E
    this : Fact (LE.le m m0)
    h_mem : Membership.mem (MeasureTheory.lpMeas E 𝕜 m 2 μ) (MeasureTheory.indicat …
    ind : Subtype fun x => Membership.mem (MeasureTheory.lpMeas E 𝕜 m 2 μ) x := ⟨M …
    h_coe_ind : Eq (↑ind) (MeasureTheory.indicatorConstLp 2 ⋯ hμs c)
    ⊢ Eq (↑((orthogonalProjection (MeasureTheory.lpMeas E 𝕜 m 2 μ)) (MeasureTheory …
  -/
  have h_orth_mem := orthogonalProjection_mem_subspace_eq_self ind
  /-
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_7
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : E
    this : Fact (LE.le m m0)
    h_mem : Membership.mem (MeasureTheory.lpMeas E 𝕜 m 2 μ) (MeasureTheory.indicat …
    ind : Subtype fun x => Membership.mem (MeasureTheory.lpMeas E 𝕜 m 2 μ) x := ⟨M …
    h_coe_ind : Eq (↑ind) (MeasureTheory.indicatorConstLp 2 ⋯ hμs c)
    h_orth_mem : Eq ((orthogonalProjection (MeasureTheory.lpMeas E 𝕜 m 2 μ)) ↑ind) …
    ⊢ Eq (↑((orthogonalProjection (MeasureTheory.lpMeas E 𝕜 m 2 μ)) (MeasureTheory …
  -/
  rw [← h_coe_ind, h_orth_mem]
  /-
    🎉 no goals
  -/


theorem inner_condexpL2_eq_inner_fun (hm : m ≤ m0) (f g : α →₂[μ] E)
    (hg : AEStronglyMeasurable' m g μ) :
    ⟪(condexpL2 E 𝕜 hm f : α →₂[μ] E), g⟫₂ = ⟪f, g⟫₂ := by
  /-
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_7
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp E 2 μ) x
    hg : MeasureTheory.AEStronglyMeasurable' m (↑↑g) μ
    ⊢ Eq (Inner.inner (↑((MeasureTheory.condexpL2 E 𝕜 hm) f)) g) (Inner.inner f g)
  -/
  symm
  /-
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_7
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp E 2 μ) x
    hg : MeasureTheory.AEStronglyMeasurable' m (↑↑g) μ
    ⊢ Eq (Inner.inner f g) (Inner.inner (↑((MeasureTheory.condexpL2 E 𝕜 hm) f)) g)
  -/
  rw [← sub_eq_zero, ← inner_sub_left, condexpL2]
  /-
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_7
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp E 2 μ) x
    hg : MeasureTheory.AEStronglyMeasurable' m (↑↑g) μ
    ⊢ Eq (Inner.inner (HSub.hSub f ↑((orthogonalProjection (MeasureTheory.lpMeas E …
  -/
  simp only [mem_lpMeas_iff_aeStronglyMeasurable'.mpr hg, orthogonalProjection_inner_eq_zero f g]
  /-
    🎉 no goals
  -/


theorem integral_condexpL2_eq_of_fin_meas_real (f : Lp 𝕜 2 μ) (hs : MeasurableSet[m] s)
    (hμs : μ s ≠ ∞) : ∫ x in s, (condexpL2 𝕜 𝕜 hm f : α → 𝕜) x ∂μ = ∫ x in s, f x ∂μ := by
  /-
    α : Type u_1
    𝕜 : Type u_7
    inst✝ : RCLike 𝕜
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp 𝕜 2 μ) x
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun x => ↑↑↑((MeasureTheory.condex …
  -/
  rw [← L2.inner_indicatorConstLp_one (𝕜 := 𝕜) (hm s hs) hμs f]
  have h_eq_inner : ∫ x in s, (condexpL2 𝕜 𝕜 hm f : α → 𝕜) x ∂μ =
      inner (indicatorConstLp 2 (hm s hs) hμs (1 : 𝕜)) (condexpL2 𝕜 𝕜 hm f) := by
    rw [L2.inner_indicatorConstLp_one (hm s hs) hμs]
  /-
    α : Type u_1
    𝕜 : Type u_7
    inst✝ : RCLike 𝕜
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp 𝕜 2 μ) x
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    h_eq_inner : Eq (MeasureTheory.integral (μ.restrict s) fun x => ↑↑↑((MeasureTh …
    ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun x => ↑↑↑((MeasureTheory.condex …
  -/
  rw [h_eq_inner, ← inner_condexpL2_left_eq_right, condexpL2_indicator_of_measurable hm hs hμs]
  /-
    🎉 no goals
  -/


theorem lintegral_nnnorm_condexpL2_le (hs : MeasurableSet[m] s) (hμs : μ s ≠ ∞) (f : Lp ℝ 2 μ) :
    ∫⁻ x in s, ‖(condexpL2 ℝ ℝ hm f : α → ℝ) x‖₊ ∂μ ≤ ∫⁻ x in s, ‖f x‖₊ ∂μ := by
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp Real 2 μ) x
    ⊢ LE.le (MeasureTheory.lintegral (μ.restrict s) fun x => ↑(NNNorm.nnnorm (↑↑↑( …
  -/
  let h_meas := lpMeas.aeStronglyMeasurable' (condexpL2 ℝ ℝ hm f)
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp Real 2 μ) x
    h_meas : MeasureTheory.AEStronglyMeasurable' m (↑↑↑((MeasureTheory.condexpL2 R …
    ⊢ LE.le (MeasureTheory.lintegral (μ.restrict s) fun x => ↑(NNNorm.nnnorm (↑↑↑( …
  -/
  let g := h_meas.choose
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp Real 2 μ) x
    h_meas : MeasureTheory.AEStronglyMeasurable' m (↑↑↑((MeasureTheory.condexpL2 R …
    g : α → Real := Exists.choose h_meas
    ⊢ LE.le (MeasureTheory.lintegral (μ.restrict s) fun x => ↑(NNNorm.nnnorm (↑↑↑( …
  -/
  have hg_meas : StronglyMeasurable[m] g := h_meas.choose_spec.1
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp Real 2 μ) x
    h_meas : MeasureTheory.AEStronglyMeasurable' m (↑↑↑((MeasureTheory.condexpL2 R …
    g : α → Real := Exists.choose h_meas
    hg_meas : MeasureTheory.StronglyMeasurable g
    ⊢ LE.le (MeasureTheory.lintegral (μ.restrict s) fun x => ↑(NNNorm.nnnorm (↑↑↑( …
  -/
  have hg_eq : g =ᵐ[μ] condexpL2 ℝ ℝ hm f := h_meas.choose_spec.2.symm
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp Real 2 μ) x
    h_meas : MeasureTheory.AEStronglyMeasurable' m (↑↑↑((MeasureTheory.condexpL2 R …
    g : α → Real := Exists.choose h_meas
    hg_meas : MeasureTheory.StronglyMeasurable g
    hg_eq : (MeasureTheory.ae μ).EventuallyEq g ↑↑↑((MeasureTheory.condexpL2 Real  …
    ⊢ LE.le (MeasureTheory.lintegral (μ.restrict s) fun x => ↑(NNNorm.nnnorm (↑↑↑( …
  -/
  have hg_eq_restrict : g =ᵐ[μ.restrict s] condexpL2 ℝ ℝ hm f := ae_restrict_of_ae hg_eq
  have hg_nnnorm_eq : (fun x => (‖g x‖₊ : ℝ≥0∞)) =ᵐ[μ.restrict s] fun x =>
      (‖(condexpL2 ℝ ℝ hm f : α → ℝ) x‖₊ : ℝ≥0∞) := by
    refine hg_eq_restrict.mono fun x hx => ?_
    dsimp only
    simp_rw [hx]
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp Real 2 μ) x
    h_meas : MeasureTheory.AEStronglyMeasurable' m (↑↑↑((MeasureTheory.condexpL2 R …
    g : α → Real := Exists.choose h_meas
    hg_meas : MeasureTheory.StronglyMeasurable g
    hg_eq : (MeasureTheory.ae μ).EventuallyEq g ↑↑↑((MeasureTheory.condexpL2 Real  …
    hg_eq_restrict : (MeasureTheory.ae (μ.restrict s)).EventuallyEq g ↑↑↑((Measure …
    hg_nnnorm_eq : (MeasureTheory.ae (μ.restrict s)).EventuallyEq (fun x => ↑(NNNo …
    ⊢ LE.le (MeasureTheory.lintegral (μ.restrict s) fun x => ↑(NNNorm.nnnorm (↑↑↑( …
  -/
  rw [lintegral_congr_ae hg_nnnorm_eq.symm]
  refine lintegral_nnnorm_le_of_forall_fin_meas_integral_eq
    hm (Lp.stronglyMeasurable f) ?_ ?_ ?_ ?_ hs hμs
    /-
      case refine_1
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      hm : LE.le m m0
      hs : MeasurableSet s
      hμs : Ne (μ s) Top.top
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp Real 2 μ) x
      h_meas : MeasureTheory.AEStronglyMeasurable' m (↑↑↑((MeasureTheory.condexpL2 R …
      g : α → Real := Exists.choose h_meas
      hg_meas : MeasureTheory.StronglyMeasurable g
      hg_eq : (MeasureTheory.ae μ).EventuallyEq g ↑↑↑((MeasureTheory.condexpL2 Real  …
      hg_eq_restrict : (MeasureTheory.ae (μ.restrict s)).EventuallyEq g ↑↑↑((Measure …
      hg_nnnorm_eq : (MeasureTheory.ae (μ.restrict s)).EventuallyEq (fun x => ↑(NNNo …
      ⊢ MeasureTheory.IntegrableOn (↑↑f) s μ
    -/
  · exact integrableOn_Lp_of_measure_ne_top f fact_one_le_two_ennreal.elim hμs
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      hm : LE.le m m0
      hs : MeasurableSet s
      hμs : Ne (μ s) Top.top
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp Real 2 μ) x
      h_meas : MeasureTheory.AEStronglyMeasurable' m (↑↑↑((MeasureTheory.condexpL2 R …
      g : α → Real := Exists.choose h_meas
      hg_meas : MeasureTheory.StronglyMeasurable g
      hg_eq : (MeasureTheory.ae μ).EventuallyEq g ↑↑↑((MeasureTheory.condexpL2 Real  …
      hg_eq_restrict : (MeasureTheory.ae (μ.restrict s)).EventuallyEq g ↑↑↑((Measure …
      hg_nnnorm_eq : (MeasureTheory.ae (μ.restrict s)).EventuallyEq (fun x => ↑(NNNo …
      ⊢ MeasureTheory.StronglyMeasurable g
    -/
  · exact hg_meas
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      hm : LE.le m m0
      hs : MeasurableSet s
      hμs : Ne (μ s) Top.top
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp Real 2 μ) x
      h_meas : MeasureTheory.AEStronglyMeasurable' m (↑↑↑((MeasureTheory.condexpL2 R …
      g : α → Real := Exists.choose h_meas
      hg_meas : MeasureTheory.StronglyMeasurable g
      hg_eq : (MeasureTheory.ae μ).EventuallyEq g ↑↑↑((MeasureTheory.condexpL2 Real  …
      hg_eq_restrict : (MeasureTheory.ae (μ.restrict s)).EventuallyEq g ↑↑↑((Measure …
      hg_nnnorm_eq : (MeasureTheory.ae (μ.restrict s)).EventuallyEq (fun x => ↑(NNNo …
      ⊢ MeasureTheory.IntegrableOn g s μ
    -/
  · rw [IntegrableOn, integrable_congr hg_eq_restrict]
    /-
      case refine_3
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      hm : LE.le m m0
      hs : MeasurableSet s
      hμs : Ne (μ s) Top.top
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp Real 2 μ) x
      h_meas : MeasureTheory.AEStronglyMeasurable' m (↑↑↑((MeasureTheory.condexpL2 R …
      g : α → Real := Exists.choose h_meas
      hg_meas : MeasureTheory.StronglyMeasurable g
      hg_eq : (MeasureTheory.ae μ).EventuallyEq g ↑↑↑((MeasureTheory.condexpL2 Real  …
      hg_eq_restrict : (MeasureTheory.ae (μ.restrict s)).EventuallyEq g ↑↑↑((Measure …
      hg_nnnorm_eq : (MeasureTheory.ae (μ.restrict s)).EventuallyEq (fun x => ↑(NNNo …
      ⊢ MeasureTheory.Integrable (↑↑↑((MeasureTheory.condexpL2 Real Real hm) f)) (μ. …
    -/
    exact integrableOn_condexpL2_of_measure_ne_top hm hμs f
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      hm : LE.le m m0
      hs : MeasurableSet s
      hμs : Ne (μ s) Top.top
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp Real 2 μ) x
      h_meas : MeasureTheory.AEStronglyMeasurable' m (↑↑↑((MeasureTheory.condexpL2 R …
      g : α → Real := Exists.choose h_meas
      hg_meas : MeasureTheory.StronglyMeasurable g
      hg_eq : (MeasureTheory.ae μ).EventuallyEq g ↑↑↑((MeasureTheory.condexpL2 Real  …
      hg_eq_restrict : (MeasureTheory.ae (μ.restrict s)).EventuallyEq g ↑↑↑((Measure …
      hg_nnnorm_eq : (MeasureTheory.ae (μ.restrict s)).EventuallyEq (fun x => ↑(NNNo …
      ⊢ ∀ (t : Set α), MeasurableSet t → LT.lt (μ t) Top.top → Eq (MeasureTheory.int …
    -/
  · intro t ht hμt
    /-
      case refine_4
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      hm : LE.le m m0
      hs : MeasurableSet s
      hμs : Ne (μ s) Top.top
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp Real 2 μ) x
      h_meas : MeasureTheory.AEStronglyMeasurable' m (↑↑↑((MeasureTheory.condexpL2 R …
      g : α → Real := Exists.choose h_meas
      hg_meas : MeasureTheory.StronglyMeasurable g
      hg_eq : (MeasureTheory.ae μ).EventuallyEq g ↑↑↑((MeasureTheory.condexpL2 Real  …
      hg_eq_restrict : (MeasureTheory.ae (μ.restrict s)).EventuallyEq g ↑↑↑((Measure …
      hg_nnnorm_eq : (MeasureTheory.ae (μ.restrict s)).EventuallyEq (fun x => ↑(NNNo …
      t : Set α
      ht : MeasurableSet t
      hμt : LT.lt (μ t) Top.top
      ⊢ Eq (MeasureTheory.integral (μ.restrict t) fun x => g x) (MeasureTheory.integ …
    -/
    rw [← integral_condexpL2_eq_of_fin_meas_real f ht hμt.ne]
    /-
      case refine_4
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      hm : LE.le m m0
      hs : MeasurableSet s
      hμs : Ne (μ s) Top.top
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp Real 2 μ) x
      h_meas : MeasureTheory.AEStronglyMeasurable' m (↑↑↑((MeasureTheory.condexpL2 R …
      g : α → Real := Exists.choose h_meas
      hg_meas : MeasureTheory.StronglyMeasurable g
      hg_eq : (MeasureTheory.ae μ).EventuallyEq g ↑↑↑((MeasureTheory.condexpL2 Real  …
      hg_eq_restrict : (MeasureTheory.ae (μ.restrict s)).EventuallyEq g ↑↑↑((Measure …
      hg_nnnorm_eq : (MeasureTheory.ae (μ.restrict s)).EventuallyEq (fun x => ↑(NNNo …
      t : Set α
      ht : MeasurableSet t
      hμt : LT.lt (μ t) Top.top
      ⊢ Eq (MeasureTheory.integral (μ.restrict t) fun x => g x) (MeasureTheory.integ …
    -/
    exact setIntegral_congr_ae (hm t ht) (hg_eq.mono fun x hx _ => hx)
    /-
      🎉 no goals
    -/


theorem condexpL2_ae_eq_zero_of_ae_eq_zero (hs : MeasurableSet[m] s) (hμs : μ s ≠ ∞) {f : Lp ℝ 2 μ}
    (hf : f =ᵐ[μ.restrict s] 0) : condexpL2 ℝ ℝ hm f =ᵐ[μ.restrict s] (0 : α → ℝ) := by
  suffices h_nnnorm_eq_zero : ∫⁻ x in s, ‖(condexpL2 ℝ ℝ hm f : α → ℝ) x‖₊ ∂μ = 0 by
    rw [lintegral_eq_zero_iff] at h_nnnorm_eq_zero
    · refine h_nnnorm_eq_zero.mono fun x hx => ?_
      dsimp only at hx
      rw [Pi.zero_apply] at hx ⊢
      · rwa [ENNReal.coe_eq_zero, nnnorm_eq_zero] at hx
    · refine Measurable.coe_nnreal_ennreal (Measurable.nnnorm ?_)
      rw [lpMeas_coe]
      exact (Lp.stronglyMeasurable _).measurable
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp Real 2 μ) x
    hf : (MeasureTheory.ae (μ.restrict s)).EventuallyEq (↑↑f) 0
    ⊢ Eq (MeasureTheory.lintegral (μ.restrict s) fun x => ↑(NNNorm.nnnorm (↑↑↑((Me …
  -/
  refine le_antisymm ?_ (zero_le _)
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp Real 2 μ) x
    hf : (MeasureTheory.ae (μ.restrict s)).EventuallyEq (↑↑f) 0
    ⊢ LE.le (MeasureTheory.lintegral (μ.restrict s) fun x => ↑(NNNorm.nnnorm (↑↑↑( …
  -/
  refine (lintegral_nnnorm_condexpL2_le hs hμs f).trans (le_of_eq ?_)
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp Real 2 μ) x
    hf : (MeasureTheory.ae (μ.restrict s)).EventuallyEq (↑↑f) 0
    ⊢ Eq (MeasureTheory.lintegral (μ.restrict s) fun x => ↑(NNNorm.nnnorm (↑↑f x)) …
  -/
  rw [lintegral_eq_zero_iff]
    /-
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      hm : LE.le m m0
      hs : MeasurableSet s
      hμs : Ne (μ s) Top.top
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp Real 2 μ) x
      hf : (MeasureTheory.ae (μ.restrict s)).EventuallyEq (↑↑f) 0
      ⊢ (MeasureTheory.ae (μ.restrict s)).EventuallyEq (fun x => ↑(NNNorm.nnnorm (↑↑ …
    -/
  · refine hf.mono fun x hx => ?_
    /-
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      hm : LE.le m m0
      hs : MeasurableSet s
      hμs : Ne (μ s) Top.top
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp Real 2 μ) x
      hf : (MeasureTheory.ae (μ.restrict s)).EventuallyEq (↑↑f) 0
      x : α
      hx : Eq (↑↑f x) (0 x)
      ⊢ Eq ((fun x => ↑(NNNorm.nnnorm (↑↑f x))) x) (0 x)
    -/
    dsimp only
    /-
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      hm : LE.le m m0
      hs : MeasurableSet s
      hμs : Ne (μ s) Top.top
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp Real 2 μ) x
      hf : (MeasureTheory.ae (μ.restrict s)).EventuallyEq (↑↑f) 0
      x : α
      hx : Eq (↑↑f x) (0 x)
      ⊢ Eq (↑(NNNorm.nnnorm (↑↑f x))) (0 x)
    -/
    rw [hx]
    /-
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      hm : LE.le m m0
      hs : MeasurableSet s
      hμs : Ne (μ s) Top.top
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp Real 2 μ) x
      hf : (MeasureTheory.ae (μ.restrict s)).EventuallyEq (↑↑f) 0
      x : α
      hx : Eq (↑↑f x) (0 x)
      ⊢ Eq (↑(NNNorm.nnnorm (0 x))) (0 x)
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      hm : LE.le m m0
      hs : MeasurableSet s
      hμs : Ne (μ s) Top.top
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp Real 2 μ) x
      hf : (MeasureTheory.ae (μ.restrict s)).EventuallyEq (↑↑f) 0
      ⊢ Measurable fun x => ↑(NNNorm.nnnorm (↑↑f x))
    -/
  · exact (Lp.stronglyMeasurable _).ennnorm
    /-
      🎉 no goals
    -/


theorem lintegral_nnnorm_condexpL2_indicator_le_real (hs : MeasurableSet s) (hμs : μ s ≠ ∞)
    (ht : MeasurableSet[m] t) (hμt : μ t ≠ ∞) :
    ∫⁻ a in t, ‖(condexpL2 ℝ ℝ hm (indicatorConstLp 2 hs hμs 1) : α → ℝ) a‖₊ ∂μ ≤ μ (s ∩ t) := by
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    hm : LE.le m m0
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    ht : MeasurableSet t
    hμt : Ne (μ t) Top.top
    ⊢ LE.le (MeasureTheory.lintegral (μ.restrict t) fun a => ↑(NNNorm.nnnorm (↑↑↑( …
  -/
  refine (lintegral_nnnorm_condexpL2_le ht hμt _).trans (le_of_eq ?_)
  have h_eq :
    ∫⁻ x in t, ‖(indicatorConstLp 2 hs hμs (1 : ℝ)) x‖₊ ∂μ =
      ∫⁻ x in t, s.indicator (fun _ => (1 : ℝ≥0∞)) x ∂μ := by
    refine lintegral_congr_ae (ae_restrict_of_ae ?_)
    refine (@indicatorConstLp_coeFn _ _ _ 2 _ _ _ hs hμs (1 : ℝ)).mono fun x hx => ?_
    dsimp only
    rw [hx]
    classical
    simp_rw [Set.indicator_apply]
    split_ifs <;> simp
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    hm : LE.le m m0
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    ht : MeasurableSet t
    hμt : Ne (μ t) Top.top
    h_eq : Eq (MeasureTheory.lintegral (μ.restrict t) fun x => ↑(NNNorm.nnnorm (↑↑ …
    ⊢ Eq (MeasureTheory.lintegral (μ.restrict t) fun x => ↑(NNNorm.nnnorm (↑↑(Meas …
  -/
  rw [h_eq, lintegral_indicator hs, lintegral_const, Measure.restrict_restrict hs]
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    hm : LE.le m m0
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    ht : MeasurableSet t
    hμt : Ne (μ t) Top.top
    h_eq : Eq (MeasureTheory.lintegral (μ.restrict t) fun x => ↑(NNNorm.nnnorm (↑↑ …
    ⊢ Eq (HMul.hMul 1 ((μ.restrict (Inter.inter s t)) Set.univ)) (μ (Inter.inter s …
  -/
  simp only [one_mul, Set.univ_inter, MeasurableSet.univ, Measure.restrict_apply]
  /-
    🎉 no goals
  -/


/-- `condexpL2` commutes with taking inner products with constants. See the lemma
`condexpL2_comp_continuousLinearMap` for a more general result about commuting with continuous
linear maps. -/
theorem condexpL2_const_inner (hm : m ≤ m0) (f : Lp E 2 μ) (c : E) :
    condexpL2 𝕜 𝕜 hm (((Lp.memℒp f).const_inner c).toLp fun a => ⟪c, f a⟫) =ᵐ[μ]
    fun a => ⟪c, (condexpL2 E 𝕜 hm f : α → E) a⟫ := by
  /-
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_7
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 2 μ) x
    c : E
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑↑((MeasureTheory.condexpL2 𝕜 𝕜 hm) (Meas …
  -/
  rw [lpMeas_coe]
  have h_mem_Lp : Memℒp (fun a => ⟪c, (condexpL2 E 𝕜 hm f : α → E) a⟫) 2 μ := by
    refine Memℒp.const_inner _ ?_; rw [lpMeas_coe]; exact Lp.memℒp _
  have h_eq : h_mem_Lp.toLp _ =ᵐ[μ] fun a => ⟪c, (condexpL2 E 𝕜 hm f : α → E) a⟫ :=
    h_mem_Lp.coeFn_toLp
  /-
    α : Type u_1
    E : Type u_2
    𝕜 : Type u_7
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 2 μ) x
    c : E
    h_mem_Lp : MeasureTheory.Memℒp (fun a => Inner.inner c (↑↑↑((MeasureTheory.con …
    h_eq : (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.Memℒp.toLp (fun a => …
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑↑((MeasureTheory.condexpL2 𝕜 𝕜 hm) (Meas …
  -/
  refine EventuallyEq.trans ?_ h_eq
  refine Lp.ae_eq_of_forall_setIntegral_eq' 𝕜 hm _ _ two_ne_zero ENNReal.coe_ne_top
    (fun s _ hμs => integrableOn_condexpL2_of_measure_ne_top hm hμs.ne _) ?_ ?_ ?_ ?_
    /-
      case refine_1
      α : Type u_1
      E : Type u_2
      𝕜 : Type u_7
      inst✝³ : RCLike 𝕜
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace 𝕜 E
      inst✝ : CompleteSpace E
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 2 μ) x
      c : E
      h_mem_Lp : MeasureTheory.Memℒp (fun a => Inner.inner c (↑↑↑((MeasureTheory.con …
      h_eq : (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.Memℒp.toLp (fun a => …
      ⊢ ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → MeasureTheory.Integra …
    -/
  · intro s _ hμs
    /-
      case refine_1
      α : Type u_1
      E : Type u_2
      𝕜 : Type u_7
      inst✝³ : RCLike 𝕜
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace 𝕜 E
      inst✝ : CompleteSpace E
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 2 μ) x
      c : E
      h_mem_Lp : MeasureTheory.Memℒp (fun a => Inner.inner c (↑↑↑((MeasureTheory.con …
      h_eq : (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.Memℒp.toLp (fun a => …
      s : Set α
      a✝ : MeasurableSet s
      hμs : LT.lt (μ s) Top.top
      ⊢ MeasureTheory.IntegrableOn (↑↑(MeasureTheory.Memℒp.toLp (fun a => Inner.inne …
    -/
    rw [IntegrableOn, integrable_congr (ae_restrict_of_ae h_eq)]
    /-
      case refine_1
      α : Type u_1
      E : Type u_2
      𝕜 : Type u_7
      inst✝³ : RCLike 𝕜
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace 𝕜 E
      inst✝ : CompleteSpace E
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 2 μ) x
      c : E
      h_mem_Lp : MeasureTheory.Memℒp (fun a => Inner.inner c (↑↑↑((MeasureTheory.con …
      h_eq : (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.Memℒp.toLp (fun a => …
      s : Set α
      a✝ : MeasurableSet s
      hμs : LT.lt (μ s) Top.top
      ⊢ MeasureTheory.Integrable (fun a => Inner.inner c (↑↑↑((MeasureTheory.condexp …
    -/
    exact (integrableOn_condexpL2_of_measure_ne_top hm hμs.ne _).const_inner _
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      E : Type u_2
      𝕜 : Type u_7
      inst✝³ : RCLike 𝕜
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace 𝕜 E
      inst✝ : CompleteSpace E
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 2 μ) x
      c : E
      h_mem_Lp : MeasureTheory.Memℒp (fun a => Inner.inner c (↑↑↑((MeasureTheory.con …
      h_eq : (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.Memℒp.toLp (fun a => …
      ⊢ ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTheory.int …
    -/
  · intro s hs hμs
    rw [← lpMeas_coe, integral_condexpL2_eq_of_fin_meas_real _ hs hμs.ne,
      integral_congr_ae (ae_restrict_of_ae h_eq), lpMeas_coe, ←
      L2.inner_indicatorConstLp_eq_setIntegral_inner 𝕜 (↑(condexpL2 E 𝕜 hm f)) (hm s hs) c hμs.ne,
      ← inner_condexpL2_left_eq_right, condexpL2_indicator_of_measurable _ hs,
      L2.inner_indicatorConstLp_eq_setIntegral_inner 𝕜 f (hm s hs) c hμs.ne,
      setIntegral_congr_ae (hm s hs)
        ((Memℒp.coeFn_toLp ((Lp.memℒp f).const_inner c)).mono fun x hx _ => hx)]
    /-
      case refine_3
      α : Type u_1
      E : Type u_2
      𝕜 : Type u_7
      inst✝³ : RCLike 𝕜
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace 𝕜 E
      inst✝ : CompleteSpace E
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 2 μ) x
      c : E
      h_mem_Lp : MeasureTheory.Memℒp (fun a => Inner.inner c (↑↑↑((MeasureTheory.con …
      h_eq : (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.Memℒp.toLp (fun a => …
      ⊢ MeasureTheory.AEStronglyMeasurable' m (↑↑↑((MeasureTheory.condexpL2 𝕜 𝕜 hm)  …
    -/
  · rw [← lpMeas_coe]; exact lpMeas.aeStronglyMeasurable' _
                       /-
                         🎉 no goals
                       -/
    /-
      case refine_4
      α : Type u_1
      E : Type u_2
      𝕜 : Type u_7
      inst✝³ : RCLike 𝕜
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace 𝕜 E
      inst✝ : CompleteSpace E
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 2 μ) x
      c : E
      h_mem_Lp : MeasureTheory.Memℒp (fun a => Inner.inner c (↑↑↑((MeasureTheory.con …
      h_eq : (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.Memℒp.toLp (fun a => …
      ⊢ MeasureTheory.AEStronglyMeasurable' m (↑↑(MeasureTheory.Memℒp.toLp (fun a => …
    -/
  · refine AEStronglyMeasurable'.congr ?_ h_eq.symm
    /-
      case refine_4
      α : Type u_1
      E : Type u_2
      𝕜 : Type u_7
      inst✝³ : RCLike 𝕜
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace 𝕜 E
      inst✝ : CompleteSpace E
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E 2 μ) x
      c : E
      h_mem_Lp : MeasureTheory.Memℒp (fun a => Inner.inner c (↑↑↑((MeasureTheory.con …
      h_eq : (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.Memℒp.toLp (fun a => …
      ⊢ MeasureTheory.AEStronglyMeasurable' m (fun a => Inner.inner c (↑↑↑((MeasureT …
    -/
    exact (lpMeas.aeStronglyMeasurable' _).const_inner _
    /-
      🎉 no goals
    -/


/-- `condexpL2` verifies the equality of integrals defining the conditional expectation. -/
theorem integral_condexpL2_eq (hm : m ≤ m0) (f : Lp E' 2 μ) (hs : MeasurableSet[m] s)
    (hμs : μ s ≠ ∞) : ∫ x in s, (condexpL2 E' 𝕜 hm f : α → E') x ∂μ = ∫ x in s, f x ∂μ := by
  rw [← sub_eq_zero, lpMeas_coe, ←
    integral_sub' (integrableOn_Lp_of_measure_ne_top _ fact_one_le_two_ennreal.elim hμs)
      (integrableOn_Lp_of_measure_ne_top _ fact_one_le_two_ennreal.elim hμs)]
  /-
    α : Type u_1
    E' : Type u_3
    𝕜 : Type u_7
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup E'
    inst✝² : InnerProductSpace 𝕜 E'
    inst✝¹ : CompleteSpace E'
    inst✝ : NormedSpace Real E'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E' 2 μ) x
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun a => HSub.hSub (↑↑↑((MeasureTh …
  -/
  refine integral_eq_zero_of_forall_integral_inner_eq_zero 𝕜 _ ?_ ?_
    /-
      case refine_1
      α : Type u_1
      E' : Type u_3
      𝕜 : Type u_7
      inst✝⁴ : RCLike 𝕜
      inst✝³ : NormedAddCommGroup E'
      inst✝² : InnerProductSpace 𝕜 E'
      inst✝¹ : CompleteSpace E'
      inst✝ : NormedSpace Real E'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      hm : LE.le m m0
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E' 2 μ) x
      hs : MeasurableSet s
      hμs : Ne (μ s) Top.top
      ⊢ MeasureTheory.Integrable (HSub.hSub ↑↑↑((MeasureTheory.condexpL2 E' 𝕜 hm) f) …
    -/
  · rw [integrable_congr (ae_restrict_of_ae (Lp.coeFn_sub (↑(condexpL2 E' 𝕜 hm f)) f).symm)]
    /-
      case refine_1
      α : Type u_1
      E' : Type u_3
      𝕜 : Type u_7
      inst✝⁴ : RCLike 𝕜
      inst✝³ : NormedAddCommGroup E'
      inst✝² : InnerProductSpace 𝕜 E'
      inst✝¹ : CompleteSpace E'
      inst✝ : NormedSpace Real E'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      hm : LE.le m m0
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E' 2 μ) x
      hs : MeasurableSet s
      hμs : Ne (μ s) Top.top
      ⊢ MeasureTheory.Integrable (↑↑(HSub.hSub (↑((MeasureTheory.condexpL2 E' 𝕜 hm)  …
    -/
    exact integrableOn_Lp_of_measure_ne_top _ fact_one_le_two_ennreal.elim hμs
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    α : Type u_1
    E' : Type u_3
    𝕜 : Type u_7
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup E'
    inst✝² : InnerProductSpace 𝕜 E'
    inst✝¹ : CompleteSpace E'
    inst✝ : NormedSpace Real E'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E' 2 μ) x
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    ⊢ ∀ (c : E'), Eq (MeasureTheory.integral (μ.restrict s) fun x => Inner.inner c …
  -/
  intro c
  /-
    case refine_2
    α : Type u_1
    E' : Type u_3
    𝕜 : Type u_7
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup E'
    inst✝² : InnerProductSpace 𝕜 E'
    inst✝¹ : CompleteSpace E'
    inst✝ : NormedSpace Real E'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E' 2 μ) x
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : E'
    ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun x => Inner.inner c (HSub.hSub  …
  -/
  simp_rw [Pi.sub_apply, inner_sub_right]
  rw [integral_sub
      ((integrableOn_Lp_of_measure_ne_top _ fact_one_le_two_ennreal.elim hμs).const_inner c)
      ((integrableOn_Lp_of_measure_ne_top _ fact_one_le_two_ennreal.elim hμs).const_inner c)]
  /-
    case refine_2
    α : Type u_1
    E' : Type u_3
    𝕜 : Type u_7
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup E'
    inst✝² : InnerProductSpace 𝕜 E'
    inst✝¹ : CompleteSpace E'
    inst✝ : NormedSpace Real E'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E' 2 μ) x
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : E'
    ⊢ Eq (HSub.hSub (MeasureTheory.integral (μ.restrict s) fun a => Inner.inner c  …
  -/
  have h_ae_eq_f := Memℒp.coeFn_toLp (E := 𝕜) ((Lp.memℒp f).const_inner c)
  rw [← lpMeas_coe, sub_eq_zero, ←
    setIntegral_congr_ae (hm s hs) ((condexpL2_const_inner hm f c).mono fun x hx _ => hx), ←
    setIntegral_congr_ae (hm s hs) (h_ae_eq_f.mono fun x hx _ => hx)]
  /-
    case refine_2
    α : Type u_1
    E' : Type u_3
    𝕜 : Type u_7
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup E'
    inst✝² : InnerProductSpace 𝕜 E'
    inst✝¹ : CompleteSpace E'
    inst✝ : NormedSpace Real E'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E' 2 μ) x
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : E'
    h_ae_eq_f : (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.Memℒp.toLp (fun …
    ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun x => ↑↑↑((MeasureTheory.condex …
  -/
  exact integral_condexpL2_eq_of_fin_meas_real _ hs hμs
  /-
    🎉 no goals
  -/


theorem condexpL2_comp_continuousLinearMap (hm : m ≤ m0) (T : E' →L[ℝ] E'') (f : α →₂[μ] E') :
    (condexpL2 E'' 𝕜' hm (T.compLp f) : α →₂[μ] E'') =ᵐ[μ]
    T.compLp (condexpL2 E' 𝕜 hm f : α →₂[μ] E') := by
  refine Lp.ae_eq_of_forall_setIntegral_eq' 𝕜' hm _ _ two_ne_zero ENNReal.coe_ne_top
    (fun s _ hμs => integrableOn_condexpL2_of_measure_ne_top hm hμs.ne _) (fun s _ hμs =>
      integrableOn_Lp_of_measure_ne_top _ fact_one_le_two_ennreal.elim hμs.ne) ?_ ?_ ?_
    /-
      case refine_1
      α : Type u_1
      E' : Type u_3
      𝕜 : Type u_7
      inst✝⁹ : RCLike 𝕜
      inst✝⁸ : NormedAddCommGroup E'
      inst✝⁷ : InnerProductSpace 𝕜 E'
      inst✝⁶ : CompleteSpace E'
      inst✝⁵ : NormedSpace Real E'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E'' : Type u_8
      𝕜' : Type u_9
      inst✝⁴ : RCLike 𝕜'
      inst✝³ : NormedAddCommGroup E''
      inst✝² : InnerProductSpace 𝕜' E''
      inst✝¹ : CompleteSpace E''
      inst✝ : NormedSpace Real E''
      hm : LE.le m m0
      T : ContinuousLinearMap (RingHom.id Real) E' E''
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E' 2 μ) x
      ⊢ ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → Eq (MeasureTheory.int …
    -/
  · intro s hs hμs
    rw [T.setIntegral_compLp _ (hm s hs),
      T.integral_comp_comm
        (integrableOn_Lp_of_measure_ne_top _ fact_one_le_two_ennreal.elim hμs.ne),
      ← lpMeas_coe, ← lpMeas_coe, integral_condexpL2_eq hm f hs hμs.ne,
      integral_condexpL2_eq hm (T.compLp f) hs hμs.ne, T.setIntegral_compLp _ (hm s hs),
      T.integral_comp_comm
        (integrableOn_Lp_of_measure_ne_top f fact_one_le_two_ennreal.elim hμs.ne)]
    /-
      case refine_2
      α : Type u_1
      E' : Type u_3
      𝕜 : Type u_7
      inst✝⁹ : RCLike 𝕜
      inst✝⁸ : NormedAddCommGroup E'
      inst✝⁷ : InnerProductSpace 𝕜 E'
      inst✝⁶ : CompleteSpace E'
      inst✝⁵ : NormedSpace Real E'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E'' : Type u_8
      𝕜' : Type u_9
      inst✝⁴ : RCLike 𝕜'
      inst✝³ : NormedAddCommGroup E''
      inst✝² : InnerProductSpace 𝕜' E''
      inst✝¹ : CompleteSpace E''
      inst✝ : NormedSpace Real E''
      hm : LE.le m m0
      T : ContinuousLinearMap (RingHom.id Real) E' E''
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E' 2 μ) x
      ⊢ MeasureTheory.AEStronglyMeasurable' m (↑↑↑((MeasureTheory.condexpL2 E'' 𝕜' h …
    -/
  · rw [← lpMeas_coe]; exact lpMeas.aeStronglyMeasurable' _
                       /-
                         🎉 no goals
                       -/
    /-
      case refine_3
      α : Type u_1
      E' : Type u_3
      𝕜 : Type u_7
      inst✝⁹ : RCLike 𝕜
      inst✝⁸ : NormedAddCommGroup E'
      inst✝⁷ : InnerProductSpace 𝕜 E'
      inst✝⁶ : CompleteSpace E'
      inst✝⁵ : NormedSpace Real E'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E'' : Type u_8
      𝕜' : Type u_9
      inst✝⁴ : RCLike 𝕜'
      inst✝³ : NormedAddCommGroup E''
      inst✝² : InnerProductSpace 𝕜' E''
      inst✝¹ : CompleteSpace E''
      inst✝ : NormedSpace Real E''
      hm : LE.le m m0
      T : ContinuousLinearMap (RingHom.id Real) E' E''
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E' 2 μ) x
      ⊢ MeasureTheory.AEStronglyMeasurable' m (↑↑(T.compLp ↑((MeasureTheory.condexpL …
    -/
  · have h_coe := T.coeFn_compLp (condexpL2 E' 𝕜 hm f : α →₂[μ] E')
    /-
      case refine_3
      α : Type u_1
      E' : Type u_3
      𝕜 : Type u_7
      inst✝⁹ : RCLike 𝕜
      inst✝⁸ : NormedAddCommGroup E'
      inst✝⁷ : InnerProductSpace 𝕜 E'
      inst✝⁶ : CompleteSpace E'
      inst✝⁵ : NormedSpace Real E'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E'' : Type u_8
      𝕜' : Type u_9
      inst✝⁴ : RCLike 𝕜'
      inst✝³ : NormedAddCommGroup E''
      inst✝² : InnerProductSpace 𝕜' E''
      inst✝¹ : CompleteSpace E''
      inst✝ : NormedSpace Real E''
      hm : LE.le m m0
      T : ContinuousLinearMap (RingHom.id Real) E' E''
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E' 2 μ) x
      h_coe : Filter.Eventually (fun a => Eq (↑↑(T.compLp ↑((MeasureTheory.condexpL2 …
      ⊢ MeasureTheory.AEStronglyMeasurable' m (↑↑(T.compLp ↑((MeasureTheory.condexpL …
    -/
    rw [← EventuallyEq] at h_coe
    /-
      case refine_3
      α : Type u_1
      E' : Type u_3
      𝕜 : Type u_7
      inst✝⁹ : RCLike 𝕜
      inst✝⁸ : NormedAddCommGroup E'
      inst✝⁷ : InnerProductSpace 𝕜 E'
      inst✝⁶ : CompleteSpace E'
      inst✝⁵ : NormedSpace Real E'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E'' : Type u_8
      𝕜' : Type u_9
      inst✝⁴ : RCLike 𝕜'
      inst✝³ : NormedAddCommGroup E''
      inst✝² : InnerProductSpace 𝕜' E''
      inst✝¹ : CompleteSpace E''
      inst✝ : NormedSpace Real E''
      hm : LE.le m m0
      T : ContinuousLinearMap (RingHom.id Real) E' E''
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E' 2 μ) x
      h_coe : (MeasureTheory.ae μ).EventuallyEq ↑↑(T.compLp ↑((MeasureTheory.condexp …
      ⊢ MeasureTheory.AEStronglyMeasurable' m (↑↑(T.compLp ↑((MeasureTheory.condexpL …
    -/
    refine AEStronglyMeasurable'.congr ?_ h_coe.symm
    /-
      case refine_3
      α : Type u_1
      E' : Type u_3
      𝕜 : Type u_7
      inst✝⁹ : RCLike 𝕜
      inst✝⁸ : NormedAddCommGroup E'
      inst✝⁷ : InnerProductSpace 𝕜 E'
      inst✝⁶ : CompleteSpace E'
      inst✝⁵ : NormedSpace Real E'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      E'' : Type u_8
      𝕜' : Type u_9
      inst✝⁴ : RCLike 𝕜'
      inst✝³ : NormedAddCommGroup E''
      inst✝² : InnerProductSpace 𝕜' E''
      inst✝¹ : CompleteSpace E''
      inst✝ : NormedSpace Real E''
      hm : LE.le m m0
      T : ContinuousLinearMap (RingHom.id Real) E' E''
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp E' 2 μ) x
      h_coe : (MeasureTheory.ae μ).EventuallyEq ↑↑(T.compLp ↑((MeasureTheory.condexp …
      ⊢ MeasureTheory.AEStronglyMeasurable' m (fun a => T (↑↑↑((MeasureTheory.condex …
    -/
    exact (lpMeas.aeStronglyMeasurable' (condexpL2 E' 𝕜 hm f)).continuous_comp T.continuous
    /-
      🎉 no goals
    -/


theorem condexpL2_indicator_ae_eq_smul (hm : m ≤ m0) (hs : MeasurableSet s) (hμs : μ s ≠ ∞)
    (x : E') :
    condexpL2 E' 𝕜 hm (indicatorConstLp 2 hs hμs x) =ᵐ[μ] fun a =>
      (condexpL2 ℝ ℝ hm (indicatorConstLp 2 hs hμs (1 : ℝ)) : α → ℝ) a • x := by
  /-
    α : Type u_1
    E' : Type u_3
    𝕜 : Type u_7
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup E'
    inst✝² : InnerProductSpace 𝕜 E'
    inst✝¹ : CompleteSpace E'
    inst✝ : NormedSpace Real E'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : E'
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑↑((MeasureTheory.condexpL2 E' 𝕜 hm) (Mea …
  -/
  rw [indicatorConstLp_eq_toSpanSingleton_compLp hs hμs x]
  have h_comp :=
    condexpL2_comp_continuousLinearMap ℝ 𝕜 hm (toSpanSingleton ℝ x)
      (indicatorConstLp 2 hs hμs (1 : ℝ))
  /-
    α : Type u_1
    E' : Type u_3
    𝕜 : Type u_7
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup E'
    inst✝² : InnerProductSpace 𝕜 E'
    inst✝¹ : CompleteSpace E'
    inst✝ : NormedSpace Real E'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : E'
    h_comp : (MeasureTheory.ae μ).EventuallyEq ↑↑↑((MeasureTheory.condexpL2 E' 𝕜 h …
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑↑((MeasureTheory.condexpL2 E' 𝕜 hm) ((Co …
  -/
  rw [← lpMeas_coe] at h_comp
  /-
    α : Type u_1
    E' : Type u_3
    𝕜 : Type u_7
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup E'
    inst✝² : InnerProductSpace 𝕜 E'
    inst✝¹ : CompleteSpace E'
    inst✝ : NormedSpace Real E'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : E'
    h_comp : (MeasureTheory.ae μ).EventuallyEq ↑↑↑((MeasureTheory.condexpL2 E' 𝕜 h …
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑↑((MeasureTheory.condexpL2 E' 𝕜 hm) ((Co …
  -/
  refine h_comp.trans ?_
  /-
    α : Type u_1
    E' : Type u_3
    𝕜 : Type u_7
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup E'
    inst✝² : InnerProductSpace 𝕜 E'
    inst✝¹ : CompleteSpace E'
    inst✝ : NormedSpace Real E'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : E'
    h_comp : (MeasureTheory.ae μ).EventuallyEq ↑↑↑((MeasureTheory.condexpL2 E' 𝕜 h …
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑((ContinuousLinearMap.toSpanSingleton Re …
  -/
  exact (toSpanSingleton ℝ x).coeFn_compLp _
  /-
    🎉 no goals
  -/


theorem condexpL2_indicator_eq_toSpanSingleton_comp (hm : m ≤ m0) (hs : MeasurableSet s)
    (hμs : μ s ≠ ∞) (x : E') : (condexpL2 E' 𝕜 hm (indicatorConstLp 2 hs hμs x) : α →₂[μ] E') =
    (toSpanSingleton ℝ x).compLp (condexpL2 ℝ ℝ hm (indicatorConstLp 2 hs hμs 1)) := by
  /-
    α : Type u_1
    E' : Type u_3
    𝕜 : Type u_7
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup E'
    inst✝² : InnerProductSpace 𝕜 E'
    inst✝¹ : CompleteSpace E'
    inst✝ : NormedSpace Real E'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : E'
    ⊢ Eq (↑((MeasureTheory.condexpL2 E' 𝕜 hm) (MeasureTheory.indicatorConstLp 2 hs …
  -/
  ext1
  /-
    case h
    α : Type u_1
    E' : Type u_3
    𝕜 : Type u_7
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup E'
    inst✝² : InnerProductSpace 𝕜 E'
    inst✝¹ : CompleteSpace E'
    inst✝ : NormedSpace Real E'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : E'
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑↑((MeasureTheory.condexpL2 E' 𝕜 hm) (Mea …
  -/
  rw [← lpMeas_coe]
  /-
    case h
    α : Type u_1
    E' : Type u_3
    𝕜 : Type u_7
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup E'
    inst✝² : InnerProductSpace 𝕜 E'
    inst✝¹ : CompleteSpace E'
    inst✝ : NormedSpace Real E'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : E'
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑↑((MeasureTheory.condexpL2 E' 𝕜 hm) (Mea …
  -/
  refine (condexpL2_indicator_ae_eq_smul 𝕜 hm hs hμs x).trans ?_
  have h_comp := (toSpanSingleton ℝ x).coeFn_compLp
    (condexpL2 ℝ ℝ hm (indicatorConstLp 2 hs hμs 1) : α →₂[μ] ℝ)
  /-
    case h
    α : Type u_1
    E' : Type u_3
    𝕜 : Type u_7
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup E'
    inst✝² : InnerProductSpace 𝕜 E'
    inst✝¹ : CompleteSpace E'
    inst✝ : NormedSpace Real E'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : E'
    h_comp : Filter.Eventually (fun a => Eq (↑↑((ContinuousLinearMap.toSpanSinglet …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun a => HSMul.hSMul (↑↑↑((MeasureTheory. …
  -/
  rw [← EventuallyEq] at h_comp
  /-
    case h
    α : Type u_1
    E' : Type u_3
    𝕜 : Type u_7
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup E'
    inst✝² : InnerProductSpace 𝕜 E'
    inst✝¹ : CompleteSpace E'
    inst✝ : NormedSpace Real E'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : E'
    h_comp : (MeasureTheory.ae μ).EventuallyEq ↑↑((ContinuousLinearMap.toSpanSingl …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun a => HSMul.hSMul (↑↑↑((MeasureTheory. …
  -/
  refine EventuallyEq.trans ?_ h_comp.symm
  /-
    case h
    α : Type u_1
    E' : Type u_3
    𝕜 : Type u_7
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup E'
    inst✝² : InnerProductSpace 𝕜 E'
    inst✝¹ : CompleteSpace E'
    inst✝ : NormedSpace Real E'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : E'
    h_comp : (MeasureTheory.ae μ).EventuallyEq ↑↑((ContinuousLinearMap.toSpanSingl …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun a => HSMul.hSMul (↑↑↑((MeasureTheory. …
  -/
  filter_upwards with y using rfl
  /-
    🎉 no goals
  -/


theorem setLIntegral_nnnorm_condexpL2_indicator_le (hm : m ≤ m0) (hs : MeasurableSet s)
    (hμs : μ s ≠ ∞) (x : E') {t : Set α} (ht : MeasurableSet[m] t) (hμt : μ t ≠ ∞) :
    ∫⁻ a in t, ‖(condexpL2 E' 𝕜 hm (indicatorConstLp 2 hs hμs x) : α → E') a‖₊ ∂μ ≤
    μ (s ∩ t) * ‖x‖₊ :=
  calc
    ∫⁻ a in t, ‖(condexpL2 E' 𝕜 hm (indicatorConstLp 2 hs hμs x) : α → E') a‖₊ ∂μ =
        ∫⁻ a in t, ‖(condexpL2 ℝ ℝ hm (indicatorConstLp 2 hs hμs 1) : α → ℝ) a • x‖₊ ∂μ :=
      setLIntegral_congr_fun (hm t ht)
                                                                              /-
                                                                                α : Type u_1
                                                                                E' : Type u_3
                                                                                𝕜 : Type u_7
                                                                                inst✝⁴ : RCLike 𝕜
                                                                                inst✝³ : NormedAddCommGroup E'
                                                                                inst✝² : InnerProductSpace 𝕜 E'
                                                                                inst✝¹ : CompleteSpace E'
                                                                                inst✝ : NormedSpace Real E'
                                                                                m m0 : MeasurableSpace α
                                                                                μ : MeasureTheory.Measure α
                                                                                s : Set α
                                                                                hm : LE.le m m0
                                                                                hs : MeasurableSet s
                                                                                hμs : Ne (μ s) Top.top
                                                                                x : E'
                                                                                t : Set α
                                                                                ht : MeasurableSet t
                                                                                hμt : Ne (μ t) Top.top
                                                                                a : α
                                                                                ha : Eq (↑↑↑((MeasureTheory.condexpL2 E' 𝕜 hm) (MeasureTheory.indicatorConstLp …
                                                                                x✝ : Membership.mem t a
                                                                                ⊢ Eq ↑(NNNorm.nnnorm (↑↑↑((MeasureTheory.condexpL2 E' 𝕜 hm) (MeasureTheory.ind …
                                                                              -/
        ((condexpL2_indicator_ae_eq_smul 𝕜 hm hs hμs x).mono fun a ha _ => by rw [ha])
                                                                              /-
                                                                                🎉 no goals
                                                                              -/
    _ = (∫⁻ a in t, ‖(condexpL2 ℝ ℝ hm (indicatorConstLp 2 hs hμs 1) : α → ℝ) a‖₊ ∂μ) * ‖x‖₊ := by
      /-
        α : Type u_1
        E' : Type u_3
        𝕜 : Type u_7
        inst✝⁴ : RCLike 𝕜
        inst✝³ : NormedAddCommGroup E'
        inst✝² : InnerProductSpace 𝕜 E'
        inst✝¹ : CompleteSpace E'
        inst✝ : NormedSpace Real E'
        m m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        s : Set α
        hm : LE.le m m0
        hs : MeasurableSet s
        hμs : Ne (μ s) Top.top
        x : E'
        t : Set α
        ht : MeasurableSet t
        hμt : Ne (μ t) Top.top
        ⊢ Eq (MeasureTheory.lintegral (μ.restrict t) fun a => ↑(NNNorm.nnnorm (HSMul.h …
      -/
      simp_rw [nnnorm_smul, ENNReal.coe_mul]
      /-
        α : Type u_1
        E' : Type u_3
        𝕜 : Type u_7
        inst✝⁴ : RCLike 𝕜
        inst✝³ : NormedAddCommGroup E'
        inst✝² : InnerProductSpace 𝕜 E'
        inst✝¹ : CompleteSpace E'
        inst✝ : NormedSpace Real E'
        m m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        s : Set α
        hm : LE.le m m0
        hs : MeasurableSet s
        hμs : Ne (μ s) Top.top
        x : E'
        t : Set α
        ht : MeasurableSet t
        hμt : Ne (μ t) Top.top
        ⊢ Eq (MeasureTheory.lintegral (μ.restrict t) fun a => HMul.hMul ↑(NNNorm.nnnor …
      -/
      rw [lintegral_mul_const, lpMeas_coe]
      /-
        case hf
        α : Type u_1
        E' : Type u_3
        𝕜 : Type u_7
        inst✝⁴ : RCLike 𝕜
        inst✝³ : NormedAddCommGroup E'
        inst✝² : InnerProductSpace 𝕜 E'
        inst✝¹ : CompleteSpace E'
        inst✝ : NormedSpace Real E'
        m m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        s : Set α
        hm : LE.le m m0
        hs : MeasurableSet s
        hμs : Ne (μ s) Top.top
        x : E'
        t : Set α
        ht : MeasurableSet t
        hμt : Ne (μ t) Top.top
        ⊢ Measurable fun a => ↑(NNNorm.nnnorm (↑↑↑((MeasureTheory.condexpL2 Real Real  …
      -/
      exact (Lp.stronglyMeasurable _).ennnorm
      /-
        🎉 no goals
      -/
    _ ≤ μ (s ∩ t) * ‖x‖₊ :=
      mul_le_mul_right' (lintegral_nnnorm_condexpL2_indicator_le_real hs hμs ht hμt) _


@[deprecated (since := "2024-06-29")]
alias set_lintegral_nnnorm_condexpL2_indicator_le := setLIntegral_nnnorm_condexpL2_indicator_le


theorem lintegral_nnnorm_condexpL2_indicator_le (hm : m ≤ m0) (hs : MeasurableSet s) (hμs : μ s ≠ ∞)
    (x : E') [SigmaFinite (μ.trim hm)] :
    ∫⁻ a, ‖(condexpL2 E' 𝕜 hm (indicatorConstLp 2 hs hμs x) : α → E') a‖₊ ∂μ ≤ μ s * ‖x‖₊ := by
  /-
    α : Type u_1
    E' : Type u_3
    𝕜 : Type u_7
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E'
    inst✝³ : InnerProductSpace 𝕜 E'
    inst✝² : CompleteSpace E'
    inst✝¹ : NormedSpace Real E'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : E'
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (↑↑↑((MeasureTheor …
  -/
  refine lintegral_le_of_forall_fin_meas_trim_le hm (μ s * ‖x‖₊) fun t ht hμt => ?_
  /-
    α : Type u_1
    E' : Type u_3
    𝕜 : Type u_7
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E'
    inst✝³ : InnerProductSpace 𝕜 E'
    inst✝² : CompleteSpace E'
    inst✝¹ : NormedSpace Real E'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : E'
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    t : Set α
    ht : MeasurableSet t
    hμt : Ne (μ t) Top.top
    ⊢ LE.le (MeasureTheory.lintegral (μ.restrict t) fun x_1 => ↑(NNNorm.nnnorm (↑↑ …
  -/
  refine (setLIntegral_nnnorm_condexpL2_indicator_le hm hs hμs x ht hμt).trans ?_
  /-
    α : Type u_1
    E' : Type u_3
    𝕜 : Type u_7
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E'
    inst✝³ : InnerProductSpace 𝕜 E'
    inst✝² : CompleteSpace E'
    inst✝¹ : NormedSpace Real E'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : E'
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    t : Set α
    ht : MeasurableSet t
    hμt : Ne (μ t) Top.top
    ⊢ LE.le (HMul.hMul (μ (Inter.inter s t)) ↑(NNNorm.nnnorm x)) (HMul.hMul (μ s)  …
  -/
  gcongr
  /-
    case bc.h
    α : Type u_1
    E' : Type u_3
    𝕜 : Type u_7
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E'
    inst✝³ : InnerProductSpace 𝕜 E'
    inst✝² : CompleteSpace E'
    inst✝¹ : NormedSpace Real E'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : E'
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    t : Set α
    ht : MeasurableSet t
    hμt : Ne (μ t) Top.top
    ⊢ HasSubset.Subset (Inter.inter s t) s
  -/
  apply Set.inter_subset_left
  /-
    🎉 no goals
  -/


/-- If the measure `μ.trim hm` is sigma-finite, then the conditional expectation of a measurable set
with finite measure is integrable. -/
theorem integrable_condexpL2_indicator (hm : m ≤ m0) [SigmaFinite (μ.trim hm)]
    (hs : MeasurableSet s) (hμs : μ s ≠ ∞) (x : E') :
    Integrable (ε := E') (condexpL2 E' 𝕜 hm (indicatorConstLp 2 hs hμs x)) μ := by
  refine integrable_of_forall_fin_meas_le' hm (μ s * ‖x‖₊)
    (ENNReal.mul_lt_top hμs.lt_top ENNReal.coe_lt_top) ?_ ?_
    /-
      case refine_1
      α : Type u_1
      E' : Type u_3
      𝕜 : Type u_7
      inst✝⁵ : RCLike 𝕜
      inst✝⁴ : NormedAddCommGroup E'
      inst✝³ : InnerProductSpace 𝕜 E'
      inst✝² : CompleteSpace E'
      inst✝¹ : NormedSpace Real E'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      hs : MeasurableSet s
      hμs : Ne (μ s) Top.top
      x : E'
      ⊢ MeasureTheory.AEStronglyMeasurable (↑↑↑((MeasureTheory.condexpL2 E' 𝕜 hm) (M …
    -/
  · rw [lpMeas_coe]; exact Lp.aestronglyMeasurable _
                     /-
                       🎉 no goals
                     -/
  · refine fun t ht hμt =>
      (setLIntegral_nnnorm_condexpL2_indicator_le hm hs hμs x ht hμt).trans ?_
    /-
      case refine_2
      α : Type u_1
      E' : Type u_3
      𝕜 : Type u_7
      inst✝⁵ : RCLike 𝕜
      inst✝⁴ : NormedAddCommGroup E'
      inst✝³ : InnerProductSpace 𝕜 E'
      inst✝² : CompleteSpace E'
      inst✝¹ : NormedSpace Real E'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      hs : MeasurableSet s
      hμs : Ne (μ s) Top.top
      x : E'
      t : Set α
      ht : MeasurableSet t
      hμt : Ne (μ t) Top.top
      ⊢ LE.le (HMul.hMul (μ (Inter.inter s t)) ↑(NNNorm.nnnorm x)) (HMul.hMul (μ s)  …
    -/
    gcongr
    /-
      case refine_2.bc.h
      α : Type u_1
      E' : Type u_3
      𝕜 : Type u_7
      inst✝⁵ : RCLike 𝕜
      inst✝⁴ : NormedAddCommGroup E'
      inst✝³ : InnerProductSpace 𝕜 E'
      inst✝² : CompleteSpace E'
      inst✝¹ : NormedSpace Real E'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      hs : MeasurableSet s
      hμs : Ne (μ s) Top.top
      x : E'
      t : Set α
      ht : MeasurableSet t
      hμt : Ne (μ t) Top.top
      ⊢ HasSubset.Subset (Inter.inter s t) s
    -/
    apply Set.inter_subset_left
    /-
      🎉 no goals
    -/


/-- Conditional expectation of the indicator of a measurable set with finite measure, in L2. -/
noncomputable def condexpIndSMul (hm : m ≤ m0) (hs : MeasurableSet s) (hμs : μ s ≠ ∞) (x : G) :
    Lp G 2 μ :=
  (toSpanSingleton ℝ x).compLpL 2 μ (condexpL2 ℝ ℝ hm (indicatorConstLp 2 hs hμs (1 : ℝ)))


theorem aeStronglyMeasurable'_condexpIndSMul (hm : m ≤ m0) (hs : MeasurableSet s) (hμs : μ s ≠ ∞)
    (x : G) : AEStronglyMeasurable' m (condexpIndSMul hm hs hμs x) μ := by
  have h : AEStronglyMeasurable' m (condexpL2 ℝ ℝ hm (indicatorConstLp 2 hs hμs 1) : α → ℝ) μ :=
    aeStronglyMeasurable'_condexpL2 _ _
  /-
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝ : NormedSpace Real G
    hm : LE.le m m0
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : G
    h : MeasureTheory.AEStronglyMeasurable' m (↑↑↑((MeasureTheory.condexpL2 Real R …
    ⊢ MeasureTheory.AEStronglyMeasurable' m (↑↑(MeasureTheory.condexpIndSMul hm hs …
  -/
  rw [condexpIndSMul]
  suffices AEStronglyMeasurable' m
      (toSpanSingleton ℝ x ∘ condexpL2 ℝ ℝ hm (indicatorConstLp 2 hs hμs 1)) μ by
    refine AEStronglyMeasurable'.congr this ?_
    refine EventuallyEq.trans ?_ (coeFn_compLpL _ _).symm
    rfl
  /-
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝ : NormedSpace Real G
    hm : LE.le m m0
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : G
    h : MeasureTheory.AEStronglyMeasurable' m (↑↑↑((MeasureTheory.condexpL2 Real R …
    ⊢ MeasureTheory.AEStronglyMeasurable' m (Function.comp ⇑(ContinuousLinearMap.t …
  -/
  exact AEStronglyMeasurable'.continuous_comp (toSpanSingleton ℝ x).continuous h
  /-
    🎉 no goals
  -/


theorem condexpIndSMul_add (hs : MeasurableSet s) (hμs : μ s ≠ ∞) (x y : G) :
    condexpIndSMul hm hs hμs (x + y) = condexpIndSMul hm hs hμs x + condexpIndSMul hm hs hμs y := by
  /-
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝ : NormedSpace Real G
    hm : LE.le m m0
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x y : G
    ⊢ Eq (MeasureTheory.condexpIndSMul hm hs hμs (HAdd.hAdd x y)) (HAdd.hAdd (Meas …
  -/
  simp_rw [condexpIndSMul]; rw [toSpanSingleton_add, add_compLpL, add_apply]
                            /-
                              🎉 no goals
                            -/


theorem condexpIndSMul_smul (hs : MeasurableSet s) (hμs : μ s ≠ ∞) (c : ℝ) (x : G) :
    condexpIndSMul hm hs hμs (c • x) = c • condexpIndSMul hm hs hμs x := by
  /-
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝ : NormedSpace Real G
    hm : LE.le m m0
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : Real
    x : G
    ⊢ Eq (MeasureTheory.condexpIndSMul hm hs hμs (HSMul.hSMul c x)) (HSMul.hSMul c …
  -/
  simp_rw [condexpIndSMul]; rw [toSpanSingleton_smul, smul_compLpL, smul_apply]
                            /-
                              🎉 no goals
                            -/


theorem condexpIndSMul_smul' [NormedSpace ℝ F] [SMulCommClass ℝ 𝕜 F] (hs : MeasurableSet s)
    (hμs : μ s ≠ ∞) (c : 𝕜) (x : F) :
    condexpIndSMul hm hs hμs (c • x) = c • condexpIndSMul hm hs hμs x := by
  rw [condexpIndSMul, condexpIndSMul, toSpanSingleton_smul',
    (toSpanSingleton ℝ x).smul_compLpL c, smul_apply]


theorem condexpIndSMul_ae_eq_smul (hm : m ≤ m0) (hs : MeasurableSet s) (hμs : μ s ≠ ∞) (x : G) :
    condexpIndSMul hm hs hμs x =ᵐ[μ] fun a =>
      (condexpL2 ℝ ℝ hm (indicatorConstLp 2 hs hμs 1) : α → ℝ) a • x :=
  (toSpanSingleton ℝ x).coeFn_compLpL _


theorem setLIntegral_nnnorm_condexpIndSMul_le (hm : m ≤ m0) (hs : MeasurableSet s) (hμs : μ s ≠ ∞)
    (x : G) {t : Set α} (ht : MeasurableSet[m] t) (hμt : μ t ≠ ∞) :
    (∫⁻ a in t, ‖condexpIndSMul hm hs hμs x a‖₊ ∂μ) ≤ μ (s ∩ t) * ‖x‖₊ :=
  calc
    ∫⁻ a in t, ‖condexpIndSMul hm hs hμs x a‖₊ ∂μ =
        ∫⁻ a in t, ‖(condexpL2 ℝ ℝ hm (indicatorConstLp 2 hs hμs 1) : α → ℝ) a • x‖₊ ∂μ :=
      setLIntegral_congr_fun (hm t ht)
                                                                       /-
                                                                         α : Type u_1
                                                                         G : Type u_5
                                                                         inst✝¹ : NormedAddCommGroup G
                                                                         m m0 : MeasurableSpace α
                                                                         μ : MeasureTheory.Measure α
                                                                         s : Set α
                                                                         inst✝ : NormedSpace Real G
                                                                         hm : LE.le m m0
                                                                         hs : MeasurableSet s
                                                                         hμs : Ne (μ s) Top.top
                                                                         x : G
                                                                         t : Set α
                                                                         ht : MeasurableSet t
                                                                         hμt : Ne (μ t) Top.top
                                                                         a : α
                                                                         ha : Eq (↑↑(MeasureTheory.condexpIndSMul hm hs hμs x) a) ((fun a => HSMul.hSMu …
                                                                         x✝ : Membership.mem t a
                                                                         ⊢ Eq ↑(NNNorm.nnnorm (↑↑(MeasureTheory.condexpIndSMul hm hs hμs x) a)) ↑(NNNor …
                                                                       -/
        ((condexpIndSMul_ae_eq_smul hm hs hμs x).mono fun a ha _ => by rw [ha])
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
    _ = (∫⁻ a in t, ‖(condexpL2 ℝ ℝ hm (indicatorConstLp 2 hs hμs 1) : α → ℝ) a‖₊ ∂μ) * ‖x‖₊ := by
      /-
        α : Type u_1
        G : Type u_5
        inst✝¹ : NormedAddCommGroup G
        m m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        s : Set α
        inst✝ : NormedSpace Real G
        hm : LE.le m m0
        hs : MeasurableSet s
        hμs : Ne (μ s) Top.top
        x : G
        t : Set α
        ht : MeasurableSet t
        hμt : Ne (μ t) Top.top
        ⊢ Eq (MeasureTheory.lintegral (μ.restrict t) fun a => ↑(NNNorm.nnnorm (HSMul.h …
      -/
      simp_rw [nnnorm_smul, ENNReal.coe_mul]
      /-
        α : Type u_1
        G : Type u_5
        inst✝¹ : NormedAddCommGroup G
        m m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        s : Set α
        inst✝ : NormedSpace Real G
        hm : LE.le m m0
        hs : MeasurableSet s
        hμs : Ne (μ s) Top.top
        x : G
        t : Set α
        ht : MeasurableSet t
        hμt : Ne (μ t) Top.top
        ⊢ Eq (MeasureTheory.lintegral (μ.restrict t) fun a => HMul.hMul ↑(NNNorm.nnnor …
      -/
      rw [lintegral_mul_const, lpMeas_coe]
      /-
        case hf
        α : Type u_1
        G : Type u_5
        inst✝¹ : NormedAddCommGroup G
        m m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        s : Set α
        inst✝ : NormedSpace Real G
        hm : LE.le m m0
        hs : MeasurableSet s
        hμs : Ne (μ s) Top.top
        x : G
        t : Set α
        ht : MeasurableSet t
        hμt : Ne (μ t) Top.top
        ⊢ Measurable fun a => ↑(NNNorm.nnnorm (↑↑↑((MeasureTheory.condexpL2 Real Real  …
      -/
      exact (Lp.stronglyMeasurable _).ennnorm
      /-
        🎉 no goals
      -/
    _ ≤ μ (s ∩ t) * ‖x‖₊ :=
      mul_le_mul_right' (lintegral_nnnorm_condexpL2_indicator_le_real hs hμs ht hμt) _


@[deprecated (since := "2024-06-29")]
alias set_lintegral_nnnorm_condexpIndSMul_le := setLIntegral_nnnorm_condexpIndSMul_le


theorem lintegral_nnnorm_condexpIndSMul_le (hm : m ≤ m0) (hs : MeasurableSet s) (hμs : μ s ≠ ∞)
    (x : G) [SigmaFinite (μ.trim hm)] : ∫⁻ a, ‖condexpIndSMul hm hs hμs x a‖₊ ∂μ ≤ μ s * ‖x‖₊ := by
  /-
    α : Type u_1
    G : Type u_5
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : G
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (↑↑(MeasureTheory. …
  -/
  refine lintegral_le_of_forall_fin_meas_trim_le hm (μ s * ‖x‖₊) fun t ht hμt => ?_
  /-
    α : Type u_1
    G : Type u_5
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : G
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    t : Set α
    ht : MeasurableSet t
    hμt : Ne (μ t) Top.top
    ⊢ LE.le (MeasureTheory.lintegral (μ.restrict t) fun x_1 => ↑(NNNorm.nnnorm (↑↑ …
  -/
  refine (setLIntegral_nnnorm_condexpIndSMul_le hm hs hμs x ht hμt).trans ?_
  /-
    α : Type u_1
    G : Type u_5
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : G
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    t : Set α
    ht : MeasurableSet t
    hμt : Ne (μ t) Top.top
    ⊢ LE.le (HMul.hMul (μ (Inter.inter s t)) ↑(NNNorm.nnnorm x)) (HMul.hMul (μ s)  …
  -/
  gcongr
  /-
    case bc.h
    α : Type u_1
    G : Type u_5
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : G
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    t : Set α
    ht : MeasurableSet t
    hμt : Ne (μ t) Top.top
    ⊢ HasSubset.Subset (Inter.inter s t) s
  -/
  apply Set.inter_subset_left
  /-
    🎉 no goals
  -/


/-- If the measure `μ.trim hm` is sigma-finite, then the conditional expectation of a measurable set
with finite measure is integrable. -/
theorem integrable_condexpIndSMul (hm : m ≤ m0) [SigmaFinite (μ.trim hm)] (hs : MeasurableSet s)
    (hμs : μ s ≠ ∞) (x : G) : Integrable (condexpIndSMul hm hs hμs x) μ := by
  refine integrable_of_forall_fin_meas_le' hm (μ s * ‖x‖₊)
    (ENNReal.mul_lt_top hμs.lt_top ENNReal.coe_lt_top) ?_ ?_
    /-
      case refine_1
      α : Type u_1
      G : Type u_5
      inst✝² : NormedAddCommGroup G
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      inst✝¹ : NormedSpace Real G
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      hs : MeasurableSet s
      hμs : Ne (μ s) Top.top
      x : G
      ⊢ MeasureTheory.AEStronglyMeasurable (↑↑(MeasureTheory.condexpIndSMul hm hs hμ …
    -/
  · exact Lp.aestronglyMeasurable _
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      G : Type u_5
      inst✝² : NormedAddCommGroup G
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      inst✝¹ : NormedSpace Real G
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      hs : MeasurableSet s
      hμs : Ne (μ s) Top.top
      x : G
      ⊢ ∀ (s_1 : Set α), MeasurableSet s_1 → Ne (μ s_1) Top.top → LE.le (MeasureTheo …
    -/
  · refine fun t ht hμt => (setLIntegral_nnnorm_condexpIndSMul_le hm hs hμs x ht hμt).trans ?_
    /-
      case refine_2
      α : Type u_1
      G : Type u_5
      inst✝² : NormedAddCommGroup G
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      inst✝¹ : NormedSpace Real G
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      hs : MeasurableSet s
      hμs : Ne (μ s) Top.top
      x : G
      t : Set α
      ht : MeasurableSet t
      hμt : Ne (μ t) Top.top
      ⊢ LE.le (HMul.hMul (μ (Inter.inter s t)) ↑(NNNorm.nnnorm x)) (HMul.hMul (μ s)  …
    -/
    gcongr
    /-
      case refine_2.bc.h
      α : Type u_1
      G : Type u_5
      inst✝² : NormedAddCommGroup G
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      inst✝¹ : NormedSpace Real G
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      hs : MeasurableSet s
      hμs : Ne (μ s) Top.top
      x : G
      t : Set α
      ht : MeasurableSet t
      hμt : Ne (μ t) Top.top
      ⊢ HasSubset.Subset (Inter.inter s t) s
    -/
    apply Set.inter_subset_left
    /-
      🎉 no goals
    -/


theorem condexpIndSMul_empty {x : G} : condexpIndSMul hm MeasurableSet.empty
    ((measure_empty (μ := μ)).le.trans_lt ENNReal.coe_lt_top).ne x = 0 := by
  /-
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedSpace Real G
    hm : LE.le m m0
    x : G
    ⊢ Eq (MeasureTheory.condexpIndSMul hm ⋯ ⋯ x) 0
  -/
  rw [condexpIndSMul, indicatorConstLp_empty]
  /-
    α : Type u_1
    G : Type u_5
    inst✝¹ : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : NormedSpace Real G
    hm : LE.le m m0
    x : G
    ⊢ Eq ((ContinuousLinearMap.compLpL 2 μ (ContinuousLinearMap.toSpanSingleton Re …
  -/
  simp only [Submodule.coe_zero, ContinuousLinearMap.map_zero]
  /-
    🎉 no goals
  -/


theorem setIntegral_condexpL2_indicator (hs : MeasurableSet[m] s) (ht : MeasurableSet t)
    (hμs : μ s ≠ ∞) (hμt : μ t ≠ ∞) :
    ∫ x in s, (condexpL2 ℝ ℝ hm (indicatorConstLp 2 ht hμt 1) : α → ℝ) x ∂μ = (μ (t ∩ s)).toReal :=
  calc
    ∫ x in s, (condexpL2 ℝ ℝ hm (indicatorConstLp 2 ht hμt 1) : α → ℝ) x ∂μ =
        ∫ x in s, indicatorConstLp 2 ht hμt (1 : ℝ) x ∂μ :=
      @integral_condexpL2_eq α _ ℝ _ _ _ _ _ _ _ _ _ hm (indicatorConstLp 2 ht hμt (1 : ℝ)) hs hμs
    _ = (μ (t ∩ s)).toReal • (1 : ℝ) := setIntegral_indicatorConstLp (hm s hs) ht hμt 1
                                 /-
                                   α : Type u_1
                                   m m0 : MeasurableSpace α
                                   μ : MeasureTheory.Measure α
                                   s t : Set α
                                   hm : LE.le m m0
                                   hs : MeasurableSet s
                                   ht : MeasurableSet t
                                   hμs : Ne (μ s) Top.top
                                   hμt : Ne (μ t) Top.top
                                   ⊢ Eq (HSMul.hSMul (μ (Inter.inter t s)).toReal 1) (μ (Inter.inter t s)).toReal
                                 -/
    _ = (μ (t ∩ s)).toReal := by rw [smul_eq_mul, mul_one]
                                 /-
                                   🎉 no goals
                                 -/


@[deprecated (since := "2024-04-17")]
alias set_integral_condexpL2_indicator := setIntegral_condexpL2_indicator


theorem setIntegral_condexpIndSMul (hs : MeasurableSet[m] s) (ht : MeasurableSet t)
    (hμs : μ s ≠ ∞) (hμt : μ t ≠ ∞) (x : G') :
    ∫ a in s, (condexpIndSMul hm ht hμt x) a ∂μ = (μ (t ∩ s)).toReal • x :=
  calc
    ∫ a in s, (condexpIndSMul hm ht hμt x) a ∂μ =
        ∫ a in s, (condexpL2 ℝ ℝ hm (indicatorConstLp 2 ht hμt 1) : α → ℝ) a • x ∂μ :=
      setIntegral_congr_ae (hm s hs)
        ((condexpIndSMul_ae_eq_smul hm ht hμt x).mono fun _ hx _ => hx)
    _ = (∫ a in s, (condexpL2 ℝ ℝ hm (indicatorConstLp 2 ht hμt 1) : α → ℝ) a ∂μ) • x :=
      (integral_smul_const _ x)
                                     /-
                                       α : Type u_1
                                       G' : Type u_6
                                       inst✝² : NormedAddCommGroup G'
                                       inst✝¹ : NormedSpace Real G'
                                       inst✝ : CompleteSpace G'
                                       m m0 : MeasurableSpace α
                                       μ : MeasureTheory.Measure α
                                       s t : Set α
                                       hm : LE.le m m0
                                       hs : MeasurableSet s
                                       ht : MeasurableSet t
                                       hμs : Ne (μ s) Top.top
                                       hμt : Ne (μ t) Top.top
                                       x : G'
                                       ⊢ Eq (HSMul.hSMul (MeasureTheory.integral (μ.restrict s) fun a => ↑↑↑((Measure …
                                     -/
    _ = (μ (t ∩ s)).toReal • x := by rw [setIntegral_condexpL2_indicator hs ht hμs hμt]
                                     /-
                                       🎉 no goals
                                     -/


@[deprecated (since := "2024-04-17")]
alias set_integral_condexpIndSMul := setIntegral_condexpIndSMul


theorem condexpL2_indicator_nonneg (hm : m ≤ m0) (hs : MeasurableSet s) (hμs : μ s ≠ ∞)
    [SigmaFinite (μ.trim hm)] : (0 : α → ℝ) ≤ᵐ[μ]
    condexpL2 ℝ ℝ hm (indicatorConstLp 2 hs hμs 1) := by
  have h : AEStronglyMeasurable' m (condexpL2 ℝ ℝ hm (indicatorConstLp 2 hs hμs 1) : α → ℝ) μ :=
    aeStronglyMeasurable'_condexpL2 _ _
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    h : MeasureTheory.AEStronglyMeasurable' m (↑↑↑((MeasureTheory.condexpL2 Real R …
    ⊢ (MeasureTheory.ae μ).EventuallyLE 0 ↑↑↑((MeasureTheory.condexpL2 Real Real h …
  -/
  refine EventuallyLE.trans_eq ?_ h.ae_eq_mk.symm
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    h : MeasureTheory.AEStronglyMeasurable' m (↑↑↑((MeasureTheory.condexpL2 Real R …
    ⊢ (MeasureTheory.ae μ).EventuallyLE 0 (MeasureTheory.AEStronglyMeasurable'.mk  …
  -/
  refine @ae_le_of_ae_le_trim _ _ _ _ _ _ hm (0 : α → ℝ) _ ?_
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    h : MeasureTheory.AEStronglyMeasurable' m (↑↑↑((MeasureTheory.condexpL2 Real R …
    ⊢ (MeasureTheory.ae (μ.trim hm)).EventuallyLE 0 (MeasureTheory.AEStronglyMeasu …
  -/
  refine ae_nonneg_of_forall_setIntegral_nonneg_of_sigmaFinite ?_ ?_
    /-
      case refine_1
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      hm : LE.le m m0
      hs : MeasurableSet s
      hμs : Ne (μ s) Top.top
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      h : MeasureTheory.AEStronglyMeasurable' m (↑↑↑((MeasureTheory.condexpL2 Real R …
      ⊢ ∀ (s_1 : Set α), MeasurableSet s_1 → LT.lt ((μ.trim hm) s_1) Top.top → Measu …
    -/
  · rintro t - -
    /-
      case refine_1
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      hm : LE.le m m0
      hs : MeasurableSet s
      hμs : Ne (μ s) Top.top
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      h : MeasureTheory.AEStronglyMeasurable' m (↑↑↑((MeasureTheory.condexpL2 Real R …
      t : Set α
      ⊢ MeasureTheory.IntegrableOn (MeasureTheory.AEStronglyMeasurable'.mk (↑↑↑((Mea …
    -/
    refine @Integrable.integrableOn _ _ m _ _ _ _ ?_
    /-
      case refine_1
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      hm : LE.le m m0
      hs : MeasurableSet s
      hμs : Ne (μ s) Top.top
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      h : MeasureTheory.AEStronglyMeasurable' m (↑↑↑((MeasureTheory.condexpL2 Real R …
      t : Set α
      ⊢ MeasureTheory.Integrable (MeasureTheory.AEStronglyMeasurable'.mk (↑↑↑((Measu …
    -/
    refine Integrable.trim hm ?_ ?_
      /-
        case refine_1.refine_1
        α : Type u_1
        m m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        s : Set α
        hm : LE.le m m0
        hs : MeasurableSet s
        hμs : Ne (μ s) Top.top
        inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
        h : MeasureTheory.AEStronglyMeasurable' m (↑↑↑((MeasureTheory.condexpL2 Real R …
        t : Set α
        ⊢ MeasureTheory.Integrable (MeasureTheory.AEStronglyMeasurable'.mk (↑↑↑((Measu …
      -/
    · rw [integrable_congr h.ae_eq_mk.symm]
      /-
        case refine_1.refine_1
        α : Type u_1
        m m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        s : Set α
        hm : LE.le m m0
        hs : MeasurableSet s
        hμs : Ne (μ s) Top.top
        inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
        h : MeasureTheory.AEStronglyMeasurable' m (↑↑↑((MeasureTheory.condexpL2 Real R …
        t : Set α
        ⊢ MeasureTheory.Integrable (↑↑↑((MeasureTheory.condexpL2 Real Real hm) (Measur …
      -/
      exact integrable_condexpL2_indicator hm hs hμs _
      /-
        🎉 no goals
      -/
      /-
        case refine_1.refine_2
        α : Type u_1
        m m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        s : Set α
        hm : LE.le m m0
        hs : MeasurableSet s
        hμs : Ne (μ s) Top.top
        inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
        h : MeasureTheory.AEStronglyMeasurable' m (↑↑↑((MeasureTheory.condexpL2 Real R …
        t : Set α
        ⊢ MeasureTheory.StronglyMeasurable (MeasureTheory.AEStronglyMeasurable'.mk (↑↑ …
      -/
    · exact h.stronglyMeasurable_mk
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      hm : LE.le m m0
      hs : MeasurableSet s
      hμs : Ne (μ s) Top.top
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      h : MeasureTheory.AEStronglyMeasurable' m (↑↑↑((MeasureTheory.condexpL2 Real R …
      ⊢ ∀ (s_1 : Set α), MeasurableSet s_1 → LT.lt ((μ.trim hm) s_1) Top.top → LE.le …
    -/
  · intro t ht hμt
    /-
      case refine_2
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      hm : LE.le m m0
      hs : MeasurableSet s
      hμs : Ne (μ s) Top.top
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      h : MeasureTheory.AEStronglyMeasurable' m (↑↑↑((MeasureTheory.condexpL2 Real R …
      t : Set α
      ht : MeasurableSet t
      hμt : LT.lt ((μ.trim hm) t) Top.top
      ⊢ LE.le 0 (MeasureTheory.integral ((μ.trim hm).restrict t) fun x => MeasureThe …
    -/
    rw [← setIntegral_trim hm h.stronglyMeasurable_mk ht]
    have h_ae :
      ∀ᵐ x ∂μ, x ∈ t → h.mk _ x = (condexpL2 ℝ ℝ hm (indicatorConstLp 2 hs hμs 1) : α → ℝ) x := by
      filter_upwards [h.ae_eq_mk] with x hx
      exact fun _ => hx.symm
    rw [setIntegral_congr_ae (hm t ht) h_ae,
      setIntegral_condexpL2_indicator ht hs ((le_trim hm).trans_lt hμt).ne hμs]
    /-
      case refine_2
      α : Type u_1
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      hm : LE.le m m0
      hs : MeasurableSet s
      hμs : Ne (μ s) Top.top
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      h : MeasureTheory.AEStronglyMeasurable' m (↑↑↑((MeasureTheory.condexpL2 Real R …
      t : Set α
      ht : MeasurableSet t
      hμt : LT.lt ((μ.trim hm) t) Top.top
      h_ae : Filter.Eventually (fun x => Membership.mem t x → Eq (MeasureTheory.AESt …
      ⊢ LE.le 0 (μ (Inter.inter s t)).toReal
    -/
    exact ENNReal.toReal_nonneg
    /-
      🎉 no goals
    -/


theorem condexpIndSMul_nonneg {E} [NormedLatticeAddCommGroup E] [NormedSpace ℝ E] [OrderedSMul ℝ E]
    [SigmaFinite (μ.trim hm)] (hs : MeasurableSet s) (hμs : μ s ≠ ∞) (x : E) (hx : 0 ≤ x) :
    (0 : α → E) ≤ᵐ[μ] condexpIndSMul hm hs hμs x := by
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    E : Type u_10
    inst✝³ : NormedLatticeAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : OrderedSMul Real E
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : E
    hx : LE.le 0 x
    ⊢ (MeasureTheory.ae μ).EventuallyLE 0 ↑↑(MeasureTheory.condexpIndSMul hm hs hμ …
  -/
  refine EventuallyLE.trans_eq ?_ (condexpIndSMul_ae_eq_smul hm hs hμs x).symm
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    E : Type u_10
    inst✝³ : NormedLatticeAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : OrderedSMul Real E
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : E
    hx : LE.le 0 x
    ⊢ (MeasureTheory.ae μ).EventuallyLE 0 fun a => HSMul.hSMul (↑↑↑((MeasureTheory …
  -/
  filter_upwards [condexpL2_indicator_nonneg hm hs hμs] with a ha
  /-
    case h
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    E : Type u_10
    inst✝³ : NormedLatticeAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : OrderedSMul Real E
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : E
    hx : LE.le 0 x
    a : α
    ha : LE.le (0 a) (↑↑↑((MeasureTheory.condexpL2 Real Real hm) (MeasureTheory.in …
    ⊢ LE.le (0 a) (HSMul.hSMul (↑↑↑((MeasureTheory.condexpL2 Real Real hm) (Measur …
  -/
  exact smul_nonneg ha hx
  /-
    🎉 no goals
  -/


