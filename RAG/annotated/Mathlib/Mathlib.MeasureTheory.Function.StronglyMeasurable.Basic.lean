local infixr:25 " →ₛ " => SimpleFunc


/-- A function is `StronglyMeasurable` if it is the limit of simple functions. -/
def StronglyMeasurable [MeasurableSpace α] (f : α → β) : Prop :=
  ∃ fs : ℕ → α →ₛ β, ∀ x, Tendsto (fun n => fs n x) atTop (𝓝 (f x))


/-- The notation for StronglyMeasurable giving the measurable space instance explicitly. -/
scoped notation "StronglyMeasurable[" m "]" => @MeasureTheory.StronglyMeasurable _ _ _ m


/-- A function is `FinStronglyMeasurable` with respect to a measure if it is the limit of simple
  functions with support with finite measure. -/
def FinStronglyMeasurable [Zero β]
    {_ : MeasurableSpace α} (f : α → β) (μ : Measure α := by volume_tac) : Prop :=
  ∃ fs : ℕ → α →ₛ β, (∀ n, μ (support (fs n)) < ∞) ∧ ∀ x, Tendsto (fun n => fs n x) atTop (𝓝 (f x))


/-- A function is `AEStronglyMeasurable` with respect to a measure `μ` if it is almost everywhere
equal to the limit of a sequence of simple functions. -/
@[fun_prop]
def AEStronglyMeasurable
    {_ : MeasurableSpace α} (f : α → β) (μ : Measure α := by volume_tac) : Prop :=
  ∃ g, StronglyMeasurable g ∧ f =ᵐ[μ] g


/-- A function is `AEFinStronglyMeasurable` with respect to a measure if it is almost everywhere
equal to the limit of a sequence of simple functions with support with finite measure. -/
def AEFinStronglyMeasurable
    [Zero β] {_ : MeasurableSpace α} (f : α → β) (μ : Measure α := by volume_tac) : Prop :=
  ∃ g, FinStronglyMeasurable g μ ∧ f =ᵐ[μ] g


@[aesop 30% apply (rule_sets := [Measurable])]
protected theorem StronglyMeasurable.aestronglyMeasurable {α β} {_ : MeasurableSpace α}
    [TopologicalSpace β] {f : α → β} {μ : Measure α} (hf : StronglyMeasurable f) :
    AEStronglyMeasurable f μ :=
  ⟨f, hf, EventuallyEq.refl _ _⟩


@[simp]
theorem Subsingleton.stronglyMeasurable {α β} [MeasurableSpace α] [TopologicalSpace β]
    [Subsingleton β] (f : α → β) : StronglyMeasurable f := by
  /-
    α : Type u_5
    β : Type u_6
    inst✝² : MeasurableSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : Subsingleton β
    f : α → β
    ⊢ MeasureTheory.StronglyMeasurable f
  -/
  let f_sf : α →ₛ β := ⟨f, fun x => ?_, Set.Subsingleton.finite Set.subsingleton_of_subsingleton⟩
    /-
      case refine_2
      α : Type u_5
      β : Type u_6
      inst✝² : MeasurableSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : Subsingleton β
      f : α → β
      f_sf : MeasureTheory.SimpleFunc α β := { toFun := f, measurableSet_fiber' := ⋯ …
      ⊢ MeasureTheory.StronglyMeasurable f
    -/
  · exact ⟨fun _ => f_sf, fun x => tendsto_const_nhds⟩
    /-
      🎉 no goals
    -/
  · have h_univ : f ⁻¹' {x} = Set.univ := by
      ext1 y
      simp [eq_iff_true_of_subsingleton]
    /-
      case refine_1
      α : Type u_5
      β : Type u_6
      inst✝² : MeasurableSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : Subsingleton β
      f : α → β
      x : β
      h_univ : Eq (Set.preimage f (Singleton.singleton x)) Set.univ
      ⊢ MeasurableSet (Set.preimage f (Singleton.singleton x))
    -/
    rw [h_univ]
    /-
      case refine_1
      α : Type u_5
      β : Type u_6
      inst✝² : MeasurableSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : Subsingleton β
      f : α → β
      x : β
      h_univ : Eq (Set.preimage f (Singleton.singleton x)) Set.univ
      ⊢ MeasurableSet Set.univ
    -/
    exact MeasurableSet.univ
    /-
      🎉 no goals
    -/


theorem SimpleFunc.stronglyMeasurable {α β} {_ : MeasurableSpace α} [TopologicalSpace β]
    (f : α →ₛ β) : StronglyMeasurable f :=
  ⟨fun _ => f, fun _ => tendsto_const_nhds⟩


@[nontriviality]
theorem StronglyMeasurable.of_finite [Finite α] {_ : MeasurableSpace α}
    [MeasurableSingletonClass α] [TopologicalSpace β]
    {f : α → β} : StronglyMeasurable f :=
  ⟨fun _ => SimpleFunc.ofFinite f, fun _ => tendsto_const_nhds⟩


@[deprecated (since := "2024-02-05")]
alias stronglyMeasurable_of_fintype := StronglyMeasurable.of_finite


@[deprecated StronglyMeasurable.of_finite (since := "2024-02-06")]
theorem stronglyMeasurable_of_isEmpty [IsEmpty α] {_ : MeasurableSpace α} [TopologicalSpace β]
    (f : α → β) : StronglyMeasurable f :=
  .of_finite


theorem stronglyMeasurable_const {α β} {_ : MeasurableSpace α} [TopologicalSpace β] {b : β} :
    StronglyMeasurable fun _ : α => b :=
  ⟨fun _ => SimpleFunc.const α b, fun _ => tendsto_const_nhds⟩


@[to_additive]
theorem stronglyMeasurable_one {α β} {_ : MeasurableSpace α} [TopologicalSpace β] [One β] :
    StronglyMeasurable (1 : α → β) :=
  stronglyMeasurable_const


/-- A version of `stronglyMeasurable_const` that assumes `f x = f y` for all `x, y`.
This version works for functions between empty types. -/
theorem stronglyMeasurable_const' {α β} {m : MeasurableSpace α} [TopologicalSpace β] {f : α → β}
    (hf : ∀ x y, f x = f y) : StronglyMeasurable f := by
  /-
    α : Type u_5
    β : Type u_6
    m : MeasurableSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    hf : ∀ (x y : α), Eq (f x) (f y)
    ⊢ MeasureTheory.StronglyMeasurable f
  -/
  nontriviality α
  /-
    α : Type u_5
    β : Type u_6
    m : MeasurableSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    hf : ∀ (x y : α), Eq (f x) (f y)
    a✝ : Nontrivial α
    ⊢ MeasureTheory.StronglyMeasurable f
  -/
  inhabit α
  /-
    α : Type u_5
    β : Type u_6
    m : MeasurableSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    hf : ∀ (x y : α), Eq (f x) (f y)
    a✝ : Nontrivial α
    inhabited_h : Inhabited α
    ⊢ MeasureTheory.StronglyMeasurable f
  -/
  convert stronglyMeasurable_const (β := β) using 1
  /-
    case h.e'_5
    α : Type u_5
    β : Type u_6
    m : MeasurableSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    hf : ∀ (x y : α), Eq (f x) (f y)
    a✝ : Nontrivial α
    inhabited_h : Inhabited α
    ⊢ Eq f fun x => ?convert_3
  -/
  exact funext fun x => hf x default
  /-
    🎉 no goals
  -/

-- Porting note: changed binding type of `MeasurableSpace α`.

@[simp]
theorem Subsingleton.stronglyMeasurable' {α β} [MeasurableSpace α] [TopologicalSpace β]
    [Subsingleton α] (f : α → β) : StronglyMeasurable f :=
                                          /-
                                            α : Type u_5
                                            β : Type u_6
                                            inst✝² : MeasurableSpace α
                                            inst✝¹ : TopologicalSpace β
                                            inst✝ : Subsingleton α
                                            f : α → β
                                            x y : α
                                            ⊢ Eq (f x) (f y)
                                          -/
  stronglyMeasurable_const' fun x y => by rw [Subsingleton.elim x y]
                                          /-
                                            🎉 no goals
                                          -/


/-- A sequence of simple functions such that
`∀ x, Tendsto (fun n => hf.approx n x) atTop (𝓝 (f x))`.
That property is given by `stronglyMeasurable.tendsto_approx`. -/
protected noncomputable def approx {_ : MeasurableSpace α} (hf : StronglyMeasurable f) :
    ℕ → α →ₛ β :=
  hf.choose


protected theorem tendsto_approx {_ : MeasurableSpace α} (hf : StronglyMeasurable f) :
    ∀ x, Tendsto (fun n => hf.approx n x) atTop (𝓝 (f x)) :=
  hf.choose_spec


/-- Similar to `stronglyMeasurable.approx`, but enforces that the norm of every function in the
sequence is less than `c` everywhere. If `‖f x‖ ≤ c` this sequence of simple functions verifies
`Tendsto (fun n => hf.approxBounded n x) atTop (𝓝 (f x))`. -/
noncomputable def approxBounded {_ : MeasurableSpace α} [Norm β] [SMul ℝ β]
    (hf : StronglyMeasurable f) (c : ℝ) : ℕ → SimpleFunc α β := fun n =>
  (hf.approx n).map fun x => min 1 (c / ‖x‖) • x


theorem tendsto_approxBounded_of_norm_le {β} {f : α → β} [NormedAddCommGroup β] [NormedSpace ℝ β]
    {m : MeasurableSpace α} (hf : StronglyMeasurable[m] f) {c : ℝ} {x : α} (hfx : ‖f x‖ ≤ c) :
    Tendsto (fun n => hf.approxBounded c n x) atTop (𝓝 (f x)) := by
  /-
    α : Type u_1
    β : Type u_5
    f : α → β
    inst✝¹ : NormedAddCommGroup β
    inst✝ : NormedSpace Real β
    m : MeasurableSpace α
    hf : MeasureTheory.StronglyMeasurable f
    c : Real
    x : α
    hfx : LE.le (Norm.norm (f x)) c
    ⊢ Filter.Tendsto (fun n => (hf.approxBounded c n) x) Filter.atTop (nhds (f x))
  -/
  have h_tendsto := hf.tendsto_approx x
  /-
    α : Type u_1
    β : Type u_5
    f : α → β
    inst✝¹ : NormedAddCommGroup β
    inst✝ : NormedSpace Real β
    m : MeasurableSpace α
    hf : MeasureTheory.StronglyMeasurable f
    c : Real
    x : α
    hfx : LE.le (Norm.norm (f x)) c
    h_tendsto : Filter.Tendsto (fun n => (hf.approx n) x) Filter.atTop (nhds (f x))
    ⊢ Filter.Tendsto (fun n => (hf.approxBounded c n) x) Filter.atTop (nhds (f x))
  -/
  simp only [StronglyMeasurable.approxBounded, SimpleFunc.coe_map, Function.comp_apply]
  /-
    α : Type u_1
    β : Type u_5
    f : α → β
    inst✝¹ : NormedAddCommGroup β
    inst✝ : NormedSpace Real β
    m : MeasurableSpace α
    hf : MeasureTheory.StronglyMeasurable f
    c : Real
    x : α
    hfx : LE.le (Norm.norm (f x)) c
    h_tendsto : Filter.Tendsto (fun n => (hf.approx n) x) Filter.atTop (nhds (f x))
    ⊢ Filter.Tendsto (fun n => HSMul.hSMul (Min.min 1 (HDiv.hDiv c (Norm.norm ((hf …
  -/
  by_cases hfx0 : ‖f x‖ = 0
    /-
      case pos
      α : Type u_1
      β : Type u_5
      f : α → β
      inst✝¹ : NormedAddCommGroup β
      inst✝ : NormedSpace Real β
      m : MeasurableSpace α
      hf : MeasureTheory.StronglyMeasurable f
      c : Real
      x : α
      hfx : LE.le (Norm.norm (f x)) c
      h_tendsto : Filter.Tendsto (fun n => (hf.approx n) x) Filter.atTop (nhds (f x))
      hfx0 : Eq (Norm.norm (f x)) 0
      ⊢ Filter.Tendsto (fun n => HSMul.hSMul (Min.min 1 (HDiv.hDiv c (Norm.norm ((hf …
    -/
  · rw [norm_eq_zero] at hfx0
    /-
      case pos
      α : Type u_1
      β : Type u_5
      f : α → β
      inst✝¹ : NormedAddCommGroup β
      inst✝ : NormedSpace Real β
      m : MeasurableSpace α
      hf : MeasureTheory.StronglyMeasurable f
      c : Real
      x : α
      hfx : LE.le (Norm.norm (f x)) c
      h_tendsto : Filter.Tendsto (fun n => (hf.approx n) x) Filter.atTop (nhds (f x))
      hfx0 : Eq (f x) 0
      ⊢ Filter.Tendsto (fun n => HSMul.hSMul (Min.min 1 (HDiv.hDiv c (Norm.norm ((hf …
    -/
    rw [hfx0] at h_tendsto ⊢
    have h_tendsto_norm : Tendsto (fun n => ‖hf.approx n x‖) atTop (𝓝 0) := by
      convert h_tendsto.norm
      rw [norm_zero]
    /-
      case pos
      α : Type u_1
      β : Type u_5
      f : α → β
      inst✝¹ : NormedAddCommGroup β
      inst✝ : NormedSpace Real β
      m : MeasurableSpace α
      hf : MeasureTheory.StronglyMeasurable f
      c : Real
      x : α
      hfx : LE.le (Norm.norm (f x)) c
      h_tendsto : Filter.Tendsto (fun n => (hf.approx n) x) Filter.atTop (nhds 0)
      hfx0 : Eq (f x) 0
      h_tendsto_norm : Filter.Tendsto (fun n => Norm.norm ((hf.approx n) x)) Filter. …
      ⊢ Filter.Tendsto (fun n => HSMul.hSMul (Min.min 1 (HDiv.hDiv c (Norm.norm ((hf …
    -/
    refine squeeze_zero_norm (fun n => ?_) h_tendsto_norm
    calc
      ‖min 1 (c / ‖hf.approx n x‖) • hf.approx n x‖ =
          ‖min 1 (c / ‖hf.approx n x‖)‖ * ‖hf.approx n x‖ :=
        norm_smul _ _
      _ ≤ ‖(1 : ℝ)‖ * ‖hf.approx n x‖ := by
        refine mul_le_mul_of_nonneg_right ?_ (norm_nonneg _)
        rw [norm_one, Real.norm_of_nonneg]
        · exact min_le_left _ _
        · exact le_min zero_le_one (div_nonneg ((norm_nonneg _).trans hfx) (norm_nonneg _))
      _ = ‖hf.approx n x‖ := by rw [norm_one, one_mul]
  /-
    case neg
    α : Type u_1
    β : Type u_5
    f : α → β
    inst✝¹ : NormedAddCommGroup β
    inst✝ : NormedSpace Real β
    m : MeasurableSpace α
    hf : MeasureTheory.StronglyMeasurable f
    c : Real
    x : α
    hfx : LE.le (Norm.norm (f x)) c
    h_tendsto : Filter.Tendsto (fun n => (hf.approx n) x) Filter.atTop (nhds (f x))
    hfx0 : Not (Eq (Norm.norm (f x)) 0)
    ⊢ Filter.Tendsto (fun n => HSMul.hSMul (Min.min 1 (HDiv.hDiv c (Norm.norm ((hf …
  -/
  rw [← one_smul ℝ (f x)]
  /-
    case neg
    α : Type u_1
    β : Type u_5
    f : α → β
    inst✝¹ : NormedAddCommGroup β
    inst✝ : NormedSpace Real β
    m : MeasurableSpace α
    hf : MeasureTheory.StronglyMeasurable f
    c : Real
    x : α
    hfx : LE.le (Norm.norm (f x)) c
    h_tendsto : Filter.Tendsto (fun n => (hf.approx n) x) Filter.atTop (nhds (f x))
    hfx0 : Not (Eq (Norm.norm (f x)) 0)
    ⊢ Filter.Tendsto (fun n => HSMul.hSMul (Min.min 1 (HDiv.hDiv c (Norm.norm ((hf …
  -/
  refine Tendsto.smul ?_ h_tendsto
  have : min 1 (c / ‖f x‖) = 1 := by
    rw [min_eq_left_iff, one_le_div (lt_of_le_of_ne (norm_nonneg _) (Ne.symm hfx0))]
    exact hfx
  /-
    case neg
    α : Type u_1
    β : Type u_5
    f : α → β
    inst✝¹ : NormedAddCommGroup β
    inst✝ : NormedSpace Real β
    m : MeasurableSpace α
    hf : MeasureTheory.StronglyMeasurable f
    c : Real
    x : α
    hfx : LE.le (Norm.norm (f x)) c
    h_tendsto : Filter.Tendsto (fun n => (hf.approx n) x) Filter.atTop (nhds (f x))
    hfx0 : Not (Eq (Norm.norm (f x)) 0)
    this : Eq (Min.min 1 (HDiv.hDiv c (Norm.norm (f x)))) 1
    ⊢ Filter.Tendsto (fun n => Min.min 1 (HDiv.hDiv c (Norm.norm ((hf.approx n) x) …
  -/
  nth_rw 2 [this.symm]
  /-
    case neg
    α : Type u_1
    β : Type u_5
    f : α → β
    inst✝¹ : NormedAddCommGroup β
    inst✝ : NormedSpace Real β
    m : MeasurableSpace α
    hf : MeasureTheory.StronglyMeasurable f
    c : Real
    x : α
    hfx : LE.le (Norm.norm (f x)) c
    h_tendsto : Filter.Tendsto (fun n => (hf.approx n) x) Filter.atTop (nhds (f x))
    hfx0 : Not (Eq (Norm.norm (f x)) 0)
    this : Eq (Min.min 1 (HDiv.hDiv c (Norm.norm (f x)))) 1
    ⊢ Filter.Tendsto (fun n => Min.min 1 (HDiv.hDiv c (Norm.norm ((hf.approx n) x) …
  -/
  refine Tendsto.min tendsto_const_nhds ?_
  /-
    case neg
    α : Type u_1
    β : Type u_5
    f : α → β
    inst✝¹ : NormedAddCommGroup β
    inst✝ : NormedSpace Real β
    m : MeasurableSpace α
    hf : MeasureTheory.StronglyMeasurable f
    c : Real
    x : α
    hfx : LE.le (Norm.norm (f x)) c
    h_tendsto : Filter.Tendsto (fun n => (hf.approx n) x) Filter.atTop (nhds (f x))
    hfx0 : Not (Eq (Norm.norm (f x)) 0)
    this : Eq (Min.min 1 (HDiv.hDiv c (Norm.norm (f x)))) 1
    ⊢ Filter.Tendsto (fun n => HDiv.hDiv c (Norm.norm ((hf.approx n) x))) Filter.a …
  -/
  exact Tendsto.div tendsto_const_nhds h_tendsto.norm hfx0
  /-
    🎉 no goals
  -/


theorem tendsto_approxBounded_ae {β} {f : α → β} [NormedAddCommGroup β] [NormedSpace ℝ β]
    {m m0 : MeasurableSpace α} {μ : Measure α} (hf : StronglyMeasurable[m] f) {c : ℝ}
    (hf_bound : ∀ᵐ x ∂μ, ‖f x‖ ≤ c) :
    ∀ᵐ x ∂μ, Tendsto (fun n => hf.approxBounded c n x) atTop (𝓝 (f x)) := by
  /-
    α : Type u_1
    β : Type u_5
    f : α → β
    inst✝¹ : NormedAddCommGroup β
    inst✝ : NormedSpace Real β
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hf : MeasureTheory.StronglyMeasurable f
    c : Real
    hf_bound : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) c) (MeasureTheo …
    ⊢ Filter.Eventually (fun x => Filter.Tendsto (fun n => (hf.approxBounded c n)  …
  -/
  filter_upwards [hf_bound] with x hfx using tendsto_approxBounded_of_norm_le hf hfx
  /-
    🎉 no goals
  -/


theorem norm_approxBounded_le {β} {f : α → β} [SeminormedAddCommGroup β] [NormedSpace ℝ β]
    {m : MeasurableSpace α} {c : ℝ} (hf : StronglyMeasurable[m] f) (hc : 0 ≤ c) (n : ℕ) (x : α) :
    ‖hf.approxBounded c n x‖ ≤ c := by
  /-
    α : Type u_1
    β : Type u_5
    f : α → β
    inst✝¹ : SeminormedAddCommGroup β
    inst✝ : NormedSpace Real β
    m : MeasurableSpace α
    c : Real
    hf : MeasureTheory.StronglyMeasurable f
    hc : LE.le 0 c
    n : Nat
    x : α
    ⊢ LE.le (Norm.norm ((hf.approxBounded c n) x)) c
  -/
  simp only [StronglyMeasurable.approxBounded, SimpleFunc.coe_map, Function.comp_apply]
  /-
    α : Type u_1
    β : Type u_5
    f : α → β
    inst✝¹ : SeminormedAddCommGroup β
    inst✝ : NormedSpace Real β
    m : MeasurableSpace α
    c : Real
    hf : MeasureTheory.StronglyMeasurable f
    hc : LE.le 0 c
    n : Nat
    x : α
    ⊢ LE.le (Norm.norm (HSMul.hSMul (Min.min 1 (HDiv.hDiv c (Norm.norm ((hf.approx …
  -/
  refine (norm_smul_le _ _).trans ?_
  /-
    α : Type u_1
    β : Type u_5
    f : α → β
    inst✝¹ : SeminormedAddCommGroup β
    inst✝ : NormedSpace Real β
    m : MeasurableSpace α
    c : Real
    hf : MeasureTheory.StronglyMeasurable f
    hc : LE.le 0 c
    n : Nat
    x : α
    ⊢ LE.le (HMul.hMul (Norm.norm (Min.min 1 (HDiv.hDiv c (Norm.norm ((hf.approx n …
  -/
  by_cases h0 : ‖hf.approx n x‖ = 0
    /-
      case pos
      α : Type u_1
      β : Type u_5
      f : α → β
      inst✝¹ : SeminormedAddCommGroup β
      inst✝ : NormedSpace Real β
      m : MeasurableSpace α
      c : Real
      hf : MeasureTheory.StronglyMeasurable f
      hc : LE.le 0 c
      n : Nat
      x : α
      h0 : Eq (Norm.norm ((hf.approx n) x)) 0
      ⊢ LE.le (HMul.hMul (Norm.norm (Min.min 1 (HDiv.hDiv c (Norm.norm ((hf.approx n …
    -/
  · simp only [h0, _root_.div_zero, min_eq_right, zero_le_one, norm_zero, mul_zero]
    /-
      case pos
      α : Type u_1
      β : Type u_5
      f : α → β
      inst✝¹ : SeminormedAddCommGroup β
      inst✝ : NormedSpace Real β
      m : MeasurableSpace α
      c : Real
      hf : MeasureTheory.StronglyMeasurable f
      hc : LE.le 0 c
      n : Nat
      x : α
      h0 : Eq (Norm.norm ((hf.approx n) x)) 0
      ⊢ LE.le 0 c
    -/
    exact hc
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    β : Type u_5
    f : α → β
    inst✝¹ : SeminormedAddCommGroup β
    inst✝ : NormedSpace Real β
    m : MeasurableSpace α
    c : Real
    hf : MeasureTheory.StronglyMeasurable f
    hc : LE.le 0 c
    n : Nat
    x : α
    h0 : Not (Eq (Norm.norm ((hf.approx n) x)) 0)
    ⊢ LE.le (HMul.hMul (Norm.norm (Min.min 1 (HDiv.hDiv c (Norm.norm ((hf.approx n …
  -/
  rcases le_total ‖hf.approx n x‖ c with h | h
    /-
      case neg.inl
      α : Type u_1
      β : Type u_5
      f : α → β
      inst✝¹ : SeminormedAddCommGroup β
      inst✝ : NormedSpace Real β
      m : MeasurableSpace α
      c : Real
      hf : MeasureTheory.StronglyMeasurable f
      hc : LE.le 0 c
      n : Nat
      x : α
      h0 : Not (Eq (Norm.norm ((hf.approx n) x)) 0)
      h : LE.le (Norm.norm ((hf.approx n) x)) c
      ⊢ LE.le (HMul.hMul (Norm.norm (Min.min 1 (HDiv.hDiv c (Norm.norm ((hf.approx n …
    -/
  · rw [min_eq_left _]
      /-
        case neg.inl
        α : Type u_1
        β : Type u_5
        f : α → β
        inst✝¹ : SeminormedAddCommGroup β
        inst✝ : NormedSpace Real β
        m : MeasurableSpace α
        c : Real
        hf : MeasureTheory.StronglyMeasurable f
        hc : LE.le 0 c
        n : Nat
        x : α
        h0 : Not (Eq (Norm.norm ((hf.approx n) x)) 0)
        h : LE.le (Norm.norm ((hf.approx n) x)) c
        ⊢ LE.le (HMul.hMul (Norm.norm 1) (Norm.norm ((hf.approx n) x))) c
      -/
    · simpa only [norm_one, one_mul] using h
      /-
        🎉 no goals
      -/
      /-
        α : Type u_1
        β : Type u_5
        f : α → β
        inst✝¹ : SeminormedAddCommGroup β
        inst✝ : NormedSpace Real β
        m : MeasurableSpace α
        c : Real
        hf : MeasureTheory.StronglyMeasurable f
        hc : LE.le 0 c
        n : Nat
        x : α
        h0 : Not (Eq (Norm.norm ((hf.approx n) x)) 0)
        h : LE.le (Norm.norm ((hf.approx n) x)) c
        ⊢ LE.le 1 (HDiv.hDiv c (Norm.norm ((hf.approx n) x)))
      -/
    · rwa [one_le_div (lt_of_le_of_ne (norm_nonneg _) (Ne.symm h0))]
      /-
        🎉 no goals
      -/
    /-
      case neg.inr
      α : Type u_1
      β : Type u_5
      f : α → β
      inst✝¹ : SeminormedAddCommGroup β
      inst✝ : NormedSpace Real β
      m : MeasurableSpace α
      c : Real
      hf : MeasureTheory.StronglyMeasurable f
      hc : LE.le 0 c
      n : Nat
      x : α
      h0 : Not (Eq (Norm.norm ((hf.approx n) x)) 0)
      h : LE.le c (Norm.norm ((hf.approx n) x))
      ⊢ LE.le (HMul.hMul (Norm.norm (Min.min 1 (HDiv.hDiv c (Norm.norm ((hf.approx n …
    -/
  · rw [min_eq_right _]
    · rw [norm_div, norm_norm, mul_comm, mul_div, div_eq_mul_inv, mul_comm, ← mul_assoc,
        inv_mul_cancel₀ h0, one_mul, Real.norm_of_nonneg hc]
      /-
        α : Type u_1
        β : Type u_5
        f : α → β
        inst✝¹ : SeminormedAddCommGroup β
        inst✝ : NormedSpace Real β
        m : MeasurableSpace α
        c : Real
        hf : MeasureTheory.StronglyMeasurable f
        hc : LE.le 0 c
        n : Nat
        x : α
        h0 : Not (Eq (Norm.norm ((hf.approx n) x)) 0)
        h : LE.le c (Norm.norm ((hf.approx n) x))
        ⊢ LE.le (HDiv.hDiv c (Norm.norm ((hf.approx n) x))) 1
      -/
    · rwa [div_le_one (lt_of_le_of_ne (norm_nonneg _) (Ne.symm h0))]
      /-
        🎉 no goals
      -/


theorem _root_.stronglyMeasurable_bot_iff [Nonempty β] [T2Space β] :
    StronglyMeasurable[⊥] f ↔ ∃ c, f = fun _ => c := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    inst✝² : TopologicalSpace β
    inst✝¹ : Nonempty β
    inst✝ : T2Space β
    ⊢ Iff (MeasureTheory.StronglyMeasurable f) (Exists fun c => Eq f fun x => c)
  -/
  cases' isEmpty_or_nonempty α with hα hα
  · simp only [@Subsingleton.stronglyMeasurable' _ _ ⊥ _ _ f,
      eq_iff_true_of_subsingleton, exists_const]
  /-
    case inr
    α : Type u_1
    β : Type u_2
    f : α → β
    inst✝² : TopologicalSpace β
    inst✝¹ : Nonempty β
    inst✝ : T2Space β
    hα : Nonempty α
    ⊢ Iff (MeasureTheory.StronglyMeasurable f) (Exists fun c => Eq f fun x => c)
  -/
  refine ⟨fun hf => ?_, fun hf_eq => ?_⟩
    /-
      case inr.refine_1
      α : Type u_1
      β : Type u_2
      f : α → β
      inst✝² : TopologicalSpace β
      inst✝¹ : Nonempty β
      inst✝ : T2Space β
      hα : Nonempty α
      hf : MeasureTheory.StronglyMeasurable f
      ⊢ Exists fun c => Eq f fun x => c
    -/
  · refine ⟨f hα.some, ?_⟩
    /-
      case inr.refine_1
      α : Type u_1
      β : Type u_2
      f : α → β
      inst✝² : TopologicalSpace β
      inst✝¹ : Nonempty β
      inst✝ : T2Space β
      hα : Nonempty α
      hf : MeasureTheory.StronglyMeasurable f
      ⊢ Eq f fun x => f hα.some
    -/
    let fs := hf.approx
    /-
      case inr.refine_1
      α : Type u_1
      β : Type u_2
      f : α → β
      inst✝² : TopologicalSpace β
      inst✝¹ : Nonempty β
      inst✝ : T2Space β
      hα : Nonempty α
      hf : MeasureTheory.StronglyMeasurable f
      fs : Nat → MeasureTheory.SimpleFunc α β := hf.approx
      ⊢ Eq f fun x => f hα.some
    -/
    have h_fs_tendsto : ∀ x, Tendsto (fun n => fs n x) atTop (𝓝 (f x)) := hf.tendsto_approx
    /-
      case inr.refine_1
      α : Type u_1
      β : Type u_2
      f : α → β
      inst✝² : TopologicalSpace β
      inst✝¹ : Nonempty β
      inst✝ : T2Space β
      hα : Nonempty α
      hf : MeasureTheory.StronglyMeasurable f
      fs : Nat → MeasureTheory.SimpleFunc α β := hf.approx
      h_fs_tendsto : ∀ (x : α), Filter.Tendsto (fun n => (fs n) x) Filter.atTop (nhd …
      ⊢ Eq f fun x => f hα.some
    -/
    have : ∀ n, ∃ c, ∀ x, fs n x = c := fun n => SimpleFunc.simpleFunc_bot (fs n)
    /-
      case inr.refine_1
      α : Type u_1
      β : Type u_2
      f : α → β
      inst✝² : TopologicalSpace β
      inst✝¹ : Nonempty β
      inst✝ : T2Space β
      hα : Nonempty α
      hf : MeasureTheory.StronglyMeasurable f
      fs : Nat → MeasureTheory.SimpleFunc α β := hf.approx
      h_fs_tendsto : ∀ (x : α), Filter.Tendsto (fun n => (fs n) x) Filter.atTop (nhd …
      this : ∀ (n : Nat), Exists fun c => ∀ (x : α), Eq ((fs n) x) c
      ⊢ Eq f fun x => f hα.some
    -/
    let cs n := (this n).choose
    /-
      case inr.refine_1
      α : Type u_1
      β : Type u_2
      f : α → β
      inst✝² : TopologicalSpace β
      inst✝¹ : Nonempty β
      inst✝ : T2Space β
      hα : Nonempty α
      hf : MeasureTheory.StronglyMeasurable f
      fs : Nat → MeasureTheory.SimpleFunc α β := hf.approx
      h_fs_tendsto : ∀ (x : α), Filter.Tendsto (fun n => (fs n) x) Filter.atTop (nhd …
      this : ∀ (n : Nat), Exists fun c => ∀ (x : α), Eq ((fs n) x) c
      cs : Nat → β := fun n => ⋯.choose
      ⊢ Eq f fun x => f hα.some
    -/
    have h_cs_eq : ∀ n, ⇑(fs n) = fun _ => cs n := fun n => funext (this n).choose_spec
    /-
      case inr.refine_1
      α : Type u_1
      β : Type u_2
      f : α → β
      inst✝² : TopologicalSpace β
      inst✝¹ : Nonempty β
      inst✝ : T2Space β
      hα : Nonempty α
      hf : MeasureTheory.StronglyMeasurable f
      fs : Nat → MeasureTheory.SimpleFunc α β := hf.approx
      h_fs_tendsto : ∀ (x : α), Filter.Tendsto (fun n => (fs n) x) Filter.atTop (nhd …
      this : ∀ (n : Nat), Exists fun c => ∀ (x : α), Eq ((fs n) x) c
      cs : Nat → β := fun n => ⋯.choose
      h_cs_eq : ∀ (n : Nat), Eq ⇑(fs n) fun x => cs n
      ⊢ Eq f fun x => f hα.some
    -/
    conv at h_fs_tendsto => enter [x, 1, n]; rw [h_cs_eq]
    /-
      case inr.refine_1
      α : Type u_1
      β : Type u_2
      f : α → β
      inst✝² : TopologicalSpace β
      inst✝¹ : Nonempty β
      inst✝ : T2Space β
      hα : Nonempty α
      hf : MeasureTheory.StronglyMeasurable f
      fs : Nat → MeasureTheory.SimpleFunc α β := hf.approx
      this : ∀ (n : Nat), Exists fun c => ∀ (x : α), Eq ((fs n) x) c
      cs : Nat → β := fun n => ⋯.choose
      h_fs_tendsto : ∀ (x : α), Filter.Tendsto (fun n => (fun x => cs n) x) Filter.a …
      h_cs_eq : ∀ (n : Nat), Eq ⇑(fs n) fun x => cs n
      ⊢ Eq f fun x => f hα.some
    -/
    have h_tendsto : Tendsto cs atTop (𝓝 (f hα.some)) := h_fs_tendsto hα.some
    /-
      case inr.refine_1
      α : Type u_1
      β : Type u_2
      f : α → β
      inst✝² : TopologicalSpace β
      inst✝¹ : Nonempty β
      inst✝ : T2Space β
      hα : Nonempty α
      hf : MeasureTheory.StronglyMeasurable f
      fs : Nat → MeasureTheory.SimpleFunc α β := hf.approx
      this : ∀ (n : Nat), Exists fun c => ∀ (x : α), Eq ((fs n) x) c
      cs : Nat → β := fun n => ⋯.choose
      h_fs_tendsto : ∀ (x : α), Filter.Tendsto (fun n => (fun x => cs n) x) Filter.a …
      h_cs_eq : ∀ (n : Nat), Eq ⇑(fs n) fun x => cs n
      h_tendsto : Filter.Tendsto cs Filter.atTop (nhds (f hα.some))
      ⊢ Eq f fun x => f hα.some
    -/
    ext1 x
    /-
      case inr.refine_1.h
      α : Type u_1
      β : Type u_2
      f : α → β
      inst✝² : TopologicalSpace β
      inst✝¹ : Nonempty β
      inst✝ : T2Space β
      hα : Nonempty α
      hf : MeasureTheory.StronglyMeasurable f
      fs : Nat → MeasureTheory.SimpleFunc α β := hf.approx
      this : ∀ (n : Nat), Exists fun c => ∀ (x : α), Eq ((fs n) x) c
      cs : Nat → β := fun n => ⋯.choose
      h_fs_tendsto : ∀ (x : α), Filter.Tendsto (fun n => (fun x => cs n) x) Filter.a …
      h_cs_eq : ∀ (n : Nat), Eq ⇑(fs n) fun x => cs n
      h_tendsto : Filter.Tendsto cs Filter.atTop (nhds (f hα.some))
      x : α
      ⊢ Eq (f x) (f hα.some)
    -/
    exact tendsto_nhds_unique (h_fs_tendsto x) h_tendsto
    /-
      🎉 no goals
    -/
    /-
      case inr.refine_2
      α : Type u_1
      β : Type u_2
      f : α → β
      inst✝² : TopologicalSpace β
      inst✝¹ : Nonempty β
      inst✝ : T2Space β
      hα : Nonempty α
      hf_eq : Exists fun c => Eq f fun x => c
      ⊢ MeasureTheory.StronglyMeasurable f
    -/
  · obtain ⟨c, rfl⟩ := hf_eq
    /-
      case inr.refine_2.intro
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace β
      inst✝¹ : Nonempty β
      inst✝ : T2Space β
      hα : Nonempty α
      c : β
      ⊢ MeasureTheory.StronglyMeasurable fun x => c
    -/
    exact stronglyMeasurable_const
    /-
      🎉 no goals
    -/


theorem finStronglyMeasurable_of_set_sigmaFinite [TopologicalSpace β] [Zero β]
    {m : MeasurableSpace α} {μ : Measure α} (hf_meas : StronglyMeasurable f) {t : Set α}
    (ht : MeasurableSet t) (hft_zero : ∀ x ∈ tᶜ, f x = 0) (htμ : SigmaFinite (μ.restrict t)) :
    FinStronglyMeasurable f μ := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    inst✝¹ : TopologicalSpace β
    inst✝ : Zero β
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hf_meas : MeasureTheory.StronglyMeasurable f
    t : Set α
    ht : MeasurableSet t
    hft_zero : ∀ (x : α), Membership.mem (HasCompl.compl t) x → Eq (f x) 0
    htμ : MeasureTheory.SigmaFinite (μ.restrict t)
    ⊢ MeasureTheory.FinStronglyMeasurable f μ
  -/
  haveI : SigmaFinite (μ.restrict t) := htμ
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    inst✝¹ : TopologicalSpace β
    inst✝ : Zero β
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hf_meas : MeasureTheory.StronglyMeasurable f
    t : Set α
    ht : MeasurableSet t
    hft_zero : ∀ (x : α), Membership.mem (HasCompl.compl t) x → Eq (f x) 0
    htμ this : MeasureTheory.SigmaFinite (μ.restrict t)
    ⊢ MeasureTheory.FinStronglyMeasurable f μ
  -/
  let S := spanningSets (μ.restrict t)
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    inst✝¹ : TopologicalSpace β
    inst✝ : Zero β
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hf_meas : MeasureTheory.StronglyMeasurable f
    t : Set α
    ht : MeasurableSet t
    hft_zero : ∀ (x : α), Membership.mem (HasCompl.compl t) x → Eq (f x) 0
    htμ this : MeasureTheory.SigmaFinite (μ.restrict t)
    S : Nat → Set α := MeasureTheory.spanningSets (μ.restrict t)
    ⊢ MeasureTheory.FinStronglyMeasurable f μ
  -/
  have hS_meas : ∀ n, MeasurableSet (S n) := measurableSet_spanningSets (μ.restrict t)
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    inst✝¹ : TopologicalSpace β
    inst✝ : Zero β
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hf_meas : MeasureTheory.StronglyMeasurable f
    t : Set α
    ht : MeasurableSet t
    hft_zero : ∀ (x : α), Membership.mem (HasCompl.compl t) x → Eq (f x) 0
    htμ this : MeasureTheory.SigmaFinite (μ.restrict t)
    S : Nat → Set α := MeasureTheory.spanningSets (μ.restrict t)
    hS_meas : ∀ (n : Nat), MeasurableSet (S n)
    ⊢ MeasureTheory.FinStronglyMeasurable f μ
  -/
  let f_approx := hf_meas.approx
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    inst✝¹ : TopologicalSpace β
    inst✝ : Zero β
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hf_meas : MeasureTheory.StronglyMeasurable f
    t : Set α
    ht : MeasurableSet t
    hft_zero : ∀ (x : α), Membership.mem (HasCompl.compl t) x → Eq (f x) 0
    htμ this : MeasureTheory.SigmaFinite (μ.restrict t)
    S : Nat → Set α := MeasureTheory.spanningSets (μ.restrict t)
    hS_meas : ∀ (n : Nat), MeasurableSet (S n)
    f_approx : Nat → MeasureTheory.SimpleFunc α β := hf_meas.approx
    ⊢ MeasureTheory.FinStronglyMeasurable f μ
  -/
  let fs n := SimpleFunc.restrict (f_approx n) (S n ∩ t)
  have h_fs_t_compl : ∀ n, ∀ x, x ∉ t → fs n x = 0 := by
    intro n x hxt
    rw [SimpleFunc.restrict_apply _ ((hS_meas n).inter ht)]
    refine Set.indicator_of_not_mem ?_ _
    simp [hxt]
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    inst✝¹ : TopologicalSpace β
    inst✝ : Zero β
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hf_meas : MeasureTheory.StronglyMeasurable f
    t : Set α
    ht : MeasurableSet t
    hft_zero : ∀ (x : α), Membership.mem (HasCompl.compl t) x → Eq (f x) 0
    htμ this : MeasureTheory.SigmaFinite (μ.restrict t)
    S : Nat → Set α := MeasureTheory.spanningSets (μ.restrict t)
    hS_meas : ∀ (n : Nat), MeasurableSet (S n)
    f_approx : Nat → MeasureTheory.SimpleFunc α β := hf_meas.approx
    fs : Nat → MeasureTheory.SimpleFunc α β := fun n => (f_approx n).restrict (Int …
    h_fs_t_compl : ∀ (n : Nat) (x : α), Not (Membership.mem t x) → Eq ((fs n) x) 0
    ⊢ MeasureTheory.FinStronglyMeasurable f μ
  -/
  refine ⟨fs, ?_, fun x => ?_⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      f : α → β
      inst✝¹ : TopologicalSpace β
      inst✝ : Zero β
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hf_meas : MeasureTheory.StronglyMeasurable f
      t : Set α
      ht : MeasurableSet t
      hft_zero : ∀ (x : α), Membership.mem (HasCompl.compl t) x → Eq (f x) 0
      htμ this : MeasureTheory.SigmaFinite (μ.restrict t)
      S : Nat → Set α := MeasureTheory.spanningSets (μ.restrict t)
      hS_meas : ∀ (n : Nat), MeasurableSet (S n)
      f_approx : Nat → MeasureTheory.SimpleFunc α β := hf_meas.approx
      fs : Nat → MeasureTheory.SimpleFunc α β := fun n => (f_approx n).restrict (Int …
      h_fs_t_compl : ∀ (n : Nat) (x : α), Not (Membership.mem t x) → Eq ((fs n) x) 0
      ⊢ ∀ (n : Nat), LT.lt (μ (Function.support ⇑(fs n))) Top.top
    -/
  · simp_rw [SimpleFunc.support_eq, ← Finset.mem_coe]
    classical
    refine fun n => measure_biUnion_lt_top {y ∈ (fs n).range | y ≠ 0}.finite_toSet fun y hy => ?_
    rw [SimpleFunc.restrict_preimage_singleton _ ((hS_meas n).inter ht)]
    swap
    · letI : (y : β) → Decidable (y = 0) := fun y => Classical.propDecidable _
      rw [Finset.mem_coe, Finset.mem_filter] at hy
      exact hy.2
    refine (measure_mono Set.inter_subset_left).trans_lt ?_
    have h_lt_top := measure_spanningSets_lt_top (μ.restrict t) n
    rwa [Measure.restrict_apply' ht] at h_lt_top
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      f : α → β
      inst✝¹ : TopologicalSpace β
      inst✝ : Zero β
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hf_meas : MeasureTheory.StronglyMeasurable f
      t : Set α
      ht : MeasurableSet t
      hft_zero : ∀ (x : α), Membership.mem (HasCompl.compl t) x → Eq (f x) 0
      htμ this : MeasureTheory.SigmaFinite (μ.restrict t)
      S : Nat → Set α := MeasureTheory.spanningSets (μ.restrict t)
      hS_meas : ∀ (n : Nat), MeasurableSet (S n)
      f_approx : Nat → MeasureTheory.SimpleFunc α β := hf_meas.approx
      fs : Nat → MeasureTheory.SimpleFunc α β := fun n => (f_approx n).restrict (Int …
      h_fs_t_compl : ∀ (n : Nat) (x : α), Not (Membership.mem t x) → Eq ((fs n) x) 0
      x : α
      ⊢ Filter.Tendsto (fun n => (fs n) x) Filter.atTop (nhds (f x))
    -/
  · by_cases hxt : x ∈ t
    /-
      case pos
      α : Type u_1
      β : Type u_2
      f : α → β
      inst✝¹ : TopologicalSpace β
      inst✝ : Zero β
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hf_meas : MeasureTheory.StronglyMeasurable f
      t : Set α
      ht : MeasurableSet t
      hft_zero : ∀ (x : α), Membership.mem (HasCompl.compl t) x → Eq (f x) 0
      htμ this : MeasureTheory.SigmaFinite (μ.restrict t)
      S : Nat → Set α := MeasureTheory.spanningSets (μ.restrict t)
      hS_meas : ∀ (n : Nat), MeasurableSet (S n)
      f_approx : Nat → MeasureTheory.SimpleFunc α β := hf_meas.approx
      fs : Nat → MeasureTheory.SimpleFunc α β := fun n => (f_approx n).restrict (Int …
      h_fs_t_compl : ∀ (n : Nat) (x : α), Not (Membership.mem t x) → Eq ((fs n) x) 0
      x : α
      hxt : Membership.mem t x
      ⊢ Filter.Tendsto (fun n => (fs n) x) Filter.atTop (nhds (f x))
    -/
    swap
      /-
        case neg
        α : Type u_1
        β : Type u_2
        f : α → β
        inst✝¹ : TopologicalSpace β
        inst✝ : Zero β
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        hf_meas : MeasureTheory.StronglyMeasurable f
        t : Set α
        ht : MeasurableSet t
        hft_zero : ∀ (x : α), Membership.mem (HasCompl.compl t) x → Eq (f x) 0
        htμ this : MeasureTheory.SigmaFinite (μ.restrict t)
        S : Nat → Set α := MeasureTheory.spanningSets (μ.restrict t)
        hS_meas : ∀ (n : Nat), MeasurableSet (S n)
        f_approx : Nat → MeasureTheory.SimpleFunc α β := hf_meas.approx
        fs : Nat → MeasureTheory.SimpleFunc α β := fun n => (f_approx n).restrict (Int …
        h_fs_t_compl : ∀ (n : Nat) (x : α), Not (Membership.mem t x) → Eq ((fs n) x) 0
        x : α
        hxt : Not (Membership.mem t x)
        ⊢ Filter.Tendsto (fun n => (fs n) x) Filter.atTop (nhds (f x))
      -/
    · rw [funext fun n => h_fs_t_compl n x hxt, hft_zero x hxt]
      /-
        case neg
        α : Type u_1
        β : Type u_2
        f : α → β
        inst✝¹ : TopologicalSpace β
        inst✝ : Zero β
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        hf_meas : MeasureTheory.StronglyMeasurable f
        t : Set α
        ht : MeasurableSet t
        hft_zero : ∀ (x : α), Membership.mem (HasCompl.compl t) x → Eq (f x) 0
        htμ this : MeasureTheory.SigmaFinite (μ.restrict t)
        S : Nat → Set α := MeasureTheory.spanningSets (μ.restrict t)
        hS_meas : ∀ (n : Nat), MeasurableSet (S n)
        f_approx : Nat → MeasureTheory.SimpleFunc α β := hf_meas.approx
        fs : Nat → MeasureTheory.SimpleFunc α β := fun n => (f_approx n).restrict (Int …
        h_fs_t_compl : ∀ (n : Nat) (x : α), Not (Membership.mem t x) → Eq ((fs n) x) 0
        x : α
        hxt : Not (Membership.mem t x)
        ⊢ Filter.Tendsto (fun n => 0) Filter.atTop (nhds 0)
      -/
      exact tendsto_const_nhds
      /-
        🎉 no goals
      -/
    /-
      case pos
      α : Type u_1
      β : Type u_2
      f : α → β
      inst✝¹ : TopologicalSpace β
      inst✝ : Zero β
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hf_meas : MeasureTheory.StronglyMeasurable f
      t : Set α
      ht : MeasurableSet t
      hft_zero : ∀ (x : α), Membership.mem (HasCompl.compl t) x → Eq (f x) 0
      htμ this : MeasureTheory.SigmaFinite (μ.restrict t)
      S : Nat → Set α := MeasureTheory.spanningSets (μ.restrict t)
      hS_meas : ∀ (n : Nat), MeasurableSet (S n)
      f_approx : Nat → MeasureTheory.SimpleFunc α β := hf_meas.approx
      fs : Nat → MeasureTheory.SimpleFunc α β := fun n => (f_approx n).restrict (Int …
      h_fs_t_compl : ∀ (n : Nat) (x : α), Not (Membership.mem t x) → Eq ((fs n) x) 0
      x : α
      hxt : Membership.mem t x
      ⊢ Filter.Tendsto (fun n => (fs n) x) Filter.atTop (nhds (f x))
    -/
    have h : Tendsto (fun n => (f_approx n) x) atTop (𝓝 (f x)) := hf_meas.tendsto_approx x
    obtain ⟨n₁, hn₁⟩ : ∃ n, ∀ m, n ≤ m → fs m x = f_approx m x := by
      obtain ⟨n, hn⟩ : ∃ n, ∀ m, n ≤ m → x ∈ S m ∩ t := by
        rsuffices ⟨n, hn⟩ : ∃ n, ∀ m, n ≤ m → x ∈ S m
        · exact ⟨n, fun m hnm => Set.mem_inter (hn m hnm) hxt⟩
        rsuffices ⟨n, hn⟩ : ∃ n, x ∈ S n
        · exact ⟨n, fun m hnm => monotone_spanningSets (μ.restrict t) hnm hn⟩
        rw [← Set.mem_iUnion, iUnion_spanningSets (μ.restrict t)]
        trivial
      refine ⟨n, fun m hnm => ?_⟩
      simp_rw [fs, SimpleFunc.restrict_apply _ ((hS_meas m).inter ht),
        Set.indicator_of_mem (hn m hnm)]
    /-
      case pos.intro
      α : Type u_1
      β : Type u_2
      f : α → β
      inst✝¹ : TopologicalSpace β
      inst✝ : Zero β
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hf_meas : MeasureTheory.StronglyMeasurable f
      t : Set α
      ht : MeasurableSet t
      hft_zero : ∀ (x : α), Membership.mem (HasCompl.compl t) x → Eq (f x) 0
      htμ this : MeasureTheory.SigmaFinite (μ.restrict t)
      S : Nat → Set α := MeasureTheory.spanningSets (μ.restrict t)
      hS_meas : ∀ (n : Nat), MeasurableSet (S n)
      f_approx : Nat → MeasureTheory.SimpleFunc α β := hf_meas.approx
      fs : Nat → MeasureTheory.SimpleFunc α β := fun n => (f_approx n).restrict (Int …
      h_fs_t_compl : ∀ (n : Nat) (x : α), Not (Membership.mem t x) → Eq ((fs n) x) 0
      x : α
      hxt : Membership.mem t x
      h : Filter.Tendsto (fun n => (f_approx n) x) Filter.atTop (nhds (f x))
      n₁ : Nat
      hn₁ : ∀ (m_1 : Nat), LE.le n₁ m_1 → Eq ((fs m_1) x) ((f_approx m_1) x)
      ⊢ Filter.Tendsto (fun n => (fs n) x) Filter.atTop (nhds (f x))
    -/
    rw [tendsto_atTop'] at h ⊢
    /-
      case pos.intro
      α : Type u_1
      β : Type u_2
      f : α → β
      inst✝¹ : TopologicalSpace β
      inst✝ : Zero β
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hf_meas : MeasureTheory.StronglyMeasurable f
      t : Set α
      ht : MeasurableSet t
      hft_zero : ∀ (x : α), Membership.mem (HasCompl.compl t) x → Eq (f x) 0
      htμ this : MeasureTheory.SigmaFinite (μ.restrict t)
      S : Nat → Set α := MeasureTheory.spanningSets (μ.restrict t)
      hS_meas : ∀ (n : Nat), MeasurableSet (S n)
      f_approx : Nat → MeasureTheory.SimpleFunc α β := hf_meas.approx
      fs : Nat → MeasureTheory.SimpleFunc α β := fun n => (f_approx n).restrict (Int …
      h_fs_t_compl : ∀ (n : Nat) (x : α), Not (Membership.mem t x) → Eq ((fs n) x) 0
      x : α
      hxt : Membership.mem t x
      h : ∀ (s : Set β), Membership.mem (nhds (f x)) s → Exists fun a => ∀ (b : Nat) …
      n₁ : Nat
      hn₁ : ∀ (m_1 : Nat), LE.le n₁ m_1 → Eq ((fs m_1) x) ((f_approx m_1) x)
      ⊢ ∀ (s : Set β), Membership.mem (nhds (f x)) s → Exists fun a => ∀ (b : Nat),  …
    -/
    intro s hs
    /-
      case pos.intro
      α : Type u_1
      β : Type u_2
      f : α → β
      inst✝¹ : TopologicalSpace β
      inst✝ : Zero β
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hf_meas : MeasureTheory.StronglyMeasurable f
      t : Set α
      ht : MeasurableSet t
      hft_zero : ∀ (x : α), Membership.mem (HasCompl.compl t) x → Eq (f x) 0
      htμ this : MeasureTheory.SigmaFinite (μ.restrict t)
      S : Nat → Set α := MeasureTheory.spanningSets (μ.restrict t)
      hS_meas : ∀ (n : Nat), MeasurableSet (S n)
      f_approx : Nat → MeasureTheory.SimpleFunc α β := hf_meas.approx
      fs : Nat → MeasureTheory.SimpleFunc α β := fun n => (f_approx n).restrict (Int …
      h_fs_t_compl : ∀ (n : Nat) (x : α), Not (Membership.mem t x) → Eq ((fs n) x) 0
      x : α
      hxt : Membership.mem t x
      h : ∀ (s : Set β), Membership.mem (nhds (f x)) s → Exists fun a => ∀ (b : Nat) …
      n₁ : Nat
      hn₁ : ∀ (m_1 : Nat), LE.le n₁ m_1 → Eq ((fs m_1) x) ((f_approx m_1) x)
      s : Set β
      hs : Membership.mem (nhds (f x)) s
      ⊢ Exists fun a => ∀ (b : Nat), GE.ge b a → Membership.mem s ((fs b) x)
    -/
    obtain ⟨n₂, hn₂⟩ := h s hs
    /-
      case pos.intro.intro
      α : Type u_1
      β : Type u_2
      f : α → β
      inst✝¹ : TopologicalSpace β
      inst✝ : Zero β
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hf_meas : MeasureTheory.StronglyMeasurable f
      t : Set α
      ht : MeasurableSet t
      hft_zero : ∀ (x : α), Membership.mem (HasCompl.compl t) x → Eq (f x) 0
      htμ this : MeasureTheory.SigmaFinite (μ.restrict t)
      S : Nat → Set α := MeasureTheory.spanningSets (μ.restrict t)
      hS_meas : ∀ (n : Nat), MeasurableSet (S n)
      f_approx : Nat → MeasureTheory.SimpleFunc α β := hf_meas.approx
      fs : Nat → MeasureTheory.SimpleFunc α β := fun n => (f_approx n).restrict (Int …
      h_fs_t_compl : ∀ (n : Nat) (x : α), Not (Membership.mem t x) → Eq ((fs n) x) 0
      x : α
      hxt : Membership.mem t x
      h : ∀ (s : Set β), Membership.mem (nhds (f x)) s → Exists fun a => ∀ (b : Nat) …
      n₁ : Nat
      hn₁ : ∀ (m_1 : Nat), LE.le n₁ m_1 → Eq ((fs m_1) x) ((f_approx m_1) x)
      s : Set β
      hs : Membership.mem (nhds (f x)) s
      n₂ : Nat
      hn₂ : ∀ (b : Nat), GE.ge b n₂ → Membership.mem s ((f_approx b) x)
      ⊢ Exists fun a => ∀ (b : Nat), GE.ge b a → Membership.mem s ((fs b) x)
    -/
    refine ⟨max n₁ n₂, fun m hm => ?_⟩
    /-
      case pos.intro.intro
      α : Type u_1
      β : Type u_2
      f : α → β
      inst✝¹ : TopologicalSpace β
      inst✝ : Zero β
      m✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hf_meas : MeasureTheory.StronglyMeasurable f
      t : Set α
      ht : MeasurableSet t
      hft_zero : ∀ (x : α), Membership.mem (HasCompl.compl t) x → Eq (f x) 0
      htμ this : MeasureTheory.SigmaFinite (μ.restrict t)
      S : Nat → Set α := MeasureTheory.spanningSets (μ.restrict t)
      hS_meas : ∀ (n : Nat), MeasurableSet (S n)
      f_approx : Nat → MeasureTheory.SimpleFunc α β := hf_meas.approx
      fs : Nat → MeasureTheory.SimpleFunc α β := fun n => (f_approx n).restrict (Int …
      h_fs_t_compl : ∀ (n : Nat) (x : α), Not (Membership.mem t x) → Eq ((fs n) x) 0
      x : α
      hxt : Membership.mem t x
      h : ∀ (s : Set β), Membership.mem (nhds (f x)) s → Exists fun a => ∀ (b : Nat) …
      n₁ : Nat
      hn₁ : ∀ (m : Nat), LE.le n₁ m → Eq ((fs m) x) ((f_approx m) x)
      s : Set β
      hs : Membership.mem (nhds (f x)) s
      n₂ : Nat
      hn₂ : ∀ (b : Nat), GE.ge b n₂ → Membership.mem s ((f_approx b) x)
      m : Nat
      hm : GE.ge m (Max.max n₁ n₂)
      ⊢ Membership.mem s ((fs m) x)
    -/
    rw [hn₁ m ((le_max_left _ _).trans hm.le)]
    /-
      case pos.intro.intro
      α : Type u_1
      β : Type u_2
      f : α → β
      inst✝¹ : TopologicalSpace β
      inst✝ : Zero β
      m✝ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hf_meas : MeasureTheory.StronglyMeasurable f
      t : Set α
      ht : MeasurableSet t
      hft_zero : ∀ (x : α), Membership.mem (HasCompl.compl t) x → Eq (f x) 0
      htμ this : MeasureTheory.SigmaFinite (μ.restrict t)
      S : Nat → Set α := MeasureTheory.spanningSets (μ.restrict t)
      hS_meas : ∀ (n : Nat), MeasurableSet (S n)
      f_approx : Nat → MeasureTheory.SimpleFunc α β := hf_meas.approx
      fs : Nat → MeasureTheory.SimpleFunc α β := fun n => (f_approx n).restrict (Int …
      h_fs_t_compl : ∀ (n : Nat) (x : α), Not (Membership.mem t x) → Eq ((fs n) x) 0
      x : α
      hxt : Membership.mem t x
      h : ∀ (s : Set β), Membership.mem (nhds (f x)) s → Exists fun a => ∀ (b : Nat) …
      n₁ : Nat
      hn₁ : ∀ (m : Nat), LE.le n₁ m → Eq ((fs m) x) ((f_approx m) x)
      s : Set β
      hs : Membership.mem (nhds (f x)) s
      n₂ : Nat
      hn₂ : ∀ (b : Nat), GE.ge b n₂ → Membership.mem s ((f_approx b) x)
      m : Nat
      hm : GE.ge m (Max.max n₁ n₂)
      ⊢ Membership.mem s ((f_approx m) x)
    -/
    exact hn₂ m ((le_max_right _ _).trans hm.le)
    /-
      🎉 no goals
    -/


/-- If the measure is sigma-finite, all strongly measurable functions are
  `FinStronglyMeasurable`. -/
@[aesop 5% apply (rule_sets := [Measurable])]
protected theorem finStronglyMeasurable [TopologicalSpace β] [Zero β] {m0 : MeasurableSpace α}
    (hf : StronglyMeasurable f) (μ : Measure α) [SigmaFinite μ] : FinStronglyMeasurable f μ :=
                                                                     /-
                                                                       α : Type u_1
                                                                       β : Type u_2
                                                                       f : α → β
                                                                       inst✝² : TopologicalSpace β
                                                                       inst✝¹ : Zero β
                                                                       m0 : MeasurableSpace α
                                                                       hf : MeasureTheory.StronglyMeasurable f
                                                                       μ : MeasureTheory.Measure α
                                                                       inst✝ : MeasureTheory.SigmaFinite μ
                                                                       ⊢ ∀ (x : α), Membership.mem (HasCompl.compl Set.univ) x → Eq (f x) 0
                                                                     -/
  hf.finStronglyMeasurable_of_set_sigmaFinite MeasurableSet.univ (by simp)
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
        /-
          α : Type u_1
          β : Type u_2
          f : α → β
          inst✝² : TopologicalSpace β
          inst✝¹ : Zero β
          m0 : MeasurableSpace α
          hf : MeasureTheory.StronglyMeasurable f
          μ : MeasureTheory.Measure α
          inst✝ : MeasureTheory.SigmaFinite μ
          ⊢ MeasureTheory.SigmaFinite (μ.restrict Set.univ)
        -/
    (by rwa [Measure.restrict_univ])
        /-
          🎉 no goals
        -/


/-- A strongly measurable function is measurable. -/
@[aesop 5% apply (rule_sets := [Measurable])]
protected theorem measurable {_ : MeasurableSpace α} [TopologicalSpace β] [PseudoMetrizableSpace β]
    [MeasurableSpace β] [BorelSpace β] (hf : StronglyMeasurable f) : Measurable f :=
  measurable_of_tendsto_metrizable (fun n => (hf.approx n).measurable)
    (tendsto_pi_nhds.mpr hf.tendsto_approx)


/-- A strongly measurable function is almost everywhere measurable. -/
@[aesop 5% apply (rule_sets := [Measurable])]
protected theorem aemeasurable {_ : MeasurableSpace α} [TopologicalSpace β]
    [PseudoMetrizableSpace β] [MeasurableSpace β] [BorelSpace β] {μ : Measure α}
    (hf : StronglyMeasurable f) : AEMeasurable f μ :=
  hf.measurable.aemeasurable


theorem _root_.Continuous.comp_stronglyMeasurable {_ : MeasurableSpace α} [TopologicalSpace β]
    [TopologicalSpace γ] {g : β → γ} {f : α → β} (hg : Continuous g) (hf : StronglyMeasurable f) :
    StronglyMeasurable fun x => g (f x) :=
  ⟨fun n => SimpleFunc.map g (hf.approx n), fun x => (hg.tendsto _).comp (hf.tendsto_approx x)⟩


@[to_additive]
nonrec theorem measurableSet_mulSupport {m : MeasurableSpace α} [One β] [TopologicalSpace β]
    [MetrizableSpace β] (hf : StronglyMeasurable f) : MeasurableSet (mulSupport f) := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    m : MeasurableSpace α
    inst✝² : One β
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace.MetrizableSpace β
    hf : MeasureTheory.StronglyMeasurable f
    ⊢ MeasurableSet (Function.mulSupport f)
  -/
  borelize β
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    m : MeasurableSpace α
    inst✝² : One β
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace.MetrizableSpace β
    hf : MeasureTheory.StronglyMeasurable f
    this✝¹ : MeasurableSpace β := borel β
    this✝ : BorelSpace β
    ⊢ MeasurableSet (Function.mulSupport f)
  -/
  exact measurableSet_mulSupport hf.measurable
  /-
    🎉 no goals
  -/


protected theorem mono {m m' : MeasurableSpace α} [TopologicalSpace β]
    (hf : StronglyMeasurable[m'] f) (h_mono : m' ≤ m) : StronglyMeasurable[m] f := by
  let f_approx : ℕ → @SimpleFunc α m β := fun n =>
    @SimpleFunc.mk α m β
      (hf.approx n)
      (fun x => h_mono _ (SimpleFunc.measurableSet_fiber' _ x))
      (SimpleFunc.finite_range (hf.approx n))
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    m m' : MeasurableSpace α
    inst✝ : TopologicalSpace β
    hf : MeasureTheory.StronglyMeasurable f
    h_mono : LE.le m' m
    f_approx : Nat → MeasureTheory.SimpleFunc α β := fun n => { toFun := ⇑(hf.appr …
    ⊢ MeasureTheory.StronglyMeasurable f
  -/
  exact ⟨f_approx, hf.tendsto_approx⟩
  /-
    🎉 no goals
  -/


protected theorem prod_mk {m : MeasurableSpace α} [TopologicalSpace β] [TopologicalSpace γ]
    {f : α → β} {g : α → γ} (hf : StronglyMeasurable f) (hg : StronglyMeasurable g) :
    StronglyMeasurable fun x => (f x, g x) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    m : MeasurableSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace γ
    f : α → β
    g : α → γ
    hf : MeasureTheory.StronglyMeasurable f
    hg : MeasureTheory.StronglyMeasurable g
    ⊢ MeasureTheory.StronglyMeasurable fun x => { fst := f x, snd := g x }
  -/
  refine ⟨fun n => SimpleFunc.pair (hf.approx n) (hg.approx n), fun x => ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    m : MeasurableSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace γ
    f : α → β
    g : α → γ
    hf : MeasureTheory.StronglyMeasurable f
    hg : MeasureTheory.StronglyMeasurable g
    x : α
    ⊢ Filter.Tendsto (fun n => ((fun n => (hf.approx n).pair (hg.approx n)) n) x)  …
  -/
  rw [nhds_prod_eq]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    m : MeasurableSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace γ
    f : α → β
    g : α → γ
    hf : MeasureTheory.StronglyMeasurable f
    hg : MeasureTheory.StronglyMeasurable g
    x : α
    ⊢ Filter.Tendsto (fun n => ((fun n => (hf.approx n).pair (hg.approx n)) n) x)  …
  -/
  exact Tendsto.prod_mk (hf.tendsto_approx x) (hg.tendsto_approx x)
  /-
    🎉 no goals
  -/


theorem comp_measurable [TopologicalSpace β] {_ : MeasurableSpace α} {_ : MeasurableSpace γ}
    {f : α → β} {g : γ → α} (hf : StronglyMeasurable f) (hg : Measurable g) :
    StronglyMeasurable (f ∘ g) :=
  ⟨fun n => SimpleFunc.comp (hf.approx n) g hg, fun x => hf.tendsto_approx (g x)⟩


theorem of_uncurry_left [TopologicalSpace β] {_ : MeasurableSpace α} {_ : MeasurableSpace γ}
    {f : α → γ → β} (hf : StronglyMeasurable (uncurry f)) {x : α} : StronglyMeasurable (f x) :=
  hf.comp_measurable measurable_prod_mk_left


theorem of_uncurry_right [TopologicalSpace β] {_ : MeasurableSpace α} {_ : MeasurableSpace γ}
    {f : α → γ → β} (hf : StronglyMeasurable (uncurry f)) {y : γ} :
    StronglyMeasurable fun x => f x y :=
  hf.comp_measurable measurable_prod_mk_right


protected theorem prod_swap {_ : MeasurableSpace α} {_ : MeasurableSpace β} [TopologicalSpace γ]
    {f : β × α → γ} (hf : StronglyMeasurable f) :
    StronglyMeasurable (fun z : α × β => f z.swap) :=
  hf.comp_measurable measurable_swap


protected theorem fst {_ : MeasurableSpace α} [mβ : MeasurableSpace β] [TopologicalSpace γ]
    {f : α → γ} (hf : StronglyMeasurable f) :
    StronglyMeasurable (fun z : α × β => f z.1) :=
  hf.comp_measurable measurable_fst


protected theorem snd [mα : MeasurableSpace α] {_ : MeasurableSpace β} [TopologicalSpace γ]
    {f : β → γ} (hf : StronglyMeasurable f) :
    StronglyMeasurable (fun z : α × β => f z.2) :=
  hf.comp_measurable measurable_snd


@[to_additive (attr := aesop safe 20 apply (rule_sets := [Measurable]))]
protected theorem mul [Mul β] [ContinuousMul β] (hf : StronglyMeasurable f)
    (hg : StronglyMeasurable g) : StronglyMeasurable (f * g) :=
  ⟨fun n => hf.approx n * hg.approx n, fun x => (hf.tendsto_approx x).mul (hg.tendsto_approx x)⟩


@[to_additive (attr := measurability)]
theorem mul_const [Mul β] [ContinuousMul β] (hf : StronglyMeasurable f) (c : β) :
    StronglyMeasurable fun x => f x * c :=
  hf.mul stronglyMeasurable_const


@[to_additive (attr := measurability)]
theorem const_mul [Mul β] [ContinuousMul β] (hf : StronglyMeasurable f) (c : β) :
    StronglyMeasurable fun x => c * f x :=
  stronglyMeasurable_const.mul hf


@[to_additive (attr := aesop safe 20 apply (rule_sets := [Measurable])) const_nsmul]
protected theorem pow [Monoid β] [ContinuousMul β] (hf : StronglyMeasurable f) (n : ℕ) :
    StronglyMeasurable (f ^ n) :=
  ⟨fun k => hf.approx k ^ n, fun x => (hf.tendsto_approx x).pow n⟩


@[to_additive (attr := measurability)]
protected theorem inv [Inv β] [ContinuousInv β] (hf : StronglyMeasurable f) :
    StronglyMeasurable f⁻¹ :=
  ⟨fun n => (hf.approx n)⁻¹, fun x => (hf.tendsto_approx x).inv⟩


@[to_additive (attr := aesop safe 20 apply (rule_sets := [Measurable]))]
protected theorem div [Div β] [ContinuousDiv β] (hf : StronglyMeasurable f)
    (hg : StronglyMeasurable g) : StronglyMeasurable (f / g) :=
  ⟨fun n => hf.approx n / hg.approx n, fun x => (hf.tendsto_approx x).div' (hg.tendsto_approx x)⟩


@[to_additive]
theorem mul_iff_right [CommGroup β] [TopologicalGroup β] (hf : StronglyMeasurable f) :
    StronglyMeasurable (f * g) ↔ StronglyMeasurable g :=
                                   /-
                                     α : Type u_1
                                     β : Type u_2
                                     f g : α → β
                                     mα : MeasurableSpace α
                                     inst✝² : TopologicalSpace β
                                     inst✝¹ : CommGroup β
                                     inst✝ : TopologicalGroup β
                                     hf : MeasureTheory.StronglyMeasurable f
                                     h : MeasureTheory.StronglyMeasurable (HMul.hMul f g)
                                     ⊢ Eq g (HMul.hMul (HMul.hMul f g) (Inv.inv f))
                                   -/
  ⟨fun h ↦ show g = f * g * f⁻¹ by simp only [mul_inv_cancel_comm] ▸ h.mul hf.inv,
                                   /-
                                     🎉 no goals
                                   -/
    fun h ↦ hf.mul h⟩


@[to_additive]
theorem mul_iff_left [CommGroup β] [TopologicalGroup β] (hf : StronglyMeasurable f) :
    StronglyMeasurable (g * f) ↔ StronglyMeasurable g :=
  mul_comm g f ▸ mul_iff_right hf


@[to_additive (attr := aesop safe 20 apply (rule_sets := [Measurable]))]
protected theorem smul {𝕜} [TopologicalSpace 𝕜] [SMul 𝕜 β] [ContinuousSMul 𝕜 β] {f : α → 𝕜}
    {g : α → β} (hf : StronglyMeasurable f) (hg : StronglyMeasurable g) :
    StronglyMeasurable fun x => f x • g x :=
  continuous_smul.comp_stronglyMeasurable (hf.prod_mk hg)


@[to_additive (attr := measurability)]
protected theorem const_smul {𝕜} [SMul 𝕜 β] [ContinuousConstSMul 𝕜 β] (hf : StronglyMeasurable f)
    (c : 𝕜) : StronglyMeasurable (c • f) :=
  ⟨fun n => c • hf.approx n, fun x => (hf.tendsto_approx x).const_smul c⟩


@[to_additive (attr := measurability)]
protected theorem const_smul' {𝕜} [SMul 𝕜 β] [ContinuousConstSMul 𝕜 β] (hf : StronglyMeasurable f)
    (c : 𝕜) : StronglyMeasurable fun x => c • f x :=
  hf.const_smul c


@[to_additive (attr := measurability)]
protected theorem smul_const {𝕜} [TopologicalSpace 𝕜] [SMul 𝕜 β] [ContinuousSMul 𝕜 β] {f : α → 𝕜}
    (hf : StronglyMeasurable f) (c : β) : StronglyMeasurable fun x => f x • c :=
  continuous_smul.comp_stronglyMeasurable (hf.prod_mk stronglyMeasurable_const)


/-- In a normed vector space, the addition of a measurable function and a strongly measurable
function is measurable. Note that this is not true without further second-countability assumptions
for the addition of two measurable functions. -/
theorem _root_.Measurable.add_stronglyMeasurable
    {α E : Type*} {_ : MeasurableSpace α} [AddGroup E] [TopologicalSpace E]
    [MeasurableSpace E] [BorelSpace E] [ContinuousAdd E] [PseudoMetrizableSpace E]
    {g f : α → E} (hg : Measurable g) (hf : StronglyMeasurable f) :
    Measurable (g + f) := by
  /-
    α : Type u_5
    E : Type u_6
    x✝ : MeasurableSpace α
    inst✝⁵ : AddGroup E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : ContinuousAdd E
    inst✝ : TopologicalSpace.PseudoMetrizableSpace E
    g f : α → E
    hg : Measurable g
    hf : MeasureTheory.StronglyMeasurable f
    ⊢ Measurable (HAdd.hAdd g f)
  -/
  rcases hf with ⟨φ, hφ⟩
  have : Tendsto (fun n x ↦ g x + φ n x) atTop (𝓝 (g + f)) :=
    tendsto_pi_nhds.2 (fun x ↦ tendsto_const_nhds.add (hφ x))
  /-
    case intro
    α : Type u_5
    E : Type u_6
    x✝ : MeasurableSpace α
    inst✝⁵ : AddGroup E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : ContinuousAdd E
    inst✝ : TopologicalSpace.PseudoMetrizableSpace E
    g f : α → E
    hg : Measurable g
    φ : Nat → MeasureTheory.SimpleFunc α E
    hφ : ∀ (x : α), Filter.Tendsto (fun n => (φ n) x) Filter.atTop (nhds (f x))
    this : Filter.Tendsto (fun n x => HAdd.hAdd (g x) ((φ n) x)) Filter.atTop (nhd …
    ⊢ Measurable (HAdd.hAdd g f)
  -/
  apply measurable_of_tendsto_metrizable (fun n ↦ ?_) this
  /-
    α : Type u_5
    E : Type u_6
    x✝ : MeasurableSpace α
    inst✝⁵ : AddGroup E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : ContinuousAdd E
    inst✝ : TopologicalSpace.PseudoMetrizableSpace E
    g f : α → E
    hg : Measurable g
    φ : Nat → MeasureTheory.SimpleFunc α E
    hφ : ∀ (x : α), Filter.Tendsto (fun n => (φ n) x) Filter.atTop (nhds (f x))
    this : Filter.Tendsto (fun n x => HAdd.hAdd (g x) ((φ n) x)) Filter.atTop (nhd …
    n : Nat
    ⊢ Measurable fun x => HAdd.hAdd (g x) ((φ n) x)
  -/
  exact hg.add_simpleFunc _
  /-
    🎉 no goals
  -/


/-- In a normed vector space, the subtraction of a measurable function and a strongly measurable
function is measurable. Note that this is not true without further second-countability assumptions
for the subtraction of two measurable functions. -/
theorem _root_.Measurable.sub_stronglyMeasurable
    {α E : Type*} {_ : MeasurableSpace α} [AddCommGroup E] [TopologicalSpace E]
    [MeasurableSpace E] [BorelSpace E] [ContinuousAdd E] [ContinuousNeg E] [PseudoMetrizableSpace E]
    {g f : α → E} (hg : Measurable g) (hf : StronglyMeasurable f) :
    Measurable (g - f) := by
  /-
    α : Type u_5
    E : Type u_6
    x✝ : MeasurableSpace α
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : MeasurableSpace E
    inst✝³ : BorelSpace E
    inst✝² : ContinuousAdd E
    inst✝¹ : ContinuousNeg E
    inst✝ : TopologicalSpace.PseudoMetrizableSpace E
    g f : α → E
    hg : Measurable g
    hf : MeasureTheory.StronglyMeasurable f
    ⊢ Measurable (HSub.hSub g f)
  -/
  rw [sub_eq_add_neg]
  /-
    α : Type u_5
    E : Type u_6
    x✝ : MeasurableSpace α
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : MeasurableSpace E
    inst✝³ : BorelSpace E
    inst✝² : ContinuousAdd E
    inst✝¹ : ContinuousNeg E
    inst✝ : TopologicalSpace.PseudoMetrizableSpace E
    g f : α → E
    hg : Measurable g
    hf : MeasureTheory.StronglyMeasurable f
    ⊢ Measurable (HAdd.hAdd g (Neg.neg f))
  -/
  exact hg.add_stronglyMeasurable hf.neg
  /-
    🎉 no goals
  -/


/-- In a normed vector space, the addition of a strongly measurable function and a measurable
function is measurable. Note that this is not true without further second-countability assumptions
for the addition of two measurable functions. -/
theorem _root_.Measurable.stronglyMeasurable_add
    {α E : Type*} {_ : MeasurableSpace α} [AddGroup E] [TopologicalSpace E]
    [MeasurableSpace E] [BorelSpace E] [ContinuousAdd E] [PseudoMetrizableSpace E]
    {g f : α → E} (hg : Measurable g) (hf : StronglyMeasurable f) :
    Measurable (f + g) := by
  /-
    α : Type u_5
    E : Type u_6
    x✝ : MeasurableSpace α
    inst✝⁵ : AddGroup E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : ContinuousAdd E
    inst✝ : TopologicalSpace.PseudoMetrizableSpace E
    g f : α → E
    hg : Measurable g
    hf : MeasureTheory.StronglyMeasurable f
    ⊢ Measurable (HAdd.hAdd f g)
  -/
  rcases hf with ⟨φ, hφ⟩
  have : Tendsto (fun n x ↦ φ n x + g x) atTop (𝓝 (f + g)) :=
    tendsto_pi_nhds.2 (fun x ↦ (hφ x).add tendsto_const_nhds)
  /-
    case intro
    α : Type u_5
    E : Type u_6
    x✝ : MeasurableSpace α
    inst✝⁵ : AddGroup E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : ContinuousAdd E
    inst✝ : TopologicalSpace.PseudoMetrizableSpace E
    g f : α → E
    hg : Measurable g
    φ : Nat → MeasureTheory.SimpleFunc α E
    hφ : ∀ (x : α), Filter.Tendsto (fun n => (φ n) x) Filter.atTop (nhds (f x))
    this : Filter.Tendsto (fun n x => HAdd.hAdd ((φ n) x) (g x)) Filter.atTop (nhd …
    ⊢ Measurable (HAdd.hAdd f g)
  -/
  apply measurable_of_tendsto_metrizable (fun n ↦ ?_) this
  /-
    α : Type u_5
    E : Type u_6
    x✝ : MeasurableSpace α
    inst✝⁵ : AddGroup E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : MeasurableSpace E
    inst✝² : BorelSpace E
    inst✝¹ : ContinuousAdd E
    inst✝ : TopologicalSpace.PseudoMetrizableSpace E
    g f : α → E
    hg : Measurable g
    φ : Nat → MeasureTheory.SimpleFunc α E
    hφ : ∀ (x : α), Filter.Tendsto (fun n => (φ n) x) Filter.atTop (nhds (f x))
    this : Filter.Tendsto (fun n x => HAdd.hAdd ((φ n) x) (g x)) Filter.atTop (nhd …
    n : Nat
    ⊢ Measurable fun x => HAdd.hAdd ((φ n) x) (g x)
  -/
  exact hg.simpleFunc_add _
  /-
    🎉 no goals
  -/


theorem _root_.stronglyMeasurable_const_smul_iff {m : MeasurableSpace α} (c : G) :
    (StronglyMeasurable fun x => c • f x) ↔ StronglyMeasurable f :=
               /-
                 α : Type u_1
                 β : Type u_2
                 f : α → β
                 G : Type u_6
                 inst✝³ : TopologicalSpace β
                 inst✝² : Group G
                 inst✝¹ : MulAction G β
                 inst✝ : ContinuousConstSMul G β
                 m : MeasurableSpace α
                 c : G
                 h : MeasureTheory.StronglyMeasurable fun x => HSMul.hSMul c (f x)
                 ⊢ MeasureTheory.StronglyMeasurable f
               -/
  ⟨fun h => by simpa only [inv_smul_smul] using h.const_smul' c⁻¹, fun h => h.const_smul c⟩
               /-
                 🎉 no goals
               -/


nonrec theorem _root_.IsUnit.stronglyMeasurable_const_smul_iff {_ : MeasurableSpace α} {c : M}
    (hc : IsUnit c) :
    (StronglyMeasurable fun x => c • f x) ↔ StronglyMeasurable f :=
  let ⟨u, hu⟩ := hc
  hu ▸ stronglyMeasurable_const_smul_iff u


theorem _root_.stronglyMeasurable_const_smul_iff₀ {_ : MeasurableSpace α} {c : G₀} (hc : c ≠ 0) :
    (StronglyMeasurable fun x => c • f x) ↔ StronglyMeasurable f :=
  (IsUnit.mk0 _ hc).stronglyMeasurable_const_smul_iff


@[aesop safe 20 (rule_sets := [Measurable])]
protected theorem sup [Max β] [ContinuousSup β] (hf : StronglyMeasurable f)
    (hg : StronglyMeasurable g) : StronglyMeasurable (f ⊔ g) :=
  ⟨fun n => hf.approx n ⊔ hg.approx n, fun x =>
    (hf.tendsto_approx x).sup_nhds (hg.tendsto_approx x)⟩


@[aesop safe 20 (rule_sets := [Measurable])]
protected theorem inf [Min β] [ContinuousInf β] (hf : StronglyMeasurable f)
    (hg : StronglyMeasurable g) : StronglyMeasurable (f ⊓ g) :=
  ⟨fun n => hf.approx n ⊓ hg.approx n, fun x =>
    (hf.tendsto_approx x).inf_nhds (hg.tendsto_approx x)⟩


@[to_additive (attr := measurability)]
theorem _root_.List.stronglyMeasurable_prod' (l : List (α → M))
    (hl : ∀ f ∈ l, StronglyMeasurable f) : StronglyMeasurable l.prod := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝² : Monoid M
    inst✝¹ : TopologicalSpace M
    inst✝ : ContinuousMul M
    m : MeasurableSpace α
    l : List (α → M)
    hl : ∀ (f : α → M), Membership.mem l f → MeasureTheory.StronglyMeasurable f
    ⊢ MeasureTheory.StronglyMeasurable l.prod
  -/
  induction' l with f l ihl; · exact stronglyMeasurable_one
                               /-
                                 🎉 no goals
                               -/
  /-
    case cons
    α : Type u_1
    M : Type u_5
    inst✝² : Monoid M
    inst✝¹ : TopologicalSpace M
    inst✝ : ContinuousMul M
    m : MeasurableSpace α
    f : α → M
    l : List (α → M)
    ihl : (∀ (f : α → M), Membership.mem l f → MeasureTheory.StronglyMeasurable f) …
    hl : ∀ (f_1 : α → M), Membership.mem (List.cons f l) f_1 → MeasureTheory.Stron …
    ⊢ MeasureTheory.StronglyMeasurable (List.cons f l).prod
  -/
  rw [List.forall_mem_cons] at hl
  /-
    case cons
    α : Type u_1
    M : Type u_5
    inst✝² : Monoid M
    inst✝¹ : TopologicalSpace M
    inst✝ : ContinuousMul M
    m : MeasurableSpace α
    f : α → M
    l : List (α → M)
    ihl : (∀ (f : α → M), Membership.mem l f → MeasureTheory.StronglyMeasurable f) …
    hl : And (MeasureTheory.StronglyMeasurable f) (∀ (x : α → M), Membership.mem l …
    ⊢ MeasureTheory.StronglyMeasurable (List.cons f l).prod
  -/
  rw [List.prod_cons]
  /-
    case cons
    α : Type u_1
    M : Type u_5
    inst✝² : Monoid M
    inst✝¹ : TopologicalSpace M
    inst✝ : ContinuousMul M
    m : MeasurableSpace α
    f : α → M
    l : List (α → M)
    ihl : (∀ (f : α → M), Membership.mem l f → MeasureTheory.StronglyMeasurable f) …
    hl : And (MeasureTheory.StronglyMeasurable f) (∀ (x : α → M), Membership.mem l …
    ⊢ MeasureTheory.StronglyMeasurable (HMul.hMul f l.prod)
  -/
  exact hl.1.mul (ihl hl.2)
  /-
    🎉 no goals
  -/


@[to_additive (attr := measurability)]
theorem _root_.List.stronglyMeasurable_prod (l : List (α → M))
    (hl : ∀ f ∈ l, StronglyMeasurable f) :
    StronglyMeasurable fun x => (l.map fun f : α → M => f x).prod := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝² : Monoid M
    inst✝¹ : TopologicalSpace M
    inst✝ : ContinuousMul M
    m : MeasurableSpace α
    l : List (α → M)
    hl : ∀ (f : α → M), Membership.mem l f → MeasureTheory.StronglyMeasurable f
    ⊢ MeasureTheory.StronglyMeasurable fun x => (List.map (fun f => f x) l).prod
  -/
  simpa only [← Pi.list_prod_apply] using l.stronglyMeasurable_prod' hl
  /-
    🎉 no goals
  -/


@[to_additive (attr := measurability)]
theorem _root_.Multiset.stronglyMeasurable_prod' (l : Multiset (α → M))
    (hl : ∀ f ∈ l, StronglyMeasurable f) : StronglyMeasurable l.prod := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝² : CommMonoid M
    inst✝¹ : TopologicalSpace M
    inst✝ : ContinuousMul M
    m : MeasurableSpace α
    l : Multiset (α → M)
    hl : ∀ (f : α → M), Membership.mem l f → MeasureTheory.StronglyMeasurable f
    ⊢ MeasureTheory.StronglyMeasurable l.prod
  -/
  rcases l with ⟨l⟩
  /-
    case mk
    α : Type u_1
    M : Type u_5
    inst✝² : CommMonoid M
    inst✝¹ : TopologicalSpace M
    inst✝ : ContinuousMul M
    m : MeasurableSpace α
    l✝ : Multiset (α → M)
    l : List (α → M)
    hl : ∀ (f : α → M), Membership.mem (Quot.mk (⇑(List.isSetoid (α → M))) l) f →  …
    ⊢ MeasureTheory.StronglyMeasurable (Multiset.prod (Quot.mk (⇑(List.isSetoid (α …
  -/
  simpa using l.stronglyMeasurable_prod' (by simpa using hl)
  /-
    🎉 no goals
  -/


@[to_additive (attr := measurability)]
theorem _root_.Multiset.stronglyMeasurable_prod (s : Multiset (α → M))
    (hs : ∀ f ∈ s, StronglyMeasurable f) :
    StronglyMeasurable fun x => (s.map fun f : α → M => f x).prod := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝² : CommMonoid M
    inst✝¹ : TopologicalSpace M
    inst✝ : ContinuousMul M
    m : MeasurableSpace α
    s : Multiset (α → M)
    hs : ∀ (f : α → M), Membership.mem s f → MeasureTheory.StronglyMeasurable f
    ⊢ MeasureTheory.StronglyMeasurable fun x => (Multiset.map (fun f => f x) s).prod
  -/
  simpa only [← Pi.multiset_prod_apply] using s.stronglyMeasurable_prod' hs
  /-
    🎉 no goals
  -/


@[to_additive (attr := measurability)]
theorem _root_.Finset.stronglyMeasurable_prod' {ι : Type*} {f : ι → α → M} (s : Finset ι)
    (hf : ∀ i ∈ s, StronglyMeasurable (f i)) : StronglyMeasurable (∏ i ∈ s, f i) :=
  Finset.prod_induction _ _ (fun _a _b ha hb => ha.mul hb) (@stronglyMeasurable_one α M _ _ _) hf


@[to_additive (attr := measurability)]
theorem _root_.Finset.stronglyMeasurable_prod {ι : Type*} {f : ι → α → M} (s : Finset ι)
    (hf : ∀ i ∈ s, StronglyMeasurable (f i)) : StronglyMeasurable fun a => ∏ i ∈ s, f i a := by
  /-
    α : Type u_1
    M : Type u_5
    inst✝² : CommMonoid M
    inst✝¹ : TopologicalSpace M
    inst✝ : ContinuousMul M
    m : MeasurableSpace α
    ι : Type u_6
    f : ι → α → M
    s : Finset ι
    hf : ∀ (i : ι), Membership.mem s i → MeasureTheory.StronglyMeasurable (f i)
    ⊢ MeasureTheory.StronglyMeasurable fun a => s.prod fun i => f i a
  -/
  simpa only [← Finset.prod_apply] using s.stronglyMeasurable_prod' hf
  /-
    🎉 no goals
  -/


/-- The range of a strongly measurable function is separable. -/
protected theorem isSeparable_range {m : MeasurableSpace α} [TopologicalSpace β]
    (hf : StronglyMeasurable f) : TopologicalSpace.IsSeparable (range f) := by
  have : IsSeparable (closure (⋃ n, range (hf.approx n))) :=
    .closure <| .iUnion fun n => (hf.approx n).finite_range.isSeparable
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    m : MeasurableSpace α
    inst✝ : TopologicalSpace β
    hf : MeasureTheory.StronglyMeasurable f
    this : TopologicalSpace.IsSeparable (closure (Set.iUnion fun n => Set.range ⇑( …
    ⊢ TopologicalSpace.IsSeparable (Set.range f)
  -/
  apply this.mono
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    m : MeasurableSpace α
    inst✝ : TopologicalSpace β
    hf : MeasureTheory.StronglyMeasurable f
    this : TopologicalSpace.IsSeparable (closure (Set.iUnion fun n => Set.range ⇑( …
    ⊢ HasSubset.Subset (Set.range f) (closure (Set.iUnion fun n => Set.range ⇑(hf. …
  -/
  rintro _ ⟨x, rfl⟩
  /-
    case intro
    α : Type u_1
    β : Type u_2
    f : α → β
    m : MeasurableSpace α
    inst✝ : TopologicalSpace β
    hf : MeasureTheory.StronglyMeasurable f
    this : TopologicalSpace.IsSeparable (closure (Set.iUnion fun n => Set.range ⇑( …
    x : α
    ⊢ Membership.mem (closure (Set.iUnion fun n => Set.range ⇑(hf.approx n))) (f x)
  -/
  apply mem_closure_of_tendsto (hf.tendsto_approx x)
  /-
    case intro
    α : Type u_1
    β : Type u_2
    f : α → β
    m : MeasurableSpace α
    inst✝ : TopologicalSpace β
    hf : MeasureTheory.StronglyMeasurable f
    this : TopologicalSpace.IsSeparable (closure (Set.iUnion fun n => Set.range ⇑( …
    x : α
    ⊢ Filter.Eventually (fun x_1 => Membership.mem (Set.iUnion fun n => Set.range  …
  -/
  filter_upwards with n
  /-
    case intro.h
    α : Type u_1
    β : Type u_2
    f : α → β
    m : MeasurableSpace α
    inst✝ : TopologicalSpace β
    hf : MeasureTheory.StronglyMeasurable f
    this : TopologicalSpace.IsSeparable (closure (Set.iUnion fun n => Set.range ⇑( …
    x : α
    n : Nat
    ⊢ Membership.mem (Set.iUnion fun n => Set.range ⇑(hf.approx n)) ((hf.approx n) …
  -/
  apply mem_iUnion_of_mem n
  /-
    case intro.h
    α : Type u_1
    β : Type u_2
    f : α → β
    m : MeasurableSpace α
    inst✝ : TopologicalSpace β
    hf : MeasureTheory.StronglyMeasurable f
    this : TopologicalSpace.IsSeparable (closure (Set.iUnion fun n => Set.range ⇑( …
    x : α
    n : Nat
    ⊢ Membership.mem (Set.range ⇑(hf.approx n)) ((hf.approx n) x)
  -/
  exact mem_range_self _
  /-
    🎉 no goals
  -/


theorem separableSpace_range_union_singleton {_ : MeasurableSpace α} [TopologicalSpace β]
    [PseudoMetrizableSpace β] (hf : StronglyMeasurable f) {b : β} :
    SeparableSpace (range f ∪ {b} : Set β) :=
  letI := pseudoMetrizableSpacePseudoMetric β
  (hf.isSeparable_range.union (finite_singleton _).isSeparable).separableSpace


/-- In a space with second countable topology, measurable implies strongly measurable. -/
@[aesop 90% apply (rule_sets := [Measurable])]
theorem _root_.Measurable.stronglyMeasurable [TopologicalSpace β] [PseudoMetrizableSpace β]
    [SecondCountableTopology β] [OpensMeasurableSpace β] (hf : Measurable f) :
    StronglyMeasurable f := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    mα : MeasurableSpace α
    inst✝⁴ : MeasurableSpace β
    inst✝³ : TopologicalSpace β
    inst✝² : TopologicalSpace.PseudoMetrizableSpace β
    inst✝¹ : SecondCountableTopology β
    inst✝ : OpensMeasurableSpace β
    hf : Measurable f
    ⊢ MeasureTheory.StronglyMeasurable f
  -/
  letI := pseudoMetrizableSpacePseudoMetric β
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    mα : MeasurableSpace α
    inst✝⁴ : MeasurableSpace β
    inst✝³ : TopologicalSpace β
    inst✝² : TopologicalSpace.PseudoMetrizableSpace β
    inst✝¹ : SecondCountableTopology β
    inst✝ : OpensMeasurableSpace β
    hf : Measurable f
    this : PseudoMetricSpace β := TopologicalSpace.pseudoMetrizableSpacePseudoMetr …
    ⊢ MeasureTheory.StronglyMeasurable f
  -/
  nontriviality β; inhabit β
  exact ⟨SimpleFunc.approxOn f hf Set.univ default (Set.mem_univ _), fun x ↦
    SimpleFunc.tendsto_approxOn hf (Set.mem_univ _) (by rw [closure_univ]; simp)⟩


/-- In a space with second countable topology, strongly measurable and measurable are equivalent. -/
theorem _root_.stronglyMeasurable_iff_measurable [TopologicalSpace β] [MetrizableSpace β]
    [BorelSpace β] [SecondCountableTopology β] : StronglyMeasurable f ↔ Measurable f :=
  ⟨fun h => h.measurable, fun h => Measurable.stronglyMeasurable h⟩


@[measurability]
theorem _root_.stronglyMeasurable_id [TopologicalSpace α] [PseudoMetrizableSpace α]
    [OpensMeasurableSpace α] [SecondCountableTopology α] : StronglyMeasurable (id : α → α) :=
  measurable_id.stronglyMeasurable


/-- A function is strongly measurable if and only if it is measurable and has separable
range. -/
theorem _root_.stronglyMeasurable_iff_measurable_separable {m : MeasurableSpace α}
    [TopologicalSpace β] [PseudoMetrizableSpace β] [MeasurableSpace β] [BorelSpace β] :
    StronglyMeasurable f ↔ Measurable f ∧ IsSeparable (range f) := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    m : MeasurableSpace α
    inst✝³ : TopologicalSpace β
    inst✝² : TopologicalSpace.PseudoMetrizableSpace β
    inst✝¹ : MeasurableSpace β
    inst✝ : BorelSpace β
    ⊢ Iff (MeasureTheory.StronglyMeasurable f) (And (Measurable f) (TopologicalSpa …
  -/
  refine ⟨fun H ↦ ⟨H.measurable, H.isSeparable_range⟩, fun ⟨Hm, Hsep⟩  ↦ ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    m : MeasurableSpace α
    inst✝³ : TopologicalSpace β
    inst✝² : TopologicalSpace.PseudoMetrizableSpace β
    inst✝¹ : MeasurableSpace β
    inst✝ : BorelSpace β
    x✝ : And (Measurable f) (TopologicalSpace.IsSeparable (Set.range f))
    Hm : Measurable f
    Hsep : TopologicalSpace.IsSeparable (Set.range f)
    ⊢ MeasureTheory.StronglyMeasurable f
  -/
  have := Hsep.secondCountableTopology
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    m : MeasurableSpace α
    inst✝³ : TopologicalSpace β
    inst✝² : TopologicalSpace.PseudoMetrizableSpace β
    inst✝¹ : MeasurableSpace β
    inst✝ : BorelSpace β
    x✝ : And (Measurable f) (TopologicalSpace.IsSeparable (Set.range f))
    Hm : Measurable f
    Hsep : TopologicalSpace.IsSeparable (Set.range f)
    this : SecondCountableTopology ↑(Set.range f)
    ⊢ MeasureTheory.StronglyMeasurable f
  -/
  have Hm' : StronglyMeasurable (rangeFactorization f) := Hm.subtype_mk.stronglyMeasurable
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    m : MeasurableSpace α
    inst✝³ : TopologicalSpace β
    inst✝² : TopologicalSpace.PseudoMetrizableSpace β
    inst✝¹ : MeasurableSpace β
    inst✝ : BorelSpace β
    x✝ : And (Measurable f) (TopologicalSpace.IsSeparable (Set.range f))
    Hm : Measurable f
    Hsep : TopologicalSpace.IsSeparable (Set.range f)
    this : SecondCountableTopology ↑(Set.range f)
    Hm' : MeasureTheory.StronglyMeasurable (Set.rangeFactorization f)
    ⊢ MeasureTheory.StronglyMeasurable f
  -/
  exact continuous_subtype_val.comp_stronglyMeasurable Hm'
  /-
    🎉 no goals
  -/


/-- A continuous function is strongly measurable when either the source space or the target space
is second-countable. -/
theorem _root_.Continuous.stronglyMeasurable [MeasurableSpace α] [TopologicalSpace α]
    [OpensMeasurableSpace α] [TopologicalSpace β] [PseudoMetrizableSpace β]
    [h : SecondCountableTopologyEither α β] {f : α → β} (hf : Continuous f) :
    StronglyMeasurable f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : MeasurableSpace α
    inst✝³ : TopologicalSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace.PseudoMetrizableSpace β
    h : SecondCountableTopologyEither α β
    f : α → β
    hf : Continuous f
    ⊢ MeasureTheory.StronglyMeasurable f
  -/
  borelize β
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : MeasurableSpace α
    inst✝³ : TopologicalSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace.PseudoMetrizableSpace β
    h : SecondCountableTopologyEither α β
    f : α → β
    hf : Continuous f
    this✝¹ : MeasurableSpace β := borel β
    this✝ : BorelSpace β
    ⊢ MeasureTheory.StronglyMeasurable f
  -/
  cases h.out
    /-
      case inl
      α : Type u_1
      β : Type u_2
      inst✝⁴ : MeasurableSpace α
      inst✝³ : TopologicalSpace α
      inst✝² : OpensMeasurableSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : TopologicalSpace.PseudoMetrizableSpace β
      h : SecondCountableTopologyEither α β
      f : α → β
      hf : Continuous f
      this✝¹ : MeasurableSpace β := borel β
      this✝ : BorelSpace β
      h✝ : SecondCountableTopology α
      ⊢ MeasureTheory.StronglyMeasurable f
    -/
  · rw [stronglyMeasurable_iff_measurable_separable]
    /-
      case inl
      α : Type u_1
      β : Type u_2
      inst✝⁴ : MeasurableSpace α
      inst✝³ : TopologicalSpace α
      inst✝² : OpensMeasurableSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : TopologicalSpace.PseudoMetrizableSpace β
      h : SecondCountableTopologyEither α β
      f : α → β
      hf : Continuous f
      this✝¹ : MeasurableSpace β := borel β
      this✝ : BorelSpace β
      h✝ : SecondCountableTopology α
      ⊢ And (Measurable f) (TopologicalSpace.IsSeparable (Set.range f))
    -/
    refine ⟨hf.measurable, ?_⟩
    /-
      case inl
      α : Type u_1
      β : Type u_2
      inst✝⁴ : MeasurableSpace α
      inst✝³ : TopologicalSpace α
      inst✝² : OpensMeasurableSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : TopologicalSpace.PseudoMetrizableSpace β
      h : SecondCountableTopologyEither α β
      f : α → β
      hf : Continuous f
      this✝¹ : MeasurableSpace β := borel β
      this✝ : BorelSpace β
      h✝ : SecondCountableTopology α
      ⊢ TopologicalSpace.IsSeparable (Set.range f)
    -/
    exact isSeparable_range hf
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      β : Type u_2
      inst✝⁴ : MeasurableSpace α
      inst✝³ : TopologicalSpace α
      inst✝² : OpensMeasurableSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : TopologicalSpace.PseudoMetrizableSpace β
      h : SecondCountableTopologyEither α β
      f : α → β
      hf : Continuous f
      this✝¹ : MeasurableSpace β := borel β
      this✝ : BorelSpace β
      h✝ : SecondCountableTopology β
      ⊢ MeasureTheory.StronglyMeasurable f
    -/
  · exact hf.measurable.stronglyMeasurable
    /-
      🎉 no goals
    -/


/-- A continuous function whose support is contained in a compact set is strongly measurable. -/
@[to_additive]
theorem _root_.Continuous.stronglyMeasurable_of_mulSupport_subset_isCompact
    [MeasurableSpace α] [TopologicalSpace α] [OpensMeasurableSpace α] [MeasurableSpace β]
    [TopologicalSpace β] [PseudoMetrizableSpace β] [BorelSpace β] [One β] {f : α → β}
    (hf : Continuous f) {k : Set α} (hk : IsCompact k)
    (h'f : mulSupport f ⊆ k) : StronglyMeasurable f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁷ : MeasurableSpace α
    inst✝⁶ : TopologicalSpace α
    inst✝⁵ : OpensMeasurableSpace α
    inst✝⁴ : MeasurableSpace β
    inst✝³ : TopologicalSpace β
    inst✝² : TopologicalSpace.PseudoMetrizableSpace β
    inst✝¹ : BorelSpace β
    inst✝ : One β
    f : α → β
    hf : Continuous f
    k : Set α
    hk : IsCompact k
    h'f : HasSubset.Subset (Function.mulSupport f) k
    ⊢ MeasureTheory.StronglyMeasurable f
  -/
  letI : PseudoMetricSpace β := pseudoMetrizableSpacePseudoMetric β
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁷ : MeasurableSpace α
    inst✝⁶ : TopologicalSpace α
    inst✝⁵ : OpensMeasurableSpace α
    inst✝⁴ : MeasurableSpace β
    inst✝³ : TopologicalSpace β
    inst✝² : TopologicalSpace.PseudoMetrizableSpace β
    inst✝¹ : BorelSpace β
    inst✝ : One β
    f : α → β
    hf : Continuous f
    k : Set α
    hk : IsCompact k
    h'f : HasSubset.Subset (Function.mulSupport f) k
    this : PseudoMetricSpace β := TopologicalSpace.pseudoMetrizableSpacePseudoMetr …
    ⊢ MeasureTheory.StronglyMeasurable f
  -/
  rw [stronglyMeasurable_iff_measurable_separable]
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁷ : MeasurableSpace α
    inst✝⁶ : TopologicalSpace α
    inst✝⁵ : OpensMeasurableSpace α
    inst✝⁴ : MeasurableSpace β
    inst✝³ : TopologicalSpace β
    inst✝² : TopologicalSpace.PseudoMetrizableSpace β
    inst✝¹ : BorelSpace β
    inst✝ : One β
    f : α → β
    hf : Continuous f
    k : Set α
    hk : IsCompact k
    h'f : HasSubset.Subset (Function.mulSupport f) k
    this : PseudoMetricSpace β := TopologicalSpace.pseudoMetrizableSpacePseudoMetr …
    ⊢ And (Measurable f) (TopologicalSpace.IsSeparable (Set.range f))
  -/
  exact ⟨hf.measurable, (isCompact_range_of_mulSupport_subset_isCompact hf hk h'f).isSeparable⟩
  /-
    🎉 no goals
  -/


/-- A continuous function with compact support is strongly measurable. -/
@[to_additive]
theorem _root_.Continuous.stronglyMeasurable_of_hasCompactMulSupport
    [MeasurableSpace α] [TopologicalSpace α] [OpensMeasurableSpace α] [MeasurableSpace β]
    [TopologicalSpace β] [PseudoMetrizableSpace β] [BorelSpace β] [One β] {f : α → β}
    (hf : Continuous f) (h'f : HasCompactMulSupport f) : StronglyMeasurable f :=
  hf.stronglyMeasurable_of_mulSupport_subset_isCompact h'f (subset_mulTSupport f)


/-- A continuous function with compact support on a product space is strongly measurable for the
product sigma-algebra. The subtlety is that we do not assume that the spaces are separable, so the
product of the Borel sigma algebras might not contain all open sets, but still it contains enough
of them to approximate compactly supported continuous functions. -/
lemma _root_.HasCompactSupport.stronglyMeasurable_of_prod {X Y : Type*} [Zero α]
    [TopologicalSpace X] [TopologicalSpace Y] [MeasurableSpace X] [MeasurableSpace Y]
    [OpensMeasurableSpace X] [OpensMeasurableSpace Y] [TopologicalSpace α] [PseudoMetrizableSpace α]
    {f : X × Y → α} (hf : Continuous f) (h'f : HasCompactSupport f) :
    StronglyMeasurable f := by
  /-
    α : Type u_1
    X : Type u_5
    Y : Type u_6
    inst✝⁸ : Zero α
    inst✝⁷ : TopologicalSpace X
    inst✝⁶ : TopologicalSpace Y
    inst✝⁵ : MeasurableSpace X
    inst✝⁴ : MeasurableSpace Y
    inst✝³ : OpensMeasurableSpace X
    inst✝² : OpensMeasurableSpace Y
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace.PseudoMetrizableSpace α
    f : Prod X Y → α
    hf : Continuous f
    h'f : HasCompactSupport f
    ⊢ MeasureTheory.StronglyMeasurable f
  -/
  borelize α
  /-
    α : Type u_1
    X : Type u_5
    Y : Type u_6
    inst✝⁸ : Zero α
    inst✝⁷ : TopologicalSpace X
    inst✝⁶ : TopologicalSpace Y
    inst✝⁵ : MeasurableSpace X
    inst✝⁴ : MeasurableSpace Y
    inst✝³ : OpensMeasurableSpace X
    inst✝² : OpensMeasurableSpace Y
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace.PseudoMetrizableSpace α
    f : Prod X Y → α
    hf : Continuous f
    h'f : HasCompactSupport f
    this✝¹ : MeasurableSpace α := borel α
    this✝ : BorelSpace α
    ⊢ MeasureTheory.StronglyMeasurable f
  -/
  apply stronglyMeasurable_iff_measurable_separable.2 ⟨h'f.measurable_of_prod hf, ?_⟩
  /-
    α : Type u_1
    X : Type u_5
    Y : Type u_6
    inst✝⁸ : Zero α
    inst✝⁷ : TopologicalSpace X
    inst✝⁶ : TopologicalSpace Y
    inst✝⁵ : MeasurableSpace X
    inst✝⁴ : MeasurableSpace Y
    inst✝³ : OpensMeasurableSpace X
    inst✝² : OpensMeasurableSpace Y
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace.PseudoMetrizableSpace α
    f : Prod X Y → α
    hf : Continuous f
    h'f : HasCompactSupport f
    this✝¹ : MeasurableSpace α := borel α
    this✝ : BorelSpace α
    ⊢ TopologicalSpace.IsSeparable (Set.range f)
  -/
  letI : PseudoMetricSpace α := pseudoMetrizableSpacePseudoMetric α
  /-
    α : Type u_1
    X : Type u_5
    Y : Type u_6
    inst✝⁸ : Zero α
    inst✝⁷ : TopologicalSpace X
    inst✝⁶ : TopologicalSpace Y
    inst✝⁵ : MeasurableSpace X
    inst✝⁴ : MeasurableSpace Y
    inst✝³ : OpensMeasurableSpace X
    inst✝² : OpensMeasurableSpace Y
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace.PseudoMetrizableSpace α
    f : Prod X Y → α
    hf : Continuous f
    h'f : HasCompactSupport f
    this✝¹ : MeasurableSpace α := borel α
    this✝ : BorelSpace α
    this : PseudoMetricSpace α := TopologicalSpace.pseudoMetrizableSpacePseudoMetr …
    ⊢ TopologicalSpace.IsSeparable (Set.range f)
  -/
  exact IsCompact.isSeparable (s := range f) (h'f.isCompact_range hf)
  /-
    🎉 no goals
  -/


/-- If `g` is a topological embedding, then `f` is strongly measurable iff `g ∘ f` is. -/
theorem _root_.Embedding.comp_stronglyMeasurable_iff {m : MeasurableSpace α} [TopologicalSpace β]
    [PseudoMetrizableSpace β] [TopologicalSpace γ] [PseudoMetrizableSpace γ] {g : β → γ} {f : α → β}
    (hg : IsEmbedding g) : (StronglyMeasurable fun x => g (f x)) ↔ StronglyMeasurable f := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    m : MeasurableSpace α
    inst✝³ : TopologicalSpace β
    inst✝² : TopologicalSpace.PseudoMetrizableSpace β
    inst✝¹ : TopologicalSpace γ
    inst✝ : TopologicalSpace.PseudoMetrizableSpace γ
    g : β → γ
    f : α → β
    hg : Topology.IsEmbedding g
    ⊢ Iff (MeasureTheory.StronglyMeasurable fun x => g (f x)) (MeasureTheory.Stron …
  -/
  letI := pseudoMetrizableSpacePseudoMetric γ
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    m : MeasurableSpace α
    inst✝³ : TopologicalSpace β
    inst✝² : TopologicalSpace.PseudoMetrizableSpace β
    inst✝¹ : TopologicalSpace γ
    inst✝ : TopologicalSpace.PseudoMetrizableSpace γ
    g : β → γ
    f : α → β
    hg : Topology.IsEmbedding g
    this : PseudoMetricSpace γ := TopologicalSpace.pseudoMetrizableSpacePseudoMetr …
    ⊢ Iff (MeasureTheory.StronglyMeasurable fun x => g (f x)) (MeasureTheory.Stron …
  -/
  borelize β γ
  refine
    ⟨fun H => stronglyMeasurable_iff_measurable_separable.2 ⟨?_, ?_⟩, fun H =>
      hg.continuous.comp_stronglyMeasurable H⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      m : MeasurableSpace α
      inst✝³ : TopologicalSpace β
      inst✝² : TopologicalSpace.PseudoMetrizableSpace β
      inst✝¹ : TopologicalSpace γ
      inst✝ : TopologicalSpace.PseudoMetrizableSpace γ
      g : β → γ
      f : α → β
      hg : Topology.IsEmbedding g
      this : PseudoMetricSpace γ := TopologicalSpace.pseudoMetrizableSpacePseudoMetr …
      this✝³ : MeasurableSpace β := borel β
      this✝² : BorelSpace β
      this✝¹ : MeasurableSpace γ := borel γ
      this✝ : BorelSpace γ
      H : MeasureTheory.StronglyMeasurable fun x => g (f x)
      ⊢ Measurable f
    -/
  · let G : β → range g := rangeFactorization g
    have hG : IsClosedEmbedding G :=
      { hg.codRestrict _ _ with
        isClosed_range := by
          rw [surjective_onto_range.range_eq]
          exact isClosed_univ }
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      m : MeasurableSpace α
      inst✝³ : TopologicalSpace β
      inst✝² : TopologicalSpace.PseudoMetrizableSpace β
      inst✝¹ : TopologicalSpace γ
      inst✝ : TopologicalSpace.PseudoMetrizableSpace γ
      g : β → γ
      f : α → β
      hg : Topology.IsEmbedding g
      this : PseudoMetricSpace γ := TopologicalSpace.pseudoMetrizableSpacePseudoMetr …
      this✝³ : MeasurableSpace β := borel β
      this✝² : BorelSpace β
      this✝¹ : MeasurableSpace γ := borel γ
      this✝ : BorelSpace γ
      H : MeasureTheory.StronglyMeasurable fun x => g (f x)
      G : β → ↑(Set.range g) := Set.rangeFactorization g
      hG : Topology.IsClosedEmbedding G
      ⊢ Measurable f
    -/
    have : Measurable (G ∘ f) := Measurable.subtype_mk H.measurable
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      m : MeasurableSpace α
      inst✝³ : TopologicalSpace β
      inst✝² : TopologicalSpace.PseudoMetrizableSpace β
      inst✝¹ : TopologicalSpace γ
      inst✝ : TopologicalSpace.PseudoMetrizableSpace γ
      g : β → γ
      f : α → β
      hg : Topology.IsEmbedding g
      this✝⁴ : PseudoMetricSpace γ := TopologicalSpace.pseudoMetrizableSpacePseudoMe …
      this✝³ : MeasurableSpace β := borel β
      this✝² : BorelSpace β
      this✝¹ : MeasurableSpace γ := borel γ
      this✝ : BorelSpace γ
      H : MeasureTheory.StronglyMeasurable fun x => g (f x)
      G : β → ↑(Set.range g) := Set.rangeFactorization g
      hG : Topology.IsClosedEmbedding G
      this : Measurable (Function.comp G f)
      ⊢ Measurable f
    -/
    exact hG.measurableEmbedding.measurable_comp_iff.1 this
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      m : MeasurableSpace α
      inst✝³ : TopologicalSpace β
      inst✝² : TopologicalSpace.PseudoMetrizableSpace β
      inst✝¹ : TopologicalSpace γ
      inst✝ : TopologicalSpace.PseudoMetrizableSpace γ
      g : β → γ
      f : α → β
      hg : Topology.IsEmbedding g
      this : PseudoMetricSpace γ := TopologicalSpace.pseudoMetrizableSpacePseudoMetr …
      this✝³ : MeasurableSpace β := borel β
      this✝² : BorelSpace β
      this✝¹ : MeasurableSpace γ := borel γ
      this✝ : BorelSpace γ
      H : MeasureTheory.StronglyMeasurable fun x => g (f x)
      ⊢ TopologicalSpace.IsSeparable (Set.range f)
    -/
  · have : IsSeparable (g ⁻¹' range (g ∘ f)) := hg.isSeparable_preimage H.isSeparable_range
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      m : MeasurableSpace α
      inst✝³ : TopologicalSpace β
      inst✝² : TopologicalSpace.PseudoMetrizableSpace β
      inst✝¹ : TopologicalSpace γ
      inst✝ : TopologicalSpace.PseudoMetrizableSpace γ
      g : β → γ
      f : α → β
      hg : Topology.IsEmbedding g
      this✝⁴ : PseudoMetricSpace γ := TopologicalSpace.pseudoMetrizableSpacePseudoMe …
      this✝³ : MeasurableSpace β := borel β
      this✝² : BorelSpace β
      this✝¹ : MeasurableSpace γ := borel γ
      this✝ : BorelSpace γ
      H : MeasureTheory.StronglyMeasurable fun x => g (f x)
      this : TopologicalSpace.IsSeparable (Set.preimage g (Set.range (Function.comp  …
      ⊢ TopologicalSpace.IsSeparable (Set.range f)
    -/
    rwa [range_comp, hg.injective.preimage_image] at this
    /-
      🎉 no goals
    -/


/-- A sequential limit of strongly measurable functions is strongly measurable. -/
theorem _root_.stronglyMeasurable_of_tendsto {ι : Type*} {m : MeasurableSpace α}
    [TopologicalSpace β] [PseudoMetrizableSpace β] (u : Filter ι) [NeBot u] [IsCountablyGenerated u]
    {f : ι → α → β} {g : α → β} (hf : ∀ i, StronglyMeasurable (f i)) (lim : Tendsto f u (𝓝 g)) :
    StronglyMeasurable g := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_5
    m : MeasurableSpace α
    inst✝³ : TopologicalSpace β
    inst✝² : TopologicalSpace.PseudoMetrizableSpace β
    u : Filter ι
    inst✝¹ : u.NeBot
    inst✝ : u.IsCountablyGenerated
    f : ι → α → β
    g : α → β
    hf : ∀ (i : ι), MeasureTheory.StronglyMeasurable (f i)
    lim : Filter.Tendsto f u (nhds g)
    ⊢ MeasureTheory.StronglyMeasurable g
  -/
  borelize β
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_5
    m : MeasurableSpace α
    inst✝³ : TopologicalSpace β
    inst✝² : TopologicalSpace.PseudoMetrizableSpace β
    u : Filter ι
    inst✝¹ : u.NeBot
    inst✝ : u.IsCountablyGenerated
    f : ι → α → β
    g : α → β
    hf : ∀ (i : ι), MeasureTheory.StronglyMeasurable (f i)
    lim : Filter.Tendsto f u (nhds g)
    this✝¹ : MeasurableSpace β := borel β
    this✝ : BorelSpace β
    ⊢ MeasureTheory.StronglyMeasurable g
  -/
  refine stronglyMeasurable_iff_measurable_separable.2 ⟨?_, ?_⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      ι : Type u_5
      m : MeasurableSpace α
      inst✝³ : TopologicalSpace β
      inst✝² : TopologicalSpace.PseudoMetrizableSpace β
      u : Filter ι
      inst✝¹ : u.NeBot
      inst✝ : u.IsCountablyGenerated
      f : ι → α → β
      g : α → β
      hf : ∀ (i : ι), MeasureTheory.StronglyMeasurable (f i)
      lim : Filter.Tendsto f u (nhds g)
      this✝¹ : MeasurableSpace β := borel β
      this✝ : BorelSpace β
      ⊢ Measurable g
    -/
  · exact measurable_of_tendsto_metrizable' u (fun i => (hf i).measurable) lim
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      ι : Type u_5
      m : MeasurableSpace α
      inst✝³ : TopologicalSpace β
      inst✝² : TopologicalSpace.PseudoMetrizableSpace β
      u : Filter ι
      inst✝¹ : u.NeBot
      inst✝ : u.IsCountablyGenerated
      f : ι → α → β
      g : α → β
      hf : ∀ (i : ι), MeasureTheory.StronglyMeasurable (f i)
      lim : Filter.Tendsto f u (nhds g)
      this✝¹ : MeasurableSpace β := borel β
      this✝ : BorelSpace β
      ⊢ TopologicalSpace.IsSeparable (Set.range g)
    -/
  · rcases u.exists_seq_tendsto with ⟨v, hv⟩
    have : IsSeparable (closure (⋃ i, range (f (v i)))) :=
      .closure <| .iUnion fun i => (hf (v i)).isSeparable_range
    /-
      case refine_2.intro
      α : Type u_1
      β : Type u_2
      ι : Type u_5
      m : MeasurableSpace α
      inst✝³ : TopologicalSpace β
      inst✝² : TopologicalSpace.PseudoMetrizableSpace β
      u : Filter ι
      inst✝¹ : u.NeBot
      inst✝ : u.IsCountablyGenerated
      f : ι → α → β
      g : α → β
      hf : ∀ (i : ι), MeasureTheory.StronglyMeasurable (f i)
      lim : Filter.Tendsto f u (nhds g)
      this✝¹ : MeasurableSpace β := borel β
      this✝ : BorelSpace β
      v : Nat → ι
      hv : Filter.Tendsto v Filter.atTop u
      this : TopologicalSpace.IsSeparable (closure (Set.iUnion fun i => Set.range (f …
      ⊢ TopologicalSpace.IsSeparable (Set.range g)
    -/
    apply this.mono
    /-
      case refine_2.intro
      α : Type u_1
      β : Type u_2
      ι : Type u_5
      m : MeasurableSpace α
      inst✝³ : TopologicalSpace β
      inst✝² : TopologicalSpace.PseudoMetrizableSpace β
      u : Filter ι
      inst✝¹ : u.NeBot
      inst✝ : u.IsCountablyGenerated
      f : ι → α → β
      g : α → β
      hf : ∀ (i : ι), MeasureTheory.StronglyMeasurable (f i)
      lim : Filter.Tendsto f u (nhds g)
      this✝¹ : MeasurableSpace β := borel β
      this✝ : BorelSpace β
      v : Nat → ι
      hv : Filter.Tendsto v Filter.atTop u
      this : TopologicalSpace.IsSeparable (closure (Set.iUnion fun i => Set.range (f …
      ⊢ HasSubset.Subset (Set.range g) (closure (Set.iUnion fun i => Set.range (f (v …
    -/
    rintro _ ⟨x, rfl⟩
    /-
      case refine_2.intro.intro
      α : Type u_1
      β : Type u_2
      ι : Type u_5
      m : MeasurableSpace α
      inst✝³ : TopologicalSpace β
      inst✝² : TopologicalSpace.PseudoMetrizableSpace β
      u : Filter ι
      inst✝¹ : u.NeBot
      inst✝ : u.IsCountablyGenerated
      f : ι → α → β
      g : α → β
      hf : ∀ (i : ι), MeasureTheory.StronglyMeasurable (f i)
      lim : Filter.Tendsto f u (nhds g)
      this✝¹ : MeasurableSpace β := borel β
      this✝ : BorelSpace β
      v : Nat → ι
      hv : Filter.Tendsto v Filter.atTop u
      this : TopologicalSpace.IsSeparable (closure (Set.iUnion fun i => Set.range (f …
      x : α
      ⊢ Membership.mem (closure (Set.iUnion fun i => Set.range (f (v i)))) (g x)
    -/
    rw [tendsto_pi_nhds] at lim
    /-
      case refine_2.intro.intro
      α : Type u_1
      β : Type u_2
      ι : Type u_5
      m : MeasurableSpace α
      inst✝³ : TopologicalSpace β
      inst✝² : TopologicalSpace.PseudoMetrizableSpace β
      u : Filter ι
      inst✝¹ : u.NeBot
      inst✝ : u.IsCountablyGenerated
      f : ι → α → β
      g : α → β
      hf : ∀ (i : ι), MeasureTheory.StronglyMeasurable (f i)
      lim : ∀ (x : α), Filter.Tendsto (fun i => f i x) u (nhds (g x))
      this✝¹ : MeasurableSpace β := borel β
      this✝ : BorelSpace β
      v : Nat → ι
      hv : Filter.Tendsto v Filter.atTop u
      this : TopologicalSpace.IsSeparable (closure (Set.iUnion fun i => Set.range (f …
      x : α
      ⊢ Membership.mem (closure (Set.iUnion fun i => Set.range (f (v i)))) (g x)
    -/
    apply mem_closure_of_tendsto ((lim x).comp hv)
    /-
      case refine_2.intro.intro
      α : Type u_1
      β : Type u_2
      ι : Type u_5
      m : MeasurableSpace α
      inst✝³ : TopologicalSpace β
      inst✝² : TopologicalSpace.PseudoMetrizableSpace β
      u : Filter ι
      inst✝¹ : u.NeBot
      inst✝ : u.IsCountablyGenerated
      f : ι → α → β
      g : α → β
      hf : ∀ (i : ι), MeasureTheory.StronglyMeasurable (f i)
      lim : ∀ (x : α), Filter.Tendsto (fun i => f i x) u (nhds (g x))
      this✝¹ : MeasurableSpace β := borel β
      this✝ : BorelSpace β
      v : Nat → ι
      hv : Filter.Tendsto v Filter.atTop u
      this : TopologicalSpace.IsSeparable (closure (Set.iUnion fun i => Set.range (f …
      x : α
      ⊢ Filter.Eventually (fun x_1 => Membership.mem (Set.iUnion fun i => Set.range  …
    -/
    filter_upwards with n
    /-
      case refine_2.intro.intro.h
      α : Type u_1
      β : Type u_2
      ι : Type u_5
      m : MeasurableSpace α
      inst✝³ : TopologicalSpace β
      inst✝² : TopologicalSpace.PseudoMetrizableSpace β
      u : Filter ι
      inst✝¹ : u.NeBot
      inst✝ : u.IsCountablyGenerated
      f : ι → α → β
      g : α → β
      hf : ∀ (i : ι), MeasureTheory.StronglyMeasurable (f i)
      lim : ∀ (x : α), Filter.Tendsto (fun i => f i x) u (nhds (g x))
      this✝¹ : MeasurableSpace β := borel β
      this✝ : BorelSpace β
      v : Nat → ι
      hv : Filter.Tendsto v Filter.atTop u
      this : TopologicalSpace.IsSeparable (closure (Set.iUnion fun i => Set.range (f …
      x : α
      n : Nat
      ⊢ Membership.mem (Set.iUnion fun i => Set.range (f (v i))) (Function.comp (fun …
    -/
    apply mem_iUnion_of_mem n
    /-
      case refine_2.intro.intro.h
      α : Type u_1
      β : Type u_2
      ι : Type u_5
      m : MeasurableSpace α
      inst✝³ : TopologicalSpace β
      inst✝² : TopologicalSpace.PseudoMetrizableSpace β
      u : Filter ι
      inst✝¹ : u.NeBot
      inst✝ : u.IsCountablyGenerated
      f : ι → α → β
      g : α → β
      hf : ∀ (i : ι), MeasureTheory.StronglyMeasurable (f i)
      lim : ∀ (x : α), Filter.Tendsto (fun i => f i x) u (nhds (g x))
      this✝¹ : MeasurableSpace β := borel β
      this✝ : BorelSpace β
      v : Nat → ι
      hv : Filter.Tendsto v Filter.atTop u
      this : TopologicalSpace.IsSeparable (closure (Set.iUnion fun i => Set.range (f …
      x : α
      n : Nat
      ⊢ Membership.mem (Set.range (f (v n))) (Function.comp (fun i => f i x) v n)
    -/
    exact mem_range_self _
    /-
      🎉 no goals
    -/


protected theorem piecewise {m : MeasurableSpace α} [TopologicalSpace β] {s : Set α}
    {_ : DecidablePred (· ∈ s)} (hs : MeasurableSet s) (hf : StronglyMeasurable f)
    (hg : StronglyMeasurable g) : StronglyMeasurable (Set.piecewise s f g) := by
  /-
    α : Type u_1
    β : Type u_2
    f g : α → β
    m : MeasurableSpace α
    inst✝ : TopologicalSpace β
    s : Set α
    x✝ : DecidablePred fun x => Membership.mem s x
    hs : MeasurableSet s
    hf : MeasureTheory.StronglyMeasurable f
    hg : MeasureTheory.StronglyMeasurable g
    ⊢ MeasureTheory.StronglyMeasurable (s.piecewise f g)
  -/
  refine ⟨fun n => SimpleFunc.piecewise s hs (hf.approx n) (hg.approx n), fun x => ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    f g : α → β
    m : MeasurableSpace α
    inst✝ : TopologicalSpace β
    s : Set α
    x✝ : DecidablePred fun x => Membership.mem s x
    hs : MeasurableSet s
    hf : MeasureTheory.StronglyMeasurable f
    hg : MeasureTheory.StronglyMeasurable g
    x : α
    ⊢ Filter.Tendsto (fun n => ((fun n => MeasureTheory.SimpleFunc.piecewise s hs  …
  -/
  by_cases hx : x ∈ s
  · simpa [@Set.piecewise_eq_of_mem _ _ _ _ _ (fun _ => Classical.propDecidable _) _ hx,
      hx] using hf.tendsto_approx x
  · simpa [@Set.piecewise_eq_of_not_mem _ _ _ _ _ (fun _ => Classical.propDecidable _) _ hx,
      hx] using hg.tendsto_approx x


/-- this is slightly different from `StronglyMeasurable.piecewise`. It can be used to show
`StronglyMeasurable (ite (x=0) 0 1)` by
`exact StronglyMeasurable.ite (measurableSet_singleton 0) stronglyMeasurable_const
stronglyMeasurable_const`, but replacing `StronglyMeasurable.ite` by
`StronglyMeasurable.piecewise` in that example proof does not work. -/
protected theorem ite {_ : MeasurableSpace α} [TopologicalSpace β] {p : α → Prop}
    {_ : DecidablePred p} (hp : MeasurableSet { a : α | p a }) (hf : StronglyMeasurable f)
    (hg : StronglyMeasurable g) : StronglyMeasurable fun x => ite (p x) (f x) (g x) :=
  StronglyMeasurable.piecewise hp hf hg


@[measurability]
theorem _root_.MeasurableEmbedding.stronglyMeasurable_extend {f : α → β} {g : α → γ} {g' : γ → β}
    {mα : MeasurableSpace α} {mγ : MeasurableSpace γ} [TopologicalSpace β]
    (hg : MeasurableEmbedding g) (hf : StronglyMeasurable f) (hg' : StronglyMeasurable g') :
    StronglyMeasurable (Function.extend g f g') := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → β
    g : α → γ
    g' : γ → β
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    inst✝ : TopologicalSpace β
    hg : MeasurableEmbedding g
    hf : MeasureTheory.StronglyMeasurable f
    hg' : MeasureTheory.StronglyMeasurable g'
    ⊢ MeasureTheory.StronglyMeasurable (Function.extend g f g')
  -/
  refine ⟨fun n => SimpleFunc.extend (hf.approx n) g hg (hg'.approx n), ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → β
    g : α → γ
    g' : γ → β
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    inst✝ : TopologicalSpace β
    hg : MeasurableEmbedding g
    hf : MeasureTheory.StronglyMeasurable f
    hg' : MeasureTheory.StronglyMeasurable g'
    ⊢ ∀ (x : γ), Filter.Tendsto (fun n => ((fun n => (hf.approx n).extend g hg (hg …
  -/
  intro x
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → β
    g : α → γ
    g' : γ → β
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    inst✝ : TopologicalSpace β
    hg : MeasurableEmbedding g
    hf : MeasureTheory.StronglyMeasurable f
    hg' : MeasureTheory.StronglyMeasurable g'
    x : γ
    ⊢ Filter.Tendsto (fun n => ((fun n => (hf.approx n).extend g hg (hg'.approx n) …
  -/
  by_cases hx : ∃ y, g y = x
    /-
      case pos
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      f : α → β
      g : α → γ
      g' : γ → β
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      inst✝ : TopologicalSpace β
      hg : MeasurableEmbedding g
      hf : MeasureTheory.StronglyMeasurable f
      hg' : MeasureTheory.StronglyMeasurable g'
      x : γ
      hx : Exists fun y => Eq (g y) x
      ⊢ Filter.Tendsto (fun n => ((fun n => (hf.approx n).extend g hg (hg'.approx n) …
    -/
  · rcases hx with ⟨y, rfl⟩
    simpa only [SimpleFunc.extend_apply, hg.injective, Injective.extend_apply] using
      hf.tendsto_approx y
  · simpa only [hx, SimpleFunc.extend_apply', not_false_iff, extend_apply'] using
      hg'.tendsto_approx x


theorem _root_.MeasurableEmbedding.exists_stronglyMeasurable_extend {f : α → β} {g : α → γ}
    {_ : MeasurableSpace α} {_ : MeasurableSpace γ} [TopologicalSpace β]
    (hg : MeasurableEmbedding g) (hf : StronglyMeasurable f) (hne : γ → Nonempty β) :
    ∃ f' : γ → β, StronglyMeasurable f' ∧ f' ∘ g = f :=
  ⟨Function.extend g f fun x => Classical.choice (hne x),
    hg.stronglyMeasurable_extend hf (stronglyMeasurable_const' fun _ _ => rfl),
    funext fun _ => hg.injective.extend_apply _ _ _⟩


theorem _root_.stronglyMeasurable_of_stronglyMeasurable_union_cover {m : MeasurableSpace α}
    [TopologicalSpace β] {f : α → β} (s t : Set α) (hs : MeasurableSet s) (ht : MeasurableSet t)
    (h : univ ⊆ s ∪ t) (hc : StronglyMeasurable fun a : s => f a)
    (hd : StronglyMeasurable fun a : t => f a) : StronglyMeasurable f := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    s t : Set α
    hs : MeasurableSet s
    ht : MeasurableSet t
    h : HasSubset.Subset Set.univ (Union.union s t)
    hc : MeasureTheory.StronglyMeasurable fun a => f ↑a
    hd : MeasureTheory.StronglyMeasurable fun a => f ↑a
    ⊢ MeasureTheory.StronglyMeasurable f
  -/
  nontriviality β; inhabit β
  suffices Function.extend Subtype.val (fun x : s ↦ f x)
      (Function.extend (↑) (fun x : t ↦ f x) fun _ ↦ default) = f from
    this ▸ (MeasurableEmbedding.subtype_coe hs).stronglyMeasurable_extend hc <|
      (MeasurableEmbedding.subtype_coe ht).stronglyMeasurable_extend hd stronglyMeasurable_const
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    s t : Set α
    hs : MeasurableSet s
    ht : MeasurableSet t
    h : HasSubset.Subset Set.univ (Union.union s t)
    hc : MeasureTheory.StronglyMeasurable fun a => f ↑a
    hd : MeasureTheory.StronglyMeasurable fun a => f ↑a
    a✝ : Nontrivial β
    inhabited_h : Inhabited β
    ⊢ Eq (Function.extend Subtype.val (fun x => f ↑x) (Function.extend Subtype.val …
  -/
  ext x
  /-
    case h
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    s t : Set α
    hs : MeasurableSet s
    ht : MeasurableSet t
    h : HasSubset.Subset Set.univ (Union.union s t)
    hc : MeasureTheory.StronglyMeasurable fun a => f ↑a
    hd : MeasureTheory.StronglyMeasurable fun a => f ↑a
    a✝ : Nontrivial β
    inhabited_h : Inhabited β
    x : α
    ⊢ Eq (Function.extend Subtype.val (fun x => f ↑x) (Function.extend Subtype.val …
  -/
  by_cases hxs : x ∈ s
    /-
      case pos
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      s t : Set α
      hs : MeasurableSet s
      ht : MeasurableSet t
      h : HasSubset.Subset Set.univ (Union.union s t)
      hc : MeasureTheory.StronglyMeasurable fun a => f ↑a
      hd : MeasureTheory.StronglyMeasurable fun a => f ↑a
      a✝ : Nontrivial β
      inhabited_h : Inhabited β
      x : α
      hxs : Membership.mem s x
      ⊢ Eq (Function.extend Subtype.val (fun x => f ↑x) (Function.extend Subtype.val …
    -/
  · lift x to s using hxs
    /-
      case pos.intro
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      s t : Set α
      hs : MeasurableSet s
      ht : MeasurableSet t
      h : HasSubset.Subset Set.univ (Union.union s t)
      hc : MeasureTheory.StronglyMeasurable fun a => f ↑a
      hd : MeasureTheory.StronglyMeasurable fun a => f ↑a
      a✝ : Nontrivial β
      inhabited_h : Inhabited β
      x : Subtype fun x => Membership.mem s x
      ⊢ Eq (Function.extend Subtype.val (fun x => f ↑x) (Function.extend Subtype.val …
    -/
    simp [Subtype.coe_injective.extend_apply]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      s t : Set α
      hs : MeasurableSet s
      ht : MeasurableSet t
      h : HasSubset.Subset Set.univ (Union.union s t)
      hc : MeasureTheory.StronglyMeasurable fun a => f ↑a
      hd : MeasureTheory.StronglyMeasurable fun a => f ↑a
      a✝ : Nontrivial β
      inhabited_h : Inhabited β
      x : α
      hxs : Not (Membership.mem s x)
      ⊢ Eq (Function.extend Subtype.val (fun x => f ↑x) (Function.extend Subtype.val …
    -/
  · lift x to t using (h trivial).resolve_left hxs
    /-
      case neg.intro
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      s t : Set α
      hs : MeasurableSet s
      ht : MeasurableSet t
      h : HasSubset.Subset Set.univ (Union.union s t)
      hc : MeasureTheory.StronglyMeasurable fun a => f ↑a
      hd : MeasureTheory.StronglyMeasurable fun a => f ↑a
      a✝ : Nontrivial β
      inhabited_h : Inhabited β
      x : Subtype fun x => Membership.mem t x
      hxs : Not (Membership.mem s ↑x)
      ⊢ Eq (Function.extend Subtype.val (fun x => f ↑x) (Function.extend Subtype.val …
    -/
    rw [extend_apply', Subtype.coe_injective.extend_apply]
    /-
      case neg.intro.hb
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      inst✝ : TopologicalSpace β
      f : α → β
      s t : Set α
      hs : MeasurableSet s
      ht : MeasurableSet t
      h : HasSubset.Subset Set.univ (Union.union s t)
      hc : MeasureTheory.StronglyMeasurable fun a => f ↑a
      hd : MeasureTheory.StronglyMeasurable fun a => f ↑a
      a✝ : Nontrivial β
      inhabited_h : Inhabited β
      x : Subtype fun x => Membership.mem t x
      hxs : Not (Membership.mem s ↑x)
      ⊢ Not (Exists fun a => Eq ↑a ↑x)
    -/
    exact fun ⟨y, hy⟩ ↦ hxs <| hy ▸ y.2
    /-
      🎉 no goals
    -/


theorem _root_.stronglyMeasurable_of_restrict_of_restrict_compl {_ : MeasurableSpace α}
    [TopologicalSpace β] {f : α → β} {s : Set α} (hs : MeasurableSet s)
    (h₁ : StronglyMeasurable (s.restrict f)) (h₂ : StronglyMeasurable (sᶜ.restrict f)) :
    StronglyMeasurable f :=
  stronglyMeasurable_of_stronglyMeasurable_union_cover s sᶜ hs hs.compl (union_compl_self s).ge h₁
    h₂


@[measurability]
protected theorem indicator {_ : MeasurableSpace α} [TopologicalSpace β] [Zero β]
    (hf : StronglyMeasurable f) {s : Set α} (hs : MeasurableSet s) :
    StronglyMeasurable (s.indicator f) :=
  hf.piecewise hs stronglyMeasurable_const


@[aesop safe 20 apply (rule_sets := [Measurable])]
protected theorem dist {_ : MeasurableSpace α} {β : Type*} [PseudoMetricSpace β] {f g : α → β}
    (hf : StronglyMeasurable f) (hg : StronglyMeasurable g) :
    StronglyMeasurable fun x => dist (f x) (g x) :=
  continuous_dist.comp_stronglyMeasurable (hf.prod_mk hg)


@[measurability]
protected theorem norm {_ : MeasurableSpace α} {β : Type*} [SeminormedAddCommGroup β] {f : α → β}
    (hf : StronglyMeasurable f) : StronglyMeasurable fun x => ‖f x‖ :=
  continuous_norm.comp_stronglyMeasurable hf


@[measurability]
protected theorem nnnorm {_ : MeasurableSpace α} {β : Type*} [SeminormedAddCommGroup β] {f : α → β}
    (hf : StronglyMeasurable f) : StronglyMeasurable fun x => ‖f x‖₊ :=
  continuous_nnnorm.comp_stronglyMeasurable hf


@[measurability]
protected theorem ennnorm {_ : MeasurableSpace α} {β : Type*} [SeminormedAddCommGroup β]
    {f : α → β} (hf : StronglyMeasurable f) : Measurable fun a => (‖f a‖₊ : ℝ≥0∞) :=
  (ENNReal.continuous_coe.comp_stronglyMeasurable hf.nnnorm).measurable


@[measurability]
protected theorem real_toNNReal {_ : MeasurableSpace α} {f : α → ℝ} (hf : StronglyMeasurable f) :
    StronglyMeasurable fun x => (f x).toNNReal :=
  continuous_real_toNNReal.comp_stronglyMeasurable hf


theorem measurableSet_eq_fun {m : MeasurableSpace α} {E} [TopologicalSpace E] [MetrizableSpace E]
    {f g : α → E} (hf : StronglyMeasurable f) (hg : StronglyMeasurable g) :
    MeasurableSet { x | f x = g x } := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    E : Type u_5
    inst✝¹ : TopologicalSpace E
    inst✝ : TopologicalSpace.MetrizableSpace E
    f g : α → E
    hf : MeasureTheory.StronglyMeasurable f
    hg : MeasureTheory.StronglyMeasurable g
    ⊢ MeasurableSet (setOf fun x => Eq (f x) (g x))
  -/
  borelize (E × E)
  /-
    α : Type u_1
    m : MeasurableSpace α
    E : Type u_5
    inst✝¹ : TopologicalSpace E
    inst✝ : TopologicalSpace.MetrizableSpace E
    f g : α → E
    hf : MeasureTheory.StronglyMeasurable f
    hg : MeasureTheory.StronglyMeasurable g
    this✝¹ : MeasurableSpace (Prod E E) := borel (Prod E E)
    this✝ : BorelSpace (Prod E E)
    ⊢ MeasurableSet (setOf fun x => Eq (f x) (g x))
  -/
  exact (hf.prod_mk hg).measurable isClosed_diagonal.measurableSet
  /-
    🎉 no goals
  -/


theorem measurableSet_lt {m : MeasurableSpace α} [TopologicalSpace β] [LinearOrder β]
    [OrderClosedTopology β] [PseudoMetrizableSpace β] {f g : α → β} (hf : StronglyMeasurable f)
    (hg : StronglyMeasurable g) : MeasurableSet { a | f a < g a } := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝³ : TopologicalSpace β
    inst✝² : LinearOrder β
    inst✝¹ : OrderClosedTopology β
    inst✝ : TopologicalSpace.PseudoMetrizableSpace β
    f g : α → β
    hf : MeasureTheory.StronglyMeasurable f
    hg : MeasureTheory.StronglyMeasurable g
    ⊢ MeasurableSet (setOf fun a => LT.lt (f a) (g a))
  -/
  borelize (β × β)
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝³ : TopologicalSpace β
    inst✝² : LinearOrder β
    inst✝¹ : OrderClosedTopology β
    inst✝ : TopologicalSpace.PseudoMetrizableSpace β
    f g : α → β
    hf : MeasureTheory.StronglyMeasurable f
    hg : MeasureTheory.StronglyMeasurable g
    this✝¹ : MeasurableSpace (Prod β β) := borel (Prod β β)
    this✝ : BorelSpace (Prod β β)
    ⊢ MeasurableSet (setOf fun a => LT.lt (f a) (g a))
  -/
  exact (hf.prod_mk hg).measurable isOpen_lt_prod.measurableSet
  /-
    🎉 no goals
  -/


theorem measurableSet_le {m : MeasurableSpace α} [TopologicalSpace β] [Preorder β]
    [OrderClosedTopology β] [PseudoMetrizableSpace β] {f g : α → β} (hf : StronglyMeasurable f)
    (hg : StronglyMeasurable g) : MeasurableSet { a | f a ≤ g a } := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝³ : TopologicalSpace β
    inst✝² : Preorder β
    inst✝¹ : OrderClosedTopology β
    inst✝ : TopologicalSpace.PseudoMetrizableSpace β
    f g : α → β
    hf : MeasureTheory.StronglyMeasurable f
    hg : MeasureTheory.StronglyMeasurable g
    ⊢ MeasurableSet (setOf fun a => LE.le (f a) (g a))
  -/
  borelize (β × β)
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝³ : TopologicalSpace β
    inst✝² : Preorder β
    inst✝¹ : OrderClosedTopology β
    inst✝ : TopologicalSpace.PseudoMetrizableSpace β
    f g : α → β
    hf : MeasureTheory.StronglyMeasurable f
    hg : MeasureTheory.StronglyMeasurable g
    this✝¹ : MeasurableSpace (Prod β β) := borel (Prod β β)
    this✝ : BorelSpace (Prod β β)
    ⊢ MeasurableSet (setOf fun a => LE.le (f a) (g a))
  -/
  exact (hf.prod_mk hg).measurable isClosed_le_prod.measurableSet
  /-
    🎉 no goals
  -/


theorem stronglyMeasurable_in_set {m : MeasurableSpace α} [TopologicalSpace β] [Zero β] {s : Set α}
    {f : α → β} (hs : MeasurableSet s) (hf : StronglyMeasurable f)
    (hf_zero : ∀ x, x ∉ s → f x = 0) :
    ∃ fs : ℕ → α →ₛ β,
      (∀ x, Tendsto (fun n => fs n x) atTop (𝓝 (f x))) ∧ ∀ x ∉ s, ∀ n, fs n x = 0 := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : Zero β
    s : Set α
    f : α → β
    hs : MeasurableSet s
    hf : MeasureTheory.StronglyMeasurable f
    hf_zero : ∀ (x : α), Not (Membership.mem s x) → Eq (f x) 0
    ⊢ Exists fun fs => And (∀ (x : α), Filter.Tendsto (fun n => (fs n) x) Filter.a …
  -/
  let g_seq_s : ℕ → @SimpleFunc α m β := fun n => (hf.approx n).restrict s
  have hg_eq : ∀ x ∈ s, ∀ n, g_seq_s n x = hf.approx n x := by
    intro x hx n
    rw [SimpleFunc.coe_restrict _ hs, Set.indicator_of_mem hx]
  have hg_zero : ∀ x ∉ s, ∀ n, g_seq_s n x = 0 := by
    intro x hx n
    rw [SimpleFunc.coe_restrict _ hs, Set.indicator_of_not_mem hx]
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : Zero β
    s : Set α
    f : α → β
    hs : MeasurableSet s
    hf : MeasureTheory.StronglyMeasurable f
    hf_zero : ∀ (x : α), Not (Membership.mem s x) → Eq (f x) 0
    g_seq_s : Nat → MeasureTheory.SimpleFunc α β := fun n => (hf.approx n).restric …
    hg_eq : ∀ (x : α), Membership.mem s x → ∀ (n : Nat), Eq ((g_seq_s n) x) ((hf.a …
    hg_zero : ∀ (x : α), Not (Membership.mem s x) → ∀ (n : Nat), Eq ((g_seq_s n) x …
    ⊢ Exists fun fs => And (∀ (x : α), Filter.Tendsto (fun n => (fs n) x) Filter.a …
  -/
  refine ⟨g_seq_s, fun x => ?_, hg_zero⟩
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : Zero β
    s : Set α
    f : α → β
    hs : MeasurableSet s
    hf : MeasureTheory.StronglyMeasurable f
    hf_zero : ∀ (x : α), Not (Membership.mem s x) → Eq (f x) 0
    g_seq_s : Nat → MeasureTheory.SimpleFunc α β := fun n => (hf.approx n).restric …
    hg_eq : ∀ (x : α), Membership.mem s x → ∀ (n : Nat), Eq ((g_seq_s n) x) ((hf.a …
    hg_zero : ∀ (x : α), Not (Membership.mem s x) → ∀ (n : Nat), Eq ((g_seq_s n) x …
    x : α
    ⊢ Filter.Tendsto (fun n => (g_seq_s n) x) Filter.atTop (nhds (f x))
  -/
  by_cases hx : x ∈ s
    /-
      case pos
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : Zero β
      s : Set α
      f : α → β
      hs : MeasurableSet s
      hf : MeasureTheory.StronglyMeasurable f
      hf_zero : ∀ (x : α), Not (Membership.mem s x) → Eq (f x) 0
      g_seq_s : Nat → MeasureTheory.SimpleFunc α β := fun n => (hf.approx n).restric …
      hg_eq : ∀ (x : α), Membership.mem s x → ∀ (n : Nat), Eq ((g_seq_s n) x) ((hf.a …
      hg_zero : ∀ (x : α), Not (Membership.mem s x) → ∀ (n : Nat), Eq ((g_seq_s n) x …
      x : α
      hx : Membership.mem s x
      ⊢ Filter.Tendsto (fun n => (g_seq_s n) x) Filter.atTop (nhds (f x))
    -/
  · simp_rw [hg_eq x hx]
    /-
      case pos
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : Zero β
      s : Set α
      f : α → β
      hs : MeasurableSet s
      hf : MeasureTheory.StronglyMeasurable f
      hf_zero : ∀ (x : α), Not (Membership.mem s x) → Eq (f x) 0
      g_seq_s : Nat → MeasureTheory.SimpleFunc α β := fun n => (hf.approx n).restric …
      hg_eq : ∀ (x : α), Membership.mem s x → ∀ (n : Nat), Eq ((g_seq_s n) x) ((hf.a …
      hg_zero : ∀ (x : α), Not (Membership.mem s x) → ∀ (n : Nat), Eq ((g_seq_s n) x …
      x : α
      hx : Membership.mem s x
      ⊢ Filter.Tendsto (fun n => (hf.approx n) x) Filter.atTop (nhds (f x))
    -/
    exact hf.tendsto_approx x
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : Zero β
      s : Set α
      f : α → β
      hs : MeasurableSet s
      hf : MeasureTheory.StronglyMeasurable f
      hf_zero : ∀ (x : α), Not (Membership.mem s x) → Eq (f x) 0
      g_seq_s : Nat → MeasureTheory.SimpleFunc α β := fun n => (hf.approx n).restric …
      hg_eq : ∀ (x : α), Membership.mem s x → ∀ (n : Nat), Eq ((g_seq_s n) x) ((hf.a …
      hg_zero : ∀ (x : α), Not (Membership.mem s x) → ∀ (n : Nat), Eq ((g_seq_s n) x …
      x : α
      hx : Not (Membership.mem s x)
      ⊢ Filter.Tendsto (fun n => (g_seq_s n) x) Filter.atTop (nhds (f x))
    -/
  · simp_rw [hg_zero x hx, hf_zero x hx]
    /-
      case neg
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : Zero β
      s : Set α
      f : α → β
      hs : MeasurableSet s
      hf : MeasureTheory.StronglyMeasurable f
      hf_zero : ∀ (x : α), Not (Membership.mem s x) → Eq (f x) 0
      g_seq_s : Nat → MeasureTheory.SimpleFunc α β := fun n => (hf.approx n).restric …
      hg_eq : ∀ (x : α), Membership.mem s x → ∀ (n : Nat), Eq ((g_seq_s n) x) ((hf.a …
      hg_zero : ∀ (x : α), Not (Membership.mem s x) → ∀ (n : Nat), Eq ((g_seq_s n) x …
      x : α
      hx : Not (Membership.mem s x)
      ⊢ Filter.Tendsto (fun n => 0) Filter.atTop (nhds 0)
    -/
    exact tendsto_const_nhds
    /-
      🎉 no goals
    -/


/-- If the restriction to a set `s` of a σ-algebra `m` is included in the restriction to `s` of
another σ-algebra `m₂` (hypothesis `hs`), the set `s` is `m` measurable and a function `f` supported
on `s` is `m`-strongly-measurable, then `f` is also `m₂`-strongly-measurable. -/
theorem stronglyMeasurable_of_measurableSpace_le_on {α E} {m m₂ : MeasurableSpace α}
    [TopologicalSpace E] [Zero E] {s : Set α} {f : α → E} (hs_m : MeasurableSet[m] s)
    (hs : ∀ t, MeasurableSet[m] (s ∩ t) → MeasurableSet[m₂] (s ∩ t))
    (hf : StronglyMeasurable[m] f) (hf_zero : ∀ x ∉ s, f x = 0) :
    StronglyMeasurable[m₂] f := by
  have hs_m₂ : MeasurableSet[m₂] s := by
    rw [← Set.inter_univ s]
    refine hs Set.univ ?_
    rwa [Set.inter_univ]
  /-
    α : Type u_5
    E : Type u_6
    m m₂ : MeasurableSpace α
    inst✝¹ : TopologicalSpace E
    inst✝ : Zero E
    s : Set α
    f : α → E
    hs_m : MeasurableSet s
    hs : ∀ (t : Set α), MeasurableSet (Inter.inter s t) → MeasurableSet (Inter.int …
    hf : MeasureTheory.StronglyMeasurable f
    hf_zero : ∀ (x : α), Not (Membership.mem s x) → Eq (f x) 0
    hs_m₂ : MeasurableSet s
    ⊢ MeasureTheory.StronglyMeasurable f
  -/
  obtain ⟨g_seq_s, hg_seq_tendsto, hg_seq_zero⟩ := stronglyMeasurable_in_set hs_m hf hf_zero
  let g_seq_s₂ : ℕ → @SimpleFunc α m₂ E := fun n =>
    { toFun := g_seq_s n
      measurableSet_fiber' := fun x => by
        rw [← Set.inter_univ (g_seq_s n ⁻¹' {x}), ← Set.union_compl_self s,
          Set.inter_union_distrib_left, Set.inter_comm (g_seq_s n ⁻¹' {x})]
        refine MeasurableSet.union (hs _ (hs_m.inter ?_)) ?_
        · exact @SimpleFunc.measurableSet_fiber _ _ m _ _
        by_cases hx : x = 0
        · suffices g_seq_s n ⁻¹' {x} ∩ sᶜ = sᶜ by
            rw [this]
            exact hs_m₂.compl
          ext1 y
          rw [hx, Set.mem_inter_iff, Set.mem_preimage, Set.mem_singleton_iff]
          exact ⟨fun h => h.2, fun h => ⟨hg_seq_zero y h n, h⟩⟩
        · suffices g_seq_s n ⁻¹' {x} ∩ sᶜ = ∅ by
            rw [this]
            exact MeasurableSet.empty
          ext1 y
          simp only [mem_inter_iff, mem_preimage, mem_singleton_iff, mem_compl_iff,
            mem_empty_iff_false, iff_false, not_and, not_not_mem]
          refine Function.mtr fun hys => ?_
          rw [hg_seq_zero y hys n]
          exact Ne.symm hx
      finite_range' := @SimpleFunc.finite_range _ _ m (g_seq_s n) }
  /-
    case intro.intro
    α : Type u_5
    E : Type u_6
    m m₂ : MeasurableSpace α
    inst✝¹ : TopologicalSpace E
    inst✝ : Zero E
    s : Set α
    f : α → E
    hs_m : MeasurableSet s
    hs : ∀ (t : Set α), MeasurableSet (Inter.inter s t) → MeasurableSet (Inter.int …
    hf : MeasureTheory.StronglyMeasurable f
    hf_zero : ∀ (x : α), Not (Membership.mem s x) → Eq (f x) 0
    hs_m₂ : MeasurableSet s
    g_seq_s : Nat → MeasureTheory.SimpleFunc α E
    hg_seq_tendsto : ∀ (x : α), Filter.Tendsto (fun n => (g_seq_s n) x) Filter.atT …
    hg_seq_zero : ∀ (x : α), Not (Membership.mem s x) → ∀ (n : Nat), Eq ((g_seq_s  …
    g_seq_s₂ : Nat → MeasureTheory.SimpleFunc α E := fun n => { toFun := ⇑(g_seq_s …
    ⊢ MeasureTheory.StronglyMeasurable f
  -/
  exact ⟨g_seq_s₂, hg_seq_tendsto⟩
  /-
    🎉 no goals
  -/


/-- If a function `f` is strongly measurable w.r.t. a sub-σ-algebra `m` and the measure is σ-finite
on `m`, then there exists spanning measurable sets with finite measure on which `f` has bounded
norm. In particular, `f` is integrable on each of those sets. -/
theorem exists_spanning_measurableSet_norm_le [SeminormedAddCommGroup β] {m m0 : MeasurableSpace α}
    (hm : m ≤ m0) (hf : StronglyMeasurable[m] f) (μ : Measure α) [SigmaFinite (μ.trim hm)] :
    ∃ s : ℕ → Set α,
      (∀ n, MeasurableSet[m] (s n) ∧ μ (s n) < ∞ ∧ ∀ x ∈ s n, ‖f x‖ ≤ n) ∧
      ⋃ i, s i = Set.univ := by
  obtain ⟨s, hs, hs_univ⟩ :=
    @exists_spanning_measurableSet_le _ m _ hf.nnnorm.measurable (μ.trim hm) _
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    f : α → β
    inst✝¹ : SeminormedAddCommGroup β
    m m0 : MeasurableSpace α
    hm : LE.le m m0
    hf : MeasureTheory.StronglyMeasurable f
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    s : Nat → Set α
    hs : ∀ (n : Nat), And (MeasurableSet (s n)) (And (LT.lt ((μ.trim hm) (s n)) To …
    hs_univ : Eq (Set.iUnion fun i => s i) Set.univ
    ⊢ Exists fun s => And (∀ (n : Nat), And (MeasurableSet (s n)) (And (LT.lt (μ ( …
  -/
  refine ⟨s, fun n ↦ ⟨(hs n).1, (le_trim hm).trans_lt (hs n).2.1, fun x hx ↦ ?_⟩, hs_univ⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    f : α → β
    inst✝¹ : SeminormedAddCommGroup β
    m m0 : MeasurableSpace α
    hm : LE.le m m0
    hf : MeasureTheory.StronglyMeasurable f
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    s : Nat → Set α
    hs : ∀ (n : Nat), And (MeasurableSet (s n)) (And (LT.lt ((μ.trim hm) (s n)) To …
    hs_univ : Eq (Set.iUnion fun i => s i) Set.univ
    n : Nat
    x : α
    hx : Membership.mem (s n) x
    ⊢ LE.le (Norm.norm (f x)) ↑n
  -/
  have hx_nnnorm : ‖f x‖₊ ≤ n := (hs n).2.2 x hx
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    f : α → β
    inst✝¹ : SeminormedAddCommGroup β
    m m0 : MeasurableSpace α
    hm : LE.le m m0
    hf : MeasureTheory.StronglyMeasurable f
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    s : Nat → Set α
    hs : ∀ (n : Nat), And (MeasurableSet (s n)) (And (LT.lt ((μ.trim hm) (s n)) To …
    hs_univ : Eq (Set.iUnion fun i => s i) Set.univ
    n : Nat
    x : α
    hx : Membership.mem (s n) x
    hx_nnnorm : LE.le (NNNorm.nnnorm (f x)) ↑n
    ⊢ LE.le (Norm.norm (f x)) ↑n
  -/
  rw [← coe_nnnorm]
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    f : α → β
    inst✝¹ : SeminormedAddCommGroup β
    m m0 : MeasurableSpace α
    hm : LE.le m m0
    hf : MeasureTheory.StronglyMeasurable f
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    s : Nat → Set α
    hs : ∀ (n : Nat), And (MeasurableSet (s n)) (And (LT.lt ((μ.trim hm) (s n)) To …
    hs_univ : Eq (Set.iUnion fun i => s i) Set.univ
    n : Nat
    x : α
    hx : Membership.mem (s n) x
    hx_nnnorm : LE.le (NNNorm.nnnorm (f x)) ↑n
    ⊢ LE.le ↑(NNNorm.nnnorm (f x)) ↑n
  -/
  norm_cast
  /-
    🎉 no goals
  -/


theorem finStronglyMeasurable_zero {α β} {m : MeasurableSpace α} {μ : Measure α} [Zero β]
    [TopologicalSpace β] : FinStronglyMeasurable (0 : α → β) μ :=
  ⟨0, by
    simp only [Pi.zero_apply, SimpleFunc.coe_zero, support_zero', measure_empty,
      zero_lt_top, forall_const],
    fun _ => tendsto_const_nhds⟩


theorem aefinStronglyMeasurable [Zero β] [TopologicalSpace β] (hf : FinStronglyMeasurable f μ) :
    AEFinStronglyMeasurable f μ :=
  ⟨f, hf, ae_eq_refl f⟩


/-- A sequence of simple functions such that `∀ x, Tendsto (fun n ↦ hf.approx n x) atTop (𝓝 (f x))`
and `∀ n, μ (support (hf.approx n)) < ∞`. These properties are given by
`FinStronglyMeasurable.tendsto_approx` and `FinStronglyMeasurable.fin_support_approx`. -/
protected noncomputable def approx : ℕ → α →ₛ β :=
  hf.choose


protected theorem fin_support_approx : ∀ n, μ (support (hf.approx n)) < ∞ :=
  hf.choose_spec.1


protected theorem tendsto_approx : ∀ x, Tendsto (fun n => hf.approx n x) atTop (𝓝 (f x)) :=
  hf.choose_spec.2


/-- A finitely strongly measurable function is strongly measurable. -/
@[aesop 5% apply (rule_sets := [Measurable])]
protected theorem stronglyMeasurable [Zero β] [TopologicalSpace β]
    (hf : FinStronglyMeasurable f μ) : StronglyMeasurable f :=
  ⟨hf.approx, hf.tendsto_approx⟩


theorem exists_set_sigmaFinite [Zero β] [TopologicalSpace β] [T2Space β]
    (hf : FinStronglyMeasurable f μ) :
    ∃ t, MeasurableSet t ∧ (∀ x ∈ tᶜ, f x = 0) ∧ SigmaFinite (μ.restrict t) := by
  /-
    α : Type u_1
    β : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → β
    inst✝² : Zero β
    inst✝¹ : TopologicalSpace β
    inst✝ : T2Space β
    hf : MeasureTheory.FinStronglyMeasurable f μ
    ⊢ Exists fun t => And (MeasurableSet t) (And (∀ (x : α), Membership.mem (HasCo …
  -/
  rcases hf with ⟨fs, hT_lt_top, h_approx⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → β
    inst✝² : Zero β
    inst✝¹ : TopologicalSpace β
    inst✝ : T2Space β
    fs : Nat → MeasureTheory.SimpleFunc α β
    hT_lt_top : ∀ (n : Nat), LT.lt (μ (Function.support ⇑(fs n))) Top.top
    h_approx : ∀ (x : α), Filter.Tendsto (fun n => (fs n) x) Filter.atTop (nhds (f …
    ⊢ Exists fun t => And (MeasurableSet t) (And (∀ (x : α), Membership.mem (HasCo …
  -/
  let T n := support (fs n)
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → β
    inst✝² : Zero β
    inst✝¹ : TopologicalSpace β
    inst✝ : T2Space β
    fs : Nat → MeasureTheory.SimpleFunc α β
    hT_lt_top : ∀ (n : Nat), LT.lt (μ (Function.support ⇑(fs n))) Top.top
    h_approx : ∀ (x : α), Filter.Tendsto (fun n => (fs n) x) Filter.atTop (nhds (f …
    T : Nat → Set α := fun n => Function.support ⇑(fs n)
    ⊢ Exists fun t => And (MeasurableSet t) (And (∀ (x : α), Membership.mem (HasCo …
  -/
  have hT_meas : ∀ n, MeasurableSet (T n) := fun n => SimpleFunc.measurableSet_support (fs n)
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → β
    inst✝² : Zero β
    inst✝¹ : TopologicalSpace β
    inst✝ : T2Space β
    fs : Nat → MeasureTheory.SimpleFunc α β
    hT_lt_top : ∀ (n : Nat), LT.lt (μ (Function.support ⇑(fs n))) Top.top
    h_approx : ∀ (x : α), Filter.Tendsto (fun n => (fs n) x) Filter.atTop (nhds (f …
    T : Nat → Set α := fun n => Function.support ⇑(fs n)
    hT_meas : ∀ (n : Nat), MeasurableSet (T n)
    ⊢ Exists fun t => And (MeasurableSet t) (And (∀ (x : α), Membership.mem (HasCo …
  -/
  let t := ⋃ n, T n
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → β
    inst✝² : Zero β
    inst✝¹ : TopologicalSpace β
    inst✝ : T2Space β
    fs : Nat → MeasureTheory.SimpleFunc α β
    hT_lt_top : ∀ (n : Nat), LT.lt (μ (Function.support ⇑(fs n))) Top.top
    h_approx : ∀ (x : α), Filter.Tendsto (fun n => (fs n) x) Filter.atTop (nhds (f …
    T : Nat → Set α := fun n => Function.support ⇑(fs n)
    hT_meas : ∀ (n : Nat), MeasurableSet (T n)
    t : Set α := Set.iUnion fun n => T n
    ⊢ Exists fun t => And (MeasurableSet t) (And (∀ (x : α), Membership.mem (HasCo …
  -/
  refine ⟨t, MeasurableSet.iUnion hT_meas, ?_, ?_⟩
  · have h_fs_zero : ∀ n, ∀ x ∈ tᶜ, fs n x = 0 := by
      intro n x hxt
      rw [Set.mem_compl_iff, Set.mem_iUnion, not_exists] at hxt
      simpa [T] using hxt n
    /-
      case intro.intro.refine_1
      α : Type u_1
      β : Type u_2
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → β
      inst✝² : Zero β
      inst✝¹ : TopologicalSpace β
      inst✝ : T2Space β
      fs : Nat → MeasureTheory.SimpleFunc α β
      hT_lt_top : ∀ (n : Nat), LT.lt (μ (Function.support ⇑(fs n))) Top.top
      h_approx : ∀ (x : α), Filter.Tendsto (fun n => (fs n) x) Filter.atTop (nhds (f …
      T : Nat → Set α := fun n => Function.support ⇑(fs n)
      hT_meas : ∀ (n : Nat), MeasurableSet (T n)
      t : Set α := Set.iUnion fun n => T n
      h_fs_zero : ∀ (n : Nat) (x : α), Membership.mem (HasCompl.compl t) x → Eq ((fs …
      ⊢ ∀ (x : α), Membership.mem (HasCompl.compl t) x → Eq (f x) 0
    -/
    refine fun x hxt => tendsto_nhds_unique (h_approx x) ?_
    /-
      case intro.intro.refine_1
      α : Type u_1
      β : Type u_2
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → β
      inst✝² : Zero β
      inst✝¹ : TopologicalSpace β
      inst✝ : T2Space β
      fs : Nat → MeasureTheory.SimpleFunc α β
      hT_lt_top : ∀ (n : Nat), LT.lt (μ (Function.support ⇑(fs n))) Top.top
      h_approx : ∀ (x : α), Filter.Tendsto (fun n => (fs n) x) Filter.atTop (nhds (f …
      T : Nat → Set α := fun n => Function.support ⇑(fs n)
      hT_meas : ∀ (n : Nat), MeasurableSet (T n)
      t : Set α := Set.iUnion fun n => T n
      h_fs_zero : ∀ (n : Nat) (x : α), Membership.mem (HasCompl.compl t) x → Eq ((fs …
      x : α
      hxt : Membership.mem (HasCompl.compl t) x
      ⊢ Filter.Tendsto (fun n => (fs n) x) Filter.atTop (nhds 0)
    -/
    rw [funext fun n => h_fs_zero n x hxt]
    /-
      case intro.intro.refine_1
      α : Type u_1
      β : Type u_2
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → β
      inst✝² : Zero β
      inst✝¹ : TopologicalSpace β
      inst✝ : T2Space β
      fs : Nat → MeasureTheory.SimpleFunc α β
      hT_lt_top : ∀ (n : Nat), LT.lt (μ (Function.support ⇑(fs n))) Top.top
      h_approx : ∀ (x : α), Filter.Tendsto (fun n => (fs n) x) Filter.atTop (nhds (f …
      T : Nat → Set α := fun n => Function.support ⇑(fs n)
      hT_meas : ∀ (n : Nat), MeasurableSet (T n)
      t : Set α := Set.iUnion fun n => T n
      h_fs_zero : ∀ (n : Nat) (x : α), Membership.mem (HasCompl.compl t) x → Eq ((fs …
      x : α
      hxt : Membership.mem (HasCompl.compl t) x
      ⊢ Filter.Tendsto (fun n => 0) Filter.atTop (nhds 0)
    -/
    exact tendsto_const_nhds
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2
      α : Type u_1
      β : Type u_2
      m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : α → β
      inst✝² : Zero β
      inst✝¹ : TopologicalSpace β
      inst✝ : T2Space β
      fs : Nat → MeasureTheory.SimpleFunc α β
      hT_lt_top : ∀ (n : Nat), LT.lt (μ (Function.support ⇑(fs n))) Top.top
      h_approx : ∀ (x : α), Filter.Tendsto (fun n => (fs n) x) Filter.atTop (nhds (f …
      T : Nat → Set α := fun n => Function.support ⇑(fs n)
      hT_meas : ∀ (n : Nat), MeasurableSet (T n)
      t : Set α := Set.iUnion fun n => T n
      ⊢ MeasureTheory.SigmaFinite (μ.restrict t)
    -/
  · refine ⟨⟨⟨fun n => tᶜ ∪ T n, fun _ => trivial, fun n => ?_, ?_⟩⟩⟩
    · rw [Measure.restrict_apply' (MeasurableSet.iUnion hT_meas), Set.union_inter_distrib_right,
        Set.compl_inter_self t, Set.empty_union]
      /-
        case intro.intro.refine_2.refine_1
        α : Type u_1
        β : Type u_2
        m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → β
        inst✝² : Zero β
        inst✝¹ : TopologicalSpace β
        inst✝ : T2Space β
        fs : Nat → MeasureTheory.SimpleFunc α β
        hT_lt_top : ∀ (n : Nat), LT.lt (μ (Function.support ⇑(fs n))) Top.top
        h_approx : ∀ (x : α), Filter.Tendsto (fun n => (fs n) x) Filter.atTop (nhds (f …
        T : Nat → Set α := fun n => Function.support ⇑(fs n)
        hT_meas : ∀ (n : Nat), MeasurableSet (T n)
        t : Set α := Set.iUnion fun n => T n
        n : Nat
        ⊢ LT.lt (μ (Inter.inter (T n) (Set.iUnion fun b => T b))) Top.top
      -/
      exact (measure_mono Set.inter_subset_left).trans_lt (hT_lt_top n)
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.refine_2.refine_2
        α : Type u_1
        β : Type u_2
        m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → β
        inst✝² : Zero β
        inst✝¹ : TopologicalSpace β
        inst✝ : T2Space β
        fs : Nat → MeasureTheory.SimpleFunc α β
        hT_lt_top : ∀ (n : Nat), LT.lt (μ (Function.support ⇑(fs n))) Top.top
        h_approx : ∀ (x : α), Filter.Tendsto (fun n => (fs n) x) Filter.atTop (nhds (f …
        T : Nat → Set α := fun n => Function.support ⇑(fs n)
        hT_meas : ∀ (n : Nat), MeasurableSet (T n)
        t : Set α := Set.iUnion fun n => T n
        ⊢ Eq (Set.iUnion fun i => (fun n => Union.union (HasCompl.compl t) (T n)) i) S …
      -/
    · rw [← Set.union_iUnion tᶜ T]
      /-
        case intro.intro.refine_2.refine_2
        α : Type u_1
        β : Type u_2
        m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : α → β
        inst✝² : Zero β
        inst✝¹ : TopologicalSpace β
        inst✝ : T2Space β
        fs : Nat → MeasureTheory.SimpleFunc α β
        hT_lt_top : ∀ (n : Nat), LT.lt (μ (Function.support ⇑(fs n))) Top.top
        h_approx : ∀ (x : α), Filter.Tendsto (fun n => (fs n) x) Filter.atTop (nhds (f …
        T : Nat → Set α := fun n => Function.support ⇑(fs n)
        hT_meas : ∀ (n : Nat), MeasurableSet (T n)
        t : Set α := Set.iUnion fun n => T n
        ⊢ Eq (Union.union (HasCompl.compl t) (Set.iUnion fun i => T i)) Set.univ
      -/
      exact Set.compl_union_self _
      /-
        🎉 no goals
      -/


/-- A finitely strongly measurable function is measurable. -/
protected theorem measurable [Zero β] [TopologicalSpace β] [PseudoMetrizableSpace β]
    [MeasurableSpace β] [BorelSpace β] (hf : FinStronglyMeasurable f μ) : Measurable f :=
  hf.stronglyMeasurable.measurable


@[aesop safe 20 (rule_sets := [Measurable])]
protected theorem mul [MonoidWithZero β] [ContinuousMul β] (hf : FinStronglyMeasurable f μ)
    (hg : FinStronglyMeasurable g μ) : FinStronglyMeasurable (f * g) μ := by
  refine
    ⟨fun n => hf.approx n * hg.approx n, ?_, fun x =>
      (hf.tendsto_approx x).mul (hg.tendsto_approx x)⟩
  /-
    α : Type u_1
    β : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → β
    inst✝² : TopologicalSpace β
    inst✝¹ : MonoidWithZero β
    inst✝ : ContinuousMul β
    hf : MeasureTheory.FinStronglyMeasurable f μ
    hg : MeasureTheory.FinStronglyMeasurable g μ
    ⊢ ∀ (n : Nat), LT.lt (μ (Function.support ⇑((fun n => HMul.hMul (hf.approx n)  …
  -/
  intro n
  /-
    α : Type u_1
    β : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → β
    inst✝² : TopologicalSpace β
    inst✝¹ : MonoidWithZero β
    inst✝ : ContinuousMul β
    hf : MeasureTheory.FinStronglyMeasurable f μ
    hg : MeasureTheory.FinStronglyMeasurable g μ
    n : Nat
    ⊢ LT.lt (μ (Function.support ⇑((fun n => HMul.hMul (hf.approx n) (hg.approx n) …
  -/
  exact (measure_mono (support_mul_subset_left _ _)).trans_lt (hf.fin_support_approx n)
  /-
    🎉 no goals
  -/


@[aesop safe 20 (rule_sets := [Measurable])]
protected theorem add [AddMonoid β] [ContinuousAdd β] (hf : FinStronglyMeasurable f μ)
    (hg : FinStronglyMeasurable g μ) : FinStronglyMeasurable (f + g) μ :=
  ⟨fun n => hf.approx n + hg.approx n, fun n =>
    (measure_mono (Function.support_add _ _)).trans_lt
      ((measure_union_le _ _).trans_lt
        (ENNReal.add_lt_top.mpr ⟨hf.fin_support_approx n, hg.fin_support_approx n⟩)),
    fun x => (hf.tendsto_approx x).add (hg.tendsto_approx x)⟩


@[measurability]
protected theorem neg [AddGroup β] [TopologicalAddGroup β] (hf : FinStronglyMeasurable f μ) :
    FinStronglyMeasurable (-f) μ := by
  /-
    α : Type u_1
    β : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → β
    inst✝² : TopologicalSpace β
    inst✝¹ : AddGroup β
    inst✝ : TopologicalAddGroup β
    hf : MeasureTheory.FinStronglyMeasurable f μ
    ⊢ MeasureTheory.FinStronglyMeasurable (Neg.neg f) μ
  -/
  refine ⟨fun n => -hf.approx n, fun n => ?_, fun x => (hf.tendsto_approx x).neg⟩
  /-
    α : Type u_1
    β : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → β
    inst✝² : TopologicalSpace β
    inst✝¹ : AddGroup β
    inst✝ : TopologicalAddGroup β
    hf : MeasureTheory.FinStronglyMeasurable f μ
    n : Nat
    ⊢ LT.lt (μ (Function.support ⇑((fun n => Neg.neg (hf.approx n)) n))) Top.top
  -/
  suffices μ (Function.support fun x => -(hf.approx n) x) < ∞ by convert this
  /-
    α : Type u_1
    β : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → β
    inst✝² : TopologicalSpace β
    inst✝¹ : AddGroup β
    inst✝ : TopologicalAddGroup β
    hf : MeasureTheory.FinStronglyMeasurable f μ
    n : Nat
    ⊢ LT.lt (μ (Function.support fun x => Neg.neg ((hf.approx n) x))) Top.top
  -/
  rw [Function.support_neg (hf.approx n)]
  /-
    α : Type u_1
    β : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → β
    inst✝² : TopologicalSpace β
    inst✝¹ : AddGroup β
    inst✝ : TopologicalAddGroup β
    hf : MeasureTheory.FinStronglyMeasurable f μ
    n : Nat
    ⊢ LT.lt (μ (Function.support ⇑(hf.approx n))) Top.top
  -/
  exact hf.fin_support_approx n
  /-
    🎉 no goals
  -/


@[measurability]
protected theorem sub [AddGroup β] [ContinuousSub β] (hf : FinStronglyMeasurable f μ)
    (hg : FinStronglyMeasurable g μ) : FinStronglyMeasurable (f - g) μ :=
  ⟨fun n => hf.approx n - hg.approx n, fun n =>
    (measure_mono (Function.support_sub _ _)).trans_lt
      ((measure_union_le _ _).trans_lt
        (ENNReal.add_lt_top.mpr ⟨hf.fin_support_approx n, hg.fin_support_approx n⟩)),
    fun x => (hf.tendsto_approx x).sub (hg.tendsto_approx x)⟩


@[measurability]
protected theorem const_smul {𝕜} [TopologicalSpace 𝕜] [AddMonoid β] [Monoid 𝕜]
    [DistribMulAction 𝕜 β] [ContinuousSMul 𝕜 β] (hf : FinStronglyMeasurable f μ) (c : 𝕜) :
    FinStronglyMeasurable (c • f) μ := by
  /-
    α : Type u_1
    β : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → β
    inst✝⁵ : TopologicalSpace β
    𝕜 : Type u_5
    inst✝⁴ : TopologicalSpace 𝕜
    inst✝³ : AddMonoid β
    inst✝² : Monoid 𝕜
    inst✝¹ : DistribMulAction 𝕜 β
    inst✝ : ContinuousSMul 𝕜 β
    hf : MeasureTheory.FinStronglyMeasurable f μ
    c : 𝕜
    ⊢ MeasureTheory.FinStronglyMeasurable (HSMul.hSMul c f) μ
  -/
  refine ⟨fun n => c • hf.approx n, fun n => ?_, fun x => (hf.tendsto_approx x).const_smul c⟩
  /-
    α : Type u_1
    β : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → β
    inst✝⁵ : TopologicalSpace β
    𝕜 : Type u_5
    inst✝⁴ : TopologicalSpace 𝕜
    inst✝³ : AddMonoid β
    inst✝² : Monoid 𝕜
    inst✝¹ : DistribMulAction 𝕜 β
    inst✝ : ContinuousSMul 𝕜 β
    hf : MeasureTheory.FinStronglyMeasurable f μ
    c : 𝕜
    n : Nat
    ⊢ LT.lt (μ (Function.support ⇑((fun n => HSMul.hSMul c (hf.approx n)) n))) Top …
  -/
  rw [SimpleFunc.coe_smul]
  /-
    α : Type u_1
    β : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : α → β
    inst✝⁵ : TopologicalSpace β
    𝕜 : Type u_5
    inst✝⁴ : TopologicalSpace 𝕜
    inst✝³ : AddMonoid β
    inst✝² : Monoid 𝕜
    inst✝¹ : DistribMulAction 𝕜 β
    inst✝ : ContinuousSMul 𝕜 β
    hf : MeasureTheory.FinStronglyMeasurable f μ
    c : 𝕜
    n : Nat
    ⊢ LT.lt (μ (Function.support (HSMul.hSMul c ⇑(hf.approx n)))) Top.top
  -/
  exact (measure_mono (support_const_smul_subset c _)).trans_lt (hf.fin_support_approx n)
  /-
    🎉 no goals
  -/


@[aesop safe 20 (rule_sets := [Measurable])]
protected theorem sup [SemilatticeSup β] [ContinuousSup β] (hf : FinStronglyMeasurable f μ)
    (hg : FinStronglyMeasurable g μ) : FinStronglyMeasurable (f ⊔ g) μ := by
  refine
    ⟨fun n => hf.approx n ⊔ hg.approx n, fun n => ?_, fun x =>
      (hf.tendsto_approx x).sup_nhds (hg.tendsto_approx x)⟩
  /-
    α : Type u_1
    β : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → β
    inst✝³ : TopologicalSpace β
    inst✝² : Zero β
    inst✝¹ : SemilatticeSup β
    inst✝ : ContinuousSup β
    hf : MeasureTheory.FinStronglyMeasurable f μ
    hg : MeasureTheory.FinStronglyMeasurable g μ
    n : Nat
    ⊢ LT.lt (μ (Function.support ⇑((fun n => Max.max (hf.approx n) (hg.approx n))  …
  -/
  refine (measure_mono (support_sup _ _)).trans_lt ?_
  /-
    α : Type u_1
    β : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → β
    inst✝³ : TopologicalSpace β
    inst✝² : Zero β
    inst✝¹ : SemilatticeSup β
    inst✝ : ContinuousSup β
    hf : MeasureTheory.FinStronglyMeasurable f μ
    hg : MeasureTheory.FinStronglyMeasurable g μ
    n : Nat
    ⊢ LT.lt (μ (Union.union (Function.support ⇑(hf.approx n)) (Function.support ⇑( …
  -/
  exact measure_union_lt_top_iff.mpr ⟨hf.fin_support_approx n, hg.fin_support_approx n⟩
  /-
    🎉 no goals
  -/


@[aesop safe 20 (rule_sets := [Measurable])]
protected theorem inf [SemilatticeInf β] [ContinuousInf β] (hf : FinStronglyMeasurable f μ)
    (hg : FinStronglyMeasurable g μ) : FinStronglyMeasurable (f ⊓ g) μ := by
  refine
    ⟨fun n => hf.approx n ⊓ hg.approx n, fun n => ?_, fun x =>
      (hf.tendsto_approx x).inf_nhds (hg.tendsto_approx x)⟩
  /-
    α : Type u_1
    β : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → β
    inst✝³ : TopologicalSpace β
    inst✝² : Zero β
    inst✝¹ : SemilatticeInf β
    inst✝ : ContinuousInf β
    hf : MeasureTheory.FinStronglyMeasurable f μ
    hg : MeasureTheory.FinStronglyMeasurable g μ
    n : Nat
    ⊢ LT.lt (μ (Function.support ⇑((fun n => Min.min (hf.approx n) (hg.approx n))  …
  -/
  refine (measure_mono (support_inf _ _)).trans_lt ?_
  /-
    α : Type u_1
    β : Type u_2
    m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : α → β
    inst✝³ : TopologicalSpace β
    inst✝² : Zero β
    inst✝¹ : SemilatticeInf β
    inst✝ : ContinuousInf β
    hf : MeasureTheory.FinStronglyMeasurable f μ
    hg : MeasureTheory.FinStronglyMeasurable g μ
    n : Nat
    ⊢ LT.lt (μ (Union.union (Function.support ⇑(hf.approx n)) (Function.support ⇑( …
  -/
  exact measure_union_lt_top_iff.mpr ⟨hf.fin_support_approx n, hg.fin_support_approx n⟩
  /-
    🎉 no goals
  -/


theorem finStronglyMeasurable_iff_stronglyMeasurable_and_exists_set_sigmaFinite {α β} {f : α → β}
    [TopologicalSpace β] [T2Space β] [Zero β] {_ : MeasurableSpace α} {μ : Measure α} :
    FinStronglyMeasurable f μ ↔
      StronglyMeasurable f ∧
        ∃ t, MeasurableSet t ∧ (∀ x ∈ tᶜ, f x = 0) ∧ SigmaFinite (μ.restrict t) :=
  ⟨fun hf => ⟨hf.stronglyMeasurable, hf.exists_set_sigmaFinite⟩, fun hf =>
    hf.1.finStronglyMeasurable_of_set_sigmaFinite hf.2.choose_spec.1 hf.2.choose_spec.2.1
      hf.2.choose_spec.2.2⟩


theorem aefinStronglyMeasurable_zero {α β} {_ : MeasurableSpace α} (μ : Measure α) [Zero β]
    [TopologicalSpace β] : AEFinStronglyMeasurable (0 : α → β) μ :=
  ⟨0, finStronglyMeasurable_zero, EventuallyEq.rfl⟩


@[measurability]
theorem aestronglyMeasurable_const {α β} {_ : MeasurableSpace α} {μ : Measure α}
    [TopologicalSpace β] {b : β} : AEStronglyMeasurable (fun _ : α => b) μ :=
  stronglyMeasurable_const.aestronglyMeasurable


@[to_additive (attr := measurability)]
theorem aestronglyMeasurable_one {α β} {_ : MeasurableSpace α} {μ : Measure α} [TopologicalSpace β]
    [One β] : AEStronglyMeasurable (1 : α → β) μ :=
  stronglyMeasurable_one.aestronglyMeasurable


@[simp]
theorem Subsingleton.aestronglyMeasurable {_ : MeasurableSpace α} [TopologicalSpace β]
    [Subsingleton β] {μ : Measure α} (f : α → β) : AEStronglyMeasurable f μ :=
  (Subsingleton.stronglyMeasurable f).aestronglyMeasurable


@[simp]
theorem Subsingleton.aestronglyMeasurable' {_ : MeasurableSpace α} [TopologicalSpace β]
    [Subsingleton α] {μ : Measure α} (f : α → β) : AEStronglyMeasurable f μ :=
  (Subsingleton.stronglyMeasurable' f).aestronglyMeasurable


@[simp]
theorem aestronglyMeasurable_zero_measure [MeasurableSpace α] [TopologicalSpace β] (f : α → β) :
    AEStronglyMeasurable f (0 : Measure α) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    ⊢ MeasureTheory.AEStronglyMeasurable f 0
  -/
  nontriviality α
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    a✝ : Nontrivial α
    ⊢ MeasureTheory.AEStronglyMeasurable f 0
  -/
  inhabit α
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    a✝ : Nontrivial α
    inhabited_h : Inhabited α
    ⊢ MeasureTheory.AEStronglyMeasurable f 0
  -/
  exact ⟨fun _ => f default, stronglyMeasurable_const, rfl⟩
  /-
    🎉 no goals
  -/


@[measurability]
theorem SimpleFunc.aestronglyMeasurable {_ : MeasurableSpace α} {μ : Measure α} [TopologicalSpace β]
    (f : α →ₛ β) : AEStronglyMeasurable f μ :=
  f.stronglyMeasurable.aestronglyMeasurable


lemma of_finite [DiscreteMeasurableSpace α] [Finite α] : AEStronglyMeasurable f μ :=
  ⟨_, .of_finite, ae_eq_rfl⟩


/-- A `StronglyMeasurable` function such that `f =ᵐ[μ] hf.mk f`. See lemmas
`stronglyMeasurable_mk` and `ae_eq_mk`. -/
protected noncomputable def mk (f : α → β) (hf : AEStronglyMeasurable f μ) : α → β :=
  hf.choose


theorem stronglyMeasurable_mk (hf : AEStronglyMeasurable f μ) : StronglyMeasurable (hf.mk f) :=
  hf.choose_spec.1


theorem measurable_mk [PseudoMetrizableSpace β] [MeasurableSpace β] [BorelSpace β]
    (hf : AEStronglyMeasurable f μ) : Measurable (hf.mk f) :=
  hf.stronglyMeasurable_mk.measurable


theorem ae_eq_mk (hf : AEStronglyMeasurable f μ) : f =ᵐ[μ] hf.mk f :=
  hf.choose_spec.2


@[aesop 5% apply (rule_sets := [Measurable])]
protected theorem aemeasurable {β} [MeasurableSpace β] [TopologicalSpace β]
    [PseudoMetrizableSpace β] [BorelSpace β] {f : α → β} (hf : AEStronglyMeasurable f μ) :
    AEMeasurable f μ :=
  ⟨hf.mk f, hf.stronglyMeasurable_mk.measurable, hf.ae_eq_mk⟩


theorem congr (hf : AEStronglyMeasurable f μ) (h : f =ᵐ[μ] g) : AEStronglyMeasurable g μ :=
  ⟨hf.mk f, hf.stronglyMeasurable_mk, h.symm.trans hf.ae_eq_mk⟩


theorem _root_.aestronglyMeasurable_congr (h : f =ᵐ[μ] g) :
    AEStronglyMeasurable f μ ↔ AEStronglyMeasurable g μ :=
  ⟨fun hf => hf.congr h, fun hg => hg.congr h.symm⟩


theorem mono_measure {ν : Measure α} (hf : AEStronglyMeasurable f μ) (h : ν ≤ μ) :
    AEStronglyMeasurable f ν :=
  ⟨hf.mk f, hf.stronglyMeasurable_mk, Eventually.filter_mono (ae_mono h) hf.ae_eq_mk⟩


protected lemma mono_ac (h : ν ≪ μ) (hμ : AEStronglyMeasurable f μ) : AEStronglyMeasurable f ν :=
  let ⟨g, hg, hg'⟩ := hμ; ⟨g, hg, h.ae_eq hg'⟩


@[deprecated (since := "2024-02-15")] protected alias mono' := AEStronglyMeasurable.mono_ac


theorem mono_set {s t} (h : s ⊆ t) (ht : AEStronglyMeasurable f (μ.restrict t)) :
    AEStronglyMeasurable f (μ.restrict s) :=
  ht.mono_measure (restrict_mono h le_rfl)


protected theorem restrict (hfm : AEStronglyMeasurable f μ) {s} :
    AEStronglyMeasurable f (μ.restrict s) :=
  hfm.mono_measure Measure.restrict_le_self


theorem ae_mem_imp_eq_mk {s} (h : AEStronglyMeasurable f (μ.restrict s)) :
    ∀ᵐ x ∂μ, x ∈ s → f x = h.mk f x :=
  ae_imp_of_ae_restrict h.ae_eq_mk


/-- The composition of a continuous function and an ae strongly measurable function is ae strongly
measurable. -/
theorem _root_.Continuous.comp_aestronglyMeasurable {g : β → γ} {f : α → β} (hg : Continuous g)
    (hf : AEStronglyMeasurable f μ) : AEStronglyMeasurable (fun x => g (f x)) μ :=
  ⟨_, hg.comp_stronglyMeasurable hf.stronglyMeasurable_mk, EventuallyEq.fun_comp hf.ae_eq_mk g⟩


/-- A continuous function from `α` to `β` is ae strongly measurable when one of the two spaces is
second countable. -/
theorem _root_.Continuous.aestronglyMeasurable [TopologicalSpace α] [OpensMeasurableSpace α]
    [PseudoMetrizableSpace β] [SecondCountableTopologyEither α β] (hf : Continuous f) :
    AEStronglyMeasurable f μ :=
  hf.stronglyMeasurable.aestronglyMeasurable


protected theorem prod_mk {f : α → β} {g : α → γ} (hf : AEStronglyMeasurable f μ)
    (hg : AEStronglyMeasurable g μ) : AEStronglyMeasurable (fun x => (f x, g x)) μ :=
  ⟨fun x => (hf.mk f x, hg.mk g x), hf.stronglyMeasurable_mk.prod_mk hg.stronglyMeasurable_mk,
    hf.ae_eq_mk.prod_mk hg.ae_eq_mk⟩


/-- The composition of a continuous function of two variables and two ae strongly measurable
functions is ae strongly measurable. -/
theorem _root_.Continuous.comp_aestronglyMeasurable₂
    {β' : Type*} [TopologicalSpace β']
    {g : β → β' → γ} {f : α → β} {f' : α → β'} (hg : Continuous g.uncurry)
    (hf : AEStronglyMeasurable f μ) (h'f : AEStronglyMeasurable f' μ) :
    AEStronglyMeasurable (fun x => g (f x) (f' x)) μ :=
  hg.comp_aestronglyMeasurable (hf.prod_mk h'f)


/-- In a space with second countable topology, measurable implies ae strongly measurable. -/
@[fun_prop, aesop unsafe 30% apply (rule_sets := [Measurable])]
theorem _root_.Measurable.aestronglyMeasurable {_ : MeasurableSpace α} {μ : Measure α}
    [MeasurableSpace β] [PseudoMetrizableSpace β] [SecondCountableTopology β]
    [OpensMeasurableSpace β] (hf : Measurable f) : AEStronglyMeasurable f μ :=
  hf.stronglyMeasurable.aestronglyMeasurable


@[to_additive (attr := aesop safe 20 apply (rule_sets := [Measurable]))]
protected theorem mul [Mul β] [ContinuousMul β] (hf : AEStronglyMeasurable f μ)
    (hg : AEStronglyMeasurable g μ) : AEStronglyMeasurable (f * g) μ :=
  ⟨hf.mk f * hg.mk g, hf.stronglyMeasurable_mk.mul hg.stronglyMeasurable_mk,
    hf.ae_eq_mk.mul hg.ae_eq_mk⟩


@[to_additive (attr := measurability)]
protected theorem mul_const [Mul β] [ContinuousMul β] (hf : AEStronglyMeasurable f μ) (c : β) :
    AEStronglyMeasurable (fun x => f x * c) μ :=
  hf.mul aestronglyMeasurable_const


@[to_additive (attr := measurability)]
protected theorem const_mul [Mul β] [ContinuousMul β] (hf : AEStronglyMeasurable f μ) (c : β) :
    AEStronglyMeasurable (fun x => c * f x) μ :=
  aestronglyMeasurable_const.mul hf


@[to_additive (attr := measurability)]
protected theorem inv [Inv β] [ContinuousInv β] (hf : AEStronglyMeasurable f μ) :
    AEStronglyMeasurable f⁻¹ μ :=
  ⟨(hf.mk f)⁻¹, hf.stronglyMeasurable_mk.inv, hf.ae_eq_mk.inv⟩


@[to_additive (attr := aesop safe 20 apply (rule_sets := [Measurable]))]
protected theorem div [Group β] [TopologicalGroup β] (hf : AEStronglyMeasurable f μ)
    (hg : AEStronglyMeasurable g μ) : AEStronglyMeasurable (f / g) μ :=
  ⟨hf.mk f / hg.mk g, hf.stronglyMeasurable_mk.div hg.stronglyMeasurable_mk,
    hf.ae_eq_mk.div hg.ae_eq_mk⟩


@[to_additive]
theorem mul_iff_right [CommGroup β] [TopologicalGroup β] (hf : AEStronglyMeasurable f μ) :
    AEStronglyMeasurable (f * g) μ ↔ AEStronglyMeasurable g μ :=
                                   /-
                                     α : Type u_1
                                     β : Type u_2
                                     m : MeasurableSpace α
                                     μ : MeasureTheory.Measure α
                                     inst✝² : TopologicalSpace β
                                     f g : α → β
                                     inst✝¹ : CommGroup β
                                     inst✝ : TopologicalGroup β
                                     hf : MeasureTheory.AEStronglyMeasurable f μ
                                     h : MeasureTheory.AEStronglyMeasurable (HMul.hMul f g) μ
                                     ⊢ Eq g (HMul.hMul (HMul.hMul f g) (Inv.inv f))
                                   -/
  ⟨fun h ↦ show g = f * g * f⁻¹ by simp only [mul_inv_cancel_comm] ▸ h.mul hf.inv,
                                   /-
                                     🎉 no goals
                                   -/
    fun h ↦ hf.mul h⟩


@[to_additive]
theorem mul_iff_left [CommGroup β] [TopologicalGroup β] (hf : AEStronglyMeasurable f μ) :
    AEStronglyMeasurable (g * f) μ ↔ AEStronglyMeasurable g μ :=
  mul_comm g f ▸ AEStronglyMeasurable.mul_iff_right hf


@[to_additive (attr := aesop safe 20 apply (rule_sets := [Measurable]))]
protected theorem smul {𝕜} [TopologicalSpace 𝕜] [SMul 𝕜 β] [ContinuousSMul 𝕜 β] {f : α → 𝕜}
    {g : α → β} (hf : AEStronglyMeasurable f μ) (hg : AEStronglyMeasurable g μ) :
    AEStronglyMeasurable (fun x => f x • g x) μ :=
  continuous_smul.comp_aestronglyMeasurable (hf.prod_mk hg)


@[to_additive (attr := aesop safe 20 apply (rule_sets := [Measurable])) const_nsmul]
protected theorem pow [Monoid β] [ContinuousMul β] (hf : AEStronglyMeasurable f μ) (n : ℕ) :
    AEStronglyMeasurable (f ^ n) μ :=
  ⟨hf.mk f ^ n, hf.stronglyMeasurable_mk.pow _, hf.ae_eq_mk.pow_const _⟩


@[to_additive (attr := measurability)]
protected theorem const_smul {𝕜} [SMul 𝕜 β] [ContinuousConstSMul 𝕜 β]
    (hf : AEStronglyMeasurable f μ) (c : 𝕜) : AEStronglyMeasurable (c • f) μ :=
  ⟨c • hf.mk f, hf.stronglyMeasurable_mk.const_smul c, hf.ae_eq_mk.const_smul c⟩


@[to_additive (attr := measurability)]
protected theorem const_smul' {𝕜} [SMul 𝕜 β] [ContinuousConstSMul 𝕜 β]
    (hf : AEStronglyMeasurable f μ) (c : 𝕜) : AEStronglyMeasurable (fun x => c • f x) μ :=
  hf.const_smul c


@[to_additive (attr := measurability)]
protected theorem smul_const {𝕜} [TopologicalSpace 𝕜] [SMul 𝕜 β] [ContinuousSMul 𝕜 β] {f : α → 𝕜}
    (hf : AEStronglyMeasurable f μ) (c : β) : AEStronglyMeasurable (fun x => f x • c) μ :=
  continuous_smul.comp_aestronglyMeasurable (hf.prod_mk aestronglyMeasurable_const)


@[aesop safe 20 apply (rule_sets := [Measurable])]
protected theorem sup [SemilatticeSup β] [ContinuousSup β] (hf : AEStronglyMeasurable f μ)
    (hg : AEStronglyMeasurable g μ) : AEStronglyMeasurable (f ⊔ g) μ :=
  ⟨hf.mk f ⊔ hg.mk g, hf.stronglyMeasurable_mk.sup hg.stronglyMeasurable_mk,
    hf.ae_eq_mk.sup hg.ae_eq_mk⟩


@[aesop safe 20 apply (rule_sets := [Measurable])]
protected theorem inf [SemilatticeInf β] [ContinuousInf β] (hf : AEStronglyMeasurable f μ)
    (hg : AEStronglyMeasurable g μ) : AEStronglyMeasurable (f ⊓ g) μ :=
  ⟨hf.mk f ⊓ hg.mk g, hf.stronglyMeasurable_mk.inf hg.stronglyMeasurable_mk,
    hf.ae_eq_mk.inf hg.ae_eq_mk⟩


@[to_additive (attr := measurability)]
theorem _root_.List.aestronglyMeasurable_prod' (l : List (α → M))
    (hl : ∀ f ∈ l, AEStronglyMeasurable f μ) : AEStronglyMeasurable l.prod μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    M : Type u_5
    inst✝² : Monoid M
    inst✝¹ : TopologicalSpace M
    inst✝ : ContinuousMul M
    l : List (α → M)
    hl : ∀ (f : α → M), Membership.mem l f → MeasureTheory.AEStronglyMeasurable f μ
    ⊢ MeasureTheory.AEStronglyMeasurable l.prod μ
  -/
  induction' l with f l ihl; · exact aestronglyMeasurable_one
                               /-
                                 🎉 no goals
                               -/
  /-
    case cons
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    M : Type u_5
    inst✝² : Monoid M
    inst✝¹ : TopologicalSpace M
    inst✝ : ContinuousMul M
    f : α → M
    l : List (α → M)
    ihl : (∀ (f : α → M), Membership.mem l f → MeasureTheory.AEStronglyMeasurable  …
    hl : ∀ (f_1 : α → M), Membership.mem (List.cons f l) f_1 → MeasureTheory.AEStr …
    ⊢ MeasureTheory.AEStronglyMeasurable (List.cons f l).prod μ
  -/
  rw [List.forall_mem_cons] at hl
  /-
    case cons
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    M : Type u_5
    inst✝² : Monoid M
    inst✝¹ : TopologicalSpace M
    inst✝ : ContinuousMul M
    f : α → M
    l : List (α → M)
    ihl : (∀ (f : α → M), Membership.mem l f → MeasureTheory.AEStronglyMeasurable  …
    hl : And (MeasureTheory.AEStronglyMeasurable f μ) (∀ (x : α → M), Membership.m …
    ⊢ MeasureTheory.AEStronglyMeasurable (List.cons f l).prod μ
  -/
  rw [List.prod_cons]
  /-
    case cons
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    M : Type u_5
    inst✝² : Monoid M
    inst✝¹ : TopologicalSpace M
    inst✝ : ContinuousMul M
    f : α → M
    l : List (α → M)
    ihl : (∀ (f : α → M), Membership.mem l f → MeasureTheory.AEStronglyMeasurable  …
    hl : And (MeasureTheory.AEStronglyMeasurable f μ) (∀ (x : α → M), Membership.m …
    ⊢ MeasureTheory.AEStronglyMeasurable (HMul.hMul f l.prod) μ
  -/
  exact hl.1.mul (ihl hl.2)
  /-
    🎉 no goals
  -/


@[to_additive (attr := measurability)]
theorem _root_.List.aestronglyMeasurable_prod
    (l : List (α → M)) (hl : ∀ f ∈ l, AEStronglyMeasurable f μ) :
    AEStronglyMeasurable (fun x => (l.map fun f : α → M => f x).prod) μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    M : Type u_5
    inst✝² : Monoid M
    inst✝¹ : TopologicalSpace M
    inst✝ : ContinuousMul M
    l : List (α → M)
    hl : ∀ (f : α → M), Membership.mem l f → MeasureTheory.AEStronglyMeasurable f μ
    ⊢ MeasureTheory.AEStronglyMeasurable (fun x => (List.map (fun f => f x) l).pro …
  -/
  simpa only [← Pi.list_prod_apply] using l.aestronglyMeasurable_prod' hl
  /-
    🎉 no goals
  -/


@[to_additive (attr := measurability)]
theorem _root_.Multiset.aestronglyMeasurable_prod' (l : Multiset (α → M))
    (hl : ∀ f ∈ l, AEStronglyMeasurable f μ) : AEStronglyMeasurable l.prod μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    M : Type u_5
    inst✝² : CommMonoid M
    inst✝¹ : TopologicalSpace M
    inst✝ : ContinuousMul M
    l : Multiset (α → M)
    hl : ∀ (f : α → M), Membership.mem l f → MeasureTheory.AEStronglyMeasurable f μ
    ⊢ MeasureTheory.AEStronglyMeasurable l.prod μ
  -/
  rcases l with ⟨l⟩
  /-
    case mk
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    M : Type u_5
    inst✝² : CommMonoid M
    inst✝¹ : TopologicalSpace M
    inst✝ : ContinuousMul M
    l✝ : Multiset (α → M)
    l : List (α → M)
    hl : ∀ (f : α → M), Membership.mem (Quot.mk (⇑(List.isSetoid (α → M))) l) f →  …
    ⊢ MeasureTheory.AEStronglyMeasurable (Multiset.prod (Quot.mk (⇑(List.isSetoid  …
  -/
  simpa using l.aestronglyMeasurable_prod' (by simpa using hl)
  /-
    🎉 no goals
  -/


@[to_additive (attr := measurability)]
theorem _root_.Multiset.aestronglyMeasurable_prod (s : Multiset (α → M))
    (hs : ∀ f ∈ s, AEStronglyMeasurable f μ) :
    AEStronglyMeasurable (fun x => (s.map fun f : α → M => f x).prod) μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    M : Type u_5
    inst✝² : CommMonoid M
    inst✝¹ : TopologicalSpace M
    inst✝ : ContinuousMul M
    s : Multiset (α → M)
    hs : ∀ (f : α → M), Membership.mem s f → MeasureTheory.AEStronglyMeasurable f μ
    ⊢ MeasureTheory.AEStronglyMeasurable (fun x => (Multiset.map (fun f => f x) s) …
  -/
  simpa only [← Pi.multiset_prod_apply] using s.aestronglyMeasurable_prod' hs
  /-
    🎉 no goals
  -/


@[to_additive (attr := measurability)]
theorem _root_.Finset.aestronglyMeasurable_prod' {ι : Type*} {f : ι → α → M} (s : Finset ι)
    (hf : ∀ i ∈ s, AEStronglyMeasurable (f i) μ) : AEStronglyMeasurable (∏ i ∈ s, f i) μ :=
  Multiset.aestronglyMeasurable_prod' _ fun _g hg =>
    let ⟨_i, hi, hg⟩ := Multiset.mem_map.1 hg
    hg ▸ hf _ hi


@[to_additive (attr := measurability)]
theorem _root_.Finset.aestronglyMeasurable_prod {ι : Type*} {f : ι → α → M} (s : Finset ι)
    (hf : ∀ i ∈ s, AEStronglyMeasurable (f i) μ) :
    AEStronglyMeasurable (fun a => ∏ i ∈ s, f i a) μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    M : Type u_5
    inst✝² : CommMonoid M
    inst✝¹ : TopologicalSpace M
    inst✝ : ContinuousMul M
    ι : Type u_6
    f : ι → α → M
    s : Finset ι
    hf : ∀ (i : ι), Membership.mem s i → MeasureTheory.AEStronglyMeasurable (f i) μ
    ⊢ MeasureTheory.AEStronglyMeasurable (fun a => s.prod fun i => f i a) μ
  -/
  simpa only [← Finset.prod_apply] using s.aestronglyMeasurable_prod' hf
  /-
    🎉 no goals
  -/


/-- In a space with second countable topology, measurable implies strongly measurable. -/
@[aesop 90% apply (rule_sets := [Measurable])]
theorem _root_.AEMeasurable.aestronglyMeasurable [PseudoMetrizableSpace β] [OpensMeasurableSpace β]
    [SecondCountableTopology β] (hf : AEMeasurable f μ) : AEStronglyMeasurable f μ :=
  ⟨hf.mk f, hf.measurable_mk.stronglyMeasurable, hf.ae_eq_mk⟩


@[measurability]
theorem _root_.aestronglyMeasurable_id {α : Type*} [TopologicalSpace α] [PseudoMetrizableSpace α]
    {_ : MeasurableSpace α} [OpensMeasurableSpace α] [SecondCountableTopology α] {μ : Measure α} :
    AEStronglyMeasurable (id : α → α) μ :=
  aemeasurable_id.aestronglyMeasurable


/-- In a space with second countable topology, strongly measurable and measurable are equivalent. -/
theorem _root_.aestronglyMeasurable_iff_aemeasurable [PseudoMetrizableSpace β] [BorelSpace β]
    [SecondCountableTopology β] : AEStronglyMeasurable f μ ↔ AEMeasurable f μ :=
  ⟨fun h => h.aemeasurable, fun h => h.aestronglyMeasurable⟩


@[aesop safe 20 apply (rule_sets := [Measurable])]
protected theorem dist {β : Type*} [PseudoMetricSpace β] {f g : α → β}
    (hf : AEStronglyMeasurable f μ) (hg : AEStronglyMeasurable g μ) :
    AEStronglyMeasurable (fun x => dist (f x) (g x)) μ :=
  continuous_dist.comp_aestronglyMeasurable (hf.prod_mk hg)


@[measurability]
protected theorem norm {β : Type*} [SeminormedAddCommGroup β] {f : α → β}
    (hf : AEStronglyMeasurable f μ) : AEStronglyMeasurable (fun x => ‖f x‖) μ :=
  continuous_norm.comp_aestronglyMeasurable hf


@[measurability]
protected theorem nnnorm {β : Type*} [SeminormedAddCommGroup β] {f : α → β}
    (hf : AEStronglyMeasurable f μ) : AEStronglyMeasurable (fun x => ‖f x‖₊) μ :=
  continuous_nnnorm.comp_aestronglyMeasurable hf


@[measurability]
protected theorem ennnorm {β : Type*} [SeminormedAddCommGroup β] {f : α → β}
    (hf : AEStronglyMeasurable f μ) : AEMeasurable (fun a => (‖f a‖₊ : ℝ≥0∞)) μ :=
  (ENNReal.continuous_coe.comp_aestronglyMeasurable hf.nnnorm).aemeasurable


@[aesop safe 20 apply (rule_sets := [Measurable])]
protected theorem edist {β : Type*} [SeminormedAddCommGroup β] {f g : α → β}
    (hf : AEStronglyMeasurable f μ) (hg : AEStronglyMeasurable g μ) :
    AEMeasurable (fun a => edist (f a) (g a)) μ :=
  (continuous_edist.comp_aestronglyMeasurable (hf.prod_mk hg)).aemeasurable


@[measurability]
protected theorem real_toNNReal {f : α → ℝ} (hf : AEStronglyMeasurable f μ) :
    AEStronglyMeasurable (fun x => (f x).toNNReal) μ :=
  continuous_real_toNNReal.comp_aestronglyMeasurable hf


theorem _root_.aestronglyMeasurable_indicator_iff [Zero β] {s : Set α} (hs : MeasurableSet s) :
    AEStronglyMeasurable (indicator s f) μ ↔ AEStronglyMeasurable f (μ.restrict s) := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : TopologicalSpace β
    f : α → β
    inst✝ : Zero β
    s : Set α
    hs : MeasurableSet s
    ⊢ Iff (MeasureTheory.AEStronglyMeasurable (s.indicator f) μ) (MeasureTheory.AE …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : TopologicalSpace β
      f : α → β
      inst✝ : Zero β
      s : Set α
      hs : MeasurableSet s
      ⊢ MeasureTheory.AEStronglyMeasurable (s.indicator f) μ → MeasureTheory.AEStron …
    -/
  · intro h
    /-
      case mp
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : TopologicalSpace β
      f : α → β
      inst✝ : Zero β
      s : Set α
      hs : MeasurableSet s
      h : MeasureTheory.AEStronglyMeasurable (s.indicator f) μ
      ⊢ MeasureTheory.AEStronglyMeasurable f (μ.restrict s)
    -/
    exact (h.mono_measure Measure.restrict_le_self).congr (indicator_ae_eq_restrict hs)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : TopologicalSpace β
      f : α → β
      inst✝ : Zero β
      s : Set α
      hs : MeasurableSet s
      ⊢ MeasureTheory.AEStronglyMeasurable f (μ.restrict s) → MeasureTheory.AEStrong …
    -/
  · intro h
    /-
      case mpr
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : TopologicalSpace β
      f : α → β
      inst✝ : Zero β
      s : Set α
      hs : MeasurableSet s
      h : MeasureTheory.AEStronglyMeasurable f (μ.restrict s)
      ⊢ MeasureTheory.AEStronglyMeasurable (s.indicator f) μ
    -/
    refine ⟨indicator s (h.mk f), h.stronglyMeasurable_mk.indicator hs, ?_⟩
    have A : s.indicator f =ᵐ[μ.restrict s] s.indicator (h.mk f) :=
      (indicator_ae_eq_restrict hs).trans (h.ae_eq_mk.trans <| (indicator_ae_eq_restrict hs).symm)
    have B : s.indicator f =ᵐ[μ.restrict sᶜ] s.indicator (h.mk f) :=
      (indicator_ae_eq_restrict_compl hs).trans (indicator_ae_eq_restrict_compl hs).symm
    /-
      case mpr
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : TopologicalSpace β
      f : α → β
      inst✝ : Zero β
      s : Set α
      hs : MeasurableSet s
      h : MeasureTheory.AEStronglyMeasurable f (μ.restrict s)
      A : (MeasureTheory.ae (μ.restrict s)).EventuallyEq (s.indicator f) (s.indicato …
      B : (MeasureTheory.ae (μ.restrict (HasCompl.compl s))).EventuallyEq (s.indicat …
      ⊢ (MeasureTheory.ae μ).EventuallyEq (s.indicator f) (s.indicator (MeasureTheor …
    -/
    exact ae_of_ae_restrict_of_ae_restrict_compl _ A B
    /-
      🎉 no goals
    -/


@[measurability]
protected theorem indicator [Zero β] (hfm : AEStronglyMeasurable f μ) {s : Set α}
    (hs : MeasurableSet s) : AEStronglyMeasurable (s.indicator f) μ :=
  (aestronglyMeasurable_indicator_iff hs).mpr hfm.restrict


theorem nullMeasurableSet_eq_fun {E} [TopologicalSpace E] [MetrizableSpace E] {f g : α → E}
    (hf : AEStronglyMeasurable f μ) (hg : AEStronglyMeasurable g μ) :
    NullMeasurableSet { x | f x = g x } μ := by
  apply
    (hf.stronglyMeasurable_mk.measurableSet_eq_fun
          hg.stronglyMeasurable_mk).nullMeasurableSet.congr
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_5
    inst✝¹ : TopologicalSpace E
    inst✝ : TopologicalSpace.MetrizableSpace E
    f g : α → E
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hg : MeasureTheory.AEStronglyMeasurable g μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq (setOf fun x => Eq (MeasureTheory.AEStrong …
  -/
  filter_upwards [hf.ae_eq_mk, hg.ae_eq_mk] with x hfx hgx
  /-
    case h
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_5
    inst✝¹ : TopologicalSpace E
    inst✝ : TopologicalSpace.MetrizableSpace E
    f g : α → E
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hg : MeasureTheory.AEStronglyMeasurable g μ
    x : α
    hfx : Eq (f x) (MeasureTheory.AEStronglyMeasurable.mk f hf x)
    hgx : Eq (g x) (MeasureTheory.AEStronglyMeasurable.mk g hg x)
    ⊢ Eq (setOf (fun x => Eq (MeasureTheory.AEStronglyMeasurable.mk f hf x) (Measu …
  -/
  change (hf.mk f x = hg.mk g x) = (f x = g x)
  /-
    case h
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    E : Type u_5
    inst✝¹ : TopologicalSpace E
    inst✝ : TopologicalSpace.MetrizableSpace E
    f g : α → E
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hg : MeasureTheory.AEStronglyMeasurable g μ
    x : α
    hfx : Eq (f x) (MeasureTheory.AEStronglyMeasurable.mk f hf x)
    hgx : Eq (g x) (MeasureTheory.AEStronglyMeasurable.mk g hg x)
    ⊢ Eq (Eq (MeasureTheory.AEStronglyMeasurable.mk f hf x) (MeasureTheory.AEStron …
  -/
  simp only [hfx, hgx]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma nullMeasurableSet_mulSupport {E} [TopologicalSpace E] [MetrizableSpace E] [One E] {f : α → E}
    (hf : AEStronglyMeasurable f μ) : NullMeasurableSet (mulSupport f) μ :=
  (hf.nullMeasurableSet_eq_fun stronglyMeasurable_const.aestronglyMeasurable).compl


theorem nullMeasurableSet_lt [LinearOrder β] [OrderClosedTopology β] [PseudoMetrizableSpace β]
    {f g : α → β} (hf : AEStronglyMeasurable f μ) (hg : AEStronglyMeasurable g μ) :
    NullMeasurableSet { a | f a < g a } μ := by
  apply
    (hf.stronglyMeasurable_mk.measurableSet_lt hg.stronglyMeasurable_mk).nullMeasurableSet.congr
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : TopologicalSpace β
    inst✝² : LinearOrder β
    inst✝¹ : OrderClosedTopology β
    inst✝ : TopologicalSpace.PseudoMetrizableSpace β
    f g : α → β
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hg : MeasureTheory.AEStronglyMeasurable g μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq (setOf fun a => LT.lt (MeasureTheory.AEStr …
  -/
  filter_upwards [hf.ae_eq_mk, hg.ae_eq_mk] with x hfx hgx
  /-
    case h
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : TopologicalSpace β
    inst✝² : LinearOrder β
    inst✝¹ : OrderClosedTopology β
    inst✝ : TopologicalSpace.PseudoMetrizableSpace β
    f g : α → β
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hg : MeasureTheory.AEStronglyMeasurable g μ
    x : α
    hfx : Eq (f x) (MeasureTheory.AEStronglyMeasurable.mk f hf x)
    hgx : Eq (g x) (MeasureTheory.AEStronglyMeasurable.mk g hg x)
    ⊢ Eq (setOf (fun a => LT.lt (MeasureTheory.AEStronglyMeasurable.mk f hf a) (Me …
  -/
  change (hf.mk f x < hg.mk g x) = (f x < g x)
  /-
    case h
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : TopologicalSpace β
    inst✝² : LinearOrder β
    inst✝¹ : OrderClosedTopology β
    inst✝ : TopologicalSpace.PseudoMetrizableSpace β
    f g : α → β
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hg : MeasureTheory.AEStronglyMeasurable g μ
    x : α
    hfx : Eq (f x) (MeasureTheory.AEStronglyMeasurable.mk f hf x)
    hgx : Eq (g x) (MeasureTheory.AEStronglyMeasurable.mk g hg x)
    ⊢ Eq (LT.lt (MeasureTheory.AEStronglyMeasurable.mk f hf x) (MeasureTheory.AESt …
  -/
  simp only [hfx, hgx]
  /-
    🎉 no goals
  -/


theorem nullMeasurableSet_le [Preorder β] [OrderClosedTopology β] [PseudoMetrizableSpace β]
    {f g : α → β} (hf : AEStronglyMeasurable f μ) (hg : AEStronglyMeasurable g μ) :
    NullMeasurableSet { a | f a ≤ g a } μ := by
  apply
    (hf.stronglyMeasurable_mk.measurableSet_le hg.stronglyMeasurable_mk).nullMeasurableSet.congr
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : TopologicalSpace β
    inst✝² : Preorder β
    inst✝¹ : OrderClosedTopology β
    inst✝ : TopologicalSpace.PseudoMetrizableSpace β
    f g : α → β
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hg : MeasureTheory.AEStronglyMeasurable g μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq (setOf fun a => LE.le (MeasureTheory.AEStr …
  -/
  filter_upwards [hf.ae_eq_mk, hg.ae_eq_mk] with x hfx hgx
  /-
    case h
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : TopologicalSpace β
    inst✝² : Preorder β
    inst✝¹ : OrderClosedTopology β
    inst✝ : TopologicalSpace.PseudoMetrizableSpace β
    f g : α → β
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hg : MeasureTheory.AEStronglyMeasurable g μ
    x : α
    hfx : Eq (f x) (MeasureTheory.AEStronglyMeasurable.mk f hf x)
    hgx : Eq (g x) (MeasureTheory.AEStronglyMeasurable.mk g hg x)
    ⊢ Eq (setOf (fun a => LE.le (MeasureTheory.AEStronglyMeasurable.mk f hf a) (Me …
  -/
  change (hf.mk f x ≤ hg.mk g x) = (f x ≤ g x)
  /-
    case h
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : TopologicalSpace β
    inst✝² : Preorder β
    inst✝¹ : OrderClosedTopology β
    inst✝ : TopologicalSpace.PseudoMetrizableSpace β
    f g : α → β
    hf : MeasureTheory.AEStronglyMeasurable f μ
    hg : MeasureTheory.AEStronglyMeasurable g μ
    x : α
    hfx : Eq (f x) (MeasureTheory.AEStronglyMeasurable.mk f hf x)
    hgx : Eq (g x) (MeasureTheory.AEStronglyMeasurable.mk g hg x)
    ⊢ Eq (LE.le (MeasureTheory.AEStronglyMeasurable.mk f hf x) (MeasureTheory.AESt …
  -/
  simp only [hfx, hgx]
  /-
    🎉 no goals
  -/


theorem _root_.aestronglyMeasurable_of_aestronglyMeasurable_trim {α} {m m0 : MeasurableSpace α}
    {μ : Measure α} (hm : m ≤ m0) {f : α → β} (hf : AEStronglyMeasurable f (μ.trim hm)) :
    AEStronglyMeasurable f μ :=
  ⟨hf.mk f, StronglyMeasurable.mono hf.stronglyMeasurable_mk hm, ae_eq_of_ae_eq_trim hf.ae_eq_mk⟩


theorem comp_aemeasurable {γ : Type*} {_ : MeasurableSpace γ} {_ : MeasurableSpace α} {f : γ → α}
    {μ : Measure γ} (hg : AEStronglyMeasurable g (Measure.map f μ)) (hf : AEMeasurable f μ) :
    AEStronglyMeasurable (g ∘ f) μ :=
  ⟨hg.mk g ∘ hf.mk f, hg.stronglyMeasurable_mk.comp_measurable hf.measurable_mk,
    (ae_eq_comp hf hg.ae_eq_mk).trans (hf.ae_eq_mk.fun_comp (hg.mk g))⟩


theorem comp_measurable {γ : Type*} {_ : MeasurableSpace γ} {_ : MeasurableSpace α} {f : γ → α}
    {μ : Measure γ} (hg : AEStronglyMeasurable g (Measure.map f μ)) (hf : Measurable f) :
    AEStronglyMeasurable (g ∘ f) μ :=
  hg.comp_aemeasurable hf.aemeasurable


theorem comp_quasiMeasurePreserving {γ : Type*} {_ : MeasurableSpace γ} {_ : MeasurableSpace α}
    {f : γ → α} {μ : Measure γ} {ν : Measure α} (hg : AEStronglyMeasurable g ν)
    (hf : QuasiMeasurePreserving f μ ν) : AEStronglyMeasurable (g ∘ f) μ :=
  (hg.mono_ac hf.absolutelyContinuous).comp_measurable hf.measurable


theorem isSeparable_ae_range (hf : AEStronglyMeasurable f μ) :
    ∃ t : Set β, IsSeparable t ∧ ∀ᵐ x ∂μ, f x ∈ t := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : TopologicalSpace β
    f : α → β
    hf : MeasureTheory.AEStronglyMeasurable f μ
    ⊢ Exists fun t => And (TopologicalSpace.IsSeparable t) (Filter.Eventually (fun …
  -/
  refine ⟨range (hf.mk f), hf.stronglyMeasurable_mk.isSeparable_range, ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : TopologicalSpace β
    f : α → β
    hf : MeasureTheory.AEStronglyMeasurable f μ
    ⊢ Filter.Eventually (fun x => Membership.mem (Set.range (MeasureTheory.AEStron …
  -/
  filter_upwards [hf.ae_eq_mk] with x hx
  /-
    case h
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : TopologicalSpace β
    f : α → β
    hf : MeasureTheory.AEStronglyMeasurable f μ
    x : α
    hx : Eq (f x) (MeasureTheory.AEStronglyMeasurable.mk f hf x)
    ⊢ Membership.mem (Set.range (MeasureTheory.AEStronglyMeasurable.mk f hf)) (f x)
  -/
  simp [hx]
  /-
    🎉 no goals
  -/


/-- A function is almost everywhere strongly measurable if and only if it is almost everywhere
measurable, and up to a zero measure set its range is contained in a separable set. -/
theorem _root_.aestronglyMeasurable_iff_aemeasurable_separable [PseudoMetrizableSpace β]
    [MeasurableSpace β] [BorelSpace β] :
    AEStronglyMeasurable f μ ↔
      AEMeasurable f μ ∧ ∃ t : Set β, IsSeparable t ∧ ∀ᵐ x ∂μ, f x ∈ t := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : TopologicalSpace β
    f : α → β
    inst✝² : TopologicalSpace.PseudoMetrizableSpace β
    inst✝¹ : MeasurableSpace β
    inst✝ : BorelSpace β
    ⊢ Iff (MeasureTheory.AEStronglyMeasurable f μ) (And (AEMeasurable f μ) (Exists …
  -/
  refine ⟨fun H => ⟨H.aemeasurable, H.isSeparable_ae_range⟩, ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : TopologicalSpace β
    f : α → β
    inst✝² : TopologicalSpace.PseudoMetrizableSpace β
    inst✝¹ : MeasurableSpace β
    inst✝ : BorelSpace β
    ⊢ And (AEMeasurable f μ) (Exists fun t => And (TopologicalSpace.IsSeparable t) …
  -/
  rintro ⟨H, ⟨t, t_sep, ht⟩⟩
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : TopologicalSpace β
    f : α → β
    inst✝² : TopologicalSpace.PseudoMetrizableSpace β
    inst✝¹ : MeasurableSpace β
    inst✝ : BorelSpace β
    H : AEMeasurable f μ
    t : Set β
    t_sep : TopologicalSpace.IsSeparable t
    ht : Filter.Eventually (fun x => Membership.mem t (f x)) (MeasureTheory.ae μ)
    ⊢ MeasureTheory.AEStronglyMeasurable f μ
  -/
  rcases eq_empty_or_nonempty t with (rfl | h₀)
    /-
      case intro.intro.intro.inl
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝³ : TopologicalSpace β
      f : α → β
      inst✝² : TopologicalSpace.PseudoMetrizableSpace β
      inst✝¹ : MeasurableSpace β
      inst✝ : BorelSpace β
      H : AEMeasurable f μ
      t_sep : TopologicalSpace.IsSeparable EmptyCollection.emptyCollection
      ht : Filter.Eventually (fun x => Membership.mem EmptyCollection.emptyCollectio …
      ⊢ MeasureTheory.AEStronglyMeasurable f μ
    -/
  · simp only [mem_empty_iff_false, eventually_false_iff_eq_bot, ae_eq_bot] at ht
    /-
      case intro.intro.intro.inl
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝³ : TopologicalSpace β
      f : α → β
      inst✝² : TopologicalSpace.PseudoMetrizableSpace β
      inst✝¹ : MeasurableSpace β
      inst✝ : BorelSpace β
      H : AEMeasurable f μ
      t_sep : TopologicalSpace.IsSeparable EmptyCollection.emptyCollection
      ht : Eq μ 0
      ⊢ MeasureTheory.AEStronglyMeasurable f μ
    -/
    rw [ht]
    /-
      case intro.intro.intro.inl
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝³ : TopologicalSpace β
      f : α → β
      inst✝² : TopologicalSpace.PseudoMetrizableSpace β
      inst✝¹ : MeasurableSpace β
      inst✝ : BorelSpace β
      H : AEMeasurable f μ
      t_sep : TopologicalSpace.IsSeparable EmptyCollection.emptyCollection
      ht : Eq μ 0
      ⊢ MeasureTheory.AEStronglyMeasurable f 0
    -/
    exact aestronglyMeasurable_zero_measure f
    /-
      🎉 no goals
    -/
  · obtain ⟨g, g_meas, gt, fg⟩ : ∃ g : α → β, Measurable g ∧ range g ⊆ t ∧ f =ᵐ[μ] g :=
      H.exists_ae_eq_range_subset ht h₀
    /-
      case intro.intro.intro.inr.intro.intro.intro
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝³ : TopologicalSpace β
      f : α → β
      inst✝² : TopologicalSpace.PseudoMetrizableSpace β
      inst✝¹ : MeasurableSpace β
      inst✝ : BorelSpace β
      H : AEMeasurable f μ
      t : Set β
      t_sep : TopologicalSpace.IsSeparable t
      ht : Filter.Eventually (fun x => Membership.mem t (f x)) (MeasureTheory.ae μ)
      h₀ : t.Nonempty
      g : α → β
      g_meas : Measurable g
      gt : HasSubset.Subset (Set.range g) t
      fg : (MeasureTheory.ae μ).EventuallyEq f g
      ⊢ MeasureTheory.AEStronglyMeasurable f μ
    -/
    refine ⟨g, ?_, fg⟩
    /-
      case intro.intro.intro.inr.intro.intro.intro
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝³ : TopologicalSpace β
      f : α → β
      inst✝² : TopologicalSpace.PseudoMetrizableSpace β
      inst✝¹ : MeasurableSpace β
      inst✝ : BorelSpace β
      H : AEMeasurable f μ
      t : Set β
      t_sep : TopologicalSpace.IsSeparable t
      ht : Filter.Eventually (fun x => Membership.mem t (f x)) (MeasureTheory.ae μ)
      h₀ : t.Nonempty
      g : α → β
      g_meas : Measurable g
      gt : HasSubset.Subset (Set.range g) t
      fg : (MeasureTheory.ae μ).EventuallyEq f g
      ⊢ MeasureTheory.StronglyMeasurable g
    -/
    exact stronglyMeasurable_iff_measurable_separable.2 ⟨g_meas, t_sep.mono gt⟩
    /-
      🎉 no goals
    -/


theorem _root_.aestronglyMeasurable_iff_nullMeasurable_separable [PseudoMetrizableSpace β]
    [MeasurableSpace β] [BorelSpace β] :
    AEStronglyMeasurable f μ ↔
      NullMeasurable f μ ∧ ∃ t : Set β, IsSeparable t ∧ ∀ᵐ x ∂μ, f x ∈ t :=
  aestronglyMeasurable_iff_aemeasurable_separable.trans <| and_congr_left fun ⟨_, hsep, h⟩ ↦
    have := hsep.secondCountableTopology
    ⟨AEMeasurable.nullMeasurable, fun hf ↦ hf.aemeasurable_of_aerange h⟩


theorem _root_.MeasurableEmbedding.aestronglyMeasurable_map_iff {γ : Type*}
    {mγ : MeasurableSpace γ} {mα : MeasurableSpace α} {f : γ → α} {μ : Measure γ}
    (hf : MeasurableEmbedding f) {g : α → β} :
    AEStronglyMeasurable g (Measure.map f μ) ↔ AEStronglyMeasurable (g ∘ f) μ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : TopologicalSpace β
    γ : Type u_5
    mγ : MeasurableSpace γ
    mα : MeasurableSpace α
    f : γ → α
    μ : MeasureTheory.Measure γ
    hf : MeasurableEmbedding f
    g : α → β
    ⊢ Iff (MeasureTheory.AEStronglyMeasurable g (MeasureTheory.Measure.map f μ)) ( …
  -/
  refine ⟨fun H => H.comp_measurable hf.measurable, ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : TopologicalSpace β
    γ : Type u_5
    mγ : MeasurableSpace γ
    mα : MeasurableSpace α
    f : γ → α
    μ : MeasureTheory.Measure γ
    hf : MeasurableEmbedding f
    g : α → β
    ⊢ MeasureTheory.AEStronglyMeasurable (Function.comp g f) μ → MeasureTheory.AES …
  -/
  rintro ⟨g₁, hgm₁, heq⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    inst✝ : TopologicalSpace β
    γ : Type u_5
    mγ : MeasurableSpace γ
    mα : MeasurableSpace α
    f : γ → α
    μ : MeasureTheory.Measure γ
    hf : MeasurableEmbedding f
    g : α → β
    g₁ : γ → β
    hgm₁ : MeasureTheory.StronglyMeasurable g₁
    heq : (MeasureTheory.ae μ).EventuallyEq (Function.comp g f) g₁
    ⊢ MeasureTheory.AEStronglyMeasurable g (MeasureTheory.Measure.map f μ)
  -/
  rcases hf.exists_stronglyMeasurable_extend hgm₁ fun x => ⟨g x⟩ with ⟨g₂, hgm₂, rfl⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝ : TopologicalSpace β
    γ : Type u_5
    mγ : MeasurableSpace γ
    mα : MeasurableSpace α
    f : γ → α
    μ : MeasureTheory.Measure γ
    hf : MeasurableEmbedding f
    g g₂ : α → β
    hgm₂ : MeasureTheory.StronglyMeasurable g₂
    hgm₁ : MeasureTheory.StronglyMeasurable (Function.comp g₂ f)
    heq : (MeasureTheory.ae μ).EventuallyEq (Function.comp g f) (Function.comp g₂ f)
    ⊢ MeasureTheory.AEStronglyMeasurable g (MeasureTheory.Measure.map f μ)
  -/
  exact ⟨g₂, hgm₂, hf.ae_map_iff.2 heq⟩
  /-
    🎉 no goals
  -/


theorem _root_.Topology.IsEmbedding.aestronglyMeasurable_comp_iff [PseudoMetrizableSpace β]
    [PseudoMetrizableSpace γ] {g : β → γ} {f : α → β} (hg : IsEmbedding g) :
    AEStronglyMeasurable (fun x => g (f x)) μ ↔ AEStronglyMeasurable f μ := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : TopologicalSpace β
    inst✝² : TopologicalSpace γ
    inst✝¹ : TopologicalSpace.PseudoMetrizableSpace β
    inst✝ : TopologicalSpace.PseudoMetrizableSpace γ
    g : β → γ
    f : α → β
    hg : Topology.IsEmbedding g
    ⊢ Iff (MeasureTheory.AEStronglyMeasurable (fun x => g (f x)) μ) (MeasureTheory …
  -/
  letI := pseudoMetrizableSpacePseudoMetric γ
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : TopologicalSpace β
    inst✝² : TopologicalSpace γ
    inst✝¹ : TopologicalSpace.PseudoMetrizableSpace β
    inst✝ : TopologicalSpace.PseudoMetrizableSpace γ
    g : β → γ
    f : α → β
    hg : Topology.IsEmbedding g
    this : PseudoMetricSpace γ := TopologicalSpace.pseudoMetrizableSpacePseudoMetr …
    ⊢ Iff (MeasureTheory.AEStronglyMeasurable (fun x => g (f x)) μ) (MeasureTheory …
  -/
  borelize β γ
  refine
    ⟨fun H => aestronglyMeasurable_iff_aemeasurable_separable.2 ⟨?_, ?_⟩, fun H =>
      hg.continuous.comp_aestronglyMeasurable H⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝³ : TopologicalSpace β
      inst✝² : TopologicalSpace γ
      inst✝¹ : TopologicalSpace.PseudoMetrizableSpace β
      inst✝ : TopologicalSpace.PseudoMetrizableSpace γ
      g : β → γ
      f : α → β
      hg : Topology.IsEmbedding g
      this : PseudoMetricSpace γ := TopologicalSpace.pseudoMetrizableSpacePseudoMetr …
      this✝³ : MeasurableSpace β := borel β
      this✝² : BorelSpace β
      this✝¹ : MeasurableSpace γ := borel γ
      this✝ : BorelSpace γ
      H : MeasureTheory.AEStronglyMeasurable (fun x => g (f x)) μ
      ⊢ AEMeasurable f μ
    -/
  · let G : β → range g := rangeFactorization g
    have hG : IsClosedEmbedding G :=
      { hg.codRestrict _ _ with
        isClosed_range := by rw [surjective_onto_range.range_eq]; exact isClosed_univ }
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝³ : TopologicalSpace β
      inst✝² : TopologicalSpace γ
      inst✝¹ : TopologicalSpace.PseudoMetrizableSpace β
      inst✝ : TopologicalSpace.PseudoMetrizableSpace γ
      g : β → γ
      f : α → β
      hg : Topology.IsEmbedding g
      this : PseudoMetricSpace γ := TopologicalSpace.pseudoMetrizableSpacePseudoMetr …
      this✝³ : MeasurableSpace β := borel β
      this✝² : BorelSpace β
      this✝¹ : MeasurableSpace γ := borel γ
      this✝ : BorelSpace γ
      H : MeasureTheory.AEStronglyMeasurable (fun x => g (f x)) μ
      G : β → ↑(Set.range g) := Set.rangeFactorization g
      hG : Topology.IsClosedEmbedding G
      ⊢ AEMeasurable f μ
    -/
    have : AEMeasurable (G ∘ f) μ := AEMeasurable.subtype_mk H.aemeasurable
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝³ : TopologicalSpace β
      inst✝² : TopologicalSpace γ
      inst✝¹ : TopologicalSpace.PseudoMetrizableSpace β
      inst✝ : TopologicalSpace.PseudoMetrizableSpace γ
      g : β → γ
      f : α → β
      hg : Topology.IsEmbedding g
      this✝⁴ : PseudoMetricSpace γ := TopologicalSpace.pseudoMetrizableSpacePseudoMe …
      this✝³ : MeasurableSpace β := borel β
      this✝² : BorelSpace β
      this✝¹ : MeasurableSpace γ := borel γ
      this✝ : BorelSpace γ
      H : MeasureTheory.AEStronglyMeasurable (fun x => g (f x)) μ
      G : β → ↑(Set.range g) := Set.rangeFactorization g
      hG : Topology.IsClosedEmbedding G
      this : AEMeasurable (Function.comp G f) μ
      ⊢ AEMeasurable f μ
    -/
    exact hG.measurableEmbedding.aemeasurable_comp_iff.1 this
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝³ : TopologicalSpace β
      inst✝² : TopologicalSpace γ
      inst✝¹ : TopologicalSpace.PseudoMetrizableSpace β
      inst✝ : TopologicalSpace.PseudoMetrizableSpace γ
      g : β → γ
      f : α → β
      hg : Topology.IsEmbedding g
      this : PseudoMetricSpace γ := TopologicalSpace.pseudoMetrizableSpacePseudoMetr …
      this✝³ : MeasurableSpace β := borel β
      this✝² : BorelSpace β
      this✝¹ : MeasurableSpace γ := borel γ
      this✝ : BorelSpace γ
      H : MeasureTheory.AEStronglyMeasurable (fun x => g (f x)) μ
      ⊢ Exists fun t => And (TopologicalSpace.IsSeparable t) (Filter.Eventually (fun …
    -/
  · rcases (aestronglyMeasurable_iff_aemeasurable_separable.1 H).2 with ⟨t, ht, h't⟩
    /-
      case refine_2.intro.intro
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝³ : TopologicalSpace β
      inst✝² : TopologicalSpace γ
      inst✝¹ : TopologicalSpace.PseudoMetrizableSpace β
      inst✝ : TopologicalSpace.PseudoMetrizableSpace γ
      g : β → γ
      f : α → β
      hg : Topology.IsEmbedding g
      this : PseudoMetricSpace γ := TopologicalSpace.pseudoMetrizableSpacePseudoMetr …
      this✝³ : MeasurableSpace β := borel β
      this✝² : BorelSpace β
      this✝¹ : MeasurableSpace γ := borel γ
      this✝ : BorelSpace γ
      H : MeasureTheory.AEStronglyMeasurable (fun x => g (f x)) μ
      t : Set γ
      ht : TopologicalSpace.IsSeparable t
      h't : Filter.Eventually (fun x => Membership.mem t (g (f x))) (MeasureTheory.a …
      ⊢ Exists fun t => And (TopologicalSpace.IsSeparable t) (Filter.Eventually (fun …
    -/
    exact ⟨g ⁻¹' t, hg.isSeparable_preimage ht, h't⟩
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-26")]
alias _root_.Embedding.aestronglyMeasurable_comp_iff := IsEmbedding.aestronglyMeasurable_comp_iff


/-- An almost everywhere sequential limit of almost everywhere strongly measurable functions is
almost everywhere strongly measurable. -/
theorem _root_.aestronglyMeasurable_of_tendsto_ae {ι : Type*} [PseudoMetrizableSpace β]
    (u : Filter ι) [NeBot u] [IsCountablyGenerated u] {f : ι → α → β} {g : α → β}
    (hf : ∀ i, AEStronglyMeasurable (f i) μ) (lim : ∀ᵐ x ∂μ, Tendsto (fun n => f n x) u (𝓝 (g x))) :
    AEStronglyMeasurable g μ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : TopologicalSpace β
    ι : Type u_5
    inst✝² : TopologicalSpace.PseudoMetrizableSpace β
    u : Filter ι
    inst✝¹ : u.NeBot
    inst✝ : u.IsCountablyGenerated
    f : ι → α → β
    g : α → β
    hf : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
    lim : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) u (nhds (g x …
    ⊢ MeasureTheory.AEStronglyMeasurable g μ
  -/
  borelize β
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝³ : TopologicalSpace β
    ι : Type u_5
    inst✝² : TopologicalSpace.PseudoMetrizableSpace β
    u : Filter ι
    inst✝¹ : u.NeBot
    inst✝ : u.IsCountablyGenerated
    f : ι → α → β
    g : α → β
    hf : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
    lim : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) u (nhds (g x …
    this✝¹ : MeasurableSpace β := borel β
    this✝ : BorelSpace β
    ⊢ MeasureTheory.AEStronglyMeasurable g μ
  -/
  refine aestronglyMeasurable_iff_aemeasurable_separable.2 ⟨?_, ?_⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝³ : TopologicalSpace β
      ι : Type u_5
      inst✝² : TopologicalSpace.PseudoMetrizableSpace β
      u : Filter ι
      inst✝¹ : u.NeBot
      inst✝ : u.IsCountablyGenerated
      f : ι → α → β
      g : α → β
      hf : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
      lim : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) u (nhds (g x …
      this✝¹ : MeasurableSpace β := borel β
      this✝ : BorelSpace β
      ⊢ AEMeasurable g μ
    -/
  · exact aemeasurable_of_tendsto_metrizable_ae _ (fun n => (hf n).aemeasurable) lim
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝³ : TopologicalSpace β
      ι : Type u_5
      inst✝² : TopologicalSpace.PseudoMetrizableSpace β
      u : Filter ι
      inst✝¹ : u.NeBot
      inst✝ : u.IsCountablyGenerated
      f : ι → α → β
      g : α → β
      hf : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
      lim : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) u (nhds (g x …
      this✝¹ : MeasurableSpace β := borel β
      this✝ : BorelSpace β
      ⊢ Exists fun t => And (TopologicalSpace.IsSeparable t) (Filter.Eventually (fun …
    -/
  · rcases u.exists_seq_tendsto with ⟨v, hv⟩
    have : ∀ n : ℕ, ∃ t : Set β, IsSeparable t ∧ f (v n) ⁻¹' t ∈ ae μ := fun n =>
      (aestronglyMeasurable_iff_aemeasurable_separable.1 (hf (v n))).2
    /-
      case refine_2.intro
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝³ : TopologicalSpace β
      ι : Type u_5
      inst✝² : TopologicalSpace.PseudoMetrizableSpace β
      u : Filter ι
      inst✝¹ : u.NeBot
      inst✝ : u.IsCountablyGenerated
      f : ι → α → β
      g : α → β
      hf : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
      lim : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) u (nhds (g x …
      this✝¹ : MeasurableSpace β := borel β
      this✝ : BorelSpace β
      v : Nat → ι
      hv : Filter.Tendsto v Filter.atTop u
      this : ∀ (n : Nat), Exists fun t => And (TopologicalSpace.IsSeparable t) (Memb …
      ⊢ Exists fun t => And (TopologicalSpace.IsSeparable t) (Filter.Eventually (fun …
    -/
    choose t t_sep ht using this
    /-
      case refine_2.intro
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝³ : TopologicalSpace β
      ι : Type u_5
      inst✝² : TopologicalSpace.PseudoMetrizableSpace β
      u : Filter ι
      inst✝¹ : u.NeBot
      inst✝ : u.IsCountablyGenerated
      f : ι → α → β
      g : α → β
      hf : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
      lim : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) u (nhds (g x …
      this✝¹ : MeasurableSpace β := borel β
      this✝ : BorelSpace β
      v : Nat → ι
      hv : Filter.Tendsto v Filter.atTop u
      t : Nat → Set β
      t_sep : ∀ (n : Nat), TopologicalSpace.IsSeparable (t n)
      ht : ∀ (n : Nat), Membership.mem (MeasureTheory.ae μ) (Set.preimage (f (v n))  …
      ⊢ Exists fun t => And (TopologicalSpace.IsSeparable t) (Filter.Eventually (fun …
    -/
    refine ⟨closure (⋃ i, t i), .closure <| .iUnion t_sep, ?_⟩
    /-
      case refine_2.intro
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝³ : TopologicalSpace β
      ι : Type u_5
      inst✝² : TopologicalSpace.PseudoMetrizableSpace β
      u : Filter ι
      inst✝¹ : u.NeBot
      inst✝ : u.IsCountablyGenerated
      f : ι → α → β
      g : α → β
      hf : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
      lim : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) u (nhds (g x …
      this✝¹ : MeasurableSpace β := borel β
      this✝ : BorelSpace β
      v : Nat → ι
      hv : Filter.Tendsto v Filter.atTop u
      t : Nat → Set β
      t_sep : ∀ (n : Nat), TopologicalSpace.IsSeparable (t n)
      ht : ∀ (n : Nat), Membership.mem (MeasureTheory.ae μ) (Set.preimage (f (v n))  …
      ⊢ Filter.Eventually (fun x => Membership.mem (closure (Set.iUnion fun i => t i …
    -/
    filter_upwards [ae_all_iff.2 ht, lim] with x hx h'x
    /-
      case h
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝³ : TopologicalSpace β
      ι : Type u_5
      inst✝² : TopologicalSpace.PseudoMetrizableSpace β
      u : Filter ι
      inst✝¹ : u.NeBot
      inst✝ : u.IsCountablyGenerated
      f : ι → α → β
      g : α → β
      hf : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
      lim : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) u (nhds (g x …
      this✝¹ : MeasurableSpace β := borel β
      this✝ : BorelSpace β
      v : Nat → ι
      hv : Filter.Tendsto v Filter.atTop u
      t : Nat → Set β
      t_sep : ∀ (n : Nat), TopologicalSpace.IsSeparable (t n)
      ht : ∀ (n : Nat), Membership.mem (MeasureTheory.ae μ) (Set.preimage (f (v n))  …
      x : α
      hx : ∀ (i : Nat), Membership.mem (t i) (f (v i) x)
      h'x : Filter.Tendsto (fun n => f n x) u (nhds (g x))
      ⊢ Membership.mem (closure (Set.iUnion fun i => t i)) (g x)
    -/
    apply mem_closure_of_tendsto (h'x.comp hv)
    /-
      case h
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝³ : TopologicalSpace β
      ι : Type u_5
      inst✝² : TopologicalSpace.PseudoMetrizableSpace β
      u : Filter ι
      inst✝¹ : u.NeBot
      inst✝ : u.IsCountablyGenerated
      f : ι → α → β
      g : α → β
      hf : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable (f i) μ
      lim : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) u (nhds (g x …
      this✝¹ : MeasurableSpace β := borel β
      this✝ : BorelSpace β
      v : Nat → ι
      hv : Filter.Tendsto v Filter.atTop u
      t : Nat → Set β
      t_sep : ∀ (n : Nat), TopologicalSpace.IsSeparable (t n)
      ht : ∀ (n : Nat), Membership.mem (MeasureTheory.ae μ) (Set.preimage (f (v n))  …
      x : α
      hx : ∀ (i : Nat), Membership.mem (t i) (f (v i) x)
      h'x : Filter.Tendsto (fun n => f n x) u (nhds (g x))
      ⊢ Filter.Eventually (fun x_1 => Membership.mem (Set.iUnion fun i => t i) (Func …
    -/
    filter_upwards with n using mem_iUnion_of_mem n (hx n)
    /-
      🎉 no goals
    -/


/-- If a sequence of almost everywhere strongly measurable functions converges almost everywhere,
one can select a strongly measurable function as the almost everywhere limit. -/
theorem _root_.exists_stronglyMeasurable_limit_of_tendsto_ae [PseudoMetrizableSpace β]
    {f : ℕ → α → β} (hf : ∀ n, AEStronglyMeasurable (f n) μ)
    (h_ae_tendsto : ∀ᵐ x ∂μ, ∃ l : β, Tendsto (fun n => f n x) atTop (𝓝 l)) :
    ∃ f_lim : α → β, StronglyMeasurable f_lim ∧
      ∀ᵐ x ∂μ, Tendsto (fun n => f n x) atTop (𝓝 (f_lim x)) := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace.PseudoMetrizableSpace β
    f : Nat → α → β
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    h_ae_tendsto : Filter.Eventually (fun x => Exists fun l => Filter.Tendsto (fun …
    ⊢ Exists fun f_lim => And (MeasureTheory.StronglyMeasurable f_lim) (Filter.Eve …
  -/
  borelize β
  obtain ⟨g, _, hg⟩ :
    ∃ g : α → β, Measurable g ∧ ∀ᵐ x ∂μ, Tendsto (fun n => f n x) atTop (𝓝 (g x)) :=
    measurable_limit_of_tendsto_metrizable_ae (fun n => (hf n).aemeasurable) h_ae_tendsto
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace.PseudoMetrizableSpace β
    f : Nat → α → β
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    h_ae_tendsto : Filter.Eventually (fun x => Exists fun l => Filter.Tendsto (fun …
    this✝¹ : MeasurableSpace β := borel β
    this✝ : BorelSpace β
    g : α → β
    left✝ : Measurable g
    hg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop  …
    ⊢ Exists fun f_lim => And (MeasureTheory.StronglyMeasurable f_lim) (Filter.Eve …
  -/
  have Hg : AEStronglyMeasurable g μ := aestronglyMeasurable_of_tendsto_ae _ hf hg
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace.PseudoMetrizableSpace β
    f : Nat → α → β
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    h_ae_tendsto : Filter.Eventually (fun x => Exists fun l => Filter.Tendsto (fun …
    this✝¹ : MeasurableSpace β := borel β
    this✝ : BorelSpace β
    g : α → β
    left✝ : Measurable g
    hg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop  …
    Hg : MeasureTheory.AEStronglyMeasurable g μ
    ⊢ Exists fun f_lim => And (MeasureTheory.StronglyMeasurable f_lim) (Filter.Eve …
  -/
  refine ⟨Hg.mk g, Hg.stronglyMeasurable_mk, ?_⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace.PseudoMetrizableSpace β
    f : Nat → α → β
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    h_ae_tendsto : Filter.Eventually (fun x => Exists fun l => Filter.Tendsto (fun …
    this✝¹ : MeasurableSpace β := borel β
    this✝ : BorelSpace β
    g : α → β
    left✝ : Measurable g
    hg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop  …
    Hg : MeasureTheory.AEStronglyMeasurable g μ
    ⊢ Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop (nh …
  -/
  filter_upwards [hg, Hg.ae_eq_mk] with x hx h'x
  /-
    case h
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace.PseudoMetrizableSpace β
    f : Nat → α → β
    hf : ∀ (n : Nat), MeasureTheory.AEStronglyMeasurable (f n) μ
    h_ae_tendsto : Filter.Eventually (fun x => Exists fun l => Filter.Tendsto (fun …
    this✝¹ : MeasurableSpace β := borel β
    this✝ : BorelSpace β
    g : α → β
    left✝ : Measurable g
    hg : Filter.Eventually (fun x => Filter.Tendsto (fun n => f n x) Filter.atTop  …
    Hg : MeasureTheory.AEStronglyMeasurable g μ
    x : α
    hx : Filter.Tendsto (fun n => f n x) Filter.atTop (nhds (g x))
    h'x : Eq (g x) (MeasureTheory.AEStronglyMeasurable.mk g Hg x)
    ⊢ Filter.Tendsto (fun n => f n x) Filter.atTop (nhds (MeasureTheory.AEStrongly …
  -/
  rwa [h'x] at hx
  /-
    🎉 no goals
  -/


theorem piecewise {s : Set α} [DecidablePred (· ∈ s)]
    (hs : MeasurableSet s) (hf : AEStronglyMeasurable f (μ.restrict s))
    (hg : AEStronglyMeasurable g (μ.restrict sᶜ)) :
    AEStronglyMeasurable (s.piecewise f g) μ := by
  refine ⟨s.piecewise (hf.mk f) (hg.mk g),
    StronglyMeasurable.piecewise hs hf.stronglyMeasurable_mk hg.stronglyMeasurable_mk, ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : TopologicalSpace β
    f g : α → β
    s : Set α
    inst✝ : DecidablePred fun x => Membership.mem s x
    hs : MeasurableSet s
    hf : MeasureTheory.AEStronglyMeasurable f (μ.restrict s)
    hg : MeasureTheory.AEStronglyMeasurable g (μ.restrict (HasCompl.compl s))
    ⊢ (MeasureTheory.ae μ).EventuallyEq (s.piecewise f g) (s.piecewise (MeasureThe …
  -/
  refine ae_of_ae_restrict_of_ae_restrict_compl s ?_ ?_
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : TopologicalSpace β
      f g : α → β
      s : Set α
      inst✝ : DecidablePred fun x => Membership.mem s x
      hs : MeasurableSet s
      hf : MeasureTheory.AEStronglyMeasurable f (μ.restrict s)
      hg : MeasureTheory.AEStronglyMeasurable g (μ.restrict (HasCompl.compl s))
      ⊢ Filter.Eventually (fun x => Eq (s.piecewise f g x) (s.piecewise (MeasureTheo …
    -/
  · have h := hf.ae_eq_mk
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : TopologicalSpace β
      f g : α → β
      s : Set α
      inst✝ : DecidablePred fun x => Membership.mem s x
      hs : MeasurableSet s
      hf : MeasureTheory.AEStronglyMeasurable f (μ.restrict s)
      hg : MeasureTheory.AEStronglyMeasurable g (μ.restrict (HasCompl.compl s))
      h : (MeasureTheory.ae (μ.restrict s)).EventuallyEq f (MeasureTheory.AEStrongly …
      ⊢ Filter.Eventually (fun x => Eq (s.piecewise f g x) (s.piecewise (MeasureTheo …
    -/
    rw [Filter.EventuallyEq, ae_restrict_iff' hs] at h
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : TopologicalSpace β
      f g : α → β
      s : Set α
      inst✝ : DecidablePred fun x => Membership.mem s x
      hs : MeasurableSet s
      hf : MeasureTheory.AEStronglyMeasurable f (μ.restrict s)
      hg : MeasureTheory.AEStronglyMeasurable g (μ.restrict (HasCompl.compl s))
      h : Filter.Eventually (fun x => Membership.mem s x → Eq (f x) (MeasureTheory.A …
      ⊢ Filter.Eventually (fun x => Eq (s.piecewise f g x) (s.piecewise (MeasureTheo …
    -/
    rw [ae_restrict_iff' hs]
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : TopologicalSpace β
      f g : α → β
      s : Set α
      inst✝ : DecidablePred fun x => Membership.mem s x
      hs : MeasurableSet s
      hf : MeasureTheory.AEStronglyMeasurable f (μ.restrict s)
      hg : MeasureTheory.AEStronglyMeasurable g (μ.restrict (HasCompl.compl s))
      h : Filter.Eventually (fun x => Membership.mem s x → Eq (f x) (MeasureTheory.A …
      ⊢ Filter.Eventually (fun x => Membership.mem s x → Eq (s.piecewise f g x) (s.p …
    -/
    filter_upwards [h] with x hx
    /-
      case h
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : TopologicalSpace β
      f g : α → β
      s : Set α
      inst✝ : DecidablePred fun x => Membership.mem s x
      hs : MeasurableSet s
      hf : MeasureTheory.AEStronglyMeasurable f (μ.restrict s)
      hg : MeasureTheory.AEStronglyMeasurable g (μ.restrict (HasCompl.compl s))
      h : Filter.Eventually (fun x => Membership.mem s x → Eq (f x) (MeasureTheory.A …
      x : α
      hx : Membership.mem s x → Eq (f x) (MeasureTheory.AEStronglyMeasurable.mk f hf …
      ⊢ Membership.mem s x → Eq (s.piecewise f g x) (s.piecewise (MeasureTheory.AESt …
    -/
    intro hx_mem
    /-
      case h
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : TopologicalSpace β
      f g : α → β
      s : Set α
      inst✝ : DecidablePred fun x => Membership.mem s x
      hs : MeasurableSet s
      hf : MeasureTheory.AEStronglyMeasurable f (μ.restrict s)
      hg : MeasureTheory.AEStronglyMeasurable g (μ.restrict (HasCompl.compl s))
      h : Filter.Eventually (fun x => Membership.mem s x → Eq (f x) (MeasureTheory.A …
      x : α
      hx : Membership.mem s x → Eq (f x) (MeasureTheory.AEStronglyMeasurable.mk f hf …
      hx_mem : Membership.mem s x
      ⊢ Eq (s.piecewise f g x) (s.piecewise (MeasureTheory.AEStronglyMeasurable.mk f …
    -/
    simp only [hx_mem, Set.piecewise_eq_of_mem, hx hx_mem]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : TopologicalSpace β
      f g : α → β
      s : Set α
      inst✝ : DecidablePred fun x => Membership.mem s x
      hs : MeasurableSet s
      hf : MeasureTheory.AEStronglyMeasurable f (μ.restrict s)
      hg : MeasureTheory.AEStronglyMeasurable g (μ.restrict (HasCompl.compl s))
      ⊢ Filter.Eventually (fun x => Eq (s.piecewise f g x) (s.piecewise (MeasureTheo …
    -/
  · have h := hg.ae_eq_mk
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : TopologicalSpace β
      f g : α → β
      s : Set α
      inst✝ : DecidablePred fun x => Membership.mem s x
      hs : MeasurableSet s
      hf : MeasureTheory.AEStronglyMeasurable f (μ.restrict s)
      hg : MeasureTheory.AEStronglyMeasurable g (μ.restrict (HasCompl.compl s))
      h : (MeasureTheory.ae (μ.restrict (HasCompl.compl s))).EventuallyEq g (Measure …
      ⊢ Filter.Eventually (fun x => Eq (s.piecewise f g x) (s.piecewise (MeasureTheo …
    -/
    rw [Filter.EventuallyEq, ae_restrict_iff' hs.compl] at h
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : TopologicalSpace β
      f g : α → β
      s : Set α
      inst✝ : DecidablePred fun x => Membership.mem s x
      hs : MeasurableSet s
      hf : MeasureTheory.AEStronglyMeasurable f (μ.restrict s)
      hg : MeasureTheory.AEStronglyMeasurable g (μ.restrict (HasCompl.compl s))
      h : Filter.Eventually (fun x => Membership.mem (HasCompl.compl s) x → Eq (g x) …
      ⊢ Filter.Eventually (fun x => Eq (s.piecewise f g x) (s.piecewise (MeasureTheo …
    -/
    rw [ae_restrict_iff' hs.compl]
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : TopologicalSpace β
      f g : α → β
      s : Set α
      inst✝ : DecidablePred fun x => Membership.mem s x
      hs : MeasurableSet s
      hf : MeasureTheory.AEStronglyMeasurable f (μ.restrict s)
      hg : MeasureTheory.AEStronglyMeasurable g (μ.restrict (HasCompl.compl s))
      h : Filter.Eventually (fun x => Membership.mem (HasCompl.compl s) x → Eq (g x) …
      ⊢ Filter.Eventually (fun x => Membership.mem (HasCompl.compl s) x → Eq (s.piec …
    -/
    filter_upwards [h] with x hx
    /-
      case h
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : TopologicalSpace β
      f g : α → β
      s : Set α
      inst✝ : DecidablePred fun x => Membership.mem s x
      hs : MeasurableSet s
      hf : MeasureTheory.AEStronglyMeasurable f (μ.restrict s)
      hg : MeasureTheory.AEStronglyMeasurable g (μ.restrict (HasCompl.compl s))
      h : Filter.Eventually (fun x => Membership.mem (HasCompl.compl s) x → Eq (g x) …
      x : α
      hx : Membership.mem (HasCompl.compl s) x → Eq (g x) (MeasureTheory.AEStronglyM …
      ⊢ Membership.mem (HasCompl.compl s) x → Eq (s.piecewise f g x) (s.piecewise (M …
    -/
    intro hx_mem
    /-
      case h
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : TopologicalSpace β
      f g : α → β
      s : Set α
      inst✝ : DecidablePred fun x => Membership.mem s x
      hs : MeasurableSet s
      hf : MeasureTheory.AEStronglyMeasurable f (μ.restrict s)
      hg : MeasureTheory.AEStronglyMeasurable g (μ.restrict (HasCompl.compl s))
      h : Filter.Eventually (fun x => Membership.mem (HasCompl.compl s) x → Eq (g x) …
      x : α
      hx : Membership.mem (HasCompl.compl s) x → Eq (g x) (MeasureTheory.AEStronglyM …
      hx_mem : Membership.mem (HasCompl.compl s) x
      ⊢ Eq (s.piecewise f g x) (s.piecewise (MeasureTheory.AEStronglyMeasurable.mk f …
    -/
    rw [Set.mem_compl_iff] at hx_mem
    /-
      case h
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝¹ : TopologicalSpace β
      f g : α → β
      s : Set α
      inst✝ : DecidablePred fun x => Membership.mem s x
      hs : MeasurableSet s
      hf : MeasureTheory.AEStronglyMeasurable f (μ.restrict s)
      hg : MeasureTheory.AEStronglyMeasurable g (μ.restrict (HasCompl.compl s))
      h : Filter.Eventually (fun x => Membership.mem (HasCompl.compl s) x → Eq (g x) …
      x : α
      hx : Membership.mem (HasCompl.compl s) x → Eq (g x) (MeasureTheory.AEStronglyM …
      hx_mem : Not (Membership.mem s x)
      ⊢ Eq (s.piecewise f g x) (s.piecewise (MeasureTheory.AEStronglyMeasurable.mk f …
    -/
    simp only [hx_mem, not_false_eq_true, Set.piecewise_eq_of_not_mem, hx hx_mem]
    /-
      🎉 no goals
    -/


theorem sum_measure [PseudoMetrizableSpace β] {m : MeasurableSpace α} {μ : ι → Measure α}
    (h : ∀ i, AEStronglyMeasurable f (μ i)) : AEStronglyMeasurable f (Measure.sum μ) := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_4
    inst✝² : Countable ι
    inst✝¹ : TopologicalSpace β
    f : α → β
    inst✝ : TopologicalSpace.PseudoMetrizableSpace β
    m : MeasurableSpace α
    μ : ι → MeasureTheory.Measure α
    h : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable f (μ i)
    ⊢ MeasureTheory.AEStronglyMeasurable f (MeasureTheory.Measure.sum μ)
  -/
  borelize β
  refine
    aestronglyMeasurable_iff_aemeasurable_separable.2
      ⟨AEMeasurable.sum_measure fun i => (h i).aemeasurable, ?_⟩
  have A : ∀ i : ι, ∃ t : Set β, IsSeparable t ∧ f ⁻¹' t ∈ ae (μ i) := fun i =>
    (aestronglyMeasurable_iff_aemeasurable_separable.1 (h i)).2
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_4
    inst✝² : Countable ι
    inst✝¹ : TopologicalSpace β
    f : α → β
    inst✝ : TopologicalSpace.PseudoMetrizableSpace β
    m : MeasurableSpace α
    μ : ι → MeasureTheory.Measure α
    h : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable f (μ i)
    this✝¹ : MeasurableSpace β := borel β
    this✝ : BorelSpace β
    A : ∀ (i : ι), Exists fun t => And (TopologicalSpace.IsSeparable t) (Membershi …
    ⊢ Exists fun t => And (TopologicalSpace.IsSeparable t) (Filter.Eventually (fun …
  -/
  choose t t_sep ht using A
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_4
    inst✝² : Countable ι
    inst✝¹ : TopologicalSpace β
    f : α → β
    inst✝ : TopologicalSpace.PseudoMetrizableSpace β
    m : MeasurableSpace α
    μ : ι → MeasureTheory.Measure α
    h : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable f (μ i)
    this✝¹ : MeasurableSpace β := borel β
    this✝ : BorelSpace β
    t : ι → Set β
    t_sep : ∀ (i : ι), TopologicalSpace.IsSeparable (t i)
    ht : ∀ (i : ι), Membership.mem (MeasureTheory.ae (μ i)) (Set.preimage f (t i))
    ⊢ Exists fun t => And (TopologicalSpace.IsSeparable t) (Filter.Eventually (fun …
  -/
  refine ⟨⋃ i, t i, .iUnion t_sep, ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_4
    inst✝² : Countable ι
    inst✝¹ : TopologicalSpace β
    f : α → β
    inst✝ : TopologicalSpace.PseudoMetrizableSpace β
    m : MeasurableSpace α
    μ : ι → MeasureTheory.Measure α
    h : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable f (μ i)
    this✝¹ : MeasurableSpace β := borel β
    this✝ : BorelSpace β
    t : ι → Set β
    t_sep : ∀ (i : ι), TopologicalSpace.IsSeparable (t i)
    ht : ∀ (i : ι), Membership.mem (MeasureTheory.ae (μ i)) (Set.preimage f (t i))
    ⊢ Filter.Eventually (fun x => Membership.mem (Set.iUnion fun i => t i) (f x))  …
  -/
  simp only [Measure.ae_sum_eq, mem_iUnion, eventually_iSup]
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_4
    inst✝² : Countable ι
    inst✝¹ : TopologicalSpace β
    f : α → β
    inst✝ : TopologicalSpace.PseudoMetrizableSpace β
    m : MeasurableSpace α
    μ : ι → MeasureTheory.Measure α
    h : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable f (μ i)
    this✝¹ : MeasurableSpace β := borel β
    this✝ : BorelSpace β
    t : ι → Set β
    t_sep : ∀ (i : ι), TopologicalSpace.IsSeparable (t i)
    ht : ∀ (i : ι), Membership.mem (MeasureTheory.ae (μ i)) (Set.preimage f (t i))
    ⊢ ∀ (b : ι), Filter.Eventually (fun x => Exists fun i => Membership.mem (t i)  …
  -/
  intro i
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_4
    inst✝² : Countable ι
    inst✝¹ : TopologicalSpace β
    f : α → β
    inst✝ : TopologicalSpace.PseudoMetrizableSpace β
    m : MeasurableSpace α
    μ : ι → MeasureTheory.Measure α
    h : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable f (μ i)
    this✝¹ : MeasurableSpace β := borel β
    this✝ : BorelSpace β
    t : ι → Set β
    t_sep : ∀ (i : ι), TopologicalSpace.IsSeparable (t i)
    ht : ∀ (i : ι), Membership.mem (MeasureTheory.ae (μ i)) (Set.preimage f (t i))
    i : ι
    ⊢ Filter.Eventually (fun x => Exists fun i => Membership.mem (t i) (f x)) (Mea …
  -/
  filter_upwards [ht i] with x hx
  /-
    case h
    α : Type u_1
    β : Type u_2
    ι : Type u_4
    inst✝² : Countable ι
    inst✝¹ : TopologicalSpace β
    f : α → β
    inst✝ : TopologicalSpace.PseudoMetrizableSpace β
    m : MeasurableSpace α
    μ : ι → MeasureTheory.Measure α
    h : ∀ (i : ι), MeasureTheory.AEStronglyMeasurable f (μ i)
    this✝¹ : MeasurableSpace β := borel β
    this✝ : BorelSpace β
    t : ι → Set β
    t_sep : ∀ (i : ι), TopologicalSpace.IsSeparable (t i)
    ht : ∀ (i : ι), Membership.mem (MeasureTheory.ae (μ i)) (Set.preimage f (t i))
    i : ι
    x : α
    hx : Membership.mem (Set.preimage f (t i)) x
    ⊢ Exists fun i => Membership.mem (t i) (f x)
  -/
  exact ⟨i, hx⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem _root_.aestronglyMeasurable_sum_measure_iff [PseudoMetrizableSpace β]
    {_m : MeasurableSpace α} {μ : ι → Measure α} :
    AEStronglyMeasurable f (sum μ) ↔ ∀ i, AEStronglyMeasurable f (μ i) :=
  ⟨fun h _ => h.mono_measure (Measure.le_sum _ _), sum_measure⟩


@[simp]
theorem _root_.aestronglyMeasurable_add_measure_iff [PseudoMetrizableSpace β] {ν : Measure α} :
    AEStronglyMeasurable f (μ + ν) ↔ AEStronglyMeasurable f μ ∧ AEStronglyMeasurable f ν := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : TopologicalSpace β
    f : α → β
    inst✝ : TopologicalSpace.PseudoMetrizableSpace β
    ν : MeasureTheory.Measure α
    ⊢ Iff (MeasureTheory.AEStronglyMeasurable f (HAdd.hAdd μ ν)) (And (MeasureTheo …
  -/
  rw [← sum_cond, aestronglyMeasurable_sum_measure_iff, Bool.forall_bool, and_comm]
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : TopologicalSpace β
    f : α → β
    inst✝ : TopologicalSpace.PseudoMetrizableSpace β
    ν : MeasureTheory.Measure α
    ⊢ Iff (And (MeasureTheory.AEStronglyMeasurable f (cond Bool.true μ ν)) (Measur …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[measurability]
theorem add_measure [PseudoMetrizableSpace β] {ν : Measure α} {f : α → β}
    (hμ : AEStronglyMeasurable f μ) (hν : AEStronglyMeasurable f ν) :
    AEStronglyMeasurable f (μ + ν) :=
  aestronglyMeasurable_add_measure_iff.2 ⟨hμ, hν⟩


@[measurability]
protected theorem iUnion [PseudoMetrizableSpace β] {s : ι → Set α}
    (h : ∀ i, AEStronglyMeasurable f (μ.restrict (s i))) :
    AEStronglyMeasurable f (μ.restrict (⋃ i, s i)) :=
  (sum_measure h).mono_measure <| restrict_iUnion_le


@[simp]
theorem _root_.aestronglyMeasurable_iUnion_iff [PseudoMetrizableSpace β] {s : ι → Set α} :
    AEStronglyMeasurable f (μ.restrict (⋃ i, s i)) ↔
      ∀ i, AEStronglyMeasurable f (μ.restrict (s i)) :=
  ⟨fun h _ => h.mono_measure <| restrict_mono (subset_iUnion _ _) le_rfl,
    AEStronglyMeasurable.iUnion⟩


@[simp]
theorem _root_.aestronglyMeasurable_union_iff [PseudoMetrizableSpace β] {s t : Set α} :
    AEStronglyMeasurable f (μ.restrict (s ∪ t)) ↔
      AEStronglyMeasurable f (μ.restrict s) ∧ AEStronglyMeasurable f (μ.restrict t) := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : TopologicalSpace β
    f : α → β
    inst✝ : TopologicalSpace.PseudoMetrizableSpace β
    s t : Set α
    ⊢ Iff (MeasureTheory.AEStronglyMeasurable f (μ.restrict (Union.union s t))) (A …
  -/
  simp only [union_eq_iUnion, aestronglyMeasurable_iUnion_iff, Bool.forall_bool, cond, and_comm]
  /-
    🎉 no goals
  -/


theorem aestronglyMeasurable_uIoc_iff [LinearOrder α] [PseudoMetrizableSpace β] {f : α → β}
    {a b : α} :
    AEStronglyMeasurable f (μ.restrict <| uIoc a b) ↔
      AEStronglyMeasurable f (μ.restrict <| Ioc a b) ∧
        AEStronglyMeasurable f (μ.restrict <| Ioc b a) := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace β
    inst✝¹ : LinearOrder α
    inst✝ : TopologicalSpace.PseudoMetrizableSpace β
    f : α → β
    a b : α
    ⊢ Iff (MeasureTheory.AEStronglyMeasurable f (μ.restrict (Set.uIoc a b))) (And  …
  -/
  rw [uIoc_eq_union, aestronglyMeasurable_union_iff]
  /-
    🎉 no goals
  -/


@[measurability]
theorem smul_measure {R : Type*} [Monoid R] [DistribMulAction R ℝ≥0∞] [IsScalarTower R ℝ≥0∞ ℝ≥0∞]
    (h : AEStronglyMeasurable f μ) (c : R) : AEStronglyMeasurable f (c • μ) :=
  ⟨h.mk f, h.stronglyMeasurable_mk, ae_smul_measure h.ae_eq_mk c⟩


theorem _root_.aestronglyMeasurable_const_smul_iff (c : G) :
    AEStronglyMeasurable (fun x => c • f x) μ ↔ AEStronglyMeasurable f μ :=
               /-
                 α : Type u_1
                 β : Type u_2
                 m : MeasurableSpace α
                 μ : MeasureTheory.Measure α
                 inst✝³ : TopologicalSpace β
                 f : α → β
                 G : Type u_6
                 inst✝² : Group G
                 inst✝¹ : MulAction G β
                 inst✝ : ContinuousConstSMul G β
                 c : G
                 h : MeasureTheory.AEStronglyMeasurable (fun x => HSMul.hSMul c (f x)) μ
                 ⊢ MeasureTheory.AEStronglyMeasurable f μ
               -/
  ⟨fun h => by simpa only [inv_smul_smul] using h.const_smul' c⁻¹, fun h => h.const_smul c⟩
               /-
                 🎉 no goals
               -/


nonrec theorem _root_.IsUnit.aestronglyMeasurable_const_smul_iff {c : M} (hc : IsUnit c) :
    AEStronglyMeasurable (fun x => c • f x) μ ↔ AEStronglyMeasurable f μ :=
  let ⟨u, hu⟩ := hc
  hu ▸ aestronglyMeasurable_const_smul_iff u


theorem _root_.aestronglyMeasurable_const_smul_iff₀ {c : G₀} (hc : c ≠ 0) :
    AEStronglyMeasurable (fun x => c • f x) μ ↔ AEStronglyMeasurable f μ :=
  (IsUnit.mk0 _ hc).aestronglyMeasurable_const_smul_iff


/-- A `fin_strongly_measurable` function such that `f =ᵐ[μ] hf.mk f`. See lemmas
`fin_strongly_measurable_mk` and `ae_eq_mk`. -/
protected noncomputable def mk (f : α → β) (hf : AEFinStronglyMeasurable f μ) : α → β :=
  hf.choose


theorem finStronglyMeasurable_mk (hf : AEFinStronglyMeasurable f μ) :
    FinStronglyMeasurable (hf.mk f) μ :=
  hf.choose_spec.1


theorem ae_eq_mk (hf : AEFinStronglyMeasurable f μ) : f =ᵐ[μ] hf.mk f :=
  hf.choose_spec.2


@[aesop 10% apply (rule_sets := [Measurable])]
protected theorem aemeasurable {β} [Zero β] [MeasurableSpace β] [TopologicalSpace β]
    [PseudoMetrizableSpace β] [BorelSpace β] {f : α → β} (hf : AEFinStronglyMeasurable f μ) :
    AEMeasurable f μ :=
  ⟨hf.mk f, hf.finStronglyMeasurable_mk.measurable, hf.ae_eq_mk⟩


@[aesop safe 20 (rule_sets := [Measurable])]
protected theorem mul [MonoidWithZero β] [ContinuousMul β] (hf : AEFinStronglyMeasurable f μ)
    (hg : AEFinStronglyMeasurable g μ) : AEFinStronglyMeasurable (f * g) μ :=
  ⟨hf.mk f * hg.mk g, hf.finStronglyMeasurable_mk.mul hg.finStronglyMeasurable_mk,
    hf.ae_eq_mk.mul hg.ae_eq_mk⟩


@[aesop safe 20 (rule_sets := [Measurable])]
protected theorem add [AddMonoid β] [ContinuousAdd β] (hf : AEFinStronglyMeasurable f μ)
    (hg : AEFinStronglyMeasurable g μ) : AEFinStronglyMeasurable (f + g) μ :=
  ⟨hf.mk f + hg.mk g, hf.finStronglyMeasurable_mk.add hg.finStronglyMeasurable_mk,
    hf.ae_eq_mk.add hg.ae_eq_mk⟩


@[measurability]
protected theorem neg [AddGroup β] [TopologicalAddGroup β] (hf : AEFinStronglyMeasurable f μ) :
    AEFinStronglyMeasurable (-f) μ :=
  ⟨-hf.mk f, hf.finStronglyMeasurable_mk.neg, hf.ae_eq_mk.neg⟩


@[measurability]
protected theorem sub [AddGroup β] [ContinuousSub β] (hf : AEFinStronglyMeasurable f μ)
    (hg : AEFinStronglyMeasurable g μ) : AEFinStronglyMeasurable (f - g) μ :=
  ⟨hf.mk f - hg.mk g, hf.finStronglyMeasurable_mk.sub hg.finStronglyMeasurable_mk,
    hf.ae_eq_mk.sub hg.ae_eq_mk⟩


@[measurability]
protected theorem const_smul {𝕜} [TopologicalSpace 𝕜] [AddMonoid β] [Monoid 𝕜]
    [DistribMulAction 𝕜 β] [ContinuousSMul 𝕜 β] (hf : AEFinStronglyMeasurable f μ) (c : 𝕜) :
    AEFinStronglyMeasurable (c • f) μ :=
  ⟨c • hf.mk f, hf.finStronglyMeasurable_mk.const_smul c, hf.ae_eq_mk.const_smul c⟩


@[aesop safe 20 (rule_sets := [Measurable])]
protected theorem sup [SemilatticeSup β] [ContinuousSup β] (hf : AEFinStronglyMeasurable f μ)
    (hg : AEFinStronglyMeasurable g μ) : AEFinStronglyMeasurable (f ⊔ g) μ :=
  ⟨hf.mk f ⊔ hg.mk g, hf.finStronglyMeasurable_mk.sup hg.finStronglyMeasurable_mk,
    hf.ae_eq_mk.sup hg.ae_eq_mk⟩


@[aesop safe 20 (rule_sets := [Measurable])]
protected theorem inf [SemilatticeInf β] [ContinuousInf β] (hf : AEFinStronglyMeasurable f μ)
    (hg : AEFinStronglyMeasurable g μ) : AEFinStronglyMeasurable (f ⊓ g) μ :=
  ⟨hf.mk f ⊓ hg.mk g, hf.finStronglyMeasurable_mk.inf hg.finStronglyMeasurable_mk,
    hf.ae_eq_mk.inf hg.ae_eq_mk⟩


theorem exists_set_sigmaFinite (hf : AEFinStronglyMeasurable f μ) :
    ∃ t, MeasurableSet t ∧ f =ᵐ[μ.restrict tᶜ] 0 ∧ SigmaFinite (μ.restrict t) := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace β
    f : α → β
    inst✝¹ : Zero β
    inst✝ : T2Space β
    hf : MeasureTheory.AEFinStronglyMeasurable f μ
    ⊢ Exists fun t => And (MeasurableSet t) (And ((MeasureTheory.ae (μ.restrict (H …
  -/
  rcases hf with ⟨g, hg, hfg⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace β
    f : α → β
    inst✝¹ : Zero β
    inst✝ : T2Space β
    g : α → β
    hg : MeasureTheory.FinStronglyMeasurable g μ
    hfg : (MeasureTheory.ae μ).EventuallyEq f g
    ⊢ Exists fun t => And (MeasurableSet t) (And ((MeasureTheory.ae (μ.restrict (H …
  -/
  obtain ⟨t, ht, hgt_zero, htμ⟩ := hg.exists_set_sigmaFinite
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace β
    f : α → β
    inst✝¹ : Zero β
    inst✝ : T2Space β
    g : α → β
    hg : MeasureTheory.FinStronglyMeasurable g μ
    hfg : (MeasureTheory.ae μ).EventuallyEq f g
    t : Set α
    ht : MeasurableSet t
    hgt_zero : ∀ (x : α), Membership.mem (HasCompl.compl t) x → Eq (g x) 0
    htμ : MeasureTheory.SigmaFinite (μ.restrict t)
    ⊢ Exists fun t => And (MeasurableSet t) (And ((MeasureTheory.ae (μ.restrict (H …
  -/
  refine ⟨t, ht, ?_, htμ⟩
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace β
    f : α → β
    inst✝¹ : Zero β
    inst✝ : T2Space β
    g : α → β
    hg : MeasureTheory.FinStronglyMeasurable g μ
    hfg : (MeasureTheory.ae μ).EventuallyEq f g
    t : Set α
    ht : MeasurableSet t
    hgt_zero : ∀ (x : α), Membership.mem (HasCompl.compl t) x → Eq (g x) 0
    htμ : MeasureTheory.SigmaFinite (μ.restrict t)
    ⊢ (MeasureTheory.ae (μ.restrict (HasCompl.compl t))).EventuallyEq f 0
  -/
  refine EventuallyEq.trans (ae_restrict_of_ae hfg) ?_
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace β
    f : α → β
    inst✝¹ : Zero β
    inst✝ : T2Space β
    g : α → β
    hg : MeasureTheory.FinStronglyMeasurable g μ
    hfg : (MeasureTheory.ae μ).EventuallyEq f g
    t : Set α
    ht : MeasurableSet t
    hgt_zero : ∀ (x : α), Membership.mem (HasCompl.compl t) x → Eq (g x) 0
    htμ : MeasureTheory.SigmaFinite (μ.restrict t)
    ⊢ (MeasureTheory.ae (μ.restrict (HasCompl.compl t))).EventuallyEq g 0
  -/
  rw [EventuallyEq, ae_restrict_iff' ht.compl]
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace β
    f : α → β
    inst✝¹ : Zero β
    inst✝ : T2Space β
    g : α → β
    hg : MeasureTheory.FinStronglyMeasurable g μ
    hfg : (MeasureTheory.ae μ).EventuallyEq f g
    t : Set α
    ht : MeasurableSet t
    hgt_zero : ∀ (x : α), Membership.mem (HasCompl.compl t) x → Eq (g x) 0
    htμ : MeasureTheory.SigmaFinite (μ.restrict t)
    ⊢ Filter.Eventually (fun x => Membership.mem (HasCompl.compl t) x → Eq (g x) ( …
  -/
  exact Eventually.of_forall hgt_zero
  /-
    🎉 no goals
  -/


/-- A measurable set `t` such that `f =ᵐ[μ.restrict tᶜ] 0` and `sigma_finite (μ.restrict t)`. -/
def sigmaFiniteSet (hf : AEFinStronglyMeasurable f μ) : Set α :=
  hf.exists_set_sigmaFinite.choose


protected theorem measurableSet (hf : AEFinStronglyMeasurable f μ) :
    MeasurableSet hf.sigmaFiniteSet :=
  hf.exists_set_sigmaFinite.choose_spec.1


theorem ae_eq_zero_compl (hf : AEFinStronglyMeasurable f μ) :
    f =ᵐ[μ.restrict hf.sigmaFiniteSetᶜ] 0 :=
  hf.exists_set_sigmaFinite.choose_spec.2.1


instance sigmaFinite_restrict (hf : AEFinStronglyMeasurable f μ) :
    SigmaFinite (μ.restrict hf.sigmaFiniteSet) :=
  hf.exists_set_sigmaFinite.choose_spec.2.2


/-- In a space with second countable topology and a sigma-finite measure, `FinStronglyMeasurable`
  and `Measurable` are equivalent. -/
theorem finStronglyMeasurable_iff_measurable {_m0 : MeasurableSpace α} (μ : Measure α)
    [SigmaFinite μ] : FinStronglyMeasurable f μ ↔ Measurable f :=
  ⟨fun h => h.measurable, fun h => (Measurable.stronglyMeasurable h).finStronglyMeasurable μ⟩


/-- In a space with second countable topology and a sigma-finite measure, a measurable function
is `FinStronglyMeasurable`. -/
@[aesop 90% apply (rule_sets := [Measurable])]
theorem finStronglyMeasurable_of_measurable {_m0 : MeasurableSpace α} (μ : Measure α)
    [SigmaFinite μ] (hf : Measurable f) : FinStronglyMeasurable f μ :=
  (finStronglyMeasurable_iff_measurable μ).mpr hf


/-- In a space with second countable topology and a sigma-finite measure,
  `AEFinStronglyMeasurable` and `AEMeasurable` are equivalent. -/
theorem aefinStronglyMeasurable_iff_aemeasurable {_m0 : MeasurableSpace α} (μ : Measure α)
    [SigmaFinite μ] : AEFinStronglyMeasurable f μ ↔ AEMeasurable f μ := by
  /-
    α : Type u_1
    G : Type u_5
    inst✝⁴ : SeminormedAddCommGroup G
    inst✝³ : MeasurableSpace G
    inst✝² : BorelSpace G
    inst✝¹ : SecondCountableTopology G
    f : α → G
    _m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    ⊢ Iff (MeasureTheory.AEFinStronglyMeasurable f μ) (AEMeasurable f μ)
  -/
  simp_rw [AEFinStronglyMeasurable, AEMeasurable, finStronglyMeasurable_iff_measurable]
  /-
    🎉 no goals
  -/


/-- In a space with second countable topology and a sigma-finite measure,
  an `AEMeasurable` function is `AEFinStronglyMeasurable`. -/
@[aesop 90% apply (rule_sets := [Measurable])]
theorem aefinStronglyMeasurable_of_aemeasurable {_m0 : MeasurableSpace α} (μ : Measure α)
    [SigmaFinite μ] (hf : AEMeasurable f μ) : AEFinStronglyMeasurable f μ :=
  (aefinStronglyMeasurable_iff_aemeasurable μ).mpr hf


theorem measurable_uncurry_of_continuous_of_measurable {α β ι : Type*} [TopologicalSpace ι]
    [MetrizableSpace ι] [MeasurableSpace ι] [SecondCountableTopology ι] [OpensMeasurableSpace ι]
    {mβ : MeasurableSpace β} [TopologicalSpace β] [PseudoMetrizableSpace β] [BorelSpace β]
    {m : MeasurableSpace α} {u : ι → α → β} (hu_cont : ∀ x, Continuous fun i => u i x)
    (h : ∀ i, Measurable (u i)) : Measurable (Function.uncurry u) := by
  obtain ⟨t_sf, ht_sf⟩ :
    ∃ t : ℕ → SimpleFunc ι ι, ∀ j x, Tendsto (fun n => u (t n j) x) atTop (𝓝 <| u j x) := by
    have h_str_meas : StronglyMeasurable (id : ι → ι) := stronglyMeasurable_id
    refine ⟨h_str_meas.approx, fun j x => ?_⟩
    exact ((hu_cont x).tendsto j).comp (h_str_meas.tendsto_approx j)
  /-
    case intro
    α : Type u_5
    β : Type u_6
    ι : Type u_7
    inst✝⁷ : TopologicalSpace ι
    inst✝⁶ : TopologicalSpace.MetrizableSpace ι
    inst✝⁵ : MeasurableSpace ι
    inst✝⁴ : SecondCountableTopology ι
    inst✝³ : OpensMeasurableSpace ι
    mβ : MeasurableSpace β
    inst✝² : TopologicalSpace β
    inst✝¹ : TopologicalSpace.PseudoMetrizableSpace β
    inst✝ : BorelSpace β
    m : MeasurableSpace α
    u : ι → α → β
    hu_cont : ∀ (x : α), Continuous fun i => u i x
    h : ∀ (i : ι), Measurable (u i)
    t_sf : Nat → MeasureTheory.SimpleFunc ι ι
    ht_sf : ∀ (j : ι) (x : α), Filter.Tendsto (fun n => u ((t_sf n) j) x) Filter.a …
    ⊢ Measurable (Function.uncurry u)
  -/
  let U (n : ℕ) (p : ι × α) := u (t_sf n p.fst) p.snd
  have h_tendsto : Tendsto U atTop (𝓝 fun p => u p.fst p.snd) := by
    rw [tendsto_pi_nhds]
    exact fun p => ht_sf p.fst p.snd
  /-
    case intro
    α : Type u_5
    β : Type u_6
    ι : Type u_7
    inst✝⁷ : TopologicalSpace ι
    inst✝⁶ : TopologicalSpace.MetrizableSpace ι
    inst✝⁵ : MeasurableSpace ι
    inst✝⁴ : SecondCountableTopology ι
    inst✝³ : OpensMeasurableSpace ι
    mβ : MeasurableSpace β
    inst✝² : TopologicalSpace β
    inst✝¹ : TopologicalSpace.PseudoMetrizableSpace β
    inst✝ : BorelSpace β
    m : MeasurableSpace α
    u : ι → α → β
    hu_cont : ∀ (x : α), Continuous fun i => u i x
    h : ∀ (i : ι), Measurable (u i)
    t_sf : Nat → MeasureTheory.SimpleFunc ι ι
    ht_sf : ∀ (j : ι) (x : α), Filter.Tendsto (fun n => u ((t_sf n) j) x) Filter.a …
    U : Nat → Prod ι α → β := fun n p => u ((t_sf n) p.1) p.2
    h_tendsto : Filter.Tendsto U Filter.atTop (nhds fun p => u p.1 p.2)
    ⊢ Measurable (Function.uncurry u)
  -/
  refine measurable_of_tendsto_metrizable (fun n => ?_) h_tendsto
  have h_meas : Measurable fun p : (t_sf n).range × α => u (↑p.fst) p.snd := by
    have :
      (fun p : ↥(t_sf n).range × α => u (↑p.fst) p.snd) =
        (fun p : α × (t_sf n).range => u (↑p.snd) p.fst) ∘ Prod.swap :=
      rfl
    rw [this, @measurable_swap_iff α (↥(t_sf n).range) β m]
    exact measurable_from_prod_countable fun j => h j
  have :
    (fun p : ι × α => u (t_sf n p.fst) p.snd) =
      (fun p : ↥(t_sf n).range × α => u p.fst p.snd) ∘ fun p : ι × α =>
        (⟨t_sf n p.fst, SimpleFunc.mem_range_self _ _⟩, p.snd) :=
    rfl
  /-
    case intro
    α : Type u_5
    β : Type u_6
    ι : Type u_7
    inst✝⁷ : TopologicalSpace ι
    inst✝⁶ : TopologicalSpace.MetrizableSpace ι
    inst✝⁵ : MeasurableSpace ι
    inst✝⁴ : SecondCountableTopology ι
    inst✝³ : OpensMeasurableSpace ι
    mβ : MeasurableSpace β
    inst✝² : TopologicalSpace β
    inst✝¹ : TopologicalSpace.PseudoMetrizableSpace β
    inst✝ : BorelSpace β
    m : MeasurableSpace α
    u : ι → α → β
    hu_cont : ∀ (x : α), Continuous fun i => u i x
    h : ∀ (i : ι), Measurable (u i)
    t_sf : Nat → MeasureTheory.SimpleFunc ι ι
    ht_sf : ∀ (j : ι) (x : α), Filter.Tendsto (fun n => u ((t_sf n) j) x) Filter.a …
    U : Nat → Prod ι α → β := fun n p => u ((t_sf n) p.1) p.2
    h_tendsto : Filter.Tendsto U Filter.atTop (nhds fun p => u p.1 p.2)
    n : Nat
    h_meas : Measurable fun p => u (↑p.1) p.2
    this : Eq (fun p => u ((t_sf n) p.1) p.2) (Function.comp (fun p => u (↑p.1) p. …
    ⊢ Measurable (U n)
  -/
  simp_rw [U, this]
  /-
    case intro
    α : Type u_5
    β : Type u_6
    ι : Type u_7
    inst✝⁷ : TopologicalSpace ι
    inst✝⁶ : TopologicalSpace.MetrizableSpace ι
    inst✝⁵ : MeasurableSpace ι
    inst✝⁴ : SecondCountableTopology ι
    inst✝³ : OpensMeasurableSpace ι
    mβ : MeasurableSpace β
    inst✝² : TopologicalSpace β
    inst✝¹ : TopologicalSpace.PseudoMetrizableSpace β
    inst✝ : BorelSpace β
    m : MeasurableSpace α
    u : ι → α → β
    hu_cont : ∀ (x : α), Continuous fun i => u i x
    h : ∀ (i : ι), Measurable (u i)
    t_sf : Nat → MeasureTheory.SimpleFunc ι ι
    ht_sf : ∀ (j : ι) (x : α), Filter.Tendsto (fun n => u ((t_sf n) j) x) Filter.a …
    U : Nat → Prod ι α → β := fun n p => u ((t_sf n) p.1) p.2
    h_tendsto : Filter.Tendsto U Filter.atTop (nhds fun p => u p.1 p.2)
    n : Nat
    h_meas : Measurable fun p => u (↑p.1) p.2
    this : Eq (fun p => u ((t_sf n) p.1) p.2) (Function.comp (fun p => u (↑p.1) p. …
    ⊢ Measurable (Function.comp (fun p => u (↑p.1) p.2) fun p => { fst := ⟨(t_sf n …
  -/
  refine h_meas.comp (Measurable.prod_mk ?_ measurable_snd)
  /-
    case intro
    α : Type u_5
    β : Type u_6
    ι : Type u_7
    inst✝⁷ : TopologicalSpace ι
    inst✝⁶ : TopologicalSpace.MetrizableSpace ι
    inst✝⁵ : MeasurableSpace ι
    inst✝⁴ : SecondCountableTopology ι
    inst✝³ : OpensMeasurableSpace ι
    mβ : MeasurableSpace β
    inst✝² : TopologicalSpace β
    inst✝¹ : TopologicalSpace.PseudoMetrizableSpace β
    inst✝ : BorelSpace β
    m : MeasurableSpace α
    u : ι → α → β
    hu_cont : ∀ (x : α), Continuous fun i => u i x
    h : ∀ (i : ι), Measurable (u i)
    t_sf : Nat → MeasureTheory.SimpleFunc ι ι
    ht_sf : ∀ (j : ι) (x : α), Filter.Tendsto (fun n => u ((t_sf n) j) x) Filter.a …
    U : Nat → Prod ι α → β := fun n p => u ((t_sf n) p.1) p.2
    h_tendsto : Filter.Tendsto U Filter.atTop (nhds fun p => u p.1 p.2)
    n : Nat
    h_meas : Measurable fun p => u (↑p.1) p.2
    this : Eq (fun p => u ((t_sf n) p.1) p.2) (Function.comp (fun p => u (↑p.1) p. …
    ⊢ Measurable fun p => ⟨(t_sf n) p.1, ⋯⟩
  -/
  exact ((t_sf n).measurable.comp measurable_fst).subtype_mk
  /-
    🎉 no goals
  -/


theorem stronglyMeasurable_uncurry_of_continuous_of_stronglyMeasurable {α β ι : Type*}
    [TopologicalSpace ι] [MetrizableSpace ι] [MeasurableSpace ι] [SecondCountableTopology ι]
    [OpensMeasurableSpace ι] [TopologicalSpace β] [PseudoMetrizableSpace β] [MeasurableSpace α]
    {u : ι → α → β} (hu_cont : ∀ x, Continuous fun i => u i x) (h : ∀ i, StronglyMeasurable (u i)) :
    StronglyMeasurable (Function.uncurry u) := by
  /-
    α : Type u_5
    β : Type u_6
    ι : Type u_7
    inst✝⁷ : TopologicalSpace ι
    inst✝⁶ : TopologicalSpace.MetrizableSpace ι
    inst✝⁵ : MeasurableSpace ι
    inst✝⁴ : SecondCountableTopology ι
    inst✝³ : OpensMeasurableSpace ι
    inst✝² : TopologicalSpace β
    inst✝¹ : TopologicalSpace.PseudoMetrizableSpace β
    inst✝ : MeasurableSpace α
    u : ι → α → β
    hu_cont : ∀ (x : α), Continuous fun i => u i x
    h : ∀ (i : ι), MeasureTheory.StronglyMeasurable (u i)
    ⊢ MeasureTheory.StronglyMeasurable (Function.uncurry u)
  -/
  borelize β
  obtain ⟨t_sf, ht_sf⟩ :
    ∃ t : ℕ → SimpleFunc ι ι, ∀ j x, Tendsto (fun n => u (t n j) x) atTop (𝓝 <| u j x) := by
    have h_str_meas : StronglyMeasurable (id : ι → ι) := stronglyMeasurable_id
    refine ⟨h_str_meas.approx, fun j x => ?_⟩
    exact ((hu_cont x).tendsto j).comp (h_str_meas.tendsto_approx j)
  /-
    case intro
    α : Type u_5
    β : Type u_6
    ι : Type u_7
    inst✝⁷ : TopologicalSpace ι
    inst✝⁶ : TopologicalSpace.MetrizableSpace ι
    inst✝⁵ : MeasurableSpace ι
    inst✝⁴ : SecondCountableTopology ι
    inst✝³ : OpensMeasurableSpace ι
    inst✝² : TopologicalSpace β
    inst✝¹ : TopologicalSpace.PseudoMetrizableSpace β
    inst✝ : MeasurableSpace α
    u : ι → α → β
    hu_cont : ∀ (x : α), Continuous fun i => u i x
    h : ∀ (i : ι), MeasureTheory.StronglyMeasurable (u i)
    this✝¹ : MeasurableSpace β := borel β
    this✝ : BorelSpace β
    t_sf : Nat → MeasureTheory.SimpleFunc ι ι
    ht_sf : ∀ (j : ι) (x : α), Filter.Tendsto (fun n => u ((t_sf n) j) x) Filter.a …
    ⊢ MeasureTheory.StronglyMeasurable (Function.uncurry u)
  -/
  let U (n : ℕ) (p : ι × α) := u (t_sf n p.fst) p.snd
  have h_tendsto : Tendsto U atTop (𝓝 fun p => u p.fst p.snd) := by
    rw [tendsto_pi_nhds]
    exact fun p => ht_sf p.fst p.snd
  /-
    case intro
    α : Type u_5
    β : Type u_6
    ι : Type u_7
    inst✝⁷ : TopologicalSpace ι
    inst✝⁶ : TopologicalSpace.MetrizableSpace ι
    inst✝⁵ : MeasurableSpace ι
    inst✝⁴ : SecondCountableTopology ι
    inst✝³ : OpensMeasurableSpace ι
    inst✝² : TopologicalSpace β
    inst✝¹ : TopologicalSpace.PseudoMetrizableSpace β
    inst✝ : MeasurableSpace α
    u : ι → α → β
    hu_cont : ∀ (x : α), Continuous fun i => u i x
    h : ∀ (i : ι), MeasureTheory.StronglyMeasurable (u i)
    this✝¹ : MeasurableSpace β := borel β
    this✝ : BorelSpace β
    t_sf : Nat → MeasureTheory.SimpleFunc ι ι
    ht_sf : ∀ (j : ι) (x : α), Filter.Tendsto (fun n => u ((t_sf n) j) x) Filter.a …
    U : Nat → Prod ι α → β := fun n p => u ((t_sf n) p.1) p.2
    h_tendsto : Filter.Tendsto U Filter.atTop (nhds fun p => u p.1 p.2)
    ⊢ MeasureTheory.StronglyMeasurable (Function.uncurry u)
  -/
  refine stronglyMeasurable_of_tendsto _ (fun n => ?_) h_tendsto
  have h_str_meas : StronglyMeasurable fun p : (t_sf n).range × α => u (↑p.fst) p.snd := by
    refine stronglyMeasurable_iff_measurable_separable.2 ⟨?_, ?_⟩
    · have :
        (fun p : ↥(t_sf n).range × α => u (↑p.fst) p.snd) =
          (fun p : α × (t_sf n).range => u (↑p.snd) p.fst) ∘ Prod.swap :=
        rfl
      rw [this, measurable_swap_iff]
      exact measurable_from_prod_countable fun j => (h j).measurable
    · have : IsSeparable (⋃ i : (t_sf n).range, range (u i)) :=
        .iUnion fun i => (h i).isSeparable_range
      apply this.mono
      rintro _ ⟨⟨i, x⟩, rfl⟩
      simp only [mem_iUnion, mem_range]
      exact ⟨i, x, rfl⟩
  have :
    (fun p : ι × α => u (t_sf n p.fst) p.snd) =
      (fun p : ↥(t_sf n).range × α => u p.fst p.snd) ∘ fun p : ι × α =>
        (⟨t_sf n p.fst, SimpleFunc.mem_range_self _ _⟩, p.snd) :=
    rfl
  /-
    case intro
    α : Type u_5
    β : Type u_6
    ι : Type u_7
    inst✝⁷ : TopologicalSpace ι
    inst✝⁶ : TopologicalSpace.MetrizableSpace ι
    inst✝⁵ : MeasurableSpace ι
    inst✝⁴ : SecondCountableTopology ι
    inst✝³ : OpensMeasurableSpace ι
    inst✝² : TopologicalSpace β
    inst✝¹ : TopologicalSpace.PseudoMetrizableSpace β
    inst✝ : MeasurableSpace α
    u : ι → α → β
    hu_cont : ∀ (x : α), Continuous fun i => u i x
    h : ∀ (i : ι), MeasureTheory.StronglyMeasurable (u i)
    this✝¹ : MeasurableSpace β := borel β
    this✝ : BorelSpace β
    t_sf : Nat → MeasureTheory.SimpleFunc ι ι
    ht_sf : ∀ (j : ι) (x : α), Filter.Tendsto (fun n => u ((t_sf n) j) x) Filter.a …
    U : Nat → Prod ι α → β := fun n p => u ((t_sf n) p.1) p.2
    h_tendsto : Filter.Tendsto U Filter.atTop (nhds fun p => u p.1 p.2)
    n : Nat
    h_str_meas : MeasureTheory.StronglyMeasurable fun p => u (↑p.1) p.2
    this : Eq (fun p => u ((t_sf n) p.1) p.2) (Function.comp (fun p => u (↑p.1) p. …
    ⊢ MeasureTheory.StronglyMeasurable (U n)
  -/
  simp_rw [U, this]
  /-
    case intro
    α : Type u_5
    β : Type u_6
    ι : Type u_7
    inst✝⁷ : TopologicalSpace ι
    inst✝⁶ : TopologicalSpace.MetrizableSpace ι
    inst✝⁵ : MeasurableSpace ι
    inst✝⁴ : SecondCountableTopology ι
    inst✝³ : OpensMeasurableSpace ι
    inst✝² : TopologicalSpace β
    inst✝¹ : TopologicalSpace.PseudoMetrizableSpace β
    inst✝ : MeasurableSpace α
    u : ι → α → β
    hu_cont : ∀ (x : α), Continuous fun i => u i x
    h : ∀ (i : ι), MeasureTheory.StronglyMeasurable (u i)
    this✝¹ : MeasurableSpace β := borel β
    this✝ : BorelSpace β
    t_sf : Nat → MeasureTheory.SimpleFunc ι ι
    ht_sf : ∀ (j : ι) (x : α), Filter.Tendsto (fun n => u ((t_sf n) j) x) Filter.a …
    U : Nat → Prod ι α → β := fun n p => u ((t_sf n) p.1) p.2
    h_tendsto : Filter.Tendsto U Filter.atTop (nhds fun p => u p.1 p.2)
    n : Nat
    h_str_meas : MeasureTheory.StronglyMeasurable fun p => u (↑p.1) p.2
    this : Eq (fun p => u ((t_sf n) p.1) p.2) (Function.comp (fun p => u (↑p.1) p. …
    ⊢ MeasureTheory.StronglyMeasurable (Function.comp (fun p => u (↑p.1) p.2) fun  …
  -/
  refine h_str_meas.comp_measurable (Measurable.prod_mk ?_ measurable_snd)
  /-
    case intro
    α : Type u_5
    β : Type u_6
    ι : Type u_7
    inst✝⁷ : TopologicalSpace ι
    inst✝⁶ : TopologicalSpace.MetrizableSpace ι
    inst✝⁵ : MeasurableSpace ι
    inst✝⁴ : SecondCountableTopology ι
    inst✝³ : OpensMeasurableSpace ι
    inst✝² : TopologicalSpace β
    inst✝¹ : TopologicalSpace.PseudoMetrizableSpace β
    inst✝ : MeasurableSpace α
    u : ι → α → β
    hu_cont : ∀ (x : α), Continuous fun i => u i x
    h : ∀ (i : ι), MeasureTheory.StronglyMeasurable (u i)
    this✝¹ : MeasurableSpace β := borel β
    this✝ : BorelSpace β
    t_sf : Nat → MeasureTheory.SimpleFunc ι ι
    ht_sf : ∀ (j : ι) (x : α), Filter.Tendsto (fun n => u ((t_sf n) j) x) Filter.a …
    U : Nat → Prod ι α → β := fun n p => u ((t_sf n) p.1) p.2
    h_tendsto : Filter.Tendsto U Filter.atTop (nhds fun p => u p.1 p.2)
    n : Nat
    h_str_meas : MeasureTheory.StronglyMeasurable fun p => u (↑p.1) p.2
    this : Eq (fun p => u ((t_sf n) p.1) p.2) (Function.comp (fun p => u (↑p.1) p. …
    ⊢ Measurable fun p => ⟨(t_sf n) p.1, ⋯⟩
  -/
  exact ((t_sf n).measurable.comp measurable_fst).subtype_mk
  /-
    🎉 no goals
  -/


