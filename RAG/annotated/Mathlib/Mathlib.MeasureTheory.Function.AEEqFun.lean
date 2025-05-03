/-- The equivalence relation of being almost everywhere equal for almost everywhere strongly
measurable functions. -/
def Measure.aeEqSetoid (μ : Measure α) : Setoid { f : α → β // AEStronglyMeasurable f μ } :=
  ⟨fun f g => (f : α → β) =ᵐ[μ] g, fun {f} => ae_eq_refl f.val, fun {_ _} => ae_eq_symm,
    fun {_ _ _} => ae_eq_trans⟩


/-- The space of equivalence classes of almost everywhere strongly measurable functions, where two
    strongly measurable functions are equivalent if they agree almost everywhere, i.e.,
    they differ on a set of measure `0`. -/
def AEEqFun (μ : Measure α) : Type _ :=
  Quotient (μ.aeEqSetoid β)


@[inherit_doc MeasureTheory.AEEqFun]
notation:25 α " →ₘ[" μ "] " β => AEEqFun α β μ


/-- Construct the equivalence class `[f]` of an almost everywhere measurable function `f`, based
    on the equivalence relation of being almost everywhere equal. -/
def mk {β : Type*} [TopologicalSpace β] (f : α → β) (hf : AEStronglyMeasurable f μ) : α →ₘ[μ] β :=
  Quotient.mk'' ⟨f, hf⟩


open scoped Classical in
/-- Coercion from a space of equivalence classes of almost everywhere strongly measurable
functions to functions. We ensure that if `f` has a constant representative,
then we choose that one. -/
@[coe]
def cast (f : α →ₘ[μ] β) : α → β :=
  if h : ∃ (b : β), f = mk (const α b) aestronglyMeasurable_const then
    const α <| Classical.choose h else
    AEStronglyMeasurable.mk _ (Quotient.out f : { f : α → β // AEStronglyMeasurable f μ }).2


/-- A measurable representative of an `AEEqFun` [f] -/
instance instCoeFun : CoeFun (α →ₘ[μ] β) fun _ => α → β := ⟨cast⟩


protected theorem stronglyMeasurable (f : α →ₘ[μ] β) : StronglyMeasurable f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : TopologicalSpace β
    f : MeasureTheory.AEEqFun α β μ
    ⊢ MeasureTheory.StronglyMeasurable ↑f
  -/
  simp only [cast]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : TopologicalSpace β
    f : MeasureTheory.AEEqFun α β μ
    ⊢ MeasureTheory.StronglyMeasurable (dite (Exists fun b => Eq f (MeasureTheory. …
  -/
  split_ifs with h
    /-
      case pos
      α : Type u_1
      β : Type u_2
      inst✝¹ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : TopologicalSpace β
      f : MeasureTheory.AEEqFun α β μ
      h : Exists fun b => Eq f (MeasureTheory.AEEqFun.mk (Function.const α b) ⋯)
      ⊢ MeasureTheory.StronglyMeasurable (Function.const α (Classical.choose h))
    -/
  · exact stronglyMeasurable_const
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      inst✝¹ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : TopologicalSpace β
      f : MeasureTheory.AEEqFun α β μ
      h : Not (Exists fun b => Eq f (MeasureTheory.AEEqFun.mk (Function.const α b) ⋯))
      ⊢ MeasureTheory.StronglyMeasurable (MeasureTheory.AEStronglyMeasurable.mk ↑(Qu …
    -/
  · apply AEStronglyMeasurable.stronglyMeasurable_mk
    /-
      🎉 no goals
    -/


protected theorem aestronglyMeasurable (f : α →ₘ[μ] β) : AEStronglyMeasurable f μ :=
  f.stronglyMeasurable.aestronglyMeasurable


protected theorem measurable [PseudoMetrizableSpace β] [MeasurableSpace β] [BorelSpace β]
    (f : α →ₘ[μ] β) : Measurable f :=
  f.stronglyMeasurable.measurable


protected theorem aemeasurable [PseudoMetrizableSpace β] [MeasurableSpace β] [BorelSpace β]
    (f : α →ₘ[μ] β) : AEMeasurable f μ :=
  f.measurable.aemeasurable


@[simp]
theorem quot_mk_eq_mk (f : α → β) (hf) :
    (Quot.mk (@Setoid.r _ <| μ.aeEqSetoid β) ⟨f, hf⟩ : α →ₘ[μ] β) = mk f hf :=
  rfl


@[simp]
theorem mk_eq_mk {f g : α → β} {hf hg} : (mk f hf : α →ₘ[μ] β) = mk g hg ↔ f =ᵐ[μ] g :=
  Quotient.eq''


@[simp]
theorem mk_coeFn (f : α →ₘ[μ] β) : mk f f.aestronglyMeasurable = f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : TopologicalSpace β
    f : MeasureTheory.AEEqFun α β μ
    ⊢ Eq (MeasureTheory.AEEqFun.mk ↑f ⋯) f
  -/
  conv_lhs => simp only [cast]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : TopologicalSpace β
    f : MeasureTheory.AEEqFun α β μ
    ⊢ Eq (MeasureTheory.AEEqFun.mk (dite (Exists fun b => Eq f (MeasureTheory.AEEq …
  -/
  split_ifs with h
    /-
      case pos
      α : Type u_1
      β : Type u_2
      inst✝¹ : MeasurableSpace α
      μ : MeasureTheory.Measure α
      inst✝ : TopologicalSpace β
      f : MeasureTheory.AEEqFun α β μ
      h : Exists fun b => Eq f (MeasureTheory.AEEqFun.mk (Function.const α b) ⋯)
      ⊢ Eq (MeasureTheory.AEEqFun.mk (Function.const α (Classical.choose h)) ⋯) f
    -/
  · exact Classical.choose_spec h |>.symm
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : TopologicalSpace β
    f : MeasureTheory.AEEqFun α β μ
    h : Not (Exists fun b => Eq f (MeasureTheory.AEEqFun.mk (Function.const α b) ⋯))
    ⊢ Eq (MeasureTheory.AEEqFun.mk (MeasureTheory.AEStronglyMeasurable.mk ↑(Quotie …
  -/
  conv_rhs => rw [← Quotient.out_eq' f]
  /-
    case neg
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : TopologicalSpace β
    f : MeasureTheory.AEEqFun α β μ
    h : Not (Exists fun b => Eq f (MeasureTheory.AEEqFun.mk (Function.const α b) ⋯))
    ⊢ Eq (MeasureTheory.AEEqFun.mk (MeasureTheory.AEStronglyMeasurable.mk ↑(Quotie …
  -/
  rw [← mk, mk_eq_mk]
  /-
    case neg
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : TopologicalSpace β
    f : MeasureTheory.AEEqFun α β μ
    h : Not (Exists fun b => Eq f (MeasureTheory.AEEqFun.mk (Function.const α b) ⋯))
    ⊢ (MeasureTheory.ae μ).EventuallyEq (MeasureTheory.AEStronglyMeasurable.mk ↑(Q …
  -/
  exact (AEStronglyMeasurable.ae_eq_mk _).symm
  /-
    🎉 no goals
  -/


@[ext]
theorem ext {f g : α →ₘ[μ] β} (h : f =ᵐ[μ] g) : f = g := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : TopologicalSpace β
    f g : MeasureTheory.AEEqFun α β μ
    h : (MeasureTheory.ae μ).EventuallyEq ↑f ↑g
    ⊢ Eq f g
  -/
  rwa [← f.mk_coeFn, ← g.mk_coeFn, mk_eq_mk]
  /-
    🎉 no goals
  -/


theorem coeFn_mk (f : α → β) (hf) : (mk f hf : α →ₘ[μ] β) =ᵐ[μ] f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : TopologicalSpace β
    f : α → β
    hf : MeasureTheory.AEStronglyMeasurable f μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq (↑(MeasureTheory.AEEqFun.mk f hf)) f
  -/
  rw [← mk_eq_mk, mk_coeFn]
  /-
    🎉 no goals
  -/


@[elab_as_elim]
theorem induction_on (f : α →ₘ[μ] β) {p : (α →ₘ[μ] β) → Prop} (H : ∀ f hf, p (mk f hf)) : p f :=
  Quotient.inductionOn' f <| Subtype.forall.2 H


@[elab_as_elim]
theorem induction_on₂ {α' β' : Type*} [MeasurableSpace α'] [TopologicalSpace β'] {μ' : Measure α'}
    (f : α →ₘ[μ] β) (f' : α' →ₘ[μ'] β') {p : (α →ₘ[μ] β) → (α' →ₘ[μ'] β') → Prop}
    (H : ∀ f hf f' hf', p (mk f hf) (mk f' hf')) : p f f' :=
  induction_on f fun f hf => induction_on f' <| H f hf


@[elab_as_elim]
theorem induction_on₃ {α' β' : Type*} [MeasurableSpace α'] [TopologicalSpace β'] {μ' : Measure α'}
    {α'' β'' : Type*} [MeasurableSpace α''] [TopologicalSpace β''] {μ'' : Measure α''}
    (f : α →ₘ[μ] β) (f' : α' →ₘ[μ'] β') (f'' : α'' →ₘ[μ''] β'')
    {p : (α →ₘ[μ] β) → (α' →ₘ[μ'] β') → (α'' →ₘ[μ''] β'') → Prop}
    (H : ∀ f hf f' hf' f'' hf'', p (mk f hf) (mk f' hf') (mk f'' hf'')) : p f f' f'' :=
  induction_on f fun f hf => induction_on₂ f' f'' <| H f hf


/-- Composition of an almost everywhere equal function and a quasi measure preserving function.

See also `AEEqFun.compMeasurePreserving`. -/
def compQuasiMeasurePreserving (g : β →ₘ[ν] γ) (f : α → β) (hf : QuasiMeasurePreserving f μ ν) :
    α →ₘ[μ] γ :=
  Quotient.liftOn' g (fun g ↦ mk (g ∘ f) <| g.2.comp_quasiMeasurePreserving hf) fun _ _ h ↦
    mk_eq_mk.2 <| h.comp_tendsto hf.tendsto_ae


@[simp]
theorem compQuasiMeasurePreserving_mk {g : β → γ} (hg : AEStronglyMeasurable g ν)
    (hf : QuasiMeasurePreserving f μ ν) :
    (mk g hg).compQuasiMeasurePreserving f hf = mk (g ∘ f) (hg.comp_quasiMeasurePreserving hf) :=
  rfl


theorem compQuasiMeasurePreserving_eq_mk (g : β →ₘ[ν] γ) (hf : QuasiMeasurePreserving f μ ν) :
    g.compQuasiMeasurePreserving f hf =
      mk (g ∘ f) (g.aestronglyMeasurable.comp_quasiMeasurePreserving hf) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : TopologicalSpace γ
    inst✝ : MeasurableSpace β
    ν : MeasureTheory.Measure β
    f : α → β
    g : MeasureTheory.AEEqFun β γ ν
    hf : MeasureTheory.Measure.QuasiMeasurePreserving f μ ν
    ⊢ Eq (g.compQuasiMeasurePreserving f hf) (MeasureTheory.AEEqFun.mk (Function.c …
  -/
  rw [← compQuasiMeasurePreserving_mk g.aestronglyMeasurable hf, mk_coeFn]
  /-
    🎉 no goals
  -/


theorem coeFn_compQuasiMeasurePreserving (g : β →ₘ[ν] γ) (hf : QuasiMeasurePreserving f μ ν) :
    g.compQuasiMeasurePreserving f hf =ᵐ[μ] g ∘ f := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : TopologicalSpace γ
    inst✝ : MeasurableSpace β
    ν : MeasureTheory.Measure β
    f : α → β
    g : MeasureTheory.AEEqFun β γ ν
    hf : MeasureTheory.Measure.QuasiMeasurePreserving f μ ν
    ⊢ (MeasureTheory.ae μ).EventuallyEq (↑(g.compQuasiMeasurePreserving f hf)) (Fu …
  -/
  rw [compQuasiMeasurePreserving_eq_mk]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : TopologicalSpace γ
    inst✝ : MeasurableSpace β
    ν : MeasureTheory.Measure β
    f : α → β
    g : MeasureTheory.AEEqFun β γ ν
    hf : MeasureTheory.Measure.QuasiMeasurePreserving f μ ν
    ⊢ (MeasureTheory.ae μ).EventuallyEq (↑(MeasureTheory.AEEqFun.mk (Function.comp …
  -/
  apply coeFn_mk
  /-
    🎉 no goals
  -/


/-- Composition of an almost everywhere equal function and a quasi measure preserving function.

This is an important special case of `AEEqFun.compQuasiMeasurePreserving`. We use a separate
definition so that lemmas that need `f` to be measure preserving can be `@[simp]` lemmas. -/
def compMeasurePreserving (g : β →ₘ[ν] γ) (f : α → β) (hf : MeasurePreserving f μ ν) : α →ₘ[μ] γ :=
  g.compQuasiMeasurePreserving f hf.quasiMeasurePreserving


@[simp]
theorem compMeasurePreserving_mk (hg : AEStronglyMeasurable g ν) (hf : MeasurePreserving f μ ν) :
    (mk g hg).compMeasurePreserving f hf =
      mk (g ∘ f) (hg.comp_quasiMeasurePreserving hf.quasiMeasurePreserving) :=
  rfl


theorem compMeasurePreserving_eq_mk (g : β →ₘ[ν] γ) (hf : MeasurePreserving f μ ν) :
    g.compMeasurePreserving f hf =
      mk (g ∘ f) (g.aestronglyMeasurable.comp_quasiMeasurePreserving hf.quasiMeasurePreserving) :=
  g.compQuasiMeasurePreserving_eq_mk _


theorem coeFn_compMeasurePreserving (g : β →ₘ[ν] γ) (hf : MeasurePreserving f μ ν) :
    g.compMeasurePreserving f hf =ᵐ[μ] g ∘ f :=
  g.coeFn_compQuasiMeasurePreserving _


/-- Given a continuous function `g : β → γ`, and an almost everywhere equal function `[f] : α →ₘ β`,
    return the equivalence class of `g ∘ f`, i.e., the almost everywhere equal function
    `[g ∘ f] : α →ₘ γ`. -/
def comp (g : β → γ) (hg : Continuous g) (f : α →ₘ[μ] β) : α →ₘ[μ] γ :=
  Quotient.liftOn' f (fun f => mk (g ∘ (f : α → β)) (hg.comp_aestronglyMeasurable f.2))
    fun _ _ H => mk_eq_mk.2 <| H.fun_comp g


@[simp]
theorem comp_mk (g : β → γ) (hg : Continuous g) (f : α → β) (hf) :
    comp g hg (mk f hf : α →ₘ[μ] β) = mk (g ∘ f) (hg.comp_aestronglyMeasurable hf) :=
  rfl


theorem comp_eq_mk (g : β → γ) (hg : Continuous g) (f : α →ₘ[μ] β) :
    comp g hg f = mk (g ∘ f) (hg.comp_aestronglyMeasurable f.aestronglyMeasurable) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace γ
    g : β → γ
    hg : Continuous g
    f : MeasureTheory.AEEqFun α β μ
    ⊢ Eq (MeasureTheory.AEEqFun.comp g hg f) (MeasureTheory.AEEqFun.mk (Function.c …
  -/
  rw [← comp_mk g hg f f.aestronglyMeasurable, mk_coeFn]
  /-
    🎉 no goals
  -/


theorem coeFn_comp (g : β → γ) (hg : Continuous g) (f : α →ₘ[μ] β) : comp g hg f =ᵐ[μ] g ∘ f := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace γ
    g : β → γ
    hg : Continuous g
    f : MeasureTheory.AEEqFun α β μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq (↑(MeasureTheory.AEEqFun.comp g hg f)) (Fu …
  -/
  rw [comp_eq_mk]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace γ
    g : β → γ
    hg : Continuous g
    f : MeasureTheory.AEEqFun α β μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq (↑(MeasureTheory.AEEqFun.mk (Function.comp …
  -/
  apply coeFn_mk
  /-
    🎉 no goals
  -/


theorem comp_compQuasiMeasurePreserving
    {β : Type*} [MeasurableSpace β] {ν} (g : γ → δ) (hg : Continuous g)
    (f : β →ₘ[ν] γ) {φ : α → β} (hφ : Measure.QuasiMeasurePreserving φ μ ν) :
    (comp g hg f).compQuasiMeasurePreserving φ hφ =
      comp g hg (f.compQuasiMeasurePreserving φ hφ) := by
  /-
    α : Type u_1
    γ : Type u_3
    δ : Type u_4
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace δ
    inst✝¹ : TopologicalSpace γ
    β : Type u_5
    inst✝ : MeasurableSpace β
    ν : MeasureTheory.Measure β
    g : γ → δ
    hg : Continuous g
    f : MeasureTheory.AEEqFun β γ ν
    φ : α → β
    hφ : MeasureTheory.Measure.QuasiMeasurePreserving φ μ ν
    ⊢ Eq ((MeasureTheory.AEEqFun.comp g hg f).compQuasiMeasurePreserving φ hφ) (Me …
  -/
  rcases f; rfl
            /-
              🎉 no goals
            -/


/-- Given a measurable function `g : β → γ`, and an almost everywhere equal function `[f] : α →ₘ β`,
    return the equivalence class of `g ∘ f`, i.e., the almost everywhere equal function
    `[g ∘ f] : α →ₘ γ`. This requires that `γ` has a second countable topology. -/
def compMeasurable (g : β → γ) (hg : Measurable g) (f : α →ₘ[μ] β) : α →ₘ[μ] γ :=
  Quotient.liftOn' f
    (fun f' => mk (g ∘ (f' : α → β)) (hg.comp_aemeasurable f'.2.aemeasurable).aestronglyMeasurable)
    fun _ _ H => mk_eq_mk.2 <| H.fun_comp g


@[simp]
theorem compMeasurable_mk (g : β → γ) (hg : Measurable g) (f : α → β)
    (hf : AEStronglyMeasurable f μ) :
    compMeasurable g hg (mk f hf : α →ₘ[μ] β) =
      mk (g ∘ f) (hg.comp_aemeasurable hf.aemeasurable).aestronglyMeasurable :=
  rfl


theorem compMeasurable_eq_mk (g : β → γ) (hg : Measurable g) (f : α →ₘ[μ] β) :
    compMeasurable g hg f =
    mk (g ∘ f) (hg.comp_aemeasurable f.aemeasurable).aestronglyMeasurable := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁸ : TopologicalSpace β
    inst✝⁷ : TopologicalSpace γ
    inst✝⁶ : MeasurableSpace β
    inst✝⁵ : TopologicalSpace.PseudoMetrizableSpace β
    inst✝⁴ : BorelSpace β
    inst✝³ : MeasurableSpace γ
    inst✝² : TopologicalSpace.PseudoMetrizableSpace γ
    inst✝¹ : OpensMeasurableSpace γ
    inst✝ : SecondCountableTopology γ
    g : β → γ
    hg : Measurable g
    f : MeasureTheory.AEEqFun α β μ
    ⊢ Eq (MeasureTheory.AEEqFun.compMeasurable g hg f) (MeasureTheory.AEEqFun.mk ( …
  -/
  rw [← compMeasurable_mk g hg f f.aestronglyMeasurable, mk_coeFn]
  /-
    🎉 no goals
  -/


theorem coeFn_compMeasurable (g : β → γ) (hg : Measurable g) (f : α →ₘ[μ] β) :
    compMeasurable g hg f =ᵐ[μ] g ∘ f := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁸ : TopologicalSpace β
    inst✝⁷ : TopologicalSpace γ
    inst✝⁶ : MeasurableSpace β
    inst✝⁵ : TopologicalSpace.PseudoMetrizableSpace β
    inst✝⁴ : BorelSpace β
    inst✝³ : MeasurableSpace γ
    inst✝² : TopologicalSpace.PseudoMetrizableSpace γ
    inst✝¹ : OpensMeasurableSpace γ
    inst✝ : SecondCountableTopology γ
    g : β → γ
    hg : Measurable g
    f : MeasureTheory.AEEqFun α β μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq (↑(MeasureTheory.AEEqFun.compMeasurable g  …
  -/
  rw [compMeasurable_eq_mk]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁹ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝⁸ : TopologicalSpace β
    inst✝⁷ : TopologicalSpace γ
    inst✝⁶ : MeasurableSpace β
    inst✝⁵ : TopologicalSpace.PseudoMetrizableSpace β
    inst✝⁴ : BorelSpace β
    inst✝³ : MeasurableSpace γ
    inst✝² : TopologicalSpace.PseudoMetrizableSpace γ
    inst✝¹ : OpensMeasurableSpace γ
    inst✝ : SecondCountableTopology γ
    g : β → γ
    hg : Measurable g
    f : MeasureTheory.AEEqFun α β μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq (↑(MeasureTheory.AEEqFun.mk (Function.comp …
  -/
  apply coeFn_mk
  /-
    🎉 no goals
  -/


/-- The class of `x ↦ (f x, g x)`. -/
def pair (f : α →ₘ[μ] β) (g : α →ₘ[μ] γ) : α →ₘ[μ] β × γ :=
  Quotient.liftOn₂' f g (fun f g => mk (fun x => (f.1 x, g.1 x)) (f.2.prod_mk g.2))
    fun _f _g _f' _g' Hf Hg => mk_eq_mk.2 <| Hf.prod_mk Hg


@[simp]
theorem pair_mk_mk (f : α → β) (hf) (g : α → γ) (hg) :
    (mk f hf : α →ₘ[μ] β).pair (mk g hg) = mk (fun x => (f x, g x)) (hf.prod_mk hg) :=
  rfl


theorem pair_eq_mk (f : α →ₘ[μ] β) (g : α →ₘ[μ] γ) :
    f.pair g =
      mk (fun x => (f x, g x)) (f.aestronglyMeasurable.prod_mk g.aestronglyMeasurable) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace γ
    f : MeasureTheory.AEEqFun α β μ
    g : MeasureTheory.AEEqFun α γ μ
    ⊢ Eq (f.pair g) (MeasureTheory.AEEqFun.mk (fun x => { fst := ↑f x, snd := ↑g x …
  -/
  simp only [← pair_mk_mk, mk_coeFn, f.aestronglyMeasurable, g.aestronglyMeasurable]
  /-
    🎉 no goals
  -/


theorem coeFn_pair (f : α →ₘ[μ] β) (g : α →ₘ[μ] γ) : f.pair g =ᵐ[μ] fun x => (f x, g x) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace γ
    f : MeasureTheory.AEEqFun α β μ
    g : MeasureTheory.AEEqFun α γ μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑(f.pair g) fun x => { fst := ↑f x, snd := …
  -/
  rw [pair_eq_mk]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace γ
    f : MeasureTheory.AEEqFun α β μ
    g : MeasureTheory.AEEqFun α γ μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑(MeasureTheory.AEEqFun.mk (fun x => { fst …
  -/
  apply coeFn_mk
  /-
    🎉 no goals
  -/


/-- Given a continuous function `g : β → γ → δ`, and almost everywhere equal functions
    `[f₁] : α →ₘ β` and `[f₂] : α →ₘ γ`, return the equivalence class of the function
    `fun a => g (f₁ a) (f₂ a)`, i.e., the almost everywhere equal function
    `[fun a => g (f₁ a) (f₂ a)] : α →ₘ γ` -/
def comp₂ (g : β → γ → δ) (hg : Continuous (uncurry g)) (f₁ : α →ₘ[μ] β) (f₂ : α →ₘ[μ] γ) :
    α →ₘ[μ] δ :=
  comp _ hg (f₁.pair f₂)


@[simp]
theorem comp₂_mk_mk (g : β → γ → δ) (hg : Continuous (uncurry g)) (f₁ : α → β) (f₂ : α → γ)
    (hf₁ hf₂) :
    comp₂ g hg (mk f₁ hf₁ : α →ₘ[μ] β) (mk f₂ hf₂) =
      mk (fun a => g (f₁ a) (f₂ a)) (hg.comp_aestronglyMeasurable (hf₁.prod_mk hf₂)) :=
  rfl


theorem comp₂_eq_pair (g : β → γ → δ) (hg : Continuous (uncurry g)) (f₁ : α →ₘ[μ] β)
    (f₂ : α →ₘ[μ] γ) : comp₂ g hg f₁ f₂ = comp _ hg (f₁.pair f₂) :=
  rfl


theorem comp₂_eq_mk (g : β → γ → δ) (hg : Continuous (uncurry g)) (f₁ : α →ₘ[μ] β)
    (f₂ : α →ₘ[μ] γ) : comp₂ g hg f₁ f₂ = mk (fun a => g (f₁ a) (f₂ a))
      (hg.comp_aestronglyMeasurable (f₁.aestronglyMeasurable.prod_mk f₂.aestronglyMeasurable)) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace δ
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace γ
    g : β → γ → δ
    hg : Continuous (Function.uncurry g)
    f₁ : MeasureTheory.AEEqFun α β μ
    f₂ : MeasureTheory.AEEqFun α γ μ
    ⊢ Eq (MeasureTheory.AEEqFun.comp₂ g hg f₁ f₂) (MeasureTheory.AEEqFun.mk (fun a …
  -/
  rw [comp₂_eq_pair, pair_eq_mk, comp_mk]; rfl
                                           /-
                                             🎉 no goals
                                           -/


theorem coeFn_comp₂ (g : β → γ → δ) (hg : Continuous (uncurry g)) (f₁ : α →ₘ[μ] β)
    (f₂ : α →ₘ[μ] γ) : comp₂ g hg f₁ f₂ =ᵐ[μ] fun a => g (f₁ a) (f₂ a) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace δ
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace γ
    g : β → γ → δ
    hg : Continuous (Function.uncurry g)
    f₁ : MeasureTheory.AEEqFun α β μ
    f₂ : MeasureTheory.AEEqFun α γ μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑(MeasureTheory.AEEqFun.comp₂ g hg f₁ f₂)  …
  -/
  rw [comp₂_eq_mk]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace δ
    inst✝¹ : TopologicalSpace β
    inst✝ : TopologicalSpace γ
    g : β → γ → δ
    hg : Continuous (Function.uncurry g)
    f₁ : MeasureTheory.AEEqFun α β μ
    f₂ : MeasureTheory.AEEqFun α γ μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑(MeasureTheory.AEEqFun.mk (fun a => g (↑f …
  -/
  apply coeFn_mk
  /-
    🎉 no goals
  -/


/-- Given a measurable function `g : β → γ → δ`, and almost everywhere equal functions
    `[f₁] : α →ₘ β` and `[f₂] : α →ₘ γ`, return the equivalence class of the function
    `fun a => g (f₁ a) (f₂ a)`, i.e., the almost everywhere equal function
    `[fun a => g (f₁ a) (f₂ a)] : α →ₘ γ`. This requires `δ` to have second-countable topology. -/
def comp₂Measurable (g : β → γ → δ) (hg : Measurable (uncurry g)) (f₁ : α →ₘ[μ] β)
    (f₂ : α →ₘ[μ] γ) : α →ₘ[μ] δ :=
  compMeasurable _ hg (f₁.pair f₂)


@[simp]
theorem comp₂Measurable_mk_mk (g : β → γ → δ) (hg : Measurable (uncurry g)) (f₁ : α → β)
    (f₂ : α → γ) (hf₁ hf₂) :
    comp₂Measurable g hg (mk f₁ hf₁ : α →ₘ[μ] β) (mk f₂ hf₂) =
      mk (fun a => g (f₁ a) (f₂ a))
        (hg.comp_aemeasurable (hf₁.aemeasurable.prod_mk hf₂.aemeasurable)).aestronglyMeasurable :=
  rfl


theorem comp₂Measurable_eq_pair (g : β → γ → δ) (hg : Measurable (uncurry g)) (f₁ : α →ₘ[μ] β)
    (f₂ : α →ₘ[μ] γ) : comp₂Measurable g hg f₁ f₂ = compMeasurable _ hg (f₁.pair f₂) :=
  rfl


theorem comp₂Measurable_eq_mk (g : β → γ → δ) (hg : Measurable (uncurry g)) (f₁ : α →ₘ[μ] β)
    (f₂ : α →ₘ[μ] γ) :
    comp₂Measurable g hg f₁ f₂ =
      mk (fun a => g (f₁ a) (f₂ a))
        (hg.comp_aemeasurable (f₁.aemeasurable.prod_mk f₂.aemeasurable)).aestronglyMeasurable := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    inst✝¹⁴ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹³ : TopologicalSpace δ
    inst✝¹² : TopologicalSpace β
    inst✝¹¹ : TopologicalSpace γ
    inst✝¹⁰ : MeasurableSpace β
    inst✝⁹ : TopologicalSpace.PseudoMetrizableSpace β
    inst✝⁸ : BorelSpace β
    inst✝⁷ : MeasurableSpace γ
    inst✝⁶ : TopologicalSpace.PseudoMetrizableSpace γ
    inst✝⁵ : BorelSpace γ
    inst✝⁴ : SecondCountableTopologyEither β γ
    inst✝³ : MeasurableSpace δ
    inst✝² : TopologicalSpace.PseudoMetrizableSpace δ
    inst✝¹ : OpensMeasurableSpace δ
    inst✝ : SecondCountableTopology δ
    g : β → γ → δ
    hg : Measurable (Function.uncurry g)
    f₁ : MeasureTheory.AEEqFun α β μ
    f₂ : MeasureTheory.AEEqFun α γ μ
    ⊢ Eq (MeasureTheory.AEEqFun.comp₂Measurable g hg f₁ f₂) (MeasureTheory.AEEqFun …
  -/
  rw [comp₂Measurable_eq_pair, pair_eq_mk, compMeasurable_mk]; rfl
                                                               /-
                                                                 🎉 no goals
                                                               -/


theorem coeFn_comp₂Measurable (g : β → γ → δ) (hg : Measurable (uncurry g)) (f₁ : α →ₘ[μ] β)
    (f₂ : α →ₘ[μ] γ) : comp₂Measurable g hg f₁ f₂ =ᵐ[μ] fun a => g (f₁ a) (f₂ a) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    inst✝¹⁴ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹³ : TopologicalSpace δ
    inst✝¹² : TopologicalSpace β
    inst✝¹¹ : TopologicalSpace γ
    inst✝¹⁰ : MeasurableSpace β
    inst✝⁹ : TopologicalSpace.PseudoMetrizableSpace β
    inst✝⁸ : BorelSpace β
    inst✝⁷ : MeasurableSpace γ
    inst✝⁶ : TopologicalSpace.PseudoMetrizableSpace γ
    inst✝⁵ : BorelSpace γ
    inst✝⁴ : SecondCountableTopologyEither β γ
    inst✝³ : MeasurableSpace δ
    inst✝² : TopologicalSpace.PseudoMetrizableSpace δ
    inst✝¹ : OpensMeasurableSpace δ
    inst✝ : SecondCountableTopology δ
    g : β → γ → δ
    hg : Measurable (Function.uncurry g)
    f₁ : MeasureTheory.AEEqFun α β μ
    f₂ : MeasureTheory.AEEqFun α γ μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑(MeasureTheory.AEEqFun.comp₂Measurable g  …
  -/
  rw [comp₂Measurable_eq_mk]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    inst✝¹⁴ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹³ : TopologicalSpace δ
    inst✝¹² : TopologicalSpace β
    inst✝¹¹ : TopologicalSpace γ
    inst✝¹⁰ : MeasurableSpace β
    inst✝⁹ : TopologicalSpace.PseudoMetrizableSpace β
    inst✝⁸ : BorelSpace β
    inst✝⁷ : MeasurableSpace γ
    inst✝⁶ : TopologicalSpace.PseudoMetrizableSpace γ
    inst✝⁵ : BorelSpace γ
    inst✝⁴ : SecondCountableTopologyEither β γ
    inst✝³ : MeasurableSpace δ
    inst✝² : TopologicalSpace.PseudoMetrizableSpace δ
    inst✝¹ : OpensMeasurableSpace δ
    inst✝ : SecondCountableTopology δ
    g : β → γ → δ
    hg : Measurable (Function.uncurry g)
    f₁ : MeasureTheory.AEEqFun α β μ
    f₂ : MeasureTheory.AEEqFun α γ μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑(MeasureTheory.AEEqFun.mk (fun a => g (↑f …
  -/
  apply coeFn_mk
  /-
    🎉 no goals
  -/


/-- Interpret `f : α →ₘ[μ] β` as a germ at `ae μ` forgetting that `f` is almost everywhere
    strongly measurable. -/
def toGerm (f : α →ₘ[μ] β) : Germ (ae μ) β :=
  Quotient.liftOn' f (fun f => ((f : α → β) : Germ (ae μ) β)) fun _ _ H => Germ.coe_eq.2 H


@[simp]
theorem mk_toGerm (f : α → β) (hf) : (mk f hf : α →ₘ[μ] β).toGerm = f :=
  rfl


                                                                 /-
                                                                   α : Type u_1
                                                                   β : Type u_2
                                                                   inst✝¹ : MeasurableSpace α
                                                                   μ : MeasureTheory.Measure α
                                                                   inst✝ : TopologicalSpace β
                                                                   f : MeasureTheory.AEEqFun α β μ
                                                                   ⊢ Eq f.toGerm ↑↑f
                                                                 -/
theorem toGerm_eq (f : α →ₘ[μ] β) : f.toGerm = (f : α → β) := by rw [← mk_toGerm, mk_coeFn]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


theorem toGerm_injective : Injective (toGerm : (α →ₘ[μ] β) → Germ (ae μ) β) := fun f g H =>
                             /-
                               α : Type u_1
                               β : Type u_2
                               inst✝¹ : MeasurableSpace α
                               μ : MeasureTheory.Measure α
                               inst✝ : TopologicalSpace β
                               f g : MeasureTheory.AEEqFun α β μ
                               H : Eq f.toGerm g.toGerm
                               ⊢ Eq ↑↑f ↑↑g
                             -/
  ext <| Germ.coe_eq.1 <| by rwa [← toGerm_eq, ← toGerm_eq]
                             /-
                               🎉 no goals
                             -/


@[simp]
theorem compQuasiMeasurePreserving_toGerm {β : Type*} [MeasurableSpace β] {f : α → β} {ν}
    (g : β →ₘ[ν] γ) (hf : Measure.QuasiMeasurePreserving f μ ν) :
    (g.compQuasiMeasurePreserving f hf).toGerm = g.toGerm.compTendsto f hf.tendsto_ae := by
  /-
    α : Type u_1
    γ : Type u_3
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : TopologicalSpace γ
    β : Type u_5
    inst✝ : MeasurableSpace β
    f : α → β
    ν : MeasureTheory.Measure β
    g : MeasureTheory.AEEqFun β γ ν
    hf : MeasureTheory.Measure.QuasiMeasurePreserving f μ ν
    ⊢ Eq (g.compQuasiMeasurePreserving f hf).toGerm (g.toGerm.compTendsto f ⋯)
  -/
  rcases g; rfl
            /-
              🎉 no goals
            -/


@[simp]
theorem compMeasurePreserving_toGerm {β : Type*} [MeasurableSpace β] {f : α → β} {ν}
    (g : β →ₘ[ν] γ) (hf : MeasurePreserving f μ ν) :
    (g.compMeasurePreserving f hf).toGerm =
      g.toGerm.compTendsto f hf.quasiMeasurePreserving.tendsto_ae :=
  compQuasiMeasurePreserving_toGerm _ _


theorem comp_toGerm (g : β → γ) (hg : Continuous g) (f : α →ₘ[μ] β) :
    (comp g hg f).toGerm = f.toGerm.map g :=
                               /-
                                 α : Type u_1
                                 β : Type u_2
                                 γ : Type u_3
                                 inst✝² : MeasurableSpace α
                                 μ : MeasureTheory.Measure α
                                 inst✝¹ : TopologicalSpace β
                                 inst✝ : TopologicalSpace γ
                                 g : β → γ
                                 hg : Continuous g
                                 f✝ : MeasureTheory.AEEqFun α β μ
                                 f : α → β
                                 x✝ : MeasureTheory.AEStronglyMeasurable f μ
                                 ⊢ Eq (MeasureTheory.AEEqFun.comp g hg (MeasureTheory.AEEqFun.mk f x✝)).toGerm  …
                               -/
  induction_on f fun f _ => by simp
                               /-
                                 🎉 no goals
                               -/


theorem compMeasurable_toGerm [MeasurableSpace β] [BorelSpace β] [PseudoMetrizableSpace β]
    [PseudoMetrizableSpace γ] [SecondCountableTopology γ] [MeasurableSpace γ]
    [OpensMeasurableSpace γ] (g : β → γ) (hg : Measurable g) (f : α →ₘ[μ] β) :
    (compMeasurable g hg f).toGerm = f.toGerm.map g :=
                               /-
                                 α : Type u_1
                                 β : Type u_2
                                 γ : Type u_3
                                 inst✝⁹ : MeasurableSpace α
                                 μ : MeasureTheory.Measure α
                                 inst✝⁸ : TopologicalSpace β
                                 inst✝⁷ : TopologicalSpace γ
                                 inst✝⁶ : MeasurableSpace β
                                 inst✝⁵ : BorelSpace β
                                 inst✝⁴ : TopologicalSpace.PseudoMetrizableSpace β
                                 inst✝³ : TopologicalSpace.PseudoMetrizableSpace γ
                                 inst✝² : SecondCountableTopology γ
                                 inst✝¹ : MeasurableSpace γ
                                 inst✝ : OpensMeasurableSpace γ
                                 g : β → γ
                                 hg : Measurable g
                                 f✝ : MeasureTheory.AEEqFun α β μ
                                 f : α → β
                                 x✝ : MeasureTheory.AEStronglyMeasurable f μ
                                 ⊢ Eq (MeasureTheory.AEEqFun.compMeasurable g hg (MeasureTheory.AEEqFun.mk f x✝ …
                               -/
  induction_on f fun f _ => by simp
                               /-
                                 🎉 no goals
                               -/


theorem comp₂_toGerm (g : β → γ → δ) (hg : Continuous (uncurry g)) (f₁ : α →ₘ[μ] β)
    (f₂ : α →ₘ[μ] γ) : (comp₂ g hg f₁ f₂).toGerm = f₁.toGerm.map₂ g f₂.toGerm :=
                                          /-
                                            α : Type u_1
                                            β : Type u_2
                                            γ : Type u_3
                                            δ : Type u_4
                                            inst✝³ : MeasurableSpace α
                                            μ : MeasureTheory.Measure α
                                            inst✝² : TopologicalSpace δ
                                            inst✝¹ : TopologicalSpace β
                                            inst✝ : TopologicalSpace γ
                                            g : β → γ → δ
                                            hg : Continuous (Function.uncurry g)
                                            f₁✝ : MeasureTheory.AEEqFun α β μ
                                            f₂✝ : MeasureTheory.AEEqFun α γ μ
                                            f₁ : α → β
                                            x✝¹ : MeasureTheory.AEStronglyMeasurable f₁ μ
                                            f₂ : α → γ
                                            x✝ : MeasureTheory.AEStronglyMeasurable f₂ μ
                                            ⊢ Eq (MeasureTheory.AEEqFun.comp₂ g hg (MeasureTheory.AEEqFun.mk f₁ x✝¹) (Meas …
                                          -/
  induction_on₂ f₁ f₂ fun f₁ _ f₂ _ => by simp
                                          /-
                                            🎉 no goals
                                          -/


theorem comp₂Measurable_toGerm [PseudoMetrizableSpace β] [MeasurableSpace β] [BorelSpace β]
    [PseudoMetrizableSpace γ] [SecondCountableTopologyEither β γ]
    [MeasurableSpace γ] [BorelSpace γ] [PseudoMetrizableSpace δ] [SecondCountableTopology δ]
    [MeasurableSpace δ] [OpensMeasurableSpace δ] (g : β → γ → δ) (hg : Measurable (uncurry g))
    (f₁ : α →ₘ[μ] β) (f₂ : α →ₘ[μ] γ) :
    (comp₂Measurable g hg f₁ f₂).toGerm = f₁.toGerm.map₂ g f₂.toGerm :=
                                          /-
                                            α : Type u_1
                                            β : Type u_2
                                            γ : Type u_3
                                            δ : Type u_4
                                            inst✝¹⁴ : MeasurableSpace α
                                            μ : MeasureTheory.Measure α
                                            inst✝¹³ : TopologicalSpace δ
                                            inst✝¹² : TopologicalSpace β
                                            inst✝¹¹ : TopologicalSpace γ
                                            inst✝¹⁰ : TopologicalSpace.PseudoMetrizableSpace β
                                            inst✝⁹ : MeasurableSpace β
                                            inst✝⁸ : BorelSpace β
                                            inst✝⁷ : TopologicalSpace.PseudoMetrizableSpace γ
                                            inst✝⁶ : SecondCountableTopologyEither β γ
                                            inst✝⁵ : MeasurableSpace γ
                                            inst✝⁴ : BorelSpace γ
                                            inst✝³ : TopologicalSpace.PseudoMetrizableSpace δ
                                            inst✝² : SecondCountableTopology δ
                                            inst✝¹ : MeasurableSpace δ
                                            inst✝ : OpensMeasurableSpace δ
                                            g : β → γ → δ
                                            hg : Measurable (Function.uncurry g)
                                            f₁✝ : MeasureTheory.AEEqFun α β μ
                                            f₂✝ : MeasureTheory.AEEqFun α γ μ
                                            f₁ : α → β
                                            x✝¹ : MeasureTheory.AEStronglyMeasurable f₁ μ
                                            f₂ : α → γ
                                            x✝ : MeasureTheory.AEStronglyMeasurable f₂ μ
                                            ⊢ Eq (MeasureTheory.AEEqFun.comp₂Measurable g hg (MeasureTheory.AEEqFun.mk f₁  …
                                          -/
  induction_on₂ f₁ f₂ fun f₁ _ f₂ _ => by simp
                                          /-
                                            🎉 no goals
                                          -/


/-- Given a predicate `p` and an equivalence class `[f]`, return true if `p` holds of `f a`
    for almost all `a` -/
def LiftPred (p : β → Prop) (f : α →ₘ[μ] β) : Prop :=
  f.toGerm.LiftPred p


/-- Given a relation `r` and equivalence class `[f]` and `[g]`, return true if `r` holds of
    `(f a, g a)` for almost all `a` -/
def LiftRel (r : β → γ → Prop) (f : α →ₘ[μ] β) (g : α →ₘ[μ] γ) : Prop :=
  f.toGerm.LiftRel r g.toGerm


theorem liftRel_mk_mk {r : β → γ → Prop} {f : α → β} {g : α → γ} {hf hg} :
    LiftRel r (mk f hf : α →ₘ[μ] β) (mk g hg) ↔ ∀ᵐ a ∂μ, r (f a) (g a) :=
  Iff.rfl


theorem liftRel_iff_coeFn {r : β → γ → Prop} {f : α →ₘ[μ] β} {g : α →ₘ[μ] γ} :
                                                 /-
                                                   α : Type u_1
                                                   β : Type u_2
                                                   γ : Type u_3
                                                   inst✝² : MeasurableSpace α
                                                   μ : MeasureTheory.Measure α
                                                   inst✝¹ : TopologicalSpace β
                                                   inst✝ : TopologicalSpace γ
                                                   r : β → γ → Prop
                                                   f : MeasureTheory.AEEqFun α β μ
                                                   g : MeasureTheory.AEEqFun α γ μ
                                                   ⊢ Iff (MeasureTheory.AEEqFun.LiftRel r f g) (Filter.Eventually (fun a => r (↑f …
                                                 -/
    LiftRel r f g ↔ ∀ᵐ a ∂μ, r (f a) (g a) := by rw [← liftRel_mk_mk, mk_coeFn, mk_coeFn]
                                                 /-
                                                   🎉 no goals
                                                 -/


instance instPreorder [Preorder β] : Preorder (α →ₘ[μ] β) :=
  Preorder.lift toGerm


@[simp]
theorem mk_le_mk [Preorder β] {f g : α → β} (hf hg) : (mk f hf : α →ₘ[μ] β) ≤ mk g hg ↔ f ≤ᵐ[μ] g :=
  Iff.rfl


@[simp, norm_cast]
theorem coeFn_le [Preorder β] {f g : α →ₘ[μ] β} : (f : α → β) ≤ᵐ[μ] g ↔ f ≤ g :=
  liftRel_iff_coeFn.symm


instance instPartialOrder [PartialOrder β] : PartialOrder (α →ₘ[μ] β) :=
  PartialOrder.lift toGerm toGerm_injective


instance instSup : Max (α →ₘ[μ] β) where max f g := AEEqFun.comp₂ (· ⊔ ·) continuous_sup f g


theorem coeFn_sup (f g : α →ₘ[μ] β) : ⇑(f ⊔ g) =ᵐ[μ] fun x => f x ⊔ g x :=
  coeFn_comp₂ _ _ _ _


protected theorem le_sup_left (f g : α →ₘ[μ] β) : f ≤ f ⊔ g := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace β
    inst✝¹ : SemilatticeSup β
    inst✝ : ContinuousSup β
    f g : MeasureTheory.AEEqFun α β μ
    ⊢ LE.le f (Max.max f g)
  -/
  rw [← coeFn_le]
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace β
    inst✝¹ : SemilatticeSup β
    inst✝ : ContinuousSup β
    f g : MeasureTheory.AEEqFun α β μ
    ⊢ (MeasureTheory.ae μ).EventuallyLE ↑f ↑(Max.max f g)
  -/
  filter_upwards [coeFn_sup f g] with _ ha
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace β
    inst✝¹ : SemilatticeSup β
    inst✝ : ContinuousSup β
    f g : MeasureTheory.AEEqFun α β μ
    a✝ : α
    ha : Eq (↑(Max.max f g) a✝) (Max.max (↑f a✝) (↑g a✝))
    ⊢ LE.le (↑f a✝) (↑(Max.max f g) a✝)
  -/
  rw [ha]
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace β
    inst✝¹ : SemilatticeSup β
    inst✝ : ContinuousSup β
    f g : MeasureTheory.AEEqFun α β μ
    a✝ : α
    ha : Eq (↑(Max.max f g) a✝) (Max.max (↑f a✝) (↑g a✝))
    ⊢ LE.le (↑f a✝) (Max.max (↑f a✝) (↑g a✝))
  -/
  exact le_sup_left
  /-
    🎉 no goals
  -/


protected theorem le_sup_right (f g : α →ₘ[μ] β) : g ≤ f ⊔ g := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace β
    inst✝¹ : SemilatticeSup β
    inst✝ : ContinuousSup β
    f g : MeasureTheory.AEEqFun α β μ
    ⊢ LE.le g (Max.max f g)
  -/
  rw [← coeFn_le]
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace β
    inst✝¹ : SemilatticeSup β
    inst✝ : ContinuousSup β
    f g : MeasureTheory.AEEqFun α β μ
    ⊢ (MeasureTheory.ae μ).EventuallyLE ↑g ↑(Max.max f g)
  -/
  filter_upwards [coeFn_sup f g] with _ ha
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace β
    inst✝¹ : SemilatticeSup β
    inst✝ : ContinuousSup β
    f g : MeasureTheory.AEEqFun α β μ
    a✝ : α
    ha : Eq (↑(Max.max f g) a✝) (Max.max (↑f a✝) (↑g a✝))
    ⊢ LE.le (↑g a✝) (↑(Max.max f g) a✝)
  -/
  rw [ha]
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace β
    inst✝¹ : SemilatticeSup β
    inst✝ : ContinuousSup β
    f g : MeasureTheory.AEEqFun α β μ
    a✝ : α
    ha : Eq (↑(Max.max f g) a✝) (Max.max (↑f a✝) (↑g a✝))
    ⊢ LE.le (↑g a✝) (Max.max (↑f a✝) (↑g a✝))
  -/
  exact le_sup_right
  /-
    🎉 no goals
  -/


protected theorem sup_le (f g f' : α →ₘ[μ] β) (hf : f ≤ f') (hg : g ≤ f') : f ⊔ g ≤ f' := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace β
    inst✝¹ : SemilatticeSup β
    inst✝ : ContinuousSup β
    f g f' : MeasureTheory.AEEqFun α β μ
    hf : LE.le f f'
    hg : LE.le g f'
    ⊢ LE.le (Max.max f g) f'
  -/
  rw [← coeFn_le] at hf hg ⊢
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace β
    inst✝¹ : SemilatticeSup β
    inst✝ : ContinuousSup β
    f g f' : MeasureTheory.AEEqFun α β μ
    hf : (MeasureTheory.ae μ).EventuallyLE ↑f ↑f'
    hg : (MeasureTheory.ae μ).EventuallyLE ↑g ↑f'
    ⊢ (MeasureTheory.ae μ).EventuallyLE ↑(Max.max f g) ↑f'
  -/
  filter_upwards [hf, hg, coeFn_sup f g] with _ haf hag ha_sup
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace β
    inst✝¹ : SemilatticeSup β
    inst✝ : ContinuousSup β
    f g f' : MeasureTheory.AEEqFun α β μ
    hf : (MeasureTheory.ae μ).EventuallyLE ↑f ↑f'
    hg : (MeasureTheory.ae μ).EventuallyLE ↑g ↑f'
    a✝ : α
    haf : LE.le (↑f a✝) (↑f' a✝)
    hag : LE.le (↑g a✝) (↑f' a✝)
    ha_sup : Eq (↑(Max.max f g) a✝) (Max.max (↑f a✝) (↑g a✝))
    ⊢ LE.le (↑(Max.max f g) a✝) (↑f' a✝)
  -/
  rw [ha_sup]
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace β
    inst✝¹ : SemilatticeSup β
    inst✝ : ContinuousSup β
    f g f' : MeasureTheory.AEEqFun α β μ
    hf : (MeasureTheory.ae μ).EventuallyLE ↑f ↑f'
    hg : (MeasureTheory.ae μ).EventuallyLE ↑g ↑f'
    a✝ : α
    haf : LE.le (↑f a✝) (↑f' a✝)
    hag : LE.le (↑g a✝) (↑f' a✝)
    ha_sup : Eq (↑(Max.max f g) a✝) (Max.max (↑f a✝) (↑g a✝))
    ⊢ LE.le (Max.max (↑f a✝) (↑g a✝)) (↑f' a✝)
  -/
  exact sup_le haf hag
  /-
    🎉 no goals
  -/


instance instInf : Min (α →ₘ[μ] β) where min f g := AEEqFun.comp₂ (· ⊓ ·) continuous_inf f g


theorem coeFn_inf (f g : α →ₘ[μ] β) : ⇑(f ⊓ g) =ᵐ[μ] fun x => f x ⊓ g x :=
  coeFn_comp₂ _ _ _ _


protected theorem inf_le_left (f g : α →ₘ[μ] β) : f ⊓ g ≤ f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace β
    inst✝¹ : SemilatticeInf β
    inst✝ : ContinuousInf β
    f g : MeasureTheory.AEEqFun α β μ
    ⊢ LE.le (Min.min f g) f
  -/
  rw [← coeFn_le]
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace β
    inst✝¹ : SemilatticeInf β
    inst✝ : ContinuousInf β
    f g : MeasureTheory.AEEqFun α β μ
    ⊢ (MeasureTheory.ae μ).EventuallyLE ↑(Min.min f g) ↑f
  -/
  filter_upwards [coeFn_inf f g] with _ ha
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace β
    inst✝¹ : SemilatticeInf β
    inst✝ : ContinuousInf β
    f g : MeasureTheory.AEEqFun α β μ
    a✝ : α
    ha : Eq (↑(Min.min f g) a✝) (Min.min (↑f a✝) (↑g a✝))
    ⊢ LE.le (↑(Min.min f g) a✝) (↑f a✝)
  -/
  rw [ha]
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace β
    inst✝¹ : SemilatticeInf β
    inst✝ : ContinuousInf β
    f g : MeasureTheory.AEEqFun α β μ
    a✝ : α
    ha : Eq (↑(Min.min f g) a✝) (Min.min (↑f a✝) (↑g a✝))
    ⊢ LE.le (Min.min (↑f a✝) (↑g a✝)) (↑f a✝)
  -/
  exact inf_le_left
  /-
    🎉 no goals
  -/


protected theorem inf_le_right (f g : α →ₘ[μ] β) : f ⊓ g ≤ g := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace β
    inst✝¹ : SemilatticeInf β
    inst✝ : ContinuousInf β
    f g : MeasureTheory.AEEqFun α β μ
    ⊢ LE.le (Min.min f g) g
  -/
  rw [← coeFn_le]
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace β
    inst✝¹ : SemilatticeInf β
    inst✝ : ContinuousInf β
    f g : MeasureTheory.AEEqFun α β μ
    ⊢ (MeasureTheory.ae μ).EventuallyLE ↑(Min.min f g) ↑g
  -/
  filter_upwards [coeFn_inf f g] with _ ha
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace β
    inst✝¹ : SemilatticeInf β
    inst✝ : ContinuousInf β
    f g : MeasureTheory.AEEqFun α β μ
    a✝ : α
    ha : Eq (↑(Min.min f g) a✝) (Min.min (↑f a✝) (↑g a✝))
    ⊢ LE.le (↑(Min.min f g) a✝) (↑g a✝)
  -/
  rw [ha]
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace β
    inst✝¹ : SemilatticeInf β
    inst✝ : ContinuousInf β
    f g : MeasureTheory.AEEqFun α β μ
    a✝ : α
    ha : Eq (↑(Min.min f g) a✝) (Min.min (↑f a✝) (↑g a✝))
    ⊢ LE.le (Min.min (↑f a✝) (↑g a✝)) (↑g a✝)
  -/
  exact inf_le_right
  /-
    🎉 no goals
  -/


protected theorem le_inf (f' f g : α →ₘ[μ] β) (hf : f' ≤ f) (hg : f' ≤ g) : f' ≤ f ⊓ g := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace β
    inst✝¹ : SemilatticeInf β
    inst✝ : ContinuousInf β
    f' f g : MeasureTheory.AEEqFun α β μ
    hf : LE.le f' f
    hg : LE.le f' g
    ⊢ LE.le f' (Min.min f g)
  -/
  rw [← coeFn_le] at hf hg ⊢
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace β
    inst✝¹ : SemilatticeInf β
    inst✝ : ContinuousInf β
    f' f g : MeasureTheory.AEEqFun α β μ
    hf : (MeasureTheory.ae μ).EventuallyLE ↑f' ↑f
    hg : (MeasureTheory.ae μ).EventuallyLE ↑f' ↑g
    ⊢ (MeasureTheory.ae μ).EventuallyLE ↑f' ↑(Min.min f g)
  -/
  filter_upwards [hf, hg, coeFn_inf f g] with _ haf hag ha_inf
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace β
    inst✝¹ : SemilatticeInf β
    inst✝ : ContinuousInf β
    f' f g : MeasureTheory.AEEqFun α β μ
    hf : (MeasureTheory.ae μ).EventuallyLE ↑f' ↑f
    hg : (MeasureTheory.ae μ).EventuallyLE ↑f' ↑g
    a✝ : α
    haf : LE.le (↑f' a✝) (↑f a✝)
    hag : LE.le (↑f' a✝) (↑g a✝)
    ha_inf : Eq (↑(Min.min f g) a✝) (Min.min (↑f a✝) (↑g a✝))
    ⊢ LE.le (↑f' a✝) (↑(Min.min f g) a✝)
  -/
  rw [ha_inf]
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝³ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝² : TopologicalSpace β
    inst✝¹ : SemilatticeInf β
    inst✝ : ContinuousInf β
    f' f g : MeasureTheory.AEEqFun α β μ
    hf : (MeasureTheory.ae μ).EventuallyLE ↑f' ↑f
    hg : (MeasureTheory.ae μ).EventuallyLE ↑f' ↑g
    a✝ : α
    haf : LE.le (↑f' a✝) (↑f a✝)
    hag : LE.le (↑f' a✝) (↑g a✝)
    ha_inf : Eq (↑(Min.min f g) a✝) (Min.min (↑f a✝) (↑g a✝))
    ⊢ LE.le (↑f' a✝) (Min.min (↑f a✝) (↑g a✝))
  -/
  exact le_inf haf hag
  /-
    🎉 no goals
  -/


instance instLattice [Lattice β] [TopologicalLattice β] : Lattice (α →ₘ[μ] β) :=
  { AEEqFun.instPartialOrder with
    sup := max
    le_sup_left := AEEqFun.le_sup_left
    le_sup_right := AEEqFun.le_sup_right
    sup_le := AEEqFun.sup_le
    inf := min
    inf_le_left := AEEqFun.inf_le_left
    inf_le_right := AEEqFun.inf_le_right
    le_inf := AEEqFun.le_inf }


/-- The equivalence class of a constant function: `[fun _ : α => b]`, based on the equivalence
relation of being almost everywhere equal -/
def const (b : β) : α →ₘ[μ] β :=
  mk (fun _ : α ↦ b) aestronglyMeasurable_const


theorem coeFn_const (b : β) : (const α b : α →ₘ[μ] β) =ᵐ[μ] Function.const α b :=
  coeFn_mk _ _


/-- If the measure is nonzero, we can strengthen `coeFn_const` to get an equality. -/
@[simp]
theorem coeFn_const_eq [NeZero μ] (b : β) (x : α) : (const α b : α →ₘ[μ] β) x = b := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : TopologicalSpace β
    inst✝ : NeZero μ
    b : β
    x : α
    ⊢ Eq (↑(MeasureTheory.AEEqFun.const α b) x) b
  -/
  simp only [cast]
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : TopologicalSpace β
    inst✝ : NeZero μ
    b : β
    x : α
    ⊢ Eq (dite (Exists fun b_1 => Eq (MeasureTheory.AEEqFun.const α b) (MeasureThe …
  -/
  split_ifs with h; swap; · exact h.elim ⟨b, rfl⟩
                            /-
                              🎉 no goals
                            -/
  /-
    case pos
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : TopologicalSpace β
    inst✝ : NeZero μ
    b : β
    x : α
    h : Exists fun b_1 => Eq (MeasureTheory.AEEqFun.const α b) (MeasureTheory.AEEq …
    ⊢ Eq (Function.const α (Classical.choose h) x) b
  -/
  have := Classical.choose_spec h
  /-
    case pos
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : TopologicalSpace β
    inst✝ : NeZero μ
    b : β
    x : α
    h : Exists fun b_1 => Eq (MeasureTheory.AEEqFun.const α b) (MeasureTheory.AEEq …
    this : Eq (MeasureTheory.AEEqFun.const α b) (MeasureTheory.AEEqFun.mk (Functio …
    ⊢ Eq (Function.const α (Classical.choose h) x) b
  -/
  set b' := Classical.choose h
  /-
    case pos
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : TopologicalSpace β
    inst✝ : NeZero μ
    b : β
    x : α
    h : Exists fun b_1 => Eq (MeasureTheory.AEEqFun.const α b) (MeasureTheory.AEEq …
    b' : β := Classical.choose h
    this : Eq (MeasureTheory.AEEqFun.const α b) (MeasureTheory.AEEqFun.mk (Functio …
    ⊢ Eq (Function.const α b' x) b
  -/
  simp_rw [const, mk_eq_mk, EventuallyEq, ← const_def, eventually_const] at this
  /-
    case pos
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : TopologicalSpace β
    inst✝ : NeZero μ
    b : β
    x : α
    h : Exists fun b_1 => Eq (MeasureTheory.AEEqFun.const α b) (MeasureTheory.AEEq …
    b' : β := Classical.choose h
    this : Eq b b'
    ⊢ Eq (Function.const α b' x) b
  -/
  rw [Function.const, this]
  /-
    🎉 no goals
  -/


instance instInhabited [Inhabited β] : Inhabited (α →ₘ[μ] β) :=
  ⟨const α default⟩


@[to_additive]
instance instOne [One β] : One (α →ₘ[μ] β) :=
  ⟨const α 1⟩


@[to_additive]
theorem one_def [One β] : (1 : α →ₘ[μ] β) = mk (fun _ : α => 1) aestronglyMeasurable_const :=
  rfl


@[to_additive]
theorem coeFn_one [One β] : ⇑(1 : α →ₘ[μ] β) =ᵐ[μ] 1 :=
  coeFn_const ..


@[to_additive (attr := simp)]
theorem coeFn_one_eq [NeZero μ] [One β] {x : α} : (1 : α →ₘ[μ] β) x = 1 :=
  coeFn_const_eq ..


@[to_additive (attr := simp)]
theorem one_toGerm [One β] : (1 : α →ₘ[μ] β).toGerm = 1 :=
  rfl

-- Note we set up the scalar actions before the `Monoid` structures in case we want to
-- try to override the `nsmul` or `zsmul` fields in future.

instance instSMul : SMul 𝕜 (α →ₘ[μ] γ) :=
  ⟨fun c f => comp (c • ·) (continuous_id.const_smul c) f⟩


@[simp]
theorem smul_mk (c : 𝕜) (f : α → γ) (hf : AEStronglyMeasurable f μ) :
    c • (mk f hf : α →ₘ[μ] γ) = mk (c • f) (hf.const_smul _) :=
  rfl


theorem coeFn_smul (c : 𝕜) (f : α →ₘ[μ] γ) : ⇑(c • f) =ᵐ[μ] c • ⇑f :=
  coeFn_comp _ _ _


theorem smul_toGerm (c : 𝕜) (f : α →ₘ[μ] γ) : (c • f).toGerm = c • f.toGerm :=
  comp_toGerm _ _ _


instance instSMulCommClass [SMulCommClass 𝕜 𝕜' γ] : SMulCommClass 𝕜 𝕜' (α →ₘ[μ] γ) :=
                                              /-
                                                α : Type u_1
                                                β : Type u_2
                                                γ : Type u_3
                                                δ : Type u_4
                                                inst✝⁸ : MeasurableSpace α
                                                μ ν : MeasureTheory.Measure α
                                                inst✝⁷ : TopologicalSpace δ
                                                inst✝⁶ : TopologicalSpace β
                                                inst✝⁵ : TopologicalSpace γ
                                                𝕜 : Type u_5
                                                𝕜' : Type u_6
                                                inst✝⁴ : SMul 𝕜 γ
                                                inst✝³ : ContinuousConstSMul 𝕜 γ
                                                inst✝² : SMul 𝕜' γ
                                                inst✝¹ : ContinuousConstSMul 𝕜' γ
                                                inst✝ : SMulCommClass 𝕜 𝕜' γ
                                                a : 𝕜
                                                b : 𝕜'
                                                f✝ : MeasureTheory.AEEqFun α γ μ
                                                f : α → γ
                                                hf : MeasureTheory.AEStronglyMeasurable f μ
                                                ⊢ Eq (HSMul.hSMul a (HSMul.hSMul b (MeasureTheory.AEEqFun.mk f hf))) (HSMul.hS …
                                              -/
  ⟨fun a b f => induction_on f fun f hf => by simp_rw [smul_mk, smul_comm]⟩
                                              /-
                                                🎉 no goals
                                              -/


instance instIsScalarTower [SMul 𝕜 𝕜'] [IsScalarTower 𝕜 𝕜' γ] : IsScalarTower 𝕜 𝕜' (α →ₘ[μ] γ) :=
                                              /-
                                                α : Type u_1
                                                β : Type u_2
                                                γ : Type u_3
                                                δ : Type u_4
                                                inst✝⁹ : MeasurableSpace α
                                                μ ν : MeasureTheory.Measure α
                                                inst✝⁸ : TopologicalSpace δ
                                                inst✝⁷ : TopologicalSpace β
                                                inst✝⁶ : TopologicalSpace γ
                                                𝕜 : Type u_5
                                                𝕜' : Type u_6
                                                inst✝⁵ : SMul 𝕜 γ
                                                inst✝⁴ : ContinuousConstSMul 𝕜 γ
                                                inst✝³ : SMul 𝕜' γ
                                                inst✝² : ContinuousConstSMul 𝕜' γ
                                                inst✝¹ : SMul 𝕜 𝕜'
                                                inst✝ : IsScalarTower 𝕜 𝕜' γ
                                                a : 𝕜
                                                b : 𝕜'
                                                f✝ : MeasureTheory.AEEqFun α γ μ
                                                f : α → γ
                                                hf : MeasureTheory.AEStronglyMeasurable f μ
                                                ⊢ Eq (HSMul.hSMul (HSMul.hSMul a b) (MeasureTheory.AEEqFun.mk f hf)) (HSMul.hS …
                                              -/
  ⟨fun a b f => induction_on f fun f hf => by simp_rw [smul_mk, smul_assoc]⟩
                                              /-
                                                🎉 no goals
                                              -/


instance instIsCentralScalar [SMul 𝕜ᵐᵒᵖ γ] [IsCentralScalar 𝕜 γ] : IsCentralScalar 𝕜 (α →ₘ[μ] γ) :=
                                            /-
                                              α : Type u_1
                                              β : Type u_2
                                              γ : Type u_3
                                              δ : Type u_4
                                              inst✝⁹ : MeasurableSpace α
                                              μ ν : MeasureTheory.Measure α
                                              inst✝⁸ : TopologicalSpace δ
                                              inst✝⁷ : TopologicalSpace β
                                              inst✝⁶ : TopologicalSpace γ
                                              𝕜 : Type u_5
                                              𝕜' : Type u_6
                                              inst✝⁵ : SMul 𝕜 γ
                                              inst✝⁴ : ContinuousConstSMul 𝕜 γ
                                              inst✝³ : SMul 𝕜' γ
                                              inst✝² : ContinuousConstSMul 𝕜' γ
                                              inst✝¹ : SMul (MulOpposite 𝕜) γ
                                              inst✝ : IsCentralScalar 𝕜 γ
                                              a : 𝕜
                                              f✝ : MeasureTheory.AEEqFun α γ μ
                                              f : α → γ
                                              hf : MeasureTheory.AEStronglyMeasurable f μ
                                              ⊢ Eq (HSMul.hSMul (MulOpposite.op a) (MeasureTheory.AEEqFun.mk f hf)) (HSMul.h …
                                            -/
  ⟨fun a f => induction_on f fun f hf => by simp_rw [smul_mk, op_smul_eq_smul]⟩
                                            /-
                                              🎉 no goals
                                            -/


@[to_additive]
instance instMul : Mul (α →ₘ[μ] γ) :=
  ⟨comp₂ (· * ·) continuous_mul⟩


@[to_additive (attr := simp)]
theorem mk_mul_mk (f g : α → γ) (hf : AEStronglyMeasurable f μ) (hg : AEStronglyMeasurable g μ) :
    (mk f hf : α →ₘ[μ] γ) * mk g hg = mk (f * g) (hf.mul hg) :=
  rfl


@[to_additive]
theorem coeFn_mul (f g : α →ₘ[μ] γ) : ⇑(f * g) =ᵐ[μ] f * g :=
  coeFn_comp₂ _ _ _ _


@[to_additive (attr := simp)]
theorem mul_toGerm (f g : α →ₘ[μ] γ) : (f * g).toGerm = f.toGerm * g.toGerm :=
  comp₂_toGerm _ _ _ _


instance instAddMonoid [AddMonoid γ] [ContinuousAdd γ] : AddMonoid (α →ₘ[μ] γ) :=
  toGerm_injective.addMonoid toGerm zero_toGerm add_toGerm fun _ _ => smul_toGerm _ _


instance instAddCommMonoid [AddCommMonoid γ] [ContinuousAdd γ] : AddCommMonoid (α →ₘ[μ] γ) :=
  toGerm_injective.addCommMonoid toGerm zero_toGerm add_toGerm fun _ _ => smul_toGerm _ _


instance instPowNat : Pow (α →ₘ[μ] γ) ℕ :=
  ⟨fun f n => comp _ (continuous_pow n) f⟩


@[simp]
theorem mk_pow (f : α → γ) (hf) (n : ℕ) :
    (mk f hf : α →ₘ[μ] γ) ^ n =
      mk (f ^ n) ((_root_.continuous_pow n).comp_aestronglyMeasurable hf) :=
  rfl


theorem coeFn_pow (f : α →ₘ[μ] γ) (n : ℕ) : ⇑(f ^ n) =ᵐ[μ] (⇑f) ^ n :=
  coeFn_comp _ _ _


@[simp]
theorem pow_toGerm (f : α →ₘ[μ] γ) (n : ℕ) : (f ^ n).toGerm = f.toGerm ^ n :=
  comp_toGerm _ _ _


@[to_additive existing]
instance instMonoid : Monoid (α →ₘ[μ] γ) :=
  toGerm_injective.monoid toGerm one_toGerm mul_toGerm pow_toGerm


/-- `AEEqFun.toGerm` as a `MonoidHom`. -/
@[to_additive (attr := simps) "`AEEqFun.toGerm` as an `AddMonoidHom`."]
def toGermMonoidHom : (α →ₘ[μ] γ) →* (ae μ).Germ γ where
  toFun := toGerm
  map_one' := one_toGerm
  map_mul' := mul_toGerm


@[to_additive existing]
instance instCommMonoid [CommMonoid γ] [ContinuousMul γ] : CommMonoid (α →ₘ[μ] γ) :=
  toGerm_injective.commMonoid toGerm one_toGerm mul_toGerm pow_toGerm


@[to_additive]
instance instInv : Inv (α →ₘ[μ] γ) :=
  ⟨comp Inv.inv continuous_inv⟩


@[to_additive (attr := simp)]
theorem inv_mk (f : α → γ) (hf) : (mk f hf : α →ₘ[μ] γ)⁻¹ = mk f⁻¹ hf.inv :=
  rfl


@[to_additive]
theorem coeFn_inv (f : α →ₘ[μ] γ) : ⇑f⁻¹ =ᵐ[μ] f⁻¹ :=
  coeFn_comp _ _ _


@[to_additive]
theorem inv_toGerm (f : α →ₘ[μ] γ) : f⁻¹.toGerm = f.toGerm⁻¹ :=
  comp_toGerm _ _ _


@[to_additive]
instance instDiv : Div (α →ₘ[μ] γ) :=
  ⟨comp₂ Div.div continuous_div'⟩


@[to_additive (attr := simp, nolint simpNF)] -- Porting note: LHS does not simplify.
theorem mk_div (f g : α → γ) (hf : AEStronglyMeasurable f μ) (hg : AEStronglyMeasurable g μ) :
    mk (f / g) (hf.div hg) = (mk f hf : α →ₘ[μ] γ) / mk g hg :=
  rfl


@[to_additive]
theorem coeFn_div (f g : α →ₘ[μ] γ) : ⇑(f / g) =ᵐ[μ] f / g :=
  coeFn_comp₂ _ _ _ _


@[to_additive]
theorem div_toGerm (f g : α →ₘ[μ] γ) : (f / g).toGerm = f.toGerm / g.toGerm :=
  comp₂_toGerm _ _ _ _


instance instPowInt : Pow (α →ₘ[μ] γ) ℤ :=
  ⟨fun f n => comp _ (continuous_zpow n) f⟩


@[simp]
theorem mk_zpow (f : α → γ) (hf) (n : ℤ) :
    (mk f hf : α →ₘ[μ] γ) ^ n = mk (f ^ n) ((continuous_zpow n).comp_aestronglyMeasurable hf) :=
  rfl


theorem coeFn_zpow (f : α →ₘ[μ] γ) (n : ℤ) : ⇑(f ^ n) =ᵐ[μ] (⇑f) ^ n :=
  coeFn_comp _ _ _


@[simp]
theorem zpow_toGerm (f : α →ₘ[μ] γ) (n : ℤ) : (f ^ n).toGerm = f.toGerm ^ n :=
  comp_toGerm _ _ _


instance instAddGroup [AddGroup γ] [TopologicalAddGroup γ] : AddGroup (α →ₘ[μ] γ) :=
  toGerm_injective.addGroup toGerm zero_toGerm add_toGerm neg_toGerm sub_toGerm
    (fun _ _ => smul_toGerm _ _) fun _ _ => smul_toGerm _ _


instance instAddCommGroup [AddCommGroup γ] [TopologicalAddGroup γ] : AddCommGroup (α →ₘ[μ] γ) :=
  { add_comm := add_comm }


@[to_additive existing]
instance instGroup [Group γ] [TopologicalGroup γ] : Group (α →ₘ[μ] γ) :=
  toGerm_injective.group _ one_toGerm mul_toGerm inv_toGerm div_toGerm pow_toGerm zpow_toGerm


@[to_additive existing]
instance instCommGroup [CommGroup γ] [TopologicalGroup γ] : CommGroup (α →ₘ[μ] γ) :=
  { mul_comm := mul_comm }


instance instMulAction [Monoid 𝕜] [MulAction 𝕜 γ] [ContinuousConstSMul 𝕜 γ] :
    MulAction 𝕜 (α →ₘ[μ] γ) :=
  toGerm_injective.mulAction toGerm smul_toGerm


instance instDistribMulAction [Monoid 𝕜] [AddMonoid γ] [ContinuousAdd γ] [DistribMulAction 𝕜 γ]
    [ContinuousConstSMul 𝕜 γ] : DistribMulAction 𝕜 (α →ₘ[μ] γ) :=
  toGerm_injective.distribMulAction (toGermAddMonoidHom : (α →ₘ[μ] γ) →+ _) fun c : 𝕜 =>
    smul_toGerm c


instance instModule [Semiring 𝕜] [AddCommMonoid γ] [ContinuousAdd γ] [Module 𝕜 γ]
    [ContinuousConstSMul 𝕜 γ] : Module 𝕜 (α →ₘ[μ] γ) :=
  toGerm_injective.module 𝕜 (toGermAddMonoidHom : (α →ₘ[μ] γ) →+ _) smul_toGerm


/-- For `f : α → ℝ≥0∞`, define `∫ [f]` to be `∫ f` -/
def lintegral (f : α →ₘ[μ] ℝ≥0∞) : ℝ≥0∞ :=
  Quotient.liftOn' f (fun f => ∫⁻ a, (f : α → ℝ≥0∞) a ∂μ) fun _ _ => lintegral_congr_ae


@[simp]
theorem lintegral_mk (f : α → ℝ≥0∞) (hf) : (mk f hf : α →ₘ[μ] ℝ≥0∞).lintegral = ∫⁻ a, f a ∂μ :=
  rfl


theorem lintegral_coeFn (f : α →ₘ[μ] ℝ≥0∞) : ∫⁻ a, f a ∂μ = f.lintegral := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : MeasureTheory.AEEqFun α ENNReal μ
    ⊢ Eq (MeasureTheory.lintegral μ fun a => ↑f a) f.lintegral
  -/
  rw [← lintegral_mk, mk_coeFn]
  /-
    🎉 no goals
  -/


@[simp]
nonrec theorem lintegral_zero : lintegral (0 : α →ₘ[μ] ℝ≥0∞) = 0 :=
  lintegral_zero


@[simp]
theorem lintegral_eq_zero_iff {f : α →ₘ[μ] ℝ≥0∞} : lintegral f = 0 ↔ f = 0 :=
  induction_on f fun _f hf => (lintegral_eq_zero_iff' hf.aemeasurable).trans mk_eq_mk.symm


theorem lintegral_add (f g : α →ₘ[μ] ℝ≥0∞) : lintegral (f + g) = lintegral f + lintegral g :=
                                       /-
                                         α : Type u_1
                                         inst✝ : MeasurableSpace α
                                         μ : MeasureTheory.Measure α
                                         f✝ g✝ : MeasureTheory.AEEqFun α ENNReal μ
                                         f : α → ENNReal
                                         hf : MeasureTheory.AEStronglyMeasurable f μ
                                         g : α → ENNReal
                                         x✝ : MeasureTheory.AEStronglyMeasurable g μ
                                         ⊢ Eq (HAdd.hAdd (MeasureTheory.AEEqFun.mk f hf) (MeasureTheory.AEEqFun.mk g x✝ …
                                       -/
  induction_on₂ f g fun f hf g _ => by simp [lintegral_add_left' hf.aemeasurable]
                                       /-
                                         🎉 no goals
                                       -/


theorem lintegral_mono {f g : α →ₘ[μ] ℝ≥0∞} : f ≤ g → lintegral f ≤ lintegral g :=
  induction_on₂ f g fun _f _ _g _ hfg => lintegral_mono_ae hfg


theorem coeFn_abs {β} [TopologicalSpace β] [Lattice β] [TopologicalLattice β] [AddGroup β]
    [TopologicalAddGroup β] (f : α →ₘ[μ] β) : ⇑|f| =ᵐ[μ] fun x => |f x| := by
  /-
    α : Type u_1
    inst✝⁵ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_5
    inst✝⁴ : TopologicalSpace β
    inst✝³ : Lattice β
    inst✝² : TopologicalLattice β
    inst✝¹ : AddGroup β
    inst✝ : TopologicalAddGroup β
    f : MeasureTheory.AEEqFun α β μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑(abs f) fun x => abs (↑f x)
  -/
  simp_rw [abs]
  /-
    α : Type u_1
    inst✝⁵ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_5
    inst✝⁴ : TopologicalSpace β
    inst✝³ : Lattice β
    inst✝² : TopologicalLattice β
    inst✝¹ : AddGroup β
    inst✝ : TopologicalAddGroup β
    f : MeasureTheory.AEEqFun α β μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑(Max.max f (Neg.neg f)) fun x => Max.max  …
  -/
  filter_upwards [AEEqFun.coeFn_sup f (-f), AEEqFun.coeFn_neg f] with x hx_sup hx_neg
  /-
    case h
    α : Type u_1
    inst✝⁵ : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_5
    inst✝⁴ : TopologicalSpace β
    inst✝³ : Lattice β
    inst✝² : TopologicalLattice β
    inst✝¹ : AddGroup β
    inst✝ : TopologicalAddGroup β
    f : MeasureTheory.AEEqFun α β μ
    x : α
    hx_sup : Eq (↑(Max.max f (Neg.neg f)) x) (Max.max (↑f x) (↑(Neg.neg f) x))
    hx_neg : Eq (↑(Neg.neg f) x) (Neg.neg (↑f) x)
    ⊢ Eq (↑(Max.max f (Neg.neg f)) x) (Max.max (↑f x) (Neg.neg (↑f x)))
  -/
  rw [hx_sup, hx_neg, Pi.neg_apply]
  /-
    🎉 no goals
  -/


/-- Positive part of an `AEEqFun`. -/
def posPart (f : α →ₘ[μ] γ) : α →ₘ[μ] γ :=
  comp (fun x => max x 0) (continuous_id.max continuous_const) f


@[simp]
theorem posPart_mk (f : α → γ) (hf) :
    posPart (mk f hf : α →ₘ[μ] γ) =
      mk (fun x => max (f x) 0)
        ((continuous_id.max continuous_const).comp_aestronglyMeasurable hf) :=
  rfl


theorem coeFn_posPart (f : α →ₘ[μ] γ) : ⇑(posPart f) =ᵐ[μ] fun a => max (f a) 0 :=
  coeFn_comp _ _ _


/-- The equivalence class of `μ`-almost-everywhere measurable functions associated to a continuous
map. -/
def toAEEqFun (f : C(α, β)) : α →ₘ[μ] β :=
  AEEqFun.mk f f.continuous.aestronglyMeasurable


theorem coeFn_toAEEqFun (f : C(α, β)) : f.toAEEqFun μ =ᵐ[μ] f :=
  AEEqFun.coeFn_mk f _


/-- The `MulHom` from the group of continuous maps from `α` to `β` to the group of equivalence
classes of `μ`-almost-everywhere measurable functions. -/
@[to_additive "The `AddHom` from the group of continuous maps from `α` to `β` to the group of
equivalence classes of `μ`-almost-everywhere measurable functions."]
def toAEEqFunMulHom : C(α, β) →* α →ₘ[μ] β where
  toFun := ContinuousMap.toAEEqFun μ
  map_one' := rfl
  map_mul' f g :=
    AEEqFun.mk_mul_mk _ _ f.continuous.aestronglyMeasurable g.continuous.aestronglyMeasurable


/-- The linear map from the group of continuous maps from `α` to `β` to the group of equivalence
classes of `μ`-almost-everywhere measurable functions. -/
def toAEEqFunLinearMap : C(α, γ) →ₗ[𝕜] α →ₘ[μ] γ :=
  { toAEEqFunAddHom μ with
    map_smul' := fun c f => AEEqFun.smul_mk c f f.continuous.aestronglyMeasurable }


