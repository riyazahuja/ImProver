/-- A function `f` from a measurable space to any type is called *simple*,
if every preimage `f ⁻¹' {x}` is measurable, and the range is finite. This structure bundles
a function with these properties. -/
structure SimpleFunc.{u, v} (α : Type u) [MeasurableSpace α] (β : Type v) where
  toFun : α → β
  measurableSet_fiber' : ∀ x, MeasurableSet (toFun ⁻¹' {x})
  finite_range' : (Set.range toFun).Finite


local infixr:25 " →ₛ " => SimpleFunc


instance instFunLike : FunLike (α →ₛ β) α β where
  coe := toFun
  coe_injective' | ⟨_, _, _⟩, ⟨_, _, _⟩, rfl => rfl


theorem coe_injective ⦃f g : α →ₛ β⦄ (H : (f : α → β) = g) : f = g := DFunLike.ext' H


@[ext]
theorem ext {f g : α →ₛ β} (H : ∀ a, f a = g a) : f = g := DFunLike.ext _ _ H


theorem finite_range (f : α →ₛ β) : (Set.range f).Finite :=
  f.finite_range'


theorem measurableSet_fiber (f : α →ₛ β) (x : β) : MeasurableSet (f ⁻¹' {x}) :=
  f.measurableSet_fiber' x


@[simp] theorem coe_mk (f : α → β) (h h') : ⇑(mk f h h') = f := rfl


theorem apply_mk (f : α → β) (h h') (x : α) : SimpleFunc.mk f h h' x = f x :=
  rfl


/-- Simple function defined on a finite type. -/
def ofFinite [Finite α] [MeasurableSingletonClass α] (f : α → β) : α →ₛ β where
  toFun := f
  measurableSet_fiber' x := (toFinite (f ⁻¹' {x})).measurableSet
  finite_range' := Set.finite_range f


@[deprecated (since := "2024-02-05")] alias ofFintype := ofFinite


/-- Simple function defined on the empty type. -/
def ofIsEmpty [IsEmpty α] : α →ₛ β := ofFinite isEmptyElim


/-- Range of a simple function `α →ₛ β` as a `Finset β`. -/
protected def range (f : α →ₛ β) : Finset β :=
  f.finite_range.toFinset


@[simp]
theorem mem_range {f : α →ₛ β} {b} : b ∈ f.range ↔ b ∈ range f :=
  Finite.mem_toFinset _


theorem mem_range_self (f : α →ₛ β) (x : α) : f x ∈ f.range :=
  mem_range.2 ⟨x, rfl⟩


@[simp]
theorem coe_range (f : α →ₛ β) : (↑f.range : Set β) = Set.range f :=
  f.finite_range.coe_toFinset


theorem mem_range_of_measure_ne_zero {f : α →ₛ β} {x : β} {μ : Measure α} (H : μ (f ⁻¹' {x}) ≠ 0) :
    x ∈ f.range :=
  let ⟨a, ha⟩ := nonempty_of_measure_ne_zero H
  mem_range.2 ⟨a, ha⟩


theorem forall_mem_range {f : α →ₛ β} {p : β → Prop} : (∀ y ∈ f.range, p y) ↔ ∀ x, p (f x) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : MeasurableSpace α
    f : MeasureTheory.SimpleFunc α β
    p : β → Prop
    ⊢ Iff (∀ (y : β), Membership.mem f.range y → p y) (∀ (x : α), p (f x))
  -/
  simp only [mem_range, Set.forall_mem_range]
  /-
    🎉 no goals
  -/


theorem exists_range_iff {f : α →ₛ β} {p : β → Prop} : (∃ y ∈ f.range, p y) ↔ ∃ x, p (f x) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : MeasurableSpace α
    f : MeasureTheory.SimpleFunc α β
    p : β → Prop
    ⊢ Iff (Exists fun y => And (Membership.mem f.range y) (p y)) (Exists fun x =>  …
  -/
  simpa only [mem_range, exists_prop] using Set.exists_range_iff
  /-
    🎉 no goals
  -/


theorem preimage_eq_empty_iff (f : α →ₛ β) (b : β) : f ⁻¹' {b} = ∅ ↔ b ∉ f.range :=
  preimage_singleton_eq_empty.trans <| not_congr mem_range.symm


theorem exists_forall_le [Nonempty β] [Preorder β] [IsDirected β (· ≤ ·)] (f : α →ₛ β) :
    ∃ C, ∀ x, f x ≤ C :=
  f.range.exists_le.imp fun _ => forall_mem_range.1


/-- Constant function as a `SimpleFunc`. -/
def const (α) {β} [MeasurableSpace α] (b : β) : α →ₛ β :=
  ⟨fun _ => b, fun _ => MeasurableSet.const _, finite_range_const⟩


instance instInhabited [Inhabited β] : Inhabited (α →ₛ β) :=
  ⟨const _ default⟩


theorem const_apply (a : α) (b : β) : (const α b) a = b :=
  rfl


@[simp]
theorem coe_const (b : β) : ⇑(const α b) = Function.const α b :=
  rfl


@[simp]
theorem range_const (α) [MeasurableSpace α] [Nonempty α] (b : β) : (const α b).range = {b} :=
                             /-
                               β : Type u_2
                               α : Type u_5
                               inst✝¹ : MeasurableSpace α
                               inst✝ : Nonempty α
                               b : β
                               ⊢ Eq ↑(MeasureTheory.SimpleFunc.const α b).range ↑(Singleton.singleton b)
                             -/
  Finset.coe_injective <| by simp (config := { unfoldPartialApp := true }) [Function.const]
                             /-
                               🎉 no goals
                             -/


theorem range_const_subset (α) [MeasurableSpace α] (b : β) : (const α b).range ⊆ {b} :=
                            /-
                              β : Type u_2
                              α : Type u_5
                              inst✝ : MeasurableSpace α
                              b : β
                              ⊢ HasSubset.Subset ↑(MeasureTheory.SimpleFunc.const α b).range ↑(Singleton.sin …
                            -/
  Finset.coe_subset.1 <| by simp
                            /-
                              🎉 no goals
                            -/


theorem simpleFunc_bot {α} (f : @SimpleFunc α ⊥ β) [Nonempty β] : ∃ c, ∀ x, f x = c := by
  /-
    β : Type u_2
    α : Type u_5
    f : MeasureTheory.SimpleFunc α β
    inst✝ : Nonempty β
    ⊢ Exists fun c => ∀ (x : α), Eq (f x) c
  -/
  have hf_meas := @SimpleFunc.measurableSet_fiber α _ ⊥ f
  /-
    β : Type u_2
    α : Type u_5
    f : MeasureTheory.SimpleFunc α β
    inst✝ : Nonempty β
    hf_meas : ∀ (x : β), MeasurableSet (Set.preimage (⇑f) (Singleton.singleton x))
    ⊢ Exists fun c => ∀ (x : α), Eq (f x) c
  -/
  simp_rw [MeasurableSpace.measurableSet_bot_iff] at hf_meas
  /-
    β : Type u_2
    α : Type u_5
    f : MeasureTheory.SimpleFunc α β
    inst✝ : Nonempty β
    hf_meas : ∀ (x : β), Or (Eq (Set.preimage (⇑f) (Singleton.singleton x)) EmptyC …
    ⊢ Exists fun c => ∀ (x : α), Eq (f x) c
  -/
  exact (exists_eq_const_of_preimage_singleton hf_meas).imp fun c hc ↦ congr_fun hc
  /-
    🎉 no goals
  -/


theorem simpleFunc_bot' {α} [Nonempty β] (f : @SimpleFunc α ⊥ β) :
    ∃ c, f = @SimpleFunc.const α _ ⊥ c :=
  letI : MeasurableSpace α := ⊥; (simpleFunc_bot f).imp fun _ ↦ ext


theorem measurableSet_cut (r : α → β → Prop) (f : α →ₛ β) (h : ∀ b, MeasurableSet { a | r a b }) :
    MeasurableSet { a | r a (f a) } := by
  have : { a | r a (f a) } = ⋃ b ∈ range f, { a | r a b } ∩ f ⁻¹' {b} := by
    ext a
    suffices r a (f a) ↔ ∃ i, r a (f i) ∧ f a = f i by simpa
    exact ⟨fun h => ⟨a, ⟨h, rfl⟩⟩, fun ⟨a', ⟨h', e⟩⟩ => e.symm ▸ h'⟩
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : MeasurableSpace α
    r : α → β → Prop
    f : MeasureTheory.SimpleFunc α β
    h : ∀ (b : β), MeasurableSet (setOf fun a => r a b)
    this : Eq (setOf fun a => r a (f a)) (Set.iUnion fun b => Set.iUnion fun h =>  …
    ⊢ MeasurableSet (setOf fun a => r a (f a))
  -/
  rw [this]
  exact
    MeasurableSet.biUnion f.finite_range.countable fun b _ =>
      MeasurableSet.inter (h b) (f.measurableSet_fiber _)


@[measurability]
theorem measurableSet_preimage (f : α →ₛ β) (s) : MeasurableSet (f ⁻¹' s) :=
  measurableSet_cut (fun _ b => b ∈ s) f fun b => MeasurableSet.const (b ∈ s)


/-- A simple function is measurable -/
@[measurability, fun_prop]
protected theorem measurable [MeasurableSpace β] (f : α →ₛ β) : Measurable f := fun s _ =>
  measurableSet_preimage f s


@[measurability]
protected theorem aemeasurable [MeasurableSpace β] {μ : Measure α} (f : α →ₛ β) :
    AEMeasurable f μ :=
  f.measurable.aemeasurable


protected theorem sum_measure_preimage_singleton (f : α →ₛ β) {μ : Measure α} (s : Finset β) :
    (∑ y ∈ s, μ (f ⁻¹' {y})) = μ (f ⁻¹' ↑s) :=
  sum_measure_preimage_singleton _ fun _ _ => f.measurableSet_fiber _


theorem sum_range_measure_preimage_singleton (f : α →ₛ β) (μ : Measure α) :
    (∑ y ∈ f.range, μ (f ⁻¹' {y})) = μ univ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : MeasurableSpace α
    f : MeasureTheory.SimpleFunc α β
    μ : MeasureTheory.Measure α
    ⊢ Eq (f.range.sum fun y => μ (Set.preimage (⇑f) (Singleton.singleton y))) (μ S …
  -/
  rw [f.sum_measure_preimage_singleton, coe_range, preimage_range]
  /-
    🎉 no goals
  -/


/-- If-then-else as a `SimpleFunc`. -/
def piecewise (s : Set α) (hs : MeasurableSet s) (f g : α →ₛ β) : α →ₛ β :=
  ⟨s.piecewise f g, fun _ =>
    letI : MeasurableSpace β := ⊤
    f.measurable.piecewise hs g.measurable trivial,
    (f.finite_range.union g.finite_range).subset range_ite_subset⟩


@[simp]
theorem coe_piecewise {s : Set α} (hs : MeasurableSet s) (f g : α →ₛ β) :
    ⇑(piecewise s hs f g) = s.piecewise f g :=
  rfl


theorem piecewise_apply {s : Set α} (hs : MeasurableSet s) (f g : α →ₛ β) (a) :
    piecewise s hs f g a = if a ∈ s then f a else g a :=
  rfl


@[simp]
theorem piecewise_compl {s : Set α} (hs : MeasurableSet sᶜ) (f g : α →ₛ β) :
    piecewise sᶜ hs f g = piecewise s hs.of_compl g f :=
                      /-
                        α : Type u_1
                        β : Type u_2
                        inst✝ : MeasurableSpace α
                        s : Set α
                        hs : MeasurableSet (HasCompl.compl s)
                        f g : MeasureTheory.SimpleFunc α β
                        ⊢ Eq ⇑(MeasureTheory.SimpleFunc.piecewise (HasCompl.compl s) hs f g) ⇑(Measure …
                      -/
  coe_injective <| by simp [hs]
                      /-
                        🎉 no goals
                      -/


@[simp]
theorem piecewise_univ (f g : α →ₛ β) : piecewise univ MeasurableSet.univ f g = f :=
                      /-
                        α : Type u_1
                        β : Type u_2
                        inst✝ : MeasurableSpace α
                        f g : MeasureTheory.SimpleFunc α β
                        ⊢ Eq ⇑(MeasureTheory.SimpleFunc.piecewise Set.univ ⋯ f g) ⇑f
                      -/
  coe_injective <| by simp
                      /-
                        🎉 no goals
                      -/


@[simp]
theorem piecewise_empty (f g : α →ₛ β) : piecewise ∅ MeasurableSet.empty f g = g :=
                      /-
                        α : Type u_1
                        β : Type u_2
                        inst✝ : MeasurableSpace α
                        f g : MeasureTheory.SimpleFunc α β
                        ⊢ Eq ⇑(MeasureTheory.SimpleFunc.piecewise EmptyCollection.emptyCollection ⋯ f  …
                      -/
  coe_injective <| by simp
                      /-
                        🎉 no goals
                      -/


@[simp]
theorem piecewise_same (f : α →ₛ β) {s : Set α} (hs : MeasurableSet s) :
    piecewise s hs f f = f :=
  coe_injective <| Set.piecewise_same _ _


theorem support_indicator [Zero β] {s : Set α} (hs : MeasurableSet s) (f : α →ₛ β) :
    Function.support (f.piecewise s hs (SimpleFunc.const α 0)) = s ∩ Function.support f :=
  Set.support_indicator


theorem range_indicator {s : Set α} (hs : MeasurableSet s) (hs_nonempty : s.Nonempty)
    (hs_ne_univ : s ≠ univ) (x y : β) :
    (piecewise s hs (const α x) (const α y)).range = {x, y} := by
  simp only [← Finset.coe_inj, coe_range, coe_piecewise, range_piecewise, coe_const,
    Finset.coe_insert, Finset.coe_singleton, hs_nonempty.image_const,
    (nonempty_compl.2 hs_ne_univ).image_const, singleton_union, Function.const]


theorem measurable_bind [MeasurableSpace γ] (f : α →ₛ β) (g : β → α → γ)
    (hg : ∀ b, Measurable (g b)) : Measurable fun a => g (f a) a := fun s hs =>
  f.measurableSet_cut (fun a b => g b a ∈ s) fun b => hg b hs


/-- If `f : α →ₛ β` is a simple function and `g : β → α →ₛ γ` is a family of simple functions,
then `f.bind g` binds the first argument of `g` to `f`. In other words, `f.bind g a = g (f a) a`. -/
def bind (f : α →ₛ β) (g : β → α →ₛ γ) : α →ₛ γ :=
  ⟨fun a => g (f a) a, fun c =>
    f.measurableSet_cut (fun a b => g b a = c) fun b => (g b).measurableSet_preimage {c},
    (f.finite_range.biUnion fun b _ => (g b).finite_range).subset <| by
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        δ : Type u_4
        inst✝ : MeasurableSpace α
        f : MeasureTheory.SimpleFunc α β
        g : β → MeasureTheory.SimpleFunc α γ
        ⊢ HasSubset.Subset (Set.range fun a => (g (f a)) a) (Set.iUnion fun i => Set.i …
      -/
      rintro _ ⟨a, rfl⟩; simp⟩
                         /-
                           🎉 no goals
                         -/


@[simp]
theorem bind_apply (f : α →ₛ β) (g : β → α →ₛ γ) (a) : f.bind g a = g (f a) a :=
  rfl


/-- Given a function `g : β → γ` and a simple function `f : α →ₛ β`, `f.map g` return the simple
    function `g ∘ f : α →ₛ γ` -/
def map (g : β → γ) (f : α →ₛ β) : α →ₛ γ :=
  bind f (const α ∘ g)


theorem map_apply (g : β → γ) (f : α →ₛ β) (a) : f.map g a = g (f a) :=
  rfl


theorem map_map (g : β → γ) (h : γ → δ) (f : α →ₛ β) : (f.map g).map h = f.map (h ∘ g) :=
  rfl


@[simp]
theorem coe_map (g : β → γ) (f : α →ₛ β) : (f.map g : α → γ) = g ∘ f :=
  rfl


@[simp]
theorem range_map [DecidableEq γ] (g : β → γ) (f : α →ₛ β) : (f.map g).range = f.range.image g :=
                             /-
                               α : Type u_1
                               β : Type u_2
                               γ : Type u_3
                               inst✝¹ : MeasurableSpace α
                               inst✝ : DecidableEq γ
                               g : β → γ
                               f : MeasureTheory.SimpleFunc α β
                               ⊢ Eq ↑(MeasureTheory.SimpleFunc.map g f).range ↑(Finset.image g f.range)
                             -/
  Finset.coe_injective <| by simp only [coe_range, coe_map, Finset.coe_image, range_comp]
                             /-
                               🎉 no goals
                             -/


@[simp]
theorem map_const (g : β → γ) (b : β) : (const α b).map g = const α (g b) :=
  rfl


theorem map_preimage (f : α →ₛ β) (g : β → γ) (s : Set γ) :
    f.map g ⁻¹' s = f ⁻¹' ↑{b ∈ f.range | g b ∈ s} := by
  simp only [coe_range, sep_mem_eq, coe_map, Finset.coe_filter,
    ← mem_preimage, inter_comm, preimage_inter_range, ← Finset.mem_coe]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝ : MeasurableSpace α
    f : MeasureTheory.SimpleFunc α β
    g : β → γ
    s : Set γ
    ⊢ Eq (Set.preimage (Function.comp g ⇑f) s) (Set.preimage (⇑f) (Set.preimage g  …
  -/
  exact preimage_comp
  /-
    🎉 no goals
  -/


theorem map_preimage_singleton (f : α →ₛ β) (g : β → γ) (c : γ) :
    f.map g ⁻¹' {c} = f ⁻¹' ↑{b ∈ f.range | g b = c} :=
  map_preimage _ _ _


/-- Composition of a `SimpleFun` and a measurable function is a `SimpleFunc`. -/
def comp [MeasurableSpace β] (f : β →ₛ γ) (g : α → β) (hgm : Measurable g) : α →ₛ γ where
  toFun := f ∘ g
  finite_range' := f.finite_range.subset <| Set.range_comp_subset_range _ _
  measurableSet_fiber' z := hgm (f.measurableSet_fiber z)


@[simp]
theorem coe_comp [MeasurableSpace β] (f : β →ₛ γ) {g : α → β} (hgm : Measurable g) :
    ⇑(f.comp g hgm) = f ∘ g :=
  rfl


theorem range_comp_subset_range [MeasurableSpace β] (f : β →ₛ γ) {g : α → β} (hgm : Measurable g) :
    (f.comp g hgm).range ⊆ f.range :=
                            /-
                              α : Type u_1
                              β : Type u_2
                              γ : Type u_3
                              inst✝¹ : MeasurableSpace α
                              inst✝ : MeasurableSpace β
                              f : MeasureTheory.SimpleFunc β γ
                              g : α → β
                              hgm : Measurable g
                              ⊢ HasSubset.Subset ↑(f.comp g hgm).range ↑f.range
                            -/
  Finset.coe_subset.1 <| by simp only [coe_range, coe_comp, Set.range_comp_subset_range]
                            /-
                              🎉 no goals
                            -/


/-- Extend a `SimpleFunc` along a measurable embedding: `f₁.extend g hg f₂` is the function
`F : β →ₛ γ` such that `F ∘ g = f₁` and `F y = f₂ y` whenever `y ∉ range g`. -/
def extend [MeasurableSpace β] (f₁ : α →ₛ γ) (g : α → β) (hg : MeasurableEmbedding g)
    (f₂ : β →ₛ γ) : β →ₛ γ where
  toFun := Function.extend g f₁ f₂
  finite_range' :=
    (f₁.finite_range.union <| f₂.finite_range.subset (image_subset_range _ _)).subset
      (range_extend_subset _ _ _)
  measurableSet_fiber' := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      inst✝¹ : MeasurableSpace α
      inst✝ : MeasurableSpace β
      f₁ : MeasureTheory.SimpleFunc α γ
      g : α → β
      hg : MeasurableEmbedding g
      f₂ : MeasureTheory.SimpleFunc β γ
      ⊢ ∀ (x : γ), MeasurableSet (Set.preimage (Function.extend g ⇑f₁ ⇑f₂) (Singleto …
    -/
    letI : MeasurableSpace γ := ⊤; haveI : MeasurableSingletonClass γ := ⟨fun _ => trivial⟩
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      inst✝¹ : MeasurableSpace α
      inst✝ : MeasurableSpace β
      f₁ : MeasureTheory.SimpleFunc α γ
      g : α → β
      hg : MeasurableEmbedding g
      f₂ : MeasureTheory.SimpleFunc β γ
      this✝ : MeasurableSpace γ := Top.top
      this : MeasurableSingletonClass γ
      ⊢ ∀ (x : γ), MeasurableSet (Set.preimage (Function.extend g ⇑f₁ ⇑f₂) (Singleto …
    -/
    exact fun x => hg.measurable_extend f₁.measurable f₂.measurable (measurableSet_singleton _)
    /-
      🎉 no goals
    -/


@[simp]
theorem extend_apply [MeasurableSpace β] (f₁ : α →ₛ γ) {g : α → β} (hg : MeasurableEmbedding g)
    (f₂ : β →ₛ γ) (x : α) : (f₁.extend g hg f₂) (g x) = f₁ x :=
  hg.injective.extend_apply _ _ _


@[simp]
theorem extend_apply' [MeasurableSpace β] (f₁ : α →ₛ γ) {g : α → β} (hg : MeasurableEmbedding g)
    (f₂ : β →ₛ γ) {y : β} (h : ¬∃ x, g x = y) : (f₁.extend g hg f₂) y = f₂ y :=
  Function.extend_apply' _ _ _ h


@[simp]
theorem extend_comp_eq' [MeasurableSpace β] (f₁ : α →ₛ γ) {g : α → β} (hg : MeasurableEmbedding g)
    (f₂ : β →ₛ γ) : f₁.extend g hg f₂ ∘ g = f₁ :=
  funext fun _ => extend_apply _ _ _ _


@[simp]
theorem extend_comp_eq [MeasurableSpace β] (f₁ : α →ₛ γ) {g : α → β} (hg : MeasurableEmbedding g)
    (f₂ : β →ₛ γ) : (f₁.extend g hg f₂).comp g hg.measurable = f₁ :=
  coe_injective <| extend_comp_eq' _ hg _


/-- If `f` is a simple function taking values in `β → γ` and `g` is another simple function
with the same domain and codomain `β`, then `f.seq g = f a (g a)`. -/
def seq (f : α →ₛ β → γ) (g : α →ₛ β) : α →ₛ γ :=
  f.bind fun f => g.map f


@[simp]
theorem seq_apply (f : α →ₛ β → γ) (g : α →ₛ β) (a : α) : f.seq g a = f a (g a) :=
  rfl


/-- Combine two simple functions `f : α →ₛ β` and `g : α →ₛ β`
into `fun a => (f a, g a)`. -/
def pair (f : α →ₛ β) (g : α →ₛ γ) : α →ₛ β × γ :=
  (f.map Prod.mk).seq g


@[simp]
theorem pair_apply (f : α →ₛ β) (g : α →ₛ γ) (a) : pair f g a = (f a, g a) :=
  rfl


theorem pair_preimage (f : α →ₛ β) (g : α →ₛ γ) (s : Set β) (t : Set γ) :
    pair f g ⁻¹' s ×ˢ t = f ⁻¹' s ∩ g ⁻¹' t :=
  rfl

-- A special form of `pair_preimage`

theorem pair_preimage_singleton (f : α →ₛ β) (g : α →ₛ γ) (b : β) (c : γ) :
    pair f g ⁻¹' {(b, c)} = f ⁻¹' {b} ∩ g ⁻¹' {c} := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝ : MeasurableSpace α
    f : MeasureTheory.SimpleFunc α β
    g : MeasureTheory.SimpleFunc α γ
    b : β
    c : γ
    ⊢ Eq (Set.preimage (⇑(f.pair g)) (Singleton.singleton { fst := b, snd := c })) …
  -/
  rw [← singleton_prod_singleton]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝ : MeasurableSpace α
    f : MeasureTheory.SimpleFunc α β
    g : MeasureTheory.SimpleFunc α γ
    b : β
    c : γ
    ⊢ Eq (Set.preimage (⇑(f.pair g)) (SProd.sprod (Singleton.singleton b) (Singlet …
  -/
  exact pair_preimage _ _ _ _
  /-
    🎉 no goals
  -/


@[simp] theorem map_fst_pair (f : α →ₛ β) (g : α →ₛ γ) : (f.pair g).map Prod.fst = f := rfl

@[simp] theorem map_snd_pair (f : α →ₛ β) (g : α →ₛ γ) : (f.pair g).map Prod.snd = g := rfl


@[simp]
                                                             /-
                                                               α : Type u_1
                                                               β : Type u_2
                                                               inst✝ : MeasurableSpace α
                                                               f : MeasureTheory.SimpleFunc α β
                                                               ⊢ Eq (f.bind (MeasureTheory.SimpleFunc.const α)) f
                                                             -/
theorem bind_const (f : α →ₛ β) : f.bind (const α) = f := by ext; simp
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[to_additive]
instance instOne [One β] : One (α →ₛ β) :=
  ⟨const α 1⟩


@[to_additive]
instance instMul [Mul β] : Mul (α →ₛ β) :=
  ⟨fun f g => (f.map (· * ·)).seq g⟩


@[to_additive]
instance instDiv [Div β] : Div (α →ₛ β) :=
  ⟨fun f g => (f.map (· / ·)).seq g⟩


@[to_additive]
instance instInv [Inv β] : Inv (α →ₛ β) :=
  ⟨fun f => f.map Inv.inv⟩


instance instSup [Max β] : Max (α →ₛ β) :=
  ⟨fun f g => (f.map (· ⊔ ·)).seq g⟩


instance instInf [Min β] : Min (α →ₛ β) :=
  ⟨fun f g => (f.map (· ⊓ ·)).seq g⟩


instance instLE [LE β] : LE (α →ₛ β) :=
  ⟨fun f g => ∀ a, f a ≤ g a⟩


@[to_additive (attr := simp)]
theorem const_one [One β] : const α (1 : β) = 1 :=
  rfl


@[to_additive (attr := simp, norm_cast)]
theorem coe_one [One β] : ⇑(1 : α →ₛ β) = 1 :=
  rfl


@[to_additive (attr := simp, norm_cast)]
theorem coe_mul [Mul β] (f g : α →ₛ β) : ⇑(f * g) = ⇑f * ⇑g :=
  rfl


@[to_additive (attr := simp, norm_cast)]
theorem coe_inv [Inv β] (f : α →ₛ β) : ⇑(f⁻¹) = (⇑f)⁻¹ :=
  rfl


@[to_additive (attr := simp, norm_cast)]
theorem coe_div [Div β] (f g : α →ₛ β) : ⇑(f / g) = ⇑f / ⇑g :=
  rfl


@[simp, norm_cast]
theorem coe_le [Preorder β] {f g : α →ₛ β} : (f : α → β) ≤ g ↔ f ≤ g :=
  Iff.rfl


@[simp, norm_cast]
theorem coe_sup [Max β] (f g : α →ₛ β) : ⇑(f ⊔ g) = ⇑f ⊔ ⇑g :=
  rfl


@[simp, norm_cast]
theorem coe_inf [Min β] (f g : α →ₛ β) : ⇑(f ⊓ g) = ⇑f ⊓ ⇑g :=
  rfl


@[to_additive]
theorem mul_apply [Mul β] (f g : α →ₛ β) (a : α) : (f * g) a = f a * g a :=
  rfl


@[to_additive]
theorem div_apply [Div β] (f g : α →ₛ β) (x : α) : (f / g) x = f x / g x :=
  rfl


@[to_additive]
theorem inv_apply [Inv β] (f : α →ₛ β) (x : α) : f⁻¹ x = (f x)⁻¹ :=
  rfl


theorem sup_apply [Max β] (f g : α →ₛ β) (a : α) : (f ⊔ g) a = f a ⊔ g a :=
  rfl


theorem inf_apply [Min β] (f g : α →ₛ β) (a : α) : (f ⊓ g) a = f a ⊓ g a :=
  rfl


@[to_additive (attr := simp)]
theorem range_one [Nonempty α] [One β] : (1 : α →ₛ β).range = {1} :=
                         /-
                           α : Type u_1
                           β : Type u_2
                           inst✝² : MeasurableSpace α
                           inst✝¹ : Nonempty α
                           inst✝ : One β
                           x : β
                           ⊢ Iff (Membership.mem (MeasureTheory.SimpleFunc.range 1) x) (Membership.mem (S …
                         -/
  Finset.ext fun x => by simp [eq_comm]
                         /-
                           🎉 no goals
                         -/


@[simp]
theorem range_eq_empty_of_isEmpty {β} [hα : IsEmpty α] (f : α →ₛ β) : f.range = ∅ := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    β : Type u_5
    hα : IsEmpty α
    f : MeasureTheory.SimpleFunc α β
    ⊢ Eq f.range EmptyCollection.emptyCollection
  -/
  rw [← Finset.not_nonempty_iff_eq_empty]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    β : Type u_5
    hα : IsEmpty α
    f : MeasureTheory.SimpleFunc α β
    ⊢ Not f.range.Nonempty
  -/
  by_contra h
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    β : Type u_5
    hα : IsEmpty α
    f : MeasureTheory.SimpleFunc α β
    h : f.range.Nonempty
    ⊢ False
  -/
  obtain ⟨y, hy_mem⟩ := h
  /-
    case intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    β : Type u_5
    hα : IsEmpty α
    f : MeasureTheory.SimpleFunc α β
    y : β
    hy_mem : Membership.mem f.range y
    ⊢ False
  -/
  rw [SimpleFunc.mem_range, Set.mem_range] at hy_mem
  /-
    case intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    β : Type u_5
    hα : IsEmpty α
    f : MeasureTheory.SimpleFunc α β
    y : β
    hy_mem : Exists fun y_1 => Eq (f y_1) y
    ⊢ False
  -/
  obtain ⟨x, hxy⟩ := hy_mem
  /-
    case intro.intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    β : Type u_5
    hα : IsEmpty α
    f : MeasureTheory.SimpleFunc α β
    y : β
    x : α
    hxy : Eq (f x) y
    ⊢ False
  -/
  rw [isEmpty_iff] at hα
  /-
    case intro.intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    β : Type u_5
    hα : α → False
    f : MeasureTheory.SimpleFunc α β
    y : β
    x : α
    hxy : Eq (f x) y
    ⊢ False
  -/
  exact hα x
  /-
    🎉 no goals
  -/


theorem eq_zero_of_mem_range_zero [Zero β] : ∀ {y : β}, y ∈ (0 : α →ₛ β).range → y = 0 :=
  @(forall_mem_range.2 fun _ => rfl)


@[to_additive]
theorem mul_eq_map₂ [Mul β] (f g : α →ₛ β) : f * g = (pair f g).map fun p : β × β => p.1 * p.2 :=
  rfl


theorem sup_eq_map₂ [Max β] (f g : α →ₛ β) : f ⊔ g = (pair f g).map fun p : β × β => p.1 ⊔ p.2 :=
  rfl


@[to_additive]
theorem const_mul_eq_map [Mul β] (f : α →ₛ β) (b : β) : const α b * f = f.map fun a => b * a :=
  rfl


@[to_additive]
theorem map_mul [Mul β] [Mul γ] {g : β → γ} (hg : ∀ x y, g (x * y) = g x * g y) (f₁ f₂ : α →ₛ β) :
    (f₁ * f₂).map g = f₁.map g * f₂.map g :=
  ext fun _ => hg _ _


@[to_additive]
instance instSMul [SMul K β] : SMul K (α →ₛ β) :=
  ⟨fun k f => f.map (k • ·)⟩


@[to_additive (attr := simp)]
theorem coe_smul [SMul K β] (c : K) (f : α →ₛ β) : ⇑(c • f) = c • ⇑f :=
  rfl


@[to_additive (attr := simp)]
theorem smul_apply [SMul K β] (k : K) (f : α →ₛ β) (a : α) : (k • f) a = k • f a :=
  rfl


instance hasNatSMul [AddMonoid β] : SMul ℕ (α →ₛ β) := inferInstance


@[to_additive existing hasNatSMul]
instance hasNatPow [Monoid β] : Pow (α →ₛ β) ℕ :=
  ⟨fun f n => f.map (· ^ n)⟩


@[simp]
theorem coe_pow [Monoid β] (f : α →ₛ β) (n : ℕ) : ⇑(f ^ n) = (⇑f) ^ n :=
  rfl


theorem pow_apply [Monoid β] (n : ℕ) (f : α →ₛ β) (a : α) : (f ^ n) a = f a ^ n :=
  rfl


instance hasIntPow [DivInvMonoid β] : Pow (α →ₛ β) ℤ :=
  ⟨fun f n => f.map (· ^ n)⟩


@[simp]
theorem coe_zpow [DivInvMonoid β] (f : α →ₛ β) (z : ℤ) : ⇑(f ^ z) = (⇑f) ^ z :=
  rfl


theorem zpow_apply [DivInvMonoid β] (z : ℤ) (f : α →ₛ β) (a : α) : (f ^ z) a = f a ^ z :=
  rfl

-- TODO: work out how to generate these instances with `to_additive`, which gets confused by the
-- argument order swap between `coe_smul` and `coe_pow`.

instance instAddMonoid [AddMonoid β] : AddMonoid (α →ₛ β) :=
  Function.Injective.addMonoid (fun f => show α → β from f) coe_injective coe_zero coe_add
    fun _ _ => coe_smul _ _


instance instAddCommMonoid [AddCommMonoid β] : AddCommMonoid (α →ₛ β) :=
  Function.Injective.addCommMonoid (fun f => show α → β from f) coe_injective coe_zero coe_add
    fun _ _ => coe_smul _ _


instance instAddGroup [AddGroup β] : AddGroup (α →ₛ β) :=
  Function.Injective.addGroup (fun f => show α → β from f) coe_injective coe_zero coe_add coe_neg
    coe_sub (fun _ _ => coe_smul _ _) fun _ _ => coe_smul _ _


instance instAddCommGroup [AddCommGroup β] : AddCommGroup (α →ₛ β) :=
  Function.Injective.addCommGroup (fun f => show α → β from f) coe_injective coe_zero coe_add
    coe_neg coe_sub (fun _ _ => coe_smul _ _) fun _ _ => coe_smul _ _


@[to_additive existing]
instance instMonoid [Monoid β] : Monoid (α →ₛ β) :=
  Function.Injective.monoid (fun f => show α → β from f) coe_injective coe_one coe_mul coe_pow


@[to_additive existing]
instance instCommMonoid [CommMonoid β] : CommMonoid (α →ₛ β) :=
  Function.Injective.commMonoid (fun f => show α → β from f) coe_injective coe_one coe_mul coe_pow


@[to_additive existing]
instance instGroup [Group β] : Group (α →ₛ β) :=
  Function.Injective.group (fun f => show α → β from f) coe_injective coe_one coe_mul coe_inv
    coe_div coe_pow coe_zpow


@[to_additive existing]
instance instCommGroup [CommGroup β] : CommGroup (α →ₛ β) :=
  Function.Injective.commGroup (fun f => show α → β from f) coe_injective coe_one coe_mul coe_inv
    coe_div coe_pow coe_zpow


instance instModule [Semiring K] [AddCommMonoid β] [Module K β] : Module K (α →ₛ β) :=
  Function.Injective.module K ⟨⟨fun f => show α → β from f, coe_zero⟩, coe_add⟩
    coe_injective coe_smul


theorem smul_eq_map [SMul K β] (k : K) (f : α →ₛ β) : k • f = f.map (k • ·) :=
  rfl


instance instPreorder : Preorder (α →ₛ β) := Preorder.lift (⇑)


@[norm_cast] lemma coe_le_coe : ⇑f ≤ g ↔ f ≤ g := .rfl

@[simp, norm_cast] lemma coe_lt_coe : ⇑f < g ↔ f < g := .rfl


@[simp] lemma mk_le_mk {f g : α → β} {hf hg hf' hg'} : mk f hf hf' ≤ mk g hg hg' ↔ f ≤ g := Iff.rfl

@[simp] lemma mk_lt_mk {f g : α → β} {hf hg hf' hg'} : mk f hf hf' < mk g hg hg' ↔ f < g := Iff.rfl


@[gcongr] protected alias ⟨_, GCongr.mk_le_mk⟩ := mk_le_mk

@[gcongr] protected alias ⟨_, GCongr.mk_lt_mk⟩ := mk_lt_mk

@[gcongr] protected alias ⟨_, GCongr.coe_le_coe⟩ := coe_le_coe

@[gcongr] protected alias ⟨_, GCongr.coe_lt_coe⟩ := coe_lt_coe


@[gcongr]
lemma piecewise_mono (hf : ∀ a ∈ s, f₁ a ≤ f₂ a) (hg : ∀ a ∉ s, g₁ a ≤ g₂ a) :
    piecewise s hs f₁ g₁ ≤ piecewise s hs f₂ g₂ := Set.piecewise_mono hf hg


instance instPartialOrder [PartialOrder β] : PartialOrder (α →ₛ β) :=
  { SimpleFunc.instPreorder with
    le_antisymm := fun _f _g hfg hgf => ext fun a => le_antisymm (hfg a) (hgf a) }


instance instOrderBot [LE β] [OrderBot β] : OrderBot (α →ₛ β) where
  bot := const α ⊥
  bot_le _ _ := bot_le


instance instOrderTop [LE β] [OrderTop β] : OrderTop (α →ₛ β) where
  top := const α ⊤
  le_top _ _ := le_top


instance instSemilatticeInf [SemilatticeInf β] : SemilatticeInf (α →ₛ β) :=
  { SimpleFunc.instPartialOrder with
    inf := (· ⊓ ·)
    inf_le_left := fun _ _ _ => inf_le_left
    inf_le_right := fun _ _ _ => inf_le_right
    le_inf := fun _f _g _h hfh hgh a => le_inf (hfh a) (hgh a) }


instance instSemilatticeSup [SemilatticeSup β] : SemilatticeSup (α →ₛ β) :=
  { SimpleFunc.instPartialOrder with
    sup := (· ⊔ ·)
    le_sup_left := fun _ _ _ => le_sup_left
    le_sup_right := fun _ _ _ => le_sup_right
    sup_le := fun _f _g _h hfh hgh a => sup_le (hfh a) (hgh a) }


instance instLattice [Lattice β] : Lattice (α →ₛ β) :=
  { SimpleFunc.instSemilatticeSup, SimpleFunc.instSemilatticeInf with }


instance instBoundedOrder [LE β] [BoundedOrder β] : BoundedOrder (α →ₛ β) :=
  { SimpleFunc.instOrderBot, SimpleFunc.instOrderTop with }


theorem finset_sup_apply [SemilatticeSup β] [OrderBot β] {f : γ → α →ₛ β} (s : Finset γ) (a : α) :
    s.sup f a = s.sup fun c => f c a := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : MeasurableSpace α
    inst✝¹ : SemilatticeSup β
    inst✝ : OrderBot β
    f : γ → MeasureTheory.SimpleFunc α β
    s : Finset γ
    a : α
    ⊢ Eq ((s.sup f) a) (s.sup fun c => (f c) a)
  -/
  refine Finset.induction_on s rfl ?_
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : MeasurableSpace α
    inst✝¹ : SemilatticeSup β
    inst✝ : OrderBot β
    f : γ → MeasureTheory.SimpleFunc α β
    s : Finset γ
    a : α
    ⊢ ∀ ⦃a_1 : γ⦄ {s : Finset γ}, Not (Membership.mem s a_1) → Eq ((s.sup f) a) (s …
  -/
  intro a s _ ih
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : MeasurableSpace α
    inst✝¹ : SemilatticeSup β
    inst✝ : OrderBot β
    f : γ → MeasureTheory.SimpleFunc α β
    s✝ : Finset γ
    a✝¹ : α
    a : γ
    s : Finset γ
    a✝ : Not (Membership.mem s a)
    ih : Eq ((s.sup f) a✝¹) (s.sup fun c => (f c) a✝¹)
    ⊢ Eq (((Insert.insert a s).sup f) a✝¹) ((Insert.insert a s).sup fun c => (f c) …
  -/
  rw [Finset.sup_insert, Finset.sup_insert, sup_apply, ih]
  /-
    🎉 no goals
  -/


/-- Restrict a simple function `f : α →ₛ β` to a set `s`. If `s` is measurable,
then `f.restrict s a = if a ∈ s then f a else 0`, otherwise `f.restrict s = const α 0`. -/
def restrict (f : α →ₛ β) (s : Set α) : α →ₛ β :=
  if hs : MeasurableSet s then piecewise s hs f 0 else 0


theorem restrict_of_not_measurable {f : α →ₛ β} {s : Set α} (hs : ¬MeasurableSet s) :
    restrict f s = 0 :=
  dif_neg hs


@[simp]
theorem coe_restrict (f : α →ₛ β) {s : Set α} (hs : MeasurableSet s) :
    ⇑(restrict f s) = indicator s f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : Zero β
    f : MeasureTheory.SimpleFunc α β
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq (⇑(f.restrict s)) (s.indicator ⇑f)
  -/
  rw [restrict, dif_pos hs, coe_piecewise, coe_zero, piecewise_eq_indicator]
  /-
    🎉 no goals
  -/


@[simp]
                                                               /-
                                                                 α : Type u_1
                                                                 β : Type u_2
                                                                 inst✝¹ : MeasurableSpace α
                                                                 inst✝ : Zero β
                                                                 f : MeasureTheory.SimpleFunc α β
                                                                 ⊢ Eq (f.restrict Set.univ) f
                                                               -/
theorem restrict_univ (f : α →ₛ β) : restrict f univ = f := by simp [restrict]
                                                               /-
                                                                 🎉 no goals
                                                               -/


@[simp]
                                                             /-
                                                               α : Type u_1
                                                               β : Type u_2
                                                               inst✝¹ : MeasurableSpace α
                                                               inst✝ : Zero β
                                                               f : MeasureTheory.SimpleFunc α β
                                                               ⊢ Eq (f.restrict EmptyCollection.emptyCollection) 0
                                                             -/
theorem restrict_empty (f : α →ₛ β) : restrict f ∅ = 0 := by simp [restrict]
                                                             /-
                                                               🎉 no goals
                                                             -/


theorem map_restrict_of_zero [Zero γ] {g : β → γ} (hg : g 0 = 0) (f : α →ₛ β) (s : Set α) :
    (f.restrict s).map g = (f.map g).restrict s :=
  ext fun x =>
                                    /-
                                      α : Type u_1
                                      β : Type u_2
                                      γ : Type u_3
                                      inst✝² : MeasurableSpace α
                                      inst✝¹ : Zero β
                                      inst✝ : Zero γ
                                      g : β → γ
                                      hg : Eq (g 0) 0
                                      f : MeasureTheory.SimpleFunc α β
                                      s : Set α
                                      x : α
                                      hs : MeasurableSet s
                                      ⊢ Eq ((MeasureTheory.SimpleFunc.map g (f.restrict s)) x) (((MeasureTheory.Simp …
                                    -/
    if hs : MeasurableSet s then by simp [hs, Set.indicator_comp_of_zero hg]
                                    /-
                                      🎉 no goals
                                    -/
            /-
              α : Type u_1
              β : Type u_2
              γ : Type u_3
              inst✝² : MeasurableSpace α
              inst✝¹ : Zero β
              inst✝ : Zero γ
              g : β → γ
              hg : Eq (g 0) 0
              f : MeasureTheory.SimpleFunc α β
              s : Set α
              x : α
              hs : Not (MeasurableSet s)
              ⊢ Eq ((MeasureTheory.SimpleFunc.map g (f.restrict s)) x) (((MeasureTheory.Simp …
            -/
    else by simp [restrict_of_not_measurable hs, hg]
            /-
              🎉 no goals
            -/


theorem map_coe_ennreal_restrict (f : α →ₛ ℝ≥0) (s : Set α) :
    (f.restrict s).map ((↑) : ℝ≥0 → ℝ≥0∞) = (f.map (↑)).restrict s :=
  map_restrict_of_zero ENNReal.coe_zero _ _


theorem map_coe_nnreal_restrict (f : α →ₛ ℝ≥0) (s : Set α) :
    (f.restrict s).map ((↑) : ℝ≥0 → ℝ) = (f.map (↑)).restrict s :=
  map_restrict_of_zero NNReal.coe_zero _ _


theorem restrict_apply (f : α →ₛ β) {s : Set α} (hs : MeasurableSet s) (a) :
                                           /-
                                             α : Type u_1
                                             β : Type u_2
                                             inst✝¹ : MeasurableSpace α
                                             inst✝ : Zero β
                                             f : MeasureTheory.SimpleFunc α β
                                             s : Set α
                                             hs : MeasurableSet s
                                             a : α
                                             ⊢ Eq ((f.restrict s) a) (s.indicator (⇑f) a)
                                           -/
    restrict f s a = indicator s f a := by simp only [f.coe_restrict hs]
                                           /-
                                             🎉 no goals
                                           -/


theorem restrict_preimage (f : α →ₛ β) {s : Set α} (hs : MeasurableSet s) {t : Set β}
    (ht : (0 : β) ∉ t) : restrict f s ⁻¹' t = s ∩ f ⁻¹' t := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : Zero β
    f : MeasureTheory.SimpleFunc α β
    s : Set α
    hs : MeasurableSet s
    t : Set β
    ht : Not (Membership.mem t 0)
    ⊢ Eq (Set.preimage (⇑(f.restrict s)) t) (Inter.inter s (Set.preimage (⇑f) t))
  -/
  simp [hs, indicator_preimage_of_not_mem _ _ ht, inter_comm]
  /-
    🎉 no goals
  -/


theorem restrict_preimage_singleton (f : α →ₛ β) {s : Set α} (hs : MeasurableSet s) {r : β}
    (hr : r ≠ 0) : restrict f s ⁻¹' {r} = s ∩ f ⁻¹' {r} :=
  f.restrict_preimage hs hr.symm


theorem mem_restrict_range {r : β} {s : Set α} {f : α →ₛ β} (hs : MeasurableSet s) :
    r ∈ (restrict f s).range ↔ r = 0 ∧ s ≠ univ ∨ r ∈ f '' s := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : Zero β
    r : β
    s : Set α
    f : MeasureTheory.SimpleFunc α β
    hs : MeasurableSet s
    ⊢ Iff (Membership.mem (f.restrict s).range r) (Or (And (Eq r 0) (Ne s Set.univ …
  -/
  rw [← Finset.mem_coe, coe_range, coe_restrict _ hs, mem_range_indicator]
  /-
    🎉 no goals
  -/


theorem mem_image_of_mem_range_restrict {r : β} {s : Set α} {f : α →ₛ β}
    (hr : r ∈ (restrict f s).range) (h0 : r ≠ 0) : r ∈ f '' s :=
                                  /-
                                    α : Type u_1
                                    β : Type u_2
                                    inst✝¹ : MeasurableSpace α
                                    inst✝ : Zero β
                                    r : β
                                    s : Set α
                                    f : MeasureTheory.SimpleFunc α β
                                    hr : Membership.mem (f.restrict s).range r
                                    h0 : Ne r 0
                                    hs : MeasurableSet s
                                    ⊢ Membership.mem (Set.image (⇑f) s) r
                                  -/
  if hs : MeasurableSet s then by simpa [mem_restrict_range hs, h0, -mem_range] using hr
                                  /-
                                    🎉 no goals
                                  -/
  else by
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹ : MeasurableSpace α
      inst✝ : Zero β
      r : β
      s : Set α
      f : MeasureTheory.SimpleFunc α β
      hr : Membership.mem (f.restrict s).range r
      h0 : Ne r 0
      hs : Not (MeasurableSet s)
      ⊢ Membership.mem (Set.image (⇑f) s) r
    -/
    rw [restrict_of_not_measurable hs] at hr
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹ : MeasurableSpace α
      inst✝ : Zero β
      r : β
      s : Set α
      f : MeasureTheory.SimpleFunc α β
      hr : Membership.mem (MeasureTheory.SimpleFunc.range 0) r
      h0 : Ne r 0
      hs : Not (MeasurableSet s)
      ⊢ Membership.mem (Set.image (⇑f) s) r
    -/
    exact (h0 <| eq_zero_of_mem_range_zero hr).elim
    /-
      🎉 no goals
    -/


@[gcongr, mono]
theorem restrict_mono [Preorder β] (s : Set α) {f g : α →ₛ β} (H : f ≤ g) :
    f.restrict s ≤ g.restrict s :=
  if hs : MeasurableSet s then fun x => by
    /-
      α : Type u_1
      β : Type u_2
      inst✝² : MeasurableSpace α
      inst✝¹ : Zero β
      inst✝ : Preorder β
      s : Set α
      f g : MeasureTheory.SimpleFunc α β
      H : LE.le f g
      hs : MeasurableSet s
      x : α
      ⊢ LE.le ((f.restrict s) x) ((g.restrict s) x)
    -/
    simp only [coe_restrict _ hs, indicator_le_indicator (H x)]
    /-
      🎉 no goals
    -/
          /-
            α : Type u_1
            β : Type u_2
            inst✝² : MeasurableSpace α
            inst✝¹ : Zero β
            inst✝ : Preorder β
            s : Set α
            f g : MeasureTheory.SimpleFunc α β
            H : LE.le f g
            hs : Not (MeasurableSet s)
            ⊢ LE.le (f.restrict s) (g.restrict s)
          -/
  else by simp only [restrict_of_not_measurable hs, le_refl]
          /-
            🎉 no goals
          -/


/-- Fix a sequence `i : ℕ → β`. Given a function `α → β`, its `n`-th approximation
by simple functions is defined so that in case `β = ℝ≥0∞` it sends each `a` to the supremum
of the set `{i k | k ≤ n ∧ i k ≤ f a}`, see `approx_apply` and `iSup_approx_apply` for details. -/
def approx (i : ℕ → β) (f : α → β) (n : ℕ) : α →ₛ β :=
  (Finset.range n).sup fun k => restrict (const α (i k)) { a : α | i k ≤ f a }


theorem approx_apply [TopologicalSpace β] [OrderClosedTopology β] [MeasurableSpace β]
    [OpensMeasurableSpace β] {i : ℕ → β} {f : α → β} {n : ℕ} (a : α) (hf : Measurable f) :
    (approx i f n : α →ₛ β) a = (Finset.range n).sup fun k => if i k ≤ f a then i k else 0 := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁷ : MeasurableSpace α
    inst✝⁶ : SemilatticeSup β
    inst✝⁵ : OrderBot β
    inst✝⁴ : Zero β
    inst✝³ : TopologicalSpace β
    inst✝² : OrderClosedTopology β
    inst✝¹ : MeasurableSpace β
    inst✝ : OpensMeasurableSpace β
    i : Nat → β
    f : α → β
    n : Nat
    a : α
    hf : Measurable f
    ⊢ Eq ((MeasureTheory.SimpleFunc.approx i f n) a) ((Finset.range n).sup fun k = …
  -/
  dsimp only [approx]
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁷ : MeasurableSpace α
    inst✝⁶ : SemilatticeSup β
    inst✝⁵ : OrderBot β
    inst✝⁴ : Zero β
    inst✝³ : TopologicalSpace β
    inst✝² : OrderClosedTopology β
    inst✝¹ : MeasurableSpace β
    inst✝ : OpensMeasurableSpace β
    i : Nat → β
    f : α → β
    n : Nat
    a : α
    hf : Measurable f
    ⊢ Eq (((Finset.range n).sup fun k => (MeasureTheory.SimpleFunc.const α (i k)). …
  -/
  rw [finset_sup_apply]
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁷ : MeasurableSpace α
    inst✝⁶ : SemilatticeSup β
    inst✝⁵ : OrderBot β
    inst✝⁴ : Zero β
    inst✝³ : TopologicalSpace β
    inst✝² : OrderClosedTopology β
    inst✝¹ : MeasurableSpace β
    inst✝ : OpensMeasurableSpace β
    i : Nat → β
    f : α → β
    n : Nat
    a : α
    hf : Measurable f
    ⊢ Eq ((Finset.range n).sup fun c => ((MeasureTheory.SimpleFunc.const α (i c)). …
  -/
  congr
  /-
    case e_f
    α : Type u_1
    β : Type u_2
    inst✝⁷ : MeasurableSpace α
    inst✝⁶ : SemilatticeSup β
    inst✝⁵ : OrderBot β
    inst✝⁴ : Zero β
    inst✝³ : TopologicalSpace β
    inst✝² : OrderClosedTopology β
    inst✝¹ : MeasurableSpace β
    inst✝ : OpensMeasurableSpace β
    i : Nat → β
    f : α → β
    n : Nat
    a : α
    hf : Measurable f
    ⊢ Eq (fun c => ((MeasureTheory.SimpleFunc.const α (i c)).restrict (setOf fun a …
  -/
  funext k
  /-
    case e_f.h
    α : Type u_1
    β : Type u_2
    inst✝⁷ : MeasurableSpace α
    inst✝⁶ : SemilatticeSup β
    inst✝⁵ : OrderBot β
    inst✝⁴ : Zero β
    inst✝³ : TopologicalSpace β
    inst✝² : OrderClosedTopology β
    inst✝¹ : MeasurableSpace β
    inst✝ : OpensMeasurableSpace β
    i : Nat → β
    f : α → β
    n : Nat
    a : α
    hf : Measurable f
    k : Nat
    ⊢ Eq (((MeasureTheory.SimpleFunc.const α (i k)).restrict (setOf fun a => LE.le …
  -/
  rw [restrict_apply]
    /-
      case e_f.h
      α : Type u_1
      β : Type u_2
      inst✝⁷ : MeasurableSpace α
      inst✝⁶ : SemilatticeSup β
      inst✝⁵ : OrderBot β
      inst✝⁴ : Zero β
      inst✝³ : TopologicalSpace β
      inst✝² : OrderClosedTopology β
      inst✝¹ : MeasurableSpace β
      inst✝ : OpensMeasurableSpace β
      i : Nat → β
      f : α → β
      n : Nat
      a : α
      hf : Measurable f
      k : Nat
      ⊢ Eq ((setOf fun a => LE.le (i k) (f a)).indicator (⇑(MeasureTheory.SimpleFunc …
    -/
  · simp only [coe_const, mem_setOf_eq, indicator_apply, Function.const_apply]
    /-
      🎉 no goals
    -/
    /-
      case e_f.h.hs
      α : Type u_1
      β : Type u_2
      inst✝⁷ : MeasurableSpace α
      inst✝⁶ : SemilatticeSup β
      inst✝⁵ : OrderBot β
      inst✝⁴ : Zero β
      inst✝³ : TopologicalSpace β
      inst✝² : OrderClosedTopology β
      inst✝¹ : MeasurableSpace β
      inst✝ : OpensMeasurableSpace β
      i : Nat → β
      f : α → β
      n : Nat
      a : α
      hf : Measurable f
      k : Nat
      ⊢ MeasurableSet (setOf fun a => LE.le (i k) (f a))
    -/
  · exact hf measurableSet_Ici
    /-
      🎉 no goals
    -/


theorem monotone_approx (i : ℕ → β) (f : α → β) : Monotone (approx i f) := fun _ _ h =>
  Finset.sup_mono <| Finset.range_subset.2 h


theorem approx_comp [TopologicalSpace β] [OrderClosedTopology β] [MeasurableSpace β]
    [OpensMeasurableSpace β] [MeasurableSpace γ] {i : ℕ → β} {f : γ → β} {g : α → γ} {n : ℕ} (a : α)
    (hf : Measurable f) (hg : Measurable g) :
    (approx i (f ∘ g) n : α →ₛ β) a = (approx i f n : γ →ₛ β) (g a) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝⁸ : MeasurableSpace α
    inst✝⁷ : SemilatticeSup β
    inst✝⁶ : OrderBot β
    inst✝⁵ : Zero β
    inst✝⁴ : TopologicalSpace β
    inst✝³ : OrderClosedTopology β
    inst✝² : MeasurableSpace β
    inst✝¹ : OpensMeasurableSpace β
    inst✝ : MeasurableSpace γ
    i : Nat → β
    f : γ → β
    g : α → γ
    n : Nat
    a : α
    hf : Measurable f
    hg : Measurable g
    ⊢ Eq ((MeasureTheory.SimpleFunc.approx i (Function.comp f g) n) a) ((MeasureTh …
  -/
  rw [approx_apply _ hf, approx_apply _ (hf.comp hg), Function.comp_apply]
  /-
    🎉 no goals
  -/


theorem iSup_approx_apply [TopologicalSpace β] [CompleteLattice β] [OrderClosedTopology β] [Zero β]
    [MeasurableSpace β] [OpensMeasurableSpace β] (i : ℕ → β) (f : α → β) (a : α) (hf : Measurable f)
    (h_zero : (0 : β) = ⊥) : ⨆ n, (approx i f n : α →ₛ β) a = ⨆ (k) (_ : i k ≤ f a), i k := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁶ : MeasurableSpace α
    inst✝⁵ : TopologicalSpace β
    inst✝⁴ : CompleteLattice β
    inst✝³ : OrderClosedTopology β
    inst✝² : Zero β
    inst✝¹ : MeasurableSpace β
    inst✝ : OpensMeasurableSpace β
    i : Nat → β
    f : α → β
    a : α
    hf : Measurable f
    h_zero : Eq 0 Bot.bot
    ⊢ Eq (iSup fun n => (MeasureTheory.SimpleFunc.approx i f n) a) (iSup fun k =>  …
  -/
  refine le_antisymm (iSup_le fun n => ?_) (iSup_le fun k => iSup_le fun hk => ?_)
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      inst✝⁶ : MeasurableSpace α
      inst✝⁵ : TopologicalSpace β
      inst✝⁴ : CompleteLattice β
      inst✝³ : OrderClosedTopology β
      inst✝² : Zero β
      inst✝¹ : MeasurableSpace β
      inst✝ : OpensMeasurableSpace β
      i : Nat → β
      f : α → β
      a : α
      hf : Measurable f
      h_zero : Eq 0 Bot.bot
      n : Nat
      ⊢ LE.le ((MeasureTheory.SimpleFunc.approx i f n) a) (iSup fun k => iSup fun x  …
    -/
  · rw [approx_apply a hf, h_zero]
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      inst✝⁶ : MeasurableSpace α
      inst✝⁵ : TopologicalSpace β
      inst✝⁴ : CompleteLattice β
      inst✝³ : OrderClosedTopology β
      inst✝² : Zero β
      inst✝¹ : MeasurableSpace β
      inst✝ : OpensMeasurableSpace β
      i : Nat → β
      f : α → β
      a : α
      hf : Measurable f
      h_zero : Eq 0 Bot.bot
      n : Nat
      ⊢ LE.le ((Finset.range n).sup fun k => ite (LE.le (i k) (f a)) (i k) Bot.bot)  …
    -/
    refine Finset.sup_le fun k _ => ?_
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      inst✝⁶ : MeasurableSpace α
      inst✝⁵ : TopologicalSpace β
      inst✝⁴ : CompleteLattice β
      inst✝³ : OrderClosedTopology β
      inst✝² : Zero β
      inst✝¹ : MeasurableSpace β
      inst✝ : OpensMeasurableSpace β
      i : Nat → β
      f : α → β
      a : α
      hf : Measurable f
      h_zero : Eq 0 Bot.bot
      n k : Nat
      x✝ : Membership.mem (Finset.range n) k
      ⊢ LE.le (ite (LE.le (i k) (f a)) (i k) Bot.bot) (iSup fun k => iSup fun x => i …
    -/
    split_ifs with h
      /-
        case pos
        α : Type u_1
        β : Type u_2
        inst✝⁶ : MeasurableSpace α
        inst✝⁵ : TopologicalSpace β
        inst✝⁴ : CompleteLattice β
        inst✝³ : OrderClosedTopology β
        inst✝² : Zero β
        inst✝¹ : MeasurableSpace β
        inst✝ : OpensMeasurableSpace β
        i : Nat → β
        f : α → β
        a : α
        hf : Measurable f
        h_zero : Eq 0 Bot.bot
        n k : Nat
        x✝ : Membership.mem (Finset.range n) k
        h : LE.le (i k) (f a)
        ⊢ LE.le (i k) (iSup fun k => iSup fun x => i k)
      -/
    · exact le_iSup_of_le k (le_iSup (fun _ : i k ≤ f a => i k) h)
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        β : Type u_2
        inst✝⁶ : MeasurableSpace α
        inst✝⁵ : TopologicalSpace β
        inst✝⁴ : CompleteLattice β
        inst✝³ : OrderClosedTopology β
        inst✝² : Zero β
        inst✝¹ : MeasurableSpace β
        inst✝ : OpensMeasurableSpace β
        i : Nat → β
        f : α → β
        a : α
        hf : Measurable f
        h_zero : Eq 0 Bot.bot
        n k : Nat
        x✝ : Membership.mem (Finset.range n) k
        h : Not (LE.le (i k) (f a))
        ⊢ LE.le Bot.bot (iSup fun k => iSup fun x => i k)
      -/
    · exact bot_le
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      inst✝⁶ : MeasurableSpace α
      inst✝⁵ : TopologicalSpace β
      inst✝⁴ : CompleteLattice β
      inst✝³ : OrderClosedTopology β
      inst✝² : Zero β
      inst✝¹ : MeasurableSpace β
      inst✝ : OpensMeasurableSpace β
      i : Nat → β
      f : α → β
      a : α
      hf : Measurable f
      h_zero : Eq 0 Bot.bot
      k : Nat
      hk : LE.le (i k) (f a)
      ⊢ LE.le (i k) (iSup fun n => (MeasureTheory.SimpleFunc.approx i f n) a)
    -/
  · refine le_iSup_of_le (k + 1) ?_
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      inst✝⁶ : MeasurableSpace α
      inst✝⁵ : TopologicalSpace β
      inst✝⁴ : CompleteLattice β
      inst✝³ : OrderClosedTopology β
      inst✝² : Zero β
      inst✝¹ : MeasurableSpace β
      inst✝ : OpensMeasurableSpace β
      i : Nat → β
      f : α → β
      a : α
      hf : Measurable f
      h_zero : Eq 0 Bot.bot
      k : Nat
      hk : LE.le (i k) (f a)
      ⊢ LE.le (i k) ((MeasureTheory.SimpleFunc.approx i f (HAdd.hAdd k 1)) a)
    -/
    rw [approx_apply a hf]
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      inst✝⁶ : MeasurableSpace α
      inst✝⁵ : TopologicalSpace β
      inst✝⁴ : CompleteLattice β
      inst✝³ : OrderClosedTopology β
      inst✝² : Zero β
      inst✝¹ : MeasurableSpace β
      inst✝ : OpensMeasurableSpace β
      i : Nat → β
      f : α → β
      a : α
      hf : Measurable f
      h_zero : Eq 0 Bot.bot
      k : Nat
      hk : LE.le (i k) (f a)
      ⊢ LE.le (i k) ((Finset.range (HAdd.hAdd k 1)).sup fun k => ite (LE.le (i k) (f …
    -/
    have : k ∈ Finset.range (k + 1) := Finset.mem_range.2 (Nat.lt_succ_self _)
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      inst✝⁶ : MeasurableSpace α
      inst✝⁵ : TopologicalSpace β
      inst✝⁴ : CompleteLattice β
      inst✝³ : OrderClosedTopology β
      inst✝² : Zero β
      inst✝¹ : MeasurableSpace β
      inst✝ : OpensMeasurableSpace β
      i : Nat → β
      f : α → β
      a : α
      hf : Measurable f
      h_zero : Eq 0 Bot.bot
      k : Nat
      hk : LE.le (i k) (f a)
      this : Membership.mem (Finset.range (HAdd.hAdd k 1)) k
      ⊢ LE.le (i k) ((Finset.range (HAdd.hAdd k 1)).sup fun k => ite (LE.le (i k) (f …
    -/
    refine le_trans (le_of_eq ?_) (Finset.le_sup this)
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      inst✝⁶ : MeasurableSpace α
      inst✝⁵ : TopologicalSpace β
      inst✝⁴ : CompleteLattice β
      inst✝³ : OrderClosedTopology β
      inst✝² : Zero β
      inst✝¹ : MeasurableSpace β
      inst✝ : OpensMeasurableSpace β
      i : Nat → β
      f : α → β
      a : α
      hf : Measurable f
      h_zero : Eq 0 Bot.bot
      k : Nat
      hk : LE.le (i k) (f a)
      this : Membership.mem (Finset.range (HAdd.hAdd k 1)) k
      ⊢ Eq (i k) (ite (LE.le (i k) (f a)) (i k) 0)
    -/
    rw [if_pos hk]
    /-
      🎉 no goals
    -/


/-- A sequence of `ℝ≥0∞`s such that its range is the set of non-negative rational numbers. -/
def ennrealRatEmbed (n : ℕ) : ℝ≥0∞ :=
  ENNReal.ofReal ((Encodable.decode (α := ℚ) n).getD (0 : ℚ))


theorem ennrealRatEmbed_encode (q : ℚ) :
    ennrealRatEmbed (Encodable.encode q) = Real.toNNReal q := by
  /-
    q : Rat
    ⊢ Eq (MeasureTheory.SimpleFunc.ennrealRatEmbed (Encodable.encode q)) ↑(↑q).toN …
  -/
  rw [ennrealRatEmbed, Encodable.encodek]; rfl
                                           /-
                                             🎉 no goals
                                           -/


/-- Approximate a function `α → ℝ≥0∞` by a sequence of simple functions. -/
def eapprox : (α → ℝ≥0∞) → ℕ → α →ₛ ℝ≥0∞ :=
  approx ennrealRatEmbed


theorem eapprox_lt_top (f : α → ℝ≥0∞) (n : ℕ) (a : α) : eapprox f n a < ∞ := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → ENNReal
    n : Nat
    a : α
    ⊢ LT.lt ((MeasureTheory.SimpleFunc.eapprox f n) a) Top.top
  -/
  simp only [eapprox, approx, finset_sup_apply, Finset.mem_range, ENNReal.bot_eq_zero, restrict]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → ENNReal
    n : Nat
    a : α
    ⊢ LT.lt ((Finset.range n).sup fun c => (dite (MeasurableSet (setOf fun a => LE …
  -/
  rw [Finset.sup_lt_iff (α := ℝ≥0∞) WithTop.top_pos]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → ENNReal
    n : Nat
    a : α
    ⊢ ∀ (b : Nat), Membership.mem (Finset.range n) b → LT.lt ((dite (MeasurableSet …
  -/
  intro b _
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → ENNReal
    n : Nat
    a : α
    b : Nat
    a✝ : Membership.mem (Finset.range n) b
    ⊢ LT.lt ((dite (MeasurableSet (setOf fun a => LE.le (MeasureTheory.SimpleFunc. …
  -/
  split_ifs
    /-
      case pos
      α : Type u_1
      inst✝ : MeasurableSpace α
      f : α → ENNReal
      n : Nat
      a : α
      b : Nat
      a✝ : Membership.mem (Finset.range n) b
      h✝ : MeasurableSet (setOf fun a => LE.le (MeasureTheory.SimpleFunc.ennrealRatE …
      ⊢ LT.lt ((MeasureTheory.SimpleFunc.piecewise (setOf fun a => LE.le (MeasureThe …
    -/
  · simp only [coe_zero, coe_piecewise, piecewise_eq_indicator, coe_const]
    calc
      { a : α | ennrealRatEmbed b ≤ f a }.indicator (fun _ => ennrealRatEmbed b) a ≤
          ennrealRatEmbed b :=
        indicator_le_self _ _ a
      _ < ⊤ := ENNReal.coe_lt_top
    /-
      case neg
      α : Type u_1
      inst✝ : MeasurableSpace α
      f : α → ENNReal
      n : Nat
      a : α
      b : Nat
      a✝ : Membership.mem (Finset.range n) b
      h✝ : Not (MeasurableSet (setOf fun a => LE.le (MeasureTheory.SimpleFunc.ennrea …
      ⊢ LT.lt (0 a) Top.top
    -/
  · exact WithTop.top_pos
    /-
      🎉 no goals
    -/


@[mono]
theorem monotone_eapprox (f : α → ℝ≥0∞) : Monotone (eapprox f) :=
  monotone_approx _ f


@[gcongr]
lemma eapprox_mono {m n : ℕ} (hmn : m ≤ n) : eapprox f m ≤ eapprox f n := monotone_eapprox _ hmn


lemma iSup_eapprox_apply (hf : Measurable f) (a : α) : ⨆ n, (eapprox f n : α →ₛ ℝ≥0∞) a = f a := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → ENNReal
    hf : Measurable f
    a : α
    ⊢ Eq (iSup fun n => (MeasureTheory.SimpleFunc.eapprox f n) a) (f a)
  -/
  rw [eapprox, iSup_approx_apply ennrealRatEmbed f a hf rfl]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → ENNReal
    hf : Measurable f
    a : α
    ⊢ Eq (iSup fun k => iSup fun x => MeasureTheory.SimpleFunc.ennrealRatEmbed k)  …
  -/
  refine le_antisymm (iSup_le fun i => iSup_le fun hi => hi) (le_of_not_gt ?_)
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → ENNReal
    hf : Measurable f
    a : α
    ⊢ Not (GT.gt (f a) (iSup fun k => iSup fun x => MeasureTheory.SimpleFunc.ennre …
  -/
  intro h
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → ENNReal
    hf : Measurable f
    a : α
    h : GT.gt (f a) (iSup fun k => iSup fun x => MeasureTheory.SimpleFunc.ennrealR …
    ⊢ False
  -/
  rcases ENNReal.lt_iff_exists_rat_btwn.1 h with ⟨q, _, lt_q, q_lt⟩
  have :
    (Real.toNNReal q : ℝ≥0∞) ≤ ⨆ (k : ℕ) (_ : ennrealRatEmbed k ≤ f a), ennrealRatEmbed k := by
    refine le_iSup_of_le (Encodable.encode q) ?_
    rw [ennrealRatEmbed_encode q]
    exact le_iSup_of_le (le_of_lt q_lt) le_rfl
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → ENNReal
    hf : Measurable f
    a : α
    h : GT.gt (f a) (iSup fun k => iSup fun x => MeasureTheory.SimpleFunc.ennrealR …
    q : Rat
    left✝ : LE.le 0 q
    lt_q : LT.lt (iSup fun k => iSup fun x => MeasureTheory.SimpleFunc.ennrealRatE …
    q_lt : LT.lt (↑(↑q).toNNReal) (f a)
    this : LE.le (↑(↑q).toNNReal) (iSup fun k => iSup fun x => MeasureTheory.Simpl …
    ⊢ False
  -/
  exact lt_irrefl _ (lt_of_le_of_lt this lt_q)
  /-
    🎉 no goals
  -/


lemma iSup_coe_eapprox (hf : Measurable f) : ⨆ n, ⇑(eapprox f n) = f := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → ENNReal
    hf : Measurable f
    ⊢ Eq (iSup fun n => ⇑(MeasureTheory.SimpleFunc.eapprox f n)) f
  -/
  simpa [funext_iff] using iSup_eapprox_apply hf
  /-
    🎉 no goals
  -/


theorem eapprox_comp [MeasurableSpace γ] {f : γ → ℝ≥0∞} {g : α → γ} {n : ℕ} (hf : Measurable f)
    (hg : Measurable g) : (eapprox (f ∘ g) n : α → ℝ≥0∞) = (eapprox f n : γ →ₛ ℝ≥0∞) ∘ g :=
  funext fun a => approx_comp a hf hg


lemma tendsto_eapprox {f : α → ℝ≥0∞} (hf_meas : Measurable f) (a : α) :
    Tendsto (fun n ↦ eapprox f n a) atTop (𝓝 (f a)) := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → ENNReal
    hf_meas : Measurable f
    a : α
    ⊢ Filter.Tendsto (fun n => (MeasureTheory.SimpleFunc.eapprox f n) a) Filter.at …
  -/
  nth_rw 2 [← iSup_coe_eapprox hf_meas]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → ENNReal
    hf_meas : Measurable f
    a : α
    ⊢ Filter.Tendsto (fun n => (MeasureTheory.SimpleFunc.eapprox f n) a) Filter.at …
  -/
  rw [iSup_apply]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → ENNReal
    hf_meas : Measurable f
    a : α
    ⊢ Filter.Tendsto (fun n => (MeasureTheory.SimpleFunc.eapprox f n) a) Filter.at …
  -/
  exact tendsto_atTop_iSup fun _ _ hnm ↦ monotone_eapprox f hnm a
  /-
    🎉 no goals
  -/


/-- Approximate a function `α → ℝ≥0∞` by a series of simple functions taking their values
in `ℝ≥0`. -/
def eapproxDiff (f : α → ℝ≥0∞) : ℕ → α →ₛ ℝ≥0
  | 0 => (eapprox f 0).map ENNReal.toNNReal
  | n + 1 => (eapprox f (n + 1) - eapprox f n).map ENNReal.toNNReal


theorem sum_eapproxDiff (f : α → ℝ≥0∞) (n : ℕ) (a : α) :
    (∑ k ∈ Finset.range (n + 1), (eapproxDiff f k a : ℝ≥0∞)) = eapprox f n a := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → ENNReal
    n : Nat
    a : α
    ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun k => ↑((MeasureTheory.SimpleFunc. …
  -/
  induction' n with n IH
    /-
      case zero
      α : Type u_1
      inst✝ : MeasurableSpace α
      f : α → ENNReal
      a : α
      ⊢ Eq ((Finset.range (HAdd.hAdd 0 1)).sum fun k => ↑((MeasureTheory.SimpleFunc. …
    -/
  · simp only [Nat.zero_add, Finset.sum_singleton, Finset.range_one]
    /-
      case zero
      α : Type u_1
      inst✝ : MeasurableSpace α
      f : α → ENNReal
      a : α
      ⊢ Eq (↑((MeasureTheory.SimpleFunc.eapproxDiff f 0) a)) ((MeasureTheory.SimpleF …
    -/
    rfl
    /-
      🎉 no goals
    -/
  · rw [Finset.sum_range_succ, IH, eapproxDiff, coe_map, Function.comp_apply,
      coe_sub, Pi.sub_apply, ENNReal.coe_toNNReal,
      add_tsub_cancel_of_le (monotone_eapprox f (Nat.le_succ _) _)]
    /-
      case succ
      α : Type u_1
      inst✝ : MeasurableSpace α
      f : α → ENNReal
      a : α
      n : Nat
      IH : Eq ((Finset.range (HAdd.hAdd n 1)).sum fun k => ↑((MeasureTheory.SimpleFu …
      ⊢ Ne (HSub.hSub ((MeasureTheory.SimpleFunc.eapprox f (HAdd.hAdd n 1)) a) ((Mea …
    -/
    apply (lt_of_le_of_lt _ (eapprox_lt_top f (n + 1) a)).ne
    /-
      α : Type u_1
      inst✝ : MeasurableSpace α
      f : α → ENNReal
      a : α
      n : Nat
      IH : Eq ((Finset.range (HAdd.hAdd n 1)).sum fun k => ↑((MeasureTheory.SimpleFu …
      ⊢ LE.le (HSub.hSub ((MeasureTheory.SimpleFunc.eapprox f (HAdd.hAdd n 1)) a) (( …
    -/
    rw [tsub_le_iff_right]
    /-
      α : Type u_1
      inst✝ : MeasurableSpace α
      f : α → ENNReal
      a : α
      n : Nat
      IH : Eq ((Finset.range (HAdd.hAdd n 1)).sum fun k => ↑((MeasureTheory.SimpleFu …
      ⊢ LE.le ((MeasureTheory.SimpleFunc.eapprox f (HAdd.hAdd n 1)) a) (HAdd.hAdd (( …
    -/
    exact le_self_add
    /-
      🎉 no goals
    -/


theorem tsum_eapproxDiff (f : α → ℝ≥0∞) (hf : Measurable f) (a : α) :
    (∑' n, (eapproxDiff f n a : ℝ≥0∞)) = f a := by
  simp_rw [ENNReal.tsum_eq_iSup_nat' (tendsto_add_atTop_nat 1), sum_eapproxDiff,
    iSup_eapprox_apply hf a]


/-- Integral of a simple function whose codomain is `ℝ≥0∞`. -/
def lintegral {_m : MeasurableSpace α} (f : α →ₛ ℝ≥0∞) (μ : Measure α) : ℝ≥0∞ :=
  ∑ x ∈ f.range, x * μ (f ⁻¹' {x})


theorem lintegral_eq_of_subset (f : α →ₛ ℝ≥0∞) {s : Finset ℝ≥0∞}
    (hs : ∀ x, f x ≠ 0 → μ (f ⁻¹' {f x}) ≠ 0 → f x ∈ s) :
    f.lintegral μ = ∑ x ∈ s, x * μ (f ⁻¹' {x}) := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : MeasureTheory.SimpleFunc α ENNReal
    s : Finset ENNReal
    hs : ∀ (x : α), Ne (f x) 0 → Ne (μ (Set.preimage (⇑f) (Singleton.singleton (f  …
    ⊢ Eq (f.lintegral μ) (s.sum fun x => HMul.hMul x (μ (Set.preimage (⇑f) (Single …
  -/
  refine Finset.sum_bij_ne_zero (fun r _ _ => r) ?_ ?_ ?_ ?_
    /-
      case refine_1
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : MeasureTheory.SimpleFunc α ENNReal
      s : Finset ENNReal
      hs : ∀ (x : α), Ne (f x) 0 → Ne (μ (Set.preimage (⇑f) (Singleton.singleton (f  …
      ⊢ ∀ (a : ENNReal) (h₁ : Membership.mem f.range a) (h₂ : Ne (HMul.hMul a (μ (Se …
    -/
  · simpa only [forall_mem_range, mul_ne_zero_iff, and_imp]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : MeasureTheory.SimpleFunc α ENNReal
      s : Finset ENNReal
      hs : ∀ (x : α), Ne (f x) 0 → Ne (μ (Set.preimage (⇑f) (Singleton.singleton (f  …
      ⊢ ∀ (a₁ : ENNReal) (h₁₁ : Membership.mem f.range a₁) (h₁₂ : Ne (HMul.hMul a₁ ( …
    -/
  · intros
    /-
      case refine_2
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : MeasureTheory.SimpleFunc α ENNReal
      s : Finset ENNReal
      hs : ∀ (x : α), Ne (f x) 0 → Ne (μ (Set.preimage (⇑f) (Singleton.singleton (f  …
      a₁✝ : ENNReal
      h₁₁✝ : Membership.mem f.range a₁✝
      h₁₂✝ : Ne (HMul.hMul a₁✝ (μ (Set.preimage (⇑f) (Singleton.singleton a₁✝)))) 0
      a₂✝ : ENNReal
      h₂₁✝ : Membership.mem f.range a₂✝
      h₂₂✝ : Ne (HMul.hMul a₂✝ (μ (Set.preimage (⇑f) (Singleton.singleton a₂✝)))) 0
      a✝ : Eq ((fun r x x => r) a₁✝ h₁₁✝ h₁₂✝) ((fun r x x => r) a₂✝ h₂₁✝ h₂₂✝)
      ⊢ Eq a₁✝ a₂✝
    -/
    assumption
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : MeasureTheory.SimpleFunc α ENNReal
      s : Finset ENNReal
      hs : ∀ (x : α), Ne (f x) 0 → Ne (μ (Set.preimage (⇑f) (Singleton.singleton (f  …
      ⊢ ∀ (b : ENNReal), Membership.mem s b → Ne (HMul.hMul b (μ (Set.preimage (⇑f)  …
    -/
  · intro b _ hb
    /-
      case refine_3
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : MeasureTheory.SimpleFunc α ENNReal
      s : Finset ENNReal
      hs : ∀ (x : α), Ne (f x) 0 → Ne (μ (Set.preimage (⇑f) (Singleton.singleton (f  …
      b : ENNReal
      a✝ : Membership.mem s b
      hb : Ne (HMul.hMul b (μ (Set.preimage (⇑f) (Singleton.singleton b)))) 0
      ⊢ Exists fun a => Exists fun h₁ => Exists fun h₂ => Eq ((fun r x x => r) a h₁  …
    -/
    refine ⟨b, ?_, hb, rfl⟩
    /-
      case refine_3
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : MeasureTheory.SimpleFunc α ENNReal
      s : Finset ENNReal
      hs : ∀ (x : α), Ne (f x) 0 → Ne (μ (Set.preimage (⇑f) (Singleton.singleton (f  …
      b : ENNReal
      a✝ : Membership.mem s b
      hb : Ne (HMul.hMul b (μ (Set.preimage (⇑f) (Singleton.singleton b)))) 0
      ⊢ Membership.mem f.range b
    -/
    rw [mem_range, ← preimage_singleton_nonempty]
    /-
      case refine_3
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : MeasureTheory.SimpleFunc α ENNReal
      s : Finset ENNReal
      hs : ∀ (x : α), Ne (f x) 0 → Ne (μ (Set.preimage (⇑f) (Singleton.singleton (f  …
      b : ENNReal
      a✝ : Membership.mem s b
      hb : Ne (HMul.hMul b (μ (Set.preimage (⇑f) (Singleton.singleton b)))) 0
      ⊢ (Set.preimage (⇑f) (Singleton.singleton b)).Nonempty
    -/
    exact nonempty_of_measure_ne_zero (mul_ne_zero_iff.1 hb).2
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : MeasureTheory.SimpleFunc α ENNReal
      s : Finset ENNReal
      hs : ∀ (x : α), Ne (f x) 0 → Ne (μ (Set.preimage (⇑f) (Singleton.singleton (f  …
      ⊢ ∀ (a : ENNReal) (h₁ : Membership.mem f.range a) (h₂ : Ne (HMul.hMul a (μ (Se …
    -/
  · intros
    /-
      case refine_4
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : MeasureTheory.SimpleFunc α ENNReal
      s : Finset ENNReal
      hs : ∀ (x : α), Ne (f x) 0 → Ne (μ (Set.preimage (⇑f) (Singleton.singleton (f  …
      a✝ : ENNReal
      h₁✝ : Membership.mem f.range a✝
      h₂✝ : Ne (HMul.hMul a✝ (μ (Set.preimage (⇑f) (Singleton.singleton a✝)))) 0
      ⊢ Eq (HMul.hMul a✝ (μ (Set.preimage (⇑f) (Singleton.singleton a✝)))) (HMul.hMu …
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem lintegral_eq_of_subset' (f : α →ₛ ℝ≥0∞) {s : Finset ℝ≥0∞} (hs : f.range \ {0} ⊆ s) :
    f.lintegral μ = ∑ x ∈ s, x * μ (f ⁻¹' {x}) :=
  f.lintegral_eq_of_subset fun x hfx _ =>
    hs <| Finset.mem_sdiff.2 ⟨f.mem_range_self x, mt Finset.mem_singleton.1 hfx⟩


/-- Calculate the integral of `(g ∘ f)`, where `g : β → ℝ≥0∞` and `f : α →ₛ β`. -/
theorem map_lintegral (g : β → ℝ≥0∞) (f : α →ₛ β) :
    (f.map g).lintegral μ = ∑ x ∈ f.range, g x * μ (f ⁻¹' {x}) := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    g : β → ENNReal
    f : MeasureTheory.SimpleFunc α β
    ⊢ Eq ((MeasureTheory.SimpleFunc.map g f).lintegral μ) (f.range.sum fun x => HM …
  -/
  simp only [lintegral, range_map]
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    g : β → ENNReal
    f : MeasureTheory.SimpleFunc α β
    ⊢ Eq ((Finset.image g f.range).sum fun x => HMul.hMul x (μ (Set.preimage (⇑(Me …
  -/
  refine Finset.sum_image' _ fun b hb => ?_
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    g : β → ENNReal
    f : MeasureTheory.SimpleFunc α β
    b : β
    hb : Membership.mem f.range b
    ⊢ Eq (HMul.hMul (g b) (μ (Set.preimage (⇑(MeasureTheory.SimpleFunc.map g f)) ( …
  -/
  rcases mem_range.1 hb with ⟨a, rfl⟩
  /-
    case intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    g : β → ENNReal
    f : MeasureTheory.SimpleFunc α β
    a : α
    hb : Membership.mem f.range (f a)
    ⊢ Eq (HMul.hMul (g (f a)) (μ (Set.preimage (⇑(MeasureTheory.SimpleFunc.map g f …
  -/
  rw [map_preimage_singleton, ← f.sum_measure_preimage_singleton, Finset.mul_sum]
  /-
    case intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    g : β → ENNReal
    f : MeasureTheory.SimpleFunc α β
    a : α
    hb : Membership.mem f.range (f a)
    ⊢ Eq ((Finset.filter (fun b => Eq (g b) (g (f a))) f.range).sum fun i => HMul. …
  -/
  refine Finset.sum_congr ?_ ?_
    /-
      case intro.refine_1
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      g : β → ENNReal
      f : MeasureTheory.SimpleFunc α β
      a : α
      hb : Membership.mem f.range (f a)
      ⊢ Eq (Finset.filter (fun b => Eq (g b) (g (f a))) f.range) (Finset.filter (fun …
    -/
  · congr
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_2
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      g : β → ENNReal
      f : MeasureTheory.SimpleFunc α β
      a : α
      hb : Membership.mem f.range (f a)
      ⊢ ∀ (x : β), Membership.mem (Finset.filter (fun j => Eq (g j) (g (f a))) f.ran …
    -/
  · intro x
    /-
      case intro.refine_2
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      g : β → ENNReal
      f : MeasureTheory.SimpleFunc α β
      a : α
      hb : Membership.mem f.range (f a)
      x : β
      ⊢ Membership.mem (Finset.filter (fun j => Eq (g j) (g (f a))) f.range) x → Eq  …
    -/
    simp only [Finset.mem_filter]
    /-
      case intro.refine_2
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      g : β → ENNReal
      f : MeasureTheory.SimpleFunc α β
      a : α
      hb : Membership.mem f.range (f a)
      x : β
      ⊢ And (Membership.mem f.range x) (Eq (g x) (g (f a))) → Eq (HMul.hMul (g (f a) …
    -/
    rintro ⟨_, h⟩
    /-
      case intro.refine_2.intro
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      g : β → ENNReal
      f : MeasureTheory.SimpleFunc α β
      a : α
      hb : Membership.mem f.range (f a)
      x : β
      left✝ : Membership.mem f.range x
      h : Eq (g x) (g (f a))
      ⊢ Eq (HMul.hMul (g (f a)) (μ (Set.preimage (⇑f) (Singleton.singleton x)))) (HM …
    -/
    rw [h]
    /-
      🎉 no goals
    -/


theorem add_lintegral (f g : α →ₛ ℝ≥0∞) : (f + g).lintegral μ = f.lintegral μ + g.lintegral μ :=
  calc
    (f + g).lintegral μ =
        ∑ x ∈ (pair f g).range, (x.1 * μ (pair f g ⁻¹' {x}) + x.2 * μ (pair f g ⁻¹' {x})) := by
      /-
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f g : MeasureTheory.SimpleFunc α ENNReal
        ⊢ Eq ((HAdd.hAdd f g).lintegral μ) ((f.pair g).range.sum fun x => HAdd.hAdd (H …
      -/
      rw [add_eq_map₂, map_lintegral]; exact Finset.sum_congr rfl fun a _ => add_mul _ _ _
                                       /-
                                         🎉 no goals
                                       -/
    _ = (∑ x ∈ (pair f g).range, x.1 * μ (pair f g ⁻¹' {x})) +
          ∑ x ∈ (pair f g).range, x.2 * μ (pair f g ⁻¹' {x}) := by
      /-
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f g : MeasureTheory.SimpleFunc α ENNReal
        ⊢ Eq ((f.pair g).range.sum fun x => HAdd.hAdd (HMul.hMul x.1 (μ (Set.preimage  …
      -/
      rw [Finset.sum_add_distrib]
      /-
        🎉 no goals
      -/
    _ = ((pair f g).map Prod.fst).lintegral μ + ((pair f g).map Prod.snd).lintegral μ := by
      /-
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f g : MeasureTheory.SimpleFunc α ENNReal
        ⊢ Eq (HAdd.hAdd ((f.pair g).range.sum fun x => HMul.hMul x.1 (μ (Set.preimage  …
      -/
      rw [map_lintegral, map_lintegral]
      /-
        🎉 no goals
      -/
    _ = lintegral f μ + lintegral g μ := rfl


theorem const_mul_lintegral (f : α →ₛ ℝ≥0∞) (x : ℝ≥0∞) :
    (const α x * f).lintegral μ = x * f.lintegral μ :=
  calc
    (f.map fun a => x * a).lintegral μ = ∑ r ∈ f.range, x * r * μ (f ⁻¹' {r}) := map_lintegral _ _
                                                   /-
                                                     α : Type u_1
                                                     m : MeasurableSpace α
                                                     μ : MeasureTheory.Measure α
                                                     f : MeasureTheory.SimpleFunc α ENNReal
                                                     x : ENNReal
                                                     ⊢ Eq (f.range.sum fun r => HMul.hMul (HMul.hMul x r) (μ (Set.preimage (⇑f) (Si …
                                                   -/
    _ = x * ∑ r ∈ f.range, r * μ (f ⁻¹' {r}) := by simp_rw [Finset.mul_sum, mul_assoc]
                                                   /-
                                                     🎉 no goals
                                                   -/


/-- Integral of a simple function `α →ₛ ℝ≥0∞` as a bilinear map. -/
def lintegralₗ {m : MeasurableSpace α} : (α →ₛ ℝ≥0∞) →ₗ[ℝ≥0∞] Measure α →ₗ[ℝ≥0∞] ℝ≥0∞ where
  toFun f :=
    { toFun := lintegral f
                     /-
                       α : Type u_1
                       β : Type u_2
                       γ : Type u_3
                       δ : Type u_4
                       m✝ : MeasurableSpace α
                       μ ν : MeasureTheory.Measure α
                       m : MeasurableSpace α
                       f : MeasureTheory.SimpleFunc α ENNReal
                       ⊢ ∀ (x y : MeasureTheory.Measure α), Eq (f.lintegral (HAdd.hAdd x y)) (HAdd.hA …
                     -/
      map_add' := by simp [lintegral, mul_add, Finset.sum_add_distrib]
                     /-
                       🎉 no goals
                     -/
      map_smul' := fun c μ => by
        /-
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          δ : Type u_4
          m✝ : MeasurableSpace α
          μ✝ ν : MeasureTheory.Measure α
          m : MeasurableSpace α
          f : MeasureTheory.SimpleFunc α ENNReal
          c : ENNReal
          μ : MeasureTheory.Measure α
          ⊢ Eq ({ toFun := f.lintegral, map_add' := ⋯ }.toFun (HSMul.hSMul c μ)) (HSMul. …
        -/
        simp [lintegral, mul_left_comm _ c, Finset.mul_sum, Measure.smul_apply c] }
        /-
          🎉 no goals
        -/
  map_add' f g := LinearMap.ext fun _ => add_lintegral f g
  map_smul' c f := LinearMap.ext fun _ => const_mul_lintegral f c


@[simp]
theorem zero_lintegral : (0 : α →ₛ ℝ≥0∞).lintegral μ = 0 :=
  LinearMap.ext_iff.1 lintegralₗ.map_zero μ


theorem lintegral_add {ν} (f : α →ₛ ℝ≥0∞) : f.lintegral (μ + ν) = f.lintegral μ + f.lintegral ν :=
  (lintegralₗ f).map_add μ ν


theorem lintegral_smul (f : α →ₛ ℝ≥0∞) (c : ℝ≥0∞) : f.lintegral (c • μ) = c • f.lintegral μ :=
  (lintegralₗ f).map_smul c μ


@[simp]
theorem lintegral_zero [MeasurableSpace α] (f : α →ₛ ℝ≥0∞) : f.lintegral 0 = 0 :=
  (lintegralₗ f).map_zero


theorem lintegral_finset_sum {ι} (f : α →ₛ ℝ≥0∞) (μ : ι → Measure α) (s : Finset ι) :
    f.lintegral (∑ i ∈ s, μ i) = ∑ i ∈ s, f.lintegral (μ i) :=
  map_sum (lintegralₗ f) ..


theorem lintegral_sum {m : MeasurableSpace α} {ι} (f : α →ₛ ℝ≥0∞) (μ : ι → Measure α) :
    f.lintegral (Measure.sum μ) = ∑' i, f.lintegral (μ i) := by
  simp only [lintegral, Measure.sum_apply, f.measurableSet_preimage, ← Finset.tsum_subtype, ←
    ENNReal.tsum_mul_left]
  /-
    α : Type u_1
    m : MeasurableSpace α
    ι : Type u_5
    f : MeasureTheory.SimpleFunc α ENNReal
    μ : ι → MeasureTheory.Measure α
    ⊢ Eq (tsum fun x => tsum fun i => HMul.hMul (↑x) ((μ i) (Set.preimage (⇑f) (Si …
  -/
  apply ENNReal.tsum_comm
  /-
    🎉 no goals
  -/


theorem restrict_lintegral (f : α →ₛ ℝ≥0∞) {s : Set α} (hs : MeasurableSet s) :
    (restrict f s).lintegral μ = ∑ r ∈ f.range, r * μ (f ⁻¹' {r} ∩ s) :=
  calc
    (restrict f s).lintegral μ = ∑ r ∈ f.range, r * μ (restrict f s ⁻¹' {r}) :=
      lintegral_eq_of_subset _ fun x hx =>
        if hxs : x ∈ s then fun _ => by
          /-
            α : Type u_1
            m : MeasurableSpace α
            μ : MeasureTheory.Measure α
            f : MeasureTheory.SimpleFunc α ENNReal
            s : Set α
            hs : MeasurableSet s
            x : α
            hx : Ne ((f.restrict s) x) 0
            hxs : Membership.mem s x
            x✝ : Ne (μ (Set.preimage (⇑(f.restrict s)) (Singleton.singleton ((f.restrict s …
            ⊢ Membership.mem f.range ((f.restrict s) x)
          -/
          simp only [f.restrict_apply hs, indicator_of_mem hxs, mem_range_self]
          /-
            🎉 no goals
          -/
                                    /-
                                      α : Type u_1
                                      m : MeasurableSpace α
                                      μ : MeasureTheory.Measure α
                                      f : MeasureTheory.SimpleFunc α ENNReal
                                      s : Set α
                                      hs : MeasurableSet s
                                      x : α
                                      hx : Ne ((f.restrict s) x) 0
                                      hxs : Not (Membership.mem s x)
                                      ⊢ Eq ((f.restrict s) x) 0
                                    -/
        else False.elim <| hx <| by simp [*]
                                    /-
                                      🎉 no goals
                                    -/
    _ = ∑ r ∈ f.range, r * μ (f ⁻¹' {r} ∩ s) :=
      Finset.sum_congr rfl <|
        forall_mem_range.2 fun b =>
                                  /-
                                    α : Type u_1
                                    m : MeasurableSpace α
                                    μ : MeasureTheory.Measure α
                                    f : MeasureTheory.SimpleFunc α ENNReal
                                    s : Set α
                                    hs : MeasurableSet s
                                    b : α
                                    hb : Eq (f b) 0
                                    ⊢ Eq (HMul.hMul (f b) (μ (Set.preimage (⇑(f.restrict s)) (Singleton.singleton  …
                                  -/
          if hb : f b = 0 then by simp only [hb, zero_mul]
                                  /-
                                    🎉 no goals
                                  -/
                  /-
                    α : Type u_1
                    m : MeasurableSpace α
                    μ : MeasureTheory.Measure α
                    f : MeasureTheory.SimpleFunc α ENNReal
                    s : Set α
                    hs : MeasurableSet s
                    b : α
                    hb : Not (Eq (f b) 0)
                    ⊢ Eq (HMul.hMul (f b) (μ (Set.preimage (⇑(f.restrict s)) (Singleton.singleton  …
                  -/
          else by rw [restrict_preimage_singleton _ hs hb, inter_comm]
                  /-
                    🎉 no goals
                  -/


theorem lintegral_restrict {m : MeasurableSpace α} (f : α →ₛ ℝ≥0∞) (s : Set α) (μ : Measure α) :
    f.lintegral (μ.restrict s) = ∑ y ∈ f.range, y * μ (f ⁻¹' {y} ∩ s) := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    f : MeasureTheory.SimpleFunc α ENNReal
    s : Set α
    μ : MeasureTheory.Measure α
    ⊢ Eq (f.lintegral (μ.restrict s)) (f.range.sum fun y => HMul.hMul y (μ (Inter. …
  -/
  simp only [lintegral, Measure.restrict_apply, f.measurableSet_preimage]
  /-
    🎉 no goals
  -/


theorem restrict_lintegral_eq_lintegral_restrict (f : α →ₛ ℝ≥0∞) {s : Set α}
    (hs : MeasurableSet s) : (restrict f s).lintegral μ = f.lintegral (μ.restrict s) := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : MeasureTheory.SimpleFunc α ENNReal
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq ((f.restrict s).lintegral μ) (f.lintegral (μ.restrict s))
  -/
  rw [f.restrict_lintegral hs, lintegral_restrict]
  /-
    🎉 no goals
  -/


theorem lintegral_restrict_iUnion_of_directed {ι : Type*} [Countable ι]
    (f : α →ₛ ℝ≥0∞) {s : ι → Set α} (hd : Directed (· ⊆ ·) s) (μ : Measure α) :
    f.lintegral (μ.restrict (⋃ i, s i)) = ⨆ i, f.lintegral (μ.restrict (s i)) := by
  simp only [lintegral, Measure.restrict_iUnion_apply_eq_iSup hd (measurableSet_preimage ..),
    ENNReal.mul_iSup]
  /-
    α : Type u_1
    m : MeasurableSpace α
    ι : Type u_5
    inst✝ : Countable ι
    f : MeasureTheory.SimpleFunc α ENNReal
    s : ι → Set α
    hd : Directed (fun x1 x2 => HasSubset.Subset x1 x2) s
    μ : MeasureTheory.Measure α
    ⊢ Eq (f.range.sum fun x => iSup fun i => HMul.hMul x ((μ.restrict (s i)) (Set. …
  -/
  refine finsetSum_iSup fun i j ↦ (hd i j).imp fun k ⟨hik, hjk⟩ ↦ fun a ↦ ?_
  -- TODO https://github.com/leanprover-community/mathlib4/pull/14739 make `gcongr` close this goal
  /-
    α : Type u_1
    m : MeasurableSpace α
    ι : Type u_5
    inst✝ : Countable ι
    f : MeasureTheory.SimpleFunc α ENNReal
    s : ι → Set α
    hd : Directed (fun x1 x2 => HasSubset.Subset x1 x2) s
    μ : MeasureTheory.Measure α
    i j k : ι
    x✝ : And ((fun x1 x2 => HasSubset.Subset x1 x2) (s i) (s k)) ((fun x1 x2 => Ha …
    hik : (fun x1 x2 => HasSubset.Subset x1 x2) (s i) (s k)
    hjk : (fun x1 x2 => HasSubset.Subset x1 x2) (s j) (s k)
    a : ENNReal
    ⊢ And (LE.le (HMul.hMul a ((μ.restrict (s i)) (Set.preimage (⇑f) (Singleton.si …
  -/
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
  constructor <;> · gcongr; refine Measure.restrict_mono ?_ le_rfl _; assumption
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


theorem const_lintegral (c : ℝ≥0∞) : (const α c).lintegral μ = c * μ univ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    c : ENNReal
    ⊢ Eq ((MeasureTheory.SimpleFunc.const α c).lintegral μ) (HMul.hMul c (μ Set.un …
  -/
  rw [lintegral]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    c : ENNReal
    ⊢ Eq ((MeasureTheory.SimpleFunc.const α c).range.sum fun x => HMul.hMul x (μ ( …
  -/
  cases isEmpty_or_nonempty α
    /-
      case inl
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      c : ENNReal
      h✝ : IsEmpty α
      ⊢ Eq ((MeasureTheory.SimpleFunc.const α c).range.sum fun x => HMul.hMul x (μ ( …
    -/
  · simp [μ.eq_zero_of_isEmpty]
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      c : ENNReal
      h✝ : Nonempty α
      ⊢ Eq ((MeasureTheory.SimpleFunc.const α c).range.sum fun x => HMul.hMul x (μ ( …
    -/
  · simp only [range_const, coe_const, Finset.sum_singleton]
    /-
      case inr
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      c : ENNReal
      h✝ : Nonempty α
      ⊢ Eq (HMul.hMul c (μ (Set.preimage (Function.const α c) (Singleton.singleton c …
    -/
    unfold Function.const; rw [preimage_const_of_mem (mem_singleton c)]
                           /-
                             🎉 no goals
                           -/


theorem const_lintegral_restrict (c : ℝ≥0∞) (s : Set α) :
    (const α c).lintegral (μ.restrict s) = c * μ s := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    c : ENNReal
    s : Set α
    ⊢ Eq ((MeasureTheory.SimpleFunc.const α c).lintegral (μ.restrict s)) (HMul.hMu …
  -/
  rw [const_lintegral, Measure.restrict_apply MeasurableSet.univ, univ_inter]
  /-
    🎉 no goals
  -/


theorem restrict_const_lintegral (c : ℝ≥0∞) {s : Set α} (hs : MeasurableSet s) :
    ((const α c).restrict s).lintegral μ = c * μ s := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    c : ENNReal
    s : Set α
    hs : MeasurableSet s
    ⊢ Eq (((MeasureTheory.SimpleFunc.const α c).restrict s).lintegral μ) (HMul.hMu …
  -/
  rw [restrict_lintegral_eq_lintegral_restrict _ hs, const_lintegral_restrict]
  /-
    🎉 no goals
  -/


@[gcongr]
theorem lintegral_mono_fun {f g : α →ₛ ℝ≥0∞} (h : f ≤ g) : f.lintegral μ ≤ g.lintegral μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f g : MeasureTheory.SimpleFunc α ENNReal
    h : LE.le f g
    ⊢ LE.le (f.lintegral μ) (g.lintegral μ)
  -/
  refine Monotone.of_left_le_map_sup (f := (lintegral · μ)) (fun f g ↦ ?_) h
  calc
    f.lintegral μ = ((pair f g).map Prod.fst).lintegral μ := by rw [map_fst_pair]
    _ ≤ ((pair f g).map fun p ↦ p.1 ⊔ p.2).lintegral μ := by
      simp only [map_lintegral]
      gcongr
      exact le_sup_left


theorem le_sup_lintegral (f g : α →ₛ ℝ≥0∞) : f.lintegral μ ⊔ g.lintegral μ ≤ (f ⊔ g).lintegral μ :=
  Monotone.le_map_sup (fun _ _ ↦ lintegral_mono_fun) f g


@[gcongr]
theorem lintegral_mono_measure {f : α →ₛ ℝ≥0∞} (h : μ ≤ ν) : f.lintegral μ ≤ f.lintegral ν := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    f : MeasureTheory.SimpleFunc α ENNReal
    h : LE.le μ ν
    ⊢ LE.le (f.lintegral μ) (f.lintegral ν)
  -/
  simp only [lintegral]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    f : MeasureTheory.SimpleFunc α ENNReal
    h : LE.le μ ν
    ⊢ LE.le (f.range.sum fun x => HMul.hMul x (μ (Set.preimage (⇑f) (Singleton.sin …
  -/
  gcongr
  /-
    case h.bc
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    f : MeasureTheory.SimpleFunc α ENNReal
    h : LE.le μ ν
    i✝ : ENNReal
    a✝ : Membership.mem f.range i✝
    ⊢ LE.le (μ (Set.preimage (⇑f) (Singleton.singleton i✝))) (ν (Set.preimage (⇑f) …
  -/
  apply h
  /-
    🎉 no goals
  -/


/-- `SimpleFunc.lintegral` is monotone both in function and in measure. -/
@[mono, gcongr]
theorem lintegral_mono {f g : α →ₛ ℝ≥0∞} (hfg : f ≤ g) (hμν : μ ≤ ν) :
    f.lintegral μ ≤ g.lintegral ν :=
  (lintegral_mono_fun hfg).trans (lintegral_mono_measure hμν)


/-- `SimpleFunc.lintegral` depends only on the measures of `f ⁻¹' {y}`. -/
theorem lintegral_eq_of_measure_preimage [MeasurableSpace β] {f : α →ₛ ℝ≥0∞} {g : β →ₛ ℝ≥0∞}
    {ν : Measure β} (H : ∀ y, μ (f ⁻¹' {y}) = ν (g ⁻¹' {y})) : f.lintegral μ = g.lintegral ν := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasurableSpace β
    f : MeasureTheory.SimpleFunc α ENNReal
    g : MeasureTheory.SimpleFunc β ENNReal
    ν : MeasureTheory.Measure β
    H : ∀ (y : ENNReal), Eq (μ (Set.preimage (⇑f) (Singleton.singleton y))) (ν (Se …
    ⊢ Eq (f.lintegral μ) (g.lintegral ν)
  -/
  simp only [lintegral, ← H]
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasurableSpace β
    f : MeasureTheory.SimpleFunc α ENNReal
    g : MeasureTheory.SimpleFunc β ENNReal
    ν : MeasureTheory.Measure β
    H : ∀ (y : ENNReal), Eq (μ (Set.preimage (⇑f) (Singleton.singleton y))) (ν (Se …
    ⊢ Eq (f.range.sum fun x => HMul.hMul x (μ (Set.preimage (⇑f) (Singleton.single …
  -/
  apply lintegral_eq_of_subset
  /-
    case hs
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasurableSpace β
    f : MeasureTheory.SimpleFunc α ENNReal
    g : MeasureTheory.SimpleFunc β ENNReal
    ν : MeasureTheory.Measure β
    H : ∀ (y : ENNReal), Eq (μ (Set.preimage (⇑f) (Singleton.singleton y))) (ν (Se …
    ⊢ ∀ (x : α), Ne (f x) 0 → Ne (μ (Set.preimage (⇑f) (Singleton.singleton (f x)) …
  -/
  simp only [H]
  /-
    case hs
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasurableSpace β
    f : MeasureTheory.SimpleFunc α ENNReal
    g : MeasureTheory.SimpleFunc β ENNReal
    ν : MeasureTheory.Measure β
    H : ∀ (y : ENNReal), Eq (μ (Set.preimage (⇑f) (Singleton.singleton y))) (ν (Se …
    ⊢ ∀ (x : α), Ne (f x) 0 → Ne (ν (Set.preimage (⇑g) (Singleton.singleton (f x)) …
  -/
  intros
  /-
    case hs
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasurableSpace β
    f : MeasureTheory.SimpleFunc α ENNReal
    g : MeasureTheory.SimpleFunc β ENNReal
    ν : MeasureTheory.Measure β
    H : ∀ (y : ENNReal), Eq (μ (Set.preimage (⇑f) (Singleton.singleton y))) (ν (Se …
    x✝ : α
    a✝¹ : Ne (f x✝) 0
    a✝ : Ne (ν (Set.preimage (⇑g) (Singleton.singleton (f x✝)))) 0
    ⊢ Membership.mem g.range (f x✝)
  -/
  exact mem_range_of_measure_ne_zero ‹_›
  /-
    🎉 no goals
  -/


/-- If two simple functions are equal a.e., then their `lintegral`s are equal. -/
theorem lintegral_congr {f g : α →ₛ ℝ≥0∞} (h : f =ᵐ[μ] g) : f.lintegral μ = g.lintegral μ :=
  lintegral_eq_of_measure_preimage fun y =>
                                                                /-
                                                                  α : Type u_1
                                                                  m : MeasurableSpace α
                                                                  μ : MeasureTheory.Measure α
                                                                  f g : MeasureTheory.SimpleFunc α ENNReal
                                                                  h : (MeasureTheory.ae μ).EventuallyEq ⇑f ⇑g
                                                                  y : ENNReal
                                                                  x : α
                                                                  hx : Eq (f x) (g x)
                                                                  ⊢ Iff (Membership.mem (Set.preimage (⇑f) (Singleton.singleton y)) x) (Membersh …
                                                                -/
    measure_congr <| Eventually.set_eq <| h.mono fun x hx => by simp [hx]
                                                                /-
                                                                  🎉 no goals
                                                                -/


theorem lintegral_map' {β} [MeasurableSpace β] {μ' : Measure β} (f : α →ₛ ℝ≥0∞) (g : β →ₛ ℝ≥0∞)
    (m' : α → β) (eq : ∀ a, f a = g (m' a)) (h : ∀ s, MeasurableSet s → μ' s = μ (m' ⁻¹' s)) :
    f.lintegral μ = g.lintegral μ' :=
  lintegral_eq_of_measure_preimage fun y => by
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      β : Type u_5
      inst✝ : MeasurableSpace β
      μ' : MeasureTheory.Measure β
      f : MeasureTheory.SimpleFunc α ENNReal
      g : MeasureTheory.SimpleFunc β ENNReal
      m' : α → β
      eq : ∀ (a : α), Eq (f a) (g (m' a))
      h : ∀ (s : Set β), MeasurableSet s → Eq (μ' s) (μ (Set.preimage m' s))
      y : ENNReal
      ⊢ Eq (μ (Set.preimage (⇑f) (Singleton.singleton y))) (μ' (Set.preimage (⇑g) (S …
    -/
    simp only [preimage, eq]
    /-
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      β : Type u_5
      inst✝ : MeasurableSpace β
      μ' : MeasureTheory.Measure β
      f : MeasureTheory.SimpleFunc α ENNReal
      g : MeasureTheory.SimpleFunc β ENNReal
      m' : α → β
      eq : ∀ (a : α), Eq (f a) (g (m' a))
      h : ∀ (s : Set β), MeasurableSet s → Eq (μ' s) (μ (Set.preimage m' s))
      y : ENNReal
      ⊢ Eq (μ (setOf fun x => Membership.mem (Singleton.singleton y) (g (m' x)))) (μ …
    -/
    exact (h (g ⁻¹' {y}) (g.measurableSet_preimage _)).symm
    /-
      🎉 no goals
    -/


theorem lintegral_map {β} [MeasurableSpace β] (g : β →ₛ ℝ≥0∞) {f : α → β} (hf : Measurable f) :
    g.lintegral (Measure.map f μ) = (g.comp f hf).lintegral μ :=
  Eq.symm <| lintegral_map' _ _ f (fun _ => rfl) fun _s hs => Measure.map_apply hf hs


theorem support_eq [MeasurableSpace α] [Zero β] (f : α →ₛ β) :
    support f = ⋃ y ∈ {y ∈ f.range | y ≠ 0}, f ⁻¹' {y} :=
  Set.ext fun x => by
    simp only [mem_support, Set.mem_preimage, mem_filter, mem_range_self, true_and, exists_prop,
      mem_iUnion, Set.mem_range, mem_singleton_iff, exists_eq_right']


theorem measurableSet_support [MeasurableSpace α] (f : α →ₛ β) : MeasurableSet (support f) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Zero β
    inst✝ : MeasurableSpace α
    f : MeasureTheory.SimpleFunc α β
    ⊢ MeasurableSet (Function.support ⇑f)
  -/
  rw [f.support_eq]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Zero β
    inst✝ : MeasurableSpace α
    f : MeasureTheory.SimpleFunc α β
    ⊢ MeasurableSet (Set.iUnion fun y => Set.iUnion fun h => Set.preimage (⇑f) (Si …
  -/
  exact Finset.measurableSet_biUnion _ fun y _ => measurableSet_fiber _ _
  /-
    🎉 no goals
  -/


lemma measure_support_lt_top (f : α →ₛ β) (hf : ∀ y, y ≠ 0 → μ (f ⁻¹' {y}) < ∞) :
    μ (support f) < ∞ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : Zero β
    μ : MeasureTheory.Measure α
    f : MeasureTheory.SimpleFunc α β
    hf : ∀ (y : β), Ne y 0 → LT.lt (μ (Set.preimage (⇑f) (Singleton.singleton y))) …
    ⊢ LT.lt (μ (Function.support ⇑f)) Top.top
  -/
  rw [support_eq]
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : Zero β
    μ : MeasureTheory.Measure α
    f : MeasureTheory.SimpleFunc α β
    hf : ∀ (y : β), Ne y 0 → LT.lt (μ (Set.preimage (⇑f) (Singleton.singleton y))) …
    ⊢ LT.lt (μ (Set.iUnion fun y => Set.iUnion fun h => Set.preimage (⇑f) (Singlet …
  -/
  refine (measure_biUnion_finset_le _ _).trans_lt (ENNReal.sum_lt_top.mpr fun y hy => ?_)
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : Zero β
    μ : MeasureTheory.Measure α
    f : MeasureTheory.SimpleFunc α β
    hf : ∀ (y : β), Ne y 0 → LT.lt (μ (Set.preimage (⇑f) (Singleton.singleton y))) …
    y : β
    hy : Membership.mem (Finset.filter (fun y => Ne y 0) f.range) y
    ⊢ LT.lt (μ (Set.preimage (⇑f) (Singleton.singleton y))) Top.top
  -/
  rw [Finset.mem_filter] at hy
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : Zero β
    μ : MeasureTheory.Measure α
    f : MeasureTheory.SimpleFunc α β
    hf : ∀ (y : β), Ne y 0 → LT.lt (μ (Set.preimage (⇑f) (Singleton.singleton y))) …
    y : β
    hy : And (Membership.mem f.range y) (Ne y 0)
    ⊢ LT.lt (μ (Set.preimage (⇑f) (Singleton.singleton y))) Top.top
  -/
  exact hf y hy.2
  /-
    🎉 no goals
  -/


/-- A `SimpleFunc` has finite measure support if it is equal to `0` outside of a set of finite
measure. -/
protected def FinMeasSupp {_m : MeasurableSpace α} (f : α →ₛ β) (μ : Measure α) : Prop :=
  f =ᶠ[μ.cofinite] 0


theorem finMeasSupp_iff_support : f.FinMeasSupp μ ↔ μ (support f) < ∞ :=
  Iff.rfl


theorem finMeasSupp_iff : f.FinMeasSupp μ ↔ ∀ y, y ≠ 0 → μ (f ⁻¹' {y}) < ∞ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    inst✝ : Zero β
    μ : MeasureTheory.Measure α
    f : MeasureTheory.SimpleFunc α β
    ⊢ Iff (f.FinMeasSupp μ) (∀ (y : β), Ne y 0 → LT.lt (μ (Set.preimage (⇑f) (Sing …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      inst✝ : Zero β
      μ : MeasureTheory.Measure α
      f : MeasureTheory.SimpleFunc α β
      ⊢ f.FinMeasSupp μ → ∀ (y : β), Ne y 0 → LT.lt (μ (Set.preimage (⇑f) (Singleton …
    -/
  · refine fun h y hy => lt_of_le_of_lt (measure_mono ?_) h
    /-
      case mp
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      inst✝ : Zero β
      μ : MeasureTheory.Measure α
      f : MeasureTheory.SimpleFunc α β
      h : f.FinMeasSupp μ
      y : β
      hy : Ne y 0
      ⊢ HasSubset.Subset (Set.preimage (⇑f) (Singleton.singleton y)) (HasCompl.compl …
    -/
    exact fun x hx (H : f x = 0) => hy <| H ▸ Eq.symm hx
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      inst✝ : Zero β
      μ : MeasureTheory.Measure α
      f : MeasureTheory.SimpleFunc α β
      ⊢ (∀ (y : β), Ne y 0 → LT.lt (μ (Set.preimage (⇑f) (Singleton.singleton y))) T …
    -/
  · intro H
    /-
      case mpr
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      inst✝ : Zero β
      μ : MeasureTheory.Measure α
      f : MeasureTheory.SimpleFunc α β
      H : ∀ (y : β), Ne y 0 → LT.lt (μ (Set.preimage (⇑f) (Singleton.singleton y)))  …
      ⊢ f.FinMeasSupp μ
    -/
    rw [finMeasSupp_iff_support, support_eq]
    /-
      case mpr
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      inst✝ : Zero β
      μ : MeasureTheory.Measure α
      f : MeasureTheory.SimpleFunc α β
      H : ∀ (y : β), Ne y 0 → LT.lt (μ (Set.preimage (⇑f) (Singleton.singleton y)))  …
      ⊢ LT.lt (μ (Set.iUnion fun y => Set.iUnion fun h => Set.preimage (⇑f) (Singlet …
    -/
    exact measure_biUnion_lt_top (finite_toSet _) fun y hy ↦ H y (mem_filter.1 hy).2
    /-
      🎉 no goals
    -/


theorem meas_preimage_singleton_ne_zero (h : f.FinMeasSupp μ) {y : β} (hy : y ≠ 0) :
    μ (f ⁻¹' {y}) < ∞ :=
  finMeasSupp_iff.1 h y hy


protected theorem map {g : β → γ} (hf : f.FinMeasSupp μ) (hg : g 0 = 0) : (f.map g).FinMeasSupp μ :=
  flip lt_of_le_of_lt hf (measure_mono <| support_comp_subset hg f)


theorem of_map {g : β → γ} (h : (f.map g).FinMeasSupp μ) (hg : ∀ b, g b = 0 → b = 0) :
    f.FinMeasSupp μ :=
  flip lt_of_le_of_lt h <| measure_mono <| support_subset_comp @(hg) _


theorem map_iff {g : β → γ} (hg : ∀ {b}, g b = 0 ↔ b = 0) :
    (f.map g).FinMeasSupp μ ↔ f.FinMeasSupp μ :=
  ⟨fun h => h.of_map fun _ => hg.1, fun h => h.map <| hg.2 rfl⟩


protected theorem pair {g : α →ₛ γ} (hf : f.FinMeasSupp μ) (hg : g.FinMeasSupp μ) :
    (pair f g).FinMeasSupp μ :=
  calc
    μ (support <| pair f g) = μ (support f ∪ support g) := congr_arg μ <| support_prod_mk f g
    _ ≤ μ (support f) + μ (support g) := measure_union_le _ _
    _ < _ := add_lt_top.2 ⟨hf, hg⟩


protected theorem map₂ [Zero δ] (hf : f.FinMeasSupp μ) {g : α →ₛ γ} (hg : g.FinMeasSupp μ)
    {op : β → γ → δ} (H : op 0 0 = 0) : ((pair f g).map (Function.uncurry op)).FinMeasSupp μ :=
  (hf.pair hg).map H


protected theorem add {β} [AddMonoid β] {f g : α →ₛ β} (hf : f.FinMeasSupp μ)
    (hg : g.FinMeasSupp μ) : (f + g).FinMeasSupp μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_5
    inst✝ : AddMonoid β
    f g : MeasureTheory.SimpleFunc α β
    hf : f.FinMeasSupp μ
    hg : g.FinMeasSupp μ
    ⊢ (HAdd.hAdd f g).FinMeasSupp μ
  -/
  rw [add_eq_map₂]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_5
    inst✝ : AddMonoid β
    f g : MeasureTheory.SimpleFunc α β
    hf : f.FinMeasSupp μ
    hg : g.FinMeasSupp μ
    ⊢ (MeasureTheory.SimpleFunc.map (fun p => HAdd.hAdd p.1 p.2) (f.pair g)).FinMe …
  -/
  exact hf.map₂ hg (zero_add 0)
  /-
    🎉 no goals
  -/


protected theorem mul {β} [MonoidWithZero β] {f g : α →ₛ β} (hf : f.FinMeasSupp μ)
    (hg : g.FinMeasSupp μ) : (f * g).FinMeasSupp μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_5
    inst✝ : MonoidWithZero β
    f g : MeasureTheory.SimpleFunc α β
    hf : f.FinMeasSupp μ
    hg : g.FinMeasSupp μ
    ⊢ (HMul.hMul f g).FinMeasSupp μ
  -/
  rw [mul_eq_map₂]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    β : Type u_5
    inst✝ : MonoidWithZero β
    f g : MeasureTheory.SimpleFunc α β
    hf : f.FinMeasSupp μ
    hg : g.FinMeasSupp μ
    ⊢ (MeasureTheory.SimpleFunc.map (fun p => HMul.hMul p.1 p.2) (f.pair g)).FinMe …
  -/
  exact hf.map₂ hg (zero_mul 0)
  /-
    🎉 no goals
  -/


theorem lintegral_lt_top {f : α →ₛ ℝ≥0∞} (hm : f.FinMeasSupp μ) (hf : ∀ᵐ a ∂μ, f a ≠ ∞) :
    f.lintegral μ < ∞ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : MeasureTheory.SimpleFunc α ENNReal
    hm : f.FinMeasSupp μ
    hf : Filter.Eventually (fun a => Ne (f a) Top.top) (MeasureTheory.ae μ)
    ⊢ LT.lt (f.lintegral μ) Top.top
  -/
  refine sum_lt_top.2 fun a ha => ?_
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : MeasureTheory.SimpleFunc α ENNReal
    hm : f.FinMeasSupp μ
    hf : Filter.Eventually (fun a => Ne (f a) Top.top) (MeasureTheory.ae μ)
    a : ENNReal
    ha : Membership.mem f.range a
    ⊢ LT.lt (HMul.hMul a (μ (Set.preimage (⇑f) (Singleton.singleton a)))) Top.top
  -/
  rcases eq_or_ne a ∞ with (rfl | ha)
    /-
      case inl
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : MeasureTheory.SimpleFunc α ENNReal
      hm : f.FinMeasSupp μ
      hf : Filter.Eventually (fun a => Ne (f a) Top.top) (MeasureTheory.ae μ)
      ha : Membership.mem f.range Top.top
      ⊢ LT.lt (HMul.hMul Top.top (μ (Set.preimage (⇑f) (Singleton.singleton Top.top) …
    -/
  · simp only [ae_iff, Ne, Classical.not_not] at hf
    /-
      case inl
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : MeasureTheory.SimpleFunc α ENNReal
      hm : f.FinMeasSupp μ
      ha : Membership.mem f.range Top.top
      hf : Eq (μ (setOf fun a => Eq (f a) Top.top)) 0
      ⊢ LT.lt (HMul.hMul Top.top (μ (Set.preimage (⇑f) (Singleton.singleton Top.top) …
    -/
    simp [Set.preimage, hf]
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      m : MeasurableSpace α
      μ : MeasureTheory.Measure α
      f : MeasureTheory.SimpleFunc α ENNReal
      hm : f.FinMeasSupp μ
      hf : Filter.Eventually (fun a => Ne (f a) Top.top) (MeasureTheory.ae μ)
      a : ENNReal
      ha✝ : Membership.mem f.range a
      ha : Ne a Top.top
      ⊢ LT.lt (HMul.hMul a (μ (Set.preimage (⇑f) (Singleton.singleton a)))) Top.top
    -/
  · by_cases ha0 : a = 0
      /-
        case pos
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : MeasureTheory.SimpleFunc α ENNReal
        hm : f.FinMeasSupp μ
        hf : Filter.Eventually (fun a => Ne (f a) Top.top) (MeasureTheory.ae μ)
        a : ENNReal
        ha✝ : Membership.mem f.range a
        ha : Ne a Top.top
        ha0 : Eq a 0
        ⊢ LT.lt (HMul.hMul a (μ (Set.preimage (⇑f) (Singleton.singleton a)))) Top.top
      -/
    · subst a
      /-
        case pos
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : MeasureTheory.SimpleFunc α ENNReal
        hm : f.FinMeasSupp μ
        hf : Filter.Eventually (fun a => Ne (f a) Top.top) (MeasureTheory.ae μ)
        ha✝ : Membership.mem f.range 0
        ha : Ne 0 Top.top
        ⊢ LT.lt (HMul.hMul 0 (μ (Set.preimage (⇑f) (Singleton.singleton 0)))) Top.top
      -/
      simp
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        m : MeasurableSpace α
        μ : MeasureTheory.Measure α
        f : MeasureTheory.SimpleFunc α ENNReal
        hm : f.FinMeasSupp μ
        hf : Filter.Eventually (fun a => Ne (f a) Top.top) (MeasureTheory.ae μ)
        a : ENNReal
        ha✝ : Membership.mem f.range a
        ha : Ne a Top.top
        ha0 : Not (Eq a 0)
        ⊢ LT.lt (HMul.hMul a (μ (Set.preimage (⇑f) (Singleton.singleton a)))) Top.top
      -/
    · exact mul_lt_top ha.lt_top (finMeasSupp_iff.1 hm _ ha0)
      /-
        🎉 no goals
      -/


theorem of_lintegral_ne_top {f : α →ₛ ℝ≥0∞} (h : f.lintegral μ ≠ ∞) : f.FinMeasSupp μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : MeasureTheory.SimpleFunc α ENNReal
    h : Ne (f.lintegral μ) Top.top
    ⊢ f.FinMeasSupp μ
  -/
  refine finMeasSupp_iff.2 fun b hb => ?_
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : MeasureTheory.SimpleFunc α ENNReal
    h : Ne (f.lintegral μ) Top.top
    b : ENNReal
    hb : Ne b 0
    ⊢ LT.lt (μ (Set.preimage (⇑f) (Singleton.singleton b))) Top.top
  -/
  rw [f.lintegral_eq_of_subset' (Finset.subset_insert b _)] at h
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : MeasureTheory.SimpleFunc α ENNReal
    b : ENNReal
    h : Ne ((Insert.insert b (SDiff.sdiff f.range (Singleton.singleton 0))).sum fu …
    hb : Ne b 0
    ⊢ LT.lt (μ (Set.preimage (⇑f) (Singleton.singleton b))) Top.top
  -/
  refine ENNReal.lt_top_of_mul_ne_top_right ?_ hb
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : MeasureTheory.SimpleFunc α ENNReal
    b : ENNReal
    h : Ne ((Insert.insert b (SDiff.sdiff f.range (Singleton.singleton 0))).sum fu …
    hb : Ne b 0
    ⊢ Ne (HMul.hMul b (μ (Set.preimage (⇑f) (Singleton.singleton b)))) Top.top
  -/
  exact (lt_top_of_sum_ne_top h (Finset.mem_insert_self _ _)).ne
  /-
    🎉 no goals
  -/


theorem iff_lintegral_lt_top {f : α →ₛ ℝ≥0∞} (hf : ∀ᵐ a ∂μ, f a ≠ ∞) :
    f.FinMeasSupp μ ↔ f.lintegral μ < ∞ :=
  ⟨fun h => h.lintegral_lt_top hf, fun h => of_lintegral_ne_top h.ne⟩


lemma measure_support_lt_top_of_lintegral_ne_top {f : α →ₛ ℝ≥0∞} (hf : f.lintegral μ ≠ ∞) :
    μ (support f) < ∞ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : MeasureTheory.SimpleFunc α ENNReal
    hf : Ne (f.lintegral μ) Top.top
    ⊢ LT.lt (μ (Function.support ⇑f)) Top.top
  -/
  refine measure_support_lt_top f ?_
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : MeasureTheory.SimpleFunc α ENNReal
    hf : Ne (f.lintegral μ) Top.top
    ⊢ ∀ (y : ENNReal), Ne y 0 → LT.lt (μ (Set.preimage (⇑f) (Singleton.singleton y …
  -/
  rw [← finMeasSupp_iff]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ : MeasureTheory.Measure α
    f : MeasureTheory.SimpleFunc α ENNReal
    hf : Ne (f.lintegral μ) Top.top
    ⊢ f.FinMeasSupp μ
  -/
  exact FinMeasSupp.of_lintegral_ne_top hf
  /-
    🎉 no goals
  -/


/-- To prove something for an arbitrary simple function, it suffices to show
that the property holds for (multiples of) characteristic functions and is closed under
addition (of functions with disjoint support).

It is possible to make the hypotheses in `h_add` a bit stronger, and such conditions can be added
once we need them (for example it is only necessary to consider the case where `g` is a multiple
of a characteristic function, and that this multiple doesn't appear in the image of `f`) -/
@[elab_as_elim]
protected theorem induction {α γ} [MeasurableSpace α] [AddMonoid γ] {P : SimpleFunc α γ → Prop}
    (h_ind :
      ∀ (c) {s} (hs : MeasurableSet s),
        P (SimpleFunc.piecewise s hs (SimpleFunc.const _ c) (SimpleFunc.const _ 0)))
    (h_add : ∀ ⦃f g : SimpleFunc α γ⦄, Disjoint (support f) (support g) → P f → P g → P (f + g))
    (f : SimpleFunc α γ) : P f := by
  /-
    α : Type u_5
    γ : Type u_6
    inst✝¹ : MeasurableSpace α
    inst✝ : AddMonoid γ
    P : MeasureTheory.SimpleFunc α γ → Prop
    h_ind : ∀ (c : γ) {s : Set α} (hs : MeasurableSet s), P (MeasureTheory.SimpleF …
    h_add : ∀ ⦃f g : MeasureTheory.SimpleFunc α γ⦄, Disjoint (Function.support ⇑f) …
    f : MeasureTheory.SimpleFunc α γ
    ⊢ P f
  -/
  generalize h : f.range \ {0} = s
  /-
    α : Type u_5
    γ : Type u_6
    inst✝¹ : MeasurableSpace α
    inst✝ : AddMonoid γ
    P : MeasureTheory.SimpleFunc α γ → Prop
    h_ind : ∀ (c : γ) {s : Set α} (hs : MeasurableSet s), P (MeasureTheory.SimpleF …
    h_add : ∀ ⦃f g : MeasureTheory.SimpleFunc α γ⦄, Disjoint (Function.support ⇑f) …
    f : MeasureTheory.SimpleFunc α γ
    s : Finset γ
    h : Eq (SDiff.sdiff f.range (Singleton.singleton 0)) s
    ⊢ P f
  -/
  rw [← Finset.coe_inj, Finset.coe_sdiff, Finset.coe_singleton, SimpleFunc.coe_range] at h
  induction s using Finset.induction generalizing f with
  | empty =>
    rw [Finset.coe_empty, diff_eq_empty, range_subset_singleton] at h
    convert h_ind 0 MeasurableSet.univ
    ext x
    simp [h]
  | @insert x s hxs ih =>
    have mx := f.measurableSet_preimage {x}
    let g := SimpleFunc.piecewise (f ⁻¹' {x}) mx 0 f
    have Pg : P g := by
      apply ih
      simp only [g, SimpleFunc.coe_piecewise, range_piecewise]
      rw [image_compl_preimage, union_diff_distrib, diff_diff_comm, h, Finset.coe_insert,
        insert_diff_self_of_not_mem, diff_eq_empty.mpr, Set.empty_union]
      · rw [Set.image_subset_iff]
        convert Set.subset_univ _
        exact preimage_const_of_mem (mem_singleton _)
      · rwa [Finset.mem_coe]
    convert h_add _ Pg (h_ind x mx)
    · ext1 y
      by_cases hy : y ∈ f ⁻¹' {x}
      · simpa [g, piecewise_eq_of_mem _ _ _ hy, -piecewise_eq_indicator]
      · simp [g, piecewise_eq_of_not_mem _ _ _ hy, -piecewise_eq_indicator]
    rw [disjoint_iff_inf_le]
    rintro y
    by_cases hy : y ∈ f ⁻¹' {x}
    · simp [g, piecewise_eq_of_mem _ _ _ hy, -piecewise_eq_indicator]
    · simp [piecewise_eq_of_not_mem _ _ _ hy, -piecewise_eq_indicator]


/-- In a topological vector space, the addition of a measurable function and a simple function is
measurable. -/
theorem _root_.Measurable.add_simpleFunc
    {E : Type*} {_ : MeasurableSpace α} [MeasurableSpace E] [AddGroup E] [MeasurableAdd E]
    {g : α → E} (hg : Measurable g) (f : SimpleFunc α E) :
    Measurable (g + (f : α → E)) := by
  classical
  induction' f using SimpleFunc.induction with c s hs f f' hff' hf hf'
  · simp only [SimpleFunc.const_zero, SimpleFunc.coe_piecewise, SimpleFunc.coe_const,
      SimpleFunc.coe_zero]
    change Measurable (g + s.piecewise (Function.const α c) (0 : α → E))
    rw [← s.piecewise_same g, ← piecewise_add]
    exact Measurable.piecewise hs (hg.add_const _) (hg.add_const _)
  · have : (g + ↑(f + f'))
        = (Function.support f).piecewise (g + (f : α → E)) (g + f') := by
      ext x
      by_cases hx : x ∈ Function.support f
      · simpa only [SimpleFunc.coe_add, Pi.add_apply, Function.mem_support, ne_eq, not_not,
          Set.piecewise_eq_of_mem _ _ _ hx, _root_.add_right_inj, add_right_eq_self]
          using Set.disjoint_left.1 hff' hx
      · simpa only [SimpleFunc.coe_add, Pi.add_apply, Function.mem_support, ne_eq, not_not,
          Set.piecewise_eq_of_not_mem _ _ _ hx, _root_.add_right_inj, add_left_eq_self] using hx
    rw [this]
    exact Measurable.piecewise f.measurableSet_support hf hf'


/-- In a topological vector space, the addition of a simple function and a measurable function is
measurable. -/
theorem _root_.Measurable.simpleFunc_add
    {E : Type*} {_ : MeasurableSpace α} [MeasurableSpace E] [AddGroup E] [MeasurableAdd E]
    {g : α → E} (hg : Measurable g) (f : SimpleFunc α E) :
    Measurable ((f : α → E) + g) := by
  classical
  induction' f using SimpleFunc.induction with c s hs f f' hff' hf hf'
  · simp only [SimpleFunc.const_zero, SimpleFunc.coe_piecewise, SimpleFunc.coe_const,
      SimpleFunc.coe_zero]
    change Measurable (s.piecewise (Function.const α c) (0 : α → E) + g)
    rw [← s.piecewise_same g, ← piecewise_add]
    exact Measurable.piecewise hs (hg.const_add _) (hg.const_add _)
  · have : (↑(f + f') + g)
        = (Function.support f).piecewise ((f : α → E) + g) (f' + g) := by
      ext x
      by_cases hx : x ∈ Function.support f
      · simpa only [coe_add, Pi.add_apply, Function.mem_support, ne_eq, not_not,
          Set.piecewise_eq_of_mem _ _ _ hx, _root_.add_left_inj, add_right_eq_self]
          using Set.disjoint_left.1 hff' hx
      · simpa only [SimpleFunc.coe_add, Pi.add_apply, Function.mem_support, ne_eq, not_not,
          Set.piecewise_eq_of_not_mem _ _ _ hx, _root_.add_left_inj, add_left_eq_self] using hx
    rw [this]
    exact Measurable.piecewise f.measurableSet_support hf hf'


/-- To prove something for an arbitrary measurable function into `ℝ≥0∞`, it suffices to show
that the property holds for (multiples of) characteristic functions and is closed under addition
and supremum of increasing sequences of functions.

It is possible to make the hypotheses in the induction steps a bit stronger, and such conditions
can be added once we need them (for example in `h_add` it is only necessary to consider the sum of
a simple function with a multiple of a characteristic function and that the intersection
of their images is a subset of `{0}`. -/
@[elab_as_elim]
theorem Measurable.ennreal_induction {P : (α → ℝ≥0∞) → Prop}
    (h_ind : ∀ (c : ℝ≥0∞) ⦃s⦄, MeasurableSet s → P (Set.indicator s fun _ => c))
    (h_add :
      ∀ ⦃f g : α → ℝ≥0∞⦄,
        Disjoint (support f) (support g) → Measurable f → Measurable g → P f → P g → P (f + g))
    (h_iSup :
      ∀ ⦃f : ℕ → α → ℝ≥0∞⦄, (∀ n, Measurable (f n)) → Monotone f → (∀ n, P (f n)) →
        P fun x => ⨆ n, f n x)
    ⦃f : α → ℝ≥0∞⦄ (hf : Measurable f) : P f := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    P : (α → ENNReal) → Prop
    h_ind : ∀ (c : ENNReal) ⦃s : Set α⦄, MeasurableSet s → P (s.indicator fun x => …
    h_add : ∀ ⦃f g : α → ENNReal⦄, Disjoint (Function.support f) (Function.support …
    h_iSup : ∀ ⦃f : Nat → α → ENNReal⦄, (∀ (n : Nat), Measurable (f n)) → Monotone …
    f : α → ENNReal
    hf : Measurable f
    ⊢ P f
  -/
  convert h_iSup (fun n => (eapprox f n).measurable) (monotone_eapprox f) _ using 2
    /-
      case h.e'_1.h
      α : Type u_1
      mα : MeasurableSpace α
      P : (α → ENNReal) → Prop
      h_ind : ∀ (c : ENNReal) ⦃s : Set α⦄, MeasurableSet s → P (s.indicator fun x => …
      h_add : ∀ ⦃f g : α → ENNReal⦄, Disjoint (Function.support f) (Function.support …
      h_iSup : ∀ ⦃f : Nat → α → ENNReal⦄, (∀ (n : Nat), Measurable (f n)) → Monotone …
      f : α → ENNReal
      hf : Measurable f
      x✝ : α
      ⊢ Eq (f x✝) (iSup fun n => (MeasureTheory.SimpleFunc.eapprox f n) x✝)
    -/
  · rw [iSup_eapprox_apply hf]
    /-
      🎉 no goals
    -/
  · exact fun n =>
      SimpleFunc.induction (fun c s hs => h_ind c hs)
        (fun f g hfg hf hg => h_add hfg f.measurable g.measurable hf hg) (eapprox f n)


/-- To prove something for an arbitrary measurable function into `ℝ≥0∞`, it suffices to show
that the property holds for (multiples of) characteristic functions with finite mass according to
some sigma-finite measure and is closed under addition and supremum of increasing sequences of
functions.

It is possible to make the hypotheses in the induction steps a bit stronger, and such conditions
can be added once we need them (for example in `h_add` it is only necessary to consider the sum of
a simple function with a multiple of a characteristic function and that the intersection
of their images is a subset of `{0}`. -/
@[elab_as_elim]
lemma Measurable.ennreal_sigmaFinite_induction [SigmaFinite μ] {P : (α → ℝ≥0∞) → Prop}
    (h_ind : ∀ (c : ℝ≥0∞) ⦃s⦄, MeasurableSet s → μ s < ∞ → P (Set.indicator s fun _ ↦ c))
    (h_add :
      ∀ ⦃f g : α → ℝ≥0∞⦄,
        Disjoint (support f) (support g) → Measurable f → Measurable g → P f → P g → P (f + g))
    (h_iSup :
      ∀ ⦃f : ℕ → α → ℝ≥0∞⦄, (∀ n, Measurable (f n)) → Monotone f → (∀ n, P (f n)) →
        P fun x => ⨆ n, f n x)
    ⦃f : α → ℝ≥0∞⦄ (hf : Measurable f) : P f := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    P : (α → ENNReal) → Prop
    h_ind : ∀ (c : ENNReal) ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → P …
    h_add : ∀ ⦃f g : α → ENNReal⦄, Disjoint (Function.support f) (Function.support …
    h_iSup : ∀ ⦃f : Nat → α → ENNReal⦄, (∀ (n : Nat), Measurable (f n)) → Monotone …
    f : α → ENNReal
    hf : Measurable f
    ⊢ P f
  -/
  refine Measurable.ennreal_induction (fun c s hs ↦ ?_) h_add h_iSup hf
  convert h_iSup (f := fun n ↦ (s ∩ spanningSets μ n).indicator fun _ ↦ c)
    (fun n ↦ measurable_const.indicator (hs.inter (measurableSet_spanningSets ..)))
    (fun m n hmn a ↦ Set.indicator_le_indicator_of_subset (by gcongr) (by simp) _)
    (fun n ↦ h_ind _ (hs.inter (measurableSet_spanningSets ..))
      (measure_inter_lt_top_of_right_ne_top (measure_spanningSets_lt_top ..).ne)) with a
  /-
    case h.e'_1.h
    α : Type u_1
    mα : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    P : (α → ENNReal) → Prop
    h_ind : ∀ (c : ENNReal) ⦃s : Set α⦄, MeasurableSet s → LT.lt (μ s) Top.top → P …
    h_add : ∀ ⦃f g : α → ENNReal⦄, Disjoint (Function.support f) (Function.support …
    h_iSup : ∀ ⦃f : Nat → α → ENNReal⦄, (∀ (n : Nat), Measurable (f n)) → Monotone …
    f : α → ENNReal
    hf : Measurable f
    c : ENNReal
    s : Set α
    hs : MeasurableSet s
    a : α
    ⊢ Eq (s.indicator (fun x => c) a) (iSup fun n => (Inter.inter s (MeasureTheory …
  -/
  simp [← Set.indicator_iUnion_apply (M := ℝ≥0∞) rfl, ← Set.inter_iUnion]
  /-
    🎉 no goals
  -/

