/-- A completion of `α` is the data of a complete separated uniform space (from the same universe)
and a map from `α` with dense range and inducing the original uniform structure on `α`. -/
structure AbstractCompletion (α : Type u) [UniformSpace α] where
  /-- The underlying space of the completion. -/
  space : Type u
  /-- A map from a space to its completion. -/
  coe : α → space
  /-- The completion carries a uniform structure. -/
  uniformStruct : UniformSpace space
  /-- The completion is complete. -/
  complete : CompleteSpace space
  /-- The completion is a T₀ space. -/
  separation : T0Space space
  /-- The map into the completion is uniform-inducing. -/
  isUniformInducing : IsUniformInducing coe
  /-- The map into the completion has dense range. -/
  dense : DenseRange coe


local notation "hatα" => pkg.space


local notation "ι" => pkg.coe


@[deprecated (since := "2024-10-08")] alias uniformInducing := isUniformInducing


/-- If `α` is complete, then it is an abstract completion of itself. -/
def ofComplete [T0Space α] [CompleteSpace α] : AbstractCompletion α :=
  mk α id inferInstance inferInstance inferInstance .id denseRange_id


theorem closure_range : closure (range ι) = univ :=
  pkg.dense.closure_range


theorem isDenseInducing : IsDenseInducing ι :=
  ⟨pkg.isUniformInducing.isInducing, pkg.dense⟩


theorem uniformContinuous_coe : UniformContinuous ι :=
  IsUniformInducing.uniformContinuous pkg.isUniformInducing


theorem continuous_coe : Continuous ι :=
  pkg.uniformContinuous_coe.continuous


@[elab_as_elim]
theorem induction_on {p : hatα → Prop} (a : hatα) (hp : IsClosed { a | p a }) (ih : ∀ a, p (ι a)) :
    p a :=
  isClosed_property pkg.dense hp ih a


protected theorem funext [TopologicalSpace β] [T2Space β] {f g : hatα → β} (hf : Continuous f)
    (hg : Continuous g) (h : ∀ a, f (ι a) = g (ι a)) : f = g :=
  funext fun a => pkg.induction_on a (isClosed_eq hf hg) h


/-- Extension of maps to completions -/
protected def extend (f : α → β) : hatα → β :=
  if UniformContinuous f then pkg.isDenseInducing.extend f else fun x => f (pkg.dense.some x)


theorem extend_def (hf : UniformContinuous f) : pkg.extend f = pkg.isDenseInducing.extend f :=
  if_pos hf


theorem extend_coe [T2Space β] (hf : UniformContinuous f) (a : α) : (pkg.extend f) (ι a) = f a := by
  /-
    α : Type u_1
    inst✝² : UniformSpace α
    pkg : AbstractCompletion α
    β : Type u_2
    inst✝¹ : UniformSpace β
    f : α → β
    inst✝ : T2Space β
    hf : UniformContinuous f
    a : α
    ⊢ Eq (pkg.extend f (pkg.coe a)) (f a)
  -/
  rw [pkg.extend_def hf]
  /-
    α : Type u_1
    inst✝² : UniformSpace α
    pkg : AbstractCompletion α
    β : Type u_2
    inst✝¹ : UniformSpace β
    f : α → β
    inst✝ : T2Space β
    hf : UniformContinuous f
    a : α
    ⊢ Eq (⋯.extend f (pkg.coe a)) (f a)
  -/
  exact pkg.isDenseInducing.extend_eq hf.continuous a
  /-
    🎉 no goals
  -/


theorem uniformContinuous_extend : UniformContinuous (pkg.extend f) := by
  /-
    α : Type u_1
    inst✝² : UniformSpace α
    pkg : AbstractCompletion α
    β : Type u_2
    inst✝¹ : UniformSpace β
    f : α → β
    inst✝ : CompleteSpace β
    ⊢ UniformContinuous (pkg.extend f)
  -/
  by_cases hf : UniformContinuous f
    /-
      case pos
      α : Type u_1
      inst✝² : UniformSpace α
      pkg : AbstractCompletion α
      β : Type u_2
      inst✝¹ : UniformSpace β
      f : α → β
      inst✝ : CompleteSpace β
      hf : UniformContinuous f
      ⊢ UniformContinuous (pkg.extend f)
    -/
  · rw [pkg.extend_def hf]
    /-
      case pos
      α : Type u_1
      inst✝² : UniformSpace α
      pkg : AbstractCompletion α
      β : Type u_2
      inst✝¹ : UniformSpace β
      f : α → β
      inst✝ : CompleteSpace β
      hf : UniformContinuous f
      ⊢ UniformContinuous (⋯.extend f)
    -/
    exact uniformContinuous_uniformly_extend pkg.isUniformInducing pkg.dense hf
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝² : UniformSpace α
      pkg : AbstractCompletion α
      β : Type u_2
      inst✝¹ : UniformSpace β
      f : α → β
      inst✝ : CompleteSpace β
      hf : Not (UniformContinuous f)
      ⊢ UniformContinuous (pkg.extend f)
    -/
  · change UniformContinuous (ite _ _ _)
    /-
      case neg
      α : Type u_1
      inst✝² : UniformSpace α
      pkg : AbstractCompletion α
      β : Type u_2
      inst✝¹ : UniformSpace β
      f : α → β
      inst✝ : CompleteSpace β
      hf : Not (UniformContinuous f)
      ⊢ UniformContinuous (ite (UniformContinuous f) (⋯.extend f) fun x => f (⋯.some …
    -/
    rw [if_neg hf]
    /-
      case neg
      α : Type u_1
      inst✝² : UniformSpace α
      pkg : AbstractCompletion α
      β : Type u_2
      inst✝¹ : UniformSpace β
      f : α → β
      inst✝ : CompleteSpace β
      hf : Not (UniformContinuous f)
      ⊢ UniformContinuous fun x => f (⋯.some x)
    -/
    exact uniformContinuous_of_const fun a b => by congr 1
    /-
      🎉 no goals
    -/


theorem continuous_extend : Continuous (pkg.extend f) :=
  pkg.uniformContinuous_extend.continuous


theorem extend_unique (hf : UniformContinuous f) {g : hatα → β} (hg : UniformContinuous g)
    (h : ∀ a : α, f a = g (ι a)) : pkg.extend f = g := by
  /-
    α : Type u_1
    inst✝³ : UniformSpace α
    pkg : AbstractCompletion α
    β : Type u_2
    inst✝² : UniformSpace β
    f : α → β
    inst✝¹ : CompleteSpace β
    inst✝ : T0Space β
    hf : UniformContinuous f
    g : pkg.space → β
    hg : UniformContinuous g
    h : ∀ (a : α), Eq (f a) (g (pkg.coe a))
    ⊢ Eq (pkg.extend f) g
  -/
  apply pkg.funext pkg.continuous_extend hg.continuous
  /-
    α : Type u_1
    inst✝³ : UniformSpace α
    pkg : AbstractCompletion α
    β : Type u_2
    inst✝² : UniformSpace β
    f : α → β
    inst✝¹ : CompleteSpace β
    inst✝ : T0Space β
    hf : UniformContinuous f
    g : pkg.space → β
    hg : UniformContinuous g
    h : ∀ (a : α), Eq (f a) (g (pkg.coe a))
    ⊢ ∀ (a : α), Eq (pkg.extend f (pkg.coe a)) (g (pkg.coe a))
  -/
  simpa only [pkg.extend_coe hf] using h
  /-
    🎉 no goals
  -/


@[simp]
theorem extend_comp_coe {f : hatα → β} (hf : UniformContinuous f) : pkg.extend (f ∘ ι) = f :=
  funext fun x =>
    pkg.induction_on x (isClosed_eq pkg.continuous_extend hf.continuous) fun y =>
      pkg.extend_coe (hf.comp <| pkg.uniformContinuous_coe) y


local notation "hatβ" => pkg'.space


local notation "ι'" => pkg'.coe


/-- Lifting maps to completions -/
protected def map (f : α → β) : hatα → hatβ :=
  pkg.extend (ι' ∘ f)


local notation "map" => pkg.map pkg'


theorem uniformContinuous_map : UniformContinuous (map f) :=
  pkg.uniformContinuous_extend


@[continuity]
theorem continuous_map : Continuous (map f) :=
  pkg.continuous_extend


@[simp]
theorem map_coe (hf : UniformContinuous f) (a : α) : map f (ι a) = ι' (f a) :=
  pkg.extend_coe (pkg'.uniformContinuous_coe.comp hf) a


theorem map_unique {f : α → β} {g : hatα → hatβ} (hg : UniformContinuous g)
    (h : ∀ a, ι' (f a) = g (ι a)) : map f = g :=
  pkg.funext (pkg.continuous_map _ _) hg.continuous <| by
    /-
      α : Type u_1
      inst✝¹ : UniformSpace α
      pkg : AbstractCompletion α
      β : Type u_2
      inst✝ : UniformSpace β
      pkg' : AbstractCompletion β
      f : α → β
      g : pkg.space → pkg'.space
      hg : UniformContinuous g
      h : ∀ (a : α), Eq (pkg'.coe (f a)) (g (pkg.coe a))
      ⊢ ∀ (a : α), Eq (pkg.map pkg' f (pkg.coe a)) (g (pkg.coe a))
    -/
    intro a
    /-
      α : Type u_1
      inst✝¹ : UniformSpace α
      pkg : AbstractCompletion α
      β : Type u_2
      inst✝ : UniformSpace β
      pkg' : AbstractCompletion β
      f : α → β
      g : pkg.space → pkg'.space
      hg : UniformContinuous g
      h : ∀ (a : α), Eq (pkg'.coe (f a)) (g (pkg.coe a))
      a : α
      ⊢ Eq (pkg.map pkg' f (pkg.coe a)) (g (pkg.coe a))
    -/
    change pkg.extend (ι' ∘ f) _ = _
    /-
      α : Type u_1
      inst✝¹ : UniformSpace α
      pkg : AbstractCompletion α
      β : Type u_2
      inst✝ : UniformSpace β
      pkg' : AbstractCompletion β
      f : α → β
      g : pkg.space → pkg'.space
      hg : UniformContinuous g
      h : ∀ (a : α), Eq (pkg'.coe (f a)) (g (pkg.coe a))
      a : α
      ⊢ Eq (pkg.extend (Function.comp pkg'.coe f) (pkg.coe a)) (g (pkg.coe a))
    -/
    simp_rw [Function.comp_def, h, ← comp_apply (f := g)]
    /-
      α : Type u_1
      inst✝¹ : UniformSpace α
      pkg : AbstractCompletion α
      β : Type u_2
      inst✝ : UniformSpace β
      pkg' : AbstractCompletion β
      f : α → β
      g : pkg.space → pkg'.space
      hg : UniformContinuous g
      h : ∀ (a : α), Eq (pkg'.coe (f a)) (g (pkg.coe a))
      a : α
      ⊢ Eq (pkg.extend (fun x => Function.comp g pkg.coe x) (pkg.coe a)) (Function.c …
    -/
    rw [pkg.extend_coe (hg.comp pkg.uniformContinuous_coe)]
    /-
      🎉 no goals
    -/


@[simp]
theorem map_id : pkg.map pkg id = id :=
  pkg.map_unique pkg uniformContinuous_id fun _ => rfl


theorem extend_map [CompleteSpace γ] [T0Space γ] {f : β → γ} {g : α → β}
    (hf : UniformContinuous f) (hg : UniformContinuous g) :
    pkg'.extend f ∘ map g = pkg.extend (f ∘ g) :=
  pkg.funext (pkg'.continuous_extend.comp (pkg.continuous_map pkg' _)) pkg.continuous_extend
    fun a => by
    /-
      α : Type u_1
      inst✝⁴ : UniformSpace α
      pkg : AbstractCompletion α
      β : Type u_2
      inst✝³ : UniformSpace β
      pkg' : AbstractCompletion β
      γ : Type u_3
      inst✝² : UniformSpace γ
      inst✝¹ : CompleteSpace γ
      inst✝ : T0Space γ
      f : β → γ
      g : α → β
      hf : UniformContinuous f
      hg : UniformContinuous g
      a : α
      ⊢ Eq (Function.comp (pkg'.extend f) (pkg.map pkg' g) (pkg.coe a)) (pkg.extend  …
    -/
    rw [pkg.extend_coe (hf.comp hg), comp_apply, pkg.map_coe pkg' hg, pkg'.extend_coe hf]
    /-
      α : Type u_1
      inst✝⁴ : UniformSpace α
      pkg : AbstractCompletion α
      β : Type u_2
      inst✝³ : UniformSpace β
      pkg' : AbstractCompletion β
      γ : Type u_3
      inst✝² : UniformSpace γ
      inst✝¹ : CompleteSpace γ
      inst✝ : T0Space γ
      f : β → γ
      g : α → β
      hf : UniformContinuous f
      hg : UniformContinuous g
      a : α
      ⊢ Eq (f (g a)) (Function.comp f g a)
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem map_comp {g : β → γ} {f : α → β} (hg : UniformContinuous g) (hf : UniformContinuous f) :
    pkg'.map pkg'' g ∘ pkg.map pkg' f = pkg.map pkg'' (g ∘ f) :=
  pkg.extend_map pkg' (pkg''.uniformContinuous_coe.comp hg) hf


/-- The comparison map between two completions of the same uniform space. -/
def compare : pkg.space → pkg'.space :=
  pkg.extend pkg'.coe


theorem uniformContinuous_compare : UniformContinuous (pkg.compare pkg') :=
  pkg.uniformContinuous_extend


theorem compare_coe (a : α) : pkg.compare pkg' (pkg.coe a) = pkg'.coe a :=
  pkg.extend_coe pkg'.uniformContinuous_coe a


theorem inverse_compare : pkg.compare pkg' ∘ pkg'.compare pkg = id := by
  /-
    α : Type u_1
    inst✝ : UniformSpace α
    pkg pkg' : AbstractCompletion α
    ⊢ Eq (Function.comp (pkg.compare pkg') (pkg'.compare pkg)) id
  -/
  have uc := pkg.uniformContinuous_compare pkg'
  /-
    α : Type u_1
    inst✝ : UniformSpace α
    pkg pkg' : AbstractCompletion α
    uc : UniformContinuous (pkg.compare pkg')
    ⊢ Eq (Function.comp (pkg.compare pkg') (pkg'.compare pkg)) id
  -/
  have uc' := pkg'.uniformContinuous_compare pkg
  /-
    α : Type u_1
    inst✝ : UniformSpace α
    pkg pkg' : AbstractCompletion α
    uc : UniformContinuous (pkg.compare pkg')
    uc' : UniformContinuous (pkg'.compare pkg)
    ⊢ Eq (Function.comp (pkg.compare pkg') (pkg'.compare pkg)) id
  -/
  apply pkg'.funext (uc.comp uc').continuous continuous_id
  /-
    α : Type u_1
    inst✝ : UniformSpace α
    pkg pkg' : AbstractCompletion α
    uc : UniformContinuous (pkg.compare pkg')
    uc' : UniformContinuous (pkg'.compare pkg)
    ⊢ ∀ (a : α), Eq (Function.comp (pkg.compare pkg') (pkg'.compare pkg) (pkg'.coe …
  -/
  intro a
  /-
    α : Type u_1
    inst✝ : UniformSpace α
    pkg pkg' : AbstractCompletion α
    uc : UniformContinuous (pkg.compare pkg')
    uc' : UniformContinuous (pkg'.compare pkg)
    a : α
    ⊢ Eq (Function.comp (pkg.compare pkg') (pkg'.compare pkg) (pkg'.coe a)) (id (p …
  -/
  rw [comp_apply, pkg'.compare_coe pkg, pkg.compare_coe pkg']
  /-
    α : Type u_1
    inst✝ : UniformSpace α
    pkg pkg' : AbstractCompletion α
    uc : UniformContinuous (pkg.compare pkg')
    uc' : UniformContinuous (pkg'.compare pkg)
    a : α
    ⊢ Eq (pkg'.coe a) (id (pkg'.coe a))
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The uniform bijection between two completions of the same uniform space. -/
def compareEquiv : pkg.space ≃ᵤ pkg'.space where
  toFun := pkg.compare pkg'
  invFun := pkg'.compare pkg
  left_inv := congr_fun (pkg'.inverse_compare pkg)
  right_inv := congr_fun (pkg.inverse_compare pkg')
  uniformContinuous_toFun := uniformContinuous_compare _ _
  uniformContinuous_invFun := uniformContinuous_compare _ _


theorem uniformContinuous_compareEquiv : UniformContinuous (pkg.compareEquiv pkg') :=
  pkg.uniformContinuous_compare pkg'


theorem uniformContinuous_compareEquiv_symm : UniformContinuous (pkg.compareEquiv pkg').symm :=
  pkg'.uniformContinuous_compare pkg



theorem compare_comp_eq_compare (γ : Type*) [TopologicalSpace γ]
    [T3Space γ] {f : α → γ} (cont_f : Continuous f) :
    letI := pkg.uniformStruct.toTopologicalSpace
    letI := pkg'.uniformStruct.toTopologicalSpace
    (∀ a : pkg.space,
      Filter.Tendsto f (Filter.comap pkg.coe (𝓝 a)) (𝓝 ((pkg.isDenseInducing.extend f) a))) →
      pkg.isDenseInducing.extend f ∘ pkg'.compare pkg = pkg'.isDenseInducing.extend f := by
  /-
    α : Type u_1
    inst✝² : UniformSpace α
    pkg pkg' : AbstractCompletion α
    γ : Type u_3
    inst✝¹ : TopologicalSpace γ
    inst✝ : T3Space γ
    f : α → γ
    cont_f : Continuous f
    ⊢ (∀ (a : pkg.space), Filter.Tendsto f (Filter.comap pkg.coe (nhds a)) (nhds ( …
  -/
  let _ := pkg'.uniformStruct
  /-
    α : Type u_1
    inst✝² : UniformSpace α
    pkg pkg' : AbstractCompletion α
    γ : Type u_3
    inst✝¹ : TopologicalSpace γ
    inst✝ : T3Space γ
    f : α → γ
    cont_f : Continuous f
    x✝ : UniformSpace pkg'.space := pkg'.uniformStruct
    ⊢ (∀ (a : pkg.space), Filter.Tendsto f (Filter.comap pkg.coe (nhds a)) (nhds ( …
  -/
  let _ := pkg.uniformStruct
  /-
    α : Type u_1
    inst✝² : UniformSpace α
    pkg pkg' : AbstractCompletion α
    γ : Type u_3
    inst✝¹ : TopologicalSpace γ
    inst✝ : T3Space γ
    f : α → γ
    cont_f : Continuous f
    x✝¹ : UniformSpace pkg'.space := pkg'.uniformStruct
    x✝ : UniformSpace pkg.space := pkg.uniformStruct
    ⊢ (∀ (a : pkg.space), Filter.Tendsto f (Filter.comap pkg.coe (nhds a)) (nhds ( …
  -/
  intro h
  have (x : α) : (pkg.isDenseInducing.extend f ∘ pkg'.compare pkg) (pkg'.coe x) = f x := by
    simp only [Function.comp_apply, compare_coe, IsDenseInducing.extend_eq _ cont_f, implies_true]
  apply (IsDenseInducing.extend_unique (AbstractCompletion.isDenseInducing _) this
    (Continuous.comp _ (uniformContinuous_compare pkg' pkg).continuous )).symm
  /-
    α : Type u_1
    inst✝² : UniformSpace α
    pkg pkg' : AbstractCompletion α
    γ : Type u_3
    inst✝¹ : TopologicalSpace γ
    inst✝ : T3Space γ
    f : α → γ
    cont_f : Continuous f
    x✝¹ : UniformSpace pkg'.space := pkg'.uniformStruct
    x✝ : UniformSpace pkg.space := pkg.uniformStruct
    h : ∀ (a : pkg.space), Filter.Tendsto f (Filter.comap pkg.coe (nhds a)) (nhds  …
    this : ∀ (x : α), Eq (Function.comp (⋯.extend f) (pkg'.compare pkg) (pkg'.coe  …
    ⊢ Continuous (⋯.extend f)
  -/
  apply IsDenseInducing.continuous_extend
  /-
    case hf
    α : Type u_1
    inst✝² : UniformSpace α
    pkg pkg' : AbstractCompletion α
    γ : Type u_3
    inst✝¹ : TopologicalSpace γ
    inst✝ : T3Space γ
    f : α → γ
    cont_f : Continuous f
    x✝¹ : UniformSpace pkg'.space := pkg'.uniformStruct
    x✝ : UniformSpace pkg.space := pkg.uniformStruct
    h : ∀ (a : pkg.space), Filter.Tendsto f (Filter.comap pkg.coe (nhds a)) (nhds  …
    this : ∀ (x : α), Eq (Function.comp (⋯.extend f) (pkg'.compare pkg) (pkg'.coe  …
    ⊢ ∀ (b : pkg.space), Exists fun c => Filter.Tendsto f (Filter.comap pkg.coe (n …
  -/
  exact fun a ↦ ⟨(pkg.isDenseInducing.extend f) a, h a⟩
  /-
    🎉 no goals
  -/


/-- Products of completions -/
protected def prod : AbstractCompletion (α × β) where
  space := hatα × hatβ
  coe p := ⟨ι p.1, ι' p.2⟩
  uniformStruct := inferInstance
  complete := inferInstance
  separation := inferInstance
  isUniformInducing := IsUniformInducing.prod pkg.isUniformInducing pkg'.isUniformInducing
  dense := pkg.dense.prodMap pkg'.dense


/-- Extend two variable map to completions. -/
protected def extend₂ (f : α → β → γ) : hatα → hatβ → γ :=
  curry <| (pkg.prod pkg').extend (uncurry f)


theorem extension₂_coe_coe (hf : UniformContinuous <| uncurry f) (a : α) (b : β) :
    pkg.extend₂ pkg' f (ι a) (ι' b) = f a b :=
  show (pkg.prod pkg').extend (uncurry f) ((pkg.prod pkg').coe (a, b)) = uncurry f (a, b) from
    (pkg.prod pkg').extend_coe hf _


theorem uniformContinuous_extension₂ : UniformContinuous₂ (pkg.extend₂ pkg' f) := by
  /-
    α : Type u_1
    inst✝³ : UniformSpace α
    pkg : AbstractCompletion α
    β : Type u_2
    inst✝² : UniformSpace β
    pkg' : AbstractCompletion β
    γ : Type u_3
    inst✝¹ : UniformSpace γ
    f : α → β → γ
    inst✝ : CompleteSpace γ
    ⊢ UniformContinuous₂ (pkg.extend₂ pkg' f)
  -/
  rw [uniformContinuous₂_def, AbstractCompletion.extend₂, uncurry_curry]
  /-
    α : Type u_1
    inst✝³ : UniformSpace α
    pkg : AbstractCompletion α
    β : Type u_2
    inst✝² : UniformSpace β
    pkg' : AbstractCompletion β
    γ : Type u_3
    inst✝¹ : UniformSpace γ
    f : α → β → γ
    inst✝ : CompleteSpace γ
    ⊢ UniformContinuous ((pkg.prod pkg').extend (Function.uncurry f))
  -/
  apply uniformContinuous_extend
  /-
    🎉 no goals
  -/


local notation "hatγ" => pkg''.space


local notation "ι''" => pkg''.coe


local notation f " ∘₂ " g => bicompr f g


/-- Lift two variable maps to completions. -/
protected def map₂ (f : α → β → γ) : hatα → hatβ → hatγ :=
  pkg.extend₂ pkg' (pkg''.coe ∘₂ f)


theorem uniformContinuous_map₂ (f : α → β → γ) : UniformContinuous₂ (pkg.map₂ pkg' pkg'' f) :=
  AbstractCompletion.uniformContinuous_extension₂ pkg pkg' _


theorem continuous_map₂ {δ} [TopologicalSpace δ] {f : α → β → γ} {a : δ → hatα} {b : δ → hatβ}
    (ha : Continuous a) (hb : Continuous b) :
    Continuous fun d : δ => pkg.map₂ pkg' pkg'' f (a d) (b d) :=
  ((pkg.uniformContinuous_map₂ pkg' pkg'' f).continuous.comp (Continuous.prod_mk ha hb) : _)


theorem map₂_coe_coe (a : α) (b : β) (f : α → β → γ) (hf : UniformContinuous₂ f) :
    pkg.map₂ pkg' pkg'' f (ι a) (ι' b) = ι'' (f a b) :=
  pkg.extension₂_coe_coe (f := pkg''.coe ∘₂ f) pkg' (pkg''.uniformContinuous_coe.comp hf) a b


