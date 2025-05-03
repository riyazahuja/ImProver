/-- A map `f : α → β` between uniform spaces is called *uniform inducing* if the uniformity filter
on `α` is the pullback of the uniformity filter on `β` under `Prod.map f f`. If `α` is a separated
space, then this implies that `f` is injective, hence it is a `IsUniformEmbedding`. -/
@[mk_iff]
structure IsUniformInducing (f : α → β) : Prop where
  /-- The uniformity filter on the domain is the pullback of the uniformity filter on the codomain
  under `Prod.map f f`. -/
  comap_uniformity : comap (fun x : α × α => (f x.1, f x.2)) (𝓤 β) = 𝓤 α


@[deprecated (since := "2024-10-08")] alias UniformInducing := IsUniformInducing


lemma isUniformInducing_iff_uniformSpace {f : α → β} :
    IsUniformInducing f ↔ ‹UniformSpace β›.comap f = ‹UniformSpace α› := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    f : α → β
    ⊢ Iff (IsUniformInducing f) (Eq (UniformSpace.comap f inst✝) inst✝¹)
  -/
  rw [isUniformInducing_iff, UniformSpace.ext_iff, Filter.ext_iff]
  /-
    α : Type u
    β : Type v
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    f : α → β
    ⊢ Iff (∀ (s : Set (Prod α α)), Iff (Membership.mem (Filter.comap (fun x => { f …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-05")]
alias uniformInducing_iff_uniformSpace := isUniformInducing_iff_uniformSpace


protected alias ⟨IsUniformInducing.comap_uniformSpace, _⟩ := isUniformInducing_iff_uniformSpace


@[deprecated (since := "2024-10-08")] alias UniformInducing.comap_uniformSpace :=
  IsUniformInducing.comap_uniformSpace


lemma isUniformInducing_iff' {f : α → β} :
    IsUniformInducing f ↔ UniformContinuous f ∧ comap (Prod.map f f) (𝓤 β) ≤ 𝓤 α := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    f : α → β
    ⊢ Iff (IsUniformInducing f) (And (UniformContinuous f) (LE.le (Filter.comap (P …
  -/
  rw [isUniformInducing_iff, UniformContinuous, tendsto_iff_comap, le_antisymm_iff, and_comm]; rfl
                                                                                               /-
                                                                                                 🎉 no goals
                                                                                               -/


@[deprecated (since := "2024-10-05")]
alias uniformInducing_iff' := isUniformInducing_iff'


protected lemma Filter.HasBasis.isUniformInducing_iff {ι ι'} {p : ι → Prop} {p' : ι' → Prop} {s s'}
    (h : (𝓤 α).HasBasis p s) (h' : (𝓤 β).HasBasis p' s') {f : α → β} :
    IsUniformInducing f ↔
      (∀ i, p' i → ∃ j, p j ∧ ∀ x y, (x, y) ∈ s j → (f x, f y) ∈ s' i) ∧
        (∀ j, p j → ∃ i, p' i ∧ ∀ x y, (f x, f y) ∈ s' i → (x, y) ∈ s j) := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    ι : Sort u_1
    ι' : Sort u_2
    p : ι → Prop
    p' : ι' → Prop
    s : ι → Set (Prod α α)
    s' : ι' → Set (Prod β β)
    h : (uniformity α).HasBasis p s
    h' : (uniformity β).HasBasis p' s'
    f : α → β
    ⊢ Iff (IsUniformInducing f) (And (∀ (i : ι'), p' i → Exists fun j => And (p j) …
  -/
  simp [isUniformInducing_iff', h.uniformContinuous_iff h', (h'.comap _).le_basis_iff h, subset_def]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-05")]
alias Filter.HasBasis.uniformInducing_iff := Filter.HasBasis.isUniformInducing_iff


theorem IsUniformInducing.mk' {f : α → β}
    (h : ∀ s, s ∈ 𝓤 α ↔ ∃ t ∈ 𝓤 β, ∀ x y : α, (f x, f y) ∈ t → (x, y) ∈ s) : IsUniformInducing f :=
      /-
        α : Type u
        β : Type v
        inst✝¹ : UniformSpace α
        inst✝ : UniformSpace β
        f : α → β
        h : ∀ (s : Set (Prod α α)), Iff (Membership.mem (uniformity α) s) (Exists fun  …
        ⊢ Eq (Filter.comap (fun x => { fst := f x.1, snd := f x.2 }) (uniformity β)) ( …
      -/
  ⟨by simp [eq_comm, Filter.ext_iff, subset_def, h]⟩
      /-
        🎉 no goals
      -/


@[deprecated (since := "2024-10-05")]
alias UniformInducing.mk' := IsUniformInducing.mk'


theorem IsUniformInducing.id : IsUniformInducing (@id α) :=
      /-
        α : Type u
        inst✝ : UniformSpace α
        ⊢ Eq (Filter.comap (fun x => { fst := _root_.id x.1, snd := _root_.id x.2 }) ( …
      -/
  ⟨by rw [← Prod.map_def, Prod.map_id, comap_id]⟩
      /-
        🎉 no goals
      -/


@[deprecated (since := "2024-10-05")]
alias uniformInducing_id := IsUniformInducing.id


theorem IsUniformInducing.comp {g : β → γ} (hg : IsUniformInducing g) {f : α → β}
    (hf : IsUniformInducing f) : IsUniformInducing (g ∘ f) :=
      /-
        α : Type u
        β : Type v
        γ : Type w
        inst✝² : UniformSpace α
        inst✝¹ : UniformSpace β
        inst✝ : UniformSpace γ
        g : β → γ
        hg : IsUniformInducing g
        f : α → β
        hf : IsUniformInducing f
        ⊢ Eq (Filter.comap (fun x => { fst := Function.comp g f x.1, snd := Function.c …
      -/
  ⟨by rw [← hf.1, ← hg.1, comap_comap]; rfl⟩
                                        /-
                                          🎉 no goals
                                        -/


@[deprecated (since := "2024-10-05")]
alias UniformInducing.comp := IsUniformInducing.comp


theorem IsUniformInducing.of_comp_iff {g : β → γ} (hg : IsUniformInducing g) {f : α → β} :
    IsUniformInducing (g ∘ f) ↔ IsUniformInducing f := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    inst✝² : UniformSpace α
    inst✝¹ : UniformSpace β
    inst✝ : UniformSpace γ
    g : β → γ
    hg : IsUniformInducing g
    f : α → β
    ⊢ Iff (IsUniformInducing (Function.comp g f)) (IsUniformInducing f)
  -/
  refine ⟨fun h ↦ ?_, hg.comp⟩
  rw [isUniformInducing_iff, ← hg.comap_uniformity, comap_comap, ← h.comap_uniformity,
    Function.comp_def, Function.comp_def]


@[deprecated (since := "2024-10-05")]
alias UniformInducing.of_comp_iff := IsUniformInducing.of_comp_iff


theorem IsUniformInducing.basis_uniformity {f : α → β} (hf : IsUniformInducing f) {ι : Sort*}
    {p : ι → Prop} {s : ι → Set (β × β)} (H : (𝓤 β).HasBasis p s) :
    (𝓤 α).HasBasis p fun i => Prod.map f f ⁻¹' s i :=
  hf.1 ▸ H.comap _


@[deprecated (since := "2024-10-05")]
alias UniformInducing.basis_uniformity := IsUniformInducing.basis_uniformity


theorem IsUniformInducing.cauchy_map_iff {f : α → β} (hf : IsUniformInducing f) {F : Filter α} :
    Cauchy (map f F) ↔ Cauchy F := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    f : α → β
    hf : IsUniformInducing f
    F : Filter α
    ⊢ Iff (Cauchy (Filter.map f F)) (Cauchy F)
  -/
  simp only [Cauchy, map_neBot_iff, prod_map_map_eq, map_le_iff_le_comap, ← hf.comap_uniformity]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-05")]
alias UniformInducing.cauchy_map_iff := IsUniformInducing.cauchy_map_iff


theorem IsUniformInducing.of_comp {f : α → β} {g : β → γ} (hf : UniformContinuous f)
    (hg : UniformContinuous g) (hgf : IsUniformInducing (g ∘ f)) : IsUniformInducing f := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    inst✝² : UniformSpace α
    inst✝¹ : UniformSpace β
    inst✝ : UniformSpace γ
    f : α → β
    g : β → γ
    hf : UniformContinuous f
    hg : UniformContinuous g
    hgf : IsUniformInducing (Function.comp g f)
    ⊢ IsUniformInducing f
  -/
  refine ⟨le_antisymm ?_ hf.le_comap⟩
  /-
    α : Type u
    β : Type v
    γ : Type w
    inst✝² : UniformSpace α
    inst✝¹ : UniformSpace β
    inst✝ : UniformSpace γ
    f : α → β
    g : β → γ
    hf : UniformContinuous f
    hg : UniformContinuous g
    hgf : IsUniformInducing (Function.comp g f)
    ⊢ LE.le (Filter.comap (fun x => { fst := f x.1, snd := f x.2 }) (uniformity β) …
  -/
  rw [← hgf.1, ← Prod.map_def, ← Prod.map_def, ← Prod.map_comp_map f f g g, ← comap_comap]
  /-
    α : Type u
    β : Type v
    γ : Type w
    inst✝² : UniformSpace α
    inst✝¹ : UniformSpace β
    inst✝ : UniformSpace γ
    f : α → β
    g : β → γ
    hf : UniformContinuous f
    hg : UniformContinuous g
    hgf : IsUniformInducing (Function.comp g f)
    ⊢ LE.le (Filter.comap (Prod.map f f) (uniformity β)) (Filter.comap (Prod.map f …
  -/
  exact comap_mono hg.le_comap
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-05")]
alias uniformInducing_of_compose := IsUniformInducing.of_comp


theorem IsUniformInducing.uniformContinuous {f : α → β} (hf : IsUniformInducing f) :
    UniformContinuous f := (isUniformInducing_iff'.1 hf).1


@[deprecated (since := "2024-10-05")]
alias UniformInducing.uniformContinuous := IsUniformInducing.uniformContinuous


theorem IsUniformInducing.uniformContinuous_iff {f : α → β} {g : β → γ} (hg : IsUniformInducing g) :
    UniformContinuous f ↔ UniformContinuous (g ∘ f) := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    inst✝² : UniformSpace α
    inst✝¹ : UniformSpace β
    inst✝ : UniformSpace γ
    f : α → β
    g : β → γ
    hg : IsUniformInducing g
    ⊢ Iff (UniformContinuous f) (UniformContinuous (Function.comp g f))
  -/
  dsimp only [UniformContinuous, Tendsto]
  /-
    α : Type u
    β : Type v
    γ : Type w
    inst✝² : UniformSpace α
    inst✝¹ : UniformSpace β
    inst✝ : UniformSpace γ
    f : α → β
    g : β → γ
    hg : IsUniformInducing g
    ⊢ Iff (LE.le (Filter.map (fun x => { fst := f x.1, snd := f x.2 }) (uniformity …
  -/
  simp only [← hg.comap_uniformity, ← map_le_iff_le_comap, Filter.map_map, Function.comp_def]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-05")]
alias UniformInducing.uniformContinuous_iff := IsUniformInducing.uniformContinuous_iff


protected theorem IsUniformInducing.isUniformInducing_comp_iff {f : α → β} {g : β → γ}
    (hg : IsUniformInducing g) : IsUniformInducing (g ∘ f) ↔ IsUniformInducing f := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    inst✝² : UniformSpace α
    inst✝¹ : UniformSpace β
    inst✝ : UniformSpace γ
    f : α → β
    g : β → γ
    hg : IsUniformInducing g
    ⊢ Iff (IsUniformInducing (Function.comp g f)) (IsUniformInducing f)
  -/
  simp only [isUniformInducing_iff, ← hg.comap_uniformity, comap_comap, Function.comp_def]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-05")]
alias UniformInducing.uniformInducing_comp_iff := IsUniformInducing.isUniformInducing_comp_iff


theorem IsUniformInducing.uniformContinuousOn_iff {f : α → β} {g : β → γ} {S : Set α}
    (hg : IsUniformInducing g) :
    UniformContinuousOn f S ↔ UniformContinuousOn (g ∘ f) S := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    inst✝² : UniformSpace α
    inst✝¹ : UniformSpace β
    inst✝ : UniformSpace γ
    f : α → β
    g : β → γ
    S : Set α
    hg : IsUniformInducing g
    ⊢ Iff (UniformContinuousOn f S) (UniformContinuousOn (Function.comp g f) S)
  -/
  dsimp only [UniformContinuousOn, Tendsto]
  /-
    α : Type u
    β : Type v
    γ : Type w
    inst✝² : UniformSpace α
    inst✝¹ : UniformSpace β
    inst✝ : UniformSpace γ
    f : α → β
    g : β → γ
    S : Set α
    hg : IsUniformInducing g
    ⊢ Iff (LE.le (Filter.map (fun x => { fst := f x.1, snd := f x.2 }) (Min.min (u …
  -/
  rw [← hg.comap_uniformity, ← map_le_iff_le_comap, Filter.map_map, comp_def, comp_def]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-05")]
alias UniformInducing.uniformContinuousOn_iff := IsUniformInducing.uniformContinuousOn_iff


theorem IsUniformInducing.isInducing {f : α → β} (h : IsUniformInducing f) : IsInducing f := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    f : α → β
    h : IsUniformInducing f
    ⊢ Topology.IsInducing f
  -/
  obtain rfl := h.comap_uniformSpace
  /-
    α : Type u
    β : Type v
    inst✝ : UniformSpace β
    f : α → β
    h : IsUniformInducing f
    ⊢ Topology.IsInducing f
  -/
  exact .induced f
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-28")]
alias IsUniformInducing.inducing := IsUniformInducing.isInducing


@[deprecated (since := "2024-10-05")]
alias UniformInducing.isInducing := IsUniformInducing.isInducing


@[deprecated (since := "2024-10-28")] alias UniformInducing.inducing := UniformInducing.isInducing


theorem IsUniformInducing.prod {α' : Type*} {β' : Type*} [UniformSpace α'] [UniformSpace β']
    {e₁ : α → α'} {e₂ : β → β'} (h₁ : IsUniformInducing e₁) (h₂ : IsUniformInducing e₂) :
    IsUniformInducing fun p : α × β => (e₁ p.1, e₂ p.2) :=
      /-
        α : Type u
        β : Type v
        inst✝³ : UniformSpace α
        inst✝² : UniformSpace β
        α' : Type u_1
        β' : Type u_2
        inst✝¹ : UniformSpace α'
        inst✝ : UniformSpace β'
        e₁ : α → α'
        e₂ : β → β'
        h₁ : IsUniformInducing e₁
        h₂ : IsUniformInducing e₂
        ⊢ Eq (Filter.comap (fun x => { fst := { fst := e₁ x.1.1, snd := e₂ x.1.2 }, sn …
      -/
  ⟨by simp [Function.comp_def, uniformity_prod, ← h₁.1, ← h₂.1, comap_inf, comap_comap]⟩
      /-
        🎉 no goals
      -/


@[deprecated (since := "2024-10-05")]
alias UniformInducing.prod := IsUniformInducing.prod


lemma IsUniformInducing.isDenseInducing (h : IsUniformInducing f) (hd : DenseRange f) :
    IsDenseInducing f where
  toIsInducing := h.isInducing
  dense := hd


@[deprecated (since := "2024-10-05")]
alias UniformInducing.isDenseInducing := IsUniformInducing.isDenseInducing


lemma SeparationQuotient.isUniformInducing_mk :
    IsUniformInducing (mk : α → SeparationQuotient α) :=
  ⟨comap_mk_uniformity⟩


@[deprecated (since := "2024-10-05")]
alias SeparationQuotient.uniformInducing_mk := SeparationQuotient.isUniformInducing_mk


protected theorem IsUniformInducing.injective [T0Space α] {f : α → β} (h : IsUniformInducing f) :
    Injective f :=
  h.isInducing.injective


@[deprecated (since := "2024-10-05")]
alias UniformInducing.injective := IsUniformInducing.injective


/-- A map `f : α → β` between uniform spaces is a *uniform embedding* if it is uniform inducing and
injective. If `α` is a separated space, then the latter assumption follows from the former. -/
@[mk_iff]
structure IsUniformEmbedding (f : α → β) extends IsUniformInducing f : Prop where
  /-- A uniform embedding is injective. -/
  injective : Function.Injective f


lemma IsUniformEmbedding.isUniformInducing (hf : IsUniformEmbedding f) : IsUniformInducing f :=
  hf.toIsUniformInducing


@[deprecated (since := "2024-10-03")] alias UniformEmbedding := IsUniformEmbedding


theorem isUniformEmbedding_iff' {f : α → β} :
    IsUniformEmbedding f ↔
      Injective f ∧ UniformContinuous f ∧ comap (Prod.map f f) (𝓤 β) ≤ 𝓤 α := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    f : α → β
    ⊢ Iff (IsUniformEmbedding f) (And (Function.Injective f) (And (UniformContinuo …
  -/
  rw [isUniformEmbedding_iff, and_comm, isUniformInducing_iff']
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-01")]
alias uniformEmbedding_iff' := isUniformEmbedding_iff'


theorem Filter.HasBasis.isUniformEmbedding_iff' {ι ι'} {p : ι → Prop} {p' : ι' → Prop} {s s'}
    (h : (𝓤 α).HasBasis p s) (h' : (𝓤 β).HasBasis p' s') {f : α → β} :
    IsUniformEmbedding f ↔ Injective f ∧
      (∀ i, p' i → ∃ j, p j ∧ ∀ x y, (x, y) ∈ s j → (f x, f y) ∈ s' i) ∧
        (∀ j, p j → ∃ i, p' i ∧ ∀ x y, (f x, f y) ∈ s' i → (x, y) ∈ s j) := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    ι : Sort u_1
    ι' : Sort u_2
    p : ι → Prop
    p' : ι' → Prop
    s : ι → Set (Prod α α)
    s' : ι' → Set (Prod β β)
    h : (uniformity α).HasBasis p s
    h' : (uniformity β).HasBasis p' s'
    f : α → β
    ⊢ Iff (IsUniformEmbedding f) (And (Function.Injective f) (And (∀ (i : ι'), p'  …
  -/
  rw [isUniformEmbedding_iff, and_comm, h.isUniformInducing_iff h']
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-01")]
alias Filter.HasBasis.uniformEmbedding_iff' := Filter.HasBasis.isUniformEmbedding_iff'


theorem Filter.HasBasis.isUniformEmbedding_iff {ι ι'} {p : ι → Prop} {p' : ι' → Prop} {s s'}
    (h : (𝓤 α).HasBasis p s) (h' : (𝓤 β).HasBasis p' s') {f : α → β} :
    IsUniformEmbedding f ↔ Injective f ∧ UniformContinuous f ∧
      (∀ j, p j → ∃ i, p' i ∧ ∀ x y, (f x, f y) ∈ s' i → (x, y) ∈ s j) := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    ι : Sort u_1
    ι' : Sort u_2
    p : ι → Prop
    p' : ι' → Prop
    s : ι → Set (Prod α α)
    s' : ι' → Set (Prod β β)
    h : (uniformity α).HasBasis p s
    h' : (uniformity β).HasBasis p' s'
    f : α → β
    ⊢ Iff (IsUniformEmbedding f) (And (Function.Injective f) (And (UniformContinuo …
  -/
  simp only [h.isUniformEmbedding_iff' h', h.uniformContinuous_iff h']
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-01")]
alias Filter.HasBasis.uniformEmbedding_iff := Filter.HasBasis.isUniformEmbedding_iff


theorem isUniformEmbedding_subtype_val {p : α → Prop} :
    IsUniformEmbedding (Subtype.val : Subtype p → α) :=
  { comap_uniformity := rfl
    injective := Subtype.val_injective }


@[deprecated (since := "2024-10-01")]
alias uniformEmbedding_subtype_val := isUniformEmbedding_subtype_val


theorem isUniformEmbedding_set_inclusion {s t : Set α} (hst : s ⊆ t) :
    IsUniformEmbedding (inclusion hst) where
                         /-
                           α : Type u
                           inst✝ : UniformSpace α
                           s t : Set α
                           hst : HasSubset.Subset s t
                           ⊢ Eq (Filter.comap (fun x => { fst := Set.inclusion hst x.1, snd := Set.inclus …
                         -/
  comap_uniformity := by rw [uniformity_subtype, uniformity_subtype, comap_comap]; rfl
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/
  injective := inclusion_injective hst


@[deprecated (since := "2024-10-01")]
alias uniformEmbedding_set_inclusion := isUniformEmbedding_set_inclusion


theorem IsUniformEmbedding.comp {g : β → γ} (hg : IsUniformEmbedding g) {f : α → β}
    (hf : IsUniformEmbedding f) : IsUniformEmbedding (g ∘ f) where
  toIsUniformInducing := hg.isUniformInducing.comp hf.isUniformInducing
  injective := hg.injective.comp hf.injective


@[deprecated (since := "2024-10-01")]
alias UniformEmbedding.comp := IsUniformEmbedding.comp


theorem IsUniformEmbedding.of_comp_iff {g : β → γ} (hg : IsUniformEmbedding g) {f : α → β} :
    IsUniformEmbedding (g ∘ f) ↔ IsUniformEmbedding f := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    inst✝² : UniformSpace α
    inst✝¹ : UniformSpace β
    inst✝ : UniformSpace γ
    g : β → γ
    hg : IsUniformEmbedding g
    f : α → β
    ⊢ Iff (IsUniformEmbedding (Function.comp g f)) (IsUniformEmbedding f)
  -/
  simp_rw [isUniformEmbedding_iff, hg.isUniformInducing.of_comp_iff, hg.injective.of_comp_iff f]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-01")]
alias UniformEmbedding.of_comp_iff := IsUniformEmbedding.of_comp_iff


theorem Equiv.isUniformEmbedding {α β : Type*} [UniformSpace α] [UniformSpace β] (f : α ≃ β)
    (h₁ : UniformContinuous f) (h₂ : UniformContinuous f.symm) : IsUniformEmbedding f :=
                                                 /-
                                                   α : Type u_1
                                                   β : Type u_2
                                                   inst✝¹ : UniformSpace α
                                                   inst✝ : UniformSpace β
                                                   f : Equiv α β
                                                   h₁ : UniformContinuous ⇑f
                                                   h₂ : UniformContinuous ⇑f.symm
                                                   ⊢ LE.le (Filter.comap (Prod.map ⇑f ⇑f) (uniformity β)) (uniformity α)
                                                 -/
  isUniformEmbedding_iff'.2 ⟨f.injective, h₁, by rwa [← Equiv.prodCongr_apply, ← map_equiv_symm]⟩
                                                 /-
                                                   🎉 no goals
                                                 -/


@[deprecated (since := "2024-10-01")]
alias Equiv.uniformEmbedding := Equiv.isUniformEmbedding


theorem isUniformEmbedding_inl : IsUniformEmbedding (Sum.inl : α → α ⊕ β) :=
  isUniformEmbedding_iff'.2 ⟨Sum.inl_injective, uniformContinuous_inl, fun s hs =>
    ⟨Prod.map Sum.inl Sum.inl '' s ∪ range (Prod.map Sum.inr Sum.inr),
      union_mem_sup (image_mem_map hs) range_mem_map,
                    /-
                      α : Type u
                      β : Type v
                      inst✝¹ : UniformSpace α
                      inst✝ : UniformSpace β
                      s : Set (Prod α α)
                      hs : Membership.mem (uniformity α) s
                      x : Prod α α
                      h : Membership.mem (Set.preimage (Prod.map Sum.inl Sum.inl) (Union.union (Set. …
                      ⊢ Membership.mem s x
                    -/
      fun x h => by simpa [Prod.map_apply'] using h⟩⟩
                    /-
                      🎉 no goals
                    -/


@[deprecated (since := "2024-10-01")]
alias uniformEmbedding_inl := isUniformEmbedding_inl


theorem isUniformEmbedding_inr : IsUniformEmbedding (Sum.inr : β → α ⊕ β) :=
  isUniformEmbedding_iff'.2 ⟨Sum.inr_injective, uniformContinuous_inr, fun s hs =>
    ⟨range (Prod.map Sum.inl Sum.inl) ∪ Prod.map Sum.inr Sum.inr '' s,
      union_mem_sup range_mem_map (image_mem_map hs),
                    /-
                      α : Type u
                      β : Type v
                      inst✝¹ : UniformSpace α
                      inst✝ : UniformSpace β
                      s : Set (Prod β β)
                      hs : Membership.mem (uniformity β) s
                      x : Prod β β
                      h : Membership.mem (Set.preimage (Prod.map Sum.inr Sum.inr) (Union.union (Set. …
                      ⊢ Membership.mem s x
                    -/
      fun x h => by simpa [Prod.map_apply'] using h⟩⟩
                    /-
                      🎉 no goals
                    -/


@[deprecated (since := "2024-10-01")]
alias uniformEmbedding_inr := isUniformEmbedding_inr


/-- If the domain of a `IsUniformInducing` map `f` is a T₀ space, then `f` is injective,
hence it is a `IsUniformEmbedding`. -/
protected theorem IsUniformInducing.isUniformEmbedding [T0Space α] {f : α → β}
    (hf : IsUniformInducing f) : IsUniformEmbedding f :=
  ⟨hf, hf.isInducing.injective⟩


@[deprecated (since := "2024-10-05")]
alias UniformInducing.isUniformEmbedding := IsUniformInducing.isUniformEmbedding


@[deprecated (since := "2024-10-01")]
alias IsUniformInducing.uniformEmbedding := IsUniformInducing.isUniformEmbedding


theorem isUniformEmbedding_iff_isUniformInducing [T0Space α] {f : α → β} :
    IsUniformEmbedding f ↔ IsUniformInducing f :=
  ⟨IsUniformEmbedding.isUniformInducing, IsUniformInducing.isUniformEmbedding⟩


@[deprecated (since := "2024-10-05")]
alias isUniformEmbedding_iff_uniformInducing := isUniformEmbedding_iff_isUniformInducing


@[deprecated (since := "2024-10-01")]
alias uniformEmbedding_iff_isUniformInducing := isUniformEmbedding_iff_isUniformInducing


/-- If a map `f : α → β` sends any two distinct points to point that are **not** related by a fixed
`s ∈ 𝓤 β`, then `f` is uniform inducing with respect to the discrete uniformity on `α`:
the preimage of `𝓤 β` under `Prod.map f f` is the principal filter generated by the diagonal in
`α × α`. -/
theorem comap_uniformity_of_spaced_out {α} {f : α → β} {s : Set (β × β)} (hs : s ∈ 𝓤 β)
    (hf : Pairwise fun x y => (f x, f y) ∉ s) : comap (Prod.map f f) (𝓤 β) = 𝓟 idRel := by
  /-
    β : Type v
    inst✝ : UniformSpace β
    α : Type u_1
    f : α → β
    s : Set (Prod β β)
    hs : Membership.mem (uniformity β) s
    hf : Pairwise fun x y => Not (Membership.mem s { fst := f x, snd := f y })
    ⊢ Eq (Filter.comap (Prod.map f f) (uniformity β)) (Filter.principal idRel)
  -/
  refine le_antisymm ?_ (@refl_le_uniformity α (UniformSpace.comap f _))
  calc
    comap (Prod.map f f) (𝓤 β) ≤ comap (Prod.map f f) (𝓟 s) := comap_mono (le_principal_iff.2 hs)
    _ = 𝓟 (Prod.map f f ⁻¹' s) := comap_principal
    _ ≤ 𝓟 idRel := principal_mono.2 ?_
  /-
    β : Type v
    inst✝ : UniformSpace β
    α : Type u_1
    f : α → β
    s : Set (Prod β β)
    hs : Membership.mem (uniformity β) s
    hf : Pairwise fun x y => Not (Membership.mem s { fst := f x, snd := f y })
    ⊢ HasSubset.Subset (Set.preimage (Prod.map f f) s) idRel
  -/
  rintro ⟨x, y⟩; simpa [not_imp_not] using @hf x y
                 /-
                   🎉 no goals
                 -/


/-- If a map `f : α → β` sends any two distinct points to point that are **not** related by a fixed
`s ∈ 𝓤 β`, then `f` is a uniform embedding with respect to the discrete uniformity on `α`. -/
theorem isUniformEmbedding_of_spaced_out {α} {f : α → β} {s : Set (β × β)} (hs : s ∈ 𝓤 β)
    (hf : Pairwise fun x y => (f x, f y) ∉ s) : @IsUniformEmbedding α β ⊥ ‹_› f := by
  /-
    β : Type v
    inst✝ : UniformSpace β
    α : Type u_1
    f : α → β
    s : Set (Prod β β)
    hs : Membership.mem (uniformity β) s
    hf : Pairwise fun x y => Not (Membership.mem s { fst := f x, snd := f y })
    ⊢ IsUniformEmbedding f
  -/
  let _ : UniformSpace α := ⊥; have := discreteTopology_bot α
  /-
    β : Type v
    inst✝ : UniformSpace β
    α : Type u_1
    f : α → β
    s : Set (Prod β β)
    hs : Membership.mem (uniformity β) s
    hf : Pairwise fun x y => Not (Membership.mem s { fst := f x, snd := f y })
    x✝ : UniformSpace α := Bot.bot
    this : DiscreteTopology α
    ⊢ IsUniformEmbedding f
  -/
  exact IsUniformInducing.isUniformEmbedding ⟨comap_uniformity_of_spaced_out hs hf⟩
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-01")]
alias uniformEmbedding_of_spaced_out := isUniformEmbedding_of_spaced_out


protected lemma IsUniformEmbedding.isEmbedding {f : α → β} (h : IsUniformEmbedding f) :
    IsEmbedding f where
  toIsInducing := h.toIsUniformInducing.isInducing
  injective := h.injective


@[deprecated (since := "2024-10-26")]
alias IsUniformEmbedding.embedding := IsUniformEmbedding.isEmbedding


@[deprecated (since := "2024-10-01")]
alias UniformEmbedding.embedding := IsUniformEmbedding.isEmbedding


theorem IsUniformEmbedding.isDenseEmbedding {f : α → β} (h : IsUniformEmbedding f)
    (hd : DenseRange f) : IsDenseEmbedding f :=
  { h.isEmbedding with dense := hd }


@[deprecated (since := "2024-10-01")]
alias UniformEmbedding.isDenseEmbedding := IsUniformEmbedding.isDenseEmbedding


@[deprecated (since := "2024-09-30")]
alias IsUniformEmbedding.denseEmbedding := IsUniformEmbedding.isDenseEmbedding


theorem isClosedEmbedding_of_spaced_out {α} [TopologicalSpace α] [DiscreteTopology α]
    [T0Space β] {f : α → β} {s : Set (β × β)} (hs : s ∈ 𝓤 β)
    (hf : Pairwise fun x y => (f x, f y) ∉ s) : IsClosedEmbedding f := by
  /-
    β : Type v
    inst✝³ : UniformSpace β
    α : Type u_1
    inst✝² : TopologicalSpace α
    inst✝¹ : DiscreteTopology α
    inst✝ : T0Space β
    f : α → β
    s : Set (Prod β β)
    hs : Membership.mem (uniformity β) s
    hf : Pairwise fun x y => Not (Membership.mem s { fst := f x, snd := f y })
    ⊢ Topology.IsClosedEmbedding f
  -/
  rcases @DiscreteTopology.eq_bot α _ _ with rfl; let _ : UniformSpace α := ⊥
  exact
    { (isUniformEmbedding_of_spaced_out hs hf).isEmbedding with
      isClosed_range := isClosed_range_of_spaced_out hs hf }


@[deprecated (since := "2024-10-20")]
alias closedEmbedding_of_spaced_out := isClosedEmbedding_of_spaced_out


theorem closure_image_mem_nhds_of_isUniformInducing {s : Set (α × α)} {e : α → β} (b : β)
    (he₁ : IsUniformInducing e) (he₂ : IsDenseInducing e) (hs : s ∈ 𝓤 α) :
    ∃ a, closure (e '' { a' | (a, a') ∈ s }) ∈ 𝓝 b := by
  obtain ⟨U, ⟨hU, hUo, hsymm⟩, hs⟩ :
    ∃ U, (U ∈ 𝓤 β ∧ IsOpen U ∧ SymmetricRel U) ∧ Prod.map e e ⁻¹' U ⊆ s := by
      rwa [← he₁.comap_uniformity, (uniformity_hasBasis_open_symmetric.comap _).mem_iff] at hs
  /-
    case intro.intro.intro.intro
    α : Type u
    β : Type v
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    s : Set (Prod α α)
    e : α → β
    b : β
    he₁ : IsUniformInducing e
    he₂ : IsDenseInducing e
    hs✝ : Membership.mem (uniformity α) s
    U : Set (Prod β β)
    hs : HasSubset.Subset (Set.preimage (Prod.map e e) U) s
    hU : Membership.mem (uniformity β) U
    hUo : IsOpen U
    hsymm : SymmetricRel U
    ⊢ Exists fun a => Membership.mem (nhds b) (closure (Set.image e (setOf fun a'  …
  -/
  rcases he₂.dense.mem_nhds (UniformSpace.ball_mem_nhds b hU) with ⟨a, ha⟩
  /-
    case intro.intro.intro.intro.intro
    α : Type u
    β : Type v
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    s : Set (Prod α α)
    e : α → β
    b : β
    he₁ : IsUniformInducing e
    he₂ : IsDenseInducing e
    hs✝ : Membership.mem (uniformity α) s
    U : Set (Prod β β)
    hs : HasSubset.Subset (Set.preimage (Prod.map e e) U) s
    hU : Membership.mem (uniformity β) U
    hUo : IsOpen U
    hsymm : SymmetricRel U
    a : α
    ha : Membership.mem (UniformSpace.ball b U) (e a)
    ⊢ Exists fun a => Membership.mem (nhds b) (closure (Set.image e (setOf fun a'  …
  -/
  refine ⟨a, mem_of_superset ?_ (closure_mono <| image_subset _ <| UniformSpace.ball_mono hs a)⟩
  /-
    case intro.intro.intro.intro.intro
    α : Type u
    β : Type v
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    s : Set (Prod α α)
    e : α → β
    b : β
    he₁ : IsUniformInducing e
    he₂ : IsDenseInducing e
    hs✝ : Membership.mem (uniformity α) s
    U : Set (Prod β β)
    hs : HasSubset.Subset (Set.preimage (Prod.map e e) U) s
    hU : Membership.mem (uniformity β) U
    hUo : IsOpen U
    hsymm : SymmetricRel U
    a : α
    ha : Membership.mem (UniformSpace.ball b U) (e a)
    ⊢ Membership.mem (nhds b) (closure (Set.image e (UniformSpace.ball a (Set.prei …
  -/
  have ho : IsOpen (UniformSpace.ball (e a) U) := UniformSpace.isOpen_ball (e a) hUo
  /-
    case intro.intro.intro.intro.intro
    α : Type u
    β : Type v
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    s : Set (Prod α α)
    e : α → β
    b : β
    he₁ : IsUniformInducing e
    he₂ : IsDenseInducing e
    hs✝ : Membership.mem (uniformity α) s
    U : Set (Prod β β)
    hs : HasSubset.Subset (Set.preimage (Prod.map e e) U) s
    hU : Membership.mem (uniformity β) U
    hUo : IsOpen U
    hsymm : SymmetricRel U
    a : α
    ha : Membership.mem (UniformSpace.ball b U) (e a)
    ho : IsOpen (UniformSpace.ball (e a) U)
    ⊢ Membership.mem (nhds b) (closure (Set.image e (UniformSpace.ball a (Set.prei …
  -/
  refine mem_of_superset (ho.mem_nhds <| (UniformSpace.mem_ball_symmetry hsymm).2 ha) fun y hy => ?_
  /-
    case intro.intro.intro.intro.intro
    α : Type u
    β : Type v
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    s : Set (Prod α α)
    e : α → β
    b : β
    he₁ : IsUniformInducing e
    he₂ : IsDenseInducing e
    hs✝ : Membership.mem (uniformity α) s
    U : Set (Prod β β)
    hs : HasSubset.Subset (Set.preimage (Prod.map e e) U) s
    hU : Membership.mem (uniformity β) U
    hUo : IsOpen U
    hsymm : SymmetricRel U
    a : α
    ha : Membership.mem (UniformSpace.ball b U) (e a)
    ho : IsOpen (UniformSpace.ball (e a) U)
    y : β
    hy : Membership.mem (UniformSpace.ball (e a) U) y
    ⊢ Membership.mem (closure (Set.image e (UniformSpace.ball a (Set.preimage (Pro …
  -/
  refine mem_closure_iff_nhds.2 fun V hV => ?_
  /-
    case intro.intro.intro.intro.intro
    α : Type u
    β : Type v
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    s : Set (Prod α α)
    e : α → β
    b : β
    he₁ : IsUniformInducing e
    he₂ : IsDenseInducing e
    hs✝ : Membership.mem (uniformity α) s
    U : Set (Prod β β)
    hs : HasSubset.Subset (Set.preimage (Prod.map e e) U) s
    hU : Membership.mem (uniformity β) U
    hUo : IsOpen U
    hsymm : SymmetricRel U
    a : α
    ha : Membership.mem (UniformSpace.ball b U) (e a)
    ho : IsOpen (UniformSpace.ball (e a) U)
    y : β
    hy : Membership.mem (UniformSpace.ball (e a) U) y
    V : Set β
    hV : Membership.mem (nhds y) V
    ⊢ (Inter.inter V (Set.image e (UniformSpace.ball a (Set.preimage (Prod.map e e …
  -/
  rcases he₂.dense.mem_nhds (inter_mem hV (ho.mem_nhds hy)) with ⟨x, hxV, hxU⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro
    α : Type u
    β : Type v
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    s : Set (Prod α α)
    e : α → β
    b : β
    he₁ : IsUniformInducing e
    he₂ : IsDenseInducing e
    hs✝ : Membership.mem (uniformity α) s
    U : Set (Prod β β)
    hs : HasSubset.Subset (Set.preimage (Prod.map e e) U) s
    hU : Membership.mem (uniformity β) U
    hUo : IsOpen U
    hsymm : SymmetricRel U
    a : α
    ha : Membership.mem (UniformSpace.ball b U) (e a)
    ho : IsOpen (UniformSpace.ball (e a) U)
    y : β
    hy : Membership.mem (UniformSpace.ball (e a) U) y
    V : Set β
    hV : Membership.mem (nhds y) V
    x : α
    hxV : Membership.mem V (e x)
    hxU : Membership.mem (UniformSpace.ball (e a) U) (e x)
    ⊢ (Inter.inter V (Set.image e (UniformSpace.ball a (Set.preimage (Prod.map e e …
  -/
  exact ⟨e x, hxV, mem_image_of_mem e hxU⟩
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-05")]
alias closure_image_mem_nhds_of_uniformInducing := closure_image_mem_nhds_of_isUniformInducing


theorem isUniformEmbedding_subtypeEmb (p : α → Prop) {e : α → β} (ue : IsUniformEmbedding e)
    (de : IsDenseEmbedding e) : IsUniformEmbedding (IsDenseEmbedding.subtypeEmb p e) :=
  { comap_uniformity := by
      simp [comap_comap, Function.comp_def, IsDenseEmbedding.subtypeEmb, uniformity_subtype,
        ue.comap_uniformity.symm]
    injective := (de.subtype p).injective }


@[deprecated (since := "2024-10-01")]
alias uniformEmbedding_subtypeEmb := isUniformEmbedding_subtypeEmb


theorem IsUniformEmbedding.prod {α' : Type*} {β' : Type*} [UniformSpace α'] [UniformSpace β']
    {e₁ : α → α'} {e₂ : β → β'} (h₁ : IsUniformEmbedding e₁) (h₂ : IsUniformEmbedding e₂) :
    IsUniformEmbedding fun p : α × β => (e₁ p.1, e₂ p.2) where
  toIsUniformInducing := h₁.isUniformInducing.prod h₂.isUniformInducing
  injective := h₁.injective.prodMap h₂.injective


@[deprecated (since := "2024-10-01")]
alias UniformEmbedding.prod := IsUniformEmbedding.prod


/-- A set is complete iff its image under a uniform inducing map is complete. -/
theorem isComplete_image_iff {m : α → β} {s : Set α} (hm : IsUniformInducing m) :
    IsComplete (m '' s) ↔ IsComplete s := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    m : α → β
    s : Set α
    hm : IsUniformInducing m
    ⊢ Iff (IsComplete (Set.image m s)) (IsComplete s)
  -/
  have fact1 : SurjOn (map m) (Iic <| 𝓟 s) (Iic <| 𝓟 <| m '' s) := surjOn_image .. |>.filter_map_Iic
  /-
    α : Type u
    β : Type v
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    m : α → β
    s : Set α
    hm : IsUniformInducing m
    fact1 : Set.SurjOn (Filter.map m) (Set.Iic (Filter.principal s)) (Set.Iic (Fil …
    ⊢ Iff (IsComplete (Set.image m s)) (IsComplete s)
  -/
  have fact2 : MapsTo (map m) (Iic <| 𝓟 s) (Iic <| 𝓟 <| m '' s) := mapsTo_image .. |>.filter_map_Iic
  simp_rw [IsComplete, imp.swap (a := Cauchy _), ← mem_Iic (b := 𝓟 _), fact1.forall fact2,
    hm.cauchy_map_iff, exists_mem_image, map_le_iff_le_comap, hm.isInducing.nhds_eq_comap]


/-- If `f : X → Y` is an `IsUniformInducing` map, the image `f '' s` of a set `s` is complete
  if and only if `s` is complete. -/
theorem IsUniformInducing.isComplete_iff {f : α → β} {s : Set α} (hf : IsUniformInducing f) :
    IsComplete (f '' s) ↔ IsComplete s := isComplete_image_iff hf


@[deprecated (since := "2024-10-05")]
alias UniformInducing.isComplete_iff := IsUniformInducing.isComplete_iff


/-- If `f : X → Y` is an `IsUniformEmbedding`, the image `f '' s` of a set `s` is complete
  if and only if `s` is complete. -/
theorem IsUniformEmbedding.isComplete_iff {f : α → β} {s : Set α} (hf : IsUniformEmbedding f) :
    IsComplete (f '' s) ↔ IsComplete s := hf.isUniformInducing.isComplete_iff


@[deprecated (since := "2024-10-01")]
alias UniformEmbedding.isComplete_iff := IsUniformEmbedding.isComplete_iff


/-- Sets of a subtype are complete iff their image under the coercion is complete. -/
theorem Subtype.isComplete_iff {p : α → Prop} {s : Set { x // p x }} :
    IsComplete s ↔ IsComplete ((↑) '' s : Set α) :=
  isUniformEmbedding_subtype_val.isComplete_iff.symm


alias ⟨isComplete_of_complete_image, _⟩ := isComplete_image_iff


theorem completeSpace_iff_isComplete_range {f : α → β} (hf : IsUniformInducing f) :
    CompleteSpace α ↔ IsComplete (range f) := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    f : α → β
    hf : IsUniformInducing f
    ⊢ Iff (CompleteSpace α) (IsComplete (Set.range f))
  -/
  rw [completeSpace_iff_isComplete_univ, ← isComplete_image_iff hf, image_univ]
  /-
    🎉 no goals
  -/


alias ⟨_, IsUniformInducing.completeSpace⟩ := completeSpace_iff_isComplete_range


@[deprecated (since := "2024-10-08")] alias UniformInducing.completeSpace :=
  IsUniformInducing.completeSpace


lemma IsUniformInducing.isComplete_range [CompleteSpace α] (hf : IsUniformInducing f) :
    IsComplete (range f) :=
  (completeSpace_iff_isComplete_range hf).1 ‹_›


@[deprecated (since := "2024-10-05")]
alias UniformInducing.isComplete_range := IsUniformInducing.isComplete_range


/-- If `f` is a surjective uniform inducing map,
then its domain is a complete space iff its codomain is a complete space.
See also `_root_.completeSpace_congr` for a version that assumes `f` to be an equivalence. -/
theorem IsUniformInducing.completeSpace_congr {f : α → β} (hf : IsUniformInducing f)
    (hsurj : f.Surjective) : CompleteSpace α ↔ CompleteSpace β := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    f : α → β
    hf : IsUniformInducing f
    hsurj : Function.Surjective f
    ⊢ Iff (CompleteSpace α) (CompleteSpace β)
  -/
  rw [completeSpace_iff_isComplete_range hf, hsurj.range_eq, completeSpace_iff_isComplete_univ]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-05")]
alias UniformInducing.completeSpace_congr := IsUniformInducing.completeSpace_congr


theorem SeparationQuotient.completeSpace_iff :
    CompleteSpace (SeparationQuotient α) ↔ CompleteSpace α :=
  .symm <| isUniformInducing_mk.completeSpace_congr surjective_mk


instance SeparationQuotient.instCompleteSpace [CompleteSpace α] :
    CompleteSpace (SeparationQuotient α) :=
  completeSpace_iff.2 ‹_›


/-- See also `IsUniformInducing.completeSpace_congr`
for a version that works for non-injective maps. -/
theorem completeSpace_congr {e : α ≃ β} (he : IsUniformEmbedding e) :
    CompleteSpace α ↔ CompleteSpace β :=
  he.completeSpace_congr e.surjective


theorem completeSpace_coe_iff_isComplete {s : Set α} : CompleteSpace s ↔ IsComplete s := by
  rw [completeSpace_iff_isComplete_range isUniformEmbedding_subtype_val.isUniformInducing,
    Subtype.range_coe]


alias ⟨_, IsComplete.completeSpace_coe⟩ := completeSpace_coe_iff_isComplete


theorem IsClosed.completeSpace_coe [CompleteSpace α] {s : Set α} (hs : IsClosed s) :
    CompleteSpace s :=
  hs.isComplete.completeSpace_coe


theorem completeSpace_ulift_iff : CompleteSpace (ULift α) ↔ CompleteSpace α :=
  IsUniformInducing.completeSpace_congr ⟨rfl⟩ ULift.down_surjective


/-- The lift of a complete space to another universe is still complete. -/
instance ULift.instCompleteSpace [CompleteSpace α] : CompleteSpace (ULift α) :=
  completeSpace_ulift_iff.2 ‹_›


theorem completeSpace_extension {m : β → α} (hm : IsUniformInducing m) (dense : DenseRange m)
    (h : ∀ f : Filter β, Cauchy f → ∃ x : α, map m f ≤ 𝓝 x) : CompleteSpace α :=
  ⟨fun {f : Filter α} (hf : Cauchy f) =>
    let p : Set (α × α) → Set α → Set α := fun s t => { y : α | ∃ x : α, x ∈ t ∧ (x, y) ∈ s }
    let g := (𝓤 α).lift fun s => f.lift' (p s)
    have mp₀ : Monotone p := fun _ _ h _ _ ⟨x, xs, xa⟩ => ⟨x, xs, h xa⟩
    have mp₁ : ∀ {s}, Monotone (p s) := fun h _ ⟨y, ya, yxs⟩ => ⟨y, h ya, yxs⟩
    have : f ≤ g := le_iInf₂ fun _ hs => le_iInf₂ fun _ ht =>
      le_principal_iff.mpr <| mem_of_superset ht fun x hx => ⟨x, hx, refl_mem_uniformity hs⟩
    have : NeBot g := hf.left.mono this
    have : NeBot (comap m g) :=
      comap_neBot fun _ ht =>
        let ⟨t', ht', ht_mem⟩ := (mem_lift_sets <| monotone_lift' monotone_const mp₀).mp ht
        let ⟨_, ht'', ht'_sub⟩ := (mem_lift'_sets mp₁).mp ht_mem
        let ⟨x, hx⟩ := hf.left.nonempty_of_mem ht''
        have h₀ : NeBot (𝓝[range m] x) := dense.nhdsWithin_neBot x
        have h₁ : { y | (x, y) ∈ t' } ∈ 𝓝[range m] x :=
          @mem_inf_of_left α (𝓝 x) (𝓟 (range m)) _ <| mem_nhds_left x ht'
        have h₂ : range m ∈ 𝓝[range m] x :=
          @mem_inf_of_right α (𝓝 x) (𝓟 (range m)) _ <| Subset.refl _
        have : { y | (x, y) ∈ t' } ∩ range m ∈ 𝓝[range m] x := @inter_mem α (𝓝[range m] x) _ _ h₁ h₂
        let ⟨_, xyt', b, b_eq⟩ := h₀.nonempty_of_mem this
        ⟨b, b_eq.symm ▸ ht'_sub ⟨x, hx, xyt'⟩⟩
    have : Cauchy g :=
      ⟨‹NeBot g›, fun _ hs =>
        let ⟨s₁, hs₁, comp_s₁⟩ := comp_mem_uniformity_sets hs
        let ⟨s₂, hs₂, comp_s₂⟩ := comp_mem_uniformity_sets hs₁
        let ⟨t, ht, (prod_t : t ×ˢ t ⊆ s₂)⟩ := mem_prod_same_iff.mp (hf.right hs₂)
        have hg₁ : p (preimage Prod.swap s₁) t ∈ g :=
          mem_lift (symm_le_uniformity hs₁) <| @mem_lift' α α f _ t ht
        have hg₂ : p s₂ t ∈ g := mem_lift hs₂ <| @mem_lift' α α f _ t ht
        have hg : p (Prod.swap ⁻¹' s₁) t ×ˢ p s₂ t ∈ g ×ˢ g := @prod_mem_prod α α _ _ g g hg₁ hg₂
        (g ×ˢ g).sets_of_superset hg fun ⟨_, _⟩ ⟨⟨c₁, c₁t, hc₁⟩, ⟨c₂, c₂t, hc₂⟩⟩ =>
          have : (c₁, c₂) ∈ t ×ˢ t := ⟨c₁t, c₂t⟩
          comp_s₁ <| prod_mk_mem_compRel hc₁ <| comp_s₂ <| prod_mk_mem_compRel (prod_t this) hc₂⟩
    have : Cauchy (Filter.comap m g) := ‹Cauchy g›.comap' (le_of_eq hm.comap_uniformity) ‹_›
    let ⟨x, (hx : map m (Filter.comap m g) ≤ 𝓝 x)⟩ := h _ this
    have : ClusterPt x (map m (Filter.comap m g)) :=
      (le_nhds_iff_adhp_of_cauchy (this.map hm.uniformContinuous)).mp hx
    have : ClusterPt x g := this.mono map_comap_le
    ⟨x,
      calc
                    /-
                      α : Type u
                      β : Type v
                      inst✝¹ : UniformSpace α
                      inst✝ : UniformSpace β
                      m : β → α
                      hm : IsUniformInducing m
                      dense : DenseRange m
                      h : ∀ (f : Filter β), Cauchy f → Exists fun x => LE.le (Filter.map m f) (nhds x)
                      f : Filter α
                      hf : Cauchy f
                      p : Set (Prod α α) → Set α → Set α := fun s t => setOf fun y => Exists fun x = …
                      g : Filter α := (uniformity α).lift fun s => f.lift' (p s)
                      mp₀ : Monotone p
                      mp₁ : ∀ {s : Set (Prod α α)}, Monotone (p s)
                      this✝⁵ : LE.le f g
                      this✝⁴ : g.NeBot
                      this✝³ : (Filter.comap m g).NeBot
                      this✝² : Cauchy g
                      this✝¹ : Cauchy (Filter.comap m g)
                      x : α
                      hx : LE.le (Filter.map m (Filter.comap m g)) (nhds x)
                      this✝ : ClusterPt x (Filter.map m (Filter.comap m g))
                      this : ClusterPt x g
                      ⊢ LE.le f g
                    -/
        f ≤ g := by assumption
                    /-
                      🎉 no goals
                    -/
        _ ≤ 𝓝 x := le_nhds_of_cauchy_adhp ‹Cauchy g› this
        ⟩⟩


lemma totallyBounded_image_iff {f : α → β} {s : Set α} (hf : IsUniformInducing f) :
    TotallyBounded (f '' s) ↔ TotallyBounded s := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    f : α → β
    s : Set α
    hf : IsUniformInducing f
    ⊢ Iff (TotallyBounded (Set.image f s)) (TotallyBounded s)
  -/
  refine ⟨fun hs ↦ ?_, fun h ↦ h.image hf.uniformContinuous⟩
  /-
    α : Type u
    β : Type v
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    f : α → β
    s : Set α
    hf : IsUniformInducing f
    hs : TotallyBounded (Set.image f s)
    ⊢ TotallyBounded s
  -/
  simp_rw [(hf.basis_uniformity (basis_sets _)).totallyBounded_iff]
  /-
    α : Type u
    β : Type v
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    f : α → β
    s : Set α
    hf : IsUniformInducing f
    hs : TotallyBounded (Set.image f s)
    ⊢ ∀ (i : Set (Prod β β)), Membership.mem (uniformity β) i → Exists fun t => An …
  -/
  intro t ht
  /-
    α : Type u
    β : Type v
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    f : α → β
    s : Set α
    hf : IsUniformInducing f
    hs : TotallyBounded (Set.image f s)
    t : Set (Prod β β)
    ht : Membership.mem (uniformity β) t
    ⊢ Exists fun t_1 => And t_1.Finite (HasSubset.Subset s (Set.iUnion fun y => Se …
  -/
  rcases exists_subset_image_finite_and.1 (hs.exists_subset ht) with ⟨u, -, hfin, h⟩
  /-
    case intro.intro.intro
    α : Type u
    β : Type v
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    f : α → β
    s : Set α
    hf : IsUniformInducing f
    hs : TotallyBounded (Set.image f s)
    t : Set (Prod β β)
    ht : Membership.mem (uniformity β) t
    u : Set α
    hfin : u.Finite
    h : HasSubset.Subset (Set.image f s) (Set.iUnion fun y => Set.iUnion fun h =>  …
    ⊢ Exists fun t_1 => And t_1.Finite (HasSubset.Subset s (Set.iUnion fun y => Se …
  -/
  use u, hfin
  /-
    case right
    α : Type u
    β : Type v
    inst✝¹ : UniformSpace α
    inst✝ : UniformSpace β
    f : α → β
    s : Set α
    hf : IsUniformInducing f
    hs : TotallyBounded (Set.image f s)
    t : Set (Prod β β)
    ht : Membership.mem (uniformity β) t
    u : Set α
    hfin : u.Finite
    h : HasSubset.Subset (Set.image f s) (Set.iUnion fun y => Set.iUnion fun h =>  …
    ⊢ HasSubset.Subset s (Set.iUnion fun y => Set.iUnion fun h => setOf fun x => M …
  -/
  rwa [biUnion_image, image_subset_iff, preimage_iUnion₂] at h
  /-
    🎉 no goals
  -/


theorem totallyBounded_preimage {f : α → β} {s : Set β} (hf : IsUniformInducing f)
    (hs : TotallyBounded s) : TotallyBounded (f ⁻¹' s) :=
  (totallyBounded_image_iff hf).1 <| hs.subset <| image_preimage_subset ..


instance CompleteSpace.sum [CompleteSpace α] [CompleteSpace β] : CompleteSpace (α ⊕ β) := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    inst✝⁴ : UniformSpace α
    inst✝³ : UniformSpace β
    inst✝² : UniformSpace γ
    f : α → β
    inst✝¹ : CompleteSpace α
    inst✝ : CompleteSpace β
    ⊢ CompleteSpace (Sum α β)
  -/
  rw [completeSpace_iff_isComplete_univ, ← range_inl_union_range_inr]
  exact isUniformEmbedding_inl.isUniformInducing.isComplete_range.union
    isUniformEmbedding_inr.isUniformInducing.isComplete_range


theorem isUniformEmbedding_comap {α : Type*} {β : Type*} {f : α → β} [u : UniformSpace β]
    (hf : Function.Injective f) : @IsUniformEmbedding α β (UniformSpace.comap f u) u f :=
  @IsUniformEmbedding.mk _ _ (UniformSpace.comap f u) _ _
    (@IsUniformInducing.mk _ _ (UniformSpace.comap f u) _ _ rfl) hf


@[deprecated (since := "2024-10-01")]
alias uniformEmbedding_comap := isUniformEmbedding_comap


/-- Pull back a uniform space structure by an embedding, adjusting the new uniform structure to
make sure that its topology is defeq to the original one. -/
def Topology.IsEmbedding.comapUniformSpace {α β} [TopologicalSpace α] [u : UniformSpace β]
    (f : α → β) (h : IsEmbedding f) : UniformSpace α :=
  (u.comap f).replaceTopology h.eq_induced


@[deprecated (since := "2024-10-26")]
alias Embedding.comapUniformSpace := IsEmbedding.comapUniformSpace


theorem Embedding.to_isUniformEmbedding {α β} [TopologicalSpace α] [u : UniformSpace β] (f : α → β)
    (h : IsEmbedding f) : @IsUniformEmbedding α β (h.comapUniformSpace f) u f :=
  let _ := h.comapUniformSpace f
  { comap_uniformity := rfl
    injective := h.injective }


@[deprecated (since := "2024-10-01")]
alias Embedding.to_uniformEmbedding := Embedding.to_isUniformEmbedding


local notation "ψ" => IsDenseInducing.extend (IsUniformInducing.isDenseInducing h_e h_dense) f


include h_e h_dense h_f in
theorem uniformly_extend_exists [CompleteSpace γ] (a : α) : ∃ c, Tendsto f (comap e (𝓝 a)) (𝓝 c) :=
  let de := h_e.isDenseInducing h_dense
  have : Cauchy (𝓝 a) := cauchy_nhds
  have : Cauchy (comap e (𝓝 a)) :=
    this.comap' (le_of_eq h_e.comap_uniformity) (de.comap_nhds_neBot _)
  have : Cauchy (map f (comap e (𝓝 a))) := this.map h_f
  CompleteSpace.complete this


theorem uniform_extend_subtype [CompleteSpace γ] {p : α → Prop} {e : α → β} {f : α → γ} {b : β}
    {s : Set α} (hf : UniformContinuous fun x : Subtype p => f x.val) (he : IsUniformEmbedding e)
    (hd : ∀ x : β, x ∈ closure (range e)) (hb : closure (e '' s) ∈ 𝓝 b) (hs : IsClosed s)
    (hp : ∀ x ∈ s, p x) : ∃ c, Tendsto f (comap e (𝓝 b)) (𝓝 c) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝³ : UniformSpace α
    inst✝² : UniformSpace β
    inst✝¹ : UniformSpace γ
    inst✝ : CompleteSpace γ
    p : α → Prop
    e : α → β
    f : α → γ
    b : β
    s : Set α
    hf : UniformContinuous fun x => f ↑x
    he : IsUniformEmbedding e
    hd : ∀ (x : β), Membership.mem (closure (Set.range e)) x
    hb : Membership.mem (nhds b) (closure (Set.image e s))
    hs : IsClosed s
    hp : ∀ (x : α), Membership.mem s x → p x
    ⊢ Exists fun c => Filter.Tendsto f (Filter.comap e (nhds b)) (nhds c)
  -/
  have de : IsDenseEmbedding e := he.isDenseEmbedding hd
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝³ : UniformSpace α
    inst✝² : UniformSpace β
    inst✝¹ : UniformSpace γ
    inst✝ : CompleteSpace γ
    p : α → Prop
    e : α → β
    f : α → γ
    b : β
    s : Set α
    hf : UniformContinuous fun x => f ↑x
    he : IsUniformEmbedding e
    hd : ∀ (x : β), Membership.mem (closure (Set.range e)) x
    hb : Membership.mem (nhds b) (closure (Set.image e s))
    hs : IsClosed s
    hp : ∀ (x : α), Membership.mem s x → p x
    de : IsDenseEmbedding e
    ⊢ Exists fun c => Filter.Tendsto f (Filter.comap e (nhds b)) (nhds c)
  -/
  have de' : IsDenseEmbedding (IsDenseEmbedding.subtypeEmb p e) := de.subtype p
  have ue' : IsUniformEmbedding (IsDenseEmbedding.subtypeEmb p e) :=
    isUniformEmbedding_subtypeEmb _ he de
  have : b ∈ closure (e '' { x | p x }) :=
    (closure_mono <| monotone_image <| hp) (mem_of_mem_nhds hb)
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝³ : UniformSpace α
    inst✝² : UniformSpace β
    inst✝¹ : UniformSpace γ
    inst✝ : CompleteSpace γ
    p : α → Prop
    e : α → β
    f : α → γ
    b : β
    s : Set α
    hf : UniformContinuous fun x => f ↑x
    he : IsUniformEmbedding e
    hd : ∀ (x : β), Membership.mem (closure (Set.range e)) x
    hb : Membership.mem (nhds b) (closure (Set.image e s))
    hs : IsClosed s
    hp : ∀ (x : α), Membership.mem s x → p x
    de : IsDenseEmbedding e
    de' : IsDenseEmbedding (IsDenseEmbedding.subtypeEmb p e)
    ue' : IsUniformEmbedding (IsDenseEmbedding.subtypeEmb p e)
    this : Membership.mem (closure (Set.image e (setOf fun x => p x))) b
    ⊢ Exists fun c => Filter.Tendsto f (Filter.comap e (nhds b)) (nhds c)
  -/
  let ⟨c, hc⟩ := uniformly_extend_exists ue'.isUniformInducing de'.dense hf ⟨b, this⟩
  replace hc : Tendsto (f ∘ Subtype.val (p := p)) (((𝓝 b).comap e).comap Subtype.val) (𝓝 c) := by
    simpa only [nhds_subtype_eq_comap, comap_comap, IsDenseEmbedding.subtypeEmb_coe] using hc
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝³ : UniformSpace α
    inst✝² : UniformSpace β
    inst✝¹ : UniformSpace γ
    inst✝ : CompleteSpace γ
    p : α → Prop
    e : α → β
    f : α → γ
    b : β
    s : Set α
    hf : UniformContinuous fun x => f ↑x
    he : IsUniformEmbedding e
    hd : ∀ (x : β), Membership.mem (closure (Set.range e)) x
    hb : Membership.mem (nhds b) (closure (Set.image e s))
    hs : IsClosed s
    hp : ∀ (x : α), Membership.mem s x → p x
    de : IsDenseEmbedding e
    de' : IsDenseEmbedding (IsDenseEmbedding.subtypeEmb p e)
    ue' : IsUniformEmbedding (IsDenseEmbedding.subtypeEmb p e)
    this : Membership.mem (closure (Set.image e (setOf fun x => p x))) b
    c : γ
    hc : Filter.Tendsto (Function.comp f Subtype.val) (Filter.comap Subtype.val (F …
    ⊢ Exists fun c => Filter.Tendsto f (Filter.comap e (nhds b)) (nhds c)
  -/
  refine ⟨c, (tendsto_comap'_iff ?_).1 hc⟩
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝³ : UniformSpace α
    inst✝² : UniformSpace β
    inst✝¹ : UniformSpace γ
    inst✝ : CompleteSpace γ
    p : α → Prop
    e : α → β
    f : α → γ
    b : β
    s : Set α
    hf : UniformContinuous fun x => f ↑x
    he : IsUniformEmbedding e
    hd : ∀ (x : β), Membership.mem (closure (Set.range e)) x
    hb : Membership.mem (nhds b) (closure (Set.image e s))
    hs : IsClosed s
    hp : ∀ (x : α), Membership.mem s x → p x
    de : IsDenseEmbedding e
    de' : IsDenseEmbedding (IsDenseEmbedding.subtypeEmb p e)
    ue' : IsUniformEmbedding (IsDenseEmbedding.subtypeEmb p e)
    this : Membership.mem (closure (Set.image e (setOf fun x => p x))) b
    c : γ
    hc : Filter.Tendsto (Function.comp f Subtype.val) (Filter.comap Subtype.val (F …
    ⊢ Membership.mem (Filter.comap e (nhds b)) (Set.range Subtype.val)
  -/
  rw [Subtype.range_coe_subtype]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝³ : UniformSpace α
    inst✝² : UniformSpace β
    inst✝¹ : UniformSpace γ
    inst✝ : CompleteSpace γ
    p : α → Prop
    e : α → β
    f : α → γ
    b : β
    s : Set α
    hf : UniformContinuous fun x => f ↑x
    he : IsUniformEmbedding e
    hd : ∀ (x : β), Membership.mem (closure (Set.range e)) x
    hb : Membership.mem (nhds b) (closure (Set.image e s))
    hs : IsClosed s
    hp : ∀ (x : α), Membership.mem s x → p x
    de : IsDenseEmbedding e
    de' : IsDenseEmbedding (IsDenseEmbedding.subtypeEmb p e)
    ue' : IsUniformEmbedding (IsDenseEmbedding.subtypeEmb p e)
    this : Membership.mem (closure (Set.image e (setOf fun x => p x))) b
    c : γ
    hc : Filter.Tendsto (Function.comp f Subtype.val) (Filter.comap Subtype.val (F …
    ⊢ Membership.mem (Filter.comap e (nhds b)) (setOf fun x => p x)
  -/
  exact ⟨_, hb, by rwa [← de.isInducing.closure_eq_preimage_closure_image, hs.closure_eq]⟩
  /-
    🎉 no goals
  -/


include h_e h_f in
theorem uniformly_extend_spec [CompleteSpace γ] (a : α) : Tendsto f (comap e (𝓝 a)) (𝓝 (ψ a)) := by
  simpa only [IsDenseInducing.extend] using
    tendsto_nhds_limUnder (uniformly_extend_exists h_e ‹_› h_f _)


include h_f in
theorem uniformContinuous_uniformly_extend [CompleteSpace γ] : UniformContinuous ψ := fun d hd =>
  let ⟨s, hs, hs_comp⟩ := comp3_mem_uniformity hd
  have h_pnt : ∀ {a m}, m ∈ 𝓝 a → ∃ c ∈ f '' (e ⁻¹' m), (c, ψ a) ∈ s ∧ (ψ a, c) ∈ s :=
    fun {a m} hm =>
    have nb : NeBot (map f (comap e (𝓝 a))) :=
      ((h_e.isDenseInducing h_dense).comap_nhds_neBot _).map _
    have :
      f '' (e ⁻¹' m) ∩ ({ c | (c, ψ a) ∈ s } ∩ { c | (ψ a, c) ∈ s }) ∈ map f (comap e (𝓝 a)) :=
      inter_mem (image_mem_map <| preimage_mem_comap <| hm)
        (uniformly_extend_spec h_e h_dense h_f _
          (inter_mem (mem_nhds_right _ hs) (mem_nhds_left _ hs)))
    nb.nonempty_of_mem this
  have : (Prod.map f f) ⁻¹' s ∈ 𝓤 β := h_f hs
  have : (Prod.map f f) ⁻¹' s ∈ comap (Prod.map e e) (𝓤 α) := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝³ : UniformSpace α
      inst✝² : UniformSpace β
      inst✝¹ : UniformSpace γ
      e : β → α
      h_e : IsUniformInducing e
      h_dense : DenseRange e
      f : β → γ
      h_f : UniformContinuous f
      inst✝ : CompleteSpace γ
      d : Set (Prod γ γ)
      hd : Membership.mem (uniformity γ) d
      s : Set (Prod γ γ)
      hs : Membership.mem (uniformity γ) s
      hs_comp : HasSubset.Subset (compRel s (compRel s s)) d
      h_pnt : ∀ {a : α} {m : Set α}, Membership.mem (nhds a) m → Exists fun c => And …
      this : Membership.mem (uniformity β) (Set.preimage (Prod.map f f) s)
      ⊢ Membership.mem (Filter.comap (Prod.map e e) (uniformity α)) (Set.preimage (P …
    -/
    rwa [← h_e.comap_uniformity] at this
    /-
      🎉 no goals
    -/
  let ⟨t, ht, ts⟩ := this
  show (Prod.map ψ ψ) ⁻¹' d ∈ 𝓤 α from
    mem_of_superset (interior_mem_uniformity ht) fun ⟨x₁, x₂⟩ hx_t => by
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        inst✝³ : UniformSpace α
        inst✝² : UniformSpace β
        inst✝¹ : UniformSpace γ
        e : β → α
        h_e : IsUniformInducing e
        h_dense : DenseRange e
        f : β → γ
        h_f : UniformContinuous f
        inst✝ : CompleteSpace γ
        d : Set (Prod γ γ)
        hd : Membership.mem (uniformity γ) d
        s : Set (Prod γ γ)
        hs : Membership.mem (uniformity γ) s
        hs_comp : HasSubset.Subset (compRel s (compRel s s)) d
        h_pnt : ∀ {a : α} {m : Set α}, Membership.mem (nhds a) m → Exists fun c => And …
        this✝ : Membership.mem (uniformity β) (Set.preimage (Prod.map f f) s)
        this : Membership.mem (Filter.comap (Prod.map e e) (uniformity α)) (Set.preima …
        t : Set (Prod α α)
        ht : Membership.mem (uniformity α) t
        ts : HasSubset.Subset (Set.preimage (Prod.map e e) t) (Set.preimage (Prod.map  …
        x✝ : Prod α α
        x₁ x₂ : α
        hx_t : Membership.mem (interior t) { fst := x₁, snd := x₂ }
        ⊢ Membership.mem (Set.preimage (Prod.map (⋯.extend f) (⋯.extend f)) d) { fst : …
      -/
      have : interior t ∈ 𝓝 (x₁, x₂) := isOpen_interior.mem_nhds hx_t
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        inst✝³ : UniformSpace α
        inst✝² : UniformSpace β
        inst✝¹ : UniformSpace γ
        e : β → α
        h_e : IsUniformInducing e
        h_dense : DenseRange e
        f : β → γ
        h_f : UniformContinuous f
        inst✝ : CompleteSpace γ
        d : Set (Prod γ γ)
        hd : Membership.mem (uniformity γ) d
        s : Set (Prod γ γ)
        hs : Membership.mem (uniformity γ) s
        hs_comp : HasSubset.Subset (compRel s (compRel s s)) d
        h_pnt : ∀ {a : α} {m : Set α}, Membership.mem (nhds a) m → Exists fun c => And …
        this✝¹ : Membership.mem (uniformity β) (Set.preimage (Prod.map f f) s)
        this✝ : Membership.mem (Filter.comap (Prod.map e e) (uniformity α)) (Set.preim …
        t : Set (Prod α α)
        ht : Membership.mem (uniformity α) t
        ts : HasSubset.Subset (Set.preimage (Prod.map e e) t) (Set.preimage (Prod.map  …
        x✝ : Prod α α
        x₁ x₂ : α
        hx_t : Membership.mem (interior t) { fst := x₁, snd := x₂ }
        this : Membership.mem (nhds { fst := x₁, snd := x₂ }) (interior t)
        ⊢ Membership.mem (Set.preimage (Prod.map (⋯.extend f) (⋯.extend f)) d) { fst : …
      -/
      let ⟨m₁, hm₁, m₂, hm₂, (hm : m₁ ×ˢ m₂ ⊆ interior t)⟩ := mem_nhds_prod_iff.mp this
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        inst✝³ : UniformSpace α
        inst✝² : UniformSpace β
        inst✝¹ : UniformSpace γ
        e : β → α
        h_e : IsUniformInducing e
        h_dense : DenseRange e
        f : β → γ
        h_f : UniformContinuous f
        inst✝ : CompleteSpace γ
        d : Set (Prod γ γ)
        hd : Membership.mem (uniformity γ) d
        s : Set (Prod γ γ)
        hs : Membership.mem (uniformity γ) s
        hs_comp : HasSubset.Subset (compRel s (compRel s s)) d
        h_pnt : ∀ {a : α} {m : Set α}, Membership.mem (nhds a) m → Exists fun c => And …
        this✝¹ : Membership.mem (uniformity β) (Set.preimage (Prod.map f f) s)
        this✝ : Membership.mem (Filter.comap (Prod.map e e) (uniformity α)) (Set.preim …
        t : Set (Prod α α)
        ht : Membership.mem (uniformity α) t
        ts : HasSubset.Subset (Set.preimage (Prod.map e e) t) (Set.preimage (Prod.map  …
        x✝ : Prod α α
        x₁ x₂ : α
        hx_t : Membership.mem (interior t) { fst := x₁, snd := x₂ }
        this : Membership.mem (nhds { fst := x₁, snd := x₂ }) (interior t)
        m₁ : Set α
        hm₁ : Membership.mem (nhds x₁) m₁
        m₂ : Set α
        hm₂ : Membership.mem (nhds x₂) m₂
        hm : HasSubset.Subset (SProd.sprod m₁ m₂) (interior t)
        ⊢ Membership.mem (Set.preimage (Prod.map (⋯.extend f) (⋯.extend f)) d) { fst : …
      -/
      obtain ⟨_, ⟨a, ha₁, rfl⟩, _, ha₂⟩ := h_pnt hm₁
      /-
        case intro.intro.intro.intro.intro
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        inst✝³ : UniformSpace α
        inst✝² : UniformSpace β
        inst✝¹ : UniformSpace γ
        e : β → α
        h_e : IsUniformInducing e
        h_dense : DenseRange e
        f : β → γ
        h_f : UniformContinuous f
        inst✝ : CompleteSpace γ
        d : Set (Prod γ γ)
        hd : Membership.mem (uniformity γ) d
        s : Set (Prod γ γ)
        hs : Membership.mem (uniformity γ) s
        hs_comp : HasSubset.Subset (compRel s (compRel s s)) d
        h_pnt : ∀ {a : α} {m : Set α}, Membership.mem (nhds a) m → Exists fun c => And …
        this✝¹ : Membership.mem (uniformity β) (Set.preimage (Prod.map f f) s)
        this✝ : Membership.mem (Filter.comap (Prod.map e e) (uniformity α)) (Set.preim …
        t : Set (Prod α α)
        ht : Membership.mem (uniformity α) t
        ts : HasSubset.Subset (Set.preimage (Prod.map e e) t) (Set.preimage (Prod.map  …
        x✝ : Prod α α
        x₁ x₂ : α
        hx_t : Membership.mem (interior t) { fst := x₁, snd := x₂ }
        this : Membership.mem (nhds { fst := x₁, snd := x₂ }) (interior t)
        m₁ : Set α
        hm₁ : Membership.mem (nhds x₁) m₁
        m₂ : Set α
        hm₂ : Membership.mem (nhds x₂) m₂
        hm : HasSubset.Subset (SProd.sprod m₁ m₂) (interior t)
        a : β
        ha₁ : Membership.mem (Set.preimage e m₁) a
        left✝ : Membership.mem s { fst := f a, snd := ⋯.extend f x₁ }
        ha₂ : Membership.mem s { fst := ⋯.extend f x₁, snd := f a }
        ⊢ Membership.mem (Set.preimage (Prod.map (⋯.extend f) (⋯.extend f)) d) { fst : …
      -/
      obtain ⟨_, ⟨b, hb₁, rfl⟩, hb₂, _⟩ := h_pnt hm₂
      have : Prod.map f f (a, b) ∈ s :=
        ts <| mem_preimage.2 <| interior_subset (@hm (e a, e b) ⟨ha₁, hb₁⟩)
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        inst✝³ : UniformSpace α
        inst✝² : UniformSpace β
        inst✝¹ : UniformSpace γ
        e : β → α
        h_e : IsUniformInducing e
        h_dense : DenseRange e
        f : β → γ
        h_f : UniformContinuous f
        inst✝ : CompleteSpace γ
        d : Set (Prod γ γ)
        hd : Membership.mem (uniformity γ) d
        s : Set (Prod γ γ)
        hs : Membership.mem (uniformity γ) s
        hs_comp : HasSubset.Subset (compRel s (compRel s s)) d
        h_pnt : ∀ {a : α} {m : Set α}, Membership.mem (nhds a) m → Exists fun c => And …
        this✝² : Membership.mem (uniformity β) (Set.preimage (Prod.map f f) s)
        this✝¹ : Membership.mem (Filter.comap (Prod.map e e) (uniformity α)) (Set.prei …
        t : Set (Prod α α)
        ht : Membership.mem (uniformity α) t
        ts : HasSubset.Subset (Set.preimage (Prod.map e e) t) (Set.preimage (Prod.map  …
        x✝ : Prod α α
        x₁ x₂ : α
        hx_t : Membership.mem (interior t) { fst := x₁, snd := x₂ }
        this✝ : Membership.mem (nhds { fst := x₁, snd := x₂ }) (interior t)
        m₁ : Set α
        hm₁ : Membership.mem (nhds x₁) m₁
        m₂ : Set α
        hm₂ : Membership.mem (nhds x₂) m₂
        hm : HasSubset.Subset (SProd.sprod m₁ m₂) (interior t)
        a : β
        ha₁ : Membership.mem (Set.preimage e m₁) a
        left✝ : Membership.mem s { fst := f a, snd := ⋯.extend f x₁ }
        ha₂ : Membership.mem s { fst := ⋯.extend f x₁, snd := f a }
        b : β
        hb₁ : Membership.mem (Set.preimage e m₂) b
        hb₂ : Membership.mem s { fst := f b, snd := ⋯.extend f x₂ }
        right✝ : Membership.mem s { fst := ⋯.extend f x₂, snd := f b }
        this : Membership.mem s (Prod.map f f { fst := a, snd := b })
        ⊢ Membership.mem (Set.preimage (Prod.map (⋯.extend f) (⋯.extend f)) d) { fst : …
      -/
      exact hs_comp ⟨f a, ha₂, ⟨f b, this, hb₂⟩⟩
      /-
        🎉 no goals
      -/


include h_f in
theorem uniformly_extend_of_ind (b : β) : ψ (e b) = f b :=
  IsDenseInducing.extend_eq_at _ h_f.continuous.continuousAt


theorem uniformly_extend_unique {g : α → γ} (hg : ∀ b, g (e b) = f b) (hc : Continuous g) : ψ = g :=
  IsDenseInducing.extend_unique _ hg hc


