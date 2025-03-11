/-- A set `s` is called totally disconnected if every subset `t ⊆ s` which is preconnected is
a subsingleton, ie either empty or a singleton. -/
def IsTotallyDisconnected (s : Set α) : Prop :=
  ∀ t, t ⊆ s → IsPreconnected t → t.Subsingleton


theorem isTotallyDisconnected_empty : IsTotallyDisconnected (∅ : Set α) := fun _ ht _ _ x_in _ _ =>
  (ht x_in).elim


theorem isTotallyDisconnected_singleton {x} : IsTotallyDisconnected ({x} : Set α) := fun _ ht _ =>
  subsingleton_singleton.anti ht


/-- A space is totally disconnected if all of its connected components are singletons. -/
@[mk_iff]
class TotallyDisconnectedSpace (α : Type u) [TopologicalSpace α] : Prop where
  /-- The universal set `Set.univ` in a totally disconnected space is totally disconnected. -/
  isTotallyDisconnected_univ : IsTotallyDisconnected (univ : Set α)


theorem IsPreconnected.subsingleton [TotallyDisconnectedSpace α] {s : Set α}
    (h : IsPreconnected s) : s.Subsingleton :=
  TotallyDisconnectedSpace.isTotallyDisconnected_univ s (subset_univ s) h


instance Pi.totallyDisconnectedSpace {α : Type*} {β : α → Type*}
    [∀ a, TopologicalSpace (β a)] [∀ a, TotallyDisconnectedSpace (β a)] :
    TotallyDisconnectedSpace (∀ a : α, β a) :=
  ⟨fun t _ h2 =>
    have this : ∀ a, IsPreconnected ((fun x : ∀ a, β a => x a) '' t) := fun a =>
      h2.image (fun x => x a) (continuous_apply a).continuousOn
    fun x x_in y y_in => funext fun a => (this a).subsingleton ⟨x, x_in, rfl⟩ ⟨y, y_in, rfl⟩⟩


instance Prod.totallyDisconnectedSpace [TopologicalSpace β] [TotallyDisconnectedSpace α]
    [TotallyDisconnectedSpace β] : TotallyDisconnectedSpace (α × β) :=
  ⟨fun t _ h2 =>
    have H1 : IsPreconnected (Prod.fst '' t) := h2.image Prod.fst continuous_fst.continuousOn
    have H2 : IsPreconnected (Prod.snd '' t) := h2.image Prod.snd continuous_snd.continuousOn
    fun x hx y hy =>
    Prod.ext (H1.subsingleton ⟨x, hx, rfl⟩ ⟨y, hy, rfl⟩)
      (H2.subsingleton ⟨x, hx, rfl⟩ ⟨y, hy, rfl⟩)⟩


instance [TopologicalSpace β] [TotallyDisconnectedSpace α] [TotallyDisconnectedSpace β] :
    TotallyDisconnectedSpace (α ⊕ β) := by
  /-
    α : Type u
    β : Type v
    ι : Type u_1
    π : ι → Type u_2
    inst✝³ : TopologicalSpace α
    s t u v : Set α
    inst✝² : TopologicalSpace β
    inst✝¹ : TotallyDisconnectedSpace α
    inst✝ : TotallyDisconnectedSpace β
    ⊢ TotallyDisconnectedSpace (Sum α β)
  -/
  refine ⟨fun s _ hs => ?_⟩
  /-
    α : Type u
    β : Type v
    ι : Type u_1
    π : ι → Type u_2
    inst✝³ : TopologicalSpace α
    s✝ t u v : Set α
    inst✝² : TopologicalSpace β
    inst✝¹ : TotallyDisconnectedSpace α
    inst✝ : TotallyDisconnectedSpace β
    s : Set (Sum α β)
    x✝ : HasSubset.Subset s Set.univ
    hs : IsPreconnected s
    ⊢ s.Subsingleton
  -/
  obtain ⟨t, ht, rfl⟩ | ⟨t, ht, rfl⟩ := Sum.isPreconnected_iff.1 hs
    /-
      case inl.intro.intro
      α : Type u
      β : Type v
      ι : Type u_1
      π : ι → Type u_2
      inst✝³ : TopologicalSpace α
      s t✝ u v : Set α
      inst✝² : TopologicalSpace β
      inst✝¹ : TotallyDisconnectedSpace α
      inst✝ : TotallyDisconnectedSpace β
      t : Set α
      ht : IsPreconnected t
      x✝ : HasSubset.Subset (Set.image Sum.inl t) Set.univ
      hs : IsPreconnected (Set.image Sum.inl t)
      ⊢ (Set.image Sum.inl t).Subsingleton
    -/
  · exact ht.subsingleton.image _
    /-
      🎉 no goals
    -/
    /-
      case inr.intro.intro
      α : Type u
      β : Type v
      ι : Type u_1
      π : ι → Type u_2
      inst✝³ : TopologicalSpace α
      s t✝ u v : Set α
      inst✝² : TopologicalSpace β
      inst✝¹ : TotallyDisconnectedSpace α
      inst✝ : TotallyDisconnectedSpace β
      t : Set β
      ht : IsPreconnected t
      x✝ : HasSubset.Subset (Set.image Sum.inr t) Set.univ
      hs : IsPreconnected (Set.image Sum.inr t)
      ⊢ (Set.image Sum.inr t).Subsingleton
    -/
  · exact ht.subsingleton.image _
    /-
      🎉 no goals
    -/


instance [∀ i, TopologicalSpace (π i)] [∀ i, TotallyDisconnectedSpace (π i)] :
    TotallyDisconnectedSpace (Σi, π i) := by
  /-
    α : Type u
    β : Type v
    ι : Type u_1
    π : ι → Type u_2
    inst✝² : TopologicalSpace α
    s t u v : Set α
    inst✝¹ : (i : ι) → TopologicalSpace (π i)
    inst✝ : ∀ (i : ι), TotallyDisconnectedSpace (π i)
    ⊢ TotallyDisconnectedSpace (Sigma fun i => π i)
  -/
  refine ⟨fun s _ hs => ?_⟩
  /-
    α : Type u
    β : Type v
    ι : Type u_1
    π : ι → Type u_2
    inst✝² : TopologicalSpace α
    s✝ t u v : Set α
    inst✝¹ : (i : ι) → TopologicalSpace (π i)
    inst✝ : ∀ (i : ι), TotallyDisconnectedSpace (π i)
    s : Set (Sigma fun i => π i)
    x✝ : HasSubset.Subset s Set.univ
    hs : IsPreconnected s
    ⊢ s.Subsingleton
  -/
  obtain rfl | h := s.eq_empty_or_nonempty
    /-
      case inl
      α : Type u
      β : Type v
      ι : Type u_1
      π : ι → Type u_2
      inst✝² : TopologicalSpace α
      s t u v : Set α
      inst✝¹ : (i : ι) → TopologicalSpace (π i)
      inst✝ : ∀ (i : ι), TotallyDisconnectedSpace (π i)
      x✝ : HasSubset.Subset EmptyCollection.emptyCollection Set.univ
      hs : IsPreconnected EmptyCollection.emptyCollection
      ⊢ EmptyCollection.emptyCollection.Subsingleton
    -/
  · exact subsingleton_empty
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u
      β : Type v
      ι : Type u_1
      π : ι → Type u_2
      inst✝² : TopologicalSpace α
      s✝ t u v : Set α
      inst✝¹ : (i : ι) → TopologicalSpace (π i)
      inst✝ : ∀ (i : ι), TotallyDisconnectedSpace (π i)
      s : Set (Sigma fun i => π i)
      x✝ : HasSubset.Subset s Set.univ
      hs : IsPreconnected s
      h : s.Nonempty
      ⊢ s.Subsingleton
    -/
  · obtain ⟨a, t, ht, rfl⟩ := Sigma.isConnected_iff.1 ⟨h, hs⟩
    /-
      case inr.intro.intro.intro
      α : Type u
      β : Type v
      ι : Type u_1
      π : ι → Type u_2
      inst✝² : TopologicalSpace α
      s t✝ u v : Set α
      inst✝¹ : (i : ι) → TopologicalSpace (π i)
      inst✝ : ∀ (i : ι), TotallyDisconnectedSpace (π i)
      a : ι
      t : Set (π a)
      ht : IsConnected t
      x✝ : HasSubset.Subset (Set.image (Sigma.mk a) t) Set.univ
      hs : IsPreconnected (Set.image (Sigma.mk a) t)
      h : (Set.image (Sigma.mk a) t).Nonempty
      ⊢ (Set.image (Sigma.mk a) t).Subsingleton
    -/
    exact ht.isPreconnected.subsingleton.image _
    /-
      🎉 no goals
    -/

-- Porting note: reformulated using `Pairwise`

/-- Let `X` be a topological space, and suppose that for all distinct `x,y ∈ X`, there
  is some clopen set `U` such that `x ∈ U` and `y ∉ U`. Then `X` is totally disconnected. -/
theorem isTotallyDisconnected_of_isClopen_set {X : Type*} [TopologicalSpace X]
    (hX : Pairwise fun x y => ∃ (U : Set X), IsClopen U ∧ x ∈ U ∧ y ∉ U) :
    IsTotallyDisconnected (Set.univ : Set X) := by
  /-
    X : Type u_3
    inst✝ : TopologicalSpace X
    hX : Pairwise fun x y => Exists fun U => And (IsClopen U) (And (Membership.mem …
    ⊢ IsTotallyDisconnected Set.univ
  -/
  rintro S - hS
  /-
    X : Type u_3
    inst✝ : TopologicalSpace X
    hX : Pairwise fun x y => Exists fun U => And (IsClopen U) (And (Membership.mem …
    S : Set X
    hS : IsPreconnected S
    ⊢ S.Subsingleton
  -/
  unfold Set.Subsingleton
  /-
    X : Type u_3
    inst✝ : TopologicalSpace X
    hX : Pairwise fun x y => Exists fun U => And (IsClopen U) (And (Membership.mem …
    S : Set X
    hS : IsPreconnected S
    ⊢ ∀ ⦃x : X⦄, Membership.mem S x → ∀ ⦃y : X⦄, Membership.mem S y → Eq x y
  -/
  by_contra! h_contra
  /-
    X : Type u_3
    inst✝ : TopologicalSpace X
    hX : Pairwise fun x y => Exists fun U => And (IsClopen U) (And (Membership.mem …
    S : Set X
    hS : IsPreconnected S
    h_contra : Exists fun ⦃x⦄ => And (Membership.mem S x) (Exists fun ⦃y⦄ => And ( …
    ⊢ False
  -/
  rcases h_contra with ⟨x, hx, y, hy, hxy⟩
  /-
    case intro.intro.intro.intro
    X : Type u_3
    inst✝ : TopologicalSpace X
    hX : Pairwise fun x y => Exists fun U => And (IsClopen U) (And (Membership.mem …
    S : Set X
    hS : IsPreconnected S
    x : X
    hx : Membership.mem S x
    y : X
    hy : Membership.mem S y
    hxy : Ne x y
    ⊢ False
  -/
  obtain ⟨U, hU, hxU, hyU⟩ := hX hxy
  specialize
    hS U Uᶜ hU.2 hU.compl.2 (fun a _ => em (a ∈ U)) ⟨x, hx, hxU⟩ ⟨y, hy, hyU⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro
    X : Type u_3
    inst✝ : TopologicalSpace X
    hX : Pairwise fun x y => Exists fun U => And (IsClopen U) (And (Membership.mem …
    S : Set X
    x : X
    hx : Membership.mem S x
    y : X
    hy : Membership.mem S y
    hxy : Ne x y
    U : Set X
    hU : IsClopen U
    hxU : Membership.mem U x
    hyU : Not (Membership.mem U y)
    hS : (Inter.inter S (Inter.inter U (HasCompl.compl U))).Nonempty
    ⊢ False
  -/
  rw [inter_compl_self, Set.inter_empty] at hS
  /-
    case intro.intro.intro.intro.intro.intro.intro
    X : Type u_3
    inst✝ : TopologicalSpace X
    hX : Pairwise fun x y => Exists fun U => And (IsClopen U) (And (Membership.mem …
    S : Set X
    x : X
    hx : Membership.mem S x
    y : X
    hy : Membership.mem S y
    hxy : Ne x y
    U : Set X
    hU : IsClopen U
    hxU : Membership.mem U x
    hyU : Not (Membership.mem U y)
    hS : EmptyCollection.emptyCollection.Nonempty
    ⊢ False
  -/
  exact Set.not_nonempty_empty hS
  /-
    🎉 no goals
  -/


/-- A space is totally disconnected iff its connected components are subsingletons. -/
theorem totallyDisconnectedSpace_iff_connectedComponent_subsingleton :
    TotallyDisconnectedSpace α ↔ ∀ x : α, (connectedComponent x).Subsingleton := by
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    ⊢ Iff (TotallyDisconnectedSpace α) (∀ (x : α), (connectedComponent x).Subsingl …
  -/
  constructor
    /-
      case mp
      α : Type u
      inst✝ : TopologicalSpace α
      ⊢ TotallyDisconnectedSpace α → ∀ (x : α), (connectedComponent x).Subsingleton
    -/
  · intro h x
    /-
      case mp
      α : Type u
      inst✝ : TopologicalSpace α
      h : TotallyDisconnectedSpace α
      x : α
      ⊢ (connectedComponent x).Subsingleton
    -/
    apply h.1
      /-
        case mp.a
        α : Type u
        inst✝ : TopologicalSpace α
        h : TotallyDisconnectedSpace α
        x : α
        ⊢ HasSubset.Subset (connectedComponent x) Set.univ
      -/
    · exact subset_univ _
      /-
        🎉 no goals
      -/
    /-
      case mp.a
      α : Type u
      inst✝ : TopologicalSpace α
      h : TotallyDisconnectedSpace α
      x : α
      ⊢ IsPreconnected (connectedComponent x)
    -/
    exact isPreconnected_connectedComponent
    /-
      🎉 no goals
    -/
  /-
    case mpr
    α : Type u
    inst✝ : TopologicalSpace α
    ⊢ (∀ (x : α), (connectedComponent x).Subsingleton) → TotallyDisconnectedSpace α
  -/
  intro h; constructor
  /-
    case mpr.isTotallyDisconnected_univ
    α : Type u
    inst✝ : TopologicalSpace α
    h : ∀ (x : α), (connectedComponent x).Subsingleton
    ⊢ IsTotallyDisconnected Set.univ
  -/
  intro s s_sub hs
  /-
    case mpr.isTotallyDisconnected_univ
    α : Type u
    inst✝ : TopologicalSpace α
    h : ∀ (x : α), (connectedComponent x).Subsingleton
    s : Set α
    s_sub : HasSubset.Subset s Set.univ
    hs : IsPreconnected s
    ⊢ s.Subsingleton
  -/
  rcases eq_empty_or_nonempty s with (rfl | ⟨x, x_in⟩)
    /-
      case mpr.isTotallyDisconnected_univ.inl
      α : Type u
      inst✝ : TopologicalSpace α
      h : ∀ (x : α), (connectedComponent x).Subsingleton
      s_sub : HasSubset.Subset EmptyCollection.emptyCollection Set.univ
      hs : IsPreconnected EmptyCollection.emptyCollection
      ⊢ EmptyCollection.emptyCollection.Subsingleton
    -/
  · exact subsingleton_empty
    /-
      🎉 no goals
    -/
    /-
      case mpr.isTotallyDisconnected_univ.inr.intro
      α : Type u
      inst✝ : TopologicalSpace α
      h : ∀ (x : α), (connectedComponent x).Subsingleton
      s : Set α
      s_sub : HasSubset.Subset s Set.univ
      hs : IsPreconnected s
      x : α
      x_in : Membership.mem s x
      ⊢ s.Subsingleton
    -/
  · exact (h x).anti (hs.subset_connectedComponent x_in)
    /-
      🎉 no goals
    -/


/-- A space is totally disconnected iff its connected components are singletons. -/
theorem totallyDisconnectedSpace_iff_connectedComponent_singleton :
    TotallyDisconnectedSpace α ↔ ∀ x : α, connectedComponent x = {x} := by
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    ⊢ Iff (TotallyDisconnectedSpace α) (∀ (x : α), Eq (connectedComponent x) (Sing …
  -/
  rw [totallyDisconnectedSpace_iff_connectedComponent_subsingleton]
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    ⊢ Iff (∀ (x : α), (connectedComponent x).Subsingleton) (∀ (x : α), Eq (connect …
  -/
  refine forall_congr' fun x => ?_
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    x : α
    ⊢ Iff (connectedComponent x).Subsingleton (Eq (connectedComponent x) (Singleto …
  -/
  rw [subsingleton_iff_singleton]
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    x : α
    ⊢ Membership.mem (connectedComponent x) x
  -/
  exact mem_connectedComponent
  /-
    🎉 no goals
  -/


@[simp] theorem connectedComponent_eq_singleton [TotallyDisconnectedSpace α] (x : α) :
    connectedComponent x = {x} :=
  totallyDisconnectedSpace_iff_connectedComponent_singleton.1 ‹_› x


/-- The image of a connected component in a totally disconnected space is a singleton. -/
@[simp]
theorem Continuous.image_connectedComponent_eq_singleton {β : Type*} [TopologicalSpace β]
    [TotallyDisconnectedSpace β] {f : α → β} (h : Continuous f) (a : α) :
    f '' connectedComponent a = {f a} :=
  (Set.subsingleton_iff_singleton <| mem_image_of_mem f mem_connectedComponent).mp
    (isPreconnected_connectedComponent.image f h.continuousOn).subsingleton


theorem isTotallyDisconnected_of_totallyDisconnectedSpace [TotallyDisconnectedSpace α] (s : Set α) :
    IsTotallyDisconnected s := fun t _ ht =>
  TotallyDisconnectedSpace.isTotallyDisconnected_univ _ t.subset_univ ht


theorem isTotallyDisconnected_of_image [TopologicalSpace β] {f : α → β} (hf : ContinuousOn f s)
    (hf' : Injective f) (h : IsTotallyDisconnected (f '' s)) : IsTotallyDisconnected s :=
  fun _t hts ht _x x_in _y y_in =>
  hf' <|
    h _ (image_subset f hts) (ht.image f <| hf.mono hts) (mem_image_of_mem f x_in)
      (mem_image_of_mem f y_in)


lemma Topology.IsEmbedding.isTotallyDisconnected [TopologicalSpace β] {f : α → β} {s : Set α}
    (hf : IsEmbedding f) (h : IsTotallyDisconnected (f '' s)) : IsTotallyDisconnected s :=
  isTotallyDisconnected_of_image hf.continuous.continuousOn hf.injective h


@[deprecated (since := "2024-10-26")]
alias Embedding.isTotallyDisconnected := IsEmbedding.isTotallyDisconnected


lemma Topology.IsEmbedding.isTotallyDisconnected_image [TopologicalSpace β] {f : α → β} {s : Set α}
    (hf : IsEmbedding f) : IsTotallyDisconnected (f '' s) ↔ IsTotallyDisconnected s := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    s : Set α
    hf : Topology.IsEmbedding f
    ⊢ Iff (IsTotallyDisconnected (Set.image f s)) (IsTotallyDisconnected s)
  -/
  refine ⟨hf.isTotallyDisconnected, fun hs u hus hu ↦ ?_⟩
  obtain ⟨v, hvs, rfl⟩ : ∃ v, v ⊆ s ∧ f '' v = u :=
    ⟨f ⁻¹' u ∩ s, inter_subset_right, by rwa [image_preimage_inter, inter_eq_left]⟩
  /-
    case intro.intro
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    s : Set α
    hf : Topology.IsEmbedding f
    hs : IsTotallyDisconnected s
    v : Set α
    hvs : HasSubset.Subset v s
    hus : HasSubset.Subset (Set.image f v) (Set.image f s)
    hu : IsPreconnected (Set.image f v)
    ⊢ (Set.image f v).Subsingleton
  -/
  rw [hf.isInducing.isPreconnected_image] at hu
  /-
    case intro.intro
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    s : Set α
    hf : Topology.IsEmbedding f
    hs : IsTotallyDisconnected s
    v : Set α
    hvs : HasSubset.Subset v s
    hus : HasSubset.Subset (Set.image f v) (Set.image f s)
    hu : IsPreconnected v
    ⊢ (Set.image f v).Subsingleton
  -/
  exact (hs v hvs hu).image _
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-26")]
alias Embedding.isTotallyDisconnected_image := IsEmbedding.isTotallyDisconnected_image


lemma Topology.IsEmbedding.isTotallyDisconnected_range [TopologicalSpace β] {f : α → β}
    (hf : IsEmbedding f) : IsTotallyDisconnected (range f) ↔ TotallyDisconnectedSpace α := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    f : α → β
    hf : Topology.IsEmbedding f
    ⊢ Iff (IsTotallyDisconnected (Set.range f)) (TotallyDisconnectedSpace α)
  -/
  rw [totallyDisconnectedSpace_iff, ← image_univ, hf.isTotallyDisconnected_image]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-26")]
alias Embedding.isTotallyDisconnected_range := IsEmbedding.isTotallyDisconnected_range


lemma totallyDisconnectedSpace_subtype_iff {s : Set α} :
    TotallyDisconnectedSpace s ↔ IsTotallyDisconnected s := by
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    s : Set α
    ⊢ Iff (TotallyDisconnectedSpace ↑s) (IsTotallyDisconnected s)
  -/
  rw [← IsEmbedding.subtypeVal.isTotallyDisconnected_range, Subtype.range_val]
  /-
    🎉 no goals
  -/


instance Subtype.totallyDisconnectedSpace {α : Type*} {p : α → Prop} [TopologicalSpace α]
    [TotallyDisconnectedSpace α] : TotallyDisconnectedSpace (Subtype p) :=
  totallyDisconnectedSpace_subtype_iff.2 (isTotallyDisconnected_of_totallyDisconnectedSpace _)


/-- A set `s` is called totally separated if any two points of this set can be separated
by two disjoint open sets covering `s`. -/
def IsTotallySeparated (s : Set α) : Prop :=
  Set.Pairwise s fun x y =>
  ∃ u v : Set α, IsOpen u ∧ IsOpen v ∧ x ∈ u ∧ y ∈ v ∧ s ⊆ u ∪ v ∧ Disjoint u v


theorem isTotallySeparated_empty : IsTotallySeparated (∅ : Set α) := fun _ => False.elim


theorem isTotallySeparated_singleton {x} : IsTotallySeparated ({x} : Set α) := fun _ hp _ hq hpq =>
  (hpq <| (eq_of_mem_singleton hp).symm ▸ (eq_of_mem_singleton hq).symm).elim


theorem isTotallyDisconnected_of_isTotallySeparated {s : Set α} (H : IsTotallySeparated s) :
    IsTotallyDisconnected s := by
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    s : Set α
    H : IsTotallySeparated s
    ⊢ IsTotallyDisconnected s
  -/
  intro t hts ht x x_in y y_in
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    s : Set α
    H : IsTotallySeparated s
    t : Set α
    hts : HasSubset.Subset t s
    ht : IsPreconnected t
    x : α
    x_in : Membership.mem t x
    y : α
    y_in : Membership.mem t y
    ⊢ Eq x y
  -/
  by_contra h
  obtain
    ⟨u : Set α, v : Set α, hu : IsOpen u, hv : IsOpen v, hxu : x ∈ u, hyv : y ∈ v, hs : s ⊆ u ∪ v,
      huv⟩ :=
    H (hts x_in) (hts y_in) h
  /-
    case intro.intro.intro.intro.intro.intro.intro
    α : Type u
    inst✝ : TopologicalSpace α
    s : Set α
    H : IsTotallySeparated s
    t : Set α
    hts : HasSubset.Subset t s
    ht : IsPreconnected t
    x : α
    x_in : Membership.mem t x
    y : α
    y_in : Membership.mem t y
    h : Not (Eq x y)
    u v : Set α
    hu : IsOpen u
    hv : IsOpen v
    hxu : Membership.mem u x
    hyv : Membership.mem v y
    hs : HasSubset.Subset s (Union.union u v)
    huv : Disjoint u v
    ⊢ False
  -/
  refine (ht _ _ hu hv (hts.trans hs) ⟨x, x_in, hxu⟩ ⟨y, y_in, hyv⟩).ne_empty ?_
  /-
    case intro.intro.intro.intro.intro.intro.intro
    α : Type u
    inst✝ : TopologicalSpace α
    s : Set α
    H : IsTotallySeparated s
    t : Set α
    hts : HasSubset.Subset t s
    ht : IsPreconnected t
    x : α
    x_in : Membership.mem t x
    y : α
    y_in : Membership.mem t y
    h : Not (Eq x y)
    u v : Set α
    hu : IsOpen u
    hv : IsOpen v
    hxu : Membership.mem u x
    hyv : Membership.mem v y
    hs : HasSubset.Subset s (Union.union u v)
    huv : Disjoint u v
    ⊢ Eq (Inter.inter t (Inter.inter u v)) EmptyCollection.emptyCollection
  -/
  rw [huv.inter_eq, inter_empty]
  /-
    🎉 no goals
  -/


alias IsTotallySeparated.isTotallyDisconnected := isTotallyDisconnected_of_isTotallySeparated


/-- A space is totally separated if any two points can be separated by two disjoint open sets
covering the whole space. -/
@[mk_iff] class TotallySeparatedSpace (α : Type u) [TopologicalSpace α] : Prop where
  /-- The universal set `Set.univ` in a totally separated space is totally separated. -/
  isTotallySeparated_univ : IsTotallySeparated (univ : Set α)

-- see Note [lower instance priority]

instance (priority := 100) TotallySeparatedSpace.totallyDisconnectedSpace (α : Type u)
    [TopologicalSpace α] [TotallySeparatedSpace α] : TotallyDisconnectedSpace α :=
  ⟨TotallySeparatedSpace.isTotallySeparated_univ.isTotallyDisconnected⟩

-- see Note [lower instance priority]

instance (priority := 100) TotallySeparatedSpace.of_discrete (α : Type*) [TopologicalSpace α]
    [DiscreteTopology α] : TotallySeparatedSpace α :=
  ⟨fun _ _ b _ h => ⟨{b}ᶜ, {b}, isOpen_discrete _, isOpen_discrete _, h, rfl,
    (compl_union_self _).symm.subset, disjoint_compl_left⟩⟩


theorem totallySeparatedSpace_iff_exists_isClopen {α : Type*} [TopologicalSpace α] :
    TotallySeparatedSpace α ↔ ∀ x y : α, x ≠ y → ∃ U : Set α, IsClopen U ∧ x ∈ U ∧ y ∈ Uᶜ := by
  /-
    α : Type u_3
    inst✝ : TopologicalSpace α
    ⊢ Iff (TotallySeparatedSpace α) (∀ (x y : α), Ne x y → Exists fun U => And (Is …
  -/
  simp only [totallySeparatedSpace_iff, IsTotallySeparated, Set.Pairwise, mem_univ, true_implies]
  refine forall₃_congr fun x y _ ↦
    ⟨fun ⟨U, V, hU, hV, Ux, Vy, f, disj⟩ ↦ ?_, fun ⟨U, hU, Ux, Ucy⟩ ↦ ?_⟩
  · exact ⟨U, isClopen_of_disjoint_cover_open f hU hV disj,
      Ux, fun Uy ↦ Set.disjoint_iff.mp disj ⟨Uy, Vy⟩⟩
    /-
      case refine_2
      α : Type u_3
      inst✝ : TopologicalSpace α
      x y : α
      x✝¹ : Ne x y
      x✝ : Exists fun U => And (IsClopen U) (And (Membership.mem U x) (Membership.me …
      U : Set α
      hU : IsClopen U
      Ux : Membership.mem U x
      Ucy : Membership.mem (HasCompl.compl U) y
      ⊢ Exists fun u => Exists fun v => And (IsOpen u) (And (IsOpen v) (And (Members …
    -/
  · exact ⟨U, Uᶜ, hU.2, hU.compl.2, Ux, Ucy, (Set.union_compl_self U).ge, disjoint_compl_right⟩
    /-
      🎉 no goals
    -/


theorem exists_isClopen_of_totally_separated {α : Type*} [TopologicalSpace α]
    [TotallySeparatedSpace α] {x y : α} (hxy : x ≠ y) :
    ∃ U : Set α, IsClopen U ∧ x ∈ U ∧ y ∈ Uᶜ :=
  totallySeparatedSpace_iff_exists_isClopen.mp ‹_› _ _ hxy


theorem Continuous.image_eq_of_connectedComponent_eq (h : Continuous f) (a b : α)
    (hab : connectedComponent a = connectedComponent b) : f a = f b :=
  singleton_eq_singleton_iff.1 <|
    h.image_connectedComponent_eq_singleton a ▸
      h.image_connectedComponent_eq_singleton b ▸ hab ▸ rfl


/--
The lift to `connectedComponents α` of a continuous map from `α` to a totally disconnected space
-/
def Continuous.connectedComponentsLift (h : Continuous f) : ConnectedComponents α → β := fun x =>
  Quotient.liftOn' x f h.image_eq_of_connectedComponent_eq


@[continuity]
theorem Continuous.connectedComponentsLift_continuous (h : Continuous f) :
    Continuous h.connectedComponentsLift :=
                           /-
                             α : Type u
                             β : Type v
                             inst✝² : TopologicalSpace α
                             inst✝¹ : TopologicalSpace β
                             inst✝ : TotallyDisconnectedSpace β
                             f : α → β
                             h : Continuous f
                             ⊢ ∀ (a b : α), (connectedComponentSetoid α) a b → Eq (f a) (f b)
                           -/
  h.quotient_liftOn' <| by convert h.image_eq_of_connectedComponent_eq
                           /-
                             🎉 no goals
                           -/


@[simp]
theorem Continuous.connectedComponentsLift_apply_coe (h : Continuous f) (x : α) :
    h.connectedComponentsLift x = f x :=
  rfl


@[simp]
theorem Continuous.connectedComponentsLift_comp_coe (h : Continuous f) :
    h.connectedComponentsLift ∘ (↑) = f :=
  rfl


theorem connectedComponents_lift_unique' {β : Sort*} {g₁ g₂ : ConnectedComponents α → β}
    (hg : g₁ ∘ ((↑) : α → ConnectedComponents α) = g₂ ∘ (↑)) : g₁ = g₂ :=
  ConnectedComponents.surjective_coe.injective_comp_right hg


theorem Continuous.connectedComponentsLift_unique (h : Continuous f) (g : ConnectedComponents α → β)
    (hg : g ∘ (↑) = f) : g = h.connectedComponentsLift :=
  connectedComponents_lift_unique' <| hg.trans h.connectedComponentsLift_comp_coe.symm


instance ConnectedComponents.totallyDisconnectedSpace :
    TotallyDisconnectedSpace (ConnectedComponents α) := by
  /-
    α : Type u
    β : Type v
    ι : Type u_1
    π : ι → Type u_2
    inst✝² : TopologicalSpace α
    s t u v : Set α
    inst✝¹ : TopologicalSpace β
    inst✝ : TotallyDisconnectedSpace β
    f : α → β
    ⊢ TotallyDisconnectedSpace (ConnectedComponents α)
  -/
  rw [totallyDisconnectedSpace_iff_connectedComponent_singleton]
  /-
    α : Type u
    β : Type v
    ι : Type u_1
    π : ι → Type u_2
    inst✝² : TopologicalSpace α
    s t u v : Set α
    inst✝¹ : TopologicalSpace β
    inst✝ : TotallyDisconnectedSpace β
    f : α → β
    ⊢ ∀ (x : ConnectedComponents α), Eq (connectedComponent x) (Singleton.singleto …
  -/
  refine ConnectedComponents.surjective_coe.forall.2 fun x => ?_
  rw [← ConnectedComponents.isQuotientMap_coe.image_connectedComponent, ←
    connectedComponents_preimage_singleton, image_preimage_eq _ ConnectedComponents.surjective_coe]
  /-
    case h_fibers
    α : Type u
    β : Type v
    ι : Type u_1
    π : ι → Type u_2
    inst✝² : TopologicalSpace α
    s t u v : Set α
    inst✝¹ : TopologicalSpace β
    inst✝ : TotallyDisconnectedSpace β
    f : α → β
    x : α
    ⊢ ∀ (y : ConnectedComponents α), IsConnected (Set.preimage ConnectedComponents …
  -/
  refine ConnectedComponents.surjective_coe.forall.2 fun y => ?_
  /-
    case h_fibers
    α : Type u
    β : Type v
    ι : Type u_1
    π : ι → Type u_2
    inst✝² : TopologicalSpace α
    s t u v : Set α
    inst✝¹ : TopologicalSpace β
    inst✝ : TotallyDisconnectedSpace β
    f : α → β
    x y : α
    ⊢ IsConnected (Set.preimage ConnectedComponents.mk (Singleton.singleton (Conne …
  -/
  rw [connectedComponents_preimage_singleton]
  /-
    case h_fibers
    α : Type u
    β : Type v
    ι : Type u_1
    π : ι → Type u_2
    inst✝² : TopologicalSpace α
    s t u v : Set α
    inst✝¹ : TopologicalSpace β
    inst✝ : TotallyDisconnectedSpace β
    f : α → β
    x y : α
    ⊢ IsConnected (connectedComponent y)
  -/
  exact isConnected_connectedComponent
  /-
    🎉 no goals
  -/


/-- Functoriality of `connectedComponents` -/
def Continuous.connectedComponentsMap {β : Type*} [TopologicalSpace β] {f : α → β}
    (h : Continuous f) : ConnectedComponents α → ConnectedComponents β :=
  Continuous.connectedComponentsLift (ConnectedComponents.continuous_coe.comp h)


theorem Continuous.connectedComponentsMap_continuous {β : Type*} [TopologicalSpace β] {f : α → β}
    (h : Continuous f) : Continuous h.connectedComponentsMap :=
  Continuous.connectedComponentsLift_continuous (ConnectedComponents.continuous_coe.comp h)


/-- A preconnected set `s` has the property that every map to a
discrete space that is continuous on `s` is constant on `s` -/
theorem IsPreconnected.constant {Y : Type*} [TopologicalSpace Y] [DiscreteTopology Y] {s : Set α}
    (hs : IsPreconnected s) {f : α → Y} (hf : ContinuousOn f s) {x y : α} (hx : x ∈ s)
    (hy : y ∈ s) : f x = f y :=
  (hs.image f hf).subsingleton (mem_image_of_mem f hx) (mem_image_of_mem f hy)


/-- A `PreconnectedSpace` version of `isPreconnected.constant` -/
theorem PreconnectedSpace.constant {Y : Type*} [TopologicalSpace Y] [DiscreteTopology Y]
    (hp : PreconnectedSpace α) {f : α → Y} (hf : Continuous f) {x y : α} : f x = f y :=
  IsPreconnected.constant hp.isPreconnected_univ (Continuous.continuousOn hf) trivial trivial


/-- Refinement of `IsPreconnected.constant` only assuming the map factors through a
discrete subset of the target. -/
theorem IsPreconnected.constant_of_mapsTo {S : Set α} (hS : IsPreconnected S)
    {β} [TopologicalSpace β] {T : Set β} [DiscreteTopology T] {f : α → β} (hc : ContinuousOn f S)
    (hTm : MapsTo f S T) {x y : α} (hx : x ∈ S) (hy : y ∈ S) : f x = f y := by
  /-
    α : Type u
    inst✝² : TopologicalSpace α
    S : Set α
    hS : IsPreconnected S
    β : Type u_3
    inst✝¹ : TopologicalSpace β
    T : Set β
    inst✝ : DiscreteTopology ↑T
    f : α → β
    hc : ContinuousOn f S
    hTm : Set.MapsTo f S T
    x y : α
    hx : Membership.mem S x
    hy : Membership.mem S y
    ⊢ Eq (f x) (f y)
  -/
  let F : S → T := hTm.restrict f S T
  /-
    α : Type u
    inst✝² : TopologicalSpace α
    S : Set α
    hS : IsPreconnected S
    β : Type u_3
    inst✝¹ : TopologicalSpace β
    T : Set β
    inst✝ : DiscreteTopology ↑T
    f : α → β
    hc : ContinuousOn f S
    hTm : Set.MapsTo f S T
    x y : α
    hx : Membership.mem S x
    hy : Membership.mem S y
    F : ↑S → ↑T := Set.MapsTo.restrict f S T hTm
    ⊢ Eq (f x) (f y)
  -/
  suffices F ⟨x, hx⟩ = F ⟨y, hy⟩ by rwa [← Subtype.coe_inj] at this
  /-
    α : Type u
    inst✝² : TopologicalSpace α
    S : Set α
    hS : IsPreconnected S
    β : Type u_3
    inst✝¹ : TopologicalSpace β
    T : Set β
    inst✝ : DiscreteTopology ↑T
    f : α → β
    hc : ContinuousOn f S
    hTm : Set.MapsTo f S T
    x y : α
    hx : Membership.mem S x
    hy : Membership.mem S y
    F : ↑S → ↑T := Set.MapsTo.restrict f S T hTm
    ⊢ Eq (F ⟨x, hx⟩) (F ⟨y, hy⟩)
  -/
  exact (isPreconnected_iff_preconnectedSpace.mp hS).constant (hc.restrict_mapsTo _)
  /-
    🎉 no goals
  -/


/-- A version of `IsPreconnected.constant_of_mapsTo` that assumes that the codomain is nonempty and
proves that `f` is equal to `const α y` on `S` for some `y ∈ T`. -/
theorem IsPreconnected.eqOn_const_of_mapsTo {S : Set α} (hS : IsPreconnected S)
    {β} [TopologicalSpace β] {T : Set β} [DiscreteTopology T] {f : α → β} (hc : ContinuousOn f S)
    (hTm : MapsTo f S T) (hne : T.Nonempty) : ∃ y ∈ T, EqOn f (const α y) S := by
  /-
    α : Type u
    inst✝² : TopologicalSpace α
    S : Set α
    hS : IsPreconnected S
    β : Type u_3
    inst✝¹ : TopologicalSpace β
    T : Set β
    inst✝ : DiscreteTopology ↑T
    f : α → β
    hc : ContinuousOn f S
    hTm : Set.MapsTo f S T
    hne : T.Nonempty
    ⊢ Exists fun y => And (Membership.mem T y) (Set.EqOn f (Function.const α y) S)
  -/
  rcases S.eq_empty_or_nonempty with (rfl | ⟨x, hx⟩)
    /-
      case inl
      α : Type u
      inst✝² : TopologicalSpace α
      β : Type u_3
      inst✝¹ : TopologicalSpace β
      T : Set β
      inst✝ : DiscreteTopology ↑T
      f : α → β
      hne : T.Nonempty
      hS : IsPreconnected EmptyCollection.emptyCollection
      hc : ContinuousOn f EmptyCollection.emptyCollection
      hTm : Set.MapsTo f EmptyCollection.emptyCollection T
      ⊢ Exists fun y => And (Membership.mem T y) (Set.EqOn f (Function.const α y) Em …
    -/
  · exact hne.imp fun _ hy => ⟨hy, eqOn_empty _ _⟩
    /-
      🎉 no goals
    -/
    /-
      case inr.intro
      α : Type u
      inst✝² : TopologicalSpace α
      S : Set α
      hS : IsPreconnected S
      β : Type u_3
      inst✝¹ : TopologicalSpace β
      T : Set β
      inst✝ : DiscreteTopology ↑T
      f : α → β
      hc : ContinuousOn f S
      hTm : Set.MapsTo f S T
      hne : T.Nonempty
      x : α
      hx : Membership.mem S x
      ⊢ Exists fun y => And (Membership.mem T y) (Set.EqOn f (Function.const α y) S)
    -/
  · exact ⟨f x, hTm hx, fun x' hx' => hS.constant_of_mapsTo hc hTm hx' hx⟩
    /-
      🎉 no goals
    -/

