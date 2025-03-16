protected lemma Topology.IsEmbedding.toPullbackDiag (f : X → Y) : IsEmbedding (toPullbackDiag f) :=
  .mk' _ (injective_toPullbackDiag f) fun x ↦ by
    /-
      X : Type u_1
      Y : Sort u_2
      inst✝ : TopologicalSpace X
      f : X → Y
      x : X
      ⊢ Eq (Filter.comap (toPullbackDiag f) (nhds (toPullbackDiag f x))) (nhds x)
    -/
    rw [toPullbackDiag, nhds_induced, Filter.comap_comap, nhds_prod_eq, Filter.comap_prod]
    /-
      X : Type u_1
      Y : Sort u_2
      inst✝ : TopologicalSpace X
      f : X → Y
      x : X
      ⊢ Eq (Min.min (Filter.comap (Function.comp Prod.fst (Function.comp Subtype.val …
    -/
    erw [Filter.comap_id, inf_idem]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-26")]
alias embedding_toPullbackDiag := IsEmbedding.toPullbackDiag


lemma Continuous.mapPullback {X₁ X₂ Y₁ Y₂ Z₁ Z₂}
    [TopologicalSpace X₁] [TopologicalSpace X₂] [TopologicalSpace Z₁] [TopologicalSpace Z₂]
    {f₁ : X₁ → Y₁} {g₁ : Z₁ → Y₁} {f₂ : X₂ → Y₂} {g₂ : Z₂ → Y₂}
    {mapX : X₁ → X₂} (contX : Continuous mapX) {mapY : Y₁ → Y₂}
    {mapZ : Z₁ → Z₂} (contZ : Continuous mapZ)
    {commX : f₂ ∘ mapX = mapY ∘ f₁} {commZ : g₂ ∘ mapZ = mapY ∘ g₁} :
    Continuous (Function.mapPullback mapX mapY mapZ commX commZ) := by
  /-
    X₁ : Type u_1
    X₂ : Type u_2
    Y₁ : Sort u_3
    Y₂ : Sort u_4
    Z₁ : Type u_5
    Z₂ : Type u_6
    inst✝³ : TopologicalSpace X₁
    inst✝² : TopologicalSpace X₂
    inst✝¹ : TopologicalSpace Z₁
    inst✝ : TopologicalSpace Z₂
    f₁ : X₁ → Y₁
    g₁ : Z₁ → Y₁
    f₂ : X₂ → Y₂
    g₂ : Z₂ → Y₂
    mapX : X₁ → X₂
    contX : Continuous mapX
    mapY : Y₁ → Y₂
    mapZ : Z₁ → Z₂
    contZ : Continuous mapZ
    commX : Eq (Function.comp f₂ mapX) (Function.comp mapY f₁)
    commZ : Eq (Function.comp g₂ mapZ) (Function.comp mapY g₁)
    ⊢ Continuous (Function.mapPullback mapX mapY mapZ commX commZ)
  -/
  refine continuous_induced_rng.mpr (continuous_prod_mk.mpr ⟨?_, ?_⟩) <;>
  /-
    case refine_1
    X₁ : Type u_1
    X₂ : Type u_2
    Y₁ : Sort u_3
    Y₂ : Sort u_4
    Z₁ : Type u_5
    Z₂ : Type u_6
    inst✝³ : TopologicalSpace X₁
    inst✝² : TopologicalSpace X₂
    inst✝¹ : TopologicalSpace Z₁
    inst✝ : TopologicalSpace Z₂
    f₁ : X₁ → Y₁
    g₁ : Z₁ → Y₁
    f₂ : X₂ → Y₂
    g₂ : Z₂ → Y₂
    mapX : X₁ → X₂
    contX : Continuous mapX
    mapY : Y₁ → Y₂
    mapZ : Z₁ → Z₂
    contZ : Continuous mapZ
    commX : Eq (Function.comp f₂ mapX) (Function.comp mapY f₁)
    commZ : Eq (Function.comp g₂ mapZ) (Function.comp mapY g₁)
    ⊢ Continuous fun x => mapX x.fst
  -/
  /-
    🎉 no goals
  -/
  apply_rules [continuous_fst, continuous_snd, continuous_subtype_val, Continuous.comp]
  /-
    🎉 no goals
  -/


/-- A function from a topological space `X` to a type `Y` is a separated map if any two distinct
  points in `X` with the same image in `Y` can be separated by open neighborhoods. -/
def IsSeparatedMap (f : X → Y) : Prop := ∀ x₁ x₂, f x₁ = f x₂ →
    x₁ ≠ x₂ → ∃ s₁ s₂, IsOpen s₁ ∧ IsOpen s₂ ∧ x₁ ∈ s₁ ∧ x₂ ∈ s₂ ∧ Disjoint s₁ s₂


lemma t2space_iff_isSeparatedMap (y : Y) : T2Space X ↔ IsSeparatedMap fun _ : X ↦ y :=
  ⟨fun ⟨t2⟩ _ _ _ hne ↦ t2 hne, fun sep ↦ ⟨fun x₁ x₂ hne ↦ sep x₁ x₂ rfl hne⟩⟩


lemma T2Space.isSeparatedMap [T2Space X] (f : X → Y) : IsSeparatedMap f := fun _ _ _ ↦ t2_separation


lemma Function.Injective.isSeparatedMap {f : X → Y} (inj : f.Injective) : IsSeparatedMap f :=
  fun _ _ he hne ↦ (hne (inj he)).elim


lemma isSeparatedMap_iff_disjoint_nhds {f : X → Y} : IsSeparatedMap f ↔
    ∀ x₁ x₂, f x₁ = f x₂ → x₁ ≠ x₂ → Disjoint (𝓝 x₁) (𝓝 x₂) :=
  forall₃_congr fun x x' _ ↦ by simp only [(nhds_basis_opens x).disjoint_iff (nhds_basis_opens x'),
    exists_prop, ← exists_and_left, and_assoc, and_comm, and_left_comm]


lemma isSeparatedMap_iff_nhds {f : X → Y} : IsSeparatedMap f ↔
    ∀ x₁ x₂, f x₁ = f x₂ → x₁ ≠ x₂ → ∃ s₁ ∈ 𝓝 x₁, ∃ s₂ ∈ 𝓝 x₂, Disjoint s₁ s₂ := by
  /-
    X : Type u_1
    Y : Sort u_2
    inst✝ : TopologicalSpace X
    f : X → Y
    ⊢ Iff (IsSeparatedMap f) (∀ (x₁ x₂ : X), Eq (f x₁) (f x₂) → Ne x₁ x₂ → Exists  …
  -/
  simp_rw [isSeparatedMap_iff_disjoint_nhds, Filter.disjoint_iff]
  /-
    🎉 no goals
  -/


open Set Filter in
theorem isSeparatedMap_iff_isClosed_diagonal {f : X → Y} :
    IsSeparatedMap f ↔ IsClosed f.pullbackDiagonal := by
  simp_rw [isSeparatedMap_iff_nhds, ← isOpen_compl_iff, isOpen_iff_mem_nhds,
    Subtype.forall, Prod.forall, nhds_induced, nhds_prod_eq]
  /-
    X : Type u_1
    Y : Sort u_2
    inst✝ : TopologicalSpace X
    f : X → Y
    ⊢ Iff (∀ (x₁ x₂ : X), Eq (f x₁) (f x₂) → Ne x₁ x₂ → Exists fun s₁ => And (Memb …
  -/
  refine forall₄_congr fun x₁ x₂ _ _ ↦ ⟨fun h ↦ ?_, fun ⟨t, ht, t_sub⟩ ↦ ?_⟩
    /-
      case refine_1
      X : Type u_1
      Y : Sort u_2
      inst✝ : TopologicalSpace X
      f : X → Y
      x₁ x₂ : X
      x✝¹ : Eq (f x₁) (f x₂)
      x✝ : Ne x₁ x₂
      h : Exists fun s₁ => And (Membership.mem (nhds x₁) s₁) (Exists fun s₂ => And ( …
      ⊢ Membership.mem (Filter.comap Subtype.val (SProd.sprod (nhds x₁) (nhds x₂)))  …
    -/
  · simp_rw [← Filter.disjoint_iff, ← compl_diagonal_mem_prod] at h
    /-
      case refine_1
      X : Type u_1
      Y : Sort u_2
      inst✝ : TopologicalSpace X
      f : X → Y
      x₁ x₂ : X
      x✝¹ : Eq (f x₁) (f x₂)
      x✝ : Ne x₁ x₂
      h : Membership.mem (SProd.sprod (nhds x₁) (nhds x₂)) (HasCompl.compl (Set.diag …
      ⊢ Membership.mem (Filter.comap Subtype.val (SProd.sprod (nhds x₁) (nhds x₂)))  …
    -/
    exact ⟨_, h, subset_rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      X : Type u_1
      Y : Sort u_2
      inst✝ : TopologicalSpace X
      f : X → Y
      x₁ x₂ : X
      x✝² : Eq (f x₁) (f x₂)
      x✝¹ : Ne x₁ x₂
      x✝ : Membership.mem (Filter.comap Subtype.val (SProd.sprod (nhds x₁) (nhds x₂) …
      t : Set (Prod X X)
      ht : Membership.mem (SProd.sprod (nhds x₁) (nhds x₂)) t
      t_sub : HasSubset.Subset (Set.preimage Subtype.val t) (HasCompl.compl (Functio …
      ⊢ Exists fun s₁ => And (Membership.mem (nhds x₁) s₁) (Exists fun s₂ => And (Me …
    -/
  · obtain ⟨s₁, h₁, s₂, h₂, s_sub⟩ := mem_prod_iff.mp ht
    /-
      case refine_2.intro.intro.intro.intro
      X : Type u_1
      Y : Sort u_2
      inst✝ : TopologicalSpace X
      f : X → Y
      x₁ x₂ : X
      x✝² : Eq (f x₁) (f x₂)
      x✝¹ : Ne x₁ x₂
      x✝ : Membership.mem (Filter.comap Subtype.val (SProd.sprod (nhds x₁) (nhds x₂) …
      t : Set (Prod X X)
      ht : Membership.mem (SProd.sprod (nhds x₁) (nhds x₂)) t
      t_sub : HasSubset.Subset (Set.preimage Subtype.val t) (HasCompl.compl (Functio …
      s₁ : Set X
      h₁ : Membership.mem (nhds x₁) s₁
      s₂ : Set X
      h₂ : Membership.mem (nhds x₂) s₂
      s_sub : HasSubset.Subset (SProd.sprod s₁ s₂) t
      ⊢ Exists fun s₁ => And (Membership.mem (nhds x₁) s₁) (Exists fun s₂ => And (Me …
    -/
    exact ⟨s₁, h₁, s₂, h₂, disjoint_left.2 fun x h₁ h₂ ↦ @t_sub ⟨(x, x), rfl⟩ (s_sub ⟨h₁, h₂⟩) rfl⟩
    /-
      🎉 no goals
    -/


theorem isSeparatedMap_iff_isClosedEmbedding {f : X → Y} :
    IsSeparatedMap f ↔ IsClosedEmbedding (toPullbackDiag f) := by
  /-
    X : Type u_1
    Y : Sort u_2
    inst✝ : TopologicalSpace X
    f : X → Y
    ⊢ Iff (IsSeparatedMap f) (Topology.IsClosedEmbedding (toPullbackDiag f))
  -/
  rw [isSeparatedMap_iff_isClosed_diagonal, ← range_toPullbackDiag]
  /-
    X : Type u_1
    Y : Sort u_2
    inst✝ : TopologicalSpace X
    f : X → Y
    ⊢ Iff (IsClosed (Set.range (toPullbackDiag f))) (Topology.IsClosedEmbedding (t …
  -/
  exact ⟨fun h ↦ ⟨.toPullbackDiag f, h⟩, fun h ↦ h.isClosed_range⟩
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-20")]
alias isSeparatedMap_iff_closedEmbedding := isSeparatedMap_iff_isClosedEmbedding


theorem isSeparatedMap_iff_isClosedMap {f : X → Y} :
    IsSeparatedMap f ↔ IsClosedMap (toPullbackDiag f) :=
  isSeparatedMap_iff_isClosedEmbedding.trans
    ⟨IsClosedEmbedding.isClosedMap, .of_continuous_injective_isClosedMap
      (IsEmbedding.toPullbackDiag f).continuous (injective_toPullbackDiag f)⟩


open Function.Pullback in
theorem IsSeparatedMap.pullback {f : X → Y} (sep : IsSeparatedMap f) (g : A → Y) :
    IsSeparatedMap (@snd X Y A f g) := by
  /-
    X : Type u_1
    Y : Sort u_2
    A : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace A
    f : X → Y
    sep : IsSeparatedMap f
    g : A → Y
    ⊢ IsSeparatedMap Function.Pullback.snd
  -/
  rw [isSeparatedMap_iff_isClosed_diagonal] at sep ⊢
  /-
    X : Type u_1
    Y : Sort u_2
    A : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace A
    f : X → Y
    sep : IsClosed (Function.pullbackDiagonal f)
    g : A → Y
    ⊢ IsClosed (Function.pullbackDiagonal Function.Pullback.snd)
  -/
  rw [← preimage_map_fst_pullbackDiagonal]
  /-
    X : Type u_1
    Y : Sort u_2
    A : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace A
    f : X → Y
    sep : IsClosed (Function.pullbackDiagonal f)
    g : A → Y
    ⊢ IsClosed (Set.preimage Function.PullbackSelf.map_fst (Function.pullbackDiago …
  -/
  refine sep.preimage (Continuous.mapPullback ?_ ?_) <;>
  /-
    case refine_1
    X : Type u_1
    Y : Sort u_2
    A : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace A
    f : X → Y
    sep : IsClosed (Function.pullbackDiagonal f)
    g : A → Y
    ⊢ Continuous Function.Pullback.fst
  -/
  /-
    🎉 no goals
  -/
  apply_rules [continuous_fst, continuous_subtype_val, Continuous.comp]
  /-
    🎉 no goals
  -/


theorem IsSeparatedMap.comp_left {A} {f : X → Y} (sep : IsSeparatedMap f) {g : Y → A}
    (inj : g.Injective) : IsSeparatedMap (g ∘ f) := fun x₁ x₂ he ↦ sep x₁ x₂ (inj he)


theorem IsSeparatedMap.comp_right {f : X → Y} (sep : IsSeparatedMap f) {g : A → X}
    (cont : Continuous g) (inj : g.Injective) : IsSeparatedMap (f ∘ g) := by
  /-
    X : Type u_1
    Y : Sort u_2
    A : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace A
    f : X → Y
    sep : IsSeparatedMap f
    g : A → X
    cont : Continuous g
    inj : Function.Injective g
    ⊢ IsSeparatedMap (Function.comp f g)
  -/
  rw [isSeparatedMap_iff_isClosed_diagonal] at sep ⊢
  /-
    X : Type u_1
    Y : Sort u_2
    A : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace A
    f : X → Y
    sep : IsClosed (Function.pullbackDiagonal f)
    g : A → X
    cont : Continuous g
    inj : Function.Injective g
    ⊢ IsClosed (Function.pullbackDiagonal (Function.comp f g))
  -/
  rw [← inj.preimage_pullbackDiagonal]
  /-
    X : Type u_1
    Y : Sort u_2
    A : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace A
    f : X → Y
    sep : IsClosed (Function.pullbackDiagonal f)
    g : A → X
    cont : Continuous g
    inj : Function.Injective g
    ⊢ IsClosed (Set.preimage (Function.mapPullback g id g ⋯ ⋯) (Function.pullbackD …
  -/
  exact sep.preimage (cont.mapPullback cont)
  /-
    🎉 no goals
  -/


/-- A function from a topological space `X` is locally injective if every point of `X`
  has a neighborhood on which `f` is injective. -/
def IsLocallyInjective (f : X → Y) : Prop := ∀ x : X, ∃ U, IsOpen U ∧ x ∈ U ∧ U.InjOn f


lemma Function.Injective.IsLocallyInjective {f : X → Y} (inj : f.Injective) :
    IsLocallyInjective f := fun _ ↦ ⟨_, isOpen_univ, trivial, fun _ _ _ _ ↦ @inj _ _⟩


lemma isLocallyInjective_iff_nhds {f : X → Y} :
    IsLocallyInjective f ↔ ∀ x : X, ∃ U ∈ 𝓝 x, U.InjOn f := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    f : X → Y
    ⊢ Iff (IsLocallyInjective f) (∀ (x : X), Exists fun U => And (Membership.mem ( …
  -/
  constructor <;> intro h x
    /-
      case mp
      X : Type u_1
      Y : Type u_2
      inst✝ : TopologicalSpace X
      f : X → Y
      h : IsLocallyInjective f
      x : X
      ⊢ Exists fun U => And (Membership.mem (nhds x) U) (Set.InjOn f U)
    -/
  · obtain ⟨U, ho, hm, hi⟩ := h x; exact ⟨U, ho.mem_nhds hm, hi⟩
                                   /-
                                     🎉 no goals
                                   -/
    /-
      case mpr
      X : Type u_1
      Y : Type u_2
      inst✝ : TopologicalSpace X
      f : X → Y
      h : ∀ (x : X), Exists fun U => And (Membership.mem (nhds x) U) (Set.InjOn f U)
      x : X
      ⊢ Exists fun U => And (IsOpen U) (And (Membership.mem U x) (Set.InjOn f U))
    -/
  · obtain ⟨U, hn, hi⟩ := h x
    /-
      case mpr.intro.intro
      X : Type u_1
      Y : Type u_2
      inst✝ : TopologicalSpace X
      f : X → Y
      h : ∀ (x : X), Exists fun U => And (Membership.mem (nhds x) U) (Set.InjOn f U)
      x : X
      U : Set X
      hn : Membership.mem (nhds x) U
      hi : Set.InjOn f U
      ⊢ Exists fun U => And (IsOpen U) (And (Membership.mem U x) (Set.InjOn f U))
    -/
    exact ⟨interior U, isOpen_interior, mem_interior_iff_mem_nhds.mpr hn, hi.mono interior_subset⟩
    /-
      🎉 no goals
    -/


theorem isLocallyInjective_iff_isOpen_diagonal {f : X → Y} :
    IsLocallyInjective f ↔ IsOpen f.pullbackDiagonal := by
  simp_rw [isLocallyInjective_iff_nhds, isOpen_iff_mem_nhds,
    Subtype.forall, Prod.forall, nhds_induced, nhds_prod_eq, Filter.mem_comap]
  /-
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    f : X → Y
    ⊢ Iff (∀ (x : X), Exists fun U => And (Membership.mem (nhds x) U) (Set.InjOn f …
  -/
  refine ⟨?_, fun h x ↦ ?_⟩
    /-
      case refine_1
      X : Type u_1
      Y : Type u_2
      inst✝ : TopologicalSpace X
      f : X → Y
      ⊢ (∀ (x : X), Exists fun U => And (Membership.mem (nhds x) U) (Set.InjOn f U)) …
    -/
  · rintro h x x' hx (rfl : x = x')
    /-
      case refine_1
      X : Type u_1
      Y : Type u_2
      inst✝ : TopologicalSpace X
      f : X → Y
      h : ∀ (x : X), Exists fun U => And (Membership.mem (nhds x) U) (Set.InjOn f U)
      x : X
      hx : Eq (f x) (f x)
      ⊢ Exists fun t => And (Membership.mem (SProd.sprod (nhds x) (nhds x)) t) (HasS …
    -/
    obtain ⟨U, hn, hi⟩ := h x
    /-
      case refine_1.intro.intro
      X : Type u_1
      Y : Type u_2
      inst✝ : TopologicalSpace X
      f : X → Y
      h : ∀ (x : X), Exists fun U => And (Membership.mem (nhds x) U) (Set.InjOn f U)
      x : X
      hx : Eq (f x) (f x)
      U : Set X
      hn : Membership.mem (nhds x) U
      hi : Set.InjOn f U
      ⊢ Exists fun t => And (Membership.mem (SProd.sprod (nhds x) (nhds x)) t) (HasS …
    -/
    exact ⟨_, Filter.prod_mem_prod hn hn, fun {p} hp ↦ hi hp.1 hp.2 p.2⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      X : Type u_1
      Y : Type u_2
      inst✝ : TopologicalSpace X
      f : X → Y
      h : ∀ (a b : X) (b_1 : Eq (f a) (f b)), Membership.mem (Function.pullbackDiago …
      x : X
      ⊢ Exists fun U => And (Membership.mem (nhds x) U) (Set.InjOn f U)
    -/
  · obtain ⟨t, ht, t_sub⟩ := h x x rfl rfl
    /-
      case refine_2.intro.intro
      X : Type u_1
      Y : Type u_2
      inst✝ : TopologicalSpace X
      f : X → Y
      h : ∀ (a b : X) (b_1 : Eq (f a) (f b)), Membership.mem (Function.pullbackDiago …
      x : X
      t : Set (Prod X X)
      ht : Membership.mem (SProd.sprod (nhds x) (nhds x)) t
      t_sub : HasSubset.Subset (Set.preimage Subtype.val t) (Function.pullbackDiagon …
      ⊢ Exists fun U => And (Membership.mem (nhds x) U) (Set.InjOn f U)
    -/
    obtain ⟨t₁, h₁, t₂, h₂, prod_sub⟩ := Filter.mem_prod_iff.mp ht
    exact ⟨t₁ ∩ t₂, Filter.inter_mem h₁ h₂,
      fun x₁ h₁ x₂ h₂ he ↦ @t_sub ⟨(x₁, x₂), he⟩ (prod_sub ⟨h₁.1, h₂.2⟩)⟩


theorem IsLocallyInjective_iff_isOpenEmbedding {f : X → Y} :
    IsLocallyInjective f ↔ IsOpenEmbedding (toPullbackDiag f) := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    f : X → Y
    ⊢ Iff (IsLocallyInjective f) (Topology.IsOpenEmbedding (toPullbackDiag f))
  -/
  rw [isLocallyInjective_iff_isOpen_diagonal, ← range_toPullbackDiag]
  /-
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    f : X → Y
    ⊢ Iff (IsOpen (Set.range (toPullbackDiag f))) (Topology.IsOpenEmbedding (toPul …
  -/
  exact ⟨fun h ↦ ⟨.toPullbackDiag f, h⟩, fun h ↦ h.isOpen_range⟩
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-18")]
alias IsLocallyInjective_iff_openEmbedding := IsLocallyInjective_iff_isOpenEmbedding


theorem isLocallyInjective_iff_isOpenMap {f : X → Y} :
    IsLocallyInjective f ↔ IsOpenMap (toPullbackDiag f) :=
  IsLocallyInjective_iff_isOpenEmbedding.trans
    ⟨IsOpenEmbedding.isOpenMap, .of_continuous_injective_isOpenMap
      (IsEmbedding.toPullbackDiag f).continuous (injective_toPullbackDiag f)⟩


theorem discreteTopology_iff_locallyInjective (y : Y) :
    DiscreteTopology X ↔ IsLocallyInjective fun _ : X ↦ y := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    y : Y
    ⊢ Iff (DiscreteTopology X) (IsLocallyInjective fun x => y)
  -/
  rw [discreteTopology_iff_singleton_mem_nhds, isLocallyInjective_iff_nhds]
  /-
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    y : Y
    ⊢ Iff (∀ (x : X), Membership.mem (nhds x) (Singleton.singleton x)) (∀ (x : X), …
  -/
  refine forall_congr' fun x ↦ ⟨fun h ↦ ⟨{x}, h, Set.injOn_singleton _ _⟩, fun ⟨U, hU, inj⟩ ↦ ?_⟩
  /-
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    y : Y
    x : X
    x✝ : Exists fun U => And (Membership.mem (nhds x) U) (Set.InjOn (fun x => y) U)
    U : Set X
    hU : Membership.mem (nhds x) U
    inj : Set.InjOn (fun x => y) U
    ⊢ Membership.mem (nhds x) (Singleton.singleton x)
  -/
  convert hU; ext x'; refine ⟨?_, fun h ↦ inj h (mem_of_mem_nhds hU) rfl⟩
  /-
    case h.e'_5.h
    X : Type u_1
    Y : Type u_2
    inst✝ : TopologicalSpace X
    y : Y
    x : X
    x✝ : Exists fun U => And (Membership.mem (nhds x) U) (Set.InjOn (fun x => y) U)
    U : Set X
    hU : Membership.mem (nhds x) U
    inj : Set.InjOn (fun x => y) U
    x' : X
    ⊢ Membership.mem (Singleton.singleton x) x' → Membership.mem U x'
  -/
  rintro rfl; exact mem_of_mem_nhds hU
              /-
                🎉 no goals
              -/


theorem IsLocallyInjective.comp_left {A} {f : X → Y} (hf : IsLocallyInjective f) {g : Y → A}
    (hg : g.Injective) : IsLocallyInjective (g ∘ f) :=
  fun x ↦ let ⟨U, hU, hx, inj⟩ := hf x; ⟨U, hU, hx, hg.comp_injOn inj⟩


theorem IsLocallyInjective.comp_right {f : X → Y} (hf : IsLocallyInjective f) {g : A → X}
    (cont : Continuous g) (hg : g.Injective) : IsLocallyInjective (f ∘ g) := by
  /-
    X : Type u_1
    Y : Type u_2
    A : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace A
    f : X → Y
    hf : IsLocallyInjective f
    g : A → X
    cont : Continuous g
    hg : Function.Injective g
    ⊢ IsLocallyInjective (Function.comp f g)
  -/
  rw [isLocallyInjective_iff_isOpen_diagonal] at hf ⊢
  /-
    X : Type u_1
    Y : Type u_2
    A : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace A
    f : X → Y
    hf : IsOpen (Function.pullbackDiagonal f)
    g : A → X
    cont : Continuous g
    hg : Function.Injective g
    ⊢ IsOpen (Function.pullbackDiagonal (Function.comp f g))
  -/
  rw [← hg.preimage_pullbackDiagonal]
  /-
    X : Type u_1
    Y : Type u_2
    A : Type u_3
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace A
    f : X → Y
    hf : IsOpen (Function.pullbackDiagonal f)
    g : A → X
    cont : Continuous g
    hg : Function.Injective g
    ⊢ IsOpen (Set.preimage (Function.mapPullback g id g ⋯ ⋯) (Function.pullbackDia …
  -/
  apply hf.preimage (cont.mapPullback cont)
  /-
    🎉 no goals
  -/


set_option linter.unusedVariables false in
theorem IsSeparatedMap.isClosed_eqLocus (sep : IsSeparatedMap f) (he : f ∘ g₁ = f ∘ g₂) :
    IsClosed {a | g₁ a = g₂ a} :=
  let g : A → f.Pullback f := fun a ↦ ⟨⟨g₁ a, g₂ a⟩, congr_fun he a⟩
                                                             /-
                                                               X : Type u_1
                                                               Y : Sort u_2
                                                               A : Type u_3
                                                               inst✝¹ : TopologicalSpace X
                                                               inst✝ : TopologicalSpace A
                                                               f : X → Y
                                                               g₁ g₂ : A → X
                                                               h₁ : Continuous g₁
                                                               h₂ : Continuous g₂
                                                               sep : IsSeparatedMap f
                                                               he : Eq (Function.comp f g₁) (Function.comp f g₂)
                                                               g : A → Function.Pullback f f := fun a => ⟨{ fst := g₁ a, snd := g₂ a }, ⋯⟩
                                                               ⊢ Continuous g
                                                             -/
  (isSeparatedMap_iff_isClosed_diagonal.mp sep).preimage (by fun_prop : Continuous g)
                                                             /-
                                                               🎉 no goals
                                                             -/


set_option linter.unusedVariables false in
theorem IsLocallyInjective.isOpen_eqLocus (inj : IsLocallyInjective f) (he : f ∘ g₁ = f ∘ g₂) :
    IsOpen {a | g₁ a = g₂ a} :=
  let g : A → f.Pullback f := fun a ↦ ⟨⟨g₁ a, g₂ a⟩, congr_fun he a⟩
                                                               /-
                                                                 X : Type u_1
                                                                 Y : Type u_2
                                                                 A : Type u_3
                                                                 inst✝¹ : TopologicalSpace X
                                                                 inst✝ : TopologicalSpace A
                                                                 f : X → Y
                                                                 g₁ g₂ : A → X
                                                                 h₁ : Continuous g₁
                                                                 h₂ : Continuous g₂
                                                                 inj : IsLocallyInjective f
                                                                 he : Eq (Function.comp f g₁) (Function.comp f g₂)
                                                                 g : A → Function.Pullback f f := fun a => ⟨{ fst := g₁ a, snd := g₂ a }, ⋯⟩
                                                                 ⊢ Continuous g
                                                               -/
  (isLocallyInjective_iff_isOpen_diagonal.mp inj).preimage (by fun_prop : Continuous g)
                                                               /-
                                                                 🎉 no goals
                                                               -/


/-- If `p` is a locally injective separated map, and `A` is a connected space,
  then two lifts `g₁, g₂ : A → E` of a map `f : A → X` are equal if they agree at one point. -/
theorem eq_of_comp_eq
    [PreconnectedSpace A] (h₁ : Continuous g₁) (h₂ : Continuous g₂)
    (he : p ∘ g₁ = p ∘ g₂) (a : A) (ha : g₁ a = g₂ a) : g₁ = g₂ := funext fun a' ↦ by
  apply (IsClopen.eq_univ ⟨sep.isClosed_eqLocus h₁ h₂ he, inj.isOpen_eqLocus h₁ h₂ he⟩ ⟨a, ha⟩).symm
    ▸ Set.mem_univ a'


theorem eqOn_of_comp_eqOn (hs : IsPreconnected s) (h₁ : ContinuousOn g₁ s) (h₂ : ContinuousOn g₂ s)
    (he : s.EqOn (p ∘ g₁) (p ∘ g₂)) {a : A} (has : a ∈ s) (ha : g₁ a = g₂ a) : s.EqOn g₁ g₂ := by
  /-
    X : Type u_1
    E : Type u_2
    A : Type u_3
    inst✝¹ : TopologicalSpace E
    inst✝ : TopologicalSpace A
    p : E → X
    s : Set A
    g₁ g₂ : A → E
    sep : IsSeparatedMap p
    inj : IsLocallyInjective p
    hs : IsPreconnected s
    h₁ : ContinuousOn g₁ s
    h₂ : ContinuousOn g₂ s
    he : Set.EqOn (Function.comp p g₁) (Function.comp p g₂) s
    a : A
    has : Membership.mem s a
    ha : Eq (g₁ a) (g₂ a)
    ⊢ Set.EqOn g₁ g₂ s
  -/
  rw [← Set.restrict_eq_restrict_iff] at he ⊢
  /-
    X : Type u_1
    E : Type u_2
    A : Type u_3
    inst✝¹ : TopologicalSpace E
    inst✝ : TopologicalSpace A
    p : E → X
    s : Set A
    g₁ g₂ : A → E
    sep : IsSeparatedMap p
    inj : IsLocallyInjective p
    hs : IsPreconnected s
    h₁ : ContinuousOn g₁ s
    h₂ : ContinuousOn g₂ s
    he : Eq (s.restrict (Function.comp p g₁)) (s.restrict (Function.comp p g₂))
    a : A
    has : Membership.mem s a
    ha : Eq (g₁ a) (g₂ a)
    ⊢ Eq (s.restrict g₁) (s.restrict g₂)
  -/
  rw [continuousOn_iff_continuous_restrict] at h₁ h₂
  /-
    X : Type u_1
    E : Type u_2
    A : Type u_3
    inst✝¹ : TopologicalSpace E
    inst✝ : TopologicalSpace A
    p : E → X
    s : Set A
    g₁ g₂ : A → E
    sep : IsSeparatedMap p
    inj : IsLocallyInjective p
    hs : IsPreconnected s
    h₁ : Continuous (s.restrict g₁)
    h₂ : Continuous (s.restrict g₂)
    he : Eq (s.restrict (Function.comp p g₁)) (s.restrict (Function.comp p g₂))
    a : A
    has : Membership.mem s a
    ha : Eq (g₁ a) (g₂ a)
    ⊢ Eq (s.restrict g₁) (s.restrict g₂)
  -/
  rw [isPreconnected_iff_preconnectedSpace] at hs
  /-
    X : Type u_1
    E : Type u_2
    A : Type u_3
    inst✝¹ : TopologicalSpace E
    inst✝ : TopologicalSpace A
    p : E → X
    s : Set A
    g₁ g₂ : A → E
    sep : IsSeparatedMap p
    inj : IsLocallyInjective p
    hs : PreconnectedSpace ↑s
    h₁ : Continuous (s.restrict g₁)
    h₂ : Continuous (s.restrict g₂)
    he : Eq (s.restrict (Function.comp p g₁)) (s.restrict (Function.comp p g₂))
    a : A
    has : Membership.mem s a
    ha : Eq (g₁ a) (g₂ a)
    ⊢ Eq (s.restrict g₁) (s.restrict g₂)
  -/
  exact sep.eq_of_comp_eq inj h₁ h₂ he ⟨a, has⟩ ha
  /-
    🎉 no goals
  -/


theorem const_of_comp [PreconnectedSpace A] (cont : Continuous g)
    (he : ∀ a a', p (g a) = p (g a')) (a a') : g a = g a' :=
  congr_fun (sep.eq_of_comp_eq inj cont continuous_const (funext fun a ↦ he a a') a' rfl) a


theorem constOn_of_comp (hs : IsPreconnected s) (cont : ContinuousOn g s)
    (he : ∀ a ∈ s, ∀ a' ∈ s, p (g a) = p (g a'))
    {a a'} (ha : a ∈ s) (ha' : a' ∈ s) : g a = g a' :=
  sep.eqOn_of_comp_eqOn inj hs cont continuous_const.continuousOn
    (fun a ha ↦ he a ha a' ha') ha' rfl ha


