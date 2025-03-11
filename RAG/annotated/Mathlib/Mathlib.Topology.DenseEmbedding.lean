/-- `i : α → β` is "dense inducing" if it has dense range and the topology on `α`
  is the one induced by `i` from the topology on `β`. -/
structure IsDenseInducing [TopologicalSpace α] [TopologicalSpace β] (i : α → β)
    extends IsInducing i : Prop where
  /-- The range of a dense inducing map is a dense set. -/
  protected dense : DenseRange i


lemma isInducing (di : IsDenseInducing i) : IsInducing i := di.toIsInducing


theorem nhds_eq_comap (di : IsDenseInducing i) : ∀ a : α, 𝓝 a = comap i (𝓝 <| i a) :=
  di.isInducing.nhds_eq_comap


protected theorem continuous (di : IsDenseInducing i) : Continuous i :=
  di.isInducing.continuous


theorem closure_range (di : IsDenseInducing i) : closure (range i) = univ :=
  di.dense.closure_range


protected theorem preconnectedSpace [PreconnectedSpace α] (di : IsDenseInducing i) :
    PreconnectedSpace β :=
  di.dense.preconnectedSpace di.continuous


theorem closure_image_mem_nhds {s : Set α} {a : α} (di : IsDenseInducing i) (hs : s ∈ 𝓝 a) :
    closure (i '' s) ∈ 𝓝 (i a) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    i : α → β
    s : Set α
    a : α
    di : IsDenseInducing i
    hs : Membership.mem (nhds a) s
    ⊢ Membership.mem (nhds (i a)) (closure (Set.image i s))
  -/
  rw [di.nhds_eq_comap a, ((nhds_basis_opens _).comap _).mem_iff] at hs
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    i : α → β
    s : Set α
    a : α
    di : IsDenseInducing i
    hs : Exists fun i_1 => And (And (Membership.mem i_1 (i a)) (IsOpen i_1)) (HasS …
    ⊢ Membership.mem (nhds (i a)) (closure (Set.image i s))
  -/
  rcases hs with ⟨U, ⟨haU, hUo⟩, sub : i ⁻¹' U ⊆ s⟩
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    i : α → β
    s : Set α
    a : α
    di : IsDenseInducing i
    U : Set β
    sub : HasSubset.Subset (Set.preimage i U) s
    haU : Membership.mem U (i a)
    hUo : IsOpen U
    ⊢ Membership.mem (nhds (i a)) (closure (Set.image i s))
  -/
  refine mem_of_superset (hUo.mem_nhds haU) ?_
  calc
    U ⊆ closure (i '' (i ⁻¹' U)) := di.dense.subset_closure_image_preimage_of_isOpen hUo
    _ ⊆ closure (i '' s) := closure_mono (image_subset i sub)


theorem dense_image (di : IsDenseInducing i) {s : Set α} : Dense (i '' s) ↔ Dense s := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    i : α → β
    di : IsDenseInducing i
    s : Set α
    ⊢ Iff (Dense (Set.image i s)) (Dense s)
  -/
  refine ⟨fun H x => ?_, di.dense.dense_image di.continuous⟩
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    i : α → β
    di : IsDenseInducing i
    s : Set α
    H : Dense (Set.image i s)
    x : α
    ⊢ Membership.mem (closure s) x
  -/
  rw [di.isInducing.closure_eq_preimage_closure_image, H.closure_eq, preimage_univ]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : TopologicalSpace β
    i : α → β
    di : IsDenseInducing i
    s : Set α
    H : Dense (Set.image i s)
    x : α
    ⊢ Membership.mem Set.univ x
  -/
  trivial
  /-
    🎉 no goals
  -/


/-- If `i : α → β` is a dense embedding with dense complement of the range, then any compact set in
`α` has empty interior. -/
theorem interior_compact_eq_empty [T2Space β] (di : IsDenseInducing i) (hd : Dense (range i)ᶜ)
    {s : Set α} (hs : IsCompact s) : interior s = ∅ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    i : α → β
    inst✝ : T2Space β
    di : IsDenseInducing i
    hd : Dense (HasCompl.compl (Set.range i))
    s : Set α
    hs : IsCompact s
    ⊢ Eq (interior s) EmptyCollection.emptyCollection
  -/
  refine eq_empty_iff_forall_not_mem.2 fun x hx => ?_
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    i : α → β
    inst✝ : T2Space β
    di : IsDenseInducing i
    hd : Dense (HasCompl.compl (Set.range i))
    s : Set α
    hs : IsCompact s
    x : α
    hx : Membership.mem (interior s) x
    ⊢ False
  -/
  rw [mem_interior_iff_mem_nhds] at hx
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    i : α → β
    inst✝ : T2Space β
    di : IsDenseInducing i
    hd : Dense (HasCompl.compl (Set.range i))
    s : Set α
    hs : IsCompact s
    x : α
    hx : Membership.mem (nhds x) s
    ⊢ False
  -/
  have := di.closure_image_mem_nhds hx
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    i : α → β
    inst✝ : T2Space β
    di : IsDenseInducing i
    hd : Dense (HasCompl.compl (Set.range i))
    s : Set α
    hs : IsCompact s
    x : α
    hx : Membership.mem (nhds x) s
    this : Membership.mem (nhds (i x)) (closure (Set.image i s))
    ⊢ False
  -/
  rw [(hs.image di.continuous).isClosed.closure_eq] at this
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    i : α → β
    inst✝ : T2Space β
    di : IsDenseInducing i
    hd : Dense (HasCompl.compl (Set.range i))
    s : Set α
    hs : IsCompact s
    x : α
    hx : Membership.mem (nhds x) s
    this : Membership.mem (nhds (i x)) (Set.image i s)
    ⊢ False
  -/
  rcases hd.inter_nhds_nonempty this with ⟨y, hyi, hys⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    i : α → β
    inst✝ : T2Space β
    di : IsDenseInducing i
    hd : Dense (HasCompl.compl (Set.range i))
    s : Set α
    hs : IsCompact s
    x : α
    hx : Membership.mem (nhds x) s
    this : Membership.mem (nhds (i x)) (Set.image i s)
    y : β
    hyi : Membership.mem (HasCompl.compl (Set.range i)) y
    hys : Membership.mem (Set.image i s) y
    ⊢ False
  -/
  exact hyi (image_subset_range _ _ hys)
  /-
    🎉 no goals
  -/


/-- The product of two dense inducings is a dense inducing -/
protected theorem prodMap [TopologicalSpace γ] [TopologicalSpace δ] {e₁ : α → β} {e₂ : γ → δ}
    (de₁ : IsDenseInducing e₁) (de₂ : IsDenseInducing e₂) :
    IsDenseInducing (Prod.map e₁ e₂) where
  toIsInducing := de₁.isInducing.prodMap de₂.isInducing
  dense := de₁.dense.prodMap de₂.dense


@[deprecated (since := "2024-10-06")]
protected alias prod := IsDenseInducing.prodMap


/-- If the domain of a `IsDenseInducing` map is a separable space, then so is the codomain. -/
protected theorem separableSpace [SeparableSpace α] (di : IsDenseInducing i) : SeparableSpace β :=
  di.dense.separableSpace di.continuous


/--
```
 γ -f→ α
g↓     ↓e
 δ -h→ β
```
-/
theorem tendsto_comap_nhds_nhds {d : δ} {a : α} (di : IsDenseInducing i)
    (H : Tendsto h (𝓝 d) (𝓝 (i a))) (comm : h ∘ g = i ∘ f) : Tendsto f (comap g (𝓝 d)) (𝓝 a) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    i : α → β
    inst✝ : TopologicalSpace δ
    f : γ → α
    g : γ → δ
    h : δ → β
    d : δ
    a : α
    di : IsDenseInducing i
    H : Filter.Tendsto h (nhds d) (nhds (i a))
    comm : Eq (Function.comp h g) (Function.comp i f)
    ⊢ Filter.Tendsto f (Filter.comap g (nhds d)) (nhds a)
  -/
  have lim1 : map g (comap g (𝓝 d)) ≤ 𝓝 d := map_comap_le
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    i : α → β
    inst✝ : TopologicalSpace δ
    f : γ → α
    g : γ → δ
    h : δ → β
    d : δ
    a : α
    di : IsDenseInducing i
    H : Filter.Tendsto h (nhds d) (nhds (i a))
    comm : Eq (Function.comp h g) (Function.comp i f)
    lim1 : LE.le (Filter.map g (Filter.comap g (nhds d))) (nhds d)
    ⊢ Filter.Tendsto f (Filter.comap g (nhds d)) (nhds a)
  -/
  replace lim1 : map h (map g (comap g (𝓝 d))) ≤ map h (𝓝 d) := map_mono lim1
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    i : α → β
    inst✝ : TopologicalSpace δ
    f : γ → α
    g : γ → δ
    h : δ → β
    d : δ
    a : α
    di : IsDenseInducing i
    H : Filter.Tendsto h (nhds d) (nhds (i a))
    comm : Eq (Function.comp h g) (Function.comp i f)
    lim1 : LE.le (Filter.map h (Filter.map g (Filter.comap g (nhds d)))) (Filter.m …
    ⊢ Filter.Tendsto f (Filter.comap g (nhds d)) (nhds a)
  -/
  rw [Filter.map_map, comm, ← Filter.map_map, map_le_iff_le_comap] at lim1
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    i : α → β
    inst✝ : TopologicalSpace δ
    f : γ → α
    g : γ → δ
    h : δ → β
    d : δ
    a : α
    di : IsDenseInducing i
    H : Filter.Tendsto h (nhds d) (nhds (i a))
    comm : Eq (Function.comp h g) (Function.comp i f)
    lim1 : LE.le (Filter.map f (Filter.comap g (nhds d))) (Filter.comap i (Filter. …
    ⊢ Filter.Tendsto f (Filter.comap g (nhds d)) (nhds a)
  -/
  have lim2 : comap i (map h (𝓝 d)) ≤ comap i (𝓝 (i a)) := comap_mono H
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    i : α → β
    inst✝ : TopologicalSpace δ
    f : γ → α
    g : γ → δ
    h : δ → β
    d : δ
    a : α
    di : IsDenseInducing i
    H : Filter.Tendsto h (nhds d) (nhds (i a))
    comm : Eq (Function.comp h g) (Function.comp i f)
    lim1 : LE.le (Filter.map f (Filter.comap g (nhds d))) (Filter.comap i (Filter. …
    lim2 : LE.le (Filter.comap i (Filter.map h (nhds d))) (Filter.comap i (nhds (i …
    ⊢ Filter.Tendsto f (Filter.comap g (nhds d)) (nhds a)
  -/
  rw [← di.nhds_eq_comap] at lim2
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    i : α → β
    inst✝ : TopologicalSpace δ
    f : γ → α
    g : γ → δ
    h : δ → β
    d : δ
    a : α
    di : IsDenseInducing i
    H : Filter.Tendsto h (nhds d) (nhds (i a))
    comm : Eq (Function.comp h g) (Function.comp i f)
    lim1 : LE.le (Filter.map f (Filter.comap g (nhds d))) (Filter.comap i (Filter. …
    lim2 : LE.le (Filter.comap i (Filter.map h (nhds d))) (nhds a)
    ⊢ Filter.Tendsto f (Filter.comap g (nhds d)) (nhds a)
  -/
  exact le_trans lim1 lim2
  /-
    🎉 no goals
  -/


protected theorem nhdsWithin_neBot (di : IsDenseInducing i) (b : β) : NeBot (𝓝[range i] b) :=
  di.dense.nhdsWithin_neBot b


theorem comap_nhds_neBot (di : IsDenseInducing i) (b : β) : NeBot (comap i (𝓝 b)) :=
  comap_neBot fun s hs => by
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      i : α → β
      di : IsDenseInducing i
      b : β
      s : Set β
      hs : Membership.mem (nhds b) s
      ⊢ Exists fun a => Membership.mem s (i a)
    -/
    rcases mem_closure_iff_nhds.1 (di.dense b) s hs with ⟨_, ⟨ha, a, rfl⟩⟩
    /-
      case intro.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝¹ : TopologicalSpace α
      inst✝ : TopologicalSpace β
      i : α → β
      di : IsDenseInducing i
      b : β
      s : Set β
      hs : Membership.mem (nhds b) s
      a : α
      ha : Membership.mem s (i a)
      ⊢ Exists fun a => Membership.mem s (i a)
    -/
    exact ⟨a, ha⟩
    /-
      🎉 no goals
    -/


/-- If `i : α → β` is a dense inducing, then any function `f : α → γ` "extends" to a function `g =
  IsDenseInducing.extend di f : β → γ`. If `γ` is Hausdorff and `f` has a continuous extension, then
  `g` is the unique such extension. In general, `g` might not be continuous or even extend `f`. -/
def extend (di : IsDenseInducing i) (f : α → γ) (b : β) : γ :=
  @limUnder _ _ _ ⟨f (di.dense.some b)⟩ (comap i (𝓝 b)) f


theorem extend_eq_of_tendsto [T2Space γ] (di : IsDenseInducing i) {b : β} {c : γ} {f : α → γ}
    (hf : Tendsto f (comap i (𝓝 b)) (𝓝 c)) : di.extend f b = c :=
  haveI := di.comap_nhds_neBot
  hf.limUnder_eq


theorem extend_eq_at [T2Space γ] (di : IsDenseInducing i) {f : α → γ} {a : α}
    (hf : ContinuousAt f a) : di.extend f (i a) = f a :=
  extend_eq_of_tendsto _ <| di.nhds_eq_comap a ▸ hf


theorem extend_eq_at' [T2Space γ] (di : IsDenseInducing i) {f : α → γ} {a : α} (c : γ)
    (hf : Tendsto f (𝓝 a) (𝓝 c)) : di.extend f (i a) = f a :=
  di.extend_eq_at (continuousAt_of_tendsto_nhds hf)


theorem extend_eq [T2Space γ] (di : IsDenseInducing i) {f : α → γ} (hf : Continuous f) (a : α) :
    di.extend f (i a) = f a :=
  di.extend_eq_at hf.continuousAt


/-- Variation of `extend_eq` where we ask that `f` has a limit along `comap i (𝓝 b)` for each
`b : β`. This is a strictly stronger assumption than continuity of `f`, but in a lot of cases
you'd have to prove it anyway to use `continuous_extend`, so this avoids doing the work twice. -/
theorem extend_eq' [T2Space γ] {f : α → γ} (di : IsDenseInducing i)
    (hf : ∀ b, ∃ c, Tendsto f (comap i (𝓝 b)) (𝓝 c)) (a : α) : di.extend f (i a) = f a := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    i : α → β
    inst✝¹ : TopologicalSpace γ
    inst✝ : T2Space γ
    f : α → γ
    di : IsDenseInducing i
    hf : ∀ (b : β), Exists fun c => Filter.Tendsto f (Filter.comap i (nhds b)) (nh …
    a : α
    ⊢ Eq (di.extend f (i a)) (f a)
  -/
  rcases hf (i a) with ⟨b, hb⟩
  /-
    case intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    i : α → β
    inst✝¹ : TopologicalSpace γ
    inst✝ : T2Space γ
    f : α → γ
    di : IsDenseInducing i
    hf : ∀ (b : β), Exists fun c => Filter.Tendsto f (Filter.comap i (nhds b)) (nh …
    a : α
    b : γ
    hb : Filter.Tendsto f (Filter.comap i (nhds (i a))) (nhds b)
    ⊢ Eq (di.extend f (i a)) (f a)
  -/
  refine di.extend_eq_at' b ?_
  /-
    case intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    i : α → β
    inst✝¹ : TopologicalSpace γ
    inst✝ : T2Space γ
    f : α → γ
    di : IsDenseInducing i
    hf : ∀ (b : β), Exists fun c => Filter.Tendsto f (Filter.comap i (nhds b)) (nh …
    a : α
    b : γ
    hb : Filter.Tendsto f (Filter.comap i (nhds (i a))) (nhds b)
    ⊢ Filter.Tendsto f (nhds a) (nhds b)
  -/
  rwa [← di.isInducing.nhds_eq_comap] at hb
  /-
    🎉 no goals
  -/


theorem extend_unique_at [T2Space γ] {b : β} {f : α → γ} {g : β → γ} (di : IsDenseInducing i)
    (hf : ∀ᶠ x in comap i (𝓝 b), g (i x) = f x) (hg : ContinuousAt g b) : di.extend f b = g b := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    i : α → β
    inst✝¹ : TopologicalSpace γ
    inst✝ : T2Space γ
    b : β
    f : α → γ
    g : β → γ
    di : IsDenseInducing i
    hf : Filter.Eventually (fun x => Eq (g (i x)) (f x)) (Filter.comap i (nhds b))
    hg : ContinuousAt g b
    ⊢ Eq (di.extend f b) (g b)
  -/
  refine di.extend_eq_of_tendsto fun s hs => mem_map.2 ?_
  suffices ∀ᶠ x : α in comap i (𝓝 b), g (i x) ∈ s from
    hf.mp (this.mono fun x hgx hfx => hfx ▸ hgx)
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    i : α → β
    inst✝¹ : TopologicalSpace γ
    inst✝ : T2Space γ
    b : β
    f : α → γ
    g : β → γ
    di : IsDenseInducing i
    hf : Filter.Eventually (fun x => Eq (g (i x)) (f x)) (Filter.comap i (nhds b))
    hg : ContinuousAt g b
    s : Set γ
    hs : Membership.mem (nhds (g b)) s
    ⊢ Filter.Eventually (fun x => Membership.mem s (g (i x))) (Filter.comap i (nhd …
  -/
  clear hf f
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    i : α → β
    inst✝¹ : TopologicalSpace γ
    inst✝ : T2Space γ
    b : β
    g : β → γ
    di : IsDenseInducing i
    hg : ContinuousAt g b
    s : Set γ
    hs : Membership.mem (nhds (g b)) s
    ⊢ Filter.Eventually (fun x => Membership.mem s (g (i x))) (Filter.comap i (nhd …
  -/
  refine eventually_comap.2 ((hg.eventually hs).mono ?_)
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    i : α → β
    inst✝¹ : TopologicalSpace γ
    inst✝ : T2Space γ
    b : β
    g : β → γ
    di : IsDenseInducing i
    hg : ContinuousAt g b
    s : Set γ
    hs : Membership.mem (nhds (g b)) s
    ⊢ ∀ (x : β), s (g x) → ∀ (a : α), Eq (i a) x → Membership.mem s (g (i a))
  -/
  rintro _ hxs x rfl
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    i : α → β
    inst✝¹ : TopologicalSpace γ
    inst✝ : T2Space γ
    b : β
    g : β → γ
    di : IsDenseInducing i
    hg : ContinuousAt g b
    s : Set γ
    hs : Membership.mem (nhds (g b)) s
    x : α
    hxs : s (g (i x))
    ⊢ Membership.mem s (g (i x))
  -/
  exact hxs
  /-
    🎉 no goals
  -/


theorem extend_unique [T2Space γ] {f : α → γ} {g : β → γ} (di : IsDenseInducing i)
    (hf : ∀ x, g (i x) = f x) (hg : Continuous g) : di.extend f = g :=
  funext fun _ => extend_unique_at di (Eventually.of_forall hf) hg.continuousAt


theorem continuousAt_extend [T3Space γ] {b : β} {f : α → γ} (di : IsDenseInducing i)
    (hf : ∀ᶠ x in 𝓝 b, ∃ c, Tendsto f (comap i <| 𝓝 x) (𝓝 c)) : ContinuousAt (di.extend f) b := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    i : α → β
    inst✝¹ : TopologicalSpace γ
    inst✝ : T3Space γ
    b : β
    f : α → γ
    di : IsDenseInducing i
    hf : Filter.Eventually (fun x => Exists fun c => Filter.Tendsto f (Filter.coma …
    ⊢ ContinuousAt (di.extend f) b
  -/
  set φ := di.extend f
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    i : α → β
    inst✝¹ : TopologicalSpace γ
    inst✝ : T3Space γ
    b : β
    f : α → γ
    di : IsDenseInducing i
    hf : Filter.Eventually (fun x => Exists fun c => Filter.Tendsto f (Filter.coma …
    φ : β → γ := di.extend f
    ⊢ ContinuousAt φ b
  -/
  haveI := di.comap_nhds_neBot
  suffices ∀ V' ∈ 𝓝 (φ b), IsClosed V' → φ ⁻¹' V' ∈ 𝓝 b by
    simpa [ContinuousAt, (closed_nhds_basis (φ b)).tendsto_right_iff]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    i : α → β
    inst✝¹ : TopologicalSpace γ
    inst✝ : T3Space γ
    b : β
    f : α → γ
    di : IsDenseInducing i
    hf : Filter.Eventually (fun x => Exists fun c => Filter.Tendsto f (Filter.coma …
    φ : β → γ := di.extend f
    this : ∀ (b : β), (Filter.comap i (nhds b)).NeBot
    ⊢ ∀ (V' : Set γ), Membership.mem (nhds (φ b)) V' → IsClosed V' → Membership.me …
  -/
  intro V' V'_in V'_closed
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    i : α → β
    inst✝¹ : TopologicalSpace γ
    inst✝ : T3Space γ
    b : β
    f : α → γ
    di : IsDenseInducing i
    hf : Filter.Eventually (fun x => Exists fun c => Filter.Tendsto f (Filter.coma …
    φ : β → γ := di.extend f
    this : ∀ (b : β), (Filter.comap i (nhds b)).NeBot
    V' : Set γ
    V'_in : Membership.mem (nhds (φ b)) V'
    V'_closed : IsClosed V'
    ⊢ Membership.mem (nhds b) (Set.preimage φ V')
  -/
  set V₁ := { x | Tendsto f (comap i <| 𝓝 x) (𝓝 <| φ x) }
  have V₁_in : V₁ ∈ 𝓝 b := by
    filter_upwards [hf]
    rintro x ⟨c, hc⟩
    rwa [← di.extend_eq_of_tendsto hc] at hc
  obtain ⟨V₂, V₂_in, V₂_op, hV₂⟩ : ∃ V₂ ∈ 𝓝 b, IsOpen V₂ ∧ ∀ x ∈ i ⁻¹' V₂, f x ∈ V' := by
    simpa [and_assoc] using
      ((nhds_basis_opens' b).comap i).tendsto_left_iff.mp (mem_of_mem_nhds V₁_in : b ∈ V₁) V' V'_in
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    i : α → β
    inst✝¹ : TopologicalSpace γ
    inst✝ : T3Space γ
    b : β
    f : α → γ
    di : IsDenseInducing i
    hf : Filter.Eventually (fun x => Exists fun c => Filter.Tendsto f (Filter.coma …
    φ : β → γ := di.extend f
    this : ∀ (b : β), (Filter.comap i (nhds b)).NeBot
    V' : Set γ
    V'_in : Membership.mem (nhds (φ b)) V'
    V'_closed : IsClosed V'
    V₁ : Set β := setOf fun x => Filter.Tendsto f (Filter.comap i (nhds x)) (nhds  …
    V₁_in : Membership.mem (nhds b) V₁
    V₂ : Set β
    V₂_in : Membership.mem (nhds b) V₂
    V₂_op : IsOpen V₂
    hV₂ : ∀ (x : α), Membership.mem (Set.preimage i V₂) x → Membership.mem V' (f x)
    ⊢ Membership.mem (nhds b) (Set.preimage φ V')
  -/
  suffices ∀ x ∈ V₁ ∩ V₂, φ x ∈ V' by filter_upwards [inter_mem V₁_in V₂_in] using this
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    i : α → β
    inst✝¹ : TopologicalSpace γ
    inst✝ : T3Space γ
    b : β
    f : α → γ
    di : IsDenseInducing i
    hf : Filter.Eventually (fun x => Exists fun c => Filter.Tendsto f (Filter.coma …
    φ : β → γ := di.extend f
    this : ∀ (b : β), (Filter.comap i (nhds b)).NeBot
    V' : Set γ
    V'_in : Membership.mem (nhds (φ b)) V'
    V'_closed : IsClosed V'
    V₁ : Set β := setOf fun x => Filter.Tendsto f (Filter.comap i (nhds x)) (nhds  …
    V₁_in : Membership.mem (nhds b) V₁
    V₂ : Set β
    V₂_in : Membership.mem (nhds b) V₂
    V₂_op : IsOpen V₂
    hV₂ : ∀ (x : α), Membership.mem (Set.preimage i V₂) x → Membership.mem V' (f x)
    ⊢ ∀ (x : β), Membership.mem (Inter.inter V₁ V₂) x → Membership.mem V' (φ x)
  -/
  rintro x ⟨x_in₁, x_in₂⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    i : α → β
    inst✝¹ : TopologicalSpace γ
    inst✝ : T3Space γ
    b : β
    f : α → γ
    di : IsDenseInducing i
    hf : Filter.Eventually (fun x => Exists fun c => Filter.Tendsto f (Filter.coma …
    φ : β → γ := di.extend f
    this : ∀ (b : β), (Filter.comap i (nhds b)).NeBot
    V' : Set γ
    V'_in : Membership.mem (nhds (φ b)) V'
    V'_closed : IsClosed V'
    V₁ : Set β := setOf fun x => Filter.Tendsto f (Filter.comap i (nhds x)) (nhds  …
    V₁_in : Membership.mem (nhds b) V₁
    V₂ : Set β
    V₂_in : Membership.mem (nhds b) V₂
    V₂_op : IsOpen V₂
    hV₂ : ∀ (x : α), Membership.mem (Set.preimage i V₂) x → Membership.mem V' (f x)
    x : β
    x_in₁ : Membership.mem V₁ x
    x_in₂ : Membership.mem V₂ x
    ⊢ Membership.mem V' (φ x)
  -/
  have hV₂x : V₂ ∈ 𝓝 x := IsOpen.mem_nhds V₂_op x_in₂
  /-
    case intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    i : α → β
    inst✝¹ : TopologicalSpace γ
    inst✝ : T3Space γ
    b : β
    f : α → γ
    di : IsDenseInducing i
    hf : Filter.Eventually (fun x => Exists fun c => Filter.Tendsto f (Filter.coma …
    φ : β → γ := di.extend f
    this : ∀ (b : β), (Filter.comap i (nhds b)).NeBot
    V' : Set γ
    V'_in : Membership.mem (nhds (φ b)) V'
    V'_closed : IsClosed V'
    V₁ : Set β := setOf fun x => Filter.Tendsto f (Filter.comap i (nhds x)) (nhds  …
    V₁_in : Membership.mem (nhds b) V₁
    V₂ : Set β
    V₂_in : Membership.mem (nhds b) V₂
    V₂_op : IsOpen V₂
    hV₂ : ∀ (x : α), Membership.mem (Set.preimage i V₂) x → Membership.mem V' (f x)
    x : β
    x_in₁ : Membership.mem V₁ x
    x_in₂ : Membership.mem V₂ x
    hV₂x : Membership.mem (nhds x) V₂
    ⊢ Membership.mem V' (φ x)
  -/
  apply V'_closed.mem_of_tendsto x_in₁
  /-
    case intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    i : α → β
    inst✝¹ : TopologicalSpace γ
    inst✝ : T3Space γ
    b : β
    f : α → γ
    di : IsDenseInducing i
    hf : Filter.Eventually (fun x => Exists fun c => Filter.Tendsto f (Filter.coma …
    φ : β → γ := di.extend f
    this : ∀ (b : β), (Filter.comap i (nhds b)).NeBot
    V' : Set γ
    V'_in : Membership.mem (nhds (φ b)) V'
    V'_closed : IsClosed V'
    V₁ : Set β := setOf fun x => Filter.Tendsto f (Filter.comap i (nhds x)) (nhds  …
    V₁_in : Membership.mem (nhds b) V₁
    V₂ : Set β
    V₂_in : Membership.mem (nhds b) V₂
    V₂_op : IsOpen V₂
    hV₂ : ∀ (x : α), Membership.mem (Set.preimage i V₂) x → Membership.mem V' (f x)
    x : β
    x_in₁ : Membership.mem V₁ x
    x_in₂ : Membership.mem V₂ x
    hV₂x : Membership.mem (nhds x) V₂
    ⊢ Filter.Eventually (fun x => Membership.mem V' (f x)) (Filter.comap i (nhds x))
  -/
  use V₂
  /-
    case h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace β
    i : α → β
    inst✝¹ : TopologicalSpace γ
    inst✝ : T3Space γ
    b : β
    f : α → γ
    di : IsDenseInducing i
    hf : Filter.Eventually (fun x => Exists fun c => Filter.Tendsto f (Filter.coma …
    φ : β → γ := di.extend f
    this : ∀ (b : β), (Filter.comap i (nhds b)).NeBot
    V' : Set γ
    V'_in : Membership.mem (nhds (φ b)) V'
    V'_closed : IsClosed V'
    V₁ : Set β := setOf fun x => Filter.Tendsto f (Filter.comap i (nhds x)) (nhds  …
    V₁_in : Membership.mem (nhds b) V₁
    V₂ : Set β
    V₂_in : Membership.mem (nhds b) V₂
    V₂_op : IsOpen V₂
    hV₂ : ∀ (x : α), Membership.mem (Set.preimage i V₂) x → Membership.mem V' (f x)
    x : β
    x_in₁ : Membership.mem V₁ x
    x_in₂ : Membership.mem V₂ x
    hV₂x : Membership.mem (nhds x) V₂
    ⊢ And (Membership.mem (nhds x) V₂) (HasSubset.Subset (Set.preimage i V₂) (setO …
  -/
  tauto
  /-
    🎉 no goals
  -/


theorem continuous_extend [T3Space γ] {f : α → γ} (di : IsDenseInducing i)
    (hf : ∀ b, ∃ c, Tendsto f (comap i (𝓝 b)) (𝓝 c)) : Continuous (di.extend f) :=
  continuous_iff_continuousAt.mpr fun _ => di.continuousAt_extend <| univ_mem' hf


theorem mk' (i : α → β) (c : Continuous i) (dense : ∀ x, x ∈ closure (range i))
    (H : ∀ (a : α), ∀ s ∈ 𝓝 a, ∃ t ∈ 𝓝 (i a), ∀ b, i b ∈ t → b ∈ s) : IsDenseInducing i where
  toIsInducing := isInducing_iff_nhds.2 fun a =>
                                             /-
                                               α : Type u_1
                                               β : Type u_2
                                               inst✝¹ : TopologicalSpace α
                                               inst✝ : TopologicalSpace β
                                               i : α → β
                                               c : Continuous i
                                               dense : ∀ (x : β), Membership.mem (closure (Set.range i)) x
                                               H : ∀ (a : α) (s : Set α), Membership.mem (nhds a) s → Exists fun t => And (Me …
                                               a : α
                                               ⊢ LE.le (Filter.comap i (nhds (i a))) (nhds a)
                                             -/
      le_antisymm (c.tendsto _).le_comap (by simpa [Filter.le_def] using H a)
                                             /-
                                               🎉 no goals
                                             -/
  dense := dense


/-- A dense embedding is an embedding with dense image. -/
structure IsDenseEmbedding [TopologicalSpace α] [TopologicalSpace β] (e : α → β) extends
  IsDenseInducing e : Prop where
  /-- A dense embedding is injective. -/
  injective : Function.Injective e


lemma IsDenseEmbedding.mk' [TopologicalSpace α] [TopologicalSpace β] (e : α → β) (c : Continuous e)
    (dense : DenseRange e) (injective : Function.Injective e)
    (H : ∀ (a : α), ∀ s ∈ 𝓝 a, ∃ t ∈ 𝓝 (e a), ∀ b, e b ∈ t → b ∈ s) : IsDenseEmbedding e :=
  { IsDenseInducing.mk' e c dense H with injective }


@[deprecated (since := "2024-09-30")]
alias DenseEmbedding.mk' := IsDenseEmbedding.mk'


lemma isDenseInducing (de : IsDenseEmbedding e) : IsDenseInducing e := de.toIsDenseInducing


theorem inj_iff (de : IsDenseEmbedding e) {x y} : e x = e y ↔ x = y :=
  de.injective.eq_iff


theorem isEmbedding (de : IsDenseEmbedding e) : IsEmbedding e where __ := de


@[deprecated (since := "2024-10-26")]
alias to_embedding := isEmbedding


/-- If the domain of a `IsDenseEmbedding` is a separable space, then so is its codomain. -/
protected theorem separableSpace [SeparableSpace α] (de : IsDenseEmbedding e) : SeparableSpace β :=
  de.isDenseInducing.separableSpace


/-- The product of two dense embeddings is a dense embedding. -/
protected theorem prodMap {e₁ : α → β} {e₂ : γ → δ} (de₁ : IsDenseEmbedding e₁)
    (de₂ : IsDenseEmbedding e₂) : IsDenseEmbedding fun p : α × γ => (e₁ p.1, e₂ p.2) where
  toIsDenseInducing := de₁.isDenseInducing.prodMap de₂.isDenseInducing
  injective := de₁.injective.prodMap de₂.injective


@[deprecated (since := "2024-10-06")] protected alias prod := IsDenseEmbedding.prodMap


/-- The dense embedding of a subtype inside its closure. -/
@[simps]
def subtypeEmb {α : Type*} (p : α → Prop) (e : α → β) (x : { x // p x }) :
    { x // x ∈ closure (e '' { x | p x }) } :=
  ⟨e x, subset_closure <| mem_image_of_mem e x.prop⟩


protected theorem subtype (de : IsDenseEmbedding e) (p : α → Prop) :
    IsDenseEmbedding (subtypeEmb p e) where
  dense :=
    dense_iff_closure_eq.2 <| by
      /-
        α : Type u_1
        β : Type u_2
        inst✝¹ : TopologicalSpace α
        inst✝ : TopologicalSpace β
        e : α → β
        de : IsDenseEmbedding e
        p : α → Prop
        ⊢ Eq (closure (Set.range (IsDenseEmbedding.subtypeEmb p e))) Set.univ
      -/
      ext ⟨x, hx⟩
      /-
        case h.mk
        α : Type u_1
        β : Type u_2
        inst✝¹ : TopologicalSpace α
        inst✝ : TopologicalSpace β
        e : α → β
        de : IsDenseEmbedding e
        p : α → Prop
        x : β
        hx : Membership.mem (closure (Set.image e (setOf fun x => p x))) x
        ⊢ Iff (Membership.mem (closure (Set.range (IsDenseEmbedding.subtypeEmb p e)))  …
      -/
      rw [image_eq_range] at hx
      /-
        case h.mk
        α : Type u_1
        β : Type u_2
        inst✝¹ : TopologicalSpace α
        inst✝ : TopologicalSpace β
        e : α → β
        de : IsDenseEmbedding e
        p : α → Prop
        x : β
        hx✝ : Membership.mem (closure (Set.image e (setOf fun x => p x))) x
        hx : Membership.mem (closure (Set.range fun x => e ↑x)) x
        ⊢ Iff (Membership.mem (closure (Set.range (IsDenseEmbedding.subtypeEmb p e)))  …
      -/
      simpa [closure_subtype, ← range_comp, (· ∘ ·)]
      /-
        🎉 no goals
      -/
  injective := (de.injective.comp Subtype.coe_injective).codRestrict _
  eq_induced :=
    (induced_iff_nhds_eq _).2 fun ⟨x, hx⟩ => by
      simp [subtypeEmb, nhds_subtype_eq_comap, de.isInducing.nhds_eq_comap, comap_comap,
        Function.comp_def]


theorem dense_image (de : IsDenseEmbedding e) {s : Set α} : Dense (e '' s) ↔ Dense s :=
  de.isDenseInducing.dense_image


protected lemma id {α : Type*} [TopologicalSpace α] : IsDenseEmbedding (id : α → α) :=
  { IsEmbedding.id with dense := denseRange_id }


@[deprecated (since := "2024-09-30")]
alias denseEmbedding_id := IsDenseEmbedding.id


theorem Dense.isDenseEmbedding_val [TopologicalSpace α] {s : Set α} (hs : Dense s) :
    IsDenseEmbedding ((↑) : s → α) :=
  { IsEmbedding.subtypeVal with dense := hs.denseRange_val }


@[deprecated (since := "2024-09-30")]
alias Dense.denseEmbedding_val := Dense.isDenseEmbedding_val


theorem isClosed_property [TopologicalSpace β] {e : α → β} {p : β → Prop} (he : DenseRange e)
    (hp : IsClosed { x | p x }) (h : ∀ a, p (e a)) : ∀ b, p b :=
  have : univ ⊆ { b | p b } :=
    calc
      univ = closure (range e) := he.closure_range.symm
      _ ⊆ closure { b | p b } := closure_mono <| range_subset_iff.mpr h
      _ = _ := hp.closure_eq

  fun _ => this trivial


theorem isClosed_property2 [TopologicalSpace β] {e : α → β} {p : β → β → Prop} (he : DenseRange e)
    (hp : IsClosed { q : β × β | p q.1 q.2 }) (h : ∀ a₁ a₂, p (e a₁) (e a₂)) : ∀ b₁ b₂, p b₁ b₂ :=
  have : ∀ q : β × β, p q.1 q.2 := isClosed_property (he.prodMap he) hp fun _ => h _ _
  fun b₁ b₂ => this ⟨b₁, b₂⟩


theorem isClosed_property3 [TopologicalSpace β] {e : α → β} {p : β → β → β → Prop}
    (he : DenseRange e) (hp : IsClosed { q : β × β × β | p q.1 q.2.1 q.2.2 })
    (h : ∀ a₁ a₂ a₃, p (e a₁) (e a₂) (e a₃)) : ∀ b₁ b₂ b₃, p b₁ b₂ b₃ :=
  have : ∀ q : β × β × β, p q.1 q.2.1 q.2.2 :=
    isClosed_property (he.prodMap <| he.prodMap he) hp fun _ => h _ _ _
  fun b₁ b₂ b₃ => this ⟨b₁, b₂, b₃⟩


@[elab_as_elim]
theorem DenseRange.induction_on [TopologicalSpace β] {e : α → β} (he : DenseRange e) {p : β → Prop}
    (b₀ : β) (hp : IsClosed { b | p b }) (ih : ∀ a : α, p <| e a) : p b₀ :=
  isClosed_property he hp ih b₀


@[elab_as_elim]
theorem DenseRange.induction_on₂ [TopologicalSpace β] {e : α → β} {p : β → β → Prop}
    (he : DenseRange e) (hp : IsClosed { q : β × β | p q.1 q.2 }) (h : ∀ a₁ a₂, p (e a₁) (e a₂))
    (b₁ b₂ : β) : p b₁ b₂ :=
  isClosed_property2 he hp h _ _


@[elab_as_elim]
theorem DenseRange.induction_on₃ [TopologicalSpace β] {e : α → β} {p : β → β → β → Prop}
    (he : DenseRange e) (hp : IsClosed { q : β × β × β | p q.1 q.2.1 q.2.2 })
    (h : ∀ a₁ a₂ a₃, p (e a₁) (e a₂) (e a₃)) (b₁ b₂ b₃ : β) : p b₁ b₂ b₃ :=
  isClosed_property3 he hp h _ _ _


/-- Two continuous functions to a t2-space that agree on the dense range of a function are equal. -/
theorem DenseRange.equalizer (hfd : DenseRange f) {g h : β → γ} (hg : Continuous g)
    (hh : Continuous h) (H : g ∘ f = h ∘ f) : g = h :=
  funext fun y => hfd.induction_on y (isClosed_eq hg hh) <| congr_fun H


theorem Filter.HasBasis.hasBasis_of_isDenseInducing [TopologicalSpace α] [TopologicalSpace β]
    [T3Space β] {ι : Type*} {s : ι → Set α} {p : ι → Prop} {x : α} (h : (𝓝 x).HasBasis p s)
    {f : α → β} (hf : IsDenseInducing f) : (𝓝 (f x)).HasBasis p fun i => closure <| f '' s i := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : T3Space β
    ι : Type u_5
    s : ι → Set α
    p : ι → Prop
    x : α
    h : (nhds x).HasBasis p s
    f : α → β
    hf : IsDenseInducing f
    ⊢ (nhds (f x)).HasBasis p fun i => closure (Set.image f (s i))
  -/
  rw [Filter.hasBasis_iff] at h ⊢
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : T3Space β
    ι : Type u_5
    s : ι → Set α
    p : ι → Prop
    x : α
    h : ∀ (t : Set α), Iff (Membership.mem (nhds x) t) (Exists fun i => And (p i)  …
    f : α → β
    hf : IsDenseInducing f
    ⊢ ∀ (t : Set β), Iff (Membership.mem (nhds (f x)) t) (Exists fun i => And (p i …
  -/
  intro T
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : TopologicalSpace β
    inst✝ : T3Space β
    ι : Type u_5
    s : ι → Set α
    p : ι → Prop
    x : α
    h : ∀ (t : Set α), Iff (Membership.mem (nhds x) t) (Exists fun i => And (p i)  …
    f : α → β
    hf : IsDenseInducing f
    T : Set β
    ⊢ Iff (Membership.mem (nhds (f x)) T) (Exists fun i => And (p i) (HasSubset.Su …
  -/
  refine ⟨fun hT => ?_, fun hT => ?_⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : T3Space β
      ι : Type u_5
      s : ι → Set α
      p : ι → Prop
      x : α
      h : ∀ (t : Set α), Iff (Membership.mem (nhds x) t) (Exists fun i => And (p i)  …
      f : α → β
      hf : IsDenseInducing f
      T : Set β
      hT : Membership.mem (nhds (f x)) T
      ⊢ Exists fun i => And (p i) (HasSubset.Subset (closure (Set.image f (s i))) T)
    -/
  · obtain ⟨T', hT₁, hT₂, hT₃⟩ := exists_mem_nhds_isClosed_subset hT
    have hT₄ : f ⁻¹' T' ∈ 𝓝 x := by
      rw [hf.isInducing.nhds_eq_comap x]
      exact ⟨T', hT₁, Subset.rfl⟩
    /-
      case refine_1.intro.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : T3Space β
      ι : Type u_5
      s : ι → Set α
      p : ι → Prop
      x : α
      h : ∀ (t : Set α), Iff (Membership.mem (nhds x) t) (Exists fun i => And (p i)  …
      f : α → β
      hf : IsDenseInducing f
      T : Set β
      hT : Membership.mem (nhds (f x)) T
      T' : Set β
      hT₁ : Membership.mem (nhds (f x)) T'
      hT₂ : IsClosed T'
      hT₃ : HasSubset.Subset T' T
      hT₄ : Membership.mem (nhds x) (Set.preimage f T')
      ⊢ Exists fun i => And (p i) (HasSubset.Subset (closure (Set.image f (s i))) T)
    -/
    obtain ⟨i, hi, hi'⟩ := (h _).mp hT₄
    exact
      ⟨i, hi,
        (closure_mono (image_subset f hi')).trans
          (Subset.trans (closure_minimal (image_preimage_subset _ _) hT₂) hT₃)⟩
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : T3Space β
      ι : Type u_5
      s : ι → Set α
      p : ι → Prop
      x : α
      h : ∀ (t : Set α), Iff (Membership.mem (nhds x) t) (Exists fun i => And (p i)  …
      f : α → β
      hf : IsDenseInducing f
      T : Set β
      hT : Exists fun i => And (p i) (HasSubset.Subset (closure (Set.image f (s i))) …
      ⊢ Membership.mem (nhds (f x)) T
    -/
  · obtain ⟨i, hi, hi'⟩ := hT
    /-
      case refine_2.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : T3Space β
      ι : Type u_5
      s : ι → Set α
      p : ι → Prop
      x : α
      h : ∀ (t : Set α), Iff (Membership.mem (nhds x) t) (Exists fun i => And (p i)  …
      f : α → β
      hf : IsDenseInducing f
      T : Set β
      i : ι
      hi : p i
      hi' : HasSubset.Subset (closure (Set.image f (s i))) T
      ⊢ Membership.mem (nhds (f x)) T
    -/
    suffices closure (f '' s i) ∈ 𝓝 (f x) by filter_upwards [this] using hi'
    /-
      case refine_2.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : T3Space β
      ι : Type u_5
      s : ι → Set α
      p : ι → Prop
      x : α
      h : ∀ (t : Set α), Iff (Membership.mem (nhds x) t) (Exists fun i => And (p i)  …
      f : α → β
      hf : IsDenseInducing f
      T : Set β
      i : ι
      hi : p i
      hi' : HasSubset.Subset (closure (Set.image f (s i))) T
      ⊢ Membership.mem (nhds (f x)) (closure (Set.image f (s i)))
    -/
    replace h := (h (s i)).mpr ⟨i, hi, Subset.rfl⟩
    /-
      case refine_2.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝² : TopologicalSpace α
      inst✝¹ : TopologicalSpace β
      inst✝ : T3Space β
      ι : Type u_5
      s : ι → Set α
      p : ι → Prop
      x : α
      f : α → β
      hf : IsDenseInducing f
      T : Set β
      i : ι
      hi : p i
      hi' : HasSubset.Subset (closure (Set.image f (s i))) T
      h : Membership.mem (nhds x) (s i)
      ⊢ Membership.mem (nhds (f x)) (closure (Set.image f (s i)))
    -/
    exact hf.closure_image_mem_nhds h
    /-
      🎉 no goals
    -/

