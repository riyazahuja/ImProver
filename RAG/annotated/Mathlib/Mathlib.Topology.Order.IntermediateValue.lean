/-- Intermediate value theorem for two functions: if `f` and `g` are two continuous functions
on a preconnected space and `f a ≤ g a` and `g b ≤ f b`, then for some `x` we have `f x = g x`. -/
theorem intermediate_value_univ₂ [PreconnectedSpace X] {a b : X} {f g : X → α} (hf : Continuous f)
    (hg : Continuous g) (ha : f a ≤ g a) (hb : g b ≤ f b) : ∃ x, f x = g x := by
  obtain ⟨x, _, hfg, hgf⟩ : (univ ∩ { x | f x ≤ g x ∧ g x ≤ f x }).Nonempty :=
    isPreconnected_closed_iff.1 PreconnectedSpace.isPreconnected_univ _ _ (isClosed_le hf hg)
      (isClosed_le hg hf) (fun _ _ => le_total _ _) ⟨a, trivial, ha⟩ ⟨b, trivial, hb⟩
  /-
    case intro.intro.intro
    X : Type u
    α : Type v
    inst✝⁴ : TopologicalSpace X
    inst✝³ : LinearOrder α
    inst✝² : TopologicalSpace α
    inst✝¹ : OrderClosedTopology α
    inst✝ : PreconnectedSpace X
    a b : X
    f g : X → α
    hf : Continuous f
    hg : Continuous g
    ha : LE.le (f a) (g a)
    hb : LE.le (g b) (f b)
    x : X
    left✝ : Membership.mem Set.univ x
    hfg : LE.le (f x) (g x)
    hgf : LE.le (g x) (f x)
    ⊢ Exists fun x => Eq (f x) (g x)
  -/
  exact ⟨x, le_antisymm hfg hgf⟩
  /-
    🎉 no goals
  -/


theorem intermediate_value_univ₂_eventually₁ [PreconnectedSpace X] {a : X} {l : Filter X} [NeBot l]
    {f g : X → α} (hf : Continuous f) (hg : Continuous g) (ha : f a ≤ g a) (he : g ≤ᶠ[l] f) :
    ∃ x, f x = g x :=
  let ⟨_, h⟩ := he.exists; intermediate_value_univ₂ hf hg ha h


theorem intermediate_value_univ₂_eventually₂ [PreconnectedSpace X] {l₁ l₂ : Filter X} [NeBot l₁]
    [NeBot l₂] {f g : X → α} (hf : Continuous f) (hg : Continuous g) (he₁ : f ≤ᶠ[l₁] g)
    (he₂ : g ≤ᶠ[l₂] f) : ∃ x, f x = g x :=
  let ⟨_, h₁⟩ := he₁.exists
  let ⟨_, h₂⟩ := he₂.exists
  intermediate_value_univ₂ hf hg h₁ h₂


/-- Intermediate value theorem for two functions: if `f` and `g` are two functions continuous
on a preconnected set `s` and for some `a b ∈ s` we have `f a ≤ g a` and `g b ≤ f b`,
then for some `x ∈ s` we have `f x = g x`. -/
theorem IsPreconnected.intermediate_value₂ {s : Set X} (hs : IsPreconnected s) {a b : X}
    (ha : a ∈ s) (hb : b ∈ s) {f g : X → α} (hf : ContinuousOn f s) (hg : ContinuousOn g s)
    (ha' : f a ≤ g a) (hb' : g b ≤ f b) : ∃ x ∈ s, f x = g x :=
  let ⟨x, hx⟩ :=
    @intermediate_value_univ₂ s α _ _ _ _ (Subtype.preconnectedSpace hs) ⟨a, ha⟩ ⟨b, hb⟩ _ _
      (continuousOn_iff_continuous_restrict.1 hf) (continuousOn_iff_continuous_restrict.1 hg) ha'
      hb'
  ⟨x, x.2, hx⟩


theorem IsPreconnected.intermediate_value₂_eventually₁ {s : Set X} (hs : IsPreconnected s) {a : X}
    {l : Filter X} (ha : a ∈ s) [NeBot l] (hl : l ≤ 𝓟 s) {f g : X → α} (hf : ContinuousOn f s)
    (hg : ContinuousOn g s) (ha' : f a ≤ g a) (he : g ≤ᶠ[l] f) : ∃ x ∈ s, f x = g x := by
  /-
    X : Type u
    α : Type v
    inst✝⁴ : TopologicalSpace X
    inst✝³ : LinearOrder α
    inst✝² : TopologicalSpace α
    inst✝¹ : OrderClosedTopology α
    s : Set X
    hs : IsPreconnected s
    a : X
    l : Filter X
    ha : Membership.mem s a
    inst✝ : l.NeBot
    hl : LE.le l (Filter.principal s)
    f g : X → α
    hf : ContinuousOn f s
    hg : ContinuousOn g s
    ha' : LE.le (f a) (g a)
    he : l.EventuallyLE g f
    ⊢ Exists fun x => And (Membership.mem s x) (Eq (f x) (g x))
  -/
  rw [continuousOn_iff_continuous_restrict] at hf hg
  obtain ⟨b, h⟩ :=
    @intermediate_value_univ₂_eventually₁ _ _ _ _ _ _ (Subtype.preconnectedSpace hs) ⟨a, ha⟩ _
      (comap_coe_neBot_of_le_principal hl) _ _ hf hg ha' (he.comap _)
  /-
    case intro
    X : Type u
    α : Type v
    inst✝⁴ : TopologicalSpace X
    inst✝³ : LinearOrder α
    inst✝² : TopologicalSpace α
    inst✝¹ : OrderClosedTopology α
    s : Set X
    hs : IsPreconnected s
    a : X
    l : Filter X
    ha : Membership.mem s a
    inst✝ : l.NeBot
    hl : LE.le l (Filter.principal s)
    f g : X → α
    hf : Continuous (s.restrict f)
    hg : Continuous (s.restrict g)
    ha' : LE.le (f a) (g a)
    he : l.EventuallyLE g f
    b : ↑s
    h : Eq (s.restrict f b) (s.restrict g b)
    ⊢ Exists fun x => And (Membership.mem s x) (Eq (f x) (g x))
  -/
  exact ⟨b, b.prop, h⟩
  /-
    🎉 no goals
  -/


theorem IsPreconnected.intermediate_value₂_eventually₂ {s : Set X} (hs : IsPreconnected s)
    {l₁ l₂ : Filter X} [NeBot l₁] [NeBot l₂] (hl₁ : l₁ ≤ 𝓟 s) (hl₂ : l₂ ≤ 𝓟 s) {f g : X → α}
    (hf : ContinuousOn f s) (hg : ContinuousOn g s) (he₁ : f ≤ᶠ[l₁] g) (he₂ : g ≤ᶠ[l₂] f) :
    ∃ x ∈ s, f x = g x := by
  /-
    X : Type u
    α : Type v
    inst✝⁵ : TopologicalSpace X
    inst✝⁴ : LinearOrder α
    inst✝³ : TopologicalSpace α
    inst✝² : OrderClosedTopology α
    s : Set X
    hs : IsPreconnected s
    l₁ l₂ : Filter X
    inst✝¹ : l₁.NeBot
    inst✝ : l₂.NeBot
    hl₁ : LE.le l₁ (Filter.principal s)
    hl₂ : LE.le l₂ (Filter.principal s)
    f g : X → α
    hf : ContinuousOn f s
    hg : ContinuousOn g s
    he₁ : l₁.EventuallyLE f g
    he₂ : l₂.EventuallyLE g f
    ⊢ Exists fun x => And (Membership.mem s x) (Eq (f x) (g x))
  -/
  rw [continuousOn_iff_continuous_restrict] at hf hg
  obtain ⟨b, h⟩ :=
    @intermediate_value_univ₂_eventually₂ _ _ _ _ _ _ (Subtype.preconnectedSpace hs) _ _
      (comap_coe_neBot_of_le_principal hl₁) (comap_coe_neBot_of_le_principal hl₂) _ _ hf hg
      (he₁.comap _) (he₂.comap _)
  /-
    case intro
    X : Type u
    α : Type v
    inst✝⁵ : TopologicalSpace X
    inst✝⁴ : LinearOrder α
    inst✝³ : TopologicalSpace α
    inst✝² : OrderClosedTopology α
    s : Set X
    hs : IsPreconnected s
    l₁ l₂ : Filter X
    inst✝¹ : l₁.NeBot
    inst✝ : l₂.NeBot
    hl₁ : LE.le l₁ (Filter.principal s)
    hl₂ : LE.le l₂ (Filter.principal s)
    f g : X → α
    hf : Continuous (s.restrict f)
    hg : Continuous (s.restrict g)
    he₁ : l₁.EventuallyLE f g
    he₂ : l₂.EventuallyLE g f
    b : ↑s
    h : Eq (s.restrict f b) (s.restrict g b)
    ⊢ Exists fun x => And (Membership.mem s x) (Eq (f x) (g x))
  -/
  exact ⟨b, b.prop, h⟩
  /-
    🎉 no goals
  -/


/-- **Intermediate Value Theorem** for continuous functions on connected sets. -/
theorem IsPreconnected.intermediate_value {s : Set X} (hs : IsPreconnected s) {a b : X} (ha : a ∈ s)
    (hb : b ∈ s) {f : X → α} (hf : ContinuousOn f s) : Icc (f a) (f b) ⊆ f '' s := fun _x hx =>
  hs.intermediate_value₂ ha hb hf continuousOn_const hx.1 hx.2


theorem IsPreconnected.intermediate_value_Ico {s : Set X} (hs : IsPreconnected s) {a : X}
    {l : Filter X} (ha : a ∈ s) [NeBot l] (hl : l ≤ 𝓟 s) {f : X → α} (hf : ContinuousOn f s) {v : α}
    (ht : Tendsto f l (𝓝 v)) : Ico (f a) v ⊆ f '' s := fun _ h =>
  hs.intermediate_value₂_eventually₁ ha hl hf continuousOn_const h.1 (ht.eventually_const_le h.2)


theorem IsPreconnected.intermediate_value_Ioc {s : Set X} (hs : IsPreconnected s) {a : X}
    {l : Filter X} (ha : a ∈ s) [NeBot l] (hl : l ≤ 𝓟 s) {f : X → α} (hf : ContinuousOn f s) {v : α}
    (ht : Tendsto f l (𝓝 v)) : Ioc v (f a) ⊆ f '' s := fun _ h =>
  (hs.intermediate_value₂_eventually₁ ha hl continuousOn_const hf h.2
    (ht.eventually_le_const h.1)).imp fun _ h => h.imp_right Eq.symm


theorem IsPreconnected.intermediate_value_Ioo {s : Set X} (hs : IsPreconnected s) {l₁ l₂ : Filter X}
    [NeBot l₁] [NeBot l₂] (hl₁ : l₁ ≤ 𝓟 s) (hl₂ : l₂ ≤ 𝓟 s) {f : X → α} (hf : ContinuousOn f s)
    {v₁ v₂ : α} (ht₁ : Tendsto f l₁ (𝓝 v₁)) (ht₂ : Tendsto f l₂ (𝓝 v₂)) :
    Ioo v₁ v₂ ⊆ f '' s := fun _ h =>
  hs.intermediate_value₂_eventually₂ hl₁ hl₂ hf continuousOn_const
    (ht₁.eventually_le_const h.1) (ht₂.eventually_const_le h.2)


theorem IsPreconnected.intermediate_value_Ici {s : Set X} (hs : IsPreconnected s) {a : X}
    {l : Filter X} (ha : a ∈ s) [NeBot l] (hl : l ≤ 𝓟 s) {f : X → α} (hf : ContinuousOn f s)
    (ht : Tendsto f l atTop) : Ici (f a) ⊆ f '' s := fun y h =>
  hs.intermediate_value₂_eventually₁ ha hl hf continuousOn_const h (tendsto_atTop.1 ht y)


theorem IsPreconnected.intermediate_value_Iic {s : Set X} (hs : IsPreconnected s) {a : X}
    {l : Filter X} (ha : a ∈ s) [NeBot l] (hl : l ≤ 𝓟 s) {f : X → α} (hf : ContinuousOn f s)
    (ht : Tendsto f l atBot) : Iic (f a) ⊆ f '' s := fun y h =>
  (hs.intermediate_value₂_eventually₁ ha hl continuousOn_const hf h (tendsto_atBot.1 ht y)).imp
    fun _ h => h.imp_right Eq.symm


theorem IsPreconnected.intermediate_value_Ioi {s : Set X} (hs : IsPreconnected s) {l₁ l₂ : Filter X}
    [NeBot l₁] [NeBot l₂] (hl₁ : l₁ ≤ 𝓟 s) (hl₂ : l₂ ≤ 𝓟 s) {f : X → α} (hf : ContinuousOn f s)
    {v : α} (ht₁ : Tendsto f l₁ (𝓝 v)) (ht₂ : Tendsto f l₂ atTop) : Ioi v ⊆ f '' s := fun y h =>
  hs.intermediate_value₂_eventually₂ hl₁ hl₂ hf continuousOn_const
    (ht₁.eventually_le_const h) (ht₂.eventually_ge_atTop y)


theorem IsPreconnected.intermediate_value_Iio {s : Set X} (hs : IsPreconnected s) {l₁ l₂ : Filter X}
    [NeBot l₁] [NeBot l₂] (hl₁ : l₁ ≤ 𝓟 s) (hl₂ : l₂ ≤ 𝓟 s) {f : X → α} (hf : ContinuousOn f s)
    {v : α} (ht₁ : Tendsto f l₁ atBot) (ht₂ : Tendsto f l₂ (𝓝 v)) : Iio v ⊆ f '' s := fun y h =>
  hs.intermediate_value₂_eventually₂ hl₁ hl₂ hf continuousOn_const (ht₁.eventually_le_atBot y)
    (ht₂.eventually_const_le h)


theorem IsPreconnected.intermediate_value_Iii {s : Set X} (hs : IsPreconnected s) {l₁ l₂ : Filter X}
    [NeBot l₁] [NeBot l₂] (hl₁ : l₁ ≤ 𝓟 s) (hl₂ : l₂ ≤ 𝓟 s) {f : X → α} (hf : ContinuousOn f s)
    (ht₁ : Tendsto f l₁ atBot) (ht₂ : Tendsto f l₂ atTop) : univ ⊆ f '' s := fun y _ =>
  hs.intermediate_value₂_eventually₂ hl₁ hl₂ hf continuousOn_const (ht₁.eventually_le_atBot y)
    (ht₂.eventually_ge_atTop y)


/-- **Intermediate Value Theorem** for continuous functions on connected spaces. -/
theorem intermediate_value_univ [PreconnectedSpace X] (a b : X) {f : X → α} (hf : Continuous f) :
    Icc (f a) (f b) ⊆ range f := fun _ hx => intermediate_value_univ₂ hf continuous_const hx.1 hx.2


/-- **Intermediate Value Theorem** for continuous functions on connected spaces. -/
theorem mem_range_of_exists_le_of_exists_ge [PreconnectedSpace X] {c : α} {f : X → α}
    (hf : Continuous f) (h₁ : ∃ a, f a ≤ c) (h₂ : ∃ b, c ≤ f b) : c ∈ range f :=
  let ⟨a, ha⟩ := h₁; let ⟨b, hb⟩ := h₂; intermediate_value_univ a b hf ⟨ha, hb⟩


/-- If a preconnected set contains endpoints of an interval, then it includes the whole interval. -/
theorem IsPreconnected.Icc_subset {s : Set α} (hs : IsPreconnected s) {a b : α} (ha : a ∈ s)
    (hb : b ∈ s) : Icc a b ⊆ s := by
  /-
    α : Type v
    inst✝² : LinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderClosedTopology α
    s : Set α
    hs : IsPreconnected s
    a b : α
    ha : Membership.mem s a
    hb : Membership.mem s b
    ⊢ HasSubset.Subset (Set.Icc a b) s
  -/
  simpa only [image_id] using hs.intermediate_value ha hb continuousOn_id
  /-
    🎉 no goals
  -/


theorem IsPreconnected.ordConnected {s : Set α} (h : IsPreconnected s) : OrdConnected s :=
  ⟨fun _ hx _ hy => h.Icc_subset hx hy⟩


/-- If a preconnected set contains endpoints of an interval, then it includes the whole interval. -/
theorem IsConnected.Icc_subset {s : Set α} (hs : IsConnected s) {a b : α} (ha : a ∈ s)
    (hb : b ∈ s) : Icc a b ⊆ s :=
  hs.2.Icc_subset ha hb


/-- If preconnected set in a linear order space is unbounded below and above, then it is the whole
space. -/
theorem IsPreconnected.eq_univ_of_unbounded {s : Set α} (hs : IsPreconnected s) (hb : ¬BddBelow s)
    (ha : ¬BddAbove s) : s = univ := by
  /-
    α : Type v
    inst✝² : LinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderClosedTopology α
    s : Set α
    hs : IsPreconnected s
    hb : Not (BddBelow s)
    ha : Not (BddAbove s)
    ⊢ Eq s Set.univ
  -/
  refine eq_univ_of_forall fun x => ?_
  /-
    α : Type v
    inst✝² : LinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderClosedTopology α
    s : Set α
    hs : IsPreconnected s
    hb : Not (BddBelow s)
    ha : Not (BddAbove s)
    x : α
    ⊢ Membership.mem s x
  -/
  obtain ⟨y, ys, hy⟩ : ∃ y ∈ s, y < x := not_bddBelow_iff.1 hb x
  /-
    case intro.intro
    α : Type v
    inst✝² : LinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderClosedTopology α
    s : Set α
    hs : IsPreconnected s
    hb : Not (BddBelow s)
    ha : Not (BddAbove s)
    x y : α
    ys : Membership.mem s y
    hy : LT.lt y x
    ⊢ Membership.mem s x
  -/
  obtain ⟨z, zs, hz⟩ : ∃ z ∈ s, x < z := not_bddAbove_iff.1 ha x
  /-
    case intro.intro.intro.intro
    α : Type v
    inst✝² : LinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderClosedTopology α
    s : Set α
    hs : IsPreconnected s
    hb : Not (BddBelow s)
    ha : Not (BddAbove s)
    x y : α
    ys : Membership.mem s y
    hy : LT.lt y x
    z : α
    zs : Membership.mem s z
    hz : LT.lt x z
    ⊢ Membership.mem s x
  -/
  exact hs.Icc_subset ys zs ⟨le_of_lt hy, le_of_lt hz⟩
  /-
    🎉 no goals
  -/


/-- A bounded connected subset of a conditionally complete linear order includes the open interval
`(Inf s, Sup s)`. -/
theorem IsConnected.Ioo_csInf_csSup_subset {s : Set α} (hs : IsConnected s) (hb : BddBelow s)
    (ha : BddAbove s) : Ioo (sInf s) (sSup s) ⊆ s := fun _x hx =>
  let ⟨_y, ys, hy⟩ := (isGLB_lt_iff (isGLB_csInf hs.nonempty hb)).1 hx.1
  let ⟨_z, zs, hz⟩ := (lt_isLUB_iff (isLUB_csSup hs.nonempty ha)).1 hx.2
  hs.Icc_subset ys zs ⟨hy.le, hz.le⟩


theorem eq_Icc_csInf_csSup_of_connected_bdd_closed {s : Set α} (hc : IsConnected s)
    (hb : BddBelow s) (ha : BddAbove s) (hcl : IsClosed s) : s = Icc (sInf s) (sSup s) :=
  (subset_Icc_csInf_csSup hb ha).antisymm <|
    hc.Icc_subset (hcl.csInf_mem hc.nonempty hb) (hcl.csSup_mem hc.nonempty ha)


theorem IsPreconnected.Ioi_csInf_subset {s : Set α} (hs : IsPreconnected s) (hb : BddBelow s)
    (ha : ¬BddAbove s) : Ioi (sInf s) ⊆ s := fun x hx =>
  have sne : s.Nonempty := nonempty_of_not_bddAbove ha
  let ⟨_y, ys, hy⟩ : ∃ y ∈ s, y < x := (isGLB_lt_iff (isGLB_csInf sne hb)).1 hx
  let ⟨_z, zs, hz⟩ : ∃ z ∈ s, x < z := not_bddAbove_iff.1 ha x
  hs.Icc_subset ys zs ⟨hy.le, hz.le⟩


theorem IsPreconnected.Iio_csSup_subset {s : Set α} (hs : IsPreconnected s) (hb : ¬BddBelow s)
    (ha : BddAbove s) : Iio (sSup s) ⊆ s :=
  IsPreconnected.Ioi_csInf_subset (α := αᵒᵈ) hs ha hb


/-- A preconnected set in a conditionally complete linear order is either one of the intervals
`[Inf s, Sup s]`, `[Inf s, Sup s)`, `(Inf s, Sup s]`, `(Inf s, Sup s)`, `[Inf s, +∞)`,
`(Inf s, +∞)`, `(-∞, Sup s]`, `(-∞, Sup s)`, `(-∞, +∞)`, or `∅`. The converse statement requires
`α` to be densely ordered. -/
theorem IsPreconnected.mem_intervals {s : Set α} (hs : IsPreconnected s) :
    s ∈
      ({Icc (sInf s) (sSup s), Ico (sInf s) (sSup s), Ioc (sInf s) (sSup s), Ioo (sInf s) (sSup s),
          Ici (sInf s), Ioi (sInf s), Iic (sSup s), Iio (sSup s), univ, ∅} : Set (Set α)) := by
  /-
    α : Type u
    inst✝² : ConditionallyCompleteLinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    s : Set α
    hs : IsPreconnected s
    ⊢ Membership.mem (Insert.insert (Set.Icc (InfSet.sInf s) (SupSet.sSup s)) (Ins …
  -/
  rcases s.eq_empty_or_nonempty with (rfl | hne)
    /-
      case inl
      α : Type u
      inst✝² : ConditionallyCompleteLinearOrder α
      inst✝¹ : TopologicalSpace α
      inst✝ : OrderTopology α
      hs : IsPreconnected EmptyCollection.emptyCollection
      ⊢ Membership.mem (Insert.insert (Set.Icc (InfSet.sInf EmptyCollection.emptyCol …
    -/
  · apply_rules [Or.inr, mem_singleton]
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u
    inst✝² : ConditionallyCompleteLinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    s : Set α
    hs : IsPreconnected s
    hne : s.Nonempty
    ⊢ Membership.mem (Insert.insert (Set.Icc (InfSet.sInf s) (SupSet.sSup s)) (Ins …
  -/
  have hs' : IsConnected s := ⟨hne, hs⟩
  /-
    case inr
    α : Type u
    inst✝² : ConditionallyCompleteLinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    s : Set α
    hs : IsPreconnected s
    hne : s.Nonempty
    hs' : IsConnected s
    ⊢ Membership.mem (Insert.insert (Set.Icc (InfSet.sInf s) (SupSet.sSup s)) (Ins …
  -/
  by_cases hb : BddBelow s <;> by_cases ha : BddAbove s
  · refine mem_of_subset_of_mem ?_ <| mem_Icc_Ico_Ioc_Ioo_of_subset_of_subset
      (hs'.Ioo_csInf_csSup_subset hb ha) (subset_Icc_csInf_csSup hb ha)
    simp only [insert_subset_iff, mem_insert_iff, mem_singleton_iff, true_or, or_true,
      singleton_subset_iff, and_self]
    /-
      case neg
      α : Type u
      inst✝² : ConditionallyCompleteLinearOrder α
      inst✝¹ : TopologicalSpace α
      inst✝ : OrderTopology α
      s : Set α
      hs : IsPreconnected s
      hne : s.Nonempty
      hs' : IsConnected s
      hb : BddBelow s
      ha : Not (BddAbove s)
      ⊢ Membership.mem (Insert.insert (Set.Icc (InfSet.sInf s) (SupSet.sSup s)) (Ins …
    -/
  · refine Or.inr <| Or.inr <| Or.inr <| Or.inr ?_
    cases'
      mem_Ici_Ioi_of_subset_of_subset (hs.Ioi_csInf_subset hb ha) fun x hx => csInf_le hb hx with
      hs hs
      /-
        case neg.inl
        α : Type u
        inst✝² : ConditionallyCompleteLinearOrder α
        inst✝¹ : TopologicalSpace α
        inst✝ : OrderTopology α
        s : Set α
        hs✝ : IsPreconnected s
        hne : s.Nonempty
        hs' : IsConnected s
        hb : BddBelow s
        ha : Not (BddAbove s)
        hs : Eq s (Set.Ici (InfSet.sInf s))
        ⊢ Membership.mem (Insert.insert (Set.Ici (InfSet.sInf s)) (Insert.insert (Set. …
      -/
    · exact Or.inl hs
      /-
        🎉 no goals
      -/
      /-
        case neg.inr
        α : Type u
        inst✝² : ConditionallyCompleteLinearOrder α
        inst✝¹ : TopologicalSpace α
        inst✝ : OrderTopology α
        s : Set α
        hs✝ : IsPreconnected s
        hne : s.Nonempty
        hs' : IsConnected s
        hb : BddBelow s
        ha : Not (BddAbove s)
        hs : Membership.mem (Singleton.singleton (Set.Ioi (InfSet.sInf s))) s
        ⊢ Membership.mem (Insert.insert (Set.Ici (InfSet.sInf s)) (Insert.insert (Set. …
      -/
    · exact Or.inr (Or.inl hs)
      /-
        🎉 no goals
      -/
    /-
      case pos
      α : Type u
      inst✝² : ConditionallyCompleteLinearOrder α
      inst✝¹ : TopologicalSpace α
      inst✝ : OrderTopology α
      s : Set α
      hs : IsPreconnected s
      hne : s.Nonempty
      hs' : IsConnected s
      hb : Not (BddBelow s)
      ha : BddAbove s
      ⊢ Membership.mem (Insert.insert (Set.Icc (InfSet.sInf s) (SupSet.sSup s)) (Ins …
    -/
  · iterate 6 apply Or.inr
    cases' mem_Iic_Iio_of_subset_of_subset (hs.Iio_csSup_subset hb ha) fun x hx => le_csSup ha hx
      with hs hs
      /-
        case pos.h.h.h.h.h.h.inl
        α : Type u
        inst✝² : ConditionallyCompleteLinearOrder α
        inst✝¹ : TopologicalSpace α
        inst✝ : OrderTopology α
        s : Set α
        hs✝ : IsPreconnected s
        hne : s.Nonempty
        hs' : IsConnected s
        hb : Not (BddBelow s)
        ha : BddAbove s
        hs : Eq s (Set.Iic (SupSet.sSup s))
        ⊢ Membership.mem (Insert.insert (Set.Iic (SupSet.sSup s)) (Insert.insert (Set. …
      -/
    · exact Or.inl hs
      /-
        🎉 no goals
      -/
      /-
        case pos.h.h.h.h.h.h.inr
        α : Type u
        inst✝² : ConditionallyCompleteLinearOrder α
        inst✝¹ : TopologicalSpace α
        inst✝ : OrderTopology α
        s : Set α
        hs✝ : IsPreconnected s
        hne : s.Nonempty
        hs' : IsConnected s
        hb : Not (BddBelow s)
        ha : BddAbove s
        hs : Membership.mem (Singleton.singleton (Set.Iio (SupSet.sSup s))) s
        ⊢ Membership.mem (Insert.insert (Set.Iic (SupSet.sSup s)) (Insert.insert (Set. …
      -/
    · exact Or.inr (Or.inl hs)
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u
      inst✝² : ConditionallyCompleteLinearOrder α
      inst✝¹ : TopologicalSpace α
      inst✝ : OrderTopology α
      s : Set α
      hs : IsPreconnected s
      hne : s.Nonempty
      hs' : IsConnected s
      hb : Not (BddBelow s)
      ha : Not (BddAbove s)
      ⊢ Membership.mem (Insert.insert (Set.Icc (InfSet.sInf s) (SupSet.sSup s)) (Ins …
    -/
  · iterate 8 apply Or.inr
    /-
      case neg.h.h.h.h.h.h.h.h
      α : Type u
      inst✝² : ConditionallyCompleteLinearOrder α
      inst✝¹ : TopologicalSpace α
      inst✝ : OrderTopology α
      s : Set α
      hs : IsPreconnected s
      hne : s.Nonempty
      hs' : IsConnected s
      hb : Not (BddBelow s)
      ha : Not (BddAbove s)
      ⊢ Membership.mem (Insert.insert Set.univ (Singleton.singleton EmptyCollection. …
    -/
    exact Or.inl (hs.eq_univ_of_unbounded hb ha)
    /-
      🎉 no goals
    -/


/-- A preconnected set is either one of the intervals `Icc`, `Ico`, `Ioc`, `Ioo`, `Ici`, `Ioi`,
`Iic`, `Iio`, or `univ`, or `∅`. The converse statement requires `α` to be densely ordered. Though
one can represent `∅` as `(Inf ∅, Inf ∅)`, we include it into the list of possible cases to improve
readability. -/
theorem setOf_isPreconnected_subset_of_ordered :
    { s : Set α | IsPreconnected s } ⊆
      -- bounded intervals
      (range (uncurry Icc) ∪ range (uncurry Ico) ∪ range (uncurry Ioc) ∪ range (uncurry Ioo)) ∪
      -- unbounded intervals and `univ`
      (range Ici ∪ range Ioi ∪ range Iic ∪ range Iio ∪ {univ, ∅}) := by
  /-
    α : Type u
    inst✝² : ConditionallyCompleteLinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    ⊢ HasSubset.Subset (setOf fun s => IsPreconnected s) (Union.union (Union.union …
  -/
  intro s hs
  /-
    α : Type u
    inst✝² : ConditionallyCompleteLinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    s : Set α
    hs : Membership.mem (setOf fun s => IsPreconnected s) s
    ⊢ Membership.mem (Union.union (Union.union (Union.union (Union.union (Set.rang …
  -/
  rcases hs.mem_intervals with (hs | hs | hs | hs | hs | hs | hs | hs | hs | hs) <;> rw [hs] <;>
    simp only [union_insert, union_singleton, mem_insert_iff, mem_union, mem_range, Prod.exists,
      uncurry_apply_pair, exists_apply_eq_apply, true_or, or_true, exists_apply_eq_apply2]


/-- A "continuous induction principle" for a closed interval: if a set `s` meets `[a, b]`
on a closed subset, contains `a`, and the set `s ∩ [a, b)` has no maximal point, then `b ∈ s`. -/
theorem IsClosed.mem_of_ge_of_forall_exists_gt {a b : α} {s : Set α} (hs : IsClosed (s ∩ Icc a b))
    (ha : a ∈ s) (hab : a ≤ b) (hgt : ∀ x ∈ s ∩ Ico a b, (s ∩ Ioc x b).Nonempty) : b ∈ s := by
  /-
    α : Type u
    inst✝² : ConditionallyCompleteLinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    a b : α
    s : Set α
    hs : IsClosed (Inter.inter s (Set.Icc a b))
    ha : Membership.mem s a
    hab : LE.le a b
    hgt : ∀ (x : α), Membership.mem (Inter.inter s (Set.Ico a b)) x → (Inter.inter …
    ⊢ Membership.mem s b
  -/
  let S := s ∩ Icc a b
  /-
    α : Type u
    inst✝² : ConditionallyCompleteLinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    a b : α
    s : Set α
    hs : IsClosed (Inter.inter s (Set.Icc a b))
    ha : Membership.mem s a
    hab : LE.le a b
    hgt : ∀ (x : α), Membership.mem (Inter.inter s (Set.Ico a b)) x → (Inter.inter …
    S : Set α := Inter.inter s (Set.Icc a b)
    ⊢ Membership.mem s b
  -/
  replace ha : a ∈ S := ⟨ha, left_mem_Icc.2 hab⟩
  /-
    α : Type u
    inst✝² : ConditionallyCompleteLinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    a b : α
    s : Set α
    hs : IsClosed (Inter.inter s (Set.Icc a b))
    hab : LE.le a b
    hgt : ∀ (x : α), Membership.mem (Inter.inter s (Set.Ico a b)) x → (Inter.inter …
    S : Set α := Inter.inter s (Set.Icc a b)
    ha : Membership.mem S a
    ⊢ Membership.mem s b
  -/
  have Sbd : BddAbove S := ⟨b, fun z hz => hz.2.2⟩
  /-
    α : Type u
    inst✝² : ConditionallyCompleteLinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    a b : α
    s : Set α
    hs : IsClosed (Inter.inter s (Set.Icc a b))
    hab : LE.le a b
    hgt : ∀ (x : α), Membership.mem (Inter.inter s (Set.Ico a b)) x → (Inter.inter …
    S : Set α := Inter.inter s (Set.Icc a b)
    ha : Membership.mem S a
    Sbd : BddAbove S
    ⊢ Membership.mem s b
  -/
  let c := sSup (s ∩ Icc a b)
  /-
    α : Type u
    inst✝² : ConditionallyCompleteLinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    a b : α
    s : Set α
    hs : IsClosed (Inter.inter s (Set.Icc a b))
    hab : LE.le a b
    hgt : ∀ (x : α), Membership.mem (Inter.inter s (Set.Ico a b)) x → (Inter.inter …
    S : Set α := Inter.inter s (Set.Icc a b)
    ha : Membership.mem S a
    Sbd : BddAbove S
    c : α := SupSet.sSup (Inter.inter s (Set.Icc a b))
    ⊢ Membership.mem s b
  -/
  have c_mem : c ∈ S := hs.csSup_mem ⟨_, ha⟩ Sbd
  /-
    α : Type u
    inst✝² : ConditionallyCompleteLinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    a b : α
    s : Set α
    hs : IsClosed (Inter.inter s (Set.Icc a b))
    hab : LE.le a b
    hgt : ∀ (x : α), Membership.mem (Inter.inter s (Set.Ico a b)) x → (Inter.inter …
    S : Set α := Inter.inter s (Set.Icc a b)
    ha : Membership.mem S a
    Sbd : BddAbove S
    c : α := SupSet.sSup (Inter.inter s (Set.Icc a b))
    c_mem : Membership.mem S c
    ⊢ Membership.mem s b
  -/
  have c_le : c ≤ b := csSup_le ⟨_, ha⟩ fun x hx => hx.2.2
  /-
    α : Type u
    inst✝² : ConditionallyCompleteLinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    a b : α
    s : Set α
    hs : IsClosed (Inter.inter s (Set.Icc a b))
    hab : LE.le a b
    hgt : ∀ (x : α), Membership.mem (Inter.inter s (Set.Ico a b)) x → (Inter.inter …
    S : Set α := Inter.inter s (Set.Icc a b)
    ha : Membership.mem S a
    Sbd : BddAbove S
    c : α := SupSet.sSup (Inter.inter s (Set.Icc a b))
    c_mem : Membership.mem S c
    c_le : LE.le c b
    ⊢ Membership.mem s b
  -/
  cases' eq_or_lt_of_le c_le with hc hc
    /-
      case inl
      α : Type u
      inst✝² : ConditionallyCompleteLinearOrder α
      inst✝¹ : TopologicalSpace α
      inst✝ : OrderTopology α
      a b : α
      s : Set α
      hs : IsClosed (Inter.inter s (Set.Icc a b))
      hab : LE.le a b
      hgt : ∀ (x : α), Membership.mem (Inter.inter s (Set.Ico a b)) x → (Inter.inter …
      S : Set α := Inter.inter s (Set.Icc a b)
      ha : Membership.mem S a
      Sbd : BddAbove S
      c : α := SupSet.sSup (Inter.inter s (Set.Icc a b))
      c_mem : Membership.mem S c
      c_le : LE.le c b
      hc : Eq c b
      ⊢ Membership.mem s b
    -/
  · exact hc ▸ c_mem.1
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u
    inst✝² : ConditionallyCompleteLinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    a b : α
    s : Set α
    hs : IsClosed (Inter.inter s (Set.Icc a b))
    hab : LE.le a b
    hgt : ∀ (x : α), Membership.mem (Inter.inter s (Set.Ico a b)) x → (Inter.inter …
    S : Set α := Inter.inter s (Set.Icc a b)
    ha : Membership.mem S a
    Sbd : BddAbove S
    c : α := SupSet.sSup (Inter.inter s (Set.Icc a b))
    c_mem : Membership.mem S c
    c_le : LE.le c b
    hc : LT.lt c b
    ⊢ Membership.mem s b
  -/
  exfalso
  /-
    case inr
    α : Type u
    inst✝² : ConditionallyCompleteLinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    a b : α
    s : Set α
    hs : IsClosed (Inter.inter s (Set.Icc a b))
    hab : LE.le a b
    hgt : ∀ (x : α), Membership.mem (Inter.inter s (Set.Ico a b)) x → (Inter.inter …
    S : Set α := Inter.inter s (Set.Icc a b)
    ha : Membership.mem S a
    Sbd : BddAbove S
    c : α := SupSet.sSup (Inter.inter s (Set.Icc a b))
    c_mem : Membership.mem S c
    c_le : LE.le c b
    hc : LT.lt c b
    ⊢ False
  -/
  rcases hgt c ⟨c_mem.1, c_mem.2.1, hc⟩ with ⟨x, xs, cx, xb⟩
  /-
    case inr.intro.intro.intro
    α : Type u
    inst✝² : ConditionallyCompleteLinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    a b : α
    s : Set α
    hs : IsClosed (Inter.inter s (Set.Icc a b))
    hab : LE.le a b
    hgt : ∀ (x : α), Membership.mem (Inter.inter s (Set.Ico a b)) x → (Inter.inter …
    S : Set α := Inter.inter s (Set.Icc a b)
    ha : Membership.mem S a
    Sbd : BddAbove S
    c : α := SupSet.sSup (Inter.inter s (Set.Icc a b))
    c_mem : Membership.mem S c
    c_le : LE.le c b
    hc : LT.lt c b
    x : α
    xs : Membership.mem s x
    cx : LT.lt c x
    xb : LE.le x b
    ⊢ False
  -/
  exact not_lt_of_le (le_csSup Sbd ⟨xs, le_trans (le_csSup Sbd ha) (le_of_lt cx), xb⟩) cx
  /-
    🎉 no goals
  -/


/-- A "continuous induction principle" for a closed interval: if a set `s` meets `[a, b]`
on a closed subset, contains `a`, and for any `a ≤ x < y ≤ b`, `x ∈ s`, the set `s ∩ (x, y]`
is not empty, then `[a, b] ⊆ s`. -/
theorem IsClosed.Icc_subset_of_forall_exists_gt {a b : α} {s : Set α} (hs : IsClosed (s ∩ Icc a b))
    (ha : a ∈ s) (hgt : ∀ x ∈ s ∩ Ico a b, ∀ y ∈ Ioi x, (s ∩ Ioc x y).Nonempty) : Icc a b ⊆ s := by
  /-
    α : Type u
    inst✝² : ConditionallyCompleteLinearOrder α
    inst✝¹ : TopologicalSpace α
    inst✝ : OrderTopology α
    a b : α
    s : Set α
    hs : IsClosed (Inter.inter s (Set.Icc a b))
    ha : Membership.mem s a
    hgt : ∀ (x : α), Membership.mem (Inter.inter s (Set.Ico a b)) x → ∀ (y : α), M …
    ⊢ HasSubset.Subset (Set.Icc a b) s
  -/
  intro y hy
  have : IsClosed (s ∩ Icc a y) := by
    suffices s ∩ Icc a y = s ∩ Icc a b ∩ Icc a y by
      rw [this]
      exact IsClosed.inter hs isClosed_Icc
    rw [inter_assoc]
    congr
    exact (inter_eq_self_of_subset_right <| Icc_subset_Icc_right hy.2).symm
  exact
    IsClosed.mem_of_ge_of_forall_exists_gt this ha hy.1 fun x hx =>
      hgt x ⟨hx.1, Ico_subset_Ico_right hy.2 hx.2⟩ y hx.2.2


/-- A "continuous induction principle" for a closed interval: if a set `s` meets `[a, b]`
on a closed subset, contains `a`, and for any `x ∈ s ∩ [a, b)` the set `s` includes some open
neighborhood of `x` within `(x, +∞)`, then `[a, b] ⊆ s`. -/
theorem IsClosed.Icc_subset_of_forall_mem_nhdsWithin {a b : α} {s : Set α}
    (hs : IsClosed (s ∩ Icc a b)) (ha : a ∈ s) (hgt : ∀ x ∈ s ∩ Ico a b, s ∈ 𝓝[>] x) :
    Icc a b ⊆ s := by
  /-
    α : Type u
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : TopologicalSpace α
    inst✝¹ : OrderTopology α
    inst✝ : DenselyOrdered α
    a b : α
    s : Set α
    hs : IsClosed (Inter.inter s (Set.Icc a b))
    ha : Membership.mem s a
    hgt : ∀ (x : α), Membership.mem (Inter.inter s (Set.Ico a b)) x → Membership.m …
    ⊢ HasSubset.Subset (Set.Icc a b) s
  -/
  apply hs.Icc_subset_of_forall_exists_gt ha
  /-
    α : Type u
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : TopologicalSpace α
    inst✝¹ : OrderTopology α
    inst✝ : DenselyOrdered α
    a b : α
    s : Set α
    hs : IsClosed (Inter.inter s (Set.Icc a b))
    ha : Membership.mem s a
    hgt : ∀ (x : α), Membership.mem (Inter.inter s (Set.Ico a b)) x → Membership.m …
    ⊢ ∀ (x : α), Membership.mem (Inter.inter s (Set.Ico a b)) x → ∀ (y : α), Membe …
  -/
  rintro x ⟨hxs, hxab⟩ y hyxb
  /-
    case intro
    α : Type u
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : TopologicalSpace α
    inst✝¹ : OrderTopology α
    inst✝ : DenselyOrdered α
    a b : α
    s : Set α
    hs : IsClosed (Inter.inter s (Set.Icc a b))
    ha : Membership.mem s a
    hgt : ∀ (x : α), Membership.mem (Inter.inter s (Set.Ico a b)) x → Membership.m …
    x : α
    hxs : Membership.mem s x
    hxab : Membership.mem (Set.Ico a b) x
    y : α
    hyxb : Membership.mem (Set.Ioi x) y
    ⊢ (Inter.inter s (Set.Ioc x y)).Nonempty
  -/
  have : s ∩ Ioc x y ∈ 𝓝[>] x := inter_mem (hgt x ⟨hxs, hxab⟩) (Ioc_mem_nhdsGT hyxb)
  /-
    case intro
    α : Type u
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : TopologicalSpace α
    inst✝¹ : OrderTopology α
    inst✝ : DenselyOrdered α
    a b : α
    s : Set α
    hs : IsClosed (Inter.inter s (Set.Icc a b))
    ha : Membership.mem s a
    hgt : ∀ (x : α), Membership.mem (Inter.inter s (Set.Ico a b)) x → Membership.m …
    x : α
    hxs : Membership.mem s x
    hxab : Membership.mem (Set.Ico a b) x
    y : α
    hyxb : Membership.mem (Set.Ioi x) y
    this : Membership.mem (nhdsWithin x (Set.Ioi x)) (Inter.inter s (Set.Ioc x y))
    ⊢ (Inter.inter s (Set.Ioc x y)).Nonempty
  -/
  exact (nhdsGT_neBot_of_exists_gt ⟨b, hxab.2⟩).nonempty_of_mem this
  /-
    🎉 no goals
  -/


theorem isPreconnected_Icc_aux (x y : α) (s t : Set α) (hxy : x ≤ y) (hs : IsClosed s)
    (ht : IsClosed t) (hab : Icc a b ⊆ s ∪ t) (hx : x ∈ Icc a b ∩ s) (hy : y ∈ Icc a b ∩ t) :
    (Icc a b ∩ (s ∩ t)).Nonempty := by
  /-
    α : Type u
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : TopologicalSpace α
    inst✝¹ : OrderTopology α
    inst✝ : DenselyOrdered α
    a b x y : α
    s t : Set α
    hxy : LE.le x y
    hs : IsClosed s
    ht : IsClosed t
    hab : HasSubset.Subset (Set.Icc a b) (Union.union s t)
    hx : Membership.mem (Inter.inter (Set.Icc a b) s) x
    hy : Membership.mem (Inter.inter (Set.Icc a b) t) y
    ⊢ (Inter.inter (Set.Icc a b) (Inter.inter s t)).Nonempty
  -/
  have xyab : Icc x y ⊆ Icc a b := Icc_subset_Icc hx.1.1 hy.1.2
  /-
    α : Type u
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : TopologicalSpace α
    inst✝¹ : OrderTopology α
    inst✝ : DenselyOrdered α
    a b x y : α
    s t : Set α
    hxy : LE.le x y
    hs : IsClosed s
    ht : IsClosed t
    hab : HasSubset.Subset (Set.Icc a b) (Union.union s t)
    hx : Membership.mem (Inter.inter (Set.Icc a b) s) x
    hy : Membership.mem (Inter.inter (Set.Icc a b) t) y
    xyab : HasSubset.Subset (Set.Icc x y) (Set.Icc a b)
    ⊢ (Inter.inter (Set.Icc a b) (Inter.inter s t)).Nonempty
  -/
  by_contra hst
  suffices Icc x y ⊆ s from
    hst ⟨y, xyab <| right_mem_Icc.2 hxy, this <| right_mem_Icc.2 hxy, hy.2⟩
  /-
    α : Type u
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : TopologicalSpace α
    inst✝¹ : OrderTopology α
    inst✝ : DenselyOrdered α
    a b x y : α
    s t : Set α
    hxy : LE.le x y
    hs : IsClosed s
    ht : IsClosed t
    hab : HasSubset.Subset (Set.Icc a b) (Union.union s t)
    hx : Membership.mem (Inter.inter (Set.Icc a b) s) x
    hy : Membership.mem (Inter.inter (Set.Icc a b) t) y
    xyab : HasSubset.Subset (Set.Icc x y) (Set.Icc a b)
    hst : Not (Inter.inter (Set.Icc a b) (Inter.inter s t)).Nonempty
    ⊢ HasSubset.Subset (Set.Icc x y) s
  -/
  apply (IsClosed.inter hs isClosed_Icc).Icc_subset_of_forall_mem_nhdsWithin hx.2
  /-
    α : Type u
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : TopologicalSpace α
    inst✝¹ : OrderTopology α
    inst✝ : DenselyOrdered α
    a b x y : α
    s t : Set α
    hxy : LE.le x y
    hs : IsClosed s
    ht : IsClosed t
    hab : HasSubset.Subset (Set.Icc a b) (Union.union s t)
    hx : Membership.mem (Inter.inter (Set.Icc a b) s) x
    hy : Membership.mem (Inter.inter (Set.Icc a b) t) y
    xyab : HasSubset.Subset (Set.Icc x y) (Set.Icc a b)
    hst : Not (Inter.inter (Set.Icc a b) (Inter.inter s t)).Nonempty
    ⊢ ∀ (x_1 : α), Membership.mem (Inter.inter s (Set.Ico x y)) x_1 → Membership.m …
  -/
  rintro z ⟨zs, hz⟩
  /-
    case intro
    α : Type u
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : TopologicalSpace α
    inst✝¹ : OrderTopology α
    inst✝ : DenselyOrdered α
    a b x y : α
    s t : Set α
    hxy : LE.le x y
    hs : IsClosed s
    ht : IsClosed t
    hab : HasSubset.Subset (Set.Icc a b) (Union.union s t)
    hx : Membership.mem (Inter.inter (Set.Icc a b) s) x
    hy : Membership.mem (Inter.inter (Set.Icc a b) t) y
    xyab : HasSubset.Subset (Set.Icc x y) (Set.Icc a b)
    hst : Not (Inter.inter (Set.Icc a b) (Inter.inter s t)).Nonempty
    z : α
    zs : Membership.mem s z
    hz : Membership.mem (Set.Ico x y) z
    ⊢ Membership.mem (nhdsWithin z (Set.Ioi z)) s
  -/
  have zt : z ∈ tᶜ := fun zt => hst ⟨z, xyab <| Ico_subset_Icc_self hz, zs, zt⟩
  have : tᶜ ∩ Ioc z y ∈ 𝓝[>] z := by
    rw [← nhdsWithin_Ioc_eq_nhdsGT hz.2]
    exact mem_nhdsWithin.2 ⟨tᶜ, ht.isOpen_compl, zt, Subset.rfl⟩
  /-
    case intro
    α : Type u
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : TopologicalSpace α
    inst✝¹ : OrderTopology α
    inst✝ : DenselyOrdered α
    a b x y : α
    s t : Set α
    hxy : LE.le x y
    hs : IsClosed s
    ht : IsClosed t
    hab : HasSubset.Subset (Set.Icc a b) (Union.union s t)
    hx : Membership.mem (Inter.inter (Set.Icc a b) s) x
    hy : Membership.mem (Inter.inter (Set.Icc a b) t) y
    xyab : HasSubset.Subset (Set.Icc x y) (Set.Icc a b)
    hst : Not (Inter.inter (Set.Icc a b) (Inter.inter s t)).Nonempty
    z : α
    zs : Membership.mem s z
    hz : Membership.mem (Set.Ico x y) z
    zt : Membership.mem (HasCompl.compl t) z
    this : Membership.mem (nhdsWithin z (Set.Ioi z)) (Inter.inter (HasCompl.compl  …
    ⊢ Membership.mem (nhdsWithin z (Set.Ioi z)) s
  -/
  apply mem_of_superset this
  /-
    case intro
    α : Type u
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : TopologicalSpace α
    inst✝¹ : OrderTopology α
    inst✝ : DenselyOrdered α
    a b x y : α
    s t : Set α
    hxy : LE.le x y
    hs : IsClosed s
    ht : IsClosed t
    hab : HasSubset.Subset (Set.Icc a b) (Union.union s t)
    hx : Membership.mem (Inter.inter (Set.Icc a b) s) x
    hy : Membership.mem (Inter.inter (Set.Icc a b) t) y
    xyab : HasSubset.Subset (Set.Icc x y) (Set.Icc a b)
    hst : Not (Inter.inter (Set.Icc a b) (Inter.inter s t)).Nonempty
    z : α
    zs : Membership.mem s z
    hz : Membership.mem (Set.Ico x y) z
    zt : Membership.mem (HasCompl.compl t) z
    this : Membership.mem (nhdsWithin z (Set.Ioi z)) (Inter.inter (HasCompl.compl  …
    ⊢ HasSubset.Subset (Inter.inter (HasCompl.compl t) (Set.Ioc z y)) s
  -/
  have : Ioc z y ⊆ s ∪ t := fun w hw => hab (xyab ⟨le_trans hz.1 (le_of_lt hw.1), hw.2⟩)
  /-
    case intro
    α : Type u
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : TopologicalSpace α
    inst✝¹ : OrderTopology α
    inst✝ : DenselyOrdered α
    a b x y : α
    s t : Set α
    hxy : LE.le x y
    hs : IsClosed s
    ht : IsClosed t
    hab : HasSubset.Subset (Set.Icc a b) (Union.union s t)
    hx : Membership.mem (Inter.inter (Set.Icc a b) s) x
    hy : Membership.mem (Inter.inter (Set.Icc a b) t) y
    xyab : HasSubset.Subset (Set.Icc x y) (Set.Icc a b)
    hst : Not (Inter.inter (Set.Icc a b) (Inter.inter s t)).Nonempty
    z : α
    zs : Membership.mem s z
    hz : Membership.mem (Set.Ico x y) z
    zt : Membership.mem (HasCompl.compl t) z
    this✝ : Membership.mem (nhdsWithin z (Set.Ioi z)) (Inter.inter (HasCompl.compl …
    this : HasSubset.Subset (Set.Ioc z y) (Union.union s t)
    ⊢ HasSubset.Subset (Inter.inter (HasCompl.compl t) (Set.Ioc z y)) s
  -/
  exact fun w ⟨wt, wzy⟩ => (this wzy).elim id fun h => (wt h).elim
  /-
    🎉 no goals
  -/


/-- A closed interval in a densely ordered conditionally complete linear order is preconnected. -/
theorem isPreconnected_Icc : IsPreconnected (Icc a b) :=
  isPreconnected_closed_iff.2
    (by
      /-
        α : Type u
        inst✝³ : ConditionallyCompleteLinearOrder α
        inst✝² : TopologicalSpace α
        inst✝¹ : OrderTopology α
        inst✝ : DenselyOrdered α
        a b : α
        ⊢ ∀ (t t' : Set α), IsClosed t → IsClosed t' → HasSubset.Subset (Set.Icc a b)  …
      -/
      rintro s t hs ht hab ⟨x, hx⟩ ⟨y, hy⟩
      -- This used to use `wlog`, but it was causing timeouts.
      /-
        case intro.intro
        α : Type u
        inst✝³ : ConditionallyCompleteLinearOrder α
        inst✝² : TopologicalSpace α
        inst✝¹ : OrderTopology α
        inst✝ : DenselyOrdered α
        a b : α
        s t : Set α
        hs : IsClosed s
        ht : IsClosed t
        hab : HasSubset.Subset (Set.Icc a b) (Union.union s t)
        x : α
        hx : Membership.mem (Inter.inter (Set.Icc a b) s) x
        y : α
        hy : Membership.mem (Inter.inter (Set.Icc a b) t) y
        ⊢ (Inter.inter (Set.Icc a b) (Inter.inter s t)).Nonempty
      -/
      rcases le_total x y with h | h
        /-
          case intro.intro.inl
          α : Type u
          inst✝³ : ConditionallyCompleteLinearOrder α
          inst✝² : TopologicalSpace α
          inst✝¹ : OrderTopology α
          inst✝ : DenselyOrdered α
          a b : α
          s t : Set α
          hs : IsClosed s
          ht : IsClosed t
          hab : HasSubset.Subset (Set.Icc a b) (Union.union s t)
          x : α
          hx : Membership.mem (Inter.inter (Set.Icc a b) s) x
          y : α
          hy : Membership.mem (Inter.inter (Set.Icc a b) t) y
          h : LE.le x y
          ⊢ (Inter.inter (Set.Icc a b) (Inter.inter s t)).Nonempty
        -/
      · exact isPreconnected_Icc_aux x y s t h hs ht hab hx hy
        /-
          🎉 no goals
        -/
        /-
          case intro.intro.inr
          α : Type u
          inst✝³ : ConditionallyCompleteLinearOrder α
          inst✝² : TopologicalSpace α
          inst✝¹ : OrderTopology α
          inst✝ : DenselyOrdered α
          a b : α
          s t : Set α
          hs : IsClosed s
          ht : IsClosed t
          hab : HasSubset.Subset (Set.Icc a b) (Union.union s t)
          x : α
          hx : Membership.mem (Inter.inter (Set.Icc a b) s) x
          y : α
          hy : Membership.mem (Inter.inter (Set.Icc a b) t) y
          h : LE.le y x
          ⊢ (Inter.inter (Set.Icc a b) (Inter.inter s t)).Nonempty
        -/
      · rw [inter_comm s t]
        /-
          case intro.intro.inr
          α : Type u
          inst✝³ : ConditionallyCompleteLinearOrder α
          inst✝² : TopologicalSpace α
          inst✝¹ : OrderTopology α
          inst✝ : DenselyOrdered α
          a b : α
          s t : Set α
          hs : IsClosed s
          ht : IsClosed t
          hab : HasSubset.Subset (Set.Icc a b) (Union.union s t)
          x : α
          hx : Membership.mem (Inter.inter (Set.Icc a b) s) x
          y : α
          hy : Membership.mem (Inter.inter (Set.Icc a b) t) y
          h : LE.le y x
          ⊢ (Inter.inter (Set.Icc a b) (Inter.inter t s)).Nonempty
        -/
        rw [union_comm s t] at hab
        /-
          case intro.intro.inr
          α : Type u
          inst✝³ : ConditionallyCompleteLinearOrder α
          inst✝² : TopologicalSpace α
          inst✝¹ : OrderTopology α
          inst✝ : DenselyOrdered α
          a b : α
          s t : Set α
          hs : IsClosed s
          ht : IsClosed t
          hab : HasSubset.Subset (Set.Icc a b) (Union.union t s)
          x : α
          hx : Membership.mem (Inter.inter (Set.Icc a b) s) x
          y : α
          hy : Membership.mem (Inter.inter (Set.Icc a b) t) y
          h : LE.le y x
          ⊢ (Inter.inter (Set.Icc a b) (Inter.inter t s)).Nonempty
        -/
        exact isPreconnected_Icc_aux y x t s h ht hs hab hy hx)
        /-
          🎉 no goals
        -/


theorem isPreconnected_uIcc : IsPreconnected ([[a, b]]) :=
  isPreconnected_Icc


theorem Set.OrdConnected.isPreconnected {s : Set α} (h : s.OrdConnected) : IsPreconnected s :=
  isPreconnected_of_forall_pair fun x hx y hy =>
    ⟨[[x, y]], h.uIcc_subset hx hy, left_mem_uIcc, right_mem_uIcc, isPreconnected_uIcc⟩


theorem isPreconnected_iff_ordConnected {s : Set α} : IsPreconnected s ↔ OrdConnected s :=
  ⟨IsPreconnected.ordConnected, Set.OrdConnected.isPreconnected⟩


theorem isPreconnected_Ici : IsPreconnected (Ici a) :=
  ordConnected_Ici.isPreconnected


theorem isPreconnected_Iic : IsPreconnected (Iic a) :=
  ordConnected_Iic.isPreconnected


theorem isPreconnected_Iio : IsPreconnected (Iio a) :=
  ordConnected_Iio.isPreconnected


theorem isPreconnected_Ioi : IsPreconnected (Ioi a) :=
  ordConnected_Ioi.isPreconnected


theorem isPreconnected_Ioo : IsPreconnected (Ioo a b) :=
  ordConnected_Ioo.isPreconnected


theorem isPreconnected_Ioc : IsPreconnected (Ioc a b) :=
  ordConnected_Ioc.isPreconnected


theorem isPreconnected_Ico : IsPreconnected (Ico a b) :=
  ordConnected_Ico.isPreconnected


theorem isConnected_Ici : IsConnected (Ici a) :=
  ⟨nonempty_Ici, isPreconnected_Ici⟩


theorem isConnected_Iic : IsConnected (Iic a) :=
  ⟨nonempty_Iic, isPreconnected_Iic⟩


theorem isConnected_Ioi [NoMaxOrder α] : IsConnected (Ioi a) :=
  ⟨nonempty_Ioi, isPreconnected_Ioi⟩


theorem isConnected_Iio [NoMinOrder α] : IsConnected (Iio a) :=
  ⟨nonempty_Iio, isPreconnected_Iio⟩


theorem isConnected_Icc (h : a ≤ b) : IsConnected (Icc a b) :=
  ⟨nonempty_Icc.2 h, isPreconnected_Icc⟩


theorem isConnected_Ioo (h : a < b) : IsConnected (Ioo a b) :=
  ⟨nonempty_Ioo.2 h, isPreconnected_Ioo⟩


theorem isConnected_Ioc (h : a < b) : IsConnected (Ioc a b) :=
  ⟨nonempty_Ioc.2 h, isPreconnected_Ioc⟩


theorem isConnected_Ico (h : a < b) : IsConnected (Ico a b) :=
  ⟨nonempty_Ico.2 h, isPreconnected_Ico⟩


instance (priority := 100) ordered_connected_space : PreconnectedSpace α :=
  ⟨ordConnected_univ.isPreconnected⟩


/-- In a dense conditionally complete linear order, the set of preconnected sets is exactly
the set of the intervals `Icc`, `Ico`, `Ioc`, `Ioo`, `Ici`, `Ioi`, `Iic`, `Iio`, `(-∞, +∞)`,
or `∅`. Though one can represent `∅` as `(sInf s, sInf s)`, we include it into the list of
possible cases to improve readability. -/
theorem setOf_isPreconnected_eq_of_ordered :
    { s : Set α | IsPreconnected s } =
      -- bounded intervals
      range (uncurry Icc) ∪ range (uncurry Ico) ∪ range (uncurry Ioc) ∪ range (uncurry Ioo) ∪
      -- unbounded intervals and `univ`
      (range Ici ∪ range Ioi ∪ range Iic ∪ range Iio ∪ {univ, ∅}) := by
  /-
    α : Type u
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : TopologicalSpace α
    inst✝¹ : OrderTopology α
    inst✝ : DenselyOrdered α
    ⊢ Eq (setOf fun s => IsPreconnected s) (Union.union (Union.union (Union.union  …
  -/
  refine Subset.antisymm setOf_isPreconnected_subset_of_ordered ?_
  simp only [subset_def, forall_mem_range, uncurry, or_imp, forall_and, mem_union,
    mem_setOf_eq, insert_eq, mem_singleton_iff, forall_eq, forall_true_iff, and_true,
    isPreconnected_Icc, isPreconnected_Ico, isPreconnected_Ioc, isPreconnected_Ioo,
    isPreconnected_Ioi, isPreconnected_Iio, isPreconnected_Ici, isPreconnected_Iic,
    isPreconnected_univ, isPreconnected_empty]


/-- This lemmas characterizes when a subset `s` of a densely ordered conditionally complete linear
order is totally disconnected with respect to the order topology: between any two distinct points
of `s` must lie a point not in `s`. -/
lemma isTotallyDisconnected_iff_lt {s : Set α} :
    IsTotallyDisconnected s ↔ ∀ x ∈ s, ∀ y ∈ s, x < y → ∃ z ∉ s, z ∈ Ioo x y := by
  simp only [IsTotallyDisconnected, isPreconnected_iff_ordConnected, ← not_nontrivial_iff,
    nontrivial_iff_exists_lt, not_exists, not_and]
  /-
    α : Type u
    inst✝³ : ConditionallyCompleteLinearOrder α
    inst✝² : TopologicalSpace α
    inst✝¹ : OrderTopology α
    inst✝ : DenselyOrdered α
    s : Set α
    ⊢ Iff (∀ (t : Set α), HasSubset.Subset t s → t.OrdConnected → ∀ (x : α), Membe …
  -/
  refine ⟨fun h x hx y hy hxy ↦ ?_, fun h t hts ht x hx y hy hxy ↦ ?_⟩
    /-
      case refine_1
      α : Type u
      inst✝³ : ConditionallyCompleteLinearOrder α
      inst✝² : TopologicalSpace α
      inst✝¹ : OrderTopology α
      inst✝ : DenselyOrdered α
      s : Set α
      h : ∀ (t : Set α), HasSubset.Subset t s → t.OrdConnected → ∀ (x : α), Membersh …
      x : α
      hx : Membership.mem s x
      y : α
      hy : Membership.mem s y
      hxy : LT.lt x y
      ⊢ Exists fun z => And (Not (Membership.mem s z)) (Membership.mem (Set.Ioo x y) …
    -/
  · simp_rw [← not_ordConnected_inter_Icc_iff hx hy]
    /-
      case refine_1
      α : Type u
      inst✝³ : ConditionallyCompleteLinearOrder α
      inst✝² : TopologicalSpace α
      inst✝¹ : OrderTopology α
      inst✝ : DenselyOrdered α
      s : Set α
      h : ∀ (t : Set α), HasSubset.Subset t s → t.OrdConnected → ∀ (x : α), Membersh …
      x : α
      hx : Membership.mem s x
      y : α
      hy : Membership.mem s y
      hxy : LT.lt x y
      ⊢ Not (Inter.inter s (Set.Icc x y)).OrdConnected
    -/
    exact fun hs ↦ h _ inter_subset_left hs _ ⟨hx, le_rfl, hxy.le⟩ _ ⟨hy, hxy.le, le_rfl⟩ hxy
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u
      inst✝³ : ConditionallyCompleteLinearOrder α
      inst✝² : TopologicalSpace α
      inst✝¹ : OrderTopology α
      inst✝ : DenselyOrdered α
      s : Set α
      h : ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem s y → LT.lt x y  …
      t : Set α
      hts : HasSubset.Subset t s
      ht : t.OrdConnected
      x : α
      hx : Membership.mem t x
      y : α
      hy : Membership.mem t y
      hxy : LT.lt x y
      ⊢ False
    -/
  · obtain ⟨z, h1z, h2z⟩ := h x (hts hx) y (hts hy) hxy
    /-
      case refine_2.intro.intro
      α : Type u
      inst✝³ : ConditionallyCompleteLinearOrder α
      inst✝² : TopologicalSpace α
      inst✝¹ : OrderTopology α
      inst✝ : DenselyOrdered α
      s : Set α
      h : ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem s y → LT.lt x y  …
      t : Set α
      hts : HasSubset.Subset t s
      ht : t.OrdConnected
      x : α
      hx : Membership.mem t x
      y : α
      hy : Membership.mem t y
      hxy : LT.lt x y
      z : α
      h1z : Not (Membership.mem s z)
      h2z : Membership.mem (Set.Ioo x y) z
      ⊢ False
    -/
    exact h1z <| hts <| ht.1 hx hy ⟨h2z.1.le, h2z.2.le⟩
    /-
      🎉 no goals
    -/


/-- **Intermediate Value Theorem** for continuous functions on closed intervals, case
`f a ≤ t ≤ f b`. -/
theorem intermediate_value_Icc {a b : α} (hab : a ≤ b) {f : α → δ} (hf : ContinuousOn f (Icc a b)) :
    Icc (f a) (f b) ⊆ f '' Icc a b :=
  isPreconnected_Icc.intermediate_value (left_mem_Icc.2 hab) (right_mem_Icc.2 hab) hf


/-- **Intermediate Value Theorem** for continuous functions on closed intervals, case
`f a ≥ t ≥ f b`. -/
theorem intermediate_value_Icc' {a b : α} (hab : a ≤ b) {f : α → δ}
    (hf : ContinuousOn f (Icc a b)) : Icc (f b) (f a) ⊆ f '' Icc a b :=
  isPreconnected_Icc.intermediate_value (right_mem_Icc.2 hab) (left_mem_Icc.2 hab) hf


/-- **Intermediate Value Theorem** for continuous functions on closed intervals, unordered case. -/
theorem intermediate_value_uIcc {a b : α} {f : α → δ} (hf : ContinuousOn f [[a, b]]) :
    [[f a, f b]] ⊆ f '' uIcc a b := by
  /-
    α : Type u
    inst✝⁶ : ConditionallyCompleteLinearOrder α
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : OrderTopology α
    inst✝³ : DenselyOrdered α
    δ : Type u_1
    inst✝² : LinearOrder δ
    inst✝¹ : TopologicalSpace δ
    inst✝ : OrderClosedTopology δ
    a b : α
    f : α → δ
    hf : ContinuousOn f (Set.uIcc a b)
    ⊢ HasSubset.Subset (Set.uIcc (f a) (f b)) (Set.image f (Set.uIcc a b))
  -/
                                 /-
                                   🎉 no goals
                                 -/
  cases le_total (f a) (f b) <;> simp [*, isPreconnected_uIcc.intermediate_value]
                                 /-
                                   🎉 no goals
                                 -/


/-- If `f : α → α` is continuous on `[[a, b]]`, `a ≤ f a`, and `f b ≤ b`,
then `f` has a fixed point on `[[a, b]]`. -/
theorem exists_mem_uIcc_isFixedPt {a b : α} {f : α → α} (hf : ContinuousOn f (uIcc a b))
    (ha : a ≤ f a) (hb : f b ≤ b) : ∃ c ∈ [[a, b]], IsFixedPt f c :=
  isPreconnected_uIcc.intermediate_value₂ right_mem_uIcc left_mem_uIcc hf continuousOn_id hb ha


/-- If `f : α → α` is continuous on `[a, b]`, `a ≤ b`, `a ≤ f a`, and `f b ≤ b`,
then `f` has a fixed point on `[a, b]`.

In particular, if `[a, b]` is forward-invariant under `f`,
then `f` has a fixed point on `[a, b]`, see `exists_mem_Icc_isFixedPt_of_mapsTo`. -/
theorem exists_mem_Icc_isFixedPt {a b : α} {f : α → α} (hf : ContinuousOn f (Icc a b))
    (hle : a ≤ b) (ha : a ≤ f a) (hb : f b ≤ b) : ∃ c ∈ Icc a b, IsFixedPt f c :=
  isPreconnected_Icc.intermediate_value₂
    (right_mem_Icc.2 hle) (left_mem_Icc.2 hle) hf continuousOn_id hb ha


/-- If a closed interval is forward-invariant under a continuous map `f : α → α`,
then this map has a fixed point on this interval. -/
theorem exists_mem_Icc_isFixedPt_of_mapsTo {a b : α} {f : α → α} (hf : ContinuousOn f (Icc a b))
    (hle : a ≤ b) (hmaps : MapsTo f (Icc a b) (Icc a b)) : ∃ c ∈ Icc a b, IsFixedPt f c :=
  exists_mem_Icc_isFixedPt hf hle (hmaps <| left_mem_Icc.2 hle).1 (hmaps <| right_mem_Icc.2 hle).2


theorem intermediate_value_Ico {a b : α} (hab : a ≤ b) {f : α → δ} (hf : ContinuousOn f (Icc a b)) :
    Ico (f a) (f b) ⊆ f '' Ico a b :=
  Or.elim (eq_or_lt_of_le hab) (fun he _ h => absurd h.2 (not_lt_of_le (he ▸ h.1))) fun hlt =>
    @IsPreconnected.intermediate_value_Ico _ _ _ _ _ _ _ isPreconnected_Ico _ _ ⟨refl a, hlt⟩
      (right_nhdsWithin_Ico_neBot hlt) inf_le_right _ (hf.mono Ico_subset_Icc_self) _
      ((hf.continuousWithinAt ⟨hab, refl b⟩).mono Ico_subset_Icc_self)


theorem intermediate_value_Ico' {a b : α} (hab : a ≤ b) {f : α → δ}
    (hf : ContinuousOn f (Icc a b)) : Ioc (f b) (f a) ⊆ f '' Ico a b :=
  Or.elim (eq_or_lt_of_le hab) (fun he _ h => absurd h.1 (not_lt_of_le (he ▸ h.2))) fun hlt =>
    @IsPreconnected.intermediate_value_Ioc _ _ _ _ _ _ _ isPreconnected_Ico _ _ ⟨refl a, hlt⟩
      (right_nhdsWithin_Ico_neBot hlt) inf_le_right _ (hf.mono Ico_subset_Icc_self) _
      ((hf.continuousWithinAt ⟨hab, refl b⟩).mono Ico_subset_Icc_self)


theorem intermediate_value_Ioc {a b : α} (hab : a ≤ b) {f : α → δ} (hf : ContinuousOn f (Icc a b)) :
    Ioc (f a) (f b) ⊆ f '' Ioc a b :=
  Or.elim (eq_or_lt_of_le hab) (fun he _ h => absurd h.2 (not_le_of_lt (he ▸ h.1))) fun hlt =>
    @IsPreconnected.intermediate_value_Ioc _ _ _ _ _ _ _ isPreconnected_Ioc _ _ ⟨hlt, refl b⟩
      (left_nhdsWithin_Ioc_neBot hlt) inf_le_right _ (hf.mono Ioc_subset_Icc_self) _
      ((hf.continuousWithinAt ⟨refl a, hab⟩).mono Ioc_subset_Icc_self)


theorem intermediate_value_Ioc' {a b : α} (hab : a ≤ b) {f : α → δ}
    (hf : ContinuousOn f (Icc a b)) : Ico (f b) (f a) ⊆ f '' Ioc a b :=
  Or.elim (eq_or_lt_of_le hab) (fun he _ h => absurd h.1 (not_le_of_lt (he ▸ h.2))) fun hlt =>
    @IsPreconnected.intermediate_value_Ico _ _ _ _ _ _ _ isPreconnected_Ioc _ _ ⟨hlt, refl b⟩
      (left_nhdsWithin_Ioc_neBot hlt) inf_le_right _ (hf.mono Ioc_subset_Icc_self) _
      ((hf.continuousWithinAt ⟨refl a, hab⟩).mono Ioc_subset_Icc_self)


theorem intermediate_value_Ioo {a b : α} (hab : a ≤ b) {f : α → δ} (hf : ContinuousOn f (Icc a b)) :
    Ioo (f a) (f b) ⊆ f '' Ioo a b :=
  Or.elim (eq_or_lt_of_le hab) (fun he _ h => absurd h.2 (not_lt_of_lt (he ▸ h.1))) fun hlt =>
    @IsPreconnected.intermediate_value_Ioo _ _ _ _ _ _ _ isPreconnected_Ioo _ _
      (left_nhdsWithin_Ioo_neBot hlt) (right_nhdsWithin_Ioo_neBot hlt) inf_le_right inf_le_right _
      (hf.mono Ioo_subset_Icc_self) _ _
      ((hf.continuousWithinAt ⟨refl a, hab⟩).mono Ioo_subset_Icc_self)
      ((hf.continuousWithinAt ⟨hab, refl b⟩).mono Ioo_subset_Icc_self)


theorem intermediate_value_Ioo' {a b : α} (hab : a ≤ b) {f : α → δ}
    (hf : ContinuousOn f (Icc a b)) : Ioo (f b) (f a) ⊆ f '' Ioo a b :=
  Or.elim (eq_or_lt_of_le hab) (fun he _ h => absurd h.1 (not_lt_of_lt (he ▸ h.2))) fun hlt =>
    @IsPreconnected.intermediate_value_Ioo _ _ _ _ _ _ _ isPreconnected_Ioo _ _
      (right_nhdsWithin_Ioo_neBot hlt) (left_nhdsWithin_Ioo_neBot hlt) inf_le_right inf_le_right _
      (hf.mono Ioo_subset_Icc_self) _ _
      ((hf.continuousWithinAt ⟨hab, refl b⟩).mono Ioo_subset_Icc_self)
      ((hf.continuousWithinAt ⟨refl a, hab⟩).mono Ioo_subset_Icc_self)


/-- **Intermediate value theorem**: if `f` is continuous on an order-connected set `s` and `a`,
`b` are two points of this set, then `f` sends `s` to a superset of `Icc (f x) (f y)`. -/
theorem ContinuousOn.surjOn_Icc {s : Set α} [hs : OrdConnected s] {f : α → δ}
    (hf : ContinuousOn f s) {a b : α} (ha : a ∈ s) (hb : b ∈ s) : SurjOn f s (Icc (f a) (f b)) :=
  hs.isPreconnected.intermediate_value ha hb hf


/-- **Intermediate value theorem**: if `f` is continuous on an order-connected set `s` and `a`,
`b` are two points of this set, then `f` sends `s` to a superset of `[f x, f y]`. -/
theorem ContinuousOn.surjOn_uIcc {s : Set α} [hs : OrdConnected s] {f : α → δ}
    (hf : ContinuousOn f s) {a b : α} (ha : a ∈ s) (hb : b ∈ s) :
    SurjOn f s (uIcc (f a) (f b)) := by
  /-
    α : Type u
    inst✝⁶ : ConditionallyCompleteLinearOrder α
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : OrderTopology α
    inst✝³ : DenselyOrdered α
    δ : Type u_1
    inst✝² : LinearOrder δ
    inst✝¹ : TopologicalSpace δ
    inst✝ : OrderClosedTopology δ
    s : Set α
    hs : s.OrdConnected
    f : α → δ
    hf : ContinuousOn f s
    a b : α
    ha : Membership.mem s a
    hb : Membership.mem s b
    ⊢ Set.SurjOn f s (Set.uIcc (f a) (f b))
  -/
                                                 /-
                                                   🎉 no goals
                                                 -/
  rcases le_total (f a) (f b) with hab | hab <;> simp [hf.surjOn_Icc, *]
                                                 /-
                                                   🎉 no goals
                                                 -/


/-- A continuous function which tendsto `Filter.atTop` along `Filter.atTop` and to `atBot` along
`at_bot` is surjective. -/
theorem Continuous.surjective {f : α → δ} (hf : Continuous f) (h_top : Tendsto f atTop atTop)
    (h_bot : Tendsto f atBot atBot) : Function.Surjective f := fun p =>
  mem_range_of_exists_le_of_exists_ge hf (h_bot.eventually (eventually_le_atBot p)).exists
    (h_top.eventually (eventually_ge_atTop p)).exists


/-- A continuous function which tendsto `Filter.atBot` along `Filter.atTop` and to `Filter.atTop`
along `atBot` is surjective. -/
theorem Continuous.surjective' {f : α → δ} (hf : Continuous f) (h_top : Tendsto f atBot atTop)
    (h_bot : Tendsto f atTop atBot) : Function.Surjective f :=
  Continuous.surjective (α := αᵒᵈ) hf h_top h_bot


/-- If a function `f : α → β` is continuous on a nonempty interval `s`, its restriction to `s`
tends to `at_bot : Filter β` along `at_bot : Filter ↥s` and tends to `Filter.atTop : Filter β` along
`Filter.atTop : Filter ↥s`, then the restriction of `f` to `s` is surjective. We formulate the
conclusion as `Function.surjOn f s Set.univ`. -/
theorem ContinuousOn.surjOn_of_tendsto {f : α → δ} {s : Set α} [OrdConnected s] (hs : s.Nonempty)
    (hf : ContinuousOn f s) (hbot : Tendsto (fun x : s => f x) atBot atBot)
    (htop : Tendsto (fun x : s => f x) atTop atTop) : SurjOn f s univ :=
  haveI := Classical.inhabited_of_nonempty hs.to_subtype
  surjOn_iff_surjective.2 <| hf.restrict.surjective htop hbot


/-- If a function `f : α → β` is continuous on a nonempty interval `s`, its restriction to `s`
tends to `Filter.atTop : Filter β` along `Filter.atBot : Filter ↥s` and tends to
`Filter.atBot : Filter β` along `Filter.atTop : Filter ↥s`, then the restriction of `f` to `s` is
surjective. We formulate the conclusion as `Function.surjOn f s Set.univ`. -/
theorem ContinuousOn.surjOn_of_tendsto' {f : α → δ} {s : Set α} [OrdConnected s] (hs : s.Nonempty)
    (hf : ContinuousOn f s) (hbot : Tendsto (fun x : s => f x) atBot atTop)
    (htop : Tendsto (fun x : s => f x) atTop atBot) : SurjOn f s univ :=
  ContinuousOn.surjOn_of_tendsto (δ := δᵒᵈ) hs hf hbot htop


theorem Continuous.strictMono_of_inj_boundedOrder [BoundedOrder α] {f : α → δ}
    (hf_c : Continuous f) (hf : f ⊥ ≤ f ⊤) (hf_i : Injective f) : StrictMono f := by
  /-
    α : Type u
    inst✝⁷ : ConditionallyCompleteLinearOrder α
    inst✝⁶ : TopologicalSpace α
    inst✝⁵ : OrderTopology α
    inst✝⁴ : DenselyOrdered α
    δ : Type u_1
    inst✝³ : LinearOrder δ
    inst✝² : TopologicalSpace δ
    inst✝¹ : OrderClosedTopology δ
    inst✝ : BoundedOrder α
    f : α → δ
    hf_c : Continuous f
    hf : LE.le (f Bot.bot) (f Top.top)
    hf_i : Function.Injective f
    ⊢ StrictMono f
  -/
  intro a b hab
  /-
    α : Type u
    inst✝⁷ : ConditionallyCompleteLinearOrder α
    inst✝⁶ : TopologicalSpace α
    inst✝⁵ : OrderTopology α
    inst✝⁴ : DenselyOrdered α
    δ : Type u_1
    inst✝³ : LinearOrder δ
    inst✝² : TopologicalSpace δ
    inst✝¹ : OrderClosedTopology δ
    inst✝ : BoundedOrder α
    f : α → δ
    hf_c : Continuous f
    hf : LE.le (f Bot.bot) (f Top.top)
    hf_i : Function.Injective f
    a b : α
    hab : LT.lt a b
    ⊢ LT.lt (f a) (f b)
  -/
  by_contra! h
  /-
    α : Type u
    inst✝⁷ : ConditionallyCompleteLinearOrder α
    inst✝⁶ : TopologicalSpace α
    inst✝⁵ : OrderTopology α
    inst✝⁴ : DenselyOrdered α
    δ : Type u_1
    inst✝³ : LinearOrder δ
    inst✝² : TopologicalSpace δ
    inst✝¹ : OrderClosedTopology δ
    inst✝ : BoundedOrder α
    f : α → δ
    hf_c : Continuous f
    hf : LE.le (f Bot.bot) (f Top.top)
    hf_i : Function.Injective f
    a b : α
    hab : LT.lt a b
    h : LE.le (f b) (f a)
    ⊢ False
  -/
  have H : f b < f a := lt_of_le_of_ne h <| hf_i.ne hab.ne'
  /-
    α : Type u
    inst✝⁷ : ConditionallyCompleteLinearOrder α
    inst✝⁶ : TopologicalSpace α
    inst✝⁵ : OrderTopology α
    inst✝⁴ : DenselyOrdered α
    δ : Type u_1
    inst✝³ : LinearOrder δ
    inst✝² : TopologicalSpace δ
    inst✝¹ : OrderClosedTopology δ
    inst✝ : BoundedOrder α
    f : α → δ
    hf_c : Continuous f
    hf : LE.le (f Bot.bot) (f Top.top)
    hf_i : Function.Injective f
    a b : α
    hab : LT.lt a b
    h : LE.le (f b) (f a)
    H : LT.lt (f b) (f a)
    ⊢ False
  -/
  by_cases ha : f a ≤ f ⊥
    /-
      case pos
      α : Type u
      inst✝⁷ : ConditionallyCompleteLinearOrder α
      inst✝⁶ : TopologicalSpace α
      inst✝⁵ : OrderTopology α
      inst✝⁴ : DenselyOrdered α
      δ : Type u_1
      inst✝³ : LinearOrder δ
      inst✝² : TopologicalSpace δ
      inst✝¹ : OrderClosedTopology δ
      inst✝ : BoundedOrder α
      f : α → δ
      hf_c : Continuous f
      hf : LE.le (f Bot.bot) (f Top.top)
      hf_i : Function.Injective f
      a b : α
      hab : LT.lt a b
      h : LE.le (f b) (f a)
      H : LT.lt (f b) (f a)
      ha : LE.le (f a) (f Bot.bot)
      ⊢ False
    -/
  · obtain ⟨u, hu⟩ := intermediate_value_Ioc le_top hf_c.continuousOn ⟨H.trans_le ha, hf⟩
    /-
      case pos.intro
      α : Type u
      inst✝⁷ : ConditionallyCompleteLinearOrder α
      inst✝⁶ : TopologicalSpace α
      inst✝⁵ : OrderTopology α
      inst✝⁴ : DenselyOrdered α
      δ : Type u_1
      inst✝³ : LinearOrder δ
      inst✝² : TopologicalSpace δ
      inst✝¹ : OrderClosedTopology δ
      inst✝ : BoundedOrder α
      f : α → δ
      hf_c : Continuous f
      hf : LE.le (f Bot.bot) (f Top.top)
      hf_i : Function.Injective f
      a b : α
      hab : LT.lt a b
      h : LE.le (f b) (f a)
      H : LT.lt (f b) (f a)
      ha : LE.le (f a) (f Bot.bot)
      u : α
      hu : And (Membership.mem (Set.Ioc b Top.top) u) (Eq (f u) (f Bot.bot))
      ⊢ False
    -/
    have : u = ⊥ := hf_i hu.2
    /-
      case pos.intro
      α : Type u
      inst✝⁷ : ConditionallyCompleteLinearOrder α
      inst✝⁶ : TopologicalSpace α
      inst✝⁵ : OrderTopology α
      inst✝⁴ : DenselyOrdered α
      δ : Type u_1
      inst✝³ : LinearOrder δ
      inst✝² : TopologicalSpace δ
      inst✝¹ : OrderClosedTopology δ
      inst✝ : BoundedOrder α
      f : α → δ
      hf_c : Continuous f
      hf : LE.le (f Bot.bot) (f Top.top)
      hf_i : Function.Injective f
      a b : α
      hab : LT.lt a b
      h : LE.le (f b) (f a)
      H : LT.lt (f b) (f a)
      ha : LE.le (f a) (f Bot.bot)
      u : α
      hu : And (Membership.mem (Set.Ioc b Top.top) u) (Eq (f u) (f Bot.bot))
      this : Eq u Bot.bot
      ⊢ False
    -/
    aesop
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u
      inst✝⁷ : ConditionallyCompleteLinearOrder α
      inst✝⁶ : TopologicalSpace α
      inst✝⁵ : OrderTopology α
      inst✝⁴ : DenselyOrdered α
      δ : Type u_1
      inst✝³ : LinearOrder δ
      inst✝² : TopologicalSpace δ
      inst✝¹ : OrderClosedTopology δ
      inst✝ : BoundedOrder α
      f : α → δ
      hf_c : Continuous f
      hf : LE.le (f Bot.bot) (f Top.top)
      hf_i : Function.Injective f
      a b : α
      hab : LT.lt a b
      h : LE.le (f b) (f a)
      H : LT.lt (f b) (f a)
      ha : Not (LE.le (f a) (f Bot.bot))
      ⊢ False
    -/
  · by_cases hb : f ⊥ < f b
      /-
        case pos
        α : Type u
        inst✝⁷ : ConditionallyCompleteLinearOrder α
        inst✝⁶ : TopologicalSpace α
        inst✝⁵ : OrderTopology α
        inst✝⁴ : DenselyOrdered α
        δ : Type u_1
        inst✝³ : LinearOrder δ
        inst✝² : TopologicalSpace δ
        inst✝¹ : OrderClosedTopology δ
        inst✝ : BoundedOrder α
        f : α → δ
        hf_c : Continuous f
        hf : LE.le (f Bot.bot) (f Top.top)
        hf_i : Function.Injective f
        a b : α
        hab : LT.lt a b
        h : LE.le (f b) (f a)
        H : LT.lt (f b) (f a)
        ha : Not (LE.le (f a) (f Bot.bot))
        hb : LT.lt (f Bot.bot) (f b)
        ⊢ False
      -/
    · obtain ⟨u, hu⟩ := intermediate_value_Ioo bot_le hf_c.continuousOn ⟨hb, H⟩
      /-
        case pos.intro
        α : Type u
        inst✝⁷ : ConditionallyCompleteLinearOrder α
        inst✝⁶ : TopologicalSpace α
        inst✝⁵ : OrderTopology α
        inst✝⁴ : DenselyOrdered α
        δ : Type u_1
        inst✝³ : LinearOrder δ
        inst✝² : TopologicalSpace δ
        inst✝¹ : OrderClosedTopology δ
        inst✝ : BoundedOrder α
        f : α → δ
        hf_c : Continuous f
        hf : LE.le (f Bot.bot) (f Top.top)
        hf_i : Function.Injective f
        a b : α
        hab : LT.lt a b
        h : LE.le (f b) (f a)
        H : LT.lt (f b) (f a)
        ha : Not (LE.le (f a) (f Bot.bot))
        hb : LT.lt (f Bot.bot) (f b)
        u : α
        hu : And (Membership.mem (Set.Ioo Bot.bot a) u) (Eq (f u) (f b))
        ⊢ False
      -/
      rw [hf_i hu.2] at hu
      /-
        case pos.intro
        α : Type u
        inst✝⁷ : ConditionallyCompleteLinearOrder α
        inst✝⁶ : TopologicalSpace α
        inst✝⁵ : OrderTopology α
        inst✝⁴ : DenselyOrdered α
        δ : Type u_1
        inst✝³ : LinearOrder δ
        inst✝² : TopologicalSpace δ
        inst✝¹ : OrderClosedTopology δ
        inst✝ : BoundedOrder α
        f : α → δ
        hf_c : Continuous f
        hf : LE.le (f Bot.bot) (f Top.top)
        hf_i : Function.Injective f
        a b : α
        hab : LT.lt a b
        h : LE.le (f b) (f a)
        H : LT.lt (f b) (f a)
        ha : Not (LE.le (f a) (f Bot.bot))
        hb : LT.lt (f Bot.bot) (f b)
        u : α
        hu : And (Membership.mem (Set.Ioo Bot.bot a) b) (Eq (f b) (f b))
        ⊢ False
      -/
      exact (hab.trans hu.1.2).false
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u
        inst✝⁷ : ConditionallyCompleteLinearOrder α
        inst✝⁶ : TopologicalSpace α
        inst✝⁵ : OrderTopology α
        inst✝⁴ : DenselyOrdered α
        δ : Type u_1
        inst✝³ : LinearOrder δ
        inst✝² : TopologicalSpace δ
        inst✝¹ : OrderClosedTopology δ
        inst✝ : BoundedOrder α
        f : α → δ
        hf_c : Continuous f
        hf : LE.le (f Bot.bot) (f Top.top)
        hf_i : Function.Injective f
        a b : α
        hab : LT.lt a b
        h : LE.le (f b) (f a)
        H : LT.lt (f b) (f a)
        ha : Not (LE.le (f a) (f Bot.bot))
        hb : Not (LT.lt (f Bot.bot) (f b))
        ⊢ False
      -/
    · push_neg at ha hb
      /-
        case neg
        α : Type u
        inst✝⁷ : ConditionallyCompleteLinearOrder α
        inst✝⁶ : TopologicalSpace α
        inst✝⁵ : OrderTopology α
        inst✝⁴ : DenselyOrdered α
        δ : Type u_1
        inst✝³ : LinearOrder δ
        inst✝² : TopologicalSpace δ
        inst✝¹ : OrderClosedTopology δ
        inst✝ : BoundedOrder α
        f : α → δ
        hf_c : Continuous f
        hf : LE.le (f Bot.bot) (f Top.top)
        hf_i : Function.Injective f
        a b : α
        hab : LT.lt a b
        h : LE.le (f b) (f a)
        H : LT.lt (f b) (f a)
        ha : LT.lt (f Bot.bot) (f a)
        hb : LE.le (f b) (f Bot.bot)
        ⊢ False
      -/
      replace hb : f b < f ⊥ := lt_of_le_of_ne hb <| hf_i.ne (lt_of_lt_of_le' hab bot_le).ne'
      /-
        case neg
        α : Type u
        inst✝⁷ : ConditionallyCompleteLinearOrder α
        inst✝⁶ : TopologicalSpace α
        inst✝⁵ : OrderTopology α
        inst✝⁴ : DenselyOrdered α
        δ : Type u_1
        inst✝³ : LinearOrder δ
        inst✝² : TopologicalSpace δ
        inst✝¹ : OrderClosedTopology δ
        inst✝ : BoundedOrder α
        f : α → δ
        hf_c : Continuous f
        hf : LE.le (f Bot.bot) (f Top.top)
        hf_i : Function.Injective f
        a b : α
        hab : LT.lt a b
        h : LE.le (f b) (f a)
        H : LT.lt (f b) (f a)
        ha : LT.lt (f Bot.bot) (f a)
        hb : LT.lt (f b) (f Bot.bot)
        ⊢ False
      -/
      obtain ⟨u, hu⟩ := intermediate_value_Ioo' hab.le hf_c.continuousOn ⟨hb, ha⟩
      /-
        case neg.intro
        α : Type u
        inst✝⁷ : ConditionallyCompleteLinearOrder α
        inst✝⁶ : TopologicalSpace α
        inst✝⁵ : OrderTopology α
        inst✝⁴ : DenselyOrdered α
        δ : Type u_1
        inst✝³ : LinearOrder δ
        inst✝² : TopologicalSpace δ
        inst✝¹ : OrderClosedTopology δ
        inst✝ : BoundedOrder α
        f : α → δ
        hf_c : Continuous f
        hf : LE.le (f Bot.bot) (f Top.top)
        hf_i : Function.Injective f
        a b : α
        hab : LT.lt a b
        h : LE.le (f b) (f a)
        H : LT.lt (f b) (f a)
        ha : LT.lt (f Bot.bot) (f a)
        hb : LT.lt (f b) (f Bot.bot)
        u : α
        hu : And (Membership.mem (Set.Ioo a b) u) (Eq (f u) (f Bot.bot))
        ⊢ False
      -/
      have : u = ⊥ := hf_i hu.2
      /-
        case neg.intro
        α : Type u
        inst✝⁷ : ConditionallyCompleteLinearOrder α
        inst✝⁶ : TopologicalSpace α
        inst✝⁵ : OrderTopology α
        inst✝⁴ : DenselyOrdered α
        δ : Type u_1
        inst✝³ : LinearOrder δ
        inst✝² : TopologicalSpace δ
        inst✝¹ : OrderClosedTopology δ
        inst✝ : BoundedOrder α
        f : α → δ
        hf_c : Continuous f
        hf : LE.le (f Bot.bot) (f Top.top)
        hf_i : Function.Injective f
        a b : α
        hab : LT.lt a b
        h : LE.le (f b) (f a)
        H : LT.lt (f b) (f a)
        ha : LT.lt (f Bot.bot) (f a)
        hb : LT.lt (f b) (f Bot.bot)
        u : α
        hu : And (Membership.mem (Set.Ioo a b) u) (Eq (f u) (f Bot.bot))
        this : Eq u Bot.bot
        ⊢ False
      -/
      aesop
      /-
        🎉 no goals
      -/


theorem Continuous.strictAnti_of_inj_boundedOrder [BoundedOrder α] {f : α → δ}
    (hf_c : Continuous f) (hf : f ⊤ ≤ f ⊥) (hf_i : Injective f) : StrictAnti f :=
  hf_c.strictMono_of_inj_boundedOrder (δ := δᵒᵈ) hf hf_i


theorem Continuous.strictMono_of_inj_boundedOrder' [BoundedOrder α] {f : α → δ}
    (hf_c : Continuous f) (hf_i : Injective f) : StrictMono f ∨ StrictAnti f :=
  (le_total (f ⊥) (f ⊤)).imp
    (hf_c.strictMono_of_inj_boundedOrder · hf_i)
    (hf_c.strictAnti_of_inj_boundedOrder · hf_i)


/-- Suppose `α` is equipped with a conditionally complete linear dense order and `f : α → δ` is
continuous and injective. Then `f` is strictly monotone (increasing) if
it is strictly monotone (increasing) on some closed interval `[a, b]`. -/
theorem Continuous.strictMonoOn_of_inj_rigidity {f : α → δ}
    (hf_c : Continuous f) (hf_i : Injective f) {a b : α} (hab : a < b)
    (hf_mono : StrictMonoOn f (Icc a b)) : StrictMono f := by
  /-
    α : Type u
    inst✝⁶ : ConditionallyCompleteLinearOrder α
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : OrderTopology α
    inst✝³ : DenselyOrdered α
    δ : Type u_1
    inst✝² : LinearOrder δ
    inst✝¹ : TopologicalSpace δ
    inst✝ : OrderClosedTopology δ
    f : α → δ
    hf_c : Continuous f
    hf_i : Function.Injective f
    a b : α
    hab : LT.lt a b
    hf_mono : StrictMonoOn f (Set.Icc a b)
    ⊢ StrictMono f
  -/
  intro x y hxy
  /-
    α : Type u
    inst✝⁶ : ConditionallyCompleteLinearOrder α
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : OrderTopology α
    inst✝³ : DenselyOrdered α
    δ : Type u_1
    inst✝² : LinearOrder δ
    inst✝¹ : TopologicalSpace δ
    inst✝ : OrderClosedTopology δ
    f : α → δ
    hf_c : Continuous f
    hf_i : Function.Injective f
    a b : α
    hab : LT.lt a b
    hf_mono : StrictMonoOn f (Set.Icc a b)
    x y : α
    hxy : LT.lt x y
    ⊢ LT.lt (f x) (f y)
  -/
  let s := min a x
  /-
    α : Type u
    inst✝⁶ : ConditionallyCompleteLinearOrder α
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : OrderTopology α
    inst✝³ : DenselyOrdered α
    δ : Type u_1
    inst✝² : LinearOrder δ
    inst✝¹ : TopologicalSpace δ
    inst✝ : OrderClosedTopology δ
    f : α → δ
    hf_c : Continuous f
    hf_i : Function.Injective f
    a b : α
    hab : LT.lt a b
    hf_mono : StrictMonoOn f (Set.Icc a b)
    x y : α
    hxy : LT.lt x y
    s : α := Min.min a x
    ⊢ LT.lt (f x) (f y)
  -/
  let t := max b y
  /-
    α : Type u
    inst✝⁶ : ConditionallyCompleteLinearOrder α
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : OrderTopology α
    inst✝³ : DenselyOrdered α
    δ : Type u_1
    inst✝² : LinearOrder δ
    inst✝¹ : TopologicalSpace δ
    inst✝ : OrderClosedTopology δ
    f : α → δ
    hf_c : Continuous f
    hf_i : Function.Injective f
    a b : α
    hab : LT.lt a b
    hf_mono : StrictMonoOn f (Set.Icc a b)
    x y : α
    hxy : LT.lt x y
    s : α := Min.min a x
    t : α := Max.max b y
    ⊢ LT.lt (f x) (f y)
  -/
  have hsa : s ≤ a := min_le_left a x
  /-
    α : Type u
    inst✝⁶ : ConditionallyCompleteLinearOrder α
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : OrderTopology α
    inst✝³ : DenselyOrdered α
    δ : Type u_1
    inst✝² : LinearOrder δ
    inst✝¹ : TopologicalSpace δ
    inst✝ : OrderClosedTopology δ
    f : α → δ
    hf_c : Continuous f
    hf_i : Function.Injective f
    a b : α
    hab : LT.lt a b
    hf_mono : StrictMonoOn f (Set.Icc a b)
    x y : α
    hxy : LT.lt x y
    s : α := Min.min a x
    t : α := Max.max b y
    hsa : LE.le s a
    ⊢ LT.lt (f x) (f y)
  -/
  have hbt : b ≤ t := le_max_left b y
  have hf_mono_st : StrictMonoOn f (Icc s t) ∨ StrictAntiOn f (Icc s t) := by
    have : Fact (s ≤ t) := ⟨hsa.trans <| hbt.trans' hab.le⟩
    have := Continuous.strictMono_of_inj_boundedOrder' (f := Set.restrict (Icc s t) f)
      hf_c.continuousOn.restrict hf_i.injOn.injective
    exact this.imp strictMono_restrict.mp strictAntiOn_iff_strictAnti.mpr
  have (h : StrictAntiOn f (Icc s t)) : False := by
    have : Icc a b ⊆ Icc s t := Icc_subset_Icc hsa hbt
    replace : StrictAntiOn f (Icc a b) := StrictAntiOn.mono h this
    replace : IsAntichain (· ≤ ·) (Icc a b) :=
      IsAntichain.of_strictMonoOn_antitoneOn hf_mono this.antitoneOn
    exact this.not_lt (left_mem_Icc.mpr (le_of_lt hab)) (right_mem_Icc.mpr (le_of_lt hab)) hab
  /-
    α : Type u
    inst✝⁶ : ConditionallyCompleteLinearOrder α
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : OrderTopology α
    inst✝³ : DenselyOrdered α
    δ : Type u_1
    inst✝² : LinearOrder δ
    inst✝¹ : TopologicalSpace δ
    inst✝ : OrderClosedTopology δ
    f : α → δ
    hf_c : Continuous f
    hf_i : Function.Injective f
    a b : α
    hab : LT.lt a b
    hf_mono : StrictMonoOn f (Set.Icc a b)
    x y : α
    hxy : LT.lt x y
    s : α := Min.min a x
    t : α := Max.max b y
    hsa : LE.le s a
    hbt : LE.le b t
    hf_mono_st : Or (StrictMonoOn f (Set.Icc s t)) (StrictAntiOn f (Set.Icc s t))
    this : StrictAntiOn f (Set.Icc s t) → False
    ⊢ LT.lt (f x) (f y)
  -/
  replace hf_mono_st : StrictMonoOn f (Icc s t) := hf_mono_st.resolve_right this
  /-
    α : Type u
    inst✝⁶ : ConditionallyCompleteLinearOrder α
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : OrderTopology α
    inst✝³ : DenselyOrdered α
    δ : Type u_1
    inst✝² : LinearOrder δ
    inst✝¹ : TopologicalSpace δ
    inst✝ : OrderClosedTopology δ
    f : α → δ
    hf_c : Continuous f
    hf_i : Function.Injective f
    a b : α
    hab : LT.lt a b
    hf_mono : StrictMonoOn f (Set.Icc a b)
    x y : α
    hxy : LT.lt x y
    s : α := Min.min a x
    t : α := Max.max b y
    hsa : LE.le s a
    hbt : LE.le b t
    this : StrictAntiOn f (Set.Icc s t) → False
    hf_mono_st : StrictMonoOn f (Set.Icc s t)
    ⊢ LT.lt (f x) (f y)
  -/
  have hsx : s ≤ x := min_le_right a x
  /-
    α : Type u
    inst✝⁶ : ConditionallyCompleteLinearOrder α
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : OrderTopology α
    inst✝³ : DenselyOrdered α
    δ : Type u_1
    inst✝² : LinearOrder δ
    inst✝¹ : TopologicalSpace δ
    inst✝ : OrderClosedTopology δ
    f : α → δ
    hf_c : Continuous f
    hf_i : Function.Injective f
    a b : α
    hab : LT.lt a b
    hf_mono : StrictMonoOn f (Set.Icc a b)
    x y : α
    hxy : LT.lt x y
    s : α := Min.min a x
    t : α := Max.max b y
    hsa : LE.le s a
    hbt : LE.le b t
    this : StrictAntiOn f (Set.Icc s t) → False
    hf_mono_st : StrictMonoOn f (Set.Icc s t)
    hsx : LE.le s x
    ⊢ LT.lt (f x) (f y)
  -/
  have hyt : y ≤ t := le_max_right b y
  /-
    α : Type u
    inst✝⁶ : ConditionallyCompleteLinearOrder α
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : OrderTopology α
    inst✝³ : DenselyOrdered α
    δ : Type u_1
    inst✝² : LinearOrder δ
    inst✝¹ : TopologicalSpace δ
    inst✝ : OrderClosedTopology δ
    f : α → δ
    hf_c : Continuous f
    hf_i : Function.Injective f
    a b : α
    hab : LT.lt a b
    hf_mono : StrictMonoOn f (Set.Icc a b)
    x y : α
    hxy : LT.lt x y
    s : α := Min.min a x
    t : α := Max.max b y
    hsa : LE.le s a
    hbt : LE.le b t
    this : StrictAntiOn f (Set.Icc s t) → False
    hf_mono_st : StrictMonoOn f (Set.Icc s t)
    hsx : LE.le s x
    hyt : LE.le y t
    ⊢ LT.lt (f x) (f y)
  -/
  replace : Icc x y ⊆ Icc s t := Icc_subset_Icc hsx hyt
  /-
    α : Type u
    inst✝⁶ : ConditionallyCompleteLinearOrder α
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : OrderTopology α
    inst✝³ : DenselyOrdered α
    δ : Type u_1
    inst✝² : LinearOrder δ
    inst✝¹ : TopologicalSpace δ
    inst✝ : OrderClosedTopology δ
    f : α → δ
    hf_c : Continuous f
    hf_i : Function.Injective f
    a b : α
    hab : LT.lt a b
    hf_mono : StrictMonoOn f (Set.Icc a b)
    x y : α
    hxy : LT.lt x y
    s : α := Min.min a x
    t : α := Max.max b y
    hsa : LE.le s a
    hbt : LE.le b t
    hf_mono_st : StrictMonoOn f (Set.Icc s t)
    hsx : LE.le s x
    hyt : LE.le y t
    this : HasSubset.Subset (Set.Icc x y) (Set.Icc s t)
    ⊢ LT.lt (f x) (f y)
  -/
  replace : StrictMonoOn f (Icc x y) := StrictMonoOn.mono hf_mono_st this
  /-
    α : Type u
    inst✝⁶ : ConditionallyCompleteLinearOrder α
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : OrderTopology α
    inst✝³ : DenselyOrdered α
    δ : Type u_1
    inst✝² : LinearOrder δ
    inst✝¹ : TopologicalSpace δ
    inst✝ : OrderClosedTopology δ
    f : α → δ
    hf_c : Continuous f
    hf_i : Function.Injective f
    a b : α
    hab : LT.lt a b
    hf_mono : StrictMonoOn f (Set.Icc a b)
    x y : α
    hxy : LT.lt x y
    s : α := Min.min a x
    t : α := Max.max b y
    hsa : LE.le s a
    hbt : LE.le b t
    hf_mono_st : StrictMonoOn f (Set.Icc s t)
    hsx : LE.le s x
    hyt : LE.le y t
    this : StrictMonoOn f (Set.Icc x y)
    ⊢ LT.lt (f x) (f y)
  -/
  exact this (left_mem_Icc.mpr (le_of_lt hxy)) (right_mem_Icc.mpr (le_of_lt hxy)) hxy
  /-
    🎉 no goals
  -/


/-- Suppose `f : [a, b] → δ` is
continuous and injective. Then `f` is strictly monotone (increasing) if `f(a) ≤ f(b)`. -/
theorem ContinuousOn.strictMonoOn_of_injOn_Icc {a b : α} {f : α → δ}
    (hab : a ≤ b) (hfab : f a ≤ f b)
    (hf_c : ContinuousOn f (Icc a b)) (hf_i : InjOn f (Icc a b)) :
    StrictMonoOn f (Icc a b) := by
  /-
    α : Type u
    inst✝⁶ : ConditionallyCompleteLinearOrder α
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : OrderTopology α
    inst✝³ : DenselyOrdered α
    δ : Type u_1
    inst✝² : LinearOrder δ
    inst✝¹ : TopologicalSpace δ
    inst✝ : OrderClosedTopology δ
    a b : α
    f : α → δ
    hab : LE.le a b
    hfab : LE.le (f a) (f b)
    hf_c : ContinuousOn f (Set.Icc a b)
    hf_i : Set.InjOn f (Set.Icc a b)
    ⊢ StrictMonoOn f (Set.Icc a b)
  -/
  have : Fact (a ≤ b) := ⟨hab⟩
  /-
    α : Type u
    inst✝⁶ : ConditionallyCompleteLinearOrder α
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : OrderTopology α
    inst✝³ : DenselyOrdered α
    δ : Type u_1
    inst✝² : LinearOrder δ
    inst✝¹ : TopologicalSpace δ
    inst✝ : OrderClosedTopology δ
    a b : α
    f : α → δ
    hab : LE.le a b
    hfab : LE.le (f a) (f b)
    hf_c : ContinuousOn f (Set.Icc a b)
    hf_i : Set.InjOn f (Set.Icc a b)
    this : Fact (LE.le a b)
    ⊢ StrictMonoOn f (Set.Icc a b)
  -/
  refine StrictMono.of_restrict ?_
  /-
    α : Type u
    inst✝⁶ : ConditionallyCompleteLinearOrder α
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : OrderTopology α
    inst✝³ : DenselyOrdered α
    δ : Type u_1
    inst✝² : LinearOrder δ
    inst✝¹ : TopologicalSpace δ
    inst✝ : OrderClosedTopology δ
    a b : α
    f : α → δ
    hab : LE.le a b
    hfab : LE.le (f a) (f b)
    hf_c : ContinuousOn f (Set.Icc a b)
    hf_i : Set.InjOn f (Set.Icc a b)
    this : Fact (LE.le a b)
    ⊢ StrictMono ((Set.Icc a b).restrict f)
  -/
  set g : Icc a b → δ := Set.restrict (Icc a b) f
  /-
    α : Type u
    inst✝⁶ : ConditionallyCompleteLinearOrder α
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : OrderTopology α
    inst✝³ : DenselyOrdered α
    δ : Type u_1
    inst✝² : LinearOrder δ
    inst✝¹ : TopologicalSpace δ
    inst✝ : OrderClosedTopology δ
    a b : α
    f : α → δ
    hab : LE.le a b
    hfab : LE.le (f a) (f b)
    hf_c : ContinuousOn f (Set.Icc a b)
    hf_i : Set.InjOn f (Set.Icc a b)
    this : Fact (LE.le a b)
    g : ↑(Set.Icc a b) → δ := (Set.Icc a b).restrict f
    ⊢ StrictMono g
  -/
  have hgab : g ⊥ ≤ g ⊤ := by aesop
  /-
    α : Type u
    inst✝⁶ : ConditionallyCompleteLinearOrder α
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : OrderTopology α
    inst✝³ : DenselyOrdered α
    δ : Type u_1
    inst✝² : LinearOrder δ
    inst✝¹ : TopologicalSpace δ
    inst✝ : OrderClosedTopology δ
    a b : α
    f : α → δ
    hab : LE.le a b
    hfab : LE.le (f a) (f b)
    hf_c : ContinuousOn f (Set.Icc a b)
    hf_i : Set.InjOn f (Set.Icc a b)
    this : Fact (LE.le a b)
    g : ↑(Set.Icc a b) → δ := (Set.Icc a b).restrict f
    hgab : LE.le (g Bot.bot) (g Top.top)
    ⊢ StrictMono g
  -/
  exact Continuous.strictMono_of_inj_boundedOrder (f := g) hf_c.restrict hgab hf_i.injective
  /-
    🎉 no goals
  -/


/-- Suppose `f : [a, b] → δ` is
continuous and injective. Then `f` is strictly antitone (decreasing) if `f(b) ≤ f(a)`. -/
theorem ContinuousOn.strictAntiOn_of_injOn_Icc {a b : α} {f : α → δ}
    (hab : a ≤ b) (hfab : f b ≤ f a)
    (hf_c : ContinuousOn f (Icc a b)) (hf_i : InjOn f (Icc a b)) :
    StrictAntiOn f (Icc a b) := ContinuousOn.strictMonoOn_of_injOn_Icc (δ := δᵒᵈ) hab hfab hf_c hf_i


/-- Suppose `f : [a, b] → δ` is continuous and injective. Then `f` is strictly monotone
or antitone (increasing or decreasing). -/
theorem ContinuousOn.strictMonoOn_of_injOn_Icc' {a b : α} {f : α → δ} (hab : a ≤ b)
    (hf_c : ContinuousOn f (Icc a b)) (hf_i : InjOn f (Icc a b)) :
    StrictMonoOn f (Icc a b) ∨ StrictAntiOn f (Icc a b) :=
  (le_total (f a) (f b)).imp
    (ContinuousOn.strictMonoOn_of_injOn_Icc hab · hf_c hf_i)
    (ContinuousOn.strictAntiOn_of_injOn_Icc hab · hf_c hf_i)


/-- Suppose `α` is equipped with a conditionally complete linear dense order and `f : α → δ` is
continuous and injective. Then `f` is strictly monotone or antitone (increasing or decreasing). -/
theorem Continuous.strictMono_of_inj {f : α → δ}
    (hf_c : Continuous f) (hf_i : Injective f) : StrictMono f ∨ StrictAnti f := by
  have H {c d : α} (hcd : c < d) : StrictMono f ∨ StrictAnti f :=
    (hf_c.continuousOn.strictMonoOn_of_injOn_Icc' hcd.le hf_i.injOn).imp
      (hf_c.strictMonoOn_of_inj_rigidity hf_i hcd)
      (hf_c.strictMonoOn_of_inj_rigidity (δ := δᵒᵈ) hf_i hcd)
  /-
    α : Type u
    inst✝⁶ : ConditionallyCompleteLinearOrder α
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : OrderTopology α
    inst✝³ : DenselyOrdered α
    δ : Type u_1
    inst✝² : LinearOrder δ
    inst✝¹ : TopologicalSpace δ
    inst✝ : OrderClosedTopology δ
    f : α → δ
    hf_c : Continuous f
    hf_i : Function.Injective f
    H : ∀ {c d : α}, LT.lt c d → Or (StrictMono f) (StrictAnti f)
    ⊢ Or (StrictMono f) (StrictAnti f)
  -/
  by_cases hn : Nonempty α
    /-
      case pos
      α : Type u
      inst✝⁶ : ConditionallyCompleteLinearOrder α
      inst✝⁵ : TopologicalSpace α
      inst✝⁴ : OrderTopology α
      inst✝³ : DenselyOrdered α
      δ : Type u_1
      inst✝² : LinearOrder δ
      inst✝¹ : TopologicalSpace δ
      inst✝ : OrderClosedTopology δ
      f : α → δ
      hf_c : Continuous f
      hf_i : Function.Injective f
      H : ∀ {c d : α}, LT.lt c d → Or (StrictMono f) (StrictAnti f)
      hn : Nonempty α
      ⊢ Or (StrictMono f) (StrictAnti f)
    -/
  · let a : α := Classical.choice ‹_›
    /-
      case pos
      α : Type u
      inst✝⁶ : ConditionallyCompleteLinearOrder α
      inst✝⁵ : TopologicalSpace α
      inst✝⁴ : OrderTopology α
      inst✝³ : DenselyOrdered α
      δ : Type u_1
      inst✝² : LinearOrder δ
      inst✝¹ : TopologicalSpace δ
      inst✝ : OrderClosedTopology δ
      f : α → δ
      hf_c : Continuous f
      hf_i : Function.Injective f
      H : ∀ {c d : α}, LT.lt c d → Or (StrictMono f) (StrictAnti f)
      hn : Nonempty α
      a : α := Classical.choice hn
      ⊢ Or (StrictMono f) (StrictAnti f)
    -/
    by_cases h : ∃ b : α, a ≠ b
      /-
        case pos
        α : Type u
        inst✝⁶ : ConditionallyCompleteLinearOrder α
        inst✝⁵ : TopologicalSpace α
        inst✝⁴ : OrderTopology α
        inst✝³ : DenselyOrdered α
        δ : Type u_1
        inst✝² : LinearOrder δ
        inst✝¹ : TopologicalSpace δ
        inst✝ : OrderClosedTopology δ
        f : α → δ
        hf_c : Continuous f
        hf_i : Function.Injective f
        H : ∀ {c d : α}, LT.lt c d → Or (StrictMono f) (StrictAnti f)
        hn : Nonempty α
        a : α := Classical.choice hn
        h : Exists fun b => Ne a b
        ⊢ Or (StrictMono f) (StrictAnti f)
      -/
    · choose b hb using h
      /-
        case pos
        α : Type u
        inst✝⁶ : ConditionallyCompleteLinearOrder α
        inst✝⁵ : TopologicalSpace α
        inst✝⁴ : OrderTopology α
        inst✝³ : DenselyOrdered α
        δ : Type u_1
        inst✝² : LinearOrder δ
        inst✝¹ : TopologicalSpace δ
        inst✝ : OrderClosedTopology δ
        f : α → δ
        hf_c : Continuous f
        hf_i : Function.Injective f
        H : ∀ {c d : α}, LT.lt c d → Or (StrictMono f) (StrictAnti f)
        hn : Nonempty α
        a : α := Classical.choice hn
        b : α
        hb : Ne a b
        ⊢ Or (StrictMono f) (StrictAnti f)
      -/
      by_cases hab : a < b
        /-
          case pos
          α : Type u
          inst✝⁶ : ConditionallyCompleteLinearOrder α
          inst✝⁵ : TopologicalSpace α
          inst✝⁴ : OrderTopology α
          inst✝³ : DenselyOrdered α
          δ : Type u_1
          inst✝² : LinearOrder δ
          inst✝¹ : TopologicalSpace δ
          inst✝ : OrderClosedTopology δ
          f : α → δ
          hf_c : Continuous f
          hf_i : Function.Injective f
          H : ∀ {c d : α}, LT.lt c d → Or (StrictMono f) (StrictAnti f)
          hn : Nonempty α
          a : α := Classical.choice hn
          b : α
          hb : Ne a b
          hab : LT.lt a b
          ⊢ Or (StrictMono f) (StrictAnti f)
        -/
      · exact H hab
        /-
          🎉 no goals
        -/
        /-
          case neg
          α : Type u
          inst✝⁶ : ConditionallyCompleteLinearOrder α
          inst✝⁵ : TopologicalSpace α
          inst✝⁴ : OrderTopology α
          inst✝³ : DenselyOrdered α
          δ : Type u_1
          inst✝² : LinearOrder δ
          inst✝¹ : TopologicalSpace δ
          inst✝ : OrderClosedTopology δ
          f : α → δ
          hf_c : Continuous f
          hf_i : Function.Injective f
          H : ∀ {c d : α}, LT.lt c d → Or (StrictMono f) (StrictAnti f)
          hn : Nonempty α
          a : α := Classical.choice hn
          b : α
          hb : Ne a b
          hab : Not (LT.lt a b)
          ⊢ Or (StrictMono f) (StrictAnti f)
        -/
      · push_neg at hab
        /-
          case neg
          α : Type u
          inst✝⁶ : ConditionallyCompleteLinearOrder α
          inst✝⁵ : TopologicalSpace α
          inst✝⁴ : OrderTopology α
          inst✝³ : DenselyOrdered α
          δ : Type u_1
          inst✝² : LinearOrder δ
          inst✝¹ : TopologicalSpace δ
          inst✝ : OrderClosedTopology δ
          f : α → δ
          hf_c : Continuous f
          hf_i : Function.Injective f
          H : ∀ {c d : α}, LT.lt c d → Or (StrictMono f) (StrictAnti f)
          hn : Nonempty α
          a : α := Classical.choice hn
          b : α
          hb : Ne a b
          hab : LE.le b a
          ⊢ Or (StrictMono f) (StrictAnti f)
        -/
        have : b < a := by exact Ne.lt_of_le (id (Ne.symm hb)) hab
        /-
          case neg
          α : Type u
          inst✝⁶ : ConditionallyCompleteLinearOrder α
          inst✝⁵ : TopologicalSpace α
          inst✝⁴ : OrderTopology α
          inst✝³ : DenselyOrdered α
          δ : Type u_1
          inst✝² : LinearOrder δ
          inst✝¹ : TopologicalSpace δ
          inst✝ : OrderClosedTopology δ
          f : α → δ
          hf_c : Continuous f
          hf_i : Function.Injective f
          H : ∀ {c d : α}, LT.lt c d → Or (StrictMono f) (StrictAnti f)
          hn : Nonempty α
          a : α := Classical.choice hn
          b : α
          hb : Ne a b
          hab : LE.le b a
          this : LT.lt b a
          ⊢ Or (StrictMono f) (StrictAnti f)
        -/
        exact H this
        /-
          🎉 no goals
        -/
      /-
        case neg
        α : Type u
        inst✝⁶ : ConditionallyCompleteLinearOrder α
        inst✝⁵ : TopologicalSpace α
        inst✝⁴ : OrderTopology α
        inst✝³ : DenselyOrdered α
        δ : Type u_1
        inst✝² : LinearOrder δ
        inst✝¹ : TopologicalSpace δ
        inst✝ : OrderClosedTopology δ
        f : α → δ
        hf_c : Continuous f
        hf_i : Function.Injective f
        H : ∀ {c d : α}, LT.lt c d → Or (StrictMono f) (StrictAnti f)
        hn : Nonempty α
        a : α := Classical.choice hn
        h : Not (Exists fun b => Ne a b)
        ⊢ Or (StrictMono f) (StrictAnti f)
      -/
    · push_neg at h
      /-
        case neg
        α : Type u
        inst✝⁶ : ConditionallyCompleteLinearOrder α
        inst✝⁵ : TopologicalSpace α
        inst✝⁴ : OrderTopology α
        inst✝³ : DenselyOrdered α
        δ : Type u_1
        inst✝² : LinearOrder δ
        inst✝¹ : TopologicalSpace δ
        inst✝ : OrderClosedTopology δ
        f : α → δ
        hf_c : Continuous f
        hf_i : Function.Injective f
        H : ∀ {c d : α}, LT.lt c d → Or (StrictMono f) (StrictAnti f)
        hn : Nonempty α
        a : α := Classical.choice hn
        h : ∀ (b : α), Eq a b
        ⊢ Or (StrictMono f) (StrictAnti f)
      -/
      haveI : Subsingleton α := ⟨fun c d => Trans.trans (h c).symm (h d)⟩
      /-
        case neg
        α : Type u
        inst✝⁶ : ConditionallyCompleteLinearOrder α
        inst✝⁵ : TopologicalSpace α
        inst✝⁴ : OrderTopology α
        inst✝³ : DenselyOrdered α
        δ : Type u_1
        inst✝² : LinearOrder δ
        inst✝¹ : TopologicalSpace δ
        inst✝ : OrderClosedTopology δ
        f : α → δ
        hf_c : Continuous f
        hf_i : Function.Injective f
        H : ∀ {c d : α}, LT.lt c d → Or (StrictMono f) (StrictAnti f)
        hn : Nonempty α
        a : α := Classical.choice hn
        h : ∀ (b : α), Eq a b
        this : Subsingleton α
        ⊢ Or (StrictMono f) (StrictAnti f)
      -/
      exact Or.inl <| Subsingleton.strictMono f
      /-
        🎉 no goals
      -/
    /-
      case neg
      α : Type u
      inst✝⁶ : ConditionallyCompleteLinearOrder α
      inst✝⁵ : TopologicalSpace α
      inst✝⁴ : OrderTopology α
      inst✝³ : DenselyOrdered α
      δ : Type u_1
      inst✝² : LinearOrder δ
      inst✝¹ : TopologicalSpace δ
      inst✝ : OrderClosedTopology δ
      f : α → δ
      hf_c : Continuous f
      hf_i : Function.Injective f
      H : ∀ {c d : α}, LT.lt c d → Or (StrictMono f) (StrictAnti f)
      hn : Not (Nonempty α)
      ⊢ Or (StrictMono f) (StrictAnti f)
    -/
  · aesop
    /-
      🎉 no goals
    -/


/-- Every continuous injective `f : (a, b) → δ` is strictly monotone
or antitone (increasing or decreasing). -/
theorem ContinuousOn.strictMonoOn_of_injOn_Ioo {a b : α} {f : α → δ} (hab : a < b)
    (hf_c : ContinuousOn f (Ioo a b)) (hf_i : InjOn f (Ioo a b)) :
    StrictMonoOn f (Ioo a b) ∨ StrictAntiOn f (Ioo a b) := by
  /-
    α : Type u
    inst✝⁶ : ConditionallyCompleteLinearOrder α
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : OrderTopology α
    inst✝³ : DenselyOrdered α
    δ : Type u_1
    inst✝² : LinearOrder δ
    inst✝¹ : TopologicalSpace δ
    inst✝ : OrderClosedTopology δ
    a b : α
    f : α → δ
    hab : LT.lt a b
    hf_c : ContinuousOn f (Set.Ioo a b)
    hf_i : Set.InjOn f (Set.Ioo a b)
    ⊢ Or (StrictMonoOn f (Set.Ioo a b)) (StrictAntiOn f (Set.Ioo a b))
  -/
  haveI : Inhabited (Ioo a b) := Classical.inhabited_of_nonempty (nonempty_Ioo_subtype hab)
  /-
    α : Type u
    inst✝⁶ : ConditionallyCompleteLinearOrder α
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : OrderTopology α
    inst✝³ : DenselyOrdered α
    δ : Type u_1
    inst✝² : LinearOrder δ
    inst✝¹ : TopologicalSpace δ
    inst✝ : OrderClosedTopology δ
    a b : α
    f : α → δ
    hab : LT.lt a b
    hf_c : ContinuousOn f (Set.Ioo a b)
    hf_i : Set.InjOn f (Set.Ioo a b)
    this : Inhabited ↑(Set.Ioo a b)
    ⊢ Or (StrictMonoOn f (Set.Ioo a b)) (StrictAntiOn f (Set.Ioo a b))
  -/
  let g : Ioo a b → δ := Set.restrict (Ioo a b) f
  have : StrictMono g ∨ StrictAnti g :=
    Continuous.strictMono_of_inj hf_c.restrict hf_i.injective
  /-
    α : Type u
    inst✝⁶ : ConditionallyCompleteLinearOrder α
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : OrderTopology α
    inst✝³ : DenselyOrdered α
    δ : Type u_1
    inst✝² : LinearOrder δ
    inst✝¹ : TopologicalSpace δ
    inst✝ : OrderClosedTopology δ
    a b : α
    f : α → δ
    hab : LT.lt a b
    hf_c : ContinuousOn f (Set.Ioo a b)
    hf_i : Set.InjOn f (Set.Ioo a b)
    this✝ : Inhabited ↑(Set.Ioo a b)
    g : ↑(Set.Ioo a b) → δ := (Set.Ioo a b).restrict f
    this : Or (StrictMono g) (StrictAnti g)
    ⊢ Or (StrictMonoOn f (Set.Ioo a b)) (StrictAntiOn f (Set.Ioo a b))
  -/
  exact this.imp strictMono_restrict.mp strictAntiOn_iff_strictAnti.mpr
  /-
    🎉 no goals
  -/

