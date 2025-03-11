/-- The forward image of a measurable space under a function. `map f m` contains the sets
  `s : Set β` whose preimage under `f` is measurable. -/
protected def map (f : α → β) (m : MeasurableSpace α) : MeasurableSpace β where
  MeasurableSet' s := MeasurableSet[m] <| f ⁻¹' s
  measurableSet_empty := m.measurableSet_empty
  measurableSet_compl _ hs := m.measurableSet_compl _ hs
                                  /-
                                    α : Type u_1
                                    β : Type u_2
                                    γ : Type u_3
                                    δ : Type u_4
                                    δ' : Type u_5
                                    ι : Sort uι
                                    s : Set α
                                    m✝ m₁ m₂ : MeasurableSpace α
                                    m' : MeasurableSpace β
                                    f✝¹ : α → β
                                    g : β → α
                                    f✝ : α → β
                                    m : MeasurableSpace α
                                    f : Nat → Set β
                                    hf : ∀ (i : Nat), (fun s => MeasurableSet (Set.preimage f✝ s)) (f i)
                                    ⊢ (fun s => MeasurableSet (Set.preimage f✝ s)) (Set.iUnion fun i => f i)
                                  -/
  measurableSet_iUnion f hf := by simpa only [preimage_iUnion] using m.measurableSet_iUnion _ hf
                                  /-
                                    🎉 no goals
                                  -/


lemma map_def {s : Set β} : MeasurableSet[m.map f] s ↔ MeasurableSet[m] (f ⁻¹' s) := Iff.rfl


@[simp]
theorem map_id : m.map id = m :=
  MeasurableSpace.ext fun _ => Iff.rfl


@[simp]
theorem map_comp {f : α → β} {g : β → γ} : (m.map f).map g = m.map (g ∘ f) :=
  MeasurableSpace.ext fun _ => Iff.rfl


/-- The reverse image of a measurable space under a function. `comap f m` contains the sets
  `s : Set α` such that `s` is the `f`-preimage of a measurable set in `β`. -/
protected def comap (f : α → β) (m : MeasurableSpace β) : MeasurableSpace α where
  MeasurableSet' s := ∃ s', MeasurableSet[m] s' ∧ f ⁻¹' s' = s
  measurableSet_empty := ⟨∅, m.measurableSet_empty, rfl⟩
  measurableSet_compl := fun _ ⟨s', h₁, h₂⟩ => ⟨s'ᶜ, m.measurableSet_compl _ h₁, h₂ ▸ rfl⟩
  measurableSet_iUnion s hs :=
    let ⟨s', hs'⟩ := Classical.axiom_of_choice hs
                                                                   /-
                                                                     α : Type u_1
                                                                     β : Type u_2
                                                                     γ : Type u_3
                                                                     δ : Type u_4
                                                                     δ' : Type u_5
                                                                     ι : Sort uι
                                                                     s✝ : Set α
                                                                     m✝ m₁ m₂ : MeasurableSpace α
                                                                     m' : MeasurableSpace β
                                                                     f✝ : α → β
                                                                     g : β → α
                                                                     f : α → β
                                                                     m : MeasurableSpace β
                                                                     s : Nat → Set α
                                                                     hs : ∀ (i : Nat), (fun s => Exists fun s' => And (MeasurableSet s') (Eq (Set.p …
                                                                     s' : Nat → Set β
                                                                     hs' : ∀ (x : Nat), And (MeasurableSet (s' x)) (Eq (Set.preimage f (s' x)) (s x))
                                                                     ⊢ Eq (Set.preimage f (Set.iUnion fun i => s' i)) (Set.iUnion fun i => s i)
                                                                   -/
    ⟨⋃ i, s' i, m.measurableSet_iUnion _ fun i => (hs' i).left, by simp [hs']⟩
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


lemma measurableSet_comap {m : MeasurableSpace β} :
    MeasurableSet[m.comap f] s ↔ ∃ s', MeasurableSet[m] s' ∧ f ⁻¹' s' = s := .rfl


theorem comap_eq_generateFrom (m : MeasurableSpace β) (f : α → β) :
    m.comap f = generateFrom { t | ∃ s, MeasurableSet s ∧ f ⁻¹' s = t } :=
  (@generateFrom_measurableSet _ (.comap f m)).symm


@[simp]
theorem comap_id : m.comap id = m :=
  MeasurableSpace.ext fun s => ⟨fun ⟨_, hs', h⟩ => h ▸ hs', fun h => ⟨s, h, rfl⟩⟩


@[simp]
theorem comap_comp {f : β → α} {g : γ → β} : (m.comap f).comap g = m.comap (f ∘ g) :=
  MeasurableSpace.ext fun _ =>
    ⟨fun ⟨_, ⟨u, h, hu⟩, ht⟩ => ⟨u, h, ht ▸ hu ▸ rfl⟩, fun ⟨t, h, ht⟩ => ⟨f ⁻¹' t, ⟨_, h, rfl⟩, ht⟩⟩


theorem comap_le_iff_le_map {f : α → β} : m'.comap f ≤ m ↔ m' ≤ m.map f :=
  ⟨fun h _s hs => h _ ⟨_, hs, rfl⟩, fun h _s ⟨_t, ht, heq⟩ => heq ▸ h _ ht⟩


theorem gc_comap_map (f : α → β) :
    GaloisConnection (MeasurableSpace.comap f) (MeasurableSpace.map f) := fun _ _ =>
  comap_le_iff_le_map


theorem map_mono (h : m₁ ≤ m₂) : m₁.map f ≤ m₂.map f :=
  (gc_comap_map f).monotone_u h


theorem monotone_map : Monotone (MeasurableSpace.map f) := fun _ _ => map_mono


theorem comap_mono (h : m₁ ≤ m₂) : m₁.comap g ≤ m₂.comap g :=
  (gc_comap_map g).monotone_l h


theorem monotone_comap : Monotone (MeasurableSpace.comap g) := fun _ _ h => comap_mono h


@[simp]
theorem comap_bot : (⊥ : MeasurableSpace α).comap g = ⊥ :=
  (gc_comap_map g).l_bot


@[simp]
theorem comap_sup : (m₁ ⊔ m₂).comap g = m₁.comap g ⊔ m₂.comap g :=
  (gc_comap_map g).l_sup


@[simp]
theorem comap_iSup {m : ι → MeasurableSpace α} : (⨆ i, m i).comap g = ⨆ i, (m i).comap g :=
  (gc_comap_map g).l_iSup


@[simp]
theorem map_top : (⊤ : MeasurableSpace α).map f = ⊤ :=
  (gc_comap_map f).u_top


@[simp]
theorem map_inf : (m₁ ⊓ m₂).map f = m₁.map f ⊓ m₂.map f :=
  (gc_comap_map f).u_inf


@[simp]
theorem map_iInf {m : ι → MeasurableSpace α} : (⨅ i, m i).map f = ⨅ i, (m i).map f :=
  (gc_comap_map f).u_iInf


theorem comap_map_le : (m.map f).comap f ≤ m :=
  (gc_comap_map f).l_u_le _


theorem le_map_comap : m ≤ (m.comap g).map g :=
  (gc_comap_map g).le_u_l _


@[simp] theorem map_const {m} (b : β) : MeasurableSpace.map (fun _a : α ↦ b) m = ⊤ :=
                               /-
                                 α : Type u_1
                                 β : Type u_2
                                 m : MeasurableSpace α
                                 b : β
                                 s : Set β
                                 x✝ : MeasurableSet s
                                 ⊢ MeasurableSet s
                               -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
  eq_top_iff.2 <| fun s _ ↦ by rw [map_def]; by_cases h : b ∈ s <;> simp [h]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


@[simp] theorem comap_const {m} (b : β) : MeasurableSpace.comap (fun _a : α => b) m = ⊥ :=
                     /-
                       α : Type u_1
                       β : Type u_2
                       m : MeasurableSpace β
                       b : β
                       ⊢ LE.le (MeasurableSpace.comap (fun _a => b) m) Bot.bot
                     -/
                                                              /-
                                                                🎉 no goals
                                                              -/
  eq_bot_iff.2 <| by rintro _ ⟨s, -, rfl⟩; by_cases b ∈ s <;> simp [*]
                                                              /-
                                                                🎉 no goals
                                                              -/


theorem comap_generateFrom {f : α → β} {s : Set (Set β)} :
    (generateFrom s).comap f = generateFrom (preimage f '' s) :=
  le_antisymm
    (comap_le_iff_le_map.2 <|
      generateFrom_le fun _t hts => GenerateMeasurable.basic _ <| mem_image_of_mem _ <| hts)
    (generateFrom_le fun _t ⟨u, hu, Eq⟩ => Eq ▸ ⟨u, GenerateMeasurable.basic _ hu, rfl⟩)


theorem measurable_iff_le_map {m₁ : MeasurableSpace α} {m₂ : MeasurableSpace β} {f : α → β} :
    Measurable f ↔ m₂ ≤ m₁.map f :=
  Iff.rfl


alias ⟨Measurable.le_map, Measurable.of_le_map⟩ := measurable_iff_le_map


theorem measurable_iff_comap_le {m₁ : MeasurableSpace α} {m₂ : MeasurableSpace β} {f : α → β} :
    Measurable f ↔ m₂.comap f ≤ m₁ :=
  comap_le_iff_le_map.symm


alias ⟨Measurable.comap_le, Measurable.of_comap_le⟩ := measurable_iff_comap_le


theorem comap_measurable {m : MeasurableSpace β} (f : α → β) : Measurable[m.comap f] f :=
  fun s hs => ⟨s, hs, rfl⟩


theorem Measurable.mono {ma ma' : MeasurableSpace α} {mb mb' : MeasurableSpace β} {f : α → β}
    (hf : @Measurable α β ma mb f) (ha : ma ≤ ma') (hb : mb' ≤ mb) : @Measurable α β ma' mb' f :=
  fun _t ht => ha _ <| hf <| hb _ ht


lemma Measurable.iSup' {mα : ι → MeasurableSpace α} {_ : MeasurableSpace β} {f : α → β} (i₀ : ι)
    (h : Measurable[mα i₀] f) :
    Measurable[⨆ i, mα i] f :=
  h.mono (le_iSup mα i₀) le_rfl


lemma Measurable.sup_of_left {mα mα' : MeasurableSpace α} {_ : MeasurableSpace β} {f : α → β}
    (h : Measurable[mα] f) :
    Measurable[mα ⊔ mα'] f :=
  h.mono le_sup_left le_rfl


lemma Measurable.sup_of_right {mα mα' : MeasurableSpace α} {_ : MeasurableSpace β} {f : α → β}
    (h : Measurable[mα'] f) :
    Measurable[mα ⊔ mα'] f :=
  h.mono le_sup_right le_rfl


theorem measurable_id'' {m mα : MeasurableSpace α} (hm : m ≤ mα) : @Measurable α α mα m id :=
  measurable_id.mono le_rfl hm

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: add TC `DiscreteMeasurable` + instances


@[measurability]
theorem measurable_from_top [MeasurableSpace β] {f : α → β} : Measurable[⊤] f := fun _ _ => trivial


theorem measurable_generateFrom [MeasurableSpace α] {s : Set (Set β)} {f : α → β}
    (h : ∀ t ∈ s, MeasurableSet (f ⁻¹' t)) : @Measurable _ _ _ (generateFrom s) f :=
  Measurable.of_le_map <| generateFrom_le h


@[nontriviality, measurability]
theorem Subsingleton.measurable [Subsingleton α] : Measurable f := fun _ _ =>
  @Subsingleton.measurableSet α _ _ _


@[nontriviality, measurability]
theorem measurable_of_subsingleton_codomain [Subsingleton β] (f : α → β) : Measurable f :=
  fun s _ => Subsingleton.set_cases MeasurableSet.empty MeasurableSet.univ s


@[to_additive (attr := measurability, fun_prop)]
theorem measurable_one [One α] : Measurable (1 : β → α) :=
  @measurable_const _ _ _ _ 1


theorem measurable_of_empty [IsEmpty α] (f : α → β) : Measurable f :=
  Subsingleton.measurable


theorem measurable_of_empty_codomain [IsEmpty β] (f : α → β) : Measurable f :=
  measurable_of_subsingleton_codomain f


/-- A version of `measurable_const` that assumes `f x = f y` for all `x, y`. This version works
for functions between empty types. -/
theorem measurable_const' {f : β → α} (hf : ∀ x y, f x = f y) : Measurable f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    f : β → α
    hf : ∀ (x y : β), Eq (f x) (f y)
    ⊢ Measurable f
  -/
  nontriviality β
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    f : β → α
    hf : ∀ (x y : β), Eq (f x) (f y)
    a✝ : Nontrivial β
    ⊢ Measurable f
  -/
  inhabit β
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    f : β → α
    hf : ∀ (x y : β), Eq (f x) (f y)
    a✝ : Nontrivial β
    inhabited_h : Inhabited β
    ⊢ Measurable f
  -/
  convert @measurable_const α β _ _ (f default) using 2
  /-
    case h.e'_5.h
    α : Type u_1
    β : Type u_2
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSpace β
    f : β → α
    hf : ∀ (x y : β), Eq (f x) (f y)
    a✝ : Nontrivial β
    inhabited_h : Inhabited β
    x✝ : β
    ⊢ Eq (f x✝) (f Inhabited.default)
  -/
  apply hf
  /-
    🎉 no goals
  -/


@[measurability]
theorem measurable_natCast [NatCast α] (n : ℕ) : Measurable (n : β → α) :=
  @measurable_const α _ _ _ n


@[measurability]
theorem measurable_intCast [IntCast α] (n : ℤ) : Measurable (n : β → α) :=
  @measurable_const α _ _ _ n


theorem measurable_of_countable [Countable α] [MeasurableSingletonClass α] (f : α → β) :
    Measurable f := fun s _ =>
  (f ⁻¹' s).to_countable.measurableSet


theorem measurable_of_finite [Finite α] [MeasurableSingletonClass α] (f : α → β) : Measurable f :=
  measurable_of_countable f


@[measurability]
theorem Measurable.iterate {f : α → α} (hf : Measurable f) : ∀ n, Measurable f^[n]
  | 0 => measurable_id
  | n + 1 => (Measurable.iterate hf n).comp hf


@[measurability]
theorem measurableSet_preimage {t : Set β} (hf : Measurable f) (ht : MeasurableSet t) :
    MeasurableSet (f ⁻¹' t) :=
  hf ht


protected theorem MeasurableSet.preimage {t : Set β} (ht : MeasurableSet t) (hf : Measurable f) :
    MeasurableSet (f ⁻¹' t) :=
  hf ht


@[measurability, fun_prop]
protected theorem Measurable.piecewise {_ : DecidablePred (· ∈ s)} (hs : MeasurableSet s)
    (hf : Measurable f) (hg : Measurable g) : Measurable (piecewise s f g) := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    f g : α → β
    m : MeasurableSpace α
    mβ : MeasurableSpace β
    x✝ : DecidablePred fun x => Membership.mem s x
    hs : MeasurableSet s
    hf : Measurable f
    hg : Measurable g
    ⊢ Measurable (s.piecewise f g)
  -/
  intro t ht
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    f g : α → β
    m : MeasurableSpace α
    mβ : MeasurableSpace β
    x✝ : DecidablePred fun x => Membership.mem s x
    hs : MeasurableSet s
    hf : Measurable f
    hg : Measurable g
    t : Set β
    ht : MeasurableSet t
    ⊢ MeasurableSet (Set.preimage (s.piecewise f g) t)
  -/
  rw [piecewise_preimage]
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    f g : α → β
    m : MeasurableSpace α
    mβ : MeasurableSpace β
    x✝ : DecidablePred fun x => Membership.mem s x
    hs : MeasurableSet s
    hf : Measurable f
    hg : Measurable g
    t : Set β
    ht : MeasurableSet t
    ⊢ MeasurableSet (s.ite (Set.preimage f t) (Set.preimage g t))
  -/
  exact hs.ite (hf ht) (hg ht)
  /-
    🎉 no goals
  -/


/-- This is slightly different from `Measurable.piecewise`. It can be used to show
`Measurable (ite (x=0) 0 1)` by
`exact Measurable.ite (measurableSet_singleton 0) measurable_const measurable_const`,
but replacing `Measurable.ite` by `Measurable.piecewise` in that example proof does not work. -/
theorem Measurable.ite {p : α → Prop} {_ : DecidablePred p} (hp : MeasurableSet { a : α | p a })
    (hf : Measurable f) (hg : Measurable g) : Measurable fun x => ite (p x) (f x) (g x) :=
  Measurable.piecewise hp hf hg


@[measurability, fun_prop]
theorem Measurable.indicator [Zero β] (hf : Measurable f) (hs : MeasurableSet s) :
    Measurable (s.indicator f) :=
  hf.piecewise hs measurable_const


/-- The measurability of a set `A` is equivalent to the measurability of the indicator function
which takes a constant value `b ≠ 0` on a set `A` and `0` elsewhere. -/
lemma measurable_indicator_const_iff [Zero β] [MeasurableSingletonClass β] (b : β) [NeZero b] :
    Measurable (s.indicator (fun (_ : α) ↦ b)) ↔ MeasurableSet s := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    m : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝² : Zero β
    inst✝¹ : MeasurableSingletonClass β
    b : β
    inst✝ : NeZero b
    ⊢ Iff (Measurable (s.indicator fun x => b)) (MeasurableSet s)
  -/
  constructor <;> intro h
    /-
      case mp
      α : Type u_1
      β : Type u_2
      s : Set α
      m : MeasurableSpace α
      mβ : MeasurableSpace β
      inst✝² : Zero β
      inst✝¹ : MeasurableSingletonClass β
      b : β
      inst✝ : NeZero b
      h : Measurable (s.indicator fun x => b)
      ⊢ MeasurableSet s
    -/
  · convert h (MeasurableSet.singleton (0 : β)).compl
    /-
      case h.e'_3
      α : Type u_1
      β : Type u_2
      s : Set α
      m : MeasurableSpace α
      mβ : MeasurableSpace β
      inst✝² : Zero β
      inst✝¹ : MeasurableSingletonClass β
      b : β
      inst✝ : NeZero b
      h : Measurable (s.indicator fun x => b)
      ⊢ Eq s (Set.preimage (s.indicator fun x => b) (HasCompl.compl (Singleton.singl …
    -/
    ext a
    /-
      case h.e'_3.h
      α : Type u_1
      β : Type u_2
      s : Set α
      m : MeasurableSpace α
      mβ : MeasurableSpace β
      inst✝² : Zero β
      inst✝¹ : MeasurableSingletonClass β
      b : β
      inst✝ : NeZero b
      h : Measurable (s.indicator fun x => b)
      a : α
      ⊢ Iff (Membership.mem s a) (Membership.mem (Set.preimage (s.indicator fun x => …
    -/
    simp [NeZero.ne b]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      β : Type u_2
      s : Set α
      m : MeasurableSpace α
      mβ : MeasurableSpace β
      inst✝² : Zero β
      inst✝¹ : MeasurableSingletonClass β
      b : β
      inst✝ : NeZero b
      h : MeasurableSet s
      ⊢ Measurable (s.indicator fun x => b)
    -/
  · exact measurable_const.indicator h
    /-
      🎉 no goals
    -/


@[to_additive (attr := measurability)]
theorem measurableSet_mulSupport [One β] [MeasurableSingletonClass β] (hf : Measurable f) :
    MeasurableSet (mulSupport f) :=
  hf (measurableSet_singleton 1).compl


/-- If a function coincides with a measurable function outside of a countable set, it is
measurable. -/
theorem Measurable.measurable_of_countable_ne [MeasurableSingletonClass α] (hf : Measurable f)
    (h : Set.Countable { x | f x ≠ g x }) : Measurable g := by
  /-
    α : Type u_1
    β : Type u_2
    f g : α → β
    m : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝ : MeasurableSingletonClass α
    hf : Measurable f
    h : (setOf fun x => Ne (f x) (g x)).Countable
    ⊢ Measurable g
  -/
  intro t ht
  have : g ⁻¹' t = g ⁻¹' t ∩ { x | f x = g x }ᶜ ∪ g ⁻¹' t ∩ { x | f x = g x } := by
    simp [← inter_union_distrib_left]
  /-
    α : Type u_1
    β : Type u_2
    f g : α → β
    m : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝ : MeasurableSingletonClass α
    hf : Measurable f
    h : (setOf fun x => Ne (f x) (g x)).Countable
    t : Set β
    ht : MeasurableSet t
    this : Eq (Set.preimage g t) (Union.union (Inter.inter (Set.preimage g t) (Has …
    ⊢ MeasurableSet (Set.preimage g t)
  -/
  rw [this]
  /-
    α : Type u_1
    β : Type u_2
    f g : α → β
    m : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝ : MeasurableSingletonClass α
    hf : Measurable f
    h : (setOf fun x => Ne (f x) (g x)).Countable
    t : Set β
    ht : MeasurableSet t
    this : Eq (Set.preimage g t) (Union.union (Inter.inter (Set.preimage g t) (Has …
    ⊢ MeasurableSet (Union.union (Inter.inter (Set.preimage g t) (HasCompl.compl ( …
  -/
  refine (h.mono inter_subset_right).measurableSet.union ?_
  have : g ⁻¹' t ∩ { x : α | f x = g x } = f ⁻¹' t ∩ { x : α | f x = g x } := by
    ext x
    simp +contextual
  /-
    α : Type u_1
    β : Type u_2
    f g : α → β
    m : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝ : MeasurableSingletonClass α
    hf : Measurable f
    h : (setOf fun x => Ne (f x) (g x)).Countable
    t : Set β
    ht : MeasurableSet t
    this✝ : Eq (Set.preimage g t) (Union.union (Inter.inter (Set.preimage g t) (Ha …
    this : Eq (Inter.inter (Set.preimage g t) (setOf fun x => Eq (f x) (g x))) (In …
    ⊢ MeasurableSet (Inter.inter (Set.preimage g t) (setOf fun x => Eq (f x) (g x)))
  -/
  rw [this]
  /-
    α : Type u_1
    β : Type u_2
    f g : α → β
    m : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝ : MeasurableSingletonClass α
    hf : Measurable f
    h : (setOf fun x => Ne (f x) (g x)).Countable
    t : Set β
    ht : MeasurableSet t
    this✝ : Eq (Set.preimage g t) (Union.union (Inter.inter (Set.preimage g t) (Ha …
    this : Eq (Inter.inter (Set.preimage g t) (setOf fun x => Eq (f x) (g x))) (In …
    ⊢ MeasurableSet (Inter.inter (Set.preimage f t) (setOf fun x => Eq (f x) (g x)))
  -/
  exact (hf ht).inter h.measurableSet.of_compl
  /-
    🎉 no goals
  -/


theorem measurable_to_countable [MeasurableSpace α] [Countable α] [MeasurableSpace β] {f : β → α}
    (h : ∀ y, MeasurableSet (f ⁻¹' {f y})) : Measurable f := fun s _ => by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : Countable α
    inst✝ : MeasurableSpace β
    f : β → α
    h : ∀ (y : β), MeasurableSet (Set.preimage f (Singleton.singleton (f y)))
    s : Set α
    x✝ : MeasurableSet s
    ⊢ MeasurableSet (Set.preimage f s)
  -/
  rw [← biUnion_preimage_singleton]
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : Countable α
    inst✝ : MeasurableSpace β
    f : β → α
    h : ∀ (y : β), MeasurableSet (Set.preimage f (Singleton.singleton (f y)))
    s : Set α
    x✝ : MeasurableSet s
    ⊢ MeasurableSet (Set.iUnion fun y => Set.iUnion fun h => Set.preimage f (Singl …
  -/
  refine MeasurableSet.iUnion fun y => MeasurableSet.iUnion fun hy => ?_
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : MeasurableSpace α
    inst✝¹ : Countable α
    inst✝ : MeasurableSpace β
    f : β → α
    h : ∀ (y : β), MeasurableSet (Set.preimage f (Singleton.singleton (f y)))
    s : Set α
    x✝ : MeasurableSet s
    y : α
    hy : Membership.mem s y
    ⊢ MeasurableSet (Set.preimage f (Singleton.singleton y))
  -/
  by_cases hyf : y ∈ range f
    /-
      case pos
      α : Type u_1
      β : Type u_2
      inst✝² : MeasurableSpace α
      inst✝¹ : Countable α
      inst✝ : MeasurableSpace β
      f : β → α
      h : ∀ (y : β), MeasurableSet (Set.preimage f (Singleton.singleton (f y)))
      s : Set α
      x✝ : MeasurableSet s
      y : α
      hy : Membership.mem s y
      hyf : Membership.mem (Set.range f) y
      ⊢ MeasurableSet (Set.preimage f (Singleton.singleton y))
    -/
  · rcases hyf with ⟨y, rfl⟩
    /-
      case pos.intro
      α : Type u_1
      β : Type u_2
      inst✝² : MeasurableSpace α
      inst✝¹ : Countable α
      inst✝ : MeasurableSpace β
      f : β → α
      h : ∀ (y : β), MeasurableSet (Set.preimage f (Singleton.singleton (f y)))
      s : Set α
      x✝ : MeasurableSet s
      y : β
      hy : Membership.mem s (f y)
      ⊢ MeasurableSet (Set.preimage f (Singleton.singleton (f y)))
    -/
    apply h
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      inst✝² : MeasurableSpace α
      inst✝¹ : Countable α
      inst✝ : MeasurableSpace β
      f : β → α
      h : ∀ (y : β), MeasurableSet (Set.preimage f (Singleton.singleton (f y)))
      s : Set α
      x✝ : MeasurableSet s
      y : α
      hy : Membership.mem s y
      hyf : Not (Membership.mem (Set.range f) y)
      ⊢ MeasurableSet (Set.preimage f (Singleton.singleton y))
    -/
  · simp only [preimage_singleton_eq_empty.2 hyf, MeasurableSet.empty]
    /-
      🎉 no goals
    -/


theorem measurable_to_countable' [MeasurableSpace α] [Countable α] [MeasurableSpace β] {f : β → α}
    (h : ∀ x, MeasurableSet (f ⁻¹' {x})) : Measurable f :=
  measurable_to_countable fun y => h (f y)


theorem ENat.measurable_iff {α : Type*} [MeasurableSpace α] {f : α → ℕ∞} :
    Measurable f ↔ ∀ n : ℕ, MeasurableSet (f ⁻¹' {↑n}) := by
  /-
    α : Type u_6
    inst✝ : MeasurableSpace α
    f : α → ENat
    ⊢ Iff (Measurable f) (∀ (n : Nat), MeasurableSet (Set.preimage f (Singleton.si …
  -/
  refine ⟨fun hf n ↦ hf <| measurableSet_singleton _, fun h ↦ measurable_to_countable' fun n ↦ ?_⟩
  cases n with
  | top =>
    rw [← WithTop.none_eq_top, ← compl_range_some, preimage_compl, ← iUnion_singleton_eq_range,
      preimage_iUnion]
    exact .compl <| .iUnion h
  | coe n => exact h n


@[measurability]
theorem measurable_unit [MeasurableSpace α] (f : Unit → α) : Measurable f :=
  measurable_from_top


instance _root_.ULift.instMeasurableSpace : MeasurableSpace (ULift α) :=
  ‹MeasurableSpace α›.map ULift.up


lemma measurable_down : Measurable (ULift.down : ULift α → α) := fun _ ↦ id

lemma measurable_up : Measurable (ULift.up : α → ULift α) := fun _ ↦ id


@[simp] lemma measurableSet_preimage_down {s : Set α} :
    MeasurableSet (ULift.down ⁻¹' s) ↔ MeasurableSet s := Iff.rfl

@[simp] lemma measurableSet_preimage_up {s : Set (ULift α)} :
    MeasurableSet (ULift.up ⁻¹' s) ↔ MeasurableSet s := Iff.rfl


@[measurability]
theorem measurable_from_nat {f : ℕ → α} : Measurable f :=
  measurable_from_top


theorem measurable_to_nat {f : α → ℕ} : (∀ y, MeasurableSet (f ⁻¹' {f y})) → Measurable f :=
  measurable_to_countable


theorem measurable_to_bool {f : α → Bool} (h : MeasurableSet (f ⁻¹' {true})) : Measurable f := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → Bool
    h : MeasurableSet (Set.preimage f (Singleton.singleton Bool.true))
    ⊢ Measurable f
  -/
  apply measurable_to_countable'
  /-
    case h
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → Bool
    h : MeasurableSet (Set.preimage f (Singleton.singleton Bool.true))
    ⊢ ∀ (x : Bool), MeasurableSet (Set.preimage f (Singleton.singleton x))
  -/
  rintro (- | -)
    /-
      case h.false
      α : Type u_1
      inst✝ : MeasurableSpace α
      f : α → Bool
      h : MeasurableSet (Set.preimage f (Singleton.singleton Bool.true))
      ⊢ MeasurableSet (Set.preimage f (Singleton.singleton Bool.false))
    -/
  · convert h.compl
    /-
      case h.e'_3
      α : Type u_1
      inst✝ : MeasurableSpace α
      f : α → Bool
      h : MeasurableSet (Set.preimage f (Singleton.singleton Bool.true))
      ⊢ Eq (Set.preimage f (Singleton.singleton Bool.false)) (HasCompl.compl (Set.pr …
    -/
    rw [← preimage_compl, Bool.compl_singleton, Bool.not_true]
    /-
      🎉 no goals
    -/
  /-
    case h.true
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → Bool
    h : MeasurableSet (Set.preimage f (Singleton.singleton Bool.true))
    ⊢ MeasurableSet (Set.preimage f (Singleton.singleton Bool.true))
  -/
  exact h
  /-
    🎉 no goals
  -/


theorem measurable_to_prop {f : α → Prop} (h : MeasurableSet (f ⁻¹' {True})) : Measurable f := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → Prop
    h : MeasurableSet (Set.preimage f (Singleton.singleton True))
    ⊢ Measurable f
  -/
  refine measurable_to_countable' fun x => ?_
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    f : α → Prop
    h : MeasurableSet (Set.preimage f (Singleton.singleton True))
    x : Prop
    ⊢ MeasurableSet (Set.preimage f (Singleton.singleton x))
  -/
  by_cases hx : x
    /-
      case pos
      α : Type u_1
      inst✝ : MeasurableSpace α
      f : α → Prop
      h : MeasurableSet (Set.preimage f (Singleton.singleton True))
      x : Prop
      hx : x
      ⊢ MeasurableSet (Set.preimage f (Singleton.singleton x))
    -/
  · simpa [hx] using h
    /-
      🎉 no goals
    -/
  · simpa only [hx, ← preimage_compl, Prop.compl_singleton, not_true, preimage_singleton_false]
      using h.compl


theorem measurable_findGreatest' {p : α → ℕ → Prop} [∀ x, DecidablePred (p x)] {N : ℕ}
    (hN : ∀ k ≤ N, MeasurableSet { x | Nat.findGreatest (p x) N = k }) :
    Measurable fun x => Nat.findGreatest (p x) N :=
  measurable_to_nat fun _ => hN _ N.findGreatest_le


theorem measurable_findGreatest {p : α → ℕ → Prop} [∀ x, DecidablePred (p x)] {N}
    (hN : ∀ k ≤ N, MeasurableSet { x | p x k }) : Measurable fun x => Nat.findGreatest (p x) N := by
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    p : α → Nat → Prop
    inst✝ : (x : α) → DecidablePred (p x)
    N : Nat
    hN : ∀ (k : Nat), LE.le k N → MeasurableSet (setOf fun x => p x k)
    ⊢ Measurable fun x => Nat.findGreatest (p x) N
  -/
  refine measurable_findGreatest' fun k hk => ?_
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    p : α → Nat → Prop
    inst✝ : (x : α) → DecidablePred (p x)
    N : Nat
    hN : ∀ (k : Nat), LE.le k N → MeasurableSet (setOf fun x => p x k)
    k : Nat
    hk : LE.le k N
    ⊢ MeasurableSet (setOf fun x => Eq (Nat.findGreatest (p x) N) k)
  -/
  simp only [Nat.findGreatest_eq_iff, setOf_and, setOf_forall, ← compl_setOf]
  repeat' apply_rules [MeasurableSet.inter, MeasurableSet.const, MeasurableSet.iInter,
    MeasurableSet.compl, hN] <;> try intros


theorem measurable_find {p : α → ℕ → Prop} [∀ x, DecidablePred (p x)] (hp : ∀ x, ∃ N, p x N)
    (hm : ∀ k, MeasurableSet { x | p x k }) : Measurable fun x => Nat.find (hp x) := by
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    p : α → Nat → Prop
    inst✝ : (x : α) → DecidablePred (p x)
    hp : ∀ (x : α), Exists fun N => p x N
    hm : ∀ (k : Nat), MeasurableSet (setOf fun x => p x k)
    ⊢ Measurable fun x => Nat.find ⋯
  -/
  refine measurable_to_nat fun x => ?_
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    p : α → Nat → Prop
    inst✝ : (x : α) → DecidablePred (p x)
    hp : ∀ (x : α), Exists fun N => p x N
    hm : ∀ (k : Nat), MeasurableSet (setOf fun x => p x k)
    x : α
    ⊢ MeasurableSet (Set.preimage (fun x => Nat.find ⋯) (Singleton.singleton (Nat. …
  -/
  rw [preimage_find_eq_disjointed (fun k => {x | p x k})]
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    p : α → Nat → Prop
    inst✝ : (x : α) → DecidablePred (p x)
    hp : ∀ (x : α), Exists fun N => p x N
    hm : ∀ (k : Nat), MeasurableSet (setOf fun x => p x k)
    x : α
    ⊢ MeasurableSet (disjointed (fun k => setOf fun x => p x k) (Nat.find ⋯))
  -/
  exact MeasurableSet.disjointed hm _
  /-
    🎉 no goals
  -/


instance Quot.instMeasurableSpace {α} {r : α → α → Prop} [m : MeasurableSpace α] :
    MeasurableSpace (Quot r) :=
  m.map (Quot.mk r)


instance Quotient.instMeasurableSpace {α} {s : Setoid α} [m : MeasurableSpace α] :
    MeasurableSpace (Quotient s) :=
  m.map Quotient.mk''


@[to_additive]
instance QuotientGroup.measurableSpace {G} [Group G] [MeasurableSpace G] (S : Subgroup G) :
    MeasurableSpace (G ⧸ S) :=
  Quotient.instMeasurableSpace


theorem measurableSet_quotient {s : Setoid α} {t : Set (Quotient s)} :
    MeasurableSet t ↔ MeasurableSet (Quotient.mk'' ⁻¹' t) :=
  Iff.rfl


theorem measurable_from_quotient {s : Setoid α} {f : Quotient s → β} :
    Measurable f ↔ Measurable (f ∘ Quotient.mk'') :=
  Iff.rfl


@[measurability]
theorem measurable_quotient_mk' [s : Setoid α] : Measurable (Quotient.mk' : α → Quotient s) :=
  fun _ => id


@[measurability]
theorem measurable_quotient_mk'' {s : Setoid α} : Measurable (Quotient.mk'' : α → Quotient s) :=
  fun _ => id


@[measurability]
theorem measurable_quot_mk {r : α → α → Prop} : Measurable (Quot.mk r) := fun _ => id


@[to_additive (attr := measurability)]
theorem QuotientGroup.measurable_coe {G} [Group G] [MeasurableSpace G] {S : Subgroup G} :
    Measurable ((↑) : G → G ⧸ S) :=
  measurable_quotient_mk''


@[to_additive]
nonrec theorem QuotientGroup.measurable_from_quotient {G} [Group G] [MeasurableSpace G]
    {S : Subgroup G} {f : G ⧸ S → α} : Measurable f ↔ Measurable (f ∘ ((↑) : G → G ⧸ S)) :=
  measurable_from_quotient


instance Quotient.instDiscreteMeasurableSpace {α} {s : Setoid α} [MeasurableSpace α]
    [DiscreteMeasurableSpace α] : DiscreteMeasurableSpace (Quotient s) where
  forall_measurableSet _ := measurableSet_quotient.2 .of_discrete


@[to_additive]
instance QuotientGroup.instDiscreteMeasurableSpace {G} [Group G] [MeasurableSpace G]
    [DiscreteMeasurableSpace G] (S : Subgroup G) : DiscreteMeasurableSpace (G ⧸ S) :=
  Quotient.instDiscreteMeasurableSpace


instance Subtype.instMeasurableSpace {α} {p : α → Prop} [m : MeasurableSpace α] :
    MeasurableSpace (Subtype p) :=
  m.comap ((↑) : _ → α)


@[measurability]
theorem measurable_subtype_coe {p : α → Prop} : Measurable ((↑) : Subtype p → α) :=
  MeasurableSpace.le_map_comap


instance Subtype.instMeasurableSingletonClass {p : α → Prop} [MeasurableSingletonClass α] :
    MeasurableSingletonClass (Subtype p) where
  measurableSet_singleton x :=
    ⟨{(x : α)}, measurableSet_singleton (x : α), by
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        δ : Type u_4
        δ' : Type u_5
        ι : Sort uι
        s : Set α
        inst✝¹ : MeasurableSpace α
        p : α → Prop
        inst✝ : MeasurableSingletonClass α
        x : Subtype p
        ⊢ Eq (Set.preimage Subtype.val (Singleton.singleton ↑x)) (Singleton.singleton x)
      -/
      rw [← image_singleton, preimage_image_eq _ Subtype.val_injective]⟩
      /-
        🎉 no goals
      -/


theorem MeasurableSet.of_subtype_image {s : Set α} {t : Set s}
    (h : MeasurableSet (Subtype.val '' t)) : MeasurableSet t :=
  ⟨_, h, preimage_image_eq _ Subtype.val_injective⟩


theorem MeasurableSet.subtype_image {s : Set α} {t : Set s} (hs : MeasurableSet s) :
    MeasurableSet t → MeasurableSet (((↑) : s → α) '' t) := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    s : Set α
    t : Set ↑s
    hs : MeasurableSet s
    ⊢ MeasurableSet t → MeasurableSet (Set.image Subtype.val t)
  -/
  rintro ⟨u, hu, rfl⟩
  /-
    case intro.intro
    α : Type u_1
    m : MeasurableSpace α
    s : Set α
    hs : MeasurableSet s
    u : Set α
    hu : MeasurableSet u
    ⊢ MeasurableSet (Set.image Subtype.val (Set.preimage Subtype.val u))
  -/
  rw [Subtype.image_preimage_coe]
  /-
    case intro.intro
    α : Type u_1
    m : MeasurableSpace α
    s : Set α
    hs : MeasurableSet s
    u : Set α
    hu : MeasurableSet u
    ⊢ MeasurableSet (Inter.inter s u)
  -/
  exact hs.inter hu
  /-
    🎉 no goals
  -/


@[measurability]
theorem Measurable.subtype_coe {p : β → Prop} {f : α → Subtype p} (hf : Measurable f) :
    Measurable fun a : α => (f a : β) :=
  measurable_subtype_coe.comp hf


alias Measurable.subtype_val := Measurable.subtype_coe


@[measurability]
theorem Measurable.subtype_mk {p : β → Prop} {f : α → β} (hf : Measurable f) {h : ∀ x, p (f x)} :
    Measurable fun x => (⟨f x, h x⟩ : Subtype p) := fun t ⟨s, hs⟩ =>
            /-
              α : Type u_1
              β : Type u_2
              m : MeasurableSpace α
              mβ : MeasurableSpace β
              p : β → Prop
              f : α → β
              hf : Measurable f
              h : ∀ (x : α), p (f x)
              t : Set (Subtype p)
              x✝ : MeasurableSet t
              s : Set β
              hs : And (MeasurableSet s) (Eq (Set.preimage Subtype.val s) t)
              ⊢ MeasurableSet (Set.preimage (fun x => ⟨f x, ⋯⟩) (Set.preimage Subtype.val s))
            -/
  hs.2 ▸ by simp only [← preimage_comp, Function.comp_def, Subtype.coe_mk, hf hs.1]
            /-
              🎉 no goals
            -/


@[measurability]
protected theorem Measurable.rangeFactorization {f : α → β} (hf : Measurable f) :
    Measurable (rangeFactorization f) :=
  hf.subtype_mk


theorem Measurable.subtype_map {f : α → β} {p : α → Prop} {q : β → Prop} (hf : Measurable f)
    (hpq : ∀ x, p x → q (f x)) : Measurable (Subtype.map f hpq) :=
  (hf.comp measurable_subtype_coe).subtype_mk


theorem measurable_inclusion {s t : Set α} (h : s ⊆ t) : Measurable (inclusion h) :=
  measurable_id.subtype_map h


theorem MeasurableSet.image_inclusion' {s t : Set α} (h : s ⊆ t) {u : Set s}
    (hs : MeasurableSet (Subtype.val ⁻¹' s : Set t)) (hu : MeasurableSet u) :
    MeasurableSet (inclusion h '' u) := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    s t : Set α
    h : HasSubset.Subset s t
    u : Set ↑s
    hs : MeasurableSet (Set.preimage Subtype.val s)
    hu : MeasurableSet u
    ⊢ MeasurableSet (Set.image (Set.inclusion h) u)
  -/
  rcases hu with ⟨u, hu, rfl⟩
  /-
    case intro.intro
    α : Type u_1
    m : MeasurableSpace α
    s t : Set α
    h : HasSubset.Subset s t
    hs : MeasurableSet (Set.preimage Subtype.val s)
    u : Set α
    hu : MeasurableSet u
    ⊢ MeasurableSet (Set.image (Set.inclusion h) (Set.preimage Subtype.val u))
  -/
  convert (measurable_subtype_coe hu).inter hs
  /-
    case h.e'_3
    α : Type u_1
    m : MeasurableSpace α
    s t : Set α
    h : HasSubset.Subset s t
    hs : MeasurableSet (Set.preimage Subtype.val s)
    u : Set α
    hu : MeasurableSet u
    ⊢ Eq (Set.image (Set.inclusion h) (Set.preimage Subtype.val u)) (Inter.inter ( …
  -/
  ext ⟨x, hx⟩
  /-
    case h.e'_3.h.mk
    α : Type u_1
    m : MeasurableSpace α
    s t : Set α
    h : HasSubset.Subset s t
    hs : MeasurableSet (Set.preimage Subtype.val s)
    u : Set α
    hu : MeasurableSet u
    x : α
    hx : Membership.mem t x
    ⊢ Iff (Membership.mem (Set.image (Set.inclusion h) (Set.preimage Subtype.val u …
  -/
  simpa [@and_comm _ (_ = x)] using and_comm
  /-
    🎉 no goals
  -/


theorem MeasurableSet.image_inclusion {s t : Set α} (h : s ⊆ t) {u : Set s}
    (hs : MeasurableSet s) (hu : MeasurableSet u) :
    MeasurableSet (inclusion h '' u) :=
  (measurable_subtype_coe hs).image_inclusion' h hu


theorem MeasurableSet.of_union_cover {s t u : Set α} (hs : MeasurableSet s) (ht : MeasurableSet t)
    (h : univ ⊆ s ∪ t) (hsu : MeasurableSet (((↑) : s → α) ⁻¹' u))
    (htu : MeasurableSet (((↑) : t → α) ⁻¹' u)) : MeasurableSet u := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    s t u : Set α
    hs : MeasurableSet s
    ht : MeasurableSet t
    h : HasSubset.Subset Set.univ (Union.union s t)
    hsu : MeasurableSet (Set.preimage Subtype.val u)
    htu : MeasurableSet (Set.preimage Subtype.val u)
    ⊢ MeasurableSet u
  -/
  convert (hs.subtype_image hsu).union (ht.subtype_image htu)
  /-
    case h.e'_3
    α : Type u_1
    m : MeasurableSpace α
    s t u : Set α
    hs : MeasurableSet s
    ht : MeasurableSet t
    h : HasSubset.Subset Set.univ (Union.union s t)
    hsu : MeasurableSet (Set.preimage Subtype.val u)
    htu : MeasurableSet (Set.preimage Subtype.val u)
    ⊢ Eq u (Union.union (Set.image Subtype.val (Set.preimage Subtype.val u)) (Set. …
  -/
  simp [image_preimage_eq_inter_range, ← inter_union_distrib_left, univ_subset_iff.1 h]
  /-
    🎉 no goals
  -/


theorem measurable_of_measurable_union_cover {f : α → β} (s t : Set α) (hs : MeasurableSet s)
    (ht : MeasurableSet t) (h : univ ⊆ s ∪ t) (hc : Measurable fun a : s => f a)
    (hd : Measurable fun a : t => f a) : Measurable f := fun _u hu =>
  .of_union_cover hs ht h (hc hu) (hd hu)


theorem measurable_of_restrict_of_restrict_compl {f : α → β} {s : Set α} (hs : MeasurableSet s)
    (h₁ : Measurable (s.restrict f)) (h₂ : Measurable (sᶜ.restrict f)) : Measurable f :=
  measurable_of_measurable_union_cover s sᶜ hs hs.compl (union_compl_self s).ge h₁ h₂


theorem Measurable.dite [∀ x, Decidable (x ∈ s)] {f : s → β} (hf : Measurable f)
    {g : (sᶜ : Set α) → β} (hg : Measurable g) (hs : MeasurableSet s) :
    Measurable fun x => if hx : x ∈ s then f ⟨x, hx⟩ else g ⟨x, hx⟩ :=
                                                  /-
                                                    α : Type u_1
                                                    β : Type u_2
                                                    s : Set α
                                                    m : MeasurableSpace α
                                                    mβ : MeasurableSpace β
                                                    inst✝ : (x : α) → Decidable (Membership.mem s x)
                                                    f : ↑s → β
                                                    hf : Measurable f
                                                    g : ↑(HasCompl.compl s) → β
                                                    hg : Measurable g
                                                    hs : MeasurableSet s
                                                    ⊢ Measurable (s.restrict fun x => _root_.dite (Membership.mem s x) (fun hx =>  …
                                                  -/
                                                  /-
                                                    🎉 no goals
                                                  -/
  measurable_of_restrict_of_restrict_compl hs (by simpa) (by simpa)
                                                             /-
                                                               🎉 no goals
                                                             -/


theorem measurable_of_measurable_on_compl_finite [MeasurableSingletonClass α] {f : α → β}
    (s : Set α) (hs : s.Finite) (hf : Measurable (sᶜ.restrict f)) : Measurable f :=
  have := hs.to_subtype
  measurable_of_restrict_of_restrict_compl hs.measurableSet (measurable_of_finite _) hf


theorem measurable_of_measurable_on_compl_singleton [MeasurableSingletonClass α] {f : α → β} (a : α)
    (hf : Measurable ({ x | x ≠ a }.restrict f)) : Measurable f :=
  measurable_of_measurable_on_compl_finite {a} (finite_singleton a) hf


/-- The *measurable atom* of `x` is the intersection of all the measurable sets countaining `x`.
It is measurable when the space is countable (or more generally when the measurable space is
countably generated). -/
def measurableAtom (x : β) : Set β :=
  ⋂ (s : Set β) (_h's : x ∈ s) (_hs : MeasurableSet s), s


@[simp] lemma mem_measurableAtom_self (x : β) : x ∈ measurableAtom x := by
  /-
    β : Type u_2
    inst✝ : MeasurableSpace β
    x : β
    ⊢ Membership.mem (measurableAtom x) x
  -/
  simp +contextual [measurableAtom]
  /-
    🎉 no goals
  -/


lemma mem_of_mem_measurableAtom {x y : β} (h : y ∈ measurableAtom x) {s : Set β}
    (hs : MeasurableSet s) (hxs : x ∈ s) : y ∈ s := by
  /-
    β : Type u_2
    inst✝ : MeasurableSpace β
    x y : β
    h : Membership.mem (measurableAtom x) y
    s : Set β
    hs : MeasurableSet s
    hxs : Membership.mem s x
    ⊢ Membership.mem s y
  -/
  simp only [measurableAtom, mem_iInter] at h
  /-
    β : Type u_2
    inst✝ : MeasurableSpace β
    x y : β
    s : Set β
    hs : MeasurableSet s
    hxs : Membership.mem s x
    h : ∀ (i : Set β), Membership.mem i x → MeasurableSet i → Membership.mem i y
    ⊢ Membership.mem s y
  -/
  exact h s hxs hs
  /-
    🎉 no goals
  -/


lemma measurableAtom_subset {s : Set β} {x : β} (hs : MeasurableSet s) (hx : x ∈ s) :
    measurableAtom x ⊆ s :=
                                              /-
                                                β : Type u_2
                                                inst✝ : MeasurableSpace β
                                                s : Set β
                                                x : β
                                                hs : MeasurableSet s
                                                hx : Membership.mem s x
                                                a : β
                                                ⊢ Membership.mem (Set.iInter fun _hs => s) a → Membership.mem s a
                                              -/
  iInter₂_subset_of_subset s hx fun ⦃a⦄ ↦ (by simp [hs])
                                              /-
                                                🎉 no goals
                                              -/


@[simp] lemma measurableAtom_of_measurableSingletonClass [MeasurableSingletonClass β] (x : β) :
    measurableAtom x = {x} :=
                                                                              /-
                                                                                β : Type u_2
                                                                                inst✝¹ : MeasurableSpace β
                                                                                inst✝ : MeasurableSingletonClass β
                                                                                x : β
                                                                                ⊢ HasSubset.Subset (Singleton.singleton x) (measurableAtom x)
                                                                              -/
  Subset.antisymm (measurableAtom_subset (measurableSet_singleton x) rfl) (by simp)
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


lemma MeasurableSet.measurableAtom_of_countable [Countable β] (x : β) :
    MeasurableSet (measurableAtom x) := by
  have : ∀ (y : β), y ∉ measurableAtom x → ∃ s, x ∈ s ∧ MeasurableSet s ∧ y ∉ s :=
    fun y hy ↦ by simpa [measurableAtom] using hy
  /-
    β : Type u_2
    inst✝¹ : MeasurableSpace β
    inst✝ : Countable β
    x : β
    this : ∀ (y : β), Not (Membership.mem (measurableAtom x) y) → Exists fun s =>  …
    ⊢ MeasurableSet (measurableAtom x)
  -/
  choose! s hs using this
  have : measurableAtom x = ⋂ (y ∈ (measurableAtom x)ᶜ), s y := by
    apply Subset.antisymm
    · intro z hz
      simp only [mem_iInter, mem_compl_iff]
      intro i hi
      exact mem_of_mem_measurableAtom hz (hs i hi).2.1 (hs i hi).1
    · apply compl_subset_compl.1
      intro z hz
      simp only [compl_iInter, mem_iUnion, mem_compl_iff, exists_prop]
      exact ⟨z, hz, (hs z hz).2.2⟩
  /-
    β : Type u_2
    inst✝¹ : MeasurableSpace β
    inst✝ : Countable β
    x : β
    s : β → Set β
    hs : ∀ (y : β), Not (Membership.mem (measurableAtom x) y) → And (Membership.me …
    this : Eq (measurableAtom x) (Set.iInter fun y => Set.iInter fun h => s y)
    ⊢ MeasurableSet (measurableAtom x)
  -/
  rw [this]
  /-
    β : Type u_2
    inst✝¹ : MeasurableSpace β
    inst✝ : Countable β
    x : β
    s : β → Set β
    hs : ∀ (y : β), Not (Membership.mem (measurableAtom x) y) → And (Membership.me …
    this : Eq (measurableAtom x) (Set.iInter fun y => Set.iInter fun h => s y)
    ⊢ MeasurableSet (Set.iInter fun y => Set.iInter fun h => s y)
  -/
  exact MeasurableSet.biInter (to_countable (measurableAtom x)ᶜ) (fun i hi ↦ (hs i hi).2.1)
  /-
    🎉 no goals
  -/


/-- A `MeasurableSpace` structure on the product of two measurable spaces. -/
def MeasurableSpace.prod {α β} (m₁ : MeasurableSpace α) (m₂ : MeasurableSpace β) :
    MeasurableSpace (α × β) :=
  m₁.comap Prod.fst ⊔ m₂.comap Prod.snd


instance Prod.instMeasurableSpace {α β} [m₁ : MeasurableSpace α] [m₂ : MeasurableSpace β] :
    MeasurableSpace (α × β) :=
  m₁.prod m₂


@[measurability]
theorem measurable_fst {_ : MeasurableSpace α} {_ : MeasurableSpace β} :
    Measurable (Prod.fst : α × β → α) :=
  Measurable.of_comap_le le_sup_left


@[measurability]
theorem measurable_snd {_ : MeasurableSpace α} {_ : MeasurableSpace β} :
    Measurable (Prod.snd : α × β → β) :=
  Measurable.of_comap_le le_sup_right


@[fun_prop]
theorem Measurable.fst {f : α → β × γ} (hf : Measurable f) : Measurable fun a : α => (f a).1 :=
  measurable_fst.comp hf


@[fun_prop]
theorem Measurable.snd {f : α → β × γ} (hf : Measurable f) : Measurable fun a : α => (f a).2 :=
  measurable_snd.comp hf


@[measurability]
theorem Measurable.prod {f : α → β × γ} (hf₁ : Measurable fun a => (f a).1)
    (hf₂ : Measurable fun a => (f a).2) : Measurable f :=
  Measurable.of_le_map <|
    sup_le
      (by
        /-
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          m : MeasurableSpace α
          mβ : MeasurableSpace β
          mγ : MeasurableSpace γ
          f : α → Prod β γ
          hf₁ : Measurable fun a => (f a).1
          hf₂ : Measurable fun a => (f a).2
          ⊢ LE.le (MeasurableSpace.comap Prod.fst mβ) (MeasurableSpace.map f m)
        -/
        rw [MeasurableSpace.comap_le_iff_le_map, MeasurableSpace.map_comp]
        /-
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          m : MeasurableSpace α
          mβ : MeasurableSpace β
          mγ : MeasurableSpace γ
          f : α → Prod β γ
          hf₁ : Measurable fun a => (f a).1
          hf₂ : Measurable fun a => (f a).2
          ⊢ LE.le mβ (MeasurableSpace.map (Function.comp Prod.fst f) m)
        -/
        exact hf₁)
        /-
          🎉 no goals
        -/
      (by
        /-
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          m : MeasurableSpace α
          mβ : MeasurableSpace β
          mγ : MeasurableSpace γ
          f : α → Prod β γ
          hf₁ : Measurable fun a => (f a).1
          hf₂ : Measurable fun a => (f a).2
          ⊢ LE.le (MeasurableSpace.comap Prod.snd mγ) (MeasurableSpace.map f m)
        -/
        rw [MeasurableSpace.comap_le_iff_le_map, MeasurableSpace.map_comp]
        /-
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          m : MeasurableSpace α
          mβ : MeasurableSpace β
          mγ : MeasurableSpace γ
          f : α → Prod β γ
          hf₁ : Measurable fun a => (f a).1
          hf₂ : Measurable fun a => (f a).2
          ⊢ LE.le mγ (MeasurableSpace.map (Function.comp Prod.snd f) m)
        -/
        exact hf₂)
        /-
          🎉 no goals
        -/


@[fun_prop]
theorem Measurable.prod_mk {β γ} {_ : MeasurableSpace β} {_ : MeasurableSpace γ} {f : α → β}
    {g : α → γ} (hf : Measurable f) (hg : Measurable g) : Measurable fun a : α => (f a, g a) :=
  Measurable.prod hf hg


@[fun_prop]
theorem Measurable.prod_map [MeasurableSpace δ] {f : α → β} {g : γ → δ} (hf : Measurable f)
    (hg : Measurable g) : Measurable (Prod.map f g) :=
  (hf.comp measurable_fst).prod_mk (hg.comp measurable_snd)


theorem measurable_prod_mk_left {x : α} : Measurable (@Prod.mk _ β x) :=
  measurable_const.prod_mk measurable_id


theorem measurable_prod_mk_right {y : β} : Measurable fun x : α => (x, y) :=
  measurable_id.prod_mk measurable_const


theorem Measurable.of_uncurry_left {f : α → β → γ} (hf : Measurable (uncurry f)) {x : α} :
    Measurable (f x) :=
  hf.comp measurable_prod_mk_left


theorem Measurable.of_uncurry_right {f : α → β → γ} (hf : Measurable (uncurry f)) {y : β} :
    Measurable fun x => f x y :=
  hf.comp measurable_prod_mk_right


theorem measurable_prod {f : α → β × γ} :
    Measurable f ↔ (Measurable fun a => (f a).1) ∧ Measurable fun a => (f a).2 :=
  ⟨fun hf => ⟨measurable_fst.comp hf, measurable_snd.comp hf⟩, fun h => Measurable.prod h.1 h.2⟩


@[fun_prop, measurability]
theorem measurable_swap : Measurable (Prod.swap : α × β → β × α) :=
  Measurable.prod measurable_snd measurable_fst


theorem measurable_swap_iff {_ : MeasurableSpace γ} {f : α × β → γ} :
    Measurable (f ∘ Prod.swap) ↔ Measurable f :=
  ⟨fun hf => hf.comp measurable_swap, fun hf => hf.comp measurable_swap⟩


@[measurability]
protected theorem MeasurableSet.prod {s : Set α} {t : Set β} (hs : MeasurableSet s)
    (ht : MeasurableSet t) : MeasurableSet (s ×ˢ t) :=
  MeasurableSet.inter (measurable_fst hs) (measurable_snd ht)


theorem measurableSet_prod_of_nonempty {s : Set α} {t : Set β} (h : (s ×ˢ t).Nonempty) :
    MeasurableSet (s ×ˢ t) ↔ MeasurableSet s ∧ MeasurableSet t := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    mβ : MeasurableSpace β
    s : Set α
    t : Set β
    h : (SProd.sprod s t).Nonempty
    ⊢ Iff (MeasurableSet (SProd.sprod s t)) (And (MeasurableSet s) (MeasurableSet  …
  -/
  rcases h with ⟨⟨x, y⟩, hx, hy⟩
  /-
    case intro.mk.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    mβ : MeasurableSpace β
    s : Set α
    t : Set β
    x : α
    y : β
    hx : Membership.mem s { fst := x, snd := y }.1
    hy : Membership.mem t { fst := x, snd := y }.2
    ⊢ Iff (MeasurableSet (SProd.sprod s t)) (And (MeasurableSet s) (MeasurableSet  …
  -/
  refine ⟨fun hst => ?_, fun h => h.1.prod h.2⟩
  /-
    case intro.mk.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    mβ : MeasurableSpace β
    s : Set α
    t : Set β
    x : α
    y : β
    hx : Membership.mem s { fst := x, snd := y }.1
    hy : Membership.mem t { fst := x, snd := y }.2
    hst : MeasurableSet (SProd.sprod s t)
    ⊢ And (MeasurableSet s) (MeasurableSet t)
  -/
  have : MeasurableSet ((fun x => (x, y)) ⁻¹' s ×ˢ t) := measurable_prod_mk_right hst
  /-
    case intro.mk.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    mβ : MeasurableSpace β
    s : Set α
    t : Set β
    x : α
    y : β
    hx : Membership.mem s { fst := x, snd := y }.1
    hy : Membership.mem t { fst := x, snd := y }.2
    hst : MeasurableSet (SProd.sprod s t)
    this : MeasurableSet (Set.preimage (fun x => { fst := x, snd := y }) (SProd.sp …
    ⊢ And (MeasurableSet s) (MeasurableSet t)
  -/
  have : MeasurableSet (Prod.mk x ⁻¹' s ×ˢ t) := measurable_prod_mk_left hst
  /-
    case intro.mk.intro
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    mβ : MeasurableSpace β
    s : Set α
    t : Set β
    x : α
    y : β
    hx : Membership.mem s { fst := x, snd := y }.1
    hy : Membership.mem t { fst := x, snd := y }.2
    hst : MeasurableSet (SProd.sprod s t)
    this✝ : MeasurableSet (Set.preimage (fun x => { fst := x, snd := y }) (SProd.s …
    this : MeasurableSet (Set.preimage (Prod.mk x) (SProd.sprod s t))
    ⊢ And (MeasurableSet s) (MeasurableSet t)
  -/
  simp_all
  /-
    🎉 no goals
  -/


theorem measurableSet_prod {s : Set α} {t : Set β} :
    MeasurableSet (s ×ˢ t) ↔ MeasurableSet s ∧ MeasurableSet t ∨ s = ∅ ∨ t = ∅ := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    mβ : MeasurableSpace β
    s : Set α
    t : Set β
    ⊢ Iff (MeasurableSet (SProd.sprod s t)) (Or (And (MeasurableSet s) (Measurable …
  -/
  rcases (s ×ˢ t).eq_empty_or_nonempty with h | h
    /-
      case inl
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      mβ : MeasurableSpace β
      s : Set α
      t : Set β
      h : Eq (SProd.sprod s t) EmptyCollection.emptyCollection
      ⊢ Iff (MeasurableSet (SProd.sprod s t)) (Or (And (MeasurableSet s) (Measurable …
    -/
  · simp [h, prod_eq_empty_iff.mp h]
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      β : Type u_2
      m : MeasurableSpace α
      mβ : MeasurableSpace β
      s : Set α
      t : Set β
      h : (SProd.sprod s t).Nonempty
      ⊢ Iff (MeasurableSet (SProd.sprod s t)) (Or (And (MeasurableSet s) (Measurable …
    -/
  · simp [← not_nonempty_iff_eq_empty, prod_nonempty_iff.mp h, measurableSet_prod_of_nonempty h]
    /-
      🎉 no goals
    -/


theorem measurableSet_swap_iff {s : Set (α × β)} :
    MeasurableSet (Prod.swap ⁻¹' s) ↔ MeasurableSet s :=
  ⟨fun hs => measurable_swap hs, fun hs => measurable_swap hs⟩


instance Prod.instMeasurableSingletonClass
    [MeasurableSingletonClass α] [MeasurableSingletonClass β] :
    MeasurableSingletonClass (α × β) :=
  ⟨fun ⟨a, b⟩ => @singleton_prod_singleton _ _ a b ▸ .prod (.singleton a) (.singleton b)⟩


theorem measurable_from_prod_countable' [Countable β]
    {_ : MeasurableSpace γ} {f : α × β → γ} (hf : ∀ y, Measurable fun x => f (x, y))
    (h'f : ∀ y y' x, y' ∈ measurableAtom y → f (x, y') = f (x, y)) :
    Measurable f := fun s hs => by
  have : f ⁻¹' s = ⋃ y, ((fun x => f (x, y)) ⁻¹' s) ×ˢ (measurableAtom y : Set β) := by
    ext1 ⟨x, y⟩
    simp only [mem_preimage, mem_iUnion, mem_prod]
    refine ⟨fun h ↦ ⟨y, h, mem_measurableAtom_self y⟩, ?_⟩
    rintro ⟨y', hy's, hy'⟩
    rwa [h'f y' y x hy']
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    m : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝ : Countable β
    x✝ : MeasurableSpace γ
    f : Prod α β → γ
    hf : ∀ (y : β), Measurable fun x => f { fst := x, snd := y }
    h'f : ∀ (y y' : β) (x : α), Membership.mem (measurableAtom y) y' → Eq (f { fst …
    s : Set γ
    hs : MeasurableSet s
    this : Eq (Set.preimage f s) (Set.iUnion fun y => SProd.sprod (Set.preimage (f …
    ⊢ MeasurableSet (Set.preimage f s)
  -/
  rw [this]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    m : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝ : Countable β
    x✝ : MeasurableSpace γ
    f : Prod α β → γ
    hf : ∀ (y : β), Measurable fun x => f { fst := x, snd := y }
    h'f : ∀ (y y' : β) (x : α), Membership.mem (measurableAtom y) y' → Eq (f { fst …
    s : Set γ
    hs : MeasurableSet s
    this : Eq (Set.preimage f s) (Set.iUnion fun y => SProd.sprod (Set.preimage (f …
    ⊢ MeasurableSet (Set.iUnion fun y => SProd.sprod (Set.preimage (fun x => f { f …
  -/
  exact .iUnion (fun y ↦ (hf y hs).prod (.measurableAtom_of_countable y))
  /-
    🎉 no goals
  -/


theorem measurable_from_prod_countable [Countable β] [MeasurableSingletonClass β]
    {_ : MeasurableSpace γ} {f : α × β → γ} (hf : ∀ y, Measurable fun x => f (x, y)) :
    Measurable f :=
                                         /-
                                           α : Type u_1
                                           β : Type u_2
                                           γ : Type u_3
                                           m : MeasurableSpace α
                                           mβ : MeasurableSpace β
                                           inst✝¹ : Countable β
                                           inst✝ : MeasurableSingletonClass β
                                           x✝ : MeasurableSpace γ
                                           f : Prod α β → γ
                                           hf : ∀ (y : β), Measurable fun x => f { fst := x, snd := y }
                                           ⊢ ∀ (y y' : β) (x : α), Membership.mem (measurableAtom y) y' → Eq (f { fst :=  …
                                         -/
  measurable_from_prod_countable' hf (by simp +contextual)
                                         /-
                                           🎉 no goals
                                         -/


/-- A piecewise function on countably many pieces is measurable if all the data is measurable. -/
@[measurability]
theorem Measurable.find {_ : MeasurableSpace α} {f : ℕ → α → β} {p : ℕ → α → Prop}
    [∀ n, DecidablePred (p n)] (hf : ∀ n, Measurable (f n)) (hp : ∀ n, MeasurableSet { x | p n x })
    (h : ∀ x, ∃ n, p n x) : Measurable fun x => f (Nat.find (h x)) x :=
  have : Measurable fun p : α × ℕ => f p.2 p.1 := measurable_from_prod_countable fun n => hf n
  this.comp (Measurable.prod_mk measurable_id (measurable_find h hp))


/-- Let `t i` be a countable covering of a set `T` by measurable sets. Let `f i : t i → β` be a
family of functions that agree on the intersections `t i ∩ t j`. Then the function
`Set.iUnionLift t f _ _ : T → β`, defined as `f i ⟨x, hx⟩` for `hx : x ∈ t i`, is measurable. -/
theorem measurable_iUnionLift [Countable ι] {t : ι → Set α} {f : ∀ i, t i → β}
    (htf : ∀ (i j) (x : α) (hxi : x ∈ t i) (hxj : x ∈ t j), f i ⟨x, hxi⟩ = f j ⟨x, hxj⟩)
    {T : Set α} (hT : T ⊆ ⋃ i, t i) (htm : ∀ i, MeasurableSet (t i)) (hfm : ∀ i, Measurable (f i)) :
    Measurable (iUnionLift t f htf T hT) := fun s hs => by
  /-
    α : Type u_1
    β : Type u_2
    ι : Sort uι
    m : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝ : Countable ι
    t : ι → Set α
    f : (i : ι) → ↑(t i) → β
    htf : ∀ (i j : ι) (x : α) (hxi : Membership.mem (t i) x) (hxj : Membership.mem …
    T : Set α
    hT : HasSubset.Subset T (Set.iUnion fun i => t i)
    htm : ∀ (i : ι), MeasurableSet (t i)
    hfm : ∀ (i : ι), Measurable (f i)
    s : Set β
    hs : MeasurableSet s
    ⊢ MeasurableSet (Set.preimage (Set.iUnionLift t f htf T hT) s)
  -/
  rw [preimage_iUnionLift]
  /-
    α : Type u_1
    β : Type u_2
    ι : Sort uι
    m : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝ : Countable ι
    t : ι → Set α
    f : (i : ι) → ↑(t i) → β
    htf : ∀ (i j : ι) (x : α) (hxi : Membership.mem (t i) x) (hxj : Membership.mem …
    T : Set α
    hT : HasSubset.Subset T (Set.iUnion fun i => t i)
    htm : ∀ (i : ι), MeasurableSet (t i)
    hfm : ∀ (i : ι), Measurable (f i)
    s : Set β
    hs : MeasurableSet s
    ⊢ MeasurableSet (Set.preimage (Set.inclusion hT) (Set.iUnion fun i => Set.imag …
  -/
  exact .preimage (.iUnion fun i => .image_inclusion _ (htm _) (hfm i hs)) (measurable_inclusion _)
  /-
    🎉 no goals
  -/


/-- Let `t i` be a countable covering of `α` by measurable sets. Let `f i : t i → β` be a family of
functions that agree on the intersections `t i ∩ t j`. Then the function `Set.liftCover t f _ _`,
defined as `f i ⟨x, hx⟩` for `hx : x ∈ t i`, is measurable. -/
theorem measurable_liftCover [Countable ι] (t : ι → Set α) (htm : ∀ i, MeasurableSet (t i))
    (f : ∀ i, t i → β) (hfm : ∀ i, Measurable (f i))
    (hf : ∀ (i j) (x : α) (hxi : x ∈ t i) (hxj : x ∈ t j), f i ⟨x, hxi⟩ = f j ⟨x, hxj⟩)
    (htU : ⋃ i, t i = univ) :
    Measurable (liftCover t f hf htU) := fun s hs => by
  /-
    α : Type u_1
    β : Type u_2
    ι : Sort uι
    m : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝ : Countable ι
    t : ι → Set α
    htm : ∀ (i : ι), MeasurableSet (t i)
    f : (i : ι) → ↑(t i) → β
    hfm : ∀ (i : ι), Measurable (f i)
    hf : ∀ (i j : ι) (x : α) (hxi : Membership.mem (t i) x) (hxj : Membership.mem  …
    htU : Eq (Set.iUnion fun i => t i) Set.univ
    s : Set β
    hs : MeasurableSet s
    ⊢ MeasurableSet (Set.preimage (Set.liftCover t f hf htU) s)
  -/
  rw [preimage_liftCover]
  /-
    α : Type u_1
    β : Type u_2
    ι : Sort uι
    m : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝ : Countable ι
    t : ι → Set α
    htm : ∀ (i : ι), MeasurableSet (t i)
    f : (i : ι) → ↑(t i) → β
    hfm : ∀ (i : ι), Measurable (f i)
    hf : ∀ (i j : ι) (x : α) (hxi : Membership.mem (t i) x) (hxj : Membership.mem  …
    htU : Eq (Set.iUnion fun i => t i) Set.univ
    s : Set β
    hs : MeasurableSet s
    ⊢ MeasurableSet (Set.iUnion fun i => Set.image Subtype.val (Set.preimage (f i) …
  -/
  exact .iUnion fun i => .subtype_image (htm i) <| hfm i hs
  /-
    🎉 no goals
  -/


/-- Let `t i` be a nonempty countable family of measurable sets in `α`. Let `g i : α → β` be a
family of measurable functions such that `g i` agrees with `g j` on `t i ∩ t j`. Then there exists
a measurable function `f : α → β` that agrees with each `g i` on `t i`.

We only need the assumption `[Nonempty ι]` to prove `[Nonempty (α → β)]`. -/
theorem exists_measurable_piecewise {ι} [Countable ι] [Nonempty ι] (t : ι → Set α)
    (t_meas : ∀ n, MeasurableSet (t n)) (g : ι → α → β) (hg : ∀ n, Measurable (g n))
    (ht : Pairwise fun i j => EqOn (g i) (g j) (t i ∩ t j)) :
    ∃ f : α → β, Measurable f ∧ ∀ n, EqOn f (g n) (t n) := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    mβ : MeasurableSpace β
    ι : Type u_6
    inst✝¹ : Countable ι
    inst✝ : Nonempty ι
    t : ι → Set α
    t_meas : ∀ (n : ι), MeasurableSet (t n)
    g : ι → α → β
    hg : ∀ (n : ι), Measurable (g n)
    ht : Pairwise fun i j => Set.EqOn (g i) (g j) (Inter.inter (t i) (t j))
    ⊢ Exists fun f => And (Measurable f) (∀ (n : ι), Set.EqOn f (g n) (t n))
  -/
  inhabit ι
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    mβ : MeasurableSpace β
    ι : Type u_6
    inst✝¹ : Countable ι
    inst✝ : Nonempty ι
    t : ι → Set α
    t_meas : ∀ (n : ι), MeasurableSet (t n)
    g : ι → α → β
    hg : ∀ (n : ι), Measurable (g n)
    ht : Pairwise fun i j => Set.EqOn (g i) (g j) (Inter.inter (t i) (t j))
    inhabited_h : Inhabited ι
    ⊢ Exists fun f => And (Measurable f) (∀ (n : ι), Set.EqOn f (g n) (t n))
  -/
  set g' : (i : ι) → t i → β := fun i => g i ∘ (↑)
  -- see https://github.com/leanprover-community/mathlib4/issues/2184
  have ht' : ∀ (i j) (x : α) (hxi : x ∈ t i) (hxj : x ∈ t j), g' i ⟨x, hxi⟩ = g' j ⟨x, hxj⟩ := by
    intro i j x hxi hxj
    rcases eq_or_ne i j with rfl | hij
    · rfl
    · exact ht hij ⟨hxi, hxj⟩
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    mβ : MeasurableSpace β
    ι : Type u_6
    inst✝¹ : Countable ι
    inst✝ : Nonempty ι
    t : ι → Set α
    t_meas : ∀ (n : ι), MeasurableSet (t n)
    g : ι → α → β
    hg : ∀ (n : ι), Measurable (g n)
    ht : Pairwise fun i j => Set.EqOn (g i) (g j) (Inter.inter (t i) (t j))
    inhabited_h : Inhabited ι
    g' : (i : ι) → ↑(t i) → β := fun i => Function.comp (g i) Subtype.val
    ht' : ∀ (i j : ι) (x : α) (hxi : Membership.mem (t i) x) (hxj : Membership.mem …
    ⊢ Exists fun f => And (Measurable f) (∀ (n : ι), Set.EqOn f (g n) (t n))
  -/
  set f : (⋃ i, t i) → β := iUnionLift t g' ht' _ Subset.rfl
  have hfm : Measurable f := measurable_iUnionLift _ _ t_meas
    (fun i => (hg i).comp measurable_subtype_coe)
  classical
    refine ⟨fun x => if hx : x ∈ ⋃ i, t i then f ⟨x, hx⟩ else g default x,
      hfm.dite ((hg default).comp measurable_subtype_coe) (.iUnion t_meas), fun i x hx => ?_⟩
    simp only [dif_pos (mem_iUnion.2 ⟨i, hx⟩)]
    exact iUnionLift_of_mem ⟨x, mem_iUnion.2 ⟨i, hx⟩⟩ hx


instance MeasurableSpace.pi [m : ∀ a, MeasurableSpace (π a)] : MeasurableSpace (∀ a, π a) :=
  ⨆ a, (m a).comap fun b => b a


theorem measurable_pi_iff {g : α → ∀ a, π a} : Measurable g ↔ ∀ a, Measurable fun x => g x a := by
  simp_rw [measurable_iff_comap_le, MeasurableSpace.pi, MeasurableSpace.comap_iSup,
    MeasurableSpace.comap_comp, Function.comp_def, iSup_le_iff]


@[fun_prop, aesop safe 100 apply (rule_sets := [Measurable])]
theorem measurable_pi_apply (a : δ) : Measurable fun f : ∀ a, π a => f a :=
  measurable_pi_iff.1 measurable_id a


@[aesop safe 100 apply (rule_sets := [Measurable])]
theorem Measurable.eval {a : δ} {g : α → ∀ a, π a} (hg : Measurable g) :
    Measurable fun x => g x a :=
  (measurable_pi_apply a).comp hg


@[fun_prop, aesop safe 100 apply (rule_sets := [Measurable])]
theorem measurable_pi_lambda (f : α → ∀ a, π a) (hf : ∀ a, Measurable fun c => f c a) :
    Measurable f :=
  measurable_pi_iff.mpr hf


/-- The function `(f, x) ↦ update f a x : (Π a, π a) × π a → Π a, π a` is measurable. -/
theorem measurable_update'  {a : δ} [DecidableEq δ] :
    Measurable (fun p : (∀ i, π i) × π a ↦ update p.1 a p.2) := by
  /-
    δ : Type u_4
    π : δ → Type u_6
    inst✝¹ : (a : δ) → MeasurableSpace (π a)
    a : δ
    inst✝ : DecidableEq δ
    ⊢ Measurable fun p => Function.update p.1 a p.2
  -/
  rw [measurable_pi_iff]
  /-
    δ : Type u_4
    π : δ → Type u_6
    inst✝¹ : (a : δ) → MeasurableSpace (π a)
    a : δ
    inst✝ : DecidableEq δ
    ⊢ ∀ (a_1 : δ), Measurable fun x => Function.update x.1 a x.2 a_1
  -/
  intro j
  /-
    δ : Type u_4
    π : δ → Type u_6
    inst✝¹ : (a : δ) → MeasurableSpace (π a)
    a : δ
    inst✝ : DecidableEq δ
    j : δ
    ⊢ Measurable fun x => Function.update x.1 a x.2 j
  -/
  dsimp [update]
  /-
    δ : Type u_4
    π : δ → Type u_6
    inst✝¹ : (a : δ) → MeasurableSpace (π a)
    a : δ
    inst✝ : DecidableEq δ
    j : δ
    ⊢ Measurable fun x => dite (Eq j a) (fun h => Eq.rec x.2 ⋯) fun h => x.1 j
  -/
  split_ifs with h
    /-
      case pos
      δ : Type u_4
      π : δ → Type u_6
      inst✝¹ : (a : δ) → MeasurableSpace (π a)
      a : δ
      inst✝ : DecidableEq δ
      j : δ
      h : Eq j a
      ⊢ Measurable fun x => Eq.rec x.2 ⋯
    -/
  · subst h
    /-
      case pos
      δ : Type u_4
      π : δ → Type u_6
      inst✝¹ : (a : δ) → MeasurableSpace (π a)
      inst✝ : DecidableEq δ
      j : δ
      ⊢ Measurable fun x => Eq.rec x.2 ⋯
    -/
    dsimp
    /-
      case pos
      δ : Type u_4
      π : δ → Type u_6
      inst✝¹ : (a : δ) → MeasurableSpace (π a)
      inst✝ : DecidableEq δ
      j : δ
      ⊢ Measurable fun x => x.2
    -/
    exact measurable_snd
    /-
      🎉 no goals
    -/
    /-
      case neg
      δ : Type u_4
      π : δ → Type u_6
      inst✝¹ : (a : δ) → MeasurableSpace (π a)
      a : δ
      inst✝ : DecidableEq δ
      j : δ
      h : Not (Eq j a)
      ⊢ Measurable fun x => x.1 j
    -/
  · exact measurable_pi_iff.1 measurable_fst _
    /-
      🎉 no goals
    -/


theorem measurable_uniqueElim [Unique δ] :
    Measurable (uniqueElim : π (default : δ) → ∀ i, π i) := by
  /-
    δ : Type u_4
    π : δ → Type u_6
    inst✝¹ : (a : δ) → MeasurableSpace (π a)
    inst✝ : Unique δ
    ⊢ Measurable uniqueElim
  -/
  simp_rw [measurable_pi_iff, Unique.forall_iff, uniqueElim_default]; exact measurable_id
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


theorem measurable_updateFinset [DecidableEq δ] {s : Finset δ} {x : ∀ i, π i} :
    Measurable (updateFinset x s) := by
  /-
    δ : Type u_4
    π : δ → Type u_6
    inst✝¹ : (a : δ) → MeasurableSpace (π a)
    inst✝ : DecidableEq δ
    s : Finset δ
    x : (i : δ) → π i
    ⊢ Measurable (Function.updateFinset x s)
  -/
  simp (config := { unfoldPartialApp := true }) only [updateFinset, measurable_pi_iff]
  /-
    δ : Type u_4
    π : δ → Type u_6
    inst✝¹ : (a : δ) → MeasurableSpace (π a)
    inst✝ : DecidableEq δ
    s : Finset δ
    x : (i : δ) → π i
    ⊢ ∀ (a : δ), Measurable fun x_1 => dite (Membership.mem s a) (fun hi => x_1 ⟨a …
  -/
  intro i
  /-
    δ : Type u_4
    π : δ → Type u_6
    inst✝¹ : (a : δ) → MeasurableSpace (π a)
    inst✝ : DecidableEq δ
    s : Finset δ
    x : (i : δ) → π i
    i : δ
    ⊢ Measurable fun x_1 => dite (Membership.mem s i) (fun hi => x_1 ⟨i, hi⟩) fun  …
  -/
                         /-
                           🎉 no goals
                         -/
  by_cases h : i ∈ s <;> simp [h, measurable_pi_apply]
                         /-
                           🎉 no goals
                         -/


/-- The function `update f a : π a → Π a, π a` is always measurable.
  This doesn't require `f` to be measurable.
  This should not be confused with the statement that `update f a x` is measurable. -/
@[measurability]
theorem measurable_update (f : ∀ a : δ, π a) {a : δ} [DecidableEq δ] : Measurable (update f a) :=
  measurable_update'.comp measurable_prod_mk_left


theorem measurable_update_left {a : δ} [DecidableEq δ] {x : π a} :
    Measurable (update · a x) :=
  measurable_update'.comp measurable_prod_mk_right


@[measurability, fun_prop]
theorem Set.measurable_restrict (s : Set δ) : Measurable (s.restrict (π := π)) :=
  measurable_pi_lambda _ fun _ ↦ measurable_pi_apply _


@[measurability, fun_prop]
theorem Set.measurable_restrict₂ {s t : Set δ} (hst : s ⊆ t) :
    Measurable (restrict₂ (π := π) hst) :=
  measurable_pi_lambda _ fun _ ↦ measurable_pi_apply _


@[measurability, fun_prop]
theorem Finset.measurable_restrict (s : Finset δ) : Measurable (s.restrict (π := π)) :=
  measurable_pi_lambda _ fun _ ↦ measurable_pi_apply _


@[measurability, fun_prop]
theorem Finset.measurable_restrict₂ {s t : Finset δ} (hst : s ⊆ t) :
    Measurable (Finset.restrict₂ (π := π) hst) :=
  measurable_pi_lambda _ fun _ ↦ measurable_pi_apply _


@[measurability, fun_prop]
theorem Set.measurable_restrict_apply (s : Set α) {f : α → γ} (hf : Measurable f) :
    Measurable (s.restrict f) := hf.comp measurable_subtype_coe


@[measurability, fun_prop]
theorem Set.measurable_restrict₂_apply {s t : Set α} (hst : s ⊆ t)
    {f : t → γ} (hf : Measurable f) :
    Measurable (restrict₂ (π := fun _ ↦ γ) hst f) := hf.comp (measurable_inclusion hst)


@[measurability, fun_prop]
theorem Finset.measurable_restrict_apply (s : Finset α) {f : α → γ} (hf : Measurable f) :
    Measurable (s.restrict f) := hf.comp measurable_subtype_coe


@[measurability, fun_prop]
theorem Finset.measurable_restrict₂_apply {s t : Finset α} (hst : s ⊆ t)
    {f : t → γ} (hf : Measurable f) :
    Measurable (restrict₂ (π := fun _ ↦ γ) hst f) := hf.comp (measurable_inclusion hst)


variable (π) in
theorem measurable_eq_mp {i i' : δ} (h : i = i') : Measurable (congr_arg π h).mp := by
  /-
    δ : Type u_4
    π : δ → Type u_6
    inst✝ : (a : δ) → MeasurableSpace (π a)
    i i' : δ
    h : Eq i i'
    ⊢ Measurable ⋯.mp
  -/
  cases h
  /-
    case refl
    δ : Type u_4
    π : δ → Type u_6
    inst✝ : (a : δ) → MeasurableSpace (π a)
    i : δ
    ⊢ Measurable ⋯.mp
  -/
  exact measurable_id
  /-
    🎉 no goals
  -/


variable (π) in
theorem Measurable.eq_mp {β} [MeasurableSpace β] {i i' : δ} (h : i = i') {f : β → π i}
    (hf : Measurable f) : Measurable fun x => (congr_arg π h).mp (f x) :=
  (measurable_eq_mp π h).comp hf


theorem measurable_piCongrLeft (f : δ' ≃ δ) : Measurable (piCongrLeft π f) := by
  /-
    δ : Type u_4
    δ' : Type u_5
    π : δ → Type u_6
    inst✝ : (a : δ) → MeasurableSpace (π a)
    f : Equiv δ' δ
    ⊢ Measurable ⇑(Equiv.piCongrLeft π f)
  -/
  rw [measurable_pi_iff]
  /-
    δ : Type u_4
    δ' : Type u_5
    π : δ → Type u_6
    inst✝ : (a : δ) → MeasurableSpace (π a)
    f : Equiv δ' δ
    ⊢ ∀ (a : δ), Measurable fun x => (Equiv.piCongrLeft π f) x a
  -/
  intro i
  /-
    δ : Type u_4
    δ' : Type u_5
    π : δ → Type u_6
    inst✝ : (a : δ) → MeasurableSpace (π a)
    f : Equiv δ' δ
    i : δ
    ⊢ Measurable fun x => (Equiv.piCongrLeft π f) x i
  -/
  simp_rw [piCongrLeft_apply_eq_cast]
  /-
    δ : Type u_4
    δ' : Type u_5
    π : δ → Type u_6
    inst✝ : (a : δ) → MeasurableSpace (π a)
    f : Equiv δ' δ
    i : δ
    ⊢ Measurable fun x => cast ⋯ (x (f.symm i))
  -/
  exact Measurable.eq_mp π (f.apply_symm_apply i) <| measurable_pi_apply <| f.symm i
  /-
    🎉 no goals
  -/

/- Even though we cannot use projection notation, we still keep a dot to be consistent with similar
  lemmas, like `MeasurableSet.prod`. -/

@[measurability]
protected theorem MeasurableSet.pi {s : Set δ} {t : ∀ i : δ, Set (π i)} (hs : s.Countable)
    (ht : ∀ i ∈ s, MeasurableSet (t i)) : MeasurableSet (s.pi t) := by
  /-
    δ : Type u_4
    π : δ → Type u_6
    inst✝ : (a : δ) → MeasurableSpace (π a)
    s : Set δ
    t : (i : δ) → Set (π i)
    hs : s.Countable
    ht : ∀ (i : δ), Membership.mem s i → MeasurableSet (t i)
    ⊢ MeasurableSet (s.pi t)
  -/
  rw [pi_def]
  /-
    δ : Type u_4
    π : δ → Type u_6
    inst✝ : (a : δ) → MeasurableSpace (π a)
    s : Set δ
    t : (i : δ) → Set (π i)
    hs : s.Countable
    ht : ∀ (i : δ), Membership.mem s i → MeasurableSet (t i)
    ⊢ MeasurableSet (Set.iInter fun a => Set.iInter fun h => Set.preimage (Functio …
  -/
  exact MeasurableSet.biInter hs fun i hi => measurable_pi_apply _ (ht i hi)
  /-
    🎉 no goals
  -/


protected theorem MeasurableSet.univ_pi [Countable δ] {t : ∀ i : δ, Set (π i)}
    (ht : ∀ i, MeasurableSet (t i)) : MeasurableSet (pi univ t) :=
  MeasurableSet.pi (to_countable _) fun i _ => ht i


theorem measurableSet_pi_of_nonempty {s : Set δ} {t : ∀ i, Set (π i)} (hs : s.Countable)
    (h : (pi s t).Nonempty) : MeasurableSet (pi s t) ↔ ∀ i ∈ s, MeasurableSet (t i) := by
  classical
    rcases h with ⟨f, hf⟩
    refine ⟨fun hst i hi => ?_, MeasurableSet.pi hs⟩
    convert measurable_update f (a := i) hst
    rw [update_preimage_pi hi]
    exact fun j hj _ => hf j hj


theorem measurableSet_pi {s : Set δ} {t : ∀ i, Set (π i)} (hs : s.Countable) :
    MeasurableSet (pi s t) ↔ (∀ i ∈ s, MeasurableSet (t i)) ∨ pi s t = ∅ := by
  /-
    δ : Type u_4
    π : δ → Type u_6
    inst✝ : (a : δ) → MeasurableSpace (π a)
    s : Set δ
    t : (i : δ) → Set (π i)
    hs : s.Countable
    ⊢ Iff (MeasurableSet (s.pi t)) (Or (∀ (i : δ), Membership.mem s i → Measurable …
  -/
  rcases (pi s t).eq_empty_or_nonempty with h | h
    /-
      case inl
      δ : Type u_4
      π : δ → Type u_6
      inst✝ : (a : δ) → MeasurableSpace (π a)
      s : Set δ
      t : (i : δ) → Set (π i)
      hs : s.Countable
      h : Eq (s.pi t) EmptyCollection.emptyCollection
      ⊢ Iff (MeasurableSet (s.pi t)) (Or (∀ (i : δ), Membership.mem s i → Measurable …
    -/
  · simp [h]
    /-
      🎉 no goals
    -/
    /-
      case inr
      δ : Type u_4
      π : δ → Type u_6
      inst✝ : (a : δ) → MeasurableSpace (π a)
      s : Set δ
      t : (i : δ) → Set (π i)
      hs : s.Countable
      h : (s.pi t).Nonempty
      ⊢ Iff (MeasurableSet (s.pi t)) (Or (∀ (i : δ), Membership.mem s i → Measurable …
    -/
  · simp [measurableSet_pi_of_nonempty hs, h, ← not_nonempty_iff_eq_empty]
    /-
      🎉 no goals
    -/


instance Pi.instMeasurableSingletonClass [Countable δ] [∀ a, MeasurableSingletonClass (π a)] :
    MeasurableSingletonClass (∀ a, π a) :=
  ⟨fun f => univ_pi_singleton f ▸ MeasurableSet.univ_pi fun t => measurableSet_singleton (f t)⟩


@[measurability]
theorem measurable_piEquivPiSubtypeProd_symm (p : δ → Prop) [DecidablePred p] :
    Measurable (Equiv.piEquivPiSubtypeProd p π).symm := by
  /-
    δ : Type u_4
    π : δ → Type u_6
    inst✝¹ : (a : δ) → MeasurableSpace (π a)
    p : δ → Prop
    inst✝ : DecidablePred p
    ⊢ Measurable ⇑(Equiv.piEquivPiSubtypeProd p π).symm
  -/
  refine measurable_pi_iff.2 fun j => ?_
  /-
    δ : Type u_4
    π : δ → Type u_6
    inst✝¹ : (a : δ) → MeasurableSpace (π a)
    p : δ → Prop
    inst✝ : DecidablePred p
    j : δ
    ⊢ Measurable fun x => (Equiv.piEquivPiSubtypeProd p π).symm x j
  -/
  by_cases hj : p j
    /-
      case pos
      δ : Type u_4
      π : δ → Type u_6
      inst✝¹ : (a : δ) → MeasurableSpace (π a)
      p : δ → Prop
      inst✝ : DecidablePred p
      j : δ
      hj : p j
      ⊢ Measurable fun x => (Equiv.piEquivPiSubtypeProd p π).symm x j
    -/
  · simp only [hj, dif_pos, Equiv.piEquivPiSubtypeProd_symm_apply]
    have : Measurable fun (f : ∀ i : { x // p x }, π i.1) => f ⟨j, hj⟩ :=
      measurable_pi_apply (π := fun i : {x // p x} => π i.1) ⟨j, hj⟩
    /-
      case pos
      δ : Type u_4
      π : δ → Type u_6
      inst✝¹ : (a : δ) → MeasurableSpace (π a)
      p : δ → Prop
      inst✝ : DecidablePred p
      j : δ
      hj : p j
      this : Measurable fun f => f ⟨j, hj⟩
      ⊢ Measurable fun x => x.1 ⟨j, ⋯⟩
    -/
    exact Measurable.comp this measurable_fst
    /-
      🎉 no goals
    -/
    /-
      case neg
      δ : Type u_4
      π : δ → Type u_6
      inst✝¹ : (a : δ) → MeasurableSpace (π a)
      p : δ → Prop
      inst✝ : DecidablePred p
      j : δ
      hj : Not (p j)
      ⊢ Measurable fun x => (Equiv.piEquivPiSubtypeProd p π).symm x j
    -/
  · simp only [hj, Equiv.piEquivPiSubtypeProd_symm_apply, dif_neg, not_false_iff]
    have : Measurable fun (f : ∀ i : { x // ¬p x }, π i.1) => f ⟨j, hj⟩ :=
      measurable_pi_apply (π := fun i : {x // ¬p x} => π i.1) ⟨j, hj⟩
    /-
      case neg
      δ : Type u_4
      π : δ → Type u_6
      inst✝¹ : (a : δ) → MeasurableSpace (π a)
      p : δ → Prop
      inst✝ : DecidablePred p
      j : δ
      hj : Not (p j)
      this : Measurable fun f => f ⟨j, hj⟩
      ⊢ Measurable fun x => x.2 ⟨j, ⋯⟩
    -/
    exact Measurable.comp this measurable_snd
    /-
      🎉 no goals
    -/


@[measurability]
theorem measurable_piEquivPiSubtypeProd (p : δ → Prop) [DecidablePred p] :
    Measurable (Equiv.piEquivPiSubtypeProd p π) :=
  (measurable_pi_iff.2 fun _ => measurable_pi_apply _).prod_mk
    (measurable_pi_iff.2 fun _ => measurable_pi_apply _)


instance TProd.instMeasurableSpace (π : δ → Type*) [∀ x, MeasurableSpace (π x)] :
    ∀ l : List δ, MeasurableSpace (List.TProd π l)
  | [] => PUnit.instMeasurableSpace
  | _::is => @Prod.instMeasurableSpace _ _ _ (TProd.instMeasurableSpace π is)


theorem measurable_tProd_mk (l : List δ) : Measurable (@TProd.mk δ π l) := by
  /-
    δ : Type u_4
    π : δ → Type u_6
    inst✝ : (x : δ) → MeasurableSpace (π x)
    l : List δ
    ⊢ Measurable (List.TProd.mk l)
  -/
  induction' l with i l ih
    /-
      case nil
      δ : Type u_4
      π : δ → Type u_6
      inst✝ : (x : δ) → MeasurableSpace (π x)
      ⊢ Measurable (List.TProd.mk List.nil)
    -/
  · exact measurable_const
    /-
      🎉 no goals
    -/
    /-
      case cons
      δ : Type u_4
      π : δ → Type u_6
      inst✝ : (x : δ) → MeasurableSpace (π x)
      i : δ
      l : List δ
      ih : Measurable (List.TProd.mk l)
      ⊢ Measurable (List.TProd.mk (List.cons i l))
    -/
  · exact (measurable_pi_apply i).prod_mk ih
    /-
      🎉 no goals
    -/


theorem measurable_tProd_elim [DecidableEq δ] :
    ∀ {l : List δ} {i : δ} (hi : i ∈ l), Measurable fun v : TProd π l => v.elim hi
  | i::is, j, hj => by
    /-
      δ : Type u_4
      π : δ → Type u_6
      inst✝¹ : (x : δ) → MeasurableSpace (π x)
      inst✝ : DecidableEq δ
      i : δ
      is : List δ
      j : δ
      hj : Membership.mem (List.cons i is) j
      ⊢ Measurable fun v => v.elim hj
    -/
    by_cases hji : j = i
      /-
        case pos
        δ : Type u_4
        π : δ → Type u_6
        inst✝¹ : (x : δ) → MeasurableSpace (π x)
        inst✝ : DecidableEq δ
        i : δ
        is : List δ
        j : δ
        hj : Membership.mem (List.cons i is) j
        hji : Eq j i
        ⊢ Measurable fun v => v.elim hj
      -/
    · subst hji
      /-
        case pos
        δ : Type u_4
        π : δ → Type u_6
        inst✝¹ : (x : δ) → MeasurableSpace (π x)
        inst✝ : DecidableEq δ
        is : List δ
        j : δ
        hj : Membership.mem (List.cons j is) j
        ⊢ Measurable fun v => v.elim hj
      -/
      simpa using measurable_fst
      /-
        🎉 no goals
      -/
      /-
        case neg
        δ : Type u_4
        π : δ → Type u_6
        inst✝¹ : (x : δ) → MeasurableSpace (π x)
        inst✝ : DecidableEq δ
        i : δ
        is : List δ
        j : δ
        hj : Membership.mem (List.cons i is) j
        hji : Not (Eq j i)
        ⊢ Measurable fun v => v.elim hj
      -/
    · simp only [TProd.elim_of_ne _ hji]
      /-
        case neg
        δ : Type u_4
        π : δ → Type u_6
        inst✝¹ : (x : δ) → MeasurableSpace (π x)
        inst✝ : DecidableEq δ
        i : δ
        is : List δ
        j : δ
        hj : Membership.mem (List.cons i is) j
        hji : Not (Eq j i)
        ⊢ Measurable fun v => List.TProd.elim v.2 ⋯
      -/
      rw [mem_cons] at hj
      /-
        case neg
        δ : Type u_4
        π : δ → Type u_6
        inst✝¹ : (x : δ) → MeasurableSpace (π x)
        inst✝ : DecidableEq δ
        i : δ
        is : List δ
        j : δ
        hj✝ : Membership.mem (List.cons i is) j
        hj : Or (Eq j i) (Membership.mem is j)
        hji : Not (Eq j i)
        ⊢ Measurable fun v => List.TProd.elim v.2 ⋯
      -/
      exact (measurable_tProd_elim (hj.resolve_left hji)).comp measurable_snd
      /-
        🎉 no goals
      -/


theorem measurable_tProd_elim' [DecidableEq δ] {l : List δ} (h : ∀ i, i ∈ l) :
    Measurable (TProd.elim' h : TProd π l → ∀ i, π i) :=
  measurable_pi_lambda _ fun i => measurable_tProd_elim (h i)


theorem MeasurableSet.tProd (l : List δ) {s : ∀ i, Set (π i)} (hs : ∀ i, MeasurableSet (s i)) :
    MeasurableSet (Set.tprod l s) := by
  /-
    δ : Type u_4
    π : δ → Type u_6
    inst✝ : (x : δ) → MeasurableSpace (π x)
    l : List δ
    s : (i : δ) → Set (π i)
    hs : ∀ (i : δ), MeasurableSet (s i)
    ⊢ MeasurableSet (Set.tprod l s)
  -/
  induction' l with i l ih
    /-
      case nil
      δ : Type u_4
      π : δ → Type u_6
      inst✝ : (x : δ) → MeasurableSpace (π x)
      s : (i : δ) → Set (π i)
      hs : ∀ (i : δ), MeasurableSet (s i)
      ⊢ MeasurableSet (Set.tprod List.nil s)
    -/
  · exact MeasurableSet.univ
    /-
      🎉 no goals
    -/
    /-
      case cons
      δ : Type u_4
      π : δ → Type u_6
      inst✝ : (x : δ) → MeasurableSpace (π x)
      s : (i : δ) → Set (π i)
      hs : ∀ (i : δ), MeasurableSet (s i)
      i : δ
      l : List δ
      ih : MeasurableSet (Set.tprod l s)
      ⊢ MeasurableSet (Set.tprod (List.cons i l) s)
    -/
  · exact (hs i).prod ih
    /-
      🎉 no goals
    -/


instance Sum.instMeasurableSpace {α β} [m₁ : MeasurableSpace α] [m₂ : MeasurableSpace β] :
    MeasurableSpace (α ⊕ β) :=
  m₁.map Sum.inl ⊓ m₂.map Sum.inr


@[measurability]
theorem measurable_inl [MeasurableSpace α] [MeasurableSpace β] : Measurable (@Sum.inl α β) :=
  Measurable.of_le_map inf_le_left


@[measurability]
theorem measurable_inr [MeasurableSpace α] [MeasurableSpace β] : Measurable (@Sum.inr α β) :=
  Measurable.of_le_map inf_le_right


theorem measurableSet_sum_iff {s : Set (α ⊕ β)} :
    MeasurableSet s ↔ MeasurableSet (Sum.inl ⁻¹' s) ∧ MeasurableSet (Sum.inr ⁻¹' s) :=
  Iff.rfl


theorem measurable_sum {_ : MeasurableSpace γ} {f : α ⊕ β → γ} (hl : Measurable (f ∘ Sum.inl))
    (hr : Measurable (f ∘ Sum.inr)) : Measurable f :=
  Measurable.of_comap_le <|
    le_inf (MeasurableSpace.comap_le_iff_le_map.2 <| hl)
      (MeasurableSpace.comap_le_iff_le_map.2 <| hr)


@[measurability]
theorem Measurable.sumElim {_ : MeasurableSpace γ} {f : α → γ} {g : β → γ} (hf : Measurable f)
    (hg : Measurable g) : Measurable (Sum.elim f g) :=
  measurable_sum hf hg


theorem Measurable.sumMap {_ : MeasurableSpace γ} {_ : MeasurableSpace δ} {f : α → β} {g : γ → δ}
    (hf : Measurable f) (hg : Measurable g) : Measurable (Sum.map f g) :=
  (measurable_inl.comp hf).sumElim (measurable_inr.comp hg)


@[simp] theorem measurableSet_inl_image {s : Set α} :
    MeasurableSet (Sum.inl '' s : Set (α ⊕ β)) ↔ MeasurableSet s := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    mβ : MeasurableSpace β
    s : Set α
    ⊢ Iff (MeasurableSet (Set.image Sum.inl s)) (MeasurableSet s)
  -/
  simp [measurableSet_sum_iff, Sum.inl_injective.preimage_image]
  /-
    🎉 no goals
  -/


alias ⟨_, MeasurableSet.inl_image⟩ := measurableSet_inl_image


@[simp] theorem measurableSet_inr_image {s : Set β} :
    MeasurableSet (Sum.inr '' s : Set (α ⊕ β)) ↔ MeasurableSet s := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    mβ : MeasurableSpace β
    s : Set β
    ⊢ Iff (MeasurableSet (Set.image Sum.inr s)) (MeasurableSet s)
  -/
  simp [measurableSet_sum_iff, Sum.inr_injective.preimage_image]
  /-
    🎉 no goals
  -/


alias ⟨_, MeasurableSet.inr_image⟩ := measurableSet_inr_image


theorem measurableSet_range_inl [MeasurableSpace α] :
    MeasurableSet (range Sum.inl : Set (α ⊕ β)) := by
  /-
    α : Type u_1
    β : Type u_2
    mβ : MeasurableSpace β
    inst✝ : MeasurableSpace α
    ⊢ MeasurableSet (Set.range Sum.inl)
  -/
  rw [← image_univ]
  /-
    α : Type u_1
    β : Type u_2
    mβ : MeasurableSpace β
    inst✝ : MeasurableSpace α
    ⊢ MeasurableSet (Set.image Sum.inl Set.univ)
  -/
  exact MeasurableSet.univ.inl_image
  /-
    🎉 no goals
  -/


theorem measurableSet_range_inr [MeasurableSpace α] :
    MeasurableSet (range Sum.inr : Set (α ⊕ β)) := by
  /-
    α : Type u_1
    β : Type u_2
    mβ : MeasurableSpace β
    inst✝ : MeasurableSpace α
    ⊢ MeasurableSet (Set.range Sum.inr)
  -/
  rw [← image_univ]
  /-
    α : Type u_1
    β : Type u_2
    mβ : MeasurableSpace β
    inst✝ : MeasurableSpace α
    ⊢ MeasurableSet (Set.image Sum.inr Set.univ)
  -/
  exact MeasurableSet.univ.inr_image
  /-
    🎉 no goals
  -/


instance Sigma.instMeasurableSpace {α} {β : α → Type*} [m : ∀ a, MeasurableSpace (β a)] :
    MeasurableSpace (Sigma β) :=
  ⨅ a, (m a).map (Sigma.mk a)


@[simp] theorem measurableSet_setOf : MeasurableSet {a | p a} ↔ Measurable p :=
                                    /-
                                      α : Type u_1
                                      inst✝ : MeasurableSpace α
                                      p : α → Prop
                                      h : MeasurableSet (setOf fun a => p a)
                                      ⊢ MeasurableSet (Set.preimage p (Singleton.singleton True))
                                    -/
  ⟨fun h ↦ measurable_to_prop <| by simpa only [preimage_singleton_true], fun h => by
                                    /-
                                      🎉 no goals
                                    -/
    /-
      α : Type u_1
      inst✝ : MeasurableSpace α
      p : α → Prop
      h : Measurable p
      ⊢ MeasurableSet (setOf fun a => p a)
    -/
    simpa using h (measurableSet_singleton True)⟩
    /-
      🎉 no goals
    -/


@[simp] theorem measurable_mem : Measurable (· ∈ s) ↔ MeasurableSet s := measurableSet_setOf.symm


alias ⟨_, Measurable.setOf⟩ := measurableSet_setOf


alias ⟨_, MeasurableSet.mem⟩ := measurable_mem


lemma Measurable.not (hp : Measurable p) : Measurable (¬ p ·) :=
  measurableSet_setOf.1 hp.setOf.compl


lemma Measurable.and (hp : Measurable p) (hq : Measurable q) : Measurable fun a ↦ p a ∧ q a :=
  measurableSet_setOf.1 <| hp.setOf.inter hq.setOf


lemma Measurable.or (hp : Measurable p) (hq : Measurable q) : Measurable fun a ↦ p a ∨ q a :=
  measurableSet_setOf.1 <| hp.setOf.union hq.setOf


lemma Measurable.imp (hp : Measurable p) (hq : Measurable q) : Measurable fun a ↦ p a → q a :=
  measurableSet_setOf.1 <| hp.setOf.himp hq.setOf


lemma Measurable.iff (hp : Measurable p) (hq : Measurable q) : Measurable fun a ↦ p a ↔ q a :=
                              /-
                                α : Type u_1
                                inst✝ : MeasurableSpace α
                                p q : α → Prop
                                hp : Measurable p
                                hq : Measurable q
                                ⊢ MeasurableSet (_root_.setOf fun a => Iff (p a) (q a))
                              -/
  measurableSet_setOf.1 <| by simp_rw [iff_iff_implies_and_implies]; exact hq.setOf.bihimp hp.setOf
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


lemma Measurable.forall [Countable ι] {p : ι → α → Prop} (hp : ∀ i, Measurable (p i)) :
    Measurable fun a ↦ ∀ i, p i a :=
                              /-
                                α : Type u_1
                                ι : Sort uι
                                inst✝¹ : MeasurableSpace α
                                inst✝ : Countable ι
                                p : ι → α → Prop
                                hp : ∀ (i : ι), Measurable (p i)
                                ⊢ MeasurableSet (_root_.setOf fun a => ∀ (i : ι), p i a)
                              -/
  measurableSet_setOf.1 <| by rw [setOf_forall]; exact MeasurableSet.iInter fun i ↦ (hp i).setOf
                                                 /-
                                                   🎉 no goals
                                                 -/


lemma Measurable.exists [Countable ι] {p : ι → α → Prop} (hp : ∀ i, Measurable (p i)) :
    Measurable fun a ↦ ∃ i, p i a :=
                              /-
                                α : Type u_1
                                ι : Sort uι
                                inst✝¹ : MeasurableSpace α
                                inst✝ : Countable ι
                                p : ι → α → Prop
                                hp : ∀ (i : ι), Measurable (p i)
                                ⊢ MeasurableSet (_root_.setOf fun a => Exists fun i => p i a)
                              -/
  measurableSet_setOf.1 <| by rw [setOf_exists]; exact MeasurableSet.iUnion fun i ↦ (hp i).setOf
                                                 /-
                                                   🎉 no goals
                                                 -/


/-- This instance is useful when talking about Bernoulli sequences of random variables or binomial
random graphs. -/
                                                                 /-
                                                                   α : Type u_1
                                                                   β : Type u_2
                                                                   γ : Type u_3
                                                                   δ : Type u_4
                                                                   δ' : Type u_5
                                                                   ι : Sort uι
                                                                   s : Set α
                                                                   inst✝ : MeasurableSpace β
                                                                   g : β → Set α
                                                                   ⊢ MeasurableSpace (Set α)
                                                                 -/
instance Set.instMeasurableSpace : MeasurableSpace (Set α) := by unfold Set; infer_instance
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


instance Set.instMeasurableSingletonClass [Countable α] : MeasurableSingletonClass (Set α) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    δ' : Type u_5
    ι : Sort uι
    s : Set α
    inst✝¹ : MeasurableSpace β
    g : β → Set α
    inst✝ : Countable α
    ⊢ MeasurableSingletonClass (Set α)
  -/
  unfold Set; infer_instance
              /-
                🎉 no goals
              -/


lemma measurable_set_iff : Measurable g ↔ ∀ a, Measurable fun x ↦ a ∈ g x := measurable_pi_iff


@[aesop safe 100 apply (rule_sets := [Measurable])]
lemma measurable_set_mem (a : α) : Measurable fun s : Set α ↦ a ∈ s := measurable_pi_apply _


@[aesop safe 100 apply (rule_sets := [Measurable])]
lemma measurable_set_not_mem (a : α) : Measurable fun s : Set α ↦ a ∉ s :=
  (Measurable.of_discrete (f := Not)).comp <| measurable_set_mem a


@[aesop safe 100 apply (rule_sets := [Measurable])]
lemma measurableSet_mem (a : α) : MeasurableSet {s : Set α | a ∈ s} :=
  measurableSet_setOf.2 <| measurable_set_mem _


@[aesop safe 100 apply (rule_sets := [Measurable])]
lemma measurableSet_not_mem (a : α) : MeasurableSet {s : Set α | a ∉ s} :=
  measurableSet_setOf.2 <| measurable_set_not_mem _


lemma measurable_compl : Measurable ((·ᶜ) : Set α → Set α) :=
  measurable_set_iff.2 fun _ ↦ measurable_set_not_mem _


lemma MeasurableSet.setOf_finite [Countable α] : MeasurableSet {s : Set α | s.Finite} :=
  Countable.setOf_finite.measurableSet


lemma MeasurableSet.setOf_infinite [Countable α] : MeasurableSet {s : Set α | s.Infinite} :=
  .setOf_finite |> .compl


lemma MeasurableSet.sep_finite [Countable α] {S : Set (Set α)} (hS : MeasurableSet S) :
    MeasurableSet {s ∈ S | s.Finite} :=
  hS.inter .setOf_finite


lemma MeasurableSet.sep_infinite [Countable α] {S : Set (Set α)} (hS : MeasurableSet S) :
    MeasurableSet {s ∈ S | s.Infinite} :=
  hS.inter .setOf_infinite


/-- The sigma-algebra generated by a single set `s` is `{∅, s, sᶜ, univ}`. -/
@[simp] theorem generateFrom_singleton (s : Set α) :
    generateFrom {s} = MeasurableSpace.comap (· ∈ s) ⊤ := by
  classical
  letI : MeasurableSpace α := generateFrom {s}
  refine le_antisymm (generateFrom_le fun t ht => ⟨{True}, trivial, by simp [ht.symm]⟩) ?_
  rintro _ ⟨u, -, rfl⟩
  exact (show MeasurableSet s from GenerateMeasurable.basic _ <| mem_singleton s).mem trivial


lemma generateFrom_singleton_le {m : MeasurableSpace α} {s : Set α} (hs : MeasurableSet s) :
    MeasurableSpace.generateFrom {s} ≤ m :=
  generateFrom_le (fun _ ht ↦ mem_singleton_iff.1 ht ▸ hs)


theorem measurableSet_generateFrom_singleton_iff {s t : Set α} :
    MeasurableSet[MeasurableSpace.generateFrom {s}] t ↔ t = ∅ ∨ t = s ∨ t = sᶜ ∨ t = univ := by
  /-
    α : Type u_1
    s t : Set α
    ⊢ Iff (MeasurableSet t) (Or (Eq t EmptyCollection.emptyCollection) (Or (Eq t s …
  -/
  simp_rw [MeasurableSpace.generateFrom_singleton]
  /-
    α : Type u_1
    s t : Set α
    ⊢ Iff (MeasurableSet t) (Or (Eq t EmptyCollection.emptyCollection) (Or (Eq t s …
  -/
  change t ∈ {t | _} ↔ _
  /-
    α : Type u_1
    s t : Set α
    ⊢ Iff (Membership.mem (setOf fun t_1 => Exists fun s' => And (MeasurableSet s' …
  -/
  simp_rw [MeasurableSpace.measurableSet_top, true_and, mem_setOf_eq]
  /-
    α : Type u_1
    s t : Set α
    ⊢ Iff (Exists fun s' => Eq (Set.preimage (fun x => Membership.mem s x) s') t)  …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      s t : Set α
      ⊢ (Exists fun s' => Eq (Set.preimage (fun x => Membership.mem s x) s') t) → Or …
    -/
  · rintro ⟨x, rfl⟩
    /-
      case mp.intro
      α : Type u_1
      s : Set α
      x : Set Prop
      ⊢ Or (Eq (Set.preimage (fun x => Membership.mem s x) x) EmptyCollection.emptyC …
    -/
    by_cases hT : True ∈ x
      /-
        case pos
        α : Type u_1
        s : Set α
        x : Set Prop
        hT : Membership.mem x True
        ⊢ Or (Eq (Set.preimage (fun x => Membership.mem s x) x) EmptyCollection.emptyC …
      -/
    · by_cases hF : False ∈ x
        /-
          case pos
          α : Type u_1
          s : Set α
          x : Set Prop
          hT : Membership.mem x True
          hF : Membership.mem x False
          ⊢ Or (Eq (Set.preimage (fun x => Membership.mem s x) x) EmptyCollection.emptyC …
        -/
      · refine Or.inr <| Or.inr <| Or.inr <| subset_antisymm (subset_univ _) ?_
        /-
          case pos
          α : Type u_1
          s : Set α
          x : Set Prop
          hT : Membership.mem x True
          hF : Membership.mem x False
          ⊢ HasSubset.Subset Set.univ (Set.preimage (fun x => Membership.mem s x) x)
        -/
        suffices x = univ by simp only [this, preimage_univ, subset_refl]
        /-
          case pos
          α : Type u_1
          s : Set α
          x : Set Prop
          hT : Membership.mem x True
          hF : Membership.mem x False
          ⊢ Eq x Set.univ
        -/
        refine subset_antisymm (subset_univ _) ?_
        /-
          case pos
          α : Type u_1
          s : Set α
          x : Set Prop
          hT : Membership.mem x True
          hF : Membership.mem x False
          ⊢ HasSubset.Subset Set.univ x
        -/
        rw [univ_eq_true_false]
        /-
          case pos
          α : Type u_1
          s : Set α
          x : Set Prop
          hT : Membership.mem x True
          hF : Membership.mem x False
          ⊢ HasSubset.Subset (Insert.insert True (Singleton.singleton False)) x
        -/
        rintro - (rfl | rfl)
          /-
            case pos.inl
            α : Type u_1
            s : Set α
            x : Set Prop
            hT : Membership.mem x True
            hF : Membership.mem x False
            ⊢ Membership.mem x True
          -/
        · assumption
          /-
            🎉 no goals
          -/
          /-
            case pos.inr
            α : Type u_1
            s : Set α
            x : Set Prop
            hT : Membership.mem x True
            hF : Membership.mem x False
            ⊢ Membership.mem x False
          -/
        · assumption
          /-
            🎉 no goals
          -/
      · have hx : x = {True} := by
          ext p
          refine ⟨fun hp ↦ mem_singleton_iff.2 ?_, fun hp ↦ hp ▸ hT⟩
          by_contra hpneg
          rw [eq_iff_iff, iff_true, ← false_iff] at hpneg
          exact hF (by convert hp)
        /-
          case neg
          α : Type u_1
          s : Set α
          x : Set Prop
          hT : Membership.mem x True
          hF : Not (Membership.mem x False)
          hx : Eq x (Singleton.singleton True)
          ⊢ Or (Eq (Set.preimage (fun x => Membership.mem s x) x) EmptyCollection.emptyC …
        -/
        simp [hx]
        /-
          🎉 no goals
        -/
      /-
        case neg
        α : Type u_1
        s : Set α
        x : Set Prop
        hT : Not (Membership.mem x True)
        ⊢ Or (Eq (Set.preimage (fun x => Membership.mem s x) x) EmptyCollection.emptyC …
      -/
    · by_cases hF : False ∈ x
      · have hx : x = {False} := by
          ext p
          refine ⟨fun hp ↦ mem_singleton_iff.2 ?_, fun hp ↦ hp ▸ hF⟩
          by_contra hpneg
          simp only [eq_iff_iff, iff_false, not_not] at hpneg
          refine hT ?_
          convert hp
          simpa
        /-
          case pos
          α : Type u_1
          s : Set α
          x : Set Prop
          hT : Not (Membership.mem x True)
          hF : Membership.mem x False
          hx : Eq x (Singleton.singleton False)
          ⊢ Or (Eq (Set.preimage (fun x => Membership.mem s x) x) EmptyCollection.emptyC …
        -/
        refine Or.inr <| Or.inr <| Or.inl <| ?_
        /-
          case pos
          α : Type u_1
          s : Set α
          x : Set Prop
          hT : Not (Membership.mem x True)
          hF : Membership.mem x False
          hx : Eq x (Singleton.singleton False)
          ⊢ Eq (Set.preimage (fun x => Membership.mem s x) x) (HasCompl.compl s)
        -/
        simp [hx]
        /-
          case pos
          α : Type u_1
          s : Set α
          x : Set Prop
          hT : Not (Membership.mem x True)
          hF : Membership.mem x False
          hx : Eq x (Singleton.singleton False)
          ⊢ Eq (setOf fun a => Not (Membership.mem s a)) (HasCompl.compl s)
        -/
        rfl
        /-
          🎉 no goals
        -/
        /-
          case neg
          α : Type u_1
          s : Set α
          x : Set Prop
          hT : Not (Membership.mem x True)
          hF : Not (Membership.mem x False)
          ⊢ Or (Eq (Set.preimage (fun x => Membership.mem s x) x) EmptyCollection.emptyC …
        -/
      · refine Or.inl <| subset_antisymm ?_ <| empty_subset _
        suffices x ⊆ ∅ by
          rw [subset_empty_iff] at this
          simp only [this, preimage_empty, subset_refl]
        /-
          case neg
          α : Type u_1
          s : Set α
          x : Set Prop
          hT : Not (Membership.mem x True)
          hF : Not (Membership.mem x False)
          ⊢ HasSubset.Subset x EmptyCollection.emptyCollection
        -/
        intro p hp
        /-
          case neg
          α : Type u_1
          s : Set α
          x : Set Prop
          hT : Not (Membership.mem x True)
          hF : Not (Membership.mem x False)
          p : Prop
          hp : Membership.mem x p
          ⊢ Membership.mem EmptyCollection.emptyCollection p
        -/
        fin_cases p
          /-
            case neg.«_@»._hyg.14998.«0»
            α : Type u_1
            s : Set α
            x : Set Prop
            hT : Not (Membership.mem x True)
            hF : Not (Membership.mem x False)
            hp : Membership.mem x True
            ⊢ Membership.mem EmptyCollection.emptyCollection True
          -/
        · contradiction
          /-
            🎉 no goals
          -/
          /-
            case neg.«_@»._hyg.14998.«1»
            α : Type u_1
            s : Set α
            x : Set Prop
            hT : Not (Membership.mem x True)
            hF : Not (Membership.mem x False)
            hp : Membership.mem x False
            ⊢ Membership.mem EmptyCollection.emptyCollection False
          -/
        · contradiction
          /-
            🎉 no goals
          -/
    /-
      case mpr
      α : Type u_1
      s t : Set α
      ⊢ Or (Eq t EmptyCollection.emptyCollection) (Or (Eq t s) (Or (Eq t (HasCompl.c …
    -/
  · rintro (rfl | rfl | rfl | rfl)
    /-
      case mpr.inl
      α : Type u_1
      s : Set α
      ⊢ Exists fun s' => Eq (Set.preimage (fun x => Membership.mem s x) s') EmptyCol …
    -/
    on_goal 1 => use ∅
    /-
      case h
      α : Type u_1
      s : Set α
      ⊢ Eq (Set.preimage (fun x => Membership.mem s x) EmptyCollection.emptyCollecti …
    -/
    on_goal 2 => use {True}
    /-
      case h
      α : Type u_1
      s : Set α
      ⊢ Eq (Set.preimage (fun x => Membership.mem s x) EmptyCollection.emptyCollecti …
    -/
    on_goal 3 => use {False}
    /-
      case h
      α : Type u_1
      s : Set α
      ⊢ Eq (Set.preimage (fun x => Membership.mem s x) EmptyCollection.emptyCollecti …
    -/
    on_goal 4 => use Set.univ
    all_goals
      simp [compl_def]


/-- A filter `f` is measurably generates if each `s ∈ f` includes a measurable `t ∈ f`. -/
class IsMeasurablyGenerated (f : Filter α) : Prop where
  exists_measurable_subset : ∀ ⦃s⦄, s ∈ f → ∃ t ∈ f, MeasurableSet t ∧ t ⊆ s


instance isMeasurablyGenerated_bot : IsMeasurablyGenerated (⊥ : Filter α) :=
  ⟨fun _ _ => ⟨∅, mem_bot, MeasurableSet.empty, empty_subset _⟩⟩


instance isMeasurablyGenerated_top : IsMeasurablyGenerated (⊤ : Filter α) :=
  ⟨fun _s hs => ⟨univ, univ_mem, MeasurableSet.univ, fun x _ => hs x⟩⟩


theorem Eventually.exists_measurable_mem {f : Filter α} [IsMeasurablyGenerated f] {p : α → Prop}
    (h : ∀ᶠ x in f, p x) : ∃ s ∈ f, MeasurableSet s ∧ ∀ x ∈ s, p x :=
  IsMeasurablyGenerated.exists_measurable_subset h


theorem Eventually.exists_measurable_mem_of_smallSets {f : Filter α} [IsMeasurablyGenerated f]
    {p : Set α → Prop} (h : ∀ᶠ s in f.smallSets, p s) : ∃ s ∈ f, MeasurableSet s ∧ p s :=
  let ⟨_s, hsf, hs⟩ := eventually_smallSets.1 h
  let ⟨t, htf, htm, hts⟩ := IsMeasurablyGenerated.exists_measurable_subset hsf
  ⟨t, htf, htm, hs t hts⟩


instance inf_isMeasurablyGenerated (f g : Filter α) [IsMeasurablyGenerated f]
    [IsMeasurablyGenerated g] : IsMeasurablyGenerated (f ⊓ g) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    δ' : Type u_5
    ι : Sort uι
    s : Set α
    inst✝² : MeasurableSpace α
    f g : Filter α
    inst✝¹ : f.IsMeasurablyGenerated
    inst✝ : g.IsMeasurablyGenerated
    ⊢ (Min.min f g).IsMeasurablyGenerated
  -/
  constructor
  /-
    case exists_measurable_subset
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    δ' : Type u_5
    ι : Sort uι
    s : Set α
    inst✝² : MeasurableSpace α
    f g : Filter α
    inst✝¹ : f.IsMeasurablyGenerated
    inst✝ : g.IsMeasurablyGenerated
    ⊢ ∀ ⦃s : Set α⦄, Membership.mem (Min.min f g) s → Exists fun t => And (Members …
  -/
  rintro t ⟨sf, hsf, sg, hsg, rfl⟩
  /-
    case exists_measurable_subset.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    δ' : Type u_5
    ι : Sort uι
    s : Set α
    inst✝² : MeasurableSpace α
    f g : Filter α
    inst✝¹ : f.IsMeasurablyGenerated
    inst✝ : g.IsMeasurablyGenerated
    sf : Set α
    hsf : Membership.mem f sf
    sg : Set α
    hsg : Membership.mem g sg
    ⊢ Exists fun t => And (Membership.mem (Min.min f g) t) (And (MeasurableSet t)  …
  -/
  rcases IsMeasurablyGenerated.exists_measurable_subset hsf with ⟨s'f, hs'f, hmf, hs'sf⟩
  /-
    case exists_measurable_subset.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    δ' : Type u_5
    ι : Sort uι
    s : Set α
    inst✝² : MeasurableSpace α
    f g : Filter α
    inst✝¹ : f.IsMeasurablyGenerated
    inst✝ : g.IsMeasurablyGenerated
    sf : Set α
    hsf : Membership.mem f sf
    sg : Set α
    hsg : Membership.mem g sg
    s'f : Set α
    hs'f : Membership.mem f s'f
    hmf : MeasurableSet s'f
    hs'sf : HasSubset.Subset s'f sf
    ⊢ Exists fun t => And (Membership.mem (Min.min f g) t) (And (MeasurableSet t)  …
  -/
  rcases IsMeasurablyGenerated.exists_measurable_subset hsg with ⟨s'g, hs'g, hmg, hs'sg⟩
  /-
    case exists_measurable_subset.intro.intro.intro.intro.intro.intro.intro.intro. …
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    δ' : Type u_5
    ι : Sort uι
    s : Set α
    inst✝² : MeasurableSpace α
    f g : Filter α
    inst✝¹ : f.IsMeasurablyGenerated
    inst✝ : g.IsMeasurablyGenerated
    sf : Set α
    hsf : Membership.mem f sf
    sg : Set α
    hsg : Membership.mem g sg
    s'f : Set α
    hs'f : Membership.mem f s'f
    hmf : MeasurableSet s'f
    hs'sf : HasSubset.Subset s'f sf
    s'g : Set α
    hs'g : Membership.mem g s'g
    hmg : MeasurableSet s'g
    hs'sg : HasSubset.Subset s'g sg
    ⊢ Exists fun t => And (Membership.mem (Min.min f g) t) (And (MeasurableSet t)  …
  -/
  refine ⟨s'f ∩ s'g, inter_mem_inf hs'f hs'g, hmf.inter hmg, ?_⟩
  /-
    case exists_measurable_subset.intro.intro.intro.intro.intro.intro.intro.intro. …
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    δ' : Type u_5
    ι : Sort uι
    s : Set α
    inst✝² : MeasurableSpace α
    f g : Filter α
    inst✝¹ : f.IsMeasurablyGenerated
    inst✝ : g.IsMeasurablyGenerated
    sf : Set α
    hsf : Membership.mem f sf
    sg : Set α
    hsg : Membership.mem g sg
    s'f : Set α
    hs'f : Membership.mem f s'f
    hmf : MeasurableSet s'f
    hs'sf : HasSubset.Subset s'f sf
    s'g : Set α
    hs'g : Membership.mem g s'g
    hmg : MeasurableSet s'g
    hs'sg : HasSubset.Subset s'g sg
    ⊢ HasSubset.Subset (Inter.inter s'f s'g) (Inter.inter sf sg)
  -/
  exact inter_subset_inter hs'sf hs'sg
  /-
    🎉 no goals
  -/


theorem principal_isMeasurablyGenerated_iff {s : Set α} :
    IsMeasurablyGenerated (𝓟 s) ↔ MeasurableSet s := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : Set α
    ⊢ Iff (Filter.principal s).IsMeasurablyGenerated (MeasurableSet s)
  -/
  refine ⟨?_, fun hs => ⟨fun t ht => ⟨s, mem_principal_self s, hs, ht⟩⟩⟩
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : Set α
    ⊢ (Filter.principal s).IsMeasurablyGenerated → MeasurableSet s
  -/
  rintro ⟨hs⟩
  /-
    case mk
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : Set α
    hs : ∀ ⦃s_1 : Set α⦄, Membership.mem (Filter.principal s) s_1 → Exists fun t = …
    ⊢ MeasurableSet s
  -/
  rcases hs (mem_principal_self s) with ⟨t, ht, htm, hts⟩
  /-
    case mk.intro.intro.intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : Set α
    hs : ∀ ⦃s_1 : Set α⦄, Membership.mem (Filter.principal s) s_1 → Exists fun t = …
    t : Set α
    ht : Membership.mem (Filter.principal s) t
    htm : MeasurableSet t
    hts : HasSubset.Subset t s
    ⊢ MeasurableSet s
  -/
  have : t = s := hts.antisymm ht
  /-
    case mk.intro.intro.intro
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : Set α
    hs : ∀ ⦃s_1 : Set α⦄, Membership.mem (Filter.principal s) s_1 → Exists fun t = …
    t : Set α
    ht : Membership.mem (Filter.principal s) t
    htm : MeasurableSet t
    hts : HasSubset.Subset t s
    this : Eq t s
    ⊢ MeasurableSet s
  -/
  rwa [← this]
  /-
    🎉 no goals
  -/


alias ⟨_, _root_.MeasurableSet.principal_isMeasurablyGenerated⟩ :=
  principal_isMeasurablyGenerated_iff


instance iInf_isMeasurablyGenerated {f : ι → Filter α} [∀ i, IsMeasurablyGenerated (f i)] :
    IsMeasurablyGenerated (⨅ i, f i) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    δ' : Type u_5
    ι : Sort uι
    s : Set α
    inst✝¹ : MeasurableSpace α
    f : ι → Filter α
    inst✝ : ∀ (i : ι), (f i).IsMeasurablyGenerated
    ⊢ (iInf fun i => f i).IsMeasurablyGenerated
  -/
  refine ⟨fun s hs => ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    δ' : Type u_5
    ι : Sort uι
    s✝ : Set α
    inst✝¹ : MeasurableSpace α
    f : ι → Filter α
    inst✝ : ∀ (i : ι), (f i).IsMeasurablyGenerated
    s : Set α
    hs : Membership.mem (iInf fun i => f i) s
    ⊢ Exists fun t => And (Membership.mem (iInf fun i => f i) t) (And (MeasurableS …
  -/
  rw [← Equiv.plift.surjective.iInf_comp, mem_iInf] at hs
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    δ' : Type u_5
    ι : Sort uι
    s✝ : Set α
    inst✝¹ : MeasurableSpace α
    f : ι → Filter α
    inst✝ : ∀ (i : ι), (f i).IsMeasurablyGenerated
    s : Set α
    hs : Exists fun I => And I.Finite (Exists fun V => And (∀ (i : ↑I), Membership …
    ⊢ Exists fun t => And (Membership.mem (iInf fun i => f i) t) (And (MeasurableS …
  -/
  rcases hs with ⟨t, ht, ⟨V, hVf, rfl⟩⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    δ' : Type u_5
    ι : Sort uι
    s : Set α
    inst✝¹ : MeasurableSpace α
    f : ι → Filter α
    inst✝ : ∀ (i : ι), (f i).IsMeasurablyGenerated
    t : Set (PLift ι)
    ht : t.Finite
    V : ↑t → Set α
    hVf : ∀ (i : ↑t), Membership.mem (f (Equiv.plift ↑i)) (V i)
    ⊢ Exists fun t_1 => And (Membership.mem (iInf fun i => f i) t_1) (And (Measura …
  -/
  choose U hUf hU using fun i => IsMeasurablyGenerated.exists_measurable_subset (hVf i)
  /-
    case intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    δ' : Type u_5
    ι : Sort uι
    s : Set α
    inst✝¹ : MeasurableSpace α
    f : ι → Filter α
    inst✝ : ∀ (i : ι), (f i).IsMeasurablyGenerated
    t : Set (PLift ι)
    ht : t.Finite
    V : ↑t → Set α
    hVf : ∀ (i : ↑t), Membership.mem (f (Equiv.plift ↑i)) (V i)
    U : ↑t → Set α
    hUf : ∀ (i : ↑t), Membership.mem (f (Equiv.plift ↑i)) (U i)
    hU : ∀ (i : ↑t), And (MeasurableSet (U i)) (HasSubset.Subset (U i) (V i))
    ⊢ Exists fun t_1 => And (Membership.mem (iInf fun i => f i) t_1) (And (Measura …
  -/
  refine ⟨⋂ i : t, U i, ?_, ?_, ?_⟩
    /-
      case intro.intro.intro.intro.refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      δ' : Type u_5
      ι : Sort uι
      s : Set α
      inst✝¹ : MeasurableSpace α
      f : ι → Filter α
      inst✝ : ∀ (i : ι), (f i).IsMeasurablyGenerated
      t : Set (PLift ι)
      ht : t.Finite
      V : ↑t → Set α
      hVf : ∀ (i : ↑t), Membership.mem (f (Equiv.plift ↑i)) (V i)
      U : ↑t → Set α
      hUf : ∀ (i : ↑t), Membership.mem (f (Equiv.plift ↑i)) (U i)
      hU : ∀ (i : ↑t), And (MeasurableSet (U i)) (HasSubset.Subset (U i) (V i))
      ⊢ Membership.mem (iInf fun i => f i) (Set.iInter fun i => U i)
    -/
  · rw [← Equiv.plift.surjective.iInf_comp, mem_iInf]
    /-
      case intro.intro.intro.intro.refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      δ' : Type u_5
      ι : Sort uι
      s : Set α
      inst✝¹ : MeasurableSpace α
      f : ι → Filter α
      inst✝ : ∀ (i : ι), (f i).IsMeasurablyGenerated
      t : Set (PLift ι)
      ht : t.Finite
      V : ↑t → Set α
      hVf : ∀ (i : ↑t), Membership.mem (f (Equiv.plift ↑i)) (V i)
      U : ↑t → Set α
      hUf : ∀ (i : ↑t), Membership.mem (f (Equiv.plift ↑i)) (U i)
      hU : ∀ (i : ↑t), And (MeasurableSet (U i)) (HasSubset.Subset (U i) (V i))
      ⊢ Exists fun I => And I.Finite (Exists fun V => And (∀ (i : ↑I), Membership.me …
    -/
    exact ⟨t, ht, U, hUf, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.refine_2
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      δ' : Type u_5
      ι : Sort uι
      s : Set α
      inst✝¹ : MeasurableSpace α
      f : ι → Filter α
      inst✝ : ∀ (i : ι), (f i).IsMeasurablyGenerated
      t : Set (PLift ι)
      ht : t.Finite
      V : ↑t → Set α
      hVf : ∀ (i : ↑t), Membership.mem (f (Equiv.plift ↑i)) (V i)
      U : ↑t → Set α
      hUf : ∀ (i : ↑t), Membership.mem (f (Equiv.plift ↑i)) (U i)
      hU : ∀ (i : ↑t), And (MeasurableSet (U i)) (HasSubset.Subset (U i) (V i))
      ⊢ MeasurableSet (Set.iInter fun i => U i)
    -/
  · haveI := ht.countable.toEncodable.countable
    /-
      case intro.intro.intro.intro.refine_2
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      δ' : Type u_5
      ι : Sort uι
      s : Set α
      inst✝¹ : MeasurableSpace α
      f : ι → Filter α
      inst✝ : ∀ (i : ι), (f i).IsMeasurablyGenerated
      t : Set (PLift ι)
      ht : t.Finite
      V : ↑t → Set α
      hVf : ∀ (i : ↑t), Membership.mem (f (Equiv.plift ↑i)) (V i)
      U : ↑t → Set α
      hUf : ∀ (i : ↑t), Membership.mem (f (Equiv.plift ↑i)) (U i)
      hU : ∀ (i : ↑t), And (MeasurableSet (U i)) (HasSubset.Subset (U i) (V i))
      this : Countable ↑t
      ⊢ MeasurableSet (Set.iInter fun i => U i)
    -/
    exact MeasurableSet.iInter fun i => (hU i).1
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.refine_3
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      δ' : Type u_5
      ι : Sort uι
      s : Set α
      inst✝¹ : MeasurableSpace α
      f : ι → Filter α
      inst✝ : ∀ (i : ι), (f i).IsMeasurablyGenerated
      t : Set (PLift ι)
      ht : t.Finite
      V : ↑t → Set α
      hVf : ∀ (i : ↑t), Membership.mem (f (Equiv.plift ↑i)) (V i)
      U : ↑t → Set α
      hUf : ∀ (i : ↑t), Membership.mem (f (Equiv.plift ↑i)) (U i)
      hU : ∀ (i : ↑t), And (MeasurableSet (U i)) (HasSubset.Subset (U i) (V i))
      ⊢ HasSubset.Subset (Set.iInter fun i => U i) (Set.iInter fun i => V i)
    -/
  · exact iInter_mono fun i => (hU i).2
    /-
      🎉 no goals
    -/


/-- The set of points for which a sequence of measurable functions converges to a given value
is measurable. -/
@[measurability]
lemma measurableSet_tendsto {_ : MeasurableSpace β} [MeasurableSpace γ]
    [Countable δ] {l : Filter δ} [l.IsCountablyGenerated]
    (l' : Filter γ) [l'.IsCountablyGenerated] [hl' : l'.IsMeasurablyGenerated]
    {f : δ → β → γ} (hf : ∀ i, Measurable (f i)) :
    MeasurableSet { x | Tendsto (fun n ↦ f n x) l l' } := by
  /-
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    x✝ : MeasurableSpace β
    inst✝³ : MeasurableSpace γ
    inst✝² : Countable δ
    l : Filter δ
    inst✝¹ : l.IsCountablyGenerated
    l' : Filter γ
    inst✝ : l'.IsCountablyGenerated
    hl' : l'.IsMeasurablyGenerated
    f : δ → β → γ
    hf : ∀ (i : δ), Measurable (f i)
    ⊢ MeasurableSet (setOf fun x => Filter.Tendsto (fun n => f n x) l l')
  -/
  rcases l.exists_antitone_basis with ⟨u, hu⟩
  rcases (Filter.hasBasis_self.mpr hl'.exists_measurable_subset).exists_antitone_subbasis with
    ⟨v, v_meas, hv⟩
  /-
    case intro.intro.intro
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    x✝ : MeasurableSpace β
    inst✝³ : MeasurableSpace γ
    inst✝² : Countable δ
    l : Filter δ
    inst✝¹ : l.IsCountablyGenerated
    l' : Filter γ
    inst✝ : l'.IsCountablyGenerated
    hl' : l'.IsMeasurablyGenerated
    f : δ → β → γ
    hf : ∀ (i : δ), Measurable (f i)
    u : Nat → Set δ
    hu : l.HasAntitoneBasis u
    v : Nat → Set γ
    v_meas : ∀ (i : Nat), And (Membership.mem l' (v i)) (MeasurableSet (v i))
    hv : l'.HasAntitoneBasis fun i => id (v i)
    ⊢ MeasurableSet (setOf fun x => Filter.Tendsto (fun n => f n x) l l')
  -/
  simp only [hu.tendsto_iff hv.toHasBasis, true_imp_iff, true_and, setOf_forall, setOf_exists]
  exact .iInter fun n ↦ .iUnion fun _ ↦ .biInter (to_countable _) fun i _ ↦
    (v_meas n).2.preimage (hf i)


/-- We say that a collection of sets is countably spanning if a countable subset spans the
whole type. This is a useful condition in various parts of measure theory. For example, it is
a needed condition to show that the product of two collections generate the product sigma algebra,
see `generateFrom_prod_eq`. -/
def IsCountablySpanning (C : Set (Set α)) : Prop :=
  ∃ s : ℕ → Set α, (∀ n, s n ∈ C) ∧ ⋃ n, s n = univ


theorem isCountablySpanning_measurableSet [MeasurableSpace α] :
    IsCountablySpanning { s : Set α | MeasurableSet s } :=
  ⟨fun _ => univ, fun _ => MeasurableSet.univ, iUnion_const _⟩


/-- Rectangles of countably spanning sets are countably spanning. -/
lemma IsCountablySpanning.prod {C : Set (Set α)} {D : Set (Set β)} (hC : IsCountablySpanning C)
    (hD : IsCountablySpanning D) : IsCountablySpanning (image2 (· ×ˢ ·) C D) := by
  /-
    α : Type u_1
    β : Type u_2
    C : Set (Set α)
    D : Set (Set β)
    hC : IsCountablySpanning C
    hD : IsCountablySpanning D
    ⊢ IsCountablySpanning (Set.image2 (fun x1 x2 => SProd.sprod x1 x2) C D)
  -/
  rcases hC, hD with ⟨⟨s, h1s, h2s⟩, t, h1t, h2t⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    C : Set (Set α)
    D : Set (Set β)
    s : Nat → Set α
    h1s : ∀ (n : Nat), Membership.mem C (s n)
    h2s : Eq (Set.iUnion fun n => s n) Set.univ
    t : Nat → Set β
    h1t : ∀ (n : Nat), Membership.mem D (t n)
    h2t : Eq (Set.iUnion fun n => t n) Set.univ
    ⊢ IsCountablySpanning (Set.image2 (fun x1 x2 => SProd.sprod x1 x2) C D)
  -/
  refine ⟨fun n => s n.unpair.1 ×ˢ t n.unpair.2, fun n => mem_image2_of_mem (h1s _) (h1t _), ?_⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    C : Set (Set α)
    D : Set (Set β)
    s : Nat → Set α
    h1s : ∀ (n : Nat), Membership.mem C (s n)
    h2s : Eq (Set.iUnion fun n => s n) Set.univ
    t : Nat → Set β
    h1t : ∀ (n : Nat), Membership.mem D (t n)
    h2t : Eq (Set.iUnion fun n => t n) Set.univ
    ⊢ Eq (Set.iUnion fun n => (fun n => SProd.sprod (s (Nat.unpair n).1) (t (Nat.u …
  -/
  rw [iUnion_unpair_prod, h2s, h2t, univ_prod_univ]
  /-
    🎉 no goals
  -/


protected theorem iUnion_of_monotone_of_frequently
    {ι : Type*} [Preorder ι] [(atTop : Filter ι).IsCountablyGenerated] {s : ι → Set α}
    (hsm : Monotone s) (hs : ∃ᶠ i in atTop, MeasurableSet (s i)) : MeasurableSet (⋃ i, s i) := by
  /-
    α : Type u_1
    inst✝² : MeasurableSpace α
    ι : Type u_6
    inst✝¹ : Preorder ι
    inst✝ : Filter.atTop.IsCountablyGenerated
    s : ι → Set α
    hsm : Monotone s
    hs : Filter.Frequently (fun i => MeasurableSet (s i)) Filter.atTop
    ⊢ MeasurableSet (Set.iUnion fun i => s i)
  -/
  rcases exists_seq_forall_of_frequently hs with ⟨x, hx, hxm⟩
  /-
    case intro.intro
    α : Type u_1
    inst✝² : MeasurableSpace α
    ι : Type u_6
    inst✝¹ : Preorder ι
    inst✝ : Filter.atTop.IsCountablyGenerated
    s : ι → Set α
    hsm : Monotone s
    hs : Filter.Frequently (fun i => MeasurableSet (s i)) Filter.atTop
    x : Nat → ι
    hx : Filter.Tendsto x Filter.atTop Filter.atTop
    hxm : ∀ (n : Nat), MeasurableSet (s (x n))
    ⊢ MeasurableSet (Set.iUnion fun i => s i)
  -/
  rw [← hsm.iUnion_comp_tendsto_atTop hx]
  /-
    case intro.intro
    α : Type u_1
    inst✝² : MeasurableSpace α
    ι : Type u_6
    inst✝¹ : Preorder ι
    inst✝ : Filter.atTop.IsCountablyGenerated
    s : ι → Set α
    hsm : Monotone s
    hs : Filter.Frequently (fun i => MeasurableSet (s i)) Filter.atTop
    x : Nat → ι
    hx : Filter.Tendsto x Filter.atTop Filter.atTop
    hxm : ∀ (n : Nat), MeasurableSet (s (x n))
    ⊢ MeasurableSet (Set.iUnion fun a => s (x a))
  -/
  exact .iUnion hxm
  /-
    🎉 no goals
  -/


protected theorem iInter_of_antitone_of_frequently
    {ι : Type*} [Preorder ι] [(atTop : Filter ι).IsCountablyGenerated] {s : ι → Set α}
    (hsm : Antitone s) (hs : ∃ᶠ i in atTop, MeasurableSet (s i)) : MeasurableSet (⋂ i, s i) := by
  /-
    α : Type u_1
    inst✝² : MeasurableSpace α
    ι : Type u_6
    inst✝¹ : Preorder ι
    inst✝ : Filter.atTop.IsCountablyGenerated
    s : ι → Set α
    hsm : Antitone s
    hs : Filter.Frequently (fun i => MeasurableSet (s i)) Filter.atTop
    ⊢ MeasurableSet (Set.iInter fun i => s i)
  -/
  rw [← compl_iff, compl_iInter]
  /-
    α : Type u_1
    inst✝² : MeasurableSpace α
    ι : Type u_6
    inst✝¹ : Preorder ι
    inst✝ : Filter.atTop.IsCountablyGenerated
    s : ι → Set α
    hsm : Antitone s
    hs : Filter.Frequently (fun i => MeasurableSet (s i)) Filter.atTop
    ⊢ MeasurableSet (Set.iUnion fun i => HasCompl.compl (s i))
  -/
  exact .iUnion_of_monotone_of_frequently (compl_anti.comp hsm) <| hs.mono fun _ ↦ .compl
  /-
    🎉 no goals
  -/


protected theorem iUnion_of_monotone {ι : Type*} [Preorder ι] [IsDirected ι (· ≤ ·)]
    [(atTop : Filter ι).IsCountablyGenerated] {s : ι → Set α}
    (hsm : Monotone s) (hs : ∀ i, MeasurableSet (s i)) : MeasurableSet (⋃ i, s i) := by
  cases isEmpty_or_nonempty ι with
  | inl _ => simp
  | inr _ => exact .iUnion_of_monotone_of_frequently hsm <| .of_forall hs


protected theorem iInter_of_antitone {ι : Type*} [Preorder ι] [IsDirected ι (· ≤ ·)]
    [(atTop : Filter ι).IsCountablyGenerated] {s : ι → Set α}
    (hsm : Antitone s) (hs : ∀ i, MeasurableSet (s i)) : MeasurableSet (⋂ i, s i) := by
  /-
    α : Type u_1
    inst✝³ : MeasurableSpace α
    ι : Type u_6
    inst✝² : Preorder ι
    inst✝¹ : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝ : Filter.atTop.IsCountablyGenerated
    s : ι → Set α
    hsm : Antitone s
    hs : ∀ (i : ι), MeasurableSet (s i)
    ⊢ MeasurableSet (Set.iInter fun i => s i)
  -/
  rw [← compl_iff, compl_iInter]
  /-
    α : Type u_1
    inst✝³ : MeasurableSpace α
    ι : Type u_6
    inst✝² : Preorder ι
    inst✝¹ : IsDirected ι fun x1 x2 => LE.le x1 x2
    inst✝ : Filter.atTop.IsCountablyGenerated
    s : ι → Set α
    hsm : Antitone s
    hs : ∀ (i : ι), MeasurableSet (s i)
    ⊢ MeasurableSet (Set.iUnion fun i => HasCompl.compl (s i))
  -/
  exact .iUnion_of_monotone (compl_anti.comp hsm) fun i ↦ (hs i).compl
  /-
    🎉 no goals
  -/


instance Subtype.instMembership : Membership α (Subtype (MeasurableSet : Set α → Prop)) :=
  ⟨fun s a => a ∈ (s : Set α)⟩


@[simp]
theorem mem_coe (a : α) (s : Subtype (MeasurableSet : Set α → Prop)) : a ∈ (s : Set α) ↔ a ∈ s :=
  Iff.rfl


instance Subtype.instEmptyCollection : EmptyCollection (Subtype (MeasurableSet : Set α → Prop)) :=
  ⟨⟨∅, MeasurableSet.empty⟩⟩


@[simp]
theorem coe_empty : ↑(∅ : Subtype (MeasurableSet : Set α → Prop)) = (∅ : Set α) :=
  rfl


instance Subtype.instInsert [MeasurableSingletonClass α] :
    Insert α (Subtype (MeasurableSet : Set α → Prop)) :=
  ⟨fun a s => ⟨insert a (s : Set α), s.prop.insert a⟩⟩


@[simp]
theorem coe_insert [MeasurableSingletonClass α] (a : α)
    (s : Subtype (MeasurableSet : Set α → Prop)) :
    ↑(Insert.insert a s) = (Insert.insert a s : Set α) :=
  rfl


instance Subtype.instSingleton [MeasurableSingletonClass α] :
    Singleton α (Subtype (MeasurableSet : Set α → Prop)) :=
  ⟨fun a => ⟨{a}, .singleton _⟩⟩


@[simp] theorem coe_singleton [MeasurableSingletonClass α] (a : α) :
    ↑({a} : Subtype (MeasurableSet : Set α → Prop)) = ({a} : Set α) :=
  rfl


instance Subtype.instLawfulSingleton [MeasurableSingletonClass α] :
    LawfulSingleton α (Subtype (MeasurableSet : Set α → Prop)) :=
  ⟨fun _ => Subtype.eq <| insert_emptyc_eq _⟩


instance Subtype.instHasCompl : HasCompl (Subtype (MeasurableSet : Set α → Prop)) :=
  ⟨fun x => ⟨xᶜ, x.prop.compl⟩⟩


@[simp]
theorem coe_compl (s : Subtype (MeasurableSet : Set α → Prop)) : ↑sᶜ = (sᶜ : Set α) :=
  rfl


instance Subtype.instUnion : Union (Subtype (MeasurableSet : Set α → Prop)) :=
  ⟨fun x y => ⟨(x : Set α) ∪ y, x.prop.union y.prop⟩⟩


@[simp]
theorem coe_union (s t : Subtype (MeasurableSet : Set α → Prop)) : ↑(s ∪ t) = (s ∪ t : Set α) :=
  rfl


instance Subtype.instSup : Max (Subtype (MeasurableSet : Set α → Prop)) :=
  ⟨fun x y => x ∪ y⟩


@[simp]
protected theorem sup_eq_union (s t : {s : Set α // MeasurableSet s}) : s ⊔ t = s ∪ t := rfl


instance Subtype.instInter : Inter (Subtype (MeasurableSet : Set α → Prop)) :=
  ⟨fun x y => ⟨x ∩ y, x.prop.inter y.prop⟩⟩


@[simp]
theorem coe_inter (s t : Subtype (MeasurableSet : Set α → Prop)) : ↑(s ∩ t) = (s ∩ t : Set α) :=
  rfl


instance Subtype.instInf : Min (Subtype (MeasurableSet : Set α → Prop)) :=
  ⟨fun x y => x ∩ y⟩


@[simp]
protected theorem inf_eq_inter (s t : {s : Set α // MeasurableSet s}) : s ⊓ t = s ∩ t := rfl


instance Subtype.instSDiff : SDiff (Subtype (MeasurableSet : Set α → Prop)) :=
  ⟨fun x y => ⟨x \ y, x.prop.diff y.prop⟩⟩

-- TODO: Why does it complain that `x ⇨ y` is noncomputable?

noncomputable instance Subtype.instHImp : HImp (Subtype (MeasurableSet : Set α → Prop)) where
  himp x y := ⟨x ⇨ y, x.prop.himp y.prop⟩


@[simp]
theorem coe_sdiff (s t : Subtype (MeasurableSet : Set α → Prop)) : ↑(s \ t) = (s : Set α) \ t :=
  rfl


@[simp]
lemma coe_himp (s t : Subtype (MeasurableSet : Set α → Prop)) : ↑(s ⇨ t) = (s ⇨ t : Set α) := rfl


instance Subtype.instBot : Bot (Subtype (MeasurableSet : Set α → Prop)) := ⟨∅⟩


@[simp]
theorem coe_bot : ↑(⊥ : Subtype (MeasurableSet : Set α → Prop)) = (⊥ : Set α) :=
  rfl


instance Subtype.instTop : Top (Subtype (MeasurableSet : Set α → Prop)) :=
  ⟨⟨Set.univ, MeasurableSet.univ⟩⟩


@[simp]
theorem coe_top : ↑(⊤ : Subtype (MeasurableSet : Set α → Prop)) = (⊤ : Set α) :=
  rfl


noncomputable instance Subtype.instBooleanAlgebra :
    BooleanAlgebra (Subtype (MeasurableSet : Set α → Prop)) :=
  Subtype.coe_injective.booleanAlgebra _ coe_union coe_inter coe_top coe_bot coe_compl coe_sdiff
    coe_himp


@[measurability]
theorem measurableSet_blimsup {s : ℕ → Set α} {p : ℕ → Prop} (h : ∀ n, p n → MeasurableSet (s n)) :
    MeasurableSet <| blimsup s atTop p := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : Nat → Set α
    p : Nat → Prop
    h : ∀ (n : Nat), p n → MeasurableSet (s n)
    ⊢ MeasurableSet (Filter.blimsup s Filter.atTop p)
  -/
  simp only [blimsup_eq_iInf_biSup_of_nat, iSup_eq_iUnion, iInf_eq_iInter]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : Nat → Set α
    p : Nat → Prop
    h : ∀ (n : Nat), p n → MeasurableSet (s n)
    ⊢ MeasurableSet (Set.iInter fun i => Set.iUnion fun j => Set.iUnion fun x => s …
  -/
  exact .iInter fun _ => .iUnion fun m => .iUnion fun hm => h m hm.1
  /-
    🎉 no goals
  -/


@[measurability]
theorem measurableSet_bliminf {s : ℕ → Set α} {p : ℕ → Prop} (h : ∀ n, p n → MeasurableSet (s n)) :
    MeasurableSet <| Filter.bliminf s Filter.atTop p := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : Nat → Set α
    p : Nat → Prop
    h : ∀ (n : Nat), p n → MeasurableSet (s n)
    ⊢ MeasurableSet (Filter.bliminf s Filter.atTop p)
  -/
  simp only [Filter.bliminf_eq_iSup_biInf_of_nat, iInf_eq_iInter, iSup_eq_iUnion]
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : Nat → Set α
    p : Nat → Prop
    h : ∀ (n : Nat), p n → MeasurableSet (s n)
    ⊢ MeasurableSet (Set.iUnion fun i => Set.iInter fun j => Set.iInter fun x => s …
  -/
  exact .iUnion fun n => .iInter fun m => .iInter fun hm => h m hm.1
  /-
    🎉 no goals
  -/


@[measurability]
theorem measurableSet_limsup {s : ℕ → Set α} (hs : ∀ n, MeasurableSet <| s n) :
    MeasurableSet <| Filter.limsup s Filter.atTop := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : Nat → Set α
    hs : ∀ (n : Nat), MeasurableSet (s n)
    ⊢ MeasurableSet (Filter.limsup s Filter.atTop)
  -/
  simpa only [← blimsup_true] using measurableSet_blimsup fun n _ => hs n
  /-
    🎉 no goals
  -/


@[measurability]
theorem measurableSet_liminf {s : ℕ → Set α} (hs : ∀ n, MeasurableSet <| s n) :
    MeasurableSet <| Filter.liminf s Filter.atTop := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    s : Nat → Set α
    hs : ∀ (n : Nat), MeasurableSet (s n)
    ⊢ MeasurableSet (Filter.liminf s Filter.atTop)
  -/
  simpa only [← bliminf_true] using measurableSet_bliminf fun n _ => hs n
  /-
    🎉 no goals
  -/


