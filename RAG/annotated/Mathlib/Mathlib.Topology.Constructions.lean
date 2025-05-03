instance {r : X → X → Prop} [t : TopologicalSpace X] : TopologicalSpace (Quot r) :=
  coinduced (Quot.mk r) t


instance instTopologicalSpaceQuotient {s : Setoid X} [t : TopologicalSpace X] :
    TopologicalSpace (Quotient s) :=
  coinduced Quotient.mk' t


instance instTopologicalSpaceProd [t₁ : TopologicalSpace X] [t₂ : TopologicalSpace Y] :
    TopologicalSpace (X × Y) :=
  induced Prod.fst t₁ ⊓ induced Prod.snd t₂


instance instTopologicalSpaceSum [t₁ : TopologicalSpace X] [t₂ : TopologicalSpace Y] :
    TopologicalSpace (X ⊕ Y) :=
  coinduced Sum.inl t₁ ⊔ coinduced Sum.inr t₂


instance instTopologicalSpaceSigma {ι : Type*} {X : ι → Type v} [t₂ : ∀ i, TopologicalSpace (X i)] :
    TopologicalSpace (Sigma X) :=
  ⨆ i, coinduced (Sigma.mk i) (t₂ i)


instance Pi.topologicalSpace {ι : Type*} {Y : ι → Type v} [t₂ : (i : ι) → TopologicalSpace (Y i)] :
    TopologicalSpace ((i : ι) → Y i) :=
  ⨅ i, induced (fun f => f i) (t₂ i)


instance ULift.topologicalSpace [t : TopologicalSpace X] : TopologicalSpace (ULift.{v, u} X) :=
  t.induced ULift.down


instance : TopologicalSpace (Additive X) := ‹TopologicalSpace X›

instance : TopologicalSpace (Multiplicative X) := ‹TopologicalSpace X›


instance [DiscreteTopology X] : DiscreteTopology (Additive X) := ‹DiscreteTopology X›

instance [DiscreteTopology X] : DiscreteTopology (Multiplicative X) := ‹DiscreteTopology X›


theorem continuous_ofMul : Continuous (ofMul : X → Additive X) := continuous_id


theorem continuous_toMul : Continuous (toMul : Additive X → X) := continuous_id


theorem continuous_ofAdd : Continuous (ofAdd : X → Multiplicative X) := continuous_id


theorem continuous_toAdd : Continuous (toAdd : Multiplicative X → X) := continuous_id


theorem isOpenMap_ofMul : IsOpenMap (ofMul : X → Additive X) := IsOpenMap.id


theorem isOpenMap_toMul : IsOpenMap (toMul : Additive X → X) := IsOpenMap.id


theorem isOpenMap_ofAdd : IsOpenMap (ofAdd : X → Multiplicative X) := IsOpenMap.id


theorem isOpenMap_toAdd : IsOpenMap (toAdd : Multiplicative X → X) := IsOpenMap.id


theorem isClosedMap_ofMul : IsClosedMap (ofMul : X → Additive X) := IsClosedMap.id


theorem isClosedMap_toMul : IsClosedMap (toMul : Additive X → X) := IsClosedMap.id


theorem isClosedMap_ofAdd : IsClosedMap (ofAdd : X → Multiplicative X) := IsClosedMap.id


theorem isClosedMap_toAdd : IsClosedMap (toAdd : Multiplicative X → X) := IsClosedMap.id


theorem nhds_ofMul (x : X) : 𝓝 (ofMul x) = map ofMul (𝓝 x) := rfl


theorem nhds_ofAdd (x : X) : 𝓝 (ofAdd x) = map ofAdd (𝓝 x) := rfl


theorem nhds_toMul (x : Additive X) : 𝓝 x.toMul = map toMul (𝓝 x) := rfl


theorem nhds_toAdd (x : Multiplicative X) : 𝓝 x.toAdd = map toAdd (𝓝 x) := rfl


instance OrderDual.instTopologicalSpace : TopologicalSpace Xᵒᵈ := ‹_›

instance OrderDual.instDiscreteTopology [DiscreteTopology X] : DiscreteTopology Xᵒᵈ := ‹_›


theorem continuous_toDual : Continuous (toDual : X → Xᵒᵈ) := continuous_id


theorem continuous_ofDual : Continuous (ofDual : Xᵒᵈ → X) := continuous_id


theorem isOpenMap_toDual : IsOpenMap (toDual : X → Xᵒᵈ) := IsOpenMap.id


theorem isOpenMap_ofDual : IsOpenMap (ofDual : Xᵒᵈ → X) := IsOpenMap.id


theorem isClosedMap_toDual : IsClosedMap (toDual : X → Xᵒᵈ) := IsClosedMap.id


theorem isClosedMap_ofDual : IsClosedMap (ofDual : Xᵒᵈ → X) := IsClosedMap.id


theorem nhds_toDual (x : X) : 𝓝 (toDual x) = map toDual (𝓝 x) := rfl


theorem nhds_ofDual (x : X) : 𝓝 (ofDual x) = map ofDual (𝓝 x) := rfl


instance OrderDual.instNeBotNhdsWithinIoi [(𝓝[<] x).NeBot] : (𝓝[>] toDual x).NeBot := ‹_›

instance OrderDual.instNeBotNhdsWithinIio [(𝓝[>] x).NeBot] : (𝓝[<] toDual x).NeBot := ‹_›


theorem Quotient.preimage_mem_nhds [TopologicalSpace X] [s : Setoid X] {V : Set <| Quotient s}
    {x : X} (hs : V ∈ 𝓝 (Quotient.mk' x)) : Quotient.mk' ⁻¹' V ∈ 𝓝 x :=
  preimage_nhds_coinduced hs


/-- The image of a dense set under `Quotient.mk'` is a dense set. -/
theorem Dense.quotient [Setoid X] [TopologicalSpace X] {s : Set X} (H : Dense s) :
    Dense (Quotient.mk' '' s) :=
  Quotient.mk''_surjective.denseRange.dense_image continuous_coinduced_rng H


/-- The composition of `Quotient.mk'` and a function with dense range has dense range. -/
theorem DenseRange.quotient [Setoid X] [TopologicalSpace X] {f : Y → X} (hf : DenseRange f) :
    DenseRange (Quotient.mk' ∘ f) :=
  Quotient.mk''_surjective.denseRange.comp hf continuous_coinduced_rng


theorem continuous_map_of_le {α : Type*} [TopologicalSpace α]
    {s t : Setoid α} (h : s ≤ t) : Continuous (Setoid.map_of_le h) :=
  continuous_coinduced_rng


theorem continuous_map_sInf {α : Type*} [TopologicalSpace α]
    {S : Set (Setoid α)} {s : Setoid α} (h : s ∈ S) : Continuous (Setoid.map_sInf h) :=
  continuous_coinduced_rng


instance {p : X → Prop} [TopologicalSpace X] [DiscreteTopology X] : DiscreteTopology (Subtype p) :=
  ⟨bot_unique fun s _ => ⟨(↑) '' s, isOpen_discrete _, preimage_image_eq _ Subtype.val_injective⟩⟩


instance Sum.discreteTopology [TopologicalSpace X] [TopologicalSpace Y] [h : DiscreteTopology X]
    [hY : DiscreteTopology Y] : DiscreteTopology (X ⊕ Y) :=
                          /-
                            X : Type u
                            Y : Type v
                            Z : Type u_1
                            W : Type u_2
                            ε : Type u_3
                            ζ : Type u_4
                            inst✝¹ : TopologicalSpace X
                            inst✝ : TopologicalSpace Y
                            h : DiscreteTopology X
                            hY : DiscreteTopology Y
                            ⊢ And (Eq (TopologicalSpace.coinduced Sum.inl inst✝¹) Bot.bot) (Eq (Topologica …
                          -/
  ⟨sup_eq_bot_iff.2 <| by simp [h.eq_bot, hY.eq_bot]⟩
                          /-
                            🎉 no goals
                          -/


instance Sigma.discreteTopology {ι : Type*} {Y : ι → Type v} [∀ i, TopologicalSpace (Y i)]
    [h : ∀ i, DiscreteTopology (Y i)] : DiscreteTopology (Sigma Y) :=
                             /-
                               X : Type u
                               Y✝ : Type v
                               Z : Type u_1
                               W : Type u_2
                               ε : Type u_3
                               ζ : Type u_4
                               ι : Type u_5
                               Y : ι → Type v
                               inst✝ : (i : ι) → TopologicalSpace (Y i)
                               h : ∀ (i : ι), DiscreteTopology (Y i)
                               x✝ : ι
                               ⊢ Eq (TopologicalSpace.coinduced (Sigma.mk x✝) (inst✝ x✝)) Bot.bot
                             -/
  ⟨iSup_eq_bot.2 fun _ => by simp only [(h _).eq_bot, coinduced_bot]⟩
                             /-
                               🎉 no goals
                             -/


@[simp] lemma comap_nhdsWithin_range {α β} [TopologicalSpace β] (f : α → β) (y : β) :
    comap f (𝓝[range f] y) = comap f (𝓝 y) := comap_inf_principal_range


theorem mem_nhds_subtype (s : Set X) (x : { x // x ∈ s }) (t : Set { x // x ∈ s }) :
    t ∈ 𝓝 x ↔ ∃ u ∈ 𝓝 (x : X), Subtype.val ⁻¹' u ⊆ t :=
  mem_nhds_induced _ x t


theorem nhds_subtype (s : Set X) (x : { x // x ∈ s }) : 𝓝 x = comap (↑) (𝓝 (x : X)) :=
  nhds_induced _ x


lemma nhds_subtype_eq_comap_nhdsWithin (s : Set X) (x : { x // x ∈ s }) :
    𝓝 x = comap (↑) (𝓝[s] (x : X)) := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    x : Subtype fun x => Membership.mem s x
    ⊢ Eq (nhds x) (Filter.comap Subtype.val (nhdsWithin (↑x) s))
  -/
  rw [nhds_subtype, ← comap_nhdsWithin_range, Subtype.range_val]
  /-
    🎉 no goals
  -/


theorem nhdsWithin_subtype_eq_bot_iff {s t : Set X} {x : s} :
    𝓝[((↑) : s → X) ⁻¹' t] x = ⊥ ↔ 𝓝[t] (x : X) ⊓ 𝓟 s = ⊥ := by
  rw [inf_principal_eq_bot_iff_comap, nhdsWithin, nhdsWithin, comap_inf, comap_principal,
    nhds_induced]


theorem nhds_ne_subtype_eq_bot_iff {S : Set X} {x : S} :
    𝓝[≠] x = ⊥ ↔ 𝓝[≠] (x : X) ⊓ 𝓟 S = ⊥ := by
  rw [← nhdsWithin_subtype_eq_bot_iff, preimage_compl, ← image_singleton,
    Subtype.coe_injective.preimage_image]


theorem nhds_ne_subtype_neBot_iff {S : Set X} {x : S} :
    (𝓝[≠] x).NeBot ↔ (𝓝[≠] (x : X) ⊓ 𝓟 S).NeBot := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    S : Set X
    x : ↑S
    ⊢ Iff (nhdsWithin x (HasCompl.compl (Singleton.singleton x))).NeBot (Min.min ( …
  -/
  rw [neBot_iff, neBot_iff, not_iff_not, nhds_ne_subtype_eq_bot_iff]
  /-
    🎉 no goals
  -/


theorem discreteTopology_subtype_iff {S : Set X} :
    DiscreteTopology S ↔ ∀ x ∈ S, 𝓝[≠] x ⊓ 𝓟 S = ⊥ := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    S : Set X
    ⊢ Iff (DiscreteTopology ↑S) (∀ (x : X), Membership.mem S x → Eq (Min.min (nhds …
  -/
  simp_rw [discreteTopology_iff_nhds_ne, SetCoe.forall', nhds_ne_subtype_eq_bot_iff]
  /-
    🎉 no goals
  -/


/-- A type synonym equipped with the topology whose open sets are the empty set and the sets with
finite complements. -/
def CofiniteTopology (X : Type*) := X


/-- The identity equivalence between `` and `CofiniteTopology `. -/
def of : X ≃ CofiniteTopology X :=
  Equiv.refl X


instance [Inhabited X] : Inhabited (CofiniteTopology X) where default := of default


instance : TopologicalSpace (CofiniteTopology X) where
  IsOpen s := s.Nonempty → Set.Finite sᶜ
                    /-
                      X : Type u
                      Y : Type v
                      Z : Type u_1
                      W : Type u_2
                      ε : Type u_3
                      ζ : Type u_4
                      ⊢ (fun s => s.Nonempty → (HasCompl.compl s).Finite) Set.univ
                    -/
  isOpen_univ := by simp
                    /-
                      🎉 no goals
                    -/
  isOpen_inter s t := by
    /-
      X : Type u
      Y : Type v
      Z : Type u_1
      W : Type u_2
      ε : Type u_3
      ζ : Type u_4
      s t : Set (CofiniteTopology X)
      ⊢ (fun s => s.Nonempty → (HasCompl.compl s).Finite) s → (fun s => s.Nonempty → …
    -/
    rintro hs ht ⟨x, hxs, hxt⟩
    /-
      case intro.intro
      X : Type u
      Y : Type v
      Z : Type u_1
      W : Type u_2
      ε : Type u_3
      ζ : Type u_4
      s t : Set (CofiniteTopology X)
      hs : s.Nonempty → (HasCompl.compl s).Finite
      ht : t.Nonempty → (HasCompl.compl t).Finite
      x : CofiniteTopology X
      hxs : Membership.mem s x
      hxt : Membership.mem t x
      ⊢ (HasCompl.compl (Inter.inter s t)).Finite
    -/
    rw [compl_inter]
    /-
      case intro.intro
      X : Type u
      Y : Type v
      Z : Type u_1
      W : Type u_2
      ε : Type u_3
      ζ : Type u_4
      s t : Set (CofiniteTopology X)
      hs : s.Nonempty → (HasCompl.compl s).Finite
      ht : t.Nonempty → (HasCompl.compl t).Finite
      x : CofiniteTopology X
      hxs : Membership.mem s x
      hxt : Membership.mem t x
      ⊢ (Union.union (HasCompl.compl s) (HasCompl.compl t)).Finite
    -/
    exact (hs ⟨x, hxs⟩).union (ht ⟨x, hxt⟩)
    /-
      🎉 no goals
    -/
  isOpen_sUnion := by
    /-
      X : Type u
      Y : Type v
      Z : Type u_1
      W : Type u_2
      ε : Type u_3
      ζ : Type u_4
      ⊢ ∀ (s : Set (Set (CofiniteTopology X))), (∀ (t : Set (CofiniteTopology X)), M …
    -/
    rintro s h ⟨x, t, hts, hzt⟩
    /-
      case intro.intro.intro
      X : Type u
      Y : Type v
      Z : Type u_1
      W : Type u_2
      ε : Type u_3
      ζ : Type u_4
      s : Set (Set (CofiniteTopology X))
      h : ∀ (t : Set (CofiniteTopology X)), Membership.mem s t → (fun s => s.Nonempt …
      x : CofiniteTopology X
      t : Set (CofiniteTopology X)
      hts : Membership.mem s t
      hzt : Membership.mem t x
      ⊢ (HasCompl.compl s.sUnion).Finite
    -/
    rw [compl_sUnion]
    /-
      case intro.intro.intro
      X : Type u
      Y : Type v
      Z : Type u_1
      W : Type u_2
      ε : Type u_3
      ζ : Type u_4
      s : Set (Set (CofiniteTopology X))
      h : ∀ (t : Set (CofiniteTopology X)), Membership.mem s t → (fun s => s.Nonempt …
      x : CofiniteTopology X
      t : Set (CofiniteTopology X)
      hts : Membership.mem s t
      hzt : Membership.mem t x
      ⊢ (Set.image HasCompl.compl s).sInter.Finite
    -/
    exact Finite.sInter (mem_image_of_mem _ hts) (h t hts ⟨x, hzt⟩)
    /-
      🎉 no goals
    -/


theorem isOpen_iff {s : Set (CofiniteTopology X)} : IsOpen s ↔ s.Nonempty → sᶜ.Finite :=
  Iff.rfl


theorem isOpen_iff' {s : Set (CofiniteTopology X)} : IsOpen s ↔ s = ∅ ∨ sᶜ.Finite := by
  /-
    X : Type u
    s : Set (CofiniteTopology X)
    ⊢ Iff (IsOpen s) (Or (Eq s EmptyCollection.emptyCollection) (HasCompl.compl s) …
  -/
  simp only [isOpen_iff, nonempty_iff_ne_empty, or_iff_not_imp_left]
  /-
    🎉 no goals
  -/


theorem isClosed_iff {s : Set (CofiniteTopology X)} : IsClosed s ↔ s = univ ∨ s.Finite := by
  /-
    X : Type u
    s : Set (CofiniteTopology X)
    ⊢ Iff (IsClosed s) (Or (Eq s Set.univ) s.Finite)
  -/
  simp only [← isOpen_compl_iff, isOpen_iff', compl_compl, compl_empty_iff]
  /-
    🎉 no goals
  -/


theorem nhds_eq (x : CofiniteTopology X) : 𝓝 x = pure x ⊔ cofinite := by
  /-
    X : Type u
    x : CofiniteTopology X
    ⊢ Eq (nhds x) (Max.max (Pure.pure x) Filter.cofinite)
  -/
  ext U
  /-
    case h
    X : Type u
    x : CofiniteTopology X
    U : Set (CofiniteTopology X)
    ⊢ Iff (Membership.mem (nhds x) U) (Membership.mem (Max.max (Pure.pure x) Filte …
  -/
  rw [mem_nhds_iff]
  /-
    case h
    X : Type u
    x : CofiniteTopology X
    U : Set (CofiniteTopology X)
    ⊢ Iff (Exists fun t => And (HasSubset.Subset t U) (And (IsOpen t) (Membership. …
  -/
  constructor
    /-
      case h.mp
      X : Type u
      x : CofiniteTopology X
      U : Set (CofiniteTopology X)
      ⊢ (Exists fun t => And (HasSubset.Subset t U) (And (IsOpen t) (Membership.mem  …
    -/
  · rintro ⟨V, hVU, V_op, haV⟩
    /-
      case h.mp.intro.intro.intro
      X : Type u
      x : CofiniteTopology X
      U V : Set (CofiniteTopology X)
      hVU : HasSubset.Subset V U
      V_op : IsOpen V
      haV : Membership.mem V x
      ⊢ Membership.mem (Max.max (Pure.pure x) Filter.cofinite) U
    -/
    exact mem_sup.mpr ⟨hVU haV, mem_of_superset (V_op ⟨_, haV⟩) hVU⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      X : Type u
      x : CofiniteTopology X
      U : Set (CofiniteTopology X)
      ⊢ Membership.mem (Max.max (Pure.pure x) Filter.cofinite) U → Exists fun t => A …
    -/
  · rintro ⟨hU : x ∈ U, hU' : Uᶜ.Finite⟩
    /-
      case h.mpr.intro
      X : Type u
      x : CofiniteTopology X
      U : Set (CofiniteTopology X)
      hU : Membership.mem U x
      hU' : (HasCompl.compl U).Finite
      ⊢ Exists fun t => And (HasSubset.Subset t U) (And (IsOpen t) (Membership.mem t …
    -/
    exact ⟨U, Subset.rfl, fun _ => hU', hU⟩
    /-
      🎉 no goals
    -/


theorem mem_nhds_iff {x : CofiniteTopology X} {s : Set (CofiniteTopology X)} :
                                      /-
                                        X : Type u
                                        x : CofiniteTopology X
                                        s : Set (CofiniteTopology X)
                                        ⊢ Iff (Membership.mem (nhds x) s) (And (Membership.mem s x) (HasCompl.compl s) …
                                      -/
    s ∈ 𝓝 x ↔ x ∈ s ∧ sᶜ.Finite := by simp [nhds_eq]
                                      /-
                                        🎉 no goals
                                      -/


@[simp] theorem continuous_prod_mk {f : X → Y} {g : X → Z} :
    (Continuous fun x => (f x, g x)) ↔ Continuous f ∧ Continuous g :=
  (@continuous_inf_rng X (Y × Z) _ _ (TopologicalSpace.induced Prod.fst _)
    (TopologicalSpace.induced Prod.snd _)).trans <|
    continuous_induced_rng.and continuous_induced_rng


@[continuity]
theorem continuous_fst : Continuous (@Prod.fst X Y) :=
  (continuous_prod_mk.1 continuous_id).1


/-- Postcomposing `f` with `Prod.fst` is continuous -/
@[fun_prop]
theorem Continuous.fst {f : X → Y × Z} (hf : Continuous f) : Continuous fun x : X => (f x).1 :=
  continuous_fst.comp hf


/-- Precomposing `f` with `Prod.fst` is continuous -/
theorem Continuous.fst' {f : X → Z} (hf : Continuous f) : Continuous fun x : X × Y => f x.fst :=
  hf.comp continuous_fst


theorem continuousAt_fst {p : X × Y} : ContinuousAt Prod.fst p :=
  continuous_fst.continuousAt


/-- Postcomposing `f` with `Prod.fst` is continuous at `x` -/
@[fun_prop]
theorem ContinuousAt.fst {f : X → Y × Z} {x : X} (hf : ContinuousAt f x) :
    ContinuousAt (fun x : X => (f x).1) x :=
  continuousAt_fst.comp hf


/-- Precomposing `f` with `Prod.fst` is continuous at `(x, y)` -/
theorem ContinuousAt.fst' {f : X → Z} {x : X} {y : Y} (hf : ContinuousAt f x) :
    ContinuousAt (fun x : X × Y => f x.fst) (x, y) :=
  ContinuousAt.comp hf continuousAt_fst


/-- Precomposing `f` with `Prod.fst` is continuous at `x : X × Y` -/
theorem ContinuousAt.fst'' {f : X → Z} {x : X × Y} (hf : ContinuousAt f x.fst) :
    ContinuousAt (fun x : X × Y => f x.fst) x :=
  hf.comp continuousAt_fst


theorem Filter.Tendsto.fst_nhds {X} {l : Filter X} {f : X → Y × Z} {p : Y × Z}
    (h : Tendsto f l (𝓝 p)) : Tendsto (fun a ↦ (f a).1) l (𝓝 <| p.1) :=
  continuousAt_fst.tendsto.comp h


@[continuity]
theorem continuous_snd : Continuous (@Prod.snd X Y) :=
  (continuous_prod_mk.1 continuous_id).2


/-- Postcomposing `f` with `Prod.snd` is continuous -/
@[fun_prop]
theorem Continuous.snd {f : X → Y × Z} (hf : Continuous f) : Continuous fun x : X => (f x).2 :=
  continuous_snd.comp hf


/-- Precomposing `f` with `Prod.snd` is continuous -/
theorem Continuous.snd' {f : Y → Z} (hf : Continuous f) : Continuous fun x : X × Y => f x.snd :=
  hf.comp continuous_snd


theorem continuousAt_snd {p : X × Y} : ContinuousAt Prod.snd p :=
  continuous_snd.continuousAt


/-- Postcomposing `f` with `Prod.snd` is continuous at `x` -/
@[fun_prop]
theorem ContinuousAt.snd {f : X → Y × Z} {x : X} (hf : ContinuousAt f x) :
    ContinuousAt (fun x : X => (f x).2) x :=
  continuousAt_snd.comp hf


/-- Precomposing `f` with `Prod.snd` is continuous at `(x, y)` -/
theorem ContinuousAt.snd' {f : Y → Z} {x : X} {y : Y} (hf : ContinuousAt f y) :
    ContinuousAt (fun x : X × Y => f x.snd) (x, y) :=
  ContinuousAt.comp hf continuousAt_snd


/-- Precomposing `f` with `Prod.snd` is continuous at `x : X × Y` -/
theorem ContinuousAt.snd'' {f : Y → Z} {x : X × Y} (hf : ContinuousAt f x.snd) :
    ContinuousAt (fun x : X × Y => f x.snd) x :=
  hf.comp continuousAt_snd


theorem Filter.Tendsto.snd_nhds {X} {l : Filter X} {f : X → Y × Z} {p : Y × Z}
    (h : Tendsto f l (𝓝 p)) : Tendsto (fun a ↦ (f a).2) l (𝓝 <| p.2) :=
  continuousAt_snd.tendsto.comp h


@[continuity, fun_prop]
theorem Continuous.prod_mk {f : Z → X} {g : Z → Y} (hf : Continuous f) (hg : Continuous g) :
    Continuous fun x => (f x, g x) :=
  continuous_prod_mk.2 ⟨hf, hg⟩


@[continuity]
theorem Continuous.Prod.mk (x : X) : Continuous fun y : Y => (x, y) :=
  continuous_const.prod_mk continuous_id


@[continuity]
theorem Continuous.Prod.mk_left (y : Y) : Continuous fun x : X => (x, y) :=
  continuous_id.prod_mk continuous_const


/-- If `f x y` is continuous in `x` for all `y ∈ s`,
then the set of `x` such that `f x` maps `s` to `t` is closed. -/
lemma IsClosed.setOf_mapsTo {α : Type*} {f : X → α → Z} {s : Set α} {t : Set Z} (ht : IsClosed t)
    (hf : ∀ a ∈ s, Continuous (f · a)) : IsClosed {x | MapsTo (f x) s t} := by
  /-
    X : Type u
    Z : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Z
    α : Type u_5
    f : X → α → Z
    s : Set α
    t : Set Z
    ht : IsClosed t
    hf : ∀ (a : α), Membership.mem s a → Continuous fun x => f x a
    ⊢ IsClosed (setOf fun x => Set.MapsTo (f x) s t)
  -/
  simpa only [MapsTo, setOf_forall] using isClosed_biInter fun y hy ↦ ht.preimage (hf y hy)
  /-
    🎉 no goals
  -/


theorem Continuous.comp₂ {g : X × Y → Z} (hg : Continuous g) {e : W → X} (he : Continuous e)
    {f : W → Y} (hf : Continuous f) : Continuous fun w => g (e w, f w) :=
  hg.comp <| he.prod_mk hf


theorem Continuous.comp₃ {g : X × Y × Z → ε} (hg : Continuous g) {e : W → X} (he : Continuous e)
    {f : W → Y} (hf : Continuous f) {k : W → Z} (hk : Continuous k) :
    Continuous fun w => g (e w, f w, k w) :=
  hg.comp₂ he <| hf.prod_mk hk


theorem Continuous.comp₄ {g : X × Y × Z × ζ → ε} (hg : Continuous g) {e : W → X} (he : Continuous e)
    {f : W → Y} (hf : Continuous f) {k : W → Z} (hk : Continuous k) {l : W → ζ}
    (hl : Continuous l) : Continuous fun w => g (e w, f w, k w, l w) :=
  hg.comp₃ he hf <| hk.prod_mk hl


@[continuity]
theorem Continuous.prodMap {f : Z → X} {g : W → Y} (hf : Continuous f) (hg : Continuous g) :
    Continuous (Prod.map f g) :=
  hf.fst'.prod_mk hg.snd'


@[deprecated (since := "2024-10-05")] alias Continuous.prod_map := Continuous.prodMap


/-- A version of `continuous_inf_dom_left` for binary functions -/
theorem continuous_inf_dom_left₂ {X Y Z} {f : X → Y → Z} {ta1 ta2 : TopologicalSpace X}
    {tb1 tb2 : TopologicalSpace Y} {tc1 : TopologicalSpace Z}
            /-
              X✝ : Type u
              Y✝ : Type v
              Z✝ : Type u_1
              W : Type u_2
              ε : Type u_3
              ζ : Type u_4
              inst✝⁵ : TopologicalSpace X✝
              inst✝⁴ : TopologicalSpace Y✝
              inst✝³ : TopologicalSpace Z✝
              inst✝² : TopologicalSpace W
              inst✝¹ : TopologicalSpace ε
              inst✝ : TopologicalSpace ζ
              X : Type ?u.27269
              Y : Type ?u.27275
              Z : Type ?u.27281
              f : X → Y → Z
              ta1 ta2 : TopologicalSpace X
              tb1 tb2 : TopologicalSpace Y
              tc1 : TopologicalSpace Z
              ⊢ Sort ?u.27283
            -/
    (h : by haveI := ta1; haveI := tb1; exact Continuous fun p : X × Y => f p.1 p.2) : by
                                        /-
                                          🎉 no goals
                                        -/
    /-
      X✝ : Type u
      Y✝ : Type v
      Z✝ : Type u_1
      W : Type u_2
      ε : Type u_3
      ζ : Type u_4
      inst✝⁵ : TopologicalSpace X✝
      inst✝⁴ : TopologicalSpace Y✝
      inst✝³ : TopologicalSpace Z✝
      inst✝² : TopologicalSpace W
      inst✝¹ : TopologicalSpace ε
      inst✝ : TopologicalSpace ζ
      X : Type ?u.27269
      Y : Type ?u.27275
      Z : Type ?u.27281
      f : X → Y → Z
      ta1 ta2 : TopologicalSpace X
      tb1 tb2 : TopologicalSpace Y
      tc1 : TopologicalSpace Z
      h : Continuous fun p => f p.1 p.2
      ⊢ Sort ?u.27286
    -/
    haveI := ta1 ⊓ ta2; haveI := tb1 ⊓ tb2; exact Continuous fun p : X × Y => f p.1 p.2 := by
                                            /-
                                              🎉 no goals
                                            -/
  /-
    X : Type u_5
    Y : Type u_6
    Z : Type u_7
    f : X → Y → Z
    ta1 ta2 : TopologicalSpace X
    tb1 tb2 : TopologicalSpace Y
    tc1 : TopologicalSpace Z
    h : Continuous fun p => f p.1 p.2
    ⊢ Continuous fun p => f p.1 p.2
  -/
  have ha := @continuous_inf_dom_left _ _ id ta1 ta2 ta1 (@continuous_id _ (id _))
  /-
    X : Type u_5
    Y : Type u_6
    Z : Type u_7
    f : X → Y → Z
    ta1 ta2 : TopologicalSpace X
    tb1 tb2 : TopologicalSpace Y
    tc1 : TopologicalSpace Z
    h : Continuous fun p => f p.1 p.2
    ha : Continuous id
    ⊢ Continuous fun p => f p.1 p.2
  -/
  have hb := @continuous_inf_dom_left _ _ id tb1 tb2 tb1 (@continuous_id _ (id _))
  /-
    X : Type u_5
    Y : Type u_6
    Z : Type u_7
    f : X → Y → Z
    ta1 ta2 : TopologicalSpace X
    tb1 tb2 : TopologicalSpace Y
    tc1 : TopologicalSpace Z
    h : Continuous fun p => f p.1 p.2
    ha : Continuous id
    hb : Continuous id
    ⊢ Continuous fun p => f p.1 p.2
  -/
  have h_continuous_id := @Continuous.prodMap _ _ _ _ ta1 tb1 (ta1 ⊓ ta2) (tb1 ⊓ tb2) _ _ ha hb
  /-
    X : Type u_5
    Y : Type u_6
    Z : Type u_7
    f : X → Y → Z
    ta1 ta2 : TopologicalSpace X
    tb1 tb2 : TopologicalSpace Y
    tc1 : TopologicalSpace Z
    h : Continuous fun p => f p.1 p.2
    ha : Continuous id
    hb : Continuous id
    h_continuous_id : Continuous (Prod.map id id)
    ⊢ Continuous fun p => f p.1 p.2
  -/
  exact @Continuous.comp _ _ _ (id _) (id _) _ _ _ h h_continuous_id
  /-
    🎉 no goals
  -/


/-- A version of `continuous_inf_dom_right` for binary functions -/
theorem continuous_inf_dom_right₂ {X Y Z} {f : X → Y → Z} {ta1 ta2 : TopologicalSpace X}
    {tb1 tb2 : TopologicalSpace Y} {tc1 : TopologicalSpace Z}
            /-
              X✝ : Type u
              Y✝ : Type v
              Z✝ : Type u_1
              W : Type u_2
              ε : Type u_3
              ζ : Type u_4
              inst✝⁵ : TopologicalSpace X✝
              inst✝⁴ : TopologicalSpace Y✝
              inst✝³ : TopologicalSpace Z✝
              inst✝² : TopologicalSpace W
              inst✝¹ : TopologicalSpace ε
              inst✝ : TopologicalSpace ζ
              X : Type ?u.28020
              Y : Type ?u.28026
              Z : Type ?u.28032
              f : X → Y → Z
              ta1 ta2 : TopologicalSpace X
              tb1 tb2 : TopologicalSpace Y
              tc1 : TopologicalSpace Z
              ⊢ Sort ?u.28034
            -/
    (h : by haveI := ta2; haveI := tb2; exact Continuous fun p : X × Y => f p.1 p.2) : by
                                        /-
                                          🎉 no goals
                                        -/
    /-
      X✝ : Type u
      Y✝ : Type v
      Z✝ : Type u_1
      W : Type u_2
      ε : Type u_3
      ζ : Type u_4
      inst✝⁵ : TopologicalSpace X✝
      inst✝⁴ : TopologicalSpace Y✝
      inst✝³ : TopologicalSpace Z✝
      inst✝² : TopologicalSpace W
      inst✝¹ : TopologicalSpace ε
      inst✝ : TopologicalSpace ζ
      X : Type ?u.28020
      Y : Type ?u.28026
      Z : Type ?u.28032
      f : X → Y → Z
      ta1 ta2 : TopologicalSpace X
      tb1 tb2 : TopologicalSpace Y
      tc1 : TopologicalSpace Z
      h : Continuous fun p => f p.1 p.2
      ⊢ Sort ?u.28037
    -/
    haveI := ta1 ⊓ ta2; haveI := tb1 ⊓ tb2; exact Continuous fun p : X × Y => f p.1 p.2 := by
                                            /-
                                              🎉 no goals
                                            -/
  /-
    X : Type u_5
    Y : Type u_6
    Z : Type u_7
    f : X → Y → Z
    ta1 ta2 : TopologicalSpace X
    tb1 tb2 : TopologicalSpace Y
    tc1 : TopologicalSpace Z
    h : Continuous fun p => f p.1 p.2
    ⊢ Continuous fun p => f p.1 p.2
  -/
  have ha := @continuous_inf_dom_right _ _ id ta1 ta2 ta2 (@continuous_id _ (id _))
  /-
    X : Type u_5
    Y : Type u_6
    Z : Type u_7
    f : X → Y → Z
    ta1 ta2 : TopologicalSpace X
    tb1 tb2 : TopologicalSpace Y
    tc1 : TopologicalSpace Z
    h : Continuous fun p => f p.1 p.2
    ha : Continuous id
    ⊢ Continuous fun p => f p.1 p.2
  -/
  have hb := @continuous_inf_dom_right _ _ id tb1 tb2 tb2 (@continuous_id _ (id _))
  /-
    X : Type u_5
    Y : Type u_6
    Z : Type u_7
    f : X → Y → Z
    ta1 ta2 : TopologicalSpace X
    tb1 tb2 : TopologicalSpace Y
    tc1 : TopologicalSpace Z
    h : Continuous fun p => f p.1 p.2
    ha : Continuous id
    hb : Continuous id
    ⊢ Continuous fun p => f p.1 p.2
  -/
  have h_continuous_id := @Continuous.prodMap _ _ _ _ ta2 tb2 (ta1 ⊓ ta2) (tb1 ⊓ tb2) _ _ ha hb
  /-
    X : Type u_5
    Y : Type u_6
    Z : Type u_7
    f : X → Y → Z
    ta1 ta2 : TopologicalSpace X
    tb1 tb2 : TopologicalSpace Y
    tc1 : TopologicalSpace Z
    h : Continuous fun p => f p.1 p.2
    ha : Continuous id
    hb : Continuous id
    h_continuous_id : Continuous (Prod.map id id)
    ⊢ Continuous fun p => f p.1 p.2
  -/
  exact @Continuous.comp _ _ _ (id _) (id _) _ _ _ h h_continuous_id
  /-
    🎉 no goals
  -/


/-- A version of `continuous_sInf_dom` for binary functions -/
theorem continuous_sInf_dom₂ {X Y Z} {f : X → Y → Z} {tas : Set (TopologicalSpace X)}
    {tbs : Set (TopologicalSpace Y)} {tX : TopologicalSpace X} {tY : TopologicalSpace Y}
    {tc : TopologicalSpace Z} (hX : tX ∈ tas) (hY : tY ∈ tbs)
    (hf : Continuous fun p : X × Y => f p.1 p.2) : by
    /-
      X✝ : Type u
      Y✝ : Type v
      Z✝ : Type u_1
      W : Type u_2
      ε : Type u_3
      ζ : Type u_4
      inst✝⁵ : TopologicalSpace X✝
      inst✝⁴ : TopologicalSpace Y✝
      inst✝³ : TopologicalSpace Z✝
      inst✝² : TopologicalSpace W
      inst✝¹ : TopologicalSpace ε
      inst✝ : TopologicalSpace ζ
      X : Type ?u.28772
      Y : Type ?u.28776
      Z : Type ?u.28785
      f : X → Y → Z
      tas : Set (TopologicalSpace X)
      tbs : Set (TopologicalSpace Y)
      tX : TopologicalSpace X
      tY : TopologicalSpace Y
      tc : TopologicalSpace Z
      hX : Membership.mem tas tX
      hY : Membership.mem tbs tY
      hf : Continuous fun p => f p.1 p.2
      ⊢ Sort ?u.28968
    -/
    haveI := sInf tas; haveI := sInf tbs
    /-
      X✝ : Type u
      Y✝ : Type v
      Z✝ : Type u_1
      W : Type u_2
      ε : Type u_3
      ζ : Type u_4
      inst✝⁵ : TopologicalSpace X✝
      inst✝⁴ : TopologicalSpace Y✝
      inst✝³ : TopologicalSpace Z✝
      inst✝² : TopologicalSpace W
      inst✝¹ : TopologicalSpace ε
      inst✝ : TopologicalSpace ζ
      X : Type ?u.28772
      Y : Type ?u.28776
      Z : Type ?u.28785
      f : X → Y → Z
      tas : Set (TopologicalSpace X)
      tbs : Set (TopologicalSpace Y)
      tX : TopologicalSpace X
      tY : TopologicalSpace Y
      tc : TopologicalSpace Z
      hX : Membership.mem tas tX
      hY : Membership.mem tbs tY
      hf : Continuous fun p => f p.1 p.2
      this✝ : TopologicalSpace X
      this : TopologicalSpace Y
      ⊢ Sort ?u.28968
    -/
    exact @Continuous _ _ _ tc fun p : X × Y => f p.1 p.2 := by
    /-
      🎉 no goals
    -/
  /-
    X : Type u_5
    Y : Type u_6
    Z : Type u_7
    f : X → Y → Z
    tas : Set (TopologicalSpace X)
    tbs : Set (TopologicalSpace Y)
    tX : TopologicalSpace X
    tY : TopologicalSpace Y
    tc : TopologicalSpace Z
    hX : Membership.mem tas tX
    hY : Membership.mem tbs tY
    hf : Continuous fun p => f p.1 p.2
    ⊢ Continuous fun p => f p.1 p.2
  -/
  have hX := continuous_sInf_dom hX continuous_id
  /-
    X : Type u_5
    Y : Type u_6
    Z : Type u_7
    f : X → Y → Z
    tas : Set (TopologicalSpace X)
    tbs : Set (TopologicalSpace Y)
    tX : TopologicalSpace X
    tY : TopologicalSpace Y
    tc : TopologicalSpace Z
    hX✝ : Membership.mem tas tX
    hY : Membership.mem tbs tY
    hf : Continuous fun p => f p.1 p.2
    hX : Continuous id
    ⊢ Continuous fun p => f p.1 p.2
  -/
  have hY := continuous_sInf_dom hY continuous_id
  /-
    X : Type u_5
    Y : Type u_6
    Z : Type u_7
    f : X → Y → Z
    tas : Set (TopologicalSpace X)
    tbs : Set (TopologicalSpace Y)
    tX : TopologicalSpace X
    tY : TopologicalSpace Y
    tc : TopologicalSpace Z
    hX✝ : Membership.mem tas tX
    hY✝ : Membership.mem tbs tY
    hf : Continuous fun p => f p.1 p.2
    hX : Continuous id
    hY : Continuous id
    ⊢ Continuous fun p => f p.1 p.2
  -/
  have h_continuous_id := @Continuous.prodMap _ _ _ _ tX tY (sInf tas) (sInf tbs) _ _ hX hY
  /-
    X : Type u_5
    Y : Type u_6
    Z : Type u_7
    f : X → Y → Z
    tas : Set (TopologicalSpace X)
    tbs : Set (TopologicalSpace Y)
    tX : TopologicalSpace X
    tY : TopologicalSpace Y
    tc : TopologicalSpace Z
    hX✝ : Membership.mem tas tX
    hY✝ : Membership.mem tbs tY
    hf : Continuous fun p => f p.1 p.2
    hX : Continuous id
    hY : Continuous id
    h_continuous_id : Continuous (Prod.map id id)
    ⊢ Continuous fun p => f p.1 p.2
  -/
  exact @Continuous.comp _ _ _ (id _) (id _) _ _ _ hf h_continuous_id
  /-
    🎉 no goals
  -/


theorem Filter.Eventually.prod_inl_nhds {p : X → Prop} {x : X} (h : ∀ᶠ x in 𝓝 x, p x) (y : Y) :
    ∀ᶠ x in 𝓝 (x, y), p (x : X × Y).1 :=
  continuousAt_fst h


theorem Filter.Eventually.prod_inr_nhds {p : Y → Prop} {y : Y} (h : ∀ᶠ x in 𝓝 y, p x) (x : X) :
    ∀ᶠ x in 𝓝 (x, y), p (x : X × Y).2 :=
  continuousAt_snd h


theorem Filter.Eventually.prod_mk_nhds {px : X → Prop} {x} (hx : ∀ᶠ x in 𝓝 x, px x) {py : Y → Prop}
    {y} (hy : ∀ᶠ y in 𝓝 y, py y) : ∀ᶠ p in 𝓝 (x, y), px (p : X × Y).1 ∧ py p.2 :=
  (hx.prod_inl_nhds y).and (hy.prod_inr_nhds x)


theorem continuous_swap : Continuous (Prod.swap : X × Y → Y × X) :=
  continuous_snd.prod_mk continuous_fst


lemma isClosedMap_swap : IsClosedMap (Prod.swap : X × Y → Y × X) := fun s hs ↦ by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set (Prod X Y)
    hs : IsClosed s
    ⊢ IsClosed (Set.image Prod.swap s)
  -/
  rw [image_swap_eq_preimage_swap]
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set (Prod X Y)
    hs : IsClosed s
    ⊢ IsClosed (Set.preimage Prod.swap s)
  -/
  exact hs.preimage continuous_swap
  /-
    🎉 no goals
  -/


theorem Continuous.uncurry_left {f : X → Y → Z} (x : X) (h : Continuous (uncurry f)) :
    Continuous (f x) :=
  h.comp (Continuous.Prod.mk _)


theorem Continuous.uncurry_right {f : X → Y → Z} (y : Y) (h : Continuous (uncurry f)) :
    Continuous fun a => f a y :=
  h.comp (Continuous.Prod.mk_left _)


@[deprecated (since := "2024-03-09")] alias continuous_uncurry_left := Continuous.uncurry_left

@[deprecated (since := "2024-03-09")] alias continuous_uncurry_right := Continuous.uncurry_right


theorem continuous_curry {g : X × Y → Z} (x : X) (h : Continuous g) : Continuous (curry g x) :=
  Continuous.uncurry_left x h


theorem IsOpen.prod {s : Set X} {t : Set Y} (hs : IsOpen s) (ht : IsOpen t) : IsOpen (s ×ˢ t) :=
  (hs.preimage continuous_fst).inter (ht.preimage continuous_snd)

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: Lean fails to find `t₁` and `t₂` by unification

theorem nhds_prod_eq {x : X} {y : Y} : 𝓝 (x, y) = 𝓝 x ×ˢ 𝓝 y := by
  rw [prod_eq_inf, instTopologicalSpaceProd, nhds_inf (t₁ := TopologicalSpace.induced Prod.fst _)
    (t₂ := TopologicalSpace.induced Prod.snd _), nhds_induced, nhds_induced]


theorem nhdsWithin_prod_eq (x : X) (y : Y) (s : Set X) (t : Set Y) :
    𝓝[s ×ˢ t] (x, y) = 𝓝[s] x ×ˢ 𝓝[t] y := by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    x : X
    y : Y
    s : Set X
    t : Set Y
    ⊢ Eq (nhdsWithin { fst := x, snd := y } (SProd.sprod s t)) (SProd.sprod (nhdsW …
  -/
  simp only [nhdsWithin, nhds_prod_eq, ← prod_inf_prod, prod_principal_principal]
  /-
    🎉 no goals
  -/


instance Prod.instNeBotNhdsWithinIio [Preorder X] [Preorder Y] {x : X × Y}
    [hx₁ : (𝓝[<] x.1).NeBot] [hx₂ : (𝓝[<] x.2).NeBot] : (𝓝[<] x).NeBot := by
  /-
    X : Type u
    Y : Type v
    Z : Type u_1
    W : Type u_2
    ε : Type u_3
    ζ : Type u_4
    inst✝⁷ : TopologicalSpace X
    inst✝⁶ : TopologicalSpace Y
    inst✝⁵ : TopologicalSpace Z
    inst✝⁴ : TopologicalSpace W
    inst✝³ : TopologicalSpace ε
    inst✝² : TopologicalSpace ζ
    inst✝¹ : Preorder X
    inst✝ : Preorder Y
    x : Prod X Y
    hx₁ : (nhdsWithin x.1 (Set.Iio x.1)).NeBot
    hx₂ : (nhdsWithin x.2 (Set.Iio x.2)).NeBot
    ⊢ (nhdsWithin x (Set.Iio x)).NeBot
  -/
  refine (hx₁.prod hx₂).mono ?_
  /-
    X : Type u
    Y : Type v
    Z : Type u_1
    W : Type u_2
    ε : Type u_3
    ζ : Type u_4
    inst✝⁷ : TopologicalSpace X
    inst✝⁶ : TopologicalSpace Y
    inst✝⁵ : TopologicalSpace Z
    inst✝⁴ : TopologicalSpace W
    inst✝³ : TopologicalSpace ε
    inst✝² : TopologicalSpace ζ
    inst✝¹ : Preorder X
    inst✝ : Preorder Y
    x : Prod X Y
    hx₁ : (nhdsWithin x.1 (Set.Iio x.1)).NeBot
    hx₂ : (nhdsWithin x.2 (Set.Iio x.2)).NeBot
    ⊢ LE.le (SProd.sprod (nhdsWithin x.1 (Set.Iio x.1)) (nhdsWithin x.2 (Set.Iio x …
  -/
  rw [← nhdsWithin_prod_eq]
  /-
    X : Type u
    Y : Type v
    Z : Type u_1
    W : Type u_2
    ε : Type u_3
    ζ : Type u_4
    inst✝⁷ : TopologicalSpace X
    inst✝⁶ : TopologicalSpace Y
    inst✝⁵ : TopologicalSpace Z
    inst✝⁴ : TopologicalSpace W
    inst✝³ : TopologicalSpace ε
    inst✝² : TopologicalSpace ζ
    inst✝¹ : Preorder X
    inst✝ : Preorder Y
    x : Prod X Y
    hx₁ : (nhdsWithin x.1 (Set.Iio x.1)).NeBot
    hx₂ : (nhdsWithin x.2 (Set.Iio x.2)).NeBot
    ⊢ LE.le (nhdsWithin { fst := x.1, snd := x.2 } (SProd.sprod (Set.Iio x.1) (Set …
  -/
  exact nhdsWithin_mono _ fun _ ⟨h₁, h₂⟩ ↦ Prod.lt_iff.2 <| .inl ⟨h₁, h₂.le⟩
  /-
    🎉 no goals
  -/


instance Prod.instNeBotNhdsWithinIoi [Preorder X] [Preorder Y] {x : X × Y}
    [(𝓝[>] x.1).NeBot] [(𝓝[>] x.2).NeBot] : (𝓝[>] x).NeBot :=
  Prod.instNeBotNhdsWithinIio (X := Xᵒᵈ) (Y := Yᵒᵈ)
    (x := (OrderDual.toDual x.1, OrderDual.toDual x.2))


theorem mem_nhds_prod_iff {x : X} {y : Y} {s : Set (X × Y)} :
                                                          /-
                                                            X : Type u
                                                            Y : Type v
                                                            inst✝¹ : TopologicalSpace X
                                                            inst✝ : TopologicalSpace Y
                                                            x : X
                                                            y : Y
                                                            s : Set (Prod X Y)
                                                            ⊢ Iff (Membership.mem (nhds { fst := x, snd := y }) s) (Exists fun u => And (M …
                                                          -/
    s ∈ 𝓝 (x, y) ↔ ∃ u ∈ 𝓝 x, ∃ v ∈ 𝓝 y, u ×ˢ v ⊆ s := by rw [nhds_prod_eq, mem_prod_iff]
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem mem_nhdsWithin_prod_iff {x : X} {y : Y} {s : Set (X × Y)} {tx : Set X} {ty : Set Y} :
    s ∈ 𝓝[tx ×ˢ ty] (x, y) ↔ ∃ u ∈ 𝓝[tx] x, ∃ v ∈ 𝓝[ty] y, u ×ˢ v ⊆ s := by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    x : X
    y : Y
    s : Set (Prod X Y)
    tx : Set X
    ty : Set Y
    ⊢ Iff (Membership.mem (nhdsWithin { fst := x, snd := y } (SProd.sprod tx ty))  …
  -/
  rw [nhdsWithin_prod_eq, mem_prod_iff]
  /-
    🎉 no goals
  -/


theorem Filter.HasBasis.prod_nhds {ιX ιY : Type*} {px : ιX → Prop} {py : ιY → Prop}
    {sx : ιX → Set X} {sy : ιY → Set Y} {x : X} {y : Y} (hx : (𝓝 x).HasBasis px sx)
    (hy : (𝓝 y).HasBasis py sy) :
    (𝓝 (x, y)).HasBasis (fun i : ιX × ιY => px i.1 ∧ py i.2) fun i => sx i.1 ×ˢ sy i.2 := by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    ιX : Type u_5
    ιY : Type u_6
    px : ιX → Prop
    py : ιY → Prop
    sx : ιX → Set X
    sy : ιY → Set Y
    x : X
    y : Y
    hx : (nhds x).HasBasis px sx
    hy : (nhds y).HasBasis py sy
    ⊢ (nhds { fst := x, snd := y }).HasBasis (fun i => And (px i.1) (py i.2)) fun  …
  -/
  rw [nhds_prod_eq]
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    ιX : Type u_5
    ιY : Type u_6
    px : ιX → Prop
    py : ιY → Prop
    sx : ιX → Set X
    sy : ιY → Set Y
    x : X
    y : Y
    hx : (nhds x).HasBasis px sx
    hy : (nhds y).HasBasis py sy
    ⊢ (SProd.sprod (nhds x) (nhds y)).HasBasis (fun i => And (px i.1) (py i.2)) fu …
  -/
  exact hx.prod hy
  /-
    🎉 no goals
  -/


theorem Filter.HasBasis.prod_nhds' {ιX ιY : Type*} {pX : ιX → Prop} {pY : ιY → Prop}
    {sx : ιX → Set X} {sy : ιY → Set Y} {p : X × Y} (hx : (𝓝 p.1).HasBasis pX sx)
    (hy : (𝓝 p.2).HasBasis pY sy) :
    (𝓝 p).HasBasis (fun i : ιX × ιY => pX i.1 ∧ pY i.2) fun i => sx i.1 ×ˢ sy i.2 :=
  hx.prod_nhds hy


theorem MapClusterPt.curry_prodMap {α β : Type*}
    {f : α → X} {g : β → Y} {la : Filter α} {lb : Filter β} {x : X} {y : Y}
    (hf : MapClusterPt x la f) (hg : MapClusterPt y lb g) :
    MapClusterPt (x, y) (la.curry lb) (.map f g) := by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    α : Type u_5
    β : Type u_6
    f : α → X
    g : β → Y
    la : Filter α
    lb : Filter β
    x : X
    y : Y
    hf : MapClusterPt x la f
    hg : MapClusterPt y lb g
    ⊢ MapClusterPt { fst := x, snd := y } (la.curry lb) (Prod.map f g)
  -/
  rw [mapClusterPt_iff] at hf hg
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    α : Type u_5
    β : Type u_6
    f : α → X
    g : β → Y
    la : Filter α
    lb : Filter β
    x : X
    y : Y
    hf : ∀ (s : Set X), Membership.mem (nhds x) s → Filter.Frequently (fun a => Me …
    hg : ∀ (s : Set Y), Membership.mem (nhds y) s → Filter.Frequently (fun a => Me …
    ⊢ MapClusterPt { fst := x, snd := y } (la.curry lb) (Prod.map f g)
  -/
  rw [((𝓝 x).basis_sets.prod_nhds (𝓝 y).basis_sets).mapClusterPt_iff_frequently]
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    α : Type u_5
    β : Type u_6
    f : α → X
    g : β → Y
    la : Filter α
    lb : Filter β
    x : X
    y : Y
    hf : ∀ (s : Set X), Membership.mem (nhds x) s → Filter.Frequently (fun a => Me …
    hg : ∀ (s : Set Y), Membership.mem (nhds y) s → Filter.Frequently (fun a => Me …
    ⊢ ∀ (i : Prod (Set X) (Set Y)), And (Membership.mem (nhds x) i.1) (Membership. …
  -/
  rintro ⟨s, t⟩ ⟨hs, ht⟩
  /-
    case mk.intro
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    α : Type u_5
    β : Type u_6
    f : α → X
    g : β → Y
    la : Filter α
    lb : Filter β
    x : X
    y : Y
    hf : ∀ (s : Set X), Membership.mem (nhds x) s → Filter.Frequently (fun a => Me …
    hg : ∀ (s : Set Y), Membership.mem (nhds y) s → Filter.Frequently (fun a => Me …
    s : Set X
    t : Set Y
    hs : Membership.mem (nhds x) { fst := s, snd := t }.1
    ht : Membership.mem (nhds y) { fst := s, snd := t }.2
    ⊢ Filter.Frequently (fun a => Membership.mem (SProd.sprod (id { fst := s, snd  …
  -/
  rw [frequently_curry_iff]
  /-
    case mk.intro
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    α : Type u_5
    β : Type u_6
    f : α → X
    g : β → Y
    la : Filter α
    lb : Filter β
    x : X
    y : Y
    hf : ∀ (s : Set X), Membership.mem (nhds x) s → Filter.Frequently (fun a => Me …
    hg : ∀ (s : Set Y), Membership.mem (nhds y) s → Filter.Frequently (fun a => Me …
    s : Set X
    t : Set Y
    hs : Membership.mem (nhds x) { fst := s, snd := t }.1
    ht : Membership.mem (nhds y) { fst := s, snd := t }.2
    ⊢ Filter.Frequently (fun x => Filter.Frequently (fun y => Membership.mem (SPro …
  -/
  exact (hf s hs).mono fun x hx ↦ (hg t ht).mono fun y hy ↦ ⟨hx, hy⟩
  /-
    🎉 no goals
  -/


theorem MapClusterPt.prodMap {α β : Type*}
    {f : α → X} {g : β → Y} {la : Filter α} {lb : Filter β} {x : X} {y : Y}
    (hf : MapClusterPt x la f) (hg : MapClusterPt y lb g) :
    MapClusterPt (x, y) (la ×ˢ lb) (.map f g) :=
  (hf.curry_prodMap hg).mono <| map_mono curry_le_prod


theorem mem_nhds_prod_iff' {x : X} {y : Y} {s : Set (X × Y)} :
    s ∈ 𝓝 (x, y) ↔ ∃ u v, IsOpen u ∧ x ∈ u ∧ IsOpen v ∧ y ∈ v ∧ u ×ˢ v ⊆ s :=
  ((nhds_basis_opens x).prod_nhds (nhds_basis_opens y)).mem_iff.trans <| by
    /-
      X : Type u
      Y : Type v
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      x : X
      y : Y
      s : Set (Prod X Y)
      ⊢ Iff (Exists fun i => And (And (And (Membership.mem i.1 x) (IsOpen i.1)) (And …
    -/
    simp only [Prod.exists, and_comm, and_assoc, and_left_comm]
    /-
      🎉 no goals
    -/


theorem Prod.tendsto_iff {X} (seq : X → Y × Z) {f : Filter X} (p : Y × Z) :
    Tendsto seq f (𝓝 p) ↔
      Tendsto (fun n => (seq n).fst) f (𝓝 p.fst) ∧ Tendsto (fun n => (seq n).snd) f (𝓝 p.snd) := by
  /-
    Y : Type v
    Z : Type u_1
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    X : Type u_5
    seq : X → Prod Y Z
    f : Filter X
    p : Prod Y Z
    ⊢ Iff (Filter.Tendsto seq f (nhds p)) (And (Filter.Tendsto (fun n => (seq n).1 …
  -/
  rw [nhds_prod_eq, Filter.tendsto_prod_iff']
  /-
    🎉 no goals
  -/


instance [DiscreteTopology X] [DiscreteTopology Y] : DiscreteTopology (X × Y) :=
  discreteTopology_iff_nhds.2 fun (a, b) => by
    /-
      X : Type u
      Y : Type v
      Z : Type u_1
      W : Type u_2
      ε : Type u_3
      ζ : Type u_4
      inst✝⁷ : TopologicalSpace X
      inst✝⁶ : TopologicalSpace Y
      inst✝⁵ : TopologicalSpace Z
      inst✝⁴ : TopologicalSpace W
      inst✝³ : TopologicalSpace ε
      inst✝² : TopologicalSpace ζ
      inst✝¹ : DiscreteTopology X
      inst✝ : DiscreteTopology Y
      x✝ : Prod X Y
      a : X
      b : Y
      ⊢ Eq (nhds { fst := a, snd := b }) (Pure.pure { fst := a, snd := b })
    -/
    rw [nhds_prod_eq, nhds_discrete X, nhds_discrete Y, prod_pure_pure]
    /-
      🎉 no goals
    -/


theorem prod_mem_nhds_iff {s : Set X} {t : Set Y} {x : X} {y : Y} :
                                                /-
                                                  X : Type u
                                                  Y : Type v
                                                  inst✝¹ : TopologicalSpace X
                                                  inst✝ : TopologicalSpace Y
                                                  s : Set X
                                                  t : Set Y
                                                  x : X
                                                  y : Y
                                                  ⊢ Iff (Membership.mem (nhds { fst := x, snd := y }) (SProd.sprod s t)) (And (M …
                                                -/
    s ×ˢ t ∈ 𝓝 (x, y) ↔ s ∈ 𝓝 x ∧ t ∈ 𝓝 y := by rw [nhds_prod_eq, prod_mem_prod_iff]
                                                /-
                                                  🎉 no goals
                                                -/


theorem prod_mem_nhds {s : Set X} {t : Set Y} {x : X} {y : Y} (hx : s ∈ 𝓝 x) (hy : t ∈ 𝓝 y) :
    s ×ˢ t ∈ 𝓝 (x, y) :=
  prod_mem_nhds_iff.2 ⟨hx, hy⟩


theorem isOpen_setOf_disjoint_nhds_nhds : IsOpen { p : X × X | Disjoint (𝓝 p.1) (𝓝 p.2) } := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    ⊢ IsOpen (setOf fun p => Disjoint (nhds p.1) (nhds p.2))
  -/
  simp only [isOpen_iff_mem_nhds, Prod.forall, mem_setOf_eq]
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    ⊢ ∀ (a b : X), Disjoint (nhds a) (nhds b) → Membership.mem (nhds { fst := a, s …
  -/
  intro x y h
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    x y : X
    h : Disjoint (nhds x) (nhds y)
    ⊢ Membership.mem (nhds { fst := x, snd := y }) (setOf fun p => Disjoint (nhds  …
  -/
  obtain ⟨U, hU, V, hV, hd⟩ := ((nhds_basis_opens x).disjoint_iff (nhds_basis_opens y)).mp h
  exact mem_nhds_prod_iff'.mpr ⟨U, V, hU.2, hU.1, hV.2, hV.1, fun ⟨x', y'⟩ ⟨hx', hy'⟩ =>
    disjoint_of_disjoint_of_mem hd (hU.2.mem_nhds hx') (hV.2.mem_nhds hy')⟩


theorem Filter.Eventually.prod_nhds {p : X → Prop} {q : Y → Prop} {x : X} {y : Y}
    (hx : ∀ᶠ x in 𝓝 x, p x) (hy : ∀ᶠ y in 𝓝 y, q y) : ∀ᶠ z : X × Y in 𝓝 (x, y), p z.1 ∧ q z.2 :=
  prod_mem_nhds hx hy


theorem nhds_swap (x : X) (y : Y) : 𝓝 (x, y) = (𝓝 (y, x)).map Prod.swap := by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    x : X
    y : Y
    ⊢ Eq (nhds { fst := x, snd := y }) (Filter.map Prod.swap (nhds { fst := y, snd …
  -/
  rw [nhds_prod_eq, Filter.prod_comm, nhds_prod_eq]; rfl
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem Filter.Tendsto.prod_mk_nhds {γ} {x : X} {y : Y} {f : Filter γ} {mx : γ → X} {my : γ → Y}
    (hx : Tendsto mx f (𝓝 x)) (hy : Tendsto my f (𝓝 y)) :
    Tendsto (fun c => (mx c, my c)) f (𝓝 (x, y)) := by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    γ : Type u_5
    x : X
    y : Y
    f : Filter γ
    mx : γ → X
    my : γ → Y
    hx : Filter.Tendsto mx f (nhds x)
    hy : Filter.Tendsto my f (nhds y)
    ⊢ Filter.Tendsto (fun c => { fst := mx c, snd := my c }) f (nhds { fst := x, s …
  -/
  rw [nhds_prod_eq]; exact Filter.Tendsto.prod_mk hx hy
                     /-
                       🎉 no goals
                     -/


theorem Filter.Eventually.curry_nhds {p : X × Y → Prop} {x : X} {y : Y}
    (h : ∀ᶠ x in 𝓝 (x, y), p x) : ∀ᶠ x' in 𝓝 x, ∀ᶠ y' in 𝓝 y, p (x', y') := by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    p : Prod X Y → Prop
    x : X
    y : Y
    h : Filter.Eventually (fun x => p x) (nhds { fst := x, snd := y })
    ⊢ Filter.Eventually (fun x' => Filter.Eventually (fun y' => p { fst := x', snd …
  -/
  rw [nhds_prod_eq] at h
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    p : Prod X Y → Prop
    x : X
    y : Y
    h : Filter.Eventually (fun x => p x) (SProd.sprod (nhds x) (nhds y))
    ⊢ Filter.Eventually (fun x' => Filter.Eventually (fun y' => p { fst := x', snd …
  -/
  exact h.curry
  /-
    🎉 no goals
  -/


@[fun_prop]
theorem ContinuousAt.prod {f : X → Y} {g : X → Z} {x : X} (hf : ContinuousAt f x)
    (hg : ContinuousAt g x) : ContinuousAt (fun x => (f x, g x)) x :=
  hf.prod_mk_nhds hg


theorem ContinuousAt.prodMap {f : X → Z} {g : Y → W} {p : X × Y} (hf : ContinuousAt f p.fst)
    (hg : ContinuousAt g p.snd) : ContinuousAt (Prod.map f g) p :=
  hf.fst''.prod hg.snd''


@[deprecated (since := "2024-10-05")] alias ContinuousAt.prod_map := ContinuousAt.prodMap


/-- A version of `ContinuousAt.prodMap` that avoids `Prod.fst`/`Prod.snd`
by assuming that the point is `(x, y)`. -/
theorem ContinuousAt.prodMap' {f : X → Z} {g : Y → W} {x : X} {y : Y} (hf : ContinuousAt f x)
    (hg : ContinuousAt g y) : ContinuousAt (Prod.map f g) (x, y) :=
  hf.prodMap hg


@[deprecated (since := "2024-10-05")] alias ContinuousAt.prod_map' := ContinuousAt.prodMap'


theorem ContinuousAt.comp₂ {f : Y × Z → W} {g : X → Y} {h : X → Z} {x : X}
    (hf : ContinuousAt f (g x, h x)) (hg : ContinuousAt g x) (hh : ContinuousAt h x) :
    ContinuousAt (fun x ↦ f (g x, h x)) x :=
  ContinuousAt.comp hf (hg.prod hh)


theorem ContinuousAt.comp₂_of_eq {f : Y × Z → W} {g : X → Y} {h : X → Z} {x : X} {y : Y × Z}
    (hf : ContinuousAt f y) (hg : ContinuousAt g x) (hh : ContinuousAt h x) (e : (g x, h x) = y) :
    ContinuousAt (fun x ↦ f (g x, h x)) x := by
  /-
    X : Type u
    Y : Type v
    Z : Type u_1
    W : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : TopologicalSpace Z
    inst✝ : TopologicalSpace W
    f : Prod Y Z → W
    g : X → Y
    h : X → Z
    x : X
    y : Prod Y Z
    hf : ContinuousAt f y
    hg : ContinuousAt g x
    hh : ContinuousAt h x
    e : Eq { fst := g x, snd := h x } y
    ⊢ ContinuousAt (fun x => f { fst := g x, snd := h x }) x
  -/
  rw [← e] at hf
  /-
    X : Type u
    Y : Type v
    Z : Type u_1
    W : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : TopologicalSpace Z
    inst✝ : TopologicalSpace W
    f : Prod Y Z → W
    g : X → Y
    h : X → Z
    x : X
    y : Prod Y Z
    hf : ContinuousAt f { fst := g x, snd := h x }
    hg : ContinuousAt g x
    hh : ContinuousAt h x
    e : Eq { fst := g x, snd := h x } y
    ⊢ ContinuousAt (fun x => f { fst := g x, snd := h x }) x
  -/
  exact hf.comp₂ hg hh
  /-
    🎉 no goals
  -/


/-- Continuous functions on products are continuous in their first argument -/
theorem Continuous.curry_left {f : X × Y → Z} (hf : Continuous f) {y : Y} :
    Continuous fun x ↦ f (x, y) :=
  hf.comp (continuous_id.prod_mk continuous_const)

alias Continuous.along_fst := Continuous.curry_left


/-- Continuous functions on products are continuous in their second argument -/
theorem Continuous.curry_right {f : X × Y → Z} (hf : Continuous f) {x : X} :
    Continuous fun y ↦ f (x, y) :=
  hf.comp (continuous_const.prod_mk continuous_id)

alias Continuous.along_snd := Continuous.curry_right

-- todo: prove a version of `generateFrom_union` with `image2 (∩) s t` in the LHS and use it here

theorem prod_generateFrom_generateFrom_eq {X Y : Type*} {s : Set (Set X)} {t : Set (Set Y)}
    (hs : ⋃₀ s = univ) (ht : ⋃₀ t = univ) :
    @instTopologicalSpaceProd X Y (generateFrom s) (generateFrom t) =
      generateFrom (image2 (·  ×ˢ ·) s t) :=
  let G := generateFrom (image2  (·  ×ˢ ·) s t)
  le_antisymm
    (le_generateFrom fun _ ⟨_, hu, _, hv, g_eq⟩ =>
      g_eq.symm ▸
        @IsOpen.prod _ _ (generateFrom s) (generateFrom t) _ _ (GenerateOpen.basic _ hu)
          (GenerateOpen.basic _ hv))
    (le_inf
      (coinduced_le_iff_le_induced.mp <|
        le_generateFrom fun u hu =>
          have : ⋃ v ∈ t, u ×ˢ v = Prod.fst ⁻¹' u := by
            /-
              X : Type u_5
              Y : Type u_6
              s : Set (Set X)
              t : Set (Set Y)
              hs : Eq s.sUnion Set.univ
              ht : Eq t.sUnion Set.univ
              G : TopologicalSpace (Prod X Y) := TopologicalSpace.generateFrom (Set.image2 ( …
              u : Set X
              hu : Membership.mem s u
              ⊢ Eq (Set.iUnion fun v => Set.iUnion fun h => SProd.sprod u v) (Set.preimage P …
            -/
            simp_rw [← prod_iUnion, ← sUnion_eq_biUnion, ht, prod_univ]
            /-
              🎉 no goals
            -/
          show G.IsOpen (Prod.fst ⁻¹' u) by
            /-
              X : Type u_5
              Y : Type u_6
              s : Set (Set X)
              t : Set (Set Y)
              hs : Eq s.sUnion Set.univ
              ht : Eq t.sUnion Set.univ
              G : TopologicalSpace (Prod X Y) := TopologicalSpace.generateFrom (Set.image2 ( …
              u : Set X
              hu : Membership.mem s u
              this : Eq (Set.iUnion fun v => Set.iUnion fun h => SProd.sprod u v) (Set.preim …
              ⊢ TopologicalSpace.IsOpen (Set.preimage Prod.fst u)
            -/
            rw [← this]
            exact
              isOpen_iUnion fun v =>
                isOpen_iUnion fun hv => GenerateOpen.basic _ ⟨_, hu, _, hv, rfl⟩)
      (coinduced_le_iff_le_induced.mp <|
        le_generateFrom fun v hv =>
          have : ⋃ u ∈ s, u ×ˢ v = Prod.snd ⁻¹' v := by
            /-
              X : Type u_5
              Y : Type u_6
              s : Set (Set X)
              t : Set (Set Y)
              hs : Eq s.sUnion Set.univ
              ht : Eq t.sUnion Set.univ
              G : TopologicalSpace (Prod X Y) := TopologicalSpace.generateFrom (Set.image2 ( …
              v : Set Y
              hv : Membership.mem t v
              ⊢ Eq (Set.iUnion fun u => Set.iUnion fun h => SProd.sprod u v) (Set.preimage P …
            -/
            simp_rw [← iUnion_prod_const, ← sUnion_eq_biUnion, hs, univ_prod]
            /-
              🎉 no goals
            -/
          show G.IsOpen (Prod.snd ⁻¹' v) by
            /-
              X : Type u_5
              Y : Type u_6
              s : Set (Set X)
              t : Set (Set Y)
              hs : Eq s.sUnion Set.univ
              ht : Eq t.sUnion Set.univ
              G : TopologicalSpace (Prod X Y) := TopologicalSpace.generateFrom (Set.image2 ( …
              v : Set Y
              hv : Membership.mem t v
              this : Eq (Set.iUnion fun u => Set.iUnion fun h => SProd.sprod u v) (Set.preim …
              ⊢ TopologicalSpace.IsOpen (Set.preimage Prod.snd v)
            -/
            rw [← this]
            exact
              isOpen_iUnion fun u =>
                isOpen_iUnion fun hu => GenerateOpen.basic _ ⟨_, hu, _, hv, rfl⟩))

-- todo: use the previous lemma?

theorem prod_eq_generateFrom :
    instTopologicalSpaceProd =
      generateFrom { g | ∃ (s : Set X) (t : Set Y), IsOpen s ∧ IsOpen t ∧ g = s ×ˢ t } :=
  le_antisymm (le_generateFrom fun _ ⟨_, _, hs, ht, g_eq⟩ => g_eq.symm ▸ hs.prod ht)
    (le_inf
      (forall_mem_image.2 fun t ht =>
                                          /-
                                            X : Type u
                                            Y : Type v
                                            inst✝¹ : TopologicalSpace X
                                            inst✝ : TopologicalSpace Y
                                            t : Set X
                                            ht : Membership.mem inst✝¹.1 t
                                            ⊢ And (IsOpen t) (And (IsOpen Set.univ) (Eq (Set.preimage Prod.fst t) (SProd.s …
                                          -/
        GenerateOpen.basic _ ⟨t, univ, by simpa [Set.prod_eq] using ht⟩)
                                          /-
                                            🎉 no goals
                                          -/
      (forall_mem_image.2 fun t ht =>
                                          /-
                                            X : Type u
                                            Y : Type v
                                            inst✝¹ : TopologicalSpace X
                                            inst✝ : TopologicalSpace Y
                                            t : Set Y
                                            ht : Membership.mem inst✝.1 t
                                            ⊢ And (IsOpen Set.univ) (And (IsOpen t) (Eq (Set.preimage Prod.snd t) (SProd.s …
                                          -/
        GenerateOpen.basic _ ⟨univ, t, by simpa [Set.prod_eq] using ht⟩))
                                          /-
                                            🎉 no goals
                                          -/

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: align with `mem_nhds_prod_iff'`

theorem isOpen_prod_iff {s : Set (X × Y)} :
    IsOpen s ↔ ∀ a b, (a, b) ∈ s →
      ∃ u v, IsOpen u ∧ IsOpen v ∧ a ∈ u ∧ b ∈ v ∧ u ×ˢ v ⊆ s :=
                                  /-
                                    X : Type u
                                    Y : Type v
                                    inst✝¹ : TopologicalSpace X
                                    inst✝ : TopologicalSpace Y
                                    s : Set (Prod X Y)
                                    ⊢ Iff (∀ (x : Prod X Y), Membership.mem s x → Membership.mem (nhds x) s) (∀ (a …
                                  -/
  isOpen_iff_mem_nhds.trans <| by simp_rw [Prod.forall, mem_nhds_prod_iff', and_left_comm]
                                  /-
                                    🎉 no goals
                                  -/


/-- A product of induced topologies is induced by the product map -/
theorem prod_induced_induced {X Z} (f : X → Y) (g : Z → W) :
    @instTopologicalSpaceProd X Z (induced f ‹_›) (induced g ‹_›) =
      induced (fun p => (f p.1, g p.2)) instTopologicalSpaceProd := by
  /-
    Y : Type v
    W : Type u_2
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace W
    X : Type u_5
    Z : Type u_6
    f : X → Y
    g : Z → W
    ⊢ Eq instTopologicalSpaceProd (TopologicalSpace.induced (fun p => { fst := f p …
  -/
  delta instTopologicalSpaceProd
  /-
    Y : Type v
    W : Type u_2
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace W
    X : Type u_5
    Z : Type u_6
    f : X → Y
    g : Z → W
    ⊢ Eq (Min.min (TopologicalSpace.induced Prod.fst (TopologicalSpace.induced f i …
  -/
  simp_rw [induced_inf, induced_compose]
  /-
    Y : Type v
    W : Type u_2
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace W
    X : Type u_5
    Z : Type u_6
    f : X → Y
    g : Z → W
    ⊢ Eq (Min.min (TopologicalSpace.induced (Function.comp f Prod.fst) inst✝¹) (To …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Given a neighborhood `s` of `(x, x)`, then `(x, x)` has a square open neighborhood
  that is a subset of `s`. -/
theorem exists_nhds_square {s : Set (X × X)} {x : X} (hx : s ∈ 𝓝 (x, x)) :
    ∃ U : Set X, IsOpen U ∧ x ∈ U ∧ U ×ˢ U ⊆ s := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set (Prod X X)
    x : X
    hx : Membership.mem (nhds { fst := x, snd := x }) s
    ⊢ Exists fun U => And (IsOpen U) (And (Membership.mem U x) (HasSubset.Subset ( …
  -/
  simpa [nhds_prod_eq, (nhds_basis_opens x).prod_self.mem_iff, and_assoc, and_left_comm] using hx
  /-
    🎉 no goals
  -/


/-- `Prod.fst` maps neighborhood of `x : X × Y` within the section `Prod.snd ⁻¹' {x.2}`
to `𝓝 x.1`. -/
theorem map_fst_nhdsWithin (x : X × Y) : map Prod.fst (𝓝[Prod.snd ⁻¹' {x.2}] x) = 𝓝 x.1 := by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    x : Prod X Y
    ⊢ Eq (Filter.map Prod.fst (nhdsWithin x (Set.preimage Prod.snd (Singleton.sing …
  -/
  refine le_antisymm (continuousAt_fst.mono_left inf_le_left) fun s hs => ?_
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    x : Prod X Y
    s : Set X
    hs : Membership.mem (Filter.map Prod.fst (nhdsWithin x (Set.preimage Prod.snd  …
    ⊢ Membership.mem (nhds x.1) s
  -/
  rcases x with ⟨x, y⟩
  /-
    case mk
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set X
    x : X
    y : Y
    hs : Membership.mem (Filter.map Prod.fst (nhdsWithin { fst := x, snd := y } (S …
    ⊢ Membership.mem (nhds { fst := x, snd := y }.1) s
  -/
  rw [mem_map, nhdsWithin, mem_inf_principal, mem_nhds_prod_iff] at hs
  /-
    case mk
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set X
    x : X
    y : Y
    hs : Exists fun u => And (Membership.mem (nhds x) u) (Exists fun v => And (Mem …
    ⊢ Membership.mem (nhds { fst := x, snd := y }.1) s
  -/
  rcases hs with ⟨u, hu, v, hv, H⟩
  /-
    case mk.intro.intro.intro.intro
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set X
    x : X
    y : Y
    u : Set X
    hu : Membership.mem (nhds x) u
    v : Set Y
    hv : Membership.mem (nhds y) v
    H : HasSubset.Subset (SProd.sprod u v) (setOf fun x_1 => Membership.mem (Set.p …
    ⊢ Membership.mem (nhds { fst := x, snd := y }.1) s
  -/
  simp only [prod_subset_iff, mem_singleton_iff, mem_setOf_eq, mem_preimage] at H
  /-
    case mk.intro.intro.intro.intro
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set X
    x : X
    y : Y
    u : Set X
    hu : Membership.mem (nhds x) u
    v : Set Y
    hv : Membership.mem (nhds y) v
    H : ∀ (x : X), Membership.mem u x → ∀ (y_1 : Y), Membership.mem v y_1 → Eq y_1 …
    ⊢ Membership.mem (nhds { fst := x, snd := y }.1) s
  -/
  exact mem_of_superset hu fun z hz => H _ hz _ (mem_of_mem_nhds hv) rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem map_fst_nhds (x : X × Y) : map Prod.fst (𝓝 x) = 𝓝 x.1 :=
  le_antisymm continuousAt_fst <| (map_fst_nhdsWithin x).symm.trans_le (map_mono inf_le_left)


/-- The first projection in a product of topological spaces sends open sets to open sets. -/
theorem isOpenMap_fst : IsOpenMap (@Prod.fst X Y) :=
  isOpenMap_iff_nhds_le.2 fun x => (map_fst_nhds x).ge


/-- `Prod.snd` maps neighborhood of `x : X × Y` within the section `Prod.fst ⁻¹' {x.1}`
to `𝓝 x.2`. -/
theorem map_snd_nhdsWithin (x : X × Y) : map Prod.snd (𝓝[Prod.fst ⁻¹' {x.1}] x) = 𝓝 x.2 := by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    x : Prod X Y
    ⊢ Eq (Filter.map Prod.snd (nhdsWithin x (Set.preimage Prod.fst (Singleton.sing …
  -/
  refine le_antisymm (continuousAt_snd.mono_left inf_le_left) fun s hs => ?_
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    x : Prod X Y
    s : Set Y
    hs : Membership.mem (Filter.map Prod.snd (nhdsWithin x (Set.preimage Prod.fst  …
    ⊢ Membership.mem (nhds x.2) s
  -/
  rcases x with ⟨x, y⟩
  /-
    case mk
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set Y
    x : X
    y : Y
    hs : Membership.mem (Filter.map Prod.snd (nhdsWithin { fst := x, snd := y } (S …
    ⊢ Membership.mem (nhds { fst := x, snd := y }.2) s
  -/
  rw [mem_map, nhdsWithin, mem_inf_principal, mem_nhds_prod_iff] at hs
  /-
    case mk
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set Y
    x : X
    y : Y
    hs : Exists fun u => And (Membership.mem (nhds x) u) (Exists fun v => And (Mem …
    ⊢ Membership.mem (nhds { fst := x, snd := y }.2) s
  -/
  rcases hs with ⟨u, hu, v, hv, H⟩
  /-
    case mk.intro.intro.intro.intro
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set Y
    x : X
    y : Y
    u : Set X
    hu : Membership.mem (nhds x) u
    v : Set Y
    hv : Membership.mem (nhds y) v
    H : HasSubset.Subset (SProd.sprod u v) (setOf fun x_1 => Membership.mem (Set.p …
    ⊢ Membership.mem (nhds { fst := x, snd := y }.2) s
  -/
  simp only [prod_subset_iff, mem_singleton_iff, mem_setOf_eq, mem_preimage] at H
  /-
    case mk.intro.intro.intro.intro
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set Y
    x : X
    y : Y
    u : Set X
    hu : Membership.mem (nhds x) u
    v : Set Y
    hv : Membership.mem (nhds y) v
    H : ∀ (x_1 : X), Membership.mem u x_1 → ∀ (y : Y), Membership.mem v y → Eq x_1 …
    ⊢ Membership.mem (nhds { fst := x, snd := y }.2) s
  -/
  exact mem_of_superset hv fun z hz => H _ (mem_of_mem_nhds hu) _ hz rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem map_snd_nhds (x : X × Y) : map Prod.snd (𝓝 x) = 𝓝 x.2 :=
  le_antisymm continuousAt_snd <| (map_snd_nhdsWithin x).symm.trans_le (map_mono inf_le_left)


/-- The second projection in a product of topological spaces sends open sets to open sets. -/
theorem isOpenMap_snd : IsOpenMap (@Prod.snd X Y) :=
  isOpenMap_iff_nhds_le.2 fun x => (map_snd_nhds x).ge


/-- A product set is open in a product space if and only if each factor is open, or one of them is
empty -/
theorem isOpen_prod_iff' {s : Set X} {t : Set Y} :
    IsOpen (s ×ˢ t) ↔ IsOpen s ∧ IsOpen t ∨ s = ∅ ∨ t = ∅ := by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set X
    t : Set Y
    ⊢ Iff (IsOpen (SProd.sprod s t)) (Or (And (IsOpen s) (IsOpen t)) (Or (Eq s Emp …
  -/
  rcases (s ×ˢ t).eq_empty_or_nonempty with h | h
    /-
      case inl
      X : Type u
      Y : Type v
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      s : Set X
      t : Set Y
      h : Eq (SProd.sprod s t) EmptyCollection.emptyCollection
      ⊢ Iff (IsOpen (SProd.sprod s t)) (Or (And (IsOpen s) (IsOpen t)) (Or (Eq s Emp …
    -/
  · simp [h, prod_eq_empty_iff.1 h]
    /-
      🎉 no goals
    -/
    /-
      case inr
      X : Type u
      Y : Type v
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      s : Set X
      t : Set Y
      h : (SProd.sprod s t).Nonempty
      ⊢ Iff (IsOpen (SProd.sprod s t)) (Or (And (IsOpen s) (IsOpen t)) (Or (Eq s Emp …
    -/
  · have st : s.Nonempty ∧ t.Nonempty := prod_nonempty_iff.1 h
    /-
      case inr
      X : Type u
      Y : Type v
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      s : Set X
      t : Set Y
      h : (SProd.sprod s t).Nonempty
      st : And s.Nonempty t.Nonempty
      ⊢ Iff (IsOpen (SProd.sprod s t)) (Or (And (IsOpen s) (IsOpen t)) (Or (Eq s Emp …
    -/
    constructor
      /-
        case inr.mp
        X : Type u
        Y : Type v
        inst✝¹ : TopologicalSpace X
        inst✝ : TopologicalSpace Y
        s : Set X
        t : Set Y
        h : (SProd.sprod s t).Nonempty
        st : And s.Nonempty t.Nonempty
        ⊢ IsOpen (SProd.sprod s t) → Or (And (IsOpen s) (IsOpen t)) (Or (Eq s EmptyCol …
      -/
    · intro (H : IsOpen (s ×ˢ t))
      /-
        case inr.mp
        X : Type u
        Y : Type v
        inst✝¹ : TopologicalSpace X
        inst✝ : TopologicalSpace Y
        s : Set X
        t : Set Y
        h : (SProd.sprod s t).Nonempty
        st : And s.Nonempty t.Nonempty
        H : IsOpen (SProd.sprod s t)
        ⊢ Or (And (IsOpen s) (IsOpen t)) (Or (Eq s EmptyCollection.emptyCollection) (E …
      -/
      refine Or.inl ⟨?_, ?_⟩
        /-
          case inr.mp.refine_1
          X : Type u
          Y : Type v
          inst✝¹ : TopologicalSpace X
          inst✝ : TopologicalSpace Y
          s : Set X
          t : Set Y
          h : (SProd.sprod s t).Nonempty
          st : And s.Nonempty t.Nonempty
          H : IsOpen (SProd.sprod s t)
          ⊢ IsOpen s
        -/
      · simpa only [fst_image_prod _ st.2] using isOpenMap_fst _ H
        /-
          🎉 no goals
        -/
        /-
          case inr.mp.refine_2
          X : Type u
          Y : Type v
          inst✝¹ : TopologicalSpace X
          inst✝ : TopologicalSpace Y
          s : Set X
          t : Set Y
          h : (SProd.sprod s t).Nonempty
          st : And s.Nonempty t.Nonempty
          H : IsOpen (SProd.sprod s t)
          ⊢ IsOpen t
        -/
      · simpa only [snd_image_prod st.1 t] using isOpenMap_snd _ H
        /-
          🎉 no goals
        -/
      /-
        case inr.mpr
        X : Type u
        Y : Type v
        inst✝¹ : TopologicalSpace X
        inst✝ : TopologicalSpace Y
        s : Set X
        t : Set Y
        h : (SProd.sprod s t).Nonempty
        st : And s.Nonempty t.Nonempty
        ⊢ Or (And (IsOpen s) (IsOpen t)) (Or (Eq s EmptyCollection.emptyCollection) (E …
      -/
    · intro H
      /-
        case inr.mpr
        X : Type u
        Y : Type v
        inst✝¹ : TopologicalSpace X
        inst✝ : TopologicalSpace Y
        s : Set X
        t : Set Y
        h : (SProd.sprod s t).Nonempty
        st : And s.Nonempty t.Nonempty
        H : Or (And (IsOpen s) (IsOpen t)) (Or (Eq s EmptyCollection.emptyCollection)  …
        ⊢ IsOpen (SProd.sprod s t)
      -/
      simp only [st.1.ne_empty, st.2.ne_empty, not_false_iff, or_false] at H
      /-
        case inr.mpr
        X : Type u
        Y : Type v
        inst✝¹ : TopologicalSpace X
        inst✝ : TopologicalSpace Y
        s : Set X
        t : Set Y
        h : (SProd.sprod s t).Nonempty
        st : And s.Nonempty t.Nonempty
        H : And (IsOpen s) (IsOpen t)
        ⊢ IsOpen (SProd.sprod s t)
      -/
      exact H.1.prod H.2
      /-
        🎉 no goals
      -/


theorem isQuotientMap_fst [Nonempty Y] : IsQuotientMap (Prod.fst : X × Y → X) :=
  isOpenMap_fst.isQuotientMap continuous_fst Prod.fst_surjective


@[deprecated (since := "2024-10-22")]
alias quotientMap_fst := isQuotientMap_fst


theorem isQuotientMap_snd [Nonempty X] : IsQuotientMap (Prod.snd : X × Y → Y) :=
  isOpenMap_snd.isQuotientMap continuous_snd Prod.snd_surjective


@[deprecated (since := "2024-10-22")]
alias quotientMap_snd := isQuotientMap_snd


theorem closure_prod_eq {s : Set X} {t : Set Y} : closure (s ×ˢ t) = closure s ×ˢ closure t :=
  ext fun ⟨a, b⟩ => by
    /-
      X : Type u
      Y : Type v
      inst✝¹ : TopologicalSpace X
      inst✝ : TopologicalSpace Y
      s : Set X
      t : Set Y
      x✝ : Prod X Y
      a : X
      b : Y
      ⊢ Iff (Membership.mem (closure (SProd.sprod s t)) { fst := a, snd := b }) (Mem …
    -/
    simp_rw [mem_prod, mem_closure_iff_nhdsWithin_neBot, nhdsWithin_prod_eq, prod_neBot]
    /-
      🎉 no goals
    -/


theorem interior_prod_eq (s : Set X) (t : Set Y) : interior (s ×ˢ t) = interior s ×ˢ interior t :=
                       /-
                         X : Type u
                         Y : Type v
                         inst✝¹ : TopologicalSpace X
                         inst✝ : TopologicalSpace Y
                         s : Set X
                         t : Set Y
                         x✝ : Prod X Y
                         a : X
                         b : Y
                         ⊢ Iff (Membership.mem (interior (SProd.sprod s t)) { fst := a, snd := b }) (Me …
                       -/
  ext fun ⟨a, b⟩ => by simp only [mem_interior_iff_mem_nhds, mem_prod, prod_mem_nhds_iff]
                       /-
                         🎉 no goals
                       -/


theorem frontier_prod_eq (s : Set X) (t : Set Y) :
    frontier (s ×ˢ t) = closure s ×ˢ frontier t ∪ frontier s ×ˢ closure t := by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set X
    t : Set Y
    ⊢ Eq (frontier (SProd.sprod s t)) (Union.union (SProd.sprod (closure s) (front …
  -/
  simp only [frontier, closure_prod_eq, interior_prod_eq, prod_diff_prod]
  /-
    🎉 no goals
  -/


@[simp]
theorem frontier_prod_univ_eq (s : Set X) :
    frontier (s ×ˢ (univ : Set Y)) = frontier s ×ˢ univ := by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set X
    ⊢ Eq (frontier (SProd.sprod s Set.univ)) (SProd.sprod (frontier s) Set.univ)
  -/
  simp [frontier_prod_eq]
  /-
    🎉 no goals
  -/


@[simp]
theorem frontier_univ_prod_eq (s : Set Y) :
    frontier ((univ : Set X) ×ˢ s) = univ ×ˢ frontier s := by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set Y
    ⊢ Eq (frontier (SProd.sprod Set.univ s)) (SProd.sprod Set.univ (frontier s))
  -/
  simp [frontier_prod_eq]
  /-
    🎉 no goals
  -/


theorem map_mem_closure₂ {f : X → Y → Z} {x : X} {y : Y} {s : Set X} {t : Set Y} {u : Set Z}
    (hf : Continuous (uncurry f)) (hx : x ∈ closure s) (hy : y ∈ closure t)
    (h : ∀ a ∈ s, ∀ b ∈ t, f a b ∈ u) : f x y ∈ closure u :=
                                            /-
                                              X : Type u
                                              Y : Type v
                                              Z : Type u_1
                                              inst✝² : TopologicalSpace X
                                              inst✝¹ : TopologicalSpace Y
                                              inst✝ : TopologicalSpace Z
                                              f : X → Y → Z
                                              x : X
                                              y : Y
                                              s : Set X
                                              t : Set Y
                                              u : Set Z
                                              hf : Continuous (Function.uncurry f)
                                              hx : Membership.mem (closure s) x
                                              hy : Membership.mem (closure t) y
                                              h : ∀ (a : X), Membership.mem s a → ∀ (b : Y), Membership.mem t b → Membership …
                                              ⊢ Membership.mem (closure (SProd.sprod s t)) { fst := x, snd := y }
                                            -/
  have H₁ : (x, y) ∈ closure (s ×ˢ t) := by simpa only [closure_prod_eq] using mk_mem_prod hx hy
                                            /-
                                              🎉 no goals
                                            -/
  have H₂ : MapsTo (uncurry f) (s ×ˢ t) u := forall_prod_set.2 h
  H₂.closure hf H₁


theorem IsClosed.prod {s₁ : Set X} {s₂ : Set Y} (h₁ : IsClosed s₁) (h₂ : IsClosed s₂) :
    IsClosed (s₁ ×ˢ s₂) :=
                                   /-
                                     X : Type u
                                     Y : Type v
                                     inst✝¹ : TopologicalSpace X
                                     inst✝ : TopologicalSpace Y
                                     s₁ : Set X
                                     s₂ : Set Y
                                     h₁ : IsClosed s₁
                                     h₂ : IsClosed s₂
                                     ⊢ Eq (closure (SProd.sprod s₁ s₂)) (SProd.sprod s₁ s₂)
                                   -/
  closure_eq_iff_isClosed.mp <| by simp only [h₁.closure_eq, h₂.closure_eq, closure_prod_eq]
                                   /-
                                     🎉 no goals
                                   -/


/-- The product of two dense sets is a dense set. -/
theorem Dense.prod {s : Set X} {t : Set Y} (hs : Dense s) (ht : Dense t) : Dense (s ×ˢ t) :=
  fun x => by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set X
    t : Set Y
    hs : Dense s
    ht : Dense t
    x : Prod X Y
    ⊢ Membership.mem (closure (SProd.sprod s t)) x
  -/
  rw [closure_prod_eq]
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set X
    t : Set Y
    hs : Dense s
    ht : Dense t
    x : Prod X Y
    ⊢ Membership.mem (SProd.sprod (closure s) (closure t)) x
  -/
  exact ⟨hs x.1, ht x.2⟩
  /-
    🎉 no goals
  -/


/-- If `f` and `g` are maps with dense range, then `Prod.map f g` has dense range. -/
theorem DenseRange.prodMap {ι : Type*} {κ : Type*} {f : ι → Y} {g : κ → Z} (hf : DenseRange f)
    (hg : DenseRange g) : DenseRange (Prod.map f g) := by
  /-
    Y : Type v
    Z : Type u_1
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    ι : Type u_5
    κ : Type u_6
    f : ι → Y
    g : κ → Z
    hf : DenseRange f
    hg : DenseRange g
    ⊢ DenseRange (Prod.map f g)
  -/
  simpa only [DenseRange, prod_range_range_eq] using hf.prod hg
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-05")] alias DenseRange.prod_map := DenseRange.prodMap


lemma Topology.IsInducing.prodMap {f : X → Y} {g : Z → W} (hf : IsInducing f) (hg : IsInducing g) :
    IsInducing (Prod.map f g) :=
  isInducing_iff_nhds.2 fun (x, z) => by simp_rw [Prod.map_def, nhds_prod_eq, hf.nhds_eq_comap,
    hg.nhds_eq_comap, prod_comap_comap_eq]


@[deprecated (since := "2024-10-28")] alias Inducing.prodMap := IsInducing.prodMap


@[deprecated (since := "2024-10-05")] alias Inducing.prod_map := IsInducing.prodMap


@[simp]
lemma Topology.isInducing_const_prod {x : X} {f : Y → Z} :
    IsInducing (fun x' => (x, f x')) ↔ IsInducing f := by
  simp_rw [isInducing_iff, instTopologicalSpaceProd, induced_inf, induced_compose,
    Function.comp_def, induced_const, top_inf_eq]


@[deprecated (since := "2024-10-28")] alias inducing_const_prod := isInducing_const_prod


@[simp]
lemma Topology.isInducing_prod_const {y : Y} {f : X → Z} :
    IsInducing (fun x => (f x, y)) ↔ IsInducing f := by
  simp_rw [isInducing_iff, instTopologicalSpaceProd, induced_inf, induced_compose,
    Function.comp_def, induced_const, inf_top_eq]


@[deprecated (since := "2024-10-28")] alias inducing_prod_const := isInducing_prod_const


lemma Topology.IsEmbedding.prodMap {f : X → Y} {g : Z → W} (hf : IsEmbedding f)
    (hg : IsEmbedding g) : IsEmbedding (Prod.map f g) where
  toIsInducing := hf.isInducing.prodMap hg.isInducing
  injective := hf.injective.prodMap hg.injective


@[deprecated (since := "2024-10-08")] alias Embedding.prodMap := Topology.IsEmbedding.prodMap

@[deprecated (since := "2024-10-05")] alias Embedding.prod_map := Topology.IsEmbedding.prodMap


protected theorem IsOpenMap.prodMap {f : X → Y} {g : Z → W} (hf : IsOpenMap f) (hg : IsOpenMap g) :
    IsOpenMap (Prod.map f g) := by
  /-
    X : Type u
    Y : Type v
    Z : Type u_1
    W : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : TopologicalSpace Z
    inst✝ : TopologicalSpace W
    f : X → Y
    g : Z → W
    hf : IsOpenMap f
    hg : IsOpenMap g
    ⊢ IsOpenMap (Prod.map f g)
  -/
  rw [isOpenMap_iff_nhds_le]
  /-
    X : Type u
    Y : Type v
    Z : Type u_1
    W : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : TopologicalSpace Z
    inst✝ : TopologicalSpace W
    f : X → Y
    g : Z → W
    hf : IsOpenMap f
    hg : IsOpenMap g
    ⊢ ∀ (x : Prod X Z), LE.le (nhds (Prod.map f g x)) (Filter.map (Prod.map f g) ( …
  -/
  rintro ⟨a, b⟩
  /-
    case mk
    X : Type u
    Y : Type v
    Z : Type u_1
    W : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : TopologicalSpace Z
    inst✝ : TopologicalSpace W
    f : X → Y
    g : Z → W
    hf : IsOpenMap f
    hg : IsOpenMap g
    a : X
    b : Z
    ⊢ LE.le (nhds (Prod.map f g { fst := a, snd := b })) (Filter.map (Prod.map f g …
  -/
  rw [nhds_prod_eq, nhds_prod_eq, ← Filter.prod_map_map_eq']
  /-
    case mk
    X : Type u
    Y : Type v
    Z : Type u_1
    W : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : TopologicalSpace Z
    inst✝ : TopologicalSpace W
    f : X → Y
    g : Z → W
    hf : IsOpenMap f
    hg : IsOpenMap g
    a : X
    b : Z
    ⊢ LE.le (SProd.sprod (nhds (Prod.map f g { fst := a, snd := b }).1) (nhds (Pro …
  -/
  exact Filter.prod_mono (hf.nhds_le a) (hg.nhds_le b)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-05")] alias IsOpenMap.prod := IsOpenMap.prodMap


protected lemma Topology.IsOpenEmbedding.prodMap {f : X → Y} {g : Z → W} (hf : IsOpenEmbedding f)
    (hg : IsOpenEmbedding g) : IsOpenEmbedding (Prod.map f g) :=
  .of_isEmbedding_isOpenMap (hf.1.prodMap hg.1) (hf.isOpenMap.prodMap hg.isOpenMap)


@[deprecated (since := "2024-10-18")]
alias OpenEmbedding.prodMap := IsOpenEmbedding.prodMap


@[deprecated (since := "2024-10-05")] alias IsOpenEmbedding.prod := IsOpenEmbedding.prodMap


lemma isEmbedding_graph {f : X → Y} (hf : Continuous f) : IsEmbedding fun x => (x, f x) :=
  .of_comp (continuous_id.prod_mk hf) continuous_fst .id


@[deprecated (since := "2024-10-26")]
alias embedding_graph := isEmbedding_graph


lemma isEmbedding_prodMk (x : X) : IsEmbedding (Prod.mk x : Y → X × Y) :=
  .of_comp (Continuous.Prod.mk x) continuous_snd .id


@[deprecated (since := "2024-10-26")]
alias embedding_prod_mk := isEmbedding_prodMk


theorem IsOpenQuotientMap.prodMap {f : X → Y} {g : Z → W} (hf : IsOpenQuotientMap f)
    (hg : IsOpenQuotientMap g) : IsOpenQuotientMap (Prod.map f g) :=
  ⟨.prodMap hf.1 hg.1, .prodMap hf.2 hg.2, .prodMap hf.3 hg.3⟩


lemma continuous_bool_rng [TopologicalSpace X] {f : X → Bool} (b : Bool) :
    Continuous f ↔ IsClopen (f ⁻¹' {b}) := by
  rw [continuous_discrete_rng, Bool.forall_bool' b, IsClopen, ← isOpen_compl_iff, ← preimage_compl,
    Bool.compl_singleton, and_comm]


theorem continuous_sum_dom {f : X ⊕ Y → Z} :
    Continuous f ↔ Continuous (f ∘ Sum.inl) ∧ Continuous (f ∘ Sum.inr) :=
  (continuous_sup_dom (t₁ := TopologicalSpace.coinduced Sum.inl _)
    (t₂ := TopologicalSpace.coinduced Sum.inr _)).trans <|
    continuous_coinduced_dom.and continuous_coinduced_dom


theorem continuous_sum_elim {f : X → Z} {g : Y → Z} :
    Continuous (Sum.elim f g) ↔ Continuous f ∧ Continuous g :=
  continuous_sum_dom


@[continuity, fun_prop]
theorem Continuous.sum_elim {f : X → Z} {g : Y → Z} (hf : Continuous f) (hg : Continuous g) :
    Continuous (Sum.elim f g) :=
  continuous_sum_elim.2 ⟨hf, hg⟩


@[continuity, fun_prop]
theorem continuous_isLeft : Continuous (isLeft : X ⊕ Y → Bool) :=
  continuous_sum_dom.2 ⟨continuous_const, continuous_const⟩


@[continuity, fun_prop]
theorem continuous_isRight : Continuous (isRight : X ⊕ Y → Bool) :=
  continuous_sum_dom.2 ⟨continuous_const, continuous_const⟩


@[continuity, fun_prop]
theorem continuous_inl : Continuous (@inl X Y) := ⟨fun _ => And.left⟩


@[continuity, fun_prop]
theorem continuous_inr : Continuous (@inr X Y) := ⟨fun _ => And.right⟩


@[fun_prop, continuity]
lemma continuous_sum_swap : Continuous (@Sum.swap X Y) :=
  Continuous.sum_elim continuous_inr continuous_inl


theorem isOpen_sum_iff {s : Set (X ⊕ Y)} : IsOpen s ↔ IsOpen (inl ⁻¹' s) ∧ IsOpen (inr ⁻¹' s) :=
  Iff.rfl


theorem isClosed_sum_iff {s : Set (X ⊕ Y)} :
    IsClosed s ↔ IsClosed (inl ⁻¹' s) ∧ IsClosed (inr ⁻¹' s) := by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set (Sum X Y)
    ⊢ Iff (IsClosed s) (And (IsClosed (Set.preimage Sum.inl s)) (IsClosed (Set.pre …
  -/
  simp only [← isOpen_compl_iff, isOpen_sum_iff, preimage_compl]
  /-
    🎉 no goals
  -/


theorem isOpenMap_inl : IsOpenMap (@inl X Y) := fun u hu => by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    u : Set X
    hu : IsOpen u
    ⊢ IsOpen (Set.image Sum.inl u)
  -/
  simpa [isOpen_sum_iff, preimage_image_eq u Sum.inl_injective]
  /-
    🎉 no goals
  -/


theorem isOpenMap_inr : IsOpenMap (@inr X Y) := fun u hu => by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    u : Set Y
    hu : IsOpen u
    ⊢ IsOpen (Set.image Sum.inr u)
  -/
  simpa [isOpen_sum_iff, preimage_image_eq u Sum.inr_injective]
  /-
    🎉 no goals
  -/


protected lemma Topology.IsOpenEmbedding.inl : IsOpenEmbedding (@inl X Y) :=
  .of_continuous_injective_isOpenMap continuous_inl inl_injective isOpenMap_inl


@[deprecated (since := "2024-10-30")] alias isOpenEmbedding_inl := IsOpenEmbedding.inl


@[deprecated (since := "2024-10-18")]
alias openEmbedding_inl := IsOpenEmbedding.inl


protected lemma Topology.IsOpenEmbedding.inr : IsOpenEmbedding (@inr X Y) :=
  .of_continuous_injective_isOpenMap continuous_inr inr_injective isOpenMap_inr


@[deprecated (since := "2024-10-30")] alias isOpenEmbedding_inr := IsOpenEmbedding.inr


@[deprecated (since := "2024-10-18")]
alias openEmbedding_inr := IsOpenEmbedding.inr


protected lemma Topology.IsEmbedding.inl : IsEmbedding (@inl X Y) := IsOpenEmbedding.inl.1

protected lemma Topology.IsEmbedding.inr : IsEmbedding (@inr X Y) := IsOpenEmbedding.inr.1


@[deprecated (since := "2024-10-26")]
alias embedding_inr := IsEmbedding.inr


lemma isOpen_range_inl : IsOpen (range (inl : X → X ⊕ Y)) := IsOpenEmbedding.inl.2

lemma isOpen_range_inr : IsOpen (range (inr : Y → X ⊕ Y)) := IsOpenEmbedding.inr.2


theorem isClosed_range_inl : IsClosed (range (inl : X → X ⊕ Y)) := by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    ⊢ IsClosed (Set.range Sum.inl)
  -/
  rw [← isOpen_compl_iff, compl_range_inl]
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    ⊢ IsOpen (Set.range Sum.inr)
  -/
  exact isOpen_range_inr
  /-
    🎉 no goals
  -/


theorem isClosed_range_inr : IsClosed (range (inr : Y → X ⊕ Y)) := by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    ⊢ IsClosed (Set.range Sum.inr)
  -/
  rw [← isOpen_compl_iff, compl_range_inr]
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    ⊢ IsOpen (Set.range Sum.inl)
  -/
  exact isOpen_range_inl
  /-
    🎉 no goals
  -/


theorem Topology.IsClosedEmbedding.inl : IsClosedEmbedding (inl : X → X ⊕ Y) :=
  ⟨.inl, isClosed_range_inl⟩


@[deprecated (since := "2024-10-30")] alias isClosedEmbedding_inl := IsClosedEmbedding.inl


@[deprecated (since := "2024-10-20")]
alias closedEmbedding_inl := IsClosedEmbedding.inl


theorem Topology.IsClosedEmbedding.inr : IsClosedEmbedding (inr : Y → X ⊕ Y) :=
  ⟨.inr, isClosed_range_inr⟩


@[deprecated (since := "2024-10-30")] alias isClosedEmbedding_inr := IsClosedEmbedding.inr


@[deprecated (since := "2024-10-20")]
alias closedEmbedding_inr := IsClosedEmbedding.inr


theorem nhds_inl (x : X) : 𝓝 (inl x : X ⊕ Y) = map inl (𝓝 x) :=
  (IsOpenEmbedding.inl.map_nhds_eq _).symm


theorem nhds_inr (y : Y) : 𝓝 (inr y : X ⊕ Y) = map inr (𝓝 y) :=
  (IsOpenEmbedding.inr.map_nhds_eq _).symm


@[simp]
theorem continuous_sum_map {f : X → Y} {g : Z → W} :
    Continuous (Sum.map f g) ↔ Continuous f ∧ Continuous g :=
  continuous_sum_elim.trans <|
    IsEmbedding.inl.continuous_iff.symm.and IsEmbedding.inr.continuous_iff.symm


@[continuity, fun_prop]
theorem Continuous.sum_map {f : X → Y} {g : Z → W} (hf : Continuous f) (hg : Continuous g) :
    Continuous (Sum.map f g) :=
  continuous_sum_map.2 ⟨hf, hg⟩


theorem isOpenMap_sum {f : X ⊕ Y → Z} :
    IsOpenMap f ↔ (IsOpenMap fun a => f (inl a)) ∧ IsOpenMap fun b => f (inr b) := by
  /-
    X : Type u
    Y : Type v
    Z : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    f : Sum X Y → Z
    ⊢ Iff (IsOpenMap f) (And (IsOpenMap fun a => f (Sum.inl a)) (IsOpenMap fun b = …
  -/
  simp only [isOpenMap_iff_nhds_le, Sum.forall, nhds_inl, nhds_inr, Filter.map_map, comp_def]
  /-
    🎉 no goals
  -/


theorem IsOpenMap.sumMap {f : X → Y} {g : Z → W} (hf : IsOpenMap f) (hg : IsOpenMap g) :
    IsOpenMap (Sum.map f g) := by
  /-
    X : Type u
    Y : Type v
    Z : Type u_1
    W : Type u_2
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : TopologicalSpace Z
    inst✝ : TopologicalSpace W
    f : X → Y
    g : Z → W
    hf : IsOpenMap f
    hg : IsOpenMap g
    ⊢ IsOpenMap (Sum.map f g)
  -/
  exact isOpenMap_sum.2 ⟨isOpenMap_inl.comp hf,isOpenMap_inr.comp hg⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem isOpenMap_sum_elim {f : X → Z} {g : Y → Z} :
    IsOpenMap (Sum.elim f g) ↔ IsOpenMap f ∧ IsOpenMap g := by
  /-
    X : Type u
    Y : Type v
    Z : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    f : X → Z
    g : Y → Z
    ⊢ Iff (IsOpenMap (Sum.elim f g)) (And (IsOpenMap f) (IsOpenMap g))
  -/
  simp only [isOpenMap_sum, elim_inl, elim_inr]
  /-
    🎉 no goals
  -/


theorem IsOpenMap.sum_elim {f : X → Z} {g : Y → Z} (hf : IsOpenMap f) (hg : IsOpenMap g) :
    IsOpenMap (Sum.elim f g) :=
  isOpenMap_sum_elim.2 ⟨hf, hg⟩


theorem isClosedMap_sum {f : X ⊕ Y → Z} :
    IsClosedMap f ↔ (IsClosedMap fun a => f (.inl a)) ∧ IsClosedMap fun b => f (.inr b) := by
  /-
    X : Type u
    Y : Type v
    Z : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : TopologicalSpace Y
    inst✝ : TopologicalSpace Z
    f : Sum X Y → Z
    ⊢ Iff (IsClosedMap f) (And (IsClosedMap fun a => f (Sum.inl a)) (IsClosedMap f …
  -/
  constructor
    /-
      case mp
      X : Type u
      Y : Type v
      Z : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : TopologicalSpace Z
      f : Sum X Y → Z
      ⊢ IsClosedMap f → And (IsClosedMap fun a => f (Sum.inl a)) (IsClosedMap fun b  …
    -/
  · intro h
    /-
      case mp
      X : Type u
      Y : Type v
      Z : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : TopologicalSpace Z
      f : Sum X Y → Z
      h : IsClosedMap f
      ⊢ And (IsClosedMap fun a => f (Sum.inl a)) (IsClosedMap fun b => f (Sum.inr b))
    -/
    exact ⟨h.comp IsClosedEmbedding.inl.isClosedMap, h.comp IsClosedEmbedding.inr.isClosedMap⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      X : Type u
      Y : Type v
      Z : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : TopologicalSpace Z
      f : Sum X Y → Z
      ⊢ And (IsClosedMap fun a => f (Sum.inl a)) (IsClosedMap fun b => f (Sum.inr b) …
    -/
  · rintro h Z hZ
    /-
      case mpr
      X : Type u
      Y : Type v
      Z✝ : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : TopologicalSpace Z✝
      f : Sum X Y → Z✝
      h : And (IsClosedMap fun a => f (Sum.inl a)) (IsClosedMap fun b => f (Sum.inr  …
      Z : Set (Sum X Y)
      hZ : IsClosed Z
      ⊢ IsClosed (Set.image f Z)
    -/
    rw [isClosed_sum_iff] at hZ
    /-
      case mpr
      X : Type u
      Y : Type v
      Z✝ : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : TopologicalSpace Z✝
      f : Sum X Y → Z✝
      h : And (IsClosedMap fun a => f (Sum.inl a)) (IsClosedMap fun b => f (Sum.inr  …
      Z : Set (Sum X Y)
      hZ : And (IsClosed (Set.preimage Sum.inl Z)) (IsClosed (Set.preimage Sum.inr Z))
      ⊢ IsClosed (Set.image f Z)
    -/
    convert (h.1 _ hZ.1).union (h.2 _ hZ.2)
    /-
      case h.e'_3
      X : Type u
      Y : Type v
      Z✝ : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : TopologicalSpace Z✝
      f : Sum X Y → Z✝
      h : And (IsClosedMap fun a => f (Sum.inl a)) (IsClosedMap fun b => f (Sum.inr  …
      Z : Set (Sum X Y)
      hZ : And (IsClosed (Set.preimage Sum.inl Z)) (IsClosed (Set.preimage Sum.inr Z))
      ⊢ Eq (Set.image f Z) (Union.union (Set.image (fun a => f (Sum.inl a)) (Set.pre …
    -/
    ext
    /-
      case h.e'_3.h
      X : Type u
      Y : Type v
      Z✝ : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : TopologicalSpace Y
      inst✝ : TopologicalSpace Z✝
      f : Sum X Y → Z✝
      h : And (IsClosedMap fun a => f (Sum.inl a)) (IsClosedMap fun b => f (Sum.inr  …
      Z : Set (Sum X Y)
      hZ : And (IsClosed (Set.preimage Sum.inl Z)) (IsClosed (Set.preimage Sum.inr Z))
      x✝ : Z✝
      ⊢ Iff (Membership.mem (Set.image f Z) x✝) (Membership.mem (Union.union (Set.im …
    -/
    simp only [mem_image, Sum.exists, mem_union, mem_preimage]
    /-
      🎉 no goals
    -/


lemma Topology.IsInducing.subtypeVal {t : Set Y} : IsInducing ((↑) : t → Y) := ⟨rfl⟩


@[deprecated (since := "2024-10-28")] alias inducing_subtype_val := IsInducing.subtypeVal


lemma Topology.IsInducing.of_codRestrict {f : X → Y} {t : Set Y} (ht : ∀ x, f x ∈ t)
    (h : IsInducing (t.codRestrict f ht)) : IsInducing f := subtypeVal.comp h


@[deprecated (since := "2024-10-28")] alias Inducing.of_codRestrict := IsInducing.of_codRestrict


lemma Topology.IsEmbedding.subtypeVal : IsEmbedding ((↑) : Subtype p → X) :=
  ⟨.subtypeVal, Subtype.coe_injective⟩


@[deprecated (since := "2024-10-26")]
alias embedding_subtype_val := IsEmbedding.subtypeVal


theorem Topology.IsClosedEmbedding.subtypeVal (h : IsClosed {a | p a}) :
    IsClosedEmbedding ((↑) : Subtype p → X) :=
                   /-
                     X : Type u
                     inst✝ : TopologicalSpace X
                     p : X → Prop
                     h : IsClosed (setOf fun a => p a)
                     ⊢ IsClosed (Set.range Subtype.val)
                   -/
  ⟨.subtypeVal, by rwa [Subtype.range_coe_subtype]⟩
                   /-
                     🎉 no goals
                   -/


@[deprecated (since := "2024-10-20")]
alias closedEmbedding_subtype_val := IsClosedEmbedding.subtypeVal


@[continuity, fun_prop]
theorem continuous_subtype_val : Continuous (@Subtype.val X p) :=
  continuous_induced_dom


theorem Continuous.subtype_val {f : Y → Subtype p} (hf : Continuous f) :
    Continuous fun x => (f x : X) :=
  continuous_subtype_val.comp hf


theorem IsOpen.isOpenEmbedding_subtypeVal {s : Set X} (hs : IsOpen s) :
    IsOpenEmbedding ((↑) : s → X) :=
  ⟨.subtypeVal, (@Subtype.range_coe _ s).symm ▸ hs⟩


@[deprecated (since := "2024-10-18")]
alias IsOpen.openEmbedding_subtype_val := IsOpen.isOpenEmbedding_subtypeVal


theorem IsOpen.isOpenMap_subtype_val {s : Set X} (hs : IsOpen s) : IsOpenMap ((↑) : s → X) :=
  hs.isOpenEmbedding_subtypeVal.isOpenMap


theorem IsOpenMap.restrict {f : X → Y} (hf : IsOpenMap f) {s : Set X} (hs : IsOpen s) :
    IsOpenMap (s.restrict f) :=
  hf.comp hs.isOpenMap_subtype_val


lemma IsClosed.isClosedEmbedding_subtypeVal {s : Set X} (hs : IsClosed s) :
    IsClosedEmbedding ((↑) : s → X) := .subtypeVal hs


@[deprecated (since := "2024-10-20")]
alias IsClosed.closedEmbedding_subtype_val := IsClosed.isClosedEmbedding_subtypeVal


theorem IsClosed.isClosedMap_subtype_val {s : Set X} (hs : IsClosed s) :
    IsClosedMap ((↑) : s → X) :=
  hs.isClosedEmbedding_subtypeVal.isClosedMap


@[continuity, fun_prop]
theorem Continuous.subtype_mk {f : Y → X} (h : Continuous f) (hp : ∀ x, p (f x)) :
    Continuous fun x => (⟨f x, hp x⟩ : Subtype p) :=
  continuous_induced_rng.2 h


theorem Continuous.subtype_map {f : X → Y} (h : Continuous f) {q : Y → Prop}
    (hpq : ∀ x, p x → q (f x)) : Continuous (Subtype.map f hpq) :=
  (h.comp continuous_subtype_val).subtype_mk _


theorem continuous_inclusion {s t : Set X} (h : s ⊆ t) : Continuous (inclusion h) :=
  continuous_id.subtype_map h


theorem continuousAt_subtype_val {p : X → Prop} {x : Subtype p} :
    ContinuousAt ((↑) : Subtype p → X) x :=
  continuous_subtype_val.continuousAt


theorem Subtype.dense_iff {s : Set X} {t : Set s} : Dense t ↔ s ⊆ closure ((↑) '' t) := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    t : Set ↑s
    ⊢ Iff (Dense t) (HasSubset.Subset s (closure (Set.image Subtype.val t)))
  -/
  rw [IsInducing.subtypeVal.dense_iff, SetCoe.forall]
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    t : Set ↑s
    ⊢ Iff (∀ (x : X) (h : Membership.mem s x), Membership.mem (closure (Set.image  …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem map_nhds_subtype_val {s : Set X} (x : s) : map ((↑) : s → X) (𝓝 x) = 𝓝[s] ↑x := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    x : ↑s
    ⊢ Eq (Filter.map Subtype.val (nhds x)) (nhdsWithin (↑x) s)
  -/
  rw [IsInducing.subtypeVal.map_nhds_eq, Subtype.range_val]
  /-
    🎉 no goals
  -/


theorem map_nhds_subtype_coe_eq_nhds {x : X} (hx : p x) (h : ∀ᶠ x in 𝓝 x, p x) :
    map ((↑) : Subtype p → X) (𝓝 ⟨x, hx⟩) = 𝓝 x :=
                                /-
                                  X : Type u
                                  inst✝ : TopologicalSpace X
                                  p : X → Prop
                                  x : X
                                  hx : p x
                                  h : Filter.Eventually (fun x => p x) (nhds x)
                                  ⊢ Membership.mem (nhds ↑⟨x, hx⟩) (Set.range Subtype.val)
                                -/
  map_nhds_induced_of_mem <| by rw [Subtype.range_val]; exact h
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem nhds_subtype_eq_comap {x : X} {h : p x} : 𝓝 (⟨x, h⟩ : Subtype p) = comap (↑) (𝓝 x) :=
  nhds_induced _ _


theorem tendsto_subtype_rng {Y : Type*} {p : X → Prop} {l : Filter Y} {f : Y → Subtype p} :
    ∀ {x : Subtype p}, Tendsto f l (𝓝 x) ↔ Tendsto (fun x => (f x : X)) l (𝓝 (x : X))
                  /-
                    X : Type u
                    inst✝ : TopologicalSpace X
                    Y : Type u_5
                    p : X → Prop
                    l : Filter Y
                    f : Y → Subtype p
                    a : X
                    ha : p a
                    ⊢ Iff (Filter.Tendsto f l (nhds ⟨a, ha⟩)) (Filter.Tendsto (fun x => ↑(f x)) l  …
                  -/
  | ⟨a, ha⟩ => by rw [nhds_subtype_eq_comap, tendsto_comap_iff]; rfl
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


theorem closure_subtype {x : { a // p a }} {s : Set { a // p a }} :
    x ∈ closure s ↔ (x : X) ∈ closure (((↑) : _ → X) '' s) :=
  closure_induced


@[simp]
theorem continuousAt_codRestrict_iff {f : X → Y} {t : Set Y} (h1 : ∀ x, f x ∈ t) {x : X} :
    ContinuousAt (codRestrict f t h1) x ↔ ContinuousAt f x :=
  IsInducing.subtypeVal.continuousAt_iff


alias ⟨_, ContinuousAt.codRestrict⟩ := continuousAt_codRestrict_iff


theorem ContinuousAt.restrict {f : X → Y} {s : Set X} {t : Set Y} (h1 : MapsTo f s t) {x : s}
    (h2 : ContinuousAt f x) : ContinuousAt (h1.restrict f s t) x :=
  (h2.comp continuousAt_subtype_val).codRestrict _


theorem ContinuousAt.restrictPreimage {f : X → Y} {s : Set Y} {x : f ⁻¹' s} (h : ContinuousAt f x) :
    ContinuousAt (s.restrictPreimage f) x :=
  h.restrict _


@[continuity, fun_prop]
theorem Continuous.codRestrict {f : X → Y} {s : Set Y} (hf : Continuous f) (hs : ∀ a, f a ∈ s) :
    Continuous (s.codRestrict f hs) :=
  hf.subtype_mk hs


@[continuity, fun_prop]
theorem Continuous.restrict {f : X → Y} {s : Set X} {t : Set Y} (h1 : MapsTo f s t)
    (h2 : Continuous f) : Continuous (h1.restrict f s t) :=
  (h2.comp continuous_subtype_val).codRestrict _


@[continuity, fun_prop]
theorem Continuous.restrictPreimage {f : X → Y} {s : Set Y} (h : Continuous f) :
    Continuous (s.restrictPreimage f) :=
  h.restrict _


theorem Topology.IsInducing.codRestrict {e : X → Y} (he : IsInducing e) {s : Set Y}
    (hs : ∀ x, e x ∈ s) : IsInducing (codRestrict e s hs) :=
  he.of_comp (he.continuous.codRestrict hs) continuous_subtype_val


@[deprecated (since := "2024-10-28")] alias Inducing.codRestrict := IsInducing.codRestrict


protected lemma Topology.IsEmbedding.codRestrict {e : X → Y} (he : IsEmbedding e) (s : Set Y)
    (hs : ∀ x, e x ∈ s) : IsEmbedding (codRestrict e s hs) :=
  he.of_comp (he.continuous.codRestrict hs) continuous_subtype_val


@[deprecated (since := "2024-10-26")]
alias Embedding.codRestrict := IsEmbedding.codRestrict


protected lemma Topology.IsEmbedding.inclusion {s t : Set X} (h : s ⊆ t) :
    IsEmbedding (inclusion h) := IsEmbedding.subtypeVal.codRestrict _ _


@[deprecated (since := "2024-10-26")]
alias embedding_inclusion := IsEmbedding.inclusion


/-- Let `s, t ⊆ X` be two subsets of a topological space `X`.  If `t ⊆ s` and the topology induced
by `X`on `s` is discrete, then also the topology induces on `t` is discrete. -/
theorem DiscreteTopology.of_subset {X : Type*} [TopologicalSpace X] {s t : Set X}
    (_ : DiscreteTopology s) (ts : t ⊆ s) : DiscreteTopology t :=
  (IsEmbedding.inclusion ts).discreteTopology


/-- Let `s` be a discrete subset of a topological space. Then the preimage of `s` by
a continuous injective map is also discrete. -/
theorem DiscreteTopology.preimage_of_continuous_injective {X Y : Type*} [TopologicalSpace X]
    [TopologicalSpace Y] (s : Set Y) [DiscreteTopology s] {f : X → Y} (hc : Continuous f)
    (hinj : Function.Injective f) : DiscreteTopology (f ⁻¹' s) :=
  DiscreteTopology.of_continuous_injective (β := s) (Continuous.restrict
        /-
          X : Type u_5
          Y : Type u_6
          inst✝² : TopologicalSpace X
          inst✝¹ : TopologicalSpace Y
          s : Set Y
          inst✝ : DiscreteTopology ↑s
          f : X → Y
          hc : Continuous f
          hinj : Function.Injective f
          ⊢ Set.MapsTo f (Set.preimage f s) s
        -/
    (by exact fun _ x ↦ x) hc) ((MapsTo.restrict_inj _).mpr hinj.injOn)
        /-
          🎉 no goals
        -/


/-- If `f : X → Y` is a quotient map,
then its restriction to the preimage of an open set is a quotient map too. -/
theorem Topology.IsQuotientMap.restrictPreimage_isOpen {f : X → Y} (hf : IsQuotientMap f)
    {s : Set Y} (hs : IsOpen s) : IsQuotientMap (s.restrictPreimage f) := by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    f : X → Y
    hf : Topology.IsQuotientMap f
    s : Set Y
    hs : IsOpen s
    ⊢ Topology.IsQuotientMap (s.restrictPreimage f)
  -/
  refine isQuotientMap_iff.2 ⟨hf.surjective.restrictPreimage _, fun U ↦ ?_⟩
  rw [hs.isOpenEmbedding_subtypeVal.isOpen_iff_image_isOpen, ← hf.isOpen_preimage,
    (hs.preimage hf.continuous).isOpenEmbedding_subtypeVal.isOpen_iff_image_isOpen,
    image_val_preimage_restrictPreimage]


@[deprecated (since := "2024-10-22")]
alias QuotientMap.restrictPreimage_isOpen := IsQuotientMap.restrictPreimage_isOpen


open scoped Set.Notation in
lemma isClosed_preimage_val {s t : Set X} : IsClosed (s ↓∩ t) ↔ s ∩ closure (s ∩ t) ⊆ t := by
  rw [← closure_eq_iff_isClosed, IsEmbedding.subtypeVal.closure_eq_preimage_closure_image,
    ← Subtype.val_injective.image_injective.eq_iff, Subtype.image_preimage_coe,
    Subtype.image_preimage_coe, subset_antisymm_iff, and_iff_left, Set.subset_inter_iff,
    and_iff_right]
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s t : Set X
    ⊢ HasSubset.Subset (Inter.inter s (closure (Inter.inter s t))) s
  -/
  exacts [Set.inter_subset_left, Set.subset_inter Set.inter_subset_left subset_closure]
  /-
    🎉 no goals
  -/


theorem frontier_inter_open_inter {s t : Set X} (ht : IsOpen t) :
    frontier (s ∩ t) ∩ t = frontier s ∩ t := by
  simp only [Set.inter_comm _ t, ← Subtype.preimage_coe_eq_preimage_coe_iff,
    ht.isOpenMap_subtype_val.preimage_frontier_eq_frontier_preimage continuous_subtype_val,
    Subtype.preimage_coe_self_inter]


theorem isQuotientMap_quot_mk : IsQuotientMap (@Quot.mk X r) :=
  ⟨Quot.exists_rep, rfl⟩


@[deprecated (since := "2024-10-22")]
alias quotientMap_quot_mk := isQuotientMap_quot_mk


@[continuity, fun_prop]
theorem continuous_quot_mk : Continuous (@Quot.mk X r) :=
  continuous_coinduced_rng


@[continuity, fun_prop]
theorem continuous_quot_lift {f : X → Y} (hr : ∀ a b, r a b → f a = f b) (h : Continuous f) :
    Continuous (Quot.lift f hr : Quot r → Y) :=
  continuous_coinduced_dom.2 h


theorem isQuotientMap_quotient_mk' : IsQuotientMap (@Quotient.mk' X s) :=
  isQuotientMap_quot_mk


@[deprecated (since := "2024-10-22")]
alias quotientMap_quotient_mk' := isQuotientMap_quotient_mk'


theorem continuous_quotient_mk' : Continuous (@Quotient.mk' X s) :=
  continuous_coinduced_rng


theorem Continuous.quotient_lift {f : X → Y} (h : Continuous f) (hs : ∀ a b, a ≈ b → f a = f b) :
    Continuous (Quotient.lift f hs : Quotient s → Y) :=
  continuous_coinduced_dom.2 h


theorem Continuous.quotient_liftOn' {f : X → Y} (h : Continuous f)
    (hs : ∀ a b, s a b → f a = f b) :
    Continuous (fun x => Quotient.liftOn' x f hs : Quotient s → Y) :=
  h.quotient_lift hs


@[continuity, fun_prop]
theorem Continuous.quotient_map' {t : Setoid Y} {f : X → Y} (hf : Continuous f)
    (H : (s.r ⇒ t.r) f f) : Continuous (Quotient.map' f H) :=
  (continuous_quotient_mk'.comp hf).quotient_lift _


theorem continuous_pi_iff : Continuous f ↔ ∀ i, Continuous fun a => f a i := by
  /-
    X : Type u
    ι : Type u_5
    π : ι → Type u_6
    inst✝ : TopologicalSpace X
    T : (i : ι) → TopologicalSpace (π i)
    f : X → (i : ι) → π i
    ⊢ Iff (Continuous f) (∀ (i : ι), Continuous fun a => f a i)
  -/
  simp only [continuous_iInf_rng, continuous_induced_rng, comp_def]
  /-
    🎉 no goals
  -/


@[continuity, fun_prop]
theorem continuous_pi (h : ∀ i, Continuous fun a => f a i) : Continuous f :=
  continuous_pi_iff.2 h


@[continuity, fun_prop]
theorem continuous_apply (i : ι) : Continuous fun p : ∀ i, π i => p i :=
  continuous_iInf_dom continuous_induced_dom


@[continuity]
theorem continuous_apply_apply {ρ : κ → ι → Type*} [∀ j i, TopologicalSpace (ρ j i)] (j : κ)
    (i : ι) : Continuous fun p : ∀ j, ∀ i, ρ j i => p j i :=
  (continuous_apply i).comp (continuous_apply j)


theorem continuousAt_apply (i : ι) (x : ∀ i, π i) : ContinuousAt (fun p : ∀ i, π i => p i) x :=
  (continuous_apply i).continuousAt


theorem Filter.Tendsto.apply_nhds {l : Filter Y} {f : Y → ∀ i, π i} {x : ∀ i, π i}
    (h : Tendsto f l (𝓝 x)) (i : ι) : Tendsto (fun a => f a i) l (𝓝 <| x i) :=
  (continuousAt_apply i _).tendsto.comp h


@[fun_prop]
protected theorem Continuous.piMap {Y : ι → Type*} [∀ i, TopologicalSpace (Y i)]
    {f : ∀ i, π i → Y i} (hf : ∀ i, Continuous (f i)) : Continuous (Pi.map f) :=
  continuous_pi fun i ↦ (hf i).comp (continuous_apply i)


theorem nhds_pi {a : ∀ i, π i} : 𝓝 a = pi fun i => 𝓝 (a i) := by
  /-
    ι : Type u_5
    π : ι → Type u_6
    T : (i : ι) → TopologicalSpace (π i)
    a : (i : ι) → π i
    ⊢ Eq (nhds a) (Filter.pi fun i => nhds (a i))
  -/
  simp only [nhds_iInf, nhds_induced, Filter.pi]
  /-
    🎉 no goals
  -/


protected theorem IsOpenMap.piMap {Y : ι → Type*} [∀ i, TopologicalSpace (Y i)] {f : ∀ i, π i → Y i}
    (hfo : ∀ i, IsOpenMap (f i)) (hsurj : ∀ᶠ i in cofinite, Surjective (f i)) :
    IsOpenMap (Pi.map f) := by
  /-
    ι : Type u_5
    π : ι → Type u_6
    T : (i : ι) → TopologicalSpace (π i)
    Y : ι → Type u_8
    inst✝ : (i : ι) → TopologicalSpace (Y i)
    f : (i : ι) → π i → Y i
    hfo : ∀ (i : ι), IsOpenMap (f i)
    hsurj : Filter.Eventually (fun i => Function.Surjective (f i)) Filter.cofinite
    ⊢ IsOpenMap (Pi.map f)
  -/
  refine IsOpenMap.of_nhds_le fun x ↦ ?_
  /-
    ι : Type u_5
    π : ι → Type u_6
    T : (i : ι) → TopologicalSpace (π i)
    Y : ι → Type u_8
    inst✝ : (i : ι) → TopologicalSpace (Y i)
    f : (i : ι) → π i → Y i
    hfo : ∀ (i : ι), IsOpenMap (f i)
    hsurj : Filter.Eventually (fun i => Function.Surjective (f i)) Filter.cofinite
    x : (i : ι) → π i
    ⊢ LE.le (nhds (Pi.map f x)) (Filter.map (Pi.map f) (nhds x))
  -/
  rw [nhds_pi, nhds_pi, map_piMap_pi hsurj]
  /-
    ι : Type u_5
    π : ι → Type u_6
    T : (i : ι) → TopologicalSpace (π i)
    Y : ι → Type u_8
    inst✝ : (i : ι) → TopologicalSpace (Y i)
    f : (i : ι) → π i → Y i
    hfo : ∀ (i : ι), IsOpenMap (f i)
    hsurj : Filter.Eventually (fun i => Function.Surjective (f i)) Filter.cofinite
    x : (i : ι) → π i
    ⊢ LE.le (Filter.pi fun i => nhds (Pi.map f x i)) (Filter.pi fun i => Filter.ma …
  -/
  exact Filter.pi_mono fun i ↦ (hfo i).nhds_le _
  /-
    🎉 no goals
  -/


protected theorem IsOpenQuotientMap.piMap {Y : ι → Type*} [∀ i, TopologicalSpace (Y i)]
    {f : ∀ i, π i → Y i} (hf : ∀ i, IsOpenQuotientMap (f i)) : IsOpenQuotientMap (Pi.map f) :=
  ⟨.piMap fun i ↦ (hf i).1, .piMap fun i ↦ (hf i).2, .piMap (fun i ↦ (hf i).3) <|
    .of_forall fun i ↦ (hf i).1⟩


theorem tendsto_pi_nhds {f : Y → ∀ i, π i} {g : ∀ i, π i} {u : Filter Y} :
    Tendsto f u (𝓝 g) ↔ ∀ x, Tendsto (fun i => f i x) u (𝓝 (g x)) := by
  /-
    Y : Type v
    ι : Type u_5
    π : ι → Type u_6
    T : (i : ι) → TopologicalSpace (π i)
    f : Y → (i : ι) → π i
    g : (i : ι) → π i
    u : Filter Y
    ⊢ Iff (Filter.Tendsto f u (nhds g)) (∀ (x : ι), Filter.Tendsto (fun i => f i x …
  -/
  rw [nhds_pi, Filter.tendsto_pi]
  /-
    🎉 no goals
  -/


theorem continuousAt_pi {f : X → ∀ i, π i} {x : X} :
    ContinuousAt f x ↔ ∀ i, ContinuousAt (fun y => f y i) x :=
  tendsto_pi_nhds


@[fun_prop]
theorem continuousAt_pi' {f : X → ∀ i, π i} {x : X} (hf : ∀ i, ContinuousAt (fun y => f y i) x) :
    ContinuousAt f x :=
  continuousAt_pi.2 hf


@[fun_prop]
protected theorem ContinuousAt.piMap {Y : ι → Type*} [∀ i, TopologicalSpace (Y i)]
    {f : ∀ i, π i → Y i} {x : ∀ i, π i} (hf : ∀ i, ContinuousAt (f i) (x i)) :
    ContinuousAt (Pi.map f) x :=
  continuousAt_pi.2 fun i ↦ (hf i).comp (continuousAt_apply i x)


theorem Pi.continuous_precomp' {ι' : Type*} (φ : ι' → ι) :
    Continuous (fun (f : (∀ i, π i)) (j : ι') ↦ f (φ j)) :=
  continuous_pi fun j ↦ continuous_apply (φ j)


theorem Pi.continuous_precomp {ι' : Type*} (φ : ι' → ι) :
    Continuous (· ∘ φ : (ι → X) → (ι' → X)) :=
  Pi.continuous_precomp' φ


theorem Pi.continuous_postcomp' {X : ι → Type*} [∀ i, TopologicalSpace (X i)]
    {g : ∀ i, π i → X i} (hg : ∀ i, Continuous (g i)) :
    Continuous (fun (f : (∀ i, π i)) (i : ι) ↦ g i (f i)) :=
  continuous_pi fun i ↦ (hg i).comp <| continuous_apply i


theorem Pi.continuous_postcomp [TopologicalSpace Y] {g : X → Y} (hg : Continuous g) :
    Continuous (g ∘ · : (ι → X) → (ι → Y)) :=
  Pi.continuous_postcomp' fun _ ↦ hg


lemma Pi.induced_precomp' {ι' : Type*} (φ : ι' → ι) :
    induced (fun (f : (∀ i, π i)) (j : ι') ↦ f (φ j)) Pi.topologicalSpace =
    ⨅ i', induced (eval (φ i')) (T (φ i')) := by
  /-
    ι : Type u_5
    π : ι → Type u_6
    T : (i : ι) → TopologicalSpace (π i)
    ι' : Type u_8
    φ : ι' → ι
    ⊢ Eq (TopologicalSpace.induced (fun f j => f (φ j)) Pi.topologicalSpace) (iInf …
  -/
  simp [Pi.topologicalSpace, induced_iInf, induced_compose, comp_def]
  /-
    🎉 no goals
  -/


lemma Pi.induced_precomp [TopologicalSpace Y] {ι' : Type*} (φ : ι' → ι) :
    induced (· ∘ φ) Pi.topologicalSpace =
    ⨅ i', induced (eval (φ i')) ‹TopologicalSpace Y› :=
  induced_precomp' φ


@[continuity, fun_prop]
lemma Pi.continuous_restrict (S : Set ι) :
    Continuous (S.restrict : (∀ i : ι, π i) → (∀ i : S, π i)) :=
  Pi.continuous_precomp' ((↑) : S → ι)


@[continuity, fun_prop]
lemma Pi.continuous_restrict₂ {s t : Set ι} (hst : s ⊆ t) : Continuous (restrict₂ (π := π) hst) :=
  continuous_pi fun _ ↦ continuous_apply _


@[continuity, fun_prop]
theorem Finset.continuous_restrict (s : Finset ι) : Continuous (s.restrict (π := π)) :=
  continuous_pi fun _ ↦ continuous_apply _


@[continuity, fun_prop]
theorem Finset.continuous_restrict₂ {s t : Finset ι} (hst : s ⊆ t) :
    Continuous (Finset.restrict₂ (π := π) hst) :=
  continuous_pi fun _ ↦ continuous_apply _


@[continuity, fun_prop]
theorem Pi.continuous_restrict_apply (s : Set X) {f : X → Z} (hf : Continuous f) :
    Continuous (s.restrict f) := hf.comp continuous_subtype_val


@[continuity, fun_prop]
theorem Pi.continuous_restrict₂_apply {s t : Set X} (hst : s ⊆ t)
    {f : t → Z} (hf : Continuous f) :
    Continuous (restrict₂ (π := fun _ ↦ Z) hst f) := hf.comp (continuous_inclusion hst)


@[continuity, fun_prop]
theorem Finset.continuous_restrict_apply (s : Finset X) {f : X → Z} (hf : Continuous f) :
    Continuous (s.restrict f) := hf.comp continuous_subtype_val


@[continuity, fun_prop]
theorem Finset.continuous_restrict₂_apply {s t : Finset X} (hst : s ⊆ t)
    {f : t → Z} (hf : Continuous f) :
    Continuous (restrict₂ (π := fun _ ↦ Z) hst f) := hf.comp (continuous_inclusion hst)


lemma Pi.induced_restrict (S : Set ι) :
    induced (S.restrict) Pi.topologicalSpace =
    ⨅ i ∈ S, induced (eval i) (T i) := by
  simp (config := { unfoldPartialApp := true }) [← iInf_subtype'', ← induced_precomp' ((↑) : S → ι),
    restrict]


lemma Pi.induced_restrict_sUnion (𝔖 : Set (Set ι)) :
    induced (⋃₀ 𝔖).restrict (Pi.topologicalSpace (Y := fun i : (⋃₀ 𝔖) ↦ π i)) =
    ⨅ S ∈ 𝔖, induced S.restrict Pi.topologicalSpace := by
  /-
    ι : Type u_5
    π : ι → Type u_6
    T : (i : ι) → TopologicalSpace (π i)
    𝔖 : Set (Set ι)
    ⊢ Eq (TopologicalSpace.induced 𝔖.sUnion.restrict Pi.topologicalSpace) (iInf fu …
  -/
  simp_rw [Pi.induced_restrict, iInf_sUnion]
  /-
    🎉 no goals
  -/


theorem Filter.Tendsto.update [DecidableEq ι] {l : Filter Y} {f : Y → ∀ i, π i} {x : ∀ i, π i}
    (hf : Tendsto f l (𝓝 x)) (i : ι) {g : Y → π i} {xi : π i} (hg : Tendsto g l (𝓝 xi)) :
    Tendsto (fun a => update (f a) i (g a)) l (𝓝 <| update x i xi) :=
                                /-
                                  Y : Type v
                                  ι : Type u_5
                                  π : ι → Type u_6
                                  T : (i : ι) → TopologicalSpace (π i)
                                  inst✝ : DecidableEq ι
                                  l : Filter Y
                                  f : Y → (i : ι) → π i
                                  x : (i : ι) → π i
                                  hf : Filter.Tendsto f l (nhds x)
                                  i : ι
                                  g : Y → π i
                                  xi : π i
                                  hg : Filter.Tendsto g l (nhds xi)
                                  j : ι
                                  ⊢ Filter.Tendsto (fun i_1 => Function.update (f i_1) i (g i_1) j) l (nhds (Fun …
                                -/
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
  tendsto_pi_nhds.2 fun j => by rcases eq_or_ne j i with (rfl | hj) <;> simp [*, hf.apply_nhds]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


theorem ContinuousAt.update [DecidableEq ι] {x : X} (hf : ContinuousAt f x) (i : ι) {g : X → π i}
    (hg : ContinuousAt g x) : ContinuousAt (fun a => update (f a) i (g a)) x :=
  hf.tendsto.update i hg


theorem Continuous.update [DecidableEq ι] (hf : Continuous f) (i : ι) {g : X → π i}
    (hg : Continuous g) : Continuous fun a => update (f a) i (g a) :=
  continuous_iff_continuousAt.2 fun _ => hf.continuousAt.update i hg.continuousAt


/-- `Function.update f i x` is continuous in `(f, x)`. -/
@[continuity, fun_prop]
theorem continuous_update [DecidableEq ι] (i : ι) :
    Continuous fun f : (∀ j, π j) × π i => update f.1 i f.2 :=
  continuous_fst.update i continuous_snd


/-- `Pi.mulSingle i x` is continuous in `x`. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: restore @[continuity]
@[to_additive "`Pi.single i x` is continuous in `x`."]
theorem continuous_mulSingle [∀ i, One (π i)] [DecidableEq ι] (i : ι) :
    Continuous fun x => (Pi.mulSingle i x : ∀ i, π i) :=
  continuous_const.update _ continuous_id


theorem Filter.Tendsto.finCons
    {f : Y → π 0} {g : Y → ∀ j : Fin n, π j.succ} {l : Filter Y} {x : π 0} {y : ∀ j, π (Fin.succ j)}
    (hf : Tendsto f l (𝓝 x)) (hg : Tendsto g l (𝓝 y)) :
    Tendsto (fun a => Fin.cons (f a) (g a)) l (𝓝 <| Fin.cons x y) :=
                                           /-
                                             Y : Type v
                                             n : Nat
                                             π : Fin (HAdd.hAdd n 1) → Type u_8
                                             inst✝ : (i : Fin (HAdd.hAdd n 1)) → TopologicalSpace (π i)
                                             f : Y → π 0
                                             g : Y → (j : Fin n) → π j.succ
                                             l : Filter Y
                                             x : π 0
                                             y : (j : Fin n) → π j.succ
                                             hf : Filter.Tendsto f l (nhds x)
                                             hg : Filter.Tendsto g l (nhds y)
                                             j : Fin (HAdd.hAdd n 1)
                                             ⊢ Filter.Tendsto (fun i => Fin.cons (f i) (g i) 0) l (nhds (Fin.cons x y 0))
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
  tendsto_pi_nhds.2 fun j => Fin.cases (by simpa) (by simpa using tendsto_pi_nhds.1 hg) j
                                                      /-
                                                        🎉 no goals
                                                      -/


theorem ContinuousAt.finCons {f : X → π 0} {g : X → ∀ j : Fin n, π (Fin.succ j)} {x : X}
    (hf : ContinuousAt f x) (hg : ContinuousAt g x) :
    ContinuousAt (fun a => Fin.cons (f a) (g a)) x :=
  hf.tendsto.finCons hg


theorem Continuous.finCons {f : X → π 0} {g : X → ∀ j : Fin n, π (Fin.succ j)}
    (hf : Continuous f) (hg : Continuous g) : Continuous fun a => Fin.cons (f a) (g a) :=
  continuous_iff_continuousAt.2 fun _ => hf.continuousAt.finCons hg.continuousAt


theorem Filter.Tendsto.matrixVecCons
    {f : Y → Z} {g : Y → Fin n → Z} {l : Filter Y} {x : Z} {y : Fin n → Z}
    (hf : Tendsto f l (𝓝 x)) (hg : Tendsto g l (𝓝 y)) :
    Tendsto (fun a => Matrix.vecCons (f a) (g a)) l (𝓝 <| Matrix.vecCons x y) :=
  hf.finCons hg


theorem ContinuousAt.matrixVecCons
    {f : X → Z} {g : X → Fin n → Z} {x : X} (hf : ContinuousAt f x) (hg : ContinuousAt g x) :
    ContinuousAt (fun a => Matrix.vecCons (f a) (g a)) x :=
  hf.finCons hg


theorem Continuous.matrixVecCons
    {f : X → Z} {g : X → Fin n → Z} (hf : Continuous f) (hg : Continuous g) :
    Continuous fun a => Matrix.vecCons (f a) (g a) :=
  hf.finCons hg


theorem Filter.Tendsto.finSnoc
    {f : Y → ∀ j : Fin n, π j.castSucc} {g : Y → π (Fin.last _)}
    {l : Filter Y} {x : ∀ j, π (Fin.castSucc j)} {y : π (Fin.last _)}
    (hf : Tendsto f l (𝓝 x)) (hg : Tendsto g l (𝓝 y)) :
    Tendsto (fun a => Fin.snoc (f a) (g a)) l (𝓝 <| Fin.snoc x y) :=
                                               /-
                                                 Y : Type v
                                                 n : Nat
                                                 π : Fin (HAdd.hAdd n 1) → Type u_8
                                                 inst✝ : (i : Fin (HAdd.hAdd n 1)) → TopologicalSpace (π i)
                                                 f : Y → (j : Fin n) → π j.castSucc
                                                 g : Y → π (Fin.last n)
                                                 l : Filter Y
                                                 x : (j : Fin n) → π j.castSucc
                                                 y : π (Fin.last n)
                                                 hf : Filter.Tendsto f l (nhds x)
                                                 hg : Filter.Tendsto g l (nhds y)
                                                 j : Fin (HAdd.hAdd n 1)
                                                 ⊢ Filter.Tendsto (fun i => Fin.snoc (f i) (g i) (Fin.last n)) l (nhds (Fin.sno …
                                               -/
                                               /-
                                                 🎉 no goals
                                               -/
  tendsto_pi_nhds.2 fun j => Fin.lastCases (by simpa) (by simpa using tendsto_pi_nhds.1 hf) j
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem ContinuousAt.finSnoc {f : X → ∀ j : Fin n, π j.castSucc} {g : X → π (Fin.last _)} {x : X}
    (hf : ContinuousAt f x) (hg : ContinuousAt g x) :
    ContinuousAt (fun a => Fin.snoc (f a) (g a)) x :=
  hf.tendsto.finSnoc hg


theorem Continuous.finSnoc {f : X → ∀ j : Fin n, π j.castSucc} {g : X → π (Fin.last _)}
    (hf : Continuous f) (hg : Continuous g) : Continuous fun a => Fin.snoc (f a) (g a) :=
  continuous_iff_continuousAt.2 fun _ => hf.continuousAt.finSnoc hg.continuousAt


theorem Filter.Tendsto.finInsertNth
    (i : Fin (n + 1)) {f : Y → π i} {g : Y → ∀ j : Fin n, π (i.succAbove j)} {l : Filter Y}
    {x : π i} {y : ∀ j, π (i.succAbove j)} (hf : Tendsto f l (𝓝 x)) (hg : Tendsto g l (𝓝 y)) :
    Tendsto (fun a => i.insertNth (f a) (g a)) l (𝓝 <| i.insertNth x y) :=
                                                      /-
                                                        Y : Type v
                                                        n : Nat
                                                        π : Fin (HAdd.hAdd n 1) → Type u_8
                                                        inst✝ : (i : Fin (HAdd.hAdd n 1)) → TopologicalSpace (π i)
                                                        i : Fin (HAdd.hAdd n 1)
                                                        f : Y → π i
                                                        g : Y → (j : Fin n) → π (i.succAbove j)
                                                        l : Filter Y
                                                        x : π i
                                                        y : (j : Fin n) → π (i.succAbove j)
                                                        hf : Filter.Tendsto f l (nhds x)
                                                        hg : Filter.Tendsto g l (nhds y)
                                                        j : Fin (HAdd.hAdd n 1)
                                                        ⊢ Filter.Tendsto (fun i_1 => i.insertNth (f i_1) (g i_1) i) l (nhds (i.insertN …
                                                      -/
                                                      /-
                                                        🎉 no goals
                                                      -/
  tendsto_pi_nhds.2 fun j => Fin.succAboveCases i (by simpa) (by simpa using tendsto_pi_nhds.1 hg) j
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[deprecated (since := "2025-01-02")]
alias Filter.Tendsto.fin_insertNth := Filter.Tendsto.finInsertNth


theorem ContinuousAt.finInsertNth
    (i : Fin (n + 1)) {f : X → π i} {g : X → ∀ j : Fin n, π (i.succAbove j)} {x : X}
    (hf : ContinuousAt f x) (hg : ContinuousAt g x) :
    ContinuousAt (fun a => i.insertNth (f a) (g a)) x :=
  hf.tendsto.finInsertNth i hg


@[deprecated (since := "2025-01-02")]
alias ContinuousAt.fin_insertNth := ContinuousAt.finInsertNth


theorem Continuous.finInsertNth
    (i : Fin (n + 1)) {f : X → π i} {g : X → ∀ j : Fin n, π (i.succAbove j)}
    (hf : Continuous f) (hg : Continuous g) : Continuous fun a => i.insertNth (f a) (g a) :=
  continuous_iff_continuousAt.2 fun _ => hf.continuousAt.finInsertNth i hg.continuousAt


@[deprecated (since := "2025-01-02")]
alias Continuous.fin_insertNth := Continuous.finInsertNth


theorem isOpen_set_pi {i : Set ι} {s : ∀ a, Set (π a)} (hi : i.Finite)
    (hs : ∀ a ∈ i, IsOpen (s a)) : IsOpen (pi i s) := by
  /-
    ι : Type u_5
    π : ι → Type u_6
    T : (i : ι) → TopologicalSpace (π i)
    i : Set ι
    s : (a : ι) → Set (π a)
    hi : i.Finite
    hs : ∀ (a : ι), Membership.mem i a → IsOpen (s a)
    ⊢ IsOpen (i.pi s)
  -/
  rw [pi_def]; exact hi.isOpen_biInter fun a ha => (hs _ ha).preimage (continuous_apply _)
               /-
                 🎉 no goals
               -/


theorem isOpen_pi_iff {s : Set (∀ a, π a)} :
    IsOpen s ↔
      ∀ f, f ∈ s → ∃ (I : Finset ι) (u : ∀ a, Set (π a)),
        (∀ a, a ∈ I → IsOpen (u a) ∧ f a ∈ u a) ∧ (I : Set ι).pi u ⊆ s := by
  /-
    ι : Type u_5
    π : ι → Type u_6
    T : (i : ι) → TopologicalSpace (π i)
    s : Set ((a : ι) → π a)
    ⊢ Iff (IsOpen s) (∀ (f : (a : ι) → π a), Membership.mem s f → Exists fun I =>  …
  -/
  rw [isOpen_iff_nhds]
  /-
    ι : Type u_5
    π : ι → Type u_6
    T : (i : ι) → TopologicalSpace (π i)
    s : Set ((a : ι) → π a)
    ⊢ Iff (∀ (x : (a : ι) → π a), Membership.mem s x → LE.le (nhds x) (Filter.prin …
  -/
  simp_rw [le_principal_iff, nhds_pi, Filter.mem_pi', mem_nhds_iff]
  /-
    ι : Type u_5
    π : ι → Type u_6
    T : (i : ι) → TopologicalSpace (π i)
    s : Set ((a : ι) → π a)
    ⊢ Iff (∀ (x : (a : ι) → π a), Membership.mem s x → Exists fun I => Exists fun  …
  -/
  refine forall₂_congr fun a _ => ⟨?_, ?_⟩
    /-
      case refine_1
      ι : Type u_5
      π : ι → Type u_6
      T : (i : ι) → TopologicalSpace (π i)
      s : Set ((a : ι) → π a)
      a : (a : ι) → π a
      x✝ : Membership.mem s a
      ⊢ (Exists fun I => Exists fun t => And (∀ (i : ι), Exists fun t_1 => And (HasS …
    -/
  · rintro ⟨I, t, ⟨h1, h2⟩⟩
    /-
      case refine_1.intro.intro.intro
      ι : Type u_5
      π : ι → Type u_6
      T : (i : ι) → TopologicalSpace (π i)
      s : Set ((a : ι) → π a)
      a : (a : ι) → π a
      x✝ : Membership.mem s a
      I : Finset ι
      t : (i : ι) → Set (π i)
      h1 : ∀ (i : ι), Exists fun t_1 => And (HasSubset.Subset t_1 (t i)) (And (IsOpe …
      h2 : HasSubset.Subset ((↑I).pi t) s
      ⊢ Exists fun I => Exists fun u => And (∀ (a_1 : ι), Membership.mem I a_1 → And …
    -/
    refine ⟨I, fun a => eval a '' (I : Set ι).pi fun a => (h1 a).choose, fun i hi => ?_, ?_⟩
    · simp_rw [eval_image_pi (Finset.mem_coe.mpr hi)
          (pi_nonempty_iff.mpr fun i => ⟨_, fun _ => (h1 i).choose_spec.2.2⟩)]
      /-
        case refine_1.intro.intro.intro.refine_1
        ι : Type u_5
        π : ι → Type u_6
        T : (i : ι) → TopologicalSpace (π i)
        s : Set ((a : ι) → π a)
        a : (a : ι) → π a
        x✝ : Membership.mem s a
        I : Finset ι
        t : (i : ι) → Set (π i)
        h1 : ∀ (i : ι), Exists fun t_1 => And (HasSubset.Subset t_1 (t i)) (And (IsOpe …
        h2 : HasSubset.Subset ((↑I).pi t) s
        i : ι
        hi : Membership.mem I i
        ⊢ And (IsOpen ⋯.choose) (Membership.mem ⋯.choose (a i))
      -/
      exact (h1 i).choose_spec.2
      /-
        🎉 no goals
      -/
    · exact Subset.trans
        (pi_mono fun i hi => (eval_image_pi_subset hi).trans (h1 i).choose_spec.1) h2
    /-
      case refine_2
      ι : Type u_5
      π : ι → Type u_6
      T : (i : ι) → TopologicalSpace (π i)
      s : Set ((a : ι) → π a)
      a : (a : ι) → π a
      x✝ : Membership.mem s a
      ⊢ (Exists fun I => Exists fun u => And (∀ (a_1 : ι), Membership.mem I a_1 → An …
    -/
  · rintro ⟨I, t, ⟨h1, h2⟩⟩
    classical
    refine ⟨I, fun a => ite (a ∈ I) (t a) univ, fun i => ?_, ?_⟩
    · by_cases hi : i ∈ I
      · use t i
        simp_rw [if_pos hi]
        exact ⟨Subset.rfl, (h1 i) hi⟩
      · use univ
        simp_rw [if_neg hi]
        exact ⟨Subset.rfl, isOpen_univ, mem_univ _⟩
    · rw [← univ_pi_ite]
      simp only [← ite_and, ← Finset.mem_coe, and_self_iff, univ_pi_ite, h2]


theorem isOpen_pi_iff' [Finite ι] {s : Set (∀ a, π a)} :
    IsOpen s ↔
      ∀ f, f ∈ s → ∃ u : ∀ a, Set (π a), (∀ a, IsOpen (u a) ∧ f a ∈ u a) ∧ univ.pi u ⊆ s := by
  /-
    ι : Type u_5
    π : ι → Type u_6
    T : (i : ι) → TopologicalSpace (π i)
    inst✝ : Finite ι
    s : Set ((a : ι) → π a)
    ⊢ Iff (IsOpen s) (∀ (f : (a : ι) → π a), Membership.mem s f → Exists fun u =>  …
  -/
  cases nonempty_fintype ι
  /-
    case intro
    ι : Type u_5
    π : ι → Type u_6
    T : (i : ι) → TopologicalSpace (π i)
    inst✝ : Finite ι
    s : Set ((a : ι) → π a)
    val✝ : Fintype ι
    ⊢ Iff (IsOpen s) (∀ (f : (a : ι) → π a), Membership.mem s f → Exists fun u =>  …
  -/
  rw [isOpen_iff_nhds]
  /-
    case intro
    ι : Type u_5
    π : ι → Type u_6
    T : (i : ι) → TopologicalSpace (π i)
    inst✝ : Finite ι
    s : Set ((a : ι) → π a)
    val✝ : Fintype ι
    ⊢ Iff (∀ (x : (a : ι) → π a), Membership.mem s x → LE.le (nhds x) (Filter.prin …
  -/
  simp_rw [le_principal_iff, nhds_pi, Filter.mem_pi', mem_nhds_iff]
  /-
    case intro
    ι : Type u_5
    π : ι → Type u_6
    T : (i : ι) → TopologicalSpace (π i)
    inst✝ : Finite ι
    s : Set ((a : ι) → π a)
    val✝ : Fintype ι
    ⊢ Iff (∀ (x : (a : ι) → π a), Membership.mem s x → Exists fun I => Exists fun  …
  -/
  refine forall₂_congr fun a _ => ⟨?_, ?_⟩
    /-
      case intro.refine_1
      ι : Type u_5
      π : ι → Type u_6
      T : (i : ι) → TopologicalSpace (π i)
      inst✝ : Finite ι
      s : Set ((a : ι) → π a)
      val✝ : Fintype ι
      a : (a : ι) → π a
      x✝ : Membership.mem s a
      ⊢ (Exists fun I => Exists fun t => And (∀ (i : ι), Exists fun t_1 => And (HasS …
    -/
  · rintro ⟨I, t, ⟨h1, h2⟩⟩
    refine
      ⟨fun i => (h1 i).choose,
        ⟨fun i => (h1 i).choose_spec.2,
          (pi_mono fun i _ => (h1 i).choose_spec.1).trans (Subset.trans ?_ h2)⟩⟩
    /-
      case intro.refine_1.intro.intro.intro
      ι : Type u_5
      π : ι → Type u_6
      T : (i : ι) → TopologicalSpace (π i)
      inst✝ : Finite ι
      s : Set ((a : ι) → π a)
      val✝ : Fintype ι
      a : (a : ι) → π a
      x✝ : Membership.mem s a
      I : Finset ι
      t : (i : ι) → Set (π i)
      h1 : ∀ (i : ι), Exists fun t_1 => And (HasSubset.Subset t_1 (t i)) (And (IsOpe …
      h2 : HasSubset.Subset ((↑I).pi t) s
      ⊢ HasSubset.Subset (Set.univ.pi t) ((↑I).pi t)
    -/
    rw [← pi_inter_compl (I : Set ι)]
    /-
      case intro.refine_1.intro.intro.intro
      ι : Type u_5
      π : ι → Type u_6
      T : (i : ι) → TopologicalSpace (π i)
      inst✝ : Finite ι
      s : Set ((a : ι) → π a)
      val✝ : Fintype ι
      a : (a : ι) → π a
      x✝ : Membership.mem s a
      I : Finset ι
      t : (i : ι) → Set (π i)
      h1 : ∀ (i : ι), Exists fun t_1 => And (HasSubset.Subset t_1 (t i)) (And (IsOpe …
      h2 : HasSubset.Subset ((↑I).pi t) s
      ⊢ HasSubset.Subset (Inter.inter ((↑I).pi t) ((HasCompl.compl ↑I).pi t)) ((↑I). …
    -/
    exact inter_subset_left
    /-
      🎉 no goals
    -/
  · exact fun ⟨u, ⟨h1, _⟩⟩ =>
      ⟨Finset.univ, u, ⟨fun i => ⟨u i, ⟨rfl.subset, h1 i⟩⟩, by rwa [Finset.coe_univ]⟩⟩


theorem isClosed_set_pi {i : Set ι} {s : ∀ a, Set (π a)} (hs : ∀ a ∈ i, IsClosed (s a)) :
    IsClosed (pi i s) := by
  /-
    ι : Type u_5
    π : ι → Type u_6
    T : (i : ι) → TopologicalSpace (π i)
    i : Set ι
    s : (a : ι) → Set (π a)
    hs : ∀ (a : ι), Membership.mem i a → IsClosed (s a)
    ⊢ IsClosed (i.pi s)
  -/
  rw [pi_def]; exact isClosed_biInter fun a ha => (hs _ ha).preimage (continuous_apply _)
               /-
                 🎉 no goals
               -/


theorem mem_nhds_of_pi_mem_nhds {I : Set ι} {s : ∀ i, Set (π i)} (a : ∀ i, π i) (hs : I.pi s ∈ 𝓝 a)
    {i : ι} (hi : i ∈ I) : s i ∈ 𝓝 (a i) := by
  /-
    ι : Type u_5
    π : ι → Type u_6
    T : (i : ι) → TopologicalSpace (π i)
    I : Set ι
    s : (i : ι) → Set (π i)
    a : (i : ι) → π i
    hs : Membership.mem (nhds a) (I.pi s)
    i : ι
    hi : Membership.mem I i
    ⊢ Membership.mem (nhds (a i)) (s i)
  -/
  rw [nhds_pi] at hs; exact mem_of_pi_mem_pi hs hi
                      /-
                        🎉 no goals
                      -/


theorem set_pi_mem_nhds {i : Set ι} {s : ∀ a, Set (π a)} {x : ∀ a, π a} (hi : i.Finite)
    (hs : ∀ a ∈ i, s a ∈ 𝓝 (x a)) : pi i s ∈ 𝓝 x := by
  /-
    ι : Type u_5
    π : ι → Type u_6
    T : (i : ι) → TopologicalSpace (π i)
    i : Set ι
    s : (a : ι) → Set (π a)
    x : (a : ι) → π a
    hi : i.Finite
    hs : ∀ (a : ι), Membership.mem i a → Membership.mem (nhds (x a)) (s a)
    ⊢ Membership.mem (nhds x) (i.pi s)
  -/
  rw [pi_def, biInter_mem hi]
  /-
    ι : Type u_5
    π : ι → Type u_6
    T : (i : ι) → TopologicalSpace (π i)
    i : Set ι
    s : (a : ι) → Set (π a)
    x : (a : ι) → π a
    hi : i.Finite
    hs : ∀ (a : ι), Membership.mem i a → Membership.mem (nhds (x a)) (s a)
    ⊢ ∀ (i_1 : ι), Membership.mem i i_1 → Membership.mem (nhds x) (Set.preimage (F …
  -/
  exact fun a ha => (continuous_apply a).continuousAt (hs a ha)
  /-
    🎉 no goals
  -/


theorem set_pi_mem_nhds_iff {I : Set ι} (hI : I.Finite) {s : ∀ i, Set (π i)} (a : ∀ i, π i) :
    I.pi s ∈ 𝓝 a ↔ ∀ i : ι, i ∈ I → s i ∈ 𝓝 (a i) := by
  /-
    ι : Type u_5
    π : ι → Type u_6
    T : (i : ι) → TopologicalSpace (π i)
    I : Set ι
    hI : I.Finite
    s : (i : ι) → Set (π i)
    a : (i : ι) → π i
    ⊢ Iff (Membership.mem (nhds a) (I.pi s)) (∀ (i : ι), Membership.mem I i → Memb …
  -/
  rw [nhds_pi, pi_mem_pi_iff hI]
  /-
    🎉 no goals
  -/


theorem interior_pi_set {I : Set ι} (hI : I.Finite) {s : ∀ i, Set (π i)} :
    interior (pi I s) = I.pi fun i => interior (s i) := by
  /-
    ι : Type u_5
    π : ι → Type u_6
    T : (i : ι) → TopologicalSpace (π i)
    I : Set ι
    hI : I.Finite
    s : (i : ι) → Set (π i)
    ⊢ Eq (interior (I.pi s)) (I.pi fun i => interior (s i))
  -/
  ext a
  /-
    case h
    ι : Type u_5
    π : ι → Type u_6
    T : (i : ι) → TopologicalSpace (π i)
    I : Set ι
    hI : I.Finite
    s : (i : ι) → Set (π i)
    a : (i : ι) → π i
    ⊢ Iff (Membership.mem (interior (I.pi s)) a) (Membership.mem (I.pi fun i => in …
  -/
  simp only [Set.mem_pi, mem_interior_iff_mem_nhds, set_pi_mem_nhds_iff hI]
  /-
    🎉 no goals
  -/


theorem exists_finset_piecewise_mem_of_mem_nhds [DecidableEq ι] {s : Set (∀ a, π a)} {x : ∀ a, π a}
    (hs : s ∈ 𝓝 x) (y : ∀ a, π a) : ∃ I : Finset ι, I.piecewise x y ∈ s := by
  /-
    ι : Type u_5
    π : ι → Type u_6
    T : (i : ι) → TopologicalSpace (π i)
    inst✝ : DecidableEq ι
    s : Set ((a : ι) → π a)
    x : (a : ι) → π a
    hs : Membership.mem (nhds x) s
    y : (a : ι) → π a
    ⊢ Exists fun I => Membership.mem s (I.piecewise x y)
  -/
  simp only [nhds_pi, Filter.mem_pi'] at hs
  /-
    ι : Type u_5
    π : ι → Type u_6
    T : (i : ι) → TopologicalSpace (π i)
    inst✝ : DecidableEq ι
    s : Set ((a : ι) → π a)
    x y : (a : ι) → π a
    hs : Exists fun I => Exists fun t => And (∀ (i : ι), Membership.mem (nhds (x i …
    ⊢ Exists fun I => Membership.mem s (I.piecewise x y)
  -/
  rcases hs with ⟨I, t, htx, hts⟩
  /-
    case intro.intro.intro
    ι : Type u_5
    π : ι → Type u_6
    T : (i : ι) → TopologicalSpace (π i)
    inst✝ : DecidableEq ι
    s : Set ((a : ι) → π a)
    x y : (a : ι) → π a
    I : Finset ι
    t : (i : ι) → Set (π i)
    htx : ∀ (i : ι), Membership.mem (nhds (x i)) (t i)
    hts : HasSubset.Subset ((↑I).pi t) s
    ⊢ Exists fun I => Membership.mem s (I.piecewise x y)
  -/
  refine ⟨I, hts fun i hi => ?_⟩
  /-
    case intro.intro.intro
    ι : Type u_5
    π : ι → Type u_6
    T : (i : ι) → TopologicalSpace (π i)
    inst✝ : DecidableEq ι
    s : Set ((a : ι) → π a)
    x y : (a : ι) → π a
    I : Finset ι
    t : (i : ι) → Set (π i)
    htx : ∀ (i : ι), Membership.mem (nhds (x i)) (t i)
    hts : HasSubset.Subset ((↑I).pi t) s
    i : ι
    hi : Membership.mem (↑I) i
    ⊢ Membership.mem (t i) (I.piecewise x y i)
  -/
  simpa [Finset.mem_coe.1 hi] using mem_of_mem_nhds (htx i)
  /-
    🎉 no goals
  -/


theorem pi_generateFrom_eq {π : ι → Type*} {g : ∀ a, Set (Set (π a))} :
    (@Pi.topologicalSpace ι π fun a => generateFrom (g a)) =
      generateFrom
        { t | ∃ (s : ∀ a, Set (π a)) (i : Finset ι), (∀ a ∈ i, s a ∈ g a) ∧ t = pi (↑i) s } := by
  /-
    ι : Type u_5
    π : ι → Type u_8
    g : (a : ι) → Set (Set (π a))
    ⊢ Eq Pi.topologicalSpace (TopologicalSpace.generateFrom (setOf fun t => Exists …
  -/
  refine le_antisymm ?_ ?_
    /-
      case refine_1
      ι : Type u_5
      π : ι → Type u_8
      g : (a : ι) → Set (Set (π a))
      ⊢ LE.le Pi.topologicalSpace (TopologicalSpace.generateFrom (setOf fun t => Exi …
    -/
  · apply le_generateFrom
    /-
      case refine_1.h
      ι : Type u_5
      π : ι → Type u_8
      g : (a : ι) → Set (Set (π a))
      ⊢ ∀ (s : Set ((i : ι) → π i)), Membership.mem (setOf fun t => Exists fun s =>  …
    -/
    rintro _ ⟨s, i, hi, rfl⟩
    /-
      case refine_1.h.intro.intro.intro
      ι : Type u_5
      π : ι → Type u_8
      g : (a : ι) → Set (Set (π a))
      s : (a : ι) → Set (π a)
      i : Finset ι
      hi : ∀ (a : ι), Membership.mem i a → Membership.mem (g a) (s a)
      ⊢ IsOpen ((↑i).pi s)
    -/
    letI := fun a => generateFrom (g a)
    /-
      case refine_1.h.intro.intro.intro
      ι : Type u_5
      π : ι → Type u_8
      g : (a : ι) → Set (Set (π a))
      s : (a : ι) → Set (π a)
      i : Finset ι
      hi : ∀ (a : ι), Membership.mem i a → Membership.mem (g a) (s a)
      this : (a : ι) → TopologicalSpace (π a) := fun a => TopologicalSpace.generateF …
      ⊢ IsOpen ((↑i).pi s)
    -/
    exact isOpen_set_pi i.finite_toSet (fun a ha => GenerateOpen.basic _ (hi a ha))
    /-
      🎉 no goals
    -/
  · classical
    refine le_iInf fun i => coinduced_le_iff_le_induced.1 <| le_generateFrom fun s hs => ?_
    refine GenerateOpen.basic _ ⟨update (fun i => univ) i s, {i}, ?_⟩
    simp [hs]


theorem pi_eq_generateFrom :
    Pi.topologicalSpace =
      generateFrom
        { g | ∃ (s : ∀ a, Set (π a)) (i : Finset ι), (∀ a ∈ i, IsOpen (s a)) ∧ g = pi (↑i) s } :=
  calc Pi.topologicalSpace
  _ = @Pi.topologicalSpace ι π fun _ => generateFrom { s | IsOpen s } := by
    /-
      ι : Type u_5
      π : ι → Type u_6
      T : (i : ι) → TopologicalSpace (π i)
      ⊢ Eq Pi.topologicalSpace Pi.topologicalSpace
    -/
    simp only [generateFrom_setOf_isOpen]
    /-
      🎉 no goals
    -/
  _ = _ := pi_generateFrom_eq


theorem pi_generateFrom_eq_finite {π : ι → Type*} {g : ∀ a, Set (Set (π a))} [Finite ι]
    (hg : ∀ a, ⋃₀ g a = univ) :
    (@Pi.topologicalSpace ι π fun a => generateFrom (g a)) =
      generateFrom { t | ∃ s : ∀ a, Set (π a), (∀ a, s a ∈ g a) ∧ t = pi univ s } := by
  /-
    ι : Type u_5
    π : ι → Type u_8
    g : (a : ι) → Set (Set (π a))
    inst✝ : Finite ι
    hg : ∀ (a : ι), Eq (g a).sUnion Set.univ
    ⊢ Eq Pi.topologicalSpace (TopologicalSpace.generateFrom (setOf fun t => Exists …
  -/
  cases nonempty_fintype ι
  /-
    case intro
    ι : Type u_5
    π : ι → Type u_8
    g : (a : ι) → Set (Set (π a))
    inst✝ : Finite ι
    hg : ∀ (a : ι), Eq (g a).sUnion Set.univ
    val✝ : Fintype ι
    ⊢ Eq Pi.topologicalSpace (TopologicalSpace.generateFrom (setOf fun t => Exists …
  -/
  rw [pi_generateFrom_eq]
  /-
    case intro
    ι : Type u_5
    π : ι → Type u_8
    g : (a : ι) → Set (Set (π a))
    inst✝ : Finite ι
    hg : ∀ (a : ι), Eq (g a).sUnion Set.univ
    val✝ : Fintype ι
    ⊢ Eq (TopologicalSpace.generateFrom (setOf fun t => Exists fun s => Exists fun …
  -/
  refine le_antisymm (generateFrom_anti ?_) (le_generateFrom ?_)
    /-
      case intro.refine_1
      ι : Type u_5
      π : ι → Type u_8
      g : (a : ι) → Set (Set (π a))
      inst✝ : Finite ι
      hg : ∀ (a : ι), Eq (g a).sUnion Set.univ
      val✝ : Fintype ι
      ⊢ HasSubset.Subset (setOf fun t => Exists fun s => And (∀ (a : ι), Membership. …
    -/
  · exact fun s ⟨t, ht, Eq⟩ => ⟨t, Finset.univ, by simp [ht, Eq]⟩
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_2
      ι : Type u_5
      π : ι → Type u_8
      g : (a : ι) → Set (Set (π a))
      inst✝ : Finite ι
      hg : ∀ (a : ι), Eq (g a).sUnion Set.univ
      val✝ : Fintype ι
      ⊢ ∀ (s : Set ((i : ι) → π i)), Membership.mem (setOf fun t => Exists fun s =>  …
    -/
  · rintro s ⟨t, i, ht, rfl⟩
    /-
      case intro.refine_2.intro.intro.intro
      ι : Type u_5
      π : ι → Type u_8
      g : (a : ι) → Set (Set (π a))
      inst✝ : Finite ι
      hg : ∀ (a : ι), Eq (g a).sUnion Set.univ
      val✝ : Fintype ι
      t : (a : ι) → Set (π a)
      i : Finset ι
      ht : ∀ (a : ι), Membership.mem i a → Membership.mem (g a) (t a)
      ⊢ IsOpen ((↑i).pi t)
    -/
    letI := generateFrom { t | ∃ s : ∀ a, Set (π a), (∀ a, s a ∈ g a) ∧ t = pi univ s }
    /-
      case intro.refine_2.intro.intro.intro
      ι : Type u_5
      π : ι → Type u_8
      g : (a : ι) → Set (Set (π a))
      inst✝ : Finite ι
      hg : ∀ (a : ι), Eq (g a).sUnion Set.univ
      val✝ : Fintype ι
      t : (a : ι) → Set (π a)
      i : Finset ι
      ht : ∀ (a : ι), Membership.mem i a → Membership.mem (g a) (t a)
      this : TopologicalSpace ((i : ι) → π i) := TopologicalSpace.generateFrom (setO …
      ⊢ IsOpen ((↑i).pi t)
    -/
    refine isOpen_iff_forall_mem_open.2 fun f hf => ?_
    /-
      case intro.refine_2.intro.intro.intro
      ι : Type u_5
      π : ι → Type u_8
      g : (a : ι) → Set (Set (π a))
      inst✝ : Finite ι
      hg : ∀ (a : ι), Eq (g a).sUnion Set.univ
      val✝ : Fintype ι
      t : (a : ι) → Set (π a)
      i : Finset ι
      ht : ∀ (a : ι), Membership.mem i a → Membership.mem (g a) (t a)
      this : TopologicalSpace ((i : ι) → π i) := TopologicalSpace.generateFrom (setO …
      f : (i : ι) → π i
      hf : Membership.mem ((↑i).pi t) f
      ⊢ Exists fun t_1 => And (HasSubset.Subset t_1 ((↑i).pi t)) (And (IsOpen t_1) ( …
    -/
    choose c hcg hfc using fun a => sUnion_eq_univ_iff.1 (hg a) (f a)
    /-
      case intro.refine_2.intro.intro.intro
      ι : Type u_5
      π : ι → Type u_8
      g : (a : ι) → Set (Set (π a))
      inst✝ : Finite ι
      hg : ∀ (a : ι), Eq (g a).sUnion Set.univ
      val✝ : Fintype ι
      t : (a : ι) → Set (π a)
      i : Finset ι
      ht : ∀ (a : ι), Membership.mem i a → Membership.mem (g a) (t a)
      this : TopologicalSpace ((i : ι) → π i) := TopologicalSpace.generateFrom (setO …
      f : (i : ι) → π i
      hf : Membership.mem ((↑i).pi t) f
      c : (a : ι) → Set (π a)
      hcg : ∀ (a : ι), Membership.mem (g a) (c a)
      hfc : ∀ (a : ι), Membership.mem (c a) (f a)
      ⊢ Exists fun t_1 => And (HasSubset.Subset t_1 ((↑i).pi t)) (And (IsOpen t_1) ( …
    -/
    refine ⟨pi i t ∩ pi ((↑i)ᶜ : Set ι) c, inter_subset_left, ?_, ⟨hf, fun a _ => hfc a⟩⟩
    classical
    rw [← univ_pi_piecewise]
    refine GenerateOpen.basic _ ⟨_, fun a => ?_, rfl⟩
    by_cases a ∈ i <;> simp [*]


theorem induced_to_pi {X : Type*} (f : X → ∀ i, π i) :
    induced f Pi.topologicalSpace = ⨅ i, induced (f · i) inferInstance := by
  /-
    ι : Type u_5
    π : ι → Type u_6
    T : (i : ι) → TopologicalSpace (π i)
    X : Type u_8
    f : X → (i : ι) → π i
    ⊢ Eq (TopologicalSpace.induced f Pi.topologicalSpace) (iInf fun i => Topologic …
  -/
  simp_rw [Pi.topologicalSpace, induced_iInf, induced_compose, Function.comp_def]
  /-
    🎉 no goals
  -/


/-- Suppose `π i` is a family of topological spaces indexed by `i : ι`, and `X` is a type
endowed with a family of maps `f i : X → π i` for every `i : ι`, hence inducing a
map `g : X → Π i, π i`. This lemma shows that infimum of the topologies on `X` induced by
the `f i` as `i : ι` varies is simply the topology on `X` induced by `g : X → Π i, π i`
where `Π i, π i` is endowed with the usual product topology. -/
theorem inducing_iInf_to_pi {X : Type*} (f : ∀ i, X → π i) :
    @IsInducing X (∀ i, π i) (⨅ i, induced (f i) inferInstance) _ fun x i => f i x :=
  letI := ⨅ i, induced (f i) inferInstance; ⟨(induced_to_pi _).symm⟩


/-- A finite product of discrete spaces is discrete. -/
instance Pi.discreteTopology : DiscreteTopology (∀ i, π i) :=
  singletons_open_iff_discrete.mp fun x => by
    /-
      X : Type u
      Y : Type v
      Z : Type u_1
      W : Type u_2
      ε : Type u_3
      ζ : Type u_4
      ι : Type u_5
      π : ι → Type u_6
      κ : Type u_7
      inst✝³ : TopologicalSpace X
      T : (i : ι) → TopologicalSpace (π i)
      f : X → (i : ι) → π i
      inst✝² : TopologicalSpace Z
      inst✝¹ : Finite ι
      inst✝ : ∀ (i : ι), DiscreteTopology (π i)
      x : (i : ι) → π i
      ⊢ IsOpen (Singleton.singleton x)
    -/
    rw [← univ_pi_singleton]
    /-
      X : Type u
      Y : Type v
      Z : Type u_1
      W : Type u_2
      ε : Type u_3
      ζ : Type u_4
      ι : Type u_5
      π : ι → Type u_6
      κ : Type u_7
      inst✝³ : TopologicalSpace X
      T : (i : ι) → TopologicalSpace (π i)
      f : X → (i : ι) → π i
      inst✝² : TopologicalSpace Z
      inst✝¹ : Finite ι
      inst✝ : ∀ (i : ι), DiscreteTopology (π i)
      x : (i : ι) → π i
      ⊢ IsOpen (Set.univ.pi fun i => Singleton.singleton (x i))
    -/
    exact isOpen_set_pi finite_univ fun i _ => (isOpen_discrete {x i})
    /-
      🎉 no goals
    -/


@[continuity, fun_prop]
theorem continuous_sigmaMk {i : ι} : Continuous (@Sigma.mk ι σ i) :=
  continuous_iSup_rng continuous_coinduced_rng

-- Porting note: the proof was `by simp only [isOpen_iSup_iff, isOpen_coinduced]`

theorem isOpen_sigma_iff {s : Set (Sigma σ)} : IsOpen s ↔ ∀ i, IsOpen (Sigma.mk i ⁻¹' s) := by
  /-
    ι : Type u_5
    σ : ι → Type u_7
    inst✝ : (i : ι) → TopologicalSpace (σ i)
    s : Set (Sigma σ)
    ⊢ Iff (IsOpen s) (∀ (i : ι), IsOpen (Set.preimage (Sigma.mk i) s))
  -/
  delta instTopologicalSpaceSigma
  /-
    ι : Type u_5
    σ : ι → Type u_7
    inst✝ : (i : ι) → TopologicalSpace (σ i)
    s : Set (Sigma σ)
    ⊢ Iff (IsOpen s) (∀ (i : ι), IsOpen (Set.preimage (Sigma.mk i) s))
  -/
  rw [isOpen_iSup_iff]
  /-
    ι : Type u_5
    σ : ι → Type u_7
    inst✝ : (i : ι) → TopologicalSpace (σ i)
    s : Set (Sigma σ)
    ⊢ Iff (∀ (i : ι), IsOpen s) (∀ (i : ι), IsOpen (Set.preimage (Sigma.mk i) s))
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem isClosed_sigma_iff {s : Set (Sigma σ)} : IsClosed s ↔ ∀ i, IsClosed (Sigma.mk i ⁻¹' s) := by
  /-
    ι : Type u_5
    σ : ι → Type u_7
    inst✝ : (i : ι) → TopologicalSpace (σ i)
    s : Set (Sigma σ)
    ⊢ Iff (IsClosed s) (∀ (i : ι), IsClosed (Set.preimage (Sigma.mk i) s))
  -/
  simp only [← isOpen_compl_iff, isOpen_sigma_iff, preimage_compl]
  /-
    🎉 no goals
  -/


theorem isOpenMap_sigmaMk {i : ι} : IsOpenMap (@Sigma.mk ι σ i) := by
  /-
    ι : Type u_5
    σ : ι → Type u_7
    inst✝ : (i : ι) → TopologicalSpace (σ i)
    i : ι
    ⊢ IsOpenMap (Sigma.mk i)
  -/
  intro s hs
  /-
    ι : Type u_5
    σ : ι → Type u_7
    inst✝ : (i : ι) → TopologicalSpace (σ i)
    i : ι
    s : Set (σ i)
    hs : IsOpen s
    ⊢ IsOpen (Set.image (Sigma.mk i) s)
  -/
  rw [isOpen_sigma_iff]
  /-
    ι : Type u_5
    σ : ι → Type u_7
    inst✝ : (i : ι) → TopologicalSpace (σ i)
    i : ι
    s : Set (σ i)
    hs : IsOpen s
    ⊢ ∀ (i_1 : ι), IsOpen (Set.preimage (Sigma.mk i_1) (Set.image (Sigma.mk i) s))
  -/
  intro j
  /-
    ι : Type u_5
    σ : ι → Type u_7
    inst✝ : (i : ι) → TopologicalSpace (σ i)
    i : ι
    s : Set (σ i)
    hs : IsOpen s
    j : ι
    ⊢ IsOpen (Set.preimage (Sigma.mk j) (Set.image (Sigma.mk i) s))
  -/
  rcases eq_or_ne j i with (rfl | hne)
    /-
      case inl
      ι : Type u_5
      σ : ι → Type u_7
      inst✝ : (i : ι) → TopologicalSpace (σ i)
      j : ι
      s : Set (σ j)
      hs : IsOpen s
      ⊢ IsOpen (Set.preimage (Sigma.mk j) (Set.image (Sigma.mk j) s))
    -/
  · rwa [preimage_image_eq _ sigma_mk_injective]
    /-
      🎉 no goals
    -/
    /-
      case inr
      ι : Type u_5
      σ : ι → Type u_7
      inst✝ : (i : ι) → TopologicalSpace (σ i)
      i : ι
      s : Set (σ i)
      hs : IsOpen s
      j : ι
      hne : Ne j i
      ⊢ IsOpen (Set.preimage (Sigma.mk j) (Set.image (Sigma.mk i) s))
    -/
  · rw [preimage_image_sigmaMk_of_ne hne]
    /-
      case inr
      ι : Type u_5
      σ : ι → Type u_7
      inst✝ : (i : ι) → TopologicalSpace (σ i)
      i : ι
      s : Set (σ i)
      hs : IsOpen s
      j : ι
      hne : Ne j i
      ⊢ IsOpen EmptyCollection.emptyCollection
    -/
    exact isOpen_empty
    /-
      🎉 no goals
    -/


theorem isOpen_range_sigmaMk {i : ι} : IsOpen (range (@Sigma.mk ι σ i)) :=
  isOpenMap_sigmaMk.isOpen_range


theorem isClosedMap_sigmaMk {i : ι} : IsClosedMap (@Sigma.mk ι σ i) := by
  /-
    ι : Type u_5
    σ : ι → Type u_7
    inst✝ : (i : ι) → TopologicalSpace (σ i)
    i : ι
    ⊢ IsClosedMap (Sigma.mk i)
  -/
  intro s hs
  /-
    ι : Type u_5
    σ : ι → Type u_7
    inst✝ : (i : ι) → TopologicalSpace (σ i)
    i : ι
    s : Set (σ i)
    hs : IsClosed s
    ⊢ IsClosed (Set.image (Sigma.mk i) s)
  -/
  rw [isClosed_sigma_iff]
  /-
    ι : Type u_5
    σ : ι → Type u_7
    inst✝ : (i : ι) → TopologicalSpace (σ i)
    i : ι
    s : Set (σ i)
    hs : IsClosed s
    ⊢ ∀ (i_1 : ι), IsClosed (Set.preimage (Sigma.mk i_1) (Set.image (Sigma.mk i) s))
  -/
  intro j
  /-
    ι : Type u_5
    σ : ι → Type u_7
    inst✝ : (i : ι) → TopologicalSpace (σ i)
    i : ι
    s : Set (σ i)
    hs : IsClosed s
    j : ι
    ⊢ IsClosed (Set.preimage (Sigma.mk j) (Set.image (Sigma.mk i) s))
  -/
  rcases eq_or_ne j i with (rfl | hne)
    /-
      case inl
      ι : Type u_5
      σ : ι → Type u_7
      inst✝ : (i : ι) → TopologicalSpace (σ i)
      j : ι
      s : Set (σ j)
      hs : IsClosed s
      ⊢ IsClosed (Set.preimage (Sigma.mk j) (Set.image (Sigma.mk j) s))
    -/
  · rwa [preimage_image_eq _ sigma_mk_injective]
    /-
      🎉 no goals
    -/
    /-
      case inr
      ι : Type u_5
      σ : ι → Type u_7
      inst✝ : (i : ι) → TopologicalSpace (σ i)
      i : ι
      s : Set (σ i)
      hs : IsClosed s
      j : ι
      hne : Ne j i
      ⊢ IsClosed (Set.preimage (Sigma.mk j) (Set.image (Sigma.mk i) s))
    -/
  · rw [preimage_image_sigmaMk_of_ne hne]
    /-
      case inr
      ι : Type u_5
      σ : ι → Type u_7
      inst✝ : (i : ι) → TopologicalSpace (σ i)
      i : ι
      s : Set (σ i)
      hs : IsClosed s
      j : ι
      hne : Ne j i
      ⊢ IsClosed EmptyCollection.emptyCollection
    -/
    exact isClosed_empty
    /-
      🎉 no goals
    -/


theorem isClosed_range_sigmaMk {i : ι} : IsClosed (range (@Sigma.mk ι σ i)) :=
  isClosedMap_sigmaMk.isClosed_range


lemma Topology.IsOpenEmbedding.sigmaMk {i : ι} : IsOpenEmbedding (@Sigma.mk ι σ i) :=
  .of_continuous_injective_isOpenMap continuous_sigmaMk sigma_mk_injective isOpenMap_sigmaMk


@[deprecated (since := "2024-10-30")] alias isOpenEmbedding_sigmaMk := IsOpenEmbedding.sigmaMk


@[deprecated (since := "2024-10-18")]
alias openEmbedding_sigmaMk := IsOpenEmbedding.sigmaMk


lemma Topology.IsClosedEmbedding.sigmaMk {i : ι} : IsClosedEmbedding (@Sigma.mk ι σ i) :=
  .of_continuous_injective_isClosedMap continuous_sigmaMk sigma_mk_injective isClosedMap_sigmaMk


@[deprecated (since := "2024-10-30")] alias isClosedEmbedding_sigmaMk := IsClosedEmbedding.sigmaMk


@[deprecated (since := "2024-10-20")]
alias closedEmbedding_sigmaMk := IsClosedEmbedding.sigmaMk


lemma Topology.IsEmbedding.sigmaMk {i : ι} : IsEmbedding (@Sigma.mk ι σ i) :=
  IsClosedEmbedding.sigmaMk.1


@[deprecated (since := "2024-10-26")]
alias embedding_sigmaMk := IsEmbedding.sigmaMk


theorem Sigma.nhds_mk (i : ι) (x : σ i) : 𝓝 (⟨i, x⟩ : Sigma σ) = Filter.map (Sigma.mk i) (𝓝 x) :=
  (IsOpenEmbedding.sigmaMk.map_nhds_eq x).symm


theorem Sigma.nhds_eq (x : Sigma σ) : 𝓝 x = Filter.map (Sigma.mk x.1) (𝓝 x.2) := by
  /-
    ι : Type u_5
    σ : ι → Type u_7
    inst✝ : (i : ι) → TopologicalSpace (σ i)
    x : Sigma σ
    ⊢ Eq (nhds x) (Filter.map (Sigma.mk x.fst) (nhds x.snd))
  -/
  cases x
  /-
    case mk
    ι : Type u_5
    σ : ι → Type u_7
    inst✝ : (i : ι) → TopologicalSpace (σ i)
    fst✝ : ι
    snd✝ : σ fst✝
    ⊢ Eq (nhds ⟨fst✝, snd✝⟩) (Filter.map (Sigma.mk ⟨fst✝, snd✝⟩.fst) (nhds ⟨fst✝,  …
  -/
  apply Sigma.nhds_mk
  /-
    🎉 no goals
  -/


theorem comap_sigmaMk_nhds (i : ι) (x : σ i) : comap (Sigma.mk i) (𝓝 ⟨i, x⟩) = 𝓝 x :=
  (IsEmbedding.sigmaMk.nhds_eq_comap _).symm


theorem isOpen_sigma_fst_preimage (s : Set ι) : IsOpen (Sigma.fst ⁻¹' s : Set (Σ a, σ a)) := by
  /-
    ι : Type u_5
    σ : ι → Type u_7
    inst✝ : (i : ι) → TopologicalSpace (σ i)
    s : Set ι
    ⊢ IsOpen (Set.preimage Sigma.fst s)
  -/
  rw [← biUnion_of_singleton s, preimage_iUnion₂]
  /-
    ι : Type u_5
    σ : ι → Type u_7
    inst✝ : (i : ι) → TopologicalSpace (σ i)
    s : Set ι
    ⊢ IsOpen (Set.iUnion fun i => Set.iUnion fun j => Set.preimage Sigma.fst (Sing …
  -/
  simp only [← range_sigmaMk]
  /-
    ι : Type u_5
    σ : ι → Type u_7
    inst✝ : (i : ι) → TopologicalSpace (σ i)
    s : Set ι
    ⊢ IsOpen (Set.iUnion fun i => Set.iUnion fun x => Set.range (Sigma.mk i))
  -/
  exact isOpen_biUnion fun _ _ => isOpen_range_sigmaMk
  /-
    🎉 no goals
  -/


/-- A map out of a sum type is continuous iff its restriction to each summand is. -/
@[simp]
theorem continuous_sigma_iff {f : Sigma σ → X} :
    Continuous f ↔ ∀ i, Continuous fun a => f ⟨i, a⟩ := by
  /-
    X : Type u
    ι : Type u_5
    σ : ι → Type u_7
    inst✝¹ : (i : ι) → TopologicalSpace (σ i)
    inst✝ : TopologicalSpace X
    f : Sigma σ → X
    ⊢ Iff (Continuous f) (∀ (i : ι), Continuous fun a => f ⟨i, a⟩)
  -/
  delta instTopologicalSpaceSigma
  /-
    X : Type u
    ι : Type u_5
    σ : ι → Type u_7
    inst✝¹ : (i : ι) → TopologicalSpace (σ i)
    inst✝ : TopologicalSpace X
    f : Sigma σ → X
    ⊢ Iff (Continuous f) (∀ (i : ι), Continuous fun a => f ⟨i, a⟩)
  -/
  rw [continuous_iSup_dom]
  /-
    X : Type u
    ι : Type u_5
    σ : ι → Type u_7
    inst✝¹ : (i : ι) → TopologicalSpace (σ i)
    inst✝ : TopologicalSpace X
    f : Sigma σ → X
    ⊢ Iff (∀ (i : ι), Continuous f) (∀ (i : ι), Continuous fun a => f ⟨i, a⟩)
  -/
  exact forall_congr' fun _ => continuous_coinduced_dom
  /-
    🎉 no goals
  -/


/-- A map out of a sum type is continuous if its restriction to each summand is. -/
@[continuity, fun_prop]
theorem continuous_sigma {f : Sigma σ → X} (hf : ∀ i, Continuous fun a => f ⟨i, a⟩) :
    Continuous f :=
  continuous_sigma_iff.2 hf


/-- A map defined on a sigma type (a.k.a. the disjoint union of an indexed family of topological
spaces) is inducing iff its restriction to each component is inducing and each the image of each
component under `f` can be separated from the images of all other components by an open set. -/
theorem inducing_sigma {f : Sigma σ → X} :
    IsInducing f ↔ (∀ i, IsInducing (f ∘ Sigma.mk i)) ∧
      (∀ i, ∃ U, IsOpen U ∧ ∀ x, f x ∈ U ↔ x.1 = i) := by
  /-
    X : Type u
    ι : Type u_5
    σ : ι → Type u_7
    inst✝¹ : (i : ι) → TopologicalSpace (σ i)
    inst✝ : TopologicalSpace X
    f : Sigma σ → X
    ⊢ Iff (Topology.IsInducing f) (And (∀ (i : ι), Topology.IsInducing (Function.c …
  -/
  refine ⟨fun h ↦ ⟨fun i ↦ h.comp IsEmbedding.sigmaMk.1, fun i ↦ ?_⟩, ?_⟩
    /-
      case refine_1
      X : Type u
      ι : Type u_5
      σ : ι → Type u_7
      inst✝¹ : (i : ι) → TopologicalSpace (σ i)
      inst✝ : TopologicalSpace X
      f : Sigma σ → X
      h : Topology.IsInducing f
      i : ι
      ⊢ Exists fun U => And (IsOpen U) (∀ (x : Sigma σ), Iff (Membership.mem U (f x) …
    -/
  · rcases h.isOpen_iff.1 (isOpen_range_sigmaMk (i := i)) with ⟨U, hUo, hU⟩
    /-
      case refine_1.intro.intro
      X : Type u
      ι : Type u_5
      σ : ι → Type u_7
      inst✝¹ : (i : ι) → TopologicalSpace (σ i)
      inst✝ : TopologicalSpace X
      f : Sigma σ → X
      h : Topology.IsInducing f
      i : ι
      U : Set X
      hUo : IsOpen U
      hU : Eq (Set.preimage f U) (Set.range (Sigma.mk i))
      ⊢ Exists fun U => And (IsOpen U) (∀ (x : Sigma σ), Iff (Membership.mem U (f x) …
    -/
    refine ⟨U, hUo, ?_⟩
    /-
      case refine_1.intro.intro
      X : Type u
      ι : Type u_5
      σ : ι → Type u_7
      inst✝¹ : (i : ι) → TopologicalSpace (σ i)
      inst✝ : TopologicalSpace X
      f : Sigma σ → X
      h : Topology.IsInducing f
      i : ι
      U : Set X
      hUo : IsOpen U
      hU : Eq (Set.preimage f U) (Set.range (Sigma.mk i))
      ⊢ ∀ (x : Sigma σ), Iff (Membership.mem U (f x)) (Eq x.fst i)
    -/
    simpa [Set.ext_iff] using hU
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      X : Type u
      ι : Type u_5
      σ : ι → Type u_7
      inst✝¹ : (i : ι) → TopologicalSpace (σ i)
      inst✝ : TopologicalSpace X
      f : Sigma σ → X
      ⊢ And (∀ (i : ι), Topology.IsInducing (Function.comp f (Sigma.mk i))) (∀ (i :  …
    -/
  · refine fun ⟨h₁, h₂⟩ ↦ isInducing_iff_nhds.2 fun ⟨i, x⟩ ↦ ?_
    /-
      case refine_2
      X : Type u
      ι : Type u_5
      σ : ι → Type u_7
      inst✝¹ : (i : ι) → TopologicalSpace (σ i)
      inst✝ : TopologicalSpace X
      f : Sigma σ → X
      x✝¹ : And (∀ (i : ι), Topology.IsInducing (Function.comp f (Sigma.mk i))) (∀ ( …
      h₁ : ∀ (i : ι), Topology.IsInducing (Function.comp f (Sigma.mk i))
      h₂ : ∀ (i : ι), Exists fun U => And (IsOpen U) (∀ (x : Sigma σ), Iff (Membersh …
      x✝ : Sigma σ
      i : ι
      x : σ i
      ⊢ Eq (nhds ⟨i, x⟩) (Filter.comap f (nhds (f ⟨i, x⟩)))
    -/
    rw [Sigma.nhds_mk, (h₁ i).nhds_eq_comap, comp_apply, ← comap_comap, map_comap_of_mem]
    /-
      case refine_2
      X : Type u
      ι : Type u_5
      σ : ι → Type u_7
      inst✝¹ : (i : ι) → TopologicalSpace (σ i)
      inst✝ : TopologicalSpace X
      f : Sigma σ → X
      x✝¹ : And (∀ (i : ι), Topology.IsInducing (Function.comp f (Sigma.mk i))) (∀ ( …
      h₁ : ∀ (i : ι), Topology.IsInducing (Function.comp f (Sigma.mk i))
      h₂ : ∀ (i : ι), Exists fun U => And (IsOpen U) (∀ (x : Sigma σ), Iff (Membersh …
      x✝ : Sigma σ
      i : ι
      x : σ i
      ⊢ Membership.mem (Filter.comap f (nhds (f ⟨i, x⟩))) (Set.range (Sigma.mk i))
    -/
    rcases h₂ i with ⟨U, hUo, hU⟩
    /-
      case refine_2.intro.intro
      X : Type u
      ι : Type u_5
      σ : ι → Type u_7
      inst✝¹ : (i : ι) → TopologicalSpace (σ i)
      inst✝ : TopologicalSpace X
      f : Sigma σ → X
      x✝¹ : And (∀ (i : ι), Topology.IsInducing (Function.comp f (Sigma.mk i))) (∀ ( …
      h₁ : ∀ (i : ι), Topology.IsInducing (Function.comp f (Sigma.mk i))
      h₂ : ∀ (i : ι), Exists fun U => And (IsOpen U) (∀ (x : Sigma σ), Iff (Membersh …
      x✝ : Sigma σ
      i : ι
      x : σ i
      U : Set X
      hUo : IsOpen U
      hU : ∀ (x : Sigma σ), Iff (Membership.mem U (f x)) (Eq x.fst i)
      ⊢ Membership.mem (Filter.comap f (nhds (f ⟨i, x⟩))) (Set.range (Sigma.mk i))
    -/
    filter_upwards [preimage_mem_comap <| hUo.mem_nhds <| (hU _).2 rfl] with y hy
    /-
      case h
      X : Type u
      ι : Type u_5
      σ : ι → Type u_7
      inst✝¹ : (i : ι) → TopologicalSpace (σ i)
      inst✝ : TopologicalSpace X
      f : Sigma σ → X
      x✝¹ : And (∀ (i : ι), Topology.IsInducing (Function.comp f (Sigma.mk i))) (∀ ( …
      h₁ : ∀ (i : ι), Topology.IsInducing (Function.comp f (Sigma.mk i))
      h₂ : ∀ (i : ι), Exists fun U => And (IsOpen U) (∀ (x : Sigma σ), Iff (Membersh …
      x✝ : Sigma σ
      i : ι
      x : σ i
      U : Set X
      hUo : IsOpen U
      hU : ∀ (x : Sigma σ), Iff (Membership.mem U (f x)) (Eq x.fst i)
      y : Sigma σ
      hy : Membership.mem (Set.preimage f U) y
      ⊢ Membership.mem (Set.range (Sigma.mk i)) y
    -/
    simpa [hU] using hy
    /-
      🎉 no goals
    -/


@[simp 1100]
theorem continuous_sigma_map {f₁ : ι → κ} {f₂ : ∀ i, σ i → τ (f₁ i)} :
    Continuous (Sigma.map f₁ f₂) ↔ ∀ i, Continuous (f₂ i) :=
  continuous_sigma_iff.trans <| by
    /-
      ι : Type u_5
      κ : Type u_6
      σ : ι → Type u_7
      τ : κ → Type u_8
      inst✝¹ : (i : ι) → TopologicalSpace (σ i)
      inst✝ : (k : κ) → TopologicalSpace (τ k)
      f₁ : ι → κ
      f₂ : (i : ι) → σ i → τ (f₁ i)
      ⊢ Iff (∀ (i : ι), Continuous fun a => Sigma.map f₁ f₂ ⟨i, a⟩) (∀ (i : ι), Cont …
    -/
    simp only [Sigma.map, IsEmbedding.sigmaMk.continuous_iff, comp_def]
    /-
      🎉 no goals
    -/


@[continuity, fun_prop]
theorem Continuous.sigma_map {f₁ : ι → κ} {f₂ : ∀ i, σ i → τ (f₁ i)} (hf : ∀ i, Continuous (f₂ i)) :
    Continuous (Sigma.map f₁ f₂) :=
  continuous_sigma_map.2 hf


theorem isOpenMap_sigma {f : Sigma σ → X} : IsOpenMap f ↔ ∀ i, IsOpenMap fun a => f ⟨i, a⟩ := by
  /-
    X : Type u
    ι : Type u_5
    σ : ι → Type u_7
    inst✝¹ : (i : ι) → TopologicalSpace (σ i)
    inst✝ : TopologicalSpace X
    f : Sigma σ → X
    ⊢ Iff (IsOpenMap f) (∀ (i : ι), IsOpenMap fun a => f ⟨i, a⟩)
  -/
  simp only [isOpenMap_iff_nhds_le, Sigma.forall, Sigma.nhds_eq, map_map, comp_def]
  /-
    🎉 no goals
  -/


theorem isOpenMap_sigma_map {f₁ : ι → κ} {f₂ : ∀ i, σ i → τ (f₁ i)} :
    IsOpenMap (Sigma.map f₁ f₂) ↔ ∀ i, IsOpenMap (f₂ i) :=
  isOpenMap_sigma.trans <|
    forall_congr' fun i => (@IsOpenEmbedding.sigmaMk _ _ _ (f₁ i)).isOpenMap_iff.symm


lemma Topology.isInducing_sigmaMap {f₁ : ι → κ} {f₂ : ∀ i, σ i → τ (f₁ i)}
    (h₁ : Injective f₁) : IsInducing (Sigma.map f₁ f₂) ↔ ∀ i, IsInducing (f₂ i) := by
  simp only [isInducing_iff_nhds, Sigma.forall, Sigma.nhds_mk, Sigma.map_mk,
    ← map_sigma_mk_comap h₁, map_inj sigma_mk_injective]


@[deprecated (since := "2024-10-28")] alias inducing_sigma_map := isInducing_sigmaMap


lemma Topology.isEmbedding_sigmaMap {f₁ : ι → κ} {f₂ : ∀ i, σ i → τ (f₁ i)}
    (h : Injective f₁) : IsEmbedding (Sigma.map f₁ f₂) ↔ ∀ i, IsEmbedding (f₂ i) := by
  simp only [isEmbedding_iff, Injective.sigma_map, isInducing_sigmaMap h, forall_and,
    h.sigma_map_iff]


@[deprecated (since := "2024-10-26")]
alias embedding_sigma_map := isEmbedding_sigmaMap


lemma Topology.isOpenEmbedding_sigmaMap {f₁ : ι → κ} {f₂ : ∀ i, σ i → τ (f₁ i)} (h : Injective f₁) :
    IsOpenEmbedding (Sigma.map f₁ f₂) ↔ ∀ i, IsOpenEmbedding (f₂ i) := by
  simp only [isOpenEmbedding_iff_isEmbedding_isOpenMap, isOpenMap_sigma_map, isEmbedding_sigmaMap h,
    forall_and]


@[deprecated (since := "2024-10-30")] alias isOpenEmbedding_sigma_map := isOpenEmbedding_sigmaMap


@[deprecated (since := "2024-10-18")]
alias openEmbedding_sigma_map := isOpenEmbedding_sigmaMap


theorem ULift.isOpen_iff [TopologicalSpace X] {s : Set (ULift.{v} X)} :
    IsOpen s ↔ IsOpen (ULift.up ⁻¹' s) := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set (ULift.{v, u} X)
    ⊢ Iff (IsOpen s) (IsOpen (Set.preimage ULift.up s))
  -/
  rw [ULift.topologicalSpace, ← Equiv.ulift_apply, ← Equiv.ulift.coinduced_symm, ← isOpen_coinduced]
  /-
    🎉 no goals
  -/


theorem ULift.isClosed_iff [TopologicalSpace X] {s : Set (ULift.{v} X)} :
    IsClosed s ↔ IsClosed (ULift.up ⁻¹' s) := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set (ULift.{v, u} X)
    ⊢ Iff (IsClosed s) (IsClosed (Set.preimage ULift.up s))
  -/
  rw [← isOpen_compl_iff, ← isOpen_compl_iff, isOpen_iff, preimage_compl]
  /-
    🎉 no goals
  -/


@[continuity]
theorem continuous_uLift_down [TopologicalSpace X] : Continuous (ULift.down : ULift.{v, u} X → X) :=
  continuous_induced_dom


@[continuity]
theorem continuous_uLift_up [TopologicalSpace X] : Continuous (ULift.up : X → ULift.{v, u} X) :=
  continuous_induced_rng.2 continuous_id


lemma Topology.IsEmbedding.uliftDown [TopologicalSpace X] :
    IsEmbedding (ULift.down : ULift.{v, u} X → X) := ⟨⟨rfl⟩, ULift.down_injective⟩


@[deprecated (since := "2024-10-26")]
alias embedding_uLift_down := IsEmbedding.uliftDown


lemma Topology.IsClosedEmbedding.uliftDown [TopologicalSpace X] :
    IsClosedEmbedding (ULift.down : ULift.{v, u} X → X) :=
                  /-
                    X : Type u
                    inst✝ : TopologicalSpace X
                    ⊢ IsClosed (Set.range ULift.down)
                  -/
  ⟨.uliftDown, by simp only [ULift.down_surjective.range_eq, isClosed_univ]⟩
                  /-
                    🎉 no goals
                  -/


@[deprecated (since := "2024-10-30")]
alias ULift.isClosedEmbedding_down := IsClosedEmbedding.uliftDown


@[deprecated (since := "2024-10-20")]
alias ULift.closedEmbedding_down := IsClosedEmbedding.uliftDown


instance [TopologicalSpace X] [DiscreteTopology X] : DiscreteTopology (ULift X) :=
  IsEmbedding.uliftDown.discreteTopology


theorem IsOpen.trans (ht : IsOpen t) (hs : IsOpen s) : IsOpen (t : Set X) := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    t : Set ↑s
    ht : IsOpen t
    hs : IsOpen s
    ⊢ IsOpen (Set.image Subtype.val t)
  -/
  rcases isOpen_induced_iff.mp ht with ⟨s', hs', rfl⟩
  /-
    case intro.intro
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    hs : IsOpen s
    s' : Set X
    hs' : IsOpen s'
    ht : IsOpen (Set.preimage Subtype.val s')
    ⊢ IsOpen (Set.image Subtype.val (Set.preimage Subtype.val s'))
  -/
  rw [Subtype.image_preimage_coe]
  /-
    case intro.intro
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    hs : IsOpen s
    s' : Set X
    hs' : IsOpen s'
    ht : IsOpen (Set.preimage Subtype.val s')
    ⊢ IsOpen (Inter.inter s s')
  -/
  exact hs.inter hs'
  /-
    🎉 no goals
  -/


theorem IsClosed.trans (ht : IsClosed t) (hs : IsClosed s) : IsClosed (t : Set X) := by
  /-
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    t : Set ↑s
    ht : IsClosed t
    hs : IsClosed s
    ⊢ IsClosed (Set.image Subtype.val t)
  -/
  rcases isClosed_induced_iff.mp ht with ⟨s', hs', rfl⟩
  /-
    case intro.intro
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    hs : IsClosed s
    s' : Set X
    hs' : IsClosed s'
    ht : IsClosed (Set.preimage Subtype.val s')
    ⊢ IsClosed (Set.image Subtype.val (Set.preimage Subtype.val s'))
  -/
  rw [Subtype.image_preimage_coe]
  /-
    case intro.intro
    X : Type u
    inst✝ : TopologicalSpace X
    s : Set X
    hs : IsClosed s
    s' : Set X
    hs' : IsClosed s'
    ht : IsClosed (Set.preimage Subtype.val s')
    ⊢ IsClosed (Inter.inter s s')
  -/
  exact hs.inter hs'
  /-
    🎉 no goals
  -/


/-- The product of a neighborhood of `s` and a neighborhood of `t` is a neighborhood of `s ×ˢ t`,
formulated in terms of a filter inequality. -/
theorem nhdsSet_prod_le (s : Set X) (t : Set Y) : 𝓝ˢ (s ×ˢ t) ≤ 𝓝ˢ s ×ˢ 𝓝ˢ t :=
  ((hasBasis_nhdsSet _).prod (hasBasis_nhdsSet _)).ge_iff.2 fun (_u, _v) ⟨⟨huo, hsu⟩, hvo, htv⟩ ↦
    (huo.prod hvo).mem_nhdsSet.2 <| prod_mono hsu htv


theorem Filter.eventually_nhdsSet_prod_iff {p : X × Y → Prop} :
    (∀ᶠ q in 𝓝ˢ (s ×ˢ t), p q) ↔
      ∀ x ∈ s, ∀ y ∈ t,
          ∃ px : X → Prop, (∀ᶠ x' in 𝓝 x, px x') ∧ ∃ py : Y → Prop, (∀ᶠ y' in 𝓝 y, py y') ∧
            ∀ {x : X}, px x → ∀ {y : Y}, py y → p (x, y) := by
  /-
    X : Type u
    Y : Type v
    inst✝¹ : TopologicalSpace X
    inst✝ : TopologicalSpace Y
    s : Set X
    t : Set Y
    p : Prod X Y → Prop
    ⊢ Iff (Filter.Eventually (fun q => p q) (nhdsSet (SProd.sprod s t))) (∀ (x : X …
  -/
  simp_rw [eventually_nhdsSet_iff_forall, forall_prod_set, nhds_prod_eq, eventually_prod_iff]
  /-
    🎉 no goals
  -/


theorem Filter.Eventually.prod_nhdsSet {p : X × Y → Prop} {px : X → Prop} {py : Y → Prop}
    (hp : ∀ {x : X}, px x → ∀ {y : Y}, py y → p (x, y)) (hs : ∀ᶠ x in 𝓝ˢ s, px x)
    (ht : ∀ᶠ y in 𝓝ˢ t, py y) : ∀ᶠ q in 𝓝ˢ (s ×ˢ t), p q :=
  nhdsSet_prod_le _ _ (mem_of_superset (prod_mem_prod hs ht) fun _ ⟨hx, hy⟩ ↦ hp hx hy)


