/-- The `MeasurableSpace` of sets which are measurable with respect to a given σ-algebra `m`
on `α`, modulo a given σ-filter `l` on `α`. -/
def EventuallyMeasurableSpace (l : Filter α) [CountableInterFilter l] : MeasurableSpace α where
  MeasurableSet' s := ∃ t, MeasurableSet t ∧ s =ᶠ[l] t
  measurableSet_empty := ⟨∅, MeasurableSet.empty, EventuallyEq.refl _ _ ⟩
  measurableSet_compl := fun _ ⟨t, ht, hts⟩ => ⟨tᶜ, ht.compl, hts.compl⟩
  measurableSet_iUnion s hs := by
    /-
      α : Type u_1
      m : MeasurableSpace α
      s✝ t : Set α
      l : Filter α
      inst✝ : CountableInterFilter l
      s : Nat → Set α
      hs : ∀ (i : Nat), (fun s => Exists fun t => And (MeasurableSet t) (l.Eventuall …
      ⊢ (fun s => Exists fun t => And (MeasurableSet t) (l.EventuallyEq s t)) (Set.i …
    -/
    choose t ht hts using hs
    /-
      α : Type u_1
      m : MeasurableSpace α
      s✝ t✝ : Set α
      l : Filter α
      inst✝ : CountableInterFilter l
      s t : Nat → Set α
      ht : ∀ (i : Nat), MeasurableSet (t i)
      hts : ∀ (i : Nat), l.EventuallyEq (s i) (t i)
      ⊢ Exists fun t => And (MeasurableSet t) (l.EventuallyEq (Set.iUnion fun i => s …
    -/
    exact ⟨⋃ i, t i, MeasurableSet.iUnion ht, EventuallyEq.countable_iUnion hts⟩
    /-
      🎉 no goals
    -/


/-- We say a set `s` is an `EventuallyMeasurableSet` with respect to a given
σ-algebra `m` and σ-filter `l` if it differs from a set in `m` by a set in
the dual ideal of `l`. -/
def EventuallyMeasurableSet (l : Filter α) [CountableInterFilter l]  (s : Set α) : Prop :=
  @MeasurableSet _ (EventuallyMeasurableSpace m l) s


theorem MeasurableSet.eventuallyMeasurableSet (hs : MeasurableSet s) :
    EventuallyMeasurableSet m l s :=
  ⟨s, hs, EventuallyEq.refl _ _⟩


theorem EventuallyMeasurableSpace.measurable_le : m ≤ EventuallyMeasurableSpace m l :=
  fun _ hs => hs.eventuallyMeasurableSet


theorem eventuallyMeasurableSet_of_mem_filter (hs : s ∈ l) : EventuallyMeasurableSet m l s :=
  ⟨univ, MeasurableSet.univ, eventuallyEq_univ.mpr hs⟩


/-- A set which is `EventuallyEq` to an `EventuallyMeasurableSet`
is an `EventuallyMeasurableSet`. -/
theorem EventuallyMeasurableSet.congr
    (ht : EventuallyMeasurableSet m l t) (hst : s =ᶠ[l] t) : EventuallyMeasurableSet m l s := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    s t : Set α
    l : Filter α
    inst✝ : CountableInterFilter l
    ht : EventuallyMeasurableSet m l t
    hst : l.EventuallyEq s t
    ⊢ EventuallyMeasurableSet m l s
  -/
  rcases ht with ⟨t', ht', htt'⟩
  /-
    case intro.intro
    α : Type u_1
    m : MeasurableSpace α
    s t : Set α
    l : Filter α
    inst✝ : CountableInterFilter l
    hst : l.EventuallyEq s t
    t' : Set α
    ht' : MeasurableSet t'
    htt' : l.EventuallyEq t t'
    ⊢ EventuallyMeasurableSet m l s
  -/
  exact ⟨t', ht', hst.trans htt'⟩
  /-
    🎉 no goals
  -/


instance measurableSingleton [MeasurableSingletonClass α] :
    @MeasurableSingletonClass α (EventuallyMeasurableSpace m l) :=
  @MeasurableSingletonClass.mk _ (_) <| fun x => (MeasurableSet.singleton x).eventuallyMeasurableSet


/-- We say a function is `EventuallyMeasurable` with respect to a given
σ-algebra `m` and σ-filter `l` if the preimage of any measurable set is equal to some
`m`-measurable set modulo `l`.
Warning: This is not always the same as being equal to some `m`-measurable function modulo `l`.
In general it is weaker. See `Measurable.eventuallyMeasurable_of_eventuallyEq`.
*TODO*: Add lemmas about when these are equivalent. -/
def EventuallyMeasurable (f : α → β) : Prop := @Measurable _ _ (EventuallyMeasurableSpace m l) _ f


theorem Measurable.eventuallyMeasurable (hf : Measurable f) : EventuallyMeasurable m l f :=
  hf.le EventuallyMeasurableSpace.measurable_le


theorem Measurable.comp_eventuallyMeasurable (hh : Measurable h) (hf : EventuallyMeasurable m l f) :
    EventuallyMeasurable m l (h ∘ f) :=
  hh.comp hf


/-- A function which is `EventuallyEq` to some `EventuallyMeasurable` function
is `EventuallyMeasurable`.-/
theorem EventuallyMeasurable.congr
    (hf : EventuallyMeasurable m l f) (hgf : g =ᶠ[l] f) : EventuallyMeasurable m l g :=
  fun _ hs => EventuallyMeasurableSet.congr (hf hs)
    (hgf.preimage _)


/-- A function which is `EventuallyEq` to some `Measurable` function is `EventuallyMeasurable`.-/
theorem Measurable.eventuallyMeasurable_of_eventuallyEq
    (hf : Measurable f) (hgf : g =ᶠ[l] f) : EventuallyMeasurable m l g :=
  hf.eventuallyMeasurable.congr hgf


