/-- A standard Borel space is a measurable space arising as the Borel sets of some Polish topology.
This is useful in situations where a space has no natural topology or
the natural topology in a space is non-Polish.

To endow a standard Borel space `α` with a compatible Polish topology, use
`letI := upgradeStandardBorel α`. One can then use `eq_borel_upgradeStandardBorel α` to
rewrite the `MeasurableSpace α` instance to `borel α t`, where `t` is the new topology. -/
class StandardBorelSpace [MeasurableSpace α] : Prop where
  /-- There exists a compatible Polish topology. -/
  polish : ∃ _ : TopologicalSpace α, BorelSpace α ∧ PolishSpace α


/-- A convenience class similar to `UpgradedPolishSpace`. No instance should be registered.
Instead one should use `letI := upgradeStandardBorel α`. -/
class UpgradedStandardBorel extends MeasurableSpace α, TopologicalSpace α,
  BorelSpace α, PolishSpace α


/-- Use as `letI := upgradeStandardBorel α` to endow a standard Borel space `α` with
a compatible Polish topology.

Warning: following this with `borelize α` will cause an error. Instead, one can
rewrite with `eq_borel_upgradeStandardBorel α`.
TODO: fix the corresponding bug in `borelize`. -/
noncomputable
def upgradeStandardBorel [MeasurableSpace α] [h : StandardBorelSpace α] :
    UpgradedStandardBorel α := by
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    h : StandardBorelSpace α
    ⊢ UpgradedStandardBorel α
  -/
  choose τ hb hp using h.polish
  /-
    α : Type u_1
    inst✝ : MeasurableSpace α
    h : StandardBorelSpace α
    τ : TopologicalSpace α
    hb : BorelSpace α
    hp : PolishSpace α
    ⊢ UpgradedStandardBorel α
  -/
  constructor
  /-
    🎉 no goals
  -/


/-- The `MeasurableSpace α` instance on a `StandardBorelSpace` `α` is equal to
the borel sets of `upgradeStandardBorel α`. -/
theorem eq_borel_upgradeStandardBorel [MeasurableSpace α] [StandardBorelSpace α] :
    ‹MeasurableSpace α› = @borel _ (upgradeStandardBorel α).toTopologicalSpace :=
  @BorelSpace.measurable_eq _ (upgradeStandardBorel α).toTopologicalSpace _
    (upgradeStandardBorel α).toBorelSpace


instance (priority := 100) standardBorel_of_polish [τ : TopologicalSpace α]
                                                                /-
                                                                  α : Type u_1
                                                                  inst✝² : MeasurableSpace α
                                                                  τ : TopologicalSpace α
                                                                  inst✝¹ : BorelSpace α
                                                                  inst✝ : PolishSpace α
                                                                  ⊢ StandardBorelSpace α
                                                                -/
    [BorelSpace α] [PolishSpace α] : StandardBorelSpace α := by exists τ
                                                                /-
                                                                  🎉 no goals
                                                                -/

-- See note [lower instance priority]

instance (priority := 100) standardBorelSpace_of_discreteMeasurableSpace [DiscreteMeasurableSpace α]
    [Countable α] : StandardBorelSpace α :=
  let _ : TopologicalSpace α := ⊥
  have : DiscreteTopology α := ⟨rfl⟩
  inferInstance

-- See note [lower instance priority]

instance (priority := 100) countablyGenerated_of_standardBorel [StandardBorelSpace α] :
    MeasurableSpace.CountablyGenerated α :=
  letI := upgradeStandardBorel α
  inferInstance

-- See note [lower instance priority]

instance (priority := 100) measurableSingleton_of_standardBorel [StandardBorelSpace α] :
    MeasurableSingletonClass α :=
  letI := upgradeStandardBorel α
  inferInstance


/-- A product of two standard Borel spaces is standard Borel. -/
instance prod [StandardBorelSpace α] [StandardBorelSpace β] : StandardBorelSpace (α × β) :=
  letI := upgradeStandardBorel α
  letI := upgradeStandardBorel β
  inferInstance


/-- A product of countably many standard Borel spaces is standard Borel. -/
instance pi_countable {ι : Type*} [Countable ι] {α : ι → Type*} [∀ n, MeasurableSpace (α n)]
    [∀ n, StandardBorelSpace (α n)] : StandardBorelSpace (∀ n, α n) :=
  letI := fun n => upgradeStandardBorel (α n)
  inferInstance


/-- An analytic set is a set which is the continuous image of some Polish space. There are several
equivalent characterizations of this definition. For the definition, we pick one that avoids
universe issues: a set is analytic if and only if it is a continuous image of `ℕ → ℕ` (or if it
is empty). The above more usual characterization is given
in `analyticSet_iff_exists_polishSpace_range`.

Warning: these are analytic sets in the context of descriptive set theory (which is why they are
registered in the namespace `MeasureTheory`). They have nothing to do with analytic sets in the
context of complex analysis. -/
irreducible_def AnalyticSet (s : Set α) : Prop :=
  s = ∅ ∨ ∃ f : (ℕ → ℕ) → α, Continuous f ∧ range f = s


theorem analyticSet_empty : AnalyticSet (∅ : Set α) := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    ⊢ MeasureTheory.AnalyticSet EmptyCollection.emptyCollection
  -/
  rw [AnalyticSet]
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    ⊢ Or (Eq EmptyCollection.emptyCollection EmptyCollection.emptyCollection) (Exi …
  -/
  exact Or.inl rfl
  /-
    🎉 no goals
  -/


theorem analyticSet_range_of_polishSpace {β : Type*} [TopologicalSpace β] [PolishSpace β]
    {f : β → α} (f_cont : Continuous f) : AnalyticSet (range f) := by
  /-
    α : Type u_1
    inst✝² : TopologicalSpace α
    β : Type u_3
    inst✝¹ : TopologicalSpace β
    inst✝ : PolishSpace β
    f : β → α
    f_cont : Continuous f
    ⊢ MeasureTheory.AnalyticSet (Set.range f)
  -/
  cases isEmpty_or_nonempty β
    /-
      case inl
      α : Type u_1
      inst✝² : TopologicalSpace α
      β : Type u_3
      inst✝¹ : TopologicalSpace β
      inst✝ : PolishSpace β
      f : β → α
      f_cont : Continuous f
      h✝ : IsEmpty β
      ⊢ MeasureTheory.AnalyticSet (Set.range f)
    -/
  · rw [range_eq_empty]
    /-
      case inl
      α : Type u_1
      inst✝² : TopologicalSpace α
      β : Type u_3
      inst✝¹ : TopologicalSpace β
      inst✝ : PolishSpace β
      f : β → α
      f_cont : Continuous f
      h✝ : IsEmpty β
      ⊢ MeasureTheory.AnalyticSet EmptyCollection.emptyCollection
    -/
    exact analyticSet_empty
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝² : TopologicalSpace α
      β : Type u_3
      inst✝¹ : TopologicalSpace β
      inst✝ : PolishSpace β
      f : β → α
      f_cont : Continuous f
      h✝ : Nonempty β
      ⊢ MeasureTheory.AnalyticSet (Set.range f)
    -/
  · rw [AnalyticSet]
    obtain ⟨g, g_cont, hg⟩ : ∃ g : (ℕ → ℕ) → β, Continuous g ∧ Surjective g :=
      exists_nat_nat_continuous_surjective β
    /-
      case inr.intro.intro
      α : Type u_1
      inst✝² : TopologicalSpace α
      β : Type u_3
      inst✝¹ : TopologicalSpace β
      inst✝ : PolishSpace β
      f : β → α
      f_cont : Continuous f
      h✝ : Nonempty β
      g : (Nat → Nat) → β
      g_cont : Continuous g
      hg : Function.Surjective g
      ⊢ Or (Eq (Set.range f) EmptyCollection.emptyCollection) (Exists fun f_1 => And …
    -/
    refine Or.inr ⟨f ∘ g, f_cont.comp g_cont, ?_⟩
    /-
      case inr.intro.intro
      α : Type u_1
      inst✝² : TopologicalSpace α
      β : Type u_3
      inst✝¹ : TopologicalSpace β
      inst✝ : PolishSpace β
      f : β → α
      f_cont : Continuous f
      h✝ : Nonempty β
      g : (Nat → Nat) → β
      g_cont : Continuous g
      hg : Function.Surjective g
      ⊢ Eq (Set.range (Function.comp f g)) (Set.range f)
    -/
    rw [hg.range_comp]
    /-
      🎉 no goals
    -/


/-- The image of an open set under a continuous map is analytic. -/
theorem _root_.IsOpen.analyticSet_image {β : Type*} [TopologicalSpace β] [PolishSpace β]
    {s : Set β} (hs : IsOpen s) {f : β → α} (f_cont : Continuous f) : AnalyticSet (f '' s) := by
  /-
    α : Type u_1
    inst✝² : TopologicalSpace α
    β : Type u_3
    inst✝¹ : TopologicalSpace β
    inst✝ : PolishSpace β
    s : Set β
    hs : IsOpen s
    f : β → α
    f_cont : Continuous f
    ⊢ MeasureTheory.AnalyticSet (Set.image f s)
  -/
  rw [image_eq_range]
  /-
    α : Type u_1
    inst✝² : TopologicalSpace α
    β : Type u_3
    inst✝¹ : TopologicalSpace β
    inst✝ : PolishSpace β
    s : Set β
    hs : IsOpen s
    f : β → α
    f_cont : Continuous f
    ⊢ MeasureTheory.AnalyticSet (Set.range fun x => f ↑x)
  -/
  haveI : PolishSpace s := hs.polishSpace
  /-
    α : Type u_1
    inst✝² : TopologicalSpace α
    β : Type u_3
    inst✝¹ : TopologicalSpace β
    inst✝ : PolishSpace β
    s : Set β
    hs : IsOpen s
    f : β → α
    f_cont : Continuous f
    this : PolishSpace ↑s
    ⊢ MeasureTheory.AnalyticSet (Set.range fun x => f ↑x)
  -/
  exact analyticSet_range_of_polishSpace (f_cont.comp continuous_subtype_val)
  /-
    🎉 no goals
  -/


/-- A set is analytic if and only if it is the continuous image of some Polish space. -/
theorem analyticSet_iff_exists_polishSpace_range {s : Set α} :
    AnalyticSet s ↔
      ∃ (β : Type) (h : TopologicalSpace β) (_ : @PolishSpace β h) (f : β → α),
        @Continuous _ _ h _ f ∧ range f = s := by
  /-
    α : Type u_1
    inst✝ : TopologicalSpace α
    s : Set α
    ⊢ Iff (MeasureTheory.AnalyticSet s) (Exists fun β => Exists fun h => Exists fu …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      inst✝ : TopologicalSpace α
      s : Set α
      ⊢ MeasureTheory.AnalyticSet s → Exists fun β => Exists fun h => Exists fun x = …
    -/
  · intro h
    /-
      case mp
      α : Type u_1
      inst✝ : TopologicalSpace α
      s : Set α
      h : MeasureTheory.AnalyticSet s
      ⊢ Exists fun β => Exists fun h => Exists fun x => Exists fun f => And (Continu …
    -/
    rw [AnalyticSet] at h
    /-
      case mp
      α : Type u_1
      inst✝ : TopologicalSpace α
      s : Set α
      h : Or (Eq s EmptyCollection.emptyCollection) (Exists fun f => And (Continuous …
      ⊢ Exists fun β => Exists fun h => Exists fun x => Exists fun f => And (Continu …
    -/
    cases' h with h h
      /-
        case mp.inl
        α : Type u_1
        inst✝ : TopologicalSpace α
        s : Set α
        h : Eq s EmptyCollection.emptyCollection
        ⊢ Exists fun β => Exists fun h => Exists fun x => Exists fun f => And (Continu …
      -/
    · refine ⟨Empty, inferInstance, inferInstance, Empty.elim, continuous_bot, ?_⟩
      /-
        case mp.inl
        α : Type u_1
        inst✝ : TopologicalSpace α
        s : Set α
        h : Eq s EmptyCollection.emptyCollection
        ⊢ Eq (Set.range Empty.elim) s
      -/
      rw [h]
      /-
        case mp.inl
        α : Type u_1
        inst✝ : TopologicalSpace α
        s : Set α
        h : Eq s EmptyCollection.emptyCollection
        ⊢ Eq (Set.range Empty.elim) EmptyCollection.emptyCollection
      -/
      exact range_eq_empty _
      /-
        🎉 no goals
      -/
      /-
        case mp.inr
        α : Type u_1
        inst✝ : TopologicalSpace α
        s : Set α
        h : Exists fun f => And (Continuous f) (Eq (Set.range f) s)
        ⊢ Exists fun β => Exists fun h => Exists fun x => Exists fun f => And (Continu …
      -/
    · exact ⟨ℕ → ℕ, inferInstance, inferInstance, h⟩
      /-
        🎉 no goals
      -/
    /-
      case mpr
      α : Type u_1
      inst✝ : TopologicalSpace α
      s : Set α
      ⊢ (Exists fun β => Exists fun h => Exists fun x => Exists fun f => And (Contin …
    -/
  · rintro ⟨β, h, h', f, f_cont, f_range⟩
    /-
      case mpr.intro.intro.intro.intro.intro
      α : Type u_1
      inst✝ : TopologicalSpace α
      s : Set α
      β : Type
      h : TopologicalSpace β
      h' : PolishSpace β
      f : β → α
      f_cont : Continuous f
      f_range : Eq (Set.range f) s
      ⊢ MeasureTheory.AnalyticSet s
    -/
    rw [← f_range]
    /-
      case mpr.intro.intro.intro.intro.intro
      α : Type u_1
      inst✝ : TopologicalSpace α
      s : Set α
      β : Type
      h : TopologicalSpace β
      h' : PolishSpace β
      f : β → α
      f_cont : Continuous f
      f_range : Eq (Set.range f) s
      ⊢ MeasureTheory.AnalyticSet (Set.range f)
    -/
    exact analyticSet_range_of_polishSpace f_cont
    /-
      🎉 no goals
    -/


/-- The continuous image of an analytic set is analytic -/
theorem AnalyticSet.image_of_continuousOn {β : Type*} [TopologicalSpace β] {s : Set α}
    (hs : AnalyticSet s) {f : α → β} (hf : ContinuousOn f s) : AnalyticSet (f '' s) := by
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    β : Type u_3
    inst✝ : TopologicalSpace β
    s : Set α
    hs : MeasureTheory.AnalyticSet s
    f : α → β
    hf : ContinuousOn f s
    ⊢ MeasureTheory.AnalyticSet (Set.image f s)
  -/
  rcases analyticSet_iff_exists_polishSpace_range.1 hs with ⟨γ, γtop, γpolish, g, g_cont, gs⟩
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    β : Type u_3
    inst✝ : TopologicalSpace β
    s : Set α
    hs : MeasureTheory.AnalyticSet s
    f : α → β
    hf : ContinuousOn f s
    γ : Type
    γtop : TopologicalSpace γ
    γpolish : PolishSpace γ
    g : γ → α
    g_cont : Continuous g
    gs : Eq (Set.range g) s
    ⊢ MeasureTheory.AnalyticSet (Set.image f s)
  -/
  have : f '' s = range (f ∘ g) := by rw [range_comp, gs]
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    β : Type u_3
    inst✝ : TopologicalSpace β
    s : Set α
    hs : MeasureTheory.AnalyticSet s
    f : α → β
    hf : ContinuousOn f s
    γ : Type
    γtop : TopologicalSpace γ
    γpolish : PolishSpace γ
    g : γ → α
    g_cont : Continuous g
    gs : Eq (Set.range g) s
    this : Eq (Set.image f s) (Set.range (Function.comp f g))
    ⊢ MeasureTheory.AnalyticSet (Set.image f s)
  -/
  rw [this]
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    β : Type u_3
    inst✝ : TopologicalSpace β
    s : Set α
    hs : MeasureTheory.AnalyticSet s
    f : α → β
    hf : ContinuousOn f s
    γ : Type
    γtop : TopologicalSpace γ
    γpolish : PolishSpace γ
    g : γ → α
    g_cont : Continuous g
    gs : Eq (Set.range g) s
    this : Eq (Set.image f s) (Set.range (Function.comp f g))
    ⊢ MeasureTheory.AnalyticSet (Set.range (Function.comp f g))
  -/
  apply analyticSet_range_of_polishSpace
  /-
    case intro.intro.intro.intro.intro.f_cont
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    β : Type u_3
    inst✝ : TopologicalSpace β
    s : Set α
    hs : MeasureTheory.AnalyticSet s
    f : α → β
    hf : ContinuousOn f s
    γ : Type
    γtop : TopologicalSpace γ
    γpolish : PolishSpace γ
    g : γ → α
    g_cont : Continuous g
    gs : Eq (Set.range g) s
    this : Eq (Set.image f s) (Set.range (Function.comp f g))
    ⊢ Continuous (Function.comp f g)
  -/
  apply hf.comp_continuous g_cont fun x => _
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    β : Type u_3
    inst✝ : TopologicalSpace β
    s : Set α
    hs : MeasureTheory.AnalyticSet s
    f : α → β
    hf : ContinuousOn f s
    γ : Type
    γtop : TopologicalSpace γ
    γpolish : PolishSpace γ
    g : γ → α
    g_cont : Continuous g
    gs : Eq (Set.range g) s
    this : Eq (Set.image f s) (Set.range (Function.comp f g))
    ⊢ ∀ (x : γ), Membership.mem s (g x)
  -/
  rw [← gs]
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    β : Type u_3
    inst✝ : TopologicalSpace β
    s : Set α
    hs : MeasureTheory.AnalyticSet s
    f : α → β
    hf : ContinuousOn f s
    γ : Type
    γtop : TopologicalSpace γ
    γpolish : PolishSpace γ
    g : γ → α
    g_cont : Continuous g
    gs : Eq (Set.range g) s
    this : Eq (Set.image f s) (Set.range (Function.comp f g))
    ⊢ ∀ (x : γ), Membership.mem (Set.range g) (g x)
  -/
  exact mem_range_self
  /-
    🎉 no goals
  -/


theorem AnalyticSet.image_of_continuous {β : Type*} [TopologicalSpace β] {s : Set α}
    (hs : AnalyticSet s) {f : α → β} (hf : Continuous f) : AnalyticSet (f '' s) :=
  hs.image_of_continuousOn hf.continuousOn


/-- A countable intersection of analytic sets is analytic. -/
theorem AnalyticSet.iInter [hι : Nonempty ι] [Countable ι] [T2Space α] {s : ι → Set α}
    (hs : ∀ n, AnalyticSet (s n)) : AnalyticSet (⋂ n, s n) := by
  /-
    α : Type u_1
    ι : Type u_2
    inst✝² : TopologicalSpace α
    hι : Nonempty ι
    inst✝¹ : Countable ι
    inst✝ : T2Space α
    s : ι → Set α
    hs : ∀ (n : ι), MeasureTheory.AnalyticSet (s n)
    ⊢ MeasureTheory.AnalyticSet (Set.iInter fun n => s n)
  -/
  rcases hι with ⟨i₀⟩
  /- For the proof, write each `s n` as the continuous image under a map `f n` of a
    Polish space `β n`. The product space `γ = Π n, β n` is also Polish, and so is the subset
    `t` of sequences `x n` for which `f n (x n)` is independent of `n`. The set `t` is Polish, and
    the range of `x ↦ f 0 (x 0)` on `t` is exactly `⋂ n, s n`, so this set is analytic. -/
  choose β hβ h'β f f_cont f_range using fun n =>
    analyticSet_iff_exists_polishSpace_range.1 (hs n)
  /-
    case intro
    α : Type u_1
    ι : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : Countable ι
    inst✝ : T2Space α
    s : ι → Set α
    hs : ∀ (n : ι), MeasureTheory.AnalyticSet (s n)
    i₀ : ι
    β : ι → Type
    hβ : (n : ι) → TopologicalSpace (β n)
    h'β : ∀ (n : ι), PolishSpace (β n)
    f : (n : ι) → β n → α
    f_cont : ∀ (n : ι), Continuous (f n)
    f_range : ∀ (n : ι), Eq (Set.range (f n)) (s n)
    ⊢ MeasureTheory.AnalyticSet (Set.iInter fun n => s n)
  -/
  let γ := ∀ n, β n
  /-
    case intro
    α : Type u_1
    ι : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : Countable ι
    inst✝ : T2Space α
    s : ι → Set α
    hs : ∀ (n : ι), MeasureTheory.AnalyticSet (s n)
    i₀ : ι
    β : ι → Type
    hβ : (n : ι) → TopologicalSpace (β n)
    h'β : ∀ (n : ι), PolishSpace (β n)
    f : (n : ι) → β n → α
    f_cont : ∀ (n : ι), Continuous (f n)
    f_range : ∀ (n : ι), Eq (Set.range (f n)) (s n)
    γ : Type u_2 := (n : ι) → β n
    ⊢ MeasureTheory.AnalyticSet (Set.iInter fun n => s n)
  -/
  let t : Set γ := ⋂ n, { x | f n (x n) = f i₀ (x i₀) }
  have t_closed : IsClosed t := by
    apply isClosed_iInter
    intro n
    exact
      isClosed_eq ((f_cont n).comp (continuous_apply n)) ((f_cont i₀).comp (continuous_apply i₀))
  /-
    case intro
    α : Type u_1
    ι : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : Countable ι
    inst✝ : T2Space α
    s : ι → Set α
    hs : ∀ (n : ι), MeasureTheory.AnalyticSet (s n)
    i₀ : ι
    β : ι → Type
    hβ : (n : ι) → TopologicalSpace (β n)
    h'β : ∀ (n : ι), PolishSpace (β n)
    f : (n : ι) → β n → α
    f_cont : ∀ (n : ι), Continuous (f n)
    f_range : ∀ (n : ι), Eq (Set.range (f n)) (s n)
    γ : Type u_2 := (n : ι) → β n
    t : Set γ := Set.iInter fun n => setOf fun x => Eq (f n (x n)) (f i₀ (x i₀))
    t_closed : IsClosed t
    ⊢ MeasureTheory.AnalyticSet (Set.iInter fun n => s n)
  -/
  haveI : PolishSpace t := t_closed.polishSpace
  /-
    case intro
    α : Type u_1
    ι : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : Countable ι
    inst✝ : T2Space α
    s : ι → Set α
    hs : ∀ (n : ι), MeasureTheory.AnalyticSet (s n)
    i₀ : ι
    β : ι → Type
    hβ : (n : ι) → TopologicalSpace (β n)
    h'β : ∀ (n : ι), PolishSpace (β n)
    f : (n : ι) → β n → α
    f_cont : ∀ (n : ι), Continuous (f n)
    f_range : ∀ (n : ι), Eq (Set.range (f n)) (s n)
    γ : Type u_2 := (n : ι) → β n
    t : Set γ := Set.iInter fun n => setOf fun x => Eq (f n (x n)) (f i₀ (x i₀))
    t_closed : IsClosed t
    this : PolishSpace ↑t
    ⊢ MeasureTheory.AnalyticSet (Set.iInter fun n => s n)
  -/
  let F : t → α := fun x => f i₀ ((x : γ) i₀)
  /-
    case intro
    α : Type u_1
    ι : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : Countable ι
    inst✝ : T2Space α
    s : ι → Set α
    hs : ∀ (n : ι), MeasureTheory.AnalyticSet (s n)
    i₀ : ι
    β : ι → Type
    hβ : (n : ι) → TopologicalSpace (β n)
    h'β : ∀ (n : ι), PolishSpace (β n)
    f : (n : ι) → β n → α
    f_cont : ∀ (n : ι), Continuous (f n)
    f_range : ∀ (n : ι), Eq (Set.range (f n)) (s n)
    γ : Type u_2 := (n : ι) → β n
    t : Set γ := Set.iInter fun n => setOf fun x => Eq (f n (x n)) (f i₀ (x i₀))
    t_closed : IsClosed t
    this : PolishSpace ↑t
    F : ↑t → α := fun x => f i₀ (↑x i₀)
    ⊢ MeasureTheory.AnalyticSet (Set.iInter fun n => s n)
  -/
  have F_cont : Continuous F := (f_cont i₀).comp ((continuous_apply i₀).comp continuous_subtype_val)
  have F_range : range F = ⋂ n : ι, s n := by
    apply Subset.antisymm
    · rintro y ⟨x, rfl⟩
      refine mem_iInter.2 fun n => ?_
      have : f n ((x : γ) n) = F x := (mem_iInter.1 x.2 n : _)
      rw [← this, ← f_range n]
      exact mem_range_self _
    · intro y hy
      have A : ∀ n, ∃ x : β n, f n x = y := by
        intro n
        rw [← mem_range, f_range n]
        exact mem_iInter.1 hy n
      choose x hx using A
      have xt : x ∈ t := by
        refine mem_iInter.2 fun n => ?_
        simp [γ, t, F, hx]
      refine ⟨⟨x, xt⟩, ?_⟩
      exact hx i₀
  /-
    case intro
    α : Type u_1
    ι : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : Countable ι
    inst✝ : T2Space α
    s : ι → Set α
    hs : ∀ (n : ι), MeasureTheory.AnalyticSet (s n)
    i₀ : ι
    β : ι → Type
    hβ : (n : ι) → TopologicalSpace (β n)
    h'β : ∀ (n : ι), PolishSpace (β n)
    f : (n : ι) → β n → α
    f_cont : ∀ (n : ι), Continuous (f n)
    f_range : ∀ (n : ι), Eq (Set.range (f n)) (s n)
    γ : Type u_2 := (n : ι) → β n
    t : Set γ := Set.iInter fun n => setOf fun x => Eq (f n (x n)) (f i₀ (x i₀))
    t_closed : IsClosed t
    this : PolishSpace ↑t
    F : ↑t → α := fun x => f i₀ (↑x i₀)
    F_cont : Continuous F
    F_range : Eq (Set.range F) (Set.iInter fun n => s n)
    ⊢ MeasureTheory.AnalyticSet (Set.iInter fun n => s n)
  -/
  rw [← F_range]
  /-
    case intro
    α : Type u_1
    ι : Type u_2
    inst✝² : TopologicalSpace α
    inst✝¹ : Countable ι
    inst✝ : T2Space α
    s : ι → Set α
    hs : ∀ (n : ι), MeasureTheory.AnalyticSet (s n)
    i₀ : ι
    β : ι → Type
    hβ : (n : ι) → TopologicalSpace (β n)
    h'β : ∀ (n : ι), PolishSpace (β n)
    f : (n : ι) → β n → α
    f_cont : ∀ (n : ι), Continuous (f n)
    f_range : ∀ (n : ι), Eq (Set.range (f n)) (s n)
    γ : Type u_2 := (n : ι) → β n
    t : Set γ := Set.iInter fun n => setOf fun x => Eq (f n (x n)) (f i₀ (x i₀))
    t_closed : IsClosed t
    this : PolishSpace ↑t
    F : ↑t → α := fun x => f i₀ (↑x i₀)
    F_cont : Continuous F
    F_range : Eq (Set.range F) (Set.iInter fun n => s n)
    ⊢ MeasureTheory.AnalyticSet (Set.range F)
  -/
  exact analyticSet_range_of_polishSpace F_cont
  /-
    🎉 no goals
  -/


/-- A countable union of analytic sets is analytic. -/
theorem AnalyticSet.iUnion [Countable ι] {s : ι → Set α} (hs : ∀ n, AnalyticSet (s n)) :
    AnalyticSet (⋃ n, s n) := by
  /- For the proof, write each `s n` as the continuous image under a map `f n` of a
    Polish space `β n`. The union space `γ = Σ n, β n` is also Polish, and the map `F : γ → α` which
    coincides with `f n` on `β n` sends it to `⋃ n, s n`. -/
  choose β hβ h'β f f_cont f_range using fun n =>
    analyticSet_iff_exists_polishSpace_range.1 (hs n)
  /-
    α : Type u_1
    ι : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : Countable ι
    s : ι → Set α
    hs : ∀ (n : ι), MeasureTheory.AnalyticSet (s n)
    β : ι → Type
    hβ : (n : ι) → TopologicalSpace (β n)
    h'β : ∀ (n : ι), PolishSpace (β n)
    f : (n : ι) → β n → α
    f_cont : ∀ (n : ι), Continuous (f n)
    f_range : ∀ (n : ι), Eq (Set.range (f n)) (s n)
    ⊢ MeasureTheory.AnalyticSet (Set.iUnion fun n => s n)
  -/
  let γ := Σn, β n
  /-
    α : Type u_1
    ι : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : Countable ι
    s : ι → Set α
    hs : ∀ (n : ι), MeasureTheory.AnalyticSet (s n)
    β : ι → Type
    hβ : (n : ι) → TopologicalSpace (β n)
    h'β : ∀ (n : ι), PolishSpace (β n)
    f : (n : ι) → β n → α
    f_cont : ∀ (n : ι), Continuous (f n)
    f_range : ∀ (n : ι), Eq (Set.range (f n)) (s n)
    γ : Type (max u_2 0) := Sigma fun n => β n
    ⊢ MeasureTheory.AnalyticSet (Set.iUnion fun n => s n)
  -/
  let F : γ → α := fun ⟨n, x⟩ ↦ f n x
  /-
    α : Type u_1
    ι : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : Countable ι
    s : ι → Set α
    hs : ∀ (n : ι), MeasureTheory.AnalyticSet (s n)
    β : ι → Type
    hβ : (n : ι) → TopologicalSpace (β n)
    h'β : ∀ (n : ι), PolishSpace (β n)
    f : (n : ι) → β n → α
    f_cont : ∀ (n : ι), Continuous (f n)
    f_range : ∀ (n : ι), Eq (Set.range (f n)) (s n)
    γ : Type (max u_2 0) := Sigma fun n => β n
    F : γ → α := fun x => MeasureTheory.AnalyticSet.iUnion.match_1 β (fun x => α)  …
    ⊢ MeasureTheory.AnalyticSet (Set.iUnion fun n => s n)
  -/
  have F_cont : Continuous F := continuous_sigma f_cont
  have F_range : range F = ⋃ n, s n := by
    simp only [γ, F, range_sigma_eq_iUnion_range, f_range]
  /-
    α : Type u_1
    ι : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : Countable ι
    s : ι → Set α
    hs : ∀ (n : ι), MeasureTheory.AnalyticSet (s n)
    β : ι → Type
    hβ : (n : ι) → TopologicalSpace (β n)
    h'β : ∀ (n : ι), PolishSpace (β n)
    f : (n : ι) → β n → α
    f_cont : ∀ (n : ι), Continuous (f n)
    f_range : ∀ (n : ι), Eq (Set.range (f n)) (s n)
    γ : Type (max u_2 0) := Sigma fun n => β n
    F : γ → α := fun x => MeasureTheory.AnalyticSet.iUnion.match_1 β (fun x => α)  …
    F_cont : Continuous F
    F_range : Eq (Set.range F) (Set.iUnion fun n => s n)
    ⊢ MeasureTheory.AnalyticSet (Set.iUnion fun n => s n)
  -/
  rw [← F_range]
  /-
    α : Type u_1
    ι : Type u_2
    inst✝¹ : TopologicalSpace α
    inst✝ : Countable ι
    s : ι → Set α
    hs : ∀ (n : ι), MeasureTheory.AnalyticSet (s n)
    β : ι → Type
    hβ : (n : ι) → TopologicalSpace (β n)
    h'β : ∀ (n : ι), PolishSpace (β n)
    f : (n : ι) → β n → α
    f_cont : ∀ (n : ι), Continuous (f n)
    f_range : ∀ (n : ι), Eq (Set.range (f n)) (s n)
    γ : Type (max u_2 0) := Sigma fun n => β n
    F : γ → α := fun x => MeasureTheory.AnalyticSet.iUnion.match_1 β (fun x => α)  …
    F_cont : Continuous F
    F_range : Eq (Set.range F) (Set.iUnion fun n => s n)
    ⊢ MeasureTheory.AnalyticSet (Set.range F)
  -/
  exact analyticSet_range_of_polishSpace F_cont
  /-
    🎉 no goals
  -/


theorem _root_.IsClosed.analyticSet [PolishSpace α] {s : Set α} (hs : IsClosed s) :
    AnalyticSet s := by
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : PolishSpace α
    s : Set α
    hs : IsClosed s
    ⊢ MeasureTheory.AnalyticSet s
  -/
  haveI : PolishSpace s := hs.polishSpace
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : PolishSpace α
    s : Set α
    hs : IsClosed s
    this : PolishSpace ↑s
    ⊢ MeasureTheory.AnalyticSet s
  -/
  rw [← @Subtype.range_val α s]
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : PolishSpace α
    s : Set α
    hs : IsClosed s
    this : PolishSpace ↑s
    ⊢ MeasureTheory.AnalyticSet (Set.range Subtype.val)
  -/
  exact analyticSet_range_of_polishSpace continuous_subtype_val
  /-
    🎉 no goals
  -/


/-- Given a Borel-measurable set in a Polish space, there exists a finer Polish topology making
it clopen. This is in fact an equivalence, see `isClopenable_iff_measurableSet`. -/
theorem _root_.MeasurableSet.isClopenable [PolishSpace α] [MeasurableSpace α] [BorelSpace α]
    {s : Set α} (hs : MeasurableSet s) : IsClopenable s := by
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : PolishSpace α
    inst✝¹ : MeasurableSpace α
    inst✝ : BorelSpace α
    s : Set α
    hs : MeasurableSet s
    ⊢ PolishSpace.IsClopenable s
  -/
  revert s
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : PolishSpace α
    inst✝¹ : MeasurableSpace α
    inst✝ : BorelSpace α
    ⊢ ∀ {s : Set α}, MeasurableSet s → PolishSpace.IsClopenable s
  -/
  apply MeasurableSet.induction_on_open
    /-
      case isOpen
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : PolishSpace α
      inst✝¹ : MeasurableSpace α
      inst✝ : BorelSpace α
      ⊢ ∀ (U : Set α), IsOpen U → PolishSpace.IsClopenable U
    -/
  · exact fun u hu => hu.isClopenable
    /-
      🎉 no goals
    -/
    /-
      case compl
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : PolishSpace α
      inst✝¹ : MeasurableSpace α
      inst✝ : BorelSpace α
      ⊢ ∀ (t : Set α), MeasurableSet t → PolishSpace.IsClopenable t → PolishSpace.Is …
    -/
  · exact fun u _ h'u => h'u.compl
    /-
      🎉 no goals
    -/
    /-
      case iUnion
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : PolishSpace α
      inst✝¹ : MeasurableSpace α
      inst✝ : BorelSpace α
      ⊢ ∀ (f : Nat → Set α), Pairwise (Function.onFun Disjoint f) → (∀ (i : Nat), Me …
    -/
  · exact fun f _ _ hf => IsClopenable.iUnion hf
    /-
      🎉 no goals
    -/


/-- A Borel-measurable set in a Polish space is analytic. -/
theorem _root_.MeasurableSet.analyticSet {α : Type*} [t : TopologicalSpace α] [PolishSpace α]
    [MeasurableSpace α] [BorelSpace α] {s : Set α} (hs : MeasurableSet s) : AnalyticSet s := by
  /- For a short proof (avoiding measurable induction), one sees `s` as a closed set for a finer
    topology `t'`. It is analytic for this topology. As the identity from `t'` to `t` is continuous
    and the image of an analytic set is analytic, it follows that `s` is also analytic for `t`. -/
  obtain ⟨t', t't, t'_polish, s_closed, _⟩ :
      ∃ t' : TopologicalSpace α, t' ≤ t ∧ @PolishSpace α t' ∧ IsClosed[t'] s ∧ IsOpen[t'] s :=
    hs.isClopenable
  /-
    case intro.intro.intro.intro
    α : Type u_3
    t : TopologicalSpace α
    inst✝² : PolishSpace α
    inst✝¹ : MeasurableSpace α
    inst✝ : BorelSpace α
    s : Set α
    hs : MeasurableSet s
    t' : TopologicalSpace α
    t't : LE.le t' t
    t'_polish : PolishSpace α
    s_closed : IsClosed s
    right✝ : IsOpen s
    ⊢ MeasureTheory.AnalyticSet s
  -/
  have A := @IsClosed.analyticSet α t' t'_polish s s_closed
  /-
    case intro.intro.intro.intro
    α : Type u_3
    t : TopologicalSpace α
    inst✝² : PolishSpace α
    inst✝¹ : MeasurableSpace α
    inst✝ : BorelSpace α
    s : Set α
    hs : MeasurableSet s
    t' : TopologicalSpace α
    t't : LE.le t' t
    t'_polish : PolishSpace α
    s_closed : IsClosed s
    right✝ : IsOpen s
    A : MeasureTheory.AnalyticSet s
    ⊢ MeasureTheory.AnalyticSet s
  -/
  convert @AnalyticSet.image_of_continuous α t' α t s A id (continuous_id_of_le t't)
  /-
    case h.e'_3
    α : Type u_3
    t : TopologicalSpace α
    inst✝² : PolishSpace α
    inst✝¹ : MeasurableSpace α
    inst✝ : BorelSpace α
    s : Set α
    hs : MeasurableSet s
    t' : TopologicalSpace α
    t't : LE.le t' t
    t'_polish : PolishSpace α
    s_closed : IsClosed s
    right✝ : IsOpen s
    A : MeasureTheory.AnalyticSet s
    ⊢ Eq s (Set.image id s)
  -/
  simp only [id, image_id']
  /-
    🎉 no goals
  -/


/-- Given a Borel-measurable function from a Polish space to a second-countable space, there exists
a finer Polish topology on the source space for which the function is continuous. -/
theorem _root_.Measurable.exists_continuous {α β : Type*} [t : TopologicalSpace α] [PolishSpace α]
    [MeasurableSpace α] [BorelSpace α] [tβ : TopologicalSpace β] [MeasurableSpace β]
    [OpensMeasurableSpace β] {f : α → β} [SecondCountableTopology (range f)] (hf : Measurable f) :
    ∃ t' : TopologicalSpace α, t' ≤ t ∧ @Continuous α β t' tβ f ∧ @PolishSpace α t' := by
  obtain ⟨b, b_count, -, hb⟩ :
      ∃ b : Set (Set (range f)), b.Countable ∧ ∅ ∉ b ∧ IsTopologicalBasis b :=
    exists_countable_basis (range f)
  /-
    case intro.intro.intro
    α : Type u_3
    β : Type u_4
    t : TopologicalSpace α
    inst✝⁵ : PolishSpace α
    inst✝⁴ : MeasurableSpace α
    inst✝³ : BorelSpace α
    tβ : TopologicalSpace β
    inst✝² : MeasurableSpace β
    inst✝¹ : OpensMeasurableSpace β
    f : α → β
    inst✝ : SecondCountableTopology ↑(Set.range f)
    hf : Measurable f
    b : Set (Set ↑(Set.range f))
    b_count : b.Countable
    hb : TopologicalSpace.IsTopologicalBasis b
    ⊢ Exists fun t' => And (LE.le t' t) (And (Continuous f) (PolishSpace α))
  -/
  haveI : Countable b := b_count.to_subtype
  have : ∀ s : b, IsClopenable (rangeFactorization f ⁻¹' s) := fun s ↦ by
    apply MeasurableSet.isClopenable
    exact hf.subtype_mk (hb.isOpen s.2).measurableSet
  /-
    case intro.intro.intro
    α : Type u_3
    β : Type u_4
    t : TopologicalSpace α
    inst✝⁵ : PolishSpace α
    inst✝⁴ : MeasurableSpace α
    inst✝³ : BorelSpace α
    tβ : TopologicalSpace β
    inst✝² : MeasurableSpace β
    inst✝¹ : OpensMeasurableSpace β
    f : α → β
    inst✝ : SecondCountableTopology ↑(Set.range f)
    hf : Measurable f
    b : Set (Set ↑(Set.range f))
    b_count : b.Countable
    hb : TopologicalSpace.IsTopologicalBasis b
    this✝ : Countable ↑b
    this : ∀ (s : ↑b), PolishSpace.IsClopenable (Set.preimage (Set.rangeFactorizat …
    ⊢ Exists fun t' => And (LE.le t' t) (And (Continuous f) (PolishSpace α))
  -/
  choose T Tt Tpolish _ Topen using this
  obtain ⟨t', t'T, t't, t'_polish⟩ :
      ∃ t' : TopologicalSpace α, (∀ i, t' ≤ T i) ∧ t' ≤ t ∧ @PolishSpace α t' :=
    exists_polishSpace_forall_le (t := t) T Tt Tpolish
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_3
    β : Type u_4
    t : TopologicalSpace α
    inst✝⁵ : PolishSpace α
    inst✝⁴ : MeasurableSpace α
    inst✝³ : BorelSpace α
    tβ : TopologicalSpace β
    inst✝² : MeasurableSpace β
    inst✝¹ : OpensMeasurableSpace β
    f : α → β
    inst✝ : SecondCountableTopology ↑(Set.range f)
    hf : Measurable f
    b : Set (Set ↑(Set.range f))
    b_count : b.Countable
    hb : TopologicalSpace.IsTopologicalBasis b
    this : Countable ↑b
    T : ↑b → TopologicalSpace α
    Tt : ∀ (s : ↑b), LE.le (T s) t
    Tpolish : ∀ (s : ↑b), PolishSpace α
    h✝ : ∀ (s : ↑b), IsClosed (Set.preimage (Set.rangeFactorization f) ↑s)
    Topen : ∀ (s : ↑b), IsOpen (Set.preimage (Set.rangeFactorization f) ↑s)
    t' : TopologicalSpace α
    t'T : ∀ (i : ↑b), LE.le t' (T i)
    t't : LE.le t' t
    t'_polish : PolishSpace α
    ⊢ Exists fun t' => And (LE.le t' t) (And (Continuous f) (PolishSpace α))
  -/
  refine ⟨t', t't, ?_, t'_polish⟩
  have : Continuous[t', _] (rangeFactorization f) :=
    hb.continuous_iff.2 fun s hs => t'T ⟨s, hs⟩ _ (Topen ⟨s, hs⟩)
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_3
    β : Type u_4
    t : TopologicalSpace α
    inst✝⁵ : PolishSpace α
    inst✝⁴ : MeasurableSpace α
    inst✝³ : BorelSpace α
    tβ : TopologicalSpace β
    inst✝² : MeasurableSpace β
    inst✝¹ : OpensMeasurableSpace β
    f : α → β
    inst✝ : SecondCountableTopology ↑(Set.range f)
    hf : Measurable f
    b : Set (Set ↑(Set.range f))
    b_count : b.Countable
    hb : TopologicalSpace.IsTopologicalBasis b
    this✝ : Countable ↑b
    T : ↑b → TopologicalSpace α
    Tt : ∀ (s : ↑b), LE.le (T s) t
    Tpolish : ∀ (s : ↑b), PolishSpace α
    h✝ : ∀ (s : ↑b), IsClosed (Set.preimage (Set.rangeFactorization f) ↑s)
    Topen : ∀ (s : ↑b), IsOpen (Set.preimage (Set.rangeFactorization f) ↑s)
    t' : TopologicalSpace α
    t'T : ∀ (i : ↑b), LE.le t' (T i)
    t't : LE.le t' t
    t'_polish : PolishSpace α
    this : Continuous (Set.rangeFactorization f)
    ⊢ Continuous f
  -/
  exact continuous_subtype_val.comp this
  /-
    🎉 no goals
  -/


/-- The image of a measurable set in a standard Borel space under a measurable map
is an analytic set. -/
theorem _root_.MeasurableSet.analyticSet_image {X Y : Type*} [MeasurableSpace X]
    [StandardBorelSpace X] [TopologicalSpace Y] [MeasurableSpace Y]
    [OpensMeasurableSpace Y] {f : X → Y} [SecondCountableTopology (range f)] {s : Set X}
    (hs : MeasurableSet s) (hf : Measurable f) : AnalyticSet (f '' s) := by
  /-
    X : Type u_3
    Y : Type u_4
    inst✝⁵ : MeasurableSpace X
    inst✝⁴ : StandardBorelSpace X
    inst✝³ : TopologicalSpace Y
    inst✝² : MeasurableSpace Y
    inst✝¹ : OpensMeasurableSpace Y
    f : X → Y
    inst✝ : SecondCountableTopology ↑(Set.range f)
    s : Set X
    hs : MeasurableSet s
    hf : Measurable f
    ⊢ MeasureTheory.AnalyticSet (Set.image f s)
  -/
  letI := upgradeStandardBorel X
  /-
    X : Type u_3
    Y : Type u_4
    inst✝⁵ : MeasurableSpace X
    inst✝⁴ : StandardBorelSpace X
    inst✝³ : TopologicalSpace Y
    inst✝² : MeasurableSpace Y
    inst✝¹ : OpensMeasurableSpace Y
    f : X → Y
    inst✝ : SecondCountableTopology ↑(Set.range f)
    s : Set X
    hs : MeasurableSet s
    hf : Measurable f
    this : UpgradedStandardBorel X := upgradeStandardBorel X
    ⊢ MeasureTheory.AnalyticSet (Set.image f s)
  -/
  rw [eq_borel_upgradeStandardBorel X] at hs
  /-
    X : Type u_3
    Y : Type u_4
    inst✝⁵ : MeasurableSpace X
    inst✝⁴ : StandardBorelSpace X
    inst✝³ : TopologicalSpace Y
    inst✝² : MeasurableSpace Y
    inst✝¹ : OpensMeasurableSpace Y
    f : X → Y
    inst✝ : SecondCountableTopology ↑(Set.range f)
    s : Set X
    hs : MeasurableSet s
    hf : Measurable f
    this : UpgradedStandardBorel X := upgradeStandardBorel X
    ⊢ MeasureTheory.AnalyticSet (Set.image f s)
  -/
  rcases hf.exists_continuous with ⟨τ', hle, hfc, hτ'⟩
  /-
    case intro.intro.intro
    X : Type u_3
    Y : Type u_4
    inst✝⁵ : MeasurableSpace X
    inst✝⁴ : StandardBorelSpace X
    inst✝³ : TopologicalSpace Y
    inst✝² : MeasurableSpace Y
    inst✝¹ : OpensMeasurableSpace Y
    f : X → Y
    inst✝ : SecondCountableTopology ↑(Set.range f)
    s : Set X
    hs : MeasurableSet s
    hf : Measurable f
    this : UpgradedStandardBorel X := upgradeStandardBorel X
    τ' : TopologicalSpace X
    hle : LE.le τ' UpgradedStandardBorel.toTopologicalSpace
    hfc : Continuous f
    hτ' : PolishSpace X
    ⊢ MeasureTheory.AnalyticSet (Set.image f s)
  -/
  letI m' : MeasurableSpace X := @borel _ τ'
  /-
    case intro.intro.intro
    X : Type u_3
    Y : Type u_4
    inst✝⁵ : MeasurableSpace X
    inst✝⁴ : StandardBorelSpace X
    inst✝³ : TopologicalSpace Y
    inst✝² : MeasurableSpace Y
    inst✝¹ : OpensMeasurableSpace Y
    f : X → Y
    inst✝ : SecondCountableTopology ↑(Set.range f)
    s : Set X
    hs : MeasurableSet s
    hf : Measurable f
    this : UpgradedStandardBorel X := upgradeStandardBorel X
    τ' : TopologicalSpace X
    hle : LE.le τ' UpgradedStandardBorel.toTopologicalSpace
    hfc : Continuous f
    hτ' : PolishSpace X
    m' : MeasurableSpace X := borel X
    ⊢ MeasureTheory.AnalyticSet (Set.image f s)
  -/
  haveI b' : BorelSpace X := ⟨rfl⟩
  /-
    case intro.intro.intro
    X : Type u_3
    Y : Type u_4
    inst✝⁵ : MeasurableSpace X
    inst✝⁴ : StandardBorelSpace X
    inst✝³ : TopologicalSpace Y
    inst✝² : MeasurableSpace Y
    inst✝¹ : OpensMeasurableSpace Y
    f : X → Y
    inst✝ : SecondCountableTopology ↑(Set.range f)
    s : Set X
    hs : MeasurableSet s
    hf : Measurable f
    this : UpgradedStandardBorel X := upgradeStandardBorel X
    τ' : TopologicalSpace X
    hle : LE.le τ' UpgradedStandardBorel.toTopologicalSpace
    hfc : Continuous f
    hτ' : PolishSpace X
    m' : MeasurableSpace X := borel X
    b' : BorelSpace X
    ⊢ MeasureTheory.AnalyticSet (Set.image f s)
  -/
  have hle := borel_anti hle
  /-
    case intro.intro.intro
    X : Type u_3
    Y : Type u_4
    inst✝⁵ : MeasurableSpace X
    inst✝⁴ : StandardBorelSpace X
    inst✝³ : TopologicalSpace Y
    inst✝² : MeasurableSpace Y
    inst✝¹ : OpensMeasurableSpace Y
    f : X → Y
    inst✝ : SecondCountableTopology ↑(Set.range f)
    s : Set X
    hs : MeasurableSet s
    hf : Measurable f
    this : UpgradedStandardBorel X := upgradeStandardBorel X
    τ' : TopologicalSpace X
    hle✝ : LE.le τ' UpgradedStandardBorel.toTopologicalSpace
    hfc : Continuous f
    hτ' : PolishSpace X
    m' : MeasurableSpace X := borel X
    b' : BorelSpace X
    hle : LE.le (borel X) (borel X)
    ⊢ MeasureTheory.AnalyticSet (Set.image f s)
  -/
  exact (hle _ hs).analyticSet.image_of_continuous hfc
  /-
    🎉 no goals
  -/


/-- Preimage of an analytic set is an analytic set. -/
protected lemma AnalyticSet.preimage {X Y : Type*} [TopologicalSpace X] [TopologicalSpace Y]
    [PolishSpace X] [T2Space Y] {s : Set Y} (hs : AnalyticSet s) {f : X → Y} (hf : Continuous f) :
    AnalyticSet (f ⁻¹' s) := by
  /-
    X : Type u_3
    Y : Type u_4
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : PolishSpace X
    inst✝ : T2Space Y
    s : Set Y
    hs : MeasureTheory.AnalyticSet s
    f : X → Y
    hf : Continuous f
    ⊢ MeasureTheory.AnalyticSet (Set.preimage f s)
  -/
  rcases analyticSet_iff_exists_polishSpace_range.1 hs with ⟨Z, _, _, g, hg, rfl⟩
  /-
    case intro.intro.intro.intro.intro
    X : Type u_3
    Y : Type u_4
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : PolishSpace X
    inst✝ : T2Space Y
    f : X → Y
    hf : Continuous f
    Z : Type
    w✝¹ : TopologicalSpace Z
    w✝ : PolishSpace Z
    g : Z → Y
    hg : Continuous g
    hs : MeasureTheory.AnalyticSet (Set.range g)
    ⊢ MeasureTheory.AnalyticSet (Set.preimage f (Set.range g))
  -/
  have : IsClosed {x : X × Z | f x.1 = g x.2} := isClosed_eq hf.fst' hg.snd'
  /-
    case intro.intro.intro.intro.intro
    X : Type u_3
    Y : Type u_4
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : PolishSpace X
    inst✝ : T2Space Y
    f : X → Y
    hf : Continuous f
    Z : Type
    w✝¹ : TopologicalSpace Z
    w✝ : PolishSpace Z
    g : Z → Y
    hg : Continuous g
    hs : MeasureTheory.AnalyticSet (Set.range g)
    this : IsClosed (setOf fun x => Eq (f x.1) (g x.2))
    ⊢ MeasureTheory.AnalyticSet (Set.preimage f (Set.range g))
  -/
  convert this.analyticSet.image_of_continuous continuous_fst
  /-
    case h.e'_3
    X : Type u_3
    Y : Type u_4
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : PolishSpace X
    inst✝ : T2Space Y
    f : X → Y
    hf : Continuous f
    Z : Type
    w✝¹ : TopologicalSpace Z
    w✝ : PolishSpace Z
    g : Z → Y
    hg : Continuous g
    hs : MeasureTheory.AnalyticSet (Set.range g)
    this : IsClosed (setOf fun x => Eq (f x.1) (g x.2))
    ⊢ Eq (Set.preimage f (Set.range g)) (Set.image Prod.fst (setOf fun x => Eq (f  …
  -/
  ext x
  /-
    case h.e'_3.h
    X : Type u_3
    Y : Type u_4
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : PolishSpace X
    inst✝ : T2Space Y
    f : X → Y
    hf : Continuous f
    Z : Type
    w✝¹ : TopologicalSpace Z
    w✝ : PolishSpace Z
    g : Z → Y
    hg : Continuous g
    hs : MeasureTheory.AnalyticSet (Set.range g)
    this : IsClosed (setOf fun x => Eq (f x.1) (g x.2))
    x : X
    ⊢ Iff (Membership.mem (Set.preimage f (Set.range g)) x) (Membership.mem (Set.i …
  -/
  simp [eq_comm]
  /-
    🎉 no goals
  -/


/-- Two sets `u` and `v` in a measurable space are measurably separable if there
exists a measurable set containing `u` and disjoint from `v`.
This is mostly interesting for Borel-separable sets. -/
def MeasurablySeparable {α : Type*} [MeasurableSpace α] (s t : Set α) : Prop :=
  ∃ u, s ⊆ u ∧ Disjoint t u ∧ MeasurableSet u


theorem MeasurablySeparable.iUnion [Countable ι] {α : Type*} [MeasurableSpace α] {s t : ι → Set α}
    (h : ∀ m n, MeasurablySeparable (s m) (t n)) : MeasurablySeparable (⋃ n, s n) (⋃ m, t m) := by
  /-
    ι : Type u_2
    inst✝¹ : Countable ι
    α : Type u_3
    inst✝ : MeasurableSpace α
    s t : ι → Set α
    h : ∀ (m n : ι), MeasureTheory.MeasurablySeparable (s m) (t n)
    ⊢ MeasureTheory.MeasurablySeparable (Set.iUnion fun n => s n) (Set.iUnion fun  …
  -/
  choose u hsu htu hu using h
  /-
    ι : Type u_2
    inst✝¹ : Countable ι
    α : Type u_3
    inst✝ : MeasurableSpace α
    s t : ι → Set α
    u : ι → ι → Set α
    hsu : ∀ (m n : ι), HasSubset.Subset (s m) (u m n)
    htu : ∀ (m n : ι), Disjoint (t n) (u m n)
    hu : ∀ (m n : ι), MeasurableSet (u m n)
    ⊢ MeasureTheory.MeasurablySeparable (Set.iUnion fun n => s n) (Set.iUnion fun  …
  -/
  refine ⟨⋃ m, ⋂ n, u m n, ?_, ?_, ?_⟩
    /-
      case refine_1
      ι : Type u_2
      inst✝¹ : Countable ι
      α : Type u_3
      inst✝ : MeasurableSpace α
      s t : ι → Set α
      u : ι → ι → Set α
      hsu : ∀ (m n : ι), HasSubset.Subset (s m) (u m n)
      htu : ∀ (m n : ι), Disjoint (t n) (u m n)
      hu : ∀ (m n : ι), MeasurableSet (u m n)
      ⊢ HasSubset.Subset (Set.iUnion fun n => s n) (Set.iUnion fun m => Set.iInter f …
    -/
  · refine iUnion_subset fun m => subset_iUnion_of_subset m ?_
    /-
      case refine_1
      ι : Type u_2
      inst✝¹ : Countable ι
      α : Type u_3
      inst✝ : MeasurableSpace α
      s t : ι → Set α
      u : ι → ι → Set α
      hsu : ∀ (m n : ι), HasSubset.Subset (s m) (u m n)
      htu : ∀ (m n : ι), Disjoint (t n) (u m n)
      hu : ∀ (m n : ι), MeasurableSet (u m n)
      m : ι
      ⊢ HasSubset.Subset (s m) (Set.iInter fun n => u m n)
    -/
    exact subset_iInter fun n => hsu m n
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ι : Type u_2
      inst✝¹ : Countable ι
      α : Type u_3
      inst✝ : MeasurableSpace α
      s t : ι → Set α
      u : ι → ι → Set α
      hsu : ∀ (m n : ι), HasSubset.Subset (s m) (u m n)
      htu : ∀ (m n : ι), Disjoint (t n) (u m n)
      hu : ∀ (m n : ι), MeasurableSet (u m n)
      ⊢ Disjoint (Set.iUnion fun m => t m) (Set.iUnion fun m => Set.iInter fun n =>  …
    -/
  · simp_rw [disjoint_iUnion_left, disjoint_iUnion_right]
    /-
      case refine_2
      ι : Type u_2
      inst✝¹ : Countable ι
      α : Type u_3
      inst✝ : MeasurableSpace α
      s t : ι → Set α
      u : ι → ι → Set α
      hsu : ∀ (m n : ι), HasSubset.Subset (s m) (u m n)
      htu : ∀ (m n : ι), Disjoint (t n) (u m n)
      hu : ∀ (m n : ι), MeasurableSet (u m n)
      ⊢ ∀ (i i_1 : ι), Disjoint (t i) (Set.iInter fun n => u i_1 n)
    -/
    intro n m
    /-
      case refine_2
      ι : Type u_2
      inst✝¹ : Countable ι
      α : Type u_3
      inst✝ : MeasurableSpace α
      s t : ι → Set α
      u : ι → ι → Set α
      hsu : ∀ (m n : ι), HasSubset.Subset (s m) (u m n)
      htu : ∀ (m n : ι), Disjoint (t n) (u m n)
      hu : ∀ (m n : ι), MeasurableSet (u m n)
      n m : ι
      ⊢ Disjoint (t n) (Set.iInter fun n => u m n)
    -/
    apply Disjoint.mono_right _ (htu m n)
    /-
      ι : Type u_2
      inst✝¹ : Countable ι
      α : Type u_3
      inst✝ : MeasurableSpace α
      s t : ι → Set α
      u : ι → ι → Set α
      hsu : ∀ (m n : ι), HasSubset.Subset (s m) (u m n)
      htu : ∀ (m n : ι), Disjoint (t n) (u m n)
      hu : ∀ (m n : ι), MeasurableSet (u m n)
      n m : ι
      ⊢ LE.le (Set.iInter fun n => u m n) (u m n)
    -/
    apply iInter_subset
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      ι : Type u_2
      inst✝¹ : Countable ι
      α : Type u_3
      inst✝ : MeasurableSpace α
      s t : ι → Set α
      u : ι → ι → Set α
      hsu : ∀ (m n : ι), HasSubset.Subset (s m) (u m n)
      htu : ∀ (m n : ι), Disjoint (t n) (u m n)
      hu : ∀ (m n : ι), MeasurableSet (u m n)
      ⊢ MeasurableSet (Set.iUnion fun m => Set.iInter fun n => u m n)
    -/
  · refine MeasurableSet.iUnion fun m => ?_
    /-
      case refine_3
      ι : Type u_2
      inst✝¹ : Countable ι
      α : Type u_3
      inst✝ : MeasurableSpace α
      s t : ι → Set α
      u : ι → ι → Set α
      hsu : ∀ (m n : ι), HasSubset.Subset (s m) (u m n)
      htu : ∀ (m n : ι), Disjoint (t n) (u m n)
      hu : ∀ (m n : ι), MeasurableSet (u m n)
      m : ι
      ⊢ MeasurableSet (Set.iInter fun n => u m n)
    -/
    exact MeasurableSet.iInter fun n => hu m n
    /-
      🎉 no goals
    -/


/-- The hard part of the Lusin separation theorem saying that two disjoint analytic sets are
contained in disjoint Borel sets (see the full statement in `AnalyticSet.measurablySeparable`).
Here, we prove this when our analytic sets are the ranges of functions from `ℕ → ℕ`.
-/
theorem measurablySeparable_range_of_disjoint [T2Space α] [MeasurableSpace α]
    [OpensMeasurableSpace α] {f g : (ℕ → ℕ) → α} (hf : Continuous f) (hg : Continuous g)
    (h : Disjoint (range f) (range g)) : MeasurablySeparable (range f) (range g) := by
  /- We follow [Kechris, *Classical Descriptive Set Theory* (Theorem 14.7)][kechris1995].
    If the ranges are not Borel-separated, then one can find two cylinders of length one whose
    images are not Borel-separated, and then two smaller cylinders of length two whose images are
    not Borel-separated, and so on. One thus gets two sequences of cylinders, that decrease to two
    points `x` and `y`. Their images are different by the disjointness assumption, hence contained
    in two disjoint open sets by the T2 property. By continuity, long enough cylinders around `x`
    and `y` have images which are separated by these two disjoint open sets, a contradiction.
    -/
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : T2Space α
    inst✝¹ : MeasurableSpace α
    inst✝ : OpensMeasurableSpace α
    f g : (Nat → Nat) → α
    hf : Continuous f
    hg : Continuous g
    h : Disjoint (Set.range f) (Set.range g)
    ⊢ MeasureTheory.MeasurablySeparable (Set.range f) (Set.range g)
  -/
  by_contra hfg
  have I : ∀ n x y, ¬MeasurablySeparable (f '' cylinder x n) (g '' cylinder y n) →
      ∃ x' y', x' ∈ cylinder x n ∧ y' ∈ cylinder y n ∧
      ¬MeasurablySeparable (f '' cylinder x' (n + 1)) (g '' cylinder y' (n + 1)) := by
    intro n x y
    contrapose!
    intro H
    rw [← iUnion_cylinder_update x n, ← iUnion_cylinder_update y n, image_iUnion, image_iUnion]
    refine MeasurablySeparable.iUnion fun i j => ?_
    exact H _ _ (update_mem_cylinder _ _ _) (update_mem_cylinder _ _ _)
  -- consider the set of pairs of cylinders of some length whose images are not Borel-separated
  let A :=
    { p : ℕ × (ℕ → ℕ) × (ℕ → ℕ) //
      ¬MeasurablySeparable (f '' cylinder p.2.1 p.1) (g '' cylinder p.2.2 p.1) }
  -- for each such pair, one can find longer cylinders whose images are not Borel-separated either
  have : ∀ p : A, ∃ q : A,
      q.1.1 = p.1.1 + 1 ∧ q.1.2.1 ∈ cylinder p.1.2.1 p.1.1 ∧ q.1.2.2 ∈ cylinder p.1.2.2 p.1.1 := by
    rintro ⟨⟨n, x, y⟩, hp⟩
    rcases I n x y hp with ⟨x', y', hx', hy', h'⟩
    exact ⟨⟨⟨n + 1, x', y'⟩, h'⟩, rfl, hx', hy'⟩
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : T2Space α
    inst✝¹ : MeasurableSpace α
    inst✝ : OpensMeasurableSpace α
    f g : (Nat → Nat) → α
    hf : Continuous f
    hg : Continuous g
    h : Disjoint (Set.range f) (Set.range g)
    hfg : Not (MeasureTheory.MeasurablySeparable (Set.range f) (Set.range g))
    I : ∀ (n : Nat) (x y : Nat → Nat), Not (MeasureTheory.MeasurablySeparable (Set …
    A : Type := Subtype fun p => Not (MeasureTheory.MeasurablySeparable (Set.image …
    this : ∀ (p : A), Exists fun q => And (Eq (↑q).1 (HAdd.hAdd (↑p).1 1)) (And (M …
    ⊢ False
  -/
  choose F hFn hFx hFy using this
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : T2Space α
    inst✝¹ : MeasurableSpace α
    inst✝ : OpensMeasurableSpace α
    f g : (Nat → Nat) → α
    hf : Continuous f
    hg : Continuous g
    h : Disjoint (Set.range f) (Set.range g)
    hfg : Not (MeasureTheory.MeasurablySeparable (Set.range f) (Set.range g))
    I : ∀ (n : Nat) (x y : Nat → Nat), Not (MeasureTheory.MeasurablySeparable (Set …
    A : Type := Subtype fun p => Not (MeasureTheory.MeasurablySeparable (Set.image …
    F : A → A
    hFn : ∀ (p : A), Eq (↑(F p)).1 (HAdd.hAdd (↑p).1 1)
    hFx : ∀ (p : A), Membership.mem (PiNat.cylinder (↑p).2.1 (↑p).1) (↑(F p)).2.1
    hFy : ∀ (p : A), Membership.mem (PiNat.cylinder (↑p).2.2 (↑p).1) (↑(F p)).2.2
    ⊢ False
  -/
  let p0 : A := ⟨⟨0, fun _ => 0, fun _ => 0⟩, by simp [hfg]⟩
  -- construct inductively decreasing sequences of cylinders whose images are not separated
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : T2Space α
    inst✝¹ : MeasurableSpace α
    inst✝ : OpensMeasurableSpace α
    f g : (Nat → Nat) → α
    hf : Continuous f
    hg : Continuous g
    h : Disjoint (Set.range f) (Set.range g)
    hfg : Not (MeasureTheory.MeasurablySeparable (Set.range f) (Set.range g))
    I : ∀ (n : Nat) (x y : Nat → Nat), Not (MeasureTheory.MeasurablySeparable (Set …
    A : Type := Subtype fun p => Not (MeasureTheory.MeasurablySeparable (Set.image …
    F : A → A
    hFn : ∀ (p : A), Eq (↑(F p)).1 (HAdd.hAdd (↑p).1 1)
    hFx : ∀ (p : A), Membership.mem (PiNat.cylinder (↑p).2.1 (↑p).1) (↑(F p)).2.1
    hFy : ∀ (p : A), Membership.mem (PiNat.cylinder (↑p).2.2 (↑p).1) (↑(F p)).2.2
    p0 : A := ⟨{ fst := 0, snd := { fst := fun x => 0, snd := fun x => 0 } }, ⋯⟩
    ⊢ False
  -/
  let p : ℕ → A := fun n => F^[n] p0
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : T2Space α
    inst✝¹ : MeasurableSpace α
    inst✝ : OpensMeasurableSpace α
    f g : (Nat → Nat) → α
    hf : Continuous f
    hg : Continuous g
    h : Disjoint (Set.range f) (Set.range g)
    hfg : Not (MeasureTheory.MeasurablySeparable (Set.range f) (Set.range g))
    I : ∀ (n : Nat) (x y : Nat → Nat), Not (MeasureTheory.MeasurablySeparable (Set …
    A : Type := Subtype fun p => Not (MeasureTheory.MeasurablySeparable (Set.image …
    F : A → A
    hFn : ∀ (p : A), Eq (↑(F p)).1 (HAdd.hAdd (↑p).1 1)
    hFx : ∀ (p : A), Membership.mem (PiNat.cylinder (↑p).2.1 (↑p).1) (↑(F p)).2.1
    hFy : ∀ (p : A), Membership.mem (PiNat.cylinder (↑p).2.2 (↑p).1) (↑(F p)).2.2
    p0 : A := ⟨{ fst := 0, snd := { fst := fun x => 0, snd := fun x => 0 } }, ⋯⟩
    p : Nat → A := fun n => Nat.iterate F n p0
    ⊢ False
  -/
  have prec : ∀ n, p (n + 1) = F (p n) := fun n => by simp only [p, iterate_succ', Function.comp]
  -- check that at the `n`-th step we deal with cylinders of length `n`
  have pn_fst : ∀ n, (p n).1.1 = n := by
    intro n
    induction' n with n IH
    · rfl
    · simp only [prec, hFn, IH]
  -- check that the cylinders we construct are indeed decreasing, by checking that the coordinates
  -- are stationary.
  have Ix : ∀ m n, m + 1 ≤ n → (p n).1.2.1 m = (p (m + 1)).1.2.1 m := by
    intro m
    apply Nat.le_induction
    · rfl
    intro n hmn IH
    have I : (F (p n)).val.snd.fst m = (p n).val.snd.fst m := by
      apply hFx (p n) m
      rw [pn_fst]
      exact hmn
    rw [prec, I, IH]
  have Iy : ∀ m n, m + 1 ≤ n → (p n).1.2.2 m = (p (m + 1)).1.2.2 m := by
    intro m
    apply Nat.le_induction
    · rfl
    intro n hmn IH
    have I : (F (p n)).val.snd.snd m = (p n).val.snd.snd m := by
      apply hFy (p n) m
      rw [pn_fst]
      exact hmn
    rw [prec, I, IH]
  -- denote by `x` and `y` the limit points of these two sequences of cylinders.
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : T2Space α
    inst✝¹ : MeasurableSpace α
    inst✝ : OpensMeasurableSpace α
    f g : (Nat → Nat) → α
    hf : Continuous f
    hg : Continuous g
    h : Disjoint (Set.range f) (Set.range g)
    hfg : Not (MeasureTheory.MeasurablySeparable (Set.range f) (Set.range g))
    I : ∀ (n : Nat) (x y : Nat → Nat), Not (MeasureTheory.MeasurablySeparable (Set …
    A : Type := Subtype fun p => Not (MeasureTheory.MeasurablySeparable (Set.image …
    F : A → A
    hFn : ∀ (p : A), Eq (↑(F p)).1 (HAdd.hAdd (↑p).1 1)
    hFx : ∀ (p : A), Membership.mem (PiNat.cylinder (↑p).2.1 (↑p).1) (↑(F p)).2.1
    hFy : ∀ (p : A), Membership.mem (PiNat.cylinder (↑p).2.2 (↑p).1) (↑(F p)).2.2
    p0 : A := ⟨{ fst := 0, snd := { fst := fun x => 0, snd := fun x => 0 } }, ⋯⟩
    p : Nat → A := fun n => Nat.iterate F n p0
    prec : ∀ (n : Nat), Eq (p (HAdd.hAdd n 1)) (F (p n))
    pn_fst : ∀ (n : Nat), Eq (↑(p n)).1 n
    Ix : ∀ (m n : Nat), LE.le (HAdd.hAdd m 1) n → Eq ((↑(p n)).2.1 m) ((↑(p (HAdd. …
    Iy : ∀ (m n : Nat), LE.le (HAdd.hAdd m 1) n → Eq ((↑(p n)).2.2 m) ((↑(p (HAdd. …
    ⊢ False
  -/
  set x : ℕ → ℕ := fun n => (p (n + 1)).1.2.1 n with hx
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : T2Space α
    inst✝¹ : MeasurableSpace α
    inst✝ : OpensMeasurableSpace α
    f g : (Nat → Nat) → α
    hf : Continuous f
    hg : Continuous g
    h : Disjoint (Set.range f) (Set.range g)
    hfg : Not (MeasureTheory.MeasurablySeparable (Set.range f) (Set.range g))
    I : ∀ (n : Nat) (x y : Nat → Nat), Not (MeasureTheory.MeasurablySeparable (Set …
    A : Type := Subtype fun p => Not (MeasureTheory.MeasurablySeparable (Set.image …
    F : A → A
    hFn : ∀ (p : A), Eq (↑(F p)).1 (HAdd.hAdd (↑p).1 1)
    hFx : ∀ (p : A), Membership.mem (PiNat.cylinder (↑p).2.1 (↑p).1) (↑(F p)).2.1
    hFy : ∀ (p : A), Membership.mem (PiNat.cylinder (↑p).2.2 (↑p).1) (↑(F p)).2.2
    p0 : A := ⟨{ fst := 0, snd := { fst := fun x => 0, snd := fun x => 0 } }, ⋯⟩
    p : Nat → A := fun n => Nat.iterate F n p0
    prec : ∀ (n : Nat), Eq (p (HAdd.hAdd n 1)) (F (p n))
    pn_fst : ∀ (n : Nat), Eq (↑(p n)).1 n
    Ix : ∀ (m n : Nat), LE.le (HAdd.hAdd m 1) n → Eq ((↑(p n)).2.1 m) ((↑(p (HAdd. …
    Iy : ∀ (m n : Nat), LE.le (HAdd.hAdd m 1) n → Eq ((↑(p n)).2.2 m) ((↑(p (HAdd. …
    x : Nat → Nat := fun n => (↑(p (HAdd.hAdd n 1))).2.1 n
    hx : Eq x fun n => (↑(p (HAdd.hAdd n 1))).2.1 n
    ⊢ False
  -/
  set y : ℕ → ℕ := fun n => (p (n + 1)).1.2.2 n with hy
  -- by design, the cylinders around these points have images which are not Borel-separable.
  have M : ∀ n, ¬MeasurablySeparable (f '' cylinder x n) (g '' cylinder y n) := by
    intro n
    convert (p n).2 using 3
    · rw [pn_fst, ← mem_cylinder_iff_eq, mem_cylinder_iff]
      intro i hi
      rw [hx]
      exact (Ix i n hi).symm
    · rw [pn_fst, ← mem_cylinder_iff_eq, mem_cylinder_iff]
      intro i hi
      rw [hy]
      exact (Iy i n hi).symm
  -- consider two open sets separating `f x` and `g y`.
  obtain ⟨u, v, u_open, v_open, xu, yv, huv⟩ :
      ∃ u v : Set α, IsOpen u ∧ IsOpen v ∧ f x ∈ u ∧ g y ∈ v ∧ Disjoint u v := by
    apply t2_separation
    exact disjoint_iff_forall_ne.1 h (mem_range_self _) (mem_range_self _)
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : T2Space α
    inst✝¹ : MeasurableSpace α
    inst✝ : OpensMeasurableSpace α
    f g : (Nat → Nat) → α
    hf : Continuous f
    hg : Continuous g
    h : Disjoint (Set.range f) (Set.range g)
    hfg : Not (MeasureTheory.MeasurablySeparable (Set.range f) (Set.range g))
    I : ∀ (n : Nat) (x y : Nat → Nat), Not (MeasureTheory.MeasurablySeparable (Set …
    A : Type := Subtype fun p => Not (MeasureTheory.MeasurablySeparable (Set.image …
    F : A → A
    hFn : ∀ (p : A), Eq (↑(F p)).1 (HAdd.hAdd (↑p).1 1)
    hFx : ∀ (p : A), Membership.mem (PiNat.cylinder (↑p).2.1 (↑p).1) (↑(F p)).2.1
    hFy : ∀ (p : A), Membership.mem (PiNat.cylinder (↑p).2.2 (↑p).1) (↑(F p)).2.2
    p0 : A := ⟨{ fst := 0, snd := { fst := fun x => 0, snd := fun x => 0 } }, ⋯⟩
    p : Nat → A := fun n => Nat.iterate F n p0
    prec : ∀ (n : Nat), Eq (p (HAdd.hAdd n 1)) (F (p n))
    pn_fst : ∀ (n : Nat), Eq (↑(p n)).1 n
    Ix : ∀ (m n : Nat), LE.le (HAdd.hAdd m 1) n → Eq ((↑(p n)).2.1 m) ((↑(p (HAdd. …
    Iy : ∀ (m n : Nat), LE.le (HAdd.hAdd m 1) n → Eq ((↑(p n)).2.2 m) ((↑(p (HAdd. …
    x : Nat → Nat := fun n => (↑(p (HAdd.hAdd n 1))).2.1 n
    hx : Eq x fun n => (↑(p (HAdd.hAdd n 1))).2.1 n
    y : Nat → Nat := fun n => (↑(p (HAdd.hAdd n 1))).2.2 n
    hy : Eq y fun n => (↑(p (HAdd.hAdd n 1))).2.2 n
    M : ∀ (n : Nat), Not (MeasureTheory.MeasurablySeparable (Set.image f (PiNat.cy …
    u v : Set α
    u_open : IsOpen u
    v_open : IsOpen v
    xu : Membership.mem u (f x)
    yv : Membership.mem v (g y)
    huv : Disjoint u v
    ⊢ False
  -/
  letI : MetricSpace (ℕ → ℕ) := metricSpaceNatNat
  obtain ⟨εx, εxpos, hεx⟩ : ∃ (εx : ℝ), εx > 0 ∧ Metric.ball x εx ⊆ f ⁻¹' u := by
    apply Metric.mem_nhds_iff.1
    exact hf.continuousAt.preimage_mem_nhds (u_open.mem_nhds xu)
  obtain ⟨εy, εypos, hεy⟩ : ∃ (εy : ℝ), εy > 0 ∧ Metric.ball y εy ⊆ g ⁻¹' v := by
    apply Metric.mem_nhds_iff.1
    exact hg.continuousAt.preimage_mem_nhds (v_open.mem_nhds yv)
  obtain ⟨n, hn⟩ : ∃ n : ℕ, (1 / 2 : ℝ) ^ n < min εx εy :=
    exists_pow_lt_of_lt_one (lt_min εxpos εypos) (by norm_num)
  -- for large enough `n`, these open sets separate the images of long cylinders around `x` and `y`
  have B : MeasurablySeparable (f '' cylinder x n) (g '' cylinder y n) := by
    refine ⟨u, ?_, ?_, u_open.measurableSet⟩
    · rw [image_subset_iff]
      apply Subset.trans _ hεx
      intro z hz
      rw [mem_cylinder_iff_dist_le] at hz
      exact hz.trans_lt (hn.trans_le (min_le_left _ _))
    · refine Disjoint.mono_left ?_ huv.symm
      change g '' cylinder y n ⊆ v
      rw [image_subset_iff]
      apply Subset.trans _ hεy
      intro z hz
      rw [mem_cylinder_iff_dist_le] at hz
      exact hz.trans_lt (hn.trans_le (min_le_right _ _))
  -- this is a contradiction.
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : T2Space α
    inst✝¹ : MeasurableSpace α
    inst✝ : OpensMeasurableSpace α
    f g : (Nat → Nat) → α
    hf : Continuous f
    hg : Continuous g
    h : Disjoint (Set.range f) (Set.range g)
    hfg : Not (MeasureTheory.MeasurablySeparable (Set.range f) (Set.range g))
    I : ∀ (n : Nat) (x y : Nat → Nat), Not (MeasureTheory.MeasurablySeparable (Set …
    A : Type := Subtype fun p => Not (MeasureTheory.MeasurablySeparable (Set.image …
    F : A → A
    hFn : ∀ (p : A), Eq (↑(F p)).1 (HAdd.hAdd (↑p).1 1)
    hFx : ∀ (p : A), Membership.mem (PiNat.cylinder (↑p).2.1 (↑p).1) (↑(F p)).2.1
    hFy : ∀ (p : A), Membership.mem (PiNat.cylinder (↑p).2.2 (↑p).1) (↑(F p)).2.2
    p0 : A := ⟨{ fst := 0, snd := { fst := fun x => 0, snd := fun x => 0 } }, ⋯⟩
    p : Nat → A := fun n => Nat.iterate F n p0
    prec : ∀ (n : Nat), Eq (p (HAdd.hAdd n 1)) (F (p n))
    pn_fst : ∀ (n : Nat), Eq (↑(p n)).1 n
    Ix : ∀ (m n : Nat), LE.le (HAdd.hAdd m 1) n → Eq ((↑(p n)).2.1 m) ((↑(p (HAdd. …
    Iy : ∀ (m n : Nat), LE.le (HAdd.hAdd m 1) n → Eq ((↑(p n)).2.2 m) ((↑(p (HAdd. …
    x : Nat → Nat := fun n => (↑(p (HAdd.hAdd n 1))).2.1 n
    hx : Eq x fun n => (↑(p (HAdd.hAdd n 1))).2.1 n
    y : Nat → Nat := fun n => (↑(p (HAdd.hAdd n 1))).2.2 n
    hy : Eq y fun n => (↑(p (HAdd.hAdd n 1))).2.2 n
    M : ∀ (n : Nat), Not (MeasureTheory.MeasurablySeparable (Set.image f (PiNat.cy …
    u v : Set α
    u_open : IsOpen u
    v_open : IsOpen v
    xu : Membership.mem u (f x)
    yv : Membership.mem v (g y)
    huv : Disjoint u v
    this : MetricSpace (Nat → Nat) := PiNat.metricSpaceNatNat
    εx : Real
    εxpos : GT.gt εx 0
    hεx : HasSubset.Subset (Metric.ball x εx) (Set.preimage f u)
    εy : Real
    εypos : GT.gt εy 0
    hεy : HasSubset.Subset (Metric.ball y εy) (Set.preimage g v)
    n : Nat
    hn : LT.lt (HPow.hPow (1 / 2) n) (Min.min εx εy)
    B : MeasureTheory.MeasurablySeparable (Set.image f (PiNat.cylinder x n)) (Set. …
    ⊢ False
  -/
  exact M n B
  /-
    🎉 no goals
  -/


/-- The **Lusin separation theorem**: if two analytic sets are disjoint, then they are contained in
disjoint Borel sets. -/
theorem AnalyticSet.measurablySeparable [T2Space α] [MeasurableSpace α] [OpensMeasurableSpace α]
    {s t : Set α} (hs : AnalyticSet s) (ht : AnalyticSet t) (h : Disjoint s t) :
    MeasurablySeparable s t := by
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : T2Space α
    inst✝¹ : MeasurableSpace α
    inst✝ : OpensMeasurableSpace α
    s t : Set α
    hs : MeasureTheory.AnalyticSet s
    ht : MeasureTheory.AnalyticSet t
    h : Disjoint s t
    ⊢ MeasureTheory.MeasurablySeparable s t
  -/
  rw [AnalyticSet] at hs ht
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : T2Space α
    inst✝¹ : MeasurableSpace α
    inst✝ : OpensMeasurableSpace α
    s t : Set α
    hs : Or (Eq s EmptyCollection.emptyCollection) (Exists fun f => And (Continuou …
    ht : Or (Eq t EmptyCollection.emptyCollection) (Exists fun f => And (Continuou …
    h : Disjoint s t
    ⊢ MeasureTheory.MeasurablySeparable s t
  -/
  rcases hs with (rfl | ⟨f, f_cont, rfl⟩)
    /-
      case inl
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : T2Space α
      inst✝¹ : MeasurableSpace α
      inst✝ : OpensMeasurableSpace α
      t : Set α
      ht : Or (Eq t EmptyCollection.emptyCollection) (Exists fun f => And (Continuou …
      h : Disjoint EmptyCollection.emptyCollection t
      ⊢ MeasureTheory.MeasurablySeparable EmptyCollection.emptyCollection t
    -/
  · refine ⟨∅, Subset.refl _, by simp, MeasurableSet.empty⟩
    /-
      🎉 no goals
    -/
  /-
    case inr.intro.intro
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : T2Space α
    inst✝¹ : MeasurableSpace α
    inst✝ : OpensMeasurableSpace α
    t : Set α
    ht : Or (Eq t EmptyCollection.emptyCollection) (Exists fun f => And (Continuou …
    f : (Nat → Nat) → α
    f_cont : Continuous f
    h : Disjoint (Set.range f) t
    ⊢ MeasureTheory.MeasurablySeparable (Set.range f) t
  -/
  rcases ht with (rfl | ⟨g, g_cont, rfl⟩)
    /-
      case inr.intro.intro.inl
      α : Type u_1
      inst✝³ : TopologicalSpace α
      inst✝² : T2Space α
      inst✝¹ : MeasurableSpace α
      inst✝ : OpensMeasurableSpace α
      f : (Nat → Nat) → α
      f_cont : Continuous f
      h : Disjoint (Set.range f) EmptyCollection.emptyCollection
      ⊢ MeasureTheory.MeasurablySeparable (Set.range f) EmptyCollection.emptyCollect …
    -/
  · exact ⟨univ, subset_univ _, by simp, MeasurableSet.univ⟩
    /-
      🎉 no goals
    -/
  /-
    case inr.intro.intro.inr.intro.intro
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : T2Space α
    inst✝¹ : MeasurableSpace α
    inst✝ : OpensMeasurableSpace α
    f : (Nat → Nat) → α
    f_cont : Continuous f
    g : (Nat → Nat) → α
    g_cont : Continuous g
    h : Disjoint (Set.range f) (Set.range g)
    ⊢ MeasureTheory.MeasurablySeparable (Set.range f) (Set.range g)
  -/
  exact measurablySeparable_range_of_disjoint f_cont g_cont h
  /-
    🎉 no goals
  -/


/-- **Suslin's Theorem**: in a Hausdorff topological space, an analytic set with an analytic
complement is measurable. -/
theorem AnalyticSet.measurableSet_of_compl [T2Space α] [MeasurableSpace α] [OpensMeasurableSpace α]
    {s : Set α} (hs : AnalyticSet s) (hsc : AnalyticSet sᶜ) : MeasurableSet s := by
  /-
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : T2Space α
    inst✝¹ : MeasurableSpace α
    inst✝ : OpensMeasurableSpace α
    s : Set α
    hs : MeasureTheory.AnalyticSet s
    hsc : MeasureTheory.AnalyticSet (HasCompl.compl s)
    ⊢ MeasurableSet s
  -/
  rcases hs.measurablySeparable hsc disjoint_compl_right with ⟨u, hsu, hdu, hmu⟩
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : T2Space α
    inst✝¹ : MeasurableSpace α
    inst✝ : OpensMeasurableSpace α
    s : Set α
    hs : MeasureTheory.AnalyticSet s
    hsc : MeasureTheory.AnalyticSet (HasCompl.compl s)
    u : Set α
    hsu : HasSubset.Subset s u
    hdu : Disjoint (HasCompl.compl s) u
    hmu : MeasurableSet u
    ⊢ MeasurableSet s
  -/
  obtain rfl : s = u := hsu.antisymm (disjoint_compl_left_iff_subset.1 hdu)
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝³ : TopologicalSpace α
    inst✝² : T2Space α
    inst✝¹ : MeasurableSpace α
    inst✝ : OpensMeasurableSpace α
    s : Set α
    hs : MeasureTheory.AnalyticSet s
    hsc : MeasureTheory.AnalyticSet (HasCompl.compl s)
    hsu : HasSubset.Subset s s
    hdu : Disjoint (HasCompl.compl s) s
    hmu : MeasurableSet s
    ⊢ MeasurableSet s
  -/
  exact hmu
  /-
    🎉 no goals
  -/


/-- If `f : X → Z` is a surjective Borel measurable map from a standard Borel space
to a countably separated measurable space, then the preimage of a set `s`
is measurable if and only if the set is measurable.
One implication is the definition of measurability, the other one heavily relies on `X` being a
standard Borel space. -/
theorem measurableSet_preimage_iff_of_surjective [CountablySeparated Z]
    {f : X → Z} (hf : Measurable f) (hsurj : Surjective f) {s : Set Z} :
    MeasurableSet (f ⁻¹' s) ↔ MeasurableSet s := by
  /-
    X : Type u_3
    Z : Type u_5
    inst✝³ : MeasurableSpace X
    inst✝² : StandardBorelSpace X
    inst✝¹ : MeasurableSpace Z
    inst✝ : MeasurableSpace.CountablySeparated Z
    f : X → Z
    hf : Measurable f
    hsurj : Function.Surjective f
    s : Set Z
    ⊢ Iff (MeasurableSet (Set.preimage f s)) (MeasurableSet s)
  -/
  refine ⟨fun h => ?_, fun h => hf h⟩
  /-
    X : Type u_3
    Z : Type u_5
    inst✝³ : MeasurableSpace X
    inst✝² : StandardBorelSpace X
    inst✝¹ : MeasurableSpace Z
    inst✝ : MeasurableSpace.CountablySeparated Z
    f : X → Z
    hf : Measurable f
    hsurj : Function.Surjective f
    s : Set Z
    h : MeasurableSet (Set.preimage f s)
    ⊢ MeasurableSet s
  -/
  rcases exists_opensMeasurableSpace_of_countablySeparated Z with ⟨τ, _, _, _⟩
  /-
    case intro.intro.intro
    X : Type u_3
    Z : Type u_5
    inst✝³ : MeasurableSpace X
    inst✝² : StandardBorelSpace X
    inst✝¹ : MeasurableSpace Z
    inst✝ : MeasurableSpace.CountablySeparated Z
    f : X → Z
    hf : Measurable f
    hsurj : Function.Surjective f
    s : Set Z
    h : MeasurableSet (Set.preimage f s)
    τ : TopologicalSpace Z
    left✝¹ : SecondCountableTopology Z
    left✝ : T4Space Z
    right✝ : OpensMeasurableSpace Z
    ⊢ MeasurableSet s
  -/
  apply AnalyticSet.measurableSet_of_compl
    /-
      case intro.intro.intro.hs
      X : Type u_3
      Z : Type u_5
      inst✝³ : MeasurableSpace X
      inst✝² : StandardBorelSpace X
      inst✝¹ : MeasurableSpace Z
      inst✝ : MeasurableSpace.CountablySeparated Z
      f : X → Z
      hf : Measurable f
      hsurj : Function.Surjective f
      s : Set Z
      h : MeasurableSet (Set.preimage f s)
      τ : TopologicalSpace Z
      left✝¹ : SecondCountableTopology Z
      left✝ : T4Space Z
      right✝ : OpensMeasurableSpace Z
      ⊢ MeasureTheory.AnalyticSet s
    -/
  · rw [← image_preimage_eq s hsurj]
    /-
      case intro.intro.intro.hs
      X : Type u_3
      Z : Type u_5
      inst✝³ : MeasurableSpace X
      inst✝² : StandardBorelSpace X
      inst✝¹ : MeasurableSpace Z
      inst✝ : MeasurableSpace.CountablySeparated Z
      f : X → Z
      hf : Measurable f
      hsurj : Function.Surjective f
      s : Set Z
      h : MeasurableSet (Set.preimage f s)
      τ : TopologicalSpace Z
      left✝¹ : SecondCountableTopology Z
      left✝ : T4Space Z
      right✝ : OpensMeasurableSpace Z
      ⊢ MeasureTheory.AnalyticSet (Set.image f (Set.preimage f s))
    -/
    exact h.analyticSet_image hf
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.hsc
      X : Type u_3
      Z : Type u_5
      inst✝³ : MeasurableSpace X
      inst✝² : StandardBorelSpace X
      inst✝¹ : MeasurableSpace Z
      inst✝ : MeasurableSpace.CountablySeparated Z
      f : X → Z
      hf : Measurable f
      hsurj : Function.Surjective f
      s : Set Z
      h : MeasurableSet (Set.preimage f s)
      τ : TopologicalSpace Z
      left✝¹ : SecondCountableTopology Z
      left✝ : T4Space Z
      right✝ : OpensMeasurableSpace Z
      ⊢ MeasureTheory.AnalyticSet (HasCompl.compl s)
    -/
  · rw [← image_preimage_eq sᶜ hsurj]
    /-
      case intro.intro.intro.hsc
      X : Type u_3
      Z : Type u_5
      inst✝³ : MeasurableSpace X
      inst✝² : StandardBorelSpace X
      inst✝¹ : MeasurableSpace Z
      inst✝ : MeasurableSpace.CountablySeparated Z
      f : X → Z
      hf : Measurable f
      hsurj : Function.Surjective f
      s : Set Z
      h : MeasurableSet (Set.preimage f s)
      τ : TopologicalSpace Z
      left✝¹ : SecondCountableTopology Z
      left✝ : T4Space Z
      right✝ : OpensMeasurableSpace Z
      ⊢ MeasureTheory.AnalyticSet (Set.image f (Set.preimage f (HasCompl.compl s)))
    -/
    exact h.compl.analyticSet_image hf
    /-
      🎉 no goals
    -/


theorem map_measurableSpace_eq [CountablySeparated Z]
    {f : X → Z} (hf : Measurable f)
    (hsurj : Surjective f) : MeasurableSpace.map f ‹MeasurableSpace X› = ‹MeasurableSpace Z› :=
  MeasurableSpace.ext fun _ => hf.measurableSet_preimage_iff_of_surjective hsurj


theorem map_measurableSpace_eq_borel [SecondCountableTopology Y] {f : X → Y} (hf : Measurable f)
    (hsurj : Surjective f) : MeasurableSpace.map f ‹MeasurableSpace X› = borel Y := by
  /-
    X : Type u_3
    Y : Type u_4
    inst✝⁶ : MeasurableSpace X
    inst✝⁵ : StandardBorelSpace X
    inst✝⁴ : TopologicalSpace Y
    inst✝³ : T0Space Y
    inst✝² : MeasurableSpace Y
    inst✝¹ : OpensMeasurableSpace Y
    inst✝ : SecondCountableTopology Y
    f : X → Y
    hf : Measurable f
    hsurj : Function.Surjective f
    ⊢ Eq (MeasurableSpace.map f inst✝⁶) (borel Y)
  -/
  have d := hf.mono le_rfl OpensMeasurableSpace.borel_le
  /-
    X : Type u_3
    Y : Type u_4
    inst✝⁶ : MeasurableSpace X
    inst✝⁵ : StandardBorelSpace X
    inst✝⁴ : TopologicalSpace Y
    inst✝³ : T0Space Y
    inst✝² : MeasurableSpace Y
    inst✝¹ : OpensMeasurableSpace Y
    inst✝ : SecondCountableTopology Y
    f : X → Y
    hf : Measurable f
    hsurj : Function.Surjective f
    d : Measurable f
    ⊢ Eq (MeasurableSpace.map f inst✝⁶) (borel Y)
  -/
  letI := borel Y; haveI : BorelSpace Y := ⟨rfl⟩
  /-
    X : Type u_3
    Y : Type u_4
    inst✝⁶ : MeasurableSpace X
    inst✝⁵ : StandardBorelSpace X
    inst✝⁴ : TopologicalSpace Y
    inst✝³ : T0Space Y
    inst✝² : MeasurableSpace Y
    inst✝¹ : OpensMeasurableSpace Y
    inst✝ : SecondCountableTopology Y
    f : X → Y
    hf : Measurable f
    hsurj : Function.Surjective f
    d : Measurable f
    this✝ : MeasurableSpace Y := borel Y
    this : BorelSpace Y
    ⊢ Eq (MeasurableSpace.map f inst✝⁶) (borel Y)
  -/
  exact d.map_measurableSpace_eq hsurj
  /-
    🎉 no goals
  -/


theorem borelSpace_codomain [SecondCountableTopology Y] {f : X → Y} (hf : Measurable f)
    (hsurj : Surjective f) : BorelSpace Y :=
  ⟨(hf.map_measurableSpace_eq hsurj).symm.trans <| hf.map_measurableSpace_eq_borel hsurj⟩


/-- If `f : X → Z` is a Borel measurable map from a standard Borel space to a
countably separated measurable space then the preimage of a set `s` is measurable
if and only if the set is measurable in `Set.range f`. -/
theorem measurableSet_preimage_iff_preimage_val {f : X → Z} [CountablySeparated (range f)]
    (hf : Measurable f) {s : Set Z} :
    MeasurableSet (f ⁻¹' s) ↔ MeasurableSet ((↑) ⁻¹' s : Set (range f)) :=
  have hf' : Measurable (rangeFactorization f) := hf.subtype_mk
  hf'.measurableSet_preimage_iff_of_surjective (s := Subtype.val ⁻¹' s) surjective_onto_range


/-- If `f : X → Z` is a Borel measurable map from a standard Borel space to a
countably separated measurable space and the range of `f` is measurable,
then the preimage of a set `s` is measurable
if and only if the intersection with `Set.range f` is measurable. -/
theorem measurableSet_preimage_iff_inter_range {f : X → Z} [CountablySeparated (range f)]
    (hf : Measurable f) (hr : MeasurableSet (range f)) {s : Set Z} :
    MeasurableSet (f ⁻¹' s) ↔ MeasurableSet (s ∩ range f) := by
  rw [hf.measurableSet_preimage_iff_preimage_val, inter_comm,
    ← (MeasurableEmbedding.subtype_coe hr).measurableSet_image, Subtype.image_preimage_coe]


/-- If `f : X → Z` is a Borel measurable map from a standard Borel space
to a countably separated measurable space,
then for any measurable space `β` and `g : Z → β`, the composition `g ∘ f` is
measurable if and only if the restriction of `g` to the range of `f` is measurable. -/
theorem measurable_comp_iff_restrict {f : X → Z}
    [CountablySeparated (range f)]
    (hf : Measurable f) {g : Z → β} : Measurable (g ∘ f) ↔ Measurable (restrict (range f) g) :=
  forall₂_congr fun s _ => measurableSet_preimage_iff_preimage_val hf (s := g ⁻¹' s)


/-- If `f : X → Z` is a surjective Borel measurable map from a standard Borel space
to a countably separated measurable space,
then for any measurable space `α` and `g : Z → α`, the composition
`g ∘ f` is measurable if and only if `g` is measurable. -/
theorem measurable_comp_iff_of_surjective [CountablySeparated Z]
    {f : X → Z} (hf : Measurable f) (hsurj : Surjective f)
    {g : Z → β} : Measurable (g ∘ f) ↔ Measurable g :=
  forall₂_congr fun s _ => measurableSet_preimage_iff_of_surjective hf hsurj (s := g ⁻¹' s)


theorem Continuous.map_eq_borel {X Y : Type*} [TopologicalSpace X] [PolishSpace X]
    [MeasurableSpace X] [BorelSpace X] [TopologicalSpace Y] [T0Space Y] [SecondCountableTopology Y]
    {f : X → Y} (hf : Continuous f) (hsurj : Surjective f) :
    MeasurableSpace.map f ‹MeasurableSpace X› = borel Y := by
  /-
    X : Type u_3
    Y : Type u_4
    inst✝⁶ : TopologicalSpace X
    inst✝⁵ : PolishSpace X
    inst✝⁴ : MeasurableSpace X
    inst✝³ : BorelSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : T0Space Y
    inst✝ : SecondCountableTopology Y
    f : X → Y
    hf : Continuous f
    hsurj : Function.Surjective f
    ⊢ Eq (MeasurableSpace.map f inst✝⁴) (borel Y)
  -/
  borelize Y
  /-
    X : Type u_3
    Y : Type u_4
    inst✝⁶ : TopologicalSpace X
    inst✝⁵ : PolishSpace X
    inst✝⁴ : MeasurableSpace X
    inst✝³ : BorelSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : T0Space Y
    inst✝ : SecondCountableTopology Y
    f : X → Y
    hf : Continuous f
    hsurj : Function.Surjective f
    this✝¹ : MeasurableSpace Y := borel Y
    this✝ : BorelSpace Y
    ⊢ Eq (MeasurableSpace.map f inst✝⁴) (borel Y)
  -/
  exact hf.measurable.map_measurableSpace_eq hsurj
  /-
    🎉 no goals
  -/


theorem Continuous.map_borel_eq {X Y : Type*} [TopologicalSpace X] [PolishSpace X]
    [TopologicalSpace Y] [T0Space Y] [SecondCountableTopology Y] {f : X → Y} (hf : Continuous f)
    (hsurj : Surjective f) : MeasurableSpace.map f (borel X) = borel Y := by
  /-
    X : Type u_3
    Y : Type u_4
    inst✝⁴ : TopologicalSpace X
    inst✝³ : PolishSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : T0Space Y
    inst✝ : SecondCountableTopology Y
    f : X → Y
    hf : Continuous f
    hsurj : Function.Surjective f
    ⊢ Eq (MeasurableSpace.map f (borel X)) (borel Y)
  -/
  borelize X
  /-
    X : Type u_3
    Y : Type u_4
    inst✝⁴ : TopologicalSpace X
    inst✝³ : PolishSpace X
    inst✝² : TopologicalSpace Y
    inst✝¹ : T0Space Y
    inst✝ : SecondCountableTopology Y
    f : X → Y
    hf : Continuous f
    hsurj : Function.Surjective f
    this✝¹ : MeasurableSpace X := borel X
    this✝ : BorelSpace X
    ⊢ Eq (MeasurableSpace.map f (borel X)) (borel Y)
  -/
  exact hf.map_eq_borel hsurj
  /-
    🎉 no goals
  -/


instance Quotient.borelSpace {X : Type*} [TopologicalSpace X] [PolishSpace X] [MeasurableSpace X]
    [BorelSpace X] {s : Setoid X} [T0Space (Quotient s)] [SecondCountableTopology (Quotient s)] :
    BorelSpace (Quotient s) :=
  ⟨continuous_quotient_mk'.map_eq_borel Quotient.mk'_surjective⟩


/-- When the subgroup `N < G` is not necessarily `Normal`, we have a `CosetSpace` as opposed
to `QuotientGroup` (the next `instance`).
TODO: typeclass inference should normally find this, but currently doesn't.
E.g., `MeasurableSMul G (G ⧸ Γ)` fails to synthesize, even though `G ⧸ Γ` is the quotient
of `G` by the action of `Γ`; it seems unable to pick up the `BorelSpace` instance. -/
@[to_additive AddCosetSpace.borelSpace]
instance CosetSpace.borelSpace {G : Type*} [TopologicalSpace G] [PolishSpace G] [Group G]
    [MeasurableSpace G] [BorelSpace G] {N : Subgroup G} [T2Space (G ⧸ N)]
    [SecondCountableTopology (G ⧸ N)] : BorelSpace (G ⧸ N) := Quotient.borelSpace


@[to_additive]
instance QuotientGroup.borelSpace {G : Type*} [TopologicalSpace G] [PolishSpace G] [Group G]
    [TopologicalGroup G] [MeasurableSpace G] [BorelSpace G] {N : Subgroup G} [N.Normal]
    [IsClosed (N : Set G)] : BorelSpace (G ⧸ N) :=
  ⟨continuous_mk.map_eq_borel mk_surjective⟩


/-- The **Lusin-Souslin theorem**: the range of a continuous injective function defined on a Polish
space is Borel-measurable. -/
theorem measurableSet_range_of_continuous_injective {β : Type*} [TopologicalSpace γ]
    [PolishSpace γ] [TopologicalSpace β] [T2Space β] [MeasurableSpace β] [OpensMeasurableSpace β]
    {f : γ → β} (f_cont : Continuous f) (f_inj : Injective f) :
    MeasurableSet (range f) := by
  /- We follow [Fremlin, *Measure Theory* (volume 4, 423I)][fremlin_vol4].
    Let `b = {s i}` be a countable basis for `α`. When `s i` and `s j` are disjoint, their images
    are disjoint analytic sets, hence by the separation theorem one can find a Borel-measurable set
    `q i j` separating them.
    Let `E i = closure (f '' s i) ∩ ⋂ j, q i j \ q j i`. It contains `f '' (s i)` and it is
    measurable. Let `F n = ⋃ E i`, where the union is taken over those `i` for which `diam (s i)`
    is bounded by some number `u n` tending to `0` with `n`.
    We claim that `range f = ⋂ F n`, from which the measurability is obvious. The inclusion `⊆` is
    straightforward. To show `⊇`, consider a point `x` in the intersection. For each `n`, it belongs
    to some `E i` with `diam (s i) ≤ u n`. Pick a point `y i ∈ s i`. We claim that for such `i`
    and `j`, the intersection `s i ∩ s j` is nonempty: if it were empty, then thanks to the
    separating set `q i j` in the definition of `E i` one could not have `x ∈ E i ∩ E j`.
    Since these two sets have small diameter, it follows that `y i` and `y j` are close.
    Thus, `y` is a Cauchy sequence, converging to a limit `z`. We claim that `f z = x`, completing
    the proof.
    Otherwise, one could find open sets `v` and `w` separating `f z` from `x`. Then, for large `n`,
    the image `f '' (s i)` would be included in `v` by continuity of `f`, so its closure would be
    contained in the closure of `v`, and therefore it would be disjoint from `w`. This is a
    contradiction since `x` belongs both to this closure and to `w`. -/
  /-
    γ : Type u_3
    β : Type u_4
    inst✝⁵ : TopologicalSpace γ
    inst✝⁴ : PolishSpace γ
    inst✝³ : TopologicalSpace β
    inst✝² : T2Space β
    inst✝¹ : MeasurableSpace β
    inst✝ : OpensMeasurableSpace β
    f : γ → β
    f_cont : Continuous f
    f_inj : Function.Injective f
    ⊢ MeasurableSet (Set.range f)
  -/
  letI := upgradePolishSpace γ
  obtain ⟨b, b_count, b_nonempty, hb⟩ :
    ∃ b : Set (Set γ), b.Countable ∧ ∅ ∉ b ∧ IsTopologicalBasis b := exists_countable_basis γ
  /-
    case intro.intro.intro
    γ : Type u_3
    β : Type u_4
    inst✝⁵ : TopologicalSpace γ
    inst✝⁴ : PolishSpace γ
    inst✝³ : TopologicalSpace β
    inst✝² : T2Space β
    inst✝¹ : MeasurableSpace β
    inst✝ : OpensMeasurableSpace β
    f : γ → β
    f_cont : Continuous f
    f_inj : Function.Injective f
    this : UpgradedPolishSpace γ := upgradePolishSpace γ
    b : Set (Set γ)
    b_count : b.Countable
    b_nonempty : Not (Membership.mem b EmptyCollection.emptyCollection)
    hb : TopologicalSpace.IsTopologicalBasis b
    ⊢ MeasurableSet (Set.range f)
  -/
  haveI : Encodable b := b_count.toEncodable
  /-
    case intro.intro.intro
    γ : Type u_3
    β : Type u_4
    inst✝⁵ : TopologicalSpace γ
    inst✝⁴ : PolishSpace γ
    inst✝³ : TopologicalSpace β
    inst✝² : T2Space β
    inst✝¹ : MeasurableSpace β
    inst✝ : OpensMeasurableSpace β
    f : γ → β
    f_cont : Continuous f
    f_inj : Function.Injective f
    this✝ : UpgradedPolishSpace γ := upgradePolishSpace γ
    b : Set (Set γ)
    b_count : b.Countable
    b_nonempty : Not (Membership.mem b EmptyCollection.emptyCollection)
    hb : TopologicalSpace.IsTopologicalBasis b
    this : Encodable ↑b
    ⊢ MeasurableSet (Set.range f)
  -/
  let A := { p : b × b // Disjoint (p.1 : Set γ) p.2 }
  -- for each pair of disjoint sets in the topological basis `b`, consider Borel sets separating
  -- their images, by injectivity of `f` and the Lusin separation theorem.
  have : ∀ p : A, ∃ q : Set β,
      f '' (p.1.1 : Set γ) ⊆ q ∧ Disjoint (f '' (p.1.2 : Set γ)) q ∧ MeasurableSet q := by
    intro p
    apply
      AnalyticSet.measurablySeparable ((hb.isOpen p.1.1.2).analyticSet_image f_cont)
        ((hb.isOpen p.1.2.2).analyticSet_image f_cont)
    exact Disjoint.image p.2 f_inj.injOn (subset_univ _) (subset_univ _)
  /-
    case intro.intro.intro
    γ : Type u_3
    β : Type u_4
    inst✝⁵ : TopologicalSpace γ
    inst✝⁴ : PolishSpace γ
    inst✝³ : TopologicalSpace β
    inst✝² : T2Space β
    inst✝¹ : MeasurableSpace β
    inst✝ : OpensMeasurableSpace β
    f : γ → β
    f_cont : Continuous f
    f_inj : Function.Injective f
    this✝¹ : UpgradedPolishSpace γ := upgradePolishSpace γ
    b : Set (Set γ)
    b_count : b.Countable
    b_nonempty : Not (Membership.mem b EmptyCollection.emptyCollection)
    hb : TopologicalSpace.IsTopologicalBasis b
    this✝ : Encodable ↑b
    A : Type (max 0 u_3) := Subtype fun p => Disjoint ↑p.1 ↑p.2
    this : ∀ (p : A), Exists fun q => And (HasSubset.Subset (Set.image f ↑(↑p).1)  …
    ⊢ MeasurableSet (Set.range f)
  -/
  choose q hq1 hq2 q_meas using this
  -- define sets `E i` and `F n` as in the proof sketch above
  let E : b → Set β := fun s =>
    closure (f '' s) ∩ ⋂ (t : b) (ht : Disjoint s.1 t.1), q ⟨(s, t), ht⟩ \ q ⟨(t, s), ht.symm⟩
  obtain ⟨u, u_anti, u_pos, u_lim⟩ :
      ∃ u : ℕ → ℝ, StrictAnti u ∧ (∀ n : ℕ, 0 < u n) ∧ Tendsto u atTop (𝓝 0) :=
    exists_seq_strictAnti_tendsto (0 : ℝ)
  /-
    case intro.intro.intro.intro.intro.intro
    γ : Type u_3
    β : Type u_4
    inst✝⁵ : TopologicalSpace γ
    inst✝⁴ : PolishSpace γ
    inst✝³ : TopologicalSpace β
    inst✝² : T2Space β
    inst✝¹ : MeasurableSpace β
    inst✝ : OpensMeasurableSpace β
    f : γ → β
    f_cont : Continuous f
    f_inj : Function.Injective f
    this✝ : UpgradedPolishSpace γ := upgradePolishSpace γ
    b : Set (Set γ)
    b_count : b.Countable
    b_nonempty : Not (Membership.mem b EmptyCollection.emptyCollection)
    hb : TopologicalSpace.IsTopologicalBasis b
    this : Encodable ↑b
    A : Type (max 0 u_3) := Subtype fun p => Disjoint ↑p.1 ↑p.2
    q : A → Set β
    hq1 : ∀ (p : A), HasSubset.Subset (Set.image f ↑(↑p).1) (q p)
    hq2 : ∀ (p : A), Disjoint (Set.image f ↑(↑p).2) (q p)
    q_meas : ∀ (p : A), MeasurableSet (q p)
    E : ↑b → Set β := fun s => Inter.inter (closure (Set.image f ↑s)) (Set.iInter  …
    u : Nat → Real
    u_anti : StrictAnti u
    u_pos : ∀ (n : Nat), LT.lt 0 (u n)
    u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
    ⊢ MeasurableSet (Set.range f)
  -/
  let F : ℕ → Set β := fun n => ⋃ (s : b) (_ : IsBounded s.1 ∧ diam s.1 ≤ u n), E s
  -- it is enough to show that `range f = ⋂ F n`, as the latter set is obviously measurable.
  suffices range f = ⋂ n, F n by
    have E_meas : ∀ s : b, MeasurableSet (E s) := by
      intro b
      refine isClosed_closure.measurableSet.inter ?_
      refine MeasurableSet.iInter fun s => ?_
      exact MeasurableSet.iInter fun hs => (q_meas _).diff (q_meas _)
    have F_meas : ∀ n, MeasurableSet (F n) := by
      intro n
      refine MeasurableSet.iUnion fun s => ?_
      exact MeasurableSet.iUnion fun _ => E_meas _
    rw [this]
    exact MeasurableSet.iInter fun n => F_meas n
  -- we check both inclusions.
  /-
    case intro.intro.intro.intro.intro.intro
    γ : Type u_3
    β : Type u_4
    inst✝⁵ : TopologicalSpace γ
    inst✝⁴ : PolishSpace γ
    inst✝³ : TopologicalSpace β
    inst✝² : T2Space β
    inst✝¹ : MeasurableSpace β
    inst✝ : OpensMeasurableSpace β
    f : γ → β
    f_cont : Continuous f
    f_inj : Function.Injective f
    this✝ : UpgradedPolishSpace γ := upgradePolishSpace γ
    b : Set (Set γ)
    b_count : b.Countable
    b_nonempty : Not (Membership.mem b EmptyCollection.emptyCollection)
    hb : TopologicalSpace.IsTopologicalBasis b
    this : Encodable ↑b
    A : Type (max 0 u_3) := Subtype fun p => Disjoint ↑p.1 ↑p.2
    q : A → Set β
    hq1 : ∀ (p : A), HasSubset.Subset (Set.image f ↑(↑p).1) (q p)
    hq2 : ∀ (p : A), Disjoint (Set.image f ↑(↑p).2) (q p)
    q_meas : ∀ (p : A), MeasurableSet (q p)
    E : ↑b → Set β := fun s => Inter.inter (closure (Set.image f ↑s)) (Set.iInter  …
    u : Nat → Real
    u_anti : StrictAnti u
    u_pos : ∀ (n : Nat), LT.lt 0 (u n)
    u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
    F : Nat → Set β := fun n => Set.iUnion fun s => Set.iUnion fun x => E s
    ⊢ Eq (Set.range f) (Set.iInter fun n => F n)
  -/
  apply Subset.antisymm
  -- we start with the easy inclusion `range f ⊆ ⋂ F n`. One just needs to unfold the definitions.
    /-
      case intro.intro.intro.intro.intro.intro.h₁
      γ : Type u_3
      β : Type u_4
      inst✝⁵ : TopologicalSpace γ
      inst✝⁴ : PolishSpace γ
      inst✝³ : TopologicalSpace β
      inst✝² : T2Space β
      inst✝¹ : MeasurableSpace β
      inst✝ : OpensMeasurableSpace β
      f : γ → β
      f_cont : Continuous f
      f_inj : Function.Injective f
      this✝ : UpgradedPolishSpace γ := upgradePolishSpace γ
      b : Set (Set γ)
      b_count : b.Countable
      b_nonempty : Not (Membership.mem b EmptyCollection.emptyCollection)
      hb : TopologicalSpace.IsTopologicalBasis b
      this : Encodable ↑b
      A : Type (max 0 u_3) := Subtype fun p => Disjoint ↑p.1 ↑p.2
      q : A → Set β
      hq1 : ∀ (p : A), HasSubset.Subset (Set.image f ↑(↑p).1) (q p)
      hq2 : ∀ (p : A), Disjoint (Set.image f ↑(↑p).2) (q p)
      q_meas : ∀ (p : A), MeasurableSet (q p)
      E : ↑b → Set β := fun s => Inter.inter (closure (Set.image f ↑s)) (Set.iInter  …
      u : Nat → Real
      u_anti : StrictAnti u
      u_pos : ∀ (n : Nat), LT.lt 0 (u n)
      u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
      F : Nat → Set β := fun n => Set.iUnion fun s => Set.iUnion fun x => E s
      ⊢ HasSubset.Subset (Set.range f) (Set.iInter fun n => F n)
    -/
  · rintro x ⟨y, rfl⟩
    /-
      case intro.intro.intro.intro.intro.intro.h₁.intro
      γ : Type u_3
      β : Type u_4
      inst✝⁵ : TopologicalSpace γ
      inst✝⁴ : PolishSpace γ
      inst✝³ : TopologicalSpace β
      inst✝² : T2Space β
      inst✝¹ : MeasurableSpace β
      inst✝ : OpensMeasurableSpace β
      f : γ → β
      f_cont : Continuous f
      f_inj : Function.Injective f
      this✝ : UpgradedPolishSpace γ := upgradePolishSpace γ
      b : Set (Set γ)
      b_count : b.Countable
      b_nonempty : Not (Membership.mem b EmptyCollection.emptyCollection)
      hb : TopologicalSpace.IsTopologicalBasis b
      this : Encodable ↑b
      A : Type (max 0 u_3) := Subtype fun p => Disjoint ↑p.1 ↑p.2
      q : A → Set β
      hq1 : ∀ (p : A), HasSubset.Subset (Set.image f ↑(↑p).1) (q p)
      hq2 : ∀ (p : A), Disjoint (Set.image f ↑(↑p).2) (q p)
      q_meas : ∀ (p : A), MeasurableSet (q p)
      E : ↑b → Set β := fun s => Inter.inter (closure (Set.image f ↑s)) (Set.iInter  …
      u : Nat → Real
      u_anti : StrictAnti u
      u_pos : ∀ (n : Nat), LT.lt 0 (u n)
      u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
      F : Nat → Set β := fun n => Set.iUnion fun s => Set.iUnion fun x => E s
      y : γ
      ⊢ Membership.mem (Set.iInter fun n => F n) (f y)
    -/
    refine mem_iInter.2 fun n => ?_
    obtain ⟨s, sb, ys, hs⟩ : ∃ (s : Set γ), s ∈ b ∧ y ∈ s ∧ s ⊆ ball y (u n / 2) := by
      apply hb.mem_nhds_iff.1
      exact ball_mem_nhds _ (half_pos (u_pos n))
    have diam_s : diam s ≤ u n := by
      apply (diam_mono hs isBounded_ball).trans
      convert diam_ball (x := y) (half_pos (u_pos n)).le
      ring
    /-
      case intro.intro.intro.intro.intro.intro.h₁.intro.intro.intro.intro
      γ : Type u_3
      β : Type u_4
      inst✝⁵ : TopologicalSpace γ
      inst✝⁴ : PolishSpace γ
      inst✝³ : TopologicalSpace β
      inst✝² : T2Space β
      inst✝¹ : MeasurableSpace β
      inst✝ : OpensMeasurableSpace β
      f : γ → β
      f_cont : Continuous f
      f_inj : Function.Injective f
      this✝ : UpgradedPolishSpace γ := upgradePolishSpace γ
      b : Set (Set γ)
      b_count : b.Countable
      b_nonempty : Not (Membership.mem b EmptyCollection.emptyCollection)
      hb : TopologicalSpace.IsTopologicalBasis b
      this : Encodable ↑b
      A : Type (max 0 u_3) := Subtype fun p => Disjoint ↑p.1 ↑p.2
      q : A → Set β
      hq1 : ∀ (p : A), HasSubset.Subset (Set.image f ↑(↑p).1) (q p)
      hq2 : ∀ (p : A), Disjoint (Set.image f ↑(↑p).2) (q p)
      q_meas : ∀ (p : A), MeasurableSet (q p)
      E : ↑b → Set β := fun s => Inter.inter (closure (Set.image f ↑s)) (Set.iInter  …
      u : Nat → Real
      u_anti : StrictAnti u
      u_pos : ∀ (n : Nat), LT.lt 0 (u n)
      u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
      F : Nat → Set β := fun n => Set.iUnion fun s => Set.iUnion fun x => E s
      y : γ
      n : Nat
      s : Set γ
      sb : Membership.mem b s
      ys : Membership.mem s y
      hs : HasSubset.Subset s (Metric.ball y (HDiv.hDiv (u n) 2))
      diam_s : LE.le (Metric.diam s) (u n)
      ⊢ Membership.mem (F n) (f y)
    -/
    refine mem_iUnion.2 ⟨⟨s, sb⟩, ?_⟩
    /-
      case intro.intro.intro.intro.intro.intro.h₁.intro.intro.intro.intro
      γ : Type u_3
      β : Type u_4
      inst✝⁵ : TopologicalSpace γ
      inst✝⁴ : PolishSpace γ
      inst✝³ : TopologicalSpace β
      inst✝² : T2Space β
      inst✝¹ : MeasurableSpace β
      inst✝ : OpensMeasurableSpace β
      f : γ → β
      f_cont : Continuous f
      f_inj : Function.Injective f
      this✝ : UpgradedPolishSpace γ := upgradePolishSpace γ
      b : Set (Set γ)
      b_count : b.Countable
      b_nonempty : Not (Membership.mem b EmptyCollection.emptyCollection)
      hb : TopologicalSpace.IsTopologicalBasis b
      this : Encodable ↑b
      A : Type (max 0 u_3) := Subtype fun p => Disjoint ↑p.1 ↑p.2
      q : A → Set β
      hq1 : ∀ (p : A), HasSubset.Subset (Set.image f ↑(↑p).1) (q p)
      hq2 : ∀ (p : A), Disjoint (Set.image f ↑(↑p).2) (q p)
      q_meas : ∀ (p : A), MeasurableSet (q p)
      E : ↑b → Set β := fun s => Inter.inter (closure (Set.image f ↑s)) (Set.iInter  …
      u : Nat → Real
      u_anti : StrictAnti u
      u_pos : ∀ (n : Nat), LT.lt 0 (u n)
      u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
      F : Nat → Set β := fun n => Set.iUnion fun s => Set.iUnion fun x => E s
      y : γ
      n : Nat
      s : Set γ
      sb : Membership.mem b s
      ys : Membership.mem s y
      hs : HasSubset.Subset s (Metric.ball y (HDiv.hDiv (u n) 2))
      diam_s : LE.le (Metric.diam s) (u n)
      ⊢ Membership.mem (Set.iUnion fun x => E ⟨s, sb⟩) (f y)
    -/
    refine mem_iUnion.2 ⟨⟨isBounded_ball.subset hs, diam_s⟩, ?_⟩
    /-
      case intro.intro.intro.intro.intro.intro.h₁.intro.intro.intro.intro
      γ : Type u_3
      β : Type u_4
      inst✝⁵ : TopologicalSpace γ
      inst✝⁴ : PolishSpace γ
      inst✝³ : TopologicalSpace β
      inst✝² : T2Space β
      inst✝¹ : MeasurableSpace β
      inst✝ : OpensMeasurableSpace β
      f : γ → β
      f_cont : Continuous f
      f_inj : Function.Injective f
      this✝ : UpgradedPolishSpace γ := upgradePolishSpace γ
      b : Set (Set γ)
      b_count : b.Countable
      b_nonempty : Not (Membership.mem b EmptyCollection.emptyCollection)
      hb : TopologicalSpace.IsTopologicalBasis b
      this : Encodable ↑b
      A : Type (max 0 u_3) := Subtype fun p => Disjoint ↑p.1 ↑p.2
      q : A → Set β
      hq1 : ∀ (p : A), HasSubset.Subset (Set.image f ↑(↑p).1) (q p)
      hq2 : ∀ (p : A), Disjoint (Set.image f ↑(↑p).2) (q p)
      q_meas : ∀ (p : A), MeasurableSet (q p)
      E : ↑b → Set β := fun s => Inter.inter (closure (Set.image f ↑s)) (Set.iInter  …
      u : Nat → Real
      u_anti : StrictAnti u
      u_pos : ∀ (n : Nat), LT.lt 0 (u n)
      u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
      F : Nat → Set β := fun n => Set.iUnion fun s => Set.iUnion fun x => E s
      y : γ
      n : Nat
      s : Set γ
      sb : Membership.mem b s
      ys : Membership.mem s y
      hs : HasSubset.Subset s (Metric.ball y (HDiv.hDiv (u n) 2))
      diam_s : LE.le (Metric.diam s) (u n)
      ⊢ Membership.mem (E ⟨s, sb⟩) (f y)
    -/
    apply mem_inter (subset_closure (mem_image_of_mem _ ys))
    /-
      case intro.intro.intro.intro.intro.intro.h₁.intro.intro.intro.intro
      γ : Type u_3
      β : Type u_4
      inst✝⁵ : TopologicalSpace γ
      inst✝⁴ : PolishSpace γ
      inst✝³ : TopologicalSpace β
      inst✝² : T2Space β
      inst✝¹ : MeasurableSpace β
      inst✝ : OpensMeasurableSpace β
      f : γ → β
      f_cont : Continuous f
      f_inj : Function.Injective f
      this✝ : UpgradedPolishSpace γ := upgradePolishSpace γ
      b : Set (Set γ)
      b_count : b.Countable
      b_nonempty : Not (Membership.mem b EmptyCollection.emptyCollection)
      hb : TopologicalSpace.IsTopologicalBasis b
      this : Encodable ↑b
      A : Type (max 0 u_3) := Subtype fun p => Disjoint ↑p.1 ↑p.2
      q : A → Set β
      hq1 : ∀ (p : A), HasSubset.Subset (Set.image f ↑(↑p).1) (q p)
      hq2 : ∀ (p : A), Disjoint (Set.image f ↑(↑p).2) (q p)
      q_meas : ∀ (p : A), MeasurableSet (q p)
      E : ↑b → Set β := fun s => Inter.inter (closure (Set.image f ↑s)) (Set.iInter  …
      u : Nat → Real
      u_anti : StrictAnti u
      u_pos : ∀ (n : Nat), LT.lt 0 (u n)
      u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
      F : Nat → Set β := fun n => Set.iUnion fun s => Set.iUnion fun x => E s
      y : γ
      n : Nat
      s : Set γ
      sb : Membership.mem b s
      ys : Membership.mem s y
      hs : HasSubset.Subset s (Metric.ball y (HDiv.hDiv (u n) 2))
      diam_s : LE.le (Metric.diam s) (u n)
      ⊢ Membership.mem (Set.iInter fun t => Set.iInter fun ht => SDiff.sdiff (q ⟨{ f …
    -/
    refine mem_iInter.2 fun t => mem_iInter.2 fun ht => ⟨?_, ?_⟩
      /-
        case intro.intro.intro.intro.intro.intro.h₁.intro.intro.intro.intro.refine_1
        γ : Type u_3
        β : Type u_4
        inst✝⁵ : TopologicalSpace γ
        inst✝⁴ : PolishSpace γ
        inst✝³ : TopologicalSpace β
        inst✝² : T2Space β
        inst✝¹ : MeasurableSpace β
        inst✝ : OpensMeasurableSpace β
        f : γ → β
        f_cont : Continuous f
        f_inj : Function.Injective f
        this✝ : UpgradedPolishSpace γ := upgradePolishSpace γ
        b : Set (Set γ)
        b_count : b.Countable
        b_nonempty : Not (Membership.mem b EmptyCollection.emptyCollection)
        hb : TopologicalSpace.IsTopologicalBasis b
        this : Encodable ↑b
        A : Type (max 0 u_3) := Subtype fun p => Disjoint ↑p.1 ↑p.2
        q : A → Set β
        hq1 : ∀ (p : A), HasSubset.Subset (Set.image f ↑(↑p).1) (q p)
        hq2 : ∀ (p : A), Disjoint (Set.image f ↑(↑p).2) (q p)
        q_meas : ∀ (p : A), MeasurableSet (q p)
        E : ↑b → Set β := fun s => Inter.inter (closure (Set.image f ↑s)) (Set.iInter  …
        u : Nat → Real
        u_anti : StrictAnti u
        u_pos : ∀ (n : Nat), LT.lt 0 (u n)
        u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
        F : Nat → Set β := fun n => Set.iUnion fun s => Set.iUnion fun x => E s
        y : γ
        n : Nat
        s : Set γ
        sb : Membership.mem b s
        ys : Membership.mem s y
        hs : HasSubset.Subset s (Metric.ball y (HDiv.hDiv (u n) 2))
        diam_s : LE.le (Metric.diam s) (u n)
        t : ↑b
        ht : Disjoint ↑⟨s, sb⟩ ↑t
        ⊢ Membership.mem (q ⟨{ fst := ⟨s, sb⟩, snd := t }, ht⟩) (f y)
      -/
    · apply hq1
      /-
        case intro.intro.intro.intro.intro.intro.h₁.intro.intro.intro.intro.refine_1.a
        γ : Type u_3
        β : Type u_4
        inst✝⁵ : TopologicalSpace γ
        inst✝⁴ : PolishSpace γ
        inst✝³ : TopologicalSpace β
        inst✝² : T2Space β
        inst✝¹ : MeasurableSpace β
        inst✝ : OpensMeasurableSpace β
        f : γ → β
        f_cont : Continuous f
        f_inj : Function.Injective f
        this✝ : UpgradedPolishSpace γ := upgradePolishSpace γ
        b : Set (Set γ)
        b_count : b.Countable
        b_nonempty : Not (Membership.mem b EmptyCollection.emptyCollection)
        hb : TopologicalSpace.IsTopologicalBasis b
        this : Encodable ↑b
        A : Type (max 0 u_3) := Subtype fun p => Disjoint ↑p.1 ↑p.2
        q : A → Set β
        hq1 : ∀ (p : A), HasSubset.Subset (Set.image f ↑(↑p).1) (q p)
        hq2 : ∀ (p : A), Disjoint (Set.image f ↑(↑p).2) (q p)
        q_meas : ∀ (p : A), MeasurableSet (q p)
        E : ↑b → Set β := fun s => Inter.inter (closure (Set.image f ↑s)) (Set.iInter  …
        u : Nat → Real
        u_anti : StrictAnti u
        u_pos : ∀ (n : Nat), LT.lt 0 (u n)
        u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
        F : Nat → Set β := fun n => Set.iUnion fun s => Set.iUnion fun x => E s
        y : γ
        n : Nat
        s : Set γ
        sb : Membership.mem b s
        ys : Membership.mem s y
        hs : HasSubset.Subset s (Metric.ball y (HDiv.hDiv (u n) 2))
        diam_s : LE.le (Metric.diam s) (u n)
        t : ↑b
        ht : Disjoint ↑⟨s, sb⟩ ↑t
        ⊢ Membership.mem (Set.image f ↑(↑⟨{ fst := ⟨s, sb⟩, snd := t }, ht⟩).1) (f y)
      -/
      exact mem_image_of_mem _ ys
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.intro.intro.intro.h₁.intro.intro.intro.intro.refine_2
        γ : Type u_3
        β : Type u_4
        inst✝⁵ : TopologicalSpace γ
        inst✝⁴ : PolishSpace γ
        inst✝³ : TopologicalSpace β
        inst✝² : T2Space β
        inst✝¹ : MeasurableSpace β
        inst✝ : OpensMeasurableSpace β
        f : γ → β
        f_cont : Continuous f
        f_inj : Function.Injective f
        this✝ : UpgradedPolishSpace γ := upgradePolishSpace γ
        b : Set (Set γ)
        b_count : b.Countable
        b_nonempty : Not (Membership.mem b EmptyCollection.emptyCollection)
        hb : TopologicalSpace.IsTopologicalBasis b
        this : Encodable ↑b
        A : Type (max 0 u_3) := Subtype fun p => Disjoint ↑p.1 ↑p.2
        q : A → Set β
        hq1 : ∀ (p : A), HasSubset.Subset (Set.image f ↑(↑p).1) (q p)
        hq2 : ∀ (p : A), Disjoint (Set.image f ↑(↑p).2) (q p)
        q_meas : ∀ (p : A), MeasurableSet (q p)
        E : ↑b → Set β := fun s => Inter.inter (closure (Set.image f ↑s)) (Set.iInter  …
        u : Nat → Real
        u_anti : StrictAnti u
        u_pos : ∀ (n : Nat), LT.lt 0 (u n)
        u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
        F : Nat → Set β := fun n => Set.iUnion fun s => Set.iUnion fun x => E s
        y : γ
        n : Nat
        s : Set γ
        sb : Membership.mem b s
        ys : Membership.mem s y
        hs : HasSubset.Subset s (Metric.ball y (HDiv.hDiv (u n) 2))
        diam_s : LE.le (Metric.diam s) (u n)
        t : ↑b
        ht : Disjoint ↑⟨s, sb⟩ ↑t
        ⊢ Not (Membership.mem (q ⟨{ fst := t, snd := ⟨s, sb⟩ }, ⋯⟩) (f y))
      -/
    · apply disjoint_left.1 (hq2 ⟨(t, ⟨s, sb⟩), ht.symm⟩)
      /-
        case intro.intro.intro.intro.intro.intro.h₁.intro.intro.intro.intro.refine_2.a
        γ : Type u_3
        β : Type u_4
        inst✝⁵ : TopologicalSpace γ
        inst✝⁴ : PolishSpace γ
        inst✝³ : TopologicalSpace β
        inst✝² : T2Space β
        inst✝¹ : MeasurableSpace β
        inst✝ : OpensMeasurableSpace β
        f : γ → β
        f_cont : Continuous f
        f_inj : Function.Injective f
        this✝ : UpgradedPolishSpace γ := upgradePolishSpace γ
        b : Set (Set γ)
        b_count : b.Countable
        b_nonempty : Not (Membership.mem b EmptyCollection.emptyCollection)
        hb : TopologicalSpace.IsTopologicalBasis b
        this : Encodable ↑b
        A : Type (max 0 u_3) := Subtype fun p => Disjoint ↑p.1 ↑p.2
        q : A → Set β
        hq1 : ∀ (p : A), HasSubset.Subset (Set.image f ↑(↑p).1) (q p)
        hq2 : ∀ (p : A), Disjoint (Set.image f ↑(↑p).2) (q p)
        q_meas : ∀ (p : A), MeasurableSet (q p)
        E : ↑b → Set β := fun s => Inter.inter (closure (Set.image f ↑s)) (Set.iInter  …
        u : Nat → Real
        u_anti : StrictAnti u
        u_pos : ∀ (n : Nat), LT.lt 0 (u n)
        u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
        F : Nat → Set β := fun n => Set.iUnion fun s => Set.iUnion fun x => E s
        y : γ
        n : Nat
        s : Set γ
        sb : Membership.mem b s
        ys : Membership.mem s y
        hs : HasSubset.Subset s (Metric.ball y (HDiv.hDiv (u n) 2))
        diam_s : LE.le (Metric.diam s) (u n)
        t : ↑b
        ht : Disjoint ↑⟨s, sb⟩ ↑t
        ⊢ Membership.mem (Set.image f ↑(↑⟨{ fst := t, snd := ⟨s, sb⟩ }, ⋯⟩).2) (f y)
      -/
      exact mem_image_of_mem _ ys
      /-
        🎉 no goals
      -/
  -- Now, let us prove the harder inclusion `⋂ F n ⊆ range f`.
    /-
      case intro.intro.intro.intro.intro.intro.h₂
      γ : Type u_3
      β : Type u_4
      inst✝⁵ : TopologicalSpace γ
      inst✝⁴ : PolishSpace γ
      inst✝³ : TopologicalSpace β
      inst✝² : T2Space β
      inst✝¹ : MeasurableSpace β
      inst✝ : OpensMeasurableSpace β
      f : γ → β
      f_cont : Continuous f
      f_inj : Function.Injective f
      this✝ : UpgradedPolishSpace γ := upgradePolishSpace γ
      b : Set (Set γ)
      b_count : b.Countable
      b_nonempty : Not (Membership.mem b EmptyCollection.emptyCollection)
      hb : TopologicalSpace.IsTopologicalBasis b
      this : Encodable ↑b
      A : Type (max 0 u_3) := Subtype fun p => Disjoint ↑p.1 ↑p.2
      q : A → Set β
      hq1 : ∀ (p : A), HasSubset.Subset (Set.image f ↑(↑p).1) (q p)
      hq2 : ∀ (p : A), Disjoint (Set.image f ↑(↑p).2) (q p)
      q_meas : ∀ (p : A), MeasurableSet (q p)
      E : ↑b → Set β := fun s => Inter.inter (closure (Set.image f ↑s)) (Set.iInter  …
      u : Nat → Real
      u_anti : StrictAnti u
      u_pos : ∀ (n : Nat), LT.lt 0 (u n)
      u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
      F : Nat → Set β := fun n => Set.iUnion fun s => Set.iUnion fun x => E s
      ⊢ HasSubset.Subset (Set.iInter fun n => F n) (Set.range f)
    -/
  · intro x hx
    -- pick for each `n` a good set `s n` of small diameter for which `x ∈ E (s n)`.
    have C1 : ∀ n, ∃ (s : b) (_ : IsBounded s.1 ∧ diam s.1 ≤ u n), x ∈ E s := fun n => by
      simpa only [F, mem_iUnion] using mem_iInter.1 hx n
    /-
      case intro.intro.intro.intro.intro.intro.h₂
      γ : Type u_3
      β : Type u_4
      inst✝⁵ : TopologicalSpace γ
      inst✝⁴ : PolishSpace γ
      inst✝³ : TopologicalSpace β
      inst✝² : T2Space β
      inst✝¹ : MeasurableSpace β
      inst✝ : OpensMeasurableSpace β
      f : γ → β
      f_cont : Continuous f
      f_inj : Function.Injective f
      this✝ : UpgradedPolishSpace γ := upgradePolishSpace γ
      b : Set (Set γ)
      b_count : b.Countable
      b_nonempty : Not (Membership.mem b EmptyCollection.emptyCollection)
      hb : TopologicalSpace.IsTopologicalBasis b
      this : Encodable ↑b
      A : Type (max 0 u_3) := Subtype fun p => Disjoint ↑p.1 ↑p.2
      q : A → Set β
      hq1 : ∀ (p : A), HasSubset.Subset (Set.image f ↑(↑p).1) (q p)
      hq2 : ∀ (p : A), Disjoint (Set.image f ↑(↑p).2) (q p)
      q_meas : ∀ (p : A), MeasurableSet (q p)
      E : ↑b → Set β := fun s => Inter.inter (closure (Set.image f ↑s)) (Set.iInter  …
      u : Nat → Real
      u_anti : StrictAnti u
      u_pos : ∀ (n : Nat), LT.lt 0 (u n)
      u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
      F : Nat → Set β := fun n => Set.iUnion fun s => Set.iUnion fun x => E s
      x : β
      hx : Membership.mem (Set.iInter fun n => F n) x
      C1 : ∀ (n : Nat), Exists fun s => Exists fun x_1 => Membership.mem (E s) x
      ⊢ Membership.mem (Set.range f) x
    -/
    choose s hs hxs using C1
    have C2 : ∀ n, (s n).1.Nonempty := by
      intro n
      rw [nonempty_iff_ne_empty]
      intro hn
      have := (s n).2
      rw [hn] at this
      exact b_nonempty this
    -- choose a point `y n ∈ s n`.
    /-
      case intro.intro.intro.intro.intro.intro.h₂
      γ : Type u_3
      β : Type u_4
      inst✝⁵ : TopologicalSpace γ
      inst✝⁴ : PolishSpace γ
      inst✝³ : TopologicalSpace β
      inst✝² : T2Space β
      inst✝¹ : MeasurableSpace β
      inst✝ : OpensMeasurableSpace β
      f : γ → β
      f_cont : Continuous f
      f_inj : Function.Injective f
      this✝ : UpgradedPolishSpace γ := upgradePolishSpace γ
      b : Set (Set γ)
      b_count : b.Countable
      b_nonempty : Not (Membership.mem b EmptyCollection.emptyCollection)
      hb : TopologicalSpace.IsTopologicalBasis b
      this : Encodable ↑b
      A : Type (max 0 u_3) := Subtype fun p => Disjoint ↑p.1 ↑p.2
      q : A → Set β
      hq1 : ∀ (p : A), HasSubset.Subset (Set.image f ↑(↑p).1) (q p)
      hq2 : ∀ (p : A), Disjoint (Set.image f ↑(↑p).2) (q p)
      q_meas : ∀ (p : A), MeasurableSet (q p)
      E : ↑b → Set β := fun s => Inter.inter (closure (Set.image f ↑s)) (Set.iInter  …
      u : Nat → Real
      u_anti : StrictAnti u
      u_pos : ∀ (n : Nat), LT.lt 0 (u n)
      u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
      F : Nat → Set β := fun n => Set.iUnion fun s => Set.iUnion fun x => E s
      x : β
      hx : Membership.mem (Set.iInter fun n => F n) x
      s : Nat → ↑b
      hs : ∀ (n : Nat), And (Bornology.IsBounded ↑(s n)) (LE.le (Metric.diam ↑(s n)) …
      hxs : ∀ (n : Nat), Membership.mem (E (s n)) x
      C2 : ∀ (n : Nat), (↑(s n)).Nonempty
      ⊢ Membership.mem (Set.range f) x
    -/
    choose y hy using C2
    have I : ∀ m n, ((s m).1 ∩ (s n).1).Nonempty := by
      intro m n
      rw [← not_disjoint_iff_nonempty_inter]
      by_contra! h
      have A : x ∈ q ⟨(s m, s n), h⟩ \ q ⟨(s n, s m), h.symm⟩ :=
        haveI := mem_iInter.1 (hxs m).2 (s n)
        (mem_iInter.1 this h : _)
      have B : x ∈ q ⟨(s n, s m), h.symm⟩ \ q ⟨(s m, s n), h⟩ :=
        haveI := mem_iInter.1 (hxs n).2 (s m)
        (mem_iInter.1 this h.symm : _)
      exact A.2 B.1
    -- the points `y n` are nearby, and therefore they form a Cauchy sequence.
    have cauchy_y : CauchySeq y := by
      have : Tendsto (fun n => 2 * u n) atTop (𝓝 0) := by
        simpa only [mul_zero] using u_lim.const_mul 2
      refine cauchySeq_of_le_tendsto_0' (fun n => 2 * u n) (fun m n hmn => ?_) this
      rcases I m n with ⟨z, zsm, zsn⟩
      calc
        dist (y m) (y n) ≤ dist (y m) z + dist z (y n) := dist_triangle _ _ _
        _ ≤ u m + u n :=
          (add_le_add ((dist_le_diam_of_mem (hs m).1 (hy m) zsm).trans (hs m).2)
            ((dist_le_diam_of_mem (hs n).1 zsn (hy n)).trans (hs n).2))
        _ ≤ 2 * u m := by linarith [u_anti.antitone hmn]
    /-
      case intro.intro.intro.intro.intro.intro.h₂
      γ : Type u_3
      β : Type u_4
      inst✝⁵ : TopologicalSpace γ
      inst✝⁴ : PolishSpace γ
      inst✝³ : TopologicalSpace β
      inst✝² : T2Space β
      inst✝¹ : MeasurableSpace β
      inst✝ : OpensMeasurableSpace β
      f : γ → β
      f_cont : Continuous f
      f_inj : Function.Injective f
      this✝ : UpgradedPolishSpace γ := upgradePolishSpace γ
      b : Set (Set γ)
      b_count : b.Countable
      b_nonempty : Not (Membership.mem b EmptyCollection.emptyCollection)
      hb : TopologicalSpace.IsTopologicalBasis b
      this : Encodable ↑b
      A : Type (max 0 u_3) := Subtype fun p => Disjoint ↑p.1 ↑p.2
      q : A → Set β
      hq1 : ∀ (p : A), HasSubset.Subset (Set.image f ↑(↑p).1) (q p)
      hq2 : ∀ (p : A), Disjoint (Set.image f ↑(↑p).2) (q p)
      q_meas : ∀ (p : A), MeasurableSet (q p)
      E : ↑b → Set β := fun s => Inter.inter (closure (Set.image f ↑s)) (Set.iInter  …
      u : Nat → Real
      u_anti : StrictAnti u
      u_pos : ∀ (n : Nat), LT.lt 0 (u n)
      u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
      F : Nat → Set β := fun n => Set.iUnion fun s => Set.iUnion fun x => E s
      x : β
      hx : Membership.mem (Set.iInter fun n => F n) x
      s : Nat → ↑b
      hs : ∀ (n : Nat), And (Bornology.IsBounded ↑(s n)) (LE.le (Metric.diam ↑(s n)) …
      hxs : ∀ (n : Nat), Membership.mem (E (s n)) x
      y : Nat → γ
      hy : ∀ (n : Nat), Membership.mem (↑(s n)) (y n)
      I : ∀ (m n : Nat), (Inter.inter ↑(s m) ↑(s n)).Nonempty
      cauchy_y : CauchySeq y
      ⊢ Membership.mem (Set.range f) x
    -/
    haveI : Nonempty γ := ⟨y 0⟩
    -- let `z` be its limit.
    /-
      case intro.intro.intro.intro.intro.intro.h₂
      γ : Type u_3
      β : Type u_4
      inst✝⁵ : TopologicalSpace γ
      inst✝⁴ : PolishSpace γ
      inst✝³ : TopologicalSpace β
      inst✝² : T2Space β
      inst✝¹ : MeasurableSpace β
      inst✝ : OpensMeasurableSpace β
      f : γ → β
      f_cont : Continuous f
      f_inj : Function.Injective f
      this✝¹ : UpgradedPolishSpace γ := upgradePolishSpace γ
      b : Set (Set γ)
      b_count : b.Countable
      b_nonempty : Not (Membership.mem b EmptyCollection.emptyCollection)
      hb : TopologicalSpace.IsTopologicalBasis b
      this✝ : Encodable ↑b
      A : Type (max 0 u_3) := Subtype fun p => Disjoint ↑p.1 ↑p.2
      q : A → Set β
      hq1 : ∀ (p : A), HasSubset.Subset (Set.image f ↑(↑p).1) (q p)
      hq2 : ∀ (p : A), Disjoint (Set.image f ↑(↑p).2) (q p)
      q_meas : ∀ (p : A), MeasurableSet (q p)
      E : ↑b → Set β := fun s => Inter.inter (closure (Set.image f ↑s)) (Set.iInter  …
      u : Nat → Real
      u_anti : StrictAnti u
      u_pos : ∀ (n : Nat), LT.lt 0 (u n)
      u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
      F : Nat → Set β := fun n => Set.iUnion fun s => Set.iUnion fun x => E s
      x : β
      hx : Membership.mem (Set.iInter fun n => F n) x
      s : Nat → ↑b
      hs : ∀ (n : Nat), And (Bornology.IsBounded ↑(s n)) (LE.le (Metric.diam ↑(s n)) …
      hxs : ∀ (n : Nat), Membership.mem (E (s n)) x
      y : Nat → γ
      hy : ∀ (n : Nat), Membership.mem (↑(s n)) (y n)
      I : ∀ (m n : Nat), (Inter.inter ↑(s m) ↑(s n)).Nonempty
      cauchy_y : CauchySeq y
      this : Nonempty γ
      ⊢ Membership.mem (Set.range f) x
    -/
    let z := limUnder atTop y
    /-
      case intro.intro.intro.intro.intro.intro.h₂
      γ : Type u_3
      β : Type u_4
      inst✝⁵ : TopologicalSpace γ
      inst✝⁴ : PolishSpace γ
      inst✝³ : TopologicalSpace β
      inst✝² : T2Space β
      inst✝¹ : MeasurableSpace β
      inst✝ : OpensMeasurableSpace β
      f : γ → β
      f_cont : Continuous f
      f_inj : Function.Injective f
      this✝¹ : UpgradedPolishSpace γ := upgradePolishSpace γ
      b : Set (Set γ)
      b_count : b.Countable
      b_nonempty : Not (Membership.mem b EmptyCollection.emptyCollection)
      hb : TopologicalSpace.IsTopologicalBasis b
      this✝ : Encodable ↑b
      A : Type (max 0 u_3) := Subtype fun p => Disjoint ↑p.1 ↑p.2
      q : A → Set β
      hq1 : ∀ (p : A), HasSubset.Subset (Set.image f ↑(↑p).1) (q p)
      hq2 : ∀ (p : A), Disjoint (Set.image f ↑(↑p).2) (q p)
      q_meas : ∀ (p : A), MeasurableSet (q p)
      E : ↑b → Set β := fun s => Inter.inter (closure (Set.image f ↑s)) (Set.iInter  …
      u : Nat → Real
      u_anti : StrictAnti u
      u_pos : ∀ (n : Nat), LT.lt 0 (u n)
      u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
      F : Nat → Set β := fun n => Set.iUnion fun s => Set.iUnion fun x => E s
      x : β
      hx : Membership.mem (Set.iInter fun n => F n) x
      s : Nat → ↑b
      hs : ∀ (n : Nat), And (Bornology.IsBounded ↑(s n)) (LE.le (Metric.diam ↑(s n)) …
      hxs : ∀ (n : Nat), Membership.mem (E (s n)) x
      y : Nat → γ
      hy : ∀ (n : Nat), Membership.mem (↑(s n)) (y n)
      I : ∀ (m n : Nat), (Inter.inter ↑(s m) ↑(s n)).Nonempty
      cauchy_y : CauchySeq y
      this : Nonempty γ
      z : γ := limUnder Filter.atTop y
      ⊢ Membership.mem (Set.range f) x
    -/
    have y_lim : Tendsto y atTop (𝓝 z) := cauchy_y.tendsto_limUnder
    suffices f z = x by
      rw [← this]
      exact mem_range_self _
    -- assume for a contradiction that `f z ≠ x`.
    /-
      case intro.intro.intro.intro.intro.intro.h₂
      γ : Type u_3
      β : Type u_4
      inst✝⁵ : TopologicalSpace γ
      inst✝⁴ : PolishSpace γ
      inst✝³ : TopologicalSpace β
      inst✝² : T2Space β
      inst✝¹ : MeasurableSpace β
      inst✝ : OpensMeasurableSpace β
      f : γ → β
      f_cont : Continuous f
      f_inj : Function.Injective f
      this✝¹ : UpgradedPolishSpace γ := upgradePolishSpace γ
      b : Set (Set γ)
      b_count : b.Countable
      b_nonempty : Not (Membership.mem b EmptyCollection.emptyCollection)
      hb : TopologicalSpace.IsTopologicalBasis b
      this✝ : Encodable ↑b
      A : Type (max 0 u_3) := Subtype fun p => Disjoint ↑p.1 ↑p.2
      q : A → Set β
      hq1 : ∀ (p : A), HasSubset.Subset (Set.image f ↑(↑p).1) (q p)
      hq2 : ∀ (p : A), Disjoint (Set.image f ↑(↑p).2) (q p)
      q_meas : ∀ (p : A), MeasurableSet (q p)
      E : ↑b → Set β := fun s => Inter.inter (closure (Set.image f ↑s)) (Set.iInter  …
      u : Nat → Real
      u_anti : StrictAnti u
      u_pos : ∀ (n : Nat), LT.lt 0 (u n)
      u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
      F : Nat → Set β := fun n => Set.iUnion fun s => Set.iUnion fun x => E s
      x : β
      hx : Membership.mem (Set.iInter fun n => F n) x
      s : Nat → ↑b
      hs : ∀ (n : Nat), And (Bornology.IsBounded ↑(s n)) (LE.le (Metric.diam ↑(s n)) …
      hxs : ∀ (n : Nat), Membership.mem (E (s n)) x
      y : Nat → γ
      hy : ∀ (n : Nat), Membership.mem (↑(s n)) (y n)
      I : ∀ (m n : Nat), (Inter.inter ↑(s m) ↑(s n)).Nonempty
      cauchy_y : CauchySeq y
      this : Nonempty γ
      z : γ := limUnder Filter.atTop y
      y_lim : Filter.Tendsto y Filter.atTop (nhds z)
      ⊢ Eq (f z) x
    -/
    by_contra! hne
    -- introduce disjoint open sets `v` and `w` separating `f z` from `x`.
    /-
      case intro.intro.intro.intro.intro.intro.h₂
      γ : Type u_3
      β : Type u_4
      inst✝⁵ : TopologicalSpace γ
      inst✝⁴ : PolishSpace γ
      inst✝³ : TopologicalSpace β
      inst✝² : T2Space β
      inst✝¹ : MeasurableSpace β
      inst✝ : OpensMeasurableSpace β
      f : γ → β
      f_cont : Continuous f
      f_inj : Function.Injective f
      this✝¹ : UpgradedPolishSpace γ := upgradePolishSpace γ
      b : Set (Set γ)
      b_count : b.Countable
      b_nonempty : Not (Membership.mem b EmptyCollection.emptyCollection)
      hb : TopologicalSpace.IsTopologicalBasis b
      this✝ : Encodable ↑b
      A : Type (max 0 u_3) := Subtype fun p => Disjoint ↑p.1 ↑p.2
      q : A → Set β
      hq1 : ∀ (p : A), HasSubset.Subset (Set.image f ↑(↑p).1) (q p)
      hq2 : ∀ (p : A), Disjoint (Set.image f ↑(↑p).2) (q p)
      q_meas : ∀ (p : A), MeasurableSet (q p)
      E : ↑b → Set β := fun s => Inter.inter (closure (Set.image f ↑s)) (Set.iInter  …
      u : Nat → Real
      u_anti : StrictAnti u
      u_pos : ∀ (n : Nat), LT.lt 0 (u n)
      u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
      F : Nat → Set β := fun n => Set.iUnion fun s => Set.iUnion fun x => E s
      x : β
      hx : Membership.mem (Set.iInter fun n => F n) x
      s : Nat → ↑b
      hs : ∀ (n : Nat), And (Bornology.IsBounded ↑(s n)) (LE.le (Metric.diam ↑(s n)) …
      hxs : ∀ (n : Nat), Membership.mem (E (s n)) x
      y : Nat → γ
      hy : ∀ (n : Nat), Membership.mem (↑(s n)) (y n)
      I : ∀ (m n : Nat), (Inter.inter ↑(s m) ↑(s n)).Nonempty
      cauchy_y : CauchySeq y
      this : Nonempty γ
      z : γ := limUnder Filter.atTop y
      y_lim : Filter.Tendsto y Filter.atTop (nhds z)
      hne : Ne (f z) x
      ⊢ False
    -/
    obtain ⟨v, w, v_open, w_open, fzv, xw, hvw⟩ := t2_separation hne
    obtain ⟨δ, δpos, hδ⟩ : ∃ δ > (0 : ℝ), ball z δ ⊆ f ⁻¹' v := by
      apply Metric.mem_nhds_iff.1
      exact f_cont.continuousAt.preimage_mem_nhds (v_open.mem_nhds fzv)
    obtain ⟨n, hn⟩ : ∃ n, u n + dist (y n) z < δ :=
      haveI : Tendsto (fun n => u n + dist (y n) z) atTop (𝓝 0) := by
        simpa only [add_zero] using u_lim.add (tendsto_iff_dist_tendsto_zero.1 y_lim)
      ((tendsto_order.1 this).2 _ δpos).exists
    -- for large enough `n`, the image of `s n` is contained in `v`, by continuity of `f`.
    have fsnv : f '' s n ⊆ v := by
      rw [image_subset_iff]
      apply Subset.trans _ hδ
      intro a ha
      calc
        dist a z ≤ dist a (y n) + dist (y n) z := dist_triangle _ _ _
        _ ≤ u n + dist (y n) z :=
          (add_le_add_right ((dist_le_diam_of_mem (hs n).1 ha (hy n)).trans (hs n).2) _)
        _ < δ := hn
    -- as `x` belongs to the closure of `f '' (s n)`, it belongs to the closure of `v`.
    /-
      case intro.intro.intro.intro.intro.intro.h₂.intro.intro.intro.intro.intro.intr …
      γ : Type u_3
      β : Type u_4
      inst✝⁵ : TopologicalSpace γ
      inst✝⁴ : PolishSpace γ
      inst✝³ : TopologicalSpace β
      inst✝² : T2Space β
      inst✝¹ : MeasurableSpace β
      inst✝ : OpensMeasurableSpace β
      f : γ → β
      f_cont : Continuous f
      f_inj : Function.Injective f
      this✝¹ : UpgradedPolishSpace γ := upgradePolishSpace γ
      b : Set (Set γ)
      b_count : b.Countable
      b_nonempty : Not (Membership.mem b EmptyCollection.emptyCollection)
      hb : TopologicalSpace.IsTopologicalBasis b
      this✝ : Encodable ↑b
      A : Type (max 0 u_3) := Subtype fun p => Disjoint ↑p.1 ↑p.2
      q : A → Set β
      hq1 : ∀ (p : A), HasSubset.Subset (Set.image f ↑(↑p).1) (q p)
      hq2 : ∀ (p : A), Disjoint (Set.image f ↑(↑p).2) (q p)
      q_meas : ∀ (p : A), MeasurableSet (q p)
      E : ↑b → Set β := fun s => Inter.inter (closure (Set.image f ↑s)) (Set.iInter  …
      u : Nat → Real
      u_anti : StrictAnti u
      u_pos : ∀ (n : Nat), LT.lt 0 (u n)
      u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
      F : Nat → Set β := fun n => Set.iUnion fun s => Set.iUnion fun x => E s
      x : β
      hx : Membership.mem (Set.iInter fun n => F n) x
      s : Nat → ↑b
      hs : ∀ (n : Nat), And (Bornology.IsBounded ↑(s n)) (LE.le (Metric.diam ↑(s n)) …
      hxs : ∀ (n : Nat), Membership.mem (E (s n)) x
      y : Nat → γ
      hy : ∀ (n : Nat), Membership.mem (↑(s n)) (y n)
      I : ∀ (m n : Nat), (Inter.inter ↑(s m) ↑(s n)).Nonempty
      cauchy_y : CauchySeq y
      this : Nonempty γ
      z : γ := limUnder Filter.atTop y
      y_lim : Filter.Tendsto y Filter.atTop (nhds z)
      hne : Ne (f z) x
      v w : Set β
      v_open : IsOpen v
      w_open : IsOpen w
      fzv : Membership.mem v (f z)
      xw : Membership.mem w x
      hvw : Disjoint v w
      δ : Real
      δpos : GT.gt δ 0
      hδ : HasSubset.Subset (Metric.ball z δ) (Set.preimage f v)
      n : Nat
      hn : LT.lt (HAdd.hAdd (u n) (Dist.dist (y n) z)) δ
      fsnv : HasSubset.Subset (Set.image f ↑(s n)) v
      ⊢ False
    -/
    have : x ∈ closure v := closure_mono fsnv (hxs n).1
    -- this is a contradiction, as `x` is supposed to belong to `w`, which is disjoint from
    -- the closure of `v`.
    /-
      case intro.intro.intro.intro.intro.intro.h₂.intro.intro.intro.intro.intro.intr …
      γ : Type u_3
      β : Type u_4
      inst✝⁵ : TopologicalSpace γ
      inst✝⁴ : PolishSpace γ
      inst✝³ : TopologicalSpace β
      inst✝² : T2Space β
      inst✝¹ : MeasurableSpace β
      inst✝ : OpensMeasurableSpace β
      f : γ → β
      f_cont : Continuous f
      f_inj : Function.Injective f
      this✝² : UpgradedPolishSpace γ := upgradePolishSpace γ
      b : Set (Set γ)
      b_count : b.Countable
      b_nonempty : Not (Membership.mem b EmptyCollection.emptyCollection)
      hb : TopologicalSpace.IsTopologicalBasis b
      this✝¹ : Encodable ↑b
      A : Type (max 0 u_3) := Subtype fun p => Disjoint ↑p.1 ↑p.2
      q : A → Set β
      hq1 : ∀ (p : A), HasSubset.Subset (Set.image f ↑(↑p).1) (q p)
      hq2 : ∀ (p : A), Disjoint (Set.image f ↑(↑p).2) (q p)
      q_meas : ∀ (p : A), MeasurableSet (q p)
      E : ↑b → Set β := fun s => Inter.inter (closure (Set.image f ↑s)) (Set.iInter  …
      u : Nat → Real
      u_anti : StrictAnti u
      u_pos : ∀ (n : Nat), LT.lt 0 (u n)
      u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
      F : Nat → Set β := fun n => Set.iUnion fun s => Set.iUnion fun x => E s
      x : β
      hx : Membership.mem (Set.iInter fun n => F n) x
      s : Nat → ↑b
      hs : ∀ (n : Nat), And (Bornology.IsBounded ↑(s n)) (LE.le (Metric.diam ↑(s n)) …
      hxs : ∀ (n : Nat), Membership.mem (E (s n)) x
      y : Nat → γ
      hy : ∀ (n : Nat), Membership.mem (↑(s n)) (y n)
      I : ∀ (m n : Nat), (Inter.inter ↑(s m) ↑(s n)).Nonempty
      cauchy_y : CauchySeq y
      this✝ : Nonempty γ
      z : γ := limUnder Filter.atTop y
      y_lim : Filter.Tendsto y Filter.atTop (nhds z)
      hne : Ne (f z) x
      v w : Set β
      v_open : IsOpen v
      w_open : IsOpen w
      fzv : Membership.mem v (f z)
      xw : Membership.mem w x
      hvw : Disjoint v w
      δ : Real
      δpos : GT.gt δ 0
      hδ : HasSubset.Subset (Metric.ball z δ) (Set.preimage f v)
      n : Nat
      hn : LT.lt (HAdd.hAdd (u n) (Dist.dist (y n) z)) δ
      fsnv : HasSubset.Subset (Set.image f ↑(s n)) v
      this : Membership.mem (closure v) x
      ⊢ False
    -/
    exact disjoint_left.1 (hvw.closure_left w_open) this xw
    /-
      🎉 no goals
    -/


theorem _root_.IsClosed.measurableSet_image_of_continuousOn_injOn
    [TopologicalSpace γ] [PolishSpace γ] {β : Type*} [TopologicalSpace β] [T2Space β]
    [MeasurableSpace β] [OpensMeasurableSpace β] {s : Set γ} (hs : IsClosed s) {f : γ → β}
    (f_cont : ContinuousOn f s) (f_inj : InjOn f s) : MeasurableSet (f '' s) := by
  /-
    γ : Type u_3
    inst✝⁵ : TopologicalSpace γ
    inst✝⁴ : PolishSpace γ
    β : Type u_4
    inst✝³ : TopologicalSpace β
    inst✝² : T2Space β
    inst✝¹ : MeasurableSpace β
    inst✝ : OpensMeasurableSpace β
    s : Set γ
    hs : IsClosed s
    f : γ → β
    f_cont : ContinuousOn f s
    f_inj : Set.InjOn f s
    ⊢ MeasurableSet (Set.image f s)
  -/
  rw [image_eq_range]
  /-
    γ : Type u_3
    inst✝⁵ : TopologicalSpace γ
    inst✝⁴ : PolishSpace γ
    β : Type u_4
    inst✝³ : TopologicalSpace β
    inst✝² : T2Space β
    inst✝¹ : MeasurableSpace β
    inst✝ : OpensMeasurableSpace β
    s : Set γ
    hs : IsClosed s
    f : γ → β
    f_cont : ContinuousOn f s
    f_inj : Set.InjOn f s
    ⊢ MeasurableSet (Set.range fun x => f ↑x)
  -/
  haveI : PolishSpace s := IsClosed.polishSpace hs
  /-
    γ : Type u_3
    inst✝⁵ : TopologicalSpace γ
    inst✝⁴ : PolishSpace γ
    β : Type u_4
    inst✝³ : TopologicalSpace β
    inst✝² : T2Space β
    inst✝¹ : MeasurableSpace β
    inst✝ : OpensMeasurableSpace β
    s : Set γ
    hs : IsClosed s
    f : γ → β
    f_cont : ContinuousOn f s
    f_inj : Set.InjOn f s
    this : PolishSpace ↑s
    ⊢ MeasurableSet (Set.range fun x => f ↑x)
  -/
  apply measurableSet_range_of_continuous_injective
    /-
      case f_cont
      γ : Type u_3
      inst✝⁵ : TopologicalSpace γ
      inst✝⁴ : PolishSpace γ
      β : Type u_4
      inst✝³ : TopologicalSpace β
      inst✝² : T2Space β
      inst✝¹ : MeasurableSpace β
      inst✝ : OpensMeasurableSpace β
      s : Set γ
      hs : IsClosed s
      f : γ → β
      f_cont : ContinuousOn f s
      f_inj : Set.InjOn f s
      this : PolishSpace ↑s
      ⊢ Continuous fun x => f ↑x
    -/
  · rwa [continuousOn_iff_continuous_restrict] at f_cont
    /-
      🎉 no goals
    -/
    /-
      case f_inj
      γ : Type u_3
      inst✝⁵ : TopologicalSpace γ
      inst✝⁴ : PolishSpace γ
      β : Type u_4
      inst✝³ : TopologicalSpace β
      inst✝² : T2Space β
      inst✝¹ : MeasurableSpace β
      inst✝ : OpensMeasurableSpace β
      s : Set γ
      hs : IsClosed s
      f : γ → β
      f_cont : ContinuousOn f s
      f_inj : Set.InjOn f s
      this : PolishSpace ↑s
      ⊢ Function.Injective fun x => f ↑x
    -/
  · rwa [injOn_iff_injective] at f_inj
    /-
      🎉 no goals
    -/


/-- The Lusin-Souslin theorem: if `s` is Borel-measurable in a Polish space, then its image under
a continuous injective map is also Borel-measurable. -/
theorem _root_.MeasurableSet.image_of_continuousOn_injOn [OpensMeasurableSpace β]
    [tγ : TopologicalSpace γ] [PolishSpace γ] [MeasurableSpace γ] [BorelSpace γ]
    (hs : MeasurableSet s)
    (f_cont : ContinuousOn f s) (f_inj : InjOn f s) : MeasurableSet (f '' s) := by
  obtain ⟨t', t't, t'_polish, s_closed, _⟩ :
      ∃ t' : TopologicalSpace γ, t' ≤ tγ ∧ @PolishSpace γ t' ∧ IsClosed[t'] s ∧ IsOpen[t'] s :=
    hs.isClopenable
  exact
    @IsClosed.measurableSet_image_of_continuousOn_injOn γ t' t'_polish β _ _ _ _ s s_closed f
      (f_cont.mono_dom t't) f_inj


/-- The Lusin-Souslin theorem: if `s` is Borel-measurable in a standard Borel space,
then its image under a measurable injective map taking values in a
countably separate measurable space is also Borel-measurable. -/
theorem _root_.MeasurableSet.image_of_measurable_injOn {f : γ → α}
    [MeasurableSpace.CountablySeparated α]
    [MeasurableSpace γ] [StandardBorelSpace γ]
    (hs : MeasurableSet s) (f_meas : Measurable f) (f_inj : InjOn f s) :
    MeasurableSet (f '' s) := by
  /-
    γ : Type u_3
    α : Type u_4
    inst✝³ : MeasurableSpace α
    s : Set γ
    f : γ → α
    inst✝² : MeasurableSpace.CountablySeparated α
    inst✝¹ : MeasurableSpace γ
    inst✝ : StandardBorelSpace γ
    hs : MeasurableSet s
    f_meas : Measurable f
    f_inj : Set.InjOn f s
    ⊢ MeasurableSet (Set.image f s)
  -/
  letI := upgradeStandardBorel γ
  /-
    γ : Type u_3
    α : Type u_4
    inst✝³ : MeasurableSpace α
    s : Set γ
    f : γ → α
    inst✝² : MeasurableSpace.CountablySeparated α
    inst✝¹ : MeasurableSpace γ
    inst✝ : StandardBorelSpace γ
    hs : MeasurableSet s
    f_meas : Measurable f
    f_inj : Set.InjOn f s
    this : UpgradedStandardBorel γ := upgradeStandardBorel γ
    ⊢ MeasurableSet (Set.image f s)
  -/
  let tγ : TopologicalSpace γ := inferInstance
  /-
    γ : Type u_3
    α : Type u_4
    inst✝³ : MeasurableSpace α
    s : Set γ
    f : γ → α
    inst✝² : MeasurableSpace.CountablySeparated α
    inst✝¹ : MeasurableSpace γ
    inst✝ : StandardBorelSpace γ
    hs : MeasurableSet s
    f_meas : Measurable f
    f_inj : Set.InjOn f s
    this : UpgradedStandardBorel γ := upgradeStandardBorel γ
    tγ : TopologicalSpace γ := inferInstance
    ⊢ MeasurableSet (Set.image f s)
  -/
  rcases exists_opensMeasurableSpace_of_countablySeparated α with ⟨τ, _, _, _⟩
  -- for a finer Polish topology, `f` is continuous. Therefore, one may apply the corresponding
  -- result for continuous maps.
  obtain ⟨t', t't, f_cont, t'_polish⟩ :
      ∃ t' : TopologicalSpace γ, t' ≤ tγ ∧ @Continuous γ _ t' _ f ∧ @PolishSpace γ t' :=
    f_meas.exists_continuous
  have M : MeasurableSet[@borel γ t'] s :=
    @Continuous.measurable γ γ t' (@borel γ t')
      (@BorelSpace.opensMeasurable γ t' (@borel γ t') (@BorelSpace.mk _ _ (borel γ) rfl))
      tγ _ _ _ (continuous_id_of_le t't) s hs
  exact
    @MeasurableSet.image_of_continuousOn_injOn γ
      _ _ _ _  s f _ t' t'_polish (@borel γ t') (@BorelSpace.mk _ _ (borel γ) rfl)
      M (@Continuous.continuousOn γ _ t' _ f s f_cont) f_inj


/-- An injective continuous function on a Polish space is a measurable embedding. -/
theorem _root_.Continuous.measurableEmbedding [BorelSpace β]
    [TopologicalSpace γ] [PolishSpace γ] [MeasurableSpace γ] [BorelSpace γ]
    (f_cont : Continuous f) (f_inj : Injective f) :
    MeasurableEmbedding f :=
  { injective := f_inj
    measurable := f_cont.measurable
    measurableSet_image' := fun _u hu =>
      hu.image_of_continuousOn_injOn f_cont.continuousOn f_inj.injOn }


/-- If `s` is Borel-measurable in a Polish space and `f` is continuous injective on `s`, then
the restriction of `f` to `s` is a measurable embedding. -/
theorem _root_.ContinuousOn.measurableEmbedding [BorelSpace β]
    [TopologicalSpace γ] [PolishSpace γ] [MeasurableSpace γ] [BorelSpace γ]
    (hs : MeasurableSet s) (f_cont : ContinuousOn f s)
    (f_inj : InjOn f s) : MeasurableEmbedding (s.restrict f) :=
  { injective := injOn_iff_injective.1 f_inj
    measurable := (continuousOn_iff_continuous_restrict.1 f_cont).measurable
    measurableSet_image' := by
      /-
        γ : Type u_3
        β : Type u_5
        inst✝⁶ : MeasurableSpace β
        tβ : TopologicalSpace β
        inst✝⁵ : T2Space β
        s : Set γ
        f : γ → β
        inst✝⁴ : BorelSpace β
        inst✝³ : TopologicalSpace γ
        inst✝² : PolishSpace γ
        inst✝¹ : MeasurableSpace γ
        inst✝ : BorelSpace γ
        hs : MeasurableSet s
        f_cont : ContinuousOn f s
        f_inj : Set.InjOn f s
        ⊢ ∀ ⦃s_1 : Set ↑s⦄, MeasurableSet s_1 → MeasurableSet (Set.image (s.restrict f …
      -/
      intro u hu
      have A : MeasurableSet (((↑) : s → γ) '' u) :=
        (MeasurableEmbedding.subtype_coe hs).measurableSet_image.2 hu
      have B : MeasurableSet (f '' (((↑) : s → γ) '' u)) :=
        A.image_of_continuousOn_injOn (f_cont.mono (Subtype.coe_image_subset s u))
          (f_inj.mono (Subtype.coe_image_subset s u))
      /-
        γ : Type u_3
        β : Type u_5
        inst✝⁶ : MeasurableSpace β
        tβ : TopologicalSpace β
        inst✝⁵ : T2Space β
        s : Set γ
        f : γ → β
        inst✝⁴ : BorelSpace β
        inst✝³ : TopologicalSpace γ
        inst✝² : PolishSpace γ
        inst✝¹ : MeasurableSpace γ
        inst✝ : BorelSpace γ
        hs : MeasurableSet s
        f_cont : ContinuousOn f s
        f_inj : Set.InjOn f s
        u : Set ↑s
        hu : MeasurableSet u
        A : MeasurableSet (Set.image Subtype.val u)
        B : MeasurableSet (Set.image f (Set.image Subtype.val u))
        ⊢ MeasurableSet (Set.image (s.restrict f) u)
      -/
      rwa [← image_comp] at B }
      /-
        🎉 no goals
      -/


/-- An injective measurable function from a standard Borel space to a
countably separated measurable space is a measurable embedding. -/
theorem _root_.Measurable.measurableEmbedding {f : γ → α}
    [MeasurableSpace.CountablySeparated α]
    [MeasurableSpace γ] [StandardBorelSpace γ]
    (f_meas : Measurable f) (f_inj : Injective f) : MeasurableEmbedding f :=
  { injective := f_inj
    measurable := f_meas
    measurableSet_image' := fun _u hu => hu.image_of_measurable_injOn f_meas f_inj.injOn }


/-- If one Polish topology on a type refines another, they have the same Borel sets. -/
theorem borel_eq_borel_of_le {t t' : TopologicalSpace γ}
    (ht : PolishSpace (h := t)) (ht' : PolishSpace (h := t')) (hle : t ≤ t') :
    @borel _ t = @borel _ t' := by
  /-
    γ : Type u_3
    t t' : TopologicalSpace γ
    ht : PolishSpace γ
    ht' : PolishSpace γ
    hle : LE.le t t'
    ⊢ Eq (borel γ) (borel γ)
  -/
  refine le_antisymm ?_ (borel_anti hle)
  /-
    γ : Type u_3
    t t' : TopologicalSpace γ
    ht : PolishSpace γ
    ht' : PolishSpace γ
    hle : LE.le t t'
    ⊢ LE.le (borel γ) (borel γ)
  -/
  intro s hs
  have e := @Continuous.measurableEmbedding
    _ _ (@borel _ t') t' _ _ (@BorelSpace.mk _ _ (borel γ) rfl)
    t _ (@borel _ t) (@BorelSpace.mk _ t (@borel _ t) rfl) (continuous_id_of_le hle) injective_id
  /-
    γ : Type u_3
    t t' : TopologicalSpace γ
    ht : PolishSpace γ
    ht' : PolishSpace γ
    hle : LE.le t t'
    s : Set γ
    hs : MeasurableSet s
    e : MeasurableEmbedding id
    ⊢ MeasurableSet s
  -/
  convert e.measurableSet_image.2 hs
  /-
    case h.e'_3
    γ : Type u_3
    t t' : TopologicalSpace γ
    ht : PolishSpace γ
    ht' : PolishSpace γ
    hle : LE.le t t'
    s : Set γ
    hs : MeasurableSet s
    e : MeasurableEmbedding id
    ⊢ Eq s (Set.image id s)
  -/
  simp only [id_eq, image_id']
  /-
    🎉 no goals
  -/


/-- In a Polish space, a set is clopenable if and only if it is Borel-measurable. -/
theorem isClopenable_iff_measurableSet
    [tγ : TopologicalSpace γ] [PolishSpace γ] [MeasurableSpace γ] [BorelSpace γ] :
    IsClopenable s ↔ MeasurableSet s := by
  -- we already know that a measurable set is clopenable. Conversely, assume that `s` is clopenable.
  /-
    γ : Type u_3
    s : Set γ
    tγ : TopologicalSpace γ
    inst✝² : PolishSpace γ
    inst✝¹ : MeasurableSpace γ
    inst✝ : BorelSpace γ
    ⊢ Iff (PolishSpace.IsClopenable s) (MeasurableSet s)
  -/
  refine ⟨fun hs => ?_, fun hs => hs.isClopenable⟩
  /-
    γ : Type u_3
    s : Set γ
    tγ : TopologicalSpace γ
    inst✝² : PolishSpace γ
    inst✝¹ : MeasurableSpace γ
    inst✝ : BorelSpace γ
    hs : PolishSpace.IsClopenable s
    ⊢ MeasurableSet s
  -/
  borelize γ
  -- consider a finer topology `t'` in which `s` is open and closed.
  obtain ⟨t', t't, t'_polish, _, s_open⟩ :
    ∃ t' : TopologicalSpace γ, t' ≤ tγ ∧ @PolishSpace γ t' ∧ IsClosed[t'] s ∧ IsOpen[t'] s := hs
  /-
    case intro.intro.intro.intro
    γ : Type u_3
    s : Set γ
    tγ : TopologicalSpace γ
    inst✝¹ : PolishSpace γ
    inst✝ : BorelSpace γ
    this✝ : MeasurableSpace γ := borel γ
    t' : TopologicalSpace γ
    t't : LE.le t' tγ
    t'_polish : PolishSpace γ
    left✝ : IsClosed s
    s_open : IsOpen s
    ⊢ MeasurableSet s
  -/
  rw [← borel_eq_borel_of_le t'_polish _ t't]
    /-
      case intro.intro.intro.intro
      γ : Type u_3
      s : Set γ
      tγ : TopologicalSpace γ
      inst✝¹ : PolishSpace γ
      inst✝ : BorelSpace γ
      this✝ : MeasurableSpace γ := borel γ
      t' : TopologicalSpace γ
      t't : LE.le t' tγ
      t'_polish : PolishSpace γ
      left✝ : IsClosed s
      s_open : IsOpen s
      ⊢ MeasurableSet s
    -/
  · exact MeasurableSpace.measurableSet_generateFrom s_open
    /-
      🎉 no goals
    -/
  /-
    γ : Type u_3
    s : Set γ
    tγ : TopologicalSpace γ
    inst✝¹ : PolishSpace γ
    inst✝ : BorelSpace γ
    this✝ : MeasurableSpace γ := borel γ
    t' : TopologicalSpace γ
    t't : LE.le t' tγ
    t'_polish : PolishSpace γ
    left✝ : IsClosed s
    s_open : IsOpen s
    ⊢ PolishSpace γ
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The set of points for which a sequence of measurable functions converges to a given function
is measurable. -/
@[measurability]
lemma measurableSet_tendsto_fun [MeasurableSpace γ] [Countable ι]
    {l : Filter ι} [l.IsCountablyGenerated]
    [TopologicalSpace γ] [SecondCountableTopology γ] [PseudoMetrizableSpace γ]
    [OpensMeasurableSpace γ]
    {f : ι → β → γ} (hf : ∀ i, Measurable (f i)) {g : β → γ} (hg : Measurable g) :
    MeasurableSet { x | Tendsto (fun n ↦ f n x) l (𝓝 (g x)) } := by
  /-
    ι : Type u_2
    γ : Type u_3
    β : Type u_5
    inst✝⁷ : MeasurableSpace β
    inst✝⁶ : MeasurableSpace γ
    inst✝⁵ : Countable ι
    l : Filter ι
    inst✝⁴ : l.IsCountablyGenerated
    inst✝³ : TopologicalSpace γ
    inst✝² : SecondCountableTopology γ
    inst✝¹ : TopologicalSpace.PseudoMetrizableSpace γ
    inst✝ : OpensMeasurableSpace γ
    f : ι → β → γ
    hf : ∀ (i : ι), Measurable (f i)
    g : β → γ
    hg : Measurable g
    ⊢ MeasurableSet (setOf fun x => Filter.Tendsto (fun n => f n x) l (nhds (g x)))
  -/
  letI := TopologicalSpace.pseudoMetrizableSpacePseudoMetric γ
  /-
    ι : Type u_2
    γ : Type u_3
    β : Type u_5
    inst✝⁷ : MeasurableSpace β
    inst✝⁶ : MeasurableSpace γ
    inst✝⁵ : Countable ι
    l : Filter ι
    inst✝⁴ : l.IsCountablyGenerated
    inst✝³ : TopologicalSpace γ
    inst✝² : SecondCountableTopology γ
    inst✝¹ : TopologicalSpace.PseudoMetrizableSpace γ
    inst✝ : OpensMeasurableSpace γ
    f : ι → β → γ
    hf : ∀ (i : ι), Measurable (f i)
    g : β → γ
    hg : Measurable g
    this : PseudoMetricSpace γ := TopologicalSpace.pseudoMetrizableSpacePseudoMetr …
    ⊢ MeasurableSet (setOf fun x => Filter.Tendsto (fun n => f n x) l (nhds (g x)))
  -/
  simp_rw [tendsto_iff_dist_tendsto_zero (f := fun n ↦ f n _)]
  /-
    ι : Type u_2
    γ : Type u_3
    β : Type u_5
    inst✝⁷ : MeasurableSpace β
    inst✝⁶ : MeasurableSpace γ
    inst✝⁵ : Countable ι
    l : Filter ι
    inst✝⁴ : l.IsCountablyGenerated
    inst✝³ : TopologicalSpace γ
    inst✝² : SecondCountableTopology γ
    inst✝¹ : TopologicalSpace.PseudoMetrizableSpace γ
    inst✝ : OpensMeasurableSpace γ
    f : ι → β → γ
    hf : ∀ (i : ι), Measurable (f i)
    g : β → γ
    hg : Measurable g
    this : PseudoMetricSpace γ := TopologicalSpace.pseudoMetrizableSpacePseudoMetr …
    ⊢ MeasurableSet (setOf fun x => Filter.Tendsto (fun b => Dist.dist (f b x) (g  …
  -/
  exact measurableSet_tendsto (𝓝 0) (fun n ↦ (hf n).dist hg)
  /-
    🎉 no goals
  -/


/-- The set of points for which a measurable sequence of functions converges is measurable. -/
@[measurability]
theorem measurableSet_exists_tendsto [TopologicalSpace γ] [PolishSpace γ] [MeasurableSpace γ]
    [hγ : OpensMeasurableSpace γ] [Countable ι] {l : Filter ι}
    [l.IsCountablyGenerated] {f : ι → β → γ} (hf : ∀ i, Measurable (f i)) :
    MeasurableSet { x | ∃ c, Tendsto (fun n => f n x) l (𝓝 c) } := by
  /-
    ι : Type u_2
    γ : Type u_3
    β : Type u_5
    inst✝⁵ : MeasurableSpace β
    inst✝⁴ : TopologicalSpace γ
    inst✝³ : PolishSpace γ
    inst✝² : MeasurableSpace γ
    hγ : OpensMeasurableSpace γ
    inst✝¹ : Countable ι
    l : Filter ι
    inst✝ : l.IsCountablyGenerated
    f : ι → β → γ
    hf : ∀ (i : ι), Measurable (f i)
    ⊢ MeasurableSet (setOf fun x => Exists fun c => Filter.Tendsto (fun n => f n x …
  -/
  rcases l.eq_or_neBot with rfl | hl
    /-
      case inl
      ι : Type u_2
      γ : Type u_3
      β : Type u_5
      inst✝⁵ : MeasurableSpace β
      inst✝⁴ : TopologicalSpace γ
      inst✝³ : PolishSpace γ
      inst✝² : MeasurableSpace γ
      hγ : OpensMeasurableSpace γ
      inst✝¹ : Countable ι
      f : ι → β → γ
      hf : ∀ (i : ι), Measurable (f i)
      inst✝ : Bot.bot.IsCountablyGenerated
      ⊢ MeasurableSet (setOf fun x => Exists fun c => Filter.Tendsto (fun n => f n x …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    ι : Type u_2
    γ : Type u_3
    β : Type u_5
    inst✝⁵ : MeasurableSpace β
    inst✝⁴ : TopologicalSpace γ
    inst✝³ : PolishSpace γ
    inst✝² : MeasurableSpace γ
    hγ : OpensMeasurableSpace γ
    inst✝¹ : Countable ι
    l : Filter ι
    inst✝ : l.IsCountablyGenerated
    f : ι → β → γ
    hf : ∀ (i : ι), Measurable (f i)
    hl : l.NeBot
    ⊢ MeasurableSet (setOf fun x => Exists fun c => Filter.Tendsto (fun n => f n x …
  -/
  letI := upgradePolishSpace γ
  /-
    case inr
    ι : Type u_2
    γ : Type u_3
    β : Type u_5
    inst✝⁵ : MeasurableSpace β
    inst✝⁴ : TopologicalSpace γ
    inst✝³ : PolishSpace γ
    inst✝² : MeasurableSpace γ
    hγ : OpensMeasurableSpace γ
    inst✝¹ : Countable ι
    l : Filter ι
    inst✝ : l.IsCountablyGenerated
    f : ι → β → γ
    hf : ∀ (i : ι), Measurable (f i)
    hl : l.NeBot
    this : UpgradedPolishSpace γ := upgradePolishSpace γ
    ⊢ MeasurableSet (setOf fun x => Exists fun c => Filter.Tendsto (fun n => f n x …
  -/
  rcases l.exists_antitone_basis with ⟨u, hu⟩
  /-
    case inr.intro
    ι : Type u_2
    γ : Type u_3
    β : Type u_5
    inst✝⁵ : MeasurableSpace β
    inst✝⁴ : TopologicalSpace γ
    inst✝³ : PolishSpace γ
    inst✝² : MeasurableSpace γ
    hγ : OpensMeasurableSpace γ
    inst✝¹ : Countable ι
    l : Filter ι
    inst✝ : l.IsCountablyGenerated
    f : ι → β → γ
    hf : ∀ (i : ι), Measurable (f i)
    hl : l.NeBot
    this : UpgradedPolishSpace γ := upgradePolishSpace γ
    u : Nat → Set ι
    hu : l.HasAntitoneBasis u
    ⊢ MeasurableSet (setOf fun x => Exists fun c => Filter.Tendsto (fun n => f n x …
  -/
  simp_rw [← cauchy_map_iff_exists_tendsto]
  /-
    case inr.intro
    ι : Type u_2
    γ : Type u_3
    β : Type u_5
    inst✝⁵ : MeasurableSpace β
    inst✝⁴ : TopologicalSpace γ
    inst✝³ : PolishSpace γ
    inst✝² : MeasurableSpace γ
    hγ : OpensMeasurableSpace γ
    inst✝¹ : Countable ι
    l : Filter ι
    inst✝ : l.IsCountablyGenerated
    f : ι → β → γ
    hf : ∀ (i : ι), Measurable (f i)
    hl : l.NeBot
    this : UpgradedPolishSpace γ := upgradePolishSpace γ
    u : Nat → Set ι
    hu : l.HasAntitoneBasis u
    ⊢ MeasurableSet (setOf fun x => Cauchy (Filter.map (fun n => f n x) l))
  -/
  change MeasurableSet { x | _ ∧ _ }
  have : ∀ x, (map (f · x) l ×ˢ map (f · x) l).HasAntitoneBasis fun n =>
      ((f · x) '' u n) ×ˢ ((f · x) '' u n) := fun x => (hu.map _).prod (hu.map _)
  simp_rw [and_iff_right (hl.map _),
    Filter.HasBasis.le_basis_iff (this _).toHasBasis Metric.uniformity_basis_dist_inv_nat_succ,
    Set.setOf_forall]
  /-
    case inr.intro
    ι : Type u_2
    γ : Type u_3
    β : Type u_5
    inst✝⁵ : MeasurableSpace β
    inst✝⁴ : TopologicalSpace γ
    inst✝³ : PolishSpace γ
    inst✝² : MeasurableSpace γ
    hγ : OpensMeasurableSpace γ
    inst✝¹ : Countable ι
    l : Filter ι
    inst✝ : l.IsCountablyGenerated
    f : ι → β → γ
    hf : ∀ (i : ι), Measurable (f i)
    hl : l.NeBot
    this✝ : UpgradedPolishSpace γ := upgradePolishSpace γ
    u : Nat → Set ι
    hu : l.HasAntitoneBasis u
    this : ∀ (x : β), (SProd.sprod (Filter.map (fun x_1 => f x_1 x) l) (Filter.map …
    ⊢ MeasurableSet (Set.iInter fun i => Set.iInter fun i_1 => setOf fun x => Exis …
  -/
  refine MeasurableSet.biInter Set.countable_univ fun K _ => ?_
  /-
    case inr.intro
    ι : Type u_2
    γ : Type u_3
    β : Type u_5
    inst✝⁵ : MeasurableSpace β
    inst✝⁴ : TopologicalSpace γ
    inst✝³ : PolishSpace γ
    inst✝² : MeasurableSpace γ
    hγ : OpensMeasurableSpace γ
    inst✝¹ : Countable ι
    l : Filter ι
    inst✝ : l.IsCountablyGenerated
    f : ι → β → γ
    hf : ∀ (i : ι), Measurable (f i)
    hl : l.NeBot
    this✝ : UpgradedPolishSpace γ := upgradePolishSpace γ
    u : Nat → Set ι
    hu : l.HasAntitoneBasis u
    this : ∀ (x : β), (SProd.sprod (Filter.map (fun x_1 => f x_1 x) l) (Filter.map …
    K : Nat
    x✝ : Membership.mem (fun i => True) K
    ⊢ MeasurableSet (setOf fun x => Exists fun i => And True (HasSubset.Subset (SP …
  -/
  simp_rw [Set.setOf_exists, true_and]
  /-
    case inr.intro
    ι : Type u_2
    γ : Type u_3
    β : Type u_5
    inst✝⁵ : MeasurableSpace β
    inst✝⁴ : TopologicalSpace γ
    inst✝³ : PolishSpace γ
    inst✝² : MeasurableSpace γ
    hγ : OpensMeasurableSpace γ
    inst✝¹ : Countable ι
    l : Filter ι
    inst✝ : l.IsCountablyGenerated
    f : ι → β → γ
    hf : ∀ (i : ι), Measurable (f i)
    hl : l.NeBot
    this✝ : UpgradedPolishSpace γ := upgradePolishSpace γ
    u : Nat → Set ι
    hu : l.HasAntitoneBasis u
    this : ∀ (x : β), (SProd.sprod (Filter.map (fun x_1 => f x_1 x) l) (Filter.map …
    K : Nat
    x✝ : Membership.mem (fun i => True) K
    ⊢ MeasurableSet (Set.iUnion fun i => setOf fun x => HasSubset.Subset (SProd.sp …
  -/
  refine MeasurableSet.iUnion fun N => ?_
  /-
    case inr.intro
    ι : Type u_2
    γ : Type u_3
    β : Type u_5
    inst✝⁵ : MeasurableSpace β
    inst✝⁴ : TopologicalSpace γ
    inst✝³ : PolishSpace γ
    inst✝² : MeasurableSpace γ
    hγ : OpensMeasurableSpace γ
    inst✝¹ : Countable ι
    l : Filter ι
    inst✝ : l.IsCountablyGenerated
    f : ι → β → γ
    hf : ∀ (i : ι), Measurable (f i)
    hl : l.NeBot
    this✝ : UpgradedPolishSpace γ := upgradePolishSpace γ
    u : Nat → Set ι
    hu : l.HasAntitoneBasis u
    this : ∀ (x : β), (SProd.sprod (Filter.map (fun x_1 => f x_1 x) l) (Filter.map …
    K : Nat
    x✝ : Membership.mem (fun i => True) K
    N : Nat
    ⊢ MeasurableSet (setOf fun x => HasSubset.Subset (SProd.sprod (Set.image (fun  …
  -/
  simp_rw [prod_image_image_eq, image_subset_iff, prod_subset_iff, Set.setOf_forall]
  exact
    MeasurableSet.biInter (to_countable (u N)) fun i _ =>
      MeasurableSet.biInter (to_countable (u N)) fun j _ =>
        measurableSet_lt (Measurable.dist (hf i) (hf j)) measurable_const


/-- If `s` is a measurable set in a standard Borel space, there is a compatible Polish topology
making `s` clopen. -/
theorem _root_.MeasurableSet.isClopenable' {s : Set α} (hs : MeasurableSet s) :
    ∃ _ : TopologicalSpace α, BorelSpace α ∧ PolishSpace α ∧ IsClosed s ∧ IsOpen s := by
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    inst✝ : StandardBorelSpace α
    s : Set α
    hs : MeasurableSet s
    ⊢ Exists fun x => And (BorelSpace α) (And (PolishSpace α) (And (IsClosed s) (I …
  -/
  letI := upgradeStandardBorel α
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    inst✝ : StandardBorelSpace α
    s : Set α
    hs : MeasurableSet s
    this : UpgradedStandardBorel α := upgradeStandardBorel α
    ⊢ Exists fun x => And (BorelSpace α) (And (PolishSpace α) (And (IsClosed s) (I …
  -/
  obtain ⟨t, hle, ht, s_clopen⟩ := hs.isClopenable
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    inst✝ : StandardBorelSpace α
    s : Set α
    hs : MeasurableSet s
    this : UpgradedStandardBorel α := upgradeStandardBorel α
    t : TopologicalSpace α
    hle : LE.le t UpgradedStandardBorel.toTopologicalSpace
    ht : PolishSpace α
    s_clopen : And (IsClosed s) (IsOpen s)
    ⊢ Exists fun x => And (BorelSpace α) (And (PolishSpace α) (And (IsClosed s) (I …
  -/
  refine ⟨t, ?_, ht, s_clopen⟩
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    inst✝ : StandardBorelSpace α
    s : Set α
    hs : MeasurableSet s
    this : UpgradedStandardBorel α := upgradeStandardBorel α
    t : TopologicalSpace α
    hle : LE.le t UpgradedStandardBorel.toTopologicalSpace
    ht : PolishSpace α
    s_clopen : And (IsClosed s) (IsOpen s)
    ⊢ BorelSpace α
  -/
  constructor
  /-
    case intro.intro.intro.measurable_eq
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    inst✝ : StandardBorelSpace α
    s : Set α
    hs : MeasurableSet s
    this : UpgradedStandardBorel α := upgradeStandardBorel α
    t : TopologicalSpace α
    hle : LE.le t UpgradedStandardBorel.toTopologicalSpace
    ht : PolishSpace α
    s_clopen : And (IsClosed s) (IsOpen s)
    ⊢ Eq inst✝¹ (borel α)
  -/
  rw [eq_borel_upgradeStandardBorel α, borel_eq_borel_of_le ht _ hle]
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    inst✝ : StandardBorelSpace α
    s : Set α
    hs : MeasurableSet s
    this : UpgradedStandardBorel α := upgradeStandardBorel α
    t : TopologicalSpace α
    hle : LE.le t UpgradedStandardBorel.toTopologicalSpace
    ht : PolishSpace α
    s_clopen : And (IsClosed s) (IsOpen s)
    ⊢ PolishSpace α
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- A measurable subspace of a standard Borel space is standard Borel. -/
theorem _root_.MeasurableSet.standardBorel {s : Set α} (hs : MeasurableSet s) :
    StandardBorelSpace s := by
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    inst✝ : StandardBorelSpace α
    s : Set α
    hs : MeasurableSet s
    ⊢ StandardBorelSpace ↑s
  -/
  obtain ⟨_, _, _, s_closed, _⟩ := hs.isClopenable'
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    inst✝ : StandardBorelSpace α
    s : Set α
    hs : MeasurableSet s
    w✝ : TopologicalSpace α
    left✝¹ : BorelSpace α
    left✝ : PolishSpace α
    s_closed : IsClosed s
    right✝ : IsOpen s
    ⊢ StandardBorelSpace ↑s
  -/
  haveI := s_closed.polishSpace
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    inst✝ : StandardBorelSpace α
    s : Set α
    hs : MeasurableSet s
    w✝ : TopologicalSpace α
    left✝¹ : BorelSpace α
    left✝ : PolishSpace α
    s_closed : IsClosed s
    right✝ : IsOpen s
    this : PolishSpace ↑s
    ⊢ StandardBorelSpace ↑s
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- If two standard Borel spaces admit Borel measurable injections to one another,
then they are Borel isomorphic. -/
noncomputable def borelSchroederBernstein {f : α → β} {g : β → α} (fmeas : Measurable f)
    (finj : Function.Injective f) (gmeas : Measurable g) (ginj : Function.Injective g) : α ≃ᵐ β :=
  letI := upgradeStandardBorel α
  letI := upgradeStandardBorel β
  (fmeas.measurableEmbedding finj).schroederBernstein (gmeas.measurableEmbedding ginj)


/-- Any uncountable standard Borel space is Borel isomorphic to the Cantor space `ℕ → Bool`. -/
noncomputable def measurableEquivNatBoolOfNotCountable (h : ¬Countable α) : α ≃ᵐ (ℕ → Bool) := by
  /-
    α : Type u_1
    ι : Type u_2
    β : Type u_3
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    inst✝¹ : StandardBorelSpace α
    inst✝ : StandardBorelSpace β
    h : Not (Countable α)
    ⊢ MeasurableEquiv α (Nat → Bool)
  -/
  apply Nonempty.some
  /-
    case h
    α : Type u_1
    ι : Type u_2
    β : Type u_3
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    inst✝¹ : StandardBorelSpace α
    inst✝ : StandardBorelSpace β
    h : Not (Countable α)
    ⊢ Nonempty (MeasurableEquiv α (Nat → Bool))
  -/
  letI := upgradeStandardBorel α
  obtain ⟨f, -, fcts, finj⟩ :=
    isClosed_univ.exists_nat_bool_injection_of_not_countable (α := α)
      (by rwa [← countable_coe_iff, (Equiv.Set.univ _).countable_iff])
  obtain ⟨g, gmeas, ginj⟩ :=
    MeasurableSpace.measurable_injection_nat_bool_of_countablySeparated α
  /-
    case h.intro.intro.intro.intro.intro
    α : Type u_1
    ι : Type u_2
    β : Type u_3
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    inst✝¹ : StandardBorelSpace α
    inst✝ : StandardBorelSpace β
    h : Not (Countable α)
    this : UpgradedStandardBorel α := upgradeStandardBorel α
    f : (Nat → Bool) → α
    fcts : Continuous f
    finj : Function.Injective f
    g : α → Nat → Bool
    gmeas : Measurable g
    ginj : Function.Injective g
    ⊢ Nonempty (MeasurableEquiv α (Nat → Bool))
  -/
  exact ⟨borelSchroederBernstein gmeas ginj fcts.measurable finj⟩
  /-
    🎉 no goals
  -/


/-- The **Borel Isomorphism Theorem**: Any two uncountable standard Borel spaces are
Borel isomorphic. -/
noncomputable def measurableEquivOfNotCountable (hα : ¬Countable α) (hβ : ¬Countable β) : α ≃ᵐ β :=
  (measurableEquivNatBoolOfNotCountable hα).trans (measurableEquivNatBoolOfNotCountable hβ).symm


/-- The **Borel Isomorphism Theorem**: If two standard Borel spaces have the same cardinality,
they are Borel isomorphic. -/
noncomputable def Equiv.measurableEquiv (e : α ≃ β) : α ≃ᵐ β := by
  /-
    α : Type u_1
    ι : Type u_2
    β : Type u_3
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    inst✝¹ : StandardBorelSpace α
    inst✝ : StandardBorelSpace β
    e : Equiv α β
    ⊢ MeasurableEquiv α β
  -/
  by_cases h : Countable α
    /-
      case pos
      α : Type u_1
      ι : Type u_2
      β : Type u_3
      inst✝³ : MeasurableSpace α
      inst✝² : MeasurableSpace β
      inst✝¹ : StandardBorelSpace α
      inst✝ : StandardBorelSpace β
      e : Equiv α β
      h : Countable α
      ⊢ MeasurableEquiv α β
    -/
  · letI := Countable.of_equiv α e
    /-
      case pos
      α : Type u_1
      ι : Type u_2
      β : Type u_3
      inst✝³ : MeasurableSpace α
      inst✝² : MeasurableSpace β
      inst✝¹ : StandardBorelSpace α
      inst✝ : StandardBorelSpace β
      e : Equiv α β
      h : Countable α
      this : Countable β := Countable.of_equiv α e
      ⊢ MeasurableEquiv α β
    -/
                           /-
                             🎉 no goals
                           -/
    refine ⟨e, ?_, ?_⟩ <;> apply measurable_of_countable
                           /-
                             🎉 no goals
                           -/
  /-
    case neg
    α : Type u_1
    ι : Type u_2
    β : Type u_3
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    inst✝¹ : StandardBorelSpace α
    inst✝ : StandardBorelSpace β
    e : Equiv α β
    h : Not (Countable α)
    ⊢ MeasurableEquiv α β
  -/
  refine measurableEquivOfNotCountable h ?_
  /-
    case neg
    α : Type u_1
    ι : Type u_2
    β : Type u_3
    inst✝³ : MeasurableSpace α
    inst✝² : MeasurableSpace β
    inst✝¹ : StandardBorelSpace α
    inst✝ : StandardBorelSpace β
    e : Equiv α β
    h : Not (Countable α)
    ⊢ Not (Countable β)
  -/
  rwa [e.countable_iff] at h
  /-
    🎉 no goals
  -/


