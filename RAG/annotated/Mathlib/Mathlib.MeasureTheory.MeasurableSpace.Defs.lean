/-- A measurable space is a space equipped with a σ-algebra. -/
@[class] structure MeasurableSpace (α : Type*) where
  /-- Predicate saying that a given set is measurable. Use `MeasurableSet` in the root namespace
  instead. -/
  MeasurableSet' : Set α → Prop
  /-- The empty set is a measurable set. Use `MeasurableSet.empty` instead. -/
  measurableSet_empty : MeasurableSet' ∅
  /-- The complement of a measurable set is a measurable set. Use `MeasurableSet.compl` instead. -/
  measurableSet_compl : ∀ s, MeasurableSet' s → MeasurableSet' sᶜ
  /-- The union of a sequence of measurable sets is a measurable set. Use a more general
  `MeasurableSet.iUnion` instead. -/
  measurableSet_iUnion : ∀ f : ℕ → Set α, (∀ i, MeasurableSet' (f i)) → MeasurableSet' (⋃ i, f i)


instance [h : MeasurableSpace α] : MeasurableSpace αᵒᵈ := h


/-- `MeasurableSet s` means that `s` is measurable (in the ambient measure space on `α`) -/
def MeasurableSet [MeasurableSpace α] (s : Set α) : Prop :=
  ‹MeasurableSpace α›.MeasurableSet' s

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: `scoped[MeasureTheory]` doesn't work for unknown reason

set_option quotPrecheck false in
/-- Notation for `MeasurableSet` with respect to a non-standard σ-algebra. -/
scoped notation "MeasurableSet[" m "]" => @MeasurableSet _ m


@[simp, measurability]
theorem MeasurableSet.empty [MeasurableSpace α] : MeasurableSet (∅ : Set α) :=
  MeasurableSpace.measurableSet_empty _


@[measurability]
protected theorem MeasurableSet.compl : MeasurableSet s → MeasurableSet sᶜ :=
  MeasurableSpace.measurableSet_compl _ s


protected theorem MeasurableSet.of_compl (h : MeasurableSet sᶜ) : MeasurableSet s :=
  compl_compl s ▸ h.compl


@[simp]
theorem MeasurableSet.compl_iff : MeasurableSet sᶜ ↔ MeasurableSet s :=
  ⟨.of_compl, .compl⟩


@[simp, measurability]
protected theorem MeasurableSet.univ : MeasurableSet (univ : Set α) :=
                  /-
                    α : Type u_1
                    m : MeasurableSpace α
                    ⊢ MeasurableSet (HasCompl.compl Set.univ)
                  -/
  .of_compl <| by simp
                  /-
                    🎉 no goals
                  -/


@[nontriviality, measurability]
theorem Subsingleton.measurableSet [Subsingleton α] {s : Set α} : MeasurableSet s :=
  Subsingleton.set_cases MeasurableSet.empty MeasurableSet.univ s


theorem MeasurableSet.congr {s t : Set α} (hs : MeasurableSet s) (h : s = t) : MeasurableSet t := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    s t : Set α
    hs : MeasurableSet s
    h : Eq s t
    ⊢ MeasurableSet t
  -/
  rwa [← h]
  /-
    🎉 no goals
  -/


@[measurability]
protected theorem MeasurableSet.iUnion [Countable ι] ⦃f : ι → Set α⦄
    (h : ∀ b, MeasurableSet (f b)) : MeasurableSet (⋃ b, f b) := by
  /-
    α : Type u_1
    ι : Sort u_6
    m : MeasurableSpace α
    inst✝ : Countable ι
    f : ι → Set α
    h : ∀ (b : ι), MeasurableSet (f b)
    ⊢ MeasurableSet (Set.iUnion fun b => f b)
  -/
  cases isEmpty_or_nonempty ι
    /-
      case inl
      α : Type u_1
      ι : Sort u_6
      m : MeasurableSpace α
      inst✝ : Countable ι
      f : ι → Set α
      h : ∀ (b : ι), MeasurableSet (f b)
      h✝ : IsEmpty ι
      ⊢ MeasurableSet (Set.iUnion fun b => f b)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      ι : Sort u_6
      m : MeasurableSpace α
      inst✝ : Countable ι
      f : ι → Set α
      h : ∀ (b : ι), MeasurableSet (f b)
      h✝ : Nonempty ι
      ⊢ MeasurableSet (Set.iUnion fun b => f b)
    -/
  · rcases exists_surjective_nat ι with ⟨e, he⟩
    /-
      case inr.intro
      α : Type u_1
      ι : Sort u_6
      m : MeasurableSpace α
      inst✝ : Countable ι
      f : ι → Set α
      h : ∀ (b : ι), MeasurableSet (f b)
      h✝ : Nonempty ι
      e : Nat → ι
      he : Function.Surjective e
      ⊢ MeasurableSet (Set.iUnion fun b => f b)
    -/
    rw [← iUnion_congr_of_surjective _ he (fun _ => rfl)]
    /-
      case inr.intro
      α : Type u_1
      ι : Sort u_6
      m : MeasurableSpace α
      inst✝ : Countable ι
      f : ι → Set α
      h : ∀ (b : ι), MeasurableSet (f b)
      h✝ : Nonempty ι
      e : Nat → ι
      he : Function.Surjective e
      ⊢ MeasurableSet (Set.iUnion fun x => f (e x))
    -/
    exact m.measurableSet_iUnion _ fun _ => h _
    /-
      🎉 no goals
    -/


protected theorem MeasurableSet.biUnion {f : β → Set α} {s : Set β} (hs : s.Countable)
    (h : ∀ b ∈ s, MeasurableSet (f b)) : MeasurableSet (⋃ b ∈ s, f b) := by
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    f : β → Set α
    s : Set β
    hs : s.Countable
    h : ∀ (b : β), Membership.mem s b → MeasurableSet (f b)
    ⊢ MeasurableSet (Set.iUnion fun b => Set.iUnion fun h => f b)
  -/
  rw [biUnion_eq_iUnion]
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    f : β → Set α
    s : Set β
    hs : s.Countable
    h : ∀ (b : β), Membership.mem s b → MeasurableSet (f b)
    ⊢ MeasurableSet (Set.iUnion fun x => f ↑x)
  -/
  have := hs.to_subtype
  /-
    α : Type u_1
    β : Type u_2
    m : MeasurableSpace α
    f : β → Set α
    s : Set β
    hs : s.Countable
    h : ∀ (b : β), Membership.mem s b → MeasurableSet (f b)
    this : Countable ↑s
    ⊢ MeasurableSet (Set.iUnion fun x => f ↑x)
  -/
  exact MeasurableSet.iUnion (by simpa using h)
  /-
    🎉 no goals
  -/


theorem Set.Finite.measurableSet_biUnion {f : β → Set α} {s : Set β} (hs : s.Finite)
    (h : ∀ b ∈ s, MeasurableSet (f b)) : MeasurableSet (⋃ b ∈ s, f b) :=
  .biUnion hs.countable h


theorem Finset.measurableSet_biUnion {f : β → Set α} (s : Finset β)
    (h : ∀ b ∈ s, MeasurableSet (f b)) : MeasurableSet (⋃ b ∈ s, f b) :=
  s.finite_toSet.measurableSet_biUnion h


protected theorem MeasurableSet.sUnion {s : Set (Set α)} (hs : s.Countable)
    (h : ∀ t ∈ s, MeasurableSet t) : MeasurableSet (⋃₀ s) := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    s : Set (Set α)
    hs : s.Countable
    h : ∀ (t : Set α), Membership.mem s t → MeasurableSet t
    ⊢ MeasurableSet s.sUnion
  -/
  rw [sUnion_eq_biUnion]
  /-
    α : Type u_1
    m : MeasurableSpace α
    s : Set (Set α)
    hs : s.Countable
    h : ∀ (t : Set α), Membership.mem s t → MeasurableSet t
    ⊢ MeasurableSet (Set.iUnion fun i => Set.iUnion fun x => i)
  -/
  exact .biUnion hs h
  /-
    🎉 no goals
  -/


theorem Set.Finite.measurableSet_sUnion {s : Set (Set α)} (hs : s.Finite)
    (h : ∀ t ∈ s, MeasurableSet t) : MeasurableSet (⋃₀ s) :=
  MeasurableSet.sUnion hs.countable h


@[measurability]
theorem MeasurableSet.iInter [Countable ι] {f : ι → Set α} (h : ∀ b, MeasurableSet (f b)) :
    MeasurableSet (⋂ b, f b) :=
                  /-
                    α : Type u_1
                    ι : Sort u_6
                    m : MeasurableSpace α
                    inst✝ : Countable ι
                    f : ι → Set α
                    h : ∀ (b : ι), MeasurableSet (f b)
                    ⊢ MeasurableSet (HasCompl.compl (Set.iInter fun b => f b))
                  -/
  .of_compl <| by rw [compl_iInter]; exact .iUnion fun b => (h b).compl
                                     /-
                                       🎉 no goals
                                     -/


theorem MeasurableSet.biInter {f : β → Set α} {s : Set β} (hs : s.Countable)
    (h : ∀ b ∈ s, MeasurableSet (f b)) : MeasurableSet (⋂ b ∈ s, f b) :=
                  /-
                    α : Type u_1
                    β : Type u_2
                    m : MeasurableSpace α
                    f : β → Set α
                    s : Set β
                    hs : s.Countable
                    h : ∀ (b : β), Membership.mem s b → MeasurableSet (f b)
                    ⊢ MeasurableSet (HasCompl.compl (Set.iInter fun b => Set.iInter fun h => f b))
                  -/
  .of_compl <| by rw [compl_iInter₂]; exact .biUnion hs fun b hb => (h b hb).compl
                                      /-
                                        🎉 no goals
                                      -/


theorem Set.Finite.measurableSet_biInter {f : β → Set α} {s : Set β} (hs : s.Finite)
    (h : ∀ b ∈ s, MeasurableSet (f b)) : MeasurableSet (⋂ b ∈ s, f b) :=
 .biInter hs.countable h


theorem Finset.measurableSet_biInter {f : β → Set α} (s : Finset β)
    (h : ∀ b ∈ s, MeasurableSet (f b)) : MeasurableSet (⋂ b ∈ s, f b) :=
  s.finite_toSet.measurableSet_biInter h


theorem MeasurableSet.sInter {s : Set (Set α)} (hs : s.Countable) (h : ∀ t ∈ s, MeasurableSet t) :
    MeasurableSet (⋂₀ s) := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    s : Set (Set α)
    hs : s.Countable
    h : ∀ (t : Set α), Membership.mem s t → MeasurableSet t
    ⊢ MeasurableSet s.sInter
  -/
  rw [sInter_eq_biInter]
  /-
    α : Type u_1
    m : MeasurableSpace α
    s : Set (Set α)
    hs : s.Countable
    h : ∀ (t : Set α), Membership.mem s t → MeasurableSet t
    ⊢ MeasurableSet (Set.iInter fun i => Set.iInter fun x => i)
  -/
  exact MeasurableSet.biInter hs h
  /-
    🎉 no goals
  -/


theorem Set.Finite.measurableSet_sInter {s : Set (Set α)} (hs : s.Finite)
    (h : ∀ t ∈ s, MeasurableSet t) : MeasurableSet (⋂₀ s) :=
  MeasurableSet.sInter hs.countable h


@[simp, measurability]
protected theorem MeasurableSet.union {s₁ s₂ : Set α} (h₁ : MeasurableSet s₁)
    (h₂ : MeasurableSet s₂) : MeasurableSet (s₁ ∪ s₂) := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    s₁ s₂ : Set α
    h₁ : MeasurableSet s₁
    h₂ : MeasurableSet s₂
    ⊢ MeasurableSet (Union.union s₁ s₂)
  -/
  rw [union_eq_iUnion]
  /-
    α : Type u_1
    m : MeasurableSpace α
    s₁ s₂ : Set α
    h₁ : MeasurableSet s₁
    h₂ : MeasurableSet s₂
    ⊢ MeasurableSet (Set.iUnion fun b => cond b s₁ s₂)
  -/
  exact .iUnion (Bool.forall_bool.2 ⟨h₂, h₁⟩)
  /-
    🎉 no goals
  -/


@[simp, measurability]
protected theorem MeasurableSet.inter {s₁ s₂ : Set α} (h₁ : MeasurableSet s₁)
    (h₂ : MeasurableSet s₂) : MeasurableSet (s₁ ∩ s₂) := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    s₁ s₂ : Set α
    h₁ : MeasurableSet s₁
    h₂ : MeasurableSet s₂
    ⊢ MeasurableSet (Inter.inter s₁ s₂)
  -/
  rw [inter_eq_compl_compl_union_compl]
  /-
    α : Type u_1
    m : MeasurableSpace α
    s₁ s₂ : Set α
    h₁ : MeasurableSet s₁
    h₂ : MeasurableSet s₂
    ⊢ MeasurableSet (HasCompl.compl (Union.union (HasCompl.compl s₁) (HasCompl.com …
  -/
  exact (h₁.compl.union h₂.compl).compl
  /-
    🎉 no goals
  -/


@[simp, measurability]
protected theorem MeasurableSet.diff {s₁ s₂ : Set α} (h₁ : MeasurableSet s₁)
    (h₂ : MeasurableSet s₂) : MeasurableSet (s₁ \ s₂) :=
  h₁.inter h₂.compl


@[simp, measurability]
protected lemma MeasurableSet.himp {s₁ s₂ : Set α} (h₁ : MeasurableSet s₁) (h₂ : MeasurableSet s₂) :
                                  /-
                                    α : Type u_1
                                    m : MeasurableSpace α
                                    s₁ s₂ : Set α
                                    h₁ : MeasurableSet s₁
                                    h₂ : MeasurableSet s₂
                                    ⊢ MeasurableSet (HImp.himp s₁ s₂)
                                  -/
    MeasurableSet (s₁ ⇨ s₂) := by rw [himp_eq]; exact h₂.union h₁.compl
                                                /-
                                                  🎉 no goals
                                                -/


@[simp, measurability]
protected theorem MeasurableSet.symmDiff {s₁ s₂ : Set α} (h₁ : MeasurableSet s₁)
    (h₂ : MeasurableSet s₂) : MeasurableSet (s₁ ∆ s₂) :=
  (h₁.diff h₂).union (h₂.diff h₁)


@[simp, measurability]
protected lemma MeasurableSet.bihimp {s₁ s₂ : Set α} (h₁ : MeasurableSet s₁)
    (h₂ : MeasurableSet s₂) : MeasurableSet (s₁ ⇔ s₂) := (h₂.himp h₁).inter (h₁.himp h₂)


@[simp, measurability]
protected theorem MeasurableSet.ite {t s₁ s₂ : Set α} (ht : MeasurableSet t)
    (h₁ : MeasurableSet s₁) (h₂ : MeasurableSet s₂) : MeasurableSet (t.ite s₁ s₂) :=
  (h₁.inter ht).union (h₂.diff ht)


open Classical in
theorem MeasurableSet.ite' {s t : Set α} {p : Prop} (hs : p → MeasurableSet s)
    (ht : ¬p → MeasurableSet t) : MeasurableSet (ite p s t) := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    s t : Set α
    p : Prop
    hs : p → MeasurableSet s
    ht : Not p → MeasurableSet t
    ⊢ MeasurableSet (ite p s t)
  -/
  split_ifs with h
  /-
    case pos
    α : Type u_1
    m : MeasurableSpace α
    s t : Set α
    p : Prop
    hs : p → MeasurableSet s
    ht : Not p → MeasurableSet t
    h : p
    ⊢ MeasurableSet s
  -/
  exacts [hs h, ht h]
  /-
    🎉 no goals
  -/


@[simp, measurability]
protected theorem MeasurableSet.cond {s₁ s₂ : Set α} (h₁ : MeasurableSet s₁)
    (h₂ : MeasurableSet s₂) {i : Bool} : MeasurableSet (cond i s₁ s₂) := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    s₁ s₂ : Set α
    h₁ : MeasurableSet s₁
    h₂ : MeasurableSet s₂
    i : Bool
    ⊢ MeasurableSet (cond i s₁ s₂)
  -/
  cases i
  /-
    case false
    α : Type u_1
    m : MeasurableSpace α
    s₁ s₂ : Set α
    h₁ : MeasurableSet s₁
    h₂ : MeasurableSet s₂
    ⊢ MeasurableSet (cond Bool.false s₁ s₂)
  -/
  exacts [h₂, h₁]
  /-
    🎉 no goals
  -/


@[simp, measurability]
protected theorem MeasurableSet.disjointed {f : ℕ → Set α} (h : ∀ i, MeasurableSet (f i)) (n) :
    MeasurableSet (disjointed f n) :=
  disjointedRec (fun _ _ ht => MeasurableSet.diff ht <| h _) (h n)


protected theorem MeasurableSet.const (p : Prop) : MeasurableSet { _a : α | p } := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    p : Prop
    ⊢ MeasurableSet (setOf fun _a => p)
  -/
                 /-
                   🎉 no goals
                 -/
  by_cases p <;> simp [*]
                 /-
                   🎉 no goals
                 -/


/-- Every set has a measurable superset. Declare this as local instance as needed. -/
theorem nonempty_measurable_superset (s : Set α) : Nonempty { t // s ⊆ t ∧ MeasurableSet t } :=
  ⟨⟨univ, subset_univ s, MeasurableSet.univ⟩⟩


theorem MeasurableSpace.measurableSet_injective : Injective (@MeasurableSet α)
                                        /-
                                          α : Type u_1
                                          MeasurableSet'✝¹ : Set α → Prop
                                          measurableSet_empty✝¹ : MeasurableSet'✝¹ EmptyCollection.emptyCollection
                                          measurableSet_compl✝¹ : ∀ (s : Set α), MeasurableSet'✝¹ s → MeasurableSet'✝¹ ( …
                                          measurableSet_iUnion✝¹ : ∀ (f : Nat → Set α), (∀ (i : Nat), MeasurableSet'✝¹ ( …
                                          MeasurableSet'✝ : Set α → Prop
                                          measurableSet_empty✝ : MeasurableSet'✝ EmptyCollection.emptyCollection
                                          measurableSet_compl✝ : ∀ (s : Set α), MeasurableSet'✝ s → MeasurableSet'✝ (Has …
                                          measurableSet_iUnion✝ : ∀ (f : Nat → Set α), (∀ (i : Nat), MeasurableSet'✝ (f  …
                                          x✝ : Eq MeasurableSet MeasurableSet
                                          ⊢ Eq { MeasurableSet' := MeasurableSet'✝¹, measurableSet_empty := measurableSe …
                                        -/
  | ⟨_, _, _, _⟩, ⟨_, _, _, _⟩, _ => by congr
                                        /-
                                          🎉 no goals
                                        -/


@[ext]
theorem MeasurableSpace.ext {m₁ m₂ : MeasurableSpace α}
    (h : ∀ s : Set α, MeasurableSet[m₁] s ↔ MeasurableSet[m₂] s) : m₁ = m₂ :=
  measurableSet_injective <| funext fun s => propext (h s)


/-- A typeclass mixin for `MeasurableSpace`s such that each singleton is measurable. -/
class MeasurableSingletonClass (α : Type*) [MeasurableSpace α] : Prop where
  /-- A singleton is a measurable set. -/
  measurableSet_singleton : ∀ x, MeasurableSet ({x} : Set α)


@[simp]
lemma MeasurableSet.singleton [MeasurableSpace α] [MeasurableSingletonClass α] (a : α) :
    MeasurableSet {a} :=
  measurableSet_singleton a


@[measurability]
theorem measurableSet_eq {a : α} : MeasurableSet { x | x = a } := .singleton a


@[measurability]
protected theorem MeasurableSet.insert {s : Set α} (hs : MeasurableSet s) (a : α) :
    MeasurableSet (insert a s) :=
  .union (.singleton a) hs


@[simp]
theorem measurableSet_insert {a : α} {s : Set α} :
    MeasurableSet (insert a s) ↔ MeasurableSet s := by
  classical
  exact ⟨fun h =>
    if ha : a ∈ s then by rwa [← insert_eq_of_mem ha]
    else insert_diff_self_of_not_mem ha ▸ h.diff (.singleton _),
    fun h => h.insert a⟩


theorem Set.Subsingleton.measurableSet {s : Set α} (hs : s.Subsingleton) : MeasurableSet s :=
  hs.induction_on .empty .singleton


theorem Set.Finite.measurableSet {s : Set α} (hs : s.Finite) : MeasurableSet s :=
  Finite.induction_on hs MeasurableSet.empty fun _ _ hsm => hsm.insert _


@[measurability]
protected theorem Finset.measurableSet (s : Finset α) : MeasurableSet (↑s : Set α) :=
  s.finite_toSet.measurableSet


theorem Set.Countable.measurableSet {s : Set α} (hs : s.Countable) : MeasurableSet s := by
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSingletonClass α
    s : Set α
    hs : s.Countable
    ⊢ MeasurableSet s
  -/
  rw [← biUnion_of_singleton s]
  /-
    α : Type u_1
    inst✝¹ : MeasurableSpace α
    inst✝ : MeasurableSingletonClass α
    s : Set α
    hs : s.Countable
    ⊢ MeasurableSet (Set.iUnion fun x => Set.iUnion fun h => Singleton.singleton x)
  -/
  exact .biUnion hs fun b _ => .singleton b
  /-
    🎉 no goals
  -/


/-- Copy of a `MeasurableSpace` with a new `MeasurableSet` equal to the old one. Useful to fix
definitional equalities. -/
protected def copy (m : MeasurableSpace α) (p : Set α → Prop) (h : ∀ s, p s ↔ MeasurableSet[m] s) :
    MeasurableSpace α where
  MeasurableSet' := p
                            /-
                              α : Type u_1
                              β : Type u_2
                              γ : Type u_3
                              δ : Type u_4
                              δ' : Type u_5
                              ι : Sort u_6
                              s t u : Set α
                              m : MeasurableSpace α
                              p : Set α → Prop
                              h : ∀ (s : Set α), Iff (p s) (MeasurableSet s)
                              ⊢ p EmptyCollection.emptyCollection
                            -/
  measurableSet_empty := by simpa only [h] using m.measurableSet_empty
                            /-
                              🎉 no goals
                            -/
                            /-
                              α : Type u_1
                              β : Type u_2
                              γ : Type u_3
                              δ : Type u_4
                              δ' : Type u_5
                              ι : Sort u_6
                              s t u : Set α
                              m : MeasurableSpace α
                              p : Set α → Prop
                              h : ∀ (s : Set α), Iff (p s) (MeasurableSet s)
                              ⊢ ∀ (s : Set α), p s → p (HasCompl.compl s)
                            -/
  measurableSet_compl := by simpa only [h] using m.measurableSet_compl
                            /-
                              🎉 no goals
                            -/
                             /-
                               α : Type u_1
                               β : Type u_2
                               γ : Type u_3
                               δ : Type u_4
                               δ' : Type u_5
                               ι : Sort u_6
                               s t u : Set α
                               m : MeasurableSpace α
                               p : Set α → Prop
                               h : ∀ (s : Set α), Iff (p s) (MeasurableSet s)
                               ⊢ ∀ (f : Nat → Set α), (∀ (i : Nat), p (f i)) → p (Set.iUnion fun i => f i)
                             -/
  measurableSet_iUnion := by simpa only [h] using m.measurableSet_iUnion
                             /-
                               🎉 no goals
                             -/


lemma measurableSet_copy {m : MeasurableSpace α} {p : Set α → Prop}
    (h : ∀ s, p s ↔ MeasurableSet[m] s) {s} : MeasurableSet[.copy m p h] s ↔ p s :=
  Iff.rfl


lemma copy_eq {m : MeasurableSpace α} {p : Set α → Prop} (h : ∀ s, p s ↔ MeasurableSet[m] s) :
    m.copy p h = m :=
  ext h


instance : LE (MeasurableSpace α) where le m₁ m₂ := ∀ s, MeasurableSet[m₁] s → MeasurableSet[m₂] s


theorem le_def {α} {a b : MeasurableSpace α} : a ≤ b ↔ a.MeasurableSet' ≤ b.MeasurableSet' :=
  Iff.rfl


instance : PartialOrder (MeasurableSpace α) :=
  { PartialOrder.lift (@MeasurableSet α) measurableSet_injective with
    le := LE.le
    lt := fun m₁ m₂ => m₁ ≤ m₂ ∧ ¬m₂ ≤ m₁ }


/-- The smallest σ-algebra containing a collection `s` of basic sets -/
inductive GenerateMeasurable (s : Set (Set α)) : Set α → Prop
  | protected basic : ∀ u ∈ s, GenerateMeasurable s u
  | protected empty : GenerateMeasurable s ∅
  | protected compl : ∀ t, GenerateMeasurable s t → GenerateMeasurable s tᶜ
  | protected iUnion : ∀ f : ℕ → Set α, (∀ n, GenerateMeasurable s (f n)) →
      GenerateMeasurable s (⋃ i, f i)


/-- Construct the smallest measure space containing a collection of basic sets -/
def generateFrom (s : Set (Set α)) : MeasurableSpace α where
  MeasurableSet' := GenerateMeasurable s
  measurableSet_empty := .empty
  measurableSet_compl := .compl
  measurableSet_iUnion := .iUnion


theorem measurableSet_generateFrom {s : Set (Set α)} {t : Set α} (ht : t ∈ s) :
    MeasurableSet[generateFrom s] t :=
  .basic t ht


@[elab_as_elim]
theorem generateFrom_induction (C : Set (Set α))
    (p : ∀ s : Set α, MeasurableSet[generateFrom C] s → Prop) (hC : ∀ t ∈ C, ∀ ht, p t ht)
    (empty : p ∅ (measurableSet_empty _)) (compl : ∀ t ht, p t ht → p tᶜ ht.compl)
    (iUnion : ∀ (s : ℕ → Set α) (hs : ∀ n, MeasurableSet[generateFrom C] (s n)),
      (∀ n, p (s n) (hs n)) → p (⋃ i, s i) (.iUnion hs)) (s : Set α)
    (hs : MeasurableSet[generateFrom C] s) : p s hs := by
  /-
    α : Type u_1
    C : Set (Set α)
    p : (s : Set α) → MeasurableSet s → Prop
    hC : ∀ (t : Set α), Membership.mem C t → ∀ (ht : MeasurableSet t), p t ht
    empty : p EmptyCollection.emptyCollection ⋯
    compl : ∀ (t : Set α) (ht : MeasurableSet t), p t ht → p (HasCompl.compl t) ⋯
    iUnion : ∀ (s : Nat → Set α) (hs : ∀ (n : Nat), MeasurableSet (s n)), (∀ (n :  …
    s : Set α
    hs : MeasurableSet s
    ⊢ p s hs
  -/
  induction hs
  /-
    case basic
    α : Type u_1
    C : Set (Set α)
    p : (s : Set α) → MeasurableSet s → Prop
    hC : ∀ (t : Set α), Membership.mem C t → ∀ (ht : MeasurableSet t), p t ht
    empty : p EmptyCollection.emptyCollection ⋯
    compl : ∀ (t : Set α) (ht : MeasurableSet t), p t ht → p (HasCompl.compl t) ⋯
    iUnion : ∀ (s : Nat → Set α) (hs : ∀ (n : Nat), MeasurableSet (s n)), (∀ (n :  …
    s u✝ : Set α
    a✝ : Membership.mem C u✝
    ⊢ p u✝ ⋯
  -/
  exacts [hC _ ‹_› _, empty, compl _ ‹_› ‹_›, iUnion ‹_› ‹_› ‹_›]
  /-
    🎉 no goals
  -/


theorem generateFrom_le {s : Set (Set α)} {m : MeasurableSpace α}
    (h : ∀ t ∈ s, MeasurableSet[m] t) : generateFrom s ≤ m :=
  fun t (ht : GenerateMeasurable s t) =>
  ht.recOn h .empty (fun _ _ => .compl) fun _ _ hf => .iUnion hf


theorem generateFrom_le_iff {s : Set (Set α)} (m : MeasurableSpace α) :
    generateFrom s ≤ m ↔ s ⊆ { t | MeasurableSet[m] t } :=
  Iff.intro (fun h _ hu => h _ <| measurableSet_generateFrom hu) fun h => generateFrom_le h


@[simp]
theorem generateFrom_measurableSet [MeasurableSpace α] :
    generateFrom { s : Set α | MeasurableSet s } = ‹_› :=
  le_antisymm (generateFrom_le fun _ => id) fun _ => measurableSet_generateFrom


theorem forall_generateFrom_mem_iff_mem_iff {S : Set (Set α)} {x y : α} :
    (∀ s, MeasurableSet[generateFrom S] s → (x ∈ s ↔ y ∈ s)) ↔ (∀ s ∈ S, x ∈ s ↔ y ∈ s) := by
  /-
    α : Type u_1
    S : Set (Set α)
    x y : α
    ⊢ Iff (∀ (s : Set α), MeasurableSet s → Iff (Membership.mem s x) (Membership.m …
  -/
  refine ⟨fun H s hs ↦ H s (.basic s hs), fun H s ↦ ?_⟩
  /-
    α : Type u_1
    S : Set (Set α)
    x y : α
    H : ∀ (s : Set α), Membership.mem S s → Iff (Membership.mem s x) (Membership.m …
    s : Set α
    ⊢ MeasurableSet s → Iff (Membership.mem s x) (Membership.mem s y)
  -/
  apply generateFrom_induction
    /-
      case hC
      α : Type u_1
      S : Set (Set α)
      x y : α
      H : ∀ (s : Set α), Membership.mem S s → Iff (Membership.mem s x) (Membership.m …
      s : Set α
      ⊢ ∀ (t : Set α), Membership.mem S t → MeasurableSet t → Iff (Membership.mem t  …
    -/
  · exact fun s hs _ ↦ H s hs
    /-
      🎉 no goals
    -/
    /-
      case empty
      α : Type u_1
      S : Set (Set α)
      x y : α
      H : ∀ (s : Set α), Membership.mem S s → Iff (Membership.mem s x) (Membership.m …
      s : Set α
      ⊢ Iff (Membership.mem EmptyCollection.emptyCollection x) (Membership.mem Empty …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case compl
      α : Type u_1
      S : Set (Set α)
      x y : α
      H : ∀ (s : Set α), Membership.mem S s → Iff (Membership.mem s x) (Membership.m …
      s : Set α
      ⊢ ∀ (t : Set α), MeasurableSet t → Iff (Membership.mem t x) (Membership.mem t  …
    -/
  · exact fun _ _ ↦ Iff.not
    /-
      🎉 no goals
    -/
    /-
      case iUnion
      α : Type u_1
      S : Set (Set α)
      x y : α
      H : ∀ (s : Set α), Membership.mem S s → Iff (Membership.mem s x) (Membership.m …
      s : Set α
      ⊢ ∀ (s : Nat → Set α), (∀ (n : Nat), MeasurableSet (s n)) → (∀ (n : Nat), Iff  …
    -/
  · intro f _ hf
    /-
      case iUnion
      α : Type u_1
      S : Set (Set α)
      x y : α
      H : ∀ (s : Set α), Membership.mem S s → Iff (Membership.mem s x) (Membership.m …
      s : Set α
      f : Nat → Set α
      hs✝ : ∀ (n : Nat), MeasurableSet (f n)
      hf : ∀ (n : Nat), Iff (Membership.mem (f n) x) (Membership.mem (f n) y)
      ⊢ Iff (Membership.mem (Set.iUnion fun i => f i) x) (Membership.mem (Set.iUnion …
    -/
    simp only [mem_iUnion, hf]
    /-
      🎉 no goals
    -/


/-- If `g` is a collection of subsets of `α` such that the `σ`-algebra generated from `g` contains
the same sets as `g`, then `g` was already a `σ`-algebra. -/
protected def mkOfClosure (g : Set (Set α)) (hg : { t | MeasurableSet[generateFrom g] t } = g) :
    MeasurableSpace α :=
  (generateFrom g).copy (· ∈ g) <| Set.ext_iff.1 hg.symm


theorem mkOfClosure_sets {s : Set (Set α)} {hs : { t | MeasurableSet[generateFrom s] t } = s} :
    MeasurableSpace.mkOfClosure s hs = generateFrom s :=
  copy_eq _


/-- We get a Galois insertion between `σ`-algebras on `α` and `Set (Set α)` by using `generate_from`
  on one side and the collection of measurable sets on the other side. -/
def giGenerateFrom : GaloisInsertion (@generateFrom α) fun m => { t | MeasurableSet[m] t } where
  gc _ := generateFrom_le_iff
  le_l_u _ _ := measurableSet_generateFrom
  choice g hg := MeasurableSpace.mkOfClosure g <| le_antisymm hg <| (generateFrom_le_iff _).1 le_rfl
  choice_eq _ _ := mkOfClosure_sets


instance : CompleteLattice (MeasurableSpace α) :=
  giGenerateFrom.liftCompleteLattice


instance : Inhabited (MeasurableSpace α) := ⟨⊤⟩


@[mono]
theorem generateFrom_mono {s t : Set (Set α)} (h : s ⊆ t) : generateFrom s ≤ generateFrom t :=
  giGenerateFrom.gc.monotone_l h


theorem generateFrom_sup_generateFrom {s t : Set (Set α)} :
    generateFrom s ⊔ generateFrom t = generateFrom (s ∪ t) :=
  (@giGenerateFrom α).gc.l_sup.symm


lemma iSup_generateFrom (s : ι → Set (Set α)) :
    ⨆ i, generateFrom (s i) = generateFrom (⋃ i, s i) :=
  (@MeasurableSpace.giGenerateFrom α).gc.l_iSup.symm


@[simp]
lemma generateFrom_empty : generateFrom (∅ : Set (Set α)) = ⊥ :=
                                     /-
                                       α : Type u_1
                                       ⊢ ∀ (t : Set α), Membership.mem EmptyCollection.emptyCollection t → Measurable …
                                     -/
  le_bot_iff.mp (generateFrom_le (by simp))
                                     /-
                                       🎉 no goals
                                     -/


theorem generateFrom_singleton_empty : generateFrom {∅} = (⊥ : MeasurableSpace α) :=
                                      /-
                                        α : Type u_1
                                        ⊢ ∀ (t : Set α), Membership.mem (Singleton.singleton EmptyCollection.emptyColl …
                                      -/
  bot_unique <| generateFrom_le <| by simp [@MeasurableSet.empty α ⊥]
                                      /-
                                        🎉 no goals
                                      -/


theorem generateFrom_singleton_univ : generateFrom {Set.univ} = (⊥ : MeasurableSpace α) :=
                                      /-
                                        α : Type u_1
                                        ⊢ ∀ (t : Set α), Membership.mem (Singleton.singleton Set.univ) t → MeasurableS …
                                      -/
  bot_unique <| generateFrom_le <| by simp
                                      /-
                                        🎉 no goals
                                      -/


@[simp]
theorem generateFrom_insert_univ (S : Set (Set α)) :
    generateFrom (insert Set.univ S) = generateFrom S := by
  /-
    α : Type u_1
    S : Set (Set α)
    ⊢ Eq (MeasurableSpace.generateFrom (Insert.insert Set.univ S)) (MeasurableSpac …
  -/
  rw [insert_eq, ← generateFrom_sup_generateFrom, generateFrom_singleton_univ, bot_sup_eq]
  /-
    🎉 no goals
  -/


@[simp]
theorem generateFrom_insert_empty (S : Set (Set α)) :
    generateFrom (insert ∅ S) = generateFrom S := by
  /-
    α : Type u_1
    S : Set (Set α)
    ⊢ Eq (MeasurableSpace.generateFrom (Insert.insert EmptyCollection.emptyCollect …
  -/
  rw [insert_eq, ← generateFrom_sup_generateFrom, generateFrom_singleton_empty, bot_sup_eq]
  /-
    🎉 no goals
  -/


theorem measurableSet_bot_iff {s : Set α} : MeasurableSet[⊥] s ↔ s = ∅ ∨ s = univ :=
  let b : MeasurableSpace α :=
    { MeasurableSet' := fun s => s = ∅ ∨ s = univ
      measurableSet_empty := Or.inl rfl
                                /-
                                  α : Type u_1
                                  s : Set α
                                  ⊢ ∀ (s : Set α), (fun s => Or (Eq s EmptyCollection.emptyCollection) (Eq s Set …
                                -/
      measurableSet_compl := by simp +contextual [or_imp]
                                /-
                                  🎉 no goals
                                -/
      measurableSet_iUnion := fun _ hf => sUnion_mem_empty_univ (forall_mem_range.2 hf) }
  have : b = ⊥ :=
    bot_unique fun _ hs =>
      hs.elim (fun s => s.symm ▸ @measurableSet_empty _ ⊥) fun s =>
        s.symm ▸ @MeasurableSet.univ _ ⊥
  this ▸ Iff.rfl


@[simp, measurability] theorem measurableSet_top {s : Set α} : MeasurableSet[⊤] s := trivial


@[simp, nolint simpNF] -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: `simpNF` claims that
-- this lemma doesn't simplify LHS
theorem measurableSet_inf {m₁ m₂ : MeasurableSpace α} {s : Set α} :
    MeasurableSet[m₁ ⊓ m₂] s ↔ MeasurableSet[m₁] s ∧ MeasurableSet[m₂] s :=
  Iff.rfl


@[simp]
theorem measurableSet_sInf {ms : Set (MeasurableSpace α)} {s : Set α} :
    MeasurableSet[sInf ms] s ↔ ∀ m ∈ ms, MeasurableSet[m] s :=
                       /-
                         α : Type u_1
                         ms : Set (MeasurableSpace α)
                         s : Set α
                         ⊢ Iff (Membership.mem (Set.image (fun m => setOf fun t => MeasurableSet t) ms) …
                       -/
  show s ∈ ⋂₀ _ ↔ _ by simp
                       /-
                         🎉 no goals
                       -/


theorem measurableSet_iInf {ι} {m : ι → MeasurableSpace α} {s : Set α} :
    MeasurableSet[iInf m] s ↔ ∀ i, MeasurableSet[m i] s := by
  /-
    α : Type u_1
    ι : Sort u_7
    m : ι → MeasurableSpace α
    s : Set α
    ⊢ Iff (MeasurableSet s) (∀ (i : ι), MeasurableSet s)
  -/
  rw [iInf, measurableSet_sInf, forall_mem_range]
  /-
    🎉 no goals
  -/


theorem measurableSet_sup {m₁ m₂ : MeasurableSpace α} {s : Set α} :
    MeasurableSet[m₁ ⊔ m₂] s ↔ GenerateMeasurable (MeasurableSet[m₁] ∪ MeasurableSet[m₂]) s :=
  Iff.rfl


theorem measurableSet_sSup {ms : Set (MeasurableSpace α)} {s : Set α} :
    MeasurableSet[sSup ms] s ↔
      GenerateMeasurable { s : Set α | ∃ m ∈ ms, MeasurableSet[m] s } s := by
  /-
    α : Type u_1
    ms : Set (MeasurableSpace α)
    s : Set α
    ⊢ Iff (MeasurableSet s) (MeasurableSpace.GenerateMeasurable (setOf fun s => Ex …
  -/
  change GenerateMeasurable (⋃₀ _) _ ↔ _
  /-
    α : Type u_1
    ms : Set (MeasurableSpace α)
    s : Set α
    ⊢ Iff (MeasurableSpace.GenerateMeasurable (Set.image (fun m => setOf fun t =>  …
  -/
  simp [← setOf_exists]
  /-
    🎉 no goals
  -/


theorem measurableSet_iSup {ι} {m : ι → MeasurableSpace α} {s : Set α} :
    MeasurableSet[iSup m] s ↔ GenerateMeasurable { s : Set α | ∃ i, MeasurableSet[m i] s } s := by
  /-
    α : Type u_1
    ι : Sort u_7
    m : ι → MeasurableSpace α
    s : Set α
    ⊢ Iff (MeasurableSet s) (MeasurableSpace.GenerateMeasurable (setOf fun s => Ex …
  -/
  simp only [iSup, measurableSet_sSup, exists_range_iff]
  /-
    🎉 no goals
  -/


theorem measurableSpace_iSup_eq (m : ι → MeasurableSpace α) :
    ⨆ n, m n = generateFrom { s | ∃ n, MeasurableSet[m n] s } := by
  /-
    α : Type u_1
    ι : Sort u_6
    m : ι → MeasurableSpace α
    ⊢ Eq (iSup fun n => m n) (MeasurableSpace.generateFrom (setOf fun s => Exists  …
  -/
  ext s
  /-
    case h
    α : Type u_1
    ι : Sort u_6
    m : ι → MeasurableSpace α
    s : Set α
    ⊢ Iff (MeasurableSet s) (MeasurableSet s)
  -/
  rw [measurableSet_iSup]
  /-
    case h
    α : Type u_1
    ι : Sort u_6
    m : ι → MeasurableSpace α
    s : Set α
    ⊢ Iff (MeasurableSpace.GenerateMeasurable (setOf fun s => Exists fun i => Meas …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem generateFrom_iUnion_measurableSet (m : ι → MeasurableSpace α) :
    generateFrom (⋃ n, { t | MeasurableSet[m n] t }) = ⨆ n, m n :=
  (@giGenerateFrom α).l_iSup_u m


/-- A function `f` between measurable spaces is measurable if the preimage of every
  measurable set is measurable. -/
@[fun_prop]
def Measurable [MeasurableSpace α] [MeasurableSpace β] (f : α → β) : Prop :=
  ∀ ⦃t : Set β⦄, MeasurableSet t → MeasurableSet (f ⁻¹' t)


set_option quotPrecheck false in
/-- Notation for `Measurable` with respect to a non-standard σ-algebra in the domain. -/
scoped notation "Measurable[" m "]" => @Measurable _ _ m _

/-- Notation for `Measurable` with respect to a non-standard σ-algebra in the domain and codomain.
-/
scoped notation "Measurable[" mα ", " mβ "]" => @Measurable _ _ mα mβ


@[measurability]
theorem measurable_id {_ : MeasurableSpace α} : Measurable (@id α) := fun _ => id


@[fun_prop, measurability]
theorem measurable_id' {_ : MeasurableSpace α} : Measurable fun a : α => a := measurable_id


protected theorem Measurable.comp {_ : MeasurableSpace α} {_ : MeasurableSpace β}
    {_ : MeasurableSpace γ} {g : β → γ} {f : α → β} (hg : Measurable g) (hf : Measurable f) :
    Measurable (g ∘ f) :=
  fun _ h => hf (hg h)

-- This is needed due to reducibility issues with the `measurability` tactic.

@[fun_prop, aesop safe 50 (rule_sets := [Measurable])]
protected theorem Measurable.comp' {_ : MeasurableSpace α} {_ : MeasurableSpace β}
    {_ : MeasurableSpace γ} {g : β → γ} {f : α → β} (hg : Measurable g) (hf : Measurable f) :
    Measurable (fun x => g (f x)) := Measurable.comp hg hf


@[simp, fun_prop, measurability]
theorem measurable_const {_ : MeasurableSpace α} {_ : MeasurableSpace β} {a : α} :
    Measurable fun _ : β => a := fun s _ => .const (a ∈ s)


theorem Measurable.le {α} {m m0 : MeasurableSpace α} {_ : MeasurableSpace β} (hm : m ≤ m0)
    {f : α → β} (hf : Measurable[m] f) : Measurable[m0] f := fun _ hs => hm _ (hf hs)


/-- A typeclass mixin for `MeasurableSpace`s such that all sets are measurable. -/
class DiscreteMeasurableSpace (α : Type*) [MeasurableSpace α] : Prop where
  /-- Do not use this. Use `MeasurableSet.of_discrete` instead. -/
  forall_measurableSet : ∀ s : Set α, MeasurableSet s


instance : @DiscreteMeasurableSpace α ⊤ :=
  @DiscreteMeasurableSpace.mk _ (_) fun _ ↦ MeasurableSpace.measurableSet_top

-- See note [lower instance priority]

instance (priority := 100) MeasurableSingletonClass.toDiscreteMeasurableSpace [MeasurableSpace α]
    [MeasurableSingletonClass α] [Countable α] : DiscreteMeasurableSpace α where
  forall_measurableSet _ := (Set.to_countable _).measurableSet


@[measurability] lemma MeasurableSet.of_discrete : MeasurableSet s :=
  DiscreteMeasurableSpace.forall_measurableSet _


@[measurability, fun_prop] lemma Measurable.of_discrete : Measurable f := fun _ _ ↦ .of_discrete


@[deprecated MeasurableSet.of_discrete (since := "2024-08-25")]
lemma measurableSet_discrete (s : Set α) : MeasurableSet s := .of_discrete


@[deprecated Measurable.of_discrete (since := "2024-08-25")]
lemma measurable_discrete (f : α → β) : Measurable f := .of_discrete


/-- Warning: Creates a typeclass loop with `MeasurableSingletonClass.toDiscreteMeasurableSpace`.
To be monitored. -/
-- See note [lower instance priority]
instance (priority := 100) DiscreteMeasurableSpace.toMeasurableSingletonClass :
    MeasurableSingletonClass α where
  measurableSet_singleton _ := .of_discrete


