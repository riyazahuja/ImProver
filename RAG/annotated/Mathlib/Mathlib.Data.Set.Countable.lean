/-- A set `s` is countable if the corresponding subtype is countable,
i.e., there exists an injective map `f : s → ℕ`.

Note that this is an abbreviation, so `hs : Set.Countable s` in the proof context
is the same as an instance `Countable s`.
For a constructive version, see `Encodable`.
-/
protected def Countable (s : Set α) : Prop := Countable s


@[simp]
theorem countable_coe_iff {s : Set α} : Countable s ↔ s.Countable := .rfl


/-- Prove `Set.Countable` from a `Countable` instance on the subtype. -/
theorem to_countable (s : Set α) [Countable s] : s.Countable := ‹_›


/-- Restate `Set.Countable` as a `Countable` instance. -/
alias ⟨_root_.Countable.to_set, Countable.to_subtype⟩ := countable_coe_iff


protected theorem countable_iff_exists_injective {s : Set α} :
    s.Countable ↔ ∃ f : s → ℕ, Injective f :=
  countable_iff_exists_injective s


/-- A set `s : Set α` is countable if and only if there exists a function `α → ℕ` injective
on `s`. -/
theorem countable_iff_exists_injOn {s : Set α} : s.Countable ↔ ∃ f : α → ℕ, InjOn f s :=
  Set.countable_iff_exists_injective.trans exists_injOn_iff_injective.symm


theorem countable_iff_nonempty_encodable {s : Set α} : s.Countable ↔ Nonempty (Encodable s) :=
  Encodable.nonempty_encodable.symm


alias ⟨Countable.nonempty_encodable, _⟩ := countable_iff_nonempty_encodable


/-- Convert `Set.Countable s` to `Encodable s` (noncomputable). -/
protected def Countable.toEncodable {s : Set α} (hs : s.Countable) : Encodable s :=
  Classical.choice hs.nonempty_encodable


/-- Noncomputably enumerate elements in a set. The `default` value is used to extend the domain to
all of `ℕ`. -/
def enumerateCountable {s : Set α} (h : s.Countable) (default : α) : ℕ → α := fun n =>
  match @Encodable.decode s h.toEncodable n with
  | some y => y
  | none => default


theorem subset_range_enumerate {s : Set α} (h : s.Countable) (default : α) :
    s ⊆ range (enumerateCountable h default) := fun x hx =>
  ⟨@Encodable.encode s h.toEncodable ⟨x, hx⟩, by
    /-
      α : Type u
      s : Set α
      h : s.Countable
      default x : α
      hx : Membership.mem s x
      ⊢ Eq (Set.enumerateCountable h default (Encodable.encode ⟨x, hx⟩)) x
    -/
    letI := h.toEncodable
    /-
      α : Type u
      s : Set α
      h : s.Countable
      default x : α
      hx : Membership.mem s x
      this : Encodable ↑s := h.toEncodable
      ⊢ Eq (Set.enumerateCountable h default (Encodable.encode ⟨x, hx⟩)) x
    -/
    simp [enumerateCountable, Encodable.encodek]⟩
    /-
      🎉 no goals
    -/


lemma range_enumerateCountable_subset {s : Set α} (h : s.Countable) (default : α) :
    range (enumerateCountable h default) ⊆ insert default s := by
  /-
    α : Type u
    s : Set α
    h : s.Countable
    default : α
    ⊢ HasSubset.Subset (Set.range (Set.enumerateCountable h default)) (Insert.inse …
  -/
  refine range_subset_iff.mpr (fun n ↦ ?_)
  /-
    α : Type u
    s : Set α
    h : s.Countable
    default : α
    n : Nat
    ⊢ Membership.mem (Insert.insert default s) (Set.enumerateCountable h default n)
  -/
  rw [enumerateCountable]
  match @decode s (Countable.toEncodable h) n with
  | none => exact mem_insert _ _
  | some val => simp


lemma range_enumerateCountable_of_mem {s : Set α} (h : s.Countable) {default : α}
    (h_mem : default ∈ s) :
    range (enumerateCountable h default) = s :=
  subset_antisymm ((range_enumerateCountable_subset h _).trans_eq (insert_eq_of_mem h_mem))
    (subset_range_enumerate h default)


lemma enumerateCountable_mem {s : Set α} (h : s.Countable) {default : α} (h_mem : default ∈ s)
    (n : ℕ) :
    enumerateCountable h default n ∈ s := by
  /-
    α : Type u
    s : Set α
    h : s.Countable
    default : α
    h_mem : Membership.mem s default
    n : Nat
    ⊢ Membership.mem s (Set.enumerateCountable h default n)
  -/
  convert mem_range_self n
  /-
    case h.e'_4
    α : Type u
    s : Set α
    h : s.Countable
    default : α
    h_mem : Membership.mem s default
    n : Nat
    ⊢ Eq s (Set.range (Set.enumerateCountable h default))
  -/
  exact (range_enumerateCountable_of_mem h h_mem).symm
  /-
    🎉 no goals
  -/


theorem Countable.mono {s₁ s₂ : Set α} (h : s₁ ⊆ s₂) (hs : s₂.Countable) : s₁.Countable :=
  have := hs.to_subtype; (inclusion_injective h).countable


theorem countable_range [Countable ι] (f : ι → β) : (range f).Countable :=
  surjective_onto_range.countable.to_set


theorem countable_iff_exists_subset_range [Nonempty α] {s : Set α} :
    s.Countable ↔ ∃ f : ℕ → α, s ⊆ range f :=
  ⟨fun h => by
    /-
      α : Type u
      inst✝ : Nonempty α
      s : Set α
      h : s.Countable
      ⊢ Exists fun f => HasSubset.Subset s (Set.range f)
    -/
    inhabit α
    /-
      α : Type u
      inst✝ : Nonempty α
      s : Set α
      h : s.Countable
      inhabited_h : Inhabited α
      ⊢ Exists fun f => HasSubset.Subset s (Set.range f)
    -/
    exact ⟨enumerateCountable h default, subset_range_enumerate _ _⟩, fun ⟨f, hsf⟩ =>
    /-
      🎉 no goals
    -/
    (countable_range f).mono hsf⟩


/-- A non-empty set is countable iff there exists a surjection from the
natural numbers onto the subtype induced by the set.
-/
protected theorem countable_iff_exists_surjective {s : Set α} (hs : s.Nonempty) :
    s.Countable ↔ ∃ f : ℕ → s, Surjective f :=
  @countable_iff_exists_surjective s hs.to_subtype


alias ⟨Countable.exists_surjective, _⟩ := Set.countable_iff_exists_surjective


theorem countable_univ_iff : (univ : Set α).Countable ↔ Countable α :=
  countable_coe_iff.symm.trans (Equiv.Set.univ _).countable_iff


theorem countable_univ [Countable α] : (univ : Set α).Countable :=
  to_countable univ


theorem not_countable_univ_iff : ¬ (univ : Set α).Countable ↔ Uncountable α := by
  /-
    α : Type u
    ⊢ Iff (Not Set.univ.Countable) (Uncountable α)
  -/
  rw [countable_univ_iff, not_countable_iff]
  /-
    🎉 no goals
  -/


theorem not_countable_univ [Uncountable α] : ¬ (univ : Set α).Countable :=
  not_countable_univ_iff.2 ‹_›


/-- If `s : Set α` is a nonempty countable set, then there exists a map
`f : ℕ → α` such that `s = range f`. -/
theorem Countable.exists_eq_range {s : Set α} (hc : s.Countable) (hs : s.Nonempty) :
    ∃ f : ℕ → α, s = range f := by
  /-
    α : Type u
    s : Set α
    hc : s.Countable
    hs : s.Nonempty
    ⊢ Exists fun f => Eq s (Set.range f)
  -/
  rcases hc.exists_surjective hs with ⟨f, hf⟩
  /-
    case intro
    α : Type u
    s : Set α
    hc : s.Countable
    hs : s.Nonempty
    f : Nat → ↑s
    hf : Function.Surjective f
    ⊢ Exists fun f => Eq s (Set.range f)
  -/
  refine ⟨(↑) ∘ f, ?_⟩
  /-
    case intro
    α : Type u
    s : Set α
    hc : s.Countable
    hs : s.Nonempty
    f : Nat → ↑s
    hf : Function.Surjective f
    ⊢ Eq s (Set.range (Function.comp Subtype.val f))
  -/
  rw [hf.range_comp, Subtype.range_coe]
  /-
    🎉 no goals
  -/


@[simp] theorem countable_empty : (∅ : Set α).Countable := to_countable _


@[simp] theorem countable_singleton (a : α) : ({a} : Set α).Countable := to_countable _


theorem Countable.image {s : Set α} (hs : s.Countable) (f : α → β) : (f '' s).Countable := by
  /-
    α : Type u
    β : Type v
    s : Set α
    hs : s.Countable
    f : α → β
    ⊢ (Set.image f s).Countable
  -/
  rw [image_eq_range]
  /-
    α : Type u
    β : Type v
    s : Set α
    hs : s.Countable
    f : α → β
    ⊢ (Set.range fun x => f ↑x).Countable
  -/
  have := hs.to_subtype
  /-
    α : Type u
    β : Type v
    s : Set α
    hs : s.Countable
    f : α → β
    this : Countable ↑s
    ⊢ (Set.range fun x => f ↑x).Countable
  -/
  apply countable_range
  /-
    🎉 no goals
  -/


theorem MapsTo.countable_of_injOn {s : Set α} {t : Set β} {f : α → β} (hf : MapsTo f s t)
    (hf' : InjOn f s) (ht : t.Countable) : s.Countable :=
  have := ht.to_subtype
  have : Injective (hf.restrict f s t) := (injOn_iff_injective.1 hf').codRestrict _
  this.countable


theorem Countable.preimage_of_injOn {s : Set β} (hs : s.Countable) {f : α → β}
    (hf : InjOn f (f ⁻¹' s)) : (f ⁻¹' s).Countable :=
  (mapsTo_preimage f s).countable_of_injOn hf hs


protected theorem Countable.preimage {s : Set β} (hs : s.Countable) {f : α → β} (hf : Injective f) :
    (f ⁻¹' s).Countable :=
  hs.preimage_of_injOn hf.injOn


theorem exists_seq_iSup_eq_top_iff_countable [CompleteLattice α] {p : α → Prop} (h : ∃ x, p x) :
    (∃ s : ℕ → α, (∀ n, p (s n)) ∧ ⨆ n, s n = ⊤) ↔
      ∃ S : Set α, S.Countable ∧ (∀ s ∈ S, p s) ∧ sSup S = ⊤ := by
  /-
    α : Type u
    inst✝ : CompleteLattice α
    p : α → Prop
    h : Exists fun x => p x
    ⊢ Iff (Exists fun s => And (∀ (n : Nat), p (s n)) (Eq (iSup fun n => s n) Top. …
  -/
  constructor
    /-
      case mp
      α : Type u
      inst✝ : CompleteLattice α
      p : α → Prop
      h : Exists fun x => p x
      ⊢ (Exists fun s => And (∀ (n : Nat), p (s n)) (Eq (iSup fun n => s n) Top.top) …
    -/
  · rintro ⟨s, hps, hs⟩
    /-
      case mp.intro.intro
      α : Type u
      inst✝ : CompleteLattice α
      p : α → Prop
      h : Exists fun x => p x
      s : Nat → α
      hps : ∀ (n : Nat), p (s n)
      hs : Eq (iSup fun n => s n) Top.top
      ⊢ Exists fun S => And S.Countable (And (∀ (s : α), Membership.mem S s → p s) ( …
    -/
    refine ⟨range s, countable_range s, forall_mem_range.2 hps, ?_⟩
    /-
      case mp.intro.intro
      α : Type u
      inst✝ : CompleteLattice α
      p : α → Prop
      h : Exists fun x => p x
      s : Nat → α
      hps : ∀ (n : Nat), p (s n)
      hs : Eq (iSup fun n => s n) Top.top
      ⊢ Eq (SupSet.sSup (Set.range s)) Top.top
    -/
    rwa [sSup_range]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u
      inst✝ : CompleteLattice α
      p : α → Prop
      h : Exists fun x => p x
      ⊢ (Exists fun S => And S.Countable (And (∀ (s : α), Membership.mem S s → p s)  …
    -/
  · rintro ⟨S, hSc, hps, hS⟩
    /-
      case mpr.intro.intro.intro
      α : Type u
      inst✝ : CompleteLattice α
      p : α → Prop
      h : Exists fun x => p x
      S : Set α
      hSc : S.Countable
      hps : ∀ (s : α), Membership.mem S s → p s
      hS : Eq (SupSet.sSup S) Top.top
      ⊢ Exists fun s => And (∀ (n : Nat), p (s n)) (Eq (iSup fun n => s n) Top.top)
    -/
    rcases eq_empty_or_nonempty S with (rfl | hne)
      /-
        case mpr.intro.intro.intro.inl
        α : Type u
        inst✝ : CompleteLattice α
        p : α → Prop
        h : Exists fun x => p x
        hSc : EmptyCollection.emptyCollection.Countable
        hps : ∀ (s : α), Membership.mem EmptyCollection.emptyCollection s → p s
        hS : Eq (SupSet.sSup EmptyCollection.emptyCollection) Top.top
        ⊢ Exists fun s => And (∀ (n : Nat), p (s n)) (Eq (iSup fun n => s n) Top.top)
      -/
    · rw [sSup_empty] at hS
      /-
        case mpr.intro.intro.intro.inl
        α : Type u
        inst✝ : CompleteLattice α
        p : α → Prop
        h : Exists fun x => p x
        hSc : EmptyCollection.emptyCollection.Countable
        hps : ∀ (s : α), Membership.mem EmptyCollection.emptyCollection s → p s
        hS : Eq Bot.bot Top.top
        ⊢ Exists fun s => And (∀ (n : Nat), p (s n)) (Eq (iSup fun n => s n) Top.top)
      -/
      haveI := subsingleton_of_bot_eq_top hS
      /-
        case mpr.intro.intro.intro.inl
        α : Type u
        inst✝ : CompleteLattice α
        p : α → Prop
        h : Exists fun x => p x
        hSc : EmptyCollection.emptyCollection.Countable
        hps : ∀ (s : α), Membership.mem EmptyCollection.emptyCollection s → p s
        hS : Eq Bot.bot Top.top
        this : Subsingleton α
        ⊢ Exists fun s => And (∀ (n : Nat), p (s n)) (Eq (iSup fun n => s n) Top.top)
      -/
      rcases h with ⟨x, hx⟩
      /-
        case mpr.intro.intro.intro.inl.intro
        α : Type u
        inst✝ : CompleteLattice α
        p : α → Prop
        hSc : EmptyCollection.emptyCollection.Countable
        hps : ∀ (s : α), Membership.mem EmptyCollection.emptyCollection s → p s
        hS : Eq Bot.bot Top.top
        this : Subsingleton α
        x : α
        hx : p x
        ⊢ Exists fun s => And (∀ (n : Nat), p (s n)) (Eq (iSup fun n => s n) Top.top)
      -/
      exact ⟨fun _ => x, fun _ => hx, Subsingleton.elim _ _⟩
      /-
        🎉 no goals
      -/
      /-
        case mpr.intro.intro.intro.inr
        α : Type u
        inst✝ : CompleteLattice α
        p : α → Prop
        h : Exists fun x => p x
        S : Set α
        hSc : S.Countable
        hps : ∀ (s : α), Membership.mem S s → p s
        hS : Eq (SupSet.sSup S) Top.top
        hne : S.Nonempty
        ⊢ Exists fun s => And (∀ (n : Nat), p (s n)) (Eq (iSup fun n => s n) Top.top)
      -/
    · rcases (Set.countable_iff_exists_surjective hne).1 hSc with ⟨s, hs⟩
      /-
        case mpr.intro.intro.intro.inr.intro
        α : Type u
        inst✝ : CompleteLattice α
        p : α → Prop
        h : Exists fun x => p x
        S : Set α
        hSc : S.Countable
        hps : ∀ (s : α), Membership.mem S s → p s
        hS : Eq (SupSet.sSup S) Top.top
        hne : S.Nonempty
        s : Nat → ↑S
        hs : Function.Surjective s
        ⊢ Exists fun s => And (∀ (n : Nat), p (s n)) (Eq (iSup fun n => s n) Top.top)
      -/
      refine ⟨fun n => s n, fun n => hps _ (s n).coe_prop, ?_⟩
      /-
        case mpr.intro.intro.intro.inr.intro
        α : Type u
        inst✝ : CompleteLattice α
        p : α → Prop
        h : Exists fun x => p x
        S : Set α
        hSc : S.Countable
        hps : ∀ (s : α), Membership.mem S s → p s
        hS : Eq (SupSet.sSup S) Top.top
        hne : S.Nonempty
        s : Nat → ↑S
        hs : Function.Surjective s
        ⊢ Eq (iSup fun n => (fun n => ↑(s n)) n) Top.top
      -/
      rwa [hs.iSup_comp, ← sSup_eq_iSup']
      /-
        🎉 no goals
      -/


theorem exists_seq_cover_iff_countable {p : Set α → Prop} (h : ∃ s, p s) :
    (∃ s : ℕ → Set α, (∀ n, p (s n)) ∧ ⋃ n, s n = univ) ↔
      ∃ S : Set (Set α), S.Countable ∧ (∀ s ∈ S, p s) ∧ ⋃₀ S = univ :=
  exists_seq_iSup_eq_top_iff_countable h


theorem countable_of_injective_of_countable_image {s : Set α} {f : α → β} (hf : InjOn f s)
    (hs : (f '' s).Countable) : s.Countable :=
  (mapsTo_image _ _).countable_of_injOn hf hs


theorem countable_iUnion {t : ι → Set α} [Countable ι] (ht : ∀ i, (t i).Countable) :
    (⋃ i, t i).Countable := by
  /-
    α : Type u
    ι : Sort x
    t : ι → Set α
    inst✝ : Countable ι
    ht : ∀ (i : ι), (t i).Countable
    ⊢ (Set.iUnion fun i => t i).Countable
  -/
  have := fun i ↦ (ht i).to_subtype
  /-
    α : Type u
    ι : Sort x
    t : ι → Set α
    inst✝ : Countable ι
    ht : ∀ (i : ι), (t i).Countable
    this : ∀ (i : ι), Countable ↑(t i)
    ⊢ (Set.iUnion fun i => t i).Countable
  -/
  rw [iUnion_eq_range_psigma]
  /-
    α : Type u
    ι : Sort x
    t : ι → Set α
    inst✝ : Countable ι
    ht : ∀ (i : ι), (t i).Countable
    this : ∀ (i : ι), Countable ↑(t i)
    ⊢ (Set.range fun a => ↑a.snd).Countable
  -/
  apply countable_range
  /-
    🎉 no goals
  -/


@[simp]
theorem countable_iUnion_iff [Countable ι] {t : ι → Set α} :
    (⋃ i, t i).Countable ↔ ∀ i, (t i).Countable :=
  ⟨fun h _ => h.mono <| subset_iUnion _ _, countable_iUnion⟩


theorem Countable.biUnion_iff {s : Set α} {t : ∀ a ∈ s, Set β} (hs : s.Countable) :
    (⋃ a ∈ s, t a ‹_›).Countable ↔ ∀ a (ha : a ∈ s), (t a ha).Countable := by
  /-
    α : Type u
    β : Type v
    s : Set α
    t : (a : α) → Membership.mem s a → Set β
    hs : s.Countable
    ⊢ Iff (Set.iUnion fun a => Set.iUnion fun h => t a h).Countable (∀ (a : α) (ha …
  -/
  have := hs.to_subtype
  /-
    α : Type u
    β : Type v
    s : Set α
    t : (a : α) → Membership.mem s a → Set β
    hs : s.Countable
    this : Countable ↑s
    ⊢ Iff (Set.iUnion fun a => Set.iUnion fun h => t a h).Countable (∀ (a : α) (ha …
  -/
  rw [biUnion_eq_iUnion, countable_iUnion_iff, SetCoe.forall']
  /-
    🎉 no goals
  -/


theorem Countable.sUnion_iff {s : Set (Set α)} (hs : s.Countable) :
                                                  /-
                                                    α : Type u
                                                    s : Set (Set α)
                                                    hs : s.Countable
                                                    ⊢ Iff s.sUnion.Countable (∀ (a : Set α), Membership.mem s a → a.Countable)
                                                  -/
    (⋃₀ s).Countable ↔ ∀ a ∈ s, a.Countable := by rw [sUnion_eq_biUnion, hs.biUnion_iff]
                                                  /-
                                                    🎉 no goals
                                                  -/


alias ⟨_, Countable.biUnion⟩ := Countable.biUnion_iff


alias ⟨_, Countable.sUnion⟩ := Countable.sUnion_iff


@[simp]
theorem countable_union {s t : Set α} : (s ∪ t).Countable ↔ s.Countable ∧ t.Countable := by
  /-
    α : Type u
    s t : Set α
    ⊢ Iff (Union.union s t).Countable (And s.Countable t.Countable)
  -/
  simp [union_eq_iUnion, and_comm]
  /-
    🎉 no goals
  -/


theorem Countable.union {s t : Set α} (hs : s.Countable) (ht : t.Countable) : (s ∪ t).Countable :=
  countable_union.2 ⟨hs, ht⟩


theorem Countable.of_diff {s t : Set α} (h : (s \ t).Countable) (ht : t.Countable) : s.Countable :=
  (h.union ht).mono (subset_diff_union _ _)


@[simp]
theorem countable_insert {s : Set α} {a : α} : (insert a s).Countable ↔ s.Countable := by
  /-
    α : Type u
    s : Set α
    a : α
    ⊢ Iff (Insert.insert a s).Countable s.Countable
  -/
  simp only [insert_eq, countable_union, countable_singleton, true_and]
  /-
    🎉 no goals
  -/


protected theorem Countable.insert {s : Set α} (a : α) (h : s.Countable) : (insert a s).Countable :=
  countable_insert.2 h


theorem Finite.countable {s : Set α} (hs : s.Finite) : s.Countable :=
  have := hs.to_subtype; s.to_countable


@[nontriviality]
theorem Countable.of_subsingleton [Subsingleton α] (s : Set α) : s.Countable :=
  (Finite.of_subsingleton s).countable


theorem Subsingleton.countable {s : Set α} (hs : s.Subsingleton) : s.Countable :=
  hs.finite.countable


theorem countable_isTop (α : Type*) [PartialOrder α] : { x : α | IsTop x }.Countable :=
  (finite_isTop α).countable


theorem countable_isBot (α : Type*) [PartialOrder α] : { x : α | IsBot x }.Countable :=
  (finite_isBot α).countable


/-- The set of finite subsets of a countable set is countable. -/
theorem countable_setOf_finite_subset {s : Set α} (hs : s.Countable) :
    { t | Set.Finite t ∧ t ⊆ s }.Countable := by
  /-
    α : Type u
    s : Set α
    hs : s.Countable
    ⊢ (setOf fun t => And t.Finite (HasSubset.Subset t s)).Countable
  -/
  have := hs.to_subtype
  /-
    α : Type u
    s : Set α
    hs : s.Countable
    this : Countable ↑s
    ⊢ (setOf fun t => And t.Finite (HasSubset.Subset t s)).Countable
  -/
  refine (countable_range fun t : Finset s => Subtype.val '' (t : Set s)).mono ?_
  /-
    α : Type u
    s : Set α
    hs : s.Countable
    this : Countable ↑s
    ⊢ HasSubset.Subset (setOf fun t => And t.Finite (HasSubset.Subset t s)) (Set.r …
  -/
  rintro t ⟨ht, hts⟩
  /-
    case intro
    α : Type u
    s : Set α
    hs : s.Countable
    this : Countable ↑s
    t : Set α
    ht : t.Finite
    hts : HasSubset.Subset t s
    ⊢ Membership.mem (Set.range fun t => Set.image Subtype.val ↑t) t
  -/
  lift t to Set s using hts
  /-
    case intro.intro
    α : Type u
    s : Set α
    hs : s.Countable
    this : Countable ↑s
    t : Set ↑s
    ht : (Set.image Subtype.val t).Finite
    ⊢ Membership.mem (Set.range fun t => Set.image Subtype.val ↑t) (Set.image Subt …
  -/
  lift t to Finset s using ht.of_finite_image Subtype.val_injective.injOn
  /-
    case intro.intro.intro
    α : Type u
    s : Set α
    hs : s.Countable
    this : Countable ↑s
    t : Finset ↑s
    ht : (Set.image Subtype.val ↑t).Finite
    ⊢ Membership.mem (Set.range fun t => Set.image Subtype.val ↑t) (Set.image Subt …
  -/
  exact mem_range_self _
  /-
    🎉 no goals
  -/


/-- The set of finite sets in a countable type is countable. -/
theorem Countable.setOf_finite [Countable α] : {s : Set α | s.Finite}.Countable := by
  /-
    α : Type u
    inst✝ : Countable α
    ⊢ (setOf fun s => s.Finite).Countable
  -/
  simpa using countable_setOf_finite_subset countable_univ
  /-
    🎉 no goals
  -/


theorem countable_univ_pi {π : α → Type*} [Finite α] {s : ∀ a, Set (π a)}
    (hs : ∀ a, (s a).Countable) : (pi univ s).Countable :=
  have := fun a ↦ (hs a).to_subtype; .of_equiv _ (Equiv.Set.univPi s).symm


theorem countable_pi {π : α → Type*} [Finite α] {s : ∀ a, Set (π a)} (hs : ∀ a, (s a).Countable) :
    { f : ∀ a, π a | ∀ a, f a ∈ s a }.Countable := by
  /-
    α : Type u
    π : α → Type u_1
    inst✝ : Finite α
    s : (a : α) → Set (π a)
    hs : ∀ (a : α), (s a).Countable
    ⊢ (setOf fun f => ∀ (a : α), Membership.mem (s a) (f a)).Countable
  -/
  simpa only [← mem_univ_pi] using countable_univ_pi hs
  /-
    🎉 no goals
  -/


protected theorem Countable.prod {s : Set α} {t : Set β} (hs : s.Countable) (ht : t.Countable) :
    Set.Countable (s ×ˢ t) :=
  have := hs.to_subtype; have := ht.to_subtype; .of_equiv _ <| (Equiv.Set.prod _ _).symm


theorem Countable.image2 {s : Set α} {t : Set β} (hs : s.Countable) (ht : t.Countable)
    (f : α → β → γ) : (image2 f s t).Countable := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    s : Set α
    t : Set β
    hs : s.Countable
    ht : t.Countable
    f : α → β → γ
    ⊢ (Set.image2 f s t).Countable
  -/
  rw [← image_prod]
  /-
    α : Type u
    β : Type v
    γ : Type w
    s : Set α
    t : Set β
    hs : s.Countable
    ht : t.Countable
    f : α → β → γ
    ⊢ (Set.image (fun x => f x.1 x.2) (SProd.sprod s t)).Countable
  -/
  exact (hs.prod ht).image _
  /-
    🎉 no goals
  -/


/-- If a family of disjoint sets is included in a countable set, then only countably many of
them are nonempty. -/
theorem countable_setOf_nonempty_of_disjoint {f : β → Set α}
    (hf : Pairwise (Disjoint on f)) {s : Set α} (h'f : ∀ t, f t ⊆ s) (hs : s.Countable) :
    Set.Countable {t | (f t).Nonempty} := by
  /-
    α : Type u
    β : Type v
    f : β → Set α
    hf : Pairwise (Function.onFun Disjoint f)
    s : Set α
    h'f : ∀ (t : β), HasSubset.Subset (f t) s
    hs : s.Countable
    ⊢ (setOf fun t => (f t).Nonempty).Countable
  -/
  rw [← Set.countable_coe_iff] at hs ⊢
  have : ∀ t : {t // (f t).Nonempty}, ∃ x : s, x.1 ∈ f t := by
    rintro ⟨t, ⟨x, hx⟩⟩
    exact ⟨⟨x, (h'f t hx)⟩, hx⟩
  /-
    α : Type u
    β : Type v
    f : β → Set α
    hf : Pairwise (Function.onFun Disjoint f)
    s : Set α
    h'f : ∀ (t : β), HasSubset.Subset (f t) s
    hs : Countable ↑s
    this : ∀ (t : Subtype fun t => (f t).Nonempty), Exists fun x => Membership.mem …
    ⊢ Countable ↑(setOf fun t => (f t).Nonempty)
  -/
  choose F hF using this
  have A : Injective F := by
    rintro ⟨t, ht⟩ ⟨t', ht'⟩ htt'
    have A : (f t ∩ f t').Nonempty := by
      refine ⟨F ⟨t, ht⟩, hF ⟨t, _⟩, ?_⟩
      rw [htt']
      exact hF ⟨t', _⟩
    simp only [Subtype.mk.injEq]
    by_contra H
    exact not_disjoint_iff_nonempty_inter.2 A (hf H)
  /-
    α : Type u
    β : Type v
    f : β → Set α
    hf : Pairwise (Function.onFun Disjoint f)
    s : Set α
    h'f : ∀ (t : β), HasSubset.Subset (f t) s
    hs : Countable ↑s
    F : (Subtype fun t => (f t).Nonempty) → ↑s
    hF : ∀ (t : Subtype fun t => (f t).Nonempty), Membership.mem (f ↑t) ↑(F t)
    A : Function.Injective F
    ⊢ Countable ↑(setOf fun t => (f t).Nonempty)
  -/
  exact Injective.countable A
  /-
    🎉 no goals
  -/


theorem Finset.countable_toSet (s : Finset α) : Set.Countable (↑s : Set α) :=
  s.finite_toSet.countable

