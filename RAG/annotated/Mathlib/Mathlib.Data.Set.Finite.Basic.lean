theorem finite_def {s : Set α} : s.Finite ↔ Nonempty (Fintype s) :=
  finite_iff_nonempty_fintype s


protected alias ⟨Finite.nonempty_fintype, _⟩ := finite_def


/-- Construct a `Finite` instance for a `Set` from a `Finset` with the same elements. -/
protected theorem Finite.ofFinset {p : Set α} (s : Finset α) (H : ∀ x, x ∈ s ↔ x ∈ p) : p.Finite :=
  have := Fintype.ofFinset s H; p.toFinite


/-- A finite set coerced to a type is a `Fintype`.
This is the `Fintype` projection for a `Set.Finite`.

Note that because `Finite` isn't a typeclass, this definition will not fire if it
is made into an instance -/
protected noncomputable def Finite.fintype {s : Set α} (h : s.Finite) : Fintype s :=
  h.nonempty_fintype.some


/-- Using choice, get the `Finset` that represents this `Set`. -/
protected noncomputable def Finite.toFinset {s : Set α} (h : s.Finite) : Finset α :=
  @Set.toFinset _ _ h.fintype


theorem Finite.toFinset_eq_toFinset {s : Set α} [Fintype s] (h : s.Finite) :
    h.toFinset = s.toFinset := by
  -- Porting note: was `rw [Finite.toFinset]; congr`
  -- in Lean 4, a goal is left after `congr`
  /-
    α : Type u
    s : Set α
    inst✝ : Fintype ↑s
    h : s.Finite
    ⊢ Eq h.toFinset s.toFinset
  -/
  have : h.fintype = ‹_› := Subsingleton.elim _ _
  /-
    α : Type u
    s : Set α
    inst✝ : Fintype ↑s
    h : s.Finite
    this : Eq h.fintype inst✝
    ⊢ Eq h.toFinset s.toFinset
  -/
  rw [Finite.toFinset, this]
  /-
    🎉 no goals
  -/


@[simp]
theorem toFinite_toFinset (s : Set α) [Fintype s] : s.toFinite.toFinset = s.toFinset :=
  s.toFinite.toFinset_eq_toFinset


theorem Finite.exists_finset {s : Set α} (h : s.Finite) :
    ∃ s' : Finset α, ∀ a : α, a ∈ s' ↔ a ∈ s := by
  /-
    α : Type u
    s : Set α
    h : s.Finite
    ⊢ Exists fun s' => ∀ (a : α), Iff (Membership.mem s' a) (Membership.mem s a)
  -/
  cases h.nonempty_fintype
  /-
    case intro
    α : Type u
    s : Set α
    h : s.Finite
    val✝ : Fintype ↑s
    ⊢ Exists fun s' => ∀ (a : α), Iff (Membership.mem s' a) (Membership.mem s a)
  -/
  exact ⟨s.toFinset, fun _ => mem_toFinset⟩
  /-
    🎉 no goals
  -/


theorem Finite.exists_finset_coe {s : Set α} (h : s.Finite) : ∃ s' : Finset α, ↑s' = s := by
  /-
    α : Type u
    s : Set α
    h : s.Finite
    ⊢ Exists fun s' => Eq (↑s') s
  -/
  cases h.nonempty_fintype
  /-
    case intro
    α : Type u
    s : Set α
    h : s.Finite
    val✝ : Fintype ↑s
    ⊢ Exists fun s' => Eq (↑s') s
  -/
  exact ⟨s.toFinset, s.coe_toFinset⟩
  /-
    🎉 no goals
  -/


/-- Finite sets can be lifted to finsets. -/
instance : CanLift (Set α) (Finset α) (↑) Set.Finite where prf _ hs := hs.exists_finset_coe


@[simp]
protected theorem mem_toFinset : a ∈ hs.toFinset ↔ a ∈ s :=
  @mem_toFinset _ _ hs.fintype _


@[simp]
protected theorem coe_toFinset : (hs.toFinset : Set α) = s :=
  @coe_toFinset _ _ hs.fintype


@[simp]
protected theorem toFinset_nonempty : hs.toFinset.Nonempty ↔ s.Nonempty := by
  /-
    α : Type u
    s : Set α
    hs : s.Finite
    ⊢ Iff hs.toFinset.Nonempty s.Nonempty
  -/
  rw [← Finset.coe_nonempty, Finite.coe_toFinset]
  /-
    🎉 no goals
  -/


/-- Note that this is an equality of types not holding definitionally. Use wisely. -/
theorem coeSort_toFinset : ↥hs.toFinset = ↥s := by
  /-
    α : Type u
    s : Set α
    hs : s.Finite
    ⊢ Eq (Subtype fun x => Membership.mem hs.toFinset x) ↑s
  -/
  rw [← Finset.coe_sort_coe _, hs.coe_toFinset]
  /-
    🎉 no goals
  -/


/-- The identity map, bundled as an equivalence between the subtypes of `s : Set α` and of
`h.toFinset : Finset α`, where `h` is a proof of finiteness of `s`. -/
@[simps!] def subtypeEquivToFinset : {x // x ∈ s} ≃ {x // x ∈ hs.toFinset} :=
  (Equiv.refl α).subtypeEquiv fun _ ↦ hs.mem_toFinset.symm


@[simp]
protected theorem toFinset_inj : hs.toFinset = ht.toFinset ↔ s = t :=
  @toFinset_inj _ _ _ hs.fintype ht.fintype


@[simp]
theorem toFinset_subset {t : Finset α} : hs.toFinset ⊆ t ↔ s ⊆ t := by
  /-
    α : Type u
    s : Set α
    hs : s.Finite
    t : Finset α
    ⊢ Iff (HasSubset.Subset hs.toFinset t) (HasSubset.Subset s ↑t)
  -/
  rw [← Finset.coe_subset, Finite.coe_toFinset]
  /-
    🎉 no goals
  -/


@[simp]
theorem toFinset_ssubset {t : Finset α} : hs.toFinset ⊂ t ↔ s ⊂ t := by
  /-
    α : Type u
    s : Set α
    hs : s.Finite
    t : Finset α
    ⊢ Iff (HasSSubset.SSubset hs.toFinset t) (HasSSubset.SSubset s ↑t)
  -/
  rw [← Finset.coe_ssubset, Finite.coe_toFinset]
  /-
    🎉 no goals
  -/


@[simp]
theorem subset_toFinset {s : Finset α} : s ⊆ ht.toFinset ↔ ↑s ⊆ t := by
  /-
    α : Type u
    t : Set α
    ht : t.Finite
    s : Finset α
    ⊢ Iff (HasSubset.Subset s ht.toFinset) (HasSubset.Subset (↑s) t)
  -/
  rw [← Finset.coe_subset, Finite.coe_toFinset]
  /-
    🎉 no goals
  -/


@[simp]
theorem ssubset_toFinset {s : Finset α} : s ⊂ ht.toFinset ↔ ↑s ⊂ t := by
  /-
    α : Type u
    t : Set α
    ht : t.Finite
    s : Finset α
    ⊢ Iff (HasSSubset.SSubset s ht.toFinset) (HasSSubset.SSubset (↑s) t)
  -/
  rw [← Finset.coe_ssubset, Finite.coe_toFinset]
  /-
    🎉 no goals
  -/


@[mono]
protected theorem toFinset_subset_toFinset : hs.toFinset ⊆ ht.toFinset ↔ s ⊆ t := by
  /-
    α : Type u
    s t : Set α
    hs : s.Finite
    ht : t.Finite
    ⊢ Iff (HasSubset.Subset hs.toFinset ht.toFinset) (HasSubset.Subset s t)
  -/
  simp only [← Finset.coe_subset, Finite.coe_toFinset]
  /-
    🎉 no goals
  -/


@[mono]
protected theorem toFinset_ssubset_toFinset : hs.toFinset ⊂ ht.toFinset ↔ s ⊂ t := by
  /-
    α : Type u
    s t : Set α
    hs : s.Finite
    ht : t.Finite
    ⊢ Iff (HasSSubset.SSubset hs.toFinset ht.toFinset) (HasSSubset.SSubset s t)
  -/
  simp only [← Finset.coe_ssubset, Finite.coe_toFinset]
  /-
    🎉 no goals
  -/


protected alias ⟨_, toFinset_mono⟩ := Finite.toFinset_subset_toFinset


protected alias ⟨_, toFinset_strictMono⟩ := Finite.toFinset_ssubset_toFinset

-- Porting note: `simp` can simplify LHS but then it simplifies something
-- in the generated `Fintype {x | p x}` instance and fails to apply `Set.toFinset_setOf`

@[simp high]
protected theorem toFinset_setOf [Fintype α] (p : α → Prop) [DecidablePred p]
    (h : { x | p x }.Finite) : h.toFinset = Finset.univ.filter p := by
  /-
    α : Type u
    inst✝¹ : Fintype α
    p : α → Prop
    inst✝ : DecidablePred p
    h : (setOf fun x => p x).Finite
    ⊢ Eq h.toFinset (Finset.filter p Finset.univ)
  -/
  ext
  -- Porting note: `simp` doesn't use the `simp` lemma `Set.toFinset_setOf` without the `_`
  /-
    case h
    α : Type u
    inst✝¹ : Fintype α
    p : α → Prop
    inst✝ : DecidablePred p
    h : (setOf fun x => p x).Finite
    a✝ : α
    ⊢ Iff (Membership.mem h.toFinset a✝) (Membership.mem (Finset.filter p Finset.u …
  -/
  simp [Set.toFinset_setOf _]
  /-
    🎉 no goals
  -/


@[simp]
nonrec theorem disjoint_toFinset {hs : s.Finite} {ht : t.Finite} :
    Disjoint hs.toFinset ht.toFinset ↔ Disjoint s t :=
  @disjoint_toFinset _ _ _ hs.fintype ht.fintype


protected theorem toFinset_inter [DecidableEq α] (hs : s.Finite) (ht : t.Finite)
    (h : (s ∩ t).Finite) : h.toFinset = hs.toFinset ∩ ht.toFinset := by
  /-
    α : Type u
    s t : Set α
    inst✝ : DecidableEq α
    hs : s.Finite
    ht : t.Finite
    h : (Inter.inter s t).Finite
    ⊢ Eq h.toFinset (Inter.inter hs.toFinset ht.toFinset)
  -/
  ext
  /-
    case h
    α : Type u
    s t : Set α
    inst✝ : DecidableEq α
    hs : s.Finite
    ht : t.Finite
    h : (Inter.inter s t).Finite
    a✝ : α
    ⊢ Iff (Membership.mem h.toFinset a✝) (Membership.mem (Inter.inter hs.toFinset  …
  -/
  simp
  /-
    🎉 no goals
  -/


protected theorem toFinset_union [DecidableEq α] (hs : s.Finite) (ht : t.Finite)
    (h : (s ∪ t).Finite) : h.toFinset = hs.toFinset ∪ ht.toFinset := by
  /-
    α : Type u
    s t : Set α
    inst✝ : DecidableEq α
    hs : s.Finite
    ht : t.Finite
    h : (Union.union s t).Finite
    ⊢ Eq h.toFinset (Union.union hs.toFinset ht.toFinset)
  -/
  ext
  /-
    case h
    α : Type u
    s t : Set α
    inst✝ : DecidableEq α
    hs : s.Finite
    ht : t.Finite
    h : (Union.union s t).Finite
    a✝ : α
    ⊢ Iff (Membership.mem h.toFinset a✝) (Membership.mem (Union.union hs.toFinset  …
  -/
  simp
  /-
    🎉 no goals
  -/


protected theorem toFinset_diff [DecidableEq α] (hs : s.Finite) (ht : t.Finite)
    (h : (s \ t).Finite) : h.toFinset = hs.toFinset \ ht.toFinset := by
  /-
    α : Type u
    s t : Set α
    inst✝ : DecidableEq α
    hs : s.Finite
    ht : t.Finite
    h : (SDiff.sdiff s t).Finite
    ⊢ Eq h.toFinset (SDiff.sdiff hs.toFinset ht.toFinset)
  -/
  ext
  /-
    case h
    α : Type u
    s t : Set α
    inst✝ : DecidableEq α
    hs : s.Finite
    ht : t.Finite
    h : (SDiff.sdiff s t).Finite
    a✝ : α
    ⊢ Iff (Membership.mem h.toFinset a✝) (Membership.mem (SDiff.sdiff hs.toFinset  …
  -/
  simp
  /-
    🎉 no goals
  -/


open scoped symmDiff in
protected theorem toFinset_symmDiff [DecidableEq α] (hs : s.Finite) (ht : t.Finite)
    (h : (s ∆ t).Finite) : h.toFinset = hs.toFinset ∆ ht.toFinset := by
  /-
    α : Type u
    s t : Set α
    inst✝ : DecidableEq α
    hs : s.Finite
    ht : t.Finite
    h : (symmDiff s t).Finite
    ⊢ Eq h.toFinset (symmDiff hs.toFinset ht.toFinset)
  -/
  ext
  /-
    case h
    α : Type u
    s t : Set α
    inst✝ : DecidableEq α
    hs : s.Finite
    ht : t.Finite
    h : (symmDiff s t).Finite
    a✝ : α
    ⊢ Iff (Membership.mem h.toFinset a✝) (Membership.mem (symmDiff hs.toFinset ht. …
  -/
  simp [mem_symmDiff, Finset.mem_symmDiff]
  /-
    🎉 no goals
  -/


protected theorem toFinset_compl [DecidableEq α] [Fintype α] (hs : s.Finite) (h : sᶜ.Finite) :
    h.toFinset = hs.toFinsetᶜ := by
  /-
    α : Type u
    s : Set α
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    hs : s.Finite
    h : (HasCompl.compl s).Finite
    ⊢ Eq h.toFinset (HasCompl.compl hs.toFinset)
  -/
  ext
  /-
    case h
    α : Type u
    s : Set α
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    hs : s.Finite
    h : (HasCompl.compl s).Finite
    a✝ : α
    ⊢ Iff (Membership.mem h.toFinset a✝) (Membership.mem (HasCompl.compl hs.toFins …
  -/
  simp
  /-
    🎉 no goals
  -/


protected theorem toFinset_univ [Fintype α] (h : (Set.univ : Set α).Finite) :
    h.toFinset = Finset.univ := by
  /-
    α : Type u
    inst✝ : Fintype α
    h : Set.univ.Finite
    ⊢ Eq h.toFinset Finset.univ
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
protected theorem toFinset_eq_empty {h : s.Finite} : h.toFinset = ∅ ↔ s = ∅ :=
  @toFinset_eq_empty _ _ h.fintype


protected theorem toFinset_empty (h : (∅ : Set α).Finite) : h.toFinset = ∅ := by
  /-
    α : Type u
    h : EmptyCollection.emptyCollection.Finite
    ⊢ Eq h.toFinset EmptyCollection.emptyCollection
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
protected theorem toFinset_eq_univ [Fintype α] {h : s.Finite} :
    h.toFinset = Finset.univ ↔ s = univ :=
  @toFinset_eq_univ _ _ _ h.fintype


protected theorem toFinset_image [DecidableEq β] (f : α → β) (hs : s.Finite) (h : (f '' s).Finite) :
    h.toFinset = hs.toFinset.image f := by
  /-
    α : Type u
    β : Type v
    s : Set α
    inst✝ : DecidableEq β
    f : α → β
    hs : s.Finite
    h : (Set.image f s).Finite
    ⊢ Eq h.toFinset (Finset.image f hs.toFinset)
  -/
  ext
  /-
    case h
    α : Type u
    β : Type v
    s : Set α
    inst✝ : DecidableEq β
    f : α → β
    hs : s.Finite
    h : (Set.image f s).Finite
    a✝ : β
    ⊢ Iff (Membership.mem h.toFinset a✝) (Membership.mem (Finset.image f hs.toFins …
  -/
  simp
  /-
    🎉 no goals
  -/


protected theorem toFinset_range [DecidableEq α] [Fintype β] (f : β → α) (h : (range f).Finite) :
    h.toFinset = Finset.univ.image f := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : DecidableEq α
    inst✝ : Fintype β
    f : β → α
    h : (Set.range f).Finite
    ⊢ Eq h.toFinset (Finset.image f Finset.univ)
  -/
  ext
  /-
    case h
    α : Type u
    β : Type v
    inst✝¹ : DecidableEq α
    inst✝ : Fintype β
    f : β → α
    h : (Set.range f).Finite
    a✝ : α
    ⊢ Iff (Membership.mem h.toFinset a✝) (Membership.mem (Finset.image f Finset.un …
  -/
  simp
  /-
    🎉 no goals
  -/


instance fintypeUniv [Fintype α] : Fintype (@univ α) :=
  Fintype.ofEquiv α (Equiv.Set.univ α).symm

-- Redeclared with appropriate keys

instance fintypeTop [Fintype α] : Fintype (⊤ : Set α) := inferInstanceAs (Fintype (univ : Set α))


/-- If `(Set.univ : Set α)` is finite then `α` is a finite type. -/
noncomputable def fintypeOfFiniteUniv (H : (univ (α := α)).Finite) : Fintype α :=
  @Fintype.ofEquiv _ (univ : Set α) H.fintype (Equiv.Set.univ _)


instance fintypeUnion [DecidableEq α] (s t : Set α) [Fintype s] [Fintype t] :
    Fintype (s ∪ t : Set α) :=
                                                   /-
                                                     α : Type u
                                                     β : Type v
                                                     ι : Sort w
                                                     γ : Type x
                                                     inst✝² : DecidableEq α
                                                     s t : Set α
                                                     inst✝¹ : Fintype ↑s
                                                     inst✝ : Fintype ↑t
                                                     ⊢ ∀ (x : α), Iff (Membership.mem (Union.union s.toFinset t.toFinset) x) (Membe …
                                                   -/
  Fintype.ofFinset (s.toFinset ∪ t.toFinset) <| by simp
                                                   /-
                                                     🎉 no goals
                                                   -/


instance fintypeSep (s : Set α) (p : α → Prop) [Fintype s] [DecidablePred p] :
    Fintype ({ a ∈ s | p a } : Set α) :=
                                               /-
                                                 α : Type u
                                                 β : Type v
                                                 ι : Sort w
                                                 γ : Type x
                                                 s : Set α
                                                 p : α → Prop
                                                 inst✝¹ : Fintype ↑s
                                                 inst✝ : DecidablePred p
                                                 ⊢ ∀ (x : α), Iff (Membership.mem (Finset.filter p s.toFinset) x) (Membership.m …
                                               -/
  Fintype.ofFinset (s.toFinset.filter p) <| by simp
                                               /-
                                                 🎉 no goals
                                               -/


instance fintypeInter (s t : Set α) [DecidableEq α] [Fintype s] [Fintype t] :
    Fintype (s ∩ t : Set α) :=
                                                   /-
                                                     α : Type u
                                                     β : Type v
                                                     ι : Sort w
                                                     γ : Type x
                                                     s t : Set α
                                                     inst✝² : DecidableEq α
                                                     inst✝¹ : Fintype ↑s
                                                     inst✝ : Fintype ↑t
                                                     ⊢ ∀ (x : α), Iff (Membership.mem (Inter.inter s.toFinset t.toFinset) x) (Membe …
                                                   -/
  Fintype.ofFinset (s.toFinset ∩ t.toFinset) <| by simp
                                                   /-
                                                     🎉 no goals
                                                   -/


/-- A `Fintype` instance for set intersection where the left set has a `Fintype` instance. -/
instance fintypeInterOfLeft (s t : Set α) [Fintype s] [DecidablePred (· ∈ t)] :
    Fintype (s ∩ t : Set α) :=
                                                     /-
                                                       α : Type u
                                                       β : Type v
                                                       ι : Sort w
                                                       γ : Type x
                                                       s t : Set α
                                                       inst✝¹ : Fintype ↑s
                                                       inst✝ : DecidablePred fun x => Membership.mem t x
                                                       ⊢ ∀ (x : α), Iff (Membership.mem (Finset.filter (fun x => Membership.mem t x)  …
                                                     -/
  Fintype.ofFinset (s.toFinset.filter (· ∈ t)) <| by simp
                                                     /-
                                                       🎉 no goals
                                                     -/


/-- A `Fintype` instance for set intersection where the right set has a `Fintype` instance. -/
instance fintypeInterOfRight (s t : Set α) [Fintype t] [DecidablePred (· ∈ s)] :
    Fintype (s ∩ t : Set α) :=
                                                     /-
                                                       α : Type u
                                                       β : Type v
                                                       ι : Sort w
                                                       γ : Type x
                                                       s t : Set α
                                                       inst✝¹ : Fintype ↑t
                                                       inst✝ : DecidablePred fun x => Membership.mem s x
                                                       ⊢ ∀ (x : α), Iff (Membership.mem (Finset.filter (fun x => Membership.mem s x)  …
                                                     -/
  Fintype.ofFinset (t.toFinset.filter (· ∈ s)) <| by simp [and_comm]
                                                     /-
                                                       🎉 no goals
                                                     -/


/-- A `Fintype` structure on a set defines a `Fintype` structure on its subset. -/
def fintypeSubset (s : Set α) {t : Set α} [Fintype s] [DecidablePred (· ∈ t)] (h : t ⊆ s) :
    Fintype t := by
  /-
    α : Type u
    β : Type v
    ι : Sort w
    γ : Type x
    s t : Set α
    inst✝¹ : Fintype ↑s
    inst✝ : DecidablePred fun x => Membership.mem t x
    h : HasSubset.Subset t s
    ⊢ Fintype ↑t
  -/
  rw [← inter_eq_self_of_subset_right h]
  /-
    α : Type u
    β : Type v
    ι : Sort w
    γ : Type x
    s t : Set α
    inst✝¹ : Fintype ↑s
    inst✝ : DecidablePred fun x => Membership.mem t x
    h : HasSubset.Subset t s
    ⊢ Fintype ↑(Inter.inter s t)
  -/
  apply Set.fintypeInterOfLeft
  /-
    🎉 no goals
  -/


instance fintypeDiff [DecidableEq α] (s t : Set α) [Fintype s] [Fintype t] :
    Fintype (s \ t : Set α) :=
                                                   /-
                                                     α : Type u
                                                     β : Type v
                                                     ι : Sort w
                                                     γ : Type x
                                                     inst✝² : DecidableEq α
                                                     s t : Set α
                                                     inst✝¹ : Fintype ↑s
                                                     inst✝ : Fintype ↑t
                                                     ⊢ ∀ (x : α), Iff (Membership.mem (SDiff.sdiff s.toFinset t.toFinset) x) (Membe …
                                                   -/
  Fintype.ofFinset (s.toFinset \ t.toFinset) <| by simp
                                                   /-
                                                     🎉 no goals
                                                   -/


instance fintypeDiffLeft (s t : Set α) [Fintype s] [DecidablePred (· ∈ t)] :
    Fintype (s \ t : Set α) :=
  Set.fintypeSep s (· ∈ tᶜ)


/-- A union of sets with `Fintype` structure over a set with `Fintype` structure has a `Fintype`
structure. -/
def fintypeBiUnion [DecidableEq α] {ι : Type*} (s : Set ι) [Fintype s] (t : ι → Set α)
    (H : ∀ i ∈ s, Fintype (t i)) : Fintype (⋃ x ∈ s, t x) :=
  haveI : ∀ i : toFinset s, Fintype (t i) := fun i => H i (mem_toFinset.1 i.2)
                                                                                   /-
                                                                                     α : Type u
                                                                                     β : Type v
                                                                                     ι✝ : Sort w
                                                                                     γ : Type x
                                                                                     inst✝¹ : DecidableEq α
                                                                                     ι : Type u_1
                                                                                     s : Set ι
                                                                                     inst✝ : Fintype ↑s
                                                                                     t : ι → Set α
                                                                                     H : (i : ι) → Membership.mem s i → Fintype ↑(t i)
                                                                                     this : (i : Subtype fun x => Membership.mem s.toFinset x) → Fintype ↑(t ↑i)
                                                                                     x : α
                                                                                     ⊢ Iff (Membership.mem (s.toFinset.attach.biUnion fun x => (t ↑x).toFinset) x)  …
                                                                                   -/
  Fintype.ofFinset (s.toFinset.attach.biUnion fun x => (t x).toFinset) fun x => by simp
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


instance fintypeBiUnion' [DecidableEq α] {ι : Type*} (s : Set ι) [Fintype s] (t : ι → Set α)
    [∀ i, Fintype (t i)] : Fintype (⋃ x ∈ s, t x) :=
                                                                      /-
                                                                        α : Type u
                                                                        β : Type v
                                                                        ι✝ : Sort w
                                                                        γ : Type x
                                                                        inst✝² : DecidableEq α
                                                                        ι : Type u_1
                                                                        s : Set ι
                                                                        inst✝¹ : Fintype ↑s
                                                                        t : ι → Set α
                                                                        inst✝ : (i : ι) → Fintype ↑(t i)
                                                                        ⊢ ∀ (x : α), Iff (Membership.mem (s.toFinset.biUnion fun x => (t x).toFinset)  …
                                                                      -/
  Fintype.ofFinset (s.toFinset.biUnion fun x => (t x).toFinset) <| by simp
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


instance fintypeEmpty : Fintype (∅ : Set α) :=
                           /-
                             α : Type u
                             β : Type v
                             ι : Sort w
                             γ : Type x
                             ⊢ ∀ (x : α), Iff (Membership.mem EmptyCollection.emptyCollection x) (Membershi …
                           -/
  Fintype.ofFinset ∅ <| by simp
                           /-
                             🎉 no goals
                           -/


instance fintypeSingleton (a : α) : Fintype ({a} : Set α) :=
                             /-
                               α : Type u
                               β : Type v
                               ι : Sort w
                               γ : Type x
                               a : α
                               ⊢ ∀ (x : α), Iff (Membership.mem (Singleton.singleton a) x) (Membership.mem (S …
                             -/
  Fintype.ofFinset {a} <| by simp
                             /-
                               🎉 no goals
                             -/


/-- A `Fintype` instance for inserting an element into a `Set` using the
corresponding `insert` function on `Finset`. This requires `DecidableEq α`.
There is also `Set.fintypeInsert'` when `a ∈ s` is decidable. -/
instance fintypeInsert (a : α) (s : Set α) [DecidableEq α] [Fintype s] :
    Fintype (insert a s : Set α) :=
                                               /-
                                                 α : Type u
                                                 β : Type v
                                                 ι : Sort w
                                                 γ : Type x
                                                 a : α
                                                 s : Set α
                                                 inst✝¹ : DecidableEq α
                                                 inst✝ : Fintype ↑s
                                                 ⊢ ∀ (x : α), Iff (Membership.mem (Insert.insert a s.toFinset) x) (Membership.m …
                                               -/
  Fintype.ofFinset (insert a s.toFinset) <| by simp
                                               /-
                                                 🎉 no goals
                                               -/


/-- A `Fintype` structure on `insert a s` when inserting a new element. -/
def fintypeInsertOfNotMem {a : α} (s : Set α) [Fintype s] (h : a ∉ s) :
    Fintype (insert a s : Set α) :=
                                                                  /-
                                                                    α : Type u
                                                                    β : Type v
                                                                    ι : Sort w
                                                                    γ : Type x
                                                                    a : α
                                                                    s : Set α
                                                                    inst✝ : Fintype ↑s
                                                                    h : Not (Membership.mem s a)
                                                                    ⊢ Not (Membership.mem s.toFinset.val a)
                                                                  -/
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
  Fintype.ofFinset ⟨a ::ₘ s.toFinset.1, s.toFinset.nodup.cons (by simp [h])⟩ <| by simp
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


/-- A `Fintype` structure on `insert a s` when inserting a pre-existing element. -/
def fintypeInsertOfMem {a : α} (s : Set α) [Fintype s] (h : a ∈ s) : Fintype (insert a s : Set α) :=
                                    /-
                                      α : Type u
                                      β : Type v
                                      ι : Sort w
                                      γ : Type x
                                      a : α
                                      s : Set α
                                      inst✝ : Fintype ↑s
                                      h : Membership.mem s a
                                      ⊢ ∀ (x : α), Iff (Membership.mem s.toFinset x) (Membership.mem (Insert.insert  …
                                    -/
  Fintype.ofFinset s.toFinset <| by simp [h]
                                    /-
                                      🎉 no goals
                                    -/


/-- The `Set.fintypeInsert` instance requires decidable equality, but when `a ∈ s`
is decidable for this particular `a` we can still get a `Fintype` instance by using
`Set.fintypeInsertOfNotMem` or `Set.fintypeInsertOfMem`.

This instance pre-dates `Set.fintypeInsert`, and it is less efficient.
When `Set.decidableMemOfFintype` is made a local instance, then this instance would
override `Set.fintypeInsert` if not for the fact that its priority has been
adjusted. See Note [lower instance priority]. -/
instance (priority := 100) fintypeInsert' (a : α) (s : Set α) [Decidable <| a ∈ s] [Fintype s] :
    Fintype (insert a s : Set α) :=
  if h : a ∈ s then fintypeInsertOfMem s h else fintypeInsertOfNotMem s h


instance fintypeImage [DecidableEq β] (s : Set α) (f : α → β) [Fintype s] : Fintype (f '' s) :=
                                              /-
                                                α : Type u
                                                β : Type v
                                                ι : Sort w
                                                γ : Type x
                                                inst✝¹ : DecidableEq β
                                                s : Set α
                                                f : α → β
                                                inst✝ : Fintype ↑s
                                                ⊢ ∀ (x : β), Iff (Membership.mem (Finset.image f s.toFinset) x) (Membership.me …
                                              -/
  Fintype.ofFinset (s.toFinset.image f) <| by simp
                                              /-
                                                🎉 no goals
                                              -/


/-- If a function `f` has a partial inverse `g` and the image of `s` under `f` is a set with
a `Fintype` instance, then `s` has a `Fintype` structure as well. -/
def fintypeOfFintypeImage (s : Set α) {f : α → β} {g} (I : IsPartialInv f g) [Fintype (f '' s)] :
    Fintype s :=
  Fintype.ofFinset ⟨_, (f '' s).toFinset.2.filterMap g <| injective_of_isPartialInv_right I⟩
    fun a => by
    suffices (∃ b x, f x = b ∧ g b = some a ∧ x ∈ s) ↔ a ∈ s by
      simpa [exists_and_left.symm, and_comm, and_left_comm, and_assoc]
    /-
      α : Type u
      β : Type v
      ι : Sort w
      γ : Type x
      s : Set α
      f : α → β
      g : β → Option α
      I : Function.IsPartialInv f g
      inst✝ : Fintype ↑(Set.image f s)
      a : α
      ⊢ Iff (Exists fun b => Exists fun x => And (Eq (f x) b) (And (Eq (g b) (Option …
    -/
    rw [exists_swap]
    /-
      α : Type u
      β : Type v
      ι : Sort w
      γ : Type x
      s : Set α
      f : α → β
      g : β → Option α
      I : Function.IsPartialInv f g
      inst✝ : Fintype ↑(Set.image f s)
      a : α
      ⊢ Iff (Exists fun y => Exists fun x => And (Eq (f y) x) (And (Eq (g x) (Option …
    -/
    suffices (∃ x, x ∈ s ∧ g (f x) = some a) ↔ a ∈ s by simpa [and_comm, and_left_comm, and_assoc]
    /-
      α : Type u
      β : Type v
      ι : Sort w
      γ : Type x
      s : Set α
      f : α → β
      g : β → Option α
      I : Function.IsPartialInv f g
      inst✝ : Fintype ↑(Set.image f s)
      a : α
      ⊢ Iff (Exists fun x => And (Membership.mem s x) (Eq (g (f x)) (Option.some a)) …
    -/
    simp [I _, (injective_of_isPartialInv I).eq_iff]
    /-
      🎉 no goals
    -/


instance fintypeMap {α β} [DecidableEq β] :
    ∀ (s : Set α) (f : α → β) [Fintype s], Fintype (f <$> s) :=
  Set.fintypeImage


instance fintypeLTNat (n : ℕ) : Fintype { i | i < n } :=
                                          /-
                                            α : Type u
                                            β : Type v
                                            ι : Sort w
                                            γ : Type x
                                            n : Nat
                                            ⊢ ∀ (x : Nat), Iff (Membership.mem (Finset.range n) x) (Membership.mem (setOf  …
                                          -/
  Fintype.ofFinset (Finset.range n) <| by simp
                                          /-
                                            🎉 no goals
                                          -/


instance fintypeLENat (n : ℕ) : Fintype { i | i ≤ n } := by
  /-
    α : Type u
    β : Type v
    ι : Sort w
    γ : Type x
    n : Nat
    ⊢ Fintype ↑(setOf fun i => LE.le i n)
  -/
  simpa [Nat.lt_succ_iff] using Set.fintypeLTNat (n + 1)
  /-
    🎉 no goals
  -/


/-- This is not an instance so that it does not conflict with the one
in `Mathlib/Order/LocallyFinite.lean`. -/
def Nat.fintypeIio (n : ℕ) : Fintype (Iio n) :=
  Set.fintypeLTNat n


instance fintypeMemFinset (s : Finset α) : Fintype { a | a ∈ s } :=
  Finset.fintypeCoeSort s


/-- Gives a `Set.Finite` for the `Finset` coerced to a `Set`.
This is a wrapper around `Set.toFinite`. -/
@[simp]
theorem finite_toSet (s : Finset α) : (s : Set α).Finite :=
  Set.toFinite _


theorem finite_toSet_toFinset (s : Finset α) : s.finite_toSet.toFinset = s := by
  /-
    α : Type u
    s : Finset α
    ⊢ Eq ⋯.toFinset s
  -/
  rw [toFinite_toFinset, toFinset_coe]
  /-
    🎉 no goals
  -/


@[simp]
theorem finite_toSet (s : Multiset α) : { x | x ∈ s }.Finite := by
  /-
    α : Type u
    s : Multiset α
    ⊢ (setOf fun x => Membership.mem s x).Finite
  -/
  classical simpa only [← Multiset.mem_toFinset] using s.toFinset.finite_toSet
  /-
    🎉 no goals
  -/


@[simp]
theorem finite_toSet_toFinset [DecidableEq α] (s : Multiset α) :
    s.finite_toSet.toFinset = s.toFinset := by
  /-
    α : Type u
    inst✝ : DecidableEq α
    s : Multiset α
    ⊢ Eq ⋯.toFinset s.toFinset
  -/
  ext x
  /-
    case h
    α : Type u
    inst✝ : DecidableEq α
    s : Multiset α
    x : α
    ⊢ Iff (Membership.mem ⋯.toFinset x) (Membership.mem s.toFinset x)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem List.finite_toSet (l : List α) : { x | x ∈ l }.Finite :=
  (show Multiset α from ⟦l⟧).finite_toSet


instance finite_union (s t : Set α) [Finite s] [Finite t] : Finite (s ∪ t : Set α) := by
  /-
    α : Type u
    β : Type v
    ι : Sort w
    γ : Type x
    s t : Set α
    inst✝¹ : Finite ↑s
    inst✝ : Finite ↑t
    ⊢ Finite ↑(Union.union s t)
  -/
  cases nonempty_fintype s
  /-
    case intro
    α : Type u
    β : Type v
    ι : Sort w
    γ : Type x
    s t : Set α
    inst✝¹ : Finite ↑s
    inst✝ : Finite ↑t
    val✝ : Fintype ↑s
    ⊢ Finite ↑(Union.union s t)
  -/
  cases nonempty_fintype t
  classical
  infer_instance


instance finite_sep (s : Set α) (p : α → Prop) [Finite s] : Finite ({ a ∈ s | p a } : Set α) := by
  /-
    α : Type u
    β : Type v
    ι : Sort w
    γ : Type x
    s : Set α
    p : α → Prop
    inst✝ : Finite ↑s
    ⊢ Finite ↑(setOf fun a => And (Membership.mem s a) (p a))
  -/
  cases nonempty_fintype s
  classical
  infer_instance


protected theorem subset (s : Set α) {t : Set α} [Finite s] (h : t ⊆ s) : Finite t := by
  /-
    α : Type u
    s t : Set α
    inst✝ : Finite ↑s
    h : HasSubset.Subset t s
    ⊢ Finite ↑t
  -/
  rw [← sep_eq_of_subset h]
  /-
    α : Type u
    s t : Set α
    inst✝ : Finite ↑s
    h : HasSubset.Subset t s
    ⊢ Finite ↑(setOf fun x => And (Membership.mem s x) (Membership.mem t x))
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance finite_inter_of_right (s t : Set α) [Finite t] : Finite (s ∩ t : Set α) :=
  Finite.Set.subset t inter_subset_right


instance finite_inter_of_left (s t : Set α) [Finite s] : Finite (s ∩ t : Set α) :=
  Finite.Set.subset s inter_subset_left


instance finite_diff (s t : Set α) [Finite s] : Finite (s \ t : Set α) :=
  Finite.Set.subset s diff_subset


instance finite_insert (a : α) (s : Set α) [Finite s] : Finite (insert a s : Set α) :=
  Finite.Set.finite_union {a} s


instance finite_image (s : Set α) (f : α → β) [Finite s] : Finite (f '' s) := by
  /-
    α : Type u
    β : Type v
    ι : Sort w
    γ : Type x
    s : Set α
    f : α → β
    inst✝ : Finite ↑s
    ⊢ Finite ↑(Set.image f s)
  -/
  cases nonempty_fintype s
  classical
  infer_instance


@[nontriviality]
theorem Finite.of_subsingleton [Subsingleton α] (s : Set α) : s.Finite :=
  s.toFinite


theorem finite_univ [Finite α] : (@univ α).Finite :=
  Set.toFinite _


theorem finite_univ_iff : (@univ α).Finite ↔ Finite α := (Equiv.Set.univ α).finite_iff


alias ⟨_root_.Finite.of_finite_univ, _⟩ := finite_univ_iff


theorem Finite.subset {s : Set α} (hs : s.Finite) {t : Set α} (ht : t ⊆ s) : t.Finite := by
  /-
    α : Type u
    s : Set α
    hs : s.Finite
    t : Set α
    ht : HasSubset.Subset t s
    ⊢ t.Finite
  -/
  have := hs.to_subtype
  /-
    α : Type u
    s : Set α
    hs : s.Finite
    t : Set α
    ht : HasSubset.Subset t s
    this : Finite ↑s
    ⊢ t.Finite
  -/
  exact Finite.Set.subset _ ht
  /-
    🎉 no goals
  -/


theorem Finite.union {s t : Set α} (hs : s.Finite) (ht : t.Finite) : (s ∪ t).Finite := by
  /-
    α : Type u
    s t : Set α
    hs : s.Finite
    ht : t.Finite
    ⊢ (Union.union s t).Finite
  -/
  rw [Set.Finite] at hs ht
  /-
    α : Type u
    s t : Set α
    hs : Finite ↑s
    ht : Finite ↑t
    ⊢ (Union.union s t).Finite
  -/
  apply toFinite
  /-
    🎉 no goals
  -/


theorem Finite.finite_of_compl {s : Set α} (hs : s.Finite) (hsc : sᶜ.Finite) : Finite α := by
  /-
    α : Type u
    s : Set α
    hs : s.Finite
    hsc : (HasCompl.compl s).Finite
    ⊢ Finite α
  -/
  rw [← finite_univ_iff, ← union_compl_self s]
  /-
    α : Type u
    s : Set α
    hs : s.Finite
    hsc : (HasCompl.compl s).Finite
    ⊢ (Union.union s (HasCompl.compl s)).Finite
  -/
  exact hs.union hsc
  /-
    🎉 no goals
  -/


theorem Finite.sup {s t : Set α} : s.Finite → t.Finite → (s ⊔ t).Finite :=
  Finite.union


theorem Finite.sep {s : Set α} (hs : s.Finite) (p : α → Prop) : { a ∈ s | p a }.Finite :=
  hs.subset <| sep_subset _ _


theorem Finite.inter_of_left {s : Set α} (hs : s.Finite) (t : Set α) : (s ∩ t).Finite :=
  hs.subset inter_subset_left


theorem Finite.inter_of_right {s : Set α} (hs : s.Finite) (t : Set α) : (t ∩ s).Finite :=
  hs.subset inter_subset_right


theorem Finite.inf_of_left {s : Set α} (h : s.Finite) (t : Set α) : (s ⊓ t).Finite :=
  h.inter_of_left t


theorem Finite.inf_of_right {s : Set α} (h : s.Finite) (t : Set α) : (t ⊓ s).Finite :=
  h.inter_of_right t


protected lemma Infinite.mono {s t : Set α} (h : s ⊆ t) : s.Infinite → t.Infinite :=
  mt fun ht ↦ ht.subset h


theorem Finite.diff {s : Set α} (hs : s.Finite) (t : Set α) : (s \ t).Finite :=
  hs.subset diff_subset


theorem Finite.of_diff {s t : Set α} (hd : (s \ t).Finite) (ht : t.Finite) : s.Finite :=
  (hd.union ht).subset <| subset_diff_union _ _


@[simp]
theorem finite_empty : (∅ : Set α).Finite :=
  toFinite _


protected theorem Infinite.nonempty {s : Set α} (h : s.Infinite) : s.Nonempty :=
  nonempty_iff_ne_empty.2 <| by
    /-
      α : Type u
      s : Set α
      h : s.Infinite
      ⊢ Ne s EmptyCollection.emptyCollection
    -/
    rintro rfl
    /-
      α : Type u
      h : EmptyCollection.emptyCollection.Infinite
      ⊢ False
    -/
    exact h finite_empty
    /-
      🎉 no goals
    -/


@[simp]
theorem finite_singleton (a : α) : ({a} : Set α).Finite :=
  toFinite _


@[simp]
protected theorem Finite.insert (a : α) {s : Set α} (hs : s.Finite) : (insert a s).Finite :=
  (finite_singleton a).union hs


theorem Finite.image {s : Set α} (f : α → β) (hs : s.Finite) : (f '' s).Finite := by
  /-
    α : Type u
    β : Type v
    s : Set α
    f : α → β
    hs : s.Finite
    ⊢ (Set.image f s).Finite
  -/
  have := hs.to_subtype
  /-
    α : Type u
    β : Type v
    s : Set α
    f : α → β
    hs : s.Finite
    this : Finite ↑s
    ⊢ (Set.image f s).Finite
  -/
  apply toFinite
  /-
    🎉 no goals
  -/


lemma Finite.of_surjOn {s : Set α} {t : Set β} (f : α → β) (hf : SurjOn f s t) (hs : s.Finite) :
    t.Finite := (hs.image _).subset hf


theorem Finite.map {α β} {s : Set α} : ∀ f : α → β, s.Finite → (f <$> s).Finite :=
  Finite.image


theorem Finite.of_finite_image {s : Set α} {f : α → β} (h : (f '' s).Finite) (hi : Set.InjOn f s) :
    s.Finite :=
  have := h.to_subtype
  .of_injective _ hi.bijOn_image.bijective.injective


theorem finite_of_finite_preimage (h : (f ⁻¹' s).Finite) (hs : s ⊆ range f) : s.Finite := by
  /-
    α : Type u
    β : Type v
    f : α → β
    s : Set β
    h : (Set.preimage f s).Finite
    hs : HasSubset.Subset s (Set.range f)
    ⊢ s.Finite
  -/
  rw [← image_preimage_eq_of_subset hs]
  /-
    α : Type u
    β : Type v
    f : α → β
    s : Set β
    h : (Set.preimage f s).Finite
    hs : HasSubset.Subset s (Set.range f)
    ⊢ (Set.image f (Set.preimage f s)).Finite
  -/
  exact Finite.image f h
  /-
    🎉 no goals
  -/


theorem Finite.of_preimage (h : (f ⁻¹' s).Finite) (hf : Surjective f) : s.Finite :=
  hf.image_preimage s ▸ h.image _


theorem Finite.preimage (I : Set.InjOn f (f ⁻¹' s)) (h : s.Finite) : (f ⁻¹' s).Finite :=
  (h.subset (image_preimage_subset f s)).of_finite_image I


protected lemma Infinite.preimage (hs : s.Infinite) (hf : s ⊆ range f) : (f ⁻¹' s).Infinite :=
  fun h ↦ hs <| finite_of_finite_preimage h hf


lemma Infinite.preimage' (hs : (s ∩ range f).Infinite) : (f ⁻¹' s).Infinite :=
  (hs.preimage inter_subset_right).mono <| preimage_mono inter_subset_left


theorem Finite.preimage_embedding {s : Set β} (f : α ↪ β) (h : s.Finite) : (f ⁻¹' s).Finite :=
  h.preimage fun _ _ _ _ h' => f.injective h'


theorem finite_lt_nat (n : ℕ) : Set.Finite { i | i < n } :=
  toFinite _


theorem finite_le_nat (n : ℕ) : Set.Finite { i | i ≤ n } :=
  toFinite _


theorem Finite.surjOn_iff_bijOn_of_mapsTo (hs : s.Finite) (hm : MapsTo f s s) :
    SurjOn f s s ↔ BijOn f s s := by
  /-
    α : Type u
    s : Set α
    f : α → α
    hs : s.Finite
    hm : Set.MapsTo f s s
    ⊢ Iff (Set.SurjOn f s s) (Set.BijOn f s s)
  -/
  refine ⟨fun h ↦ ⟨hm, ?_, h⟩, BijOn.surjOn⟩
  /-
    α : Type u
    s : Set α
    f : α → α
    hs : s.Finite
    hm : Set.MapsTo f s s
    h : Set.SurjOn f s s
    ⊢ Set.InjOn f s
  -/
  have : Finite s := finite_coe_iff.mpr hs
  /-
    α : Type u
    s : Set α
    f : α → α
    hs : s.Finite
    hm : Set.MapsTo f s s
    h : Set.SurjOn f s s
    this : Finite ↑s
    ⊢ Set.InjOn f s
  -/
  exact hm.restrict_inj.mp (Finite.injective_iff_surjective.mpr <| hm.restrict_surjective_iff.mpr h)
  /-
    🎉 no goals
  -/


theorem Finite.injOn_iff_bijOn_of_mapsTo (hs : s.Finite) (hm : MapsTo f s s) :
    InjOn f s ↔ BijOn f s s := by
  /-
    α : Type u
    s : Set α
    f : α → α
    hs : s.Finite
    hm : Set.MapsTo f s s
    ⊢ Iff (Set.InjOn f s) (Set.BijOn f s s)
  -/
  refine ⟨fun h ↦ ⟨hm, h, ?_⟩, BijOn.injOn⟩
  /-
    α : Type u
    s : Set α
    f : α → α
    hs : s.Finite
    hm : Set.MapsTo f s s
    h : Set.InjOn f s
    ⊢ Set.SurjOn f s s
  -/
  have : Finite s := finite_coe_iff.mpr hs
  /-
    α : Type u
    s : Set α
    f : α → α
    hs : s.Finite
    hm : Set.MapsTo f s s
    h : Set.InjOn f s
    this : Finite ↑s
    ⊢ Set.SurjOn f s s
  -/
  exact hm.restrict_surjective_iff.mp (Finite.injective_iff_surjective.mp <| hm.restrict_inj.mpr h)
  /-
    🎉 no goals
  -/


theorem finite_mem_finset (s : Finset α) : { a | a ∈ s }.Finite :=
  toFinite _


theorem Subsingleton.finite {s : Set α} (h : s.Subsingleton) : s.Finite :=
  h.induction_on finite_empty finite_singleton


theorem Infinite.nontrivial {s : Set α} (hs : s.Infinite) : s.Nontrivial :=
  not_subsingleton_iff.1 <| mt Subsingleton.finite hs


theorem finite_preimage_inl_and_inr {s : Set (α ⊕ β)} :
    (Sum.inl ⁻¹' s).Finite ∧ (Sum.inr ⁻¹' s).Finite ↔ s.Finite :=
  ⟨fun h => image_preimage_inl_union_image_preimage_inr s ▸ (h.1.image _).union (h.2.image _),
    fun h => ⟨h.preimage Sum.inl_injective.injOn, h.preimage Sum.inr_injective.injOn⟩⟩


theorem exists_finite_iff_finset {p : Set α → Prop} :
    (∃ s : Set α, s.Finite ∧ p s) ↔ ∃ s : Finset α, p ↑s :=
  ⟨fun ⟨_, hs, hps⟩ => ⟨hs.toFinset, hs.coe_toFinset.symm ▸ hps⟩, fun ⟨s, hs⟩ =>
    ⟨s, s.finite_toSet, hs⟩⟩


theorem exists_subset_image_finite_and {f : α → β} {s : Set α} {p : Set β → Prop} :
    (∃ t ⊆ f '' s, t.Finite ∧ p t) ↔ ∃ t ⊆ s, t.Finite ∧ p (f '' t) := by
  classical
  simp_rw [@and_comm (_ ⊆ _), and_assoc, exists_finite_iff_finset, @and_comm (p _),
    Finset.subset_set_image_iff]
  aesop


theorem finite_range_ite {p : α → Prop} [DecidablePred p] {f g : α → β} (hf : (range f).Finite)
    (hg : (range g).Finite) : (range fun x => if p x then f x else g x).Finite :=
  (hf.union hg).subset range_ite_subset


theorem finite_range_const {c : β} : (range fun _ : α => c).Finite :=
  (finite_singleton c).subset range_const_subset


instance Finite.inhabited : Inhabited { s : Set α // s.Finite } :=
  ⟨⟨∅, finite_empty⟩⟩


@[simp]
theorem finite_union {s t : Set α} : (s ∪ t).Finite ↔ s.Finite ∧ t.Finite :=
  ⟨fun h => ⟨h.subset subset_union_left, h.subset subset_union_right⟩, fun ⟨hs, ht⟩ =>
    hs.union ht⟩


theorem finite_image_iff {s : Set α} {f : α → β} (hi : InjOn f s) : (f '' s).Finite ↔ s.Finite :=
  ⟨fun h => h.of_finite_image hi, Finite.image _⟩


theorem univ_finite_iff_nonempty_fintype : (univ : Set α).Finite ↔ Nonempty (Fintype α) :=
  ⟨fun h => ⟨fintypeOfFiniteUniv h⟩, fun ⟨_i⟩ => finite_univ⟩

-- Porting note: moved `@[simp]` to `Set.toFinset_singleton` because `simp` can now simplify LHS

theorem Finite.toFinset_singleton {a : α} (ha : ({a} : Set α).Finite := finite_singleton _) :
    ha.toFinset = {a} :=
  Set.toFinite_toFinset _


@[simp]
theorem Finite.toFinset_insert [DecidableEq α] {s : Set α} {a : α} (hs : (insert a s).Finite) :
    hs.toFinset = insert a (hs.subset <| subset_insert _ _).toFinset :=
                   /-
                     α : Type u
                     inst✝ : DecidableEq α
                     s : Set α
                     a : α
                     hs : (Insert.insert a s).Finite
                     ⊢ ∀ (a_1 : α), Iff (Membership.mem hs.toFinset a_1) (Membership.mem (Insert.in …
                   -/
  Finset.ext <| by simp
                   /-
                     🎉 no goals
                   -/


theorem Finite.toFinset_insert' [DecidableEq α] {a : α} {s : Set α} (hs : s.Finite) :
    (hs.insert a).toFinset = insert a hs.toFinset :=
  Finite.toFinset_insert _


theorem finite_option {s : Set (Option α)} : s.Finite ↔ { x : α | some x ∈ s }.Finite :=
  ⟨fun h => h.preimage_embedding Embedding.some, fun h =>
    ((h.image some).insert none).subset fun x =>
      x.casesOn (fun _ => Or.inl rfl) fun _ hx => Or.inr <| mem_image_of_mem _ hx⟩


@[elab_as_elim]
theorem Finite.induction_on {C : Set α → Prop} {s : Set α} (h : s.Finite) (H0 : C ∅)
    (H1 : ∀ {a s}, a ∉ s → Set.Finite s → C s → C (insert a s)) : C s := by
  /-
    α : Type u
    C : Set α → Prop
    s : Set α
    h : s.Finite
    H0 : C EmptyCollection.emptyCollection
    H1 : ∀ {a : α} {s : Set α}, Not (Membership.mem s a) → s.Finite → C s → C (Ins …
    ⊢ C s
  -/
  lift s to Finset α using h
  /-
    case intro
    α : Type u
    C : Set α → Prop
    H0 : C EmptyCollection.emptyCollection
    H1 : ∀ {a : α} {s : Set α}, Not (Membership.mem s a) → s.Finite → C s → C (Ins …
    s : Finset α
    ⊢ C ↑s
  -/
  induction' s using Finset.cons_induction_on with a s ha hs
    /-
      case intro.h₁
      α : Type u
      C : Set α → Prop
      H0 : C EmptyCollection.emptyCollection
      H1 : ∀ {a : α} {s : Set α}, Not (Membership.mem s a) → s.Finite → C s → C (Ins …
      ⊢ C ↑EmptyCollection.emptyCollection
    -/
  · rwa [Finset.coe_empty]
    /-
      🎉 no goals
    -/
    /-
      case intro.h₂
      α : Type u
      C : Set α → Prop
      H0 : C EmptyCollection.emptyCollection
      H1 : ∀ {a : α} {s : Set α}, Not (Membership.mem s a) → s.Finite → C s → C (Ins …
      a : α
      s : Finset α
      ha : Not (Membership.mem s a)
      hs : C ↑s
      ⊢ C ↑(Finset.cons a s ha)
    -/
  · rw [Finset.coe_cons]
    /-
      case intro.h₂
      α : Type u
      C : Set α → Prop
      H0 : C EmptyCollection.emptyCollection
      H1 : ∀ {a : α} {s : Set α}, Not (Membership.mem s a) → s.Finite → C s → C (Ins …
      a : α
      s : Finset α
      ha : Not (Membership.mem s a)
      hs : C ↑s
      ⊢ C (Insert.insert a ↑s)
    -/
    exact @H1 a s ha (Set.toFinite _) hs
    /-
      🎉 no goals
    -/


/-- Analogous to `Finset.induction_on'`. -/
@[elab_as_elim]
theorem Finite.induction_on' {C : Set α → Prop} {S : Set α} (h : S.Finite) (H0 : C ∅)
    (H1 : ∀ {a s}, a ∈ S → s ⊆ S → a ∉ s → C s → C (insert a s)) : C S := by
  /-
    α : Type u
    C : Set α → Prop
    S : Set α
    h : S.Finite
    H0 : C EmptyCollection.emptyCollection
    H1 : ∀ {a : α} {s : Set α}, Membership.mem S a → HasSubset.Subset s S → Not (M …
    ⊢ C S
  -/
  refine @Set.Finite.induction_on α (fun s => s ⊆ S → C s) S h (fun _ => H0) ?_ Subset.rfl
  /-
    α : Type u
    C : Set α → Prop
    S : Set α
    h : S.Finite
    H0 : C EmptyCollection.emptyCollection
    H1 : ∀ {a : α} {s : Set α}, Membership.mem S a → HasSubset.Subset s S → Not (M …
    ⊢ ∀ {a : α} {s : Set α}, Not (Membership.mem s a) → s.Finite → (fun s => HasSu …
  -/
  intro a s has _ hCs haS
  /-
    α : Type u
    C : Set α → Prop
    S : Set α
    h : S.Finite
    H0 : C EmptyCollection.emptyCollection
    H1 : ∀ {a : α} {s : Set α}, Membership.mem S a → HasSubset.Subset s S → Not (M …
    a : α
    s : Set α
    has : Not (Membership.mem s a)
    a✝ : s.Finite
    hCs : HasSubset.Subset s S → C s
    haS : HasSubset.Subset (Insert.insert a s) S
    ⊢ C (Insert.insert a s)
  -/
  rw [insert_subset_iff] at haS
  /-
    α : Type u
    C : Set α → Prop
    S : Set α
    h : S.Finite
    H0 : C EmptyCollection.emptyCollection
    H1 : ∀ {a : α} {s : Set α}, Membership.mem S a → HasSubset.Subset s S → Not (M …
    a : α
    s : Set α
    has : Not (Membership.mem s a)
    a✝ : s.Finite
    hCs : HasSubset.Subset s S → C s
    haS : And (Membership.mem S a) (HasSubset.Subset s S)
    ⊢ C (Insert.insert a s)
  -/
  exact H1 haS.1 haS.2 has (hCs haS.2)
  /-
    🎉 no goals
  -/


@[elab_as_elim]
theorem Finite.dinduction_on {C : ∀ s : Set α, s.Finite → Prop} (s : Set α) (h : s.Finite)
    (H0 : C ∅ finite_empty)
    (H1 : ∀ {a s}, a ∉ s → ∀ h : Set.Finite s, C s h → C (insert a s) (h.insert a)) : C s h :=
  have : ∀ h : s.Finite, C s h :=
    Finite.induction_on h (fun _ => H0) fun has hs ih _ => H1 has hs (ih _)
  this h


/-- If `P` is some relation between terms of `γ` and sets in `γ`, such that every finite set
`t : Set γ` has some `c : γ` related to it, then there is a recursively defined sequence `u` in `γ`
so `u n` is related to the image of `{0, 1, ..., n-1}` under `u`.

(We use this later to show sequentially compact sets are totally bounded.)
-/
theorem seq_of_forall_finite_exists {γ : Type*} {P : γ → Set γ → Prop}
    (h : ∀ t : Set γ, t.Finite → ∃ c, P c t) : ∃ u : ℕ → γ, ∀ n, P (u n) (u '' Iio n) := by
  /-
    γ : Type u_1
    P : γ → Set γ → Prop
    h : ∀ (t : Set γ), t.Finite → Exists fun c => P c t
    ⊢ Exists fun u => ∀ (n : Nat), P (u n) (Set.image u (Set.Iio n))
  -/
  haveI : Nonempty γ := (h ∅ finite_empty).nonempty
  /-
    γ : Type u_1
    P : γ → Set γ → Prop
    h : ∀ (t : Set γ), t.Finite → Exists fun c => P c t
    this : Nonempty γ
    ⊢ Exists fun u => ∀ (n : Nat), P (u n) (Set.image u (Set.Iio n))
  -/
  choose! c hc using h
  /-
    γ : Type u_1
    P : γ → Set γ → Prop
    this : Nonempty γ
    c : Set γ → γ
    hc : ∀ (t : Set γ), t.Finite → P (c t) t
    ⊢ Exists fun u => ∀ (n : Nat), P (u n) (Set.image u (Set.Iio n))
  -/
  set f : (n : ℕ) → (g : (m : ℕ) → m < n → γ) → γ := fun n g => c (range fun k : Iio n => g k.1 k.2)
  /-
    γ : Type u_1
    P : γ → Set γ → Prop
    this : Nonempty γ
    c : Set γ → γ
    hc : ∀ (t : Set γ), t.Finite → P (c t) t
    f : (n : Nat) → ((m : Nat) → LT.lt m n → γ) → γ := fun n g => c (Set.range fun …
    ⊢ Exists fun u => ∀ (n : Nat), P (u n) (Set.image u (Set.Iio n))
  -/
  set u : ℕ → γ := fun n => Nat.strongRecOn' n f
  /-
    γ : Type u_1
    P : γ → Set γ → Prop
    this : Nonempty γ
    c : Set γ → γ
    hc : ∀ (t : Set γ), t.Finite → P (c t) t
    f : (n : Nat) → ((m : Nat) → LT.lt m n → γ) → γ := fun n g => c (Set.range fun …
    u : Nat → γ := fun n => n.strongRecOn' f
    ⊢ Exists fun u => ∀ (n : Nat), P (u n) (Set.image u (Set.Iio n))
  -/
  refine ⟨u, fun n => ?_⟩
  /-
    γ : Type u_1
    P : γ → Set γ → Prop
    this : Nonempty γ
    c : Set γ → γ
    hc : ∀ (t : Set γ), t.Finite → P (c t) t
    f : (n : Nat) → ((m : Nat) → LT.lt m n → γ) → γ := fun n g => c (Set.range fun …
    u : Nat → γ := fun n => n.strongRecOn' f
    n : Nat
    ⊢ P (u n) (Set.image u (Set.Iio n))
  -/
  convert hc (u '' Iio n) ((finite_lt_nat _).image _)
  /-
    case h.e'_1
    γ : Type u_1
    P : γ → Set γ → Prop
    this : Nonempty γ
    c : Set γ → γ
    hc : ∀ (t : Set γ), t.Finite → P (c t) t
    f : (n : Nat) → ((m : Nat) → LT.lt m n → γ) → γ := fun n g => c (Set.range fun …
    u : Nat → γ := fun n => n.strongRecOn' f
    n : Nat
    ⊢ Eq (u n) (c (Set.image u (Set.Iio n)))
  -/
  rw [image_eq_range]
  /-
    case h.e'_1
    γ : Type u_1
    P : γ → Set γ → Prop
    this : Nonempty γ
    c : Set γ → γ
    hc : ∀ (t : Set γ), t.Finite → P (c t) t
    f : (n : Nat) → ((m : Nat) → LT.lt m n → γ) → γ := fun n g => c (Set.range fun …
    u : Nat → γ := fun n => n.strongRecOn' f
    n : Nat
    ⊢ Eq (u n) (c (Set.range fun x => u ↑x))
  -/
  exact Nat.strongRecOn'_beta
  /-
    🎉 no goals
  -/


theorem empty_card : Fintype.card (∅ : Set α) = 0 :=
  rfl


theorem empty_card' {h : Fintype.{u} (∅ : Set α)} : @Fintype.card (∅ : Set α) h = 0 := by
  /-
    α : Type u
    h : Fintype ↑EmptyCollection.emptyCollection
    ⊢ Eq (Fintype.card ↑EmptyCollection.emptyCollection) 0
  -/
  simp
  /-
    🎉 no goals
  -/


theorem card_fintypeInsertOfNotMem {a : α} (s : Set α) [Fintype s] (h : a ∉ s) :
    @Fintype.card _ (fintypeInsertOfNotMem s h) = Fintype.card s + 1 := by
  /-
    α : Type u
    a : α
    s : Set α
    inst✝ : Fintype ↑s
    h : Not (Membership.mem s a)
    ⊢ Eq (Fintype.card ↑(Insert.insert a s)) (HAdd.hAdd (Fintype.card ↑s) 1)
  -/
  simp [fintypeInsertOfNotMem, Fintype.card_ofFinset]
  /-
    🎉 no goals
  -/


@[simp]
theorem card_insert {a : α} (s : Set α) [Fintype s] (h : a ∉ s)
    {d : Fintype.{u} (insert a s : Set α)} : @Fintype.card _ d = Fintype.card s + 1 := by
  /-
    α : Type u
    a : α
    s : Set α
    inst✝ : Fintype ↑s
    h : Not (Membership.mem s a)
    d : Fintype ↑(Insert.insert a s)
    ⊢ Eq (Fintype.card ↑(Insert.insert a s)) (HAdd.hAdd (Fintype.card ↑s) 1)
  -/
  rw [← card_fintypeInsertOfNotMem s h]; congr!
                                         /-
                                           🎉 no goals
                                         -/


theorem card_image_of_inj_on {s : Set α} [Fintype s] {f : α → β} [Fintype (f '' s)]
    (H : ∀ x ∈ s, ∀ y ∈ s, f x = f y → x = y) : Fintype.card (f '' s) = Fintype.card s :=
  haveI := Classical.propDecidable
  calc
                                                                                       /-
                                                                                         α : Type u
                                                                                         β : Type v
                                                                                         s : Set α
                                                                                         inst✝¹ : Fintype ↑s
                                                                                         f : α → β
                                                                                         inst✝ : Fintype ↑(Set.image f s)
                                                                                         H : ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem s y → Eq (f x) ( …
                                                                                         this : (a : Prop) → Decidable a
                                                                                         ⊢ ∀ (x : β), Iff (Membership.mem (Finset.image f s.toFinset) x) (Membership.me …
                                                                                       -/
    Fintype.card (f '' s) = (s.toFinset.image f).card := Fintype.card_of_finset' _ (by simp)
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/
    _ = s.toFinset.card :=
      Finset.card_image_of_injOn fun x hx y hy hxy =>
        H x (mem_toFinset.1 hx) y (mem_toFinset.1 hy) hxy
    _ = Fintype.card s := (Fintype.card_of_finset' _ fun _ => mem_toFinset).symm


theorem card_image_of_injective (s : Set α) [Fintype s] {f : α → β} [Fintype (f '' s)]
    (H : Function.Injective f) : Fintype.card (f '' s) = Fintype.card s :=
  card_image_of_inj_on fun _ _ _ _ h => H h


@[simp]
theorem card_singleton (a : α) : Fintype.card ({a} : Set α) = 1 :=
  Fintype.card_ofSubsingleton _


theorem card_lt_card {s t : Set α} [Fintype s] [Fintype t] (h : s ⊂ t) :
    Fintype.card s < Fintype.card t :=
  Fintype.card_lt_of_injective_not_surjective (Set.inclusion h.1) (Set.inclusion_injective h.1)
    fun hst => (ssubset_iff_subset_ne.1 h).2 (eq_of_inclusion_surjective hst)


theorem card_le_card {s t : Set α} [Fintype s] [Fintype t] (hsub : s ⊆ t) :
    Fintype.card s ≤ Fintype.card t :=
  Fintype.card_le_of_injective (Set.inclusion hsub) (Set.inclusion_injective hsub)


theorem eq_of_subset_of_card_le {s t : Set α} [Fintype s] [Fintype t] (hsub : s ⊆ t)
    (hcard : Fintype.card t ≤ Fintype.card s) : s = t :=
  (eq_or_ssubset_of_subset hsub).elim id fun h => absurd hcard <| not_le_of_lt <| card_lt_card h


theorem card_range_of_injective [Fintype α] {f : α → β} (hf : Injective f) [Fintype (range f)] :
    Fintype.card (range f) = Fintype.card α :=
  Eq.symm <| Fintype.card_congr <| Equiv.ofInjective f hf


theorem Finite.card_toFinset {s : Set α} [Fintype s] (h : s.Finite) :
    h.toFinset.card = Fintype.card s :=
  Eq.symm <| Fintype.card_of_finset' _ fun _ ↦ h.mem_toFinset


theorem card_ne_eq [Fintype α] (a : α) [Fintype { x : α | x ≠ a }] :
    Fintype.card { x : α | x ≠ a } = Fintype.card α - 1 := by
  /-
    α : Type u
    inst✝¹ : Fintype α
    a : α
    inst✝ : Fintype ↑(setOf fun x => Ne x a)
    ⊢ Eq (Fintype.card ↑(setOf fun x => Ne x a)) (HSub.hSub (Fintype.card α) 1)
  -/
  haveI := Classical.decEq α
  rw [← toFinset_card, toFinset_setOf, Finset.filter_ne',
    Finset.card_erase_of_mem (Finset.mem_univ _), Finset.card_univ]


theorem infinite_univ_iff : (@univ α).Infinite ↔ Infinite α := by
  /-
    α : Type u
    ⊢ Iff Set.univ.Infinite (Infinite α)
  -/
  rw [Set.Infinite, finite_univ_iff, not_finite_iff_infinite]
  /-
    🎉 no goals
  -/


theorem infinite_univ [h : Infinite α] : (@univ α).Infinite :=
  infinite_univ_iff.2 h


lemma Infinite.exists_not_mem_finite (hs : s.Infinite) (ht : t.Finite) : ∃ a, a ∈ s ∧ a ∉ t := by
  /-
    α : Type u
    s t : Set α
    hs : s.Infinite
    ht : t.Finite
    ⊢ Exists fun a => And (Membership.mem s a) (Not (Membership.mem t a))
  -/
  by_contra! h; exact hs <| ht.subset h
                /-
                  🎉 no goals
                -/


lemma Infinite.exists_not_mem_finset (hs : s.Infinite) (t : Finset α) : ∃ a ∈ s, a ∉ t :=
  hs.exists_not_mem_finite t.finite_toSet


lemma Finite.exists_not_mem (hs : s.Finite) : ∃ a, a ∉ s := by
  /-
    α : Type u
    s : Set α
    inst✝ : Infinite α
    hs : s.Finite
    ⊢ Exists fun a => Not (Membership.mem s a)
  -/
  by_contra! h; exact infinite_univ (hs.subset fun a _ ↦ h _)
                /-
                  🎉 no goals
                -/


lemma _root_.Finset.exists_not_mem (s : Finset α) : ∃ a, a ∉ s := s.finite_toSet.exists_not_mem


/-- Embedding of `ℕ` into an infinite set. -/
noncomputable def Infinite.natEmbedding (s : Set α) (h : s.Infinite) : ℕ ↪ s :=
  h.to_subtype.natEmbedding


theorem Infinite.exists_subset_card_eq {s : Set α} (hs : s.Infinite) (n : ℕ) :
    ∃ t : Finset α, ↑t ⊆ s ∧ t.card = n :=
                                                                            /-
                                                                              α : Type u
                                                                              s : Set α
                                                                              hs : s.Infinite
                                                                              n : Nat
                                                                              ⊢ And (HasSubset.Subset (↑(Finset.map (Function.Embedding.subtype fun x => Mem …
                                                                            -/
  ⟨((Finset.range n).map (hs.natEmbedding _)).map (Embedding.subtype _), by simp⟩
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


theorem infinite_of_finite_compl [Infinite α] {s : Set α} (hs : sᶜ.Finite) : s.Infinite := fun h =>
                                 /-
                                   α : Type u
                                   inst✝ : Infinite α
                                   s : Set α
                                   hs : (HasCompl.compl s).Finite
                                   h : s.Finite
                                   ⊢ Set.univ.Finite
                                 -/
  Set.infinite_univ (α := α) (by simpa using hs.union h)
                                 /-
                                   🎉 no goals
                                 -/


theorem Finite.infinite_compl [Infinite α] {s : Set α} (hs : s.Finite) : sᶜ.Infinite := fun h =>
                                 /-
                                   α : Type u
                                   inst✝ : Infinite α
                                   s : Set α
                                   hs : s.Finite
                                   h : (HasCompl.compl s).Finite
                                   ⊢ Set.univ.Finite
                                 -/
  Set.infinite_univ (α := α) (by simpa using hs.union h)
                                 /-
                                   🎉 no goals
                                 -/


theorem Infinite.diff {s t : Set α} (hs : s.Infinite) (ht : t.Finite) : (s \ t).Infinite := fun h =>
  hs <| h.of_diff ht


@[simp]
theorem infinite_union {s t : Set α} : (s ∪ t).Infinite ↔ s.Infinite ∨ t.Infinite := by
  /-
    α : Type u
    s t : Set α
    ⊢ Iff (Union.union s t).Infinite (Or s.Infinite t.Infinite)
  -/
  simp only [Set.Infinite, finite_union, not_and_or]
  /-
    🎉 no goals
  -/


theorem Infinite.of_image (f : α → β) {s : Set α} (hs : (f '' s).Infinite) : s.Infinite :=
  mt (Finite.image f) hs


theorem infinite_image_iff {s : Set α} {f : α → β} (hi : InjOn f s) :
    (f '' s).Infinite ↔ s.Infinite :=
  not_congr <| finite_image_iff hi


theorem infinite_range_iff {f : α → β} (hi : Injective f) :
    (range f).Infinite ↔ Infinite α := by
  /-
    α : Type u
    β : Type v
    f : α → β
    hi : Function.Injective f
    ⊢ Iff (Set.range f).Infinite (Infinite α)
  -/
  rw [← image_univ, infinite_image_iff hi.injOn, infinite_univ_iff]
  /-
    🎉 no goals
  -/


protected alias ⟨_, Infinite.image⟩ := infinite_image_iff


theorem infinite_of_injOn_mapsTo {s : Set α} {t : Set β} {f : α → β} (hi : InjOn f s)
    (hm : MapsTo f s t) (hs : s.Infinite) : t.Infinite :=
  ((infinite_image_iff hi).2 hs).mono (mapsTo'.mp hm)


theorem Infinite.exists_ne_map_eq_of_mapsTo {s : Set α} {t : Set β} {f : α → β} (hs : s.Infinite)
    (hf : MapsTo f s t) (ht : t.Finite) : ∃ x ∈ s, ∃ y ∈ s, x ≠ y ∧ f x = f y := by
  /-
    α : Type u
    β : Type v
    s : Set α
    t : Set β
    f : α → β
    hs : s.Infinite
    hf : Set.MapsTo f s t
    ht : t.Finite
    ⊢ Exists fun x => And (Membership.mem s x) (Exists fun y => And (Membership.me …
  -/
  contrapose! ht
  /-
    α : Type u
    β : Type v
    s : Set α
    t : Set β
    f : α → β
    hs : s.Infinite
    hf : Set.MapsTo f s t
    ht : ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem s y → Ne x y →  …
    ⊢ Not t.Finite
  -/
  exact infinite_of_injOn_mapsTo (fun x hx y hy => not_imp_not.1 (ht x hx y hy)) hf hs
  /-
    🎉 no goals
  -/


theorem infinite_range_of_injective [Infinite α] {f : α → β} (hi : Injective f) :
    (range f).Infinite := by
  /-
    α : Type u
    β : Type v
    inst✝ : Infinite α
    f : α → β
    hi : Function.Injective f
    ⊢ (Set.range f).Infinite
  -/
  rw [← image_univ, infinite_image_iff hi.injOn]
  /-
    α : Type u
    β : Type v
    inst✝ : Infinite α
    f : α → β
    hi : Function.Injective f
    ⊢ Set.univ.Infinite
  -/
  exact infinite_univ
  /-
    🎉 no goals
  -/


theorem infinite_of_injective_forall_mem [Infinite α] {s : Set β} {f : α → β} (hi : Injective f)
    (hf : ∀ x : α, f x ∈ s) : s.Infinite := by
  /-
    α : Type u
    β : Type v
    inst✝ : Infinite α
    s : Set β
    f : α → β
    hi : Function.Injective f
    hf : ∀ (x : α), Membership.mem s (f x)
    ⊢ s.Infinite
  -/
  rw [← range_subset_iff] at hf
  /-
    α : Type u
    β : Type v
    inst✝ : Infinite α
    s : Set β
    f : α → β
    hi : Function.Injective f
    hf : HasSubset.Subset (Set.range f) s
    ⊢ s.Infinite
  -/
  exact (infinite_range_of_injective hi).mono hf
  /-
    🎉 no goals
  -/


theorem not_injOn_infinite_finite_image {f : α → β} {s : Set α} (h_inf : s.Infinite)
    (h_fin : (f '' s).Finite) : ¬InjOn f s := by
  /-
    α : Type u
    β : Type v
    f : α → β
    s : Set α
    h_inf : s.Infinite
    h_fin : (Set.image f s).Finite
    ⊢ Not (Set.InjOn f s)
  -/
  have : Finite (f '' s) := finite_coe_iff.mpr h_fin
  /-
    α : Type u
    β : Type v
    f : α → β
    s : Set α
    h_inf : s.Infinite
    h_fin : (Set.image f s).Finite
    this : Finite ↑(Set.image f s)
    ⊢ Not (Set.InjOn f s)
  -/
  have : Infinite s := infinite_coe_iff.mpr h_inf
  have h := not_injective_infinite_finite
            ((f '' s).codRestrict (s.restrict f) fun x => ⟨x, x.property, rfl⟩)
  /-
    α : Type u
    β : Type v
    f : α → β
    s : Set α
    h_inf : s.Infinite
    h_fin : (Set.image f s).Finite
    this✝ : Finite ↑(Set.image f s)
    this : Infinite ↑s
    h : Not (Function.Injective (Set.codRestrict (s.restrict f) (Set.image f s) ⋯))
    ⊢ Not (Set.InjOn f s)
  -/
  contrapose! h
  /-
    α : Type u
    β : Type v
    f : α → β
    s : Set α
    h_inf : s.Infinite
    h_fin : (Set.image f s).Finite
    this✝ : Finite ↑(Set.image f s)
    this : Infinite ↑s
    h : Set.InjOn f s
    ⊢ Function.Injective (Set.codRestrict (s.restrict f) (Set.image f s) ⋯)
  -/
  rwa [injective_codRestrict, ← injOn_iff_injective]
  /-
    🎉 no goals
  -/


theorem infinite_of_forall_exists_gt (h : ∀ a, ∃ b ∈ s, a < b) : s.Infinite := by
  /-
    α : Type u
    inst✝¹ : Preorder α
    inst✝ : Nonempty α
    s : Set α
    h : ∀ (a : α), Exists fun b => And (Membership.mem s b) (LT.lt a b)
    ⊢ s.Infinite
  -/
  inhabit α
  /-
    α : Type u
    inst✝¹ : Preorder α
    inst✝ : Nonempty α
    s : Set α
    h : ∀ (a : α), Exists fun b => And (Membership.mem s b) (LT.lt a b)
    inhabited_h : Inhabited α
    ⊢ s.Infinite
  -/
  set f : ℕ → α := fun n => Nat.recOn n (h default).choose fun _ a => (h a).choose
  /-
    α : Type u
    inst✝¹ : Preorder α
    inst✝ : Nonempty α
    s : Set α
    h : ∀ (a : α), Exists fun b => And (Membership.mem s b) (LT.lt a b)
    inhabited_h : Inhabited α
    f : Nat → α := fun n => Nat.recOn n ⋯.choose fun x a => ⋯.choose
    ⊢ s.Infinite
  -/
  have hf : ∀ n, f n ∈ s := by rintro (_ | _) <;> exact (h _).choose_spec.1
  exact infinite_of_injective_forall_mem
    (strictMono_nat_of_lt_succ fun n => (h _).choose_spec.2).injective hf


theorem infinite_of_forall_exists_lt (h : ∀ a, ∃ b ∈ s, b < a) : s.Infinite :=
  infinite_of_forall_exists_gt (α := αᵒᵈ) h


theorem finite_isTop (α : Type*) [PartialOrder α] : { x : α | IsTop x }.Finite :=
  (subsingleton_isTop α).finite


theorem finite_isBot (α : Type*) [PartialOrder α] : { x : α | IsBot x }.Finite :=
  (subsingleton_isBot α).finite


theorem Infinite.exists_lt_map_eq_of_mapsTo [LinearOrder α] {s : Set α} {t : Set β} {f : α → β}
    (hs : s.Infinite) (hf : MapsTo f s t) (ht : t.Finite) : ∃ x ∈ s, ∃ y ∈ s, x < y ∧ f x = f y :=
  let ⟨x, hx, y, hy, hxy, hf⟩ := hs.exists_ne_map_eq_of_mapsTo hf ht
  hxy.lt_or_lt.elim (fun hxy => ⟨x, hx, y, hy, hxy, hf⟩) fun hyx => ⟨y, hy, x, hx, hyx, hf.symm⟩


theorem Finite.exists_lt_map_eq_of_forall_mem [LinearOrder α] [Infinite α] {t : Set β} {f : α → β}
    (hf : ∀ a, f a ∈ t) (ht : t.Finite) : ∃ a b, a < b ∧ f a = f b := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : LinearOrder α
    inst✝ : Infinite α
    t : Set β
    f : α → β
    hf : ∀ (a : α), Membership.mem t (f a)
    ht : t.Finite
    ⊢ Exists fun a => Exists fun b => And (LT.lt a b) (Eq (f a) (f b))
  -/
  rw [← mapsTo_univ_iff] at hf
  /-
    α : Type u
    β : Type v
    inst✝¹ : LinearOrder α
    inst✝ : Infinite α
    t : Set β
    f : α → β
    hf : Set.MapsTo f Set.univ t
    ht : t.Finite
    ⊢ Exists fun a => Exists fun b => And (LT.lt a b) (Eq (f a) (f b))
  -/
  obtain ⟨a, -, b, -, h⟩ := infinite_univ.exists_lt_map_eq_of_mapsTo hf ht
  /-
    case intro.intro.intro.intro
    α : Type u
    β : Type v
    inst✝¹ : LinearOrder α
    inst✝ : Infinite α
    t : Set β
    f : α → β
    hf : Set.MapsTo f Set.univ t
    ht : t.Finite
    a b : α
    h : And (LT.lt a b) (Eq (f a) (f b))
    ⊢ Exists fun a => Exists fun b => And (LT.lt a b) (Eq (f a) (f b))
  -/
  exact ⟨a, b, h⟩
  /-
    🎉 no goals
  -/


theorem finite_range_findGreatest {P : α → ℕ → Prop} [∀ x, DecidablePred (P x)] {b : ℕ} :
    (range fun x => Nat.findGreatest (P x) b).Finite :=
  (finite_le_nat b).subset <| range_subset_iff.2 fun _ => Nat.findGreatest_le _


theorem Finite.exists_maximal_wrt [PartialOrder β] (f : α → β) (s : Set α) (h : s.Finite)
    (hs : s.Nonempty) : ∃ a ∈ s, ∀ a' ∈ s, f a ≤ f a' → f a = f a' := by
  induction s, h using Set.Finite.dinduction_on with
  | H0 => exact absurd hs not_nonempty_empty
  | @H1 a s his _ ih =>
    rcases s.eq_empty_or_nonempty with h | h
    · use a
      simp [h]
    rcases ih h with ⟨b, hb, ih⟩
    by_cases h : f b ≤ f a
    · refine ⟨a, Set.mem_insert _ _, fun c hc hac => le_antisymm hac ?_⟩
      rcases Set.mem_insert_iff.1 hc with (rfl | hcs)
      · rfl
      · rwa [← ih c hcs (le_trans h hac)]
    · refine ⟨b, Set.mem_insert_of_mem _ hb, fun c hc hbc => ?_⟩
      rcases Set.mem_insert_iff.1 hc with (rfl | hcs)
      · exact (h hbc).elim
      · exact ih c hcs hbc


/-- A version of `Finite.exists_maximal_wrt` with the (weaker) hypothesis that the image of `s`
  is finite rather than `s` itself. -/
theorem Finite.exists_maximal_wrt' [PartialOrder β] (f : α → β) (s : Set α) (h : (f '' s).Finite)
    (hs : s.Nonempty) : (∃ a ∈ s, ∀ (a' : α), a' ∈ s → f a ≤ f a' → f a = f a') := by
  /-
    α : Type u
    β : Type v
    inst✝ : PartialOrder β
    f : α → β
    s : Set α
    h : (Set.image f s).Finite
    hs : s.Nonempty
    ⊢ Exists fun a => And (Membership.mem s a) (∀ (a' : α), Membership.mem s a' →  …
  -/
  obtain ⟨_, ⟨a, ha, rfl⟩, hmax⟩ := Finite.exists_maximal_wrt id (f '' s) h (hs.image f)
  /-
    case intro.intro.intro.intro
    α : Type u
    β : Type v
    inst✝ : PartialOrder β
    f : α → β
    s : Set α
    h : (Set.image f s).Finite
    hs : s.Nonempty
    a : α
    ha : Membership.mem s a
    hmax : ∀ (a' : β), Membership.mem (Set.image f s) a' → LE.le (id (f a)) (id a' …
    ⊢ Exists fun a => And (Membership.mem s a) (∀ (a' : α), Membership.mem s a' →  …
  -/
  exact ⟨a, ha, fun a' ha' hf ↦ hmax _ (mem_image_of_mem f ha') hf⟩
  /-
    🎉 no goals
  -/


theorem Finite.exists_minimal_wrt [PartialOrder β] (f : α → β) (s : Set α) (h : s.Finite)
    (hs : s.Nonempty) : ∃ a ∈ s, ∀ a' ∈ s, f a' ≤ f a → f a = f a' :=
  Finite.exists_maximal_wrt (β := βᵒᵈ) f s h hs


/-- A version of `Finite.exists_minimal_wrt` with the (weaker) hypothesis that the image of `s`
  is finite rather than `s` itself. -/
lemma Finite.exists_minimal_wrt' [PartialOrder β] (f : α → β) (s : Set α) (h : (f '' s).Finite)
    (hs : s.Nonempty) : (∃ a ∈ s, ∀ (a' : α), a' ∈ s → f a' ≤ f a → f a = f a') :=
  Set.Finite.exists_maximal_wrt' (β := βᵒᵈ) f s h hs


lemma exists_card_eq [Infinite α] : ∀ n : ℕ, ∃ s : Finset α, s.card = n
  | 0 => ⟨∅, card_empty⟩
  | n + 1 => by
    classical
    obtain ⟨s, rfl⟩ := exists_card_eq n
    obtain ⟨a, ha⟩ := s.exists_not_mem
    exact ⟨insert a s, card_insert_of_not_mem ha⟩


/-- If a linear order does not contain any triple of elements `x < y < z`, then this type
is finite. -/
lemma Finite.of_forall_not_lt_lt (h : ∀ ⦃x y z : α⦄, x < y → y < z → False) : Finite α := by
  /-
    α : Type u
    inst✝ : LinearOrder α
    h : ∀ ⦃x y z : α⦄, LT.lt x y → LT.lt y z → False
    ⊢ Finite α
  -/
  nontriviality α
  /-
    α : Type u
    inst✝ : LinearOrder α
    h : ∀ ⦃x y z : α⦄, LT.lt x y → LT.lt y z → False
    a✝ : Nontrivial α
    ⊢ Finite α
  -/
  rcases exists_pair_ne α with ⟨x, y, hne⟩
  /-
    case intro.intro
    α : Type u
    inst✝ : LinearOrder α
    h : ∀ ⦃x y z : α⦄, LT.lt x y → LT.lt y z → False
    a✝ : Nontrivial α
    x y : α
    hne : Ne x y
    ⊢ Finite α
  -/
  refine @Finite.of_fintype α ⟨{x, y}, fun z => ?_⟩
  /-
    case intro.intro
    α : Type u
    inst✝ : LinearOrder α
    h : ∀ ⦃x y z : α⦄, LT.lt x y → LT.lt y z → False
    a✝ : Nontrivial α
    x y : α
    hne : Ne x y
    z : α
    ⊢ Membership.mem (Insert.insert x (Singleton.singleton y)) z
  -/
  simpa [hne] using eq_or_eq_or_eq_of_forall_not_lt_lt h z x y
  /-
    🎉 no goals
  -/


/-- If a set `s` does not contain any triple of elements `x < y < z`, then `s` is finite. -/
lemma Set.finite_of_forall_not_lt_lt (h : ∀ x ∈ s, ∀ y ∈ s, ∀ z ∈ s, x < y → y < z → False) :
    Set.Finite s :=
                                                        /-
                                                          α : Type u
                                                          inst✝ : LinearOrder α
                                                          s : Set α
                                                          h : ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem s y → ∀ (z : α), …
                                                          ⊢ ∀ ⦃x y z : ↑s⦄, LT.lt x y → LT.lt y z → False
                                                        -/
  @Set.toFinite _ s <| Finite.of_forall_not_lt_lt <| by simpa only [SetCoe.forall'] using h
                                                        /-
                                                          🎉 no goals
                                                        -/


lemma Directed.exists_mem_subset_of_finset_subset_biUnion {α ι : Type*} [Nonempty ι]
    {f : ι → Set α} (h : Directed (· ⊆ ·) f) {s : Finset α} (hs : (s : Set α) ⊆ ⋃ i, f i) :
    ∃ i, (s : Set α) ⊆ f i := by
  induction s using Finset.cons_induction with
  | empty => simp
  | cons b t hbt iht =>
    simp only [Finset.coe_cons, Set.insert_subset_iff, Set.mem_iUnion] at hs ⊢
    rcases hs.imp_right iht with ⟨⟨i, hi⟩, j, hj⟩
    rcases h i j with ⟨k, hik, hjk⟩
    exact ⟨k, hik hi, hj.trans hjk⟩


