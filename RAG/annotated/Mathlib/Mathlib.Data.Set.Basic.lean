instance instBooleanAlgebra : BooleanAlgebra (Set α) :=
  { (inferInstance : BooleanAlgebra (α → Prop)) with
    sup := (· ∪ ·),
    le := (· ≤ ·),
    lt := fun s t => s ⊆ t ∧ ¬t ⊆ s,
    inf := (· ∩ ·),
    bot := ∅,
    compl := (·ᶜ),
    top := univ,
    sdiff := (· \ ·) }


instance : HasSSubset (Set α) :=
  ⟨(· < ·)⟩


@[simp]
theorem top_eq_univ : (⊤ : Set α) = univ :=
  rfl


@[simp]
theorem bot_eq_empty : (⊥ : Set α) = ∅ :=
  rfl


@[simp]
theorem sup_eq_union : ((· ⊔ ·) : Set α → Set α → Set α) = (· ∪ ·) :=
  rfl


@[simp]
theorem inf_eq_inter : ((· ⊓ ·) : Set α → Set α → Set α) = (· ∩ ·) :=
  rfl


@[simp]
theorem le_eq_subset : ((· ≤ ·) : Set α → Set α → Prop) = (· ⊆ ·) :=
  rfl


@[simp]
theorem lt_eq_ssubset : ((· < ·) : Set α → Set α → Prop) = (· ⊂ ·) :=
  rfl


theorem le_iff_subset : s ≤ t ↔ s ⊆ t :=
  Iff.rfl


theorem lt_iff_ssubset : s < t ↔ s ⊂ t :=
  Iff.rfl


alias ⟨_root_.LE.le.subset, _root_.HasSubset.Subset.le⟩ := le_iff_subset


alias ⟨_root_.LT.lt.ssubset, _root_.HasSSubset.SSubset.lt⟩ := lt_iff_ssubset


instance PiSetCoe.canLift (ι : Type u) (α : ι → Type v) [∀ i, Nonempty (α i)] (s : Set ι) :
    CanLift (∀ i : s, α i) (∀ i, α i) (fun f i => f i) fun _ => True :=
  PiSubtype.canLift ι α s


instance PiSetCoe.canLift' (ι : Type u) (α : Type v) [Nonempty α] (s : Set ι) :
    CanLift (s → α) (ι → α) (fun f i => f i) fun _ => True :=
  PiSetCoe.canLift ι (fun _ => α) s


instance (s : Set α) : CoeTC s α := ⟨fun x => x.1⟩


theorem Set.coe_eq_subtype (s : Set α) : ↥s = { x // x ∈ s } :=
  rfl


@[simp]
theorem Set.coe_setOf (p : α → Prop) : ↥{ x | p x } = { x // p x } :=
  rfl


theorem SetCoe.forall {s : Set α} {p : s → Prop} : (∀ x : s, p x) ↔ ∀ (x) (h : x ∈ s), p ⟨x, h⟩ :=
  Subtype.forall


theorem SetCoe.exists {s : Set α} {p : s → Prop} :
    (∃ x : s, p x) ↔ ∃ (x : _) (h : x ∈ s), p ⟨x, h⟩ :=
  Subtype.exists


theorem SetCoe.exists' {s : Set α} {p : ∀ x, x ∈ s → Prop} :
    (∃ (x : _) (h : x ∈ s), p x h) ↔ ∃ x : s, p x.1 x.2 :=
  (@SetCoe.exists _ _ fun x => p x.1 x.2).symm


theorem SetCoe.forall' {s : Set α} {p : ∀ x, x ∈ s → Prop} :
    (∀ (x) (h : x ∈ s), p x h) ↔ ∀ x : s, p x.1 x.2 :=
  (@SetCoe.forall _ _ fun x => p x.1 x.2).symm


@[simp]
theorem set_coe_cast :
    ∀ {s t : Set α} (H' : s = t) (H : ↥s = ↥t) (x : s), cast H x = ⟨x.1, H' ▸ x.2⟩
  | _, _, rfl, _, _ => rfl


theorem SetCoe.ext {s : Set α} {a b : s} : (a : α) = b → a = b :=
  Subtype.eq


theorem SetCoe.ext_iff {s : Set α} {a b : s} : (↑a : α) = ↑b ↔ a = b :=
  Iff.intro SetCoe.ext fun h => h ▸ rfl


/-- See also `Subtype.prop` -/
theorem Subtype.mem {α : Type*} {s : Set α} (p : s) : (p : α) ∈ s :=
  p.prop


/-- Duplicate of `Eq.subset'`, which currently has elaboration problems. -/
theorem Eq.subset {α} {s t : Set α} : s = t → s ⊆ t :=
                    /-
                      α : Type u_1
                      s t : Set α
                      h₁ : Eq s t
                      x✝ : α
                      h₂ : Membership.mem s x✝
                      ⊢ Membership.mem t x✝
                    -/
  fun h₁ _ h₂ => by rw [← h₁]; exact h₂
                               /-
                                 🎉 no goals
                               -/


instance : Inhabited (Set α) :=
  ⟨∅⟩


@[trans]
theorem mem_of_mem_of_subset {x : α} {s t : Set α} (hx : x ∈ s) (h : s ⊆ t) : x ∈ t :=
  h hx


theorem forall_in_swap {p : α → β → Prop} : (∀ a ∈ s, ∀ (b), p a b) ↔ ∀ (b), ∀ a ∈ s, p a b := by
  /-
    α : Type u
    β : Type v
    s : Set α
    p : α → β → Prop
    ⊢ Iff (∀ (a : α), Membership.mem s a → ∀ (b : β), p a b) (∀ (b : β) (a : α), M …
  -/
  tauto
  /-
    🎉 no goals
  -/


theorem mem_setOf {a : α} {p : α → Prop} : a ∈ { x | p x } ↔ p a :=
  Iff.rfl


/-- This lemma is intended for use with `rw` where a membership predicate is needed,
hence the explicit argument and the equality in the reverse direction from normal.
See also `Set.mem_setOf_eq` for the reverse direction applied to an argument. -/
theorem eq_mem_setOf (p : α → Prop) : p = (· ∈ {a | p a}) := rfl


/-- If `h : a ∈ {x | p x}` then `h.out : p x`. These are definitionally equal, but this can
nevertheless be useful for various reasons, e.g. to apply further projection notation or in an
argument to `simp`. -/
theorem _root_.Membership.mem.out {p : α → Prop} {a : α} (h : a ∈ { x | p x }) : p a :=
  h


theorem nmem_setOf_iff {a : α} {p : α → Prop} : a ∉ { x | p x } ↔ ¬p a :=
  Iff.rfl


@[simp]
theorem setOf_mem_eq {s : Set α} : { x | x ∈ s } = s :=
  rfl


theorem setOf_set {s : Set α} : setOf s = s :=
  rfl


theorem setOf_app_iff {p : α → Prop} {x : α} : { x | p x } x ↔ p x :=
  Iff.rfl


theorem mem_def {a : α} {s : Set α} : a ∈ s ↔ s a :=
  Iff.rfl


theorem setOf_bijective : Bijective (setOf : (α → Prop) → Set α) :=
  bijective_id


theorem subset_setOf {p : α → Prop} {s : Set α} : s ⊆ setOf p ↔ ∀ x, x ∈ s → p x :=
  Iff.rfl


theorem setOf_subset {p : α → Prop} {s : Set α} : setOf p ⊆ s ↔ ∀ x, p x → x ∈ s :=
  Iff.rfl


@[simp]
theorem setOf_subset_setOf {p q : α → Prop} : { a | p a } ⊆ { a | q a } ↔ ∀ a, p a → q a :=
  Iff.rfl


theorem setOf_and {p q : α → Prop} : { a | p a ∧ q a } = { a | p a } ∩ { a | q a } :=
  rfl


theorem setOf_or {p q : α → Prop} : { a | p a ∨ q a } = { a | p a } ∪ { a | q a } :=
  rfl


instance : IsRefl (Set α) (· ⊆ ·) :=
                                 /-
                                   α : Type u
                                   β : Type v
                                   a b : α
                                   s s₁ s₂ t t₁ t₂ u : Set α
                                   ⊢ IsRefl (Set α) fun x1 x2 => LE.le x1 x2
                                 -/
  show IsRefl (Set α) (· ≤ ·) by infer_instance
                                 /-
                                   🎉 no goals
                                 -/


instance : IsTrans (Set α) (· ⊆ ·) :=
                                  /-
                                    α : Type u
                                    β : Type v
                                    a b : α
                                    s s₁ s₂ t t₁ t₂ u : Set α
                                    ⊢ IsTrans (Set α) fun x1 x2 => LE.le x1 x2
                                  -/
  show IsTrans (Set α) (· ≤ ·) by infer_instance
                                  /-
                                    🎉 no goals
                                  -/


instance : Trans ((· ⊆ ·) : Set α → Set α → Prop) (· ⊆ ·) (· ⊆ ·) :=
                                        /-
                                          α : Type u
                                          β : Type v
                                          a b : α
                                          s s₁ s₂ t t₁ t₂ u : Set α
                                          ⊢ Trans (fun x1 x2 => LE.le x1 x2) (fun x1 x2 => LE.le x1 x2) fun x1 x2 => LE. …
                                        -/
  show Trans (· ≤ ·) (· ≤ ·) (· ≤ ·) by infer_instance
                                        /-
                                          🎉 no goals
                                        -/


instance : IsAntisymm (Set α) (· ⊆ ·) :=
                                     /-
                                       α : Type u
                                       β : Type v
                                       a b : α
                                       s s₁ s₂ t t₁ t₂ u : Set α
                                       ⊢ IsAntisymm (Set α) fun x1 x2 => LE.le x1 x2
                                     -/
  show IsAntisymm (Set α) (· ≤ ·) by infer_instance
                                     /-
                                       🎉 no goals
                                     -/


instance : IsIrrefl (Set α) (· ⊂ ·) :=
                                   /-
                                     α : Type u
                                     β : Type v
                                     a b : α
                                     s s₁ s₂ t t₁ t₂ u : Set α
                                     ⊢ IsIrrefl (Set α) fun x1 x2 => LT.lt x1 x2
                                   -/
  show IsIrrefl (Set α) (· < ·) by infer_instance
                                   /-
                                     🎉 no goals
                                   -/


instance : IsTrans (Set α) (· ⊂ ·) :=
                                  /-
                                    α : Type u
                                    β : Type v
                                    a b : α
                                    s s₁ s₂ t t₁ t₂ u : Set α
                                    ⊢ IsTrans (Set α) fun x1 x2 => LT.lt x1 x2
                                  -/
  show IsTrans (Set α) (· < ·) by infer_instance
                                  /-
                                    🎉 no goals
                                  -/


instance : Trans ((· ⊂ ·) : Set α → Set α → Prop) (· ⊂ ·) (· ⊂ ·) :=
                                        /-
                                          α : Type u
                                          β : Type v
                                          a b : α
                                          s s₁ s₂ t t₁ t₂ u : Set α
                                          ⊢ Trans (fun x1 x2 => LT.lt x1 x2) (fun x1 x2 => LT.lt x1 x2) fun x1 x2 => LT. …
                                        -/
  show Trans (· < ·) (· < ·) (· < ·) by infer_instance
                                        /-
                                          🎉 no goals
                                        -/


instance : Trans ((· ⊂ ·) : Set α → Set α → Prop) (· ⊆ ·) (· ⊂ ·) :=
                                        /-
                                          α : Type u
                                          β : Type v
                                          a b : α
                                          s s₁ s₂ t t₁ t₂ u : Set α
                                          ⊢ Trans (fun x1 x2 => LT.lt x1 x2) (fun x1 x2 => LE.le x1 x2) fun x1 x2 => LT. …
                                        -/
  show Trans (· < ·) (· ≤ ·) (· < ·) by infer_instance
                                        /-
                                          🎉 no goals
                                        -/


instance : Trans ((· ⊆ ·) : Set α → Set α → Prop) (· ⊂ ·) (· ⊂ ·) :=
                                        /-
                                          α : Type u
                                          β : Type v
                                          a b : α
                                          s s₁ s₂ t t₁ t₂ u : Set α
                                          ⊢ Trans (fun x1 x2 => LE.le x1 x2) (fun x1 x2 => LT.lt x1 x2) fun x1 x2 => LT. …
                                        -/
  show Trans (· ≤ ·) (· < ·) (· < ·) by infer_instance
                                        /-
                                          🎉 no goals
                                        -/


instance : IsAsymm (Set α) (· ⊂ ·) :=
                                  /-
                                    α : Type u
                                    β : Type v
                                    a b : α
                                    s s₁ s₂ t t₁ t₂ u : Set α
                                    ⊢ IsAsymm (Set α) fun x1 x2 => LT.lt x1 x2
                                  -/
  show IsAsymm (Set α) (· < ·) by infer_instance
                                  /-
                                    🎉 no goals
                                  -/


instance : IsNonstrictStrictOrder (Set α) (· ⊆ ·) (· ⊂ ·) :=
  ⟨fun _ _ => Iff.rfl⟩

-- TODO(Jeremy): write a tactic to unfold specific instances of generic notation?

theorem subset_def : (s ⊆ t) = ∀ x, x ∈ s → x ∈ t :=
  rfl


theorem ssubset_def : (s ⊂ t) = (s ⊆ t ∧ ¬t ⊆ s) :=
  rfl


@[refl]
theorem Subset.refl (a : Set α) : a ⊆ a := fun _ => id


theorem Subset.rfl {s : Set α} : s ⊆ s :=
  Subset.refl s


@[trans]
theorem Subset.trans {a b c : Set α} (ab : a ⊆ b) (bc : b ⊆ c) : a ⊆ c := fun _ h => bc <| ab h


@[trans]
theorem mem_of_eq_of_mem {x y : α} {s : Set α} (hx : x = y) (h : y ∈ s) : x ∈ s :=
  hx.symm ▸ h


theorem Subset.antisymm {a b : Set α} (h₁ : a ⊆ b) (h₂ : b ⊆ a) : a = b :=
  Set.ext fun _ => ⟨@h₁ _, @h₂ _⟩


theorem Subset.antisymm_iff {a b : Set α} : a = b ↔ a ⊆ b ∧ b ⊆ a :=
  ⟨fun e => ⟨e.subset, e.symm.subset⟩, fun ⟨h₁, h₂⟩ => Subset.antisymm h₁ h₂⟩

-- an alternative name

theorem eq_of_subset_of_subset {a b : Set α} : a ⊆ b → b ⊆ a → a = b :=
  Subset.antisymm


theorem mem_of_subset_of_mem {s₁ s₂ : Set α} {a : α} (h : s₁ ⊆ s₂) : a ∈ s₁ → a ∈ s₂ :=
  @h _


theorem not_mem_subset (h : s ⊆ t) : a ∉ t → a ∉ s :=
  mt <| mem_of_subset_of_mem h


theorem not_subset : ¬s ⊆ t ↔ ∃ a ∈ s, a ∉ t := by
  /-
    α : Type u
    s t : Set α
    ⊢ Iff (Not (HasSubset.Subset s t)) (Exists fun a => And (Membership.mem s a) ( …
  -/
  simp only [subset_def, not_forall, exists_prop]
  /-
    🎉 no goals
  -/


lemma eq_of_forall_subset_iff (h : ∀ u, s ⊆ u ↔ t ⊆ u) : s = t := eq_of_forall_ge_iff h


protected theorem eq_or_ssubset_of_subset (h : s ⊆ t) : s = t ∨ s ⊂ t :=
  eq_or_lt_of_le h


theorem exists_of_ssubset {s t : Set α} (h : s ⊂ t) : ∃ x ∈ t, x ∉ s :=
  not_subset.1 h.2


protected theorem ssubset_iff_subset_ne {s t : Set α} : s ⊂ t ↔ s ⊆ t ∧ s ≠ t :=
  @lt_iff_le_and_ne (Set α) _ s t


theorem ssubset_iff_of_subset {s t : Set α} (h : s ⊆ t) : s ⊂ t ↔ ∃ x ∈ t, x ∉ s :=
  ⟨exists_of_ssubset, fun ⟨_, hxt, hxs⟩ => ⟨h, fun h => hxs <| h hxt⟩⟩


protected theorem ssubset_of_ssubset_of_subset {s₁ s₂ s₃ : Set α} (hs₁s₂ : s₁ ⊂ s₂)
    (hs₂s₃ : s₂ ⊆ s₃) : s₁ ⊂ s₃ :=
  ⟨Subset.trans hs₁s₂.1 hs₂s₃, fun hs₃s₁ => hs₁s₂.2 (Subset.trans hs₂s₃ hs₃s₁)⟩


protected theorem ssubset_of_subset_of_ssubset {s₁ s₂ s₃ : Set α} (hs₁s₂ : s₁ ⊆ s₂)
    (hs₂s₃ : s₂ ⊂ s₃) : s₁ ⊂ s₃ :=
  ⟨Subset.trans hs₁s₂ hs₂s₃.1, fun hs₃s₁ => hs₂s₃.2 (Subset.trans hs₃s₁ hs₁s₂)⟩


theorem not_mem_empty (x : α) : ¬x ∈ (∅ : Set α) :=
  id


theorem not_not_mem : ¬a ∉ s ↔ a ∈ s :=
  not_not


theorem nonempty_coe_sort {s : Set α} : Nonempty (↥s) ↔ s.Nonempty :=
  nonempty_subtype


alias ⟨_, Nonempty.coe_sort⟩ := nonempty_coe_sort


theorem nonempty_def : s.Nonempty ↔ ∃ x, x ∈ s :=
  Iff.rfl


theorem nonempty_of_mem {x} (h : x ∈ s) : s.Nonempty :=
  ⟨x, h⟩


theorem Nonempty.not_subset_empty : s.Nonempty → ¬s ⊆ ∅
  | ⟨_, hx⟩, hs => hs hx


/-- Extract a witness from `s.Nonempty`. This function might be used instead of case analysis
on the argument. Note that it makes a proof depend on the `Classical.choice` axiom. -/
protected noncomputable def Nonempty.some (h : s.Nonempty) : α :=
  Classical.choose h


protected theorem Nonempty.some_mem (h : s.Nonempty) : h.some ∈ s :=
  Classical.choose_spec h


theorem Nonempty.mono (ht : s ⊆ t) (hs : s.Nonempty) : t.Nonempty :=
  hs.imp ht


theorem nonempty_of_not_subset (h : ¬s ⊆ t) : (s \ t).Nonempty :=
  let ⟨x, xs, xt⟩ := not_subset.1 h
  ⟨x, xs, xt⟩


theorem nonempty_of_ssubset (ht : s ⊂ t) : (t \ s).Nonempty :=
  nonempty_of_not_subset ht.2


theorem Nonempty.of_diff (h : (s \ t).Nonempty) : s.Nonempty :=
  h.imp fun _ => And.left


theorem nonempty_of_ssubset' (ht : s ⊂ t) : t.Nonempty :=
  (nonempty_of_ssubset ht).of_diff


theorem Nonempty.inl (hs : s.Nonempty) : (s ∪ t).Nonempty :=
  hs.imp fun _ => Or.inl


theorem Nonempty.inr (ht : t.Nonempty) : (s ∪ t).Nonempty :=
  ht.imp fun _ => Or.inr


@[simp]
theorem union_nonempty : (s ∪ t).Nonempty ↔ s.Nonempty ∨ t.Nonempty :=
  exists_or


theorem Nonempty.left (h : (s ∩ t).Nonempty) : s.Nonempty :=
  h.imp fun _ => And.left


theorem Nonempty.right (h : (s ∩ t).Nonempty) : t.Nonempty :=
  h.imp fun _ => And.right


theorem inter_nonempty : (s ∩ t).Nonempty ↔ ∃ x, x ∈ s ∧ x ∈ t :=
  Iff.rfl


theorem inter_nonempty_iff_exists_left : (s ∩ t).Nonempty ↔ ∃ x ∈ s, x ∈ t := by
  /-
    α : Type u
    s t : Set α
    ⊢ Iff (Inter.inter s t).Nonempty (Exists fun x => And (Membership.mem s x) (Me …
  -/
  simp_rw [inter_nonempty]
  /-
    🎉 no goals
  -/


theorem inter_nonempty_iff_exists_right : (s ∩ t).Nonempty ↔ ∃ x ∈ t, x ∈ s := by
  /-
    α : Type u
    s t : Set α
    ⊢ Iff (Inter.inter s t).Nonempty (Exists fun x => And (Membership.mem t x) (Me …
  -/
  simp_rw [inter_nonempty, and_comm]
  /-
    🎉 no goals
  -/


theorem nonempty_iff_univ_nonempty : Nonempty α ↔ (univ : Set α).Nonempty :=
  ⟨fun ⟨x⟩ => ⟨x, trivial⟩, fun ⟨x, _⟩ => ⟨x⟩⟩


@[simp]
theorem univ_nonempty : ∀ [Nonempty α], (univ : Set α).Nonempty
  | ⟨x⟩ => ⟨x, trivial⟩


theorem Nonempty.to_subtype : s.Nonempty → Nonempty (↥s) :=
  nonempty_subtype.2


theorem Nonempty.to_type : s.Nonempty → Nonempty α := fun ⟨x, _⟩ => ⟨x⟩


instance univ.nonempty [Nonempty α] : Nonempty (↥(Set.univ : Set α)) :=
  Set.univ_nonempty.to_subtype

-- Redeclare for refined keys
-- `Nonempty (@Subtype _ (@Membership.mem _ (Set _) _ (@Top.top (Set _) _)))`

instance instNonemptyTop [Nonempty α] : Nonempty (⊤ : Set α) :=
  inferInstanceAs (Nonempty (univ : Set α))


theorem Nonempty.of_subtype [Nonempty (↥s)] : s.Nonempty := nonempty_subtype.mp ‹_›


@[deprecated (since := "2024-11-23")] alias nonempty_of_nonempty_subtype := Nonempty.of_subtype


theorem empty_def : (∅ : Set α) = { _x : α | False } :=
  rfl


@[simp]
theorem mem_empty_iff_false (x : α) : x ∈ (∅ : Set α) ↔ False :=
  Iff.rfl


@[simp]
theorem setOf_false : { _a : α | False } = ∅ :=
  rfl


@[simp] theorem setOf_bot : { _x : α | ⊥ } = ∅ := rfl


@[simp]
theorem empty_subset (s : Set α) : ∅ ⊆ s :=
  nofun


@[simp]
theorem subset_empty_iff {s : Set α} : s ⊆ ∅ ↔ s = ∅ :=
  (Subset.antisymm_iff.trans <| and_iff_left (empty_subset _)).symm


theorem eq_empty_iff_forall_not_mem {s : Set α} : s = ∅ ↔ ∀ x, x ∉ s :=
  subset_empty_iff.symm


theorem eq_empty_of_forall_not_mem (h : ∀ x, x ∉ s) : s = ∅ :=
  subset_empty_iff.1 h


theorem eq_empty_of_subset_empty {s : Set α} : s ⊆ ∅ → s = ∅ :=
  subset_empty_iff.1


theorem eq_empty_of_isEmpty [IsEmpty α] (s : Set α) : s = ∅ :=
  eq_empty_of_subset_empty fun x _ => isEmptyElim x


/-- There is exactly one set of a type that is empty. -/
instance uniqueEmpty [IsEmpty α] : Unique (Set α) where
  default := ∅
  uniq := eq_empty_of_isEmpty


/-- See also `Set.nonempty_iff_ne_empty`. -/
theorem not_nonempty_iff_eq_empty {s : Set α} : ¬s.Nonempty ↔ s = ∅ := by
  /-
    α : Type u
    s : Set α
    ⊢ Iff (Not s.Nonempty) (Eq s EmptyCollection.emptyCollection)
  -/
  simp only [Set.Nonempty, not_exists, eq_empty_iff_forall_not_mem]
  /-
    🎉 no goals
  -/


/-- See also `Set.not_nonempty_iff_eq_empty`. -/
theorem nonempty_iff_ne_empty : s.Nonempty ↔ s ≠ ∅ :=
  not_nonempty_iff_eq_empty.not_right


/-- See also `nonempty_iff_ne_empty'`. -/
theorem not_nonempty_iff_eq_empty' : ¬Nonempty s ↔ s = ∅ := by
  /-
    α : Type u
    s : Set α
    ⊢ Iff (Not (Nonempty ↑s)) (Eq s EmptyCollection.emptyCollection)
  -/
  rw [nonempty_subtype, not_exists, eq_empty_iff_forall_not_mem]
  /-
    🎉 no goals
  -/


/-- See also `not_nonempty_iff_eq_empty'`. -/
theorem nonempty_iff_ne_empty' : Nonempty s ↔ s ≠ ∅ :=
  not_nonempty_iff_eq_empty'.not_right


alias ⟨Nonempty.ne_empty, _⟩ := nonempty_iff_ne_empty


@[simp]
theorem not_nonempty_empty : ¬(∅ : Set α).Nonempty := fun ⟨_, hx⟩ => hx


@[simp]
theorem isEmpty_coe_sort {s : Set α} : IsEmpty (↥s) ↔ s = ∅ :=
                      /-
                        α : Type u
                        s : Set α
                        ⊢ Iff (Not (IsEmpty ↑s)) (Not (Eq s EmptyCollection.emptyCollection))
                      -/
  not_iff_not.1 <| by simpa using nonempty_iff_ne_empty
                      /-
                        🎉 no goals
                      -/


theorem eq_empty_or_nonempty (s : Set α) : s = ∅ ∨ s.Nonempty :=
  or_iff_not_imp_left.2 nonempty_iff_ne_empty.2


theorem subset_eq_empty {s t : Set α} (h : t ⊆ s) (e : s = ∅) : t = ∅ :=
  subset_empty_iff.1 <| e ▸ h


theorem forall_mem_empty {p : α → Prop} : (∀ x ∈ (∅ : Set α), p x) ↔ True :=
  iff_true_intro fun _ => False.elim

@[deprecated (since := "2024-03-23")] alias ball_empty_iff := forall_mem_empty


instance (α : Type u) : IsEmpty.{u + 1} (↥(∅ : Set α)) :=
  ⟨fun x => x.2⟩


@[simp]
theorem empty_ssubset : ∅ ⊂ s ↔ s.Nonempty :=
  (@bot_lt_iff_ne_bot (Set α) _ _ _).trans nonempty_iff_ne_empty.symm


alias ⟨_, Nonempty.empty_ssubset⟩ := empty_ssubset


@[simp]
theorem setOf_true : { _x : α | True } = univ :=
  rfl


@[simp] theorem setOf_top : { _x : α | ⊤ } = univ := rfl


@[simp]
theorem univ_eq_empty_iff : (univ : Set α) = ∅ ↔ IsEmpty α :=
  eq_empty_iff_forall_not_mem.trans
    ⟨fun H => ⟨fun x => H x trivial⟩, fun H x _ => @IsEmpty.false α H x⟩


theorem empty_ne_univ [Nonempty α] : (∅ : Set α) ≠ univ := fun e =>
  not_isEmpty_of_nonempty α <| univ_eq_empty_iff.1 e.symm


@[simp]
theorem subset_univ (s : Set α) : s ⊆ univ := fun _ _ => trivial


@[simp]
theorem univ_subset_iff {s : Set α} : univ ⊆ s ↔ s = univ :=
  @top_le_iff _ _ _ s


alias ⟨eq_univ_of_univ_subset, _⟩ := univ_subset_iff


theorem eq_univ_iff_forall {s : Set α} : s = univ ↔ ∀ x, x ∈ s :=
  univ_subset_iff.symm.trans <| forall_congr' fun _ => imp_iff_right trivial


theorem eq_univ_of_forall {s : Set α} : (∀ x, x ∈ s) → s = univ :=
  eq_univ_iff_forall.2


theorem Nonempty.eq_univ [Subsingleton α] : s.Nonempty → s = univ := by
  /-
    α : Type u
    s : Set α
    inst✝ : Subsingleton α
    ⊢ s.Nonempty → Eq s Set.univ
  -/
  rintro ⟨x, hx⟩
  /-
    case intro
    α : Type u
    s : Set α
    inst✝ : Subsingleton α
    x : α
    hx : Membership.mem s x
    ⊢ Eq s Set.univ
  -/
  exact eq_univ_of_forall fun y => by rwa [Subsingleton.elim y x]
  /-
    🎉 no goals
  -/


theorem eq_univ_of_subset {s t : Set α} (h : s ⊆ t) (hs : s = univ) : t = univ :=
  eq_univ_of_univ_subset <| (hs ▸ h : univ ⊆ t)


theorem exists_mem_of_nonempty (α) : ∀ [Nonempty α], ∃ x : α, x ∈ (univ : Set α)
  | ⟨x⟩ => ⟨x, trivial⟩


theorem ne_univ_iff_exists_not_mem {α : Type*} (s : Set α) : s ≠ univ ↔ ∃ a, a ∉ s := by
  /-
    α : Type u_1
    s : Set α
    ⊢ Iff (Ne s Set.univ) (Exists fun a => Not (Membership.mem s a))
  -/
  rw [← not_forall, ← eq_univ_iff_forall]
  /-
    🎉 no goals
  -/


theorem not_subset_iff_exists_mem_not_mem {α : Type*} {s t : Set α} :
                                      /-
                                        α : Type u_1
                                        s t : Set α
                                        ⊢ Iff (Not (HasSubset.Subset s t)) (Exists fun x => And (Membership.mem s x) ( …
                                      -/
    ¬s ⊆ t ↔ ∃ x, x ∈ s ∧ x ∉ t := by simp [subset_def]
                                      /-
                                        🎉 no goals
                                      -/


theorem univ_unique [Unique α] : @Set.univ α = {default} :=
  Set.ext fun x => iff_of_true trivial <| Subsingleton.elim x default


theorem ssubset_univ_iff : s ⊂ univ ↔ s ≠ univ :=
  lt_top_iff_ne_top


instance nontrivial_of_nonempty [Nonempty α] : Nontrivial (Set α) :=
  ⟨⟨∅, univ, empty_ne_univ⟩⟩


theorem union_def {s₁ s₂ : Set α} : s₁ ∪ s₂ = { a | a ∈ s₁ ∨ a ∈ s₂ } :=
  rfl


theorem mem_union_left {x : α} {a : Set α} (b : Set α) : x ∈ a → x ∈ a ∪ b :=
  Or.inl


theorem mem_union_right {x : α} {b : Set α} (a : Set α) : x ∈ b → x ∈ a ∪ b :=
  Or.inr


theorem mem_or_mem_of_mem_union {x : α} {a b : Set α} (H : x ∈ a ∪ b) : x ∈ a ∨ x ∈ b :=
  H


theorem MemUnion.elim {x : α} {a b : Set α} {P : Prop} (H₁ : x ∈ a ∪ b) (H₂ : x ∈ a → P)
    (H₃ : x ∈ b → P) : P :=
  Or.elim H₁ H₂ H₃


@[simp]
theorem mem_union (x : α) (a b : Set α) : x ∈ a ∪ b ↔ x ∈ a ∨ x ∈ b :=
  Iff.rfl


@[simp]
theorem union_self (a : Set α) : a ∪ a = a :=
  ext fun _ => or_self_iff


@[simp]
theorem union_empty (a : Set α) : a ∪ ∅ = a :=
  ext fun _ => iff_of_eq (or_false _)


@[simp]
theorem empty_union (a : Set α) : ∅ ∪ a = a :=
  ext fun _ => iff_of_eq (false_or _)


theorem union_comm (a b : Set α) : a ∪ b = b ∪ a :=
  ext fun _ => or_comm


theorem union_assoc (a b c : Set α) : a ∪ b ∪ c = a ∪ (b ∪ c) :=
  ext fun _ => or_assoc


instance union_isAssoc : Std.Associative (α := Set α) (· ∪ ·) :=
  ⟨union_assoc⟩


instance union_isComm : Std.Commutative (α := Set α) (· ∪ ·) :=
  ⟨union_comm⟩


theorem union_left_comm (s₁ s₂ s₃ : Set α) : s₁ ∪ (s₂ ∪ s₃) = s₂ ∪ (s₁ ∪ s₃) :=
  ext fun _ => or_left_comm


theorem union_right_comm (s₁ s₂ s₃ : Set α) : s₁ ∪ s₂ ∪ s₃ = s₁ ∪ s₃ ∪ s₂ :=
  ext fun _ => or_right_comm


@[simp]
theorem union_eq_left {s t : Set α} : s ∪ t = s ↔ t ⊆ s :=
  sup_eq_left


@[simp]
theorem union_eq_right {s t : Set α} : s ∪ t = t ↔ s ⊆ t :=
  sup_eq_right


theorem union_eq_self_of_subset_left {s t : Set α} (h : s ⊆ t) : s ∪ t = t :=
  union_eq_right.mpr h


theorem union_eq_self_of_subset_right {s t : Set α} (h : t ⊆ s) : s ∪ t = s :=
  union_eq_left.mpr h


@[simp]
theorem subset_union_left {s t : Set α} : s ⊆ s ∪ t := fun _ => Or.inl


@[simp]
theorem subset_union_right {s t : Set α} : t ⊆ s ∪ t := fun _ => Or.inr


theorem union_subset {s t r : Set α} (sr : s ⊆ r) (tr : t ⊆ r) : s ∪ t ⊆ r := fun _ =>
  Or.rec (@sr _) (@tr _)


@[simp]
theorem union_subset_iff {s t u : Set α} : s ∪ t ⊆ u ↔ s ⊆ u ∧ t ⊆ u :=
  (forall_congr' fun _ => or_imp).trans forall_and


@[gcongr]
theorem union_subset_union {s₁ s₂ t₁ t₂ : Set α} (h₁ : s₁ ⊆ s₂) (h₂ : t₁ ⊆ t₂) :
    s₁ ∪ t₁ ⊆ s₂ ∪ t₂ := fun _ => Or.imp (@h₁ _) (@h₂ _)


@[gcongr]
theorem union_subset_union_left {s₁ s₂ : Set α} (t) (h : s₁ ⊆ s₂) : s₁ ∪ t ⊆ s₂ ∪ t :=
  union_subset_union h Subset.rfl


@[gcongr]
theorem union_subset_union_right (s) {t₁ t₂ : Set α} (h : t₁ ⊆ t₂) : s ∪ t₁ ⊆ s ∪ t₂ :=
  union_subset_union Subset.rfl h


theorem subset_union_of_subset_left {s t : Set α} (h : s ⊆ t) (u : Set α) : s ⊆ t ∪ u :=
  h.trans subset_union_left


theorem subset_union_of_subset_right {s u : Set α} (h : s ⊆ u) (t : Set α) : s ⊆ t ∪ u :=
  h.trans subset_union_right

-- Porting note: replaced `⊔` in RHS

theorem union_congr_left (ht : t ⊆ s ∪ u) (hu : u ⊆ s ∪ t) : s ∪ t = s ∪ u :=
  sup_congr_left ht hu


theorem union_congr_right (hs : s ⊆ t ∪ u) (ht : t ⊆ s ∪ u) : s ∪ u = t ∪ u :=
  sup_congr_right hs ht


theorem union_eq_union_iff_left : s ∪ t = s ∪ u ↔ t ⊆ s ∪ u ∧ u ⊆ s ∪ t :=
  sup_eq_sup_iff_left


theorem union_eq_union_iff_right : s ∪ u = t ∪ u ↔ s ⊆ t ∪ u ∧ t ⊆ s ∪ u :=
  sup_eq_sup_iff_right


@[simp]
theorem union_empty_iff {s t : Set α} : s ∪ t = ∅ ↔ s = ∅ ∧ t = ∅ := by
  /-
    α : Type u
    s t : Set α
    ⊢ Iff (Eq (Union.union s t) EmptyCollection.emptyCollection) (And (Eq s EmptyC …
  -/
  simp only [← subset_empty_iff]
  /-
    α : Type u
    s t : Set α
    ⊢ Iff (HasSubset.Subset (Union.union s t) EmptyCollection.emptyCollection) (An …
  -/
  exact union_subset_iff
  /-
    🎉 no goals
  -/


@[simp]
theorem union_univ (s : Set α) : s ∪ univ = univ := sup_top_eq _


@[simp]
theorem univ_union (s : Set α) : univ ∪ s = univ := top_sup_eq _


theorem inter_def {s₁ s₂ : Set α} : s₁ ∩ s₂ = { a | a ∈ s₁ ∧ a ∈ s₂ } :=
  rfl


@[simp, mfld_simps]
theorem mem_inter_iff (x : α) (a b : Set α) : x ∈ a ∩ b ↔ x ∈ a ∧ x ∈ b :=
  Iff.rfl


theorem mem_inter {x : α} {a b : Set α} (ha : x ∈ a) (hb : x ∈ b) : x ∈ a ∩ b :=
  ⟨ha, hb⟩


theorem mem_of_mem_inter_left {x : α} {a b : Set α} (h : x ∈ a ∩ b) : x ∈ a :=
  h.left


theorem mem_of_mem_inter_right {x : α} {a b : Set α} (h : x ∈ a ∩ b) : x ∈ b :=
  h.right


@[simp]
theorem inter_self (a : Set α) : a ∩ a = a :=
  ext fun _ => and_self_iff


@[simp]
theorem inter_empty (a : Set α) : a ∩ ∅ = ∅ :=
  ext fun _ => iff_of_eq (and_false _)


@[simp]
theorem empty_inter (a : Set α) : ∅ ∩ a = ∅ :=
  ext fun _ => iff_of_eq (false_and _)


theorem inter_comm (a b : Set α) : a ∩ b = b ∩ a :=
  ext fun _ => and_comm


theorem inter_assoc (a b c : Set α) : a ∩ b ∩ c = a ∩ (b ∩ c) :=
  ext fun _ => and_assoc


instance inter_isAssoc : Std.Associative (α := Set α) (· ∩ ·) :=
  ⟨inter_assoc⟩


instance inter_isComm : Std.Commutative (α := Set α) (· ∩ ·) :=
  ⟨inter_comm⟩


theorem inter_left_comm (s₁ s₂ s₃ : Set α) : s₁ ∩ (s₂ ∩ s₃) = s₂ ∩ (s₁ ∩ s₃) :=
  ext fun _ => and_left_comm


theorem inter_right_comm (s₁ s₂ s₃ : Set α) : s₁ ∩ s₂ ∩ s₃ = s₁ ∩ s₃ ∩ s₂ :=
  ext fun _ => and_right_comm


@[simp, mfld_simps]
theorem inter_subset_left {s t : Set α} : s ∩ t ⊆ s := fun _ => And.left


@[simp]
theorem inter_subset_right {s t : Set α} : s ∩ t ⊆ t := fun _ => And.right


theorem subset_inter {s t r : Set α} (rs : r ⊆ s) (rt : r ⊆ t) : r ⊆ s ∩ t := fun _ h =>
  ⟨rs h, rt h⟩


@[simp]
theorem subset_inter_iff {s t r : Set α} : r ⊆ s ∩ t ↔ r ⊆ s ∧ r ⊆ t :=
  (forall_congr' fun _ => imp_and).trans forall_and


@[simp] lemma inter_eq_left : s ∩ t = s ↔ s ⊆ t := inf_eq_left


@[simp] lemma inter_eq_right : s ∩ t = t ↔ t ⊆ s := inf_eq_right


@[simp] lemma left_eq_inter : s = s ∩ t ↔ s ⊆ t := left_eq_inf


@[simp] lemma right_eq_inter : t = s ∩ t ↔ t ⊆ s := right_eq_inf


theorem inter_eq_self_of_subset_left {s t : Set α} : s ⊆ t → s ∩ t = s :=
  inter_eq_left.mpr


theorem inter_eq_self_of_subset_right {s t : Set α} : t ⊆ s → s ∩ t = t :=
  inter_eq_right.mpr


theorem inter_congr_left (ht : s ∩ u ⊆ t) (hu : s ∩ t ⊆ u) : s ∩ t = s ∩ u :=
  inf_congr_left ht hu


theorem inter_congr_right (hs : t ∩ u ⊆ s) (ht : s ∩ u ⊆ t) : s ∩ u = t ∩ u :=
  inf_congr_right hs ht


theorem inter_eq_inter_iff_left : s ∩ t = s ∩ u ↔ s ∩ u ⊆ t ∧ s ∩ t ⊆ u :=
  inf_eq_inf_iff_left


theorem inter_eq_inter_iff_right : s ∩ u = t ∩ u ↔ t ∩ u ⊆ s ∧ s ∩ u ⊆ t :=
  inf_eq_inf_iff_right


@[simp, mfld_simps]
theorem inter_univ (a : Set α) : a ∩ univ = a := inf_top_eq _


@[simp, mfld_simps]
theorem univ_inter (a : Set α) : univ ∩ a = a := top_inf_eq _


@[gcongr]
theorem inter_subset_inter {s₁ s₂ t₁ t₂ : Set α} (h₁ : s₁ ⊆ t₁) (h₂ : s₂ ⊆ t₂) :
    s₁ ∩ s₂ ⊆ t₁ ∩ t₂ := fun _ => And.imp (@h₁ _) (@h₂ _)


@[gcongr]
theorem inter_subset_inter_left {s t : Set α} (u : Set α) (H : s ⊆ t) : s ∩ u ⊆ t ∩ u :=
  inter_subset_inter H Subset.rfl


@[gcongr]
theorem inter_subset_inter_right {s t : Set α} (u : Set α) (H : s ⊆ t) : u ∩ s ⊆ u ∩ t :=
  inter_subset_inter Subset.rfl H


theorem union_inter_cancel_left {s t : Set α} : (s ∪ t) ∩ s = s :=
  inter_eq_self_of_subset_right subset_union_left


theorem union_inter_cancel_right {s t : Set α} : (s ∪ t) ∩ t = t :=
  inter_eq_self_of_subset_right subset_union_right


theorem inter_setOf_eq_sep (s : Set α) (p : α → Prop) : s ∩ {a | p a} = {a ∈ s | p a} :=
  rfl


theorem setOf_inter_eq_sep (p : α → Prop) (s : Set α) : {a | p a} ∩ s = {a ∈ s | p a} :=
  inter_comm _ _


theorem inter_union_distrib_left (s t u : Set α) : s ∩ (t ∪ u) = s ∩ t ∪ s ∩ u :=
  inf_sup_left _ _ _


theorem union_inter_distrib_right (s t u : Set α) : (s ∪ t) ∩ u = s ∩ u ∪ t ∩ u :=
  inf_sup_right _ _ _


theorem union_inter_distrib_left (s t u : Set α) : s ∪ t ∩ u = (s ∪ t) ∩ (s ∪ u) :=
  sup_inf_left _ _ _


theorem inter_union_distrib_right (s t u : Set α) : s ∩ t ∪ u = (s ∪ u) ∩ (t ∪ u) :=
  sup_inf_right _ _ _


@[deprecated (since := "2024-03-22")] alias inter_distrib_left := inter_union_distrib_left

@[deprecated (since := "2024-03-22")] alias inter_distrib_right := union_inter_distrib_right

@[deprecated (since := "2024-03-22")] alias union_distrib_left := union_inter_distrib_left

@[deprecated (since := "2024-03-22")] alias union_distrib_right := inter_union_distrib_right


theorem union_union_distrib_left (s t u : Set α) : s ∪ (t ∪ u) = s ∪ t ∪ (s ∪ u) :=
  sup_sup_distrib_left _ _ _


theorem union_union_distrib_right (s t u : Set α) : s ∪ t ∪ u = s ∪ u ∪ (t ∪ u) :=
  sup_sup_distrib_right _ _ _


theorem inter_inter_distrib_left (s t u : Set α) : s ∩ (t ∩ u) = s ∩ t ∩ (s ∩ u) :=
  inf_inf_distrib_left _ _ _


theorem inter_inter_distrib_right (s t u : Set α) : s ∩ t ∩ u = s ∩ u ∩ (t ∩ u) :=
  inf_inf_distrib_right _ _ _


theorem union_union_union_comm (s t u v : Set α) : s ∪ t ∪ (u ∪ v) = s ∪ u ∪ (t ∪ v) :=
  sup_sup_sup_comm _ _ _ _


theorem inter_inter_inter_comm (s t u v : Set α) : s ∩ t ∩ (u ∩ v) = s ∩ u ∩ (t ∩ v) :=
  inf_inf_inf_comm _ _ _ _


theorem insert_def (x : α) (s : Set α) : insert x s = { y | y = x ∨ y ∈ s } :=
  rfl


@[simp]
theorem subset_insert (x : α) (s : Set α) : s ⊆ insert x s := fun _ => Or.inr


theorem mem_insert (x : α) (s : Set α) : x ∈ insert x s :=
  Or.inl rfl


theorem mem_insert_of_mem {x : α} {s : Set α} (y : α) : x ∈ s → x ∈ insert y s :=
  Or.inr


theorem eq_or_mem_of_mem_insert {x a : α} {s : Set α} : x ∈ insert a s → x = a ∨ x ∈ s :=
  id


theorem mem_of_mem_insert_of_ne : b ∈ insert a s → b ≠ a → b ∈ s :=
  Or.resolve_left


theorem eq_of_not_mem_of_mem_insert : b ∈ insert a s → b ∉ s → b = a :=
  Or.resolve_right


@[simp]
theorem mem_insert_iff {x a : α} {s : Set α} : x ∈ insert a s ↔ x = a ∨ x ∈ s :=
  Iff.rfl


@[simp]
theorem insert_eq_of_mem {a : α} {s : Set α} (h : a ∈ s) : insert a s = s :=
  ext fun _ => or_iff_right_of_imp fun e => e.symm ▸ h


theorem ne_insert_of_not_mem {s : Set α} (t : Set α) {a : α} : a ∉ s → s ≠ insert a t :=
  mt fun e => e.symm ▸ mem_insert _ _


@[simp]
theorem insert_eq_self : insert a s = s ↔ a ∈ s :=
  ⟨fun h => h ▸ mem_insert _ _, insert_eq_of_mem⟩


theorem insert_ne_self : insert a s ≠ s ↔ a ∉ s :=
  insert_eq_self.not


theorem insert_subset_iff : insert a s ⊆ t ↔ a ∈ t ∧ s ⊆ t := by
  /-
    α : Type u
    a : α
    s t : Set α
    ⊢ Iff (HasSubset.Subset (Insert.insert a s) t) (And (Membership.mem t a) (HasS …
  -/
  simp only [subset_def, mem_insert_iff, or_imp, forall_and, forall_eq]
  /-
    🎉 no goals
  -/


theorem insert_subset (ha : a ∈ t) (hs : s ⊆ t) : insert a s ⊆ t :=
  insert_subset_iff.mpr ⟨ha, hs⟩


theorem insert_subset_insert (h : s ⊆ t) : insert a s ⊆ insert a t := fun _ => Or.imp_right (@h _)


@[simp] theorem insert_subset_insert_iff (ha : a ∉ s) : insert a s ⊆ insert a t ↔ s ⊆ t := by
  /-
    α : Type u
    a : α
    s t : Set α
    ha : Not (Membership.mem s a)
    ⊢ Iff (HasSubset.Subset (Insert.insert a s) (Insert.insert a t)) (HasSubset.Su …
  -/
  refine ⟨fun h x hx => ?_, insert_subset_insert⟩
  /-
    α : Type u
    a : α
    s t : Set α
    ha : Not (Membership.mem s a)
    h : HasSubset.Subset (Insert.insert a s) (Insert.insert a t)
    x : α
    hx : Membership.mem s x
    ⊢ Membership.mem t x
  -/
  rcases h (subset_insert _ _ hx) with (rfl | hxt)
  /-
    case inl
    α : Type u
    s t : Set α
    x : α
    hx : Membership.mem s x
    ha : Not (Membership.mem s x)
    h : HasSubset.Subset (Insert.insert x s) (Insert.insert x t)
    ⊢ Membership.mem t x
  -/
  exacts [(ha hx).elim, hxt]
  /-
    🎉 no goals
  -/


theorem subset_insert_iff_of_not_mem (ha : a ∉ s) : s ⊆ insert a t ↔ s ⊆ t :=
  forall₂_congr fun _ hb => or_iff_right <| ne_of_mem_of_not_mem hb ha


theorem ssubset_iff_insert {s t : Set α} : s ⊂ t ↔ ∃ a ∉ s, insert a s ⊆ t := by
  /-
    α : Type u
    s t : Set α
    ⊢ Iff (HasSSubset.SSubset s t) (Exists fun a => And (Not (Membership.mem s a)) …
  -/
  simp only [insert_subset_iff, exists_and_right, ssubset_def, not_subset]
  /-
    α : Type u
    s t : Set α
    ⊢ Iff (And (HasSubset.Subset s t) (Exists fun a => And (Membership.mem t a) (N …
  -/
  aesop
  /-
    🎉 no goals
  -/


theorem ssubset_insert {s : Set α} {a : α} (h : a ∉ s) : s ⊂ insert a s :=
  ssubset_iff_insert.2 ⟨a, h, Subset.rfl⟩


theorem insert_comm (a b : α) (s : Set α) : insert a (insert b s) = insert b (insert a s) :=
  ext fun _ => or_left_comm


theorem insert_idem (a : α) (s : Set α) : insert a (insert a s) = insert a s :=
  insert_eq_of_mem <| mem_insert _ _


theorem insert_union : insert a s ∪ t = insert a (s ∪ t) :=
  ext fun _ => or_assoc


@[simp]
theorem union_insert : s ∪ insert a t = insert a (s ∪ t) :=
  ext fun _ => or_left_comm


@[simp]
theorem insert_nonempty (a : α) (s : Set α) : (insert a s).Nonempty :=
  ⟨a, mem_insert a s⟩


instance (a : α) (s : Set α) : Nonempty (insert a s : Set α) :=
  (insert_nonempty a s).to_subtype


theorem insert_inter_distrib (a : α) (s t : Set α) : insert a (s ∩ t) = insert a s ∩ insert a t :=
  ext fun _ => or_and_left


theorem insert_union_distrib (a : α) (s t : Set α) : insert a (s ∪ t) = insert a s ∪ insert a t :=
  ext fun _ => or_or_distrib_left


theorem insert_inj (ha : a ∉ s) : insert a s = insert b s ↔ a = b :=
  ⟨fun h => eq_of_not_mem_of_mem_insert (h ▸ mem_insert a s) ha,
    congr_arg (fun x => insert x s)⟩

-- useful in proofs by induction

theorem forall_of_forall_insert {P : α → Prop} {a : α} {s : Set α} (H : ∀ x, x ∈ insert a s → P x)
    (x) (h : x ∈ s) : P x :=
  H _ (Or.inr h)


theorem forall_insert_of_forall {P : α → Prop} {a : α} {s : Set α} (H : ∀ x, x ∈ s → P x) (ha : P a)
    (x) (h : x ∈ insert a s) : P x :=
  h.elim (fun e => e.symm ▸ ha) (H _)

/- Porting note: ∃ x ∈ insert a s, P x is parsed as ∃ x, x ∈ insert a s ∧ P x,
 where in Lean3 it was parsed as `∃ x, ∃ (h : x ∈ insert a s), P x` -/

theorem exists_mem_insert {P : α → Prop} {a : α} {s : Set α} :
    (∃ x ∈ insert a s, P x) ↔ (P a ∨ ∃ x ∈ s, P x) := by
  /-
    α : Type u
    P : α → Prop
    a : α
    s : Set α
    ⊢ Iff (Exists fun x => And (Membership.mem (Insert.insert a s) x) (P x)) (Or ( …
  -/
  simp [mem_insert_iff, or_and_right, exists_and_left, exists_or]
  /-
    🎉 no goals
  -/

@[deprecated (since := "2024-03-23")] alias bex_insert_iff := exists_mem_insert


theorem forall_mem_insert {P : α → Prop} {a : α} {s : Set α} :
    (∀ x ∈ insert a s, P x) ↔ P a ∧ ∀ x ∈ s, P x :=
  forall₂_or_left.trans <| and_congr_left' forall_eq

@[deprecated (since := "2024-03-23")] alias ball_insert_iff := forall_mem_insert


/-- Inserting an element to a set is equivalent to the option type. -/
def subtypeInsertEquivOption
    [DecidableEq α] {t : Set α} {x : α} (h : x ∉ t) :
    { i // i ∈ insert x t } ≃ Option { i // i ∈ t } where
  toFun y := if h : ↑y = x then none else some ⟨y, (mem_insert_iff.mp y.2).resolve_left h⟩
  invFun y := (y.elim ⟨x, mem_insert _ _⟩) fun z => ⟨z, mem_insert_of_mem _ z.2⟩
  left_inv y := by
    /-
      α : Type u
      β : Type v
      a b : α
      s s₁ s₂ t✝ t₁ t₂ u : Set α
      inst✝ : DecidableEq α
      t : Set α
      x : α
      h : Not (Membership.mem t x)
      y : Subtype fun i => Membership.mem (Insert.insert x t) i
      ⊢ Eq ((fun y => y.elim ⟨x, ⋯⟩ fun z => ⟨↑z, ⋯⟩) ((fun y => dite (Eq (↑y) x) (f …
    -/
    by_cases h : ↑y = x
      /-
        case pos
        α : Type u
        β : Type v
        a b : α
        s s₁ s₂ t✝ t₁ t₂ u : Set α
        inst✝ : DecidableEq α
        t : Set α
        x : α
        h✝ : Not (Membership.mem t x)
        y : Subtype fun i => Membership.mem (Insert.insert x t) i
        h : Eq (↑y) x
        ⊢ Eq ((fun y => y.elim ⟨x, ⋯⟩ fun z => ⟨↑z, ⋯⟩) ((fun y => dite (Eq (↑y) x) (f …
      -/
    · simp only [Subtype.ext_iff, h, Option.elim, dif_pos, Subtype.coe_mk]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u
        β : Type v
        a b : α
        s s₁ s₂ t✝ t₁ t₂ u : Set α
        inst✝ : DecidableEq α
        t : Set α
        x : α
        h✝ : Not (Membership.mem t x)
        y : Subtype fun i => Membership.mem (Insert.insert x t) i
        h : Not (Eq (↑y) x)
        ⊢ Eq ((fun y => y.elim ⟨x, ⋯⟩ fun z => ⟨↑z, ⋯⟩) ((fun y => dite (Eq (↑y) x) (f …
      -/
    · simp only [h, Option.elim, dif_neg, not_false_iff, Subtype.coe_eta, Subtype.coe_mk]
      /-
        🎉 no goals
      -/
  right_inv := by
    /-
      α : Type u
      β : Type v
      a b : α
      s s₁ s₂ t✝ t₁ t₂ u : Set α
      inst✝ : DecidableEq α
      t : Set α
      x : α
      h : Not (Membership.mem t x)
      ⊢ Function.RightInverse (fun y => y.elim ⟨x, ⋯⟩ fun z => ⟨↑z, ⋯⟩) fun y => dit …
    -/
    rintro (_ | y)
      /-
        case none
        α : Type u
        β : Type v
        a b : α
        s s₁ s₂ t✝ t₁ t₂ u : Set α
        inst✝ : DecidableEq α
        t : Set α
        x : α
        h : Not (Membership.mem t x)
        ⊢ Eq ((fun y => dite (Eq (↑y) x) (fun h => Option.none) fun h => Option.some ⟨ …
      -/
    · simp only [Option.elim, dif_pos]
      /-
        🎉 no goals
      -/
    · have : ↑y ≠ x := by
        rintro ⟨⟩
        exact h y.2
      /-
        case some
        α : Type u
        β : Type v
        a b : α
        s s₁ s₂ t✝ t₁ t₂ u : Set α
        inst✝ : DecidableEq α
        t : Set α
        x : α
        h : Not (Membership.mem t x)
        y : Subtype fun i => Membership.mem t i
        this : Ne (↑y) x
        ⊢ Eq ((fun y => dite (Eq (↑y) x) (fun h => Option.none) fun h => Option.some ⟨ …
      -/
      simp only [this, Option.elim, Subtype.eta, dif_neg, not_false_iff, Subtype.coe_mk]
      /-
        🎉 no goals
      -/


instance : LawfulSingleton α (Set α) :=
  ⟨fun x => Set.ext fun a => by
    /-
      α : Type u
      β : Type v
      a✝ b : α
      s s₁ s₂ t t₁ t₂ u : Set α
      x a : α
      ⊢ Iff (Membership.mem (Insert.insert x EmptyCollection.emptyCollection) a) (Me …
    -/
    simp only [mem_empty_iff_false, mem_insert_iff, or_false]
    /-
      α : Type u
      β : Type v
      a✝ b : α
      s s₁ s₂ t t₁ t₂ u : Set α
      x a : α
      ⊢ Iff (Eq a x) (Membership.mem (Singleton.singleton x) a)
    -/
    exact Iff.rfl⟩
    /-
      🎉 no goals
    -/


theorem singleton_def (a : α) : ({a} : Set α) = insert a ∅ :=
  (insert_emptyc_eq a).symm


@[simp]
theorem mem_singleton_iff {a b : α} : a ∈ ({b} : Set α) ↔ a = b :=
  Iff.rfl


theorem not_mem_singleton_iff {a b : α} : a ∉ ({b} : Set α) ↔ a ≠ b :=
  Iff.rfl


@[simp]
theorem setOf_eq_eq_singleton {a : α} : { n | n = a } = {a} :=
  rfl


@[simp]
theorem setOf_eq_eq_singleton' {a : α} : { x | a = x } = {a} :=
  ext fun _ => eq_comm

-- TODO: again, annotation needed
--Porting note (https://github.com/leanprover-community/mathlib4/issues/11119): removed `simp` attribute

theorem mem_singleton (a : α) : a ∈ ({a} : Set α) :=
  @rfl _ _


theorem eq_of_mem_singleton {x y : α} (h : x ∈ ({y} : Set α)) : x = y :=
  h


@[simp]
theorem singleton_eq_singleton_iff {x y : α} : {x} = ({y} : Set α) ↔ x = y :=
  Set.ext_iff.trans eq_iff_eq_cancel_left


theorem singleton_injective : Injective (singleton : α → Set α) := fun _ _ =>
  singleton_eq_singleton_iff.mp


theorem mem_singleton_of_eq {x y : α} (H : x = y) : x ∈ ({y} : Set α) :=
  H


theorem insert_eq (x : α) (s : Set α) : insert x s = ({x} : Set α) ∪ s :=
  rfl


@[simp]
theorem singleton_nonempty (a : α) : ({a} : Set α).Nonempty :=
  ⟨a, rfl⟩


@[simp]
theorem singleton_ne_empty (a : α) : ({a} : Set α) ≠ ∅ :=
  (singleton_nonempty _).ne_empty


theorem empty_ssubset_singleton : (∅ : Set α) ⊂ {a} :=
  (singleton_nonempty _).empty_ssubset


@[simp]
theorem singleton_subset_iff {a : α} {s : Set α} : {a} ⊆ s ↔ a ∈ s :=
  forall_eq


                                                                       /-
                                                                         α : Type u
                                                                         a b : α
                                                                         ⊢ Iff (HasSubset.Subset (Singleton.singleton a) (Singleton.singleton b)) (Eq a …
                                                                       -/
theorem singleton_subset_singleton : ({a} : Set α) ⊆ {b} ↔ a = b := by simp
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


theorem set_compr_eq_eq_singleton {a : α} : { b | b = a } = {a} :=
  rfl


@[simp]
theorem singleton_union : {a} ∪ s = insert a s :=
  rfl


@[simp]
theorem union_singleton : s ∪ {a} = insert a s :=
  union_comm _ _


@[simp]
theorem singleton_inter_nonempty : ({a} ∩ s).Nonempty ↔ a ∈ s := by
  /-
    α : Type u
    a : α
    s : Set α
    ⊢ Iff (Inter.inter (Singleton.singleton a) s).Nonempty (Membership.mem s a)
  -/
  simp only [Set.Nonempty, mem_inter_iff, mem_singleton_iff, exists_eq_left]
  /-
    🎉 no goals
  -/


@[simp]
theorem inter_singleton_nonempty : (s ∩ {a}).Nonempty ↔ a ∈ s := by
  /-
    α : Type u
    a : α
    s : Set α
    ⊢ Iff (Inter.inter s (Singleton.singleton a)).Nonempty (Membership.mem s a)
  -/
  rw [inter_comm, singleton_inter_nonempty]
  /-
    🎉 no goals
  -/


@[simp]
theorem singleton_inter_eq_empty : {a} ∩ s = ∅ ↔ a ∉ s :=
  not_nonempty_iff_eq_empty.symm.trans singleton_inter_nonempty.not


@[simp]
theorem inter_singleton_eq_empty : s ∩ {a} = ∅ ↔ a ∉ s := by
  /-
    α : Type u
    a : α
    s : Set α
    ⊢ Iff (Eq (Inter.inter s (Singleton.singleton a)) EmptyCollection.emptyCollect …
  -/
  rw [inter_comm, singleton_inter_eq_empty]
  /-
    🎉 no goals
  -/


theorem nmem_singleton_empty {s : Set α} : s ∉ ({∅} : Set (Set α)) ↔ s.Nonempty :=
  nonempty_iff_ne_empty.symm


instance uniqueSingleton (a : α) : Unique (↥({a} : Set α)) :=
  ⟨⟨⟨a, mem_singleton a⟩⟩, fun ⟨_, h⟩ => Subtype.eq h⟩


theorem eq_singleton_iff_unique_mem : s = {a} ↔ a ∈ s ∧ ∀ x ∈ s, x = a :=
  Subset.antisymm_iff.trans <| and_comm.trans <| and_congr_left' singleton_subset_iff


theorem eq_singleton_iff_nonempty_unique_mem : s = {a} ↔ s.Nonempty ∧ ∀ x ∈ s, x = a :=
  eq_singleton_iff_unique_mem.trans <|
    and_congr_left fun H => ⟨fun h' => ⟨_, h'⟩, fun ⟨x, h⟩ => H x h ▸ h⟩

-- while `simp` is capable of proving this, it is not capable of turning the LHS into the RHS.

@[simp]
theorem default_coe_singleton (x : α) : (default : ({x} : Set α)) = ⟨x, rfl⟩ :=
  rfl


theorem mem_sep (xs : x ∈ s) (px : p x) : x ∈ { x ∈ s | p x } :=
  ⟨xs, px⟩


@[simp]
theorem sep_mem_eq : { x ∈ s | x ∈ t } = s ∩ t :=
  rfl


@[simp]
theorem mem_sep_iff : x ∈ { x ∈ s | p x } ↔ x ∈ s ∧ p x :=
  Iff.rfl


theorem sep_ext_iff : { x ∈ s | p x } = { x ∈ s | q x } ↔ ∀ x ∈ s, p x ↔ q x := by
  /-
    α : Type u
    s : Set α
    p q : α → Prop
    ⊢ Iff (Eq (setOf fun x => And (Membership.mem s x) (p x)) (setOf fun x => And  …
  -/
  simp_rw [Set.ext_iff, mem_sep_iff, and_congr_right_iff]
  /-
    🎉 no goals
  -/


theorem sep_eq_of_subset (h : s ⊆ t) : { x ∈ t | x ∈ s } = s :=
  inter_eq_self_of_subset_right h


@[simp]
theorem sep_subset (s : Set α) (p : α → Prop) : { x ∈ s | p x } ⊆ s := fun _ => And.left


@[simp]
theorem sep_eq_self_iff_mem_true : { x ∈ s | p x } = s ↔ ∀ x ∈ s, p x := by
  /-
    α : Type u
    s : Set α
    p : α → Prop
    ⊢ Iff (Eq (setOf fun x => And (Membership.mem s x) (p x)) s) (∀ (x : α), Membe …
  -/
  simp_rw [Set.ext_iff, mem_sep_iff, and_iff_left_iff_imp]
  /-
    🎉 no goals
  -/


@[simp]
theorem sep_eq_empty_iff_mem_false : { x ∈ s | p x } = ∅ ↔ ∀ x ∈ s, ¬p x := by
  /-
    α : Type u
    s : Set α
    p : α → Prop
    ⊢ Iff (Eq (setOf fun x => And (Membership.mem s x) (p x)) EmptyCollection.empt …
  -/
  simp_rw [Set.ext_iff, mem_sep_iff, mem_empty_iff_false, iff_false, not_and]
  /-
    🎉 no goals
  -/


theorem sep_true : { x ∈ s | True } = s :=
  inter_univ s


theorem sep_false : { x ∈ s | False } = ∅ :=
  inter_empty s


theorem sep_empty (p : α → Prop) : { x ∈ (∅ : Set α) | p x } = ∅ :=
  empty_inter {x | p x}


theorem sep_univ : { x ∈ (univ : Set α) | p x } = { x | p x } :=
  univ_inter {x | p x}


@[simp]
theorem sep_union : { x | (x ∈ s ∨ x ∈ t) ∧ p x } = { x ∈ s | p x } ∪ { x ∈ t | p x } :=
  union_inter_distrib_right { x | x ∈ s } { x | x ∈ t } p


@[simp]
theorem sep_inter : { x | (x ∈ s ∧ x ∈ t) ∧ p x } = { x ∈ s | p x } ∩ { x ∈ t | p x } :=
  inter_inter_distrib_right s t {x | p x}


@[simp]
theorem sep_and : { x ∈ s | p x ∧ q x } = { x ∈ s | p x } ∩ { x ∈ s | q x } :=
  inter_inter_distrib_left s {x | p x} {x | q x}


@[simp]
theorem sep_or : { x ∈ s | p x ∨ q x } = { x ∈ s | p x } ∪ { x ∈ s | q x } :=
  inter_union_distrib_left s p q


@[simp]
theorem sep_setOf : { x ∈ { y | p y } | q x } = { x | p x ∧ q x } :=
  rfl


@[simp]
theorem subset_singleton_iff {α : Type*} {s : Set α} {x : α} : s ⊆ {x} ↔ ∀ y ∈ s, y = x :=
  Iff.rfl


theorem subset_singleton_iff_eq {s : Set α} {x : α} : s ⊆ {x} ↔ s = ∅ ∨ s = {x} := by
  /-
    α : Type u
    s : Set α
    x : α
    ⊢ Iff (HasSubset.Subset s (Singleton.singleton x)) (Or (Eq s EmptyCollection.e …
  -/
  obtain rfl | hs := s.eq_empty_or_nonempty
    /-
      case inl
      α : Type u
      x : α
      ⊢ Iff (HasSubset.Subset EmptyCollection.emptyCollection (Singleton.singleton x …
    -/
  · exact ⟨fun _ => Or.inl rfl, fun _ => empty_subset _⟩
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u
      s : Set α
      x : α
      hs : s.Nonempty
      ⊢ Iff (HasSubset.Subset s (Singleton.singleton x)) (Or (Eq s EmptyCollection.e …
    -/
  · simp [eq_singleton_iff_nonempty_unique_mem, hs, hs.ne_empty]
    /-
      🎉 no goals
    -/


theorem Nonempty.subset_singleton_iff (h : s.Nonempty) : s ⊆ {a} ↔ s = {a} :=
  subset_singleton_iff_eq.trans <| or_iff_right h.ne_empty


theorem ssubset_singleton_iff {s : Set α} {x : α} : s ⊂ {x} ↔ s = ∅ := by
  rw [ssubset_iff_subset_ne, subset_singleton_iff_eq, or_and_right, and_not_self_iff, or_false,
    and_iff_left_iff_imp]
  /-
    α : Type u
    s : Set α
    x : α
    ⊢ Eq s EmptyCollection.emptyCollection → Ne s (Singleton.singleton x)
  -/
  exact fun h => h ▸ (singleton_ne_empty _).symm
  /-
    🎉 no goals
  -/


theorem eq_empty_of_ssubset_singleton {s : Set α} {x : α} (hs : s ⊂ {x}) : s = ∅ :=
  ssubset_singleton_iff.1 hs


theorem eq_of_nonempty_of_subsingleton {α} [Subsingleton α] (s t : Set α) [Nonempty s]
    [Nonempty t] : s = t :=
  Nonempty.of_subtype.eq_univ.trans Nonempty.of_subtype.eq_univ.symm


theorem eq_of_nonempty_of_subsingleton' {α} [Subsingleton α] {s : Set α} (t : Set α)
    (hs : s.Nonempty) [Nonempty t] : s = t :=
  have := hs.to_subtype; eq_of_nonempty_of_subsingleton s t


theorem Nonempty.eq_zero [Subsingleton α] [Zero α] {s : Set α} (h : s.Nonempty) :
    s = {0} := eq_of_nonempty_of_subsingleton' {0} h


theorem Nonempty.eq_one [Subsingleton α] [One α] {s : Set α} (h : s.Nonempty) :
    s = {1} := eq_of_nonempty_of_subsingleton' {1} h


protected theorem disjoint_iff : Disjoint s t ↔ s ∩ t ⊆ ∅ :=
  disjoint_iff_inf_le


theorem disjoint_iff_inter_eq_empty : Disjoint s t ↔ s ∩ t = ∅ :=
  disjoint_iff


theorem _root_.Disjoint.inter_eq : Disjoint s t → s ∩ t = ∅ :=
  Disjoint.eq_bot


theorem disjoint_left : Disjoint s t ↔ ∀ ⦃a⦄, a ∈ s → a ∉ t :=
  disjoint_iff_inf_le.trans <| forall_congr' fun _ => not_and


                                                                   /-
                                                                     α : Type u
                                                                     s t : Set α
                                                                     ⊢ Iff (Disjoint s t) (∀ ⦃a : α⦄, Membership.mem t a → Not (Membership.mem s a))
                                                                   -/
theorem disjoint_right : Disjoint s t ↔ ∀ ⦃a⦄, a ∈ t → a ∉ s := by rw [disjoint_comm, disjoint_left]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


lemma not_disjoint_iff : ¬Disjoint s t ↔ ∃ x, x ∈ s ∧ x ∈ t :=
  Set.disjoint_iff.not.trans <| not_forall.trans <| exists_congr fun _ ↦ not_not


lemma not_disjoint_iff_nonempty_inter : ¬ Disjoint s t ↔ (s ∩ t).Nonempty := not_disjoint_iff


alias ⟨_, Nonempty.not_disjoint⟩ := not_disjoint_iff_nonempty_inter


lemma disjoint_or_nonempty_inter (s t : Set α) : Disjoint s t ∨ (s ∩ t).Nonempty :=
  (em _).imp_right not_disjoint_iff_nonempty_inter.1


lemma disjoint_iff_forall_ne : Disjoint s t ↔ ∀ ⦃a⦄, a ∈ s → ∀ ⦃b⦄, b ∈ t → a ≠ b := by
  /-
    α : Type u
    s t : Set α
    ⊢ Iff (Disjoint s t) (∀ ⦃a : α⦄, Membership.mem s a → ∀ ⦃b : α⦄, Membership.me …
  -/
  simp only [Ne, disjoint_left, @imp_not_comm _ (_ = _), forall_eq']
  /-
    🎉 no goals
  -/


alias ⟨_root_.Disjoint.ne_of_mem, _⟩ := disjoint_iff_forall_ne


lemma disjoint_of_subset_left (h : s ⊆ u) (d : Disjoint u t) : Disjoint s t := d.mono_left h

lemma disjoint_of_subset_right (h : t ⊆ u) (d : Disjoint s u) : Disjoint s t := d.mono_right h


lemma disjoint_of_subset (hs : s₁ ⊆ s₂) (ht : t₁ ⊆ t₂) (h : Disjoint s₂ t₂) : Disjoint s₁ t₁ :=
  h.mono hs ht


@[simp]
lemma disjoint_union_left : Disjoint (s ∪ t) u ↔ Disjoint s u ∧ Disjoint t u := disjoint_sup_left


@[simp]
lemma disjoint_union_right : Disjoint s (t ∪ u) ↔ Disjoint s t ∧ Disjoint s u := disjoint_sup_right


@[simp] lemma disjoint_empty (s : Set α) : Disjoint s ∅ := disjoint_bot_right

@[simp] lemma empty_disjoint (s : Set α) : Disjoint ∅ s := disjoint_bot_left


@[simp] lemma univ_disjoint : Disjoint univ s ↔ s = ∅ := top_disjoint

@[simp] lemma disjoint_univ : Disjoint s univ ↔ s = ∅ := disjoint_top


lemma disjoint_sdiff_left : Disjoint (t \ s) s := disjoint_sdiff_self_left


lemma disjoint_sdiff_right : Disjoint s (t \ s) := disjoint_sdiff_self_right

-- TODO: prove this in terms of a lattice lemma

theorem disjoint_sdiff_inter : Disjoint (s \ t) (s ∩ t) :=
  disjoint_of_subset_right inter_subset_right disjoint_sdiff_left


theorem diff_union_diff_cancel (hts : t ⊆ s) (hut : u ⊆ t) : s \ t ∪ t \ u = s \ u :=
  sdiff_sup_sdiff_cancel hts hut


theorem diff_diff_eq_sdiff_union (h : u ⊆ s) : s \ (t \ u) = s \ t ∪ u := sdiff_sdiff_eq_sdiff_sup h


@[simp default+1]
                                                             /-
                                                               α : Type u
                                                               a : α
                                                               s : Set α
                                                               ⊢ Iff (Disjoint (Singleton.singleton a) s) (Not (Membership.mem s a))
                                                             -/
lemma disjoint_singleton_left : Disjoint {a} s ↔ a ∉ s := by simp [Set.disjoint_iff, subset_def]
                                                             /-
                                                               🎉 no goals
                                                             -/


@[simp]
lemma disjoint_singleton_right : Disjoint s {a} ↔ a ∉ s :=
  disjoint_comm.trans disjoint_singleton_left


lemma disjoint_singleton : Disjoint ({a} : Set α) {b} ↔ a ≠ b := by
  /-
    α : Type u
    a b : α
    ⊢ Iff (Disjoint (Singleton.singleton a) (Singleton.singleton b)) (Ne a b)
  -/
  simp
  /-
    🎉 no goals
  -/


lemma subset_diff : s ⊆ t \ u ↔ s ⊆ t ∧ Disjoint s u := le_iff_subset.symm.trans le_sdiff


lemma ssubset_iff_sdiff_singleton : s ⊂ t ↔ ∃ a ∈ t, s ⊆ t \ {a} := by
  /-
    α : Type u
    s t : Set α
    ⊢ Iff (HasSSubset.SSubset s t) (Exists fun a => And (Membership.mem t a) (HasS …
  -/
  simp [ssubset_iff_insert, subset_diff, insert_subset_iff]; aesop
                                                             /-
                                                               🎉 no goals
                                                             -/


theorem inter_diff_distrib_left (s t u : Set α) : s ∩ (t \ u) = (s ∩ t) \ (s ∩ u) :=
  inf_sdiff_distrib_left _ _ _


theorem inter_diff_distrib_right (s t u : Set α) : s \ t ∩ u = (s ∩ u) \ (t ∩ u) :=
  inf_sdiff_distrib_right _ _ _


theorem disjoint_of_subset_iff_left_eq_empty (h : s ⊆ t) :
    Disjoint s t ↔ s = ∅ := by
  /-
    α : Type u
    s t : Set α
    h : HasSubset.Subset s t
    ⊢ Iff (Disjoint s t) (Eq s EmptyCollection.emptyCollection)
  -/
  simp only [disjoint_iff, inf_eq_left.mpr h, bot_eq_empty]
  /-
    🎉 no goals
  -/


theorem compl_def (s : Set α) : sᶜ = { x | x ∉ s } :=
  rfl


theorem mem_compl {s : Set α} {x : α} (h : x ∉ s) : x ∈ sᶜ :=
  h


theorem compl_setOf {α} (p : α → Prop) : { a | p a }ᶜ = { a | ¬p a } :=
  rfl


theorem not_mem_of_mem_compl {s : Set α} {x : α} (h : x ∈ sᶜ) : x ∉ s :=
  h


theorem not_mem_compl_iff {x : α} : x ∉ sᶜ ↔ x ∈ s :=
  not_not


@[simp]
theorem inter_compl_self (s : Set α) : s ∩ sᶜ = ∅ :=
  inf_compl_eq_bot


@[simp]
theorem compl_inter_self (s : Set α) : sᶜ ∩ s = ∅ :=
  compl_inf_eq_bot


@[simp]
theorem compl_empty : (∅ : Set α)ᶜ = univ :=
  compl_bot


@[simp]
theorem compl_union (s t : Set α) : (s ∪ t)ᶜ = sᶜ ∩ tᶜ :=
  compl_sup


theorem compl_inter (s t : Set α) : (s ∩ t)ᶜ = sᶜ ∪ tᶜ :=
  compl_inf


@[simp]
theorem compl_univ : (univ : Set α)ᶜ = ∅ :=
  compl_top


@[simp]
theorem compl_empty_iff {s : Set α} : sᶜ = ∅ ↔ s = univ :=
  compl_eq_bot


@[simp]
theorem compl_univ_iff {s : Set α} : sᶜ = univ ↔ s = ∅ :=
  compl_eq_top


theorem compl_ne_univ : sᶜ ≠ univ ↔ s.Nonempty :=
  compl_univ_iff.not.trans nonempty_iff_ne_empty.symm


theorem nonempty_compl : sᶜ.Nonempty ↔ s ≠ univ :=
  (ne_univ_iff_exists_not_mem s).symm


@[simp] lemma nonempty_compl_of_nontrivial [Nontrivial α] (x : α) : Set.Nonempty {x}ᶜ := by
  /-
    α : Type u
    inst✝ : Nontrivial α
    x : α
    ⊢ (HasCompl.compl (Singleton.singleton x)).Nonempty
  -/
  obtain ⟨y, hy⟩ := exists_ne x
  /-
    case intro
    α : Type u
    inst✝ : Nontrivial α
    x y : α
    hy : Ne y x
    ⊢ (HasCompl.compl (Singleton.singleton x)).Nonempty
  -/
  exact ⟨y, by simp [hy]⟩
  /-
    🎉 no goals
  -/


theorem mem_compl_singleton_iff {a x : α} : x ∈ ({a} : Set α)ᶜ ↔ x ≠ a :=
  Iff.rfl


theorem compl_singleton_eq (a : α) : ({a} : Set α)ᶜ = { x | x ≠ a } :=
  rfl


@[simp]
theorem compl_ne_eq_singleton (a : α) : ({ x | x ≠ a } : Set α)ᶜ = {a} :=
  compl_compl _


theorem union_eq_compl_compl_inter_compl (s t : Set α) : s ∪ t = (sᶜ ∩ tᶜ)ᶜ :=
  ext fun _ => or_iff_not_and_not


theorem inter_eq_compl_compl_union_compl (s t : Set α) : s ∩ t = (sᶜ ∪ tᶜ)ᶜ :=
  ext fun _ => and_iff_not_or_not


@[simp]
theorem union_compl_self (s : Set α) : s ∪ sᶜ = univ :=
  eq_univ_iff_forall.2 fun _ => em _


@[simp]
                                                           /-
                                                             α : Type u
                                                             s : Set α
                                                             ⊢ Eq (Union.union (HasCompl.compl s) s) Set.univ
                                                           -/
theorem compl_union_self (s : Set α) : sᶜ ∪ s = univ := by rw [union_comm, union_compl_self]
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem compl_subset_comm : sᶜ ⊆ t ↔ tᶜ ⊆ s :=
  @compl_le_iff_compl_le _ s _ _


theorem subset_compl_comm : s ⊆ tᶜ ↔ t ⊆ sᶜ :=
  @le_compl_iff_le_compl _ _ _ t


@[simp]
theorem compl_subset_compl : sᶜ ⊆ tᶜ ↔ t ⊆ s :=
  @compl_le_compl_iff_le (Set α) _ _ _


@[gcongr] theorem compl_subset_compl_of_subset (h : t ⊆ s) : sᶜ ⊆ tᶜ := compl_subset_compl.2 h


theorem subset_compl_iff_disjoint_left : s ⊆ tᶜ ↔ Disjoint t s :=
  @le_compl_iff_disjoint_left (Set α) _ _ _


theorem subset_compl_iff_disjoint_right : s ⊆ tᶜ ↔ Disjoint s t :=
  @le_compl_iff_disjoint_right (Set α) _ _ _


theorem disjoint_compl_left_iff_subset : Disjoint sᶜ t ↔ t ⊆ s :=
  disjoint_compl_left_iff


theorem disjoint_compl_right_iff_subset : Disjoint s tᶜ ↔ s ⊆ t :=
  disjoint_compl_right_iff


alias ⟨_, _root_.Disjoint.subset_compl_right⟩ := subset_compl_iff_disjoint_right


alias ⟨_, _root_.Disjoint.subset_compl_left⟩ := subset_compl_iff_disjoint_left


alias ⟨_, _root_.HasSubset.Subset.disjoint_compl_left⟩ := disjoint_compl_left_iff_subset


alias ⟨_, _root_.HasSubset.Subset.disjoint_compl_right⟩ := disjoint_compl_right_iff_subset


theorem subset_union_compl_iff_inter_subset {s t u : Set α} : s ⊆ t ∪ uᶜ ↔ s ∩ u ⊆ t :=
  (@isCompl_compl _ u _).le_sup_right_iff_inf_left_le


theorem compl_subset_iff_union {s t : Set α} : sᶜ ⊆ t ↔ s ∪ t = univ :=
  Iff.symm <| eq_univ_iff_forall.trans <| forall_congr' fun _ => or_iff_not_imp_left


@[simp]
theorem subset_compl_singleton_iff {a : α} {s : Set α} : s ⊆ {a}ᶜ ↔ a ∉ s :=
  subset_compl_comm.trans singleton_subset_iff


theorem inter_subset (a b c : Set α) : a ∩ b ⊆ c ↔ a ⊆ bᶜ ∪ c :=
  forall_congr' fun _ => and_imp.trans <| imp_congr_right fun _ => imp_iff_not_or


theorem inter_compl_nonempty_iff {s t : Set α} : (s ∩ tᶜ).Nonempty ↔ ¬s ⊆ t :=
                                                /-
                                                  α : Type u
                                                  s t : Set α
                                                  x : α
                                                  ⊢ Iff (And (Membership.mem s x) (Not (Membership.mem t x))) (Membership.mem (I …
                                                -/
  (not_subset.trans <| exists_congr fun x => by simp [mem_compl]).symm
                                                /-
                                                  🎉 no goals
                                                -/


theorem not_mem_diff_of_mem {s t : Set α} {x : α} (hx : x ∈ t) : x ∉ s \ t := fun h => h.2 hx


theorem mem_of_mem_diff {s t : Set α} {x : α} (h : x ∈ s \ t) : x ∈ s :=
  h.left


theorem not_mem_of_mem_diff {s t : Set α} {x : α} (h : x ∈ s \ t) : x ∉ t :=
  h.right


                                                                 /-
                                                                   α : Type u
                                                                   s t : Set α
                                                                   ⊢ Eq (SDiff.sdiff s t) (Inter.inter (HasCompl.compl t) s)
                                                                 -/
theorem diff_eq_compl_inter {s t : Set α} : s \ t = tᶜ ∩ s := by rw [diff_eq, inter_comm]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


theorem diff_nonempty {s t : Set α} : (s \ t).Nonempty ↔ ¬s ⊆ t :=
  inter_compl_nonempty_iff

@[deprecated (since := "2024-08-27")] alias nonempty_diff := diff_nonempty


theorem diff_subset {s t : Set α} : s \ t ⊆ s := show s \ t ≤ s from sdiff_le


theorem diff_subset_compl (s t : Set α) : s \ t ⊆ tᶜ :=
  diff_eq_compl_inter ▸ inter_subset_left


theorem union_diff_cancel' {s t u : Set α} (h₁ : s ⊆ t) (h₂ : t ⊆ u) : t ∪ u \ s = u :=
  sup_sdiff_cancel' h₁ h₂


theorem union_diff_cancel {s t : Set α} (h : s ⊆ t) : s ∪ t \ s = t :=
  sup_sdiff_cancel_right h


theorem union_diff_cancel_left {s t : Set α} (h : s ∩ t ⊆ ∅) : (s ∪ t) \ s = t :=
  Disjoint.sup_sdiff_cancel_left <| disjoint_iff_inf_le.2 h


theorem union_diff_cancel_right {s t : Set α} (h : s ∩ t ⊆ ∅) : (s ∪ t) \ t = s :=
  Disjoint.sup_sdiff_cancel_right <| disjoint_iff_inf_le.2 h


@[simp]
theorem union_diff_left {s t : Set α} : (s ∪ t) \ s = t \ s :=
  sup_sdiff_left_self


@[simp]
theorem union_diff_right {s t : Set α} : (s ∪ t) \ t = s \ t :=
  sup_sdiff_right_self


theorem union_diff_distrib {s t u : Set α} : (s ∪ t) \ u = s \ u ∪ t \ u :=
  sup_sdiff


theorem inter_diff_assoc (a b c : Set α) : (a ∩ b) \ c = a ∩ (b \ c) :=
  inf_sdiff_assoc


@[simp]
theorem inter_diff_self (a b : Set α) : a ∩ (b \ a) = ∅ :=
  inf_sdiff_self_right


@[simp]
theorem inter_union_diff (s t : Set α) : s ∩ t ∪ s \ t = s :=
  sup_inf_sdiff s t


@[simp]
theorem diff_union_inter (s t : Set α) : s \ t ∪ s ∩ t = s := by
  /-
    α : Type u
    s t : Set α
    ⊢ Eq (Union.union (SDiff.sdiff s t) (Inter.inter s t)) s
  -/
  rw [union_comm]
  /-
    α : Type u
    s t : Set α
    ⊢ Eq (Union.union (Inter.inter s t) (SDiff.sdiff s t)) s
  -/
  exact sup_inf_sdiff _ _
  /-
    🎉 no goals
  -/


@[simp]
theorem inter_union_compl (s t : Set α) : s ∩ t ∪ s ∩ tᶜ = s :=
  inter_union_diff _ _


@[gcongr]
theorem diff_subset_diff {s₁ s₂ t₁ t₂ : Set α} : s₁ ⊆ s₂ → t₂ ⊆ t₁ → s₁ \ t₁ ⊆ s₂ \ t₂ :=
  show s₁ ≤ s₂ → t₂ ≤ t₁ → s₁ \ t₁ ≤ s₂ \ t₂ from sdiff_le_sdiff


@[gcongr]
theorem diff_subset_diff_left {s₁ s₂ t : Set α} (h : s₁ ⊆ s₂) : s₁ \ t ⊆ s₂ \ t :=
  sdiff_le_sdiff_right ‹s₁ ≤ s₂›


@[gcongr]
theorem diff_subset_diff_right {s t u : Set α} (h : t ⊆ u) : s \ u ⊆ s \ t :=
  sdiff_le_sdiff_left ‹t ≤ u›


theorem diff_subset_diff_iff_subset {r : Set α} (hs : s ⊆ r) (ht : t ⊆ r) :
    r \ s ⊆ r \ t ↔ t ⊆ s :=
  sdiff_le_sdiff_iff_le hs ht


theorem compl_eq_univ_diff (s : Set α) : sᶜ = univ \ s :=
  top_sdiff.symm


@[simp]
theorem empty_diff (s : Set α) : (∅ \ s : Set α) = ∅ :=
  bot_sdiff


theorem diff_eq_empty {s t : Set α} : s \ t = ∅ ↔ s ⊆ t :=
  sdiff_eq_bot_iff


@[simp]
theorem diff_empty {s : Set α} : s \ ∅ = s :=
  sdiff_bot


@[simp]
theorem diff_univ (s : Set α) : s \ univ = ∅ :=
  diff_eq_empty.2 (subset_univ s)


theorem diff_diff {u : Set α} : (s \ t) \ u = s \ (t ∪ u) :=
  sdiff_sdiff_left

-- the following statement contains parentheses to help the reader

theorem diff_diff_comm {s t u : Set α} : (s \ t) \ u = (s \ u) \ t :=
  sdiff_sdiff_comm


theorem diff_subset_iff {s t u : Set α} : s \ t ⊆ u ↔ s ⊆ t ∪ u :=
  show s \ t ≤ u ↔ s ≤ t ∪ u from sdiff_le_iff


theorem subset_diff_union (s t : Set α) : s ⊆ s \ t ∪ t :=
  show s ≤ s \ t ∪ t from le_sdiff_sup


theorem diff_union_of_subset {s t : Set α} (h : t ⊆ s) : s \ t ∪ t = s :=
  Subset.antisymm (union_subset diff_subset h) (subset_diff_union _ _)


@[simp]
theorem diff_singleton_subset_iff {x : α} {s t : Set α} : s \ {x} ⊆ t ↔ s ⊆ insert x t := by
  /-
    α : Type u
    x : α
    s t : Set α
    ⊢ Iff (HasSubset.Subset (SDiff.sdiff s (Singleton.singleton x)) t) (HasSubset. …
  -/
  rw [← union_singleton, union_comm]
  /-
    α : Type u
    x : α
    s t : Set α
    ⊢ Iff (HasSubset.Subset (SDiff.sdiff s (Singleton.singleton x)) t) (HasSubset. …
  -/
  apply diff_subset_iff
  /-
    🎉 no goals
  -/


theorem subset_diff_singleton {x : α} {s t : Set α} (h : s ⊆ t) (hx : x ∉ s) : s ⊆ t \ {x} :=
  subset_inter h <| subset_compl_comm.1 <| singleton_subset_iff.2 hx


theorem subset_insert_diff_singleton (x : α) (s : Set α) : s ⊆ insert x (s \ {x}) := by
  /-
    α : Type u
    x : α
    s : Set α
    ⊢ HasSubset.Subset s (Insert.insert x (SDiff.sdiff s (Singleton.singleton x)))
  -/
  rw [← diff_singleton_subset_iff]
  /-
    🎉 no goals
  -/


theorem diff_subset_comm {s t u : Set α} : s \ t ⊆ u ↔ s \ u ⊆ t :=
  show s \ t ≤ u ↔ s \ u ≤ t from sdiff_le_comm


theorem diff_inter {s t u : Set α} : s \ (t ∩ u) = s \ t ∪ s \ u :=
  sdiff_inf


theorem diff_inter_diff {s t u : Set α} : s \ t ∩ (s \ u) = s \ (t ∪ u) :=
  sdiff_sup.symm


theorem diff_compl : s \ tᶜ = s ∩ t :=
  sdiff_compl


theorem diff_diff_right {s t u : Set α} : s \ (t \ u) = s \ t ∪ s ∩ u :=
  sdiff_sdiff_right'


theorem diff_insert_of_not_mem {x : α} (h : x ∉ s) : s \ insert x t = s \ t := by
  /-
    α : Type u
    s t : Set α
    x : α
    h : Not (Membership.mem s x)
    ⊢ Eq (SDiff.sdiff s (Insert.insert x t)) (SDiff.sdiff s t)
  -/
  refine Subset.antisymm (diff_subset_diff (refl _) (subset_insert ..)) fun y hy ↦ ?_
  /-
    α : Type u
    s t : Set α
    x : α
    h : Not (Membership.mem s x)
    y : α
    hy : Membership.mem (SDiff.sdiff s t) y
    ⊢ Membership.mem (SDiff.sdiff s (Insert.insert x t)) y
  -/
  simp only [mem_diff, mem_insert_iff, not_or] at hy ⊢
  /-
    α : Type u
    s t : Set α
    x : α
    h : Not (Membership.mem s x)
    y : α
    hy : And (Membership.mem s y) (Not (Membership.mem t y))
    ⊢ And (Membership.mem s y) (And (Not (Eq y x)) (Not (Membership.mem t y)))
  -/
  exact ⟨hy.1, fun hxy ↦ h <| hxy ▸ hy.1, hy.2⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem insert_diff_of_mem (s) (h : a ∈ t) : insert a s \ t = s \ t := by
  /-
    α : Type u
    a : α
    t s : Set α
    h : Membership.mem t a
    ⊢ Eq (SDiff.sdiff (Insert.insert a s) t) (SDiff.sdiff s t)
  -/
  ext
  /-
    case h
    α : Type u
    a : α
    t s : Set α
    h : Membership.mem t a
    x✝ : α
    ⊢ Iff (Membership.mem (SDiff.sdiff (Insert.insert a s) t) x✝) (Membership.mem  …
  -/
                  /-
                    🎉 no goals
                  -/
  constructor <;> simp +contextual [or_imp, h]
                  /-
                    🎉 no goals
                  -/


theorem insert_diff_of_not_mem (s) (h : a ∉ t) : insert a s \ t = insert a (s \ t) := by
  classical
    ext x
    by_cases h' : x ∈ t
    · simp [h, h', ne_of_mem_of_not_mem h' h]
    · simp [h, h']


theorem insert_diff_self_of_not_mem {a : α} {s : Set α} (h : a ∉ s) : insert a s \ {a} = s := by
  /-
    α : Type u
    a : α
    s : Set α
    h : Not (Membership.mem s a)
    ⊢ Eq (SDiff.sdiff (Insert.insert a s) (Singleton.singleton a)) s
  -/
  ext x
  /-
    case h
    α : Type u
    a : α
    s : Set α
    h : Not (Membership.mem s a)
    x : α
    ⊢ Iff (Membership.mem (SDiff.sdiff (Insert.insert a s) (Singleton.singleton a) …
  -/
  simp [and_iff_left_of_imp (ne_of_mem_of_not_mem · h)]
  /-
    🎉 no goals
  -/


@[simp]
theorem insert_diff_eq_singleton {a : α} {s : Set α} (h : a ∉ s) : insert a s \ s = {a} := by
  /-
    α : Type u
    a : α
    s : Set α
    h : Not (Membership.mem s a)
    ⊢ Eq (SDiff.sdiff (Insert.insert a s) s) (Singleton.singleton a)
  -/
  ext
  rw [Set.mem_diff, Set.mem_insert_iff, Set.mem_singleton_iff, or_and_right, and_not_self_iff,
    or_false, and_iff_left_iff_imp]
  /-
    case h
    α : Type u
    a : α
    s : Set α
    h : Not (Membership.mem s a)
    x✝ : α
    ⊢ Eq x✝ a → Not (Membership.mem s x✝)
  -/
  rintro rfl
  /-
    case h
    α : Type u
    s : Set α
    x✝ : α
    h : Not (Membership.mem s x✝)
    ⊢ Not (Membership.mem s x✝)
  -/
  exact h
  /-
    🎉 no goals
  -/


theorem inter_insert_of_mem (h : a ∈ s) : s ∩ insert a t = insert a (s ∩ t) := by
  /-
    α : Type u
    a : α
    s t : Set α
    h : Membership.mem s a
    ⊢ Eq (Inter.inter s (Insert.insert a t)) (Insert.insert a (Inter.inter s t))
  -/
  rw [insert_inter_distrib, insert_eq_of_mem h]
  /-
    🎉 no goals
  -/


theorem insert_inter_of_mem (h : a ∈ t) : insert a s ∩ t = insert a (s ∩ t) := by
  /-
    α : Type u
    a : α
    s t : Set α
    h : Membership.mem t a
    ⊢ Eq (Inter.inter (Insert.insert a s) t) (Insert.insert a (Inter.inter s t))
  -/
  rw [insert_inter_distrib, insert_eq_of_mem h]
  /-
    🎉 no goals
  -/


theorem inter_insert_of_not_mem (h : a ∉ s) : s ∩ insert a t = s ∩ t :=
  ext fun _ => and_congr_right fun hx => or_iff_right <| ne_of_mem_of_not_mem hx h


theorem insert_inter_of_not_mem (h : a ∉ t) : insert a s ∩ t = s ∩ t :=
  ext fun _ => and_congr_left fun hx => or_iff_right <| ne_of_mem_of_not_mem hx h


@[simp]
theorem union_diff_self {s t : Set α} : s ∪ t \ s = s ∪ t :=
  sup_sdiff_self _ _


@[simp]
theorem diff_union_self {s t : Set α} : s \ t ∪ t = s ∪ t :=
  sdiff_sup_self _ _


@[simp]
theorem diff_inter_self {a b : Set α} : b \ a ∩ a = ∅ :=
  inf_sdiff_self_left


@[simp]
theorem diff_inter_self_eq_diff {s t : Set α} : s \ (t ∩ s) = s \ t :=
  sdiff_inf_self_right _ _


@[simp]
theorem diff_self_inter {s t : Set α} : s \ (s ∩ t) = s \ t :=
  sdiff_inf_self_left _ _


@[simp]
theorem diff_singleton_eq_self {a : α} {s : Set α} (h : a ∉ s) : s \ {a} = s :=
                                     /-
                                       α : Type u
                                       a : α
                                       s : Set α
                                       h : Not (Membership.mem s a)
                                       ⊢ Disjoint (Singleton.singleton a) s
                                     -/
  sdiff_eq_self_iff_disjoint.2 <| by simp [h]
                                     /-
                                       🎉 no goals
                                     -/


@[simp]
theorem diff_singleton_sSubset {s : Set α} {a : α} : s \ {a} ⊂ s ↔ a ∈ s :=
                                                            /-
                                                              α : Type u
                                                              s : Set α
                                                              a : α
                                                              ⊢ Iff (Not (Disjoint s (Singleton.singleton a))) (Membership.mem s a)
                                                            -/
  sdiff_le.lt_iff_ne.trans <| sdiff_eq_left.not.trans <| by simp
                                                            /-
                                                              🎉 no goals
                                                            -/


@[simp]
theorem insert_diff_singleton {a : α} {s : Set α} : insert a (s \ {a}) = insert a s := by
  /-
    α : Type u
    a : α
    s : Set α
    ⊢ Eq (Insert.insert a (SDiff.sdiff s (Singleton.singleton a))) (Insert.insert  …
  -/
  simp [insert_eq, union_diff_self, -union_singleton, -singleton_union]
  /-
    🎉 no goals
  -/


theorem insert_diff_singleton_comm (hab : a ≠ b) (s : Set α) :
    insert a (s \ {b}) = insert a s \ {b} := by
  simp_rw [← union_singleton, union_diff_distrib,
    diff_singleton_eq_self (mem_singleton_iff.not.2 hab.symm)]


theorem diff_self {s : Set α} : s \ s = ∅ :=
  sdiff_self


theorem diff_diff_right_self (s t : Set α) : s \ (s \ t) = s ∩ t :=
  sdiff_sdiff_right_self


theorem diff_diff_cancel_left {s t : Set α} (h : s ⊆ t) : t \ (t \ s) = s :=
  sdiff_sdiff_eq_self h


theorem mem_diff_singleton {x y : α} {s : Set α} : x ∈ s \ {y} ↔ x ∈ s ∧ x ≠ y :=
  Iff.rfl


theorem mem_diff_singleton_empty {t : Set (Set α)} : s ∈ t \ {∅} ↔ s ∈ t ∧ s.Nonempty :=
  mem_diff_singleton.trans <| and_congr_right' nonempty_iff_ne_empty.symm


theorem subset_insert_iff {s t : Set α} {x : α} :
    s ⊆ insert x t ↔ s ⊆ t ∨ (x ∈ s ∧ s \ {x} ⊆ t) := by
  /-
    α : Type u
    s t : Set α
    x : α
    ⊢ Iff (HasSubset.Subset s (Insert.insert x t)) (Or (HasSubset.Subset s t) (And …
  -/
  rw [← diff_singleton_subset_iff]
  /-
    α : Type u
    s t : Set α
    x : α
    ⊢ Iff (HasSubset.Subset (SDiff.sdiff s (Singleton.singleton x)) t) (Or (HasSub …
  -/
  by_cases hx : x ∈ s
    /-
      case pos
      α : Type u
      s t : Set α
      x : α
      hx : Membership.mem s x
      ⊢ Iff (HasSubset.Subset (SDiff.sdiff s (Singleton.singleton x)) t) (Or (HasSub …
    -/
  · rw [and_iff_right hx, or_iff_right_of_imp diff_subset.trans]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u
    s t : Set α
    x : α
    hx : Not (Membership.mem s x)
    ⊢ Iff (HasSubset.Subset (SDiff.sdiff s (Singleton.singleton x)) t) (Or (HasSub …
  -/
  rw [diff_singleton_eq_self hx, or_iff_left_of_imp And.right]
  /-
    🎉 no goals
  -/


theorem union_eq_diff_union_diff_union_inter (s t : Set α) : s ∪ t = s \ t ∪ t \ s ∪ s ∩ t :=
  sup_eq_sdiff_sup_sdiff_sup_inf


theorem pair_eq_singleton (a : α) : ({a, a} : Set α) = {a} :=
  union_self _


theorem pair_comm (a b : α) : ({a, b} : Set α) = {b, a} :=
  union_comm _ _


theorem pair_eq_pair_iff {x y z w : α} :
    ({x, y} : Set α) = {z, w} ↔ x = z ∧ y = w ∨ x = w ∧ y = z := by
  /-
    α : Type u
    x y z w : α
    ⊢ Iff (Eq (Insert.insert x (Singleton.singleton y)) (Insert.insert z (Singleto …
  -/
  simp [subset_antisymm_iff, insert_subset_iff]; aesop
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem pair_diff_left (hne : a ≠ b) : ({a, b} : Set α) \ {a} = {b} := by
  /-
    α : Type u
    a b : α
    hne : Ne a b
    ⊢ Eq (SDiff.sdiff (Insert.insert a (Singleton.singleton b)) (Singleton.singlet …
  -/
  rw [insert_diff_of_mem _ (mem_singleton a), diff_singleton_eq_self (by simpa)]
  /-
    🎉 no goals
  -/


theorem pair_diff_right (hne : a ≠ b) : ({a, b} : Set α) \ {b} = {a} := by
  /-
    α : Type u
    a b : α
    hne : Ne a b
    ⊢ Eq (SDiff.sdiff (Insert.insert a (Singleton.singleton b)) (Singleton.singlet …
  -/
  rw [pair_comm, pair_diff_left hne.symm]
  /-
    🎉 no goals
  -/


theorem pair_subset_iff : {a, b} ⊆ s ↔ a ∈ s ∧ b ∈ s := by
  /-
    α : Type u
    a b : α
    s : Set α
    ⊢ Iff (HasSubset.Subset (Insert.insert a (Singleton.singleton b)) s) (And (Mem …
  -/
  rw [insert_subset_iff, singleton_subset_iff]
  /-
    🎉 no goals
  -/


theorem pair_subset (ha : a ∈ s) (hb : b ∈ s) : {a, b} ⊆ s :=
  pair_subset_iff.2 ⟨ha,hb⟩


theorem subset_pair_iff : s ⊆ {a, b} ↔ ∀ x ∈ s, x = a ∨ x = b := by
  /-
    α : Type u
    a b : α
    s : Set α
    ⊢ Iff (HasSubset.Subset s (Insert.insert a (Singleton.singleton b))) (∀ (x : α …
  -/
  simp [subset_def]
  /-
    🎉 no goals
  -/


theorem subset_pair_iff_eq {x y : α} : s ⊆ {x, y} ↔ s = ∅ ∨ s = {x} ∨ s = {y} ∨ s = {x, y} := by
  /-
    α : Type u
    s : Set α
    x y : α
    ⊢ Iff (HasSubset.Subset s (Insert.insert x (Singleton.singleton y))) (Or (Eq s …
  -/
  refine ⟨?_, by rintro (rfl | rfl | rfl | rfl) <;> simp [pair_subset_iff]⟩
  rw [subset_insert_iff, subset_singleton_iff_eq, subset_singleton_iff_eq,
    ← subset_empty_iff (s := s \ {x}), diff_subset_iff, union_empty, subset_singleton_iff_eq]
  /-
    α : Type u
    s : Set α
    x y : α
    ⊢ Or (Or (Eq s EmptyCollection.emptyCollection) (Eq s (Singleton.singleton y)) …
  -/
  have h : x ∈ s → {y} = s \ {x} → s = {x,y} := fun h₁ h₂ ↦ by simp [h₁, h₂]
  /-
    α : Type u
    s : Set α
    x y : α
    h : Membership.mem s x → Eq (Singleton.singleton y) (SDiff.sdiff s (Singleton. …
    ⊢ Or (Or (Eq s EmptyCollection.emptyCollection) (Eq s (Singleton.singleton y)) …
  -/
  tauto
  /-
    🎉 no goals
  -/


theorem Nonempty.subset_pair_iff_eq (hs : s.Nonempty) :
    s ⊆ {a, b} ↔ s = {a} ∨ s = {b} ∨ s = {a, b} := by
  /-
    α : Type u
    a b : α
    s : Set α
    hs : s.Nonempty
    ⊢ Iff (HasSubset.Subset s (Insert.insert a (Singleton.singleton b))) (Or (Eq s …
  -/
  rw [Set.subset_pair_iff_eq, or_iff_right]; exact hs.ne_empty
                                             /-
                                               🎉 no goals
                                             -/


theorem mem_powerset {x s : Set α} (h : x ⊆ s) : x ∈ 𝒫 s := @h


theorem subset_of_mem_powerset {x s : Set α} (h : x ∈ 𝒫 s) : x ⊆ s := @h


@[simp]
theorem mem_powerset_iff (x s : Set α) : x ∈ 𝒫 s ↔ x ⊆ s :=
  Iff.rfl


theorem powerset_inter (s t : Set α) : 𝒫(s ∩ t) = 𝒫 s ∩ 𝒫 t :=
  ext fun _ => subset_inter_iff


@[simp]
theorem powerset_mono : 𝒫 s ⊆ 𝒫 t ↔ s ⊆ t :=
  ⟨fun h => @h _ (fun _ h => h), fun h _ hu _ ha => h (hu ha)⟩


theorem monotone_powerset : Monotone (powerset : Set α → Set (Set α)) := fun _ _ => powerset_mono.2


@[simp]
theorem powerset_nonempty : (𝒫 s).Nonempty :=
  ⟨∅, fun _ h => empty_subset s h⟩


@[simp]
theorem powerset_empty : 𝒫(∅ : Set α) = {∅} :=
  ext fun _ => subset_empty_iff


@[simp]
theorem powerset_univ : 𝒫(univ : Set α) = univ :=
  eq_univ_of_forall subset_univ


/-- The powerset of a singleton contains only `∅` and the singleton itself. -/
theorem powerset_singleton (x : α) : 𝒫({x} : Set α) = {∅, {x}} := by
  /-
    α : Type u
    x : α
    ⊢ Eq (Singleton.singleton x).powerset (Insert.insert EmptyCollection.emptyColl …
  -/
  ext y
  /-
    case h
    α : Type u
    x : α
    y : Set α
    ⊢ Iff (Membership.mem (Singleton.singleton x).powerset y) (Membership.mem (Ins …
  -/
  rw [mem_powerset_iff, subset_singleton_iff_eq, mem_insert_iff, mem_singleton_iff]
  /-
    🎉 no goals
  -/


theorem mem_dite (p : Prop) [Decidable p] (s : p → Set α) (t : ¬ p → Set α) (x : α) :
    (x ∈ if h : p then s h else t h) ↔ (∀ h : p, x ∈ s h) ∧ ∀ h : ¬p, x ∈ t h := by
  /-
    α : Type u
    p : Prop
    inst✝ : Decidable p
    s : p → Set α
    t : Not p → Set α
    x : α
    ⊢ Iff (Membership.mem (dite p (fun h => s h) fun h => t h) x) (And (∀ (h : p), …
  -/
  split_ifs with hp
    /-
      case pos
      α : Type u
      p : Prop
      inst✝ : Decidable p
      s : p → Set α
      t : Not p → Set α
      x : α
      hp : p
      ⊢ Iff (Membership.mem (s hp) x) (And (∀ (h : p), Membership.mem (s h) x) (∀ (h …
    -/
  · exact ⟨fun hx => ⟨fun _ => hx, fun hnp => (hnp hp).elim⟩, fun hx => hx.1 hp⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u
      p : Prop
      inst✝ : Decidable p
      s : p → Set α
      t : Not p → Set α
      x : α
      hp : Not p
      ⊢ Iff (Membership.mem (t hp) x) (And (∀ (h : p), Membership.mem (s h) x) (∀ (h …
    -/
  · exact ⟨fun hx => ⟨fun h => (hp h).elim, fun _ => hx⟩, fun hx => hx.2 hp⟩
    /-
      🎉 no goals
    -/


theorem mem_dite_univ_right (p : Prop) [Decidable p] (t : p → Set α) (x : α) :
    (x ∈ if h : p then t h else univ) ↔ ∀ h : p, x ∈ t h := by
  /-
    α : Type u
    p : Prop
    inst✝ : Decidable p
    t : p → Set α
    x : α
    ⊢ Iff (Membership.mem (dite p (fun h => t h) fun h => Set.univ) x) (∀ (h : p), …
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> simp_all
                /-
                  🎉 no goals
                -/


@[simp]
theorem mem_ite_univ_right (p : Prop) [Decidable p] (t : Set α) (x : α) :
    x ∈ ite p t Set.univ ↔ p → x ∈ t :=
  mem_dite_univ_right p (fun _ => t) x


theorem mem_dite_univ_left (p : Prop) [Decidable p] (t : ¬p → Set α) (x : α) :
    (x ∈ if h : p then univ else t h) ↔ ∀ h : ¬p, x ∈ t h := by
  /-
    α : Type u
    p : Prop
    inst✝ : Decidable p
    t : Not p → Set α
    x : α
    ⊢ Iff (Membership.mem (dite p (fun h => Set.univ) fun h => t h) x) (∀ (h : Not …
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> simp_all
                /-
                  🎉 no goals
                -/


@[simp]
theorem mem_ite_univ_left (p : Prop) [Decidable p] (t : Set α) (x : α) :
    x ∈ ite p Set.univ t ↔ ¬p → x ∈ t :=
  mem_dite_univ_left p (fun _ => t) x


theorem mem_dite_empty_right (p : Prop) [Decidable p] (t : p → Set α) (x : α) :
    (x ∈ if h : p then t h else ∅) ↔ ∃ h : p, x ∈ t h := by
  /-
    α : Type u
    p : Prop
    inst✝ : Decidable p
    t : p → Set α
    x : α
    ⊢ Iff (Membership.mem (dite p (fun h => t h) fun h => EmptyCollection.emptyCol …
  -/
  simp only [mem_dite, mem_empty_iff_false, imp_false, not_not]
  /-
    α : Type u
    p : Prop
    inst✝ : Decidable p
    t : p → Set α
    x : α
    ⊢ Iff (And (∀ (h : p), Membership.mem (t h) x) p) (Exists fun h => Membership. …
  -/
  exact ⟨fun h => ⟨h.2, h.1 h.2⟩, fun ⟨h₁, h₂⟩ => ⟨fun _ => h₂, h₁⟩⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem mem_ite_empty_right (p : Prop) [Decidable p] (t : Set α) (x : α) :
    x ∈ ite p t ∅ ↔ p ∧ x ∈ t :=
                                                    /-
                                                      α : Type u
                                                      p : Prop
                                                      inst✝ : Decidable p
                                                      t : Set α
                                                      x : α
                                                      ⊢ Iff (Exists fun h => Membership.mem t x) (And p (Membership.mem t x))
                                                    -/
  (mem_dite_empty_right p (fun _ => t) x).trans (by simp)
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem mem_dite_empty_left (p : Prop) [Decidable p] (t : ¬p → Set α) (x : α) :
    (x ∈ if h : p then ∅ else t h) ↔ ∃ h : ¬p, x ∈ t h := by
  /-
    α : Type u
    p : Prop
    inst✝ : Decidable p
    t : Not p → Set α
    x : α
    ⊢ Iff (Membership.mem (dite p (fun h => EmptyCollection.emptyCollection) fun h …
  -/
  simp only [mem_dite, mem_empty_iff_false, imp_false]
  /-
    α : Type u
    p : Prop
    inst✝ : Decidable p
    t : Not p → Set α
    x : α
    ⊢ Iff (And (Not p) (∀ (h : Not p), Membership.mem (t h) x)) (Exists fun h => M …
  -/
  exact ⟨fun h => ⟨h.1, h.2 h.1⟩, fun ⟨h₁, h₂⟩ => ⟨fun h => h₁ h, fun _ => h₂⟩⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem mem_ite_empty_left (p : Prop) [Decidable p] (t : Set α) (x : α) :
    x ∈ ite p ∅ t ↔ ¬p ∧ x ∈ t :=
                                                   /-
                                                     α : Type u
                                                     p : Prop
                                                     inst✝ : Decidable p
                                                     t : Set α
                                                     x : α
                                                     ⊢ Iff (Exists fun h => Membership.mem t x) (And (Not p) (Membership.mem t x))
                                                   -/
  (mem_dite_empty_left p (fun _ => t) x).trans (by simp)
                                                   /-
                                                     🎉 no goals
                                                   -/


/-- `ite` for sets: `Set.ite t s s' ∩ t = s ∩ t`, `Set.ite t s s' ∩ tᶜ = s' ∩ tᶜ`.
Defined as `s ∩ t ∪ s' \ t`. -/
protected def ite (t s s' : Set α) : Set α :=
  s ∩ t ∪ s' \ t


@[simp]
theorem ite_inter_self (t s s' : Set α) : t.ite s s' ∩ t = s ∩ t := by
  /-
    α : Type u
    t s s' : Set α
    ⊢ Eq (Inter.inter (t.ite s s') t) (Inter.inter s t)
  -/
  rw [Set.ite, union_inter_distrib_right, diff_inter_self, inter_assoc, inter_self, union_empty]
  /-
    🎉 no goals
  -/


@[simp]
theorem ite_compl (t s s' : Set α) : tᶜ.ite s s' = t.ite s' s := by
  /-
    α : Type u
    t s s' : Set α
    ⊢ Eq ((HasCompl.compl t).ite s s') (t.ite s' s)
  -/
  rw [Set.ite, Set.ite, diff_compl, union_comm, diff_eq]
  /-
    🎉 no goals
  -/


@[simp]
theorem ite_inter_compl_self (t s s' : Set α) : t.ite s s' ∩ tᶜ = s' ∩ tᶜ := by
  /-
    α : Type u
    t s s' : Set α
    ⊢ Eq (Inter.inter (t.ite s s') (HasCompl.compl t)) (Inter.inter s' (HasCompl.c …
  -/
  rw [← ite_compl, ite_inter_self]
  /-
    🎉 no goals
  -/


@[simp]
theorem ite_diff_self (t s s' : Set α) : t.ite s s' \ t = s' \ t :=
  ite_inter_compl_self t s s'


@[simp]
theorem ite_same (t s : Set α) : t.ite s s = s :=
  inter_union_diff _ _


@[simp]
                                                         /-
                                                           α : Type u
                                                           s t : Set α
                                                           ⊢ Eq (s.ite s t) (Union.union s t)
                                                         -/
theorem ite_left (s t : Set α) : s.ite s t = s ∪ t := by simp [Set.ite]
                                                         /-
                                                           🎉 no goals
                                                         -/


@[simp]
                                                          /-
                                                            α : Type u
                                                            s t : Set α
                                                            ⊢ Eq (s.ite t s) (Inter.inter t s)
                                                          -/
theorem ite_right (s t : Set α) : s.ite t s = t ∩ s := by simp [Set.ite]
                                                          /-
                                                            🎉 no goals
                                                          -/


@[simp]
                                                             /-
                                                               α : Type u
                                                               s s' : Set α
                                                               ⊢ Eq (EmptyCollection.emptyCollection.ite s s') s'
                                                             -/
theorem ite_empty (s s' : Set α) : Set.ite ∅ s s' = s' := by simp [Set.ite]
                                                             /-
                                                               🎉 no goals
                                                             -/


@[simp]
                                                              /-
                                                                α : Type u
                                                                s s' : Set α
                                                                ⊢ Eq (Set.univ.ite s s') s
                                                              -/
theorem ite_univ (s s' : Set α) : Set.ite univ s s' = s := by simp [Set.ite]
                                                              /-
                                                                🎉 no goals
                                                              -/


@[simp]
                                                               /-
                                                                 α : Type u
                                                                 t s : Set α
                                                                 ⊢ Eq (t.ite EmptyCollection.emptyCollection s) (SDiff.sdiff s t)
                                                               -/
theorem ite_empty_left (t s : Set α) : t.ite ∅ s = s \ t := by simp [Set.ite]
                                                               /-
                                                                 🎉 no goals
                                                               -/


@[simp]
                                                                /-
                                                                  α : Type u
                                                                  t s : Set α
                                                                  ⊢ Eq (t.ite s EmptyCollection.emptyCollection) (Inter.inter s t)
                                                                -/
theorem ite_empty_right (t s : Set α) : t.ite s ∅ = s ∩ t := by simp [Set.ite]
                                                                /-
                                                                  🎉 no goals
                                                                -/


theorem ite_mono (t : Set α) {s₁ s₁' s₂ s₂' : Set α} (h : s₁ ⊆ s₂) (h' : s₁' ⊆ s₂') :
    t.ite s₁ s₁' ⊆ t.ite s₂ s₂' :=
  union_subset_union (inter_subset_inter_left _ h) (inter_subset_inter_left _ h')


theorem ite_subset_union (t s s' : Set α) : t.ite s s' ⊆ s ∪ s' :=
  union_subset_union inter_subset_left diff_subset


theorem inter_subset_ite (t s s' : Set α) : s ∩ s' ⊆ t.ite s s' :=
  ite_same t (s ∩ s') ▸ ite_mono _ inter_subset_left inter_subset_right


theorem ite_inter_inter (t s₁ s₂ s₁' s₂' : Set α) :
    t.ite (s₁ ∩ s₂) (s₁' ∩ s₂') = t.ite s₁ s₁' ∩ t.ite s₂ s₂' := by
  /-
    α : Type u
    t s₁ s₂ s₁' s₂' : Set α
    ⊢ Eq (t.ite (Inter.inter s₁ s₂) (Inter.inter s₁' s₂')) (Inter.inter (t.ite s₁  …
  -/
  ext x
  /-
    case h
    α : Type u
    t s₁ s₂ s₁' s₂' : Set α
    x : α
    ⊢ Iff (Membership.mem (t.ite (Inter.inter s₁ s₂) (Inter.inter s₁' s₂')) x) (Me …
  -/
  simp only [Set.ite, Set.mem_inter_iff, Set.mem_diff, Set.mem_union]
  /-
    case h
    α : Type u
    t s₁ s₂ s₁' s₂' : Set α
    x : α
    ⊢ Iff (Or (And (And (Membership.mem s₁ x) (Membership.mem s₂ x)) (Membership.m …
  -/
  tauto
  /-
    🎉 no goals
  -/


theorem ite_inter (t s₁ s₂ s : Set α) : t.ite (s₁ ∩ s) (s₂ ∩ s) = t.ite s₁ s₂ ∩ s := by
  /-
    α : Type u
    t s₁ s₂ s : Set α
    ⊢ Eq (t.ite (Inter.inter s₁ s) (Inter.inter s₂ s)) (Inter.inter (t.ite s₁ s₂) s)
  -/
  rw [ite_inter_inter, ite_same]
  /-
    🎉 no goals
  -/


theorem ite_inter_of_inter_eq (t : Set α) {s₁ s₂ s : Set α} (h : s₁ ∩ s = s₂ ∩ s) :
                                   /-
                                     α : Type u
                                     t s₁ s₂ s : Set α
                                     h : Eq (Inter.inter s₁ s) (Inter.inter s₂ s)
                                     ⊢ Eq (Inter.inter (t.ite s₁ s₂) s) (Inter.inter s₁ s)
                                   -/
    t.ite s₁ s₂ ∩ s = s₁ ∩ s := by rw [← ite_inter, ← h, ite_same]
                                   /-
                                     🎉 no goals
                                   -/


theorem subset_ite {t s s' u : Set α} : u ⊆ t.ite s s' ↔ u ∩ t ⊆ s ∧ u \ t ⊆ s' := by
  /-
    α : Type u
    t s s' u : Set α
    ⊢ Iff (HasSubset.Subset u (t.ite s s')) (And (HasSubset.Subset (Inter.inter u  …
  -/
  simp only [subset_def, ← forall_and]
  /-
    α : Type u
    t s s' u : Set α
    ⊢ Iff (∀ (x : α), Membership.mem u x → Membership.mem (t.ite s s') x) (∀ (x :  …
  -/
  refine forall_congr' fun x => ?_
  /-
    α : Type u
    t s s' u : Set α
    x : α
    ⊢ Iff (Membership.mem u x → Membership.mem (t.ite s s') x) (And (Membership.me …
  -/
                          /-
                            🎉 no goals
                          -/
  by_cases hx : x ∈ t <;> simp [*, Set.ite]
                          /-
                            🎉 no goals
                          -/


theorem ite_eq_of_subset_left (t : Set α) {s₁ s₂ : Set α} (h : s₁ ⊆ s₂) :
    t.ite s₁ s₂ = s₁ ∪ (s₂ \ t) := by
  /-
    α : Type u
    t s₁ s₂ : Set α
    h : HasSubset.Subset s₁ s₂
    ⊢ Eq (t.ite s₁ s₂) (Union.union s₁ (SDiff.sdiff s₂ t))
  -/
  ext x
  /-
    case h
    α : Type u
    t s₁ s₂ : Set α
    h : HasSubset.Subset s₁ s₂
    x : α
    ⊢ Iff (Membership.mem (t.ite s₁ s₂) x) (Membership.mem (Union.union s₁ (SDiff. …
  -/
                          /-
                            🎉 no goals
                          -/
  by_cases hx : x ∈ t <;> simp [*, Set.ite, or_iff_right_of_imp (@h x)]
                          /-
                            🎉 no goals
                          -/


theorem ite_eq_of_subset_right (t : Set α) {s₁ s₂ : Set α} (h : s₂ ⊆ s₁) :
    t.ite s₁ s₂ = (s₁ ∩ t) ∪ s₂ := by
  /-
    α : Type u
    t s₁ s₂ : Set α
    h : HasSubset.Subset s₂ s₁
    ⊢ Eq (t.ite s₁ s₂) (Union.union (Inter.inter s₁ t) s₂)
  -/
  ext x
  /-
    case h
    α : Type u
    t s₁ s₂ : Set α
    h : HasSubset.Subset s₂ s₁
    x : α
    ⊢ Iff (Membership.mem (t.ite s₁ s₂) x) (Membership.mem (Union.union (Inter.int …
  -/
                          /-
                            🎉 no goals
                          -/
  by_cases hx : x ∈ t <;> simp [*, Set.ite, or_iff_left_of_imp (@h x)]
                          /-
                            🎉 no goals
                          -/


theorem monotoneOn_iff_monotone : MonotoneOn f s ↔
    Monotone fun a : s => f a := by
  /-
    α : Type u
    β : Type v
    s : Set α
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : α → β
    ⊢ Iff (MonotoneOn f s) (Monotone fun a => f ↑a)
  -/
  simp [Monotone, MonotoneOn]
  /-
    🎉 no goals
  -/


theorem antitoneOn_iff_antitone : AntitoneOn f s ↔
    Antitone fun a : s => f a := by
  /-
    α : Type u
    β : Type v
    s : Set α
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : α → β
    ⊢ Iff (AntitoneOn f s) (Antitone fun a => f ↑a)
  -/
  simp [Antitone, AntitoneOn]
  /-
    🎉 no goals
  -/


theorem strictMonoOn_iff_strictMono : StrictMonoOn f s ↔
    StrictMono fun a : s => f a := by
  /-
    α : Type u
    β : Type v
    s : Set α
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : α → β
    ⊢ Iff (StrictMonoOn f s) (StrictMono fun a => f ↑a)
  -/
  simp [StrictMono, StrictMonoOn]
  /-
    🎉 no goals
  -/


theorem strictAntiOn_iff_strictAnti : StrictAntiOn f s ↔
    StrictAnti fun a : s => f a := by
  /-
    α : Type u
    β : Type v
    s : Set α
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : α → β
    ⊢ Iff (StrictAntiOn f s) (StrictAnti fun a => f ↑a)
  -/
  simp [StrictAnti, StrictAntiOn]
  /-
    🎉 no goals
  -/


/-- A function between linear orders which is neither monotone nor antitone makes a dent upright or
downright. -/
theorem not_monotoneOn_not_antitoneOn_iff_exists_le_le :
    ¬MonotoneOn f s ∧ ¬AntitoneOn f s ↔
      ∃ᵉ (a ∈ s) (b ∈ s) (c ∈ s), a ≤ b ∧ b ≤ c ∧
        (f a < f b ∧ f c < f b ∨ f b < f a ∧ f b < f c) := by
  simp [monotoneOn_iff_monotone, antitoneOn_iff_antitone, and_assoc, exists_and_left,
    not_monotone_not_antitone_iff_exists_le_le, @and_left_comm (_ ∈ s)]


/-- A function between linear orders which is neither monotone nor antitone makes a dent upright or
downright. -/
theorem not_monotoneOn_not_antitoneOn_iff_exists_lt_lt :
    ¬MonotoneOn f s ∧ ¬AntitoneOn f s ↔
      ∃ᵉ (a ∈ s) (b ∈ s) (c ∈ s), a < b ∧ b < c ∧
        (f a < f b ∧ f c < f b ∨ f b < f a ∧ f b < f c) := by
  simp [monotoneOn_iff_monotone, antitoneOn_iff_antitone, and_assoc, exists_and_left,
    not_monotone_not_antitone_iff_exists_lt_lt, @and_left_comm (_ ∈ s)]


theorem Injective.nonempty_apply_iff {f : Set α → Set β} (hf : Injective f) (h2 : f ∅ = ∅)
    {s : Set α} : (f s).Nonempty ↔ s.Nonempty := by
  /-
    α : Type u_1
    β : Type u_2
    f : Set α → Set β
    hf : Function.Injective f
    h2 : Eq (f EmptyCollection.emptyCollection) EmptyCollection.emptyCollection
    s : Set α
    ⊢ Iff (f s).Nonempty s.Nonempty
  -/
  rw [nonempty_iff_ne_empty, ← h2, nonempty_iff_ne_empty, hf.ne_iff]
  /-
    🎉 no goals
  -/


lemma preimage_fst_singleton_eq_range : (Prod.fst ⁻¹' {a} : Set (α × β)) = range (a, ·) := by
  /-
    α : Type u_1
    β : Type u_2
    a : α
    ⊢ Eq (Set.preimage Prod.fst (Singleton.singleton a)) (Set.range fun x => { fst …
  -/
  aesop
  /-
    🎉 no goals
  -/


lemma preimage_snd_singleton_eq_range : (Prod.snd ⁻¹' {b} : Set (α × β)) = range (·, b) := by
  /-
    α : Type u_1
    β : Type u_2
    b : β
    ⊢ Eq (Set.preimage Prod.snd (Singleton.singleton b)) (Set.range fun x => { fst …
  -/
  aesop
  /-
    🎉 no goals
  -/


/-- `inclusion` is the "identity" function between two subsets `s` and `t`, where `s ⊆ t` -/
abbrev inclusion (h : s ⊆ t) : s → t := fun x : s => (⟨x, h x.2⟩ : t)


theorem inclusion_self (x : s) : inclusion Subset.rfl x = x := by
  /-
    α : Type u_1
    s : Set α
    x : ↑s
    ⊢ Eq (Set.inclusion ⋯ x) x
  -/
  cases x
  /-
    case mk
    α : Type u_1
    s : Set α
    val✝ : α
    property✝ : Membership.mem s val✝
    ⊢ Eq (Set.inclusion ⋯ ⟨val✝, property✝⟩) ⟨val✝, property✝⟩
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem inclusion_eq_id (h : s ⊆ s) : inclusion h = id :=
  funext inclusion_self


@[simp]
theorem inclusion_mk {h : s ⊆ t} (a : α) (ha : a ∈ s) : inclusion h ⟨a, ha⟩ = ⟨a, h ha⟩ :=
  rfl


theorem inclusion_right (h : s ⊆ t) (x : t) (m : (x : α) ∈ s) : inclusion h ⟨x, m⟩ = x := by
  /-
    α : Type u_1
    s t : Set α
    h : HasSubset.Subset s t
    x : ↑t
    m : Membership.mem s ↑x
    ⊢ Eq (Set.inclusion h ⟨↑x, m⟩) x
  -/
  cases x
  /-
    case mk
    α : Type u_1
    s t : Set α
    h : HasSubset.Subset s t
    val✝ : α
    property✝ : Membership.mem t val✝
    m : Membership.mem s ↑⟨val✝, property✝⟩
    ⊢ Eq (Set.inclusion h ⟨↑⟨val✝, property✝⟩, m⟩) ⟨val✝, property✝⟩
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem inclusion_inclusion (hst : s ⊆ t) (htu : t ⊆ u) (x : s) :
    inclusion htu (inclusion hst x) = inclusion (hst.trans htu) x := by
  /-
    α : Type u_1
    s t u : Set α
    hst : HasSubset.Subset s t
    htu : HasSubset.Subset t u
    x : ↑s
    ⊢ Eq (Set.inclusion htu (Set.inclusion hst x)) (Set.inclusion ⋯ x)
  -/
  cases x
  /-
    case mk
    α : Type u_1
    s t u : Set α
    hst : HasSubset.Subset s t
    htu : HasSubset.Subset t u
    val✝ : α
    property✝ : Membership.mem s val✝
    ⊢ Eq (Set.inclusion htu (Set.inclusion hst ⟨val✝, property✝⟩)) (Set.inclusion  …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem inclusion_comp_inclusion {α} {s t u : Set α} (hst : s ⊆ t) (htu : t ⊆ u) :
    inclusion htu ∘ inclusion hst = inclusion (hst.trans htu) :=
  funext (inclusion_inclusion hst htu)


@[simp]
theorem coe_inclusion (h : s ⊆ t) (x : s) : (inclusion h x : α) = (x : α) :=
  rfl


theorem val_comp_inclusion (h : s ⊆ t) : Subtype.val ∘ inclusion h = Subtype.val :=
  rfl


theorem inclusion_injective (h : s ⊆ t) : Injective (inclusion h)
  | ⟨_, _⟩, ⟨_, _⟩ => Subtype.ext_iff_val.2 ∘ Subtype.ext_iff_val.1


theorem inclusion_inj (h : s ⊆ t) {x y : s} : inclusion h x = inclusion h y ↔ x = y :=
  (inclusion_injective h).eq_iff


theorem eq_of_inclusion_surjective {s t : Set α} {h : s ⊆ t}
    (h_surj : Function.Surjective (inclusion h)) : s = t := by
  /-
    α : Type u_1
    s t : Set α
    h : HasSubset.Subset s t
    h_surj : Function.Surjective (Set.inclusion h)
    ⊢ Eq s t
  -/
  refine Set.Subset.antisymm h (fun x hx => ?_)
  /-
    α : Type u_1
    s t : Set α
    h : HasSubset.Subset s t
    h_surj : Function.Surjective (Set.inclusion h)
    x : α
    hx : Membership.mem t x
    ⊢ Membership.mem s x
  -/
  obtain ⟨y, hy⟩ := h_surj ⟨x, hx⟩
  /-
    case intro
    α : Type u_1
    s t : Set α
    h : HasSubset.Subset s t
    h_surj : Function.Surjective (Set.inclusion h)
    x : α
    hx : Membership.mem t x
    y : ↑s
    hy : Eq (Set.inclusion h y) ⟨x, hx⟩
    ⊢ Membership.mem s x
  -/
  exact mem_of_eq_of_mem (congr_arg Subtype.val hy).symm y.prop
  /-
    🎉 no goals
  -/


theorem inclusion_le_inclusion [Preorder α] {s t : Set α} (h : s ⊆ t) {x y : s} :
    inclusion h x ≤ inclusion h y ↔ x ≤ y := Iff.rfl


theorem inclusion_lt_inclusion [Preorder α] {s t : Set α} (h : s ⊆ t) {x y : s} :
    inclusion h x < inclusion h y ↔ x < y := Iff.rfl


theorem eq_univ_of_nonempty {s : Set α} : s.Nonempty → s = univ := fun ⟨x, hx⟩ =>
  eq_univ_of_forall fun y => Subsingleton.elim x y ▸ hx


@[elab_as_elim]
theorem set_cases {p : Set α → Prop} (h0 : p ∅) (h1 : p univ) (s) : p s :=
  (s.eq_empty_or_nonempty.elim fun h => h.symm ▸ h0) fun h => (eq_univ_of_nonempty h).symm ▸ h1


theorem mem_iff_nonempty {α : Type*} [Subsingleton α] {s : Set α} {x : α} : x ∈ s ↔ s.Nonempty :=
  ⟨fun hx => ⟨x, hx⟩, fun ⟨y, hy⟩ => Subsingleton.elim y x ▸ hy⟩


instance decidableSdiff [Decidable (a ∈ s)] [Decidable (a ∈ t)] : Decidable (a ∈ s \ t) :=
  inferInstanceAs (Decidable (a ∈ s ∧ a ∉ t))


instance decidableInter [Decidable (a ∈ s)] [Decidable (a ∈ t)] : Decidable (a ∈ s ∩ t) :=
  inferInstanceAs (Decidable (a ∈ s ∧ a ∈ t))


instance decidableUnion [Decidable (a ∈ s)] [Decidable (a ∈ t)] : Decidable (a ∈ s ∪ t) :=
  inferInstanceAs (Decidable (a ∈ s ∨ a ∈ t))


instance decidableCompl [Decidable (a ∈ s)] : Decidable (a ∈ sᶜ) :=
  inferInstanceAs (Decidable (a ∉ s))


                                                                                  /-
                                                                                    α : Type u
                                                                                    s t : Set α
                                                                                    a b : α
                                                                                    ⊢ Not (Membership.mem EmptyCollection.emptyCollection a)
                                                                                  -/
instance decidableEmptyset : Decidable (a ∈ (∅ : Set α)) := Decidable.isFalse (by simp)
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


                                                                      /-
                                                                        α : Type u
                                                                        s t : Set α
                                                                        a b : α
                                                                        ⊢ Membership.mem Set.univ a
                                                                      -/
instance decidableUniv : Decidable (a ∈ univ) := Decidable.isTrue (by simp)
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


instance decidableInsert [Decidable (a = b)] [Decidable (a ∈ s)] : Decidable (a ∈ insert b s) :=
  inferInstanceAs (Decidable (_ ∨ _))

-- Porting note: Lean 3 unfolded `{a}` before finding instances but Lean 4 needs additional help

instance decidableSingleton [Decidable (a = b)] : Decidable (a ∈ ({b} : Set α)) :=
  inferInstanceAs (Decidable (a = b))


instance decidableSetOf (p : α → Prop) [Decidable (p a)] : Decidable (a ∈ { a | p a }) := by
  /-
    α : Type u
    s t : Set α
    a b : α
    p : α → Prop
    inst✝ : Decidable (p a)
    ⊢ Decidable (Membership.mem (setOf fun a => p a) a)
  -/
  assumption
  /-
    🎉 no goals
  -/


theorem Monotone.inter [Preorder β] {f g : β → Set α} (hf : Monotone f) (hg : Monotone g) :
    Monotone fun x => f x ∩ g x :=
  hf.inf hg


theorem MonotoneOn.inter [Preorder β] {f g : β → Set α} {s : Set β} (hf : MonotoneOn f s)
    (hg : MonotoneOn g s) : MonotoneOn (fun x => f x ∩ g x) s :=
  hf.inf hg


theorem Antitone.inter [Preorder β] {f g : β → Set α} (hf : Antitone f) (hg : Antitone g) :
    Antitone fun x => f x ∩ g x :=
  hf.inf hg


theorem AntitoneOn.inter [Preorder β] {f g : β → Set α} {s : Set β} (hf : AntitoneOn f s)
    (hg : AntitoneOn g s) : AntitoneOn (fun x => f x ∩ g x) s :=
  hf.inf hg


theorem Monotone.union [Preorder β] {f g : β → Set α} (hf : Monotone f) (hg : Monotone g) :
    Monotone fun x => f x ∪ g x :=
  hf.sup hg


theorem MonotoneOn.union [Preorder β] {f g : β → Set α} {s : Set β} (hf : MonotoneOn f s)
    (hg : MonotoneOn g s) : MonotoneOn (fun x => f x ∪ g x) s :=
  hf.sup hg


theorem Antitone.union [Preorder β] {f g : β → Set α} (hf : Antitone f) (hg : Antitone g) :
    Antitone fun x => f x ∪ g x :=
  hf.sup hg


theorem AntitoneOn.union [Preorder β] {f g : β → Set α} {s : Set β} (hf : AntitoneOn f s)
    (hg : AntitoneOn g s) : AntitoneOn (fun x => f x ∪ g x) s :=
  hf.sup hg


theorem monotone_setOf [Preorder α] {p : α → β → Prop} (hp : ∀ b, Monotone fun a => p a b) :
    Monotone fun a => { b | p a b } := fun _ _ h b => hp b h


theorem antitone_setOf [Preorder α] {p : α → β → Prop} (hp : ∀ b, Antitone fun a => p a b) :
    Antitone fun a => { b | p a b } := fun _ _ h b => hp b h


/-- Quantifying over a set is antitone in the set -/
theorem antitone_bforall {P : α → Prop} : Antitone fun s : Set α => ∀ x ∈ s, P x :=
  fun _ _ hst h x hx => h x <| hst hx


theorem union_left (hs : Disjoint s u) (ht : Disjoint t u) : Disjoint (s ∪ t) u :=
  hs.sup_left ht


theorem union_right (ht : Disjoint s t) (hu : Disjoint s u) : Disjoint s (t ∪ u) :=
  ht.sup_right hu


theorem inter_left (u : Set α) (h : Disjoint s t) : Disjoint (s ∩ u) t :=
  h.inf_left _


theorem inter_left' (u : Set α) (h : Disjoint s t) : Disjoint (u ∩ s) t :=
  h.inf_left' _


theorem inter_right (u : Set α) (h : Disjoint s t) : Disjoint s (t ∩ u) :=
  h.inf_right _


theorem inter_right' (u : Set α) (h : Disjoint s t) : Disjoint s (u ∩ t) :=
  h.inf_right' _


theorem subset_left_of_subset_union (h : s ⊆ t ∪ u) (hac : Disjoint s u) : s ⊆ t :=
  hac.left_le_of_le_sup_right h


theorem subset_right_of_subset_union (h : s ⊆ t ∪ u) (hab : Disjoint s t) : s ⊆ u :=
  hab.left_le_of_le_sup_left h


@[simp] theorem Prop.compl_singleton (p : Prop) : ({p}ᶜ : Set Prop) = {¬p} :=
                 /-
                   p q : Prop
                   ⊢ Iff (Membership.mem (HasCompl.compl (Singleton.singleton p)) q) (Membership. …
                 -/
  ext fun q ↦ by simpa [@Iff.comm q] using not_iff
                 /-
                   🎉 no goals
                 -/


