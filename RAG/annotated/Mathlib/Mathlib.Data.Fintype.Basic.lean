/-- `Fintype α` means that `α` is finite, i.e. there are only
  finitely many distinct elements of type `α`. The evidence of this
  is a finset `elems` (a list up to permutation without duplicates),
  together with a proof that everything of type `α` is in the list. -/
class Fintype (α : Type*) where
  /-- The `Finset` containing all elements of a `Fintype` -/
  elems : Finset α
  /-- A proof that `elems` contains every element of the type -/
  complete : ∀ x : α, x ∈ elems


/-- `univ` is the universal finite set of type `Finset α` implied from
  the assumption `Fintype α`. -/
def univ : Finset α :=
  @Fintype.elems α _


@[simp]
theorem mem_univ (x : α) : x ∈ (univ : Finset α) :=
  Fintype.complete x

-- Porting note: removing @[simp], simp can prove it

theorem mem_univ_val : ∀ x, x ∈ (univ : Finset α).1 :=
  mem_univ


                                                         /-
                                                           α : Type u_1
                                                           inst✝ : Fintype α
                                                           s : Finset α
                                                           ⊢ Iff (Eq s Finset.univ) (∀ (x : α), Membership.mem s x)
                                                         -/
theorem eq_univ_iff_forall : s = univ ↔ ∀ x, x ∈ s := by simp [Finset.ext_iff]
                                                         /-
                                                           🎉 no goals
                                                         -/


theorem eq_univ_of_forall : (∀ x, x ∈ s) → s = univ :=
  eq_univ_iff_forall.2


@[simp, norm_cast]
                                                                 /-
                                                                   α : Type u_1
                                                                   inst✝ : Fintype α
                                                                   ⊢ Eq (↑Finset.univ) Set.univ
                                                                 -/
theorem coe_univ : ↑(univ : Finset α) = (Set.univ : Set α) := by ext; simp
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[simp, norm_cast]
                                                              /-
                                                                α : Type u_1
                                                                inst✝ : Fintype α
                                                                s : Finset α
                                                                ⊢ Iff (Eq (↑s) Set.univ) (Eq s Finset.univ)
                                                              -/
theorem coe_eq_univ : (s : Set α) = Set.univ ↔ s = univ := by rw [← coe_univ, coe_inj]
                                                              /-
                                                                🎉 no goals
                                                              -/


theorem Nonempty.eq_univ [Subsingleton α] : s.Nonempty → s = univ := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    s : Finset α
    inst✝ : Subsingleton α
    ⊢ s.Nonempty → Eq s Finset.univ
  -/
  rintro ⟨x, hx⟩
  /-
    case intro
    α : Type u_1
    inst✝¹ : Fintype α
    s : Finset α
    inst✝ : Subsingleton α
    x : α
    hx : Membership.mem s x
    ⊢ Eq s Finset.univ
  -/
  exact eq_univ_of_forall fun y => by rwa [Subsingleton.elim y x]
  /-
    🎉 no goals
  -/


theorem univ_nonempty_iff : (univ : Finset α).Nonempty ↔ Nonempty α := by
  /-
    α : Type u_1
    inst✝ : Fintype α
    ⊢ Iff Finset.univ.Nonempty (Nonempty α)
  -/
  rw [← coe_nonempty, coe_univ, Set.nonempty_iff_univ_nonempty]
  /-
    🎉 no goals
  -/


@[simp, aesop unsafe apply (rule_sets := [finsetNonempty])]
theorem univ_nonempty [Nonempty α] : (univ : Finset α).Nonempty :=
  univ_nonempty_iff.2 ‹_›


theorem univ_eq_empty_iff : (univ : Finset α) = ∅ ↔ IsEmpty α := by
  /-
    α : Type u_1
    inst✝ : Fintype α
    ⊢ Iff (Eq Finset.univ EmptyCollection.emptyCollection) (IsEmpty α)
  -/
  rw [← not_nonempty_iff, ← univ_nonempty_iff, not_nonempty_iff_eq_empty]
  /-
    🎉 no goals
  -/


theorem univ_nontrivial_iff :
    (Finset.univ : Finset α).Nontrivial ↔ Nontrivial α := by
  /-
    α : Type u_1
    inst✝ : Fintype α
    ⊢ Iff Finset.univ.Nontrivial (Nontrivial α)
  -/
  rw [Finset.Nontrivial, Finset.coe_univ, Set.nontrivial_univ_iff]
  /-
    🎉 no goals
  -/


theorem univ_nontrivial [h : Nontrivial α] :
    (Finset.univ : Finset α).Nontrivial :=
  univ_nontrivial_iff.mpr h


@[simp]
theorem univ_eq_empty [IsEmpty α] : (univ : Finset α) = ∅ :=
  univ_eq_empty_iff.2 ‹_›


@[simp]
theorem univ_unique [Unique α] : (univ : Finset α) = {default} :=
  Finset.ext fun x => iff_of_true (mem_univ _) <| mem_singleton.2 <| Subsingleton.elim x default


@[simp]
theorem subset_univ (s : Finset α) : s ⊆ univ := fun a _ => mem_univ a


instance boundedOrder : BoundedOrder (Finset α) :=
  { inferInstanceAs (OrderBot (Finset α)) with
    top := univ
    le_top := subset_univ }


@[simp]
theorem top_eq_univ : (⊤ : Finset α) = univ :=
  rfl


theorem ssubset_univ_iff {s : Finset α} : s ⊂ univ ↔ s ≠ univ :=
  @lt_top_iff_ne_top _ _ _ s


@[simp]
theorem univ_subset_iff {s : Finset α} : univ ⊆ s ↔ s = univ :=
  @top_le_iff _ _ _ s


theorem codisjoint_left : Codisjoint s t ↔ ∀ ⦃a⦄, a ∉ s → a ∈ t := by
  /-
    α : Type u_1
    inst✝ : Fintype α
    s t : Finset α
    ⊢ Iff (Codisjoint s t) (∀ ⦃a : α⦄, Not (Membership.mem s a) → Membership.mem t …
  -/
  classical simp [codisjoint_iff, eq_univ_iff_forall, or_iff_not_imp_left]
  /-
    🎉 no goals
  -/


theorem codisjoint_right : Codisjoint s t ↔ ∀ ⦃a⦄, a ∉ t → a ∈ s :=
  codisjoint_comm.trans codisjoint_left


instance booleanAlgebra [DecidableEq α] : BooleanAlgebra (Finset α) :=
  GeneralizedBooleanAlgebra.toBooleanAlgebra


/-- Elaborate set builder notation for `Finset`.

* `{x | p x}` is elaborated as `Finset.filter (fun x ↦ p x) Finset.univ` if the expected type is
  `Finset ?α`.
* `{x : α | p x}` is elaborated as `Finset.filter (fun x : α ↦ p x) Finset.univ` if the expected
  type is `Finset ?α`.
* `{x ∉ s | p x}` is elaborated as `Finset.filter (fun x ↦ p x) sᶜ` if either the expected type is
  `Finset ?α` or the expected type is not `Set ?α` and `s` has expected type `Finset ?α`.
* `{x ≠ a | p x}` is elaborated as `Finset.filter (fun x ↦ p x) {a}ᶜ` if the expected type is
  `Finset ?α`.

See also
* `Data.Set.Defs` for the `Set` builder notation elaborator that this elaborator partly overrides.
* `Data.Finset.Basic` for the `Finset` builder notation elaborator partly overriding this one for
  syntax of the form `{x ∈ s | p x}`.
* `Data.Fintype.Basic` for the `Finset` builder notation elaborator handling syntax of the form
  `{x | p x}`, `{x : α | p x}`, `{x ∉ s | p x}`, `{x ≠ a | p x}`.
* `Order.LocallyFinite.Basic` for the `Finset` builder notation elaborator handling syntax of the
  form `{x ≤ a | p x}`, `{x ≥ a | p x}`, `{x < a | p x}`, `{x > a | p x}`.

TODO: Write a delaborator
-/
@[term_elab setBuilder]
def elabFinsetBuilderSetOf : TermElab
  | `({ $x:ident | $p }), expectedType? => do
    -- If the expected type is not known to be `Finset ?α`, give up.
    unless ← knownToBeFinsetNotSet expectedType? do throwUnsupportedSyntax
    elabTerm (← `(Finset.filter (fun $x:ident ↦ $p) Finset.univ)) expectedType?
  | `({ $x:ident : $t | $p }), expectedType? => do
    -- If the expected type is not known to be `Finset ?α`, give up.
    unless ← knownToBeFinsetNotSet expectedType? do throwUnsupportedSyntax
    elabTerm (← `(Finset.filter (fun $x:ident : $t ↦ $p) Finset.univ)) expectedType?
  | `({ $x:ident ∉ $s:term | $p }), expectedType? => do
    -- If the expected type is known to be `Set ?α`, give up. If it is not known to be `Set ?α` or
    -- `Finset ?α`, check the expected type of `s`.
    unless ← knownToBeFinsetNotSet expectedType? do
      let ty ← try whnfR (← inferType (← elabTerm s none)) catch _ => throwUnsupportedSyntax
      -- If the expected type of `s` is not known to be `Finset ?α`, give up.
      match_expr ty with
      | Finset _ => pure ()
      | _ => throwUnsupportedSyntax
    -- Finally, we can elaborate the syntax as a finset.
    -- TODO: Seems a bit wasteful to have computed the expected type but still use `expectedType?`.
    elabTerm (← `(Finset.filter (fun $x:ident ↦ $p) $sᶜ)) expectedType?
  | `({ $x:ident ≠ $a | $p }), expectedType? => do
    -- If the expected type is not known to be `Finset ?α`, give up.
    unless ← knownToBeFinsetNotSet expectedType? do throwUnsupportedSyntax
    elabTerm (← `(Finset.filter (fun $x:ident ↦ $p) (singleton $a)ᶜ)) expectedType?
  | _, _ => throwUnsupportedSyntax


theorem sdiff_eq_inter_compl (s t : Finset α) : s \ t = s ∩ tᶜ :=
  sdiff_eq


theorem compl_eq_univ_sdiff (s : Finset α) : sᶜ = univ \ s :=
  rfl


@[simp]
                                         /-
                                           α : Type u_1
                                           inst✝¹ : Fintype α
                                           s : Finset α
                                           inst✝ : DecidableEq α
                                           a : α
                                           ⊢ Iff (Membership.mem (HasCompl.compl s) a) (Not (Membership.mem s a))
                                         -/
theorem mem_compl : a ∈ sᶜ ↔ a ∉ s := by simp [compl_eq_univ_sdiff]
                                         /-
                                           🎉 no goals
                                         -/


                                             /-
                                               α : Type u_1
                                               inst✝¹ : Fintype α
                                               s : Finset α
                                               inst✝ : DecidableEq α
                                               a : α
                                               ⊢ Iff (Not (Membership.mem (HasCompl.compl s) a)) (Membership.mem s a)
                                             -/
theorem not_mem_compl : a ∉ sᶜ ↔ a ∈ s := by rw [mem_compl, not_not]
                                             /-
                                               🎉 no goals
                                             -/


@[simp, norm_cast]
theorem coe_compl (s : Finset α) : ↑sᶜ = (↑s : Set α)ᶜ :=
  Set.ext fun _ => mem_compl


@[simp] lemma compl_subset_compl : sᶜ ⊆ tᶜ ↔ t ⊆ s := @compl_le_compl_iff_le (Finset α) _ _ _

@[simp] lemma compl_ssubset_compl : sᶜ ⊂ tᶜ ↔ t ⊂ s := @compl_lt_compl_iff_lt (Finset α) _ _ _


lemma subset_compl_comm : s ⊆ tᶜ ↔ t ⊆ sᶜ := le_compl_iff_le_compl (α := Finset α)


@[simp] lemma subset_compl_singleton : s ⊆ {a}ᶜ ↔ a ∉ s := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    s : Finset α
    inst✝ : DecidableEq α
    a : α
    ⊢ Iff (HasSubset.Subset s (HasCompl.compl (Singleton.singleton a))) (Not (Memb …
  -/
  rw [subset_compl_comm, singleton_subset_iff, mem_compl]
  /-
    🎉 no goals
  -/


@[simp]
theorem compl_empty : (∅ : Finset α)ᶜ = univ :=
  compl_bot


@[simp]
theorem compl_univ : (univ : Finset α)ᶜ = ∅ :=
  compl_top


@[simp]
theorem compl_eq_empty_iff (s : Finset α) : sᶜ = ∅ ↔ s = univ :=
  compl_eq_bot


@[simp]
theorem compl_eq_univ_iff (s : Finset α) : sᶜ = univ ↔ s = ∅ :=
  compl_eq_top


@[simp]
theorem union_compl (s : Finset α) : s ∪ sᶜ = univ :=
  sup_compl_eq_top


@[simp]
theorem inter_compl (s : Finset α) : s ∩ sᶜ = ∅ :=
  inf_compl_eq_bot


@[simp]
theorem compl_union (s t : Finset α) : (s ∪ t)ᶜ = sᶜ ∩ tᶜ :=
  compl_sup


@[simp]
theorem compl_inter (s t : Finset α) : (s ∩ t)ᶜ = sᶜ ∪ tᶜ :=
  compl_inf


@[simp]
theorem compl_erase : (s.erase a)ᶜ = insert a sᶜ := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    s : Finset α
    inst✝ : DecidableEq α
    a : α
    ⊢ Eq (HasCompl.compl (s.erase a)) (Insert.insert a (HasCompl.compl s))
  -/
  ext
  /-
    case h
    α : Type u_1
    inst✝¹ : Fintype α
    s : Finset α
    inst✝ : DecidableEq α
    a a✝ : α
    ⊢ Iff (Membership.mem (HasCompl.compl (s.erase a)) a✝) (Membership.mem (Insert …
  -/
  simp only [or_iff_not_imp_left, mem_insert, not_and, mem_compl, mem_erase]
  /-
    🎉 no goals
  -/


@[simp]
theorem compl_insert : (insert a s)ᶜ = sᶜ.erase a := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    s : Finset α
    inst✝ : DecidableEq α
    a : α
    ⊢ Eq (HasCompl.compl (Insert.insert a s)) ((HasCompl.compl s).erase a)
  -/
  ext
  /-
    case h
    α : Type u_1
    inst✝¹ : Fintype α
    s : Finset α
    inst✝ : DecidableEq α
    a a✝ : α
    ⊢ Iff (Membership.mem (HasCompl.compl (Insert.insert a s)) a✝) (Membership.mem …
  -/
  simp only [not_or, mem_insert, mem_compl, mem_erase]
  /-
    🎉 no goals
  -/


theorem insert_compl_insert (ha : a ∉ s) : insert a (insert a s)ᶜ = sᶜ := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    s : Finset α
    inst✝ : DecidableEq α
    a : α
    ha : Not (Membership.mem s a)
    ⊢ Eq (Insert.insert a (HasCompl.compl (Insert.insert a s))) (HasCompl.compl s)
  -/
  simp_rw [compl_insert, insert_erase (mem_compl.2 ha)]
  /-
    🎉 no goals
  -/


@[simp]
theorem insert_compl_self (x : α) : insert x ({x}ᶜ : Finset α) = univ := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    x : α
    ⊢ Eq (Insert.insert x (HasCompl.compl (Singleton.singleton x))) Finset.univ
  -/
  rw [← compl_erase, erase_singleton, compl_empty]
  /-
    🎉 no goals
  -/


@[simp]
theorem compl_filter (p : α → Prop) [DecidablePred p] [∀ x, Decidable ¬p x] :
    (univ.filter p)ᶜ = univ.filter fun x => ¬p x :=
            /-
              α : Type u_1
              inst✝³ : Fintype α
              inst✝² : DecidableEq α
              p : α → Prop
              inst✝¹ : DecidablePred p
              inst✝ : (x : α) → Decidable (Not (p x))
              ⊢ ∀ (a : α), Iff (Membership.mem (HasCompl.compl (Finset.filter p Finset.univ) …
            -/
  ext <| by simp
            /-
              🎉 no goals
            -/


theorem compl_ne_univ_iff_nonempty (s : Finset α) : sᶜ ≠ univ ↔ s.Nonempty := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    s : Finset α
    ⊢ Iff (Ne (HasCompl.compl s) Finset.univ) s.Nonempty
  -/
  simp [eq_univ_iff_forall, Finset.Nonempty]
  /-
    🎉 no goals
  -/


theorem compl_singleton (a : α) : ({a} : Finset α)ᶜ = univ.erase a := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    a : α
    ⊢ Eq (HasCompl.compl (Singleton.singleton a)) (Finset.univ.erase a)
  -/
  rw [compl_eq_univ_sdiff, sdiff_singleton_eq_erase]
  /-
    🎉 no goals
  -/


theorem insert_inj_on' (s : Finset α) : Set.InjOn (fun a => insert a s) (sᶜ : Finset α) := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    s : Finset α
    ⊢ Set.InjOn (fun a => Insert.insert a s) ↑(HasCompl.compl s)
  -/
  rw [coe_compl]
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    s : Finset α
    ⊢ Set.InjOn (fun a => Insert.insert a s) (HasCompl.compl ↑s)
  -/
  exact s.insert_inj_on
  /-
    🎉 no goals
  -/


theorem image_univ_of_surjective [Fintype β] {f : β → α} (hf : Surjective f) :
    univ.image f = univ :=
  eq_univ_of_forall <| hf.forall.2 fun _ => mem_image_of_mem _ <| mem_univ _


@[simp]
theorem image_univ_equiv [Fintype β] (f : β ≃ α) : univ.image f = univ :=
  Finset.image_univ_of_surjective f.surjective


                                                             /-
                                                               α : Type u_1
                                                               inst✝¹ : Fintype α
                                                               inst✝ : DecidableEq α
                                                               s : Finset α
                                                               ⊢ Eq (Inter.inter Finset.univ s) s
                                                             -/
@[simp] lemma univ_inter (s : Finset α) : univ ∩ s = s := by ext a; simp
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


                                                             /-
                                                               α : Type u_1
                                                               inst✝¹ : Fintype α
                                                               inst✝ : DecidableEq α
                                                               s : Finset α
                                                               ⊢ Eq (Inter.inter s Finset.univ) s
                                                             -/
@[simp] lemma inter_univ (s : Finset α) : s ∩ univ = s := by rw [inter_comm, univ_inter]
                                                             /-
                                                               🎉 no goals
                                                             -/


@[simp] lemma inter_eq_univ : s ∩ t = univ ↔ s = univ ∧ t = univ := inf_eq_top_iff


lemma singleton_eq_univ [Subsingleton α] (a : α) : ({a} : Finset α) = univ := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : Subsingleton α
    a : α
    ⊢ Eq (Singleton.singleton a) Finset.univ
  -/
  ext b; simp [Subsingleton.elim a b]
         /-
           🎉 no goals
         -/



theorem map_univ_of_surjective [Fintype β] {f : β ↪ α} (hf : Surjective f) : univ.map f = univ :=
  eq_univ_of_forall <| hf.forall.2 fun _ => mem_map_of_mem _ <| mem_univ _


@[simp]
theorem map_univ_equiv [Fintype β] (f : β ≃ α) : univ.map f.toEmbedding = univ :=
  map_univ_of_surjective f.surjective


theorem univ_map_equiv_to_embedding {α β : Type*} [Fintype α] [Fintype β] (e : α ≃ β) :
    univ.map e.toEmbedding = univ :=
                                                                        /-
                                                                          α : Type u_4
                                                                          β : Type u_5
                                                                          inst✝¹ : Fintype α
                                                                          inst✝ : Fintype β
                                                                          e : Equiv α β
                                                                          b : β
                                                                          ⊢ Eq (e.toEmbedding (e.symm b)) b
                                                                        -/
  eq_univ_iff_forall.mpr fun b => mem_map.mpr ⟨e.symm b, mem_univ _, by simp⟩
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


@[simp]
theorem univ_filter_exists (f : α → β) [Fintype β] [DecidablePred fun y => ∃ x, f x = y]
    [DecidableEq β] : (Finset.univ.filter fun y => ∃ x, f x = y) = Finset.univ.image f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : Fintype α
    f : α → β
    inst✝² : Fintype β
    inst✝¹ : DecidablePred fun y => Exists fun x => Eq (f x) y
    inst✝ : DecidableEq β
    ⊢ Eq (Finset.filter (fun y => Exists fun x => Eq (f x) y) Finset.univ) (Finset …
  -/
  ext
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝³ : Fintype α
    f : α → β
    inst✝² : Fintype β
    inst✝¹ : DecidablePred fun y => Exists fun x => Eq (f x) y
    inst✝ : DecidableEq β
    a✝ : β
    ⊢ Iff (Membership.mem (Finset.filter (fun y => Exists fun x => Eq (f x) y) Fin …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Note this is a special case of `(Finset.image_preimage f univ _).symm`. -/
theorem univ_filter_mem_range (f : α → β) [Fintype β] [DecidablePred fun y => y ∈ Set.range f]
    [DecidableEq β] : (Finset.univ.filter fun y => y ∈ Set.range f) = Finset.univ.image f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : Fintype α
    f : α → β
    inst✝² : Fintype β
    inst✝¹ : DecidablePred fun y => Membership.mem (Set.range f) y
    inst✝ : DecidableEq β
    ⊢ Eq (Finset.filter (fun y => Membership.mem (Set.range f) y) Finset.univ) (Fi …
  -/
  letI : DecidablePred (fun y => ∃ x, f x = y) := by simpa using ‹_›
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : Fintype α
    f : α → β
    inst✝² : Fintype β
    inst✝¹ : DecidablePred fun y => Membership.mem (Set.range f) y
    inst✝ : DecidableEq β
    this : DecidablePred fun y => Exists fun x => Eq (f x) y := inst✝¹
    ⊢ Eq (Finset.filter (fun y => Membership.mem (Set.range f) y) Finset.univ) (Fi …
  -/
  exact univ_filter_exists f
  /-
    🎉 no goals
  -/


theorem coe_filter_univ (p : α → Prop) [DecidablePred p] :
                                                /-
                                                  α : Type u_1
                                                  inst✝¹ : Fintype α
                                                  p : α → Prop
                                                  inst✝ : DecidablePred p
                                                  ⊢ Eq (↑(Finset.filter p Finset.univ)) (setOf fun x => p x)
                                                -/
    (univ.filter p : Set α) = { x | p x } := by simp
                                                /-
                                                  🎉 no goals
                                                -/


@[simp] lemma subtype_eq_univ {p : α → Prop} [DecidablePred p] [Fintype {a // p a}] :
                                                  /-
                                                    α : Type u_1
                                                    s : Finset α
                                                    p : α → Prop
                                                    inst✝¹ : DecidablePred p
                                                    inst✝ : Fintype (Subtype fun a => p a)
                                                    ⊢ Iff (Eq (Finset.subtype p s) Finset.univ) (∀ ⦃a : α⦄, p a → Membership.mem s …
                                                  -/
    s.subtype p = univ ↔ ∀ ⦃a⦄, p a → a ∈ s := by simp [Finset.ext_iff]
                                                  /-
                                                    🎉 no goals
                                                  -/


@[simp] lemma subtype_univ [Fintype α] (p : α → Prop) [DecidablePred p] [Fintype {a // p a}] :
                                /-
                                  α : Type u_1
                                  inst✝² : Fintype α
                                  p : α → Prop
                                  inst✝¹ : DecidablePred p
                                  inst✝ : Fintype (Subtype fun a => p a)
                                  ⊢ Eq (Finset.subtype p Finset.univ) Finset.univ
                                -/
    univ.subtype p = univ := by simp
                                /-
                                  🎉 no goals
                                -/


lemma univ_map_subtype [Fintype α] (p : α → Prop) [DecidablePred p] [Fintype {a // p a}] :
    univ.map (Function.Embedding.subtype p) = univ.filter p := by
  /-
    α : Type u_1
    inst✝² : Fintype α
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : Fintype (Subtype fun a => p a)
    ⊢ Eq (Finset.map (Function.Embedding.subtype p) Finset.univ) (Finset.filter p  …
  -/
  rw [← subtype_map, subtype_univ]
  /-
    🎉 no goals
  -/


lemma univ_val_map_subtype_val [Fintype α] (p : α → Prop) [DecidablePred p] [Fintype {a // p a}] :
    univ.val.map ((↑) : { a // p a } → α) = (univ.filter p).val := by
  /-
    α : Type u_1
    inst✝² : Fintype α
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : Fintype (Subtype fun a => p a)
    ⊢ Eq (Multiset.map Subtype.val Finset.univ.val) (Finset.filter p Finset.univ). …
  -/
  apply (map_val (Function.Embedding.subtype p) univ).symm.trans
  /-
    α : Type u_1
    inst✝² : Fintype α
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : Fintype (Subtype fun a => p a)
    ⊢ Eq (Finset.map (Function.Embedding.subtype p) Finset.univ).val (Finset.filte …
  -/
  apply congr_arg
  /-
    case h
    α : Type u_1
    inst✝² : Fintype α
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : Fintype (Subtype fun a => p a)
    ⊢ Eq (Finset.map (Function.Embedding.subtype p) Finset.univ) (Finset.filter p  …
  -/
  apply univ_map_subtype
  /-
    🎉 no goals
  -/


lemma univ_val_map_subtype_restrict [Fintype α] (f : α → β)
    (p : α → Prop) [DecidablePred p] [Fintype {a // p a}] :
    univ.val.map (Subtype.restrict p f) = (univ.filter p).val.map f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : Fintype α
    f : α → β
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : Fintype (Subtype fun a => p a)
    ⊢ Eq (Multiset.map (Subtype.restrict p f) Finset.univ.val) (Multiset.map f (Fi …
  -/
  rw [← univ_val_map_subtype_val, Multiset.map_map, Subtype.restrict_def]
  /-
    🎉 no goals
  -/


instance decidablePiFintype {α} {β : α → Type*} [∀ a, DecidableEq (β a)] [Fintype α] :
    DecidableEq (∀ a, β a) := fun f g =>
  decidable_of_iff (∀ a ∈ @Fintype.elems α _, f a = g a)
        /-
          α✝ : Type u_1
          β✝ : Type u_2
          γ : Type u_3
          α : Type ?u.36819
          β : α → Type u_4
          inst✝¹ : (a : α) → DecidableEq (β a)
          inst✝ : Fintype α
          f g : (a : α) → β a
          ⊢ Iff (∀ (a : α), Membership.mem Fintype.elems a → Eq (f a) (g a)) (Eq f g)
        -/
    (by simp [funext_iff, Fintype.complete])
        /-
          🎉 no goals
        -/


instance decidableForallFintype {p : α → Prop} [DecidablePred p] [Fintype α] :
    Decidable (∀ a, p a) :=
                                              /-
                                                α : Type u_1
                                                β : Type u_2
                                                γ : Type u_3
                                                p : α → Prop
                                                inst✝¹ : DecidablePred p
                                                inst✝ : Fintype α
                                                ⊢ Iff (∀ (a : α), Membership.mem Finset.univ a → p a) (∀ (a : α), p a)
                                              -/
  decidable_of_iff (∀ a ∈ @univ α _, p a) (by simp)
                                              /-
                                                🎉 no goals
                                              -/


instance decidableExistsFintype {p : α → Prop} [DecidablePred p] [Fintype α] :
    Decidable (∃ a, p a) :=
                                              /-
                                                α : Type u_1
                                                β : Type u_2
                                                γ : Type u_3
                                                p : α → Prop
                                                inst✝¹ : DecidablePred p
                                                inst✝ : Fintype α
                                                ⊢ Iff (Exists fun a => And (Membership.mem Finset.univ a) (p a)) (Exists fun a …
                                              -/
  decidable_of_iff (∃ a ∈ @univ α _, p a) (by simp)
                                              /-
                                                🎉 no goals
                                              -/


instance decidableMemRangeFintype [Fintype α] [DecidableEq β] (f : α → β) :
    DecidablePred (· ∈ Set.range f) := fun _ => Fintype.decidableExistsFintype


instance decidableSubsingleton [Fintype α] [DecidableEq α] {s : Set α} [DecidablePred (· ∈ s)] :
    Decidable s.Subsingleton := decidable_of_iff (∀ a ∈ s, ∀ b ∈ s, a = b) Iff.rfl


instance decidableEqEquivFintype [DecidableEq β] [Fintype α] : DecidableEq (α ≃ β) := fun a b =>
  decidable_of_iff (a.1 = b.1) Equiv.coe_fn_injective.eq_iff


instance decidableEqEmbeddingFintype [DecidableEq β] [Fintype α] : DecidableEq (α ↪ β) := fun a b =>
  decidable_of_iff ((a : α → β) = b) Function.Embedding.coe_injective.eq_iff


@[to_additive]
instance decidableEqMulEquivFintype {α β : Type*} [DecidableEq β] [Fintype α] [Mul α] [Mul β] :
    DecidableEq (α ≃* β) :=
  fun a b => decidable_of_iff ((a : α → β) = b) (Injective.eq_iff DFunLike.coe_injective)


instance decidableInjectiveFintype [DecidableEq α] [DecidableEq β] [Fintype α] :
                                                              /-
                                                                α : Type u_1
                                                                β : Type u_2
                                                                γ : Type u_3
                                                                inst✝² : DecidableEq α
                                                                inst✝¹ : DecidableEq β
                                                                inst✝ : Fintype α
                                                                x : α → β
                                                                ⊢ Decidable (Function.Injective x)
                                                              -/
    DecidablePred (Injective : (α → β) → Prop) := fun x => by unfold Injective; infer_instance
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


instance decidableSurjectiveFintype [DecidableEq β] [Fintype α] [Fintype β] :
                                                               /-
                                                                 α : Type u_1
                                                                 β : Type u_2
                                                                 γ : Type u_3
                                                                 inst✝² : DecidableEq β
                                                                 inst✝¹ : Fintype α
                                                                 inst✝ : Fintype β
                                                                 x : α → β
                                                                 ⊢ Decidable (Function.Surjective x)
                                                               -/
    DecidablePred (Surjective : (α → β) → Prop) := fun x => by unfold Surjective; infer_instance
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


instance decidableBijectiveFintype [DecidableEq α] [DecidableEq β] [Fintype α] [Fintype β] :
                                                              /-
                                                                α : Type u_1
                                                                β : Type u_2
                                                                γ : Type u_3
                                                                inst✝³ : DecidableEq α
                                                                inst✝² : DecidableEq β
                                                                inst✝¹ : Fintype α
                                                                inst✝ : Fintype β
                                                                x : α → β
                                                                ⊢ Decidable (Function.Bijective x)
                                                              -/
    DecidablePred (Bijective : (α → β) → Prop) := fun x => by unfold Bijective; infer_instance
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


instance decidableRightInverseFintype [DecidableEq α] [Fintype α] (f : α → β) (g : β → α) :
    Decidable (Function.RightInverse f g) :=
                                       /-
                                         α : Type u_1
                                         β : Type u_2
                                         γ : Type u_3
                                         inst✝¹ : DecidableEq α
                                         inst✝ : Fintype α
                                         f : α → β
                                         g : β → α
                                         ⊢ Decidable (∀ (x : α), Eq (g (f x)) x)
                                       -/
  show Decidable (∀ x, g (f x) = x) by infer_instance
                                       /-
                                         🎉 no goals
                                       -/


instance decidableLeftInverseFintype [DecidableEq β] [Fintype β] (f : α → β) (g : β → α) :
    Decidable (Function.LeftInverse f g) :=
                                       /-
                                         α : Type u_1
                                         β : Type u_2
                                         γ : Type u_3
                                         inst✝¹ : DecidableEq β
                                         inst✝ : Fintype β
                                         f : α → β
                                         g : β → α
                                         ⊢ Decidable (∀ (x : β), Eq (f (g x)) x)
                                       -/
  show Decidable (∀ x, f (g x) = x) by infer_instance
                                       /-
                                         🎉 no goals
                                       -/


/-- Construct a proof of `Fintype α` from a universal multiset -/
def ofMultiset [DecidableEq α] (s : Multiset α) (H : ∀ x : α, x ∈ s) : Fintype α :=
                  /-
                    α : Type u_1
                    β : Type u_2
                    γ : Type u_3
                    inst✝ : DecidableEq α
                    s : Multiset α
                    H : ∀ (x : α), Membership.mem s x
                    ⊢ ∀ (x : α), Membership.mem s.toFinset x
                  -/
  ⟨s.toFinset, by simpa using H⟩
                  /-
                    🎉 no goals
                  -/


/-- Construct a proof of `Fintype α` from a universal list -/
def ofList [DecidableEq α] (l : List α) (H : ∀ x : α, x ∈ l) : Fintype α :=
                  /-
                    α : Type u_1
                    β : Type u_2
                    γ : Type u_3
                    inst✝ : DecidableEq α
                    l : List α
                    H : ∀ (x : α), Membership.mem l x
                    ⊢ ∀ (x : α), Membership.mem l.toFinset x
                  -/
  ⟨l.toFinset, by simpa using H⟩
                  /-
                    🎉 no goals
                  -/


instance subsingleton (α : Type*) : Subsingleton (Fintype α) :=
                               /-
                                 α✝ : Type u_1
                                 β : Type u_2
                                 γ : Type u_3
                                 α : Type u_4
                                 x✝¹ x✝ : Fintype α
                                 s₁ : Finset α
                                 h₁ : ∀ (x : α), Membership.mem s₁ x
                                 s₂ : Finset α
                                 h₂ : ∀ (x : α), Membership.mem s₂ x
                                 ⊢ Eq { elems := s₁, complete := h₁ } { elems := s₂, complete := h₂ }
                               -/
  ⟨fun ⟨s₁, h₁⟩ ⟨s₂, h₂⟩ => by congr; simp [Finset.ext_iff, h₁, h₂]⟩
                                      /-
                                        🎉 no goals
                                      -/


instance (α : Type*) : Lean.Meta.FastSubsingleton (Fintype α) := {}


/-- Given a predicate that can be represented by a finset, the subtype
associated to the predicate is a fintype. -/
protected def subtype {p : α → Prop} (s : Finset α) (H : ∀ x : α, x ∈ s ↔ p x) :
    Fintype { x // p x } :=
  ⟨⟨s.1.pmap Subtype.mk fun x => (H x).1, s.nodup.pmap fun _ _ _ _ => congr_arg Subtype.val⟩,
    fun ⟨x, px⟩ => Multiset.mem_pmap.2 ⟨x, (H x).2 px, rfl⟩⟩


/-- Construct a fintype from a finset with the same elements. -/
def ofFinset {p : Set α} (s : Finset α) (H : ∀ x, x ∈ s ↔ x ∈ p) : Fintype p :=
  Fintype.subtype s H


/-- If `f : α → β` is a bijection and `α` is a fintype, then `β` is also a fintype. -/
def ofBijective [Fintype α] (f : α → β) (H : Function.Bijective f) : Fintype β :=
  ⟨univ.map ⟨f, H.1⟩, fun b =>
    let ⟨_, e⟩ := H.2 b
    e ▸ mem_map_of_mem _ (mem_univ _)⟩


/-- If `f : α → β` is a surjection and `α` is a fintype, then `β` is also a fintype. -/
def ofSurjective [DecidableEq β] [Fintype α] (f : α → β) (H : Function.Surjective f) : Fintype β :=
  ⟨univ.image f, fun b =>
    let ⟨_, e⟩ := H b
    e ▸ mem_image_of_mem _ (mem_univ _)⟩


@[simp]
                                                                     /-
                                                                       α : Type u_1
                                                                       inst✝¹ : Fintype α
                                                                       inst✝ : DecidableEq α
                                                                       s : Finset α
                                                                       ⊢ Eq (Finset.filter (fun x => Membership.mem s x) Finset.univ) s
                                                                     -/
lemma filter_univ_mem (s : Finset α) : univ.filter (· ∈ s) = s := by simp [filter_mem_eq_inter]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


instance decidableCodisjoint : Decidable (Codisjoint s t) :=
  decidable_of_iff _ codisjoint_left.symm


instance decidableIsCompl : Decidable (IsCompl s t) :=
  decidable_of_iff' _ isCompl_iff


/-- The inverse of an `hf : injective` function `f : α → β`, of the type `↥(Set.range f) → α`.
This is the computable version of `Function.invFun` that requires `Fintype α` and `DecidableEq β`,
or the function version of applying `(Equiv.ofInjective f hf).symm`.
This function should not usually be used for actual computation because for most cases,
an explicit inverse can be stated that has better computational properties.
This function computes by checking all terms `a : α` to find the `f a = b`, so it is O(N) where
`N = Fintype.card α`.
-/
def invOfMemRange : Set.range f → α := fun b =>
  Finset.choose (fun a => f a = b) Finset.univ
                             /-
                               α : Type u_1
                               β : Type u_2
                               γ : Type u_3
                               inst✝¹ : Fintype α
                               inst✝ : DecidableEq β
                               f : α → β
                               hf : Function.Injective f
                               b : ↑(Set.range f)
                               ⊢ ∀ (a : α), Iff (Eq (f a) ↑b) (And (Membership.mem Finset.univ a) ((fun a =>  …
                             -/
    ((existsUnique_congr (by simp)).mp (hf.existsUnique_of_mem_range b.property))
                             /-
                               🎉 no goals
                             -/


theorem left_inv_of_invOfMemRange (b : Set.range f) : f (hf.invOfMemRange b) = b :=
  (Finset.choose_spec (fun a => f a = b) _ _).right


@[simp]
theorem right_inv_of_invOfMemRange (a : α) : hf.invOfMemRange ⟨f a, Set.mem_range_self a⟩ = a :=
  hf (Finset.choose_spec (fun a' => f a' = f a) _ _).right


theorem invFun_restrict [Nonempty α] : (Set.range f).restrict (invFun f) = hf.invOfMemRange := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : Fintype α
    inst✝¹ : DecidableEq β
    f : α → β
    hf : Function.Injective f
    inst✝ : Nonempty α
    ⊢ Eq ((Set.range f).restrict (Function.invFun f)) hf.invOfMemRange
  -/
  ext ⟨b, h⟩
  /-
    case h.mk
    α : Type u_1
    β : Type u_2
    inst✝² : Fintype α
    inst✝¹ : DecidableEq β
    f : α → β
    hf : Function.Injective f
    inst✝ : Nonempty α
    b : β
    h : Membership.mem (Set.range f) b
    ⊢ Eq ((Set.range f).restrict (Function.invFun f) ⟨b, h⟩) (hf.invOfMemRange ⟨b, …
  -/
  apply hf
  /-
    case h.mk.a
    α : Type u_1
    β : Type u_2
    inst✝² : Fintype α
    inst✝¹ : DecidableEq β
    f : α → β
    hf : Function.Injective f
    inst✝ : Nonempty α
    b : β
    h : Membership.mem (Set.range f) b
    ⊢ Eq (f ((Set.range f).restrict (Function.invFun f) ⟨b, h⟩)) (f (hf.invOfMemRa …
  -/
  simp [hf.left_inv_of_invOfMemRange, @invFun_eq _ _ _ f b (Set.mem_range.mp h)]
  /-
    🎉 no goals
  -/


theorem invOfMemRange_surjective : Function.Surjective hf.invOfMemRange := fun a =>
                                   /-
                                     α : Type u_1
                                     β : Type u_2
                                     inst✝¹ : Fintype α
                                     inst✝ : DecidableEq β
                                     f : α → β
                                     hf : Function.Injective f
                                     a : α
                                     ⊢ Eq (hf.invOfMemRange ⟨f a, ⋯⟩) a
                                   -/
  ⟨⟨f a, Set.mem_range_self a⟩, by simp⟩
                                   /-
                                     🎉 no goals
                                   -/


/-- The inverse of an embedding `f : α ↪ β`, of the type `↥(Set.range f) → α`.
This is the computable version of `Function.invFun` that requires `Fintype α` and `DecidableEq β`,
or the function version of applying `(Equiv.ofInjective f f.injective).symm`.
This function should not usually be used for actual computation because for most cases,
an explicit inverse can be stated that has better computational properties.
This function computes by checking all terms `a : α` to find the `f a = b`, so it is O(N) where
`N = Fintype.card α`.
-/
def invOfMemRange : α :=
  f.injective.invOfMemRange b


@[simp]
theorem left_inv_of_invOfMemRange : f (f.invOfMemRange b) = b :=
  f.injective.left_inv_of_invOfMemRange b


@[simp]
theorem right_inv_of_invOfMemRange (a : α) : f.invOfMemRange ⟨f a, Set.mem_range_self a⟩ = a :=
  f.injective.right_inv_of_invOfMemRange a


theorem invFun_restrict [Nonempty α] : (Set.range f).restrict (invFun f) = f.invOfMemRange := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : Fintype α
    inst✝¹ : DecidableEq β
    f : Function.Embedding α β
    inst✝ : Nonempty α
    ⊢ Eq ((Set.range ⇑f).restrict (Function.invFun ⇑f)) f.invOfMemRange
  -/
  ext ⟨b, h⟩
  /-
    case h.mk
    α : Type u_1
    β : Type u_2
    inst✝² : Fintype α
    inst✝¹ : DecidableEq β
    f : Function.Embedding α β
    inst✝ : Nonempty α
    b : β
    h : Membership.mem (Set.range ⇑f) b
    ⊢ Eq ((Set.range ⇑f).restrict (Function.invFun ⇑f) ⟨b, h⟩) (f.invOfMemRange ⟨b …
  -/
  apply f.injective
  /-
    case h.mk.a
    α : Type u_1
    β : Type u_2
    inst✝² : Fintype α
    inst✝¹ : DecidableEq β
    f : Function.Embedding α β
    inst✝ : Nonempty α
    b : β
    h : Membership.mem (Set.range ⇑f) b
    ⊢ Eq (f ((Set.range ⇑f).restrict (Function.invFun ⇑f) ⟨b, h⟩)) (f (f.invOfMemR …
  -/
  simp [f.left_inv_of_invOfMemRange, @invFun_eq _ _ _ f b (Set.mem_range.mp h)]
  /-
    🎉 no goals
  -/


theorem invOfMemRange_surjective : Function.Surjective f.invOfMemRange := fun a =>
                                   /-
                                     α : Type u_1
                                     β : Type u_2
                                     inst✝¹ : Fintype α
                                     inst✝ : DecidableEq β
                                     f : Function.Embedding α β
                                     a : α
                                     ⊢ Eq (f.invOfMemRange ⟨f a, ⋯⟩) a
                                   -/
  ⟨⟨f a, Set.mem_range_self a⟩, by simp⟩
                                   /-
                                     🎉 no goals
                                   -/


/-- Given an injective function to a fintype, the domain is also a
fintype. This is noncomputable because injectivity alone cannot be
used to construct preimages. -/
noncomputable def ofInjective [Fintype β] (f : α → β) (H : Function.Injective f) : Fintype α :=
  letI := Classical.dec
  if hα : Nonempty α then
    letI := Classical.inhabited_of_nonempty hα
    ofSurjective (invFun f) (invFun_surjective H)
  else ⟨∅, fun x => (hα ⟨x⟩).elim⟩


/-- If `f : α ≃ β` and `α` is a fintype, then `β` is also a fintype. -/
def ofEquiv (α : Type*) [Fintype α] (f : α ≃ β) : Fintype β :=
  ofBijective _ f.bijective


/-- Any subsingleton type with a witness is a fintype (with one term). -/
def ofSubsingleton (a : α) [Subsingleton α] : Fintype α :=
  ⟨{a}, fun _ => Finset.mem_singleton.2 (Subsingleton.elim _ _)⟩

-- In principle, this could be a `simp` theorem but it applies to any occurrence of `univ` and
-- required unification of the (possibly very complex) `Fintype` instances.

theorem univ_ofSubsingleton (a : α) [Subsingleton α] : @univ _ (ofSubsingleton a) = {a} :=
  rfl


/-- An empty type is a fintype. Not registered as an instance, to make sure that there aren't two
conflicting `Fintype ι` instances around when casing over whether a fintype `ι` is empty or not. -/
def ofIsEmpty [IsEmpty α] : Fintype α :=
  ⟨∅, isEmptyElim⟩


/-- Note: this lemma is specifically about `Fintype.ofIsEmpty`. For a statement about
arbitrary `Fintype` instances, use `Finset.univ_eq_empty`. -/
theorem univ_ofIsEmpty [IsEmpty α] : @univ α Fintype.ofIsEmpty = ∅ :=
  rfl


instance : Fintype Empty := Fintype.ofIsEmpty

instance : Fintype PEmpty := Fintype.ofIsEmpty


/-- Construct a finset enumerating a set `s`, given a `Fintype` instance. -/
def toFinset (s : Set α) [Fintype s] : Finset α :=
  (@Finset.univ s _).map <| Function.Embedding.subtype _


@[congr]
theorem toFinset_congr {s t : Set α} [Fintype s] [Fintype t] (h : s = t) :
                                  /-
                                    α : Type u_1
                                    s t : Set α
                                    inst✝¹ : Fintype ↑s
                                    inst✝ : Fintype ↑t
                                    h : Eq s t
                                    ⊢ Eq s.toFinset t.toFinset
                                  -/
    toFinset s = toFinset t := by subst h; congr!
                                           /-
                                             🎉 no goals
                                           -/


@[simp]
theorem mem_toFinset {s : Set α} [Fintype s] {a : α} : a ∈ s.toFinset ↔ a ∈ s := by
  /-
    α : Type u_1
    s : Set α
    inst✝ : Fintype ↑s
    a : α
    ⊢ Iff (Membership.mem s.toFinset a) (Membership.mem s a)
  -/
  simp [toFinset]
  /-
    🎉 no goals
  -/


/-- Many `Fintype` instances for sets are defined using an extensionally equal `Finset`.
Rewriting `s.toFinset` with `Set.toFinset_ofFinset` replaces the term with such a `Finset`. -/
theorem toFinset_ofFinset {p : Set α} (s : Finset α) (H : ∀ x, x ∈ s ↔ x ∈ p) :
    @Set.toFinset _ p (Fintype.ofFinset s H) = s :=
                         /-
                           α : Type u_1
                           p : Set α
                           s : Finset α
                           H : ∀ (x : α), Iff (Membership.mem s x) (Membership.mem p x)
                           x : α
                           ⊢ Iff (Membership.mem p.toFinset x) (Membership.mem s x)
                         -/
  Finset.ext fun x => by rw [@mem_toFinset _ _ (id _), H]
                         /-
                           🎉 no goals
                         -/


/-- Membership of a set with a `Fintype` instance is decidable.

Using this as an instance leads to potential loops with `Subtype.fintype` under certain decidability
assumptions, so it should only be declared a local instance. -/
def decidableMemOfFintype [DecidableEq α] (s : Set α) [Fintype s] (a) : Decidable (a ∈ s) :=
  decidable_of_iff _ mem_toFinset


@[simp]
theorem coe_toFinset (s : Set α) [Fintype s] : (↑s.toFinset : Set α) = s :=
  Set.ext fun _ => mem_toFinset


@[simp]
theorem toFinset_nonempty {s : Set α} [Fintype s] : s.toFinset.Nonempty ↔ s.Nonempty := by
  /-
    α : Type u_1
    s : Set α
    inst✝ : Fintype ↑s
    ⊢ Iff s.toFinset.Nonempty s.Nonempty
  -/
  rw [← Finset.coe_nonempty, coe_toFinset]
  /-
    🎉 no goals
  -/


@[aesop safe apply (rule_sets := [finsetNonempty])]
alias ⟨_, Aesop.toFinset_nonempty_of_nonempty⟩ := toFinset_nonempty


@[simp]
theorem toFinset_inj {s t : Set α} [Fintype s] [Fintype t] : s.toFinset = t.toFinset ↔ s = t :=
               /-
                 α : Type u_1
                 s t : Set α
                 inst✝¹ : Fintype ↑s
                 inst✝ : Fintype ↑t
                 h : Eq s.toFinset t.toFinset
                 ⊢ Eq s t
               -/
               /-
                 🎉 no goals
               -/
  ⟨fun h => by rw [← s.coe_toFinset, h, t.coe_toFinset], fun h => by simp [h]⟩
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[mono]
theorem toFinset_subset_toFinset [Fintype s] [Fintype t] : s.toFinset ⊆ t.toFinset ↔ s ⊆ t := by
  /-
    α : Type u_1
    s t : Set α
    inst✝¹ : Fintype ↑s
    inst✝ : Fintype ↑t
    ⊢ Iff (HasSubset.Subset s.toFinset t.toFinset) (HasSubset.Subset s t)
  -/
  simp [Finset.subset_iff, Set.subset_def]
  /-
    🎉 no goals
  -/


@[simp]
theorem toFinset_ssubset [Fintype s] {t : Finset α} : s.toFinset ⊂ t ↔ s ⊂ t := by
  /-
    α : Type u_1
    s : Set α
    inst✝ : Fintype ↑s
    t : Finset α
    ⊢ Iff (HasSSubset.SSubset s.toFinset t) (HasSSubset.SSubset s ↑t)
  -/
  rw [← Finset.coe_ssubset, coe_toFinset]
  /-
    🎉 no goals
  -/


@[simp]
theorem subset_toFinset {s : Finset α} [Fintype t] : s ⊆ t.toFinset ↔ ↑s ⊆ t := by
  /-
    α : Type u_1
    t : Set α
    s : Finset α
    inst✝ : Fintype ↑t
    ⊢ Iff (HasSubset.Subset s t.toFinset) (HasSubset.Subset (↑s) t)
  -/
  rw [← Finset.coe_subset, coe_toFinset]
  /-
    🎉 no goals
  -/


@[simp]
theorem ssubset_toFinset {s : Finset α} [Fintype t] : s ⊂ t.toFinset ↔ ↑s ⊂ t := by
  /-
    α : Type u_1
    t : Set α
    s : Finset α
    inst✝ : Fintype ↑t
    ⊢ Iff (HasSSubset.SSubset s t.toFinset) (HasSSubset.SSubset (↑s) t)
  -/
  rw [← Finset.coe_ssubset, coe_toFinset]
  /-
    🎉 no goals
  -/


@[mono]
theorem toFinset_ssubset_toFinset [Fintype s] [Fintype t] : s.toFinset ⊂ t.toFinset ↔ s ⊂ t := by
  /-
    α : Type u_1
    s t : Set α
    inst✝¹ : Fintype ↑s
    inst✝ : Fintype ↑t
    ⊢ Iff (HasSSubset.SSubset s.toFinset t.toFinset) (HasSSubset.SSubset s t)
  -/
  simp only [Finset.ssubset_def, toFinset_subset_toFinset, ssubset_def]
  /-
    🎉 no goals
  -/


@[simp]
theorem toFinset_subset [Fintype s] {t : Finset α} : s.toFinset ⊆ t ↔ s ⊆ t := by
  /-
    α : Type u_1
    s : Set α
    inst✝ : Fintype ↑s
    t : Finset α
    ⊢ Iff (HasSubset.Subset s.toFinset t) (HasSubset.Subset s ↑t)
  -/
  rw [← Finset.coe_subset, coe_toFinset]
  /-
    🎉 no goals
  -/


alias ⟨_, toFinset_mono⟩ := toFinset_subset_toFinset


alias ⟨_, toFinset_strict_mono⟩ := toFinset_ssubset_toFinset


@[simp]
theorem disjoint_toFinset [Fintype s] [Fintype t] :
                                                        /-
                                                          α : Type u_1
                                                          s t : Set α
                                                          inst✝¹ : Fintype ↑s
                                                          inst✝ : Fintype ↑t
                                                          ⊢ Iff (Disjoint s.toFinset t.toFinset) (Disjoint s t)
                                                        -/
    Disjoint s.toFinset t.toFinset ↔ Disjoint s t := by simp only [← disjoint_coe, coe_toFinset]
                                                        /-
                                                          🎉 no goals
                                                        -/


@[simp]
theorem toFinset_inter [Fintype (s ∩ t : Set _)] : (s ∩ t).toFinset = s.toFinset ∩ t.toFinset := by
  /-
    α : Type u_1
    s t : Set α
    inst✝³ : DecidableEq α
    inst✝² : Fintype ↑s
    inst✝¹ : Fintype ↑t
    inst✝ : Fintype ↑(Inter.inter s t)
    ⊢ Eq (Inter.inter s t).toFinset (Inter.inter s.toFinset t.toFinset)
  -/
  ext
  /-
    case h
    α : Type u_1
    s t : Set α
    inst✝³ : DecidableEq α
    inst✝² : Fintype ↑s
    inst✝¹ : Fintype ↑t
    inst✝ : Fintype ↑(Inter.inter s t)
    a✝ : α
    ⊢ Iff (Membership.mem (Inter.inter s t).toFinset a✝) (Membership.mem (Inter.in …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem toFinset_union [Fintype (s ∪ t : Set _)] : (s ∪ t).toFinset = s.toFinset ∪ t.toFinset := by
  /-
    α : Type u_1
    s t : Set α
    inst✝³ : DecidableEq α
    inst✝² : Fintype ↑s
    inst✝¹ : Fintype ↑t
    inst✝ : Fintype ↑(Union.union s t)
    ⊢ Eq (Union.union s t).toFinset (Union.union s.toFinset t.toFinset)
  -/
  ext
  /-
    case h
    α : Type u_1
    s t : Set α
    inst✝³ : DecidableEq α
    inst✝² : Fintype ↑s
    inst✝¹ : Fintype ↑t
    inst✝ : Fintype ↑(Union.union s t)
    a✝ : α
    ⊢ Iff (Membership.mem (Union.union s t).toFinset a✝) (Membership.mem (Union.un …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem toFinset_diff [Fintype (s \ t : Set _)] : (s \ t).toFinset = s.toFinset \ t.toFinset := by
  /-
    α : Type u_1
    s t : Set α
    inst✝³ : DecidableEq α
    inst✝² : Fintype ↑s
    inst✝¹ : Fintype ↑t
    inst✝ : Fintype ↑(SDiff.sdiff s t)
    ⊢ Eq (SDiff.sdiff s t).toFinset (SDiff.sdiff s.toFinset t.toFinset)
  -/
  ext
  /-
    case h
    α : Type u_1
    s t : Set α
    inst✝³ : DecidableEq α
    inst✝² : Fintype ↑s
    inst✝¹ : Fintype ↑t
    inst✝ : Fintype ↑(SDiff.sdiff s t)
    a✝ : α
    ⊢ Iff (Membership.mem (SDiff.sdiff s t).toFinset a✝) (Membership.mem (SDiff.sd …
  -/
  simp
  /-
    🎉 no goals
  -/


open scoped symmDiff in
@[simp]
theorem toFinset_symmDiff [Fintype (s ∆ t : Set _)] :
    (s ∆ t).toFinset = s.toFinset ∆ t.toFinset := by
  /-
    α : Type u_1
    s t : Set α
    inst✝³ : DecidableEq α
    inst✝² : Fintype ↑s
    inst✝¹ : Fintype ↑t
    inst✝ : Fintype ↑(symmDiff s t)
    ⊢ Eq (symmDiff s t).toFinset (symmDiff s.toFinset t.toFinset)
  -/
  ext
  /-
    case h
    α : Type u_1
    s t : Set α
    inst✝³ : DecidableEq α
    inst✝² : Fintype ↑s
    inst✝¹ : Fintype ↑t
    inst✝ : Fintype ↑(symmDiff s t)
    a✝ : α
    ⊢ Iff (Membership.mem (symmDiff s t).toFinset a✝) (Membership.mem (symmDiff s. …
  -/
  simp [mem_symmDiff, Finset.mem_symmDiff]
  /-
    🎉 no goals
  -/


@[simp]
theorem toFinset_compl [Fintype α] [Fintype (sᶜ : Set _)] : sᶜ.toFinset = s.toFinsetᶜ := by
  /-
    α : Type u_1
    s : Set α
    inst✝³ : DecidableEq α
    inst✝² : Fintype ↑s
    inst✝¹ : Fintype α
    inst✝ : Fintype ↑(HasCompl.compl s)
    ⊢ Eq (HasCompl.compl s).toFinset (HasCompl.compl s.toFinset)
  -/
  ext
  /-
    case h
    α : Type u_1
    s : Set α
    inst✝³ : DecidableEq α
    inst✝² : Fintype ↑s
    inst✝¹ : Fintype α
    inst✝ : Fintype ↑(HasCompl.compl s)
    a✝ : α
    ⊢ Iff (Membership.mem (HasCompl.compl s).toFinset a✝) (Membership.mem (HasComp …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem toFinset_empty [Fintype (∅ : Set α)] : (∅ : Set α).toFinset = ∅ := by
  /-
    α : Type u_1
    inst✝ : Fintype ↑EmptyCollection.emptyCollection
    ⊢ Eq EmptyCollection.emptyCollection.toFinset EmptyCollection.emptyCollection
  -/
  ext
  /-
    case h
    α : Type u_1
    inst✝ : Fintype ↑EmptyCollection.emptyCollection
    a✝ : α
    ⊢ Iff (Membership.mem EmptyCollection.emptyCollection.toFinset a✝) (Membership …
  -/
  simp
  /-
    🎉 no goals
  -/

/- TODO Without the coercion arrow (`↥`) there is an elaboration bug in the following two;
it essentially infers `Fintype.{v} (Set.univ.{u} : Set α)` with `v` and `u` distinct.
Reported in https://github.com/leanprover-community/lean/issues/672 -/

@[simp]
theorem toFinset_univ [Fintype α] [Fintype (Set.univ : Set α)] :
    (Set.univ : Set α).toFinset = Finset.univ := by
  /-
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : Fintype ↑Set.univ
    ⊢ Eq Set.univ.toFinset Finset.univ
  -/
  ext
  /-
    case h
    α : Type u_1
    inst✝¹ : Fintype α
    inst✝ : Fintype ↑Set.univ
    a✝ : α
    ⊢ Iff (Membership.mem Set.univ.toFinset a✝) (Membership.mem Finset.univ a✝)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem toFinset_eq_empty [Fintype s] : s.toFinset = ∅ ↔ s = ∅ := by
  /-
    α : Type u_1
    s : Set α
    inst✝ : Fintype ↑s
    ⊢ Iff (Eq s.toFinset EmptyCollection.emptyCollection) (Eq s EmptyCollection.em …
  -/
  let A : Fintype (∅ : Set α) := Fintype.ofIsEmpty
  /-
    α : Type u_1
    s : Set α
    inst✝ : Fintype ↑s
    A : Fintype ↑EmptyCollection.emptyCollection := Fintype.ofIsEmpty
    ⊢ Iff (Eq s.toFinset EmptyCollection.emptyCollection) (Eq s EmptyCollection.em …
  -/
  rw [← toFinset_empty, toFinset_inj]
  /-
    🎉 no goals
  -/


@[simp]
theorem toFinset_eq_univ [Fintype α] [Fintype s] : s.toFinset = Finset.univ ↔ s = univ := by
  /-
    α : Type u_1
    s : Set α
    inst✝¹ : Fintype α
    inst✝ : Fintype ↑s
    ⊢ Iff (Eq s.toFinset Finset.univ) (Eq s Set.univ)
  -/
  rw [← coe_inj, coe_toFinset, coe_univ]
  /-
    🎉 no goals
  -/


@[simp]
theorem toFinset_setOf [Fintype α] (p : α → Prop) [DecidablePred p] [Fintype { x | p x }] :
    Set.toFinset {x | p x} = Finset.univ.filter p := by
  /-
    α : Type u_1
    inst✝² : Fintype α
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : Fintype ↑(setOf fun x => p x)
    ⊢ Eq (setOf fun x => p x).toFinset (Finset.filter p Finset.univ)
  -/
  ext
  /-
    case h
    α : Type u_1
    inst✝² : Fintype α
    p : α → Prop
    inst✝¹ : DecidablePred p
    inst✝ : Fintype ↑(setOf fun x => p x)
    a✝ : α
    ⊢ Iff (Membership.mem (setOf fun x => p x).toFinset a✝) (Membership.mem (Finse …
  -/
  simp
  /-
    🎉 no goals
  -/

--@[simp] Porting note: removing simp, simp can prove it

theorem toFinset_ssubset_univ [Fintype α] {s : Set α} [Fintype s] :
                                              /-
                                                α : Type u_1
                                                inst✝¹ : Fintype α
                                                s : Set α
                                                inst✝ : Fintype ↑s
                                                ⊢ Iff (HasSSubset.SSubset s.toFinset Finset.univ) (HasSSubset.SSubset s Set.un …
                                              -/
    s.toFinset ⊂ Finset.univ ↔ s ⊂ univ := by rw [← coe_ssubset, coe_toFinset, coe_univ]
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
theorem toFinset_image [DecidableEq β] (f : α → β) (s : Set α) [Fintype s] [Fintype (f '' s)] :
    (f '' s).toFinset = s.toFinset.image f :=
                             /-
                               α : Type u_1
                               β : Type u_2
                               inst✝² : DecidableEq β
                               f : α → β
                               s : Set α
                               inst✝¹ : Fintype ↑s
                               inst✝ : Fintype ↑(Set.image f s)
                               ⊢ Eq ↑(Set.image f s).toFinset ↑(Finset.image f s.toFinset)
                             -/
  Finset.coe_injective <| by simp
                             /-
                               🎉 no goals
                             -/


@[simp]
theorem toFinset_range [DecidableEq α] [Fintype β] (f : β → α) [Fintype (Set.range f)] :
    (Set.range f).toFinset = Finset.univ.image f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : DecidableEq α
    inst✝¹ : Fintype β
    f : β → α
    inst✝ : Fintype ↑(Set.range f)
    ⊢ Eq (Set.range f).toFinset (Finset.image f Finset.univ)
  -/
  ext
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝² : DecidableEq α
    inst✝¹ : Fintype β
    f : β → α
    inst✝ : Fintype ↑(Set.range f)
    a✝ : α
    ⊢ Iff (Membership.mem (Set.range f).toFinset a✝) (Membership.mem (Finset.image …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp] -- Porting note: new attribute
theorem toFinset_singleton (a : α) [Fintype ({a} : Set α)] : ({a} : Set α).toFinset = {a} := by
  /-
    α : Type u_1
    a : α
    inst✝ : Fintype ↑(Singleton.singleton a)
    ⊢ Eq (Singleton.singleton a).toFinset (Singleton.singleton a)
  -/
  ext
  /-
    case h
    α : Type u_1
    a : α
    inst✝ : Fintype ↑(Singleton.singleton a)
    a✝ : α
    ⊢ Iff (Membership.mem (Singleton.singleton a).toFinset a✝) (Membership.mem (Si …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem toFinset_insert [DecidableEq α] {a : α} {s : Set α} [Fintype (insert a s : Set α)]
    [Fintype s] : (insert a s).toFinset = insert a s.toFinset := by
  /-
    α : Type u_1
    inst✝² : DecidableEq α
    a : α
    s : Set α
    inst✝¹ : Fintype ↑(Insert.insert a s)
    inst✝ : Fintype ↑s
    ⊢ Eq (Insert.insert a s).toFinset (Insert.insert a s.toFinset)
  -/
  ext
  /-
    case h
    α : Type u_1
    inst✝² : DecidableEq α
    a : α
    s : Set α
    inst✝¹ : Fintype ↑(Insert.insert a s)
    inst✝ : Fintype ↑s
    a✝ : α
    ⊢ Iff (Membership.mem (Insert.insert a s).toFinset a✝) (Membership.mem (Insert …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem filter_mem_univ_eq_toFinset [Fintype α] (s : Set α) [Fintype s] [DecidablePred (· ∈ s)] :
    Finset.univ.filter (· ∈ s) = s.toFinset := by
  /-
    α : Type u_1
    inst✝² : Fintype α
    s : Set α
    inst✝¹ : Fintype ↑s
    inst✝ : DecidablePred fun x => Membership.mem s x
    ⊢ Eq (Finset.filter (fun x => Membership.mem s x) Finset.univ) s.toFinset
  -/
  ext
  simp only [Finset.mem_univ, decide_eq_true_eq, forall_true_left, mem_filter,
    true_and, mem_toFinset]


@[simp]
theorem Finset.toFinset_coe (s : Finset α) [Fintype (s : Set α)] : (s : Set α).toFinset = s :=
  ext fun _ => Set.mem_toFinset


instance Fin.fintype (n : ℕ) : Fintype (Fin n) :=
  ⟨⟨List.finRange n, List.nodup_finRange n⟩, List.mem_finRange⟩


theorem Fin.univ_def (n : ℕ) : (univ : Finset (Fin n)) = ⟨List.finRange n, List.nodup_finRange n⟩ :=
  rfl


/-- See also `nonempty_encodable`, `nonempty_denumerable`. -/
theorem nonempty_fintype (α : Type*) [Finite α] : Nonempty (Fintype α) := by
  /-
    α : Type u_4
    inst✝ : Finite α
    ⊢ Nonempty (Fintype α)
  -/
  rcases Finite.exists_equiv_fin α with ⟨n, ⟨e⟩⟩
  /-
    case intro.intro
    α : Type u_4
    inst✝ : Finite α
    n : Nat
    e : Equiv α (Fin n)
    ⊢ Nonempty (Fintype α)
  -/
  exact ⟨.ofEquiv _ e.symm⟩
  /-
    🎉 no goals
  -/


@[simp] theorem List.toFinset_finRange (n : ℕ) : (List.finRange n).toFinset = Finset.univ := by
  /-
    n : Nat
    ⊢ Eq (List.finRange n).toFinset Finset.univ
  -/
  ext; simp
       /-
         🎉 no goals
       -/


@[simp] theorem Fin.univ_val_map {n : ℕ} (f : Fin n → α) :
    Finset.univ.val.map f = List.ofFn f := by
  /-
    α : Type u_1
    n : Nat
    f : Fin n → α
    ⊢ Eq (Multiset.map f Finset.univ.val) ↑(List.ofFn f)
  -/
  simp [List.ofFn_eq_map, univ_def]
  /-
    🎉 no goals
  -/


theorem Fin.univ_image_def {n : ℕ} [DecidableEq α] (f : Fin n → α) :
    Finset.univ.image f = (List.ofFn f).toFinset := by
  /-
    α : Type u_1
    n : Nat
    inst✝ : DecidableEq α
    f : Fin n → α
    ⊢ Eq (Finset.image f Finset.univ) (List.ofFn f).toFinset
  -/
  simp [Finset.image]
  /-
    🎉 no goals
  -/


theorem Fin.univ_map_def {n : ℕ} (f : Fin n ↪ α) :
    Finset.univ.map f = ⟨List.ofFn f, List.nodup_ofFn.mpr f.injective⟩ := by
  /-
    α : Type u_1
    n : Nat
    f : Function.Embedding (Fin n) α
    ⊢ Eq (Finset.map f Finset.univ) { val := ↑(List.ofFn ⇑f), nodup := ⋯ }
  -/
  simp [Finset.map]
  /-
    🎉 no goals
  -/


@[simp]
theorem Fin.image_succAbove_univ {n : ℕ} (i : Fin (n + 1)) : univ.image i.succAbove = {i}ᶜ := by
  /-
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    ⊢ Eq (Finset.image i.succAbove Finset.univ) (HasCompl.compl (Singleton.singlet …
  -/
  ext m
  /-
    case h
    n : Nat
    i m : Fin (HAdd.hAdd n 1)
    ⊢ Iff (Membership.mem (Finset.image i.succAbove Finset.univ) m) (Membership.me …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem Fin.image_succ_univ (n : ℕ) : (univ : Finset (Fin n)).image Fin.succ = {0}ᶜ := by
  /-
    n : Nat
    ⊢ Eq (Finset.image Fin.succ Finset.univ) (HasCompl.compl (Singleton.singleton  …
  -/
  rw [← Fin.succAbove_zero, Fin.image_succAbove_univ]
  /-
    🎉 no goals
  -/


@[simp]
theorem Fin.image_castSucc (n : ℕ) :
    (univ : Finset (Fin n)).image Fin.castSucc = {Fin.last n}ᶜ := by
  /-
    n : Nat
    ⊢ Eq (Finset.image Fin.castSucc Finset.univ) (HasCompl.compl (Singleton.single …
  -/
  rw [← Fin.succAbove_last, Fin.image_succAbove_univ]
  /-
    🎉 no goals
  -/

/- The following three lemmas use `Finset.cons` instead of `insert` and `Finset.map` instead of
`Finset.image` to reduce proof obligations downstream. -/

/-- Embed `Fin n` into `Fin (n + 1)` by prepending zero to the `univ` -/
theorem Fin.univ_succ (n : ℕ) :
    (univ : Finset (Fin (n + 1))) =
                                                                    /-
                                                                      α : Type u_1
                                                                      β : Type u_2
                                                                      γ : Type u_3
                                                                      n : Nat
                                                                      ⊢ Not (Membership.mem (Finset.map { toFun := Fin.succ, inj' := ⋯ } Finset.univ …
                                                                    -/
      Finset.cons 0 (univ.map ⟨Fin.succ, Fin.succ_injective _⟩) (by simp [map_eq_image]) := by
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
  /-
    n : Nat
    ⊢ Eq Finset.univ (Finset.cons 0 (Finset.map { toFun := Fin.succ, inj' := ⋯ } F …
  -/
  simp [map_eq_image]
  /-
    🎉 no goals
  -/


/-- Embed `Fin n` into `Fin (n + 1)` by appending a new `Fin.last n` to the `univ` -/
theorem Fin.univ_castSuccEmb (n : ℕ) :
    (univ : Finset (Fin (n + 1))) =
                                                              /-
                                                                α : Type u_1
                                                                β : Type u_2
                                                                γ : Type u_3
                                                                n : Nat
                                                                ⊢ Not (Membership.mem (Finset.map Fin.castSuccEmb Finset.univ) (Fin.last n))
                                                              -/
      Finset.cons (Fin.last n) (univ.map Fin.castSuccEmb) (by simp [map_eq_image]) := by
                                                              /-
                                                                🎉 no goals
                                                              -/
  /-
    n : Nat
    ⊢ Eq Finset.univ (Finset.cons (Fin.last n) (Finset.map Fin.castSuccEmb Finset. …
  -/
  simp [map_eq_image]
  /-
    🎉 no goals
  -/


/-- Embed `Fin n` into `Fin (n + 1)` by inserting
around a specified pivot `p : Fin (n + 1)` into the `univ` -/
theorem Fin.univ_succAbove (n : ℕ) (p : Fin (n + 1)) :
                                                                                       /-
                                                                                         α : Type u_1
                                                                                         β : Type u_2
                                                                                         γ : Type u_3
                                                                                         n : Nat
                                                                                         p : Fin (HAdd.hAdd n 1)
                                                                                         ⊢ Not (Membership.mem (Finset.map p.succAboveEmb Finset.univ) p)
                                                                                       -/
    (univ : Finset (Fin (n + 1))) = Finset.cons p (univ.map <| Fin.succAboveEmb p) (by simp) := by
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/
  /-
    n : Nat
    p : Fin (HAdd.hAdd n 1)
    ⊢ Eq Finset.univ (Finset.cons p (Finset.map p.succAboveEmb Finset.univ) ⋯)
  -/
  simp [map_eq_image]
  /-
    🎉 no goals
  -/


@[simp] theorem Fin.univ_image_get [DecidableEq α] (l : List α) :
    Finset.univ.image l.get = l.toFinset := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    l : List α
    ⊢ Eq (Finset.image l.get Finset.univ) l.toFinset
  -/
  simp [univ_image_def]
  /-
    🎉 no goals
  -/


@[simp] theorem Fin.univ_image_getElem' [DecidableEq β] (l : List α) (f : α → β) :
    Finset.univ.image (fun i : Fin l.length => f <| l[(i : Nat)]) = (l.map f).toFinset := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : DecidableEq β
    l : List α
    f : α → β
    ⊢ Eq (Finset.image (fun i => f (GetElem.getElem l ↑i ⋯)) Finset.univ) (List.ma …
  -/
  simp only [univ_image_def, List.ofFn_getElem_eq_map]
  /-
    🎉 no goals
  -/


theorem Fin.univ_image_get' [DecidableEq β] (l : List α) (f : α → β) :
    Finset.univ.image (f <| l.get ·) = (l.map f).toFinset := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : DecidableEq β
    l : List α
    f : α → β
    ⊢ Eq (Finset.image (fun x => f (l.get x)) Finset.univ) (List.map f l).toFinset
  -/
  simp
  /-
    🎉 no goals
  -/


@[instance]
def Unique.fintype {α : Type*} [Unique α] : Fintype α :=
  Fintype.ofSubsingleton default


/-- Short-circuit instance to decrease search for `Unique.fintype`,
since that relies on a subsingleton elimination for `Unique`. -/
instance Fintype.subtypeEq (y : α) : Fintype { x // x = y } :=
                          /-
                            α : Type u_1
                            β : Type u_2
                            γ : Type u_3
                            y : α
                            ⊢ ∀ (x : α), Iff (Membership.mem (Singleton.singleton y) x) (Eq x y)
                          -/
  Fintype.subtype {y} (by simp)
                          /-
                            🎉 no goals
                          -/


/-- Short-circuit instance to decrease search for `Unique.fintype`,
since that relies on a subsingleton elimination for `Unique`. -/
instance Fintype.subtypeEq' (y : α) : Fintype { x // y = x } :=
                          /-
                            α : Type u_1
                            β : Type u_2
                            γ : Type u_3
                            y : α
                            ⊢ ∀ (x : α), Iff (Membership.mem (Singleton.singleton y) x) (Eq y x)
                          -/
  Fintype.subtype {y} (by simp [eq_comm])
                          /-
                            🎉 no goals
                          -/

-- Porting note: removing @[simp], simp can prove it

theorem Fintype.univ_empty : @univ Empty _ = ∅ :=
  rfl

--@[simp] Porting note: removing simp, simp can prove it

theorem Fintype.univ_pempty : @univ PEmpty _ = ∅ :=
  rfl


instance Unit.fintype : Fintype Unit :=
  Fintype.ofSubsingleton ()


theorem Fintype.univ_unit : @univ Unit _ = {()} :=
  rfl


instance PUnit.fintype : Fintype PUnit :=
  Fintype.ofSubsingleton PUnit.unit

--@[simp] Porting note: removing simp, simp can prove it

theorem Fintype.univ_punit : @univ PUnit _ = {PUnit.unit} :=
  rfl


instance Bool.fintype : Fintype Bool :=
                      /-
                        α : Type u_1
                        β : Type u_2
                        γ : Type u_3
                        ⊢ (Insert.insert Bool.true (Singleton.singleton Bool.false)).Nodup
                      -/
                      /-
                        🎉 no goals
                      -/
                                                     /-
                                                       🎉 no goals
                                                     -/
  ⟨⟨{true, false}, by simp⟩, fun x => by cases x <;> simp⟩
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp]
theorem Fintype.univ_bool : @univ Bool _ = {true, false} :=
  rfl


instance Additive.fintype : ∀ [Fintype α], Fintype (Additive α) :=
  Fintype.ofEquiv α Additive.ofMul


instance Multiplicative.fintype : ∀ [Fintype α], Fintype (Multiplicative α) :=
  Fintype.ofEquiv α Multiplicative.ofAdd


/-- Given that `α × β` is a fintype, `α` is also a fintype. -/
def Fintype.prodLeft {α β} [DecidableEq α] [Fintype (α × β)] [Nonempty β] : Fintype α :=
                                                 /-
                                                   α✝ : Type u_1
                                                   β✝ : Type u_2
                                                   γ : Type u_3
                                                   α : Type ?u.93936
                                                   β : Type ?u.93935
                                                   inst✝² : DecidableEq α
                                                   inst✝¹ : Fintype (Prod α β)
                                                   inst✝ : Nonempty β
                                                   a : α
                                                   ⊢ Membership.mem (Finset.image Prod.fst Finset.univ) a
                                                 -/
  ⟨(@univ (α × β) _).image Prod.fst, fun a => by simp⟩
                                                 /-
                                                   🎉 no goals
                                                 -/


/-- Given that `α × β` is a fintype, `β` is also a fintype. -/
def Fintype.prodRight {α β} [DecidableEq β] [Fintype (α × β)] [Nonempty α] : Fintype β :=
                                                 /-
                                                   α✝ : Type u_1
                                                   β✝ : Type u_2
                                                   γ : Type u_3
                                                   α : Type ?u.96661
                                                   β : Type ?u.96660
                                                   inst✝² : DecidableEq β
                                                   inst✝¹ : Fintype (Prod α β)
                                                   inst✝ : Nonempty α
                                                   b : β
                                                   ⊢ Membership.mem (Finset.image Prod.snd Finset.univ) b
                                                 -/
  ⟨(@univ (α × β) _).image Prod.snd, fun b => by simp⟩
                                                 /-
                                                   🎉 no goals
                                                 -/


instance ULift.fintype (α : Type*) [Fintype α] : Fintype (ULift α) :=
  Fintype.ofEquiv _ Equiv.ulift.symm


instance PLift.fintype (α : Type*) [Fintype α] : Fintype (PLift α) :=
  Fintype.ofEquiv _ Equiv.plift.symm


instance OrderDual.fintype (α : Type*) [Fintype α] : Fintype αᵒᵈ :=
  ‹Fintype α›


instance OrderDual.finite (α : Type*) [Finite α] : Finite αᵒᵈ :=
  ‹Finite α›


instance Lex.fintype (α : Type*) [Fintype α] : Fintype (Lex α) :=
  ‹Fintype α›


instance Finset.fintypeCoeSort {α : Type u} (s : Finset α) : Fintype s :=
  ⟨s.attach, s.mem_attach⟩


@[simp]
theorem Finset.univ_eq_attach {α : Type u} (s : Finset α) : (univ : Finset s) = s.attach :=
  rfl


theorem Fintype.coe_image_univ [Fintype α] [DecidableEq β] {f : α → β} :
    ↑(Finset.image f Finset.univ) = Set.range f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Fintype α
    inst✝ : DecidableEq β
    f : α → β
    ⊢ Eq (↑(Finset.image f Finset.univ)) (Set.range f)
  -/
  ext x
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝¹ : Fintype α
    inst✝ : DecidableEq β
    f : α → β
    x : β
    ⊢ Iff (Membership.mem (↑(Finset.image f Finset.univ)) x) (Membership.mem (Set. …
  -/
  simp
  /-
    🎉 no goals
  -/


instance List.Subtype.fintype [DecidableEq α] (l : List α) : Fintype { x // x ∈ l } :=
  Fintype.ofList l.attach l.mem_attach


instance Multiset.Subtype.fintype [DecidableEq α] (s : Multiset α) : Fintype { x // x ∈ s } :=
  Fintype.ofMultiset s.attach s.mem_attach


instance Finset.Subtype.fintype (s : Finset α) : Fintype { x // x ∈ s } :=
  ⟨s.attach, s.mem_attach⟩


instance FinsetCoe.fintype (s : Finset α) : Fintype (↑s : Set α) :=
  Finset.Subtype.fintype s


theorem Finset.attach_eq_univ {s : Finset α} : s.attach = Finset.univ :=
  rfl


instance PLift.fintypeProp (p : Prop) [Decidable p] : Fintype (PLift p) :=
                                             /-
                                               α : Type u_1
                                               β : Type u_2
                                               γ : Type u_3
                                               p : Prop
                                               inst✝ : Decidable p
                                               x✝ : PLift p
                                               h : p
                                               ⊢ Membership.mem (dite p (fun h => Singleton.singleton { down := h }) fun h => …
                                             -/
  ⟨if h : p then {⟨h⟩} else ∅, fun ⟨h⟩ => by simp [h]⟩
                                             /-
                                               🎉 no goals
                                             -/


instance Prop.fintype : Fintype Prop :=
                      /-
                        α : Type u_1
                        β : Type u_2
                        γ : Type u_3
                        ⊢ (Insert.insert True (Singleton.singleton False)).Nodup
                      -/
                      /-
                        🎉 no goals
                      -/
  ⟨⟨{True, False}, by simp [true_ne_false]⟩, by simpa using em⟩
                                                /-
                                                  🎉 no goals
                                                -/


@[simp]
theorem Fintype.univ_Prop : (Finset.univ : Finset Prop) = {True, False} :=
                         /-
                           ⊢ Eq Finset.univ.val (Insert.insert True (Singleton.singleton False)).val
                         -/
  Finset.eq_of_veq <| by simp; rfl
                               /-
                                 🎉 no goals
                               -/


instance Subtype.fintype (p : α → Prop) [DecidablePred p] [Fintype α] : Fintype { x // p x } :=
                                      /-
                                        α : Type u_1
                                        β : Type u_2
                                        γ : Type u_3
                                        p : α → Prop
                                        inst✝¹ : DecidablePred p
                                        inst✝ : Fintype α
                                        ⊢ ∀ (x : α), Iff (Membership.mem (Finset.filter p Finset.univ) x) (p x)
                                      -/
  Fintype.subtype (univ.filter p) (by simp)
                                      /-
                                        🎉 no goals
                                      -/


/-- A set on a fintype, when coerced to a type, is a fintype. -/
def setFintype [Fintype α] (s : Set α) [DecidablePred (· ∈ s)] : Fintype s :=
  Subtype.fintype fun x => x ∈ s


/-- Given `Fintype α`, `finsetEquivSet` is the equiv between `Finset α` and `Set α`. (All
sets on a finite type are finite.) -/
noncomputable def finsetEquivSet : Finset α ≃ Set α where
  toFun := (↑)
               /-
                 α : Type u_1
                 β : Type u_2
                 γ : Type u_3
                 inst✝ : Fintype α
                 ⊢ Set α → Finset α
               -/
  invFun := by classical exact fun s => s.toFinset
               /-
                 🎉 no goals
               -/
                   /-
                     α : Type u_1
                     β : Type u_2
                     γ : Type u_3
                     inst✝ : Fintype α
                     s : Finset α
                     ⊢ Eq (↑s).toFinset s
                   -/
  left_inv s := by convert Finset.toFinset_coe s
                   /-
                     🎉 no goals
                   -/
                    /-
                      α : Type u_1
                      β : Type u_2
                      γ : Type u_3
                      inst✝ : Fintype α
                      s : Set α
                      ⊢ Eq (↑s.toFinset) s
                    -/
  right_inv s := by classical exact s.coe_toFinset
                    /-
                      🎉 no goals
                    -/


@[simp, norm_cast] lemma coe_finsetEquivSet : ⇑finsetEquivSet = ((↑) : Finset α → Set α) := rfl


@[simp] lemma finsetEquivSet_apply (s : Finset α) : finsetEquivSet s = s := rfl


@[simp] lemma finsetEquivSet_symm_apply (s : Set α) [Fintype s] :
                                             /-
                                               α : Type u_1
                                               inst✝¹ : Fintype α
                                               s : Set α
                                               inst✝ : Fintype ↑s
                                               ⊢ Eq (Fintype.finsetEquivSet.symm s) s.toFinset
                                             -/
    finsetEquivSet.symm s = s.toFinset := by simp [finsetEquivSet]
                                             /-
                                               🎉 no goals
                                             -/


/-- Given a fintype `α`, `finsetOrderIsoSet` is the order isomorphism between `Finset α` and `Set α`
(all sets on a finite type are finite). -/
@[simps toEquiv]
noncomputable def finsetOrderIsoSet : Finset α ≃o Set α where
  toEquiv := finsetEquivSet
  map_rel_iff' := Finset.coe_subset


@[simp, norm_cast]
lemma coe_finsetOrderIsoSet : ⇑finsetOrderIsoSet = ((↑) : Finset α → Set α) := rfl


@[simp] lemma coe_finsetOrderIsoSet_symm :
    ⇑(finsetOrderIsoSet : Finset α ≃o Set α).symm = ⇑finsetEquivSet.symm := rfl


instance Quotient.fintype [Fintype α] (s : Setoid α) [DecidableRel ((· ≈ ·) : α → α → Prop)] :
    Fintype (Quotient s) :=
  Fintype.ofSurjective Quotient.mk'' Quotient.mk''_surjective


instance PSigma.fintypePropLeft {α : Prop} {β : α → Type*} [Decidable α] [∀ a, Fintype (β a)] :
    Fintype (Σ'a, β a) :=
  if h : α then Fintype.ofEquiv (β h) ⟨fun x => ⟨h, x⟩, PSigma.snd, fun _ => rfl, fun ⟨_, _⟩ => rfl⟩
  else ⟨∅, fun x => (h x.1).elim⟩


instance PSigma.fintypePropRight {α : Type*} {β : α → Prop} [∀ a, Decidable (β a)] [Fintype α] :
    Fintype (Σ'a, β a) :=
  Fintype.ofEquiv { a // β a }
    ⟨fun ⟨x, y⟩ => ⟨x, y⟩, fun ⟨x, y⟩ => ⟨x, y⟩, fun ⟨_, _⟩ => rfl, fun ⟨_, _⟩ => rfl⟩


instance PSigma.fintypePropProp {α : Prop} {β : α → Prop} [Decidable α] [∀ a, Decidable (β a)] :
    Fintype (Σ'a, β a) :=
                                                           /-
                                                             α✝ : Type u_1
                                                             β✝ : Type u_2
                                                             γ : Type u_3
                                                             α : Prop
                                                             β : α → Prop
                                                             inst✝¹ : Decidable α
                                                             inst✝ : (a : α) → Decidable (β a)
                                                             h : Exists fun a => β a
                                                             x✝ : PSigma fun a => β a
                                                             fst✝ : α
                                                             snd✝ : β fst✝
                                                             ⊢ Membership.mem (Singleton.singleton ⟨⋯, ⋯⟩) ⟨fst✝, snd✝⟩
                                                           -/
  if h : ∃ a, β a then ⟨{⟨h.fst, h.snd⟩}, fun ⟨_, _⟩ => by simp⟩ else ⟨∅, fun ⟨x, y⟩ =>
                                                           /-
                                                             🎉 no goals
                                                           -/
    (h ⟨x, y⟩).elim⟩


instance pfunFintype (p : Prop) [Decidable p] (α : p → Type*) [∀ hp, Fintype (α hp)] :
    Fintype (∀ hp : p, α hp) :=
  if hp : p then Fintype.ofEquiv (α hp) ⟨fun a _ => a, fun f => f hp, fun _ => rfl, fun _ => rfl⟩
  else ⟨singleton fun h => (hp h).elim, fun h => mem_singleton.2
                        /-
                          α✝ : Type u_1
                          β : Type u_2
                          γ : Type u_3
                          p : Prop
                          inst✝¹ : Decidable p
                          α : p → Type u_4
                          inst✝ : (hp : p) → Fintype (α hp)
                          hp : Not p
                          h : (hp : p) → α hp
                          x : p
                          ⊢ Eq (h x) ⋯.elim
                        -/
    (funext fun x => by contradiction)⟩
                        /-
                          🎉 no goals
                        -/


theorem mem_image_univ_iff_mem_range {α β : Type*} [Fintype α] [DecidableEq β] {f : α → β}
                                                       /-
                                                         α : Type u_4
                                                         β : Type u_5
                                                         inst✝¹ : Fintype α
                                                         inst✝ : DecidableEq β
                                                         f : α → β
                                                         b : β
                                                         ⊢ Iff (Membership.mem (Finset.image f Finset.univ) b) (Membership.mem (Set.ran …
                                                       -/
    {b : β} : b ∈ univ.image f ↔ b ∈ Set.range f := by simp
                                                       /-
                                                         🎉 no goals
                                                       -/


/-- Given a fintype `α` and a predicate `p`, associate to a proof that there is a unique element of
`α` satisfying `p` this unique element, as an element of the corresponding subtype. -/
def chooseX (hp : ∃! a : α, p a) : { a // p a } :=
                            /-
                              α : Type u_1
                              β : Type u_2
                              γ : Type u_3
                              inst✝¹ : Fintype α
                              p : α → Prop
                              inst✝ : DecidablePred p
                              hp : ExistsUnique fun a => p a
                              ⊢ ExistsUnique fun a => And (Membership.mem Finset.univ a) (p a)
                            -/
  ⟨Finset.choose p univ (by simpa), Finset.choose_property _ _ _⟩
                            /-
                              🎉 no goals
                            -/


/-- Given a fintype `α` and a predicate `p`, associate to a proof that there is a unique element of
`α` satisfying `p` this unique element, as an element of `α`. -/
def choose (hp : ∃! a, p a) : α :=
  chooseX p hp


theorem choose_spec (hp : ∃! a, p a) : p (choose p hp) :=
  (chooseX p hp).property

-- @[simp] Porting note: removing simp, never applies

theorem choose_subtype_eq {α : Type*} (p : α → Prop) [Fintype { a : α // p a }] [DecidableEq α]
    (x : { a : α // p a })
    (h : ∃! a : { a // p a }, (a : α) = x :=
                              /-
                                α✝ : Type u_1
                                β : Type u_2
                                γ : Type u_3
                                inst✝³ : Fintype α✝
                                p✝ : α✝ → Prop
                                inst✝² : DecidablePred p✝
                                α : Type u_4
                                p : α → Prop
                                inst✝¹ : Fintype (Subtype fun a => p a)
                                inst✝ : DecidableEq α
                                x y : Subtype fun a => p a
                                hy : (fun a => Eq ↑a ↑x) y
                                ⊢ Eq y x
                              -/
      ⟨x, rfl, fun y hy => by simpa [Subtype.ext_iff] using hy⟩) :
                              /-
                                🎉 no goals
                              -/
    Fintype.choose (fun y : { a : α // p a } => (y : α) = x) h = x := by
  /-
    α : Type u_4
    p : α → Prop
    inst✝¹ : Fintype (Subtype fun a => p a)
    inst✝ : DecidableEq α
    x : Subtype fun a => p a
    h : optParam (ExistsUnique fun a => Eq ↑a ↑x) ⋯
    ⊢ Eq (Fintype.choose (fun y => Eq ↑y ↑x) h) x
  -/
  rw [Subtype.ext_iff, Fintype.choose_spec (fun y : { a : α // p a } => (y : α) = x) _]
  /-
    🎉 no goals
  -/


/-- `bijInv f` is the unique inverse to a bijection `f`. This acts
  as a computable alternative to `Function.invFun`. -/
def bijInv (f_bij : Bijective f) (b : β) : α :=
  Fintype.choose (fun a => f a = b)
    (by
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        inst✝¹ : Fintype α
        inst✝ : DecidableEq β
        f : α → β
        f_bij : Function.Bijective f
        b : β
        ⊢ ExistsUnique fun a => (fun a => Eq (f a) b) a
      -/
      rcases f_bij.right b with ⟨a', fa_eq_b⟩
      /-
        case intro
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        inst✝¹ : Fintype α
        inst✝ : DecidableEq β
        f : α → β
        f_bij : Function.Bijective f
        b : β
        a' : α
        fa_eq_b : Eq (f a') b
        ⊢ ExistsUnique fun a => (fun a => Eq (f a) b) a
      -/
      rw [← fa_eq_b]
      /-
        case intro
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        inst✝¹ : Fintype α
        inst✝ : DecidableEq β
        f : α → β
        f_bij : Function.Bijective f
        b : β
        a' : α
        fa_eq_b : Eq (f a') b
        ⊢ ExistsUnique fun a => (fun a => Eq (f a) (f a')) a
      -/
      exact ⟨a', ⟨rfl, fun a h => f_bij.left h⟩⟩)
      /-
        🎉 no goals
      -/


theorem leftInverse_bijInv (f_bij : Bijective f) : LeftInverse (bijInv f_bij) f := fun a =>
  f_bij.left (choose_spec (fun a' => f a' = f a) _)


theorem rightInverse_bijInv (f_bij : Bijective f) : RightInverse (bijInv f_bij) f := fun b =>
  choose_spec (fun a' => f a' = b) _


theorem bijective_bijInv (f_bij : Bijective f) : Bijective (bijInv f_bij) :=
  ⟨(rightInverse_bijInv _).injective, (leftInverse_bijInv _).surjective⟩


/-- For `s : Multiset α`, we can lift the existential statement that `∃ x, x ∈ s` to a `Trunc α`.
-/
def truncOfMultisetExistsMem {α} (s : Multiset α) : (∃ x, x ∈ s) → Trunc α :=
  Quotient.recOnSubsingleton s fun l h =>
    match l, h with
                              /-
                                α✝ : Type u_1
                                β : Type u_2
                                γ : Type u_3
                                α : Type ?u.116792
                                s : Multiset α
                                l : List α
                                h : Exists fun x => Membership.mem (Quotient.mk (List.isSetoid α) l) x
                                x✝ : Exists fun x => Membership.mem (Quotient.mk (List.isSetoid α) List.nil) x
                                ⊢ False
                              -/
    | [], _ => False.elim (by tauto)
                              /-
                                🎉 no goals
                              -/
    | a :: _, _ => Trunc.mk a


/-- A `Nonempty` `Fintype` constructively contains an element.
-/
def truncOfNonemptyFintype (α) [Nonempty α] [Fintype α] : Trunc α :=
                                               /-
                                                 α✝ : Type u_1
                                                 β : Type u_2
                                                 γ : Type u_3
                                                 α : Type ?u.117637
                                                 inst✝¹ : Nonempty α
                                                 inst✝ : Fintype α
                                                 ⊢ Exists fun x => Membership.mem Finset.univ.val x
                                               -/
  truncOfMultisetExistsMem Finset.univ.val (by simp)
                                               /-
                                                 🎉 no goals
                                               -/


/-- By iterating over the elements of a fintype, we can lift an existential statement `∃ a, P a`
to `Trunc (Σ' a, P a)`, containing data.
-/
def truncSigmaOfExists {α} [Fintype α] {P : α → Prop} [DecidablePred P] (h : ∃ a, P a) :
    Trunc (Σ'a, P a) :=
  @truncOfNonemptyFintype (Σ'a, P a) ((Exists.elim h) fun a ha => ⟨⟨a, ha⟩⟩) _


@[simp]
theorem count_univ [DecidableEq α] (a : α) : count a Finset.univ.val = 1 :=
  count_eq_one_of_mem Finset.univ.nodup (Finset.mem_univ _)


@[simp]
theorem map_univ_val_equiv (e : α ≃ β) :
    map e univ.val = univ.val := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Fintype α
    inst✝ : Fintype β
    e : Equiv α β
    ⊢ Eq (Multiset.map (⇑e) Finset.univ.val) Finset.univ.val
  -/
  rw [← congr_arg Finset.val (Finset.map_univ_equiv e), Finset.map_val, Equiv.coe_toEmbedding]
  /-
    🎉 no goals
  -/


/-- For functions on finite sets, they are bijections iff they map universes into universes. -/
@[simp]
theorem bijective_iff_map_univ_eq_univ (f : α → β) :
    f.Bijective ↔ map f (Finset.univ : Finset α).val = univ.val :=
  ⟨fun bij ↦ congr_arg (·.val) (map_univ_equiv <| Equiv.ofBijective f bij),
    fun eq ↦ ⟨
      fun a₁ a₂ ↦ inj_on_of_nodup_map (eq.symm ▸ univ.nodup) _ (mem_univ a₁) _ (mem_univ a₂),
      fun b ↦ have ⟨a, _, h⟩ := mem_map.mp (eq.symm ▸ mem_univ_val b); ⟨a, h⟩⟩⟩


/-- Auxiliary definition to show `exists_seq_of_forall_finset_exists`. -/
noncomputable def seqOfForallFinsetExistsAux {α : Type*} [DecidableEq α] (P : α → Prop)
    (r : α → α → Prop) (h : ∀ s : Finset α, ∃ y, (∀ x ∈ s, P x) → P y ∧ ∀ x ∈ s, r x y) : ℕ → α
  | n =>
    Classical.choose
      (h
        (Finset.image (fun i : Fin n => seqOfForallFinsetExistsAux P r h i)
          (Finset.univ : Finset (Fin n))))


/-- Induction principle to build a sequence, by adding one point at a time satisfying a given
relation with respect to all the previously chosen points.

More precisely, Assume that, for any finite set `s`, one can find another point satisfying
some relation `r` with respect to all the points in `s`. Then one may construct a
function `f : ℕ → α` such that `r (f m) (f n)` holds whenever `m < n`.
We also ensure that all constructed points satisfy a given predicate `P`. -/
theorem exists_seq_of_forall_finset_exists {α : Type*} (P : α → Prop) (r : α → α → Prop)
    (h : ∀ s : Finset α, (∀ x ∈ s, P x) → ∃ y, P y ∧ ∀ x ∈ s, r x y) :
    ∃ f : ℕ → α, (∀ n, P (f n)) ∧ ∀ m n, m < n → r (f m) (f n) := by
  classical
    have : Nonempty α := by
      rcases h ∅ (by simp) with ⟨y, _⟩
      exact ⟨y⟩
    choose! F hF using h
    have h' : ∀ s : Finset α, ∃ y, (∀ x ∈ s, P x) → P y ∧ ∀ x ∈ s, r x y := fun s => ⟨F s, hF s⟩
    set f := seqOfForallFinsetExistsAux P r h' with hf
    have A : ∀ n : ℕ, P (f n) := by
      intro n
      induction' n using Nat.strong_induction_on with n IH
      have IH' : ∀ x : Fin n, P (f x) := fun n => IH n.1 n.2
      rw [hf, seqOfForallFinsetExistsAux]
      exact
        (Classical.choose_spec
            (h' (Finset.image (fun i : Fin n => f i) (Finset.univ : Finset (Fin n))))
            (by simp [IH'])).1
    refine ⟨f, A, fun m n hmn => ?_⟩
    conv_rhs => rw [hf]
    rw [seqOfForallFinsetExistsAux]
    apply
      (Classical.choose_spec
          (h' (Finset.image (fun i : Fin n => f i) (Finset.univ : Finset (Fin n)))) (by simp [A])).2
    exact Finset.mem_image.2 ⟨⟨m, hmn⟩, Finset.mem_univ _, rfl⟩


/-- Induction principle to build a sequence, by adding one point at a time satisfying a given
symmetric relation with respect to all the previously chosen points.

More precisely, Assume that, for any finite set `s`, one can find another point satisfying
some relation `r` with respect to all the points in `s`. Then one may construct a
function `f : ℕ → α` such that `r (f m) (f n)` holds whenever `m ≠ n`.
We also ensure that all constructed points satisfy a given predicate `P`. -/
theorem exists_seq_of_forall_finset_exists' {α : Type*} (P : α → Prop) (r : α → α → Prop)
    [IsSymm α r] (h : ∀ s : Finset α, (∀ x ∈ s, P x) → ∃ y, P y ∧ ∀ x ∈ s, r x y) :
    ∃ f : ℕ → α, (∀ n, P (f n)) ∧ Pairwise (r on f) := by
  /-
    α : Type u_4
    P : α → Prop
    r : α → α → Prop
    inst✝ : IsSymm α r
    h : ∀ (s : Finset α), (∀ (x : α), Membership.mem s x → P x) → Exists fun y =>  …
    ⊢ Exists fun f => And (∀ (n : Nat), P (f n)) (Pairwise (Function.onFun r f))
  -/
  rcases exists_seq_of_forall_finset_exists P r h with ⟨f, hf, hf'⟩
  /-
    case intro.intro
    α : Type u_4
    P : α → Prop
    r : α → α → Prop
    inst✝ : IsSymm α r
    h : ∀ (s : Finset α), (∀ (x : α), Membership.mem s x → P x) → Exists fun y =>  …
    f : Nat → α
    hf : ∀ (n : Nat), P (f n)
    hf' : ∀ (m n : Nat), LT.lt m n → r (f m) (f n)
    ⊢ Exists fun f => And (∀ (n : Nat), P (f n)) (Pairwise (Function.onFun r f))
  -/
  refine ⟨f, hf, fun m n hmn => ?_⟩
  /-
    case intro.intro
    α : Type u_4
    P : α → Prop
    r : α → α → Prop
    inst✝ : IsSymm α r
    h : ∀ (s : Finset α), (∀ (x : α), Membership.mem s x → P x) → Exists fun y =>  …
    f : Nat → α
    hf : ∀ (n : Nat), P (f n)
    hf' : ∀ (m n : Nat), LT.lt m n → r (f m) (f n)
    m n : Nat
    hmn : Ne m n
    ⊢ Function.onFun r f m n
  -/
  rcases lt_trichotomy m n with (h | rfl | h)
    /-
      case intro.intro.inl
      α : Type u_4
      P : α → Prop
      r : α → α → Prop
      inst✝ : IsSymm α r
      h✝ : ∀ (s : Finset α), (∀ (x : α), Membership.mem s x → P x) → Exists fun y => …
      f : Nat → α
      hf : ∀ (n : Nat), P (f n)
      hf' : ∀ (m n : Nat), LT.lt m n → r (f m) (f n)
      m n : Nat
      hmn : Ne m n
      h : LT.lt m n
      ⊢ Function.onFun r f m n
    -/
  · exact hf' m n h
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.inr.inl
      α : Type u_4
      P : α → Prop
      r : α → α → Prop
      inst✝ : IsSymm α r
      h : ∀ (s : Finset α), (∀ (x : α), Membership.mem s x → P x) → Exists fun y =>  …
      f : Nat → α
      hf : ∀ (n : Nat), P (f n)
      hf' : ∀ (m n : Nat), LT.lt m n → r (f m) (f n)
      m : Nat
      hmn : Ne m m
      ⊢ Function.onFun r f m m
    -/
  · exact (hmn rfl).elim
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.inr.inr
      α : Type u_4
      P : α → Prop
      r : α → α → Prop
      inst✝ : IsSymm α r
      h✝ : ∀ (s : Finset α), (∀ (x : α), Membership.mem s x → P x) → Exists fun y => …
      f : Nat → α
      hf : ∀ (n : Nat), P (f n)
      hf' : ∀ (m n : Nat), LT.lt m n → r (f m) (f n)
      m n : Nat
      hmn : Ne m n
      h : LT.lt n m
      ⊢ Function.onFun r f m n
    -/
  · unfold Function.onFun
    /-
      case intro.intro.inr.inr
      α : Type u_4
      P : α → Prop
      r : α → α → Prop
      inst✝ : IsSymm α r
      h✝ : ∀ (s : Finset α), (∀ (x : α), Membership.mem s x → P x) → Exists fun y => …
      f : Nat → α
      hf : ∀ (n : Nat), P (f n)
      hf' : ∀ (m n : Nat), LT.lt m n → r (f m) (f n)
      m n : Nat
      hmn : Ne m n
      h : LT.lt n m
      ⊢ r (f m) (f n)
    -/
    apply symm
    /-
      case intro.intro.inr.inr.a
      α : Type u_4
      P : α → Prop
      r : α → α → Prop
      inst✝ : IsSymm α r
      h✝ : ∀ (s : Finset α), (∀ (x : α), Membership.mem s x → P x) → Exists fun y => …
      f : Nat → α
      hf : ∀ (n : Nat), P (f n)
      hf' : ∀ (m n : Nat), LT.lt m n → r (f m) (f n)
      m n : Nat
      hmn : Ne m n
      h : LT.lt n m
      ⊢ r (f n) (f m)
    -/
    exact hf' n m h
    /-
      🎉 no goals
    -/


/-- `finset% t` elaborates `t` as a `Finset`.
If `t` is a `Set`, then inserts `Set.toFinset`.
Does not make use of the expected type; useful for big operators over finsets.
```
#check finset% Finset.range 2 -- Finset Nat
#check finset% (Set.univ : Set Bool) -- Finset Bool
```
-/
elab (name := finsetStx) "finset% " t:term : term => do
  let u ← mkFreshLevelMVar
  let ty ← mkFreshExprMVar (mkSort (.succ u))
  let x ← Elab.Term.elabTerm t (mkApp (.const ``Finset [u]) ty)
  let xty ← whnfR (← inferType x)
  if xty.isAppOfArity ``Set 1 then
    Elab.Term.elabAppArgs (.const ``Set.toFinset [u]) #[] #[.expr x] none false false
  else
    return x


open Lean.Elab.Term.Quotation in
/-- `quot_precheck` for the `finset%` syntax. -/
@[quot_precheck finsetStx] def precheckFinsetStx : Precheck
  | `(finset% $t) => precheck t
  | _ => Elab.throwUnsupportedSyntax

