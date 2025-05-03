/-- Preimage of `s : Finset β` under a map `f` injective on `f ⁻¹' s` as a `Finset`. -/
noncomputable def preimage (s : Finset β) (f : α → β) (hf : Set.InjOn f (f ⁻¹' ↑s)) : Finset α :=
  (s.finite_toSet.preimage hf).toFinset


@[simp]
theorem mem_preimage {f : α → β} {s : Finset β} {hf : Set.InjOn f (f ⁻¹' ↑s)} {x : α} :
    x ∈ preimage s f hf ↔ f x ∈ s :=
  Set.Finite.mem_toFinset _


@[simp, norm_cast]
theorem coe_preimage {f : α → β} (s : Finset β) (hf : Set.InjOn f (f ⁻¹' ↑s)) :
    (↑(preimage s f hf) : Set α) = f ⁻¹' ↑s :=
  Set.Finite.coe_toFinset _


@[simp]
                                                      /-
                                                        α : Type u
                                                        β : Type v
                                                        ι : Sort w
                                                        γ : Type x
                                                        f : α → β
                                                        ⊢ Set.InjOn f (Set.preimage f ↑EmptyCollection.emptyCollection)
                                                      -/
theorem preimage_empty {f : α → β} : preimage ∅ f (by simp [InjOn]) = ∅ :=
                                                      /-
                                                        🎉 no goals
                                                      -/
                           /-
                             α : Type u
                             β : Type v
                             f : α → β
                             ⊢ Eq ↑(EmptyCollection.emptyCollection.preimage f ⋯) ↑EmptyCollection.emptyCol …
                           -/
  Finset.coe_injective (by simp)
                           /-
                             🎉 no goals
                           -/


@[simp]
theorem preimage_univ {f : α → β} [Fintype α] [Fintype β] (hf) : preimage univ f hf = univ :=
                           /-
                             α : Type u
                             β : Type v
                             f : α → β
                             inst✝¹ : Fintype α
                             inst✝ : Fintype β
                             hf : Set.InjOn f (Set.preimage f ↑Finset.univ)
                             ⊢ Eq ↑(Finset.univ.preimage f hf) ↑Finset.univ
                           -/
  Finset.coe_injective (by simp)
                           /-
                             🎉 no goals
                           -/


@[simp]
theorem preimage_inter [DecidableEq α] [DecidableEq β] {f : α → β} {s t : Finset β}
    (hs : Set.InjOn f (f ⁻¹' ↑s)) (ht : Set.InjOn f (f ⁻¹' ↑t)) :
    (preimage (s ∩ t) f fun _ hx₁ _ hx₂ =>
        hs (mem_of_mem_inter_left hx₁) (mem_of_mem_inter_left hx₂)) =
      preimage s f hs ∩ preimage t f ht :=
                           /-
                             α : Type u
                             β : Type v
                             inst✝¹ : DecidableEq α
                             inst✝ : DecidableEq β
                             f : α → β
                             s t : Finset β
                             hs : Set.InjOn f (Set.preimage f ↑s)
                             ht : Set.InjOn f (Set.preimage f ↑t)
                             ⊢ Eq ↑((Inter.inter s t).preimage f ⋯) ↑(Inter.inter (s.preimage f hs) (t.prei …
                           -/
  Finset.coe_injective (by simp)
                           /-
                             🎉 no goals
                           -/


@[simp]
theorem preimage_union [DecidableEq α] [DecidableEq β] {f : α → β} {s t : Finset β} (hst) :
    preimage (s ∪ t) f hst =
      (preimage s f fun _ hx₁ _ hx₂ => hst (mem_union_left _ hx₁) (mem_union_left _ hx₂)) ∪
        preimage t f fun _ hx₁ _ hx₂ => hst (mem_union_right _ hx₁) (mem_union_right _ hx₂) :=
                           /-
                             α : Type u
                             β : Type v
                             inst✝¹ : DecidableEq α
                             inst✝ : DecidableEq β
                             f : α → β
                             s t : Finset β
                             hst : Set.InjOn f (Set.preimage f ↑(Union.union s t))
                             ⊢ Eq ↑((Union.union s t).preimage f hst) ↑(Union.union (s.preimage f ⋯) (t.pre …
                           -/
  Finset.coe_injective (by simp)
                           /-
                             🎉 no goals
                           -/


@[simp, nolint simpNF] -- Porting note: linter complains that LHS doesn't simplify
theorem preimage_compl [DecidableEq α] [DecidableEq β] [Fintype α] [Fintype β] {f : α → β}
    (s : Finset β) (hf : Function.Injective f) :
    preimage sᶜ f hf.injOn = (preimage s f hf.injOn)ᶜ :=
                           /-
                             α : Type u
                             β : Type v
                             inst✝³ : DecidableEq α
                             inst✝² : DecidableEq β
                             inst✝¹ : Fintype α
                             inst✝ : Fintype β
                             f : α → β
                             s : Finset β
                             hf : Function.Injective f
                             ⊢ Eq ↑((HasCompl.compl s).preimage f ⋯) ↑(HasCompl.compl (s.preimage f ⋯))
                           -/
  Finset.coe_injective (by simp)
                           /-
                             🎉 no goals
                           -/


@[simp]
lemma preimage_map (f : α ↪ β) (s : Finset α) : (s.map f).preimage f f.injective.injOn = s :=
                      /-
                        α : Type u
                        β : Type v
                        f : Function.Embedding α β
                        s : Finset α
                        ⊢ Eq ↑((Finset.map f s).preimage ⇑f ⋯) ↑s
                      -/
  coe_injective <| by simp only [coe_preimage, coe_map, Set.preimage_image_eq _ f.injective]
                      /-
                        🎉 no goals
                      -/


theorem monotone_preimage {f : α → β} (h : Injective f) :
    Monotone fun s => preimage s f h.injOn := fun _ _ H _ hx =>
  mem_preimage.2 (H <| mem_preimage.1 hx)


theorem image_subset_iff_subset_preimage [DecidableEq β] {f : α → β} {s : Finset α} {t : Finset β}
    (hf : Set.InjOn f (f ⁻¹' ↑t)) : s.image f ⊆ t ↔ s ⊆ t.preimage f hf :=
                               /-
                                 α : Type u
                                 β : Type v
                                 inst✝ : DecidableEq β
                                 f : α → β
                                 s : Finset α
                                 t : Finset β
                                 hf : Set.InjOn f (Set.preimage f ↑t)
                                 ⊢ Iff (∀ (x : α), Membership.mem s x → Membership.mem t (f x)) (HasSubset.Subs …
                               -/
  image_subset_iff.trans <| by simp only [subset_iff, mem_preimage]
                               /-
                                 🎉 no goals
                               -/


theorem map_subset_iff_subset_preimage {f : α ↪ β} {s : Finset α} {t : Finset β} :
    s.map f ⊆ t ↔ s ⊆ t.preimage f f.injective.injOn := by
  /-
    α : Type u
    β : Type v
    f : Function.Embedding α β
    s : Finset α
    t : Finset β
    ⊢ Iff (HasSubset.Subset (Finset.map f s) t) (HasSubset.Subset s (t.preimage ⇑f …
  -/
  classical rw [map_eq_image, image_subset_iff_subset_preimage]
  /-
    🎉 no goals
  -/


lemma card_preimage (s : Finset β) (f : α → β) (hf) [DecidablePred (· ∈ Set.range f)] :
    (s.preimage f hf).card = {x ∈ s | x ∈ Set.range f}.card :=
                  /-
                    α : Type u
                    β : Type v
                    s : Finset β
                    f : α → β
                    hf : Set.InjOn f (Set.preimage f ↑s)
                    inst✝ : DecidablePred fun x => Membership.mem (Set.range f) x
                    ⊢ ∀ (a : α), Membership.mem (s.preimage f hf) a → Membership.mem (Finset.filte …
                  -/
                  /-
                    🎉 no goals
                  -/
                            /-
                              🎉 no goals
                            -/
  card_nbij f (by simp) (by simpa) (fun b hb ↦ by aesop)
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem image_preimage [DecidableEq β] (f : α → β) (s : Finset β) [∀ x, Decidable (x ∈ Set.range f)]
    (hf : Set.InjOn f (f ⁻¹' ↑s)) : image f (preimage s f hf) = s.filter fun x => x ∈ Set.range f :=
  Finset.coe_inj.1 <| by
    simp only [coe_image, coe_preimage, coe_filter, Set.image_preimage_eq_inter_range,
                         /-
                           α : Type u
                           β : Type v
                           inst✝¹ : DecidableEq β
                           f : α → β
                           s : Finset β
                           inst✝ : (x : β) → Decidable (Membership.mem (Set.range f) x)
                           hf : Set.InjOn f (Set.preimage f ↑s)
                           ⊢ Eq (setOf fun x => And (Membership.mem (↑s) x) (Membership.mem (Set.range f) …
                         -/
      ← Set.sep_mem_eq]; rfl
                         /-
                           🎉 no goals
                         -/


theorem image_preimage_of_bij [DecidableEq β] (f : α → β) (s : Finset β)
    (hf : Set.BijOn f (f ⁻¹' ↑s) ↑s) : image f (preimage s f hf.injOn) = s :=
                         /-
                           α : Type u
                           β : Type v
                           inst✝ : DecidableEq β
                           f : α → β
                           s : Finset β
                           hf : Set.BijOn f (Set.preimage f ↑s) ↑s
                           ⊢ Eq ↑(Finset.image f (s.preimage f ⋯)) ↑s
                         -/
  Finset.coe_inj.1 <| by simpa using hf.image_eq
                         /-
                           🎉 no goals
                         -/


lemma preimage_subset_of_subset_image [DecidableEq β] {f : α → β} {s : Finset β} {t : Finset α}
    (hs : s ⊆ t.image f) {hf} : s.preimage f hf ⊆ t := by
  /-
    α : Type u
    β : Type v
    inst✝ : DecidableEq β
    f : α → β
    s : Finset β
    t : Finset α
    hs : HasSubset.Subset s (Finset.image f t)
    hf : Set.InjOn f (Set.preimage f ↑s)
    ⊢ HasSubset.Subset (s.preimage f hf) t
  -/
  rw [← coe_subset, coe_preimage]; exact Set.preimage_subset (mod_cast hs) hf
                                   /-
                                     🎉 no goals
                                   -/


theorem preimage_subset {f : α ↪ β} {s : Finset β} {t : Finset α} (hs : s ⊆ t.map f) :
    s.preimage f f.injective.injOn ⊆ t := fun _ h => (mem_map' f).1 (hs (mem_preimage.1 h))


theorem subset_map_iff {f : α ↪ β} {s : Finset β} {t : Finset α} :
    s ⊆ t.map f ↔ ∃ u ⊆ t, s = u.map f := by
  classical
  simp_rw [map_eq_image, subset_image_iff, eq_comm]


theorem sigma_preimage_mk {β : α → Type*} [DecidableEq α] (s : Finset (Σa, β a)) (t : Finset α) :
    (t.sigma fun a => s.preimage (Sigma.mk a) sigma_mk_injective.injOn) =
      s.filter fun a => a.1 ∈ t := by
  /-
    α : Type u
    β : α → Type u_1
    inst✝ : DecidableEq α
    s : Finset (Sigma fun a => β a)
    t : Finset α
    ⊢ Eq (t.sigma fun a => s.preimage (Sigma.mk a) ⋯) (Finset.filter (fun a => Mem …
  -/
  ext x
  /-
    case h
    α : Type u
    β : α → Type u_1
    inst✝ : DecidableEq α
    s : Finset (Sigma fun a => β a)
    t : Finset α
    x : Sigma fun i => β i
    ⊢ Iff (Membership.mem (t.sigma fun a => s.preimage (Sigma.mk a) ⋯) x) (Members …
  -/
  simp [and_comm]
  /-
    🎉 no goals
  -/


theorem sigma_preimage_mk_of_subset {β : α → Type*} [DecidableEq α] (s : Finset (Σa, β a))
    {t : Finset α} (ht : s.image Sigma.fst ⊆ t) :
    (t.sigma fun a => s.preimage (Sigma.mk a) sigma_mk_injective.injOn) = s := by
  /-
    α : Type u
    β : α → Type u_1
    inst✝ : DecidableEq α
    s : Finset (Sigma fun a => β a)
    t : Finset α
    ht : HasSubset.Subset (Finset.image Sigma.fst s) t
    ⊢ Eq (t.sigma fun a => s.preimage (Sigma.mk a) ⋯) s
  -/
  rw [sigma_preimage_mk, filter_true_of_mem <| image_subset_iff.1 ht]
  /-
    🎉 no goals
  -/


theorem sigma_image_fst_preimage_mk {β : α → Type*} [DecidableEq α] (s : Finset (Σa, β a)) :
    ((s.image Sigma.fst).sigma fun a => s.preimage (Sigma.mk a) sigma_mk_injective.injOn) =
      s :=
  s.sigma_preimage_mk_of_subset (Subset.refl _)


@[simp] lemma preimage_inl (s : Finset (α ⊕ β)) :
    s.preimage Sum.inl Sum.inl_injective.injOn = s.toLeft := by
  /-
    α : Type u
    β : Type v
    s : Finset (Sum α β)
    ⊢ Eq (s.preimage Sum.inl ⋯) s.toLeft
  -/
  ext x; simp
         /-
           🎉 no goals
         -/


@[simp] lemma preimage_inr (s : Finset (α ⊕ β)) :
    s.preimage Sum.inr Sum.inr_injective.injOn = s.toRight := by
  /-
    α : Type u
    β : Type v
    s : Finset (Sum α β)
    ⊢ Eq (s.preimage Sum.inr ⋯) s.toRight
  -/
  ext x; simp
         /-
           🎉 no goals
         -/


