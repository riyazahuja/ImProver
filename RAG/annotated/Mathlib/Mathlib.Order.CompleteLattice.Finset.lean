/-- Supremum of `s i`, `i : ι`, is equal to the supremum over `t : Finset ι` of suprema
`⨆ i ∈ t, s i`. This version assumes `ι` is a `Type*`. See `iSup_eq_iSup_finset'` for a version
that works for `ι : Sort*`. -/
theorem iSup_eq_iSup_finset (s : ι → α) : ⨆ i, s i = ⨆ t : Finset ι, ⨆ i ∈ t, s i := by
  classical
  refine le_antisymm ?_ ?_
  · exact iSup_le fun b => le_iSup_of_le {b} <| le_iSup_of_le b <| le_iSup_of_le (by simp) <| le_rfl
  · exact iSup_le fun t => iSup_le fun b => iSup_le fun _ => le_iSup _ _


/-- Supremum of `s i`, `i : ι`, is equal to the supremum over `t : Finset ι` of suprema
`⨆ i ∈ t, s i`. This version works for `ι : Sort*`. See `iSup_eq_iSup_finset` for a version
that assumes `ι : Type*` but has no `PLift`s. -/
theorem iSup_eq_iSup_finset' (s : ι' → α) :
    ⨆ i, s i = ⨆ t : Finset (PLift ι'), ⨆ i ∈ t, s (PLift.down i) := by
  /-
    α : Type u_2
    ι' : Sort u_7
    inst✝ : CompleteLattice α
    s : ι' → α
    ⊢ Eq (iSup fun i => s i) (iSup fun t => iSup fun i => iSup fun h => s i.down)
  -/
  rw [← iSup_eq_iSup_finset, ← Equiv.plift.surjective.iSup_comp]; rfl
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


/-- Infimum of `s i`, `i : ι`, is equal to the infimum over `t : Finset ι` of infima
`⨅ i ∈ t, s i`. This version assumes `ι` is a `Type*`. See `iInf_eq_iInf_finset'` for a version
that works for `ι : Sort*`. -/
theorem iInf_eq_iInf_finset (s : ι → α) : ⨅ i, s i = ⨅ (t : Finset ι) (i ∈ t), s i :=
  @iSup_eq_iSup_finset αᵒᵈ _ _ _


/-- Infimum of `s i`, `i : ι`, is equal to the infimum over `t : Finset ι` of infima
`⨅ i ∈ t, s i`. This version works for `ι : Sort*`. See `iInf_eq_iInf_finset` for a version
that assumes `ι : Type*` but has no `PLift`s. -/
theorem iInf_eq_iInf_finset' (s : ι' → α) :
    ⨅ i, s i = ⨅ t : Finset (PLift ι'), ⨅ i ∈ t, s (PLift.down i) :=
  @iSup_eq_iSup_finset' αᵒᵈ _ _ _


/-- Union of an indexed family of sets `s : ι → Set α` is equal to the union of the unions
of finite subfamilies. This version assumes `ι : Type*`. See also `iUnion_eq_iUnion_finset'` for
a version that works for `ι : Sort*`. -/
theorem iUnion_eq_iUnion_finset (s : ι → Set α) : ⋃ i, s i = ⋃ t : Finset ι, ⋃ i ∈ t, s i :=
  iSup_eq_iSup_finset s


/-- Union of an indexed family of sets `s : ι → Set α` is equal to the union of the unions
of finite subfamilies. This version works for `ι : Sort*`. See also `iUnion_eq_iUnion_finset` for
a version that assumes `ι : Type*` but avoids `PLift`s in the right hand side. -/
theorem iUnion_eq_iUnion_finset' (s : ι' → Set α) :
    ⋃ i, s i = ⋃ t : Finset (PLift ι'), ⋃ i ∈ t, s (PLift.down i) :=
  iSup_eq_iSup_finset' s


/-- Intersection of an indexed family of sets `s : ι → Set α` is equal to the intersection of the
intersections of finite subfamilies. This version assumes `ι : Type*`. See also
`iInter_eq_iInter_finset'` for a version that works for `ι : Sort*`. -/
theorem iInter_eq_iInter_finset (s : ι → Set α) : ⋂ i, s i = ⋂ t : Finset ι, ⋂ i ∈ t, s i :=
  iInf_eq_iInf_finset s


/-- Intersection of an indexed family of sets `s : ι → Set α` is equal to the intersection of the
intersections of finite subfamilies. This version works for `ι : Sort*`. See also
`iInter_eq_iInter_finset` for a version that assumes `ι : Type*` but avoids `PLift`s in the right
hand side. -/
theorem iInter_eq_iInter_finset' (s : ι' → Set α) :
    ⋂ i, s i = ⋂ t : Finset (PLift ι'), ⋂ i ∈ t, s (PLift.down i) :=
  iInf_eq_iInf_finset' s


theorem maximal_iff_forall_insert (hP : ∀ ⦃s t⦄, P t → s ⊆ t → P s) :
    Maximal P s ↔ P s ∧ ∀ x ∉ s, ¬ P (insert x s) := by
  /-
    α : Type u_2
    inst✝ : DecidableEq α
    P : Finset α → Prop
    s : Finset α
    hP : ∀ ⦃s t : Finset α⦄, P t → HasSubset.Subset s t → P s
    ⊢ Iff (Maximal P s) (And (P s) (∀ (x : α), Not (Membership.mem s x) → Not (P ( …
  -/
  simp only [Maximal, and_congr_right_iff]
  exact fun _ ↦ ⟨fun h x hxs hx ↦ hxs <| h hx (subset_insert _ _) (mem_insert_self x s),
    fun h t ht hst x hxt ↦ by_contra fun hxs ↦ h x hxs (hP ht (insert_subset hxt hst))⟩


theorem minimal_iff_forall_diff_singleton (hP : ∀ ⦃s t⦄, P t → t ⊆ s → P s) :
    Minimal P s ↔ P s ∧ ∀ x ∈ s, ¬ P (s.erase x) where
                                     /-
                                       α : Type u_2
                                       inst✝ : DecidableEq α
                                       P : Finset α → Prop
                                       s : Finset α
                                       hP : ∀ ⦃s t : Finset α⦄, P t → HasSubset.Subset t s → P s
                                       h : Minimal P s
                                       x : α
                                       hxs : Membership.mem s x
                                       hx : P (s.erase x)
                                       ⊢ False
                                     -/
  mp h := ⟨h.prop, fun x hxs hx ↦ by simpa using h.le_of_le hx (erase_subset _ _) hxs⟩
                                     /-
                                       🎉 no goals
                                     -/
  mpr h := ⟨h.1, fun t ht hts x hxs ↦ by_contra fun hxt ↦
    h.2 x hxs <| hP ht (subset_erase.2 ⟨hts, hxt⟩)⟩


theorem iSup_coe [SupSet β] (f : α → β) (s : Finset α) : ⨆ x ∈ (↑s : Set α), f x = ⨆ x ∈ s, f x :=
  rfl


theorem iInf_coe [InfSet β] (f : α → β) (s : Finset α) : ⨅ x ∈ (↑s : Set α), f x = ⨅ x ∈ s, f x :=
  rfl


                                                                                     /-
                                                                                       α : Type u_2
                                                                                       β : Type u_3
                                                                                       inst✝ : CompleteLattice β
                                                                                       a : α
                                                                                       s : α → β
                                                                                       ⊢ Eq (iSup fun x => iSup fun h => s x) (s a)
                                                                                     -/
theorem iSup_singleton (a : α) (s : α → β) : ⨆ x ∈ ({a} : Finset α), s x = s a := by simp
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


                                                                                     /-
                                                                                       α : Type u_2
                                                                                       β : Type u_3
                                                                                       inst✝ : CompleteLattice β
                                                                                       a : α
                                                                                       s : α → β
                                                                                       ⊢ Eq (iInf fun x => iInf fun h => s x) (s a)
                                                                                     -/
theorem iInf_singleton (a : α) (s : α → β) : ⨅ x ∈ ({a} : Finset α), s x = s a := by simp
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


theorem iSup_option_toFinset (o : Option α) (f : α → β) : ⨆ x ∈ o.toFinset, f x = ⨆ x ∈ o, f x := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝ : CompleteLattice β
    o : Option α
    f : α → β
    ⊢ Eq (iSup fun x => iSup fun h => f x) (iSup fun x => iSup fun h => f x)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem iInf_option_toFinset (o : Option α) (f : α → β) : ⨅ x ∈ o.toFinset, f x = ⨅ x ∈ o, f x :=
  @iSup_option_toFinset _ βᵒᵈ _ _ _


theorem iSup_union {f : α → β} {s t : Finset α} :
                                                           /-
                                                             α : Type u_2
                                                             β : Type u_3
                                                             inst✝¹ : CompleteLattice β
                                                             inst✝ : DecidableEq α
                                                             f : α → β
                                                             s t : Finset α
                                                             ⊢ Eq (iSup fun x => iSup fun h => f x) (Max.max (iSup fun x => iSup fun h => f …
                                                           -/
    ⨆ x ∈ s ∪ t, f x = (⨆ x ∈ s, f x) ⊔ ⨆ x ∈ t, f x := by simp [iSup_or, iSup_sup_eq]
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem iInf_union {f : α → β} {s t : Finset α} :
    ⨅ x ∈ s ∪ t, f x = (⨅ x ∈ s, f x) ⊓ ⨅ x ∈ t, f x :=
  @iSup_union α βᵒᵈ _ _ _ _ _


theorem iSup_insert (a : α) (s : Finset α) (t : α → β) :
    ⨆ x ∈ insert a s, t x = t a ⊔ ⨆ x ∈ s, t x := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : CompleteLattice β
    inst✝ : DecidableEq α
    a : α
    s : Finset α
    t : α → β
    ⊢ Eq (iSup fun x => iSup fun h => t x) (Max.max (t a) (iSup fun x => iSup fun  …
  -/
  rw [insert_eq]
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : CompleteLattice β
    inst✝ : DecidableEq α
    a : α
    s : Finset α
    t : α → β
    ⊢ Eq (iSup fun x => iSup fun h => t x) (Max.max (t a) (iSup fun x => iSup fun  …
  -/
  simp only [iSup_union, Finset.iSup_singleton]
  /-
    🎉 no goals
  -/


theorem iInf_insert (a : α) (s : Finset α) (t : α → β) :
    ⨅ x ∈ insert a s, t x = t a ⊓ ⨅ x ∈ s, t x :=
  @iSup_insert α βᵒᵈ _ _ _ _ _


theorem iSup_finset_image {f : γ → α} {g : α → β} {s : Finset γ} :
                                                  /-
                                                    α : Type u_2
                                                    β : Type u_3
                                                    γ : Type u_4
                                                    inst✝¹ : CompleteLattice β
                                                    inst✝ : DecidableEq α
                                                    f : γ → α
                                                    g : α → β
                                                    s : Finset γ
                                                    ⊢ Eq (iSup fun x => iSup fun h => g x) (iSup fun y => iSup fun h => g (f y))
                                                  -/
    ⨆ x ∈ s.image f, g x = ⨆ y ∈ s, g (f y) := by rw [← iSup_coe, coe_image, iSup_image, iSup_coe]
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem iInf_finset_image {f : γ → α} {g : α → β} {s : Finset γ} :
                                                  /-
                                                    α : Type u_2
                                                    β : Type u_3
                                                    γ : Type u_4
                                                    inst✝¹ : CompleteLattice β
                                                    inst✝ : DecidableEq α
                                                    f : γ → α
                                                    g : α → β
                                                    s : Finset γ
                                                    ⊢ Eq (iInf fun x => iInf fun h => g x) (iInf fun y => iInf fun h => g (f y))
                                                  -/
    ⨅ x ∈ s.image f, g x = ⨅ y ∈ s, g (f y) := by rw [← iInf_coe, coe_image, iInf_image, iInf_coe]
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem iSup_insert_update {x : α} {t : Finset α} (f : α → β) {s : β} (hx : x ∉ t) :
    ⨆ i ∈ insert x t, Function.update f x s i = s ⊔ ⨆ i ∈ t, f i := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : CompleteLattice β
    inst✝ : DecidableEq α
    x : α
    t : Finset α
    f : α → β
    s : β
    hx : Not (Membership.mem t x)
    ⊢ Eq (iSup fun i => iSup fun h => Function.update f x s i) (Max.max s (iSup fu …
  -/
  simp only [Finset.iSup_insert, update_self]
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : CompleteLattice β
    inst✝ : DecidableEq α
    x : α
    t : Finset α
    f : α → β
    s : β
    hx : Not (Membership.mem t x)
    ⊢ Eq (Max.max s (iSup fun x_1 => iSup fun h => Function.update f x s x_1)) (Ma …
  -/
  rcongr (i hi); apply update_of_ne; rintro rfl; exact hx hi
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem iInf_insert_update {x : α} {t : Finset α} (f : α → β) {s : β} (hx : x ∉ t) :
    ⨅ i ∈ insert x t, update f x s i = s ⊓ ⨅ i ∈ t, f i :=
  @iSup_insert_update α βᵒᵈ _ _ _ _ f _ hx


theorem iSup_biUnion (s : Finset γ) (t : γ → Finset α) (f : α → β) :
                                                            /-
                                                              α : Type u_2
                                                              β : Type u_3
                                                              γ : Type u_4
                                                              inst✝¹ : CompleteLattice β
                                                              inst✝ : DecidableEq α
                                                              s : Finset γ
                                                              t : γ → Finset α
                                                              f : α → β
                                                              ⊢ Eq (iSup fun y => iSup fun h => f y) (iSup fun x => iSup fun h => iSup fun y …
                                                            -/
    ⨆ y ∈ s.biUnion t, f y = ⨆ (x ∈ s) (y ∈ t x), f y := by simp [@iSup_comm _ α, iSup_and]
                                                            /-
                                                              🎉 no goals
                                                            -/


theorem iInf_biUnion (s : Finset γ) (t : γ → Finset α) (f : α → β) :
    ⨅ y ∈ s.biUnion t, f y = ⨅ (x ∈ s) (y ∈ t x), f y :=
  @iSup_biUnion _ βᵒᵈ _ _ _ _ _ _


theorem set_biUnion_coe (s : Finset α) (t : α → Set β) : ⋃ x ∈ (↑s : Set α), t x = ⋃ x ∈ s, t x :=
  rfl


theorem set_biInter_coe (s : Finset α) (t : α → Set β) : ⋂ x ∈ (↑s : Set α), t x = ⋂ x ∈ s, t x :=
  rfl


theorem set_biUnion_singleton (a : α) (s : α → Set β) : ⋃ x ∈ ({a} : Finset α), s x = s a :=
  iSup_singleton a s


theorem set_biInter_singleton (a : α) (s : α → Set β) : ⋂ x ∈ ({a} : Finset α), s x = s a :=
  iInf_singleton a s


@[simp]
theorem set_biUnion_preimage_singleton (f : α → β) (s : Finset β) :
    ⋃ y ∈ s, f ⁻¹' {y} = f ⁻¹' s :=
  Set.biUnion_preimage_singleton f s


theorem set_biUnion_option_toFinset (o : Option α) (f : α → Set β) :
    ⋃ x ∈ o.toFinset, f x = ⋃ x ∈ o, f x :=
  iSup_option_toFinset o f


theorem set_biInter_option_toFinset (o : Option α) (f : α → Set β) :
    ⋂ x ∈ o.toFinset, f x = ⋂ x ∈ o, f x :=
  iInf_option_toFinset o f


theorem subset_set_biUnion_of_mem {s : Finset α} {f : α → Set β} {x : α} (h : x ∈ s) :
    f x ⊆ ⋃ y ∈ s, f y :=
                                                     /-
                                                       α : Type u_2
                                                       β : Type u_3
                                                       s : Finset α
                                                       f : α → Set β
                                                       x : α
                                                       h : Membership.mem s x
                                                       ⊢ LE.le (f x) (iSup fun h => f x)
                                                     -/
  show f x ≤ ⨆ y ∈ s, f y from le_iSup_of_le x <| by simp only [h, iSup_pos, le_refl]
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem set_biUnion_union (s t : Finset α) (u : α → Set β) :
    ⋃ x ∈ s ∪ t, u x = (⋃ x ∈ s, u x) ∪ ⋃ x ∈ t, u x :=
  iSup_union


theorem set_biInter_inter (s t : Finset α) (u : α → Set β) :
    ⋂ x ∈ s ∪ t, u x = (⋂ x ∈ s, u x) ∩ ⋂ x ∈ t, u x :=
  iInf_union


theorem set_biUnion_insert (a : α) (s : Finset α) (t : α → Set β) :
    ⋃ x ∈ insert a s, t x = t a ∪ ⋃ x ∈ s, t x :=
  iSup_insert a s t


theorem set_biInter_insert (a : α) (s : Finset α) (t : α → Set β) :
    ⋂ x ∈ insert a s, t x = t a ∩ ⋂ x ∈ s, t x :=
  iInf_insert a s t


theorem set_biUnion_finset_image {f : γ → α} {g : α → Set β} {s : Finset γ} :
    ⋃ x ∈ s.image f, g x = ⋃ y ∈ s, g (f y) :=
  iSup_finset_image


theorem set_biInter_finset_image {f : γ → α} {g : α → Set β} {s : Finset γ} :
    ⋂ x ∈ s.image f, g x = ⋂ y ∈ s, g (f y) :=
  iInf_finset_image


theorem set_biUnion_insert_update {x : α} {t : Finset α} (f : α → Set β) {s : Set β} (hx : x ∉ t) :
    ⋃ i ∈ insert x t, @update _ _ _ f x s i = s ∪ ⋃ i ∈ t, f i :=
  iSup_insert_update f hx


theorem set_biInter_insert_update {x : α} {t : Finset α} (f : α → Set β) {s : Set β} (hx : x ∉ t) :
    ⋂ i ∈ insert x t, @update _ _ _ f x s i = s ∩ ⋂ i ∈ t, f i :=
  iInf_insert_update f hx


theorem set_biUnion_biUnion (s : Finset γ) (t : γ → Finset α) (f : α → Set β) :
    ⋃ y ∈ s.biUnion t, f y = ⋃ (x ∈ s) (y ∈ t x), f y :=
  iSup_biUnion s t f


theorem set_biInter_biUnion (s : Finset γ) (t : γ → Finset α) (f : α → Set β) :
    ⋂ y ∈ s.biUnion t, f y = ⋂ (x ∈ s) (y ∈ t x), f y :=
  iInf_biUnion s t f


