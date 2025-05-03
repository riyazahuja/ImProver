/-- Restrict domain of a function `f` to a set `s`. Same as `Subtype.restrict` but this version
takes an argument `↥s` instead of `Subtype s`. -/
def restrict (s : Set α) (f : ∀ a : α, π a) : ∀ a : s, π a := fun x => f x


theorem restrict_def (s : Set α) : s.restrict (π := π) = fun f x ↦ f x := rfl


theorem restrict_eq (f : α → β) (s : Set α) : s.restrict f = f ∘ Subtype.val :=
  rfl


@[simp]
theorem restrict_apply (f : (a : α) → π a) (s : Set α) (x : s) : s.restrict f x = f x :=
  rfl


theorem restrict_eq_iff {f : ∀ a, π a} {s : Set α} {g : ∀ a : s, π a} :
    restrict s f = g ↔ ∀ (a) (ha : a ∈ s), f a = g ⟨a, ha⟩ :=
  funext_iff.trans Subtype.forall


theorem eq_restrict_iff {s : Set α} {f : ∀ a : s, π a} {g : ∀ a, π a} :
    f = restrict s g ↔ ∀ (a) (ha : a ∈ s), f ⟨a, ha⟩ = g a :=
  funext_iff.trans Subtype.forall


@[simp]
theorem range_restrict (f : α → β) (s : Set α) : Set.range (s.restrict f) = f '' s :=
  (range_comp _ _).trans <| congr_arg (f '' ·) Subtype.range_coe


theorem image_restrict (f : α → β) (s t : Set α) :
    s.restrict f '' (Subtype.val ⁻¹' t) = f '' (t ∩ s) := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    s t : Set α
    ⊢ Eq (Set.image (s.restrict f) (Set.preimage Subtype.val t)) (Set.image f (Int …
  -/
  rw [restrict_eq, image_comp, image_preimage_eq_inter_range, Subtype.range_coe]
  /-
    🎉 no goals
  -/


@[simp]
theorem restrict_dite {s : Set α} [∀ x, Decidable (x ∈ s)] (f : ∀ a ∈ s, β)
    (g : ∀ a ∉ s, β) :
    (s.restrict fun a => if h : a ∈ s then f a h else g a h) = (fun a : s => f a a.2) :=
  funext fun a => dif_pos a.2


@[simp]
theorem restrict_dite_compl {s : Set α} [∀ x, Decidable (x ∈ s)] (f : ∀ a ∈ s, β)
    (g : ∀ a ∉ s, β) :
    (sᶜ.restrict fun a => if h : a ∈ s then f a h else g a h) = (fun a : (sᶜ : Set α) => g a a.2) :=
  funext fun a => dif_neg a.2


@[simp]
theorem restrict_ite (f g : α → β) (s : Set α) [∀ x, Decidable (x ∈ s)] :
    (s.restrict fun a => if a ∈ s then f a else g a) = s.restrict f :=
  restrict_dite _ _


@[simp]
theorem restrict_ite_compl (f g : α → β) (s : Set α) [∀ x, Decidable (x ∈ s)] :
    (sᶜ.restrict fun a => if a ∈ s then f a else g a) = sᶜ.restrict g :=
  restrict_dite_compl _ _


@[simp]
theorem restrict_piecewise (f g : α → β) (s : Set α) [∀ x, Decidable (x ∈ s)] :
    s.restrict (piecewise s f g) = s.restrict f :=
  restrict_ite _ _ _


@[simp]
theorem restrict_piecewise_compl (f g : α → β) (s : Set α) [∀ x, Decidable (x ∈ s)] :
    sᶜ.restrict (piecewise s f g) = sᶜ.restrict g :=
  restrict_ite_compl _ _ _


theorem restrict_extend_range (f : α → β) (g : α → γ) (g' : β → γ) :
    (range f).restrict (extend f g g') = fun x => g x.coe_prop.choose := by
  classical
  exact restrict_dite _ _


@[simp]
theorem restrict_extend_compl_range (f : α → β) (g : α → γ) (g' : β → γ) :
    (range f)ᶜ.restrict (extend f g g') = g' ∘ Subtype.val := by
  classical
  exact restrict_dite_compl _ _


/-- If a function `f` is restricted to a set `t`, and `s ⊆ t`, this is the restriction to `s`. -/
@[simp]
def restrict₂ {s t : Set α} (hst : s ⊆ t) (f : ∀ a : t, π a) : ∀ a : s, π a :=
  fun x => f ⟨x.1, hst x.2⟩


theorem restrict₂_def {s t : Set α} (hst : s ⊆ t) :
    restrict₂ (π := π) hst = fun f x ↦ f ⟨x.1, hst x.2⟩ := rfl


theorem restrict₂_comp_restrict {s t : Set α} (hst : s ⊆ t) :
    (restrict₂ (π := π) hst) ∘ t.restrict = s.restrict := rfl


theorem restrict₂_comp_restrict₂ {s t u : Set α} (hst : s ⊆ t) (htu : t ⊆ u) :
    (restrict₂ (π := π) hst) ∘ (restrict₂ htu) = restrict₂ (hst.trans htu) := rfl


theorem range_extend_subset (f : α → β) (g : α → γ) (g' : β → γ) :
    range (extend f g g') ⊆ range g ∪ g' '' (range f)ᶜ := by
  classical
  rintro _ ⟨y, rfl⟩
  rw [extend_def]
  split_ifs with h
  exacts [Or.inl (mem_range_self _), Or.inr (mem_image_of_mem _ h)]


theorem range_extend {f : α → β} (hf : Injective f) (g : α → γ) (g' : β → γ) :
    range (extend f g g') = range g ∪ g' '' (range f)ᶜ := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → β
    hf : Function.Injective f
    g : α → γ
    g' : β → γ
    ⊢ Eq (Set.range (Function.extend f g g')) (Union.union (Set.range g) (Set.imag …
  -/
  refine (range_extend_subset _ _ _).antisymm ?_
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → β
    hf : Function.Injective f
    g : α → γ
    g' : β → γ
    ⊢ HasSubset.Subset (Union.union (Set.range g) (Set.image g' (HasCompl.compl (S …
  -/
  rintro z (⟨x, rfl⟩ | ⟨y, hy, rfl⟩)
  /-
    case inl.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → β
    hf : Function.Injective f
    g : α → γ
    g' : β → γ
    x : α
    ⊢ Membership.mem (Set.range (Function.extend f g g')) (g x)
  -/
  exacts [⟨f x, hf.extend_apply _ _ _⟩, ⟨y, extend_apply' _ _ _ hy⟩]
  /-
    🎉 no goals
  -/


/-- Restrict codomain of a function `f` to a set `s`. Same as `Subtype.coind` but this version
has codomain `↥s` instead of `Subtype s`. -/
def codRestrict (f : ι → α) (s : Set α) (h : ∀ x, f x ∈ s) : ι → s := fun x => ⟨f x, h x⟩


@[simp]
theorem val_codRestrict_apply (f : ι → α) (s : Set α) (h : ∀ x, f x ∈ s) (x : ι) :
    (codRestrict f s h x : α) = f x :=
  rfl


@[simp]
theorem restrict_comp_codRestrict {f : ι → α} {g : α → β} {b : Set α} (h : ∀ x, f x ∈ b) :
    b.restrict g ∘ b.codRestrict f h = g ∘ f :=
  rfl


@[simp]
theorem injective_codRestrict {f : ι → α} {s : Set α} (h : ∀ x, f x ∈ s) :
    Injective (codRestrict f s h) ↔ Injective f := by
  /-
    α : Type u_1
    ι : Sort u_5
    f : ι → α
    s : Set α
    h : ∀ (x : ι), Membership.mem s (f x)
    ⊢ Iff (Function.Injective (Set.codRestrict f s h)) (Function.Injective f)
  -/
  simp only [Injective, Subtype.ext_iff, val_codRestrict_apply]
  /-
    🎉 no goals
  -/


alias ⟨_, _root_.Function.Injective.codRestrict⟩ := injective_codRestrict


@[simp]
theorem eqOn_empty (f₁ f₂ : α → β) : EqOn f₁ f₂ ∅ := fun _ => False.elim


@[simp]
theorem eqOn_singleton : Set.EqOn f₁ f₂ {a} ↔ f₁ a = f₂ a := by
  /-
    α : Type u_1
    β : Type u_2
    f₁ f₂ : α → β
    a : α
    ⊢ Iff (Set.EqOn f₁ f₂ (Singleton.singleton a)) (Eq (f₁ a) (f₂ a))
  -/
  simp [Set.EqOn]
  /-
    🎉 no goals
  -/


@[simp]
theorem eqOn_univ (f₁ f₂ : α → β) : EqOn f₁ f₂ univ ↔ f₁ = f₂ := by
  /-
    α : Type u_1
    β : Type u_2
    f₁ f₂ : α → β
    ⊢ Iff (Set.EqOn f₁ f₂ Set.univ) (Eq f₁ f₂)
  -/
  simp [EqOn, funext_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem restrict_eq_restrict_iff : restrict s f₁ = restrict s f₂ ↔ EqOn f₁ f₂ s :=
  restrict_eq_iff


@[symm]
theorem EqOn.symm (h : EqOn f₁ f₂ s) : EqOn f₂ f₁ s := fun _ hx => (h hx).symm


theorem eqOn_comm : EqOn f₁ f₂ s ↔ EqOn f₂ f₁ s :=
  ⟨EqOn.symm, EqOn.symm⟩

-- This can not be tagged as `@[refl]` with the current argument order.
-- See note below at `EqOn.trans`.

theorem eqOn_refl (f : α → β) (s : Set α) : EqOn f f s := fun _ _ => rfl

-- Note: this was formerly tagged with `@[trans]`, and although the `trans` attribute accepted it
-- the `trans` tactic could not use it.
-- An update to the trans tactic coming in https://github.com/leanprover-community/mathlib4/pull/7014 will reject this attribute.
-- It can be restored by changing the argument order from `EqOn f₁ f₂ s` to `EqOn s f₁ f₂`.
-- This change will be made separately: [zulip](https://leanprover.zulipchat.com/#narrow/stream/287929-mathlib4/topic/Reordering.20arguments.20of.20.60Set.2EEqOn.60/near/390467581).

theorem EqOn.trans (h₁ : EqOn f₁ f₂ s) (h₂ : EqOn f₂ f₃ s) : EqOn f₁ f₃ s := fun _ hx =>
  (h₁ hx).trans (h₂ hx)


theorem EqOn.image_eq (heq : EqOn f₁ f₂ s) : f₁ '' s = f₂ '' s :=
  image_congr heq


/-- Variant of `EqOn.image_eq`, for one function being the identity. -/
theorem EqOn.image_eq_self {f : α → α} (h : Set.EqOn f id s) : f '' s = s := by
  /-
    α : Type u_1
    s : Set α
    f : α → α
    h : Set.EqOn f id s
    ⊢ Eq (Set.image f s) s
  -/
  rw [h.image_eq, image_id]
  /-
    🎉 no goals
  -/


theorem EqOn.inter_preimage_eq (heq : EqOn f₁ f₂ s) (t : Set β) : s ∩ f₁ ⁻¹' t = s ∩ f₂ ⁻¹' t :=
                                                  /-
                                                    α : Type u_1
                                                    β : Type u_2
                                                    s : Set α
                                                    f₁ f₂ : α → β
                                                    heq : Set.EqOn f₁ f₂ s
                                                    t : Set β
                                                    x : α
                                                    hx : Membership.mem s x
                                                    ⊢ Iff (Membership.mem (Set.preimage f₁ t) x) (Membership.mem (Set.preimage f₂  …
                                                  -/
  ext fun x => and_congr_right_iff.2 fun hx => by rw [mem_preimage, mem_preimage, heq hx]
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem EqOn.mono (hs : s₁ ⊆ s₂) (hf : EqOn f₁ f₂ s₂) : EqOn f₁ f₂ s₁ := fun _ hx => hf (hs hx)


@[simp]
theorem eqOn_union : EqOn f₁ f₂ (s₁ ∪ s₂) ↔ EqOn f₁ f₂ s₁ ∧ EqOn f₁ f₂ s₂ :=
  forall₂_or_left


theorem EqOn.union (h₁ : EqOn f₁ f₂ s₁) (h₂ : EqOn f₁ f₂ s₂) : EqOn f₁ f₂ (s₁ ∪ s₂) :=
  eqOn_union.2 ⟨h₁, h₂⟩


theorem EqOn.comp_left (h : s.EqOn f₁ f₂) : s.EqOn (g ∘ f₁) (g ∘ f₂) := fun _ ha =>
  congr_arg _ <| h ha


@[simp]
theorem eqOn_range {ι : Sort*} {f : ι → α} {g₁ g₂ : α → β} :
    EqOn g₁ g₂ (range f) ↔ g₁ ∘ f = g₂ ∘ f :=
  forall_mem_range.trans <| funext_iff.symm


alias ⟨EqOn.comp_eq, _⟩ := eqOn_range


theorem MapsTo.restrict_commutes (f : α → β) (s : Set α) (t : Set β) (h : MapsTo f s t) :
    Subtype.val ∘ h.restrict f s t = f ∘ Subtype.val :=
  rfl


@[simp]
theorem MapsTo.val_restrict_apply (h : MapsTo f s t) (x : s) : (h.restrict f s t x : β) = f x :=
  rfl


theorem MapsTo.coe_iterate_restrict {f : α → α} (h : MapsTo f s s) (x : s) (k : ℕ) :
    h.restrict^[k] x = f^[k] x := by
  induction k with
  | zero => simp
  | succ k ih => simp only [iterate_succ', comp_apply, val_restrict_apply, ih]


/-- Restricting the domain and then the codomain is the same as `MapsTo.restrict`. -/
@[simp]
theorem codRestrict_restrict (h : ∀ x : s, f x ∈ t) :
    codRestrict (s.restrict f) t h = MapsTo.restrict f s t fun x hx => h ⟨x, hx⟩ :=
  rfl


/-- Reverse of `Set.codRestrict_restrict`. -/
theorem MapsTo.restrict_eq_codRestrict (h : MapsTo f s t) :
    h.restrict f s t = codRestrict (s.restrict f) t fun x => h x.2 :=
  rfl


theorem MapsTo.coe_restrict (h : Set.MapsTo f s t) :
    Subtype.val ∘ h.restrict f s t = s.restrict f :=
  rfl


theorem MapsTo.range_restrict (f : α → β) (s : Set α) (t : Set β) (h : MapsTo f s t) :
    range (h.restrict f s t) = Subtype.val ⁻¹' (f '' s) :=
  Set.range_subtype_map f h


theorem mapsTo_iff_exists_map_subtype : MapsTo f s t ↔ ∃ g : s → t, ∀ x : s, f x = g x :=
  ⟨fun h => ⟨h.restrict f s t, fun _ => rfl⟩, fun ⟨g, hg⟩ x hx => by
    /-
      α : Type u_1
      β : Type u_2
      s : Set α
      t : Set β
      f : α → β
      x✝ : Exists fun g => ∀ (x : ↑s), Eq (f ↑x) ↑(g x)
      x : α
      hx : Membership.mem s x
      g : ↑s → ↑t
      hg : ∀ (x : ↑s), Eq (f ↑x) ↑(g x)
      ⊢ Membership.mem t (f x)
    -/
    rw [hg ⟨x, hx⟩]
    /-
      α : Type u_1
      β : Type u_2
      s : Set α
      t : Set β
      f : α → β
      x✝ : Exists fun g => ∀ (x : ↑s), Eq (f ↑x) ↑(g x)
      x : α
      hx : Membership.mem s x
      g : ↑s → ↑t
      hg : ∀ (x : ↑s), Eq (f ↑x) ↑(g x)
      ⊢ Membership.mem t ↑(g ⟨x, hx⟩)
    -/
    apply Subtype.coe_prop⟩
    /-
      🎉 no goals
    -/


theorem mapsTo' : MapsTo f s t ↔ f '' s ⊆ t :=
  image_subset_iff.symm


theorem mapsTo_prod_map_diagonal : MapsTo (Prod.map f f) (diagonal α) (diagonal β) :=
  diagonal_subset_iff.2 fun _ => rfl


theorem MapsTo.subset_preimage (hf : MapsTo f s t) : s ⊆ f ⁻¹' t := hf


theorem mapsTo_iff_subset_preimage : MapsTo f s t ↔ s ⊆ f ⁻¹' t := Iff.rfl


@[simp]
theorem mapsTo_singleton {x : α} : MapsTo f {x} t ↔ f x ∈ t :=
  singleton_subset_iff


theorem mapsTo_empty (f : α → β) (t : Set β) : MapsTo f ∅ t :=
  empty_subset _


@[simp] theorem mapsTo_empty_iff : MapsTo f s ∅ ↔ s = ∅ := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    f : α → β
    ⊢ Iff (Set.MapsTo f s EmptyCollection.emptyCollection) (Eq s EmptyCollection.e …
  -/
  simp [mapsTo', subset_empty_iff]
  /-
    🎉 no goals
  -/


/-- If `f` maps `s` to `t` and `s` is non-empty, `t` is non-empty. -/
theorem MapsTo.nonempty (h : MapsTo f s t) (hs : s.Nonempty) : t.Nonempty :=
  (hs.image f).mono (mapsTo'.mp h)


theorem MapsTo.image_subset (h : MapsTo f s t) : f '' s ⊆ t :=
  mapsTo'.1 h


theorem MapsTo.congr (h₁ : MapsTo f₁ s t) (h : EqOn f₁ f₂ s) : MapsTo f₂ s t := fun _ hx =>
  h hx ▸ h₁ hx


theorem EqOn.comp_right (hg : t.EqOn g₁ g₂) (hf : s.MapsTo f t) : s.EqOn (g₁ ∘ f) (g₂ ∘ f) :=
  fun _ ha => hg <| hf ha


theorem EqOn.mapsTo_iff (H : EqOn f₁ f₂ s) : MapsTo f₁ s t ↔ MapsTo f₂ s t :=
  ⟨fun h => h.congr H, fun h => h.congr H.symm⟩


theorem MapsTo.comp (h₁ : MapsTo g t p) (h₂ : MapsTo f s t) : MapsTo (g ∘ f) s p := fun _ h =>
  h₁ (h₂ h)


theorem mapsTo_id (s : Set α) : MapsTo id s s := fun _ => id


theorem MapsTo.iterate {f : α → α} {s : Set α} (h : MapsTo f s s) : ∀ n, MapsTo f^[n] s s
  | 0 => fun _ => id
  | n + 1 => (MapsTo.iterate h n).comp h


theorem MapsTo.iterate_restrict {f : α → α} {s : Set α} (h : MapsTo f s s) (n : ℕ) :
    (h.restrict f s s)^[n] = (h.iterate n).restrict _ _ _ := by
  /-
    α : Type u_1
    f : α → α
    s : Set α
    h : Set.MapsTo f s s
    n : Nat
    ⊢ Eq (Nat.iterate (Set.MapsTo.restrict f s s h) n) (Set.MapsTo.restrict (Nat.i …
  -/
  funext x
  /-
    case h
    α : Type u_1
    f : α → α
    s : Set α
    h : Set.MapsTo f s s
    n : Nat
    x : ↑s
    ⊢ Eq (Nat.iterate (Set.MapsTo.restrict f s s h) n x) (Set.MapsTo.restrict (Nat …
  -/
  rw [Subtype.ext_iff, MapsTo.val_restrict_apply]
  induction n generalizing x with
  | zero => rfl
  | succ n ihn => simp [Nat.iterate, ihn]


lemma mapsTo_of_subsingleton' [Subsingleton β] (f : α → β) (h : s.Nonempty → t.Nonempty) :
    MapsTo f s t :=
  fun a ha ↦ Subsingleton.mem_iff_nonempty.2 <| h ⟨a, ha⟩


lemma mapsTo_of_subsingleton [Subsingleton α] (f : α → α) (s : Set α) : MapsTo f s s :=
  mapsTo_of_subsingleton' _ id


theorem MapsTo.mono (hf : MapsTo f s₁ t₁) (hs : s₂ ⊆ s₁) (ht : t₁ ⊆ t₂) : MapsTo f s₂ t₂ :=
  fun _ hx => ht (hf <| hs hx)


theorem MapsTo.mono_left (hf : MapsTo f s₁ t) (hs : s₂ ⊆ s₁) : MapsTo f s₂ t := fun _ hx =>
  hf (hs hx)


theorem MapsTo.mono_right (hf : MapsTo f s t₁) (ht : t₁ ⊆ t₂) : MapsTo f s t₂ := fun _ hx =>
  ht (hf hx)


theorem MapsTo.union_union (h₁ : MapsTo f s₁ t₁) (h₂ : MapsTo f s₂ t₂) :
    MapsTo f (s₁ ∪ s₂) (t₁ ∪ t₂) := fun _ hx =>
  hx.elim (fun hx => Or.inl <| h₁ hx) fun hx => Or.inr <| h₂ hx


theorem MapsTo.union (h₁ : MapsTo f s₁ t) (h₂ : MapsTo f s₂ t) : MapsTo f (s₁ ∪ s₂) t :=
  union_self t ▸ h₁.union_union h₂


@[simp]
theorem mapsTo_union : MapsTo f (s₁ ∪ s₂) t ↔ MapsTo f s₁ t ∧ MapsTo f s₂ t :=
  ⟨fun h =>
    ⟨h.mono subset_union_left (Subset.refl t),
      h.mono subset_union_right (Subset.refl t)⟩,
    fun h => h.1.union h.2⟩


theorem MapsTo.inter (h₁ : MapsTo f s t₁) (h₂ : MapsTo f s t₂) : MapsTo f s (t₁ ∩ t₂) := fun _ hx =>
  ⟨h₁ hx, h₂ hx⟩


lemma MapsTo.insert (h : MapsTo f s t) (x : α) : MapsTo f (insert x s) (insert (f x) t) := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    f : α → β
    h : Set.MapsTo f s t
    x : α
    ⊢ Set.MapsTo f (Insert.insert x s) (Insert.insert (f x) t)
  -/
  simpa [← singleton_union] using h.mono_right subset_union_right
  /-
    🎉 no goals
  -/


theorem MapsTo.inter_inter (h₁ : MapsTo f s₁ t₁) (h₂ : MapsTo f s₂ t₂) :
    MapsTo f (s₁ ∩ s₂) (t₁ ∩ t₂) := fun _ hx => ⟨h₁ hx.1, h₂ hx.2⟩


@[simp]
theorem mapsTo_inter : MapsTo f s (t₁ ∩ t₂) ↔ MapsTo f s t₁ ∧ MapsTo f s t₂ :=
  ⟨fun h =>
    ⟨h.mono (Subset.refl s) inter_subset_left,
      h.mono (Subset.refl s) inter_subset_right⟩,
    fun h => h.1.inter h.2⟩


theorem mapsTo_univ (f : α → β) (s : Set α) : MapsTo f s univ := fun _ _ => trivial


theorem mapsTo_range (f : α → β) (s : Set α) : MapsTo f s (range f) :=
  (mapsTo_image f s).mono (Subset.refl s) (image_subset_range _ _)


@[simp]
theorem mapsTo_image_iff {f : α → β} {g : γ → α} {s : Set γ} {t : Set β} :
    MapsTo f (g '' s) t ↔ MapsTo (f ∘ g) s t :=
  ⟨fun h c hc => h ⟨c, hc, rfl⟩, fun h _ ⟨_, hc⟩ => hc.2 ▸ h hc.1⟩


lemma MapsTo.comp_left (g : β → γ) (hf : MapsTo f s t) : MapsTo (g ∘ f) s (g '' t) :=
  fun x hx ↦ ⟨f x, hf hx, rfl⟩


lemma MapsTo.comp_right {s : Set β} {t : Set γ} (hg : MapsTo g s t) (f : α → β) :
    MapsTo (g ∘ f) (f ⁻¹' s) t := fun _ hx ↦ hg hx


@[simp]
lemma mapsTo_univ_iff : MapsTo f univ t ↔ ∀ x, f x ∈ t :=
  ⟨fun h _ => h (mem_univ _), fun h x _ => h x⟩


@[simp]
lemma mapsTo_range_iff {g : ι → α} : MapsTo f (range g) t ↔ ∀ i, f (g i) ∈ t :=
  forall_mem_range


theorem surjective_mapsTo_image_restrict (f : α → β) (s : Set α) :
    Surjective ((mapsTo_image f s).restrict f s (f '' s)) := fun ⟨_, x, hs, hxy⟩ =>
  ⟨⟨x, hs⟩, Subtype.ext hxy⟩


theorem MapsTo.mem_iff (h : MapsTo f s t) (hc : MapsTo f sᶜ tᶜ) {x} : f x ∈ t ↔ x ∈ s :=
  ⟨fun ht => by_contra fun hs => hc hs ht, fun hx => h hx⟩


variable (f s) in
theorem image_restrictPreimage :
    t.restrictPreimage f '' (Subtype.val ⁻¹' s) = Subtype.val ⁻¹' (f '' s) := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    f : α → β
    ⊢ Eq (Set.image (t.restrictPreimage f) (Set.preimage Subtype.val s)) (Set.prei …
  -/
  delta Set.restrictPreimage
  rw [← (Subtype.coe_injective).image_injective.eq_iff, ← image_comp, MapsTo.restrict_commutes,
    image_comp, Subtype.image_preimage_coe, Subtype.image_preimage_coe, image_preimage_inter]


variable (f) in
theorem range_restrictPreimage : range (t.restrictPreimage f) = Subtype.val ⁻¹' range f := by
  /-
    α : Type u_1
    β : Type u_2
    t : Set β
    f : α → β
    ⊢ Eq (Set.range (t.restrictPreimage f)) (Set.preimage Subtype.val (Set.range f))
  -/
  simp only [← image_univ, ← image_restrictPreimage, preimage_univ]
  /-
    🎉 no goals
  -/


@[simp]
theorem restrictPreimage_mk (h : a ∈ f ⁻¹' t) : t.restrictPreimage f ⟨a, h⟩ = ⟨f a, h⟩ := rfl


theorem image_val_preimage_restrictPreimage {u : Set t} :
    Subtype.val '' (t.restrictPreimage f ⁻¹' u) = f ⁻¹' (Subtype.val '' u) := by
  /-
    α : Type u_1
    β : Type u_2
    t : Set β
    f : α → β
    u : Set ↑t
    ⊢ Eq (Set.image Subtype.val (Set.preimage (t.restrictPreimage f) u)) (Set.prei …
  -/
  ext
  /-
    case h
    α : Type u_1
    β : Type u_2
    t : Set β
    f : α → β
    u : Set ↑t
    x✝ : α
    ⊢ Iff (Membership.mem (Set.image Subtype.val (Set.preimage (t.restrictPreimage …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem preimage_restrictPreimage {u : Set t} :
    t.restrictPreimage f ⁻¹' u = (fun a : f ⁻¹' t ↦ f a) ⁻¹' (Subtype.val '' u) := by
  rw [← preimage_preimage (g := f) (f := Subtype.val), ← image_val_preimage_restrictPreimage,
    preimage_image_eq _ Subtype.val_injective]


lemma restrictPreimage_injective (hf : Injective f) : Injective (t.restrictPreimage f) :=
  fun _ _ e => Subtype.coe_injective <| hf <| Subtype.mk.inj e


lemma restrictPreimage_surjective (hf : Surjective f) : Surjective (t.restrictPreimage f) :=
  fun x => ⟨⟨_, ((hf x).choose_spec.symm ▸ x.2 : _ ∈ t)⟩, Subtype.ext (hf x).choose_spec⟩


lemma restrictPreimage_bijective (hf : Bijective f) : Bijective (t.restrictPreimage f) :=
  ⟨t.restrictPreimage_injective hf.1, t.restrictPreimage_surjective hf.2⟩


alias _root_.Function.Injective.restrictPreimage := Set.restrictPreimage_injective

alias _root_.Function.Surjective.restrictPreimage := Set.restrictPreimage_surjective

alias _root_.Function.Bijective.restrictPreimage := Set.restrictPreimage_bijective


theorem Subsingleton.injOn (hs : s.Subsingleton) (f : α → β) : InjOn f s := fun _ hx _ hy _ =>
  hs hx hy


@[simp]
theorem injOn_empty (f : α → β) : InjOn f ∅ :=
  subsingleton_empty.injOn f

@[simp]
theorem injOn_singleton (f : α → β) (a : α) : InjOn f {a} :=
  subsingleton_singleton.injOn f


                                                                            /-
                                                                              α : Type u_1
                                                                              β : Type u_2
                                                                              f : α → β
                                                                              a b : α
                                                                              ⊢ Iff (Set.InjOn f (Insert.insert a (Singleton.singleton b))) (Eq (f a) (f b)  …
                                                                            -/
@[simp] lemma injOn_pair {b : α} : InjOn f {a, b} ↔ f a = f b → a = b := by unfold InjOn; aesop
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


theorem InjOn.eq_iff {x y} (h : InjOn f s) (hx : x ∈ s) (hy : y ∈ s) : f x = f y ↔ x = y :=
  ⟨h hx hy, fun h => h ▸ rfl⟩


theorem InjOn.ne_iff {x y} (h : InjOn f s) (hx : x ∈ s) (hy : y ∈ s) : f x ≠ f y ↔ x ≠ y :=
  (h.eq_iff hx hy).not


alias ⟨_, InjOn.ne⟩ := InjOn.ne_iff


theorem InjOn.congr (h₁ : InjOn f₁ s) (h : EqOn f₁ f₂ s) : InjOn f₂ s := fun _ hx _ hy =>
  h hx ▸ h hy ▸ h₁ hx hy


theorem EqOn.injOn_iff (H : EqOn f₁ f₂ s) : InjOn f₁ s ↔ InjOn f₂ s :=
  ⟨fun h => h.congr H, fun h => h.congr H.symm⟩


theorem InjOn.mono (h : s₁ ⊆ s₂) (ht : InjOn f s₂) : InjOn f s₁ := fun _ hx _ hy H =>
  ht (h hx) (h hy) H


theorem injOn_union (h : Disjoint s₁ s₂) :
    InjOn f (s₁ ∪ s₂) ↔ InjOn f s₁ ∧ InjOn f s₂ ∧ ∀ x ∈ s₁, ∀ y ∈ s₂, f x ≠ f y := by
  /-
    α : Type u_1
    β : Type u_2
    s₁ s₂ : Set α
    f : α → β
    h : Disjoint s₁ s₂
    ⊢ Iff (Set.InjOn f (Union.union s₁ s₂)) (And (Set.InjOn f s₁) (And (Set.InjOn  …
  -/
  refine ⟨fun H => ⟨H.mono subset_union_left, H.mono subset_union_right, ?_⟩, ?_⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      s₁ s₂ : Set α
      f : α → β
      h : Disjoint s₁ s₂
      H : Set.InjOn f (Union.union s₁ s₂)
      ⊢ ∀ (x : α), Membership.mem s₁ x → ∀ (y : α), Membership.mem s₂ y → Ne (f x) ( …
    -/
  · intro x hx y hy hxy
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      s₁ s₂ : Set α
      f : α → β
      h : Disjoint s₁ s₂
      H : Set.InjOn f (Union.union s₁ s₂)
      x : α
      hx : Membership.mem s₁ x
      y : α
      hy : Membership.mem s₂ y
      hxy : Eq (f x) (f y)
      ⊢ False
    -/
    obtain rfl : x = y := H (Or.inl hx) (Or.inr hy) hxy
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      s₁ s₂ : Set α
      f : α → β
      h : Disjoint s₁ s₂
      H : Set.InjOn f (Union.union s₁ s₂)
      x : α
      hx : Membership.mem s₁ x
      hy : Membership.mem s₂ x
      hxy : Eq (f x) (f x)
      ⊢ False
    -/
    exact h.le_bot ⟨hx, hy⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      s₁ s₂ : Set α
      f : α → β
      h : Disjoint s₁ s₂
      ⊢ And (Set.InjOn f s₁) (And (Set.InjOn f s₂) (∀ (x : α), Membership.mem s₁ x → …
    -/
  · rintro ⟨h₁, h₂, h₁₂⟩
    /-
      case refine_2.intro.intro
      α : Type u_1
      β : Type u_2
      s₁ s₂ : Set α
      f : α → β
      h : Disjoint s₁ s₂
      h₁ : Set.InjOn f s₁
      h₂ : Set.InjOn f s₂
      h₁₂ : ∀ (x : α), Membership.mem s₁ x → ∀ (y : α), Membership.mem s₂ y → Ne (f  …
      ⊢ Set.InjOn f (Union.union s₁ s₂)
    -/
    rintro x (hx | hx) y (hy | hy) hxy
    /-
      case refine_2.intro.intro.inl.inl
      α : Type u_1
      β : Type u_2
      s₁ s₂ : Set α
      f : α → β
      h : Disjoint s₁ s₂
      h₁ : Set.InjOn f s₁
      h₂ : Set.InjOn f s₂
      h₁₂ : ∀ (x : α), Membership.mem s₁ x → ∀ (y : α), Membership.mem s₂ y → Ne (f  …
      x : α
      hx : Membership.mem s₁ x
      y : α
      hy : Membership.mem s₁ y
      hxy : Eq (f x) (f y)
      ⊢ Eq x y
    -/
    exacts [h₁ hx hy hxy, (h₁₂ _ hx _ hy hxy).elim, (h₁₂ _ hy _ hx hxy.symm).elim, h₂ hx hy hxy]
    /-
      🎉 no goals
    -/


theorem injOn_insert {f : α → β} {s : Set α} {a : α} (has : a ∉ s) :
    Set.InjOn f (insert a s) ↔ Set.InjOn f s ∧ f a ∉ f '' s := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    s : Set α
    a : α
    has : Not (Membership.mem s a)
    ⊢ Iff (Set.InjOn f (Insert.insert a s)) (And (Set.InjOn f s) (Not (Membership. …
  -/
  rw [← union_singleton, injOn_union (disjoint_singleton_right.2 has)]
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    s : Set α
    a : α
    has : Not (Membership.mem s a)
    ⊢ Iff (And (Set.InjOn f s) (And (Set.InjOn f (Singleton.singleton a)) (∀ (x :  …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem injective_iff_injOn_univ : Injective f ↔ InjOn f univ :=
  ⟨fun h _ _ _ _ hxy => h hxy, fun h _ _ heq => h trivial trivial heq⟩


theorem injOn_of_injective (h : Injective f) {s : Set α} : InjOn f s := fun _ _ _ _ hxy => h hxy


alias _root_.Function.Injective.injOn := injOn_of_injective

-- A specialization of `injOn_of_injective` for `Subtype.val`.

theorem injOn_subtype_val {s : Set { x // p x }} : Set.InjOn Subtype.val s :=
  Subtype.coe_injective.injOn


lemma injOn_id (s : Set α) : InjOn id s := injective_id.injOn


theorem InjOn.comp (hg : InjOn g t) (hf : InjOn f s) (h : MapsTo f s t) : InjOn (g ∘ f) s :=
  fun _ hx _ hy heq => hf hx hy <| hg (h hx) (h hy) heq


lemma InjOn.image_of_comp (h : InjOn (g ∘ f) s) : InjOn g (f '' s) :=
  forall_mem_image.2 fun _x hx ↦ forall_mem_image.2 fun _y hy heq ↦ congr_arg f <| h hx hy heq


lemma InjOn.iterate {f : α → α} {s : Set α} (h : InjOn f s) (hf : MapsTo f s s) :
    ∀ n, InjOn f^[n] s
  | 0 => injOn_id _
  | (n + 1) => (h.iterate hf n).comp h hf


lemma injOn_of_subsingleton [Subsingleton α] (f : α → β) (s : Set α) : InjOn f s :=
  (injective_of_subsingleton _).injOn


theorem _root_.Function.Injective.injOn_range (h : Injective (g ∘ f)) : InjOn g (range f) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → β
    g : β → γ
    h : Function.Injective (Function.comp g f)
    ⊢ Set.InjOn g (Set.range f)
  -/
  rintro _ ⟨x, rfl⟩ _ ⟨y, rfl⟩ H
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → β
    g : β → γ
    h : Function.Injective (Function.comp g f)
    x y : α
    H : Eq (g (f x)) (g (f y))
    ⊢ Eq (f x) (f y)
  -/
  exact congr_arg f (h H)
  /-
    🎉 no goals
  -/


theorem injOn_iff_injective : InjOn f s ↔ Injective (s.restrict f) :=
  ⟨fun H a b h => Subtype.eq <| H a.2 b.2 h, fun H a as b bs h =>
    congr_arg Subtype.val <| @H ⟨a, as⟩ ⟨b, bs⟩ h⟩


alias ⟨InjOn.injective, _⟩ := Set.injOn_iff_injective


theorem MapsTo.restrict_inj (h : MapsTo f s t) : Injective (h.restrict f s t) ↔ InjOn f s := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    f : α → β
    h : Set.MapsTo f s t
    ⊢ Iff (Function.Injective (Set.MapsTo.restrict f s t h)) (Set.InjOn f s)
  -/
  rw [h.restrict_eq_codRestrict, injective_codRestrict, injOn_iff_injective]
  /-
    🎉 no goals
  -/


theorem exists_injOn_iff_injective [Nonempty β] :
    (∃ f : α → β, InjOn f s) ↔ ∃ f : s → β, Injective f :=
  ⟨fun ⟨_, hf⟩ => ⟨_, hf.injective⟩,
   fun ⟨f, hf⟩ => by
    /-
      α : Type u_1
      β : Type u_2
      s : Set α
      inst✝ : Nonempty β
      x✝ : Exists fun f => Function.Injective f
      f : ↑s → β
      hf : Function.Injective f
      ⊢ Exists fun f => Set.InjOn f s
    -/
    lift f to α → β using trivial
    /-
      case intro
      α : Type u_1
      β : Type u_2
      s : Set α
      inst✝ : Nonempty β
      x✝ : Exists fun f => Function.Injective f
      f : α → β
      hf : Function.Injective fun i => f ↑i
      ⊢ Exists fun f => Set.InjOn f s
    -/
    exact ⟨f, injOn_iff_injective.2 hf⟩⟩
    /-
      🎉 no goals
    -/


theorem injOn_preimage {B : Set (Set β)} (hB : B ⊆ 𝒫 range f) : InjOn (preimage f) B :=
  fun s hs t ht hst => (preimage_eq_preimage' (@hB s hs) (@hB t ht)).1 hst
-- Porting note: is there a semi-implicit variable problem with `⊆`?


theorem InjOn.mem_of_mem_image {x} (hf : InjOn f s) (hs : s₁ ⊆ s) (h : x ∈ s) (h₁ : f x ∈ f '' s₁) :
    x ∈ s₁ :=
  let ⟨_, h', Eq⟩ := h₁
  hf (hs h') h Eq ▸ h'


theorem InjOn.mem_image_iff {x} (hf : InjOn f s) (hs : s₁ ⊆ s) (hx : x ∈ s) :
    f x ∈ f '' s₁ ↔ x ∈ s₁ :=
  ⟨hf.mem_of_mem_image hs hx, mem_image_of_mem f⟩


theorem InjOn.preimage_image_inter (hf : InjOn f s) (hs : s₁ ⊆ s) : f ⁻¹' (f '' s₁) ∩ s = s₁ :=
  ext fun _ => ⟨fun ⟨h₁, h₂⟩ => hf.mem_of_mem_image hs h₂ h₁, fun h => ⟨mem_image_of_mem _ h, hs h⟩⟩


theorem EqOn.cancel_left (h : s.EqOn (g ∘ f₁) (g ∘ f₂)) (hg : t.InjOn g) (hf₁ : s.MapsTo f₁ t)
    (hf₂ : s.MapsTo f₂ t) : s.EqOn f₁ f₂ := fun _ ha => hg (hf₁ ha) (hf₂ ha) (h ha)


theorem InjOn.cancel_left (hg : t.InjOn g) (hf₁ : s.MapsTo f₁ t) (hf₂ : s.MapsTo f₂ t) :
    s.EqOn (g ∘ f₁) (g ∘ f₂) ↔ s.EqOn f₁ f₂ :=
  ⟨fun h => h.cancel_left hg hf₁ hf₂, EqOn.comp_left⟩


lemma InjOn.image_inter {s t u : Set α} (hf : u.InjOn f) (hs : s ⊆ u) (ht : t ⊆ u) :
    f '' (s ∩ t) = f '' s ∩ f '' t := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    s t u : Set α
    hf : Set.InjOn f u
    hs : HasSubset.Subset s u
    ht : HasSubset.Subset t u
    ⊢ Eq (Set.image f (Inter.inter s t)) (Inter.inter (Set.image f s) (Set.image f …
  -/
  apply Subset.antisymm (image_inter_subset _ _ _)
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    s t u : Set α
    hf : Set.InjOn f u
    hs : HasSubset.Subset s u
    ht : HasSubset.Subset t u
    ⊢ HasSubset.Subset (Inter.inter (Set.image f s) (Set.image f t)) (Set.image f  …
  -/
  intro x ⟨⟨y, ys, hy⟩, ⟨z, zt, hz⟩⟩
  have : y = z := by
    apply hf (hs ys) (ht zt)
    rwa [← hz] at hy
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    s t u : Set α
    hf : Set.InjOn f u
    hs : HasSubset.Subset s u
    ht : HasSubset.Subset t u
    x : β
    y : α
    ys : Membership.mem s y
    hy : Eq (f y) x
    z : α
    zt : Membership.mem t z
    hz : Eq (f z) x
    this : Eq y z
    ⊢ Membership.mem (Set.image f (Inter.inter s t)) x
  -/
  rw [← this] at zt
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    s t u : Set α
    hf : Set.InjOn f u
    hs : HasSubset.Subset s u
    ht : HasSubset.Subset t u
    x : β
    y : α
    ys : Membership.mem s y
    hy : Eq (f y) x
    z : α
    zt : Membership.mem t y
    hz : Eq (f z) x
    this : Eq y z
    ⊢ Membership.mem (Set.image f (Inter.inter s t)) x
  -/
  exact ⟨y, ⟨ys, zt⟩, hy⟩
  /-
    🎉 no goals
  -/


lemma InjOn.image (h : s.InjOn f) : s.powerset.InjOn (image f) :=
                            /-
                              α : Type u_1
                              β : Type u_2
                              s : Set α
                              f : α → β
                              h : Set.InjOn f s
                              s₁ : Set α
                              hs₁ : Membership.mem s.powerset s₁
                              s₂ : Set α
                              hs₂ : Membership.mem s.powerset s₂
                              h' : Eq (Set.image f s₁) (Set.image f s₂)
                              ⊢ Eq s₁ s₂
                            -/
  fun s₁ hs₁ s₂ hs₂ h' ↦ by rw [← h.preimage_image_inter hs₁, h', h.preimage_image_inter hs₂]
                            /-
                              🎉 no goals
                            -/


theorem InjOn.image_eq_image_iff (h : s.InjOn f) (h₁ : s₁ ⊆ s) (h₂ : s₂ ⊆ s) :
    f '' s₁ = f '' s₂ ↔ s₁ = s₂ :=
  h.image.eq_iff h₁ h₂


lemma InjOn.image_subset_image_iff (h : s.InjOn f) (h₁ : s₁ ⊆ s) (h₂ : s₂ ⊆ s) :
    f '' s₁ ⊆ f '' s₂ ↔ s₁ ⊆ s₂ := by
  /-
    α : Type u_1
    β : Type u_2
    s s₁ s₂ : Set α
    f : α → β
    h : Set.InjOn f s
    h₁ : HasSubset.Subset s₁ s
    h₂ : HasSubset.Subset s₂ s
    ⊢ Iff (HasSubset.Subset (Set.image f s₁) (Set.image f s₂)) (HasSubset.Subset s …
  -/
  refine ⟨fun h' ↦ ?_, image_subset _⟩
  /-
    α : Type u_1
    β : Type u_2
    s s₁ s₂ : Set α
    f : α → β
    h : Set.InjOn f s
    h₁ : HasSubset.Subset s₁ s
    h₂ : HasSubset.Subset s₂ s
    h' : HasSubset.Subset (Set.image f s₁) (Set.image f s₂)
    ⊢ HasSubset.Subset s₁ s₂
  -/
  rw [← h.preimage_image_inter h₁, ← h.preimage_image_inter h₂]
  /-
    α : Type u_1
    β : Type u_2
    s s₁ s₂ : Set α
    f : α → β
    h : Set.InjOn f s
    h₁ : HasSubset.Subset s₁ s
    h₂ : HasSubset.Subset s₂ s
    h' : HasSubset.Subset (Set.image f s₁) (Set.image f s₂)
    ⊢ HasSubset.Subset (Inter.inter (Set.preimage f (Set.image f s₁)) s) (Inter.in …
  -/
  exact inter_subset_inter_left _ (preimage_mono h')
  /-
    🎉 no goals
  -/


lemma InjOn.image_ssubset_image_iff (h : s.InjOn f) (h₁ : s₁ ⊆ s) (h₂ : s₂ ⊆ s) :
    f '' s₁ ⊂ f '' s₂ ↔ s₁ ⊂ s₂ := by
  /-
    α : Type u_1
    β : Type u_2
    s s₁ s₂ : Set α
    f : α → β
    h : Set.InjOn f s
    h₁ : HasSubset.Subset s₁ s
    h₂ : HasSubset.Subset s₂ s
    ⊢ Iff (HasSSubset.SSubset (Set.image f s₁) (Set.image f s₂)) (HasSSubset.SSubs …
  -/
  simp_rw [ssubset_def, h.image_subset_image_iff h₁ h₂, h.image_subset_image_iff h₂ h₁]
  /-
    🎉 no goals
  -/

-- TODO: can this move to a better place?

theorem _root_.Disjoint.image {s t u : Set α} {f : α → β} (h : Disjoint s t) (hf : u.InjOn f)
    (hs : s ⊆ u) (ht : t ⊆ u) : Disjoint (f '' s) (f '' t) := by
  /-
    α : Type u_1
    β : Type u_2
    s t u : Set α
    f : α → β
    h : Disjoint s t
    hf : Set.InjOn f u
    hs : HasSubset.Subset s u
    ht : HasSubset.Subset t u
    ⊢ Disjoint (Set.image f s) (Set.image f t)
  -/
  rw [disjoint_iff_inter_eq_empty] at h ⊢
  /-
    α : Type u_1
    β : Type u_2
    s t u : Set α
    f : α → β
    h : Eq (Inter.inter s t) EmptyCollection.emptyCollection
    hf : Set.InjOn f u
    hs : HasSubset.Subset s u
    ht : HasSubset.Subset t u
    ⊢ Eq (Inter.inter (Set.image f s) (Set.image f t)) EmptyCollection.emptyCollec …
  -/
  rw [← hf.image_inter hs ht, h, image_empty]
  /-
    🎉 no goals
  -/


lemma InjOn.image_diff {t : Set α} (h : s.InjOn f) : f '' (s \ t) = f '' s \ f '' (s ∩ t) := by
  refine subset_antisymm (subset_diff.2 ⟨image_subset f diff_subset, ?_⟩)
    (diff_subset_iff.2 (by rw [← image_union, inter_union_diff]))
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    f : α → β
    t : Set α
    h : Set.InjOn f s
    ⊢ Disjoint (Set.image f (SDiff.sdiff s t)) (Set.image f (Inter.inter s t))
  -/
  exact Disjoint.image disjoint_sdiff_inter h diff_subset inter_subset_left
  /-
    🎉 no goals
  -/


lemma InjOn.image_diff_subset {f : α → β} {t : Set α} (h : InjOn f s) (hst : t ⊆ s) :
    f '' (s \ t) = f '' s \ f '' t := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    f : α → β
    t : Set α
    h : Set.InjOn f s
    hst : HasSubset.Subset t s
    ⊢ Eq (Set.image f (SDiff.sdiff s t)) (SDiff.sdiff (Set.image f s) (Set.image f …
  -/
  rw [h.image_diff, inter_eq_self_of_subset_right hst]
  /-
    🎉 no goals
  -/


alias image_diff_of_injOn := InjOn.image_diff_subset


theorem InjOn.imageFactorization_injective (h : InjOn f s) :
    Injective (s.imageFactorization f) :=
                              /-
                                α : Type u_1
                                β : Type u_2
                                s : Set α
                                f : α → β
                                h : Set.InjOn f s
                                x✝¹ x✝ : ↑s
                                x : α
                                hx : Membership.mem s x
                                y : α
                                hy : Membership.mem s y
                                h' : Eq (Set.imageFactorization f s ⟨x, hx⟩) (Set.imageFactorization f s ⟨y, h …
                                ⊢ Eq ⟨x, hx⟩ ⟨y, hy⟩
                              -/
  fun ⟨x, hx⟩ ⟨y, hy⟩ h' ↦ by simpa [imageFactorization, h.eq_iff hx hy] using h'
                              /-
                                🎉 no goals
                              -/


@[simp] theorem imageFactorization_injective_iff : Injective (s.imageFactorization f) ↔ InjOn f s :=
                          /-
                            α : Type u_1
                            β : Type u_2
                            s : Set α
                            f : α → β
                            h : Function.Injective (Set.imageFactorization f s)
                            x : α
                            hx : Membership.mem s x
                            y : α
                            hy : Membership.mem s y
                            x✝ : Eq (f x) (f y)
                            ⊢ Eq x y
                          -/
  ⟨fun h x hx y hy _ ↦ by simpa using @h ⟨x, hx⟩ ⟨y, hy⟩ (by simpa [imageFactorization]),
                          /-
                            🎉 no goals
                          -/
    InjOn.imageFactorization_injective⟩


                                                                          /-
                                                                            α : Type u_1
                                                                            β : Type u_2
                                                                            s : Set α
                                                                            f : α → β
                                                                            x : Prod α β
                                                                            ⊢ Iff (Membership.mem (Set.graphOn f s) x) (And (Membership.mem s x.1) (Eq (f  …
                                                                          -/
@[simp] lemma mem_graphOn : x ∈ s.graphOn f ↔ x.1 ∈ s ∧ f x.1 = x.2 := by aesop (add simp graphOn)
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


@[simp] lemma graphOn_empty (f : α → β) : graphOn f ∅ = ∅ := image_empty _

@[simp] lemma graphOn_eq_empty : graphOn f s = ∅ ↔ s = ∅ := image_eq_empty

@[simp] lemma graphOn_nonempty : (s.graphOn f).Nonempty ↔ s.Nonempty := image_nonempty


protected alias ⟨_, Nonempty.graphOn⟩ := graphOn_nonempty


@[simp]
lemma graphOn_union (f : α → β) (s t : Set α) : graphOn f (s ∪ t) = graphOn f s ∪ graphOn f t :=
  image_union ..


@[simp]
lemma graphOn_singleton (f : α → β) (x : α) : graphOn f {x} = {(x, f x)} :=
  image_singleton ..


@[simp]
lemma graphOn_insert (f : α → β) (x : α) (s : Set α) :
    graphOn f (insert x s) = insert (x, f x) (graphOn f s) :=
  image_insert_eq ..


@[simp]
lemma image_fst_graphOn (f : α → β) (s : Set α) : Prod.fst '' graphOn f s = s := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    s : Set α
    ⊢ Eq (Set.image Prod.fst (Set.graphOn f s)) s
  -/
  simp [graphOn, image_image]
  /-
    🎉 no goals
  -/


                                                                                     /-
                                                                                       α : Type u_1
                                                                                       β : Type u_2
                                                                                       s : Set α
                                                                                       f : α → β
                                                                                       ⊢ Eq (Set.image Prod.snd (Set.graphOn f s)) (Set.image f s)
                                                                                     -/
@[simp] lemma image_snd_graphOn (f : α → β) : Prod.snd '' s.graphOn f = f '' s := by ext x; simp
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/


                                                           /-
                                                             α : Type u_1
                                                             β : Type u_2
                                                             s : Set α
                                                             f : α → β
                                                             ⊢ Set.InjOn Prod.fst (Set.graphOn f s)
                                                           -/
lemma fst_injOn_graph : (s.graphOn f).InjOn Prod.fst := by aesop (add simp InjOn)
                                                           /-
                                                             🎉 no goals
                                                           -/


lemma graphOn_comp (s : Set α) (f : α → β) (g : β → γ) :
    s.graphOn (g ∘ f) = (fun x ↦ (x.1, g x.2)) '' s.graphOn f := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    s : Set α
    f : α → β
    g : β → γ
    ⊢ Eq (Set.graphOn (Function.comp g f) s) (Set.image (fun x => { fst := x.1, sn …
  -/
  simpa using image_comp (fun x ↦ (x.1, g x.2)) (fun x ↦ (x, f x)) _
  /-
    🎉 no goals
  -/


lemma graphOn_univ_eq_range : univ.graphOn f = range fun x ↦ (x, f x) := image_univ


@[simp] lemma graphOn_inj {g : α → β} : s.graphOn f = s.graphOn g ↔ s.EqOn f g := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    f g : α → β
    ⊢ Iff (Eq (Set.graphOn f s) (Set.graphOn g s)) (Set.EqOn f g s)
  -/
  simp [Set.ext_iff, funext_iff, forall_swap, EqOn]
  /-
    🎉 no goals
  -/


                                                                                   /-
                                                                                     α : Type u_1
                                                                                     β : Type u_2
                                                                                     f g : α → β
                                                                                     ⊢ Iff (Eq (Set.graphOn f Set.univ) (Set.graphOn g Set.univ)) (Eq f g)
                                                                                   -/
lemma graphOn_univ_inj {g : α → β} : univ.graphOn f = univ.graphOn g ↔ f = g := by simp
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


lemma graphOn_univ_injective : Injective (univ.graphOn : (α → β) → Set (α × β)) :=
  fun _f _g ↦ graphOn_univ_inj.1


lemma exists_eq_graphOn_image_fst [Nonempty β] {s : Set (α × β)} :
    (∃ f : α → β, s = graphOn f (Prod.fst '' s)) ↔ InjOn Prod.fst s := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : Nonempty β
    s : Set (Prod α β)
    ⊢ Iff (Exists fun f => Eq s (Set.graphOn f (Set.image Prod.fst s))) (Set.InjOn …
  -/
  refine ⟨?_, fun h ↦ ?_⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      inst✝ : Nonempty β
      s : Set (Prod α β)
      ⊢ (Exists fun f => Eq s (Set.graphOn f (Set.image Prod.fst s))) → Set.InjOn Pr …
    -/
  · rintro ⟨f, hf⟩
    /-
      case refine_1.intro
      α : Type u_1
      β : Type u_2
      inst✝ : Nonempty β
      s : Set (Prod α β)
      f : α → β
      hf : Eq s (Set.graphOn f (Set.image Prod.fst s))
      ⊢ Set.InjOn Prod.fst s
    -/
    rw [hf]
    /-
      case refine_1.intro
      α : Type u_1
      β : Type u_2
      inst✝ : Nonempty β
      s : Set (Prod α β)
      f : α → β
      hf : Eq s (Set.graphOn f (Set.image Prod.fst s))
      ⊢ Set.InjOn Prod.fst (Set.graphOn f (Set.image Prod.fst s))
    -/
    exact InjOn.image_of_comp <| injOn_id _
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      inst✝ : Nonempty β
      s : Set (Prod α β)
      h : Set.InjOn Prod.fst s
      ⊢ Exists fun f => Eq s (Set.graphOn f (Set.image Prod.fst s))
    -/
  · have : ∀ x ∈ Prod.fst '' s, ∃ y, (x, y) ∈ s := forall_mem_image.2 fun (x, y) h ↦ ⟨y, h⟩
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      inst✝ : Nonempty β
      s : Set (Prod α β)
      h : Set.InjOn Prod.fst s
      this : ∀ (x : α), Membership.mem (Set.image Prod.fst s) x → Exists fun y => Me …
      ⊢ Exists fun f => Eq s (Set.graphOn f (Set.image Prod.fst s))
    -/
    choose! f hf using this
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      inst✝ : Nonempty β
      s : Set (Prod α β)
      h : Set.InjOn Prod.fst s
      f : α → β
      hf : ∀ (x : α), Membership.mem (Set.image Prod.fst s) x → Membership.mem s { f …
      ⊢ Exists fun f => Eq s (Set.graphOn f (Set.image Prod.fst s))
    -/
    rw [forall_mem_image] at hf
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      inst✝ : Nonempty β
      s : Set (Prod α β)
      h : Set.InjOn Prod.fst s
      f : α → β
      hf : ∀ ⦃x : Prod α β⦄, Membership.mem s x → Membership.mem s { fst := x.1, snd …
      ⊢ Exists fun f => Eq s (Set.graphOn f (Set.image Prod.fst s))
    -/
    use f
    /-
      case h
      α : Type u_1
      β : Type u_2
      inst✝ : Nonempty β
      s : Set (Prod α β)
      h : Set.InjOn Prod.fst s
      f : α → β
      hf : ∀ ⦃x : Prod α β⦄, Membership.mem s x → Membership.mem s { fst := x.1, snd …
      ⊢ Eq s (Set.graphOn f (Set.image Prod.fst s))
    -/
    rw [graphOn, image_image, EqOn.image_eq_self]
    /-
      case h
      α : Type u_1
      β : Type u_2
      inst✝ : Nonempty β
      s : Set (Prod α β)
      h : Set.InjOn Prod.fst s
      f : α → β
      hf : ∀ ⦃x : Prod α β⦄, Membership.mem s x → Membership.mem s { fst := x.1, snd …
      ⊢ Set.EqOn (fun x => { fst := x.1, snd := f x.1 }) id s
    -/
    exact fun x hx ↦ h (hf hx) hx rfl
    /-
      🎉 no goals
    -/


lemma exists_eq_graphOn [Nonempty β] {s : Set (α × β)} :
    (∃ f t, s = graphOn f t) ↔ InjOn Prod.fst s :=
                                  /-
                                    α : Type u_1
                                    β : Type u_2
                                    inst✝ : Nonempty β
                                    s : Set (Prod α β)
                                    x✝ : Exists fun f => Exists fun t => Eq s (Set.graphOn f t)
                                    f : α → β
                                    t : Set α
                                    hs : Eq s (Set.graphOn f t)
                                    ⊢ Eq s (Set.graphOn f (Set.image Prod.fst s))
                                  -/
  .trans ⟨fun ⟨f, t, hs⟩ ↦ ⟨f, by rw [hs, image_fst_graphOn]⟩, fun ⟨f, hf⟩ ↦ ⟨f, _, hf⟩⟩
                                  /-
                                    🎉 no goals
                                  -/
    exists_eq_graphOn_image_fst


lemma graphOn_prod_graphOn (s : Set α) (t : Set β) (f : α → γ) (g : β → δ) :
    s.graphOn f ×ˢ t.graphOn g = Equiv.prodProdProdComm .. ⁻¹' (s ×ˢ t).graphOn (Prod.map f g) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    s : Set α
    t : Set β
    f : α → γ
    g : β → δ
    ⊢ Eq (SProd.sprod (Set.graphOn f s) (Set.graphOn g t)) (Set.preimage (⇑(Equiv. …
  -/
  aesop
  /-
    🎉 no goals
  -/


lemma graphOn_prod_prodMap (s : Set α) (t : Set β) (f : α → γ) (g : β → δ) :
    (s ×ˢ t).graphOn (Prod.map f g) = Equiv.prodProdProdComm .. ⁻¹' s.graphOn f ×ˢ t.graphOn g := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    s : Set α
    t : Set β
    f : α → γ
    g : β → δ
    ⊢ Eq (Set.graphOn (Prod.map f g) (SProd.sprod s t)) (Set.preimage (⇑(Equiv.pro …
  -/
  aesop
  /-
    🎉 no goals
  -/


theorem SurjOn.subset_range (h : SurjOn f s t) : t ⊆ range f :=
  Subset.trans h <| image_subset_range f s


theorem surjOn_iff_exists_map_subtype :
    SurjOn f s t ↔ ∃ (t' : Set β) (g : s → t'), t ⊆ t' ∧ Surjective g ∧ ∀ x : s, f x = g x :=
  ⟨fun h =>
    ⟨_, (mapsTo_image f s).restrict f s _, h, surjective_mapsTo_image_restrict _ _, fun _ => rfl⟩,
    fun ⟨t', g, htt', hg, hfg⟩ y hy =>
    let ⟨x, hx⟩ := hg ⟨y, htt' hy⟩
                /-
                  α : Type u_1
                  β : Type u_2
                  s : Set α
                  t : Set β
                  f : α → β
                  x✝ : Exists fun t' => Exists fun g => And (HasSubset.Subset t t') (And (Functi …
                  y : β
                  hy : Membership.mem t y
                  t' : Set β
                  g : ↑s → ↑t'
                  htt' : HasSubset.Subset t t'
                  hg : Function.Surjective g
                  hfg : ∀ (x : ↑s), Eq (f ↑x) ↑(g x)
                  x : ↑s
                  hx : Eq (g x) ⟨y, ⋯⟩
                  ⊢ Eq (f ↑x) y
                -/
    ⟨x, x.2, by rw [hfg, hx, Subtype.coe_mk]⟩⟩
                /-
                  🎉 no goals
                -/


theorem surjOn_empty (f : α → β) (s : Set α) : SurjOn f s ∅ :=
  empty_subset _


@[simp] theorem surjOn_empty_iff : SurjOn f ∅ t ↔ t = ∅ := by
  /-
    α : Type u_1
    β : Type u_2
    t : Set β
    f : α → β
    ⊢ Iff (Set.SurjOn f EmptyCollection.emptyCollection t) (Eq t EmptyCollection.e …
  -/
  simp [SurjOn, subset_empty_iff]
  /-
    🎉 no goals
  -/


@[simp] lemma surjOn_singleton : SurjOn f s {b} ↔ b ∈ f '' s := singleton_subset_iff


theorem surjOn_image (f : α → β) (s : Set α) : SurjOn f s (f '' s) :=
  Subset.rfl


theorem SurjOn.comap_nonempty (h : SurjOn f s t) (ht : t.Nonempty) : s.Nonempty :=
  (ht.mono h).of_image


theorem SurjOn.congr (h : SurjOn f₁ s t) (H : EqOn f₁ f₂ s) : SurjOn f₂ s t := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    f₁ f₂ : α → β
    h : Set.SurjOn f₁ s t
    H : Set.EqOn f₁ f₂ s
    ⊢ Set.SurjOn f₂ s t
  -/
  rwa [SurjOn, ← H.image_eq]
  /-
    🎉 no goals
  -/


theorem EqOn.surjOn_iff (h : EqOn f₁ f₂ s) : SurjOn f₁ s t ↔ SurjOn f₂ s t :=
  ⟨fun H => H.congr h, fun H => H.congr h.symm⟩


theorem SurjOn.mono (hs : s₁ ⊆ s₂) (ht : t₁ ⊆ t₂) (hf : SurjOn f s₁ t₂) : SurjOn f s₂ t₁ :=
  Subset.trans ht <| Subset.trans hf <| image_subset _ hs


theorem SurjOn.union (h₁ : SurjOn f s t₁) (h₂ : SurjOn f s t₂) : SurjOn f s (t₁ ∪ t₂) := fun _ hx =>
  hx.elim (fun hx => h₁ hx) fun hx => h₂ hx


theorem SurjOn.union_union (h₁ : SurjOn f s₁ t₁) (h₂ : SurjOn f s₂ t₂) :
    SurjOn f (s₁ ∪ s₂) (t₁ ∪ t₂) :=
  (h₁.mono subset_union_left (Subset.refl _)).union
    (h₂.mono subset_union_right (Subset.refl _))


theorem SurjOn.inter_inter (h₁ : SurjOn f s₁ t₁) (h₂ : SurjOn f s₂ t₂) (h : InjOn f (s₁ ∪ s₂)) :
    SurjOn f (s₁ ∩ s₂) (t₁ ∩ t₂) := by
  /-
    α : Type u_1
    β : Type u_2
    s₁ s₂ : Set α
    t₁ t₂ : Set β
    f : α → β
    h₁ : Set.SurjOn f s₁ t₁
    h₂ : Set.SurjOn f s₂ t₂
    h : Set.InjOn f (Union.union s₁ s₂)
    ⊢ Set.SurjOn f (Inter.inter s₁ s₂) (Inter.inter t₁ t₂)
  -/
  intro y hy
  /-
    α : Type u_1
    β : Type u_2
    s₁ s₂ : Set α
    t₁ t₂ : Set β
    f : α → β
    h₁ : Set.SurjOn f s₁ t₁
    h₂ : Set.SurjOn f s₂ t₂
    h : Set.InjOn f (Union.union s₁ s₂)
    y : β
    hy : Membership.mem (Inter.inter t₁ t₂) y
    ⊢ Membership.mem (Set.image f (Inter.inter s₁ s₂)) y
  -/
  rcases h₁ hy.1 with ⟨x₁, hx₁, rfl⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    s₁ s₂ : Set α
    t₁ t₂ : Set β
    f : α → β
    h₁ : Set.SurjOn f s₁ t₁
    h₂ : Set.SurjOn f s₂ t₂
    h : Set.InjOn f (Union.union s₁ s₂)
    x₁ : α
    hx₁ : Membership.mem s₁ x₁
    hy : Membership.mem (Inter.inter t₁ t₂) (f x₁)
    ⊢ Membership.mem (Set.image f (Inter.inter s₁ s₂)) (f x₁)
  -/
  rcases h₂ hy.2 with ⟨x₂, hx₂, heq⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    s₁ s₂ : Set α
    t₁ t₂ : Set β
    f : α → β
    h₁ : Set.SurjOn f s₁ t₁
    h₂ : Set.SurjOn f s₂ t₂
    h : Set.InjOn f (Union.union s₁ s₂)
    x₁ : α
    hx₁ : Membership.mem s₁ x₁
    hy : Membership.mem (Inter.inter t₁ t₂) (f x₁)
    x₂ : α
    hx₂ : Membership.mem s₂ x₂
    heq : Eq (f x₂) (f x₁)
    ⊢ Membership.mem (Set.image f (Inter.inter s₁ s₂)) (f x₁)
  -/
  obtain rfl : x₁ = x₂ := h (Or.inl hx₁) (Or.inr hx₂) heq.symm
  /-
    case intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    s₁ s₂ : Set α
    t₁ t₂ : Set β
    f : α → β
    h₁ : Set.SurjOn f s₁ t₁
    h₂ : Set.SurjOn f s₂ t₂
    h : Set.InjOn f (Union.union s₁ s₂)
    x₁ : α
    hx₁ : Membership.mem s₁ x₁
    hy : Membership.mem (Inter.inter t₁ t₂) (f x₁)
    hx₂ : Membership.mem s₂ x₁
    heq : Eq (f x₁) (f x₁)
    ⊢ Membership.mem (Set.image f (Inter.inter s₁ s₂)) (f x₁)
  -/
  exact mem_image_of_mem f ⟨hx₁, hx₂⟩
  /-
    🎉 no goals
  -/


theorem SurjOn.inter (h₁ : SurjOn f s₁ t) (h₂ : SurjOn f s₂ t) (h : InjOn f (s₁ ∪ s₂)) :
    SurjOn f (s₁ ∩ s₂) t :=
  inter_self t ▸ h₁.inter_inter h₂ h

-- Porting note: Why does `simp` not call `refl` by itself?

                                                  /-
                                                    α : Type u_1
                                                    s : Set α
                                                    ⊢ Set.SurjOn id s s
                                                  -/
lemma surjOn_id (s : Set α) : SurjOn id s s := by simp [SurjOn, subset_rfl]
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem SurjOn.comp (hg : SurjOn g t p) (hf : SurjOn f s t) : SurjOn (g ∘ f) s p :=
  Subset.trans hg <| Subset.trans (image_subset g hf) <| image_comp g f s ▸ Subset.refl _


lemma SurjOn.iterate {f : α → α} {s : Set α} (h : SurjOn f s s) : ∀ n, SurjOn f^[n] s s
  | 0 => surjOn_id _
  | (n + 1) => (h.iterate n).comp h


lemma SurjOn.comp_left (hf : SurjOn f s t) (g : β → γ) : SurjOn (g ∘ f) s (g '' t) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    s : Set α
    t : Set β
    f : α → β
    hf : Set.SurjOn f s t
    g : β → γ
    ⊢ Set.SurjOn (Function.comp g f) s (Set.image g t)
  -/
  rw [SurjOn, image_comp g f]; exact image_subset _ hf
                               /-
                                 🎉 no goals
                               -/


lemma SurjOn.comp_right {s : Set β} {t : Set γ} (hf : Surjective f) (hg : SurjOn g s t) :
    SurjOn (g ∘ f) (f ⁻¹' s) t := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → β
    g : β → γ
    s : Set β
    t : Set γ
    hf : Function.Surjective f
    hg : Set.SurjOn g s t
    ⊢ Set.SurjOn (Function.comp g f) (Set.preimage f s) t
  -/
  rwa [SurjOn, image_comp g f, image_preimage_eq _ hf]
  /-
    🎉 no goals
  -/


lemma surjOn_of_subsingleton' [Subsingleton β] (f : α → β) (h : t.Nonempty → s.Nonempty) :
    SurjOn f s t :=
  fun _ ha ↦ Subsingleton.mem_iff_nonempty.2 <| (h ⟨_, ha⟩).image _


lemma surjOn_of_subsingleton [Subsingleton α] (f : α → α) (s : Set α) : SurjOn f s s :=
  surjOn_of_subsingleton' _ id


theorem surjective_iff_surjOn_univ : Surjective f ↔ SurjOn f univ univ := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    ⊢ Iff (Function.Surjective f) (Set.SurjOn f Set.univ Set.univ)
  -/
  simp [Surjective, SurjOn, subset_def]
  /-
    🎉 no goals
  -/


theorem surjOn_iff_surjective : SurjOn f s univ ↔ Surjective (s.restrict f) :=
  ⟨fun H b =>
    let ⟨a, as, e⟩ := @H b trivial
    ⟨⟨a, as⟩, e⟩,
    fun H b _ =>
    let ⟨⟨a, as⟩, e⟩ := H b
    ⟨a, as, e⟩⟩


@[simp]
theorem MapsTo.restrict_surjective_iff (h : MapsTo f s t) :
    Surjective (MapsTo.restrict _ _ _ h) ↔ SurjOn f s t := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    f : α → β
    h : Set.MapsTo f s t
    ⊢ Iff (Function.Surjective (Set.MapsTo.restrict f s t h)) (Set.SurjOn f s t)
  -/
  refine ⟨fun h' b hb ↦ ?_, fun h' ⟨b, hb⟩ ↦ ?_⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      s : Set α
      t : Set β
      f : α → β
      h : Set.MapsTo f s t
      h' : Function.Surjective (Set.MapsTo.restrict f s t h)
      b : β
      hb : Membership.mem t b
      ⊢ Membership.mem (Set.image f s) b
    -/
  · obtain ⟨⟨a, ha⟩, ha'⟩ := h' ⟨b, hb⟩
    /-
      case refine_1.intro.mk
      α : Type u_1
      β : Type u_2
      s : Set α
      t : Set β
      f : α → β
      h : Set.MapsTo f s t
      h' : Function.Surjective (Set.MapsTo.restrict f s t h)
      b : β
      hb : Membership.mem t b
      a : α
      ha : Membership.mem s a
      ha' : Eq (Set.MapsTo.restrict f s t h ⟨a, ha⟩) ⟨b, hb⟩
      ⊢ Membership.mem (Set.image f s) b
    -/
    replace ha' : f a = b := by simpa [Subtype.ext_iff] using ha'
    /-
      case refine_1.intro.mk
      α : Type u_1
      β : Type u_2
      s : Set α
      t : Set β
      f : α → β
      h : Set.MapsTo f s t
      h' : Function.Surjective (Set.MapsTo.restrict f s t h)
      b : β
      hb : Membership.mem t b
      a : α
      ha : Membership.mem s a
      ha' : Eq (f a) b
      ⊢ Membership.mem (Set.image f s) b
    -/
    rw [← ha']
    /-
      case refine_1.intro.mk
      α : Type u_1
      β : Type u_2
      s : Set α
      t : Set β
      f : α → β
      h : Set.MapsTo f s t
      h' : Function.Surjective (Set.MapsTo.restrict f s t h)
      b : β
      hb : Membership.mem t b
      a : α
      ha : Membership.mem s a
      ha' : Eq (f a) b
      ⊢ Membership.mem (Set.image f s) (f a)
    -/
    exact mem_image_of_mem f ha
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      s : Set α
      t : Set β
      f : α → β
      h : Set.MapsTo f s t
      h' : Set.SurjOn f s t
      x✝ : ↑t
      b : β
      hb : Membership.mem t b
      ⊢ Exists fun a => Eq (Set.MapsTo.restrict f s t h a) ⟨b, hb⟩
    -/
  · obtain ⟨a, ha, rfl⟩ := h' hb
    /-
      case refine_2.intro.intro
      α : Type u_1
      β : Type u_2
      s : Set α
      t : Set β
      f : α → β
      h : Set.MapsTo f s t
      h' : Set.SurjOn f s t
      x✝ : ↑t
      a : α
      ha : Membership.mem s a
      hb : Membership.mem t (f a)
      ⊢ Exists fun a_1 => Eq (Set.MapsTo.restrict f s t h a_1) ⟨f a, hb⟩
    -/
    exact ⟨⟨a, ha⟩, rfl⟩
    /-
      🎉 no goals
    -/


theorem SurjOn.image_eq_of_mapsTo (h₁ : SurjOn f s t) (h₂ : MapsTo f s t) : f '' s = t :=
  eq_of_subset_of_subset h₂.image_subset h₁


theorem image_eq_iff_surjOn_mapsTo : f '' s = t ↔ s.SurjOn f t ∧ s.MapsTo f t := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    f : α → β
    ⊢ Iff (Eq (Set.image f s) t) (And (Set.SurjOn f s t) (Set.MapsTo f s t))
  -/
  refine ⟨?_, fun h => h.1.image_eq_of_mapsTo h.2⟩
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    f : α → β
    ⊢ Eq (Set.image f s) t → And (Set.SurjOn f s t) (Set.MapsTo f s t)
  -/
  rintro rfl
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    f : α → β
    ⊢ And (Set.SurjOn f s (Set.image f s)) (Set.MapsTo f s (Set.image f s))
  -/
  exact ⟨s.surjOn_image f, s.mapsTo_image f⟩
  /-
    🎉 no goals
  -/


lemma SurjOn.image_preimage (h : Set.SurjOn f s t) (ht : t₁ ⊆ t) : f '' (f ⁻¹' t₁) = t₁ :=
  image_preimage_eq_iff.2 fun _ hx ↦ mem_range_of_mem_image f s <| h <| ht hx


theorem SurjOn.mapsTo_compl (h : SurjOn f s t) (h' : Injective f) : MapsTo f sᶜ tᶜ :=
  fun _ hs ht =>
  let ⟨_, hx', HEq⟩ := h ht
  hs <| h' HEq ▸ hx'


theorem MapsTo.surjOn_compl (h : MapsTo f s t) (h' : Surjective f) : SurjOn f sᶜ tᶜ :=
  h'.forall.2 fun _ ht => (mem_image_of_mem _) fun hs => ht (h hs)


theorem EqOn.cancel_right (hf : s.EqOn (g₁ ∘ f) (g₂ ∘ f)) (hf' : s.SurjOn f t) : t.EqOn g₁ g₂ := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    s : Set α
    t : Set β
    f : α → β
    g₁ g₂ : β → γ
    hf : Set.EqOn (Function.comp g₁ f) (Function.comp g₂ f) s
    hf' : Set.SurjOn f s t
    ⊢ Set.EqOn g₁ g₂ t
  -/
  intro b hb
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    s : Set α
    t : Set β
    f : α → β
    g₁ g₂ : β → γ
    hf : Set.EqOn (Function.comp g₁ f) (Function.comp g₂ f) s
    hf' : Set.SurjOn f s t
    b : β
    hb : Membership.mem t b
    ⊢ Eq (g₁ b) (g₂ b)
  -/
  obtain ⟨a, ha, rfl⟩ := hf' hb
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    s : Set α
    t : Set β
    f : α → β
    g₁ g₂ : β → γ
    hf : Set.EqOn (Function.comp g₁ f) (Function.comp g₂ f) s
    hf' : Set.SurjOn f s t
    a : α
    ha : Membership.mem s a
    hb : Membership.mem t (f a)
    ⊢ Eq (g₁ (f a)) (g₂ (f a))
  -/
  exact hf ha
  /-
    🎉 no goals
  -/


theorem SurjOn.cancel_right (hf : s.SurjOn f t) (hf' : s.MapsTo f t) :
    s.EqOn (g₁ ∘ f) (g₂ ∘ f) ↔ t.EqOn g₁ g₂ :=
  ⟨fun h => h.cancel_right hf, fun h => h.comp_right hf'⟩


theorem eqOn_comp_right_iff : s.EqOn (g₁ ∘ f) (g₂ ∘ f) ↔ (f '' s).EqOn g₁ g₂ :=
  (s.surjOn_image f).cancel_right <| s.mapsTo_image f


theorem SurjOn.forall {p : β → Prop} (hf : s.SurjOn f t) (hf' : s.MapsTo f t) :
    (∀ y ∈ t, p y) ↔ (∀ x ∈ s, p (f x)) :=
  ⟨fun H x hx ↦ H (f x) (hf' hx), fun H _y hy ↦ let ⟨x, hx, hxy⟩ := hf hy; hxy ▸ H x hx⟩


theorem BijOn.mapsTo (h : BijOn f s t) : MapsTo f s t :=
  h.left


theorem BijOn.injOn (h : BijOn f s t) : InjOn f s :=
  h.right.left


theorem BijOn.surjOn (h : BijOn f s t) : SurjOn f s t :=
  h.right.right


theorem BijOn.mk (h₁ : MapsTo f s t) (h₂ : InjOn f s) (h₃ : SurjOn f s t) : BijOn f s t :=
  ⟨h₁, h₂, h₃⟩


theorem bijOn_empty (f : α → β) : BijOn f ∅ ∅ :=
  ⟨mapsTo_empty f ∅, injOn_empty f, surjOn_empty f ∅⟩


@[simp] theorem bijOn_empty_iff_left : BijOn f s ∅ ↔ s = ∅ :=
              /-
                α : Type u_1
                β : Type u_2
                s : Set α
                f : α → β
                h : Set.BijOn f s EmptyCollection.emptyCollection
                ⊢ Eq s EmptyCollection.emptyCollection
              -/
              /-
                🎉 no goals
              -/
  ⟨fun h ↦ by simpa using h.mapsTo, by rintro rfl; exact bijOn_empty f⟩
                                                   /-
                                                     🎉 no goals
                                                   -/


@[simp] theorem bijOn_empty_iff_right : BijOn f ∅ t ↔ t = ∅ :=
              /-
                α : Type u_1
                β : Type u_2
                t : Set β
                f : α → β
                h : Set.BijOn f EmptyCollection.emptyCollection t
                ⊢ Eq t EmptyCollection.emptyCollection
              -/
              /-
                🎉 no goals
              -/
  ⟨fun h ↦ by simpa using h.surjOn, by rintro rfl; exact bijOn_empty f⟩
                                                   /-
                                                     🎉 no goals
                                                   -/


                                                                /-
                                                                  α : Type u_1
                                                                  β : Type u_2
                                                                  f : α → β
                                                                  a : α
                                                                  b : β
                                                                  ⊢ Iff (Set.BijOn f (Singleton.singleton a) (Singleton.singleton b)) (Eq (f a) b)
                                                                -/
@[simp] lemma bijOn_singleton : BijOn f {a} {b} ↔ f a = b := by simp [BijOn, eq_comm]
                                                                /-
                                                                  🎉 no goals
                                                                -/


theorem BijOn.inter_mapsTo (h₁ : BijOn f s₁ t₁) (h₂ : MapsTo f s₂ t₂) (h₃ : s₁ ∩ f ⁻¹' t₂ ⊆ s₂) :
    BijOn f (s₁ ∩ s₂) (t₁ ∩ t₂) :=
  ⟨h₁.mapsTo.inter_inter h₂, h₁.injOn.mono inter_subset_left, fun _ hy =>
    let ⟨x, hx, hxy⟩ := h₁.surjOn hy.1
    ⟨x, ⟨hx, h₃ ⟨hx, hxy.symm.subst hy.2⟩⟩, hxy⟩⟩


theorem MapsTo.inter_bijOn (h₁ : MapsTo f s₁ t₁) (h₂ : BijOn f s₂ t₂) (h₃ : s₂ ∩ f ⁻¹' t₁ ⊆ s₁) :
    BijOn f (s₁ ∩ s₂) (t₁ ∩ t₂) :=
  inter_comm s₂ s₁ ▸ inter_comm t₂ t₁ ▸ h₂.inter_mapsTo h₁ h₃


theorem BijOn.inter (h₁ : BijOn f s₁ t₁) (h₂ : BijOn f s₂ t₂) (h : InjOn f (s₁ ∪ s₂)) :
    BijOn f (s₁ ∩ s₂) (t₁ ∩ t₂) :=
  ⟨h₁.mapsTo.inter_inter h₂.mapsTo, h₁.injOn.mono inter_subset_left,
    h₁.surjOn.inter_inter h₂.surjOn h⟩


theorem BijOn.union (h₁ : BijOn f s₁ t₁) (h₂ : BijOn f s₂ t₂) (h : InjOn f (s₁ ∪ s₂)) :
    BijOn f (s₁ ∪ s₂) (t₁ ∪ t₂) :=
  ⟨h₁.mapsTo.union_union h₂.mapsTo, h, h₁.surjOn.union_union h₂.surjOn⟩


theorem BijOn.subset_range (h : BijOn f s t) : t ⊆ range f :=
  h.surjOn.subset_range


theorem InjOn.bijOn_image (h : InjOn f s) : BijOn f s (f '' s) :=
  BijOn.mk (mapsTo_image f s) h (Subset.refl _)


theorem BijOn.congr (h₁ : BijOn f₁ s t) (h : EqOn f₁ f₂ s) : BijOn f₂ s t :=
  BijOn.mk (h₁.mapsTo.congr h) (h₁.injOn.congr h) (h₁.surjOn.congr h)


theorem EqOn.bijOn_iff (H : EqOn f₁ f₂ s) : BijOn f₁ s t ↔ BijOn f₂ s t :=
  ⟨fun h => h.congr H, fun h => h.congr H.symm⟩


theorem BijOn.image_eq (h : BijOn f s t) : f '' s = t :=
  h.surjOn.image_eq_of_mapsTo h.mapsTo


lemma BijOn.forall {p : β → Prop} (hf : BijOn f s t) : (∀ b ∈ t, p b) ↔ ∀ a ∈ s, p (f a) where
  mp h _ ha := h _ <| hf.mapsTo ha
                   /-
                     α : Type u_1
                     β : Type u_2
                     s : Set α
                     t : Set β
                     f : α → β
                     p : β → Prop
                     hf : Set.BijOn f s t
                     h : ∀ (a : α), Membership.mem s a → p (f a)
                     b : β
                     hb : Membership.mem t b
                     ⊢ p b
                   -/
  mpr h b hb := by obtain ⟨a, ha, rfl⟩ := hf.surjOn hb; exact h _ ha
                                                        /-
                                                          🎉 no goals
                                                        -/


lemma BijOn.exists {p : β → Prop} (hf : BijOn f s t) : (∃ b ∈ t, p b) ↔ ∃ a ∈ s, p (f a) where
           /-
             α : Type u_1
             β : Type u_2
             s : Set α
             t : Set β
             f : α → β
             p : β → Prop
             hf : Set.BijOn f s t
             ⊢ (Exists fun b => And (Membership.mem t b) (p b)) → Exists fun a => And (Memb …
           -/
  mp := by rintro ⟨b, hb, h⟩; obtain ⟨a, ha, rfl⟩ := hf.surjOn hb; exact ⟨a, ha, h⟩
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
            /-
              α : Type u_1
              β : Type u_2
              s : Set α
              t : Set β
              f : α → β
              p : β → Prop
              hf : Set.BijOn f s t
              ⊢ (Exists fun a => And (Membership.mem s a) (p (f a))) → Exists fun b => And ( …
            -/
  mpr := by rintro ⟨a, ha, h⟩; exact ⟨f a, hf.mapsTo ha, h⟩
                               /-
                                 🎉 no goals
                               -/


lemma _root_.Equiv.image_eq_iff_bijOn (e : α ≃ β) : e '' s = t ↔ BijOn e s t :=
  ⟨fun h ↦ ⟨(mapsTo_image e s).mono_right h.subset, e.injective.injOn, h ▸ surjOn_image e s⟩,
  BijOn.image_eq⟩


lemma bijOn_id (s : Set α) : BijOn id s s := ⟨s.mapsTo_id, s.injOn_id, s.surjOn_id⟩


theorem BijOn.comp (hg : BijOn g t p) (hf : BijOn f s t) : BijOn (g ∘ f) s p :=
  BijOn.mk (hg.mapsTo.comp hf.mapsTo) (hg.injOn.comp hf.injOn hf.mapsTo) (hg.surjOn.comp hf.surjOn)


lemma BijOn.iterate {f : α → α} {s : Set α} (h : BijOn f s s) : ∀ n, BijOn f^[n] s s
  | 0 => s.bijOn_id
  | (n + 1) => (h.iterate n).comp h


lemma bijOn_of_subsingleton' [Subsingleton α] [Subsingleton β] (f : α → β)
    (h : s.Nonempty ↔ t.Nonempty) : BijOn f s t :=
  ⟨mapsTo_of_subsingleton' _ h.1, injOn_of_subsingleton _ _, surjOn_of_subsingleton' _ h.2⟩


lemma bijOn_of_subsingleton [Subsingleton α] (f : α → α) (s : Set α) : BijOn f s s :=
  bijOn_of_subsingleton' _ Iff.rfl


theorem BijOn.bijective (h : BijOn f s t) : Bijective (h.mapsTo.restrict f s t) :=
  ⟨fun x y h' => Subtype.ext <| h.injOn x.2 y.2 <| Subtype.ext_iff.1 h', fun ⟨_, hy⟩ =>
    let ⟨x, hx, hxy⟩ := h.surjOn hy
    ⟨⟨x, hx⟩, Subtype.eq hxy⟩⟩


theorem bijective_iff_bijOn_univ : Bijective f ↔ BijOn f univ univ :=
  Iff.intro
    (fun h =>
      let ⟨inj, surj⟩ := h
      ⟨mapsTo_univ f _, inj.injOn, Iff.mp surjective_iff_surjOn_univ surj⟩)
    fun h =>
    let ⟨_map, inj, surj⟩ := h
    ⟨Iff.mpr injective_iff_injOn_univ inj, Iff.mpr surjective_iff_surjOn_univ surj⟩


alias ⟨_root_.Function.Bijective.bijOn_univ, _⟩ := bijective_iff_bijOn_univ


theorem BijOn.compl (hst : BijOn f s t) (hf : Bijective f) : BijOn f sᶜ tᶜ :=
  ⟨hst.surjOn.mapsTo_compl hf.1, hf.1.injOn, hst.mapsTo.surjOn_compl hf.2⟩


theorem BijOn.subset_right {r : Set β} (hf : BijOn f s t) (hrt : r ⊆ t) :
    BijOn f (s ∩ f ⁻¹' r) r := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    f : α → β
    r : Set β
    hf : Set.BijOn f s t
    hrt : HasSubset.Subset r t
    ⊢ Set.BijOn f (Inter.inter s (Set.preimage f r)) r
  -/
  refine ⟨inter_subset_right, hf.injOn.mono inter_subset_left, fun x hx ↦ ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    f : α → β
    r : Set β
    hf : Set.BijOn f s t
    hrt : HasSubset.Subset r t
    x : β
    hx : Membership.mem r x
    ⊢ Membership.mem (Set.image f (Inter.inter s (Set.preimage f r))) x
  -/
  obtain ⟨y, hy, rfl⟩ := hf.surjOn (hrt hx)
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    f : α → β
    r : Set β
    hf : Set.BijOn f s t
    hrt : HasSubset.Subset r t
    y : α
    hy : Membership.mem s y
    hx : Membership.mem r (f y)
    ⊢ Membership.mem (Set.image f (Inter.inter s (Set.preimage f r))) (f y)
  -/
  exact ⟨y, ⟨hy, hx⟩, rfl⟩
  /-
    🎉 no goals
  -/


theorem BijOn.subset_left {r : Set α} (hf : BijOn f s t) (hrs : r ⊆ s) :
    BijOn f r (f '' r) :=
  (hf.injOn.mono hrs).bijOn_image


theorem BijOn.insert_iff (ha : a ∉ s) (hfa : f a ∉ t) :
    BijOn f (insert a s) (insert (f a) t) ↔ BijOn f s t where
  mp h := by
    /-
      α : Type u_1
      β : Type u_2
      s : Set α
      t : Set β
      f : α → β
      a : α
      ha : Not (Membership.mem s a)
      hfa : Not (Membership.mem t (f a))
      h : Set.BijOn f (Insert.insert a s) (Insert.insert (f a) t)
      ⊢ Set.BijOn f s t
    -/
    have := congrArg (· \ {f a}) (image_insert_eq ▸ h.image_eq)
    /-
      α : Type u_1
      β : Type u_2
      s : Set α
      t : Set β
      f : α → β
      a : α
      ha : Not (Membership.mem s a)
      hfa : Not (Membership.mem t (f a))
      h : Set.BijOn f (Insert.insert a s) (Insert.insert (f a) t)
      this : Eq ((fun x => SDiff.sdiff x (Singleton.singleton (f a))) (Insert.insert …
      ⊢ Set.BijOn f s t
    -/
    simp only [mem_singleton_iff, insert_diff_of_mem] at this
    /-
      α : Type u_1
      β : Type u_2
      s : Set α
      t : Set β
      f : α → β
      a : α
      ha : Not (Membership.mem s a)
      hfa : Not (Membership.mem t (f a))
      h : Set.BijOn f (Insert.insert a s) (Insert.insert (f a) t)
      this : Eq (SDiff.sdiff (Set.image f s) (Singleton.singleton (f a))) (SDiff.sdi …
      ⊢ Set.BijOn f s t
    -/
    rw [diff_singleton_eq_self hfa, diff_singleton_eq_self] at this
    · exact ⟨by simp [← this, mapsTo'], h.injOn.mono (subset_insert ..),
        by simp [← this, surjOn_image]⟩
    /-
      α : Type u_1
      β : Type u_2
      s : Set α
      t : Set β
      f : α → β
      a : α
      ha : Not (Membership.mem s a)
      hfa : Not (Membership.mem t (f a))
      h : Set.BijOn f (Insert.insert a s) (Insert.insert (f a) t)
      this : Eq (SDiff.sdiff (Set.image f s) (Singleton.singleton (f a))) t
      ⊢ Not (Membership.mem (Set.image f s) (f a))
    -/
    simp only [mem_image, not_exists, not_and]
    /-
      α : Type u_1
      β : Type u_2
      s : Set α
      t : Set β
      f : α → β
      a : α
      ha : Not (Membership.mem s a)
      hfa : Not (Membership.mem t (f a))
      h : Set.BijOn f (Insert.insert a s) (Insert.insert (f a) t)
      this : Eq (SDiff.sdiff (Set.image f s) (Singleton.singleton (f a))) t
      ⊢ ∀ (x : α), Membership.mem s x → Not (Eq (f x) (f a))
    -/
    intro x hx
    /-
      α : Type u_1
      β : Type u_2
      s : Set α
      t : Set β
      f : α → β
      a : α
      ha : Not (Membership.mem s a)
      hfa : Not (Membership.mem t (f a))
      h : Set.BijOn f (Insert.insert a s) (Insert.insert (f a) t)
      this : Eq (SDiff.sdiff (Set.image f s) (Singleton.singleton (f a))) t
      x : α
      hx : Membership.mem s x
      ⊢ Not (Eq (f x) (f a))
    -/
    rw [h.injOn.eq_iff (by simp [hx]) (by simp)]
    /-
      α : Type u_1
      β : Type u_2
      s : Set α
      t : Set β
      f : α → β
      a : α
      ha : Not (Membership.mem s a)
      hfa : Not (Membership.mem t (f a))
      h : Set.BijOn f (Insert.insert a s) (Insert.insert (f a) t)
      this : Eq (SDiff.sdiff (Set.image f s) (Singleton.singleton (f a))) t
      x : α
      hx : Membership.mem s x
      ⊢ Not (Eq x a)
    -/
    exact ha ∘ (· ▸ hx)
    /-
      🎉 no goals
    -/
  mpr h := by
    /-
      α : Type u_1
      β : Type u_2
      s : Set α
      t : Set β
      f : α → β
      a : α
      ha : Not (Membership.mem s a)
      hfa : Not (Membership.mem t (f a))
      h : Set.BijOn f s t
      ⊢ Set.BijOn f (Insert.insert a s) (Insert.insert (f a) t)
    -/
    repeat rw [insert_eq]
    /-
      α : Type u_1
      β : Type u_2
      s : Set α
      t : Set β
      f : α → β
      a : α
      ha : Not (Membership.mem s a)
      hfa : Not (Membership.mem t (f a))
      h : Set.BijOn f s t
      ⊢ Set.BijOn f (Union.union (Singleton.singleton a) s) (Union.union (Singleton. …
    -/
    refine (bijOn_singleton.mpr rfl).union h ?_
    simp only [singleton_union, injOn_insert fun x ↦ (hfa (h.mapsTo x)), h.injOn, mem_image,
      not_exists, not_and, true_and]
    /-
      α : Type u_1
      β : Type u_2
      s : Set α
      t : Set β
      f : α → β
      a : α
      ha : Not (Membership.mem s a)
      hfa : Not (Membership.mem t (f a))
      h : Set.BijOn f s t
      ⊢ ∀ (x : α), Membership.mem s x → Not (Eq (f x) (f a))
    -/
    exact fun _ hx h₂ ↦ hfa (h₂ ▸ h.mapsTo hx)
    /-
      🎉 no goals
    -/


theorem BijOn.insert (h₁ : BijOn f s t) (h₂ : f a ∉ t) :
    BijOn f (insert a s) (insert (f a) t) :=
  (insert_iff (h₂ <| h₁.mapsTo ·) h₂).mpr h₁


theorem BijOn.sdiff_singleton (h₁ : BijOn f s t) (h₂ : a ∈ s) :
    BijOn f (s \ {a}) (t \ {f a}) := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    f : α → β
    a : α
    h₁ : Set.BijOn f s t
    h₂ : Membership.mem s a
    ⊢ Set.BijOn f (SDiff.sdiff s (Singleton.singleton a)) (SDiff.sdiff t (Singleto …
  -/
  convert h₁.subset_left diff_subset
  /-
    case h.e'_5
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    f : α → β
    a : α
    h₁ : Set.BijOn f s t
    h₂ : Membership.mem s a
    ⊢ Eq (SDiff.sdiff t (Singleton.singleton (f a))) (Set.image f (SDiff.sdiff s ( …
  -/
  simp [h₁.injOn.image_diff, h₁.image_eq, h₂, inter_eq_self_of_subset_right]
  /-
    🎉 no goals
  -/


theorem eqOn (h : LeftInvOn f' f s) : EqOn (f' ∘ f) id s :=
  h


theorem eq (h : LeftInvOn f' f s) {x} (hx : x ∈ s) : f' (f x) = x :=
  h hx


theorem congr_left (h₁ : LeftInvOn f₁' f s) {t : Set β} (h₁' : MapsTo f s t)
    (heq : EqOn f₁' f₂' t) : LeftInvOn f₂' f s := fun _ hx => heq (h₁' hx) ▸ h₁ hx


theorem congr_right (h₁ : LeftInvOn f₁' f₁ s) (heq : EqOn f₁ f₂ s) : LeftInvOn f₁' f₂ s :=
  fun _ hx => heq hx ▸ h₁ hx


theorem injOn (h : LeftInvOn f₁' f s) : InjOn f s := fun x₁ h₁ x₂ h₂ heq =>
  calc
    x₁ = f₁' (f x₁) := Eq.symm <| h h₁
    _ = f₁' (f x₂) := congr_arg f₁' heq
    _ = x₂ := h h₂


theorem surjOn (h : LeftInvOn f' f s) (hf : MapsTo f s t) : SurjOn f' t s := fun x hx =>
  ⟨f x, hf hx, h hx⟩


theorem mapsTo (h : LeftInvOn f' f s) (hf : SurjOn f s t) :
    MapsTo f' t s := fun y hy => by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    f : α → β
    f' : β → α
    h : Set.LeftInvOn f' f s
    hf : Set.SurjOn f s t
    y : β
    hy : Membership.mem t y
    ⊢ Membership.mem s (f' y)
  -/
  let ⟨x, hs, hx⟩ := hf hy
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    f : α → β
    f' : β → α
    h : Set.LeftInvOn f' f s
    hf : Set.SurjOn f s t
    y : β
    hy : Membership.mem t y
    x : α
    hs : Membership.mem s x
    hx : Eq (f x) y
    ⊢ Membership.mem s (f' y)
  -/
  rwa [← hx, h hs]
  /-
    🎉 no goals
  -/


lemma _root_.Set.leftInvOn_id (s : Set α) : LeftInvOn id id s := fun _ _ ↦ rfl


theorem comp (hf' : LeftInvOn f' f s) (hg' : LeftInvOn g' g t) (hf : MapsTo f s t) :
    LeftInvOn (f' ∘ g') (g ∘ f) s := fun x h =>
  calc
    (f' ∘ g') ((g ∘ f) x) = f' (f x) := congr_arg f' (hg' (hf h))
    _ = x := hf' h


theorem mono (hf : LeftInvOn f' f s) (ht : s₁ ⊆ s) : LeftInvOn f' f s₁ := fun _ hx =>
  hf (ht hx)


theorem image_inter' (hf : LeftInvOn f' f s) : f '' (s₁ ∩ s) = f' ⁻¹' s₁ ∩ f '' s := by
  /-
    α : Type u_1
    β : Type u_2
    s s₁ : Set α
    f : α → β
    f' : β → α
    hf : Set.LeftInvOn f' f s
    ⊢ Eq (Set.image f (Inter.inter s₁ s)) (Inter.inter (Set.preimage f' s₁) (Set.i …
  -/
  apply Subset.antisymm
    /-
      case h₁
      α : Type u_1
      β : Type u_2
      s s₁ : Set α
      f : α → β
      f' : β → α
      hf : Set.LeftInvOn f' f s
      ⊢ HasSubset.Subset (Set.image f (Inter.inter s₁ s)) (Inter.inter (Set.preimage …
    -/
  · rintro _ ⟨x, ⟨h₁, h⟩, rfl⟩
    /-
      case h₁.intro.intro.intro
      α : Type u_1
      β : Type u_2
      s s₁ : Set α
      f : α → β
      f' : β → α
      hf : Set.LeftInvOn f' f s
      x : α
      h₁ : Membership.mem s₁ x
      h : Membership.mem s x
      ⊢ Membership.mem (Inter.inter (Set.preimage f' s₁) (Set.image f s)) (f x)
    -/
    exact ⟨by rwa [mem_preimage, hf h], mem_image_of_mem _ h⟩
    /-
      🎉 no goals
    -/
    /-
      case h₂
      α : Type u_1
      β : Type u_2
      s s₁ : Set α
      f : α → β
      f' : β → α
      hf : Set.LeftInvOn f' f s
      ⊢ HasSubset.Subset (Inter.inter (Set.preimage f' s₁) (Set.image f s)) (Set.ima …
    -/
  · rintro _ ⟨h₁, ⟨x, h, rfl⟩⟩
    /-
      case h₂.intro.intro.intro
      α : Type u_1
      β : Type u_2
      s s₁ : Set α
      f : α → β
      f' : β → α
      hf : Set.LeftInvOn f' f s
      x : α
      h : Membership.mem s x
      h₁ : Membership.mem (Set.preimage f' s₁) (f x)
      ⊢ Membership.mem (Set.image f (Inter.inter s₁ s)) (f x)
    -/
    exact mem_image_of_mem _ ⟨by rwa [← hf h], h⟩
    /-
      🎉 no goals
    -/


theorem image_inter (hf : LeftInvOn f' f s) :
    f '' (s₁ ∩ s) = f' ⁻¹' (s₁ ∩ s) ∩ f '' s := by
  /-
    α : Type u_1
    β : Type u_2
    s s₁ : Set α
    f : α → β
    f' : β → α
    hf : Set.LeftInvOn f' f s
    ⊢ Eq (Set.image f (Inter.inter s₁ s)) (Inter.inter (Set.preimage f' (Inter.int …
  -/
  rw [hf.image_inter']
  /-
    α : Type u_1
    β : Type u_2
    s s₁ : Set α
    f : α → β
    f' : β → α
    hf : Set.LeftInvOn f' f s
    ⊢ Eq (Inter.inter (Set.preimage f' s₁) (Set.image f s)) (Inter.inter (Set.prei …
  -/
  refine Subset.antisymm ?_ (inter_subset_inter_left _ (preimage_mono inter_subset_left))
  /-
    α : Type u_1
    β : Type u_2
    s s₁ : Set α
    f : α → β
    f' : β → α
    hf : Set.LeftInvOn f' f s
    ⊢ HasSubset.Subset (Inter.inter (Set.preimage f' s₁) (Set.image f s)) (Inter.i …
  -/
  rintro _ ⟨h₁, x, hx, rfl⟩; exact ⟨⟨h₁, by rwa [hf hx]⟩, mem_image_of_mem _ hx⟩
                             /-
                               🎉 no goals
                             -/


theorem image_image (hf : LeftInvOn f' f s) : f' '' (f '' s) = s := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    f : α → β
    f' : β → α
    hf : Set.LeftInvOn f' f s
    ⊢ Eq (Set.image f' (Set.image f s)) s
  -/
  rw [Set.image_image, image_congr hf, image_id']
  /-
    🎉 no goals
  -/


theorem image_image' (hf : LeftInvOn f' f s) (hs : s₁ ⊆ s) : f' '' (f '' s₁) = s₁ :=
  (hf.mono hs).image_image


theorem eqOn (h : RightInvOn f' f t) : EqOn (f ∘ f') id t :=
  h


theorem eq (h : RightInvOn f' f t) {y} (hy : y ∈ t) : f (f' y) = y :=
  h hy


theorem _root_.Set.LeftInvOn.rightInvOn_image (h : LeftInvOn f' f s) : RightInvOn f' f (f '' s) :=
  fun _y ⟨_x, hx, heq⟩ => heq ▸ (congr_arg f <| h.eq hx)


theorem congr_left (h₁ : RightInvOn f₁' f t) (heq : EqOn f₁' f₂' t) :
    RightInvOn f₂' f t :=
  h₁.congr_right heq


theorem congr_right (h₁ : RightInvOn f' f₁ t) (hg : MapsTo f' t s) (heq : EqOn f₁ f₂ s) :
    RightInvOn f' f₂ t :=
  LeftInvOn.congr_left h₁ hg heq


theorem surjOn (hf : RightInvOn f' f t) (hf' : MapsTo f' t s) : SurjOn f s t :=
  LeftInvOn.surjOn hf hf'


theorem mapsTo (h : RightInvOn f' f t) (hf : SurjOn f' t s) : MapsTo f s t :=
  LeftInvOn.mapsTo h hf


lemma _root_.Set.rightInvOn_id (s : Set α) : RightInvOn id id s := fun _ _ ↦ rfl


theorem comp (hf : RightInvOn f' f t) (hg : RightInvOn g' g p) (g'pt : MapsTo g' p t) :
    RightInvOn (f' ∘ g') (g ∘ f) p :=
  LeftInvOn.comp hg hf g'pt


theorem mono (hf : RightInvOn f' f t) (ht : t₁ ⊆ t) : RightInvOn f' f t₁ :=
  LeftInvOn.mono hf ht

theorem InjOn.rightInvOn_of_leftInvOn (hf : InjOn f s) (hf' : LeftInvOn f f' t)
    (h₁ : MapsTo f s t) (h₂ : MapsTo f' t s) : RightInvOn f f' s := fun _ h =>
  hf (h₂ <| h₁ h) h (hf' (h₁ h))


theorem eqOn_of_leftInvOn_of_rightInvOn (h₁ : LeftInvOn f₁' f s) (h₂ : RightInvOn f₂' f t)
    (h : MapsTo f₂' t s) : EqOn f₁' f₂' t := fun y hy =>
  calc
    f₁' y = (f₁' ∘ f ∘ f₂') y := congr_arg f₁' (h₂ hy).symm
    _ = f₂' y := h₁ (h hy)


theorem SurjOn.leftInvOn_of_rightInvOn (hf : SurjOn f s t) (hf' : RightInvOn f f' s) :
    LeftInvOn f f' t := fun y hy => by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    f : α → β
    f' : β → α
    hf : Set.SurjOn f s t
    hf' : Set.RightInvOn f f' s
    y : β
    hy : Membership.mem t y
    ⊢ Eq (f (f' y)) y
  -/
  let ⟨x, hx, heq⟩ := hf hy
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    f : α → β
    f' : β → α
    hf : Set.SurjOn f s t
    hf' : Set.RightInvOn f f' s
    y : β
    hy : Membership.mem t y
    x : α
    hx : Membership.mem s x
    heq : Eq (f x) y
    ⊢ Eq (f (f' y)) y
  -/
  rw [← heq, hf' hx]
  /-
    🎉 no goals
  -/


lemma _root_.Set.invOn_id (s : Set α) : InvOn id id s s := ⟨s.leftInvOn_id, s.rightInvOn_id⟩


lemma comp (hf : InvOn f' f s t) (hg : InvOn g' g t p) (fst : MapsTo f s t)
    (g'pt : MapsTo g' p t) :
    InvOn (f' ∘ g') (g ∘ f) s p :=
  ⟨hf.1.comp hg.1 fst, hf.2.comp hg.2 g'pt⟩


@[symm]
theorem symm (h : InvOn f' f s t) : InvOn f f' t s :=
  ⟨h.right, h.left⟩


theorem mono (h : InvOn f' f s t) (hs : s₁ ⊆ s) (ht : t₁ ⊆ t) : InvOn f' f s₁ t₁ :=
  ⟨h.1.mono hs, h.2.mono ht⟩


/-- If functions `f'` and `f` are inverse on `s` and `t`, `f` maps `s` into `t`, and `f'` maps `t`
into `s`, then `f` is a bijection between `s` and `t`. The `mapsTo` arguments can be deduced from
`surjOn` statements using `LeftInvOn.mapsTo` and `RightInvOn.mapsTo`. -/
theorem bijOn (h : InvOn f' f s t) (hf : MapsTo f s t) (hf' : MapsTo f' t s) : BijOn f s t :=
  ⟨hf, h.left.injOn, h.right.surjOn hf'⟩


/-- Construct the inverse for a function `f` on domain `s`. This function is a right inverse of `f`
on `f '' s`. For a computable version, see `Function.Embedding.invOfMemRange`. -/
noncomputable def invFunOn [Nonempty α] (f : α → β) (s : Set α) (b : β) : α :=
  if h : ∃ a, a ∈ s ∧ f a = b then Classical.choose h else Classical.choice ‹Nonempty α›


theorem invFunOn_pos (h : ∃ a ∈ s, f a = b) : invFunOn f s b ∈ s ∧ f (invFunOn f s b) = b := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    f : α → β
    b : β
    inst✝ : Nonempty α
    h : Exists fun a => And (Membership.mem s a) (Eq (f a) b)
    ⊢ And (Membership.mem s (Function.invFunOn f s b)) (Eq (f (Function.invFunOn f …
  -/
  rw [invFunOn, dif_pos h]
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    f : α → β
    b : β
    inst✝ : Nonempty α
    h : Exists fun a => And (Membership.mem s a) (Eq (f a) b)
    ⊢ And (Membership.mem s (Classical.choose h)) (Eq (f (Classical.choose h)) b)
  -/
  exact Classical.choose_spec h
  /-
    🎉 no goals
  -/


theorem invFunOn_mem (h : ∃ a ∈ s, f a = b) : invFunOn f s b ∈ s :=
  (invFunOn_pos h).left


theorem invFunOn_eq (h : ∃ a ∈ s, f a = b) : f (invFunOn f s b) = b :=
  (invFunOn_pos h).right


theorem invFunOn_neg (h : ¬∃ a ∈ s, f a = b) : invFunOn f s b = Classical.choice ‹Nonempty α› := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    f : α → β
    b : β
    inst✝ : Nonempty α
    h : Not (Exists fun a => And (Membership.mem s a) (Eq (f a) b))
    ⊢ Eq (Function.invFunOn f s b) (Classical.choice inst✝)
  -/
  rw [invFunOn, dif_neg h]
  /-
    🎉 no goals
  -/


@[simp]
theorem invFunOn_apply_mem (h : a ∈ s) : invFunOn f s (f a) ∈ s :=
  invFunOn_mem ⟨a, h, rfl⟩


theorem invFunOn_apply_eq (h : a ∈ s) : f (invFunOn f s (f a)) = f a :=
  invFunOn_eq ⟨a, h, rfl⟩


theorem InjOn.leftInvOn_invFunOn [Nonempty α] (h : InjOn f s) : LeftInvOn (invFunOn f s) f s :=
  fun _a ha => h (invFunOn_apply_mem ha) ha (invFunOn_apply_eq ha)


theorem InjOn.invFunOn_image [Nonempty α] (h : InjOn f s₂) (ht : s₁ ⊆ s₂) :
    invFunOn f s₂ '' (f '' s₁) = s₁ :=
  h.leftInvOn_invFunOn.image_image' ht


theorem _root_.Function.leftInvOn_invFunOn_of_subset_image_image [Nonempty α]
    (h : s ⊆ (invFunOn f s) '' (f '' s)) : LeftInvOn (invFunOn f s) f s :=
  fun x hx ↦ by
    /-
      α : Type u_1
      β : Type u_2
      s : Set α
      f : α → β
      inst✝ : Nonempty α
      h : HasSubset.Subset s (Set.image (Function.invFunOn f s) (Set.image f s))
      x : α
      hx : Membership.mem s x
      ⊢ Eq (Function.invFunOn f s (f x)) x
    -/
    obtain ⟨-, ⟨x, hx', rfl⟩, rfl⟩ := h hx
    /-
      case intro.intro.intro.intro
      α : Type u_1
      β : Type u_2
      s : Set α
      f : α → β
      inst✝ : Nonempty α
      h : HasSubset.Subset s (Set.image (Function.invFunOn f s) (Set.image f s))
      x : α
      hx' : Membership.mem s x
      hx : Membership.mem s (Function.invFunOn f s (f x))
      ⊢ Eq (Function.invFunOn f s (f (Function.invFunOn f s (f x)))) (Function.invFu …
    -/
    rw [invFunOn_apply_eq (f := f) hx']
    /-
      🎉 no goals
    -/


theorem injOn_iff_invFunOn_image_image_eq_self [Nonempty α] :
    InjOn f s ↔ (invFunOn f s) '' (f '' s) = s :=
  ⟨fun h ↦ h.invFunOn_image Subset.rfl, fun h ↦
    (Function.leftInvOn_invFunOn_of_subset_image_image h.symm.subset).injOn⟩


theorem _root_.Function.invFunOn_injOn_image [Nonempty α] (f : α → β) (s : Set α) :
    Set.InjOn (invFunOn f s) (f '' s) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : Nonempty α
    f : α → β
    s : Set α
    ⊢ Set.InjOn (Function.invFunOn f s) (Set.image f s)
  -/
  rintro _ ⟨x, hx, rfl⟩ _ ⟨x', hx', rfl⟩ he
  /-
    case intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    inst✝ : Nonempty α
    f : α → β
    s : Set α
    x : α
    hx : Membership.mem s x
    x' : α
    hx' : Membership.mem s x'
    he : Eq (Function.invFunOn f s (f x)) (Function.invFunOn f s (f x'))
    ⊢ Eq (f x) (f x')
  -/
  rw [← invFunOn_apply_eq (f := f) hx, he, invFunOn_apply_eq (f := f) hx']
  /-
    🎉 no goals
  -/


theorem _root_.Function.invFunOn_image_image_subset [Nonempty α] (f : α → β) (s : Set α) :
    (invFunOn f s) '' (f '' s) ⊆ s := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : Nonempty α
    f : α → β
    s : Set α
    ⊢ HasSubset.Subset (Set.image (Function.invFunOn f s) (Set.image f s)) s
  -/
  rintro _ ⟨_, ⟨x,hx,rfl⟩, rfl⟩; exact invFunOn_apply_mem hx
                                 /-
                                   🎉 no goals
                                 -/


theorem SurjOn.rightInvOn_invFunOn [Nonempty α] (h : SurjOn f s t) :
    RightInvOn (invFunOn f s) f t := fun _y hy => invFunOn_eq <| h hy


theorem BijOn.invOn_invFunOn [Nonempty α] (h : BijOn f s t) : InvOn (invFunOn f s) f s t :=
  ⟨h.injOn.leftInvOn_invFunOn, h.surjOn.rightInvOn_invFunOn⟩


theorem SurjOn.invOn_invFunOn [Nonempty α] (h : SurjOn f s t) :
    InvOn (invFunOn f s) f (invFunOn f s '' t) t := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    f : α → β
    inst✝ : Nonempty α
    h : Set.SurjOn f s t
    ⊢ Set.InvOn (Function.invFunOn f s) f (Set.image (Function.invFunOn f s) t) t
  -/
  refine ⟨?_, h.rightInvOn_invFunOn⟩
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    f : α → β
    inst✝ : Nonempty α
    h : Set.SurjOn f s t
    ⊢ Set.LeftInvOn (Function.invFunOn f s) f (Set.image (Function.invFunOn f s) t)
  -/
  rintro _ ⟨y, hy, rfl⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    f : α → β
    inst✝ : Nonempty α
    h : Set.SurjOn f s t
    y : β
    hy : Membership.mem t y
    ⊢ Eq (Function.invFunOn f s (f (Function.invFunOn f s y))) (Function.invFunOn  …
  -/
  rw [h.rightInvOn_invFunOn hy]
  /-
    🎉 no goals
  -/


theorem SurjOn.mapsTo_invFunOn [Nonempty α] (h : SurjOn f s t) : MapsTo (invFunOn f s) t s :=
  fun _y hy => mem_preimage.2 <| invFunOn_mem <| h hy


/-- This lemma is a special case of `rightInvOn_invFunOn.image_image'`; it may make more sense
to use the other lemma directly in an application. -/
theorem SurjOn.image_invFunOn_image_of_subset [Nonempty α] {r : Set β} (hf : SurjOn f s t)
    (hrt : r ⊆ t) : f '' (f.invFunOn s '' r) = r :=
  hf.rightInvOn_invFunOn.image_image' hrt


/-- This lemma is a special case of `rightInvOn_invFunOn.image_image`; it may make more sense
to use the other lemma directly in an application. -/
theorem SurjOn.image_invFunOn_image [Nonempty α] (hf : SurjOn f s t) :
    f '' (f.invFunOn s '' t) = t :=
  hf.rightInvOn_invFunOn.image_image


theorem SurjOn.bijOn_subset [Nonempty α] (h : SurjOn f s t) : BijOn f (invFunOn f s '' t) t := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    f : α → β
    inst✝ : Nonempty α
    h : Set.SurjOn f s t
    ⊢ Set.BijOn f (Set.image (Function.invFunOn f s) t) t
  -/
  refine h.invOn_invFunOn.bijOn ?_ (mapsTo_image _ _)
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    f : α → β
    inst✝ : Nonempty α
    h : Set.SurjOn f s t
    ⊢ Set.MapsTo f (Set.image (Function.invFunOn f s) t) t
  -/
  rintro _ ⟨y, hy, rfl⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    f : α → β
    inst✝ : Nonempty α
    h : Set.SurjOn f s t
    y : β
    hy : Membership.mem t y
    ⊢ Membership.mem t (f (Function.invFunOn f s y))
  -/
  rwa [h.rightInvOn_invFunOn hy]
  /-
    🎉 no goals
  -/


theorem surjOn_iff_exists_bijOn_subset : SurjOn f s t ↔ ∃ s' ⊆ s, BijOn f s' t := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    f : α → β
    ⊢ Iff (Set.SurjOn f s t) (Exists fun s' => And (HasSubset.Subset s' s) (Set.Bi …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      β : Type u_2
      s : Set α
      t : Set β
      f : α → β
      ⊢ Set.SurjOn f s t → Exists fun s' => And (HasSubset.Subset s' s) (Set.BijOn f …
    -/
  · rcases eq_empty_or_nonempty t with (rfl | ht)
      /-
        case mp.inl
        α : Type u_1
        β : Type u_2
        s : Set α
        f : α → β
        ⊢ Set.SurjOn f s EmptyCollection.emptyCollection → Exists fun s' => And (HasSu …
      -/
    · exact fun _ => ⟨∅, empty_subset _, bijOn_empty f⟩
      /-
        🎉 no goals
      -/
      /-
        case mp.inr
        α : Type u_1
        β : Type u_2
        s : Set α
        t : Set β
        f : α → β
        ht : t.Nonempty
        ⊢ Set.SurjOn f s t → Exists fun s' => And (HasSubset.Subset s' s) (Set.BijOn f …
      -/
    · intro h
      /-
        case mp.inr
        α : Type u_1
        β : Type u_2
        s : Set α
        t : Set β
        f : α → β
        ht : t.Nonempty
        h : Set.SurjOn f s t
        ⊢ Exists fun s' => And (HasSubset.Subset s' s) (Set.BijOn f s' t)
      -/
      haveI : Nonempty α := ⟨Classical.choose (h.comap_nonempty ht)⟩
      /-
        case mp.inr
        α : Type u_1
        β : Type u_2
        s : Set α
        t : Set β
        f : α → β
        ht : t.Nonempty
        h : Set.SurjOn f s t
        this : Nonempty α
        ⊢ Exists fun s' => And (HasSubset.Subset s' s) (Set.BijOn f s' t)
      -/
      exact ⟨_, h.mapsTo_invFunOn.image_subset, h.bijOn_subset⟩
      /-
        🎉 no goals
      -/
    /-
      case mpr
      α : Type u_1
      β : Type u_2
      s : Set α
      t : Set β
      f : α → β
      ⊢ (Exists fun s' => And (HasSubset.Subset s' s) (Set.BijOn f s' t)) → Set.Surj …
    -/
  · rintro ⟨s', hs', hfs'⟩
    /-
      case mpr.intro.intro
      α : Type u_1
      β : Type u_2
      s : Set α
      t : Set β
      f : α → β
      s' : Set α
      hs' : HasSubset.Subset s' s
      hfs' : Set.BijOn f s' t
      ⊢ Set.SurjOn f s t
    -/
    exact hfs'.surjOn.mono hs' (Subset.refl _)
    /-
      🎉 no goals
    -/


alias ⟨SurjOn.exists_bijOn_subset, _⟩ := Set.surjOn_iff_exists_bijOn_subset


lemma exists_subset_bijOn : ∃ s' ⊆ s, BijOn f s' (f '' s) :=
  surjOn_iff_exists_bijOn_subset.mp (surjOn_image f s)


lemma exists_image_eq_and_injOn : ∃ u, f '' u =  f '' s ∧ InjOn f u :=
  let ⟨u, _, hfu⟩ := exists_subset_bijOn s f
  ⟨u, hfu.image_eq, hfu.injOn⟩


lemma exists_image_eq_injOn_of_subset_range (ht : t ⊆ range f) :
    ∃ s, f '' s = t ∧ InjOn f s :=
  image_preimage_eq_of_subset ht ▸ exists_image_eq_and_injOn _ _


/-- If `f` maps `s` bijectively to `t` and a set `t'` is contained in the image of some `s₁ ⊇ s`,
then `s₁` has a subset containing `s` that `f` maps bijectively to `t'`.-/
theorem BijOn.exists_extend_of_subset {t' : Set β} (h : BijOn f s t) (hss₁ : s ⊆ s₁) (htt' : t ⊆ t')
    (ht' : SurjOn f s₁ t') : ∃ s', s ⊆ s' ∧ s' ⊆ s₁ ∧ Set.BijOn f s' t' := by
  /-
    α : Type u_1
    β : Type u_2
    s s₁ : Set α
    t : Set β
    f : α → β
    t' : Set β
    h : Set.BijOn f s t
    hss₁ : HasSubset.Subset s s₁
    htt' : HasSubset.Subset t t'
    ht' : Set.SurjOn f s₁ t'
    ⊢ Exists fun s' => And (HasSubset.Subset s s') (And (HasSubset.Subset s' s₁) ( …
  -/
  obtain ⟨r, hrss, hbij⟩ := exists_subset_bijOn ((s₁ ∩ f ⁻¹' t') \ f ⁻¹' t) f
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    s s₁ : Set α
    t : Set β
    f : α → β
    t' : Set β
    h : Set.BijOn f s t
    hss₁ : HasSubset.Subset s s₁
    htt' : HasSubset.Subset t t'
    ht' : Set.SurjOn f s₁ t'
    r : Set α
    hrss : HasSubset.Subset r (SDiff.sdiff (Inter.inter s₁ (Set.preimage f t')) (S …
    hbij : Set.BijOn f r (Set.image f (SDiff.sdiff (Inter.inter s₁ (Set.preimage f …
    ⊢ Exists fun s' => And (HasSubset.Subset s s') (And (HasSubset.Subset s' s₁) ( …
  -/
  rw [image_diff_preimage, image_inter_preimage] at hbij
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    s s₁ : Set α
    t : Set β
    f : α → β
    t' : Set β
    h : Set.BijOn f s t
    hss₁ : HasSubset.Subset s s₁
    htt' : HasSubset.Subset t t'
    ht' : Set.SurjOn f s₁ t'
    r : Set α
    hrss : HasSubset.Subset r (SDiff.sdiff (Inter.inter s₁ (Set.preimage f t')) (S …
    hbij : Set.BijOn f r (SDiff.sdiff (Inter.inter (Set.image f s₁) t') t)
    ⊢ Exists fun s' => And (HasSubset.Subset s s') (And (HasSubset.Subset s' s₁) ( …
  -/
  refine ⟨s ∪ r, subset_union_left, ?_, ?_, ?_, fun y hyt' ↦ ?_⟩
    /-
      case intro.intro.refine_1
      α : Type u_1
      β : Type u_2
      s s₁ : Set α
      t : Set β
      f : α → β
      t' : Set β
      h : Set.BijOn f s t
      hss₁ : HasSubset.Subset s s₁
      htt' : HasSubset.Subset t t'
      ht' : Set.SurjOn f s₁ t'
      r : Set α
      hrss : HasSubset.Subset r (SDiff.sdiff (Inter.inter s₁ (Set.preimage f t')) (S …
      hbij : Set.BijOn f r (SDiff.sdiff (Inter.inter (Set.image f s₁) t') t)
      ⊢ HasSubset.Subset (Union.union s r) s₁
    -/
  · exact union_subset hss₁ <| hrss.trans <| diff_subset.trans inter_subset_left
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2
      α : Type u_1
      β : Type u_2
      s s₁ : Set α
      t : Set β
      f : α → β
      t' : Set β
      h : Set.BijOn f s t
      hss₁ : HasSubset.Subset s s₁
      htt' : HasSubset.Subset t t'
      ht' : Set.SurjOn f s₁ t'
      r : Set α
      hrss : HasSubset.Subset r (SDiff.sdiff (Inter.inter s₁ (Set.preimage f t')) (S …
      hbij : Set.BijOn f r (SDiff.sdiff (Inter.inter (Set.image f s₁) t') t)
      ⊢ Set.MapsTo f (Union.union s r) t'
    -/
  · rw [mapsTo', image_union, hbij.image_eq, h.image_eq, union_subset_iff]
    /-
      case intro.intro.refine_2
      α : Type u_1
      β : Type u_2
      s s₁ : Set α
      t : Set β
      f : α → β
      t' : Set β
      h : Set.BijOn f s t
      hss₁ : HasSubset.Subset s s₁
      htt' : HasSubset.Subset t t'
      ht' : Set.SurjOn f s₁ t'
      r : Set α
      hrss : HasSubset.Subset r (SDiff.sdiff (Inter.inter s₁ (Set.preimage f t')) (S …
      hbij : Set.BijOn f r (SDiff.sdiff (Inter.inter (Set.image f s₁) t') t)
      ⊢ And (HasSubset.Subset t t') (HasSubset.Subset (SDiff.sdiff (Inter.inter (Set …
    -/
    exact ⟨htt', diff_subset.trans inter_subset_right⟩
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_3
      α : Type u_1
      β : Type u_2
      s s₁ : Set α
      t : Set β
      f : α → β
      t' : Set β
      h : Set.BijOn f s t
      hss₁ : HasSubset.Subset s s₁
      htt' : HasSubset.Subset t t'
      ht' : Set.SurjOn f s₁ t'
      r : Set α
      hrss : HasSubset.Subset r (SDiff.sdiff (Inter.inter s₁ (Set.preimage f t')) (S …
      hbij : Set.BijOn f r (SDiff.sdiff (Inter.inter (Set.image f s₁) t') t)
      ⊢ Set.InjOn f (Union.union s r)
    -/
  · rw [injOn_union, and_iff_right h.injOn, and_iff_right hbij.injOn]
      /-
        case intro.intro.refine_3
        α : Type u_1
        β : Type u_2
        s s₁ : Set α
        t : Set β
        f : α → β
        t' : Set β
        h : Set.BijOn f s t
        hss₁ : HasSubset.Subset s s₁
        htt' : HasSubset.Subset t t'
        ht' : Set.SurjOn f s₁ t'
        r : Set α
        hrss : HasSubset.Subset r (SDiff.sdiff (Inter.inter s₁ (Set.preimage f t')) (S …
        hbij : Set.BijOn f r (SDiff.sdiff (Inter.inter (Set.image f s₁) t') t)
        ⊢ ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem r y → Ne (f x) (f y)
      -/
    · refine fun x hxs y hyr hxy ↦ (hrss hyr).2 ?_
      /-
        case intro.intro.refine_3
        α : Type u_1
        β : Type u_2
        s s₁ : Set α
        t : Set β
        f : α → β
        t' : Set β
        h : Set.BijOn f s t
        hss₁ : HasSubset.Subset s s₁
        htt' : HasSubset.Subset t t'
        ht' : Set.SurjOn f s₁ t'
        r : Set α
        hrss : HasSubset.Subset r (SDiff.sdiff (Inter.inter s₁ (Set.preimage f t')) (S …
        hbij : Set.BijOn f r (SDiff.sdiff (Inter.inter (Set.image f s₁) t') t)
        x : α
        hxs : Membership.mem s x
        y : α
        hyr : Membership.mem r y
        hxy : Eq (f x) (f y)
        ⊢ Membership.mem (Set.preimage f t) y
      -/
      rw [← h.image_eq]
      /-
        case intro.intro.refine_3
        α : Type u_1
        β : Type u_2
        s s₁ : Set α
        t : Set β
        f : α → β
        t' : Set β
        h : Set.BijOn f s t
        hss₁ : HasSubset.Subset s s₁
        htt' : HasSubset.Subset t t'
        ht' : Set.SurjOn f s₁ t'
        r : Set α
        hrss : HasSubset.Subset r (SDiff.sdiff (Inter.inter s₁ (Set.preimage f t')) (S …
        hbij : Set.BijOn f r (SDiff.sdiff (Inter.inter (Set.image f s₁) t') t)
        x : α
        hxs : Membership.mem s x
        y : α
        hyr : Membership.mem r y
        hxy : Eq (f x) (f y)
        ⊢ Membership.mem (Set.preimage f (Set.image f s)) y
      -/
      exact ⟨x, hxs, hxy⟩
      /-
        🎉 no goals
      -/
    /-
      case intro.intro.refine_3
      α : Type u_1
      β : Type u_2
      s s₁ : Set α
      t : Set β
      f : α → β
      t' : Set β
      h : Set.BijOn f s t
      hss₁ : HasSubset.Subset s s₁
      htt' : HasSubset.Subset t t'
      ht' : Set.SurjOn f s₁ t'
      r : Set α
      hrss : HasSubset.Subset r (SDiff.sdiff (Inter.inter s₁ (Set.preimage f t')) (S …
      hbij : Set.BijOn f r (SDiff.sdiff (Inter.inter (Set.image f s₁) t') t)
      ⊢ Disjoint s r
    -/
    exact (subset_diff.1 hrss).2.symm.mono_left h.mapsTo
    /-
      🎉 no goals
    -/
  /-
    case intro.intro.refine_4
    α : Type u_1
    β : Type u_2
    s s₁ : Set α
    t : Set β
    f : α → β
    t' : Set β
    h : Set.BijOn f s t
    hss₁ : HasSubset.Subset s s₁
    htt' : HasSubset.Subset t t'
    ht' : Set.SurjOn f s₁ t'
    r : Set α
    hrss : HasSubset.Subset r (SDiff.sdiff (Inter.inter s₁ (Set.preimage f t')) (S …
    hbij : Set.BijOn f r (SDiff.sdiff (Inter.inter (Set.image f s₁) t') t)
    y : β
    hyt' : Membership.mem t' y
    ⊢ Membership.mem (Set.image f (Union.union s r)) y
  -/
  rw [image_union, h.image_eq, hbij.image_eq, union_diff_self]
  /-
    case intro.intro.refine_4
    α : Type u_1
    β : Type u_2
    s s₁ : Set α
    t : Set β
    f : α → β
    t' : Set β
    h : Set.BijOn f s t
    hss₁ : HasSubset.Subset s s₁
    htt' : HasSubset.Subset t t'
    ht' : Set.SurjOn f s₁ t'
    r : Set α
    hrss : HasSubset.Subset r (SDiff.sdiff (Inter.inter s₁ (Set.preimage f t')) (S …
    hbij : Set.BijOn f r (SDiff.sdiff (Inter.inter (Set.image f s₁) t') t)
    y : β
    hyt' : Membership.mem t' y
    ⊢ Membership.mem (Union.union t (Inter.inter (Set.image f s₁) t')) y
  -/
  exact .inr ⟨ht' hyt', hyt'⟩
  /-
    🎉 no goals
  -/


/-- If `f` maps `s` bijectively to `t`, and `t'` is a superset of `t` contained in the range of `f`,
then `f` maps some superset of `s` bijectively to `t'`. -/
theorem BijOn.exists_extend {t' : Set β} (h : BijOn f s t) (htt' : t ⊆ t') (ht' : t' ⊆ range f) :
    ∃ s', s ⊆ s' ∧ BijOn f s' t' := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    f : α → β
    t' : Set β
    h : Set.BijOn f s t
    htt' : HasSubset.Subset t t'
    ht' : HasSubset.Subset t' (Set.range f)
    ⊢ Exists fun s' => And (HasSubset.Subset s s') (Set.BijOn f s' t')
  -/
  simpa using h.exists_extend_of_subset (subset_univ s) htt' (by simpa [SurjOn])
  /-
    🎉 no goals
  -/


theorem InjOn.exists_subset_injOn_subset_range_eq {r : Set α} (hinj : InjOn f r) (hrs : r ⊆ s) :
    ∃ u : Set α, r ⊆ u ∧ u ⊆ s ∧ f '' u = f '' s ∧ InjOn f u := by
  obtain ⟨u, hru, hus, h⟩ := hinj.bijOn_image.exists_extend_of_subset hrs
    (image_subset f hrs) Subset.rfl
  /-
    case intro.intro.intro
    α : Type u_1
    β : Type u_2
    s : Set α
    f : α → β
    r : Set α
    hinj : Set.InjOn f r
    hrs : HasSubset.Subset r s
    u : Set α
    hru : HasSubset.Subset r u
    hus : HasSubset.Subset u s
    h : Set.BijOn f u (Set.image f s)
    ⊢ Exists fun u => And (HasSubset.Subset r u) (And (HasSubset.Subset u s) (And  …
  -/
  exact ⟨u, hru, hus, h.image_eq, h.injOn⟩
  /-
    🎉 no goals
  -/


theorem preimage_invFun_of_mem [n : Nonempty α] {f : α → β} (hf : Injective f) {s : Set α}
    (h : Classical.choice n ∈ s) : invFun f ⁻¹' s = f '' s ∪ (range f)ᶜ := by
  /-
    α : Type u_1
    β : Type u_2
    n : Nonempty α
    f : α → β
    hf : Function.Injective f
    s : Set α
    h : Membership.mem s (Classical.choice n)
    ⊢ Eq (Set.preimage (Function.invFun f) s) (Union.union (Set.image f s) (HasCom …
  -/
  ext x
  /-
    case h
    α : Type u_1
    β : Type u_2
    n : Nonempty α
    f : α → β
    hf : Function.Injective f
    s : Set α
    h : Membership.mem s (Classical.choice n)
    x : β
    ⊢ Iff (Membership.mem (Set.preimage (Function.invFun f) s) x) (Membership.mem  …
  -/
  rcases em (x ∈ range f) with (⟨a, rfl⟩ | hx)
  · simp only [mem_preimage, mem_union, mem_compl_iff, mem_range_self, not_true, or_false,
      leftInverse_invFun hf _, hf.mem_set_image]
    /-
      case h.inr
      α : Type u_1
      β : Type u_2
      n : Nonempty α
      f : α → β
      hf : Function.Injective f
      s : Set α
      h : Membership.mem s (Classical.choice n)
      x : β
      hx : Not (Membership.mem (Set.range f) x)
      ⊢ Iff (Membership.mem (Set.preimage (Function.invFun f) s) x) (Membership.mem  …
    -/
  · simp only [mem_preimage, invFun_neg hx, h, hx, mem_union, mem_compl_iff, not_false_iff, or_true]
    /-
      🎉 no goals
    -/


theorem preimage_invFun_of_not_mem [n : Nonempty α] {f : α → β} (hf : Injective f) {s : Set α}
    (h : Classical.choice n ∉ s) : invFun f ⁻¹' s = f '' s := by
  /-
    α : Type u_1
    β : Type u_2
    n : Nonempty α
    f : α → β
    hf : Function.Injective f
    s : Set α
    h : Not (Membership.mem s (Classical.choice n))
    ⊢ Eq (Set.preimage (Function.invFun f) s) (Set.image f s)
  -/
  ext x
  /-
    case h
    α : Type u_1
    β : Type u_2
    n : Nonempty α
    f : α → β
    hf : Function.Injective f
    s : Set α
    h : Not (Membership.mem s (Classical.choice n))
    x : β
    ⊢ Iff (Membership.mem (Set.preimage (Function.invFun f) s) x) (Membership.mem  …
  -/
  rcases em (x ∈ range f) with (⟨a, rfl⟩ | hx)
    /-
      case h.inl.intro
      α : Type u_1
      β : Type u_2
      n : Nonempty α
      f : α → β
      hf : Function.Injective f
      s : Set α
      h : Not (Membership.mem s (Classical.choice n))
      a : α
      ⊢ Iff (Membership.mem (Set.preimage (Function.invFun f) s) (f a)) (Membership. …
    -/
  · rw [mem_preimage, leftInverse_invFun hf, hf.mem_set_image]
    /-
      🎉 no goals
    -/
    /-
      case h.inr
      α : Type u_1
      β : Type u_2
      n : Nonempty α
      f : α → β
      hf : Function.Injective f
      s : Set α
      h : Not (Membership.mem s (Classical.choice n))
      x : β
      hx : Not (Membership.mem (Set.range f) x)
      ⊢ Iff (Membership.mem (Set.preimage (Function.invFun f) s) x) (Membership.mem  …
    -/
  · have : x ∉ f '' s := fun h' => hx (image_subset_range _ _ h')
    /-
      case h.inr
      α : Type u_1
      β : Type u_2
      n : Nonempty α
      f : α → β
      hf : Function.Injective f
      s : Set α
      h : Not (Membership.mem s (Classical.choice n))
      x : β
      hx : Not (Membership.mem (Set.range f) x)
      this : Not (Membership.mem (Set.image f s) x)
      ⊢ Iff (Membership.mem (Set.preimage (Function.invFun f) s) x) (Membership.mem  …
    -/
    simp only [mem_preimage, invFun_neg hx, h, this]
    /-
      🎉 no goals
    -/


lemma BijOn.symm {g : β → α} (h : InvOn f g t s) (hf : BijOn f s t) : BijOn g t s :=
  ⟨h.2.mapsTo hf.surjOn, h.1.injOn, h.2.surjOn hf.mapsTo⟩


lemma bijOn_comm {g : β → α} (h : InvOn f g t s) : BijOn f s t ↔ BijOn g t s :=
  ⟨BijOn.symm h, BijOn.symm h.symm⟩


@[simp]
theorem piecewise_empty [∀ i : α, Decidable (i ∈ (∅ : Set α))] : piecewise ∅ f g = g := by
  /-
    α : Type u_1
    δ : α → Sort u_7
    f g : (i : α) → δ i
    inst✝ : (i : α) → Decidable (Membership.mem EmptyCollection.emptyCollection i)
    ⊢ Eq (EmptyCollection.emptyCollection.piecewise f g) g
  -/
  ext i
  /-
    case h
    α : Type u_1
    δ : α → Sort u_7
    f g : (i : α) → δ i
    inst✝ : (i : α) → Decidable (Membership.mem EmptyCollection.emptyCollection i)
    i : α
    ⊢ Eq (EmptyCollection.emptyCollection.piecewise f g i) (g i)
  -/
  simp [piecewise]
  /-
    🎉 no goals
  -/


@[simp]
theorem piecewise_univ [∀ i : α, Decidable (i ∈ (Set.univ : Set α))] :
    piecewise Set.univ f g = f := by
  /-
    α : Type u_1
    δ : α → Sort u_7
    f g : (i : α) → δ i
    inst✝ : (i : α) → Decidable (Membership.mem Set.univ i)
    ⊢ Eq (Set.univ.piecewise f g) f
  -/
  ext i
  /-
    case h
    α : Type u_1
    δ : α → Sort u_7
    f g : (i : α) → δ i
    inst✝ : (i : α) → Decidable (Membership.mem Set.univ i)
    i : α
    ⊢ Eq (Set.univ.piecewise f g i) (f i)
  -/
  simp [piecewise]
  /-
    🎉 no goals
  -/

--@[simp] -- Porting note: simpNF linter complains

theorem piecewise_insert_self {j : α} [∀ i, Decidable (i ∈ insert j s)] :
                                             /-
                                               α : Type u_1
                                               δ : α → Sort u_7
                                               s : Set α
                                               f g : (i : α) → δ i
                                               j : α
                                               inst✝ : (i : α) → Decidable (Membership.mem (Insert.insert j s) i)
                                               ⊢ Eq ((Insert.insert j s).piecewise f g j) (f j)
                                             -/
    (insert j s).piecewise f g j = f j := by simp [piecewise]
                                             /-
                                               🎉 no goals
                                             -/


theorem piecewise_insert [DecidableEq α] (j : α) [∀ i, Decidable (i ∈ insert j s)] :
    (insert j s).piecewise f g = Function.update (s.piecewise f g) j (f j) := by
  /-
    α : Type u_1
    δ : α → Sort u_7
    s : Set α
    f g : (i : α) → δ i
    inst✝² : (j : α) → Decidable (Membership.mem s j)
    inst✝¹ : DecidableEq α
    j : α
    inst✝ : (i : α) → Decidable (Membership.mem (Insert.insert j s) i)
    ⊢ Eq ((Insert.insert j s).piecewise f g) (Function.update (s.piecewise f g) j  …
  -/
  simp (config := { unfoldPartialApp := true }) only [piecewise, mem_insert_iff]
  /-
    α : Type u_1
    δ : α → Sort u_7
    s : Set α
    f g : (i : α) → δ i
    inst✝² : (j : α) → Decidable (Membership.mem s j)
    inst✝¹ : DecidableEq α
    j : α
    inst✝ : (i : α) → Decidable (Membership.mem (Insert.insert j s) i)
    ⊢ Eq (fun i => ite (Or (Eq i j) (Membership.mem s i)) (f i) (g i)) (Function.u …
  -/
  ext i
  /-
    case h
    α : Type u_1
    δ : α → Sort u_7
    s : Set α
    f g : (i : α) → δ i
    inst✝² : (j : α) → Decidable (Membership.mem s j)
    inst✝¹ : DecidableEq α
    j : α
    inst✝ : (i : α) → Decidable (Membership.mem (Insert.insert j s) i)
    i : α
    ⊢ Eq (ite (Or (Eq i j) (Membership.mem s i)) (f i) (g i)) (Function.update (fu …
  -/
  by_cases h : i = j
    /-
      case pos
      α : Type u_1
      δ : α → Sort u_7
      s : Set α
      f g : (i : α) → δ i
      inst✝² : (j : α) → Decidable (Membership.mem s j)
      inst✝¹ : DecidableEq α
      j : α
      inst✝ : (i : α) → Decidable (Membership.mem (Insert.insert j s) i)
      i : α
      h : Eq i j
      ⊢ Eq (ite (Or (Eq i j) (Membership.mem s i)) (f i) (g i)) (Function.update (fu …
    -/
  · rw [h]
    /-
      case pos
      α : Type u_1
      δ : α → Sort u_7
      s : Set α
      f g : (i : α) → δ i
      inst✝² : (j : α) → Decidable (Membership.mem s j)
      inst✝¹ : DecidableEq α
      j : α
      inst✝ : (i : α) → Decidable (Membership.mem (Insert.insert j s) i)
      i : α
      h : Eq i j
      ⊢ Eq (ite (Or (Eq j j) (Membership.mem s j)) (f j) (g j)) (Function.update (fu …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      δ : α → Sort u_7
      s : Set α
      f g : (i : α) → δ i
      inst✝² : (j : α) → Decidable (Membership.mem s j)
      inst✝¹ : DecidableEq α
      j : α
      inst✝ : (i : α) → Decidable (Membership.mem (Insert.insert j s) i)
      i : α
      h : Not (Eq i j)
      ⊢ Eq (ite (Or (Eq i j) (Membership.mem s i)) (f i) (g i)) (Function.update (fu …
    -/
                            /-
                              🎉 no goals
                            -/
  · by_cases h' : i ∈ s <;> simp [h, h']
                            /-
                              🎉 no goals
                            -/


@[simp]
theorem piecewise_eq_of_mem {i : α} (hi : i ∈ s) : s.piecewise f g i = f i :=
  if_pos hi


@[simp]
theorem piecewise_eq_of_not_mem {i : α} (hi : i ∉ s) : s.piecewise f g i = g i :=
  if_neg hi


theorem piecewise_singleton (x : α) [∀ y, Decidable (y ∈ ({x} : Set α))] [DecidableEq α]
    (f g : α → β) : piecewise {x} f g = Function.update g x (f x) := by
  /-
    α : Type u_1
    β : Type u_2
    x : α
    inst✝¹ : (y : α) → Decidable (Membership.mem (Singleton.singleton x) y)
    inst✝ : DecidableEq α
    f g : α → β
    ⊢ Eq ((Singleton.singleton x).piecewise f g) (Function.update g x (f x))
  -/
  ext y
  /-
    case h
    α : Type u_1
    β : Type u_2
    x : α
    inst✝¹ : (y : α) → Decidable (Membership.mem (Singleton.singleton x) y)
    inst✝ : DecidableEq α
    f g : α → β
    y : α
    ⊢ Eq ((Singleton.singleton x).piecewise f g y) (Function.update g x (f x) y)
  -/
  by_cases hy : y = x
    /-
      case pos
      α : Type u_1
      β : Type u_2
      x : α
      inst✝¹ : (y : α) → Decidable (Membership.mem (Singleton.singleton x) y)
      inst✝ : DecidableEq α
      f g : α → β
      y : α
      hy : Eq y x
      ⊢ Eq ((Singleton.singleton x).piecewise f g y) (Function.update g x (f x) y)
    -/
  · subst y
    /-
      case pos
      α : Type u_1
      β : Type u_2
      x : α
      inst✝¹ : (y : α) → Decidable (Membership.mem (Singleton.singleton x) y)
      inst✝ : DecidableEq α
      f g : α → β
      ⊢ Eq ((Singleton.singleton x).piecewise f g x) (Function.update g x (f x) x)
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      x : α
      inst✝¹ : (y : α) → Decidable (Membership.mem (Singleton.singleton x) y)
      inst✝ : DecidableEq α
      f g : α → β
      y : α
      hy : Not (Eq y x)
      ⊢ Eq ((Singleton.singleton x).piecewise f g y) (Function.update g x (f x) y)
    -/
  · simp [hy]
    /-
      🎉 no goals
    -/


theorem piecewise_eqOn (f g : α → β) : EqOn (s.piecewise f g) f s := fun _ =>
  piecewise_eq_of_mem _ _ _


theorem piecewise_eqOn_compl (f g : α → β) : EqOn (s.piecewise f g) g sᶜ := fun _ =>
  piecewise_eq_of_not_mem _ _ _


theorem piecewise_le {δ : α → Type*} [∀ i, Preorder (δ i)] {s : Set α} [∀ j, Decidable (j ∈ s)]
    {f₁ f₂ g : ∀ i, δ i} (h₁ : ∀ i ∈ s, f₁ i ≤ g i) (h₂ : ∀ i ∉ s, f₂ i ≤ g i) :
                                                           /-
                                                             α : Type u_1
                                                             δ : α → Type u_8
                                                             inst✝¹ : (i : α) → Preorder (δ i)
                                                             s : Set α
                                                             inst✝ : (j : α) → Decidable (Membership.mem s j)
                                                             f₁ f₂ g : (i : α) → δ i
                                                             h₁ : ∀ (i : α), Membership.mem s i → LE.le (f₁ i) (g i)
                                                             h₂ : ∀ (i : α), Not (Membership.mem s i) → LE.le (f₂ i) (g i)
                                                             i : α
                                                             h : Membership.mem s i
                                                             ⊢ LE.le (s.piecewise f₁ f₂ i) (g i)
                                                           -/
                                                           /-
                                                             🎉 no goals
                                                           -/
    s.piecewise f₁ f₂ ≤ g := fun i => if h : i ∈ s then by simp [*] else by simp [*]
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


theorem le_piecewise {δ : α → Type*} [∀ i, Preorder (δ i)] {s : Set α} [∀ j, Decidable (j ∈ s)]
    {f₁ f₂ g : ∀ i, δ i} (h₁ : ∀ i ∈ s, g i ≤ f₁ i) (h₂ : ∀ i ∉ s, g i ≤ f₂ i) :
    g ≤ s.piecewise f₁ f₂ :=
  @piecewise_le α (fun i => (δ i)ᵒᵈ) _ s _ _ _ _ h₁ h₂


@[gcongr]
theorem piecewise_mono {δ : α → Type*} [∀ i, Preorder (δ i)] {s : Set α}
    [∀ j, Decidable (j ∈ s)] {f₁ f₂ g₁ g₂ : ∀ i, δ i} (h₁ : ∀ i ∈ s, f₁ i ≤ g₁ i)
    (h₂ : ∀ i ∉ s, f₂ i ≤ g₂ i) : s.piecewise f₁ f₂ ≤ s.piecewise g₁ g₂ := by
  /-
    α : Type u_1
    δ : α → Type u_8
    inst✝¹ : (i : α) → Preorder (δ i)
    s : Set α
    inst✝ : (j : α) → Decidable (Membership.mem s j)
    f₁ f₂ g₁ g₂ : (i : α) → δ i
    h₁ : ∀ (i : α), Membership.mem s i → LE.le (f₁ i) (g₁ i)
    h₂ : ∀ (i : α), Not (Membership.mem s i) → LE.le (f₂ i) (g₂ i)
    ⊢ LE.le (s.piecewise f₁ f₂) (s.piecewise g₁ g₂)
  -/
                                    /-
                                      🎉 no goals
                                    -/
  apply piecewise_le <;> intros <;> simp [*]
                                    /-
                                      🎉 no goals
                                    -/


@[deprecated (since := "2024-10-06")] alias piecewise_le_piecewise := piecewise_mono


@[simp]
theorem piecewise_insert_of_ne {i j : α} (h : i ≠ j) [∀ i, Decidable (i ∈ insert j s)] :
                                                           /-
                                                             α : Type u_1
                                                             δ : α → Sort u_7
                                                             s : Set α
                                                             f g : (i : α) → δ i
                                                             inst✝¹ : (j : α) → Decidable (Membership.mem s j)
                                                             i j : α
                                                             h : Ne i j
                                                             inst✝ : (i : α) → Decidable (Membership.mem (Insert.insert j s) i)
                                                             ⊢ Eq ((Insert.insert j s).piecewise f g i) (s.piecewise f g i)
                                                           -/
    (insert j s).piecewise f g i = s.piecewise f g i := by simp [piecewise, h]
                                                           /-
                                                             🎉 no goals
                                                           -/


@[simp]
theorem piecewise_compl [∀ i, Decidable (i ∈ sᶜ)] : sᶜ.piecewise f g = s.piecewise g f :=
                                        /-
                                          α : Type u_1
                                          δ : α → Sort u_7
                                          s : Set α
                                          f g : (i : α) → δ i
                                          inst✝¹ : (j : α) → Decidable (Membership.mem s j)
                                          inst✝ : (i : α) → Decidable (Membership.mem (HasCompl.compl s) i)
                                          x : α
                                          hx : Membership.mem s x
                                          ⊢ Eq ((HasCompl.compl s).piecewise f g x) (s.piecewise g f x)
                                        -/
                                        /-
                                          🎉 no goals
                                        -/
  funext fun x => if hx : x ∈ s then by simp [hx] else by simp [hx]
                                                          /-
                                                            🎉 no goals
                                                          -/


@[simp]
theorem piecewise_range_comp {ι : Sort*} (f : ι → α) [∀ j, Decidable (j ∈ range f)]
    (g₁ g₂ : α → β) : (range f).piecewise g₁ g₂ ∘ f = g₁ ∘ f :=
  (piecewise_eqOn ..).comp_eq


theorem MapsTo.piecewise_ite {s s₁ s₂ : Set α} {t t₁ t₂ : Set β} {f₁ f₂ : α → β}
    [∀ i, Decidable (i ∈ s)] (h₁ : MapsTo f₁ (s₁ ∩ s) (t₁ ∩ t))
    (h₂ : MapsTo f₂ (s₂ ∩ sᶜ) (t₂ ∩ tᶜ)) :
    MapsTo (s.piecewise f₁ f₂) (s.ite s₁ s₂) (t.ite t₁ t₂) := by
  /-
    α : Type u_1
    β : Type u_2
    s s₁ s₂ : Set α
    t t₁ t₂ : Set β
    f₁ f₂ : α → β
    inst✝ : (i : α) → Decidable (Membership.mem s i)
    h₁ : Set.MapsTo f₁ (Inter.inter s₁ s) (Inter.inter t₁ t)
    h₂ : Set.MapsTo f₂ (Inter.inter s₂ (HasCompl.compl s)) (Inter.inter t₂ (HasCom …
    ⊢ Set.MapsTo (s.piecewise f₁ f₂) (s.ite s₁ s₂) (t.ite t₁ t₂)
  -/
  refine (h₁.congr ?_).union_union (h₂.congr ?_)
  exacts [(piecewise_eqOn s f₁ f₂).symm.mono inter_subset_right,
    (piecewise_eqOn_compl s f₁ f₂).symm.mono inter_subset_right]


theorem eqOn_piecewise {f f' g : α → β} {t} :
    EqOn (s.piecewise f f') g t ↔ EqOn f g (t ∩ s) ∧ EqOn f' g (t ∩ sᶜ) := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    inst✝ : (j : α) → Decidable (Membership.mem s j)
    f f' g : α → β
    t : Set α
    ⊢ Iff (Set.EqOn (s.piecewise f f') g t) (And (Set.EqOn f g (Inter.inter t s))  …
  -/
  simp only [EqOn, ← forall_and]
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    inst✝ : (j : α) → Decidable (Membership.mem s j)
    f f' g : α → β
    t : Set α
    ⊢ Iff (∀ ⦃x : α⦄, Membership.mem t x → Eq (s.piecewise f f' x) (g x)) (∀ (x :  …
  -/
                                                       /-
                                                         🎉 no goals
                                                       -/
  refine forall_congr' fun a => ?_; by_cases a ∈ s <;> simp [*]
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem EqOn.piecewise_ite' {f f' g : α → β} {t t'} (h : EqOn f g (t ∩ s))
    (h' : EqOn f' g (t' ∩ sᶜ)) : EqOn (s.piecewise f f') g (s.ite t t') := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    inst✝ : (j : α) → Decidable (Membership.mem s j)
    f f' g : α → β
    t t' : Set α
    h : Set.EqOn f g (Inter.inter t s)
    h' : Set.EqOn f' g (Inter.inter t' (HasCompl.compl s))
    ⊢ Set.EqOn (s.piecewise f f') g (s.ite t t')
  -/
  simp [eqOn_piecewise, *]
  /-
    🎉 no goals
  -/


theorem EqOn.piecewise_ite {f f' g : α → β} {t t'} (h : EqOn f g t) (h' : EqOn f' g t') :
    EqOn (s.piecewise f f') g (s.ite t t') :=
  (h.mono inter_subset_left).piecewise_ite' s (h'.mono inter_subset_left)


theorem piecewise_preimage (f g : α → β) (t) : s.piecewise f g ⁻¹' t = s.ite (f ⁻¹' t) (g ⁻¹' t) :=
                  /-
                    α : Type u_1
                    β : Type u_2
                    s : Set α
                    inst✝ : (j : α) → Decidable (Membership.mem s j)
                    f g : α → β
                    t : Set β
                    x : α
                    ⊢ Iff (Membership.mem (Set.preimage (s.piecewise f g) t) x) (Membership.mem (s …
                  -/
                                     /-
                                       🎉 no goals
                                     -/
  ext fun x => by by_cases x ∈ s <;> simp [*, Set.ite]
                                     /-
                                       🎉 no goals
                                     -/


theorem apply_piecewise {δ' : α → Sort*} (h : ∀ i, δ i → δ' i) {x : α} :
    h x (s.piecewise f g x) = s.piecewise (fun x => h x (f x)) (fun x => h x (g x)) x := by
  /-
    α : Type u_1
    δ : α → Sort u_7
    s : Set α
    f g : (i : α) → δ i
    inst✝ : (j : α) → Decidable (Membership.mem s j)
    δ' : α → Sort u_8
    h : (i : α) → δ i → δ' i
    x : α
    ⊢ Eq (h x (s.piecewise f g x)) (s.piecewise (fun x => h x (f x)) (fun x => h x …
  -/
                          /-
                            🎉 no goals
                          -/
  by_cases hx : x ∈ s <;> simp [hx]
                          /-
                            🎉 no goals
                          -/


theorem apply_piecewise₂ {δ' δ'' : α → Sort*} (f' g' : ∀ i, δ' i) (h : ∀ i, δ i → δ' i → δ'' i)
    {x : α} :
    h x (s.piecewise f g x) (s.piecewise f' g' x) =
      s.piecewise (fun x => h x (f x) (f' x)) (fun x => h x (g x) (g' x)) x := by
  /-
    α : Type u_1
    δ : α → Sort u_7
    s : Set α
    f g : (i : α) → δ i
    inst✝ : (j : α) → Decidable (Membership.mem s j)
    δ' : α → Sort u_8
    δ'' : α → Sort u_9
    f' g' : (i : α) → δ' i
    h : (i : α) → δ i → δ' i → δ'' i
    x : α
    ⊢ Eq (h x (s.piecewise f g x) (s.piecewise f' g' x)) (s.piecewise (fun x => h  …
  -/
                          /-
                            🎉 no goals
                          -/
  by_cases hx : x ∈ s <;> simp [hx]
                          /-
                            🎉 no goals
                          -/


theorem piecewise_op {δ' : α → Sort*} (h : ∀ i, δ i → δ' i) :
    (s.piecewise (fun x => h x (f x)) fun x => h x (g x)) = fun x => h x (s.piecewise f g x) :=
  funext fun _ => (apply_piecewise _ _ _ _).symm


theorem piecewise_op₂ {δ' δ'' : α → Sort*} (f' g' : ∀ i, δ' i) (h : ∀ i, δ i → δ' i → δ'' i) :
    (s.piecewise (fun x => h x (f x) (f' x)) fun x => h x (g x) (g' x)) = fun x =>
      h x (s.piecewise f g x) (s.piecewise f' g' x) :=
  funext fun _ => (apply_piecewise₂ _ _ _ _ _ _).symm


@[simp]
theorem piecewise_same : s.piecewise f f = f := by
  /-
    α : Type u_1
    δ : α → Sort u_7
    s : Set α
    f : (i : α) → δ i
    inst✝ : (j : α) → Decidable (Membership.mem s j)
    ⊢ Eq (s.piecewise f f) f
  -/
  ext x
  /-
    case h
    α : Type u_1
    δ : α → Sort u_7
    s : Set α
    f : (i : α) → δ i
    inst✝ : (j : α) → Decidable (Membership.mem s j)
    x : α
    ⊢ Eq (s.piecewise f f x) (f x)
  -/
                          /-
                            🎉 no goals
                          -/
  by_cases hx : x ∈ s <;> simp [hx]
                          /-
                            🎉 no goals
                          -/


theorem range_piecewise (f g : α → β) : range (s.piecewise f g) = f '' s ∪ g '' sᶜ := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    inst✝ : (j : α) → Decidable (Membership.mem s j)
    f g : α → β
    ⊢ Eq (Set.range (s.piecewise f g)) (Union.union (Set.image f s) (Set.image g ( …
  -/
  ext y; constructor
    /-
      case h.mp
      α : Type u_1
      β : Type u_2
      s : Set α
      inst✝ : (j : α) → Decidable (Membership.mem s j)
      f g : α → β
      y : β
      ⊢ Membership.mem (Set.range (s.piecewise f g)) y → Membership.mem (Union.union …
    -/
  · rintro ⟨x, rfl⟩
    /-
      case h.mp.intro
      α : Type u_1
      β : Type u_2
      s : Set α
      inst✝ : (j : α) → Decidable (Membership.mem s j)
      f g : α → β
      x : α
      ⊢ Membership.mem (Union.union (Set.image f s) (Set.image g (HasCompl.compl s)) …
    -/
                                                       /-
                                                         🎉 no goals
                                                       -/
    by_cases h : x ∈ s <;> [left; right] <;> use x <;> simp [h]
                                                       /-
                                                         🎉 no goals
                                                       -/
    /-
      case h.mpr
      α : Type u_1
      β : Type u_2
      s : Set α
      inst✝ : (j : α) → Decidable (Membership.mem s j)
      f g : α → β
      y : β
      ⊢ Membership.mem (Union.union (Set.image f s) (Set.image g (HasCompl.compl s)) …
    -/
                                                       /-
                                                         🎉 no goals
                                                       -/
  · rintro (⟨x, hx, rfl⟩ | ⟨x, hx, rfl⟩) <;> use x <;> simp_all
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem injective_piecewise_iff {f g : α → β} :
    Injective (s.piecewise f g) ↔
      InjOn f s ∧ InjOn g sᶜ ∧ ∀ x ∈ s, ∀ y ∉ s, f x ≠ g y := by
  rw [injective_iff_injOn_univ, ← union_compl_self s, injOn_union (@disjoint_compl_right _ _ s),
    (piecewise_eqOn s f g).injOn_iff, (piecewise_eqOn_compl s f g).injOn_iff]
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    inst✝ : (j : α) → Decidable (Membership.mem s j)
    f g : α → β
    ⊢ Iff (And (Set.InjOn f s) (And (Set.InjOn g (HasCompl.compl s)) (∀ (x : α), M …
  -/
  refine and_congr Iff.rfl (and_congr Iff.rfl <| forall₄_congr fun x hx y hy => ?_)
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    inst✝ : (j : α) → Decidable (Membership.mem s j)
    f g : α → β
    x : α
    hx : Membership.mem s x
    y : α
    hy : Membership.mem (HasCompl.compl s) y
    ⊢ Iff (Ne (s.piecewise f g x) (s.piecewise f g y)) (Ne (f x) (g y))
  -/
  rw [piecewise_eq_of_mem s f g hx, piecewise_eq_of_not_mem s f g hy]
  /-
    🎉 no goals
  -/


theorem piecewise_mem_pi {δ : α → Type*} {t : Set α} {t' : ∀ i, Set (δ i)} {f g} (hf : f ∈ pi t t')
    (hg : g ∈ pi t t') : s.piecewise f g ∈ pi t t' := by
  /-
    α : Type u_1
    s : Set α
    inst✝ : (j : α) → Decidable (Membership.mem s j)
    δ : α → Type u_8
    t : Set α
    t' : (i : α) → Set (δ i)
    f g : (i : α) → δ i
    hf : Membership.mem (t.pi t') f
    hg : Membership.mem (t.pi t') g
    ⊢ Membership.mem (t.pi t') (s.piecewise f g)
  -/
  intro i ht
  /-
    α : Type u_1
    s : Set α
    inst✝ : (j : α) → Decidable (Membership.mem s j)
    δ : α → Type u_8
    t : Set α
    t' : (i : α) → Set (δ i)
    f g : (i : α) → δ i
    hf : Membership.mem (t.pi t') f
    hg : Membership.mem (t.pi t') g
    i : α
    ht : Membership.mem t i
    ⊢ Membership.mem (t' i) (s.piecewise f g i)
  -/
                          /-
                            🎉 no goals
                          -/
  by_cases hs : i ∈ s <;> simp [hf i ht, hg i ht, hs]
                          /-
                            🎉 no goals
                          -/


@[simp]
theorem pi_piecewise {ι : Type*} {α : ι → Type*} (s s' : Set ι) (t t' : ∀ i, Set (α i))
    [∀ x, Decidable (x ∈ s')] : pi s (s'.piecewise t t') = pi (s ∩ s') t ∩ pi (s \ s') t' :=
  pi_if _ _ _


theorem univ_pi_piecewise {ι : Type*} {α : ι → Type*} (s : Set ι) (t t' : ∀ i, Set (α i))
    [∀ x, Decidable (x ∈ s)] : pi univ (s.piecewise t t') = pi s t ∩ pi sᶜ t' := by
  /-
    ι : Type u_8
    α : ι → Type u_9
    s : Set ι
    t t' : (i : ι) → Set (α i)
    inst✝ : (x : ι) → Decidable (Membership.mem s x)
    ⊢ Eq (Set.univ.pi (s.piecewise t t')) (Inter.inter (s.pi t) ((HasCompl.compl s …
  -/
  simp [compl_eq_univ_diff]
  /-
    🎉 no goals
  -/


theorem univ_pi_piecewise_univ {ι : Type*} {α : ι → Type*} (s : Set ι) (t : ∀ i, Set (α i))
                                                                                    /-
                                                                                      ι : Type u_8
                                                                                      α : ι → Type u_9
                                                                                      s : Set ι
                                                                                      t : (i : ι) → Set (α i)
                                                                                      inst✝ : (x : ι) → Decidable (Membership.mem s x)
                                                                                      ⊢ Eq (Set.univ.pi (s.piecewise t fun x => Set.univ)) (s.pi t)
                                                                                    -/
    [∀ x, Decidable (x ∈ s)] : pi univ (s.piecewise t fun _ => univ) = pi s t := by simp
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


theorem Injective.comp_injOn (hg : Injective g) (hf : s.InjOn f) : s.InjOn (g ∘ f) :=
  hg.injOn.comp hf (mapsTo_univ _ _)


theorem Surjective.surjOn (hf : Surjective f) (s : Set β) : SurjOn f univ s :=
  (surjective_iff_surjOn_univ.1 hf).mono (Subset.refl _) (subset_univ _)


theorem LeftInverse.leftInvOn {g : β → α} (h : LeftInverse f g) (s : Set β) : LeftInvOn f g s :=
  fun x _ => h x


theorem RightInverse.rightInvOn {g : β → α} (h : RightInverse f g) (s : Set α) :
    RightInvOn f g s := fun x _ => h x


theorem LeftInverse.rightInvOn_range {g : β → α} (h : LeftInverse f g) :
    RightInvOn f g (range g) :=
  forall_mem_range.2 fun i => congr_arg g (h i)


theorem mapsTo_image (h : Semiconj f fa fb) (ha : MapsTo fa s t) : MapsTo fb (f '' s) (f '' t) :=
  fun _y ⟨x, hx, hy⟩ => hy ▸ ⟨fa x, ha hx, h x⟩


theorem mapsTo_range (h : Semiconj f fa fb) : MapsTo fb (range f) (range f) := fun _y ⟨x, hy⟩ =>
  hy ▸ ⟨fa x, h x⟩


theorem surjOn_image (h : Semiconj f fa fb) (ha : SurjOn fa s t) : SurjOn fb (f '' s) (f '' t) := by
  /-
    α : Type u_1
    β : Type u_2
    fa : α → α
    fb : β → β
    f : α → β
    s t : Set α
    h : Function.Semiconj f fa fb
    ha : Set.SurjOn fa s t
    ⊢ Set.SurjOn fb (Set.image f s) (Set.image f t)
  -/
  rintro y ⟨x, hxt, rfl⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    fa : α → α
    fb : β → β
    f : α → β
    s t : Set α
    h : Function.Semiconj f fa fb
    ha : Set.SurjOn fa s t
    x : α
    hxt : Membership.mem t x
    ⊢ Membership.mem (Set.image fb (Set.image f s)) (f x)
  -/
  rcases ha hxt with ⟨x, hxs, rfl⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    fa : α → α
    fb : β → β
    f : α → β
    s t : Set α
    h : Function.Semiconj f fa fb
    ha : Set.SurjOn fa s t
    x : α
    hxs : Membership.mem s x
    hxt : Membership.mem t (fa x)
    ⊢ Membership.mem (Set.image fb (Set.image f s)) (f (fa x))
  -/
  rw [h x]
  /-
    case intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    fa : α → α
    fb : β → β
    f : α → β
    s t : Set α
    h : Function.Semiconj f fa fb
    ha : Set.SurjOn fa s t
    x : α
    hxs : Membership.mem s x
    hxt : Membership.mem t (fa x)
    ⊢ Membership.mem (Set.image fb (Set.image f s)) (fb (f x))
  -/
  exact mem_image_of_mem _ (mem_image_of_mem _ hxs)
  /-
    🎉 no goals
  -/


theorem surjOn_range (h : Semiconj f fa fb) (ha : Surjective fa) :
    SurjOn fb (range f) (range f) := by
  /-
    α : Type u_1
    β : Type u_2
    fa : α → α
    fb : β → β
    f : α → β
    h : Function.Semiconj f fa fb
    ha : Function.Surjective fa
    ⊢ Set.SurjOn fb (Set.range f) (Set.range f)
  -/
  rw [← image_univ]
  /-
    α : Type u_1
    β : Type u_2
    fa : α → α
    fb : β → β
    f : α → β
    h : Function.Semiconj f fa fb
    ha : Function.Surjective fa
    ⊢ Set.SurjOn fb (Set.image f Set.univ) (Set.image f Set.univ)
  -/
  exact h.surjOn_image (ha.surjOn univ)
  /-
    🎉 no goals
  -/


theorem injOn_image (h : Semiconj f fa fb) (ha : InjOn fa s) (hf : InjOn f (fa '' s)) :
    InjOn fb (f '' s) := by
  /-
    α : Type u_1
    β : Type u_2
    fa : α → α
    fb : β → β
    f : α → β
    s : Set α
    h : Function.Semiconj f fa fb
    ha : Set.InjOn fa s
    hf : Set.InjOn f (Set.image fa s)
    ⊢ Set.InjOn fb (Set.image f s)
  -/
  rintro _ ⟨x, hx, rfl⟩ _ ⟨y, hy, rfl⟩ H
  /-
    case intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    fa : α → α
    fb : β → β
    f : α → β
    s : Set α
    h : Function.Semiconj f fa fb
    ha : Set.InjOn fa s
    hf : Set.InjOn f (Set.image fa s)
    x : α
    hx : Membership.mem s x
    y : α
    hy : Membership.mem s y
    H : Eq (fb (f x)) (fb (f y))
    ⊢ Eq (f x) (f y)
  -/
  simp only [← h.eq] at H
  /-
    case intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    fa : α → α
    fb : β → β
    f : α → β
    s : Set α
    h : Function.Semiconj f fa fb
    ha : Set.InjOn fa s
    hf : Set.InjOn f (Set.image fa s)
    x : α
    hx : Membership.mem s x
    y : α
    hy : Membership.mem s y
    H : Eq (f (fa x)) (f (fa y))
    ⊢ Eq (f x) (f y)
  -/
  exact congr_arg f (ha hx hy <| hf (mem_image_of_mem fa hx) (mem_image_of_mem fa hy) H)
  /-
    🎉 no goals
  -/


theorem injOn_range (h : Semiconj f fa fb) (ha : Injective fa) (hf : InjOn f (range fa)) :
    InjOn fb (range f) := by
  /-
    α : Type u_1
    β : Type u_2
    fa : α → α
    fb : β → β
    f : α → β
    h : Function.Semiconj f fa fb
    ha : Function.Injective fa
    hf : Set.InjOn f (Set.range fa)
    ⊢ Set.InjOn fb (Set.range f)
  -/
  rw [← image_univ] at *
  /-
    α : Type u_1
    β : Type u_2
    fa : α → α
    fb : β → β
    f : α → β
    h : Function.Semiconj f fa fb
    ha : Function.Injective fa
    hf : Set.InjOn f (Set.image fa Set.univ)
    ⊢ Set.InjOn fb (Set.image f Set.univ)
  -/
  exact h.injOn_image ha.injOn hf
  /-
    🎉 no goals
  -/


theorem bijOn_image (h : Semiconj f fa fb) (ha : BijOn fa s t) (hf : InjOn f t) :
    BijOn fb (f '' s) (f '' t) :=
  ⟨h.mapsTo_image ha.mapsTo, h.injOn_image ha.injOn (ha.image_eq.symm ▸ hf),
    h.surjOn_image ha.surjOn⟩


theorem bijOn_range (h : Semiconj f fa fb) (ha : Bijective fa) (hf : Injective f) :
    BijOn fb (range f) (range f) := by
  /-
    α : Type u_1
    β : Type u_2
    fa : α → α
    fb : β → β
    f : α → β
    h : Function.Semiconj f fa fb
    ha : Function.Bijective fa
    hf : Function.Injective f
    ⊢ Set.BijOn fb (Set.range f) (Set.range f)
  -/
  rw [← image_univ]
  /-
    α : Type u_1
    β : Type u_2
    fa : α → α
    fb : β → β
    f : α → β
    h : Function.Semiconj f fa fb
    ha : Function.Bijective fa
    hf : Function.Injective f
    ⊢ Set.BijOn fb (Set.image f Set.univ) (Set.image f Set.univ)
  -/
  exact h.bijOn_image (bijective_iff_bijOn_univ.1 ha) hf.injOn
  /-
    🎉 no goals
  -/


theorem mapsTo_preimage (h : Semiconj f fa fb) {s t : Set β} (hb : MapsTo fb s t) :
                                                    /-
                                                      α : Type u_1
                                                      β : Type u_2
                                                      fa : α → α
                                                      fb : β → β
                                                      f : α → β
                                                      h : Function.Semiconj f fa fb
                                                      s t : Set β
                                                      hb : Set.MapsTo fb s t
                                                      x : α
                                                      hx : Membership.mem (Set.preimage f s) x
                                                      ⊢ Membership.mem (Set.preimage f t) (fa x)
                                                    -/
    MapsTo fa (f ⁻¹' s) (f ⁻¹' t) := fun x hx => by simp only [mem_preimage, h x, hb hx]
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem injOn_preimage (h : Semiconj f fa fb) {s : Set β} (hb : InjOn fb s)
    (hf : InjOn f (f ⁻¹' s)) : InjOn fa (f ⁻¹' s) := by
  /-
    α : Type u_1
    β : Type u_2
    fa : α → α
    fb : β → β
    f : α → β
    h : Function.Semiconj f fa fb
    s : Set β
    hb : Set.InjOn fb s
    hf : Set.InjOn f (Set.preimage f s)
    ⊢ Set.InjOn fa (Set.preimage f s)
  -/
  intro x hx y hy H
  /-
    α : Type u_1
    β : Type u_2
    fa : α → α
    fb : β → β
    f : α → β
    h : Function.Semiconj f fa fb
    s : Set β
    hb : Set.InjOn fb s
    hf : Set.InjOn f (Set.preimage f s)
    x : α
    hx : Membership.mem (Set.preimage f s) x
    y : α
    hy : Membership.mem (Set.preimage f s) y
    H : Eq (fa x) (fa y)
    ⊢ Eq x y
  -/
  have := congr_arg f H
  /-
    α : Type u_1
    β : Type u_2
    fa : α → α
    fb : β → β
    f : α → β
    h : Function.Semiconj f fa fb
    s : Set β
    hb : Set.InjOn fb s
    hf : Set.InjOn f (Set.preimage f s)
    x : α
    hx : Membership.mem (Set.preimage f s) x
    y : α
    hy : Membership.mem (Set.preimage f s) y
    H : Eq (fa x) (fa y)
    this : Eq (f (fa x)) (f (fa y))
    ⊢ Eq x y
  -/
  rw [h.eq, h.eq] at this
  /-
    α : Type u_1
    β : Type u_2
    fa : α → α
    fb : β → β
    f : α → β
    h : Function.Semiconj f fa fb
    s : Set β
    hb : Set.InjOn fb s
    hf : Set.InjOn f (Set.preimage f s)
    x : α
    hx : Membership.mem (Set.preimage f s) x
    y : α
    hy : Membership.mem (Set.preimage f s) y
    H : Eq (fa x) (fa y)
    this : Eq (fb (f x)) (fb (f y))
    ⊢ Eq x y
  -/
  exact hf hx hy (hb hx hy this)
  /-
    🎉 no goals
  -/


theorem update_comp_eq_of_not_mem_range' {α : Sort*} {β : Type*} {γ : β → Sort*} [DecidableEq β]
    (g : ∀ b, γ b) {f : α → β} {i : β} (a : γ i) (h : i ∉ Set.range f) :
    (fun j => update g i a (f j)) = fun j => g (f j) :=
  (update_comp_eq_of_forall_ne' _ _) fun x hx => h ⟨x, hx⟩


/-- Non-dependent version of `Function.update_comp_eq_of_not_mem_range'` -/
theorem update_comp_eq_of_not_mem_range {α : Sort*} {β : Type*} {γ : Sort*} [DecidableEq β]
    (g : β → γ) {f : α → β} {i : β} (a : γ) (h : i ∉ Set.range f) : update g i a ∘ f = g ∘ f :=
  update_comp_eq_of_not_mem_range' g a h


theorem insert_injOn (s : Set α) : sᶜ.InjOn fun a => insert a s := fun _a ha _ _ =>
  (insert_inj ha).1


lemma apply_eq_of_range_eq_singleton {f : α → β} {b : β} (h : range f = {b}) (a : α) :
    f a = b := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    b : β
    h : Eq (Set.range f) (Singleton.singleton b)
    a : α
    ⊢ Eq (f a) b
  -/
  simpa only [h, mem_singleton_iff] using mem_range_self (f := f) a
  /-
    🎉 no goals
  -/


protected lemma MapsTo.extendDomain (h : MapsTo g s t) :
    MapsTo (g.extendDomain f) ((↑) ∘ f '' s) ((↑) ∘ f '' t) := by
  /-
    α : Type u_1
    β : Type u_2
    p : β → Prop
    inst✝ : DecidablePred p
    f : Equiv α (Subtype p)
    g : Equiv.Perm α
    s t : Set α
    h : Set.MapsTo (⇑g) s t
    ⊢ Set.MapsTo (⇑(g.extendDomain f)) (Set.image (Function.comp Subtype.val ⇑f) s …
  -/
  rintro _ ⟨a, ha, rfl⟩; exact ⟨_, h ha, by simp_rw [Function.comp_apply, extendDomain_apply_image]⟩
                         /-
                           🎉 no goals
                         -/


protected lemma SurjOn.extendDomain (h : SurjOn g s t) :
    SurjOn (g.extendDomain f) ((↑) ∘ f '' s) ((↑) ∘ f '' t) := by
  /-
    α : Type u_1
    β : Type u_2
    p : β → Prop
    inst✝ : DecidablePred p
    f : Equiv α (Subtype p)
    g : Equiv.Perm α
    s t : Set α
    h : Set.SurjOn (⇑g) s t
    ⊢ Set.SurjOn (⇑(g.extendDomain f)) (Set.image (Function.comp Subtype.val ⇑f) s …
  -/
  rintro _ ⟨a, ha, rfl⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    p : β → Prop
    inst✝ : DecidablePred p
    f : Equiv α (Subtype p)
    g : Equiv.Perm α
    s t : Set α
    h : Set.SurjOn (⇑g) s t
    a : α
    ha : Membership.mem t a
    ⊢ Membership.mem (Set.image (⇑(g.extendDomain f)) (Set.image (Function.comp Su …
  -/
  obtain ⟨b, hb, rfl⟩ := h ha
  /-
    case intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    p : β → Prop
    inst✝ : DecidablePred p
    f : Equiv α (Subtype p)
    g : Equiv.Perm α
    s t : Set α
    h : Set.SurjOn (⇑g) s t
    b : α
    hb : Membership.mem s b
    ha : Membership.mem t (g b)
    ⊢ Membership.mem (Set.image (⇑(g.extendDomain f)) (Set.image (Function.comp Su …
  -/
  exact ⟨_, ⟨_, hb, rfl⟩, by simp_rw [Function.comp_apply, extendDomain_apply_image]⟩
  /-
    🎉 no goals
  -/


protected lemma BijOn.extendDomain (h : BijOn g s t) :
    BijOn (g.extendDomain f) ((↑) ∘ f '' s) ((↑) ∘ f '' t) :=
  ⟨h.mapsTo.extendDomain, (g.extendDomain f).injective.injOn, h.surjOn.extendDomain⟩


protected lemma LeftInvOn.extendDomain (h : LeftInvOn g₁ g₂ s) :
    LeftInvOn (g₁.extendDomain f) (g₂.extendDomain f) ((↑) ∘ f '' s) := by
  /-
    α : Type u_1
    β : Type u_2
    p : β → Prop
    inst✝ : DecidablePred p
    f : Equiv α (Subtype p)
    g₁ g₂ : Equiv.Perm α
    s : Set α
    h : Set.LeftInvOn (⇑g₁) (⇑g₂) s
    ⊢ Set.LeftInvOn (⇑(g₁.extendDomain f)) (⇑(g₂.extendDomain f)) (Set.image (Func …
  -/
  rintro _ ⟨a, ha, rfl⟩; simp_rw [Function.comp_apply, extendDomain_apply_image, h ha]
                         /-
                           🎉 no goals
                         -/


protected lemma RightInvOn.extendDomain (h : RightInvOn g₁ g₂ t) :
    RightInvOn (g₁.extendDomain f) (g₂.extendDomain f) ((↑) ∘ f '' t) := by
  /-
    α : Type u_1
    β : Type u_2
    p : β → Prop
    inst✝ : DecidablePred p
    f : Equiv α (Subtype p)
    g₁ g₂ : Equiv.Perm α
    t : Set α
    h : Set.RightInvOn (⇑g₁) (⇑g₂) t
    ⊢ Set.RightInvOn (⇑(g₁.extendDomain f)) (⇑(g₂.extendDomain f)) (Set.image (Fun …
  -/
  rintro _ ⟨a, ha, rfl⟩; simp_rw [Function.comp_apply, extendDomain_apply_image, h ha]
                         /-
                           🎉 no goals
                         -/


protected lemma InvOn.extendDomain (h : InvOn g₁ g₂ s t) :
    InvOn (g₁.extendDomain f) (g₂.extendDomain f) ((↑) ∘ f '' s) ((↑) ∘ f '' t) :=
  ⟨h.1.extendDomain, h.2.extendDomain⟩


lemma InjOn.prodMap (h₁ : s₁.InjOn f₁) (h₂ : s₂.InjOn f₂) :
    (s₁ ×ˢ s₂).InjOn fun x ↦ (f₁ x.1, f₂ x.2) :=
                     /-
                       α₁ : Type u_7
                       α₂ : Type u_8
                       β₁ : Type u_9
                       β₂ : Type u_10
                       s₁ : Set α₁
                       s₂ : Set α₂
                       f₁ : α₁ → β₁
                       f₂ : α₂ → β₂
                       h₁ : Set.InjOn f₁ s₁
                       h₂ : Set.InjOn f₂ s₂
                       x : Prod α₁ α₂
                       hx : Membership.mem (SProd.sprod s₁ s₂) x
                       y : Prod α₁ α₂
                       hy : Membership.mem (SProd.sprod s₁ s₂) y
                       ⊢ Eq ((fun x => { fst := f₁ x.1, snd := f₂ x.2 }) x) ((fun x => { fst := f₁ x. …
                     -/
  fun x hx y hy ↦ by simp_rw [Prod.ext_iff]; exact And.imp (h₁ hx.1 hy.1) (h₂ hx.2 hy.2)
                                             /-
                                               🎉 no goals
                                             -/


lemma SurjOn.prodMap (h₁ : SurjOn f₁ s₁ t₁) (h₂ : SurjOn f₂ s₂ t₂) :
    SurjOn (fun x ↦ (f₁ x.1, f₂ x.2)) (s₁ ×ˢ s₂) (t₁ ×ˢ t₂) := by
  /-
    α₁ : Type u_7
    α₂ : Type u_8
    β₁ : Type u_9
    β₂ : Type u_10
    s₁ : Set α₁
    s₂ : Set α₂
    t₁ : Set β₁
    t₂ : Set β₂
    f₁ : α₁ → β₁
    f₂ : α₂ → β₂
    h₁ : Set.SurjOn f₁ s₁ t₁
    h₂ : Set.SurjOn f₂ s₂ t₂
    ⊢ Set.SurjOn (fun x => { fst := f₁ x.1, snd := f₂ x.2 }) (SProd.sprod s₁ s₂) ( …
  -/
  rintro x hx
  /-
    α₁ : Type u_7
    α₂ : Type u_8
    β₁ : Type u_9
    β₂ : Type u_10
    s₁ : Set α₁
    s₂ : Set α₂
    t₁ : Set β₁
    t₂ : Set β₂
    f₁ : α₁ → β₁
    f₂ : α₂ → β₂
    h₁ : Set.SurjOn f₁ s₁ t₁
    h₂ : Set.SurjOn f₂ s₂ t₂
    x : Prod β₁ β₂
    hx : Membership.mem (SProd.sprod t₁ t₂) x
    ⊢ Membership.mem (Set.image (fun x => { fst := f₁ x.1, snd := f₂ x.2 }) (SProd …
  -/
  obtain ⟨a₁, ha₁, hx₁⟩ := h₁ hx.1
  /-
    case intro.intro
    α₁ : Type u_7
    α₂ : Type u_8
    β₁ : Type u_9
    β₂ : Type u_10
    s₁ : Set α₁
    s₂ : Set α₂
    t₁ : Set β₁
    t₂ : Set β₂
    f₁ : α₁ → β₁
    f₂ : α₂ → β₂
    h₁ : Set.SurjOn f₁ s₁ t₁
    h₂ : Set.SurjOn f₂ s₂ t₂
    x : Prod β₁ β₂
    hx : Membership.mem (SProd.sprod t₁ t₂) x
    a₁ : α₁
    ha₁ : Membership.mem s₁ a₁
    hx₁ : Eq (f₁ a₁) x.1
    ⊢ Membership.mem (Set.image (fun x => { fst := f₁ x.1, snd := f₂ x.2 }) (SProd …
  -/
  obtain ⟨a₂, ha₂, hx₂⟩ := h₂ hx.2
  /-
    case intro.intro.intro.intro
    α₁ : Type u_7
    α₂ : Type u_8
    β₁ : Type u_9
    β₂ : Type u_10
    s₁ : Set α₁
    s₂ : Set α₂
    t₁ : Set β₁
    t₂ : Set β₂
    f₁ : α₁ → β₁
    f₂ : α₂ → β₂
    h₁ : Set.SurjOn f₁ s₁ t₁
    h₂ : Set.SurjOn f₂ s₂ t₂
    x : Prod β₁ β₂
    hx : Membership.mem (SProd.sprod t₁ t₂) x
    a₁ : α₁
    ha₁ : Membership.mem s₁ a₁
    hx₁ : Eq (f₁ a₁) x.1
    a₂ : α₂
    ha₂ : Membership.mem s₂ a₂
    hx₂ : Eq (f₂ a₂) x.2
    ⊢ Membership.mem (Set.image (fun x => { fst := f₁ x.1, snd := f₂ x.2 }) (SProd …
  -/
  exact ⟨(a₁, a₂), ⟨ha₁, ha₂⟩, Prod.ext hx₁ hx₂⟩
  /-
    🎉 no goals
  -/


lemma MapsTo.prodMap (h₁ : MapsTo f₁ s₁ t₁) (h₂ : MapsTo f₂ s₂ t₂) :
    MapsTo (fun x ↦ (f₁ x.1, f₂ x.2)) (s₁ ×ˢ s₂) (t₁ ×ˢ t₂) :=
  fun _x hx ↦ ⟨h₁ hx.1, h₂ hx.2⟩


lemma BijOn.prodMap (h₁ : BijOn f₁ s₁ t₁) (h₂ : BijOn f₂ s₂ t₂) :
    BijOn (fun x ↦ (f₁ x.1, f₂ x.2)) (s₁ ×ˢ s₂) (t₁ ×ˢ t₂) :=
  ⟨h₁.mapsTo.prodMap h₂.mapsTo, h₁.injOn.prodMap h₂.injOn, h₁.surjOn.prodMap h₂.surjOn⟩


lemma LeftInvOn.prodMap (h₁ : LeftInvOn g₁ f₁ s₁) (h₂ : LeftInvOn g₂ f₂ s₂) :
    LeftInvOn (fun x ↦ (g₁ x.1, g₂ x.2)) (fun x ↦ (f₁ x.1, f₂ x.2)) (s₁ ×ˢ s₂) :=
  fun _x hx ↦ Prod.ext (h₁ hx.1) (h₂ hx.2)


lemma RightInvOn.prodMap (h₁ : RightInvOn g₁ f₁ t₁) (h₂ : RightInvOn g₂ f₂ t₂) :
    RightInvOn (fun x ↦ (g₁ x.1, g₂ x.2)) (fun x ↦ (f₁ x.1, f₂ x.2)) (t₁ ×ˢ t₂) :=
  fun _x hx ↦ Prod.ext (h₁ hx.1) (h₂ hx.2)


lemma InvOn.prodMap (h₁ : InvOn g₁ f₁ s₁ t₁) (h₂ : InvOn g₂ f₂ s₂ t₂) :
    InvOn (fun x ↦ (g₁ x.1, g₂ x.2)) (fun x ↦ (f₁ x.1, f₂ x.2)) (s₁ ×ˢ s₂) (t₁ ×ˢ t₂) :=
  ⟨h₁.1.prodMap h₂.1, h₁.2.prodMap h₂.2⟩


lemma bijOn' (h₁ : MapsTo e s t) (h₂ : MapsTo e.symm t s) : BijOn e s t :=
  ⟨h₁, e.injective.injOn, fun b hb ↦ ⟨e.symm b, h₂ hb, apply_symm_apply _ _⟩⟩


protected lemma bijOn (h : ∀ a, e a ∈ t ↔ a ∈ s) : BijOn e s t :=
                                                      /-
                                                        α : Type u_1
                                                        β : Type u_2
                                                        e : Equiv α β
                                                        s : Set α
                                                        t : Set β
                                                        h : ∀ (a : α), Iff (Membership.mem t (e a)) (Membership.mem s a)
                                                        b : β
                                                        hb : Membership.mem t b
                                                        ⊢ Membership.mem t (e (e.symm b))
                                                      -/
  e.bijOn' (fun _ ↦ (h _).2) fun b hb ↦ (h _).1 <| by rwa [apply_symm_apply]
                                                      /-
                                                        🎉 no goals
                                                      -/


lemma invOn : InvOn e e.symm t s :=
  ⟨e.rightInverse_symm.leftInvOn _, e.leftInverse_symm.leftInvOn _⟩


lemma bijOn_image : BijOn e s (e '' s) := e.injective.injOn.bijOn_image

lemma bijOn_symm_image : BijOn e.symm (e '' s) s := e.bijOn_image.symm e.invOn


@[simp] lemma bijOn_symm : BijOn e.symm t s ↔ BijOn e s t := bijOn_comm e.symm.invOn


alias ⟨_root_.Set.BijOn.of_equiv_symm, _root_.Set.BijOn.equiv_symm⟩ := bijOn_symm


lemma bijOn_swap (ha : a ∈ s) (hb : b ∈ s) : BijOn (swap a b) s s :=
  (swap a b).bijOn fun x ↦ by
    /-
      α : Type u_1
      s : Set α
      inst✝ : DecidableEq α
      a b : α
      ha : Membership.mem s a
      hb : Membership.mem s b
      x : α
      ⊢ Iff (Membership.mem s ((Equiv.swap a b) x)) (Membership.mem s x)
    -/
    obtain rfl | hxa := eq_or_ne x a <;>
    /-
      case inl
      α : Type u_1
      s : Set α
      inst✝ : DecidableEq α
      b : α
      hb : Membership.mem s b
      x : α
      ha : Membership.mem s x
      ⊢ Iff (Membership.mem s ((Equiv.swap x b) x)) (Membership.mem s x)
    -/
    obtain rfl | hxb := eq_or_ne x b <;>
    /-
      case inl.inl
      α : Type u_1
      s : Set α
      inst✝ : DecidableEq α
      x : α
      ha hb : Membership.mem s x
      ⊢ Iff (Membership.mem s ((Equiv.swap x x) x)) (Membership.mem s x)
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    simp [*, swap_apply_of_ne_of_ne]
    /-
      🎉 no goals
    -/


/-- **Vertical line test** for functions.

Let `f : α → β × γ` be a function to a product. Assume that `f` is surjective on the first factor
and that the image of `f` intersects every "vertical line" `{(b, c) | c : γ}` at most once.
Then the image of `f` is the graph of some monoid homomorphism `f' : β → γ`. -/
lemma exists_range_eq_graphOn_univ {f : α → β × γ} (hf₁ : Surjective (Prod.fst ∘ f))
    (hf : ∀ g₁ g₂, (f g₁).1 = (f g₂).1 → (f g₁).2 = (f g₂).2) :
    ∃ f' : β → γ, range f = univ.graphOn f' := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → Prod β γ
    hf₁ : Function.Surjective (Function.comp Prod.fst f)
    hf : ∀ (g₁ g₂ : α), Eq (f g₁).1 (f g₂).1 → Eq (f g₁).2 (f g₂).2
    ⊢ Exists fun f' => Eq (Set.range f) (Set.graphOn f' Set.univ)
  -/
  refine ⟨fun h ↦ (f (hf₁ h).choose).snd, ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → Prod β γ
    hf₁ : Function.Surjective (Function.comp Prod.fst f)
    hf : ∀ (g₁ g₂ : α), Eq (f g₁).1 (f g₂).1 → Eq (f g₁).2 (f g₂).2
    ⊢ Eq (Set.range f) (Set.graphOn (fun h => (f ⋯.choose).2) Set.univ)
  -/
  ext x
  /-
    case h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → Prod β γ
    hf₁ : Function.Surjective (Function.comp Prod.fst f)
    hf : ∀ (g₁ g₂ : α), Eq (f g₁).1 (f g₂).1 → Eq (f g₁).2 (f g₂).2
    x : Prod β γ
    ⊢ Iff (Membership.mem (Set.range f) x) (Membership.mem (Set.graphOn (fun h =>  …
  -/
  simp only [mem_range, comp_apply, mem_graphOn, mem_univ, true_and]
  /-
    case h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → Prod β γ
    hf₁ : Function.Surjective (Function.comp Prod.fst f)
    hf : ∀ (g₁ g₂ : α), Eq (f g₁).1 (f g₂).1 → Eq (f g₁).2 (f g₂).2
    x : Prod β γ
    ⊢ Iff (Exists fun y => Eq (f y) x) (Eq (f ⋯.choose).2 x.2)
  -/
  refine ⟨?_, fun hi ↦ ⟨(hf₁ x.1).choose, Prod.ext (hf₁ x.1).choose_spec hi⟩⟩
  /-
    case h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → Prod β γ
    hf₁ : Function.Surjective (Function.comp Prod.fst f)
    hf : ∀ (g₁ g₂ : α), Eq (f g₁).1 (f g₂).1 → Eq (f g₁).2 (f g₂).2
    x : Prod β γ
    ⊢ (Exists fun y => Eq (f y) x) → Eq (f ⋯.choose).2 x.2
  -/
  rintro ⟨g, rfl⟩
  /-
    case h.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → Prod β γ
    hf₁ : Function.Surjective (Function.comp Prod.fst f)
    hf : ∀ (g₁ g₂ : α), Eq (f g₁).1 (f g₂).1 → Eq (f g₁).2 (f g₂).2
    g : α
    ⊢ Eq (f ⋯.choose).2 (f g).2
  -/
  exact hf _ _ (hf₁ (f g).1).choose_spec
  /-
    🎉 no goals
  -/


/-- **Line test** for equivalences.

Let `f : α → β × γ` be a homomorphism to a product of monoids. Assume that `f` is surjective on both
factors and that the image of `f` intersects every "vertical line" `{(b, c) | c : γ}` and every
"horizontal line" `{(b, c) | b : β}` at most once. Then the image of `f` is the graph of some
equivalence `f' : β ≃ γ`. -/
lemma exists_equiv_range_eq_graphOn_univ {f : α → β × γ} (hf₁ : Surjective (Prod.fst ∘ f))
    (hf₂ : Surjective (Prod.snd ∘ f)) (hf : ∀ g₁ g₂, (f g₁).1 = (f g₂).1 ↔ (f g₁).2 = (f g₂).2) :
    ∃ e : β ≃ γ, range f = univ.graphOn e := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → Prod β γ
    hf₁ : Function.Surjective (Function.comp Prod.fst f)
    hf₂ : Function.Surjective (Function.comp Prod.snd f)
    hf : ∀ (g₁ g₂ : α), Iff (Eq (f g₁).1 (f g₂).1) (Eq (f g₁).2 (f g₂).2)
    ⊢ Exists fun e => Eq (Set.range f) (Set.graphOn (⇑e) Set.univ)
  -/
  obtain ⟨e₁, he₁⟩ := exists_range_eq_graphOn_univ hf₁ fun _ _ ↦ (hf _ _).1
  obtain ⟨e₂, he₂⟩ := exists_range_eq_graphOn_univ (f := Equiv.prodComm _ _ ∘ f) (by simpa) <|
    by simp [hf]
  have he₁₂ h i : e₁ h = i ↔ e₂ i = h := by
    rw [Set.ext_iff] at he₁ he₂
    aesop (add simp [Prod.swap_eq_iff_eq_swap])
  exact ⟨
  { toFun := e₁
    invFun := e₂
    left_inv := fun h ↦ by rw [← he₁₂]
    right_inv := fun i ↦ by rw [he₁₂] }, he₁⟩


/-- **Vertical line test** for functions.

Let `s : Set (β × γ)` be a set in a product. Assume that `s` maps bijectively to the first factor.
Then `s` is the graph of some function `f : β → γ`. -/
lemma exists_eq_mgraphOn_univ {s : Set (β × γ)}
    (hs₁ : Bijective (Prod.fst ∘ (Subtype.val : s → β × γ))) : ∃ f : β → γ, s = univ.graphOn f := by
  simpa using exists_range_eq_graphOn_univ hs₁.surjective
    fun a b h ↦ congr_arg (Prod.snd ∘ (Subtype.val : s → β × γ)) (hs₁.injective h)


