@[simp]
theorem preimage_empty : f ⁻¹' ∅ = ∅ :=
  rfl


theorem preimage_congr {f g : α → β} {s : Set β} (h : ∀ x : α, f x = g x) : f ⁻¹' s = g ⁻¹' s := by
  /-
    α : Type u_1
    β : Type u_2
    f g : α → β
    s : Set β
    h : ∀ (x : α), Eq (f x) (g x)
    ⊢ Eq (Set.preimage f s) (Set.preimage g s)
  -/
  congr with x
  /-
    case h
    α : Type u_1
    β : Type u_2
    f g : α → β
    s : Set β
    h : ∀ (x : α), Eq (f x) (g x)
    x : α
    ⊢ Iff (Membership.mem (Set.preimage f s) x) (Membership.mem (Set.preimage g s) …
  -/
  simp [h]
  /-
    🎉 no goals
  -/


@[gcongr]
theorem preimage_mono {s t : Set β} (h : s ⊆ t) : f ⁻¹' s ⊆ f ⁻¹' t := fun _ hx => h hx


@[simp, mfld_simps]
theorem preimage_univ : f ⁻¹' univ = univ :=
  rfl


theorem subset_preimage_univ {s : Set α} : s ⊆ f ⁻¹' univ :=
  subset_univ _


@[simp, mfld_simps]
theorem preimage_inter {s t : Set β} : f ⁻¹' (s ∩ t) = f ⁻¹' s ∩ f ⁻¹' t :=
  rfl


@[simp]
theorem preimage_union {s t : Set β} : f ⁻¹' (s ∪ t) = f ⁻¹' s ∪ f ⁻¹' t :=
  rfl


@[simp]
theorem preimage_compl {s : Set β} : f ⁻¹' sᶜ = (f ⁻¹' s)ᶜ :=
  rfl


@[simp]
theorem preimage_diff (f : α → β) (s t : Set β) : f ⁻¹' (s \ t) = f ⁻¹' s \ f ⁻¹' t :=
  rfl


open scoped symmDiff in
@[simp]
lemma preimage_symmDiff {f : α → β} (s t : Set β) : f ⁻¹' (s ∆ t) = (f ⁻¹' s) ∆ (f ⁻¹' t) :=
  rfl


@[simp]
theorem preimage_ite (f : α → β) (s t₁ t₂ : Set β) :
    f ⁻¹' s.ite t₁ t₂ = (f ⁻¹' s).ite (f ⁻¹' t₁) (f ⁻¹' t₂) :=
  rfl


@[simp]
theorem preimage_setOf_eq {p : α → Prop} {f : β → α} : f ⁻¹' { a | p a } = { a | p (f a) } :=
  rfl


@[simp]
theorem preimage_id_eq : preimage (id : α → α) = id :=
  rfl


@[mfld_simps]
theorem preimage_id {s : Set α} : id ⁻¹' s = s :=
  rfl


@[simp, mfld_simps]
theorem preimage_id' {s : Set α} : (fun x => x) ⁻¹' s = s :=
  rfl


@[simp]
theorem preimage_const_of_mem {b : β} {s : Set β} (h : b ∈ s) : (fun _ : α => b) ⁻¹' s = univ :=
  eq_univ_of_forall fun _ => h


@[simp]
theorem preimage_const_of_not_mem {b : β} {s : Set β} (h : b ∉ s) : (fun _ : α => b) ⁻¹' s = ∅ :=
  eq_empty_of_subset_empty fun _ hx => h hx


theorem preimage_const (b : β) (s : Set β) [Decidable (b ∈ s)] :
    (fun _ : α => b) ⁻¹' s = if b ∈ s then univ else ∅ := by
  /-
    α : Type u_1
    β : Type u_2
    b : β
    s : Set β
    inst✝ : Decidable (Membership.mem s b)
    ⊢ Eq (Set.preimage (fun x => b) s) (ite (Membership.mem s b) Set.univ EmptyCol …
  -/
  split_ifs with hb
  /-
    case pos
    α : Type u_1
    β : Type u_2
    b : β
    s : Set β
    inst✝ : Decidable (Membership.mem s b)
    hb : Membership.mem s b
    ⊢ Eq (Set.preimage (fun x => b) s) Set.univ
  -/
  exacts [preimage_const_of_mem hb, preimage_const_of_not_mem hb]
  /-
    🎉 no goals
  -/


/-- If preimage of each singleton under `f : α → β` is either empty or the whole type,
then `f` is a constant. -/
lemma exists_eq_const_of_preimage_singleton [Nonempty β] {f : α → β}
    (hf : ∀ b : β, f ⁻¹' {b} = ∅ ∨ f ⁻¹' {b} = univ) : ∃ b, f = const α b := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : Nonempty β
    f : α → β
    hf : ∀ (b : β), Or (Eq (Set.preimage f (Singleton.singleton b)) EmptyCollectio …
    ⊢ Exists fun b => Eq f (Function.const α b)
  -/
  rcases em (∃ b, f ⁻¹' {b} = univ) with ⟨b, hb⟩ | hf'
    /-
      case inl.intro
      α : Type u_1
      β : Type u_2
      inst✝ : Nonempty β
      f : α → β
      hf : ∀ (b : β), Or (Eq (Set.preimage f (Singleton.singleton b)) EmptyCollectio …
      b : β
      hb : Eq (Set.preimage f (Singleton.singleton b)) Set.univ
      ⊢ Exists fun b => Eq f (Function.const α b)
    -/
  · exact ⟨b, funext fun x ↦ eq_univ_iff_forall.1 hb x⟩
    /-
      🎉 no goals
    -/
  · have : ∀ x b, f x ≠ b := fun x b ↦
      eq_empty_iff_forall_not_mem.1 ((hf b).resolve_right fun h ↦ hf' ⟨b, h⟩) x
    /-
      case inr
      α : Type u_1
      β : Type u_2
      inst✝ : Nonempty β
      f : α → β
      hf : ∀ (b : β), Or (Eq (Set.preimage f (Singleton.singleton b)) EmptyCollectio …
      hf' : Not (Exists fun b => Eq (Set.preimage f (Singleton.singleton b)) Set.univ)
      this : ∀ (x : α) (b : β), Ne (f x) b
      ⊢ Exists fun b => Eq f (Function.const α b)
    -/
    exact ⟨Classical.arbitrary β, funext fun x ↦ absurd rfl (this x _)⟩
    /-
      🎉 no goals
    -/


theorem preimage_comp {s : Set γ} : g ∘ f ⁻¹' s = f ⁻¹' (g ⁻¹' s) :=
  rfl


theorem preimage_comp_eq : preimage (g ∘ f) = preimage f ∘ preimage g :=
  rfl


theorem preimage_iterate_eq {f : α → α} {n : ℕ} : Set.preimage f^[n] = (Set.preimage f)^[n] := by
  induction n with
  | zero => simp
  | succ n ih => rw [iterate_succ, iterate_succ', preimage_comp_eq, ih]


theorem preimage_preimage {g : β → γ} {f : α → β} {s : Set γ} :
    f ⁻¹' (g ⁻¹' s) = (fun x => g (f x)) ⁻¹' s :=
  preimage_comp.symm


theorem eq_preimage_subtype_val_iff {p : α → Prop} {s : Set (Subtype p)} {t : Set α} :
    s = Subtype.val ⁻¹' t ↔ ∀ (x) (h : p x), (⟨x, h⟩ : Subtype p) ∈ s ↔ x ∈ t :=
  ⟨fun s_eq x h => by
    /-
      α : Type u_1
      p : α → Prop
      s : Set (Subtype p)
      t : Set α
      s_eq : Eq s (Set.preimage Subtype.val t)
      x : α
      h : p x
      ⊢ Iff (Membership.mem s ⟨x, h⟩) (Membership.mem t x)
    -/
    rw [s_eq]
    /-
      α : Type u_1
      p : α → Prop
      s : Set (Subtype p)
      t : Set α
      s_eq : Eq s (Set.preimage Subtype.val t)
      x : α
      h : p x
      ⊢ Iff (Membership.mem (Set.preimage Subtype.val t) ⟨x, h⟩) (Membership.mem t x)
    -/
    /-
      🎉 no goals
    -/
    simp, fun h => ext fun ⟨x, hx⟩ => by simp [h]⟩
                                         /-
                                           🎉 no goals
                                         -/


theorem nonempty_of_nonempty_preimage {s : Set β} {f : α → β} (hf : (f ⁻¹' s).Nonempty) :
    s.Nonempty :=
  let ⟨x, hx⟩ := hf
  ⟨f x, hx⟩


                                                                                        /-
                                                                                          α : Type u_1
                                                                                          p : α → Prop
                                                                                          ⊢ Eq (Set.preimage p (Singleton.singleton True)) (setOf fun a => p a)
                                                                                        -/
@[simp] theorem preimage_singleton_true (p : α → Prop) : p ⁻¹' {True} = {a | p a} := by ext; simp
                                                                                             /-
                                                                                               🎉 no goals
                                                                                             -/


                                                                                           /-
                                                                                             α : Type u_1
                                                                                             p : α → Prop
                                                                                             ⊢ Eq (Set.preimage p (Singleton.singleton False)) (setOf fun a => Not (p a))
                                                                                           -/
@[simp] theorem preimage_singleton_false (p : α → Prop) : p ⁻¹' {False} = {a | ¬p a} := by ext; simp
                                                                                                /-
                                                                                                  🎉 no goals
                                                                                                -/


theorem preimage_subtype_coe_eq_compl {s u v : Set α} (hsuv : s ⊆ u ∪ v)
    (H : s ∩ (u ∩ v) = ∅) : ((↑) : s → α) ⁻¹' u = ((↑) ⁻¹' v)ᶜ := by
  /-
    α : Type u_1
    s u v : Set α
    hsuv : HasSubset.Subset s (Union.union u v)
    H : Eq (Inter.inter s (Inter.inter u v)) EmptyCollection.emptyCollection
    ⊢ Eq (Set.preimage Subtype.val u) (HasCompl.compl (Set.preimage Subtype.val v))
  -/
  ext ⟨x, x_in_s⟩
  /-
    case h.mk
    α : Type u_1
    s u v : Set α
    hsuv : HasSubset.Subset s (Union.union u v)
    H : Eq (Inter.inter s (Inter.inter u v)) EmptyCollection.emptyCollection
    x : α
    x_in_s : Membership.mem s x
    ⊢ Iff (Membership.mem (Set.preimage Subtype.val u) ⟨x, x_in_s⟩) (Membership.me …
  -/
  constructor
    /-
      case h.mk.mp
      α : Type u_1
      s u v : Set α
      hsuv : HasSubset.Subset s (Union.union u v)
      H : Eq (Inter.inter s (Inter.inter u v)) EmptyCollection.emptyCollection
      x : α
      x_in_s : Membership.mem s x
      ⊢ Membership.mem (Set.preimage Subtype.val u) ⟨x, x_in_s⟩ → Membership.mem (Ha …
    -/
  · intro x_in_u x_in_v
    /-
      case h.mk.mp
      α : Type u_1
      s u v : Set α
      hsuv : HasSubset.Subset s (Union.union u v)
      H : Eq (Inter.inter s (Inter.inter u v)) EmptyCollection.emptyCollection
      x : α
      x_in_s : Membership.mem s x
      x_in_u : Membership.mem (Set.preimage Subtype.val u) ⟨x, x_in_s⟩
      x_in_v : Membership.mem (Set.preimage Subtype.val v) ⟨x, x_in_s⟩
      ⊢ False
    -/
    exact eq_empty_iff_forall_not_mem.mp H x ⟨x_in_s, ⟨x_in_u, x_in_v⟩⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mk.mpr
      α : Type u_1
      s u v : Set α
      hsuv : HasSubset.Subset s (Union.union u v)
      H : Eq (Inter.inter s (Inter.inter u v)) EmptyCollection.emptyCollection
      x : α
      x_in_s : Membership.mem s x
      ⊢ Membership.mem (HasCompl.compl (Set.preimage Subtype.val v)) ⟨x, x_in_s⟩ → M …
    -/
  · intro hx
    /-
      case h.mk.mpr
      α : Type u_1
      s u v : Set α
      hsuv : HasSubset.Subset s (Union.union u v)
      H : Eq (Inter.inter s (Inter.inter u v)) EmptyCollection.emptyCollection
      x : α
      x_in_s : Membership.mem s x
      hx : Membership.mem (HasCompl.compl (Set.preimage Subtype.val v)) ⟨x, x_in_s⟩
      ⊢ Membership.mem (Set.preimage Subtype.val u) ⟨x, x_in_s⟩
    -/
    exact Or.elim (hsuv x_in_s) id fun hx' => hx.elim hx'
    /-
      🎉 no goals
    -/


lemma preimage_subset {s t} (hs : s ⊆ f '' t) (hf : Set.InjOn f (f ⁻¹' s)) : f ⁻¹' s ⊆ t := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    s : Set β
    t : Set α
    hs : HasSubset.Subset s (Set.image f t)
    hf : Set.InjOn f (Set.preimage f s)
    ⊢ HasSubset.Subset (Set.preimage f s) t
  -/
  rintro a ha
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    s : Set β
    t : Set α
    hs : HasSubset.Subset s (Set.image f t)
    hf : Set.InjOn f (Set.preimage f s)
    a : α
    ha : Membership.mem (Set.preimage f s) a
    ⊢ Membership.mem t a
  -/
  obtain ⟨b, hb, hba⟩ := hs ha
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    f : α → β
    s : Set β
    t : Set α
    hs : HasSubset.Subset s (Set.image f t)
    hf : Set.InjOn f (Set.preimage f s)
    a : α
    ha : Membership.mem (Set.preimage f s) a
    b : α
    hb : Membership.mem t b
    hba : Eq (f b) (f a)
    ⊢ Membership.mem t a
  -/
  rwa [hf ha _ hba.symm]
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    s : Set β
    t : Set α
    hs : HasSubset.Subset s (Set.image f t)
    hf : Set.InjOn f (Set.preimage f s)
    a : α
    ha : Membership.mem (Set.preimage f s) a
    b : α
    hb : Membership.mem t b
    hba : Eq (f b) (f a)
    ⊢ Membership.mem (Set.preimage f s) b
  -/
  simpa [hba]
  /-
    🎉 no goals
  -/


@[deprecated mem_image (since := "2024-03-23")]
theorem mem_image_iff_bex {f : α → β} {s : Set α} {y : β} :
    y ∈ f '' s ↔ ∃ (x : _) (_ : x ∈ s), f x = y :=
  bex_def.symm


theorem image_eta (f : α → β) : f '' s = (fun x => f x) '' s :=
  rfl


theorem _root_.Function.Injective.mem_set_image {f : α → β} (hf : Injective f) {s : Set α} {a : α} :
    f a ∈ f '' s ↔ a ∈ s :=
  ⟨fun ⟨_, hb, Eq⟩ => hf Eq ▸ hb, mem_image_of_mem f⟩


lemma preimage_subset_of_surjOn {t : Set β} (hf : Injective f) (h : SurjOn f s t) :
    f ⁻¹' t ⊆ s := fun _ hx ↦
  hf.mem_set_image.1 <| h hx


theorem forall_mem_image {f : α → β} {s : Set α} {p : β → Prop} :
                                                       /-
                                                         α : Type u_1
                                                         β : Type u_2
                                                         f : α → β
                                                         s : Set α
                                                         p : β → Prop
                                                         ⊢ Iff (∀ (y : β), Membership.mem (Set.image f s) y → p y) (∀ ⦃x : α⦄, Membersh …
                                                       -/
    (∀ y ∈ f '' s, p y) ↔ ∀ ⦃x⦄, x ∈ s → p (f x) := by simp
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem exists_mem_image {f : α → β} {s : Set α} {p : β → Prop} :
                                                 /-
                                                   α : Type u_1
                                                   β : Type u_2
                                                   f : α → β
                                                   s : Set α
                                                   p : β → Prop
                                                   ⊢ Iff (Exists fun y => And (Membership.mem (Set.image f s) y) (p y)) (Exists f …
                                                 -/
    (∃ y ∈ f '' s, p y) ↔ ∃ x ∈ s, p (f x) := by simp
                                                 /-
                                                   🎉 no goals
                                                 -/


@[deprecated (since := "2024-02-21")] alias ball_image_iff := forall_mem_image

@[deprecated (since := "2024-02-21")] alias bex_image_iff := exists_mem_image

@[deprecated (since := "2024-02-21")] alias ⟨_, ball_image_of_ball⟩ := forall_mem_image


@[deprecated forall_mem_image (since := "2024-02-21")]
theorem mem_image_elim {f : α → β} {s : Set α} {C : β → Prop} (h : ∀ x : α, x ∈ s → C (f x)) :
    ∀ {y : β}, y ∈ f '' s → C y := forall_mem_image.2 h _


@[deprecated forall_mem_image (since := "2024-02-21")]
theorem mem_image_elim_on {f : α → β} {s : Set α} {C : β → Prop} {y : β} (h_y : y ∈ f '' s)
    (h : ∀ x : α, x ∈ s → C (f x)) : C y := forall_mem_image.2 h _ h_y

-- Porting note: used to be `safe`

@[congr]
theorem image_congr {f g : α → β} {s : Set α} (h : ∀ a ∈ s, f a = g a) : f '' s = g '' s := by
  /-
    α : Type u_1
    β : Type u_2
    f g : α → β
    s : Set α
    h : ∀ (a : α), Membership.mem s a → Eq (f a) (g a)
    ⊢ Eq (Set.image f s) (Set.image g s)
  -/
  ext x
  /-
    case h
    α : Type u_1
    β : Type u_2
    f g : α → β
    s : Set α
    h : ∀ (a : α), Membership.mem s a → Eq (f a) (g a)
    x : β
    ⊢ Iff (Membership.mem (Set.image f s) x) (Membership.mem (Set.image g s) x)
  -/
  exact exists_congr fun a ↦ and_congr_right fun ha ↦ by rw [h a ha]
  /-
    🎉 no goals
  -/


/-- A common special case of `image_congr` -/
theorem image_congr' {f g : α → β} {s : Set α} (h : ∀ x : α, f x = g x) : f '' s = g '' s :=
  image_congr fun x _ => h x


@[gcongr]
lemma image_mono (h : s ⊆ t) : f '' s ⊆ f '' t := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    s t : Set α
    h : HasSubset.Subset s t
    ⊢ HasSubset.Subset (Set.image f s) (Set.image f t)
  -/
  rintro - ⟨a, ha, rfl⟩; exact mem_image_of_mem f (h ha)
                         /-
                           🎉 no goals
                         -/


                                                                                          /-
                                                                                            α : Type u_1
                                                                                            β : Type u_2
                                                                                            γ : Type u_3
                                                                                            f : β → γ
                                                                                            g : α → β
                                                                                            a : Set α
                                                                                            ⊢ Eq (Set.image (Function.comp f g) a) (Set.image f (Set.image g a))
                                                                                          -/
theorem image_comp (f : β → γ) (g : α → β) (a : Set α) : f ∘ g '' a = f '' (g '' a) := by aesop
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


                                                                            /-
                                                                              α : Type u_1
                                                                              β : Type u_2
                                                                              γ : Type u_3
                                                                              f : α → β
                                                                              g : β → γ
                                                                              ⊢ Eq (Set.image (Function.comp g f)) (Function.comp (Set.image g) (Set.image f))
                                                                            -/
theorem image_comp_eq {g : β → γ} : image (g ∘ f) = image g ∘ image f := by ext; simp
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


/-- A variant of `image_comp`, useful for rewriting -/
theorem image_image (g : β → γ) (f : α → β) (s : Set α) : g '' (f '' s) = (fun x => g (f x)) '' s :=
  (image_comp g f s).symm


theorem image_comm {β'} {f : β → γ} {g : α → β} {f' : α → β'} {g' : β' → γ}
    (h_comm : ∀ a, f (g a) = g' (f' a)) : (s.image g).image f = (s.image f').image g' := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    s : Set α
    β' : Type u_5
    f : β → γ
    g : α → β
    f' : α → β'
    g' : β' → γ
    h_comm : ∀ (a : α), Eq (f (g a)) (g' (f' a))
    ⊢ Eq (Set.image f (Set.image g s)) (Set.image g' (Set.image f' s))
  -/
  simp_rw [image_image, h_comm]
  /-
    🎉 no goals
  -/


theorem _root_.Function.Semiconj.set_image {f : α → β} {ga : α → α} {gb : β → β}
    (h : Function.Semiconj f ga gb) : Function.Semiconj (image f) (image ga) (image gb) := fun _ =>
  image_comm h


theorem _root_.Function.Commute.set_image {f g : α → α} (h : Function.Commute f g) :
    Function.Commute (image f) (image g) :=
  Function.Semiconj.set_image h


/-- Image is monotone with respect to `⊆`. See `Set.monotone_image` for the statement in
terms of `≤`. -/
@[gcongr]
theorem image_subset {a b : Set α} (f : α → β) (h : a ⊆ b) : f '' a ⊆ f '' b := by
  /-
    α : Type u_1
    β : Type u_2
    a b : Set α
    f : α → β
    h : HasSubset.Subset a b
    ⊢ HasSubset.Subset (Set.image f a) (Set.image f b)
  -/
  simp only [subset_def, mem_image]
  /-
    α : Type u_1
    β : Type u_2
    a b : Set α
    f : α → β
    h : HasSubset.Subset a b
    ⊢ ∀ (x : β), (Exists fun x_1 => And (Membership.mem a x_1) (Eq (f x_1) x)) → E …
  -/
  exact fun x => fun ⟨w, h1, h2⟩ => ⟨w, h h1, h2⟩
  /-
    🎉 no goals
  -/


/-- `Set.image` is monotone. See `Set.image_subset` for the statement in terms of `⊆`. -/
lemma monotone_image {f : α → β} : Monotone (image f) := fun _ _ => image_subset _


theorem image_union (f : α → β) (s t : Set α) : f '' (s ∪ t) = f '' s ∪ f '' t :=
  ext fun x =>
        /-
          α : Type u_1
          β : Type u_2
          f : α → β
          s t : Set α
          x : β
          ⊢ Membership.mem (Set.image f (Union.union s t)) x → Membership.mem (Union.uni …
        -/
                                                     /-
                                                       🎉 no goals
                                                     -/
    ⟨by rintro ⟨a, h | h, rfl⟩ <;> [left; right] <;> exact ⟨_, h, rfl⟩, by
                                                     /-
                                                       🎉 no goals
                                                     -/
      /-
        α : Type u_1
        β : Type u_2
        f : α → β
        s t : Set α
        x : β
        ⊢ Membership.mem (Union.union (Set.image f s) (Set.image f t)) x → Membership. …
      -/
      rintro (⟨a, h, rfl⟩ | ⟨a, h, rfl⟩) <;> refine ⟨_, ?_, rfl⟩
        /-
          case inl.intro.intro
          α : Type u_1
          β : Type u_2
          f : α → β
          s t : Set α
          a : α
          h : Membership.mem s a
          ⊢ Membership.mem (Union.union s t) a
        -/
      · exact mem_union_left t h
        /-
          🎉 no goals
        -/
        /-
          case inr.intro.intro
          α : Type u_1
          β : Type u_2
          f : α → β
          s t : Set α
          a : α
          h : Membership.mem t a
          ⊢ Membership.mem (Union.union s t) a
        -/
      · exact mem_union_right s h⟩
        /-
          🎉 no goals
        -/


@[simp]
theorem image_empty (f : α → β) : f '' ∅ = ∅ := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    ⊢ Eq (Set.image f EmptyCollection.emptyCollection) EmptyCollection.emptyCollec …
  -/
  ext
  /-
    case h
    α : Type u_1
    β : Type u_2
    f : α → β
    x✝ : β
    ⊢ Iff (Membership.mem (Set.image f EmptyCollection.emptyCollection) x✝) (Membe …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem image_inter_subset (f : α → β) (s t : Set α) : f '' (s ∩ t) ⊆ f '' s ∩ f '' t :=
  subset_inter (image_subset _ inter_subset_left) (image_subset _ inter_subset_right)


theorem image_inter_on {f : α → β} {s t : Set α} (h : ∀ x ∈ t, ∀ y ∈ s, f x = f y → x = y) :
    f '' (s ∩ t) = f '' s ∩ f '' t :=
  (image_inter_subset _ _ _).antisymm
    fun b ⟨⟨a₁, ha₁, h₁⟩, ⟨a₂, ha₂, h₂⟩⟩ ↦
                                          /-
                                            α : Type u_1
                                            β : Type u_2
                                            f : α → β
                                            s t : Set α
                                            h : ∀ (x : α), Membership.mem t x → ∀ (y : α), Membership.mem s y → Eq (f x) ( …
                                            b : β
                                            x✝ : Membership.mem (Inter.inter (Set.image f s) (Set.image f t)) b
                                            a₁ : α
                                            ha₁ : Membership.mem s a₁
                                            h₁ : Eq (f a₁) b
                                            a₂ : α
                                            ha₂ : Membership.mem t a₂
                                            h₂ : Eq (f a₂) b
                                            ⊢ Eq (f a₂) (f a₁)
                                          -/
      have : a₂ = a₁ := h _ ha₂ _ ha₁ (by simp [*])
                                          /-
                                            🎉 no goals
                                          -/
      ⟨a₁, ⟨ha₁, this ▸ ha₂⟩, h₁⟩


theorem image_inter {f : α → β} {s t : Set α} (H : Injective f) : f '' (s ∩ t) = f '' s ∩ f '' t :=
  image_inter_on fun _ _ _ _ h => H h


theorem image_univ_of_surjective {ι : Type*} {f : ι → β} (H : Surjective f) : f '' univ = univ :=
                          /-
                            β : Type u_2
                            ι : Type u_5
                            f : ι → β
                            H : Function.Surjective f
                            ⊢ ∀ (x : β), Membership.mem (Set.image f Set.univ) x
                          -/
  eq_univ_of_forall <| by simpa [image]
                          /-
                            🎉 no goals
                          -/


@[simp]
theorem image_singleton {f : α → β} {a : α} : f '' {a} = {f a} := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    a : α
    ⊢ Eq (Set.image f (Singleton.singleton a)) (Singleton.singleton (f a))
  -/
  ext
  /-
    case h
    α : Type u_1
    β : Type u_2
    f : α → β
    a : α
    x✝ : β
    ⊢ Iff (Membership.mem (Set.image f (Singleton.singleton a)) x✝) (Membership.me …
  -/
  simp [image, eq_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem Nonempty.image_const {s : Set α} (hs : s.Nonempty) (a : β) : (fun _ => a) '' s = {a} :=
  ext fun _ =>
    ⟨fun ⟨_, _, h⟩ => h ▸ mem_singleton _, fun h =>
      (eq_of_mem_singleton h).symm ▸ hs.imp fun _ hy => ⟨hy, rfl⟩⟩


@[simp, mfld_simps]
theorem image_eq_empty {α β} {f : α → β} {s : Set α} : f '' s = ∅ ↔ s = ∅ := by
  /-
    α : Type u_5
    β : Type u_6
    f : α → β
    s : Set α
    ⊢ Iff (Eq (Set.image f s) EmptyCollection.emptyCollection) (Eq s EmptyCollecti …
  -/
  simp only [eq_empty_iff_forall_not_mem]
  /-
    α : Type u_5
    β : Type u_6
    f : α → β
    s : Set α
    ⊢ Iff (∀ (x : β), Not (Membership.mem (Set.image f s) x)) (∀ (x : α), Not (Mem …
  -/
  exact ⟨fun H a ha => H _ ⟨_, ha, rfl⟩, fun H b ⟨_, ha, _⟩ => H _ ha⟩
  /-
    🎉 no goals
  -/

-- Porting note: `compl` is already defined in `Data.Set.Defs`

theorem preimage_compl_eq_image_compl [BooleanAlgebra α] (S : Set α) :
    HasCompl.compl ⁻¹' S = HasCompl.compl '' S :=
  Set.ext fun x =>
    ⟨fun h => ⟨xᶜ, h, compl_compl x⟩, fun h =>
      Exists.elim h fun _ hy => (compl_eq_comm.mp hy.2).symm.subst hy.1⟩


theorem mem_compl_image [BooleanAlgebra α] (t : α) (S : Set α) :
    t ∈ HasCompl.compl '' S ↔ tᶜ ∈ S := by
  /-
    α : Type u_1
    inst✝ : BooleanAlgebra α
    t : α
    S : Set α
    ⊢ Iff (Membership.mem (Set.image HasCompl.compl S) t) (Membership.mem S (HasCo …
  -/
  simp [← preimage_compl_eq_image_compl]
  /-
    🎉 no goals
  -/


@[simp]
                                                    /-
                                                      α : Type u_1
                                                      ⊢ Eq (Set.image id) id
                                                    -/
theorem image_id_eq : image (id : α → α) = id := by ext; simp
                                                         /-
                                                           🎉 no goals
                                                         -/


/-- A variant of `image_id` -/
@[simp]
theorem image_id' (s : Set α) : (fun x => x) '' s = s := by
  /-
    α : Type u_1
    s : Set α
    ⊢ Eq (Set.image (fun x => x) s) s
  -/
  ext
  /-
    case h
    α : Type u_1
    s : Set α
    x✝ : α
    ⊢ Iff (Membership.mem (Set.image (fun x => x) s) x✝) (Membership.mem s x✝)
  -/
  simp
  /-
    🎉 no goals
  -/


                                                 /-
                                                   α : Type u_1
                                                   s : Set α
                                                   ⊢ Eq (Set.image id s) s
                                                 -/
theorem image_id (s : Set α) : id '' s = s := by simp
                                                 /-
                                                   🎉 no goals
                                                 -/


lemma image_iterate_eq {f : α → α} {n : ℕ} : image (f^[n]) = (image f)^[n] := by
  induction n with
  | zero => simp
  | succ n ih => rw [iterate_succ', iterate_succ', ← ih, image_comp_eq]


theorem compl_compl_image [BooleanAlgebra α] (S : Set α) :
    HasCompl.compl '' (HasCompl.compl '' S) = S := by
  /-
    α : Type u_1
    inst✝ : BooleanAlgebra α
    S : Set α
    ⊢ Eq (Set.image HasCompl.compl (Set.image HasCompl.compl S)) S
  -/
  rw [← image_comp, compl_comp_compl, image_id]
  /-
    🎉 no goals
  -/


theorem image_insert_eq {f : α → β} {a : α} {s : Set α} :
    f '' insert a s = insert (f a) (f '' s) := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    a : α
    s : Set α
    ⊢ Eq (Set.image f (Insert.insert a s)) (Insert.insert (f a) (Set.image f s))
  -/
  ext
  /-
    case h
    α : Type u_1
    β : Type u_2
    f : α → β
    a : α
    s : Set α
    x✝ : β
    ⊢ Iff (Membership.mem (Set.image f (Insert.insert a s)) x✝) (Membership.mem (I …
  -/
  simp [and_or_left, exists_or, eq_comm, or_comm, and_comm]
  /-
    🎉 no goals
  -/


theorem image_pair (f : α → β) (a b : α) : f '' {a, b} = {f a, f b} := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    a b : α
    ⊢ Eq (Set.image f (Insert.insert a (Singleton.singleton b))) (Insert.insert (f …
  -/
  simp only [image_insert_eq, image_singleton]
  /-
    🎉 no goals
  -/


theorem image_subset_preimage_of_inverse {f : α → β} {g : β → α} (I : LeftInverse g f) (s : Set α) :
    f '' s ⊆ g ⁻¹' s := fun _ ⟨a, h, e⟩ => e ▸ ((I a).symm ▸ h : g (f a) ∈ s)


theorem preimage_subset_image_of_inverse {f : α → β} {g : β → α} (I : LeftInverse g f) (s : Set β) :
    f ⁻¹' s ⊆ g '' s := fun b h => ⟨f b, h, I b⟩


theorem image_eq_preimage_of_inverse {f : α → β} {g : β → α} (h₁ : LeftInverse g f)
    (h₂ : RightInverse g f) : image f = preimage g :=
  funext fun s =>
    Subset.antisymm (image_subset_preimage_of_inverse h₁ s) (preimage_subset_image_of_inverse h₂ s)


theorem mem_image_iff_of_inverse {f : α → β} {g : β → α} {b : β} {s : Set α} (h₁ : LeftInverse g f)
    (h₂ : RightInverse g f) : b ∈ f '' s ↔ g b ∈ s := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    g : β → α
    b : β
    s : Set α
    h₁ : Function.LeftInverse g f
    h₂ : Function.RightInverse g f
    ⊢ Iff (Membership.mem (Set.image f s) b) (Membership.mem s (g b))
  -/
  rw [image_eq_preimage_of_inverse h₁ h₂]; rfl
                                           /-
                                             🎉 no goals
                                           -/


theorem image_compl_subset {f : α → β} {s : Set α} (H : Injective f) : f '' sᶜ ⊆ (f '' s)ᶜ :=
                                   /-
                                     α : Type u_1
                                     β : Type u_2
                                     f : α → β
                                     s : Set α
                                     H : Function.Injective f
                                     ⊢ Disjoint (Set.image f s) (Set.image f (HasCompl.compl s))
                                   -/
  Disjoint.subset_compl_left <| by simp [disjoint_iff_inf_le, ← image_inter H]
                                   /-
                                     🎉 no goals
                                   -/


theorem subset_image_compl {f : α → β} {s : Set α} (H : Surjective f) : (f '' s)ᶜ ⊆ f '' sᶜ :=
  compl_subset_iff_union.2 <| by
    /-
      α : Type u_1
      β : Type u_2
      f : α → β
      s : Set α
      H : Function.Surjective f
      ⊢ Eq (Union.union (Set.image f s) (Set.image f (HasCompl.compl s))) Set.univ
    -/
    rw [← image_union]
    /-
      α : Type u_1
      β : Type u_2
      f : α → β
      s : Set α
      H : Function.Surjective f
      ⊢ Eq (Set.image f (Union.union s (HasCompl.compl s))) Set.univ
    -/
    simp [image_univ_of_surjective H]
    /-
      🎉 no goals
    -/


theorem image_compl_eq {f : α → β} {s : Set α} (H : Bijective f) : f '' sᶜ = (f '' s)ᶜ :=
  Subset.antisymm (image_compl_subset H.1) (subset_image_compl H.2)


theorem subset_image_diff (f : α → β) (s t : Set α) : f '' s \ f '' t ⊆ f '' (s \ t) := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    s t : Set α
    ⊢ HasSubset.Subset (SDiff.sdiff (Set.image f s) (Set.image f t)) (Set.image f  …
  -/
  rw [diff_subset_iff, ← image_union, union_diff_self]
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    s t : Set α
    ⊢ HasSubset.Subset (Set.image f s) (Set.image f (Union.union t s))
  -/
  exact image_subset f subset_union_right
  /-
    🎉 no goals
  -/


open scoped symmDiff in
theorem subset_image_symmDiff : (f '' s) ∆ (f '' t) ⊆ f '' s ∆ t :=
  (union_subset_union (subset_image_diff _ _ _) <| subset_image_diff _ _ _).trans
    (superset_of_eq (image_union _ _ _))


theorem image_diff {f : α → β} (hf : Injective f) (s t : Set α) : f '' (s \ t) = f '' s \ f '' t :=
  Subset.antisymm
    (Subset.trans (image_inter_subset _ _ _) <| inter_subset_inter_right _ <| image_compl_subset hf)
    (subset_image_diff f s t)


open scoped symmDiff in
theorem image_symmDiff (hf : Injective f) (s t : Set α) : f '' s ∆ t = (f '' s) ∆ (f '' t) := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    hf : Function.Injective f
    s t : Set α
    ⊢ Eq (Set.image f (symmDiff s t)) (symmDiff (Set.image f s) (Set.image f t))
  -/
  simp_rw [Set.symmDiff_def, image_union, image_diff hf]
  /-
    🎉 no goals
  -/


theorem Nonempty.image (f : α → β) {s : Set α} : s.Nonempty → (f '' s).Nonempty
  | ⟨x, hx⟩ => ⟨f x, mem_image_of_mem f hx⟩


theorem Nonempty.of_image {f : α → β} {s : Set α} : (f '' s).Nonempty → s.Nonempty
  | ⟨_, x, hx, _⟩ => ⟨x, hx⟩


@[simp]
theorem image_nonempty {f : α → β} {s : Set α} : (f '' s).Nonempty ↔ s.Nonempty :=
  ⟨Nonempty.of_image, fun h => h.image f⟩


theorem Nonempty.preimage {s : Set β} (hs : s.Nonempty) {f : α → β} (hf : Surjective f) :
    (f ⁻¹' s).Nonempty :=
  let ⟨y, hy⟩ := hs
  let ⟨x, hx⟩ := hf y
  ⟨x, mem_preimage.2 <| hx.symm ▸ hy⟩


instance (f : α → β) (s : Set α) [Nonempty s] : Nonempty (f '' s) :=
  (Set.Nonempty.image f .of_subtype).to_subtype


/-- image and preimage are a Galois connection -/
@[simp]
theorem image_subset_iff {s : Set α} {t : Set β} {f : α → β} : f '' s ⊆ t ↔ s ⊆ f ⁻¹' t :=
  forall_mem_image


theorem image_preimage_subset (f : α → β) (s : Set β) : f '' (f ⁻¹' s) ⊆ s :=
  image_subset_iff.2 Subset.rfl


theorem subset_preimage_image (f : α → β) (s : Set α) : s ⊆ f ⁻¹' (f '' s) := fun _ =>
  mem_image_of_mem f


theorem preimage_image_univ {f : α → β} : f ⁻¹' (f '' univ) = univ :=
  Subset.antisymm (fun _ _ => trivial) (subset_preimage_image f univ)


@[simp]
theorem preimage_image_eq {f : α → β} (s : Set α) (h : Injective f) : f ⁻¹' (f '' s) = s :=
  Subset.antisymm (fun _ ⟨_, hy, e⟩ => h e ▸ hy) (subset_preimage_image f s)


@[simp]
theorem image_preimage_eq {f : α → β} (s : Set β) (h : Surjective f) : f '' (f ⁻¹' s) = s :=
  Subset.antisymm (image_preimage_subset f s) fun x hx =>
    let ⟨y, e⟩ := h x
    ⟨y, (e.symm ▸ hx : f y ∈ s), e⟩


@[simp]
theorem Nonempty.subset_preimage_const {s : Set α} (hs : Set.Nonempty s) (t : Set β) (a : β) :
    s ⊆ (fun _ => a) ⁻¹' t ↔ a ∈ t := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    hs : s.Nonempty
    t : Set β
    a : β
    ⊢ Iff (HasSubset.Subset s (Set.preimage (fun x => a) t)) (Membership.mem t a)
  -/
  rw [← image_subset_iff, hs.image_const, singleton_subset_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_eq_preimage {f : β → α} (hf : Surjective f) : f ⁻¹' s = f ⁻¹' t ↔ s = t :=
  Iff.intro
                 /-
                   α : Type u_1
                   β : Type u_2
                   s t : Set α
                   f : β → α
                   hf : Function.Surjective f
                   eq : Eq (Set.preimage f s) (Set.preimage f t)
                   ⊢ Eq s t
                 -/
    fun eq => by rw [← image_preimage_eq s hf, ← image_preimage_eq t hf, eq]
                 /-
                   🎉 no goals
                 -/
    fun eq => eq ▸ rfl


theorem image_inter_preimage (f : α → β) (s : Set α) (t : Set β) :
    f '' (s ∩ f ⁻¹' t) = f '' s ∩ t := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    s : Set α
    t : Set β
    ⊢ Eq (Set.image f (Inter.inter s (Set.preimage f t))) (Inter.inter (Set.image  …
  -/
  apply Subset.antisymm
  · calc
      f '' (s ∩ f ⁻¹' t) ⊆ f '' s ∩ f '' (f ⁻¹' t) := image_inter_subset _ _ _
      _ ⊆ f '' s ∩ t := inter_subset_inter_right _ (image_preimage_subset f t)
    /-
      case h₂
      α : Type u_1
      β : Type u_2
      f : α → β
      s : Set α
      t : Set β
      ⊢ HasSubset.Subset (Inter.inter (Set.image f s) t) (Set.image f (Inter.inter s …
    -/
  · rintro _ ⟨⟨x, h', rfl⟩, h⟩
    /-
      case h₂.intro.intro.intro
      α : Type u_1
      β : Type u_2
      f : α → β
      s : Set α
      t : Set β
      x : α
      h' : Membership.mem s x
      h : Membership.mem t (f x)
      ⊢ Membership.mem (Set.image f (Inter.inter s (Set.preimage f t))) (f x)
    -/
    exact ⟨x, ⟨h', h⟩, rfl⟩
    /-
      🎉 no goals
    -/


theorem image_preimage_inter (f : α → β) (s : Set α) (t : Set β) :
                                          /-
                                            α : Type u_1
                                            β : Type u_2
                                            f : α → β
                                            s : Set α
                                            t : Set β
                                            ⊢ Eq (Set.image f (Inter.inter (Set.preimage f t) s)) (Inter.inter t (Set.imag …
                                          -/
    f '' (f ⁻¹' t ∩ s) = t ∩ f '' s := by simp only [inter_comm, image_inter_preimage]
                                          /-
                                            🎉 no goals
                                          -/


@[simp]
theorem image_inter_nonempty_iff {f : α → β} {s : Set α} {t : Set β} :
    (f '' s ∩ t).Nonempty ↔ (s ∩ f ⁻¹' t).Nonempty := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    s : Set α
    t : Set β
    ⊢ Iff (Inter.inter (Set.image f s) t).Nonempty (Inter.inter s (Set.preimage f  …
  -/
  rw [← image_inter_preimage, image_nonempty]
  /-
    🎉 no goals
  -/


theorem image_diff_preimage {f : α → β} {s : Set α} {t : Set β} :
                                          /-
                                            α : Type u_1
                                            β : Type u_2
                                            f : α → β
                                            s : Set α
                                            t : Set β
                                            ⊢ Eq (Set.image f (SDiff.sdiff s (Set.preimage f t))) (SDiff.sdiff (Set.image  …
                                          -/
    f '' (s \ f ⁻¹' t) = f '' s \ t := by simp_rw [diff_eq, ← preimage_compl, image_inter_preimage]
                                          /-
                                            🎉 no goals
                                          -/


theorem compl_image : image (compl : Set α → Set α) = preimage compl :=
  image_eq_preimage_of_inverse compl_compl compl_compl


theorem compl_image_set_of {p : Set α → Prop} : compl '' { s | p s } = { s | p sᶜ } :=
  congr_fun compl_image p


theorem inter_preimage_subset (s : Set α) (t : Set β) (f : α → β) :
    s ∩ f ⁻¹' t ⊆ f ⁻¹' (f '' s ∩ t) := fun _ h => ⟨mem_image_of_mem _ h.left, h.right⟩


theorem union_preimage_subset (s : Set α) (t : Set β) (f : α → β) :
    s ∪ f ⁻¹' t ⊆ f ⁻¹' (f '' s ∪ t) := fun _ h =>
  Or.elim h (fun l => Or.inl <| mem_image_of_mem _ l) fun r => Or.inr r


theorem subset_image_union (f : α → β) (s : Set α) (t : Set β) : f '' (s ∪ f ⁻¹' t) ⊆ f '' s ∪ t :=
  image_subset_iff.2 (union_preimage_subset _ _ _)


theorem preimage_subset_iff {A : Set α} {B : Set β} {f : α → β} :
    f ⁻¹' B ⊆ A ↔ ∀ a : α, f a ∈ B → a ∈ A :=
  Iff.rfl


theorem image_eq_image {f : α → β} (hf : Injective f) : f '' s = f '' t ↔ s = t :=
  Iff.symm <|
    (Iff.intro fun eq => eq ▸ rfl) fun eq => by
      /-
        α : Type u_1
        β : Type u_2
        s t : Set α
        f : α → β
        hf : Function.Injective f
        eq : Eq (Set.image f s) (Set.image f t)
        ⊢ Eq s t
      -/
      rw [← preimage_image_eq s hf, ← preimage_image_eq t hf, eq]
      /-
        🎉 no goals
      -/


theorem subset_image_iff {t : Set β} :
    t ⊆ f '' s ↔ ∃ u, u ⊆ s ∧ f '' u = t := by
  refine ⟨fun h ↦ ⟨f ⁻¹' t ∩ s, inter_subset_right, ?_⟩,
    fun ⟨u, hu, hu'⟩ ↦ hu'.symm ▸ image_mono hu⟩
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    s : Set α
    t : Set β
    h : HasSubset.Subset t (Set.image f s)
    ⊢ Eq (Set.image f (Inter.inter (Set.preimage f t) s)) t
  -/
  rwa [image_preimage_inter, inter_eq_left]
  /-
    🎉 no goals
  -/


@[simp]
lemma exists_subset_image_iff {p : Set β → Prop} : (∃ t ⊆ f '' s, p t) ↔ ∃ t ⊆ s, p (f '' t) := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    s : Set α
    p : Set β → Prop
    ⊢ Iff (Exists fun t => And (HasSubset.Subset t (Set.image f s)) (p t)) (Exists …
  -/
  simp [subset_image_iff]
  /-
    🎉 no goals
  -/


@[simp]
lemma forall_subset_image_iff {p : Set β → Prop} : (∀ t ⊆ f '' s, p t) ↔ ∀ t ⊆ s, p (f '' t) := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    s : Set α
    p : Set β → Prop
    ⊢ Iff (∀ (t : Set β), HasSubset.Subset t (Set.image f s) → p t) (∀ (t : Set α) …
  -/
  simp [subset_image_iff]
  /-
    🎉 no goals
  -/


theorem image_subset_image_iff {f : α → β} (hf : Injective f) : f '' s ⊆ f '' t ↔ s ⊆ t := by
  /-
    α : Type u_1
    β : Type u_2
    s t : Set α
    f : α → β
    hf : Function.Injective f
    ⊢ Iff (HasSubset.Subset (Set.image f s) (Set.image f t)) (HasSubset.Subset s t)
  -/
  refine Iff.symm <| (Iff.intro (image_subset f)) fun h => ?_
  /-
    α : Type u_1
    β : Type u_2
    s t : Set α
    f : α → β
    hf : Function.Injective f
    h : HasSubset.Subset (Set.image f s) (Set.image f t)
    ⊢ HasSubset.Subset s t
  -/
  rw [← preimage_image_eq s hf, ← preimage_image_eq t hf]
  /-
    α : Type u_1
    β : Type u_2
    s t : Set α
    f : α → β
    hf : Function.Injective f
    h : HasSubset.Subset (Set.image f s) (Set.image f t)
    ⊢ HasSubset.Subset (Set.preimage f (Set.image f s)) (Set.preimage f (Set.image …
  -/
  exact preimage_mono h
  /-
    🎉 no goals
  -/


theorem prod_quotient_preimage_eq_image [s : Setoid α] (g : Quotient s → β) {h : α → β}
    (Hh : h = g ∘ Quotient.mk'') (r : Set (β × β)) :
    { x : Quotient s × Quotient s | (g x.1, g x.2) ∈ r } =
      (fun a : α × α => (⟦a.1⟧, ⟦a.2⟧)) '' ((fun a : α × α => (h a.1, h a.2)) ⁻¹' r) :=
  Hh.symm ▸
    Set.ext fun ⟨a₁, a₂⟩ =>
      ⟨Quot.induction_on₂ a₁ a₂ fun a₁ a₂ h => ⟨(a₁, a₂), h, rfl⟩, fun ⟨⟨b₁, b₂⟩, h₁, h₂⟩ =>
        show (g a₁, g a₂) ∈ r from
          have h₃ : ⟦b₁⟧ = a₁ ∧ ⟦b₂⟧ = a₂ := Prod.ext_iff.1 h₂
          h₃.1 ▸ h₃.2 ▸ h₁⟩


theorem exists_image_iff (f : α → β) (x : Set α) (P : β → Prop) :
    (∃ a : f '' x, P a) ↔ ∃ a : x, P (f a) :=
  ⟨fun ⟨a, h⟩ => ⟨⟨_, a.prop.choose_spec.1⟩, a.prop.choose_spec.2.symm ▸ h⟩, fun ⟨a, h⟩ =>
    ⟨⟨_, _, a.prop, rfl⟩, h⟩⟩


theorem imageFactorization_eq {f : α → β} {s : Set α} :
    Subtype.val ∘ imageFactorization f s = f ∘ Subtype.val :=
  funext fun _ => rfl


theorem surjective_onto_image {f : α → β} {s : Set α} : Surjective (imageFactorization f s) :=
  fun ⟨_, ⟨a, ha, rfl⟩⟩ => ⟨⟨a, ha⟩, rfl⟩


/-- If the only elements outside `s` are those left fixed by `σ`, then mapping by `σ` has no effect.
-/
theorem image_perm {s : Set α} {σ : Equiv.Perm α} (hs : { a : α | σ a ≠ a } ⊆ s) : σ '' s = s := by
  /-
    α : Type u_1
    s : Set α
    σ : Equiv.Perm α
    hs : HasSubset.Subset (setOf fun a => Ne (σ a) a) s
    ⊢ Eq (Set.image (⇑σ) s) s
  -/
  ext i
  /-
    case h
    α : Type u_1
    s : Set α
    σ : Equiv.Perm α
    hs : HasSubset.Subset (setOf fun a => Ne (σ a) a) s
    i : α
    ⊢ Iff (Membership.mem (Set.image (⇑σ) s) i) (Membership.mem s i)
  -/
  obtain hi | hi := eq_or_ne (σ i) i
    /-
      case h.inl
      α : Type u_1
      s : Set α
      σ : Equiv.Perm α
      hs : HasSubset.Subset (setOf fun a => Ne (σ a) a) s
      i : α
      hi : Eq (σ i) i
      ⊢ Iff (Membership.mem (Set.image (⇑σ) s) i) (Membership.mem s i)
    -/
  · refine ⟨?_, fun h => ⟨i, h, hi⟩⟩
    /-
      case h.inl
      α : Type u_1
      s : Set α
      σ : Equiv.Perm α
      hs : HasSubset.Subset (setOf fun a => Ne (σ a) a) s
      i : α
      hi : Eq (σ i) i
      ⊢ Membership.mem (Set.image (⇑σ) s) i → Membership.mem s i
    -/
    rintro ⟨j, hj, h⟩
    /-
      case h.inl.intro.intro
      α : Type u_1
      s : Set α
      σ : Equiv.Perm α
      hs : HasSubset.Subset (setOf fun a => Ne (σ a) a) s
      i : α
      hi : Eq (σ i) i
      j : α
      hj : Membership.mem s j
      h : Eq (σ j) i
      ⊢ Membership.mem s i
    -/
    rwa [σ.injective (hi.trans h.symm)]
    /-
      🎉 no goals
    -/
    /-
      case h.inr
      α : Type u_1
      s : Set α
      σ : Equiv.Perm α
      hs : HasSubset.Subset (setOf fun a => Ne (σ a) a) s
      i : α
      hi : Ne (σ i) i
      ⊢ Iff (Membership.mem (Set.image (⇑σ) s) i) (Membership.mem s i)
    -/
  · refine iff_of_true ⟨σ.symm i, hs fun h => hi ?_, σ.apply_symm_apply _⟩ (hs hi)
    /-
      case h.inr
      α : Type u_1
      s : Set α
      σ : Equiv.Perm α
      hs : HasSubset.Subset (setOf fun a => Ne (σ a) a) s
      i : α
      hi : Ne (σ i) i
      h : Eq (σ ((Equiv.symm σ) i)) ((Equiv.symm σ) i)
      ⊢ Eq (σ i) i
    -/
                              /-
                                🎉 no goals
                              -/
    convert congr_arg σ h <;> exact (σ.apply_symm_apply _).symm
                              /-
                                🎉 no goals
                              -/


/-- The powerset of `{a} ∪ s` is `𝒫 s` together with `{a} ∪ t` for each `t ∈ 𝒫 s`. -/
theorem powerset_insert (s : Set α) (a : α) : 𝒫 insert a s = 𝒫 s ∪ insert a '' 𝒫 s := by
  /-
    α : Type u_1
    s : Set α
    a : α
    ⊢ Eq (Insert.insert a s).powerset (Union.union s.powerset (Set.image (Insert.i …
  -/
  ext t
  /-
    case h
    α : Type u_1
    s : Set α
    a : α
    t : Set α
    ⊢ Iff (Membership.mem (Insert.insert a s).powerset t) (Membership.mem (Union.u …
  -/
  simp_rw [mem_union, mem_image, mem_powerset_iff]
  /-
    case h
    α : Type u_1
    s : Set α
    a : α
    t : Set α
    ⊢ Iff (HasSubset.Subset t (Insert.insert a s)) (Or (HasSubset.Subset t s) (Exi …
  -/
  constructor
    /-
      case h.mp
      α : Type u_1
      s : Set α
      a : α
      t : Set α
      ⊢ HasSubset.Subset t (Insert.insert a s) → Or (HasSubset.Subset t s) (Exists f …
    -/
  · intro h
    /-
      case h.mp
      α : Type u_1
      s : Set α
      a : α
      t : Set α
      h : HasSubset.Subset t (Insert.insert a s)
      ⊢ Or (HasSubset.Subset t s) (Exists fun x => And (HasSubset.Subset x s) (Eq (I …
    -/
    by_cases hs : a ∈ t
      /-
        case pos
        α : Type u_1
        s : Set α
        a : α
        t : Set α
        h : HasSubset.Subset t (Insert.insert a s)
        hs : Membership.mem t a
        ⊢ Or (HasSubset.Subset t s) (Exists fun x => And (HasSubset.Subset x s) (Eq (I …
      -/
    · right
      /-
        case pos.h
        α : Type u_1
        s : Set α
        a : α
        t : Set α
        h : HasSubset.Subset t (Insert.insert a s)
        hs : Membership.mem t a
        ⊢ Exists fun x => And (HasSubset.Subset x s) (Eq (Insert.insert a x) t)
      -/
      refine ⟨t \ {a}, ?_, ?_⟩
        /-
          case pos.h.refine_1
          α : Type u_1
          s : Set α
          a : α
          t : Set α
          h : HasSubset.Subset t (Insert.insert a s)
          hs : Membership.mem t a
          ⊢ HasSubset.Subset (SDiff.sdiff t (Singleton.singleton a)) s
        -/
      · rw [diff_singleton_subset_iff]
        /-
          case pos.h.refine_1
          α : Type u_1
          s : Set α
          a : α
          t : Set α
          h : HasSubset.Subset t (Insert.insert a s)
          hs : Membership.mem t a
          ⊢ HasSubset.Subset t (Insert.insert a s)
        -/
        assumption
        /-
          🎉 no goals
        -/
        /-
          case pos.h.refine_2
          α : Type u_1
          s : Set α
          a : α
          t : Set α
          h : HasSubset.Subset t (Insert.insert a s)
          hs : Membership.mem t a
          ⊢ Eq (Insert.insert a (SDiff.sdiff t (Singleton.singleton a))) t
        -/
      · rw [insert_diff_singleton, insert_eq_of_mem hs]
        /-
          🎉 no goals
        -/
      /-
        case neg
        α : Type u_1
        s : Set α
        a : α
        t : Set α
        h : HasSubset.Subset t (Insert.insert a s)
        hs : Not (Membership.mem t a)
        ⊢ Or (HasSubset.Subset t s) (Exists fun x => And (HasSubset.Subset x s) (Eq (I …
      -/
    · left
      /-
        case neg.h
        α : Type u_1
        s : Set α
        a : α
        t : Set α
        h : HasSubset.Subset t (Insert.insert a s)
        hs : Not (Membership.mem t a)
        ⊢ HasSubset.Subset t s
      -/
      exact (subset_insert_iff_of_not_mem hs).mp h
      /-
        🎉 no goals
      -/
    /-
      case h.mpr
      α : Type u_1
      s : Set α
      a : α
      t : Set α
      ⊢ Or (HasSubset.Subset t s) (Exists fun x => And (HasSubset.Subset x s) (Eq (I …
    -/
  · rintro (h | ⟨s', h₁, rfl⟩)
      /-
        case h.mpr.inl
        α : Type u_1
        s : Set α
        a : α
        t : Set α
        h : HasSubset.Subset t s
        ⊢ HasSubset.Subset t (Insert.insert a s)
      -/
    · exact subset_trans h (subset_insert a s)
      /-
        🎉 no goals
      -/
      /-
        case h.mpr.inr.intro.intro
        α : Type u_1
        s : Set α
        a : α
        s' : Set α
        h₁ : HasSubset.Subset s' s
        ⊢ HasSubset.Subset (Insert.insert a s') (Insert.insert a s)
      -/
    · exact insert_subset_insert h₁
      /-
        🎉 no goals
      -/


                                                                                    /-
                                                                                      α : Type u_1
                                                                                      ι : Sort u_4
                                                                                      f : ι → α
                                                                                      p : α → Prop
                                                                                      ⊢ Iff (∀ (a : α), Membership.mem (Set.range f) a → p a) (∀ (i : ι), p (f i))
                                                                                    -/
theorem forall_mem_range {p : α → Prop} : (∀ a ∈ range f, p a) ↔ ∀ i, p (f i) := by simp
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


@[deprecated (since := "2024-02-21")] alias forall_range_iff := forall_mem_range


theorem forall_subtype_range_iff {p : range f → Prop} :
    (∀ a : range f, p a) ↔ ∀ i, p ⟨f i, mem_range_self _⟩ :=
  ⟨fun H _ => H _, fun H ⟨y, i, hi⟩ => by
    /-
      α : Type u_1
      ι : Sort u_4
      f : ι → α
      p : ↑(Set.range f) → Prop
      H : ∀ (i : ι), p ⟨f i, ⋯⟩
      x✝ : ↑(Set.range f)
      y : α
      i : ι
      hi : Eq (f i) y
      ⊢ p ⟨y, ⋯⟩
    -/
    subst hi
    /-
      α : Type u_1
      ι : Sort u_4
      f : ι → α
      p : ↑(Set.range f) → Prop
      H : ∀ (i : ι), p ⟨f i, ⋯⟩
      x✝ : ↑(Set.range f)
      i : ι
      ⊢ p ⟨f i, ⋯⟩
    -/
    apply H⟩
    /-
      🎉 no goals
    -/


                                                                                    /-
                                                                                      α : Type u_1
                                                                                      ι : Sort u_4
                                                                                      f : ι → α
                                                                                      p : α → Prop
                                                                                      ⊢ Iff (Exists fun a => And (Membership.mem (Set.range f) a) (p a)) (Exists fun …
                                                                                    -/
theorem exists_range_iff {p : α → Prop} : (∃ a ∈ range f, p a) ↔ ∃ i, p (f i) := by simp
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


@[deprecated (since := "2024-03-10")]
alias exists_range_iff' := exists_range_iff


theorem exists_subtype_range_iff {p : range f → Prop} :
    (∃ a : range f, p a) ↔ ∃ i, p ⟨f i, mem_range_self _⟩ :=
  ⟨fun ⟨⟨a, i, hi⟩, ha⟩ => by
    /-
      α : Type u_1
      ι : Sort u_4
      f : ι → α
      p : ↑(Set.range f) → Prop
      x✝ : Exists fun a => p a
      a : α
      i : ι
      hi : Eq (f i) a
      ha : p ⟨a, ⋯⟩
      ⊢ Exists fun i => p ⟨f i, ⋯⟩
    -/
    subst a
    /-
      α : Type u_1
      ι : Sort u_4
      f : ι → α
      p : ↑(Set.range f) → Prop
      x✝ : Exists fun a => p a
      i : ι
      ha : p ⟨f i, ⋯⟩
      ⊢ Exists fun i => p ⟨f i, ⋯⟩
    -/
    exact ⟨i, ha⟩,
    /-
      🎉 no goals
    -/
   fun ⟨_, hi⟩ => ⟨_, hi⟩⟩


theorem range_eq_univ : range f = univ ↔ Surjective f :=
  eq_univ_iff_forall


@[deprecated (since := "2024-11-11")] alias range_iff_surjective := range_eq_univ


alias ⟨_, _root_.Function.Surjective.range_eq⟩ := range_eq_univ


@[simp]
theorem subset_range_of_surjective {f : α → β} (h : Surjective f) (s : Set β) :
    s ⊆ range f := Surjective.range_eq h ▸ subset_univ s


@[simp]
theorem image_univ {f : α → β} : f '' univ = range f := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    ⊢ Eq (Set.image f Set.univ) (Set.range f)
  -/
  ext
  /-
    case h
    α : Type u_1
    β : Type u_2
    f : α → β
    x✝ : β
    ⊢ Iff (Membership.mem (Set.image f Set.univ) x✝) (Membership.mem (Set.range f) …
  -/
  simp [image, range]
  /-
    🎉 no goals
  -/


lemma image_compl_eq_range_diff_image {f : α → β} (hf : Injective f) (s : Set α) :
                                     /-
                                       α : Type u_1
                                       β : Type u_2
                                       f : α → β
                                       hf : Function.Injective f
                                       s : Set α
                                       ⊢ Eq (Set.image f (HasCompl.compl s)) (SDiff.sdiff (Set.range f) (Set.image f  …
                                     -/
    f '' sᶜ = range f \ f '' s := by rw [← image_univ, ← image_diff hf, compl_eq_univ_diff]
                                     /-
                                       🎉 no goals
                                     -/


/-- Alias of `Set.image_compl_eq_range_sdiff_image`. -/
lemma range_diff_image {f : α → β} (hf : Injective f) (s : Set α) : range f \ f '' s = f '' sᶜ := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    hf : Function.Injective f
    s : Set α
    ⊢ Eq (SDiff.sdiff (Set.range f) (Set.image f s)) (Set.image f (HasCompl.compl  …
  -/
  rw [image_compl_eq_range_diff_image hf]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_eq_univ_iff {f : α → β} {s} : f ⁻¹' s = univ ↔ range f ⊆ s := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    s : Set β
    ⊢ Iff (Eq (Set.preimage f s) Set.univ) (HasSubset.Subset (Set.range f) s)
  -/
  rw [← univ_subset_iff, ← image_subset_iff, image_univ]
  /-
    🎉 no goals
  -/


theorem image_subset_range (f : α → β) (s) : f '' s ⊆ range f := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    s : Set α
    ⊢ HasSubset.Subset (Set.image f s) (Set.range f)
  -/
  rw [← image_univ]; exact image_subset _ (subset_univ _)
                     /-
                       🎉 no goals
                     -/


theorem mem_range_of_mem_image (f : α → β) (s) {x : β} (h : x ∈ f '' s) : x ∈ range f :=
  image_subset_range f s h


theorem _root_.Nat.mem_range_succ (i : ℕ) : i ∈ range Nat.succ ↔ 0 < i :=
  ⟨by
    /-
      i : Nat
      ⊢ Membership.mem (Set.range Nat.succ) i → LT.lt 0 i
    -/
    rintro ⟨n, rfl⟩
    /-
      case intro
      n : Nat
      ⊢ LT.lt 0 n.succ
    -/
    exact Nat.succ_pos n, fun h => ⟨_, Nat.succ_pred_eq_of_pos h⟩⟩
    /-
      🎉 no goals
    -/


theorem Nonempty.preimage' {s : Set β} (hs : s.Nonempty) {f : α → β} (hf : s ⊆ range f) :
    (f ⁻¹' s).Nonempty :=
  let ⟨_, hy⟩ := hs
  let ⟨x, hx⟩ := hf hy
  ⟨x, Set.mem_preimage.2 <| hx.symm ▸ hy⟩


                                                                                /-
                                                                                  α : Type u_1
                                                                                  β : Type u_2
                                                                                  ι : Sort u_4
                                                                                  g : α → β
                                                                                  f : ι → α
                                                                                  ⊢ Eq (Set.range (Function.comp g f)) (Set.image g (Set.range f))
                                                                                -/
theorem range_comp (g : α → β) (f : ι → α) : range (g ∘ f) = g '' range f := by aesop
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


theorem range_subset_iff : range f ⊆ s ↔ ∀ y, f y ∈ s :=
  forall_mem_range


theorem range_subset_range_iff_exists_comp {f : α → γ} {g : β → γ} :
    range f ⊆ range g ↔ ∃ h : α → β, f = g ∘ h := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → γ
    g : β → γ
    ⊢ Iff (HasSubset.Subset (Set.range f) (Set.range g)) (Exists fun h => Eq f (Fu …
  -/
  simp only [range_subset_iff, mem_range, Classical.skolem, funext_iff, (· ∘ ·), eq_comm]
  /-
    🎉 no goals
  -/


theorem range_eq_iff (f : α → β) (s : Set β) :
    range f = s ↔ (∀ a, f a ∈ s) ∧ ∀ b ∈ s, ∃ a, f a = b := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    s : Set β
    ⊢ Iff (Eq (Set.range f) s) (And (∀ (a : α), Membership.mem s (f a)) (∀ (b : β) …
  -/
  rw [← range_subset_iff]
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    s : Set β
    ⊢ Iff (Eq (Set.range f) s) (And (HasSubset.Subset (Set.range f) s) (∀ (b : β), …
  -/
  exact le_antisymm_iff
  /-
    🎉 no goals
  -/


theorem range_comp_subset_range (f : α → β) (g : β → γ) : range (g ∘ f) ⊆ range g := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → β
    g : β → γ
    ⊢ HasSubset.Subset (Set.range (Function.comp g f)) (Set.range g)
  -/
  rw [range_comp]; apply image_subset_range
                   /-
                     🎉 no goals
                   -/


theorem range_nonempty_iff_nonempty : (range f).Nonempty ↔ Nonempty ι :=
  ⟨fun ⟨_, x, _⟩ => ⟨x⟩, fun ⟨x⟩ => ⟨f x, mem_range_self x⟩⟩


theorem range_nonempty [h : Nonempty ι] (f : ι → α) : (range f).Nonempty :=
  range_nonempty_iff_nonempty.2 h


@[simp]
theorem range_eq_empty_iff {f : ι → α} : range f = ∅ ↔ IsEmpty ι := by
  /-
    α : Type u_1
    ι : Sort u_4
    f : ι → α
    ⊢ Iff (Eq (Set.range f) EmptyCollection.emptyCollection) (IsEmpty ι)
  -/
  rw [← not_nonempty_iff, ← range_nonempty_iff_nonempty, not_nonempty_iff_eq_empty]
  /-
    🎉 no goals
  -/


theorem range_eq_empty [IsEmpty ι] (f : ι → α) : range f = ∅ :=
  range_eq_empty_iff.2 ‹_›


instance instNonemptyRange [Nonempty ι] (f : ι → α) : Nonempty (range f) :=
  (range_nonempty f).to_subtype


@[simp]
theorem image_union_image_compl_eq_range (f : α → β) : f '' s ∪ f '' sᶜ = range f := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    f : α → β
    ⊢ Eq (Union.union (Set.image f s) (Set.image f (HasCompl.compl s))) (Set.range …
  -/
  rw [← image_union, ← image_univ, ← union_compl_self]
  /-
    🎉 no goals
  -/


theorem insert_image_compl_eq_range (f : α → β) (x : α) : insert (f x) (f '' {x}ᶜ) = range f := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    x : α
    ⊢ Eq (Insert.insert (f x) (Set.image f (HasCompl.compl (Singleton.singleton x) …
  -/
  rw [← image_insert_eq, insert_eq, union_compl_self, image_univ]
  /-
    🎉 no goals
  -/


theorem image_preimage_eq_range_inter {f : α → β} {t : Set β} : f '' (f ⁻¹' t) = range f ∩ t :=
  ext fun x =>
    ⟨fun ⟨_, hx, HEq⟩ => HEq ▸ ⟨mem_range_self _, hx⟩, fun ⟨⟨y, h_eq⟩, hx⟩ =>
                                                       /-
                                                         α : Type u_1
                                                         β : Type u_2
                                                         f : α → β
                                                         t : Set β
                                                         x : β
                                                         x✝ : Membership.mem (Inter.inter (Set.range f) t) x
                                                         y : α
                                                         h_eq : Eq (f y) x
                                                         hx : Membership.mem t x
                                                         ⊢ Membership.mem (Set.preimage f t) y
                                                       -/
      h_eq ▸ mem_image_of_mem f <| show y ∈ f ⁻¹' t by rw [preimage, mem_setOf, h_eq]; exact hx⟩
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/


theorem image_preimage_eq_inter_range {f : α → β} {t : Set β} : f '' (f ⁻¹' t) = t ∩ range f := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    t : Set β
    ⊢ Eq (Set.image f (Set.preimage f t)) (Inter.inter t (Set.range f))
  -/
  rw [image_preimage_eq_range_inter, inter_comm]
  /-
    🎉 no goals
  -/


theorem image_preimage_eq_of_subset {f : α → β} {s : Set β} (hs : s ⊆ range f) :
                             /-
                               α : Type u_1
                               β : Type u_2
                               f : α → β
                               s : Set β
                               hs : HasSubset.Subset s (Set.range f)
                               ⊢ Eq (Set.image f (Set.preimage f s)) s
                             -/
    f '' (f ⁻¹' s) = s := by rw [image_preimage_eq_range_inter, inter_eq_self_of_subset_right hs]
                             /-
                               🎉 no goals
                             -/


theorem image_preimage_eq_iff {f : α → β} {s : Set β} : f '' (f ⁻¹' s) = s ↔ s ⊆ range f :=
  ⟨by
    /-
      α : Type u_1
      β : Type u_2
      f : α → β
      s : Set β
      ⊢ Eq (Set.image f (Set.preimage f s)) s → HasSubset.Subset s (Set.range f)
    -/
    intro h
    /-
      α : Type u_1
      β : Type u_2
      f : α → β
      s : Set β
      h : Eq (Set.image f (Set.preimage f s)) s
      ⊢ HasSubset.Subset s (Set.range f)
    -/
    rw [← h]
    /-
      α : Type u_1
      β : Type u_2
      f : α → β
      s : Set β
      h : Eq (Set.image f (Set.preimage f s)) s
      ⊢ HasSubset.Subset (Set.image f (Set.preimage f s)) (Set.range f)
    -/
    apply image_subset_range,
    /-
      🎉 no goals
    -/
   image_preimage_eq_of_subset⟩


theorem subset_range_iff_exists_image_eq {f : α → β} {s : Set β} : s ⊆ range f ↔ ∃ t, f '' t = s :=
  ⟨fun h => ⟨_, image_preimage_eq_iff.2 h⟩, fun ⟨_, ht⟩ => ht ▸ image_subset_range _ _⟩


theorem range_image (f : α → β) : range (image f) = 𝒫 range f :=
  ext fun _ => subset_range_iff_exists_image_eq.symm


@[simp]
theorem exists_subset_range_and_iff {f : α → β} {p : Set β → Prop} :
    (∃ s, s ⊆ range f ∧ p s) ↔ ∃ s, p (f '' s) := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    p : Set β → Prop
    ⊢ Iff (Exists fun s => And (HasSubset.Subset s (Set.range f)) (p s)) (Exists f …
  -/
  rw [← exists_range_iff, range_image]; rfl
                                        /-
                                          🎉 no goals
                                        -/


@[deprecated exists_subset_range_and_iff (since := "2024-06-06")]
theorem exists_subset_range_iff {f : α → β} {p : Set β → Prop} :
                                                               /-
                                                                 α : Type u_1
                                                                 β : Type u_2
                                                                 f : α → β
                                                                 p : Set β → Prop
                                                                 ⊢ Iff (Exists fun s => Exists fun x => p s) (Exists fun s => p (Set.image f s))
                                                               -/
    (∃ (s : _) (_ : s ⊆ range f), p s) ↔ ∃ s, p (f '' s) := by simp
                                                               /-
                                                                 🎉 no goals
                                                               -/


@[simp]
theorem forall_subset_range_iff {f : α → β} {p : Set β → Prop} :
    (∀ s, s ⊆ range f → p s) ↔ ∀ s, p (f '' s) := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    p : Set β → Prop
    ⊢ Iff (∀ (s : Set β), HasSubset.Subset s (Set.range f) → p s) (∀ (s : Set α),  …
  -/
  rw [← forall_mem_range, range_image]; rfl
                                        /-
                                          🎉 no goals
                                        -/


@[simp]
theorem preimage_subset_preimage_iff {s t : Set α} {f : β → α} (hs : s ⊆ range f) :
    f ⁻¹' s ⊆ f ⁻¹' t ↔ s ⊆ t := by
  /-
    α : Type u_1
    β : Type u_2
    s t : Set α
    f : β → α
    hs : HasSubset.Subset s (Set.range f)
    ⊢ Iff (HasSubset.Subset (Set.preimage f s) (Set.preimage f t)) (HasSubset.Subs …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      β : Type u_2
      s t : Set α
      f : β → α
      hs : HasSubset.Subset s (Set.range f)
      ⊢ HasSubset.Subset (Set.preimage f s) (Set.preimage f t) → HasSubset.Subset s t
    -/
  · intro h x hx
    /-
      case mp
      α : Type u_1
      β : Type u_2
      s t : Set α
      f : β → α
      hs : HasSubset.Subset s (Set.range f)
      h : HasSubset.Subset (Set.preimage f s) (Set.preimage f t)
      x : α
      hx : Membership.mem s x
      ⊢ Membership.mem t x
    -/
    rcases hs hx with ⟨y, rfl⟩
    /-
      case mp.intro
      α : Type u_1
      β : Type u_2
      s t : Set α
      f : β → α
      hs : HasSubset.Subset s (Set.range f)
      h : HasSubset.Subset (Set.preimage f s) (Set.preimage f t)
      y : β
      hx : Membership.mem s (f y)
      ⊢ Membership.mem t (f y)
    -/
    exact h hx
    /-
      🎉 no goals
    -/
  /-
    case mpr
    α : Type u_1
    β : Type u_2
    s t : Set α
    f : β → α
    hs : HasSubset.Subset s (Set.range f)
    ⊢ HasSubset.Subset s t → HasSubset.Subset (Set.preimage f s) (Set.preimage f t)
  -/
  intro h x; apply h
             /-
               🎉 no goals
             -/


theorem preimage_eq_preimage' {s t : Set α} {f : β → α} (hs : s ⊆ range f) (ht : t ⊆ range f) :
    f ⁻¹' s = f ⁻¹' t ↔ s = t := by
  /-
    α : Type u_1
    β : Type u_2
    s t : Set α
    f : β → α
    hs : HasSubset.Subset s (Set.range f)
    ht : HasSubset.Subset t (Set.range f)
    ⊢ Iff (Eq (Set.preimage f s) (Set.preimage f t)) (Eq s t)
  -/
  constructor
    /-
      case mp
      α : Type u_1
      β : Type u_2
      s t : Set α
      f : β → α
      hs : HasSubset.Subset s (Set.range f)
      ht : HasSubset.Subset t (Set.range f)
      ⊢ Eq (Set.preimage f s) (Set.preimage f t) → Eq s t
    -/
  · intro h
    /-
      case mp
      α : Type u_1
      β : Type u_2
      s t : Set α
      f : β → α
      hs : HasSubset.Subset s (Set.range f)
      ht : HasSubset.Subset t (Set.range f)
      h : Eq (Set.preimage f s) (Set.preimage f t)
      ⊢ Eq s t
    -/
    apply Subset.antisymm
      /-
        case mp.h₁
        α : Type u_1
        β : Type u_2
        s t : Set α
        f : β → α
        hs : HasSubset.Subset s (Set.range f)
        ht : HasSubset.Subset t (Set.range f)
        h : Eq (Set.preimage f s) (Set.preimage f t)
        ⊢ HasSubset.Subset s t
      -/
    · rw [← preimage_subset_preimage_iff hs, h]
      /-
        🎉 no goals
      -/
      /-
        case mp.h₂
        α : Type u_1
        β : Type u_2
        s t : Set α
        f : β → α
        hs : HasSubset.Subset s (Set.range f)
        ht : HasSubset.Subset t (Set.range f)
        h : Eq (Set.preimage f s) (Set.preimage f t)
        ⊢ HasSubset.Subset t s
      -/
    · rw [← preimage_subset_preimage_iff ht, h]
      /-
        🎉 no goals
      -/
  /-
    case mpr
    α : Type u_1
    β : Type u_2
    s t : Set α
    f : β → α
    hs : HasSubset.Subset s (Set.range f)
    ht : HasSubset.Subset t (Set.range f)
    ⊢ Eq s t → Eq (Set.preimage f s) (Set.preimage f t)
  -/
  rintro rfl; rfl
              /-
                🎉 no goals
              -/

-- Porting note:
-- @[simp] `simp` can prove this

theorem preimage_inter_range {f : α → β} {s : Set β} : f ⁻¹' (s ∩ range f) = f ⁻¹' s :=
  Set.ext fun x => and_iff_left ⟨x, rfl⟩

-- Porting note:
-- @[simp] `simp` can prove this

theorem preimage_range_inter {f : α → β} {s : Set β} : f ⁻¹' (range f ∩ s) = f ⁻¹' s := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    s : Set β
    ⊢ Eq (Set.preimage f (Inter.inter (Set.range f) s)) (Set.preimage f s)
  -/
  rw [inter_comm, preimage_inter_range]
  /-
    🎉 no goals
  -/


theorem preimage_image_preimage {f : α → β} {s : Set β} : f ⁻¹' (f '' (f ⁻¹' s)) = f ⁻¹' s := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    s : Set β
    ⊢ Eq (Set.preimage f (Set.image f (Set.preimage f s))) (Set.preimage f s)
  -/
  rw [image_preimage_eq_range_inter, preimage_range_inter]
  /-
    🎉 no goals
  -/


@[simp, mfld_simps]
theorem range_id : range (@id α) = univ :=
  range_eq_univ.2 surjective_id


@[simp, mfld_simps]
theorem range_id' : (range fun x : α => x) = univ :=
  range_id


@[simp]
theorem _root_.Prod.range_fst [Nonempty β] : range (Prod.fst : α × β → α) = univ :=
  Prod.fst_surjective.range_eq


@[simp]
theorem _root_.Prod.range_snd [Nonempty α] : range (Prod.snd : α × β → β) = univ :=
  Prod.snd_surjective.range_eq


@[simp]
theorem range_eval {α : ι → Sort _} [∀ i, Nonempty (α i)] (i : ι) :
    range (eval i : (∀ i, α i) → α i) = univ :=
  (surjective_eval i).range_eq


                                                                    /-
                                                                      α : Type u_1
                                                                      β : Type u_2
                                                                      ⊢ Eq (Set.range Sum.inl) (setOf fun x => Eq x.isLeft Bool.true)
                                                                    -/
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/
theorem range_inl : range (@Sum.inl α β) = {x | Sum.isLeft x} := by ext (_|_) <;> simp
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/

                                                                     /-
                                                                       α : Type u_1
                                                                       β : Type u_2
                                                                       ⊢ Eq (Set.range Sum.inr) (setOf fun x => Eq x.isRight Bool.true)
                                                                     -/
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/
theorem range_inr : range (@Sum.inr α β) = {x | Sum.isRight x} := by ext (_|_) <;> simp
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


theorem isCompl_range_inl_range_inr : IsCompl (range <| @Sum.inl α β) (range Sum.inr) :=
  IsCompl.of_le
    (by
      /-
        α : Type u_1
        β : Type u_2
        ⊢ LE.le (Min.min (Set.range Sum.inl) (Set.range Sum.inr)) Bot.bot
      -/
      rintro y ⟨⟨x₁, rfl⟩, ⟨x₂, h⟩⟩
      /-
        case intro.intro.intro
        α : Type u_1
        β : Type u_2
        x₁ : α
        x₂ : β
        h : Eq (Sum.inr x₂) (Sum.inl x₁)
        ⊢ Membership.mem Bot.bot (Sum.inl x₁)
      -/
      exact Sum.noConfusion h)
      /-
        🎉 no goals
      -/
        /-
          α : Type u_1
          β : Type u_2
          ⊢ LE.le Top.top (Max.max (Set.range Sum.inl) (Set.range Sum.inr))
        -/
                                               /-
                                                 🎉 no goals
                                               -/
    (by rintro (x | y) - <;> [left; right] <;> exact mem_range_self _)
                                               /-
                                                 🎉 no goals
                                               -/


@[simp]
theorem range_inl_union_range_inr : range (Sum.inl : α → α ⊕ β) ∪ range Sum.inr = univ :=
  isCompl_range_inl_range_inr.sup_eq_top


@[simp]
theorem range_inl_inter_range_inr : range (Sum.inl : α → α ⊕ β) ∩ range Sum.inr = ∅ :=
  isCompl_range_inl_range_inr.inf_eq_bot


@[simp]
theorem range_inr_union_range_inl : range (Sum.inr : β → α ⊕ β) ∪ range Sum.inl = univ :=
  isCompl_range_inl_range_inr.symm.sup_eq_top


@[simp]
theorem range_inr_inter_range_inl : range (Sum.inr : β → α ⊕ β) ∩ range Sum.inl = ∅ :=
  isCompl_range_inl_range_inr.symm.inf_eq_bot


@[simp]
theorem preimage_inl_image_inr (s : Set β) : Sum.inl ⁻¹' (@Sum.inr α β '' s) = ∅ := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set β
    ⊢ Eq (Set.preimage Sum.inl (Set.image Sum.inr s)) EmptyCollection.emptyCollect …
  -/
  ext
  /-
    case h
    α : Type u_1
    β : Type u_2
    s : Set β
    x✝ : α
    ⊢ Iff (Membership.mem (Set.preimage Sum.inl (Set.image Sum.inr s)) x✝) (Member …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_inr_image_inl (s : Set α) : Sum.inr ⁻¹' (@Sum.inl α β '' s) = ∅ := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    ⊢ Eq (Set.preimage Sum.inr (Set.image Sum.inl s)) EmptyCollection.emptyCollect …
  -/
  ext
  /-
    case h
    α : Type u_1
    β : Type u_2
    s : Set α
    x✝ : β
    ⊢ Iff (Membership.mem (Set.preimage Sum.inr (Set.image Sum.inl s)) x✝) (Member …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_inl_range_inr : Sum.inl ⁻¹' range (Sum.inr : β → α ⊕ β) = ∅ := by
  /-
    α : Type u_1
    β : Type u_2
    ⊢ Eq (Set.preimage Sum.inl (Set.range Sum.inr)) EmptyCollection.emptyCollection
  -/
  rw [← image_univ, preimage_inl_image_inr]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_inr_range_inl : Sum.inr ⁻¹' range (Sum.inl : α → α ⊕ β) = ∅ := by
  /-
    α : Type u_1
    β : Type u_2
    ⊢ Eq (Set.preimage Sum.inr (Set.range Sum.inl)) EmptyCollection.emptyCollection
  -/
  rw [← image_univ, preimage_inr_image_inl]
  /-
    🎉 no goals
  -/


@[simp]
theorem compl_range_inl : (range (Sum.inl : α → α ⊕ β))ᶜ = range (Sum.inr : β → α ⊕ β) :=
  IsCompl.compl_eq isCompl_range_inl_range_inr


@[simp]
theorem compl_range_inr : (range (Sum.inr : β → α ⊕ β))ᶜ = range (Sum.inl : α → α ⊕ β) :=
  IsCompl.compl_eq isCompl_range_inl_range_inr.symm


theorem image_preimage_inl_union_image_preimage_inr (s : Set (α ⊕ β)) :
    Sum.inl '' (Sum.inl ⁻¹' s) ∪ Sum.inr '' (Sum.inr ⁻¹' s) = s := by
  rw [image_preimage_eq_inter_range, image_preimage_eq_inter_range, ← inter_union_distrib_left,
    range_inl_union_range_inr, inter_univ]


@[simp]
theorem range_quot_mk (r : α → α → Prop) : range (Quot.mk r) = univ :=
  Quot.mk_surjective.range_eq


@[simp]
theorem range_quot_lift {r : ι → ι → Prop} (hf : ∀ x y, r x y → f x = f y) :
    range (Quot.lift f hf) = range f :=
  ext fun _ => Quot.mk_surjective.exists


@[simp]
theorem range_quotient_mk {s : Setoid α} : range (Quotient.mk s) = univ :=
  range_quot_mk _


@[simp]
theorem range_quotient_lift [s : Setoid ι] (hf) :
    range (Quotient.lift f hf : Quotient s → α) = range f :=
  range_quot_lift _


@[simp]
theorem range_quotient_mk' {s : Setoid α} : range (Quotient.mk' : α → Quotient s) = univ :=
  range_quot_mk _


lemma Quotient.range_mk'' {sa : Setoid α} : range (Quotient.mk'' (s₁ := sa)) = univ :=
  range_quotient_mk


@[simp]
theorem range_quotient_lift_on' {s : Setoid ι} (hf) :
    (range fun x : Quotient s => Quotient.liftOn' x f hf) = range f :=
  range_quot_lift _


instance canLift (c) (p) [CanLift α β c p] :
    CanLift (Set α) (Set β) (c '' ·) fun s => ∀ x ∈ s, p x where
  prf _ hs := subset_range_iff_exists_image_eq.mp fun x hx => CanLift.prf _ (hs x hx)


theorem range_const_subset {c : α} : (range fun _ : ι => c) ⊆ {c} :=
  range_subset_iff.2 fun _ => rfl


@[simp]
theorem range_const : ∀ [Nonempty ι] {c : α}, (range fun _ : ι => c) = {c}
  | ⟨x⟩, _ =>
    (Subset.antisymm range_const_subset) fun _ hy =>
      (mem_singleton_iff.1 hy).symm ▸ mem_range_self x


theorem range_subtype_map {p : α → Prop} {q : β → Prop} (f : α → β) (h : ∀ x, p x → q (f x)) :
    range (Subtype.map f h) = (↑) ⁻¹' (f '' { x | p x }) := by
  /-
    α : Type u_1
    β : Type u_2
    p : α → Prop
    q : β → Prop
    f : α → β
    h : ∀ (x : α), p x → q (f x)
    ⊢ Eq (Set.range (Subtype.map f h)) (Set.preimage Subtype.val (Set.image f (set …
  -/
  ext ⟨x, hx⟩
  /-
    case h.mk
    α : Type u_1
    β : Type u_2
    p : α → Prop
    q : β → Prop
    f : α → β
    h : ∀ (x : α), p x → q (f x)
    x : β
    hx : q x
    ⊢ Iff (Membership.mem (Set.range (Subtype.map f h)) ⟨x, hx⟩) (Membership.mem ( …
  -/
  simp_rw [mem_preimage, mem_range, mem_image, Subtype.exists, Subtype.map]
  /-
    case h.mk
    α : Type u_1
    β : Type u_2
    p : α → Prop
    q : β → Prop
    f : α → β
    h : ∀ (x : α), p x → q (f x)
    x : β
    hx : q x
    ⊢ Iff (Exists fun a => Exists fun h_1 => Eq ⟨f a, ⋯⟩ ⟨x, hx⟩) (Exists fun x_1  …
  -/
  simp only [Subtype.mk.injEq, exists_prop, mem_setOf_eq]
  /-
    🎉 no goals
  -/


theorem image_swap_eq_preimage_swap : image (@Prod.swap α β) = preimage Prod.swap :=
  image_eq_preimage_of_inverse Prod.swap_leftInverse Prod.swap_rightInverse


theorem preimage_singleton_nonempty {f : α → β} {y : β} : (f ⁻¹' {y}).Nonempty ↔ y ∈ range f :=
  Iff.rfl


theorem preimage_singleton_eq_empty {f : α → β} {y : β} : f ⁻¹' {y} = ∅ ↔ y ∉ range f :=
  not_nonempty_iff_eq_empty.symm.trans preimage_singleton_nonempty.not


theorem range_subset_singleton {f : ι → α} {x : α} : range f ⊆ {x} ↔ f = const ι x := by
  /-
    α : Type u_1
    ι : Sort u_4
    f : ι → α
    x : α
    ⊢ Iff (HasSubset.Subset (Set.range f) (Singleton.singleton x)) (Eq f (Function …
  -/
  simp [range_subset_iff, funext_iff, mem_singleton]
  /-
    🎉 no goals
  -/


theorem image_compl_preimage {f : α → β} {s : Set β} : f '' (f ⁻¹' s)ᶜ = range f \ s := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    s : Set β
    ⊢ Eq (Set.image f (HasCompl.compl (Set.preimage f s))) (SDiff.sdiff (Set.range …
  -/
  rw [compl_eq_univ_diff, image_diff_preimage, image_univ]
  /-
    🎉 no goals
  -/


theorem rangeFactorization_eq {f : ι → β} : Subtype.val ∘ rangeFactorization f = f :=
  funext fun _ => rfl


@[simp]
theorem rangeFactorization_coe (f : ι → β) (a : ι) : (rangeFactorization f a : β) = f a :=
  rfl


@[simp]
theorem coe_comp_rangeFactorization (f : ι → β) : (↑) ∘ rangeFactorization f = f := rfl


theorem surjective_onto_range : Surjective (rangeFactorization f) := fun ⟨_, ⟨i, rfl⟩⟩ => ⟨i, rfl⟩


theorem image_eq_range (f : α → β) (s : Set α) : f '' s = range fun x : s => f x := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    s : Set α
    ⊢ Eq (Set.image f s) (Set.range fun x => f ↑x)
  -/
  ext
  /-
    case h
    α : Type u_1
    β : Type u_2
    f : α → β
    s : Set α
    x✝ : β
    ⊢ Iff (Membership.mem (Set.image f s) x✝) (Membership.mem (Set.range fun x =>  …
  -/
  constructor
    /-
      case h.mp
      α : Type u_1
      β : Type u_2
      f : α → β
      s : Set α
      x✝ : β
      ⊢ Membership.mem (Set.image f s) x✝ → Membership.mem (Set.range fun x => f ↑x) …
    -/
  · rintro ⟨x, h1, h2⟩
    /-
      case h.mp.intro.intro
      α : Type u_1
      β : Type u_2
      f : α → β
      s : Set α
      x✝ : β
      x : α
      h1 : Membership.mem s x
      h2 : Eq (f x) x✝
      ⊢ Membership.mem (Set.range fun x => f ↑x) x✝
    -/
    exact ⟨⟨x, h1⟩, h2⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      α : Type u_1
      β : Type u_2
      f : α → β
      s : Set α
      x✝ : β
      ⊢ Membership.mem (Set.range fun x => f ↑x) x✝ → Membership.mem (Set.image f s) …
    -/
  · rintro ⟨⟨x, h1⟩, h2⟩
    /-
      case h.mpr.intro.mk
      α : Type u_1
      β : Type u_2
      f : α → β
      s : Set α
      x✝ : β
      x : α
      h1 : Membership.mem s x
      h2 : Eq ((fun x => f ↑x) ⟨x, h1⟩) x✝
      ⊢ Membership.mem (Set.image f s) x✝
    -/
    exact ⟨x, h1, h2⟩
    /-
      🎉 no goals
    -/


theorem _root_.Sum.range_eq (f : α ⊕ β → γ) :
    range f = range (f ∘ Sum.inl) ∪ range (f ∘ Sum.inr) :=
  ext fun _ => Sum.exists


@[simp]
theorem Sum.elim_range (f : α → γ) (g : β → γ) : range (Sum.elim f g) = range f ∪ range g :=
  Sum.range_eq _


theorem range_ite_subset' {p : Prop} [Decidable p] {f g : α → β} :
    range (if p then f else g) ⊆ range f ∪ range g := by
  /-
    α : Type u_1
    β : Type u_2
    p : Prop
    inst✝ : Decidable p
    f g : α → β
    ⊢ HasSubset.Subset (Set.range (ite p f g)) (Union.union (Set.range f) (Set.ran …
  -/
  by_cases h : p
    /-
      case pos
      α : Type u_1
      β : Type u_2
      p : Prop
      inst✝ : Decidable p
      f g : α → β
      h : p
      ⊢ HasSubset.Subset (Set.range (ite p f g)) (Union.union (Set.range f) (Set.ran …
    -/
  · rw [if_pos h]
    /-
      case pos
      α : Type u_1
      β : Type u_2
      p : Prop
      inst✝ : Decidable p
      f g : α → β
      h : p
      ⊢ HasSubset.Subset (Set.range f) (Union.union (Set.range f) (Set.range g))
    -/
    exact subset_union_left
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      p : Prop
      inst✝ : Decidable p
      f g : α → β
      h : Not p
      ⊢ HasSubset.Subset (Set.range (ite p f g)) (Union.union (Set.range f) (Set.ran …
    -/
  · rw [if_neg h]
    /-
      case neg
      α : Type u_1
      β : Type u_2
      p : Prop
      inst✝ : Decidable p
      f g : α → β
      h : Not p
      ⊢ HasSubset.Subset (Set.range g) (Union.union (Set.range f) (Set.range g))
    -/
    exact subset_union_right
    /-
      🎉 no goals
    -/


theorem range_ite_subset {p : α → Prop} [DecidablePred p] {f g : α → β} :
    (range fun x => if p x then f x else g x) ⊆ range f ∪ range g := by
  /-
    α : Type u_1
    β : Type u_2
    p : α → Prop
    inst✝ : DecidablePred p
    f g : α → β
    ⊢ HasSubset.Subset (Set.range fun x => ite (p x) (f x) (g x)) (Union.union (Se …
  -/
  rw [range_subset_iff]; intro x; by_cases h : p x
    /-
      case pos
      α : Type u_1
      β : Type u_2
      p : α → Prop
      inst✝ : DecidablePred p
      f g : α → β
      x : α
      h : p x
      ⊢ Membership.mem (Union.union (Set.range f) (Set.range g)) (ite (p x) (f x) (g …
    -/
  · simp only [if_pos h, mem_union, mem_range, exists_apply_eq_apply, true_or]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      p : α → Prop
      inst✝ : DecidablePred p
      f g : α → β
      x : α
      h : Not (p x)
      ⊢ Membership.mem (Union.union (Set.range f) (Set.range g)) (ite (p x) (f x) (g …
    -/
  · simp [if_neg h, mem_union, mem_range_self]
    /-
      🎉 no goals
    -/


@[simp]
theorem preimage_range (f : α → β) : f ⁻¹' range f = univ :=
  eq_univ_of_forall mem_range_self


/-- The range of a function from a `Unique` type contains just the
function applied to its single value. -/
theorem range_unique [h : Unique ι] : range f = {f default} := by
  /-
    α : Type u_1
    ι : Sort u_4
    f : ι → α
    h : Unique ι
    ⊢ Eq (Set.range f) (Singleton.singleton (f Inhabited.default))
  -/
  ext x
  /-
    case h
    α : Type u_1
    ι : Sort u_4
    f : ι → α
    h : Unique ι
    x : α
    ⊢ Iff (Membership.mem (Set.range f) x) (Membership.mem (Singleton.singleton (f …
  -/
  rw [mem_range]
  /-
    case h
    α : Type u_1
    ι : Sort u_4
    f : ι → α
    h : Unique ι
    x : α
    ⊢ Iff (Exists fun y => Eq (f y) x) (Membership.mem (Singleton.singleton (f Inh …
  -/
  constructor
    /-
      case h.mp
      α : Type u_1
      ι : Sort u_4
      f : ι → α
      h : Unique ι
      x : α
      ⊢ (Exists fun y => Eq (f y) x) → Membership.mem (Singleton.singleton (f Inhabi …
    -/
  · rintro ⟨i, hi⟩
    /-
      case h.mp.intro
      α : Type u_1
      ι : Sort u_4
      f : ι → α
      h : Unique ι
      x : α
      i : ι
      hi : Eq (f i) x
      ⊢ Membership.mem (Singleton.singleton (f Inhabited.default)) x
    -/
    rw [h.uniq i] at hi
    /-
      case h.mp.intro
      α : Type u_1
      ι : Sort u_4
      f : ι → α
      h : Unique ι
      x : α
      i : ι
      hi : Eq (f Inhabited.default) x
      ⊢ Membership.mem (Singleton.singleton (f Inhabited.default)) x
    -/
    exact hi ▸ mem_singleton _
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      α : Type u_1
      ι : Sort u_4
      f : ι → α
      h : Unique ι
      x : α
      ⊢ Membership.mem (Singleton.singleton (f Inhabited.default)) x → Exists fun y  …
    -/
  · exact fun h => ⟨default, h.symm⟩
    /-
      🎉 no goals
    -/


theorem range_diff_image_subset (f : α → β) (s : Set α) : range f \ f '' s ⊆ f '' sᶜ :=
  fun _ ⟨⟨x, h₁⟩, h₂⟩ => ⟨x, fun h => h₂ ⟨x, h, h₁⟩, h₁⟩


@[simp]
theorem range_inclusion (h : s ⊆ t) : range (inclusion h) = { x : t | (x : α) ∈ s } := by
  /-
    α : Type u_1
    s t : Set α
    h : HasSubset.Subset s t
    ⊢ Eq (Set.range (Set.inclusion h)) (setOf fun x => Membership.mem s ↑x)
  -/
  ext ⟨x, hx⟩
  /-
    case h.mk
    α : Type u_1
    s t : Set α
    h : HasSubset.Subset s t
    x : α
    hx : Membership.mem t x
    ⊢ Iff (Membership.mem (Set.range (Set.inclusion h)) ⟨x, hx⟩) (Membership.mem ( …
  -/
  simp
  /-
    🎉 no goals
  -/

-- When `f` is injective, see also `Equiv.ofInjective`.

theorem leftInverse_rangeSplitting (f : α → β) :
    LeftInverse (rangeFactorization f) (rangeSplitting f) := fun x => by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    x : ↑(Set.range f)
    ⊢ Eq (Set.rangeFactorization f (Set.rangeSplitting f x)) x
  -/
  ext
  /-
    case a
    α : Type u_1
    β : Type u_2
    f : α → β
    x : ↑(Set.range f)
    ⊢ Eq ↑(Set.rangeFactorization f (Set.rangeSplitting f x)) ↑x
  -/
  simp only [rangeFactorization_coe]
  /-
    case a
    α : Type u_1
    β : Type u_2
    f : α → β
    x : ↑(Set.range f)
    ⊢ Eq (f (Set.rangeSplitting f x)) ↑x
  -/
  apply apply_rangeSplitting
  /-
    🎉 no goals
  -/


theorem rangeSplitting_injective (f : α → β) : Injective (rangeSplitting f) :=
  (leftInverse_rangeSplitting f).injective


theorem rightInverse_rangeSplitting {f : α → β} (h : Injective f) :
    RightInverse (rangeFactorization f) (rangeSplitting f) :=
  (leftInverse_rangeSplitting f).rightInverse_of_injective fun _ _ hxy =>
    h <| Subtype.ext_iff.1 hxy


theorem preimage_rangeSplitting {f : α → β} (hf : Injective f) :
    preimage (rangeSplitting f) = image (rangeFactorization f) :=
  (image_eq_preimage_of_inverse (rightInverse_rangeSplitting hf)
      (leftInverse_rangeSplitting f)).symm


theorem isCompl_range_some_none (α : Type*) : IsCompl (range (some : α → Option α)) {none} :=
  IsCompl.of_le (fun _ ⟨⟨_, ha⟩, (hn : _ = none)⟩ => Option.some_ne_none _ (ha.trans hn))
    fun x _ => Option.casesOn x (Or.inr rfl) fun _ => Or.inl <| mem_range_self _


@[simp]
theorem compl_range_some (α : Type*) : (range (some : α → Option α))ᶜ = {none} :=
  (isCompl_range_some_none α).compl_eq


@[simp]
theorem range_some_inter_none (α : Type*) : range (some : α → Option α) ∩ {none} = ∅ :=
  (isCompl_range_some_none α).inf_eq_bot

-- Porting note:
-- @[simp] `simp` can prove this

theorem range_some_union_none (α : Type*) : range (some : α → Option α) ∪ {none} = univ :=
  (isCompl_range_some_none α).sup_eq_top


@[simp]
theorem insert_none_range_some (α : Type*) : insert none (range (some : α → Option α)) = univ :=
  (isCompl_range_some_none α).symm.sup_eq_top


/-- The image of a subsingleton is a subsingleton. -/
theorem Subsingleton.image (hs : s.Subsingleton) (f : α → β) : (f '' s).Subsingleton :=
  fun _ ⟨_, hx, Hx⟩ _ ⟨_, hy, Hy⟩ => Hx ▸ Hy ▸ congr_arg f (hs hx hy)


/-- The preimage of a subsingleton under an injective map is a subsingleton. -/
theorem Subsingleton.preimage {s : Set β} (hs : s.Subsingleton)
    (hf : Function.Injective f) : (f ⁻¹' s).Subsingleton := fun _ ha _ hb => hf <| hs ha hb


/-- If the image of a set under an injective map is a subsingleton, the set is a subsingleton. -/
theorem subsingleton_of_image (hf : Function.Injective f) (s : Set α)
    (hs : (f '' s).Subsingleton) : s.Subsingleton :=
  (hs.preimage hf).anti <| subset_preimage_image _ _


/-- If the preimage of a set under a surjective map is a subsingleton,
the set is a subsingleton. -/
theorem subsingleton_of_preimage (hf : Function.Surjective f) (s : Set β)
    (hs : (f ⁻¹' s).Subsingleton) : s.Subsingleton := fun fx hx fy hy => by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    hf : Function.Surjective f
    s : Set β
    hs : (Set.preimage f s).Subsingleton
    fx : β
    hx : Membership.mem s fx
    fy : β
    hy : Membership.mem s fy
    ⊢ Eq fx fy
  -/
  rcases hf fx, hf fy with ⟨⟨x, rfl⟩, ⟨y, rfl⟩⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    f : α → β
    hf : Function.Surjective f
    s : Set β
    hs : (Set.preimage f s).Subsingleton
    x : α
    hx : Membership.mem s (f x)
    y : α
    hy : Membership.mem s (f y)
    ⊢ Eq (f x) (f y)
  -/
  exact congr_arg f (hs hx hy)
  /-
    🎉 no goals
  -/


theorem subsingleton_range {α : Sort*} [Subsingleton α] (f : α → β) : (range f).Subsingleton :=
  forall_mem_range.2 fun x => forall_mem_range.2 fun y => congr_arg f (Subsingleton.elim x y)


/-- The preimage of a nontrivial set under a surjective map is nontrivial. -/
theorem Nontrivial.preimage {s : Set β} (hs : s.Nontrivial)
    (hf : Function.Surjective f) : (f ⁻¹' s).Nontrivial := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    s : Set β
    hs : s.Nontrivial
    hf : Function.Surjective f
    ⊢ (Set.preimage f s).Nontrivial
  -/
  rcases hs with ⟨fx, hx, fy, hy, hxy⟩
  /-
    case intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    f : α → β
    s : Set β
    hf : Function.Surjective f
    fx : β
    hx : Membership.mem s fx
    fy : β
    hy : Membership.mem s fy
    hxy : Ne fx fy
    ⊢ (Set.preimage f s).Nontrivial
  -/
  rcases hf fx, hf fy with ⟨⟨x, rfl⟩, ⟨y, rfl⟩⟩
  /-
    case intro.intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    f : α → β
    s : Set β
    hf : Function.Surjective f
    x : α
    hx : Membership.mem s (f x)
    y : α
    hy : Membership.mem s (f y)
    hxy : Ne (f x) (f y)
    ⊢ (Set.preimage f s).Nontrivial
  -/
  exact ⟨x, hx, y, hy, mt (congr_arg f) hxy⟩
  /-
    🎉 no goals
  -/


/-- The image of a nontrivial set under an injective map is nontrivial. -/
theorem Nontrivial.image (hs : s.Nontrivial) (hf : Function.Injective f) :
    (f '' s).Nontrivial :=
  let ⟨x, hx, y, hy, hxy⟩ := hs
  ⟨f x, mem_image_of_mem f hx, f y, mem_image_of_mem f hy, hf.ne hxy⟩


theorem Nontrivial.image_of_injOn (hs : s.Nontrivial) (hf : s.InjOn f) :
    (f '' s).Nontrivial := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    f : α → β
    hs : s.Nontrivial
    hf : Set.InjOn f s
    ⊢ (Set.image f s).Nontrivial
  -/
  obtain ⟨x, hx, y, hy, hxy⟩ := hs
  /-
    case intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    s : Set α
    f : α → β
    hf : Set.InjOn f s
    x : α
    hx : Membership.mem s x
    y : α
    hy : Membership.mem s y
    hxy : Ne x y
    ⊢ (Set.image f s).Nontrivial
  -/
  exact ⟨f x, mem_image_of_mem _ hx, f y, mem_image_of_mem _ hy, (hxy <| hf hx hy ·)⟩
  /-
    🎉 no goals
  -/


/-- If the image of a set is nontrivial, the set is nontrivial. -/
theorem nontrivial_of_image (f : α → β) (s : Set α) (hs : (f '' s).Nontrivial) : s.Nontrivial :=
  let ⟨_, ⟨x, hx, rfl⟩, _, ⟨y, hy, rfl⟩, hxy⟩ := hs
  ⟨x, hx, y, hy, mt (congr_arg f) hxy⟩


@[simp]
theorem image_nontrivial (hf : f.Injective) : (f '' s).Nontrivial ↔ s.Nontrivial :=
  ⟨nontrivial_of_image f s, fun h ↦ h.image hf⟩


@[simp]
theorem InjOn.image_nontrivial_iff (hf : s.InjOn f) :
    (f '' s).Nontrivial ↔ s.Nontrivial :=
  ⟨nontrivial_of_image f s, fun h ↦ h.image_of_injOn hf⟩


/-- If the preimage of a set under an injective map is nontrivial, the set is nontrivial. -/
theorem nontrivial_of_preimage (hf : Function.Injective f) (s : Set β)
    (hs : (f ⁻¹' s).Nontrivial) : s.Nontrivial :=
  (hs.image hf).mono <| image_preimage_subset _ _


theorem Surjective.preimage_injective (hf : Surjective f) : Injective (preimage f) := fun _ _ =>
  (preimage_eq_preimage hf).1


theorem Injective.preimage_image (hf : Injective f) (s : Set α) : f ⁻¹' (f '' s) = s :=
  preimage_image_eq s hf


theorem Injective.preimage_surjective (hf : Injective f) : Surjective (preimage f) := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    hf : Function.Injective f
    ⊢ Function.Surjective (Set.preimage f)
  -/
  intro s
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    hf : Function.Injective f
    s : Set α
    ⊢ Exists fun a => Eq (Set.preimage f a) s
  -/
  use f '' s
  /-
    case h
    α : Type u_1
    β : Type u_2
    f : α → β
    hf : Function.Injective f
    s : Set α
    ⊢ Eq (Set.preimage f (Set.image f s)) s
  -/
  rw [hf.preimage_image]
  /-
    🎉 no goals
  -/


theorem Injective.subsingleton_image_iff (hf : Injective f) {s : Set α} :
    (f '' s).Subsingleton ↔ s.Subsingleton :=
  ⟨subsingleton_of_image hf s, fun h => h.image f⟩


theorem Surjective.image_preimage (hf : Surjective f) (s : Set β) : f '' (f ⁻¹' s) = s :=
  image_preimage_eq s hf


theorem Surjective.image_surjective (hf : Surjective f) : Surjective (image f) := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    hf : Function.Surjective f
    ⊢ Function.Surjective (Set.image f)
  -/
  intro s
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    hf : Function.Surjective f
    s : Set β
    ⊢ Exists fun a => Eq (Set.image f a) s
  -/
  use f ⁻¹' s
  /-
    case h
    α : Type u_1
    β : Type u_2
    f : α → β
    hf : Function.Surjective f
    s : Set β
    ⊢ Eq (Set.image f (Set.preimage f s)) s
  -/
  rw [hf.image_preimage]
  /-
    🎉 no goals
  -/


@[simp]
theorem Surjective.nonempty_preimage (hf : Surjective f) {s : Set β} :
                                          /-
                                            α : Type u_1
                                            β : Type u_2
                                            f : α → β
                                            hf : Function.Surjective f
                                            s : Set β
                                            ⊢ Iff (Set.preimage f s).Nonempty s.Nonempty
                                          -/
    (f ⁻¹' s).Nonempty ↔ s.Nonempty := by rw [← image_nonempty, hf.image_preimage]
                                          /-
                                            🎉 no goals
                                          -/


theorem Injective.image_injective (hf : Injective f) : Injective (image f) := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    hf : Function.Injective f
    ⊢ Function.Injective (Set.image f)
  -/
  intro s t h
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    hf : Function.Injective f
    s t : Set α
    h : Eq (Set.image f s) (Set.image f t)
    ⊢ Eq s t
  -/
  rw [← preimage_image_eq s hf, ← preimage_image_eq t hf, h]
  /-
    🎉 no goals
  -/


lemma Injective.image_strictMono (inj : Function.Injective f) : StrictMono (image f) :=
  monotone_image.strictMono_of_injective inj.image_injective


theorem Surjective.preimage_subset_preimage_iff {s t : Set β} (hf : Surjective f) :
    f ⁻¹' s ⊆ f ⁻¹' t ↔ s ⊆ t := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    s t : Set β
    hf : Function.Surjective f
    ⊢ Iff (HasSubset.Subset (Set.preimage f s) (Set.preimage f t)) (HasSubset.Subs …
  -/
  apply Set.preimage_subset_preimage_iff
  /-
    case hs
    α : Type u_1
    β : Type u_2
    f : α → β
    s t : Set β
    hf : Function.Surjective f
    ⊢ HasSubset.Subset s (Set.range f)
  -/
  rw [hf.range_eq]
  /-
    case hs
    α : Type u_1
    β : Type u_2
    f : α → β
    s t : Set β
    hf : Function.Surjective f
    ⊢ HasSubset.Subset s Set.univ
  -/
  apply subset_univ
  /-
    🎉 no goals
  -/


theorem Surjective.range_comp {ι' : Sort*} {f : ι → ι'} (hf : Surjective f) (g : ι' → α) :
    range (g ∘ f) = range g :=
  ext fun y => (@Surjective.exists _ _ _ hf fun x => g x = y).symm


theorem Injective.mem_range_iff_existsUnique (hf : Injective f) {b : β} :
    b ∈ range f ↔ ∃! a, f a = b :=
  ⟨fun ⟨a, h⟩ => ⟨a, h, fun _ ha => hf (ha.trans h.symm)⟩, ExistsUnique.exists⟩


alias ⟨Injective.existsUnique_of_mem_range, _⟩ := Injective.mem_range_iff_existsUnique


@[deprecated (since := "2024-09-25")]
alias Injective.mem_range_iff_exists_unique := Injective.mem_range_iff_existsUnique


@[deprecated (since := "2024-09-25")]
alias Injective.exists_unique_of_mem_range := Injective.existsUnique_of_mem_range


theorem Injective.compl_image_eq (hf : Injective f) (s : Set α) :
    (f '' s)ᶜ = f '' sᶜ ∪ (range f)ᶜ := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    hf : Function.Injective f
    s : Set α
    ⊢ Eq (HasCompl.compl (Set.image f s)) (Union.union (Set.image f (HasCompl.comp …
  -/
  ext y
  /-
    case h
    α : Type u_1
    β : Type u_2
    f : α → β
    hf : Function.Injective f
    s : Set α
    y : β
    ⊢ Iff (Membership.mem (HasCompl.compl (Set.image f s)) y) (Membership.mem (Uni …
  -/
  rcases em (y ∈ range f) with (⟨x, rfl⟩ | hx)
    /-
      case h.inl.intro
      α : Type u_1
      β : Type u_2
      f : α → β
      hf : Function.Injective f
      s : Set α
      x : α
      ⊢ Iff (Membership.mem (HasCompl.compl (Set.image f s)) (f x)) (Membership.mem  …
    -/
  · simp [hf.eq_iff]
    /-
      🎉 no goals
    -/
    /-
      case h.inr
      α : Type u_1
      β : Type u_2
      f : α → β
      hf : Function.Injective f
      s : Set α
      y : β
      hx : Not (Membership.mem (Set.range f) y)
      ⊢ Iff (Membership.mem (HasCompl.compl (Set.image f s)) y) (Membership.mem (Uni …
    -/
  · rw [mem_range, not_exists] at hx
    /-
      case h.inr
      α : Type u_1
      β : Type u_2
      f : α → β
      hf : Function.Injective f
      s : Set α
      y : β
      hx : ∀ (x : α), Not (Eq (f x) y)
      ⊢ Iff (Membership.mem (HasCompl.compl (Set.image f s)) y) (Membership.mem (Uni …
    -/
    simp [hx]
    /-
      🎉 no goals
    -/


theorem LeftInverse.image_image {g : β → α} (h : LeftInverse g f) (s : Set α) :
                            /-
                              α : Type u_1
                              β : Type u_2
                              f : α → β
                              g : β → α
                              h : Function.LeftInverse g f
                              s : Set α
                              ⊢ Eq (Set.image g (Set.image f s)) s
                            -/
    g '' (f '' s) = s := by rw [← image_comp, h.comp_eq_id, image_id]
                            /-
                              🎉 no goals
                            -/


theorem LeftInverse.preimage_preimage {g : β → α} (h : LeftInverse g f) (s : Set α) :
                              /-
                                α : Type u_1
                                β : Type u_2
                                f : α → β
                                g : β → α
                                h : Function.LeftInverse g f
                                s : Set α
                                ⊢ Eq (Set.preimage f (Set.preimage g s)) s
                              -/
    f ⁻¹' (g ⁻¹' s) = s := by rw [← preimage_comp, h.comp_eq_id, preimage_id]
                              /-
                                🎉 no goals
                              -/


protected theorem Involutive.preimage {f : α → α} (hf : Involutive f) : Involutive (preimage f) :=
  hf.rightInverse.preimage_preimage


@[simp] lemma range_comp {α : Type*} (f : ι' → α) (e : E) : range (f ∘ e) = range f :=
  (EquivLike.surjective _).range_comp _


theorem coe_image {p : α → Prop} {s : Set (Subtype p)} :
    (↑) '' s = { x | ∃ h : p x, (⟨x, h⟩ : Subtype p) ∈ s } :=
  Set.ext fun a =>
    ⟨fun ⟨⟨_, ha'⟩, in_s, h_eq⟩ => h_eq ▸ ⟨ha', in_s⟩, fun ⟨ha, in_s⟩ => ⟨⟨a, ha⟩, in_s, rfl⟩⟩


@[simp]
theorem coe_image_of_subset {s t : Set α} (h : t ⊆ s) : (↑) '' { x : ↥s | ↑x ∈ t } = t := by
  /-
    α : Type u_1
    s t : Set α
    h : HasSubset.Subset t s
    ⊢ Eq (Set.image Subtype.val (setOf fun x => Membership.mem t ↑x)) t
  -/
  ext x
  /-
    case h
    α : Type u_1
    s t : Set α
    h : HasSubset.Subset t s
    x : α
    ⊢ Iff (Membership.mem (Set.image Subtype.val (setOf fun x => Membership.mem t  …
  -/
  rw [mem_image]
  /-
    case h
    α : Type u_1
    s t : Set α
    h : HasSubset.Subset t s
    x : α
    ⊢ Iff (Exists fun x_1 => And (Membership.mem (setOf fun x => Membership.mem t  …
  -/
  exact ⟨fun ⟨_, hx', hx⟩ => hx ▸ hx', fun hx => ⟨⟨x, h hx⟩, hx, rfl⟩⟩
  /-
    🎉 no goals
  -/


theorem range_coe {s : Set α} : range ((↑) : s → α) = s := by
  /-
    α : Type u_1
    s : Set α
    ⊢ Eq (Set.range Subtype.val) s
  -/
  rw [← image_univ]
  /-
    α : Type u_1
    s : Set α
    ⊢ Eq (Set.image Subtype.val Set.univ) s
  -/
  simp [-image_univ, coe_image]
  /-
    🎉 no goals
  -/


/-- A variant of `range_coe`. Try to use `range_coe` if possible.
  This version is useful when defining a new type that is defined as the subtype of something.
  In that case, the coercion doesn't fire anymore. -/
theorem range_val {s : Set α} : range (Subtype.val : s → α) = s :=
  range_coe


/-- We make this the simp lemma instead of `range_coe`. The reason is that if we write
  for `s : Set α` the function `(↑) : s → α`, then the inferred implicit arguments of `(↑)` are
  `↑α (fun x ↦ x ∈ s)`. -/
@[simp]
theorem range_coe_subtype {p : α → Prop} : range ((↑) : Subtype p → α) = { x | p x } :=
  range_coe


@[simp]
theorem coe_preimage_self (s : Set α) : ((↑) : s → α) ⁻¹' s = univ := by
  /-
    α : Type u_1
    s : Set α
    ⊢ Eq (Set.preimage Subtype.val s) Set.univ
  -/
  rw [← preimage_range, range_coe]
  /-
    🎉 no goals
  -/


theorem range_val_subtype {p : α → Prop} : range (Subtype.val : Subtype p → α) = { x | p x } :=
  range_coe


theorem coe_image_subset (s : Set α) (t : Set s) : ((↑) : s → α) '' t ⊆ s :=
  fun x ⟨y, _, yvaleq⟩ => by
  /-
    α : Type u_1
    s : Set α
    t : Set ↑s
    x : α
    x✝ : Membership.mem (Set.image Subtype.val t) x
    y : Subtype fun x => Membership.mem s x
    left✝ : Membership.mem t y
    yvaleq : Eq (↑y) x
    ⊢ Membership.mem s x
  -/
  rw [← yvaleq]; exact y.property
                 /-
                   🎉 no goals
                 -/


theorem coe_image_univ (s : Set α) : ((↑) : s → α) '' Set.univ = s :=
  image_univ.trans range_coe


@[simp]
theorem image_preimage_coe (s t : Set α) : ((↑) : s → α) '' (((↑) : s → α) ⁻¹' t) = s ∩ t :=
  image_preimage_eq_range_inter.trans <| congr_arg (· ∩ t) range_coe


theorem image_preimage_val (s t : Set α) : (Subtype.val : s → α) '' (Subtype.val ⁻¹' t) = s ∩ t :=
  image_preimage_coe s t


theorem preimage_coe_eq_preimage_coe_iff {s t u : Set α} :
    ((↑) : s → α) ⁻¹' t = ((↑) : s → α) ⁻¹' u ↔ s ∩ t = s ∩ u := by
  /-
    α : Type u_1
    s t u : Set α
    ⊢ Iff (Eq (Set.preimage Subtype.val t) (Set.preimage Subtype.val u)) (Eq (Inte …
  -/
  rw [← image_preimage_coe, ← image_preimage_coe, coe_injective.image_injective.eq_iff]
  /-
    🎉 no goals
  -/


theorem preimage_coe_self_inter (s t : Set α) :
    ((↑) : s → α) ⁻¹' (s ∩ t) = ((↑) : s → α) ⁻¹' t := by
  /-
    α : Type u_1
    s t : Set α
    ⊢ Eq (Set.preimage Subtype.val (Inter.inter s t)) (Set.preimage Subtype.val t)
  -/
  rw [preimage_coe_eq_preimage_coe_iff, ← inter_assoc, inter_self]
  /-
    🎉 no goals
  -/

-- Porting note:
-- @[simp] `simp` can prove this

theorem preimage_coe_inter_self (s t : Set α) :
    ((↑) : s → α) ⁻¹' (t ∩ s) = ((↑) : s → α) ⁻¹' t := by
  /-
    α : Type u_1
    s t : Set α
    ⊢ Eq (Set.preimage Subtype.val (Inter.inter t s)) (Set.preimage Subtype.val t)
  -/
  rw [inter_comm, preimage_coe_self_inter]
  /-
    🎉 no goals
  -/


theorem preimage_val_eq_preimage_val_iff (s t u : Set α) :
    (Subtype.val : s → α) ⁻¹' t = Subtype.val ⁻¹' u ↔ s ∩ t = s ∩ u :=
  preimage_coe_eq_preimage_coe_iff


lemma preimage_val_subset_preimage_val_iff (s t u : Set α) :
    (Subtype.val ⁻¹' t : Set s) ⊆ Subtype.val ⁻¹' u ↔ s ∩ t ⊆ s ∩ u := by
  /-
    α : Type u_1
    s t u : Set α
    ⊢ Iff (HasSubset.Subset (Set.preimage Subtype.val t) (Set.preimage Subtype.val …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      s t u : Set α
      ⊢ HasSubset.Subset (Set.preimage Subtype.val t) (Set.preimage Subtype.val u) → …
    -/
  · rw [← image_preimage_coe, ← image_preimage_coe]
    /-
      case mp
      α : Type u_1
      s t u : Set α
      ⊢ HasSubset.Subset (Set.preimage Subtype.val t) (Set.preimage Subtype.val u) → …
    -/
    exact image_subset _
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      s t u : Set α
      ⊢ HasSubset.Subset (Inter.inter s t) (Inter.inter s u) → HasSubset.Subset (Set …
    -/
  · intro h x a
    /-
      case mpr
      α : Type u_1
      s t u : Set α
      h : HasSubset.Subset (Inter.inter s t) (Inter.inter s u)
      x : ↑s
      a : Membership.mem (Set.preimage Subtype.val t) x
      ⊢ Membership.mem (Set.preimage Subtype.val u) x
    -/
    exact (h ⟨x.2, a⟩).2
    /-
      🎉 no goals
    -/


theorem exists_set_subtype {t : Set α} (p : Set α → Prop) :
    (∃ s : Set t, p (((↑) : t → α) '' s)) ↔ ∃ s : Set α, s ⊆ t ∧ p s := by
  /-
    α : Type u_1
    t : Set α
    p : Set α → Prop
    ⊢ Iff (Exists fun s => p (Set.image Subtype.val s)) (Exists fun s => And (HasS …
  -/
  rw [← exists_subset_range_and_iff, range_coe]
  /-
    🎉 no goals
  -/


theorem forall_set_subtype {t : Set α} (p : Set α → Prop) :
    (∀ s : Set t, p (((↑) : t → α) '' s)) ↔ ∀ s : Set α, s ⊆ t → p s := by
  /-
    α : Type u_1
    t : Set α
    p : Set α → Prop
    ⊢ Iff (∀ (s : Set ↑t), p (Set.image Subtype.val s)) (∀ (s : Set α), HasSubset. …
  -/
  rw [← forall_subset_range_iff, range_coe]
  /-
    🎉 no goals
  -/


theorem preimage_coe_nonempty {s t : Set α} :
    (((↑) : s → α) ⁻¹' t).Nonempty ↔ (s ∩ t).Nonempty := by
  /-
    α : Type u_1
    s t : Set α
    ⊢ Iff (Set.preimage Subtype.val t).Nonempty (Inter.inter s t).Nonempty
  -/
  rw [← image_preimage_coe, image_nonempty]
  /-
    🎉 no goals
  -/


theorem preimage_coe_eq_empty {s t : Set α} : ((↑) : s → α) ⁻¹' t = ∅ ↔ s ∩ t = ∅ := by
  /-
    α : Type u_1
    s t : Set α
    ⊢ Iff (Eq (Set.preimage Subtype.val t) EmptyCollection.emptyCollection) (Eq (I …
  -/
  simp [← not_nonempty_iff_eq_empty, preimage_coe_nonempty]
  /-
    🎉 no goals
  -/

-- Porting note:
-- @[simp] `simp` can prove this

theorem preimage_coe_compl (s : Set α) : ((↑) : s → α) ⁻¹' sᶜ = ∅ :=
  preimage_coe_eq_empty.2 (inter_compl_self s)


@[simp]
theorem preimage_coe_compl' (s : Set α) :
    (fun x : (sᶜ : Set α) => (x : α)) ⁻¹' s = ∅ :=
  preimage_coe_eq_empty.2 (compl_inter_self s)


theorem injective_iff {α β} {f : Option α → β} :
    Injective f ↔ Injective (f ∘ some) ∧ f none ∉ range (f ∘ some) := by
  /-
    α : Type u_1
    β : Type u_2
    f : Option α → β
    ⊢ Iff (Function.Injective f) (And (Function.Injective (Function.comp f Option. …
  -/
  simp only [mem_range, not_exists, (· ∘ ·)]
  refine
    ⟨fun hf => ⟨hf.comp (Option.some_injective _), fun x => hf.ne <| Option.some_ne_none _⟩, ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    f : Option α → β
    ⊢ And (Function.Injective (Function.comp f Option.some)) (∀ (x : α), Not (Eq ( …
  -/
  rintro ⟨h_some, h_none⟩ (_ | a) (_ | b) hab
  /-
    case intro.none.none
    α : Type u_1
    β : Type u_2
    f : Option α → β
    h_some : Function.Injective (Function.comp f Option.some)
    h_none : ∀ (x : α), Not (Eq (f (Option.some x)) (f Option.none))
    hab : Eq (f Option.none) (f Option.none)
    ⊢ Eq Option.none Option.none
  -/
  exacts [rfl, (h_none _ hab.symm).elim, (h_none _ hab).elim, congr_arg some (h_some hab)]
  /-
    🎉 no goals
  -/


theorem range_eq {α β} (f : Option α → β) : range f = insert (f none) (range (f ∘ some)) :=
  Set.ext fun _ => Option.exists.trans <| eq_comm.or Iff.rfl


theorem WithBot.range_eq {α β} (f : WithBot α → β) :
    range f = insert (f ⊥) (range (f ∘ WithBot.some : α → β)) :=
  Option.range_eq f


theorem WithTop.range_eq {α β} (f : WithTop α → β) :
    range f = insert (f ⊤) (range (f ∘ WithBot.some : α → β)) :=
  Option.range_eq f


@[simp]
theorem preimage_injective : Injective (preimage f) ↔ Surjective f := by
  /-
    α : Type u
    β : Type v
    f : α → β
    ⊢ Iff (Function.Injective (Set.preimage f)) (Function.Surjective f)
  -/
  refine ⟨fun h y => ?_, Surjective.preimage_injective⟩
  obtain ⟨x, hx⟩ : (f ⁻¹' {y}).Nonempty := by
    rw [h.nonempty_apply_iff preimage_empty]
    apply singleton_nonempty
  /-
    case intro
    α : Type u
    β : Type v
    f : α → β
    h : Function.Injective (Set.preimage f)
    y : β
    x : α
    hx : Membership.mem (Set.preimage f (Singleton.singleton y)) x
    ⊢ Exists fun a => Eq (f a) y
  -/
  exact ⟨x, hx⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_surjective : Surjective (preimage f) ↔ Injective f := by
  /-
    α : Type u
    β : Type v
    f : α → β
    ⊢ Iff (Function.Surjective (Set.preimage f)) (Function.Injective f)
  -/
  refine ⟨fun h x x' hx => ?_, Injective.preimage_surjective⟩
  /-
    α : Type u
    β : Type v
    f : α → β
    h : Function.Surjective (Set.preimage f)
    x x' : α
    hx : Eq (f x) (f x')
    ⊢ Eq x x'
  -/
  rcases h {x} with ⟨s, hs⟩; have := mem_singleton x
  /-
    case intro
    α : Type u
    β : Type v
    f : α → β
    h : Function.Surjective (Set.preimage f)
    x x' : α
    hx : Eq (f x) (f x')
    s : Set β
    hs : Eq (Set.preimage f s) (Singleton.singleton x)
    this : Membership.mem (Singleton.singleton x) x
    ⊢ Eq x x'
  -/
  rwa [← hs, mem_preimage, hx, ← mem_preimage, hs, mem_singleton_iff, eq_comm] at this
  /-
    🎉 no goals
  -/


@[simp]
theorem image_surjective : Surjective (image f) ↔ Surjective f := by
  /-
    α : Type u
    β : Type v
    f : α → β
    ⊢ Iff (Function.Surjective (Set.image f)) (Function.Surjective f)
  -/
  refine ⟨fun h y => ?_, Surjective.image_surjective⟩
  /-
    α : Type u
    β : Type v
    f : α → β
    h : Function.Surjective (Set.image f)
    y : β
    ⊢ Exists fun a => Eq (f a) y
  -/
  rcases h {y} with ⟨s, hs⟩
  /-
    case intro
    α : Type u
    β : Type v
    f : α → β
    h : Function.Surjective (Set.image f)
    y : β
    s : Set α
    hs : Eq (Set.image f s) (Singleton.singleton y)
    ⊢ Exists fun a => Eq (f a) y
  -/
  have := mem_singleton y; rw [← hs] at this; rcases this with ⟨x, _, hx⟩
  /-
    case intro.intro.intro
    α : Type u
    β : Type v
    f : α → β
    h : Function.Surjective (Set.image f)
    y : β
    s : Set α
    hs : Eq (Set.image f s) (Singleton.singleton y)
    x : α
    left✝ : Membership.mem s x
    hx : Eq (f x) y
    ⊢ Exists fun a => Eq (f a) y
  -/
  exact ⟨x, hx⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem image_injective : Injective (image f) ↔ Injective f := by
  /-
    α : Type u
    β : Type v
    f : α → β
    ⊢ Iff (Function.Injective (Set.image f)) (Function.Injective f)
  -/
  refine ⟨fun h x x' hx => ?_, Injective.image_injective⟩
  /-
    α : Type u
    β : Type v
    f : α → β
    h : Function.Injective (Set.image f)
    x x' : α
    hx : Eq (f x) (f x')
    ⊢ Eq x x'
  -/
  rw [← singleton_eq_singleton_iff]; apply h
  /-
    case a
    α : Type u
    β : Type v
    f : α → β
    h : Function.Injective (Set.image f)
    x x' : α
    hx : Eq (f x) (f x')
    ⊢ Eq (Set.image f (Singleton.singleton x)) (Set.image f (Singleton.singleton x …
  -/
  rw [image_singleton, image_singleton, hx]
  /-
    🎉 no goals
  -/


theorem preimage_eq_iff_eq_image {f : α → β} (hf : Bijective f) {s t} :
                                   /-
                                     α : Type u
                                     β : Type v
                                     f : α → β
                                     hf : Function.Bijective f
                                     s : Set β
                                     t : Set α
                                     ⊢ Iff (Eq (Set.preimage f s) t) (Eq s (Set.image f t))
                                   -/
    f ⁻¹' s = t ↔ s = f '' t := by rw [← image_eq_image hf.1, hf.2.image_preimage]
                                   /-
                                     🎉 no goals
                                   -/


theorem eq_preimage_iff_image_eq {f : α → β} (hf : Bijective f) {s t} :
                                   /-
                                     α : Type u
                                     β : Type v
                                     f : α → β
                                     hf : Function.Bijective f
                                     s : Set α
                                     t : Set β
                                     ⊢ Iff (Eq s (Set.preimage f t)) (Eq (Set.image f s) t)
                                   -/
    s = f ⁻¹' t ↔ f '' s = t := by rw [← image_eq_image hf.1, hf.2.image_preimage]
                                   /-
                                     🎉 no goals
                                   -/


theorem Disjoint.preimage (f : α → β) {s t : Set β} (h : Disjoint s t) :
    Disjoint (f ⁻¹' s) (f ⁻¹' t) :=
  disjoint_iff_inf_le.mpr fun _ hx => h.le_bot hx


lemma Codisjoint.preimage (f : α → β) {s t : Set β} (h : Codisjoint s t) :
    Codisjoint (f ⁻¹' s) (f ⁻¹' t) := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    s t : Set β
    h : Codisjoint s t
    ⊢ Codisjoint (Set.preimage f s) (Set.preimage f t)
  -/
  simp only [codisjoint_iff_le_sup, Set.sup_eq_union, top_le_iff, ← Set.preimage_union] at h ⊢
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    s t : Set β
    h : Eq (Union.union s t) Top.top
    ⊢ Eq (Set.preimage f (Union.union s t)) Top.top
  -/
  rw [h]; rfl
          /-
            🎉 no goals
          -/


lemma IsCompl.preimage (f : α → β) {s t : Set β} (h : IsCompl s t) :
    IsCompl (f ⁻¹' s) (f ⁻¹' t) :=
  ⟨h.1.preimage f, h.2.preimage f⟩


theorem disjoint_image_image {f : β → α} {g : γ → α} {s : Set β} {t : Set γ}
    (h : ∀ b ∈ s, ∀ c ∈ t, f b ≠ g c) : Disjoint (f '' s) (g '' t) :=
                                /-
                                  α : Type u_1
                                  β : Type u_2
                                  γ : Type u_3
                                  f : β → α
                                  g : γ → α
                                  s : Set β
                                  t : Set γ
                                  h : ∀ (b : β), Membership.mem s b → ∀ (c : γ), Membership.mem t c → Ne (f b) ( …
                                  ⊢ LE.le (Min.min (Set.image f s) (Set.image g t)) Bot.bot
                                -/
  disjoint_iff_inf_le.mpr <| by rintro a ⟨⟨b, hb, eq⟩, c, hc, rfl⟩; exact h b hb c hc eq
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem disjoint_image_of_injective (hf : Injective f) {s t : Set α} (hd : Disjoint s t) :
    Disjoint (f '' s) (f '' t) :=
  disjoint_image_image fun _ hx _ hy => hf.ne fun H => Set.disjoint_iff.1 hd ⟨hx, H.symm ▸ hy⟩


theorem _root_.Disjoint.of_image (h : Disjoint (f '' s) (f '' t)) : Disjoint s t :=
  disjoint_iff_inf_le.mpr fun _ hx =>
    disjoint_left.1 h (mem_image_of_mem _ hx.1) (mem_image_of_mem _ hx.2)


@[simp]
theorem disjoint_image_iff (hf : Injective f) : Disjoint (f '' s) (f '' t) ↔ Disjoint s t :=
  ⟨Disjoint.of_image, disjoint_image_of_injective hf⟩


theorem _root_.Disjoint.of_preimage (hf : Surjective f) {s t : Set β}
    (h : Disjoint (f ⁻¹' s) (f ⁻¹' t)) : Disjoint s t := by
  rw [disjoint_iff_inter_eq_empty, ← image_preimage_eq (_ ∩ _) hf, preimage_inter, h.inter_eq,
    image_empty]


@[simp]
theorem disjoint_preimage_iff (hf : Surjective f) {s t : Set β} :
    Disjoint (f ⁻¹' s) (f ⁻¹' t) ↔ Disjoint s t :=
  ⟨Disjoint.of_preimage hf, Disjoint.preimage _⟩


theorem preimage_eq_empty {s : Set β} (h : Disjoint s (range f)) :
    f ⁻¹' s = ∅ := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    s : Set β
    h : Disjoint s (Set.range f)
    ⊢ Eq (Set.preimage f s) EmptyCollection.emptyCollection
  -/
  simpa using h.preimage f
  /-
    🎉 no goals
  -/


theorem preimage_eq_empty_iff {s : Set β} : f ⁻¹' s = ∅ ↔ Disjoint s (range f) :=
  ⟨fun h => by
    simp only [eq_empty_iff_forall_not_mem, disjoint_iff_inter_eq_empty, not_exists, mem_inter_iff,
      not_and, mem_range, mem_preimage] at h ⊢
    /-
      α : Type u_1
      β : Type u_2
      f : α → β
      s : Set β
      h : ∀ (x : α), Not (Membership.mem s (f x))
      ⊢ ∀ (x : β), Membership.mem s x → ∀ (x_1 : α), Not (Eq (f x_1) x)
    -/
    intro y hy x hx
    /-
      α : Type u_1
      β : Type u_2
      f : α → β
      s : Set β
      h : ∀ (x : α), Not (Membership.mem s (f x))
      y : β
      hy : Membership.mem s y
      x : α
      hx : Eq (f x) y
      ⊢ False
    -/
    rw [← hx] at hy
    /-
      α : Type u_1
      β : Type u_2
      f : α → β
      s : Set β
      h : ∀ (x : α), Not (Membership.mem s (f x))
      y : β
      x : α
      hy : Membership.mem s (f x)
      hx : Eq (f x) y
      ⊢ False
    -/
    exact h x hy,
    /-
      🎉 no goals
    -/
  preimage_eq_empty⟩


lemma sigma_mk_preimage_image' (h : i ≠ j) : Sigma.mk j ⁻¹' (Sigma.mk i '' s) = ∅ := by
  /-
    α : Type u_1
    β : α → Type u_2
    i j : α
    s : Set (β i)
    h : Ne i j
    ⊢ Eq (Set.preimage (Sigma.mk j) (Set.image (Sigma.mk i) s)) EmptyCollection.em …
  -/
  simp [image, h]
  /-
    🎉 no goals
  -/


lemma sigma_mk_preimage_image_eq_self : Sigma.mk i ⁻¹' (Sigma.mk i '' s) = s := by
  /-
    α : Type u_1
    β : α → Type u_2
    i : α
    s : Set (β i)
    ⊢ Eq (Set.preimage (Sigma.mk i) (Set.image (Sigma.mk i) s)) s
  -/
  simp [image]
  /-
    🎉 no goals
  -/


