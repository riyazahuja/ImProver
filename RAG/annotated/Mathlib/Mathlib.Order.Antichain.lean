protected theorem Symmetric.compl (h : Symmetric r) : Symmetric rᶜ := fun _ _ hr hr' =>
  hr <| h hr'


/-- An antichain is a set such that no two distinct elements are related. -/
def IsAntichain (r : α → α → Prop) (s : Set α) : Prop :=
  s.Pairwise rᶜ


protected theorem subset (hs : IsAntichain r s) (h : t ⊆ s) : IsAntichain r t :=
  hs.mono h


theorem mono (hs : IsAntichain r₁ s) (h : r₂ ≤ r₁) : IsAntichain r₂ s :=
  hs.mono' <| compl_le_compl h


theorem mono_on (hs : IsAntichain r₁ s) (h : s.Pairwise fun ⦃a b⦄ => r₂ a b → r₁ a b) :
    IsAntichain r₂ s :=
  hs.imp_on <| h.imp fun _ _ h h₁ h₂ => h₁ <| h h₂


protected theorem eq (hs : IsAntichain r s) {a b : α} (ha : a ∈ s) (hb : b ∈ s) (h : r a b) :
    a = b :=
  Set.Pairwise.eq hs ha hb <| not_not_intro h


protected theorem eq' (hs : IsAntichain r s) {a b : α} (ha : a ∈ s) (hb : b ∈ s) (h : r b a) :
    a = b :=
  (hs.eq hb ha h).symm


protected theorem isAntisymm (h : IsAntichain r univ) : IsAntisymm α r :=
  ⟨fun _ _ ha _ => h.eq trivial trivial ha⟩


protected theorem subsingleton [IsTrichotomous α r] (h : IsAntichain r s) : s.Subsingleton := by
  /-
    α : Type u_1
    r : α → α → Prop
    s : Set α
    inst✝ : IsTrichotomous α r
    h : IsAntichain r s
    ⊢ s.Subsingleton
  -/
  rintro a ha b hb
  /-
    α : Type u_1
    r : α → α → Prop
    s : Set α
    inst✝ : IsTrichotomous α r
    h : IsAntichain r s
    a : α
    ha : Membership.mem s a
    b : α
    hb : Membership.mem s b
    ⊢ Eq a b
  -/
  obtain hab | hab | hab := trichotomous_of r a b
    /-
      case inl
      α : Type u_1
      r : α → α → Prop
      s : Set α
      inst✝ : IsTrichotomous α r
      h : IsAntichain r s
      a : α
      ha : Membership.mem s a
      b : α
      hb : Membership.mem s b
      hab : r a b
      ⊢ Eq a b
    -/
  · exact h.eq ha hb hab
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      α : Type u_1
      r : α → α → Prop
      s : Set α
      inst✝ : IsTrichotomous α r
      h : IsAntichain r s
      a : α
      ha : Membership.mem s a
      b : α
      hb : Membership.mem s b
      hab : Eq a b
      ⊢ Eq a b
    -/
  · exact hab
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      α : Type u_1
      r : α → α → Prop
      s : Set α
      inst✝ : IsTrichotomous α r
      h : IsAntichain r s
      a : α
      ha : Membership.mem s a
      b : α
      hb : Membership.mem s b
      hab : r b a
      ⊢ Eq a b
    -/
  · exact h.eq' ha hb hab
    /-
      🎉 no goals
    -/


protected theorem flip (hs : IsAntichain r s) : IsAntichain (flip r) s := fun _ ha _ hb h =>
  hs hb ha h.symm


theorem swap (hs : IsAntichain r s) : IsAntichain (swap r) s :=
  hs.flip


theorem image (hs : IsAntichain r s) (f : α → β) (h : ∀ ⦃a b⦄, r' (f a) (f b) → r a b) :
    IsAntichain r' (f '' s) := by
  /-
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    r' : β → β → Prop
    s : Set α
    hs : IsAntichain r s
    f : α → β
    h : ∀ ⦃a b : α⦄, r' (f a) (f b) → r a b
    ⊢ IsAntichain r' (Set.image f s)
  -/
  rintro _ ⟨b, hb, rfl⟩ _ ⟨c, hc, rfl⟩ hbc hr
  /-
    case intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    r' : β → β → Prop
    s : Set α
    hs : IsAntichain r s
    f : α → β
    h : ∀ ⦃a b : α⦄, r' (f a) (f b) → r a b
    b : α
    hb : Membership.mem s b
    c : α
    hc : Membership.mem s c
    hbc : Ne (f b) (f c)
    hr : r' (f b) (f c)
    ⊢ False
  -/
  exact hs hb hc (ne_of_apply_ne _ hbc) (h hr)
  /-
    🎉 no goals
  -/


theorem preimage (hs : IsAntichain r s) {f : β → α} (hf : Injective f)
    (h : ∀ ⦃a b⦄, r' a b → r (f a) (f b)) : IsAntichain r' (f ⁻¹' s) := fun _ hb _ hc hbc hr =>
  hs hb hc (hf.ne hbc) <| h hr


theorem _root_.isAntichain_insert :
    IsAntichain r (insert a s) ↔ IsAntichain r s ∧ ∀ ⦃b⦄, b ∈ s → a ≠ b → ¬r a b ∧ ¬r b a :=
  Set.pairwise_insert


protected theorem insert (hs : IsAntichain r s) (hl : ∀ ⦃b⦄, b ∈ s → a ≠ b → ¬r b a)
    (hr : ∀ ⦃b⦄, b ∈ s → a ≠ b → ¬r a b) : IsAntichain r (insert a s) :=
  isAntichain_insert.2 ⟨hs, fun _ hb hab => ⟨hr hb hab, hl hb hab⟩⟩


theorem _root_.isAntichain_insert_of_symmetric (hr : Symmetric r) :
    IsAntichain r (insert a s) ↔ IsAntichain r s ∧ ∀ ⦃b⦄, b ∈ s → a ≠ b → ¬r a b :=
  pairwise_insert_of_symmetric hr.compl


theorem insert_of_symmetric (hs : IsAntichain r s) (hr : Symmetric r)
    (h : ∀ ⦃b⦄, b ∈ s → a ≠ b → ¬r a b) : IsAntichain r (insert a s) :=
  (isAntichain_insert_of_symmetric hr).2 ⟨hs, h⟩


theorem image_relEmbedding (hs : IsAntichain r s) (φ : r ↪r r') : IsAntichain r' (φ '' s) := by
  /-
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    r' : β → β → Prop
    s : Set α
    hs : IsAntichain r s
    φ : RelEmbedding r r'
    ⊢ IsAntichain r' (Set.image (⇑φ) s)
  -/
  intro b hb b' hb' h₁ h₂
  /-
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    r' : β → β → Prop
    s : Set α
    hs : IsAntichain r s
    φ : RelEmbedding r r'
    b : β
    hb : Membership.mem (Set.image (⇑φ) s) b
    b' : β
    hb' : Membership.mem (Set.image (⇑φ) s) b'
    h₁ : Ne b b'
    h₂ : r' b b'
    ⊢ False
  -/
  rw [Set.mem_image] at hb hb'
  /-
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    r' : β → β → Prop
    s : Set α
    hs : IsAntichain r s
    φ : RelEmbedding r r'
    b : β
    hb : Exists fun x => And (Membership.mem s x) (Eq (φ x) b)
    b' : β
    hb' : Exists fun x => And (Membership.mem s x) (Eq (φ x) b')
    h₁ : Ne b b'
    h₂ : r' b b'
    ⊢ False
  -/
  obtain ⟨⟨a, has, rfl⟩, ⟨a', has', rfl⟩⟩ := hb, hb'
  /-
    case intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    r' : β → β → Prop
    s : Set α
    hs : IsAntichain r s
    φ : RelEmbedding r r'
    a : α
    has : Membership.mem s a
    a' : α
    has' : Membership.mem s a'
    h₁ : Ne (φ a) (φ a')
    h₂ : r' (φ a) (φ a')
    ⊢ False
  -/
  exact hs has has' (fun haa' => h₁ (by rw [haa'])) (φ.map_rel_iff.mp h₂)
  /-
    🎉 no goals
  -/


theorem preimage_relEmbedding {t : Set β} (ht : IsAntichain r' t) (φ : r ↪r r') :
    IsAntichain r (φ ⁻¹' t) := fun _ ha _s ha' hne hle =>
  ht ha ha' (fun h => hne (φ.injective h)) (φ.map_rel_iff.mpr hle)


theorem image_relIso (hs : IsAntichain r s) (φ : r ≃r r') : IsAntichain r' (φ '' s) :=
  hs.image_relEmbedding φ.toRelEmbedding


theorem preimage_relIso {t : Set β} (hs : IsAntichain r' t) (φ : r ≃r r') :
    IsAntichain r (φ ⁻¹' t) :=
  hs.preimage_relEmbedding φ.toRelEmbedding


theorem image_relEmbedding_iff {φ : r ↪r r'} : IsAntichain r' (φ '' s) ↔ IsAntichain r s :=
  ⟨fun h => (φ.injective.preimage_image s).subst (h.preimage_relEmbedding φ), fun h =>
    h.image_relEmbedding φ⟩


theorem image_relIso_iff {φ : r ≃r r'} : IsAntichain r' (φ '' s) ↔ IsAntichain r s :=
  @image_relEmbedding_iff _ _ _ _ _ (φ : r ↪r r')


theorem image_embedding [LE α] [LE β] (hs : IsAntichain (· ≤ ·) s) (φ : α ↪o β) :
    IsAntichain (· ≤ ·) (φ '' s) :=
  image_relEmbedding hs _


theorem preimage_embedding [LE α] [LE β] {t : Set β} (ht : IsAntichain (· ≤ ·) t) (φ : α ↪o β) :
    IsAntichain (· ≤ ·) (φ ⁻¹' t) :=
  preimage_relEmbedding ht _


theorem image_embedding_iff [LE α] [LE β] {φ : α ↪o β} :
    IsAntichain (· ≤ ·) (φ '' s) ↔ IsAntichain (· ≤ ·) s :=
  image_relEmbedding_iff


theorem image_iso [LE α] [LE β] (hs : IsAntichain (· ≤ ·) s) (φ : α ≃o β) :
    IsAntichain (· ≤ ·) (φ '' s) :=
  image_relEmbedding hs _


theorem image_iso_iff [LE α] [LE β] {φ : α ≃o β} :
    IsAntichain (· ≤ ·) (φ '' s) ↔ IsAntichain (· ≤ ·) s :=
  image_relEmbedding_iff


theorem preimage_iso [LE α] [LE β] {t : Set β} (ht : IsAntichain (· ≤ ·) t) (φ : α ≃o β) :
    IsAntichain (· ≤ ·) (φ ⁻¹' t) :=
  preimage_relEmbedding ht _


theorem preimage_iso_iff [LE α] [LE β] {t : Set β} {φ : α ≃o β} :
    IsAntichain (· ≤ ·) (φ ⁻¹' t) ↔ IsAntichain (· ≤ ·) t :=
  ⟨fun h => (φ.image_preimage t).subst (h.image_iso φ), fun h => h.preimage_iso _⟩


theorem to_dual [LE α] (hs : IsAntichain (· ≤ ·) s) : @IsAntichain αᵒᵈ (· ≤ ·) s :=
  fun _ ha _ hb hab => hs hb ha hab.symm


theorem to_dual_iff [LE α] : IsAntichain (· ≤ ·) s ↔ @IsAntichain αᵒᵈ (· ≤ ·) s :=
  ⟨to_dual, to_dual⟩


theorem image_compl [BooleanAlgebra α] (hs : IsAntichain (· ≤ ·) s) :
    IsAntichain (· ≤ ·) (compl '' s) :=
  (hs.image_embedding (OrderIso.compl α).toOrderEmbedding).flip


theorem preimage_compl [BooleanAlgebra α] (hs : IsAntichain (· ≤ ·) s) :
    IsAntichain (· ≤ ·) (compl ⁻¹' s) := fun _ ha _ ha' hne hle =>
  hs ha' ha (fun h => hne (compl_inj_iff.mp h.symm)) (compl_le_compl hle)


theorem isAntichain_singleton (a : α) (r : α → α → Prop) : IsAntichain r {a} :=
  pairwise_singleton _ _


theorem Set.Subsingleton.isAntichain (hs : s.Subsingleton) (r : α → α → Prop) : IsAntichain r s :=
  hs.pairwise _


theorem IsAntichain.not_lt (hs : IsAntichain (· ≤ ·) s) (ha : a ∈ s) (hb : b ∈ s) : ¬a < b :=
  fun h => hs ha hb h.ne h.le


theorem isAntichain_and_least_iff : IsAntichain (· ≤ ·) s ∧ IsLeast s a ↔ s = {a} :=
  ⟨fun h => eq_singleton_iff_unique_mem.2 ⟨h.2.1, fun _ hb => h.1.eq' hb h.2.1 (h.2.2 hb)⟩, by
    /-
      α : Type u_1
      s : Set α
      a : α
      inst✝ : Preorder α
      ⊢ Eq s (Singleton.singleton a) → And (IsAntichain (fun x1 x2 => LE.le x1 x2) s …
    -/
    rintro rfl
    /-
      α : Type u_1
      a : α
      inst✝ : Preorder α
      ⊢ And (IsAntichain (fun x1 x2 => LE.le x1 x2) (Singleton.singleton a)) (IsLeas …
    -/
    exact ⟨isAntichain_singleton _ _, isLeast_singleton⟩⟩
    /-
      🎉 no goals
    -/


theorem isAntichain_and_greatest_iff : IsAntichain (· ≤ ·) s ∧ IsGreatest s a ↔ s = {a} :=
  ⟨fun h => eq_singleton_iff_unique_mem.2 ⟨h.2.1, fun _ hb => h.1.eq hb h.2.1 (h.2.2 hb)⟩, by
    /-
      α : Type u_1
      s : Set α
      a : α
      inst✝ : Preorder α
      ⊢ Eq s (Singleton.singleton a) → And (IsAntichain (fun x1 x2 => LE.le x1 x2) s …
    -/
    rintro rfl
    /-
      α : Type u_1
      a : α
      inst✝ : Preorder α
      ⊢ And (IsAntichain (fun x1 x2 => LE.le x1 x2) (Singleton.singleton a)) (IsGrea …
    -/
    exact ⟨isAntichain_singleton _ _, isGreatest_singleton⟩⟩
    /-
      🎉 no goals
    -/


theorem IsAntichain.least_iff (hs : IsAntichain (· ≤ ·) s) : IsLeast s a ↔ s = {a} :=
  (and_iff_right hs).symm.trans isAntichain_and_least_iff


theorem IsAntichain.greatest_iff (hs : IsAntichain (· ≤ ·) s) : IsGreatest s a ↔ s = {a} :=
  (and_iff_right hs).symm.trans isAntichain_and_greatest_iff


theorem IsLeast.antichain_iff (hs : IsLeast s a) : IsAntichain (· ≤ ·) s ↔ s = {a} :=
  (and_iff_left hs).symm.trans isAntichain_and_least_iff


theorem IsGreatest.antichain_iff (hs : IsGreatest s a) : IsAntichain (· ≤ ·) s ↔ s = {a} :=
  (and_iff_left hs).symm.trans isAntichain_and_greatest_iff


theorem IsAntichain.bot_mem_iff [OrderBot α] (hs : IsAntichain (· ≤ ·) s) : ⊥ ∈ s ↔ s = {⊥} :=
  isLeast_bot_iff.symm.trans hs.least_iff


theorem IsAntichain.top_mem_iff [OrderTop α] (hs : IsAntichain (· ≤ ·) s) : ⊤ ∈ s ↔ s = {⊤} :=
  isGreatest_top_iff.symm.trans hs.greatest_iff


theorem isAntichain_iff_forall_not_lt :
    IsAntichain (· ≤ ·) s ↔ ∀ ⦃a⦄, a ∈ s → ∀ ⦃b⦄, b ∈ s → ¬a < b :=
  ⟨fun hs _ ha _ => hs.not_lt ha, fun hs _ ha _ hb h h' => hs ha hb <| h'.lt_of_ne h⟩


/-- A strong (upward) antichain is a set such that no two distinct elements are related to a common
element. -/
def IsStrongAntichain (r : α → α → Prop) (s : Set α) : Prop :=
  s.Pairwise fun a b => ∀ c, ¬r a c ∨ ¬r b c


protected theorem subset (hs : IsStrongAntichain r s) (h : t ⊆ s) : IsStrongAntichain r t :=
  hs.mono h


theorem mono (hs : IsStrongAntichain r₁ s) (h : r₂ ≤ r₁) : IsStrongAntichain r₂ s :=
  hs.mono' fun _ _ hab c => (hab c).imp (compl_le_compl h _ _) (compl_le_compl h _ _)


theorem eq (hs : IsStrongAntichain r s) {a b c : α} (ha : a ∈ s) (hb : b ∈ s) (hac : r a c)
    (hbc : r b c) : a = b :=
  (Set.Pairwise.eq hs ha hb) fun h =>
    False.elim <| (h c).elim (not_not_intro hac) (not_not_intro hbc)


protected theorem isAntichain [IsRefl α r] (h : IsStrongAntichain r s) : IsAntichain r s :=
  h.imp fun _ b hab => (hab b).resolve_right (not_not_intro <| refl _)


protected theorem subsingleton [IsDirected α r] (h : IsStrongAntichain r s) : s.Subsingleton :=
  fun a ha b hb =>
  let ⟨_, hac, hbc⟩ := directed_of r a b
  h.eq ha hb hac hbc


protected theorem flip [IsSymm α r] (hs : IsStrongAntichain r s) : IsStrongAntichain (flip r) s :=
  fun _ ha _ hb h c => (hs ha hb h c).imp (mt <| symm_of r) (mt <| symm_of r)


theorem swap [IsSymm α r] (hs : IsStrongAntichain r s) : IsStrongAntichain (swap r) s :=
  hs.flip


theorem image (hs : IsStrongAntichain r s) {f : α → β} (hf : Surjective f)
    (h : ∀ a b, r' (f a) (f b) → r a b) : IsStrongAntichain r' (f '' s) := by
  /-
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    r' : β → β → Prop
    s : Set α
    hs : IsStrongAntichain r s
    f : α → β
    hf : Function.Surjective f
    h : ∀ (a b : α), r' (f a) (f b) → r a b
    ⊢ IsStrongAntichain r' (Set.image f s)
  -/
  rintro _ ⟨a, ha, rfl⟩ _ ⟨b, hb, rfl⟩ hab c
  /-
    case intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    r' : β → β → Prop
    s : Set α
    hs : IsStrongAntichain r s
    f : α → β
    hf : Function.Surjective f
    h : ∀ (a b : α), r' (f a) (f b) → r a b
    a : α
    ha : Membership.mem s a
    b : α
    hb : Membership.mem s b
    hab : Ne (f a) (f b)
    c : β
    ⊢ Or (Not (r' (f a) c)) (Not (r' (f b) c))
  -/
  obtain ⟨c, rfl⟩ := hf c
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    β : Type u_2
    r : α → α → Prop
    r' : β → β → Prop
    s : Set α
    hs : IsStrongAntichain r s
    f : α → β
    hf : Function.Surjective f
    h : ∀ (a b : α), r' (f a) (f b) → r a b
    a : α
    ha : Membership.mem s a
    b : α
    hb : Membership.mem s b
    hab : Ne (f a) (f b)
    c : α
    ⊢ Or (Not (r' (f a) (f c))) (Not (r' (f b) (f c)))
  -/
  exact (hs ha hb (ne_of_apply_ne _ hab) _).imp (mt <| h _ _) (mt <| h _ _)
  /-
    🎉 no goals
  -/


theorem preimage (hs : IsStrongAntichain r s) {f : β → α} (hf : Injective f)
    (h : ∀ a b, r' a b → r (f a) (f b)) : IsStrongAntichain r' (f ⁻¹' s) := fun _ ha _ hb hab _ =>
  (hs ha hb (hf.ne hab) _).imp (mt <| h _ _) (mt <| h _ _)


theorem _root_.isStrongAntichain_insert :
    IsStrongAntichain r (insert a s) ↔
      IsStrongAntichain r s ∧ ∀ ⦃b⦄, b ∈ s → a ≠ b → ∀ c, ¬r a c ∨ ¬r b c :=
  Set.pairwise_insert_of_symmetric fun _ _ h c => (h c).symm


protected theorem insert (hs : IsStrongAntichain r s)
    (h : ∀ ⦃b⦄, b ∈ s → a ≠ b → ∀ c, ¬r a c ∨ ¬r b c) : IsStrongAntichain r (insert a s) :=
  isStrongAntichain_insert.2 ⟨hs, h⟩


theorem Set.Subsingleton.isStrongAntichain (hs : s.Subsingleton) (r : α → α → Prop) :
    IsStrongAntichain r s :=
  hs.pairwise _


lemma IsAntichain.of_strictMonoOn_antitoneOn (hf : StrictMonoOn f s) (hf' : AntitoneOn f s) :
    IsAntichain (· ≤ ·) s :=
  fun _a ha _b hb hab' hab ↦ (hf ha hb <| hab.lt_of_ne hab').not_le (hf' ha hb hab)


lemma IsAntichain.of_monotoneOn_strictAntiOn (hf : MonotoneOn f s) (hf' : StrictAntiOn f s) :
    IsAntichain (· ≤ ·) s :=
  fun _a ha _b hb hab' hab ↦ (hf ha hb hab).not_lt (hf' ha hb <| hab.lt_of_ne hab')


@[inherit_doc]
local infixl:50 " ≺ " => StrongLT


/-- A weak antichain in `Π i, α i` is a set such that no two distinct elements are strongly less
than each other. -/
def IsWeakAntichain (s : Set (∀ i, α i)) : Prop :=
  IsAntichain (· ≺ ·) s


protected theorem subset (hs : IsWeakAntichain s) : t ⊆ s → IsWeakAntichain t :=
  IsAntichain.subset hs


protected theorem eq (hs : IsWeakAntichain s) : a ∈ s → b ∈ s → a ≺ b → a = b :=
  IsAntichain.eq hs


protected theorem insert (hs : IsWeakAntichain s) :
    (∀ ⦃b⦄, b ∈ s → a ≠ b → ¬b ≺ a) →
      (∀ ⦃b⦄, b ∈ s → a ≠ b → ¬a ≺ b) → IsWeakAntichain (insert a s) :=
  IsAntichain.insert hs


theorem _root_.isWeakAntichain_insert :
    IsWeakAntichain (insert a s) ↔ IsWeakAntichain s ∧ ∀ ⦃b⦄, b ∈ s → a ≠ b → ¬a ≺ b ∧ ¬b ≺ a :=
  isAntichain_insert


protected theorem IsAntichain.isWeakAntichain (hs : IsAntichain (· ≤ ·) s) : IsWeakAntichain s :=
  hs.mono fun _ _ => le_of_strongLT


theorem Set.Subsingleton.isWeakAntichain (hs : s.Subsingleton) : IsWeakAntichain s :=
  hs.isAntichain _


