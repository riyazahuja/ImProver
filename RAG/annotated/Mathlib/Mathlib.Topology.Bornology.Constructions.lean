instance Prod.instBornology : Bornology (α × β) where
  cobounded' := (cobounded α).coprod (cobounded β)
  le_cofinite' :=
    @coprod_cofinite α β ▸ coprod_mono ‹Bornology α›.le_cofinite ‹Bornology β›.le_cofinite


instance Pi.instBornology : Bornology (∀ i, π i) where
  cobounded' := Filter.coprodᵢ fun i => cobounded (π i)
  le_cofinite' := iSup_le fun _ ↦ (comap_mono (Bornology.le_cofinite _)).trans (comap_cofinite_le _)


/-- Inverse image of a bornology. -/
abbrev Bornology.induced {α β : Type*} [Bornology β] (f : α → β) : Bornology α where
  cobounded' := comap f (cobounded β)
  le_cofinite' := (comap_mono (Bornology.le_cofinite β)).trans (comap_cofinite_le _)


instance {p : α → Prop} : Bornology (Subtype p) :=
  Bornology.induced (Subtype.val : Subtype p → α)


theorem cobounded_prod : cobounded (α × β) = (cobounded α).coprod (cobounded β) :=
  rfl


theorem isBounded_image_fst_and_snd {s : Set (α × β)} :
    IsBounded (Prod.fst '' s) ∧ IsBounded (Prod.snd '' s) ↔ IsBounded s :=
  compl_mem_coprod.symm


lemma IsBounded.image_fst {s : Set (α × β)} (hs : IsBounded s) : IsBounded (Prod.fst '' s) :=
  (isBounded_image_fst_and_snd.2 hs).1


lemma IsBounded.image_snd {s : Set (α × β)} (hs : IsBounded s) : IsBounded (Prod.snd '' s) :=
  (isBounded_image_fst_and_snd.2 hs).2


theorem IsBounded.fst_of_prod (h : IsBounded (s ×ˢ t)) (ht : t.Nonempty) : IsBounded s :=
  fst_image_prod s ht ▸ h.image_fst


theorem IsBounded.snd_of_prod (h : IsBounded (s ×ˢ t)) (hs : s.Nonempty) : IsBounded t :=
  snd_image_prod hs t ▸ h.image_snd


theorem IsBounded.prod (hs : IsBounded s) (ht : IsBounded t) : IsBounded (s ×ˢ t) :=
  isBounded_image_fst_and_snd.1
    ⟨hs.subset <| fst_image_prod_subset _ _, ht.subset <| snd_image_prod_subset _ _⟩


theorem isBounded_prod_of_nonempty (hne : Set.Nonempty (s ×ˢ t)) :
    IsBounded (s ×ˢ t) ↔ IsBounded s ∧ IsBounded t :=
  ⟨fun h => ⟨h.fst_of_prod hne.snd, h.snd_of_prod hne.fst⟩, fun h => h.1.prod h.2⟩


theorem isBounded_prod : IsBounded (s ×ˢ t) ↔ s = ∅ ∨ t = ∅ ∨ IsBounded s ∧ IsBounded t := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Bornology α
    inst✝ : Bornology β
    s : Set α
    t : Set β
    ⊢ Iff (Bornology.IsBounded (SProd.sprod s t)) (Or (Eq s EmptyCollection.emptyC …
  -/
  rcases s.eq_empty_or_nonempty with (rfl | hs); · simp
                                                   /-
                                                     🎉 no goals
                                                   -/
  /-
    case inr
    α : Type u_1
    β : Type u_2
    inst✝¹ : Bornology α
    inst✝ : Bornology β
    s : Set α
    t : Set β
    hs : s.Nonempty
    ⊢ Iff (Bornology.IsBounded (SProd.sprod s t)) (Or (Eq s EmptyCollection.emptyC …
  -/
  rcases t.eq_empty_or_nonempty with (rfl | ht); · simp
                                                   /-
                                                     🎉 no goals
                                                   -/
  /-
    case inr.inr
    α : Type u_1
    β : Type u_2
    inst✝¹ : Bornology α
    inst✝ : Bornology β
    s : Set α
    t : Set β
    hs : s.Nonempty
    ht : t.Nonempty
    ⊢ Iff (Bornology.IsBounded (SProd.sprod s t)) (Or (Eq s EmptyCollection.emptyC …
  -/
  simp only [hs.ne_empty, ht.ne_empty, isBounded_prod_of_nonempty (hs.prod ht), false_or]
  /-
    🎉 no goals
  -/


theorem isBounded_prod_self : IsBounded (s ×ˢ s) ↔ IsBounded s := by
  /-
    α : Type u_1
    inst✝ : Bornology α
    s : Set α
    ⊢ Iff (Bornology.IsBounded (SProd.sprod s s)) (Bornology.IsBounded s)
  -/
  rcases s.eq_empty_or_nonempty with (rfl | hs); · simp
                                                   /-
                                                     🎉 no goals
                                                   -/
  /-
    case inr
    α : Type u_1
    inst✝ : Bornology α
    s : Set α
    hs : s.Nonempty
    ⊢ Iff (Bornology.IsBounded (SProd.sprod s s)) (Bornology.IsBounded s)
  -/
  exact (isBounded_prod_of_nonempty (hs.prod hs)).trans and_self_iff
  /-
    🎉 no goals
  -/


theorem cobounded_pi : cobounded (∀ i, π i) = Filter.coprodᵢ fun i => cobounded (π i) :=
  rfl


theorem forall_isBounded_image_eval_iff {s : Set (∀ i, π i)} :
    (∀ i, IsBounded (eval i '' s)) ↔ IsBounded s :=
  compl_mem_coprodᵢ.symm


lemma IsBounded.image_eval {s : Set (∀ i, π i)} (hs : IsBounded s) (i : ι) :
    IsBounded (eval i '' s) :=
  forall_isBounded_image_eval_iff.2 hs i


theorem IsBounded.pi (h : ∀ i, IsBounded (S i)) : IsBounded (pi univ S) :=
  forall_isBounded_image_eval_iff.1 fun i => (h i).subset eval_image_univ_pi_subset


theorem isBounded_pi_of_nonempty (hne : (pi univ S).Nonempty) :
    IsBounded (pi univ S) ↔ ∀ i, IsBounded (S i) :=
  ⟨fun H i => @eval_image_univ_pi _ _ _ i hne ▸ forall_isBounded_image_eval_iff.2 H i, IsBounded.pi⟩


theorem isBounded_pi : IsBounded (pi univ S) ↔ (∃ i, S i = ∅) ∨ ∀ i, IsBounded (S i) := by
  /-
    ι : Type u_3
    π : ι → Type u_4
    inst✝ : (i : ι) → Bornology (π i)
    S : (i : ι) → Set (π i)
    ⊢ Iff (Bornology.IsBounded (Set.univ.pi S)) (Or (Exists fun i => Eq (S i) Empt …
  -/
  by_cases hne : ∃ i, S i = ∅
    /-
      case pos
      ι : Type u_3
      π : ι → Type u_4
      inst✝ : (i : ι) → Bornology (π i)
      S : (i : ι) → Set (π i)
      hne : Exists fun i => Eq (S i) EmptyCollection.emptyCollection
      ⊢ Iff (Bornology.IsBounded (Set.univ.pi S)) (Or (Exists fun i => Eq (S i) Empt …
    -/
  · simp [hne, univ_pi_eq_empty_iff.2 hne]
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u_3
      π : ι → Type u_4
      inst✝ : (i : ι) → Bornology (π i)
      S : (i : ι) → Set (π i)
      hne : Not (Exists fun i => Eq (S i) EmptyCollection.emptyCollection)
      ⊢ Iff (Bornology.IsBounded (Set.univ.pi S)) (Or (Exists fun i => Eq (S i) Empt …
    -/
  · simp only [hne, false_or]
    /-
      case neg
      ι : Type u_3
      π : ι → Type u_4
      inst✝ : (i : ι) → Bornology (π i)
      S : (i : ι) → Set (π i)
      hne : Not (Exists fun i => Eq (S i) EmptyCollection.emptyCollection)
      ⊢ Iff (Bornology.IsBounded (Set.univ.pi S)) (∀ (i : ι), Bornology.IsBounded (S …
    -/
    simp only [not_exists, ← Ne.eq_def, ← nonempty_iff_ne_empty, ← univ_pi_nonempty_iff] at hne
    /-
      case neg
      ι : Type u_3
      π : ι → Type u_4
      inst✝ : (i : ι) → Bornology (π i)
      S : (i : ι) → Set (π i)
      hne : (Set.univ.pi S).Nonempty
      ⊢ Iff (Bornology.IsBounded (Set.univ.pi S)) (∀ (i : ι), Bornology.IsBounded (S …
    -/
    exact isBounded_pi_of_nonempty hne
    /-
      🎉 no goals
    -/


theorem isBounded_induced {α β : Type*} [Bornology β] {f : α → β} {s : Set α} :
    @IsBounded α (Bornology.induced f) s ↔ IsBounded (f '' s) :=
  compl_mem_comap


theorem isBounded_image_subtype_val {p : α → Prop} {s : Set { x // p x }} :
    IsBounded (Subtype.val '' s) ↔ IsBounded s :=
  isBounded_induced.symm


instance [BoundedSpace α] [BoundedSpace β] : BoundedSpace (α × β) := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    π : ι → Type u_4
    inst✝⁴ : Bornology α
    inst✝³ : Bornology β
    inst✝² : (i : ι) → Bornology (π i)
    inst✝¹ : BoundedSpace α
    inst✝ : BoundedSpace β
    ⊢ BoundedSpace (Prod α β)
  -/
  simp [← cobounded_eq_bot_iff, cobounded_prod]
  /-
    🎉 no goals
  -/


instance [∀ i, BoundedSpace (π i)] : BoundedSpace (∀ i, π i) := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    π : ι → Type u_4
    inst✝³ : Bornology α
    inst✝² : Bornology β
    inst✝¹ : (i : ι) → Bornology (π i)
    inst✝ : ∀ (i : ι), BoundedSpace (π i)
    ⊢ BoundedSpace ((i : ι) → π i)
  -/
  simp [← cobounded_eq_bot_iff, cobounded_pi]
  /-
    🎉 no goals
  -/


theorem boundedSpace_induced_iff {α β : Type*} [Bornology β] {f : α → β} :
    @BoundedSpace α (Bornology.induced f) ↔ IsBounded (range f) := by
  /-
    α : Type u_5
    β : Type u_6
    inst✝ : Bornology β
    f : α → β
    ⊢ Iff (BoundedSpace α) (Bornology.IsBounded (Set.range f))
  -/
  rw [← @isBounded_univ _ (Bornology.induced f), isBounded_induced, image_univ]
  /-
    🎉 no goals
  -/
-- Porting note: had to explicitly provided the bornology to `isBounded_univ`.


theorem boundedSpace_subtype_iff {p : α → Prop} :
    BoundedSpace (Subtype p) ↔ IsBounded { x | p x } := by
  /-
    α : Type u_1
    inst✝ : Bornology α
    p : α → Prop
    ⊢ Iff (BoundedSpace (Subtype p)) (Bornology.IsBounded (setOf fun x => p x))
  -/
  rw [boundedSpace_induced_iff, Subtype.range_coe_subtype]
  /-
    🎉 no goals
  -/


theorem boundedSpace_val_set_iff {s : Set α} : BoundedSpace s ↔ IsBounded s :=
  boundedSpace_subtype_iff


alias ⟨_, Bornology.IsBounded.boundedSpace_subtype⟩ := boundedSpace_subtype_iff


alias ⟨_, Bornology.IsBounded.boundedSpace_val⟩ := boundedSpace_val_set_iff


instance [BoundedSpace α] {p : α → Prop} : BoundedSpace (Subtype p) :=
  (IsBounded.all { x | p x }).boundedSpace_subtype


instance : Bornology (Additive α) :=
  ‹Bornology α›


instance : Bornology (Multiplicative α) :=
  ‹Bornology α›


instance [BoundedSpace α] : BoundedSpace (Additive α) :=
  ‹BoundedSpace α›


instance [BoundedSpace α] : BoundedSpace (Multiplicative α) :=
  ‹BoundedSpace α›


instance : Bornology αᵒᵈ :=
  ‹Bornology α›


instance [BoundedSpace α] : BoundedSpace αᵒᵈ :=
  ‹BoundedSpace α›

