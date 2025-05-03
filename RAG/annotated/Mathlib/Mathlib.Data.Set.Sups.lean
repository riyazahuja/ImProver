/-- Notation typeclass for pointwise supremum `⊻`. -/
class HasSups (α : Type*) where
  /-- The point-wise supremum `a ⊔ b` of `a, b : α`. -/
  sups : α → α → α


/-- Notation typeclass for pointwise infimum `⊼`. -/
class HasInfs (α : Type*) where
  /-- The point-wise infimum `a ⊓ b` of `a, b : α`. -/
  infs : α → α → α

-- This notation is meant to have higher precedence than `⊔` and `⊓`, but still within the
-- realm of other binary notation.

@[inherit_doc]
infixl:74 " ⊻ " => HasSups.sups


@[inherit_doc]
infixl:75 " ⊼ " => HasInfs.infs


/-- `s ⊻ t` is the set of elements of the form `a ⊔ b` where `a ∈ s`, `b ∈ t`. -/
protected def hasSups : HasSups (Set α) :=
  ⟨image2 (· ⊔ ·)⟩


@[simp]
                                                                 /-
                                                                   α : Type u_2
                                                                   inst✝ : SemilatticeSup α
                                                                   s t : Set α
                                                                   c : α
                                                                   ⊢ Iff (Membership.mem (HasSups.sups s t) c) (Exists fun a => And (Membership.m …
                                                                 -/
theorem mem_sups : c ∈ s ⊻ t ↔ ∃ a ∈ s, ∃ b ∈ t, a ⊔ b = c := by simp [(· ⊻ ·)]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


theorem sup_mem_sups : a ∈ s → b ∈ t → a ⊔ b ∈ s ⊻ t :=
  mem_image2_of_mem


theorem sups_subset : s₁ ⊆ s₂ → t₁ ⊆ t₂ → s₁ ⊻ t₁ ⊆ s₂ ⊻ t₂ :=
  image2_subset


theorem sups_subset_left : t₁ ⊆ t₂ → s ⊻ t₁ ⊆ s ⊻ t₂ :=
  image2_subset_left


theorem sups_subset_right : s₁ ⊆ s₂ → s₁ ⊻ t ⊆ s₂ ⊻ t :=
  image2_subset_right


theorem image_subset_sups_left : b ∈ t → (fun a => a ⊔ b) '' s ⊆ s ⊻ t :=
  image_subset_image2_left


theorem image_subset_sups_right : a ∈ s → (· ⊔ ·) a '' t ⊆ s ⊻ t :=
  image_subset_image2_right


theorem forall_sups_iff {p : α → Prop} : (∀ c ∈ s ⊻ t, p c) ↔ ∀ a ∈ s, ∀ b ∈ t, p (a ⊔ b) :=
  forall_mem_image2


@[simp]
theorem sups_subset_iff : s ⊻ t ⊆ u ↔ ∀ a ∈ s, ∀ b ∈ t, a ⊔ b ∈ u :=
  image2_subset_iff


@[simp]
theorem sups_nonempty : (s ⊻ t).Nonempty ↔ s.Nonempty ∧ t.Nonempty :=
  image2_nonempty_iff


protected theorem Nonempty.sups : s.Nonempty → t.Nonempty → (s ⊻ t).Nonempty :=
  Nonempty.image2


theorem Nonempty.of_sups_left : (s ⊻ t).Nonempty → s.Nonempty :=
  Nonempty.of_image2_left


theorem Nonempty.of_sups_right : (s ⊻ t).Nonempty → t.Nonempty :=
  Nonempty.of_image2_right


@[simp]
theorem empty_sups : ∅ ⊻ t = ∅ :=
  image2_empty_left


@[simp]
theorem sups_empty : s ⊻ ∅ = ∅ :=
  image2_empty_right


@[simp]
theorem sups_eq_empty : s ⊻ t = ∅ ↔ s = ∅ ∨ t = ∅ :=
  image2_eq_empty_iff


@[simp]
theorem singleton_sups : {a} ⊻ t = t.image fun b => a ⊔ b :=
  image2_singleton_left


@[simp]
theorem sups_singleton : s ⊻ {b} = s.image fun a => a ⊔ b :=
  image2_singleton_right


theorem singleton_sups_singleton : ({a} ⊻ {b} : Set α) = {a ⊔ b} :=
  image2_singleton


theorem sups_union_left : (s₁ ∪ s₂) ⊻ t = s₁ ⊻ t ∪ s₂ ⊻ t :=
  image2_union_left


theorem sups_union_right : s ⊻ (t₁ ∪ t₂) = s ⊻ t₁ ∪ s ⊻ t₂ :=
  image2_union_right


theorem sups_inter_subset_left : (s₁ ∩ s₂) ⊻ t ⊆ s₁ ⊻ t ∩ s₂ ⊻ t :=
  image2_inter_subset_left


theorem sups_inter_subset_right : s ⊻ (t₁ ∩ t₂) ⊆ s ⊻ t₁ ∩ s ⊻ t₂ :=
  image2_inter_subset_right


lemma image_sups (f : F) (s t : Set α) : f '' (s ⊻ t) = f '' s ⊻ f '' t :=
  image_image2_distrib <| map_sup f


lemma subset_sups_self : s ⊆ s ⊻ s := fun _a ha ↦ mem_sups.2 ⟨_, ha, _, ha, sup_idem _⟩

lemma sups_subset_self : s ⊻ s ⊆ s ↔ SupClosed s := sups_subset_iff


@[simp] lemma sups_eq_self : s ⊻ s = s ↔ SupClosed s :=
  subset_sups_self.le.le_iff_eq.symm.trans sups_subset_self


lemma sep_sups_le (s t : Set α) (a : α) :
                                                                  /-
                                                                    α : Type u_2
                                                                    inst✝ : SemilatticeSup α
                                                                    s t : Set α
                                                                    a : α
                                                                    ⊢ Eq (setOf fun b => And (Membership.mem (HasSups.sups s t) b) (LE.le b a)) (H …
                                                                  -/
    {b ∈ s ⊻ t | b ≤ a} = {b ∈ s | b ≤ a} ⊻ {b ∈ t | b ≤ a} := by ext; aesop
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


theorem iUnion_image_sup_left : ⋃ a ∈ s, (· ⊔ ·) a '' t = s ⊻ t :=
  iUnion_image_left _


theorem iUnion_image_sup_right : ⋃ b ∈ t, (· ⊔ b) '' s = s ⊻ t :=
  iUnion_image_right _


@[simp]
theorem image_sup_prod (s t : Set α) : Set.image2 (· ⊔ ·) s t = s ⊻ t := rfl


theorem sups_assoc : s ⊻ t ⊻ u = s ⊻ (t ⊻ u) := image2_assoc sup_assoc


theorem sups_comm : s ⊻ t = t ⊻ s := image2_comm sup_comm


theorem sups_left_comm : s ⊻ (t ⊻ u) = t ⊻ (s ⊻ u) :=
  image2_left_comm sup_left_comm


theorem sups_right_comm : s ⊻ t ⊻ u = s ⊻ u ⊻ t :=
  image2_right_comm sup_right_comm


theorem sups_sups_sups_comm : s ⊻ t ⊻ (u ⊻ v) = s ⊻ u ⊻ (t ⊻ v) :=
  image2_image2_image2_comm sup_sup_sup_comm


/-- `s ⊼ t` is the set of elements of the form `a ⊓ b` where `a ∈ s`, `b ∈ t`. -/
protected def hasInfs : HasInfs (Set α) :=
  ⟨image2 (· ⊓ ·)⟩


@[simp]
                                                                 /-
                                                                   α : Type u_2
                                                                   inst✝ : SemilatticeInf α
                                                                   s t : Set α
                                                                   c : α
                                                                   ⊢ Iff (Membership.mem (HasInfs.infs s t) c) (Exists fun a => And (Membership.m …
                                                                 -/
theorem mem_infs : c ∈ s ⊼ t ↔ ∃ a ∈ s, ∃ b ∈ t, a ⊓ b = c := by simp [(· ⊼ ·)]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


theorem inf_mem_infs : a ∈ s → b ∈ t → a ⊓ b ∈ s ⊼ t :=
  mem_image2_of_mem


theorem infs_subset : s₁ ⊆ s₂ → t₁ ⊆ t₂ → s₁ ⊼ t₁ ⊆ s₂ ⊼ t₂ :=
  image2_subset


theorem infs_subset_left : t₁ ⊆ t₂ → s ⊼ t₁ ⊆ s ⊼ t₂ :=
  image2_subset_left


theorem infs_subset_right : s₁ ⊆ s₂ → s₁ ⊼ t ⊆ s₂ ⊼ t :=
  image2_subset_right


theorem image_subset_infs_left : b ∈ t → (fun a => a ⊓ b) '' s ⊆ s ⊼ t :=
  image_subset_image2_left


theorem image_subset_infs_right : a ∈ s → (a ⊓ ·) '' t ⊆ s ⊼ t :=
  image_subset_image2_right


theorem forall_infs_iff {p : α → Prop} : (∀ c ∈ s ⊼ t, p c) ↔ ∀ a ∈ s, ∀ b ∈ t, p (a ⊓ b) :=
  forall_mem_image2


@[simp]
theorem infs_subset_iff : s ⊼ t ⊆ u ↔ ∀ a ∈ s, ∀ b ∈ t, a ⊓ b ∈ u :=
  image2_subset_iff


@[simp]
theorem infs_nonempty : (s ⊼ t).Nonempty ↔ s.Nonempty ∧ t.Nonempty :=
  image2_nonempty_iff


protected theorem Nonempty.infs : s.Nonempty → t.Nonempty → (s ⊼ t).Nonempty :=
  Nonempty.image2


theorem Nonempty.of_infs_left : (s ⊼ t).Nonempty → s.Nonempty :=
  Nonempty.of_image2_left


theorem Nonempty.of_infs_right : (s ⊼ t).Nonempty → t.Nonempty :=
  Nonempty.of_image2_right


@[simp]
theorem empty_infs : ∅ ⊼ t = ∅ :=
  image2_empty_left


@[simp]
theorem infs_empty : s ⊼ ∅ = ∅ :=
  image2_empty_right


@[simp]
theorem infs_eq_empty : s ⊼ t = ∅ ↔ s = ∅ ∨ t = ∅ :=
  image2_eq_empty_iff


@[simp]
theorem singleton_infs : {a} ⊼ t = t.image fun b => a ⊓ b :=
  image2_singleton_left


@[simp]
theorem infs_singleton : s ⊼ {b} = s.image fun a => a ⊓ b :=
  image2_singleton_right


theorem singleton_infs_singleton : ({a} ⊼ {b} : Set α) = {a ⊓ b} :=
  image2_singleton


theorem infs_union_left : (s₁ ∪ s₂) ⊼ t = s₁ ⊼ t ∪ s₂ ⊼ t :=
  image2_union_left


theorem infs_union_right : s ⊼ (t₁ ∪ t₂) = s ⊼ t₁ ∪ s ⊼ t₂ :=
  image2_union_right


theorem infs_inter_subset_left : (s₁ ∩ s₂) ⊼ t ⊆ s₁ ⊼ t ∩ s₂ ⊼ t :=
  image2_inter_subset_left


theorem infs_inter_subset_right : s ⊼ (t₁ ∩ t₂) ⊆ s ⊼ t₁ ∩ s ⊼ t₂ :=
  image2_inter_subset_right


lemma image_infs (f : F) (s t : Set α) : f '' (s ⊼ t) = f '' s ⊼ f '' t :=
  image_image2_distrib <| map_inf f


lemma subset_infs_self : s ⊆ s ⊼ s := fun _a ha ↦ mem_infs.2 ⟨_, ha, _, ha, inf_idem _⟩

lemma infs_self_subset : s ⊼ s ⊆ s ↔ InfClosed s := infs_subset_iff


@[simp] lemma infs_self : s ⊼ s = s ↔ InfClosed s :=
  subset_infs_self.le.le_iff_eq.symm.trans infs_self_subset


lemma sep_infs_le (s t : Set α) (a : α) :
                                                                  /-
                                                                    α : Type u_2
                                                                    inst✝ : SemilatticeInf α
                                                                    s t : Set α
                                                                    a : α
                                                                    ⊢ Eq (setOf fun b => And (Membership.mem (HasInfs.infs s t) b) (LE.le a b)) (H …
                                                                  -/
    {b ∈ s ⊼ t | a ≤ b} = {b ∈ s | a ≤ b} ⊼ {b ∈ t | a ≤ b} := by ext; aesop
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


theorem iUnion_image_inf_left : ⋃ a ∈ s, (a ⊓ ·) '' t = s ⊼ t :=
  iUnion_image_left _


theorem iUnion_image_inf_right : ⋃ b ∈ t, (· ⊓ b) '' s = s ⊼ t :=
  iUnion_image_right _


@[simp]
theorem image_inf_prod (s t : Set α) : Set.image2 (fun x x_1 => x ⊓ x_1) s t = s ⊼ t := by
  have : (s ×ˢ t).image (uncurry (· ⊓ ·)) = Set.image2 (fun x x_1 => x ⊓ x_1) s t := by
    simp only [Set.image_uncurry_prod]
  /-
    α : Type u_2
    inst✝ : SemilatticeInf α
    s t : Set α
    this : Eq (Set.image (Function.uncurry fun x1 x2 => Min.min x1 x2) (SProd.spro …
    ⊢ Eq (Set.image2 (fun x x_1 => Min.min x x_1) s t) (HasInfs.infs s t)
  -/
  rw [← this]
  /-
    α : Type u_2
    inst✝ : SemilatticeInf α
    s t : Set α
    this : Eq (Set.image (Function.uncurry fun x1 x2 => Min.min x1 x2) (SProd.spro …
    ⊢ Eq (Set.image (Function.uncurry fun x1 x2 => Min.min x1 x2) (SProd.sprod s t …
  -/
  exact image_uncurry_prod _ _ _
  /-
    🎉 no goals
  -/


theorem infs_assoc : s ⊼ t ⊼ u = s ⊼ (t ⊼ u) := image2_assoc inf_assoc


theorem infs_comm : s ⊼ t = t ⊼ s := image2_comm inf_comm


theorem infs_left_comm : s ⊼ (t ⊼ u) = t ⊼ (s ⊼ u) :=
  image2_left_comm inf_left_comm


theorem infs_right_comm : s ⊼ t ⊼ u = s ⊼ u ⊼ t :=
  image2_right_comm inf_right_comm


theorem infs_infs_infs_comm : s ⊼ t ⊼ (u ⊼ v) = s ⊼ u ⊼ (t ⊼ v) :=
  image2_image2_image2_comm inf_inf_inf_comm


theorem sups_infs_subset_left : s ⊻ t ⊼ u ⊆ (s ⊻ t) ⊼ (s ⊻ u) :=
  image2_distrib_subset_left sup_inf_left


theorem sups_infs_subset_right : t ⊼ u ⊻ s ⊆ (t ⊻ s) ⊼ (u ⊻ s) :=
  image2_distrib_subset_right sup_inf_right


theorem infs_sups_subset_left : s ⊼ (t ⊻ u) ⊆ s ⊼ t ⊻ s ⊼ u :=
  image2_distrib_subset_left inf_sup_left


theorem infs_sups_subset_right : (t ⊻ u) ⊼ s ⊆ t ⊼ s ⊻ u ⊼ s :=
  image2_distrib_subset_right inf_sup_right


@[simp]
theorem upperClosure_sups [SemilatticeSup α] (s t : Set α) :
    upperClosure (s ⊻ t) = upperClosure s ⊔ upperClosure t := by
  /-
    α : Type u_2
    inst✝ : SemilatticeSup α
    s t : Set α
    ⊢ Eq (upperClosure (HasSups.sups s t)) (Max.max (upperClosure s) (upperClosure …
  -/
  ext a
  simp only [SetLike.mem_coe, mem_upperClosure, Set.mem_sups, exists_and_left, exists_prop,
    UpperSet.coe_sup, Set.mem_inter_iff]
  /-
    case a.h
    α : Type u_2
    inst✝ : SemilatticeSup α
    s t : Set α
    a : α
    ⊢ Iff (Exists fun a_1 => And (Exists fun a => And (Membership.mem s a) (Exists …
  -/
  constructor
    /-
      case a.h.mp
      α : Type u_2
      inst✝ : SemilatticeSup α
      s t : Set α
      a : α
      ⊢ (Exists fun a_1 => And (Exists fun a => And (Membership.mem s a) (Exists fun …
    -/
  · rintro ⟨_, ⟨b, hb, c, hc, rfl⟩, ha⟩
    /-
      case a.h.mp.intro.intro.intro.intro.intro.intro
      α : Type u_2
      inst✝ : SemilatticeSup α
      s t : Set α
      a b : α
      hb : Membership.mem s b
      c : α
      hc : Membership.mem t c
      ha : LE.le (Max.max b c) a
      ⊢ And (Exists fun a_1 => And (Membership.mem s a_1) (LE.le a_1 a)) (Exists fun …
    -/
    exact ⟨⟨b, hb, le_sup_left.trans ha⟩, c, hc, le_sup_right.trans ha⟩
    /-
      🎉 no goals
    -/
    /-
      case a.h.mpr
      α : Type u_2
      inst✝ : SemilatticeSup α
      s t : Set α
      a : α
      ⊢ And (Exists fun a_1 => And (Membership.mem s a_1) (LE.le a_1 a)) (Exists fun …
    -/
  · rintro ⟨⟨b, hb, hab⟩, c, hc, hac⟩
    /-
      case a.h.mpr.intro.intro.intro.intro.intro
      α : Type u_2
      inst✝ : SemilatticeSup α
      s t : Set α
      a b : α
      hb : Membership.mem s b
      hab : LE.le b a
      c : α
      hc : Membership.mem t c
      hac : LE.le c a
      ⊢ Exists fun a_1 => And (Exists fun a => And (Membership.mem s a) (Exists fun  …
    -/
    exact ⟨_, ⟨b, hb, c, hc, rfl⟩, sup_le hab hac⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem lowerClosure_infs [SemilatticeInf α] (s t : Set α) :
    lowerClosure (s ⊼ t) = lowerClosure s ⊓ lowerClosure t := by
  /-
    α : Type u_2
    inst✝ : SemilatticeInf α
    s t : Set α
    ⊢ Eq (lowerClosure (HasInfs.infs s t)) (Min.min (lowerClosure s) (lowerClosure …
  -/
  ext a
  simp only [SetLike.mem_coe, mem_lowerClosure, Set.mem_infs, exists_and_left, exists_prop,
    LowerSet.coe_sup, Set.mem_inter_iff]
  /-
    case a.h
    α : Type u_2
    inst✝ : SemilatticeInf α
    s t : Set α
    a : α
    ⊢ Iff (Exists fun a_1 => And (Exists fun a => And (Membership.mem s a) (Exists …
  -/
  constructor
    /-
      case a.h.mp
      α : Type u_2
      inst✝ : SemilatticeInf α
      s t : Set α
      a : α
      ⊢ (Exists fun a_1 => And (Exists fun a => And (Membership.mem s a) (Exists fun …
    -/
  · rintro ⟨_, ⟨b, hb, c, hc, rfl⟩, ha⟩
    /-
      case a.h.mp.intro.intro.intro.intro.intro.intro
      α : Type u_2
      inst✝ : SemilatticeInf α
      s t : Set α
      a b : α
      hb : Membership.mem s b
      c : α
      hc : Membership.mem t c
      ha : LE.le a (Min.min b c)
      ⊢ Membership.mem (Min.min (lowerClosure s) (lowerClosure t)) a
    -/
    exact ⟨⟨b, hb, ha.trans inf_le_left⟩, c, hc, ha.trans inf_le_right⟩
    /-
      🎉 no goals
    -/
    /-
      case a.h.mpr
      α : Type u_2
      inst✝ : SemilatticeInf α
      s t : Set α
      a : α
      ⊢ Membership.mem (Min.min (lowerClosure s) (lowerClosure t)) a → Exists fun a_ …
    -/
  · rintro ⟨⟨b, hb, hab⟩, c, hc, hac⟩
    /-
      case a.h.mpr.intro.intro.intro.intro.intro
      α : Type u_2
      inst✝ : SemilatticeInf α
      s t : Set α
      a b : α
      hb : Membership.mem s b
      hab : LE.le a b
      c : α
      hc : Membership.mem t c
      hac : LE.le a c
      ⊢ Exists fun a_1 => And (Exists fun a => And (Membership.mem s a) (Exists fun  …
    -/
    exact ⟨_, ⟨b, hb, c, hc, rfl⟩, le_inf hab hac⟩
    /-
      🎉 no goals
    -/

