theorem mem_upperBounds_image (Hf : MonotoneOn f t) (Hst : s ⊆ t) (Has : a ∈ upperBounds s)
    (Hat : a ∈ t) : f a ∈ upperBounds (f '' s) :=
  forall_mem_image.2 fun _ H => Hf (Hst H) Hat (Has H)


theorem mem_upperBounds_image_self (Hf : MonotoneOn f t) :
    a ∈ upperBounds t → a ∈ t → f a ∈ upperBounds (f '' t) :=
  Hf.mem_upperBounds_image subset_rfl


theorem mem_lowerBounds_image (Hf : MonotoneOn f t) (Hst : s ⊆ t) (Has : a ∈ lowerBounds s)
    (Hat : a ∈ t) : f a ∈ lowerBounds (f '' s) :=
  forall_mem_image.2 fun _ H => Hf Hat (Hst H) (Has H)


theorem mem_lowerBounds_image_self (Hf : MonotoneOn f t) :
    a ∈ lowerBounds t → a ∈ t → f a ∈ lowerBounds (f '' t) :=
  Hf.mem_lowerBounds_image subset_rfl


theorem image_upperBounds_subset_upperBounds_image (Hf : MonotoneOn f t) (Hst : s ⊆ t) :
    f '' (upperBounds s ∩ t) ⊆ upperBounds (f '' s) := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : α → β
    s t : Set α
    Hf : MonotoneOn f t
    Hst : HasSubset.Subset s t
    ⊢ HasSubset.Subset (Set.image f (Inter.inter (upperBounds s) t)) (upperBounds  …
  -/
  rintro _ ⟨a, ha, rfl⟩
  /-
    case intro.intro
    α : Type u
    β : Type v
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : α → β
    s t : Set α
    Hf : MonotoneOn f t
    Hst : HasSubset.Subset s t
    a : α
    ha : Membership.mem (Inter.inter (upperBounds s) t) a
    ⊢ Membership.mem (upperBounds (Set.image f s)) (f a)
  -/
  exact Hf.mem_upperBounds_image Hst ha.1 ha.2
  /-
    🎉 no goals
  -/


theorem image_lowerBounds_subset_lowerBounds_image (Hf : MonotoneOn f t) (Hst : s ⊆ t) :
    f '' (lowerBounds s ∩ t) ⊆ lowerBounds (f '' s) :=
  Hf.dual.image_upperBounds_subset_upperBounds_image Hst


/-- The image under a monotone function on a set `t` of a subset which has an upper bound in `t`
  is bounded above. -/
theorem map_bddAbove (Hf : MonotoneOn f t) (Hst : s ⊆ t) :
    (upperBounds s ∩ t).Nonempty → BddAbove (f '' s) := fun ⟨C, hs, ht⟩ =>
  ⟨f C, Hf.mem_upperBounds_image Hst hs ht⟩


/-- The image under a monotone function on a set `t` of a subset which has a lower bound in `t`
  is bounded below. -/
theorem map_bddBelow (Hf : MonotoneOn f t) (Hst : s ⊆ t) :
    (lowerBounds s ∩ t).Nonempty → BddBelow (f '' s) := fun ⟨C, hs, ht⟩ =>
  ⟨f C, Hf.mem_lowerBounds_image Hst hs ht⟩


/-- A monotone map sends a least element of a set to a least element of its image. -/
theorem map_isLeast (Hf : MonotoneOn f t) (Ha : IsLeast t a) : IsLeast (f '' t) (f a) :=
  ⟨mem_image_of_mem _ Ha.1, Hf.mem_lowerBounds_image_self Ha.2 Ha.1⟩


/-- A monotone map sends a greatest element of a set to a greatest element of its image. -/
theorem map_isGreatest (Hf : MonotoneOn f t) (Ha : IsGreatest t a) : IsGreatest (f '' t) (f a) :=
  ⟨mem_image_of_mem _ Ha.1, Hf.mem_upperBounds_image_self Ha.2 Ha.1⟩


theorem mem_upperBounds_image (Hf : AntitoneOn f t) (Hst : s ⊆ t) (Has : a ∈ lowerBounds s) :
    a ∈ t → f a ∈ upperBounds (f '' s) :=
  Hf.dual_right.mem_lowerBounds_image Hst Has


theorem mem_upperBounds_image_self (Hf : AntitoneOn f t) :
    a ∈ lowerBounds t → a ∈ t → f a ∈ upperBounds (f '' t) :=
  Hf.dual_right.mem_lowerBounds_image_self


theorem mem_lowerBounds_image (Hf : AntitoneOn f t) (Hst : s ⊆ t) :
    a ∈ upperBounds s → a ∈ t → f a ∈ lowerBounds (f '' s) :=
  Hf.dual_right.mem_upperBounds_image Hst


theorem mem_lowerBounds_image_self (Hf : AntitoneOn f t) :
    a ∈ upperBounds t → a ∈ t → f a ∈ lowerBounds (f '' t) :=
  Hf.dual_right.mem_upperBounds_image_self


theorem image_lowerBounds_subset_upperBounds_image (Hf : AntitoneOn f t) (Hst : s ⊆ t) :
    f '' (lowerBounds s ∩ t) ⊆ upperBounds (f '' s) :=
  Hf.dual_right.image_lowerBounds_subset_lowerBounds_image Hst


theorem image_upperBounds_subset_lowerBounds_image (Hf : AntitoneOn f t) (Hst : s ⊆ t) :
    f '' (upperBounds s ∩ t) ⊆ lowerBounds (f '' s) :=
  Hf.dual_right.image_upperBounds_subset_upperBounds_image Hst


/-- The image under an antitone function of a set which is bounded above is bounded below. -/
theorem map_bddAbove (Hf : AntitoneOn f t) (Hst : s ⊆ t) :
    (upperBounds s ∩ t).Nonempty → BddBelow (f '' s) :=
  Hf.dual_right.map_bddAbove Hst


/-- The image under an antitone function of a set which is bounded below is bounded above. -/
theorem map_bddBelow (Hf : AntitoneOn f t) (Hst : s ⊆ t) :
    (lowerBounds s ∩ t).Nonempty → BddAbove (f '' s) :=
  Hf.dual_right.map_bddBelow Hst


/-- An antitone map sends a greatest element of a set to a least element of its image. -/
theorem map_isGreatest (Hf : AntitoneOn f t) : IsGreatest t a → IsLeast (f '' t) (f a) :=
  Hf.dual_right.map_isGreatest


/-- An antitone map sends a least element of a set to a greatest element of its image. -/
theorem map_isLeast (Hf : AntitoneOn f t) : IsLeast t a → IsGreatest (f '' t) (f a) :=
  Hf.dual_right.map_isLeast


theorem mem_upperBounds_image (Ha : a ∈ upperBounds s) : f a ∈ upperBounds (f '' s) :=
  forall_mem_image.2 fun _ H => Hf (Ha H)


theorem mem_lowerBounds_image (Ha : a ∈ lowerBounds s) : f a ∈ lowerBounds (f '' s) :=
  forall_mem_image.2 fun _ H => Hf (Ha H)


theorem image_upperBounds_subset_upperBounds_image :
    f '' upperBounds s ⊆ upperBounds (f '' s) := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : α → β
    Hf : Monotone f
    s : Set α
    ⊢ HasSubset.Subset (Set.image f (upperBounds s)) (upperBounds (Set.image f s))
  -/
  rintro _ ⟨a, ha, rfl⟩
  /-
    case intro.intro
    α : Type u
    β : Type v
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    f : α → β
    Hf : Monotone f
    s : Set α
    a : α
    ha : Membership.mem (upperBounds s) a
    ⊢ Membership.mem (upperBounds (Set.image f s)) (f a)
  -/
  exact Hf.mem_upperBounds_image ha
  /-
    🎉 no goals
  -/


theorem image_lowerBounds_subset_lowerBounds_image : f '' lowerBounds s ⊆ lowerBounds (f '' s) :=
  Hf.dual.image_upperBounds_subset_upperBounds_image


/-- The image under a monotone function of a set which is bounded above is bounded above. See also
`BddAbove.image2`. -/
theorem map_bddAbove : BddAbove s → BddAbove (f '' s)
  | ⟨C, hC⟩ => ⟨f C, Hf.mem_upperBounds_image hC⟩


/-- The image under a monotone function of a set which is bounded below is bounded below. See also
`BddBelow.image2`. -/
theorem map_bddBelow : BddBelow s → BddBelow (f '' s)
  | ⟨C, hC⟩ => ⟨f C, Hf.mem_lowerBounds_image hC⟩


/-- A monotone map sends a least element of a set to a least element of its image. -/
theorem map_isLeast (Ha : IsLeast s a) : IsLeast (f '' s) (f a) :=
  ⟨mem_image_of_mem _ Ha.1, Hf.mem_lowerBounds_image Ha.2⟩


/-- A monotone map sends a greatest element of a set to a greatest element of its image. -/
theorem map_isGreatest (Ha : IsGreatest s a) : IsGreatest (f '' s) (f a) :=
  ⟨mem_image_of_mem _ Ha.1, Hf.mem_upperBounds_image Ha.2⟩


theorem mem_upperBounds_image : a ∈ lowerBounds s → f a ∈ upperBounds (f '' s) :=
  hf.dual_right.mem_lowerBounds_image


theorem mem_lowerBounds_image : a ∈ upperBounds s → f a ∈ lowerBounds (f '' s) :=
  hf.dual_right.mem_upperBounds_image


theorem image_lowerBounds_subset_upperBounds_image : f '' lowerBounds s ⊆ upperBounds (f '' s) :=
  hf.dual_right.image_lowerBounds_subset_lowerBounds_image


theorem image_upperBounds_subset_lowerBounds_image : f '' upperBounds s ⊆ lowerBounds (f '' s) :=
  hf.dual_right.image_upperBounds_subset_upperBounds_image


/-- The image under an antitone function of a set which is bounded above is bounded below. -/
theorem map_bddAbove : BddAbove s → BddBelow (f '' s) :=
  hf.dual_right.map_bddAbove


/-- The image under an antitone function of a set which is bounded below is bounded above. -/
theorem map_bddBelow : BddBelow s → BddAbove (f '' s) :=
  hf.dual_right.map_bddBelow


/-- An antitone map sends a greatest element of a set to a least element of its image. -/
theorem map_isGreatest : IsGreatest s a → IsLeast (f '' s) (f a) :=
  hf.dual_right.map_isGreatest


/-- An antitone map sends a least element of a set to a greatest element of its image. -/
theorem map_isLeast : IsLeast s a → IsGreatest (f '' s) (f a) :=
  hf.dual_right.map_isLeast


lemma StrictMono.mem_upperBounds_image (hf : StrictMono f) :
                                                         /-
                                                           α : Type u
                                                           β : Type v
                                                           inst✝¹ : LinearOrder α
                                                           inst✝ : Preorder β
                                                           f : α → β
                                                           a : α
                                                           s : Set α
                                                           hf : StrictMono f
                                                           ⊢ Iff (Membership.mem (upperBounds (Set.image f s)) (f a)) (Membership.mem (up …
                                                         -/
    f a ∈ upperBounds (f '' s) ↔ a ∈ upperBounds s := by simp [upperBounds, hf.le_iff_le]
                                                         /-
                                                           🎉 no goals
                                                         -/


lemma StrictMono.mem_lowerBounds_image (hf : StrictMono f) :
                                                          /-
                                                            α : Type u
                                                            β : Type v
                                                            inst✝¹ : LinearOrder α
                                                            inst✝ : Preorder β
                                                            f : α → β
                                                            a : α
                                                            s : Set α
                                                            hf : StrictMono f
                                                            ⊢ Iff (Membership.mem (lowerBounds (Set.image f s)) (f a)) (Membership.mem (lo …
                                                          -/
    f a ∈ lowerBounds (f '' s) ↔ a ∈ lowerBounds s :=  by simp [lowerBounds, hf.le_iff_le]
                                                          /-
                                                            🎉 no goals
                                                          -/


lemma StrictMono.map_isLeast (hf : StrictMono f) : IsLeast (f '' s) (f a) ↔ IsLeast s a := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : LinearOrder α
    inst✝ : Preorder β
    f : α → β
    a : α
    s : Set α
    hf : StrictMono f
    ⊢ Iff (IsLeast (Set.image f s) (f a)) (IsLeast s a)
  -/
  simp [IsLeast, hf.injective.eq_iff, hf.mem_lowerBounds_image]
  /-
    🎉 no goals
  -/


lemma StrictMono.map_isGreatest (hf : StrictMono f) :
    IsGreatest (f '' s) (f a) ↔ IsGreatest s a := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : LinearOrder α
    inst✝ : Preorder β
    f : α → β
    a : α
    s : Set α
    hf : StrictMono f
    ⊢ Iff (IsGreatest (Set.image f s) (f a)) (IsGreatest s a)
  -/
  simp [IsGreatest, hf.injective.eq_iff, hf.mem_upperBounds_image]
  /-
    🎉 no goals
  -/


lemma StrictAnti.mem_upperBounds_image (hf : StrictAnti f) :
    f a ∈ upperBounds (f '' s) ↔ a ∈ lowerBounds s := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : LinearOrder α
    inst✝ : Preorder β
    f : α → β
    a : α
    s : Set α
    hf : StrictAnti f
    ⊢ Iff (Membership.mem (upperBounds (Set.image f s)) (f a)) (Membership.mem (lo …
  -/
  simp [upperBounds, lowerBounds, hf.le_iff_le]
  /-
    🎉 no goals
  -/


lemma StrictAnti.mem_lowerBounds_image (hf : StrictAnti f) :
    f a ∈ lowerBounds (f '' s) ↔ a ∈ upperBounds s := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : LinearOrder α
    inst✝ : Preorder β
    f : α → β
    a : α
    s : Set α
    hf : StrictAnti f
    ⊢ Iff (Membership.mem (lowerBounds (Set.image f s)) (f a)) (Membership.mem (up …
  -/
  simp [upperBounds, lowerBounds, hf.le_iff_le]
  /-
    🎉 no goals
  -/


lemma StrictAnti.map_isLeast (hf : StrictAnti f) : IsLeast (f '' s) (f a) ↔ IsGreatest s a := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : LinearOrder α
    inst✝ : Preorder β
    f : α → β
    a : α
    s : Set α
    hf : StrictAnti f
    ⊢ Iff (IsLeast (Set.image f s) (f a)) (IsGreatest s a)
  -/
  simp [IsLeast, IsGreatest, hf.injective.eq_iff, hf.mem_lowerBounds_image]
  /-
    🎉 no goals
  -/


lemma StrictAnti.map_isGreatest (hf : StrictAnti f) : IsGreatest (f '' s) (f a) ↔ IsLeast s a := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : LinearOrder α
    inst✝ : Preorder β
    f : α → β
    a : α
    s : Set α
    hf : StrictAnti f
    ⊢ Iff (IsGreatest (Set.image f s) (f a)) (IsLeast s a)
  -/
  simp [IsLeast, IsGreatest, hf.injective.eq_iff, hf.mem_upperBounds_image]
  /-
    🎉 no goals
  -/


theorem mem_upperBounds_image2 (ha : a ∈ upperBounds s) (hb : b ∈ upperBounds t) :
    f a b ∈ upperBounds (image2 f s t) :=
  forall_mem_image2.2 fun _ hx _ hy => (h₀ _ <| ha hx).trans <| h₁ _ <| hb hy


theorem mem_lowerBounds_image2 (ha : a ∈ lowerBounds s) (hb : b ∈ lowerBounds t) :
    f a b ∈ lowerBounds (image2 f s t) :=
  forall_mem_image2.2 fun _ hx _ hy => (h₀ _ <| ha hx).trans <| h₁ _ <| hb hy


theorem image2_upperBounds_upperBounds_subset :
    image2 f (upperBounds s) (upperBounds t) ⊆ upperBounds (image2 f s t) :=
  image2_subset_iff.2 fun _ ha _ hb ↦ mem_upperBounds_image2 h₀ h₁ ha hb


theorem image2_lowerBounds_lowerBounds_subset :
    image2 f (lowerBounds s) (lowerBounds t) ⊆ lowerBounds (image2 f s t) :=
  image2_subset_iff.2 fun _ ha _ hb ↦ mem_lowerBounds_image2 h₀ h₁ ha hb


/-- See also `Monotone.map_bddAbove`. -/
protected theorem BddAbove.image2 :
    BddAbove s → BddAbove t → BddAbove (image2 f s t) := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    inst✝² : Preorder α
    inst✝¹ : Preorder β
    inst✝ : Preorder γ
    f : α → β → γ
    s : Set α
    t : Set β
    h₀ : ∀ (b : β), Monotone (Function.swap f b)
    h₁ : ∀ (a : α), Monotone (f a)
    ⊢ BddAbove s → BddAbove t → BddAbove (Set.image2 f s t)
  -/
  rintro ⟨a, ha⟩ ⟨b, hb⟩
  /-
    case intro.intro
    α : Type u
    β : Type v
    γ : Type w
    inst✝² : Preorder α
    inst✝¹ : Preorder β
    inst✝ : Preorder γ
    f : α → β → γ
    s : Set α
    t : Set β
    h₀ : ∀ (b : β), Monotone (Function.swap f b)
    h₁ : ∀ (a : α), Monotone (f a)
    a : α
    ha : Membership.mem (upperBounds s) a
    b : β
    hb : Membership.mem (upperBounds t) b
    ⊢ BddAbove (Set.image2 f s t)
  -/
  exact ⟨f a b, mem_upperBounds_image2 h₀ h₁ ha hb⟩
  /-
    🎉 no goals
  -/


/-- See also `Monotone.map_bddBelow`. -/
protected theorem BddBelow.image2 :
    BddBelow s → BddBelow t → BddBelow (image2 f s t) := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    inst✝² : Preorder α
    inst✝¹ : Preorder β
    inst✝ : Preorder γ
    f : α → β → γ
    s : Set α
    t : Set β
    h₀ : ∀ (b : β), Monotone (Function.swap f b)
    h₁ : ∀ (a : α), Monotone (f a)
    ⊢ BddBelow s → BddBelow t → BddBelow (Set.image2 f s t)
  -/
  rintro ⟨a, ha⟩ ⟨b, hb⟩
  /-
    case intro.intro
    α : Type u
    β : Type v
    γ : Type w
    inst✝² : Preorder α
    inst✝¹ : Preorder β
    inst✝ : Preorder γ
    f : α → β → γ
    s : Set α
    t : Set β
    h₀ : ∀ (b : β), Monotone (Function.swap f b)
    h₁ : ∀ (a : α), Monotone (f a)
    a : α
    ha : Membership.mem (lowerBounds s) a
    b : β
    hb : Membership.mem (lowerBounds t) b
    ⊢ BddBelow (Set.image2 f s t)
  -/
  exact ⟨f a b, mem_lowerBounds_image2 h₀ h₁ ha hb⟩
  /-
    🎉 no goals
  -/


protected theorem IsGreatest.image2 (ha : IsGreatest s a) (hb : IsGreatest t b) :
    IsGreatest (image2 f s t) (f a b) :=
  ⟨mem_image2_of_mem ha.1 hb.1, mem_upperBounds_image2 h₀ h₁ ha.2 hb.2⟩


protected theorem IsLeast.image2 (ha : IsLeast s a) (hb : IsLeast t b) :
    IsLeast (image2 f s t) (f a b) :=
  ⟨mem_image2_of_mem ha.1 hb.1, mem_lowerBounds_image2 h₀ h₁ ha.2 hb.2⟩


theorem mem_upperBounds_image2_of_mem_upperBounds_of_mem_lowerBounds (ha : a ∈ upperBounds s)
    (hb : b ∈ lowerBounds t) : f a b ∈ upperBounds (image2 f s t) :=
  forall_mem_image2.2 fun _ hx _ hy => (h₀ _ <| ha hx).trans <| h₁ _ <| hb hy


theorem mem_lowerBounds_image2_of_mem_lowerBounds_of_mem_upperBounds (ha : a ∈ lowerBounds s)
    (hb : b ∈ upperBounds t) : f a b ∈ lowerBounds (image2 f s t) :=
  forall_mem_image2.2 fun _ hx _ hy => (h₀ _ <| ha hx).trans <| h₁ _ <| hb hy


theorem image2_upperBounds_lowerBounds_subset_upperBounds_image2 :
    image2 f (upperBounds s) (lowerBounds t) ⊆ upperBounds (image2 f s t) :=
  image2_subset_iff.2 fun _ ha _ hb ↦
    mem_upperBounds_image2_of_mem_upperBounds_of_mem_lowerBounds h₀ h₁ ha hb


theorem image2_lowerBounds_upperBounds_subset_lowerBounds_image2 :
    image2 f (lowerBounds s) (upperBounds t) ⊆ lowerBounds (image2 f s t) :=
  image2_subset_iff.2 fun _ ha _ hb ↦
    mem_lowerBounds_image2_of_mem_lowerBounds_of_mem_upperBounds h₀ h₁ ha hb


theorem BddAbove.bddAbove_image2_of_bddBelow :
    BddAbove s → BddBelow t → BddAbove (Set.image2 f s t) := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    inst✝² : Preorder α
    inst✝¹ : Preorder β
    inst✝ : Preorder γ
    f : α → β → γ
    s : Set α
    t : Set β
    h₀ : ∀ (b : β), Monotone (Function.swap f b)
    h₁ : ∀ (a : α), Antitone (f a)
    ⊢ BddAbove s → BddBelow t → BddAbove (Set.image2 f s t)
  -/
  rintro ⟨a, ha⟩ ⟨b, hb⟩
  /-
    case intro.intro
    α : Type u
    β : Type v
    γ : Type w
    inst✝² : Preorder α
    inst✝¹ : Preorder β
    inst✝ : Preorder γ
    f : α → β → γ
    s : Set α
    t : Set β
    h₀ : ∀ (b : β), Monotone (Function.swap f b)
    h₁ : ∀ (a : α), Antitone (f a)
    a : α
    ha : Membership.mem (upperBounds s) a
    b : β
    hb : Membership.mem (lowerBounds t) b
    ⊢ BddAbove (Set.image2 f s t)
  -/
  exact ⟨f a b, mem_upperBounds_image2_of_mem_upperBounds_of_mem_lowerBounds h₀ h₁ ha hb⟩
  /-
    🎉 no goals
  -/


theorem BddBelow.bddBelow_image2_of_bddAbove :
    BddBelow s → BddAbove t → BddBelow (Set.image2 f s t) := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    inst✝² : Preorder α
    inst✝¹ : Preorder β
    inst✝ : Preorder γ
    f : α → β → γ
    s : Set α
    t : Set β
    h₀ : ∀ (b : β), Monotone (Function.swap f b)
    h₁ : ∀ (a : α), Antitone (f a)
    ⊢ BddBelow s → BddAbove t → BddBelow (Set.image2 f s t)
  -/
  rintro ⟨a, ha⟩ ⟨b, hb⟩
  /-
    case intro.intro
    α : Type u
    β : Type v
    γ : Type w
    inst✝² : Preorder α
    inst✝¹ : Preorder β
    inst✝ : Preorder γ
    f : α → β → γ
    s : Set α
    t : Set β
    h₀ : ∀ (b : β), Monotone (Function.swap f b)
    h₁ : ∀ (a : α), Antitone (f a)
    a : α
    ha : Membership.mem (lowerBounds s) a
    b : β
    hb : Membership.mem (upperBounds t) b
    ⊢ BddBelow (Set.image2 f s t)
  -/
  exact ⟨f a b, mem_lowerBounds_image2_of_mem_lowerBounds_of_mem_upperBounds h₀ h₁ ha hb⟩
  /-
    🎉 no goals
  -/


theorem IsGreatest.isGreatest_image2_of_isLeast (ha : IsGreatest s a) (hb : IsLeast t b) :
    IsGreatest (Set.image2 f s t) (f a b) :=
  ⟨mem_image2_of_mem ha.1 hb.1,
    mem_upperBounds_image2_of_mem_upperBounds_of_mem_lowerBounds h₀ h₁ ha.2 hb.2⟩


theorem IsLeast.isLeast_image2_of_isGreatest (ha : IsLeast s a) (hb : IsGreatest t b) :
    IsLeast (Set.image2 f s t) (f a b) :=
  ⟨mem_image2_of_mem ha.1 hb.1,
    mem_lowerBounds_image2_of_mem_lowerBounds_of_mem_upperBounds h₀ h₁ ha.2 hb.2⟩


theorem mem_upperBounds_image2_of_mem_lowerBounds (ha : a ∈ lowerBounds s)
    (hb : b ∈ lowerBounds t) : f a b ∈ upperBounds (image2 f s t) :=
  forall_mem_image2.2 fun _ hx _ hy => (h₀ _ <| ha hx).trans <| h₁ _ <| hb hy


theorem mem_lowerBounds_image2_of_mem_upperBounds (ha : a ∈ upperBounds s)
    (hb : b ∈ upperBounds t) : f a b ∈ lowerBounds (image2 f s t) :=
  forall_mem_image2.2 fun _ hx _ hy => (h₀ _ <| ha hx).trans <| h₁ _ <| hb hy


theorem image2_upperBounds_upperBounds_subset_upperBounds_image2 :
    image2 f (lowerBounds s) (lowerBounds t) ⊆ upperBounds (image2 f s t) :=
  image2_subset_iff.2 fun _ ha _ hb ↦
    mem_upperBounds_image2_of_mem_lowerBounds h₀ h₁ ha hb


theorem image2_lowerBounds_lowerBounds_subset_lowerBounds_image2 :
    image2 f (upperBounds s) (upperBounds t) ⊆ lowerBounds (image2 f s t) :=
  image2_subset_iff.2 fun _ ha _ hb ↦
    mem_lowerBounds_image2_of_mem_upperBounds h₀ h₁ ha hb


theorem BddBelow.image2_bddAbove : BddBelow s → BddBelow t → BddAbove (Set.image2 f s t) := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    inst✝² : Preorder α
    inst✝¹ : Preorder β
    inst✝ : Preorder γ
    f : α → β → γ
    s : Set α
    t : Set β
    h₀ : ∀ (b : β), Antitone (Function.swap f b)
    h₁ : ∀ (a : α), Antitone (f a)
    ⊢ BddBelow s → BddBelow t → BddAbove (Set.image2 f s t)
  -/
  rintro ⟨a, ha⟩ ⟨b, hb⟩
  /-
    case intro.intro
    α : Type u
    β : Type v
    γ : Type w
    inst✝² : Preorder α
    inst✝¹ : Preorder β
    inst✝ : Preorder γ
    f : α → β → γ
    s : Set α
    t : Set β
    h₀ : ∀ (b : β), Antitone (Function.swap f b)
    h₁ : ∀ (a : α), Antitone (f a)
    a : α
    ha : Membership.mem (lowerBounds s) a
    b : β
    hb : Membership.mem (lowerBounds t) b
    ⊢ BddAbove (Set.image2 f s t)
  -/
  exact ⟨f a b, mem_upperBounds_image2_of_mem_lowerBounds h₀ h₁ ha hb⟩
  /-
    🎉 no goals
  -/


theorem BddAbove.image2_bddBelow : BddAbove s → BddAbove t → BddBelow (Set.image2 f s t) := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    inst✝² : Preorder α
    inst✝¹ : Preorder β
    inst✝ : Preorder γ
    f : α → β → γ
    s : Set α
    t : Set β
    h₀ : ∀ (b : β), Antitone (Function.swap f b)
    h₁ : ∀ (a : α), Antitone (f a)
    ⊢ BddAbove s → BddAbove t → BddBelow (Set.image2 f s t)
  -/
  rintro ⟨a, ha⟩ ⟨b, hb⟩
  /-
    case intro.intro
    α : Type u
    β : Type v
    γ : Type w
    inst✝² : Preorder α
    inst✝¹ : Preorder β
    inst✝ : Preorder γ
    f : α → β → γ
    s : Set α
    t : Set β
    h₀ : ∀ (b : β), Antitone (Function.swap f b)
    h₁ : ∀ (a : α), Antitone (f a)
    a : α
    ha : Membership.mem (upperBounds s) a
    b : β
    hb : Membership.mem (upperBounds t) b
    ⊢ BddBelow (Set.image2 f s t)
  -/
  exact ⟨f a b, mem_lowerBounds_image2_of_mem_upperBounds h₀ h₁ ha hb⟩
  /-
    🎉 no goals
  -/


theorem IsLeast.isGreatest_image2 (ha : IsLeast s a) (hb : IsLeast t b) :
    IsGreatest (Set.image2 f s t) (f a b) :=
  ⟨mem_image2_of_mem ha.1 hb.1, mem_upperBounds_image2_of_mem_lowerBounds h₀ h₁ ha.2 hb.2⟩


theorem IsGreatest.isLeast_image2 (ha : IsGreatest s a) (hb : IsGreatest t b) :
    IsLeast (Set.image2 f s t) (f a b) :=
  ⟨mem_image2_of_mem ha.1 hb.1, mem_lowerBounds_image2_of_mem_upperBounds h₀ h₁ ha.2 hb.2⟩


theorem mem_upperBounds_image2_of_mem_upperBounds_of_mem_upperBounds (ha : a ∈ lowerBounds s)
    (hb : b ∈ upperBounds t) : f a b ∈ upperBounds (image2 f s t) :=
  forall_mem_image2.2 fun _ hx _ hy => (h₀ _ <| ha hx).trans <| h₁ _ <| hb hy


theorem mem_lowerBounds_image2_of_mem_lowerBounds_of_mem_lowerBounds (ha : a ∈ upperBounds s)
    (hb : b ∈ lowerBounds t) : f a b ∈ lowerBounds (image2 f s t) :=
  forall_mem_image2.2 fun _ hx _ hy => (h₀ _ <| ha hx).trans <| h₁ _ <| hb hy


theorem image2_lowerBounds_upperBounds_subset_upperBounds_image2 :
    image2 f (lowerBounds s) (upperBounds t) ⊆ upperBounds (image2 f s t) :=
  image2_subset_iff.2 fun _ ha _ hb ↦
    mem_upperBounds_image2_of_mem_upperBounds_of_mem_upperBounds h₀ h₁ ha hb


theorem image2_upperBounds_lowerBounds_subset_lowerBounds_image2 :
    image2 f (upperBounds s) (lowerBounds t) ⊆ lowerBounds (image2 f s t) :=
  image2_subset_iff.2 fun _ ha _ hb ↦
    mem_lowerBounds_image2_of_mem_lowerBounds_of_mem_lowerBounds h₀ h₁ ha hb


theorem BddBelow.bddAbove_image2_of_bddAbove :
    BddBelow s → BddAbove t → BddAbove (Set.image2 f s t) := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    inst✝² : Preorder α
    inst✝¹ : Preorder β
    inst✝ : Preorder γ
    f : α → β → γ
    s : Set α
    t : Set β
    h₀ : ∀ (b : β), Antitone (Function.swap f b)
    h₁ : ∀ (a : α), Monotone (f a)
    ⊢ BddBelow s → BddAbove t → BddAbove (Set.image2 f s t)
  -/
  rintro ⟨a, ha⟩ ⟨b, hb⟩
  /-
    case intro.intro
    α : Type u
    β : Type v
    γ : Type w
    inst✝² : Preorder α
    inst✝¹ : Preorder β
    inst✝ : Preorder γ
    f : α → β → γ
    s : Set α
    t : Set β
    h₀ : ∀ (b : β), Antitone (Function.swap f b)
    h₁ : ∀ (a : α), Monotone (f a)
    a : α
    ha : Membership.mem (lowerBounds s) a
    b : β
    hb : Membership.mem (upperBounds t) b
    ⊢ BddAbove (Set.image2 f s t)
  -/
  exact ⟨f a b, mem_upperBounds_image2_of_mem_upperBounds_of_mem_upperBounds h₀ h₁ ha hb⟩
  /-
    🎉 no goals
  -/


theorem BddAbove.bddBelow_image2_of_bddAbove :
    BddAbove s → BddBelow t → BddBelow (Set.image2 f s t) := by
  /-
    α : Type u
    β : Type v
    γ : Type w
    inst✝² : Preorder α
    inst✝¹ : Preorder β
    inst✝ : Preorder γ
    f : α → β → γ
    s : Set α
    t : Set β
    h₀ : ∀ (b : β), Antitone (Function.swap f b)
    h₁ : ∀ (a : α), Monotone (f a)
    ⊢ BddAbove s → BddBelow t → BddBelow (Set.image2 f s t)
  -/
  rintro ⟨a, ha⟩ ⟨b, hb⟩
  /-
    case intro.intro
    α : Type u
    β : Type v
    γ : Type w
    inst✝² : Preorder α
    inst✝¹ : Preorder β
    inst✝ : Preorder γ
    f : α → β → γ
    s : Set α
    t : Set β
    h₀ : ∀ (b : β), Antitone (Function.swap f b)
    h₁ : ∀ (a : α), Monotone (f a)
    a : α
    ha : Membership.mem (upperBounds s) a
    b : β
    hb : Membership.mem (lowerBounds t) b
    ⊢ BddBelow (Set.image2 f s t)
  -/
  exact ⟨f a b, mem_lowerBounds_image2_of_mem_lowerBounds_of_mem_lowerBounds h₀ h₁ ha hb⟩
  /-
    🎉 no goals
  -/


theorem IsLeast.isGreatest_image2_of_isGreatest (ha : IsLeast s a) (hb : IsGreatest t b) :
    IsGreatest (Set.image2 f s t) (f a b) :=
  ⟨mem_image2_of_mem ha.1 hb.1,
    mem_upperBounds_image2_of_mem_upperBounds_of_mem_upperBounds h₀ h₁ ha.2 hb.2⟩


theorem IsGreatest.isLeast_image2_of_isLeast (ha : IsGreatest s a) (hb : IsLeast t b) :
    IsLeast (Set.image2 f s t) (f a b) :=
  ⟨mem_image2_of_mem ha.1 hb.1,
    mem_lowerBounds_image2_of_mem_lowerBounds_of_mem_lowerBounds h₀ h₁ ha.2 hb.2⟩


lemma bddAbove_prod {s : Set (α × β)} :
    BddAbove s ↔ BddAbove (Prod.fst '' s) ∧ BddAbove (Prod.snd '' s) :=
  ⟨fun ⟨p, hp⟩ ↦ ⟨⟨p.1, forall_mem_image.2 fun _q hq ↦ (hp hq).1⟩,
    ⟨p.2, forall_mem_image.2 fun _q hq ↦ (hp hq).2⟩⟩,
    fun ⟨⟨x, hx⟩, ⟨y, hy⟩⟩ ↦ ⟨⟨x, y⟩, fun _p hp ↦
      ⟨hx <| mem_image_of_mem _ hp, hy <| mem_image_of_mem _ hp⟩⟩⟩


lemma bddBelow_prod {s : Set (α × β)} :
    BddBelow s ↔ BddBelow (Prod.fst '' s) ∧ BddBelow (Prod.snd '' s) :=
  bddAbove_prod (α := αᵒᵈ) (β := βᵒᵈ)


lemma bddAbove_range_prod {F : ι → α × β} :
    BddAbove (range F) ↔ BddAbove (range <| Prod.fst ∘ F) ∧ BddAbove (range <| Prod.snd ∘ F) := by
  /-
    ι : Sort x
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    F : ι → Prod α β
    ⊢ Iff (BddAbove (Set.range F)) (And (BddAbove (Set.range (Function.comp Prod.f …
  -/
  simp only [bddAbove_prod, ← range_comp]
  /-
    🎉 no goals
  -/


lemma bddBelow_range_prod {F : ι → α × β} :
    BddBelow (range F) ↔ BddBelow (range <| Prod.fst ∘ F) ∧ BddBelow (range <| Prod.snd ∘ F) :=
  bddAbove_range_prod (α := αᵒᵈ) (β := βᵒᵈ)


theorem isLUB_prod {s : Set (α × β)} (p : α × β) :
    IsLUB s p ↔ IsLUB (Prod.fst '' s) p.1 ∧ IsLUB (Prod.snd '' s) p.2 := by
  refine
    ⟨fun H =>
      ⟨⟨monotone_fst.mem_upperBounds_image H.1, fun a ha => ?_⟩,
        ⟨monotone_snd.mem_upperBounds_image H.1, fun a ha => ?_⟩⟩,
      fun H => ⟨?_, ?_⟩⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      inst✝¹ : Preorder α
      inst✝ : Preorder β
      s : Set (Prod α β)
      p : Prod α β
      H : IsLUB s p
      a : α
      ha : Membership.mem (upperBounds (Set.image Prod.fst s)) a
      ⊢ LE.le p.1 a
    -/
  · suffices h : (a, p.2) ∈ upperBounds s from (H.2 h).1
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      inst✝¹ : Preorder α
      inst✝ : Preorder β
      s : Set (Prod α β)
      p : Prod α β
      H : IsLUB s p
      a : α
      ha : Membership.mem (upperBounds (Set.image Prod.fst s)) a
      ⊢ Membership.mem (upperBounds s) { fst := a, snd := p.2 }
    -/
    exact fun q hq => ⟨ha <| mem_image_of_mem _ hq, (H.1 hq).2⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      inst✝¹ : Preorder α
      inst✝ : Preorder β
      s : Set (Prod α β)
      p : Prod α β
      H : IsLUB s p
      a : β
      ha : Membership.mem (upperBounds (Set.image Prod.snd s)) a
      ⊢ LE.le p.2 a
    -/
  · suffices h : (p.1, a) ∈ upperBounds s from (H.2 h).2
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      inst✝¹ : Preorder α
      inst✝ : Preorder β
      s : Set (Prod α β)
      p : Prod α β
      H : IsLUB s p
      a : β
      ha : Membership.mem (upperBounds (Set.image Prod.snd s)) a
      ⊢ Membership.mem (upperBounds s) { fst := p.1, snd := a }
    -/
    exact fun q hq => ⟨(H.1 hq).1, ha <| mem_image_of_mem _ hq⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α : Type u_1
      β : Type u_2
      inst✝¹ : Preorder α
      inst✝ : Preorder β
      s : Set (Prod α β)
      p : Prod α β
      H : And (IsLUB (Set.image Prod.fst s) p.1) (IsLUB (Set.image Prod.snd s) p.2)
      ⊢ Membership.mem (upperBounds s) p
    -/
  · exact fun q hq => ⟨H.1.1 <| mem_image_of_mem _ hq, H.2.1 <| mem_image_of_mem _ hq⟩
    /-
      🎉 no goals
    -/
  · exact fun q hq =>
      ⟨H.1.2 <| monotone_fst.mem_upperBounds_image hq,
        H.2.2 <| monotone_snd.mem_upperBounds_image hq⟩


theorem isGLB_prod {s : Set (α × β)} (p : α × β) :
    IsGLB s p ↔ IsGLB (Prod.fst '' s) p.1 ∧ IsGLB (Prod.snd '' s) p.2 :=
  @isLUB_prod αᵒᵈ βᵒᵈ _ _ _ _


lemma bddAbove_pi {s : Set (∀ a, π a)} :
    BddAbove s ↔ ∀ a, BddAbove (Function.eval a '' s) :=
  ⟨fun ⟨f, hf⟩ a ↦ ⟨f a, forall_mem_image.2 fun _ hg ↦ hf hg a⟩,
    fun h ↦ ⟨fun a ↦ (h a).some, fun _ hg a ↦ (h a).some_mem <| mem_image_of_mem _ hg⟩⟩


lemma bddBelow_pi {s : Set (∀ a, π a)} :
    BddBelow s ↔ ∀ a, BddBelow (Function.eval a '' s) :=
  bddAbove_pi (π := fun a ↦ (π a)ᵒᵈ)


lemma bddAbove_range_pi {F : ι → ∀ a, π a} :
    BddAbove (range F) ↔ ∀ a, BddAbove (range fun i ↦ F i a) := by
  /-
    α : Type u
    ι : Sort x
    π : α → Type u_1
    inst✝ : (a : α) → Preorder (π a)
    F : ι → (a : α) → π a
    ⊢ Iff (BddAbove (Set.range F)) (∀ (a : α), BddAbove (Set.range fun i => F i a))
  -/
  simp only [bddAbove_pi, ← range_comp]
  /-
    α : Type u
    ι : Sort x
    π : α → Type u_1
    inst✝ : (a : α) → Preorder (π a)
    F : ι → (a : α) → π a
    ⊢ Iff (∀ (a : α), BddAbove (Set.range (Function.comp (Function.eval a) F))) (∀ …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma bddBelow_range_pi {F : ι → ∀ a, π a} :
    BddBelow (range F) ↔ ∀ a, BddBelow (range fun i ↦ F i a) :=
  bddAbove_range_pi (π := fun a ↦ (π a)ᵒᵈ)


theorem isLUB_pi {s : Set (∀ a, π a)} {f : ∀ a, π a} :
    IsLUB s f ↔ ∀ a, IsLUB (Function.eval a '' s) (f a) := by
  classical
    refine
      ⟨fun H a => ⟨(Function.monotone_eval a).mem_upperBounds_image H.1, fun b hb => ?_⟩, fun H =>
        ⟨?_, ?_⟩⟩
    · suffices h : Function.update f a b ∈ upperBounds s from Function.update_self a b f ▸ H.2 h a
      exact fun g hg => le_update_iff.2 ⟨hb <| mem_image_of_mem _ hg, fun i _ => H.1 hg i⟩
    · exact fun g hg a => (H a).1 (mem_image_of_mem _ hg)
    · exact fun g hg a => (H a).2 ((Function.monotone_eval a).mem_upperBounds_image hg)


theorem isGLB_pi {s : Set (∀ a, π a)} {f : ∀ a, π a} :
    IsGLB s f ↔ ∀ a, IsGLB (Function.eval a '' s) (f a) :=
  @isLUB_pi α (fun a => (π a)ᵒᵈ) _ s f


theorem IsGLB.of_image [Preorder α] [Preorder β] {f : α → β} (hf : ∀ {x y}, f x ≤ f y ↔ x ≤ y)
    {s : Set α} {x : α} (hx : IsGLB (f '' s) (f x)) : IsGLB s x :=
  ⟨fun _ hy => hf.1 <| hx.1 <| mem_image_of_mem _ hy, fun _ hy =>
    hf.1 <| hx.2 <| Monotone.mem_lowerBounds_image (fun _ _ => hf.2) hy⟩


theorem IsLUB.of_image [Preorder α] [Preorder β] {f : α → β} (hf : ∀ {x y}, f x ≤ f y ↔ x ≤ y)
    {s : Set α} {x : α} (hx : IsLUB (f '' s) (f x)) : IsLUB s x :=
  ⟨fun _ hy => hf.1 <| hx.1 <| mem_image_of_mem _ hy, fun _ hy =>
    hf.1 <| hx.2 <| Monotone.mem_upperBounds_image (fun _ _ => hf.2) hy⟩


lemma BddAbove.range_mono [Preorder β] {f : α → β} (g : α → β) (h : ∀ a, f a ≤ g a)
    (hbdd : BddAbove (range g)) : BddAbove (range f) := by
  /-
    α : Type u
    β : Type v
    inst✝ : Preorder β
    f g : α → β
    h : ∀ (a : α), LE.le (f a) (g a)
    hbdd : BddAbove (Set.range g)
    ⊢ BddAbove (Set.range f)
  -/
  obtain ⟨C, hC⟩ := hbdd
  /-
    case intro
    α : Type u
    β : Type v
    inst✝ : Preorder β
    f g : α → β
    h : ∀ (a : α), LE.le (f a) (g a)
    C : β
    hC : Membership.mem (upperBounds (Set.range g)) C
    ⊢ BddAbove (Set.range f)
  -/
  use C
  /-
    case h
    α : Type u
    β : Type v
    inst✝ : Preorder β
    f g : α → β
    h : ∀ (a : α), LE.le (f a) (g a)
    C : β
    hC : Membership.mem (upperBounds (Set.range g)) C
    ⊢ Membership.mem (upperBounds (Set.range f)) C
  -/
  rintro - ⟨x, rfl⟩
  /-
    case h.intro
    α : Type u
    β : Type v
    inst✝ : Preorder β
    f g : α → β
    h : ∀ (a : α), LE.le (f a) (g a)
    C : β
    hC : Membership.mem (upperBounds (Set.range g)) C
    x : α
    ⊢ LE.le (f x) C
  -/
  exact (h x).trans (hC <| mem_range_self x)
  /-
    🎉 no goals
  -/


lemma BddBelow.range_mono [Preorder β] (f : α → β) {g : α → β} (h : ∀ a, f a ≤ g a)
    (hbdd : BddBelow (range f)) : BddBelow (range g) :=
  BddAbove.range_mono (β := βᵒᵈ) f h hbdd


lemma BddAbove.range_comp {γ : Type*} [Preorder β] [Preorder γ] {f : α → β} {g : β → γ}
    (hf : BddAbove (range f)) (hg : Monotone g) : BddAbove (range (fun x => g (f x))) := by
  /-
    α : Type u
    β : Type v
    γ : Type u_1
    inst✝¹ : Preorder β
    inst✝ : Preorder γ
    f : α → β
    g : β → γ
    hf : BddAbove (Set.range f)
    hg : Monotone g
    ⊢ BddAbove (Set.range fun x => g (f x))
  -/
  change BddAbove (range (g ∘ f))
  /-
    α : Type u
    β : Type v
    γ : Type u_1
    inst✝¹ : Preorder β
    inst✝ : Preorder γ
    f : α → β
    g : β → γ
    hf : BddAbove (Set.range f)
    hg : Monotone g
    ⊢ BddAbove (Set.range (Function.comp g f))
  -/
  simpa only [Set.range_comp] using hg.map_bddAbove hf
  /-
    🎉 no goals
  -/


lemma BddBelow.range_comp {γ : Type*} [Preorder β] [Preorder γ] {f : α → β} {g : β → γ}
    (hf : BddBelow (range f)) (hg : Monotone g) : BddBelow (range (fun x => g (f x))) := by
  /-
    α : Type u
    β : Type v
    γ : Type u_1
    inst✝¹ : Preorder β
    inst✝ : Preorder γ
    f : α → β
    g : β → γ
    hf : BddBelow (Set.range f)
    hg : Monotone g
    ⊢ BddBelow (Set.range fun x => g (f x))
  -/
  change BddBelow (range (g ∘ f))
  /-
    α : Type u
    β : Type v
    γ : Type u_1
    inst✝¹ : Preorder β
    inst✝ : Preorder γ
    f : α → β
    g : β → γ
    hf : BddBelow (Set.range f)
    hg : Monotone g
    ⊢ BddBelow (Set.range (Function.comp g f))
  -/
  simpa only [Set.range_comp] using hg.map_bddBelow hf
  /-
    🎉 no goals
  -/

