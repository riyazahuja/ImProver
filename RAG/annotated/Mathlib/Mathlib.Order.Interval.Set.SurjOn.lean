theorem surjOn_Ioo_of_monotone_surjective (h_mono : Monotone f) (h_surj : Function.Surjective f)
    (a b : α) : SurjOn f (Ioo a b) (Ioo (f a) (f b)) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : PartialOrder β
    f : α → β
    h_mono : Monotone f
    h_surj : Function.Surjective f
    a b : α
    ⊢ Set.SurjOn f (Set.Ioo a b) (Set.Ioo (f a) (f b))
  -/
  intro p hp
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : PartialOrder β
    f : α → β
    h_mono : Monotone f
    h_surj : Function.Surjective f
    a b : α
    p : β
    hp : Membership.mem (Set.Ioo (f a) (f b)) p
    ⊢ Membership.mem (Set.image f (Set.Ioo a b)) p
  -/
  rcases h_surj p with ⟨x, rfl⟩
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : PartialOrder β
    f : α → β
    h_mono : Monotone f
    h_surj : Function.Surjective f
    a b x : α
    hp : Membership.mem (Set.Ioo (f a) (f b)) (f x)
    ⊢ Membership.mem (Set.image f (Set.Ioo a b)) (f x)
  -/
  refine ⟨x, mem_Ioo.2 ?_, rfl⟩
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : PartialOrder β
    f : α → β
    h_mono : Monotone f
    h_surj : Function.Surjective f
    a b x : α
    hp : Membership.mem (Set.Ioo (f a) (f b)) (f x)
    ⊢ And (LT.lt a x) (LT.lt x b)
  -/
  contrapose! hp
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : PartialOrder β
    f : α → β
    h_mono : Monotone f
    h_surj : Function.Surjective f
    a b x : α
    hp : LT.lt a x → LE.le b x
    ⊢ Not (Membership.mem (Set.Ioo (f a) (f b)) (f x))
  -/
  exact fun h => h.2.not_le (h_mono <| hp <| h_mono.reflect_lt h.1)
  /-
    🎉 no goals
  -/


theorem surjOn_Ico_of_monotone_surjective (h_mono : Monotone f) (h_surj : Function.Surjective f)
    (a b : α) : SurjOn f (Ico a b) (Ico (f a) (f b)) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : PartialOrder β
    f : α → β
    h_mono : Monotone f
    h_surj : Function.Surjective f
    a b : α
    ⊢ Set.SurjOn f (Set.Ico a b) (Set.Ico (f a) (f b))
  -/
  obtain hab | hab := lt_or_le a b
    /-
      case inl
      α : Type u_1
      β : Type u_2
      inst✝¹ : LinearOrder α
      inst✝ : PartialOrder β
      f : α → β
      h_mono : Monotone f
      h_surj : Function.Surjective f
      a b : α
      hab : LT.lt a b
      ⊢ Set.SurjOn f (Set.Ico a b) (Set.Ico (f a) (f b))
    -/
  · intro p hp
    /-
      case inl
      α : Type u_1
      β : Type u_2
      inst✝¹ : LinearOrder α
      inst✝ : PartialOrder β
      f : α → β
      h_mono : Monotone f
      h_surj : Function.Surjective f
      a b : α
      hab : LT.lt a b
      p : β
      hp : Membership.mem (Set.Ico (f a) (f b)) p
      ⊢ Membership.mem (Set.image f (Set.Ico a b)) p
    -/
    rcases eq_left_or_mem_Ioo_of_mem_Ico hp with (rfl | hp')
      /-
        case inl.inl
        α : Type u_1
        β : Type u_2
        inst✝¹ : LinearOrder α
        inst✝ : PartialOrder β
        f : α → β
        h_mono : Monotone f
        h_surj : Function.Surjective f
        a b : α
        hab : LT.lt a b
        hp : Membership.mem (Set.Ico (f a) (f b)) (f a)
        ⊢ Membership.mem (Set.image f (Set.Ico a b)) (f a)
      -/
    · exact mem_image_of_mem f (left_mem_Ico.mpr hab)
      /-
        🎉 no goals
      -/
    · exact image_subset f Ioo_subset_Ico_self <|
        surjOn_Ioo_of_monotone_surjective h_mono h_surj a b hp'
    /-
      case inr
      α : Type u_1
      β : Type u_2
      inst✝¹ : LinearOrder α
      inst✝ : PartialOrder β
      f : α → β
      h_mono : Monotone f
      h_surj : Function.Surjective f
      a b : α
      hab : LE.le b a
      ⊢ Set.SurjOn f (Set.Ico a b) (Set.Ico (f a) (f b))
    -/
  · rw [Ico_eq_empty (h_mono hab).not_lt]
    /-
      case inr
      α : Type u_1
      β : Type u_2
      inst✝¹ : LinearOrder α
      inst✝ : PartialOrder β
      f : α → β
      h_mono : Monotone f
      h_surj : Function.Surjective f
      a b : α
      hab : LE.le b a
      ⊢ Set.SurjOn f (Set.Ico a b) EmptyCollection.emptyCollection
    -/
    exact surjOn_empty f _
    /-
      🎉 no goals
    -/


theorem surjOn_Ioc_of_monotone_surjective (h_mono : Monotone f) (h_surj : Function.Surjective f)
    (a b : α) : SurjOn f (Ioc a b) (Ioc (f a) (f b)) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : PartialOrder β
    f : α → β
    h_mono : Monotone f
    h_surj : Function.Surjective f
    a b : α
    ⊢ Set.SurjOn f (Set.Ioc a b) (Set.Ioc (f a) (f b))
  -/
  simpa using surjOn_Ico_of_monotone_surjective h_mono.dual h_surj (toDual b) (toDual a)
  /-
    🎉 no goals
  -/

-- to see that the hypothesis `a ≤ b` is necessary, consider a constant function

theorem surjOn_Icc_of_monotone_surjective (h_mono : Monotone f) (h_surj : Function.Surjective f)
    {a b : α} (hab : a ≤ b) : SurjOn f (Icc a b) (Icc (f a) (f b)) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : PartialOrder β
    f : α → β
    h_mono : Monotone f
    h_surj : Function.Surjective f
    a b : α
    hab : LE.le a b
    ⊢ Set.SurjOn f (Set.Icc a b) (Set.Icc (f a) (f b))
  -/
  intro p hp
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : PartialOrder β
    f : α → β
    h_mono : Monotone f
    h_surj : Function.Surjective f
    a b : α
    hab : LE.le a b
    p : β
    hp : Membership.mem (Set.Icc (f a) (f b)) p
    ⊢ Membership.mem (Set.image f (Set.Icc a b)) p
  -/
  rcases eq_endpoints_or_mem_Ioo_of_mem_Icc hp with (rfl | rfl | hp')
    /-
      case inl
      α : Type u_1
      β : Type u_2
      inst✝¹ : LinearOrder α
      inst✝ : PartialOrder β
      f : α → β
      h_mono : Monotone f
      h_surj : Function.Surjective f
      a b : α
      hab : LE.le a b
      hp : Membership.mem (Set.Icc (f a) (f b)) (f a)
      ⊢ Membership.mem (Set.image f (Set.Icc a b)) (f a)
    -/
  · exact ⟨a, left_mem_Icc.mpr hab, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      α : Type u_1
      β : Type u_2
      inst✝¹ : LinearOrder α
      inst✝ : PartialOrder β
      f : α → β
      h_mono : Monotone f
      h_surj : Function.Surjective f
      a b : α
      hab : LE.le a b
      hp : Membership.mem (Set.Icc (f a) (f b)) (f b)
      ⊢ Membership.mem (Set.image f (Set.Icc a b)) (f b)
    -/
  · exact ⟨b, right_mem_Icc.mpr hab, rfl⟩
    /-
      🎉 no goals
    -/
  · exact image_subset f Ioo_subset_Icc_self <|
      surjOn_Ioo_of_monotone_surjective h_mono h_surj a b hp'


theorem surjOn_Ioi_of_monotone_surjective (h_mono : Monotone f) (h_surj : Function.Surjective f)
    (a : α) : SurjOn f (Ioi a) (Ioi (f a)) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : PartialOrder β
    f : α → β
    h_mono : Monotone f
    h_surj : Function.Surjective f
    a : α
    ⊢ Set.SurjOn f (Set.Ioi a) (Set.Ioi (f a))
  -/
  rw [← compl_Iic, ← compl_compl (Ioi (f a))]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : PartialOrder β
    f : α → β
    h_mono : Monotone f
    h_surj : Function.Surjective f
    a : α
    ⊢ Set.SurjOn f (HasCompl.compl (Set.Iic a)) (HasCompl.compl (HasCompl.compl (S …
  -/
  refine MapsTo.surjOn_compl ?_ h_surj
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : PartialOrder β
    f : α → β
    h_mono : Monotone f
    h_surj : Function.Surjective f
    a : α
    ⊢ Set.MapsTo f (Set.Iic a) (HasCompl.compl (Set.Ioi (f a)))
  -/
  exact fun x hx => (h_mono hx).not_lt
  /-
    🎉 no goals
  -/


theorem surjOn_Iio_of_monotone_surjective (h_mono : Monotone f) (h_surj : Function.Surjective f)
    (a : α) : SurjOn f (Iio a) (Iio (f a)) :=
  @surjOn_Ioi_of_monotone_surjective _ _ _ _ _ h_mono.dual h_surj a


theorem surjOn_Ici_of_monotone_surjective (h_mono : Monotone f) (h_surj : Function.Surjective f)
    (a : α) : SurjOn f (Ici a) (Ici (f a)) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : PartialOrder β
    f : α → β
    h_mono : Monotone f
    h_surj : Function.Surjective f
    a : α
    ⊢ Set.SurjOn f (Set.Ici a) (Set.Ici (f a))
  -/
  rw [← Ioi_union_left, ← Ioi_union_left]
  exact
    (surjOn_Ioi_of_monotone_surjective h_mono h_surj a).union_union
      (@image_singleton _ _ f a ▸ surjOn_image _ _)


theorem surjOn_Iic_of_monotone_surjective (h_mono : Monotone f) (h_surj : Function.Surjective f)
    (a : α) : SurjOn f (Iic a) (Iic (f a)) :=
  @surjOn_Ici_of_monotone_surjective _ _ _ _ _ h_mono.dual h_surj a

