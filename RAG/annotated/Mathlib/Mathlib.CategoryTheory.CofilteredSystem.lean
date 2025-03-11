/-- This bootstraps `nonempty_sections_of_finite_inverse_system`. In this version,
the `F` functor is between categories of the same universe, and it is an easy
corollary to `TopCat.nonempty_limitCone_of_compact_t2_cofiltered_system`. -/
theorem nonempty_sections_of_finite_cofiltered_system.init {J : Type u} [SmallCategory J]
    [IsCofilteredOrEmpty J] (F : J ⥤ Type u) [hf : ∀ j, Finite (F.obj j)]
    [hne : ∀ j, Nonempty (F.obj j)] : F.sections.Nonempty := by
  /-
    J : Type u
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.IsCofilteredOrEmpty J
    F : CategoryTheory.Functor J (Type u)
    hf : ∀ (j : J), Finite (F.obj j)
    hne : ∀ (j : J), Nonempty (F.obj j)
    ⊢ F.sections.Nonempty
  -/
  let F' : J ⥤ TopCat := F ⋙ TopCat.discrete
  /-
    J : Type u
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.IsCofilteredOrEmpty J
    F : CategoryTheory.Functor J (Type u)
    hf : ∀ (j : J), Finite (F.obj j)
    hne : ∀ (j : J), Nonempty (F.obj j)
    F' : CategoryTheory.Functor J TopCat := F.comp TopCat.discrete
    ⊢ F.sections.Nonempty
  -/
  haveI : ∀ j, DiscreteTopology (F'.obj j) := fun _ => ⟨rfl⟩
  /-
    J : Type u
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.IsCofilteredOrEmpty J
    F : CategoryTheory.Functor J (Type u)
    hf : ∀ (j : J), Finite (F.obj j)
    hne : ∀ (j : J), Nonempty (F.obj j)
    F' : CategoryTheory.Functor J TopCat := F.comp TopCat.discrete
    this : ∀ (j : J), DiscreteTopology ↑(F'.obj j)
    ⊢ F.sections.Nonempty
  -/
  haveI : ∀ j, Finite (F'.obj j) := hf
  /-
    J : Type u
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.IsCofilteredOrEmpty J
    F : CategoryTheory.Functor J (Type u)
    hf : ∀ (j : J), Finite (F.obj j)
    hne : ∀ (j : J), Nonempty (F.obj j)
    F' : CategoryTheory.Functor J TopCat := F.comp TopCat.discrete
    this✝ : ∀ (j : J), DiscreteTopology ↑(F'.obj j)
    this : ∀ (j : J), Finite ↑(F'.obj j)
    ⊢ F.sections.Nonempty
  -/
  haveI : ∀ j, Nonempty (F'.obj j) := hne
  /-
    J : Type u
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.IsCofilteredOrEmpty J
    F : CategoryTheory.Functor J (Type u)
    hf : ∀ (j : J), Finite (F.obj j)
    hne : ∀ (j : J), Nonempty (F.obj j)
    F' : CategoryTheory.Functor J TopCat := F.comp TopCat.discrete
    this✝¹ : ∀ (j : J), DiscreteTopology ↑(F'.obj j)
    this✝ : ∀ (j : J), Finite ↑(F'.obj j)
    this : ∀ (j : J), Nonempty ↑(F'.obj j)
    ⊢ F.sections.Nonempty
  -/
  obtain ⟨⟨u, hu⟩⟩ := TopCat.nonempty_limitCone_of_compact_t2_cofiltered_system.{u} F'
  /-
    case intro.mk
    J : Type u
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.IsCofilteredOrEmpty J
    F : CategoryTheory.Functor J (Type u)
    hf : ∀ (j : J), Finite (F.obj j)
    hne : ∀ (j : J), Nonempty (F.obj j)
    F' : CategoryTheory.Functor J TopCat := F.comp TopCat.discrete
    this✝¹ : ∀ (j : J), DiscreteTopology ↑(F'.obj j)
    this✝ : ∀ (j : J), Finite ↑(F'.obj j)
    this : ∀ (j : J), Nonempty ↑(F'.obj j)
    u : (j : J) → ↑(F'.obj j)
    hu : Membership.mem (setOf fun u => ∀ {i j : J} (f : Quiver.Hom i j), Eq ((F'. …
    ⊢ F.sections.Nonempty
  -/
  exact ⟨u, hu⟩
  /-
    🎉 no goals
  -/


/-- The cofiltered limit of nonempty finite types is nonempty.

See `nonempty_sections_of_finite_inverse_system` for a specialization to inverse limits. -/
theorem nonempty_sections_of_finite_cofiltered_system {J : Type u} [Category.{w} J]
    [IsCofilteredOrEmpty J] (F : J ⥤ Type v) [∀ j : J, Finite (F.obj j)]
    [∀ j : J, Nonempty (F.obj j)] : F.sections.Nonempty := by
  -- Step 1: lift everything to the `max u v w` universe.
  /-
    J : Type u
    inst✝³ : CategoryTheory.Category.{w, u} J
    inst✝² : CategoryTheory.IsCofilteredOrEmpty J
    F : CategoryTheory.Functor J (Type v)
    inst✝¹ : ∀ (j : J), Finite (F.obj j)
    inst✝ : ∀ (j : J), Nonempty (F.obj j)
    ⊢ F.sections.Nonempty
  -/
  let J' : Type max w v u := AsSmall.{max w v} J
  /-
    J : Type u
    inst✝³ : CategoryTheory.Category.{w, u} J
    inst✝² : CategoryTheory.IsCofilteredOrEmpty J
    F : CategoryTheory.Functor J (Type v)
    inst✝¹ : ∀ (j : J), Finite (F.obj j)
    inst✝ : ∀ (j : J), Nonempty (F.obj j)
    J' : Type (max w v u) := CategoryTheory.AsSmall J
    ⊢ F.sections.Nonempty
  -/
  let down : J' ⥤ J := AsSmall.down
  /-
    J : Type u
    inst✝³ : CategoryTheory.Category.{w, u} J
    inst✝² : CategoryTheory.IsCofilteredOrEmpty J
    F : CategoryTheory.Functor J (Type v)
    inst✝¹ : ∀ (j : J), Finite (F.obj j)
    inst✝ : ∀ (j : J), Nonempty (F.obj j)
    J' : Type (max w v u) := CategoryTheory.AsSmall J
    down : CategoryTheory.Functor J' J := CategoryTheory.AsSmall.down
    ⊢ F.sections.Nonempty
  -/
  let F' : J' ⥤ Type max u v w := down ⋙ F ⋙ uliftFunctor.{max u w, v}
  /-
    J : Type u
    inst✝³ : CategoryTheory.Category.{w, u} J
    inst✝² : CategoryTheory.IsCofilteredOrEmpty J
    F : CategoryTheory.Functor J (Type v)
    inst✝¹ : ∀ (j : J), Finite (F.obj j)
    inst✝ : ∀ (j : J), Nonempty (F.obj j)
    J' : Type (max w v u) := CategoryTheory.AsSmall J
    down : CategoryTheory.Functor J' J := CategoryTheory.AsSmall.down
    F' : CategoryTheory.Functor J' (Type (max u v w)) := down.comp (F.comp Categor …
    ⊢ F.sections.Nonempty
  -/
  haveI : ∀ i, Nonempty (F'.obj i) := fun i => ⟨⟨Classical.arbitrary (F.obj (down.obj i))⟩⟩
  /-
    J : Type u
    inst✝³ : CategoryTheory.Category.{w, u} J
    inst✝² : CategoryTheory.IsCofilteredOrEmpty J
    F : CategoryTheory.Functor J (Type v)
    inst✝¹ : ∀ (j : J), Finite (F.obj j)
    inst✝ : ∀ (j : J), Nonempty (F.obj j)
    J' : Type (max w v u) := CategoryTheory.AsSmall J
    down : CategoryTheory.Functor J' J := CategoryTheory.AsSmall.down
    F' : CategoryTheory.Functor J' (Type (max u v w)) := down.comp (F.comp Categor …
    this : ∀ (i : J'), Nonempty (F'.obj i)
    ⊢ F.sections.Nonempty
  -/
  haveI : ∀ i, Finite (F'.obj i) := fun i => Finite.of_equiv (F.obj (down.obj i)) Equiv.ulift.symm
  -- Step 2: apply the bootstrap theorem
  /-
    J : Type u
    inst✝³ : CategoryTheory.Category.{w, u} J
    inst✝² : CategoryTheory.IsCofilteredOrEmpty J
    F : CategoryTheory.Functor J (Type v)
    inst✝¹ : ∀ (j : J), Finite (F.obj j)
    inst✝ : ∀ (j : J), Nonempty (F.obj j)
    J' : Type (max w v u) := CategoryTheory.AsSmall J
    down : CategoryTheory.Functor J' J := CategoryTheory.AsSmall.down
    F' : CategoryTheory.Functor J' (Type (max u v w)) := down.comp (F.comp Categor …
    this✝ : ∀ (i : J'), Nonempty (F'.obj i)
    this : ∀ (i : J'), Finite (F'.obj i)
    ⊢ F.sections.Nonempty
  -/
  cases isEmpty_or_nonempty J
    /-
      case inl
      J : Type u
      inst✝³ : CategoryTheory.Category.{w, u} J
      inst✝² : CategoryTheory.IsCofilteredOrEmpty J
      F : CategoryTheory.Functor J (Type v)
      inst✝¹ : ∀ (j : J), Finite (F.obj j)
      inst✝ : ∀ (j : J), Nonempty (F.obj j)
      J' : Type (max w v u) := CategoryTheory.AsSmall J
      down : CategoryTheory.Functor J' J := CategoryTheory.AsSmall.down
      F' : CategoryTheory.Functor J' (Type (max u v w)) := down.comp (F.comp Categor …
      this✝ : ∀ (i : J'), Nonempty (F'.obj i)
      this : ∀ (i : J'), Finite (F'.obj i)
      h✝ : IsEmpty J
      ⊢ F.sections.Nonempty
    -/
                     /-
                       🎉 no goals
                     -/
  · fconstructor <;> apply isEmptyElim
                     /-
                       🎉 no goals
                     -/
  /-
    case inr
    J : Type u
    inst✝³ : CategoryTheory.Category.{w, u} J
    inst✝² : CategoryTheory.IsCofilteredOrEmpty J
    F : CategoryTheory.Functor J (Type v)
    inst✝¹ : ∀ (j : J), Finite (F.obj j)
    inst✝ : ∀ (j : J), Nonempty (F.obj j)
    J' : Type (max w v u) := CategoryTheory.AsSmall J
    down : CategoryTheory.Functor J' J := CategoryTheory.AsSmall.down
    F' : CategoryTheory.Functor J' (Type (max u v w)) := down.comp (F.comp Categor …
    this✝ : ∀ (i : J'), Nonempty (F'.obj i)
    this : ∀ (i : J'), Finite (F'.obj i)
    h✝ : Nonempty J
    ⊢ F.sections.Nonempty
  -/
  haveI : IsCofiltered J := ⟨⟩
  /-
    case inr
    J : Type u
    inst✝³ : CategoryTheory.Category.{w, u} J
    inst✝² : CategoryTheory.IsCofilteredOrEmpty J
    F : CategoryTheory.Functor J (Type v)
    inst✝¹ : ∀ (j : J), Finite (F.obj j)
    inst✝ : ∀ (j : J), Nonempty (F.obj j)
    J' : Type (max w v u) := CategoryTheory.AsSmall J
    down : CategoryTheory.Functor J' J := CategoryTheory.AsSmall.down
    F' : CategoryTheory.Functor J' (Type (max u v w)) := down.comp (F.comp Categor …
    this✝¹ : ∀ (i : J'), Nonempty (F'.obj i)
    this✝ : ∀ (i : J'), Finite (F'.obj i)
    h✝ : Nonempty J
    this : CategoryTheory.IsCofiltered J
    ⊢ F.sections.Nonempty
  -/
  obtain ⟨u, hu⟩ := nonempty_sections_of_finite_cofiltered_system.init F'
  -- Step 3: interpret the results
  /-
    case inr.intro
    J : Type u
    inst✝³ : CategoryTheory.Category.{w, u} J
    inst✝² : CategoryTheory.IsCofilteredOrEmpty J
    F : CategoryTheory.Functor J (Type v)
    inst✝¹ : ∀ (j : J), Finite (F.obj j)
    inst✝ : ∀ (j : J), Nonempty (F.obj j)
    J' : Type (max w v u) := CategoryTheory.AsSmall J
    down : CategoryTheory.Functor J' J := CategoryTheory.AsSmall.down
    F' : CategoryTheory.Functor J' (Type (max u v w)) := down.comp (F.comp Categor …
    this✝¹ : ∀ (i : J'), Nonempty (F'.obj i)
    this✝ : ∀ (i : J'), Finite (F'.obj i)
    h✝ : Nonempty J
    this : CategoryTheory.IsCofiltered J
    u : (j : J') → F'.obj j
    hu : Membership.mem F'.sections u
    ⊢ F.sections.Nonempty
  -/
  use fun j => (u ⟨j⟩).down
  /-
    case h
    J : Type u
    inst✝³ : CategoryTheory.Category.{w, u} J
    inst✝² : CategoryTheory.IsCofilteredOrEmpty J
    F : CategoryTheory.Functor J (Type v)
    inst✝¹ : ∀ (j : J), Finite (F.obj j)
    inst✝ : ∀ (j : J), Nonempty (F.obj j)
    J' : Type (max w v u) := CategoryTheory.AsSmall J
    down : CategoryTheory.Functor J' J := CategoryTheory.AsSmall.down
    F' : CategoryTheory.Functor J' (Type (max u v w)) := down.comp (F.comp Categor …
    this✝¹ : ∀ (i : J'), Nonempty (F'.obj i)
    this✝ : ∀ (i : J'), Finite (F'.obj i)
    h✝ : Nonempty J
    this : CategoryTheory.IsCofiltered J
    u : (j : J') → F'.obj j
    hu : Membership.mem F'.sections u
    ⊢ Membership.mem F.sections fun j => (u { down := j }).down
  -/
  intro j j' f
  /-
    case h
    J : Type u
    inst✝³ : CategoryTheory.Category.{w, u} J
    inst✝² : CategoryTheory.IsCofilteredOrEmpty J
    F : CategoryTheory.Functor J (Type v)
    inst✝¹ : ∀ (j : J), Finite (F.obj j)
    inst✝ : ∀ (j : J), Nonempty (F.obj j)
    J' : Type (max w v u) := CategoryTheory.AsSmall J
    down : CategoryTheory.Functor J' J := CategoryTheory.AsSmall.down
    F' : CategoryTheory.Functor J' (Type (max u v w)) := down.comp (F.comp Categor …
    this✝¹ : ∀ (i : J'), Nonempty (F'.obj i)
    this✝ : ∀ (i : J'), Finite (F'.obj i)
    h✝ : Nonempty J
    this : CategoryTheory.IsCofiltered J
    u : (j : J') → F'.obj j
    hu : Membership.mem F'.sections u
    j j' : J
    f : Quiver.Hom j j'
    ⊢ Eq (F.map f ((fun j => (u { down := j }).down) j)) ((fun j => (u { down := j …
  -/
  have h := @hu (⟨j⟩ : J') (⟨j'⟩ : J') (ULift.up f)
  /-
    case h
    J : Type u
    inst✝³ : CategoryTheory.Category.{w, u} J
    inst✝² : CategoryTheory.IsCofilteredOrEmpty J
    F : CategoryTheory.Functor J (Type v)
    inst✝¹ : ∀ (j : J), Finite (F.obj j)
    inst✝ : ∀ (j : J), Nonempty (F.obj j)
    J' : Type (max w v u) := CategoryTheory.AsSmall J
    down : CategoryTheory.Functor J' J := CategoryTheory.AsSmall.down
    F' : CategoryTheory.Functor J' (Type (max u v w)) := down.comp (F.comp Categor …
    this✝¹ : ∀ (i : J'), Nonempty (F'.obj i)
    this✝ : ∀ (i : J'), Finite (F'.obj i)
    h✝ : Nonempty J
    this : CategoryTheory.IsCofiltered J
    u : (j : J') → F'.obj j
    hu : Membership.mem F'.sections u
    j j' : J
    f : Quiver.Hom j j'
    h : Eq (F'.map { down := f } (u { down := j })) (u { down := j' })
    ⊢ Eq (F.map f ((fun j => (u { down := j }).down) j)) ((fun j => (u { down := j …
  -/
  simp only [F', down, AsSmall.down, Functor.comp_map, uliftFunctor_map, Functor.op_map] at h
  /-
    case h
    J : Type u
    inst✝³ : CategoryTheory.Category.{w, u} J
    inst✝² : CategoryTheory.IsCofilteredOrEmpty J
    F : CategoryTheory.Functor J (Type v)
    inst✝¹ : ∀ (j : J), Finite (F.obj j)
    inst✝ : ∀ (j : J), Nonempty (F.obj j)
    J' : Type (max w v u) := CategoryTheory.AsSmall J
    down : CategoryTheory.Functor J' J := CategoryTheory.AsSmall.down
    F' : CategoryTheory.Functor J' (Type (max u v w)) := down.comp (F.comp Categor …
    this✝¹ : ∀ (i : J'), Nonempty (F'.obj i)
    this✝ : ∀ (i : J'), Finite (F'.obj i)
    h✝ : Nonempty J
    this : CategoryTheory.IsCofiltered J
    u : (j : J') → F'.obj j
    hu : Membership.mem F'.sections u
    j j' : J
    f : Quiver.Hom j j'
    h : Eq { down := F.map f (u { down := j }).down } (u { down := j' })
    ⊢ Eq (F.map f ((fun j => (u { down := j }).down) j)) ((fun j => (u { down := j …
  -/
  simp_rw [← h]
  /-
    🎉 no goals
  -/


/-- The inverse limit of nonempty finite types is nonempty.

See `nonempty_sections_of_finite_cofiltered_system` for a generalization to cofiltered limits.
That version applies in almost all cases, and the only difference is that this version
allows `J` to be empty.

This may be regarded as a generalization of Kőnig's lemma.
To specialize: given a locally finite connected graph, take `Jᵒᵖ` to be `ℕ` and
`F j` to be length-`j` paths that start from an arbitrary fixed vertex.
Elements of `F.sections` can be read off as infinite rays in the graph. -/
theorem nonempty_sections_of_finite_inverse_system {J : Type u} [Preorder J] [IsDirected J (· ≤ ·)]
    (F : Jᵒᵖ ⥤ Type v) [∀ j : Jᵒᵖ, Finite (F.obj j)] [∀ j : Jᵒᵖ, Nonempty (F.obj j)] :
    F.sections.Nonempty := by
  /-
    J : Type u
    inst✝³ : Preorder J
    inst✝² : IsDirected J fun x1 x2 => LE.le x1 x2
    F : CategoryTheory.Functor (Opposite J) (Type v)
    inst✝¹ : ∀ (j : Opposite J), Finite (F.obj j)
    inst✝ : ∀ (j : Opposite J), Nonempty (F.obj j)
    ⊢ F.sections.Nonempty
  -/
  cases isEmpty_or_nonempty J
    /-
      case inl
      J : Type u
      inst✝³ : Preorder J
      inst✝² : IsDirected J fun x1 x2 => LE.le x1 x2
      F : CategoryTheory.Functor (Opposite J) (Type v)
      inst✝¹ : ∀ (j : Opposite J), Finite (F.obj j)
      inst✝ : ∀ (j : Opposite J), Nonempty (F.obj j)
      h✝ : IsEmpty J
      ⊢ F.sections.Nonempty
    -/
  · haveI : IsEmpty Jᵒᵖ := ⟨fun j => isEmptyElim j.unop⟩ -- TODO: this should be a global instance
    /-
      case inl
      J : Type u
      inst✝³ : Preorder J
      inst✝² : IsDirected J fun x1 x2 => LE.le x1 x2
      F : CategoryTheory.Functor (Opposite J) (Type v)
      inst✝¹ : ∀ (j : Opposite J), Finite (F.obj j)
      inst✝ : ∀ (j : Opposite J), Nonempty (F.obj j)
      h✝ : IsEmpty J
      this : IsEmpty (Opposite J)
      ⊢ F.sections.Nonempty
    -/
    exact ⟨isEmptyElim, by apply isEmptyElim⟩
    /-
      🎉 no goals
    -/
    /-
      case inr
      J : Type u
      inst✝³ : Preorder J
      inst✝² : IsDirected J fun x1 x2 => LE.le x1 x2
      F : CategoryTheory.Functor (Opposite J) (Type v)
      inst✝¹ : ∀ (j : Opposite J), Finite (F.obj j)
      inst✝ : ∀ (j : Opposite J), Nonempty (F.obj j)
      h✝ : Nonempty J
      ⊢ F.sections.Nonempty
    -/
  · exact nonempty_sections_of_finite_cofiltered_system _
    /-
      🎉 no goals
    -/


/-- The eventual range of the functor `F : J ⥤ Type v` at index `j : J` is the intersection
of the ranges of all maps `F.map f` with `i : J` and `f : i ⟶ j`. -/
def eventualRange (j : J) :=
  ⋂ (i) (f : i ⟶ j), range (F.map f)


theorem mem_eventualRange_iff {x : F.obj j} :
    x ∈ F.eventualRange j ↔ ∀ ⦃i⦄ (f : i ⟶ j), x ∈ range (F.map f) :=
  mem_iInter₂


/-- The functor `F : J ⥤ Type v` satisfies the Mittag-Leffler condition if for all `j : J`,
there exists some `i : J` and `f : i ⟶ j` such that for all `k : J` and `g : k ⟶ j`, the range
of `F.map f` is contained in that of `F.map g`;
in other words (see `isMittagLeffler_iff_eventualRange`), the eventual range at `j` is attained
by some `f : i ⟶ j`. -/
def IsMittagLeffler : Prop :=
  ∀ j : J, ∃ (i : _) (f : i ⟶ j), ∀ ⦃k⦄ (g : k ⟶ j), range (F.map f) ⊆ range (F.map g)


theorem isMittagLeffler_iff_eventualRange :
    F.IsMittagLeffler ↔ ∀ j : J, ∃ (i : _) (f : i ⟶ j), F.eventualRange j = range (F.map f) :=
  forall_congr' fun _ =>
    exists₂_congr fun _ _ =>
      ⟨fun h => (iInter₂_subset _ _).antisymm <| subset_iInter₂ h, fun h => h ▸ iInter₂_subset⟩


theorem IsMittagLeffler.subset_image_eventualRange (h : F.IsMittagLeffler) (f : j ⟶ i) :
    F.eventualRange i ⊆ F.map f '' F.eventualRange j := by
  /-
    J : Type u
    inst✝ : CategoryTheory.Category.{u_1, u} J
    F : CategoryTheory.Functor J (Type v)
    i j : J
    h : F.IsMittagLeffler
    f : Quiver.Hom j i
    ⊢ HasSubset.Subset (F.eventualRange i) (Set.image (F.map f) (F.eventualRange j))
  -/
  obtain ⟨k, g, hg⟩ := F.isMittagLeffler_iff_eventualRange.1 h j
  /-
    case intro.intro
    J : Type u
    inst✝ : CategoryTheory.Category.{u_1, u} J
    F : CategoryTheory.Functor J (Type v)
    i j : J
    h : F.IsMittagLeffler
    f : Quiver.Hom j i
    k : J
    g : Quiver.Hom k j
    hg : Eq (F.eventualRange j) (Set.range (F.map g))
    ⊢ HasSubset.Subset (F.eventualRange i) (Set.image (F.map f) (F.eventualRange j))
  -/
  rw [hg]; intro x hx
  /-
    case intro.intro
    J : Type u
    inst✝ : CategoryTheory.Category.{u_1, u} J
    F : CategoryTheory.Functor J (Type v)
    i j : J
    h : F.IsMittagLeffler
    f : Quiver.Hom j i
    k : J
    g : Quiver.Hom k j
    hg : Eq (F.eventualRange j) (Set.range (F.map g))
    x : F.obj i
    hx : Membership.mem (F.eventualRange i) x
    ⊢ Membership.mem (Set.image (F.map f) (Set.range (F.map g))) x
  -/
  obtain ⟨x, rfl⟩ := F.mem_eventualRange_iff.1 hx (g ≫ f)
  /-
    case intro.intro.intro
    J : Type u
    inst✝ : CategoryTheory.Category.{u_1, u} J
    F : CategoryTheory.Functor J (Type v)
    i j : J
    h : F.IsMittagLeffler
    f : Quiver.Hom j i
    k : J
    g : Quiver.Hom k j
    hg : Eq (F.eventualRange j) (Set.range (F.map g))
    x : F.obj k
    hx : Membership.mem (F.eventualRange i) (F.map (CategoryTheory.CategoryStruct. …
    ⊢ Membership.mem (Set.image (F.map f) (Set.range (F.map g))) (F.map (CategoryT …
  -/
  exact ⟨_, ⟨x, rfl⟩, by rw [map_comp_apply]⟩
  /-
    🎉 no goals
  -/


theorem eventualRange_eq_range_precomp (f : i ⟶ j) (g : j ⟶ k)
    (h : F.eventualRange k = range (F.map g)) : F.eventualRange k = range (F.map <| f ≫ g) := by
  /-
    J : Type u
    inst✝ : CategoryTheory.Category.{u_1, u} J
    F : CategoryTheory.Functor J (Type v)
    i j k : J
    f : Quiver.Hom i j
    g : Quiver.Hom j k
    h : Eq (F.eventualRange k) (Set.range (F.map g))
    ⊢ Eq (F.eventualRange k) (Set.range (F.map (CategoryTheory.CategoryStruct.comp …
  -/
  apply subset_antisymm
    /-
      case a
      J : Type u
      inst✝ : CategoryTheory.Category.{u_1, u} J
      F : CategoryTheory.Functor J (Type v)
      i j k : J
      f : Quiver.Hom i j
      g : Quiver.Hom j k
      h : Eq (F.eventualRange k) (Set.range (F.map g))
      ⊢ HasSubset.Subset (F.eventualRange k) (Set.range (F.map (CategoryTheory.Categ …
    -/
  · apply iInter₂_subset
    /-
      🎉 no goals
    -/
    /-
      case a
      J : Type u
      inst✝ : CategoryTheory.Category.{u_1, u} J
      F : CategoryTheory.Functor J (Type v)
      i j k : J
      f : Quiver.Hom i j
      g : Quiver.Hom j k
      h : Eq (F.eventualRange k) (Set.range (F.map g))
      ⊢ HasSubset.Subset (Set.range (F.map (CategoryTheory.CategoryStruct.comp f g)) …
    -/
  · rw [h, F.map_comp]
    /-
      case a
      J : Type u
      inst✝ : CategoryTheory.Category.{u_1, u} J
      F : CategoryTheory.Functor J (Type v)
      i j k : J
      f : Quiver.Hom i j
      g : Quiver.Hom j k
      h : Eq (F.eventualRange k) (Set.range (F.map g))
      ⊢ HasSubset.Subset (Set.range (CategoryTheory.CategoryStruct.comp (F.map f) (F …
    -/
    apply range_comp_subset_range
    /-
      🎉 no goals
    -/


theorem isMittagLeffler_of_surjective (h : ∀ ⦃i j : J⦄ (f : i ⟶ j), (F.map f).Surjective) :
    F.IsMittagLeffler :=
                                  /-
                                    J : Type u
                                    inst✝ : CategoryTheory.Category.{u_1, u} J
                                    F : CategoryTheory.Functor J (Type v)
                                    h : ∀ ⦃i j : J⦄ (f : Quiver.Hom i j), Function.Surjective (F.map f)
                                    j k : J
                                    g : Quiver.Hom k j
                                    ⊢ HasSubset.Subset (Set.range (F.map (CategoryTheory.CategoryStruct.id j))) (S …
                                  -/
  fun j => ⟨j, 𝟙 j, fun k g => by rw [map_id, types_id, range_id, (h g).range_eq]⟩
                                  /-
                                    🎉 no goals
                                  -/


/-- The subfunctor of `F` obtained by restricting to the preimages of a set `s ∈ F.obj i`. -/
@[simps]
def toPreimages : J ⥤ Type v where
  obj j := ⋂ f : j ⟶ i, F.map f ⁻¹' s
  map g := MapsTo.restrict (F.map g) _ _ fun x h => by
    /-
      J : Type u
      inst✝ : CategoryTheory.Category.{?u.9986, u} J
      F : CategoryTheory.Functor J (Type v)
      i j k : J
      s : Set (F.obj i)
      X✝ Y✝ : J
      g : Quiver.Hom X✝ Y✝
      x : F.obj X✝
      h : Membership.mem (Set.iInter fun f => Set.preimage (F.map f) s) x
      ⊢ Membership.mem (Set.iInter fun f => Set.preimage (F.map f) s) (F.map g x)
    -/
    rw [mem_iInter] at h ⊢
    /-
      J : Type u
      inst✝ : CategoryTheory.Category.{?u.9986, u} J
      F : CategoryTheory.Functor J (Type v)
      i j k : J
      s : Set (F.obj i)
      X✝ Y✝ : J
      g : Quiver.Hom X✝ Y✝
      x : F.obj X✝
      h : ∀ (i_1 : Quiver.Hom X✝ i), Membership.mem (Set.preimage (F.map i_1) s) x
      ⊢ ∀ (i_1 : Quiver.Hom Y✝ i), Membership.mem (Set.preimage (F.map i_1) s) (F.ma …
    -/
    intro f
    /-
      J : Type u
      inst✝ : CategoryTheory.Category.{?u.9986, u} J
      F : CategoryTheory.Functor J (Type v)
      i j k : J
      s : Set (F.obj i)
      X✝ Y✝ : J
      g : Quiver.Hom X✝ Y✝
      x : F.obj X✝
      h : ∀ (i_1 : Quiver.Hom X✝ i), Membership.mem (Set.preimage (F.map i_1) s) x
      f : Quiver.Hom Y✝ i
      ⊢ Membership.mem (Set.preimage (F.map f) s) (F.map g x)
    -/
    rw [← mem_preimage, preimage_preimage, mem_preimage]
    /-
      J : Type u
      inst✝ : CategoryTheory.Category.{?u.9986, u} J
      F : CategoryTheory.Functor J (Type v)
      i j k : J
      s : Set (F.obj i)
      X✝ Y✝ : J
      g : Quiver.Hom X✝ Y✝
      x : F.obj X✝
      h : ∀ (i_1 : Quiver.Hom X✝ i), Membership.mem (Set.preimage (F.map i_1) s) x
      f : Quiver.Hom Y✝ i
      ⊢ Membership.mem s (F.map f (F.map g x))
    -/
    convert h (g ≫ f); rw [F.map_comp]; rfl
                                        /-
                                          🎉 no goals
                                        -/
  map_id j := by
    #adaptation_note /-- nightly-2024-03-16: simp was
    simp (config := { unfoldPartialApp := true }) only [MapsTo.restrict, Subtype.map, F.map_id] -/
    /-
      J : Type u
      inst✝ : CategoryTheory.Category.{?u.9986, u} J
      F : CategoryTheory.Functor J (Type v)
      i j✝ k : J
      s : Set (F.obj i)
      j : J
      ⊢ Eq ({ obj := fun j => ↑(Set.iInter fun f => Set.preimage (F.map f) s), map : …
    -/
    simp only [MapsTo.restrict, Subtype.map_def, F.map_id]
    /-
      J : Type u
      inst✝ : CategoryTheory.Category.{?u.9986, u} J
      F : CategoryTheory.Functor J (Type v)
      i j✝ k : J
      s : Set (F.obj i)
      j : J
      ⊢ Eq (fun x => ⟨CategoryTheory.CategoryStruct.id (F.obj j) ↑x, ⋯⟩) (CategoryTh …
    -/
    ext
    /-
      case h.a
      J : Type u
      inst✝ : CategoryTheory.Category.{?u.9986, u} J
      F : CategoryTheory.Functor J (Type v)
      i j✝ k : J
      s : Set (F.obj i)
      j : J
      a✝ : Subtype fun x => Membership.mem (Set.iInter fun f => Set.preimage (F.map  …
      ⊢ Eq ↑⟨CategoryTheory.CategoryStruct.id (F.obj j) ↑a✝, ⋯⟩ ↑(CategoryTheory.Cat …
    -/
    rfl
    /-
      🎉 no goals
    -/
  map_comp f g := by
    #adaptation_note /-- nightly-2024-03-16: simp was
    simp (config := { unfoldPartialApp := true }) only [MapsTo.restrict, Subtype.map, F.map_comp] -/
    /-
      J : Type u
      inst✝ : CategoryTheory.Category.{?u.9986, u} J
      F : CategoryTheory.Functor J (Type v)
      i j k : J
      s : Set (F.obj i)
      X✝ Y✝ Z✝ : J
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Eq ({ obj := fun j => ↑(Set.iInter fun f => Set.preimage (F.map f) s), map : …
    -/
    simp only [MapsTo.restrict, Subtype.map_def, F.map_comp]
    /-
      J : Type u
      inst✝ : CategoryTheory.Category.{?u.9986, u} J
      F : CategoryTheory.Functor J (Type v)
      i j k : J
      s : Set (F.obj i)
      X✝ Y✝ Z✝ : J
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Eq (fun x => ⟨CategoryTheory.CategoryStruct.comp (F.map f) (F.map g) ↑x, ⋯⟩) …
    -/
    rfl
    /-
      🎉 no goals
    -/


instance toPreimages_finite [∀ j, Finite (F.obj j)] : ∀ j, Finite ((F.toPreimages s).obj j) :=
  fun _ => Subtype.finite


theorem eventualRange_mapsTo (f : j ⟶ i) :
    (F.eventualRange j).MapsTo (F.map f) (F.eventualRange i) := fun x hx => by
  /-
    J : Type u
    inst✝¹ : CategoryTheory.Category.{u_1, u} J
    F : CategoryTheory.Functor J (Type v)
    i j : J
    inst✝ : CategoryTheory.IsCofilteredOrEmpty J
    f : Quiver.Hom j i
    x : F.obj j
    hx : Membership.mem (F.eventualRange j) x
    ⊢ Membership.mem (F.eventualRange i) (F.map f x)
  -/
  rw [mem_eventualRange_iff] at hx ⊢
  /-
    J : Type u
    inst✝¹ : CategoryTheory.Category.{u_1, u} J
    F : CategoryTheory.Functor J (Type v)
    i j : J
    inst✝ : CategoryTheory.IsCofilteredOrEmpty J
    f : Quiver.Hom j i
    x : F.obj j
    hx : ∀ ⦃i : J⦄ (f : Quiver.Hom i j), Membership.mem (Set.range (F.map f)) x
    ⊢ ∀ ⦃i_1 : J⦄ (f_1 : Quiver.Hom i_1 i), Membership.mem (Set.range (F.map f_1)) …
  -/
  intro k f'
  /-
    J : Type u
    inst✝¹ : CategoryTheory.Category.{u_1, u} J
    F : CategoryTheory.Functor J (Type v)
    i j : J
    inst✝ : CategoryTheory.IsCofilteredOrEmpty J
    f : Quiver.Hom j i
    x : F.obj j
    hx : ∀ ⦃i : J⦄ (f : Quiver.Hom i j), Membership.mem (Set.range (F.map f)) x
    k : J
    f' : Quiver.Hom k i
    ⊢ Membership.mem (Set.range (F.map f')) (F.map f x)
  -/
  obtain ⟨l, g, g', he⟩ := cospan f f'
  /-
    case intro.intro.intro
    J : Type u
    inst✝¹ : CategoryTheory.Category.{u_1, u} J
    F : CategoryTheory.Functor J (Type v)
    i j : J
    inst✝ : CategoryTheory.IsCofilteredOrEmpty J
    f : Quiver.Hom j i
    x : F.obj j
    hx : ∀ ⦃i : J⦄ (f : Quiver.Hom i j), Membership.mem (Set.range (F.map f)) x
    k : J
    f' : Quiver.Hom k i
    l : J
    g : Quiver.Hom l j
    g' : Quiver.Hom l k
    he : Eq (CategoryTheory.CategoryStruct.comp g f) (CategoryTheory.CategoryStruc …
    ⊢ Membership.mem (Set.range (F.map f')) (F.map f x)
  -/
  obtain ⟨x, rfl⟩ := hx g
  /-
    case intro.intro.intro.intro
    J : Type u
    inst✝¹ : CategoryTheory.Category.{u_1, u} J
    F : CategoryTheory.Functor J (Type v)
    i j : J
    inst✝ : CategoryTheory.IsCofilteredOrEmpty J
    f : Quiver.Hom j i
    k : J
    f' : Quiver.Hom k i
    l : J
    g : Quiver.Hom l j
    g' : Quiver.Hom l k
    he : Eq (CategoryTheory.CategoryStruct.comp g f) (CategoryTheory.CategoryStruc …
    x : F.obj l
    hx : ∀ ⦃i : J⦄ (f : Quiver.Hom i j), Membership.mem (Set.range (F.map f)) (F.m …
    ⊢ Membership.mem (Set.range (F.map f')) (F.map f (F.map g x))
  -/
  rw [← map_comp_apply, he, F.map_comp]
  /-
    case intro.intro.intro.intro
    J : Type u
    inst✝¹ : CategoryTheory.Category.{u_1, u} J
    F : CategoryTheory.Functor J (Type v)
    i j : J
    inst✝ : CategoryTheory.IsCofilteredOrEmpty J
    f : Quiver.Hom j i
    k : J
    f' : Quiver.Hom k i
    l : J
    g : Quiver.Hom l j
    g' : Quiver.Hom l k
    he : Eq (CategoryTheory.CategoryStruct.comp g f) (CategoryTheory.CategoryStruc …
    x : F.obj l
    hx : ∀ ⦃i : J⦄ (f : Quiver.Hom i j), Membership.mem (Set.range (F.map f)) (F.m …
    ⊢ Membership.mem (Set.range (F.map f')) (CategoryTheory.CategoryStruct.comp (F …
  -/
  exact ⟨_, rfl⟩
  /-
    🎉 no goals
  -/


theorem IsMittagLeffler.eq_image_eventualRange (h : F.IsMittagLeffler) (f : j ⟶ i) :
    F.eventualRange i = F.map f '' F.eventualRange j :=
  (h.subset_image_eventualRange F f).antisymm <| mapsTo'.1 (F.eventualRange_mapsTo f)


theorem eventualRange_eq_iff {f : i ⟶ j} :
    F.eventualRange j = range (F.map f) ↔
      ∀ ⦃k⦄ (g : k ⟶ i), range (F.map f) ⊆ range (F.map <| g ≫ f) := by
  /-
    J : Type u
    inst✝¹ : CategoryTheory.Category.{u_1, u} J
    F : CategoryTheory.Functor J (Type v)
    i j : J
    inst✝ : CategoryTheory.IsCofilteredOrEmpty J
    f : Quiver.Hom i j
    ⊢ Iff (Eq (F.eventualRange j) (Set.range (F.map f))) (∀ ⦃k : J⦄ (g : Quiver.Ho …
  -/
  rw [subset_antisymm_iff, eventualRange, and_iff_right (iInter₂_subset _ _), subset_iInter₂_iff]
  /-
    J : Type u
    inst✝¹ : CategoryTheory.Category.{u_1, u} J
    F : CategoryTheory.Functor J (Type v)
    i j : J
    inst✝ : CategoryTheory.IsCofilteredOrEmpty J
    f : Quiver.Hom i j
    ⊢ Iff (∀ (i_1 : J) (j_1 : Quiver.Hom i_1 j), HasSubset.Subset (Set.range (F.ma …
  -/
  refine ⟨fun h k g => h _ _, fun h j' f' => ?_⟩
  /-
    J : Type u
    inst✝¹ : CategoryTheory.Category.{u_1, u} J
    F : CategoryTheory.Functor J (Type v)
    i j : J
    inst✝ : CategoryTheory.IsCofilteredOrEmpty J
    f : Quiver.Hom i j
    h : ∀ ⦃k : J⦄ (g : Quiver.Hom k i), HasSubset.Subset (Set.range (F.map f)) (Se …
    j' : J
    f' : Quiver.Hom j' j
    ⊢ HasSubset.Subset (Set.range (F.map f)) (Set.range (F.map f'))
  -/
  obtain ⟨k, g, g', he⟩ := cospan f f'
  /-
    case intro.intro.intro
    J : Type u
    inst✝¹ : CategoryTheory.Category.{u_1, u} J
    F : CategoryTheory.Functor J (Type v)
    i j : J
    inst✝ : CategoryTheory.IsCofilteredOrEmpty J
    f : Quiver.Hom i j
    h : ∀ ⦃k : J⦄ (g : Quiver.Hom k i), HasSubset.Subset (Set.range (F.map f)) (Se …
    j' : J
    f' : Quiver.Hom j' j
    k : J
    g : Quiver.Hom k i
    g' : Quiver.Hom k j'
    he : Eq (CategoryTheory.CategoryStruct.comp g f) (CategoryTheory.CategoryStruc …
    ⊢ HasSubset.Subset (Set.range (F.map f)) (Set.range (F.map f'))
  -/
  refine (h g).trans ?_
  /-
    case intro.intro.intro
    J : Type u
    inst✝¹ : CategoryTheory.Category.{u_1, u} J
    F : CategoryTheory.Functor J (Type v)
    i j : J
    inst✝ : CategoryTheory.IsCofilteredOrEmpty J
    f : Quiver.Hom i j
    h : ∀ ⦃k : J⦄ (g : Quiver.Hom k i), HasSubset.Subset (Set.range (F.map f)) (Se …
    j' : J
    f' : Quiver.Hom j' j
    k : J
    g : Quiver.Hom k i
    g' : Quiver.Hom k j'
    he : Eq (CategoryTheory.CategoryStruct.comp g f) (CategoryTheory.CategoryStruc …
    ⊢ HasSubset.Subset (Set.range (F.map (CategoryTheory.CategoryStruct.comp g f)) …
  -/
  rw [he, F.map_comp]
  /-
    case intro.intro.intro
    J : Type u
    inst✝¹ : CategoryTheory.Category.{u_1, u} J
    F : CategoryTheory.Functor J (Type v)
    i j : J
    inst✝ : CategoryTheory.IsCofilteredOrEmpty J
    f : Quiver.Hom i j
    h : ∀ ⦃k : J⦄ (g : Quiver.Hom k i), HasSubset.Subset (Set.range (F.map f)) (Se …
    j' : J
    f' : Quiver.Hom j' j
    k : J
    g : Quiver.Hom k i
    g' : Quiver.Hom k j'
    he : Eq (CategoryTheory.CategoryStruct.comp g f) (CategoryTheory.CategoryStruc …
    ⊢ HasSubset.Subset (Set.range (CategoryTheory.CategoryStruct.comp (F.map g') ( …
  -/
  apply range_comp_subset_range
  /-
    🎉 no goals
  -/


theorem isMittagLeffler_iff_subset_range_comp : F.IsMittagLeffler ↔ ∀ j : J, ∃ (i : _) (f : i ⟶ j),
    ∀ ⦃k⦄ (g : k ⟶ i), range (F.map f) ⊆ range (F.map <| g ≫ f) := by
  /-
    J : Type u
    inst✝¹ : CategoryTheory.Category.{u_1, u} J
    F : CategoryTheory.Functor J (Type v)
    inst✝ : CategoryTheory.IsCofilteredOrEmpty J
    ⊢ Iff F.IsMittagLeffler (∀ (j : J), Exists fun i => Exists fun f => ∀ ⦃k : J⦄  …
  -/
  simp_rw [isMittagLeffler_iff_eventualRange, eventualRange_eq_iff]
  /-
    🎉 no goals
  -/


theorem IsMittagLeffler.toPreimages (h : F.IsMittagLeffler) : (F.toPreimages s).IsMittagLeffler :=
  (isMittagLeffler_iff_subset_range_comp _).2 fun j => by
    /-
      J : Type u
      inst✝¹ : CategoryTheory.Category.{u_1, u} J
      F : CategoryTheory.Functor J (Type v)
      i : J
      s : Set (F.obj i)
      inst✝ : CategoryTheory.IsCofilteredOrEmpty J
      h : F.IsMittagLeffler
      j : J
      ⊢ Exists fun i_1 => Exists fun f => ∀ ⦃k : J⦄ (g : Quiver.Hom k i_1), HasSubse …
    -/
    obtain ⟨j₁, g₁, f₁, -⟩ := IsCofilteredOrEmpty.cone_objs i j
    /-
      case intro.intro.intro
      J : Type u
      inst✝¹ : CategoryTheory.Category.{u_1, u} J
      F : CategoryTheory.Functor J (Type v)
      i : J
      s : Set (F.obj i)
      inst✝ : CategoryTheory.IsCofilteredOrEmpty J
      h : F.IsMittagLeffler
      j j₁ : J
      g₁ : Quiver.Hom j₁ i
      f₁ : Quiver.Hom j₁ j
      ⊢ Exists fun i_1 => Exists fun f => ∀ ⦃k : J⦄ (g : Quiver.Hom k i_1), HasSubse …
    -/
    obtain ⟨j₂, f₂, h₂⟩ := F.isMittagLeffler_iff_eventualRange.1 h j₁
    /-
      case intro.intro.intro.intro.intro
      J : Type u
      inst✝¹ : CategoryTheory.Category.{u_1, u} J
      F : CategoryTheory.Functor J (Type v)
      i : J
      s : Set (F.obj i)
      inst✝ : CategoryTheory.IsCofilteredOrEmpty J
      h : F.IsMittagLeffler
      j j₁ : J
      g₁ : Quiver.Hom j₁ i
      f₁ : Quiver.Hom j₁ j
      j₂ : J
      f₂ : Quiver.Hom j₂ j₁
      h₂ : Eq (F.eventualRange j₁) (Set.range (F.map f₂))
      ⊢ Exists fun i_1 => Exists fun f => ∀ ⦃k : J⦄ (g : Quiver.Hom k i_1), HasSubse …
    -/
    refine ⟨j₂, f₂ ≫ f₁, fun j₃ f₃ => ?_⟩
    /-
      case intro.intro.intro.intro.intro
      J : Type u
      inst✝¹ : CategoryTheory.Category.{u_1, u} J
      F : CategoryTheory.Functor J (Type v)
      i : J
      s : Set (F.obj i)
      inst✝ : CategoryTheory.IsCofilteredOrEmpty J
      h : F.IsMittagLeffler
      j j₁ : J
      g₁ : Quiver.Hom j₁ i
      f₁ : Quiver.Hom j₁ j
      j₂ : J
      f₂ : Quiver.Hom j₂ j₁
      h₂ : Eq (F.eventualRange j₁) (Set.range (F.map f₂))
      j₃ : J
      f₃ : Quiver.Hom j₃ j₂
      ⊢ HasSubset.Subset (Set.range ((F.toPreimages s).map (CategoryTheory.CategoryS …
    -/
    rintro _ ⟨⟨x, hx⟩, rfl⟩
    have : F.map f₂ x ∈ F.eventualRange j₁ := by
      rw [h₂]
      exact ⟨_, rfl⟩
    /-
      case intro.intro.intro.intro.intro.intro.mk
      J : Type u
      inst✝¹ : CategoryTheory.Category.{u_1, u} J
      F : CategoryTheory.Functor J (Type v)
      i : J
      s : Set (F.obj i)
      inst✝ : CategoryTheory.IsCofilteredOrEmpty J
      h : F.IsMittagLeffler
      j j₁ : J
      g₁ : Quiver.Hom j₁ i
      f₁ : Quiver.Hom j₁ j
      j₂ : J
      f₂ : Quiver.Hom j₂ j₁
      h₂ : Eq (F.eventualRange j₁) (Set.range (F.map f₂))
      j₃ : J
      f₃ : Quiver.Hom j₃ j₂
      x : F.obj j₂
      hx : Membership.mem (Set.iInter fun f => Set.preimage (F.map f) s) x
      this : Membership.mem (F.eventualRange j₁) (F.map f₂ x)
      ⊢ Membership.mem (Set.range ((F.toPreimages s).map (CategoryTheory.CategoryStr …
    -/
    obtain ⟨y, hy, h₃⟩ := h.subset_image_eventualRange F (f₃ ≫ f₂) this
    /-
      case intro.intro.intro.intro.intro.intro.mk.intro.intro
      J : Type u
      inst✝¹ : CategoryTheory.Category.{u_1, u} J
      F : CategoryTheory.Functor J (Type v)
      i : J
      s : Set (F.obj i)
      inst✝ : CategoryTheory.IsCofilteredOrEmpty J
      h : F.IsMittagLeffler
      j j₁ : J
      g₁ : Quiver.Hom j₁ i
      f₁ : Quiver.Hom j₁ j
      j₂ : J
      f₂ : Quiver.Hom j₂ j₁
      h₂ : Eq (F.eventualRange j₁) (Set.range (F.map f₂))
      j₃ : J
      f₃ : Quiver.Hom j₃ j₂
      x : F.obj j₂
      hx : Membership.mem (Set.iInter fun f => Set.preimage (F.map f) s) x
      this : Membership.mem (F.eventualRange j₁) (F.map f₂ x)
      y : F.obj j₃
      hy : Membership.mem (F.eventualRange j₃) y
      h₃ : Eq (F.map (CategoryTheory.CategoryStruct.comp f₃ f₂) y) (F.map f₂ x)
      ⊢ Membership.mem (Set.range ((F.toPreimages s).map (CategoryTheory.CategoryStr …
    -/
    refine ⟨⟨y, mem_iInter.2 fun g₂ => ?_⟩, Subtype.ext ?_⟩
      /-
        case intro.intro.intro.intro.intro.intro.mk.intro.intro.refine_1
        J : Type u
        inst✝¹ : CategoryTheory.Category.{u_1, u} J
        F : CategoryTheory.Functor J (Type v)
        i : J
        s : Set (F.obj i)
        inst✝ : CategoryTheory.IsCofilteredOrEmpty J
        h : F.IsMittagLeffler
        j j₁ : J
        g₁ : Quiver.Hom j₁ i
        f₁ : Quiver.Hom j₁ j
        j₂ : J
        f₂ : Quiver.Hom j₂ j₁
        h₂ : Eq (F.eventualRange j₁) (Set.range (F.map f₂))
        j₃ : J
        f₃ : Quiver.Hom j₃ j₂
        x : F.obj j₂
        hx : Membership.mem (Set.iInter fun f => Set.preimage (F.map f) s) x
        this : Membership.mem (F.eventualRange j₁) (F.map f₂ x)
        y : F.obj j₃
        hy : Membership.mem (F.eventualRange j₃) y
        h₃ : Eq (F.map (CategoryTheory.CategoryStruct.comp f₃ f₂) y) (F.map f₂ x)
        g₂ : Quiver.Hom j₃ i
        ⊢ Membership.mem (Set.preimage (F.map g₂) s) y
      -/
    · obtain ⟨j₄, f₄, h₄⟩ := IsCofilteredOrEmpty.cone_maps g₂ ((f₃ ≫ f₂) ≫ g₁)
      /-
        case intro.intro.intro.intro.intro.intro.mk.intro.intro.refine_1.intro.intro
        J : Type u
        inst✝¹ : CategoryTheory.Category.{u_1, u} J
        F : CategoryTheory.Functor J (Type v)
        i : J
        s : Set (F.obj i)
        inst✝ : CategoryTheory.IsCofilteredOrEmpty J
        h : F.IsMittagLeffler
        j j₁ : J
        g₁ : Quiver.Hom j₁ i
        f₁ : Quiver.Hom j₁ j
        j₂ : J
        f₂ : Quiver.Hom j₂ j₁
        h₂ : Eq (F.eventualRange j₁) (Set.range (F.map f₂))
        j₃ : J
        f₃ : Quiver.Hom j₃ j₂
        x : F.obj j₂
        hx : Membership.mem (Set.iInter fun f => Set.preimage (F.map f) s) x
        this : Membership.mem (F.eventualRange j₁) (F.map f₂ x)
        y : F.obj j₃
        hy : Membership.mem (F.eventualRange j₃) y
        h₃ : Eq (F.map (CategoryTheory.CategoryStruct.comp f₃ f₂) y) (F.map f₂ x)
        g₂ : Quiver.Hom j₃ i
        j₄ : J
        f₄ : Quiver.Hom j₄ j₃
        h₄ : Eq (CategoryTheory.CategoryStruct.comp f₄ g₂) (CategoryTheory.CategoryStr …
        ⊢ Membership.mem (Set.preimage (F.map g₂) s) y
      -/
      obtain ⟨y, rfl⟩ := F.mem_eventualRange_iff.1 hy f₄
      /-
        case intro.intro.intro.intro.intro.intro.mk.intro.intro.refine_1.intro.intro.i …
        J : Type u
        inst✝¹ : CategoryTheory.Category.{u_1, u} J
        F : CategoryTheory.Functor J (Type v)
        i : J
        s : Set (F.obj i)
        inst✝ : CategoryTheory.IsCofilteredOrEmpty J
        h : F.IsMittagLeffler
        j j₁ : J
        g₁ : Quiver.Hom j₁ i
        f₁ : Quiver.Hom j₁ j
        j₂ : J
        f₂ : Quiver.Hom j₂ j₁
        h₂ : Eq (F.eventualRange j₁) (Set.range (F.map f₂))
        j₃ : J
        f₃ : Quiver.Hom j₃ j₂
        x : F.obj j₂
        hx : Membership.mem (Set.iInter fun f => Set.preimage (F.map f) s) x
        this : Membership.mem (F.eventualRange j₁) (F.map f₂ x)
        g₂ : Quiver.Hom j₃ i
        j₄ : J
        f₄ : Quiver.Hom j₄ j₃
        h₄ : Eq (CategoryTheory.CategoryStruct.comp f₄ g₂) (CategoryTheory.CategoryStr …
        y : F.obj j₄
        hy : Membership.mem (F.eventualRange j₃) (F.map f₄ y)
        h₃ : Eq (F.map (CategoryTheory.CategoryStruct.comp f₃ f₂) (F.map f₄ y)) (F.map …
        ⊢ Membership.mem (Set.preimage (F.map g₂) s) (F.map f₄ y)
      -/
      rw [← map_comp_apply] at h₃
      rw [mem_preimage, ← map_comp_apply, h₄, ← Category.assoc, map_comp_apply, h₃,
        ← map_comp_apply]
      /-
        case intro.intro.intro.intro.intro.intro.mk.intro.intro.refine_1.intro.intro.i …
        J : Type u
        inst✝¹ : CategoryTheory.Category.{u_1, u} J
        F : CategoryTheory.Functor J (Type v)
        i : J
        s : Set (F.obj i)
        inst✝ : CategoryTheory.IsCofilteredOrEmpty J
        h : F.IsMittagLeffler
        j j₁ : J
        g₁ : Quiver.Hom j₁ i
        f₁ : Quiver.Hom j₁ j
        j₂ : J
        f₂ : Quiver.Hom j₂ j₁
        h₂ : Eq (F.eventualRange j₁) (Set.range (F.map f₂))
        j₃ : J
        f₃ : Quiver.Hom j₃ j₂
        x : F.obj j₂
        hx : Membership.mem (Set.iInter fun f => Set.preimage (F.map f) s) x
        this : Membership.mem (F.eventualRange j₁) (F.map f₂ x)
        g₂ : Quiver.Hom j₃ i
        j₄ : J
        f₄ : Quiver.Hom j₄ j₃
        h₄ : Eq (CategoryTheory.CategoryStruct.comp f₄ g₂) (CategoryTheory.CategoryStr …
        y : F.obj j₄
        hy : Membership.mem (F.eventualRange j₃) (F.map f₄ y)
        h₃ : Eq (F.map (CategoryTheory.CategoryStruct.comp f₄ (CategoryTheory.Category …
        ⊢ Membership.mem s (F.map (CategoryTheory.CategoryStruct.comp f₂ g₁) x)
      -/
      apply mem_iInter.1 hx
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.intro.intro.intro.mk.intro.intro.refine_2
        J : Type u
        inst✝¹ : CategoryTheory.Category.{u_1, u} J
        F : CategoryTheory.Functor J (Type v)
        i : J
        s : Set (F.obj i)
        inst✝ : CategoryTheory.IsCofilteredOrEmpty J
        h : F.IsMittagLeffler
        j j₁ : J
        g₁ : Quiver.Hom j₁ i
        f₁ : Quiver.Hom j₁ j
        j₂ : J
        f₂ : Quiver.Hom j₂ j₁
        h₂ : Eq (F.eventualRange j₁) (Set.range (F.map f₂))
        j₃ : J
        f₃ : Quiver.Hom j₃ j₂
        x : F.obj j₂
        hx : Membership.mem (Set.iInter fun f => Set.preimage (F.map f) s) x
        this : Membership.mem (F.eventualRange j₁) (F.map f₂ x)
        y : F.obj j₃
        hy : Membership.mem (F.eventualRange j₃) y
        h₃ : Eq (F.map (CategoryTheory.CategoryStruct.comp f₃ f₂) y) (F.map f₂ x)
        ⊢ Eq ↑((F.toPreimages s).map (CategoryTheory.CategoryStruct.comp f₃ (CategoryT …
      -/
    · simp_rw [toPreimages_map, MapsTo.val_restrict_apply]
      /-
        case intro.intro.intro.intro.intro.intro.mk.intro.intro.refine_2
        J : Type u
        inst✝¹ : CategoryTheory.Category.{u_1, u} J
        F : CategoryTheory.Functor J (Type v)
        i : J
        s : Set (F.obj i)
        inst✝ : CategoryTheory.IsCofilteredOrEmpty J
        h : F.IsMittagLeffler
        j j₁ : J
        g₁ : Quiver.Hom j₁ i
        f₁ : Quiver.Hom j₁ j
        j₂ : J
        f₂ : Quiver.Hom j₂ j₁
        h₂ : Eq (F.eventualRange j₁) (Set.range (F.map f₂))
        j₃ : J
        f₃ : Quiver.Hom j₃ j₂
        x : F.obj j₂
        hx : Membership.mem (Set.iInter fun f => Set.preimage (F.map f) s) x
        this : Membership.mem (F.eventualRange j₁) (F.map f₂ x)
        y : F.obj j₃
        hy : Membership.mem (F.eventualRange j₃) y
        h₃ : Eq (F.map (CategoryTheory.CategoryStruct.comp f₃ f₂) y) (F.map f₂ x)
        ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp f₃ (CategoryTheory.CategoryStr …
      -/
      rw [← Category.assoc, map_comp_apply, h₃, map_comp_apply]
      /-
        🎉 no goals
      -/


theorem isMittagLeffler_of_exists_finite_range
    (h : ∀ j : J, ∃ (i : _) (f : i ⟶ j), (range <| F.map f).Finite) : F.IsMittagLeffler := by
  /-
    J : Type u
    inst✝¹ : CategoryTheory.Category.{u_1, u} J
    F : CategoryTheory.Functor J (Type v)
    inst✝ : CategoryTheory.IsCofilteredOrEmpty J
    h : ∀ (j : J), Exists fun i => Exists fun f => (Set.range (F.map f)).Finite
    ⊢ F.IsMittagLeffler
  -/
  intro j
  /-
    J : Type u
    inst✝¹ : CategoryTheory.Category.{u_1, u} J
    F : CategoryTheory.Functor J (Type v)
    inst✝ : CategoryTheory.IsCofilteredOrEmpty J
    h : ∀ (j : J), Exists fun i => Exists fun f => (Set.range (F.map f)).Finite
    j : J
    ⊢ Exists fun i => Exists fun f => ∀ ⦃k : J⦄ (g : Quiver.Hom k j), HasSubset.Su …
  -/
  obtain ⟨i, hi, hf⟩ := h j
  obtain ⟨m, ⟨i, f, hm⟩, hmin⟩ := Finset.wellFoundedLT.wf.has_min
    { s : Finset (F.obj j) | ∃ (i : _) (f : i ⟶ j), ↑s = range (F.map f) }
    ⟨_, i, hi, hf.coe_toFinset⟩
  refine ⟨i, f, fun k g =>
    (directedOn_range.mp <| F.ranges_directed j).is_bot_of_is_min ⟨⟨i, f⟩, rfl⟩ ?_ _ ⟨⟨k, g⟩, rfl⟩⟩
  /-
    case intro.intro.intro.intro.intro.intro
    J : Type u
    inst✝¹ : CategoryTheory.Category.{u_1, u} J
    F : CategoryTheory.Functor J (Type v)
    inst✝ : CategoryTheory.IsCofilteredOrEmpty J
    h : ∀ (j : J), Exists fun i => Exists fun f => (Set.range (F.map f)).Finite
    j i✝ : J
    hi : Quiver.Hom i✝ j
    hf : (Set.range (F.map hi)).Finite
    m : Finset (F.obj j)
    hmin : ∀ (x : Finset (F.obj j)), Membership.mem (setOf fun s => Exists fun i = …
    i : J
    f : Quiver.Hom i j
    hm : Eq (↑m) (Set.range (F.map f))
    k : J
    g : Quiver.Hom k j
    ⊢ ∀ (a : Set (F.obj j)), Membership.mem (Set.range fun f => Set.range (F.map f …
  -/
  rintro _ ⟨⟨k', g'⟩, rfl⟩ hl
  /-
    case intro.intro.intro.intro.intro.intro.intro.mk
    J : Type u
    inst✝¹ : CategoryTheory.Category.{u_1, u} J
    F : CategoryTheory.Functor J (Type v)
    inst✝ : CategoryTheory.IsCofilteredOrEmpty J
    h : ∀ (j : J), Exists fun i => Exists fun f => (Set.range (F.map f)).Finite
    j i✝ : J
    hi : Quiver.Hom i✝ j
    hf : (Set.range (F.map hi)).Finite
    m : Finset (F.obj j)
    hmin : ∀ (x : Finset (F.obj j)), Membership.mem (setOf fun s => Exists fun i = …
    i : J
    f : Quiver.Hom i j
    hm : Eq (↑m) (Set.range (F.map f))
    k : J
    g : Quiver.Hom k j
    k' : J
    g' : Quiver.Hom k' j
    hl : LE.le ((fun f => Set.range (F.map f.snd)) ⟨k', g'⟩) ((fun f => Set.range  …
    ⊢ LE.le ((fun f => Set.range (F.map f.snd)) ⟨i, f⟩) ((fun f => Set.range (F.ma …
  -/
  refine (eq_of_le_of_not_lt hl ?_).ge
  /-
    case intro.intro.intro.intro.intro.intro.intro.mk
    J : Type u
    inst✝¹ : CategoryTheory.Category.{u_1, u} J
    F : CategoryTheory.Functor J (Type v)
    inst✝ : CategoryTheory.IsCofilteredOrEmpty J
    h : ∀ (j : J), Exists fun i => Exists fun f => (Set.range (F.map f)).Finite
    j i✝ : J
    hi : Quiver.Hom i✝ j
    hf : (Set.range (F.map hi)).Finite
    m : Finset (F.obj j)
    hmin : ∀ (x : Finset (F.obj j)), Membership.mem (setOf fun s => Exists fun i = …
    i : J
    f : Quiver.Hom i j
    hm : Eq (↑m) (Set.range (F.map f))
    k : J
    g : Quiver.Hom k j
    k' : J
    g' : Quiver.Hom k' j
    hl : LE.le ((fun f => Set.range (F.map f.snd)) ⟨k', g'⟩) ((fun f => Set.range  …
    ⊢ Not (LT.lt ((fun f => Set.range (F.map f.snd)) ⟨k', g'⟩) ((fun f => Set.rang …
  -/
  have := hmin _ ⟨k', g', (m.finite_toSet.subset <| hm.substr hl).coe_toFinset⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.mk
    J : Type u
    inst✝¹ : CategoryTheory.Category.{u_1, u} J
    F : CategoryTheory.Functor J (Type v)
    inst✝ : CategoryTheory.IsCofilteredOrEmpty J
    h : ∀ (j : J), Exists fun i => Exists fun f => (Set.range (F.map f)).Finite
    j i✝ : J
    hi : Quiver.Hom i✝ j
    hf : (Set.range (F.map hi)).Finite
    m : Finset (F.obj j)
    hmin : ∀ (x : Finset (F.obj j)), Membership.mem (setOf fun s => Exists fun i = …
    i : J
    f : Quiver.Hom i j
    hm : Eq (↑m) (Set.range (F.map f))
    k : J
    g : Quiver.Hom k j
    k' : J
    g' : Quiver.Hom k' j
    hl : LE.le ((fun f => Set.range (F.map f.snd)) ⟨k', g'⟩) ((fun f => Set.range  …
    this : Not (LT.lt ⋯.toFinset m)
    ⊢ Not (LT.lt ((fun f => Set.range (F.map f.snd)) ⟨k', g'⟩) ((fun f => Set.rang …
  -/
  rwa [Finset.lt_iff_ssubset, ← Finset.coe_ssubset, Set.Finite.coe_toFinset, hm] at this
  /-
    🎉 no goals
  -/


/-- The subfunctor of `F` obtained by restricting to the eventual range at each index. -/
@[simps]
def toEventualRanges : J ⥤ Type v where
  obj j := F.eventualRange j
  map f := (F.eventualRange_mapsTo f).restrict _ _ _
  map_id i := by
    #adaptation_note /--- nightly-2024-03-16: simp was
    simp (config := { unfoldPartialApp := true }) only [MapsTo.restrict, Subtype.map, F.map_id] -/
    /-
      J : Type u
      inst✝¹ : CategoryTheory.Category.{?u.24110, u} J
      F : CategoryTheory.Functor J (Type v)
      i✝ j k : J
      s : Set (F.obj i✝)
      inst✝ : CategoryTheory.IsCofilteredOrEmpty J
      i : J
      ⊢ Eq ({ obj := fun j => ↑(F.eventualRange j), map := fun {X Y} f => Set.MapsTo …
    -/
    simp only [MapsTo.restrict, Subtype.map_def, F.map_id]
    /-
      J : Type u
      inst✝¹ : CategoryTheory.Category.{?u.24110, u} J
      F : CategoryTheory.Functor J (Type v)
      i✝ j k : J
      s : Set (F.obj i✝)
      inst✝ : CategoryTheory.IsCofilteredOrEmpty J
      i : J
      ⊢ Eq (fun x => ⟨CategoryTheory.CategoryStruct.id (F.obj i) ↑x, ⋯⟩) (CategoryTh …
    -/
    ext
    /-
      case h.a
      J : Type u
      inst✝¹ : CategoryTheory.Category.{?u.24110, u} J
      F : CategoryTheory.Functor J (Type v)
      i✝ j k : J
      s : Set (F.obj i✝)
      inst✝ : CategoryTheory.IsCofilteredOrEmpty J
      i : J
      a✝ : Subtype fun x => Membership.mem (F.eventualRange i) x
      ⊢ Eq ↑⟨CategoryTheory.CategoryStruct.id (F.obj i) ↑a✝, ⋯⟩ ↑(CategoryTheory.Cat …
    -/
    rfl
    /-
      🎉 no goals
    -/
  map_comp _ _ := by
    #adaptation_note /-- nightly-2024-03-16: simp was
    simp (config := { unfoldPartialApp := true }) only [MapsTo.restrict, Subtype.map, F.map_comp] -/
    /-
      J : Type u
      inst✝¹ : CategoryTheory.Category.{?u.24110, u} J
      F : CategoryTheory.Functor J (Type v)
      i j k : J
      s : Set (F.obj i)
      inst✝ : CategoryTheory.IsCofilteredOrEmpty J
      X✝ Y✝ Z✝ : J
      x✝¹ : Quiver.Hom X✝ Y✝
      x✝ : Quiver.Hom Y✝ Z✝
      ⊢ Eq ({ obj := fun j => ↑(F.eventualRange j), map := fun {X Y} f => Set.MapsTo …
    -/
    simp only [MapsTo.restrict, Subtype.map_def, F.map_comp]
    /-
      J : Type u
      inst✝¹ : CategoryTheory.Category.{?u.24110, u} J
      F : CategoryTheory.Functor J (Type v)
      i j k : J
      s : Set (F.obj i)
      inst✝ : CategoryTheory.IsCofilteredOrEmpty J
      X✝ Y✝ Z✝ : J
      x✝¹ : Quiver.Hom X✝ Y✝
      x✝ : Quiver.Hom Y✝ Z✝
      ⊢ Eq (fun x => ⟨CategoryTheory.CategoryStruct.comp (F.map x✝¹) (F.map x✝) ↑x,  …
    -/
    rfl
    /-
      🎉 no goals
    -/


instance toEventualRanges_finite [∀ j, Finite (F.obj j)] : ∀ j, Finite (F.toEventualRanges.obj j) :=
  fun _ => Subtype.finite


/-- The sections of the functor `F : J ⥤ Type v` are in bijection with the sections of
`F.toEventualRanges`. -/
def toEventualRangesSectionsEquiv : F.toEventualRanges.sections ≃ F.sections where
  toFun s := ⟨_, fun f => Subtype.coe_inj.2 <| s.prop f⟩
  invFun s :=
    ⟨fun _ => ⟨_, mem_iInter₂.2 fun _ f => ⟨_, s.prop f⟩⟩, fun f => Subtype.ext <| s.prop f⟩
  left_inv _ := by
    /-
      J : Type u
      inst✝¹ : CategoryTheory.Category.{?u.26340, u} J
      F : CategoryTheory.Functor J (Type v)
      i j k : J
      s : Set (F.obj i)
      inst✝ : CategoryTheory.IsCofilteredOrEmpty J
      x✝ : ↑F.toEventualRanges.sections
      ⊢ Eq ((fun s => ⟨fun x => ⟨↑s x, ⋯⟩, ⋯⟩) ((fun s => ⟨fun {j} => ↑(↑s j), ⋯⟩) x …
    -/
    ext
    /-
      case a.h
      J : Type u
      inst✝¹ : CategoryTheory.Category.{?u.26340, u} J
      F : CategoryTheory.Functor J (Type v)
      i j k : J
      s : Set (F.obj i)
      inst✝ : CategoryTheory.IsCofilteredOrEmpty J
      x✝¹ : ↑F.toEventualRanges.sections
      x✝ : J
      ⊢ Eq (↑((fun s => ⟨fun x => ⟨↑s x, ⋯⟩, ⋯⟩) ((fun s => ⟨fun {j} => ↑(↑s j), ⋯⟩) …
    -/
    rfl
    /-
      🎉 no goals
    -/
  right_inv _ := by
    /-
      J : Type u
      inst✝¹ : CategoryTheory.Category.{?u.26340, u} J
      F : CategoryTheory.Functor J (Type v)
      i j k : J
      s : Set (F.obj i)
      inst✝ : CategoryTheory.IsCofilteredOrEmpty J
      x✝ : ↑F.sections
      ⊢ Eq ((fun s => ⟨fun {j} => ↑(↑s j), ⋯⟩) ((fun s => ⟨fun x => ⟨↑s x, ⋯⟩, ⋯⟩) x …
    -/
    ext
    /-
      case a.h
      J : Type u
      inst✝¹ : CategoryTheory.Category.{?u.26340, u} J
      F : CategoryTheory.Functor J (Type v)
      i j k : J
      s : Set (F.obj i)
      inst✝ : CategoryTheory.IsCofilteredOrEmpty J
      x✝¹ : ↑F.sections
      x✝ : J
      ⊢ Eq (↑((fun s => ⟨fun {j} => ↑(↑s j), ⋯⟩) ((fun s => ⟨fun x => ⟨↑s x, ⋯⟩, ⋯⟩) …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- If `F` satisfies the Mittag-Leffler condition, its restriction to eventual ranges is a
surjective functor. -/
theorem surjective_toEventualRanges (h : F.IsMittagLeffler) ⦃i j⦄ (f : i ⟶ j) :
    (F.toEventualRanges.map f).Surjective := fun ⟨x, hx⟩ => by
  /-
    J : Type u
    inst✝¹ : CategoryTheory.Category.{u_1, u} J
    F : CategoryTheory.Functor J (Type v)
    inst✝ : CategoryTheory.IsCofilteredOrEmpty J
    h : F.IsMittagLeffler
    i j : J
    f : Quiver.Hom i j
    x✝ : F.toEventualRanges.obj j
    x : F.obj j
    hx : Membership.mem (F.eventualRange j) x
    ⊢ Exists fun a => Eq (F.toEventualRanges.map f a) ⟨x, hx⟩
  -/
  obtain ⟨y, hy, rfl⟩ := h.subset_image_eventualRange F f hx
  /-
    case intro.intro
    J : Type u
    inst✝¹ : CategoryTheory.Category.{u_1, u} J
    F : CategoryTheory.Functor J (Type v)
    inst✝ : CategoryTheory.IsCofilteredOrEmpty J
    h : F.IsMittagLeffler
    i j : J
    f : Quiver.Hom i j
    x✝ : F.toEventualRanges.obj j
    y : F.obj i
    hy : Membership.mem (F.eventualRange i) y
    hx : Membership.mem (F.eventualRange j) (F.map f y)
    ⊢ Exists fun a => Eq (F.toEventualRanges.map f a) ⟨F.map f y, hx⟩
  -/
  exact ⟨⟨y, hy⟩, rfl⟩
  /-
    🎉 no goals
  -/


/-- If `F` is nonempty at each index and Mittag-Leffler, then so is `F.toEventualRanges`. -/
theorem toEventualRanges_nonempty (h : F.IsMittagLeffler) [∀ j : J, Nonempty (F.obj j)] (j : J) :
    Nonempty (F.toEventualRanges.obj j) := by
  /-
    J : Type u
    inst✝² : CategoryTheory.Category.{u_1, u} J
    F : CategoryTheory.Functor J (Type v)
    inst✝¹ : CategoryTheory.IsCofilteredOrEmpty J
    h : F.IsMittagLeffler
    inst✝ : ∀ (j : J), Nonempty (F.obj j)
    j : J
    ⊢ Nonempty (F.toEventualRanges.obj j)
  -/
  let ⟨i, f, h⟩ := F.isMittagLeffler_iff_eventualRange.1 h j
  /-
    J : Type u
    inst✝² : CategoryTheory.Category.{u_1, u} J
    F : CategoryTheory.Functor J (Type v)
    inst✝¹ : CategoryTheory.IsCofilteredOrEmpty J
    h✝ : F.IsMittagLeffler
    inst✝ : ∀ (j : J), Nonempty (F.obj j)
    j i : J
    f : Quiver.Hom i j
    h : Eq (F.eventualRange j) (Set.range (F.map f))
    ⊢ Nonempty (F.toEventualRanges.obj j)
  -/
  rw [toEventualRanges_obj, h]
  /-
    J : Type u
    inst✝² : CategoryTheory.Category.{u_1, u} J
    F : CategoryTheory.Functor J (Type v)
    inst✝¹ : CategoryTheory.IsCofilteredOrEmpty J
    h✝ : F.IsMittagLeffler
    inst✝ : ∀ (j : J), Nonempty (F.obj j)
    j i : J
    f : Quiver.Hom i j
    h : Eq (F.eventualRange j) (Set.range (F.map f))
    ⊢ Nonempty ↑(Set.range (F.map f))
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- If `F` has all arrows surjective, then it "factors through a poset". -/
theorem thin_diagram_of_surjective (Fsur : ∀ ⦃i j : J⦄ (f : i ⟶ j), (F.map f).Surjective) {i j}
    (f g : i ⟶ j) : F.map f = F.map g :=
  let ⟨k, φ, hφ⟩ := IsCofilteredOrEmpty.cone_maps f g
                                      /-
                                        J : Type u
                                        inst✝¹ : CategoryTheory.Category.{u_1, u} J
                                        F : CategoryTheory.Functor J (Type v)
                                        inst✝ : CategoryTheory.IsCofilteredOrEmpty J
                                        Fsur : ∀ ⦃i j : J⦄ (f : Quiver.Hom i j), Function.Surjective (F.map f)
                                        i j : J
                                        f g : Quiver.Hom i j
                                        k : J
                                        φ : Quiver.Hom k i
                                        hφ : Eq (CategoryTheory.CategoryStruct.comp φ f) (CategoryTheory.CategoryStruc …
                                        ⊢ Eq ((fun g => Function.comp g (F.map φ)) (F.map f)) ((fun g => Function.comp …
                                      -/
  (Fsur φ).injective_comp_right <| by simp_rw [← types_comp, ← F.map_comp, hφ]
                                      /-
                                        🎉 no goals
                                      -/


theorem toPreimages_nonempty_of_surjective [hFn : ∀ j : J, Nonempty (F.obj j)]
    (Fsur : ∀ ⦃i j : J⦄ (f : i ⟶ j), (F.map f).Surjective) (hs : s.Nonempty) (j) :
    Nonempty ((F.toPreimages s).obj j) := by
  /-
    J : Type u
    inst✝¹ : CategoryTheory.Category.{u_1, u} J
    F : CategoryTheory.Functor J (Type v)
    i : J
    s : Set (F.obj i)
    inst✝ : CategoryTheory.IsCofilteredOrEmpty J
    hFn : ∀ (j : J), Nonempty (F.obj j)
    Fsur : ∀ ⦃i j : J⦄ (f : Quiver.Hom i j), Function.Surjective (F.map f)
    hs : s.Nonempty
    j : J
    ⊢ Nonempty ((F.toPreimages s).obj j)
  -/
  simp only [toPreimages_obj, nonempty_coe_sort, nonempty_iInter, mem_preimage]
  /-
    J : Type u
    inst✝¹ : CategoryTheory.Category.{u_1, u} J
    F : CategoryTheory.Functor J (Type v)
    i : J
    s : Set (F.obj i)
    inst✝ : CategoryTheory.IsCofilteredOrEmpty J
    hFn : ∀ (j : J), Nonempty (F.obj j)
    Fsur : ∀ ⦃i j : J⦄ (f : Quiver.Hom i j), Function.Surjective (F.map f)
    hs : s.Nonempty
    j : J
    ⊢ Exists fun x => ∀ (i_1 : Quiver.Hom j i), Membership.mem s (F.map i_1 x)
  -/
  obtain h | ⟨⟨ji⟩⟩ := isEmpty_or_nonempty (j ⟶ i)
    /-
      case inl
      J : Type u
      inst✝¹ : CategoryTheory.Category.{u_1, u} J
      F : CategoryTheory.Functor J (Type v)
      i : J
      s : Set (F.obj i)
      inst✝ : CategoryTheory.IsCofilteredOrEmpty J
      hFn : ∀ (j : J), Nonempty (F.obj j)
      Fsur : ∀ ⦃i j : J⦄ (f : Quiver.Hom i j), Function.Surjective (F.map f)
      hs : s.Nonempty
      j : J
      h : IsEmpty (Quiver.Hom j i)
      ⊢ Exists fun x => ∀ (i_1 : Quiver.Hom j i), Membership.mem s (F.map i_1 x)
    -/
  · exact ⟨(hFn j).some, fun ji => h.elim ji⟩
    /-
      🎉 no goals
    -/
    /-
      case inr.intro
      J : Type u
      inst✝¹ : CategoryTheory.Category.{u_1, u} J
      F : CategoryTheory.Functor J (Type v)
      i : J
      s : Set (F.obj i)
      inst✝ : CategoryTheory.IsCofilteredOrEmpty J
      hFn : ∀ (j : J), Nonempty (F.obj j)
      Fsur : ∀ ⦃i j : J⦄ (f : Quiver.Hom i j), Function.Surjective (F.map f)
      hs : s.Nonempty
      j : J
      ji : Quiver.Hom j i
      ⊢ Exists fun x => ∀ (i_1 : Quiver.Hom j i), Membership.mem s (F.map i_1 x)
    -/
  · obtain ⟨y, ys⟩ := hs
    /-
      case inr.intro.intro
      J : Type u
      inst✝¹ : CategoryTheory.Category.{u_1, u} J
      F : CategoryTheory.Functor J (Type v)
      i : J
      s : Set (F.obj i)
      inst✝ : CategoryTheory.IsCofilteredOrEmpty J
      hFn : ∀ (j : J), Nonempty (F.obj j)
      Fsur : ∀ ⦃i j : J⦄ (f : Quiver.Hom i j), Function.Surjective (F.map f)
      j : J
      ji : Quiver.Hom j i
      y : F.obj i
      ys : Membership.mem s y
      ⊢ Exists fun x => ∀ (i_1 : Quiver.Hom j i), Membership.mem s (F.map i_1 x)
    -/
    obtain ⟨x, rfl⟩ := Fsur ji y
    /-
      case inr.intro.intro.intro
      J : Type u
      inst✝¹ : CategoryTheory.Category.{u_1, u} J
      F : CategoryTheory.Functor J (Type v)
      i : J
      s : Set (F.obj i)
      inst✝ : CategoryTheory.IsCofilteredOrEmpty J
      hFn : ∀ (j : J), Nonempty (F.obj j)
      Fsur : ∀ ⦃i j : J⦄ (f : Quiver.Hom i j), Function.Surjective (F.map f)
      j : J
      ji : Quiver.Hom j i
      x : F.obj j
      ys : Membership.mem s (F.map ji x)
      ⊢ Exists fun x => ∀ (i_1 : Quiver.Hom j i), Membership.mem s (F.map i_1 x)
    -/
    exact ⟨x, fun ji' => (F.thin_diagram_of_surjective Fsur ji' ji).symm ▸ ys⟩
    /-
      🎉 no goals
    -/


theorem eval_section_injective_of_eventually_injective {j}
    (Finj : ∀ (i) (f : i ⟶ j), (F.map f).Injective) (i) (f : i ⟶ j) :
    (fun s : F.sections => s.val j).Injective := by
  /-
    J : Type u
    inst✝¹ : CategoryTheory.Category.{u_1, u} J
    F : CategoryTheory.Functor J (Type v)
    inst✝ : CategoryTheory.IsCofilteredOrEmpty J
    j : J
    Finj : ∀ (i : J) (f : Quiver.Hom i j), Function.Injective (F.map f)
    i : J
    f : Quiver.Hom i j
    ⊢ Function.Injective fun s => ↑s j
  -/
  refine fun s₀ s₁ h => Subtype.ext <| funext fun k => ?_
  /-
    J : Type u
    inst✝¹ : CategoryTheory.Category.{u_1, u} J
    F : CategoryTheory.Functor J (Type v)
    inst✝ : CategoryTheory.IsCofilteredOrEmpty J
    j : J
    Finj : ∀ (i : J) (f : Quiver.Hom i j), Function.Injective (F.map f)
    i : J
    f : Quiver.Hom i j
    s₀ s₁ : ↑F.sections
    h : Eq ((fun s => ↑s j) s₀) ((fun s => ↑s j) s₁)
    k : J
    ⊢ Eq (↑s₀ k) (↑s₁ k)
  -/
  obtain ⟨m, mi, mk, _⟩ := IsCofilteredOrEmpty.cone_objs i k
  /-
    case intro.intro.intro
    J : Type u
    inst✝¹ : CategoryTheory.Category.{u_1, u} J
    F : CategoryTheory.Functor J (Type v)
    inst✝ : CategoryTheory.IsCofilteredOrEmpty J
    j : J
    Finj : ∀ (i : J) (f : Quiver.Hom i j), Function.Injective (F.map f)
    i : J
    f : Quiver.Hom i j
    s₀ s₁ : ↑F.sections
    h : Eq ((fun s => ↑s j) s₀) ((fun s => ↑s j) s₁)
    k m : J
    mi : Quiver.Hom m i
    mk : Quiver.Hom m k
    h✝ : True
    ⊢ Eq (↑s₀ k) (↑s₁ k)
  -/
  dsimp at h
  /-
    case intro.intro.intro
    J : Type u
    inst✝¹ : CategoryTheory.Category.{u_1, u} J
    F : CategoryTheory.Functor J (Type v)
    inst✝ : CategoryTheory.IsCofilteredOrEmpty J
    j : J
    Finj : ∀ (i : J) (f : Quiver.Hom i j), Function.Injective (F.map f)
    i : J
    f : Quiver.Hom i j
    s₀ s₁ : ↑F.sections
    h : Eq (↑s₀ j) (↑s₁ j)
    k m : J
    mi : Quiver.Hom m i
    mk : Quiver.Hom m k
    h✝ : True
    ⊢ Eq (↑s₀ k) (↑s₁ k)
  -/
  rw [← s₀.prop (mi ≫ f), ← s₁.prop (mi ≫ f)] at h
  /-
    case intro.intro.intro
    J : Type u
    inst✝¹ : CategoryTheory.Category.{u_1, u} J
    F : CategoryTheory.Functor J (Type v)
    inst✝ : CategoryTheory.IsCofilteredOrEmpty J
    j : J
    Finj : ∀ (i : J) (f : Quiver.Hom i j), Function.Injective (F.map f)
    i : J
    f : Quiver.Hom i j
    s₀ s₁ : ↑F.sections
    k m : J
    mi : Quiver.Hom m i
    h : Eq (F.map (CategoryTheory.CategoryStruct.comp mi f) (↑s₀ m)) (F.map (Categ …
    mk : Quiver.Hom m k
    h✝ : True
    ⊢ Eq (↑s₀ k) (↑s₁ k)
  -/
  rw [← s₀.prop mk, ← s₁.prop mk]
  /-
    case intro.intro.intro
    J : Type u
    inst✝¹ : CategoryTheory.Category.{u_1, u} J
    F : CategoryTheory.Functor J (Type v)
    inst✝ : CategoryTheory.IsCofilteredOrEmpty J
    j : J
    Finj : ∀ (i : J) (f : Quiver.Hom i j), Function.Injective (F.map f)
    i : J
    f : Quiver.Hom i j
    s₀ s₁ : ↑F.sections
    k m : J
    mi : Quiver.Hom m i
    h : Eq (F.map (CategoryTheory.CategoryStruct.comp mi f) (↑s₀ m)) (F.map (Categ …
    mk : Quiver.Hom m k
    h✝ : True
    ⊢ Eq (F.map mk (↑s₀ m)) (F.map mk (↑s₁ m))
  -/
  exact congr_arg _ (Finj m (mi ≫ f) h)
  /-
    🎉 no goals
  -/


theorem eval_section_surjective_of_surjective (i : J) :
    (fun s : F.sections => s.val i).Surjective := fun x => by
  /-
    J : Type u
    inst✝³ : CategoryTheory.Category.{u_1, u} J
    F : CategoryTheory.Functor J (Type v)
    inst✝² : CategoryTheory.IsCofilteredOrEmpty J
    inst✝¹ : ∀ (j : J), Nonempty (F.obj j)
    inst✝ : ∀ (j : J), Finite (F.obj j)
    Fsur : ∀ ⦃i j : J⦄ (f : Quiver.Hom i j), Function.Surjective (F.map f)
    i : J
    x : F.obj i
    ⊢ Exists fun a => Eq ((fun s => ↑s i) a) x
  -/
  let s : Set (F.obj i) := {x}
  /-
    J : Type u
    inst✝³ : CategoryTheory.Category.{u_1, u} J
    F : CategoryTheory.Functor J (Type v)
    inst✝² : CategoryTheory.IsCofilteredOrEmpty J
    inst✝¹ : ∀ (j : J), Nonempty (F.obj j)
    inst✝ : ∀ (j : J), Finite (F.obj j)
    Fsur : ∀ ⦃i j : J⦄ (f : Quiver.Hom i j), Function.Surjective (F.map f)
    i : J
    x : F.obj i
    s : Set (F.obj i) := Singleton.singleton x
    ⊢ Exists fun a => Eq ((fun s => ↑s i) a) x
  -/
  haveI := F.toPreimages_nonempty_of_surjective s Fsur (singleton_nonempty x)
  /-
    J : Type u
    inst✝³ : CategoryTheory.Category.{u_1, u} J
    F : CategoryTheory.Functor J (Type v)
    inst✝² : CategoryTheory.IsCofilteredOrEmpty J
    inst✝¹ : ∀ (j : J), Nonempty (F.obj j)
    inst✝ : ∀ (j : J), Finite (F.obj j)
    Fsur : ∀ ⦃i j : J⦄ (f : Quiver.Hom i j), Function.Surjective (F.map f)
    i : J
    x : F.obj i
    s : Set (F.obj i) := Singleton.singleton x
    this : ∀ (j : J), Nonempty ((F.toPreimages s).obj j)
    ⊢ Exists fun a => Eq ((fun s => ↑s i) a) x
  -/
  obtain ⟨sec, h⟩ := nonempty_sections_of_finite_cofiltered_system (F.toPreimages s)
  /-
    case intro
    J : Type u
    inst✝³ : CategoryTheory.Category.{u_1, u} J
    F : CategoryTheory.Functor J (Type v)
    inst✝² : CategoryTheory.IsCofilteredOrEmpty J
    inst✝¹ : ∀ (j : J), Nonempty (F.obj j)
    inst✝ : ∀ (j : J), Finite (F.obj j)
    Fsur : ∀ ⦃i j : J⦄ (f : Quiver.Hom i j), Function.Surjective (F.map f)
    i : J
    x : F.obj i
    s : Set (F.obj i) := Singleton.singleton x
    this : ∀ (j : J), Nonempty ((F.toPreimages s).obj j)
    sec : (j : J) → (F.toPreimages s).obj j
    h : Membership.mem (F.toPreimages s).sections sec
    ⊢ Exists fun a => Eq ((fun s => ↑s i) a) x
  -/
  refine ⟨⟨fun j => (sec j).val, fun jk => by simpa [Subtype.ext_iff] using h jk⟩, ?_⟩
    /-
      case intro
      J : Type u
      inst✝³ : CategoryTheory.Category.{u_1, u} J
      F : CategoryTheory.Functor J (Type v)
      inst✝² : CategoryTheory.IsCofilteredOrEmpty J
      inst✝¹ : ∀ (j : J), Nonempty (F.obj j)
      inst✝ : ∀ (j : J), Finite (F.obj j)
      Fsur : ∀ ⦃i j : J⦄ (f : Quiver.Hom i j), Function.Surjective (F.map f)
      i : J
      x : F.obj i
      s : Set (F.obj i) := Singleton.singleton x
      this : ∀ (j : J), Nonempty ((F.toPreimages s).obj j)
      sec : (j : J) → (F.toPreimages s).obj j
      h : Membership.mem (F.toPreimages s).sections sec
      ⊢ Eq ((fun s => ↑s i) ⟨fun j => ↑(sec j), ⋯⟩) x
    -/
  · have := (sec i).prop
    /-
      case intro
      J : Type u
      inst✝³ : CategoryTheory.Category.{u_1, u} J
      F : CategoryTheory.Functor J (Type v)
      inst✝² : CategoryTheory.IsCofilteredOrEmpty J
      inst✝¹ : ∀ (j : J), Nonempty (F.obj j)
      inst✝ : ∀ (j : J), Finite (F.obj j)
      Fsur : ∀ ⦃i j : J⦄ (f : Quiver.Hom i j), Function.Surjective (F.map f)
      i : J
      x : F.obj i
      s : Set (F.obj i) := Singleton.singleton x
      this✝ : ∀ (j : J), Nonempty ((F.toPreimages s).obj j)
      sec : (j : J) → (F.toPreimages s).obj j
      h : Membership.mem (F.toPreimages s).sections sec
      this : Membership.mem (Set.iInter fun f => Set.preimage (F.map f) s) ↑(sec i)
      ⊢ Eq ((fun s => ↑s i) ⟨fun j => ↑(sec j), ⋯⟩) x
    -/
    simp only [mem_iInter, mem_preimage, mem_singleton_iff] at this
    /-
      case intro
      J : Type u
      inst✝³ : CategoryTheory.Category.{u_1, u} J
      F : CategoryTheory.Functor J (Type v)
      inst✝² : CategoryTheory.IsCofilteredOrEmpty J
      inst✝¹ : ∀ (j : J), Nonempty (F.obj j)
      inst✝ : ∀ (j : J), Finite (F.obj j)
      Fsur : ∀ ⦃i j : J⦄ (f : Quiver.Hom i j), Function.Surjective (F.map f)
      i : J
      x : F.obj i
      s : Set (F.obj i) := Singleton.singleton x
      this✝ : ∀ (j : J), Nonempty ((F.toPreimages s).obj j)
      sec : (j : J) → (F.toPreimages s).obj j
      h : Membership.mem (F.toPreimages s).sections sec
      this : ∀ (i_1 : Quiver.Hom i i), Membership.mem s (F.map i_1 ↑(sec i))
      ⊢ Eq ((fun s => ↑s i) ⟨fun j => ↑(sec j), ⋯⟩) x
    -/
    have := this (𝟙 i)
    /-
      case intro
      J : Type u
      inst✝³ : CategoryTheory.Category.{u_1, u} J
      F : CategoryTheory.Functor J (Type v)
      inst✝² : CategoryTheory.IsCofilteredOrEmpty J
      inst✝¹ : ∀ (j : J), Nonempty (F.obj j)
      inst✝ : ∀ (j : J), Finite (F.obj j)
      Fsur : ∀ ⦃i j : J⦄ (f : Quiver.Hom i j), Function.Surjective (F.map f)
      i : J
      x : F.obj i
      s : Set (F.obj i) := Singleton.singleton x
      this✝¹ : ∀ (j : J), Nonempty ((F.toPreimages s).obj j)
      sec : (j : J) → (F.toPreimages s).obj j
      h : Membership.mem (F.toPreimages s).sections sec
      this✝ : ∀ (i_1 : Quiver.Hom i i), Membership.mem s (F.map i_1 ↑(sec i))
      this : Membership.mem s (F.map (CategoryTheory.CategoryStruct.id i) ↑(sec i))
      ⊢ Eq ((fun s => ↑s i) ⟨fun j => ↑(sec j), ⋯⟩) x
    -/
    rwa [map_id_apply] at this
    /-
      🎉 no goals
    -/


theorem eventually_injective [Nonempty J] [Finite F.sections] :
    ∃ j, ∀ (i) (f : i ⟶ j), (F.map f).Injective := by
  /-
    J : Type u
    inst✝⁵ : CategoryTheory.Category.{u_1, u} J
    F : CategoryTheory.Functor J (Type v)
    inst✝⁴ : CategoryTheory.IsCofilteredOrEmpty J
    inst✝³ : ∀ (j : J), Nonempty (F.obj j)
    inst✝² : ∀ (j : J), Finite (F.obj j)
    Fsur : ∀ ⦃i j : J⦄ (f : Quiver.Hom i j), Function.Surjective (F.map f)
    inst✝¹ : Nonempty J
    inst✝ : Finite ↑F.sections
    ⊢ Exists fun j => ∀ (i : J) (f : Quiver.Hom i j), Function.Injective (F.map f)
  -/
  haveI : ∀ j, Fintype (F.obj j) := fun j => Fintype.ofFinite (F.obj j)
  /-
    J : Type u
    inst✝⁵ : CategoryTheory.Category.{u_1, u} J
    F : CategoryTheory.Functor J (Type v)
    inst✝⁴ : CategoryTheory.IsCofilteredOrEmpty J
    inst✝³ : ∀ (j : J), Nonempty (F.obj j)
    inst✝² : ∀ (j : J), Finite (F.obj j)
    Fsur : ∀ ⦃i j : J⦄ (f : Quiver.Hom i j), Function.Surjective (F.map f)
    inst✝¹ : Nonempty J
    inst✝ : Finite ↑F.sections
    this : (j : J) → Fintype (F.obj j)
    ⊢ Exists fun j => ∀ (i : J) (f : Quiver.Hom i j), Function.Injective (F.map f)
  -/
  haveI : Fintype F.sections := Fintype.ofFinite F.sections
  have card_le : ∀ j, Fintype.card (F.obj j) ≤ Fintype.card F.sections :=
    fun j => Fintype.card_le_of_surjective _ (F.eval_section_surjective_of_surjective Fsur j)
  /-
    J : Type u
    inst✝⁵ : CategoryTheory.Category.{u_1, u} J
    F : CategoryTheory.Functor J (Type v)
    inst✝⁴ : CategoryTheory.IsCofilteredOrEmpty J
    inst✝³ : ∀ (j : J), Nonempty (F.obj j)
    inst✝² : ∀ (j : J), Finite (F.obj j)
    Fsur : ∀ ⦃i j : J⦄ (f : Quiver.Hom i j), Function.Surjective (F.map f)
    inst✝¹ : Nonempty J
    inst✝ : Finite ↑F.sections
    this✝ : (j : J) → Fintype (F.obj j)
    this : Fintype ↑F.sections
    card_le : ∀ (j : J), LE.le (Fintype.card (F.obj j)) (Fintype.card ↑F.sections)
    ⊢ Exists fun j => ∀ (i : J) (f : Quiver.Hom i j), Function.Injective (F.map f)
  -/
  let fn j := Fintype.card F.sections - Fintype.card (F.obj j)
  refine ⟨fn.argmin Nat.lt_wfRel.wf,
    fun i f => ((Fintype.bijective_iff_surjective_and_card _).2
      ⟨Fsur f, le_antisymm ?_ (Fintype.card_le_of_surjective _ <| Fsur f)⟩).1⟩
  /-
    J : Type u
    inst✝⁵ : CategoryTheory.Category.{u_1, u} J
    F : CategoryTheory.Functor J (Type v)
    inst✝⁴ : CategoryTheory.IsCofilteredOrEmpty J
    inst✝³ : ∀ (j : J), Nonempty (F.obj j)
    inst✝² : ∀ (j : J), Finite (F.obj j)
    Fsur : ∀ ⦃i j : J⦄ (f : Quiver.Hom i j), Function.Surjective (F.map f)
    inst✝¹ : Nonempty J
    inst✝ : Finite ↑F.sections
    this✝ : (j : J) → Fintype (F.obj j)
    this : Fintype ↑F.sections
    card_le : ∀ (j : J), LE.le (Fintype.card (F.obj j)) (Fintype.card ↑F.sections)
    fn : J → Nat := fun j => HSub.hSub (Fintype.card ↑F.sections) (Fintype.card (F …
    i : J
    f : Quiver.Hom i (Function.argmin fn ⋯)
    ⊢ LE.le (Fintype.card (F.obj i)) (Fintype.card (F.obj (Function.argmin fn ⋯)))
  -/
  rw [← Nat.sub_le_sub_iff_left (card_le i)]
  /-
    J : Type u
    inst✝⁵ : CategoryTheory.Category.{u_1, u} J
    F : CategoryTheory.Functor J (Type v)
    inst✝⁴ : CategoryTheory.IsCofilteredOrEmpty J
    inst✝³ : ∀ (j : J), Nonempty (F.obj j)
    inst✝² : ∀ (j : J), Finite (F.obj j)
    Fsur : ∀ ⦃i j : J⦄ (f : Quiver.Hom i j), Function.Surjective (F.map f)
    inst✝¹ : Nonempty J
    inst✝ : Finite ↑F.sections
    this✝ : (j : J) → Fintype (F.obj j)
    this : Fintype ↑F.sections
    card_le : ∀ (j : J), LE.le (Fintype.card (F.obj j)) (Fintype.card ↑F.sections)
    fn : J → Nat := fun j => HSub.hSub (Fintype.card ↑F.sections) (Fintype.card (F …
    i : J
    f : Quiver.Hom i (Function.argmin fn ⋯)
    ⊢ LE.le (HSub.hSub (Fintype.card ↑F.sections) (Fintype.card (F.obj (Function.a …
  -/
  apply fn.argmin_le
  /-
    🎉 no goals
  -/


