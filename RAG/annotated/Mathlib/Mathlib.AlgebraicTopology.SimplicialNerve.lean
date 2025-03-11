/-- A type synonym for a linear order `J`, will be equipped with a simplicial category structure. -/
@[nolint unusedArguments]
def SimplicialThickening (J : Type*) [LinearOrder J] : Type _ := J


instance (J : Type*) [LinearOrder J] : LinearOrder (SimplicialThickening J) :=
  inferInstanceAs (LinearOrder J)


/--
A path from `i` to `j` in a linear order `J` is a subset of the interval `[i, j]` in `J` containing
the endpoints.
-/
@[ext]
structure Path {J : Type*} [LinearOrder J] (i j : J) where
  /-- The underlying subset -/
  I : Set J
  left : i ∈ I := by simp
  right : j ∈ I := by simp
  left_le (k : J) (_ : k ∈ I) : i ≤ k := by simp
  le_right (k : J) (_ : k ∈ I) : k ≤ j := by simp


lemma Path.le {J : Type*} [LinearOrder J] {i j : J} (f : Path i j) : i ≤ j :=
  f.left_le _ f.right


instance {J : Type*} [LinearOrder J] (i j : J) : Category (Path i j) :=
  InducedCategory.category (fun f : Path i j ↦ f.I)


@[simps]
instance (J : Type*) [LinearOrder J] : CategoryStruct (SimplicialThickening J) where
  Hom i j := Path i j
  id i := { I := {i} }
  comp {i j k} f g := {
    I := f.I ∪ g.I
    left := Or.inl f.left
    right := Or.inr g.right
    left_le l := by
      /-
        J : Type u_1
        inst✝ : LinearOrder J
        i j k : CategoryTheory.SimplicialThickening J
        f : Quiver.Hom i j
        g : Quiver.Hom j k
        l : CategoryTheory.SimplicialThickening J
        ⊢ Membership.mem (Union.union f.I g.I) l → LE.le i l
      -/
      rintro (h | h)
      /-
        case inl
        J : Type u_1
        inst✝ : LinearOrder J
        i j k : CategoryTheory.SimplicialThickening J
        f : Quiver.Hom i j
        g : Quiver.Hom j k
        l : CategoryTheory.SimplicialThickening J
        h : Membership.mem f.I l
        ⊢ LE.le i l
      -/
      exacts [(f.left_le l h), (Path.le f).trans (g.left_le l h)]
      /-
        🎉 no goals
      -/
    le_right l := by
      /-
        J : Type u_1
        inst✝ : LinearOrder J
        i j k : CategoryTheory.SimplicialThickening J
        f : Quiver.Hom i j
        g : Quiver.Hom j k
        l : CategoryTheory.SimplicialThickening J
        ⊢ Membership.mem (Union.union f.I g.I) l → LE.le l k
      -/
      rintro (h | h)
      /-
        case inl
        J : Type u_1
        inst✝ : LinearOrder J
        i j k : CategoryTheory.SimplicialThickening J
        f : Quiver.Hom i j
        g : Quiver.Hom j k
        l : CategoryTheory.SimplicialThickening J
        h : Membership.mem f.I l
        ⊢ LE.le l k
      -/
      exacts [(f.le_right _ h).trans (Path.le g), (g.le_right l h)] }
      /-
        🎉 no goals
      -/


instance {J : Type*} [LinearOrder J] (i j : SimplicialThickening J) : Category (i ⟶ j) :=
  inferInstanceAs (Category (Path _ _))


@[ext]
lemma hom_ext {J : Type*} [LinearOrder J]
    (i j : SimplicialThickening J) (x y : i ⟶ j) (h : ∀ t, t ∈ x.I ↔ t ∈ y.I) : x = y := by
  /-
    J : Type u_1
    inst✝ : LinearOrder J
    i j : CategoryTheory.SimplicialThickening J
    x y : Quiver.Hom i j
    h : ∀ (t : CategoryTheory.SimplicialThickening J), Iff (Membership.mem x.I t)  …
    ⊢ Eq x y
  -/
  apply Path.ext
  /-
    case I
    J : Type u_1
    inst✝ : LinearOrder J
    i j : CategoryTheory.SimplicialThickening J
    x y : Quiver.Hom i j
    h : ∀ (t : CategoryTheory.SimplicialThickening J), Iff (Membership.mem x.I t)  …
    ⊢ Eq x.I y.I
  -/
  ext
  /-
    case I.h
    J : Type u_1
    inst✝ : LinearOrder J
    i j : CategoryTheory.SimplicialThickening J
    x y : Quiver.Hom i j
    h : ∀ (t : CategoryTheory.SimplicialThickening J), Iff (Membership.mem x.I t)  …
    x✝ : CategoryTheory.SimplicialThickening J
    ⊢ Iff (Membership.mem x.I x✝) (Membership.mem y.I x✝)
  -/
  apply h
  /-
    🎉 no goals
  -/


instance (J : Type*) [LinearOrder J] : Category (SimplicialThickening J) where
                  /-
                    J : Type u_1
                    inst✝ : LinearOrder J
                    X✝ Y✝ : CategoryTheory.SimplicialThickening J
                    f : Quiver.Hom X✝ Y✝
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id X✝) …
                  -/
  id_comp f := by ext; simpa using fun h ↦ h ▸ f.left
                       /-
                         🎉 no goals
                       -/
                  /-
                    J : Type u_1
                    inst✝ : LinearOrder J
                    X✝ Y✝ : CategoryTheory.SimplicialThickening J
                    f : Quiver.Hom X✝ Y✝
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.id Y …
                  -/
  comp_id f := by ext; simpa using fun h ↦ h ▸ f.right
                       /-
                         🎉 no goals
                       -/


/--
Composition of morphisms in `SimplicialThickening J`, as a functor `(i ⟶ j) × (j ⟶ k) ⥤ (i ⟶ k)`
-/
@[simps]
def compFunctor {J : Type*} [LinearOrder J]
    (i j k : SimplicialThickening J) : (i ⟶ j) × (j ⟶ k) ⥤ (i ⟶ k) where
  obj x := x.1 ≫ x.2
  map f := ⟨⟨Set.union_subset_union f.1.1.1 f.2.1.1⟩⟩


/-- The hom simplicial set of the simplicial category structure on `SimplicialThickening J` -/
abbrev Hom (i j : SimplicialThickening J) : SSet := (nerve (i ⟶ j))


/-- The identity of the simplicial category structure on `SimplicialThickening J` -/
abbrev id (i : SimplicialThickening J) : 𝟙_ SSet ⟶ Hom i i :=
                                                         /-
                                                           J : Type u_1
                                                           inst✝ : LinearOrder J
                                                           i : CategoryTheory.SimplicialThickening J
                                                           x✝² x✝¹ : Opposite SimplexCategory
                                                           x✝ : Quiver.Hom x✝² x✝¹
                                                           ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                                                         -/
  ⟨fun _ _ ↦ (Functor.const _).obj (𝟙 _), fun _ _ _ ↦ by simp; rfl⟩
                                                               /-
                                                                 🎉 no goals
                                                               -/


/-- The composition of the simplicial category structure on `SimplicialThickening J` -/
abbrev comp (i j k : SimplicialThickening J) : Hom i j ⊗ Hom j k ⟶ Hom i k :=
                                                               /-
                                                                 J : Type u_1
                                                                 inst✝ : LinearOrder J
                                                                 i j k : CategoryTheory.SimplicialThickening J
                                                                 x✝² x✝¹ : Opposite SimplexCategory
                                                                 x✝ : Quiver.Hom x✝² x✝¹
                                                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.MonoidalCategoryStru …
                                                               -/
  ⟨fun _ x ↦ x.1.prod' x.2 ⋙ compFunctor i j k, fun _ _ _ ↦ by simp; rfl⟩
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[simp]
lemma id_comp (i j : SimplicialThickening J) :
    (λ_ (Hom i j)).inv ≫ id i ▷ Hom i j ≫ comp i i j = 𝟙 (Hom i j) := by
  /-
    J : Type u_1
    inst✝ : LinearOrder J
    i j : CategoryTheory.SimplicialThickening J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [Iso.inv_comp_eq]
  /-
    J : Type u_1
    inst✝ : LinearOrder J
    i j : CategoryTheory.SimplicialThickening J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  ext
  /-
    case w.h
    J : Type u_1
    inst✝ : LinearOrder J
    i j : CategoryTheory.SimplicialThickening J
    n✝ : Opposite SimplexCategory
    a✝ : (CategoryTheory.MonoidalCategoryStruct.tensorObj CategoryTheory.MonoidalC …
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStru …
  -/
  exact Functor.ext (fun _ ↦ by simp)
  /-
    🎉 no goals
  -/


@[simp]
lemma comp_id (i j : SimplicialThickening J) :
    (ρ_ (Hom i j)).inv ≫ Hom i j ◁ id j ≫ comp i j j = 𝟙 (Hom i j) := by
  /-
    J : Type u_1
    inst✝ : LinearOrder J
    i j : CategoryTheory.SimplicialThickening J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  rw [Iso.inv_comp_eq]
  /-
    J : Type u_1
    inst✝ : LinearOrder J
    i j : CategoryTheory.SimplicialThickening J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  ext
  /-
    case w.h
    J : Type u_1
    inst✝ : LinearOrder J
    i j : CategoryTheory.SimplicialThickening J
    n✝ : Opposite SimplexCategory
    a✝ : (CategoryTheory.MonoidalCategoryStruct.tensorObj (CategoryTheory.Simplici …
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStru …
  -/
  exact Functor.ext (fun _ ↦ by simp)
  /-
    🎉 no goals
  -/


@[simp]
lemma assoc (i j k l : SimplicialThickening J) :
    (α_ (Hom i j) (Hom j k) (Hom k l)).inv ≫ comp i j k ▷ Hom k l ≫ comp i k l =
      Hom i j ◁ comp j k l ≫ comp i j l := by
  /-
    J : Type u_1
    inst✝ : LinearOrder J
    i j k l : CategoryTheory.SimplicialThickening J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  ext
  /-
    case w.h
    J : Type u_1
    inst✝ : LinearOrder J
    i j k l : CategoryTheory.SimplicialThickening J
    n✝ : Opposite SimplexCategory
    a✝ : (CategoryTheory.MonoidalCategoryStruct.tensorObj (CategoryTheory.Simplici …
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStru …
  -/
  exact Functor.ext (fun _ ↦ by simp)
  /-
    🎉 no goals
  -/


noncomputable instance (J : Type*) [LinearOrder J] :
    SimplicialCategory (SimplicialThickening J) where
  Hom := Hom
  id := id
  comp := comp
  homEquiv {i j} := (nerveEquiv _).symm.trans (SSet.unitHomEquiv _).symm


/-- Auxiliary definition for `SimplicialThickening.functorMap` -/
def orderHom {J K : Type*} [LinearOrder J] [LinearOrder K] (f : J →o K) :
    SimplicialThickening J →o SimplicialThickening K := f


/-- Auxiliary definition for `SimplicialThickening.functor` -/
noncomputable abbrev functorMap {J K : Type u} [LinearOrder J] [LinearOrder K]
    (f : J →o K) (i j : SimplicialThickening J) : (i ⟶ j) ⥤ ((orderHom f i) ⟶ (orderHom f j)) where
  obj I := ⟨f '' I.I, Set.mem_image_of_mem f I.left, Set.mem_image_of_mem f I.right,
       /-
         J K : Type u
         inst✝¹ : LinearOrder J
         inst✝ : LinearOrder K
         f : OrderHom J K
         i j : CategoryTheory.SimplicialThickening J
         I : Quiver.Hom i j
         ⊢ ∀ (k : CategoryTheory.SimplicialThickening K), Membership.mem (Set.image (⇑f …
       -/
    by rintro _ ⟨k, hk, rfl⟩; exact f.monotone (I.left_le k hk),
                              /-
                                🎉 no goals
                              -/
       /-
         J K : Type u
         inst✝¹ : LinearOrder J
         inst✝ : LinearOrder K
         f : OrderHom J K
         i j : CategoryTheory.SimplicialThickening J
         I : Quiver.Hom i j
         ⊢ ∀ (k : CategoryTheory.SimplicialThickening K), Membership.mem (Set.image (⇑f …
       -/
    by rintro _ ⟨k, hk, rfl⟩; exact f.monotone (I.le_right k hk)⟩
                              /-
                                🎉 no goals
                              -/
  map f := ⟨⟨Set.image_subset _ f.1.1⟩⟩


/--
The simplicial thickening defines a functor from the category of linear orders to the category of
simplicial categories
-/
@[simps]
noncomputable def functor {J K : Type u} [LinearOrder J] [LinearOrder K]
    (f : J →o K) : EnrichedFunctor SSet (SimplicialThickening J) (SimplicialThickening K) where
  obj := f
  map i j := nerveMap ((functorMap f i j))
  map_id i := by
    /-
      J K : Type u
      inst✝¹ : LinearOrder J
      inst✝ : LinearOrder K
      f : OrderHom J K
      i : CategoryTheory.SimplicialThickening J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eId SSet i) ((fun i j …
    -/
    ext
    /-
      case w.h
      J K : Type u
      inst✝¹ : LinearOrder J
      inst✝ : LinearOrder K
      f : OrderHom J K
      i : CategoryTheory.SimplicialThickening J
      n✝ : Opposite SimplexCategory
      a✝ : CategoryTheory.MonoidalCategoryStruct.tensorUnit.obj n✝
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.eId SSet i) ((fun i  …
    -/
    simp only [eId, EnrichedCategory.id]
    /-
      case w.h
      J K : Type u
      inst✝¹ : LinearOrder J
      inst✝ : LinearOrder K
      f : OrderHom J K
      i : CategoryTheory.SimplicialThickening J
      n✝ : Opposite SimplexCategory
      a✝ : CategoryTheory.MonoidalCategoryStruct.tensorUnit.obj n✝
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.SimplicialThickening …
    -/
    exact Functor.ext (by aesop_cat)
    /-
      🎉 no goals
    -/
  map_comp i j k := by
    /-
      J K : Type u
      inst✝¹ : LinearOrder J
      inst✝ : LinearOrder K
      f : OrderHom J K
      i j k : CategoryTheory.SimplicialThickening J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.eComp SSet i j k) ((f …
    -/
    ext
    /-
      case w.h
      J K : Type u
      inst✝¹ : LinearOrder J
      inst✝ : LinearOrder K
      f : OrderHom J K
      i j k : CategoryTheory.SimplicialThickening J
      n✝ : Opposite SimplexCategory
      a✝ : (CategoryTheory.MonoidalCategoryStruct.tensorObj (CategoryTheory.Enriched …
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.eComp SSet i j k) (( …
    -/
    simp only [eComp, EnrichedCategory.comp]
    /-
      case w.h
      J K : Type u
      inst✝¹ : LinearOrder J
      inst✝ : LinearOrder K
      f : OrderHom J K
      i j k : CategoryTheory.SimplicialThickening J
      n✝ : Opposite SimplexCategory
      a✝ : (CategoryTheory.MonoidalCategoryStruct.tensorObj (CategoryTheory.Enriched …
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.SimplicialThickening …
    -/
    exact Functor.ext (by aesop_cat)
    /-
      🎉 no goals
    -/


lemma functor_id (J : Type u) [LinearOrder J] :
    (functor (OrderHom.id (α := J))) = EnrichedFunctor.id _ _ := by
  /-
    J : Type u
    inst✝ : LinearOrder J
    ⊢ Eq (CategoryTheory.SimplicialThickening.functor OrderHom.id) (CategoryTheory …
  -/
  refine EnrichedFunctor.ext _ (fun _ ↦ rfl) fun i j ↦ ?_
  /-
    J : Type u
    inst✝ : LinearOrder J
    i j : CategoryTheory.SimplicialThickening J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.SimplicialThickening …
  -/
  ext
  /-
    case w.h
    J : Type u
    inst✝ : LinearOrder J
    i j : CategoryTheory.SimplicialThickening J
    n✝ : Opposite SimplexCategory
    a✝ : (CategoryTheory.EnrichedCategory.Hom i j).obj n✝
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((CategoryTheory.SimplicialThickenin …
  -/
  exact Functor.ext (by aesop_cat)
  /-
    🎉 no goals
  -/


lemma functor_comp {J K L : Type u} [LinearOrder J] [LinearOrder K]
    [LinearOrder L] (f : J →o K) (g : K →o L) :
    functor (g.comp f) =
      (functor f).comp _ (functor g) := by
  /-
    J K L : Type u
    inst✝² : LinearOrder J
    inst✝¹ : LinearOrder K
    inst✝ : LinearOrder L
    f : OrderHom J K
    g : OrderHom K L
    ⊢ Eq (CategoryTheory.SimplicialThickening.functor (g.comp f)) (CategoryTheory. …
  -/
  refine EnrichedFunctor.ext _ (fun _ ↦ rfl) fun i j ↦ ?_
  /-
    J K L : Type u
    inst✝² : LinearOrder J
    inst✝¹ : LinearOrder K
    inst✝ : LinearOrder L
    f : OrderHom J K
    g : OrderHom K L
    i j : CategoryTheory.SimplicialThickening J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.SimplicialThickening …
  -/
  ext
  /-
    case w.h
    J K L : Type u
    inst✝² : LinearOrder J
    inst✝¹ : LinearOrder K
    inst✝ : LinearOrder L
    f : OrderHom J K
    g : OrderHom K L
    i j : CategoryTheory.SimplicialThickening J
    n✝ : Opposite SimplexCategory
    a✝ : (CategoryTheory.EnrichedCategory.Hom i j).obj n✝
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((CategoryTheory.SimplicialThickenin …
  -/
  exact Functor.ext (by aesop_cat)
  /-
    🎉 no goals
  -/


/--
The simplicial nerve of a simplicial category `C` is defined as the simplicial set whose
`n`-simplices are given by the set of simplicial functors from the simplicial thickening of
the linear order `Fin (n + 1)` to `C`
-/
noncomputable def SimplicialNerve (C : Type u) [Category.{v} C] [SimplicialCategory C] :
    SSet.{max u v} where
  obj n := EnrichedFunctor SSet (SimplicialThickening (ULift (Fin (n.unop.len + 1)))) C
  map f := (SimplicialThickening.functor f.unop.toOrderHom.uliftMap).comp (E := C) SSet
  map_id i := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.SimplicialCategory C
      i : Opposite SimplexCategory
      ⊢ Eq ({ obj := fun n => CategoryTheory.EnrichedFunctor SSet (CategoryTheory.Si …
    -/
    change EnrichedFunctor.comp SSet (SimplicialThickening.functor (OrderHom.id)) = _
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.SimplicialCategory C
      i : Opposite SimplexCategory
      ⊢ Eq (CategoryTheory.EnrichedFunctor.comp SSet (CategoryTheory.SimplicialThick …
    -/
    rw [SimplicialThickening.functor_id]
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.SimplicialCategory C
      i : Opposite SimplexCategory
      ⊢ Eq (CategoryTheory.EnrichedFunctor.comp SSet (CategoryTheory.EnrichedFunctor …
    -/
    rfl
    /-
      🎉 no goals
    -/
  map_comp f g := by
    change EnrichedFunctor.comp SSet (SimplicialThickening.functor
      (f.unop.toOrderHom.uliftMap.comp g.unop.toOrderHom.uliftMap)) = _
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.SimplicialCategory C
      X✝ Y✝ Z✝ : Opposite SimplexCategory
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Eq (CategoryTheory.EnrichedFunctor.comp SSet (CategoryTheory.SimplicialThick …
    -/
    rw [SimplicialThickening.functor_comp]
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      inst✝ : CategoryTheory.SimplicialCategory C
      X✝ Y✝ Z✝ : Opposite SimplexCategory
      f : Quiver.Hom X✝ Y✝
      g : Quiver.Hom Y✝ Z✝
      ⊢ Eq (CategoryTheory.EnrichedFunctor.comp SSet (CategoryTheory.EnrichedFunctor …
    -/
    rfl
    /-
      🎉 no goals
    -/


