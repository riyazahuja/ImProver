/-- A `Groupoid` is a category such that all morphisms are isomorphisms. -/
class Groupoid (obj : Type u) extends Category.{v} obj : Type max u (v + 1) where
  /-- The inverse morphism -/
  inv : ∀ {X Y : obj}, (X ⟶ Y) → (Y ⟶ X)
  /-- `inv f` composed `f` is the identity -/
  inv_comp : ∀ {X Y : obj} (f : X ⟶ Y), comp (inv f) f = id Y := by aesop_cat
  /-- `f` composed with `inv f` is the identity -/
  comp_inv : ∀ {X Y : obj} (f : X ⟶ Y), comp f (inv f) = id X := by aesop_cat


/-- A `LargeGroupoid` is a groupoid
where the objects live in `Type (u+1)` while the morphisms live in `Type u`.
-/
abbrev LargeGroupoid (C : Type (u + 1)) : Type (u + 1) :=
  Groupoid.{u} C


/-- A `SmallGroupoid` is a groupoid
where the objects and morphisms live in the same universe.
-/
abbrev SmallGroupoid (C : Type u) : Type (u + 1) :=
  Groupoid.{u} C


instance (priority := 100) IsIso.of_groupoid (f : X ⟶ Y) : IsIso f :=
  ⟨⟨Groupoid.inv f, Groupoid.comp_inv f, Groupoid.inv_comp f⟩⟩


@[simp]
theorem Groupoid.inv_eq_inv (f : X ⟶ Y) : Groupoid.inv f = CategoryTheory.inv f :=
  IsIso.eq_inv_of_hom_inv_id <| Groupoid.comp_inv f


/-- `Groupoid.inv` is involutive. -/
@[simps]
def Groupoid.invEquiv : (X ⟶ Y) ≃ (Y ⟶ X) :=
                                           /-
                                             C : Type u
                                             inst✝ : CategoryTheory.Groupoid C
                                             X Y : C
                                             f : Quiver.Hom X Y
                                             ⊢ Eq (CategoryTheory.Groupoid.inv (CategoryTheory.Groupoid.inv f)) f
                                           -/
                                           /-
                                             🎉 no goals
                                           -/
  ⟨Groupoid.inv, Groupoid.inv, fun f => by simp, fun f => by simp⟩
                                                             /-
                                                               🎉 no goals
                                                             -/


instance (priority := 100) groupoidHasInvolutiveReverse : Quiver.HasInvolutiveReverse C where
  reverse' f := Groupoid.inv f
  inv' f := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Groupoid C
      X Y a✝ b✝ : C
      f : Quiver.Hom a✝ b✝
      ⊢ Eq (Quiver.reverse (Quiver.reverse f)) f
    -/
    dsimp [Quiver.reverse]
    /-
      C : Type u
      inst✝ : CategoryTheory.Groupoid C
      X Y a✝ b✝ : C
      f : Quiver.Hom a✝ b✝
      ⊢ Eq (CategoryTheory.Groupoid.inv (CategoryTheory.Groupoid.inv f)) f
    -/
    simp
    /-
      🎉 no goals
    -/


@[simp]
theorem Groupoid.reverse_eq_inv (f : X ⟶ Y) : Quiver.reverse f = Groupoid.inv f :=
  rfl


instance functorMapReverse {D : Type*} [Groupoid D] (F : C ⥤ D) : F.toPrefunctor.MapReverse where
  map_reverse' f := by
    simp only [Quiver.reverse, Quiver.HasReverse.reverse', Groupoid.inv_eq_inv,
      Functor.map_inv]


/-- In a groupoid, isomorphisms are equivalent to morphisms. -/
def Groupoid.isoEquivHom : (X ≅ Y) ≃ (X ⟶ Y) where
  toFun := Iso.hom
                                      /-
                                        C : Type u
                                        inst✝ : CategoryTheory.Groupoid C
                                        X Y : C
                                        f : Quiver.Hom X Y
                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Groupoid.inv f)) (C …
                                      -/
                                      /-
                                        🎉 no goals
                                      -/
  invFun f := ⟨f, Groupoid.inv f, (by aesop_cat), (by aesop_cat)⟩
                                                      /-
                                                        🎉 no goals
                                                      -/
  left_inv _ := Iso.ext rfl
  right_inv _ := rfl


/-- The functor from a groupoid `C` to its opposite sending every morphism to its inverse. -/
@[simps]
noncomputable def Groupoid.invFunctor : C ⥤ Cᵒᵖ where
  obj := Opposite.op
  map {_ _} f := (inv f).op


/-- A category where every morphism `IsIso` is a groupoid. -/
noncomputable def Groupoid.ofIsIso (all_is_iso : ∀ {X Y : C} (f : X ⟶ Y), IsIso f) :
    Groupoid.{v} C where
  inv := fun f => CategoryTheory.inv f
  inv_comp := fun f => Classical.choose_spec (all_is_iso f).out|>.right


/-- A category with a unique morphism between any two objects is a groupoid -/
def Groupoid.ofHomUnique (all_unique : ∀ {X Y : C}, Unique (X ⟶ Y)) : Groupoid.{v} C where
  inv _ := all_unique.default


instance InducedCategory.groupoid {C : Type u} (D : Type u₂) [Groupoid.{v} D] (F : C → D) :
    Groupoid.{v} (InducedCategory D F) :=
  { InducedCategory.category F with
    inv := fun f => Groupoid.inv f
    inv_comp := fun f => Groupoid.inv_comp f
    comp_inv := fun f => Groupoid.comp_inv f }


instance groupoidPi {I : Type u} {J : I → Type u₂} [∀ i, Groupoid.{v} (J i)] :
    Groupoid.{max u v} (∀ i : I, J i) where
  inv f := fun i : I => Groupoid.inv (f i)
                          /-
                            I : Type u
                            J : I → Type u₂
                            inst✝ : (i : I) → CategoryTheory.Groupoid (J i)
                            X✝ Y✝ : (i : I) → J i
                            f : Quiver.Hom X✝ Y✝
                            ⊢ Eq (CategoryTheory.CategoryStruct.comp f ((fun {X Y} f i => CategoryTheory.G …
                          -/
                          /-
                            I : Type u
                            J : I → Type u₂
                            inst✝ : (i : I) → CategoryTheory.Groupoid (J i)
                            X✝ Y✝ : (i : I) → J i
                            f : Quiver.Hom X✝ Y✝
                            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun {X Y} f i => CategoryTheory.Gro …
                          -/
  comp_inv := fun f => by funext i; apply Groupoid.comp_inv
                                    /-
                                      🎉 no goals
                                    -/
                                    /-
                                      🎉 no goals
                                    -/
  inv_comp := fun f => by funext i; apply Groupoid.inv_comp


instance groupoidProd {α : Type u} {β : Type v} [Groupoid.{u₂} α] [Groupoid.{v₂} β] :
    Groupoid.{max u₂ v₂} (α × β) where
  inv f := (Groupoid.inv f.1, Groupoid.inv f.2)


