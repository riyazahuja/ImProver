/-- An object `X` is isomorphic to an object `Y`, if `X ≅ Y` is not empty. -/
def IsIsomorphic : C → C → Prop := fun X Y => Nonempty (X ≅ Y)


/-- `IsIsomorphic` defines a setoid. -/
def isIsomorphicSetoid : Setoid C where
  r := IsIsomorphic
  iseqv := ⟨fun X => ⟨Iso.refl X⟩, fun ⟨α⟩ => ⟨α.symm⟩, fun ⟨α⟩ ⟨β⟩ => ⟨α.trans β⟩⟩


/-- The functor that sends each category to the quotient space of its objects up to an isomorphism.
-/
def isomorphismClasses : Cat.{v, u} ⥤ Type u where
  obj C := Quotient (isIsomorphicSetoid C.α)
  map {_ _} F := Quot.map F.obj fun _ _ ⟨f⟩ => ⟨F.mapIso f⟩
  map_id {C} := by  -- Porting note: this used to be `tidy`
    /-
      C : CategoryTheory.Cat
      ⊢ Eq ({ obj := fun C => Quotient (CategoryTheory.isIsomorphicSetoid ↑C), map : …
    -/
    dsimp; apply funext; intro x
    /-
      case h
      C : CategoryTheory.Cat
      x : Quot ⇑(CategoryTheory.isIsomorphicSetoid ↑C)
      ⊢ Eq (Quot.map (CategoryTheory.CategoryStruct.id C).obj ⋯ x) (CategoryTheory.C …
    -/
    apply @Quot.recOn _ _ _ x
      /-
        case h.h
        C : CategoryTheory.Cat
        x : Quot ⇑(CategoryTheory.isIsomorphicSetoid ↑C)
        ⊢ ∀ (a b : ↑C) (p : (CategoryTheory.isIsomorphicSetoid ↑C) a b), Eq ⋯ ⋯
      -/
    · intro _ _ p
      /-
        case h.h
        C : CategoryTheory.Cat
        x : Quot ⇑(CategoryTheory.isIsomorphicSetoid ↑C)
        a✝ b✝ : ↑C
        p : (CategoryTheory.isIsomorphicSetoid ↑C) a✝ b✝
        ⊢ Eq ⋯ ⋯
      -/
      simp only [types_id_apply]
      /-
        🎉 no goals
      -/
      /-
        case h.f
        C : CategoryTheory.Cat
        x : Quot ⇑(CategoryTheory.isIsomorphicSetoid ↑C)
        ⊢ ∀ (a : ↑C), Eq (Quot.map (CategoryTheory.CategoryStruct.id C).obj ⋯ (Quot.mk …
      -/
    · intro _
      /-
        case h.f
        C : CategoryTheory.Cat
        x : Quot ⇑(CategoryTheory.isIsomorphicSetoid ↑C)
        a✝ : ↑C
        ⊢ Eq (Quot.map (CategoryTheory.CategoryStruct.id C).obj ⋯ (Quot.mk (⇑(Category …
      -/
      rfl
      /-
        🎉 no goals
      -/
  map_comp {C D E} f g := by -- Porting note(s): idem
    /-
      C D E : CategoryTheory.Cat
      f : Quiver.Hom C D
      g : Quiver.Hom D E
      ⊢ Eq ({ obj := fun C => Quotient (CategoryTheory.isIsomorphicSetoid ↑C), map : …
    -/
    dsimp; apply funext; intro x
    /-
      case h
      C D E : CategoryTheory.Cat
      f : Quiver.Hom C D
      g : Quiver.Hom D E
      x : Quot ⇑(CategoryTheory.isIsomorphicSetoid ↑C)
      ⊢ Eq (Quot.map (CategoryTheory.CategoryStruct.comp f g).obj ⋯ x) (CategoryTheo …
    -/
    apply @Quot.recOn _ _ _ x
      /-
        case h.h
        C D E : CategoryTheory.Cat
        f : Quiver.Hom C D
        g : Quiver.Hom D E
        x : Quot ⇑(CategoryTheory.isIsomorphicSetoid ↑C)
        ⊢ ∀ (a b : ↑C) (p : (CategoryTheory.isIsomorphicSetoid ↑C) a b), Eq ⋯ ⋯
      -/
    · intro _ _ _
      /-
        case h.h
        C D E : CategoryTheory.Cat
        f : Quiver.Hom C D
        g : Quiver.Hom D E
        x : Quot ⇑(CategoryTheory.isIsomorphicSetoid ↑C)
        a✝ b✝ : ↑C
        p✝ : (CategoryTheory.isIsomorphicSetoid ↑C) a✝ b✝
        ⊢ Eq ⋯ ⋯
      -/
      simp only [types_id_apply]
      /-
        🎉 no goals
      -/
      /-
        case h.f
        C D E : CategoryTheory.Cat
        f : Quiver.Hom C D
        g : Quiver.Hom D E
        x : Quot ⇑(CategoryTheory.isIsomorphicSetoid ↑C)
        ⊢ ∀ (a : ↑C), Eq (Quot.map (CategoryTheory.CategoryStruct.comp f g).obj ⋯ (Quo …
      -/
    · intro _
      /-
        case h.f
        C D E : CategoryTheory.Cat
        f : Quiver.Hom C D
        g : Quiver.Hom D E
        x : Quot ⇑(CategoryTheory.isIsomorphicSetoid ↑C)
        a✝ : ↑C
        ⊢ Eq (Quot.map (CategoryTheory.CategoryStruct.comp f g).obj ⋯ (Quot.mk (⇑(Cate …
      -/
      rfl
      /-
        🎉 no goals
      -/


theorem Groupoid.isIsomorphic_iff_nonempty_hom {C : Type u} [Groupoid.{v} C] {X Y : C} :
    IsIsomorphic X Y ↔ Nonempty (X ⟶ Y) :=
  (Groupoid.isoEquivHom X Y).nonempty_congr


