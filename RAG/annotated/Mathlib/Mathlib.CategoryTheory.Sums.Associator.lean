/-- The associator functor `(C ⊕ D) ⊕ E ⥤ C ⊕ (D ⊕ E)` for sums of categories.
-/
def associator : (C ⊕ D) ⊕ E ⥤ C ⊕ (D ⊕ E) where
  obj X :=
    match X with
    | inl (inl X) => inl X
    | inl (inr X) => inr (inl X)
    | inr X => inr (inr X)
  map {X Y} f :=
    match X, Y, f with
    | inl (inl _), inl (inl _), f => f
    | inl (inr _), inl (inr _), f => f
    | inr _, inr _, f => f
               /-
                 C : Type u
                 inst✝² : CategoryTheory.Category.{v, u} C
                 D : Type u
                 inst✝¹ : CategoryTheory.Category.{v, u} D
                 E : Type u
                 inst✝ : CategoryTheory.Category.{v, u} E
                 ⊢ ∀ (X : Sum (Sum C D) E), Eq ({ obj := fun X => CategoryTheory.sum.associator …
               -/
                                    /-
                                      🎉 no goals
                                    -/
                                    /-
                                      🎉 no goals
                                    -/
  map_id := by rintro ((_|_)|_) <;> rfl
                                    /-
                                      🎉 no goals
                                    -/
  map_comp := by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} D
      E : Type u
      inst✝ : CategoryTheory.Category.{v, u} E
      ⊢ ∀ {X Y Z : Sum (Sum C D) E} (f : Quiver.Hom X Y) (g : Quiver.Hom Y Z), Eq ({ …
    -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
    rintro ((_|_)|_) ((_|_)|_) ((_|_)|_) f g <;> first | cases f | cases g | aesop_cat
                                                 /-
                                                   🎉 no goals
                                                 -/


@[simp]
theorem associator_obj_inl_inl (X) : (associator C D E).obj (inl (inl X)) = inl X :=
  rfl


@[simp]
theorem associator_obj_inl_inr (X) : (associator C D E).obj (inl (inr X)) = inr (inl X) :=
  rfl


@[simp]
theorem associator_obj_inr (X) : (associator C D E).obj (inr X) = inr (inr X) :=
  rfl


@[simp]
theorem associator_map_inl_inl {X Y : C} (f : inl (inl X) ⟶ inl (inl Y)) :
    (associator C D E).map f = f :=
  rfl


@[simp]
theorem associator_map_inl_inr {X Y : D} (f : inl (inr X) ⟶ inl (inr Y)) :
    (associator C D E).map f = f :=
  rfl


@[simp]
theorem associator_map_inr {X Y : E} (f : inr X ⟶ inr Y) : (associator C D E).map f = f :=
  rfl


/-- The inverse associator functor `C ⊕ (D ⊕ E) ⥤ (C ⊕ D) ⊕ E` for sums of categories.
-/
def inverseAssociator : C ⊕ (D ⊕ E) ⥤ (C ⊕ D) ⊕ E where
  obj X :=
    match X with
    | inl X => inl (inl X)
    | inr (inl X) => inl (inr X)
    | inr (inr X) => inr X
  map {X Y} f :=
    match X, Y, f with
    | inl _, inl _, f => f
    | inr (inl _), inr (inl _), f => f
    | inr (inr _), inr (inr _), f => f
               /-
                 C : Type u
                 inst✝² : CategoryTheory.Category.{v, u} C
                 D : Type u
                 inst✝¹ : CategoryTheory.Category.{v, u} D
                 E : Type u
                 inst✝ : CategoryTheory.Category.{v, u} E
                 ⊢ ∀ (X : Sum C (Sum D E)), Eq ({ obj := fun X => CategoryTheory.sum.inverseAss …
               -/
                                    /-
                                      🎉 no goals
                                    -/
                                    /-
                                      🎉 no goals
                                    -/
  map_id := by rintro (_|(_|_)) <;> rfl
                                    /-
                                      🎉 no goals
                                    -/
  map_comp := by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} D
      E : Type u
      inst✝ : CategoryTheory.Category.{v, u} E
      ⊢ ∀ {X Y Z : Sum C (Sum D E)} (f : Quiver.Hom X Y) (g : Quiver.Hom Y Z), Eq ({ …
    -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
    rintro (_|(_|_)) (_|(_|_)) (_|(_|_)) f g <;> first | cases f | cases g | aesop_cat
                                                 /-
                                                   🎉 no goals
                                                 -/


@[simp]
theorem inverseAssociator_obj_inl (X) : (inverseAssociator C D E).obj (inl X) = inl (inl X) :=
  rfl


@[simp]
theorem inverseAssociator_obj_inr_inl (X) :
    (inverseAssociator C D E).obj (inr (inl X)) = inl (inr X) :=
  rfl


@[simp]
theorem inverseAssociator_obj_inr_inr (X) : (inverseAssociator C D E).obj (inr (inr X)) = inr X :=
  rfl


@[simp]
theorem inverseAssociator_map_inl {X Y : C} (f : inl X ⟶ inl Y) :
    (inverseAssociator C D E).map f = f :=
  rfl


@[simp]
theorem inverseAssociator_map_inr_inl {X Y : D} (f : inr (inl X) ⟶ inr (inl Y)) :
    (inverseAssociator C D E).map f = f :=
  rfl


@[simp]
theorem inverseAssociator_map_inr_inr {X Y : E} (f : inr (inr X) ⟶ inr (inr Y)) :
    (inverseAssociator C D E).map f = f :=
  rfl


/-- The equivalence of categories expressing associativity of sums of categories.
-/
@[simps functor inverse]
def associativity : (C ⊕ D) ⊕ E ≌ C ⊕ (D ⊕ E) where
  functor := associator C D E
  inverse := inverseAssociator C D E
                                     /-
                                       C : Type u
                                       inst✝² : CategoryTheory.Category.{v, u} C
                                       D : Type u
                                       inst✝¹ : CategoryTheory.Category.{v, u} D
                                       E : Type u
                                       inst✝ : CategoryTheory.Category.{v, u} E
                                       ⊢ (X : Sum (Sum C D) E) → CategoryTheory.Iso ((CategoryTheory.Functor.id (Sum  …
                                     -/
                                                              /-
                                                                🎉 no goals
                                                              -/
                                                              /-
                                                                🎉 no goals
                                                              -/
  unitIso := NatIso.ofComponents (by rintro ((_ | _) | _) <;> exact Iso.refl _) (by
                                                              /-
                                                                🎉 no goals
                                                              -/
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} D
      E : Type u
      inst✝ : CategoryTheory.Category.{v, u} E
      ⊢ ∀ {X Y : Sum (Sum C D) E} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryS …
    -/
                                             /-
                                               🎉 no goals
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
    rintro ((_ | _) | _) ((_ | _) | _) f <;> first | cases f | aesop_cat)
                                             /-
                                               🎉 no goals
                                             -/
                                       /-
                                         C : Type u
                                         inst✝² : CategoryTheory.Category.{v, u} C
                                         D : Type u
                                         inst✝¹ : CategoryTheory.Category.{v, u} D
                                         E : Type u
                                         inst✝ : CategoryTheory.Category.{v, u} E
                                         ⊢ (X : Sum C (Sum D E)) → CategoryTheory.Iso (((CategoryTheory.sum.inverseAsso …
                                       -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
  counitIso := NatIso.ofComponents (by rintro (_ | (_ | _)) <;> exact Iso.refl _) (by
                                                                /-
                                                                  🎉 no goals
                                                                -/
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      D : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} D
      E : Type u
      inst✝ : CategoryTheory.Category.{v, u} E
      ⊢ ∀ {X Y : Sum C (Sum D E)} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryS …
    -/
                                             /-
                                               🎉 no goals
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
    rintro (_ | (_ | _)) (_ | (_ | _)) f <;> first | cases f | aesop_cat)
                                             /-
                                               🎉 no goals
                                             -/


instance associatorIsEquivalence : (associator C D E).IsEquivalence :=
      /-
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        D : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} D
        E : Type u
        inst✝ : CategoryTheory.Category.{v, u} E
        ⊢ (CategoryTheory.sum.associativity C D E).functor.IsEquivalence
      -/
  (by infer_instance : (associativity C D E).functor.IsEquivalence)
      /-
        🎉 no goals
      -/


instance inverseAssociatorIsEquivalence : (inverseAssociator C D E).IsEquivalence :=
      /-
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        D : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} D
        E : Type u
        inst✝ : CategoryTheory.Category.{v, u} E
        ⊢ (CategoryTheory.sum.associativity C D E).inverse.IsEquivalence
      -/
  (by infer_instance : (associativity C D E).inverse.IsEquivalence)
      /-
        🎉 no goals
      -/

-- TODO unitors?
-- TODO pentagon natural transformation? ...satisfying?

