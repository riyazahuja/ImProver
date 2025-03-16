/-- The associator functor `(C × D) × E ⥤ C × (D × E)`.
-/
@[simps]
def associator : (C × D) × E ⥤ C × D × E where
  obj X := (X.1.1, (X.1.2, X.2))
  map := @fun _ _ f => (f.1.1, (f.1.2, f.2))


/-- The inverse associator functor `C × (D × E) ⥤ (C × D) × E `.
-/
@[simps]
def inverseAssociator : C × D × E ⥤ (C × D) × E where
  obj X := ((X.1, X.2.1), X.2.2)
  map := @fun _ _ f => ((f.1, f.2.1), f.2.2)


/-- The equivalence of categories expressing associativity of products of categories.
-/
@[simps]
def associativity : (C × D) × E ≌ C × D × E where
  functor := associator C D E
  inverse := inverseAssociator C D E
  unitIso := Iso.refl _
  counitIso := Iso.refl _


instance associatorIsEquivalence : (associator C D E).IsEquivalence :=
      /-
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
        E : Type u₃
        inst✝ : CategoryTheory.Category.{v₃, u₃} E
        ⊢ (CategoryTheory.prod.associativity C D E).functor.IsEquivalence
      -/
  (by infer_instance : (associativity C D E).functor.IsEquivalence)
      /-
        🎉 no goals
      -/


instance inverseAssociatorIsEquivalence : (inverseAssociator C D E).IsEquivalence :=
      /-
        C : Type u₁
        inst✝² : CategoryTheory.Category.{v₁, u₁} C
        D : Type u₂
        inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
        E : Type u₃
        inst✝ : CategoryTheory.Category.{v₃, u₃} E
        ⊢ (CategoryTheory.prod.associativity C D E).inverse.IsEquivalence
      -/
  (by infer_instance : (associativity C D E).inverse.IsEquivalence)
      /-
        🎉 no goals
      -/

-- TODO pentagon natural transformation? ...satisfying?

