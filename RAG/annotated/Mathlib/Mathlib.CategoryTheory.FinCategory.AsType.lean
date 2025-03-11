/-- A FinCategory `α` is equivalent to a category with objects in `Type`. -/
--@[nolint unused_arguments]
abbrev ObjAsType : Type :=
  InducedCategory α (Fintype.equivFin α).symm


instance {i j : ObjAsType α} : Fintype (i ⟶ j) :=
  FinCategory.fintypeHom ((Fintype.equivFin α).symm i) _


/-- The constructed category is indeed equivalent to `α`. -/
noncomputable def objAsTypeEquiv : ObjAsType α ≌ α :=
  (inducedFunctor (Fintype.equivFin α).symm).asEquivalence


/-- A FinCategory `α` is equivalent to a fin_category with in `Type`. -/
--@[nolint unused_arguments]
abbrev AsType : Type :=
  Fin (Fintype.card α)


@[simps (config := .lemmasOnly) id comp]
noncomputable instance categoryAsType : SmallCategory (AsType α) where
  Hom i j := Fin (Fintype.card (@Quiver.Hom (ObjAsType α) _ i j))
  id _ := Fintype.equivFin _ (𝟙 _)
  comp f g := Fintype.equivFin _ ((Fintype.equivFin _).symm f ≫ (Fintype.equivFin _).symm g)


/-- The "identity" functor from `AsType α` to `ObjAsType α`. -/
@[simps]
noncomputable def asTypeToObjAsType : AsType α ⥤ ObjAsType α where
  obj := id
  map {_ _} := (Fintype.equivFin _).symm


/-- The "identity" functor from `ObjAsType α` to `AsType α`. -/
@[simps]
noncomputable def objAsTypeToAsType : ObjAsType α ⥤ AsType α where
  obj := id
  map {_ _} := Fintype.equivFin _


/-- The constructed category (`AsType α`) is equivalent to `ObjAsType α`. -/
noncomputable def asTypeEquivObjAsType : AsType α ≌ ObjAsType α where
  functor := asTypeToObjAsType α
  inverse := objAsTypeToAsType α
             /-
               α : Type u_1
               inst✝² : Fintype α
               inst✝¹ : CategoryTheory.SmallCategory α
               inst✝ : CategoryTheory.FinCategory α
               ⊢ ∀ {X Y : CategoryTheory.FinCategory.AsType α} (f : Quiver.Hom X Y), Eq (Cate …
             -/
  unitIso := NatIso.ofComponents Iso.refl
             /-
               🎉 no goals
             -/
               /-
                 α : Type u_1
                 inst✝² : Fintype α
                 inst✝¹ : CategoryTheory.SmallCategory α
                 inst✝ : CategoryTheory.FinCategory α
                 ⊢ ∀ {X Y : CategoryTheory.FinCategory.ObjAsType α} (f : Quiver.Hom X Y), Eq (C …
               -/
  counitIso := NatIso.ofComponents Iso.refl
               /-
                 🎉 no goals
               -/


noncomputable instance asTypeFinCategory : FinCategory (AsType α) where
  fintypeHom := fun _ _ => show Fintype (Fin _) from inferInstance


/-- The constructed category (`ObjAsType α`) is indeed equivalent to `α`. -/
noncomputable def equivAsType : AsType α ≌ α :=
  (asTypeEquivObjAsType α).trans (objAsTypeEquiv α)


