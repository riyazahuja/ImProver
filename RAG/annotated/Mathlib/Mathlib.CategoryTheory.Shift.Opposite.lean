/-- Construction of the naive shift on the opposite category of a category `C`:
the shiftfunctor by `n` is `(shiftFunctor C n).op`. -/
noncomputable def mkShiftCoreOp : ShiftMkCore Cᵒᵖ A where
  F n := (shiftFunctor C n).op
  zero := (NatIso.op (shiftFunctorZero C A)).symm
  add a b := (NatIso.op (shiftFunctorAdd C a b)).symm
  assoc_hom_app m₁ m₂ m₃ X :=
    Quiver.Hom.unop_inj ((shiftFunctorAdd_assoc_inv_app m₁ m₂ m₃ X.unop).trans
          /-
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.47, u_1} C
            A : Type u_2
            inst✝¹ : AddMonoid A
            inst✝ : CategoryTheory.HasShift C A
            m₁ m₂ m₃ : A
            X : Opposite C
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctorAdd C m₂ …
          -/
      (by simp [shiftFunctorAdd']))
          /-
            🎉 no goals
          -/
  zero_add_hom_app n X :=
                                                                               /-
                                                                                 C : Type u_1
                                                                                 inst✝² : CategoryTheory.Category.{?u.47, u_1} C
                                                                                 A : Type u_2
                                                                                 inst✝¹ : AddMonoid A
                                                                                 inst✝ : CategoryTheory.HasShift C A
                                                                                 n : A
                                                                                 X : Opposite C
                                                                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctor C n).ma …
                                                                               -/
    Quiver.Hom.unop_inj ((shiftFunctorAdd_zero_add_inv_app n X.unop).trans (by simp))
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
  add_zero_hom_app n X :=
                                                                               /-
                                                                                 C : Type u_1
                                                                                 inst✝² : CategoryTheory.Category.{?u.47, u_1} C
                                                                                 A : Type u_2
                                                                                 inst✝¹ : AddMonoid A
                                                                                 inst✝ : CategoryTheory.HasShift C A
                                                                                 n : A
                                                                                 X : Opposite C
                                                                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctorZero C A …
                                                                               -/
    Quiver.Hom.unop_inj ((shiftFunctorAdd_add_zero_inv_app n X.unop).trans (by simp))
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


/-- The category `OppositeShift C A` is the opposite category `Cᵒᵖ` equipped
with the naive shift: `shiftFunctor (OppositeShift C A) n` is `(shiftFunctor C n).op`. -/
@[nolint unusedArguments]
def OppositeShift (A : Type*) [AddMonoid A] [HasShift C A] := Cᵒᵖ


instance : Category (OppositeShift C A) := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{?u.9186, u_1} C
    A : Type u_2
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift C A
    ⊢ CategoryTheory.Category.{?u.9224, u_1} (CategoryTheory.OppositeShift C A)
  -/
  dsimp only [OppositeShift]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{?u.9186, u_1} C
    A : Type u_2
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift C A
    ⊢ CategoryTheory.Category.{?u.9224, u_1} (Opposite C)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


noncomputable instance : HasShift (OppositeShift C A) A :=
  hasShiftMk Cᵒᵖ A (HasShift.mkShiftCoreOp C A)


instance [HasZeroObject C] : HasZeroObject (OppositeShift C A) := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    A : Type u_2
    inst✝² : AddMonoid A
    inst✝¹ : CategoryTheory.HasShift C A
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    ⊢ CategoryTheory.Limits.HasZeroObject (CategoryTheory.OppositeShift C A)
  -/
  dsimp only [OppositeShift]
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    A : Type u_2
    inst✝² : AddMonoid A
    inst✝¹ : CategoryTheory.HasShift C A
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    ⊢ CategoryTheory.Limits.HasZeroObject (Opposite C)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance [Preadditive C] : Preadditive (OppositeShift C A) := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{?u.9834, u_1} C
    A : Type u_2
    inst✝² : AddMonoid A
    inst✝¹ : CategoryTheory.HasShift C A
    inst✝ : CategoryTheory.Preadditive C
    ⊢ CategoryTheory.Preadditive (CategoryTheory.OppositeShift C A)
  -/
  dsimp only [OppositeShift]
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{?u.9834, u_1} C
    A : Type u_2
    inst✝² : AddMonoid A
    inst✝¹ : CategoryTheory.HasShift C A
    inst✝ : CategoryTheory.Preadditive C
    ⊢ CategoryTheory.Preadditive (Opposite C)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance [Preadditive C] (n : A) [(shiftFunctor C n).Additive] :
    (shiftFunctor (OppositeShift C A) n).Additive := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    A : Type u_2
    inst✝³ : AddMonoid A
    inst✝² : CategoryTheory.HasShift C A
    inst✝¹ : CategoryTheory.Preadditive C
    n : A
    inst✝ : (CategoryTheory.shiftFunctor C n).Additive
    ⊢ (CategoryTheory.shiftFunctor (CategoryTheory.OppositeShift C A) n).Additive
  -/
  change (shiftFunctor C n).op.Additive
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    A : Type u_2
    inst✝³ : AddMonoid A
    inst✝² : CategoryTheory.HasShift C A
    inst✝¹ : CategoryTheory.Preadditive C
    n : A
    inst✝ : (CategoryTheory.shiftFunctor C n).Additive
    ⊢ (CategoryTheory.shiftFunctor C n).op.Additive
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma oppositeShiftFunctorZero_inv_app (X : OppositeShift C A) :
    (shiftFunctorZero (OppositeShift C A) A).inv.app X =
      ((shiftFunctorZero C A).hom.app X.unop).op := rfl


lemma oppositeShiftFunctorZero_hom_app (X : OppositeShift C A) :
    (shiftFunctorZero (OppositeShift C A) A).hom.app X =
      ((shiftFunctorZero C A).inv.app X.unop).op := by
  rw [← cancel_mono ((shiftFunctorZero (OppositeShift C A) A).inv.app X),
    Iso.hom_inv_id_app, oppositeShiftFunctorZero_inv_app, ← op_comp,
    Iso.hom_inv_id_app, op_id]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    A : Type u_2
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift C A
    X : CategoryTheory.OppositeShift C A
    ⊢ Eq (CategoryTheory.CategoryStruct.id ((CategoryTheory.shiftFunctor (Category …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma oppositeShiftFunctorAdd_inv_app :
    (shiftFunctorAdd (OppositeShift C A) a b).inv.app X =
      ((shiftFunctorAdd C a b).hom.app X.unop).op := rfl


lemma oppositeShiftFunctorAdd_hom_app :
    (shiftFunctorAdd (OppositeShift C A) a b).hom.app X =
      ((shiftFunctorAdd C a b).inv.app X.unop).op := by
  rw [← cancel_mono ((shiftFunctorAdd (OppositeShift C A) a b).inv.app X),
    Iso.hom_inv_id_app, oppositeShiftFunctorAdd_inv_app, ← op_comp,
    Iso.hom_inv_id_app, op_id]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    A : Type u_2
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift C A
    X : CategoryTheory.OppositeShift C A
    a b : A
    ⊢ Eq (CategoryTheory.CategoryStruct.id ((CategoryTheory.shiftFunctor (Category …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma oppositeShiftFunctorAdd'_inv_app :
    (shiftFunctorAdd' (OppositeShift C A) a b c h).inv.app X =
      ((shiftFunctorAdd' C a b c h).hom.app X.unop).op := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    A : Type u_2
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift C A
    X : CategoryTheory.OppositeShift C A
    a b c : A
    h : Eq (HAdd.hAdd a b) c
    ⊢ Eq ((CategoryTheory.shiftFunctorAdd' (CategoryTheory.OppositeShift C A) a b  …
  -/
  subst h
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    A : Type u_2
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift C A
    X : CategoryTheory.OppositeShift C A
    a b : A
    ⊢ Eq ((CategoryTheory.shiftFunctorAdd' (CategoryTheory.OppositeShift C A) a b  …
  -/
  simp only [shiftFunctorAdd'_eq_shiftFunctorAdd, oppositeShiftFunctorAdd_inv_app]
  /-
    🎉 no goals
  -/


lemma oppositeShiftFunctorAdd'_hom_app :
    (shiftFunctorAdd' (OppositeShift C A) a b c h).hom.app X =
      ((shiftFunctorAdd' C a b c h).inv.app X.unop).op := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    A : Type u_2
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift C A
    X : CategoryTheory.OppositeShift C A
    a b c : A
    h : Eq (HAdd.hAdd a b) c
    ⊢ Eq ((CategoryTheory.shiftFunctorAdd' (CategoryTheory.OppositeShift C A) a b  …
  -/
  subst h
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    A : Type u_2
    inst✝¹ : AddMonoid A
    inst✝ : CategoryTheory.HasShift C A
    X : CategoryTheory.OppositeShift C A
    a b : A
    ⊢ Eq ((CategoryTheory.shiftFunctorAdd' (CategoryTheory.OppositeShift C A) a b  …
  -/
  simp only [shiftFunctorAdd'_eq_shiftFunctorAdd, oppositeShiftFunctorAdd_hom_app]
  /-
    🎉 no goals
  -/


/--
Given a `CommShift` structure on `F`, this is the corresponding `CommShift` structure on
`F.op` (for the naive shifts on the opposite categories).
-/
@[simps]
noncomputable def commShiftOp [CommShift F A] :
    CommShift (C := OppositeShift C A) (D := OppositeShift D A) F.op A where
  iso a := (NatIso.op (F.commShiftIso a)).symm
  zero := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁵ : CategoryTheory.Category.{?u.21295, u_1} C
      inst✝⁴ : CategoryTheory.Category.{?u.21299, u_2} D
      F : CategoryTheory.Functor C D
      A : Type u_3
      inst✝³ : AddMonoid A
      inst✝² : CategoryTheory.HasShift C A
      inst✝¹ : CategoryTheory.HasShift D A
      inst✝ : F.CommShift A
      ⊢ Eq ((fun a => (CategoryTheory.NatIso.op (F.commShiftIso a)).symm) 0) (Catego …
    -/
    simp only
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁵ : CategoryTheory.Category.{?u.21295, u_1} C
      inst✝⁴ : CategoryTheory.Category.{?u.21299, u_2} D
      F : CategoryTheory.Functor C D
      A : Type u_3
      inst✝³ : AddMonoid A
      inst✝² : CategoryTheory.HasShift C A
      inst✝¹ : CategoryTheory.HasShift D A
      inst✝ : F.CommShift A
      ⊢ Eq (CategoryTheory.NatIso.op (F.commShiftIso 0)).symm (CategoryTheory.Functo …
    -/
    rw [commShiftIso_zero]
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁵ : CategoryTheory.Category.{?u.21295, u_1} C
      inst✝⁴ : CategoryTheory.Category.{?u.21299, u_2} D
      F : CategoryTheory.Functor C D
      A : Type u_3
      inst✝³ : AddMonoid A
      inst✝² : CategoryTheory.HasShift C A
      inst✝¹ : CategoryTheory.HasShift D A
      inst✝ : F.CommShift A
      ⊢ Eq (CategoryTheory.NatIso.op (CategoryTheory.Functor.CommShift.isoZero F A)) …
    -/
    ext
    simp only [op_obj, comp_obj, Iso.symm_hom, NatIso.op_inv, NatTrans.op_app,
      CommShift.isoZero_inv_app, op_comp, CommShift.isoZero_hom_app, op_map]
    /-
      case w.w.h
      C : Type u_1
      D : Type u_2
      inst✝⁵ : CategoryTheory.Category.{?u.21295, u_1} C
      inst✝⁴ : CategoryTheory.Category.{?u.21299, u_2} D
      F : CategoryTheory.Functor C D
      A : Type u_3
      inst✝³ : AddMonoid A
      inst✝² : CategoryTheory.HasShift C A
      inst✝¹ : CategoryTheory.HasShift D A
      inst✝ : F.CommShift A
      x✝ : Opposite C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map ((CategoryTheory.shiftFunctorZ …
    -/
    erw [oppositeShiftFunctorZero_inv_app, oppositeShiftFunctorZero_hom_app]
    /-
      case w.w.h
      C : Type u_1
      D : Type u_2
      inst✝⁵ : CategoryTheory.Category.{?u.21295, u_1} C
      inst✝⁴ : CategoryTheory.Category.{?u.21299, u_2} D
      F : CategoryTheory.Functor C D
      A : Type u_3
      inst✝³ : AddMonoid A
      inst✝² : CategoryTheory.HasShift C A
      inst✝¹ : CategoryTheory.HasShift D A
      inst✝ : F.CommShift A
      x✝ : Opposite C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map ((CategoryTheory.shiftFunctorZ …
    -/
    rfl
    /-
      🎉 no goals
    -/
  add a b := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁵ : CategoryTheory.Category.{?u.21295, u_1} C
      inst✝⁴ : CategoryTheory.Category.{?u.21299, u_2} D
      F : CategoryTheory.Functor C D
      A : Type u_3
      inst✝³ : AddMonoid A
      inst✝² : CategoryTheory.HasShift C A
      inst✝¹ : CategoryTheory.HasShift D A
      inst✝ : F.CommShift A
      a b : A
      ⊢ Eq ((fun a => (CategoryTheory.NatIso.op (F.commShiftIso a)).symm) (HAdd.hAdd …
    -/
    simp only
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁵ : CategoryTheory.Category.{?u.21295, u_1} C
      inst✝⁴ : CategoryTheory.Category.{?u.21299, u_2} D
      F : CategoryTheory.Functor C D
      A : Type u_3
      inst✝³ : AddMonoid A
      inst✝² : CategoryTheory.HasShift C A
      inst✝¹ : CategoryTheory.HasShift D A
      inst✝ : F.CommShift A
      a b : A
      ⊢ Eq (CategoryTheory.NatIso.op (F.commShiftIso (HAdd.hAdd a b))).symm (Categor …
    -/
    rw [commShiftIso_add]
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁵ : CategoryTheory.Category.{?u.21295, u_1} C
      inst✝⁴ : CategoryTheory.Category.{?u.21299, u_2} D
      F : CategoryTheory.Functor C D
      A : Type u_3
      inst✝³ : AddMonoid A
      inst✝² : CategoryTheory.HasShift C A
      inst✝¹ : CategoryTheory.HasShift D A
      inst✝ : F.CommShift A
      a b : A
      ⊢ Eq (CategoryTheory.NatIso.op (CategoryTheory.Functor.CommShift.isoAdd (F.com …
    -/
    ext
    simp only [op_obj, comp_obj, Iso.symm_hom, NatIso.op_inv, NatTrans.op_app,
      CommShift.isoAdd_inv_app, op_comp, Category.assoc, CommShift.isoAdd_hom_app, op_map]
    /-
      case w.w.h
      C : Type u_1
      D : Type u_2
      inst✝⁵ : CategoryTheory.Category.{?u.21295, u_1} C
      inst✝⁴ : CategoryTheory.Category.{?u.21299, u_2} D
      F : CategoryTheory.Functor C D
      A : Type u_3
      inst✝³ : AddMonoid A
      inst✝² : CategoryTheory.HasShift C A
      inst✝¹ : CategoryTheory.HasShift D A
      inst✝ : F.CommShift A
      a b : A
      x✝ : Opposite C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map ((CategoryTheory.shiftFunctorA …
    -/
    erw [oppositeShiftFunctorAdd_inv_app, oppositeShiftFunctorAdd_hom_app]
    /-
      case w.w.h
      C : Type u_1
      D : Type u_2
      inst✝⁵ : CategoryTheory.Category.{?u.21295, u_1} C
      inst✝⁴ : CategoryTheory.Category.{?u.21299, u_2} D
      F : CategoryTheory.Functor C D
      A : Type u_3
      inst✝³ : AddMonoid A
      inst✝² : CategoryTheory.HasShift C A
      inst✝¹ : CategoryTheory.HasShift D A
      inst✝ : F.CommShift A
      a b : A
      x✝ : Opposite C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map ((CategoryTheory.shiftFunctorA …
    -/
    rfl
    /-
      🎉 no goals
    -/


/--
Given a `CommShift` structure on `F.op` (for the naive shifts on the opposite categories),
this is the corresponding `CommShift` structure on `F`.
-/
@[simps]
noncomputable def commShiftUnop
    [CommShift (C := OppositeShift C A) (D := OppositeShift D A) F.op A] : CommShift F A where
  iso a := NatIso.removeOp (F.op.commShiftIso (C := OppositeShift C A)
    (D := OppositeShift D A) a).symm
  zero := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁵ : CategoryTheory.Category.{?u.31232, u_1} C
      inst✝⁴ : CategoryTheory.Category.{?u.31236, u_2} D
      F : CategoryTheory.Functor C D
      A : Type u_3
      inst✝³ : AddMonoid A
      inst✝² : CategoryTheory.HasShift C A
      inst✝¹ : CategoryTheory.HasShift D A
      inst✝ : F.op.CommShift A
      ⊢ Eq ((fun a => CategoryTheory.NatIso.removeOp (F.op.commShiftIso a).symm) 0)  …
    -/
    simp only
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁵ : CategoryTheory.Category.{?u.31232, u_1} C
      inst✝⁴ : CategoryTheory.Category.{?u.31236, u_2} D
      F : CategoryTheory.Functor C D
      A : Type u_3
      inst✝³ : AddMonoid A
      inst✝² : CategoryTheory.HasShift C A
      inst✝¹ : CategoryTheory.HasShift D A
      inst✝ : F.op.CommShift A
      ⊢ Eq (CategoryTheory.NatIso.removeOp (F.op.commShiftIso 0).symm) (CategoryTheo …
    -/
    rw [commShiftIso_zero]
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁵ : CategoryTheory.Category.{?u.31232, u_1} C
      inst✝⁴ : CategoryTheory.Category.{?u.31236, u_2} D
      F : CategoryTheory.Functor C D
      A : Type u_3
      inst✝³ : AddMonoid A
      inst✝² : CategoryTheory.HasShift C A
      inst✝¹ : CategoryTheory.HasShift D A
      inst✝ : F.op.CommShift A
      ⊢ Eq (CategoryTheory.NatIso.removeOp (CategoryTheory.Functor.CommShift.isoZero …
    -/
    ext
    simp only [comp_obj, NatIso.removeOp_hom, Iso.symm_hom, NatTrans.removeOp_app, op_obj,
      CommShift.isoZero_inv_app, op_map, unop_comp, Quiver.Hom.unop_op, CommShift.isoZero_hom_app]
    /-
      case w.w.h
      C : Type u_1
      D : Type u_2
      inst✝⁵ : CategoryTheory.Category.{?u.31232, u_1} C
      inst✝⁴ : CategoryTheory.Category.{?u.31236, u_2} D
      F : CategoryTheory.Functor C D
      A : Type u_3
      inst✝³ : AddMonoid A
      inst✝² : CategoryTheory.HasShift C A
      inst✝¹ : CategoryTheory.HasShift D A
      inst✝ : F.op.CommShift A
      x✝ : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map ((CategoryTheory.shiftFunctorZ …
    -/
    erw [oppositeShiftFunctorZero_hom_app, oppositeShiftFunctorZero_inv_app]
    /-
      case w.w.h
      C : Type u_1
      D : Type u_2
      inst✝⁵ : CategoryTheory.Category.{?u.31232, u_1} C
      inst✝⁴ : CategoryTheory.Category.{?u.31236, u_2} D
      F : CategoryTheory.Functor C D
      A : Type u_3
      inst✝³ : AddMonoid A
      inst✝² : CategoryTheory.HasShift C A
      inst✝¹ : CategoryTheory.HasShift D A
      inst✝ : F.op.CommShift A
      x✝ : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map ((CategoryTheory.shiftFunctorZ …
    -/
    rfl
    /-
      🎉 no goals
    -/
  add a b := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁵ : CategoryTheory.Category.{?u.31232, u_1} C
      inst✝⁴ : CategoryTheory.Category.{?u.31236, u_2} D
      F : CategoryTheory.Functor C D
      A : Type u_3
      inst✝³ : AddMonoid A
      inst✝² : CategoryTheory.HasShift C A
      inst✝¹ : CategoryTheory.HasShift D A
      inst✝ : F.op.CommShift A
      a b : A
      ⊢ Eq ((fun a => CategoryTheory.NatIso.removeOp (F.op.commShiftIso a).symm) (HA …
    -/
    simp only
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁵ : CategoryTheory.Category.{?u.31232, u_1} C
      inst✝⁴ : CategoryTheory.Category.{?u.31236, u_2} D
      F : CategoryTheory.Functor C D
      A : Type u_3
      inst✝³ : AddMonoid A
      inst✝² : CategoryTheory.HasShift C A
      inst✝¹ : CategoryTheory.HasShift D A
      inst✝ : F.op.CommShift A
      a b : A
      ⊢ Eq (CategoryTheory.NatIso.removeOp (F.op.commShiftIso (HAdd.hAdd a b)).symm) …
    -/
    rw [commShiftIso_add]
    /-
      C : Type u_1
      D : Type u_2
      inst✝⁵ : CategoryTheory.Category.{?u.31232, u_1} C
      inst✝⁴ : CategoryTheory.Category.{?u.31236, u_2} D
      F : CategoryTheory.Functor C D
      A : Type u_3
      inst✝³ : AddMonoid A
      inst✝² : CategoryTheory.HasShift C A
      inst✝¹ : CategoryTheory.HasShift D A
      inst✝ : F.op.CommShift A
      a b : A
      ⊢ Eq (CategoryTheory.NatIso.removeOp (CategoryTheory.Functor.CommShift.isoAdd  …
    -/
    ext
    simp only [comp_obj, NatIso.removeOp_hom, Iso.symm_hom, NatTrans.removeOp_app, op_obj,
      CommShift.isoAdd_inv_app, op_map, unop_comp, Quiver.Hom.unop_op, Category.assoc,
      CommShift.isoAdd_hom_app]
    /-
      case w.w.h
      C : Type u_1
      D : Type u_2
      inst✝⁵ : CategoryTheory.Category.{?u.31232, u_1} C
      inst✝⁴ : CategoryTheory.Category.{?u.31236, u_2} D
      F : CategoryTheory.Functor C D
      A : Type u_3
      inst✝³ : AddMonoid A
      inst✝² : CategoryTheory.HasShift C A
      inst✝¹ : CategoryTheory.HasShift D A
      inst✝ : F.op.CommShift A
      a b : A
      x✝ : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map ((CategoryTheory.shiftFunctorA …
    -/
    erw [oppositeShiftFunctorAdd_hom_app, oppositeShiftFunctorAdd_inv_app]
    /-
      case w.w.h
      C : Type u_1
      D : Type u_2
      inst✝⁵ : CategoryTheory.Category.{?u.31232, u_1} C
      inst✝⁴ : CategoryTheory.Category.{?u.31236, u_2} D
      F : CategoryTheory.Functor C D
      A : Type u_3
      inst✝³ : AddMonoid A
      inst✝² : CategoryTheory.HasShift C A
      inst✝¹ : CategoryTheory.HasShift D A
      inst✝ : F.op.CommShift A
      a b : A
      x✝ : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map ((CategoryTheory.shiftFunctorA …
    -/
    rfl
    /-
      🎉 no goals
    -/


