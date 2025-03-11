/-- A relation on morphisms is compatible with the shift by a monoid `A` when the
relation if preserved by the shift. -/
class IsCompatibleWithShift : Prop where
  /-- the condition that the relation is preserved by the shift -/
  condition : ∀ (a : A) ⦃X Y : C⦄ (f g : X ⟶ Y), r f g → r (f⟦a⟧') (g⟦a⟧')


/-- The shift by a monoid `A` induced on a quotient category `Quotient r` when the
relation `r` is compatible with the shift. -/
noncomputable instance HasShift.quotient [r.IsCompatibleWithShift A] :
    HasShift (Quotient r) A :=
  HasShift.induced (Quotient.functor r) A
    (fun a => Quotient.lift r (shiftFunctor C a ⋙ Quotient.functor r)
      (fun _ _ _ _ hfg => Quotient.sound r (HomRel.IsCompatibleWithShift.condition _ _ _ hfg)))
    (fun _ => Quotient.lift.isLift _ _ _)


/-- The functor `Quotient.functor r : C ⥤ Quotient r` commutes with the shift. -/
noncomputable instance Quotient.functor_commShift [r.IsCompatibleWithShift A] :
    (Quotient.functor r).CommShift A :=
  Functor.CommShift.ofInduced _ _ _ _

-- the construction is made irreducible in order to prevent timeouts and abuse of defeq

/-- Auxiliary definition for `Quotient.liftCommShift`. -/
noncomputable def iso (a : A) :
    shiftFunctor (Quotient r) a ⋙ lift r F hF ≅ lift r F hF ⋙ shiftFunctor D a :=
  natIsoLift r ((Functor.associator _ _ _).symm ≪≫
    isoWhiskerRight ((functor r).commShiftIso a).symm _ ≪≫
    Functor.associator _ _ _ ≪≫ isoWhiskerLeft _ (lift.isLift r F hF) ≪≫ F.commShiftIso a ≪≫
    isoWhiskerRight (lift.isLift r F hF).symm _ ≪≫ Functor.associator _ _ _)


@[simp]
lemma iso_hom_app (a : A) (X : C) :
    (iso F r hF a).hom.app ((functor r).obj X) =
      (lift r F hF).map (((functor r).commShiftIso a).inv.app X) ≫
      (F.commShiftIso a).hom.app X := by
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v', u'} D
    F : CategoryTheory.Functor C D
    r : HomRel C
    A : Type w
    inst✝⁴ : AddMonoid A
    inst✝³ : CategoryTheory.HasShift C A
    inst✝² : CategoryTheory.HasShift D A
    inst✝¹ : r.IsCompatibleWithShift A
    inst✝ : F.CommShift A
    hF : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (F.map f₁) (F.map f₂)
    a : A
    X : C
    ⊢ Eq ((CategoryTheory.Quotient.LiftCommShift.iso F r hF a).hom.app ((CategoryT …
  -/
  dsimp only [iso, natIsoLift]
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v', u'} D
    F : CategoryTheory.Functor C D
    r : HomRel C
    A : Type w
    inst✝⁴ : AddMonoid A
    inst✝³ : CategoryTheory.HasShift C A
    inst✝² : CategoryTheory.HasShift D A
    inst✝¹ : r.IsCompatibleWithShift A
    inst✝ : F.CommShift A
    hF : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (F.map f₁) (F.map f₂)
    a : A
    X : C
    ⊢ Eq ((CategoryTheory.Quotient.natTransLift r (((CategoryTheory.Quotient.funct …
  -/
  rw [natTransLift_app]
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v', u'} D
    F : CategoryTheory.Functor C D
    r : HomRel C
    A : Type w
    inst✝⁴ : AddMonoid A
    inst✝³ : CategoryTheory.HasShift C A
    inst✝² : CategoryTheory.HasShift D A
    inst✝¹ : r.IsCompatibleWithShift A
    inst✝ : F.CommShift A
    hF : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (F.map f₁) (F.map f₂)
    a : A
    X : C
    ⊢ Eq ((((CategoryTheory.Quotient.functor r).associator (CategoryTheory.shiftFu …
  -/
  dsimp
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v', u'} D
    F : CategoryTheory.Functor C D
    r : HomRel C
    A : Type w
    inst✝⁴ : AddMonoid A
    inst✝³ : CategoryTheory.HasShift C A
    inst✝² : CategoryTheory.HasShift D A
    inst✝¹ : r.IsCompatibleWithShift A
    inst✝ : F.CommShift A
    hF : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (F.map f₁) (F.map f₂)
    a : A
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id ((C …
  -/
  erw [comp_id, id_comp, id_comp, id_comp, Functor.map_id, comp_id]
  /-
    🎉 no goals
  -/


@[simp]
lemma iso_inv_app (a : A) (X : C) :
    (iso F r hF a).inv.app ((functor r).obj X) =
      (F.commShiftIso a).inv.app X ≫
      (lift r F hF).map (((functor r).commShiftIso a).hom.app X) := by
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v', u'} D
    F : CategoryTheory.Functor C D
    r : HomRel C
    A : Type w
    inst✝⁴ : AddMonoid A
    inst✝³ : CategoryTheory.HasShift C A
    inst✝² : CategoryTheory.HasShift D A
    inst✝¹ : r.IsCompatibleWithShift A
    inst✝ : F.CommShift A
    hF : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (F.map f₁) (F.map f₂)
    a : A
    X : C
    ⊢ Eq ((CategoryTheory.Quotient.LiftCommShift.iso F r hF a).inv.app ((CategoryT …
  -/
  dsimp only [iso, natIsoLift]
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v', u'} D
    F : CategoryTheory.Functor C D
    r : HomRel C
    A : Type w
    inst✝⁴ : AddMonoid A
    inst✝³ : CategoryTheory.HasShift C A
    inst✝² : CategoryTheory.HasShift D A
    inst✝¹ : r.IsCompatibleWithShift A
    inst✝ : F.CommShift A
    hF : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (F.map f₁) (F.map f₂)
    a : A
    X : C
    ⊢ Eq ((CategoryTheory.Quotient.natTransLift r (((CategoryTheory.Quotient.funct …
  -/
  rw [natTransLift_app]
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v', u'} D
    F : CategoryTheory.Functor C D
    r : HomRel C
    A : Type w
    inst✝⁴ : AddMonoid A
    inst✝³ : CategoryTheory.HasShift C A
    inst✝² : CategoryTheory.HasShift D A
    inst✝¹ : r.IsCompatibleWithShift A
    inst✝ : F.CommShift A
    hF : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (F.map f₁) (F.map f₂)
    a : A
    X : C
    ⊢ Eq ((((CategoryTheory.Quotient.functor r).associator (CategoryTheory.shiftFu …
  -/
  dsimp
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    D : Type u'
    inst✝⁵ : CategoryTheory.Category.{v', u'} D
    F : CategoryTheory.Functor C D
    r : HomRel C
    A : Type w
    inst✝⁴ : AddMonoid A
    inst✝³ : CategoryTheory.HasShift C A
    inst✝² : CategoryTheory.HasShift D A
    inst✝¹ : r.IsCompatibleWithShift A
    inst✝ : F.CommShift A
    hF : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (F.map f₁) (F.map f₂)
    a : A
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  erw [id_comp, comp_id, comp_id, comp_id, Functor.map_id, id_comp]
  /-
    🎉 no goals
  -/


/-- When `r : HomRel C` is compatible with the shift by an additive monoid, and
`F : C ⥤ D` is a functor which commutes with the shift and is compatible with `r`, then
the induced functor `Quotient.lift r F _ : Quotient r ⥤ D` also commutes with the shift. -/
noncomputable instance liftCommShift :
    (Quotient.lift r F hF).CommShift A where
  iso := LiftCommShift.iso F r hF
  zero := by
    /-
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝⁵ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      r : HomRel C
      A : Type w
      inst✝⁴ : AddMonoid A
      inst✝³ : CategoryTheory.HasShift C A
      inst✝² : CategoryTheory.HasShift D A
      inst✝¹ : r.IsCompatibleWithShift A
      inst✝ : F.CommShift A
      hF : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (F.map f₁) (F.map f₂)
      ⊢ Eq (CategoryTheory.Quotient.LiftCommShift.iso F r hF 0) (CategoryTheory.Func …
    -/
    ext1
    /-
      case w
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝⁵ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      r : HomRel C
      A : Type w
      inst✝⁴ : AddMonoid A
      inst✝³ : CategoryTheory.HasShift C A
      inst✝² : CategoryTheory.HasShift D A
      inst✝¹ : r.IsCompatibleWithShift A
      inst✝ : F.CommShift A
      hF : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (F.map f₁) (F.map f₂)
      ⊢ Eq (CategoryTheory.Quotient.LiftCommShift.iso F r hF 0).hom (CategoryTheory. …
    -/
    apply natTrans_ext
    /-
      case w.h
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝⁵ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      r : HomRel C
      A : Type w
      inst✝⁴ : AddMonoid A
      inst✝³ : CategoryTheory.HasShift C A
      inst✝² : CategoryTheory.HasShift D A
      inst✝¹ : r.IsCompatibleWithShift A
      inst✝ : F.CommShift A
      hF : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (F.map f₁) (F.map f₂)
      ⊢ Eq (CategoryTheory.whiskerLeft (CategoryTheory.Quotient.functor r) (Category …
    -/
    ext X
    /-
      case w.h.w.h
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝⁵ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      r : HomRel C
      A : Type w
      inst✝⁴ : AddMonoid A
      inst✝³ : CategoryTheory.HasShift C A
      inst✝² : CategoryTheory.HasShift D A
      inst✝¹ : r.IsCompatibleWithShift A
      inst✝ : F.CommShift A
      hF : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (F.map f₁) (F.map f₂)
      X : C
      ⊢ Eq ((CategoryTheory.whiskerLeft (CategoryTheory.Quotient.functor r) (Categor …
    -/
    dsimp
    rw [LiftCommShift.iso_hom_app, (functor r).commShiftIso_zero,
      Functor.CommShift.isoZero_hom_app, Functor.CommShift.isoZero_inv_app,
      Functor.map_comp, assoc, F.commShiftIso_zero, Functor.CommShift.isoZero_hom_app,
      lift_map_functor_map, ← F.map_comp_assoc, Iso.inv_hom_id_app]
    /-
      case w.h.w.h
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝⁵ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      r : HomRel C
      A : Type w
      inst✝⁴ : AddMonoid A
      inst✝³ : CategoryTheory.HasShift C A
      inst✝² : CategoryTheory.HasShift D A
      inst✝¹ : r.IsCompatibleWithShift A
      inst✝ : F.CommShift A
      hF : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (F.map f₁) (F.map f₂)
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Quotient.lift r F hF …
    -/
    dsimp [lift_obj_functor_obj]
    /-
      case w.h.w.h
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝⁵ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      r : HomRel C
      A : Type w
      inst✝⁴ : AddMonoid A
      inst✝³ : CategoryTheory.HasShift C A
      inst✝² : CategoryTheory.HasShift D A
      inst✝¹ : r.IsCompatibleWithShift A
      inst✝ : F.CommShift A
      hF : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (F.map f₁) (F.map f₂)
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Quotient.lift r F hF …
    -/
    rw [F.map_id, id_comp]
    /-
      🎉 no goals
    -/
  add a b := by
    /-
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝⁵ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      r : HomRel C
      A : Type w
      inst✝⁴ : AddMonoid A
      inst✝³ : CategoryTheory.HasShift C A
      inst✝² : CategoryTheory.HasShift D A
      inst✝¹ : r.IsCompatibleWithShift A
      inst✝ : F.CommShift A
      hF : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (F.map f₁) (F.map f₂)
      a b : A
      ⊢ Eq (CategoryTheory.Quotient.LiftCommShift.iso F r hF (HAdd.hAdd a b)) (Categ …
    -/
    ext1
    /-
      case w
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝⁵ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      r : HomRel C
      A : Type w
      inst✝⁴ : AddMonoid A
      inst✝³ : CategoryTheory.HasShift C A
      inst✝² : CategoryTheory.HasShift D A
      inst✝¹ : r.IsCompatibleWithShift A
      inst✝ : F.CommShift A
      hF : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (F.map f₁) (F.map f₂)
      a b : A
      ⊢ Eq (CategoryTheory.Quotient.LiftCommShift.iso F r hF (HAdd.hAdd a b)).hom (C …
    -/
    apply natTrans_ext
    /-
      case w.h
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝⁵ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      r : HomRel C
      A : Type w
      inst✝⁴ : AddMonoid A
      inst✝³ : CategoryTheory.HasShift C A
      inst✝² : CategoryTheory.HasShift D A
      inst✝¹ : r.IsCompatibleWithShift A
      inst✝ : F.CommShift A
      hF : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (F.map f₁) (F.map f₂)
      a b : A
      ⊢ Eq (CategoryTheory.whiskerLeft (CategoryTheory.Quotient.functor r) (Category …
    -/
    ext X
    /-
      case w.h.w.h
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝⁵ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      r : HomRel C
      A : Type w
      inst✝⁴ : AddMonoid A
      inst✝³ : CategoryTheory.HasShift C A
      inst✝² : CategoryTheory.HasShift D A
      inst✝¹ : r.IsCompatibleWithShift A
      inst✝ : F.CommShift A
      hF : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (F.map f₁) (F.map f₂)
      a b : A
      X : C
      ⊢ Eq ((CategoryTheory.whiskerLeft (CategoryTheory.Quotient.functor r) (Categor …
    -/
    dsimp
    rw [LiftCommShift.iso_hom_app, (functor r).commShiftIso_add, F.commShiftIso_add,
      Functor.CommShift.isoAdd_hom_app, Functor.CommShift.isoAdd_hom_app,
      Functor.CommShift.isoAdd_inv_app, Functor.map_comp, Functor.map_comp,
      Functor.map_comp, assoc, assoc, assoc, LiftCommShift.iso_hom_app, lift_map_functor_map]
    /-
      case w.h.w.h
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝⁵ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      r : HomRel C
      A : Type w
      inst✝⁴ : AddMonoid A
      inst✝³ : CategoryTheory.HasShift C A
      inst✝² : CategoryTheory.HasShift D A
      inst✝¹ : r.IsCompatibleWithShift A
      inst✝ : F.CommShift A
      hF : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (F.map f₁) (F.map f₂)
      a b : A
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Quotient.lift r F hF …
    -/
    congr 1
    rw [← cancel_epi ((shiftFunctor (Quotient r) b ⋙ lift r F hF).map
      (NatTrans.app (Functor.commShiftIso (functor r) a).hom X))]
    erw [(LiftCommShift.iso F r hF b).hom.naturality_assoc
      (((functor r).commShiftIso a).hom.app X), LiftCommShift.iso_hom_app,
      ← Functor.map_comp_assoc, Iso.hom_inv_id_app]
    /-
      case w.h.w.h.e_a
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝⁵ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      r : HomRel C
      A : Type w
      inst✝⁴ : AddMonoid A
      inst✝³ : CategoryTheory.HasShift C A
      inst✝² : CategoryTheory.HasShift D A
      inst✝¹ : r.IsCompatibleWithShift A
      inst✝ : F.CommShift A
      hF : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (F.map f₁) (F.map f₂)
      a b : A
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.shiftFunctor (Categ …
    -/
    dsimp
    simp only [Functor.comp_obj, assoc, ← Functor.map_comp_assoc, Iso.inv_hom_id_app,
      Functor.map_id, id_comp, Iso.hom_inv_id_app, lift_obj_functor_obj]


instance liftCommShift_compatibility :
    NatTrans.CommShift (Quotient.lift.isLift r F hF).hom A where
  shift_comm a := by
    /-
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝⁵ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      r : HomRel C
      A : Type w
      inst✝⁴ : AddMonoid A
      inst✝³ : CategoryTheory.HasShift C A
      inst✝² : CategoryTheory.HasShift D A
      inst✝¹ : r.IsCompatibleWithShift A
      inst✝ : F.CommShift A
      hF : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (F.map f₁) (F.map f₂)
      a : A
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Quotient.functor r) …
    -/
    ext X
    /-
      case w.h
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝⁵ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      r : HomRel C
      A : Type w
      inst✝⁴ : AddMonoid A
      inst✝³ : CategoryTheory.HasShift C A
      inst✝² : CategoryTheory.HasShift D A
      inst✝¹ : r.IsCompatibleWithShift A
      inst✝ : F.CommShift A
      hF : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (F.map f₁) (F.map f₂)
      a : A
      X : C
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (((CategoryTheory.Quotient.functor r …
    -/
    dsimp
    /-
      case w.h
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝⁵ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      r : HomRel C
      A : Type w
      inst✝⁴ : AddMonoid A
      inst✝³ : CategoryTheory.HasShift C A
      inst✝² : CategoryTheory.HasShift D A
      inst✝¹ : r.IsCompatibleWithShift A
      inst✝ : F.CommShift A
      hF : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (F.map f₁) (F.map f₂)
      a : A
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((((CategoryTheory.Quotient.functor r …
    -/
    erw [Functor.map_id, id_comp, comp_id]
    /-
      case w.h
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝⁵ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      r : HomRel C
      A : Type w
      inst✝⁴ : AddMonoid A
      inst✝³ : CategoryTheory.HasShift C A
      inst✝² : CategoryTheory.HasShift D A
      inst✝¹ : r.IsCompatibleWithShift A
      inst✝ : F.CommShift A
      hF : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (F.map f₁) (F.map f₂)
      a : A
      X : C
      ⊢ Eq ((((CategoryTheory.Quotient.functor r).comp (CategoryTheory.Quotient.lift …
    -/
    rw [Functor.commShiftIso_comp_hom_app]
    /-
      case w.h
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝⁵ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      r : HomRel C
      A : Type w
      inst✝⁴ : AddMonoid A
      inst✝³ : CategoryTheory.HasShift C A
      inst✝² : CategoryTheory.HasShift D A
      inst✝¹ : r.IsCompatibleWithShift A
      inst✝ : F.CommShift A
      hF : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (F.map f₁) (F.map f₂)
      a : A
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Quotient.lift r F hF …
    -/
    erw [LiftCommShift.iso_hom_app]
    /-
      case w.h
      C : Type u
      inst✝⁶ : CategoryTheory.Category.{v, u} C
      D : Type u'
      inst✝⁵ : CategoryTheory.Category.{v', u'} D
      F : CategoryTheory.Functor C D
      r : HomRel C
      A : Type w
      inst✝⁴ : AddMonoid A
      inst✝³ : CategoryTheory.HasShift C A
      inst✝² : CategoryTheory.HasShift D A
      inst✝¹ : r.IsCompatibleWithShift A
      inst✝ : F.CommShift A
      hF : ∀ (x y : C) (f₁ f₂ : Quiver.Hom x y), r f₁ f₂ → Eq (F.map f₁) (F.map f₂)
      a : A
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Quotient.lift r F hF …
    -/
    rw [← Functor.map_comp_assoc, Iso.hom_inv_id_app, Functor.map_id, id_comp]
    /-
      🎉 no goals
    -/


