/-- If `F` commutes with shifts, so does `F.op`, for the shifts chosen on `Cᵒᵖ` in
`CategoryTheory.Triangulated.Opposite.Basic`.
-/
noncomputable scoped instance commShiftOpInt : F.op.CommShift ℤ := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{?u.185, u_1} C
    inst✝³ : CategoryTheory.Category.{?u.189, u_2} D
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.HasShift D Int
    F : CategoryTheory.Functor C D
    inst✝ : F.CommShift Int
    ⊢ F.op.CommShift Int
  -/
  letI F' : OppositeShift C ℤ ⥤ OppositeShift D ℤ := F.op
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{?u.185, u_1} C
    inst✝³ : CategoryTheory.Category.{?u.189, u_2} D
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.HasShift D Int
    F : CategoryTheory.Functor C D
    inst✝ : F.CommShift Int
    F' : CategoryTheory.Functor (CategoryTheory.OppositeShift C Int) (CategoryTheo …
    ⊢ F.op.CommShift Int
  -/
  letI : F'.CommShift ℤ := F.commShiftOp ℤ
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{?u.185, u_1} C
    inst✝³ : CategoryTheory.Category.{?u.189, u_2} D
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.HasShift D Int
    F : CategoryTheory.Functor C D
    inst✝ : F.CommShift Int
    F' : CategoryTheory.Functor (CategoryTheory.OppositeShift C Int) (CategoryTheo …
    this : F'.CommShift Int := F.commShiftOp Int
    ⊢ F.op.CommShift Int
  -/
  apply F'.commShiftPullback
  /-
    🎉 no goals
  -/


@[reassoc]
lemma op_commShiftIso_hom_app (X : Cᵒᵖ) (n m : ℤ) (h : n + m = 0):
    (F.op.commShiftIso n).hom.app X =
      (F.map ((shiftFunctorOpIso C n m h).hom.app X).unop).op ≫
        ((F.commShiftIso m).inv.app X.unop).op ≫
        (shiftFunctorOpIso D n m h).inv.app (op (F.obj X.unop)) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.HasShift D Int
    F : CategoryTheory.Functor C D
    inst✝ : F.CommShift Int
    X : Opposite C
    n m : Int
    h : Eq (HAdd.hAdd n m) 0
    ⊢ Eq ((F.op.commShiftIso n).hom.app X) (CategoryTheory.CategoryStruct.comp (F. …
  -/
  obtain rfl : m = -n := by omega
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.HasShift D Int
    F : CategoryTheory.Functor C D
    inst✝ : F.CommShift Int
    X : Opposite C
    n : Int
    h : Eq (HAdd.hAdd n (Neg.neg n)) 0
    ⊢ Eq ((F.op.commShiftIso n).hom.app X) (CategoryTheory.CategoryStruct.comp (F. …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[reassoc]
lemma op_commShiftIso_inv_app (X : Cᵒᵖ) (n m : ℤ) (h : n + m = 0):
    (F.op.commShiftIso n).inv.app X =
      (shiftFunctorOpIso D n m h).hom.app (op (F.obj X.unop)) ≫
        ((F.commShiftIso m).hom.app X.unop).op ≫
          (F.map ((shiftFunctorOpIso C n m h).inv.app X).unop).op := by
  rw [← cancel_epi ((F.op.commShiftIso n).hom.app X), Iso.hom_inv_id_app,
    op_commShiftIso_hom_app _ X n m h, assoc, assoc]
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.HasShift D Int
    F : CategoryTheory.Functor C D
    inst✝ : F.CommShift Int
    X : Opposite C
    n m : Int
    h : Eq (HAdd.hAdd n m) 0
    ⊢ Eq (CategoryTheory.CategoryStruct.id (((CategoryTheory.shiftFunctor (Opposit …
  -/
  simp [← op_comp, ← F.map_comp]
  /-
    🎉 no goals
  -/


@[reassoc]
lemma shift_map_op {X Y : C} (f : X ⟶ Y) (n : ℤ) :
    (F.map f).op⟦n⟧' = (F.op.commShiftIso n).inv.app _ ≫
      (F.map (f.op⟦n⟧').unop).op ≫ (F.op.commShiftIso n).hom.app _ :=
  (NatIso.naturality_1 (F.op.commShiftIso n) f.op).symm


@[reassoc]
lemma map_shift_unop {X Y : Cᵒᵖ} (f : X ⟶ Y) (n : ℤ) :
    F.map ((f⟦n⟧').unop) = ((F.op.commShiftIso n).inv.app Y).unop ≫
      ((F.map f.unop).op⟦n⟧').unop ≫ ((F.op.commShiftIso n).hom.app X).unop := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝³ : CategoryTheory.Category.{u_4, u_2} D
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.HasShift D Int
    F : CategoryTheory.Functor C D
    inst✝ : F.CommShift Int
    X Y : Opposite C
    f : Quiver.Hom X Y
    n : Int
    ⊢ Eq (F.map ((CategoryTheory.shiftFunctor (Opposite C) n).map f).unop) (Catego …
  -/
  simp [shift_map_op]
  /-
    🎉 no goals
  -/


@[reassoc]
lemma map_opShiftFunctorEquivalence_unitIso_hom_app_unop (X : Cᵒᵖ) (n : ℤ) :
    F.map ((opShiftFunctorEquivalence C n).unitIso.hom.app X).unop =
      (F.commShiftIso n).hom.app _ ≫
        (((F.op).commShiftIso n).inv.app X).unop⟦n⟧' ≫
        ((opShiftFunctorEquivalence D n).unitIso.hom.app (op _)).unop := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.HasShift D Int
    F : CategoryTheory.Functor C D
    inst✝ : F.CommShift Int
    X : Opposite C
    n : Int
    ⊢ Eq (F.map ((CategoryTheory.Pretriangulated.opShiftFunctorEquivalence C n).un …
  -/
  dsimp [opShiftFunctorEquivalence]
  simp only [map_comp, unop_comp, Quiver.Hom.unop_op, assoc,
    map_shiftFunctorCompIsoId_hom_app, commShiftIso_hom_naturality_assoc,
    op_commShiftIso_inv_app _ _ _ _ (add_neg_cancel n)]
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.HasShift D Int
    F : CategoryTheory.Functor C D
    inst✝ : F.CommShift Int
    X : Opposite C
    n : Int
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.commShiftIso n).hom.app (Opposite …
  -/
  congr 3
  rw [← Functor.map_comp_assoc, ← unop_comp,
    Iso.inv_hom_id_app]
  /-
    case e_a.e_a.e_a
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.HasShift D Int
    F : CategoryTheory.Functor C D
    inst✝ : F.CommShift Int
    X : Opposite C
    n : Int
    ⊢ Eq ((CategoryTheory.shiftFunctorCompIsoId D (Neg.neg n) n ⋯).hom.app (F.obj  …
  -/
  dsimp
  /-
    case e_a.e_a.e_a
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.HasShift D Int
    F : CategoryTheory.Functor C D
    inst✝ : F.CommShift Int
    X : Opposite C
    n : Int
    ⊢ Eq ((CategoryTheory.shiftFunctorCompIsoId D (Neg.neg n) n ⋯).hom.app (F.obj  …
  -/
  rw [map_id, id_comp]
  /-
    🎉 no goals
  -/


@[reassoc]
lemma map_opShiftFunctorEquivalence_unitIso_inv_app_unop (X : Cᵒᵖ) (n : ℤ) :
    F.map ((opShiftFunctorEquivalence C n).unitIso.inv.app X).unop =
      ((opShiftFunctorEquivalence D n).unitIso.inv.app (op (F.obj X.unop))).unop ≫
        (((F.op).commShiftIso n).hom.app X).unop⟦n⟧' ≫
        ((F.commShiftIso n).inv.app _) := by
  rw [← cancel_mono (F.map ((opShiftFunctorEquivalence C n).unitIso.hom.app X).unop),
    ← F.map_comp, ← unop_comp, Iso.hom_inv_id_app,
    map_opShiftFunctorEquivalence_unitIso_hom_app_unop, assoc, assoc,
    Iso.inv_hom_id_app_assoc, ← Functor.map_comp_assoc, ← unop_comp]
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.HasShift D Int
    F : CategoryTheory.Functor C D
    inst✝ : F.CommShift Int
    X : Opposite C
    n : Int
    ⊢ Eq (F.map (CategoryTheory.CategoryStruct.id ((CategoryTheory.Functor.id (Opp …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc]
lemma map_opShiftFunctorEquivalence_counitIso_hom_app_unop (X : Cᵒᵖ) (n : ℤ) :
    F.map ((opShiftFunctorEquivalence C n).counitIso.hom.app X).unop =
      ((opShiftFunctorEquivalence D n).counitIso.hom.app (op (F.obj X.unop))).unop ≫
        (((F.commShiftIso n).inv.app X.unop).op⟦n⟧').unop ≫
          ((F.op.commShiftIso n).hom.app (op (X.unop⟦n⟧))).unop := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.HasShift D Int
    F : CategoryTheory.Functor C D
    inst✝ : F.CommShift Int
    X : Opposite C
    n : Int
    ⊢ Eq (F.map ((CategoryTheory.Pretriangulated.opShiftFunctorEquivalence C n).co …
  -/
  apply Quiver.Hom.op_inj
  /-
    case a
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.HasShift D Int
    F : CategoryTheory.Functor C D
    inst✝ : F.CommShift Int
    X : Opposite C
    n : Int
    ⊢ Eq (F.map ((CategoryTheory.Pretriangulated.opShiftFunctorEquivalence C n).co …
  -/
  dsimp [opShiftFunctorEquivalence]
  rw [assoc, F.op_commShiftIso_hom_app_assoc _ _ _ (add_neg_cancel n), map_comp,
    map_shiftFunctorCompIsoId_inv_app_assoc, op_comp, op_comp_assoc, op_comp_assoc,
    NatTrans.naturality_assoc, op_map, Iso.inv_hom_id_app_assoc, Quiver.Hom.unop_op]


@[reassoc]
lemma map_opShiftFunctorEquivalence_counitIso_inv_app_unop (X : Cᵒᵖ) (n : ℤ) :
    F.map ((opShiftFunctorEquivalence C n).counitIso.inv.app X).unop =
      ((F.op.commShiftIso n).inv.app (op (X.unop⟦n⟧))).unop ≫
        (((F.commShiftIso n).hom.app X.unop).op⟦n⟧').unop ≫
          ((opShiftFunctorEquivalence D n).counitIso.inv.app (op (F.obj X.unop))).unop := by
  rw [← cancel_epi (F.map ((opShiftFunctorEquivalence C n).counitIso.hom.app X).unop),
    ← F.map_comp, ← unop_comp, Iso.inv_hom_id_app,
    map_opShiftFunctorEquivalence_counitIso_hom_app_unop]
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.HasShift D Int
    F : CategoryTheory.Functor C D
    inst✝ : F.CommShift Int
    X : Opposite C
    n : Int
    ⊢ Eq (F.map (CategoryTheory.CategoryStruct.id ((CategoryTheory.Functor.id (Opp …
  -/
  dsimp
  simp only [map_id, assoc, Iso.unop_hom_inv_id_app_assoc, ← Functor.map_comp_assoc,
    ← unop_comp, Iso.inv_hom_id_app_assoc, ← unop_comp_assoc, ← op_comp,
    Iso.inv_hom_id_app]
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_3, u_2} D
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.HasShift D Int
    F : CategoryTheory.Functor C D
    inst✝ : F.CommShift Int
    X : Opposite C
    n : Int
    ⊢ Eq (CategoryTheory.CategoryStruct.id (F.obj (Opposite.unop X))) (CategoryThe …
  -/
  simp
  /-
    🎉 no goals
  -/


/--
If `F : C ⥤ D` commutes with shifts, this expresses the compatibility of `F.mapTriangle`
with the equivalences `Pretriangulated.triangleOpEquivalence` on `C` and `D`.
-/
@[simps!]
noncomputable def mapTriangleOpCompTriangleOpEquivalenceFunctorApp (T : Triangle C) :
    (triangleOpEquivalence D).functor.obj (op (F.mapTriangle.obj T)) ≅
      F.op.mapTriangle.obj ((triangleOpEquivalence C).functor.obj (op T)) :=
  Triangle.isoMk _ _ (Iso.refl _) (Iso.refl _) (Iso.refl _)
        /-
          C : Type u_1
          D : Type u_2
          inst✝¹² : CategoryTheory.Category.{?u.79827, u_1} C
          inst✝¹¹ : CategoryTheory.Category.{?u.79831, u_2} D
          inst✝¹⁰ : CategoryTheory.HasShift C Int
          inst✝⁹ : CategoryTheory.HasShift D Int
          F : CategoryTheory.Functor C D
          inst✝⁸ : F.CommShift Int
          inst✝⁷ : CategoryTheory.Limits.HasZeroObject C
          inst✝⁶ : CategoryTheory.Preadditive C
          inst✝⁵ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
          inst✝⁴ : CategoryTheory.Pretriangulated C
          inst✝³ : CategoryTheory.Limits.HasZeroObject D
          inst✝² : CategoryTheory.Preadditive D
          inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
          inst✝ : CategoryTheory.Pretriangulated D
          T : CategoryTheory.Pretriangulated.Triangle C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Pretriangulated.tria …
        -/
               /-
                 🎉 no goals
               -/
    (by dsimp; simp) (by dsimp; simp) (by
                                /-
                                  🎉 no goals
                                -/
      /-
        C : Type u_1
        D : Type u_2
        inst✝¹² : CategoryTheory.Category.{?u.79827, u_1} C
        inst✝¹¹ : CategoryTheory.Category.{?u.79831, u_2} D
        inst✝¹⁰ : CategoryTheory.HasShift C Int
        inst✝⁹ : CategoryTheory.HasShift D Int
        F : CategoryTheory.Functor C D
        inst✝⁸ : F.CommShift Int
        inst✝⁷ : CategoryTheory.Limits.HasZeroObject C
        inst✝⁶ : CategoryTheory.Preadditive C
        inst✝⁵ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
        inst✝⁴ : CategoryTheory.Pretriangulated C
        inst✝³ : CategoryTheory.Limits.HasZeroObject D
        inst✝² : CategoryTheory.Preadditive D
        inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
        inst✝ : CategoryTheory.Pretriangulated D
        T : CategoryTheory.Pretriangulated.Triangle C
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Pretriangulated.tria …
      -/
      dsimp
      simp only [map_comp, shift_map_op, map_id, comp_id, op_comp, op_unop,
        map_opShiftFunctorEquivalence_counitIso_inv_app_unop,
        opShiftFunctorEquivalence_inverse, opShiftFunctorEquivalence_functor,
        Quiver.Hom.op_unop, assoc, id_comp])


/--
If `F : C ⥤ D` commutes with shifts, this expresses the compatibility of `F.mapTriangle`
with the equivalences `Pretriangulated.triangleOpEquivalence` on `C` and `D`.
-/
noncomputable def mapTriangleOpCompTriangleOpEquivalenceFunctor :
    F.mapTriangle.op ⋙ (triangleOpEquivalence D).functor ≅
      (triangleOpEquivalence C).functor ⋙ F.op.mapTriangle :=
  NatIso.ofComponents
    (fun T ↦ F.mapTriangleOpCompTriangleOpEquivalenceFunctorApp T.unop)
        /-
          C : Type u_1
          D : Type u_2
          inst✝¹² : CategoryTheory.Category.{?u.96638, u_1} C
          inst✝¹¹ : CategoryTheory.Category.{?u.96642, u_2} D
          inst✝¹⁰ : CategoryTheory.HasShift C Int
          inst✝⁹ : CategoryTheory.HasShift D Int
          F : CategoryTheory.Functor C D
          inst✝⁸ : F.CommShift Int
          inst✝⁷ : CategoryTheory.Limits.HasZeroObject C
          inst✝⁶ : CategoryTheory.Preadditive C
          inst✝⁵ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
          inst✝⁴ : CategoryTheory.Pretriangulated C
          inst✝³ : CategoryTheory.Limits.HasZeroObject D
          inst✝² : CategoryTheory.Preadditive D
          inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
          inst✝ : CategoryTheory.Pretriangulated D
          ⊢ ∀ {X Y : Opposite (CategoryTheory.Pretriangulated.Triangle C)} (f : Quiver.H …
        -/
                                  /-
                                    🎉 no goals
                                  -/
                                  /-
                                    🎉 no goals
                                  -/
    (by intros; ext <;> dsimp <;> simp only [comp_id, id_comp])
                                  /-
                                    🎉 no goals
                                  -/


/--
If `F : C ⥤ D` commutes with shifts, this is the 2-commutative square of categories
`CategoryTheory.Functor.mapTriangleOpCompTriangleOpEquivalenceFunctor`.
-/
noncomputable instance :
    CatCommSq (F.mapTriangle.op) (triangleOpEquivalence C).functor
      (triangleOpEquivalence D).functor F.op.mapTriangle :=
  ⟨F.mapTriangleOpCompTriangleOpEquivalenceFunctor⟩


/--
Vertical inverse of the 2-commutative square of
`CategoryTheory.Functor.mapTriangleOpCompTriangleOpEquivalenceFunctor`.
-/
noncomputable instance :
    CatCommSq (F.op.mapTriangle) (triangleOpEquivalence C).inverse
      (triangleOpEquivalence D).inverse F.mapTriangle.op :=
  CatCommSq.vInv (F.mapTriangle.op) (triangleOpEquivalence C)
      (triangleOpEquivalence D) F.op.mapTriangle inferInstance


/--
If `F : C ⥤ D` commutes with shifts, this expresses the compatibility of `F.mapTriangle`
with the equivalences `Pretriangulated.triangleOpEquivalence` on `C` and `D`.
-/
noncomputable def opMapTriangleCompTriangleOpEquivalenceInverse :
    F.op.mapTriangle ⋙ (triangleOpEquivalence D).inverse ≅
      (triangleOpEquivalence C).inverse ⋙ F.mapTriangle.op :=
  CatCommSq.iso (F.op.mapTriangle) (triangleOpEquivalence C).inverse
      (triangleOpEquivalence D).inverse F.mapTriangle.op


/-- If `F` is triangulated, so is `F.op`.
-/
lemma isTriangulated_op [F.IsTriangulated] : F.op.IsTriangulated where
  map_distinguished T dT := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝¹³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹² : CategoryTheory.Category.{u_4, u_2} D
      inst✝¹¹ : CategoryTheory.HasShift C Int
      inst✝¹⁰ : CategoryTheory.HasShift D Int
      F : CategoryTheory.Functor C D
      inst✝⁹ : F.CommShift Int
      inst✝⁸ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁷ : CategoryTheory.Preadditive C
      inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝⁵ : CategoryTheory.Pretriangulated C
      inst✝⁴ : CategoryTheory.Limits.HasZeroObject D
      inst✝³ : CategoryTheory.Preadditive D
      inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
      inst✝¹ : CategoryTheory.Pretriangulated D
      inst✝ : F.IsTriangulated
      T : CategoryTheory.Pretriangulated.Triangle (Opposite C)
      dT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      ⊢ Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (F.op.m …
    -/
    rw [mem_distTriang_op_iff]
    exact Pretriangulated.isomorphic_distinguished _
      ((F.map_distinguished _ (unop_distinguished _ dT))) _
      (((opMapTriangleCompTriangleOpEquivalenceInverse F).symm.app T).unop)


/-- If `F.op` is triangulated, so is `F`.
-/
lemma isTriangulated_of_op [F.op.IsTriangulated] : F.IsTriangulated where
  map_distinguished T dT := by
    have := distinguished_iff_of_iso ((triangleOpEquivalence D).unitIso.app
      (Opposite.op (F.mapTriangle.obj T))).unop
    /-
      C : Type u_1
      D : Type u_2
      inst✝¹³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹² : CategoryTheory.Category.{u_4, u_2} D
      inst✝¹¹ : CategoryTheory.HasShift C Int
      inst✝¹⁰ : CategoryTheory.HasShift D Int
      F : CategoryTheory.Functor C D
      inst✝⁹ : F.CommShift Int
      inst✝⁸ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁷ : CategoryTheory.Preadditive C
      inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝⁵ : CategoryTheory.Pretriangulated C
      inst✝⁴ : CategoryTheory.Limits.HasZeroObject D
      inst✝³ : CategoryTheory.Preadditive D
      inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
      inst✝¹ : CategoryTheory.Pretriangulated D
      inst✝ : F.op.IsTriangulated
      T : CategoryTheory.Pretriangulated.Triangle C
      dT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      this : Iff (Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangl …
      ⊢ Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (F.mapT …
    -/
    rw [Functor.id_obj, Opposite.unop_op (F.mapTriangle.obj T)] at this
    rw [← this, Functor.comp_obj, ← mem_distTriang_op_iff, ← Functor.op_obj, ← Functor.comp_obj,
      distinguished_iff_of_iso ((mapTriangleOpCompTriangleOpEquivalenceFunctor F).app
      (Opposite.op T))]
    /-
      C : Type u_1
      D : Type u_2
      inst✝¹³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹² : CategoryTheory.Category.{u_4, u_2} D
      inst✝¹¹ : CategoryTheory.HasShift C Int
      inst✝¹⁰ : CategoryTheory.HasShift D Int
      F : CategoryTheory.Functor C D
      inst✝⁹ : F.CommShift Int
      inst✝⁸ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁷ : CategoryTheory.Preadditive C
      inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝⁵ : CategoryTheory.Pretriangulated C
      inst✝⁴ : CategoryTheory.Limits.HasZeroObject D
      inst✝³ : CategoryTheory.Preadditive D
      inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
      inst✝¹ : CategoryTheory.Pretriangulated D
      inst✝ : F.op.IsTriangulated
      T : CategoryTheory.Pretriangulated.Triangle C
      dT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      this : Iff (Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangl …
      ⊢ Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (((Cate …
    -/
    apply F.op.map_distinguished
    /-
      case hT
      C : Type u_1
      D : Type u_2
      inst✝¹³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹² : CategoryTheory.Category.{u_4, u_2} D
      inst✝¹¹ : CategoryTheory.HasShift C Int
      inst✝¹⁰ : CategoryTheory.HasShift D Int
      F : CategoryTheory.Functor C D
      inst✝⁹ : F.CommShift Int
      inst✝⁸ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁷ : CategoryTheory.Preadditive C
      inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝⁵ : CategoryTheory.Pretriangulated C
      inst✝⁴ : CategoryTheory.Limits.HasZeroObject D
      inst✝³ : CategoryTheory.Preadditive D
      inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
      inst✝¹ : CategoryTheory.Pretriangulated D
      inst✝ : F.op.IsTriangulated
      T : CategoryTheory.Pretriangulated.Triangle C
      dT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      this : Iff (Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangl …
      ⊢ Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles ((Categ …
    -/
    have := distinguished_iff_of_iso ((triangleOpEquivalence C).unitIso.app (Opposite.op T)).unop
    /-
      case hT
      C : Type u_1
      D : Type u_2
      inst✝¹³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹² : CategoryTheory.Category.{u_4, u_2} D
      inst✝¹¹ : CategoryTheory.HasShift C Int
      inst✝¹⁰ : CategoryTheory.HasShift D Int
      F : CategoryTheory.Functor C D
      inst✝⁹ : F.CommShift Int
      inst✝⁸ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁷ : CategoryTheory.Preadditive C
      inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝⁵ : CategoryTheory.Pretriangulated C
      inst✝⁴ : CategoryTheory.Limits.HasZeroObject D
      inst✝³ : CategoryTheory.Preadditive D
      inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
      inst✝¹ : CategoryTheory.Pretriangulated D
      inst✝ : F.op.IsTriangulated
      T : CategoryTheory.Pretriangulated.Triangle C
      dT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      this✝ : Iff (Membership.mem CategoryTheory.Pretriangulated.distinguishedTriang …
      this : Iff (Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangl …
      ⊢ Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles ((Categ …
    -/
    rw [Functor.id_obj, Opposite.unop_op T] at this
    /-
      case hT
      C : Type u_1
      D : Type u_2
      inst✝¹³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹² : CategoryTheory.Category.{u_4, u_2} D
      inst✝¹¹ : CategoryTheory.HasShift C Int
      inst✝¹⁰ : CategoryTheory.HasShift D Int
      F : CategoryTheory.Functor C D
      inst✝⁹ : F.CommShift Int
      inst✝⁸ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁷ : CategoryTheory.Preadditive C
      inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝⁵ : CategoryTheory.Pretriangulated C
      inst✝⁴ : CategoryTheory.Limits.HasZeroObject D
      inst✝³ : CategoryTheory.Preadditive D
      inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
      inst✝¹ : CategoryTheory.Pretriangulated D
      inst✝ : F.op.IsTriangulated
      T : CategoryTheory.Pretriangulated.Triangle C
      dT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      this✝ : Iff (Membership.mem CategoryTheory.Pretriangulated.distinguishedTriang …
      this : Iff (Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangl …
      ⊢ Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles ((Categ …
    -/
    rw [← this, Functor.comp_obj, ← mem_distTriang_op_iff] at dT
    /-
      case hT
      C : Type u_1
      D : Type u_2
      inst✝¹³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹² : CategoryTheory.Category.{u_4, u_2} D
      inst✝¹¹ : CategoryTheory.HasShift C Int
      inst✝¹⁰ : CategoryTheory.HasShift D Int
      F : CategoryTheory.Functor C D
      inst✝⁹ : F.CommShift Int
      inst✝⁸ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁷ : CategoryTheory.Preadditive C
      inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝⁵ : CategoryTheory.Pretriangulated C
      inst✝⁴ : CategoryTheory.Limits.HasZeroObject D
      inst✝³ : CategoryTheory.Preadditive D
      inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
      inst✝¹ : CategoryTheory.Pretriangulated D
      inst✝ : F.op.IsTriangulated
      T : CategoryTheory.Pretriangulated.Triangle C
      dT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles ((Ca …
      this✝ : Iff (Membership.mem CategoryTheory.Pretriangulated.distinguishedTriang …
      this : Iff (Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangl …
      ⊢ Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles ((Categ …
    -/
    exact dT
    /-
      🎉 no goals
    -/


/-- `F` is triangulated if and only if `F.op` is triangulated.
-/
lemma op_isTriangulated_iff : F.op.IsTriangulated ↔ F.IsTriangulated :=
  ⟨fun _ ↦ F.isTriangulated_of_op, fun _ ↦ F.isTriangulated_op⟩


