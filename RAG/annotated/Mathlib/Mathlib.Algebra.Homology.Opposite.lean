theorem imageToKernel_op {X Y Z : V} (f : X ⟶ Y) (g : Y ⟶ Z) (w : f ≫ g = 0) :
                                /-
                                  V : Type u_1
                                  inst✝¹ : CategoryTheory.Category.{?u.29, u_1} V
                                  inst✝ : CategoryTheory.Abelian V
                                  X Y Z : V
                                  f : Quiver.Hom X Y
                                  g : Quiver.Hom Y Z
                                  w : Eq (CategoryTheory.CategoryStruct.comp f g) 0
                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp g.op f.op) 0
                                -/
    imageToKernel g.op f.op (by rw [← op_comp, w, op_zero]) =
                                /-
                                  🎉 no goals
                                -/
      (imageSubobjectIso _ ≪≫ (imageOpOp _).symm).hom ≫
        (cokernel.desc f (factorThruImage g)
                  /-
                    V : Type u_1
                    inst✝¹ : CategoryTheory.Category.{?u.29, u_1} V
                    inst✝ : CategoryTheory.Abelian V
                    X Y Z : V
                    f : Quiver.Hom X Y
                    g : Quiver.Hom Y Z
                    w : Eq (CategoryTheory.CategoryStruct.comp f g) 0
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.factorThruIm …
                  -/
              (by rw [← cancel_mono (image.ι g), Category.assoc, image.fac, w, zero_comp])).op ≫
                  /-
                    🎉 no goals
                  -/
          (kernelSubobjectIso _ ≪≫ kernelOpOp _).inv := by
  /-
    V : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} V
    inst✝ : CategoryTheory.Abelian V
    X Y Z : V
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    w : Eq (CategoryTheory.CategoryStruct.comp f g) 0
    ⊢ Eq (imageToKernel g.op f.op ⋯) (CategoryTheory.CategoryStruct.comp ((Categor …
  -/
  ext
  simp only [Iso.trans_hom, Iso.symm_hom, Iso.trans_inv, kernelOpOp_inv, Category.assoc,
    imageToKernel_arrow, kernelSubobject_arrow', kernel.lift_ι, ← op_comp, cokernel.π_desc,
    ← imageSubobject_arrow, ← imageUnopOp_inv_comp_op_factorThruImage g.op]
  /-
    case h
    V : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} V
    inst✝ : CategoryTheory.Abelian V
    X Y Z : V
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    w : Eq (CategoryTheory.CategoryStruct.comp f g) 0
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.imageSubobject …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem imageToKernel_unop {X Y Z : Vᵒᵖ} (f : X ⟶ Y) (g : Y ⟶ Z) (w : f ≫ g = 0) :
                                    /-
                                      V : Type u_1
                                      inst✝¹ : CategoryTheory.Category.{?u.9343, u_1} V
                                      inst✝ : CategoryTheory.Abelian V
                                      X Y Z : Opposite V
                                      f : Quiver.Hom X Y
                                      g : Quiver.Hom Y Z
                                      w : Eq (CategoryTheory.CategoryStruct.comp f g) 0
                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp g.unop f.unop) 0
                                    -/
    imageToKernel g.unop f.unop (by rw [← unop_comp, w, unop_zero]) =
                                    /-
                                      🎉 no goals
                                    -/
      (imageSubobjectIso _ ≪≫ (imageUnopUnop _).symm).hom ≫
        (cokernel.desc f (factorThruImage g)
                  /-
                    V : Type u_1
                    inst✝¹ : CategoryTheory.Category.{?u.9343, u_1} V
                    inst✝ : CategoryTheory.Abelian V
                    X Y Z : Opposite V
                    f : Quiver.Hom X Y
                    g : Quiver.Hom Y Z
                    w : Eq (CategoryTheory.CategoryStruct.comp f g) 0
                    ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.factorThruIm …
                  -/
              (by rw [← cancel_mono (image.ι g), Category.assoc, image.fac, w, zero_comp])).unop ≫
                  /-
                    🎉 no goals
                  -/
          (kernelSubobjectIso _ ≪≫ kernelUnopUnop _).inv := by
  /-
    V : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} V
    inst✝ : CategoryTheory.Abelian V
    X Y Z : Opposite V
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    w : Eq (CategoryTheory.CategoryStruct.comp f g) 0
    ⊢ Eq (imageToKernel g.unop f.unop ⋯) (CategoryTheory.CategoryStruct.comp ((Cat …
  -/
  ext
  /-
    case h
    V : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} V
    inst✝ : CategoryTheory.Abelian V
    X Y Z : Opposite V
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    w : Eq (CategoryTheory.CategoryStruct.comp f g) 0
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (imageToKernel g.unop f.unop ⋯) (Cate …
  -/
  dsimp only [imageUnopUnop]
  simp only [Iso.trans_hom, Iso.symm_hom, Iso.trans_inv, kernelUnopUnop_inv, Category.assoc,
    imageToKernel_arrow, kernelSubobject_arrow', kernel.lift_ι, cokernel.π_desc, Iso.unop_inv,
    ← unop_comp, factorThruImage_comp_imageUnopOp_inv, Quiver.Hom.unop_op, imageSubobject_arrow]


/-- Sends a complex `X` with objects in `V` to the corresponding complex with objects in `Vᵒᵖ`. -/
@[simps]
protected def op (X : HomologicalComplex V c) : HomologicalComplex Vᵒᵖ c.symm where
  X i := op (X.X i)
  d i j := (X.d j i).op
                      /-
                        ι : Type u_1
                        V : Type u_2
                        inst✝¹ : CategoryTheory.Category.{?u.19714, u_2} V
                        c : ComplexShape ι
                        inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
                        X : HomologicalComplex V c
                        i j : ι
                        hij : Not (c.symm.Rel i j)
                        ⊢ Eq ((fun i j => (X.d j i).op) i j) 0
                      -/
  shape i j hij := by simp only; rw [X.shape j i hij, op_zero]
                                 /-
                                   🎉 no goals
                                 -/
                            /-
                              ι : Type u_1
                              V : Type u_2
                              inst✝¹ : CategoryTheory.Category.{?u.19714, u_2} V
                              c : ComplexShape ι
                              inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
                              X : HomologicalComplex V c
                              x✝⁴ x✝³ x✝² : ι
                              x✝¹ : c.symm.Rel x✝⁴ x✝³
                              x✝ : c.symm.Rel x✝³ x✝²
                              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i j => (X.d j i).op) x✝⁴ x✝³) ( …
                            -/
  d_comp_d' _ _ _ _ _ := by rw [← op_comp, X.d_comp_d, op_zero]
                            /-
                              🎉 no goals
                            -/


/-- Sends a complex `X` with objects in `V` to the corresponding complex with objects in `Vᵒᵖ`. -/
@[simps]
protected def opSymm (X : HomologicalComplex V c.symm) : HomologicalComplex Vᵒᵖ c where
  X i := op (X.X i)
  d i j := (X.d j i).op
                      /-
                        ι : Type u_1
                        V : Type u_2
                        inst✝¹ : CategoryTheory.Category.{?u.21355, u_2} V
                        c : ComplexShape ι
                        inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
                        X : HomologicalComplex V c.symm
                        i j : ι
                        hij : Not (c.Rel i j)
                        ⊢ Eq ((fun i j => (X.d j i).op) i j) 0
                      -/
  shape i j hij := by simp only; rw [X.shape j i hij, op_zero]
                                 /-
                                   🎉 no goals
                                 -/
                            /-
                              ι : Type u_1
                              V : Type u_2
                              inst✝¹ : CategoryTheory.Category.{?u.21355, u_2} V
                              c : ComplexShape ι
                              inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
                              X : HomologicalComplex V c.symm
                              x✝⁴ x✝³ x✝² : ι
                              x✝¹ : c.Rel x✝⁴ x✝³
                              x✝ : c.Rel x✝³ x✝²
                              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i j => (X.d j i).op) x✝⁴ x✝³) ( …
                            -/
  d_comp_d' _ _ _ _ _ := by rw [← op_comp, X.d_comp_d, op_zero]
                            /-
                              🎉 no goals
                            -/


/-- Sends a complex `X` with objects in `Vᵒᵖ` to the corresponding complex with objects in `V`. -/
@[simps]
protected def unop (X : HomologicalComplex Vᵒᵖ c) : HomologicalComplex V c.symm where
  X i := unop (X.X i)
  d i j := (X.d j i).unop
                      /-
                        ι : Type u_1
                        V : Type u_2
                        inst✝¹ : CategoryTheory.Category.{?u.22996, u_2} V
                        c : ComplexShape ι
                        inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
                        X : HomologicalComplex (Opposite V) c
                        i j : ι
                        hij : Not (c.symm.Rel i j)
                        ⊢ Eq ((fun i j => (X.d j i).unop) i j) 0
                      -/
  shape i j hij := by simp only; rw [X.shape j i hij, unop_zero]
                                 /-
                                   🎉 no goals
                                 -/
                            /-
                              ι : Type u_1
                              V : Type u_2
                              inst✝¹ : CategoryTheory.Category.{?u.22996, u_2} V
                              c : ComplexShape ι
                              inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
                              X : HomologicalComplex (Opposite V) c
                              x✝⁴ x✝³ x✝² : ι
                              x✝¹ : c.symm.Rel x✝⁴ x✝³
                              x✝ : c.symm.Rel x✝³ x✝²
                              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i j => (X.d j i).unop) x✝⁴ x✝³) …
                            -/
  d_comp_d' _ _ _ _ _ := by rw [← unop_comp, X.d_comp_d, unop_zero]
                            /-
                              🎉 no goals
                            -/


/-- Sends a complex `X` with objects in `Vᵒᵖ` to the corresponding complex with objects in `V`. -/
@[simps]
protected def unopSymm (X : HomologicalComplex Vᵒᵖ c.symm) : HomologicalComplex V c where
  X i := unop (X.X i)
  d i j := (X.d j i).unop
                      /-
                        ι : Type u_1
                        V : Type u_2
                        inst✝¹ : CategoryTheory.Category.{?u.24645, u_2} V
                        c : ComplexShape ι
                        inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
                        X : HomologicalComplex (Opposite V) c.symm
                        i j : ι
                        hij : Not (c.Rel i j)
                        ⊢ Eq ((fun i j => (X.d j i).unop) i j) 0
                      -/
  shape i j hij := by simp only; rw [X.shape j i hij, unop_zero]
                                 /-
                                   🎉 no goals
                                 -/
                            /-
                              ι : Type u_1
                              V : Type u_2
                              inst✝¹ : CategoryTheory.Category.{?u.24645, u_2} V
                              c : ComplexShape ι
                              inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
                              X : HomologicalComplex (Opposite V) c.symm
                              x✝⁴ x✝³ x✝² : ι
                              x✝¹ : c.Rel x✝⁴ x✝³
                              x✝ : c.Rel x✝³ x✝²
                              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i j => (X.d j i).unop) x✝⁴ x✝³) …
                            -/
  d_comp_d' _ _ _ _ _ := by rw [← unop_comp, X.d_comp_d, unop_zero]
                            /-
                              🎉 no goals
                            -/


/-- Auxiliary definition for `opEquivalence`. -/
@[simps]
def opFunctor : (HomologicalComplex V c)ᵒᵖ ⥤ HomologicalComplex Vᵒᵖ c.symm where
  obj X := (unop X).op
  map f :=
    { f := fun i => (f.unop.f i).op
                               /-
                                 ι : Type u_1
                                 V : Type u_2
                                 inst✝¹ : CategoryTheory.Category.{?u.26323, u_2} V
                                 c : ComplexShape ι
                                 inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
                                 X✝ Y✝ : Opposite (HomologicalComplex V c)
                                 f : Quiver.Hom X✝ Y✝
                                 i j : ι
                                 x✝ : c.symm.Rel i j
                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i => (f.unop.f i).op) i) (((fun …
                               -/
      comm' := fun i j _ => by simp only [op_d, ← op_comp, f.unop.comm] }
                               /-
                                 🎉 no goals
                               -/


/-- Auxiliary definition for `opEquivalence`. -/
@[simps]
def opInverse : HomologicalComplex Vᵒᵖ c.symm ⥤ (HomologicalComplex V c)ᵒᵖ where
  obj X := op X.unopSymm
  map f := Quiver.Hom.op
    { f := fun i => (f.f i).unop
                               /-
                                 ι : Type u_1
                                 V : Type u_2
                                 inst✝¹ : CategoryTheory.Category.{?u.30179, u_2} V
                                 c : ComplexShape ι
                                 inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
                                 X✝ Y✝ : HomologicalComplex (Opposite V) c.symm
                                 f : Quiver.Hom X✝ Y✝
                                 i j : ι
                                 x✝ : c.Rel i j
                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i => (f.f i).unop) i) (X✝.unopS …
                               -/
      comm' := fun i j _ => by simp only [unopSymm_d, ← unop_comp, f.comm] }
                               /-
                                 🎉 no goals
                               -/


/-- Auxiliary definition for `opEquivalence`. -/
def opUnitIso : 𝟭 (HomologicalComplex V c)ᵒᵖ ≅ opFunctor V c ⋙ opInverse V c :=
  NatIso.ofComponents
    (fun X =>
      (HomologicalComplex.Hom.isoOfComponents (fun _ => Iso.refl _) fun i j _ => by
            simp only [Iso.refl_hom, Category.id_comp, unopSymm_d, op_d, Quiver.Hom.unop_op,
              Category.comp_id] :
          (Opposite.unop X).op.unopSymm ≅ unop X).op)
    (by
      /-
        ι : Type u_1
        V : Type u_2
        inst✝¹ : CategoryTheory.Category.{?u.34094, u_2} V
        c : ComplexShape ι
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
        ⊢ ∀ {X Y : Opposite (HomologicalComplex V c)} (f : Quiver.Hom X Y), Eq (Catego …
      -/
      intro X Y f
      /-
        ι : Type u_1
        V : Type u_2
        inst✝¹ : CategoryTheory.Category.{?u.34094, u_2} V
        c : ComplexShape ι
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
        X Y : Opposite (HomologicalComplex V c)
        f : Quiver.Hom X Y
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (Opposite …
      -/
      refine Quiver.Hom.unop_inj ?_
      /-
        ι : Type u_1
        V : Type u_2
        inst✝¹ : CategoryTheory.Category.{?u.34094, u_2} V
        c : ComplexShape ι
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
        X Y : Opposite (HomologicalComplex V c)
        f : Quiver.Hom X Y
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (Opposite …
      -/
      ext x
      simp only [Quiver.Hom.unop_op, Functor.id_map, Iso.op_hom, Functor.comp_map, unop_comp,
        comp_f, Hom.isoOfComponents_hom_f]
      /-
        case h
        ι : Type u_1
        V : Type u_2
        inst✝¹ : CategoryTheory.Category.{?u.34094, u_2} V
        c : ComplexShape ι
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
        X Y : Opposite (HomologicalComplex V c)
        f : Quiver.Hom X Y
        x : ι
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Iso.refl ((Opposite.u …
      -/
      erw [Category.id_comp, Category.comp_id (f.unop.f x)])
      /-
        🎉 no goals
      -/


/-- Auxiliary definition for `opEquivalence`. -/
def opCounitIso : opInverse V c ⋙ opFunctor V c ≅ 𝟭 (HomologicalComplex Vᵒᵖ c.symm) :=
  /-
    ι : Type u_1
    V : Type u_2
    inst✝¹ : CategoryTheory.Category.{?u.37864, u_2} V
    c : ComplexShape ι
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    ⊢ ∀ {X Y : HomologicalComplex (Opposite V) c.symm} (f : Quiver.Hom X Y), Eq (C …
  -/
             /-
               ι : Type u_1
               V : Type u_2
               inst✝¹ : CategoryTheory.Category.{?u.37864, u_2} V
               c : ComplexShape ι
               inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
               X : HomologicalComplex (Opposite V) c.symm
               ⊢ ∀ (i j : ι), c.symm.Rel i j → Eq (CategoryTheory.CategoryStruct.comp ((fun x …
             -/
  NatIso.ofComponents
             /-
               🎉 no goals
             -/
  /-
    🎉 no goals
  -/
    fun X => HomologicalComplex.Hom.isoOfComponents fun _ => Iso.refl _


/-- Given a category of complexes with objects in `V`, there is a natural equivalence between its
opposite category and a category of complexes with objects in `Vᵒᵖ`. -/
@[simps]
def opEquivalence : (HomologicalComplex V c)ᵒᵖ ≌ HomologicalComplex Vᵒᵖ c.symm where
  functor := opFunctor V c
  inverse := opInverse V c
  unitIso := opUnitIso V c
  counitIso := opCounitIso V c
  functor_unitIso_comp X := by
    /-
      ι : Type u_1
      V : Type u_2
      inst✝¹ : CategoryTheory.Category.{?u.46556, u_2} V
      c : ComplexShape ι
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
      X : Opposite (HomologicalComplex V c)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((HomologicalComplex.opFunctor V c).m …
    -/
    ext
    simp only [opUnitIso, opCounitIso, NatIso.ofComponents_hom_app, Iso.op_hom, comp_f,
      opFunctor_map_f, Quiver.Hom.unop_op, Hom.isoOfComponents_hom_f]
    /-
      case h
      ι : Type u_1
      V : Type u_2
      inst✝¹ : CategoryTheory.Category.{?u.46556, u_2} V
      c : ComplexShape ι
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
      X : Opposite (HomologicalComplex V c)
      i✝ : ι
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Iso.refl ((Opposite.u …
    -/
    exact Category.comp_id _
    /-
      🎉 no goals
    -/


/-- Auxiliary definition for `unopEquivalence`. -/
@[simps]
def unopFunctor : (HomologicalComplex Vᵒᵖ c)ᵒᵖ ⥤ HomologicalComplex V c.symm where
  obj X := (unop X).unop
  map f :=
    { f := fun i => (f.unop.f i).unop
                               /-
                                 ι : Type u_1
                                 V : Type u_2
                                 inst✝¹ : CategoryTheory.Category.{?u.48375, u_2} V
                                 c : ComplexShape ι
                                 inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
                                 X✝ Y✝ : Opposite (HomologicalComplex (Opposite V) c)
                                 f : Quiver.Hom X✝ Y✝
                                 i j : ι
                                 x✝ : c.symm.Rel i j
                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i => (f.unop.f i).unop) i) (((f …
                               -/
      comm' := fun i j _ => by simp only [unop_d, ← unop_comp, f.unop.comm] }
                               /-
                                 🎉 no goals
                               -/


/-- Auxiliary definition for `unopEquivalence`. -/
@[simps]
def unopInverse : HomologicalComplex V c.symm ⥤ (HomologicalComplex Vᵒᵖ c)ᵒᵖ where
  obj X := op X.opSymm
  map f := Quiver.Hom.op
    { f := fun i => (f.f i).op
                               /-
                                 ι : Type u_1
                                 V : Type u_2
                                 inst✝¹ : CategoryTheory.Category.{?u.53271, u_2} V
                                 c : ComplexShape ι
                                 inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
                                 X✝ Y✝ : HomologicalComplex V c.symm
                                 f : Quiver.Hom X✝ Y✝
                                 i j : ι
                                 x✝ : c.Rel i j
                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i => (f.f i).op) i) (X✝.opSymm. …
                               -/
      comm' := fun i j _ => by simp only [opSymm_d, ← op_comp, f.comm] }
                               /-
                                 🎉 no goals
                               -/


/-- Auxiliary definition for `unopEquivalence`. -/
def unopUnitIso : 𝟭 (HomologicalComplex Vᵒᵖ c)ᵒᵖ ≅ unopFunctor V c ⋙ unopInverse V c :=
  NatIso.ofComponents
    (fun X =>
      (HomologicalComplex.Hom.isoOfComponents (fun _ => Iso.refl _) fun i j _ => by
            simp only [Iso.refl_hom, Category.id_comp, unopSymm_d, op_d, Quiver.Hom.unop_op,
              Category.comp_id] :
          (Opposite.unop X).op.unopSymm ≅ unop X).op)
    (by
      /-
        ι : Type u_1
        V : Type u_2
        inst✝¹ : CategoryTheory.Category.{?u.56405, u_2} V
        c : ComplexShape ι
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
        ⊢ ∀ {X Y : Opposite (HomologicalComplex (Opposite V) c)} (f : Quiver.Hom X Y), …
      -/
      intro X Y f
      /-
        ι : Type u_1
        V : Type u_2
        inst✝¹ : CategoryTheory.Category.{?u.56405, u_2} V
        c : ComplexShape ι
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
        X Y : Opposite (HomologicalComplex (Opposite V) c)
        f : Quiver.Hom X Y
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (Opposite …
      -/
      refine Quiver.Hom.unop_inj ?_
      /-
        ι : Type u_1
        V : Type u_2
        inst✝¹ : CategoryTheory.Category.{?u.56405, u_2} V
        c : ComplexShape ι
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
        X Y : Opposite (HomologicalComplex (Opposite V) c)
        f : Quiver.Hom X Y
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (Opposite …
      -/
      ext x
      simp only [Quiver.Hom.unop_op, Functor.id_map, Iso.op_hom, Functor.comp_map, unop_comp,
        comp_f, Hom.isoOfComponents_hom_f]
      /-
        case h
        ι : Type u_1
        V : Type u_2
        inst✝¹ : CategoryTheory.Category.{?u.56405, u_2} V
        c : ComplexShape ι
        inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
        X Y : Opposite (HomologicalComplex (Opposite V) c)
        f : Quiver.Hom X Y
        x : ι
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Iso.refl ((Opposite.u …
      -/
      erw [Category.id_comp, Category.comp_id (f.unop.f x)])
      /-
        🎉 no goals
      -/


/-- Auxiliary definition for `unopEquivalence`. -/
def unopCounitIso : unopInverse V c ⋙ unopFunctor V c ≅ 𝟭 (HomologicalComplex V c.symm) :=
  /-
    ι : Type u_1
    V : Type u_2
    inst✝¹ : CategoryTheory.Category.{?u.60379, u_2} V
    c : ComplexShape ι
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
    ⊢ ∀ {X Y : HomologicalComplex V c.symm} (f : Quiver.Hom X Y), Eq (CategoryTheo …
  -/
             /-
               ι : Type u_1
               V : Type u_2
               inst✝¹ : CategoryTheory.Category.{?u.60379, u_2} V
               c : ComplexShape ι
               inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
               X : HomologicalComplex V c.symm
               ⊢ ∀ (i j : ι), c.symm.Rel i j → Eq (CategoryTheory.CategoryStruct.comp ((fun x …
             -/
  NatIso.ofComponents
             /-
               🎉 no goals
             -/
  /-
    🎉 no goals
  -/
    fun X => HomologicalComplex.Hom.isoOfComponents fun _ => Iso.refl _


/-- Given a category of complexes with objects in `Vᵒᵖ`, there is a natural equivalence between its
opposite category and a category of complexes with objects in `V`. -/
@[simps]
def unopEquivalence : (HomologicalComplex Vᵒᵖ c)ᵒᵖ ≌ HomologicalComplex V c.symm where
  functor := unopFunctor V c
  inverse := unopInverse V c
  unitIso := unopUnitIso V c
  counitIso := unopCounitIso V c
  functor_unitIso_comp X := by
    /-
      ι : Type u_1
      V : Type u_2
      inst✝¹ : CategoryTheory.Category.{?u.68935, u_2} V
      c : ComplexShape ι
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
      X : Opposite (HomologicalComplex (Opposite V) c)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((HomologicalComplex.unopFunctor V c) …
    -/
    ext
    simp only [opUnitIso, opCounitIso, NatIso.ofComponents_hom_app, Iso.op_hom, comp_f,
      opFunctor_map_f, Quiver.Hom.unop_op, Hom.isoOfComponents_hom_f]
    /-
      case h
      ι : Type u_1
      V : Type u_2
      inst✝¹ : CategoryTheory.Category.{?u.68935, u_2} V
      c : ComplexShape ι
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms V
      X : Opposite (HomologicalComplex (Opposite V) c)
      i✝ : ι
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((HomologicalComplex.unopFunctor V c …
    -/
    exact Category.comp_id _
    /-
      🎉 no goals
    -/


instance (K : HomologicalComplex V c) (i : ι) [K.HasHomology i] :
    K.op.HasHomology i :=
  (inferInstance : (K.sc i).op.HasHomology)


instance (K : HomologicalComplex Vᵒᵖ c) (i : ι) [K.HasHomology i] :
    K.unop.HasHomology i :=
  (inferInstance : (K.sc i).unop.HasHomology)


instance (K : HomologicalComplex V c) (i : ι) [K.HasHomology i] :
    ((opFunctor _ _).obj (op K)).HasHomology i := by
  /-
    ι : Type u_1
    V : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_2} V
    c : ComplexShape ι
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
    K : HomologicalComplex V c
    i : ι
    inst✝ : K.HasHomology i
    ⊢ ((HomologicalComplex.opFunctor V c).obj { unop := K }).HasHomology i
  -/
  dsimp
  /-
    ι : Type u_1
    V : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_2} V
    c : ComplexShape ι
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
    K : HomologicalComplex V c
    i : ι
    inst✝ : K.HasHomology i
    ⊢ K.op.HasHomology i
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance (K : HomologicalComplex Vᵒᵖ c) (i : ι) [K.HasHomology i] :
    ((unopFunctor _ _).obj (op K)).HasHomology i := by
  /-
    ι : Type u_1
    V : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_2} V
    c : ComplexShape ι
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
    K : HomologicalComplex (Opposite V) c
    i : ι
    inst✝ : K.HasHomology i
    ⊢ ((HomologicalComplex.unopFunctor V c).obj { unop := K }).HasHomology i
  -/
  dsimp
  /-
    ι : Type u_1
    V : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_2} V
    c : ComplexShape ι
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
    K : HomologicalComplex (Opposite V) c
    i : ι
    inst✝ : K.HasHomology i
    ⊢ K.unop.HasHomology i
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- If `K` is a homological complex, then the homology of `K.op` identifies to
the opposite of the homology of `K`. -/
def homologyOp (K : HomologicalComplex V c) (i : ι) [K.HasHomology i] :
    K.op.homology i ≅ op (K.homology i) :=
  (K.sc i).homologyOpIso


/-- If `K` is a homological complex in the opposite category,
then the homology of `K.unop` identifies to the opposite of the homology of `K`. -/
def homologyUnop (K : HomologicalComplex Vᵒᵖ c) (i : ι) [K.HasHomology i] :
    K.unop.homology i ≅ unop (K.homology i) :=
  (K.unop.homologyOp i).unop


/-- The canonical isomorphism `K.op.cycles i ≅ op (K.opcycles i)`. -/
def cyclesOpIso : K.op.cycles i ≅ op (K.opcycles i) :=
  (K.sc i).cyclesOpIso


/-- The canonical isomorphism `K.op.opcycles i ≅ op (K.cycles i)`. -/
def opcyclesOpIso : K.op.opcycles i ≅ op (K.cycles i) :=
  (K.sc i).opcyclesOpIso


@[reassoc (attr := simp)]
lemma opcyclesOpIso_hom_toCycles_op :
    (K.opcyclesOpIso i).hom ≫ (K.toCycles j i).op = K.op.fromOpcycles i j := by
  /-
    ι : Type u_1
    V : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_2} V
    c : ComplexShape ι
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
    K : HomologicalComplex V c
    i : ι
    inst✝ : K.HasHomology i
    j : ι
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.opcyclesOpIso i).hom (K.toCycles j …
  -/
  by_cases hij : c.Rel j i
    /-
      case pos
      ι : Type u_1
      V : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_2} V
      c : ComplexShape ι
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
      K : HomologicalComplex V c
      i : ι
      inst✝ : K.HasHomology i
      j : ι
      hij : c.Rel j i
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.opcyclesOpIso i).hom (K.toCycles j …
    -/
  · obtain rfl := c.prev_eq' hij
    /-
      case pos
      ι : Type u_1
      V : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_2} V
      c : ComplexShape ι
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
      K : HomologicalComplex V c
      i : ι
      inst✝ : K.HasHomology i
      hij : c.Rel (c.prev i) i
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.opcyclesOpIso i).hom (K.toCycles ( …
    -/
    exact (K.sc i).opcyclesOpIso_hom_toCycles_op
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u_1
      V : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_2} V
      c : ComplexShape ι
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
      K : HomologicalComplex V c
      i : ι
      inst✝ : K.HasHomology i
      j : ι
      hij : Not (c.Rel j i)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.opcyclesOpIso i).hom (K.toCycles j …
    -/
  · rw [K.toCycles_eq_zero hij, K.op.fromOpcycles_eq_zero hij, op_zero, comp_zero]
    /-
      🎉 no goals
    -/


@[reassoc (attr := simp)]
lemma fromOpcycles_op_cyclesOpIso_inv :
    (K.fromOpcycles i j).op ≫ (K.cyclesOpIso i).inv = K.op.toCycles j i := by
  /-
    ι : Type u_1
    V : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_2} V
    c : ComplexShape ι
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
    K : HomologicalComplex V c
    i : ι
    inst✝ : K.HasHomology i
    j : ι
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.fromOpcycles i j).op (K.cyclesOpIs …
  -/
  by_cases hij : c.Rel i j
    /-
      case pos
      ι : Type u_1
      V : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_2} V
      c : ComplexShape ι
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
      K : HomologicalComplex V c
      i : ι
      inst✝ : K.HasHomology i
      j : ι
      hij : c.Rel i j
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.fromOpcycles i j).op (K.cyclesOpIs …
    -/
  · obtain rfl := c.next_eq' hij
    /-
      case pos
      ι : Type u_1
      V : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_2} V
      c : ComplexShape ι
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
      K : HomologicalComplex V c
      i : ι
      inst✝ : K.HasHomology i
      hij : c.Rel i (c.next i)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.fromOpcycles i (c.next i)).op (K.c …
    -/
    exact (K.sc i).fromOpcycles_op_cyclesOpIso_inv
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u_1
      V : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_2} V
      c : ComplexShape ι
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
      K : HomologicalComplex V c
      i : ι
      inst✝ : K.HasHomology i
      j : ι
      hij : Not (c.Rel i j)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.fromOpcycles i j).op (K.cyclesOpIs …
    -/
  · rw [K.op.toCycles_eq_zero hij, K.fromOpcycles_eq_zero hij, op_zero, zero_comp]
    /-
      🎉 no goals
    -/


@[reassoc]
lemma homologyOp_hom_naturality :
    homologyMap ((opFunctor _ _).map φ.op) _ ≫ (K.homologyOp i).hom =
      (L.homologyOp i).hom ≫ (homologyMap φ i).op :=
  ShortComplex.homologyOpIso_hom_naturality ((shortComplexFunctor V c i).map φ)


@[reassoc]
lemma opcyclesOpIso_hom_naturality :
    opcyclesMap ((opFunctor _ _).map φ.op) _ ≫ (K.opcyclesOpIso i).hom =
      (L.opcyclesOpIso i).hom ≫ (cyclesMap φ i).op :=
  ShortComplex.opcyclesOpIso_hom_naturality ((shortComplexFunctor V c i).map φ)


@[reassoc]
lemma opcyclesOpIso_inv_naturality :
    (cyclesMap φ i).op ≫ (K.opcyclesOpIso i).inv =
      (L.opcyclesOpIso i).inv ≫ opcyclesMap ((opFunctor _ _).map φ.op) _ :=
  ShortComplex.opcyclesOpIso_inv_naturality ((shortComplexFunctor V c i).map φ)


@[reassoc]
lemma cyclesOpIso_hom_naturality :
    cyclesMap ((opFunctor _ _).map φ.op) _ ≫ (K.cyclesOpIso i).hom =
      (L.cyclesOpIso i).hom ≫ (opcyclesMap φ i).op :=
  ShortComplex.cyclesOpIso_hom_naturality ((shortComplexFunctor V c i).map φ)


@[reassoc]
lemma cyclesOpIso_inv_naturality :
    (opcyclesMap φ i).op ≫ (K.cyclesOpIso i).inv =
      (L.cyclesOpIso i).inv ≫ cyclesMap ((opFunctor _ _).map φ.op) _ :=
  ShortComplex.cyclesOpIso_inv_naturality ((shortComplexFunctor V c i).map φ)


/-- The natural isomorphism `K.op.cycles i ≅ op (K.opcycles i)`. -/
@[simps!]
def cyclesOpNatIso :
    opFunctor V c ⋙ cyclesFunctor Vᵒᵖ c.symm i ≅ (opcyclesFunctor V c i).op :=
  NatIso.ofComponents (fun K ↦ (unop K).cyclesOpIso i)
    (fun _ ↦ cyclesOpIso_hom_naturality _ _)


/-- The natural isomorphism `K.op.opcycles i ≅ op (K.cycles i)`. -/
def opcyclesOpNatIso :
    opFunctor V c ⋙ opcyclesFunctor Vᵒᵖ c.symm i ≅ (cyclesFunctor V c i).op :=
  NatIso.ofComponents (fun K ↦ (unop K).opcyclesOpIso i)
    (fun _ ↦ opcyclesOpIso_hom_naturality _ _)


/-- The natural isomorphism `K.op.homology i ≅ op (K.homology i)`. -/
def homologyOpNatIso :
    opFunctor V c ⋙ homologyFunctor Vᵒᵖ c.symm i ≅ (homologyFunctor V c i).op :=
  NatIso.ofComponents (fun K ↦ (unop K).homologyOp i)
    (fun _ ↦ homologyOp_hom_naturality _ _)


instance opFunctor_additive : (@opFunctor ι V _ c _).Additive where


instance unopFunctor_additive : (@unopFunctor ι V _ c _).Additive where


