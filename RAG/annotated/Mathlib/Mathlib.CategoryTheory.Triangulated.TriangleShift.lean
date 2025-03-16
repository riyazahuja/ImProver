/-- The shift functor `Triangle C ⥤ Triangle C` by `n : ℤ` sends a triangle
to the triangle obtained by shifting the objects by `n` in `C` and by
multiplying the three morphisms by `(-1)^n`. -/
@[simps]
noncomputable def Triangle.shiftFunctor (n : ℤ) : Triangle C ⥤ Triangle C where
  obj T := Triangle.mk (n.negOnePow • T.mor₁⟦n⟧') (n.negOnePow • T.mor₂⟦n⟧')
    (n.negOnePow • T.mor₃⟦n⟧' ≫ (shiftFunctorComm C 1 n).hom.app T.obj₁)
  map f :=
    { hom₁ := f.hom₁⟦n⟧'
      hom₂ := f.hom₂⟦n⟧'
      hom₃ := f.hom₃⟦n⟧'
      comm₁ := by
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Preadditive C
          inst✝¹ : CategoryTheory.HasShift C Int
          inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
          n : Int
          X✝ Y✝ : CategoryTheory.Pretriangulated.Triangle C
          f : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun T => CategoryTheory.Pretriangul …
        -/
        dsimp
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Preadditive C
          inst✝¹ : CategoryTheory.HasShift C Int
          inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
          n : Int
          X✝ Y✝ : CategoryTheory.Pretriangulated.Triangle C
          f : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSMul.hSMul n.negOnePow ((CategoryTh …
        -/
        simp only [Linear.units_smul_comp, Linear.comp_units_smul, ← Functor.map_comp, f.comm₁]
        /-
          🎉 no goals
        -/
      comm₂ := by
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Preadditive C
          inst✝¹ : CategoryTheory.HasShift C Int
          inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
          n : Int
          X✝ Y✝ : CategoryTheory.Pretriangulated.Triangle C
          f : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun T => CategoryTheory.Pretriangul …
        -/
        dsimp
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Preadditive C
          inst✝¹ : CategoryTheory.HasShift C Int
          inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
          n : Int
          X✝ Y✝ : CategoryTheory.Pretriangulated.Triangle C
          f : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSMul.hSMul n.negOnePow ((CategoryTh …
        -/
        simp only [Linear.units_smul_comp, Linear.comp_units_smul, ← Functor.map_comp, f.comm₂]
        /-
          🎉 no goals
        -/
      comm₃ := by
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Preadditive C
          inst✝¹ : CategoryTheory.HasShift C Int
          inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
          n : Int
          X✝ Y✝ : CategoryTheory.Pretriangulated.Triangle C
          f : Quiver.Hom X✝ Y✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun T => CategoryTheory.Pretriangul …
        -/
        dsimp
        rw [Linear.units_smul_comp, Linear.comp_units_smul, ← Functor.map_comp_assoc, ← f.comm₃,
          Functor.map_comp, assoc, assoc]
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Preadditive C
          inst✝¹ : CategoryTheory.HasShift C Int
          inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
          n : Int
          X✝ Y✝ : CategoryTheory.Pretriangulated.Triangle C
          f : Quiver.Hom X✝ Y✝
          ⊢ Eq (HSMul.hSMul n.negOnePow (CategoryTheory.CategoryStruct.comp ((CategoryTh …
        -/
        erw [(shiftFunctorComm C 1 n).hom.naturality]
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Preadditive C
          inst✝¹ : CategoryTheory.HasShift C Int
          inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
          n : Int
          X✝ Y✝ : CategoryTheory.Pretriangulated.Triangle C
          f : Quiver.Hom X✝ Y✝
          ⊢ Eq (HSMul.hSMul n.negOnePow (CategoryTheory.CategoryStruct.comp ((CategoryTh …
        -/
        rfl }
        /-
          🎉 no goals
        -/


/-- The canonical isomorphism `Triangle.shiftFunctor C 0 ≅ 𝟭 (Triangle C)`. -/
@[simps!]
noncomputable def Triangle.shiftFunctorZero : Triangle.shiftFunctor C 0 ≅ 𝟭 _ :=
  NatIso.ofComponents
    (fun T => Triangle.isoMk _ _ ((CategoryTheory.shiftFunctorZero C ℤ).app _)
      ((CategoryTheory.shiftFunctorZero C ℤ).app _) ((CategoryTheory.shiftFunctorZero C ℤ).app _)
          /-
            C : Type u
            inst✝³ : CategoryTheory.Category.{v, u} C
            inst✝² : CategoryTheory.Preadditive C
            inst✝¹ : CategoryTheory.HasShift C Int
            inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
            T : CategoryTheory.Pretriangulated.Triangle C
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Pretriangulated.Tria …
          -/
          /-
            🎉 no goals
          -/
      (by aesop_cat) (by aesop_cat) (by
                         /-
                           🎉 no goals
                         -/
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Preadditive C
          inst✝¹ : CategoryTheory.HasShift C Int
          inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
          T : CategoryTheory.Pretriangulated.Triangle C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Pretriangulated.Tria …
        -/
        dsimp
        simp only [one_smul, assoc, shiftFunctorComm_zero_hom_app,
          ← Functor.map_comp, Iso.inv_hom_id_app, Functor.id_obj, Functor.map_id,
          comp_id, NatTrans.naturality, Functor.id_map]))
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Preadditive C
          inst✝¹ : CategoryTheory.HasShift C Int
          inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
          ⊢ ∀ {X Y : CategoryTheory.Pretriangulated.Triangle C} (f : Quiver.Hom X Y), Eq …
        -/
    (by aesop_cat)
        /-
          🎉 no goals
        -/


/-- The canonical isomorphism
`Triangle.shiftFunctor C n ≅ Triangle.shiftFunctor C a ⋙ Triangle.shiftFunctor C b`
when `a + b = n`. -/
@[simps!]
noncomputable def Triangle.shiftFunctorAdd' (a b n : ℤ) (h : a + b = n) :
    Triangle.shiftFunctor C n ≅ Triangle.shiftFunctor C a ⋙ Triangle.shiftFunctor C b :=
  NatIso.ofComponents
    (fun T => Triangle.isoMk _ _
      ((CategoryTheory.shiftFunctorAdd' C a b n h).app _)
      ((CategoryTheory.shiftFunctorAdd' C a b n h).app _)
      ((CategoryTheory.shiftFunctorAdd' C a b n h).app _)
      (by
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Preadditive C
          inst✝¹ : CategoryTheory.HasShift C Int
          inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
          a b n : Int
          h : Eq (HAdd.hAdd a b) n
          T : CategoryTheory.Pretriangulated.Triangle C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Pretriangulated.Tria …
        -/
        subst h
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Preadditive C
          inst✝¹ : CategoryTheory.HasShift C Int
          inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
          a b : Int
          T : CategoryTheory.Pretriangulated.Triangle C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Pretriangulated.Tria …
        -/
        dsimp
        rw [Linear.units_smul_comp, NatTrans.naturality, Linear.comp_units_smul, Functor.comp_map,
          Functor.map_units_smul, Linear.comp_units_smul, smul_smul, Int.negOnePow_add, mul_comm])
      (by
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Preadditive C
          inst✝¹ : CategoryTheory.HasShift C Int
          inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
          a b n : Int
          h : Eq (HAdd.hAdd a b) n
          T : CategoryTheory.Pretriangulated.Triangle C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Pretriangulated.Tria …
        -/
        subst h
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Preadditive C
          inst✝¹ : CategoryTheory.HasShift C Int
          inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
          a b : Int
          T : CategoryTheory.Pretriangulated.Triangle C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Pretriangulated.Tria …
        -/
        dsimp
        rw [Linear.units_smul_comp, NatTrans.naturality, Linear.comp_units_smul, Functor.comp_map,
          Functor.map_units_smul, Linear.comp_units_smul, smul_smul, Int.negOnePow_add, mul_comm])
      (by
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Preadditive C
          inst✝¹ : CategoryTheory.HasShift C Int
          inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
          a b n : Int
          h : Eq (HAdd.hAdd a b) n
          T : CategoryTheory.Pretriangulated.Triangle C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Pretriangulated.Tria …
        -/
        subst h
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Preadditive C
          inst✝¹ : CategoryTheory.HasShift C Int
          inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
          a b : Int
          T : CategoryTheory.Pretriangulated.Triangle C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Pretriangulated.Tria …
        -/
        dsimp
        rw [Linear.units_smul_comp, Linear.comp_units_smul, Functor.map_units_smul,
          Linear.units_smul_comp, Linear.comp_units_smul, smul_smul, assoc,
          Functor.map_comp, assoc]
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Preadditive C
          inst✝¹ : CategoryTheory.HasShift C Int
          inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
          a b : Int
          T : CategoryTheory.Pretriangulated.Triangle C
          ⊢ Eq (HSMul.hSMul (HAdd.hAdd a b).negOnePow (CategoryTheory.CategoryStruct.com …
        -/
        erw [← NatTrans.naturality_assoc]
        simp only [shiftFunctorAdd'_eq_shiftFunctorAdd, Int.negOnePow_add,
          shiftFunctorComm_hom_app_comp_shift_shiftFunctorAdd_hom_app, add_comm a]))
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Preadditive C
          inst✝¹ : CategoryTheory.HasShift C Int
          inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
          a b n : Int
          h : Eq (HAdd.hAdd a b) n
          ⊢ ∀ {X Y : CategoryTheory.Pretriangulated.Triangle C} (f : Quiver.Hom X Y), Eq …
        -/
    (by aesop_cat)
        /-
          🎉 no goals
        -/


/-- Rotating triangles three times identifies with the shift by `1`. -/
noncomputable def rotateRotateRotateIso :
    rotate C ⋙ rotate C ⋙ rotate C ≅ Triangle.shiftFunctor C 1 :=
  NatIso.ofComponents
    (fun T => Triangle.isoMk _ _ (Iso.refl _) (Iso.refl _) (Iso.refl _)
          /-
            C : Type u
            inst✝³ : CategoryTheory.Category.{v, u} C
            inst✝² : CategoryTheory.Preadditive C
            inst✝¹ : CategoryTheory.HasShift C Int
            inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
            T : CategoryTheory.Pretriangulated.Triangle C
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Pretriangulated.rot …
          -/
          /-
            🎉 no goals
          -/
                         /-
                           🎉 no goals
                         -/
      (by aesop_cat) (by aesop_cat) (by aesop_cat))
                                        /-
                                          🎉 no goals
                                        -/
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Preadditive C
          inst✝¹ : CategoryTheory.HasShift C Int
          inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
          ⊢ ∀ {X Y : CategoryTheory.Pretriangulated.Triangle C} (f : Quiver.Hom X Y), Eq …
        -/
    (by aesop_cat)
        /-
          🎉 no goals
        -/


/-- Rotating triangles three times backwards identifies with the shift by `-1`. -/
noncomputable def invRotateInvRotateInvRotateIso :
    invRotate C ⋙ invRotate C ⋙ invRotate C ≅ Triangle.shiftFunctor C (-1) :=
  NatIso.ofComponents
    (fun T => Triangle.isoMk _ _ (Iso.refl _) (Iso.refl _) (Iso.refl _)
          /-
            C : Type u
            inst✝³ : CategoryTheory.Category.{v, u} C
            inst✝² : CategoryTheory.Preadditive C
            inst✝¹ : CategoryTheory.HasShift C Int
            inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
            T : CategoryTheory.Pretriangulated.Triangle C
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Pretriangulated.inv …
          -/
      (by aesop_cat)
          /-
            🎉 no goals
          -/
          /-
            C : Type u
            inst✝³ : CategoryTheory.Category.{v, u} C
            inst✝² : CategoryTheory.Preadditive C
            inst✝¹ : CategoryTheory.HasShift C Int
            inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
            T : CategoryTheory.Pretriangulated.Triangle C
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Pretriangulated.inv …
          -/
      (by aesop_cat)
          /-
            🎉 no goals
          -/
      (by
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Preadditive C
          inst✝¹ : CategoryTheory.HasShift C Int
          inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
          T : CategoryTheory.Pretriangulated.Triangle C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Pretriangulated.inv …
        -/
        dsimp [shiftFunctorCompIsoId]
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Preadditive C
          inst✝¹ : CategoryTheory.HasShift C Int
          inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
          T : CategoryTheory.Pretriangulated.Triangle C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        simp [shiftFunctorComm_eq C _ _ _ (add_neg_cancel (1 : ℤ))]))
        /-
          🎉 no goals
        -/
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Preadditive C
          inst✝¹ : CategoryTheory.HasShift C Int
          inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
          ⊢ ∀ {X Y : CategoryTheory.Pretriangulated.Triangle C} (f : Quiver.Hom X Y), Eq …
        -/
    (by aesop_cat)
        /-
          🎉 no goals
        -/


/-- The inverse of the rotation of triangles can be expressed using a double
rotation and the shift by `-1`. -/
noncomputable def invRotateIsoRotateRotateShiftFunctorNegOne :
    invRotate C ≅ rotate C ⋙ rotate C ⋙ Triangle.shiftFunctor C (-1) :=
  calc
    invRotate C ≅ invRotate C ⋙ 𝟭 _ := (Functor.rightUnitor _).symm
    _ ≅ invRotate C ⋙ Triangle.shiftFunctor C 0 :=
          isoWhiskerLeft _ (Triangle.shiftFunctorZero C).symm
    _ ≅ invRotate C ⋙ Triangle.shiftFunctor C 1 ⋙ Triangle.shiftFunctor C (-1) :=
          isoWhiskerLeft _ (Triangle.shiftFunctorAdd' C 1 (-1) 0 (add_neg_cancel 1))
    _ ≅ invRotate C ⋙ (rotate C ⋙ rotate C ⋙ rotate C) ⋙ Triangle.shiftFunctor C (-1) :=
          isoWhiskerLeft _ (isoWhiskerRight (rotateRotateRotateIso C).symm _)
    _ ≅ (invRotate C ⋙ rotate C) ⋙ rotate C ⋙ rotate C ⋙ Triangle.shiftFunctor C (-1) :=
          isoWhiskerLeft _ (Functor.associator _ _ _ ≪≫
            isoWhiskerLeft _ (Functor.associator _ _ _)) ≪≫ (Functor.associator _ _ _).symm
    _ ≅ 𝟭 _ ⋙ rotate C ⋙ rotate C ⋙ Triangle.shiftFunctor C (-1) :=
          isoWhiskerRight (triangleRotation C).counitIso _
    _ ≅ _ := Functor.leftUnitor _


noncomputable instance : HasShift (Triangle C) ℤ :=
  hasShiftMk (Triangle C) ℤ
    { F := Triangle.shiftFunctor C
      zero := Triangle.shiftFunctorZero C
      add := fun a b => Triangle.shiftFunctorAdd' C a b _ rfl
      assoc_hom_app := fun a b c T => by
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.Preadditive C
          inst✝¹ : CategoryTheory.HasShift C Int
          inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
          a b c : Int
          T : CategoryTheory.Pretriangulated.Triangle C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((fun a b => CategoryTheory.Pretrian …
        -/
        ext
        all_goals
          dsimp
          rw [← shiftFunctorAdd'_assoc_hom_app a b c _ _ _ rfl rfl (add_assoc a b c)]
          dsimp only [CategoryTheory.shiftFunctorAdd']
          simp }


@[simp]
lemma shiftFunctor_eq (n : ℤ) :
    CategoryTheory.shiftFunctor (Triangle C) n = Triangle.shiftFunctor C n := rfl


@[simp]
lemma shiftFunctorZero_eq :
    CategoryTheory.shiftFunctorZero (Triangle C) ℤ = Triangle.shiftFunctorZero C :=
  ShiftMkCore.shiftFunctorZero_eq _


@[simp]
lemma shiftFunctorAdd_eq (a b : ℤ) :
    CategoryTheory.shiftFunctorAdd (Triangle C) a b =
      Triangle.shiftFunctorAdd' C a b _ rfl :=
  ShiftMkCore.shiftFunctorAdd_eq _ _ _


@[simp]
lemma shiftFunctorAdd'_eq (a b c : ℤ) (h : a + b = c) :
    CategoryTheory.shiftFunctorAdd' (Triangle C) a b c h =
      Triangle.shiftFunctorAdd' C a b c h := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.HasShift C Int
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    a b c : Int
    h : Eq (HAdd.hAdd a b) c
    ⊢ Eq (CategoryTheory.shiftFunctorAdd' (CategoryTheory.Pretriangulated.Triangle …
  -/
  subst h
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.HasShift C Int
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    a b : Int
    ⊢ Eq (CategoryTheory.shiftFunctorAdd' (CategoryTheory.Pretriangulated.Triangle …
  -/
  rw [shiftFunctorAdd'_eq_shiftFunctorAdd]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.HasShift C Int
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    a b : Int
    ⊢ Eq (CategoryTheory.shiftFunctorAdd (CategoryTheory.Pretriangulated.Triangle  …
  -/
  apply shiftFunctorAdd_eq
  /-
    🎉 no goals
  -/


