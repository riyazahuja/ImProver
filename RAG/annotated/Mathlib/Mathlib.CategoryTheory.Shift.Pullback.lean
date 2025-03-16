/-- The category `PullbackShift C φ` is equipped with a shift such that for all `a`,
the shift functor by `a` is `shiftFunctor C (φ a)`. -/
@[nolint unusedArguments]
def PullbackShift (_ : A →+ B) [HasShift C B] := C


instance : Category (PullbackShift C φ) := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{?u.258, u_1} C
    A : Type u_2
    B : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : AddMonoid B
    φ : AddMonoidHom A B
    inst✝ : CategoryTheory.HasShift C B
    ⊢ CategoryTheory.Category.{?u.334, u_1} (CategoryTheory.PullbackShift C φ)
  -/
  dsimp only [PullbackShift]
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{?u.258, u_1} C
    A : Type u_2
    B : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : AddMonoid B
    φ : AddMonoidHom A B
    inst✝ : CategoryTheory.HasShift C B
    ⊢ CategoryTheory.Category.{?u.334, u_1} C
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The shift on `PullbackShift C φ` is obtained by precomposing the shift on `C` with
the monoidal functor `Discrete.addMonoidalFunctor φ : Discrete A ⥤ Discrete B`. -/
noncomputable instance : HasShift (PullbackShift C φ) A where
  shift := Discrete.addMonoidalFunctor φ ⋙ shiftMonoidalFunctor C B


instance [HasZeroObject C] : HasZeroObject (PullbackShift C φ) := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    A : Type u_2
    B : Type u_3
    inst✝³ : AddMonoid A
    inst✝² : AddMonoid B
    φ : AddMonoidHom A B
    inst✝¹ : CategoryTheory.HasShift C B
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    ⊢ CategoryTheory.Limits.HasZeroObject (CategoryTheory.PullbackShift C φ)
  -/
  dsimp [PullbackShift]
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    A : Type u_2
    B : Type u_3
    inst✝³ : AddMonoid A
    inst✝² : AddMonoid B
    φ : AddMonoidHom A B
    inst✝¹ : CategoryTheory.HasShift C B
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    ⊢ CategoryTheory.Limits.HasZeroObject C
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance [Preadditive C] : Preadditive (PullbackShift C φ) := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{?u.1785, u_1} C
    A : Type u_2
    B : Type u_3
    inst✝³ : AddMonoid A
    inst✝² : AddMonoid B
    φ : AddMonoidHom A B
    inst✝¹ : CategoryTheory.HasShift C B
    inst✝ : CategoryTheory.Preadditive C
    ⊢ CategoryTheory.Preadditive (CategoryTheory.PullbackShift C φ)
  -/
  dsimp [PullbackShift]
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{?u.1785, u_1} C
    A : Type u_2
    B : Type u_3
    inst✝³ : AddMonoid A
    inst✝² : AddMonoid B
    φ : AddMonoidHom A B
    inst✝¹ : CategoryTheory.HasShift C B
    inst✝ : CategoryTheory.Preadditive C
    ⊢ CategoryTheory.Preadditive C
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance [Preadditive C] (a : A) [(shiftFunctor C (φ a)).Additive] :
    (shiftFunctor (PullbackShift C φ) a).Additive := by
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
    A : Type u_2
    B : Type u_3
    inst✝⁴ : AddMonoid A
    inst✝³ : AddMonoid B
    φ : AddMonoidHom A B
    inst✝² : CategoryTheory.HasShift C B
    inst✝¹ : CategoryTheory.Preadditive C
    a : A
    inst✝ : (CategoryTheory.shiftFunctor C (φ a)).Additive
    ⊢ (CategoryTheory.shiftFunctor (CategoryTheory.PullbackShift C φ) a).Additive
  -/
  change (shiftFunctor C (φ a)).Additive
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
    A : Type u_2
    B : Type u_3
    inst✝⁴ : AddMonoid A
    inst✝³ : AddMonoid B
    φ : AddMonoidHom A B
    inst✝² : CategoryTheory.HasShift C B
    inst✝¹ : CategoryTheory.Preadditive C
    a : A
    inst✝ : (CategoryTheory.shiftFunctor C (φ a)).Additive
    ⊢ (CategoryTheory.shiftFunctor C (φ a)).Additive
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- When `b = φ a`, this is the canonical
isomorphism `shiftFunctor (PullbackShift C φ) a ≅ shiftFunctor C b`. -/
noncomputable def pullbackShiftIso (a : A) (b : B) (h : b = φ a) :
                                                                         /-
                                                                           C : Type u_1
                                                                           inst✝³ : CategoryTheory.Category.{?u.2752, u_1} C
                                                                           A : Type u_2
                                                                           B : Type u_3
                                                                           inst✝² : AddMonoid A
                                                                           inst✝¹ : AddMonoid B
                                                                           φ : AddMonoidHom A B
                                                                           inst✝ : CategoryTheory.HasShift C B
                                                                           a : A
                                                                           b : B
                                                                           h : Eq b (φ a)
                                                                           ⊢ Eq (CategoryTheory.shiftFunctor (CategoryTheory.PullbackShift C φ) a) (Categ …
                                                                         -/
    shiftFunctor (PullbackShift C φ) a ≅ shiftFunctor C b := eqToIso (by subst h; rfl)
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


lemma pullbackShiftFunctorZero_inv_app :
    (shiftFunctorZero _ A).inv.app X =
                                                                       /-
                                                                         C : Type u_1
                                                                         inst✝³ : CategoryTheory.Category.{?u.4377, u_1} C
                                                                         A : Type u_2
                                                                         B : Type u_3
                                                                         inst✝² : AddMonoid A
                                                                         inst✝¹ : AddMonoid B
                                                                         φ : AddMonoidHom A B
                                                                         inst✝ : CategoryTheory.HasShift C B
                                                                         X : CategoryTheory.PullbackShift C φ
                                                                         a₁ a₂ a₃ : A
                                                                         h : Eq (HAdd.hAdd a₁ a₂) a₃
                                                                         b₁ b₂ b₃ : B
                                                                         h₁ : Eq b₁ (φ a₁)
                                                                         h₂ : Eq b₂ (φ a₂)
                                                                         h₃ : Eq b₃ (φ a₃)
                                                                         ⊢ Eq 0 (φ 0)
                                                                       -/
      (shiftFunctorZero C B).inv.app X ≫ (pullbackShiftIso C φ 0 0 (by simp)).inv.app X := by
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    A : Type u_2
    B : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : AddMonoid B
    φ : AddMonoidHom A B
    inst✝ : CategoryTheory.HasShift C B
    X : CategoryTheory.PullbackShift C φ
    ⊢ Eq ((CategoryTheory.shiftFunctorZero (CategoryTheory.PullbackShift C φ) A).i …
  -/
  change (shiftFunctorZero C B).inv.app X ≫ _ = _
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    A : Type u_2
    B : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : AddMonoid B
    φ : AddMonoidHom A B
    inst✝ : CategoryTheory.HasShift C B
    X : CategoryTheory.PullbackShift C φ
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctorZero C B …
  -/
  dsimp [Discrete.eqToHom, Discrete.addMonoidalFunctor_ε]
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    A : Type u_2
    B : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : AddMonoid B
    φ : AddMonoidHom A B
    inst✝ : CategoryTheory.HasShift C B
    X : CategoryTheory.PullbackShift C φ
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctorZero C B …
  -/
  congr 2
  /-
    case e_a.e_self
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    A : Type u_2
    B : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : AddMonoid B
    φ : AddMonoidHom A B
    inst✝ : CategoryTheory.HasShift C B
    X : CategoryTheory.PullbackShift C φ
    ⊢ Eq ((CategoryTheory.shiftMonoidalFunctor C B).map (CategoryTheory.eqToHom ⋯) …
  -/
  apply eqToHom_map
  /-
    🎉 no goals
  -/


lemma pullbackShiftFunctorZero_hom_app :
    (shiftFunctorZero _ A).hom.app X =
                                    /-
                                      C : Type u_1
                                      inst✝³ : CategoryTheory.Category.{?u.7677, u_1} C
                                      A : Type u_2
                                      B : Type u_3
                                      inst✝² : AddMonoid A
                                      inst✝¹ : AddMonoid B
                                      φ : AddMonoidHom A B
                                      inst✝ : CategoryTheory.HasShift C B
                                      X : CategoryTheory.PullbackShift C φ
                                      a₁ a₂ a₃ : A
                                      h : Eq (HAdd.hAdd a₁ a₂) a₃
                                      b₁ b₂ b₃ : B
                                      h₁ : Eq b₁ (φ a₁)
                                      h₂ : Eq b₂ (φ a₂)
                                      h₃ : Eq b₃ (φ a₃)
                                      ⊢ Eq 0 (φ 0)
                                    -/
      (pullbackShiftIso C φ 0 0 (by simp)).hom.app X ≫ (shiftFunctorZero C B).hom.app X := by
                                    /-
                                      🎉 no goals
                                    -/
  rw [← cancel_epi ((shiftFunctorZero _ A).inv.app X), Iso.inv_hom_id_app,
    pullbackShiftFunctorZero_inv_app, assoc, Iso.inv_hom_id_app_assoc, Iso.inv_hom_id_app]
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    A : Type u_2
    B : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : AddMonoid B
    φ : AddMonoidHom A B
    inst✝ : CategoryTheory.HasShift C B
    X : CategoryTheory.PullbackShift C φ
    ⊢ Eq (CategoryTheory.CategoryStruct.id ((CategoryTheory.Functor.id (CategoryTh …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma pullbackShiftFunctorZero'_inv_app :
                                                                      /-
                                                                        C : Type u_1
                                                                        inst✝³ : CategoryTheory.Category.{?u.13601, u_1} C
                                                                        A : Type u_2
                                                                        B : Type u_3
                                                                        inst✝² : AddMonoid A
                                                                        inst✝¹ : AddMonoid B
                                                                        φ : AddMonoidHom A B
                                                                        inst✝ : CategoryTheory.HasShift C B
                                                                        X : CategoryTheory.PullbackShift C φ
                                                                        a₁ a₂ a₃ : A
                                                                        h : Eq (HAdd.hAdd a₁ a₂) a₃
                                                                        b₁ b₂ b₃ : B
                                                                        h₁ : Eq b₁ (φ a₁)
                                                                        h₂ : Eq b₂ (φ a₂)
                                                                        h₃ : Eq b₃ (φ a₃)
                                                                        ⊢ Eq (φ 0) 0
                                                                      -/
    (shiftFunctorZero _ A).inv.app X = (shiftFunctorZero' C (φ 0) (by rw [map_zero])).inv.app X ≫
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
      (pullbackShiftIso C φ 0 (φ 0) rfl).inv.app X := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    A : Type u_2
    B : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : AddMonoid B
    φ : AddMonoidHom A B
    inst✝ : CategoryTheory.HasShift C B
    X : CategoryTheory.PullbackShift C φ
    ⊢ Eq ((CategoryTheory.shiftFunctorZero (CategoryTheory.PullbackShift C φ) A).i …
  -/
  rw [pullbackShiftFunctorZero_inv_app]
  simp only [Functor.id_obj, pullbackShiftIso, eqToIso.inv, eqToHom_app, shiftFunctorZero',
    Iso.trans_inv, NatTrans.comp_app, eqToIso_refl, Iso.refl_inv, NatTrans.id_app, assoc]
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    A : Type u_2
    B : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : AddMonoid B
    φ : AddMonoidHom A B
    inst✝ : CategoryTheory.HasShift C B
    X : CategoryTheory.PullbackShift C φ
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctorZero C B …
  -/
  erw [comp_id]
  /-
    🎉 no goals
  -/


lemma pullbackShiftFunctorZero'_hom_app :
    (shiftFunctorZero _ A).hom.app X = (pullbackShiftIso C φ 0 (φ 0) rfl).hom.app X ≫
                                     /-
                                       C : Type u_1
                                       inst✝³ : CategoryTheory.Category.{?u.18675, u_1} C
                                       A : Type u_2
                                       B : Type u_3
                                       inst✝² : AddMonoid A
                                       inst✝¹ : AddMonoid B
                                       φ : AddMonoidHom A B
                                       inst✝ : CategoryTheory.HasShift C B
                                       X : CategoryTheory.PullbackShift C φ
                                       a₁ a₂ a₃ : A
                                       h : Eq (HAdd.hAdd a₁ a₂) a₃
                                       b₁ b₂ b₃ : B
                                       h₁ : Eq b₁ (φ a₁)
                                       h₂ : Eq b₂ (φ a₂)
                                       h₃ : Eq b₃ (φ a₃)
                                       ⊢ Eq (φ 0) 0
                                     -/
      (shiftFunctorZero' C (φ 0) (by rw [map_zero])).hom.app X := by
                                     /-
                                       🎉 no goals
                                     -/
  rw [← cancel_epi ((shiftFunctorZero _ A).inv.app X), Iso.inv_hom_id_app,
    pullbackShiftFunctorZero'_inv_app, assoc, Iso.inv_hom_id_app_assoc, Iso.inv_hom_id_app]
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    A : Type u_2
    B : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : AddMonoid B
    φ : AddMonoidHom A B
    inst✝ : CategoryTheory.HasShift C B
    X : CategoryTheory.PullbackShift C φ
    ⊢ Eq (CategoryTheory.CategoryStruct.id ((CategoryTheory.Functor.id (CategoryTh …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma pullbackShiftFunctorAdd'_inv_app :
    (shiftFunctorAdd' _ a₁ a₂ a₃ h).inv.app X =
      (shiftFunctor (PullbackShift C φ) a₂).map ((pullbackShiftIso C φ a₁ b₁ h₁).hom.app X) ≫
        (pullbackShiftIso C φ a₂ b₂ h₂).hom.app _ ≫
                                         /-
                                           C : Type u_1
                                           inst✝³ : CategoryTheory.Category.{?u.24993, u_1} C
                                           A : Type u_2
                                           B : Type u_3
                                           inst✝² : AddMonoid A
                                           inst✝¹ : AddMonoid B
                                           φ : AddMonoidHom A B
                                           inst✝ : CategoryTheory.HasShift C B
                                           X : CategoryTheory.PullbackShift C φ
                                           a₁ a₂ a₃ : A
                                           h : Eq (HAdd.hAdd a₁ a₂) a₃
                                           b₁ b₂ b₃ : B
                                           h₁ : Eq b₁ (φ a₁)
                                           h₂ : Eq b₂ (φ a₂)
                                           h₃ : Eq b₃ (φ a₃)
                                           ⊢ Eq (HAdd.hAdd b₁ b₂) b₃
                                         -/
        (shiftFunctorAdd' C b₁ b₂ b₃ (by rw [h₁, h₂, h₃, ← h, φ.map_add])).inv.app X ≫
                                         /-
                                           🎉 no goals
                                         -/
        (pullbackShiftIso C φ a₃ b₃ h₃).inv.app X := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    A : Type u_2
    B : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : AddMonoid B
    φ : AddMonoidHom A B
    inst✝ : CategoryTheory.HasShift C B
    X : CategoryTheory.PullbackShift C φ
    a₁ a₂ a₃ : A
    h : Eq (HAdd.hAdd a₁ a₂) a₃
    b₁ b₂ b₃ : B
    h₁ : Eq b₁ (φ a₁)
    h₂ : Eq b₂ (φ a₂)
    h₃ : Eq b₃ (φ a₃)
    ⊢ Eq ((CategoryTheory.shiftFunctorAdd' (CategoryTheory.PullbackShift C φ) a₁ a …
  -/
  subst h₁ h₂ h
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    A : Type u_2
    B : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : AddMonoid B
    φ : AddMonoidHom A B
    inst✝ : CategoryTheory.HasShift C B
    X : CategoryTheory.PullbackShift C φ
    a₁ a₂ : A
    b₃ : B
    h₃ : Eq b₃ (φ (HAdd.hAdd a₁ a₂))
    ⊢ Eq ((CategoryTheory.shiftFunctorAdd' (CategoryTheory.PullbackShift C φ) a₁ a …
  -/
  obtain rfl : b₃ = φ a₁ + φ a₂ := by rw [h₃, φ.map_add]
  erw [Functor.map_id, id_comp, id_comp, shiftFunctorAdd'_eq_shiftFunctorAdd,
    shiftFunctorAdd'_eq_shiftFunctorAdd]
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    A : Type u_2
    B : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : AddMonoid B
    φ : AddMonoidHom A B
    inst✝ : CategoryTheory.HasShift C B
    X : CategoryTheory.PullbackShift C φ
    a₁ a₂ : A
    h₃ : Eq (HAdd.hAdd (φ a₁) (φ a₂)) (φ (HAdd.hAdd a₁ a₂))
    ⊢ Eq ((CategoryTheory.shiftFunctorAdd (CategoryTheory.PullbackShift C φ) a₁ a₂ …
  -/
  change _ ≫ _ = _
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    A : Type u_2
    B : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : AddMonoid B
    φ : AddMonoidHom A B
    inst✝ : CategoryTheory.HasShift C B
    X : CategoryTheory.PullbackShift C φ
    a₁ a₂ : A
    h₃ : Eq (HAdd.hAdd (φ a₁) (φ a₂)) (φ (HAdd.hAdd a₁ a₂))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.LaxMonoidal. …
  -/
  congr 1
  /-
    case e_a
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    A : Type u_2
    B : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : AddMonoid B
    φ : AddMonoidHom A B
    inst✝ : CategoryTheory.HasShift C B
    X : CategoryTheory.PullbackShift C φ
    a₁ a₂ : A
    h₃ : Eq (HAdd.hAdd (φ a₁) (φ a₂)) (φ (HAdd.hAdd a₁ a₂))
    ⊢ Eq (((CategoryTheory.shiftMonoidalFunctor C B).map (CategoryTheory.Functor.L …
  -/
  rw [Discrete.addMonoidalFunctor_μ]
  /-
    case e_a
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    A : Type u_2
    B : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : AddMonoid B
    φ : AddMonoidHom A B
    inst✝ : CategoryTheory.HasShift C B
    X : CategoryTheory.PullbackShift C φ
    a₁ a₂ : A
    h₃ : Eq (HAdd.hAdd (φ a₁) (φ a₂)) (φ (HAdd.hAdd a₁ a₂))
    ⊢ Eq (((CategoryTheory.shiftMonoidalFunctor C B).map (CategoryTheory.Discrete. …
  -/
  dsimp [Discrete.eqToHom]
  /-
    case e_a
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    A : Type u_2
    B : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : AddMonoid B
    φ : AddMonoidHom A B
    inst✝ : CategoryTheory.HasShift C B
    X : CategoryTheory.PullbackShift C φ
    a₁ a₂ : A
    h₃ : Eq (HAdd.hAdd (φ a₁) (φ a₂)) (φ (HAdd.hAdd a₁ a₂))
    ⊢ Eq (((CategoryTheory.shiftMonoidalFunctor C B).map (CategoryTheory.eqToHom ⋯ …
  -/
  congr 2
  /-
    case e_a.e_self
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    A : Type u_2
    B : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : AddMonoid B
    φ : AddMonoidHom A B
    inst✝ : CategoryTheory.HasShift C B
    X : CategoryTheory.PullbackShift C φ
    a₁ a₂ : A
    h₃ : Eq (HAdd.hAdd (φ a₁) (φ a₂)) (φ (HAdd.hAdd a₁ a₂))
    ⊢ Eq ((CategoryTheory.shiftMonoidalFunctor C B).map (CategoryTheory.eqToHom ⋯) …
  -/
  apply eqToHom_map
  /-
    🎉 no goals
  -/


lemma pullbackShiftFunctorAdd'_hom_app :
    (shiftFunctorAdd' _ a₁ a₂ a₃ h).hom.app X =
      (pullbackShiftIso C φ a₃ b₃ h₃).hom.app X ≫
                                       /-
                                         C : Type u_1
                                         inst✝³ : CategoryTheory.Category.{?u.30467, u_1} C
                                         A : Type u_2
                                         B : Type u_3
                                         inst✝² : AddMonoid A
                                         inst✝¹ : AddMonoid B
                                         φ : AddMonoidHom A B
                                         inst✝ : CategoryTheory.HasShift C B
                                         X : CategoryTheory.PullbackShift C φ
                                         a₁ a₂ a₃ : A
                                         h : Eq (HAdd.hAdd a₁ a₂) a₃
                                         b₁ b₂ b₃ : B
                                         h₁ : Eq b₁ (φ a₁)
                                         h₂ : Eq b₂ (φ a₂)
                                         h₃ : Eq b₃ (φ a₃)
                                         ⊢ Eq (HAdd.hAdd b₁ b₂) b₃
                                       -/
      (shiftFunctorAdd' C b₁ b₂ b₃ (by rw [h₁, h₂, h₃, ← h, φ.map_add])).hom.app X ≫
                                       /-
                                         🎉 no goals
                                       -/
      (pullbackShiftIso C φ a₂ b₂ h₂).inv.app _ ≫
      (shiftFunctor (PullbackShift C φ) a₂).map ((pullbackShiftIso C φ a₁ b₁ h₁).inv.app X) := by
  rw [← cancel_epi ((shiftFunctorAdd' _ a₁ a₂ a₃ h).inv.app X), Iso.inv_hom_id_app,
    pullbackShiftFunctorAdd'_inv_app φ X a₁ a₂ a₃ h b₁ b₂ b₃ h₁ h₂ h₃, assoc, assoc, assoc,
    Iso.inv_hom_id_app_assoc, Iso.inv_hom_id_app_assoc, Iso.hom_inv_id_app_assoc,
    ← Functor.map_comp, Iso.hom_inv_id_app, Functor.map_id]
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    A : Type u_2
    B : Type u_3
    inst✝² : AddMonoid A
    inst✝¹ : AddMonoid B
    φ : AddMonoidHom A B
    inst✝ : CategoryTheory.HasShift C B
    X : CategoryTheory.PullbackShift C φ
    a₁ a₂ a₃ : A
    h : Eq (HAdd.hAdd a₁ a₂) a₃
    b₁ b₂ b₃ : B
    h₁ : Eq b₁ (φ a₁)
    h₂ : Eq b₂ (φ a₂)
    h₃ : Eq b₃ (φ a₃)
    ⊢ Eq (CategoryTheory.CategoryStruct.id (((CategoryTheory.shiftFunctor (Categor …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- If `F : C ⥤ D` commutes with the shifts on `C` and `D`, then it also commutes with
their pullbacks by an additive map.
-/
noncomputable def commShiftPullback :
    F.CommShift A (C := PullbackShift C φ) (D := PullbackShift D φ) where
  iso a := isoWhiskerRight (pullbackShiftIso C φ a (φ a) rfl) F ≪≫
    F.commShiftIso (φ a) ≪≫ isoWhiskerLeft _  (pullbackShiftIso D φ a (φ a) rfl).symm
  zero := by
    /-
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{?u.41452, u_1} C
      A : Type u_2
      B : Type u_3
      inst✝⁵ : AddMonoid A
      inst✝⁴ : AddMonoid B
      φ : AddMonoidHom A B
      inst✝³ : CategoryTheory.HasShift C B
      X : CategoryTheory.PullbackShift C φ
      a₁ a₂ a₃ : A
      h : Eq (HAdd.hAdd a₁ a₂) a₃
      b₁ b₂ b₃ : B
      h₁ : Eq b₁ (φ a₁)
      h₂ : Eq b₂ (φ a₂)
      h₃ : Eq b₃ (φ a₃)
      D : Type u_4
      inst✝² : CategoryTheory.Category.{?u.42230, u_4} D
      inst✝¹ : CategoryTheory.HasShift D B
      F : CategoryTheory.Functor C D
      inst✝ : F.CommShift B
      ⊢ Eq ((fun a => (CategoryTheory.isoWhiskerRight (CategoryTheory.pullbackShiftI …
    -/
    ext
    /-
      case w.w.h
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{?u.41452, u_1} C
      A : Type u_2
      B : Type u_3
      inst✝⁵ : AddMonoid A
      inst✝⁴ : AddMonoid B
      φ : AddMonoidHom A B
      inst✝³ : CategoryTheory.HasShift C B
      X : CategoryTheory.PullbackShift C φ
      a₁ a₂ a₃ : A
      h : Eq (HAdd.hAdd a₁ a₂) a₃
      b₁ b₂ b₃ : B
      h₁ : Eq b₁ (φ a₁)
      h₂ : Eq b₂ (φ a₂)
      h₃ : Eq b₃ (φ a₃)
      D : Type u_4
      inst✝² : CategoryTheory.Category.{?u.42230, u_4} D
      inst✝¹ : CategoryTheory.HasShift D B
      F : CategoryTheory.Functor C D
      inst✝ : F.CommShift B
      x✝ : CategoryTheory.PullbackShift C φ
      ⊢ Eq (((fun a => (CategoryTheory.isoWhiskerRight (CategoryTheory.pullbackShift …
    -/
    dsimp
    simp only [F.commShiftIso_zero' (A := B) (φ 0) (by rw [map_zero]), CommShift.isoZero'_hom_app,
      assoc, CommShift.isoZero_hom_app, pullbackShiftFunctorZero'_hom_app, map_comp,
      pullbackShiftFunctorZero'_inv_app]
    /-
      case w.w.h
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{?u.41452, u_1} C
      A : Type u_2
      B : Type u_3
      inst✝⁵ : AddMonoid A
      inst✝⁴ : AddMonoid B
      φ : AddMonoidHom A B
      inst✝³ : CategoryTheory.HasShift C B
      X : CategoryTheory.PullbackShift C φ
      a₁ a₂ a₃ : A
      h : Eq (HAdd.hAdd a₁ a₂) a₃
      b₁ b₂ b₃ : B
      h₁ : Eq b₁ (φ a₁)
      h₂ : Eq b₂ (φ a₂)
      h₃ : Eq b₃ (φ a₃)
      D : Type u_4
      inst✝² : CategoryTheory.Category.{?u.42230, u_4} D
      inst✝¹ : CategoryTheory.HasShift D B
      F : CategoryTheory.Functor C D
      inst✝ : F.CommShift B
      x✝ : CategoryTheory.PullbackShift C φ
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map ((CategoryTheory.pullbackShift …
    -/
    dsimp
    /-
      case w.w.h
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{?u.41452, u_1} C
      A : Type u_2
      B : Type u_3
      inst✝⁵ : AddMonoid A
      inst✝⁴ : AddMonoid B
      φ : AddMonoidHom A B
      inst✝³ : CategoryTheory.HasShift C B
      X : CategoryTheory.PullbackShift C φ
      a₁ a₂ a₃ : A
      h : Eq (HAdd.hAdd a₁ a₂) a₃
      b₁ b₂ b₃ : B
      h₁ : Eq b₁ (φ a₁)
      h₂ : Eq b₂ (φ a₂)
      h₃ : Eq b₃ (φ a₃)
      D : Type u_4
      inst✝² : CategoryTheory.Category.{?u.42230, u_4} D
      inst✝¹ : CategoryTheory.HasShift D B
      F : CategoryTheory.Functor C D
      inst✝ : F.CommShift B
      x✝ : CategoryTheory.PullbackShift C φ
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map ((CategoryTheory.pullbackShift …
    -/
    rfl
    /-
      🎉 no goals
    -/
  add a b := by
    /-
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{?u.41452, u_1} C
      A : Type u_2
      B : Type u_3
      inst✝⁵ : AddMonoid A
      inst✝⁴ : AddMonoid B
      φ : AddMonoidHom A B
      inst✝³ : CategoryTheory.HasShift C B
      X : CategoryTheory.PullbackShift C φ
      a₁ a₂ a₃ : A
      h : Eq (HAdd.hAdd a₁ a₂) a₃
      b₁ b₂ b₃ : B
      h₁ : Eq b₁ (φ a₁)
      h₂ : Eq b₂ (φ a₂)
      h₃ : Eq b₃ (φ a₃)
      D : Type u_4
      inst✝² : CategoryTheory.Category.{?u.42230, u_4} D
      inst✝¹ : CategoryTheory.HasShift D B
      F : CategoryTheory.Functor C D
      inst✝ : F.CommShift B
      a b : A
      ⊢ Eq ((fun a => (CategoryTheory.isoWhiskerRight (CategoryTheory.pullbackShiftI …
    -/
    ext
    /-
      case w.w.h
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{?u.41452, u_1} C
      A : Type u_2
      B : Type u_3
      inst✝⁵ : AddMonoid A
      inst✝⁴ : AddMonoid B
      φ : AddMonoidHom A B
      inst✝³ : CategoryTheory.HasShift C B
      X : CategoryTheory.PullbackShift C φ
      a₁ a₂ a₃ : A
      h : Eq (HAdd.hAdd a₁ a₂) a₃
      b₁ b₂ b₃ : B
      h₁ : Eq b₁ (φ a₁)
      h₂ : Eq b₂ (φ a₂)
      h₃ : Eq b₃ (φ a₃)
      D : Type u_4
      inst✝² : CategoryTheory.Category.{?u.42230, u_4} D
      inst✝¹ : CategoryTheory.HasShift D B
      F : CategoryTheory.Functor C D
      inst✝ : F.CommShift B
      a b : A
      x✝ : CategoryTheory.PullbackShift C φ
      ⊢ Eq (((fun a => (CategoryTheory.isoWhiskerRight (CategoryTheory.pullbackShift …
    -/
    dsimp
    /-
      case w.w.h
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{?u.41452, u_1} C
      A : Type u_2
      B : Type u_3
      inst✝⁵ : AddMonoid A
      inst✝⁴ : AddMonoid B
      φ : AddMonoidHom A B
      inst✝³ : CategoryTheory.HasShift C B
      X : CategoryTheory.PullbackShift C φ
      a₁ a₂ a₃ : A
      h : Eq (HAdd.hAdd a₁ a₂) a₃
      b₁ b₂ b₃ : B
      h₁ : Eq b₁ (φ a₁)
      h₂ : Eq b₂ (φ a₂)
      h₃ : Eq b₃ (φ a₃)
      D : Type u_4
      inst✝² : CategoryTheory.Category.{?u.42230, u_4} D
      inst✝¹ : CategoryTheory.HasShift D B
      F : CategoryTheory.Functor C D
      inst✝ : F.CommShift B
      a b : A
      x✝ : CategoryTheory.PullbackShift C φ
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map ((CategoryTheory.pullbackShift …
    -/
    simp only [CommShift.isoAdd_hom_app, map_comp, assoc]
    /-
      case w.w.h
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{?u.41452, u_1} C
      A : Type u_2
      B : Type u_3
      inst✝⁵ : AddMonoid A
      inst✝⁴ : AddMonoid B
      φ : AddMonoidHom A B
      inst✝³ : CategoryTheory.HasShift C B
      X : CategoryTheory.PullbackShift C φ
      a₁ a₂ a₃ : A
      h : Eq (HAdd.hAdd a₁ a₂) a₃
      b₁ b₂ b₃ : B
      h₁ : Eq b₁ (φ a₁)
      h₂ : Eq b₂ (φ a₂)
      h₃ : Eq b₃ (φ a₃)
      D : Type u_4
      inst✝² : CategoryTheory.Category.{?u.42230, u_4} D
      inst✝¹ : CategoryTheory.HasShift D B
      F : CategoryTheory.Functor C D
      inst✝ : F.CommShift B
      a b : A
      x✝ : CategoryTheory.PullbackShift C φ
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map ((CategoryTheory.pullbackShift …
    -/
    dsimp
    rw [F.commShiftIso_add' (a := φ a) (b := φ b) (by rw [φ.map_add]),
      ← shiftFunctorAdd'_eq_shiftFunctorAdd, ← shiftFunctorAdd'_eq_shiftFunctorAdd,
      pullbackShiftFunctorAdd'_hom_app φ _ a b (a + b) rfl (φ a) (φ b) (φ (a + b)) rfl rfl rfl,
      pullbackShiftFunctorAdd'_inv_app φ _ a b (a + b) rfl (φ a) (φ b) (φ (a + b)) rfl rfl rfl]
    /-
      case w.w.h
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{?u.41452, u_1} C
      A : Type u_2
      B : Type u_3
      inst✝⁵ : AddMonoid A
      inst✝⁴ : AddMonoid B
      φ : AddMonoidHom A B
      inst✝³ : CategoryTheory.HasShift C B
      X : CategoryTheory.PullbackShift C φ
      a₁ a₂ a₃ : A
      h : Eq (HAdd.hAdd a₁ a₂) a₃
      b₁ b₂ b₃ : B
      h₁ : Eq b₁ (φ a₁)
      h₂ : Eq b₂ (φ a₂)
      h₃ : Eq b₃ (φ a₃)
      D : Type u_4
      inst✝² : CategoryTheory.Category.{?u.42230, u_4} D
      inst✝¹ : CategoryTheory.HasShift D B
      F : CategoryTheory.Functor C D
      inst✝ : F.CommShift B
      a b : A
      x✝ : CategoryTheory.PullbackShift C φ
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map ((CategoryTheory.pullbackShift …
    -/
    dsimp
    simp only [CommShift.isoAdd'_hom_app, assoc, map_comp, NatTrans.naturality_assoc,
      Iso.inv_hom_id_app_assoc]
    /-
      case w.w.h
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{?u.41452, u_1} C
      A : Type u_2
      B : Type u_3
      inst✝⁵ : AddMonoid A
      inst✝⁴ : AddMonoid B
      φ : AddMonoidHom A B
      inst✝³ : CategoryTheory.HasShift C B
      X : CategoryTheory.PullbackShift C φ
      a₁ a₂ a₃ : A
      h : Eq (HAdd.hAdd a₁ a₂) a₃
      b₁ b₂ b₃ : B
      h₁ : Eq b₁ (φ a₁)
      h₂ : Eq b₂ (φ a₂)
      h₃ : Eq b₃ (φ a₃)
      D : Type u_4
      inst✝² : CategoryTheory.Category.{?u.42230, u_4} D
      inst✝¹ : CategoryTheory.HasShift D B
      F : CategoryTheory.Functor C D
      inst✝ : F.CommShift B
      a b : A
      x✝ : CategoryTheory.PullbackShift C φ
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map ((CategoryTheory.pullbackShift …
    -/
    slice_rhs 9 10 => rw [← map_comp, Iso.inv_hom_id_app, map_id]
    /-
      case w.w.h
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{?u.41452, u_1} C
      A : Type u_2
      B : Type u_3
      inst✝⁵ : AddMonoid A
      inst✝⁴ : AddMonoid B
      φ : AddMonoidHom A B
      inst✝³ : CategoryTheory.HasShift C B
      X : CategoryTheory.PullbackShift C φ
      a₁ a₂ a₃ : A
      h : Eq (HAdd.hAdd a₁ a₂) a₃
      b₁ b₂ b₃ : B
      h₁ : Eq b₁ (φ a₁)
      h₂ : Eq b₂ (φ a₂)
      h₃ : Eq b₃ (φ a₃)
      D : Type u_4
      inst✝² : CategoryTheory.Category.{?u.42230, u_4} D
      inst✝¹ : CategoryTheory.HasShift D B
      F : CategoryTheory.Functor C D
      inst✝ : F.CommShift B
      a b : A
      x✝ : CategoryTheory.PullbackShift C φ
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map ((CategoryTheory.pullbackShift …
    -/
    erw [id_comp]
    /-
      case w.w.h
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{?u.41452, u_1} C
      A : Type u_2
      B : Type u_3
      inst✝⁵ : AddMonoid A
      inst✝⁴ : AddMonoid B
      φ : AddMonoidHom A B
      inst✝³ : CategoryTheory.HasShift C B
      X : CategoryTheory.PullbackShift C φ
      a₁ a₂ a₃ : A
      h : Eq (HAdd.hAdd a₁ a₂) a₃
      b₁ b₂ b₃ : B
      h₁ : Eq b₁ (φ a₁)
      h₂ : Eq b₂ (φ a₂)
      h₃ : Eq b₃ (φ a₃)
      D : Type u_4
      inst✝² : CategoryTheory.Category.{?u.42230, u_4} D
      inst✝¹ : CategoryTheory.HasShift D B
      F : CategoryTheory.Functor C D
      inst✝ : F.CommShift B
      a b : A
      x✝ : CategoryTheory.PullbackShift C φ
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map ((CategoryTheory.pullbackShift …
    -/
    slice_rhs 6 7 => erw [← (CommShift.iso (φ b)).hom.naturality]
    /-
      case w.w.h
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{?u.41452, u_1} C
      A : Type u_2
      B : Type u_3
      inst✝⁵ : AddMonoid A
      inst✝⁴ : AddMonoid B
      φ : AddMonoidHom A B
      inst✝³ : CategoryTheory.HasShift C B
      X : CategoryTheory.PullbackShift C φ
      a₁ a₂ a₃ : A
      h : Eq (HAdd.hAdd a₁ a₂) a₃
      b₁ b₂ b₃ : B
      h₁ : Eq b₁ (φ a₁)
      h₂ : Eq b₂ (φ a₂)
      h₃ : Eq b₃ (φ a₃)
      D : Type u_4
      inst✝² : CategoryTheory.Category.{?u.42230, u_4} D
      inst✝¹ : CategoryTheory.HasShift D B
      F : CategoryTheory.Functor C D
      inst✝ : F.CommShift B
      a b : A
      x✝ : CategoryTheory.PullbackShift C φ
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map ((CategoryTheory.pullbackShift …
    -/
    slice_rhs 4 5 => rw [← map_comp, (pullbackShiftIso C φ b (φ b) rfl).hom.naturality, map_comp]
    /-
      case w.w.h
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{?u.41452, u_1} C
      A : Type u_2
      B : Type u_3
      inst✝⁵ : AddMonoid A
      inst✝⁴ : AddMonoid B
      φ : AddMonoidHom A B
      inst✝³ : CategoryTheory.HasShift C B
      X : CategoryTheory.PullbackShift C φ
      a₁ a₂ a₃ : A
      h : Eq (HAdd.hAdd a₁ a₂) a₃
      b₁ b₂ b₃ : B
      h₁ : Eq b₁ (φ a₁)
      h₂ : Eq b₂ (φ a₂)
      h₃ : Eq b₃ (φ a₃)
      D : Type u_4
      inst✝² : CategoryTheory.Category.{?u.42230, u_4} D
      inst✝¹ : CategoryTheory.HasShift D B
      F : CategoryTheory.Functor C D
      inst✝ : F.CommShift B
      a b : A
      x✝ : CategoryTheory.PullbackShift C φ
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map ((CategoryTheory.pullbackShift …
    -/
    simp only [comp_obj, Functor.comp_map, assoc]
    /-
      case w.w.h
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{?u.41452, u_1} C
      A : Type u_2
      B : Type u_3
      inst✝⁵ : AddMonoid A
      inst✝⁴ : AddMonoid B
      φ : AddMonoidHom A B
      inst✝³ : CategoryTheory.HasShift C B
      X : CategoryTheory.PullbackShift C φ
      a₁ a₂ a₃ : A
      h : Eq (HAdd.hAdd a₁ a₂) a₃
      b₁ b₂ b₃ : B
      h₁ : Eq b₁ (φ a₁)
      h₂ : Eq b₂ (φ a₂)
      h₃ : Eq b₃ (φ a₃)
      D : Type u_4
      inst✝² : CategoryTheory.Category.{?u.42230, u_4} D
      inst✝¹ : CategoryTheory.HasShift D B
      F : CategoryTheory.Functor C D
      inst✝ : F.CommShift B
      a b : A
      x✝ : CategoryTheory.PullbackShift C φ
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map ((CategoryTheory.pullbackShift …
    -/
    slice_rhs 3 4 => rw [← map_comp, Iso.inv_hom_id_app, map_id]
    /-
      case w.w.h
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{?u.41452, u_1} C
      A : Type u_2
      B : Type u_3
      inst✝⁵ : AddMonoid A
      inst✝⁴ : AddMonoid B
      φ : AddMonoidHom A B
      inst✝³ : CategoryTheory.HasShift C B
      X : CategoryTheory.PullbackShift C φ
      a₁ a₂ a₃ : A
      h : Eq (HAdd.hAdd a₁ a₂) a₃
      b₁ b₂ b₃ : B
      h₁ : Eq b₁ (φ a₁)
      h₂ : Eq b₂ (φ a₂)
      h₃ : Eq b₃ (φ a₃)
      D : Type u_4
      inst✝² : CategoryTheory.Category.{?u.42230, u_4} D
      inst✝¹ : CategoryTheory.HasShift D B
      F : CategoryTheory.Functor C D
      inst✝ : F.CommShift B
      a b : A
      x✝ : CategoryTheory.PullbackShift C φ
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map ((CategoryTheory.pullbackShift …
    -/
    slice_rhs 4 5 => rw [← map_comp]; erw [← map_comp]; rw [Iso.inv_hom_id_app, map_id, map_id]
    /-
      case w.w.h
      C : Type u_1
      inst✝⁶ : CategoryTheory.Category.{?u.41452, u_1} C
      A : Type u_2
      B : Type u_3
      inst✝⁵ : AddMonoid A
      inst✝⁴ : AddMonoid B
      φ : AddMonoidHom A B
      inst✝³ : CategoryTheory.HasShift C B
      X : CategoryTheory.PullbackShift C φ
      a₁ a₂ a₃ : A
      h : Eq (HAdd.hAdd a₁ a₂) a₃
      b₁ b₂ b₃ : B
      h₁ : Eq b₁ (φ a₁)
      h₂ : Eq b₂ (φ a₂)
      h₃ : Eq b₃ (φ a₃)
      D : Type u_4
      inst✝² : CategoryTheory.Category.{?u.42230, u_4} D
      inst✝¹ : CategoryTheory.HasShift D B
      F : CategoryTheory.Functor C D
      inst✝ : F.CommShift B
      a b : A
      x✝ : CategoryTheory.PullbackShift C φ
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map ((CategoryTheory.pullbackShift …
    -/
    rw [id_comp, id_comp, assoc, assoc]; rfl
                                         /-
                                           🎉 no goals
                                         -/


lemma commShiftPullback_iso_eq (a : A) (b : B) (h : b = φ a) :
    letI : F.CommShift (C := PullbackShift C φ) (D := PullbackShift D φ) A := F.commShiftPullback φ
    F.commShiftIso a (C := PullbackShift C φ) (D := PullbackShift D φ) =
      isoWhiskerRight (pullbackShiftIso C φ a b h) F ≪≫ (F.commShiftIso b) ≪≫
        isoWhiskerLeft F (pullbackShiftIso D φ a b h).symm := by
  /-
    C : Type u_1
    inst✝⁶ : CategoryTheory.Category.{u_6, u_1} C
    A : Type u_2
    B : Type u_3
    inst✝⁵ : AddMonoid A
    inst✝⁴ : AddMonoid B
    φ : AddMonoidHom A B
    inst✝³ : CategoryTheory.HasShift C B
    D : Type u_4
    inst✝² : CategoryTheory.Category.{u_5, u_4} D
    inst✝¹ : CategoryTheory.HasShift D B
    F : CategoryTheory.Functor C D
    inst✝ : F.CommShift B
    a : A
    b : B
    h : Eq b (φ a)
    ⊢ Eq (F.commShiftIso a) ((CategoryTheory.isoWhiskerRight (CategoryTheory.pullb …
  -/
  obtain rfl : b = φ a := h
  /-
    C : Type u_1
    inst✝⁶ : CategoryTheory.Category.{u_6, u_1} C
    A : Type u_2
    B : Type u_3
    inst✝⁵ : AddMonoid A
    inst✝⁴ : AddMonoid B
    φ : AddMonoidHom A B
    inst✝³ : CategoryTheory.HasShift C B
    D : Type u_4
    inst✝² : CategoryTheory.Category.{u_5, u_4} D
    inst✝¹ : CategoryTheory.HasShift D B
    F : CategoryTheory.Functor C D
    inst✝ : F.CommShift B
    a : A
    ⊢ Eq (F.commShiftIso a) ((CategoryTheory.isoWhiskerRight (CategoryTheory.pullb …
  -/
  rfl
  /-
    🎉 no goals
  -/


