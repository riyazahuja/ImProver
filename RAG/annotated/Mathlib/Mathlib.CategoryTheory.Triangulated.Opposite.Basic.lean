/-- As it is unclear whether the opposite category `Cᵒᵖ` should always be equipped
with the shift by `ℤ` such that shifting by `n` on `Cᵒᵖ` corresponds to shifting
by `-n` on `C`, the user shall have to do `open CategoryTheory.Pretriangulated.Opposite`
in order to get this shift and the (pre)triangulated structure on `Cᵒᵖ`. -/

private abbrev OppositeShiftAux :=
  PullbackShift (OppositeShift C ℤ)
                                              /-
                                                C : Type u_1
                                                inst✝¹ : CategoryTheory.Category.{?u.45, u_1} C
                                                inst✝ : CategoryTheory.HasShift C Int
                                                ⊢ ∀ (a b : Int), Eq ((fun n => Neg.neg n) (HAdd.hAdd a b)) (HAdd.hAdd ((fun n  …
                                              -/
    (AddMonoidHom.mk' (fun (n : ℤ) => -n) (by intros; dsimp; omega))
                                                             /-
                                                               🎉 no goals
                                                             -/


/-- The category `Cᵒᵖ` is equipped with the shift such that the shift by `n` on `Cᵒᵖ`
corresponds to the shift by `-n` on `C`. -/
noncomputable scoped instance : HasShift Cᵒᵖ ℤ :=
  (inferInstance : HasShift (OppositeShiftAux C) ℤ)


instance [Preadditive C] [∀ (n : ℤ), (shiftFunctor C n).Additive] (n : ℤ) :
    (shiftFunctor Cᵒᵖ n).Additive :=
  (inferInstance : (shiftFunctor (OppositeShiftAux C) n).Additive)


/-- The shift functor on the opposite category identifies to the opposite functor
of a shift functor on the original category. -/
noncomputable def shiftFunctorOpIso (n m : ℤ) (hnm : n + m = 0) :
    shiftFunctor Cᵒᵖ n ≅ (shiftFunctor C m).op := eqToIso (by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{?u.1328, u_1} C
    inst✝ : CategoryTheory.HasShift C Int
    n m : Int
    hnm : Eq (HAdd.hAdd n m) 0
    ⊢ Eq (CategoryTheory.shiftFunctor (Opposite C) n) (CategoryTheory.shiftFunctor …
  -/
  obtain rfl : m = -n := by omega
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{?u.1328, u_1} C
    inst✝ : CategoryTheory.HasShift C Int
    n : Int
    hnm : Eq (HAdd.hAdd n (Neg.neg n)) 0
    ⊢ Eq (CategoryTheory.shiftFunctor (Opposite C) n) (CategoryTheory.shiftFunctor …
  -/
  rfl)
  /-
    🎉 no goals
  -/


lemma shiftFunctorZero_op_hom_app (X : Cᵒᵖ) :
    (shiftFunctorZero Cᵒᵖ ℤ).hom.app X = (shiftFunctorOpIso C 0 0 (zero_add 0)).hom.app X ≫
      ((shiftFunctorZero C ℤ).inv.app X.unop).op := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.HasShift C Int
    X : Opposite C
    ⊢ Eq ((CategoryTheory.shiftFunctorZero (Opposite C) Int).hom.app X) (CategoryT …
  -/
  erw [@pullbackShiftFunctorZero_hom_app (OppositeShift C ℤ), oppositeShiftFunctorZero_hom_app]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.HasShift C Int
    X : Opposite C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.pullbackShiftIso (Ca …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma shiftFunctorZero_op_inv_app (X : Cᵒᵖ) :
    (shiftFunctorZero Cᵒᵖ ℤ).inv.app X =
      ((shiftFunctorZero C ℤ).hom.app X.unop).op ≫
      (shiftFunctorOpIso C 0 0 (zero_add 0)).inv.app X := by
  rw [← cancel_epi ((shiftFunctorZero Cᵒᵖ ℤ).hom.app X), Iso.hom_inv_id_app,
    shiftFunctorZero_op_hom_app, assoc, ← op_comp_assoc, Iso.hom_inv_id_app, op_id,
    id_comp, Iso.hom_inv_id_app]


lemma shiftFunctorAdd'_op_hom_app (X : Cᵒᵖ) (a₁ a₂ a₃ : ℤ) (h : a₁ + a₂ = a₃)
    (b₁ b₂ b₃ : ℤ) (h₁ : a₁ + b₁ = 0) (h₂ : a₂ + b₂ = 0) (h₃ : a₃ + b₃ = 0) :
    (shiftFunctorAdd' Cᵒᵖ a₁ a₂ a₃ h).hom.app X =
      (shiftFunctorOpIso C _ _ h₃).hom.app X ≫
                                          /-
                                            C : Type u_1
                                            inst✝¹ : CategoryTheory.Category.{?u.8804, u_1} C
                                            inst✝ : CategoryTheory.HasShift C Int
                                            X : Opposite C
                                            a₁ a₂ a₃ : Int
                                            h : Eq (HAdd.hAdd a₁ a₂) a₃
                                            b₁ b₂ b₃ : Int
                                            h₁ : Eq (HAdd.hAdd a₁ b₁) 0
                                            h₂ : Eq (HAdd.hAdd a₂ b₂) 0
                                            h₃ : Eq (HAdd.hAdd a₃ b₃) 0
                                            ⊢ Eq (HAdd.hAdd b₁ b₂) b₃
                                          -/
        ((shiftFunctorAdd' C b₁ b₂ b₃ (by omega)).inv.app X.unop).op ≫
                                          /-
                                            🎉 no goals
                                          -/
        (shiftFunctorOpIso C _ _ h₂).inv.app _ ≫
        (shiftFunctor Cᵒᵖ a₂).map ((shiftFunctorOpIso C _ _ h₁).inv.app X) := by
  erw [@pullbackShiftFunctorAdd'_hom_app (OppositeShift C ℤ) _ _ _ _ _ _ _ X
    a₁ a₂ a₃ h b₁ b₂ b₃ (by dsimp; omega) (by dsimp; omega) (by dsimp; omega)]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.HasShift C Int
    X : Opposite C
    a₁ a₂ a₃ : Int
    h : Eq (HAdd.hAdd a₁ a₂) a₃
    b₁ b₂ b₃ : Int
    h₁ : Eq (HAdd.hAdd a₁ b₁) 0
    h₂ : Eq (HAdd.hAdd a₂ b₂) 0
    h₃ : Eq (HAdd.hAdd a₃ b₃) 0
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.pullbackShiftIso (Ca …
  -/
  rw [oppositeShiftFunctorAdd'_hom_app]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.HasShift C Int
    X : Opposite C
    a₁ a₂ a₃ : Int
    h : Eq (HAdd.hAdd a₁ a₂) a₃
    b₁ b₂ b₃ : Int
    h₁ : Eq (HAdd.hAdd a₁ b₁) 0
    h₂ : Eq (HAdd.hAdd a₂ b₂) 0
    h₃ : Eq (HAdd.hAdd a₃ b₃) 0
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.pullbackShiftIso (Ca …
  -/
  obtain rfl : b₁ = -a₁ := by omega
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.HasShift C Int
    X : Opposite C
    a₁ a₂ a₃ : Int
    h : Eq (HAdd.hAdd a₁ a₂) a₃
    b₂ b₃ : Int
    h₂ : Eq (HAdd.hAdd a₂ b₂) 0
    h₃ : Eq (HAdd.hAdd a₃ b₃) 0
    h₁ : Eq (HAdd.hAdd a₁ (Neg.neg a₁)) 0
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.pullbackShiftIso (Ca …
  -/
  obtain rfl : b₂ = -a₂ := by omega
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.HasShift C Int
    X : Opposite C
    a₁ a₂ a₃ : Int
    h : Eq (HAdd.hAdd a₁ a₂) a₃
    b₃ : Int
    h₃ : Eq (HAdd.hAdd a₃ b₃) 0
    h₁ : Eq (HAdd.hAdd a₁ (Neg.neg a₁)) 0
    h₂ : Eq (HAdd.hAdd a₂ (Neg.neg a₂)) 0
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.pullbackShiftIso (Ca …
  -/
  obtain rfl : b₃ = -a₃ := by omega
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.HasShift C Int
    X : Opposite C
    a₁ a₂ a₃ : Int
    h : Eq (HAdd.hAdd a₁ a₂) a₃
    h₁ : Eq (HAdd.hAdd a₁ (Neg.neg a₁)) 0
    h₂ : Eq (HAdd.hAdd a₂ (Neg.neg a₂)) 0
    h₃ : Eq (HAdd.hAdd a₃ (Neg.neg a₃)) 0
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.pullbackShiftIso (Ca …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma shiftFunctorAdd'_op_inv_app (X : Cᵒᵖ) (a₁ a₂ a₃ : ℤ) (h : a₁ + a₂ = a₃)
    (b₁ b₂ b₃ : ℤ) (h₁ : a₁ + b₁ = 0) (h₂ : a₂ + b₂ = 0) (h₃ : a₃ + b₃ = 0) :
    (shiftFunctorAdd' Cᵒᵖ a₁ a₂ a₃ h).inv.app X =
      (shiftFunctor Cᵒᵖ a₂).map ((shiftFunctorOpIso C _ _ h₁).hom.app X) ≫
      (shiftFunctorOpIso C _ _ h₂).hom.app _ ≫
                                        /-
                                          C : Type u_1
                                          inst✝¹ : CategoryTheory.Category.{?u.28094, u_1} C
                                          inst✝ : CategoryTheory.HasShift C Int
                                          X : Opposite C
                                          a₁ a₂ a₃ : Int
                                          h : Eq (HAdd.hAdd a₁ a₂) a₃
                                          b₁ b₂ b₃ : Int
                                          h₁ : Eq (HAdd.hAdd a₁ b₁) 0
                                          h₂ : Eq (HAdd.hAdd a₂ b₂) 0
                                          h₃ : Eq (HAdd.hAdd a₃ b₃) 0
                                          ⊢ Eq (HAdd.hAdd b₁ b₂) b₃
                                        -/
      ((shiftFunctorAdd' C b₁ b₂ b₃ (by omega)).hom.app X.unop).op ≫
                                        /-
                                          🎉 no goals
                                        -/
      (shiftFunctorOpIso C _ _ h₃).inv.app X := by
  rw [← cancel_epi ((shiftFunctorAdd' Cᵒᵖ a₁ a₂ a₃ h).hom.app X), Iso.hom_inv_id_app,
    shiftFunctorAdd'_op_hom_app X a₁ a₂ a₃ h b₁ b₂ b₃ h₁ h₂ h₃,
    assoc, assoc, assoc, ← Functor.map_comp_assoc, Iso.inv_hom_id_app]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.HasShift C Int
    X : Opposite C
    a₁ a₂ a₃ : Int
    h : Eq (HAdd.hAdd a₁ a₂) a₃
    b₁ b₂ b₃ : Int
    h₁ : Eq (HAdd.hAdd a₁ b₁) 0
    h₂ : Eq (HAdd.hAdd a₂ b₂) 0
    h₃ : Eq (HAdd.hAdd a₃ b₃) 0
    ⊢ Eq (CategoryTheory.CategoryStruct.id ((CategoryTheory.shiftFunctor (Opposite …
  -/
  erw [Functor.map_id, id_comp, Iso.inv_hom_id_app_assoc]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.HasShift C Int
    X : Opposite C
    a₁ a₂ a₃ : Int
    h : Eq (HAdd.hAdd a₁ a₂) a₃
    b₁ b₂ b₃ : Int
    h₁ : Eq (HAdd.hAdd a₁ b₁) 0
    h₂ : Eq (HAdd.hAdd a₂ b₂) 0
    h₃ : Eq (HAdd.hAdd a₃ b₃) 0
    ⊢ Eq (CategoryTheory.CategoryStruct.id ((CategoryTheory.shiftFunctor (Opposite …
  -/
  rw [← op_comp_assoc, Iso.hom_inv_id_app, op_id, id_comp, Iso.hom_inv_id_app]
  /-
    🎉 no goals
  -/


lemma shiftFunctor_op_map (n m : ℤ) (hnm : n + m = 0) {K L : Cᵒᵖ} (φ : K ⟶ L) :
    (shiftFunctor Cᵒᵖ n).map φ =
      (shiftFunctorOpIso C n m hnm).hom.app K ≫ ((shiftFunctor C m).map φ.unop).op ≫
        (shiftFunctorOpIso C n m hnm).inv.app L :=
  (NatIso.naturality_2 (shiftFunctorOpIso C n m hnm) φ).symm


variable (C) in
/-- The autoequivalence `Cᵒᵖ ≌ Cᵒᵖ` whose functor is `shiftFunctor Cᵒᵖ n` and whose inverse
functor is `(shiftFunctor C n).op`. Do not unfold the definitions of the unit and counit
isomorphisms: the compatibilities they satisfy are stated as separate lemmas. -/
@[simps functor inverse]
noncomputable def opShiftFunctorEquivalence (n : ℤ) : Cᵒᵖ ≌ Cᵒᵖ where
  functor := shiftFunctor Cᵒᵖ n
  inverse := (shiftFunctor C n).op
  unitIso := NatIso.op (shiftFunctorCompIsoId C (-n) n n.add_left_neg) ≪≫
    isoWhiskerRight (shiftFunctorOpIso C n (-n) n.add_right_neg).symm (shiftFunctor C n).op
  counitIso := isoWhiskerLeft _ (shiftFunctorOpIso C n (-n) n.add_right_neg) ≪≫
    NatIso.op (shiftFunctorCompIsoId C n (-n) n.add_right_neg).symm
  functor_unitIso_comp X := Quiver.Hom.unop_inj (by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.57577, u_1} C
      inst✝ : CategoryTheory.HasShift C Int
      n : Int
      X : Opposite C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctor (Opposi …
    -/
    dsimp [shiftFunctorOpIso]
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.57577, u_1} C
      inst✝ : CategoryTheory.HasShift C Int
      n : Int
      X : Opposite C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    erw [comp_id, Functor.map_id, comp_id]
    change (shiftFunctorCompIsoId C n (-n) (add_neg_cancel n)).inv.app (X.unop⟦-n⟧) ≫
      ((shiftFunctorCompIsoId C (-n) n (neg_add_cancel n)).hom.app X.unop)⟦-n⟧' = 𝟙 _
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.57577, u_1} C
      inst✝ : CategoryTheory.HasShift C Int
      n : Int
      X : Opposite C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctorCompIsoI …
    -/
    rw [shift_shiftFunctorCompIsoId_neg_add_cancel_hom_app n X.unop, Iso.inv_hom_id_app])
    /-
      🎉 no goals
    -/


@[reassoc (attr := simp)]
lemma opShiftFunctorEquivalence_unitIso_hom_naturality (n : ℤ) {X Y : Cᵒᵖ} (f : X ⟶ Y) :
    f ≫ (opShiftFunctorEquivalence C n).unitIso.hom.app Y =
      (opShiftFunctorEquivalence C n).unitIso.hom.app X ≫ (f⟦n⟧').unop⟦n⟧'.op :=
  (opShiftFunctorEquivalence C n).unitIso.hom.naturality f


@[reassoc (attr := simp)]
lemma opShiftFunctorEquivalence_unitIso_inv_naturality (n : ℤ) {X Y : Cᵒᵖ} (f : X ⟶ Y) :
    (f⟦n⟧').unop⟦n⟧'.op ≫ (opShiftFunctorEquivalence C n).unitIso.inv.app Y =
      (opShiftFunctorEquivalence C n).unitIso.inv.app X ≫ f :=
  (opShiftFunctorEquivalence C n).unitIso.inv.naturality f


@[reassoc (attr := simp)]
lemma opShiftFunctorEquivalence_counitIso_hom_naturality (n : ℤ) {X Y : Cᵒᵖ} (f : X ⟶ Y) :
    f.unop⟦n⟧'.op⟦n⟧' ≫ (opShiftFunctorEquivalence C n).counitIso.hom.app Y =
      (opShiftFunctorEquivalence C n).counitIso.hom.app X ≫ f :=
  (opShiftFunctorEquivalence C n).counitIso.hom.naturality f


@[reassoc (attr := simp)]
lemma opShiftFunctorEquivalence_counitIso_inv_naturality (n : ℤ) {X Y : Cᵒᵖ} (f : X ⟶ Y) :
    f ≫ (opShiftFunctorEquivalence C n).counitIso.inv.app Y =
      (opShiftFunctorEquivalence C n).counitIso.inv.app X ≫ f.unop⟦n⟧'.op⟦n⟧' :=
  (opShiftFunctorEquivalence C n).counitIso.inv.naturality f


lemma opShiftFunctorEquivalence_zero_unitIso_hom_app (X : Cᵒᵖ) :
    (opShiftFunctorEquivalence C 0).unitIso.hom.app X =
      ((shiftFunctorZero C ℤ).hom.app X.unop).op ≫
      (((shiftFunctorZero Cᵒᵖ ℤ).inv.app X).unop⟦(0 : ℤ)⟧').op := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.HasShift C Int
    X : Opposite C
    ⊢ Eq ((CategoryTheory.Pretriangulated.opShiftFunctorEquivalence C 0).unitIso.h …
  -/
  apply Quiver.Hom.unop_inj
  /-
    case a
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.HasShift C Int
    X : Opposite C
    ⊢ Eq ((CategoryTheory.Pretriangulated.opShiftFunctorEquivalence C 0).unitIso.h …
  -/
  dsimp [opShiftFunctorEquivalence]
  rw [shiftFunctorZero_op_inv_app, unop_comp, Quiver.Hom.unop_op, Functor.map_comp,
    shiftFunctorCompIsoId_zero_zero_hom_app, assoc]


lemma opShiftFunctorEquivalence_zero_unitIso_inv_app (X : Cᵒᵖ) :
    (opShiftFunctorEquivalence C 0).unitIso.inv.app X =
      (((shiftFunctorZero Cᵒᵖ ℤ).hom.app X).unop⟦(0 : ℤ)⟧').op ≫
        ((shiftFunctorZero C ℤ).inv.app X.unop).op := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.HasShift C Int
    X : Opposite C
    ⊢ Eq ((CategoryTheory.Pretriangulated.opShiftFunctorEquivalence C 0).unitIso.i …
  -/
  apply Quiver.Hom.unop_inj
  /-
    case a
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.HasShift C Int
    X : Opposite C
    ⊢ Eq ((CategoryTheory.Pretriangulated.opShiftFunctorEquivalence C 0).unitIso.i …
  -/
  dsimp [opShiftFunctorEquivalence]
  rw [shiftFunctorZero_op_hom_app, unop_comp, Quiver.Hom.unop_op, Functor.map_comp,
    shiftFunctorCompIsoId_zero_zero_inv_app, assoc]


lemma opShiftFunctorEquivalence_unitIso_hom_app_eq (X : Cᵒᵖ) (m n p : ℤ) (h : m + n = p) :
    (opShiftFunctorEquivalence C p).unitIso.hom.app X =
      (opShiftFunctorEquivalence C n).unitIso.hom.app X ≫
      (((opShiftFunctorEquivalence C m).unitIso.hom.app (X⟦n⟧)).unop⟦n⟧').op ≫
      ((shiftFunctorAdd' C m n p h).hom.app _).op ≫
                                        /-
                                          C : Type u_1
                                          inst✝¹ : CategoryTheory.Category.{?u.84734, u_1} C
                                          inst✝ : CategoryTheory.HasShift C Int
                                          X : Opposite C
                                          m n p : Int
                                          h : Eq (HAdd.hAdd m n) p
                                          ⊢ Eq (HAdd.hAdd n m) p
                                        -/
      (((shiftFunctorAdd' Cᵒᵖ n m p (by omega)).inv.app X).unop⟦p⟧').op := by
                                        /-
                                          🎉 no goals
                                        -/
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.HasShift C Int
    X : Opposite C
    m n p : Int
    h : Eq (HAdd.hAdd m n) p
    ⊢ Eq ((CategoryTheory.Pretriangulated.opShiftFunctorEquivalence C p).unitIso.h …
  -/
  dsimp [opShiftFunctorEquivalence]
  simp only [shiftFunctorAdd'_op_inv_app _ n m p (by omega) _ _ _ (add_neg_cancel n)
    (add_neg_cancel m) (add_neg_cancel p), shiftFunctor_op_map _ _ (add_neg_cancel m),
    Category.assoc, Iso.inv_hom_id_app_assoc]
  erw [Functor.map_id, Functor.map_id, Functor.map_id, Functor.map_id,
    id_comp, id_comp, id_comp, comp_id, comp_id]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.HasShift C Int
    X : Opposite C
    m n p : Int
    h : Eq (HAdd.hAdd m n) p
    ⊢ Eq ((CategoryTheory.shiftFunctorCompIsoId C (Neg.neg p) p ⋯).hom.app (Opposi …
  -/
  dsimp
  rw [comp_id, shiftFunctorCompIsoId_add'_hom_app _ _ _ _ _ _
    (neg_add_cancel m) (neg_add_cancel n) (neg_add_cancel p) h]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.HasShift C Int
    X : Opposite C
    m n p : Int
    h : Eq (HAdd.hAdd m n) p
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctor C p).ma …
  -/
  dsimp
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.HasShift C Int
    X : Opposite C
    m n p : Int
    h : Eq (HAdd.hAdd m n) p
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  rw [Category.assoc, Category.assoc]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.HasShift C Int
    X : Opposite C
    m n p : Int
    h : Eq (HAdd.hAdd m n) p
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctorCompIsoI …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma opShiftFunctorEquivalence_unitIso_inv_app_eq (X : Cᵒᵖ) (m n p : ℤ) (h : m + n = p) :
    (opShiftFunctorEquivalence C p).unitIso.inv.app X =
                                        /-
                                          C : Type u_1
                                          inst✝¹ : CategoryTheory.Category.{?u.132164, u_1} C
                                          inst✝ : CategoryTheory.HasShift C Int
                                          X : Opposite C
                                          m n p : Int
                                          h : Eq (HAdd.hAdd m n) p
                                          ⊢ Eq (HAdd.hAdd n m) p
                                        -/
      (((shiftFunctorAdd' Cᵒᵖ n m p (by omega)).hom.app X).unop⟦p⟧').op ≫
                                        /-
                                          🎉 no goals
                                        -/
      ((shiftFunctorAdd' C m n p h).inv.app _).op ≫
      (((opShiftFunctorEquivalence C m).unitIso.inv.app (X⟦n⟧)).unop⟦n⟧').op ≫
      (opShiftFunctorEquivalence C n).unitIso.inv.app X := by
  rw [← cancel_mono ((opShiftFunctorEquivalence C p).unitIso.hom.app X), Iso.inv_hom_id_app,
    opShiftFunctorEquivalence_unitIso_hom_app_eq _ _ _ _ h,
    Category.assoc, Category.assoc, Category.assoc, Iso.inv_hom_id_app_assoc]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.HasShift C Int
    X : Opposite C
    m n p : Int
    h : Eq (HAdd.hAdd m n) p
    ⊢ Eq (CategoryTheory.CategoryStruct.id (((CategoryTheory.Pretriangulated.opShi …
  -/
  apply Quiver.Hom.unop_inj
  /-
    case a
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.HasShift C Int
    X : Opposite C
    m n p : Int
    h : Eq (HAdd.hAdd m n) p
    ⊢ Eq (CategoryTheory.CategoryStruct.id (((CategoryTheory.Pretriangulated.opShi …
  -/
  dsimp
  simp only [Category.assoc, ← Functor.map_comp_assoc, Iso.hom_inv_id_app_assoc,
    ← unop_comp, Iso.inv_hom_id_app, Functor.comp_obj, Functor.op_obj, unop_id,
    Functor.map_id, id_comp, ← Functor.map_comp, Iso.hom_inv_id_app]


lemma shift_unop_opShiftFunctorEquivalence_counitIso_inv_app (X : Cᵒᵖ) (n : ℤ) :
    ((opShiftFunctorEquivalence C n).counitIso.inv.app X).unop⟦n⟧' =
      ((opShiftFunctorEquivalence C n).unitIso.hom.app ((Opposite.op ((X.unop)⟦n⟧)))).unop :=
  Quiver.Hom.op_inj ((opShiftFunctorEquivalence C n).unit_app_inverse X).symm


lemma shift_unop_opShiftFunctorEquivalence_counitIso_hom_app (X : Cᵒᵖ) (n : ℤ) :
    ((opShiftFunctorEquivalence C n).counitIso.hom.app X).unop⟦n⟧' =
      ((opShiftFunctorEquivalence C n).unitIso.inv.app ((Opposite.op (X.unop⟦n⟧)))).unop :=
  Quiver.Hom.op_inj ((opShiftFunctorEquivalence C n).unitInv_app_inverse X).symm


lemma opShiftFunctorEquivalence_counitIso_inv_app_shift (X : Cᵒᵖ) (n : ℤ) :
    (opShiftFunctorEquivalence C n).counitIso.inv.app (X⟦n⟧) =
      ((opShiftFunctorEquivalence C n).unitIso.hom.app X)⟦n⟧' :=
  (opShiftFunctorEquivalence C n).counitInv_app_functor X


lemma opShiftFunctorEquivalence_counitIso_hom_app_shift (X : Cᵒᵖ) (n : ℤ) :
    (opShiftFunctorEquivalence C n).counitIso.hom.app (X⟦n⟧) =
      ((opShiftFunctorEquivalence C n).unitIso.inv.app X)⟦n⟧' :=
  (opShiftFunctorEquivalence C n).counit_app_functor X


