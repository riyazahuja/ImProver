/-- `CatCommSq T L R B` expresses that there is a 2-commutative square of functors, where
the functors `T`, `L`, `R` and `B` are respectively the left, top, right and bottom functors
of the square. -/
@[ext]
class CatCommSq where
  /-- the isomorphism corresponding to a 2-commutative diagram -/
  iso' : T ⋙ R ≅ L ⋙ B


/-- Assuming `[CatCommSq T L R B]`, `iso T L R B` is the isomorphism `T ⋙ R ≅ L ⋙ B`
given by the 2-commutative square. -/
def iso [h : CatCommSq T L R B] : T ⋙ R ≅ L ⋙ B := h.iso'


/-- Horizontal composition of 2-commutative squares -/
@[simps! iso'_hom_app iso'_inv_app]
def hComp (T₁ : C₁ ⥤ C₂) (T₂ : C₂ ⥤ C₃) (V₁ : C₁ ⥤ C₄) (V₂ : C₂ ⥤ C₅) (V₃ : C₃ ⥤ C₆)
    (B₁ : C₄ ⥤ C₅) (B₂ : C₅ ⥤ C₆) [CatCommSq T₁ V₁ V₂ B₁] [CatCommSq T₂ V₂ V₃ B₂] :
    CatCommSq (T₁ ⋙ T₂) V₁ V₃ (B₁ ⋙ B₂) where
  iso' := Functor.associator _ _ _ ≪≫ isoWhiskerLeft T₁ (iso T₂ V₂ V₃ B₂) ≪≫
    (Functor.associator _ _ _).symm ≪≫ isoWhiskerRight (iso T₁ V₁ V₂ B₁) B₂ ≪≫
    Functor.associator _ _ _


/-- Vertical composition of 2-commutative squares -/
@[simps! iso'_hom_app iso'_inv_app]
def vComp (L₁ : C₁ ⥤ C₂) (L₂ : C₂ ⥤ C₃) (H₁ : C₁ ⥤ C₄) (H₂ : C₂ ⥤ C₅) (H₃ : C₃ ⥤ C₆)
    (R₁ : C₄ ⥤ C₅) (R₂ : C₅ ⥤ C₆) [CatCommSq H₁ L₁ R₁ H₂] [CatCommSq H₂ L₂ R₂ H₃] :
    CatCommSq H₁ (L₁ ⋙ L₂) (R₁ ⋙ R₂) H₃ where
  iso' := (Functor.associator _ _ _).symm ≪≫ isoWhiskerRight (iso H₁ L₁ R₁ H₂) R₂ ≪≫
      Functor.associator _ _ _ ≪≫ isoWhiskerLeft L₁ (iso H₂ L₂ R₂ H₃) ≪≫
      (Functor.associator _ _ _).symm


/-- Horizontal inverse of a 2-commutative square -/
@[simps! iso'_hom_app iso'_inv_app]
def hInv (_ : CatCommSq T.functor L R B.functor) : CatCommSq T.inverse R L B.inverse where
  iso' := isoWhiskerLeft _ (L.rightUnitor.symm ≪≫ isoWhiskerLeft L B.unitIso ≪≫
      (Functor.associator _ _ _).symm ≪≫
      isoWhiskerRight (iso T.functor L R B.functor).symm B.inverse ≪≫
      Functor.associator _ _ _  ) ≪≫ (Functor.associator _ _ _).symm ≪≫
      isoWhiskerRight T.counitIso _ ≪≫ Functor.leftUnitor _


lemma hInv_hInv (h : CatCommSq T.functor L R B.functor) :
    hInv T.symm R L B.symm (hInv T L R B h) = h := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    C₄ : Type u_4
    inst✝³ : CategoryTheory.Category.{u_8, u_1} C₁
    inst✝² : CategoryTheory.Category.{u_9, u_2} C₂
    inst✝¹ : CategoryTheory.Category.{u_10, u_3} C₃
    inst✝ : CategoryTheory.Category.{u_7, u_4} C₄
    T : CategoryTheory.Equivalence C₁ C₂
    L : CategoryTheory.Functor C₁ C₃
    R : CategoryTheory.Functor C₂ C₄
    B : CategoryTheory.Equivalence C₃ C₄
    h : CategoryTheory.CatCommSq T.functor L R B.functor
    ⊢ Eq (CategoryTheory.CatCommSq.hInv T.symm R L B.symm (CategoryTheory.CatCommS …
  -/
  ext X
  erw [← cancel_mono (B.functor.map (L.map (T.unitIso.hom.app X))),
    ← h.iso'.hom.naturality (T.unitIso.hom.app X), hInv_iso'_hom_app, hInv_iso'_inv_app]
  /-
    case iso'.w.w.h
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    C₄ : Type u_4
    inst✝³ : CategoryTheory.Category.{u_8, u_1} C₁
    inst✝² : CategoryTheory.Category.{u_9, u_2} C₂
    inst✝¹ : CategoryTheory.Category.{u_10, u_3} C₃
    inst✝ : CategoryTheory.Category.{u_7, u_4} C₄
    T : CategoryTheory.Equivalence C₁ C₂
    L : CategoryTheory.Functor C₁ C₃
    R : CategoryTheory.Functor C₂ C₄
    B : CategoryTheory.Equivalence C₃ C₄
    h : CategoryTheory.CatCommSq T.functor L R B.functor
    X : C₁
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  dsimp
  simp only [Functor.comp_obj, assoc, ← Functor.map_comp, Iso.inv_hom_id_app,
    Equivalence.counitInv_app_functor, Functor.map_id]
  simp only [Functor.map_comp, Equivalence.fun_inv_map, assoc,
    Equivalence.counitInv_functor_comp, comp_id, Iso.inv_hom_id_app_assoc]
  /-
    case iso'.w.w.h
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    C₄ : Type u_4
    inst✝³ : CategoryTheory.Category.{u_8, u_1} C₁
    inst✝² : CategoryTheory.Category.{u_9, u_2} C₂
    inst✝¹ : CategoryTheory.Category.{u_10, u_3} C₃
    inst✝ : CategoryTheory.Category.{u_7, u_4} C₄
    T : CategoryTheory.Equivalence C₁ C₂
    L : CategoryTheory.Functor C₁ C₃
    R : CategoryTheory.Functor C₂ C₄
    B : CategoryTheory.Equivalence C₃ C₄
    h : CategoryTheory.CatCommSq T.functor L R B.functor
    X : C₁
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (R.map (T.functor.map (T.unit.app X)) …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- In a square of categories, when the top and bottom functors are part
of equivalence of categories, it is equivalent to show 2-commutativity for
the functors of these equivalences or for their inverses. -/
def hInvEquiv : CatCommSq T.functor L R B.functor ≃ CatCommSq T.inverse R L B.inverse where
  toFun := hInv T L R B
  invFun := hInv T.symm R L B.symm
  left_inv := hInv_hInv T L R B
  right_inv := hInv_hInv T.symm R L B.symm


/-- Vertical inverse of a 2-commutative square -/
@[simps! iso'_hom_app iso'_inv_app]
def vInv (_ : CatCommSq T L.functor R.functor B) : CatCommSq B L.inverse R.inverse T where
  iso' := isoWhiskerRight (B.leftUnitor.symm ≪≫ isoWhiskerRight L.counitIso.symm B ≪≫
      Functor.associator _ _ _ ≪≫
      isoWhiskerLeft L.inverse (iso T L.functor R.functor B).symm) R.inverse ≪≫
      Functor.associator _ _ _ ≪≫ isoWhiskerLeft _ (Functor.associator _ _ _) ≪≫
      (Functor.associator _ _ _ ).symm ≪≫ isoWhiskerLeft _ R.unitIso.symm ≪≫
      Functor.rightUnitor _


lemma vInv_vInv (h : CatCommSq T L.functor R.functor B) :
    vInv B L.symm R.symm T (vInv T L R B h) = h := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    C₄ : Type u_4
    inst✝³ : CategoryTheory.Category.{u_8, u_1} C₁
    inst✝² : CategoryTheory.Category.{u_9, u_2} C₂
    inst✝¹ : CategoryTheory.Category.{u_10, u_3} C₃
    inst✝ : CategoryTheory.Category.{u_7, u_4} C₄
    T : CategoryTheory.Functor C₁ C₂
    L : CategoryTheory.Equivalence C₁ C₃
    R : CategoryTheory.Equivalence C₂ C₄
    B : CategoryTheory.Functor C₃ C₄
    h : CategoryTheory.CatCommSq T L.functor R.functor B
    ⊢ Eq (CategoryTheory.CatCommSq.vInv B L.symm R.symm T (CategoryTheory.CatCommS …
  -/
  ext X
  /-
    case iso'.w.w.h
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    C₄ : Type u_4
    inst✝³ : CategoryTheory.Category.{u_8, u_1} C₁
    inst✝² : CategoryTheory.Category.{u_9, u_2} C₂
    inst✝¹ : CategoryTheory.Category.{u_10, u_3} C₃
    inst✝ : CategoryTheory.Category.{u_7, u_4} C₄
    T : CategoryTheory.Functor C₁ C₂
    L : CategoryTheory.Equivalence C₁ C₃
    R : CategoryTheory.Equivalence C₂ C₄
    B : CategoryTheory.Functor C₃ C₄
    h : CategoryTheory.CatCommSq T L.functor R.functor B
    X : C₁
    ⊢ Eq (CategoryTheory.CatCommSq.iso'.hom.app X) (CategoryTheory.CatCommSq.iso'. …
  -/
  erw [vInv_iso'_hom_app, vInv_iso'_inv_app]
  /-
    case iso'.w.w.h
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    C₄ : Type u_4
    inst✝³ : CategoryTheory.Category.{u_8, u_1} C₁
    inst✝² : CategoryTheory.Category.{u_9, u_2} C₂
    inst✝¹ : CategoryTheory.Category.{u_10, u_3} C₃
    inst✝ : CategoryTheory.Category.{u_7, u_4} C₄
    T : CategoryTheory.Functor C₁ C₂
    L : CategoryTheory.Equivalence C₁ C₃
    R : CategoryTheory.Equivalence C₂ C₄
    B : CategoryTheory.Functor C₃ C₄
    h : CategoryTheory.CatCommSq T L.functor R.functor B
    X : C₁
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (R.symm.inverse.map (T.map (L.symm.co …
  -/
  dsimp
  /-
    case iso'.w.w.h
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    C₄ : Type u_4
    inst✝³ : CategoryTheory.Category.{u_8, u_1} C₁
    inst✝² : CategoryTheory.Category.{u_9, u_2} C₂
    inst✝¹ : CategoryTheory.Category.{u_10, u_3} C₃
    inst✝ : CategoryTheory.Category.{u_7, u_4} C₄
    T : CategoryTheory.Functor C₁ C₂
    L : CategoryTheory.Equivalence C₁ C₃
    R : CategoryTheory.Equivalence C₂ C₄
    B : CategoryTheory.Functor C₃ C₄
    h : CategoryTheory.CatCommSq T L.functor R.functor B
    X : C₁
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (R.functor.map (T.map (L.unitIso.hom. …
  -/
  rw [← cancel_mono (B.map (L.functor.map (NatTrans.app L.unitIso.hom X)))]
  /-
    case iso'.w.w.h
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    C₄ : Type u_4
    inst✝³ : CategoryTheory.Category.{u_8, u_1} C₁
    inst✝² : CategoryTheory.Category.{u_9, u_2} C₂
    inst✝¹ : CategoryTheory.Category.{u_10, u_3} C₃
    inst✝ : CategoryTheory.Category.{u_7, u_4} C₄
    T : CategoryTheory.Functor C₁ C₂
    L : CategoryTheory.Equivalence C₁ C₃
    R : CategoryTheory.Equivalence C₂ C₄
    B : CategoryTheory.Functor C₃ C₄
    h : CategoryTheory.CatCommSq T L.functor R.functor B
    X : C₁
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  erw [← (iso T L.functor R.functor B).hom.naturality (L.unitIso.hom.app X)]
  /-
    case iso'.w.w.h
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    C₄ : Type u_4
    inst✝³ : CategoryTheory.Category.{u_8, u_1} C₁
    inst✝² : CategoryTheory.Category.{u_9, u_2} C₂
    inst✝¹ : CategoryTheory.Category.{u_10, u_3} C₃
    inst✝ : CategoryTheory.Category.{u_7, u_4} C₄
    T : CategoryTheory.Functor C₁ C₂
    L : CategoryTheory.Equivalence C₁ C₃
    R : CategoryTheory.Equivalence C₂ C₄
    B : CategoryTheory.Functor C₃ C₄
    h : CategoryTheory.CatCommSq T L.functor R.functor B
    X : C₁
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  dsimp
  simp only [Functor.map_comp, Equivalence.fun_inv_map, Functor.comp_obj,
    Functor.id_obj, assoc, Iso.inv_hom_id_app_assoc, Iso.inv_hom_id_app, comp_id]
  erw [← B.map_comp, L.counit_app_functor, ← L.functor.map_comp, ← NatTrans.comp_app,
    Iso.inv_hom_id, NatTrans.id_app, L.functor.map_id, B.map_id, comp_id, R.counit_app_functor,
    ← R.functor.map_comp_assoc, ← R.functor.map_comp_assoc, assoc, ← NatTrans.comp_app,
    Iso.hom_inv_id, NatTrans.id_app, comp_id]


/-- In a square of categories, when the left and right functors are part
of equivalence of categories, it is equivalent to show 2-commutativity for
the functors of these equivalences or for their inverses. -/
def vInvEquiv : CatCommSq T L.functor R.functor B ≃ CatCommSq B L.inverse R.inverse T where
  toFun := vInv T L R B
  invFun := vInv B L.symm R.symm T
  left_inv := vInv_vInv T L R B
  right_inv := vInv_vInv B L.symm R.symm T


