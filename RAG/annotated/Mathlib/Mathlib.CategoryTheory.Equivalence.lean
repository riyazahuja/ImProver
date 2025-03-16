/-- We define an equivalence as a (half)-adjoint equivalence, a pair of functors with
  a unit and counit which are natural isomorphisms and the triangle law `Fη ≫ εF = 1`, or in other
  words the composite `F ⟶ FGF ⟶ F` is the identity.

  In `unit_inverse_comp`, we show that this is actually an adjoint equivalence, i.e., that the
  composite `G ⟶ GFG ⟶ G` is also the identity.

  The triangle equation is written as a family of equalities between morphisms, it is more
  complicated if we write it as an equality of natural transformations, because then we would have
  to insert natural transformations like `F ⟶ F1`.

See <https://stacks.math.columbia.edu/tag/001J>
-/
@[ext]
structure Equivalence (C : Type u₁) (D : Type u₂) [Category.{v₁} C] [Category.{v₂} D] where mk' ::
  /-- A functor in one direction -/
  functor : C ⥤ D
  /-- A functor in the other direction -/
  inverse : D ⥤ C
  /-- The composition `functor ⋙ inverse` is isomorphic to the identity -/
  unitIso : 𝟭 C ≅ functor ⋙ inverse
  /-- The composition `inverse ⋙ functor` is also isomorphic to the identity -/
  counitIso : inverse ⋙ functor ≅ 𝟭 D
  /-- The natural isomorphisms compose to the identity. -/
  functor_unitIso_comp :
    ∀ X : C, functor.map (unitIso.hom.app X) ≫ counitIso.hom.app (functor.obj X) =
      𝟙 (functor.obj X) := by aesop_cat


/-- We infix the usual notation for an equivalence -/
infixr:10 " ≌ " => Equivalence


/-- The unit of an equivalence of categories. -/
abbrev unit (e : C ≌ D) : 𝟭 C ⟶ e.functor ⋙ e.inverse :=
  e.unitIso.hom


/-- The counit of an equivalence of categories. -/
abbrev counit (e : C ≌ D) : e.inverse ⋙ e.functor ⟶ 𝟭 D :=
  e.counitIso.hom


/-- The inverse of the unit of an equivalence of categories. -/
abbrev unitInv (e : C ≌ D) : e.functor ⋙ e.inverse ⟶ 𝟭 C :=
  e.unitIso.inv


/-- The inverse of the counit of an equivalence of categories. -/
abbrev counitInv (e : C ≌ D) : 𝟭 D ⟶ e.inverse ⋙ e.functor :=
  e.counitIso.inv

/- While these abbreviations are convenient, they also cause some trouble,
preventing structure projections from unfolding. -/

@[simp]
theorem Equivalence_mk'_unit (functor inverse unit_iso counit_iso f) :
    (⟨functor, inverse, unit_iso, counit_iso, f⟩ : C ≌ D).unit = unit_iso.hom :=
  rfl


@[simp]
theorem Equivalence_mk'_counit (functor inverse unit_iso counit_iso f) :
    (⟨functor, inverse, unit_iso, counit_iso, f⟩ : C ≌ D).counit = counit_iso.hom :=
  rfl


@[simp]
theorem Equivalence_mk'_unitInv (functor inverse unit_iso counit_iso f) :
    (⟨functor, inverse, unit_iso, counit_iso, f⟩ : C ≌ D).unitInv = unit_iso.inv :=
  rfl


@[simp]
theorem Equivalence_mk'_counitInv (functor inverse unit_iso counit_iso f) :
    (⟨functor, inverse, unit_iso, counit_iso, f⟩ : C ≌ D).counitInv = counit_iso.inv :=
  rfl


@[reassoc (attr := simp)]
theorem functor_unit_comp (e : C ≌ D) (X : C) :
    e.functor.map (e.unit.app X) ≫ e.counit.app (e.functor.obj X) = 𝟙 (e.functor.obj X) :=
  e.functor_unitIso_comp X


@[reassoc (attr := simp)]
theorem counitInv_functor_comp (e : C ≌ D) (X : C) :
    e.counitInv.app (e.functor.obj X) ≫ e.functor.map (e.unitInv.app X) = 𝟙 (e.functor.obj X) := by
  erw [Iso.inv_eq_inv (e.functor.mapIso (e.unitIso.app X) ≪≫ e.counitIso.app (e.functor.obj X))
      (Iso.refl _)]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    e : CategoryTheory.Equivalence C D
    X : C
    ⊢ Eq ((e.functor.mapIso (e.unitIso.app X)).trans (e.counitIso.app (e.functor.o …
  -/
  exact e.functor_unit_comp X
  /-
    🎉 no goals
  -/


theorem counitInv_app_functor (e : C ≌ D) (X : C) :
    e.counitInv.app (e.functor.obj X) = e.functor.map (e.unit.app X) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    e : CategoryTheory.Equivalence C D
    X : C
    ⊢ Eq (e.counitInv.app (e.functor.obj X)) (e.functor.map (e.unit.app X))
  -/
  symm
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    e : CategoryTheory.Equivalence C D
    X : C
    ⊢ Eq (e.functor.map (e.unit.app X)) (e.counitInv.app (e.functor.obj X))
  -/
  erw [← Iso.comp_hom_eq_id (e.counitIso.app _), functor_unit_comp]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    e : CategoryTheory.Equivalence C D
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.id (e.functor.obj X)) (CategoryTheory.Cate …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem counit_app_functor (e : C ≌ D) (X : C) :
    e.counit.app (e.functor.obj X) = e.functor.map (e.unitInv.app X) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    e : CategoryTheory.Equivalence C D
    X : C
    ⊢ Eq (e.counit.app (e.functor.obj X)) (e.functor.map (e.unitInv.app X))
  -/
  erw [← Iso.hom_comp_eq_id (e.functor.mapIso (e.unitIso.app X)), functor_unit_comp]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    e : CategoryTheory.Equivalence C D
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.id (e.functor.obj X)) (CategoryTheory.Cate …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The other triangle equality. The proof follows the following proof in Globular:
  http://globular.science/1905.001 -/
@[reassoc (attr := simp)]
theorem unit_inverse_comp (e : C ≌ D) (Y : D) :
    e.unit.app (e.inverse.obj Y) ≫ e.inverse.map (e.counit.app Y) = 𝟙 (e.inverse.obj Y) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    e : CategoryTheory.Equivalence C D
    Y : D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (e.unit.app (e.inverse.obj Y)) (e.inv …
  -/
  rw [← id_comp (e.inverse.map _), ← map_id e.inverse, ← counitInv_functor_comp, map_comp]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    e : CategoryTheory.Equivalence C D
    Y : D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (e.unit.app (e.inverse.obj Y)) (Categ …
  -/
  dsimp
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    e : CategoryTheory.Equivalence C D
    Y : D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (e.unit.app (e.inverse.obj Y)) (Categ …
  -/
  rw [← Iso.hom_inv_id_assoc (e.unitIso.app _) (e.inverse.map (e.functor.map _)), app_hom, app_inv]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    e : CategoryTheory.Equivalence C D
    Y : D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (e.unit.app (e.inverse.obj Y)) (Categ …
  -/
  slice_lhs 2 3 => erw [e.unit.naturality]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    e : CategoryTheory.Equivalence C D
    Y : D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (e.unit.app (e.inverse.obj Y)) (Categ …
  -/
  slice_lhs 1 2 => erw [e.unit.naturality]
  slice_lhs 4 4 =>
    rw [← Iso.hom_inv_id_assoc (e.inverse.mapIso (e.counitIso.app _)) (e.unitInv.app _)]
  slice_lhs 3 4 =>
    erw [← map_comp e.inverse, e.counit.naturality]
    erw [(e.counitIso.app _).hom_inv_id, map_id]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    e : CategoryTheory.Equivalence C D
    Y : D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (e.unit.app ((CategoryTheory.Functor. …
  -/
  erw [id_comp]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    e : CategoryTheory.Equivalence C D
    Y : D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (e.unit.app ((CategoryTheory.Functor. …
  -/
  slice_lhs 2 3 => erw [← map_comp e.inverse, e.counitIso.inv.naturality, map_comp]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    e : CategoryTheory.Equivalence C D
    Y : D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (e.unit.app ((CategoryTheory.Functor. …
  -/
  slice_lhs 3 4 => erw [e.unitInv.naturality]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    e : CategoryTheory.Equivalence C D
    Y : D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (e.unit.app ((CategoryTheory.Functor. …
  -/
  slice_lhs 4 5 => erw [← map_comp (e.functor ⋙ e.inverse), (e.unitIso.app _).hom_inv_id, map_id]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    e : CategoryTheory.Equivalence C D
    Y : D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (e.unit.app ((CategoryTheory.Functor. …
  -/
  erw [id_comp]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    e : CategoryTheory.Equivalence C D
    Y : D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (e.unit.app ((CategoryTheory.Functor. …
  -/
  slice_lhs 3 4 => erw [← e.unitInv.naturality]
  slice_lhs 2 3 =>
    erw [← map_comp e.inverse, ← e.counitIso.inv.naturality, (e.counitIso.app _).hom_inv_id,
      map_id]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    e : CategoryTheory.Equivalence C D
    Y : D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (e.unit.app ((CategoryTheory.Functor. …
  -/
  erw [id_comp, (e.unitIso.app _).hom_inv_id]; rfl
                                               /-
                                                 🎉 no goals
                                               -/


@[reassoc (attr := simp)]
theorem inverse_counitInv_comp (e : C ≌ D) (Y : D) :
    e.inverse.map (e.counitInv.app Y) ≫ e.unitInv.app (e.inverse.obj Y) = 𝟙 (e.inverse.obj Y) := by
  erw [Iso.inv_eq_inv (e.unitIso.app (e.inverse.obj Y) ≪≫ e.inverse.mapIso (e.counitIso.app Y))
      (Iso.refl _)]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    e : CategoryTheory.Equivalence C D
    Y : D
    ⊢ Eq ((e.unitIso.app (e.inverse.obj Y)).trans (e.inverse.mapIso (e.counitIso.a …
  -/
  exact e.unit_inverse_comp Y
  /-
    🎉 no goals
  -/


theorem unit_app_inverse (e : C ≌ D) (Y : D) :
    e.unit.app (e.inverse.obj Y) = e.inverse.map (e.counitInv.app Y) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    e : CategoryTheory.Equivalence C D
    Y : D
    ⊢ Eq (e.unit.app (e.inverse.obj Y)) (e.inverse.map (e.counitInv.app Y))
  -/
  erw [← Iso.comp_hom_eq_id (e.inverse.mapIso (e.counitIso.app Y)), unit_inverse_comp]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    e : CategoryTheory.Equivalence C D
    Y : D
    ⊢ Eq (CategoryTheory.CategoryStruct.id (e.inverse.obj Y)) (CategoryTheory.Cate …
  -/
  dsimp
  /-
    🎉 no goals
  -/


theorem unitInv_app_inverse (e : C ≌ D) (Y : D) :
    e.unitInv.app (e.inverse.obj Y) = e.inverse.map (e.counit.app Y) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    e : CategoryTheory.Equivalence C D
    Y : D
    ⊢ Eq (e.unitInv.app (e.inverse.obj Y)) (e.inverse.map (e.counit.app Y))
  -/
  symm
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    e : CategoryTheory.Equivalence C D
    Y : D
    ⊢ Eq (e.inverse.map (e.counit.app Y)) (e.unitInv.app (e.inverse.obj Y))
  -/
  erw [← Iso.hom_comp_eq_id (e.unitIso.app _), unit_inverse_comp]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    e : CategoryTheory.Equivalence C D
    Y : D
    ⊢ Eq (CategoryTheory.CategoryStruct.id (e.inverse.obj Y)) (CategoryTheory.Cate …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[reassoc, simp]
theorem fun_inv_map (e : C ≌ D) (X Y : D) (f : X ⟶ Y) :
    e.functor.map (e.inverse.map f) = e.counit.app X ≫ f ≫ e.counitInv.app Y :=
  (NatIso.naturality_2 e.counitIso f).symm


@[reassoc, simp]
theorem inv_fun_map (e : C ≌ D) (X Y : C) (f : X ⟶ Y) :
    e.inverse.map (e.functor.map f) = e.unitInv.app X ≫ f ≫ e.unit.app Y :=
  (NatIso.naturality_1 e.unitIso f).symm


/-- If `η : 𝟭 C ≅ F ⋙ G` is part of a (not necessarily half-adjoint) equivalence, we can upgrade it
to a refined natural isomorphism `adjointifyη η : 𝟭 C ≅ F ⋙ G` which exhibits the properties
required for a half-adjoint equivalence. See `Equivalence.mk`. -/
def adjointifyη : 𝟭 C ≅ F ⋙ G := by
  calc
    𝟭 C ≅ F ⋙ G := η
    _ ≅ F ⋙ 𝟭 D ⋙ G := isoWhiskerLeft F (leftUnitor G).symm
    _ ≅ F ⋙ (G ⋙ F) ⋙ G := isoWhiskerLeft F (isoWhiskerRight ε.symm G)
    _ ≅ F ⋙ G ⋙ F ⋙ G := isoWhiskerLeft F (associator G F G)
    _ ≅ (F ⋙ G) ⋙ F ⋙ G := (associator F G (F ⋙ G)).symm
    _ ≅ 𝟭 C ⋙ F ⋙ G := isoWhiskerRight η.symm (F ⋙ G)
    _ ≅ F ⋙ G := leftUnitor (F ⋙ G)


@[reassoc]
theorem adjointify_η_ε (X : C) :
    F.map ((adjointifyη η ε).hom.app X) ≫ ε.hom.app (F.obj X) = 𝟙 (F.obj X) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    η : CategoryTheory.Iso (CategoryTheory.Functor.id C) (F.comp G)
    ε : CategoryTheory.Iso (G.comp F) (CategoryTheory.Functor.id D)
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map ((CategoryTheory.Equivalence.a …
  -/
  dsimp [adjointifyη,Trans.trans]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    η : CategoryTheory.Iso (CategoryTheory.Functor.id C) (F.comp G)
    ε : CategoryTheory.Iso (G.comp F) (CategoryTheory.Functor.id D)
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.CategoryStruct …
  -/
  simp only [comp_id, assoc, map_comp]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    η : CategoryTheory.Iso (CategoryTheory.Functor.id C) (F.comp G)
    ε : CategoryTheory.Iso (G.comp F) (CategoryTheory.Functor.id D)
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (η.hom.app X)) (CategoryTheory …
  -/
  have := ε.hom.naturality (F.map (η.inv.app X)); dsimp at this; rw [this]; clear this
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    η : CategoryTheory.Iso (CategoryTheory.Functor.id C) (F.comp G)
    ε : CategoryTheory.Iso (G.comp F) (CategoryTheory.Functor.id D)
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (η.hom.app X)) (CategoryTheory …
  -/
  rw [← assoc _ _ (F.map _)]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    η : CategoryTheory.Iso (CategoryTheory.Functor.id C) (F.comp G)
    ε : CategoryTheory.Iso (G.comp F) (CategoryTheory.Functor.id D)
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (η.hom.app X)) (CategoryTheory …
  -/
  have := ε.hom.naturality (ε.inv.app <| F.obj X); dsimp at this; rw [this]; clear this
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    η : CategoryTheory.Iso (CategoryTheory.Functor.id C) (F.comp G)
    ε : CategoryTheory.Iso (G.comp F) (CategoryTheory.Functor.id D)
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (η.hom.app X)) (CategoryTheory …
  -/
  have := (ε.app <| F.obj X).hom_inv_id; dsimp at this; rw [this]; clear this
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    η : CategoryTheory.Iso (CategoryTheory.Functor.id C) (F.comp G)
    ε : CategoryTheory.Iso (G.comp F) (CategoryTheory.Functor.id D)
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (η.hom.app X)) (CategoryTheory …
  -/
  rw [id_comp]; have := (F.mapIso <| η.app X).hom_inv_id; dsimp at this; rw [this]
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


/-- Every equivalence of categories consisting of functors `F` and `G` such that `F ⋙ G` and
    `G ⋙ F` are naturally isomorphic to identity functors can be transformed into a half-adjoint
    equivalence without changing `F` or `G`. -/
protected def mk (F : C ⥤ D) (G : D ⥤ C) (η : 𝟭 C ≅ F ⋙ G) (ε : G ⋙ F ≅ 𝟭 D) : C ≌ D :=
  ⟨F, G, adjointifyη η ε, ε, adjointify_η_ε η ε⟩


/-- Equivalence of categories is reflexive. -/
@[refl, simps]
def refl : C ≌ C :=
  ⟨𝟭 C, 𝟭 C, Iso.refl _, Iso.refl _, fun _ => Category.id_comp _⟩


instance : Inhabited (C ≌ C) :=
  ⟨refl⟩


/-- Equivalence of categories is symmetric. -/
@[symm, simps]
def symm (e : C ≌ D) : D ≌ C :=
  ⟨e.inverse, e.functor, e.counitIso.symm, e.unitIso.symm, e.inverse_counitInv_comp⟩


/-- Equivalence of categories is transitive. -/
@[trans, simps]
def trans (e : C ≌ D) (f : D ≌ E) : C ≌ E where
  functor := e.functor ⋙ f.functor
  inverse := f.inverse ⋙ e.inverse
  unitIso := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      e : CategoryTheory.Equivalence C D
      f : CategoryTheory.Equivalence D E
      ⊢ CategoryTheory.Iso (CategoryTheory.Functor.id C) ((e.functor.comp f.functor) …
    -/
    refine Iso.trans e.unitIso ?_
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      e : CategoryTheory.Equivalence C D
      f : CategoryTheory.Equivalence D E
      ⊢ CategoryTheory.Iso (e.functor.comp e.inverse) ((e.functor.comp f.functor).co …
    -/
    exact isoWhiskerLeft e.functor (isoWhiskerRight f.unitIso e.inverse)
    /-
      🎉 no goals
    -/
  counitIso := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      e : CategoryTheory.Equivalence C D
      f : CategoryTheory.Equivalence D E
      ⊢ CategoryTheory.Iso ((f.inverse.comp e.inverse).comp (e.functor.comp f.functo …
    -/
    refine Iso.trans ?_ f.counitIso
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      e : CategoryTheory.Equivalence C D
      f : CategoryTheory.Equivalence D E
      ⊢ CategoryTheory.Iso ((f.inverse.comp e.inverse).comp (e.functor.comp f.functo …
    -/
    exact isoWhiskerLeft f.inverse (isoWhiskerRight e.counitIso f.functor)
    /-
      🎉 no goals
    -/
  -- We wouldn't have needed to give this proof if we'd used `Equivalence.mk`,
  -- but we choose to avoid using that here, for the sake of good structure projection `simp`
  -- lemmas.
  functor_unitIso_comp X := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      e : CategoryTheory.Equivalence C D
      f : CategoryTheory.Equivalence D E
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((e.functor.comp f.functor).map ((e.u …
    -/
    dsimp
    rw [← f.functor.map_comp_assoc, e.functor.map_comp, ← counitInv_app_functor, fun_inv_map,
      Iso.inv_hom_id_app_assoc, assoc, Iso.inv_hom_id_app, counit_app_functor, ← Functor.map_comp]
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      e : CategoryTheory.Equivalence C D
      f : CategoryTheory.Equivalence D E
      X : C
      ⊢ Eq (f.functor.map (CategoryTheory.CategoryStruct.comp (CategoryTheory.Catego …
    -/
    erw [comp_id, Iso.hom_inv_id_app, Functor.map_id]
    /-
      🎉 no goals
    -/


/-- Composing a functor with both functors of an equivalence yields a naturally isomorphic
functor. -/
def funInvIdAssoc (e : C ≌ D) (F : C ⥤ E) : e.functor ⋙ e.inverse ⋙ F ≅ F :=
  (Functor.associator _ _ _).symm ≪≫ isoWhiskerRight e.unitIso.symm F ≪≫ F.leftUnitor


@[simp]
theorem funInvIdAssoc_hom_app (e : C ≌ D) (F : C ⥤ E) (X : C) :
    (funInvIdAssoc e F).hom.app X = F.map (e.unitInv.app X) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    e : CategoryTheory.Equivalence C D
    F : CategoryTheory.Functor C E
    X : C
    ⊢ Eq ((e.funInvIdAssoc F).hom.app X) (F.map (e.unitInv.app X))
  -/
  dsimp [funInvIdAssoc]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    e : CategoryTheory.Equivalence C D
    F : CategoryTheory.Functor C E
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (F. …
  -/
  aesop_cat
  /-
    🎉 no goals
  -/


@[simp]
theorem funInvIdAssoc_inv_app (e : C ≌ D) (F : C ⥤ E) (X : C) :
    (funInvIdAssoc e F).inv.app X = F.map (e.unit.app X) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    e : CategoryTheory.Equivalence C D
    F : CategoryTheory.Functor C E
    X : C
    ⊢ Eq ((e.funInvIdAssoc F).inv.app X) (F.map (e.unit.app X))
  -/
  dsimp [funInvIdAssoc]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    e : CategoryTheory.Equivalence C D
    F : CategoryTheory.Functor C E
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  aesop_cat
  /-
    🎉 no goals
  -/


/-- Composing a functor with both functors of an equivalence yields a naturally isomorphic
functor. -/
def invFunIdAssoc (e : C ≌ D) (F : D ⥤ E) : e.inverse ⋙ e.functor ⋙ F ≅ F :=
  (Functor.associator _ _ _).symm ≪≫ isoWhiskerRight e.counitIso F ≪≫ F.leftUnitor


@[simp]
theorem invFunIdAssoc_hom_app (e : C ≌ D) (F : D ⥤ E) (X : D) :
    (invFunIdAssoc e F).hom.app X = F.map (e.counit.app X) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    e : CategoryTheory.Equivalence C D
    F : CategoryTheory.Functor D E
    X : D
    ⊢ Eq ((e.invFunIdAssoc F).hom.app X) (F.map (e.counit.app X))
  -/
  dsimp [invFunIdAssoc]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    e : CategoryTheory.Equivalence C D
    F : CategoryTheory.Functor D E
    X : D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (F. …
  -/
  aesop_cat
  /-
    🎉 no goals
  -/


@[simp]
theorem invFunIdAssoc_inv_app (e : C ≌ D) (F : D ⥤ E) (X : D) :
    (invFunIdAssoc e F).inv.app X = F.map (e.counitInv.app X) := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    e : CategoryTheory.Equivalence C D
    F : CategoryTheory.Functor D E
    X : D
    ⊢ Eq ((e.invFunIdAssoc F).inv.app X) (F.map (e.counitInv.app X))
  -/
  dsimp [invFunIdAssoc]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u₃
    inst✝ : CategoryTheory.Category.{v₃, u₃} E
    e : CategoryTheory.Equivalence C D
    F : CategoryTheory.Functor D E
    X : D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  aesop_cat
  /-
    🎉 no goals
  -/


/-- If `C` is equivalent to `D`, then `C ⥤ E` is equivalent to `D ⥤ E`. -/
@[simps! functor inverse unitIso counitIso]
def congrLeft (e : C ≌ D) : C ⥤ E ≌ D ⥤ E where
  functor := (whiskeringLeft _ _ _).obj e.inverse
  inverse := (whiskeringLeft _ _ _).obj e.functor
              /-
                C : Type u₁
                inst✝² : CategoryTheory.Category.{v₁, u₁} C
                D : Type u₂
                inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                E : Type u₃
                inst✝ : CategoryTheory.Category.{v₃, u₃} E
                e : CategoryTheory.Equivalence C D
                ⊢ ∀ {X Y : CategoryTheory.Functor C E} (f : Quiver.Hom X Y), Eq (CategoryTheor …
              -/
  unitIso := (NatIso.ofComponents fun F => (e.funInvIdAssoc F).symm)
              /-
                🎉 no goals
              -/
                /-
                  C : Type u₁
                  inst✝² : CategoryTheory.Category.{v₁, u₁} C
                  D : Type u₂
                  inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                  E : Type u₃
                  inst✝ : CategoryTheory.Category.{v₃, u₃} E
                  e : CategoryTheory.Equivalence C D
                  ⊢ ∀ {X Y : CategoryTheory.Functor D E} (f : Quiver.Hom X Y), Eq (CategoryTheor …
                -/
  counitIso := (NatIso.ofComponents fun F => e.invFunIdAssoc F)
                /-
                  🎉 no goals
                -/
  functor_unitIso_comp F := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      e : CategoryTheory.Equivalence C D
      F : CategoryTheory.Functor C E
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.whiskeringLeft D C  …
    -/
    ext X
    /-
      case w.h
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      e : CategoryTheory.Equivalence C D
      F : CategoryTheory.Functor C E
      X : D
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (((CategoryTheory.whiskeringLeft D C …
    -/
    dsimp
    simp only [funInvIdAssoc_inv_app, id_obj, comp_obj, invFunIdAssoc_hom_app,
      Functor.comp_map, ← F.map_comp, unit_inverse_comp, map_id]


/-- If `C` is equivalent to `D`, then `E ⥤ C` is equivalent to `E ⥤ D`. -/
@[simps! functor inverse unitIso counitIso]
def congrRight (e : C ≌ D) : E ⥤ C ≌ E ⥤ D where
  functor := (whiskeringRight _ _ _).obj e.functor
  inverse := (whiskeringRight _ _ _).obj e.inverse
             /-
               C : Type u₁
               inst✝² : CategoryTheory.Category.{v₁, u₁} C
               D : Type u₂
               inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
               E : Type u₃
               inst✝ : CategoryTheory.Category.{v₃, u₃} E
               e : CategoryTheory.Equivalence C D
               ⊢ ∀ {X Y : CategoryTheory.Functor E C} (f : Quiver.Hom X Y), Eq (CategoryTheor …
             -/
  unitIso := NatIso.ofComponents
             /-
               🎉 no goals
             -/
      fun F => F.rightUnitor.symm ≪≫ isoWhiskerLeft F e.unitIso ≪≫ Functor.associator _ _ _
               /-
                 C : Type u₁
                 inst✝² : CategoryTheory.Category.{v₁, u₁} C
                 D : Type u₂
                 inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                 E : Type u₃
                 inst✝ : CategoryTheory.Category.{v₃, u₃} E
                 e : CategoryTheory.Equivalence C D
                 ⊢ ∀ {X Y : CategoryTheory.Functor E D} (f : Quiver.Hom X Y), Eq (CategoryTheor …
               -/
  counitIso := NatIso.ofComponents
               /-
                 🎉 no goals
               -/
      fun F => Functor.associator _ _ _ ≪≫ isoWhiskerLeft F e.counitIso ≪≫ F.rightUnitor


@[simp]
theorem cancel_unit_right {X Y : C} (f f' : X ⟶ Y) :
                                                        /-
                                                          C : Type u₁
                                                          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                          D : Type u₂
                                                          inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                          e : CategoryTheory.Equivalence C D
                                                          X Y : C
                                                          f f' : Quiver.Hom X Y
                                                          ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp f (e.unit.app Y)) (CategoryTheor …
                                                        -/
    f ≫ e.unit.app Y = f' ≫ e.unit.app Y ↔ f = f' := by simp only [cancel_mono]
                                                        /-
                                                          🎉 no goals
                                                        -/


@[simp]
theorem cancel_unitInv_right {X Y : C} (f f' : X ⟶ e.inverse.obj (e.functor.obj Y)) :
                                                              /-
                                                                C : Type u₁
                                                                inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                                D : Type u₂
                                                                inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                                e : CategoryTheory.Equivalence C D
                                                                X Y : C
                                                                f f' : Quiver.Hom X (e.inverse.obj (e.functor.obj Y))
                                                                ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp f (e.unitInv.app Y)) (CategoryTh …
                                                              -/
    f ≫ e.unitInv.app Y = f' ≫ e.unitInv.app Y ↔ f = f' := by simp only [cancel_mono]
                                                              /-
                                                                🎉 no goals
                                                              -/


@[simp]
theorem cancel_counit_right {X Y : D} (f f' : X ⟶ e.functor.obj (e.inverse.obj Y)) :
                                                            /-
                                                              C : Type u₁
                                                              inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                              D : Type u₂
                                                              inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                              e : CategoryTheory.Equivalence C D
                                                              X Y : D
                                                              f f' : Quiver.Hom X (e.functor.obj (e.inverse.obj Y))
                                                              ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp f (e.counit.app Y)) (CategoryThe …
                                                            -/
    f ≫ e.counit.app Y = f' ≫ e.counit.app Y ↔ f = f' := by simp only [cancel_mono]
                                                            /-
                                                              🎉 no goals
                                                            -/


@[simp]
theorem cancel_counitInv_right {X Y : D} (f f' : X ⟶ Y) :
                                                                  /-
                                                                    C : Type u₁
                                                                    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                                    D : Type u₂
                                                                    inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                                    e : CategoryTheory.Equivalence C D
                                                                    X Y : D
                                                                    f f' : Quiver.Hom X Y
                                                                    ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp f (e.counitInv.app Y)) (Category …
                                                                  -/
    f ≫ e.counitInv.app Y = f' ≫ e.counitInv.app Y ↔ f = f' := by simp only [cancel_mono]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


@[simp]
theorem cancel_unit_right_assoc {W X X' Y : C} (f : W ⟶ X) (g : X ⟶ Y) (f' : W ⟶ X') (g' : X' ⟶ Y) :
    f ≫ g ≫ e.unit.app Y = f' ≫ g' ≫ e.unit.app Y ↔ f ≫ g = f' ≫ g' := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    e : CategoryTheory.Equivalence C D
    W X X' Y : C
    f : Quiver.Hom W X
    g : Quiver.Hom X Y
    f' : Quiver.Hom W X'
    g' : Quiver.Hom X' Y
    ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct …
  -/
  simp only [← Category.assoc, cancel_mono]
  /-
    🎉 no goals
  -/


@[simp]
theorem cancel_counitInv_right_assoc {W X X' Y : D} (f : W ⟶ X) (g : X ⟶ Y) (f' : W ⟶ X')
    (g' : X' ⟶ Y) : f ≫ g ≫ e.counitInv.app Y = f' ≫ g' ≫ e.counitInv.app Y ↔ f ≫ g = f' ≫ g' := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    e : CategoryTheory.Equivalence C D
    W X X' Y : D
    f : Quiver.Hom W X
    g : Quiver.Hom X Y
    f' : Quiver.Hom W X'
    g' : Quiver.Hom X' Y
    ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct …
  -/
  simp only [← Category.assoc, cancel_mono]
  /-
    🎉 no goals
  -/


@[simp]
theorem cancel_unit_right_assoc' {W X X' Y Y' Z : C} (f : W ⟶ X) (g : X ⟶ Y) (h : Y ⟶ Z)
    (f' : W ⟶ X') (g' : X' ⟶ Y') (h' : Y' ⟶ Z) :
    f ≫ g ≫ h ≫ e.unit.app Z = f' ≫ g' ≫ h' ≫ e.unit.app Z ↔ f ≫ g ≫ h = f' ≫ g' ≫ h' := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝ : CategoryTheory.Category.{v₂, u₂} D
    e : CategoryTheory.Equivalence C D
    W X X' Y Y' Z : C
    f : Quiver.Hom W X
    g : Quiver.Hom X Y
    h : Quiver.Hom Y Z
    f' : Quiver.Hom W X'
    g' : Quiver.Hom X' Y'
    h' : Quiver.Hom Y' Z
    ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct …
  -/
  simp only [← Category.assoc, cancel_mono]
  /-
    🎉 no goals
  -/


@[simp]
theorem cancel_counitInv_right_assoc' {W X X' Y Y' Z : D} (f : W ⟶ X) (g : X ⟶ Y) (h : Y ⟶ Z)
    (f' : W ⟶ X') (g' : X' ⟶ Y') (h' : Y' ⟶ Z) :
    f ≫ g ≫ h ≫ e.counitInv.app Z = f' ≫ g' ≫ h' ≫ e.counitInv.app Z ↔
                                   /-
                                     C : Type u₁
                                     inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                     D : Type u₂
                                     inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                     e : CategoryTheory.Equivalence C D
                                     W X X' Y Y' Z : D
                                     f : Quiver.Hom W X
                                     g : Quiver.Hom X Y
                                     h : Quiver.Hom Y Z
                                     f' : Quiver.Hom W X'
                                     g' : Quiver.Hom X' Y'
                                     h' : Quiver.Hom Y' Z
                                     ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct …
                                   -/
    f ≫ g ≫ h = f' ≫ g' ≫ h' := by simp only [← Category.assoc, cancel_mono]
                                   /-
                                     🎉 no goals
                                   -/


/-- Natural number powers of an auto-equivalence.  Use `(^)` instead. -/
def powNat (e : C ≌ C) : ℕ → (C ≌ C)
  | 0 => Equivalence.refl
  | 1 => e
  | n + 2 => e.trans (powNat e (n + 1))


/-- Powers of an auto-equivalence.  Use `(^)` instead. -/
def pow (e : C ≌ C) : ℤ → (C ≌ C)
  | Int.ofNat n => e.powNat n
  | Int.negSucc n => e.symm.powNat (n + 1)


instance : Pow (C ≌ C) ℤ :=
  ⟨pow⟩


@[simp]
theorem pow_zero (e : C ≌ C) : e ^ (0 : ℤ) = Equivalence.refl :=
  rfl


@[simp]
theorem pow_one (e : C ≌ C) : e ^ (1 : ℤ) = e :=
  rfl


@[simp]
theorem pow_neg_one (e : C ≌ C) : e ^ (-1 : ℤ) = e.symm :=
  rfl

-- TODO as necessary, add the natural isomorphisms `(e^a).trans e^b ≅ e^(a+b)`.
-- At this point, we haven't even defined the category of equivalences.
-- Note: the better formulation of this would involve `HasShift`.

/-- The functor of an equivalence of categories is essentially surjective.

See <https://stacks.math.columbia.edu/tag/02C3>.
-/
instance essSurj_functor (e : C ≌ E) : e.functor.EssSurj :=
  ⟨fun Y => ⟨e.inverse.obj Y, ⟨e.counitIso.app Y⟩⟩⟩


instance essSurj_inverse (e : C ≌ E) : e.inverse.EssSurj :=
  e.symm.essSurj_functor


/-- The functor of an equivalence of categories is fully faithful. -/
def fullyFaithfulFunctor (e : C ≌ E) : e.functor.FullyFaithful where
  preimage {X Y} f := e.unitIso.hom.app X ≫ e.inverse.map f ≫ e.unitIso.inv.app Y


/-- The inverse of an equivalence of categories is fully faithful. -/
def fullyFaithfulInverse (e : C ≌ E) : e.inverse.FullyFaithful where
  preimage {X Y} f := e.counitIso.inv.app X ≫ e.functor.map f ≫ e.counitIso.hom.app Y


/-- The functor of an equivalence of categories is faithful.

See <https://stacks.math.columbia.edu/tag/02C3>.
-/
instance faithful_functor (e : C ≌ E) : e.functor.Faithful :=
  e.fullyFaithfulFunctor.faithful


instance faithful_inverse (e : C ≌ E) : e.inverse.Faithful :=
  e.fullyFaithfulInverse.faithful


/-- The functor of an equivalence of categories is full.

See <https://stacks.math.columbia.edu/tag/02C3>.
-/
instance full_functor (e : C ≌ E) : e.functor.Full :=
  e.fullyFaithfulFunctor.full


instance full_inverse (e : C ≌ E) : e.inverse.Full :=
  e.fullyFaithfulInverse.full


/-- If `e : C ≌ D` is an equivalence of categories, and `iso : e.functor ≅ G` is
an isomorphism, then there is an equivalence of categories whose functor is `G`. -/
@[simps!]
def changeFunctor (e : C ≌ D) {G : C ⥤ D} (iso : e.functor ≅ G) : C ≌ D where
  functor := G
  inverse := e.inverse
  unitIso := e.unitIso ≪≫ isoWhiskerRight iso _
  counitIso := isoWhiskerLeft _ iso.symm ≪≫ e.counitIso


/-- Compatibility of `changeFunctor` with identity isomorphisms of functors -/
                                                                                /-
                                                                                  C : Type u₁
                                                                                  inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                                                  D : Type u₂
                                                                                  inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                                                  e : CategoryTheory.Equivalence C D
                                                                                  ⊢ Eq (e.changeFunctor (CategoryTheory.Iso.refl e.functor)) e
                                                                                -/
theorem changeFunctor_refl (e : C ≌ D) : e.changeFunctor (Iso.refl _) = e := by aesop_cat
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


/-- Compatibility of `changeFunctor` with the composition of isomorphisms of functors -/
theorem changeFunctor_trans (e : C ≌ D) {G G' : C ⥤ D} (iso₁ : e.functor ≅ G) (iso₂ : G ≅ G') :
                                                                                     /-
                                                                                       C : Type u₁
                                                                                       inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                                                       D : Type u₂
                                                                                       inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                                                                       e : CategoryTheory.Equivalence C D
                                                                                       G G' : CategoryTheory.Functor C D
                                                                                       iso₁ : CategoryTheory.Iso e.functor G
                                                                                       iso₂ : CategoryTheory.Iso G G'
                                                                                       ⊢ Eq ((e.changeFunctor iso₁).changeFunctor iso₂) (e.changeFunctor (iso₁.trans  …
                                                                                     -/
    (e.changeFunctor iso₁).changeFunctor iso₂ = e.changeFunctor (iso₁ ≪≫ iso₂) := by aesop_cat
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


/-- If `e : C ≌ D` is an equivalence of categories, and `iso : e.functor ≅ G` is
an isomorphism, then there is an equivalence of categories whose inverse is `G`. -/
@[simps!]
def changeInverse (e : C ≌ D) {G : D ⥤ C} (iso : e.inverse ≅ G) : C ≌ D where
  functor := e.functor
  inverse := G
  unitIso := e.unitIso ≪≫ isoWhiskerLeft _ iso
  counitIso := isoWhiskerRight iso.symm _ ≪≫ e.counitIso
  functor_unitIso_comp X := by
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      e : CategoryTheory.Equivalence C D
      G : CategoryTheory.Functor D C
      iso : CategoryTheory.Iso e.inverse G
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (e.functor.map ((e.unitIso.trans (Cat …
    -/
    dsimp
    /-
      C : Type u₁
      inst✝² : CategoryTheory.Category.{v₁, u₁} C
      D : Type u₂
      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
      E : Type u₃
      inst✝ : CategoryTheory.Category.{v₃, u₃} E
      e : CategoryTheory.Equivalence C D
      G : CategoryTheory.Functor D C
      iso : CategoryTheory.Iso e.inverse G
      X : C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (e.functor.map (CategoryTheory.Catego …
    -/
    rw [← map_comp_assoc, assoc, iso.hom_inv_id_app, comp_id, functor_unit_comp]
    /-
      🎉 no goals
    -/


/-- A functor is an equivalence of categories if it is faithful, full and
essentially surjective. -/
class Functor.IsEquivalence (F : C ⥤ D) : Prop where
  faithful : F.Faithful := by infer_instance
  full : F.Full := by infer_instance
  essSurj : F.EssSurj := by infer_instance


instance Equivalence.isEquivalence_functor (F : C ≌ D) : IsEquivalence F.functor where


instance Equivalence.isEquivalence_inverse (F : C ≌ D) : IsEquivalence F.inverse :=
  F.symm.isEquivalence_functor


/-- To see that a functor is an equivalence, it suffices to provide an inverse functor `G` such that
    `F ⋙ G` and `G ⋙ F` are naturally isomorphic to identity functors. -/
protected lemma mk' {F : C ⥤ D} (G : D ⥤ C) (η : 𝟭 C ≅ F ⋙ G) (ε : G ⋙ F ≅ 𝟭 D) :
    IsEquivalence F :=
  inferInstanceAs (IsEquivalence (Equivalence.mk F G η ε).functor)


/-- A quasi-inverse `D ⥤ C` to a functor that `F : C ⥤ D` that is an equivalence,
i.e. faithful, full, and essentially surjective. -/
noncomputable def inv (F : C ⥤ D) [F.IsEquivalence] : D ⥤ C where
  obj X := F.objPreimage X
  map {X Y} f := F.preimage ((F.objObjPreimageIso X).hom ≫ f ≫ (F.objObjPreimageIso Y).inv)
                 /-
                   C : Type u₁
                   inst✝² : CategoryTheory.Category.{v₁, u₁} C
                   D : Type u₂
                   inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                   F : CategoryTheory.Functor C D
                   inst✝ : F.IsEquivalence
                   X : D
                   ⊢ Eq ({ obj := fun X => F.objPreimage X, map := fun {X Y} f => F.preimage (Cat …
                 -/
  map_id X := by apply F.map_injective; aesop_cat
                                        /-
                                          🎉 no goals
                                        -/
                             /-
                               C : Type u₁
                               inst✝² : CategoryTheory.Category.{v₁, u₁} C
                               D : Type u₂
                               inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                               F : CategoryTheory.Functor C D
                               inst✝ : F.IsEquivalence
                               X Y Z : D
                               f : Quiver.Hom X Y
                               g : Quiver.Hom Y Z
                               ⊢ Eq ({ obj := fun X => F.objPreimage X, map := fun {X Y} f => F.preimage (Cat …
                             -/
  map_comp {X Y Z} f g := by apply F.map_injective; simp
                                                    /-
                                                      🎉 no goals
                                                    -/


/-- Interpret a functor that is an equivalence as an equivalence.

See <https://stacks.math.columbia.edu/tag/02C3>. -/
@[simps functor]
noncomputable def asEquivalence (F : C ⥤ D) [F.IsEquivalence] : C ≌ D where
  functor := F
  inverse := F.inv
  unitIso := NatIso.ofComponents
    (fun X => (F.preimageIso <| F.objObjPreimageIso <| F.obj X).symm)
                                    /-
                                      C : Type u₁
                                      inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                      D : Type u₂
                                      inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                      F : CategoryTheory.Functor C D
                                      inst✝ : F.IsEquivalence
                                      X✝ Y✝ : C
                                      f : Quiver.Hom X✝ Y✝
                                      ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id C) …
                                    -/
      (fun f => F.map_injective (by simp [inv]))
                                    /-
                                      🎉 no goals
                                    -/
                                                           /-
                                                             C : Type u₁
                                                             inst✝² : CategoryTheory.Category.{v₁, u₁} C
                                                             D : Type u₂
                                                             inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
                                                             F : CategoryTheory.Functor C D
                                                             inst✝ : F.IsEquivalence
                                                             ⊢ ∀ {X Y : D} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.comp ((F …
                                                           -/
  counitIso := NatIso.ofComponents F.objObjPreimageIso (by simp [inv])
                                                           /-
                                                             🎉 no goals
                                                           -/


instance isEquivalence_refl : IsEquivalence (𝟭 C) :=
  Equivalence.refl.isEquivalence_functor


instance isEquivalence_inv (F : C ⥤ D) [IsEquivalence F] : IsEquivalence F.inv :=
  F.asEquivalence.symm.isEquivalence_functor


instance isEquivalence_trans (F : C ⥤ D) (G : D ⥤ E) [IsEquivalence F] [IsEquivalence G] :
    IsEquivalence (F ⋙ G) where


instance (F : C ⥤ D) [IsEquivalence F] : IsEquivalence ((whiskeringLeft C D E).obj F) :=
  (inferInstance : IsEquivalence (Equivalence.congrLeft F.asEquivalence).inverse)


instance (F : C ⥤ D) [IsEquivalence F] : IsEquivalence ((whiskeringRight E C D).obj F) :=
  (inferInstance : IsEquivalence (Equivalence.congrRight F.asEquivalence).functor)


@[simp]
theorem fun_inv_map (F : C ⥤ D) [IsEquivalence F] (X Y : D) (f : X ⟶ Y) :
    F.map (F.inv.map f) = F.asEquivalence.counit.app X ≫ f ≫ F.asEquivalence.counitInv.app Y := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝ : F.IsEquivalence
    X Y : D
    f : Quiver.Hom X Y
    ⊢ Eq (F.map (F.inv.map f)) (CategoryTheory.CategoryStruct.comp (F.asEquivalenc …
  -/
  erw [NatIso.naturality_2]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝ : F.IsEquivalence
    X Y : D
    f : Quiver.Hom X Y
    ⊢ Eq (F.map (F.inv.map f)) ((F.asEquivalence.inverse.comp F.asEquivalence.func …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem inv_fun_map (F : C ⥤ D) [IsEquivalence F] (X Y : C) (f : X ⟶ Y) :
    F.inv.map (F.map f) = F.asEquivalence.unitInv.app X ≫ f ≫ F.asEquivalence.unit.app Y := by
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝ : F.IsEquivalence
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Eq (F.inv.map (F.map f)) (CategoryTheory.CategoryStruct.comp (F.asEquivalenc …
  -/
  erw [NatIso.naturality_1]
  /-
    C : Type u₁
    inst✝² : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝¹ : CategoryTheory.Category.{v₂, u₂} D
    F : CategoryTheory.Functor C D
    inst✝ : F.IsEquivalence
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Eq (F.inv.map (F.map f)) ((F.asEquivalence.functor.comp F.asEquivalence.inve …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma isEquivalence_of_iso {F G : C ⥤ D} (e : F ≅ G) [F.IsEquivalence] : G.IsEquivalence :=
  ((asEquivalence F).changeFunctor e).isEquivalence_functor


lemma isEquivalence_iff_of_iso {F G : C ⥤ D} (e : F ≅ G) :
    F.IsEquivalence ↔ G.IsEquivalence :=
  ⟨fun _ => isEquivalence_of_iso e, fun _ => isEquivalence_of_iso e.symm⟩


/-- If `G` and `F ⋙ G` are equivalence of categories, then `F` is also an equivalence. -/
lemma isEquivalence_of_comp_right {E : Type*} [Category E] (F : C ⥤ D) (G : D ⥤ E)
    [IsEquivalence G] [IsEquivalence (F ⋙ G)] : IsEquivalence F := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D E
    inst✝¹ : G.IsEquivalence
    inst✝ : (F.comp G).IsEquivalence
    ⊢ F.IsEquivalence
  -/
  rw [isEquivalence_iff_of_iso (F.rightUnitor.symm ≪≫ isoWhiskerLeft F (G.asEquivalence.unitIso))]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D E
    inst✝¹ : G.IsEquivalence
    inst✝ : (F.comp G).IsEquivalence
    ⊢ (F.comp (G.asEquivalence.functor.comp G.asEquivalence.inverse)).IsEquivalence
  -/
  exact ((F ⋙ G).asEquivalence.trans G.asEquivalence.symm).isEquivalence_functor
  /-
    🎉 no goals
  -/


/-- If `F` and `F ⋙ G` are equivalence of categories, then `G` is also an equivalence. -/
lemma isEquivalence_of_comp_left {E : Type*} [Category E] (F : C ⥤ D) (G : D ⥤ E)
    [IsEquivalence F] [IsEquivalence (F ⋙ G)] : IsEquivalence G := by
  rw [isEquivalence_iff_of_iso (G.leftUnitor.symm ≪≫
    isoWhiskerRight F.asEquivalence.counitIso.symm G)]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    E : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} E
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D E
    inst✝¹ : F.IsEquivalence
    inst✝ : (F.comp G).IsEquivalence
    ⊢ ((F.asEquivalence.inverse.comp F.asEquivalence.functor).comp G).IsEquivalence
  -/
  exact (F.asEquivalence.symm.trans (F ⋙ G).asEquivalence).isEquivalence_functor
  /-
    🎉 no goals
  -/


instance essSurjInducedFunctor {C' : Type*} (e : C' ≃ D) : (inducedFunctor e).EssSurj where
                                  /-
                                    C : Type u₁
                                    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                    D : Type u₂
                                    inst✝ : CategoryTheory.Category.{v₂, u₂} D
                                    C' : Type u_1
                                    e : Equiv C' D
                                    Y : D
                                    ⊢ Nonempty (CategoryTheory.Iso ((CategoryTheory.inducedFunctor ⇑e).obj (e.symm …
                                  -/
  mem_essImage Y := ⟨e.symm Y, by simpa using ⟨default⟩⟩
                                  /-
                                    🎉 no goals
                                  -/


noncomputable instance inducedFunctorOfEquiv {C' : Type*} (e : C' ≃ D) :
    IsEquivalence (inducedFunctor e) where


noncomputable instance fullyFaithfulToEssImage (F : C ⥤ D) [F.Full] [F.Faithful] :
    IsEquivalence F.toEssImage where


/-- Construct an isomorphism `F ⋙ H.inverse ≅ G` from an isomorphism `F ≅ G ⋙ H.functor`. -/
@[simps!]
def compInverseIso {H : D ≌ E} (i : F ≅ G ⋙ H.functor) : F ⋙ H.inverse ≅ G :=
  isoWhiskerRight i H.inverse ≪≫
    associator G _ H.inverse ≪≫ isoWhiskerLeft G H.unitIso.symm ≪≫ G.rightUnitor


/-- Construct an isomorphism `G ≅ F ⋙ H.inverse` from an isomorphism `G ⋙ H.functor ≅ F`. -/
@[simps!]
def isoCompInverse {H : D ≌ E} (i : G ⋙ H.functor ≅ F) : G ≅ F ⋙ H.inverse :=
  G.rightUnitor.symm ≪≫ isoWhiskerLeft G H.unitIso ≪≫ (associator _ _ _).symm ≪≫
    isoWhiskerRight i H.inverse


/-- Construct an isomorphism `G.inverse ⋙ F ≅ H` from an isomorphism `F ≅ G.functor ⋙ H`. -/
@[simps!]
def inverseCompIso {G : C ≌ D} (i : F ≅ G.functor ⋙ H) : G.inverse ⋙ F ≅ H :=
  isoWhiskerLeft G.inverse i ≪≫ (associator _ _ _).symm ≪≫
    isoWhiskerRight G.counitIso H ≪≫ H.leftUnitor


/-- Construct an isomorphism `H ≅ G.inverse ⋙ F` from an isomorphism `G.functor ⋙ H ≅ F`. -/
@[simps!]
def isoInverseComp {G : C ≌ D} (i : G.functor ⋙ H ≅ F) : H ≅ G.inverse ⋙ F :=
  H.leftUnitor.symm ≪≫ isoWhiskerRight G.counitIso.symm H ≪≫ associator _ _ _
    ≪≫ isoWhiskerLeft G.inverse i


@[deprecated (since := "2024-04-06")] alias IsEquivalence := Functor.IsEquivalence

@[deprecated (since := "2024-04-06")] alias IsEquivalence.fun_inv_map := Functor.fun_inv_map

@[deprecated (since := "2024-04-06")] alias IsEquivalence.inv_fun_map := Functor.inv_fun_map

@[deprecated (since := "2024-04-06")] alias IsEquivalence.ofIso := Equivalence.changeFunctor

@[deprecated (since := "2024-04-06")]
alias IsEquivalence.ofIso_trans := Equivalence.changeFunctor_trans

@[deprecated (since := "2024-04-06")]
alias IsEquivalence.ofIso_refl := Equivalence.changeFunctor_refl

@[deprecated (since := "2024-04-06")]
alias IsEquivalence.equivOfIso := Functor.isEquivalence_iff_of_iso

@[deprecated (since := "2024-04-06")]
alias IsEquivalence.cancelCompRight := Functor.isEquivalence_of_comp_right

@[deprecated (since := "2024-04-06")]
alias IsEquivalence.cancelCompLeft := Functor.isEquivalence_of_comp_left


