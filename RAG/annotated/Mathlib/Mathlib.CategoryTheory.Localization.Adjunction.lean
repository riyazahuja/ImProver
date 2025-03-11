/-- Auxiliary definition of the unit morphism for the adjunction `Adjunction.localization` -/
noncomputable def ε : 𝟭 D₁ ⟶ G' ⋙ F' := by
  letI : Lifting L₁ W₁ ((G ⋙ F) ⋙ L₁) (G' ⋙ F') :=
    Lifting.mk (CatCommSq.hComp G F L₁ L₂ L₁ G' F').iso'.symm
  exact Localization.liftNatTrans L₁ W₁ L₁ ((G ⋙ F) ⋙ L₁) (𝟭 D₁) (G' ⋙ F')
    (whiskerRight adj.unit L₁)


lemma ε_app (X₁ : C₁) :
    (ε adj L₁ W₁ L₂ G' F').app (L₁.obj X₁) =
      L₁.map (adj.unit.app X₁) ≫ (CatCommSq.iso F L₂ L₁ F').hom.app (G.obj X₁) ≫
        F'.map ((CatCommSq.iso G L₁ L₂ G').hom.app X₁) := by
  letI : Lifting L₁ W₁ ((G ⋙ F) ⋙ L₁) (G' ⋙ F') :=
    Lifting.mk (CatCommSq.hComp G F L₁ L₂ L₁ G' F').iso'.symm
  simp only [ε, liftNatTrans_app, Lifting.iso, Iso.symm,
    Functor.id_obj, Functor.comp_obj, Lifting.id_iso', Functor.rightUnitor_hom_app,
      whiskerRight_app, CatCommSq.hComp_iso'_hom_app, id_comp]


/-- Auxiliary definition of the counit morphism for the adjunction `Adjunction.localization` -/
noncomputable def η : F' ⋙ G' ⟶ 𝟭 D₂ := by
  letI : Lifting L₂ W₂ ((F ⋙ G) ⋙ L₂) (F' ⋙ G') :=
    Lifting.mk (CatCommSq.hComp F G L₂ L₁ L₂ F' G').iso'.symm
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    D₁ : Type u_3
    D₂ : Type u_4
    inst✝⁷ : CategoryTheory.Category.{?u.11179, u_1} C₁
    inst✝⁶ : CategoryTheory.Category.{?u.11183, u_2} C₂
    inst✝⁵ : CategoryTheory.Category.{?u.11187, u_3} D₁
    inst✝⁴ : CategoryTheory.Category.{?u.11191, u_4} D₂
    G : CategoryTheory.Functor C₁ C₂
    F : CategoryTheory.Functor C₂ C₁
    adj : CategoryTheory.Adjunction G F
    L₁ : CategoryTheory.Functor C₁ D₁
    W₁ : CategoryTheory.MorphismProperty C₁
    inst✝³ : L₁.IsLocalization W₁
    L₂ : CategoryTheory.Functor C₂ D₂
    W₂ : CategoryTheory.MorphismProperty C₂
    inst✝² : L₂.IsLocalization W₂
    G' : CategoryTheory.Functor D₁ D₂
    F' : CategoryTheory.Functor D₂ D₁
    inst✝¹ : CategoryTheory.CatCommSq G L₁ L₂ G'
    inst✝ : CategoryTheory.CatCommSq F L₂ L₁ F'
    this : CategoryTheory.Localization.Lifting L₂ W₂ ((F.comp G).comp L₂) (F'.comp …
    ⊢ Quiver.Hom (F'.comp G') (CategoryTheory.Functor.id D₂)
  -/
  exact liftNatTrans L₂ W₂ ((F ⋙ G) ⋙ L₂) L₂ (F' ⋙ G') (𝟭 D₂) (whiskerRight adj.counit L₂)
  /-
    🎉 no goals
  -/


lemma η_app (X₂ : C₂) :
    (η adj L₁ L₂ W₂ G' F').app (L₂.obj X₂) =
      G'.map ((CatCommSq.iso F L₂ L₁ F').inv.app X₂) ≫
        (CatCommSq.iso G L₁ L₂ G').inv.app (F.obj X₂) ≫
        L₂.map (adj.counit.app X₂) := by
  letI : Lifting L₂ W₂ ((F ⋙ G) ⋙ L₂) (F' ⋙ G') :=
    Lifting.mk (CatCommSq.hComp F G L₂ L₁ L₂ F' G').iso'.symm
  simp only [η, liftNatTrans_app, Lifting.iso, Iso.symm, CatCommSq.hComp_iso'_inv_app,
    whiskerRight_app, Lifting.id_iso', Functor.rightUnitor_inv_app, comp_id, assoc]


/-- If `adj : G ⊣ F` is an adjunction between two categories `C₁` and `C₂` that
are equipped with localization functors `L₁ : C₁ ⥤ D₁` and `L₂ : C₂ ⥤ D₂` with
respect to `W₁ : MorphismProperty C₁` and `W₂ : MorphismProperty C₂`, and that
the functors `F : C₂ ⥤ C₁` and `G : C₁ ⥤ C₂` induce functors `F' : D₂ ⥤ D₁`
and `G' : D₁ ⥤ D₂` on the localized categories, then the adjunction `adj`
induces an adjunction `G' ⊣ F'`. -/
noncomputable def localization : G' ⊣ F' :=
  Adjunction.mkOfUnitCounit
    { unit := Localization.ε adj L₁ W₁ L₂ G' F'
      counit := Localization.η adj L₁ L₂ W₂ G' F'
      left_triangle := by
        /-
          C₁ : Type u_1
          C₂ : Type u_2
          D₁ : Type u_3
          D₂ : Type u_4
          inst✝⁷ : CategoryTheory.Category.{?u.20036, u_1} C₁
          inst✝⁶ : CategoryTheory.Category.{?u.20040, u_2} C₂
          inst✝⁵ : CategoryTheory.Category.{?u.20044, u_3} D₁
          inst✝⁴ : CategoryTheory.Category.{?u.20048, u_4} D₂
          G : CategoryTheory.Functor C₁ C₂
          F : CategoryTheory.Functor C₂ C₁
          adj : CategoryTheory.Adjunction G F
          L₁ : CategoryTheory.Functor C₁ D₁
          W₁ : CategoryTheory.MorphismProperty C₁
          inst✝³ : L₁.IsLocalization W₁
          L₂ : CategoryTheory.Functor C₂ D₂
          W₂ : CategoryTheory.MorphismProperty C₂
          inst✝² : L₂.IsLocalization W₂
          G' : CategoryTheory.Functor D₁ D₂
          F' : CategoryTheory.Functor D₂ D₁
          inst✝¹ : CategoryTheory.CatCommSq G L₁ L₂ G'
          inst✝ : CategoryTheory.CatCommSq F L₂ L₁ F'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerRight (Categor …
        -/
        apply natTrans_ext L₁ W₁
        /-
          C₁ : Type u_1
          C₂ : Type u_2
          D₁ : Type u_3
          D₂ : Type u_4
          inst✝⁷ : CategoryTheory.Category.{?u.20036, u_1} C₁
          inst✝⁶ : CategoryTheory.Category.{?u.20040, u_2} C₂
          inst✝⁵ : CategoryTheory.Category.{?u.20044, u_3} D₁
          inst✝⁴ : CategoryTheory.Category.{?u.20048, u_4} D₂
          G : CategoryTheory.Functor C₁ C₂
          F : CategoryTheory.Functor C₂ C₁
          adj : CategoryTheory.Adjunction G F
          L₁ : CategoryTheory.Functor C₁ D₁
          W₁ : CategoryTheory.MorphismProperty C₁
          inst✝³ : L₁.IsLocalization W₁
          L₂ : CategoryTheory.Functor C₂ D₂
          W₂ : CategoryTheory.MorphismProperty C₂
          inst✝² : L₂.IsLocalization W₂
          G' : CategoryTheory.Functor D₁ D₂
          F' : CategoryTheory.Functor D₂ D₁
          inst✝¹ : CategoryTheory.CatCommSq G L₁ L₂ G'
          inst✝ : CategoryTheory.CatCommSq F L₂ L₁ F'
          ⊢ ∀ (X : C₁), Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerR …
        -/
        intro X₁
        /-
          C₁ : Type u_1
          C₂ : Type u_2
          D₁ : Type u_3
          D₂ : Type u_4
          inst✝⁷ : CategoryTheory.Category.{?u.20036, u_1} C₁
          inst✝⁶ : CategoryTheory.Category.{?u.20040, u_2} C₂
          inst✝⁵ : CategoryTheory.Category.{?u.20044, u_3} D₁
          inst✝⁴ : CategoryTheory.Category.{?u.20048, u_4} D₂
          G : CategoryTheory.Functor C₁ C₂
          F : CategoryTheory.Functor C₂ C₁
          adj : CategoryTheory.Adjunction G F
          L₁ : CategoryTheory.Functor C₁ D₁
          W₁ : CategoryTheory.MorphismProperty C₁
          inst✝³ : L₁.IsLocalization W₁
          L₂ : CategoryTheory.Functor C₂ D₂
          W₂ : CategoryTheory.MorphismProperty C₂
          inst✝² : L₂.IsLocalization W₂
          G' : CategoryTheory.Functor D₁ D₂
          F' : CategoryTheory.Functor D₂ D₁
          inst✝¹ : CategoryTheory.CatCommSq G L₁ L₂ G'
          inst✝ : CategoryTheory.CatCommSq F L₂ L₁ F'
          X₁ : C₁
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerRight (Catego …
        -/
        have eq := congr_app adj.left_triangle X₁
        /-
          C₁ : Type u_1
          C₂ : Type u_2
          D₁ : Type u_3
          D₂ : Type u_4
          inst✝⁷ : CategoryTheory.Category.{?u.20036, u_1} C₁
          inst✝⁶ : CategoryTheory.Category.{?u.20040, u_2} C₂
          inst✝⁵ : CategoryTheory.Category.{?u.20044, u_3} D₁
          inst✝⁴ : CategoryTheory.Category.{?u.20048, u_4} D₂
          G : CategoryTheory.Functor C₁ C₂
          F : CategoryTheory.Functor C₂ C₁
          adj : CategoryTheory.Adjunction G F
          L₁ : CategoryTheory.Functor C₁ D₁
          W₁ : CategoryTheory.MorphismProperty C₁
          inst✝³ : L₁.IsLocalization W₁
          L₂ : CategoryTheory.Functor C₂ D₂
          W₂ : CategoryTheory.MorphismProperty C₂
          inst✝² : L₂.IsLocalization W₂
          G' : CategoryTheory.Functor D₁ D₂
          F' : CategoryTheory.Functor D₂ D₁
          inst✝¹ : CategoryTheory.CatCommSq G L₁ L₂ G'
          inst✝ : CategoryTheory.CatCommSq F L₂ L₁ F'
          X₁ : C₁
          eq : Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerRight adj. …
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerRight (Catego …
        -/
        dsimp at eq
        rw [NatTrans.comp_app, NatTrans.comp_app, whiskerRight_app, Localization.ε_app,
          Functor.associator_hom_app, id_comp, whiskerLeft_app, G'.map_comp, G'.map_comp,
          assoc, assoc]
        erw [(Localization.η adj L₁ L₂ W₂ G' F').naturality, Localization.η_app,
          assoc, assoc, ← G'.map_comp_assoc, ← G'.map_comp_assoc, assoc, Iso.hom_inv_id_app,
          comp_id, (CatCommSq.iso G L₁ L₂ G').inv.naturality_assoc, ← L₂.map_comp_assoc, eq,
          L₂.map_id, id_comp, Iso.inv_hom_id_app]
        /-
          C₁ : Type u_1
          C₂ : Type u_2
          D₁ : Type u_3
          D₂ : Type u_4
          inst✝⁷ : CategoryTheory.Category.{?u.20036, u_1} C₁
          inst✝⁶ : CategoryTheory.Category.{?u.20040, u_2} C₂
          inst✝⁵ : CategoryTheory.Category.{?u.20044, u_3} D₁
          inst✝⁴ : CategoryTheory.Category.{?u.20048, u_4} D₂
          G : CategoryTheory.Functor C₁ C₂
          F : CategoryTheory.Functor C₂ C₁
          adj : CategoryTheory.Adjunction G F
          L₁ : CategoryTheory.Functor C₁ D₁
          W₁ : CategoryTheory.MorphismProperty C₁
          inst✝³ : L₁.IsLocalization W₁
          L₂ : CategoryTheory.Functor C₂ D₂
          W₂ : CategoryTheory.MorphismProperty C₂
          inst✝² : L₂.IsLocalization W₂
          G' : CategoryTheory.Functor D₁ D₂
          F' : CategoryTheory.Functor D₂ D₁
          inst✝¹ : CategoryTheory.CatCommSq G L₁ L₂ G'
          inst✝ : CategoryTheory.CatCommSq F L₂ L₁ F'
          X₁ : C₁
          eq : Eq (CategoryTheory.CategoryStruct.comp (G.map (adj.unit.app X₁)) (adj.cou …
          ⊢ Eq (CategoryTheory.CategoryStruct.id ((L₁.comp G').obj ((CategoryTheory.Func …
        -/
        rfl
        /-
          🎉 no goals
        -/
      right_triangle := by
        /-
          C₁ : Type u_1
          C₂ : Type u_2
          D₁ : Type u_3
          D₂ : Type u_4
          inst✝⁷ : CategoryTheory.Category.{?u.20036, u_1} C₁
          inst✝⁶ : CategoryTheory.Category.{?u.20040, u_2} C₂
          inst✝⁵ : CategoryTheory.Category.{?u.20044, u_3} D₁
          inst✝⁴ : CategoryTheory.Category.{?u.20048, u_4} D₂
          G : CategoryTheory.Functor C₁ C₂
          F : CategoryTheory.Functor C₂ C₁
          adj : CategoryTheory.Adjunction G F
          L₁ : CategoryTheory.Functor C₁ D₁
          W₁ : CategoryTheory.MorphismProperty C₁
          inst✝³ : L₁.IsLocalization W₁
          L₂ : CategoryTheory.Functor C₂ D₂
          W₂ : CategoryTheory.MorphismProperty C₂
          inst✝² : L₂.IsLocalization W₂
          G' : CategoryTheory.Functor D₁ D₂
          F' : CategoryTheory.Functor D₂ D₁
          inst✝¹ : CategoryTheory.CatCommSq G L₁ L₂ G'
          inst✝ : CategoryTheory.CatCommSq F L₂ L₁ F'
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerLeft F' (Categ …
        -/
        apply natTrans_ext L₂ W₂
        /-
          C₁ : Type u_1
          C₂ : Type u_2
          D₁ : Type u_3
          D₂ : Type u_4
          inst✝⁷ : CategoryTheory.Category.{?u.20036, u_1} C₁
          inst✝⁶ : CategoryTheory.Category.{?u.20040, u_2} C₂
          inst✝⁵ : CategoryTheory.Category.{?u.20044, u_3} D₁
          inst✝⁴ : CategoryTheory.Category.{?u.20048, u_4} D₂
          G : CategoryTheory.Functor C₁ C₂
          F : CategoryTheory.Functor C₂ C₁
          adj : CategoryTheory.Adjunction G F
          L₁ : CategoryTheory.Functor C₁ D₁
          W₁ : CategoryTheory.MorphismProperty C₁
          inst✝³ : L₁.IsLocalization W₁
          L₂ : CategoryTheory.Functor C₂ D₂
          W₂ : CategoryTheory.MorphismProperty C₂
          inst✝² : L₂.IsLocalization W₂
          G' : CategoryTheory.Functor D₁ D₂
          F' : CategoryTheory.Functor D₂ D₁
          inst✝¹ : CategoryTheory.CatCommSq G L₁ L₂ G'
          inst✝ : CategoryTheory.CatCommSq F L₂ L₁ F'
          ⊢ ∀ (X : C₂), Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerL …
        -/
        intro X₂
        /-
          C₁ : Type u_1
          C₂ : Type u_2
          D₁ : Type u_3
          D₂ : Type u_4
          inst✝⁷ : CategoryTheory.Category.{?u.20036, u_1} C₁
          inst✝⁶ : CategoryTheory.Category.{?u.20040, u_2} C₂
          inst✝⁵ : CategoryTheory.Category.{?u.20044, u_3} D₁
          inst✝⁴ : CategoryTheory.Category.{?u.20048, u_4} D₂
          G : CategoryTheory.Functor C₁ C₂
          F : CategoryTheory.Functor C₂ C₁
          adj : CategoryTheory.Adjunction G F
          L₁ : CategoryTheory.Functor C₁ D₁
          W₁ : CategoryTheory.MorphismProperty C₁
          inst✝³ : L₁.IsLocalization W₁
          L₂ : CategoryTheory.Functor C₂ D₂
          W₂ : CategoryTheory.MorphismProperty C₂
          inst✝² : L₂.IsLocalization W₂
          G' : CategoryTheory.Functor D₁ D₂
          F' : CategoryTheory.Functor D₂ D₁
          inst✝¹ : CategoryTheory.CatCommSq G L₁ L₂ G'
          inst✝ : CategoryTheory.CatCommSq F L₂ L₁ F'
          X₂ : C₂
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerLeft F' (Cate …
        -/
        have eq := congr_app adj.right_triangle X₂
        /-
          C₁ : Type u_1
          C₂ : Type u_2
          D₁ : Type u_3
          D₂ : Type u_4
          inst✝⁷ : CategoryTheory.Category.{?u.20036, u_1} C₁
          inst✝⁶ : CategoryTheory.Category.{?u.20040, u_2} C₂
          inst✝⁵ : CategoryTheory.Category.{?u.20044, u_3} D₁
          inst✝⁴ : CategoryTheory.Category.{?u.20048, u_4} D₂
          G : CategoryTheory.Functor C₁ C₂
          F : CategoryTheory.Functor C₂ C₁
          adj : CategoryTheory.Adjunction G F
          L₁ : CategoryTheory.Functor C₁ D₁
          W₁ : CategoryTheory.MorphismProperty C₁
          inst✝³ : L₁.IsLocalization W₁
          L₂ : CategoryTheory.Functor C₂ D₂
          W₂ : CategoryTheory.MorphismProperty C₂
          inst✝² : L₂.IsLocalization W₂
          G' : CategoryTheory.Functor D₁ D₂
          F' : CategoryTheory.Functor D₂ D₁
          inst✝¹ : CategoryTheory.CatCommSq G L₁ L₂ G'
          inst✝ : CategoryTheory.CatCommSq F L₂ L₁ F'
          X₂ : C₂
          eq : Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerLeft F adj …
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerLeft F' (Cate …
        -/
        dsimp at eq
        rw [NatTrans.comp_app, NatTrans.comp_app, whiskerLeft_app, whiskerRight_app,
          Localization.η_app, Functor.associator_inv_app, id_comp, F'.map_comp, F'.map_comp]
        erw [← (Localization.ε _ _ _ _ _ _).naturality_assoc, Localization.ε_app,
          assoc, assoc, ← F'.map_comp_assoc, Iso.hom_inv_id_app, F'.map_id, id_comp,
          ← NatTrans.naturality, ← L₁.map_comp_assoc, eq, L₁.map_id, id_comp,
          Iso.inv_hom_id_app]
        /-
          C₁ : Type u_1
          C₂ : Type u_2
          D₁ : Type u_3
          D₂ : Type u_4
          inst✝⁷ : CategoryTheory.Category.{?u.20036, u_1} C₁
          inst✝⁶ : CategoryTheory.Category.{?u.20040, u_2} C₂
          inst✝⁵ : CategoryTheory.Category.{?u.20044, u_3} D₁
          inst✝⁴ : CategoryTheory.Category.{?u.20048, u_4} D₂
          G : CategoryTheory.Functor C₁ C₂
          F : CategoryTheory.Functor C₂ C₁
          adj : CategoryTheory.Adjunction G F
          L₁ : CategoryTheory.Functor C₁ D₁
          W₁ : CategoryTheory.MorphismProperty C₁
          inst✝³ : L₁.IsLocalization W₁
          L₂ : CategoryTheory.Functor C₂ D₂
          W₂ : CategoryTheory.MorphismProperty C₂
          inst✝² : L₂.IsLocalization W₂
          G' : CategoryTheory.Functor D₁ D₂
          F' : CategoryTheory.Functor D₂ D₁
          inst✝¹ : CategoryTheory.CatCommSq G L₁ L₂ G'
          inst✝ : CategoryTheory.CatCommSq F L₂ L₁ F'
          X₂ : C₂
          eq : Eq (CategoryTheory.CategoryStruct.comp (adj.unit.app (F.obj X₂)) (F.map ( …
          ⊢ Eq (CategoryTheory.CategoryStruct.id ((L₂.comp F').obj X₂)) ((CategoryTheory …
        -/
        rfl }
        /-
          🎉 no goals
        -/


@[simp]
lemma localization_unit_app (X₁ : C₁) :
    (adj.localization L₁ W₁ L₂ W₂ G' F').unit.app (L₁.obj X₁) =
    L₁.map (adj.unit.app X₁) ≫ (CatCommSq.iso F L₂ L₁ F').hom.app (G.obj X₁) ≫
      F'.map ((CatCommSq.iso G L₁ L₂ G').hom.app X₁) := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    D₁ : Type u_3
    D₂ : Type u_4
    inst✝⁷ : CategoryTheory.Category.{u_6, u_1} C₁
    inst✝⁶ : CategoryTheory.Category.{u_8, u_2} C₂
    inst✝⁵ : CategoryTheory.Category.{u_5, u_3} D₁
    inst✝⁴ : CategoryTheory.Category.{u_7, u_4} D₂
    G : CategoryTheory.Functor C₁ C₂
    F : CategoryTheory.Functor C₂ C₁
    adj : CategoryTheory.Adjunction G F
    L₁ : CategoryTheory.Functor C₁ D₁
    W₁ : CategoryTheory.MorphismProperty C₁
    inst✝³ : L₁.IsLocalization W₁
    L₂ : CategoryTheory.Functor C₂ D₂
    W₂ : CategoryTheory.MorphismProperty C₂
    inst✝² : L₂.IsLocalization W₂
    G' : CategoryTheory.Functor D₁ D₂
    F' : CategoryTheory.Functor D₂ D₁
    inst✝¹ : CategoryTheory.CatCommSq G L₁ L₂ G'
    inst✝ : CategoryTheory.CatCommSq F L₂ L₁ F'
    X₁ : C₁
    ⊢ Eq ((adj.localization L₁ W₁ L₂ W₂ G' F').unit.app (L₁.obj X₁)) (CategoryTheo …
  -/
  apply Localization.ε_app
  /-
    🎉 no goals
  -/


@[simp]
lemma localization_counit_app (X₂ : C₂) :
    (adj.localization L₁ W₁ L₂ W₂ G' F').counit.app (L₂.obj X₂) =
    G'.map ((CatCommSq.iso F L₂ L₁ F').inv.app X₂) ≫
      (CatCommSq.iso G L₁ L₂ G').inv.app (F.obj X₂) ≫
      L₂.map (adj.counit.app X₂) := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    D₁ : Type u_3
    D₂ : Type u_4
    inst✝⁷ : CategoryTheory.Category.{u_8, u_1} C₁
    inst✝⁶ : CategoryTheory.Category.{u_7, u_2} C₂
    inst✝⁵ : CategoryTheory.Category.{u_6, u_3} D₁
    inst✝⁴ : CategoryTheory.Category.{u_5, u_4} D₂
    G : CategoryTheory.Functor C₁ C₂
    F : CategoryTheory.Functor C₂ C₁
    adj : CategoryTheory.Adjunction G F
    L₁ : CategoryTheory.Functor C₁ D₁
    W₁ : CategoryTheory.MorphismProperty C₁
    inst✝³ : L₁.IsLocalization W₁
    L₂ : CategoryTheory.Functor C₂ D₂
    W₂ : CategoryTheory.MorphismProperty C₂
    inst✝² : L₂.IsLocalization W₂
    G' : CategoryTheory.Functor D₁ D₂
    F' : CategoryTheory.Functor D₂ D₁
    inst✝¹ : CategoryTheory.CatCommSq G L₁ L₂ G'
    inst✝ : CategoryTheory.CatCommSq F L₂ L₁ F'
    X₂ : C₂
    ⊢ Eq ((adj.localization L₁ W₁ L₂ W₂ G' F').counit.app (L₂.obj X₂)) (CategoryTh …
  -/
  apply Localization.η_app
  /-
    🎉 no goals
  -/


include adj in
lemma isLocalization [F.Full] [F.Faithful] :
    G.IsLocalization ((MorphismProperty.isomorphisms C₂).inverseImage G) := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    inst✝³ : CategoryTheory.Category.{u_6, u_1} C₁
    inst✝² : CategoryTheory.Category.{u_5, u_2} C₂
    G : CategoryTheory.Functor C₁ C₂
    F : CategoryTheory.Functor C₂ C₁
    adj : CategoryTheory.Adjunction G F
    inst✝¹ : F.Full
    inst✝ : F.Faithful
    ⊢ G.IsLocalization ((CategoryTheory.MorphismProperty.isomorphisms C₂).inverseI …
  -/
  let W := ((MorphismProperty.isomorphisms C₂).inverseImage G)
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    inst✝³ : CategoryTheory.Category.{u_6, u_1} C₁
    inst✝² : CategoryTheory.Category.{u_5, u_2} C₂
    G : CategoryTheory.Functor C₁ C₂
    F : CategoryTheory.Functor C₂ C₁
    adj : CategoryTheory.Adjunction G F
    inst✝¹ : F.Full
    inst✝ : F.Faithful
    W : CategoryTheory.MorphismProperty C₁ := (CategoryTheory.MorphismProperty.iso …
    ⊢ G.IsLocalization ((CategoryTheory.MorphismProperty.isomorphisms C₂).inverseI …
  -/
  have hG : W.IsInvertedBy G := fun _ _ _ hf => hf
  have : ∀ (X : C₁), IsIso ((whiskerRight adj.unit W.Q).app X) := fun X =>
    Localization.inverts W.Q W _ (by
      change IsIso _
      infer_instance)
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    inst✝³ : CategoryTheory.Category.{u_6, u_1} C₁
    inst✝² : CategoryTheory.Category.{u_5, u_2} C₂
    G : CategoryTheory.Functor C₁ C₂
    F : CategoryTheory.Functor C₂ C₁
    adj : CategoryTheory.Adjunction G F
    inst✝¹ : F.Full
    inst✝ : F.Faithful
    W : CategoryTheory.MorphismProperty C₁ := (CategoryTheory.MorphismProperty.iso …
    hG : W.IsInvertedBy G
    this : ∀ (X : C₁), CategoryTheory.IsIso ((CategoryTheory.whiskerRight adj.unit …
    ⊢ G.IsLocalization ((CategoryTheory.MorphismProperty.isomorphisms C₂).inverseI …
  -/
  have : IsIso (whiskerRight adj.unit W.Q) := NatIso.isIso_of_isIso_app _
  let e : W.Localization ≌ C₂ := Equivalence.mk (Localization.lift G hG W.Q) (F ⋙ W.Q)
    (liftNatIso W.Q W W.Q (G ⋙ F ⋙ W.Q) _ _
    (W.Q.leftUnitor.symm ≪≫ asIso (whiskerRight adj.unit W.Q)))
    (Functor.associator _ _ _ ≪≫ isoWhiskerLeft _ (Localization.fac G hG W.Q) ≪≫
      asIso adj.counit)
  apply Functor.IsLocalization.of_equivalence_target W.Q W G e
    (Localization.fac G hG W.Q)


