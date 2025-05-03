/-- If `F` and `F'` are both left adjoint to `G`, then they are naturally isomorphic. -/
def leftAdjointUniq {F F' : C ⥤ D} {G : D ⥤ C} (adj1 : F ⊣ G) (adj2 : F' ⊣ G) : F ≅ F' :=
  ((conjugateIsoEquiv adj1 adj2).symm (Iso.refl G)).symm


theorem homEquiv_leftAdjointUniq_hom_app {F F' : C ⥤ D} {G : D ⥤ C} (adj1 : F ⊣ G) (adj2 : F' ⊣ G)
    (x : C) : adj1.homEquiv _ _ ((leftAdjointUniq adj1 adj2).hom.app x) = adj2.unit.app x := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    F F' : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj1 : CategoryTheory.Adjunction F G
    adj2 : CategoryTheory.Adjunction F' G
    x : C
    ⊢ Eq ((adj1.homEquiv x (F'.obj x)) ((adj1.leftAdjointUniq adj2).hom.app x)) (a …
  -/
  simp [leftAdjointUniq]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem unit_leftAdjointUniq_hom {F F' : C ⥤ D} {G : D ⥤ C} (adj1 : F ⊣ G) (adj2 : F' ⊣ G) :
    adj1.unit ≫ whiskerRight (leftAdjointUniq adj1 adj2).hom G = adj2.unit := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    F F' : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj1 : CategoryTheory.Adjunction F G
    adj2 : CategoryTheory.Adjunction F' G
    ⊢ Eq (CategoryTheory.CategoryStruct.comp adj1.unit (CategoryTheory.whiskerRigh …
  -/
  ext x
  /-
    case w.h
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    F F' : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj1 : CategoryTheory.Adjunction F G
    adj2 : CategoryTheory.Adjunction F' G
    x : C
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp adj1.unit (CategoryTheory.whiskerRig …
  -/
  rw [NatTrans.comp_app, ← homEquiv_leftAdjointUniq_hom_app adj1 adj2]
  /-
    case w.h
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    F F' : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj1 : CategoryTheory.Adjunction F G
    adj2 : CategoryTheory.Adjunction F' G
    x : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj1.unit.app x) ((CategoryTheory.wh …
  -/
  simp [← G.map_comp]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem unit_leftAdjointUniq_hom_app
    {F F' : C ⥤ D} {G : D ⥤ C} (adj1 : F ⊣ G) (adj2 : F' ⊣ G) (x : C) :
    adj1.unit.app x ≫ G.map ((leftAdjointUniq adj1 adj2).hom.app x) = adj2.unit.app x := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    F F' : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj1 : CategoryTheory.Adjunction F G
    adj2 : CategoryTheory.Adjunction F' G
    x : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj1.unit.app x) (G.map ((adj1.leftA …
  -/
  rw [← unit_leftAdjointUniq_hom adj1 adj2]; rfl
                                             /-
                                               🎉 no goals
                                             -/


@[reassoc (attr := simp)]
theorem leftAdjointUniq_hom_counit {F F' : C ⥤ D} {G : D ⥤ C} (adj1 : F ⊣ G) (adj2 : F' ⊣ G) :
    whiskerLeft G (leftAdjointUniq adj1 adj2).hom ≫ adj2.counit = adj1.counit := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    F F' : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj1 : CategoryTheory.Adjunction F G
    adj2 : CategoryTheory.Adjunction F' G
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerLeft G (adj1.l …
  -/
  ext x
  simp only [Functor.comp_obj, Functor.id_obj, leftAdjointUniq, Iso.symm_hom,
    conjugateIsoEquiv_symm_apply_inv, Iso.refl_inv, NatTrans.comp_app, whiskerLeft_app,
    conjugateEquiv_symm_apply_app, NatTrans.id_app, Functor.map_id, Category.id_comp,
    Category.assoc]
  /-
    case w.h
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    F F' : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj1 : CategoryTheory.Adjunction F G
    adj2 : CategoryTheory.Adjunction F' G
    x : D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (adj2.unit.app (G.obj x))) (Ca …
  -/
  rw [← adj1.counit_naturality, ← Category.assoc, ← F.map_comp]
  /-
    case w.h
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    F F' : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj1 : CategoryTheory.Adjunction F G
    adj2 : CategoryTheory.Adjunction F' G
    x : D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map (CategoryTheory.CategoryStruct …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem leftAdjointUniq_hom_app_counit {F F' : C ⥤ D} {G : D ⥤ C} (adj1 : F ⊣ G) (adj2 : F' ⊣ G)
    (x : D) :
    (leftAdjointUniq adj1 adj2).hom.app (G.obj x) ≫ adj2.counit.app x = adj1.counit.app x := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    F F' : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj1 : CategoryTheory.Adjunction F G
    adj2 : CategoryTheory.Adjunction F' G
    x : D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((adj1.leftAdjointUniq adj2).hom.app  …
  -/
  rw [← leftAdjointUniq_hom_counit adj1 adj2]
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    F F' : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj1 : CategoryTheory.Adjunction F G
    adj2 : CategoryTheory.Adjunction F' G
    x : D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((adj1.leftAdjointUniq adj2).hom.app  …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem leftAdjointUniq_inv_app {F F' : C ⥤ D} {G : D ⥤ C} (adj1 : F ⊣ G) (adj2 : F' ⊣ G) (x : C) :
    (leftAdjointUniq adj1 adj2).inv.app x = (leftAdjointUniq adj2 adj1).hom.app x :=
  rfl


@[reassoc (attr := simp)]
theorem leftAdjointUniq_trans {F F' F'' : C ⥤ D} {G : D ⥤ C} (adj1 : F ⊣ G) (adj2 : F' ⊣ G)
    (adj3 : F'' ⊣ G) :
    (leftAdjointUniq adj1 adj2).hom ≫ (leftAdjointUniq adj2 adj3).hom =
      (leftAdjointUniq adj1 adj3).hom := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    F F' F'' : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj1 : CategoryTheory.Adjunction F G
    adj2 : CategoryTheory.Adjunction F' G
    adj3 : CategoryTheory.Adjunction F'' G
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj1.leftAdjointUniq adj2).hom (adj2 …
  -/
  simp [leftAdjointUniq]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem leftAdjointUniq_trans_app {F F' F'' : C ⥤ D} {G : D ⥤ C} (adj1 : F ⊣ G) (adj2 : F' ⊣ G)
    (adj3 : F'' ⊣ G) (x : C) :
    (leftAdjointUniq adj1 adj2).hom.app x ≫ (leftAdjointUniq adj2 adj3).hom.app x =
      (leftAdjointUniq adj1 adj3).hom.app x := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    F F' F'' : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj1 : CategoryTheory.Adjunction F G
    adj2 : CategoryTheory.Adjunction F' G
    adj3 : CategoryTheory.Adjunction F'' G
    x : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((adj1.leftAdjointUniq adj2).hom.app  …
  -/
  rw [← leftAdjointUniq_trans adj1 adj2 adj3]
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    F F' F'' : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj1 : CategoryTheory.Adjunction F G
    adj2 : CategoryTheory.Adjunction F' G
    adj3 : CategoryTheory.Adjunction F'' G
    x : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((adj1.leftAdjointUniq adj2).hom.app  …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem leftAdjointUniq_refl {F : C ⥤ D} {G : D ⥤ C} (adj1 : F ⊣ G) :
    (leftAdjointUniq adj1 adj1).hom = 𝟙 _ := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj1 : CategoryTheory.Adjunction F G
    ⊢ Eq (adj1.leftAdjointUniq adj1).hom (CategoryTheory.CategoryStruct.id F)
  -/
  simp [leftAdjointUniq]
  /-
    🎉 no goals
  -/


/-- If `G` and `G'` are both right adjoint to `F`, then they are naturally isomorphic. -/
def rightAdjointUniq {F : C ⥤ D} {G G' : D ⥤ C} (adj1 : F ⊣ G) (adj2 : F ⊣ G') : G ≅ G' :=
  conjugateIsoEquiv adj1 adj2 (Iso.refl _)


theorem homEquiv_symm_rightAdjointUniq_hom_app {F : C ⥤ D} {G G' : D ⥤ C} (adj1 : F ⊣ G)
    (adj2 : F ⊣ G') (x : D) :
    (adj2.homEquiv _ _).symm ((rightAdjointUniq adj1 adj2).hom.app x) = adj1.counit.app x := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    F : CategoryTheory.Functor C D
    G G' : CategoryTheory.Functor D C
    adj1 : CategoryTheory.Adjunction F G
    adj2 : CategoryTheory.Adjunction F G'
    x : D
    ⊢ Eq ((adj2.homEquiv (G.obj x) x).symm ((adj1.rightAdjointUniq adj2).hom.app x …
  -/
  simp [rightAdjointUniq]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem unit_rightAdjointUniq_hom_app {F : C ⥤ D} {G G' : D ⥤ C} (adj1 : F ⊣ G) (adj2 : F ⊣ G')
    (x : C) : adj1.unit.app x ≫ (rightAdjointUniq adj1 adj2).hom.app (F.obj x) =
      adj2.unit.app x := by
  simp only [Functor.id_obj, Functor.comp_obj, rightAdjointUniq, conjugateIsoEquiv_apply_hom,
    Iso.refl_hom, conjugateEquiv_apply_app, NatTrans.id_app, Functor.map_id, Category.id_comp]
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    F : CategoryTheory.Functor C D
    G G' : CategoryTheory.Functor D C
    adj1 : CategoryTheory.Adjunction F G
    adj2 : CategoryTheory.Adjunction F G'
    x : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj1.unit.app x) (CategoryTheory.Cat …
  -/
  rw [← adj2.unit_naturality_assoc, ← G'.map_comp]
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    F : CategoryTheory.Functor C D
    G G' : CategoryTheory.Functor D C
    adj1 : CategoryTheory.Adjunction F G
    adj2 : CategoryTheory.Adjunction F G'
    x : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj2.unit.app x) (G'.map (CategoryTh …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem unit_rightAdjointUniq_hom {F : C ⥤ D} {G G' : D ⥤ C} (adj1 : F ⊣ G) (adj2 : F ⊣ G') :
    adj1.unit ≫ whiskerLeft F (rightAdjointUniq adj1 adj2).hom = adj2.unit := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    F : CategoryTheory.Functor C D
    G G' : CategoryTheory.Functor D C
    adj1 : CategoryTheory.Adjunction F G
    adj2 : CategoryTheory.Adjunction F G'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp adj1.unit (CategoryTheory.whiskerLeft …
  -/
  ext x
  /-
    case w.h
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    F : CategoryTheory.Functor C D
    G G' : CategoryTheory.Functor D C
    adj1 : CategoryTheory.Adjunction F G
    adj2 : CategoryTheory.Adjunction F G'
    x : C
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp adj1.unit (CategoryTheory.whiskerLef …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem rightAdjointUniq_hom_app_counit {F : C ⥤ D} {G G' : D ⥤ C} (adj1 : F ⊣ G) (adj2 : F ⊣ G')
    (x : D) :
    F.map ((rightAdjointUniq adj1 adj2).hom.app x) ≫ adj2.counit.app x = adj1.counit.app x := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    F : CategoryTheory.Functor C D
    G G' : CategoryTheory.Functor D C
    adj1 : CategoryTheory.Adjunction F G
    adj2 : CategoryTheory.Adjunction F G'
    x : D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map ((adj1.rightAdjointUniq adj2). …
  -/
  simp [rightAdjointUniq]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem rightAdjointUniq_hom_counit {F : C ⥤ D} {G G' : D ⥤ C} (adj1 : F ⊣ G) (adj2 : F ⊣ G') :
    whiskerRight (rightAdjointUniq adj1 adj2).hom F ≫ adj2.counit = adj1.counit := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    F : CategoryTheory.Functor C D
    G G' : CategoryTheory.Functor D C
    adj1 : CategoryTheory.Adjunction F G
    adj2 : CategoryTheory.Adjunction F G'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerRight (adj1.ri …
  -/
  ext
  /-
    case w.h
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    F : CategoryTheory.Functor C D
    G G' : CategoryTheory.Functor D C
    adj1 : CategoryTheory.Adjunction F G
    adj2 : CategoryTheory.Adjunction F G'
    x✝ : D
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.whiskerRight (adj1.r …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem rightAdjointUniq_inv_app {F : C ⥤ D} {G G' : D ⥤ C} (adj1 : F ⊣ G) (adj2 : F ⊣ G')
    (x : D) : (rightAdjointUniq adj1 adj2).inv.app x = (rightAdjointUniq adj2 adj1).hom.app x :=
  rfl


@[reassoc (attr := simp)]
theorem rightAdjointUniq_trans {F : C ⥤ D} {G G' G'' : D ⥤ C} (adj1 : F ⊣ G) (adj2 : F ⊣ G')
    (adj3 : F ⊣ G'') :
    (rightAdjointUniq adj1 adj2).hom ≫ (rightAdjointUniq adj2 adj3).hom =
      (rightAdjointUniq adj1 adj3).hom := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    F : CategoryTheory.Functor C D
    G G' G'' : CategoryTheory.Functor D C
    adj1 : CategoryTheory.Adjunction F G
    adj2 : CategoryTheory.Adjunction F G'
    adj3 : CategoryTheory.Adjunction F G''
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (adj1.rightAdjointUniq adj2).hom (adj …
  -/
  simp [rightAdjointUniq]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem rightAdjointUniq_trans_app {F : C ⥤ D} {G G' G'' : D ⥤ C} (adj1 : F ⊣ G) (adj2 : F ⊣ G')
    (adj3 : F ⊣ G'') (x : D) :
    (rightAdjointUniq adj1 adj2).hom.app x ≫ (rightAdjointUniq adj2 adj3).hom.app x =
      (rightAdjointUniq adj1 adj3).hom.app x := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    F : CategoryTheory.Functor C D
    G G' G'' : CategoryTheory.Functor D C
    adj1 : CategoryTheory.Adjunction F G
    adj2 : CategoryTheory.Adjunction F G'
    adj3 : CategoryTheory.Adjunction F G''
    x : D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((adj1.rightAdjointUniq adj2).hom.app …
  -/
  rw [← rightAdjointUniq_trans adj1 adj2 adj3]
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    F : CategoryTheory.Functor C D
    G G' G'' : CategoryTheory.Functor D C
    adj1 : CategoryTheory.Adjunction F G
    adj2 : CategoryTheory.Adjunction F G'
    adj3 : CategoryTheory.Adjunction F G''
    x : D
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((adj1.rightAdjointUniq adj2).hom.app …
  -/
  rfl
  /-
    🎉 no goals
  -/



@[simp]
theorem rightAdjointUniq_refl {F : C ⥤ D} {G : D ⥤ C} (adj1 : F ⊣ G) :
    (rightAdjointUniq adj1 adj1).hom = 𝟙 _ := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj1 : CategoryTheory.Adjunction F G
    ⊢ Eq (adj1.rightAdjointUniq adj1).hom (CategoryTheory.CategoryStruct.id G)
  -/
  delta rightAdjointUniq
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Category.{u_4, u_2} D
    F : CategoryTheory.Functor C D
    G : CategoryTheory.Functor D C
    adj1 : CategoryTheory.Adjunction F G
    ⊢ Eq ((CategoryTheory.conjugateIsoEquiv adj1 adj1) (CategoryTheory.Iso.refl F) …
  -/
  simp
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-07")] alias Adjunction.natTransEquiv := conjugateEquiv

@[deprecated (since := "2024-10-07")] alias Adjunction.natIsoEquiv := conjugateIsoEquiv


