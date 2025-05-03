/-- A functor `RF : D ⥤ H` is a right derived functor of `F : C ⥤ H`
if it is equipped with a natural transformation `α : F ⟶ L ⋙ RF`
which makes it a left Kan extension of `F` along `L`,
where `L : C ⥤ D` is a localization functor for `W : MorphismProperty C`. -/
class IsRightDerivedFunctor [L.IsLocalization W] : Prop where
  isLeftKanExtension' : RF.IsLeftKanExtension α


lemma IsRightDerivedFunctor.isLeftKanExtension
    [L.IsLocalization W] [RF.IsRightDerivedFunctor α W] :
    RF.IsLeftKanExtension α :=
  IsRightDerivedFunctor.isLeftKanExtension' W


lemma isRightDerivedFunctor_iff_isLeftKanExtension [L.IsLocalization W] :
    RF.IsRightDerivedFunctor α W ↔ RF.IsLeftKanExtension α := by
  /-
    C : Type u_1
    D : Type u_2
    H : Type u_5
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Category.{u_4, u_2} D
    inst✝¹ : CategoryTheory.Category.{u_6, u_5} H
    RF : CategoryTheory.Functor D H
    F : CategoryTheory.Functor C H
    L : CategoryTheory.Functor C D
    α : Quiver.Hom F (L.comp RF)
    W : CategoryTheory.MorphismProperty C
    inst✝ : L.IsLocalization W
    ⊢ Iff (RF.IsRightDerivedFunctor α W) (RF.IsLeftKanExtension α)
  -/
  constructor
    /-
      case mp
      C : Type u_1
      D : Type u_2
      H : Type u_5
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      inst✝¹ : CategoryTheory.Category.{u_6, u_5} H
      RF : CategoryTheory.Functor D H
      F : CategoryTheory.Functor C H
      L : CategoryTheory.Functor C D
      α : Quiver.Hom F (L.comp RF)
      W : CategoryTheory.MorphismProperty C
      inst✝ : L.IsLocalization W
      ⊢ RF.IsRightDerivedFunctor α W → RF.IsLeftKanExtension α
    -/
  · exact fun _ => IsRightDerivedFunctor.isLeftKanExtension RF α W
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      D : Type u_2
      H : Type u_5
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      inst✝¹ : CategoryTheory.Category.{u_6, u_5} H
      RF : CategoryTheory.Functor D H
      F : CategoryTheory.Functor C H
      L : CategoryTheory.Functor C D
      α : Quiver.Hom F (L.comp RF)
      W : CategoryTheory.MorphismProperty C
      inst✝ : L.IsLocalization W
      ⊢ RF.IsLeftKanExtension α → RF.IsRightDerivedFunctor α W
    -/
  · exact fun h => ⟨h⟩
    /-
      🎉 no goals
    -/


variable {RF RF'} in
lemma isRightDerivedFunctor_iff_of_iso (α' : F ⟶ L ⋙ RF') (W : MorphismProperty C)
    [L.IsLocalization W] (e : RF ≅ RF') (comm : α ≫ whiskerLeft L e.hom = α') :
    RF.IsRightDerivedFunctor α W ↔ RF'.IsRightDerivedFunctor α' W := by
  /-
    C : Type u_1
    D : Type u_6
    H : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_5, u_6} D
    inst✝¹ : CategoryTheory.Category.{u_2, u_3} H
    RF RF' : CategoryTheory.Functor D H
    F : CategoryTheory.Functor C H
    L : CategoryTheory.Functor C D
    α : Quiver.Hom F (L.comp RF)
    α' : Quiver.Hom F (L.comp RF')
    W : CategoryTheory.MorphismProperty C
    inst✝ : L.IsLocalization W
    e : CategoryTheory.Iso RF RF'
    comm : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.whiskerLeft L  …
    ⊢ Iff (RF.IsRightDerivedFunctor α W) (RF'.IsRightDerivedFunctor α' W)
  -/
  simp only [isRightDerivedFunctor_iff_isLeftKanExtension]
  /-
    C : Type u_1
    D : Type u_6
    H : Type u_3
    inst✝³ : CategoryTheory.Category.{u_4, u_1} C
    inst✝² : CategoryTheory.Category.{u_5, u_6} D
    inst✝¹ : CategoryTheory.Category.{u_2, u_3} H
    RF RF' : CategoryTheory.Functor D H
    F : CategoryTheory.Functor C H
    L : CategoryTheory.Functor C D
    α : Quiver.Hom F (L.comp RF)
    α' : Quiver.Hom F (L.comp RF')
    W : CategoryTheory.MorphismProperty C
    inst✝ : L.IsLocalization W
    e : CategoryTheory.Iso RF RF'
    comm : Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.whiskerLeft L  …
    ⊢ Iff (RF.IsLeftKanExtension α) (RF'.IsLeftKanExtension α')
  -/
  exact isLeftKanExtension_iff_of_iso e _ _ comm
  /-
    🎉 no goals
  -/


/-- Constructor for natural transformations from a right derived functor. -/
noncomputable def rightDerivedDesc (G : D ⥤ H) (β : F ⟶ L ⋙ G) : RF ⟶ G :=
  have := IsRightDerivedFunctor.isLeftKanExtension RF α W
  RF.descOfIsLeftKanExtension α G β


@[reassoc (attr := simp)]
lemma rightDerived_fac (G : D ⥤ H) (β : F ⟶ L ⋙ G) :
    α ≫ whiskerLeft L (RF.rightDerivedDesc α W G β) = β :=
  have := IsRightDerivedFunctor.isLeftKanExtension RF α W
  RF.descOfIsLeftKanExtension_fac α G β


@[reassoc (attr := simp)]
lemma rightDerived_fac_app (G : D ⥤ H) (β : F ⟶ L ⋙ G) (X : C) :
    α.app X ≫ (RF.rightDerivedDesc α W G β).app (L.obj X) = β.app X :=
  have := IsRightDerivedFunctor.isLeftKanExtension RF α W
  RF.descOfIsLeftKanExtension_fac_app α G β X


include W in
lemma rightDerived_ext (G : D ⥤ H) (γ₁ γ₂ : RF ⟶ G)
    (hγ : α ≫ whiskerLeft L γ₁ = α ≫ whiskerLeft L γ₂) : γ₁ = γ₂ :=
  have := IsRightDerivedFunctor.isLeftKanExtension RF α W
  RF.hom_ext_of_isLeftKanExtension α γ₁ γ₂ hγ


/-- The natural transformation `RF ⟶ RF'` on right derived functors that is
induced by a natural transformation `F ⟶ F'`. -/
noncomputable def rightDerivedNatTrans (τ : F ⟶ F') : RF ⟶ RF' :=
  RF.rightDerivedDesc α W RF' (τ ≫ α')


@[reassoc (attr := simp)]
lemma rightDerivedNatTrans_fac (τ : F ⟶ F') :
    α ≫ whiskerLeft L (rightDerivedNatTrans RF RF' α α' W τ) = τ ≫ α' := by
  /-
    C : Type u_1
    D : Type u_6
    H : Type u_3
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_6} D
    inst✝² : CategoryTheory.Category.{u_2, u_3} H
    RF RF' : CategoryTheory.Functor D H
    F F' : CategoryTheory.Functor C H
    L : CategoryTheory.Functor C D
    α : Quiver.Hom F (L.comp RF)
    α' : Quiver.Hom F' (L.comp RF')
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : RF.IsRightDerivedFunctor α W
    τ : Quiver.Hom F F'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.whiskerLeft L (RF.r …
  -/
  dsimp only [rightDerivedNatTrans]
  /-
    C : Type u_1
    D : Type u_6
    H : Type u_3
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_6} D
    inst✝² : CategoryTheory.Category.{u_2, u_3} H
    RF RF' : CategoryTheory.Functor D H
    F F' : CategoryTheory.Functor C H
    L : CategoryTheory.Functor C D
    α : Quiver.Hom F (L.comp RF)
    α' : Quiver.Hom F' (L.comp RF')
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : RF.IsRightDerivedFunctor α W
    τ : Quiver.Hom F F'
    ⊢ Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.whiskerLeft L (RF.r …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma rightDerivedNatTrans_app (τ : F ⟶ F') (X : C) :
    α.app X ≫ (rightDerivedNatTrans RF RF' α α' W τ).app (L.obj X) =
    τ.app X ≫ α'.app X := by
  /-
    C : Type u_1
    D : Type u_6
    H : Type u_3
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_6} D
    inst✝² : CategoryTheory.Category.{u_2, u_3} H
    RF RF' : CategoryTheory.Functor D H
    F F' : CategoryTheory.Functor C H
    L : CategoryTheory.Functor C D
    α : Quiver.Hom F (L.comp RF)
    α' : Quiver.Hom F' (L.comp RF')
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : RF.IsRightDerivedFunctor α W
    τ : Quiver.Hom F F'
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (α.app X) ((RF.rightDerivedNatTrans R …
  -/
  dsimp only [rightDerivedNatTrans]
  /-
    C : Type u_1
    D : Type u_6
    H : Type u_3
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_6} D
    inst✝² : CategoryTheory.Category.{u_2, u_3} H
    RF RF' : CategoryTheory.Functor D H
    F F' : CategoryTheory.Functor C H
    L : CategoryTheory.Functor C D
    α : Quiver.Hom F (L.comp RF)
    α' : Quiver.Hom F' (L.comp RF')
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : RF.IsRightDerivedFunctor α W
    τ : Quiver.Hom F F'
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (α.app X) ((RF.rightDerivedDesc α W R …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
lemma rightDerivedNatTrans_id :
    rightDerivedNatTrans RF RF α α W (𝟙 F) = 𝟙 RF :=
                                    /-
                                      C : Type u_5
                                      D : Type u_1
                                      H : Type u_3
                                      inst✝⁴ : CategoryTheory.Category.{u_6, u_5} C
                                      inst✝³ : CategoryTheory.Category.{u_4, u_1} D
                                      inst✝² : CategoryTheory.Category.{u_2, u_3} H
                                      RF : CategoryTheory.Functor D H
                                      F : CategoryTheory.Functor C H
                                      L : CategoryTheory.Functor C D
                                      α : Quiver.Hom F (L.comp RF)
                                      W : CategoryTheory.MorphismProperty C
                                      inst✝¹ : L.IsLocalization W
                                      inst✝ : RF.IsRightDerivedFunctor α W
                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.whiskerLeft L (RF.r …
                                    -/
  rightDerived_ext RF α W _ _ _ (by aesop_cat)
                                    /-
                                      🎉 no goals
                                    -/


@[reassoc (attr := simp)]
lemma rightDerivedNatTrans_comp (τ : F ⟶ F') (τ' : F' ⟶ F'') :
    rightDerivedNatTrans RF RF' α α' W τ ≫ rightDerivedNatTrans RF' RF'' α' α'' W τ' =
    rightDerivedNatTrans RF RF'' α α'' W (τ ≫ τ') :=
                                    /-
                                      C : Type u_1
                                      D : Type u_5
                                      H : Type u_3
                                      inst✝⁵ : CategoryTheory.Category.{u_4, u_1} C
                                      inst✝⁴ : CategoryTheory.Category.{u_6, u_5} D
                                      inst✝³ : CategoryTheory.Category.{u_2, u_3} H
                                      RF RF' RF'' : CategoryTheory.Functor D H
                                      F F' F'' : CategoryTheory.Functor C H
                                      L : CategoryTheory.Functor C D
                                      α : Quiver.Hom F (L.comp RF)
                                      α' : Quiver.Hom F' (L.comp RF')
                                      α'' : Quiver.Hom F'' (L.comp RF'')
                                      W : CategoryTheory.MorphismProperty C
                                      inst✝² : L.IsLocalization W
                                      inst✝¹ : RF.IsRightDerivedFunctor α W
                                      inst✝ : RF'.IsRightDerivedFunctor α' W
                                      τ : Quiver.Hom F F'
                                      τ' : Quiver.Hom F' F''
                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp α (CategoryTheory.whiskerLeft L (Cate …
                                    -/
  rightDerived_ext RF α W _ _ _ (by aesop_cat)
                                    /-
                                      🎉 no goals
                                    -/


/-- The natural isomorphism `RF ≅ RF'` on right derived functors that is
induced by a natural isomorphism `F ≅ F'`. -/
@[simps]
noncomputable def rightDerivedNatIso (τ : F ≅ F') :
    RF ≅ RF' where
  hom := rightDerivedNatTrans RF RF' α α' W τ.hom
  inv := rightDerivedNatTrans RF' RF α' α W τ.inv


/-- Uniqueness (up to a natural isomorphism) of the right derived functor. -/
noncomputable abbrev rightDerivedUnique [RF'.IsRightDerivedFunctor α'₂ W] : RF ≅ RF' :=
  rightDerivedNatIso RF RF' α α'₂ W (Iso.refl F)


lemma isRightDerivedFunctor_iff_isIso_rightDerivedDesc (G : D ⥤ H) (β : F ⟶ L ⋙ G) :
    G.IsRightDerivedFunctor β W ↔ IsIso (RF.rightDerivedDesc α W G β) := by
  /-
    C : Type u_5
    D : Type u_3
    H : Type u_4
    inst✝⁴ : CategoryTheory.Category.{u_6, u_5} C
    inst✝³ : CategoryTheory.Category.{u_1, u_3} D
    inst✝² : CategoryTheory.Category.{u_2, u_4} H
    RF : CategoryTheory.Functor D H
    F : CategoryTheory.Functor C H
    L : CategoryTheory.Functor C D
    α : Quiver.Hom F (L.comp RF)
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : RF.IsRightDerivedFunctor α W
    G : CategoryTheory.Functor D H
    β : Quiver.Hom F (L.comp G)
    ⊢ Iff (G.IsRightDerivedFunctor β W) (CategoryTheory.IsIso (RF.rightDerivedDesc …
  -/
  rw [isRightDerivedFunctor_iff_isLeftKanExtension]
  /-
    C : Type u_5
    D : Type u_3
    H : Type u_4
    inst✝⁴ : CategoryTheory.Category.{u_6, u_5} C
    inst✝³ : CategoryTheory.Category.{u_1, u_3} D
    inst✝² : CategoryTheory.Category.{u_2, u_4} H
    RF : CategoryTheory.Functor D H
    F : CategoryTheory.Functor C H
    L : CategoryTheory.Functor C D
    α : Quiver.Hom F (L.comp RF)
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : RF.IsRightDerivedFunctor α W
    G : CategoryTheory.Functor D H
    β : Quiver.Hom F (L.comp G)
    ⊢ Iff (G.IsLeftKanExtension β) (CategoryTheory.IsIso (RF.rightDerivedDesc α W  …
  -/
  have := IsRightDerivedFunctor.isLeftKanExtension _ α W
  /-
    C : Type u_5
    D : Type u_3
    H : Type u_4
    inst✝⁴ : CategoryTheory.Category.{u_6, u_5} C
    inst✝³ : CategoryTheory.Category.{u_1, u_3} D
    inst✝² : CategoryTheory.Category.{u_2, u_4} H
    RF : CategoryTheory.Functor D H
    F : CategoryTheory.Functor C H
    L : CategoryTheory.Functor C D
    α : Quiver.Hom F (L.comp RF)
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : RF.IsRightDerivedFunctor α W
    G : CategoryTheory.Functor D H
    β : Quiver.Hom F (L.comp G)
    this : RF.IsLeftKanExtension α
    ⊢ Iff (G.IsLeftKanExtension β) (CategoryTheory.IsIso (RF.rightDerivedDesc α W  …
  -/
  exact isLeftKanExtension_iff_isIso _ α _ (by simp)
  /-
    🎉 no goals
  -/


/-- A functor `F : C ⥤ H` has a right derived functor with respect to
`W : MorphismProperty C` if it has a left Kan extension along
`W.Q : C ⥤ W.Localization` (or any localization functor `L : C ⥤ D`
for `W`, see `hasRightDerivedFunctor_iff`). -/
class HasRightDerivedFunctor : Prop where
  hasLeftKanExtension' : HasLeftKanExtension W.Q F


lemma hasRightDerivedFunctor_iff :
    F.HasRightDerivedFunctor W ↔ HasLeftKanExtension L F := by
  have : HasRightDerivedFunctor F W ↔ HasLeftKanExtension W.Q F :=
    ⟨fun h => h.hasLeftKanExtension', fun h => ⟨h⟩⟩
  /-
    C : Type u_1
    D : Type u_5
    H : Type u_2
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Category.{u_6, u_5} D
    inst✝¹ : CategoryTheory.Category.{u_4, u_2} H
    F : CategoryTheory.Functor C H
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝ : L.IsLocalization W
    this : Iff (F.HasRightDerivedFunctor W) (W.Q.HasLeftKanExtension F)
    ⊢ Iff (F.HasRightDerivedFunctor W) (L.HasLeftKanExtension F)
  -/
  rw [this, hasLeftExtension_iff_postcomp₁ (Localization.compUniqFunctor W.Q L W) F]
  /-
    🎉 no goals
  -/


include e in
lemma hasRightDerivedFunctor_iff_of_iso :
    HasRightDerivedFunctor F W ↔ HasRightDerivedFunctor F' W := by
  rw [hasRightDerivedFunctor_iff F W.Q W, hasRightDerivedFunctor_iff F' W.Q W,
    hasLeftExtension_iff_of_iso₂ W.Q e]


lemma HasRightDerivedFunctor.hasLeftKanExtension [HasRightDerivedFunctor F W] :
    HasLeftKanExtension L F := by
  /-
    C : Type u_1
    D : Type u_5
    H : Type u_2
    inst✝⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝³ : CategoryTheory.Category.{u_6, u_5} D
    inst✝² : CategoryTheory.Category.{u_4, u_2} H
    F : CategoryTheory.Functor C H
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : F.HasRightDerivedFunctor W
    ⊢ L.HasLeftKanExtension F
  -/
  simpa only [← hasRightDerivedFunctor_iff F L W]
  /-
    🎉 no goals
  -/


lemma HasRightDerivedFunctor.mk' [RF.IsRightDerivedFunctor α W] :
    HasRightDerivedFunctor F W := by
  /-
    C : Type u_1
    D : Type u_2
    H : Type u_3
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_2} D
    inst✝² : CategoryTheory.Category.{u_6, u_3} H
    RF : CategoryTheory.Functor D H
    F : CategoryTheory.Functor C H
    L : CategoryTheory.Functor C D
    α : Quiver.Hom F (L.comp RF)
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : RF.IsRightDerivedFunctor α W
    ⊢ F.HasRightDerivedFunctor W
  -/
  have := IsRightDerivedFunctor.isLeftKanExtension RF α W
  /-
    C : Type u_1
    D : Type u_2
    H : Type u_3
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Category.{u_5, u_2} D
    inst✝² : CategoryTheory.Category.{u_6, u_3} H
    RF : CategoryTheory.Functor D H
    F : CategoryTheory.Functor C H
    L : CategoryTheory.Functor C D
    α : Quiver.Hom F (L.comp RF)
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : RF.IsRightDerivedFunctor α W
    this : RF.IsLeftKanExtension α
    ⊢ F.HasRightDerivedFunctor W
  -/
  simpa only [hasRightDerivedFunctor_iff F L W] using HasLeftKanExtension.mk RF α
  /-
    🎉 no goals
  -/


/-- Given a functor `F : C ⥤ H`, and a localization functor `L : D ⥤ H` for `W`,
this is the right derived functor `D ⥤ H` of `F`, i.e. the left Kan extension
of `F` along `L`. -/
noncomputable def totalRightDerived : D ⥤ H :=
  have := HasRightDerivedFunctor.hasLeftKanExtension F L W
  leftKanExtension L F


/-- The canonical natural transformation `F ⟶ L ⋙ F.totalRightDerived L W`. -/
noncomputable def totalRightDerivedUnit : F ⟶ L ⋙ F.totalRightDerived L W :=
  have := HasRightDerivedFunctor.hasLeftKanExtension F L W
  leftKanExtensionUnit L F


instance : (F.totalRightDerived L W).IsRightDerivedFunctor
    (F.totalRightDerivedUnit L W) W where
  isLeftKanExtension' := by
    /-
      C : Type u_1
      C' : Type ?u.104458
      D : Type u_2
      D' : Type ?u.104464
      H : Type u_3
      H' : Type ?u.104470
      inst✝⁷ : CategoryTheory.Category.{u_4, u_1} C
      inst✝⁶ : CategoryTheory.Category.{?u.104478, ?u.104458} C'
      inst✝⁵ : CategoryTheory.Category.{u_5, u_2} D
      inst✝⁴ : CategoryTheory.Category.{?u.104486, ?u.104464} D'
      inst✝³ : CategoryTheory.Category.{u_6, u_3} H
      inst✝² : CategoryTheory.Category.{?u.104494, ?u.104470} H'
      RF RF' RF'' : CategoryTheory.Functor D H
      F F' F'' : CategoryTheory.Functor C H
      e : CategoryTheory.Iso F F'
      L : CategoryTheory.Functor C D
      α : Quiver.Hom F (L.comp RF)
      α' : Quiver.Hom F' (L.comp RF')
      α'' : Quiver.Hom F'' (L.comp RF'')
      α'₂ : Quiver.Hom F (L.comp RF')
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : L.IsLocalization W
      inst✝ : F.HasRightDerivedFunctor W
      ⊢ (CategoryTheory.Functor.totalRightDerived L W).IsLeftKanExtension (CategoryT …
    -/
    dsimp [totalRightDerived, totalRightDerivedUnit]
    /-
      C : Type u_1
      C' : Type ?u.104458
      D : Type u_2
      D' : Type ?u.104464
      H : Type u_3
      H' : Type ?u.104470
      inst✝⁷ : CategoryTheory.Category.{u_4, u_1} C
      inst✝⁶ : CategoryTheory.Category.{?u.104478, ?u.104458} C'
      inst✝⁵ : CategoryTheory.Category.{u_5, u_2} D
      inst✝⁴ : CategoryTheory.Category.{?u.104486, ?u.104464} D'
      inst✝³ : CategoryTheory.Category.{u_6, u_3} H
      inst✝² : CategoryTheory.Category.{?u.104494, ?u.104470} H'
      RF RF' RF'' : CategoryTheory.Functor D H
      F F' F'' : CategoryTheory.Functor C H
      e : CategoryTheory.Iso F F'
      L : CategoryTheory.Functor C D
      α : Quiver.Hom F (L.comp RF)
      α' : Quiver.Hom F' (L.comp RF')
      α'' : Quiver.Hom F'' (L.comp RF'')
      α'₂ : Quiver.Hom F (L.comp RF')
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : L.IsLocalization W
      inst✝ : F.HasRightDerivedFunctor W
      ⊢ (L.leftKanExtension F).IsLeftKanExtension (L.leftKanExtensionUnit F)
    -/
    infer_instance
    /-
      🎉 no goals
    -/


