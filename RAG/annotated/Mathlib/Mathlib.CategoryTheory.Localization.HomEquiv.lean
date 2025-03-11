/-- If `Φ : LocalizerMorphism W₁ W₂` is a morphism of localizers, `L₁` and `L₂`
are localization functors for `W₁` and `W₂`, then this is the induced map
`(L₁.obj X ⟶ L₁.obj Y) ⟶ (L₂.obj (Φ.functor.obj X) ⟶ L₂.obj (Φ.functor.obj Y))`
for all objects `X` and `Y`. -/
noncomputable def homMap (f : L₁.obj X ⟶ L₁.obj Y) :
    L₂.obj (Φ.functor.obj X) ⟶ L₂.obj (Φ.functor.obj Y) :=
  Iso.homCongr ((CatCommSq.iso _ _ _ _).symm.app _) ((CatCommSq.iso _ _ _ _).symm.app _)
    ((Φ.localizedFunctor L₁ L₂).map f)


@[simp]
lemma homMap_map (f : X ⟶ Y) :
    Φ.homMap L₁ L₂ (L₁.map f) = L₂.map (Φ.functor.map f) := by
  /-
    C₁ : Type u_2
    C₂ : Type u_3
    D₁ : Type u_5
    D₂ : Type u_6
    inst✝⁵ : CategoryTheory.Category.{u_8, u_2} C₁
    inst✝⁴ : CategoryTheory.Category.{u_10, u_3} C₂
    inst✝³ : CategoryTheory.Category.{u_11, u_5} D₁
    inst✝² : CategoryTheory.Category.{u_9, u_6} D₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    L₁ : CategoryTheory.Functor C₁ D₁
    inst✝¹ : L₁.IsLocalization W₁
    L₂ : CategoryTheory.Functor C₂ D₂
    inst✝ : L₂.IsLocalization W₂
    X Y : C₁
    f : Quiver.Hom X Y
    ⊢ Eq (Φ.homMap L₁ L₂ (L₁.map f)) (L₂.map (Φ.functor.map f))
  -/
  dsimp [homMap]
  /-
    C₁ : Type u_2
    C₂ : Type u_3
    D₁ : Type u_5
    D₂ : Type u_6
    inst✝⁵ : CategoryTheory.Category.{u_8, u_2} C₁
    inst✝⁴ : CategoryTheory.Category.{u_10, u_3} C₂
    inst✝³ : CategoryTheory.Category.{u_11, u_5} D₁
    inst✝² : CategoryTheory.Category.{u_9, u_6} D₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    L₁ : CategoryTheory.Functor C₁ D₁
    inst✝¹ : L₁.IsLocalization W₁
    L₂ : CategoryTheory.Functor C₂ D₂
    inst✝ : L₂.IsLocalization W₂
    X Y : C₁
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.CatCommSq.iso Φ.func …
  -/
  erw [← NatTrans.naturality_assoc]
  /-
    C₁ : Type u_2
    C₂ : Type u_3
    D₁ : Type u_5
    D₂ : Type u_6
    inst✝⁵ : CategoryTheory.Category.{u_8, u_2} C₁
    inst✝⁴ : CategoryTheory.Category.{u_10, u_3} C₂
    inst✝³ : CategoryTheory.Category.{u_11, u_5} D₁
    inst✝² : CategoryTheory.Category.{u_9, u_6} D₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    L₁ : CategoryTheory.Functor C₁ D₁
    inst✝¹ : L₁.IsLocalization W₁
    L₂ : CategoryTheory.Functor C₂ D₂
    inst✝ : L₂.IsLocalization W₂
    X Y : C₁
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((Φ.functor.comp L₂).map f) (Category …
  -/
  simp
  /-
    🎉 no goals
  -/


variable (X) in
@[simp]
lemma homMap_id  :
    Φ.homMap L₁ L₂ (𝟙 (L₁.obj X)) = 𝟙 (L₂.obj (Φ.functor.obj X)) := by
  /-
    C₁ : Type u_2
    C₂ : Type u_3
    D₁ : Type u_5
    D₂ : Type u_6
    inst✝⁵ : CategoryTheory.Category.{u_10, u_2} C₁
    inst✝⁴ : CategoryTheory.Category.{u_9, u_3} C₂
    inst✝³ : CategoryTheory.Category.{u_11, u_5} D₁
    inst✝² : CategoryTheory.Category.{u_8, u_6} D₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    L₁ : CategoryTheory.Functor C₁ D₁
    inst✝¹ : L₁.IsLocalization W₁
    L₂ : CategoryTheory.Functor C₂ D₂
    inst✝ : L₂.IsLocalization W₂
    X : C₁
    ⊢ Eq (Φ.homMap L₁ L₂ (CategoryTheory.CategoryStruct.id (L₁.obj X))) (CategoryT …
  -/
  simpa using Φ.homMap_map L₁ L₂ (𝟙 X)
  /-
    🎉 no goals
  -/


@[reassoc]
lemma homMap_comp (f : L₁.obj X ⟶ L₁.obj Y) (g : L₁.obj Y ⟶ L₁.obj Z) :
    Φ.homMap L₁ L₂ (f ≫ g) = Φ.homMap L₁ L₂ f ≫ Φ.homMap L₁ L₂ g := by
  /-
    C₁ : Type u_2
    C₂ : Type u_3
    D₁ : Type u_5
    D₂ : Type u_6
    inst✝⁵ : CategoryTheory.Category.{u_9, u_2} C₁
    inst✝⁴ : CategoryTheory.Category.{u_11, u_3} C₂
    inst✝³ : CategoryTheory.Category.{u_8, u_5} D₁
    inst✝² : CategoryTheory.Category.{u_10, u_6} D₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    L₁ : CategoryTheory.Functor C₁ D₁
    inst✝¹ : L₁.IsLocalization W₁
    L₂ : CategoryTheory.Functor C₂ D₂
    inst✝ : L₂.IsLocalization W₂
    X Y Z : C₁
    f : Quiver.Hom (L₁.obj X) (L₁.obj Y)
    g : Quiver.Hom (L₁.obj Y) (L₁.obj Z)
    ⊢ Eq (Φ.homMap L₁ L₂ (CategoryTheory.CategoryStruct.comp f g)) (CategoryTheory …
  -/
  simp [homMap]
  /-
    🎉 no goals
  -/


@[reassoc]
lemma homMap_apply (G : D₁ ⥤ D₂) (e : Φ.functor ⋙ L₂ ≅ L₁ ⋙ G) (f : L₁.obj X ⟶ L₁.obj Y) :
    Φ.homMap L₁ L₂ f = e.hom.app X ≫ G.map f ≫ e.inv.app Y := by
  /-
    C₁ : Type u_2
    C₂ : Type u_3
    D₁ : Type u_5
    D₂ : Type u_6
    inst✝⁵ : CategoryTheory.Category.{u_10, u_2} C₁
    inst✝⁴ : CategoryTheory.Category.{u_11, u_3} C₂
    inst✝³ : CategoryTheory.Category.{u_8, u_5} D₁
    inst✝² : CategoryTheory.Category.{u_9, u_6} D₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    L₁ : CategoryTheory.Functor C₁ D₁
    inst✝¹ : L₁.IsLocalization W₁
    L₂ : CategoryTheory.Functor C₂ D₂
    inst✝ : L₂.IsLocalization W₂
    X Y : C₁
    G : CategoryTheory.Functor D₁ D₂
    e : CategoryTheory.Iso (Φ.functor.comp L₂) (L₁.comp G)
    f : Quiver.Hom (L₁.obj X) (L₁.obj Y)
    ⊢ Eq (Φ.homMap L₁ L₂ f) (CategoryTheory.CategoryStruct.comp (e.hom.app X) (Cat …
  -/
  let G' := Φ.localizedFunctor L₁ L₂
  /-
    C₁ : Type u_2
    C₂ : Type u_3
    D₁ : Type u_5
    D₂ : Type u_6
    inst✝⁵ : CategoryTheory.Category.{u_10, u_2} C₁
    inst✝⁴ : CategoryTheory.Category.{u_11, u_3} C₂
    inst✝³ : CategoryTheory.Category.{u_8, u_5} D₁
    inst✝² : CategoryTheory.Category.{u_9, u_6} D₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    L₁ : CategoryTheory.Functor C₁ D₁
    inst✝¹ : L₁.IsLocalization W₁
    L₂ : CategoryTheory.Functor C₂ D₂
    inst✝ : L₂.IsLocalization W₂
    X Y : C₁
    G : CategoryTheory.Functor D₁ D₂
    e : CategoryTheory.Iso (Φ.functor.comp L₂) (L₁.comp G)
    f : Quiver.Hom (L₁.obj X) (L₁.obj Y)
    G' : CategoryTheory.Functor D₁ D₂ := Φ.localizedFunctor L₁ L₂
    ⊢ Eq (Φ.homMap L₁ L₂ f) (CategoryTheory.CategoryStruct.comp (e.hom.app X) (Cat …
  -/
  let e' := CatCommSq.iso Φ.functor L₁ L₂ G'
  /-
    C₁ : Type u_2
    C₂ : Type u_3
    D₁ : Type u_5
    D₂ : Type u_6
    inst✝⁵ : CategoryTheory.Category.{u_10, u_2} C₁
    inst✝⁴ : CategoryTheory.Category.{u_11, u_3} C₂
    inst✝³ : CategoryTheory.Category.{u_8, u_5} D₁
    inst✝² : CategoryTheory.Category.{u_9, u_6} D₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    L₁ : CategoryTheory.Functor C₁ D₁
    inst✝¹ : L₁.IsLocalization W₁
    L₂ : CategoryTheory.Functor C₂ D₂
    inst✝ : L₂.IsLocalization W₂
    X Y : C₁
    G : CategoryTheory.Functor D₁ D₂
    e : CategoryTheory.Iso (Φ.functor.comp L₂) (L₁.comp G)
    f : Quiver.Hom (L₁.obj X) (L₁.obj Y)
    G' : CategoryTheory.Functor D₁ D₂ := Φ.localizedFunctor L₁ L₂
    e' : CategoryTheory.Iso (Φ.functor.comp L₂) (L₁.comp G') := CategoryTheory.Cat …
    ⊢ Eq (Φ.homMap L₁ L₂ f) (CategoryTheory.CategoryStruct.comp (e.hom.app X) (Cat …
  -/
  change e'.hom.app X ≫ G'.map f ≫ e'.inv.app Y = _
  /-
    C₁ : Type u_2
    C₂ : Type u_3
    D₁ : Type u_5
    D₂ : Type u_6
    inst✝⁵ : CategoryTheory.Category.{u_10, u_2} C₁
    inst✝⁴ : CategoryTheory.Category.{u_11, u_3} C₂
    inst✝³ : CategoryTheory.Category.{u_8, u_5} D₁
    inst✝² : CategoryTheory.Category.{u_9, u_6} D₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    L₁ : CategoryTheory.Functor C₁ D₁
    inst✝¹ : L₁.IsLocalization W₁
    L₂ : CategoryTheory.Functor C₂ D₂
    inst✝ : L₂.IsLocalization W₂
    X Y : C₁
    G : CategoryTheory.Functor D₁ D₂
    e : CategoryTheory.Iso (Φ.functor.comp L₂) (L₁.comp G)
    f : Quiver.Hom (L₁.obj X) (L₁.obj Y)
    G' : CategoryTheory.Functor D₁ D₂ := Φ.localizedFunctor L₁ L₂
    e' : CategoryTheory.Iso (Φ.functor.comp L₂) (L₁.comp G') := CategoryTheory.Cat …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (e'.hom.app X) (CategoryTheory.Catego …
  -/
  letI : Localization.Lifting L₁ W₁ (Φ.functor ⋙ L₂) G := ⟨e.symm⟩
  /-
    C₁ : Type u_2
    C₂ : Type u_3
    D₁ : Type u_5
    D₂ : Type u_6
    inst✝⁵ : CategoryTheory.Category.{u_10, u_2} C₁
    inst✝⁴ : CategoryTheory.Category.{u_11, u_3} C₂
    inst✝³ : CategoryTheory.Category.{u_8, u_5} D₁
    inst✝² : CategoryTheory.Category.{u_9, u_6} D₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    L₁ : CategoryTheory.Functor C₁ D₁
    inst✝¹ : L₁.IsLocalization W₁
    L₂ : CategoryTheory.Functor C₂ D₂
    inst✝ : L₂.IsLocalization W₂
    X Y : C₁
    G : CategoryTheory.Functor D₁ D₂
    e : CategoryTheory.Iso (Φ.functor.comp L₂) (L₁.comp G)
    f : Quiver.Hom (L₁.obj X) (L₁.obj Y)
    G' : CategoryTheory.Functor D₁ D₂ := Φ.localizedFunctor L₁ L₂
    e' : CategoryTheory.Iso (Φ.functor.comp L₂) (L₁.comp G') := CategoryTheory.Cat …
    this : CategoryTheory.Localization.Lifting L₁ W₁ (Φ.functor.comp L₂) G := { is …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (e'.hom.app X) (CategoryTheory.Catego …
  -/
  let α : G' ≅ G := Localization.liftNatIso L₁ W₁ (L₁ ⋙ G') (Φ.functor ⋙ L₂) _ _ e'.symm
  have : e = e' ≪≫ isoWhiskerLeft _ α := by
    ext X
    dsimp [α]
    rw [Localization.liftNatTrans_app]
    erw [id_comp]
    rw [Iso.hom_inv_id_app_assoc]
    rfl
  /-
    C₁ : Type u_2
    C₂ : Type u_3
    D₁ : Type u_5
    D₂ : Type u_6
    inst✝⁵ : CategoryTheory.Category.{u_10, u_2} C₁
    inst✝⁴ : CategoryTheory.Category.{u_11, u_3} C₂
    inst✝³ : CategoryTheory.Category.{u_8, u_5} D₁
    inst✝² : CategoryTheory.Category.{u_9, u_6} D₂
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    L₁ : CategoryTheory.Functor C₁ D₁
    inst✝¹ : L₁.IsLocalization W₁
    L₂ : CategoryTheory.Functor C₂ D₂
    inst✝ : L₂.IsLocalization W₂
    X Y : C₁
    G : CategoryTheory.Functor D₁ D₂
    e : CategoryTheory.Iso (Φ.functor.comp L₂) (L₁.comp G)
    f : Quiver.Hom (L₁.obj X) (L₁.obj Y)
    G' : CategoryTheory.Functor D₁ D₂ := Φ.localizedFunctor L₁ L₂
    e' : CategoryTheory.Iso (Φ.functor.comp L₂) (L₁.comp G') := CategoryTheory.Cat …
    this✝ : CategoryTheory.Localization.Lifting L₁ W₁ (Φ.functor.comp L₂) G := { i …
    α : CategoryTheory.Iso G' G := CategoryTheory.Localization.liftNatIso L₁ W₁ (L …
    this : Eq e (e'.trans (CategoryTheory.isoWhiskerLeft L₁ α))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (e'.hom.app X) (CategoryTheory.Catego …
  -/
  simp [this]
  /-
    🎉 no goals
  -/


@[simp]
lemma id_homMap (f : L₁.obj X ⟶ L₁.obj Y) :
    (id W₁).homMap L₁ L₁ f = f := by
  /-
    C₁ : Type u_2
    D₁ : Type u_5
    inst✝² : CategoryTheory.Category.{u_9, u_2} C₁
    inst✝¹ : CategoryTheory.Category.{u_8, u_5} D₁
    W₁ : CategoryTheory.MorphismProperty C₁
    L₁ : CategoryTheory.Functor C₁ D₁
    inst✝ : L₁.IsLocalization W₁
    X Y : C₁
    f : Quiver.Hom (L₁.obj X) (L₁.obj Y)
    ⊢ Eq ((CategoryTheory.LocalizerMorphism.id W₁).homMap L₁ L₁ f) f
  -/
  simpa using (id W₁).homMap_apply L₁ L₁ (𝟭 D₁) (Iso.refl _) f
  /-
    🎉 no goals
  -/


@[simp]
lemma homMap_homMap (f : L₁.obj X ⟶ L₁.obj Y) :
    Ψ.homMap L₂ L₃ (Φ.homMap L₁ L₂ f) = (Φ.comp Ψ).homMap L₁ L₃ f := by
  /-
    C₁ : Type u_2
    C₂ : Type u_3
    C₃ : Type u_4
    D₁ : Type u_5
    D₂ : Type u_6
    D₃ : Type u_7
    inst✝⁸ : CategoryTheory.Category.{u_9, u_2} C₁
    inst✝⁷ : CategoryTheory.Category.{u_12, u_3} C₂
    inst✝⁶ : CategoryTheory.Category.{u_11, u_4} C₃
    inst✝⁵ : CategoryTheory.Category.{u_8, u_5} D₁
    inst✝⁴ : CategoryTheory.Category.{u_13, u_6} D₂
    inst✝³ : CategoryTheory.Category.{u_10, u_7} D₃
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    W₃ : CategoryTheory.MorphismProperty C₃
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    Ψ : CategoryTheory.LocalizerMorphism W₂ W₃
    L₁ : CategoryTheory.Functor C₁ D₁
    inst✝² : L₁.IsLocalization W₁
    L₂ : CategoryTheory.Functor C₂ D₂
    inst✝¹ : L₂.IsLocalization W₂
    L₃ : CategoryTheory.Functor C₃ D₃
    inst✝ : L₃.IsLocalization W₃
    X Y : C₁
    f : Quiver.Hom (L₁.obj X) (L₁.obj Y)
    ⊢ Eq (Ψ.homMap L₂ L₃ (Φ.homMap L₁ L₂ f)) ((Φ.comp Ψ).homMap L₁ L₃ f)
  -/
  let G := Φ.localizedFunctor L₁ L₂
  /-
    C₁ : Type u_2
    C₂ : Type u_3
    C₃ : Type u_4
    D₁ : Type u_5
    D₂ : Type u_6
    D₃ : Type u_7
    inst✝⁸ : CategoryTheory.Category.{u_9, u_2} C₁
    inst✝⁷ : CategoryTheory.Category.{u_12, u_3} C₂
    inst✝⁶ : CategoryTheory.Category.{u_11, u_4} C₃
    inst✝⁵ : CategoryTheory.Category.{u_8, u_5} D₁
    inst✝⁴ : CategoryTheory.Category.{u_13, u_6} D₂
    inst✝³ : CategoryTheory.Category.{u_10, u_7} D₃
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    W₃ : CategoryTheory.MorphismProperty C₃
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    Ψ : CategoryTheory.LocalizerMorphism W₂ W₃
    L₁ : CategoryTheory.Functor C₁ D₁
    inst✝² : L₁.IsLocalization W₁
    L₂ : CategoryTheory.Functor C₂ D₂
    inst✝¹ : L₂.IsLocalization W₂
    L₃ : CategoryTheory.Functor C₃ D₃
    inst✝ : L₃.IsLocalization W₃
    X Y : C₁
    f : Quiver.Hom (L₁.obj X) (L₁.obj Y)
    G : CategoryTheory.Functor D₁ D₂ := Φ.localizedFunctor L₁ L₂
    ⊢ Eq (Ψ.homMap L₂ L₃ (Φ.homMap L₁ L₂ f)) ((Φ.comp Ψ).homMap L₁ L₃ f)
  -/
  let G' := Ψ.localizedFunctor L₂ L₃
  /-
    C₁ : Type u_2
    C₂ : Type u_3
    C₃ : Type u_4
    D₁ : Type u_5
    D₂ : Type u_6
    D₃ : Type u_7
    inst✝⁸ : CategoryTheory.Category.{u_9, u_2} C₁
    inst✝⁷ : CategoryTheory.Category.{u_12, u_3} C₂
    inst✝⁶ : CategoryTheory.Category.{u_11, u_4} C₃
    inst✝⁵ : CategoryTheory.Category.{u_8, u_5} D₁
    inst✝⁴ : CategoryTheory.Category.{u_13, u_6} D₂
    inst✝³ : CategoryTheory.Category.{u_10, u_7} D₃
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    W₃ : CategoryTheory.MorphismProperty C₃
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    Ψ : CategoryTheory.LocalizerMorphism W₂ W₃
    L₁ : CategoryTheory.Functor C₁ D₁
    inst✝² : L₁.IsLocalization W₁
    L₂ : CategoryTheory.Functor C₂ D₂
    inst✝¹ : L₂.IsLocalization W₂
    L₃ : CategoryTheory.Functor C₃ D₃
    inst✝ : L₃.IsLocalization W₃
    X Y : C₁
    f : Quiver.Hom (L₁.obj X) (L₁.obj Y)
    G : CategoryTheory.Functor D₁ D₂ := Φ.localizedFunctor L₁ L₂
    G' : CategoryTheory.Functor D₂ D₃ := Ψ.localizedFunctor L₂ L₃
    ⊢ Eq (Ψ.homMap L₂ L₃ (Φ.homMap L₁ L₂ f)) ((Φ.comp Ψ).homMap L₁ L₃ f)
  -/
  let e : Φ.functor ⋙ L₂ ≅ L₁ ⋙ G := CatCommSq.iso _ _ _ _
  /-
    C₁ : Type u_2
    C₂ : Type u_3
    C₃ : Type u_4
    D₁ : Type u_5
    D₂ : Type u_6
    D₃ : Type u_7
    inst✝⁸ : CategoryTheory.Category.{u_9, u_2} C₁
    inst✝⁷ : CategoryTheory.Category.{u_12, u_3} C₂
    inst✝⁶ : CategoryTheory.Category.{u_11, u_4} C₃
    inst✝⁵ : CategoryTheory.Category.{u_8, u_5} D₁
    inst✝⁴ : CategoryTheory.Category.{u_13, u_6} D₂
    inst✝³ : CategoryTheory.Category.{u_10, u_7} D₃
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    W₃ : CategoryTheory.MorphismProperty C₃
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    Ψ : CategoryTheory.LocalizerMorphism W₂ W₃
    L₁ : CategoryTheory.Functor C₁ D₁
    inst✝² : L₁.IsLocalization W₁
    L₂ : CategoryTheory.Functor C₂ D₂
    inst✝¹ : L₂.IsLocalization W₂
    L₃ : CategoryTheory.Functor C₃ D₃
    inst✝ : L₃.IsLocalization W₃
    X Y : C₁
    f : Quiver.Hom (L₁.obj X) (L₁.obj Y)
    G : CategoryTheory.Functor D₁ D₂ := Φ.localizedFunctor L₁ L₂
    G' : CategoryTheory.Functor D₂ D₃ := Ψ.localizedFunctor L₂ L₃
    e : CategoryTheory.Iso (Φ.functor.comp L₂) (L₁.comp G) := CategoryTheory.CatCo …
    ⊢ Eq (Ψ.homMap L₂ L₃ (Φ.homMap L₁ L₂ f)) ((Φ.comp Ψ).homMap L₁ L₃ f)
  -/
  let e' : Ψ.functor ⋙ L₃ ≅ L₂ ⋙ G' := CatCommSq.iso _ _ _ _
  rw [Φ.homMap_apply L₁ L₂ G e, Ψ.homMap_apply L₂ L₃ G' e',
    (Φ.comp Ψ).homMap_apply L₁ L₃ (G ⋙ G')
      (Functor.associator _ _ _ ≪≫ isoWhiskerLeft _ e' ≪≫
      (Functor.associator _ _ _).symm ≪≫ isoWhiskerRight e _ ≪≫
      Functor.associator _ _ _)]
  /-
    C₁ : Type u_2
    C₂ : Type u_3
    C₃ : Type u_4
    D₁ : Type u_5
    D₂ : Type u_6
    D₃ : Type u_7
    inst✝⁸ : CategoryTheory.Category.{u_9, u_2} C₁
    inst✝⁷ : CategoryTheory.Category.{u_12, u_3} C₂
    inst✝⁶ : CategoryTheory.Category.{u_11, u_4} C₃
    inst✝⁵ : CategoryTheory.Category.{u_8, u_5} D₁
    inst✝⁴ : CategoryTheory.Category.{u_13, u_6} D₂
    inst✝³ : CategoryTheory.Category.{u_10, u_7} D₃
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    W₃ : CategoryTheory.MorphismProperty C₃
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    Ψ : CategoryTheory.LocalizerMorphism W₂ W₃
    L₁ : CategoryTheory.Functor C₁ D₁
    inst✝² : L₁.IsLocalization W₁
    L₂ : CategoryTheory.Functor C₂ D₂
    inst✝¹ : L₂.IsLocalization W₂
    L₃ : CategoryTheory.Functor C₃ D₃
    inst✝ : L₃.IsLocalization W₃
    X Y : C₁
    f : Quiver.Hom (L₁.obj X) (L₁.obj Y)
    G : CategoryTheory.Functor D₁ D₂ := Φ.localizedFunctor L₁ L₂
    G' : CategoryTheory.Functor D₂ D₃ := Ψ.localizedFunctor L₂ L₃
    e : CategoryTheory.Iso (Φ.functor.comp L₂) (L₁.comp G) := CategoryTheory.CatCo …
    e' : CategoryTheory.Iso (Ψ.functor.comp L₃) (L₂.comp G') := CategoryTheory.Cat …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (e'.hom.app (Φ.functor.obj X)) (Categ …
  -/
  dsimp
  /-
    C₁ : Type u_2
    C₂ : Type u_3
    C₃ : Type u_4
    D₁ : Type u_5
    D₂ : Type u_6
    D₃ : Type u_7
    inst✝⁸ : CategoryTheory.Category.{u_9, u_2} C₁
    inst✝⁷ : CategoryTheory.Category.{u_12, u_3} C₂
    inst✝⁶ : CategoryTheory.Category.{u_11, u_4} C₃
    inst✝⁵ : CategoryTheory.Category.{u_8, u_5} D₁
    inst✝⁴ : CategoryTheory.Category.{u_13, u_6} D₂
    inst✝³ : CategoryTheory.Category.{u_10, u_7} D₃
    W₁ : CategoryTheory.MorphismProperty C₁
    W₂ : CategoryTheory.MorphismProperty C₂
    W₃ : CategoryTheory.MorphismProperty C₃
    Φ : CategoryTheory.LocalizerMorphism W₁ W₂
    Ψ : CategoryTheory.LocalizerMorphism W₂ W₃
    L₁ : CategoryTheory.Functor C₁ D₁
    inst✝² : L₁.IsLocalization W₁
    L₂ : CategoryTheory.Functor C₂ D₂
    inst✝¹ : L₂.IsLocalization W₂
    L₃ : CategoryTheory.Functor C₃ D₃
    inst✝ : L₃.IsLocalization W₃
    X Y : C₁
    f : Quiver.Hom (L₁.obj X) (L₁.obj Y)
    G : CategoryTheory.Functor D₁ D₂ := Φ.localizedFunctor L₁ L₂
    G' : CategoryTheory.Functor D₂ D₃ := Ψ.localizedFunctor L₂ L₃
    e : CategoryTheory.Iso (Φ.functor.comp L₂) (L₁.comp G) := CategoryTheory.CatCo …
    e' : CategoryTheory.Iso (Ψ.functor.comp L₃) (L₂.comp G') := CategoryTheory.Cat …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (e'.hom.app (Φ.functor.obj X)) (Categ …
  -/
  simp only [Functor.map_comp, assoc, comp_id, id_comp]
  /-
    🎉 no goals
  -/


/-- Bijection between types of morphisms in two localized categories
for the same class of morphisms `W`. -/
@[simps (config := .lemmasOnly) apply]
noncomputable def homEquiv :
    (L₁.obj X ⟶ L₁.obj Y) ≃ (L₂.obj X ⟶ L₂.obj Y) where
  toFun := (LocalizerMorphism.id W).homMap L₁ L₂
  invFun := (LocalizerMorphism.id W).homMap L₂ L₁
  left_inv f := by
    /-
      C : Type u_1
      C₁ : Type u_2
      C₂ : Type u_3
      C₃ : Type u_4
      D₁ : Type u_5
      D₂ : Type u_6
      D₃ : Type u_7
      inst✝⁹ : CategoryTheory.Category.{?u.38249, u_1} C
      inst✝⁸ : CategoryTheory.Category.{?u.38253, u_2} C₁
      inst✝⁷ : CategoryTheory.Category.{?u.38257, u_3} C₂
      inst✝⁶ : CategoryTheory.Category.{?u.38261, u_4} C₃
      inst✝⁵ : CategoryTheory.Category.{?u.38265, u_5} D₁
      inst✝⁴ : CategoryTheory.Category.{?u.38269, u_6} D₂
      inst✝³ : CategoryTheory.Category.{?u.38273, u_7} D₃
      W : CategoryTheory.MorphismProperty C
      L₁ : CategoryTheory.Functor C D₁
      inst✝² : L₁.IsLocalization W
      L₂ : CategoryTheory.Functor C D₂
      inst✝¹ : L₂.IsLocalization W
      L₃ : CategoryTheory.Functor C D₃
      inst✝ : L₃.IsLocalization W
      X Y Z : C
      f : Quiver.Hom (L₁.obj X) (L₁.obj Y)
      ⊢ Eq ((CategoryTheory.LocalizerMorphism.id W).homMap L₂ L₁ ((CategoryTheory.Lo …
    -/
    rw [LocalizerMorphism.homMap_homMap]
    /-
      C : Type u_1
      C₁ : Type u_2
      C₂ : Type u_3
      C₃ : Type u_4
      D₁ : Type u_5
      D₂ : Type u_6
      D₃ : Type u_7
      inst✝⁹ : CategoryTheory.Category.{?u.38249, u_1} C
      inst✝⁸ : CategoryTheory.Category.{?u.38253, u_2} C₁
      inst✝⁷ : CategoryTheory.Category.{?u.38257, u_3} C₂
      inst✝⁶ : CategoryTheory.Category.{?u.38261, u_4} C₃
      inst✝⁵ : CategoryTheory.Category.{?u.38265, u_5} D₁
      inst✝⁴ : CategoryTheory.Category.{?u.38269, u_6} D₂
      inst✝³ : CategoryTheory.Category.{?u.38273, u_7} D₃
      W : CategoryTheory.MorphismProperty C
      L₁ : CategoryTheory.Functor C D₁
      inst✝² : L₁.IsLocalization W
      L₂ : CategoryTheory.Functor C D₂
      inst✝¹ : L₂.IsLocalization W
      L₃ : CategoryTheory.Functor C D₃
      inst✝ : L₃.IsLocalization W
      X Y Z : C
      f : Quiver.Hom (L₁.obj X) (L₁.obj Y)
      ⊢ Eq (((CategoryTheory.LocalizerMorphism.id W).comp (CategoryTheory.LocalizerM …
    -/
    apply LocalizerMorphism.id_homMap
    /-
      🎉 no goals
    -/
  right_inv g := by
    /-
      C : Type u_1
      C₁ : Type u_2
      C₂ : Type u_3
      C₃ : Type u_4
      D₁ : Type u_5
      D₂ : Type u_6
      D₃ : Type u_7
      inst✝⁹ : CategoryTheory.Category.{?u.38249, u_1} C
      inst✝⁸ : CategoryTheory.Category.{?u.38253, u_2} C₁
      inst✝⁷ : CategoryTheory.Category.{?u.38257, u_3} C₂
      inst✝⁶ : CategoryTheory.Category.{?u.38261, u_4} C₃
      inst✝⁵ : CategoryTheory.Category.{?u.38265, u_5} D₁
      inst✝⁴ : CategoryTheory.Category.{?u.38269, u_6} D₂
      inst✝³ : CategoryTheory.Category.{?u.38273, u_7} D₃
      W : CategoryTheory.MorphismProperty C
      L₁ : CategoryTheory.Functor C D₁
      inst✝² : L₁.IsLocalization W
      L₂ : CategoryTheory.Functor C D₂
      inst✝¹ : L₂.IsLocalization W
      L₃ : CategoryTheory.Functor C D₃
      inst✝ : L₃.IsLocalization W
      X Y Z : C
      g : Quiver.Hom (L₂.obj X) (L₂.obj Y)
      ⊢ Eq ((CategoryTheory.LocalizerMorphism.id W).homMap L₁ L₂ ((CategoryTheory.Lo …
    -/
    rw [LocalizerMorphism.homMap_homMap]
    /-
      C : Type u_1
      C₁ : Type u_2
      C₂ : Type u_3
      C₃ : Type u_4
      D₁ : Type u_5
      D₂ : Type u_6
      D₃ : Type u_7
      inst✝⁹ : CategoryTheory.Category.{?u.38249, u_1} C
      inst✝⁸ : CategoryTheory.Category.{?u.38253, u_2} C₁
      inst✝⁷ : CategoryTheory.Category.{?u.38257, u_3} C₂
      inst✝⁶ : CategoryTheory.Category.{?u.38261, u_4} C₃
      inst✝⁵ : CategoryTheory.Category.{?u.38265, u_5} D₁
      inst✝⁴ : CategoryTheory.Category.{?u.38269, u_6} D₂
      inst✝³ : CategoryTheory.Category.{?u.38273, u_7} D₃
      W : CategoryTheory.MorphismProperty C
      L₁ : CategoryTheory.Functor C D₁
      inst✝² : L₁.IsLocalization W
      L₂ : CategoryTheory.Functor C D₂
      inst✝¹ : L₂.IsLocalization W
      L₃ : CategoryTheory.Functor C D₃
      inst✝ : L₃.IsLocalization W
      X Y Z : C
      g : Quiver.Hom (L₂.obj X) (L₂.obj Y)
      ⊢ Eq (((CategoryTheory.LocalizerMorphism.id W).comp (CategoryTheory.LocalizerM …
    -/
    apply LocalizerMorphism.id_homMap
    /-
      🎉 no goals
    -/


@[simp]
lemma homEquiv_symm_apply (g : L₂.obj X ⟶ L₂.obj Y) :
    (homEquiv W L₁ L₂).symm g = homEquiv W L₂ L₁ g := rfl


lemma homEquiv_eq (G : D₁ ⥤ D₂) (e : L₁ ⋙ G ≅ L₂) (f : L₁.obj X ⟶ L₁.obj Y) :
    homEquiv W L₁ L₂ f = e.inv.app X ≫ G.map f ≫ e.hom.app Y := by
  rw [homEquiv_apply, LocalizerMorphism.homMap_apply (LocalizerMorphism.id W) L₁ L₂ G e.symm,
    Iso.symm_hom, Iso.symm_inv]


@[simp]
lemma homEquiv_refl (f : L₁.obj X ⟶ L₁.obj Y) :
    homEquiv W L₁ L₁ f = f := by
  /-
    C : Type u_1
    D₁ : Type u_5
    inst✝² : CategoryTheory.Category.{u_9, u_1} C
    inst✝¹ : CategoryTheory.Category.{u_8, u_5} D₁
    W : CategoryTheory.MorphismProperty C
    L₁ : CategoryTheory.Functor C D₁
    inst✝ : L₁.IsLocalization W
    X Y : C
    f : Quiver.Hom (L₁.obj X) (L₁.obj Y)
    ⊢ Eq ((CategoryTheory.Localization.homEquiv W L₁ L₁) f) f
  -/
  apply LocalizerMorphism.id_homMap
  /-
    🎉 no goals
  -/


lemma homEquiv_trans (f : L₁.obj X ⟶ L₁.obj Y) :
    homEquiv W L₂ L₃ (homEquiv W L₁ L₂ f) = homEquiv W L₁ L₃ f := by
  /-
    C : Type u_1
    D₁ : Type u_5
    D₂ : Type u_6
    D₃ : Type u_7
    inst✝⁶ : CategoryTheory.Category.{u_9, u_1} C
    inst✝⁵ : CategoryTheory.Category.{u_8, u_5} D₁
    inst✝⁴ : CategoryTheory.Category.{u_11, u_6} D₂
    inst✝³ : CategoryTheory.Category.{u_10, u_7} D₃
    W : CategoryTheory.MorphismProperty C
    L₁ : CategoryTheory.Functor C D₁
    inst✝² : L₁.IsLocalization W
    L₂ : CategoryTheory.Functor C D₂
    inst✝¹ : L₂.IsLocalization W
    L₃ : CategoryTheory.Functor C D₃
    inst✝ : L₃.IsLocalization W
    X Y : C
    f : Quiver.Hom (L₁.obj X) (L₁.obj Y)
    ⊢ Eq ((CategoryTheory.Localization.homEquiv W L₂ L₃) ((CategoryTheory.Localiza …
  -/
  dsimp only [homEquiv_apply]
  /-
    C : Type u_1
    D₁ : Type u_5
    D₂ : Type u_6
    D₃ : Type u_7
    inst✝⁶ : CategoryTheory.Category.{u_9, u_1} C
    inst✝⁵ : CategoryTheory.Category.{u_8, u_5} D₁
    inst✝⁴ : CategoryTheory.Category.{u_11, u_6} D₂
    inst✝³ : CategoryTheory.Category.{u_10, u_7} D₃
    W : CategoryTheory.MorphismProperty C
    L₁ : CategoryTheory.Functor C D₁
    inst✝² : L₁.IsLocalization W
    L₂ : CategoryTheory.Functor C D₂
    inst✝¹ : L₂.IsLocalization W
    L₃ : CategoryTheory.Functor C D₃
    inst✝ : L₃.IsLocalization W
    X Y : C
    f : Quiver.Hom (L₁.obj X) (L₁.obj Y)
    ⊢ Eq ((CategoryTheory.LocalizerMorphism.id W).homMap L₂ L₃ ((CategoryTheory.Lo …
  -/
  apply LocalizerMorphism.homMap_homMap
  /-
    🎉 no goals
  -/


lemma homEquiv_comp (f : L₁.obj X ⟶ L₁.obj Y) (g : L₁.obj Y ⟶ L₁.obj Z) :
    homEquiv W L₁ L₂ (f ≫ g) = homEquiv W L₁ L₂ f ≫ homEquiv W L₁ L₂ g := by
  /-
    C : Type u_1
    D₁ : Type u_5
    D₂ : Type u_6
    inst✝⁴ : CategoryTheory.Category.{u_9, u_1} C
    inst✝³ : CategoryTheory.Category.{u_8, u_5} D₁
    inst✝² : CategoryTheory.Category.{u_10, u_6} D₂
    W : CategoryTheory.MorphismProperty C
    L₁ : CategoryTheory.Functor C D₁
    inst✝¹ : L₁.IsLocalization W
    L₂ : CategoryTheory.Functor C D₂
    inst✝ : L₂.IsLocalization W
    X Y Z : C
    f : Quiver.Hom (L₁.obj X) (L₁.obj Y)
    g : Quiver.Hom (L₁.obj Y) (L₁.obj Z)
    ⊢ Eq ((CategoryTheory.Localization.homEquiv W L₁ L₂) (CategoryTheory.CategoryS …
  -/
  apply LocalizerMorphism.homMap_comp
  /-
    🎉 no goals
  -/


@[simp]
lemma homEquiv_map (f : X ⟶ Y) : homEquiv W L₁ L₂ (L₁.map f) = L₂.map f := by
  /-
    C : Type u_1
    D₁ : Type u_5
    D₂ : Type u_6
    inst✝⁴ : CategoryTheory.Category.{u_8, u_1} C
    inst✝³ : CategoryTheory.Category.{u_10, u_5} D₁
    inst✝² : CategoryTheory.Category.{u_9, u_6} D₂
    W : CategoryTheory.MorphismProperty C
    L₁ : CategoryTheory.Functor C D₁
    inst✝¹ : L₁.IsLocalization W
    L₂ : CategoryTheory.Functor C D₂
    inst✝ : L₂.IsLocalization W
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Eq ((CategoryTheory.Localization.homEquiv W L₁ L₂) (L₁.map f)) (L₂.map f)
  -/
  simp [homEquiv_apply]
  /-
    🎉 no goals
  -/


variable (X) in
@[simp]
lemma homEquiv_id : homEquiv W L₁ L₂ (𝟙 (L₁.obj X)) = 𝟙 (L₂.obj X) := by
  /-
    C : Type u_1
    D₁ : Type u_5
    D₂ : Type u_6
    inst✝⁴ : CategoryTheory.Category.{u_9, u_1} C
    inst✝³ : CategoryTheory.Category.{u_10, u_5} D₁
    inst✝² : CategoryTheory.Category.{u_8, u_6} D₂
    W : CategoryTheory.MorphismProperty C
    L₁ : CategoryTheory.Functor C D₁
    inst✝¹ : L₁.IsLocalization W
    L₂ : CategoryTheory.Functor C D₂
    inst✝ : L₂.IsLocalization W
    X : C
    ⊢ Eq ((CategoryTheory.Localization.homEquiv W L₁ L₂) (CategoryTheory.CategoryS …
  -/
  simp [homEquiv_apply]
  /-
    🎉 no goals
  -/


lemma homEquiv_isoOfHom_inv (f : Y ⟶ X) (hf : W f) :
    homEquiv W L₁ L₂ (isoOfHom L₁ W f hf).inv = (isoOfHom L₂ W f hf).inv := by
  rw [← cancel_mono (isoOfHom L₂ W f hf).hom, Iso.inv_hom_id, isoOfHom_hom,
    ← homEquiv_map W L₁ L₂ f, ← homEquiv_comp, isoOfHom_inv_hom_id, homEquiv_id]


