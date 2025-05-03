/-- Associator isomorphism for the action of bifunctors on graded objects. -/
noncomputable def mapBifunctorAssociator :
    mapBifunctorMapObj G ρ₁₂.q (mapBifunctorMapObj F₁₂ ρ₁₂.p X₁ X₂) X₃ ≅
      mapBifunctorMapObj F ρ₂₃.q X₁ (mapBifunctorMapObj G₂₃ ρ₂₃.p X₂ X₃) :=
  have := H₁₂.hasMap
  have := H₂₃.hasMap
  (mapBifunctorComp₁₂MapObjIso F₁₂ G ρ₁₂ X₁ X₂ X₃).symm ≪≫
    mapIso ((((mapTrifunctorMapIso associator I₁ I₂ I₃).app X₁).app X₂).app X₃) r ≪≫
    mapBifunctorComp₂₃MapObjIso F G₂₃ ρ₂₃ X₁ X₂ X₃


@[reassoc (attr := simp, nolint unusedHavesSuffices)]
lemma ι_mapBifunctorAssociator_hom (i₁ : I₁) (i₂ : I₂) (i₃ : I₃) (j : J) (h : r (i₁, i₂, i₃) = j) :
    ιMapBifunctor₁₂BifunctorMapObj F₁₂ G ρ₁₂ X₁ X₂ X₃ i₁ i₂ i₃ j h ≫
      (mapBifunctorAssociator associator ρ₁₂ ρ₂₃ X₁ X₂ X₃).hom j =
        ((associator.hom.app (X₁ i₁)).app (X₂ i₂)).app (X₃ i₃) ≫
          ιMapBifunctorBifunctor₂₃MapObj F G₂₃ ρ₂₃ X₁ X₂ X₃ i₁ i₂ i₃ j h := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₁₂ : Type u_3
    C₂₃ : Type u_4
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝⁹ : CategoryTheory.Category.{u_15, u_1} C₁
    inst✝⁸ : CategoryTheory.Category.{u_14, u_2} C₂
    inst✝⁷ : CategoryTheory.Category.{u_12, u_5} C₃
    inst✝⁶ : CategoryTheory.Category.{u_11, u_6} C₄
    inst✝⁵ : CategoryTheory.Category.{u_13, u_3} C₁₂
    inst✝⁴ : CategoryTheory.Category.{u_17, u_4} C₂₃
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    associator : CategoryTheory.Iso (CategoryTheory.bifunctorComp₁₂ F₁₂ G) (Catego …
    I₁ : Type u_7
    I₂ : Type u_8
    I₃ : Type u_9
    J : Type u_10
    r : Prod I₁ (Prod I₂ I₃) → J
    ρ₁₂ : CategoryTheory.GradedObject.BifunctorComp₁₂IndexData r
    ρ₂₃ : CategoryTheory.GradedObject.BifunctorComp₂₃IndexData r
    X₁ : CategoryTheory.GradedObject I₁ C₁
    X₂ : CategoryTheory.GradedObject I₂ C₂
    X₃ : CategoryTheory.GradedObject I₃ C₃
    inst✝³ : (((CategoryTheory.GradedObject.mapBifunctor F₁₂ I₁ I₂).obj X₁).obj X₂ …
    inst✝² : (((CategoryTheory.GradedObject.mapBifunctor G ρ₁₂.I₁₂ I₃).obj (Catego …
    inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor G₂₃ I₂ I₃).obj X₂).obj X₃ …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor F I₁ ρ₂₃.I₂₃).obj X₁).obj  …
    H₁₂ : CategoryTheory.GradedObject.HasGoodTrifunctor₁₂Obj F₁₂ G ρ₁₂ X₁ X₂ X₃
    H₂₃ : CategoryTheory.GradedObject.HasGoodTrifunctor₂₃Obj F G₂₃ ρ₂₃ X₁ X₂ X₃
    i₁ : I₁
    i₂ : I₂
    i₃ : I₃
    j : J
    h : Eq (r { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.ιMapBifu …
  -/
  have := H₁₂.hasMap
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₁₂ : Type u_3
    C₂₃ : Type u_4
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝⁹ : CategoryTheory.Category.{u_15, u_1} C₁
    inst✝⁸ : CategoryTheory.Category.{u_14, u_2} C₂
    inst✝⁷ : CategoryTheory.Category.{u_12, u_5} C₃
    inst✝⁶ : CategoryTheory.Category.{u_11, u_6} C₄
    inst✝⁵ : CategoryTheory.Category.{u_13, u_3} C₁₂
    inst✝⁴ : CategoryTheory.Category.{u_17, u_4} C₂₃
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    associator : CategoryTheory.Iso (CategoryTheory.bifunctorComp₁₂ F₁₂ G) (Catego …
    I₁ : Type u_7
    I₂ : Type u_8
    I₃ : Type u_9
    J : Type u_10
    r : Prod I₁ (Prod I₂ I₃) → J
    ρ₁₂ : CategoryTheory.GradedObject.BifunctorComp₁₂IndexData r
    ρ₂₃ : CategoryTheory.GradedObject.BifunctorComp₂₃IndexData r
    X₁ : CategoryTheory.GradedObject I₁ C₁
    X₂ : CategoryTheory.GradedObject I₂ C₂
    X₃ : CategoryTheory.GradedObject I₃ C₃
    inst✝³ : (((CategoryTheory.GradedObject.mapBifunctor F₁₂ I₁ I₂).obj X₁).obj X₂ …
    inst✝² : (((CategoryTheory.GradedObject.mapBifunctor G ρ₁₂.I₁₂ I₃).obj (Catego …
    inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor G₂₃ I₂ I₃).obj X₂).obj X₃ …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor F I₁ ρ₂₃.I₂₃).obj X₁).obj  …
    H₁₂ : CategoryTheory.GradedObject.HasGoodTrifunctor₁₂Obj F₁₂ G ρ₁₂ X₁ X₂ X₃
    H₂₃ : CategoryTheory.GradedObject.HasGoodTrifunctor₂₃Obj F G₂₃ ρ₂₃ X₁ X₂ X₃
    i₁ : I₁
    i₂ : I₂
    i₃ : I₃
    j : J
    h : Eq (r { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
    this : ((((CategoryTheory.GradedObject.mapTrifunctor (CategoryTheory.bifunctor …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.ιMapBifu …
  -/
  have := H₂₃.hasMap
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₁₂ : Type u_3
    C₂₃ : Type u_4
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝⁹ : CategoryTheory.Category.{u_15, u_1} C₁
    inst✝⁸ : CategoryTheory.Category.{u_14, u_2} C₂
    inst✝⁷ : CategoryTheory.Category.{u_12, u_5} C₃
    inst✝⁶ : CategoryTheory.Category.{u_11, u_6} C₄
    inst✝⁵ : CategoryTheory.Category.{u_13, u_3} C₁₂
    inst✝⁴ : CategoryTheory.Category.{u_17, u_4} C₂₃
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    associator : CategoryTheory.Iso (CategoryTheory.bifunctorComp₁₂ F₁₂ G) (Catego …
    I₁ : Type u_7
    I₂ : Type u_8
    I₃ : Type u_9
    J : Type u_10
    r : Prod I₁ (Prod I₂ I₃) → J
    ρ₁₂ : CategoryTheory.GradedObject.BifunctorComp₁₂IndexData r
    ρ₂₃ : CategoryTheory.GradedObject.BifunctorComp₂₃IndexData r
    X₁ : CategoryTheory.GradedObject I₁ C₁
    X₂ : CategoryTheory.GradedObject I₂ C₂
    X₃ : CategoryTheory.GradedObject I₃ C₃
    inst✝³ : (((CategoryTheory.GradedObject.mapBifunctor F₁₂ I₁ I₂).obj X₁).obj X₂ …
    inst✝² : (((CategoryTheory.GradedObject.mapBifunctor G ρ₁₂.I₁₂ I₃).obj (Catego …
    inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor G₂₃ I₂ I₃).obj X₂).obj X₃ …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor F I₁ ρ₂₃.I₂₃).obj X₁).obj  …
    H₁₂ : CategoryTheory.GradedObject.HasGoodTrifunctor₁₂Obj F₁₂ G ρ₁₂ X₁ X₂ X₃
    H₂₃ : CategoryTheory.GradedObject.HasGoodTrifunctor₂₃Obj F G₂₃ ρ₂₃ X₁ X₂ X₃
    i₁ : I₁
    i₂ : I₂
    i₃ : I₃
    j : J
    h : Eq (r { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
    this✝ : ((((CategoryTheory.GradedObject.mapTrifunctor (CategoryTheory.bifuncto …
    this : ((((CategoryTheory.GradedObject.mapTrifunctor (CategoryTheory.bifunctor …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.GradedObject.ιMapBifu …
  -/
  dsimp [mapBifunctorAssociator]
  rw [ι_mapBifunctorComp₁₂MapObjIso_inv_assoc, ιMapTrifunctorMapObj,
    ι_mapMap_assoc, mapTrifunctorMapNatTrans_app_app_app]
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₁₂ : Type u_3
    C₂₃ : Type u_4
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝⁹ : CategoryTheory.Category.{u_15, u_1} C₁
    inst✝⁸ : CategoryTheory.Category.{u_14, u_2} C₂
    inst✝⁷ : CategoryTheory.Category.{u_12, u_5} C₃
    inst✝⁶ : CategoryTheory.Category.{u_11, u_6} C₄
    inst✝⁵ : CategoryTheory.Category.{u_13, u_3} C₁₂
    inst✝⁴ : CategoryTheory.Category.{u_17, u_4} C₂₃
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    associator : CategoryTheory.Iso (CategoryTheory.bifunctorComp₁₂ F₁₂ G) (Catego …
    I₁ : Type u_7
    I₂ : Type u_8
    I₃ : Type u_9
    J : Type u_10
    r : Prod I₁ (Prod I₂ I₃) → J
    ρ₁₂ : CategoryTheory.GradedObject.BifunctorComp₁₂IndexData r
    ρ₂₃ : CategoryTheory.GradedObject.BifunctorComp₂₃IndexData r
    X₁ : CategoryTheory.GradedObject I₁ C₁
    X₂ : CategoryTheory.GradedObject I₂ C₂
    X₃ : CategoryTheory.GradedObject I₃ C₃
    inst✝³ : (((CategoryTheory.GradedObject.mapBifunctor F₁₂ I₁ I₂).obj X₁).obj X₂ …
    inst✝² : (((CategoryTheory.GradedObject.mapBifunctor G ρ₁₂.I₁₂ I₃).obj (Catego …
    inst✝¹ : (((CategoryTheory.GradedObject.mapBifunctor G₂₃ I₂ I₃).obj X₂).obj X₃ …
    inst✝ : (((CategoryTheory.GradedObject.mapBifunctor F I₁ ρ₂₃.I₂₃).obj X₁).obj  …
    H₁₂ : CategoryTheory.GradedObject.HasGoodTrifunctor₁₂Obj F₁₂ G ρ₁₂ X₁ X₂ X₃
    H₂₃ : CategoryTheory.GradedObject.HasGoodTrifunctor₂₃Obj F G₂₃ ρ₂₃ X₁ X₂ X₃
    i₁ : I₁
    i₂ : I₂
    i₃ : I₃
    j : J
    h : Eq (r { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
    this✝ : ((((CategoryTheory.GradedObject.mapTrifunctor (CategoryTheory.bifuncto …
    this : ((((CategoryTheory.GradedObject.mapTrifunctor (CategoryTheory.bifunctor …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((associator.hom.app (X₁ { fst := i₁ …
  -/
  erw [ι_mapBifunctorComp₂₃MapObjIso_hom]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma ι_mapBifunctorAssociator_inv (i₁ : I₁) (i₂ : I₂) (i₃ : I₃) (j : J) (h : r (i₁, i₂, i₃) = j) :
    ιMapBifunctorBifunctor₂₃MapObj F G₂₃ ρ₂₃ X₁ X₂ X₃ i₁ i₂ i₃ j h ≫
      (mapBifunctorAssociator associator ρ₁₂ ρ₂₃ X₁ X₂ X₃).inv j =
    ((associator.inv.app (X₁ i₁)).app (X₂ i₂)).app (X₃ i₃) ≫
      ιMapBifunctor₁₂BifunctorMapObj F₁₂ G ρ₁₂ X₁ X₂ X₃ i₁ i₂ i₃ j h := by
  rw [← cancel_mono ((mapBifunctorAssociator associator ρ₁₂ ρ₂₃ X₁ X₂ X₃).hom j),
    assoc, assoc, Iso.inv_hom_id_eval, comp_id, ι_mapBifunctorAssociator_hom,
    ← NatTrans.comp_app_assoc, ← NatTrans.comp_app, Iso.inv_hom_id_app,
    NatTrans.id_app, NatTrans.id_app, id_comp]


