variable (F₁₂ G) in
/-- Given bifunctors `F₁₂ : C₁ ⥤ C₂ ⥤ C₁₂`, `G : C₁₂ ⥤ C₃ ⥤ C₄`, homological complexes
`K₁ : HomologicalComplex C₁ c₁`, `K₂ : HomologicalComplex C₂ c₂` and
`K₃ : HomologicalComplex C₃ c₃`, and complexes shapes `c₁₂`, `c₄`, this asserts
that for all `i₁₂ : ι₁₂` and `i₃ : ι₃`, the functor `G(-, K₃.X i₃)` commutes with
the coproducts of the `F₁₂(X₁ i₁, X₂ i₂)` such that `π c₁ c₂ c₁₂ ⟨i₁, i₂⟩ = i₁₂`. -/
abbrev HasGoodTrifunctor₁₂Obj :=
  GradedObject.HasGoodTrifunctor₁₂Obj F₁₂ G
    (ComplexShape.ρ₁₂ c₁ c₂ c₃ c₁₂ c₄) K₁.X K₂.X K₃.X


variable (F G₂₃) in
/-- Given bifunctors `F : C₁ ⥤ C₂₃ ⥤ C₄`, `G₂₃ : C₂ ⥤ C₃ ⥤ C₂₃`, homological complexes
`K₁ : HomologicalComplex C₁ c₁`, `K₂ : HomologicalComplex C₂ c₂` and
`K₃ : HomologicalComplex C₃ c₃`, and complexes shapes `c₁₂`, `c₂₃`, `c₄`
with `ComplexShape.Associative c₁ c₂ c₃ c₁₂ c₂₃ c₄`, this asserts that for
all `i₁ : ι₁` and `i₂₃ : ι₂₃`, the functor `F(K₁.X i₁, _)` commutes with
the coproducts of the `G₂₃(K₂.X i₂, K₃.X i₃)`
such that `π c₂ c₃ c₂₃ ⟨i₂, i₃⟩ = i₂₃`. -/
abbrev HasGoodTrifunctor₂₃Obj :=
  GradedObject.HasGoodTrifunctor₂₃Obj F G₂₃
    (ComplexShape.ρ₂₃ c₁ c₂ c₃ c₁₂ c₂₃ c₄) K₁.X K₂.X K₃.X


instance :
    (((GradedObject.mapBifunctor F₁₂ ι₁ ι₂).obj K₁.X).obj K₂.X).HasMap
      (ComplexShape.π c₁ c₂ c₁₂) :=
  inferInstanceAs (HasMapBifunctor K₁ K₂ F₁₂ c₁₂)


instance :
    (((GradedObject.mapBifunctor G ι₁₂ ι₃).obj (GradedObject.mapBifunctorMapObj F₁₂
        (ComplexShape.π c₁ c₂ c₁₂) K₁.X K₂.X)).obj K₃.X).HasMap
          (ComplexShape.π c₁₂ c₃ c₄) :=
  inferInstanceAs (HasMapBifunctor (mapBifunctor K₁ K₂ F₁₂ c₁₂) K₃ G c₄)


instance :
    (((GradedObject.mapBifunctor F ι₁ ι₂₃).obj K₁.X).obj
      (GradedObject.mapBifunctorMapObj G₂₃
        (ComplexShape.π c₂ c₃ c₂₃) K₂.X K₃.X)).HasMap (ComplexShape.π c₁ c₂₃ c₄) :=
  inferInstanceAs (HasMapBifunctor K₁ (mapBifunctor K₂ K₃ G₂₃ c₂₃) F c₄)


/-- The associator isomorphism for the action of bifunctors
on homological complexes, in each degree. -/
noncomputable def mapBifunctorAssociatorX
    [H₁₂ : HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄]
    [H₂₃ : HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄](j : ι₄) :
    (mapBifunctor (mapBifunctor K₁ K₂ F₁₂ c₁₂) K₃ G c₄).X j ≅
      (mapBifunctor K₁ (mapBifunctor K₂ K₃ G₂₃ c₂₃) F c₄).X j :=
  (GradedObject.eval j).mapIso
    (GradedObject.mapBifunctorAssociator (associator := associator)
      (H₁₂ := H₁₂) (H₂₃ := H₂₃))


/-- The inclusion of a summand in `mapBifunctor (mapBifunctor K₁ K₂ F₁₂ c₁₂) K₃ G c₄`. -/
noncomputable def ι (i₁ : ι₁) (i₂ : ι₂) (i₃ : ι₃) (j : ι₄)
    (h : ComplexShape.r c₁ c₂ c₃ c₁₂ c₄ (i₁, i₂, i₃) = j) :
    (G.obj ((F₁₂.obj (K₁.X i₁)).obj (K₂.X i₂))).obj (K₃.X i₃) ⟶
      (mapBifunctor (mapBifunctor K₁ K₂ F₁₂ c₁₂) K₃ G c₄).X j :=
  GradedObject.ιMapBifunctor₁₂BifunctorMapObj _ _ (ComplexShape.ρ₁₂ c₁ c₂ c₃ c₁₂ c₄) _ _ _ _ _ _ _ h


lemma ι_eq (i₁ : ι₁) (i₂ : ι₂) (i₃ : ι₃) (i₁₂ : ι₁₂) (j : ι₄)
    (h₁₂ : ComplexShape.π c₁ c₂ c₁₂ ⟨i₁, i₂⟩ = i₁₂)
    (h : ComplexShape.π c₁₂ c₃ c₄ (i₁₂, i₃) = j) :
                                           /-
                                             C₁ : Type u_1
                                             C₂ : Type u_2
                                             C₁₂ : Type u_3
                                             C₂₃ : Type u_4
                                             C₃ : Type u_5
                                             C₄ : Type u_6
                                             inst✝²⁹ : CategoryTheory.Category.{?u.137589, u_1} C₁
                                             inst✝²⁸ : CategoryTheory.Category.{?u.137593, u_2} C₂
                                             inst✝²⁷ : CategoryTheory.Category.{?u.137597, u_5} C₃
                                             inst✝²⁶ : CategoryTheory.Category.{?u.137601, u_6} C₄
                                             inst✝²⁵ : CategoryTheory.Category.{?u.137605, u_3} C₁₂
                                             inst✝²⁴ : CategoryTheory.Category.{?u.137609, u_4} C₂₃
                                             inst✝²³ : CategoryTheory.Limits.HasZeroMorphisms C₁
                                             inst✝²² : CategoryTheory.Limits.HasZeroMorphisms C₂
                                             inst✝²¹ : CategoryTheory.Limits.HasZeroMorphisms C₃
                                             inst✝²⁰ : CategoryTheory.Preadditive C₁₂
                                             inst✝¹⁹ : CategoryTheory.Preadditive C₂₃
                                             inst✝¹⁸ : CategoryTheory.Preadditive C₄
                                             F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
                                             G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
                                             F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
                                             G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
                                             inst✝¹⁷ : F₁₂.PreservesZeroMorphisms
                                             inst✝¹⁶ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
                                             inst✝¹⁵ : G.Additive
                                             inst✝¹⁴ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
                                             inst✝¹³ : G₂₃.PreservesZeroMorphisms
                                             inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
                                             inst✝¹¹ : F.PreservesZeroMorphisms
                                             inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
                                             associator : CategoryTheory.Iso (CategoryTheory.bifunctorComp₁₂ F₁₂ G) (Catego …
                                             ι₁ : Type u_7
                                             ι₂ : Type u_8
                                             ι₃ : Type u_9
                                             ι₁₂ : Type u_10
                                             ι₂₃ : Type u_11
                                             ι₄ : Type u_12
                                             inst✝⁹ : DecidableEq ι₄
                                             c₁ : ComplexShape ι₁
                                             c₂ : ComplexShape ι₂
                                             c₃ : ComplexShape ι₃
                                             K₁ : HomologicalComplex C₁ c₁
                                             K₂ : HomologicalComplex C₂ c₂
                                             K₃ : HomologicalComplex C₃ c₃
                                             c₁₂ : ComplexShape ι₁₂
                                             c₂₃ : ComplexShape ι₂₃
                                             c₄ : ComplexShape ι₄
                                             inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
                                             inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
                                             inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
                                             inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
                                             inst✝⁴ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
                                             inst✝³ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
                                             inst✝² : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
                                             inst✝¹ : DecidableEq ι₁₂
                                             inst✝ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
                                             i₁ : ι₁
                                             i₂ : ι₂
                                             i₃ : ι₃
                                             i₁₂ : ι₁₂
                                             j : ι₄
                                             h₁₂ : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
                                             h : Eq (c₁₂.π c₃ c₄ { fst := i₁₂, snd := i₃ }) j
                                             ⊢ Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
                                           -/
    ι F₁₂ G K₁ K₂ K₃ c₁₂ c₄ i₁ i₂ i₃ j (by rw [← h, ← h₁₂]; rfl) =
                                                            /-
                                                              🎉 no goals
                                                            -/
      (G.map (ιMapBifunctor K₁ K₂ F₁₂ c₁₂ i₁ i₂ i₁₂ h₁₂)).app (K₃.X i₃) ≫
        ιMapBifunctor (mapBifunctor K₁ K₂ F₁₂ c₁₂) K₃ G c₄ i₁₂ i₃ j h := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₁₂ : Type u_3
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝¹⁹ : CategoryTheory.Category.{u_17, u_1} C₁
    inst✝¹⁸ : CategoryTheory.Category.{u_16, u_2} C₂
    inst✝¹⁷ : CategoryTheory.Category.{u_14, u_5} C₃
    inst✝¹⁶ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝¹⁵ : CategoryTheory.Category.{u_15, u_3} C₁₂
    inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹² : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹¹ : CategoryTheory.Preadditive C₁₂
    inst✝¹⁰ : CategoryTheory.Preadditive C₄
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    inst✝⁹ : F₁₂.PreservesZeroMorphisms
    inst✝⁸ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
    inst✝⁷ : G.Additive
    inst✝⁶ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₄ : Type u_12
    inst✝⁵ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₄ : ComplexShape ι₄
    inst✝⁴ : TotalComplexShape c₁ c₂ c₁₂
    inst✝³ : TotalComplexShape c₁₂ c₃ c₄
    inst✝² : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
    inst✝¹ : DecidableEq ι₁₂
    inst✝ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
    i₁ : ι₁
    i₂ : ι₂
    i₃ : ι₃
    i₁₂ : ι₁₂
    j : ι₄
    h₁₂ : Eq (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) i₁₂
    h : Eq (c₁₂.π c₃ c₄ { fst := i₁₂, snd := i₃ }) j
    ⊢ Eq (HomologicalComplex.mapBifunctor₁₂.ι F₁₂ G K₁ K₂ K₃ c₁₂ c₄ i₁ i₂ i₃ j ⋯)  …
  -/
  subst h₁₂
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₁₂ : Type u_3
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝¹⁹ : CategoryTheory.Category.{u_17, u_1} C₁
    inst✝¹⁸ : CategoryTheory.Category.{u_16, u_2} C₂
    inst✝¹⁷ : CategoryTheory.Category.{u_14, u_5} C₃
    inst✝¹⁶ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝¹⁵ : CategoryTheory.Category.{u_15, u_3} C₁₂
    inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹² : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹¹ : CategoryTheory.Preadditive C₁₂
    inst✝¹⁰ : CategoryTheory.Preadditive C₄
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    inst✝⁹ : F₁₂.PreservesZeroMorphisms
    inst✝⁸ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
    inst✝⁷ : G.Additive
    inst✝⁶ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₄ : Type u_12
    inst✝⁵ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₄ : ComplexShape ι₄
    inst✝⁴ : TotalComplexShape c₁ c₂ c₁₂
    inst✝³ : TotalComplexShape c₁₂ c₃ c₄
    inst✝² : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
    inst✝¹ : DecidableEq ι₁₂
    inst✝ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
    i₁ : ι₁
    i₂ : ι₂
    i₃ : ι₃
    j : ι₄
    h : Eq (c₁₂.π c₃ c₄ { fst := c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }, snd := i₃ } …
    ⊢ Eq (HomologicalComplex.mapBifunctor₁₂.ι F₁₂ G K₁ K₂ K₃ c₁₂ c₄ i₁ i₂ i₃ j ⋯)  …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The inclusion of a summand in `mapBifunctor (mapBifunctor K₁ K₂ F₁₂ c₁₂) K₃ G c₄`,
or zero. -/
noncomputable def ιOrZero (i₁ : ι₁) (i₂ : ι₂) (i₃ : ι₃) (j : ι₄) :
    (G.obj ((F₁₂.obj (K₁.X i₁)).obj (K₂.X i₂))).obj (K₃.X i₃) ⟶
      (mapBifunctor (mapBifunctor K₁ K₂ F₁₂ c₁₂) K₃ G c₄).X j :=
  if h : ComplexShape.r c₁ c₂ c₃ c₁₂ c₄ (i₁, i₂, i₃) = j then
    ι F₁₂ G K₁ K₂ K₃ c₁₂ c₄ i₁ i₂ i₃ j h
  else 0


lemma ιOrZero_eq (i₁ : ι₁) (i₂ : ι₂) (i₃ : ι₃) (j : ι₄)
    (h : ComplexShape.r c₁ c₂ c₃ c₁₂ c₄ (i₁, i₂, i₃) = j) :
    ιOrZero F₁₂ G K₁ K₂ K₃ c₁₂ c₄ i₁ i₂ i₃ j =
      ι F₁₂ G K₁ K₂ K₃ c₁₂ c₄ i₁ i₂ i₃ j h := dif_pos h


lemma ιOrZero_eq_zero (i₁ : ι₁) (i₂ : ι₂) (i₃ : ι₃) (j : ι₄)
    (h : ComplexShape.r c₁ c₂ c₃ c₁₂ c₄ (i₁, i₂, i₃) ≠ j) :
    ιOrZero F₁₂ G K₁ K₂ K₃ c₁₂ c₄ i₁ i₂ i₃ j = 0 := dif_neg h


variable {F₁₂ G K₁ K₂ K₃ c₁₂ c₄} in
@[ext]
lemma hom_ext
    [HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄] {j : ι₄} {A : C₄}
    {f g : (mapBifunctor (mapBifunctor K₁ K₂ F₁₂ c₁₂) K₃ G c₄).X j ⟶ A}
    (hfg : ∀ (i₁ : ι₁) (i₂ : ι₂) (i₃ : ι₃)
      (h : ComplexShape.r c₁ c₂ c₃ c₁₂ c₄ (i₁, i₂, i₃) = j),
      ι F₁₂ G K₁ K₂ K₃ c₁₂ c₄ i₁ i₂ i₃ j h ≫ f =
        ι F₁₂ G K₁ K₂ K₃ c₁₂ c₄ i₁ i₂ i₃ j h ≫ g) :
    f = g :=
  GradedObject.mapBifunctor₁₂BifunctorMapObj_ext hfg


/-- Constructor for morphisms from
`(mapBifunctor (mapBifunctor K₁ K₂ F₁₂ c₁₂) K₃ G c₄).X j`. -/
noncomputable def mapBifunctor₁₂Desc :
    (mapBifunctor (mapBifunctor K₁ K₂ F₁₂ c₁₂) K₃ G c₄).X j ⟶ A :=
  GradedObject.mapBifunctor₁₂BifunctorDesc (ρ₁₂ := ComplexShape.ρ₁₂ c₁ c₂ c₃ c₁₂ c₄) f


@[reassoc (attr := simp)]
lemma ι_mapBifunctor₁₂Desc (i₁ : ι₁) (i₂ : ι₂) (i₃ : ι₃)
    (h : ComplexShape.r c₁ c₂ c₃ c₁₂ c₄ (i₁, i₂, i₃) = j) :
    ι F₁₂ G K₁ K₂ K₃ c₁₂ c₄ i₁ i₂ i₃ j h ≫ mapBifunctor₁₂Desc f =
      f i₁ i₂ i₃ h := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₁₂ : Type u_3
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝²⁰ : CategoryTheory.Category.{u_17, u_1} C₁
    inst✝¹⁹ : CategoryTheory.Category.{u_16, u_2} C₂
    inst✝¹⁸ : CategoryTheory.Category.{u_14, u_5} C₃
    inst✝¹⁷ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝¹⁶ : CategoryTheory.Category.{u_15, u_3} C₁₂
    inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹² : CategoryTheory.Preadditive C₁₂
    inst✝¹¹ : CategoryTheory.Preadditive C₄
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    inst✝¹⁰ : F₁₂.PreservesZeroMorphisms
    inst✝⁹ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
    inst✝⁸ : G.Additive
    inst✝⁷ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₄ : Type u_12
    inst✝⁶ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₄ : ComplexShape ι₄
    inst✝⁵ : TotalComplexShape c₁ c₂ c₁₂
    inst✝⁴ : TotalComplexShape c₁₂ c₃ c₄
    inst✝³ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
    inst✝² : DecidableEq ι₁₂
    inst✝¹ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
    inst✝ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
    j : ι₄
    A : C₄
    f : (i₁ : ι₁) → (i₂ : ι₂) → (i₃ : ι₃) → Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd …
    i₁ : ι₁
    i₂ : ι₂
    i₃ : ι₃
    h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.mapBifunctor₁₂.ι  …
  -/
  apply GradedObject.ι_mapBifunctor₁₂BifunctorDesc
  /-
    🎉 no goals
  -/


/-- The first differential on a summand
of `mapBifunctor (mapBifunctor K₁ K₂ F₁₂ c₁₂) K₃ G c₄`. -/
noncomputable def d₁ (i₁ : ι₁) (i₂ : ι₂) (i₃ : ι₃) (j : ι₄) :
    (G.obj ((F₁₂.obj (K₁.X i₁)).obj (K₂.X i₂))).obj (K₃.X i₃) ⟶
      (mapBifunctor (mapBifunctor K₁ K₂ F₁₂ c₁₂) K₃ G c₄).X j :=
  (ComplexShape.ε₁ c₁₂ c₃ c₄ (ComplexShape.π c₁ c₂ c₁₂ ⟨i₁, i₂⟩, i₃) *
    ComplexShape.ε₁ c₁ c₂ c₁₂ (i₁, i₂)) •
  (G.map ((F₁₂.map (K₁.d i₁ (c₁.next i₁))).app (K₂.X i₂))).app (K₃.X i₃) ≫
    ιOrZero F₁₂ G K₁ K₂ K₃ c₁₂ c₄ _ i₂ i₃ j


lemma d₁_eq_zero (i₁ : ι₁) (i₂ : ι₂) (i₃ : ι₃) (j : ι₄) (h : ¬ c₁.Rel i₁ (c₁.next i₁)) :
    d₁ F₁₂ G K₁ K₂ K₃ c₁₂ c₄ i₁ i₂ i₃ j = 0 := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₁₂ : Type u_3
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝¹⁹ : CategoryTheory.Category.{u_17, u_1} C₁
    inst✝¹⁸ : CategoryTheory.Category.{u_16, u_2} C₂
    inst✝¹⁷ : CategoryTheory.Category.{u_14, u_5} C₃
    inst✝¹⁶ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝¹⁵ : CategoryTheory.Category.{u_15, u_3} C₁₂
    inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹² : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹¹ : CategoryTheory.Preadditive C₁₂
    inst✝¹⁰ : CategoryTheory.Preadditive C₄
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    inst✝⁹ : F₁₂.PreservesZeroMorphisms
    inst✝⁸ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
    inst✝⁷ : G.Additive
    inst✝⁶ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₄ : Type u_12
    inst✝⁵ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₄ : ComplexShape ι₄
    inst✝⁴ : TotalComplexShape c₁ c₂ c₁₂
    inst✝³ : TotalComplexShape c₁₂ c₃ c₄
    inst✝² : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
    inst✝¹ : DecidableEq ι₁₂
    inst✝ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
    i₁ : ι₁
    i₂ : ι₂
    i₃ : ι₃
    j : ι₄
    h : Not (c₁.Rel i₁ (c₁.next i₁))
    ⊢ Eq (HomologicalComplex.mapBifunctor₁₂.d₁ F₁₂ G K₁ K₂ K₃ c₁₂ c₄ i₁ i₂ i₃ j) 0
  -/
  dsimp [d₁]
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₁₂ : Type u_3
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝¹⁹ : CategoryTheory.Category.{u_17, u_1} C₁
    inst✝¹⁸ : CategoryTheory.Category.{u_16, u_2} C₂
    inst✝¹⁷ : CategoryTheory.Category.{u_14, u_5} C₃
    inst✝¹⁶ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝¹⁵ : CategoryTheory.Category.{u_15, u_3} C₁₂
    inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹² : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹¹ : CategoryTheory.Preadditive C₁₂
    inst✝¹⁰ : CategoryTheory.Preadditive C₄
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    inst✝⁹ : F₁₂.PreservesZeroMorphisms
    inst✝⁸ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
    inst✝⁷ : G.Additive
    inst✝⁶ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₄ : Type u_12
    inst✝⁵ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₄ : ComplexShape ι₄
    inst✝⁴ : TotalComplexShape c₁ c₂ c₁₂
    inst✝³ : TotalComplexShape c₁₂ c₃ c₄
    inst✝² : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
    inst✝¹ : DecidableEq ι₁₂
    inst✝ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
    i₁ : ι₁
    i₂ : ι₂
    i₃ : ι₃
    j : ι₄
    h : Not (c₁.Rel i₁ (c₁.next i₁))
    ⊢ Eq (HSMul.hSMul (HMul.hMul (c₁₂.ε₁ c₃ c₄ { fst := c₁.π c₂ c₁₂ { fst := i₁, s …
  -/
  rw [shape _ _ _ h, Functor.map_zero, zero_app, Functor.map_zero, zero_app, zero_comp, smul_zero]
  /-
    🎉 no goals
  -/


lemma d₁_eq {i₁ i₁' : ι₁} (h₁ : c₁.Rel i₁ i₁') (i₂ : ι₂) (i₃ : ι₃) (j : ι₄) :
    d₁ F₁₂ G K₁ K₂ K₃ c₁₂ c₄ i₁ i₂ i₃ j =
    (ComplexShape.ε₁ c₁₂ c₃ c₄ (ComplexShape.π c₁ c₂ c₁₂ ⟨i₁, i₂⟩, i₃) *
      ComplexShape.ε₁ c₁ c₂ c₁₂ (i₁, i₂) ) •
    (G.map ((F₁₂.map (K₁.d i₁ i₁')).app (K₂.X i₂))).app (K₃.X i₃) ≫
      ιOrZero F₁₂ G K₁ K₂ K₃ c₁₂ c₄ i₁' i₂ i₃ j := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₁₂ : Type u_3
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝¹⁹ : CategoryTheory.Category.{u_17, u_1} C₁
    inst✝¹⁸ : CategoryTheory.Category.{u_16, u_2} C₂
    inst✝¹⁷ : CategoryTheory.Category.{u_14, u_5} C₃
    inst✝¹⁶ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝¹⁵ : CategoryTheory.Category.{u_15, u_3} C₁₂
    inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹² : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹¹ : CategoryTheory.Preadditive C₁₂
    inst✝¹⁰ : CategoryTheory.Preadditive C₄
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    inst✝⁹ : F₁₂.PreservesZeroMorphisms
    inst✝⁸ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
    inst✝⁷ : G.Additive
    inst✝⁶ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₄ : Type u_12
    inst✝⁵ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₄ : ComplexShape ι₄
    inst✝⁴ : TotalComplexShape c₁ c₂ c₁₂
    inst✝³ : TotalComplexShape c₁₂ c₃ c₄
    inst✝² : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
    inst✝¹ : DecidableEq ι₁₂
    inst✝ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
    i₁ i₁' : ι₁
    h₁ : c₁.Rel i₁ i₁'
    i₂ : ι₂
    i₃ : ι₃
    j : ι₄
    ⊢ Eq (HomologicalComplex.mapBifunctor₁₂.d₁ F₁₂ G K₁ K₂ K₃ c₁₂ c₄ i₁ i₂ i₃ j) ( …
  -/
  obtain rfl := c₁.next_eq' h₁
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₁₂ : Type u_3
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝¹⁹ : CategoryTheory.Category.{u_17, u_1} C₁
    inst✝¹⁸ : CategoryTheory.Category.{u_16, u_2} C₂
    inst✝¹⁷ : CategoryTheory.Category.{u_14, u_5} C₃
    inst✝¹⁶ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝¹⁵ : CategoryTheory.Category.{u_15, u_3} C₁₂
    inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹² : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹¹ : CategoryTheory.Preadditive C₁₂
    inst✝¹⁰ : CategoryTheory.Preadditive C₄
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    inst✝⁹ : F₁₂.PreservesZeroMorphisms
    inst✝⁸ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
    inst✝⁷ : G.Additive
    inst✝⁶ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₄ : Type u_12
    inst✝⁵ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₄ : ComplexShape ι₄
    inst✝⁴ : TotalComplexShape c₁ c₂ c₁₂
    inst✝³ : TotalComplexShape c₁₂ c₃ c₄
    inst✝² : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
    inst✝¹ : DecidableEq ι₁₂
    inst✝ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
    i₁ : ι₁
    i₂ : ι₂
    i₃ : ι₃
    j : ι₄
    h₁ : c₁.Rel i₁ (c₁.next i₁)
    ⊢ Eq (HomologicalComplex.mapBifunctor₁₂.d₁ F₁₂ G K₁ K₂ K₃ c₁₂ c₄ i₁ i₂ i₃ j) ( …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The second differential on a summand
of `mapBifunctor (mapBifunctor K₁ K₂ F₁₂ c₁₂) K₃ G c₄`. -/
noncomputable def d₂ (i₁ : ι₁) (i₂ : ι₂) (i₃ : ι₃) (j : ι₄) :
    (G.obj ((F₁₂.obj (K₁.X i₁)).obj (K₂.X i₂))).obj (K₃.X i₃) ⟶
      (mapBifunctor (mapBifunctor K₁ K₂ F₁₂ c₁₂) K₃ G c₄).X j :=
  (c₁₂.ε₁ c₃ c₄ (ComplexShape.π c₁ c₂ c₁₂ ⟨i₁, i₂⟩, i₃) * c₁.ε₂ c₂ c₁₂ (i₁, i₂)) •
  (G.map ((F₁₂.obj (K₁.X i₁)).map (K₂.d i₂ (c₂.next i₂)))).app (K₃.X i₃) ≫
    ιOrZero F₁₂ G K₁ K₂ K₃ c₁₂ c₄ i₁ _ i₃ j


lemma d₂_eq_zero (i₁ : ι₁) (i₂ : ι₂) (i₃ : ι₃) (j : ι₄) (h : ¬ c₂.Rel i₂ (c₂.next i₂)) :
    d₂ F₁₂ G K₁ K₂ K₃ c₁₂ c₄ i₁ i₂ i₃ j = 0 := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₁₂ : Type u_3
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝¹⁹ : CategoryTheory.Category.{u_17, u_1} C₁
    inst✝¹⁸ : CategoryTheory.Category.{u_16, u_2} C₂
    inst✝¹⁷ : CategoryTheory.Category.{u_14, u_5} C₃
    inst✝¹⁶ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝¹⁵ : CategoryTheory.Category.{u_15, u_3} C₁₂
    inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹² : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹¹ : CategoryTheory.Preadditive C₁₂
    inst✝¹⁰ : CategoryTheory.Preadditive C₄
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    inst✝⁹ : F₁₂.PreservesZeroMorphisms
    inst✝⁸ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
    inst✝⁷ : G.Additive
    inst✝⁶ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₄ : Type u_12
    inst✝⁵ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₄ : ComplexShape ι₄
    inst✝⁴ : TotalComplexShape c₁ c₂ c₁₂
    inst✝³ : TotalComplexShape c₁₂ c₃ c₄
    inst✝² : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
    inst✝¹ : DecidableEq ι₁₂
    inst✝ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
    i₁ : ι₁
    i₂ : ι₂
    i₃ : ι₃
    j : ι₄
    h : Not (c₂.Rel i₂ (c₂.next i₂))
    ⊢ Eq (HomologicalComplex.mapBifunctor₁₂.d₂ F₁₂ G K₁ K₂ K₃ c₁₂ c₄ i₁ i₂ i₃ j) 0
  -/
  dsimp [d₂]
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₁₂ : Type u_3
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝¹⁹ : CategoryTheory.Category.{u_17, u_1} C₁
    inst✝¹⁸ : CategoryTheory.Category.{u_16, u_2} C₂
    inst✝¹⁷ : CategoryTheory.Category.{u_14, u_5} C₃
    inst✝¹⁶ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝¹⁵ : CategoryTheory.Category.{u_15, u_3} C₁₂
    inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹² : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹¹ : CategoryTheory.Preadditive C₁₂
    inst✝¹⁰ : CategoryTheory.Preadditive C₄
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    inst✝⁹ : F₁₂.PreservesZeroMorphisms
    inst✝⁸ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
    inst✝⁷ : G.Additive
    inst✝⁶ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₄ : Type u_12
    inst✝⁵ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₄ : ComplexShape ι₄
    inst✝⁴ : TotalComplexShape c₁ c₂ c₁₂
    inst✝³ : TotalComplexShape c₁₂ c₃ c₄
    inst✝² : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
    inst✝¹ : DecidableEq ι₁₂
    inst✝ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
    i₁ : ι₁
    i₂ : ι₂
    i₃ : ι₃
    j : ι₄
    h : Not (c₂.Rel i₂ (c₂.next i₂))
    ⊢ Eq (HSMul.hSMul (HMul.hMul (c₁₂.ε₁ c₃ c₄ { fst := c₁.π c₂ c₁₂ { fst := i₁, s …
  -/
  rw [shape _ _ _ h, Functor.map_zero, Functor.map_zero, zero_app, zero_comp, smul_zero]
  /-
    🎉 no goals
  -/


lemma d₂_eq (i₁ : ι₁) {i₂ i₂' : ι₂} (h₂ : c₂.Rel i₂ i₂') (i₃ : ι₃) (j : ι₄) :
    d₂ F₁₂ G K₁ K₂ K₃ c₁₂ c₄ i₁ i₂ i₃ j =
  (c₁₂.ε₁ c₃ c₄ (ComplexShape.π c₁ c₂ c₁₂ ⟨i₁, i₂⟩, i₃) * c₁.ε₂ c₂ c₁₂ (i₁, i₂)) •
    (G.map ((F₁₂.obj (K₁.X i₁)).map (K₂.d i₂ i₂'))).app (K₃.X i₃) ≫
      ιOrZero F₁₂ G K₁ K₂ K₃ c₁₂ c₄ i₁ _ i₃ j := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₁₂ : Type u_3
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝¹⁹ : CategoryTheory.Category.{u_17, u_1} C₁
    inst✝¹⁸ : CategoryTheory.Category.{u_16, u_2} C₂
    inst✝¹⁷ : CategoryTheory.Category.{u_14, u_5} C₃
    inst✝¹⁶ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝¹⁵ : CategoryTheory.Category.{u_15, u_3} C₁₂
    inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹² : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹¹ : CategoryTheory.Preadditive C₁₂
    inst✝¹⁰ : CategoryTheory.Preadditive C₄
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    inst✝⁹ : F₁₂.PreservesZeroMorphisms
    inst✝⁸ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
    inst✝⁷ : G.Additive
    inst✝⁶ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₄ : Type u_12
    inst✝⁵ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₄ : ComplexShape ι₄
    inst✝⁴ : TotalComplexShape c₁ c₂ c₁₂
    inst✝³ : TotalComplexShape c₁₂ c₃ c₄
    inst✝² : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
    inst✝¹ : DecidableEq ι₁₂
    inst✝ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
    i₁ : ι₁
    i₂ i₂' : ι₂
    h₂ : c₂.Rel i₂ i₂'
    i₃ : ι₃
    j : ι₄
    ⊢ Eq (HomologicalComplex.mapBifunctor₁₂.d₂ F₁₂ G K₁ K₂ K₃ c₁₂ c₄ i₁ i₂ i₃ j) ( …
  -/
  obtain rfl := c₂.next_eq' h₂
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₁₂ : Type u_3
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝¹⁹ : CategoryTheory.Category.{u_17, u_1} C₁
    inst✝¹⁸ : CategoryTheory.Category.{u_16, u_2} C₂
    inst✝¹⁷ : CategoryTheory.Category.{u_14, u_5} C₃
    inst✝¹⁶ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝¹⁵ : CategoryTheory.Category.{u_15, u_3} C₁₂
    inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹² : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹¹ : CategoryTheory.Preadditive C₁₂
    inst✝¹⁰ : CategoryTheory.Preadditive C₄
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    inst✝⁹ : F₁₂.PreservesZeroMorphisms
    inst✝⁸ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
    inst✝⁷ : G.Additive
    inst✝⁶ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₄ : Type u_12
    inst✝⁵ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₄ : ComplexShape ι₄
    inst✝⁴ : TotalComplexShape c₁ c₂ c₁₂
    inst✝³ : TotalComplexShape c₁₂ c₃ c₄
    inst✝² : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
    inst✝¹ : DecidableEq ι₁₂
    inst✝ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
    i₁ : ι₁
    i₂ : ι₂
    i₃ : ι₃
    j : ι₄
    h₂ : c₂.Rel i₂ (c₂.next i₂)
    ⊢ Eq (HomologicalComplex.mapBifunctor₁₂.d₂ F₁₂ G K₁ K₂ K₃ c₁₂ c₄ i₁ i₂ i₃ j) ( …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The third differential on a summand
of `mapBifunctor (mapBifunctor K₁ K₂ F₁₂ c₁₂) K₃ G c₄`. -/
noncomputable def d₃ (i₁ : ι₁) (i₂ : ι₂) (i₃ : ι₃) (j : ι₄) :
    (G.obj ((F₁₂.obj (K₁.X i₁)).obj (K₂.X i₂))).obj (K₃.X i₃) ⟶
      (mapBifunctor (mapBifunctor K₁ K₂ F₁₂ c₁₂) K₃ G c₄).X j :=
  (ComplexShape.ε₂ c₁₂ c₃ c₄ (c₁.π c₂ c₁₂ (i₁, i₂), i₃)) •
    (G.obj ((F₁₂.obj (K₁.X i₁)).obj (K₂.X i₂))).map (K₃.d i₃ (c₃.next i₃)) ≫
      ιOrZero F₁₂ G K₁ K₂ K₃ c₁₂ c₄ i₁ i₂ _ j


lemma d₃_eq_zero (i₁ : ι₁) (i₂ : ι₂) (i₃ : ι₃) (j : ι₄) (h : ¬ c₃.Rel i₃ (c₃.next i₃)) :
    d₃ F₁₂ G K₁ K₂ K₃ c₁₂ c₄ i₁ i₂ i₃ j = 0 := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₁₂ : Type u_3
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝¹⁹ : CategoryTheory.Category.{u_17, u_1} C₁
    inst✝¹⁸ : CategoryTheory.Category.{u_16, u_2} C₂
    inst✝¹⁷ : CategoryTheory.Category.{u_14, u_5} C₃
    inst✝¹⁶ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝¹⁵ : CategoryTheory.Category.{u_15, u_3} C₁₂
    inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹² : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹¹ : CategoryTheory.Preadditive C₁₂
    inst✝¹⁰ : CategoryTheory.Preadditive C₄
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    inst✝⁹ : F₁₂.PreservesZeroMorphisms
    inst✝⁸ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
    inst✝⁷ : G.Additive
    inst✝⁶ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₄ : Type u_12
    inst✝⁵ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₄ : ComplexShape ι₄
    inst✝⁴ : TotalComplexShape c₁ c₂ c₁₂
    inst✝³ : TotalComplexShape c₁₂ c₃ c₄
    inst✝² : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
    inst✝¹ : DecidableEq ι₁₂
    inst✝ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
    i₁ : ι₁
    i₂ : ι₂
    i₃ : ι₃
    j : ι₄
    h : Not (c₃.Rel i₃ (c₃.next i₃))
    ⊢ Eq (HomologicalComplex.mapBifunctor₁₂.d₃ F₁₂ G K₁ K₂ K₃ c₁₂ c₄ i₁ i₂ i₃ j) 0
  -/
  dsimp [d₃]
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₁₂ : Type u_3
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝¹⁹ : CategoryTheory.Category.{u_17, u_1} C₁
    inst✝¹⁸ : CategoryTheory.Category.{u_16, u_2} C₂
    inst✝¹⁷ : CategoryTheory.Category.{u_14, u_5} C₃
    inst✝¹⁶ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝¹⁵ : CategoryTheory.Category.{u_15, u_3} C₁₂
    inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹² : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹¹ : CategoryTheory.Preadditive C₁₂
    inst✝¹⁰ : CategoryTheory.Preadditive C₄
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    inst✝⁹ : F₁₂.PreservesZeroMorphisms
    inst✝⁸ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
    inst✝⁷ : G.Additive
    inst✝⁶ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₄ : Type u_12
    inst✝⁵ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₄ : ComplexShape ι₄
    inst✝⁴ : TotalComplexShape c₁ c₂ c₁₂
    inst✝³ : TotalComplexShape c₁₂ c₃ c₄
    inst✝² : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
    inst✝¹ : DecidableEq ι₁₂
    inst✝ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
    i₁ : ι₁
    i₂ : ι₂
    i₃ : ι₃
    j : ι₄
    h : Not (c₃.Rel i₃ (c₃.next i₃))
    ⊢ Eq (HSMul.hSMul (c₁₂.ε₂ c₃ c₄ { fst := c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }, …
  -/
  rw [shape _ _ _ h, Functor.map_zero, zero_comp, smul_zero]
  /-
    🎉 no goals
  -/


lemma d₃_eq (i₁ : ι₁) (i₂ : ι₂) {i₃ i₃' : ι₃} (h₃ : c₃.Rel i₃ i₃') (j : ι₄) :
    d₃ F₁₂ G K₁ K₂ K₃ c₁₂ c₄ i₁ i₂ i₃ j =
  (ComplexShape.ε₂ c₁₂ c₃ c₄ (c₁.π c₂ c₁₂ (i₁, i₂), i₃)) •
    (G.obj ((F₁₂.obj (K₁.X i₁)).obj (K₂.X i₂))).map (K₃.d i₃ i₃') ≫
      ιOrZero F₁₂ G K₁ K₂ K₃ c₁₂ c₄ i₁ i₂ _ j := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₁₂ : Type u_3
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝¹⁹ : CategoryTheory.Category.{u_17, u_1} C₁
    inst✝¹⁸ : CategoryTheory.Category.{u_16, u_2} C₂
    inst✝¹⁷ : CategoryTheory.Category.{u_14, u_5} C₃
    inst✝¹⁶ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝¹⁵ : CategoryTheory.Category.{u_15, u_3} C₁₂
    inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹² : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹¹ : CategoryTheory.Preadditive C₁₂
    inst✝¹⁰ : CategoryTheory.Preadditive C₄
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    inst✝⁹ : F₁₂.PreservesZeroMorphisms
    inst✝⁸ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
    inst✝⁷ : G.Additive
    inst✝⁶ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₄ : Type u_12
    inst✝⁵ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₄ : ComplexShape ι₄
    inst✝⁴ : TotalComplexShape c₁ c₂ c₁₂
    inst✝³ : TotalComplexShape c₁₂ c₃ c₄
    inst✝² : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
    inst✝¹ : DecidableEq ι₁₂
    inst✝ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
    i₁ : ι₁
    i₂ : ι₂
    i₃ i₃' : ι₃
    h₃ : c₃.Rel i₃ i₃'
    j : ι₄
    ⊢ Eq (HomologicalComplex.mapBifunctor₁₂.d₃ F₁₂ G K₁ K₂ K₃ c₁₂ c₄ i₁ i₂ i₃ j) ( …
  -/
  obtain rfl := c₃.next_eq' h₃
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₁₂ : Type u_3
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝¹⁹ : CategoryTheory.Category.{u_17, u_1} C₁
    inst✝¹⁸ : CategoryTheory.Category.{u_16, u_2} C₂
    inst✝¹⁷ : CategoryTheory.Category.{u_14, u_5} C₃
    inst✝¹⁶ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝¹⁵ : CategoryTheory.Category.{u_15, u_3} C₁₂
    inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹² : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹¹ : CategoryTheory.Preadditive C₁₂
    inst✝¹⁰ : CategoryTheory.Preadditive C₄
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    inst✝⁹ : F₁₂.PreservesZeroMorphisms
    inst✝⁸ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
    inst✝⁷ : G.Additive
    inst✝⁶ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₄ : Type u_12
    inst✝⁵ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₄ : ComplexShape ι₄
    inst✝⁴ : TotalComplexShape c₁ c₂ c₁₂
    inst✝³ : TotalComplexShape c₁₂ c₃ c₄
    inst✝² : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
    inst✝¹ : DecidableEq ι₁₂
    inst✝ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
    i₁ : ι₁
    i₂ : ι₂
    i₃ : ι₃
    j : ι₄
    h₃ : c₃.Rel i₃ (c₃.next i₃)
    ⊢ Eq (HomologicalComplex.mapBifunctor₁₂.d₃ F₁₂ G K₁ K₂ K₃ c₁₂ c₄ i₁ i₂ i₃ j) ( …
  -/
  rfl
  /-
    🎉 no goals
  -/



/-- The first differential on `mapBifunctor (mapBifunctor K₁ K₂ F₁₂ c₁₂) K₃ G c₄`. -/
noncomputable def D₁ :
    (mapBifunctor (mapBifunctor K₁ K₂ F₁₂ c₁₂) K₃ G c₄).X j ⟶
      (mapBifunctor (mapBifunctor K₁ K₂ F₁₂ c₁₂) K₃ G c₄).X j' :=
  mapBifunctor₁₂Desc (fun i₁ i₂ i₃ _ ↦ d₁ F₁₂ G K₁ K₂ K₃ c₁₂ c₄ i₁ i₂ i₃ j')


/-- The second differential on `mapBifunctor (mapBifunctor K₁ K₂ F₁₂ c₁₂) K₃ G c₄`. -/
noncomputable def D₂ :
    (mapBifunctor (mapBifunctor K₁ K₂ F₁₂ c₁₂) K₃ G c₄).X j ⟶
      (mapBifunctor (mapBifunctor K₁ K₂ F₁₂ c₁₂) K₃ G c₄).X j' :=
  mapBifunctor₁₂Desc (fun i₁ i₂ i₃ _ ↦ d₂ F₁₂ G K₁ K₂ K₃ c₁₂ c₄ i₁ i₂ i₃ j')


/-- The third differential on `mapBifunctor (mapBifunctor K₁ K₂ F₁₂ c₁₂) K₃ G c₄`. -/
noncomputable def D₃ :
    (mapBifunctor (mapBifunctor K₁ K₂ F₁₂ c₁₂) K₃ G c₄).X j ⟶
      (mapBifunctor (mapBifunctor K₁ K₂ F₁₂ c₁₂) K₃ G c₄).X j' :=
  mapBifunctor.D₂ _ _ _ _ _ _


@[reassoc (attr := simp)]
lemma ι_D₁ [HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄] :
    ι F₁₂ G K₁ K₂ K₃ c₁₂ c₄ i₁ i₂ i₃ j h ≫ D₁ F₁₂ G K₁ K₂ K₃ c₁₂ c₄ j j' =
      d₁ F₁₂ G K₁ K₂ K₃ c₁₂ c₄ i₁ i₂ i₃ j' := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₁₂ : Type u_3
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝²⁰ : CategoryTheory.Category.{u_13, u_1} C₁
    inst✝¹⁹ : CategoryTheory.Category.{u_14, u_2} C₂
    inst✝¹⁸ : CategoryTheory.Category.{u_15, u_5} C₃
    inst✝¹⁷ : CategoryTheory.Category.{u_16, u_6} C₄
    inst✝¹⁶ : CategoryTheory.Category.{u_17, u_3} C₁₂
    inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹² : CategoryTheory.Preadditive C₁₂
    inst✝¹¹ : CategoryTheory.Preadditive C₄
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    inst✝¹⁰ : F₁₂.PreservesZeroMorphisms
    inst✝⁹ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
    inst✝⁸ : G.Additive
    inst✝⁷ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₄ : Type u_12
    inst✝⁶ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₄ : ComplexShape ι₄
    inst✝⁵ : TotalComplexShape c₁ c₂ c₁₂
    inst✝⁴ : TotalComplexShape c₁₂ c₃ c₄
    inst✝³ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
    inst✝² : DecidableEq ι₁₂
    inst✝¹ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
    i₁ : ι₁
    i₂ : ι₂
    i₃ : ι₃
    j j' : ι₄
    h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
    inst✝ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.mapBifunctor₁₂.ι  …
  -/
  simp [D₁]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma ι_D₂ [HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄] :
    ι F₁₂ G K₁ K₂ K₃ c₁₂ c₄ i₁ i₂ i₃ j h ≫ D₂ F₁₂ G K₁ K₂ K₃ c₁₂ c₄ j j' =
      d₂ F₁₂ G K₁ K₂ K₃ c₁₂ c₄ i₁ i₂ i₃ j' := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₁₂ : Type u_3
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝²⁰ : CategoryTheory.Category.{u_13, u_1} C₁
    inst✝¹⁹ : CategoryTheory.Category.{u_14, u_2} C₂
    inst✝¹⁸ : CategoryTheory.Category.{u_15, u_5} C₃
    inst✝¹⁷ : CategoryTheory.Category.{u_16, u_6} C₄
    inst✝¹⁶ : CategoryTheory.Category.{u_17, u_3} C₁₂
    inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹² : CategoryTheory.Preadditive C₁₂
    inst✝¹¹ : CategoryTheory.Preadditive C₄
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    inst✝¹⁰ : F₁₂.PreservesZeroMorphisms
    inst✝⁹ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
    inst✝⁸ : G.Additive
    inst✝⁷ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₄ : Type u_12
    inst✝⁶ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₄ : ComplexShape ι₄
    inst✝⁵ : TotalComplexShape c₁ c₂ c₁₂
    inst✝⁴ : TotalComplexShape c₁₂ c₃ c₄
    inst✝³ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
    inst✝² : DecidableEq ι₁₂
    inst✝¹ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
    i₁ : ι₁
    i₂ : ι₂
    i₃ : ι₃
    j j' : ι₄
    h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
    inst✝ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.mapBifunctor₁₂.ι  …
  -/
  simp [D₂]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma ι_D₃  :
    ι F₁₂ G K₁ K₂ K₃ c₁₂ c₄ i₁ i₂ i₃ j h ≫ D₃ F₁₂ G K₁ K₂ K₃ c₁₂ c₄ j j' =
      d₃ F₁₂ G K₁ K₂ K₃ c₁₂ c₄ i₁ i₂ i₃ j' := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₁₂ : Type u_3
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝¹⁹ : CategoryTheory.Category.{u_17, u_1} C₁
    inst✝¹⁸ : CategoryTheory.Category.{u_16, u_2} C₂
    inst✝¹⁷ : CategoryTheory.Category.{u_14, u_5} C₃
    inst✝¹⁶ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝¹⁵ : CategoryTheory.Category.{u_15, u_3} C₁₂
    inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹² : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹¹ : CategoryTheory.Preadditive C₁₂
    inst✝¹⁰ : CategoryTheory.Preadditive C₄
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    inst✝⁹ : F₁₂.PreservesZeroMorphisms
    inst✝⁸ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
    inst✝⁷ : G.Additive
    inst✝⁶ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₄ : Type u_12
    inst✝⁵ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₄ : ComplexShape ι₄
    inst✝⁴ : TotalComplexShape c₁ c₂ c₁₂
    inst✝³ : TotalComplexShape c₁₂ c₃ c₄
    inst✝² : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
    inst✝¹ : DecidableEq ι₁₂
    inst✝ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
    i₁ : ι₁
    i₂ : ι₂
    i₃ : ι₃
    j j' : ι₄
    h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.mapBifunctor₁₂.ι  …
  -/
  simp only [ι_eq _ _ _ _ _ _ _ _ _ _ _ _ rfl h, D₃, assoc, mapBifunctor.ι_D₂]
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₁₂ : Type u_3
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝¹⁹ : CategoryTheory.Category.{u_17, u_1} C₁
    inst✝¹⁸ : CategoryTheory.Category.{u_16, u_2} C₂
    inst✝¹⁷ : CategoryTheory.Category.{u_14, u_5} C₃
    inst✝¹⁶ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝¹⁵ : CategoryTheory.Category.{u_15, u_3} C₁₂
    inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹² : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹¹ : CategoryTheory.Preadditive C₁₂
    inst✝¹⁰ : CategoryTheory.Preadditive C₄
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    inst✝⁹ : F₁₂.PreservesZeroMorphisms
    inst✝⁸ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
    inst✝⁷ : G.Additive
    inst✝⁶ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₄ : Type u_12
    inst✝⁵ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₄ : ComplexShape ι₄
    inst✝⁴ : TotalComplexShape c₁ c₂ c₁₂
    inst✝³ : TotalComplexShape c₁₂ c₃ c₄
    inst✝² : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
    inst✝¹ : DecidableEq ι₁₂
    inst✝ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
    i₁ : ι₁
    i₂ : ι₂
    i₃ : ι₃
    j j' : ι₄
    h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((G.map (K₁.ιMapBifunctor K₂ F₁₂ c₁₂  …
  -/
  by_cases h₁ : c₃.Rel i₃ (c₃.next i₃)
    /-
      case pos
      C₁ : Type u_1
      C₂ : Type u_2
      C₁₂ : Type u_3
      C₃ : Type u_5
      C₄ : Type u_6
      inst✝¹⁹ : CategoryTheory.Category.{u_17, u_1} C₁
      inst✝¹⁸ : CategoryTheory.Category.{u_16, u_2} C₂
      inst✝¹⁷ : CategoryTheory.Category.{u_14, u_5} C₃
      inst✝¹⁶ : CategoryTheory.Category.{u_13, u_6} C₄
      inst✝¹⁵ : CategoryTheory.Category.{u_15, u_3} C₁₂
      inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₁
      inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₂
      inst✝¹² : CategoryTheory.Limits.HasZeroMorphisms C₃
      inst✝¹¹ : CategoryTheory.Preadditive C₁₂
      inst✝¹⁰ : CategoryTheory.Preadditive C₄
      F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
      G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
      inst✝⁹ : F₁₂.PreservesZeroMorphisms
      inst✝⁸ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
      inst✝⁷ : G.Additive
      inst✝⁶ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
      ι₁ : Type u_7
      ι₂ : Type u_8
      ι₃ : Type u_9
      ι₁₂ : Type u_10
      ι₄ : Type u_12
      inst✝⁵ : DecidableEq ι₄
      c₁ : ComplexShape ι₁
      c₂ : ComplexShape ι₂
      c₃ : ComplexShape ι₃
      K₁ : HomologicalComplex C₁ c₁
      K₂ : HomologicalComplex C₂ c₂
      K₃ : HomologicalComplex C₃ c₃
      c₁₂ : ComplexShape ι₁₂
      c₄ : ComplexShape ι₄
      inst✝⁴ : TotalComplexShape c₁ c₂ c₁₂
      inst✝³ : TotalComplexShape c₁₂ c₃ c₄
      inst✝² : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
      inst✝¹ : DecidableEq ι₁₂
      inst✝ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
      i₁ : ι₁
      i₂ : ι₂
      i₃ : ι₃
      j j' : ι₄
      h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
      h₁ : c₃.Rel i₃ (c₃.next i₃)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((G.map (K₁.ιMapBifunctor K₂ F₁₂ c₁₂  …
    -/
  · rw [d₃_eq _ _ _ _ _ _ _ _ _ h₁]
    /-
      case pos
      C₁ : Type u_1
      C₂ : Type u_2
      C₁₂ : Type u_3
      C₃ : Type u_5
      C₄ : Type u_6
      inst✝¹⁹ : CategoryTheory.Category.{u_17, u_1} C₁
      inst✝¹⁸ : CategoryTheory.Category.{u_16, u_2} C₂
      inst✝¹⁷ : CategoryTheory.Category.{u_14, u_5} C₃
      inst✝¹⁶ : CategoryTheory.Category.{u_13, u_6} C₄
      inst✝¹⁵ : CategoryTheory.Category.{u_15, u_3} C₁₂
      inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₁
      inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₂
      inst✝¹² : CategoryTheory.Limits.HasZeroMorphisms C₃
      inst✝¹¹ : CategoryTheory.Preadditive C₁₂
      inst✝¹⁰ : CategoryTheory.Preadditive C₄
      F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
      G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
      inst✝⁹ : F₁₂.PreservesZeroMorphisms
      inst✝⁸ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
      inst✝⁷ : G.Additive
      inst✝⁶ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
      ι₁ : Type u_7
      ι₂ : Type u_8
      ι₃ : Type u_9
      ι₁₂ : Type u_10
      ι₄ : Type u_12
      inst✝⁵ : DecidableEq ι₄
      c₁ : ComplexShape ι₁
      c₂ : ComplexShape ι₂
      c₃ : ComplexShape ι₃
      K₁ : HomologicalComplex C₁ c₁
      K₂ : HomologicalComplex C₂ c₂
      K₃ : HomologicalComplex C₃ c₃
      c₁₂ : ComplexShape ι₁₂
      c₄ : ComplexShape ι₄
      inst✝⁴ : TotalComplexShape c₁ c₂ c₁₂
      inst✝³ : TotalComplexShape c₁₂ c₃ c₄
      inst✝² : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
      inst✝¹ : DecidableEq ι₁₂
      inst✝ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
      i₁ : ι₁
      i₂ : ι₂
      i₃ : ι₃
      j j' : ι₄
      h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
      h₁ : c₃.Rel i₃ (c₃.next i₃)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((G.map (K₁.ιMapBifunctor K₂ F₁₂ c₁₂  …
    -/
    by_cases h₂ : ComplexShape.π c₁₂ c₃ c₄ (c₁.π c₂ c₁₂ (i₁, i₂), c₃.next i₃) = j'
    · rw [mapBifunctor.d₂_eq _ _ _ _ _ h₁ _ h₂,
        ιOrZero_eq _ _ _ _ _ _ _ _ _ _ _ h₂,
        Linear.comp_units_smul, smul_left_cancel_iff,
        ι_eq _ _ _ _ _ _ _ _ _ _ _ _ rfl h₂,
        NatTrans.naturality_assoc]
    · rw [mapBifunctor.d₂_eq_zero' _ _ _ _ _ h₁ _ h₂, comp_zero,
        ιOrZero_eq_zero _ _ _ _ _ _ _ _ _ _ _ h₂, comp_zero, smul_zero]
  · rw [mapBifunctor.d₂_eq_zero _ _ _ _ _ _ _ h₁, comp_zero,
      d₃_eq_zero _ _ _ _ _ _ _ _ _ _ _ h₁]


lemma d_eq (j j' : ι₄) [HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄] :
    (mapBifunctor (mapBifunctor K₁ K₂ F₁₂ c₁₂) K₃ G c₄).d j j' =
      D₁ F₁₂ G K₁ K₂ K₃ c₁₂ c₄ j j' + D₂ F₁₂ G K₁ K₂ K₃ c₁₂ c₄ j j' +
        D₃ F₁₂ G K₁ K₂ K₃ c₁₂ c₄ j j' := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₁₂ : Type u_3
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝²⁰ : CategoryTheory.Category.{u_13, u_1} C₁
    inst✝¹⁹ : CategoryTheory.Category.{u_14, u_2} C₂
    inst✝¹⁸ : CategoryTheory.Category.{u_15, u_5} C₃
    inst✝¹⁷ : CategoryTheory.Category.{u_16, u_6} C₄
    inst✝¹⁶ : CategoryTheory.Category.{u_17, u_3} C₁₂
    inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹² : CategoryTheory.Preadditive C₁₂
    inst✝¹¹ : CategoryTheory.Preadditive C₄
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    inst✝¹⁰ : F₁₂.PreservesZeroMorphisms
    inst✝⁹ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
    inst✝⁸ : G.Additive
    inst✝⁷ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₄ : Type u_12
    inst✝⁶ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₄ : ComplexShape ι₄
    inst✝⁵ : TotalComplexShape c₁ c₂ c₁₂
    inst✝⁴ : TotalComplexShape c₁₂ c₃ c₄
    inst✝³ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
    inst✝² : DecidableEq ι₁₂
    inst✝¹ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
    j j' : ι₄
    inst✝ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
    ⊢ Eq (((K₁.mapBifunctor K₂ F₁₂ c₁₂).mapBifunctor K₃ G c₄).d j j') (HAdd.hAdd ( …
  -/
  rw [mapBifunctor.d_eq]
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₁₂ : Type u_3
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝²⁰ : CategoryTheory.Category.{u_13, u_1} C₁
    inst✝¹⁹ : CategoryTheory.Category.{u_14, u_2} C₂
    inst✝¹⁸ : CategoryTheory.Category.{u_15, u_5} C₃
    inst✝¹⁷ : CategoryTheory.Category.{u_16, u_6} C₄
    inst✝¹⁶ : CategoryTheory.Category.{u_17, u_3} C₁₂
    inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹² : CategoryTheory.Preadditive C₁₂
    inst✝¹¹ : CategoryTheory.Preadditive C₄
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    inst✝¹⁰ : F₁₂.PreservesZeroMorphisms
    inst✝⁹ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
    inst✝⁸ : G.Additive
    inst✝⁷ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₄ : Type u_12
    inst✝⁶ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₄ : ComplexShape ι₄
    inst✝⁵ : TotalComplexShape c₁ c₂ c₁₂
    inst✝⁴ : TotalComplexShape c₁₂ c₃ c₄
    inst✝³ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
    inst✝² : DecidableEq ι₁₂
    inst✝¹ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
    j j' : ι₄
    inst✝ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
    ⊢ Eq (HAdd.hAdd (HomologicalComplex.mapBifunctor.D₁ (K₁.mapBifunctor K₂ F₁₂ c₁ …
  -/
  congr 1
  /-
    case e_a
    C₁ : Type u_1
    C₂ : Type u_2
    C₁₂ : Type u_3
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝²⁰ : CategoryTheory.Category.{u_13, u_1} C₁
    inst✝¹⁹ : CategoryTheory.Category.{u_14, u_2} C₂
    inst✝¹⁸ : CategoryTheory.Category.{u_15, u_5} C₃
    inst✝¹⁷ : CategoryTheory.Category.{u_16, u_6} C₄
    inst✝¹⁶ : CategoryTheory.Category.{u_17, u_3} C₁₂
    inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹² : CategoryTheory.Preadditive C₁₂
    inst✝¹¹ : CategoryTheory.Preadditive C₄
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    inst✝¹⁰ : F₁₂.PreservesZeroMorphisms
    inst✝⁹ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
    inst✝⁸ : G.Additive
    inst✝⁷ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₄ : Type u_12
    inst✝⁶ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₄ : ComplexShape ι₄
    inst✝⁵ : TotalComplexShape c₁ c₂ c₁₂
    inst✝⁴ : TotalComplexShape c₁₂ c₃ c₄
    inst✝³ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
    inst✝² : DecidableEq ι₁₂
    inst✝¹ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
    j j' : ι₄
    inst✝ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
    ⊢ Eq (HomologicalComplex.mapBifunctor.D₁ (K₁.mapBifunctor K₂ F₁₂ c₁₂) K₃ G c₄  …
  -/
  ext i₁ i₂ i₃ h
  /-
    case e_a.hfg
    C₁ : Type u_1
    C₂ : Type u_2
    C₁₂ : Type u_3
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝²⁰ : CategoryTheory.Category.{u_13, u_1} C₁
    inst✝¹⁹ : CategoryTheory.Category.{u_14, u_2} C₂
    inst✝¹⁸ : CategoryTheory.Category.{u_15, u_5} C₃
    inst✝¹⁷ : CategoryTheory.Category.{u_16, u_6} C₄
    inst✝¹⁶ : CategoryTheory.Category.{u_17, u_3} C₁₂
    inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹² : CategoryTheory.Preadditive C₁₂
    inst✝¹¹ : CategoryTheory.Preadditive C₄
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    inst✝¹⁰ : F₁₂.PreservesZeroMorphisms
    inst✝⁹ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
    inst✝⁸ : G.Additive
    inst✝⁷ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₄ : Type u_12
    inst✝⁶ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₄ : ComplexShape ι₄
    inst✝⁵ : TotalComplexShape c₁ c₂ c₁₂
    inst✝⁴ : TotalComplexShape c₁₂ c₃ c₄
    inst✝³ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
    inst✝² : DecidableEq ι₁₂
    inst✝¹ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
    j j' : ι₄
    inst✝ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
    i₁ : ι₁
    i₂ : ι₂
    i₃ : ι₃
    h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.mapBifunctor₁₂.ι  …
  -/
  simp only [Preadditive.comp_add, ι_D₁, ι_D₂]
  /-
    case e_a.hfg
    C₁ : Type u_1
    C₂ : Type u_2
    C₁₂ : Type u_3
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝²⁰ : CategoryTheory.Category.{u_13, u_1} C₁
    inst✝¹⁹ : CategoryTheory.Category.{u_14, u_2} C₂
    inst✝¹⁸ : CategoryTheory.Category.{u_15, u_5} C₃
    inst✝¹⁷ : CategoryTheory.Category.{u_16, u_6} C₄
    inst✝¹⁶ : CategoryTheory.Category.{u_17, u_3} C₁₂
    inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹² : CategoryTheory.Preadditive C₁₂
    inst✝¹¹ : CategoryTheory.Preadditive C₄
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    inst✝¹⁰ : F₁₂.PreservesZeroMorphisms
    inst✝⁹ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
    inst✝⁸ : G.Additive
    inst✝⁷ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₄ : Type u_12
    inst✝⁶ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₄ : ComplexShape ι₄
    inst✝⁵ : TotalComplexShape c₁ c₂ c₁₂
    inst✝⁴ : TotalComplexShape c₁₂ c₃ c₄
    inst✝³ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
    inst✝² : DecidableEq ι₁₂
    inst✝¹ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
    j j' : ι₄
    inst✝ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
    i₁ : ι₁
    i₂ : ι₂
    i₃ : ι₃
    h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.mapBifunctor₁₂.ι  …
  -/
  rw [ι_eq _ _ _ _ _ _ _ _ _ _ _ _ rfl h, assoc, mapBifunctor.ι_D₁]
  /-
    case e_a.hfg
    C₁ : Type u_1
    C₂ : Type u_2
    C₁₂ : Type u_3
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝²⁰ : CategoryTheory.Category.{u_13, u_1} C₁
    inst✝¹⁹ : CategoryTheory.Category.{u_14, u_2} C₂
    inst✝¹⁸ : CategoryTheory.Category.{u_15, u_5} C₃
    inst✝¹⁷ : CategoryTheory.Category.{u_16, u_6} C₄
    inst✝¹⁶ : CategoryTheory.Category.{u_17, u_3} C₁₂
    inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹² : CategoryTheory.Preadditive C₁₂
    inst✝¹¹ : CategoryTheory.Preadditive C₄
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    inst✝¹⁰ : F₁₂.PreservesZeroMorphisms
    inst✝⁹ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
    inst✝⁸ : G.Additive
    inst✝⁷ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₄ : Type u_12
    inst✝⁶ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₄ : ComplexShape ι₄
    inst✝⁵ : TotalComplexShape c₁ c₂ c₁₂
    inst✝⁴ : TotalComplexShape c₁₂ c₃ c₄
    inst✝³ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
    inst✝² : DecidableEq ι₁₂
    inst✝¹ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
    j j' : ι₄
    inst✝ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
    i₁ : ι₁
    i₂ : ι₂
    i₃ : ι₃
    h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((G.map (K₁.ιMapBifunctor K₂ F₁₂ c₁₂  …
  -/
  set i₁₂ := ComplexShape.π c₁ c₂ c₁₂ ⟨i₁, i₂⟩
  /-
    case e_a.hfg
    C₁ : Type u_1
    C₂ : Type u_2
    C₁₂ : Type u_3
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝²⁰ : CategoryTheory.Category.{u_13, u_1} C₁
    inst✝¹⁹ : CategoryTheory.Category.{u_14, u_2} C₂
    inst✝¹⁸ : CategoryTheory.Category.{u_15, u_5} C₃
    inst✝¹⁷ : CategoryTheory.Category.{u_16, u_6} C₄
    inst✝¹⁶ : CategoryTheory.Category.{u_17, u_3} C₁₂
    inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹² : CategoryTheory.Preadditive C₁₂
    inst✝¹¹ : CategoryTheory.Preadditive C₄
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    inst✝¹⁰ : F₁₂.PreservesZeroMorphisms
    inst✝⁹ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
    inst✝⁸ : G.Additive
    inst✝⁷ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₄ : Type u_12
    inst✝⁶ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₄ : ComplexShape ι₄
    inst✝⁵ : TotalComplexShape c₁ c₂ c₁₂
    inst✝⁴ : TotalComplexShape c₁₂ c₃ c₄
    inst✝³ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
    inst✝² : DecidableEq ι₁₂
    inst✝¹ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
    j j' : ι₄
    inst✝ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
    i₁ : ι₁
    i₂ : ι₂
    i₃ : ι₃
    h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
    i₁₂ : ι₁₂ := c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((G.map (K₁.ιMapBifunctor K₂ F₁₂ c₁₂  …
  -/
  by_cases h₁ : c₁₂.Rel i₁₂ (c₁₂.next i₁₂)
    /-
      case pos
      C₁ : Type u_1
      C₂ : Type u_2
      C₁₂ : Type u_3
      C₃ : Type u_5
      C₄ : Type u_6
      inst✝²⁰ : CategoryTheory.Category.{u_13, u_1} C₁
      inst✝¹⁹ : CategoryTheory.Category.{u_14, u_2} C₂
      inst✝¹⁸ : CategoryTheory.Category.{u_15, u_5} C₃
      inst✝¹⁷ : CategoryTheory.Category.{u_16, u_6} C₄
      inst✝¹⁶ : CategoryTheory.Category.{u_17, u_3} C₁₂
      inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
      inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
      inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₃
      inst✝¹² : CategoryTheory.Preadditive C₁₂
      inst✝¹¹ : CategoryTheory.Preadditive C₄
      F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
      G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
      inst✝¹⁰ : F₁₂.PreservesZeroMorphisms
      inst✝⁹ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
      inst✝⁸ : G.Additive
      inst✝⁷ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
      ι₁ : Type u_7
      ι₂ : Type u_8
      ι₃ : Type u_9
      ι₁₂ : Type u_10
      ι₄ : Type u_12
      inst✝⁶ : DecidableEq ι₄
      c₁ : ComplexShape ι₁
      c₂ : ComplexShape ι₂
      c₃ : ComplexShape ι₃
      K₁ : HomologicalComplex C₁ c₁
      K₂ : HomologicalComplex C₂ c₂
      K₃ : HomologicalComplex C₃ c₃
      c₁₂ : ComplexShape ι₁₂
      c₄ : ComplexShape ι₄
      inst✝⁵ : TotalComplexShape c₁ c₂ c₁₂
      inst✝⁴ : TotalComplexShape c₁₂ c₃ c₄
      inst✝³ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
      inst✝² : DecidableEq ι₁₂
      inst✝¹ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
      j j' : ι₄
      inst✝ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
      i₁ : ι₁
      i₂ : ι₂
      i₃ : ι₃
      h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
      i₁₂ : ι₁₂ := c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }
      h₁ : c₁₂.Rel i₁₂ (c₁₂.next i₁₂)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((G.map (K₁.ιMapBifunctor K₂ F₁₂ c₁₂  …
    -/
  · by_cases h₂ : ComplexShape.π c₁₂ c₃ c₄ (c₁₂.next i₁₂, i₃) = j'
      /-
        case pos
        C₁ : Type u_1
        C₂ : Type u_2
        C₁₂ : Type u_3
        C₃ : Type u_5
        C₄ : Type u_6
        inst✝²⁰ : CategoryTheory.Category.{u_13, u_1} C₁
        inst✝¹⁹ : CategoryTheory.Category.{u_14, u_2} C₂
        inst✝¹⁸ : CategoryTheory.Category.{u_15, u_5} C₃
        inst✝¹⁷ : CategoryTheory.Category.{u_16, u_6} C₄
        inst✝¹⁶ : CategoryTheory.Category.{u_17, u_3} C₁₂
        inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
        inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
        inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₃
        inst✝¹² : CategoryTheory.Preadditive C₁₂
        inst✝¹¹ : CategoryTheory.Preadditive C₄
        F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
        G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
        inst✝¹⁰ : F₁₂.PreservesZeroMorphisms
        inst✝⁹ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
        inst✝⁸ : G.Additive
        inst✝⁷ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
        ι₁ : Type u_7
        ι₂ : Type u_8
        ι₃ : Type u_9
        ι₁₂ : Type u_10
        ι₄ : Type u_12
        inst✝⁶ : DecidableEq ι₄
        c₁ : ComplexShape ι₁
        c₂ : ComplexShape ι₂
        c₃ : ComplexShape ι₃
        K₁ : HomologicalComplex C₁ c₁
        K₂ : HomologicalComplex C₂ c₂
        K₃ : HomologicalComplex C₃ c₃
        c₁₂ : ComplexShape ι₁₂
        c₄ : ComplexShape ι₄
        inst✝⁵ : TotalComplexShape c₁ c₂ c₁₂
        inst✝⁴ : TotalComplexShape c₁₂ c₃ c₄
        inst✝³ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
        inst✝² : DecidableEq ι₁₂
        inst✝¹ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
        j j' : ι₄
        inst✝ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
        i₁ : ι₁
        i₂ : ι₂
        i₃ : ι₃
        h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
        i₁₂ : ι₁₂ := c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }
        h₁ : c₁₂.Rel i₁₂ (c₁₂.next i₁₂)
        h₂ : Eq (c₁₂.π c₃ c₄ { fst := c₁₂.next i₁₂, snd := i₃ }) j'
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((G.map (K₁.ιMapBifunctor K₂ F₁₂ c₁₂  …
      -/
    · rw [mapBifunctor.d₁_eq _ _ _ _ h₁ _ _ h₂]
      simp only [i₁₂, mapBifunctor.d_eq, Functor.map_add, NatTrans.app_add,
        Preadditive.add_comp, smul_add, Preadditive.comp_add, Linear.comp_units_smul]
      /-
        case pos
        C₁ : Type u_1
        C₂ : Type u_2
        C₁₂ : Type u_3
        C₃ : Type u_5
        C₄ : Type u_6
        inst✝²⁰ : CategoryTheory.Category.{u_13, u_1} C₁
        inst✝¹⁹ : CategoryTheory.Category.{u_14, u_2} C₂
        inst✝¹⁸ : CategoryTheory.Category.{u_15, u_5} C₃
        inst✝¹⁷ : CategoryTheory.Category.{u_16, u_6} C₄
        inst✝¹⁶ : CategoryTheory.Category.{u_17, u_3} C₁₂
        inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
        inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
        inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₃
        inst✝¹² : CategoryTheory.Preadditive C₁₂
        inst✝¹¹ : CategoryTheory.Preadditive C₄
        F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
        G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
        inst✝¹⁰ : F₁₂.PreservesZeroMorphisms
        inst✝⁹ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
        inst✝⁸ : G.Additive
        inst✝⁷ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
        ι₁ : Type u_7
        ι₂ : Type u_8
        ι₃ : Type u_9
        ι₁₂ : Type u_10
        ι₄ : Type u_12
        inst✝⁶ : DecidableEq ι₄
        c₁ : ComplexShape ι₁
        c₂ : ComplexShape ι₂
        c₃ : ComplexShape ι₃
        K₁ : HomologicalComplex C₁ c₁
        K₂ : HomologicalComplex C₂ c₂
        K₃ : HomologicalComplex C₃ c₃
        c₁₂ : ComplexShape ι₁₂
        c₄ : ComplexShape ι₄
        inst✝⁵ : TotalComplexShape c₁ c₂ c₁₂
        inst✝⁴ : TotalComplexShape c₁₂ c₃ c₄
        inst✝³ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
        inst✝² : DecidableEq ι₁₂
        inst✝¹ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
        j j' : ι₄
        inst✝ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
        i₁ : ι₁
        i₂ : ι₂
        i₃ : ι₃
        h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
        i₁₂ : ι₁₂ := c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }
        h₁ : c₁₂.Rel i₁₂ (c₁₂.next i₁₂)
        h₂ : Eq (c₁₂.π c₃ c₄ { fst := c₁₂.next i₁₂, snd := i₃ }) j'
        ⊢ Eq (HAdd.hAdd (HSMul.hSMul (c₁₂.ε₁ c₃ c₄ { fst := c₁.π c₂ c₁₂ { fst := i₁, s …
      -/
      congr 1
      · rw [← NatTrans.comp_app_assoc, ← Functor.map_comp,
          mapBifunctor.ι_D₁]
        /-
          case pos.e_a
          C₁ : Type u_1
          C₂ : Type u_2
          C₁₂ : Type u_3
          C₃ : Type u_5
          C₄ : Type u_6
          inst✝²⁰ : CategoryTheory.Category.{u_13, u_1} C₁
          inst✝¹⁹ : CategoryTheory.Category.{u_14, u_2} C₂
          inst✝¹⁸ : CategoryTheory.Category.{u_15, u_5} C₃
          inst✝¹⁷ : CategoryTheory.Category.{u_16, u_6} C₄
          inst✝¹⁶ : CategoryTheory.Category.{u_17, u_3} C₁₂
          inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
          inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
          inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₃
          inst✝¹² : CategoryTheory.Preadditive C₁₂
          inst✝¹¹ : CategoryTheory.Preadditive C₄
          F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
          G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
          inst✝¹⁰ : F₁₂.PreservesZeroMorphisms
          inst✝⁹ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
          inst✝⁸ : G.Additive
          inst✝⁷ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
          ι₁ : Type u_7
          ι₂ : Type u_8
          ι₃ : Type u_9
          ι₁₂ : Type u_10
          ι₄ : Type u_12
          inst✝⁶ : DecidableEq ι₄
          c₁ : ComplexShape ι₁
          c₂ : ComplexShape ι₂
          c₃ : ComplexShape ι₃
          K₁ : HomologicalComplex C₁ c₁
          K₂ : HomologicalComplex C₂ c₂
          K₃ : HomologicalComplex C₃ c₃
          c₁₂ : ComplexShape ι₁₂
          c₄ : ComplexShape ι₄
          inst✝⁵ : TotalComplexShape c₁ c₂ c₁₂
          inst✝⁴ : TotalComplexShape c₁₂ c₃ c₄
          inst✝³ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
          inst✝² : DecidableEq ι₁₂
          inst✝¹ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
          j j' : ι₄
          inst✝ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
          i₁ : ι₁
          i₂ : ι₂
          i₃ : ι₃
          h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
          i₁₂ : ι₁₂ := c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }
          h₁ : c₁₂.Rel i₁₂ (c₁₂.next i₁₂)
          h₂ : Eq (c₁₂.π c₃ c₄ { fst := c₁₂.next i₁₂, snd := i₃ }) j'
          ⊢ Eq (HSMul.hSMul (c₁₂.ε₁ c₃ c₄ { fst := c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }, …
        -/
        by_cases h₃ : c₁.Rel i₁ (c₁.next i₁)
          /-
            case pos
            C₁ : Type u_1
            C₂ : Type u_2
            C₁₂ : Type u_3
            C₃ : Type u_5
            C₄ : Type u_6
            inst✝²⁰ : CategoryTheory.Category.{u_13, u_1} C₁
            inst✝¹⁹ : CategoryTheory.Category.{u_14, u_2} C₂
            inst✝¹⁸ : CategoryTheory.Category.{u_15, u_5} C₃
            inst✝¹⁷ : CategoryTheory.Category.{u_16, u_6} C₄
            inst✝¹⁶ : CategoryTheory.Category.{u_17, u_3} C₁₂
            inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
            inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
            inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₃
            inst✝¹² : CategoryTheory.Preadditive C₁₂
            inst✝¹¹ : CategoryTheory.Preadditive C₄
            F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
            G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
            inst✝¹⁰ : F₁₂.PreservesZeroMorphisms
            inst✝⁹ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
            inst✝⁸ : G.Additive
            inst✝⁷ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
            ι₁ : Type u_7
            ι₂ : Type u_8
            ι₃ : Type u_9
            ι₁₂ : Type u_10
            ι₄ : Type u_12
            inst✝⁶ : DecidableEq ι₄
            c₁ : ComplexShape ι₁
            c₂ : ComplexShape ι₂
            c₃ : ComplexShape ι₃
            K₁ : HomologicalComplex C₁ c₁
            K₂ : HomologicalComplex C₂ c₂
            K₃ : HomologicalComplex C₃ c₃
            c₁₂ : ComplexShape ι₁₂
            c₄ : ComplexShape ι₄
            inst✝⁵ : TotalComplexShape c₁ c₂ c₁₂
            inst✝⁴ : TotalComplexShape c₁₂ c₃ c₄
            inst✝³ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
            inst✝² : DecidableEq ι₁₂
            inst✝¹ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
            j j' : ι₄
            inst✝ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
            i₁ : ι₁
            i₂ : ι₂
            i₃ : ι₃
            h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
            i₁₂ : ι₁₂ := c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }
            h₁ : c₁₂.Rel i₁₂ (c₁₂.next i₁₂)
            h₂ : Eq (c₁₂.π c₃ c₄ { fst := c₁₂.next i₁₂, snd := i₃ }) j'
            h₃ : c₁.Rel i₁ (c₁.next i₁)
            ⊢ Eq (HSMul.hSMul (c₁₂.ε₁ c₃ c₄ { fst := c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }, …
          -/
        · have h₄ := (ComplexShape.next_π₁ c₂ c₁₂ h₃ i₂).symm
          rw [mapBifunctor.d₁_eq _ _ _ _ h₃ _ _ h₄,
            d₁_eq _ _ _ _ _ _ _ h₃,
            ιOrZero_eq _ _ _ _ _ _ _ _ _ _ _ (by rw [← h₂, ← h₄]; rfl),
            ι_eq _ _ _ _ _ _ _ _ _ _ (c₁₂.next i₁₂) _ h₄ h₂,
            Functor.map_units_smul, Functor.map_comp, NatTrans.app_units_zsmul,
            NatTrans.comp_app, Linear.units_smul_comp, assoc, smul_smul]
        · rw [d₁_eq_zero _ _ _ _ _ _ _ _ _ _ _ h₃,
            mapBifunctor.d₁_eq_zero _ _ _ _ _ _ _ h₃,
            Functor.map_zero, zero_app, zero_comp, smul_zero]
      · rw [← NatTrans.comp_app_assoc, ← Functor.map_comp,
          mapBifunctor.ι_D₂]
        /-
          case pos.e_a
          C₁ : Type u_1
          C₂ : Type u_2
          C₁₂ : Type u_3
          C₃ : Type u_5
          C₄ : Type u_6
          inst✝²⁰ : CategoryTheory.Category.{u_13, u_1} C₁
          inst✝¹⁹ : CategoryTheory.Category.{u_14, u_2} C₂
          inst✝¹⁸ : CategoryTheory.Category.{u_15, u_5} C₃
          inst✝¹⁷ : CategoryTheory.Category.{u_16, u_6} C₄
          inst✝¹⁶ : CategoryTheory.Category.{u_17, u_3} C₁₂
          inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
          inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
          inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₃
          inst✝¹² : CategoryTheory.Preadditive C₁₂
          inst✝¹¹ : CategoryTheory.Preadditive C₄
          F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
          G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
          inst✝¹⁰ : F₁₂.PreservesZeroMorphisms
          inst✝⁹ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
          inst✝⁸ : G.Additive
          inst✝⁷ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
          ι₁ : Type u_7
          ι₂ : Type u_8
          ι₃ : Type u_9
          ι₁₂ : Type u_10
          ι₄ : Type u_12
          inst✝⁶ : DecidableEq ι₄
          c₁ : ComplexShape ι₁
          c₂ : ComplexShape ι₂
          c₃ : ComplexShape ι₃
          K₁ : HomologicalComplex C₁ c₁
          K₂ : HomologicalComplex C₂ c₂
          K₃ : HomologicalComplex C₃ c₃
          c₁₂ : ComplexShape ι₁₂
          c₄ : ComplexShape ι₄
          inst✝⁵ : TotalComplexShape c₁ c₂ c₁₂
          inst✝⁴ : TotalComplexShape c₁₂ c₃ c₄
          inst✝³ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
          inst✝² : DecidableEq ι₁₂
          inst✝¹ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
          j j' : ι₄
          inst✝ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
          i₁ : ι₁
          i₂ : ι₂
          i₃ : ι₃
          h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
          i₁₂ : ι₁₂ := c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }
          h₁ : c₁₂.Rel i₁₂ (c₁₂.next i₁₂)
          h₂ : Eq (c₁₂.π c₃ c₄ { fst := c₁₂.next i₁₂, snd := i₃ }) j'
          ⊢ Eq (HSMul.hSMul (c₁₂.ε₁ c₃ c₄ { fst := c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }, …
        -/
        by_cases h₃ : c₂.Rel i₂ (c₂.next i₂)
          /-
            case pos
            C₁ : Type u_1
            C₂ : Type u_2
            C₁₂ : Type u_3
            C₃ : Type u_5
            C₄ : Type u_6
            inst✝²⁰ : CategoryTheory.Category.{u_13, u_1} C₁
            inst✝¹⁹ : CategoryTheory.Category.{u_14, u_2} C₂
            inst✝¹⁸ : CategoryTheory.Category.{u_15, u_5} C₃
            inst✝¹⁷ : CategoryTheory.Category.{u_16, u_6} C₄
            inst✝¹⁶ : CategoryTheory.Category.{u_17, u_3} C₁₂
            inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
            inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
            inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₃
            inst✝¹² : CategoryTheory.Preadditive C₁₂
            inst✝¹¹ : CategoryTheory.Preadditive C₄
            F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
            G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
            inst✝¹⁰ : F₁₂.PreservesZeroMorphisms
            inst✝⁹ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
            inst✝⁸ : G.Additive
            inst✝⁷ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
            ι₁ : Type u_7
            ι₂ : Type u_8
            ι₃ : Type u_9
            ι₁₂ : Type u_10
            ι₄ : Type u_12
            inst✝⁶ : DecidableEq ι₄
            c₁ : ComplexShape ι₁
            c₂ : ComplexShape ι₂
            c₃ : ComplexShape ι₃
            K₁ : HomologicalComplex C₁ c₁
            K₂ : HomologicalComplex C₂ c₂
            K₃ : HomologicalComplex C₃ c₃
            c₁₂ : ComplexShape ι₁₂
            c₄ : ComplexShape ι₄
            inst✝⁵ : TotalComplexShape c₁ c₂ c₁₂
            inst✝⁴ : TotalComplexShape c₁₂ c₃ c₄
            inst✝³ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
            inst✝² : DecidableEq ι₁₂
            inst✝¹ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
            j j' : ι₄
            inst✝ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
            i₁ : ι₁
            i₂ : ι₂
            i₃ : ι₃
            h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
            i₁₂ : ι₁₂ := c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }
            h₁ : c₁₂.Rel i₁₂ (c₁₂.next i₁₂)
            h₂ : Eq (c₁₂.π c₃ c₄ { fst := c₁₂.next i₁₂, snd := i₃ }) j'
            h₃ : c₂.Rel i₂ (c₂.next i₂)
            ⊢ Eq (HSMul.hSMul (c₁₂.ε₁ c₃ c₄ { fst := c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }, …
          -/
        · have h₄ := (ComplexShape.next_π₂ c₁ c₁₂ i₁ h₃).symm
          rw [mapBifunctor.d₂_eq _ _ _ _ _ h₃ _ h₄,
            d₂_eq _ _ _ _ _ _ _ _ h₃,
            ιOrZero_eq _ _ _ _ _ _ _ _ _ _ _ (by rw [← h₂, ← h₄]; rfl),
            ι_eq _ _ _ _ _ _ _ _ _ _ (c₁₂.next i₁₂) _ h₄ h₂,
            Functor.map_units_smul, Functor.map_comp, NatTrans.app_units_zsmul,
            NatTrans.comp_app, Linear.units_smul_comp, assoc, smul_smul]
        · rw [d₂_eq_zero _ _ _ _ _ _ _ _ _ _ _ h₃,
            mapBifunctor.d₂_eq_zero _ _ _ _ _ _ _ h₃,
            Functor.map_zero, zero_app, zero_comp, smul_zero]
      /-
        case neg
        C₁ : Type u_1
        C₂ : Type u_2
        C₁₂ : Type u_3
        C₃ : Type u_5
        C₄ : Type u_6
        inst✝²⁰ : CategoryTheory.Category.{u_13, u_1} C₁
        inst✝¹⁹ : CategoryTheory.Category.{u_14, u_2} C₂
        inst✝¹⁸ : CategoryTheory.Category.{u_15, u_5} C₃
        inst✝¹⁷ : CategoryTheory.Category.{u_16, u_6} C₄
        inst✝¹⁶ : CategoryTheory.Category.{u_17, u_3} C₁₂
        inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
        inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
        inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₃
        inst✝¹² : CategoryTheory.Preadditive C₁₂
        inst✝¹¹ : CategoryTheory.Preadditive C₄
        F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
        G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
        inst✝¹⁰ : F₁₂.PreservesZeroMorphisms
        inst✝⁹ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
        inst✝⁸ : G.Additive
        inst✝⁷ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
        ι₁ : Type u_7
        ι₂ : Type u_8
        ι₃ : Type u_9
        ι₁₂ : Type u_10
        ι₄ : Type u_12
        inst✝⁶ : DecidableEq ι₄
        c₁ : ComplexShape ι₁
        c₂ : ComplexShape ι₂
        c₃ : ComplexShape ι₃
        K₁ : HomologicalComplex C₁ c₁
        K₂ : HomologicalComplex C₂ c₂
        K₃ : HomologicalComplex C₃ c₃
        c₁₂ : ComplexShape ι₁₂
        c₄ : ComplexShape ι₄
        inst✝⁵ : TotalComplexShape c₁ c₂ c₁₂
        inst✝⁴ : TotalComplexShape c₁₂ c₃ c₄
        inst✝³ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
        inst✝² : DecidableEq ι₁₂
        inst✝¹ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
        j j' : ι₄
        inst✝ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
        i₁ : ι₁
        i₂ : ι₂
        i₃ : ι₃
        h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
        i₁₂ : ι₁₂ := c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }
        h₁ : c₁₂.Rel i₁₂ (c₁₂.next i₁₂)
        h₂ : Not (Eq (c₁₂.π c₃ c₄ { fst := c₁₂.next i₁₂, snd := i₃ }) j')
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((G.map (K₁.ιMapBifunctor K₂ F₁₂ c₁₂  …
      -/
    · rw [mapBifunctor.d₁_eq_zero' _ _ _ _ h₁ _ _ h₂, comp_zero]
      /-
        case neg
        C₁ : Type u_1
        C₂ : Type u_2
        C₁₂ : Type u_3
        C₃ : Type u_5
        C₄ : Type u_6
        inst✝²⁰ : CategoryTheory.Category.{u_13, u_1} C₁
        inst✝¹⁹ : CategoryTheory.Category.{u_14, u_2} C₂
        inst✝¹⁸ : CategoryTheory.Category.{u_15, u_5} C₃
        inst✝¹⁷ : CategoryTheory.Category.{u_16, u_6} C₄
        inst✝¹⁶ : CategoryTheory.Category.{u_17, u_3} C₁₂
        inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
        inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
        inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₃
        inst✝¹² : CategoryTheory.Preadditive C₁₂
        inst✝¹¹ : CategoryTheory.Preadditive C₄
        F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
        G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
        inst✝¹⁰ : F₁₂.PreservesZeroMorphisms
        inst✝⁹ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
        inst✝⁸ : G.Additive
        inst✝⁷ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
        ι₁ : Type u_7
        ι₂ : Type u_8
        ι₃ : Type u_9
        ι₁₂ : Type u_10
        ι₄ : Type u_12
        inst✝⁶ : DecidableEq ι₄
        c₁ : ComplexShape ι₁
        c₂ : ComplexShape ι₂
        c₃ : ComplexShape ι₃
        K₁ : HomologicalComplex C₁ c₁
        K₂ : HomologicalComplex C₂ c₂
        K₃ : HomologicalComplex C₃ c₃
        c₁₂ : ComplexShape ι₁₂
        c₄ : ComplexShape ι₄
        inst✝⁵ : TotalComplexShape c₁ c₂ c₁₂
        inst✝⁴ : TotalComplexShape c₁₂ c₃ c₄
        inst✝³ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
        inst✝² : DecidableEq ι₁₂
        inst✝¹ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
        j j' : ι₄
        inst✝ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
        i₁ : ι₁
        i₂ : ι₂
        i₃ : ι₃
        h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
        i₁₂ : ι₁₂ := c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }
        h₁ : c₁₂.Rel i₁₂ (c₁₂.next i₁₂)
        h₂ : Not (Eq (c₁₂.π c₃ c₄ { fst := c₁₂.next i₁₂, snd := i₃ }) j')
        ⊢ Eq 0 (HAdd.hAdd (HomologicalComplex.mapBifunctor₁₂.d₁ F₁₂ G K₁ K₂ K₃ c₁₂ c₄  …
      -/
      trans 0 + 0
        /-
          C₁ : Type u_1
          C₂ : Type u_2
          C₁₂ : Type u_3
          C₃ : Type u_5
          C₄ : Type u_6
          inst✝²⁰ : CategoryTheory.Category.{u_13, u_1} C₁
          inst✝¹⁹ : CategoryTheory.Category.{u_14, u_2} C₂
          inst✝¹⁸ : CategoryTheory.Category.{u_15, u_5} C₃
          inst✝¹⁷ : CategoryTheory.Category.{u_16, u_6} C₄
          inst✝¹⁶ : CategoryTheory.Category.{u_17, u_3} C₁₂
          inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
          inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
          inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₃
          inst✝¹² : CategoryTheory.Preadditive C₁₂
          inst✝¹¹ : CategoryTheory.Preadditive C₄
          F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
          G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
          inst✝¹⁰ : F₁₂.PreservesZeroMorphisms
          inst✝⁹ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
          inst✝⁸ : G.Additive
          inst✝⁷ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
          ι₁ : Type u_7
          ι₂ : Type u_8
          ι₃ : Type u_9
          ι₁₂ : Type u_10
          ι₄ : Type u_12
          inst✝⁶ : DecidableEq ι₄
          c₁ : ComplexShape ι₁
          c₂ : ComplexShape ι₂
          c₃ : ComplexShape ι₃
          K₁ : HomologicalComplex C₁ c₁
          K₂ : HomologicalComplex C₂ c₂
          K₃ : HomologicalComplex C₃ c₃
          c₁₂ : ComplexShape ι₁₂
          c₄ : ComplexShape ι₄
          inst✝⁵ : TotalComplexShape c₁ c₂ c₁₂
          inst✝⁴ : TotalComplexShape c₁₂ c₃ c₄
          inst✝³ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
          inst✝² : DecidableEq ι₁₂
          inst✝¹ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
          j j' : ι₄
          inst✝ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
          i₁ : ι₁
          i₂ : ι₂
          i₃ : ι₃
          h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
          i₁₂ : ι₁₂ := c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }
          h₁ : c₁₂.Rel i₁₂ (c₁₂.next i₁₂)
          h₂ : Not (Eq (c₁₂.π c₃ c₄ { fst := c₁₂.next i₁₂, snd := i₃ }) j')
          ⊢ Eq 0 (HAdd.hAdd 0 0)
        -/
      · simp
        /-
          🎉 no goals
        -/
        /-
          C₁ : Type u_1
          C₂ : Type u_2
          C₁₂ : Type u_3
          C₃ : Type u_5
          C₄ : Type u_6
          inst✝²⁰ : CategoryTheory.Category.{u_13, u_1} C₁
          inst✝¹⁹ : CategoryTheory.Category.{u_14, u_2} C₂
          inst✝¹⁸ : CategoryTheory.Category.{u_15, u_5} C₃
          inst✝¹⁷ : CategoryTheory.Category.{u_16, u_6} C₄
          inst✝¹⁶ : CategoryTheory.Category.{u_17, u_3} C₁₂
          inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
          inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
          inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₃
          inst✝¹² : CategoryTheory.Preadditive C₁₂
          inst✝¹¹ : CategoryTheory.Preadditive C₄
          F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
          G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
          inst✝¹⁰ : F₁₂.PreservesZeroMorphisms
          inst✝⁹ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
          inst✝⁸ : G.Additive
          inst✝⁷ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
          ι₁ : Type u_7
          ι₂ : Type u_8
          ι₃ : Type u_9
          ι₁₂ : Type u_10
          ι₄ : Type u_12
          inst✝⁶ : DecidableEq ι₄
          c₁ : ComplexShape ι₁
          c₂ : ComplexShape ι₂
          c₃ : ComplexShape ι₃
          K₁ : HomologicalComplex C₁ c₁
          K₂ : HomologicalComplex C₂ c₂
          K₃ : HomologicalComplex C₃ c₃
          c₁₂ : ComplexShape ι₁₂
          c₄ : ComplexShape ι₄
          inst✝⁵ : TotalComplexShape c₁ c₂ c₁₂
          inst✝⁴ : TotalComplexShape c₁₂ c₃ c₄
          inst✝³ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
          inst✝² : DecidableEq ι₁₂
          inst✝¹ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
          j j' : ι₄
          inst✝ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
          i₁ : ι₁
          i₂ : ι₂
          i₃ : ι₃
          h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
          i₁₂ : ι₁₂ := c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }
          h₁ : c₁₂.Rel i₁₂ (c₁₂.next i₁₂)
          h₂ : Not (Eq (c₁₂.π c₃ c₄ { fst := c₁₂.next i₁₂, snd := i₃ }) j')
          ⊢ Eq (HAdd.hAdd 0 0) (HAdd.hAdd (HomologicalComplex.mapBifunctor₁₂.d₁ F₁₂ G K₁ …
        -/
      · congr 1
          /-
            case e_a
            C₁ : Type u_1
            C₂ : Type u_2
            C₁₂ : Type u_3
            C₃ : Type u_5
            C₄ : Type u_6
            inst✝²⁰ : CategoryTheory.Category.{u_13, u_1} C₁
            inst✝¹⁹ : CategoryTheory.Category.{u_14, u_2} C₂
            inst✝¹⁸ : CategoryTheory.Category.{u_15, u_5} C₃
            inst✝¹⁷ : CategoryTheory.Category.{u_16, u_6} C₄
            inst✝¹⁶ : CategoryTheory.Category.{u_17, u_3} C₁₂
            inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
            inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
            inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₃
            inst✝¹² : CategoryTheory.Preadditive C₁₂
            inst✝¹¹ : CategoryTheory.Preadditive C₄
            F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
            G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
            inst✝¹⁰ : F₁₂.PreservesZeroMorphisms
            inst✝⁹ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
            inst✝⁸ : G.Additive
            inst✝⁷ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
            ι₁ : Type u_7
            ι₂ : Type u_8
            ι₃ : Type u_9
            ι₁₂ : Type u_10
            ι₄ : Type u_12
            inst✝⁶ : DecidableEq ι₄
            c₁ : ComplexShape ι₁
            c₂ : ComplexShape ι₂
            c₃ : ComplexShape ι₃
            K₁ : HomologicalComplex C₁ c₁
            K₂ : HomologicalComplex C₂ c₂
            K₃ : HomologicalComplex C₃ c₃
            c₁₂ : ComplexShape ι₁₂
            c₄ : ComplexShape ι₄
            inst✝⁵ : TotalComplexShape c₁ c₂ c₁₂
            inst✝⁴ : TotalComplexShape c₁₂ c₃ c₄
            inst✝³ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
            inst✝² : DecidableEq ι₁₂
            inst✝¹ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
            j j' : ι₄
            inst✝ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
            i₁ : ι₁
            i₂ : ι₂
            i₃ : ι₃
            h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
            i₁₂ : ι₁₂ := c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }
            h₁ : c₁₂.Rel i₁₂ (c₁₂.next i₁₂)
            h₂ : Not (Eq (c₁₂.π c₃ c₄ { fst := c₁₂.next i₁₂, snd := i₃ }) j')
            ⊢ Eq 0 (HomologicalComplex.mapBifunctor₁₂.d₁ F₁₂ G K₁ K₂ K₃ c₁₂ c₄ i₁ i₂ i₃ j')
          -/
        · by_cases h₃ : c₁.Rel i₁ (c₁.next i₁)
            /-
              case pos
              C₁ : Type u_1
              C₂ : Type u_2
              C₁₂ : Type u_3
              C₃ : Type u_5
              C₄ : Type u_6
              inst✝²⁰ : CategoryTheory.Category.{u_13, u_1} C₁
              inst✝¹⁹ : CategoryTheory.Category.{u_14, u_2} C₂
              inst✝¹⁸ : CategoryTheory.Category.{u_15, u_5} C₃
              inst✝¹⁷ : CategoryTheory.Category.{u_16, u_6} C₄
              inst✝¹⁶ : CategoryTheory.Category.{u_17, u_3} C₁₂
              inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
              inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
              inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₃
              inst✝¹² : CategoryTheory.Preadditive C₁₂
              inst✝¹¹ : CategoryTheory.Preadditive C₄
              F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
              G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
              inst✝¹⁰ : F₁₂.PreservesZeroMorphisms
              inst✝⁹ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
              inst✝⁸ : G.Additive
              inst✝⁷ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
              ι₁ : Type u_7
              ι₂ : Type u_8
              ι₃ : Type u_9
              ι₁₂ : Type u_10
              ι₄ : Type u_12
              inst✝⁶ : DecidableEq ι₄
              c₁ : ComplexShape ι₁
              c₂ : ComplexShape ι₂
              c₃ : ComplexShape ι₃
              K₁ : HomologicalComplex C₁ c₁
              K₂ : HomologicalComplex C₂ c₂
              K₃ : HomologicalComplex C₃ c₃
              c₁₂ : ComplexShape ι₁₂
              c₄ : ComplexShape ι₄
              inst✝⁵ : TotalComplexShape c₁ c₂ c₁₂
              inst✝⁴ : TotalComplexShape c₁₂ c₃ c₄
              inst✝³ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
              inst✝² : DecidableEq ι₁₂
              inst✝¹ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
              j j' : ι₄
              inst✝ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
              i₁ : ι₁
              i₂ : ι₂
              i₃ : ι₃
              h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
              i₁₂ : ι₁₂ := c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }
              h₁ : c₁₂.Rel i₁₂ (c₁₂.next i₁₂)
              h₂ : Not (Eq (c₁₂.π c₃ c₄ { fst := c₁₂.next i₁₂, snd := i₃ }) j')
              h₃ : c₁.Rel i₁ (c₁.next i₁)
              ⊢ Eq 0 (HomologicalComplex.mapBifunctor₁₂.d₁ F₁₂ G K₁ K₂ K₃ c₁₂ c₄ i₁ i₂ i₃ j')
            -/
          · rw [d₁_eq _ _ _ _ _ _ _ h₃, ιOrZero_eq_zero, comp_zero, smul_zero]
            /-
              case pos.h
              C₁ : Type u_1
              C₂ : Type u_2
              C₁₂ : Type u_3
              C₃ : Type u_5
              C₄ : Type u_6
              inst✝²⁰ : CategoryTheory.Category.{u_13, u_1} C₁
              inst✝¹⁹ : CategoryTheory.Category.{u_14, u_2} C₂
              inst✝¹⁸ : CategoryTheory.Category.{u_15, u_5} C₃
              inst✝¹⁷ : CategoryTheory.Category.{u_16, u_6} C₄
              inst✝¹⁶ : CategoryTheory.Category.{u_17, u_3} C₁₂
              inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
              inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
              inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₃
              inst✝¹² : CategoryTheory.Preadditive C₁₂
              inst✝¹¹ : CategoryTheory.Preadditive C₄
              F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
              G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
              inst✝¹⁰ : F₁₂.PreservesZeroMorphisms
              inst✝⁹ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
              inst✝⁸ : G.Additive
              inst✝⁷ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
              ι₁ : Type u_7
              ι₂ : Type u_8
              ι₃ : Type u_9
              ι₁₂ : Type u_10
              ι₄ : Type u_12
              inst✝⁶ : DecidableEq ι₄
              c₁ : ComplexShape ι₁
              c₂ : ComplexShape ι₂
              c₃ : ComplexShape ι₃
              K₁ : HomologicalComplex C₁ c₁
              K₂ : HomologicalComplex C₂ c₂
              K₃ : HomologicalComplex C₃ c₃
              c₁₂ : ComplexShape ι₁₂
              c₄ : ComplexShape ι₄
              inst✝⁵ : TotalComplexShape c₁ c₂ c₁₂
              inst✝⁴ : TotalComplexShape c₁₂ c₃ c₄
              inst✝³ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
              inst✝² : DecidableEq ι₁₂
              inst✝¹ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
              j j' : ι₄
              inst✝ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
              i₁ : ι₁
              i₂ : ι₂
              i₃ : ι₃
              h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
              i₁₂ : ι₁₂ := c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }
              h₁ : c₁₂.Rel i₁₂ (c₁₂.next i₁₂)
              h₂ : Not (Eq (c₁₂.π c₃ c₄ { fst := c₁₂.next i₁₂, snd := i₃ }) j')
              h₃ : c₁.Rel i₁ (c₁.next i₁)
              ⊢ Ne (c₁.r c₂ c₃ c₁₂ c₄ { fst := c₁.next i₁, snd := { fst := i₂, snd := i₃ } } …
            -/
            dsimp [ComplexShape.r]
            /-
              case pos.h
              C₁ : Type u_1
              C₂ : Type u_2
              C₁₂ : Type u_3
              C₃ : Type u_5
              C₄ : Type u_6
              inst✝²⁰ : CategoryTheory.Category.{u_13, u_1} C₁
              inst✝¹⁹ : CategoryTheory.Category.{u_14, u_2} C₂
              inst✝¹⁸ : CategoryTheory.Category.{u_15, u_5} C₃
              inst✝¹⁷ : CategoryTheory.Category.{u_16, u_6} C₄
              inst✝¹⁶ : CategoryTheory.Category.{u_17, u_3} C₁₂
              inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
              inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
              inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₃
              inst✝¹² : CategoryTheory.Preadditive C₁₂
              inst✝¹¹ : CategoryTheory.Preadditive C₄
              F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
              G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
              inst✝¹⁰ : F₁₂.PreservesZeroMorphisms
              inst✝⁹ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
              inst✝⁸ : G.Additive
              inst✝⁷ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
              ι₁ : Type u_7
              ι₂ : Type u_8
              ι₃ : Type u_9
              ι₁₂ : Type u_10
              ι₄ : Type u_12
              inst✝⁶ : DecidableEq ι₄
              c₁ : ComplexShape ι₁
              c₂ : ComplexShape ι₂
              c₃ : ComplexShape ι₃
              K₁ : HomologicalComplex C₁ c₁
              K₂ : HomologicalComplex C₂ c₂
              K₃ : HomologicalComplex C₃ c₃
              c₁₂ : ComplexShape ι₁₂
              c₄ : ComplexShape ι₄
              inst✝⁵ : TotalComplexShape c₁ c₂ c₁₂
              inst✝⁴ : TotalComplexShape c₁₂ c₃ c₄
              inst✝³ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
              inst✝² : DecidableEq ι₁₂
              inst✝¹ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
              j j' : ι₄
              inst✝ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
              i₁ : ι₁
              i₂ : ι₂
              i₃ : ι₃
              h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
              i₁₂ : ι₁₂ := c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }
              h₁ : c₁₂.Rel i₁₂ (c₁₂.next i₁₂)
              h₂ : Not (Eq (c₁₂.π c₃ c₄ { fst := c₁₂.next i₁₂, snd := i₃ }) j')
              h₃ : c₁.Rel i₁ (c₁.next i₁)
              ⊢ Not (Eq (c₁₂.π c₃ c₄ { fst := c₁.π c₂ c₁₂ { fst := c₁.next i₁, snd := i₂ },  …
            -/
            intro h₄
            /-
              case pos.h
              C₁ : Type u_1
              C₂ : Type u_2
              C₁₂ : Type u_3
              C₃ : Type u_5
              C₄ : Type u_6
              inst✝²⁰ : CategoryTheory.Category.{u_13, u_1} C₁
              inst✝¹⁹ : CategoryTheory.Category.{u_14, u_2} C₂
              inst✝¹⁸ : CategoryTheory.Category.{u_15, u_5} C₃
              inst✝¹⁷ : CategoryTheory.Category.{u_16, u_6} C₄
              inst✝¹⁶ : CategoryTheory.Category.{u_17, u_3} C₁₂
              inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
              inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
              inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₃
              inst✝¹² : CategoryTheory.Preadditive C₁₂
              inst✝¹¹ : CategoryTheory.Preadditive C₄
              F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
              G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
              inst✝¹⁰ : F₁₂.PreservesZeroMorphisms
              inst✝⁹ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
              inst✝⁸ : G.Additive
              inst✝⁷ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
              ι₁ : Type u_7
              ι₂ : Type u_8
              ι₃ : Type u_9
              ι₁₂ : Type u_10
              ι₄ : Type u_12
              inst✝⁶ : DecidableEq ι₄
              c₁ : ComplexShape ι₁
              c₂ : ComplexShape ι₂
              c₃ : ComplexShape ι₃
              K₁ : HomologicalComplex C₁ c₁
              K₂ : HomologicalComplex C₂ c₂
              K₃ : HomologicalComplex C₃ c₃
              c₁₂ : ComplexShape ι₁₂
              c₄ : ComplexShape ι₄
              inst✝⁵ : TotalComplexShape c₁ c₂ c₁₂
              inst✝⁴ : TotalComplexShape c₁₂ c₃ c₄
              inst✝³ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
              inst✝² : DecidableEq ι₁₂
              inst✝¹ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
              j j' : ι₄
              inst✝ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
              i₁ : ι₁
              i₂ : ι₂
              i₃ : ι₃
              h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
              i₁₂ : ι₁₂ := c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }
              h₁ : c₁₂.Rel i₁₂ (c₁₂.next i₁₂)
              h₂ : Not (Eq (c₁₂.π c₃ c₄ { fst := c₁₂.next i₁₂, snd := i₃ }) j')
              h₃ : c₁.Rel i₁ (c₁.next i₁)
              h₄ : Eq (c₁₂.π c₃ c₄ { fst := c₁.π c₂ c₁₂ { fst := c₁.next i₁, snd := i₂ }, sn …
              ⊢ False
            -/
            apply h₂
            /-
              case pos.h
              C₁ : Type u_1
              C₂ : Type u_2
              C₁₂ : Type u_3
              C₃ : Type u_5
              C₄ : Type u_6
              inst✝²⁰ : CategoryTheory.Category.{u_13, u_1} C₁
              inst✝¹⁹ : CategoryTheory.Category.{u_14, u_2} C₂
              inst✝¹⁸ : CategoryTheory.Category.{u_15, u_5} C₃
              inst✝¹⁷ : CategoryTheory.Category.{u_16, u_6} C₄
              inst✝¹⁶ : CategoryTheory.Category.{u_17, u_3} C₁₂
              inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
              inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
              inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₃
              inst✝¹² : CategoryTheory.Preadditive C₁₂
              inst✝¹¹ : CategoryTheory.Preadditive C₄
              F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
              G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
              inst✝¹⁰ : F₁₂.PreservesZeroMorphisms
              inst✝⁹ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
              inst✝⁸ : G.Additive
              inst✝⁷ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
              ι₁ : Type u_7
              ι₂ : Type u_8
              ι₃ : Type u_9
              ι₁₂ : Type u_10
              ι₄ : Type u_12
              inst✝⁶ : DecidableEq ι₄
              c₁ : ComplexShape ι₁
              c₂ : ComplexShape ι₂
              c₃ : ComplexShape ι₃
              K₁ : HomologicalComplex C₁ c₁
              K₂ : HomologicalComplex C₂ c₂
              K₃ : HomologicalComplex C₃ c₃
              c₁₂ : ComplexShape ι₁₂
              c₄ : ComplexShape ι₄
              inst✝⁵ : TotalComplexShape c₁ c₂ c₁₂
              inst✝⁴ : TotalComplexShape c₁₂ c₃ c₄
              inst✝³ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
              inst✝² : DecidableEq ι₁₂
              inst✝¹ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
              j j' : ι₄
              inst✝ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
              i₁ : ι₁
              i₂ : ι₂
              i₃ : ι₃
              h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
              i₁₂ : ι₁₂ := c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }
              h₁ : c₁₂.Rel i₁₂ (c₁₂.next i₁₂)
              h₂ : Not (Eq (c₁₂.π c₃ c₄ { fst := c₁₂.next i₁₂, snd := i₃ }) j')
              h₃ : c₁.Rel i₁ (c₁.next i₁)
              h₄ : Eq (c₁₂.π c₃ c₄ { fst := c₁.π c₂ c₁₂ { fst := c₁.next i₁, snd := i₂ }, sn …
              ⊢ Eq (c₁₂.π c₃ c₄ { fst := c₁₂.next i₁₂, snd := i₃ }) j'
            -/
            rw [← h₄, ComplexShape.next_π₁ c₂ c₁₂ h₃ i₂]
            /-
              🎉 no goals
            -/
            /-
              case neg
              C₁ : Type u_1
              C₂ : Type u_2
              C₁₂ : Type u_3
              C₃ : Type u_5
              C₄ : Type u_6
              inst✝²⁰ : CategoryTheory.Category.{u_13, u_1} C₁
              inst✝¹⁹ : CategoryTheory.Category.{u_14, u_2} C₂
              inst✝¹⁸ : CategoryTheory.Category.{u_15, u_5} C₃
              inst✝¹⁷ : CategoryTheory.Category.{u_16, u_6} C₄
              inst✝¹⁶ : CategoryTheory.Category.{u_17, u_3} C₁₂
              inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
              inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
              inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₃
              inst✝¹² : CategoryTheory.Preadditive C₁₂
              inst✝¹¹ : CategoryTheory.Preadditive C₄
              F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
              G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
              inst✝¹⁰ : F₁₂.PreservesZeroMorphisms
              inst✝⁹ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
              inst✝⁸ : G.Additive
              inst✝⁷ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
              ι₁ : Type u_7
              ι₂ : Type u_8
              ι₃ : Type u_9
              ι₁₂ : Type u_10
              ι₄ : Type u_12
              inst✝⁶ : DecidableEq ι₄
              c₁ : ComplexShape ι₁
              c₂ : ComplexShape ι₂
              c₃ : ComplexShape ι₃
              K₁ : HomologicalComplex C₁ c₁
              K₂ : HomologicalComplex C₂ c₂
              K₃ : HomologicalComplex C₃ c₃
              c₁₂ : ComplexShape ι₁₂
              c₄ : ComplexShape ι₄
              inst✝⁵ : TotalComplexShape c₁ c₂ c₁₂
              inst✝⁴ : TotalComplexShape c₁₂ c₃ c₄
              inst✝³ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
              inst✝² : DecidableEq ι₁₂
              inst✝¹ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
              j j' : ι₄
              inst✝ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
              i₁ : ι₁
              i₂ : ι₂
              i₃ : ι₃
              h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
              i₁₂ : ι₁₂ := c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }
              h₁ : c₁₂.Rel i₁₂ (c₁₂.next i₁₂)
              h₂ : Not (Eq (c₁₂.π c₃ c₄ { fst := c₁₂.next i₁₂, snd := i₃ }) j')
              h₃ : Not (c₁.Rel i₁ (c₁.next i₁))
              ⊢ Eq 0 (HomologicalComplex.mapBifunctor₁₂.d₁ F₁₂ G K₁ K₂ K₃ c₁₂ c₄ i₁ i₂ i₃ j')
            -/
          · rw [d₁_eq_zero _ _ _ _ _ _ _ _ _ _ _ h₃]
            /-
              🎉 no goals
            -/
          /-
            case e_a
            C₁ : Type u_1
            C₂ : Type u_2
            C₁₂ : Type u_3
            C₃ : Type u_5
            C₄ : Type u_6
            inst✝²⁰ : CategoryTheory.Category.{u_13, u_1} C₁
            inst✝¹⁹ : CategoryTheory.Category.{u_14, u_2} C₂
            inst✝¹⁸ : CategoryTheory.Category.{u_15, u_5} C₃
            inst✝¹⁷ : CategoryTheory.Category.{u_16, u_6} C₄
            inst✝¹⁶ : CategoryTheory.Category.{u_17, u_3} C₁₂
            inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
            inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
            inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₃
            inst✝¹² : CategoryTheory.Preadditive C₁₂
            inst✝¹¹ : CategoryTheory.Preadditive C₄
            F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
            G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
            inst✝¹⁰ : F₁₂.PreservesZeroMorphisms
            inst✝⁹ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
            inst✝⁸ : G.Additive
            inst✝⁷ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
            ι₁ : Type u_7
            ι₂ : Type u_8
            ι₃ : Type u_9
            ι₁₂ : Type u_10
            ι₄ : Type u_12
            inst✝⁶ : DecidableEq ι₄
            c₁ : ComplexShape ι₁
            c₂ : ComplexShape ι₂
            c₃ : ComplexShape ι₃
            K₁ : HomologicalComplex C₁ c₁
            K₂ : HomologicalComplex C₂ c₂
            K₃ : HomologicalComplex C₃ c₃
            c₁₂ : ComplexShape ι₁₂
            c₄ : ComplexShape ι₄
            inst✝⁵ : TotalComplexShape c₁ c₂ c₁₂
            inst✝⁴ : TotalComplexShape c₁₂ c₃ c₄
            inst✝³ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
            inst✝² : DecidableEq ι₁₂
            inst✝¹ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
            j j' : ι₄
            inst✝ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
            i₁ : ι₁
            i₂ : ι₂
            i₃ : ι₃
            h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
            i₁₂ : ι₁₂ := c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }
            h₁ : c₁₂.Rel i₁₂ (c₁₂.next i₁₂)
            h₂ : Not (Eq (c₁₂.π c₃ c₄ { fst := c₁₂.next i₁₂, snd := i₃ }) j')
            ⊢ Eq 0 (HomologicalComplex.mapBifunctor₁₂.d₂ F₁₂ G K₁ K₂ K₃ c₁₂ c₄ i₁ i₂ i₃ j')
          -/
        · by_cases h₃ : c₂.Rel i₂ (c₂.next i₂)
            /-
              case pos
              C₁ : Type u_1
              C₂ : Type u_2
              C₁₂ : Type u_3
              C₃ : Type u_5
              C₄ : Type u_6
              inst✝²⁰ : CategoryTheory.Category.{u_13, u_1} C₁
              inst✝¹⁹ : CategoryTheory.Category.{u_14, u_2} C₂
              inst✝¹⁸ : CategoryTheory.Category.{u_15, u_5} C₃
              inst✝¹⁷ : CategoryTheory.Category.{u_16, u_6} C₄
              inst✝¹⁶ : CategoryTheory.Category.{u_17, u_3} C₁₂
              inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
              inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
              inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₃
              inst✝¹² : CategoryTheory.Preadditive C₁₂
              inst✝¹¹ : CategoryTheory.Preadditive C₄
              F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
              G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
              inst✝¹⁰ : F₁₂.PreservesZeroMorphisms
              inst✝⁹ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
              inst✝⁸ : G.Additive
              inst✝⁷ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
              ι₁ : Type u_7
              ι₂ : Type u_8
              ι₃ : Type u_9
              ι₁₂ : Type u_10
              ι₄ : Type u_12
              inst✝⁶ : DecidableEq ι₄
              c₁ : ComplexShape ι₁
              c₂ : ComplexShape ι₂
              c₃ : ComplexShape ι₃
              K₁ : HomologicalComplex C₁ c₁
              K₂ : HomologicalComplex C₂ c₂
              K₃ : HomologicalComplex C₃ c₃
              c₁₂ : ComplexShape ι₁₂
              c₄ : ComplexShape ι₄
              inst✝⁵ : TotalComplexShape c₁ c₂ c₁₂
              inst✝⁴ : TotalComplexShape c₁₂ c₃ c₄
              inst✝³ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
              inst✝² : DecidableEq ι₁₂
              inst✝¹ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
              j j' : ι₄
              inst✝ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
              i₁ : ι₁
              i₂ : ι₂
              i₃ : ι₃
              h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
              i₁₂ : ι₁₂ := c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }
              h₁ : c₁₂.Rel i₁₂ (c₁₂.next i₁₂)
              h₂ : Not (Eq (c₁₂.π c₃ c₄ { fst := c₁₂.next i₁₂, snd := i₃ }) j')
              h₃ : c₂.Rel i₂ (c₂.next i₂)
              ⊢ Eq 0 (HomologicalComplex.mapBifunctor₁₂.d₂ F₁₂ G K₁ K₂ K₃ c₁₂ c₄ i₁ i₂ i₃ j')
            -/
          · rw [d₂_eq _ _ _ _ _ _ _ _ h₃, ιOrZero_eq_zero, comp_zero, smul_zero]
            /-
              case pos.h
              C₁ : Type u_1
              C₂ : Type u_2
              C₁₂ : Type u_3
              C₃ : Type u_5
              C₄ : Type u_6
              inst✝²⁰ : CategoryTheory.Category.{u_13, u_1} C₁
              inst✝¹⁹ : CategoryTheory.Category.{u_14, u_2} C₂
              inst✝¹⁸ : CategoryTheory.Category.{u_15, u_5} C₃
              inst✝¹⁷ : CategoryTheory.Category.{u_16, u_6} C₄
              inst✝¹⁶ : CategoryTheory.Category.{u_17, u_3} C₁₂
              inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
              inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
              inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₃
              inst✝¹² : CategoryTheory.Preadditive C₁₂
              inst✝¹¹ : CategoryTheory.Preadditive C₄
              F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
              G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
              inst✝¹⁰ : F₁₂.PreservesZeroMorphisms
              inst✝⁹ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
              inst✝⁸ : G.Additive
              inst✝⁷ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
              ι₁ : Type u_7
              ι₂ : Type u_8
              ι₃ : Type u_9
              ι₁₂ : Type u_10
              ι₄ : Type u_12
              inst✝⁶ : DecidableEq ι₄
              c₁ : ComplexShape ι₁
              c₂ : ComplexShape ι₂
              c₃ : ComplexShape ι₃
              K₁ : HomologicalComplex C₁ c₁
              K₂ : HomologicalComplex C₂ c₂
              K₃ : HomologicalComplex C₃ c₃
              c₁₂ : ComplexShape ι₁₂
              c₄ : ComplexShape ι₄
              inst✝⁵ : TotalComplexShape c₁ c₂ c₁₂
              inst✝⁴ : TotalComplexShape c₁₂ c₃ c₄
              inst✝³ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
              inst✝² : DecidableEq ι₁₂
              inst✝¹ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
              j j' : ι₄
              inst✝ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
              i₁ : ι₁
              i₂ : ι₂
              i₃ : ι₃
              h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
              i₁₂ : ι₁₂ := c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }
              h₁ : c₁₂.Rel i₁₂ (c₁₂.next i₁₂)
              h₂ : Not (Eq (c₁₂.π c₃ c₄ { fst := c₁₂.next i₁₂, snd := i₃ }) j')
              h₃ : c₂.Rel i₂ (c₂.next i₂)
              ⊢ Ne (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := c₂.next i₂, snd := i₃ } } …
            -/
            dsimp [ComplexShape.r]
            /-
              case pos.h
              C₁ : Type u_1
              C₂ : Type u_2
              C₁₂ : Type u_3
              C₃ : Type u_5
              C₄ : Type u_6
              inst✝²⁰ : CategoryTheory.Category.{u_13, u_1} C₁
              inst✝¹⁹ : CategoryTheory.Category.{u_14, u_2} C₂
              inst✝¹⁸ : CategoryTheory.Category.{u_15, u_5} C₃
              inst✝¹⁷ : CategoryTheory.Category.{u_16, u_6} C₄
              inst✝¹⁶ : CategoryTheory.Category.{u_17, u_3} C₁₂
              inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
              inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
              inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₃
              inst✝¹² : CategoryTheory.Preadditive C₁₂
              inst✝¹¹ : CategoryTheory.Preadditive C₄
              F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
              G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
              inst✝¹⁰ : F₁₂.PreservesZeroMorphisms
              inst✝⁹ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
              inst✝⁸ : G.Additive
              inst✝⁷ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
              ι₁ : Type u_7
              ι₂ : Type u_8
              ι₃ : Type u_9
              ι₁₂ : Type u_10
              ι₄ : Type u_12
              inst✝⁶ : DecidableEq ι₄
              c₁ : ComplexShape ι₁
              c₂ : ComplexShape ι₂
              c₃ : ComplexShape ι₃
              K₁ : HomologicalComplex C₁ c₁
              K₂ : HomologicalComplex C₂ c₂
              K₃ : HomologicalComplex C₃ c₃
              c₁₂ : ComplexShape ι₁₂
              c₄ : ComplexShape ι₄
              inst✝⁵ : TotalComplexShape c₁ c₂ c₁₂
              inst✝⁴ : TotalComplexShape c₁₂ c₃ c₄
              inst✝³ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
              inst✝² : DecidableEq ι₁₂
              inst✝¹ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
              j j' : ι₄
              inst✝ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
              i₁ : ι₁
              i₂ : ι₂
              i₃ : ι₃
              h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
              i₁₂ : ι₁₂ := c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }
              h₁ : c₁₂.Rel i₁₂ (c₁₂.next i₁₂)
              h₂ : Not (Eq (c₁₂.π c₃ c₄ { fst := c₁₂.next i₁₂, snd := i₃ }) j')
              h₃ : c₂.Rel i₂ (c₂.next i₂)
              ⊢ Not (Eq (c₁₂.π c₃ c₄ { fst := c₁.π c₂ c₁₂ { fst := i₁, snd := c₂.next i₂ },  …
            -/
            intro h₄
            /-
              case pos.h
              C₁ : Type u_1
              C₂ : Type u_2
              C₁₂ : Type u_3
              C₃ : Type u_5
              C₄ : Type u_6
              inst✝²⁰ : CategoryTheory.Category.{u_13, u_1} C₁
              inst✝¹⁹ : CategoryTheory.Category.{u_14, u_2} C₂
              inst✝¹⁸ : CategoryTheory.Category.{u_15, u_5} C₃
              inst✝¹⁷ : CategoryTheory.Category.{u_16, u_6} C₄
              inst✝¹⁶ : CategoryTheory.Category.{u_17, u_3} C₁₂
              inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
              inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
              inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₃
              inst✝¹² : CategoryTheory.Preadditive C₁₂
              inst✝¹¹ : CategoryTheory.Preadditive C₄
              F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
              G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
              inst✝¹⁰ : F₁₂.PreservesZeroMorphisms
              inst✝⁹ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
              inst✝⁸ : G.Additive
              inst✝⁷ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
              ι₁ : Type u_7
              ι₂ : Type u_8
              ι₃ : Type u_9
              ι₁₂ : Type u_10
              ι₄ : Type u_12
              inst✝⁶ : DecidableEq ι₄
              c₁ : ComplexShape ι₁
              c₂ : ComplexShape ι₂
              c₃ : ComplexShape ι₃
              K₁ : HomologicalComplex C₁ c₁
              K₂ : HomologicalComplex C₂ c₂
              K₃ : HomologicalComplex C₃ c₃
              c₁₂ : ComplexShape ι₁₂
              c₄ : ComplexShape ι₄
              inst✝⁵ : TotalComplexShape c₁ c₂ c₁₂
              inst✝⁴ : TotalComplexShape c₁₂ c₃ c₄
              inst✝³ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
              inst✝² : DecidableEq ι₁₂
              inst✝¹ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
              j j' : ι₄
              inst✝ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
              i₁ : ι₁
              i₂ : ι₂
              i₃ : ι₃
              h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
              i₁₂ : ι₁₂ := c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }
              h₁ : c₁₂.Rel i₁₂ (c₁₂.next i₁₂)
              h₂ : Not (Eq (c₁₂.π c₃ c₄ { fst := c₁₂.next i₁₂, snd := i₃ }) j')
              h₃ : c₂.Rel i₂ (c₂.next i₂)
              h₄ : Eq (c₁₂.π c₃ c₄ { fst := c₁.π c₂ c₁₂ { fst := i₁, snd := c₂.next i₂ }, sn …
              ⊢ False
            -/
            apply h₂
            /-
              case pos.h
              C₁ : Type u_1
              C₂ : Type u_2
              C₁₂ : Type u_3
              C₃ : Type u_5
              C₄ : Type u_6
              inst✝²⁰ : CategoryTheory.Category.{u_13, u_1} C₁
              inst✝¹⁹ : CategoryTheory.Category.{u_14, u_2} C₂
              inst✝¹⁸ : CategoryTheory.Category.{u_15, u_5} C₃
              inst✝¹⁷ : CategoryTheory.Category.{u_16, u_6} C₄
              inst✝¹⁶ : CategoryTheory.Category.{u_17, u_3} C₁₂
              inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
              inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
              inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₃
              inst✝¹² : CategoryTheory.Preadditive C₁₂
              inst✝¹¹ : CategoryTheory.Preadditive C₄
              F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
              G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
              inst✝¹⁰ : F₁₂.PreservesZeroMorphisms
              inst✝⁹ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
              inst✝⁸ : G.Additive
              inst✝⁷ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
              ι₁ : Type u_7
              ι₂ : Type u_8
              ι₃ : Type u_9
              ι₁₂ : Type u_10
              ι₄ : Type u_12
              inst✝⁶ : DecidableEq ι₄
              c₁ : ComplexShape ι₁
              c₂ : ComplexShape ι₂
              c₃ : ComplexShape ι₃
              K₁ : HomologicalComplex C₁ c₁
              K₂ : HomologicalComplex C₂ c₂
              K₃ : HomologicalComplex C₃ c₃
              c₁₂ : ComplexShape ι₁₂
              c₄ : ComplexShape ι₄
              inst✝⁵ : TotalComplexShape c₁ c₂ c₁₂
              inst✝⁴ : TotalComplexShape c₁₂ c₃ c₄
              inst✝³ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
              inst✝² : DecidableEq ι₁₂
              inst✝¹ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
              j j' : ι₄
              inst✝ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
              i₁ : ι₁
              i₂ : ι₂
              i₃ : ι₃
              h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
              i₁₂ : ι₁₂ := c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }
              h₁ : c₁₂.Rel i₁₂ (c₁₂.next i₁₂)
              h₂ : Not (Eq (c₁₂.π c₃ c₄ { fst := c₁₂.next i₁₂, snd := i₃ }) j')
              h₃ : c₂.Rel i₂ (c₂.next i₂)
              h₄ : Eq (c₁₂.π c₃ c₄ { fst := c₁.π c₂ c₁₂ { fst := i₁, snd := c₂.next i₂ }, sn …
              ⊢ Eq (c₁₂.π c₃ c₄ { fst := c₁₂.next i₁₂, snd := i₃ }) j'
            -/
            rw [← h₄, ComplexShape.next_π₂ c₁ c₁₂ i₁ h₃]
            /-
              🎉 no goals
            -/
            /-
              case neg
              C₁ : Type u_1
              C₂ : Type u_2
              C₁₂ : Type u_3
              C₃ : Type u_5
              C₄ : Type u_6
              inst✝²⁰ : CategoryTheory.Category.{u_13, u_1} C₁
              inst✝¹⁹ : CategoryTheory.Category.{u_14, u_2} C₂
              inst✝¹⁸ : CategoryTheory.Category.{u_15, u_5} C₃
              inst✝¹⁷ : CategoryTheory.Category.{u_16, u_6} C₄
              inst✝¹⁶ : CategoryTheory.Category.{u_17, u_3} C₁₂
              inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
              inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
              inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₃
              inst✝¹² : CategoryTheory.Preadditive C₁₂
              inst✝¹¹ : CategoryTheory.Preadditive C₄
              F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
              G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
              inst✝¹⁰ : F₁₂.PreservesZeroMorphisms
              inst✝⁹ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
              inst✝⁸ : G.Additive
              inst✝⁷ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
              ι₁ : Type u_7
              ι₂ : Type u_8
              ι₃ : Type u_9
              ι₁₂ : Type u_10
              ι₄ : Type u_12
              inst✝⁶ : DecidableEq ι₄
              c₁ : ComplexShape ι₁
              c₂ : ComplexShape ι₂
              c₃ : ComplexShape ι₃
              K₁ : HomologicalComplex C₁ c₁
              K₂ : HomologicalComplex C₂ c₂
              K₃ : HomologicalComplex C₃ c₃
              c₁₂ : ComplexShape ι₁₂
              c₄ : ComplexShape ι₄
              inst✝⁵ : TotalComplexShape c₁ c₂ c₁₂
              inst✝⁴ : TotalComplexShape c₁₂ c₃ c₄
              inst✝³ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
              inst✝² : DecidableEq ι₁₂
              inst✝¹ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
              j j' : ι₄
              inst✝ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
              i₁ : ι₁
              i₂ : ι₂
              i₃ : ι₃
              h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
              i₁₂ : ι₁₂ := c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }
              h₁ : c₁₂.Rel i₁₂ (c₁₂.next i₁₂)
              h₂ : Not (Eq (c₁₂.π c₃ c₄ { fst := c₁₂.next i₁₂, snd := i₃ }) j')
              h₃ : Not (c₂.Rel i₂ (c₂.next i₂))
              ⊢ Eq 0 (HomologicalComplex.mapBifunctor₁₂.d₂ F₁₂ G K₁ K₂ K₃ c₁₂ c₄ i₁ i₂ i₃ j')
            -/
          · rw [d₂_eq_zero _ _ _ _ _ _ _ _ _ _ _ h₃]
            /-
              🎉 no goals
            -/
  · rw [mapBifunctor.d₁_eq_zero _ _ _ _ _ _ _ h₁, comp_zero,
      d₁_eq_zero, d₂_eq_zero, zero_add]
      /-
        case neg.h
        C₁ : Type u_1
        C₂ : Type u_2
        C₁₂ : Type u_3
        C₃ : Type u_5
        C₄ : Type u_6
        inst✝²⁰ : CategoryTheory.Category.{u_13, u_1} C₁
        inst✝¹⁹ : CategoryTheory.Category.{u_14, u_2} C₂
        inst✝¹⁸ : CategoryTheory.Category.{u_15, u_5} C₃
        inst✝¹⁷ : CategoryTheory.Category.{u_16, u_6} C₄
        inst✝¹⁶ : CategoryTheory.Category.{u_17, u_3} C₁₂
        inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
        inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
        inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₃
        inst✝¹² : CategoryTheory.Preadditive C₁₂
        inst✝¹¹ : CategoryTheory.Preadditive C₄
        F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
        G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
        inst✝¹⁰ : F₁₂.PreservesZeroMorphisms
        inst✝⁹ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
        inst✝⁸ : G.Additive
        inst✝⁷ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
        ι₁ : Type u_7
        ι₂ : Type u_8
        ι₃ : Type u_9
        ι₁₂ : Type u_10
        ι₄ : Type u_12
        inst✝⁶ : DecidableEq ι₄
        c₁ : ComplexShape ι₁
        c₂ : ComplexShape ι₂
        c₃ : ComplexShape ι₃
        K₁ : HomologicalComplex C₁ c₁
        K₂ : HomologicalComplex C₂ c₂
        K₃ : HomologicalComplex C₃ c₃
        c₁₂ : ComplexShape ι₁₂
        c₄ : ComplexShape ι₄
        inst✝⁵ : TotalComplexShape c₁ c₂ c₁₂
        inst✝⁴ : TotalComplexShape c₁₂ c₃ c₄
        inst✝³ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
        inst✝² : DecidableEq ι₁₂
        inst✝¹ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
        j j' : ι₄
        inst✝ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
        i₁ : ι₁
        i₂ : ι₂
        i₃ : ι₃
        h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
        i₁₂ : ι₁₂ := c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }
        h₁ : Not (c₁₂.Rel i₁₂ (c₁₂.next i₁₂))
        ⊢ Not (c₂.Rel i₂ (c₂.next i₂))
      -/
    · intro h₂
      /-
        case neg.h
        C₁ : Type u_1
        C₂ : Type u_2
        C₁₂ : Type u_3
        C₃ : Type u_5
        C₄ : Type u_6
        inst✝²⁰ : CategoryTheory.Category.{u_13, u_1} C₁
        inst✝¹⁹ : CategoryTheory.Category.{u_14, u_2} C₂
        inst✝¹⁸ : CategoryTheory.Category.{u_15, u_5} C₃
        inst✝¹⁷ : CategoryTheory.Category.{u_16, u_6} C₄
        inst✝¹⁶ : CategoryTheory.Category.{u_17, u_3} C₁₂
        inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
        inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
        inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₃
        inst✝¹² : CategoryTheory.Preadditive C₁₂
        inst✝¹¹ : CategoryTheory.Preadditive C₄
        F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
        G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
        inst✝¹⁰ : F₁₂.PreservesZeroMorphisms
        inst✝⁹ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
        inst✝⁸ : G.Additive
        inst✝⁷ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
        ι₁ : Type u_7
        ι₂ : Type u_8
        ι₃ : Type u_9
        ι₁₂ : Type u_10
        ι₄ : Type u_12
        inst✝⁶ : DecidableEq ι₄
        c₁ : ComplexShape ι₁
        c₂ : ComplexShape ι₂
        c₃ : ComplexShape ι₃
        K₁ : HomologicalComplex C₁ c₁
        K₂ : HomologicalComplex C₂ c₂
        K₃ : HomologicalComplex C₃ c₃
        c₁₂ : ComplexShape ι₁₂
        c₄ : ComplexShape ι₄
        inst✝⁵ : TotalComplexShape c₁ c₂ c₁₂
        inst✝⁴ : TotalComplexShape c₁₂ c₃ c₄
        inst✝³ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
        inst✝² : DecidableEq ι₁₂
        inst✝¹ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
        j j' : ι₄
        inst✝ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
        i₁ : ι₁
        i₂ : ι₂
        i₃ : ι₃
        h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
        i₁₂ : ι₁₂ := c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }
        h₁ : Not (c₁₂.Rel i₁₂ (c₁₂.next i₁₂))
        h₂ : c₂.Rel i₂ (c₂.next i₂)
        ⊢ False
      -/
      apply h₁
      /-
        case neg.h
        C₁ : Type u_1
        C₂ : Type u_2
        C₁₂ : Type u_3
        C₃ : Type u_5
        C₄ : Type u_6
        inst✝²⁰ : CategoryTheory.Category.{u_13, u_1} C₁
        inst✝¹⁹ : CategoryTheory.Category.{u_14, u_2} C₂
        inst✝¹⁸ : CategoryTheory.Category.{u_15, u_5} C₃
        inst✝¹⁷ : CategoryTheory.Category.{u_16, u_6} C₄
        inst✝¹⁶ : CategoryTheory.Category.{u_17, u_3} C₁₂
        inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
        inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
        inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₃
        inst✝¹² : CategoryTheory.Preadditive C₁₂
        inst✝¹¹ : CategoryTheory.Preadditive C₄
        F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
        G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
        inst✝¹⁰ : F₁₂.PreservesZeroMorphisms
        inst✝⁹ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
        inst✝⁸ : G.Additive
        inst✝⁷ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
        ι₁ : Type u_7
        ι₂ : Type u_8
        ι₃ : Type u_9
        ι₁₂ : Type u_10
        ι₄ : Type u_12
        inst✝⁶ : DecidableEq ι₄
        c₁ : ComplexShape ι₁
        c₂ : ComplexShape ι₂
        c₃ : ComplexShape ι₃
        K₁ : HomologicalComplex C₁ c₁
        K₂ : HomologicalComplex C₂ c₂
        K₃ : HomologicalComplex C₃ c₃
        c₁₂ : ComplexShape ι₁₂
        c₄ : ComplexShape ι₄
        inst✝⁵ : TotalComplexShape c₁ c₂ c₁₂
        inst✝⁴ : TotalComplexShape c₁₂ c₃ c₄
        inst✝³ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
        inst✝² : DecidableEq ι₁₂
        inst✝¹ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
        j j' : ι₄
        inst✝ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
        i₁ : ι₁
        i₂ : ι₂
        i₃ : ι₃
        h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
        i₁₂ : ι₁₂ := c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }
        h₁ : Not (c₁₂.Rel i₁₂ (c₁₂.next i₁₂))
        h₂ : c₂.Rel i₂ (c₂.next i₂)
        ⊢ c₁₂.Rel i₁₂ (c₁₂.next i₁₂)
      -/
      have := ComplexShape.rel_π₂ c₁ c₁₂ i₁ h₂
      /-
        case neg.h
        C₁ : Type u_1
        C₂ : Type u_2
        C₁₂ : Type u_3
        C₃ : Type u_5
        C₄ : Type u_6
        inst✝²⁰ : CategoryTheory.Category.{u_13, u_1} C₁
        inst✝¹⁹ : CategoryTheory.Category.{u_14, u_2} C₂
        inst✝¹⁸ : CategoryTheory.Category.{u_15, u_5} C₃
        inst✝¹⁷ : CategoryTheory.Category.{u_16, u_6} C₄
        inst✝¹⁶ : CategoryTheory.Category.{u_17, u_3} C₁₂
        inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
        inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
        inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₃
        inst✝¹² : CategoryTheory.Preadditive C₁₂
        inst✝¹¹ : CategoryTheory.Preadditive C₄
        F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
        G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
        inst✝¹⁰ : F₁₂.PreservesZeroMorphisms
        inst✝⁹ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
        inst✝⁸ : G.Additive
        inst✝⁷ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
        ι₁ : Type u_7
        ι₂ : Type u_8
        ι₃ : Type u_9
        ι₁₂ : Type u_10
        ι₄ : Type u_12
        inst✝⁶ : DecidableEq ι₄
        c₁ : ComplexShape ι₁
        c₂ : ComplexShape ι₂
        c₃ : ComplexShape ι₃
        K₁ : HomologicalComplex C₁ c₁
        K₂ : HomologicalComplex C₂ c₂
        K₃ : HomologicalComplex C₃ c₃
        c₁₂ : ComplexShape ι₁₂
        c₄ : ComplexShape ι₄
        inst✝⁵ : TotalComplexShape c₁ c₂ c₁₂
        inst✝⁴ : TotalComplexShape c₁₂ c₃ c₄
        inst✝³ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
        inst✝² : DecidableEq ι₁₂
        inst✝¹ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
        j j' : ι₄
        inst✝ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
        i₁ : ι₁
        i₂ : ι₂
        i₃ : ι₃
        h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
        i₁₂ : ι₁₂ := c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }
        h₁ : Not (c₁₂.Rel i₁₂ (c₁₂.next i₁₂))
        h₂ : c₂.Rel i₂ (c₂.next i₂)
        this : c₁₂.Rel (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) (c₁.π c₂ c₁₂ { fst := i₁ …
        ⊢ c₁₂.Rel i₁₂ (c₁₂.next i₁₂)
      -/
      rw [c₁₂.next_eq' this]
      /-
        case neg.h
        C₁ : Type u_1
        C₂ : Type u_2
        C₁₂ : Type u_3
        C₃ : Type u_5
        C₄ : Type u_6
        inst✝²⁰ : CategoryTheory.Category.{u_13, u_1} C₁
        inst✝¹⁹ : CategoryTheory.Category.{u_14, u_2} C₂
        inst✝¹⁸ : CategoryTheory.Category.{u_15, u_5} C₃
        inst✝¹⁷ : CategoryTheory.Category.{u_16, u_6} C₄
        inst✝¹⁶ : CategoryTheory.Category.{u_17, u_3} C₁₂
        inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
        inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
        inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₃
        inst✝¹² : CategoryTheory.Preadditive C₁₂
        inst✝¹¹ : CategoryTheory.Preadditive C₄
        F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
        G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
        inst✝¹⁰ : F₁₂.PreservesZeroMorphisms
        inst✝⁹ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
        inst✝⁸ : G.Additive
        inst✝⁷ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
        ι₁ : Type u_7
        ι₂ : Type u_8
        ι₃ : Type u_9
        ι₁₂ : Type u_10
        ι₄ : Type u_12
        inst✝⁶ : DecidableEq ι₄
        c₁ : ComplexShape ι₁
        c₂ : ComplexShape ι₂
        c₃ : ComplexShape ι₃
        K₁ : HomologicalComplex C₁ c₁
        K₂ : HomologicalComplex C₂ c₂
        K₃ : HomologicalComplex C₃ c₃
        c₁₂ : ComplexShape ι₁₂
        c₄ : ComplexShape ι₄
        inst✝⁵ : TotalComplexShape c₁ c₂ c₁₂
        inst✝⁴ : TotalComplexShape c₁₂ c₃ c₄
        inst✝³ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
        inst✝² : DecidableEq ι₁₂
        inst✝¹ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
        j j' : ι₄
        inst✝ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
        i₁ : ι₁
        i₂ : ι₂
        i₃ : ι₃
        h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
        i₁₂ : ι₁₂ := c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }
        h₁ : Not (c₁₂.Rel i₁₂ (c₁₂.next i₁₂))
        h₂ : c₂.Rel i₂ (c₂.next i₂)
        this : c₁₂.Rel (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) (c₁.π c₂ c₁₂ { fst := i₁ …
        ⊢ c₁₂.Rel i₁₂ (c₁.π c₂ c₁₂ { fst := i₁, snd := c₂.next i₂ })
      -/
      exact this
      /-
        🎉 no goals
      -/
      /-
        case neg.h
        C₁ : Type u_1
        C₂ : Type u_2
        C₁₂ : Type u_3
        C₃ : Type u_5
        C₄ : Type u_6
        inst✝²⁰ : CategoryTheory.Category.{u_13, u_1} C₁
        inst✝¹⁹ : CategoryTheory.Category.{u_14, u_2} C₂
        inst✝¹⁸ : CategoryTheory.Category.{u_15, u_5} C₃
        inst✝¹⁷ : CategoryTheory.Category.{u_16, u_6} C₄
        inst✝¹⁶ : CategoryTheory.Category.{u_17, u_3} C₁₂
        inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
        inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
        inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₃
        inst✝¹² : CategoryTheory.Preadditive C₁₂
        inst✝¹¹ : CategoryTheory.Preadditive C₄
        F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
        G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
        inst✝¹⁰ : F₁₂.PreservesZeroMorphisms
        inst✝⁹ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
        inst✝⁸ : G.Additive
        inst✝⁷ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
        ι₁ : Type u_7
        ι₂ : Type u_8
        ι₃ : Type u_9
        ι₁₂ : Type u_10
        ι₄ : Type u_12
        inst✝⁶ : DecidableEq ι₄
        c₁ : ComplexShape ι₁
        c₂ : ComplexShape ι₂
        c₃ : ComplexShape ι₃
        K₁ : HomologicalComplex C₁ c₁
        K₂ : HomologicalComplex C₂ c₂
        K₃ : HomologicalComplex C₃ c₃
        c₁₂ : ComplexShape ι₁₂
        c₄ : ComplexShape ι₄
        inst✝⁵ : TotalComplexShape c₁ c₂ c₁₂
        inst✝⁴ : TotalComplexShape c₁₂ c₃ c₄
        inst✝³ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
        inst✝² : DecidableEq ι₁₂
        inst✝¹ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
        j j' : ι₄
        inst✝ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
        i₁ : ι₁
        i₂ : ι₂
        i₃ : ι₃
        h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
        i₁₂ : ι₁₂ := c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }
        h₁ : Not (c₁₂.Rel i₁₂ (c₁₂.next i₁₂))
        ⊢ Not (c₁.Rel i₁ (c₁.next i₁))
      -/
    · intro h₂
      /-
        case neg.h
        C₁ : Type u_1
        C₂ : Type u_2
        C₁₂ : Type u_3
        C₃ : Type u_5
        C₄ : Type u_6
        inst✝²⁰ : CategoryTheory.Category.{u_13, u_1} C₁
        inst✝¹⁹ : CategoryTheory.Category.{u_14, u_2} C₂
        inst✝¹⁸ : CategoryTheory.Category.{u_15, u_5} C₃
        inst✝¹⁷ : CategoryTheory.Category.{u_16, u_6} C₄
        inst✝¹⁶ : CategoryTheory.Category.{u_17, u_3} C₁₂
        inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
        inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
        inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₃
        inst✝¹² : CategoryTheory.Preadditive C₁₂
        inst✝¹¹ : CategoryTheory.Preadditive C₄
        F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
        G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
        inst✝¹⁰ : F₁₂.PreservesZeroMorphisms
        inst✝⁹ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
        inst✝⁸ : G.Additive
        inst✝⁷ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
        ι₁ : Type u_7
        ι₂ : Type u_8
        ι₃ : Type u_9
        ι₁₂ : Type u_10
        ι₄ : Type u_12
        inst✝⁶ : DecidableEq ι₄
        c₁ : ComplexShape ι₁
        c₂ : ComplexShape ι₂
        c₃ : ComplexShape ι₃
        K₁ : HomologicalComplex C₁ c₁
        K₂ : HomologicalComplex C₂ c₂
        K₃ : HomologicalComplex C₃ c₃
        c₁₂ : ComplexShape ι₁₂
        c₄ : ComplexShape ι₄
        inst✝⁵ : TotalComplexShape c₁ c₂ c₁₂
        inst✝⁴ : TotalComplexShape c₁₂ c₃ c₄
        inst✝³ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
        inst✝² : DecidableEq ι₁₂
        inst✝¹ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
        j j' : ι₄
        inst✝ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
        i₁ : ι₁
        i₂ : ι₂
        i₃ : ι₃
        h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
        i₁₂ : ι₁₂ := c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }
        h₁ : Not (c₁₂.Rel i₁₂ (c₁₂.next i₁₂))
        h₂ : c₁.Rel i₁ (c₁.next i₁)
        ⊢ False
      -/
      apply h₁
      /-
        case neg.h
        C₁ : Type u_1
        C₂ : Type u_2
        C₁₂ : Type u_3
        C₃ : Type u_5
        C₄ : Type u_6
        inst✝²⁰ : CategoryTheory.Category.{u_13, u_1} C₁
        inst✝¹⁹ : CategoryTheory.Category.{u_14, u_2} C₂
        inst✝¹⁸ : CategoryTheory.Category.{u_15, u_5} C₃
        inst✝¹⁷ : CategoryTheory.Category.{u_16, u_6} C₄
        inst✝¹⁶ : CategoryTheory.Category.{u_17, u_3} C₁₂
        inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
        inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
        inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₃
        inst✝¹² : CategoryTheory.Preadditive C₁₂
        inst✝¹¹ : CategoryTheory.Preadditive C₄
        F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
        G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
        inst✝¹⁰ : F₁₂.PreservesZeroMorphisms
        inst✝⁹ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
        inst✝⁸ : G.Additive
        inst✝⁷ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
        ι₁ : Type u_7
        ι₂ : Type u_8
        ι₃ : Type u_9
        ι₁₂ : Type u_10
        ι₄ : Type u_12
        inst✝⁶ : DecidableEq ι₄
        c₁ : ComplexShape ι₁
        c₂ : ComplexShape ι₂
        c₃ : ComplexShape ι₃
        K₁ : HomologicalComplex C₁ c₁
        K₂ : HomologicalComplex C₂ c₂
        K₃ : HomologicalComplex C₃ c₃
        c₁₂ : ComplexShape ι₁₂
        c₄ : ComplexShape ι₄
        inst✝⁵ : TotalComplexShape c₁ c₂ c₁₂
        inst✝⁴ : TotalComplexShape c₁₂ c₃ c₄
        inst✝³ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
        inst✝² : DecidableEq ι₁₂
        inst✝¹ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
        j j' : ι₄
        inst✝ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
        i₁ : ι₁
        i₂ : ι₂
        i₃ : ι₃
        h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
        i₁₂ : ι₁₂ := c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }
        h₁ : Not (c₁₂.Rel i₁₂ (c₁₂.next i₁₂))
        h₂ : c₁.Rel i₁ (c₁.next i₁)
        ⊢ c₁₂.Rel i₁₂ (c₁₂.next i₁₂)
      -/
      have := ComplexShape.rel_π₁ c₂ c₁₂ h₂ i₂
      /-
        case neg.h
        C₁ : Type u_1
        C₂ : Type u_2
        C₁₂ : Type u_3
        C₃ : Type u_5
        C₄ : Type u_6
        inst✝²⁰ : CategoryTheory.Category.{u_13, u_1} C₁
        inst✝¹⁹ : CategoryTheory.Category.{u_14, u_2} C₂
        inst✝¹⁸ : CategoryTheory.Category.{u_15, u_5} C₃
        inst✝¹⁷ : CategoryTheory.Category.{u_16, u_6} C₄
        inst✝¹⁶ : CategoryTheory.Category.{u_17, u_3} C₁₂
        inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
        inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
        inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₃
        inst✝¹² : CategoryTheory.Preadditive C₁₂
        inst✝¹¹ : CategoryTheory.Preadditive C₄
        F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
        G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
        inst✝¹⁰ : F₁₂.PreservesZeroMorphisms
        inst✝⁹ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
        inst✝⁸ : G.Additive
        inst✝⁷ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
        ι₁ : Type u_7
        ι₂ : Type u_8
        ι₃ : Type u_9
        ι₁₂ : Type u_10
        ι₄ : Type u_12
        inst✝⁶ : DecidableEq ι₄
        c₁ : ComplexShape ι₁
        c₂ : ComplexShape ι₂
        c₃ : ComplexShape ι₃
        K₁ : HomologicalComplex C₁ c₁
        K₂ : HomologicalComplex C₂ c₂
        K₃ : HomologicalComplex C₃ c₃
        c₁₂ : ComplexShape ι₁₂
        c₄ : ComplexShape ι₄
        inst✝⁵ : TotalComplexShape c₁ c₂ c₁₂
        inst✝⁴ : TotalComplexShape c₁₂ c₃ c₄
        inst✝³ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
        inst✝² : DecidableEq ι₁₂
        inst✝¹ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
        j j' : ι₄
        inst✝ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
        i₁ : ι₁
        i₂ : ι₂
        i₃ : ι₃
        h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
        i₁₂ : ι₁₂ := c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }
        h₁ : Not (c₁₂.Rel i₁₂ (c₁₂.next i₁₂))
        h₂ : c₁.Rel i₁ (c₁.next i₁)
        this : c₁₂.Rel (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) (c₁.π c₂ c₁₂ { fst := c₁ …
        ⊢ c₁₂.Rel i₁₂ (c₁₂.next i₁₂)
      -/
      rw [c₁₂.next_eq' this]
      /-
        case neg.h
        C₁ : Type u_1
        C₂ : Type u_2
        C₁₂ : Type u_3
        C₃ : Type u_5
        C₄ : Type u_6
        inst✝²⁰ : CategoryTheory.Category.{u_13, u_1} C₁
        inst✝¹⁹ : CategoryTheory.Category.{u_14, u_2} C₂
        inst✝¹⁸ : CategoryTheory.Category.{u_15, u_5} C₃
        inst✝¹⁷ : CategoryTheory.Category.{u_16, u_6} C₄
        inst✝¹⁶ : CategoryTheory.Category.{u_17, u_3} C₁₂
        inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₁
        inst✝¹⁴ : CategoryTheory.Limits.HasZeroMorphisms C₂
        inst✝¹³ : CategoryTheory.Limits.HasZeroMorphisms C₃
        inst✝¹² : CategoryTheory.Preadditive C₁₂
        inst✝¹¹ : CategoryTheory.Preadditive C₄
        F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
        G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
        inst✝¹⁰ : F₁₂.PreservesZeroMorphisms
        inst✝⁹ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
        inst✝⁸ : G.Additive
        inst✝⁷ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
        ι₁ : Type u_7
        ι₂ : Type u_8
        ι₃ : Type u_9
        ι₁₂ : Type u_10
        ι₄ : Type u_12
        inst✝⁶ : DecidableEq ι₄
        c₁ : ComplexShape ι₁
        c₂ : ComplexShape ι₂
        c₃ : ComplexShape ι₃
        K₁ : HomologicalComplex C₁ c₁
        K₂ : HomologicalComplex C₂ c₂
        K₃ : HomologicalComplex C₃ c₃
        c₁₂ : ComplexShape ι₁₂
        c₄ : ComplexShape ι₄
        inst✝⁵ : TotalComplexShape c₁ c₂ c₁₂
        inst✝⁴ : TotalComplexShape c₁₂ c₃ c₄
        inst✝³ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
        inst✝² : DecidableEq ι₁₂
        inst✝¹ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
        j j' : ι₄
        inst✝ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
        i₁ : ι₁
        i₂ : ι₂
        i₃ : ι₃
        h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
        i₁₂ : ι₁₂ := c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }
        h₁ : Not (c₁₂.Rel i₁₂ (c₁₂.next i₁₂))
        h₂ : c₁.Rel i₁ (c₁.next i₁)
        this : c₁₂.Rel (c₁.π c₂ c₁₂ { fst := i₁, snd := i₂ }) (c₁.π c₂ c₁₂ { fst := c₁ …
        ⊢ c₁₂.Rel i₁₂ (c₁.π c₂ c₁₂ { fst := c₁.next i₁, snd := i₂ })
      -/
      exact this
      /-
        🎉 no goals
      -/


/-- The inclusion of a summand in `mapBifunctor K₁ (mapBifunctor K₂ K₃ G₂₃ c₂₃) F c₄`. -/
noncomputable def ι (i₁ : ι₁) (i₂ : ι₂) (i₃ : ι₃) (j : ι₄)
    (h : ComplexShape.r c₁ c₂ c₃ c₁₂ c₄ (i₁, i₂, i₃) = j) :
    (F.obj (K₁.X i₁)).obj ((G₂₃.obj (K₂.X i₂)).obj (K₃.X i₃)) ⟶
      (mapBifunctor K₁ (mapBifunctor K₂ K₃ G₂₃ c₂₃) F c₄).X j :=
  GradedObject.ιMapBifunctorBifunctor₂₃MapObj _ _ (ComplexShape.ρ₂₃ c₁ c₂ c₃ c₁₂ c₂₃ c₄)
    _ _ _ _ _ _ _ h


lemma ι_eq (i₁ : ι₁) (i₂ : ι₂) (i₃ : ι₃) (i₂₃ : ι₂₃) (j : ι₄)
    (h₂₃ : ComplexShape.π c₂ c₃ c₂₃ ⟨i₂, i₃⟩ = i₂₃)
    (h : ComplexShape.π c₁ c₂₃ c₄ (i₁, i₂₃) = j) :
    ι F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ i₁ i₂ i₃ j
          /-
            C₁ : Type u_1
            C₂ : Type u_2
            C₁₂ : Type u_3
            C₂₃ : Type u_4
            C₃ : Type u_5
            C₄ : Type u_6
            inst✝²⁹ : CategoryTheory.Category.{?u.601231, u_1} C₁
            inst✝²⁸ : CategoryTheory.Category.{?u.601235, u_2} C₂
            inst✝²⁷ : CategoryTheory.Category.{?u.601239, u_5} C₃
            inst✝²⁶ : CategoryTheory.Category.{?u.601243, u_6} C₄
            inst✝²⁵ : CategoryTheory.Category.{?u.601247, u_3} C₁₂
            inst✝²⁴ : CategoryTheory.Category.{?u.601251, u_4} C₂₃
            inst✝²³ : CategoryTheory.Limits.HasZeroMorphisms C₁
            inst✝²² : CategoryTheory.Limits.HasZeroMorphisms C₂
            inst✝²¹ : CategoryTheory.Limits.HasZeroMorphisms C₃
            inst✝²⁰ : CategoryTheory.Preadditive C₁₂
            inst✝¹⁹ : CategoryTheory.Preadditive C₂₃
            inst✝¹⁸ : CategoryTheory.Preadditive C₄
            F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
            G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
            F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
            G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
            inst✝¹⁷ : F₁₂.PreservesZeroMorphisms
            inst✝¹⁶ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
            inst✝¹⁵ : G.Additive
            inst✝¹⁴ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
            inst✝¹³ : G₂₃.PreservesZeroMorphisms
            inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
            inst✝¹¹ : F.PreservesZeroMorphisms
            inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
            associator : CategoryTheory.Iso (CategoryTheory.bifunctorComp₁₂ F₁₂ G) (Catego …
            ι₁ : Type u_7
            ι₂ : Type u_8
            ι₃ : Type u_9
            ι₁₂ : Type u_10
            ι₂₃ : Type u_11
            ι₄ : Type u_12
            inst✝⁹ : DecidableEq ι₄
            c₁ : ComplexShape ι₁
            c₂ : ComplexShape ι₂
            c₃ : ComplexShape ι₃
            K₁ : HomologicalComplex C₁ c₁
            K₂ : HomologicalComplex C₂ c₂
            K₃ : HomologicalComplex C₃ c₃
            c₁₂ : ComplexShape ι₁₂
            c₂₃ : ComplexShape ι₂₃
            c₄ : ComplexShape ι₄
            inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
            inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
            inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
            inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
            inst✝⁴ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
            inst✝³ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
            inst✝² : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
            inst✝¹ : DecidableEq ι₂₃
            inst✝ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
            i₁ : ι₁
            i₂ : ι₂
            i₃ : ι₃
            i₂₃ : ι₂₃
            j : ι₄
            h₂₃ : Eq (c₂.π c₃ c₂₃ { fst := i₂, snd := i₃ }) i₂₃
            h : Eq (c₁.π c₂₃ c₄ { fst := i₁, snd := i₂₃ }) j
            ⊢ Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
          -/
      (by rw [← h, ← h₂₃, ← ComplexShape.assoc c₁ c₂ c₃ c₁₂ c₂₃ c₄]; rfl) =
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
      (F.obj (K₁.X i₁)).map (ιMapBifunctor K₂ K₃ G₂₃ c₂₃ i₂ i₃ i₂₃ h₂₃) ≫
        ιMapBifunctor K₁ (mapBifunctor K₂ K₃ G₂₃ c₂₃) F c₄ i₁ i₂₃ j h := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₂₃ : Type u_4
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝²² : CategoryTheory.Category.{u_15, u_1} C₁
    inst✝²¹ : CategoryTheory.Category.{u_17, u_2} C₂
    inst✝²⁰ : CategoryTheory.Category.{u_16, u_5} C₃
    inst✝¹⁹ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝¹⁸ : CategoryTheory.Category.{u_14, u_4} C₂₃
    inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹⁴ : CategoryTheory.Preadditive C₂₃
    inst✝¹³ : CategoryTheory.Preadditive C₄
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    inst✝¹² : G₂₃.PreservesZeroMorphisms
    inst✝¹¹ : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
    inst✝¹⁰ : F.PreservesZeroMorphisms
    inst✝⁹ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₂₃ : Type u_11
    ι₄ : Type u_12
    inst✝⁸ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₂₃ : ComplexShape ι₂₃
    c₄ : ComplexShape ι₄
    inst✝⁷ : TotalComplexShape c₁ c₂ c₁₂
    inst✝⁶ : TotalComplexShape c₁₂ c₃ c₄
    inst✝⁵ : TotalComplexShape c₂ c₃ c₂₃
    inst✝⁴ : TotalComplexShape c₁ c₂₃ c₄
    inst✝³ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
    inst✝² : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
    inst✝¹ : DecidableEq ι₂₃
    inst✝ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
    i₁ : ι₁
    i₂ : ι₂
    i₃ : ι₃
    i₂₃ : ι₂₃
    j : ι₄
    h₂₃ : Eq (c₂.π c₃ c₂₃ { fst := i₂, snd := i₃ }) i₂₃
    h : Eq (c₁.π c₂₃ c₄ { fst := i₁, snd := i₂₃ }) j
    ⊢ Eq (HomologicalComplex.mapBifunctor₂₃.ι F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ i₁ i₂ i₃ j …
  -/
  subst h₂₃
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₂₃ : Type u_4
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝²² : CategoryTheory.Category.{u_15, u_1} C₁
    inst✝²¹ : CategoryTheory.Category.{u_17, u_2} C₂
    inst✝²⁰ : CategoryTheory.Category.{u_16, u_5} C₃
    inst✝¹⁹ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝¹⁸ : CategoryTheory.Category.{u_14, u_4} C₂₃
    inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹⁴ : CategoryTheory.Preadditive C₂₃
    inst✝¹³ : CategoryTheory.Preadditive C₄
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    inst✝¹² : G₂₃.PreservesZeroMorphisms
    inst✝¹¹ : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
    inst✝¹⁰ : F.PreservesZeroMorphisms
    inst✝⁹ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₂₃ : Type u_11
    ι₄ : Type u_12
    inst✝⁸ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₂₃ : ComplexShape ι₂₃
    c₄ : ComplexShape ι₄
    inst✝⁷ : TotalComplexShape c₁ c₂ c₁₂
    inst✝⁶ : TotalComplexShape c₁₂ c₃ c₄
    inst✝⁵ : TotalComplexShape c₂ c₃ c₂₃
    inst✝⁴ : TotalComplexShape c₁ c₂₃ c₄
    inst✝³ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
    inst✝² : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
    inst✝¹ : DecidableEq ι₂₃
    inst✝ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
    i₁ : ι₁
    i₂ : ι₂
    i₃ : ι₃
    j : ι₄
    h : Eq (c₁.π c₂₃ c₄ { fst := i₁, snd := c₂.π c₃ c₂₃ { fst := i₂, snd := i₃ } } …
    ⊢ Eq (HomologicalComplex.mapBifunctor₂₃.ι F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ i₁ i₂ i₃ j …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The inclusion of a summand in `mapBifunctor K₁ (mapBifunctor K₂ K₃ G₂₃ c₂₃) F c₄`,
or zero. -/
noncomputable def ιOrZero (i₁ : ι₁) (i₂ : ι₂) (i₃ : ι₃) (j : ι₄) :
    (F.obj (K₁.X i₁)).obj ((G₂₃.obj (K₂.X i₂)).obj (K₃.X i₃)) ⟶
      (mapBifunctor K₁ (mapBifunctor K₂ K₃ G₂₃ c₂₃) F c₄).X j :=
  if h : ComplexShape.r c₁ c₂ c₃ c₁₂ c₄ (i₁, i₂, i₃) = j then
    ι F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ i₁ i₂ i₃ j h
  else 0


lemma ιOrZero_eq (i₁ : ι₁) (i₂ : ι₂) (i₃ : ι₃) (j : ι₄)
    (h : ComplexShape.r c₁ c₂ c₃ c₁₂ c₄ (i₁, i₂, i₃) = j) :
    ιOrZero F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ i₁ i₂ i₃ j =
      ι F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ i₁ i₂ i₃ j h := dif_pos h


lemma ιOrZero_eq_zero (i₁ : ι₁) (i₂ : ι₂) (i₃ : ι₃) (j : ι₄)
    (h : ComplexShape.r c₁ c₂ c₃ c₁₂ c₄ (i₁, i₂, i₃) ≠ j) :
    ιOrZero F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ i₁ i₂ i₃ j = 0 := dif_neg h


variable {F G₂₃ K₁ K₂ K₃ c₂₃ c₄} in
lemma hom_ext {j : ι₄} {A : C₄}
    {f g : (mapBifunctor K₁ (mapBifunctor K₂ K₃ G₂₃ c₂₃) F c₄).X j ⟶ A}
    (hfg : ∀ (i₁ : ι₁) (i₂ : ι₂) (i₃ : ι₃)
      (h : ComplexShape.r c₁ c₂ c₃ c₁₂ c₄ (i₁, i₂, i₃) = j),
      ι F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ i₁ i₂ i₃ j h ≫ f =
        ι F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ i₁ i₂ i₃ j h ≫ g) :
    f = g :=
  GradedObject.mapBifunctorBifunctor₂₃MapObj_ext
    (ρ₂₃ := ComplexShape.ρ₂₃ c₁ c₂ c₃ c₁₂ c₂₃ c₄) hfg


/-- Constructor for morphisms from
`(mapBifunctor K₁ (mapBifunctor K₂ K₃ G₂₃ c₂₃) F c₄).X j`. -/
noncomputable def mapBifunctor₂₃Desc :
    (mapBifunctor K₁ (mapBifunctor K₂ K₃ G₂₃ c₂₃) F c₄).X j ⟶ A :=
  GradedObject.mapBifunctorBifunctor₂₃Desc (ρ₂₃ := ComplexShape.ρ₂₃ c₁ c₂ c₃ c₁₂ c₂₃ c₄) f


@[reassoc (attr := simp)]
lemma ι_mapBifunctor₂₃Desc (i₁ : ι₁) (i₂ : ι₂) (i₃ : ι₃)
    (h : ComplexShape.r c₁ c₂ c₃ c₁₂ c₄ (i₁, i₂, i₃) = j) :
    ι F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ i₁ i₂ i₃ j h ≫ mapBifunctor₂₃Desc c₁₂ f =
      f i₁ i₂ i₃ h := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₂₃ : Type u_4
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝²³ : CategoryTheory.Category.{u_15, u_1} C₁
    inst✝²² : CategoryTheory.Category.{u_17, u_2} C₂
    inst✝²¹ : CategoryTheory.Category.{u_16, u_5} C₃
    inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝¹⁹ : CategoryTheory.Category.{u_14, u_4} C₂₃
    inst✝¹⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹⁵ : CategoryTheory.Preadditive C₂₃
    inst✝¹⁴ : CategoryTheory.Preadditive C₄
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    inst✝¹³ : G₂₃.PreservesZeroMorphisms
    inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
    inst✝¹¹ : F.PreservesZeroMorphisms
    inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₂₃ : Type u_11
    ι₄ : Type u_12
    inst✝⁹ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₂₃ : ComplexShape ι₂₃
    c₄ : ComplexShape ι₄
    inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
    inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
    inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
    inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
    inst✝⁴ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
    inst✝³ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
    inst✝² : DecidableEq ι₂₃
    inst✝¹ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
    inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
    j : ι₄
    A : C₄
    f : (i₁ : ι₁) → (i₂ : ι₂) → (i₃ : ι₃) → Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd …
    i₁ : ι₁
    i₂ : ι₂
    i₃ : ι₃
    h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.mapBifunctor₂₃.ι  …
  -/
  apply GradedObject.ι_mapBifunctorBifunctor₂₃Desc
  /-
    🎉 no goals
  -/


/-- The first differential on a summand
of `mapBifunctor K₁ (mapBifunctor K₂ K₃ G₂₃ c₂₃) F c₄`. -/
noncomputable def d₁ (i₁ : ι₁) (i₂ : ι₂) (i₃ : ι₃) (j : ι₄) :
    (F.obj (K₁.X i₁)).obj ((G₂₃.obj (K₂.X i₂)).obj (K₃.X i₃)) ⟶
      (mapBifunctor K₁ (mapBifunctor K₂ K₃ G₂₃ c₂₃) F c₄).X j :=
  (ComplexShape.ε₁ c₁ c₂₃ c₄ (i₁, ComplexShape.π c₂ c₃ c₂₃ (i₂, i₃))) •
      ((F.map (K₁.d i₁ (c₁.next i₁)))).app ((G₂₃.obj (K₂.X i₂)).obj (K₃.X i₃)) ≫
        ιOrZero F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ _ i₂ i₃ j


lemma d₁_eq_zero (i₁ : ι₁) (i₂ : ι₂) (i₃ : ι₃) (j : ι₄) (h : ¬ c₁.Rel i₁ (c₁.next i₁)) :
    d₁ F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ i₁ i₂ i₃ j = 0 := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₂₃ : Type u_4
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝²² : CategoryTheory.Category.{u_15, u_1} C₁
    inst✝²¹ : CategoryTheory.Category.{u_17, u_2} C₂
    inst✝²⁰ : CategoryTheory.Category.{u_16, u_5} C₃
    inst✝¹⁹ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝¹⁸ : CategoryTheory.Category.{u_14, u_4} C₂₃
    inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹⁴ : CategoryTheory.Preadditive C₂₃
    inst✝¹³ : CategoryTheory.Preadditive C₄
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    inst✝¹² : G₂₃.PreservesZeroMorphisms
    inst✝¹¹ : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
    inst✝¹⁰ : F.PreservesZeroMorphisms
    inst✝⁹ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₂₃ : Type u_11
    ι₄ : Type u_12
    inst✝⁸ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₂₃ : ComplexShape ι₂₃
    c₄ : ComplexShape ι₄
    inst✝⁷ : TotalComplexShape c₁ c₂ c₁₂
    inst✝⁶ : TotalComplexShape c₁₂ c₃ c₄
    inst✝⁵ : TotalComplexShape c₂ c₃ c₂₃
    inst✝⁴ : TotalComplexShape c₁ c₂₃ c₄
    inst✝³ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
    inst✝² : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
    inst✝¹ : DecidableEq ι₂₃
    inst✝ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
    i₁ : ι₁
    i₂ : ι₂
    i₃ : ι₃
    j : ι₄
    h : Not (c₁.Rel i₁ (c₁.next i₁))
    ⊢ Eq (HomologicalComplex.mapBifunctor₂₃.d₁ F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ i₁ i₂ i₃  …
  -/
  dsimp [d₁]
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₂₃ : Type u_4
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝²² : CategoryTheory.Category.{u_15, u_1} C₁
    inst✝²¹ : CategoryTheory.Category.{u_17, u_2} C₂
    inst✝²⁰ : CategoryTheory.Category.{u_16, u_5} C₃
    inst✝¹⁹ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝¹⁸ : CategoryTheory.Category.{u_14, u_4} C₂₃
    inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹⁴ : CategoryTheory.Preadditive C₂₃
    inst✝¹³ : CategoryTheory.Preadditive C₄
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    inst✝¹² : G₂₃.PreservesZeroMorphisms
    inst✝¹¹ : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
    inst✝¹⁰ : F.PreservesZeroMorphisms
    inst✝⁹ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₂₃ : Type u_11
    ι₄ : Type u_12
    inst✝⁸ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₂₃ : ComplexShape ι₂₃
    c₄ : ComplexShape ι₄
    inst✝⁷ : TotalComplexShape c₁ c₂ c₁₂
    inst✝⁶ : TotalComplexShape c₁₂ c₃ c₄
    inst✝⁵ : TotalComplexShape c₂ c₃ c₂₃
    inst✝⁴ : TotalComplexShape c₁ c₂₃ c₄
    inst✝³ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
    inst✝² : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
    inst✝¹ : DecidableEq ι₂₃
    inst✝ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
    i₁ : ι₁
    i₂ : ι₂
    i₃ : ι₃
    j : ι₄
    h : Not (c₁.Rel i₁ (c₁.next i₁))
    ⊢ Eq (HSMul.hSMul (c₁.ε₁ c₂₃ c₄ { fst := i₁, snd := c₂.π c₃ c₂₃ { fst := i₂, s …
  -/
  rw [shape _ _ _ h, Functor.map_zero, zero_app, zero_comp, smul_zero]
  /-
    🎉 no goals
  -/


lemma d₁_eq {i₁ i₁' : ι₁} (h₁ : c₁.Rel i₁ i₁') (i₂ : ι₂) (i₃ : ι₃) (j : ι₄) :
    d₁ F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ i₁ i₂ i₃ j =
  (ComplexShape.ε₁ c₁ c₂₃ c₄ (i₁, ComplexShape.π c₂ c₃ c₂₃ (i₂, i₃))) •
    ((F.map (K₁.d i₁ i₁'))).app ((G₂₃.obj (K₂.X i₂)).obj (K₃.X i₃)) ≫
      ιOrZero F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ _ i₂ i₃ j := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₂₃ : Type u_4
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝²² : CategoryTheory.Category.{u_15, u_1} C₁
    inst✝²¹ : CategoryTheory.Category.{u_17, u_2} C₂
    inst✝²⁰ : CategoryTheory.Category.{u_16, u_5} C₃
    inst✝¹⁹ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝¹⁸ : CategoryTheory.Category.{u_14, u_4} C₂₃
    inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹⁴ : CategoryTheory.Preadditive C₂₃
    inst✝¹³ : CategoryTheory.Preadditive C₄
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    inst✝¹² : G₂₃.PreservesZeroMorphisms
    inst✝¹¹ : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
    inst✝¹⁰ : F.PreservesZeroMorphisms
    inst✝⁹ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₂₃ : Type u_11
    ι₄ : Type u_12
    inst✝⁸ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₂₃ : ComplexShape ι₂₃
    c₄ : ComplexShape ι₄
    inst✝⁷ : TotalComplexShape c₁ c₂ c₁₂
    inst✝⁶ : TotalComplexShape c₁₂ c₃ c₄
    inst✝⁵ : TotalComplexShape c₂ c₃ c₂₃
    inst✝⁴ : TotalComplexShape c₁ c₂₃ c₄
    inst✝³ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
    inst✝² : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
    inst✝¹ : DecidableEq ι₂₃
    inst✝ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
    i₁ i₁' : ι₁
    h₁ : c₁.Rel i₁ i₁'
    i₂ : ι₂
    i₃ : ι₃
    j : ι₄
    ⊢ Eq (HomologicalComplex.mapBifunctor₂₃.d₁ F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ i₁ i₂ i₃  …
  -/
  obtain rfl := c₁.next_eq' h₁
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₂₃ : Type u_4
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝²² : CategoryTheory.Category.{u_15, u_1} C₁
    inst✝²¹ : CategoryTheory.Category.{u_17, u_2} C₂
    inst✝²⁰ : CategoryTheory.Category.{u_16, u_5} C₃
    inst✝¹⁹ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝¹⁸ : CategoryTheory.Category.{u_14, u_4} C₂₃
    inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹⁴ : CategoryTheory.Preadditive C₂₃
    inst✝¹³ : CategoryTheory.Preadditive C₄
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    inst✝¹² : G₂₃.PreservesZeroMorphisms
    inst✝¹¹ : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
    inst✝¹⁰ : F.PreservesZeroMorphisms
    inst✝⁹ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₂₃ : Type u_11
    ι₄ : Type u_12
    inst✝⁸ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₂₃ : ComplexShape ι₂₃
    c₄ : ComplexShape ι₄
    inst✝⁷ : TotalComplexShape c₁ c₂ c₁₂
    inst✝⁶ : TotalComplexShape c₁₂ c₃ c₄
    inst✝⁵ : TotalComplexShape c₂ c₃ c₂₃
    inst✝⁴ : TotalComplexShape c₁ c₂₃ c₄
    inst✝³ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
    inst✝² : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
    inst✝¹ : DecidableEq ι₂₃
    inst✝ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
    i₁ : ι₁
    i₂ : ι₂
    i₃ : ι₃
    j : ι₄
    h₁ : c₁.Rel i₁ (c₁.next i₁)
    ⊢ Eq (HomologicalComplex.mapBifunctor₂₃.d₁ F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ i₁ i₂ i₃  …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The second differential on a summand
of `mapBifunctor K₁ (mapBifunctor K₂ K₃ G₂₃ c₂₃) F c₄`. -/
noncomputable def d₂ (i₁ : ι₁) (i₂ : ι₂) (i₃ : ι₃) (j : ι₄) :
    (F.obj (K₁.X i₁)).obj ((G₂₃.obj (K₂.X i₂)).obj (K₃.X i₃)) ⟶
      (mapBifunctor K₁ (mapBifunctor K₂ K₃ G₂₃ c₂₃) F c₄).X j :=
  (ComplexShape.ε₂ c₁ c₂₃ c₄ (i₁, c₂.π c₃ c₂₃ (i₂, i₃)) * ComplexShape.ε₁ c₂ c₃ c₂₃ (i₂, i₃)) •
    (F.obj (K₁.X i₁)).map ((G₂₃.map (K₂.d i₂ (c₂.next i₂))).app (K₃.X i₃)) ≫
      ιOrZero F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ i₁ _ i₃ j


lemma d₂_eq_zero (i₁ : ι₁) (i₂ : ι₂) (i₃ : ι₃) (j : ι₄) (h : ¬ c₂.Rel i₂ (c₂.next i₂)) :
    d₂ F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ i₁ i₂ i₃ j = 0 := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₂₃ : Type u_4
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝²² : CategoryTheory.Category.{u_15, u_1} C₁
    inst✝²¹ : CategoryTheory.Category.{u_17, u_2} C₂
    inst✝²⁰ : CategoryTheory.Category.{u_16, u_5} C₃
    inst✝¹⁹ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝¹⁸ : CategoryTheory.Category.{u_14, u_4} C₂₃
    inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹⁴ : CategoryTheory.Preadditive C₂₃
    inst✝¹³ : CategoryTheory.Preadditive C₄
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    inst✝¹² : G₂₃.PreservesZeroMorphisms
    inst✝¹¹ : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
    inst✝¹⁰ : F.PreservesZeroMorphisms
    inst✝⁹ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₂₃ : Type u_11
    ι₄ : Type u_12
    inst✝⁸ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₂₃ : ComplexShape ι₂₃
    c₄ : ComplexShape ι₄
    inst✝⁷ : TotalComplexShape c₁ c₂ c₁₂
    inst✝⁶ : TotalComplexShape c₁₂ c₃ c₄
    inst✝⁵ : TotalComplexShape c₂ c₃ c₂₃
    inst✝⁴ : TotalComplexShape c₁ c₂₃ c₄
    inst✝³ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
    inst✝² : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
    inst✝¹ : DecidableEq ι₂₃
    inst✝ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
    i₁ : ι₁
    i₂ : ι₂
    i₃ : ι₃
    j : ι₄
    h : Not (c₂.Rel i₂ (c₂.next i₂))
    ⊢ Eq (HomologicalComplex.mapBifunctor₂₃.d₂ F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ i₁ i₂ i₃  …
  -/
  dsimp [d₂]
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₂₃ : Type u_4
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝²² : CategoryTheory.Category.{u_15, u_1} C₁
    inst✝²¹ : CategoryTheory.Category.{u_17, u_2} C₂
    inst✝²⁰ : CategoryTheory.Category.{u_16, u_5} C₃
    inst✝¹⁹ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝¹⁸ : CategoryTheory.Category.{u_14, u_4} C₂₃
    inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹⁴ : CategoryTheory.Preadditive C₂₃
    inst✝¹³ : CategoryTheory.Preadditive C₄
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    inst✝¹² : G₂₃.PreservesZeroMorphisms
    inst✝¹¹ : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
    inst✝¹⁰ : F.PreservesZeroMorphisms
    inst✝⁹ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₂₃ : Type u_11
    ι₄ : Type u_12
    inst✝⁸ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₂₃ : ComplexShape ι₂₃
    c₄ : ComplexShape ι₄
    inst✝⁷ : TotalComplexShape c₁ c₂ c₁₂
    inst✝⁶ : TotalComplexShape c₁₂ c₃ c₄
    inst✝⁵ : TotalComplexShape c₂ c₃ c₂₃
    inst✝⁴ : TotalComplexShape c₁ c₂₃ c₄
    inst✝³ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
    inst✝² : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
    inst✝¹ : DecidableEq ι₂₃
    inst✝ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
    i₁ : ι₁
    i₂ : ι₂
    i₃ : ι₃
    j : ι₄
    h : Not (c₂.Rel i₂ (c₂.next i₂))
    ⊢ Eq (HSMul.hSMul (HMul.hMul (c₁.ε₂ c₂₃ c₄ { fst := i₁, snd := c₂.π c₃ c₂₃ { f …
  -/
  rw [shape _ _ _ h, Functor.map_zero, zero_app, Functor.map_zero, zero_comp, smul_zero]
  /-
    🎉 no goals
  -/


lemma d₂_eq (i₁ : ι₁) {i₂ i₂' : ι₂} (h₂ : c₂.Rel i₂ i₂') (i₃ : ι₃) (j : ι₄) :
    d₂ F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ i₁ i₂ i₃ j =
      (ComplexShape.ε₂ c₁ c₂₃ c₄ (i₁, c₂.π c₃ c₂₃ (i₂, i₃)) * ComplexShape.ε₁ c₂ c₃ c₂₃ (i₂, i₃)) •
        (F.obj (K₁.X i₁)).map ((G₂₃.map (K₂.d i₂ i₂')).app (K₃.X i₃)) ≫
          ιOrZero F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ i₁ _ i₃ j := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₂₃ : Type u_4
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝²² : CategoryTheory.Category.{u_15, u_1} C₁
    inst✝²¹ : CategoryTheory.Category.{u_17, u_2} C₂
    inst✝²⁰ : CategoryTheory.Category.{u_16, u_5} C₃
    inst✝¹⁹ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝¹⁸ : CategoryTheory.Category.{u_14, u_4} C₂₃
    inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹⁴ : CategoryTheory.Preadditive C₂₃
    inst✝¹³ : CategoryTheory.Preadditive C₄
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    inst✝¹² : G₂₃.PreservesZeroMorphisms
    inst✝¹¹ : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
    inst✝¹⁰ : F.PreservesZeroMorphisms
    inst✝⁹ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₂₃ : Type u_11
    ι₄ : Type u_12
    inst✝⁸ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₂₃ : ComplexShape ι₂₃
    c₄ : ComplexShape ι₄
    inst✝⁷ : TotalComplexShape c₁ c₂ c₁₂
    inst✝⁶ : TotalComplexShape c₁₂ c₃ c₄
    inst✝⁵ : TotalComplexShape c₂ c₃ c₂₃
    inst✝⁴ : TotalComplexShape c₁ c₂₃ c₄
    inst✝³ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
    inst✝² : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
    inst✝¹ : DecidableEq ι₂₃
    inst✝ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
    i₁ : ι₁
    i₂ i₂' : ι₂
    h₂ : c₂.Rel i₂ i₂'
    i₃ : ι₃
    j : ι₄
    ⊢ Eq (HomologicalComplex.mapBifunctor₂₃.d₂ F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ i₁ i₂ i₃  …
  -/
  obtain rfl := c₂.next_eq' h₂
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₂₃ : Type u_4
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝²² : CategoryTheory.Category.{u_15, u_1} C₁
    inst✝²¹ : CategoryTheory.Category.{u_17, u_2} C₂
    inst✝²⁰ : CategoryTheory.Category.{u_16, u_5} C₃
    inst✝¹⁹ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝¹⁸ : CategoryTheory.Category.{u_14, u_4} C₂₃
    inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹⁴ : CategoryTheory.Preadditive C₂₃
    inst✝¹³ : CategoryTheory.Preadditive C₄
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    inst✝¹² : G₂₃.PreservesZeroMorphisms
    inst✝¹¹ : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
    inst✝¹⁰ : F.PreservesZeroMorphisms
    inst✝⁹ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₂₃ : Type u_11
    ι₄ : Type u_12
    inst✝⁸ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₂₃ : ComplexShape ι₂₃
    c₄ : ComplexShape ι₄
    inst✝⁷ : TotalComplexShape c₁ c₂ c₁₂
    inst✝⁶ : TotalComplexShape c₁₂ c₃ c₄
    inst✝⁵ : TotalComplexShape c₂ c₃ c₂₃
    inst✝⁴ : TotalComplexShape c₁ c₂₃ c₄
    inst✝³ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
    inst✝² : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
    inst✝¹ : DecidableEq ι₂₃
    inst✝ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
    i₁ : ι₁
    i₂ : ι₂
    i₃ : ι₃
    j : ι₄
    h₂ : c₂.Rel i₂ (c₂.next i₂)
    ⊢ Eq (HomologicalComplex.mapBifunctor₂₃.d₂ F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ i₁ i₂ i₃  …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The third differential on a summand
of `mapBifunctor K₁ (mapBifunctor K₂ K₃ G₂₃ c₂₃) F c₄`. -/
noncomputable def d₃ (i₁ : ι₁) (i₂ : ι₂) (i₃ : ι₃) (j : ι₄) :
    (F.obj (K₁.X i₁)).obj ((G₂₃.obj (K₂.X i₂)).obj (K₃.X i₃)) ⟶
      (mapBifunctor K₁ (mapBifunctor K₂ K₃ G₂₃ c₂₃) F c₄).X j :=
  ((ComplexShape.ε₂ c₁ c₂₃ c₄ (i₁, ComplexShape.π c₂ c₃ c₂₃ (i₂, i₃)) *
      ComplexShape.ε₂ c₂ c₃ c₂₃ (i₂, i₃))) •
    (F.obj (K₁.X i₁)).map ((G₂₃.obj (K₂.X i₂)).map (K₃.d i₃ (c₃.next i₃))) ≫
      ιOrZero F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ i₁ i₂ _ j


lemma d₃_eq_zero (i₁ : ι₁) (i₂ : ι₂) (i₃ : ι₃) (j : ι₄) (h : ¬ c₃.Rel i₃ (c₃.next i₃)) :
    d₃ F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ i₁ i₂ i₃ j = 0 := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₂₃ : Type u_4
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝²² : CategoryTheory.Category.{u_15, u_1} C₁
    inst✝²¹ : CategoryTheory.Category.{u_17, u_2} C₂
    inst✝²⁰ : CategoryTheory.Category.{u_16, u_5} C₃
    inst✝¹⁹ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝¹⁸ : CategoryTheory.Category.{u_14, u_4} C₂₃
    inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹⁴ : CategoryTheory.Preadditive C₂₃
    inst✝¹³ : CategoryTheory.Preadditive C₄
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    inst✝¹² : G₂₃.PreservesZeroMorphisms
    inst✝¹¹ : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
    inst✝¹⁰ : F.PreservesZeroMorphisms
    inst✝⁹ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₂₃ : Type u_11
    ι₄ : Type u_12
    inst✝⁸ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₂₃ : ComplexShape ι₂₃
    c₄ : ComplexShape ι₄
    inst✝⁷ : TotalComplexShape c₁ c₂ c₁₂
    inst✝⁶ : TotalComplexShape c₁₂ c₃ c₄
    inst✝⁵ : TotalComplexShape c₂ c₃ c₂₃
    inst✝⁴ : TotalComplexShape c₁ c₂₃ c₄
    inst✝³ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
    inst✝² : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
    inst✝¹ : DecidableEq ι₂₃
    inst✝ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
    i₁ : ι₁
    i₂ : ι₂
    i₃ : ι₃
    j : ι₄
    h : Not (c₃.Rel i₃ (c₃.next i₃))
    ⊢ Eq (HomologicalComplex.mapBifunctor₂₃.d₃ F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ i₁ i₂ i₃  …
  -/
  dsimp [d₃]
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₂₃ : Type u_4
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝²² : CategoryTheory.Category.{u_15, u_1} C₁
    inst✝²¹ : CategoryTheory.Category.{u_17, u_2} C₂
    inst✝²⁰ : CategoryTheory.Category.{u_16, u_5} C₃
    inst✝¹⁹ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝¹⁸ : CategoryTheory.Category.{u_14, u_4} C₂₃
    inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹⁴ : CategoryTheory.Preadditive C₂₃
    inst✝¹³ : CategoryTheory.Preadditive C₄
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    inst✝¹² : G₂₃.PreservesZeroMorphisms
    inst✝¹¹ : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
    inst✝¹⁰ : F.PreservesZeroMorphisms
    inst✝⁹ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₂₃ : Type u_11
    ι₄ : Type u_12
    inst✝⁸ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₂₃ : ComplexShape ι₂₃
    c₄ : ComplexShape ι₄
    inst✝⁷ : TotalComplexShape c₁ c₂ c₁₂
    inst✝⁶ : TotalComplexShape c₁₂ c₃ c₄
    inst✝⁵ : TotalComplexShape c₂ c₃ c₂₃
    inst✝⁴ : TotalComplexShape c₁ c₂₃ c₄
    inst✝³ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
    inst✝² : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
    inst✝¹ : DecidableEq ι₂₃
    inst✝ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
    i₁ : ι₁
    i₂ : ι₂
    i₃ : ι₃
    j : ι₄
    h : Not (c₃.Rel i₃ (c₃.next i₃))
    ⊢ Eq (HSMul.hSMul (HMul.hMul (c₁.ε₂ c₂₃ c₄ { fst := i₁, snd := c₂.π c₃ c₂₃ { f …
  -/
  rw [shape _ _ _ h, Functor.map_zero, Functor.map_zero, zero_comp, smul_zero]
  /-
    🎉 no goals
  -/


lemma d₃_eq (i₁ : ι₁) (i₂ : ι₂) {i₃ i₃' : ι₃} (h₃ : c₃.Rel i₃ i₃') (j : ι₄) :
    d₃ F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ i₁ i₂ i₃ j =
      ((ComplexShape.ε₂ c₁ c₂₃ c₄ (i₁, ComplexShape.π c₂ c₃ c₂₃ (i₂, i₃)) *
          ComplexShape.ε₂ c₂ c₃ c₂₃ (i₂, i₃))) •
        (F.obj (K₁.X i₁)).map ((G₂₃.obj (K₂.X i₂)).map (K₃.d i₃ i₃')) ≫
        ιOrZero F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ i₁ i₂ _ j := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₂₃ : Type u_4
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝²² : CategoryTheory.Category.{u_15, u_1} C₁
    inst✝²¹ : CategoryTheory.Category.{u_17, u_2} C₂
    inst✝²⁰ : CategoryTheory.Category.{u_16, u_5} C₃
    inst✝¹⁹ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝¹⁸ : CategoryTheory.Category.{u_14, u_4} C₂₃
    inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹⁴ : CategoryTheory.Preadditive C₂₃
    inst✝¹³ : CategoryTheory.Preadditive C₄
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    inst✝¹² : G₂₃.PreservesZeroMorphisms
    inst✝¹¹ : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
    inst✝¹⁰ : F.PreservesZeroMorphisms
    inst✝⁹ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₂₃ : Type u_11
    ι₄ : Type u_12
    inst✝⁸ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₂₃ : ComplexShape ι₂₃
    c₄ : ComplexShape ι₄
    inst✝⁷ : TotalComplexShape c₁ c₂ c₁₂
    inst✝⁶ : TotalComplexShape c₁₂ c₃ c₄
    inst✝⁵ : TotalComplexShape c₂ c₃ c₂₃
    inst✝⁴ : TotalComplexShape c₁ c₂₃ c₄
    inst✝³ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
    inst✝² : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
    inst✝¹ : DecidableEq ι₂₃
    inst✝ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
    i₁ : ι₁
    i₂ : ι₂
    i₃ i₃' : ι₃
    h₃ : c₃.Rel i₃ i₃'
    j : ι₄
    ⊢ Eq (HomologicalComplex.mapBifunctor₂₃.d₃ F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ i₁ i₂ i₃  …
  -/
  obtain rfl := c₃.next_eq' h₃
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₂₃ : Type u_4
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝²² : CategoryTheory.Category.{u_15, u_1} C₁
    inst✝²¹ : CategoryTheory.Category.{u_17, u_2} C₂
    inst✝²⁰ : CategoryTheory.Category.{u_16, u_5} C₃
    inst✝¹⁹ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝¹⁸ : CategoryTheory.Category.{u_14, u_4} C₂₃
    inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹⁴ : CategoryTheory.Preadditive C₂₃
    inst✝¹³ : CategoryTheory.Preadditive C₄
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    inst✝¹² : G₂₃.PreservesZeroMorphisms
    inst✝¹¹ : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
    inst✝¹⁰ : F.PreservesZeroMorphisms
    inst✝⁹ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₂₃ : Type u_11
    ι₄ : Type u_12
    inst✝⁸ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₂₃ : ComplexShape ι₂₃
    c₄ : ComplexShape ι₄
    inst✝⁷ : TotalComplexShape c₁ c₂ c₁₂
    inst✝⁶ : TotalComplexShape c₁₂ c₃ c₄
    inst✝⁵ : TotalComplexShape c₂ c₃ c₂₃
    inst✝⁴ : TotalComplexShape c₁ c₂₃ c₄
    inst✝³ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
    inst✝² : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
    inst✝¹ : DecidableEq ι₂₃
    inst✝ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
    i₁ : ι₁
    i₂ : ι₂
    i₃ : ι₃
    j : ι₄
    h₃ : c₃.Rel i₃ (c₃.next i₃)
    ⊢ Eq (HomologicalComplex.mapBifunctor₂₃.d₃ F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ i₁ i₂ i₃  …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The first differential on `mapBifunctor K₁ (mapBifunctor K₂ K₃ G₂₃ c₂₃) F c₄`. -/
noncomputable def D₁ :
    (mapBifunctor K₁ (mapBifunctor K₂ K₃ G₂₃ c₂₃) F c₄).X j ⟶
      (mapBifunctor K₁ (mapBifunctor K₂ K₃ G₂₃ c₂₃) F c₄).X j' :=
  mapBifunctor.D₁ _ _ _ _ _ _


/-- The second differential on `mapBifunctor K₁ (mapBifunctor K₂ K₃ G₂₃ c₂₃) F c₄`. -/
noncomputable def D₂ :
    (mapBifunctor K₁ (mapBifunctor K₂ K₃ G₂₃ c₂₃) F c₄).X j ⟶
      (mapBifunctor K₁ (mapBifunctor K₂ K₃ G₂₃ c₂₃) F c₄).X j' :=
  mapBifunctor₂₃Desc c₁₂ (fun i₁ i₂ i₃ _ ↦ d₂ F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ i₁ i₂ i₃ j')


/-- The third differential on `mapBifunctor K₁ (mapBifunctor K₂ K₃ G₂₃ c₂₃) F c₄`. -/
noncomputable def D₃ :
    (mapBifunctor K₁ (mapBifunctor K₂ K₃ G₂₃ c₂₃) F c₄).X j ⟶
      (mapBifunctor K₁ (mapBifunctor K₂ K₃ G₂₃ c₂₃) F c₄).X j' :=
  mapBifunctor₂₃Desc c₁₂ (fun i₁ i₂ i₃ _ ↦ d₃ F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ i₁ i₂ i₃ j')


@[reassoc (attr := simp)]
lemma ι_D₁ :
    ι F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ i₁ i₂ i₃ j h ≫ D₁ F G₂₃ K₁ K₂ K₃ c₂₃ c₄ j j' =
      d₁ F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ i₁ i₂ i₃ j' := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₂₃ : Type u_4
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝²² : CategoryTheory.Category.{u_15, u_1} C₁
    inst✝²¹ : CategoryTheory.Category.{u_17, u_2} C₂
    inst✝²⁰ : CategoryTheory.Category.{u_16, u_5} C₃
    inst✝¹⁹ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝¹⁸ : CategoryTheory.Category.{u_14, u_4} C₂₃
    inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹⁴ : CategoryTheory.Preadditive C₂₃
    inst✝¹³ : CategoryTheory.Preadditive C₄
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    inst✝¹² : G₂₃.PreservesZeroMorphisms
    inst✝¹¹ : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
    inst✝¹⁰ : F.PreservesZeroMorphisms
    inst✝⁹ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₂₃ : Type u_11
    ι₄ : Type u_12
    inst✝⁸ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₂₃ : ComplexShape ι₂₃
    c₄ : ComplexShape ι₄
    inst✝⁷ : TotalComplexShape c₁ c₂ c₁₂
    inst✝⁶ : TotalComplexShape c₁₂ c₃ c₄
    inst✝⁵ : TotalComplexShape c₂ c₃ c₂₃
    inst✝⁴ : TotalComplexShape c₁ c₂₃ c₄
    inst✝³ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
    inst✝² : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
    inst✝¹ : DecidableEq ι₂₃
    inst✝ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
    i₁ : ι₁
    i₂ : ι₂
    i₃ : ι₃
    j j' : ι₄
    h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.mapBifunctor₂₃.ι  …
  -/
  dsimp only [D₁]
  rw [ι_eq _ _ _ _ _ _ _ _ _ _ _ _ _ rfl
      (by rw [← h, ← ComplexShape.assoc c₁ c₂ c₃ c₁₂ c₂₃ c₄]; rfl),
    assoc, mapBifunctor.ι_D₁]
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₂₃ : Type u_4
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝²² : CategoryTheory.Category.{u_15, u_1} C₁
    inst✝²¹ : CategoryTheory.Category.{u_17, u_2} C₂
    inst✝²⁰ : CategoryTheory.Category.{u_16, u_5} C₃
    inst✝¹⁹ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝¹⁸ : CategoryTheory.Category.{u_14, u_4} C₂₃
    inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹⁴ : CategoryTheory.Preadditive C₂₃
    inst✝¹³ : CategoryTheory.Preadditive C₄
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    inst✝¹² : G₂₃.PreservesZeroMorphisms
    inst✝¹¹ : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
    inst✝¹⁰ : F.PreservesZeroMorphisms
    inst✝⁹ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₂₃ : Type u_11
    ι₄ : Type u_12
    inst✝⁸ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₂₃ : ComplexShape ι₂₃
    c₄ : ComplexShape ι₄
    inst✝⁷ : TotalComplexShape c₁ c₂ c₁₂
    inst✝⁶ : TotalComplexShape c₁₂ c₃ c₄
    inst✝⁵ : TotalComplexShape c₂ c₃ c₂₃
    inst✝⁴ : TotalComplexShape c₁ c₂₃ c₄
    inst✝³ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
    inst✝² : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
    inst✝¹ : DecidableEq ι₂₃
    inst✝ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
    i₁ : ι₁
    i₂ : ι₂
    i₃ : ι₃
    j j' : ι₄
    h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.obj (K₁.X i₁)).map (K₂.ιMapBifunc …
  -/
  by_cases h₁ : c₁.Rel i₁ (c₁.next i₁)
    /-
      case pos
      C₁ : Type u_1
      C₂ : Type u_2
      C₂₃ : Type u_4
      C₃ : Type u_5
      C₄ : Type u_6
      inst✝²² : CategoryTheory.Category.{u_15, u_1} C₁
      inst✝²¹ : CategoryTheory.Category.{u_17, u_2} C₂
      inst✝²⁰ : CategoryTheory.Category.{u_16, u_5} C₃
      inst✝¹⁹ : CategoryTheory.Category.{u_13, u_6} C₄
      inst✝¹⁸ : CategoryTheory.Category.{u_14, u_4} C₂₃
      inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₁
      inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₂
      inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₃
      inst✝¹⁴ : CategoryTheory.Preadditive C₂₃
      inst✝¹³ : CategoryTheory.Preadditive C₄
      F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
      G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
      inst✝¹² : G₂₃.PreservesZeroMorphisms
      inst✝¹¹ : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
      inst✝¹⁰ : F.PreservesZeroMorphisms
      inst✝⁹ : ∀ (X₁ : C₁), (F.obj X₁).Additive
      ι₁ : Type u_7
      ι₂ : Type u_8
      ι₃ : Type u_9
      ι₁₂ : Type u_10
      ι₂₃ : Type u_11
      ι₄ : Type u_12
      inst✝⁸ : DecidableEq ι₄
      c₁ : ComplexShape ι₁
      c₂ : ComplexShape ι₂
      c₃ : ComplexShape ι₃
      K₁ : HomologicalComplex C₁ c₁
      K₂ : HomologicalComplex C₂ c₂
      K₃ : HomologicalComplex C₃ c₃
      c₁₂ : ComplexShape ι₁₂
      c₂₃ : ComplexShape ι₂₃
      c₄ : ComplexShape ι₄
      inst✝⁷ : TotalComplexShape c₁ c₂ c₁₂
      inst✝⁶ : TotalComplexShape c₁₂ c₃ c₄
      inst✝⁵ : TotalComplexShape c₂ c₃ c₂₃
      inst✝⁴ : TotalComplexShape c₁ c₂₃ c₄
      inst✝³ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
      inst✝² : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
      inst✝¹ : DecidableEq ι₂₃
      inst✝ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
      i₁ : ι₁
      i₂ : ι₂
      i₃ : ι₃
      j j' : ι₄
      h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
      h₁ : c₁.Rel i₁ (c₁.next i₁)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.obj (K₁.X i₁)).map (K₂.ιMapBifunc …
    -/
  · rw [d₁_eq _ _ _ _ _ _ _ _ h₁]
    /-
      case pos
      C₁ : Type u_1
      C₂ : Type u_2
      C₂₃ : Type u_4
      C₃ : Type u_5
      C₄ : Type u_6
      inst✝²² : CategoryTheory.Category.{u_15, u_1} C₁
      inst✝²¹ : CategoryTheory.Category.{u_17, u_2} C₂
      inst✝²⁰ : CategoryTheory.Category.{u_16, u_5} C₃
      inst✝¹⁹ : CategoryTheory.Category.{u_13, u_6} C₄
      inst✝¹⁸ : CategoryTheory.Category.{u_14, u_4} C₂₃
      inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₁
      inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₂
      inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₃
      inst✝¹⁴ : CategoryTheory.Preadditive C₂₃
      inst✝¹³ : CategoryTheory.Preadditive C₄
      F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
      G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
      inst✝¹² : G₂₃.PreservesZeroMorphisms
      inst✝¹¹ : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
      inst✝¹⁰ : F.PreservesZeroMorphisms
      inst✝⁹ : ∀ (X₁ : C₁), (F.obj X₁).Additive
      ι₁ : Type u_7
      ι₂ : Type u_8
      ι₃ : Type u_9
      ι₁₂ : Type u_10
      ι₂₃ : Type u_11
      ι₄ : Type u_12
      inst✝⁸ : DecidableEq ι₄
      c₁ : ComplexShape ι₁
      c₂ : ComplexShape ι₂
      c₃ : ComplexShape ι₃
      K₁ : HomologicalComplex C₁ c₁
      K₂ : HomologicalComplex C₂ c₂
      K₃ : HomologicalComplex C₃ c₃
      c₁₂ : ComplexShape ι₁₂
      c₂₃ : ComplexShape ι₂₃
      c₄ : ComplexShape ι₄
      inst✝⁷ : TotalComplexShape c₁ c₂ c₁₂
      inst✝⁶ : TotalComplexShape c₁₂ c₃ c₄
      inst✝⁵ : TotalComplexShape c₂ c₃ c₂₃
      inst✝⁴ : TotalComplexShape c₁ c₂₃ c₄
      inst✝³ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
      inst✝² : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
      inst✝¹ : DecidableEq ι₂₃
      inst✝ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
      i₁ : ι₁
      i₂ : ι₂
      i₃ : ι₃
      j j' : ι₄
      h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
      h₁ : c₁.Rel i₁ (c₁.next i₁)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.obj (K₁.X i₁)).map (K₂.ιMapBifunc …
    -/
    by_cases h₂ : ComplexShape.π c₁ c₂₃ c₄ (c₁.next i₁, ComplexShape.π c₂ c₃ c₂₃ (i₂, i₃)) = j'
    · rw [mapBifunctor.d₁_eq _ _ _ _ h₁ _ _ h₂, ιOrZero_eq,
        Linear.comp_units_smul, NatTrans.naturality_assoc]
        /-
          case pos
          C₁ : Type u_1
          C₂ : Type u_2
          C₂₃ : Type u_4
          C₃ : Type u_5
          C₄ : Type u_6
          inst✝²² : CategoryTheory.Category.{u_15, u_1} C₁
          inst✝²¹ : CategoryTheory.Category.{u_17, u_2} C₂
          inst✝²⁰ : CategoryTheory.Category.{u_16, u_5} C₃
          inst✝¹⁹ : CategoryTheory.Category.{u_13, u_6} C₄
          inst✝¹⁸ : CategoryTheory.Category.{u_14, u_4} C₂₃
          inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₁
          inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₂
          inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₃
          inst✝¹⁴ : CategoryTheory.Preadditive C₂₃
          inst✝¹³ : CategoryTheory.Preadditive C₄
          F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
          G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
          inst✝¹² : G₂₃.PreservesZeroMorphisms
          inst✝¹¹ : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
          inst✝¹⁰ : F.PreservesZeroMorphisms
          inst✝⁹ : ∀ (X₁ : C₁), (F.obj X₁).Additive
          ι₁ : Type u_7
          ι₂ : Type u_8
          ι₃ : Type u_9
          ι₁₂ : Type u_10
          ι₂₃ : Type u_11
          ι₄ : Type u_12
          inst✝⁸ : DecidableEq ι₄
          c₁ : ComplexShape ι₁
          c₂ : ComplexShape ι₂
          c₃ : ComplexShape ι₃
          K₁ : HomologicalComplex C₁ c₁
          K₂ : HomologicalComplex C₂ c₂
          K₃ : HomologicalComplex C₃ c₃
          c₁₂ : ComplexShape ι₁₂
          c₂₃ : ComplexShape ι₂₃
          c₄ : ComplexShape ι₄
          inst✝⁷ : TotalComplexShape c₁ c₂ c₁₂
          inst✝⁶ : TotalComplexShape c₁₂ c₃ c₄
          inst✝⁵ : TotalComplexShape c₂ c₃ c₂₃
          inst✝⁴ : TotalComplexShape c₁ c₂₃ c₄
          inst✝³ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
          inst✝² : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
          inst✝¹ : DecidableEq ι₂₃
          inst✝ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
          i₁ : ι₁
          i₂ : ι₂
          i₃ : ι₃
          j j' : ι₄
          h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
          h₁ : c₁.Rel i₁ (c₁.next i₁)
          h₂ : Eq (c₁.π c₂₃ c₄ { fst := c₁.next i₁, snd := c₂.π c₃ c₂₃ { fst := i₂, snd  …
          ⊢ Eq (HSMul.hSMul (c₁.ε₁ c₂₃ c₄ { fst := i₁, snd := c₂.π c₃ c₂₃ { fst := i₂, s …
        -/
      · rfl
        /-
          🎉 no goals
        -/
        /-
          case pos.h
          C₁ : Type u_1
          C₂ : Type u_2
          C₂₃ : Type u_4
          C₃ : Type u_5
          C₄ : Type u_6
          inst✝²² : CategoryTheory.Category.{u_15, u_1} C₁
          inst✝²¹ : CategoryTheory.Category.{u_17, u_2} C₂
          inst✝²⁰ : CategoryTheory.Category.{u_16, u_5} C₃
          inst✝¹⁹ : CategoryTheory.Category.{u_13, u_6} C₄
          inst✝¹⁸ : CategoryTheory.Category.{u_14, u_4} C₂₃
          inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₁
          inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₂
          inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₃
          inst✝¹⁴ : CategoryTheory.Preadditive C₂₃
          inst✝¹³ : CategoryTheory.Preadditive C₄
          F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
          G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
          inst✝¹² : G₂₃.PreservesZeroMorphisms
          inst✝¹¹ : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
          inst✝¹⁰ : F.PreservesZeroMorphisms
          inst✝⁹ : ∀ (X₁ : C₁), (F.obj X₁).Additive
          ι₁ : Type u_7
          ι₂ : Type u_8
          ι₃ : Type u_9
          ι₁₂ : Type u_10
          ι₂₃ : Type u_11
          ι₄ : Type u_12
          inst✝⁸ : DecidableEq ι₄
          c₁ : ComplexShape ι₁
          c₂ : ComplexShape ι₂
          c₃ : ComplexShape ι₃
          K₁ : HomologicalComplex C₁ c₁
          K₂ : HomologicalComplex C₂ c₂
          K₃ : HomologicalComplex C₃ c₃
          c₁₂ : ComplexShape ι₁₂
          c₂₃ : ComplexShape ι₂₃
          c₄ : ComplexShape ι₄
          inst✝⁷ : TotalComplexShape c₁ c₂ c₁₂
          inst✝⁶ : TotalComplexShape c₁₂ c₃ c₄
          inst✝⁵ : TotalComplexShape c₂ c₃ c₂₃
          inst✝⁴ : TotalComplexShape c₁ c₂₃ c₄
          inst✝³ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
          inst✝² : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
          inst✝¹ : DecidableEq ι₂₃
          inst✝ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
          i₁ : ι₁
          i₂ : ι₂
          i₃ : ι₃
          j j' : ι₄
          h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
          h₁ : c₁.Rel i₁ (c₁.next i₁)
          h₂ : Eq (c₁.π c₂₃ c₄ { fst := c₁.next i₁, snd := c₂.π c₃ c₂₃ { fst := i₂, snd  …
          ⊢ Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := c₁.next i₁, snd := { fst := i₂, snd := i₃ } } …
        -/
      · rw [← h₂, ← ComplexShape.assoc c₁ c₂ c₃ c₁₂ c₂₃ c₄]
        /-
          case pos.h
          C₁ : Type u_1
          C₂ : Type u_2
          C₂₃ : Type u_4
          C₃ : Type u_5
          C₄ : Type u_6
          inst✝²² : CategoryTheory.Category.{u_15, u_1} C₁
          inst✝²¹ : CategoryTheory.Category.{u_17, u_2} C₂
          inst✝²⁰ : CategoryTheory.Category.{u_16, u_5} C₃
          inst✝¹⁹ : CategoryTheory.Category.{u_13, u_6} C₄
          inst✝¹⁸ : CategoryTheory.Category.{u_14, u_4} C₂₃
          inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₁
          inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₂
          inst✝¹⁵ : CategoryTheory.Limits.HasZeroMorphisms C₃
          inst✝¹⁴ : CategoryTheory.Preadditive C₂₃
          inst✝¹³ : CategoryTheory.Preadditive C₄
          F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
          G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
          inst✝¹² : G₂₃.PreservesZeroMorphisms
          inst✝¹¹ : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
          inst✝¹⁰ : F.PreservesZeroMorphisms
          inst✝⁹ : ∀ (X₁ : C₁), (F.obj X₁).Additive
          ι₁ : Type u_7
          ι₂ : Type u_8
          ι₃ : Type u_9
          ι₁₂ : Type u_10
          ι₂₃ : Type u_11
          ι₄ : Type u_12
          inst✝⁸ : DecidableEq ι₄
          c₁ : ComplexShape ι₁
          c₂ : ComplexShape ι₂
          c₃ : ComplexShape ι₃
          K₁ : HomologicalComplex C₁ c₁
          K₂ : HomologicalComplex C₂ c₂
          K₃ : HomologicalComplex C₃ c₃
          c₁₂ : ComplexShape ι₁₂
          c₂₃ : ComplexShape ι₂₃
          c₄ : ComplexShape ι₄
          inst✝⁷ : TotalComplexShape c₁ c₂ c₁₂
          inst✝⁶ : TotalComplexShape c₁₂ c₃ c₄
          inst✝⁵ : TotalComplexShape c₂ c₃ c₂₃
          inst✝⁴ : TotalComplexShape c₁ c₂₃ c₄
          inst✝³ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
          inst✝² : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
          inst✝¹ : DecidableEq ι₂₃
          inst✝ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
          i₁ : ι₁
          i₂ : ι₂
          i₃ : ι₃
          j j' : ι₄
          h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
          h₁ : c₁.Rel i₁ (c₁.next i₁)
          h₂ : Eq (c₁.π c₂₃ c₄ { fst := c₁.next i₁, snd := c₂.π c₃ c₂₃ { fst := i₂, snd  …
          ⊢ Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := c₁.next i₁, snd := { fst := i₂, snd := i₃ } } …
        -/
        rfl
        /-
          🎉 no goals
        -/
    · rw [mapBifunctor.d₁_eq_zero' _ _ _ _ h₁ _ _ h₂, comp_zero,
        ιOrZero_eq_zero _ _ _ _ _ _ _ _ _ _ _ _
          (by simpa only [← ComplexShape.assoc c₁ c₂ c₃ c₁₂ c₂₃ c₄] using h₂),
        comp_zero, smul_zero]
  · rw [mapBifunctor.d₁_eq_zero _ _ _ _ _ _ _ h₁,
      d₁_eq_zero _ _ _ _ _ _ _ _ _ _ _ _ h₁, comp_zero]


@[reassoc (attr := simp)]
lemma ι_D₂ :
    ι F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ i₁ i₂ i₃ j h ≫ D₂ F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ j j' =
      d₂ F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ i₁ i₂ i₃ j' := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₂₃ : Type u_4
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝²³ : CategoryTheory.Category.{u_15, u_1} C₁
    inst✝²² : CategoryTheory.Category.{u_17, u_2} C₂
    inst✝²¹ : CategoryTheory.Category.{u_16, u_5} C₃
    inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝¹⁹ : CategoryTheory.Category.{u_14, u_4} C₂₃
    inst✝¹⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹⁵ : CategoryTheory.Preadditive C₂₃
    inst✝¹⁴ : CategoryTheory.Preadditive C₄
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    inst✝¹³ : G₂₃.PreservesZeroMorphisms
    inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
    inst✝¹¹ : F.PreservesZeroMorphisms
    inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₂₃ : Type u_11
    ι₄ : Type u_12
    inst✝⁹ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₂₃ : ComplexShape ι₂₃
    c₄ : ComplexShape ι₄
    inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
    inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
    inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
    inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
    inst✝⁴ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
    inst✝³ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
    inst✝² : DecidableEq ι₂₃
    inst✝¹ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
    i₁ : ι₁
    i₂ : ι₂
    i₃ : ι₃
    j j' : ι₄
    h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
    inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.mapBifunctor₂₃.ι  …
  -/
  simp [D₂]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma ι_D₃ :
    ι F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ i₁ i₂ i₃ j h ≫ D₃ F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ j j' =
      d₃ F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ i₁ i₂ i₃ j' := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₂₃ : Type u_4
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝²³ : CategoryTheory.Category.{u_15, u_1} C₁
    inst✝²² : CategoryTheory.Category.{u_17, u_2} C₂
    inst✝²¹ : CategoryTheory.Category.{u_16, u_5} C₃
    inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝¹⁹ : CategoryTheory.Category.{u_14, u_4} C₂₃
    inst✝¹⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹⁵ : CategoryTheory.Preadditive C₂₃
    inst✝¹⁴ : CategoryTheory.Preadditive C₄
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    inst✝¹³ : G₂₃.PreservesZeroMorphisms
    inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
    inst✝¹¹ : F.PreservesZeroMorphisms
    inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₂₃ : Type u_11
    ι₄ : Type u_12
    inst✝⁹ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₂₃ : ComplexShape ι₂₃
    c₄ : ComplexShape ι₄
    inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
    inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
    inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
    inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
    inst✝⁴ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
    inst✝³ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
    inst✝² : DecidableEq ι₂₃
    inst✝¹ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
    i₁ : ι₁
    i₂ : ι₂
    i₃ : ι₃
    j j' : ι₄
    h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
    inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.mapBifunctor₂₃.ι  …
  -/
  simp [D₃]
  /-
    🎉 no goals
  -/


lemma d_eq :
    (mapBifunctor K₁ (mapBifunctor K₂ K₃ G₂₃ c₂₃) F c₄).d j j' =
      D₁ F G₂₃ K₁ K₂ K₃ c₂₃ c₄ j j' + D₂ F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ j j' +
      D₃ F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ j j' := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₂₃ : Type u_4
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝²³ : CategoryTheory.Category.{u_14, u_1} C₁
    inst✝²² : CategoryTheory.Category.{u_16, u_2} C₂
    inst✝²¹ : CategoryTheory.Category.{u_17, u_5} C₃
    inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝¹⁹ : CategoryTheory.Category.{u_15, u_4} C₂₃
    inst✝¹⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹⁵ : CategoryTheory.Preadditive C₂₃
    inst✝¹⁴ : CategoryTheory.Preadditive C₄
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    inst✝¹³ : G₂₃.PreservesZeroMorphisms
    inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
    inst✝¹¹ : F.PreservesZeroMorphisms
    inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₂₃ : Type u_11
    ι₄ : Type u_12
    inst✝⁹ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₂₃ : ComplexShape ι₂₃
    c₄ : ComplexShape ι₄
    inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
    inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
    inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
    inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
    inst✝⁴ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
    inst✝³ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
    inst✝² : DecidableEq ι₂₃
    inst✝¹ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
    inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
    j j' : ι₄
    ⊢ Eq ((K₁.mapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄).d j j') (HAdd.hAdd ( …
  -/
  rw [mapBifunctor.d_eq]
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₂₃ : Type u_4
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝²³ : CategoryTheory.Category.{u_14, u_1} C₁
    inst✝²² : CategoryTheory.Category.{u_16, u_2} C₂
    inst✝²¹ : CategoryTheory.Category.{u_17, u_5} C₃
    inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝¹⁹ : CategoryTheory.Category.{u_15, u_4} C₂₃
    inst✝¹⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹⁵ : CategoryTheory.Preadditive C₂₃
    inst✝¹⁴ : CategoryTheory.Preadditive C₄
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    inst✝¹³ : G₂₃.PreservesZeroMorphisms
    inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
    inst✝¹¹ : F.PreservesZeroMorphisms
    inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₂₃ : Type u_11
    ι₄ : Type u_12
    inst✝⁹ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₂₃ : ComplexShape ι₂₃
    c₄ : ComplexShape ι₄
    inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
    inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
    inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
    inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
    inst✝⁴ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
    inst✝³ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
    inst✝² : DecidableEq ι₂₃
    inst✝¹ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
    inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
    j j' : ι₄
    ⊢ Eq (HAdd.hAdd (HomologicalComplex.mapBifunctor.D₁ K₁ (K₂.mapBifunctor K₃ G₂₃ …
  -/
  rw [add_assoc]
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₂₃ : Type u_4
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝²³ : CategoryTheory.Category.{u_14, u_1} C₁
    inst✝²² : CategoryTheory.Category.{u_16, u_2} C₂
    inst✝²¹ : CategoryTheory.Category.{u_17, u_5} C₃
    inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝¹⁹ : CategoryTheory.Category.{u_15, u_4} C₂₃
    inst✝¹⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹⁵ : CategoryTheory.Preadditive C₂₃
    inst✝¹⁴ : CategoryTheory.Preadditive C₄
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    inst✝¹³ : G₂₃.PreservesZeroMorphisms
    inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
    inst✝¹¹ : F.PreservesZeroMorphisms
    inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₂₃ : Type u_11
    ι₄ : Type u_12
    inst✝⁹ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₂₃ : ComplexShape ι₂₃
    c₄ : ComplexShape ι₄
    inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
    inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
    inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
    inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
    inst✝⁴ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
    inst✝³ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
    inst✝² : DecidableEq ι₂₃
    inst✝¹ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
    inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
    j j' : ι₄
    ⊢ Eq (HAdd.hAdd (HomologicalComplex.mapBifunctor.D₁ K₁ (K₂.mapBifunctor K₃ G₂₃ …
  -/
  congr 1
  /-
    case e_a
    C₁ : Type u_1
    C₂ : Type u_2
    C₂₃ : Type u_4
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝²³ : CategoryTheory.Category.{u_14, u_1} C₁
    inst✝²² : CategoryTheory.Category.{u_16, u_2} C₂
    inst✝²¹ : CategoryTheory.Category.{u_17, u_5} C₃
    inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝¹⁹ : CategoryTheory.Category.{u_15, u_4} C₂₃
    inst✝¹⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹⁵ : CategoryTheory.Preadditive C₂₃
    inst✝¹⁴ : CategoryTheory.Preadditive C₄
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    inst✝¹³ : G₂₃.PreservesZeroMorphisms
    inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
    inst✝¹¹ : F.PreservesZeroMorphisms
    inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₂₃ : Type u_11
    ι₄ : Type u_12
    inst✝⁹ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₂₃ : ComplexShape ι₂₃
    c₄ : ComplexShape ι₄
    inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
    inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
    inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
    inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
    inst✝⁴ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
    inst✝³ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
    inst✝² : DecidableEq ι₂₃
    inst✝¹ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
    inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
    j j' : ι₄
    ⊢ Eq (HomologicalComplex.mapBifunctor.D₂ K₁ (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄  …
  -/
  apply mapBifunctor₂₃.hom_ext (c₁₂ := c₁₂)
  /-
    case e_a
    C₁ : Type u_1
    C₂ : Type u_2
    C₂₃ : Type u_4
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝²³ : CategoryTheory.Category.{u_14, u_1} C₁
    inst✝²² : CategoryTheory.Category.{u_16, u_2} C₂
    inst✝²¹ : CategoryTheory.Category.{u_17, u_5} C₃
    inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝¹⁹ : CategoryTheory.Category.{u_15, u_4} C₂₃
    inst✝¹⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹⁵ : CategoryTheory.Preadditive C₂₃
    inst✝¹⁴ : CategoryTheory.Preadditive C₄
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    inst✝¹³ : G₂₃.PreservesZeroMorphisms
    inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
    inst✝¹¹ : F.PreservesZeroMorphisms
    inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₂₃ : Type u_11
    ι₄ : Type u_12
    inst✝⁹ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₂₃ : ComplexShape ι₂₃
    c₄ : ComplexShape ι₄
    inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
    inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
    inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
    inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
    inst✝⁴ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
    inst✝³ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
    inst✝² : DecidableEq ι₂₃
    inst✝¹ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
    inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
    j j' : ι₄
    ⊢ ∀ (i₁ : ι₁) (i₂ : ι₂) (i₃ : ι₃) (h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd  …
  -/
  intros i₁ i₂ i₃ h
  /-
    case e_a
    C₁ : Type u_1
    C₂ : Type u_2
    C₂₃ : Type u_4
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝²³ : CategoryTheory.Category.{u_14, u_1} C₁
    inst✝²² : CategoryTheory.Category.{u_16, u_2} C₂
    inst✝²¹ : CategoryTheory.Category.{u_17, u_5} C₃
    inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝¹⁹ : CategoryTheory.Category.{u_15, u_4} C₂₃
    inst✝¹⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹⁵ : CategoryTheory.Preadditive C₂₃
    inst✝¹⁴ : CategoryTheory.Preadditive C₄
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    inst✝¹³ : G₂₃.PreservesZeroMorphisms
    inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
    inst✝¹¹ : F.PreservesZeroMorphisms
    inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₂₃ : Type u_11
    ι₄ : Type u_12
    inst✝⁹ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₂₃ : ComplexShape ι₂₃
    c₄ : ComplexShape ι₄
    inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
    inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
    inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
    inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
    inst✝⁴ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
    inst✝³ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
    inst✝² : DecidableEq ι₂₃
    inst✝¹ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
    inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
    j j' : ι₄
    i₁ : ι₁
    i₂ : ι₂
    i₃ : ι₃
    h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.mapBifunctor₂₃.ι  …
  -/
  simp only [Preadditive.comp_add, ι_D₂, ι_D₃]
  rw [ι_eq _ _ _ _ _ _ _ _ _ _ _ _ _ rfl
      (by rw [← h, ← ComplexShape.assoc c₁ c₂ c₃ c₁₂ c₂₃ c₄]; rfl),
    assoc, mapBifunctor.ι_D₂]
  /-
    case e_a
    C₁ : Type u_1
    C₂ : Type u_2
    C₂₃ : Type u_4
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝²³ : CategoryTheory.Category.{u_14, u_1} C₁
    inst✝²² : CategoryTheory.Category.{u_16, u_2} C₂
    inst✝²¹ : CategoryTheory.Category.{u_17, u_5} C₃
    inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝¹⁹ : CategoryTheory.Category.{u_15, u_4} C₂₃
    inst✝¹⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹⁵ : CategoryTheory.Preadditive C₂₃
    inst✝¹⁴ : CategoryTheory.Preadditive C₄
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    inst✝¹³ : G₂₃.PreservesZeroMorphisms
    inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
    inst✝¹¹ : F.PreservesZeroMorphisms
    inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₂₃ : Type u_11
    ι₄ : Type u_12
    inst✝⁹ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₂₃ : ComplexShape ι₂₃
    c₄ : ComplexShape ι₄
    inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
    inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
    inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
    inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
    inst✝⁴ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
    inst✝³ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
    inst✝² : DecidableEq ι₂₃
    inst✝¹ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
    inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
    j j' : ι₄
    i₁ : ι₁
    i₂ : ι₂
    i₃ : ι₃
    h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.obj (K₁.X i₁)).map (K₂.ιMapBifunc …
  -/
  set i₂₃ := ComplexShape.π c₂ c₃ c₂₃ ⟨i₂, i₃⟩
  /-
    case e_a
    C₁ : Type u_1
    C₂ : Type u_2
    C₂₃ : Type u_4
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝²³ : CategoryTheory.Category.{u_14, u_1} C₁
    inst✝²² : CategoryTheory.Category.{u_16, u_2} C₂
    inst✝²¹ : CategoryTheory.Category.{u_17, u_5} C₃
    inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝¹⁹ : CategoryTheory.Category.{u_15, u_4} C₂₃
    inst✝¹⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝¹⁵ : CategoryTheory.Preadditive C₂₃
    inst✝¹⁴ : CategoryTheory.Preadditive C₄
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    inst✝¹³ : G₂₃.PreservesZeroMorphisms
    inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
    inst✝¹¹ : F.PreservesZeroMorphisms
    inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₂₃ : Type u_11
    ι₄ : Type u_12
    inst✝⁹ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₂₃ : ComplexShape ι₂₃
    c₄ : ComplexShape ι₄
    inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
    inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
    inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
    inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
    inst✝⁴ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
    inst✝³ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
    inst✝² : DecidableEq ι₂₃
    inst✝¹ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
    inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
    j j' : ι₄
    i₁ : ι₁
    i₂ : ι₂
    i₃ : ι₃
    h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
    i₂₃ : ι₂₃ := c₂.π c₃ c₂₃ { fst := i₂, snd := i₃ }
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.obj (K₁.X i₁)).map (K₂.ιMapBifunc …
  -/
  by_cases h₁ : c₂₃.Rel i₂₃ (c₂₃.next i₂₃)
    /-
      case pos
      C₁ : Type u_1
      C₂ : Type u_2
      C₂₃ : Type u_4
      C₃ : Type u_5
      C₄ : Type u_6
      inst✝²³ : CategoryTheory.Category.{u_14, u_1} C₁
      inst✝²² : CategoryTheory.Category.{u_16, u_2} C₂
      inst✝²¹ : CategoryTheory.Category.{u_17, u_5} C₃
      inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} C₄
      inst✝¹⁹ : CategoryTheory.Category.{u_15, u_4} C₂₃
      inst✝¹⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
      inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
      inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₃
      inst✝¹⁵ : CategoryTheory.Preadditive C₂₃
      inst✝¹⁴ : CategoryTheory.Preadditive C₄
      F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
      G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
      inst✝¹³ : G₂₃.PreservesZeroMorphisms
      inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
      inst✝¹¹ : F.PreservesZeroMorphisms
      inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
      ι₁ : Type u_7
      ι₂ : Type u_8
      ι₃ : Type u_9
      ι₁₂ : Type u_10
      ι₂₃ : Type u_11
      ι₄ : Type u_12
      inst✝⁹ : DecidableEq ι₄
      c₁ : ComplexShape ι₁
      c₂ : ComplexShape ι₂
      c₃ : ComplexShape ι₃
      K₁ : HomologicalComplex C₁ c₁
      K₂ : HomologicalComplex C₂ c₂
      K₃ : HomologicalComplex C₃ c₃
      c₁₂ : ComplexShape ι₁₂
      c₂₃ : ComplexShape ι₂₃
      c₄ : ComplexShape ι₄
      inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
      inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
      inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
      inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
      inst✝⁴ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
      inst✝³ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
      inst✝² : DecidableEq ι₂₃
      inst✝¹ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
      inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
      j j' : ι₄
      i₁ : ι₁
      i₂ : ι₂
      i₃ : ι₃
      h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
      i₂₃ : ι₂₃ := c₂.π c₃ c₂₃ { fst := i₂, snd := i₃ }
      h₁ : c₂₃.Rel i₂₃ (c₂₃.next i₂₃)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.obj (K₁.X i₁)).map (K₂.ιMapBifunc …
    -/
  · by_cases h₂ : ComplexShape.π c₁ c₂₃ c₄ (i₁, c₂₃.next i₂₃) = j'
    · rw [mapBifunctor.d₂_eq _ _ _ _ _ h₁ _ h₂, mapBifunctor.d_eq,
        Linear.comp_units_smul, Functor.map_add, Preadditive.add_comp,
        Preadditive.comp_add, smul_add]
      /-
        case pos
        C₁ : Type u_1
        C₂ : Type u_2
        C₂₃ : Type u_4
        C₃ : Type u_5
        C₄ : Type u_6
        inst✝²³ : CategoryTheory.Category.{u_14, u_1} C₁
        inst✝²² : CategoryTheory.Category.{u_16, u_2} C₂
        inst✝²¹ : CategoryTheory.Category.{u_17, u_5} C₃
        inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} C₄
        inst✝¹⁹ : CategoryTheory.Category.{u_15, u_4} C₂₃
        inst✝¹⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
        inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
        inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₃
        inst✝¹⁵ : CategoryTheory.Preadditive C₂₃
        inst✝¹⁴ : CategoryTheory.Preadditive C₄
        F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
        G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
        inst✝¹³ : G₂₃.PreservesZeroMorphisms
        inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
        inst✝¹¹ : F.PreservesZeroMorphisms
        inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
        ι₁ : Type u_7
        ι₂ : Type u_8
        ι₃ : Type u_9
        ι₁₂ : Type u_10
        ι₂₃ : Type u_11
        ι₄ : Type u_12
        inst✝⁹ : DecidableEq ι₄
        c₁ : ComplexShape ι₁
        c₂ : ComplexShape ι₂
        c₃ : ComplexShape ι₃
        K₁ : HomologicalComplex C₁ c₁
        K₂ : HomologicalComplex C₂ c₂
        K₃ : HomologicalComplex C₃ c₃
        c₁₂ : ComplexShape ι₁₂
        c₂₃ : ComplexShape ι₂₃
        c₄ : ComplexShape ι₄
        inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
        inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
        inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
        inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
        inst✝⁴ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
        inst✝³ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
        inst✝² : DecidableEq ι₂₃
        inst✝¹ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
        inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
        j j' : ι₄
        i₁ : ι₁
        i₂ : ι₂
        i₃ : ι₃
        h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
        i₂₃ : ι₂₃ := c₂.π c₃ c₂₃ { fst := i₂, snd := i₃ }
        h₁ : c₂₃.Rel i₂₃ (c₂₃.next i₂₃)
        h₂ : Eq (c₁.π c₂₃ c₄ { fst := i₁, snd := c₂₃.next i₂₃ }) j'
        ⊢ Eq (HAdd.hAdd (HSMul.hSMul (c₁.ε₂ c₂₃ c₄ { fst := i₁, snd := i₂₃ }) (Categor …
      -/
      congr 1
        /-
          case pos.e_a
          C₁ : Type u_1
          C₂ : Type u_2
          C₂₃ : Type u_4
          C₃ : Type u_5
          C₄ : Type u_6
          inst✝²³ : CategoryTheory.Category.{u_14, u_1} C₁
          inst✝²² : CategoryTheory.Category.{u_16, u_2} C₂
          inst✝²¹ : CategoryTheory.Category.{u_17, u_5} C₃
          inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} C₄
          inst✝¹⁹ : CategoryTheory.Category.{u_15, u_4} C₂₃
          inst✝¹⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
          inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
          inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₃
          inst✝¹⁵ : CategoryTheory.Preadditive C₂₃
          inst✝¹⁴ : CategoryTheory.Preadditive C₄
          F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
          G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
          inst✝¹³ : G₂₃.PreservesZeroMorphisms
          inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
          inst✝¹¹ : F.PreservesZeroMorphisms
          inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
          ι₁ : Type u_7
          ι₂ : Type u_8
          ι₃ : Type u_9
          ι₁₂ : Type u_10
          ι₂₃ : Type u_11
          ι₄ : Type u_12
          inst✝⁹ : DecidableEq ι₄
          c₁ : ComplexShape ι₁
          c₂ : ComplexShape ι₂
          c₃ : ComplexShape ι₃
          K₁ : HomologicalComplex C₁ c₁
          K₂ : HomologicalComplex C₂ c₂
          K₃ : HomologicalComplex C₃ c₃
          c₁₂ : ComplexShape ι₁₂
          c₂₃ : ComplexShape ι₂₃
          c₄ : ComplexShape ι₄
          inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
          inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
          inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
          inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
          inst✝⁴ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
          inst✝³ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
          inst✝² : DecidableEq ι₂₃
          inst✝¹ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
          inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
          j j' : ι₄
          i₁ : ι₁
          i₂ : ι₂
          i₃ : ι₃
          h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
          i₂₃ : ι₂₃ := c₂.π c₃ c₂₃ { fst := i₂, snd := i₃ }
          h₁ : c₂₃.Rel i₂₃ (c₂₃.next i₂₃)
          h₂ : Eq (c₁.π c₂₃ c₄ { fst := i₁, snd := c₂₃.next i₂₃ }) j'
          ⊢ Eq (HSMul.hSMul (c₁.ε₂ c₂₃ c₄ { fst := i₁, snd := i₂₃ }) (CategoryTheory.Cat …
        -/
      · rw [← Functor.map_comp_assoc, mapBifunctor.ι_D₁]
        /-
          case pos.e_a
          C₁ : Type u_1
          C₂ : Type u_2
          C₂₃ : Type u_4
          C₃ : Type u_5
          C₄ : Type u_6
          inst✝²³ : CategoryTheory.Category.{u_14, u_1} C₁
          inst✝²² : CategoryTheory.Category.{u_16, u_2} C₂
          inst✝²¹ : CategoryTheory.Category.{u_17, u_5} C₃
          inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} C₄
          inst✝¹⁹ : CategoryTheory.Category.{u_15, u_4} C₂₃
          inst✝¹⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
          inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
          inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₃
          inst✝¹⁵ : CategoryTheory.Preadditive C₂₃
          inst✝¹⁴ : CategoryTheory.Preadditive C₄
          F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
          G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
          inst✝¹³ : G₂₃.PreservesZeroMorphisms
          inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
          inst✝¹¹ : F.PreservesZeroMorphisms
          inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
          ι₁ : Type u_7
          ι₂ : Type u_8
          ι₃ : Type u_9
          ι₁₂ : Type u_10
          ι₂₃ : Type u_11
          ι₄ : Type u_12
          inst✝⁹ : DecidableEq ι₄
          c₁ : ComplexShape ι₁
          c₂ : ComplexShape ι₂
          c₃ : ComplexShape ι₃
          K₁ : HomologicalComplex C₁ c₁
          K₂ : HomologicalComplex C₂ c₂
          K₃ : HomologicalComplex C₃ c₃
          c₁₂ : ComplexShape ι₁₂
          c₂₃ : ComplexShape ι₂₃
          c₄ : ComplexShape ι₄
          inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
          inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
          inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
          inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
          inst✝⁴ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
          inst✝³ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
          inst✝² : DecidableEq ι₂₃
          inst✝¹ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
          inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
          j j' : ι₄
          i₁ : ι₁
          i₂ : ι₂
          i₃ : ι₃
          h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
          i₂₃ : ι₂₃ := c₂.π c₃ c₂₃ { fst := i₂, snd := i₃ }
          h₁ : c₂₃.Rel i₂₃ (c₂₃.next i₂₃)
          h₂ : Eq (c₁.π c₂₃ c₄ { fst := i₁, snd := c₂₃.next i₂₃ }) j'
          ⊢ Eq (HSMul.hSMul (c₁.ε₂ c₂₃ c₄ { fst := i₁, snd := i₂₃ }) (CategoryTheory.Cat …
        -/
        by_cases h₃ : c₂.Rel i₂ (c₂.next i₂)
        · rw [d₂_eq _ _ _ _ _ _ _ _ _ h₃,
            mapBifunctor.d₁_eq _ _ _ _ h₃ _ _ (ComplexShape.next_π₁ c₃ c₂₃ h₃ i₃).symm,
            Functor.map_units_smul, Functor.map_comp, Linear.units_smul_comp,
            assoc, smul_smul, smul_left_cancel_iff,
            ιOrZero_eq _ _ _ _ _ _ _ _ _ _ _ _ (by
              dsimp [ComplexShape.r]
              rw [← h₂, ComplexShape.assoc c₁ c₂ c₃ c₁₂ c₂₃ c₄,
                ComplexShape.next_π₁ c₃ c₂₃ h₃ i₃]), ι_eq]
        · rw [d₂_eq_zero _ _ _ _ _ _ _ _ _ _ _ _ h₃,
            mapBifunctor.d₁_eq_zero _ _ _ _ _ _ _ h₃,
            Functor.map_zero, zero_comp, smul_zero]
        /-
          case pos.e_a
          C₁ : Type u_1
          C₂ : Type u_2
          C₂₃ : Type u_4
          C₃ : Type u_5
          C₄ : Type u_6
          inst✝²³ : CategoryTheory.Category.{u_14, u_1} C₁
          inst✝²² : CategoryTheory.Category.{u_16, u_2} C₂
          inst✝²¹ : CategoryTheory.Category.{u_17, u_5} C₃
          inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} C₄
          inst✝¹⁹ : CategoryTheory.Category.{u_15, u_4} C₂₃
          inst✝¹⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
          inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
          inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₃
          inst✝¹⁵ : CategoryTheory.Preadditive C₂₃
          inst✝¹⁴ : CategoryTheory.Preadditive C₄
          F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
          G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
          inst✝¹³ : G₂₃.PreservesZeroMorphisms
          inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
          inst✝¹¹ : F.PreservesZeroMorphisms
          inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
          ι₁ : Type u_7
          ι₂ : Type u_8
          ι₃ : Type u_9
          ι₁₂ : Type u_10
          ι₂₃ : Type u_11
          ι₄ : Type u_12
          inst✝⁹ : DecidableEq ι₄
          c₁ : ComplexShape ι₁
          c₂ : ComplexShape ι₂
          c₃ : ComplexShape ι₃
          K₁ : HomologicalComplex C₁ c₁
          K₂ : HomologicalComplex C₂ c₂
          K₃ : HomologicalComplex C₃ c₃
          c₁₂ : ComplexShape ι₁₂
          c₂₃ : ComplexShape ι₂₃
          c₄ : ComplexShape ι₄
          inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
          inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
          inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
          inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
          inst✝⁴ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
          inst✝³ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
          inst✝² : DecidableEq ι₂₃
          inst✝¹ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
          inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
          j j' : ι₄
          i₁ : ι₁
          i₂ : ι₂
          i₃ : ι₃
          h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
          i₂₃ : ι₂₃ := c₂.π c₃ c₂₃ { fst := i₂, snd := i₃ }
          h₁ : c₂₃.Rel i₂₃ (c₂₃.next i₂₃)
          h₂ : Eq (c₁.π c₂₃ c₄ { fst := i₁, snd := c₂₃.next i₂₃ }) j'
          ⊢ Eq (HSMul.hSMul (c₁.ε₂ c₂₃ c₄ { fst := i₁, snd := i₂₃ }) (CategoryTheory.Cat …
        -/
      · rw [← Functor.map_comp_assoc, mapBifunctor.ι_D₂]
        /-
          case pos.e_a
          C₁ : Type u_1
          C₂ : Type u_2
          C₂₃ : Type u_4
          C₃ : Type u_5
          C₄ : Type u_6
          inst✝²³ : CategoryTheory.Category.{u_14, u_1} C₁
          inst✝²² : CategoryTheory.Category.{u_16, u_2} C₂
          inst✝²¹ : CategoryTheory.Category.{u_17, u_5} C₃
          inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} C₄
          inst✝¹⁹ : CategoryTheory.Category.{u_15, u_4} C₂₃
          inst✝¹⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
          inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
          inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₃
          inst✝¹⁵ : CategoryTheory.Preadditive C₂₃
          inst✝¹⁴ : CategoryTheory.Preadditive C₄
          F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
          G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
          inst✝¹³ : G₂₃.PreservesZeroMorphisms
          inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
          inst✝¹¹ : F.PreservesZeroMorphisms
          inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
          ι₁ : Type u_7
          ι₂ : Type u_8
          ι₃ : Type u_9
          ι₁₂ : Type u_10
          ι₂₃ : Type u_11
          ι₄ : Type u_12
          inst✝⁹ : DecidableEq ι₄
          c₁ : ComplexShape ι₁
          c₂ : ComplexShape ι₂
          c₃ : ComplexShape ι₃
          K₁ : HomologicalComplex C₁ c₁
          K₂ : HomologicalComplex C₂ c₂
          K₃ : HomologicalComplex C₃ c₃
          c₁₂ : ComplexShape ι₁₂
          c₂₃ : ComplexShape ι₂₃
          c₄ : ComplexShape ι₄
          inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
          inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
          inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
          inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
          inst✝⁴ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
          inst✝³ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
          inst✝² : DecidableEq ι₂₃
          inst✝¹ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
          inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
          j j' : ι₄
          i₁ : ι₁
          i₂ : ι₂
          i₃ : ι₃
          h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
          i₂₃ : ι₂₃ := c₂.π c₃ c₂₃ { fst := i₂, snd := i₃ }
          h₁ : c₂₃.Rel i₂₃ (c₂₃.next i₂₃)
          h₂ : Eq (c₁.π c₂₃ c₄ { fst := i₁, snd := c₂₃.next i₂₃ }) j'
          ⊢ Eq (HSMul.hSMul (c₁.ε₂ c₂₃ c₄ { fst := i₁, snd := i₂₃ }) (CategoryTheory.Cat …
        -/
        by_cases h₃ : c₃.Rel i₃ (c₃.next i₃)
        · rw [d₃_eq _ _ _ _ _ _ _ _ _ _ h₃,
            mapBifunctor.d₂_eq _ _ _ _ _ h₃ _ (ComplexShape.next_π₂ c₂ c₂₃ i₂ h₃).symm,
            Functor.map_units_smul, Functor.map_comp, Linear.units_smul_comp, assoc,
            smul_smul, smul_left_cancel_iff]
          rw [ιOrZero_eq _ _ _ _ _ _ _ _ _ _ _ _ (by
            dsimp [ComplexShape.r]
            rw [← h₂, ComplexShape.assoc c₁ c₂ c₃ c₁₂ c₂₃ c₄, ComplexShape.next_π₂ c₂ c₂₃ i₂ h₃]),
            ι_eq]
        · rw [d₃_eq_zero _ _ _ _ _ _ _ _ _ _ _ _ h₃,
            mapBifunctor.d₂_eq_zero _ _ _ _ _ _ _ h₃,
            Functor.map_zero, zero_comp, smul_zero]
      /-
        case neg
        C₁ : Type u_1
        C₂ : Type u_2
        C₂₃ : Type u_4
        C₃ : Type u_5
        C₄ : Type u_6
        inst✝²³ : CategoryTheory.Category.{u_14, u_1} C₁
        inst✝²² : CategoryTheory.Category.{u_16, u_2} C₂
        inst✝²¹ : CategoryTheory.Category.{u_17, u_5} C₃
        inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} C₄
        inst✝¹⁹ : CategoryTheory.Category.{u_15, u_4} C₂₃
        inst✝¹⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
        inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
        inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₃
        inst✝¹⁵ : CategoryTheory.Preadditive C₂₃
        inst✝¹⁴ : CategoryTheory.Preadditive C₄
        F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
        G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
        inst✝¹³ : G₂₃.PreservesZeroMorphisms
        inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
        inst✝¹¹ : F.PreservesZeroMorphisms
        inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
        ι₁ : Type u_7
        ι₂ : Type u_8
        ι₃ : Type u_9
        ι₁₂ : Type u_10
        ι₂₃ : Type u_11
        ι₄ : Type u_12
        inst✝⁹ : DecidableEq ι₄
        c₁ : ComplexShape ι₁
        c₂ : ComplexShape ι₂
        c₃ : ComplexShape ι₃
        K₁ : HomologicalComplex C₁ c₁
        K₂ : HomologicalComplex C₂ c₂
        K₃ : HomologicalComplex C₃ c₃
        c₁₂ : ComplexShape ι₁₂
        c₂₃ : ComplexShape ι₂₃
        c₄ : ComplexShape ι₄
        inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
        inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
        inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
        inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
        inst✝⁴ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
        inst✝³ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
        inst✝² : DecidableEq ι₂₃
        inst✝¹ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
        inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
        j j' : ι₄
        i₁ : ι₁
        i₂ : ι₂
        i₃ : ι₃
        h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
        i₂₃ : ι₂₃ := c₂.π c₃ c₂₃ { fst := i₂, snd := i₃ }
        h₁ : c₂₃.Rel i₂₃ (c₂₃.next i₂₃)
        h₂ : Not (Eq (c₁.π c₂₃ c₄ { fst := i₁, snd := c₂₃.next i₂₃ }) j')
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.obj (K₁.X i₁)).map (K₂.ιMapBifunc …
      -/
    · rw [mapBifunctor.d₂_eq_zero' _ _ _ _ _ h₁ _ h₂, comp_zero]
      /-
        case neg
        C₁ : Type u_1
        C₂ : Type u_2
        C₂₃ : Type u_4
        C₃ : Type u_5
        C₄ : Type u_6
        inst✝²³ : CategoryTheory.Category.{u_14, u_1} C₁
        inst✝²² : CategoryTheory.Category.{u_16, u_2} C₂
        inst✝²¹ : CategoryTheory.Category.{u_17, u_5} C₃
        inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} C₄
        inst✝¹⁹ : CategoryTheory.Category.{u_15, u_4} C₂₃
        inst✝¹⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
        inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
        inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₃
        inst✝¹⁵ : CategoryTheory.Preadditive C₂₃
        inst✝¹⁴ : CategoryTheory.Preadditive C₄
        F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
        G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
        inst✝¹³ : G₂₃.PreservesZeroMorphisms
        inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
        inst✝¹¹ : F.PreservesZeroMorphisms
        inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
        ι₁ : Type u_7
        ι₂ : Type u_8
        ι₃ : Type u_9
        ι₁₂ : Type u_10
        ι₂₃ : Type u_11
        ι₄ : Type u_12
        inst✝⁹ : DecidableEq ι₄
        c₁ : ComplexShape ι₁
        c₂ : ComplexShape ι₂
        c₃ : ComplexShape ι₃
        K₁ : HomologicalComplex C₁ c₁
        K₂ : HomologicalComplex C₂ c₂
        K₃ : HomologicalComplex C₃ c₃
        c₁₂ : ComplexShape ι₁₂
        c₂₃ : ComplexShape ι₂₃
        c₄ : ComplexShape ι₄
        inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
        inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
        inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
        inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
        inst✝⁴ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
        inst✝³ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
        inst✝² : DecidableEq ι₂₃
        inst✝¹ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
        inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
        j j' : ι₄
        i₁ : ι₁
        i₂ : ι₂
        i₃ : ι₃
        h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
        i₂₃ : ι₂₃ := c₂.π c₃ c₂₃ { fst := i₂, snd := i₃ }
        h₁ : c₂₃.Rel i₂₃ (c₂₃.next i₂₃)
        h₂ : Not (Eq (c₁.π c₂₃ c₄ { fst := i₁, snd := c₂₃.next i₂₃ }) j')
        ⊢ Eq 0 (HAdd.hAdd (HomologicalComplex.mapBifunctor₂₃.d₂ F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ …
      -/
      trans 0 + 0
        /-
          C₁ : Type u_1
          C₂ : Type u_2
          C₂₃ : Type u_4
          C₃ : Type u_5
          C₄ : Type u_6
          inst✝²³ : CategoryTheory.Category.{u_14, u_1} C₁
          inst✝²² : CategoryTheory.Category.{u_16, u_2} C₂
          inst✝²¹ : CategoryTheory.Category.{u_17, u_5} C₃
          inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} C₄
          inst✝¹⁹ : CategoryTheory.Category.{u_15, u_4} C₂₃
          inst✝¹⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
          inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
          inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₃
          inst✝¹⁵ : CategoryTheory.Preadditive C₂₃
          inst✝¹⁴ : CategoryTheory.Preadditive C₄
          F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
          G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
          inst✝¹³ : G₂₃.PreservesZeroMorphisms
          inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
          inst✝¹¹ : F.PreservesZeroMorphisms
          inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
          ι₁ : Type u_7
          ι₂ : Type u_8
          ι₃ : Type u_9
          ι₁₂ : Type u_10
          ι₂₃ : Type u_11
          ι₄ : Type u_12
          inst✝⁹ : DecidableEq ι₄
          c₁ : ComplexShape ι₁
          c₂ : ComplexShape ι₂
          c₃ : ComplexShape ι₃
          K₁ : HomologicalComplex C₁ c₁
          K₂ : HomologicalComplex C₂ c₂
          K₃ : HomologicalComplex C₃ c₃
          c₁₂ : ComplexShape ι₁₂
          c₂₃ : ComplexShape ι₂₃
          c₄ : ComplexShape ι₄
          inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
          inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
          inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
          inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
          inst✝⁴ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
          inst✝³ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
          inst✝² : DecidableEq ι₂₃
          inst✝¹ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
          inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
          j j' : ι₄
          i₁ : ι₁
          i₂ : ι₂
          i₃ : ι₃
          h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
          i₂₃ : ι₂₃ := c₂.π c₃ c₂₃ { fst := i₂, snd := i₃ }
          h₁ : c₂₃.Rel i₂₃ (c₂₃.next i₂₃)
          h₂ : Not (Eq (c₁.π c₂₃ c₄ { fst := i₁, snd := c₂₃.next i₂₃ }) j')
          ⊢ Eq 0 (HAdd.hAdd 0 0)
        -/
      · simp
        /-
          🎉 no goals
        -/
        /-
          C₁ : Type u_1
          C₂ : Type u_2
          C₂₃ : Type u_4
          C₃ : Type u_5
          C₄ : Type u_6
          inst✝²³ : CategoryTheory.Category.{u_14, u_1} C₁
          inst✝²² : CategoryTheory.Category.{u_16, u_2} C₂
          inst✝²¹ : CategoryTheory.Category.{u_17, u_5} C₃
          inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} C₄
          inst✝¹⁹ : CategoryTheory.Category.{u_15, u_4} C₂₃
          inst✝¹⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
          inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
          inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₃
          inst✝¹⁵ : CategoryTheory.Preadditive C₂₃
          inst✝¹⁴ : CategoryTheory.Preadditive C₄
          F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
          G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
          inst✝¹³ : G₂₃.PreservesZeroMorphisms
          inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
          inst✝¹¹ : F.PreservesZeroMorphisms
          inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
          ι₁ : Type u_7
          ι₂ : Type u_8
          ι₃ : Type u_9
          ι₁₂ : Type u_10
          ι₂₃ : Type u_11
          ι₄ : Type u_12
          inst✝⁹ : DecidableEq ι₄
          c₁ : ComplexShape ι₁
          c₂ : ComplexShape ι₂
          c₃ : ComplexShape ι₃
          K₁ : HomologicalComplex C₁ c₁
          K₂ : HomologicalComplex C₂ c₂
          K₃ : HomologicalComplex C₃ c₃
          c₁₂ : ComplexShape ι₁₂
          c₂₃ : ComplexShape ι₂₃
          c₄ : ComplexShape ι₄
          inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
          inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
          inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
          inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
          inst✝⁴ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
          inst✝³ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
          inst✝² : DecidableEq ι₂₃
          inst✝¹ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
          inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
          j j' : ι₄
          i₁ : ι₁
          i₂ : ι₂
          i₃ : ι₃
          h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
          i₂₃ : ι₂₃ := c₂.π c₃ c₂₃ { fst := i₂, snd := i₃ }
          h₁ : c₂₃.Rel i₂₃ (c₂₃.next i₂₃)
          h₂ : Not (Eq (c₁.π c₂₃ c₄ { fst := i₁, snd := c₂₃.next i₂₃ }) j')
          ⊢ Eq (HAdd.hAdd 0 0) (HAdd.hAdd (HomologicalComplex.mapBifunctor₂₃.d₂ F G₂₃ K₁ …
        -/
      · congr 1
          /-
            case e_a
            C₁ : Type u_1
            C₂ : Type u_2
            C₂₃ : Type u_4
            C₃ : Type u_5
            C₄ : Type u_6
            inst✝²³ : CategoryTheory.Category.{u_14, u_1} C₁
            inst✝²² : CategoryTheory.Category.{u_16, u_2} C₂
            inst✝²¹ : CategoryTheory.Category.{u_17, u_5} C₃
            inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} C₄
            inst✝¹⁹ : CategoryTheory.Category.{u_15, u_4} C₂₃
            inst✝¹⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
            inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
            inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₃
            inst✝¹⁵ : CategoryTheory.Preadditive C₂₃
            inst✝¹⁴ : CategoryTheory.Preadditive C₄
            F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
            G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
            inst✝¹³ : G₂₃.PreservesZeroMorphisms
            inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
            inst✝¹¹ : F.PreservesZeroMorphisms
            inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
            ι₁ : Type u_7
            ι₂ : Type u_8
            ι₃ : Type u_9
            ι₁₂ : Type u_10
            ι₂₃ : Type u_11
            ι₄ : Type u_12
            inst✝⁹ : DecidableEq ι₄
            c₁ : ComplexShape ι₁
            c₂ : ComplexShape ι₂
            c₃ : ComplexShape ι₃
            K₁ : HomologicalComplex C₁ c₁
            K₂ : HomologicalComplex C₂ c₂
            K₃ : HomologicalComplex C₃ c₃
            c₁₂ : ComplexShape ι₁₂
            c₂₃ : ComplexShape ι₂₃
            c₄ : ComplexShape ι₄
            inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
            inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
            inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
            inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
            inst✝⁴ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
            inst✝³ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
            inst✝² : DecidableEq ι₂₃
            inst✝¹ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
            inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
            j j' : ι₄
            i₁ : ι₁
            i₂ : ι₂
            i₃ : ι₃
            h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
            i₂₃ : ι₂₃ := c₂.π c₃ c₂₃ { fst := i₂, snd := i₃ }
            h₁ : c₂₃.Rel i₂₃ (c₂₃.next i₂₃)
            h₂ : Not (Eq (c₁.π c₂₃ c₄ { fst := i₁, snd := c₂₃.next i₂₃ }) j')
            ⊢ Eq 0 (HomologicalComplex.mapBifunctor₂₃.d₂ F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ i₁ i₂ i …
          -/
        · by_cases h₃ : c₂.Rel i₂ (c₂.next i₂)
            /-
              case pos
              C₁ : Type u_1
              C₂ : Type u_2
              C₂₃ : Type u_4
              C₃ : Type u_5
              C₄ : Type u_6
              inst✝²³ : CategoryTheory.Category.{u_14, u_1} C₁
              inst✝²² : CategoryTheory.Category.{u_16, u_2} C₂
              inst✝²¹ : CategoryTheory.Category.{u_17, u_5} C₃
              inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} C₄
              inst✝¹⁹ : CategoryTheory.Category.{u_15, u_4} C₂₃
              inst✝¹⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
              inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
              inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₃
              inst✝¹⁵ : CategoryTheory.Preadditive C₂₃
              inst✝¹⁴ : CategoryTheory.Preadditive C₄
              F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
              G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
              inst✝¹³ : G₂₃.PreservesZeroMorphisms
              inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
              inst✝¹¹ : F.PreservesZeroMorphisms
              inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
              ι₁ : Type u_7
              ι₂ : Type u_8
              ι₃ : Type u_9
              ι₁₂ : Type u_10
              ι₂₃ : Type u_11
              ι₄ : Type u_12
              inst✝⁹ : DecidableEq ι₄
              c₁ : ComplexShape ι₁
              c₂ : ComplexShape ι₂
              c₃ : ComplexShape ι₃
              K₁ : HomologicalComplex C₁ c₁
              K₂ : HomologicalComplex C₂ c₂
              K₃ : HomologicalComplex C₃ c₃
              c₁₂ : ComplexShape ι₁₂
              c₂₃ : ComplexShape ι₂₃
              c₄ : ComplexShape ι₄
              inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
              inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
              inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
              inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
              inst✝⁴ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
              inst✝³ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
              inst✝² : DecidableEq ι₂₃
              inst✝¹ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
              inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
              j j' : ι₄
              i₁ : ι₁
              i₂ : ι₂
              i₃ : ι₃
              h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
              i₂₃ : ι₂₃ := c₂.π c₃ c₂₃ { fst := i₂, snd := i₃ }
              h₁ : c₂₃.Rel i₂₃ (c₂₃.next i₂₃)
              h₂ : Not (Eq (c₁.π c₂₃ c₄ { fst := i₁, snd := c₂₃.next i₂₃ }) j')
              h₃ : c₂.Rel i₂ (c₂.next i₂)
              ⊢ Eq 0 (HomologicalComplex.mapBifunctor₂₃.d₂ F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ i₁ i₂ i …
            -/
          · rw [d₂_eq _ _ _ _ _ _ _ _ _ h₃, ιOrZero_eq_zero, comp_zero, smul_zero]
            /-
              case pos.h
              C₁ : Type u_1
              C₂ : Type u_2
              C₂₃ : Type u_4
              C₃ : Type u_5
              C₄ : Type u_6
              inst✝²³ : CategoryTheory.Category.{u_14, u_1} C₁
              inst✝²² : CategoryTheory.Category.{u_16, u_2} C₂
              inst✝²¹ : CategoryTheory.Category.{u_17, u_5} C₃
              inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} C₄
              inst✝¹⁹ : CategoryTheory.Category.{u_15, u_4} C₂₃
              inst✝¹⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
              inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
              inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₃
              inst✝¹⁵ : CategoryTheory.Preadditive C₂₃
              inst✝¹⁴ : CategoryTheory.Preadditive C₄
              F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
              G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
              inst✝¹³ : G₂₃.PreservesZeroMorphisms
              inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
              inst✝¹¹ : F.PreservesZeroMorphisms
              inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
              ι₁ : Type u_7
              ι₂ : Type u_8
              ι₃ : Type u_9
              ι₁₂ : Type u_10
              ι₂₃ : Type u_11
              ι₄ : Type u_12
              inst✝⁹ : DecidableEq ι₄
              c₁ : ComplexShape ι₁
              c₂ : ComplexShape ι₂
              c₃ : ComplexShape ι₃
              K₁ : HomologicalComplex C₁ c₁
              K₂ : HomologicalComplex C₂ c₂
              K₃ : HomologicalComplex C₃ c₃
              c₁₂ : ComplexShape ι₁₂
              c₂₃ : ComplexShape ι₂₃
              c₄ : ComplexShape ι₄
              inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
              inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
              inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
              inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
              inst✝⁴ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
              inst✝³ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
              inst✝² : DecidableEq ι₂₃
              inst✝¹ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
              inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
              j j' : ι₄
              i₁ : ι₁
              i₂ : ι₂
              i₃ : ι₃
              h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
              i₂₃ : ι₂₃ := c₂.π c₃ c₂₃ { fst := i₂, snd := i₃ }
              h₁ : c₂₃.Rel i₂₃ (c₂₃.next i₂₃)
              h₂ : Not (Eq (c₁.π c₂₃ c₄ { fst := i₁, snd := c₂₃.next i₂₃ }) j')
              h₃ : c₂.Rel i₂ (c₂.next i₂)
              ⊢ Ne (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := c₂.next i₂, snd := i₃ } } …
            -/
            intro h₄
            /-
              case pos.h
              C₁ : Type u_1
              C₂ : Type u_2
              C₂₃ : Type u_4
              C₃ : Type u_5
              C₄ : Type u_6
              inst✝²³ : CategoryTheory.Category.{u_14, u_1} C₁
              inst✝²² : CategoryTheory.Category.{u_16, u_2} C₂
              inst✝²¹ : CategoryTheory.Category.{u_17, u_5} C₃
              inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} C₄
              inst✝¹⁹ : CategoryTheory.Category.{u_15, u_4} C₂₃
              inst✝¹⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
              inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
              inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₃
              inst✝¹⁵ : CategoryTheory.Preadditive C₂₃
              inst✝¹⁴ : CategoryTheory.Preadditive C₄
              F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
              G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
              inst✝¹³ : G₂₃.PreservesZeroMorphisms
              inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
              inst✝¹¹ : F.PreservesZeroMorphisms
              inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
              ι₁ : Type u_7
              ι₂ : Type u_8
              ι₃ : Type u_9
              ι₁₂ : Type u_10
              ι₂₃ : Type u_11
              ι₄ : Type u_12
              inst✝⁹ : DecidableEq ι₄
              c₁ : ComplexShape ι₁
              c₂ : ComplexShape ι₂
              c₃ : ComplexShape ι₃
              K₁ : HomologicalComplex C₁ c₁
              K₂ : HomologicalComplex C₂ c₂
              K₃ : HomologicalComplex C₃ c₃
              c₁₂ : ComplexShape ι₁₂
              c₂₃ : ComplexShape ι₂₃
              c₄ : ComplexShape ι₄
              inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
              inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
              inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
              inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
              inst✝⁴ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
              inst✝³ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
              inst✝² : DecidableEq ι₂₃
              inst✝¹ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
              inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
              j j' : ι₄
              i₁ : ι₁
              i₂ : ι₂
              i₃ : ι₃
              h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
              i₂₃ : ι₂₃ := c₂.π c₃ c₂₃ { fst := i₂, snd := i₃ }
              h₁ : c₂₃.Rel i₂₃ (c₂₃.next i₂₃)
              h₂ : Not (Eq (c₁.π c₂₃ c₄ { fst := i₁, snd := c₂₃.next i₂₃ }) j')
              h₃ : c₂.Rel i₂ (c₂.next i₂)
              h₄ : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := c₂.next i₂, snd := i₃  …
              ⊢ False
            -/
            apply h₂
            /-
              case pos.h
              C₁ : Type u_1
              C₂ : Type u_2
              C₂₃ : Type u_4
              C₃ : Type u_5
              C₄ : Type u_6
              inst✝²³ : CategoryTheory.Category.{u_14, u_1} C₁
              inst✝²² : CategoryTheory.Category.{u_16, u_2} C₂
              inst✝²¹ : CategoryTheory.Category.{u_17, u_5} C₃
              inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} C₄
              inst✝¹⁹ : CategoryTheory.Category.{u_15, u_4} C₂₃
              inst✝¹⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
              inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
              inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₃
              inst✝¹⁵ : CategoryTheory.Preadditive C₂₃
              inst✝¹⁴ : CategoryTheory.Preadditive C₄
              F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
              G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
              inst✝¹³ : G₂₃.PreservesZeroMorphisms
              inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
              inst✝¹¹ : F.PreservesZeroMorphisms
              inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
              ι₁ : Type u_7
              ι₂ : Type u_8
              ι₃ : Type u_9
              ι₁₂ : Type u_10
              ι₂₃ : Type u_11
              ι₄ : Type u_12
              inst✝⁹ : DecidableEq ι₄
              c₁ : ComplexShape ι₁
              c₂ : ComplexShape ι₂
              c₃ : ComplexShape ι₃
              K₁ : HomologicalComplex C₁ c₁
              K₂ : HomologicalComplex C₂ c₂
              K₃ : HomologicalComplex C₃ c₃
              c₁₂ : ComplexShape ι₁₂
              c₂₃ : ComplexShape ι₂₃
              c₄ : ComplexShape ι₄
              inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
              inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
              inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
              inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
              inst✝⁴ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
              inst✝³ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
              inst✝² : DecidableEq ι₂₃
              inst✝¹ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
              inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
              j j' : ι₄
              i₁ : ι₁
              i₂ : ι₂
              i₃ : ι₃
              h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
              i₂₃ : ι₂₃ := c₂.π c₃ c₂₃ { fst := i₂, snd := i₃ }
              h₁ : c₂₃.Rel i₂₃ (c₂₃.next i₂₃)
              h₂ : Not (Eq (c₁.π c₂₃ c₄ { fst := i₁, snd := c₂₃.next i₂₃ }) j')
              h₃ : c₂.Rel i₂ (c₂.next i₂)
              h₄ : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := c₂.next i₂, snd := i₃  …
              ⊢ Eq (c₁.π c₂₃ c₄ { fst := i₁, snd := c₂₃.next i₂₃ }) j'
            -/
            rw [← h₄]
            /-
              case pos.h
              C₁ : Type u_1
              C₂ : Type u_2
              C₂₃ : Type u_4
              C₃ : Type u_5
              C₄ : Type u_6
              inst✝²³ : CategoryTheory.Category.{u_14, u_1} C₁
              inst✝²² : CategoryTheory.Category.{u_16, u_2} C₂
              inst✝²¹ : CategoryTheory.Category.{u_17, u_5} C₃
              inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} C₄
              inst✝¹⁹ : CategoryTheory.Category.{u_15, u_4} C₂₃
              inst✝¹⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
              inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
              inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₃
              inst✝¹⁵ : CategoryTheory.Preadditive C₂₃
              inst✝¹⁴ : CategoryTheory.Preadditive C₄
              F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
              G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
              inst✝¹³ : G₂₃.PreservesZeroMorphisms
              inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
              inst✝¹¹ : F.PreservesZeroMorphisms
              inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
              ι₁ : Type u_7
              ι₂ : Type u_8
              ι₃ : Type u_9
              ι₁₂ : Type u_10
              ι₂₃ : Type u_11
              ι₄ : Type u_12
              inst✝⁹ : DecidableEq ι₄
              c₁ : ComplexShape ι₁
              c₂ : ComplexShape ι₂
              c₃ : ComplexShape ι₃
              K₁ : HomologicalComplex C₁ c₁
              K₂ : HomologicalComplex C₂ c₂
              K₃ : HomologicalComplex C₃ c₃
              c₁₂ : ComplexShape ι₁₂
              c₂₃ : ComplexShape ι₂₃
              c₄ : ComplexShape ι₄
              inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
              inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
              inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
              inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
              inst✝⁴ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
              inst✝³ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
              inst✝² : DecidableEq ι₂₃
              inst✝¹ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
              inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
              j j' : ι₄
              i₁ : ι₁
              i₂ : ι₂
              i₃ : ι₃
              h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
              i₂₃ : ι₂₃ := c₂.π c₃ c₂₃ { fst := i₂, snd := i₃ }
              h₁ : c₂₃.Rel i₂₃ (c₂₃.next i₂₃)
              h₂ : Not (Eq (c₁.π c₂₃ c₄ { fst := i₁, snd := c₂₃.next i₂₃ }) j')
              h₃ : c₂.Rel i₂ (c₂.next i₂)
              h₄ : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := c₂.next i₂, snd := i₃  …
              ⊢ Eq (c₁.π c₂₃ c₄ { fst := i₁, snd := c₂₃.next i₂₃ }) (c₁.r c₂ c₃ c₁₂ c₄ { fst …
            -/
            dsimp [ComplexShape.r]
            /-
              case pos.h
              C₁ : Type u_1
              C₂ : Type u_2
              C₂₃ : Type u_4
              C₃ : Type u_5
              C₄ : Type u_6
              inst✝²³ : CategoryTheory.Category.{u_14, u_1} C₁
              inst✝²² : CategoryTheory.Category.{u_16, u_2} C₂
              inst✝²¹ : CategoryTheory.Category.{u_17, u_5} C₃
              inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} C₄
              inst✝¹⁹ : CategoryTheory.Category.{u_15, u_4} C₂₃
              inst✝¹⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
              inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
              inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₃
              inst✝¹⁵ : CategoryTheory.Preadditive C₂₃
              inst✝¹⁴ : CategoryTheory.Preadditive C₄
              F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
              G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
              inst✝¹³ : G₂₃.PreservesZeroMorphisms
              inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
              inst✝¹¹ : F.PreservesZeroMorphisms
              inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
              ι₁ : Type u_7
              ι₂ : Type u_8
              ι₃ : Type u_9
              ι₁₂ : Type u_10
              ι₂₃ : Type u_11
              ι₄ : Type u_12
              inst✝⁹ : DecidableEq ι₄
              c₁ : ComplexShape ι₁
              c₂ : ComplexShape ι₂
              c₃ : ComplexShape ι₃
              K₁ : HomologicalComplex C₁ c₁
              K₂ : HomologicalComplex C₂ c₂
              K₃ : HomologicalComplex C₃ c₃
              c₁₂ : ComplexShape ι₁₂
              c₂₃ : ComplexShape ι₂₃
              c₄ : ComplexShape ι₄
              inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
              inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
              inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
              inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
              inst✝⁴ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
              inst✝³ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
              inst✝² : DecidableEq ι₂₃
              inst✝¹ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
              inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
              j j' : ι₄
              i₁ : ι₁
              i₂ : ι₂
              i₃ : ι₃
              h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
              i₂₃ : ι₂₃ := c₂.π c₃ c₂₃ { fst := i₂, snd := i₃ }
              h₁ : c₂₃.Rel i₂₃ (c₂₃.next i₂₃)
              h₂ : Not (Eq (c₁.π c₂₃ c₄ { fst := i₁, snd := c₂₃.next i₂₃ }) j')
              h₃ : c₂.Rel i₂ (c₂.next i₂)
              h₄ : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := c₂.next i₂, snd := i₃  …
              ⊢ Eq (c₁.π c₂₃ c₄ { fst := i₁, snd := c₂₃.next i₂₃ }) (c₁₂.π c₃ c₄ { fst := c₁ …
            -/
            rw [ComplexShape.assoc c₁ c₂ c₃ c₁₂ c₂₃ c₄, ComplexShape.next_π₁ c₃ c₂₃ h₃ i₃]
            /-
              🎉 no goals
            -/
            /-
              case neg
              C₁ : Type u_1
              C₂ : Type u_2
              C₂₃ : Type u_4
              C₃ : Type u_5
              C₄ : Type u_6
              inst✝²³ : CategoryTheory.Category.{u_14, u_1} C₁
              inst✝²² : CategoryTheory.Category.{u_16, u_2} C₂
              inst✝²¹ : CategoryTheory.Category.{u_17, u_5} C₃
              inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} C₄
              inst✝¹⁹ : CategoryTheory.Category.{u_15, u_4} C₂₃
              inst✝¹⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
              inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
              inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₃
              inst✝¹⁵ : CategoryTheory.Preadditive C₂₃
              inst✝¹⁴ : CategoryTheory.Preadditive C₄
              F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
              G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
              inst✝¹³ : G₂₃.PreservesZeroMorphisms
              inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
              inst✝¹¹ : F.PreservesZeroMorphisms
              inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
              ι₁ : Type u_7
              ι₂ : Type u_8
              ι₃ : Type u_9
              ι₁₂ : Type u_10
              ι₂₃ : Type u_11
              ι₄ : Type u_12
              inst✝⁹ : DecidableEq ι₄
              c₁ : ComplexShape ι₁
              c₂ : ComplexShape ι₂
              c₃ : ComplexShape ι₃
              K₁ : HomologicalComplex C₁ c₁
              K₂ : HomologicalComplex C₂ c₂
              K₃ : HomologicalComplex C₃ c₃
              c₁₂ : ComplexShape ι₁₂
              c₂₃ : ComplexShape ι₂₃
              c₄ : ComplexShape ι₄
              inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
              inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
              inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
              inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
              inst✝⁴ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
              inst✝³ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
              inst✝² : DecidableEq ι₂₃
              inst✝¹ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
              inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
              j j' : ι₄
              i₁ : ι₁
              i₂ : ι₂
              i₃ : ι₃
              h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
              i₂₃ : ι₂₃ := c₂.π c₃ c₂₃ { fst := i₂, snd := i₃ }
              h₁ : c₂₃.Rel i₂₃ (c₂₃.next i₂₃)
              h₂ : Not (Eq (c₁.π c₂₃ c₄ { fst := i₁, snd := c₂₃.next i₂₃ }) j')
              h₃ : Not (c₂.Rel i₂ (c₂.next i₂))
              ⊢ Eq 0 (HomologicalComplex.mapBifunctor₂₃.d₂ F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ i₁ i₂ i …
            -/
          · rw [d₂_eq_zero _ _ _ _ _ _ _ _ _ _ _ _ h₃]
            /-
              🎉 no goals
            -/
          /-
            case e_a
            C₁ : Type u_1
            C₂ : Type u_2
            C₂₃ : Type u_4
            C₃ : Type u_5
            C₄ : Type u_6
            inst✝²³ : CategoryTheory.Category.{u_14, u_1} C₁
            inst✝²² : CategoryTheory.Category.{u_16, u_2} C₂
            inst✝²¹ : CategoryTheory.Category.{u_17, u_5} C₃
            inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} C₄
            inst✝¹⁹ : CategoryTheory.Category.{u_15, u_4} C₂₃
            inst✝¹⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
            inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
            inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₃
            inst✝¹⁵ : CategoryTheory.Preadditive C₂₃
            inst✝¹⁴ : CategoryTheory.Preadditive C₄
            F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
            G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
            inst✝¹³ : G₂₃.PreservesZeroMorphisms
            inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
            inst✝¹¹ : F.PreservesZeroMorphisms
            inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
            ι₁ : Type u_7
            ι₂ : Type u_8
            ι₃ : Type u_9
            ι₁₂ : Type u_10
            ι₂₃ : Type u_11
            ι₄ : Type u_12
            inst✝⁹ : DecidableEq ι₄
            c₁ : ComplexShape ι₁
            c₂ : ComplexShape ι₂
            c₃ : ComplexShape ι₃
            K₁ : HomologicalComplex C₁ c₁
            K₂ : HomologicalComplex C₂ c₂
            K₃ : HomologicalComplex C₃ c₃
            c₁₂ : ComplexShape ι₁₂
            c₂₃ : ComplexShape ι₂₃
            c₄ : ComplexShape ι₄
            inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
            inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
            inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
            inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
            inst✝⁴ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
            inst✝³ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
            inst✝² : DecidableEq ι₂₃
            inst✝¹ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
            inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
            j j' : ι₄
            i₁ : ι₁
            i₂ : ι₂
            i₃ : ι₃
            h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
            i₂₃ : ι₂₃ := c₂.π c₃ c₂₃ { fst := i₂, snd := i₃ }
            h₁ : c₂₃.Rel i₂₃ (c₂₃.next i₂₃)
            h₂ : Not (Eq (c₁.π c₂₃ c₄ { fst := i₁, snd := c₂₃.next i₂₃ }) j')
            ⊢ Eq 0 (HomologicalComplex.mapBifunctor₂₃.d₃ F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ i₁ i₂ i …
          -/
        · by_cases h₃ : c₃.Rel i₃ (c₃.next i₃)
            /-
              case pos
              C₁ : Type u_1
              C₂ : Type u_2
              C₂₃ : Type u_4
              C₃ : Type u_5
              C₄ : Type u_6
              inst✝²³ : CategoryTheory.Category.{u_14, u_1} C₁
              inst✝²² : CategoryTheory.Category.{u_16, u_2} C₂
              inst✝²¹ : CategoryTheory.Category.{u_17, u_5} C₃
              inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} C₄
              inst✝¹⁹ : CategoryTheory.Category.{u_15, u_4} C₂₃
              inst✝¹⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
              inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
              inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₃
              inst✝¹⁵ : CategoryTheory.Preadditive C₂₃
              inst✝¹⁴ : CategoryTheory.Preadditive C₄
              F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
              G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
              inst✝¹³ : G₂₃.PreservesZeroMorphisms
              inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
              inst✝¹¹ : F.PreservesZeroMorphisms
              inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
              ι₁ : Type u_7
              ι₂ : Type u_8
              ι₃ : Type u_9
              ι₁₂ : Type u_10
              ι₂₃ : Type u_11
              ι₄ : Type u_12
              inst✝⁹ : DecidableEq ι₄
              c₁ : ComplexShape ι₁
              c₂ : ComplexShape ι₂
              c₃ : ComplexShape ι₃
              K₁ : HomologicalComplex C₁ c₁
              K₂ : HomologicalComplex C₂ c₂
              K₃ : HomologicalComplex C₃ c₃
              c₁₂ : ComplexShape ι₁₂
              c₂₃ : ComplexShape ι₂₃
              c₄ : ComplexShape ι₄
              inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
              inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
              inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
              inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
              inst✝⁴ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
              inst✝³ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
              inst✝² : DecidableEq ι₂₃
              inst✝¹ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
              inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
              j j' : ι₄
              i₁ : ι₁
              i₂ : ι₂
              i₃ : ι₃
              h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
              i₂₃ : ι₂₃ := c₂.π c₃ c₂₃ { fst := i₂, snd := i₃ }
              h₁ : c₂₃.Rel i₂₃ (c₂₃.next i₂₃)
              h₂ : Not (Eq (c₁.π c₂₃ c₄ { fst := i₁, snd := c₂₃.next i₂₃ }) j')
              h₃ : c₃.Rel i₃ (c₃.next i₃)
              ⊢ Eq 0 (HomologicalComplex.mapBifunctor₂₃.d₃ F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ i₁ i₂ i …
            -/
          · rw [d₃_eq _ _ _ _ _ _ _ _ _ _ h₃, ιOrZero_eq_zero, comp_zero, smul_zero]
            /-
              case pos.h
              C₁ : Type u_1
              C₂ : Type u_2
              C₂₃ : Type u_4
              C₃ : Type u_5
              C₄ : Type u_6
              inst✝²³ : CategoryTheory.Category.{u_14, u_1} C₁
              inst✝²² : CategoryTheory.Category.{u_16, u_2} C₂
              inst✝²¹ : CategoryTheory.Category.{u_17, u_5} C₃
              inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} C₄
              inst✝¹⁹ : CategoryTheory.Category.{u_15, u_4} C₂₃
              inst✝¹⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
              inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
              inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₃
              inst✝¹⁵ : CategoryTheory.Preadditive C₂₃
              inst✝¹⁴ : CategoryTheory.Preadditive C₄
              F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
              G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
              inst✝¹³ : G₂₃.PreservesZeroMorphisms
              inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
              inst✝¹¹ : F.PreservesZeroMorphisms
              inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
              ι₁ : Type u_7
              ι₂ : Type u_8
              ι₃ : Type u_9
              ι₁₂ : Type u_10
              ι₂₃ : Type u_11
              ι₄ : Type u_12
              inst✝⁹ : DecidableEq ι₄
              c₁ : ComplexShape ι₁
              c₂ : ComplexShape ι₂
              c₃ : ComplexShape ι₃
              K₁ : HomologicalComplex C₁ c₁
              K₂ : HomologicalComplex C₂ c₂
              K₃ : HomologicalComplex C₃ c₃
              c₁₂ : ComplexShape ι₁₂
              c₂₃ : ComplexShape ι₂₃
              c₄ : ComplexShape ι₄
              inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
              inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
              inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
              inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
              inst✝⁴ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
              inst✝³ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
              inst✝² : DecidableEq ι₂₃
              inst✝¹ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
              inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
              j j' : ι₄
              i₁ : ι₁
              i₂ : ι₂
              i₃ : ι₃
              h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
              i₂₃ : ι₂₃ := c₂.π c₃ c₂₃ { fst := i₂, snd := i₃ }
              h₁ : c₂₃.Rel i₂₃ (c₂₃.next i₂₃)
              h₂ : Not (Eq (c₁.π c₂₃ c₄ { fst := i₁, snd := c₂₃.next i₂₃ }) j')
              h₃ : c₃.Rel i₃ (c₃.next i₃)
              ⊢ Ne (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := c₃.next i₃ } } …
            -/
            intro h₄
            /-
              case pos.h
              C₁ : Type u_1
              C₂ : Type u_2
              C₂₃ : Type u_4
              C₃ : Type u_5
              C₄ : Type u_6
              inst✝²³ : CategoryTheory.Category.{u_14, u_1} C₁
              inst✝²² : CategoryTheory.Category.{u_16, u_2} C₂
              inst✝²¹ : CategoryTheory.Category.{u_17, u_5} C₃
              inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} C₄
              inst✝¹⁹ : CategoryTheory.Category.{u_15, u_4} C₂₃
              inst✝¹⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
              inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
              inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₃
              inst✝¹⁵ : CategoryTheory.Preadditive C₂₃
              inst✝¹⁴ : CategoryTheory.Preadditive C₄
              F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
              G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
              inst✝¹³ : G₂₃.PreservesZeroMorphisms
              inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
              inst✝¹¹ : F.PreservesZeroMorphisms
              inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
              ι₁ : Type u_7
              ι₂ : Type u_8
              ι₃ : Type u_9
              ι₁₂ : Type u_10
              ι₂₃ : Type u_11
              ι₄ : Type u_12
              inst✝⁹ : DecidableEq ι₄
              c₁ : ComplexShape ι₁
              c₂ : ComplexShape ι₂
              c₃ : ComplexShape ι₃
              K₁ : HomologicalComplex C₁ c₁
              K₂ : HomologicalComplex C₂ c₂
              K₃ : HomologicalComplex C₃ c₃
              c₁₂ : ComplexShape ι₁₂
              c₂₃ : ComplexShape ι₂₃
              c₄ : ComplexShape ι₄
              inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
              inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
              inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
              inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
              inst✝⁴ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
              inst✝³ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
              inst✝² : DecidableEq ι₂₃
              inst✝¹ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
              inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
              j j' : ι₄
              i₁ : ι₁
              i₂ : ι₂
              i₃ : ι₃
              h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
              i₂₃ : ι₂₃ := c₂.π c₃ c₂₃ { fst := i₂, snd := i₃ }
              h₁ : c₂₃.Rel i₂₃ (c₂₃.next i₂₃)
              h₂ : Not (Eq (c₁.π c₂₃ c₄ { fst := i₁, snd := c₂₃.next i₂₃ }) j')
              h₃ : c₃.Rel i₃ (c₃.next i₃)
              h₄ : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := c₃.next i₃  …
              ⊢ False
            -/
            apply h₂
            /-
              case pos.h
              C₁ : Type u_1
              C₂ : Type u_2
              C₂₃ : Type u_4
              C₃ : Type u_5
              C₄ : Type u_6
              inst✝²³ : CategoryTheory.Category.{u_14, u_1} C₁
              inst✝²² : CategoryTheory.Category.{u_16, u_2} C₂
              inst✝²¹ : CategoryTheory.Category.{u_17, u_5} C₃
              inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} C₄
              inst✝¹⁹ : CategoryTheory.Category.{u_15, u_4} C₂₃
              inst✝¹⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
              inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
              inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₃
              inst✝¹⁵ : CategoryTheory.Preadditive C₂₃
              inst✝¹⁴ : CategoryTheory.Preadditive C₄
              F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
              G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
              inst✝¹³ : G₂₃.PreservesZeroMorphisms
              inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
              inst✝¹¹ : F.PreservesZeroMorphisms
              inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
              ι₁ : Type u_7
              ι₂ : Type u_8
              ι₃ : Type u_9
              ι₁₂ : Type u_10
              ι₂₃ : Type u_11
              ι₄ : Type u_12
              inst✝⁹ : DecidableEq ι₄
              c₁ : ComplexShape ι₁
              c₂ : ComplexShape ι₂
              c₃ : ComplexShape ι₃
              K₁ : HomologicalComplex C₁ c₁
              K₂ : HomologicalComplex C₂ c₂
              K₃ : HomologicalComplex C₃ c₃
              c₁₂ : ComplexShape ι₁₂
              c₂₃ : ComplexShape ι₂₃
              c₄ : ComplexShape ι₄
              inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
              inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
              inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
              inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
              inst✝⁴ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
              inst✝³ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
              inst✝² : DecidableEq ι₂₃
              inst✝¹ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
              inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
              j j' : ι₄
              i₁ : ι₁
              i₂ : ι₂
              i₃ : ι₃
              h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
              i₂₃ : ι₂₃ := c₂.π c₃ c₂₃ { fst := i₂, snd := i₃ }
              h₁ : c₂₃.Rel i₂₃ (c₂₃.next i₂₃)
              h₂ : Not (Eq (c₁.π c₂₃ c₄ { fst := i₁, snd := c₂₃.next i₂₃ }) j')
              h₃ : c₃.Rel i₃ (c₃.next i₃)
              h₄ : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := c₃.next i₃  …
              ⊢ Eq (c₁.π c₂₃ c₄ { fst := i₁, snd := c₂₃.next i₂₃ }) j'
            -/
            rw [← h₄]
            /-
              case pos.h
              C₁ : Type u_1
              C₂ : Type u_2
              C₂₃ : Type u_4
              C₃ : Type u_5
              C₄ : Type u_6
              inst✝²³ : CategoryTheory.Category.{u_14, u_1} C₁
              inst✝²² : CategoryTheory.Category.{u_16, u_2} C₂
              inst✝²¹ : CategoryTheory.Category.{u_17, u_5} C₃
              inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} C₄
              inst✝¹⁹ : CategoryTheory.Category.{u_15, u_4} C₂₃
              inst✝¹⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
              inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
              inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₃
              inst✝¹⁵ : CategoryTheory.Preadditive C₂₃
              inst✝¹⁴ : CategoryTheory.Preadditive C₄
              F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
              G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
              inst✝¹³ : G₂₃.PreservesZeroMorphisms
              inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
              inst✝¹¹ : F.PreservesZeroMorphisms
              inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
              ι₁ : Type u_7
              ι₂ : Type u_8
              ι₃ : Type u_9
              ι₁₂ : Type u_10
              ι₂₃ : Type u_11
              ι₄ : Type u_12
              inst✝⁹ : DecidableEq ι₄
              c₁ : ComplexShape ι₁
              c₂ : ComplexShape ι₂
              c₃ : ComplexShape ι₃
              K₁ : HomologicalComplex C₁ c₁
              K₂ : HomologicalComplex C₂ c₂
              K₃ : HomologicalComplex C₃ c₃
              c₁₂ : ComplexShape ι₁₂
              c₂₃ : ComplexShape ι₂₃
              c₄ : ComplexShape ι₄
              inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
              inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
              inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
              inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
              inst✝⁴ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
              inst✝³ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
              inst✝² : DecidableEq ι₂₃
              inst✝¹ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
              inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
              j j' : ι₄
              i₁ : ι₁
              i₂ : ι₂
              i₃ : ι₃
              h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
              i₂₃ : ι₂₃ := c₂.π c₃ c₂₃ { fst := i₂, snd := i₃ }
              h₁ : c₂₃.Rel i₂₃ (c₂₃.next i₂₃)
              h₂ : Not (Eq (c₁.π c₂₃ c₄ { fst := i₁, snd := c₂₃.next i₂₃ }) j')
              h₃ : c₃.Rel i₃ (c₃.next i₃)
              h₄ : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := c₃.next i₃  …
              ⊢ Eq (c₁.π c₂₃ c₄ { fst := i₁, snd := c₂₃.next i₂₃ }) (c₁.r c₂ c₃ c₁₂ c₄ { fst …
            -/
            dsimp [ComplexShape.r]
            /-
              case pos.h
              C₁ : Type u_1
              C₂ : Type u_2
              C₂₃ : Type u_4
              C₃ : Type u_5
              C₄ : Type u_6
              inst✝²³ : CategoryTheory.Category.{u_14, u_1} C₁
              inst✝²² : CategoryTheory.Category.{u_16, u_2} C₂
              inst✝²¹ : CategoryTheory.Category.{u_17, u_5} C₃
              inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} C₄
              inst✝¹⁹ : CategoryTheory.Category.{u_15, u_4} C₂₃
              inst✝¹⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
              inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
              inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₃
              inst✝¹⁵ : CategoryTheory.Preadditive C₂₃
              inst✝¹⁴ : CategoryTheory.Preadditive C₄
              F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
              G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
              inst✝¹³ : G₂₃.PreservesZeroMorphisms
              inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
              inst✝¹¹ : F.PreservesZeroMorphisms
              inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
              ι₁ : Type u_7
              ι₂ : Type u_8
              ι₃ : Type u_9
              ι₁₂ : Type u_10
              ι₂₃ : Type u_11
              ι₄ : Type u_12
              inst✝⁹ : DecidableEq ι₄
              c₁ : ComplexShape ι₁
              c₂ : ComplexShape ι₂
              c₃ : ComplexShape ι₃
              K₁ : HomologicalComplex C₁ c₁
              K₂ : HomologicalComplex C₂ c₂
              K₃ : HomologicalComplex C₃ c₃
              c₁₂ : ComplexShape ι₁₂
              c₂₃ : ComplexShape ι₂₃
              c₄ : ComplexShape ι₄
              inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
              inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
              inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
              inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
              inst✝⁴ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
              inst✝³ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
              inst✝² : DecidableEq ι₂₃
              inst✝¹ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
              inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
              j j' : ι₄
              i₁ : ι₁
              i₂ : ι₂
              i₃ : ι₃
              h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
              i₂₃ : ι₂₃ := c₂.π c₃ c₂₃ { fst := i₂, snd := i₃ }
              h₁ : c₂₃.Rel i₂₃ (c₂₃.next i₂₃)
              h₂ : Not (Eq (c₁.π c₂₃ c₄ { fst := i₁, snd := c₂₃.next i₂₃ }) j')
              h₃ : c₃.Rel i₃ (c₃.next i₃)
              h₄ : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := c₃.next i₃  …
              ⊢ Eq (c₁.π c₂₃ c₄ { fst := i₁, snd := c₂₃.next i₂₃ }) (c₁₂.π c₃ c₄ { fst := c₁ …
            -/
            rw [ComplexShape.assoc c₁ c₂ c₃ c₁₂ c₂₃ c₄, ComplexShape.next_π₂ c₂ c₂₃ i₂ h₃]
            /-
              🎉 no goals
            -/
            /-
              case neg
              C₁ : Type u_1
              C₂ : Type u_2
              C₂₃ : Type u_4
              C₃ : Type u_5
              C₄ : Type u_6
              inst✝²³ : CategoryTheory.Category.{u_14, u_1} C₁
              inst✝²² : CategoryTheory.Category.{u_16, u_2} C₂
              inst✝²¹ : CategoryTheory.Category.{u_17, u_5} C₃
              inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} C₄
              inst✝¹⁹ : CategoryTheory.Category.{u_15, u_4} C₂₃
              inst✝¹⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
              inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
              inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₃
              inst✝¹⁵ : CategoryTheory.Preadditive C₂₃
              inst✝¹⁴ : CategoryTheory.Preadditive C₄
              F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
              G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
              inst✝¹³ : G₂₃.PreservesZeroMorphisms
              inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
              inst✝¹¹ : F.PreservesZeroMorphisms
              inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
              ι₁ : Type u_7
              ι₂ : Type u_8
              ι₃ : Type u_9
              ι₁₂ : Type u_10
              ι₂₃ : Type u_11
              ι₄ : Type u_12
              inst✝⁹ : DecidableEq ι₄
              c₁ : ComplexShape ι₁
              c₂ : ComplexShape ι₂
              c₃ : ComplexShape ι₃
              K₁ : HomologicalComplex C₁ c₁
              K₂ : HomologicalComplex C₂ c₂
              K₃ : HomologicalComplex C₃ c₃
              c₁₂ : ComplexShape ι₁₂
              c₂₃ : ComplexShape ι₂₃
              c₄ : ComplexShape ι₄
              inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
              inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
              inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
              inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
              inst✝⁴ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
              inst✝³ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
              inst✝² : DecidableEq ι₂₃
              inst✝¹ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
              inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
              j j' : ι₄
              i₁ : ι₁
              i₂ : ι₂
              i₃ : ι₃
              h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
              i₂₃ : ι₂₃ := c₂.π c₃ c₂₃ { fst := i₂, snd := i₃ }
              h₁ : c₂₃.Rel i₂₃ (c₂₃.next i₂₃)
              h₂ : Not (Eq (c₁.π c₂₃ c₄ { fst := i₁, snd := c₂₃.next i₂₃ }) j')
              h₃ : Not (c₃.Rel i₃ (c₃.next i₃))
              ⊢ Eq 0 (HomologicalComplex.mapBifunctor₂₃.d₃ F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ i₁ i₂ i …
            -/
          · rw [d₃_eq_zero _ _ _ _ _ _ _ _ _ _ _ _ h₃]
            /-
              🎉 no goals
            -/
    /-
      case neg
      C₁ : Type u_1
      C₂ : Type u_2
      C₂₃ : Type u_4
      C₃ : Type u_5
      C₄ : Type u_6
      inst✝²³ : CategoryTheory.Category.{u_14, u_1} C₁
      inst✝²² : CategoryTheory.Category.{u_16, u_2} C₂
      inst✝²¹ : CategoryTheory.Category.{u_17, u_5} C₃
      inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} C₄
      inst✝¹⁹ : CategoryTheory.Category.{u_15, u_4} C₂₃
      inst✝¹⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
      inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
      inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₃
      inst✝¹⁵ : CategoryTheory.Preadditive C₂₃
      inst✝¹⁴ : CategoryTheory.Preadditive C₄
      F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
      G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
      inst✝¹³ : G₂₃.PreservesZeroMorphisms
      inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
      inst✝¹¹ : F.PreservesZeroMorphisms
      inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
      ι₁ : Type u_7
      ι₂ : Type u_8
      ι₃ : Type u_9
      ι₁₂ : Type u_10
      ι₂₃ : Type u_11
      ι₄ : Type u_12
      inst✝⁹ : DecidableEq ι₄
      c₁ : ComplexShape ι₁
      c₂ : ComplexShape ι₂
      c₃ : ComplexShape ι₃
      K₁ : HomologicalComplex C₁ c₁
      K₂ : HomologicalComplex C₂ c₂
      K₃ : HomologicalComplex C₃ c₃
      c₁₂ : ComplexShape ι₁₂
      c₂₃ : ComplexShape ι₂₃
      c₄ : ComplexShape ι₄
      inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
      inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
      inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
      inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
      inst✝⁴ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
      inst✝³ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
      inst✝² : DecidableEq ι₂₃
      inst✝¹ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
      inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
      j j' : ι₄
      i₁ : ι₁
      i₂ : ι₂
      i₃ : ι₃
      h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
      i₂₃ : ι₂₃ := c₂.π c₃ c₂₃ { fst := i₂, snd := i₃ }
      h₁ : Not (c₂₃.Rel i₂₃ (c₂₃.next i₂₃))
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((F.obj (K₁.X i₁)).map (K₂.ιMapBifunc …
    -/
  · rw [mapBifunctor.d₂_eq_zero _ _ _ _ _ _ _ h₁, comp_zero]
    /-
      case neg
      C₁ : Type u_1
      C₂ : Type u_2
      C₂₃ : Type u_4
      C₃ : Type u_5
      C₄ : Type u_6
      inst✝²³ : CategoryTheory.Category.{u_14, u_1} C₁
      inst✝²² : CategoryTheory.Category.{u_16, u_2} C₂
      inst✝²¹ : CategoryTheory.Category.{u_17, u_5} C₃
      inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} C₄
      inst✝¹⁹ : CategoryTheory.Category.{u_15, u_4} C₂₃
      inst✝¹⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
      inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
      inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₃
      inst✝¹⁵ : CategoryTheory.Preadditive C₂₃
      inst✝¹⁴ : CategoryTheory.Preadditive C₄
      F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
      G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
      inst✝¹³ : G₂₃.PreservesZeroMorphisms
      inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
      inst✝¹¹ : F.PreservesZeroMorphisms
      inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
      ι₁ : Type u_7
      ι₂ : Type u_8
      ι₃ : Type u_9
      ι₁₂ : Type u_10
      ι₂₃ : Type u_11
      ι₄ : Type u_12
      inst✝⁹ : DecidableEq ι₄
      c₁ : ComplexShape ι₁
      c₂ : ComplexShape ι₂
      c₃ : ComplexShape ι₃
      K₁ : HomologicalComplex C₁ c₁
      K₂ : HomologicalComplex C₂ c₂
      K₃ : HomologicalComplex C₃ c₃
      c₁₂ : ComplexShape ι₁₂
      c₂₃ : ComplexShape ι₂₃
      c₄ : ComplexShape ι₄
      inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
      inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
      inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
      inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
      inst✝⁴ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
      inst✝³ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
      inst✝² : DecidableEq ι₂₃
      inst✝¹ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
      inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
      j j' : ι₄
      i₁ : ι₁
      i₂ : ι₂
      i₃ : ι₃
      h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
      i₂₃ : ι₂₃ := c₂.π c₃ c₂₃ { fst := i₂, snd := i₃ }
      h₁ : Not (c₂₃.Rel i₂₃ (c₂₃.next i₂₃))
      ⊢ Eq 0 (HAdd.hAdd (HomologicalComplex.mapBifunctor₂₃.d₂ F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ …
    -/
    trans 0 + 0
      /-
        C₁ : Type u_1
        C₂ : Type u_2
        C₂₃ : Type u_4
        C₃ : Type u_5
        C₄ : Type u_6
        inst✝²³ : CategoryTheory.Category.{u_14, u_1} C₁
        inst✝²² : CategoryTheory.Category.{u_16, u_2} C₂
        inst✝²¹ : CategoryTheory.Category.{u_17, u_5} C₃
        inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} C₄
        inst✝¹⁹ : CategoryTheory.Category.{u_15, u_4} C₂₃
        inst✝¹⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
        inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
        inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₃
        inst✝¹⁵ : CategoryTheory.Preadditive C₂₃
        inst✝¹⁴ : CategoryTheory.Preadditive C₄
        F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
        G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
        inst✝¹³ : G₂₃.PreservesZeroMorphisms
        inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
        inst✝¹¹ : F.PreservesZeroMorphisms
        inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
        ι₁ : Type u_7
        ι₂ : Type u_8
        ι₃ : Type u_9
        ι₁₂ : Type u_10
        ι₂₃ : Type u_11
        ι₄ : Type u_12
        inst✝⁹ : DecidableEq ι₄
        c₁ : ComplexShape ι₁
        c₂ : ComplexShape ι₂
        c₃ : ComplexShape ι₃
        K₁ : HomologicalComplex C₁ c₁
        K₂ : HomologicalComplex C₂ c₂
        K₃ : HomologicalComplex C₃ c₃
        c₁₂ : ComplexShape ι₁₂
        c₂₃ : ComplexShape ι₂₃
        c₄ : ComplexShape ι₄
        inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
        inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
        inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
        inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
        inst✝⁴ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
        inst✝³ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
        inst✝² : DecidableEq ι₂₃
        inst✝¹ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
        inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
        j j' : ι₄
        i₁ : ι₁
        i₂ : ι₂
        i₃ : ι₃
        h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
        i₂₃ : ι₂₃ := c₂.π c₃ c₂₃ { fst := i₂, snd := i₃ }
        h₁ : Not (c₂₃.Rel i₂₃ (c₂₃.next i₂₃))
        ⊢ Eq 0 (HAdd.hAdd 0 0)
      -/
    · simp only [add_zero]
      /-
        🎉 no goals
      -/
      /-
        C₁ : Type u_1
        C₂ : Type u_2
        C₂₃ : Type u_4
        C₃ : Type u_5
        C₄ : Type u_6
        inst✝²³ : CategoryTheory.Category.{u_14, u_1} C₁
        inst✝²² : CategoryTheory.Category.{u_16, u_2} C₂
        inst✝²¹ : CategoryTheory.Category.{u_17, u_5} C₃
        inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} C₄
        inst✝¹⁹ : CategoryTheory.Category.{u_15, u_4} C₂₃
        inst✝¹⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
        inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
        inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₃
        inst✝¹⁵ : CategoryTheory.Preadditive C₂₃
        inst✝¹⁴ : CategoryTheory.Preadditive C₄
        F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
        G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
        inst✝¹³ : G₂₃.PreservesZeroMorphisms
        inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
        inst✝¹¹ : F.PreservesZeroMorphisms
        inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
        ι₁ : Type u_7
        ι₂ : Type u_8
        ι₃ : Type u_9
        ι₁₂ : Type u_10
        ι₂₃ : Type u_11
        ι₄ : Type u_12
        inst✝⁹ : DecidableEq ι₄
        c₁ : ComplexShape ι₁
        c₂ : ComplexShape ι₂
        c₃ : ComplexShape ι₃
        K₁ : HomologicalComplex C₁ c₁
        K₂ : HomologicalComplex C₂ c₂
        K₃ : HomologicalComplex C₃ c₃
        c₁₂ : ComplexShape ι₁₂
        c₂₃ : ComplexShape ι₂₃
        c₄ : ComplexShape ι₄
        inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
        inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
        inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
        inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
        inst✝⁴ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
        inst✝³ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
        inst✝² : DecidableEq ι₂₃
        inst✝¹ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
        inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
        j j' : ι₄
        i₁ : ι₁
        i₂ : ι₂
        i₃ : ι₃
        h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
        i₂₃ : ι₂₃ := c₂.π c₃ c₂₃ { fst := i₂, snd := i₃ }
        h₁ : Not (c₂₃.Rel i₂₃ (c₂₃.next i₂₃))
        ⊢ Eq (HAdd.hAdd 0 0) (HAdd.hAdd (HomologicalComplex.mapBifunctor₂₃.d₂ F G₂₃ K₁ …
      -/
    · congr 1
        /-
          case e_a
          C₁ : Type u_1
          C₂ : Type u_2
          C₂₃ : Type u_4
          C₃ : Type u_5
          C₄ : Type u_6
          inst✝²³ : CategoryTheory.Category.{u_14, u_1} C₁
          inst✝²² : CategoryTheory.Category.{u_16, u_2} C₂
          inst✝²¹ : CategoryTheory.Category.{u_17, u_5} C₃
          inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} C₄
          inst✝¹⁹ : CategoryTheory.Category.{u_15, u_4} C₂₃
          inst✝¹⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
          inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
          inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₃
          inst✝¹⁵ : CategoryTheory.Preadditive C₂₃
          inst✝¹⁴ : CategoryTheory.Preadditive C₄
          F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
          G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
          inst✝¹³ : G₂₃.PreservesZeroMorphisms
          inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
          inst✝¹¹ : F.PreservesZeroMorphisms
          inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
          ι₁ : Type u_7
          ι₂ : Type u_8
          ι₃ : Type u_9
          ι₁₂ : Type u_10
          ι₂₃ : Type u_11
          ι₄ : Type u_12
          inst✝⁹ : DecidableEq ι₄
          c₁ : ComplexShape ι₁
          c₂ : ComplexShape ι₂
          c₃ : ComplexShape ι₃
          K₁ : HomologicalComplex C₁ c₁
          K₂ : HomologicalComplex C₂ c₂
          K₃ : HomologicalComplex C₃ c₃
          c₁₂ : ComplexShape ι₁₂
          c₂₃ : ComplexShape ι₂₃
          c₄ : ComplexShape ι₄
          inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
          inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
          inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
          inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
          inst✝⁴ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
          inst✝³ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
          inst✝² : DecidableEq ι₂₃
          inst✝¹ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
          inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
          j j' : ι₄
          i₁ : ι₁
          i₂ : ι₂
          i₃ : ι₃
          h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
          i₂₃ : ι₂₃ := c₂.π c₃ c₂₃ { fst := i₂, snd := i₃ }
          h₁ : Not (c₂₃.Rel i₂₃ (c₂₃.next i₂₃))
          ⊢ Eq 0 (HomologicalComplex.mapBifunctor₂₃.d₂ F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ i₁ i₂ i …
        -/
      · rw [d₂_eq_zero]
        /-
          case e_a.h
          C₁ : Type u_1
          C₂ : Type u_2
          C₂₃ : Type u_4
          C₃ : Type u_5
          C₄ : Type u_6
          inst✝²³ : CategoryTheory.Category.{u_14, u_1} C₁
          inst✝²² : CategoryTheory.Category.{u_16, u_2} C₂
          inst✝²¹ : CategoryTheory.Category.{u_17, u_5} C₃
          inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} C₄
          inst✝¹⁹ : CategoryTheory.Category.{u_15, u_4} C₂₃
          inst✝¹⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
          inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
          inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₃
          inst✝¹⁵ : CategoryTheory.Preadditive C₂₃
          inst✝¹⁴ : CategoryTheory.Preadditive C₄
          F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
          G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
          inst✝¹³ : G₂₃.PreservesZeroMorphisms
          inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
          inst✝¹¹ : F.PreservesZeroMorphisms
          inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
          ι₁ : Type u_7
          ι₂ : Type u_8
          ι₃ : Type u_9
          ι₁₂ : Type u_10
          ι₂₃ : Type u_11
          ι₄ : Type u_12
          inst✝⁹ : DecidableEq ι₄
          c₁ : ComplexShape ι₁
          c₂ : ComplexShape ι₂
          c₃ : ComplexShape ι₃
          K₁ : HomologicalComplex C₁ c₁
          K₂ : HomologicalComplex C₂ c₂
          K₃ : HomologicalComplex C₃ c₃
          c₁₂ : ComplexShape ι₁₂
          c₂₃ : ComplexShape ι₂₃
          c₄ : ComplexShape ι₄
          inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
          inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
          inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
          inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
          inst✝⁴ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
          inst✝³ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
          inst✝² : DecidableEq ι₂₃
          inst✝¹ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
          inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
          j j' : ι₄
          i₁ : ι₁
          i₂ : ι₂
          i₃ : ι₃
          h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
          i₂₃ : ι₂₃ := c₂.π c₃ c₂₃ { fst := i₂, snd := i₃ }
          h₁ : Not (c₂₃.Rel i₂₃ (c₂₃.next i₂₃))
          ⊢ Not (c₂.Rel i₂ (c₂.next i₂))
        -/
        intro h₂
        /-
          case e_a.h
          C₁ : Type u_1
          C₂ : Type u_2
          C₂₃ : Type u_4
          C₃ : Type u_5
          C₄ : Type u_6
          inst✝²³ : CategoryTheory.Category.{u_14, u_1} C₁
          inst✝²² : CategoryTheory.Category.{u_16, u_2} C₂
          inst✝²¹ : CategoryTheory.Category.{u_17, u_5} C₃
          inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} C₄
          inst✝¹⁹ : CategoryTheory.Category.{u_15, u_4} C₂₃
          inst✝¹⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
          inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
          inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₃
          inst✝¹⁵ : CategoryTheory.Preadditive C₂₃
          inst✝¹⁴ : CategoryTheory.Preadditive C₄
          F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
          G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
          inst✝¹³ : G₂₃.PreservesZeroMorphisms
          inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
          inst✝¹¹ : F.PreservesZeroMorphisms
          inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
          ι₁ : Type u_7
          ι₂ : Type u_8
          ι₃ : Type u_9
          ι₁₂ : Type u_10
          ι₂₃ : Type u_11
          ι₄ : Type u_12
          inst✝⁹ : DecidableEq ι₄
          c₁ : ComplexShape ι₁
          c₂ : ComplexShape ι₂
          c₃ : ComplexShape ι₃
          K₁ : HomologicalComplex C₁ c₁
          K₂ : HomologicalComplex C₂ c₂
          K₃ : HomologicalComplex C₃ c₃
          c₁₂ : ComplexShape ι₁₂
          c₂₃ : ComplexShape ι₂₃
          c₄ : ComplexShape ι₄
          inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
          inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
          inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
          inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
          inst✝⁴ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
          inst✝³ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
          inst✝² : DecidableEq ι₂₃
          inst✝¹ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
          inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
          j j' : ι₄
          i₁ : ι₁
          i₂ : ι₂
          i₃ : ι₃
          h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
          i₂₃ : ι₂₃ := c₂.π c₃ c₂₃ { fst := i₂, snd := i₃ }
          h₁ : Not (c₂₃.Rel i₂₃ (c₂₃.next i₂₃))
          h₂ : c₂.Rel i₂ (c₂.next i₂)
          ⊢ False
        -/
        apply h₁
        simpa only [← ComplexShape.next_π₁ c₃ c₂₃ h₂ i₃]
          using ComplexShape.rel_π₁ c₃ c₂₃ h₂ i₃
        /-
          case e_a
          C₁ : Type u_1
          C₂ : Type u_2
          C₂₃ : Type u_4
          C₃ : Type u_5
          C₄ : Type u_6
          inst✝²³ : CategoryTheory.Category.{u_14, u_1} C₁
          inst✝²² : CategoryTheory.Category.{u_16, u_2} C₂
          inst✝²¹ : CategoryTheory.Category.{u_17, u_5} C₃
          inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} C₄
          inst✝¹⁹ : CategoryTheory.Category.{u_15, u_4} C₂₃
          inst✝¹⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
          inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
          inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₃
          inst✝¹⁵ : CategoryTheory.Preadditive C₂₃
          inst✝¹⁴ : CategoryTheory.Preadditive C₄
          F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
          G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
          inst✝¹³ : G₂₃.PreservesZeroMorphisms
          inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
          inst✝¹¹ : F.PreservesZeroMorphisms
          inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
          ι₁ : Type u_7
          ι₂ : Type u_8
          ι₃ : Type u_9
          ι₁₂ : Type u_10
          ι₂₃ : Type u_11
          ι₄ : Type u_12
          inst✝⁹ : DecidableEq ι₄
          c₁ : ComplexShape ι₁
          c₂ : ComplexShape ι₂
          c₃ : ComplexShape ι₃
          K₁ : HomologicalComplex C₁ c₁
          K₂ : HomologicalComplex C₂ c₂
          K₃ : HomologicalComplex C₃ c₃
          c₁₂ : ComplexShape ι₁₂
          c₂₃ : ComplexShape ι₂₃
          c₄ : ComplexShape ι₄
          inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
          inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
          inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
          inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
          inst✝⁴ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
          inst✝³ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
          inst✝² : DecidableEq ι₂₃
          inst✝¹ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
          inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
          j j' : ι₄
          i₁ : ι₁
          i₂ : ι₂
          i₃ : ι₃
          h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
          i₂₃ : ι₂₃ := c₂.π c₃ c₂₃ { fst := i₂, snd := i₃ }
          h₁ : Not (c₂₃.Rel i₂₃ (c₂₃.next i₂₃))
          ⊢ Eq 0 (HomologicalComplex.mapBifunctor₂₃.d₃ F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ i₁ i₂ i …
        -/
      · rw [d₃_eq_zero]
        /-
          case e_a.h
          C₁ : Type u_1
          C₂ : Type u_2
          C₂₃ : Type u_4
          C₃ : Type u_5
          C₄ : Type u_6
          inst✝²³ : CategoryTheory.Category.{u_14, u_1} C₁
          inst✝²² : CategoryTheory.Category.{u_16, u_2} C₂
          inst✝²¹ : CategoryTheory.Category.{u_17, u_5} C₃
          inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} C₄
          inst✝¹⁹ : CategoryTheory.Category.{u_15, u_4} C₂₃
          inst✝¹⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
          inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
          inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₃
          inst✝¹⁵ : CategoryTheory.Preadditive C₂₃
          inst✝¹⁴ : CategoryTheory.Preadditive C₄
          F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
          G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
          inst✝¹³ : G₂₃.PreservesZeroMorphisms
          inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
          inst✝¹¹ : F.PreservesZeroMorphisms
          inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
          ι₁ : Type u_7
          ι₂ : Type u_8
          ι₃ : Type u_9
          ι₁₂ : Type u_10
          ι₂₃ : Type u_11
          ι₄ : Type u_12
          inst✝⁹ : DecidableEq ι₄
          c₁ : ComplexShape ι₁
          c₂ : ComplexShape ι₂
          c₃ : ComplexShape ι₃
          K₁ : HomologicalComplex C₁ c₁
          K₂ : HomologicalComplex C₂ c₂
          K₃ : HomologicalComplex C₃ c₃
          c₁₂ : ComplexShape ι₁₂
          c₂₃ : ComplexShape ι₂₃
          c₄ : ComplexShape ι₄
          inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
          inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
          inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
          inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
          inst✝⁴ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
          inst✝³ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
          inst✝² : DecidableEq ι₂₃
          inst✝¹ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
          inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
          j j' : ι₄
          i₁ : ι₁
          i₂ : ι₂
          i₃ : ι₃
          h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
          i₂₃ : ι₂₃ := c₂.π c₃ c₂₃ { fst := i₂, snd := i₃ }
          h₁ : Not (c₂₃.Rel i₂₃ (c₂₃.next i₂₃))
          ⊢ Not (c₃.Rel i₃ (c₃.next i₃))
        -/
        intro h₂
        /-
          case e_a.h
          C₁ : Type u_1
          C₂ : Type u_2
          C₂₃ : Type u_4
          C₃ : Type u_5
          C₄ : Type u_6
          inst✝²³ : CategoryTheory.Category.{u_14, u_1} C₁
          inst✝²² : CategoryTheory.Category.{u_16, u_2} C₂
          inst✝²¹ : CategoryTheory.Category.{u_17, u_5} C₃
          inst✝²⁰ : CategoryTheory.Category.{u_13, u_6} C₄
          inst✝¹⁹ : CategoryTheory.Category.{u_15, u_4} C₂₃
          inst✝¹⁸ : CategoryTheory.Limits.HasZeroMorphisms C₁
          inst✝¹⁷ : CategoryTheory.Limits.HasZeroMorphisms C₂
          inst✝¹⁶ : CategoryTheory.Limits.HasZeroMorphisms C₃
          inst✝¹⁵ : CategoryTheory.Preadditive C₂₃
          inst✝¹⁴ : CategoryTheory.Preadditive C₄
          F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
          G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
          inst✝¹³ : G₂₃.PreservesZeroMorphisms
          inst✝¹² : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
          inst✝¹¹ : F.PreservesZeroMorphisms
          inst✝¹⁰ : ∀ (X₁ : C₁), (F.obj X₁).Additive
          ι₁ : Type u_7
          ι₂ : Type u_8
          ι₃ : Type u_9
          ι₁₂ : Type u_10
          ι₂₃ : Type u_11
          ι₄ : Type u_12
          inst✝⁹ : DecidableEq ι₄
          c₁ : ComplexShape ι₁
          c₂ : ComplexShape ι₂
          c₃ : ComplexShape ι₃
          K₁ : HomologicalComplex C₁ c₁
          K₂ : HomologicalComplex C₂ c₂
          K₃ : HomologicalComplex C₃ c₃
          c₁₂ : ComplexShape ι₁₂
          c₂₃ : ComplexShape ι₂₃
          c₄ : ComplexShape ι₄
          inst✝⁸ : TotalComplexShape c₁ c₂ c₁₂
          inst✝⁷ : TotalComplexShape c₁₂ c₃ c₄
          inst✝⁶ : TotalComplexShape c₂ c₃ c₂₃
          inst✝⁵ : TotalComplexShape c₁ c₂₃ c₄
          inst✝⁴ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
          inst✝³ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
          inst✝² : DecidableEq ι₂₃
          inst✝¹ : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
          inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
          j j' : ι₄
          i₁ : ι₁
          i₂ : ι₂
          i₃ : ι₃
          h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
          i₂₃ : ι₂₃ := c₂.π c₃ c₂₃ { fst := i₂, snd := i₃ }
          h₁ : Not (c₂₃.Rel i₂₃ (c₂₃.next i₂₃))
          h₂ : c₃.Rel i₃ (c₃.next i₃)
          ⊢ False
        -/
        apply h₁
        simpa only [i₂₃, ComplexShape.next_π₂ c₂ c₂₃ i₂ h₂]
          using ComplexShape.rel_π₂ c₂ c₂₃ i₂ h₂


@[reassoc (attr := simp)]
lemma ι_mapBifunctorAssociatorX_hom (i₁ : ι₁) (i₂ : ι₂) (i₃ : ι₃) (j : ι₄)
    (h : ComplexShape.r c₁ c₂ c₃ c₁₂ c₄ (i₁, i₂, i₃) = j) :
    mapBifunctor₁₂.ι F₁₂ G K₁ K₂ K₃ c₁₂ c₄ i₁ i₂ i₃ j h ≫
    (mapBifunctorAssociatorX associator K₁ K₂ K₃ c₁₂ c₂₃ c₄ j).hom =
      ((associator.hom.app (K₁.X i₁)).app (K₂.X i₂)).app (K₃.X i₃) ≫
        mapBifunctor₂₃.ι F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ i₁ i₂ i₃ j h := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₁₂ : Type u_3
    C₂₃ : Type u_4
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝³³ : CategoryTheory.Category.{u_17, u_1} C₁
    inst✝³² : CategoryTheory.Category.{u_16, u_2} C₂
    inst✝³¹ : CategoryTheory.Category.{u_14, u_5} C₃
    inst✝³⁰ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝²⁹ : CategoryTheory.Category.{u_15, u_3} C₁₂
    inst✝²⁸ : CategoryTheory.Category.{u_18, u_4} C₂₃
    inst✝²⁷ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝²⁶ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝²⁵ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝²⁴ : CategoryTheory.Preadditive C₁₂
    inst✝²³ : CategoryTheory.Preadditive C₂₃
    inst✝²² : CategoryTheory.Preadditive C₄
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    inst✝²¹ : F₁₂.PreservesZeroMorphisms
    inst✝²⁰ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
    inst✝¹⁹ : G.Additive
    inst✝¹⁸ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
    inst✝¹⁷ : G₂₃.PreservesZeroMorphisms
    inst✝¹⁶ : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
    inst✝¹⁵ : F.PreservesZeroMorphisms
    inst✝¹⁴ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    associator : CategoryTheory.Iso (CategoryTheory.bifunctorComp₁₂ F₁₂ G) (Catego …
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₂₃ : Type u_11
    ι₄ : Type u_12
    inst✝¹³ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₂₃ : ComplexShape ι₂₃
    c₄ : ComplexShape ι₄
    inst✝¹² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹¹ : TotalComplexShape c₁₂ c₃ c₄
    inst✝¹⁰ : TotalComplexShape c₂ c₃ c₂₃
    inst✝⁹ : TotalComplexShape c₁ c₂₃ c₄
    inst✝⁸ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
    inst✝⁷ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
    inst✝⁶ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
    inst✝⁵ : DecidableEq ι₁₂
    inst✝⁴ : DecidableEq ι₂₃
    inst✝³ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
    inst✝² : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
    inst✝¹ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
    inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
    i₁ : ι₁
    i₂ : ι₂
    i₃ : ι₃
    j : ι₄
    h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.mapBifunctor₁₂.ι  …
  -/
  apply GradedObject.ι_mapBifunctorAssociator_hom
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma ιOrZero_mapBifunctorAssociatorX_hom (i₁ : ι₁) (i₂ : ι₂) (i₃ : ι₃) (j : ι₄) :
    mapBifunctor₁₂.ιOrZero F₁₂ G K₁ K₂ K₃ c₁₂ c₄ i₁ i₂ i₃ j ≫
    (mapBifunctorAssociatorX associator K₁ K₂ K₃ c₁₂ c₂₃ c₄ j).hom =
      ((associator.hom.app (K₁.X i₁)).app (K₂.X i₂)).app (K₃.X i₃) ≫
        mapBifunctor₂₃.ιOrZero F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ i₁ i₂ i₃ j := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₁₂ : Type u_3
    C₂₃ : Type u_4
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝³³ : CategoryTheory.Category.{u_17, u_1} C₁
    inst✝³² : CategoryTheory.Category.{u_16, u_2} C₂
    inst✝³¹ : CategoryTheory.Category.{u_14, u_5} C₃
    inst✝³⁰ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝²⁹ : CategoryTheory.Category.{u_15, u_3} C₁₂
    inst✝²⁸ : CategoryTheory.Category.{u_18, u_4} C₂₃
    inst✝²⁷ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝²⁶ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝²⁵ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝²⁴ : CategoryTheory.Preadditive C₁₂
    inst✝²³ : CategoryTheory.Preadditive C₂₃
    inst✝²² : CategoryTheory.Preadditive C₄
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    inst✝²¹ : F₁₂.PreservesZeroMorphisms
    inst✝²⁰ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
    inst✝¹⁹ : G.Additive
    inst✝¹⁸ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
    inst✝¹⁷ : G₂₃.PreservesZeroMorphisms
    inst✝¹⁶ : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
    inst✝¹⁵ : F.PreservesZeroMorphisms
    inst✝¹⁴ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    associator : CategoryTheory.Iso (CategoryTheory.bifunctorComp₁₂ F₁₂ G) (Catego …
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₂₃ : Type u_11
    ι₄ : Type u_12
    inst✝¹³ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₂₃ : ComplexShape ι₂₃
    c₄ : ComplexShape ι₄
    inst✝¹² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹¹ : TotalComplexShape c₁₂ c₃ c₄
    inst✝¹⁰ : TotalComplexShape c₂ c₃ c₂₃
    inst✝⁹ : TotalComplexShape c₁ c₂₃ c₄
    inst✝⁸ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
    inst✝⁷ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
    inst✝⁶ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
    inst✝⁵ : DecidableEq ι₁₂
    inst✝⁴ : DecidableEq ι₂₃
    inst✝³ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
    inst✝² : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
    inst✝¹ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
    inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
    i₁ : ι₁
    i₂ : ι₂
    i₃ : ι₃
    j : ι₄
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.mapBifunctor₁₂.ιO …
  -/
  by_cases h : ComplexShape.r c₁ c₂ c₃ c₁₂ c₄ (i₁, i₂, i₃) = j
  · rw [mapBifunctor₁₂.ιOrZero_eq _ _ _ _ _ _ _ _ _ _ _ h,
      mapBifunctor₂₃.ιOrZero_eq _ _ _ _ _ _ _ _ _ _ _ _ h,
      ι_mapBifunctorAssociatorX_hom]
  · rw [mapBifunctor₁₂.ιOrZero_eq_zero _ _ _ _ _ _ _ _ _ _ _ h,
      mapBifunctor₂₃.ιOrZero_eq_zero _ _ _ _ _ _ _ _ _ _ _ _ h,
      zero_comp, comp_zero]


@[reassoc]
lemma mapBifunctorAssociatorX_hom_D₁ (j j' : ι₄) :
    (mapBifunctorAssociatorX associator K₁ K₂ K₃ c₁₂ c₂₃ c₄ j).hom ≫
      mapBifunctor₂₃.D₁ F G₂₃ K₁ K₂ K₃ c₂₃ c₄ j j' =
        mapBifunctor₁₂.D₁ F₁₂ G K₁ K₂ K₃ c₁₂ c₄ j j' ≫
        (mapBifunctorAssociatorX associator K₁ K₂ K₃ c₁₂ c₂₃ c₄ j').hom := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₁₂ : Type u_3
    C₂₃ : Type u_4
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝³³ : CategoryTheory.Category.{u_16, u_1} C₁
    inst✝³² : CategoryTheory.Category.{u_17, u_2} C₂
    inst✝³¹ : CategoryTheory.Category.{u_15, u_5} C₃
    inst✝³⁰ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝²⁹ : CategoryTheory.Category.{u_14, u_3} C₁₂
    inst✝²⁸ : CategoryTheory.Category.{u_18, u_4} C₂₃
    inst✝²⁷ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝²⁶ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝²⁵ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝²⁴ : CategoryTheory.Preadditive C₁₂
    inst✝²³ : CategoryTheory.Preadditive C₂₃
    inst✝²² : CategoryTheory.Preadditive C₄
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    inst✝²¹ : F₁₂.PreservesZeroMorphisms
    inst✝²⁰ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
    inst✝¹⁹ : G.Additive
    inst✝¹⁸ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
    inst✝¹⁷ : G₂₃.PreservesZeroMorphisms
    inst✝¹⁶ : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
    inst✝¹⁵ : F.PreservesZeroMorphisms
    inst✝¹⁴ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    associator : CategoryTheory.Iso (CategoryTheory.bifunctorComp₁₂ F₁₂ G) (Catego …
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₂₃ : Type u_11
    ι₄ : Type u_12
    inst✝¹³ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₂₃ : ComplexShape ι₂₃
    c₄ : ComplexShape ι₄
    inst✝¹² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹¹ : TotalComplexShape c₁₂ c₃ c₄
    inst✝¹⁰ : TotalComplexShape c₂ c₃ c₂₃
    inst✝⁹ : TotalComplexShape c₁ c₂₃ c₄
    inst✝⁸ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
    inst✝⁷ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
    inst✝⁶ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
    inst✝⁵ : DecidableEq ι₁₂
    inst✝⁴ : DecidableEq ι₂₃
    inst✝³ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
    inst✝² : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
    inst✝¹ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
    inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
    j j' : ι₄
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.mapBifunctorAssoc …
  -/
  ext i₁ i₂ i₃ h
  /-
    case hfg
    C₁ : Type u_1
    C₂ : Type u_2
    C₁₂ : Type u_3
    C₂₃ : Type u_4
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝³³ : CategoryTheory.Category.{u_16, u_1} C₁
    inst✝³² : CategoryTheory.Category.{u_17, u_2} C₂
    inst✝³¹ : CategoryTheory.Category.{u_15, u_5} C₃
    inst✝³⁰ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝²⁹ : CategoryTheory.Category.{u_14, u_3} C₁₂
    inst✝²⁸ : CategoryTheory.Category.{u_18, u_4} C₂₃
    inst✝²⁷ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝²⁶ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝²⁵ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝²⁴ : CategoryTheory.Preadditive C₁₂
    inst✝²³ : CategoryTheory.Preadditive C₂₃
    inst✝²² : CategoryTheory.Preadditive C₄
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    inst✝²¹ : F₁₂.PreservesZeroMorphisms
    inst✝²⁰ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
    inst✝¹⁹ : G.Additive
    inst✝¹⁸ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
    inst✝¹⁷ : G₂₃.PreservesZeroMorphisms
    inst✝¹⁶ : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
    inst✝¹⁵ : F.PreservesZeroMorphisms
    inst✝¹⁴ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    associator : CategoryTheory.Iso (CategoryTheory.bifunctorComp₁₂ F₁₂ G) (Catego …
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₂₃ : Type u_11
    ι₄ : Type u_12
    inst✝¹³ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₂₃ : ComplexShape ι₂₃
    c₄ : ComplexShape ι₄
    inst✝¹² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹¹ : TotalComplexShape c₁₂ c₃ c₄
    inst✝¹⁰ : TotalComplexShape c₂ c₃ c₂₃
    inst✝⁹ : TotalComplexShape c₁ c₂₃ c₄
    inst✝⁸ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
    inst✝⁷ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
    inst✝⁶ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
    inst✝⁵ : DecidableEq ι₁₂
    inst✝⁴ : DecidableEq ι₂₃
    inst✝³ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
    inst✝² : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
    inst✝¹ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
    inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
    j j' : ι₄
    i₁ : ι₁
    i₂ : ι₂
    i₃ : ι₃
    h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.mapBifunctor₁₂.ι  …
  -/
  rw [mapBifunctor₁₂.ι_D₁_assoc, ι_mapBifunctorAssociatorX_hom_assoc, mapBifunctor₂₃.ι_D₁]
  /-
    case hfg
    C₁ : Type u_1
    C₂ : Type u_2
    C₁₂ : Type u_3
    C₂₃ : Type u_4
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝³³ : CategoryTheory.Category.{u_16, u_1} C₁
    inst✝³² : CategoryTheory.Category.{u_17, u_2} C₂
    inst✝³¹ : CategoryTheory.Category.{u_15, u_5} C₃
    inst✝³⁰ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝²⁹ : CategoryTheory.Category.{u_14, u_3} C₁₂
    inst✝²⁸ : CategoryTheory.Category.{u_18, u_4} C₂₃
    inst✝²⁷ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝²⁶ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝²⁵ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝²⁴ : CategoryTheory.Preadditive C₁₂
    inst✝²³ : CategoryTheory.Preadditive C₂₃
    inst✝²² : CategoryTheory.Preadditive C₄
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    inst✝²¹ : F₁₂.PreservesZeroMorphisms
    inst✝²⁰ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
    inst✝¹⁹ : G.Additive
    inst✝¹⁸ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
    inst✝¹⁷ : G₂₃.PreservesZeroMorphisms
    inst✝¹⁶ : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
    inst✝¹⁵ : F.PreservesZeroMorphisms
    inst✝¹⁴ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    associator : CategoryTheory.Iso (CategoryTheory.bifunctorComp₁₂ F₁₂ G) (Catego …
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₂₃ : Type u_11
    ι₄ : Type u_12
    inst✝¹³ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₂₃ : ComplexShape ι₂₃
    c₄ : ComplexShape ι₄
    inst✝¹² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹¹ : TotalComplexShape c₁₂ c₃ c₄
    inst✝¹⁰ : TotalComplexShape c₂ c₃ c₂₃
    inst✝⁹ : TotalComplexShape c₁ c₂₃ c₄
    inst✝⁸ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
    inst✝⁷ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
    inst✝⁶ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
    inst✝⁵ : DecidableEq ι₁₂
    inst✝⁴ : DecidableEq ι₂₃
    inst✝³ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
    inst✝² : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
    inst✝¹ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
    inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
    j j' : ι₄
    i₁ : ι₁
    i₂ : ι₂
    i₃ : ι₃
    h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((associator.hom.app (K₁.X i₁)).app  …
  -/
  by_cases h₁ : c₁.Rel i₁ (c₁.next i₁)
  · have := NatTrans.naturality_app_app associator.hom
      (K₁.d i₁ (c₁.next i₁)) (K₂.X i₂) (K₃.X i₃)
    /-
      case pos
      C₁ : Type u_1
      C₂ : Type u_2
      C₁₂ : Type u_3
      C₂₃ : Type u_4
      C₃ : Type u_5
      C₄ : Type u_6
      inst✝³³ : CategoryTheory.Category.{u_16, u_1} C₁
      inst✝³² : CategoryTheory.Category.{u_17, u_2} C₂
      inst✝³¹ : CategoryTheory.Category.{u_15, u_5} C₃
      inst✝³⁰ : CategoryTheory.Category.{u_13, u_6} C₄
      inst✝²⁹ : CategoryTheory.Category.{u_14, u_3} C₁₂
      inst✝²⁸ : CategoryTheory.Category.{u_18, u_4} C₂₃
      inst✝²⁷ : CategoryTheory.Limits.HasZeroMorphisms C₁
      inst✝²⁶ : CategoryTheory.Limits.HasZeroMorphisms C₂
      inst✝²⁵ : CategoryTheory.Limits.HasZeroMorphisms C₃
      inst✝²⁴ : CategoryTheory.Preadditive C₁₂
      inst✝²³ : CategoryTheory.Preadditive C₂₃
      inst✝²² : CategoryTheory.Preadditive C₄
      F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
      G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
      F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
      G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
      inst✝²¹ : F₁₂.PreservesZeroMorphisms
      inst✝²⁰ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
      inst✝¹⁹ : G.Additive
      inst✝¹⁸ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
      inst✝¹⁷ : G₂₃.PreservesZeroMorphisms
      inst✝¹⁶ : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
      inst✝¹⁵ : F.PreservesZeroMorphisms
      inst✝¹⁴ : ∀ (X₁ : C₁), (F.obj X₁).Additive
      associator : CategoryTheory.Iso (CategoryTheory.bifunctorComp₁₂ F₁₂ G) (Catego …
      ι₁ : Type u_7
      ι₂ : Type u_8
      ι₃ : Type u_9
      ι₁₂ : Type u_10
      ι₂₃ : Type u_11
      ι₄ : Type u_12
      inst✝¹³ : DecidableEq ι₄
      c₁ : ComplexShape ι₁
      c₂ : ComplexShape ι₂
      c₃ : ComplexShape ι₃
      K₁ : HomologicalComplex C₁ c₁
      K₂ : HomologicalComplex C₂ c₂
      K₃ : HomologicalComplex C₃ c₃
      c₁₂ : ComplexShape ι₁₂
      c₂₃ : ComplexShape ι₂₃
      c₄ : ComplexShape ι₄
      inst✝¹² : TotalComplexShape c₁ c₂ c₁₂
      inst✝¹¹ : TotalComplexShape c₁₂ c₃ c₄
      inst✝¹⁰ : TotalComplexShape c₂ c₃ c₂₃
      inst✝⁹ : TotalComplexShape c₁ c₂₃ c₄
      inst✝⁸ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
      inst✝⁷ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
      inst✝⁶ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
      inst✝⁵ : DecidableEq ι₁₂
      inst✝⁴ : DecidableEq ι₂₃
      inst✝³ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
      inst✝² : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
      inst✝¹ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
      inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
      j j' : ι₄
      i₁ : ι₁
      i₂ : ι₂
      i₃ : ι₃
      h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
      h₁ : c₁.Rel i₁ (c₁.next i₁)
      this : Eq (CategoryTheory.CategoryStruct.comp ((((CategoryTheory.bifunctorComp …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((associator.hom.app (K₁.X i₁)).app  …
    -/
    dsimp at this
    rw [mapBifunctor₁₂.d₁_eq _ _ _ _ _ _ _ h₁, mapBifunctor₂₃.d₁_eq _ _ _ _ _ _ _ _ h₁,
      Linear.comp_units_smul, Linear.units_smul_comp, assoc,
        ComplexShape.associative_ε₁_eq_mul c₁ c₂ c₃ c₁₂ c₂₃ c₄,
      ιOrZero_mapBifunctorAssociatorX_hom, smul_left_cancel_iff,
      reassoc_of% this]
  · rw [mapBifunctor₁₂.d₁_eq_zero _ _ _ _ _ _ _ _ _ _ _ h₁,
      mapBifunctor₂₃.d₁_eq_zero _ _ _ _ _ _ _ _ _ _ _ _ h₁, comp_zero, zero_comp]


@[reassoc]
lemma mapBifunctorAssociatorX_hom_D₂ (j j' : ι₄) :
    (mapBifunctorAssociatorX associator K₁ K₂ K₃ c₁₂ c₂₃ c₄ j).hom ≫
      mapBifunctor₂₃.D₂ F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ j j' =
        mapBifunctor₁₂.D₂ F₁₂ G K₁ K₂ K₃ c₁₂ c₄ j j' ≫
        (mapBifunctorAssociatorX associator K₁ K₂ K₃ c₁₂ c₂₃ c₄ j').hom := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₁₂ : Type u_3
    C₂₃ : Type u_4
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝³³ : CategoryTheory.Category.{u_16, u_1} C₁
    inst✝³² : CategoryTheory.Category.{u_17, u_2} C₂
    inst✝³¹ : CategoryTheory.Category.{u_15, u_5} C₃
    inst✝³⁰ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝²⁹ : CategoryTheory.Category.{u_14, u_3} C₁₂
    inst✝²⁸ : CategoryTheory.Category.{u_18, u_4} C₂₃
    inst✝²⁷ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝²⁶ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝²⁵ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝²⁴ : CategoryTheory.Preadditive C₁₂
    inst✝²³ : CategoryTheory.Preadditive C₂₃
    inst✝²² : CategoryTheory.Preadditive C₄
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    inst✝²¹ : F₁₂.PreservesZeroMorphisms
    inst✝²⁰ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
    inst✝¹⁹ : G.Additive
    inst✝¹⁸ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
    inst✝¹⁷ : G₂₃.PreservesZeroMorphisms
    inst✝¹⁶ : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
    inst✝¹⁵ : F.PreservesZeroMorphisms
    inst✝¹⁴ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    associator : CategoryTheory.Iso (CategoryTheory.bifunctorComp₁₂ F₁₂ G) (Catego …
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₂₃ : Type u_11
    ι₄ : Type u_12
    inst✝¹³ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₂₃ : ComplexShape ι₂₃
    c₄ : ComplexShape ι₄
    inst✝¹² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹¹ : TotalComplexShape c₁₂ c₃ c₄
    inst✝¹⁰ : TotalComplexShape c₂ c₃ c₂₃
    inst✝⁹ : TotalComplexShape c₁ c₂₃ c₄
    inst✝⁸ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
    inst✝⁷ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
    inst✝⁶ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
    inst✝⁵ : DecidableEq ι₁₂
    inst✝⁴ : DecidableEq ι₂₃
    inst✝³ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
    inst✝² : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
    inst✝¹ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
    inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
    j j' : ι₄
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.mapBifunctorAssoc …
  -/
  ext i₁ i₂ i₃ h
  /-
    case hfg
    C₁ : Type u_1
    C₂ : Type u_2
    C₁₂ : Type u_3
    C₂₃ : Type u_4
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝³³ : CategoryTheory.Category.{u_16, u_1} C₁
    inst✝³² : CategoryTheory.Category.{u_17, u_2} C₂
    inst✝³¹ : CategoryTheory.Category.{u_15, u_5} C₃
    inst✝³⁰ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝²⁹ : CategoryTheory.Category.{u_14, u_3} C₁₂
    inst✝²⁸ : CategoryTheory.Category.{u_18, u_4} C₂₃
    inst✝²⁷ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝²⁶ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝²⁵ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝²⁴ : CategoryTheory.Preadditive C₁₂
    inst✝²³ : CategoryTheory.Preadditive C₂₃
    inst✝²² : CategoryTheory.Preadditive C₄
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    inst✝²¹ : F₁₂.PreservesZeroMorphisms
    inst✝²⁰ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
    inst✝¹⁹ : G.Additive
    inst✝¹⁸ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
    inst✝¹⁷ : G₂₃.PreservesZeroMorphisms
    inst✝¹⁶ : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
    inst✝¹⁵ : F.PreservesZeroMorphisms
    inst✝¹⁴ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    associator : CategoryTheory.Iso (CategoryTheory.bifunctorComp₁₂ F₁₂ G) (Catego …
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₂₃ : Type u_11
    ι₄ : Type u_12
    inst✝¹³ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₂₃ : ComplexShape ι₂₃
    c₄ : ComplexShape ι₄
    inst✝¹² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹¹ : TotalComplexShape c₁₂ c₃ c₄
    inst✝¹⁰ : TotalComplexShape c₂ c₃ c₂₃
    inst✝⁹ : TotalComplexShape c₁ c₂₃ c₄
    inst✝⁸ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
    inst✝⁷ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
    inst✝⁶ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
    inst✝⁵ : DecidableEq ι₁₂
    inst✝⁴ : DecidableEq ι₂₃
    inst✝³ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
    inst✝² : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
    inst✝¹ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
    inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
    j j' : ι₄
    i₁ : ι₁
    i₂ : ι₂
    i₃ : ι₃
    h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.mapBifunctor₁₂.ι  …
  -/
  rw [mapBifunctor₁₂.ι_D₂_assoc, ι_mapBifunctorAssociatorX_hom_assoc, mapBifunctor₂₃.ι_D₂]
  /-
    case hfg
    C₁ : Type u_1
    C₂ : Type u_2
    C₁₂ : Type u_3
    C₂₃ : Type u_4
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝³³ : CategoryTheory.Category.{u_16, u_1} C₁
    inst✝³² : CategoryTheory.Category.{u_17, u_2} C₂
    inst✝³¹ : CategoryTheory.Category.{u_15, u_5} C₃
    inst✝³⁰ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝²⁹ : CategoryTheory.Category.{u_14, u_3} C₁₂
    inst✝²⁸ : CategoryTheory.Category.{u_18, u_4} C₂₃
    inst✝²⁷ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝²⁶ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝²⁵ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝²⁴ : CategoryTheory.Preadditive C₁₂
    inst✝²³ : CategoryTheory.Preadditive C₂₃
    inst✝²² : CategoryTheory.Preadditive C₄
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    inst✝²¹ : F₁₂.PreservesZeroMorphisms
    inst✝²⁰ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
    inst✝¹⁹ : G.Additive
    inst✝¹⁸ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
    inst✝¹⁷ : G₂₃.PreservesZeroMorphisms
    inst✝¹⁶ : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
    inst✝¹⁵ : F.PreservesZeroMorphisms
    inst✝¹⁴ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    associator : CategoryTheory.Iso (CategoryTheory.bifunctorComp₁₂ F₁₂ G) (Catego …
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₂₃ : Type u_11
    ι₄ : Type u_12
    inst✝¹³ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₂₃ : ComplexShape ι₂₃
    c₄ : ComplexShape ι₄
    inst✝¹² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹¹ : TotalComplexShape c₁₂ c₃ c₄
    inst✝¹⁰ : TotalComplexShape c₂ c₃ c₂₃
    inst✝⁹ : TotalComplexShape c₁ c₂₃ c₄
    inst✝⁸ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
    inst✝⁷ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
    inst✝⁶ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
    inst✝⁵ : DecidableEq ι₁₂
    inst✝⁴ : DecidableEq ι₂₃
    inst✝³ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
    inst✝² : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
    inst✝¹ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
    inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
    j j' : ι₄
    i₁ : ι₁
    i₂ : ι₂
    i₃ : ι₃
    h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((associator.hom.app (K₁.X i₁)).app  …
  -/
  by_cases h₁ : c₂.Rel i₂ (c₂.next i₂)
    /-
      case pos
      C₁ : Type u_1
      C₂ : Type u_2
      C₁₂ : Type u_3
      C₂₃ : Type u_4
      C₃ : Type u_5
      C₄ : Type u_6
      inst✝³³ : CategoryTheory.Category.{u_16, u_1} C₁
      inst✝³² : CategoryTheory.Category.{u_17, u_2} C₂
      inst✝³¹ : CategoryTheory.Category.{u_15, u_5} C₃
      inst✝³⁰ : CategoryTheory.Category.{u_13, u_6} C₄
      inst✝²⁹ : CategoryTheory.Category.{u_14, u_3} C₁₂
      inst✝²⁸ : CategoryTheory.Category.{u_18, u_4} C₂₃
      inst✝²⁷ : CategoryTheory.Limits.HasZeroMorphisms C₁
      inst✝²⁶ : CategoryTheory.Limits.HasZeroMorphisms C₂
      inst✝²⁵ : CategoryTheory.Limits.HasZeroMorphisms C₃
      inst✝²⁴ : CategoryTheory.Preadditive C₁₂
      inst✝²³ : CategoryTheory.Preadditive C₂₃
      inst✝²² : CategoryTheory.Preadditive C₄
      F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
      G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
      F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
      G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
      inst✝²¹ : F₁₂.PreservesZeroMorphisms
      inst✝²⁰ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
      inst✝¹⁹ : G.Additive
      inst✝¹⁸ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
      inst✝¹⁷ : G₂₃.PreservesZeroMorphisms
      inst✝¹⁶ : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
      inst✝¹⁵ : F.PreservesZeroMorphisms
      inst✝¹⁴ : ∀ (X₁ : C₁), (F.obj X₁).Additive
      associator : CategoryTheory.Iso (CategoryTheory.bifunctorComp₁₂ F₁₂ G) (Catego …
      ι₁ : Type u_7
      ι₂ : Type u_8
      ι₃ : Type u_9
      ι₁₂ : Type u_10
      ι₂₃ : Type u_11
      ι₄ : Type u_12
      inst✝¹³ : DecidableEq ι₄
      c₁ : ComplexShape ι₁
      c₂ : ComplexShape ι₂
      c₃ : ComplexShape ι₃
      K₁ : HomologicalComplex C₁ c₁
      K₂ : HomologicalComplex C₂ c₂
      K₃ : HomologicalComplex C₃ c₃
      c₁₂ : ComplexShape ι₁₂
      c₂₃ : ComplexShape ι₂₃
      c₄ : ComplexShape ι₄
      inst✝¹² : TotalComplexShape c₁ c₂ c₁₂
      inst✝¹¹ : TotalComplexShape c₁₂ c₃ c₄
      inst✝¹⁰ : TotalComplexShape c₂ c₃ c₂₃
      inst✝⁹ : TotalComplexShape c₁ c₂₃ c₄
      inst✝⁸ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
      inst✝⁷ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
      inst✝⁶ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
      inst✝⁵ : DecidableEq ι₁₂
      inst✝⁴ : DecidableEq ι₂₃
      inst✝³ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
      inst✝² : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
      inst✝¹ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
      inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
      j j' : ι₄
      i₁ : ι₁
      i₂ : ι₂
      i₃ : ι₃
      h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
      h₁ : c₂.Rel i₂ (c₂.next i₂)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((associator.hom.app (K₁.X i₁)).app  …
    -/
  · have := NatTrans.naturality_app (associator.hom.app (K₁.X i₁)) (K₃.X i₃) (K₂.d i₂ (c₂.next i₂))
    /-
      case pos
      C₁ : Type u_1
      C₂ : Type u_2
      C₁₂ : Type u_3
      C₂₃ : Type u_4
      C₃ : Type u_5
      C₄ : Type u_6
      inst✝³³ : CategoryTheory.Category.{u_16, u_1} C₁
      inst✝³² : CategoryTheory.Category.{u_17, u_2} C₂
      inst✝³¹ : CategoryTheory.Category.{u_15, u_5} C₃
      inst✝³⁰ : CategoryTheory.Category.{u_13, u_6} C₄
      inst✝²⁹ : CategoryTheory.Category.{u_14, u_3} C₁₂
      inst✝²⁸ : CategoryTheory.Category.{u_18, u_4} C₂₃
      inst✝²⁷ : CategoryTheory.Limits.HasZeroMorphisms C₁
      inst✝²⁶ : CategoryTheory.Limits.HasZeroMorphisms C₂
      inst✝²⁵ : CategoryTheory.Limits.HasZeroMorphisms C₃
      inst✝²⁴ : CategoryTheory.Preadditive C₁₂
      inst✝²³ : CategoryTheory.Preadditive C₂₃
      inst✝²² : CategoryTheory.Preadditive C₄
      F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
      G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
      F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
      G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
      inst✝²¹ : F₁₂.PreservesZeroMorphisms
      inst✝²⁰ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
      inst✝¹⁹ : G.Additive
      inst✝¹⁸ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
      inst✝¹⁷ : G₂₃.PreservesZeroMorphisms
      inst✝¹⁶ : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
      inst✝¹⁵ : F.PreservesZeroMorphisms
      inst✝¹⁴ : ∀ (X₁ : C₁), (F.obj X₁).Additive
      associator : CategoryTheory.Iso (CategoryTheory.bifunctorComp₁₂ F₁₂ G) (Catego …
      ι₁ : Type u_7
      ι₂ : Type u_8
      ι₃ : Type u_9
      ι₁₂ : Type u_10
      ι₂₃ : Type u_11
      ι₄ : Type u_12
      inst✝¹³ : DecidableEq ι₄
      c₁ : ComplexShape ι₁
      c₂ : ComplexShape ι₂
      c₃ : ComplexShape ι₃
      K₁ : HomologicalComplex C₁ c₁
      K₂ : HomologicalComplex C₂ c₂
      K₃ : HomologicalComplex C₃ c₃
      c₁₂ : ComplexShape ι₁₂
      c₂₃ : ComplexShape ι₂₃
      c₄ : ComplexShape ι₄
      inst✝¹² : TotalComplexShape c₁ c₂ c₁₂
      inst✝¹¹ : TotalComplexShape c₁₂ c₃ c₄
      inst✝¹⁰ : TotalComplexShape c₂ c₃ c₂₃
      inst✝⁹ : TotalComplexShape c₁ c₂₃ c₄
      inst✝⁸ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
      inst✝⁷ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
      inst✝⁶ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
      inst✝⁵ : DecidableEq ι₁₂
      inst✝⁴ : DecidableEq ι₂₃
      inst✝³ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
      inst✝² : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
      inst✝¹ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
      inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
      j j' : ι₄
      i₁ : ι₁
      i₂ : ι₂
      i₃ : ι₃
      h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
      h₁ : c₂.Rel i₂ (c₂.next i₂)
      this : Eq (CategoryTheory.CategoryStruct.comp ((((CategoryTheory.bifunctorComp …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((associator.hom.app (K₁.X i₁)).app  …
    -/
    dsimp at this
    rw [mapBifunctor₁₂.d₂_eq _ _ _ _ _ _ _ _ h₁, mapBifunctor₂₃.d₂_eq _ _ _ _ _ _ _ _ _ h₁,
      Linear.units_smul_comp, assoc, ιOrZero_mapBifunctorAssociatorX_hom,
      reassoc_of% this, Linear.comp_units_smul,
      ComplexShape.associative_ε₂_ε₁ c₁ c₂ c₃ c₁₂ c₂₃ c₄]
  · rw [mapBifunctor₁₂.d₂_eq_zero _ _ _ _ _ _ _ _ _ _ _ h₁,
      mapBifunctor₂₃.d₂_eq_zero _ _ _ _ _ _ _ _ _ _ _ _ h₁, comp_zero, zero_comp]


@[reassoc]
lemma mapBifunctorAssociatorX_hom_D₃ (j j' : ι₄) :
    (mapBifunctorAssociatorX associator K₁ K₂ K₃ c₁₂ c₂₃ c₄ j).hom ≫
      mapBifunctor₂₃.D₃ F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄ j j' =
        mapBifunctor₁₂.D₃ F₁₂ G K₁ K₂ K₃ c₁₂ c₄ j j' ≫
        (mapBifunctorAssociatorX associator K₁ K₂ K₃ c₁₂ c₂₃ c₄ j').hom := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₁₂ : Type u_3
    C₂₃ : Type u_4
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝³³ : CategoryTheory.Category.{u_16, u_1} C₁
    inst✝³² : CategoryTheory.Category.{u_17, u_2} C₂
    inst✝³¹ : CategoryTheory.Category.{u_15, u_5} C₃
    inst✝³⁰ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝²⁹ : CategoryTheory.Category.{u_14, u_3} C₁₂
    inst✝²⁸ : CategoryTheory.Category.{u_18, u_4} C₂₃
    inst✝²⁷ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝²⁶ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝²⁵ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝²⁴ : CategoryTheory.Preadditive C₁₂
    inst✝²³ : CategoryTheory.Preadditive C₂₃
    inst✝²² : CategoryTheory.Preadditive C₄
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    inst✝²¹ : F₁₂.PreservesZeroMorphisms
    inst✝²⁰ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
    inst✝¹⁹ : G.Additive
    inst✝¹⁸ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
    inst✝¹⁷ : G₂₃.PreservesZeroMorphisms
    inst✝¹⁶ : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
    inst✝¹⁵ : F.PreservesZeroMorphisms
    inst✝¹⁴ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    associator : CategoryTheory.Iso (CategoryTheory.bifunctorComp₁₂ F₁₂ G) (Catego …
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₂₃ : Type u_11
    ι₄ : Type u_12
    inst✝¹³ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₂₃ : ComplexShape ι₂₃
    c₄ : ComplexShape ι₄
    inst✝¹² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹¹ : TotalComplexShape c₁₂ c₃ c₄
    inst✝¹⁰ : TotalComplexShape c₂ c₃ c₂₃
    inst✝⁹ : TotalComplexShape c₁ c₂₃ c₄
    inst✝⁸ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
    inst✝⁷ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
    inst✝⁶ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
    inst✝⁵ : DecidableEq ι₁₂
    inst✝⁴ : DecidableEq ι₂₃
    inst✝³ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
    inst✝² : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
    inst✝¹ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
    inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
    j j' : ι₄
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.mapBifunctorAssoc …
  -/
  ext i₁ i₂ i₃ h
  /-
    case hfg
    C₁ : Type u_1
    C₂ : Type u_2
    C₁₂ : Type u_3
    C₂₃ : Type u_4
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝³³ : CategoryTheory.Category.{u_16, u_1} C₁
    inst✝³² : CategoryTheory.Category.{u_17, u_2} C₂
    inst✝³¹ : CategoryTheory.Category.{u_15, u_5} C₃
    inst✝³⁰ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝²⁹ : CategoryTheory.Category.{u_14, u_3} C₁₂
    inst✝²⁸ : CategoryTheory.Category.{u_18, u_4} C₂₃
    inst✝²⁷ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝²⁶ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝²⁵ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝²⁴ : CategoryTheory.Preadditive C₁₂
    inst✝²³ : CategoryTheory.Preadditive C₂₃
    inst✝²² : CategoryTheory.Preadditive C₄
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    inst✝²¹ : F₁₂.PreservesZeroMorphisms
    inst✝²⁰ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
    inst✝¹⁹ : G.Additive
    inst✝¹⁸ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
    inst✝¹⁷ : G₂₃.PreservesZeroMorphisms
    inst✝¹⁶ : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
    inst✝¹⁵ : F.PreservesZeroMorphisms
    inst✝¹⁴ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    associator : CategoryTheory.Iso (CategoryTheory.bifunctorComp₁₂ F₁₂ G) (Catego …
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₂₃ : Type u_11
    ι₄ : Type u_12
    inst✝¹³ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₂₃ : ComplexShape ι₂₃
    c₄ : ComplexShape ι₄
    inst✝¹² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹¹ : TotalComplexShape c₁₂ c₃ c₄
    inst✝¹⁰ : TotalComplexShape c₂ c₃ c₂₃
    inst✝⁹ : TotalComplexShape c₁ c₂₃ c₄
    inst✝⁸ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
    inst✝⁷ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
    inst✝⁶ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
    inst✝⁵ : DecidableEq ι₁₂
    inst✝⁴ : DecidableEq ι₂₃
    inst✝³ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
    inst✝² : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
    inst✝¹ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
    inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
    j j' : ι₄
    i₁ : ι₁
    i₂ : ι₂
    i₃ : ι₃
    h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.mapBifunctor₁₂.ι  …
  -/
  rw [mapBifunctor₁₂.ι_D₃_assoc, ι_mapBifunctorAssociatorX_hom_assoc, mapBifunctor₂₃.ι_D₃]
  /-
    case hfg
    C₁ : Type u_1
    C₂ : Type u_2
    C₁₂ : Type u_3
    C₂₃ : Type u_4
    C₃ : Type u_5
    C₄ : Type u_6
    inst✝³³ : CategoryTheory.Category.{u_16, u_1} C₁
    inst✝³² : CategoryTheory.Category.{u_17, u_2} C₂
    inst✝³¹ : CategoryTheory.Category.{u_15, u_5} C₃
    inst✝³⁰ : CategoryTheory.Category.{u_13, u_6} C₄
    inst✝²⁹ : CategoryTheory.Category.{u_14, u_3} C₁₂
    inst✝²⁸ : CategoryTheory.Category.{u_18, u_4} C₂₃
    inst✝²⁷ : CategoryTheory.Limits.HasZeroMorphisms C₁
    inst✝²⁶ : CategoryTheory.Limits.HasZeroMorphisms C₂
    inst✝²⁵ : CategoryTheory.Limits.HasZeroMorphisms C₃
    inst✝²⁴ : CategoryTheory.Preadditive C₁₂
    inst✝²³ : CategoryTheory.Preadditive C₂₃
    inst✝²² : CategoryTheory.Preadditive C₄
    F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
    G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
    F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
    G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
    inst✝²¹ : F₁₂.PreservesZeroMorphisms
    inst✝²⁰ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
    inst✝¹⁹ : G.Additive
    inst✝¹⁸ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
    inst✝¹⁷ : G₂₃.PreservesZeroMorphisms
    inst✝¹⁶ : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
    inst✝¹⁵ : F.PreservesZeroMorphisms
    inst✝¹⁴ : ∀ (X₁ : C₁), (F.obj X₁).Additive
    associator : CategoryTheory.Iso (CategoryTheory.bifunctorComp₁₂ F₁₂ G) (Catego …
    ι₁ : Type u_7
    ι₂ : Type u_8
    ι₃ : Type u_9
    ι₁₂ : Type u_10
    ι₂₃ : Type u_11
    ι₄ : Type u_12
    inst✝¹³ : DecidableEq ι₄
    c₁ : ComplexShape ι₁
    c₂ : ComplexShape ι₂
    c₃ : ComplexShape ι₃
    K₁ : HomologicalComplex C₁ c₁
    K₂ : HomologicalComplex C₂ c₂
    K₃ : HomologicalComplex C₃ c₃
    c₁₂ : ComplexShape ι₁₂
    c₂₃ : ComplexShape ι₂₃
    c₄ : ComplexShape ι₄
    inst✝¹² : TotalComplexShape c₁ c₂ c₁₂
    inst✝¹¹ : TotalComplexShape c₁₂ c₃ c₄
    inst✝¹⁰ : TotalComplexShape c₂ c₃ c₂₃
    inst✝⁹ : TotalComplexShape c₁ c₂₃ c₄
    inst✝⁸ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
    inst✝⁷ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
    inst✝⁶ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
    inst✝⁵ : DecidableEq ι₁₂
    inst✝⁴ : DecidableEq ι₂₃
    inst✝³ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
    inst✝² : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
    inst✝¹ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
    inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
    j j' : ι₄
    i₁ : ι₁
    i₂ : ι₂
    i₃ : ι₃
    h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (((associator.hom.app (K₁.X i₁)).app  …
  -/
  by_cases h₁ : c₃.Rel i₃ (c₃.next i₃)
  · rw [mapBifunctor₁₂.d₃_eq _ _ _ _ _ _ _ _ _ h₁,
      mapBifunctor₂₃.d₃_eq _ _ _ _ _ _ _ _ _ _ h₁,
      Linear.comp_units_smul, Linear.units_smul_comp, assoc,
      ιOrZero_mapBifunctorAssociatorX_hom, NatTrans.naturality_assoc,
      ComplexShape.associative_ε₂_eq_mul c₁ c₂ c₃ c₁₂ c₂₃ c₄]
    /-
      case pos
      C₁ : Type u_1
      C₂ : Type u_2
      C₁₂ : Type u_3
      C₂₃ : Type u_4
      C₃ : Type u_5
      C₄ : Type u_6
      inst✝³³ : CategoryTheory.Category.{u_16, u_1} C₁
      inst✝³² : CategoryTheory.Category.{u_17, u_2} C₂
      inst✝³¹ : CategoryTheory.Category.{u_15, u_5} C₃
      inst✝³⁰ : CategoryTheory.Category.{u_13, u_6} C₄
      inst✝²⁹ : CategoryTheory.Category.{u_14, u_3} C₁₂
      inst✝²⁸ : CategoryTheory.Category.{u_18, u_4} C₂₃
      inst✝²⁷ : CategoryTheory.Limits.HasZeroMorphisms C₁
      inst✝²⁶ : CategoryTheory.Limits.HasZeroMorphisms C₂
      inst✝²⁵ : CategoryTheory.Limits.HasZeroMorphisms C₃
      inst✝²⁴ : CategoryTheory.Preadditive C₁₂
      inst✝²³ : CategoryTheory.Preadditive C₂₃
      inst✝²² : CategoryTheory.Preadditive C₄
      F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
      G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
      F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
      G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
      inst✝²¹ : F₁₂.PreservesZeroMorphisms
      inst✝²⁰ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
      inst✝¹⁹ : G.Additive
      inst✝¹⁸ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
      inst✝¹⁷ : G₂₃.PreservesZeroMorphisms
      inst✝¹⁶ : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
      inst✝¹⁵ : F.PreservesZeroMorphisms
      inst✝¹⁴ : ∀ (X₁ : C₁), (F.obj X₁).Additive
      associator : CategoryTheory.Iso (CategoryTheory.bifunctorComp₁₂ F₁₂ G) (Catego …
      ι₁ : Type u_7
      ι₂ : Type u_8
      ι₃ : Type u_9
      ι₁₂ : Type u_10
      ι₂₃ : Type u_11
      ι₄ : Type u_12
      inst✝¹³ : DecidableEq ι₄
      c₁ : ComplexShape ι₁
      c₂ : ComplexShape ι₂
      c₃ : ComplexShape ι₃
      K₁ : HomologicalComplex C₁ c₁
      K₂ : HomologicalComplex C₂ c₂
      K₃ : HomologicalComplex C₃ c₃
      c₁₂ : ComplexShape ι₁₂
      c₂₃ : ComplexShape ι₂₃
      c₄ : ComplexShape ι₄
      inst✝¹² : TotalComplexShape c₁ c₂ c₁₂
      inst✝¹¹ : TotalComplexShape c₁₂ c₃ c₄
      inst✝¹⁰ : TotalComplexShape c₂ c₃ c₂₃
      inst✝⁹ : TotalComplexShape c₁ c₂₃ c₄
      inst✝⁸ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
      inst✝⁷ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
      inst✝⁶ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
      inst✝⁵ : DecidableEq ι₁₂
      inst✝⁴ : DecidableEq ι₂₃
      inst✝³ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
      inst✝² : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
      inst✝¹ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
      inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
      j j' : ι₄
      i₁ : ι₁
      i₂ : ι₂
      i₃ : ι₃
      h : Eq (c₁.r c₂ c₃ c₁₂ c₄ { fst := i₁, snd := { fst := i₂, snd := i₃ } }) j
      h₁ : c₃.Rel i₃ (c₃.next i₃)
      ⊢ Eq (HSMul.hSMul (HMul.hMul (c₁.ε₂ c₂₃ c₄ { fst := i₁, snd := c₂.π c₃ c₂₃ { f …
    -/
    dsimp
    /-
      🎉 no goals
    -/
  · rw [mapBifunctor₁₂.d₃_eq_zero _ _ _ _ _ _ _ _ _ _ _ h₁,
      mapBifunctor₂₃.d₃_eq_zero _ _ _ _ _ _ _ _ _ _ _ _ h₁, comp_zero, zero_comp]


/-- The associator isomorphism for the action of bifunctors
on homological complexes. -/
noncomputable def mapBifunctorAssociator :
    mapBifunctor (mapBifunctor K₁ K₂ F₁₂ c₁₂) K₃ G c₄ ≅
      mapBifunctor K₁ (mapBifunctor K₂ K₃ G₂₃ c₂₃) F c₄ :=
  Hom.isoOfComponents (mapBifunctorAssociatorX associator K₁ K₂ K₃ c₁₂ c₂₃ c₄) (by
    /-
      C₁ : Type u_1
      C₂ : Type u_2
      C₁₂ : Type u_3
      C₂₃ : Type u_4
      C₃ : Type u_5
      C₄ : Type u_6
      inst✝³³ : CategoryTheory.Category.{?u.1225000, u_1} C₁
      inst✝³² : CategoryTheory.Category.{?u.1225004, u_2} C₂
      inst✝³¹ : CategoryTheory.Category.{?u.1225008, u_5} C₃
      inst✝³⁰ : CategoryTheory.Category.{?u.1225012, u_6} C₄
      inst✝²⁹ : CategoryTheory.Category.{?u.1225016, u_3} C₁₂
      inst✝²⁸ : CategoryTheory.Category.{?u.1225020, u_4} C₂₃
      inst✝²⁷ : CategoryTheory.Limits.HasZeroMorphisms C₁
      inst✝²⁶ : CategoryTheory.Limits.HasZeroMorphisms C₂
      inst✝²⁵ : CategoryTheory.Limits.HasZeroMorphisms C₃
      inst✝²⁴ : CategoryTheory.Preadditive C₁₂
      inst✝²³ : CategoryTheory.Preadditive C₂₃
      inst✝²² : CategoryTheory.Preadditive C₄
      F₁₂ : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂ C₁₂)
      G : CategoryTheory.Functor C₁₂ (CategoryTheory.Functor C₃ C₄)
      F : CategoryTheory.Functor C₁ (CategoryTheory.Functor C₂₃ C₄)
      G₂₃ : CategoryTheory.Functor C₂ (CategoryTheory.Functor C₃ C₂₃)
      inst✝²¹ : F₁₂.PreservesZeroMorphisms
      inst✝²⁰ : ∀ (X₁ : C₁), (F₁₂.obj X₁).PreservesZeroMorphisms
      inst✝¹⁹ : G.Additive
      inst✝¹⁸ : ∀ (X₁₂ : C₁₂), (G.obj X₁₂).PreservesZeroMorphisms
      inst✝¹⁷ : G₂₃.PreservesZeroMorphisms
      inst✝¹⁶ : ∀ (X₂ : C₂), (G₂₃.obj X₂).PreservesZeroMorphisms
      inst✝¹⁵ : F.PreservesZeroMorphisms
      inst✝¹⁴ : ∀ (X₁ : C₁), (F.obj X₁).Additive
      associator : CategoryTheory.Iso (CategoryTheory.bifunctorComp₁₂ F₁₂ G) (Catego …
      ι₁ : Type u_7
      ι₂ : Type u_8
      ι₃ : Type u_9
      ι₁₂ : Type u_10
      ι₂₃ : Type u_11
      ι₄ : Type u_12
      inst✝¹³ : DecidableEq ι₄
      c₁ : ComplexShape ι₁
      c₂ : ComplexShape ι₂
      c₃ : ComplexShape ι₃
      K₁ : HomologicalComplex C₁ c₁
      K₂ : HomologicalComplex C₂ c₂
      K₃ : HomologicalComplex C₃ c₃
      c₁₂ : ComplexShape ι₁₂
      c₂₃ : ComplexShape ι₂₃
      c₄ : ComplexShape ι₄
      inst✝¹² : TotalComplexShape c₁ c₂ c₁₂
      inst✝¹¹ : TotalComplexShape c₁₂ c₃ c₄
      inst✝¹⁰ : TotalComplexShape c₂ c₃ c₂₃
      inst✝⁹ : TotalComplexShape c₁ c₂₃ c₄
      inst✝⁸ : K₁.HasMapBifunctor K₂ F₁₂ c₁₂
      inst✝⁷ : K₂.HasMapBifunctor K₃ G₂₃ c₂₃
      inst✝⁶ : c₁.Associative c₂ c₃ c₁₂ c₂₃ c₄
      inst✝⁵ : DecidableEq ι₁₂
      inst✝⁴ : DecidableEq ι₂₃
      inst✝³ : (K₁.mapBifunctor K₂ F₁₂ c₁₂).HasMapBifunctor K₃ G c₄
      inst✝² : K₁.HasMapBifunctor (K₂.mapBifunctor K₃ G₂₃ c₂₃) F c₄
      inst✝¹ : HomologicalComplex.HasGoodTrifunctor₁₂Obj F₁₂ G K₁ K₂ K₃ c₁₂ c₄
      inst✝ : HomologicalComplex.HasGoodTrifunctor₂₃Obj F G₂₃ K₁ K₂ K₃ c₁₂ c₂₃ c₄
      ⊢ ∀ (i j : ι₄), c₄.Rel i j → Eq (CategoryTheory.CategoryStruct.comp (Homologic …
    -/
    intro j j' _
    simp only [mapBifunctor₁₂.d_eq, mapBifunctor₂₃.d_eq  _ _ _ _ _ c₁₂,
      Preadditive.add_comp, Preadditive.comp_add,
      mapBifunctorAssociatorX_hom_D₁, mapBifunctorAssociatorX_hom_D₂,
      mapBifunctorAssociatorX_hom_D₃])


