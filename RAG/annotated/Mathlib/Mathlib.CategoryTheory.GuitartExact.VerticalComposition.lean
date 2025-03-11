/-- Given `w : TwoSquare T L R B`, one may obtain a 2-square `TwoSquare T L' R' B` if we
provide natural transformations `α : L ⟶ L'` and `β : R' ⟶ R`. -/
@[simps!]
def whiskerVertical (α : L ⟶ L') (β : R' ⟶ R) :
    TwoSquare T L' R' B :=
  whiskerLeft _ β ≫ w ≫ whiskerRight α _


/-- A 2-square stays Guitart exact if we replace the left and right functors
by isomorphic functors. See also `whiskerVertical_iff`. -/
lemma whiskerVertical [w.GuitartExact] (α : L ≅ L') (β : R ≅ R') :
    (w.whiskerVertical α.hom β.inv).GuitartExact := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    D₁ : Type u_4
    D₂ : Type u_5
    inst✝⁴ : CategoryTheory.Category.{u_7, u_1} C₁
    inst✝³ : CategoryTheory.Category.{u_9, u_2} C₂
    inst✝² : CategoryTheory.Category.{u_8, u_4} D₁
    inst✝¹ : CategoryTheory.Category.{u_10, u_5} D₂
    T : CategoryTheory.Functor C₁ D₁
    L : CategoryTheory.Functor C₁ C₂
    R : CategoryTheory.Functor D₁ D₂
    B : CategoryTheory.Functor C₂ D₂
    w : CategoryTheory.TwoSquare T L R B
    L' : CategoryTheory.Functor C₁ C₂
    R' : CategoryTheory.Functor D₁ D₂
    inst✝ : w.GuitartExact
    α : CategoryTheory.Iso L L'
    β : CategoryTheory.Iso R R'
    ⊢ (w.whiskerVertical α.hom β.inv).GuitartExact
  -/
  rw [guitartExact_iff_initial]
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    D₁ : Type u_4
    D₂ : Type u_5
    inst✝⁴ : CategoryTheory.Category.{u_7, u_1} C₁
    inst✝³ : CategoryTheory.Category.{u_9, u_2} C₂
    inst✝² : CategoryTheory.Category.{u_8, u_4} D₁
    inst✝¹ : CategoryTheory.Category.{u_10, u_5} D₂
    T : CategoryTheory.Functor C₁ D₁
    L : CategoryTheory.Functor C₁ C₂
    R : CategoryTheory.Functor D₁ D₂
    B : CategoryTheory.Functor C₂ D₂
    w : CategoryTheory.TwoSquare T L R B
    L' : CategoryTheory.Functor C₁ C₂
    R' : CategoryTheory.Functor D₁ D₂
    inst✝ : w.GuitartExact
    α : CategoryTheory.Iso L L'
    β : CategoryTheory.Iso R R'
    ⊢ ∀ (X₂ : D₁), ((w.whiskerVertical α.hom β.inv).structuredArrowDownwards X₂).I …
  -/
  intro X₂
  let e : structuredArrowDownwards (w.whiskerVertical α.hom β.inv) X₂ ≅
      w.structuredArrowDownwards X₂ ⋙ (StructuredArrow.mapIso (β.app X₂) ).functor :=
    NatIso.ofComponents (fun f => StructuredArrow.isoMk (α.symm.app f.right) (by
      dsimp
      simp only [NatTrans.naturality_assoc, assoc, NatIso.cancel_natIso_inv_left, ← B.map_comp,
        Iso.hom_inv_id_app, B.map_id, comp_id]))
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    D₁ : Type u_4
    D₂ : Type u_5
    inst✝⁴ : CategoryTheory.Category.{u_7, u_1} C₁
    inst✝³ : CategoryTheory.Category.{u_9, u_2} C₂
    inst✝² : CategoryTheory.Category.{u_8, u_4} D₁
    inst✝¹ : CategoryTheory.Category.{u_10, u_5} D₂
    T : CategoryTheory.Functor C₁ D₁
    L : CategoryTheory.Functor C₁ C₂
    R : CategoryTheory.Functor D₁ D₂
    B : CategoryTheory.Functor C₂ D₂
    w : CategoryTheory.TwoSquare T L R B
    L' : CategoryTheory.Functor C₁ C₂
    R' : CategoryTheory.Functor D₁ D₂
    inst✝ : w.GuitartExact
    α : CategoryTheory.Iso L L'
    β : CategoryTheory.Iso R R'
    X₂ : D₁
    e : CategoryTheory.Iso ((w.whiskerVertical α.hom β.inv).structuredArrowDownwar …
    ⊢ ((w.whiskerVertical α.hom β.inv).structuredArrowDownwards X₂).Initial
  -/
  rw [Functor.initial_natIso_iff e]
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    D₁ : Type u_4
    D₂ : Type u_5
    inst✝⁴ : CategoryTheory.Category.{u_7, u_1} C₁
    inst✝³ : CategoryTheory.Category.{u_9, u_2} C₂
    inst✝² : CategoryTheory.Category.{u_8, u_4} D₁
    inst✝¹ : CategoryTheory.Category.{u_10, u_5} D₂
    T : CategoryTheory.Functor C₁ D₁
    L : CategoryTheory.Functor C₁ C₂
    R : CategoryTheory.Functor D₁ D₂
    B : CategoryTheory.Functor C₂ D₂
    w : CategoryTheory.TwoSquare T L R B
    L' : CategoryTheory.Functor C₁ C₂
    R' : CategoryTheory.Functor D₁ D₂
    inst✝ : w.GuitartExact
    α : CategoryTheory.Iso L L'
    β : CategoryTheory.Iso R R'
    X₂ : D₁
    e : CategoryTheory.Iso ((w.whiskerVertical α.hom β.inv).structuredArrowDownwar …
    ⊢ ((w.structuredArrowDownwards X₂).comp (CategoryTheory.StructuredArrow.mapIso …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- A 2-square is Guitart exact iff it is so after replacing the left and right functors by
isomorphic functors. -/
@[simp]
lemma whiskerVertical_iff (α : L ≅ L') (β : R ≅ R') :
    (w.whiskerVertical α.hom β.inv).GuitartExact ↔ w.GuitartExact := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    D₁ : Type u_4
    D₂ : Type u_5
    inst✝³ : CategoryTheory.Category.{u_8, u_1} C₁
    inst✝² : CategoryTheory.Category.{u_7, u_2} C₂
    inst✝¹ : CategoryTheory.Category.{u_10, u_4} D₁
    inst✝ : CategoryTheory.Category.{u_9, u_5} D₂
    T : CategoryTheory.Functor C₁ D₁
    L : CategoryTheory.Functor C₁ C₂
    R : CategoryTheory.Functor D₁ D₂
    B : CategoryTheory.Functor C₂ D₂
    w : CategoryTheory.TwoSquare T L R B
    L' : CategoryTheory.Functor C₁ C₂
    R' : CategoryTheory.Functor D₁ D₂
    α : CategoryTheory.Iso L L'
    β : CategoryTheory.Iso R R'
    ⊢ Iff (w.whiskerVertical α.hom β.inv).GuitartExact w.GuitartExact
  -/
  constructor
    /-
      case mp
      C₁ : Type u_1
      C₂ : Type u_2
      D₁ : Type u_4
      D₂ : Type u_5
      inst✝³ : CategoryTheory.Category.{u_8, u_1} C₁
      inst✝² : CategoryTheory.Category.{u_7, u_2} C₂
      inst✝¹ : CategoryTheory.Category.{u_10, u_4} D₁
      inst✝ : CategoryTheory.Category.{u_9, u_5} D₂
      T : CategoryTheory.Functor C₁ D₁
      L : CategoryTheory.Functor C₁ C₂
      R : CategoryTheory.Functor D₁ D₂
      B : CategoryTheory.Functor C₂ D₂
      w : CategoryTheory.TwoSquare T L R B
      L' : CategoryTheory.Functor C₁ C₂
      R' : CategoryTheory.Functor D₁ D₂
      α : CategoryTheory.Iso L L'
      β : CategoryTheory.Iso R R'
      ⊢ (w.whiskerVertical α.hom β.inv).GuitartExact → w.GuitartExact
    -/
  · intro h
    have : w = TwoSquare.whiskerVertical
        (TwoSquare.whiskerVertical w α.hom β.inv) α.inv β.hom := by
      ext X₁
      simp only [Functor.comp_obj, whiskerVertical_app, assoc, Iso.hom_inv_id_app_assoc,
        ← B.map_comp, Iso.hom_inv_id_app, B.map_id, comp_id]
    /-
      case mp
      C₁ : Type u_1
      C₂ : Type u_2
      D₁ : Type u_4
      D₂ : Type u_5
      inst✝³ : CategoryTheory.Category.{u_8, u_1} C₁
      inst✝² : CategoryTheory.Category.{u_7, u_2} C₂
      inst✝¹ : CategoryTheory.Category.{u_10, u_4} D₁
      inst✝ : CategoryTheory.Category.{u_9, u_5} D₂
      T : CategoryTheory.Functor C₁ D₁
      L : CategoryTheory.Functor C₁ C₂
      R : CategoryTheory.Functor D₁ D₂
      B : CategoryTheory.Functor C₂ D₂
      w : CategoryTheory.TwoSquare T L R B
      L' : CategoryTheory.Functor C₁ C₂
      R' : CategoryTheory.Functor D₁ D₂
      α : CategoryTheory.Iso L L'
      β : CategoryTheory.Iso R R'
      h : (w.whiskerVertical α.hom β.inv).GuitartExact
      this : Eq w ((w.whiskerVertical α.hom β.inv).whiskerVertical α.inv β.hom)
      ⊢ w.GuitartExact
    -/
    rw [this]
    /-
      case mp
      C₁ : Type u_1
      C₂ : Type u_2
      D₁ : Type u_4
      D₂ : Type u_5
      inst✝³ : CategoryTheory.Category.{u_8, u_1} C₁
      inst✝² : CategoryTheory.Category.{u_7, u_2} C₂
      inst✝¹ : CategoryTheory.Category.{u_10, u_4} D₁
      inst✝ : CategoryTheory.Category.{u_9, u_5} D₂
      T : CategoryTheory.Functor C₁ D₁
      L : CategoryTheory.Functor C₁ C₂
      R : CategoryTheory.Functor D₁ D₂
      B : CategoryTheory.Functor C₂ D₂
      w : CategoryTheory.TwoSquare T L R B
      L' : CategoryTheory.Functor C₁ C₂
      R' : CategoryTheory.Functor D₁ D₂
      α : CategoryTheory.Iso L L'
      β : CategoryTheory.Iso R R'
      h : (w.whiskerVertical α.hom β.inv).GuitartExact
      this : Eq w ((w.whiskerVertical α.hom β.inv).whiskerVertical α.inv β.hom)
      ⊢ ((w.whiskerVertical α.hom β.inv).whiskerVertical α.inv β.hom).GuitartExact
    -/
    exact whiskerVertical (w.whiskerVertical α.hom β.inv) α.symm β.symm
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C₁ : Type u_1
      C₂ : Type u_2
      D₁ : Type u_4
      D₂ : Type u_5
      inst✝³ : CategoryTheory.Category.{u_8, u_1} C₁
      inst✝² : CategoryTheory.Category.{u_7, u_2} C₂
      inst✝¹ : CategoryTheory.Category.{u_10, u_4} D₁
      inst✝ : CategoryTheory.Category.{u_9, u_5} D₂
      T : CategoryTheory.Functor C₁ D₁
      L : CategoryTheory.Functor C₁ C₂
      R : CategoryTheory.Functor D₁ D₂
      B : CategoryTheory.Functor C₂ D₂
      w : CategoryTheory.TwoSquare T L R B
      L' : CategoryTheory.Functor C₁ C₂
      R' : CategoryTheory.Functor D₁ D₂
      α : CategoryTheory.Iso L L'
      β : CategoryTheory.Iso R R'
      ⊢ w.GuitartExact → (w.whiskerVertical α.hom β.inv).GuitartExact
    -/
  · intro h
    /-
      case mpr
      C₁ : Type u_1
      C₂ : Type u_2
      D₁ : Type u_4
      D₂ : Type u_5
      inst✝³ : CategoryTheory.Category.{u_8, u_1} C₁
      inst✝² : CategoryTheory.Category.{u_7, u_2} C₂
      inst✝¹ : CategoryTheory.Category.{u_10, u_4} D₁
      inst✝ : CategoryTheory.Category.{u_9, u_5} D₂
      T : CategoryTheory.Functor C₁ D₁
      L : CategoryTheory.Functor C₁ C₂
      R : CategoryTheory.Functor D₁ D₂
      B : CategoryTheory.Functor C₂ D₂
      w : CategoryTheory.TwoSquare T L R B
      L' : CategoryTheory.Functor C₁ C₂
      R' : CategoryTheory.Functor D₁ D₂
      α : CategoryTheory.Iso L L'
      β : CategoryTheory.Iso R R'
      h : w.GuitartExact
      ⊢ (w.whiskerVertical α.hom β.inv).GuitartExact
    -/
    exact whiskerVertical w α β
    /-
      🎉 no goals
    -/


instance [w.GuitartExact] (α : L ⟶ L') (β : R' ⟶ R)
    [IsIso α] [IsIso β] : (w.whiskerVertical α β).GuitartExact :=
  whiskerVertical w (asIso α) (asIso β).symm


/-- The vertical composition of 2-squares. -/
@[simps!]
def vComp : TwoSquare H₁ (L₁ ⋙ L₂) (R₁ ⋙ R₂) H₃ :=
  (Functor.associator _ _ _).inv ≫ whiskerRight w R₂ ≫
    (Functor.associator _ _ _).hom ≫ whiskerLeft L₁ w' ≫ (Functor.associator _ _ _).inv


/-- The canonical isomorphism between
`w.structuredArrowDownwards Y₁ ⋙ w'.structuredArrowDownwards (R₁.obj Y₁)` and
`(w.vComp w').structuredArrowDownwards Y₁.` -/
def structuredArrowDownwardsComp (Y₁ : D₁) :
    w.structuredArrowDownwards Y₁ ⋙ w'.structuredArrowDownwards (R₁.obj Y₁) ≅
      (w.vComp w').structuredArrowDownwards Y₁ :=
                                /-
                                  C₁ : Type u_1
                                  C₂ : Type u_2
                                  C₃ : Type u_3
                                  D₁ : Type u_4
                                  D₂ : Type u_5
                                  D₃ : Type u_6
                                  inst✝⁵ : CategoryTheory.Category.{?u.30919, u_1} C₁
                                  inst✝⁴ : CategoryTheory.Category.{?u.30923, u_2} C₂
                                  inst✝³ : CategoryTheory.Category.{?u.30927, u_3} C₃
                                  inst✝² : CategoryTheory.Category.{?u.30931, u_4} D₁
                                  inst✝¹ : CategoryTheory.Category.{?u.30935, u_5} D₂
                                  inst✝ : CategoryTheory.Category.{?u.30939, u_6} D₃
                                  H₁ : CategoryTheory.Functor C₁ D₁
                                  L₁ : CategoryTheory.Functor C₁ C₂
                                  R₁ : CategoryTheory.Functor D₁ D₂
                                  H₂ : CategoryTheory.Functor C₂ D₂
                                  w : CategoryTheory.TwoSquare H₁ L₁ R₁ H₂
                                  L₂ : CategoryTheory.Functor C₂ C₃
                                  R₂ : CategoryTheory.Functor D₂ D₃
                                  H₃ : CategoryTheory.Functor C₃ D₃
                                  w' : CategoryTheory.TwoSquare H₂ L₂ R₂ H₃
                                  Y₁ : D₁
                                  x✝ : CategoryTheory.StructuredArrow Y₁ H₁
                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (((w.structuredArrowDownwards Y₁).com …
                                -/
                                /-
                                  🎉 no goals
                                -/
  NatIso.ofComponents (fun _ => StructuredArrow.isoMk (Iso.refl _))
  /-
    🎉 no goals
  -/


/-- The vertical composition of 2-squares. (Variant where we allow the replacement of
the vertical compositions by isomorphic functors.) -/
@[simps!]
def vComp' {L₁₂ : C₁ ⥤ C₃} {R₁₂ : D₁ ⥤ D₃} (eL : L₁ ⋙ L₂ ≅ L₁₂)
    (eR : R₁ ⋙ R₂ ≅ R₁₂) : TwoSquare H₁ L₁₂ R₁₂ H₃ :=
  (w.vComp w').whiskerVertical eL.hom eR.inv


instance vComp [hw : w.GuitartExact] [hw' : w'.GuitartExact] :
    (w.vComp w').GuitartExact := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    D₁ : Type u_4
    D₂ : Type u_5
    D₃ : Type u_6
    inst✝⁵ : CategoryTheory.Category.{u_7, u_1} C₁
    inst✝⁴ : CategoryTheory.Category.{u_9, u_2} C₂
    inst✝³ : CategoryTheory.Category.{u_11, u_3} C₃
    inst✝² : CategoryTheory.Category.{u_8, u_4} D₁
    inst✝¹ : CategoryTheory.Category.{u_10, u_5} D₂
    inst✝ : CategoryTheory.Category.{u_12, u_6} D₃
    H₁ : CategoryTheory.Functor C₁ D₁
    L₁ : CategoryTheory.Functor C₁ C₂
    R₁ : CategoryTheory.Functor D₁ D₂
    H₂ : CategoryTheory.Functor C₂ D₂
    w : CategoryTheory.TwoSquare H₁ L₁ R₁ H₂
    L₂ : CategoryTheory.Functor C₂ C₃
    R₂ : CategoryTheory.Functor D₂ D₃
    H₃ : CategoryTheory.Functor C₃ D₃
    w' : CategoryTheory.TwoSquare H₂ L₂ R₂ H₃
    hw : w.GuitartExact
    hw' : w'.GuitartExact
    ⊢ (w.vComp w').GuitartExact
  -/
  simp only [TwoSquare.guitartExact_iff_initial]
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    D₁ : Type u_4
    D₂ : Type u_5
    D₃ : Type u_6
    inst✝⁵ : CategoryTheory.Category.{u_7, u_1} C₁
    inst✝⁴ : CategoryTheory.Category.{u_9, u_2} C₂
    inst✝³ : CategoryTheory.Category.{u_11, u_3} C₃
    inst✝² : CategoryTheory.Category.{u_8, u_4} D₁
    inst✝¹ : CategoryTheory.Category.{u_10, u_5} D₂
    inst✝ : CategoryTheory.Category.{u_12, u_6} D₃
    H₁ : CategoryTheory.Functor C₁ D₁
    L₁ : CategoryTheory.Functor C₁ C₂
    R₁ : CategoryTheory.Functor D₁ D₂
    H₂ : CategoryTheory.Functor C₂ D₂
    w : CategoryTheory.TwoSquare H₁ L₁ R₁ H₂
    L₂ : CategoryTheory.Functor C₂ C₃
    R₂ : CategoryTheory.Functor D₂ D₃
    H₃ : CategoryTheory.Functor C₃ D₃
    w' : CategoryTheory.TwoSquare H₂ L₂ R₂ H₃
    hw : w.GuitartExact
    hw' : w'.GuitartExact
    ⊢ ∀ (X₂ : D₁), ((w.vComp w').structuredArrowDownwards X₂).Initial
  -/
  intro Y₁
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    D₁ : Type u_4
    D₂ : Type u_5
    D₃ : Type u_6
    inst✝⁵ : CategoryTheory.Category.{u_7, u_1} C₁
    inst✝⁴ : CategoryTheory.Category.{u_9, u_2} C₂
    inst✝³ : CategoryTheory.Category.{u_11, u_3} C₃
    inst✝² : CategoryTheory.Category.{u_8, u_4} D₁
    inst✝¹ : CategoryTheory.Category.{u_10, u_5} D₂
    inst✝ : CategoryTheory.Category.{u_12, u_6} D₃
    H₁ : CategoryTheory.Functor C₁ D₁
    L₁ : CategoryTheory.Functor C₁ C₂
    R₁ : CategoryTheory.Functor D₁ D₂
    H₂ : CategoryTheory.Functor C₂ D₂
    w : CategoryTheory.TwoSquare H₁ L₁ R₁ H₂
    L₂ : CategoryTheory.Functor C₂ C₃
    R₂ : CategoryTheory.Functor D₂ D₃
    H₃ : CategoryTheory.Functor C₃ D₃
    w' : CategoryTheory.TwoSquare H₂ L₂ R₂ H₃
    hw : w.GuitartExact
    hw' : w'.GuitartExact
    Y₁ : D₁
    ⊢ ((w.vComp w').structuredArrowDownwards Y₁).Initial
  -/
  rw [← Functor.initial_natIso_iff (structuredArrowDownwardsComp w w' Y₁)]
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    D₁ : Type u_4
    D₂ : Type u_5
    D₃ : Type u_6
    inst✝⁵ : CategoryTheory.Category.{u_7, u_1} C₁
    inst✝⁴ : CategoryTheory.Category.{u_9, u_2} C₂
    inst✝³ : CategoryTheory.Category.{u_11, u_3} C₃
    inst✝² : CategoryTheory.Category.{u_8, u_4} D₁
    inst✝¹ : CategoryTheory.Category.{u_10, u_5} D₂
    inst✝ : CategoryTheory.Category.{u_12, u_6} D₃
    H₁ : CategoryTheory.Functor C₁ D₁
    L₁ : CategoryTheory.Functor C₁ C₂
    R₁ : CategoryTheory.Functor D₁ D₂
    H₂ : CategoryTheory.Functor C₂ D₂
    w : CategoryTheory.TwoSquare H₁ L₁ R₁ H₂
    L₂ : CategoryTheory.Functor C₂ C₃
    R₂ : CategoryTheory.Functor D₂ D₃
    H₃ : CategoryTheory.Functor C₃ D₃
    w' : CategoryTheory.TwoSquare H₂ L₂ R₂ H₃
    hw : w.GuitartExact
    hw' : w'.GuitartExact
    Y₁ : D₁
    ⊢ ((w.structuredArrowDownwards Y₁).comp (w'.structuredArrowDownwards (R₁.obj Y …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance vComp' [GuitartExact w] [GuitartExact w'] {L₁₂ : C₁ ⥤ C₃}
    {R₁₂ : D₁ ⥤ D₃} (eL : L₁ ⋙ L₂ ≅ L₁₂)
    (eR : R₁ ⋙ R₂ ≅ R₁₂) : (w.vComp' w' eL eR).GuitartExact := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    D₁ : Type u_4
    D₂ : Type u_5
    D₃ : Type u_6
    inst✝⁷ : CategoryTheory.Category.{u_7, u_1} C₁
    inst✝⁶ : CategoryTheory.Category.{u_9, u_2} C₂
    inst✝⁵ : CategoryTheory.Category.{u_11, u_3} C₃
    inst✝⁴ : CategoryTheory.Category.{u_8, u_4} D₁
    inst✝³ : CategoryTheory.Category.{u_10, u_5} D₂
    inst✝² : CategoryTheory.Category.{u_12, u_6} D₃
    H₁ : CategoryTheory.Functor C₁ D₁
    L₁ : CategoryTheory.Functor C₁ C₂
    R₁ : CategoryTheory.Functor D₁ D₂
    H₂ : CategoryTheory.Functor C₂ D₂
    w : CategoryTheory.TwoSquare H₁ L₁ R₁ H₂
    L₂ : CategoryTheory.Functor C₂ C₃
    R₂ : CategoryTheory.Functor D₂ D₃
    H₃ : CategoryTheory.Functor C₃ D₃
    w' : CategoryTheory.TwoSquare H₂ L₂ R₂ H₃
    inst✝¹ : w.GuitartExact
    inst✝ : w'.GuitartExact
    L₁₂ : CategoryTheory.Functor C₁ C₃
    R₁₂ : CategoryTheory.Functor D₁ D₃
    eL : CategoryTheory.Iso (L₁.comp L₂) L₁₂
    eR : CategoryTheory.Iso (R₁.comp R₂) R₁₂
    ⊢ (w.vComp' w' eL eR).GuitartExact
  -/
  dsimp only [TwoSquare.vComp']
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    D₁ : Type u_4
    D₂ : Type u_5
    D₃ : Type u_6
    inst✝⁷ : CategoryTheory.Category.{u_7, u_1} C₁
    inst✝⁶ : CategoryTheory.Category.{u_9, u_2} C₂
    inst✝⁵ : CategoryTheory.Category.{u_11, u_3} C₃
    inst✝⁴ : CategoryTheory.Category.{u_8, u_4} D₁
    inst✝³ : CategoryTheory.Category.{u_10, u_5} D₂
    inst✝² : CategoryTheory.Category.{u_12, u_6} D₃
    H₁ : CategoryTheory.Functor C₁ D₁
    L₁ : CategoryTheory.Functor C₁ C₂
    R₁ : CategoryTheory.Functor D₁ D₂
    H₂ : CategoryTheory.Functor C₂ D₂
    w : CategoryTheory.TwoSquare H₁ L₁ R₁ H₂
    L₂ : CategoryTheory.Functor C₂ C₃
    R₂ : CategoryTheory.Functor D₂ D₃
    H₃ : CategoryTheory.Functor C₃ D₃
    w' : CategoryTheory.TwoSquare H₂ L₂ R₂ H₃
    inst✝¹ : w.GuitartExact
    inst✝ : w'.GuitartExact
    L₁₂ : CategoryTheory.Functor C₁ C₃
    R₁₂ : CategoryTheory.Functor D₁ D₃
    eL : CategoryTheory.Iso (L₁.comp L₂) L₁₂
    eR : CategoryTheory.Iso (R₁.comp R₂) R₁₂
    ⊢ ((w.vComp w').whiskerVertical eL.hom eR.inv).GuitartExact
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma vComp_iff_of_equivalences (eL : C₂ ≌ C₃) (eR : D₂ ≌ D₃)
    (w' : H₂ ⋙ eR.functor ≅ eL.functor ⋙ H₃) :
    (w.vComp w'.hom).GuitartExact ↔ w.GuitartExact := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    D₁ : Type u_4
    D₂ : Type u_5
    D₃ : Type u_6
    inst✝⁵ : CategoryTheory.Category.{u_11, u_1} C₁
    inst✝⁴ : CategoryTheory.Category.{u_7, u_2} C₂
    inst✝³ : CategoryTheory.Category.{u_8, u_3} C₃
    inst✝² : CategoryTheory.Category.{u_12, u_4} D₁
    inst✝¹ : CategoryTheory.Category.{u_9, u_5} D₂
    inst✝ : CategoryTheory.Category.{u_10, u_6} D₃
    H₁ : CategoryTheory.Functor C₁ D₁
    L₁ : CategoryTheory.Functor C₁ C₂
    R₁ : CategoryTheory.Functor D₁ D₂
    H₂ : CategoryTheory.Functor C₂ D₂
    w : CategoryTheory.TwoSquare H₁ L₁ R₁ H₂
    H₃ : CategoryTheory.Functor C₃ D₃
    eL : CategoryTheory.Equivalence C₂ C₃
    eR : CategoryTheory.Equivalence D₂ D₃
    w' : CategoryTheory.Iso (H₂.comp eR.functor) (eL.functor.comp H₃)
    ⊢ Iff (w.vComp w'.hom).GuitartExact w.GuitartExact
  -/
  constructor
    /-
      case mp
      C₁ : Type u_1
      C₂ : Type u_2
      C₃ : Type u_3
      D₁ : Type u_4
      D₂ : Type u_5
      D₃ : Type u_6
      inst✝⁵ : CategoryTheory.Category.{u_11, u_1} C₁
      inst✝⁴ : CategoryTheory.Category.{u_7, u_2} C₂
      inst✝³ : CategoryTheory.Category.{u_8, u_3} C₃
      inst✝² : CategoryTheory.Category.{u_12, u_4} D₁
      inst✝¹ : CategoryTheory.Category.{u_9, u_5} D₂
      inst✝ : CategoryTheory.Category.{u_10, u_6} D₃
      H₁ : CategoryTheory.Functor C₁ D₁
      L₁ : CategoryTheory.Functor C₁ C₂
      R₁ : CategoryTheory.Functor D₁ D₂
      H₂ : CategoryTheory.Functor C₂ D₂
      w : CategoryTheory.TwoSquare H₁ L₁ R₁ H₂
      H₃ : CategoryTheory.Functor C₃ D₃
      eL : CategoryTheory.Equivalence C₂ C₃
      eR : CategoryTheory.Equivalence D₂ D₃
      w' : CategoryTheory.Iso (H₂.comp eR.functor) (eL.functor.comp H₃)
      ⊢ (w.vComp w'.hom).GuitartExact → w.GuitartExact
    -/
  · intro hww'
    /-
      case mp
      C₁ : Type u_1
      C₂ : Type u_2
      C₃ : Type u_3
      D₁ : Type u_4
      D₂ : Type u_5
      D₃ : Type u_6
      inst✝⁵ : CategoryTheory.Category.{u_11, u_1} C₁
      inst✝⁴ : CategoryTheory.Category.{u_7, u_2} C₂
      inst✝³ : CategoryTheory.Category.{u_8, u_3} C₃
      inst✝² : CategoryTheory.Category.{u_12, u_4} D₁
      inst✝¹ : CategoryTheory.Category.{u_9, u_5} D₂
      inst✝ : CategoryTheory.Category.{u_10, u_6} D₃
      H₁ : CategoryTheory.Functor C₁ D₁
      L₁ : CategoryTheory.Functor C₁ C₂
      R₁ : CategoryTheory.Functor D₁ D₂
      H₂ : CategoryTheory.Functor C₂ D₂
      w : CategoryTheory.TwoSquare H₁ L₁ R₁ H₂
      H₃ : CategoryTheory.Functor C₃ D₃
      eL : CategoryTheory.Equivalence C₂ C₃
      eR : CategoryTheory.Equivalence D₂ D₃
      w' : CategoryTheory.Iso (H₂.comp eR.functor) (eL.functor.comp H₃)
      hww' : (w.vComp w'.hom).GuitartExact
      ⊢ w.GuitartExact
    -/
    letI : CatCommSq H₂ eL.functor eR.functor H₃ := ⟨w'⟩
    /-
      case mp
      C₁ : Type u_1
      C₂ : Type u_2
      C₃ : Type u_3
      D₁ : Type u_4
      D₂ : Type u_5
      D₃ : Type u_6
      inst✝⁵ : CategoryTheory.Category.{u_11, u_1} C₁
      inst✝⁴ : CategoryTheory.Category.{u_7, u_2} C₂
      inst✝³ : CategoryTheory.Category.{u_8, u_3} C₃
      inst✝² : CategoryTheory.Category.{u_12, u_4} D₁
      inst✝¹ : CategoryTheory.Category.{u_9, u_5} D₂
      inst✝ : CategoryTheory.Category.{u_10, u_6} D₃
      H₁ : CategoryTheory.Functor C₁ D₁
      L₁ : CategoryTheory.Functor C₁ C₂
      R₁ : CategoryTheory.Functor D₁ D₂
      H₂ : CategoryTheory.Functor C₂ D₂
      w : CategoryTheory.TwoSquare H₁ L₁ R₁ H₂
      H₃ : CategoryTheory.Functor C₃ D₃
      eL : CategoryTheory.Equivalence C₂ C₃
      eR : CategoryTheory.Equivalence D₂ D₃
      w' : CategoryTheory.Iso (H₂.comp eR.functor) (eL.functor.comp H₃)
      hww' : (w.vComp w'.hom).GuitartExact
      this : CategoryTheory.CatCommSq H₂ eL.functor eR.functor H₃ := { iso' := w' }
      ⊢ w.GuitartExact
    -/
    have hw' : CatCommSq.iso H₂ eL.functor eR.functor H₃ = w' := rfl
    /-
      case mp
      C₁ : Type u_1
      C₂ : Type u_2
      C₃ : Type u_3
      D₁ : Type u_4
      D₂ : Type u_5
      D₃ : Type u_6
      inst✝⁵ : CategoryTheory.Category.{u_11, u_1} C₁
      inst✝⁴ : CategoryTheory.Category.{u_7, u_2} C₂
      inst✝³ : CategoryTheory.Category.{u_8, u_3} C₃
      inst✝² : CategoryTheory.Category.{u_12, u_4} D₁
      inst✝¹ : CategoryTheory.Category.{u_9, u_5} D₂
      inst✝ : CategoryTheory.Category.{u_10, u_6} D₃
      H₁ : CategoryTheory.Functor C₁ D₁
      L₁ : CategoryTheory.Functor C₁ C₂
      R₁ : CategoryTheory.Functor D₁ D₂
      H₂ : CategoryTheory.Functor C₂ D₂
      w : CategoryTheory.TwoSquare H₁ L₁ R₁ H₂
      H₃ : CategoryTheory.Functor C₃ D₃
      eL : CategoryTheory.Equivalence C₂ C₃
      eR : CategoryTheory.Equivalence D₂ D₃
      w' : CategoryTheory.Iso (H₂.comp eR.functor) (eL.functor.comp H₃)
      hww' : (w.vComp w'.hom).GuitartExact
      this : CategoryTheory.CatCommSq H₂ eL.functor eR.functor H₃ := { iso' := w' }
      hw' : Eq (CategoryTheory.CatCommSq.iso H₂ eL.functor eR.functor H₃) w'
      ⊢ w.GuitartExact
    -/
    letI : CatCommSq H₃ eL.inverse eR.inverse H₂ := CatCommSq.vInvEquiv _ _ _ _ inferInstance
    /-
      case mp
      C₁ : Type u_1
      C₂ : Type u_2
      C₃ : Type u_3
      D₁ : Type u_4
      D₂ : Type u_5
      D₃ : Type u_6
      inst✝⁵ : CategoryTheory.Category.{u_11, u_1} C₁
      inst✝⁴ : CategoryTheory.Category.{u_7, u_2} C₂
      inst✝³ : CategoryTheory.Category.{u_8, u_3} C₃
      inst✝² : CategoryTheory.Category.{u_12, u_4} D₁
      inst✝¹ : CategoryTheory.Category.{u_9, u_5} D₂
      inst✝ : CategoryTheory.Category.{u_10, u_6} D₃
      H₁ : CategoryTheory.Functor C₁ D₁
      L₁ : CategoryTheory.Functor C₁ C₂
      R₁ : CategoryTheory.Functor D₁ D₂
      H₂ : CategoryTheory.Functor C₂ D₂
      w : CategoryTheory.TwoSquare H₁ L₁ R₁ H₂
      H₃ : CategoryTheory.Functor C₃ D₃
      eL : CategoryTheory.Equivalence C₂ C₃
      eR : CategoryTheory.Equivalence D₂ D₃
      w' : CategoryTheory.Iso (H₂.comp eR.functor) (eL.functor.comp H₃)
      hww' : (w.vComp w'.hom).GuitartExact
      this✝ : CategoryTheory.CatCommSq H₂ eL.functor eR.functor H₃ := { iso' := w' }
      hw' : Eq (CategoryTheory.CatCommSq.iso H₂ eL.functor eR.functor H₃) w'
      this : CategoryTheory.CatCommSq H₃ eL.inverse eR.inverse H₂ := (CategoryTheory …
      ⊢ w.GuitartExact
    -/
    let w'' := CatCommSq.iso H₃ eL.inverse eR.inverse H₂
    let α : (L₁ ⋙ eL.functor) ⋙ eL.inverse ≅ L₁ :=
      Functor.associator _ _ _ ≪≫ isoWhiskerLeft L₁ eL.unitIso.symm ≪≫ L₁.rightUnitor
    let β : (R₁ ⋙ eR.functor) ⋙ eR.inverse ≅ R₁ :=
      Functor.associator _ _ _ ≪≫ isoWhiskerLeft R₁ eR.unitIso.symm ≪≫ R₁.rightUnitor
    have : w = (w.vComp w'.hom).vComp' w''.hom α β := by
      ext X₁
      dsimp
      simp? [w'', β, α] says
        simp only [vComp'_app, Functor.comp_obj, Iso.trans_inv, isoWhiskerLeft_inv, Iso.symm_inv,
          assoc, NatTrans.comp_app, Functor.id_obj, Functor.rightUnitor_inv_app, whiskerLeft_app,
          Functor.associator_inv_app, comp_id, id_comp, vComp_app, Functor.map_comp,
          Equivalence.inv_fun_map, Iso.trans_hom, isoWhiskerLeft_hom, Iso.symm_hom,
          Functor.associator_hom_app, Functor.rightUnitor_hom_app, Iso.hom_inv_id_app_assoc,
          w'', α, β]
      erw [CatCommSq.vInv_iso'_hom_app]
      simp only [hw', assoc, ← eR.inverse.map_comp_assoc]
      rw [Equivalence.counitInv_app_functor]
      erw [← NatTrans.naturality_assoc]
      simp [← H₂.map_comp]
    /-
      case mp
      C₁ : Type u_1
      C₂ : Type u_2
      C₃ : Type u_3
      D₁ : Type u_4
      D₂ : Type u_5
      D₃ : Type u_6
      inst✝⁵ : CategoryTheory.Category.{u_11, u_1} C₁
      inst✝⁴ : CategoryTheory.Category.{u_7, u_2} C₂
      inst✝³ : CategoryTheory.Category.{u_8, u_3} C₃
      inst✝² : CategoryTheory.Category.{u_12, u_4} D₁
      inst✝¹ : CategoryTheory.Category.{u_9, u_5} D₂
      inst✝ : CategoryTheory.Category.{u_10, u_6} D₃
      H₁ : CategoryTheory.Functor C₁ D₁
      L₁ : CategoryTheory.Functor C₁ C₂
      R₁ : CategoryTheory.Functor D₁ D₂
      H₂ : CategoryTheory.Functor C₂ D₂
      w : CategoryTheory.TwoSquare H₁ L₁ R₁ H₂
      H₃ : CategoryTheory.Functor C₃ D₃
      eL : CategoryTheory.Equivalence C₂ C₃
      eR : CategoryTheory.Equivalence D₂ D₃
      w' : CategoryTheory.Iso (H₂.comp eR.functor) (eL.functor.comp H₃)
      hww' : (w.vComp w'.hom).GuitartExact
      this✝¹ : CategoryTheory.CatCommSq H₂ eL.functor eR.functor H₃ := { iso' := w' }
      hw' : Eq (CategoryTheory.CatCommSq.iso H₂ eL.functor eR.functor H₃) w'
      this✝ : CategoryTheory.CatCommSq H₃ eL.inverse eR.inverse H₂ := (CategoryTheor …
      w'' : CategoryTheory.Iso (H₃.comp eR.inverse) (eL.inverse.comp H₂) := Category …
      α : CategoryTheory.Iso ((L₁.comp eL.functor).comp eL.inverse) L₁ := (L₁.associ …
      β : CategoryTheory.Iso ((R₁.comp eR.functor).comp eR.inverse) R₁ := (R₁.associ …
      this : Eq w ((w.vComp w'.hom).vComp' w''.hom α β)
      ⊢ w.GuitartExact
    -/
    rw [this]
    /-
      case mp
      C₁ : Type u_1
      C₂ : Type u_2
      C₃ : Type u_3
      D₁ : Type u_4
      D₂ : Type u_5
      D₃ : Type u_6
      inst✝⁵ : CategoryTheory.Category.{u_11, u_1} C₁
      inst✝⁴ : CategoryTheory.Category.{u_7, u_2} C₂
      inst✝³ : CategoryTheory.Category.{u_8, u_3} C₃
      inst✝² : CategoryTheory.Category.{u_12, u_4} D₁
      inst✝¹ : CategoryTheory.Category.{u_9, u_5} D₂
      inst✝ : CategoryTheory.Category.{u_10, u_6} D₃
      H₁ : CategoryTheory.Functor C₁ D₁
      L₁ : CategoryTheory.Functor C₁ C₂
      R₁ : CategoryTheory.Functor D₁ D₂
      H₂ : CategoryTheory.Functor C₂ D₂
      w : CategoryTheory.TwoSquare H₁ L₁ R₁ H₂
      H₃ : CategoryTheory.Functor C₃ D₃
      eL : CategoryTheory.Equivalence C₂ C₃
      eR : CategoryTheory.Equivalence D₂ D₃
      w' : CategoryTheory.Iso (H₂.comp eR.functor) (eL.functor.comp H₃)
      hww' : (w.vComp w'.hom).GuitartExact
      this✝¹ : CategoryTheory.CatCommSq H₂ eL.functor eR.functor H₃ := { iso' := w' }
      hw' : Eq (CategoryTheory.CatCommSq.iso H₂ eL.functor eR.functor H₃) w'
      this✝ : CategoryTheory.CatCommSq H₃ eL.inverse eR.inverse H₂ := (CategoryTheor …
      w'' : CategoryTheory.Iso (H₃.comp eR.inverse) (eL.inverse.comp H₂) := Category …
      α : CategoryTheory.Iso ((L₁.comp eL.functor).comp eL.inverse) L₁ := (L₁.associ …
      β : CategoryTheory.Iso ((R₁.comp eR.functor).comp eR.inverse) R₁ := (R₁.associ …
      this : Eq w ((w.vComp w'.hom).vComp' w''.hom α β)
      ⊢ ((w.vComp w'.hom).vComp' w''.hom α β).GuitartExact
    -/
    infer_instance
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C₁ : Type u_1
      C₂ : Type u_2
      C₃ : Type u_3
      D₁ : Type u_4
      D₂ : Type u_5
      D₃ : Type u_6
      inst✝⁵ : CategoryTheory.Category.{u_11, u_1} C₁
      inst✝⁴ : CategoryTheory.Category.{u_7, u_2} C₂
      inst✝³ : CategoryTheory.Category.{u_8, u_3} C₃
      inst✝² : CategoryTheory.Category.{u_12, u_4} D₁
      inst✝¹ : CategoryTheory.Category.{u_9, u_5} D₂
      inst✝ : CategoryTheory.Category.{u_10, u_6} D₃
      H₁ : CategoryTheory.Functor C₁ D₁
      L₁ : CategoryTheory.Functor C₁ C₂
      R₁ : CategoryTheory.Functor D₁ D₂
      H₂ : CategoryTheory.Functor C₂ D₂
      w : CategoryTheory.TwoSquare H₁ L₁ R₁ H₂
      H₃ : CategoryTheory.Functor C₃ D₃
      eL : CategoryTheory.Equivalence C₂ C₃
      eR : CategoryTheory.Equivalence D₂ D₃
      w' : CategoryTheory.Iso (H₂.comp eR.functor) (eL.functor.comp H₃)
      ⊢ w.GuitartExact → (w.vComp w'.hom).GuitartExact
    -/
  · intro
    /-
      case mpr
      C₁ : Type u_1
      C₂ : Type u_2
      C₃ : Type u_3
      D₁ : Type u_4
      D₂ : Type u_5
      D₃ : Type u_6
      inst✝⁵ : CategoryTheory.Category.{u_11, u_1} C₁
      inst✝⁴ : CategoryTheory.Category.{u_7, u_2} C₂
      inst✝³ : CategoryTheory.Category.{u_8, u_3} C₃
      inst✝² : CategoryTheory.Category.{u_12, u_4} D₁
      inst✝¹ : CategoryTheory.Category.{u_9, u_5} D₂
      inst✝ : CategoryTheory.Category.{u_10, u_6} D₃
      H₁ : CategoryTheory.Functor C₁ D₁
      L₁ : CategoryTheory.Functor C₁ C₂
      R₁ : CategoryTheory.Functor D₁ D₂
      H₂ : CategoryTheory.Functor C₂ D₂
      w : CategoryTheory.TwoSquare H₁ L₁ R₁ H₂
      H₃ : CategoryTheory.Functor C₃ D₃
      eL : CategoryTheory.Equivalence C₂ C₃
      eR : CategoryTheory.Equivalence D₂ D₃
      w' : CategoryTheory.Iso (H₂.comp eR.functor) (eL.functor.comp H₃)
      a✝ : w.GuitartExact
      ⊢ (w.vComp w'.hom).GuitartExact
    -/
    exact vComp w w'.hom
    /-
      🎉 no goals
    -/


lemma vComp'_iff_of_equivalences (E : C₂ ≌ C₃) (E' : D₂ ≌ D₃)
    (w' : H₂ ⋙ E'.functor ≅ E.functor ⋙ H₃) {L₁₂ : C₁ ⥤ C₃}
    {R₁₂ : D₁ ⥤ D₃} (eL : L₁ ⋙ E.functor ≅ L₁₂)
    (eR : R₁ ⋙ E'.functor ≅ R₁₂) :
    (w.vComp' w'.hom eL eR).GuitartExact ↔ w.GuitartExact := by
  /-
    C₁ : Type u_1
    C₂ : Type u_2
    C₃ : Type u_3
    D₁ : Type u_4
    D₂ : Type u_5
    D₃ : Type u_6
    inst✝⁵ : CategoryTheory.Category.{u_11, u_1} C₁
    inst✝⁴ : CategoryTheory.Category.{u_7, u_2} C₂
    inst✝³ : CategoryTheory.Category.{u_8, u_3} C₃
    inst✝² : CategoryTheory.Category.{u_12, u_4} D₁
    inst✝¹ : CategoryTheory.Category.{u_9, u_5} D₂
    inst✝ : CategoryTheory.Category.{u_10, u_6} D₃
    H₁ : CategoryTheory.Functor C₁ D₁
    L₁ : CategoryTheory.Functor C₁ C₂
    R₁ : CategoryTheory.Functor D₁ D₂
    H₂ : CategoryTheory.Functor C₂ D₂
    w : CategoryTheory.TwoSquare H₁ L₁ R₁ H₂
    H₃ : CategoryTheory.Functor C₃ D₃
    E : CategoryTheory.Equivalence C₂ C₃
    E' : CategoryTheory.Equivalence D₂ D₃
    w' : CategoryTheory.Iso (H₂.comp E'.functor) (E.functor.comp H₃)
    L₁₂ : CategoryTheory.Functor C₁ C₃
    R₁₂ : CategoryTheory.Functor D₁ D₃
    eL : CategoryTheory.Iso (L₁.comp E.functor) L₁₂
    eR : CategoryTheory.Iso (R₁.comp E'.functor) R₁₂
    ⊢ Iff (w.vComp' w'.hom eL eR).GuitartExact w.GuitartExact
  -/
  rw [← vComp_iff_of_equivalences w E E' w', TwoSquare.vComp', whiskerVertical_iff]
  /-
    🎉 no goals
  -/


