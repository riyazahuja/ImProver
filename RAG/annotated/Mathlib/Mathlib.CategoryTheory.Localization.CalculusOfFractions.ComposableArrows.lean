lemma essSurj_mapComposableArrows_of_hasRightCalculusOfFractions
    [W.HasRightCalculusOfFractions] (n : ℕ) :
    (L.mapComposableArrows n).EssSurj where
  mem_essImage Y := by
    /-
      C : Type u_1
      D : Type u_2
      inst✝³ : CategoryTheory.Category.{u_3, u_1} C
      inst✝² : CategoryTheory.Category.{u_4, u_2} D
      L : CategoryTheory.Functor C D
      W : CategoryTheory.MorphismProperty C
      inst✝¹ : L.IsLocalization W
      inst✝ : W.HasRightCalculusOfFractions
      n : Nat
      Y : CategoryTheory.ComposableArrows D n
      ⊢ Membership.mem (L.mapComposableArrows n).essImage Y
    -/
    have := essSurj L W
    induction n with
    | zero =>
      obtain ⟨Y, rfl⟩ := mk₀_surjective Y
      exact ⟨mk₀ _, ⟨isoMk₀ (L.objObjPreimageIso Y)⟩⟩
    | succ n hn =>
      obtain ⟨Y, Z, f, rfl⟩ := ComposableArrows.precomp_surjective Y
      obtain ⟨Y', ⟨e⟩⟩ := hn Y
      obtain ⟨f', hf'⟩ := exists_rightFraction L W
        ((L.objObjPreimageIso Z).hom ≫ f ≫ (e.app 0).inv)
      refine ⟨Y'.precomp f'.f,
        ⟨isoMkSucc (isoOfHom L W _ f'.hs ≪≫ L.objObjPreimageIso Z) e ?_⟩⟩
      dsimp at hf' ⊢
      simp [← cancel_mono (e.inv.app 0), hf']


lemma essSurj_mapComposableArrows [W.HasLeftCalculusOfFractions] (n : ℕ) :
    (L.mapComposableArrows n).EssSurj := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Category.{u_4, u_2} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    n : Nat
    ⊢ (L.mapComposableArrows n).EssSurj
  -/
  have := essSurj_mapComposableArrows_of_hasRightCalculusOfFractions L.op W.op n
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Category.{u_4, u_2} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    n : Nat
    this : (L.op.mapComposableArrows n).EssSurj
    ⊢ (L.mapComposableArrows n).EssSurj
  -/
  have := Functor.essSurj_of_iso (L.mapComposableArrowsOpIso n).symm
  /-
    C : Type u_1
    D : Type u_2
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Category.{u_4, u_2} D
    L : CategoryTheory.Functor C D
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    n : Nat
    this✝ : (L.op.mapComposableArrows n).EssSurj
    this : ((L.mapComposableArrows n).comp (CategoryTheory.ComposableArrows.opEqui …
    ⊢ (L.mapComposableArrows n).EssSurj
  -/
  exact Functor.essSurj_of_comp_fully_faithful _ (opEquivalence D n).functor.rightOp
  /-
    🎉 no goals
  -/


