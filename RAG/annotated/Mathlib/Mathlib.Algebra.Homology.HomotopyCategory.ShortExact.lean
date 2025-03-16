@[reassoc]
lemma homologySequenceδ_quotient_mapTriangle_obj
    (T : Triangle (CochainComplex C ℤ)) (n₀ n₁ : ℤ) (h : n₀ + 1 = n₁) :
    (homologyFunctor C (up ℤ) 0).homologySequenceδ
        ((quotient C (up ℤ)).mapTriangle.obj T) n₀ n₁ h =
      (homologyFunctorFactors C (up ℤ) n₀).hom.app _ ≫
                                                                                  /-
                                                                                    C : Type u_1
                                                                                    inst✝¹ : CategoryTheory.Category.{?u.29, u_1} C
                                                                                    inst✝ : CategoryTheory.Abelian C
                                                                                    T : CategoryTheory.Pretriangulated.Triangle (CochainComplex C Int)
                                                                                    n₀ n₁ : Int
                                                                                    h : Eq (HAdd.hAdd n₀ 1) n₁
                                                                                    ⊢ Eq (HAdd.hAdd 1 n₀) n₁
                                                                                  -/
        (HomologicalComplex.homologyFunctor C (up ℤ) 0).shiftMap T.mor₃ n₀ n₁ (by omega) ≫
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/
        (homologyFunctorFactors C (up ℤ) n₁).inv.app _ := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Abelian C
    T : CategoryTheory.Pretriangulated.Triangle (CochainComplex C Int)
    n₀ n₁ : Int
    h : Eq (HAdd.hAdd n₀ 1) n₁
    ⊢ Eq ((HomotopyCategory.homologyFunctor C (ComplexShape.up Int) 0).homologySeq …
  -/
  apply homologyFunctor_shiftMap
  /-
    🎉 no goals
  -/


/-- The canonical morphism `mappingCone S.f ⟶ S.X₃` when `S` is a short complex
of cochain complexes. -/
                                                                                  /-
                                                                                    C : Type u_1
                                                                                    inst✝¹ : CategoryTheory.Category.{?u.2300, u_1} C
                                                                                    inst✝ : CategoryTheory.Abelian C
                                                                                    S : CategoryTheory.ShortComplex (CochainComplex C Int)
                                                                                    hS : S.ShortExact
                                                                                    ⊢ Eq (CochainComplex.HomComplex.δ (-1) 0 0) (CochainComplex.HomComplex.Cochain …
                                                                                  -/
noncomputable def descShortComplex : mappingCone S.f ⟶ S.X₃ := desc S.f 0 S.g (by simp)
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


@[reassoc (attr := simp)]
lemma inr_descShortComplex : inr S.f ≫ descShortComplex S = S.g := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex (CochainComplex C Int)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CochainComplex.mappingCone.inr S.f)  …
  -/
  simp [descShortComplex]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma inr_f_descShortComplex_f (n : ℤ) : (inr S.f).f n ≫ (descShortComplex S).f n = S.g.f n := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex (CochainComplex C Int)
    n : Int
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CochainComplex.mappingCone.inr S.f) …
  -/
  simp [descShortComplex]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma inl_v_descShortComplex_f (i j : ℤ) (h : i + (-1) = j) :
    (inl S.f).v i j h ≫ (descShortComplex S).f j = 0 := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex (CochainComplex C Int)
    i j : Int
    h : Eq (HAdd.hAdd i (-1)) j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CochainComplex.mappingCone.inl S.f) …
  -/
  simp [descShortComplex]
  /-
    🎉 no goals
  -/


lemma homologySequenceδ_triangleh (n₀ : ℤ) (n₁ : ℤ) (h : n₀ + 1 = n₁) :
    (homologyFunctor C (up ℤ) 0).homologySequenceδ (triangleh S.f) n₀ n₁ h =
      (homologyFunctorFactors C (up ℤ) n₀).hom.app _ ≫
        HomologicalComplex.homologyMap (descShortComplex S) n₀ ≫ hS.δ n₀ n₁ h ≫
          (homologyFunctorFactors C (up ℤ) n₁).inv.app _ := by
  /- We proceed by diagram chase. We test the identity on
     cocycles `x' : A' ⟶ (mappingCone S.f).X n₀` -/
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex (CochainComplex C Int)
    hS : S.ShortExact
    n₀ n₁ : Int
    h : Eq (HAdd.hAdd n₀ 1) n₁
    ⊢ Eq ((HomotopyCategory.homologyFunctor C (ComplexShape.up Int) 0).homologySeq …
  -/
  dsimp
  rw [← cancel_mono ((homologyFunctorFactors C (up ℤ) n₁).hom.app _),
    assoc, assoc, assoc, Iso.inv_hom_id_app,
    ← cancel_epi ((homologyFunctorFactors C (up ℤ) n₀).inv.app _), Iso.inv_hom_id_app_assoc]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex (CochainComplex C Int)
    hS : S.ShortExact
    n₀ n₁ : Int
    h : Eq (HAdd.hAdd n₀ 1) n₁
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((HomotopyCategory.homologyFunctorFac …
  -/
  apply yoneda.map_injective
  /-
    case a
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex (CochainComplex C Int)
    hS : S.ShortExact
    n₀ n₁ : Int
    h : Eq (HAdd.hAdd n₀ 1) n₁
    ⊢ Eq (CategoryTheory.yoneda.map (CategoryTheory.CategoryStruct.comp ((Homotopy …
  -/
  ext ⟨A⟩ (x : A ⟶ _)
  obtain ⟨A', π, _, x', w, hx'⟩ :=
    (mappingCone S.f).eq_liftCycles_homologyπ_up_to_refinements x n₁ (by simpa using h)
  /-
    case a.w.h.op.h.intro.intro.intro.intro.intro
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex (CochainComplex C Int)
    hS : S.ShortExact
    n₀ n₁ : Int
    h : Eq (HAdd.hAdd n₀ 1) n₁
    A : C
    x : Quiver.Hom A ((HomologicalComplex.homologyFunctor C (ComplexShape.up Int)  …
    A' : C
    π : Quiver.Hom A' A
    w✝ : CategoryTheory.Epi π
    x' : Quiver.Hom A' ((CochainComplex.mappingCone S.f).X n₀)
    w : Eq (CategoryTheory.CategoryStruct.comp x' ((CochainComplex.mappingCone S.f …
    hx' : Eq (CategoryTheory.CategoryStruct.comp π x) (CategoryTheory.CategoryStru …
    ⊢ Eq ((CategoryTheory.yoneda.map (CategoryTheory.CategoryStruct.comp ((Homotop …
  -/
  erw [homologySequenceδ_quotient_mapTriangle_obj_assoc _ _ _ h]
  /-
    case a.w.h.op.h.intro.intro.intro.intro.intro
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex (CochainComplex C Int)
    hS : S.ShortExact
    n₀ n₁ : Int
    h : Eq (HAdd.hAdd n₀ 1) n₁
    A : C
    x : Quiver.Hom A ((HomologicalComplex.homologyFunctor C (ComplexShape.up Int)  …
    A' : C
    π : Quiver.Hom A' A
    w✝ : CategoryTheory.Epi π
    x' : Quiver.Hom A' ((CochainComplex.mappingCone S.f).X n₀)
    w : Eq (CategoryTheory.CategoryStruct.comp x' ((CochainComplex.mappingCone S.f …
    hx' : Eq (CategoryTheory.CategoryStruct.comp π x) (CategoryTheory.CategoryStru …
    ⊢ Eq ((CategoryTheory.yoneda.map (CategoryTheory.CategoryStruct.comp ((Homotop …
  -/
  dsimp
  /-
    case a.w.h.op.h.intro.intro.intro.intro.intro
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex (CochainComplex C Int)
    hS : S.ShortExact
    n₀ n₁ : Int
    h : Eq (HAdd.hAdd n₀ 1) n₁
    A : C
    x : Quiver.Hom A ((HomologicalComplex.homologyFunctor C (ComplexShape.up Int)  …
    A' : C
    π : Quiver.Hom A' A
    w✝ : CategoryTheory.Epi π
    x' : Quiver.Hom A' ((CochainComplex.mappingCone S.f).X n₀)
    w : Eq (CategoryTheory.CategoryStruct.comp x' ((CochainComplex.mappingCone S.f …
    hx' : Eq (CategoryTheory.CategoryStruct.comp π x) (CategoryTheory.CategoryStru …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp x (CategoryTheory.CategoryStruct.comp …
  -/
  rw [comp_id, Iso.inv_hom_id_app_assoc, Iso.inv_hom_id_app]
  /-
    case a.w.h.op.h.intro.intro.intro.intro.intro
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex (CochainComplex C Int)
    hS : S.ShortExact
    n₀ n₁ : Int
    h : Eq (HAdd.hAdd n₀ 1) n₁
    A : C
    x : Quiver.Hom A ((HomologicalComplex.homologyFunctor C (ComplexShape.up Int)  …
    A' : C
    π : Quiver.Hom A' A
    w✝ : CategoryTheory.Epi π
    x' : Quiver.Hom A' ((CochainComplex.mappingCone S.f).X n₀)
    w : Eq (CategoryTheory.CategoryStruct.comp x' ((CochainComplex.mappingCone S.f …
    hx' : Eq (CategoryTheory.CategoryStruct.comp π x) (CategoryTheory.CategoryStru …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp x (CategoryTheory.CategoryStruct.comp …
  -/
  erw [comp_id]
  rw [← cancel_epi π, reassoc_of% hx', reassoc_of% hx',
    HomologicalComplex.homologyπ_naturality_assoc,
    HomologicalComplex.liftCycles_comp_cyclesMap_assoc]
  /- We decompose the cocycle `x'` into two morphisms `a : A' ⟶ S.X₁.X n₁`
     and `b : A' ⟶ S.X₂.X n₀` satisfying certain relations. -/
  /-
    case a.w.h.op.h.intro.intro.intro.intro.intro
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex (CochainComplex C Int)
    hS : S.ShortExact
    n₀ n₁ : Int
    h : Eq (HAdd.hAdd n₀ 1) n₁
    A : C
    x : Quiver.Hom A ((HomologicalComplex.homologyFunctor C (ComplexShape.up Int)  …
    A' : C
    π : Quiver.Hom A' A
    w✝ : CategoryTheory.Epi π
    x' : Quiver.Hom A' ((CochainComplex.mappingCone S.f).X n₀)
    w : Eq (CategoryTheory.CategoryStruct.comp x' ((CochainComplex.mappingCone S.f …
    hx' : Eq (CategoryTheory.CategoryStruct.comp π x) (CategoryTheory.CategoryStru …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CochainComplex.mappingCone S.f).lif …
  -/
  obtain ⟨a, b, hab⟩ := decomp_to _ x' n₁ h
  rw [hab, ext_to_iff _ n₁ (n₁ + 1) rfl, add_comp, assoc, assoc, inr_f_d, add_comp, assoc,
    assoc, assoc, assoc, inr_f_fst_v, comp_zero, comp_zero, add_zero, zero_comp,
    d_fst_v _ _ _ _ h, comp_neg, inl_v_fst_v_assoc, comp_neg, neg_eq_zero,
    add_comp, assoc, assoc, assoc, assoc, inr_f_snd_v, comp_id, zero_comp,
    d_snd_v _ _ _ h, comp_add, inl_v_fst_v_assoc, inl_v_snd_v_assoc, zero_comp, add_zero] at w
  /- We simplify the RHS. -/
  conv_rhs => simp only [hab, add_comp, assoc, inr_f_descShortComplex_f,
    inl_v_descShortComplex_f, comp_zero, zero_add]
  rw [hS.δ_eq n₀ n₁ (by simpa using h) (b ≫ S.g.f n₀) _ b rfl (-a)
    (by simp only [neg_comp, neg_eq_iff_add_eq_zero, w.2]) (n₁ + 1) (by simp)]
  /- We simplify the LHS. -/
  /-
    case a.w.h.op.h.intro.intro.intro.intro.intro.intro.intro
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex (CochainComplex C Int)
    hS : S.ShortExact
    n₀ n₁ : Int
    h : Eq (HAdd.hAdd n₀ 1) n₁
    A : C
    x : Quiver.Hom A ((HomologicalComplex.homologyFunctor C (ComplexShape.up Int)  …
    A' : C
    π : Quiver.Hom A' A
    w✝¹ : CategoryTheory.Epi π
    x' : Quiver.Hom A' ((CochainComplex.mappingCone S.f).X n₀)
    w✝ : Eq (CategoryTheory.CategoryStruct.comp x' ((CochainComplex.mappingCone S. …
    hx' : Eq (CategoryTheory.CategoryStruct.comp π x) (CategoryTheory.CategoryStru …
    a : Quiver.Hom A' (S.X₁.X n₁)
    b : Quiver.Hom A' (S.X₂.X n₀)
    w : And (Eq (CategoryTheory.CategoryStruct.comp a (S.X₁.d n₁ (HAdd.hAdd n₁ 1)) …
    hab : Eq x' (HAdd.hAdd (CategoryTheory.CategoryStruct.comp a ((CochainComplex. …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CochainComplex.mappingCone S.f).lif …
  -/
  dsimp [Functor.shiftMap, homologyFunctor_shift]
  rw [HomologicalComplex.homologyπ_naturality_assoc,
    HomologicalComplex.liftCycles_comp_cyclesMap_assoc,
    S.X₁.liftCycles_shift_homologyπ_assoc _ _ _ _ n₁ (by omega) (n₁ + 1) (by simp),
    Iso.inv_hom_id_app]
  /-
    case a.w.h.op.h.intro.intro.intro.intro.intro.intro.intro
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex (CochainComplex C Int)
    hS : S.ShortExact
    n₀ n₁ : Int
    h : Eq (HAdd.hAdd n₀ 1) n₁
    A : C
    x : Quiver.Hom A ((HomologicalComplex.homologyFunctor C (ComplexShape.up Int)  …
    A' : C
    π : Quiver.Hom A' A
    w✝¹ : CategoryTheory.Epi π
    x' : Quiver.Hom A' ((CochainComplex.mappingCone S.f).X n₀)
    w✝ : Eq (CategoryTheory.CategoryStruct.comp x' ((CochainComplex.mappingCone S. …
    hx' : Eq (CategoryTheory.CategoryStruct.comp π x) (CategoryTheory.CategoryStru …
    a : Quiver.Hom A' (S.X₁.X n₁)
    b : Quiver.Hom A' (S.X₂.X n₀)
    w : And (Eq (CategoryTheory.CategoryStruct.comp a (S.X₁.d n₁ (HAdd.hAdd n₁ 1)) …
    hab : Eq x' (HAdd.hAdd (CategoryTheory.CategoryStruct.comp a ((CochainComplex. …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.liftCycles S.X₁ ( …
  -/
  dsimp [homologyFunctor_shift]
  simp only [hab, add_comp, assoc, inl_v_triangle_mor₃_f_assoc,
    shiftFunctorObjXIso, neg_comp, Iso.inv_hom_id, comp_neg, comp_id,
    inr_f_triangle_mor₃_f_assoc, zero_comp, comp_zero, add_zero]


include hS in
lemma quasiIso_descShortComplex : QuasiIso (descShortComplex S) where
  quasiIsoAt n := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Abelian C
      S : CategoryTheory.ShortComplex (CochainComplex C Int)
      hS : S.ShortExact
      n : Int
      ⊢ QuasiIsoAt (CochainComplex.mappingCone.descShortComplex S) n
    -/
    rw [quasiIsoAt_iff_isIso_homologyMap]
    let φ : ((homologyFunctor C (up ℤ) 0).homologySequenceComposableArrows₅
        (triangleh S.f) n _ rfl).δlast ⟶ (composableArrows₅ hS n _ rfl).δlast :=
      homMk₄ ((homologyFunctorFactors C (up ℤ) _).hom.app _)
        ((homologyFunctorFactors C (up ℤ) _).hom.app _)
        ((homologyFunctorFactors C (up ℤ) _).hom.app _ ≫
          HomologicalComplex.homologyMap (descShortComplex S) n)
        ((homologyFunctorFactors C (up ℤ) _).hom.app _)
        ((homologyFunctorFactors C (up ℤ) _).hom.app _)
        ((homologyFunctorFactors C (up ℤ) _).hom.naturality S.f)
        (by
          erw [(homologyFunctorFactors C (up ℤ) n).hom.naturality_assoc]
          dsimp
          rw [← HomologicalComplex.homologyMap_comp, inr_descShortComplex])
        (by
          dsimp
          erw [homologySequenceδ_triangleh hS]
          simp only [Functor.comp_obj, HomologicalComplex.homologyFunctor_obj, assoc,
            Iso.inv_hom_id_app, comp_id])
        ((homologyFunctorFactors C (up ℤ) _).hom.naturality S.f)
    have : IsIso ((homologyFunctorFactors C (up ℤ) n).hom.app (mappingCone S.f) ≫
        HomologicalComplex.homologyMap (descShortComplex S) n) := by
      apply Abelian.isIso_of_epi_of_isIso_of_isIso_of_mono
        ((homologyFunctor C (up ℤ) 0).homologySequenceComposableArrows₅_exact _
          (mappingCone_triangleh_distinguished S.f) n _ rfl).δlast
        (composableArrows₅_exact hS n _ rfl).δlast φ
      all_goals dsimp [φ]; infer_instance
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Abelian C
      S : CategoryTheory.ShortComplex (CochainComplex C Int)
      hS : S.ShortExact
      n : Int
      φ : Quiver.Hom ((HomotopyCategory.homologyFunctor C (ComplexShape.up Int) 0).h …
      this : CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp ((HomotopyCate …
      ⊢ CategoryTheory.IsIso (HomologicalComplex.homologyMap (CochainComplex.mapping …
    -/
    apply IsIso.of_isIso_comp_left ((homologyFunctorFactors C (up ℤ) n).hom.app (mappingCone S.f))
    /-
      🎉 no goals
    -/


