/-- The morphism `K.opcycles i ⟶ K.cycles j` that is induced by `K.d i j`. -/
noncomputable def opcyclesToCycles [K.HasHomology i] [K.HasHomology j] :
    K.opcycles i ⟶ K.cycles j :=
                                              /-
                                                C : Type u_1
                                                ι : Type u_2
                                                inst✝⁷ : CategoryTheory.Category.{?u.291, u_1} C
                                                inst✝⁶ : CategoryTheory.Limits.HasZeroMorphisms C
                                                c : ComplexShape ι
                                                K L : HomologicalComplex C c
                                                φ : Quiver.Hom K L
                                                i j : ι
                                                inst✝⁵ : K.HasHomology i
                                                inst✝⁴ : K.HasHomology j
                                                inst✝³ : L.HasHomology i
                                                inst✝² : L.HasHomology j
                                                inst✝¹ : K.HasHomology i
                                                inst✝ : K.HasHomology j
                                                ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.fromOpcycles i j) (K.d j (c.next j …
                                              -/
  K.liftCycles (K.fromOpcycles i j) _ rfl (by simp)
                                              /-
                                                🎉 no goals
                                              -/


@[reassoc (attr := simp)]
lemma opcyclesToCycles_iCycles : K.opcyclesToCycles i j ≫ K.iCycles j = K.fromOpcycles i j := by
  /-
    C : Type u_1
    ι : Type u_2
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    c : ComplexShape ι
    K : HomologicalComplex C c
    i j : ι
    inst✝¹ : K.HasHomology i
    inst✝ : K.HasHomology j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.opcyclesToCycles i j) (K.iCycles j …
  -/
  dsimp only [opcyclesToCycles]
  /-
    C : Type u_1
    ι : Type u_2
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    c : ComplexShape ι
    K : HomologicalComplex C c
    i j : ι
    inst✝¹ : K.HasHomology i
    inst✝ : K.HasHomology j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.liftCycles (K.fromOpcycles i j) (c …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc]
lemma pOpcycles_opcyclesToCycles_iCycles :
    K.pOpcycles i ≫ K.opcyclesToCycles i j ≫ K.iCycles j = K.d i j := by
  /-
    C : Type u_1
    ι : Type u_2
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    c : ComplexShape ι
    K : HomologicalComplex C c
    i j : ι
    inst✝¹ : K.HasHomology i
    inst✝ : K.HasHomology j
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (K.pOpcycles i) (CategoryTheory.Categ …
  -/
  simp [opcyclesToCycles]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma pOpcycles_opcyclesToCycles :
    K.pOpcycles i ≫ K.opcyclesToCycles i j = K.toCycles i j := by
  simp only [← cancel_mono (K.iCycles j), assoc, opcyclesToCycles_iCycles,
    p_fromOpcycles, toCycles_i]


@[reassoc (attr := simp)]
lemma homologyι_opcyclesToCycles :
    K.homologyι i ≫ K.opcyclesToCycles i j = 0 := by
  simp only [← cancel_mono (K.iCycles j), assoc, opcyclesToCycles_iCycles,
    homologyι_comp_fromOpcycles, zero_comp]


@[reassoc (attr := simp)]
lemma opcyclesToCycles_homologyπ :
    K.opcyclesToCycles i j ≫ K.homologyπ j = 0 := by
  simp only [← cancel_epi (K.pOpcycles i),
    pOpcycles_opcyclesToCycles_assoc, toCycles_comp_homologyπ, comp_zero]


@[reassoc (attr := simp)]
lemma opcyclesToCycles_naturality :
    opcyclesMap φ i ≫ opcyclesToCycles L i j = opcyclesToCycles K i j ≫ cyclesMap φ j := by
  simp only [← cancel_mono (L.iCycles j), ← cancel_epi (K.pOpcycles i),
    assoc, p_opcyclesMap_assoc, pOpcycles_opcyclesToCycles_iCycles, Hom.comm, cyclesMap_i,
    pOpcycles_opcyclesToCycles_iCycles_assoc]


/-- The natural transformation `K.opcyclesToCycles i j : K.opcycles i ⟶ K.cycles j` for all
`K : HomologicalComplex C c`. -/
@[simps]
noncomputable def natTransOpCyclesToCycles [CategoryWithHomology C] :
    opcyclesFunctor C c i ⟶ cyclesFunctor C c j where
  app K := K.opcyclesToCycles i j


/-- The diagram `K.homology i ⟶ K.opcycles i ⟶ K.cycles j ⟶ K.homology j`. -/
@[simp]
noncomputable def composableArrows₃ [K.HasHomology i] [K.HasHomology j] :
    ComposableArrows C 3 :=
  ComposableArrows.mk₃ (K.homologyι i) (K.opcyclesToCycles i j) (K.homologyπ j)


instance [K.HasHomology i] [K.HasHomology j] :
          /-
            C : Type u_1
            ι : Type u_2
            inst✝³ : CategoryTheory.Category.{?u.12905, u_1} C
            inst✝² : CategoryTheory.Preadditive C
            c : ComplexShape ι
            K : HomologicalComplex C c
            i j : ι
            hij : c.Rel i j
            inst✝¹ : K.HasHomology i
            inst✝ : K.HasHomology j
            ⊢ LE.le 0 1
          -/
          /-
            🎉 no goals
          -/
    Mono ((composableArrows₃ K i j).map' 0 1) := by
          /-
            🎉 no goals
          -/
  /-
    C : Type u_1
    ι : Type u_2
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    c : ComplexShape ι
    K : HomologicalComplex C c
    i j : ι
    hij : c.Rel i j
    inst✝¹ : K.HasHomology i
    inst✝ : K.HasHomology j
    ⊢ CategoryTheory.Mono ((HomologicalComplex.HomologySequence.composableArrows₃  …
  -/
  dsimp
  /-
    C : Type u_1
    ι : Type u_2
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    c : ComplexShape ι
    K : HomologicalComplex C c
    i j : ι
    hij : c.Rel i j
    inst✝¹ : K.HasHomology i
    inst✝ : K.HasHomology j
    ⊢ CategoryTheory.Mono (K.homologyι i)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


set_option simprocs false in
instance [K.HasHomology i] [K.HasHomology j] :
         /-
           C : Type u_1
           ι : Type u_2
           inst✝³ : CategoryTheory.Category.{?u.14519, u_1} C
           inst✝² : CategoryTheory.Preadditive C
           c : ComplexShape ι
           K : HomologicalComplex C c
           i j : ι
           hij : c.Rel i j
           inst✝¹ : K.HasHomology i
           inst✝ : K.HasHomology j
           ⊢ LE.le 2 3
         -/
         /-
           🎉 no goals
         -/
    Epi ((composableArrows₃ K i j).map' 2 3) := by
         /-
           🎉 no goals
         -/
  /-
    C : Type u_1
    ι : Type u_2
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    c : ComplexShape ι
    K : HomologicalComplex C c
    i j : ι
    hij : c.Rel i j
    inst✝¹ : K.HasHomology i
    inst✝ : K.HasHomology j
    ⊢ CategoryTheory.Epi ((HomologicalComplex.HomologySequence.composableArrows₃ K …
  -/
  dsimp
  /-
    C : Type u_1
    ι : Type u_2
    inst✝³ : CategoryTheory.Category.{u_3, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    c : ComplexShape ι
    K : HomologicalComplex C c
    i j : ι
    hij : c.Rel i j
    inst✝¹ : K.HasHomology i
    inst✝ : K.HasHomology j
    ⊢ CategoryTheory.Epi (K.homologyπ j)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


include hij in
/-- The diagram `K.homology i ⟶ K.opcycles i ⟶ K.cycles j ⟶ K.homology j` is exact
when `c.Rel i j`. -/
lemma composableArrows₃_exact [CategoryWithHomology C] :
    (composableArrows₃ K i j).Exact := by
  /-
    C : Type u_1
    ι : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    c : ComplexShape ι
    K : HomologicalComplex C c
    i j : ι
    hij : c.Rel i j
    inst✝ : CategoryTheory.CategoryWithHomology C
    ⊢ (HomologicalComplex.HomologySequence.composableArrows₃ K i j).Exact
  -/
  let S := ShortComplex.mk (K.homologyι i) (K.opcyclesToCycles i j) (by simp)
  /-
    C : Type u_1
    ι : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    c : ComplexShape ι
    K : HomologicalComplex C c
    i j : ι
    hij : c.Rel i j
    inst✝ : CategoryTheory.CategoryWithHomology C
    S : CategoryTheory.ShortComplex C := CategoryTheory.ShortComplex.mk (K.homolog …
    ⊢ (HomologicalComplex.HomologySequence.composableArrows₃ K i j).Exact
  -/
  let S' := ShortComplex.mk (K.homologyι i) (K.fromOpcycles i j) (by simp)
  let ι : S ⟶ S' :=
    { τ₁ := 𝟙 _
      τ₂ := 𝟙 _
      τ₃ := K.iCycles j }
  have hS : S.Exact := by
    rw [ShortComplex.exact_iff_of_epi_of_isIso_of_mono ι]
    exact S'.exact_of_f_is_kernel (K.homologyIsKernel i j (c.next_eq' hij))
  /-
    C : Type u_1
    ι✝ : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    c : ComplexShape ι✝
    K : HomologicalComplex C c
    i j : ι✝
    hij : c.Rel i j
    inst✝ : CategoryTheory.CategoryWithHomology C
    S : CategoryTheory.ShortComplex C := CategoryTheory.ShortComplex.mk (K.homolog …
    S' : CategoryTheory.ShortComplex C := CategoryTheory.ShortComplex.mk (K.homolo …
    ι : Quiver.Hom S S' := { τ₁ := CategoryTheory.CategoryStruct.id S.X₁, τ₂ := Ca …
    hS : S.Exact
    ⊢ (HomologicalComplex.HomologySequence.composableArrows₃ K i j).Exact
  -/
  let T := ShortComplex.mk (K.opcyclesToCycles i j) (K.homologyπ j) (by simp)
  /-
    C : Type u_1
    ι✝ : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    c : ComplexShape ι✝
    K : HomologicalComplex C c
    i j : ι✝
    hij : c.Rel i j
    inst✝ : CategoryTheory.CategoryWithHomology C
    S : CategoryTheory.ShortComplex C := CategoryTheory.ShortComplex.mk (K.homolog …
    S' : CategoryTheory.ShortComplex C := CategoryTheory.ShortComplex.mk (K.homolo …
    ι : Quiver.Hom S S' := { τ₁ := CategoryTheory.CategoryStruct.id S.X₁, τ₂ := Ca …
    hS : S.Exact
    T : CategoryTheory.ShortComplex C := CategoryTheory.ShortComplex.mk (K.opcycle …
    ⊢ (HomologicalComplex.HomologySequence.composableArrows₃ K i j).Exact
  -/
  let T' := ShortComplex.mk (K.toCycles i j) (K.homologyπ j) (by simp)
  let π : T' ⟶ T :=
    { τ₁ := K.pOpcycles i
      τ₂ := 𝟙 _
      τ₃ := 𝟙 _ }
  have hT : T.Exact := by
    rw [← ShortComplex.exact_iff_of_epi_of_isIso_of_mono π]
    exact T'.exact_of_g_is_cokernel (K.homologyIsCokernel i j (c.prev_eq' hij))
  /-
    C : Type u_1
    ι✝ : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    c : ComplexShape ι✝
    K : HomologicalComplex C c
    i j : ι✝
    hij : c.Rel i j
    inst✝ : CategoryTheory.CategoryWithHomology C
    S : CategoryTheory.ShortComplex C := CategoryTheory.ShortComplex.mk (K.homolog …
    S' : CategoryTheory.ShortComplex C := CategoryTheory.ShortComplex.mk (K.homolo …
    ι : Quiver.Hom S S' := { τ₁ := CategoryTheory.CategoryStruct.id S.X₁, τ₂ := Ca …
    hS : S.Exact
    T : CategoryTheory.ShortComplex C := CategoryTheory.ShortComplex.mk (K.opcycle …
    T' : CategoryTheory.ShortComplex C := CategoryTheory.ShortComplex.mk (K.toCycl …
    π : Quiver.Hom T' T := { τ₁ := K.pOpcycles i, τ₂ := CategoryTheory.CategoryStr …
    hT : T.Exact
    ⊢ (HomologicalComplex.HomologySequence.composableArrows₃ K i j).Exact
  -/
  apply ComposableArrows.exact_of_δ₀
    /-
      case h
      C : Type u_1
      ι✝ : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      c : ComplexShape ι✝
      K : HomologicalComplex C c
      i j : ι✝
      hij : c.Rel i j
      inst✝ : CategoryTheory.CategoryWithHomology C
      S : CategoryTheory.ShortComplex C := CategoryTheory.ShortComplex.mk (K.homolog …
      S' : CategoryTheory.ShortComplex C := CategoryTheory.ShortComplex.mk (K.homolo …
      ι : Quiver.Hom S S' := { τ₁ := CategoryTheory.CategoryStruct.id S.X₁, τ₂ := Ca …
      hS : S.Exact
      T : CategoryTheory.ShortComplex C := CategoryTheory.ShortComplex.mk (K.opcycle …
      T' : CategoryTheory.ShortComplex C := CategoryTheory.ShortComplex.mk (K.toCycl …
      π : Quiver.Hom T' T := { τ₁ := K.pOpcycles i, τ₂ := CategoryTheory.CategoryStr …
      hT : T.Exact
      ⊢ (CategoryTheory.ComposableArrows.mk₂ ((HomologicalComplex.HomologySequence.c …
    -/
  · exact hS.exact_toComposableArrows
    /-
      🎉 no goals
    -/
    /-
      case h₀
      C : Type u_1
      ι✝ : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_1} C
      inst✝¹ : CategoryTheory.Preadditive C
      c : ComplexShape ι✝
      K : HomologicalComplex C c
      i j : ι✝
      hij : c.Rel i j
      inst✝ : CategoryTheory.CategoryWithHomology C
      S : CategoryTheory.ShortComplex C := CategoryTheory.ShortComplex.mk (K.homolog …
      S' : CategoryTheory.ShortComplex C := CategoryTheory.ShortComplex.mk (K.homolo …
      ι : Quiver.Hom S S' := { τ₁ := CategoryTheory.CategoryStruct.id S.X₁, τ₂ := Ca …
      hS : S.Exact
      T : CategoryTheory.ShortComplex C := CategoryTheory.ShortComplex.mk (K.opcycle …
      T' : CategoryTheory.ShortComplex C := CategoryTheory.ShortComplex.mk (K.toCycl …
      π : Quiver.Hom T' T := { τ₁ := K.pOpcycles i, τ₂ := CategoryTheory.CategoryStr …
      hT : T.Exact
      ⊢ (HomologicalComplex.HomologySequence.composableArrows₃ K i j).δ₀.Exact
    -/
  · exact hT.exact_toComposableArrows
    /-
      🎉 no goals
    -/


set_option simprocs false in
/-- The functor `HomologicalComplex C c ⥤ ComposableArrows C 3` that maps `K` to the
diagram `K.homology i ⟶ K.opcycles i ⟶ K.cycles j ⟶ K.homology j`. -/
@[simps]
noncomputable def composableArrows₃Functor [CategoryWithHomology C] :
    HomologicalComplex C c ⥤ ComposableArrows C 3 where
  obj K := composableArrows₃ K i j
  map {K L} φ := ComposableArrows.homMk₃ (homologyMap φ i) (opcyclesMap φ i) (cyclesMap φ j)
                          /-
                            C : Type u_1
                            ι : Type u_2
                            inst✝² : CategoryTheory.Category.{?u.22111, u_1} C
                            inst✝¹ : CategoryTheory.Preadditive C
                            c : ComplexShape ι
                            K✝ : HomologicalComplex C c
                            i j : ι
                            hij : c.Rel i j
                            inst✝ : CategoryTheory.CategoryWithHomology C
                            K L : HomologicalComplex C c
                            φ : Quiver.Hom K L
                            ⊢ Eq (CategoryTheory.CategoryStruct.comp (((fun K => HomologicalComplex.Homolo …
                          -/
                          /-
                            🎉 no goals
                          -/
                                         /-
                                           🎉 no goals
                                         -/
    (homologyMap φ j) (by aesop_cat) (by aesop_cat) (by aesop_cat)
                                                        /-
                                                          🎉 no goals
                                                        -/


/-- If `X₁ ⟶ X₂ ⟶ X₃ ⟶ 0` is an exact sequence of homological complexes, then
`X₁.opcycles i ⟶ X₂.opcycles i ⟶ X₃.opcycles i ⟶ 0` is exact. This lemma states
the exactness at `X₂.opcycles i`, while the fact that `X₂.opcycles i ⟶ X₃.opcycles i`
is an epi is an instance. -/
lemma opcycles_right_exact (S : ShortComplex (HomologicalComplex C c)) (hS : S.Exact) [Epi S.g]
    (i : ι) [S.X₁.HasHomology i] [S.X₂.HasHomology i] [S.X₃.HasHomology i] :
    (ShortComplex.mk (opcyclesMap S.f i) (opcyclesMap S.g i)
          /-
            C : Type u_1
            ι : Type u_2
            inst✝⁵ : CategoryTheory.Category.{?u.58000, u_1} C
            inst✝⁴ : CategoryTheory.Abelian C
            c : ComplexShape ι
            S : CategoryTheory.ShortComplex (HomologicalComplex C c)
            hS : S.Exact
            inst✝³ : CategoryTheory.Epi S.g
            i : ι
            inst✝² : S.X₁.HasHomology i
            inst✝¹ : S.X₂.HasHomology i
            inst✝ : S.X₃.HasHomology i
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.opcyclesMap S.f i …
          -/
      (by rw [← opcyclesMap_comp, S.zero, opcyclesMap_zero])).Exact := by
          /-
            🎉 no goals
          -/
  /-
    C : Type u_1
    ι : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁴ : CategoryTheory.Abelian C
    c : ComplexShape ι
    S : CategoryTheory.ShortComplex (HomologicalComplex C c)
    hS : S.Exact
    inst✝³ : CategoryTheory.Epi S.g
    i : ι
    inst✝² : S.X₁.HasHomology i
    inst✝¹ : S.X₂.HasHomology i
    inst✝ : S.X₃.HasHomology i
    ⊢ (CategoryTheory.ShortComplex.mk (HomologicalComplex.opcyclesMap S.f i) (Homo …
  -/
  have : Epi (ShortComplex.map S (eval C c i)).g := by dsimp; infer_instance
  /-
    C : Type u_1
    ι : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁴ : CategoryTheory.Abelian C
    c : ComplexShape ι
    S : CategoryTheory.ShortComplex (HomologicalComplex C c)
    hS : S.Exact
    inst✝³ : CategoryTheory.Epi S.g
    i : ι
    inst✝² : S.X₁.HasHomology i
    inst✝¹ : S.X₂.HasHomology i
    inst✝ : S.X₃.HasHomology i
    this : CategoryTheory.Epi (S.map (HomologicalComplex.eval C c i)).g
    ⊢ (CategoryTheory.ShortComplex.mk (HomologicalComplex.opcyclesMap S.f i) (Homo …
  -/
  have hj := (hS.map (HomologicalComplex.eval C c i)).gIsCokernel
  /-
    C : Type u_1
    ι : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁴ : CategoryTheory.Abelian C
    c : ComplexShape ι
    S : CategoryTheory.ShortComplex (HomologicalComplex C c)
    hS : S.Exact
    inst✝³ : CategoryTheory.Epi S.g
    i : ι
    inst✝² : S.X₁.HasHomology i
    inst✝¹ : S.X₂.HasHomology i
    inst✝ : S.X₃.HasHomology i
    this : CategoryTheory.Epi (S.map (HomologicalComplex.eval C c i)).g
    hj : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ …
    ⊢ (CategoryTheory.ShortComplex.mk (HomologicalComplex.opcyclesMap S.f i) (Homo …
  -/
  apply ShortComplex.exact_of_g_is_cokernel
  refine CokernelCofork.IsColimit.ofπ' _ _  (fun {A} k hk => by
    dsimp at k hk ⊢
    have H := CokernelCofork.IsColimit.desc' hj (S.X₂.pOpcycles i ≫ k) (by
      dsimp
      rw [← p_opcyclesMap_assoc, hk, comp_zero])
    dsimp at H
    refine ⟨S.X₃.descOpcycles H.1 _ rfl ?_, ?_⟩
    · rw [← cancel_epi (S.g.f (c.prev i)), comp_zero, Hom.comm_assoc, H.2,
        d_pOpcycles_assoc, zero_comp]
    · rw [← cancel_epi (S.X₂.pOpcycles i), opcyclesMap_comp_descOpcycles, p_descOpcycles, H.2])


/-- If `0 ⟶ X₁ ⟶ X₂ ⟶ X₃` is an exact sequence of homological complex, then
`0 ⟶ X₁.cycles i ⟶ X₂.cycles i ⟶ X₃.cycles i` is exact. This lemma states
the exactness at `X₂.cycles i`, while the fact that `X₁.cycles i ⟶ X₂.cycles i`
is a mono is an instance. -/
lemma cycles_left_exact (S : ShortComplex (HomologicalComplex C c)) (hS : S.Exact) [Mono S.f]
    (i : ι) [S.X₁.HasHomology i] [S.X₂.HasHomology i] [S.X₃.HasHomology i] :
    (ShortComplex.mk (cyclesMap S.f i) (cyclesMap S.g i)
          /-
            C : Type u_1
            ι : Type u_2
            inst✝⁵ : CategoryTheory.Category.{?u.67031, u_1} C
            inst✝⁴ : CategoryTheory.Abelian C
            c : ComplexShape ι
            S : CategoryTheory.ShortComplex (HomologicalComplex C c)
            hS : S.Exact
            inst✝³ : CategoryTheory.Mono S.f
            i : ι
            inst✝² : S.X₁.HasHomology i
            inst✝¹ : S.X₂.HasHomology i
            inst✝ : S.X₃.HasHomology i
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (HomologicalComplex.cyclesMap S.f i)  …
          -/
      (by rw [← cyclesMap_comp, S.zero, cyclesMap_zero])).Exact := by
          /-
            🎉 no goals
          -/
  /-
    C : Type u_1
    ι : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁴ : CategoryTheory.Abelian C
    c : ComplexShape ι
    S : CategoryTheory.ShortComplex (HomologicalComplex C c)
    hS : S.Exact
    inst✝³ : CategoryTheory.Mono S.f
    i : ι
    inst✝² : S.X₁.HasHomology i
    inst✝¹ : S.X₂.HasHomology i
    inst✝ : S.X₃.HasHomology i
    ⊢ (CategoryTheory.ShortComplex.mk (HomologicalComplex.cyclesMap S.f i) (Homolo …
  -/
  have : Mono (ShortComplex.map S (eval C c i)).f := by dsimp; infer_instance
  /-
    C : Type u_1
    ι : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁴ : CategoryTheory.Abelian C
    c : ComplexShape ι
    S : CategoryTheory.ShortComplex (HomologicalComplex C c)
    hS : S.Exact
    inst✝³ : CategoryTheory.Mono S.f
    i : ι
    inst✝² : S.X₁.HasHomology i
    inst✝¹ : S.X₂.HasHomology i
    inst✝ : S.X₃.HasHomology i
    this : CategoryTheory.Mono (S.map (HomologicalComplex.eval C c i)).f
    ⊢ (CategoryTheory.ShortComplex.mk (HomologicalComplex.cyclesMap S.f i) (Homolo …
  -/
  have hi := (hS.map (HomologicalComplex.eval C c i)).fIsKernel
  /-
    C : Type u_1
    ι : Type u_2
    inst✝⁵ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁴ : CategoryTheory.Abelian C
    c : ComplexShape ι
    S : CategoryTheory.ShortComplex (HomologicalComplex C c)
    hS : S.Exact
    inst✝³ : CategoryTheory.Mono S.f
    i : ι
    inst✝² : S.X₁.HasHomology i
    inst✝¹ : S.X₂.HasHomology i
    inst✝ : S.X₃.HasHomology i
    this : CategoryTheory.Mono (S.map (HomologicalComplex.eval C c i)).f
    hi : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι (S.ma …
    ⊢ (CategoryTheory.ShortComplex.mk (HomologicalComplex.cyclesMap S.f i) (Homolo …
  -/
  apply ShortComplex.exact_of_f_is_kernel
  exact KernelFork.IsLimit.ofι' _ _ (fun {A} k hk => by
    dsimp at k hk ⊢
    have H := KernelFork.IsLimit.lift' hi (k ≫ S.X₂.iCycles i) (by
      dsimp
      rw [assoc, ← cyclesMap_i, reassoc_of% hk, zero_comp])
    dsimp at H
    refine ⟨S.X₁.liftCycles H.1 _ rfl ?_, ?_⟩
    · rw [← cancel_mono (S.f.f _), assoc, zero_comp, ← Hom.comm, reassoc_of% H.2,
        iCycles_d, comp_zero]
    · rw [← cancel_mono (S.X₂.iCycles i), liftCycles_comp_cyclesMap, liftCycles_i, H.2])


/-- Given a short exact short complex `S : HomologicalComplex C c`, and degrees `i` and `j`
such that `c.Rel i j`, this is the snake diagram whose four lines are respectively
obtained by applying the functors `homologyFunctor C c i`, `opcyclesFunctor C c i`,
`cyclesFunctor C c j`, `homologyFunctor C c j` to `S`. Applying the snake lemma to this
gives the homology sequence of `S`. -/
@[simps]
noncomputable def snakeInput (hS : S.ShortExact) (i j : ι) (hij : c.Rel i j) :
    ShortComplex.SnakeInput C where
  L₀ := (homologyFunctor C c i).mapShortComplex.obj S
  L₁ := (opcyclesFunctor C c i).mapShortComplex.obj S
  L₂ := (cyclesFunctor C c j).mapShortComplex.obj S
  L₃ := (homologyFunctor C c j).mapShortComplex.obj S
  v₀₁ := S.mapNatTrans (natTransHomologyι C c i)
  v₁₂ := S.mapNatTrans (natTransOpCyclesToCycles C c i j)
  v₂₃ := S.mapNatTrans (natTransHomologyπ C c j)
  h₀ := by
    /-
      C : Type u_1
      ι : Type u_2
      inst✝¹ : CategoryTheory.Category.{?u.78762, u_1} C
      inst✝ : CategoryTheory.Abelian C
      c : ComplexShape ι
      S : CategoryTheory.ShortComplex (HomologicalComplex C c)
      hS✝ : S.ShortExact
      i✝ j✝ : ι
      hij✝ : c.Rel i✝ j✝
      hS : S.ShortExact
      i j : ι
      hij : c.Rel i j
      ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι (S.mapNa …
    -/
    apply ShortComplex.isLimitOfIsLimitπ
    all_goals
      exact (KernelFork.isLimitMapConeEquiv _ _).symm
        ((composableArrows₃_exact _ i j hij).exact 0).fIsKernel
  h₃ := by
    /-
      C : Type u_1
      ι : Type u_2
      inst✝¹ : CategoryTheory.Category.{?u.78762, u_1} C
      inst✝ : CategoryTheory.Abelian C
      c : ComplexShape ι
      S : CategoryTheory.ShortComplex (HomologicalComplex C c)
      hS✝ : S.ShortExact
      i✝ j✝ : ι
      hij✝ : c.Rel i✝ j✝
      hS : S.ShortExact
      i j : ι
      hij : c.Rel i j
      ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ (S …
    -/
    apply ShortComplex.isColimitOfIsColimitπ
    all_goals
      exact (CokernelCofork.isColimitMapCoconeEquiv _ _).symm
        ((composableArrows₃_exact _ i j hij).exact 1).gIsCokernel
  L₁_exact := by
    /-
      C : Type u_1
      ι : Type u_2
      inst✝¹ : CategoryTheory.Category.{?u.78762, u_1} C
      inst✝ : CategoryTheory.Abelian C
      c : ComplexShape ι
      S : CategoryTheory.ShortComplex (HomologicalComplex C c)
      hS✝ : S.ShortExact
      i✝ j✝ : ι
      hij✝ : c.Rel i✝ j✝
      hS : S.ShortExact
      i j : ι
      hij : c.Rel i j
      ⊢ ((HomologicalComplex.opcyclesFunctor C c i).mapShortComplex.obj S).Exact
    -/
    have := hS.epi_g
    /-
      C : Type u_1
      ι : Type u_2
      inst✝¹ : CategoryTheory.Category.{?u.78762, u_1} C
      inst✝ : CategoryTheory.Abelian C
      c : ComplexShape ι
      S : CategoryTheory.ShortComplex (HomologicalComplex C c)
      hS✝ : S.ShortExact
      i✝ j✝ : ι
      hij✝ : c.Rel i✝ j✝
      hS : S.ShortExact
      i j : ι
      hij : c.Rel i j
      this : CategoryTheory.Epi S.g
      ⊢ ((HomologicalComplex.opcyclesFunctor C c i).mapShortComplex.obj S).Exact
    -/
    exact opcycles_right_exact S hS.exact i
    /-
      🎉 no goals
    -/
  L₂_exact := by
    /-
      C : Type u_1
      ι : Type u_2
      inst✝¹ : CategoryTheory.Category.{?u.78762, u_1} C
      inst✝ : CategoryTheory.Abelian C
      c : ComplexShape ι
      S : CategoryTheory.ShortComplex (HomologicalComplex C c)
      hS✝ : S.ShortExact
      i✝ j✝ : ι
      hij✝ : c.Rel i✝ j✝
      hS : S.ShortExact
      i j : ι
      hij : c.Rel i j
      ⊢ ((HomologicalComplex.cyclesFunctor C c j).mapShortComplex.obj S).Exact
    -/
    have := hS.mono_f
    /-
      C : Type u_1
      ι : Type u_2
      inst✝¹ : CategoryTheory.Category.{?u.78762, u_1} C
      inst✝ : CategoryTheory.Abelian C
      c : ComplexShape ι
      S : CategoryTheory.ShortComplex (HomologicalComplex C c)
      hS✝ : S.ShortExact
      i✝ j✝ : ι
      hij✝ : c.Rel i✝ j✝
      hS : S.ShortExact
      i j : ι
      hij : c.Rel i j
      this : CategoryTheory.Mono S.f
      ⊢ ((HomologicalComplex.cyclesFunctor C c j).mapShortComplex.obj S).Exact
    -/
    /-
      C : Type u_1
      ι : Type u_2
      inst✝¹ : CategoryTheory.Category.{?u.78762, u_1} C
      inst✝ : CategoryTheory.Abelian C
      c : ComplexShape ι
      S : CategoryTheory.ShortComplex (HomologicalComplex C c)
      hS✝ : S.ShortExact
      i✝ j✝ : ι
      hij✝ : c.Rel i✝ j✝
      hS : S.ShortExact
      i j : ι
      hij : c.Rel i j
      ⊢ CategoryTheory.Epi ((HomologicalComplex.opcyclesFunctor C c i).mapShortCompl …
    -/
    exact cycles_left_exact S hS.exact j
    /-
      C : Type u_1
      ι : Type u_2
      inst✝¹ : CategoryTheory.Category.{?u.78762, u_1} C
      inst✝ : CategoryTheory.Abelian C
      c : ComplexShape ι
      S : CategoryTheory.ShortComplex (HomologicalComplex C c)
      hS✝ : S.ShortExact
      i✝ j✝ : ι
      hij✝ : c.Rel i✝ j✝
      hS : S.ShortExact
      i j : ι
      hij : c.Rel i j
      this : CategoryTheory.Epi S.g
      ⊢ CategoryTheory.Epi ((HomologicalComplex.opcyclesFunctor C c i).mapShortCompl …
    -/
    /-
      🎉 no goals
    -/
    /-
      C : Type u_1
      ι : Type u_2
      inst✝¹ : CategoryTheory.Category.{?u.78762, u_1} C
      inst✝ : CategoryTheory.Abelian C
      c : ComplexShape ι
      S : CategoryTheory.ShortComplex (HomologicalComplex C c)
      hS✝ : S.ShortExact
      i✝ j✝ : ι
      hij✝ : c.Rel i✝ j✝
      hS : S.ShortExact
      i j : ι
      hij : c.Rel i j
      this : CategoryTheory.Epi S.g
      ⊢ CategoryTheory.Epi (HomologicalComplex.opcyclesMap S.g i)
    -/
  epi_L₁_g := by
    /-
      🎉 no goals
    -/
    have := hS.epi_g
    dsimp
    infer_instance
  mono_L₂_f := by
    /-
      C : Type u_1
      ι : Type u_2
      inst✝¹ : CategoryTheory.Category.{?u.78762, u_1} C
      inst✝ : CategoryTheory.Abelian C
      c : ComplexShape ι
      S : CategoryTheory.ShortComplex (HomologicalComplex C c)
      hS✝ : S.ShortExact
      i✝ j✝ : ι
      hij✝ : c.Rel i✝ j✝
      hS : S.ShortExact
      i j : ι
      hij : c.Rel i j
      ⊢ CategoryTheory.Mono ((HomologicalComplex.cyclesFunctor C c j).mapShortComple …
    -/
    have := hS.mono_f
    /-
      C : Type u_1
      ι : Type u_2
      inst✝¹ : CategoryTheory.Category.{?u.78762, u_1} C
      inst✝ : CategoryTheory.Abelian C
      c : ComplexShape ι
      S : CategoryTheory.ShortComplex (HomologicalComplex C c)
      hS✝ : S.ShortExact
      i✝ j✝ : ι
      hij✝ : c.Rel i✝ j✝
      hS : S.ShortExact
      i j : ι
      hij : c.Rel i j
      this : CategoryTheory.Mono S.f
      ⊢ CategoryTheory.Mono ((HomologicalComplex.cyclesFunctor C c j).mapShortComple …
    -/
    dsimp
    /-
      C : Type u_1
      ι : Type u_2
      inst✝¹ : CategoryTheory.Category.{?u.78762, u_1} C
      inst✝ : CategoryTheory.Abelian C
      c : ComplexShape ι
      S : CategoryTheory.ShortComplex (HomologicalComplex C c)
      hS✝ : S.ShortExact
      i✝ j✝ : ι
      hij✝ : c.Rel i✝ j✝
      hS : S.ShortExact
      i j : ι
      hij : c.Rel i j
      this : CategoryTheory.Mono S.f
      ⊢ CategoryTheory.Mono (HomologicalComplex.cyclesMap S.f j)
    -/
    infer_instance
    /-
      🎉 no goals
    -/


/-- The connecting homoomorphism `S.X₃.homology i ⟶ S.X₁.homology j` for a short exact
short complex `S`. -/
noncomputable def δ : S.X₃.homology i ⟶ S.X₁.homology j := (snakeInput hS i j hij).δ


@[reassoc (attr := simp)]
lemma δ_comp : hS.δ i j hij ≫ HomologicalComplex.homologyMap S.f j = 0 :=
  (snakeInput hS i j hij).δ_L₃_f


@[reassoc (attr := simp)]
lemma comp_δ : HomologicalComplex.homologyMap S.g i ≫ hS.δ i j hij = 0 :=
  (snakeInput hS i j hij).L₀_g_δ


/-- Exactness of `S.X₃.homology i ⟶ S.X₁.homology j ⟶ S.X₂.homology j`. -/
lemma homology_exact₁ : (ShortComplex.mk _ _ (δ_comp hS i j hij)).Exact :=
  (snakeInput hS i j hij).L₂'_exact


include hS in
/-- Exactness of `S.X₁.homology i ⟶ S.X₂.homology i ⟶ S.X₃.homology i`. -/
lemma homology_exact₂ : (ShortComplex.mk (HomologicalComplex.homologyMap S.f i)
    (HomologicalComplex.homologyMap S.g i) (by rw [← HomologicalComplex.homologyMap_comp,
      S.zero, HomologicalComplex.homologyMap_zero])).Exact := by
  /-
    C : Type u_1
    ι : Type u_2
    inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
    inst✝ : CategoryTheory.Abelian C
    c : ComplexShape ι
    S : CategoryTheory.ShortComplex (HomologicalComplex C c)
    hS : S.ShortExact
    i : ι
    ⊢ (CategoryTheory.ShortComplex.mk (HomologicalComplex.homologyMap S.f i) (Homo …
  -/
  by_cases h : c.Rel i (c.next i)
    /-
      case pos
      C : Type u_1
      ι : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Abelian C
      c : ComplexShape ι
      S : CategoryTheory.ShortComplex (HomologicalComplex C c)
      hS : S.ShortExact
      i : ι
      h : c.Rel i (c.next i)
      ⊢ (CategoryTheory.ShortComplex.mk (HomologicalComplex.homologyMap S.f i) (Homo …
    -/
  · exact (snakeInput hS i _ h).L₀_exact
    /-
      🎉 no goals
    -/
    /-
      case neg
      C : Type u_1
      ι : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Abelian C
      c : ComplexShape ι
      S : CategoryTheory.ShortComplex (HomologicalComplex C c)
      hS : S.ShortExact
      i : ι
      h : Not (c.Rel i (c.next i))
      ⊢ (CategoryTheory.ShortComplex.mk (HomologicalComplex.homologyMap S.f i) (Homo …
    -/
  · have := hS.epi_g
    have : ∀ (K : HomologicalComplex C c), IsIso (K.homologyι i) :=
      fun K => ShortComplex.isIso_homologyι (K.sc i) (K.shape _ _ h)
    have e : S.map (HomologicalComplex.homologyFunctor C c i) ≅
        S.map (HomologicalComplex.opcyclesFunctor C c i) :=
      ShortComplex.isoMk (asIso (S.X₁.homologyι i))
        (asIso (S.X₂.homologyι i)) (asIso (S.X₃.homologyι i)) (by aesop_cat) (by aesop_cat)
    /-
      case neg
      C : Type u_1
      ι : Type u_2
      inst✝¹ : CategoryTheory.Category.{u_3, u_1} C
      inst✝ : CategoryTheory.Abelian C
      c : ComplexShape ι
      S : CategoryTheory.ShortComplex (HomologicalComplex C c)
      hS : S.ShortExact
      i : ι
      h : Not (c.Rel i (c.next i))
      this✝ : CategoryTheory.Epi S.g
      this : ∀ (K : HomologicalComplex C c), CategoryTheory.IsIso (K.homologyι i)
      e : CategoryTheory.Iso (S.map (HomologicalComplex.homologyFunctor C c i)) (S.m …
      ⊢ (CategoryTheory.ShortComplex.mk (HomologicalComplex.homologyMap S.f i) (Homo …
    -/
    exact ShortComplex.exact_of_iso e.symm (opcycles_right_exact S hS.exact i)
    /-
      🎉 no goals
    -/


/-- Exactness of `S.X₂.homology i ⟶ S.X₃.homology i ⟶ S.X₁.homology j`. -/
lemma homology_exact₃ : (ShortComplex.mk _ _ (comp_δ hS i j hij)).Exact :=
  (snakeInput hS i j hij).L₁'_exact


lemma δ_eq' {A : C} (x₃ : A ⟶ S.X₃.homology i) (x₂ : A ⟶ S.X₂.opcycles i)
    (x₁ : A ⟶ S.X₁.cycles j)
    (h₂ : x₂ ≫ HomologicalComplex.opcyclesMap S.g i = x₃ ≫ S.X₃.homologyι i)
    (h₁ : x₁ ≫ HomologicalComplex.cyclesMap S.f j = x₂ ≫ S.X₂.opcyclesToCycles i j) :
    x₃ ≫ hS.δ i j hij = x₁ ≫ S.X₁.homologyπ j :=
  (snakeInput hS i j hij).δ_eq x₃ x₂ x₁ h₂ h₁


lemma δ_eq {A : C} (x₃ : A ⟶ S.X₃.X i) (hx₃ : x₃ ≫ S.X₃.d i j = 0)
    (x₂ : A ⟶ S.X₂.X i) (hx₂ : x₂ ≫ S.g.f i = x₃)
    (x₁ : A ⟶ S.X₁.X j) (hx₁ : x₁ ≫ S.f.f j = x₂ ≫ S.X₂.d i j)
    (k : ι) (hk : c.next j = k) :
    S.X₃.liftCycles x₃ j (c.next_eq' hij) hx₃ ≫ S.X₃.homologyπ i ≫ hS.δ i j hij =
      S.X₁.liftCycles x₁ k hk (by
        /-
          C : Type u_1
          ι : Type u_2
          inst✝¹ : CategoryTheory.Category.{?u.139441, u_1} C
          inst✝ : CategoryTheory.Abelian C
          c : ComplexShape ι
          S : CategoryTheory.ShortComplex (HomologicalComplex C c)
          hS : S.ShortExact
          i j : ι
          hij : c.Rel i j
          A : C
          x₃ : Quiver.Hom A (S.X₃.X i)
          hx₃ : Eq (CategoryTheory.CategoryStruct.comp x₃ (S.X₃.d i j)) 0
          x₂ : Quiver.Hom A (S.X₂.X i)
          hx₂ : Eq (CategoryTheory.CategoryStruct.comp x₂ (S.g.f i)) x₃
          x₁ : Quiver.Hom A (S.X₁.X j)
          hx₁ : Eq (CategoryTheory.CategoryStruct.comp x₁ (S.f.f j)) (CategoryTheory.Cat …
          k : ι
          hk : Eq (c.next j) k
          ⊢ Eq (CategoryTheory.CategoryStruct.comp x₁ (S.X₁.d j k)) 0
        -/
        have := hS.mono_f
        rw [← cancel_mono (S.f.f k), assoc, ← S.f.comm, reassoc_of% hx₁,
          d_comp_d, comp_zero, zero_comp]) ≫ S.X₁.homologyπ j := by
  simpa only [assoc] using hS.δ_eq' i j hij (S.X₃.liftCycles x₃ j
    (c.next_eq' hij) hx₃ ≫ S.X₃.homologyπ i)
    (x₂ ≫ S.X₂.pOpcycles i) (S.X₁.liftCycles x₁ k hk _)
      (by simp only [assoc, HomologicalComplex.p_opcyclesMap,
        HomologicalComplex.homology_π_ι,
        HomologicalComplex.liftCycles_i_assoc, reassoc_of% hx₂])
      (by rw [← cancel_mono (S.X₂.iCycles j), HomologicalComplex.liftCycles_comp_cyclesMap,
        HomologicalComplex.liftCycles_i, assoc, assoc, opcyclesToCycles_iCycles,
        HomologicalComplex.p_fromOpcycles, hx₁])


