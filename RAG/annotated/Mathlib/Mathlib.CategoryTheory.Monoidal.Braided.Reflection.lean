/-- The uncurried retraction of the unit in the proof of `4 → 1` in `day_reflection` below. -/
private noncomputable def adjRetractionAux
    (c : C) (d : D) [IsIso (L.map (adj.unit.app ((ihom d).obj (R.obj c)) ⊗ adj.unit.app d))] :
  d ⊗ ((L ⋙ R).obj ((ihom d).obj (R.obj c))) ⟶ (R.obj c) :=
  (β_ _ _).hom ≫ (_ ◁ adj.unit.app _) ≫ adj.unit.app _ ≫
    R.map (inv (L.map (adj.unit.app _ ⊗ adj.unit.app _))) ≫ (L ⋙ R).map (β_ _ _).hom ≫
      (L ⋙ R).map ((ihom.ev _).app _) ≫ inv (adj.unit.app _)


/-- The retraction of the unit in the proof of `4 → 1` in `day_reflection` below. -/
private noncomputable def adjRetraction (c : C) (d : D)
    [IsIso (L.map (adj.unit.app ((ihom d).obj (R.obj c)) ⊗ adj.unit.app d))] :
    (L ⋙ R).obj ((ihom d).obj (R.obj c)) ⟶ ((ihom d).obj (R.obj c)) :=
  curry <| adjRetractionAux adj c d


private lemma adjRetraction_is_retraction (c : C) (d : D)
    [IsIso (L.map (adj.unit.app ((ihom d).obj (R.obj c)) ⊗ adj.unit.app d))] :
    adj.unit.app ((ihom d).obj (R.obj c)) ≫ adjRetraction adj c d = 𝟙 _ := by
  suffices (_ ◁ adj.unit.app _) ≫ adjRetractionAux adj c d = (ihom.ev _).app _ by
    simp only [id_obj, comp_obj, adjRetraction, ← curry_natural_left, this]
    simp [curry_eq]
  simp only [id_obj, comp_obj, adjRetractionAux, Functor.map_inv, Functor.comp_map,
    braiding_naturality_right_assoc]
  slice_lhs 2 3 =>
    simp only [← id_tensorHom, ← tensorHom_id, ← tensor_comp, Category.id_comp, Category.comp_id]
  slice_lhs 2 4 =>
    rw [← adj.unit_naturality_assoc]
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁷ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁶ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁵ : CategoryTheory.MonoidalCategory D
    inst✝⁴ : CategoryTheory.SymmetricCategory D
    inst✝³ : CategoryTheory.MonoidalClosed D
    R : CategoryTheory.Functor C D
    inst✝² : R.Faithful
    inst✝¹ : R.Full
    L : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction L R
    c : C
    d : D
    inst✝ : CategoryTheory.IsIso (L.map (CategoryTheory.MonoidalCategoryStruct.ten …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.BraidedCategory.braid …
  -/
  simp
  /-
    🎉 no goals
  -/


/--
Day's reflection theorem.

Let `D` be a symmetric monoidal closed category and let `C` be a reflective subcategory. Denote by
`R : C ⥤ D` the inclusion functor and by `L : D ⥤ C` the reflector. Let `u` denote the unit of the
adjunction `L ⊣ R`. Denote the internal hom by `[-,-]`. The following are equivalent:

1. `u : [d, Rc] ⟶ RL[d, Rc]` is an isomorphism,
2. `[u, 𝟙] : [RLd, Rc] ⟶ [d, Rc]` is an isomorphism,
3. `L(u ▷ d') : L(d ⊗ d') ⟶ L(RLd ⊗ d')` is an isomorphism,
4. `L(u ⊗ u) : L(d ⊗ d') ⟶ L(RLd ⊗ RLd')` is an isomorphism,

where `c, d, d'` are arbitrary objects of `C`/`D`, quantified over separately in each condition.
-/
theorem isIso_tfae : List.TFAE
    [ ∀ (c : C) (d : D), IsIso (adj.unit.app ((ihom d).obj (R.obj c)))
    , ∀ (c : C) (d : D), IsIso ((pre (adj.unit.app d)).app (R.obj c))
    , ∀ (d d' : D), IsIso (L.map ((adj.unit.app d) ▷ d'))
    , ∀ (d d' : D), IsIso (L.map ((adj.unit.app d) ⊗ (adj.unit.app d')))] := by
  tfae_have 3 → 4
  | h => by
    -- We can commute the tensor product in the condition that `L.map ((adj.unit.app d) ▷ d')` is
    -- an isomorphism:
    have h' : ∀ d d', IsIso (L.map (d ◁ (adj.unit.app d'))) := by
      intro d d'
      have := braiding_naturality (𝟙 d) (adj.unit.app d')
      rw [← Iso.eq_comp_inv, id_tensorHom] at this
      rw [this]
      simp only [map_comp, id_obj, comp_obj, tensorHom_id, Category.assoc]
      infer_instance
    intro d d'
    -- We then write the tensor product of the two units as the composition of the whiskered units,
    -- and conclude.
    have : (adj.unit.app d) ⊗ (adj.unit.app d') =
        (adj.unit.app d ▷ d') ≫ (((L ⋙ R).obj _) ◁ adj.unit.app d') := by
      simp [← tensorHom_id, ← id_tensorHom, ← tensor_comp]
    rw [this, map_comp]
    infer_instance
  tfae_have 4 → 1
  | _, _, _ => by
    -- It is enough to show that the unit is a split monomorphism, and the retraction is given
    -- by `adjRetraction` above.
    let _ : Reflective R := { L := L, adj := adj }
    have : IsIso adj.toMonad.μ := μ_iso_of_reflective (R := R)
    erw [← adj.toMonad.isSplitMono_iff_isIso_unit]
    exact ⟨⟨adjRetraction adj _ _, adjRetraction_is_retraction adj _ _⟩⟩
  tfae_have 1 → 3
  | h, d, d' => by
    rw [isIso_iff_isIso_coyoneda_map]
    intro c
    -- `w₁, w₃, w₄` are the three stacked commutative squares in the proof on nLab:
    have w₁ : (coyoneda.map (L.map (adj.unit.app d ▷ d')).op).app c = (adj.homEquiv _ _).symm ∘
        (coyoneda.map (adj.unit.app d ▷ d').op).app (R.obj c) ∘ adj.homEquiv _ _ := by ext; simp
    rw [w₁, isIso_iff_bijective]
    simp only [comp_obj, coyoneda_obj_obj, id_obj, EquivLike.comp_bijective,
      EquivLike.bijective_comp]
    -- We commute the tensor product using the auxiliary commutative square `w₂`.
    have w₂ : ((coyoneda.map (adj.unit.app d ▷ d').op).app (R.obj c)) =
        ((yoneda.obj (R.obj c)).mapIso (β_ _ _)).hom ∘
          ((coyoneda.map (d' ◁ adj.unit.app d).op).app (R.obj c)) ∘
            ((yoneda.obj (R.obj c)).mapIso (β_ _ _)).hom := by ext; simp
    rw [w₂, ← types_comp, ← types_comp, ← isIso_iff_bijective]
    refine IsIso.comp_isIso' (IsIso.comp_isIso' inferInstance ?_) inferInstance
    have w₃ : ((coyoneda.map (d' ◁ adj.unit.app d).op).app (R.obj c)) =
        ((ihom.adjunction d').homEquiv _ _).symm ∘
          ((coyoneda.map (adj.unit.app _).op).app _) ∘ (ihom.adjunction d').homEquiv _ _ := by
      ext
      simp only [id_obj, op_tensorObj, coyoneda_obj_obj, unop_tensorObj, comp_obj,
        coyoneda_map_app, Quiver.Hom.unop_op, Function.comp_apply,
        Adjunction.homEquiv_unit, Adjunction.homEquiv_counit]
      simp
    rw [w₃, isIso_iff_bijective]
    simp only [comp_obj, op_tensorObj, coyoneda_obj_obj, unop_tensorObj, id_obj,
      yoneda_obj_obj, tensorLeft_obj, EquivLike.comp_bijective, EquivLike.bijective_comp]
    have w₄ : (coyoneda.map (adj.unit.app d).op).app ((ihom d').obj (R.obj c)) ≫
        (coyoneda.obj ⟨d⟩).map (adj.unit.app ((ihom d').obj (R.obj c))) =
          (coyoneda.obj ⟨(L ⋙ R).obj d⟩).map (adj.unit.app ((ihom d').obj (R.obj c))) ≫
            (coyoneda.map (adj.unit.app d).op).app _ := by simp
    rw [← isIso_iff_bijective]
    suffices IsIso ((coyoneda.map (adj.unit.app d).op).app ((ihom d').obj (R.obj c)) ≫
        (coyoneda.obj ⟨d⟩).map (adj.unit.app ((ihom d').obj (R.obj c)))) from
      IsIso.of_isIso_comp_right _ ((coyoneda.obj ⟨d⟩).map (adj.unit.app ((ihom d').obj (R.obj c))))
    rw [w₄]
    refine IsIso.comp_isIso' inferInstance ?_
    constructor
    -- We give the inverse of the bottom map in the stack of commutative squares:
    refine ⟨fun f ↦ R.map ((adj.homEquiv _ _).symm f), ?_, by ext; simp⟩
    ext f
    simp only [comp_obj, coyoneda_obj_obj, id_obj, Adjunction.homEquiv_counit,
      map_comp, types_comp_apply, coyoneda_map_app, Quiver.Hom.unop_op, Category.assoc,
      types_id_apply]
    have : f = R.map (R.preimage f) := by simp
    rw [this]
    simp [← map_comp, ← map_comp_assoc, -map_preimage]
  tfae_have 2 ↔ 3 := by
    conv => lhs; intro c d; rw [isIso_iff_isIso_yoneda_map]
    conv => rhs; intro d d'; rw [isIso_iff_isIso_coyoneda_map]
    -- bring the quantifiers out of the `↔`:
    rw [forall_swap]; apply forall_congr'; intro d
    rw [forall_swap]; apply forall₂_congr; intro d' c
    -- `w₁, w₂,` are the two stacked commutative squares in the proof on nLab:
    have w₁ : ((coyoneda.map (L.map (adj.unit.app d ▷ d')).op).app c) =
        (adj.homEquiv _ _).symm ∘
          (coyoneda.map (adj.unit.app d ▷ d').op).app (R.obj c) ∘
            (adj.homEquiv _ _) := by ext; simp
    have w₂ : ((yoneda.map ((pre (adj.unit.app d)).app (R.obj c))).app ⟨d'⟩) =
          ((ihom.adjunction d).homEquiv _ _) ∘
            ((coyoneda.map (adj.unit.app d ▷ d').op).app (R.obj c)) ∘
              ((ihom.adjunction ((L ⋙ R).obj d)).homEquiv _ _).symm := by
      rw [← Function.comp_assoc, ((ihom.adjunction ((L ⋙ R).obj d)).homEquiv _ _).eq_comp_symm]
      ext
      simp only [id_obj, yoneda_obj_obj, comp_obj, Function.comp_apply,
        yoneda_map_app, op_tensorObj, coyoneda_obj_obj, unop_tensorObj, op_whiskerRight,
        coyoneda_map_app, unop_whiskerRight, Quiver.Hom.unop_op]
      rw [Adjunction.homEquiv_unit, Adjunction.homEquiv_unit]
      simp
    rw [w₂, w₁, isIso_iff_bijective, isIso_iff_bijective]
    simp
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁶ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁵ : CategoryTheory.Category.{u_3, u_2} D
    inst✝⁴ : CategoryTheory.MonoidalCategory D
    inst✝³ : CategoryTheory.SymmetricCategory D
    inst✝² : CategoryTheory.MonoidalClosed D
    R : CategoryTheory.Functor C D
    inst✝¹ : R.Faithful
    inst✝ : R.Full
    L : CategoryTheory.Functor D C
    adj : CategoryTheory.Adjunction L R
    tfae_3_to_4 : (∀ (d d' : D), CategoryTheory.IsIso (L.map (CategoryTheory.Monoi …
    tfae_4_to_1 : (∀ (d d' : D), CategoryTheory.IsIso (L.map (CategoryTheory.Monoi …
    tfae_1_to_3 : (∀ (c : C) (d : D), CategoryTheory.IsIso (adj.unit.app ((Categor …
    tfae_2_iff_3 : Iff (∀ (c : C) (d : D), CategoryTheory.IsIso ((CategoryTheory.M …
    ⊢ (List.cons (∀ (c : C) (d : D), CategoryTheory.IsIso (adj.unit.app ((Category …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


instance (d d' : D) : IsIso (L.map ((adj.unit.app d) ⊗ (adj.unit.app d'))) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁷ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁶ : CategoryTheory.MonoidalCategory D
    inst✝⁵ : CategoryTheory.SymmetricCategory D
    inst✝⁴ : CategoryTheory.MonoidalClosed D
    inst✝³ : CategoryTheory.MonoidalCategory C
    L : CategoryTheory.Functor D C
    inst✝² : L.Monoidal
    R : CategoryTheory.Functor C D
    inst✝¹ : R.Faithful
    inst✝ : R.Full
    adj : CategoryTheory.Adjunction L R
    d d' : D
    ⊢ CategoryTheory.IsIso (L.map (CategoryTheory.MonoidalCategoryStruct.tensorHom …
  -/
  have := δ _ _ _ ≫= μ_natural L (adj.unit.app d) (adj.unit.app d')
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁷ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁶ : CategoryTheory.MonoidalCategory D
    inst✝⁵ : CategoryTheory.SymmetricCategory D
    inst✝⁴ : CategoryTheory.MonoidalClosed D
    inst✝³ : CategoryTheory.MonoidalCategory C
    L : CategoryTheory.Functor D C
    inst✝² : L.Monoidal
    R : CategoryTheory.Functor C D
    inst✝¹ : R.Faithful
    inst✝ : R.Full
    adj : CategoryTheory.Adjunction L R
    d d' : D
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.OplaxMon …
    ⊢ CategoryTheory.IsIso (L.map (CategoryTheory.MonoidalCategoryStruct.tensorHom …
  -/
  rw [δ_μ_assoc] at this
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁷ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁶ : CategoryTheory.MonoidalCategory D
    inst✝⁵ : CategoryTheory.SymmetricCategory D
    inst✝⁴ : CategoryTheory.MonoidalClosed D
    inst✝³ : CategoryTheory.MonoidalCategory C
    L : CategoryTheory.Functor D C
    inst✝² : L.Monoidal
    R : CategoryTheory.Functor C D
    inst✝¹ : R.Faithful
    inst✝ : R.Full
    adj : CategoryTheory.Adjunction L R
    d d' : D
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.OplaxMon …
    ⊢ CategoryTheory.IsIso (L.map (CategoryTheory.MonoidalCategoryStruct.tensorHom …
  -/
  rw [← this]
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁷ : CategoryTheory.Category.{u_4, u_2} D
    inst✝⁶ : CategoryTheory.MonoidalCategory D
    inst✝⁵ : CategoryTheory.SymmetricCategory D
    inst✝⁴ : CategoryTheory.MonoidalClosed D
    inst✝³ : CategoryTheory.MonoidalCategory C
    L : CategoryTheory.Functor D C
    inst✝² : L.Monoidal
    R : CategoryTheory.Functor C D
    inst✝¹ : R.Faithful
    inst✝ : R.Full
    adj : CategoryTheory.Adjunction L R
    d d' : D
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Functor.OplaxMon …
    ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (CategoryTheory.Fun …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance (c : C) (d : D) : IsIso (adj.unit.app ((ihom d).obj (R.obj c))) := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁸ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁷ : CategoryTheory.Category.{u_3, u_2} D
    inst✝⁶ : CategoryTheory.MonoidalCategory D
    inst✝⁵ : CategoryTheory.SymmetricCategory D
    inst✝⁴ : CategoryTheory.MonoidalClosed D
    inst✝³ : CategoryTheory.MonoidalCategory C
    L : CategoryTheory.Functor D C
    inst✝² : L.Monoidal
    R : CategoryTheory.Functor C D
    inst✝¹ : R.Faithful
    inst✝ : R.Full
    adj : CategoryTheory.Adjunction L R
    c : C
    d : D
    ⊢ CategoryTheory.IsIso (adj.unit.app ((CategoryTheory.ihom d).obj (R.obj c)))
  -/
  revert c d
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁸ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁷ : CategoryTheory.Category.{u_3, u_2} D
    inst✝⁶ : CategoryTheory.MonoidalCategory D
    inst✝⁵ : CategoryTheory.SymmetricCategory D
    inst✝⁴ : CategoryTheory.MonoidalClosed D
    inst✝³ : CategoryTheory.MonoidalCategory C
    L : CategoryTheory.Functor D C
    inst✝² : L.Monoidal
    R : CategoryTheory.Functor C D
    inst✝¹ : R.Faithful
    inst✝ : R.Full
    adj : CategoryTheory.Adjunction L R
    ⊢ ∀ (c : C) (d : D), CategoryTheory.IsIso (adj.unit.app ((CategoryTheory.ihom  …
  -/
  rw [((isIso_tfae adj).out 0 3:)]
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁸ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁷ : CategoryTheory.Category.{u_3, u_2} D
    inst✝⁶ : CategoryTheory.MonoidalCategory D
    inst✝⁵ : CategoryTheory.SymmetricCategory D
    inst✝⁴ : CategoryTheory.MonoidalClosed D
    inst✝³ : CategoryTheory.MonoidalCategory C
    L : CategoryTheory.Functor D C
    inst✝² : L.Monoidal
    R : CategoryTheory.Functor C D
    inst✝¹ : R.Faithful
    inst✝ : R.Full
    adj : CategoryTheory.Adjunction L R
    ⊢ ∀ (d d' : D), CategoryTheory.IsIso (L.map (CategoryTheory.MonoidalCategorySt …
  -/
  intro d d'
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁸ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁷ : CategoryTheory.Category.{u_3, u_2} D
    inst✝⁶ : CategoryTheory.MonoidalCategory D
    inst✝⁵ : CategoryTheory.SymmetricCategory D
    inst✝⁴ : CategoryTheory.MonoidalClosed D
    inst✝³ : CategoryTheory.MonoidalCategory C
    L : CategoryTheory.Functor D C
    inst✝² : L.Monoidal
    R : CategoryTheory.Functor C D
    inst✝¹ : R.Faithful
    inst✝ : R.Full
    adj : CategoryTheory.Adjunction L R
    d d' : D
    ⊢ CategoryTheory.IsIso (L.map (CategoryTheory.MonoidalCategoryStruct.tensorHom …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- Auxiliary definition for `monoidalClosed`. -/
noncomputable def closed (c : C) : Closed c where
  rightAdj := R ⋙ (ihom (R.obj c)) ⋙ L
  adj := by
    refine ((ihom.adjunction (R.obj c)).comp adj).restrictFullyFaithful
      (FullyFaithful.ofFullyFaithful R)
      (FullyFaithful.id _) ?_ ?_
    · refine NatIso.ofComponents (fun _ ↦ (μIso L _ _).symm ≪≫
        asIso ((adj.counit.app _) ⊗ (adj.counit.app _))) (fun _ ↦ ?_)
      /-
        case refine_1
        C : Type u_1
        D : Type u_2
        inst✝⁸ : CategoryTheory.Category.{?u.92043, u_1} C
        inst✝⁷ : CategoryTheory.Category.{?u.92047, u_2} D
        inst✝⁶ : CategoryTheory.MonoidalCategory D
        inst✝⁵ : CategoryTheory.SymmetricCategory D
        inst✝⁴ : CategoryTheory.MonoidalClosed D
        inst✝³ : CategoryTheory.MonoidalCategory C
        L : CategoryTheory.Functor D C
        inst✝² : L.Monoidal
        R : CategoryTheory.Functor C D
        inst✝¹ : R.Faithful
        inst✝ : R.Full
        adj : CategoryTheory.Adjunction L R
        c X✝ Y✝ : C
        x✝ : Quiver.Hom X✝ Y✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((R.comp ((CategoryTheory.MonoidalCat …
      -/
      dsimp
      rw [Category.assoc, ← δ_natural_right_assoc,
        tensorHom_def', ← MonoidalCategory.whiskerLeft_comp_assoc,
        Adjunction.counit_naturality, whisker_exchange,
        tensorHom_def_assoc, MonoidalCategory.whiskerLeft_comp]
      /-
        case refine_2
        C : Type u_1
        D : Type u_2
        inst✝⁸ : CategoryTheory.Category.{?u.92043, u_1} C
        inst✝⁷ : CategoryTheory.Category.{?u.92047, u_2} D
        inst✝⁶ : CategoryTheory.MonoidalCategory D
        inst✝⁵ : CategoryTheory.SymmetricCategory D
        inst✝⁴ : CategoryTheory.MonoidalClosed D
        inst✝³ : CategoryTheory.MonoidalCategory C
        L : CategoryTheory.Functor D C
        inst✝² : L.Monoidal
        R : CategoryTheory.Functor C D
        inst✝¹ : R.Faithful
        inst✝ : R.Full
        adj : CategoryTheory.Adjunction L R
        c : C
        ⊢ CategoryTheory.Iso ((CategoryTheory.Functor.id C).comp (R.comp (CategoryTheo …
      -/
    · exact NatIso.ofComponents (fun _ ↦ asIso (adj.unit.app ((ihom _).obj _)))
      /-
        🎉 no goals
      -/


/--
Given a reflective functor `R : C ⥤ D` with a monoidal left adjoint, such that `D` is symmetric
monoidal closed, then `C` is monoidal closed.
-/
noncomputable def monoidalClosed : MonoidalClosed C where
  closed c := closed adj c


