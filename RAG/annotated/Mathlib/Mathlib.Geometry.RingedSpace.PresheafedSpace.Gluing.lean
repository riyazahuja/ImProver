/-- A family of gluing data consists of
1. An index type `J`
2. A presheafed space `U i` for each `i : J`.
3. A presheafed space `V i j` for each `i j : J`.
  (Note that this is `J × J → PresheafedSpace C` rather than `J → J → PresheafedSpace C` to
  connect to the limits library easier.)
4. An open immersion `f i j : V i j ⟶ U i` for each `i j : ι`.
5. A transition map `t i j : V i j ⟶ V j i` for each `i j : ι`.
such that
6. `f i i` is an isomorphism.
7. `t i i` is the identity.
8. `V i j ×[U i] V i k ⟶ V i j ⟶ V j i` factors through `V j k ×[U j] V j i ⟶ V j i` via some
    `t' : V i j ×[U i] V i k ⟶ V j k ×[U j] V j i`.
9. `t' i j k ≫ t' j k i ≫ t' k i j = 𝟙 _`.

We can then glue the spaces `U i` together by identifying `V i j` with `V j i`, such
that the `U i`'s are open subspaces of the glued space.
-/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): this linter isn't ported yet.
-- @[nolint has_nonempty_instance]
structure GlueData extends CategoryTheory.GlueData (PresheafedSpace.{u, v, v} C) where
  f_open : ∀ i j, IsOpenImmersion (f i j)


local notation "𝖣" => D.toGlueData


local notation "π₁ " i ", " j ", " k => pullback.fst (D.f i j) (D.f i k)


local notation "π₂ " i ", " j ", " k => pullback.snd (D.f i j) (D.f i k)


local notation "π₁⁻¹ " i ", " j ", " k =>
  (PresheafedSpace.IsOpenImmersion.pullbackFstOfRight (D.f i j) (D.f i k)).invApp


local notation "π₂⁻¹ " i ", " j ", " k =>
  (PresheafedSpace.IsOpenImmersion.pullbackSndOfLeft (D.f i j) (D.f i k)).invApp


/-- The glue data of topological spaces associated to a family of glue data of PresheafedSpaces. -/
abbrev toTopGlueData : TopCat.GlueData :=
  { f_open := fun i j => (D.f_open i j).base_open
    toGlueData := 𝖣.mapGlueData (forget C) }


theorem ι_isOpenEmbedding [HasLimits C] (i : D.J) : IsOpenEmbedding (𝖣.ι i).base := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.PresheafedSpace.GlueData C
    inst✝ : CategoryTheory.Limits.HasLimits C
    i : D.J
    ⊢ Topology.IsOpenEmbedding ⇑(D.ι i).base
  -/
  rw [← show _ = (𝖣.ι i).base from 𝖣.ι_gluedIso_inv (PresheafedSpace.forget _) _]
  -- Porting note: added this erewrite
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.PresheafedSpace.GlueData C
    inst✝ : CategoryTheory.Limits.HasLimits C
    i : D.J
    ⊢ Topology.IsOpenEmbedding ⇑(CategoryTheory.CategoryStruct.comp ((D.mapGlueDat …
  -/
  erw [coe_comp]
  exact (TopCat.homeoOfIso (𝖣.gluedIso (PresheafedSpace.forget _)).symm).isOpenEmbedding.comp
      (D.toTopGlueData.ι_isOpenEmbedding i)


@[deprecated (since := "2024-10-18")]
alias ι_openEmbedding := ι_isOpenEmbedding


theorem pullback_base (i j k : D.J) (S : Set (D.V (i, j)).carrier) :
    (π₂ i, j, k) '' ((π₁ i, j, k) ⁻¹' S) = D.f i k ⁻¹' (D.f i j '' S) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.PresheafedSpace.GlueData C
    i j k : D.J
    S : Set ↑↑(D.V { fst := i, snd := j })
    ⊢ Eq (Set.image (⇑(CategoryTheory.Limits.pullback.snd (D.f i j) (D.f i k)).bas …
  -/
  have eq₁ : _ = (π₁ i, j, k).base := PreservesPullback.iso_hom_fst (forget C) _ _
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.PresheafedSpace.GlueData C
    i j k : D.J
    S : Set ↑↑(D.V { fst := i, snd := j })
    eq₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PreservesP …
    ⊢ Eq (Set.image (⇑(CategoryTheory.Limits.pullback.snd (D.f i j) (D.f i k)).bas …
  -/
  have eq₂ : _ = (π₂ i, j, k).base := PreservesPullback.iso_hom_snd (forget C) _ _
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.PresheafedSpace.GlueData C
    i j k : D.J
    S : Set ↑↑(D.V { fst := i, snd := j })
    eq₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PreservesP …
    eq₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PreservesP …
    ⊢ Eq (Set.image (⇑(CategoryTheory.Limits.pullback.snd (D.f i j) (D.f i k)).bas …
  -/
  rw [← eq₁, ← eq₂]
  -- Porting note: `rw` to `erw` on `coe_comp`
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.PresheafedSpace.GlueData C
    i j k : D.J
    S : Set ↑↑(D.V { fst := i, snd := j })
    eq₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PreservesP …
    eq₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PreservesP …
    ⊢ Eq (Set.image (⇑(CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.P …
  -/
  erw [coe_comp]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.PresheafedSpace.GlueData C
    i j k : D.J
    S : Set ↑↑(D.V { fst := i, snd := j })
    eq₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PreservesP …
    eq₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PreservesP …
    ⊢ Eq (Set.image (Function.comp ⇑(CategoryTheory.Limits.pullback.snd ((Algebrai …
  -/
  rw [Set.image_comp]
  -- Porting note: `rw` to `erw` on `coe_comp`
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.PresheafedSpace.GlueData C
    i j k : D.J
    S : Set ↑↑(D.V { fst := i, snd := j })
    eq₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PreservesP …
    eq₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PreservesP …
    ⊢ Eq (Set.image (⇑(CategoryTheory.Limits.pullback.snd ((AlgebraicGeometry.Pres …
  -/
  erw [coe_comp]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.PresheafedSpace.GlueData C
    i j k : D.J
    S : Set ↑↑(D.V { fst := i, snd := j })
    eq₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PreservesP …
    eq₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PreservesP …
    ⊢ Eq (Set.image (⇑(CategoryTheory.Limits.pullback.snd ((AlgebraicGeometry.Pres …
  -/
  rw [Set.preimage_comp, Set.image_preimage_eq, TopCat.pullback_snd_image_fst_preimage]
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      i j k : D.J
      S : Set ↑↑(D.V { fst := i, snd := j })
      eq₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PreservesP …
      eq₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PreservesP …
      ⊢ Eq (Set.preimage (⇑((AlgebraicGeometry.PresheafedSpace.forget C).map (D.f i  …
    -/
  · rfl
    /-
      🎉 no goals
    -/
  /-
    case h
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.PresheafedSpace.GlueData C
    i j k : D.J
    S : Set ↑↑(D.V { fst := i, snd := j })
    eq₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PreservesP …
    eq₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PreservesP …
    ⊢ Function.Surjective ⇑(CategoryTheory.Limits.PreservesPullback.iso (Algebraic …
  -/
  rw [← TopCat.epi_iff_surjective]
  /-
    case h
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.PresheafedSpace.GlueData C
    i j k : D.J
    S : Set ↑↑(D.V { fst := i, snd := j })
    eq₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PreservesP …
    eq₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PreservesP …
    ⊢ CategoryTheory.Epi (CategoryTheory.Limits.PreservesPullback.iso (AlgebraicGe …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The red and the blue arrows in ![this diagram](https://i.imgur.com/0GiBUh6.png) commute. -/
@[simp, reassoc]
theorem f_invApp_f_app (i j k : D.J) (U : Opens (D.V (i, j)).carrier) :
    (D.f_open i j).invApp _ U ≫ (D.f i k).c.app _ =
      (π₁ i, j, k).c.app (op U) ≫
        (π₂⁻¹ i, j, k) (unop _) ≫
          (D.V _).presheaf.map
            (eqToHom
              (by
                /-
                  C : Type u
                  inst✝ : CategoryTheory.Category.{v, u} C
                  D : AlgebraicGeometry.PresheafedSpace.GlueData C
                  i j k : D.J
                  U : TopologicalSpace.Opens ↑↑(D.V { fst := i, snd := j })
                  ⊢ Eq { unop := (AlgebraicGeometry.PresheafedSpace.IsOpenImmersion.opensFunctor …
                -/
                delta IsOpenImmersion.opensFunctor
                /-
                  C : Type u
                  inst✝ : CategoryTheory.Category.{v, u} C
                  D : AlgebraicGeometry.PresheafedSpace.GlueData C
                  i j k : D.J
                  U : TopologicalSpace.Opens ↑↑(D.V { fst := i, snd := j })
                  ⊢ Eq { unop := ⋯.functor.obj (Opposite.unop { unop := (TopologicalSpace.Opens. …
                -/
                dsimp only [Functor.op, IsOpenMap.functor, Opens.map, unop_op]
                /-
                  C : Type u
                  inst✝ : CategoryTheory.Category.{v, u} C
                  D : AlgebraicGeometry.PresheafedSpace.GlueData C
                  i j k : D.J
                  U : TopologicalSpace.Opens ↑↑(D.V { fst := i, snd := j })
                  ⊢ Eq { unop := { carrier := Set.image ⇑(CategoryTheory.Limits.pullback.snd (D. …
                -/
                congr
                /-
                  case e_unop.e_carrier
                  C : Type u
                  inst✝ : CategoryTheory.Category.{v, u} C
                  D : AlgebraicGeometry.PresheafedSpace.GlueData C
                  i j k : D.J
                  U : TopologicalSpace.Opens ↑↑(D.V { fst := i, snd := j })
                  ⊢ Eq (Set.image ⇑(CategoryTheory.Limits.pullback.snd (D.f i j) (D.f i k)).base …
                -/
                apply pullback_base)) := by
                /-
                  🎉 no goals
                -/
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.PresheafedSpace.GlueData C
    i j k : D.J
    U : TopologicalSpace.Opens ↑↑(D.V { fst := i, snd := j })
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.PresheafedSpace.Is …
  -/
  have := PresheafedSpace.congr_app (@pullback.condition _ _ _ _ _ (D.f i j) (D.f i k) _)
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.PresheafedSpace.GlueData C
    i j k : D.J
    U : TopologicalSpace.Opens ↑↑(D.V { fst := i, snd := j })
    this : ∀ (U : Opposite (TopologicalSpace.Opens ↑↑(D.U i))), Eq ((CategoryTheor …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.PresheafedSpace.Is …
  -/
  dsimp only [comp_c_app] at this
  rw [← cancel_epi (inv ((D.f_open i j).invApp _ U)), IsIso.inv_hom_id_assoc,
    IsOpenImmersion.inv_invApp]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.PresheafedSpace.GlueData C
    i j k : D.J
    U : TopologicalSpace.Opens ↑↑(D.V { fst := i, snd := j })
    this : ∀ (U : Opposite (TopologicalSpace.Opens ↑↑(D.U i))), Eq (CategoryTheory …
    ⊢ Eq ((D.f i k).c.app { unop := (AlgebraicGeometry.PresheafedSpace.IsOpenImmer …
  -/
  simp_rw [Category.assoc]
  erw [(π₁ i, j, k).c.naturality_assoc, reassoc_of% this, ← Functor.map_comp_assoc,
    IsOpenImmersion.inv_naturality_assoc, IsOpenImmersion.app_invApp_assoc, ←
    (D.V (i, k)).presheaf.map_comp, ← (D.V (i, k)).presheaf.map_comp]
  -- Porting note: need to provide an explicit argument, otherwise Lean does not know which
  -- category we are talking about
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.PresheafedSpace.GlueData C
    i j k : D.J
    U : TopologicalSpace.Opens ↑↑(D.V { fst := i, snd := j })
    this : ∀ (U : Opposite (TopologicalSpace.Opens ↑↑(D.U i))), Eq (CategoryTheory …
    ⊢ Eq ((D.f i k).c.app { unop := (AlgebraicGeometry.PresheafedSpace.IsOpenImmer …
  -/
  convert (Category.comp_id ((f D.toGlueData i k).c.app _)).symm
  /-
    case h.e'_3.h.h.e'_7
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.PresheafedSpace.GlueData C
    i j k : D.J
    U : TopologicalSpace.Opens ↑↑(D.V { fst := i, snd := j })
    this : ∀ (U : Opposite (TopologicalSpace.Opens ↑↑(D.U i))), Eq (CategoryTheory …
    ⊢ Eq ((D.V { fst := i, snd := k }).presheaf.map (CategoryTheory.CategoryStruct …
  -/
  erw [(D.V (i, k)).presheaf.map_id]
  /-
    case h.e'_3.h.h.e'_7
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.PresheafedSpace.GlueData C
    i j k : D.J
    U : TopologicalSpace.Opens ↑↑(D.V { fst := i, snd := j })
    this : ∀ (U : Opposite (TopologicalSpace.Opens ↑↑(D.U i))), Eq (CategoryTheory …
    ⊢ Eq (CategoryTheory.CategoryStruct.id ((D.V { fst := i, snd := k }).presheaf. …
  -/
  rfl
  /-
    🎉 no goals
  -/


set_option backward.isDefEq.lazyWhnfCore false in -- See https://github.com/leanprover-community/mathlib4/issues/12534
/-- We can prove the `eq` along with the lemma. Thus this is bundled together here, and the
lemma itself is separated below.
-/
theorem snd_invApp_t_app' (i j k : D.J) (U : Opens (pullback (D.f i j) (D.f i k)).carrier) :
    ∃ eq,
      (π₂⁻¹ i, j, k) U ≫ (D.t k i).c.app _ ≫ (D.V (k, i)).presheaf.map (eqToHom eq) =
        (D.t' k i j).c.app _ ≫ (π₁⁻¹ k, j, i) (unop _) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.PresheafedSpace.GlueData C
    i j k : D.J
    U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i  …
    ⊢ Exists fun eq => Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.P …
  -/
  fconstructor
  -- Porting note: I don't know what the magic was in Lean3 proof, it just skipped the proof of `eq`
    /-
      case w
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      i j k : D.J
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i  …
      ⊢ Eq ((TopologicalSpace.Opens.map (D.t k i).base).op.obj { unop := (AlgebraicG …
    -/
  · delta IsOpenImmersion.opensFunctor
    /-
      case w
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      i j k : D.J
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i  …
      ⊢ Eq ((TopologicalSpace.Opens.map (D.t k i).base).op.obj { unop := ⋯.functor.o …
    -/
    dsimp only [Functor.op, Opens.map, IsOpenMap.functor, unop_op, Opens.coe_mk]
    /-
      case w
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      i j k : D.J
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i  …
      ⊢ Eq { unop := { carrier := Set.preimage (⇑(D.t k i).base) (Set.image ⇑(Catego …
    -/
    congr
    /-
      case w.e_unop.e_carrier
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      i j k : D.J
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i  …
      ⊢ Eq (Set.preimage (⇑(D.t k i).base) (Set.image ⇑(CategoryTheory.Limits.pullba …
    -/
    have := (𝖣.t_fac k i j).symm
    /-
      case w.e_unop.e_carrier
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      i j k : D.J
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i  …
      this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback. …
      ⊢ Eq (Set.preimage (⇑(D.t k i).base) (Set.image ⇑(CategoryTheory.Limits.pullba …
    -/
    rw [← IsIso.inv_comp_eq] at this
    /-
      case w.e_unop.e_carrier
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      i j k : D.J
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i  …
      this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.inv (D.t' k i j) …
      ⊢ Eq (Set.preimage (⇑(D.t k i).base) (Set.image ⇑(CategoryTheory.Limits.pullba …
    -/
    replace this := (congr_arg ((PresheafedSpace.Hom.base ·)) this).symm
    /-
      case w.e_unop.e_carrier
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      i j k : D.J
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i  …
      this : Eq (CategoryTheory.Limits.pullback.snd (D.f i j) (D.f i k)).base (Categ …
      ⊢ Eq (Set.preimage (⇑(D.t k i).base) (Set.image ⇑(CategoryTheory.Limits.pullba …
    -/
    replace this := congr_arg (ContinuousMap.toFun ·) this
    /-
      case w.e_unop.e_carrier
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      i j k : D.J
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i  …
      this : Eq ((fun x => x.toFun) (CategoryTheory.Limits.pullback.snd (D.f i j) (D …
      ⊢ Eq (Set.preimage (⇑(D.t k i).base) (Set.image ⇑(CategoryTheory.Limits.pullba …
    -/
    dsimp at this
    /-
      case w.e_unop.e_carrier
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      i j k : D.J
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i  …
      this : Eq ⇑(CategoryTheory.Limits.pullback.snd (D.f i j) (D.f i k)).base ⇑(Cat …
      ⊢ Eq (Set.preimage (⇑(D.t k i).base) (Set.image ⇑(CategoryTheory.Limits.pullba …
    -/
    rw [coe_comp, coe_comp] at this
    /-
      case w.e_unop.e_carrier
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      i j k : D.J
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i  …
      this : Eq (⇑(CategoryTheory.Limits.pullback.snd (D.f i j) (D.f i k)).base) (Fu …
      ⊢ Eq (Set.preimage (⇑(D.t k i).base) (Set.image ⇑(CategoryTheory.Limits.pullba …
    -/
    rw [this, Set.image_comp, Set.image_comp, Set.preimage_image_eq]
    /-
      case w.e_unop.e_carrier
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      i j k : D.J
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i  …
      this : Eq (⇑(CategoryTheory.Limits.pullback.snd (D.f i j) (D.f i k)).base) (Fu …
      ⊢ Eq (Set.image (⇑(CategoryTheory.Limits.pullback.fst (D.f k i) (D.f k j)).bas …
    -/
    swap
      /-
        case w.e_unop.e_carrier.h
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        D : AlgebraicGeometry.PresheafedSpace.GlueData C
        i j k : D.J
        U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i  …
        this : Eq (⇑(CategoryTheory.Limits.pullback.snd (D.f i j) (D.f i k)).base) (Fu …
        ⊢ Function.Injective ⇑(D.t k i).base
      -/
    · refine Function.HasLeftInverse.injective ⟨(D.t i k).base, fun x => ?_⟩
      /-
        case w.e_unop.e_carrier.h
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        D : AlgebraicGeometry.PresheafedSpace.GlueData C
        i j k : D.J
        U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i  …
        this : Eq (⇑(CategoryTheory.Limits.pullback.snd (D.f i j) (D.f i k)).base) (Fu …
        x : ↑↑(D.V { fst := k, snd := i })
        ⊢ Eq ((D.t i k).base ((D.t k i).base x)) x
      -/
      rw [← comp_apply, ← comp_base, D.t_inv, id_base, id_apply]
      /-
        🎉 no goals
      -/
    /-
      case w.e_unop.e_carrier
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      i j k : D.J
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i  …
      this : Eq (⇑(CategoryTheory.Limits.pullback.snd (D.f i j) (D.f i k)).base) (Fu …
      ⊢ Eq (Set.image (⇑(CategoryTheory.Limits.pullback.fst (D.f k i) (D.f k j)).bas …
    -/
    refine congr_arg (_ '' ·) ?_
    /-
      case w.e_unop.e_carrier
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      i j k : D.J
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i  …
      this : Eq (⇑(CategoryTheory.Limits.pullback.snd (D.f i j) (D.f i k)).base) (Fu …
      ⊢ Eq (Set.image ⇑(CategoryTheory.inv (D.t' k i j)).base ↑U) (Set.preimage ⇑(D. …
    -/
    refine congr_fun ?_ _
    /-
      case w.e_unop.e_carrier
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      i j k : D.J
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i  …
      this : Eq (⇑(CategoryTheory.Limits.pullback.snd (D.f i j) (D.f i k)).base) (Fu …
      ⊢ Eq (Set.image ⇑(CategoryTheory.inv (D.t' k i j)).base) (Set.preimage ⇑(D.t'  …
    -/
    refine Set.image_eq_preimage_of_inverse ?_ ?_
      /-
        case w.e_unop.e_carrier.refine_1
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        D : AlgebraicGeometry.PresheafedSpace.GlueData C
        i j k : D.J
        U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i  …
        this : Eq (⇑(CategoryTheory.Limits.pullback.snd (D.f i j) (D.f i k)).base) (Fu …
        ⊢ Function.LeftInverse ⇑(D.t' k i j).base ⇑(CategoryTheory.inv (D.t' k i j)).b …
      -/
    · intro x
      /-
        case w.e_unop.e_carrier.refine_1
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        D : AlgebraicGeometry.PresheafedSpace.GlueData C
        i j k : D.J
        U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i  …
        this : Eq (⇑(CategoryTheory.Limits.pullback.snd (D.f i j) (D.f i k)).base) (Fu …
        x : ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i k))
        ⊢ Eq ((D.t' k i j).base ((CategoryTheory.inv (D.t' k i j)).base x)) x
      -/
      rw [← comp_apply, ← comp_base, IsIso.inv_hom_id, id_base, id_apply]
      /-
        🎉 no goals
      -/
      /-
        case w.e_unop.e_carrier.refine_2
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        D : AlgebraicGeometry.PresheafedSpace.GlueData C
        i j k : D.J
        U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i  …
        this : Eq (⇑(CategoryTheory.Limits.pullback.snd (D.f i j) (D.f i k)).base) (Fu …
        ⊢ Function.RightInverse ⇑(D.t' k i j).base ⇑(CategoryTheory.inv (D.t' k i j)). …
      -/
    · intro x
      /-
        case w.e_unop.e_carrier.refine_2
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        D : AlgebraicGeometry.PresheafedSpace.GlueData C
        i j k : D.J
        U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i  …
        this : Eq (⇑(CategoryTheory.Limits.pullback.snd (D.f i j) (D.f i k)).base) (Fu …
        x : (CategoryTheory.forget TopCat).obj ↑(CategoryTheory.Limits.pullback (D.f k …
        ⊢ Eq ((CategoryTheory.inv (D.t' k i j)).base ((D.t' k i j).base x)) x
      -/
      rw [← comp_apply, ← comp_base, IsIso.hom_inv_id, id_base, id_apply]
      /-
        🎉 no goals
      -/
  · rw [← IsIso.eq_inv_comp, IsOpenImmersion.inv_invApp, Category.assoc,
      (D.t' k i j).c.naturality_assoc]
    /-
      case h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      i j k : D.J
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i  …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((D.t k i).c.app { unop := (Algebraic …
    -/
    simp_rw [← Category.assoc]
    /-
      case h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      i j k : D.J
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i  …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((D.t k i).c.app { unop := (Algebraic …
    -/
    erw [← comp_c_app]
    /-
      case h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      i j k : D.J
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i  …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((D.t k i).c.app { unop := (Algebraic …
    -/
    rw [congr_app (D.t_fac k i j), comp_c_app]
    /-
      case h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      i j k : D.J
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i  …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((D.t k i).c.app { unop := (Algebraic …
    -/
    simp_rw [Category.assoc]
    erw [IsOpenImmersion.inv_naturality, IsOpenImmersion.inv_naturality_assoc,
      IsOpenImmersion.app_inv_app'_assoc]
      /-
        case h
        C : Type u
        inst✝ : CategoryTheory.Category.{v, u} C
        D : AlgebraicGeometry.PresheafedSpace.GlueData C
        i j k : D.J
        U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i  …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((D.t k i).c.app { unop := (Algebraic …
      -/
    · simp_rw [← (𝖣.V (k, i)).presheaf.map_comp]; rfl
                                                  /-
                                                    🎉 no goals
                                                  -/
    /-
      case h.hU
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      i j k : D.J
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i  …
      ⊢ HasSubset.Subset (↑((TopologicalSpace.Opens.map (D.t k i).base).obj ((Algebr …
    -/
    rintro x ⟨y, -, eq⟩
    /-
      case h.hU.intro.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      i j k : D.J
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i  …
      x : ↑↑(D.V { fst := k, snd := i })
      y : ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i k))
      eq : Eq ((CategoryTheory.Limits.pullback.snd (D.f i j) (D.f i k)).base y) ((D. …
      ⊢ Membership.mem (Set.range ⇑(CategoryTheory.Limits.pullback.fst (D.f k i) (D. …
    -/
    replace eq := ConcreteCategory.congr_arg (𝖣.t i k).base eq
    /-
      case h.hU.intro.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      i j k : D.J
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i  …
      x : ↑↑(D.V { fst := k, snd := i })
      y : ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i k))
      eq : Eq ((D.t i k).base ((CategoryTheory.Limits.pullback.snd (D.f i j) (D.f i  …
      ⊢ Membership.mem (Set.range ⇑(CategoryTheory.Limits.pullback.fst (D.f k i) (D. …
    -/
    change ((π₂ i, j, k) ≫ D.t i k).base y = (D.t k i ≫ D.t i k).base x at eq
    /-
      case h.hU.intro.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      i j k : D.J
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i  …
      x : ↑↑(D.V { fst := k, snd := i })
      y : ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i k))
      eq : Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.s …
      ⊢ Membership.mem (Set.range ⇑(CategoryTheory.Limits.pullback.fst (D.f k i) (D. …
    -/
    rw [𝖣.t_inv, id_base, TopCat.id_app] at eq
    /-
      case h.hU.intro.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      i j k : D.J
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i  …
      x : ↑↑(D.V { fst := k, snd := i })
      y : ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i k))
      eq : Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.s …
      ⊢ Membership.mem (Set.range ⇑(CategoryTheory.Limits.pullback.fst (D.f k i) (D. …
    -/
    subst eq
    /-
      case h.hU.intro.intro
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      i j k : D.J
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i  …
      y : ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i k))
      ⊢ Membership.mem (Set.range ⇑(CategoryTheory.Limits.pullback.fst (D.f k i) (D. …
    -/
    use (inv (D.t' k i j)).base y
    /-
      case h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      i j k : D.J
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i  …
      y : ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i k))
      ⊢ Eq ((CategoryTheory.Limits.pullback.fst (D.f k i) (D.f k j)).base ((Category …
    -/
    change (inv (D.t' k i j) ≫ π₁ k, i, j).base y = _
    /-
      case h
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      i j k : D.J
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i  …
      y : ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i k))
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.inv (D.t' k i j)) (C …
    -/
    congr 2
    /-
      case h.e_a.e_self
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      i j k : D.J
      U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i  …
      y : ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i k))
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.inv (D.t' k i j)) (Ca …
    -/
    rw [IsIso.inv_comp_eq, 𝖣.t_fac_assoc, 𝖣.t_inv, Category.comp_id]
    /-
      🎉 no goals
    -/


set_option backward.isDefEq.lazyWhnfCore false in -- See https://github.com/leanprover-community/mathlib4/issues/12534
/-- The red and the blue arrows in ![this diagram](https://i.imgur.com/q6X1GJ9.png) commute. -/
@[simp, reassoc]
theorem snd_invApp_t_app (i j k : D.J) (U : Opens (pullback (D.f i j) (D.f i k)).carrier) :
    (π₂⁻¹ i, j, k) U ≫ (D.t k i).c.app _ =
      (D.t' k i j).c.app _ ≫
        (π₁⁻¹ k, j, i) (unop _) ≫
          (D.V (k, i)).presheaf.map (eqToHom (D.snd_invApp_t_app' i j k U).choose.symm) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.PresheafedSpace.GlueData C
    i j k : D.J
    U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i  …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.PresheafedSpace.Is …
  -/
  have e := (D.snd_invApp_t_app' i j k U).choose_spec
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.PresheafedSpace.GlueData C
    i j k : D.J
    U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i  …
    e : Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.PresheafedSpace. …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.PresheafedSpace.Is …
  -/
  replace e := reassoc_of% e
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.PresheafedSpace.GlueData C
    i j k : D.J
    U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i  …
    e : ∀ {Z : C} (h : Quiver.Hom ((D.V { fst := k, snd := i }).presheaf.obj { uno …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.PresheafedSpace.Is …
  -/
  rw [← e]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.PresheafedSpace.GlueData C
    i j k : D.J
    U : TopologicalSpace.Opens ↑↑(CategoryTheory.Limits.pullback (D.f i j) (D.f i  …
    e : ∀ {Z : C} (h : Quiver.Hom ((D.V { fst := k, snd := i }).presheaf.obj { uno …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.PresheafedSpace.Is …
  -/
  simp [eqToHom_map]
  /-
    🎉 no goals
  -/


theorem ι_image_preimage_eq (i j : D.J) (U : Opens (D.U i).carrier) :
    (Opens.map (𝖣.ι j).base).obj ((D.ι_isOpenEmbedding i).isOpenMap.functor.obj U) =
      (opensFunctor (D.f j i)).obj
        ((Opens.map (𝖣.t j i).base).obj ((Opens.map (𝖣.f i j).base).obj U)) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.PresheafedSpace.GlueData C
    inst✝ : CategoryTheory.Limits.HasLimits C
    i j : D.J
    U : TopologicalSpace.Opens ↑↑(D.U i)
    ⊢ Eq ((TopologicalSpace.Opens.map (D.ι j).base).obj (⋯.functor.obj U)) ((Algeb …
  -/
  ext1
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.PresheafedSpace.GlueData C
    inst✝ : CategoryTheory.Limits.HasLimits C
    i j : D.J
    U : TopologicalSpace.Opens ↑↑(D.U i)
    ⊢ Eq ↑((TopologicalSpace.Opens.map (D.ι j).base).obj (⋯.functor.obj U)) ↑((Alg …
  -/
  dsimp only [Opens.map_coe, IsOpenMap.coe_functor_obj]
  rw [← show _ = (𝖣.ι i).base from 𝖣.ι_gluedIso_inv (PresheafedSpace.forget _) i, ←
    show _ = (𝖣.ι j).base from 𝖣.ι_gluedIso_inv (PresheafedSpace.forget _) j]
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11224): change `rw` to `erw` on `coe_comp`
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.PresheafedSpace.GlueData C
    inst✝ : CategoryTheory.Limits.HasLimits C
    i j : D.J
    U : TopologicalSpace.Opens ↑↑(D.U i)
    ⊢ Eq (Set.preimage (⇑(CategoryTheory.CategoryStruct.comp ((D.mapGlueData (Alge …
  -/
  erw [coe_comp, coe_comp, coe_comp]
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.PresheafedSpace.GlueData C
    inst✝ : CategoryTheory.Limits.HasLimits C
    i j : D.J
    U : TopologicalSpace.Opens ↑↑(D.U i)
    ⊢ Eq (Set.preimage (Function.comp (Function.comp ⇑(CategoryTheory.preservesCol …
  -/
  rw [Set.image_comp, Set.preimage_comp]
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.PresheafedSpace.GlueData C
    inst✝ : CategoryTheory.Limits.HasLimits C
    i j : D.J
    U : TopologicalSpace.Opens ↑↑(D.U i)
    ⊢ Eq (Set.preimage (⇑((D.mapGlueData (AlgebraicGeometry.PresheafedSpace.forget …
  -/
  erw [Set.preimage_image_eq]
    /-
      case h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i j : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      ⊢ Eq (Set.preimage (⇑((D.mapGlueData (AlgebraicGeometry.PresheafedSpace.forget …
    -/
  · refine Eq.trans (D.toTopGlueData.preimage_image_eq_image' _ _ _) ?_
    /-
      case h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i j : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      ⊢ Eq (Set.image (⇑(CategoryTheory.CategoryStruct.comp (D.toTopGlueData.t i j)  …
    -/
    dsimp
    /-
      case h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i j : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      ⊢ Eq (Set.image (Function.comp ⇑(D.f j i).base ⇑(D.t i j).base) (Set.preimage  …
    -/
    rw [Set.image_comp]
    /-
      case h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i j : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      ⊢ Eq (Set.image (⇑(D.f j i).base) (Set.image (⇑(D.t i j).base) (Set.preimage ⇑ …
    -/
    refine congr_arg (_ '' ·) ?_
    /-
      case h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i j : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      ⊢ Eq (Set.image (⇑(D.t i j).base) (Set.preimage ⇑(D.f i j).base ↑U)) (Set.prei …
    -/
    rw [Set.eq_preimage_iff_image_eq, ← Set.image_comp]
    /-
      case h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i j : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      ⊢ Eq (Set.image (Function.comp ⇑(D.t j i).base ⇑(D.t i j).base) (Set.preimage  …
    -/
    swap
      /-
        case h.hf
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : AlgebraicGeometry.PresheafedSpace.GlueData C
        inst✝ : CategoryTheory.Limits.HasLimits C
        i j : D.J
        U : TopologicalSpace.Opens ↑↑(D.U i)
        ⊢ Function.Bijective ⇑(D.t j i).base
      -/
    · exact CategoryTheory.ConcreteCategory.bijective_of_isIso (C := TopCat) _
      /-
        🎉 no goals
      -/
    /-
      case h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i j : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      ⊢ Eq (Set.image (Function.comp ⇑(D.t j i).base ⇑(D.t i j).base) (Set.preimage  …
    -/
    change (D.t i j ≫ D.t j i).base '' _ = _
    /-
      case h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i j : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      ⊢ Eq (Set.image (⇑(CategoryTheory.CategoryStruct.comp (D.t i j) (D.t j i)).bas …
    -/
    rw [𝖣.t_inv]
    /-
      case h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i j : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      ⊢ Eq (Set.image (⇑(CategoryTheory.CategoryStruct.id (D.V { fst := i, snd := j  …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case h.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i j : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      ⊢ Function.Injective (Function.comp ⇑(CategoryTheory.preservesColimitIso (Alge …
    -/
  · rw [← coe_comp, ← TopCat.mono_iff_injective]
    /-
      case h.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i j : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      ⊢ CategoryTheory.Mono (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limi …
    -/
    infer_instance
    /-
      🎉 no goals
    -/


/-- (Implementation). The map `Γ(𝒪_{U_i}, U) ⟶ Γ(𝒪_{U_j}, 𝖣.ι j ⁻¹' (𝖣.ι i '' U))` -/
def opensImagePreimageMap (i j : D.J) (U : Opens (D.U i).carrier) :
    (D.U i).presheaf.obj (op U) ⟶
    (D.U j).presheaf.obj (op <|
      (Opens.map (𝖣.ι j).base).obj ((D.ι_isOpenEmbedding i).isOpenMap.functor.obj U)) :=
  (D.f i j).c.app (op U) ≫
    (D.t j i).c.app _ ≫
      (D.f_open j i).invApp _ (unop _) ≫
        (𝖣.U j).presheaf.map (eqToHom (D.ι_image_preimage_eq i j U)).op


theorem opensImagePreimageMap_app' (i j k : D.J) (U : Opens (D.U i).carrier) :
    ∃ eq,
      D.opensImagePreimageMap i j U ≫ (D.f j k).c.app _ =
        ((π₁ j, i, k) ≫ D.t j i ≫ D.f i j).c.app (op U) ≫
          (π₂⁻¹ j, i, k) (unop _) ≫ (D.V (j, k)).presheaf.map (eqToHom eq) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.PresheafedSpace.GlueData C
    inst✝ : CategoryTheory.Limits.HasLimits C
    i j k : D.J
    U : TopologicalSpace.Opens ↑↑(D.U i)
    ⊢ Exists fun eq => Eq (CategoryTheory.CategoryStruct.comp (D.opensImagePreimag …
  -/
  constructor
    /-
      case h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i j k : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.opensImagePreimageMap i j U) ((D.f …
    -/
  · delta opensImagePreimageMap
    /-
      case h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i j k : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp_rw [Category.assoc]
    /-
      case h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i j k : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((D.f i j).c.app { unop := U }) (Cate …
    -/
    rw [(D.f j k).c.naturality, f_invApp_f_app_assoc]
      /-
        case h
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : AlgebraicGeometry.PresheafedSpace.GlueData C
        inst✝ : CategoryTheory.Limits.HasLimits C
        i j k : D.J
        U : TopologicalSpace.Opens ↑↑(D.U i)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((D.f i j).c.app { unop := U }) (Cate …
      -/
    · erw [← (D.V (j, k)).presheaf.map_comp]
        /-
          case h
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : AlgebraicGeometry.PresheafedSpace.GlueData C
          inst✝ : CategoryTheory.Limits.HasLimits C
          i j k : D.J
          U : TopologicalSpace.Opens ↑↑(D.U i)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((D.f i j).c.app { unop := U }) (Cate …
        -/
      · simp_rw [← Category.assoc]
        /-
          case h
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          D : AlgebraicGeometry.PresheafedSpace.GlueData C
          inst✝ : CategoryTheory.Limits.HasLimits C
          i j k : D.J
          U : TopologicalSpace.Opens ↑↑(D.U i)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        erw [← comp_c_app, ← comp_c_app]
          /-
            case h
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            D : AlgebraicGeometry.PresheafedSpace.GlueData C
            inst✝ : CategoryTheory.Limits.HasLimits C
            i j k : D.J
            U : TopologicalSpace.Opens ↑↑(D.U i)
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
          -/
        · simp_rw [Category.assoc]
          /-
            case h
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            D : AlgebraicGeometry.PresheafedSpace.GlueData C
            inst✝ : CategoryTheory.Limits.HasLimits C
            i j k : D.J
            U : TopologicalSpace.Opens ↑↑(D.U i)
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.CategoryStruct.comp  …
          -/
          dsimp only [Functor.op, unop_op, Quiver.Hom.unop_op]
          /-
            case h
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            D : AlgebraicGeometry.PresheafedSpace.GlueData C
            inst✝ : CategoryTheory.Limits.HasLimits C
            i j k : D.J
            U : TopologicalSpace.Opens ↑↑(D.U i)
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.CategoryStruct.comp  …
          -/
          rw [eqToHom_map (Opens.map _), eqToHom_op, eqToHom_trans]
          /-
            case h
            C : Type u
            inst✝¹ : CategoryTheory.Category.{v, u} C
            D : AlgebraicGeometry.PresheafedSpace.GlueData C
            inst✝ : CategoryTheory.Limits.HasLimits C
            i j k : D.J
            U : TopologicalSpace.Opens ↑↑(D.U i)
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.CategoryStruct.comp  …
          -/
          congr
          /-
            🎉 no goals
          -/


/-- The red and the blue arrows in ![this diagram](https://i.imgur.com/mBzV1Rx.png) commute. -/
theorem opensImagePreimageMap_app (i j k : D.J) (U : Opens (D.U i).carrier) :
    D.opensImagePreimageMap i j U ≫ (D.f j k).c.app _ =
      ((π₁ j, i, k) ≫ D.t j i ≫ D.f i j).c.app (op U) ≫
        (π₂⁻¹ j, i, k) (unop _) ≫
          (D.V (j, k)).presheaf.map (eqToHom (opensImagePreimageMap_app' D i j k U).choose) :=
  (opensImagePreimageMap_app' D i j k U).choose_spec

-- This is proved separately since `reassoc` somehow timeouts.

theorem opensImagePreimageMap_app_assoc (i j k : D.J) (U : Opens (D.U i).carrier) {X' : C}
    (f' : _ ⟶ X') :
    D.opensImagePreimageMap i j U ≫ (D.f j k).c.app _ ≫ f' =
      ((π₁ j, i, k) ≫ D.t j i ≫ D.f i j).c.app (op U) ≫
        (π₂⁻¹ j, i, k) (unop _) ≫
          (D.V (j, k)).presheaf.map
            (eqToHom (opensImagePreimageMap_app' D i j k U).choose) ≫ f' := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.PresheafedSpace.GlueData C
    inst✝ : CategoryTheory.Limits.HasLimits C
    i j k : D.J
    U : TopologicalSpace.Opens ↑↑(D.U i)
    X' : C
    f' : Quiver.Hom (((TopCat.Presheaf.pushforward C (D.f j k).base).obj (D.V { fs …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.opensImagePreimageMap i j U) (Cate …
  -/
  simpa only [Category.assoc] using congr_arg (· ≫ f') (opensImagePreimageMap_app D i j k U)
  /-
    🎉 no goals
  -/


/-- (Implementation) Given an open subset of one of the spaces `U ⊆ Uᵢ`, the sheaf component of
the image `ι '' U` in the glued space is the limit of this diagram. -/
abbrev diagramOverOpen {i : D.J} (U : Opens (D.U i).carrier) :
    -- Porting note : ↓ these need to be explicit
    (WalkingMultispan D.diagram.fstFrom D.diagram.sndFrom)ᵒᵖ ⥤ C :=
  componentwiseDiagram 𝖣.diagram.multispan ((D.ι_isOpenEmbedding i).isOpenMap.functor.obj U)


/-- (Implementation)
The projection from the limit of `diagram_over_open` to a component of `D.U j`. -/
abbrev diagramOverOpenπ {i : D.J} (U : Opens (D.U i).carrier) (j : D.J) :=
  limit.π (D.diagramOverOpen U) (op (WalkingMultispan.right j))


/-- (Implementation) We construct the map `Γ(𝒪_{U_i}, U) ⟶ Γ(𝒪_V, U_V)` for each `V` in the gluing
diagram. We will lift these maps into `ιInvApp`. -/
def ιInvAppπApp {i : D.J} (U : Opens (D.U i).carrier) (j) :
    (𝖣.U i).presheaf.obj (op U) ⟶ (D.diagramOverOpen U).obj (op j) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.PresheafedSpace.GlueData C
    inst✝ : CategoryTheory.Limits.HasLimits C
    i : D.J
    U : TopologicalSpace.Opens ↑↑(D.U i)
    j : CategoryTheory.Limits.WalkingMultispan D.diagram.fstFrom D.diagram.sndFrom
    ⊢ Quiver.Hom ((D.U i).presheaf.obj { unop := U }) ((D.diagramOverOpen U).obj { …
  -/
  rcases j with (⟨j, k⟩ | j)
  · refine
      D.opensImagePreimageMap i j U ≫ (D.f j k).c.app _ ≫ (D.V (j, k)).presheaf.map (eqToHom ?_)
    /-
      case left.mk
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      j k : D.J
      ⊢ Eq ((TopologicalSpace.Opens.map (D.f j k).base).op.obj { unop := (Topologica …
    -/
    rw [Functor.op_obj]
    /-
      case left.mk
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      j k : D.J
      ⊢ Eq { unop := (TopologicalSpace.Opens.map (D.f j k).base).obj (Opposite.unop  …
    -/
    congr 1; ext1
    /-
      case left.mk.e_unop.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      j k : D.J
      ⊢ Eq ↑((TopologicalSpace.Opens.map (D.f j k).base).obj (Opposite.unop { unop : …
    -/
    dsimp only [Functor.op_obj, Opens.map_coe, unop_op, IsOpenMap.coe_functor_obj]
    /-
      case left.mk.e_unop.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      j k : D.J
      ⊢ Eq (Set.preimage (⇑(D.f j k).base) (Set.preimage (⇑(D.ι j).base) (Set.image  …
    -/
    rw [Set.preimage_preimage]
    /-
      case left.mk.e_unop.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      j k : D.J
      ⊢ Eq (Set.preimage (fun x => (D.ι j).base ((D.f j k).base x)) (Set.image ⇑(D.ι …
    -/
    change (D.f j k ≫ 𝖣.ι j).base ⁻¹' _ = _
    -- Porting note: used to be `congr 3`
    suffices D.f j k ≫ D.ι j = colimit.ι D.diagram.multispan (WalkingMultispan.left (j, k)) by
      rw [this]
      rfl
    /-
      case left.mk.e_unop.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      j k : D.J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.f j k) (D.ι j)) (CategoryTheory.Li …
    -/
    exact colimit.w 𝖣.diagram.multispan (WalkingMultispan.Hom.fst (j, k))
    /-
      🎉 no goals
    -/
    /-
      case right
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      j : D.diagram.R
      ⊢ Quiver.Hom ((D.U i).presheaf.obj { unop := U }) ((D.diagramOverOpen U).obj { …
    -/
  · exact D.opensImagePreimageMap i j U
    /-
      🎉 no goals
    -/

-- Porting note: time out started in `erw [... congr_app (pullbackSymmetry_hom_comp_snd _ _)]` and
-- the last congr has a very difficult `rfl : eqToHom _ ≫ eqToHom _ ≫ ... = eqToHom ... `

set_option maxHeartbeats 600000 in
/-- (Implementation) The natural map `Γ(𝒪_{U_i}, U) ⟶ Γ(𝒪_X, 𝖣.ι i '' U)`.
This forms the inverse of `(𝖣.ι i).c.app (op U)`. -/
def ιInvApp {i : D.J} (U : Opens (D.U i).carrier) :
    (D.U i).presheaf.obj (op U) ⟶ limit (D.diagramOverOpen U) :=
  limit.lift (D.diagramOverOpen U)
    { pt := (D.U i).presheaf.obj (op U)
      π :=
        { app := fun j => D.ιInvAppπApp U (unop j)
          naturality := fun {X Y} f' => by
            /-
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              D : AlgebraicGeometry.PresheafedSpace.GlueData C
              inst✝ : CategoryTheory.Limits.HasLimits C
              i : D.J
              U : TopologicalSpace.Opens ↑↑(D.U i)
              X Y : Opposite (CategoryTheory.Limits.WalkingMultispan D.diagram.fstFrom D.dia …
              f' : Quiver.Hom X Y
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const (Oppo …
            -/
            induction X using Opposite.rec' with | h X => ?_
            /-
              case h
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              D : AlgebraicGeometry.PresheafedSpace.GlueData C
              inst✝ : CategoryTheory.Limits.HasLimits C
              i : D.J
              U : TopologicalSpace.Opens ↑↑(D.U i)
              Y : Opposite (CategoryTheory.Limits.WalkingMultispan D.diagram.fstFrom D.diagr …
              X : CategoryTheory.Limits.WalkingMultispan D.diagram.fstFrom D.diagram.sndFrom
              f' : Quiver.Hom { unop := X } Y
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const (Oppo …
            -/
            induction Y using Opposite.rec' with | h Y => ?_
            /-
              case h.h
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              D : AlgebraicGeometry.PresheafedSpace.GlueData C
              inst✝ : CategoryTheory.Limits.HasLimits C
              i : D.J
              U : TopologicalSpace.Opens ↑↑(D.U i)
              X Y : CategoryTheory.Limits.WalkingMultispan D.diagram.fstFrom D.diagram.sndFrom
              f' : Quiver.Hom { unop := X } { unop := Y }
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const (Oppo …
            -/
            let f : Y ⟶ X := f'.unop; have : f' = f.op := rfl; clear_value f; subst this
            /-
              case h.h
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              D : AlgebraicGeometry.PresheafedSpace.GlueData C
              inst✝ : CategoryTheory.Limits.HasLimits C
              i : D.J
              U : TopologicalSpace.Opens ↑↑(D.U i)
              X Y : CategoryTheory.Limits.WalkingMultispan D.diagram.fstFrom D.diagram.sndFrom
              f : Quiver.Hom Y X
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const (Oppo …
            -/
            rcases f with (_ | ⟨j, k⟩ | ⟨j, k⟩)
              /-
                case h.h.id
                C : Type u
                inst✝¹ : CategoryTheory.Category.{v, u} C
                D : AlgebraicGeometry.PresheafedSpace.GlueData C
                inst✝ : CategoryTheory.Limits.HasLimits C
                i : D.J
                U : TopologicalSpace.Opens ↑↑(D.U i)
                X : CategoryTheory.Limits.WalkingMultispan D.diagram.fstFrom D.diagram.sndFrom
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const (Oppo …
              -/
            · erw [Category.id_comp, CategoryTheory.Functor.map_id]
              /-
                case h.h.id
                C : Type u
                inst✝¹ : CategoryTheory.Category.{v, u} C
                D : AlgebraicGeometry.PresheafedSpace.GlueData C
                inst✝ : CategoryTheory.Limits.HasLimits C
                i : D.J
                U : TopologicalSpace.Opens ↑↑(D.U i)
                X : CategoryTheory.Limits.WalkingMultispan D.diagram.fstFrom D.diagram.sndFrom
                ⊢ Eq ((fun j => D.ιInvAppπApp U (Opposite.unop j)) { unop := X }) (CategoryThe …
              -/
              rw [Category.comp_id]
              /-
                🎉 no goals
              -/
              /-
                case h.h.fst.mk
                C : Type u
                inst✝¹ : CategoryTheory.Category.{v, u} C
                D : AlgebraicGeometry.PresheafedSpace.GlueData C
                inst✝ : CategoryTheory.Limits.HasLimits C
                i : D.J
                U : TopologicalSpace.Opens ↑↑(D.U i)
                j k : D.J
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const (Oppo …
              -/
            · erw [Category.id_comp]; congr 1
                                      /-
                                        🎉 no goals
                                      -/
            /-
              case h.h.snd.mk
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              D : AlgebraicGeometry.PresheafedSpace.GlueData C
              inst✝ : CategoryTheory.Limits.HasLimits C
              i : D.J
              U : TopologicalSpace.Opens ↑↑(D.U i)
              j k : D.J
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const (Oppo …
            -/
            erw [Category.id_comp]
            -- It remains to show that the blue is equal to red + green in the original diagram.
            -- The proof strategy is illustrated in ![this diagram](https://i.imgur.com/mBzV1Rx.png)
            -- where we prove red = pink = light-blue = green = blue.
            change
              D.opensImagePreimageMap i j U ≫
                  (D.f j k).c.app _ ≫ (D.V (j, k)).presheaf.map (eqToHom _) =
                D.opensImagePreimageMap _ _ _ ≫
                  ((D.f k j).c.app _ ≫ (D.t j k).c.app _) ≫ (D.V (j, k)).presheaf.map (eqToHom _)
            /-
              case h.h.snd.mk
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              D : AlgebraicGeometry.PresheafedSpace.GlueData C
              inst✝ : CategoryTheory.Limits.HasLimits C
              i : D.J
              U : TopologicalSpace.Opens ↑↑(D.U i)
              j k : D.J
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.opensImagePreimageMap i j U) (Cate …
            -/
            rw [opensImagePreimageMap_app_assoc]
            /-
              case h.h.snd.mk
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              D : AlgebraicGeometry.PresheafedSpace.GlueData C
              inst✝ : CategoryTheory.Limits.HasLimits C
              i : D.J
              U : TopologicalSpace.Opens ↑↑(D.U i)
              j k : D.J
              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.CategoryStruct.comp  …
            -/
            simp_rw [Category.assoc]
            rw [opensImagePreimageMap_app_assoc, (D.t j k).c.naturality_assoc,
                snd_invApp_t_app_assoc,
                ← PresheafedSpace.comp_c_app_assoc]
            -- light-blue = green is relatively easy since the part that differs does not involve
            -- partial inverses.
            have :
              D.t' j k i ≫ (π₁ k, i, j) ≫ D.t k i ≫ 𝖣.f i k =
                (pullbackSymmetry _ _).hom ≫ (π₁ j, i, k) ≫ D.t j i ≫ D.f i j := by
              rw [← 𝖣.t_fac_assoc, 𝖣.t'_comp_eq_pullbackSymmetry_assoc,
                pullbackSymmetry_hom_comp_snd_assoc, pullback.condition, 𝖣.t_fac_assoc]
            rw [congr_app this,
                PresheafedSpace.comp_c_app_assoc (pullbackSymmetry _ _).hom]
            /-
              case h.h.snd.mk
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              D : AlgebraicGeometry.PresheafedSpace.GlueData C
              inst✝ : CategoryTheory.Limits.HasLimits C
              i : D.J
              U : TopologicalSpace.Opens ↑↑(D.U i)
              j k : D.J
              this : Eq (CategoryTheory.CategoryStruct.comp (D.t' j k i) (CategoryTheory.Cat …
              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.CategoryStruct.comp  …
            -/
            simp_rw [Category.assoc]
            /-
              case h.h.snd.mk
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              D : AlgebraicGeometry.PresheafedSpace.GlueData C
              inst✝ : CategoryTheory.Limits.HasLimits C
              i : D.J
              U : TopologicalSpace.Opens ↑↑(D.U i)
              j k : D.J
              this : Eq (CategoryTheory.CategoryStruct.comp (D.t' j k i) (CategoryTheory.Cat …
              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.CategoryStruct.comp  …
            -/
            congr 1
            rw [← IsIso.eq_inv_comp,
                IsOpenImmersion.inv_invApp]
            /-
              case h.h.snd.mk.e_a
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              D : AlgebraicGeometry.PresheafedSpace.GlueData C
              inst✝ : CategoryTheory.Limits.HasLimits C
              i : D.J
              U : TopologicalSpace.Opens ↑↑(D.U i)
              j k : D.J
              this : Eq (CategoryTheory.CategoryStruct.comp (D.t' j k i) (CategoryTheory.Cat …
              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((D.V { fst := j, snd := k }).preshea …
            -/
            simp_rw [Category.assoc]
            erw [NatTrans.naturality_assoc, ← PresheafedSpace.comp_c_app_assoc,
              congr_app (pullbackSymmetry_hom_comp_snd _ _)]
            /-
              case h.h.snd.mk.e_a
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              D : AlgebraicGeometry.PresheafedSpace.GlueData C
              inst✝ : CategoryTheory.Limits.HasLimits C
              i : D.J
              U : TopologicalSpace.Opens ↑↑(D.U i)
              j k : D.J
              this : Eq (CategoryTheory.CategoryStruct.comp (D.t' j k i) (CategoryTheory.Cat …
              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((D.V { fst := j, snd := k }).preshea …
            -/
            simp_rw [Category.assoc]
            erw [IsOpenImmersion.inv_naturality_assoc, IsOpenImmersion.inv_naturality_assoc,
              IsOpenImmersion.inv_naturality_assoc, IsOpenImmersion.app_invApp_assoc]
            /-
              case h.h.snd.mk.e_a
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              D : AlgebraicGeometry.PresheafedSpace.GlueData C
              inst✝ : CategoryTheory.Limits.HasLimits C
              i : D.J
              U : TopologicalSpace.Opens ↑↑(D.U i)
              j k : D.J
              this : Eq (CategoryTheory.CategoryStruct.comp (D.t' j k i) (CategoryTheory.Cat …
              ⊢ Eq (CategoryTheory.CategoryStruct.comp ((D.V { fst := j, snd := k }).preshea …
            -/
            rw [← (D.V (j, k)).presheaf.map_comp]
            /-
              case h.h.snd.mk.e_a
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              D : AlgebraicGeometry.PresheafedSpace.GlueData C
              inst✝ : CategoryTheory.Limits.HasLimits C
              i : D.J
              U : TopologicalSpace.Opens ↑↑(D.U i)
              j k : D.J
              this : Eq (CategoryTheory.CategoryStruct.comp (D.t' j k i) (CategoryTheory.Cat …
              ⊢ Eq ((D.V { fst := j, snd := k }).presheaf.map (CategoryTheory.CategoryStruct …
            -/
            erw [← (D.V (j, k)).presheaf.map_comp]
            /-
              case h.h.snd.mk.e_a
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              D : AlgebraicGeometry.PresheafedSpace.GlueData C
              inst✝ : CategoryTheory.Limits.HasLimits C
              i : D.J
              U : TopologicalSpace.Opens ↑↑(D.U i)
              j k : D.J
              this : Eq (CategoryTheory.CategoryStruct.comp (D.t' j k i) (CategoryTheory.Cat …
              ⊢ Eq ((D.V { fst := j, snd := k }).presheaf.map (CategoryTheory.CategoryStruct …
            -/
            repeat rw [← (D.V (j, k)).presheaf.map_comp]
            -- Porting note: was just `congr`
            /-
              case h.h.snd.mk.e_a
              C : Type u
              inst✝¹ : CategoryTheory.Category.{v, u} C
              D : AlgebraicGeometry.PresheafedSpace.GlueData C
              inst✝ : CategoryTheory.Limits.HasLimits C
              i : D.J
              U : TopologicalSpace.Opens ↑↑(D.U i)
              j k : D.J
              this : Eq (CategoryTheory.CategoryStruct.comp (D.t' j k i) (CategoryTheory.Cat …
              ⊢ Eq ((D.V { fst := j, snd := k }).presheaf.map (CategoryTheory.CategoryStruct …
            -/
            exact congr_arg ((D.V (j, k)).presheaf.map ·) rfl } }
            /-
              🎉 no goals
            -/


/-- `ιInvApp` is the left inverse of `D.ι i` on `U`. -/
theorem ιInvApp_π {i : D.J} (U : Opens (D.U i).carrier) :
    ∃ eq, D.ιInvApp U ≫ D.diagramOverOpenπ U i = (D.U i).presheaf.map (eqToHom eq) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.PresheafedSpace.GlueData C
    inst✝ : CategoryTheory.Limits.HasLimits C
    i : D.J
    U : TopologicalSpace.Opens ↑↑(D.U i)
    ⊢ Exists fun eq => Eq (CategoryTheory.CategoryStruct.comp (D.ιInvApp U) (D.dia …
  -/
  fconstructor
  -- Porting note: I don't know what the magic was in Lean3 proof, it just skipped the proof of `eq`
    /-
      case w
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      ⊢ Eq { unop := U } { unop := (TopologicalSpace.Opens.map (CategoryTheory.Limit …
    -/
  · congr; ext1; change _ = _ ⁻¹' (_ '' _); ext1 x
    /-
      case w.e_unop.h.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      x : ↑↑(D.U i)
      ⊢ Iff (Membership.mem (↑U) x) (Membership.mem (Set.preimage (⇑(CategoryTheory. …
    -/
    simp only [SetLike.mem_coe, diagram_l, diagram_r, unop_op, Set.mem_preimage, Set.mem_image]
    /-
      case w.e_unop.h.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      x : ↑↑(D.U i)
      ⊢ Iff (Membership.mem U x) (Exists fun x_1 => And (Membership.mem U x_1) (Eq ( …
    -/
    refine ⟨fun h => ⟨_, h, rfl⟩, ?_⟩
    /-
      case w.e_unop.h.h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      x : ↑↑(D.U i)
      ⊢ (Exists fun x_1 => And (Membership.mem U x_1) (Eq ((D.ι i).base x_1) ((Categ …
    -/
    rintro ⟨y, h1, h2⟩
    /-
      case w.e_unop.h.h.intro.intro
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      x y : ↑↑(D.U i)
      h1 : Membership.mem U y
      h2 : Eq ((D.ι i).base y) ((CategoryTheory.Limits.colimit.ι D.diagram.multispan …
      ⊢ Membership.mem U x
    -/
    convert h1 using 1
    /-
      case h.e'_5
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      x y : ↑↑(D.U i)
      h1 : Membership.mem U y
      h2 : Eq ((D.ι i).base y) ((CategoryTheory.Limits.colimit.ι D.diagram.multispan …
      ⊢ Eq x y
    -/
    delta ι Multicoequalizer.π at h2
    /-
      case h.e'_5
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      x y : ↑↑(D.U i)
      h1 : Membership.mem U y
      h2 : Eq ((CategoryTheory.Limits.colimit.ι D.diagram.multispan (CategoryTheory. …
      ⊢ Eq x y
    -/
    apply_fun (D.ι _).base
      /-
        case h.e'_5
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : AlgebraicGeometry.PresheafedSpace.GlueData C
        inst✝ : CategoryTheory.Limits.HasLimits C
        i : D.J
        U : TopologicalSpace.Opens ↑↑(D.U i)
        x y : ↑↑(D.U i)
        h1 : Membership.mem U y
        h2 : Eq ((CategoryTheory.Limits.colimit.ι D.diagram.multispan (CategoryTheory. …
        ⊢ Eq ((D.ι i).base x) ((D.ι i).base y)
      -/
    · exact h2.symm
      /-
        🎉 no goals
      -/
      /-
        case h.e'_5.inj
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : AlgebraicGeometry.PresheafedSpace.GlueData C
        inst✝ : CategoryTheory.Limits.HasLimits C
        i : D.J
        U : TopologicalSpace.Opens ↑↑(D.U i)
        x y : ↑↑(D.U i)
        h1 : Membership.mem U y
        h2 : Eq ((CategoryTheory.Limits.colimit.ι D.diagram.multispan (CategoryTheory. …
        ⊢ Function.Injective ⇑(D.ι i).base
      -/
    · have := D.ι_gluedIso_inv (PresheafedSpace.forget _) i
      /-
        case h.e'_5.inj
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : AlgebraicGeometry.PresheafedSpace.GlueData C
        inst✝ : CategoryTheory.Limits.HasLimits C
        i : D.J
        U : TopologicalSpace.Opens ↑↑(D.U i)
        x y : ↑↑(D.U i)
        h1 : Membership.mem U y
        h2 : Eq ((CategoryTheory.Limits.colimit.ι D.diagram.multispan (CategoryTheory. …
        this : Eq (CategoryTheory.CategoryStruct.comp ((D.mapGlueData (AlgebraicGeomet …
        ⊢ Function.Injective ⇑(D.ι i).base
      -/
      dsimp at this
      /-
        case h.e'_5.inj
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : AlgebraicGeometry.PresheafedSpace.GlueData C
        inst✝ : CategoryTheory.Limits.HasLimits C
        i : D.J
        U : TopologicalSpace.Opens ↑↑(D.U i)
        x y : ↑↑(D.U i)
        h1 : Membership.mem U y
        h2 : Eq ((CategoryTheory.Limits.colimit.ι D.diagram.multispan (CategoryTheory. …
        this : Eq (CategoryTheory.CategoryStruct.comp ((D.mapGlueData (AlgebraicGeomet …
        ⊢ Function.Injective ⇑(D.ι i).base
      -/
      rw [← this, coe_comp]
      /-
        case h.e'_5.inj
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : AlgebraicGeometry.PresheafedSpace.GlueData C
        inst✝ : CategoryTheory.Limits.HasLimits C
        i : D.J
        U : TopologicalSpace.Opens ↑↑(D.U i)
        x y : ↑↑(D.U i)
        h1 : Membership.mem U y
        h2 : Eq ((CategoryTheory.Limits.colimit.ι D.diagram.multispan (CategoryTheory. …
        this : Eq (CategoryTheory.CategoryStruct.comp ((D.mapGlueData (AlgebraicGeomet …
        ⊢ Function.Injective (Function.comp ⇑(D.gluedIso (AlgebraicGeometry.Presheafed …
      -/
      refine Function.Injective.comp ?_ (TopCat.GlueData.ι_injective D.toTopGlueData i)
      /-
        case h.e'_5.inj
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : AlgebraicGeometry.PresheafedSpace.GlueData C
        inst✝ : CategoryTheory.Limits.HasLimits C
        i : D.J
        U : TopologicalSpace.Opens ↑↑(D.U i)
        x y : ↑↑(D.U i)
        h1 : Membership.mem U y
        h2 : Eq ((CategoryTheory.Limits.colimit.ι D.diagram.multispan (CategoryTheory. …
        this : Eq (CategoryTheory.CategoryStruct.comp ((D.mapGlueData (AlgebraicGeomet …
        ⊢ Function.Injective ⇑(D.gluedIso (AlgebraicGeometry.PresheafedSpace.forget C) …
      -/
      rw [← TopCat.mono_iff_injective]
      /-
        case h.e'_5.inj
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : AlgebraicGeometry.PresheafedSpace.GlueData C
        inst✝ : CategoryTheory.Limits.HasLimits C
        i : D.J
        U : TopologicalSpace.Opens ↑↑(D.U i)
        x y : ↑↑(D.U i)
        h1 : Membership.mem U y
        h2 : Eq ((CategoryTheory.Limits.colimit.ι D.diagram.multispan (CategoryTheory. …
        this : Eq (CategoryTheory.CategoryStruct.comp ((D.mapGlueData (AlgebraicGeomet …
        ⊢ CategoryTheory.Mono (D.gluedIso (AlgebraicGeometry.PresheafedSpace.forget C) …
      -/
      infer_instance
      /-
        🎉 no goals
      -/
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.PresheafedSpace.GlueData C
    inst✝ : CategoryTheory.Limits.HasLimits C
    i : D.J
    U : TopologicalSpace.Opens ↑↑(D.U i)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.ιInvApp U) (D.diagramOverOpenπ U i …
  -/
  delta ιInvApp
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.PresheafedSpace.GlueData C
    inst✝ : CategoryTheory.Limits.HasLimits C
    i : D.J
    U : TopologicalSpace.Opens ↑↑(D.U i)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.lift (D. …
  -/
  rw [limit.lift_π]
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.PresheafedSpace.GlueData C
    inst✝ : CategoryTheory.Limits.HasLimits C
    i : D.J
    U : TopologicalSpace.Opens ↑↑(D.U i)
    ⊢ Eq ({ pt := (D.U i).presheaf.obj { unop := U }, π := { app := fun j => D.ιIn …
  -/
  change D.opensImagePreimageMap i i U = _
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.PresheafedSpace.GlueData C
    inst✝ : CategoryTheory.Limits.HasLimits C
    i : D.J
    U : TopologicalSpace.Opens ↑↑(D.U i)
    ⊢ Eq (D.opensImagePreimageMap i i U) ((D.U i).presheaf.map (CategoryTheory.eqT …
  -/
  dsimp [opensImagePreimageMap]
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.PresheafedSpace.GlueData C
    inst✝ : CategoryTheory.Limits.HasLimits C
    i : D.J
    U : TopologicalSpace.Opens ↑↑(D.U i)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((D.f i i).c.app { unop := U }) (Cate …
  -/
  rw [congr_app (D.t_id _), id_c_app, ← Functor.map_comp]
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.PresheafedSpace.GlueData C
    inst✝ : CategoryTheory.Limits.HasLimits C
    i : D.J
    U : TopologicalSpace.Opens ↑↑(D.U i)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((D.f i i).c.app { unop := U }) (Cate …
  -/
  erw [IsOpenImmersion.inv_naturality_assoc, IsOpenImmersion.app_inv_app'_assoc]
    /-
      case h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((D.U i).presheaf.map (CategoryTheory …
    -/
  · simp only [eqToHom_op, eqToHom_trans, eqToHom_map (Functor.op _), ← Functor.map_comp]
    /-
      case h
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      ⊢ Eq ((D.U i).presheaf.map (CategoryTheory.CategoryStruct.comp (CategoryTheory …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case h.hU
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      ⊢ HasSubset.Subset (↑U) (Set.range ⇑(D.f i i).base)
    -/
  · rw [Set.range_eq_univ.mpr _]
      /-
        case h.hU
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : AlgebraicGeometry.PresheafedSpace.GlueData C
        inst✝ : CategoryTheory.Limits.HasLimits C
        i : D.J
        U : TopologicalSpace.Opens ↑↑(D.U i)
        ⊢ HasSubset.Subset (↑U) Set.univ
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : AlgebraicGeometry.PresheafedSpace.GlueData C
        inst✝ : CategoryTheory.Limits.HasLimits C
        i : D.J
        U : TopologicalSpace.Opens ↑↑(D.U i)
        ⊢ Function.Surjective ⇑(D.f i i).base
      -/
    · rw [← TopCat.epi_iff_surjective]
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : AlgebraicGeometry.PresheafedSpace.GlueData C
        inst✝ : CategoryTheory.Limits.HasLimits C
        i : D.J
        U : TopologicalSpace.Opens ↑↑(D.U i)
        ⊢ CategoryTheory.Epi (D.f i i).base
      -/
      infer_instance
      /-
        🎉 no goals
      -/


/-- The `eqToHom` given by `ιInvApp_π`. -/
abbrev ιInvAppπEqMap {i : D.J} (U : Opens (D.U i).carrier) :=
  (D.U i).presheaf.map (eqToIso (D.ιInvApp_π U).choose).inv


/-- `ιInvApp` is the right inverse of `D.ι i` on `U`. -/
theorem π_ιInvApp_π (i j : D.J) (U : Opens (D.U i).carrier) :
    D.diagramOverOpenπ U i ≫ D.ιInvAppπEqMap U ≫ D.ιInvApp U ≫ D.diagramOverOpenπ U j =
      D.diagramOverOpenπ U j := by
  rw [← @cancel_mono
          (f := (componentwiseDiagram 𝖣.diagram.multispan _).map
            (Quiver.Hom.op (WalkingMultispan.Hom.snd (i, j))) ≫ 𝟙 _) ..]
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i j : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
  · simp_rw [Category.assoc]
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i j : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.diagramOverOpenπ U i) (CategoryThe …
    -/
    rw [limit.w_assoc]
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i j : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.diagramOverOpenπ U i) (CategoryThe …
    -/
    erw [limit.lift_π_assoc]
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i j : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.diagramOverOpenπ U i) (CategoryThe …
    -/
    rw [Category.comp_id, Category.comp_id]
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i j : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.diagramOverOpenπ U i) (CategoryThe …
    -/
    change _ ≫ _ ≫ (_ ≫ _) ≫ _ = _
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i j : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.diagramOverOpenπ U i) (CategoryThe …
    -/
    rw [congr_app (D.t_id _), id_c_app]
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i j : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.diagramOverOpenπ U i) (CategoryThe …
    -/
    simp_rw [Category.assoc]
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i j : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.diagramOverOpenπ U i) (CategoryThe …
    -/
    rw [← Functor.map_comp_assoc]
    -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11224): change `rw` to `erw`
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i j : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.diagramOverOpenπ U i) (CategoryThe …
    -/
    erw [IsOpenImmersion.inv_naturality_assoc]
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i j : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.diagramOverOpenπ U i) (CategoryThe …
    -/
    erw [IsOpenImmersion.app_invApp_assoc]
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i j : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.diagramOverOpenπ U i) (CategoryThe …
    -/
    iterate 3 rw [← Functor.map_comp_assoc]
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i j : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.diagramOverOpenπ U i) (CategoryThe …
    -/
    rw [NatTrans.naturality_assoc]
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i j : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.diagramOverOpenπ U i) (CategoryThe …
    -/
    erw [← (D.V (i, j)).presheaf.map_comp]
    convert
      limit.w (componentwiseDiagram 𝖣.diagram.multispan _)
        (Quiver.Hom.op (WalkingMultispan.Hom.fst (i, j)))
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i j : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      ⊢ CategoryTheory.Mono (CategoryTheory.CategoryStruct.comp ((AlgebraicGeometry. …
    -/
  · rw [Category.comp_id]
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i j : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      ⊢ CategoryTheory.Mono ((AlgebraicGeometry.PresheafedSpace.componentwiseDiagram …
    -/
    apply (config := { allowSynthFailures := true }) mono_comp
    /-
      case inst
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i j : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      ⊢ CategoryTheory.Mono ((D.diagram.multispan.map (Quiver.Hom.op (CategoryTheory …
    -/
    change Mono ((_ ≫ D.f j i).c.app _)
    /-
      case inst
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i j : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      ⊢ CategoryTheory.Mono ((CategoryTheory.CategoryStruct.comp (D.t i j) (D.f j i) …
    -/
    rw [comp_c_app]
    /-
      case inst
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i j : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      ⊢ CategoryTheory.Mono (CategoryTheory.CategoryStruct.comp ((D.f j i).c.app { u …
    -/
    apply (config := { allowSynthFailures := true }) mono_comp
      /-
        case inst
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : AlgebraicGeometry.PresheafedSpace.GlueData C
        inst✝ : CategoryTheory.Limits.HasLimits C
        i j : D.J
        U : TopologicalSpace.Opens ↑↑(D.U i)
        ⊢ CategoryTheory.Mono ((D.f j i).c.app { unop := (TopologicalSpace.Opens.map ( …
      -/
    · erw [D.ι_image_preimage_eq i j U]
      /-
        case inst
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : AlgebraicGeometry.PresheafedSpace.GlueData C
        inst✝ : CategoryTheory.Limits.HasLimits C
        i j : D.J
        U : TopologicalSpace.Opens ↑↑(D.U i)
        ⊢ CategoryTheory.Mono ((D.f j i).c.app { unop := (AlgebraicGeometry.Presheafed …
      -/
      infer_instance
      /-
        🎉 no goals
      -/
      /-
        case inst
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : AlgebraicGeometry.PresheafedSpace.GlueData C
        inst✝ : CategoryTheory.Limits.HasLimits C
        i j : D.J
        U : TopologicalSpace.Opens ↑↑(D.U i)
        ⊢ CategoryTheory.Mono ((D.t i j).c.app { unop := (TopologicalSpace.Opens.map ( …
      -/
    · have : IsIso (D.t i j).c := by apply c_isIso_of_iso
      /-
        case inst
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : AlgebraicGeometry.PresheafedSpace.GlueData C
        inst✝ : CategoryTheory.Limits.HasLimits C
        i j : D.J
        U : TopologicalSpace.Opens ↑↑(D.U i)
        this : CategoryTheory.IsIso (D.t i j).c
        ⊢ CategoryTheory.Mono ((D.t i j).c.app { unop := (TopologicalSpace.Opens.map ( …
      -/
      infer_instance
      /-
        🎉 no goals
      -/


/-- `ιInvApp` is the inverse of `D.ι i` on `U`. -/
theorem π_ιInvApp_eq_id (i : D.J) (U : Opens (D.U i).carrier) :
    D.diagramOverOpenπ U i ≫ D.ιInvAppπEqMap U ≫ D.ιInvApp U = 𝟙 _ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.PresheafedSpace.GlueData C
    inst✝ : CategoryTheory.Limits.HasLimits C
    i : D.J
    U : TopologicalSpace.Opens ↑↑(D.U i)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.diagramOverOpenπ U i) (CategoryThe …
  -/
  ext j
  /-
    case w
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.PresheafedSpace.GlueData C
    inst✝ : CategoryTheory.Limits.HasLimits C
    i : D.J
    U : TopologicalSpace.Opens ↑↑(D.U i)
    j : Opposite (CategoryTheory.Limits.WalkingMultispan D.diagram.fstFrom D.diagr …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  induction j using Opposite.rec' with | h j => ?_
  /-
    case w.h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.PresheafedSpace.GlueData C
    inst✝ : CategoryTheory.Limits.HasLimits C
    i : D.J
    U : TopologicalSpace.Opens ↑↑(D.U i)
    j : CategoryTheory.Limits.WalkingMultispan D.diagram.fstFrom D.diagram.sndFrom
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  rcases j with (⟨j, k⟩ | ⟨j⟩)
  · rw [← limit.w (componentwiseDiagram 𝖣.diagram.multispan _)
        (Quiver.Hom.op (WalkingMultispan.Hom.fst (j, k))),
      ← Category.assoc, Category.id_comp]
    /-
      case w.h.left.mk
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      j k : D.J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    congr 1
    /-
      case w.h.left.mk.e_a
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      j k : D.J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp_rw [Category.assoc]
    /-
      case w.h.left.mk.e_a
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      j k : D.J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.diagramOverOpenπ U i) (CategoryThe …
    -/
    apply π_ιInvApp_π
    /-
      🎉 no goals
    -/
    /-
      case w.h.right
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      j : D.diagram.R
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
  · simp_rw [Category.assoc]
    /-
      case w.h.right
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      j : D.diagram.R
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.diagramOverOpenπ U i) (CategoryThe …
    -/
    rw [Category.id_comp]
    /-
      case w.h.right
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      j : D.diagram.R
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.diagramOverOpenπ U i) (CategoryThe …
    -/
    apply π_ιInvApp_π
    /-
      🎉 no goals
    -/


instance componentwise_diagram_π_isIso (i : D.J) (U : Opens (D.U i).carrier) :
    IsIso (D.diagramOverOpenπ U i) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.PresheafedSpace.GlueData C
    inst✝ : CategoryTheory.Limits.HasLimits C
    i : D.J
    U : TopologicalSpace.Opens ↑↑(D.U i)
    ⊢ CategoryTheory.IsIso (D.diagramOverOpenπ U i)
  -/
  use D.ιInvAppπEqMap U ≫ D.ιInvApp U
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.PresheafedSpace.GlueData C
    inst✝ : CategoryTheory.Limits.HasLimits C
    i : D.J
    U : TopologicalSpace.Opens ↑↑(D.U i)
    ⊢ And (Eq (CategoryTheory.CategoryStruct.comp (D.diagramOverOpenπ U i) (Catego …
  -/
  constructor
    /-
      case h.left
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.diagramOverOpenπ U i) (CategoryThe …
    -/
  · apply π_ιInvApp_eq_id
    /-
      🎉 no goals
    -/
    /-
      case h.right
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
  · rw [Category.assoc, (D.ιInvApp_π _).choose_spec]
    /-
      case h.right
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i : D.J
      U : TopologicalSpace.Opens ↑↑(D.U i)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (D.ιInvAppπEqMap U) ((D.U i).presheaf …
    -/
    exact Iso.inv_hom_id ((D.U i).presheaf.mapIso (eqToIso _))
    /-
      🎉 no goals
    -/


instance ιIsOpenImmersion (i : D.J) : IsOpenImmersion (𝖣.ι i) where
  base_open := D.ι_isOpenEmbedding i
                /-
                  C : Type u
                  inst✝¹ : CategoryTheory.Category.{v, u} C
                  D : AlgebraicGeometry.PresheafedSpace.GlueData C
                  inst✝ : CategoryTheory.Limits.HasLimits C
                  i : D.J
                  U : TopologicalSpace.Opens ↑↑(D.U i)
                  ⊢ CategoryTheory.IsIso ((D.ι i).c.app { unop := ⋯.functor.obj U })
                -/
  c_iso U := by erw [← colimitPresheafObjIsoComponentwiseLimit_hom_π]; infer_instance
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


/-- The following diagram is a pullback, i.e. `Vᵢⱼ` is the intersection of `Uᵢ` and `Uⱼ` in `X`.

Vᵢⱼ ⟶ Uᵢ
 |      |
 ↓      ↓
 Uⱼ ⟶ X
-/
def vPullbackConeIsLimit (i j : D.J) : IsLimit (𝖣.vPullbackCone i j) :=
  PullbackCone.isLimitAux' _ fun s => by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      D : AlgebraicGeometry.PresheafedSpace.GlueData C
      inst✝ : CategoryTheory.Limits.HasLimits C
      i j : D.J
      s : CategoryTheory.Limits.PullbackCone (D.ι i) (D.ι j)
      ⊢ Subtype fun l => And (Eq (CategoryTheory.CategoryStruct.comp l (D.vPullbackC …
    -/
    refine ⟨?_, ?_, ?_, ?_⟩
      /-
        case refine_1
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : AlgebraicGeometry.PresheafedSpace.GlueData C
        inst✝ : CategoryTheory.Limits.HasLimits C
        i j : D.J
        s : CategoryTheory.Limits.PullbackCone (D.ι i) (D.ι j)
        ⊢ Quiver.Hom s.pt (D.vPullbackCone i j).pt
      -/
    · refine PresheafedSpace.IsOpenImmersion.lift (D.f i j) s.fst ?_
      /-
        case refine_1
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : AlgebraicGeometry.PresheafedSpace.GlueData C
        inst✝ : CategoryTheory.Limits.HasLimits C
        i j : D.J
        s : CategoryTheory.Limits.PullbackCone (D.ι i) (D.ι j)
        ⊢ HasSubset.Subset (Set.range ⇑s.fst.base) (Set.range ⇑(D.f i j).base)
      -/
      erw [← D.toTopGlueData.preimage_range j i]
      have :
        s.fst.base ≫ D.toTopGlueData.ι i =
          s.snd.base ≫ D.toTopGlueData.ι j := by
        rw [← 𝖣.ι_gluedIso_hom (PresheafedSpace.forget _) _, ←
          𝖣.ι_gluedIso_hom (PresheafedSpace.forget _) _]
        have := congr_arg PresheafedSpace.Hom.base s.condition
        rw [comp_base, comp_base] at this
        replace this := reassoc_of% this
        exact this _
      /-
        case refine_1
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : AlgebraicGeometry.PresheafedSpace.GlueData C
        inst✝ : CategoryTheory.Limits.HasLimits C
        i j : D.J
        s : CategoryTheory.Limits.PullbackCone (D.ι i) (D.ι j)
        this : Eq (CategoryTheory.CategoryStruct.comp s.fst.base (D.toTopGlueData.ι i) …
        ⊢ HasSubset.Subset (Set.range ⇑s.fst.base) (Set.preimage (⇑(D.toTopGlueData.ι  …
      -/
      rw [← Set.image_subset_iff, ← Set.image_univ, ← Set.image_comp, Set.image_univ]
      -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11224): change `rw` to `erw`
      /-
        case refine_1
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : AlgebraicGeometry.PresheafedSpace.GlueData C
        inst✝ : CategoryTheory.Limits.HasLimits C
        i j : D.J
        s : CategoryTheory.Limits.PullbackCone (D.ι i) (D.ι j)
        this : Eq (CategoryTheory.CategoryStruct.comp s.fst.base (D.toTopGlueData.ι i) …
        ⊢ HasSubset.Subset (Set.range (Function.comp ⇑(D.toTopGlueData.ι i) ⇑s.fst.bas …
      -/
      erw [← coe_comp]
      /-
        case refine_1
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : AlgebraicGeometry.PresheafedSpace.GlueData C
        inst✝ : CategoryTheory.Limits.HasLimits C
        i j : D.J
        s : CategoryTheory.Limits.PullbackCone (D.ι i) (D.ι j)
        this : Eq (CategoryTheory.CategoryStruct.comp s.fst.base (D.toTopGlueData.ι i) …
        ⊢ HasSubset.Subset (Set.range ⇑(CategoryTheory.CategoryStruct.comp s.fst.base  …
      -/
      rw [this, coe_comp, ← Set.image_univ, Set.image_comp]
      /-
        case refine_1
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : AlgebraicGeometry.PresheafedSpace.GlueData C
        inst✝ : CategoryTheory.Limits.HasLimits C
        i j : D.J
        s : CategoryTheory.Limits.PullbackCone (D.ι i) (D.ι j)
        this : Eq (CategoryTheory.CategoryStruct.comp s.fst.base (D.toTopGlueData.ι i) …
        ⊢ HasSubset.Subset (Set.image (⇑(D.toTopGlueData.ι j)) (Set.image (⇑s.snd.base …
      -/
      exact Set.image_subset_range _ _
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : AlgebraicGeometry.PresheafedSpace.GlueData C
        inst✝ : CategoryTheory.Limits.HasLimits C
        i j : D.J
        s : CategoryTheory.Limits.PullbackCone (D.ι i) (D.ι j)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.PresheafedSpace.Is …
      -/
    · apply IsOpenImmersion.lift_fac
      /-
        🎉 no goals
      -/
      /-
        case refine_3
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : AlgebraicGeometry.PresheafedSpace.GlueData C
        inst✝ : CategoryTheory.Limits.HasLimits C
        i j : D.J
        s : CategoryTheory.Limits.PullbackCone (D.ι i) (D.ι j)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.PresheafedSpace.Is …
      -/
    · rw [← cancel_mono (𝖣.ι j), Category.assoc, ← (𝖣.vPullbackCone i j).condition]
      /-
        case refine_3
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : AlgebraicGeometry.PresheafedSpace.GlueData C
        inst✝ : CategoryTheory.Limits.HasLimits C
        i j : D.J
        s : CategoryTheory.Limits.PullbackCone (D.ι i) (D.ι j)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.PresheafedSpace.Is …
      -/
      conv_rhs => rw [← s.condition]
      /-
        case refine_3
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : AlgebraicGeometry.PresheafedSpace.GlueData C
        inst✝ : CategoryTheory.Limits.HasLimits C
        i j : D.J
        s : CategoryTheory.Limits.PullbackCone (D.ι i) (D.ι j)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.PresheafedSpace.Is …
      -/
      erw [IsOpenImmersion.lift_fac_assoc]
      /-
        🎉 no goals
      -/
      /-
        case refine_4
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        D : AlgebraicGeometry.PresheafedSpace.GlueData C
        inst✝ : CategoryTheory.Limits.HasLimits C
        i j : D.J
        s : CategoryTheory.Limits.PullbackCone (D.ι i) (D.ι j)
        ⊢ ∀ {m : Quiver.Hom s.pt (D.vPullbackCone i j).pt}, Eq (CategoryTheory.Categor …
      -/
    · intro m e₁ _; rw [← cancel_mono (D.f i j)]; erw [e₁]; rw [IsOpenImmersion.lift_fac]
                                                            /-
                                                              🎉 no goals
                                                            -/


theorem ι_jointly_surjective (x : 𝖣.glued) : ∃ (i : D.J) (y : D.U i), (𝖣.ι i).base y = x :=
  𝖣.ι_jointly_surjective (PresheafedSpace.forget _ ⋙ CategoryTheory.forget TopCat) x


/-- A family of gluing data consists of
1. An index type `J`
2. A sheafed space `U i` for each `i : J`.
3. A sheafed space `V i j` for each `i j : J`.
  (Note that this is `J × J → SheafedSpace C` rather than `J → J → SheafedSpace C` to
  connect to the limits library easier.)
4. An open immersion `f i j : V i j ⟶ U i` for each `i j : ι`.
5. A transition map `t i j : V i j ⟶ V j i` for each `i j : ι`.
such that
6. `f i i` is an isomorphism.
7. `t i i` is the identity.
8. `V i j ×[U i] V i k ⟶ V i j ⟶ V j i` factors through `V j k ×[U j] V j i ⟶ V j i` via some
    `t' : V i j ×[U i] V i k ⟶ V j k ×[U j] V j i`.
9. `t' i j k ≫ t' j k i ≫ t' k i j = 𝟙 _`.

We can then glue the spaces `U i` together by identifying `V i j` with `V j i`, such
that the `U i`'s are open subspaces of the glued space.
-/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): this linter isn't ported yet.
-- @[nolint has_nonempty_instance]
structure GlueData extends CategoryTheory.GlueData (SheafedSpace.{u, v, v} C) where
  f_open : ∀ i j, SheafedSpace.IsOpenImmersion (f i j)


/-- The glue data of presheafed spaces associated to a family of glue data of sheafed spaces. -/
abbrev toPresheafedSpaceGlueData : PresheafedSpace.GlueData C :=
  { f_open := D.f_open
    toGlueData := 𝖣.mapGlueData forgetToPresheafedSpace }


/-- The gluing as sheafed spaces is isomorphic to the gluing as presheafed spaces. -/
abbrev isoPresheafedSpace :
    𝖣.glued.toPresheafedSpace ≅ D.toPresheafedSpaceGlueData.toGlueData.glued :=
  𝖣.gluedIso forgetToPresheafedSpace


theorem ι_isoPresheafedSpace_inv (i : D.J) :
    D.toPresheafedSpaceGlueData.toGlueData.ι i ≫ D.isoPresheafedSpace.inv = 𝖣.ι i :=
  𝖣.ι_gluedIso_inv _ _


instance ιIsOpenImmersion (i : D.J) : IsOpenImmersion (𝖣.ι i) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.SheafedSpace.GlueData C
    inst✝ : CategoryTheory.Limits.HasLimits C
    i : D.J
    ⊢ AlgebraicGeometry.SheafedSpace.IsOpenImmersion (D.ι i)
  -/
  rw [← D.ι_isoPresheafedSpace_inv]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.SheafedSpace.GlueData C
    inst✝ : CategoryTheory.Limits.HasLimits C
    i : D.J
    ⊢ AlgebraicGeometry.SheafedSpace.IsOpenImmersion (CategoryTheory.CategoryStruc …
  -/
  have := D.toPresheafedSpaceGlueData.ιIsOpenImmersion i
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.SheafedSpace.GlueData C
    inst✝ : CategoryTheory.Limits.HasLimits C
    i : D.J
    this : AlgebraicGeometry.PresheafedSpace.IsOpenImmersion (D.toPresheafedSpaceG …
    ⊢ AlgebraicGeometry.SheafedSpace.IsOpenImmersion (CategoryTheory.CategoryStruc …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem ι_jointly_surjective (x : 𝖣.glued) : ∃ (i : D.J) (y : D.U i), (𝖣.ι i).base y = x :=
  𝖣.ι_jointly_surjective (SheafedSpace.forget _ ⋙ CategoryTheory.forget TopCat) x


/-- The following diagram is a pullback, i.e. `Vᵢⱼ` is the intersection of `Uᵢ` and `Uⱼ` in `X`.

Vᵢⱼ ⟶ Uᵢ
 |      |
 ↓      ↓
 Uⱼ ⟶ X
-/
def vPullbackConeIsLimit (i j : D.J) : IsLimit (𝖣.vPullbackCone i j) :=
  𝖣.vPullbackConeIsLimitOfMap forgetToPresheafedSpace i j
    (D.toPresheafedSpaceGlueData.vPullbackConeIsLimit _ _)


/-- A family of gluing data consists of
1. An index type `J`
2. A locally ringed space `U i` for each `i : J`.
3. A locally ringed space `V i j` for each `i j : J`.
  (Note that this is `J × J → LocallyRingedSpace` rather than `J → J → LocallyRingedSpace` to
  connect to the limits library easier.)
4. An open immersion `f i j : V i j ⟶ U i` for each `i j : ι`.
5. A transition map `t i j : V i j ⟶ V j i` for each `i j : ι`.
such that
6. `f i i` is an isomorphism.
7. `t i i` is the identity.
8. `V i j ×[U i] V i k ⟶ V i j ⟶ V j i` factors through `V j k ×[U j] V j i ⟶ V j i` via some
    `t' : V i j ×[U i] V i k ⟶ V j k ×[U j] V j i`.
9. `t' i j k ≫ t' j k i ≫ t' k i j = 𝟙 _`.

We can then glue the spaces `U i` together by identifying `V i j` with `V j i`, such
that the `U i`'s are open subspaces of the glued space.
-/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): this linter isn't ported yet.
-- @[nolint has_nonempty_instance]
structure GlueData extends CategoryTheory.GlueData LocallyRingedSpace where
  f_open : ∀ i j, LocallyRingedSpace.IsOpenImmersion (f i j)


/-- The glue data of ringed spaces associated to a family of glue data of locally ringed spaces. -/
abbrev toSheafedSpaceGlueData : SheafedSpace.GlueData CommRingCat :=
  { f_open := D.f_open
    toGlueData := 𝖣.mapGlueData forgetToSheafedSpace }


/-- The gluing as locally ringed spaces is isomorphic to the gluing as ringed spaces. -/
abbrev isoSheafedSpace : 𝖣.glued.toSheafedSpace ≅ D.toSheafedSpaceGlueData.toGlueData.glued :=
  𝖣.gluedIso forgetToSheafedSpace


theorem ι_isoSheafedSpace_inv (i : D.J) :
    D.toSheafedSpaceGlueData.toGlueData.ι i ≫ D.isoSheafedSpace.inv = (𝖣.ι i).1 :=
  𝖣.ι_gluedIso_inv forgetToSheafedSpace i


instance ι_isOpenImmersion (i : D.J) : IsOpenImmersion (𝖣.ι i) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.LocallyRingedSpace.GlueData
    i : D.J
    ⊢ AlgebraicGeometry.LocallyRingedSpace.IsOpenImmersion (D.ι i)
  -/
  delta IsOpenImmersion; rw [← D.ι_isoSheafedSpace_inv]
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.LocallyRingedSpace.GlueData
    i : D.J
    ⊢ AlgebraicGeometry.SheafedSpace.IsOpenImmersion (CategoryTheory.CategoryStruc …
  -/
  apply (config := { allowSynthFailures := true }) PresheafedSpace.IsOpenImmersion.comp
  -- Porting note: this was automatic
  /-
    case H
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    D : AlgebraicGeometry.LocallyRingedSpace.GlueData
    i : D.J
    ⊢ AlgebraicGeometry.PresheafedSpace.IsOpenImmersion (D.toSheafedSpaceGlueData. …
  -/
  exact (D.toSheafedSpaceGlueData).ιIsOpenImmersion i
  /-
    🎉 no goals
  -/


instance (i j k : D.J) : PreservesLimit (cospan (𝖣.f i j) (𝖣.f i k)) forgetToSheafedSpace :=
  inferInstance


theorem ι_jointly_surjective (x : 𝖣.glued) : ∃ (i : D.J) (y : D.U i), (𝖣.ι i).base y = x :=
  𝖣.ι_jointly_surjective
    ((LocallyRingedSpace.forgetToSheafedSpace.{u} ⋙ SheafedSpace.forget CommRingCatMax.{u, u}) ⋙
      forget TopCat.{u}) x


/-- The following diagram is a pullback, i.e. `Vᵢⱼ` is the intersection of `Uᵢ` and `Uⱼ` in `X`.

Vᵢⱼ ⟶ Uᵢ
 |      |
 ↓      ↓
 Uⱼ ⟶ X
-/
def vPullbackConeIsLimit (i j : D.J) : IsLimit (𝖣.vPullbackCone i j) :=
  𝖣.vPullbackConeIsLimitOfMap forgetToSheafedSpace i j
    (D.toSheafedSpaceGlueData.vPullbackConeIsLimit _ _)


