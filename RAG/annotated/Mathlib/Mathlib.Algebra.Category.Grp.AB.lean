instance {J C : Type*} [Category J] [Category C] [HasColimitsOfShape J C] [Preadditive C] :
    (colim (J := J) (C := C)).Additive where


noncomputable instance :
    (colim (J := J) (C := AddCommGrp.{u})).PreservesHomology :=
  Functor.preservesHomology_of_map_exact _ (fun S hS ↦ by
    /-
      J : Type u
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsFiltered J
      S : CategoryTheory.ShortComplex (CategoryTheory.Functor J AddCommGrp)
      hS : S.Exact
      ⊢ (S.map CategoryTheory.Limits.colim).Exact
    -/
    replace hS := fun j => hS.map ((evaluation _ _).obj j)
    /-
      J : Type u
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsFiltered J
      S : CategoryTheory.ShortComplex (CategoryTheory.Functor J AddCommGrp)
      hS : ∀ (j : J), (S.map ((CategoryTheory.evaluation J AddCommGrp).obj j)).Exact
      ⊢ (S.map CategoryTheory.Limits.colim).Exact
    -/
    simp only [ShortComplex.ab_exact_iff_ker_le_range] at hS ⊢
    /-
      J : Type u
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsFiltered J
      S : CategoryTheory.ShortComplex (CategoryTheory.Functor J AddCommGrp)
      hS : ∀ (j : J), LE.le (AddMonoidHom.ker (S.map ((CategoryTheory.evaluation J A …
      ⊢ LE.le (AddMonoidHom.ker (S.map CategoryTheory.Limits.colim).g) (AddMonoidHom …
    -/
    intro x (hx : _ = _)
    /-
      J : Type u
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsFiltered J
      S : CategoryTheory.ShortComplex (CategoryTheory.Functor J AddCommGrp)
      hS : ∀ (j : J), LE.le (AddMonoidHom.ker (S.map ((CategoryTheory.evaluation J A …
      x : ↑(S.map CategoryTheory.Limits.colim).X₂
      hx : Eq ((S.map CategoryTheory.Limits.colim).g x) 0
      ⊢ Membership.mem (AddMonoidHom.range (S.map CategoryTheory.Limits.colim).f) x
    -/
    dsimp at hx
    /-
      J : Type u
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsFiltered J
      S : CategoryTheory.ShortComplex (CategoryTheory.Functor J AddCommGrp)
      hS : ∀ (j : J), LE.le (AddMonoidHom.ker (S.map ((CategoryTheory.evaluation J A …
      x : ↑(S.map CategoryTheory.Limits.colim).X₂
      hx : Eq ((CategoryTheory.Limits.colimMap S.g) x) 0
      ⊢ Membership.mem (AddMonoidHom.range (S.map CategoryTheory.Limits.colim).f) x
    -/
    rcases Concrete.colimit_exists_rep S.X₂ x with ⟨j, y, rfl⟩
    erw [← comp_apply, colimit.ι_map, comp_apply,
      ← map_zero (by exact colimit.ι S.X₃ j : (S.X₃).obj j →+ ↑(colimit S.X₃))] at hx
    rcases Concrete.colimit_exists_of_rep_eq.{u, u, u} S.X₃ _ _ hx
      with ⟨k, e₁, e₂, hk : _ = S.X₃.map e₂ 0⟩
    /-
      case intro.intro.intro.intro.intro
      J : Type u
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsFiltered J
      S : CategoryTheory.ShortComplex (CategoryTheory.Functor J AddCommGrp)
      hS : ∀ (j : J), LE.le (AddMonoidHom.ker (S.map ((CategoryTheory.evaluation J A …
      j : J
      y : (CategoryTheory.forget AddCommGrp).obj (S.X₂.obj j)
      hx : Eq ((CategoryTheory.Limits.colimit.ι S.X₃ j) ((S.g.app j) y)) ((CategoryT …
      k : J
      e₁ e₂ : Quiver.Hom j k
      hk : Eq ((S.X₃.map e₁) ((S.g.app j) y)) ((S.X₃.map e₂) 0)
      ⊢ Membership.mem (AddMonoidHom.range (S.map CategoryTheory.Limits.colim).f) (( …
    -/
    rw [map_zero, ← comp_apply, ← NatTrans.naturality, comp_apply] at hk
    /-
      case intro.intro.intro.intro.intro
      J : Type u
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsFiltered J
      S : CategoryTheory.ShortComplex (CategoryTheory.Functor J AddCommGrp)
      hS : ∀ (j : J), LE.le (AddMonoidHom.ker (S.map ((CategoryTheory.evaluation J A …
      j : J
      y : (CategoryTheory.forget AddCommGrp).obj (S.X₂.obj j)
      hx : Eq ((CategoryTheory.Limits.colimit.ι S.X₃ j) ((S.g.app j) y)) ((CategoryT …
      k : J
      e₁ e₂ : Quiver.Hom j k
      hk : Eq ((S.g.app k) ((S.X₂.map e₁) y)) 0
      ⊢ Membership.mem (AddMonoidHom.range (S.map CategoryTheory.Limits.colim).f) (( …
    -/
    rcases hS k hk with ⟨t, ht⟩
    /-
      case intro.intro.intro.intro.intro.intro
      J : Type u
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsFiltered J
      S : CategoryTheory.ShortComplex (CategoryTheory.Functor J AddCommGrp)
      hS : ∀ (j : J), LE.le (AddMonoidHom.ker (S.map ((CategoryTheory.evaluation J A …
      j : J
      y : (CategoryTheory.forget AddCommGrp).obj (S.X₂.obj j)
      hx : Eq ((CategoryTheory.Limits.colimit.ι S.X₃ j) ((S.g.app j) y)) ((CategoryT …
      k : J
      e₁ e₂ : Quiver.Hom j k
      hk : Eq ((S.g.app k) ((S.X₂.map e₁) y)) 0
      t : ↑(S.map ((CategoryTheory.evaluation J AddCommGrp).obj k)).X₁
      ht : Eq ((S.map ((CategoryTheory.evaluation J AddCommGrp).obj k)).f t) ((S.X₂. …
      ⊢ Membership.mem (AddMonoidHom.range (S.map CategoryTheory.Limits.colim).f) (( …
    -/
    use colimit.ι S.X₁ k t
    /-
      case h
      J : Type u
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsFiltered J
      S : CategoryTheory.ShortComplex (CategoryTheory.Functor J AddCommGrp)
      hS : ∀ (j : J), LE.le (AddMonoidHom.ker (S.map ((CategoryTheory.evaluation J A …
      j : J
      y : (CategoryTheory.forget AddCommGrp).obj (S.X₂.obj j)
      hx : Eq ((CategoryTheory.Limits.colimit.ι S.X₃ j) ((S.g.app j) y)) ((CategoryT …
      k : J
      e₁ e₂ : Quiver.Hom j k
      hk : Eq ((S.g.app k) ((S.X₂.map e₁) y)) 0
      t : ↑(S.map ((CategoryTheory.evaluation J AddCommGrp).obj k)).X₁
      ht : Eq ((S.map ((CategoryTheory.evaluation J AddCommGrp).obj k)).f t) ((S.X₂. …
      ⊢ Eq ((S.map CategoryTheory.Limits.colim).f ((CategoryTheory.Limits.colimit.ι  …
    -/
    erw [← comp_apply, colimit.ι_map, comp_apply, ht]
    /-
      case h
      J : Type u
      inst✝¹ : CategoryTheory.SmallCategory J
      inst✝ : CategoryTheory.IsFiltered J
      S : CategoryTheory.ShortComplex (CategoryTheory.Functor J AddCommGrp)
      hS : ∀ (j : J), LE.le (AddMonoidHom.ker (S.map ((CategoryTheory.evaluation J A …
      j : J
      y : (CategoryTheory.forget AddCommGrp).obj (S.X₂.obj j)
      hx : Eq ((CategoryTheory.Limits.colimit.ι S.X₃ j) ((S.g.app j) y)) ((CategoryT …
      k : J
      e₁ e₂ : Quiver.Hom j k
      hk : Eq ((S.g.app k) ((S.X₂.map e₁) y)) 0
      t : ↑(S.map ((CategoryTheory.evaluation J AddCommGrp).obj k)).X₁
      ht : Eq ((S.map ((CategoryTheory.evaluation J AddCommGrp).obj k)).f t) ((S.X₂. …
      ⊢ Eq ((CategoryTheory.Limits.colimit.ι S.X₂ k) ((S.X₂.map e₁) y)) ((CategoryTh …
    -/
    exact colimit.w_apply S.X₂ e₁ y)
    /-
      🎉 no goals
    -/


noncomputable instance :
    PreservesFiniteLimits <| colim (J := J) (C := AddCommGrp.{u}) := by
  /-
    J : Type u
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.IsFiltered J
    ⊢ CategoryTheory.Limits.PreservesFiniteLimits CategoryTheory.Limits.colim
  -/
  apply Functor.preservesFiniteLimits_of_preservesHomology
  /-
    🎉 no goals
  -/


instance : HasFilteredColimits (AddCommGrp.{u}) where
  HasColimitsOfShape := inferInstance


noncomputable instance : AB5 (AddCommGrp.{u}) where
  ofShape _ := { preservesFiniteLimits := inferInstance }


instance : AB4 AddCommGrp.{u} := AB4.of_AB5 _


instance : HasExactLimitsOfShape (Discrete J) (AddCommGrp.{u}) := by
  /-
    J : Type u
    inst✝¹ : CategoryTheory.SmallCategory J
    inst✝ : CategoryTheory.IsFiltered J
    ⊢ CategoryTheory.HasExactLimitsOfShape (CategoryTheory.Discrete J) AddCommGrp
  -/
  apply ( config := {allowSynthFailures := true} ) hasExactLimitsOfShape_of_preservesEpi
  exact {
    preserves {X Y} f hf := by
      let iX : limit X ≅ AddCommGrp.of ((i : J) → X.obj ⟨i⟩) := (Pi.isoLimit X).symm ≪≫
          (limit.isLimit _).conePointUniqueUpToIso (AddCommGrp.HasLimit.productLimitCone _).isLimit
      let iY : limit Y ≅ AddCommGrp.of ((i : J) → Y.obj ⟨i⟩) := (Pi.isoLimit Y).symm ≪≫
          (limit.isLimit _).conePointUniqueUpToIso (AddCommGrp.HasLimit.productLimitCone _).isLimit
      have : Pi.map (fun i ↦ f.app ⟨i⟩) = iX.inv ≫ lim.map f ≫ iY.hom := by
        simp only [AddCommGrp.coe_of, Functor.comp_obj, Discrete.functor_obj_eq_as, Discrete.mk_as,
          Pi.isoLimit, IsLimit.conePointUniqueUpToIso, limit.cone,
          AddCommGrp.HasLimit.productLimitCone, Iso.trans_inv, Functor.mapIso_inv,
          IsLimit.uniqueUpToIso_inv, Cones.forget_map, IsLimit.liftConeMorphism_hom,
          limit.isLimit_lift, Iso.symm_inv, Functor.mapIso_hom, IsLimit.uniqueUpToIso_hom, lim_obj,
          lim_map, Iso.trans_hom, Iso.symm_hom, AddCommGrp.HasLimit.lift, Functor.const_obj_obj,
          Category.assoc, limit.lift_map_assoc, Pi.cone_pt, iX, iY]
        ext g j
        change _ = (_ ≫ limit.π (Discrete.functor fun j ↦ Y.obj { as := j }) ⟨j⟩) _
        simp only [Discrete.functor_obj_eq_as, Functor.comp_obj, Discrete.mk_as, productIsProduct',
          limit.lift_π, Fan.mk_pt, Fan.mk_π_app, Pi.map_apply]
        change _ = (_ ≫ _ ≫ limit.π Y ⟨j⟩) _
        simp
      suffices Epi (iX.hom ≫ (iX.inv ≫ lim.map f ≫ iY.hom) ≫ iY.inv) by simpa using this
      suffices Epi (iX.inv ≫ lim.map f ≫ iY.hom) from inferInstance
      rw [AddCommGrp.epi_iff_surjective, ← this]
      simp_rw [CategoryTheory.NatTrans.epi_iff_epi_app, AddCommGrp.epi_iff_surjective] at hf
      refine fun b ↦ ⟨fun i ↦ (hf ⟨i⟩ (b i)).choose, ?_⟩
      funext i
      exact (hf ⟨i⟩ (b i)).choose_spec }


instance : AB4Star AddCommGrp.{u} where
  ofShape _ := inferInstance

