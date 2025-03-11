/--
A condensed object is *discrete* if it is constant as a sheaf, i.e. isomorphic to a constant sheaf.
-/
abbrev IsDiscrete (X : Condensed.{u} C) := X.IsConstant (coherentTopology CompHaus)


lemma mem_locallyConstant_essImage_of_isColimit_mapCocone (X : CondensedSet.{u})
    (h : ∀ S : Profinite.{u}, IsColimit <|
      (profiniteToCompHaus.op ⋙ X.val).mapCocone S.asLimitCone.op) :
    X ∈ CondensedSet.LocallyConstant.functor.essImage := by
  let e : CondensedSet.{u} ≌ Sheaf (coherentTopology Profinite) _ :=
    (Condensed.ProfiniteCompHaus.equivalence (Type (u + 1))).symm
  let i : (e.functor.obj X).val ≅ (e.functor.obj (LocallyConstant.functor.obj _)).val :=
    Condensed.isoLocallyConstantOfIsColimit _ h
  /-
    X : CondensedSet
    h : (S : Profinite) → CategoryTheory.Limits.IsColimit ((profiniteToCompHaus.op …
    e : CategoryTheory.Equivalence CondensedSet (CategoryTheory.Sheaf (CategoryThe …
    i : CategoryTheory.Iso (e.functor.obj X).val (e.functor.obj (CondensedSet.Loca …
    ⊢ Membership.mem CondensedSet.LocallyConstant.functor.essImage X
  -/
  exact ⟨_, ⟨e.functor.preimageIso ((sheafToPresheaf _ _).preimageIso i.symm)⟩⟩
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-25")]
alias mem_locallyContant_essImage_of_isColimit_mapCocone :=
  mem_locallyConstant_essImage_of_isColimit_mapCocone


/--
`CondensedSet.LocallyConstant.functor` is left adjoint to the forgetful functor from condensed
sets to sets.
-/
noncomputable abbrev LocallyConstant.adjunction :
    CondensedSet.LocallyConstant.functor ⊣ Condensed.underlying (Type (u+1)) :=
  CompHausLike.LocallyConstant.adjunction _ _


open CondensedSet.LocallyConstant List in
theorem isDiscrete_tfae  (X : CondensedSet.{u}) :
    TFAE
    [ X.IsDiscrete
    , IsIso ((Condensed.discreteUnderlyingAdj _).counit.app X)
    , X ∈ (Condensed.discrete _).essImage
    , X ∈ CondensedSet.LocallyConstant.functor.essImage
    , IsIso (CondensedSet.LocallyConstant.adjunction.counit.app X)
    , Sheaf.IsConstant (coherentTopology Profinite)
        ((Condensed.ProfiniteCompHaus.equivalence _).inverse.obj X)
    , ∀ S : Profinite.{u}, Nonempty
        (IsColimit <| (profiniteToCompHaus.op ⋙ X.val).mapCocone S.asLimitCone.op)
    ] := by
  /-
    X : CondensedSet
    ⊢ (List.cons (Condensed.IsDiscrete X) (List.cons (CategoryTheory.IsIso ((Conde …
  -/
  tfae_have 1 ↔ 2 := Sheaf.isConstant_iff_isIso_counit_app _ _ _
  /-
    X : CondensedSet
    tfae_1_iff_2 : Iff (Condensed.IsDiscrete X) (CategoryTheory.IsIso ((Condensed. …
    ⊢ (List.cons (Condensed.IsDiscrete X) (List.cons (CategoryTheory.IsIso ((Conde …
  -/
  tfae_have 1 ↔ 3 := ⟨fun ⟨h⟩ ↦ h, fun h ↦ ⟨h⟩⟩
  /-
    X : CondensedSet
    tfae_1_iff_2 : Iff (Condensed.IsDiscrete X) (CategoryTheory.IsIso ((Condensed. …
    tfae_1_iff_3 : Iff (Condensed.IsDiscrete X) (Membership.mem (Condensed.discret …
    ⊢ (List.cons (Condensed.IsDiscrete X) (List.cons (CategoryTheory.IsIso ((Conde …
  -/
  tfae_have 1 ↔ 4 := Sheaf.isConstant_iff_mem_essImage _ CompHaus.isTerminalPUnit adjunction _
  tfae_have 1 ↔ 5 :=
    have : functor.Faithful := inferInstance
    have : functor.Full := inferInstance
    -- These `have` statements above shouldn't be needed, but they are.
    Sheaf.isConstant_iff_isIso_counit_app' _ CompHaus.isTerminalPUnit adjunction _
  tfae_have 1 ↔ 6 :=
    (Sheaf.isConstant_iff_of_equivalence (coherentTopology Profinite)
      (coherentTopology CompHaus) profiniteToCompHaus Profinite.isTerminalPUnit
      CompHaus.isTerminalPUnit _).symm
  tfae_have 7 → 4 := fun h ↦
    mem_locallyConstant_essImage_of_isColimit_mapCocone X (fun S ↦ (h S).some)
  tfae_have 4 → 7 := fun ⟨Y, ⟨i⟩⟩ S ↦
    ⟨IsColimit.mapCoconeEquiv (isoWhiskerLeft profiniteToCompHaus.op
      ((sheafToPresheaf _ _).mapIso i))
      (Condensed.isColimitLocallyConstantPresheafDiagram Y S)⟩
  /-
    X : CondensedSet
    tfae_1_iff_2 : Iff (Condensed.IsDiscrete X) (CategoryTheory.IsIso ((Condensed. …
    tfae_1_iff_3 : Iff (Condensed.IsDiscrete X) (Membership.mem (Condensed.discret …
    tfae_1_iff_4 : Iff (Condensed.IsDiscrete X) (Membership.mem CondensedSet.Local …
    tfae_1_iff_5 : Iff (Condensed.IsDiscrete X) (CategoryTheory.IsIso (CondensedSe …
    tfae_1_iff_6 : Iff (Condensed.IsDiscrete X) (CategoryTheory.Sheaf.IsConstant ( …
    tfae_7_to_4 : (∀ (S : Profinite), Nonempty (CategoryTheory.Limits.IsColimit (( …
    tfae_4_to_7 : Membership.mem CondensedSet.LocallyConstant.functor.essImage X → …
    ⊢ (List.cons (Condensed.IsDiscrete X) (List.cons (CategoryTheory.IsIso ((Conde …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


lemma isDiscrete_iff_isDiscrete_forget (M : CondensedMod R) :
    M.IsDiscrete ↔ ((Condensed.forget R).obj M).IsDiscrete  :=
  Sheaf.isConstant_iff_forget (coherentTopology CompHaus)
    (forget (ModuleCat R)) M CompHaus.isTerminalPUnit


instance : HasLimitsOfSize.{u, u+1} (ModuleCat.{u+1} R) :=
  hasLimitsOfSizeShrink.{u, u+1, u+1, u+1} _


open CondensedMod.LocallyConstant List in
theorem isDiscrete_tfae  (M : CondensedMod.{u} R) :
    TFAE
    [ M.IsDiscrete
    , IsIso ((Condensed.discreteUnderlyingAdj _).counit.app M)
    , M ∈ (Condensed.discrete _).essImage
    , M ∈ (CondensedMod.LocallyConstant.functor R).essImage
    , IsIso ((CondensedMod.LocallyConstant.adjunction R).counit.app M)
    , Sheaf.IsConstant (coherentTopology Profinite)
        ((Condensed.ProfiniteCompHaus.equivalence _).inverse.obj M)
    , ∀ S : Profinite.{u}, Nonempty
        (IsColimit <| (profiniteToCompHaus.op ⋙ M.val).mapCocone S.asLimitCone.op)
    ] := by
  /-
    R : Type (u + 1)
    inst✝ : Ring R
    M : CondensedMod R
    ⊢ (List.cons (Condensed.IsDiscrete M) (List.cons (CategoryTheory.IsIso ((Conde …
  -/
  tfae_have 1 ↔ 2 := Sheaf.isConstant_iff_isIso_counit_app _ _ _
  /-
    R : Type (u + 1)
    inst✝ : Ring R
    M : CondensedMod R
    tfae_1_iff_2 : Iff (Condensed.IsDiscrete M) (CategoryTheory.IsIso ((Condensed. …
    ⊢ (List.cons (Condensed.IsDiscrete M) (List.cons (CategoryTheory.IsIso ((Conde …
  -/
  tfae_have 1 ↔ 3 := ⟨fun ⟨h⟩ ↦ h, fun h ↦ ⟨h⟩⟩
  /-
    R : Type (u + 1)
    inst✝ : Ring R
    M : CondensedMod R
    tfae_1_iff_2 : Iff (Condensed.IsDiscrete M) (CategoryTheory.IsIso ((Condensed. …
    tfae_1_iff_3 : Iff (Condensed.IsDiscrete M) (Membership.mem (Condensed.discret …
    ⊢ (List.cons (Condensed.IsDiscrete M) (List.cons (CategoryTheory.IsIso ((Conde …
  -/
  tfae_have 1 ↔ 4 := Sheaf.isConstant_iff_mem_essImage _ CompHaus.isTerminalPUnit (adjunction R) _
  tfae_have 1 ↔ 5 :=
    have : (functor R).Faithful := inferInstance
    have : (functor R).Full := inferInstance
    -- These `have` statements above shouldn't be needed, but they are.
    Sheaf.isConstant_iff_isIso_counit_app' _ CompHaus.isTerminalPUnit (adjunction R) _
  tfae_have 1 ↔ 6 :=
    (Sheaf.isConstant_iff_of_equivalence (coherentTopology Profinite)
      (coherentTopology CompHaus) profiniteToCompHaus Profinite.isTerminalPUnit
      CompHaus.isTerminalPUnit _).symm
  tfae_have 7 → 1 := by
    intro h
    rw [isDiscrete_iff_isDiscrete_forget, ((CondensedSet.isDiscrete_tfae _).out 0 6:)]
    intro S
    letI : PreservesFilteredColimitsOfSize.{u, u} (forget (ModuleCat R)) :=
      preservesFilteredColimitsOfSize_shrink.{u, u+1, u, u+1} _
    exact ⟨isColimitOfPreserves (forget (ModuleCat R)) (h S).some⟩
  tfae_have 1 → 7 := by
    intro h S
    rw [isDiscrete_iff_isDiscrete_forget, ((CondensedSet.isDiscrete_tfae _).out 0 6:)] at h
    letI : ReflectsFilteredColimitsOfSize.{u, u} (forget (ModuleCat R)) :=
      reflectsFilteredColimitsOfSize_shrink.{u, u+1, u, u+1} _
    exact ⟨isColimitOfReflects (forget (ModuleCat R)) (h S).some⟩
  /-
    R : Type (u + 1)
    inst✝ : Ring R
    M : CondensedMod R
    tfae_1_iff_2 : Iff (Condensed.IsDiscrete M) (CategoryTheory.IsIso ((Condensed. …
    tfae_1_iff_3 : Iff (Condensed.IsDiscrete M) (Membership.mem (Condensed.discret …
    tfae_1_iff_4 : Iff (Condensed.IsDiscrete M) (Membership.mem (CondensedMod.Loca …
    tfae_1_iff_5 : Iff (Condensed.IsDiscrete M) (CategoryTheory.IsIso ((CondensedM …
    tfae_1_iff_6 : Iff (Condensed.IsDiscrete M) (CategoryTheory.Sheaf.IsConstant ( …
    tfae_7_to_1 : (∀ (S : Profinite), Nonempty (CategoryTheory.Limits.IsColimit (( …
    tfae_1_to_7 : Condensed.IsDiscrete M → ∀ (S : Profinite), Nonempty (CategoryTh …
    ⊢ (List.cons (Condensed.IsDiscrete M) (List.cons (CategoryTheory.IsIso ((Conde …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


/--
A light condensed object is *discrete* if it is constant as a sheaf, i.e. isomorphic to a constant
sheaf.
-/
abbrev IsDiscrete (X : LightCondensed.{u} C) := X.IsConstant (coherentTopology LightProfinite)


lemma mem_locallyConstant_essImage_of_isColimit_mapCocone (X : LightCondSet.{u})
    (h : ∀ S : LightProfinite.{u}, IsColimit <|
      X.val.mapCocone (coconeRightOpOfCone S.asLimitCone)) :
    X ∈ LightCondSet.LocallyConstant.functor.essImage := by
  let i : X.val ≅ (LightCondSet.LocallyConstant.functor.obj _).val :=
    LightCondensed.isoLocallyConstantOfIsColimit _ h
  /-
    X : LightCondSet
    h : (S : LightProfinite) → CategoryTheory.Limits.IsColimit (X.val.mapCocone (C …
    i : CategoryTheory.Iso X.val (LightCondSet.LocallyConstant.functor.obj (X.val. …
    ⊢ Membership.mem LightCondSet.LocallyConstant.functor.essImage X
  -/
  exact ⟨_, ⟨((sheafToPresheaf _ _).preimageIso i.symm)⟩⟩
  /-
    🎉 no goals
  -/


/--
`LightCondSet.LocallyConstant.functor` is left adjoint to the forgetful functor from light condensed
sets to sets.
-/
noncomputable abbrev LocallyConstant.adjunction :
    LightCondSet.LocallyConstant.functor ⊣ LightCondensed.underlying (Type u) :=
  CompHausLike.LocallyConstant.adjunction _ _


open LightCondSet.LocallyConstant List in
theorem isDiscrete_tfae  (X : LightCondSet.{u}) :
    TFAE
    [ X.IsDiscrete
    , IsIso ((LightCondensed.discreteUnderlyingAdj _).counit.app X)
    , X ∈ (LightCondensed.discrete _).essImage
    , X ∈ LightCondSet.LocallyConstant.functor.essImage
    , IsIso (LightCondSet.LocallyConstant.adjunction.counit.app X)
    , ∀ S : LightProfinite.{u}, Nonempty
        (IsColimit <| X.val.mapCocone (coconeRightOpOfCone S.asLimitCone))
    ] := by
  /-
    X : LightCondSet
    ⊢ (List.cons (LightCondensed.IsDiscrete X) (List.cons (CategoryTheory.IsIso (( …
  -/
  tfae_have 1 ↔ 2 := Sheaf.isConstant_iff_isIso_counit_app _ _ _
  /-
    X : LightCondSet
    tfae_1_iff_2 : Iff (LightCondensed.IsDiscrete X) (CategoryTheory.IsIso ((Light …
    ⊢ (List.cons (LightCondensed.IsDiscrete X) (List.cons (CategoryTheory.IsIso (( …
  -/
  tfae_have 1 ↔ 3 := ⟨fun ⟨h⟩ ↦ h, fun h ↦ ⟨h⟩⟩
  /-
    X : LightCondSet
    tfae_1_iff_2 : Iff (LightCondensed.IsDiscrete X) (CategoryTheory.IsIso ((Light …
    tfae_1_iff_3 : Iff (LightCondensed.IsDiscrete X) (Membership.mem (LightCondens …
    ⊢ (List.cons (LightCondensed.IsDiscrete X) (List.cons (CategoryTheory.IsIso (( …
  -/
  tfae_have 1 ↔ 4 := Sheaf.isConstant_iff_mem_essImage _ LightProfinite.isTerminalPUnit adjunction X
  tfae_have 1 ↔ 5 :=
    have : functor.Faithful := inferInstance
    have : functor.Full := inferInstance
    -- These `have` statements above shouldn't be needed, but they are.
    Sheaf.isConstant_iff_isIso_counit_app' _ LightProfinite.isTerminalPUnit adjunction X
  tfae_have 6 → 4 := fun h ↦
    mem_locallyConstant_essImage_of_isColimit_mapCocone X (fun S ↦ (h S).some)
  tfae_have 4 → 6 := fun ⟨Y, ⟨i⟩⟩ S ↦
    ⟨IsColimit.mapCoconeEquiv ((sheafToPresheaf _ _).mapIso i)
      (LightCondensed.isColimitLocallyConstantPresheafDiagram Y S)⟩
  /-
    X : LightCondSet
    tfae_1_iff_2 : Iff (LightCondensed.IsDiscrete X) (CategoryTheory.IsIso ((Light …
    tfae_1_iff_3 : Iff (LightCondensed.IsDiscrete X) (Membership.mem (LightCondens …
    tfae_1_iff_4 : Iff (LightCondensed.IsDiscrete X) (Membership.mem LightCondSet. …
    tfae_1_iff_5 : Iff (LightCondensed.IsDiscrete X) (CategoryTheory.IsIso (LightC …
    tfae_6_to_4 : (∀ (S : LightProfinite), Nonempty (CategoryTheory.Limits.IsColim …
    tfae_4_to_6 : Membership.mem LightCondSet.LocallyConstant.functor.essImage X → …
    ⊢ (List.cons (LightCondensed.IsDiscrete X) (List.cons (CategoryTheory.IsIso (( …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


lemma isDiscrete_iff_isDiscrete_forget (M : LightCondMod R) :
    M.IsDiscrete ↔ ((LightCondensed.forget R).obj M).IsDiscrete  :=
  Sheaf.isConstant_iff_forget (coherentTopology LightProfinite)
    (forget (ModuleCat R)) M LightProfinite.isTerminalPUnit


open LightCondMod.LocallyConstant List in
theorem isDiscrete_tfae  (M : LightCondMod.{u} R) :
    TFAE
    [ M.IsDiscrete
    , IsIso ((LightCondensed.discreteUnderlyingAdj _).counit.app M)
    , M ∈ (LightCondensed.discrete _).essImage
    , M ∈ (LightCondMod.LocallyConstant.functor R).essImage
    , IsIso ((LightCondMod.LocallyConstant.adjunction R).counit.app M)
    , ∀ S : LightProfinite.{u}, Nonempty
        (IsColimit <| M.val.mapCocone (coconeRightOpOfCone S.asLimitCone))
    ] := by
  /-
    R : Type u
    inst✝ : Ring R
    M : LightCondMod R
    ⊢ (List.cons (LightCondensed.IsDiscrete M) (List.cons (CategoryTheory.IsIso (( …
  -/
  tfae_have 1 ↔ 2 := Sheaf.isConstant_iff_isIso_counit_app _ _ _
  /-
    R : Type u
    inst✝ : Ring R
    M : LightCondMod R
    tfae_1_iff_2 : Iff (LightCondensed.IsDiscrete M) (CategoryTheory.IsIso ((Light …
    ⊢ (List.cons (LightCondensed.IsDiscrete M) (List.cons (CategoryTheory.IsIso (( …
  -/
  tfae_have 1 ↔ 3 := ⟨fun ⟨h⟩ ↦ h, fun h ↦ ⟨h⟩⟩
  tfae_have 1 ↔ 4 := Sheaf.isConstant_iff_mem_essImage _
    LightProfinite.isTerminalPUnit (adjunction R) _
  tfae_have 1 ↔ 5 :=
    have : (functor R).Faithful := inferInstance
    have : (functor R).Full := inferInstance
    -- These `have` statements above shouldn't be needed, but they are.
    Sheaf.isConstant_iff_isIso_counit_app' _ LightProfinite.isTerminalPUnit (adjunction R) _
  tfae_have 6 → 1 := by
    intro h
    rw [isDiscrete_iff_isDiscrete_forget, ((LightCondSet.isDiscrete_tfae _).out 0 5:)]
    intro S
    letI : PreservesFilteredColimitsOfSize.{0, 0} (forget (ModuleCat R)) :=
      preservesFilteredColimitsOfSize_shrink.{0, u, 0, u} _
    exact ⟨isColimitOfPreserves (forget (ModuleCat R)) (h S).some⟩
  tfae_have 1 → 6 := by
    intro h S
    rw [isDiscrete_iff_isDiscrete_forget, ((LightCondSet.isDiscrete_tfae _).out 0 5:)] at h
    letI : ReflectsFilteredColimitsOfSize.{0, 0} (forget (ModuleCat R)) :=
      reflectsFilteredColimitsOfSize_shrink.{0, u, 0, u} _
    exact ⟨isColimitOfReflects (forget (ModuleCat R)) (h S).some⟩
  /-
    R : Type u
    inst✝ : Ring R
    M : LightCondMod R
    tfae_1_iff_2 : Iff (LightCondensed.IsDiscrete M) (CategoryTheory.IsIso ((Light …
    tfae_1_iff_3 : Iff (LightCondensed.IsDiscrete M) (Membership.mem (LightCondens …
    tfae_1_iff_4 : Iff (LightCondensed.IsDiscrete M) (Membership.mem (LightCondMod …
    tfae_1_iff_5 : Iff (LightCondensed.IsDiscrete M) (CategoryTheory.IsIso ((Light …
    tfae_6_to_1 : (∀ (S : LightProfinite), Nonempty (CategoryTheory.Limits.IsColim …
    tfae_1_to_6 : LightCondensed.IsDiscrete M → ∀ (S : LightProfinite), Nonempty ( …
    ⊢ (List.cons (LightCondensed.IsDiscrete M) (List.cons (CategoryTheory.IsIso (( …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


