local notation "∞" => (⊤ : ℕ∞)


/-- The sheaf of smooth functions from `M` to `N`, as a sheaf of types. -/
def smoothSheaf : TopCat.Sheaf (Type u) (TopCat.of M) :=
  (contDiffWithinAt_localInvariantProp (I := IM) (I' := I) ⊤).sheaf M N


instance smoothSheaf.coeFun (U : (Opens (TopCat.of M))ᵒᵖ) :
    CoeFun ((smoothSheaf IM I M N).presheaf.obj U) (fun _ ↦ ↑(unop U) → N) :=
  (contDiffWithinAt_localInvariantProp ⊤).sheafHasCoeToFun _ _ _


open Manifold in
/-- The object of `smoothSheaf IM I M N` for the open set `U` in `M` is
`C^∞⟮IM, (unop U : Opens M); I, N⟯`, the `(IM, I)`-smooth functions from `U` to `N`.  This is not
just a "moral" equality but a literal and definitional equality! -/
lemma smoothSheaf.obj_eq (U : (Opens (TopCat.of M))ᵒᵖ) :
    (smoothSheaf IM I M N).presheaf.obj U = C^∞⟮IM, (unop U : Opens M); I, N⟯ := rfl


/-- Canonical map from the stalk of `smoothSheaf IM I M N` at `x` to `N`, given by evaluating
sections at `x`. -/
def smoothSheaf.eval (x : M) : (smoothSheaf IM I M N).presheaf.stalk x → N :=
  TopCat.stalkToFiber (StructureGroupoid.LocalInvariantProp.localPredicate M N _) x


/-- Canonical map from the stalk of `smoothSheaf IM I M N` at `x` to `N`, given by evaluating
sections at `x`, considered as a morphism in the category of types. -/
def smoothSheaf.evalHom (x : TopCat.of M) : (smoothSheaf IM I M N).presheaf.stalk x ⟶ N :=
  TopCat.stalkToFiber (StructureGroupoid.LocalInvariantProp.localPredicate M N _) x


/-- Given manifolds `M`, `N` and an open neighbourhood `U` of a point `x : M`, the evaluation-at-`x`
map to `N` from smooth functions from  `U` to `N`. -/
def smoothSheaf.evalAt (x : TopCat.of M) (U : OpenNhds x)
    (i : (smoothSheaf IM I M N).presheaf.obj (Opposite.op U.obj)) : N :=
  i.1 ⟨x, U.2⟩


@[simp, reassoc, elementwise] lemma smoothSheaf.ι_evalHom (x : TopCat.of M) (U) :
    colimit.ι ((OpenNhds.inclusion x).op ⋙ (smoothSheaf IM I M N).val) U ≫
    smoothSheaf.evalHom IM I N x =
    smoothSheaf.evalAt _ _ _ _ _ :=
  colimit.ι_desc _ _


/-- The `eval` map is surjective at `x`. -/
lemma smoothSheaf.eval_surjective (x : M) : Function.Surjective (smoothSheaf.eval IM I N x) := by
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    EM : Type u_2
    inst✝⁹ : NormedAddCommGroup EM
    inst✝⁸ : NormedSpace 𝕜 EM
    HM : Type u_3
    inst✝⁷ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    E : Type u_4
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    H : Type u_5
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace HM M
    N : Type u
    inst✝¹ : TopologicalSpace N
    inst✝ : ChartedSpace H N
    x : M
    ⊢ Function.Surjective (smoothSheaf.eval IM I N x)
  -/
  apply TopCat.stalkToFiber_surjective
  /-
    case w
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    EM : Type u_2
    inst✝⁹ : NormedAddCommGroup EM
    inst✝⁸ : NormedSpace 𝕜 EM
    HM : Type u_3
    inst✝⁷ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    E : Type u_4
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    H : Type u_5
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace HM M
    N : Type u
    inst✝¹ : TopologicalSpace N
    inst✝ : ChartedSpace H N
    x : M
    ⊢ ∀ (t : N), Exists fun U => Exists fun f => Exists fun x_1 => Eq (f ⟨x, ⋯⟩) t
  -/
  intro n
  /-
    case w
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    EM : Type u_2
    inst✝⁹ : NormedAddCommGroup EM
    inst✝⁸ : NormedSpace 𝕜 EM
    HM : Type u_3
    inst✝⁷ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    E : Type u_4
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    H : Type u_5
    inst✝⁴ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace HM M
    N : Type u
    inst✝¹ : TopologicalSpace N
    inst✝ : ChartedSpace H N
    x : M
    n : N
    ⊢ Exists fun U => Exists fun f => Exists fun x_1 => Eq (f ⟨x, ⋯⟩) n
  -/
  exact ⟨⊤, fun _ ↦ n, contMDiff_const, rfl⟩
  /-
    🎉 no goals
  -/


instance [Nontrivial N] (x : M) : Nontrivial ((smoothSheaf IM I M N).presheaf.stalk x) :=
  (smoothSheaf.eval_surjective IM I N x).nontrivial


@[simp] lemma smoothSheaf.eval_germ (U : Opens M) (x : M) (hx : x ∈ U)
    (f : (smoothSheaf IM I M N).presheaf.obj (op U)) :
    smoothSheaf.eval IM I N (x : M) ((smoothSheaf IM I M N).presheaf.germ U x hx f) = f ⟨x, hx⟩ :=
  TopCat.stalkToFiber_germ ((contDiffWithinAt_localInvariantProp ⊤).localPredicate M N) _ _ _ _


lemma smoothSheaf.contMDiff_section {U : (Opens (TopCat.of M))ᵒᵖ}
    (f : (smoothSheaf IM I M N).presheaf.obj U) :
    ContMDiff IM I ⊤ f :=
  (contDiffWithinAt_localInvariantProp ⊤).section_spec _ _ _ _


@[deprecated (since := "2024-11-21")]
alias smoothSheaf.smooth_section := smoothSheaf.contMDiff_section


open Manifold in
@[to_additive]
noncomputable instance (U : (Opens (TopCat.of M))ᵒᵖ) :
    Group ((smoothSheaf IM I M G).presheaf.obj U) :=
  (SmoothMap.group : Group C^∞⟮IM, (unop U : Opens M); I, G⟯)


/-- The presheaf of smooth functions from `M` to `G`, for `G` a Lie group, as a presheaf of groups.
-/
@[to_additive "The presheaf of smooth functions from `M` to `G`, for `G` an additive Lie group, as a
presheaf of additive groups."]
noncomputable def smoothPresheafGroup : TopCat.Presheaf Grp.{u} (TopCat.of M) :=
  { obj := fun U ↦ Grp.of ((smoothSheaf IM I M G).presheaf.obj U)
    map := fun h ↦ Grp.ofHom <|
      SmoothMap.restrictMonoidHom IM I G <| CategoryTheory.leOfHom h.unop
    map_id := fun _ ↦ rfl
    map_comp := fun _ _ ↦ rfl }


/-- The sheaf of smooth functions from `M` to `G`, for `G` a Lie group, as a sheaf of
groups. -/
@[to_additive "The sheaf of smooth functions from `M` to `G`, for `G` an additive Lie group, as a
sheaf of additive groups."]
noncomputable def smoothSheafGroup : TopCat.Sheaf Grp.{u} (TopCat.of M) :=
  { val := smoothPresheafGroup IM I M G
    cond := by
      /-
        𝕜 : Type u_1
        inst✝²¹ : NontriviallyNormedField 𝕜
        EM : Type u_2
        inst✝²⁰ : NormedAddCommGroup EM
        inst✝¹⁹ : NormedSpace 𝕜 EM
        HM : Type u_3
        inst✝¹⁸ : TopologicalSpace HM
        IM : ModelWithCorners 𝕜 EM HM
        E : Type u_4
        inst✝¹⁷ : NormedAddCommGroup E
        inst✝¹⁶ : NormedSpace 𝕜 E
        H : Type u_5
        inst✝¹⁵ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        H' : Type u_6
        inst✝¹⁴ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E H'
        M : Type u
        inst✝¹³ : TopologicalSpace M
        inst✝¹² : ChartedSpace HM M
        N G A A' R : Type u
        inst✝¹¹ : TopologicalSpace N
        inst✝¹⁰ : ChartedSpace H N
        inst✝⁹ : TopologicalSpace G
        inst✝⁸ : ChartedSpace H G
        inst✝⁷ : TopologicalSpace A
        inst✝⁶ : ChartedSpace H A
        inst✝⁵ : TopologicalSpace A'
        inst✝⁴ : ChartedSpace H' A'
        inst✝³ : TopologicalSpace R
        inst✝² : ChartedSpace H R
        inst✝¹ : Group G
        inst✝ : LieGroup I G
        ⊢ CategoryTheory.Presheaf.IsSheaf (Opens.grothendieckTopology ↑(TopCat.of M))  …
      -/
      rw [CategoryTheory.Presheaf.isSheaf_iff_isSheaf_forget _ _ (CategoryTheory.forget Grp)]
      /-
        𝕜 : Type u_1
        inst✝²¹ : NontriviallyNormedField 𝕜
        EM : Type u_2
        inst✝²⁰ : NormedAddCommGroup EM
        inst✝¹⁹ : NormedSpace 𝕜 EM
        HM : Type u_3
        inst✝¹⁸ : TopologicalSpace HM
        IM : ModelWithCorners 𝕜 EM HM
        E : Type u_4
        inst✝¹⁷ : NormedAddCommGroup E
        inst✝¹⁶ : NormedSpace 𝕜 E
        H : Type u_5
        inst✝¹⁵ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        H' : Type u_6
        inst✝¹⁴ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E H'
        M : Type u
        inst✝¹³ : TopologicalSpace M
        inst✝¹² : ChartedSpace HM M
        N G A A' R : Type u
        inst✝¹¹ : TopologicalSpace N
        inst✝¹⁰ : ChartedSpace H N
        inst✝⁹ : TopologicalSpace G
        inst✝⁸ : ChartedSpace H G
        inst✝⁷ : TopologicalSpace A
        inst✝⁶ : ChartedSpace H A
        inst✝⁵ : TopologicalSpace A'
        inst✝⁴ : ChartedSpace H' A'
        inst✝³ : TopologicalSpace R
        inst✝² : ChartedSpace H R
        inst✝¹ : Group G
        inst✝ : LieGroup I G
        ⊢ CategoryTheory.Presheaf.IsSheaf (Opens.grothendieckTopology ↑(TopCat.of M))  …
      -/
      exact CategoryTheory.Sheaf.cond (smoothSheaf IM I M G) }
      /-
        🎉 no goals
      -/


open Manifold in
@[to_additive] noncomputable instance (U : (Opens (TopCat.of M))ᵒᵖ) :
    CommGroup ((smoothSheaf IM I M A).presheaf.obj U) :=
  (SmoothMap.commGroup : CommGroup C^∞⟮IM, (unop U : Opens M); I, A⟯)


/-- The presheaf of smooth functions from `M` to `A`, for `A` an abelian Lie group, as a
presheaf of abelian groups. -/
@[to_additive "The presheaf of smooth functions from `M` to `A`, for `A` an additive abelian Lie
group, as a presheaf of additive abelian groups."]
noncomputable def smoothPresheafCommGroup : TopCat.Presheaf CommGrp.{u} (TopCat.of M) :=
  { obj := fun U ↦ CommGrp.of ((smoothSheaf IM I M A).presheaf.obj U)
    map := fun h ↦ CommGrp.ofHom <|
      SmoothMap.restrictMonoidHom IM I A <| CategoryTheory.leOfHom h.unop
    map_id := fun _ ↦ rfl
    map_comp := fun _ _ ↦ rfl }


/-- The sheaf of smooth functions from `M` to `A`, for `A` an abelian Lie group, as a
sheaf of abelian groups. -/
@[to_additive "The sheaf of smooth functions from `M` to
`A`, for `A` an abelian additive Lie group, as a sheaf of abelian additive groups."]
noncomputable def smoothSheafCommGroup : TopCat.Sheaf CommGrp.{u} (TopCat.of M) :=
  { val := smoothPresheafCommGroup IM I M A
    cond := by
      rw [CategoryTheory.Presheaf.isSheaf_iff_isSheaf_forget _ _
        (CategoryTheory.forget CommGrp)]
      /-
        𝕜 : Type u_1
        inst✝²³ : NontriviallyNormedField 𝕜
        EM : Type u_2
        inst✝²² : NormedAddCommGroup EM
        inst✝²¹ : NormedSpace 𝕜 EM
        HM : Type u_3
        inst✝²⁰ : TopologicalSpace HM
        IM : ModelWithCorners 𝕜 EM HM
        E : Type u_4
        inst✝¹⁹ : NormedAddCommGroup E
        inst✝¹⁸ : NormedSpace 𝕜 E
        H : Type u_5
        inst✝¹⁷ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        H' : Type u_6
        inst✝¹⁶ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E H'
        M : Type u
        inst✝¹⁵ : TopologicalSpace M
        inst✝¹⁴ : ChartedSpace HM M
        N G A A' R : Type u
        inst✝¹³ : TopologicalSpace N
        inst✝¹² : ChartedSpace H N
        inst✝¹¹ : TopologicalSpace G
        inst✝¹⁰ : ChartedSpace H G
        inst✝⁹ : TopologicalSpace A
        inst✝⁸ : ChartedSpace H A
        inst✝⁷ : TopologicalSpace A'
        inst✝⁶ : ChartedSpace H' A'
        inst✝⁵ : TopologicalSpace R
        inst✝⁴ : ChartedSpace H R
        inst✝³ : CommGroup A
        inst✝² : CommGroup A'
        inst✝¹ : LieGroup I A
        inst✝ : LieGroup I' A'
        ⊢ CategoryTheory.Presheaf.IsSheaf (Opens.grothendieckTopology ↑(TopCat.of M))  …
      -/
      exact CategoryTheory.Sheaf.cond (smoothSheaf IM I M A) }
      /-
        🎉 no goals
      -/


/-- For a manifold `M` and a smooth homomorphism `φ` between abelian Lie groups `A`, `A'`, the
'left-composition-by-`φ`' morphism of sheaves from `smoothSheafCommGroup IM I M A` to
`smoothSheafCommGroup IM I' M A'`. -/
@[to_additive "For a manifold `M` and a smooth homomorphism `φ` between abelian additive Lie groups
`A`, `A'`, the 'left-composition-by-`φ`' morphism of sheaves from `smoothSheafAddCommGroup IM I M A`
to `smoothSheafAddCommGroup IM I' M A'`."]
def smoothSheafCommGroup.compLeft (φ : A →* A') (hφ : ContMDiff I I' ⊤ φ) :
    smoothSheafCommGroup IM I M A ⟶ smoothSheafCommGroup IM I' M A' :=
  CategoryTheory.Sheaf.Hom.mk <|
  { app := fun _ ↦ CommGrp.ofHom <| SmoothMap.compLeftMonoidHom _ _ φ hφ
    naturality := fun _ _ _ ↦ rfl }


open Manifold in
instance (U : (Opens (TopCat.of M))ᵒᵖ) : Ring ((smoothSheaf IM I M R).presheaf.obj U) :=
  (SmoothMap.ring : Ring C^∞⟮IM, (unop U : Opens M); I, R⟯)


/-- The presheaf of smooth functions from `M` to `R`, for `R` a smooth ring, as a presheaf
of rings. -/
def smoothPresheafRing : TopCat.Presheaf RingCat.{u} (TopCat.of M) :=
  { obj := fun U ↦ RingCat.of ((smoothSheaf IM I M R).presheaf.obj U)
    map := fun h ↦ RingCat.ofHom <|
      SmoothMap.restrictRingHom IM I R <| CategoryTheory.leOfHom h.unop
    map_id := fun _ ↦ rfl
    map_comp := fun _ _ ↦ rfl }


/-- The sheaf of smooth functions from `M` to `R`, for `R` a smooth ring, as a sheaf of
rings. -/
def smoothSheafRing : TopCat.Sheaf RingCat.{u} (TopCat.of M) :=
  { val := smoothPresheafRing IM I M R
    cond := by
      /-
        𝕜 : Type u_1
        inst✝²¹ : NontriviallyNormedField 𝕜
        EM : Type u_2
        inst✝²⁰ : NormedAddCommGroup EM
        inst✝¹⁹ : NormedSpace 𝕜 EM
        HM : Type u_3
        inst✝¹⁸ : TopologicalSpace HM
        IM : ModelWithCorners 𝕜 EM HM
        E : Type u_4
        inst✝¹⁷ : NormedAddCommGroup E
        inst✝¹⁶ : NormedSpace 𝕜 E
        H : Type u_5
        inst✝¹⁵ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        H' : Type u_6
        inst✝¹⁴ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E H'
        M : Type u
        inst✝¹³ : TopologicalSpace M
        inst✝¹² : ChartedSpace HM M
        N G A A' R : Type u
        inst✝¹¹ : TopologicalSpace N
        inst✝¹⁰ : ChartedSpace H N
        inst✝⁹ : TopologicalSpace G
        inst✝⁸ : ChartedSpace H G
        inst✝⁷ : TopologicalSpace A
        inst✝⁶ : ChartedSpace H A
        inst✝⁵ : TopologicalSpace A'
        inst✝⁴ : ChartedSpace H' A'
        inst✝³ : TopologicalSpace R
        inst✝² : ChartedSpace H R
        inst✝¹ : Ring R
        inst✝ : SmoothRing I R
        ⊢ CategoryTheory.Presheaf.IsSheaf (Opens.grothendieckTopology ↑(TopCat.of M))  …
      -/
      rw [CategoryTheory.Presheaf.isSheaf_iff_isSheaf_forget _ _ (CategoryTheory.forget RingCat)]
      /-
        𝕜 : Type u_1
        inst✝²¹ : NontriviallyNormedField 𝕜
        EM : Type u_2
        inst✝²⁰ : NormedAddCommGroup EM
        inst✝¹⁹ : NormedSpace 𝕜 EM
        HM : Type u_3
        inst✝¹⁸ : TopologicalSpace HM
        IM : ModelWithCorners 𝕜 EM HM
        E : Type u_4
        inst✝¹⁷ : NormedAddCommGroup E
        inst✝¹⁶ : NormedSpace 𝕜 E
        H : Type u_5
        inst✝¹⁵ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        H' : Type u_6
        inst✝¹⁴ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E H'
        M : Type u
        inst✝¹³ : TopologicalSpace M
        inst✝¹² : ChartedSpace HM M
        N G A A' R : Type u
        inst✝¹¹ : TopologicalSpace N
        inst✝¹⁰ : ChartedSpace H N
        inst✝⁹ : TopologicalSpace G
        inst✝⁸ : ChartedSpace H G
        inst✝⁷ : TopologicalSpace A
        inst✝⁶ : ChartedSpace H A
        inst✝⁵ : TopologicalSpace A'
        inst✝⁴ : ChartedSpace H' A'
        inst✝³ : TopologicalSpace R
        inst✝² : ChartedSpace H R
        inst✝¹ : Ring R
        inst✝ : SmoothRing I R
        ⊢ CategoryTheory.Presheaf.IsSheaf (Opens.grothendieckTopology ↑(TopCat.of M))  …
      -/
      exact CategoryTheory.Sheaf.cond (smoothSheaf IM I M R) }
      /-
        🎉 no goals
      -/


open Manifold in
instance (U : (Opens (TopCat.of M))ᵒᵖ) : CommRing ((smoothSheaf IM I M R).presheaf.obj U) :=
  (SmoothMap.commRing : CommRing C^∞⟮IM, (unop U : Opens M); I, R⟯)


/-- The presheaf of smooth functions from `M` to `R`, for `R` a smooth commutative ring, as a
presheaf of commutative rings. -/
def smoothPresheafCommRing : TopCat.Presheaf CommRingCat.{u} (TopCat.of M) :=
  { obj := fun U ↦ CommRingCat.of ((smoothSheaf IM I M R).presheaf.obj U)
    map := fun h ↦ CommRingCat.ofHom <|
      SmoothMap.restrictRingHom IM I R <| CategoryTheory.leOfHom h.unop
    map_id := fun _ ↦ rfl
    map_comp := fun _ _ ↦ rfl }


/-- The sheaf of smooth functions from `M` to `R`, for `R` a smooth commutative ring, as a sheaf of
commutative rings. -/
def smoothSheafCommRing : TopCat.Sheaf CommRingCat.{u} (TopCat.of M) :=
  { val := smoothPresheafCommRing IM I M R
    cond := by
      rw [CategoryTheory.Presheaf.isSheaf_iff_isSheaf_forget _ _
        (CategoryTheory.forget CommRingCat)]
      /-
        𝕜 : Type u_1
        inst✝²¹ : NontriviallyNormedField 𝕜
        EM : Type u_2
        inst✝²⁰ : NormedAddCommGroup EM
        inst✝¹⁹ : NormedSpace 𝕜 EM
        HM : Type u_3
        inst✝¹⁸ : TopologicalSpace HM
        IM : ModelWithCorners 𝕜 EM HM
        E : Type u_4
        inst✝¹⁷ : NormedAddCommGroup E
        inst✝¹⁶ : NormedSpace 𝕜 E
        H : Type u_5
        inst✝¹⁵ : TopologicalSpace H
        I : ModelWithCorners 𝕜 E H
        H' : Type u_6
        inst✝¹⁴ : TopologicalSpace H'
        I' : ModelWithCorners 𝕜 E H'
        M : Type u
        inst✝¹³ : TopologicalSpace M
        inst✝¹² : ChartedSpace HM M
        N G A A' R : Type u
        inst✝¹¹ : TopologicalSpace N
        inst✝¹⁰ : ChartedSpace H N
        inst✝⁹ : TopologicalSpace G
        inst✝⁸ : ChartedSpace H G
        inst✝⁷ : TopologicalSpace A
        inst✝⁶ : ChartedSpace H A
        inst✝⁵ : TopologicalSpace A'
        inst✝⁴ : ChartedSpace H' A'
        inst✝³ : TopologicalSpace R
        inst✝² : ChartedSpace H R
        inst✝¹ : CommRing R
        inst✝ : SmoothRing I R
        ⊢ CategoryTheory.Presheaf.IsSheaf (Opens.grothendieckTopology ↑(TopCat.of M))  …
      -/
      exact CategoryTheory.Sheaf.cond (smoothSheaf IM I M R) }
      /-
        🎉 no goals
      -/

-- sanity check: applying the `CommRingCat`-to-`TypeCat` forgetful functor to the sheaf-of-rings of
-- smooth functions gives the sheaf-of-types of smooth functions.

instance smoothSheafCommRing.coeFun (U : (Opens (TopCat.of M))ᵒᵖ) :
    CoeFun ((smoothSheafCommRing IM I M R).presheaf.obj U) (fun _ ↦ ↑(unop U) → R) :=
  (contDiffWithinAt_localInvariantProp ⊤).sheafHasCoeToFun _ _ _


/-- Identify the stalk at a point of the sheaf-of-commutative-rings of functions from `M` to `R`
(for `R` a smooth ring) with the stalk at that point of the corresponding sheaf of types. -/
def smoothSheafCommRing.forgetStalk (x : TopCat.of M) :
    (forget _).obj ((smoothSheafCommRing IM I M R).presheaf.stalk x) ≅
    (smoothSheaf IM I M R).presheaf.stalk x :=
  preservesColimitIso _ _


@[simp, reassoc, elementwise] lemma smoothSheafCommRing.ι_forgetStalk_hom (x : TopCat.of M) (U) :
    CategoryStruct.comp
      (Z := (smoothSheaf IM I M R).presheaf.stalk x)
      (DFunLike.coe
        (α := ((forget CommRingCat).obj ((smoothSheafCommRing IM I M R).presheaf.obj
          (op ((OpenNhds.inclusion x).obj U.unop)))))
        (colimit.ι ((OpenNhds.inclusion x).op ⋙ (smoothSheafCommRing IM I M R).presheaf) U).hom)
      (forgetStalk IM I M R x).hom =
    colimit.ι ((OpenNhds.inclusion x).op ⋙ (smoothSheaf IM I M R).presheaf) U :=
  ι_preservesColimitIso_hom (forget CommRingCat) _ _


@[simp, reassoc, elementwise] lemma smoothSheafCommRing.ι_forgetStalk_inv (x : TopCat.of M) (U) :
    colimit.ι ((OpenNhds.inclusion x).op ⋙ (smoothSheaf IM I M R).presheaf) U ≫
    (smoothSheafCommRing.forgetStalk IM I M R x).inv =
    (forget CommRingCat).map
      (colimit.ι ((OpenNhds.inclusion x).op ⋙ (smoothSheafCommRing IM I M R).presheaf) U) := by
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    EM : Type u_2
    inst✝¹¹ : NormedAddCommGroup EM
    inst✝¹⁰ : NormedSpace 𝕜 EM
    HM : Type u_3
    inst✝⁹ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    E : Type u_4
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    H : Type u_5
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace HM M
    R : Type u
    inst✝³ : TopologicalSpace R
    inst✝² : ChartedSpace H R
    inst✝¹ : CommRing R
    inst✝ : SmoothRing I R
    x : ↑(TopCat.of M)
    U : Opposite (TopologicalSpace.OpenNhds x)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι ((To …
  -/
  rw [Iso.comp_inv_eq, ← smoothSheafCommRing.ι_forgetStalk_hom, CommRingCat.forget_map]
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    EM : Type u_2
    inst✝¹¹ : NormedAddCommGroup EM
    inst✝¹⁰ : NormedSpace 𝕜 EM
    HM : Type u_3
    inst✝⁹ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    E : Type u_4
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    H : Type u_5
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace HM M
    R : Type u
    inst✝³ : TopologicalSpace R
    inst✝² : ChartedSpace H R
    inst✝¹ : CommRing R
    inst✝ : SmoothRing I R
    x : ↑(TopCat.of M)
    U : Opposite (TopologicalSpace.OpenNhds x)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (⇑(CategoryTheory.Limits.colimit.ι (( …
  -/
  simp_rw [Functor.comp_obj, Functor.op_obj]
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    EM : Type u_2
    inst✝¹¹ : NormedAddCommGroup EM
    inst✝¹⁰ : NormedSpace 𝕜 EM
    HM : Type u_3
    inst✝⁹ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    E : Type u_4
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    H : Type u_5
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace HM M
    R : Type u
    inst✝³ : TopologicalSpace R
    inst✝² : ChartedSpace H R
    inst✝¹ : CommRing R
    inst✝ : SmoothRing I R
    x : ↑(TopCat.of M)
    U : Opposite (TopologicalSpace.OpenNhds x)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (⇑(CategoryTheory.Limits.colimit.ι (( …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Given a smooth commutative ring `R` and a manifold `M`, and an open neighbourhood `U` of a point
`x : M`, the evaluation-at-`x` map to `R` from smooth functions from  `U` to `R`. -/
def smoothSheafCommRing.evalAt (x : TopCat.of M) (U : OpenNhds x) :
    (smoothSheafCommRing IM I M R).presheaf.obj (Opposite.op U.1) ⟶ CommRingCat.of R :=
  CommRingCat.ofHom (SmoothMap.evalRingHom ⟨x, U.2⟩)


/-- Canonical ring homomorphism from the stalk of `smoothSheafCommRing IM I M R` at `x` to `R`,
given by evaluating sections at `x`, considered as a morphism in the category of commutative rings.
-/
def smoothSheafCommRing.evalHom (x : TopCat.of M) :
    (smoothSheafCommRing IM I M R).presheaf.stalk x ⟶ CommRingCat.of R := by
  /-
    𝕜 : Type u_1
    inst✝²¹ : NontriviallyNormedField 𝕜
    EM : Type u_2
    inst✝²⁰ : NormedAddCommGroup EM
    inst✝¹⁹ : NormedSpace 𝕜 EM
    HM : Type u_3
    inst✝¹⁸ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    E : Type u_4
    inst✝¹⁷ : NormedAddCommGroup E
    inst✝¹⁶ : NormedSpace 𝕜 E
    H : Type u_5
    inst✝¹⁵ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    H' : Type u_6
    inst✝¹⁴ : TopologicalSpace H'
    I' : ModelWithCorners 𝕜 E H'
    M : Type u
    inst✝¹³ : TopologicalSpace M
    inst✝¹² : ChartedSpace HM M
    N G A A' R : Type u
    inst✝¹¹ : TopologicalSpace N
    inst✝¹⁰ : ChartedSpace H N
    inst✝⁹ : TopologicalSpace G
    inst✝⁸ : ChartedSpace H G
    inst✝⁷ : TopologicalSpace A
    inst✝⁶ : ChartedSpace H A
    inst✝⁵ : TopologicalSpace A'
    inst✝⁴ : ChartedSpace H' A'
    inst✝³ : TopologicalSpace R
    inst✝² : ChartedSpace H R
    inst✝¹ : CommRing R
    inst✝ : SmoothRing I R
    x : ↑(TopCat.of M)
    ⊢ Quiver.Hom ((smoothSheafCommRing IM I M R).presheaf.stalk x) (CommRingCat.of …
  -/
  refine CategoryTheory.Limits.colimit.desc _ ⟨_, ⟨fun U ↦ ?_, ?_⟩⟩
    /-
      case refine_1
      𝕜 : Type u_1
      inst✝²¹ : NontriviallyNormedField 𝕜
      EM : Type u_2
      inst✝²⁰ : NormedAddCommGroup EM
      inst✝¹⁹ : NormedSpace 𝕜 EM
      HM : Type u_3
      inst✝¹⁸ : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      E : Type u_4
      inst✝¹⁷ : NormedAddCommGroup E
      inst✝¹⁶ : NormedSpace 𝕜 E
      H : Type u_5
      inst✝¹⁵ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      H' : Type u_6
      inst✝¹⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E H'
      M : Type u
      inst✝¹³ : TopologicalSpace M
      inst✝¹² : ChartedSpace HM M
      N G A A' R : Type u
      inst✝¹¹ : TopologicalSpace N
      inst✝¹⁰ : ChartedSpace H N
      inst✝⁹ : TopologicalSpace G
      inst✝⁸ : ChartedSpace H G
      inst✝⁷ : TopologicalSpace A
      inst✝⁶ : ChartedSpace H A
      inst✝⁵ : TopologicalSpace A'
      inst✝⁴ : ChartedSpace H' A'
      inst✝³ : TopologicalSpace R
      inst✝² : ChartedSpace H R
      inst✝¹ : CommRing R
      inst✝ : SmoothRing I R
      x : ↑(TopCat.of M)
      U : Opposite (TopologicalSpace.OpenNhds x)
      ⊢ Quiver.Hom ((((CategoryTheory.whiskeringLeft (Opposite (TopologicalSpace.Ope …
    -/
  · apply smoothSheafCommRing.evalAt
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      𝕜 : Type u_1
      inst✝²¹ : NontriviallyNormedField 𝕜
      EM : Type u_2
      inst✝²⁰ : NormedAddCommGroup EM
      inst✝¹⁹ : NormedSpace 𝕜 EM
      HM : Type u_3
      inst✝¹⁸ : TopologicalSpace HM
      IM : ModelWithCorners 𝕜 EM HM
      E : Type u_4
      inst✝¹⁷ : NormedAddCommGroup E
      inst✝¹⁶ : NormedSpace 𝕜 E
      H : Type u_5
      inst✝¹⁵ : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      H' : Type u_6
      inst✝¹⁴ : TopologicalSpace H'
      I' : ModelWithCorners 𝕜 E H'
      M : Type u
      inst✝¹³ : TopologicalSpace M
      inst✝¹² : ChartedSpace HM M
      N G A A' R : Type u
      inst✝¹¹ : TopologicalSpace N
      inst✝¹⁰ : ChartedSpace H N
      inst✝⁹ : TopologicalSpace G
      inst✝⁸ : ChartedSpace H G
      inst✝⁷ : TopologicalSpace A
      inst✝⁶ : ChartedSpace H A
      inst✝⁵ : TopologicalSpace A'
      inst✝⁴ : ChartedSpace H' A'
      inst✝³ : TopologicalSpace R
      inst✝² : ChartedSpace H R
      inst✝¹ : CommRing R
      inst✝ : SmoothRing I R
      x : ↑(TopCat.of M)
      ⊢ ∀ ⦃X Y : Opposite (TopologicalSpace.OpenNhds x)⦄ (f : Quiver.Hom X Y), Eq (C …
    -/
  · aesop_cat
    /-
      🎉 no goals
    -/


/-- Canonical ring homomorphism from the stalk of `smoothSheafCommRing IM I M R` at `x` to `R`,
given by evaluating sections at `x`. -/
def smoothSheafCommRing.eval (x : M) : (smoothSheafCommRing IM I M R).presheaf.stalk x →+* R :=
  (smoothSheafCommRing.evalHom IM I M R x).hom


@[simp, reassoc, elementwise] lemma smoothSheafCommRing.ι_evalHom (x : TopCat.of M) (U) :
    colimit.ι ((OpenNhds.inclusion x).op ⋙ _) U ≫ smoothSheafCommRing.evalHom IM I M R x =
    smoothSheafCommRing.evalAt _ _ _ _ _ _ :=
  colimit.ι_desc _ _


@[simp] lemma smoothSheafCommRing.evalHom_germ (U : Opens (TopCat.of M)) (x : M) (hx : x ∈ U)
    (f : (smoothSheafCommRing IM I M R).presheaf.obj (op U)) :
    smoothSheafCommRing.evalHom IM I M R (x : TopCat.of M)
      ((smoothSheafCommRing IM I M R).presheaf.germ U x hx f)
    = f ⟨x, hx⟩ :=
  congr_arg (fun a ↦ a f) <| smoothSheafCommRing.ι_evalHom IM I M R x ⟨U, hx⟩


@[simp, reassoc, elementwise] lemma smoothSheafCommRing.forgetStalk_inv_comp_eval
    (x : TopCat.of M) :
    (smoothSheafCommRing.forgetStalk IM I M R x).inv ≫
     (DFunLike.coe (smoothSheafCommRing.evalHom IM I M R x).hom) =
    smoothSheaf.evalHom _ _ _ _ := by
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    EM : Type u_2
    inst✝¹¹ : NormedAddCommGroup EM
    inst✝¹⁰ : NormedSpace 𝕜 EM
    HM : Type u_3
    inst✝⁹ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    E : Type u_4
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    H : Type u_5
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace HM M
    R : Type u
    inst✝³ : TopologicalSpace R
    inst✝² : ChartedSpace H R
    inst✝¹ : CommRing R
    inst✝ : SmoothRing I R
    x : ↑(TopCat.of M)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (smoothSheafCommRing.forgetStalk IM I …
  -/
  apply Limits.colimit.hom_ext
  /-
    case w
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    EM : Type u_2
    inst✝¹¹ : NormedAddCommGroup EM
    inst✝¹⁰ : NormedSpace 𝕜 EM
    HM : Type u_3
    inst✝⁹ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    E : Type u_4
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    H : Type u_5
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace HM M
    R : Type u
    inst✝³ : TopologicalSpace R
    inst✝² : ChartedSpace H R
    inst✝¹ : CommRing R
    inst✝ : SmoothRing I R
    x : ↑(TopCat.of M)
    ⊢ ∀ (j : Opposite (TopologicalSpace.OpenNhds x)), Eq (CategoryTheory.CategoryS …
  -/
  intro U
  /-
    case w
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    EM : Type u_2
    inst✝¹¹ : NormedAddCommGroup EM
    inst✝¹⁰ : NormedSpace 𝕜 EM
    HM : Type u_3
    inst✝⁹ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    E : Type u_4
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    H : Type u_5
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace HM M
    R : Type u
    inst✝³ : TopologicalSpace R
    inst✝² : ChartedSpace H R
    inst✝¹ : CommRing R
    inst✝ : SmoothRing I R
    x : ↑(TopCat.of M)
    U : Opposite (TopologicalSpace.OpenNhds x)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι (((C …
  -/
  show (colimit.ι _ U) ≫ _ = colimit.ι ((OpenNhds.inclusion x).op ⋙ _) U ≫ _
  /-
    case w
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    EM : Type u_2
    inst✝¹¹ : NormedAddCommGroup EM
    inst✝¹⁰ : NormedSpace 𝕜 EM
    HM : Type u_3
    inst✝⁹ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    E : Type u_4
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    H : Type u_5
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace HM M
    R : Type u
    inst✝³ : TopologicalSpace R
    inst✝² : ChartedSpace H R
    inst✝¹ : CommRing R
    inst✝ : SmoothRing I R
    x : ↑(TopCat.of M)
    U : Opposite (TopologicalSpace.OpenNhds x)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι ((To …
  -/
  rw [smoothSheafCommRing.ι_forgetStalk_inv_assoc]
  /-
    case w
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    EM : Type u_2
    inst✝¹¹ : NormedAddCommGroup EM
    inst✝¹⁰ : NormedSpace 𝕜 EM
    HM : Type u_3
    inst✝⁹ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    E : Type u_4
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    H : Type u_5
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace HM M
    R : Type u
    inst✝³ : TopologicalSpace R
    inst✝² : ChartedSpace H R
    inst✝¹ : CommRing R
    inst✝ : SmoothRing I R
    x : ↑(TopCat.of M)
    U : Opposite (TopologicalSpace.OpenNhds x)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.forget CommRingCat). …
  -/
  convert congr_arg (fun i ↦ (forget CommRingCat).map i) (smoothSheafCommRing.ι_evalHom ..)
  /-
    case h.e'_3
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    EM : Type u_2
    inst✝¹¹ : NormedAddCommGroup EM
    inst✝¹⁰ : NormedSpace 𝕜 EM
    HM : Type u_3
    inst✝⁹ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    E : Type u_4
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    H : Type u_5
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace HM M
    R : Type u
    inst✝³ : TopologicalSpace R
    inst✝² : ChartedSpace H R
    inst✝¹ : CommRing R
    inst✝ : SmoothRing I R
    x : ↑(TopCat.of M)
    U : Opposite (TopologicalSpace.OpenNhds x)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.colimit.ι ((To …
  -/
  exact smoothSheaf.ι_evalHom IM I R x U
  /-
    🎉 no goals
  -/


@[simp, reassoc, elementwise] lemma smoothSheafCommRing.forgetStalk_hom_comp_evalHom
    (x : TopCat.of M) :
    (smoothSheafCommRing.forgetStalk IM I M R x).hom ≫ (smoothSheaf.evalHom IM I R x) =
    (forget _).map (smoothSheafCommRing.evalHom _ _ _ _ _) := by
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    EM : Type u_2
    inst✝¹¹ : NormedAddCommGroup EM
    inst✝¹⁰ : NormedSpace 𝕜 EM
    HM : Type u_3
    inst✝⁹ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    E : Type u_4
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    H : Type u_5
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace HM M
    R : Type u
    inst✝³ : TopologicalSpace R
    inst✝² : ChartedSpace H R
    inst✝¹ : CommRing R
    inst✝ : SmoothRing I R
    x : ↑(TopCat.of M)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (smoothSheafCommRing.forgetStalk IM I …
  -/
  simp_rw [← CategoryTheory.Iso.eq_inv_comp]
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    EM : Type u_2
    inst✝¹¹ : NormedAddCommGroup EM
    inst✝¹⁰ : NormedSpace 𝕜 EM
    HM : Type u_3
    inst✝⁹ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    E : Type u_4
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    H : Type u_5
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace HM M
    R : Type u
    inst✝³ : TopologicalSpace R
    inst✝² : ChartedSpace H R
    inst✝¹ : CommRing R
    inst✝ : SmoothRing I R
    x : ↑(TopCat.of M)
    ⊢ Eq (smoothSheaf.evalHom IM I R x) (CategoryTheory.CategoryStruct.comp (smoot …
  -/
  rw [← smoothSheafCommRing.forgetStalk_inv_comp_eval]
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    EM : Type u_2
    inst✝¹¹ : NormedAddCommGroup EM
    inst✝¹⁰ : NormedSpace 𝕜 EM
    HM : Type u_3
    inst✝⁹ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    E : Type u_4
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    H : Type u_5
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace HM M
    R : Type u
    inst✝³ : TopologicalSpace R
    inst✝² : ChartedSpace H R
    inst✝¹ : CommRing R
    inst✝ : SmoothRing I R
    x : ↑(TopCat.of M)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (smoothSheafCommRing.forgetStalk IM I …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma smoothSheafCommRing.eval_surjective (x) :
    Function.Surjective (smoothSheafCommRing.eval IM I M R x) := by
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    EM : Type u_2
    inst✝¹¹ : NormedAddCommGroup EM
    inst✝¹⁰ : NormedSpace 𝕜 EM
    HM : Type u_3
    inst✝⁹ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    E : Type u_4
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    H : Type u_5
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace HM M
    R : Type u
    inst✝³ : TopologicalSpace R
    inst✝² : ChartedSpace H R
    inst✝¹ : CommRing R
    inst✝ : SmoothRing I R
    x : M
    ⊢ Function.Surjective ⇑(smoothSheafCommRing.eval IM I M R x)
  -/
  intro r
  /-
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    EM : Type u_2
    inst✝¹¹ : NormedAddCommGroup EM
    inst✝¹⁰ : NormedSpace 𝕜 EM
    HM : Type u_3
    inst✝⁹ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    E : Type u_4
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    H : Type u_5
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace HM M
    R : Type u
    inst✝³ : TopologicalSpace R
    inst✝² : ChartedSpace H R
    inst✝¹ : CommRing R
    inst✝ : SmoothRing I R
    x : M
    r : R
    ⊢ Exists fun a => Eq ((smoothSheafCommRing.eval IM I M R x) a) r
  -/
  obtain ⟨y, rfl⟩ := smoothSheaf.eval_surjective IM I R x r
  /-
    case intro
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    EM : Type u_2
    inst✝¹¹ : NormedAddCommGroup EM
    inst✝¹⁰ : NormedSpace 𝕜 EM
    HM : Type u_3
    inst✝⁹ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    E : Type u_4
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    H : Type u_5
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace HM M
    R : Type u
    inst✝³ : TopologicalSpace R
    inst✝² : ChartedSpace H R
    inst✝¹ : CommRing R
    inst✝ : SmoothRing I R
    x : M
    y : (smoothSheaf IM I M R).presheaf.stalk x
    ⊢ Exists fun a => Eq ((smoothSheafCommRing.eval IM I M R x) a) (smoothSheaf.ev …
  -/
  use (smoothSheafCommRing.forgetStalk IM I M R x).inv y
  /-
    case h
    𝕜 : Type u_1
    inst✝¹² : NontriviallyNormedField 𝕜
    EM : Type u_2
    inst✝¹¹ : NormedAddCommGroup EM
    inst✝¹⁰ : NormedSpace 𝕜 EM
    HM : Type u_3
    inst✝⁹ : TopologicalSpace HM
    IM : ModelWithCorners 𝕜 EM HM
    E : Type u_4
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    H : Type u_5
    inst✝⁶ : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace HM M
    R : Type u
    inst✝³ : TopologicalSpace R
    inst✝² : ChartedSpace H R
    inst✝¹ : CommRing R
    inst✝ : SmoothRing I R
    x : M
    y : (smoothSheaf IM I M R).presheaf.stalk x
    ⊢ Eq ((smoothSheafCommRing.eval IM I M R x) ((smoothSheafCommRing.forgetStalk  …
  -/
  apply smoothSheafCommRing.forgetStalk_inv_comp_eval_apply
  /-
    🎉 no goals
  -/


instance [Nontrivial R] (x : M) : Nontrivial ((smoothSheafCommRing IM I M R).presheaf.stalk x) :=
  (smoothSheafCommRing.eval_surjective IM I M R x).nontrivial


@[simp] lemma smoothSheafCommRing.eval_germ (U : Opens M) (x : M) (hx : x ∈ U)
    (f : (smoothSheafCommRing IM I M R).presheaf.obj (op U)) :
    smoothSheafCommRing.eval IM I M R x ((smoothSheafCommRing IM I M R).presheaf.germ U x hx f)
    = f ⟨x, hx⟩ :=
  smoothSheafCommRing.evalHom_germ IM I M R U x hx f


