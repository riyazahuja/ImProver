/-- The directed system of `R`-modules of the form `R/J`, where `J` is an ideal of `R`,
determined by the functor `I`  -/
def ringModIdeals (I : D ⥤ Ideal R) : D ⥤ ModuleCat.{u} R where
  obj t := ModuleCat.of R <| R ⧸ I.obj t
  map w := ModuleCat.ofHom <| Submodule.mapQ _ _ LinearMap.id (I.map w).down.down

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO:  Once this file is ported, move this instance to the right location.

instance moduleCat_enoughProjectives' : EnoughProjectives (ModuleCat.{u} R) :=
  ModuleCat.moduleCat_enoughProjectives.{u}


/-- The diagram we will take the colimit of to define local cohomology, corresponding to the
directed system determined by the functor `I` -/
def diagram (I : D ⥤ Ideal R) (i : ℕ) : Dᵒᵖ ⥤ ModuleCat.{u} R ⥤ ModuleCat.{u} R :=
  (ringModIdeals I).op ⋙ Ext R (ModuleCat.{u} R) i


lemma hasColimitDiagram (I : D ⥤ Ideal R) (i : ℕ) :
    HasColimit (diagram I i) := by
  /-
    R : Type (max u v)
    inst✝¹ : CommRing R
    D : Type v
    inst✝ : CategoryTheory.SmallCategory D
    I : CategoryTheory.Functor D (Ideal R)
    i : Nat
    ⊢ CategoryTheory.Limits.HasColimit (localCohomology.diagram I i)
  -/
  have : HasColimitsOfShape Dᵒᵖ (AddCommGrpMax.{u, v}) := inferInstance
  /-
    R : Type (max u v)
    inst✝¹ : CommRing R
    D : Type v
    inst✝ : CategoryTheory.SmallCategory D
    I : CategoryTheory.Functor D (Ideal R)
    i : Nat
    this : CategoryTheory.Limits.HasColimitsOfShape (Opposite D) AddCommGrpMax
    ⊢ CategoryTheory.Limits.HasColimit (localCohomology.diagram I i)
  -/
  infer_instance
  /-
    🎉 no goals
  -/

/-
In this definition we do not assume any special property of the diagram `I`, but the relevant case
will be where `I` is (cofinal with) the diagram of powers of a single given ideal.

Below, we give two equivalent definitions of the usual local cohomology with support
in an ideal `J`, `localCohomology` and `localCohomology.ofSelfLERadical`.
 -/

/-- `localCohomology.ofDiagram I i` is the functor sending a module `M` over a commutative
ring `R` to the direct limit of `Ext^i(R/J, M)`, where `J` ranges over a collection of ideals
of `R`, represented as a functor `I`. -/
def ofDiagram (I : D ⥤ Ideal R) (i : ℕ) : ModuleCatMax.{u, v} R ⥤ ModuleCatMax.{u, v} R :=
  have := hasColimitDiagram.{u, v} I i
  colimit (diagram I i)


/-- Local cohomology along a composition of diagrams. -/
def diagramComp (i : ℕ) : diagram (I' ⋙ I) i ≅ I'.op ⋙ diagram I i :=
  Iso.refl _


/-- Local cohomology agrees along precomposition with a cofinal diagram. -/
@[nolint unusedHavesSuffices]
def isoOfFinal [Functor.Initial I'] (i : ℕ) :
    ofDiagram.{max u v, v'} (I' ⋙ I) i ≅ ofDiagram.{max u v', v} I i :=
  have := hasColimitDiagram.{max u v', v} I i
  have := hasColimitDiagram.{max u v, v'} (I' ⋙ I) i
  HasColimit.isoOfNatIso (diagramComp.{u} I' I i) ≪≫ Functor.Final.colimitIso _ _


/-- The functor sending a natural number `i` to the `i`-th power of the ideal `J` -/
def idealPowersDiagram (J : Ideal R) : ℕᵒᵖ ⥤ Ideal R where
  obj t := J ^ unop t
  map w := ⟨⟨Ideal.pow_le_pow_right w.unop.down.down⟩⟩


/-- The full subcategory of all ideals with radical containing `J` -/
def SelfLERadical (J : Ideal R) : Type u :=
  FullSubcategory fun J' : Ideal R => J ≤ J'.radical

-- Porting note: `deriving Category` is not able to derive this instance
-- https://github.com/leanprover-community/mathlib4/issues/5020

instance (J : Ideal R) : Category (SelfLERadical J) :=
  (FullSubcategory.category _)


instance SelfLERadical.inhabited (J : Ideal R) : Inhabited (SelfLERadical J) where
  default := ⟨J, Ideal.le_radical⟩


/-- The diagram of all ideals with radical containing `J`, represented as a functor.
This is the "largest" diagram that computes local cohomology with support in `J`. -/
def selfLERadicalDiagram (J : Ideal R) : SelfLERadical J ⥤ Ideal R :=
  fullSubcategoryInclusion _


/-- `localCohomology J i` is `i`-th the local cohomology module of a module `M` over
a commutative ring `R` with support in the ideal `J` of `R`, defined as the direct limit
of `Ext^i(R/J^t, M)` over all powers `t : ℕ`. -/
def localCohomology (J : Ideal R) (i : ℕ) : ModuleCat.{u} R ⥤ ModuleCat.{u} R :=
  ofDiagram (idealPowersDiagram J) i


/-- Local cohomology as the direct limit of `Ext^i(R/J', M)` over *all* ideals `J'` with radical
containing `J`. -/
def localCohomology.ofSelfLERadical (J : Ideal R) (i : ℕ) : ModuleCat.{u} R ⥤ ModuleCat.{u} R :=
  ofDiagram.{u} (selfLERadicalDiagram.{u} J) i


/-- Lifting `idealPowersDiagram J` from a diagram valued in `ideals R` to a diagram
valued in `SelfLERadical J`. -/
def idealPowersToSelfLERadical (J : Ideal R) : ℕᵒᵖ ⥤ SelfLERadical J :=
  FullSubcategory.lift _ (idealPowersDiagram J) fun k => by
    /-
      R : Type u
      inst✝ : CommRing R
      J : Ideal R
      k : Opposite Nat
      ⊢ LE.le J ((localCohomology.idealPowersDiagram J).obj k).radical
    -/
    change _ ≤ (J ^ unop k).radical
    /-
      R : Type u
      inst✝ : CommRing R
      J : Ideal R
      k : Opposite Nat
      ⊢ LE.le J (HPow.hPow J (Opposite.unop k)).radical
    -/
    cases' unop k with n
      /-
        case zero
        R : Type u
        inst✝ : CommRing R
        J : Ideal R
        k : Opposite Nat
        ⊢ LE.le J (HPow.hPow J 0).radical
      -/
    · simp [Ideal.radical_top, pow_zero, Ideal.one_eq_top, le_top]
      /-
        🎉 no goals
      -/
      /-
        case succ
        R : Type u
        inst✝ : CommRing R
        J : Ideal R
        k : Opposite Nat
        n : Nat
        ⊢ LE.le J (HPow.hPow J (HAdd.hAdd n 1)).radical
      -/
    · simp only [J.radical_pow n.succ_ne_zero, Ideal.le_radical]
      /-
        🎉 no goals
      -/


/-- The diagram of powers of `J` is initial in the diagram of all ideals with
radical containing `J`. This uses noetherianness. -/
instance ideal_powers_initial [hR : IsNoetherian R R] :
    Functor.Initial (idealPowersToSelfLERadical J) where
  out J' := by
    /-
      R : Type u
      inst✝ : CommRing R
      I J K : Ideal R
      hR : IsNoetherian R R
      J' : localCohomology.SelfLERadical J
      ⊢ CategoryTheory.IsConnected (CategoryTheory.CostructuredArrow (localCohomolog …
    -/
    apply (config := {allowSynthFailures := true }) zigzag_isConnected
      /-
        case inst
        R : Type u
        inst✝ : CommRing R
        I J K : Ideal R
        hR : IsNoetherian R R
        J' : localCohomology.SelfLERadical J
        ⊢ Nonempty (CategoryTheory.CostructuredArrow (localCohomology.idealPowersToSel …
      -/
    · obtain ⟨k, hk⟩ := Ideal.exists_pow_le_of_le_radical_of_fg J'.2 (isNoetherian_def.mp hR _)
      /-
        case inst.intro
        R : Type u
        inst✝ : CommRing R
        I J K : Ideal R
        hR : IsNoetherian R R
        J' : localCohomology.SelfLERadical J
        k : Nat
        hk : LE.le (HPow.hPow J k) J'.obj
        ⊢ Nonempty (CategoryTheory.CostructuredArrow (localCohomology.idealPowersToSel …
      -/
      exact ⟨CostructuredArrow.mk (⟨⟨hk⟩⟩ : (idealPowersToSelfLERadical J).obj (op k) ⟶ J')⟩
      /-
        🎉 no goals
      -/
      /-
        case h
        R : Type u
        inst✝ : CommRing R
        I J K : Ideal R
        hR : IsNoetherian R R
        J' : localCohomology.SelfLERadical J
        ⊢ ∀ (j₁ j₂ : CategoryTheory.CostructuredArrow (localCohomology.idealPowersToSe …
      -/
    · intro j1 j2
      /-
        case h
        R : Type u
        inst✝ : CommRing R
        I J K : Ideal R
        hR : IsNoetherian R R
        J' : localCohomology.SelfLERadical J
        j1 j2 : CategoryTheory.CostructuredArrow (localCohomology.idealPowersToSelfLER …
        ⊢ CategoryTheory.Zigzag j1 j2
      -/
      apply Relation.ReflTransGen.single
      -- The inclusions `J^n1 ≤ J'` and `J^n2 ≤ J'` always form a triangle, based on
      -- which exponent is larger.
      /-
        case h.hab
        R : Type u
        inst✝ : CommRing R
        I J K : Ideal R
        hR : IsNoetherian R R
        J' : localCohomology.SelfLERadical J
        j1 j2 : CategoryTheory.CostructuredArrow (localCohomology.idealPowersToSelfLER …
        ⊢ CategoryTheory.Zag j1 j2
      -/
      rcases le_total (unop j1.left) (unop j2.left) with h | h
        /-
          case h.hab.inl
          R : Type u
          inst✝ : CommRing R
          I J K : Ideal R
          hR : IsNoetherian R R
          J' : localCohomology.SelfLERadical J
          j1 j2 : CategoryTheory.CostructuredArrow (localCohomology.idealPowersToSelfLER …
          h : LE.le (Opposite.unop j1.left) (Opposite.unop j2.left)
          ⊢ CategoryTheory.Zag j1 j2
        -/
      · right; exact ⟨CostructuredArrow.homMk (homOfLE h).op rfl⟩
               /-
                 🎉 no goals
               -/
        /-
          case h.hab.inr
          R : Type u
          inst✝ : CommRing R
          I J K : Ideal R
          hR : IsNoetherian R R
          J' : localCohomology.SelfLERadical J
          j1 j2 : CategoryTheory.CostructuredArrow (localCohomology.idealPowersToSelfLER …
          h : LE.le (Opposite.unop j2.left) (Opposite.unop j1.left)
          ⊢ CategoryTheory.Zag j1 j2
        -/
      · left; exact ⟨CostructuredArrow.homMk (homOfLE h).op rfl⟩
              /-
                🎉 no goals
              -/


/-- Local cohomology (defined in terms of powers of `J`) agrees with local
cohomology computed over all ideals with radical containing `J`. -/
def isoSelfLERadical (J : Ideal.{u} R) [IsNoetherian.{u,u} R R] (i : ℕ) :
    localCohomology.ofSelfLERadical.{u} J i ≅ localCohomology.{u} J i :=
  (localCohomology.isoOfFinal.{u, u, 0} (idealPowersToSelfLERadical.{u} J)
    (selfLERadicalDiagram.{u} J) i).symm ≪≫
      HasColimit.isoOfNatIso.{0,0,u+1,u+1} (Iso.refl.{u+1,u+1} _)


/-- Casting from the full subcategory of ideals with radical containing `J` to the full
subcategory of ideals with radical containing `K`. -/
def SelfLERadical.cast (hJK : J.radical = K.radical) : SelfLERadical J ⥤ SelfLERadical K :=
  FullSubcategory.map fun L hL => by
    /-
      R : Type u
      inst✝ : CommRing R
      I J K : Ideal R
      hJK : Eq J.radical K.radical
      L : Ideal R
      hL : LE.le J L.radical
      ⊢ LE.le K L.radical
    -/
    rw [← Ideal.radical_le_radical_iff] at hL ⊢
    /-
      R : Type u
      inst✝ : CommRing R
      I J K : Ideal R
      hJK : Eq J.radical K.radical
      L : Ideal R
      hL : LE.le J.radical L.radical
      ⊢ LE.le K.radical L.radical
    -/
    exact hJK.symm.trans_le hL
    /-
      🎉 no goals
    -/

-- TODO generalize this to the equivalence of full categories for any `iff`.

/-- The equivalence of categories `SelfLERadical J ≌ SelfLERadical K`
when `J.radical = K.radical`. -/
def SelfLERadical.castEquivalence (hJK : J.radical = K.radical) :
    SelfLERadical J ≌ SelfLERadical K where
  functor := SelfLERadical.cast hJK
  inverse := SelfLERadical.cast hJK.symm
  unitIso := Iso.refl _
  counitIso := Iso.refl _


instance SelfLERadical.cast_isEquivalence (hJK : J.radical = K.radical) :
    (SelfLERadical.cast hJK).IsEquivalence :=
  (castEquivalence hJK).isEquivalence_functor


/-- The natural isomorphism between local cohomology defined using the `of_self_le_radical`
diagram, assuming `J.radical = K.radical`. -/
def SelfLERadical.isoOfSameRadical (hJK : J.radical = K.radical) (i : ℕ) :
    ofSelfLERadical J i ≅ ofSelfLERadical K i :=
  (isoOfFinal.{u, u, u} (SelfLERadical.cast hJK.symm) _ _).symm


/-- Local cohomology agrees on ideals with the same radical. -/
def isoOfSameRadical [IsNoetherian R R] (hJK : J.radical = K.radical) (i : ℕ) :
    localCohomology J i ≅ localCohomology K i :=
  (isoSelfLERadical J i).symm ≪≫ SelfLERadical.isoOfSameRadical hJK i ≪≫ isoSelfLERadical K i


