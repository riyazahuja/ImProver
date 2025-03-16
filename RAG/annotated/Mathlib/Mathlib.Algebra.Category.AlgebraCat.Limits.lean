instance semiringObj (j) : Semiring ((F ⋙ forget (AlgebraCat R)).obj j) :=
  inferInstanceAs <| Semiring (F.obj j)


instance algebraObj (j) :
    Algebra R ((F ⋙ forget (AlgebraCat R)).obj j) :=
  inferInstanceAs <| Algebra R (F.obj j)


/-- The flat sections of a functor into `AlgebraCat R` form a submodule of all sections.
-/
def sectionsSubalgebra : Subalgebra R (∀ j, F.obj j) :=
  { SemiRingCat.sectionsSubsemiring
      (F ⋙ forget₂ (AlgebraCat R) RingCat.{w} ⋙ forget₂ RingCat SemiRingCat.{w}) with
    algebraMap_mem' := fun r _ _ f => (F.map f).hom.commutes r }


instance (F : J ⥤ AlgebraCat.{w} R) : Ring (F ⋙ forget _).sections :=
  inferInstanceAs <| Ring (sectionsSubalgebra F)


instance (F : J ⥤ AlgebraCat.{w} R) : Algebra R (F ⋙ forget _).sections :=
  inferInstanceAs <| Algebra R (sectionsSubalgebra F)


instance : Small.{w} (sectionsSubalgebra F) :=
  inferInstanceAs <| Small.{w} (F ⋙ forget _).sections


instance limitSemiring :
    Ring.{w} (Types.Small.limitCone.{v, w} (F ⋙ forget (AlgebraCat.{w} R))).pt :=
  inferInstanceAs <| Ring (Shrink (sectionsSubalgebra F))


instance limitAlgebra :
    Algebra R (Types.Small.limitCone (F ⋙ forget (AlgebraCat.{w} R))).pt :=
  inferInstanceAs <| Algebra R (Shrink (sectionsSubalgebra F))


/-- `limit.π (F ⋙ forget (AlgebraCat R)) j` as a `AlgHom`. -/
def limitπAlgHom (j) :
    (Types.Small.limitCone (F ⋙ forget (AlgebraCat R))).pt →ₐ[R]
      (F ⋙ forget (AlgebraCat.{w} R)).obj j :=
  letI : Small.{w}
      (Functor.sections ((F ⋙ forget₂ _ RingCat ⋙ forget₂ _ SemiRingCat) ⋙ forget _)) :=
    inferInstanceAs <| Small.{w} (F ⋙ forget _).sections
  { SemiRingCat.limitπRingHom
      (F ⋙ forget₂ (AlgebraCat R) RingCat.{w} ⋙ forget₂ RingCat SemiRingCat.{w}) j with
    toFun := (Types.Small.limitCone (F ⋙ forget (AlgebraCat.{w} R))).π.app j
    commutes' := fun x => by
      simp only [Types.Small.limitCone_π_app, ← Shrink.algEquiv_apply _ R,
        Types.Small.limitCone_pt, AlgEquiv.commutes]
      /-
        R : Type u
        inst✝² : CommRing R
        J : Type v
        inst✝¹ : CategoryTheory.Category.{t, v} J
        F : CategoryTheory.Functor J (AlgebraCat R)
        inst✝ : Small.{w, max v w} ↑(F.comp (CategoryTheory.forget (AlgebraCat R))).se …
        j : J
        this : Small.{w, max v w} ↑((F.comp ((CategoryTheory.forget₂ (AlgebraCat R) Ri …
        x : R
        ⊢ Eq (↑((algebraMap R ↑(F.comp (CategoryTheory.forget (AlgebraCat R))).section …
      -/
      rfl
      /-
        🎉 no goals
      -/
    }


/-- Construction of a limit cone in `AlgebraCat R`.
(Internal use only; use the limits API.)
-/
def limitCone : Cone F where
  pt := AlgebraCat.of R (Types.Small.limitCone (F ⋙ forget _)).pt
  π :=
    { app := fun j ↦ ofHom <| limitπAlgHom F j
      naturality := fun _ _ f => by
        /-
          R : Type u
          inst✝² : CommRing R
          J : Type v
          inst✝¹ : CategoryTheory.Category.{t, v} J
          F : CategoryTheory.Functor J (AlgebraCat R)
          inst✝ : Small.{w, max v w} ↑(F.comp (CategoryTheory.forget (AlgebraCat R))).se …
          x✝¹ x✝ : J
          f : Quiver.Hom x✝¹ x✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const J).ob …
        -/
        ext : 1
        /-
          case hf
          R : Type u
          inst✝² : CommRing R
          J : Type v
          inst✝¹ : CategoryTheory.Category.{t, v} J
          F : CategoryTheory.Functor J (AlgebraCat R)
          inst✝ : Small.{w, max v w} ↑(F.comp (CategoryTheory.forget (AlgebraCat R))).se …
          x✝¹ x✝ : J
          f : Quiver.Hom x✝¹ x✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const J).ob …
        -/
        exact AlgHom.coe_fn_injective ((Types.Small.limitCone (F ⋙ forget _)).π.naturality f) }
        /-
          🎉 no goals
        -/


/-- Witness that the limit cone in `AlgebraCat R` is a limit cone.
(Internal use only; use the limits API.)
-/
def limitConeIsLimit : IsLimit (limitCone.{v, w} F) := by
  refine
    IsLimit.ofFaithful (forget (AlgebraCat R)) (Types.Small.limitConeIsLimit.{v, w} _)
      -- Porting note: in mathlib3 the function term
      -- `fun v => ⟨fun j => ((forget (AlgebraCat R)).mapCone s).π.app j v`
      -- was provided by unification, and the last argument `(fun s => _)` was `(fun s => rfl)`.
      (fun s => ofHom
        { toFun := _, map_one' := ?_, map_mul' := ?_, map_zero' := ?_, map_add' := ?_,
          commutes' := ?_ })
      (fun s => rfl)
    /-
      case refine_1
      R : Type u
      inst✝² : CommRing R
      J : Type v
      inst✝¹ : CategoryTheory.Category.{t, v} J
      F : CategoryTheory.Functor J (AlgebraCat R)
      inst✝ : Small.{w, max v w} ↑(F.comp (CategoryTheory.forget (AlgebraCat R))).se …
      s : CategoryTheory.Limits.Cone F
      ⊢ Eq ((equivShrink ↑(F.comp (CategoryTheory.forget (AlgebraCat R))).sections). …
    -/
  · congr
    /-
      case refine_1.e_a.e_val
      R : Type u
      inst✝² : CommRing R
      J : Type v
      inst✝¹ : CategoryTheory.Category.{t, v} J
      F : CategoryTheory.Functor J (AlgebraCat R)
      inst✝ : Small.{w, max v w} ↑(F.comp (CategoryTheory.forget (AlgebraCat R))).se …
      s : CategoryTheory.Limits.Cone F
      ⊢ Eq (fun j => ((CategoryTheory.forget (AlgebraCat R)).mapCone s).π.app j 1) 1
    -/
    ext j
    /-
      case refine_1.e_a.e_val.h
      R : Type u
      inst✝² : CommRing R
      J : Type v
      inst✝¹ : CategoryTheory.Category.{t, v} J
      F : CategoryTheory.Functor J (AlgebraCat R)
      inst✝ : Small.{w, max v w} ↑(F.comp (CategoryTheory.forget (AlgebraCat R))).se …
      s : CategoryTheory.Limits.Cone F
      j : J
      ⊢ Eq (((CategoryTheory.forget (AlgebraCat R)).mapCone s).π.app j 1) (1 j)
    -/
    simp only [Functor.mapCone_π_app, forget_map, map_one, Pi.one_apply]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u
      inst✝² : CommRing R
      J : Type v
      inst✝¹ : CategoryTheory.Category.{t, v} J
      F : CategoryTheory.Functor J (AlgebraCat R)
      inst✝ : Small.{w, max v w} ↑(F.comp (CategoryTheory.forget (AlgebraCat R))).se …
      s : CategoryTheory.Limits.Cone F
      ⊢ ∀ (x y : ↑s.1), Eq ({ toFun := fun v => (equivShrink ↑(F.comp (CategoryTheor …
    -/
  · intro x y
    /-
      case refine_2
      R : Type u
      inst✝² : CommRing R
      J : Type v
      inst✝¹ : CategoryTheory.Category.{t, v} J
      F : CategoryTheory.Functor J (AlgebraCat R)
      inst✝ : Small.{w, max v w} ↑(F.comp (CategoryTheory.forget (AlgebraCat R))).se …
      s : CategoryTheory.Limits.Cone F
      x y : ↑s.1
      ⊢ Eq ({ toFun := fun v => (equivShrink ↑(F.comp (CategoryTheory.forget (Algebr …
    -/
    ext j
    simp only [Functor.comp_obj, forget_obj, Equiv.toFun_as_coe, Functor.mapCone_pt,
      Functor.mapCone_π_app, forget_map, Equiv.symm_apply_apply,
      Types.Small.limitCone_pt, equivShrink_symm_mul]
    /-
      case refine_2.w.a.h
      R : Type u
      inst✝² : CommRing R
      J : Type v
      inst✝¹ : CategoryTheory.Category.{t, v} J
      F : CategoryTheory.Functor J (AlgebraCat R)
      inst✝ : Small.{w, max v w} ↑(F.comp (CategoryTheory.forget (AlgebraCat R))).se …
      s : CategoryTheory.Limits.Cone F
      x y : ↑s.1
      j : J
      ⊢ Eq ((s.π.app j).hom (HMul.hMul x y)) (↑(HMul.hMul ⟨fun j => (s.π.app j).hom  …
    -/
    apply map_mul
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      R : Type u
      inst✝² : CommRing R
      J : Type v
      inst✝¹ : CategoryTheory.Category.{t, v} J
      F : CategoryTheory.Functor J (AlgebraCat R)
      inst✝ : Small.{w, max v w} ↑(F.comp (CategoryTheory.forget (AlgebraCat R))).se …
      s : CategoryTheory.Limits.Cone F
      ⊢ Eq ((↑{ toFun := fun v => (equivShrink ↑(F.comp (CategoryTheory.forget (Alge …
    -/
  · ext j
    simp only [Functor.comp_obj, forget_obj, Equiv.toFun_as_coe, Functor.mapCone_pt,
      Functor.mapCone_π_app, forget_map, Equiv.symm_apply_apply,
      equivShrink_symm_zero]
    /-
      case refine_3.w.a.h
      R : Type u
      inst✝² : CommRing R
      J : Type v
      inst✝¹ : CategoryTheory.Category.{t, v} J
      F : CategoryTheory.Functor J (AlgebraCat R)
      inst✝ : Small.{w, max v w} ↑(F.comp (CategoryTheory.forget (AlgebraCat R))).se …
      s : CategoryTheory.Limits.Cone F
      j : J
      ⊢ Eq ((s.π.app j).hom 0) (↑0 j)
    -/
    apply map_zero
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      R : Type u
      inst✝² : CommRing R
      J : Type v
      inst✝¹ : CategoryTheory.Category.{t, v} J
      F : CategoryTheory.Functor J (AlgebraCat R)
      inst✝ : Small.{w, max v w} ↑(F.comp (CategoryTheory.forget (AlgebraCat R))).se …
      s : CategoryTheory.Limits.Cone F
      ⊢ ∀ (x y : ↑s.1), Eq ((↑{ toFun := fun v => (equivShrink ↑(F.comp (CategoryThe …
    -/
  · intro x y
    /-
      case refine_4
      R : Type u
      inst✝² : CommRing R
      J : Type v
      inst✝¹ : CategoryTheory.Category.{t, v} J
      F : CategoryTheory.Functor J (AlgebraCat R)
      inst✝ : Small.{w, max v w} ↑(F.comp (CategoryTheory.forget (AlgebraCat R))).se …
      s : CategoryTheory.Limits.Cone F
      x y : ↑s.1
      ⊢ Eq ((↑{ toFun := fun v => (equivShrink ↑(F.comp (CategoryTheory.forget (Alge …
    -/
    ext j
    simp only [Functor.comp_obj, forget_obj, Equiv.toFun_as_coe, Functor.mapCone_pt,
      Functor.mapCone_π_app, forget_map, Equiv.symm_apply_apply,
      Types.Small.limitCone_pt, equivShrink_symm_add]
    /-
      case refine_4.w.a.h
      R : Type u
      inst✝² : CommRing R
      J : Type v
      inst✝¹ : CategoryTheory.Category.{t, v} J
      F : CategoryTheory.Functor J (AlgebraCat R)
      inst✝ : Small.{w, max v w} ↑(F.comp (CategoryTheory.forget (AlgebraCat R))).se …
      s : CategoryTheory.Limits.Cone F
      x y : ↑s.1
      j : J
      ⊢ Eq ((s.π.app j).hom (HAdd.hAdd x y)) (↑(HAdd.hAdd ⟨fun j => (s.π.app j).hom  …
    -/
    apply map_add
    /-
      🎉 no goals
    -/
    /-
      case refine_5
      R : Type u
      inst✝² : CommRing R
      J : Type v
      inst✝¹ : CategoryTheory.Category.{t, v} J
      F : CategoryTheory.Functor J (AlgebraCat R)
      inst✝ : Small.{w, max v w} ↑(F.comp (CategoryTheory.forget (AlgebraCat R))).se …
      s : CategoryTheory.Limits.Cone F
      ⊢ ∀ (r : R), Eq ((↑↑{ toFun := fun v => (equivShrink ↑(F.comp (CategoryTheory. …
    -/
  · intro r
    /-
      case refine_5
      R : Type u
      inst✝² : CommRing R
      J : Type v
      inst✝¹ : CategoryTheory.Category.{t, v} J
      F : CategoryTheory.Functor J (AlgebraCat R)
      inst✝ : Small.{w, max v w} ↑(F.comp (CategoryTheory.forget (AlgebraCat R))).se …
      s : CategoryTheory.Limits.Cone F
      r : R
      ⊢ Eq ((↑↑{ toFun := fun v => (equivShrink ↑(F.comp (CategoryTheory.forget (Alg …
    -/
    simp only [← Shrink.algEquiv_symm_apply _ R, limitCone, Equiv.algebraMap_def, Equiv.symm_symm]
    /-
      case refine_5
      R : Type u
      inst✝² : CommRing R
      J : Type v
      inst✝¹ : CategoryTheory.Category.{t, v} J
      F : CategoryTheory.Functor J (AlgebraCat R)
      inst✝ : Small.{w, max v w} ↑(F.comp (CategoryTheory.forget (AlgebraCat R))).se …
      s : CategoryTheory.Limits.Cone F
      r : R
      ⊢ Eq ((equivShrink ↑(F.comp (CategoryTheory.forget (AlgebraCat R))).sections). …
    -/
    apply congrArg
    /-
      case refine_5.h
      R : Type u
      inst✝² : CommRing R
      J : Type v
      inst✝¹ : CategoryTheory.Category.{t, v} J
      F : CategoryTheory.Functor J (AlgebraCat R)
      inst✝ : Small.{w, max v w} ↑(F.comp (CategoryTheory.forget (AlgebraCat R))).se …
      s : CategoryTheory.Limits.Cone F
      r : R
      ⊢ Eq ⟨fun j => ((CategoryTheory.forget (AlgebraCat R)).mapCone s).π.app j ((al …
    -/
    apply Subtype.ext
    /-
      case refine_5.h.a
      R : Type u
      inst✝² : CommRing R
      J : Type v
      inst✝¹ : CategoryTheory.Category.{t, v} J
      F : CategoryTheory.Functor J (AlgebraCat R)
      inst✝ : Small.{w, max v w} ↑(F.comp (CategoryTheory.forget (AlgebraCat R))).se …
      s : CategoryTheory.Limits.Cone F
      r : R
      ⊢ Eq ↑⟨fun j => ((CategoryTheory.forget (AlgebraCat R)).mapCone s).π.app j ((a …
    -/
    ext j
    /-
      case refine_5.h.a.h
      R : Type u
      inst✝² : CommRing R
      J : Type v
      inst✝¹ : CategoryTheory.Category.{t, v} J
      F : CategoryTheory.Functor J (AlgebraCat R)
      inst✝ : Small.{w, max v w} ↑(F.comp (CategoryTheory.forget (AlgebraCat R))).se …
      s : CategoryTheory.Limits.Cone F
      r : R
      j : J
      ⊢ Eq (↑⟨fun j => ((CategoryTheory.forget (AlgebraCat R)).mapCone s).π.app j (( …
    -/
    exact (s.π.app j).hom.commutes r
    /-
      🎉 no goals
    -/


/-- The category of R-algebras has all limits. -/
lemma hasLimitsOfSize [UnivLE.{v, w}] : HasLimitsOfSize.{t, v} (AlgebraCat.{w} R) :=
  { has_limits_of_shape := fun _ _ =>
    { has_limit := fun F => HasLimit.mk
        { cone := limitCone F
          isLimit := limitConeIsLimit F } } }


instance hasLimits : HasLimits (AlgebraCat.{w} R) :=
  AlgebraCat.hasLimitsOfSize.{w, w, u}


/-- The forgetful functor from R-algebras to rings preserves all limits.
-/
instance forget₂Ring_preservesLimitsOfSize [UnivLE.{v, w}] :
    PreservesLimitsOfSize.{t, v} (forget₂ (AlgebraCat.{w} R) RingCat.{w}) where
  preservesLimitsOfShape :=
    { preservesLimit := fun {K} ↦
        preservesLimit_of_preserves_limit_cone (limitConeIsLimit K)
          (RingCat.limitConeIsLimit.{v, w}
            (_ ⋙ forget₂ (AlgebraCat.{w} R) RingCat.{w})) }


instance forget₂Ring_preservesLimits : PreservesLimits (forget₂ (AlgebraCat R) RingCat.{w}) :=
  AlgebraCat.forget₂Ring_preservesLimitsOfSize.{w, w}


/-- The forgetful functor from R-algebras to R-modules preserves all limits.
-/
instance forget₂Module_preservesLimitsOfSize [UnivLE.{v, w}] : PreservesLimitsOfSize.{t, v}
    (forget₂ (AlgebraCat.{w} R) (ModuleCat.{w} R)) where
  preservesLimitsOfShape :=
    { preservesLimit := fun {K} ↦
        preservesLimit_of_preserves_limit_cone (limitConeIsLimit K)
          (ModuleCat.HasLimits.limitConeIsLimit
            (K ⋙ forget₂ (AlgebraCat.{w} R) (ModuleCat.{w} R))) }


instance forget₂Module_preservesLimits :
    PreservesLimits (forget₂ (AlgebraCat R) (ModuleCat.{w} R)) :=
  AlgebraCat.forget₂Module_preservesLimitsOfSize.{w, w}


/-- The forgetful functor from R-algebras to types preserves all limits.
-/
instance forget_preservesLimitsOfSize [UnivLE.{v, w}] :
    PreservesLimitsOfSize.{t, v} (forget (AlgebraCat.{w} R)) where
  preservesLimitsOfShape :=
    { preservesLimit := fun {K} ↦
       preservesLimit_of_preserves_limit_cone (limitConeIsLimit K)
          (Types.Small.limitConeIsLimit.{v} (K ⋙ forget _)) }


instance forget_preservesLimits : PreservesLimits (forget (AlgebraCat.{w} R)) :=
  AlgebraCat.forget_preservesLimitsOfSize.{w, w}


