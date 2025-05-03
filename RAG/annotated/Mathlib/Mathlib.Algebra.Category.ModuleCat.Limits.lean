instance addCommGroupObj (j) :
    AddCommGroup ((F ⋙ forget (ModuleCat R)).obj j) :=
  inferInstanceAs <| AddCommGroup (F.obj j)


instance moduleObj (j) :
    Module.{u, w} R ((F ⋙ forget (ModuleCat R)).obj j) :=
  inferInstanceAs <| Module R (F.obj j)


/-- The flat sections of a functor into `ModuleCat R` form a submodule of all sections.
-/
def sectionsSubmodule : Submodule R (∀ j, F.obj j) :=
  { AddGrp.sectionsAddSubgroup.{v, w}
      (F ⋙ forget₂ (ModuleCat R) AddCommGrp.{w} ⋙
          forget₂ AddCommGrp AddGrp.{w}) with
    carrier := (F ⋙ forget (ModuleCat R)).sections
    smul_mem' := fun r s sh j j' f => by
      /-
        R : Type u
        inst✝¹ : Ring R
        J : Type v
        inst✝ : CategoryTheory.Category.{t, v} J
        F : CategoryTheory.Functor J (ModuleCat R)
        r : R
        s : (j : J) → ↑(F.obj j)
        sh : Membership.mem { carrier := (F.comp (CategoryTheory.forget (ModuleCat R)) …
        j j' : J
        f : Quiver.Hom j j'
        ⊢ Eq ((F.comp (CategoryTheory.forget (ModuleCat R))).map f (HSMul.hSMul r s j) …
      -/
      simpa [Functor.sections, forget_map] using congr_arg (r • ·) (sh f) }
      /-
        🎉 no goals
      -/


instance : AddCommMonoid (F ⋙ forget (ModuleCat R)).sections :=
  inferInstanceAs <| AddCommMonoid (sectionsSubmodule F)


instance : Module R (F ⋙ forget (ModuleCat R)).sections :=
  inferInstanceAs <| Module R (sectionsSubmodule F)


instance : Small.{w} (sectionsSubmodule F) :=
  inferInstanceAs <| Small.{w} (Functor.sections (F ⋙ forget (ModuleCat R)))

-- Adding the following instance speeds up `limitModule` noticeably,
-- by preventing a bad unfold of `limitAddCommGroup`.

instance limitAddCommMonoid :
    AddCommMonoid (Types.Small.limitCone.{v, w} (F ⋙ forget (ModuleCat.{w} R))).pt :=
  inferInstanceAs <| AddCommMonoid (Shrink (sectionsSubmodule F))


instance limitAddCommGroup :
    AddCommGroup (Types.Small.limitCone.{v, w} (F ⋙ forget (ModuleCat.{w} R))).pt :=
  inferInstanceAs <| AddCommGroup (Shrink.{w} (sectionsSubmodule F))


instance limitModule :
    Module R (Types.Small.limitCone.{v, w} (F ⋙ forget (ModuleCat.{w} R))).pt :=
  inferInstanceAs <| Module R (Shrink (sectionsSubmodule F))


/-- `limit.π (F ⋙ forget (ModuleCat.{w} R)) j` as an `R`-linear map. -/
def limitπLinearMap (j) :
    (Types.Small.limitCone (F ⋙ forget (ModuleCat.{w} R))).pt →ₗ[R]
      (F ⋙ forget (ModuleCat R)).obj j where
  toFun := (Types.Small.limitCone (F ⋙ forget (ModuleCat R))).π.app j
  map_smul' _ _ := by
    simp only [Types.Small.limitCone_π_app,
      ← Shrink.linearEquiv_apply (F ⋙ forget (ModuleCat R)).sections R, map_smul]
    /-
      R : Type u
      inst✝² : Ring R
      J : Type v
      inst✝¹ : CategoryTheory.Category.{t, v} J
      F : CategoryTheory.Functor J (ModuleCat R)
      inst✝ : Small.{w, max v w} ↑(F.comp (CategoryTheory.forget (ModuleCat R))).sec …
      j : J
      x✝¹ : R
      x✝ : (CategoryTheory.Limits.Types.Small.limitCone (F.comp (CategoryTheory.forg …
      ⊢ Eq (↑(HSMul.hSMul x✝¹ ((Shrink.linearEquiv (↑(F.comp (CategoryTheory.forget  …
    -/
    simp only [Shrink.linearEquiv_apply]
    /-
      R : Type u
      inst✝² : Ring R
      J : Type v
      inst✝¹ : CategoryTheory.Category.{t, v} J
      F : CategoryTheory.Functor J (ModuleCat R)
      inst✝ : Small.{w, max v w} ↑(F.comp (CategoryTheory.forget (ModuleCat R))).sec …
      j : J
      x✝¹ : R
      x✝ : (CategoryTheory.Limits.Types.Small.limitCone (F.comp (CategoryTheory.forg …
      ⊢ Eq (↑(HSMul.hSMul x✝¹ ((equivShrink ↑(F.comp (CategoryTheory.forget (ModuleC …
    -/
    /-
      R : Type u
      inst✝² : Ring R
      J : Type v
      inst✝¹ : CategoryTheory.Category.{t, v} J
      F : CategoryTheory.Functor J (ModuleCat R)
      inst✝ : Small.{w, max v w} ↑(F.comp (CategoryTheory.forget (ModuleCat R))).sec …
      j : J
      x✝¹ x✝ : (CategoryTheory.Limits.Types.Small.limitCone (F.comp (CategoryTheory. …
      ⊢ Eq ((CategoryTheory.Limits.Types.Small.limitCone (F.comp (CategoryTheory.for …
    -/
    rfl
    /-
      R : Type u
      inst✝² : Ring R
      J : Type v
      inst✝¹ : CategoryTheory.Category.{t, v} J
      F : CategoryTheory.Functor J (ModuleCat R)
      inst✝ : Small.{w, max v w} ↑(F.comp (CategoryTheory.forget (ModuleCat R))).sec …
      j : J
      x✝¹ x✝ : (CategoryTheory.Limits.Types.Small.limitCone (F.comp (CategoryTheory. …
      ⊢ Eq (↑(HAdd.hAdd ((equivShrink ↑(F.comp (CategoryTheory.forget (ModuleCat R)) …
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
  map_add' _ _ := by
    simp only [Types.Small.limitCone_π_app, ← Equiv.addEquiv_apply, map_add]
    rfl


/-- Construction of a limit cone in `ModuleCat R`.
(Internal use only; use the limits API.)
-/
def limitCone : Cone F where
  pt := ModuleCat.of R (Types.Small.limitCone.{v, w} (F ⋙ forget _)).pt
  π :=
    { app := fun j => ofHom (limitπLinearMap F j)
      naturality := fun _ _ f => hom_ext <| LinearMap.coe_injective <|
        ((Types.Small.limitCone (F ⋙ forget _)).π.naturality f) }


/-- Witness that the limit cone in `ModuleCat R` is a limit cone.
(Internal use only; use the limits API.)
-/
def limitConeIsLimit : IsLimit (limitCone.{t, v, w} F) := by
  refine IsLimit.ofFaithful (forget (ModuleCat R)) (Types.Small.limitConeIsLimit.{v, w} _)
    (fun s => ofHom ⟨⟨(Types.Small.limitConeIsLimit.{v, w} _).lift
                ((forget (ModuleCat R)).mapCone s), ?_⟩, ?_⟩)
    (fun s => rfl)
    /-
      case refine_1
      R : Type u
      inst✝² : Ring R
      J : Type v
      inst✝¹ : CategoryTheory.Category.{t, v} J
      F : CategoryTheory.Functor J (ModuleCat R)
      inst✝ : Small.{w, max v w} ↑(F.comp (CategoryTheory.forget (ModuleCat R))).sec …
      s : CategoryTheory.Limits.Cone F
      ⊢ ∀ (x y : ↑s.1), Eq ((CategoryTheory.Limits.Types.Small.limitConeIsLimit (F.c …
    -/
  · intro x y
    /-
      case refine_1
      R : Type u
      inst✝² : Ring R
      J : Type v
      inst✝¹ : CategoryTheory.Category.{t, v} J
      F : CategoryTheory.Functor J (ModuleCat R)
      inst✝ : Small.{w, max v w} ↑(F.comp (CategoryTheory.forget (ModuleCat R))).sec …
      s : CategoryTheory.Limits.Cone F
      x y : ↑s.1
      ⊢ Eq ((CategoryTheory.Limits.Types.Small.limitConeIsLimit (F.comp (CategoryThe …
    -/
    simp only [Types.Small.limitConeIsLimit_lift, Functor.mapCone_π_app, forget_map, map_add]
    /-
      case refine_1
      R : Type u
      inst✝² : Ring R
      J : Type v
      inst✝¹ : CategoryTheory.Category.{t, v} J
      F : CategoryTheory.Functor J (ModuleCat R)
      inst✝ : Small.{w, max v w} ↑(F.comp (CategoryTheory.forget (ModuleCat R))).sec …
      s : CategoryTheory.Limits.Cone F
      x y : ↑s.1
      ⊢ Eq ((equivShrink ↑(F.comp (CategoryTheory.forget (ModuleCat R))).sections) ⟨ …
    -/
    rw [← equivShrink_add]
    /-
      case refine_1
      R : Type u
      inst✝² : Ring R
      J : Type v
      inst✝¹ : CategoryTheory.Category.{t, v} J
      F : CategoryTheory.Functor J (ModuleCat R)
      inst✝ : Small.{w, max v w} ↑(F.comp (CategoryTheory.forget (ModuleCat R))).sec …
      s : CategoryTheory.Limits.Cone F
      x y : ↑s.1
      ⊢ Eq ((equivShrink ↑(F.comp (CategoryTheory.forget (ModuleCat R))).sections) ⟨ …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u
      inst✝² : Ring R
      J : Type v
      inst✝¹ : CategoryTheory.Category.{t, v} J
      F : CategoryTheory.Functor J (ModuleCat R)
      inst✝ : Small.{w, max v w} ↑(F.comp (CategoryTheory.forget (ModuleCat R))).sec …
      s : CategoryTheory.Limits.Cone F
      ⊢ ∀ (m : R) (x : ↑s.1), Eq ({ toFun := (CategoryTheory.Limits.Types.Small.limi …
    -/
  · intro r x
    /-
      case refine_2
      R : Type u
      inst✝² : Ring R
      J : Type v
      inst✝¹ : CategoryTheory.Category.{t, v} J
      F : CategoryTheory.Functor J (ModuleCat R)
      inst✝ : Small.{w, max v w} ↑(F.comp (CategoryTheory.forget (ModuleCat R))).sec …
      s : CategoryTheory.Limits.Cone F
      r : R
      x : ↑s.1
      ⊢ Eq ({ toFun := (CategoryTheory.Limits.Types.Small.limitConeIsLimit (F.comp ( …
    -/
    simp only [Types.Small.limitConeIsLimit_lift, Functor.mapCone_π_app, forget_map, map_smul]
    /-
      case refine_2
      R : Type u
      inst✝² : Ring R
      J : Type v
      inst✝¹ : CategoryTheory.Category.{t, v} J
      F : CategoryTheory.Functor J (ModuleCat R)
      inst✝ : Small.{w, max v w} ↑(F.comp (CategoryTheory.forget (ModuleCat R))).sec …
      s : CategoryTheory.Limits.Cone F
      r : R
      x : ↑s.1
      ⊢ Eq ((equivShrink ↑(F.comp (CategoryTheory.forget (ModuleCat R))).sections) ⟨ …
    -/
    rw [← equivShrink_smul]
    /-
      case refine_2
      R : Type u
      inst✝² : Ring R
      J : Type v
      inst✝¹ : CategoryTheory.Category.{t, v} J
      F : CategoryTheory.Functor J (ModuleCat R)
      inst✝ : Small.{w, max v w} ↑(F.comp (CategoryTheory.forget (ModuleCat R))).sec …
      s : CategoryTheory.Limits.Cone F
      r : R
      x : ↑s.1
      ⊢ Eq ((equivShrink ↑(F.comp (CategoryTheory.forget (ModuleCat R))).sections) ⟨ …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- If `(F ⋙ forget (ModuleCat R)).sections` is `u`-small, `F` has a limit. -/
instance hasLimit : HasLimit F := HasLimit.mk {
    cone := limitCone F
    isLimit := limitConeIsLimit F
  }


/-- If `J` is `u`-small, the category of `R`-modules has limits of shape `J`. -/
lemma hasLimitsOfShape [Small.{w} J] : HasLimitsOfShape J (ModuleCat.{w} R) where

-- Porting note: mathport translated this as `irreducible_def`, but as `HasLimitsOfSize`
-- is a `Prop`, declaring this as `irreducible` should presumably have no effect

/-- The category of R-modules has all limits. -/
lemma hasLimitsOfSize [UnivLE.{v, w}] : HasLimitsOfSize.{t, v} (ModuleCat.{w} R) where
  has_limits_of_shape _ := hasLimitsOfShape


instance hasLimits : HasLimits (ModuleCat.{w} R) :=
  ModuleCat.hasLimitsOfSize.{w, w, w, u}


instance (priority := high) hasLimits' : HasLimits (ModuleCat.{u} R) :=
  ModuleCat.hasLimitsOfSize.{u, u, u}


/-- An auxiliary declaration to speed up typechecking.
-/
def forget₂AddCommGroup_preservesLimitsAux :
    IsLimit ((forget₂ (ModuleCat R) AddCommGrp).mapCone (limitCone F)) :=
  letI : Small.{w} (Functor.sections ((F ⋙ forget₂ _ AddCommGrp) ⋙ forget _)) :=
    inferInstanceAs <| Small.{w} (Functor.sections (F ⋙ forget (ModuleCat R)))
  AddCommGrp.limitConeIsLimit
    (F ⋙ forget₂ (ModuleCat.{w} R) _ : J ⥤ AddCommGrp.{w})


/-- The forgetful functor from R-modules to abelian groups preserves all limits. -/
instance forget₂AddCommGroup_preservesLimit :
    PreservesLimit F (forget₂ (ModuleCat R) AddCommGrp) :=
  preservesLimit_of_preserves_limit_cone (limitConeIsLimit F)
    (forget₂AddCommGroup_preservesLimitsAux F)


/-- The forgetful functor from R-modules to abelian groups preserves all limits.
-/
instance forget₂AddCommGroup_preservesLimitsOfSize [UnivLE.{v, w}] :
    PreservesLimitsOfSize.{t, v}
      (forget₂ (ModuleCat.{w} R) AddCommGrp.{w}) where
  preservesLimitsOfShape := { preservesLimit := inferInstance }


instance forget₂AddCommGroup_preservesLimits :
    PreservesLimits (forget₂ (ModuleCat R) AddCommGrp.{w}) :=
  ModuleCat.forget₂AddCommGroup_preservesLimitsOfSize.{w, w}


/-- The forgetful functor from R-modules to types preserves all limits.
-/
instance forget_preservesLimitsOfSize [UnivLE.{v, w}] :
    PreservesLimitsOfSize.{t, v} (forget (ModuleCat.{w} R)) where
  preservesLimitsOfShape :=
    { preservesLimit := fun {K} ↦ preservesLimit_of_preserves_limit_cone (limitConeIsLimit K)
        (Types.Small.limitConeIsLimit.{v} (_ ⋙ forget _)) }


instance forget_preservesLimits : PreservesLimits (forget (ModuleCat.{w} R)) :=
  ModuleCat.forget_preservesLimitsOfSize.{w, w}


instance forget₂AddCommGroup_reflectsLimit :
    ReflectsLimit F (forget₂ (ModuleCat.{w} R) AddCommGrp) where
  reflects {c} hc := ⟨by
    /-
      R : Type u
      inst✝¹ : Ring R
      J : Type v
      inst✝ : CategoryTheory.Category.{t, v} J
      F : CategoryTheory.Functor J (ModuleCat R)
      c : CategoryTheory.Limits.Cone F
      hc : CategoryTheory.Limits.IsLimit ((CategoryTheory.forget₂ (ModuleCat R) AddC …
      ⊢ CategoryTheory.Limits.IsLimit c
    -/
    have : HasLimit (F ⋙ forget₂ (ModuleCat R) AddCommGrp) := ⟨_, hc⟩
    have : Small.{w} (Functor.sections (F ⋙ forget (ModuleCat R))) := by
      simpa only [AddCommGrp.hasLimit_iff_small_sections] using this
    /-
      R : Type u
      inst✝¹ : Ring R
      J : Type v
      inst✝ : CategoryTheory.Category.{t, v} J
      F : CategoryTheory.Functor J (ModuleCat R)
      c : CategoryTheory.Limits.Cone F
      hc : CategoryTheory.Limits.IsLimit ((CategoryTheory.forget₂ (ModuleCat R) AddC …
      this✝ : CategoryTheory.Limits.HasLimit (F.comp (CategoryTheory.forget₂ (Module …
      this : Small.{w, max v w} ↑(F.comp (CategoryTheory.forget (ModuleCat R))).sect …
      ⊢ CategoryTheory.Limits.IsLimit c
    -/
    have := reflectsLimit_of_reflectsIsomorphisms F (forget₂ (ModuleCat R) AddCommGrp)
    /-
      R : Type u
      inst✝¹ : Ring R
      J : Type v
      inst✝ : CategoryTheory.Category.{t, v} J
      F : CategoryTheory.Functor J (ModuleCat R)
      c : CategoryTheory.Limits.Cone F
      hc : CategoryTheory.Limits.IsLimit ((CategoryTheory.forget₂ (ModuleCat R) AddC …
      this✝¹ : CategoryTheory.Limits.HasLimit (F.comp (CategoryTheory.forget₂ (Modul …
      this✝ : Small.{w, max v w} ↑(F.comp (CategoryTheory.forget (ModuleCat R))).sec …
      this : CategoryTheory.Limits.ReflectsLimit F (CategoryTheory.forget₂ (ModuleCa …
      ⊢ CategoryTheory.Limits.IsLimit c
    -/
    exact isLimitOfReflects _ hc⟩
    /-
      🎉 no goals
    -/


instance forget₂AddCommGroup_reflectsLimitOfShape :
    ReflectsLimitsOfShape J (forget₂ (ModuleCat.{w} R) AddCommGrp) where


instance forget₂AddCommGroup_reflectsLimitOfSize :
    ReflectsLimitsOfSize.{t, v} (forget₂ (ModuleCat.{w} R) AddCommGrp) where


/-- The diagram (in the sense of `CategoryTheory`)
 of an unbundled `directLimit` of modules. -/
@[simps]
def directLimitDiagram : ι ⥤ ModuleCat R where
  obj i := ModuleCat.of R (G i)
  map hij := ofHom (f _ _ hij.le)
  map_id i := by
    /-
      R : Type u
      inst✝⁶ : Ring R
      J : Type v
      inst✝⁵ : CategoryTheory.Category.{t, v} J
      F : CategoryTheory.Functor J (ModuleCat R)
      ι : Type v
      inst✝⁴ : DecidableEq ι
      inst✝³ : Preorder ι
      G : ι → Type v
      inst✝² : (i : ι) → AddCommGroup (G i)
      inst✝¹ : (i : ι) → Module R (G i)
      f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
      inst✝ : DirectedSystem G fun i j h => ⇑(f i j h)
      i : ι
      ⊢ Eq ({ obj := fun i => ModuleCat.of R (G i), map := fun {X Y} hij => ModuleCa …
    -/
    ext
    /-
      case hf.h
      R : Type u
      inst✝⁶ : Ring R
      J : Type v
      inst✝⁵ : CategoryTheory.Category.{t, v} J
      F : CategoryTheory.Functor J (ModuleCat R)
      ι : Type v
      inst✝⁴ : DecidableEq ι
      inst✝³ : Preorder ι
      G : ι → Type v
      inst✝² : (i : ι) → AddCommGroup (G i)
      inst✝¹ : (i : ι) → Module R (G i)
      f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
      inst✝ : DirectedSystem G fun i j h => ⇑(f i j h)
      i : ι
      x✝ : ↑({ obj := fun i => ModuleCat.of R (G i), map := fun {X Y} hij => ModuleC …
      ⊢ Eq (({ obj := fun i => ModuleCat.of R (G i), map := fun {X Y} hij => ModuleC …
    -/
    apply Module.DirectedSystem.map_self
    /-
      🎉 no goals
    -/
  map_comp hij hjk := by
    /-
      R : Type u
      inst✝⁶ : Ring R
      J : Type v
      inst✝⁵ : CategoryTheory.Category.{t, v} J
      F : CategoryTheory.Functor J (ModuleCat R)
      ι : Type v
      inst✝⁴ : DecidableEq ι
      inst✝³ : Preorder ι
      G : ι → Type v
      inst✝² : (i : ι) → AddCommGroup (G i)
      inst✝¹ : (i : ι) → Module R (G i)
      f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
      inst✝ : DirectedSystem G fun i j h => ⇑(f i j h)
      X✝ Y✝ Z✝ : ι
      hij : Quiver.Hom X✝ Y✝
      hjk : Quiver.Hom Y✝ Z✝
      ⊢ Eq ({ obj := fun i => ModuleCat.of R (G i), map := fun {X Y} hij => ModuleCa …
    -/
    ext
    /-
      case hf.h
      R : Type u
      inst✝⁶ : Ring R
      J : Type v
      inst✝⁵ : CategoryTheory.Category.{t, v} J
      F : CategoryTheory.Functor J (ModuleCat R)
      ι : Type v
      inst✝⁴ : DecidableEq ι
      inst✝³ : Preorder ι
      G : ι → Type v
      inst✝² : (i : ι) → AddCommGroup (G i)
      inst✝¹ : (i : ι) → Module R (G i)
      f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
      inst✝ : DirectedSystem G fun i j h => ⇑(f i j h)
      X✝ Y✝ Z✝ : ι
      hij : Quiver.Hom X✝ Y✝
      hjk : Quiver.Hom Y✝ Z✝
      x✝ : ↑({ obj := fun i => ModuleCat.of R (G i), map := fun {X Y} hij => ModuleC …
      ⊢ Eq (({ obj := fun i => ModuleCat.of R (G i), map := fun {X Y} hij => ModuleC …
    -/
    symm
    /-
      case hf.h
      R : Type u
      inst✝⁶ : Ring R
      J : Type v
      inst✝⁵ : CategoryTheory.Category.{t, v} J
      F : CategoryTheory.Functor J (ModuleCat R)
      ι : Type v
      inst✝⁴ : DecidableEq ι
      inst✝³ : Preorder ι
      G : ι → Type v
      inst✝² : (i : ι) → AddCommGroup (G i)
      inst✝¹ : (i : ι) → Module R (G i)
      f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
      inst✝ : DirectedSystem G fun i j h => ⇑(f i j h)
      X✝ Y✝ Z✝ : ι
      hij : Quiver.Hom X✝ Y✝
      hjk : Quiver.Hom Y✝ Z✝
      x✝ : ↑({ obj := fun i => ModuleCat.of R (G i), map := fun {X Y} hij => ModuleC …
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp ({ obj := fun i => ModuleCat.of R (G …
    -/
    apply Module.DirectedSystem.map_map f
    /-
      🎉 no goals
    -/


/-- The `Cocone` on `directLimitDiagram` corresponding to
the unbundled `directLimit` of modules.

In `directLimitIsColimit` we show that it is a colimit cocone. -/
@[simps]
def directLimitCocone : Cocone (directLimitDiagram G f) where
  pt := ModuleCat.of R <| DirectLimit G f
  ι :=
    { app := fun x => ofHom (Module.DirectLimit.of R ι G f x)
      naturality := fun _ _ hij => by
        /-
          R : Type u
          inst✝⁷ : Ring R
          J : Type v
          inst✝⁶ : CategoryTheory.Category.{t, v} J
          F : CategoryTheory.Functor J (ModuleCat R)
          ι : Type v
          inst✝⁵ : DecidableEq ι
          inst✝⁴ : Preorder ι
          G : ι → Type v
          inst✝³ : (i : ι) → AddCommGroup (G i)
          inst✝² : (i : ι) → Module R (G i)
          f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
          inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
          inst✝ : DecidableEq ι
          x✝¹ x✝ : ι
          hij : Quiver.Hom x✝¹ x✝
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((ModuleCat.directLimitDiagram G f).m …
        -/
        ext
        /-
          case hf.h
          R : Type u
          inst✝⁷ : Ring R
          J : Type v
          inst✝⁶ : CategoryTheory.Category.{t, v} J
          F : CategoryTheory.Functor J (ModuleCat R)
          ι : Type v
          inst✝⁵ : DecidableEq ι
          inst✝⁴ : Preorder ι
          G : ι → Type v
          inst✝³ : (i : ι) → AddCommGroup (G i)
          inst✝² : (i : ι) → Module R (G i)
          f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
          inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
          inst✝ : DecidableEq ι
          x✝² x✝¹ : ι
          hij : Quiver.Hom x✝² x✝¹
          x✝ : ↑((ModuleCat.directLimitDiagram G f).obj x✝²)
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((ModuleCat.directLimitDiagram G f). …
        -/
        exact DirectLimit.of_f }
        /-
          🎉 no goals
        -/


/-- The unbundled `directLimit` of modules is a colimit
in the sense of `CategoryTheory`. -/
@[simps]
def directLimitIsColimit : IsColimit (directLimitCocone G f) where
  desc s := ofHom <|
    Module.DirectLimit.lift R ι G f (fun i => (s.ι.app i).hom) fun i j h x => by
      /-
        R : Type u
        inst✝⁷ : Ring R
        J : Type v
        inst✝⁶ : CategoryTheory.Category.{t, v} J
        F : CategoryTheory.Functor J (ModuleCat R)
        ι : Type v
        inst✝⁵ : DecidableEq ι
        inst✝⁴ : Preorder ι
        G : ι → Type v
        inst✝³ : (i : ι) → AddCommGroup (G i)
        inst✝² : (i : ι) → Module R (G i)
        f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
        inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
        inst✝ : DecidableEq ι
        s : CategoryTheory.Limits.Cocone (ModuleCat.directLimitDiagram G f)
        i j : ι
        h : LE.le i j
        x : G i
        ⊢ Eq (((fun i => (s.ι.app i).hom) j) ((f i j h) x)) (((fun i => (s.ι.app i).ho …
      -/
      simp only [Functor.const_obj_obj]
      /-
        R : Type u
        inst✝⁷ : Ring R
        J : Type v
        inst✝⁶ : CategoryTheory.Category.{t, v} J
        F : CategoryTheory.Functor J (ModuleCat R)
        ι : Type v
        inst✝⁵ : DecidableEq ι
        inst✝⁴ : Preorder ι
        G : ι → Type v
        inst✝³ : (i : ι) → AddCommGroup (G i)
        inst✝² : (i : ι) → Module R (G i)
        f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
        inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
        inst✝ : DecidableEq ι
        s : CategoryTheory.Limits.Cocone (ModuleCat.directLimitDiagram G f)
        i j : ι
        h : LE.le i j
        x : G i
        ⊢ Eq ((s.ι.app j).hom ((f i j h) x)) ((s.ι.app i).hom x)
      -/
      rw [← s.w (homOfLE h)]
      /-
        R : Type u
        inst✝⁷ : Ring R
        J : Type v
        inst✝⁶ : CategoryTheory.Category.{t, v} J
        F : CategoryTheory.Functor J (ModuleCat R)
        ι : Type v
        inst✝⁵ : DecidableEq ι
        inst✝⁴ : Preorder ι
        G : ι → Type v
        inst✝³ : (i : ι) → AddCommGroup (G i)
        inst✝² : (i : ι) → Module R (G i)
        f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
        inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
        inst✝ : DecidableEq ι
        s : CategoryTheory.Limits.Cocone (ModuleCat.directLimitDiagram G f)
        i j : ι
        h : LE.le i j
        x : G i
        ⊢ Eq ((s.ι.app j).hom ((f i j h) x)) ((CategoryTheory.CategoryStruct.comp ((Mo …
      -/
      rfl
      /-
        🎉 no goals
      -/
  fac s i := by
    /-
      R : Type u
      inst✝⁷ : Ring R
      J : Type v
      inst✝⁶ : CategoryTheory.Category.{t, v} J
      F : CategoryTheory.Functor J (ModuleCat R)
      ι : Type v
      inst✝⁵ : DecidableEq ι
      inst✝⁴ : Preorder ι
      G : ι → Type v
      inst✝³ : (i : ι) → AddCommGroup (G i)
      inst✝² : (i : ι) → Module R (G i)
      f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
      inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
      inst✝ : DecidableEq ι
      s : CategoryTheory.Limits.Cocone (ModuleCat.directLimitDiagram G f)
      i : ι
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((ModuleCat.directLimitCocone G f).ι. …
    -/
    ext
    /-
      case hf.h
      R : Type u
      inst✝⁷ : Ring R
      J : Type v
      inst✝⁶ : CategoryTheory.Category.{t, v} J
      F : CategoryTheory.Functor J (ModuleCat R)
      ι : Type v
      inst✝⁵ : DecidableEq ι
      inst✝⁴ : Preorder ι
      G : ι → Type v
      inst✝³ : (i : ι) → AddCommGroup (G i)
      inst✝² : (i : ι) → Module R (G i)
      f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
      inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
      inst✝ : DecidableEq ι
      s : CategoryTheory.Limits.Cocone (ModuleCat.directLimitDiagram G f)
      i : ι
      x✝ : ↑((ModuleCat.directLimitDiagram G f).obj i)
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((ModuleCat.directLimitCocone G f).ι …
    -/
    dsimp only [directLimitCocone, CategoryStruct.comp]
    /-
      case hf.h
      R : Type u
      inst✝⁷ : Ring R
      J : Type v
      inst✝⁶ : CategoryTheory.Category.{t, v} J
      F : CategoryTheory.Functor J (ModuleCat R)
      ι : Type v
      inst✝⁵ : DecidableEq ι
      inst✝⁴ : Preorder ι
      G : ι → Type v
      inst✝³ : (i : ι) → AddCommGroup (G i)
      inst✝² : (i : ι) → Module R (G i)
      f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
      inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
      inst✝ : DecidableEq ι
      s : CategoryTheory.Limits.Cocone (ModuleCat.directLimitDiagram G f)
      i : ι
      x✝ : ↑((ModuleCat.directLimitDiagram G f).obj i)
      ⊢ Eq (((Module.DirectLimit.lift R ι G f (fun i => (s.ι.app i).hom) ⋯).comp (Mo …
    -/
    rw [LinearMap.comp_apply]
    /-
      case hf.h
      R : Type u
      inst✝⁷ : Ring R
      J : Type v
      inst✝⁶ : CategoryTheory.Category.{t, v} J
      F : CategoryTheory.Functor J (ModuleCat R)
      ι : Type v
      inst✝⁵ : DecidableEq ι
      inst✝⁴ : Preorder ι
      G : ι → Type v
      inst✝³ : (i : ι) → AddCommGroup (G i)
      inst✝² : (i : ι) → Module R (G i)
      f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
      inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
      inst✝ : DecidableEq ι
      s : CategoryTheory.Limits.Cocone (ModuleCat.directLimitDiagram G f)
      i : ι
      x✝ : ↑((ModuleCat.directLimitDiagram G f).obj i)
      ⊢ Eq ((Module.DirectLimit.lift R ι G f (fun i => (s.ι.app i).hom) ⋯) ((Module. …
    -/
    apply DirectLimit.lift_of
    /-
      🎉 no goals
    -/
  uniq s m h := by
    have :
      s.ι.app = fun i =>
        (ofHom (DirectLimit.of R ι (fun i => G i) (fun i j H => f i j H) i)) ≫ m := by
      funext i
      rw [← h]
      rfl
    /-
      R : Type u
      inst✝⁷ : Ring R
      J : Type v
      inst✝⁶ : CategoryTheory.Category.{t, v} J
      F : CategoryTheory.Functor J (ModuleCat R)
      ι : Type v
      inst✝⁵ : DecidableEq ι
      inst✝⁴ : Preorder ι
      G : ι → Type v
      inst✝³ : (i : ι) → AddCommGroup (G i)
      inst✝² : (i : ι) → Module R (G i)
      f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
      inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
      inst✝ : DecidableEq ι
      s : CategoryTheory.Limits.Cocone (ModuleCat.directLimitDiagram G f)
      m : Quiver.Hom (ModuleCat.directLimitCocone G f).pt s.pt
      h : ∀ (j : ι), Eq (CategoryTheory.CategoryStruct.comp ((ModuleCat.directLimitC …
      this : Eq s.ι.app fun i => CategoryTheory.CategoryStruct.comp (ModuleCat.ofHom …
      ⊢ Eq m ((fun s => ModuleCat.ofHom (Module.DirectLimit.lift R ι G f (fun i => ( …
    -/
    ext
    /-
      case hf.h
      R : Type u
      inst✝⁷ : Ring R
      J : Type v
      inst✝⁶ : CategoryTheory.Category.{t, v} J
      F : CategoryTheory.Functor J (ModuleCat R)
      ι : Type v
      inst✝⁵ : DecidableEq ι
      inst✝⁴ : Preorder ι
      G : ι → Type v
      inst✝³ : (i : ι) → AddCommGroup (G i)
      inst✝² : (i : ι) → Module R (G i)
      f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
      inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
      inst✝ : DecidableEq ι
      s : CategoryTheory.Limits.Cocone (ModuleCat.directLimitDiagram G f)
      m : Quiver.Hom (ModuleCat.directLimitCocone G f).pt s.pt
      h : ∀ (j : ι), Eq (CategoryTheory.CategoryStruct.comp ((ModuleCat.directLimitC …
      this : Eq s.ι.app fun i => CategoryTheory.CategoryStruct.comp (ModuleCat.ofHom …
      x✝ : ↑(ModuleCat.directLimitCocone G f).pt
      ⊢ Eq (m.hom x✝) (((fun s => ModuleCat.ofHom (Module.DirectLimit.lift R ι G f ( …
    -/
    simp only [this]
    /-
      case hf.h
      R : Type u
      inst✝⁷ : Ring R
      J : Type v
      inst✝⁶ : CategoryTheory.Category.{t, v} J
      F : CategoryTheory.Functor J (ModuleCat R)
      ι : Type v
      inst✝⁵ : DecidableEq ι
      inst✝⁴ : Preorder ι
      G : ι → Type v
      inst✝³ : (i : ι) → AddCommGroup (G i)
      inst✝² : (i : ι) → Module R (G i)
      f : (i j : ι) → LE.le i j → LinearMap (RingHom.id R) (G i) (G j)
      inst✝¹ : DirectedSystem G fun i j h => ⇑(f i j h)
      inst✝ : DecidableEq ι
      s : CategoryTheory.Limits.Cocone (ModuleCat.directLimitDiagram G f)
      m : Quiver.Hom (ModuleCat.directLimitCocone G f).pt s.pt
      h : ∀ (j : ι), Eq (CategoryTheory.CategoryStruct.comp ((ModuleCat.directLimitC …
      this : Eq s.ι.app fun i => CategoryTheory.CategoryStruct.comp (ModuleCat.ofHom …
      x✝ : ↑(ModuleCat.directLimitCocone G f).pt
      ⊢ Eq (m.hom x✝) ((Module.DirectLimit.lift R ι G f (fun i => (CategoryTheory.Ca …
    -/
    apply Module.DirectLimit.lift_unique
    /-
      🎉 no goals
    -/


