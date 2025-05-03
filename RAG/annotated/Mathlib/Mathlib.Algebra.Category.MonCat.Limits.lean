/-- An alias for `MonCat.{max u v}`, to deal around unification issues. -/
@[to_additive (attr := nolint checkUnivs) AddMonCatMax
  "An alias for `AddMonCat.{max u v}`, to deal around unification issues."]
abbrev MonCatMax.{u1, u2} := MonCat.{max u1 u2}


@[to_additive]
instance monoidObj (j) : Monoid ((F ⋙ forget MonCat).obj j) :=
  inferInstanceAs <| Monoid (F.obj j)


/-- The flat sections of a functor into `MonCat` form a submonoid of all sections.
-/
@[to_additive
      "The flat sections of a functor into `AddMonCat` form an additive submonoid of all sections."]
def sectionsSubmonoid : Submonoid (∀ j, F.obj j) where
  carrier := (F ⋙ forget MonCat).sections
                            /-
                              J : Type v
                              inst✝ : CategoryTheory.Category.{w, v} J
                              F : CategoryTheory.Functor J MonCat
                              j j' : J
                              f : Quiver.Hom j j'
                              ⊢ Eq ((F.comp (CategoryTheory.forget MonCat)).map f (1 j)) (1 j')
                            -/
  one_mem' {j} {j'} f := by simp
    /-
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J MonCat
      a b : (j : J) → ↑(F.obj j)
      ah : Membership.mem (F.comp (CategoryTheory.forget MonCat)).sections a
      bh : Membership.mem (F.comp (CategoryTheory.forget MonCat)).sections b
      j j' : J
      f : Quiver.Hom j j'
      ⊢ Eq ((F.comp (CategoryTheory.forget MonCat)).map f (HMul.hMul a b j)) (HMul.h …
    -/
                            /-
                              🎉 no goals
                            -/
    /-
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J MonCat
      a b : (j : J) → ↑(F.obj j)
      ah : Membership.mem (F.comp (CategoryTheory.forget MonCat)).sections a
      bh : Membership.mem (F.comp (CategoryTheory.forget MonCat)).sections b
      j j' : J
      f : Quiver.Hom j j'
      ⊢ Eq ((CategoryTheory.forget MonCat).map (F.map f) (HMul.hMul (a j) (b j))) (H …
    -/
  mul_mem' {a} {b} ah bh {j} {j'} f := by
    /-
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J MonCat
      a b : (j : J) → ↑(F.obj j)
      ah : ∀ {j j' : J} (f : Quiver.Hom j j'), Eq ((F.map f) (a j)) (a j')
      bh : ∀ {j j' : J} (f : Quiver.Hom j j'), Eq ((F.map f) (b j)) (b j')
      j j' : J
      f : Quiver.Hom j j'
      ⊢ Eq ((CategoryTheory.forget MonCat).map (F.map f) (HMul.hMul (a j) (b j))) (H …
    -/
    simp only [Functor.comp_map, MonoidHom.map_mul, Pi.mul_apply]
    /-
      🎉 no goals
    -/
    dsimp [Functor.sections] at ah bh
    rw [← ah f, ← bh f, forget_map, map_mul]


@[to_additive]
instance sectionsMonoid : Monoid (F ⋙ forget MonCat.{u}).sections :=
  (sectionsSubmonoid F).toMonoid


@[to_additive]
noncomputable instance limitMonoid :
    Monoid (Types.Small.limitCone.{v, u} (F ⋙ forget MonCat.{u})).pt :=
  inferInstanceAs <| Monoid (Shrink (F ⋙ forget MonCat.{u}).sections)


/-- `limit.π (F ⋙ forget MonCat) j` as a `MonoidHom`. -/
@[to_additive "`limit.π (F ⋙ forget AddMonCat) j` as an `AddMonoidHom`."]
noncomputable def limitπMonoidHom (j : J) :
    (Types.Small.limitCone.{v, u} (F ⋙ forget MonCat.{u})).pt →*
      ((F ⋙ forget MonCat.{u}).obj j) where
  toFun := (Types.Small.limitCone.{v, u} (F ⋙ forget MonCat.{u})).π.app j
  map_one' := by
    /-
      J : Type v
      inst✝¹ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J MonCat
      inst✝ : Small.{u, max u v} ↑(F.comp (CategoryTheory.forget MonCat)).sections
      j : J
      ⊢ Eq ((CategoryTheory.Limits.Types.Small.limitCone (F.comp (CategoryTheory.for …
    -/
    simp only [Types.Small.limitCone_pt, Types.Small.limitCone_π_app, equivShrink_symm_one]
    /-
      J : Type v
      inst✝¹ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J MonCat
      inst✝ : Small.{u, max u v} ↑(F.comp (CategoryTheory.forget MonCat)).sections
      j : J
      ⊢ Eq (↑1 j) 1
    -/
    rfl
    /-
      🎉 no goals
    -/
  map_mul' _ _ := by
    /-
      J : Type v
      inst✝¹ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J MonCat
      inst✝ : Small.{u, max u v} ↑(F.comp (CategoryTheory.forget MonCat)).sections
      j : J
      x✝¹ x✝ : (CategoryTheory.Limits.Types.Small.limitCone (F.comp (CategoryTheory. …
      ⊢ Eq ({ toFun := (CategoryTheory.Limits.Types.Small.limitCone (F.comp (Categor …
    -/
    simp only [Types.Small.limitCone_pt, Types.Small.limitCone_π_app, equivShrink_symm_mul]
    /-
      J : Type v
      inst✝¹ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J MonCat
      inst✝ : Small.{u, max u v} ↑(F.comp (CategoryTheory.forget MonCat)).sections
      j : J
      x✝¹ x✝ : (CategoryTheory.Limits.Types.Small.limitCone (F.comp (CategoryTheory. …
      ⊢ Eq (↑(HMul.hMul ((equivShrink ↑(F.comp (CategoryTheory.forget MonCat)).secti …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- Construction of a limit cone in `MonCat`.
(Internal use only; use the limits API.)
-/
@[to_additive "(Internal use only; use the limits API.)"]
noncomputable def limitCone : Cone F :=
  { pt := MonCat.of (Types.Small.limitCone (F ⋙ forget _)).pt
    π :=
    { app := limitπMonoidHom F
      naturality := fun _ _ f =>
        DFunLike.coe_injective ((Types.Small.limitCone (F ⋙ forget _)).π.naturality f) } }


/-- Witness that the limit cone in `MonCat` is a limit cone.
(Internal use only; use the limits API.)
-/
@[to_additive "(Internal use only; use the limits API.)"]
noncomputable def limitConeIsLimit : IsLimit (limitCone F) := by
  refine IsLimit.ofFaithful (forget MonCat) (Types.Small.limitConeIsLimit.{v,u} _)
    (fun s => { toFun := _, map_one' := ?_, map_mul' := ?_ }) (fun s => rfl)
    /-
      case refine_1
      J : Type v
      inst✝¹ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J MonCat
      inst✝ : Small.{u, max u v} ↑(F.comp (CategoryTheory.forget MonCat)).sections
      s : CategoryTheory.Limits.Cone F
      ⊢ Eq ((equivShrink ↑(F.comp (CategoryTheory.forget MonCat)).sections).1 ⟨fun j …
    -/
  · simp only [Functor.mapCone_π_app, forget_map, map_one]
    /-
      case refine_1
      J : Type v
      inst✝¹ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J MonCat
      inst✝ : Small.{u, max u v} ↑(F.comp (CategoryTheory.forget MonCat)).sections
      s : CategoryTheory.Limits.Cone F
      ⊢ Eq ((equivShrink ↑(F.comp (CategoryTheory.forget MonCat)).sections).1 ⟨fun j …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      J : Type v
      inst✝¹ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J MonCat
      inst✝ : Small.{u, max u v} ↑(F.comp (CategoryTheory.forget MonCat)).sections
      s : CategoryTheory.Limits.Cone F
      ⊢ ∀ (x y : ↑s.pt), Eq ({ toFun := fun v => (equivShrink ↑(F.comp (CategoryTheo …
    -/
  · intro x y
    /-
      case refine_2
      J : Type v
      inst✝¹ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J MonCat
      inst✝ : Small.{u, max u v} ↑(F.comp (CategoryTheory.forget MonCat)).sections
      s : CategoryTheory.Limits.Cone F
      x y : ↑s.pt
      ⊢ Eq ({ toFun := fun v => (equivShrink ↑(F.comp (CategoryTheory.forget MonCat) …
    -/
    simp only [Functor.mapCone_π_app, forget_map, map_mul, Functor.comp_obj, Equiv.toFun_as_coe]
    /-
      case refine_2
      J : Type v
      inst✝¹ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J MonCat
      inst✝ : Small.{u, max u v} ↑(F.comp (CategoryTheory.forget MonCat)).sections
      s : CategoryTheory.Limits.Cone F
      x y : ↑s.pt
      ⊢ Eq ((equivShrink ↑(F.comp (CategoryTheory.forget MonCat)).sections) ⟨fun j = …
    -/
    rw [← equivShrink_mul]
    /-
      case refine_2
      J : Type v
      inst✝¹ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J MonCat
      inst✝ : Small.{u, max u v} ↑(F.comp (CategoryTheory.forget MonCat)).sections
      s : CategoryTheory.Limits.Cone F
      x y : ↑s.pt
      ⊢ Eq ((equivShrink ↑(F.comp (CategoryTheory.forget MonCat)).sections) ⟨fun j = …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- If `(F ⋙ forget MonCat).sections` is `u`-small, `F` has a limit. -/
@[to_additive "If `(F ⋙ forget AddMonCat).sections` is `u`-small, `F` has a limit."]
instance hasLimit : HasLimit F :=
  HasLimit.mk {
    cone := limitCone F
    isLimit := limitConeIsLimit F
  }


/-- If `J` is `u`-small, `MonCat.{u}` has limits of shape `J`. -/
@[to_additive "If `J` is `u`-small, `AddMonCat.{u}` has limits of shape `J`."]
instance hasLimitsOfShape [Small.{u} J] : HasLimitsOfShape J MonCat.{u} where
  has_limit _ := inferInstance


/-- The category of monoids has all limits. -/
@[to_additive "The category of additive monoids has all limits.",
  to_additive_relevant_arg 2]
instance hasLimitsOfSize [UnivLE.{v, u}] : HasLimitsOfSize.{w, v} MonCat.{u} where
  has_limits_of_shape _ _ := { }


@[to_additive]
instance hasLimits : HasLimits MonCat.{u} :=
  MonCat.hasLimitsOfSize.{u, u}


/-- If `J` is `u`-small, the forgetful functor from `MonCat.{u}` preserves limits of shape `J`. -/
@[to_additive "If `J` is `u`-small, the forgetful functor from `AddMonCat.{u}`\n
preserves limits of shape `J`."]
noncomputable instance forget_preservesLimitsOfShape [Small.{u} J] :
    PreservesLimitsOfShape J (forget MonCat.{u}) where
  preservesLimit {F} := preservesLimit_of_preserves_limit_cone (limitConeIsLimit F)
    (Types.Small.limitConeIsLimit (F ⋙ forget _))


/-- The forgetful functor from monoids to types preserves all limits.

This means the underlying type of a limit can be computed as a limit in the category of types. -/
@[to_additive
  "The forgetful functor from additive monoids to types preserves all limits.\n\n
  This means the underlying type of a limit can be computed as a limit in the category of types.",
  to_additive_relevant_arg 2]
noncomputable instance forget_preservesLimitsOfSize [UnivLE.{v, u}] :
    PreservesLimitsOfSize.{w, v} (forget MonCat.{u}) where
  preservesLimitsOfShape := { }


@[to_additive]
noncomputable instance forget_preservesLimits : PreservesLimits (forget MonCat.{u}) :=
  MonCat.forget_preservesLimitsOfSize.{u, u}


@[to_additive]
noncomputable instance forget_createsLimit :
    CreatesLimit F (forget MonCat.{u}) := by
  /-
    J : Type v
    inst✝¹ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J MonCat
    inst✝ : Small.{u, max u v} ↑(F.comp (CategoryTheory.forget MonCat)).sections
    ⊢ CategoryTheory.CreatesLimit F (CategoryTheory.forget MonCat)
  -/
  apply createsLimitOfReflectsIso
  /-
    case h
    J : Type v
    inst✝¹ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J MonCat
    inst✝ : Small.{u, max u v} ↑(F.comp (CategoryTheory.forget MonCat)).sections
    ⊢ (c : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.forget MonCat))) → ( …
  -/
  intro c t
  have : Small.{u} (Functor.sections (F ⋙ forget MonCat)) :=
    (Types.hasLimit_iff_small_sections _).mp (HasLimit.mk {cone := c, isLimit := t})
  refine LiftsToLimit.mk (LiftableCone.mk
    {pt := MonCat.of (Types.Small.limitCone (F ⋙ forget MonCat)).pt, π := NatTrans.mk
      (limitπMonoidHom F) (MonCat.HasLimits.limitCone F).π.naturality} (Cones.ext
      ((Types.isLimitEquivSections t).trans (equivShrink _)).symm.toIso
      (fun _ ↦ funext (fun _ ↦ by simp; rfl)))) ?_
  /-
    case h
    J : Type v
    inst✝¹ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J MonCat
    inst✝ : Small.{u, max u v} ↑(F.comp (CategoryTheory.forget MonCat)).sections
    c : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.forget MonCat))
    t : CategoryTheory.Limits.IsLimit c
    this : Small.{u, max u v} ↑(F.comp (CategoryTheory.forget MonCat)).sections
    ⊢ CategoryTheory.Limits.IsLimit { liftedCone := { pt := MonCat.of (CategoryThe …
  -/
  refine IsLimit.ofFaithful (forget MonCat.{u}) (Types.Small.limitConeIsLimit.{v,u} _) ?_ ?_
    /-
      case h.refine_1
      J : Type v
      inst✝¹ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J MonCat
      inst✝ : Small.{u, max u v} ↑(F.comp (CategoryTheory.forget MonCat)).sections
      c : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.forget MonCat))
      t : CategoryTheory.Limits.IsLimit c
      this : Small.{u, max u v} ↑(F.comp (CategoryTheory.forget MonCat)).sections
      ⊢ (s : CategoryTheory.Limits.Cone F) → Quiver.Hom s.pt { liftedCone := { pt := …
    -/
  · intro _
    refine {toFun := (Types.Small.limitConeIsLimit.{v,u} _).lift ((forget MonCat).mapCone _),
                      map_one' := by simp; rfl, map_mul' := ?_ }
      /-
        case h.refine_1
        J : Type v
        inst✝¹ : CategoryTheory.Category.{w, v} J
        F : CategoryTheory.Functor J MonCat
        inst✝ : Small.{u, max u v} ↑(F.comp (CategoryTheory.forget MonCat)).sections
        c : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.forget MonCat))
        t : CategoryTheory.Limits.IsLimit c
        this : Small.{u, max u v} ↑(F.comp (CategoryTheory.forget MonCat)).sections
        s✝ : CategoryTheory.Limits.Cone F
        ⊢ ∀ (x y : ↑s✝.pt), Eq ({ toFun := (CategoryTheory.Limits.Types.Small.limitCon …
      -/
    · intro x y
      simp only [Types.Small.limitConeIsLimit_lift, Functor.comp_obj, Functor.mapCone_pt,
          Functor.mapCone_π_app, forget_map, map_mul, mul_of]
      /-
        case h.refine_1
        J : Type v
        inst✝¹ : CategoryTheory.Category.{w, v} J
        F : CategoryTheory.Functor J MonCat
        inst✝ : Small.{u, max u v} ↑(F.comp (CategoryTheory.forget MonCat)).sections
        c : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.forget MonCat))
        t : CategoryTheory.Limits.IsLimit c
        this : Small.{u, max u v} ↑(F.comp (CategoryTheory.forget MonCat)).sections
        s✝ : CategoryTheory.Limits.Cone F
        x y : ↑s✝.pt
        ⊢ Eq ((equivShrink ↑(F.comp (CategoryTheory.forget MonCat)).sections) ⟨fun j = …
      -/
      congr
      /-
        case h.refine_1.h.e_6.h.e_val
        J : Type v
        inst✝¹ : CategoryTheory.Category.{w, v} J
        F : CategoryTheory.Functor J MonCat
        inst✝ : Small.{u, max u v} ↑(F.comp (CategoryTheory.forget MonCat)).sections
        c : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.forget MonCat))
        t : CategoryTheory.Limits.IsLimit c
        this : Small.{u, max u v} ↑(F.comp (CategoryTheory.forget MonCat)).sections
        s✝ : CategoryTheory.Limits.Cone F
        x y : ↑s✝.pt
        ⊢ Eq (fun j => HMul.hMul ((s✝.π.app j) x) ((s✝.π.app j) y)) (HMul.hMul ↑((equi …
      -/
      simp only [Functor.comp_obj, Equiv.symm_apply_apply]
      /-
        case h.refine_1.h.e_6.h.e_val
        J : Type v
        inst✝¹ : CategoryTheory.Category.{w, v} J
        F : CategoryTheory.Functor J MonCat
        inst✝ : Small.{u, max u v} ↑(F.comp (CategoryTheory.forget MonCat)).sections
        c : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.forget MonCat))
        t : CategoryTheory.Limits.IsLimit c
        this : Small.{u, max u v} ↑(F.comp (CategoryTheory.forget MonCat)).sections
        s✝ : CategoryTheory.Limits.Cone F
        x y : ↑s✝.pt
        ⊢ Eq (fun j => HMul.hMul ((s✝.π.app j) x) ((s✝.π.app j) y)) (HMul.hMul (fun j  …
      -/
      rfl
      /-
        🎉 no goals
      -/
    /-
      case h.refine_2
      J : Type v
      inst✝¹ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J MonCat
      inst✝ : Small.{u, max u v} ↑(F.comp (CategoryTheory.forget MonCat)).sections
      c : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.forget MonCat))
      t : CategoryTheory.Limits.IsLimit c
      this : Small.{u, max u v} ↑(F.comp (CategoryTheory.forget MonCat)).sections
      ⊢ ∀ (s : CategoryTheory.Limits.Cone F), Eq ((CategoryTheory.forget MonCat).map …
    -/
  · exact fun _ ↦ rfl
    /-
      🎉 no goals
    -/


@[to_additive]
noncomputable instance forget_createsLimitsOfShape :
    CreatesLimitsOfShape J (forget MonCat.{u}) where
      CreatesLimit := inferInstance


/-- The forgetful functor from monoids to types preserves all limits.
-/
@[to_additive
"The forgetful functor from additive monoids to types preserves all limits."
]
noncomputable instance forget_createsLimitsOfSize :
    CreatesLimitsOfSize.{w,v} (forget MonCat.{u}) where
      CreatesLimitsOfShape := inferInstance


@[to_additive]
noncomputable instance forget_createsLimits :
    CreatesLimits (forget MonCat.{u}) := MonCat.forget_createsLimitsOfSize.{u,u}


/-- An alias for `CommMonCat.{max u v}`, to deal around unification issues. -/
@[to_additive (attr := nolint checkUnivs) AddCommMonCatMax
  "An alias for `AddCommMonCat.{max u v}`, to deal around unification issues."]
abbrev CommMonCatMax.{u1, u2} := CommMonCat.{max u1 u2}


@[to_additive]
instance commMonoidObj (j) : CommMonoid ((F ⋙ forget CommMonCat.{u}).obj j) :=
  inferInstanceAs <| CommMonoid (F.obj j)


@[to_additive]
noncomputable instance limitCommMonoid :
    CommMonoid (Types.Small.limitCone (F ⋙ forget CommMonCat.{u})).pt :=
  letI : CommMonoid (F ⋙ forget CommMonCat.{u}).sections :=
    @Submonoid.toCommMonoid (∀ j, F.obj j) _
      (MonCat.sectionsSubmonoid (F ⋙ forget₂ CommMonCat.{u} MonCat.{u}))
  inferInstanceAs <| CommMonoid (Shrink (F ⋙ forget CommMonCat.{u}).sections)


@[to_additive]
instance : Small.{u} (Functor.sections ((F ⋙ forget₂ CommMonCat MonCat) ⋙ forget MonCat)) :=
  inferInstanceAs <| Small.{u} (Functor.sections (F ⋙ forget CommMonCat))


/-- We show that the forgetful functor `CommMonCat ⥤ MonCat` creates limits.

All we need to do is notice that the limit point has a `CommMonoid` instance available,
and then reuse the existing limit. -/
@[to_additive "We show that the forgetful functor `AddCommMonCat ⥤ AddMonCat` creates limits.\n\n
All we need to do is notice that the limit point has an `AddCommMonoid` instance available,\n
and then reuse the existing limit."]
noncomputable instance forget₂CreatesLimit : CreatesLimit F (forget₂ CommMonCat MonCat.{u}) :=
  createsLimitOfReflectsIso fun c' t =>
    { liftedCone :=
        { pt := CommMonCat.of (Types.Small.limitCone (F ⋙ forget CommMonCat)).pt
          π :=
            { app := MonCat.limitπMonoidHom (F ⋙ forget₂ CommMonCat.{u} MonCat.{u})
              naturality :=
                (MonCat.HasLimits.limitCone
                      (F ⋙ forget₂ CommMonCat MonCat.{u})).π.naturality } }
                      /-
                        J : Type v
                        inst✝¹ : CategoryTheory.Category.{w, v} J
                        F : CategoryTheory.Functor J CommMonCat
                        inst✝ : Small.{u, max u v} ↑(F.comp (CategoryTheory.forget CommMonCat)).sections
                        c' : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.forget₂ CommMonCat Mon …
                        t : CategoryTheory.Limits.IsLimit c'
                        ⊢ CategoryTheory.Iso ((CategoryTheory.forget₂ CommMonCat MonCat).mapCone { pt  …
                      -/
      validLift := by apply IsLimit.uniqueUpToIso (MonCat.HasLimits.limitConeIsLimit _) t
                      /-
                        🎉 no goals
                      -/
      makesLimit :=
        IsLimit.ofFaithful (forget₂ CommMonCat MonCat.{u})
          (MonCat.HasLimits.limitConeIsLimit _) (fun _ => _) fun _ => rfl }


/-- A choice of limit cone for a functor into `CommMonCat`.
(Generally, you'll just want to use `limit F`.)
-/
@[to_additive "A choice of limit cone for a functor into `AddCommMonCat`.
(Generally, you'll just want to use `limit F`.)"]
noncomputable def limitCone : Cone F :=
  liftLimit (limit.isLimit (F ⋙ forget₂ CommMonCat.{u} MonCat.{u}))


/-- The chosen cone is a limit cone.
(Generally, you'll just want to use `limit.cone F`.)
-/
@[to_additive
      "The chosen cone is a limit cone. (Generally, you'll just want to use\n`limit.cone F`.)"]
noncomputable def limitConeIsLimit : IsLimit (limitCone F) :=
  liftedLimitIsLimit _


/-- If `(F ⋙ forget CommMonCat).sections` is `u`-small, `F` has a limit. -/
@[to_additive "If `(F ⋙ forget AddCommMonCat).sections` is `u`-small, `F` has a limit."]
instance hasLimit : HasLimit F :=
  HasLimit.mk {
    cone := limitCone F
    isLimit := limitConeIsLimit F
  }


/-- If `J` is `u`-small, `CommMonCat.{u}` has limits of shape `J`. -/
@[to_additive "If `J` is `u`-small, `AddCommMonCat.{u}` has limits of shape `J`."]
instance hasLimitsOfShape [Small.{u} J] : HasLimitsOfShape J CommMonCat.{u} where
  has_limit _ := inferInstance


/-- The category of commutative monoids has all limits. -/
@[to_additive "The category of additive commutative monoids has all limits.",
  to_additive_relevant_arg 2]
instance hasLimitsOfSize [UnivLE.{v, u}] : HasLimitsOfSize.{w, v} CommMonCat.{u} where
  has_limits_of_shape _ _ := { }


@[to_additive]
instance hasLimits : HasLimits CommMonCat.{u} :=
  CommMonCat.hasLimitsOfSize.{u, u}


/-- The forgetful functor from commutative monoids to monoids preserves all limits.

This means the underlying type of a limit can be computed as a limit in the category of monoids. -/
@[to_additive AddCommMonCat.forget₂AddMonPreservesLimitsOfSize "The forgetful functor from
  additive commutative monoids to additive monoids preserves all limits.\n\n
  This means the underlying type of a limit can be computed as a limit in the category of additive\n
  monoids.",
  to_additive_relevant_arg 2]
instance forget₂Mon_preservesLimitsOfSize [UnivLE.{v, u}] :
    PreservesLimitsOfSize.{w, v} (forget₂ CommMonCat.{u} MonCat.{u}) where
  preservesLimitsOfShape {J} 𝒥 := { }


@[to_additive]
instance forget₂Mon_preservesLimits :
    PreservesLimits (forget₂ CommMonCat.{u} MonCat.{u}) :=
  CommMonCat.forget₂Mon_preservesLimitsOfSize.{u, u}


/-- If `J` is `u`-small, the forgetful functor from `CommMonCat.{u}` preserves limits of
shape `J`. -/
@[to_additive "If `J` is `u`-small, the forgetful functor from `AddCommMonCat.{u}`\n
preserves limits of shape `J`."]
instance forget_preservesLimitsOfShape [Small.{u} J] :
    PreservesLimitsOfShape J (forget CommMonCat.{u}) where
  preservesLimit {F} := preservesLimit_of_preserves_limit_cone (limitConeIsLimit F)
    (Types.Small.limitConeIsLimit (F ⋙ forget _))


/-- The forgetful functor from commutative monoids to types preserves all limits.

This means the underlying type of a limit can be computed as a limit in the category of types. -/
@[to_additive "The forgetful functor from additive commutative monoids to types preserves all\n
limits.\n\n
This means the underlying type of a limit can be computed as a limit in the category of types."]
instance forget_preservesLimitsOfSize [UnivLE.{v, u}] :
    PreservesLimitsOfSize.{v, v} (forget CommMonCat.{u}) where
  preservesLimitsOfShape {_} _ := { }


instance _root_.AddCommMonCat.forget_preservesLimits :
    PreservesLimits (forget AddCommMonCat.{u}) :=
  AddCommMonCat.forget_preservesLimitsOfSize.{u, u}


@[to_additive existing]
instance forget_preservesLimits : PreservesLimits (forget CommMonCat.{u}) :=
  CommMonCat.forget_preservesLimitsOfSize.{u, u}


@[to_additive]
noncomputable instance forget_createsLimit :
    CreatesLimit F (forget CommMonCat.{u}) := by
  set e : forget CommMonCat.{u} ≅ forget₂ CommMonCat.{u} MonCat.{u} ⋙ forget MonCat.{u} :=
    NatIso.ofComponents (fun _ ↦ Iso.refl _) (fun _ ↦ rfl)
  /-
    J : Type v
    inst✝¹ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J CommMonCat
    inst✝ : Small.{u, max u v} ↑(F.comp (CategoryTheory.forget CommMonCat)).sections
    e : CategoryTheory.Iso (CategoryTheory.forget CommMonCat) ((CategoryTheory.for …
    ⊢ CategoryTheory.CreatesLimit F (CategoryTheory.forget CommMonCat)
  -/
  exact createsLimitOfNatIso e.symm
  /-
    🎉 no goals
  -/


/-- The forgetful functor from commutative monoids to types preserves all limits.
-/
@[to_additive
"The forgetful functor from commutative additive monoids to types preserves all limits."
]
noncomputable instance forget_createsLimitsOfSize :
    CreatesLimitsOfSize.{w,v} (forget MonCat.{u}) where
      CreatesLimitsOfShape := inferInstance


@[to_additive]
noncomputable instance forget_createsLimits :
    CreatesLimits (forget MonCat.{u}) := CommMonCat.forget_createsLimitsOfSize.{u,u}


