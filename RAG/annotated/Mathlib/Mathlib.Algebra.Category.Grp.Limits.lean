@[to_additive]
instance groupObj (j) : Group ((F ⋙ forget Grp).obj j) :=
  inferInstanceAs <| Group (F.obj j)


/-- The flat sections of a functor into `Grp` form a subgroup of all sections.
-/
@[to_additive
  "The flat sections of a functor into `AddGrp` form an additive subgroup of all sections."]
def sectionsSubgroup : Subgroup (∀ j, F.obj j) :=
  { MonCat.sectionsSubmonoid (F ⋙ forget₂ Grp MonCat) with
    carrier := (F ⋙ forget Grp).sections
    inv_mem' := fun {a} ah j j' f => by
      /-
        J : Type v
        inst✝ : CategoryTheory.Category.{w, v} J
        F : CategoryTheory.Functor J Grp
        a : (j : J) → ↑(F.obj j)
        ah : Membership.mem { carrier := (F.comp (CategoryTheory.forget Grp)).sections …
        j j' : J
        f : Quiver.Hom j j'
        ⊢ Eq ((F.comp (CategoryTheory.forget Grp)).map f (Inv.inv a j)) (Inv.inv a j')
      -/
      simp only [Functor.comp_map, Pi.inv_apply, MonoidHom.map_inv, inv_inj]
      /-
        J : Type v
        inst✝ : CategoryTheory.Category.{w, v} J
        F : CategoryTheory.Functor J Grp
        a : (j : J) → ↑(F.obj j)
        ah : Membership.mem { carrier := (F.comp (CategoryTheory.forget Grp)).sections …
        j j' : J
        f : Quiver.Hom j j'
        ⊢ Eq ((CategoryTheory.forget Grp).map (F.map f) (Inv.inv (a j))) (Inv.inv (a j …
      -/
      dsimp [Functor.sections] at ah ⊢
      /-
        J : Type v
        inst✝ : CategoryTheory.Category.{w, v} J
        F : CategoryTheory.Functor J Grp
        a : (j : J) → ↑(F.obj j)
        ah : ∀ {j j' : J} (f : Quiver.Hom j j'), Eq ((F.map f) (a j)) (a j')
        j j' : J
        f : Quiver.Hom j j'
        ⊢ Eq ((F.map f) (Inv.inv (a j))) (Inv.inv (a j'))
      -/
      rw [(F.map f).map_inv (a j), ah f] }
      /-
        🎉 no goals
      -/


@[to_additive]
instance sectionsGroup : Group (F ⋙ forget Grp.{u}).sections :=
  (sectionsSubgroup F).toGroup


/-- The projection from `Functor.sections` to a factor as a `MonoidHom`. -/
@[to_additive "The projection from `Functor.sections` to a factor as an `AddMonoidHom`."]
def sectionsπMonoidHom (j : J) : (F ⋙ forget Grp.{u}).sections →* F.obj j where
  toFun x := x.val j
  map_one' := rfl
  map_mul' _ _ := rfl


@[to_additive]
noncomputable instance limitGroup :
    Group (Types.Small.limitCone.{v, u} (F ⋙ forget Grp.{u})).pt :=
  inferInstanceAs <| Group (Shrink (F ⋙ forget Grp.{u}).sections)


@[to_additive]
instance : Small.{u} (Functor.sections ((F ⋙ forget₂ Grp MonCat) ⋙ forget MonCat)) :=
  inferInstanceAs <| Small.{u} (Functor.sections (F ⋙ forget Grp))


/-- We show that the forgetful functor `Grp ⥤ MonCat` creates limits.

All we need to do is notice that the limit point has a `Group` instance available, and then reuse
the existing limit. -/
@[to_additive "We show that the forgetful functor `AddGrp ⥤ AddMonCat` creates limits.

All we need to do is notice that the limit point has an `AddGroup` instance available, and then
reuse the existing limit."]
noncomputable instance Forget₂.createsLimit :
    CreatesLimit F (forget₂ Grp.{u} MonCat.{u}) :=
  -- Porting note: need to add `forget₂ GrpCat MonCat` reflects isomorphism
  letI : (forget₂ Grp.{u} MonCat.{u}).ReflectsIsomorphisms :=
    CategoryTheory.reflectsIsomorphisms_forget₂ _ _
  createsLimitOfReflectsIso (K := F) (F := (forget₂ Grp.{u} MonCat.{u}))
    fun c' t =>
      have : Small.{u} (Functor.sections ((F ⋙ forget₂ Grp MonCat) ⋙ forget MonCat)) := by
        /-
          J : Type v
          inst✝¹ : CategoryTheory.Category.{w, v} J
          F : CategoryTheory.Functor J Grp
          inst✝ : Small.{u, max u v} ↑(F.comp (CategoryTheory.forget Grp)).sections
          this : (CategoryTheory.forget₂ Grp MonCat).ReflectsIsomorphisms := CategoryThe …
          c' : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.forget₂ Grp MonCat))
          t : CategoryTheory.Limits.IsLimit c'
          ⊢ Small.{u, max u v} ↑((F.comp (CategoryTheory.forget₂ Grp MonCat)).comp (Cate …
        -/
        have : HasLimit (F ⋙ forget₂ Grp MonCat) := ⟨_, t⟩
        /-
          J : Type v
          inst✝¹ : CategoryTheory.Category.{w, v} J
          F : CategoryTheory.Functor J Grp
          inst✝ : Small.{u, max u v} ↑(F.comp (CategoryTheory.forget Grp)).sections
          this✝ : (CategoryTheory.forget₂ Grp MonCat).ReflectsIsomorphisms := CategoryTh …
          c' : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.forget₂ Grp MonCat))
          t : CategoryTheory.Limits.IsLimit c'
          this : CategoryTheory.Limits.HasLimit (F.comp (CategoryTheory.forget₂ Grp MonC …
          ⊢ Small.{u, max u v} ↑((F.comp (CategoryTheory.forget₂ Grp MonCat)).comp (Cate …
        -/
        apply Concrete.small_sections_of_hasLimit (F ⋙ forget₂ Grp MonCat)
        /-
          🎉 no goals
        -/
      have : Small.{u} (Functor.sections (F ⋙ forget Grp)) := inferInstanceAs <| Small.{u}
        (Functor.sections ((F ⋙ forget₂ Grp MonCat) ⋙ forget MonCat))
      { liftedCone :=
          { pt := Grp.of (Types.Small.limitCone (F ⋙ forget Grp)).pt
            π :=
              { app := MonCat.limitπMonoidHom (F ⋙ forget₂ Grp MonCat)
                naturality :=
                  (MonCat.HasLimits.limitCone
                        (F ⋙ forget₂ Grp MonCat.{u})).π.naturality } }
                        /-
                          J : Type v
                          inst✝¹ : CategoryTheory.Category.{w, v} J
                          F : CategoryTheory.Functor J Grp
                          inst✝ : Small.{u, max u v} ↑(F.comp (CategoryTheory.forget Grp)).sections
                          this✝¹ : (CategoryTheory.forget₂ Grp MonCat).ReflectsIsomorphisms := CategoryT …
                          c' : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.forget₂ Grp MonCat))
                          t : CategoryTheory.Limits.IsLimit c'
                          this✝ : Small.{u, max u v} ↑((F.comp (CategoryTheory.forget₂ Grp MonCat)).comp …
                          this : Small.{u, max u v} ↑(F.comp (CategoryTheory.forget Grp)).sections
                          ⊢ CategoryTheory.Iso ((CategoryTheory.forget₂ Grp MonCat).mapCone { pt := Grp. …
                        -/
        validLift := by apply IsLimit.uniqueUpToIso (MonCat.HasLimits.limitConeIsLimit.{v, u} _) t
                        /-
                          🎉 no goals
                        -/
        makesLimit :=
         IsLimit.ofFaithful (forget₂ Grp MonCat.{u}) (MonCat.HasLimits.limitConeIsLimit _)
          (fun _ => _) fun _ => rfl }


/-- A choice of limit cone for a functor into `Grp`.
(Generally, you'll just want to use `limit F`.)
-/
@[to_additive "A choice of limit cone for a functor into `Grp`.
  (Generally, you'll just want to use `limit F`.)"]
noncomputable def limitCone : Cone F :=
  liftLimit (limit.isLimit (F ⋙ forget₂ Grp.{u} MonCat.{u}))


/-- The chosen cone is a limit cone.
(Generally, you'll just want to use `limit.cone F`.)
-/
@[to_additive "The chosen cone is a limit cone.
  (Generally, you'll just want to use `limit.cone F`.)"]
noncomputable def limitConeIsLimit : IsLimit (limitCone F) :=
  liftedLimitIsLimit _


/-- If `(F ⋙ forget Grp).sections` is `u`-small, `F` has a limit. -/
@[to_additive "If `(F ⋙ forget AddGrp).sections` is `u`-small, `F` has a limit."]
instance hasLimit : HasLimit F :=
  HasLimit.mk {
    cone := limitCone F
    isLimit := limitConeIsLimit F
  }


/-- A functor `F : J ⥤ Grp.{u}` has a limit iff `(F ⋙ forget Grp).sections` is
`u`-small. -/
@[to_additive "A functor `F : J ⥤ AddGrp.{u}` has a limit iff
`(F ⋙ forget AddGrp).sections` is `u`-small."]
lemma hasLimit_iff_small_sections :
    HasLimit F ↔ Small.{u} (F ⋙ forget Grp).sections := by
  /-
    J : Type v
    inst✝ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J Grp
    ⊢ Iff (CategoryTheory.Limits.HasLimit F) (Small.{u, max u v} ↑(F.comp (Categor …
  -/
  constructor
    /-
      case mp
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J Grp
      ⊢ CategoryTheory.Limits.HasLimit F → Small.{u, max u v} ↑(F.comp (CategoryTheo …
    -/
  · apply Concrete.small_sections_of_hasLimit
    /-
      🎉 no goals
    -/
    /-
      case mpr
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J Grp
      ⊢ Small.{u, max u v} ↑(F.comp (CategoryTheory.forget Grp)).sections → Category …
    -/
  · intro
    /-
      case mpr
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J Grp
      a✝ : Small.{u, max u v} ↑(F.comp (CategoryTheory.forget Grp)).sections
      ⊢ CategoryTheory.Limits.HasLimit F
    -/
    infer_instance
    /-
      🎉 no goals
    -/


/-- If `J` is `u`-small, `Grp.{u}` has limits of shape `J`. -/
@[to_additive "If `J` is `u`-small, `AddGrp.{u}` has limits of shape `J`."]
instance hasLimitsOfShape [Small.{u} J] : HasLimitsOfShape J Grp.{u} where
  has_limit _ := inferInstance


/-- The category of groups has all limits. -/
@[to_additive "The category of additive groups has all limits.",
  to_additive_relevant_arg 2]
instance hasLimitsOfSize [UnivLE.{v, u}] : HasLimitsOfSize.{w, v} Grp.{u} where
  has_limits_of_shape J _ := { }


@[to_additive]
instance hasLimits : HasLimits Grp.{u} :=
  Grp.hasLimitsOfSize.{u, u}


/-- The forgetful functor from groups to monoids preserves all limits.

This means the underlying monoid of a limit can be computed as a limit in the category of monoids.
-/
@[to_additive AddGrp.forget₂AddMonPreservesLimitsOfSize
  "The forgetful functor from additive groups to additive monoids preserves all limits.

  This means the underlying additive monoid of a limit can be computed as a limit in the category of
  additive monoids.",
  to_additive_relevant_arg 2]
instance forget₂Mon_preservesLimitsOfSize [UnivLE.{v, u}] :
    PreservesLimitsOfSize.{w, v} (forget₂ Grp.{u} MonCat.{u}) where
  preservesLimitsOfShape {J _} := { }


@[to_additive]
instance forget₂Mon_preservesLimits :
  PreservesLimits (forget₂ Grp.{u} MonCat.{u}) :=
  Grp.forget₂Mon_preservesLimitsOfSize.{u, u}


/-- If `J` is `u`-small, the forgetful functor from `Grp.{u}` preserves limits of shape `J`. -/
@[to_additive "If `J` is `u`-small, the forgetful functor from `AddGrp.{u}`\n
preserves limits of shape `J`."]
instance forget_preservesLimitsOfShape [Small.{u} J] :
    PreservesLimitsOfShape J (forget Grp.{u}) where
  preservesLimit {F} := preservesLimit_of_preserves_limit_cone (limitConeIsLimit F)
    (Types.Small.limitConeIsLimit (F ⋙ forget _))


/-- The forgetful functor from groups to types preserves all limits.

This means the underlying type of a limit can be computed as a limit in the category of types. -/
@[to_additive
  "The forgetful functor from additive groups to types preserves all limits.

  This means the underlying type of a limit can be computed as a limit in the category of types.",
  to_additive_relevant_arg 2]
instance forget_preservesLimitsOfSize :
    PreservesLimitsOfSize.{w, v} (forget Grp.{u}) := inferInstance


@[to_additive]
instance forget_preservesLimits : PreservesLimits (forget Grp.{u}) :=
  Grp.forget_preservesLimitsOfSize.{u, u}


@[to_additive]
noncomputable instance forget_createsLimit :
    CreatesLimit F (forget Grp.{u}) := by
  /-
    J : Type v
    inst✝ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J Grp
    ⊢ CategoryTheory.CreatesLimit F (CategoryTheory.forget Grp)
  -/
  set e : forget₂ Grp.{u} MonCat.{u} ⋙ forget MonCat.{u} ≅ forget Grp.{u} := Iso.refl _
  /-
    J : Type v
    inst✝ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J Grp
    e : CategoryTheory.Iso ((CategoryTheory.forget₂ Grp MonCat).comp (CategoryTheo …
    ⊢ CategoryTheory.CreatesLimit F (CategoryTheory.forget Grp)
  -/
  exact createsLimitOfNatIso e
  /-
    🎉 no goals
  -/


@[to_additive]
noncomputable instance forget_createsLimitsOfShape :
    CreatesLimitsOfShape J (forget Grp.{u}) where
  CreatesLimit := inferInstance


/-- The forgetful functor from groups to types creates all limits.
-/
@[to_additive
  "The forgetful functor from additive groups to types creates all limits.",
  to_additive_relevant_arg 2]
noncomputable instance forget_createsLimitsOfSize :
    CreatesLimitsOfSize.{w, v} (forget Grp.{u}) where
  CreatesLimitsOfShape := inferInstance

@[to_additive]
instance commGroupObj (j) : CommGroup ((F ⋙ forget CommGrp).obj j) :=
  inferInstanceAs <| CommGroup (F.obj j)


@[to_additive]
noncomputable instance limitCommGroup
    [Small.{u} (Functor.sections (F ⋙ forget CommGrp))] :
    CommGroup (Types.Small.limitCone.{v, u} (F ⋙ forget CommGrp.{u})).pt :=
  letI : CommGroup (F ⋙ forget CommGrp.{u}).sections :=
    @Subgroup.toCommGroup (∀ j, F.obj j) _
      (Grp.sectionsSubgroup (F ⋙ forget₂ CommGrp.{u} Grp.{u}))
  inferInstanceAs <| CommGroup (Shrink (F ⋙ forget CommGrp.{u}).sections)


@[to_additive]
instance : (forget₂ CommGrp.{u} Grp.{u}).ReflectsIsomorphisms :=
    reflectsIsomorphisms_forget₂ _ _


/-- We show that the forgetful functor `CommGrp ⥤ Grp` creates limits.

All we need to do is notice that the limit point has a `CommGroup` instance available,
and then reuse the existing limit.
-/
@[to_additive "We show that the forgetful functor `AddCommGrp ⥤ AddGrp` creates limits.

  All we need to do is notice that the limit point has an `AddCommGroup` instance available,
  and then reuse the existing limit."]
noncomputable instance Forget₂.createsLimit :
    CreatesLimit F (forget₂ CommGrp Grp.{u}) :=
  createsLimitOfReflectsIso (fun c hc => by
    /-
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J CommGrp
      c : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.forget₂ CommGrp Grp))
      hc : CategoryTheory.Limits.IsLimit c
      ⊢ CategoryTheory.LiftsToLimit F (CategoryTheory.forget₂ CommGrp Grp) c hc
    -/
    have : HasLimit _ := ⟨_, hc⟩
    have : Small.{u} (F ⋙ forget CommGrp).sections :=
      Concrete.small_sections_of_hasLimit (F ⋙ forget₂ CommGrp Grp)
    have : Small.{u} ((F ⋙ forget₂ CommGrp Grp ⋙ forget₂ Grp MonCat) ⋙
      forget MonCat).sections := this
    /-
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J CommGrp
      c : CategoryTheory.Limits.Cone (F.comp (CategoryTheory.forget₂ CommGrp Grp))
      hc : CategoryTheory.Limits.IsLimit c
      this✝¹ : CategoryTheory.Limits.HasLimit (F.comp (CategoryTheory.forget₂ CommGr …
      this✝ : Small.{u, max u v} ↑(F.comp (CategoryTheory.forget CommGrp)).sections
      this : Small.{u, max u v} ↑((F.comp ((CategoryTheory.forget₂ CommGrp Grp).comp …
      ⊢ CategoryTheory.LiftsToLimit F (CategoryTheory.forget₂ CommGrp Grp) c hc
    -/
    have : Small.{u} ((F ⋙ forget₂ CommGrp Grp) ⋙ forget Grp).sections := this
    exact
      { liftedCone :=
          { pt := CommGrp.of (Types.Small.limitCone.{v, u} (F ⋙ forget CommGrp)).pt
            π :=
              { app := MonCat.limitπMonoidHom
                  (F ⋙ forget₂ CommGrp Grp.{u} ⋙ forget₂ Grp MonCat.{u})
                naturality := (MonCat.HasLimits.limitCone _).π.naturality } }
        validLift := by apply IsLimit.uniqueUpToIso (Grp.limitConeIsLimit _) hc
        makesLimit :=
          IsLimit.ofFaithful (forget₂ _ Grp.{u} ⋙ forget₂ _ MonCat.{u})
            (by apply MonCat.HasLimits.limitConeIsLimit _) (fun s => _) fun s => rfl })


/-- A choice of limit cone for a functor into `CommGrp`.
(Generally, you'll just want to use `limit F`.)
-/
@[to_additive
  "A choice of limit cone for a functor into `AddCommGrp`.
  (Generally, you'll just want to use `limit F`.)"]
noncomputable def limitCone : Cone F :=
  letI : Small.{u} (Functor.sections ((F ⋙ forget₂ CommGrp Grp) ⋙ forget Grp)) :=
    inferInstanceAs <| Small (Functor.sections (F ⋙ forget CommGrp))
  liftLimit (limit.isLimit (F ⋙ forget₂ CommGrp.{u} Grp.{u}))


/-- The chosen cone is a limit cone.
(Generally, you'll just want to use `limit.cone F`.)
-/
@[to_additive
  "The chosen cone is a limit cone.
  (Generally, you'll just want to use `limit.cone F`.)"]
noncomputable def limitConeIsLimit : IsLimit (limitCone.{v, u} F) :=
  liftedLimitIsLimit _


/-- If `(F ⋙ forget CommGrp).sections` is `u`-small, `F` has a limit. -/
@[to_additive "If `(F ⋙ forget AddCommGrp).sections` is `u`-small, `F` has a limit."]
instance hasLimit : HasLimit F :=
  HasLimit.mk {
    cone := limitCone F
    isLimit := limitConeIsLimit F
  }


/-- A functor `F : J ⥤ CommGrp.{u}` has a limit iff `(F ⋙ forget CommGrp).sections` is
`u`-small. -/
@[to_additive "A functor `F : J ⥤ AddCommGrp.{u}` has a limit iff
`(F ⋙ forget AddCommGrp).sections` is `u`-small."]
lemma hasLimit_iff_small_sections :
    HasLimit F ↔ Small.{u} (F ⋙ forget CommGrp).sections := by
  /-
    J : Type v
    inst✝ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J CommGrp
    ⊢ Iff (CategoryTheory.Limits.HasLimit F) (Small.{u, max u v} ↑(F.comp (Categor …
  -/
  constructor
    /-
      case mp
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J CommGrp
      ⊢ CategoryTheory.Limits.HasLimit F → Small.{u, max u v} ↑(F.comp (CategoryTheo …
    -/
  · apply Concrete.small_sections_of_hasLimit
    /-
      🎉 no goals
    -/
    /-
      case mpr
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J CommGrp
      ⊢ Small.{u, max u v} ↑(F.comp (CategoryTheory.forget CommGrp)).sections → Cate …
    -/
  · intro
    /-
      case mpr
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J CommGrp
      a✝ : Small.{u, max u v} ↑(F.comp (CategoryTheory.forget CommGrp)).sections
      ⊢ CategoryTheory.Limits.HasLimit F
    -/
    infer_instance
    /-
      🎉 no goals
    -/


/-- If `J` is `u`-small, `CommGrp.{u}` has limits of shape `J`. -/
@[to_additive "If `J` is `u`-small, `AddCommGrp.{u}` has limits of shape `J`."]
instance hasLimitsOfShape [Small.{u} J] : HasLimitsOfShape J CommGrp.{u} where
  has_limit _ := inferInstance


/-- The category of commutative groups has all limits. -/
@[to_additive "The category of additive commutative groups has all limits.",
  to_additive_relevant_arg 2]
instance hasLimitsOfSize [UnivLE.{v, u}] : HasLimitsOfSize.{w, v} CommGrp.{u}
  where has_limits_of_shape _ _ := { }


@[to_additive]
instance hasLimits : HasLimits CommGrp.{u} :=
  CommGrp.hasLimitsOfSize.{u, u}


@[to_additive]
instance forget₂Group_preservesLimit :
    PreservesLimit F (forget₂ CommGrp.{u} Grp.{u}) where
  preserves {c} hc := ⟨by
    have : HasLimit (F ⋙ forget₂ CommGrp Grp) := by
      rw [Grp.hasLimit_iff_small_sections]
      change Small.{u} (F ⋙ forget CommGrp).sections
      rw [← CommGrp.hasLimit_iff_small_sections]
      exact ⟨_, hc⟩
    /-
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      F : CategoryTheory.Functor J CommGrp
      c : CategoryTheory.Limits.Cone F
      hc : CategoryTheory.Limits.IsLimit c
      this : CategoryTheory.Limits.HasLimit (F.comp (CategoryTheory.forget₂ CommGrp  …
      ⊢ CategoryTheory.Limits.IsLimit ((CategoryTheory.forget₂ CommGrp Grp).mapCone c)
    -/
    exact isLimitOfPreserves _ hc⟩
    /-
      🎉 no goals
    -/


@[to_additive]
instance forget₂Group_preservesLimitsOfShape :
    PreservesLimitsOfShape J (forget₂ CommGrp.{u} Grp.{u}) where


/-- The forgetful functor from commutative groups to groups preserves all limits.
(That is, the underlying group could have been computed instead as limits in the category
of groups.)
-/
@[to_additive
  "The forgetful functor from additive commutative groups to additive groups preserves all limits.
  (That is, the underlying group could have been computed instead as limits in the category
    of additive groups.)",
  to_additive_relevant_arg 2]
instance forget₂Group_preservesLimitsOfSize :
    PreservesLimitsOfSize.{w, v} (forget₂ CommGrp.{u} Grp.{u}) where


@[to_additive]
instance forget₂Group_preservesLimits :
    PreservesLimits (forget₂ CommGrp Grp.{u}) :=
  CommGrp.forget₂Group_preservesLimitsOfSize.{u, u}


/-- An auxiliary declaration to speed up typechecking.
-/
@[to_additive AddCommGrp.forget₂AddCommMon_preservesLimitsAux
  "An auxiliary declaration to speed up typechecking."]
noncomputable def forget₂CommMon_preservesLimitsAux
    [Small.{u} (F ⋙ forget CommGrp).sections] :
    IsLimit ((forget₂ CommGrp.{u} CommMonCat.{u}).mapCone (limitCone.{v, u} F)) :=
  letI : Small.{u} (Functor.sections ((F ⋙ forget₂ _ CommMonCat) ⋙ forget CommMonCat)) :=
    inferInstanceAs <| Small (Functor.sections (F ⋙ forget CommGrp))
  CommMonCat.limitConeIsLimit.{v, u} (F ⋙ forget₂ CommGrp.{u} CommMonCat.{u})


/-- If `J` is `u`-small, the forgetful functor from `CommGrp.{u}` to `CommMonCat.{u}`
preserves limits of shape `J`. -/
@[to_additive AddCommGrp.forget₂AddCommMon_preservesLimitsOfShape
  "If `J` is `u`-small, the forgetful functor from `AddCommGrp.{u}`
  to `AddCommMonCat.{u}` preserves limits of shape `J`."]
instance forget₂CommMon_preservesLimitsOfShape [Small.{u} J] :
    PreservesLimitsOfShape J (forget₂ CommGrp.{u} CommMonCat.{u}) where
  preservesLimit {F} := preservesLimit_of_preserves_limit_cone (limitConeIsLimit.{v, u} F)
      (forget₂CommMon_preservesLimitsAux.{v, u} F)


/-- The forgetful functor from commutative groups to commutative monoids preserves all limits.
(That is, the underlying commutative monoids could have been computed instead as limits
in the category of commutative monoids.)
-/
@[to_additive AddCommGrp.forget₂AddCommMon_preservesLimitsOfSize
  "The forgetful functor from additive commutative groups to additive commutative monoids
  preserves all limits. (That is, the underlying additive commutative monoids could have been
  computed instead as limits in the category of additive commutative monoids.)"]
instance forget₂CommMon_preservesLimitsOfSize [UnivLE.{v, u}] :
    PreservesLimitsOfSize.{w, v} (forget₂ CommGrp CommMonCat.{u}) where
  preservesLimitsOfShape := { }


/-- If `J` is `u`-small, the forgetful functor from `CommGrp.{u}` preserves limits of
shape `J`. -/
@[to_additive "If `J` is `u`-small, the forgetful functor from `AddCommGrp.{u}`\n
preserves limits of shape `J`."]
instance forget_preservesLimitsOfShape [Small.{u} J] :
    PreservesLimitsOfShape J (forget CommGrp.{u}) where
  preservesLimit {F} := preservesLimit_of_preserves_limit_cone (limitConeIsLimit F)
    (Types.Small.limitConeIsLimit (F ⋙ forget _))


/-- The forgetful functor from commutative groups to types preserves all limits. (That is, the
underlying types could have been computed instead as limits in the category of types.)
-/
@[to_additive
  "The forgetful functor from additive commutative groups to types preserves all limits.
  (That is, the underlying types could have been computed instead as limits in the category of
  types.)"]
instance forget_preservesLimitsOfSize :
    PreservesLimitsOfSize.{w, v} (forget CommGrp.{u}) := inferInstance


noncomputable instance _root_.AddCommGrp.forget_preservesLimits :
    PreservesLimits (forget AddCommGrp.{u}) :=
  AddCommGrp.forget_preservesLimitsOfSize.{u, u}


@[to_additive existing]
noncomputable instance forget_preservesLimits : PreservesLimits (forget CommGrp.{u}) :=
  CommGrp.forget_preservesLimitsOfSize.{u, u}


@[to_additive]
noncomputable instance forget_createsLimit :
    CreatesLimit F (forget CommGrp.{u}) := by
  /-
    J : Type v
    inst✝ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J CommGrp
    ⊢ CategoryTheory.CreatesLimit F (CategoryTheory.forget CommGrp)
  -/
  set e : forget₂ CommGrp.{u} Grp.{u} ⋙ forget Grp.{u} ≅ forget CommGrp.{u} := Iso.refl _
  /-
    J : Type v
    inst✝ : CategoryTheory.Category.{w, v} J
    F : CategoryTheory.Functor J CommGrp
    e : CategoryTheory.Iso ((CategoryTheory.forget₂ CommGrp Grp).comp (CategoryThe …
    ⊢ CategoryTheory.CreatesLimit F (CategoryTheory.forget CommGrp)
  -/
  exact createsLimitOfNatIso e
  /-
    🎉 no goals
  -/


@[to_additive]
noncomputable instance forget_createsLimitsOfShape (J : Type v) [Category.{w} J] :
    CreatesLimitsOfShape J (forget CommGrp.{u}) where
  CreatesLimit := inferInstance


/-- The forgetful functor from commutative groups to types creates all limits.
-/
@[to_additive
  "The forgetful functor from additive commutative groups to types creates all limits.",
  to_additive_relevant_arg 2]
noncomputable instance forget_createsLimitsOfSize :
    CreatesLimitsOfSize.{w, v} (forget CommGrp.{u}) where
  CreatesLimitsOfShape := inferInstance
-- Verify we can form limits indexed over smaller categories.

/-- The categorical kernel of a morphism in `AddCommGrp`
agrees with the usual group-theoretical kernel.
-/
def kernelIsoKer {G H : AddCommGrp.{u}} (f : G ⟶ H) :
    kernel f ≅ AddCommGrp.of f.ker where
  hom :=
    { toFun := fun g => ⟨kernel.ι f g, DFunLike.congr_fun (kernel.condition f) g⟩
      map_zero' := by
        /-
          J : Type v
          inst✝ : CategoryTheory.Category.{w, v} J
          G H : AddCommGrp
          f : Quiver.Hom G H
          ⊢ Eq ((fun g => ⟨(CategoryTheory.Limits.kernel.ι f) g, ⋯⟩) 0) 0
        -/
        refine Subtype.ext ?_
        /-
          J : Type v
          inst✝ : CategoryTheory.Category.{w, v} J
          G H : AddCommGrp
          f : Quiver.Hom G H
          ⊢ Eq ↑((fun g => ⟨(CategoryTheory.Limits.kernel.ι f) g, ⋯⟩) 0) ↑0
        -/
        simp [(AddSubgroup.coe_zero _).symm]
        /-
          🎉 no goals
        -/
      map_add' := fun g g' => by
        /-
          J : Type v
          inst✝ : CategoryTheory.Category.{w, v} J
          G H : AddCommGrp
          f : Quiver.Hom G H
          g g' : ↑(CategoryTheory.Limits.kernel f)
          ⊢ Eq ({ toFun := fun g => ⟨(CategoryTheory.Limits.kernel.ι f) g, ⋯⟩, map_zero' …
        -/
        refine Subtype.ext ?_
        /-
          J : Type v
          inst✝ : CategoryTheory.Category.{w, v} J
          G H : AddCommGrp
          f : Quiver.Hom G H
          g g' : ↑(CategoryTheory.Limits.kernel f)
          ⊢ Eq ↑({ toFun := fun g => ⟨(CategoryTheory.Limits.kernel.ι f) g, ⋯⟩, map_zero …
        -/
        change _ = _ + _
        /-
          J : Type v
          inst✝ : CategoryTheory.Category.{w, v} J
          G H : AddCommGrp
          f : Quiver.Hom G H
          g g' : ↑(CategoryTheory.Limits.kernel f)
          ⊢ Eq (↑({ toFun := fun g => ⟨(CategoryTheory.Limits.kernel.ι f) g, ⋯⟩, map_zer …
        -/
        dsimp
        /-
          J : Type v
          inst✝ : CategoryTheory.Category.{w, v} J
          G H : AddCommGrp
          f : Quiver.Hom G H
          g g' : ↑(CategoryTheory.Limits.kernel f)
          ⊢ Eq ((CategoryTheory.Limits.kernel.ι f) (HAdd.hAdd g g')) (HAdd.hAdd ((Catego …
        -/
        simp }
        /-
          🎉 no goals
        -/
                                                         /-
                                                           J : Type v
                                                           inst✝ : CategoryTheory.Category.{w, v} J
                                                           G H : AddCommGrp
                                                           f : Quiver.Hom G H
                                                           ⊢ Eq (CategoryTheory.CategoryStruct.comp (AddMonoidHom.ker f).subtype f) 0
                                                         -/
  inv := kernel.lift f (AddSubgroup.subtype f.ker) <| by ext x; exact x.2
                                                                /-
                                                                  🎉 no goals
                                                                -/
  hom_inv_id := by
    -- Porting note (https://github.com/leanprover-community/mathlib4/pull/11041): it would be nice to do the next two steps by a single `ext`,
    -- but this will require thinking carefully about the relative priorities of `@[ext]` lemmas.
    /-
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      G H : AddCommGrp
      f : Quiver.Hom G H
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { toFun := fun g => ⟨(CategoryTheory. …
    -/
    refine equalizer.hom_ext ?_
    /-
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      G H : AddCommGrp
      f : Quiver.Hom G H
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp { …
    -/
    ext x
    /-
      case w
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      G H : AddCommGrp
      f : Quiver.Hom G H
      x : ↑(CategoryTheory.Limits.kernel f)
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp  …
    -/
    dsimp
    /-
      case w
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      G H : AddCommGrp
      f : Quiver.Hom G H
      x : ↑(CategoryTheory.Limits.kernel f)
      ⊢ Eq ((CategoryTheory.Limits.kernel.ι f) ((CategoryTheory.Limits.kernel.lift f …
    -/
    apply DFunLike.congr_fun (kernel.lift_ι f _ _)
    /-
      🎉 no goals
    -/
  inv_hom_id := by
    /-
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      G H : AddCommGrp
      f : Quiver.Hom G H
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernel.lift f  …
    -/
    apply AddCommGrp.ext
    /-
      case w
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      G H : AddCommGrp
      f : Quiver.Hom G H
      ⊢ ∀ (x : ↑(AddCommGrp.of (Subtype fun x => Membership.mem (AddMonoidHom.ker f) …
    -/
    simp only [AddMonoidHom.coe_mk, coe_id, coe_comp]
    /-
      case w
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      G H : AddCommGrp
      f : Quiver.Hom G H
      ⊢ ∀ (x : ↑(AddCommGrp.of (Subtype fun x => Membership.mem (AddMonoidHom.ker f) …
    -/
    rintro ⟨x, mem⟩
    /-
      case w.mk
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      G H : AddCommGrp
      f : Quiver.Hom G H
      x : ↑G
      mem : Membership.mem (AddMonoidHom.ker f) x
      ⊢ Eq (Function.comp ⇑{ toFun := fun g => ⟨(CategoryTheory.Limits.kernel.ι f) g …
    -/
    refine Subtype.ext ?_
    /-
      case w.mk
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      G H : AddCommGrp
      f : Quiver.Hom G H
      x : ↑G
      mem : Membership.mem (AddMonoidHom.ker f) x
      ⊢ Eq ↑(Function.comp ⇑{ toFun := fun g => ⟨(CategoryTheory.Limits.kernel.ι f)  …
    -/
    simp only [ZeroHom.coe_mk, Function.comp_apply, id_eq]
    /-
      case w.mk
      J : Type v
      inst✝ : CategoryTheory.Category.{w, v} J
      G H : AddCommGrp
      f : Quiver.Hom G H
      x : ↑G
      mem : Membership.mem (AddMonoidHom.ker f) x
      ⊢ Eq ((CategoryTheory.Limits.kernel.ι f) ((CategoryTheory.Limits.kernel.lift f …
    -/
    apply DFunLike.congr_fun (kernel.lift_ι f _ _)
    /-
      🎉 no goals
    -/


@[simp]
theorem kernelIsoKer_hom_comp_subtype {G H : AddCommGrp.{u}} (f : G ⟶ H) :
                                                                        /-
                                                                          G H : AddCommGrp
                                                                          f : Quiver.Hom G H
                                                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (AddCommGrp.kernelIsoKer f).hom (AddM …
                                                                        -/
    (kernelIsoKer f).hom ≫ AddSubgroup.subtype f.ker = kernel.ι f := by ext; rfl
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


@[simp]
theorem kernelIsoKer_inv_comp_ι {G H : AddCommGrp.{u}} (f : G ⟶ H) :
    (kernelIsoKer f).inv ≫ kernel.ι f = AddSubgroup.subtype f.ker := by
  /-
    G H : AddCommGrp
    f : Quiver.Hom G H
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AddCommGrp.kernelIsoKer f).inv (Cate …
  -/
  ext
  /-
    case w
    G H : AddCommGrp
    f : Quiver.Hom G H
    x✝ : ↑(AddCommGrp.of (Subtype fun x => Membership.mem (AddMonoidHom.ker f) x))
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (AddCommGrp.kernelIsoKer f).inv (Cat …
  -/
  simp [kernelIsoKer]
  /-
    🎉 no goals
  -/

-- Porting note: explicitly add what to be synthesized under `simps!`, because other lemmas
-- automatically generated is not in normal form

/-- The categorical kernel inclusion for `f : G ⟶ H`, as an object over `G`,
agrees with the `AddSubgroup.subtype` map.
-/
def kernelIsoKerOver {G H : AddCommGrp.{u}} (f : G ⟶ H) :
    Over.mk (kernel.ι f) ≅ @Over.mk _ _ G (AddCommGrp.of f.ker) (AddSubgroup.subtype f.ker) :=
  /-
    J : Type v
    inst✝ : CategoryTheory.Category.{w, v} J
    G H : AddCommGrp
    f : Quiver.Hom G H
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AddCommGrp.kernelIsoKer f).hom (Cate …
  -/
  Over.isoMk (kernelIsoKer f)
  /-
    🎉 no goals
  -/


