/-- The functor `DiscreteQuotient X ⥤ Fintype` whose limit is isomorphic to `X`. -/
def fintypeDiagram : DiscreteQuotient X ⥤ FintypeCat where
  obj S := @FintypeCat.of S (Fintype.ofFinite S)
  map f := DiscreteQuotient.ofLE f.le
  -- Porting note: `map_comp` used to be proved by default by `aesop_cat`.
  -- once `aesop_cat` can prove this again, remove the entire `map_comp` here.
                     /-
                       X : Profinite
                       X✝ Y✝ Z✝ : DiscreteQuotient ↑X.toTop
                       x✝¹ : Quiver.Hom X✝ Y✝
                       x✝ : Quiver.Hom Y✝ Z✝
                       ⊢ Eq ({ obj := fun S => FintypeCat.of (Quotient S.toSetoid), map := fun {X_1 Y …
                     -/
  map_comp _ _ := by ext; aesop_cat
                          /-
                            🎉 no goals
                          -/


/-- An abbreviation for `X.fintypeDiagram ⋙ FintypeCat.toProfinite`. -/
abbrev diagram : DiscreteQuotient X ⥤ Profinite :=
  X.fintypeDiagram ⋙ FintypeCat.toProfinite


/-- A cone over `X.diagram` whose cone point is `X`. -/
def asLimitCone : CategoryTheory.Limits.Cone X.diagram :=
  { pt := X
    π := { app := fun S => ⟨S.proj, IsLocallyConstant.continuous (S.proj_isLocallyConstant)⟩ } }


instance isIso_asLimitCone_lift : IsIso ((limitConeIsLimit.{u, u} X.diagram).lift X.asLimitCone) :=
  CompHausLike.isIso_of_bijective _
    (by
      /-
        X : Profinite
        ⊢ Function.Bijective ⇑((Profinite.limitConeIsLimit X.diagram).lift X.asLimitCo …
      -/
      refine ⟨fun a b h => ?_, fun a => ?_⟩
        /-
          case refine_1
          X : Profinite
          a b : (CategoryTheory.forget (CompHausLike fun X => TotallyDisconnectedSpace ↑ …
          h : Eq (((Profinite.limitConeIsLimit X.diagram).lift X.asLimitCone) a) (((Prof …
          ⊢ Eq a b
        -/
      · refine DiscreteQuotient.eq_of_forall_proj_eq fun S => ?_
        /-
          case refine_1
          X : Profinite
          a b : (CategoryTheory.forget (CompHausLike fun X => TotallyDisconnectedSpace ↑ …
          h : Eq (((Profinite.limitConeIsLimit X.diagram).lift X.asLimitCone) a) (((Prof …
          S : DiscreteQuotient ((CategoryTheory.forget (CompHausLike fun X => TotallyDis …
          ⊢ Eq (S.proj a) (S.proj b)
        -/
        apply_fun fun f : (limitCone.{u, u} X.diagram).pt => f.val S at h
        /-
          case refine_1
          X : Profinite
          a b : (CategoryTheory.forget (CompHausLike fun X => TotallyDisconnectedSpace ↑ …
          S : DiscreteQuotient ((CategoryTheory.forget (CompHausLike fun X => TotallyDis …
          h : Eq (↑(((Profinite.limitConeIsLimit X.diagram).lift X.asLimitCone) a) S) (↑ …
          ⊢ Eq (S.proj a) (S.proj b)
        -/
        exact h
        /-
          🎉 no goals
        -/
      · obtain ⟨b, hb⟩ :=
          DiscreteQuotient.exists_of_compat (fun S => a.val S) fun _ _ h => a.prop (homOfLE h)
        /-
          case refine_2.intro
          X : Profinite
          a : (CategoryTheory.forget (CompHausLike fun X => TotallyDisconnectedSpace ↑X) …
          b : ↑X.toTop
          hb : ∀ (Q : DiscreteQuotient ↑X.toTop), Eq (Q.proj b) (↑a Q)
          ⊢ Exists fun a_1 => Eq (((Profinite.limitConeIsLimit X.diagram).lift X.asLimit …
        -/
        use b
        -- ext S : 3 -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11041): `ext` does not work, replaced with following
        -- three lines.
        /-
          case h
          X : Profinite
          a : (CategoryTheory.forget (CompHausLike fun X => TotallyDisconnectedSpace ↑X) …
          b : ↑X.toTop
          hb : ∀ (Q : DiscreteQuotient ↑X.toTop), Eq (Q.proj b) (↑a Q)
          ⊢ Eq (((Profinite.limitConeIsLimit X.diagram).lift X.asLimitCone) b) a
        -/
        apply Subtype.ext
        /-
          case h.a
          X : Profinite
          a : (CategoryTheory.forget (CompHausLike fun X => TotallyDisconnectedSpace ↑X) …
          b : ↑X.toTop
          hb : ∀ (Q : DiscreteQuotient ↑X.toTop), Eq (Q.proj b) (↑a Q)
          ⊢ Eq ↑(((Profinite.limitConeIsLimit X.diagram).lift X.asLimitCone) b) ↑a
        -/
        apply funext
        /-
          case h.a.h
          X : Profinite
          a : (CategoryTheory.forget (CompHausLike fun X => TotallyDisconnectedSpace ↑X) …
          b : ↑X.toTop
          hb : ∀ (Q : DiscreteQuotient ↑X.toTop), Eq (Q.proj b) (↑a Q)
          ⊢ ∀ (x : DiscreteQuotient ↑X.toTop), Eq (↑(((Profinite.limitConeIsLimit X.diag …
        -/
        rintro S
        -- Porting note: end replacement block
        /-
          case h.a.h
          X : Profinite
          a : (CategoryTheory.forget (CompHausLike fun X => TotallyDisconnectedSpace ↑X) …
          b : ↑X.toTop
          hb : ∀ (Q : DiscreteQuotient ↑X.toTop), Eq (Q.proj b) (↑a Q)
          S : DiscreteQuotient ↑X.toTop
          ⊢ Eq (↑(((Profinite.limitConeIsLimit X.diagram).lift X.asLimitCone) b) S) (↑a S)
        -/
        apply hb
        /-
          🎉 no goals
        -/
    )


/-- The isomorphism between `X` and the explicit limit of `X.diagram`,
induced by lifting `X.asLimitCone`.
-/
def isoAsLimitConeLift : X ≅ (limitCone.{u, u} X.diagram).pt :=
  asIso <| (limitConeIsLimit.{u, u} _).lift X.asLimitCone


/-- The isomorphism of cones `X.asLimitCone` and `Profinite.limitCone X.diagram`.
The underlying isomorphism is defeq to `X.isoAsLimitConeLift`.
-/
def asLimitConeIso : X.asLimitCone ≅ limitCone.{u, u} _ :=
  Limits.Cones.ext (isoAsLimitConeLift _) fun _ => rfl


/-- `X.asLimitCone` is indeed a limit cone. -/
def asLimit : CategoryTheory.Limits.IsLimit X.asLimitCone :=
  Limits.IsLimit.ofIsoLimit (limitConeIsLimit _) X.asLimitConeIso.symm


/-- A bundled version of `X.asLimitCone` and `X.asLimit`. -/
def lim : Limits.LimitCone X.diagram :=
  ⟨X.asLimitCone, X.asLimit⟩


