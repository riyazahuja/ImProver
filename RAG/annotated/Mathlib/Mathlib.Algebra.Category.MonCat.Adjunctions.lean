/-- The functor of adjoining a neutral element `one` to a semigroup.
 -/
@[to_additive (attr := simps) "The functor of adjoining a neutral element `zero` to a semigroup"]
def adjoinOne : Semigrp.{u} ⥤ MonCat.{u} where
  obj S := MonCat.of (WithOne S)
  map := WithOne.map
  map_id _ := WithOne.map_id
  map_comp := WithOne.map_comp


@[to_additive]
instance hasForgetToSemigroup : HasForget₂ MonCat Semigrp where
  forget₂ :=
    { obj := fun M => Semigrp.of M
      map := MonoidHom.toMulHom }


/-- The `adjoinOne`-forgetful adjunction from `Semigrp` to `MonCat`. -/
@[to_additive "The `adjoinZero`-forgetful adjunction from `AddSemigrp` to `AddMonCat`"]
def adjoinOneAdj : adjoinOne ⊣ forget₂ MonCat.{u} Semigrp.{u} :=
  Adjunction.mkOfHomEquiv
    { homEquiv := fun _ _ => WithOne.lift.symm
      homEquiv_naturality_left_symm := by
        /-
          ⊢ ∀ {X' X : Semigrp} {Y : MonCat} (f : Quiver.Hom X' X) (g : Quiver.Hom X ((Ca …
        -/
        intro S T M f g
        /-
          S T : Semigrp
          M : MonCat
          f : Quiver.Hom S T
          g : Quiver.Hom T ((CategoryTheory.forget₂ MonCat Semigrp).obj M)
          ⊢ Eq (((fun x x_1 => WithOne.lift.symm) S M).symm (CategoryTheory.CategoryStru …
        -/
        ext x
        /-
          case w
          S T : Semigrp
          M : MonCat
          f : Quiver.Hom S T
          g : Quiver.Hom T ((CategoryTheory.forget₂ MonCat Semigrp).obj M)
          x : ↑(MonCat.adjoinOne.obj S)
          ⊢ Eq ((((fun x x_1 => WithOne.lift.symm) S M).symm (CategoryTheory.CategoryStr …
        -/
        simp only [Equiv.symm_symm, adjoinOne_map, coe_comp]
        /-
          case w
          S T : Semigrp
          M : MonCat
          f : Quiver.Hom S T
          g : Quiver.Hom T ((CategoryTheory.forget₂ MonCat Semigrp).obj M)
          x : ↑(MonCat.adjoinOne.obj S)
          ⊢ Eq ((WithOne.lift (CategoryTheory.CategoryStruct.comp f g)) x) (Function.com …
        -/
        simp_rw [WithOne.map]
        /-
          case w
          S T : Semigrp
          M : MonCat
          f : Quiver.Hom S T
          g : Quiver.Hom T ((CategoryTheory.forget₂ MonCat Semigrp).obj M)
          x : ↑(MonCat.adjoinOne.obj S)
          ⊢ Eq ((WithOne.lift (CategoryTheory.CategoryStruct.comp f g)) x) (Function.com …
        -/
        cases x
          /-
            case w.none
            S T : Semigrp
            M : MonCat
            f : Quiver.Hom S T
            g : Quiver.Hom T ((CategoryTheory.forget₂ MonCat Semigrp).obj M)
            ⊢ Eq ((WithOne.lift (CategoryTheory.CategoryStruct.comp f g)) Option.none) (Fu …
          -/
        · rfl
          /-
            🎉 no goals
          -/
          /-
            case w.some
            S T : Semigrp
            M : MonCat
            f : Quiver.Hom S T
            g : Quiver.Hom T ((CategoryTheory.forget₂ MonCat Semigrp).obj M)
            val✝ : ↑S
            ⊢ Eq ((WithOne.lift (CategoryTheory.CategoryStruct.comp f g)) (Option.some val …
          -/
        · simp
          /-
            case w.some
            S T : Semigrp
            M : MonCat
            f : Quiver.Hom S T
            g : Quiver.Hom T ((CategoryTheory.forget₂ MonCat Semigrp).obj M)
            val✝ : ↑S
            ⊢ Eq ((WithOne.lift (CategoryTheory.CategoryStruct.comp f g)) (Option.some val …
          -/
          rfl }
          /-
            🎉 no goals
          -/


/-- The free functor `Type u ⥤ MonCat` sending a type `X` to the free monoid on `X`. -/
def free : Type u ⥤ MonCat.{u} where
  obj α := MonCat.of (FreeMonoid α)
  map := FreeMonoid.map
  map_id _ := FreeMonoid.hom_eq (fun _ => rfl)
  map_comp _ _ := FreeMonoid.hom_eq (fun _ => rfl)


/-- The free-forgetful adjunction for monoids. -/
def adj : free ⊣ forget MonCat.{u} :=
  Adjunction.mkOfHomEquiv
    { homEquiv := fun _ _ => FreeMonoid.lift.symm
      homEquiv_naturality_left_symm := fun _ _ => FreeMonoid.hom_eq (fun _ => rfl) }


instance : (forget MonCat.{u}).IsRightAdjoint :=
  ⟨_, ⟨adj⟩⟩


