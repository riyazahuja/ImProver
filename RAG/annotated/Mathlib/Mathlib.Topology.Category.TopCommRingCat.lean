/-- A bundled topological commutative ring. -/
structure TopCommRingCat where
  /-- carrier of a topological commutative ring. -/
  α : Type u
  [isCommRing : CommRing α]
  [isTopologicalSpace : TopologicalSpace α]
  [isTopologicalRing : TopologicalRing α]


instance : Inhabited TopCommRingCat :=
  ⟨⟨PUnit⟩⟩


instance : CoeSort TopCommRingCat (Type u) :=
  ⟨TopCommRingCat.α⟩


instance : Category TopCommRingCat.{u} where
  Hom R S := { f : R →+* S // Continuous f }
                            /-
                              R : TopCommRingCat
                              ⊢ Continuous ⇑(RingHom.id R.α)
                            -/
  id R := ⟨RingHom.id R, by rw [RingHom.id]; continuity⟩
                                             /-
                                               🎉 no goals
                                             -/
  comp f g :=
    ⟨g.val.comp f.val, by
      -- TODO automate
      /-
        X✝ Y✝ Z✝ : TopCommRingCat
        f : Quiver.Hom X✝ Y✝
        g : Quiver.Hom Y✝ Z✝
        ⊢ Continuous ⇑((↑g).comp ↑f)
      -/
      cases f
      /-
        case mk
        X✝ Y✝ Z✝ : TopCommRingCat
        g : Quiver.Hom Y✝ Z✝
        val✝ : RingHom X✝.α Y✝.α
        property✝ : Continuous ⇑val✝
        ⊢ Continuous ⇑((↑g).comp ↑⟨val✝, property✝⟩)
      -/
      cases g
      /-
        case mk.mk
        X✝ Y✝ Z✝ : TopCommRingCat
        val✝¹ : RingHom X✝.α Y✝.α
        property✝¹ : Continuous ⇑val✝¹
        val✝ : RingHom Y✝.α Z✝.α
        property✝ : Continuous ⇑val✝
        ⊢ Continuous ⇑((↑⟨val✝, property✝⟩).comp ↑⟨val✝¹, property✝¹⟩)
      -/
                                       /-
                                         🎉 no goals
                                       -/
      dsimp; apply Continuous.comp <;> assumption⟩
                                       /-
                                         🎉 no goals
                                       -/


instance : ConcreteCategory TopCommRingCat.{u} where
  forget :=
    { obj := fun R => R
      map := fun f => f.val }
  -- Porting note: Old proof was `forget_faithful := { }`
  forget_faithful :=
    { map_injective := fun {_ _ _ _} h => Subtype.ext <| RingHom.coe_inj h }


/-- Construct a bundled `TopCommRingCat` from the underlying type and the appropriate typeclasses.
-/
def of (X : Type u) [CommRing X] [TopologicalSpace X] [TopologicalRing X] : TopCommRingCat :=
  ⟨X⟩


@[simp]
theorem coe_of (X : Type u) [CommRing X] [TopologicalSpace X] [TopologicalRing X] :
    (of X : Type u) = X := rfl


instance forgetTopologicalSpace (R : TopCommRingCat) :
    TopologicalSpace ((forget TopCommRingCat).obj R) :=
  R.isTopologicalSpace


instance forgetCommRing (R : TopCommRingCat) : CommRing ((forget TopCommRingCat).obj R) :=
  R.isCommRing


instance forgetTopologicalRing (R : TopCommRingCat) :
    TopologicalRing ((forget TopCommRingCat).obj R) :=
  R.isTopologicalRing


instance hasForgetToCommRingCat : HasForget₂ TopCommRingCat CommRingCat :=
  HasForget₂.mk' (fun R => CommRingCat.of R) (fun _ => rfl)
    (fun f => CommRingCat.ofHom f.val) HEq.rfl


instance forgetToCommRingCatTopologicalSpace (R : TopCommRingCat) :
    TopologicalSpace ((forget₂ TopCommRingCat CommRingCat).obj R) :=
  R.isTopologicalSpace


/-- The forgetful functor to `TopCat`. -/
instance hasForgetToTopCat : HasForget₂ TopCommRingCat TopCat :=
  HasForget₂.mk' (fun R => TopCat.of R) (fun _ => rfl) (fun f => ⟨⇑f.1, f.2⟩) HEq.rfl


instance forgetToTopCatCommRing (R : TopCommRingCat) :
    CommRing ((forget₂ TopCommRingCat TopCat).obj R) :=
  R.isCommRing


instance forgetToTopCatTopologicalRing (R : TopCommRingCat) :
    TopologicalRing ((forget₂ TopCommRingCat TopCat).obj R) :=
  R.isTopologicalRing


/-- The forgetful functors to `Type` do not reflect isomorphisms,
but the forgetful functor from `TopCommRingCat` to `TopCat` does.
-/
instance : (forget₂ TopCommRingCat.{u} TopCat.{u}).ReflectsIsomorphisms where
  reflects {X Y} f _ := by
    -- We have an isomorphism in `TopCat`,
    /-
      X Y : TopCommRingCat
      f : Quiver.Hom X Y
      x✝ : CategoryTheory.IsIso ((CategoryTheory.forget₂ TopCommRingCat TopCat).map f)
      ⊢ CategoryTheory.IsIso f
    -/
    let i_Top := asIso ((forget₂ TopCommRingCat TopCat).map f)
    -- and a `RingEquiv`.
    /-
      X Y : TopCommRingCat
      f : Quiver.Hom X Y
      x✝ : CategoryTheory.IsIso ((CategoryTheory.forget₂ TopCommRingCat TopCat).map f)
      i_Top : CategoryTheory.Iso ((CategoryTheory.forget₂ TopCommRingCat TopCat).obj …
      ⊢ CategoryTheory.IsIso f
    -/
    let e_Ring : X ≃+* Y := { f.1, ((forget TopCat).mapIso i_Top).toEquiv with }
    -- Putting these together we obtain the isomorphism we're after:
    exact
      ⟨⟨⟨e_Ring.symm, i_Top.inv.2⟩,
          ⟨by
            ext x
            exact e_Ring.left_inv x, by
            ext x
            exact e_Ring.right_inv x⟩⟩⟩


