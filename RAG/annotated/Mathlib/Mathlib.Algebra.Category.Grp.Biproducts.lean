instance : HasBinaryBiproducts AddCommGrp :=
  HasBinaryBiproducts.of_hasBinaryProducts


instance : HasFiniteBiproducts AddCommGrp :=
  HasFiniteBiproducts.of_hasFiniteProducts

-- We now construct explicit limit data,
-- so we can compare the biproducts to the usual unbundled constructions.

/-- Construct limit data for a binary product in `AddCommGrp`, using
`AddCommGrp.of (G × H)`.
-/
@[simps cone_pt isLimit_lift]
def binaryProductLimitCone (G H : AddCommGrp.{u}) : Limits.LimitCone (pair G H) where
  cone :=
    { pt := AddCommGrp.of (G × H)
      π :=
        { app := fun j =>
            Discrete.casesOn j fun j =>
              WalkingPair.casesOn j (AddMonoidHom.fst G H) (AddMonoidHom.snd G H)
                           /-
                             G H : AddCommGrp
                             ⊢ ∀ ⦃X Y : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair⦄ (f : Qui …
                           -/
                                                       /-
                                                         🎉 no goals
                                                       -/
          naturality := by rintro ⟨⟨⟩⟩ ⟨⟨⟩⟩ ⟨⟨⟨⟩⟩⟩ <;> rfl } }
                                                       /-
                                                         🎉 no goals
                                                       -/
  isLimit :=
    { lift := fun s => AddMonoidHom.prod (s.π.app ⟨WalkingPair.left⟩) (s.π.app ⟨WalkingPair.right⟩)
                /-
                  G H : AddCommGrp
                  ⊢ ∀ (s : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair G H)) (j : Cat …
                -/
                                       /-
                                         🎉 no goals
                                       -/
      fac := by rintro s (⟨⟩ | ⟨⟩) <;> rfl
                                       /-
                                         🎉 no goals
                                       -/
      uniq := fun s m w => by
        /-
          G H : AddCommGrp
          s : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair G H)
          m : Quiver.Hom s.pt { pt := AddCommGrp.of (Prod ↑G ↑H), π := { app := fun j => …
          w : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Cat …
          ⊢ Eq m ((fun s => AddMonoidHom.prod (s.π.app { as := CategoryTheory.Limits.Wal …
        -/
        simp_rw [← w ⟨WalkingPair.left⟩, ← w ⟨WalkingPair.right⟩]
        /-
          G H : AddCommGrp
          s : CategoryTheory.Limits.Cone (CategoryTheory.Limits.pair G H)
          m : Quiver.Hom s.pt { pt := AddCommGrp.of (Prod ↑G ↑H), π := { app := fun j => …
          w : ∀ (j : CategoryTheory.Discrete CategoryTheory.Limits.WalkingPair), Eq (Cat …
          ⊢ Eq m (AddMonoidHom.prod (CategoryTheory.CategoryStruct.comp m (AddMonoidHom. …
        -/
        rfl }
        /-
          🎉 no goals
        -/


@[simp]
theorem binaryProductLimitCone_cone_π_app_left (G H : AddCommGrp.{u}) :
    (binaryProductLimitCone G H).cone.π.app ⟨WalkingPair.left⟩ = AddMonoidHom.fst G H :=
  rfl


@[simp]
theorem binaryProductLimitCone_cone_π_app_right (G H : AddCommGrp.{u}) :
    (binaryProductLimitCone G H).cone.π.app ⟨WalkingPair.right⟩ = AddMonoidHom.snd G H :=
  rfl


/-- We verify that the biproduct in `AddCommGrp` is isomorphic to
the cartesian product of the underlying types:
-/
noncomputable def biprodIsoProd (G H : AddCommGrp.{u}) :
    (G ⊞ H : AddCommGrp) ≅ AddCommGrp.of (G × H) :=
  IsLimit.conePointUniqueUpToIso (BinaryBiproduct.isLimit G H) (binaryProductLimitCone G H).isLimit


@[simp, elementwise]
theorem biprodIsoProd_inv_comp_fst (G H : AddCommGrp.{u}) :
    (biprodIsoProd G H).inv ≫ biprod.fst = AddMonoidHom.fst G H :=
  IsLimit.conePointUniqueUpToIso_inv_comp _ _ (Discrete.mk WalkingPair.left)


@[simp, elementwise]
theorem biprodIsoProd_inv_comp_snd (G H : AddCommGrp.{u}) :
    (biprodIsoProd G H).inv ≫ biprod.snd = AddMonoidHom.snd G H :=
  IsLimit.conePointUniqueUpToIso_inv_comp _ _ (Discrete.mk WalkingPair.right)


/-- The map from an arbitrary cone over an indexed family of abelian groups
to the cartesian product of those groups.
-/
-- This was marked `@[simps]` until we made `AddCommGrp.coe_of` a simp lemma,
-- after which the simp normal form linter complains.
-- The generated simp lemmas were not used in Mathlib.
-- Possible solution: higher priority function coercions that remove the `of`?
-- @[simps]
def lift (s : Fan f) : s.pt ⟶ AddCommGrp.of (∀ j, f j) where
  toFun x j := s.π.app ⟨j⟩ x
  map_zero' := by
    /-
      J : Type w
      f : J → AddCommGrp
      s : CategoryTheory.Limits.Fan f
      ⊢ Eq ((fun x j => (s.π.app { as := j }) x) 0) 0
    -/
    simp only [Functor.const_obj_obj, map_zero]
    /-
      J : Type w
      f : J → AddCommGrp
      s : CategoryTheory.Limits.Fan f
      ⊢ Eq (fun j => 0) 0
    -/
    rfl
    /-
      🎉 no goals
    -/
  map_add' x y := by
    /-
      J : Type w
      f : J → AddCommGrp
      s : CategoryTheory.Limits.Fan f
      x y : ↑s.pt
      ⊢ Eq ({ toFun := fun x j => (s.π.app { as := j }) x, map_zero' := ⋯ }.toFun (H …
    -/
    simp only [Functor.const_obj_obj, map_add]
    /-
      J : Type w
      f : J → AddCommGrp
      s : CategoryTheory.Limits.Fan f
      x y : ↑s.pt
      ⊢ Eq (fun j => HAdd.hAdd ((s.π.app { as := j }) x) ((s.π.app { as := j }) y))  …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- Construct limit data for a product in `AddCommGrp`, using
`AddCommGrp.of (∀ j, F.obj j)`.
-/
@[simps]
def productLimitCone : Limits.LimitCone (Discrete.functor f) where
  cone :=
    { pt := AddCommGrp.of (∀ j, f j)
      π := Discrete.natTrans fun j => Pi.evalAddMonoidHom (fun j => f j) j.as }
  isLimit :=
    { lift := lift.{_, u} f
      fac := fun _ _ => rfl
      uniq := fun s m w => by
        /-
          J : Type w
          f : J → AddCommGrp
          s : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor f)
          m : Quiver.Hom s.pt { pt := AddCommGrp.of ((j : J) → ↑(f j)), π := CategoryThe …
          w : ∀ (j : CategoryTheory.Discrete J), Eq (CategoryTheory.CategoryStruct.comp  …
          ⊢ Eq m (AddCommGrp.HasLimit.lift f s)
        -/
        ext x
        /-
          case w
          J : Type w
          f : J → AddCommGrp
          s : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor f)
          m : Quiver.Hom s.pt { pt := AddCommGrp.of ((j : J) → ↑(f j)), π := CategoryThe …
          w : ∀ (j : CategoryTheory.Discrete J), Eq (CategoryTheory.CategoryStruct.comp  …
          x : ↑s.pt
          ⊢ Eq (m x) ((AddCommGrp.HasLimit.lift f s) x)
        -/
        funext j
        /-
          case w.h
          J : Type w
          f : J → AddCommGrp
          s : CategoryTheory.Limits.Cone (CategoryTheory.Discrete.functor f)
          m : Quiver.Hom s.pt { pt := AddCommGrp.of ((j : J) → ↑(f j)), π := CategoryThe …
          w : ∀ (j : CategoryTheory.Discrete J), Eq (CategoryTheory.CategoryStruct.comp  …
          x : ↑s.pt
          j : J
          ⊢ Eq (m x j) ((AddCommGrp.HasLimit.lift f s) x j)
        -/
        exact congr_arg (fun g : s.pt ⟶ f j => (g : s.pt → f j) x) (w ⟨j⟩) }
        /-
          🎉 no goals
        -/


/-- We verify that the biproduct we've just defined is isomorphic to the `AddCommGrp` structure
on the dependent function type.
-/
noncomputable def biproductIsoPi (f : J → AddCommGrp.{u}) :
    (⨁ f : AddCommGrp) ≅ AddCommGrp.of (∀ j, f j) :=
  IsLimit.conePointUniqueUpToIso (biproduct.isLimit f) (productLimitCone f).isLimit


@[simp, elementwise]
theorem biproductIsoPi_inv_comp_π (f : J → AddCommGrp.{u}) (j : J) :
    (biproductIsoPi f).inv ≫ biproduct.π f j = Pi.evalAddMonoidHom (fun j => f j) j :=
  IsLimit.conePointUniqueUpToIso_inv_comp _ _ (Discrete.mk j)


