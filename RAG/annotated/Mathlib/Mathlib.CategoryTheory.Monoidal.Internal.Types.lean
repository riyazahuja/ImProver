instance monMonoid (A : Mon_ (Type u)) : Monoid A.X where
  one := A.one PUnit.unit
  mul x y := A.mul (x, y)
                  /-
                    A : Mon_ (Type u)
                    x : A.X
                    ⊢ Eq (HMul.hMul 1 x) x
                  -/
  one_mul x := by convert congr_fun A.one_mul (PUnit.unit, x)
                        /-
                          A : Mon_ (Type u)
                          x y z : A.X
                          ⊢ Eq (HMul.hMul (HMul.hMul x y) z) (HMul.hMul x (HMul.hMul y z))
                        -/
                  /-
                    🎉 no goals
                  -/
                        /-
                          🎉 no goals
                        -/
                  /-
                    A : Mon_ (Type u)
                    x : A.X
                    ⊢ Eq (HMul.hMul x 1) x
                  -/
  mul_one x := by convert congr_fun A.mul_one (x, PUnit.unit)
                  /-
                    🎉 no goals
                  -/
  mul_assoc x y z := by convert congr_fun A.mul_assoc ((x, y), z)


/-- Converting a monoid object in `Type` to a bundled monoid.
-/
noncomputable def functor : Mon_ (Type u) ⥤ MonCat.{u} where
  obj A := MonCat.of A.X
  map f :=
    { toFun := f.hom
      map_one' := congr_fun f.one_hom PUnit.unit
      map_mul' := fun x y => congr_fun f.mul_hom (x, y) }


/-- Converting a bundled monoid to a monoid object in `Type`.
-/
noncomputable def inverse : MonCat.{u} ⥤ Mon_ (Type u) where
  obj A :=
    { X := A
      one := fun _ => 1
      mul := fun p => p.1 * p.2
                    /-
                      A : MonCat
                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                    -/
      one_mul := by ext ⟨_, _⟩; dsimp; simp
                                       /-
                                         🎉 no goals
                                       -/
                    /-
                      A : MonCat
                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                    -/
      mul_one := by ext ⟨_, _⟩; dsimp; simp
                                       /-
                                         🎉 no goals
                                       -/
                      /-
                        A : MonCat
                        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
                      -/
      mul_assoc := by ext ⟨⟨x, y⟩, z⟩; simp [mul_assoc] }
                                       /-
                                         🎉 no goals
                                       -/
  map f := { hom := f }


/-- The category of internal monoid objects in `Type`
is equivalent to the category of "native" bundled monoids.
-/
noncomputable def monTypeEquivalenceMon : Mon_ (Type u) ≌ MonCat.{u} where
  functor := functor
  inverse := inverse
  unitIso :=
    NatIso.ofComponents
      (fun A =>
        { hom := { hom := 𝟙 _ }
          inv := { hom := 𝟙 _ } })
          /-
            ⊢ ∀ {X Y : Mon_ (Type u)} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStr …
          -/
      (by aesop_cat)
          /-
            🎉 no goals
          -/
  counitIso :=
    NatIso.ofComponents
      (fun A =>
        { hom :=
            { toFun := id
              map_one' := rfl
              map_mul' := fun _ _ => rfl }
          inv :=
            { toFun := id
              map_one' := rfl
              map_mul' := fun _ _ => rfl } })
          /-
            ⊢ ∀ {X Y : MonCat} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct.com …
          -/
      (by aesop_cat)
          /-
            🎉 no goals
          -/


/-- The equivalence `Mon_ (Type u) ≌ MonCat.{u}`
is naturally compatible with the forgetful functors to `Type u`.
-/
noncomputable def monTypeEquivalenceMonForget :
    MonTypeEquivalenceMon.functor ⋙ forget MonCat ≅ Mon_.forget (Type u) :=
                                                /-
                                                  ⊢ ∀ {X Y : Mon_ (Type u)} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStr …
                                                -/
  NatIso.ofComponents (fun _ => Iso.refl _) (by aesop_cat)
                                                /-
                                                  🎉 no goals
                                                -/


noncomputable instance monTypeInhabited : Inhabited (Mon_ (Type u)) :=
  ⟨MonTypeEquivalenceMon.inverse.obj (MonCat.of PUnit)⟩


instance commMonCommMonoid (A : CommMon_ (Type u)) : CommMonoid A.X :=
  { MonTypeEquivalenceMon.monMonoid A.toMon_ with
                              /-
                                A : CommMon_ (Type u)
                                x y : A.X
                                ⊢ Eq (HMul.hMul x y) (HMul.hMul y x)
                              -/
    mul_comm := fun x y => by convert congr_fun A.mul_comm (y, x) }
                              /-
                                🎉 no goals
                              -/


/-- Converting a commutative monoid object in `Type` to a bundled commutative monoid.
-/
noncomputable def functor : CommMon_ (Type u) ⥤ CommMonCat.{u} where
  obj A := CommMonCat.of A.X
  map f := MonTypeEquivalenceMon.functor.map f


/-- Converting a bundled commutative monoid to a commutative monoid object in `Type`.
-/
noncomputable def inverse : CommMonCat.{u} ⥤ CommMon_ (Type u) where
  obj A :=
    { MonTypeEquivalenceMon.inverse.obj ((forget₂ CommMonCat MonCat).obj A) with
      mul_comm := by
        /-
          A : CommMonCat
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.BraidedCategory.braid …
        -/
        ext ⟨x : A, y : A⟩
        /-
          case h.mk
          A : CommMonCat
          x y : ↑A
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.BraidedCategory.braid …
        -/
        exact CommMonoid.mul_comm y x }
        /-
          🎉 no goals
        -/
  map f := MonTypeEquivalenceMon.inverse.map ((forget₂ CommMonCat MonCat).map f)


/-- The category of internal commutative monoid objects in `Type`
is equivalent to the category of "native" bundled commutative monoids.
-/
noncomputable def commMonTypeEquivalenceCommMon : CommMon_ (Type u) ≌ CommMonCat.{u} where
  functor := functor
  inverse := inverse
  unitIso :=
    NatIso.ofComponents
      (fun A =>
        { hom := { hom := 𝟙 _ }
          inv := { hom := 𝟙 _ } })
          /-
            ⊢ ∀ {X Y : CommMon_ (Type u)} (f : Quiver.Hom X Y), Eq (CategoryTheory.Categor …
          -/
      (by aesop_cat)
          /-
            🎉 no goals
          -/
  counitIso :=
    NatIso.ofComponents
      (fun A =>
        { hom :=
            { toFun := id
              map_one' := rfl
              map_mul' := fun _ _ => rfl }
          inv :=
            { toFun := id
              map_one' := rfl
              map_mul' := fun _ _ => rfl } })
          /-
            ⊢ ∀ {X Y : CommMonCat} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct …
          -/
      (by aesop_cat)
          /-
            🎉 no goals
          -/


/-- The equivalences `Mon_ (Type u) ≌ MonCat.{u}` and `CommMon_ (Type u) ≌ CommMonCat.{u}`
are naturally compatible with the forgetful functors to `MonCat` and `Mon_ (Type u)`.
-/
noncomputable def commMonTypeEquivalenceCommMonForget :
    CommMonTypeEquivalenceCommMon.functor ⋙ forget₂ CommMonCat MonCat ≅
      CommMon_.forget₂Mon_ (Type u) ⋙ MonTypeEquivalenceMon.functor :=
                                                /-
                                                  ⊢ ∀ {X Y : CommMon_ (Type u)} (f : Quiver.Hom X Y), Eq (CategoryTheory.Categor …
                                                -/
  NatIso.ofComponents (fun _ => Iso.refl _) (by aesop_cat)
                                                /-
                                                  🎉 no goals
                                                -/


noncomputable instance commMonTypeInhabited : Inhabited (CommMon_ (Type u)) :=
  ⟨CommMonTypeEquivalenceCommMon.inverse.obj (CommMonCat.of PUnit)⟩

