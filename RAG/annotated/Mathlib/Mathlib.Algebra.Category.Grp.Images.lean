attribute [local ext] Subtype.ext_val


/-- the image of a morphism in `AddCommGrp` is just the bundling of `AddMonoidHom.range f` -/
def image : AddCommGrp :=
  AddCommGrp.of (AddMonoidHom.range f)


/-- the inclusion of `image f` into the target -/
def image.ι : image f ⟶ H :=
  f.range.subtype


instance : Mono (image.ι f) :=
  ConcreteCategory.mono_of_injective (image.ι f) Subtype.val_injective


/-- the corestriction map to the image -/
def factorThruImage : G ⟶ image f :=
  f.rangeRestrict


theorem image.fac : factorThruImage f ≫ image.ι f = f := by
  /-
    G H : AddCommGrp
    f : Quiver.Hom G H
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AddCommGrp.factorThruImage f) (AddCo …
  -/
  ext
  /-
    case w
    G H : AddCommGrp
    f : Quiver.Hom G H
    x✝ : ↑G
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (AddCommGrp.factorThruImage f) (AddC …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- the universal property for the image factorisation -/
noncomputable def image.lift (F' : MonoFactorisation f) : image f ⟶ F'.I where
  toFun := (fun x => F'.e (Classical.indefiniteDescription _ x.2).1 : image f → F'.I)
  map_zero' := by
    /-
      G H : AddCommGrp
      f : Quiver.Hom G H
      F' : CategoryTheory.Limits.MonoFactorisation f
      ⊢ Eq ((fun x => F'.e ↑(Classical.indefiniteDescription (fun x_1 => Eq (f x_1)  …
    -/
    haveI := F'.m_mono
    /-
      G H : AddCommGrp
      f : Quiver.Hom G H
      F' : CategoryTheory.Limits.MonoFactorisation f
      this : CategoryTheory.Mono F'.m
      ⊢ Eq ((fun x => F'.e ↑(Classical.indefiniteDescription (fun x_1 => Eq (f x_1)  …
    -/
    apply injective_of_mono F'.m
    /-
      case a
      G H : AddCommGrp
      f : Quiver.Hom G H
      F' : CategoryTheory.Limits.MonoFactorisation f
      this : CategoryTheory.Mono F'.m
      ⊢ Eq (F'.m ((fun x => F'.e ↑(Classical.indefiniteDescription (fun x_1 => Eq (f …
    -/
    change (F'.e ≫ F'.m) _ = _
    /-
      case a
      G H : AddCommGrp
      f : Quiver.Hom G H
      F' : CategoryTheory.Limits.MonoFactorisation f
      this : CategoryTheory.Mono F'.m
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp F'.e F'.m) ↑(Classical.indefiniteDes …
    -/
    rw [F'.fac, AddMonoidHom.map_zero]
    /-
      case a
      G H : AddCommGrp
      f : Quiver.Hom G H
      F' : CategoryTheory.Limits.MonoFactorisation f
      this : CategoryTheory.Mono F'.m
      ⊢ Eq (f ↑(Classical.indefiniteDescription (fun x => Eq (f x) ↑0) ⋯)) 0
    -/
    exact (Classical.indefiniteDescription (fun y => f y = 0) _).2
    /-
      🎉 no goals
    -/
  map_add' := by
    /-
      G H : AddCommGrp
      f : Quiver.Hom G H
      F' : CategoryTheory.Limits.MonoFactorisation f
      ⊢ ∀ (x y : ↑(AddCommGrp.image f)), Eq ({ toFun := fun x => F'.e ↑(Classical.in …
    -/
    intro x y
    /-
      G H : AddCommGrp
      f : Quiver.Hom G H
      F' : CategoryTheory.Limits.MonoFactorisation f
      x y : ↑(AddCommGrp.image f)
      ⊢ Eq ({ toFun := fun x => F'.e ↑(Classical.indefiniteDescription (fun x_1 => E …
    -/
    haveI := F'.m_mono
    /-
      G H : AddCommGrp
      f : Quiver.Hom G H
      F' : CategoryTheory.Limits.MonoFactorisation f
      x y : ↑(AddCommGrp.image f)
      this : CategoryTheory.Mono F'.m
      ⊢ Eq ({ toFun := fun x => F'.e ↑(Classical.indefiniteDescription (fun x_1 => E …
    -/
    apply injective_of_mono F'.m
    /-
      case a
      G H : AddCommGrp
      f : Quiver.Hom G H
      F' : CategoryTheory.Limits.MonoFactorisation f
      x y : ↑(AddCommGrp.image f)
      this : CategoryTheory.Mono F'.m
      ⊢ Eq (F'.m ({ toFun := fun x => F'.e ↑(Classical.indefiniteDescription (fun x_ …
    -/
    rw [AddMonoidHom.map_add]
    /-
      case a
      G H : AddCommGrp
      f : Quiver.Hom G H
      F' : CategoryTheory.Limits.MonoFactorisation f
      x y : ↑(AddCommGrp.image f)
      this : CategoryTheory.Mono F'.m
      ⊢ Eq (F'.m ({ toFun := fun x => F'.e ↑(Classical.indefiniteDescription (fun x_ …
    -/
    change (F'.e ≫ F'.m) _ = (F'.e ≫ F'.m) _ + (F'.e ≫ F'.m) _
    /-
      case a
      G H : AddCommGrp
      f : Quiver.Hom G H
      F' : CategoryTheory.Limits.MonoFactorisation f
      x y : ↑(AddCommGrp.image f)
      this : CategoryTheory.Mono F'.m
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp F'.e F'.m) ↑(Classical.indefiniteDes …
    -/
    rw [F'.fac]
    /-
      case a
      G H : AddCommGrp
      f : Quiver.Hom G H
      F' : CategoryTheory.Limits.MonoFactorisation f
      x y : ↑(AddCommGrp.image f)
      this : CategoryTheory.Mono F'.m
      ⊢ Eq (f ↑(Classical.indefiniteDescription (fun x_1 => Eq (f x_1) ↑(HAdd.hAdd x …
    -/
    rw [(Classical.indefiniteDescription (fun z => f z = _) _).2]
    /-
      case a
      G H : AddCommGrp
      f : Quiver.Hom G H
      F' : CategoryTheory.Limits.MonoFactorisation f
      x y : ↑(AddCommGrp.image f)
      this : CategoryTheory.Mono F'.m
      ⊢ Eq (↑(HAdd.hAdd x y)) (HAdd.hAdd (f ↑(Classical.indefiniteDescription (fun x …
    -/
    rw [(Classical.indefiniteDescription (fun z => f z = _) _).2]
    /-
      case a
      G H : AddCommGrp
      f : Quiver.Hom G H
      F' : CategoryTheory.Limits.MonoFactorisation f
      x y : ↑(AddCommGrp.image f)
      this : CategoryTheory.Mono F'.m
      ⊢ Eq (↑(HAdd.hAdd x y)) (HAdd.hAdd (↑x) (f ↑(Classical.indefiniteDescription ( …
    -/
    rw [(Classical.indefiniteDescription (fun z => f z = _) _).2]
    /-
      case a
      G H : AddCommGrp
      f : Quiver.Hom G H
      F' : CategoryTheory.Limits.MonoFactorisation f
      x y : ↑(AddCommGrp.image f)
      this : CategoryTheory.Mono F'.m
      ⊢ Eq (↑(HAdd.hAdd x y)) (HAdd.hAdd ↑x ↑y)
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem image.lift_fac (F' : MonoFactorisation f) : image.lift F' ≫ F'.m = image.ι f := by
  /-
    G H : AddCommGrp
    f : Quiver.Hom G H
    F' : CategoryTheory.Limits.MonoFactorisation f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AddCommGrp.image.lift F') F'.m) (Add …
  -/
  ext x
  /-
    case w
    G H : AddCommGrp
    f : Quiver.Hom G H
    F' : CategoryTheory.Limits.MonoFactorisation f
    x : ↑(AddCommGrp.image f)
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (AddCommGrp.image.lift F') F'.m) x)  …
  -/
  change (F'.e ≫ F'.m) _ = _
  /-
    case w
    G H : AddCommGrp
    f : Quiver.Hom G H
    F' : CategoryTheory.Limits.MonoFactorisation f
    x : ↑(AddCommGrp.image f)
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp F'.e F'.m) ↑(Classical.indefiniteDes …
  -/
  rw [F'.fac, (Classical.indefiniteDescription _ x.2).2]
  /-
    case w
    G H : AddCommGrp
    f : Quiver.Hom G H
    F' : CategoryTheory.Limits.MonoFactorisation f
    x : ↑(AddCommGrp.image f)
    ⊢ Eq (↑x) ((AddCommGrp.image.ι f) x)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- the factorisation of any morphism in `AddCommGrp` through a mono. -/
def monoFactorisation : MonoFactorisation f where
  I := image f
  m := image.ι f
  e := factorThruImage f


/-- the factorisation of any morphism in `AddCommGrp` through a mono has
the universal property of the image. -/
noncomputable def isImage : IsImage (monoFactorisation f) where
  lift := image.lift
  lift_fac := image.lift_fac


/-- The categorical image of a morphism in `AddCommGrp`
agrees with the usual group-theoretical range.
-/
noncomputable def imageIsoRange {G H : AddCommGrp.{0}} (f : G ⟶ H) :
    Limits.image f ≅ AddCommGrp.of f.range :=
  IsImage.isoExt (Image.isImage f) (isImage f)


