attribute [local ext] Subtype.ext_val


/-- The image of a morphism in `ModuleCat R` is just the bundling of `LinearMap.range f` -/
def image : ModuleCat R :=
  ModuleCat.of R (LinearMap.range f.hom)


/-- The inclusion of `image f` into the target -/
def image.ι : image f ⟶ H :=
  ofHom f.hom.range.subtype


instance : Mono (image.ι f) :=
  ConcreteCategory.mono_of_injective (image.ι f) Subtype.val_injective


/-- The corestriction map to the image -/
def factorThruImage : G ⟶ image f :=
  ofHom f.hom.rangeRestrict


theorem image.fac : factorThruImage f ≫ image.ι f = f :=
  rfl


/-- The universal property for the image factorisation -/
noncomputable def image.lift (F' : MonoFactorisation f) : image f ⟶ F'.I :=
  ofHom
  { toFun := (fun x => F'.e (Classical.indefiniteDescription _ x.2).1 : image f → F'.I)
    map_add' := fun x y => by
      /-
        R : Type u
        inst✝ : Ring R
        G H : ModuleCat R
        f : Quiver.Hom G H
        F' : CategoryTheory.Limits.MonoFactorisation f
        x y : Subtype fun x => Membership.mem (LinearMap.range f.hom) x
        ⊢ Eq ((fun x => F'.e.hom ↑(Classical.indefiniteDescription (fun x_1 => Eq (f.h …
      -/
      apply (mono_iff_injective F'.m).1
        /-
          case a
          R : Type u
          inst✝ : Ring R
          G H : ModuleCat R
          f : Quiver.Hom G H
          F' : CategoryTheory.Limits.MonoFactorisation f
          x y : Subtype fun x => Membership.mem (LinearMap.range f.hom) x
          ⊢ CategoryTheory.Mono F'.m
        -/
      · infer_instance
        /-
          🎉 no goals
        -/
      /-
        case a
        R : Type u
        inst✝ : Ring R
        G H : ModuleCat R
        f : Quiver.Hom G H
        F' : CategoryTheory.Limits.MonoFactorisation f
        x y : Subtype fun x => Membership.mem (LinearMap.range f.hom) x
        ⊢ Eq (F'.m.hom ((fun x => F'.e.hom ↑(Classical.indefiniteDescription (fun x_1  …
      -/
      rw [LinearMap.map_add]
      /-
        case a
        R : Type u
        inst✝ : Ring R
        G H : ModuleCat R
        f : Quiver.Hom G H
        F' : CategoryTheory.Limits.MonoFactorisation f
        x y : Subtype fun x => Membership.mem (LinearMap.range f.hom) x
        ⊢ Eq (F'.m.hom ((fun x => F'.e.hom ↑(Classical.indefiniteDescription (fun x_1  …
      -/
      change (F'.e ≫ F'.m) _ = (F'.e ≫ F'.m) _ + (F'.e ≫ F'.m) _
      /-
        case a
        R : Type u
        inst✝ : Ring R
        G H : ModuleCat R
        f : Quiver.Hom G H
        F' : CategoryTheory.Limits.MonoFactorisation f
        x y : Subtype fun x => Membership.mem (LinearMap.range f.hom) x
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp F'.e F'.m).hom ↑(Classical.indefinit …
      -/
      simp_rw [F'.fac, (Classical.indefiniteDescription (fun z => f z = _) _).2]
      /-
        case a
        R : Type u
        inst✝ : Ring R
        G H : ModuleCat R
        f : Quiver.Hom G H
        F' : CategoryTheory.Limits.MonoFactorisation f
        x y : Subtype fun x => Membership.mem (LinearMap.range f.hom) x
        ⊢ Eq (↑(HAdd.hAdd x y)) (HAdd.hAdd ↑x ↑y)
      -/
      rfl
      /-
        🎉 no goals
      -/
    map_smul' := fun c x => by
      /-
        R : Type u
        inst✝ : Ring R
        G H : ModuleCat R
        f : Quiver.Hom G H
        F' : CategoryTheory.Limits.MonoFactorisation f
        c : R
        x : Subtype fun x => Membership.mem (LinearMap.range f.hom) x
        ⊢ Eq ({ toFun := fun x => F'.e.hom ↑(Classical.indefiniteDescription (fun x_1  …
      -/
      apply (mono_iff_injective F'.m).1
        /-
          case a
          R : Type u
          inst✝ : Ring R
          G H : ModuleCat R
          f : Quiver.Hom G H
          F' : CategoryTheory.Limits.MonoFactorisation f
          c : R
          x : Subtype fun x => Membership.mem (LinearMap.range f.hom) x
          ⊢ CategoryTheory.Mono F'.m
        -/
      · infer_instance
        /-
          🎉 no goals
        -/
      /-
        case a
        R : Type u
        inst✝ : Ring R
        G H : ModuleCat R
        f : Quiver.Hom G H
        F' : CategoryTheory.Limits.MonoFactorisation f
        c : R
        x : Subtype fun x => Membership.mem (LinearMap.range f.hom) x
        ⊢ Eq (F'.m.hom ({ toFun := fun x => F'.e.hom ↑(Classical.indefiniteDescription …
      -/
      rw [LinearMap.map_smul]
      /-
        case a
        R : Type u
        inst✝ : Ring R
        G H : ModuleCat R
        f : Quiver.Hom G H
        F' : CategoryTheory.Limits.MonoFactorisation f
        c : R
        x : Subtype fun x => Membership.mem (LinearMap.range f.hom) x
        ⊢ Eq (F'.m.hom ({ toFun := fun x => F'.e.hom ↑(Classical.indefiniteDescription …
      -/
      change (F'.e ≫ F'.m) _ = _ • (F'.e ≫ F'.m) _
      /-
        case a
        R : Type u
        inst✝ : Ring R
        G H : ModuleCat R
        f : Quiver.Hom G H
        F' : CategoryTheory.Limits.MonoFactorisation f
        c : R
        x : Subtype fun x => Membership.mem (LinearMap.range f.hom) x
        ⊢ Eq ((CategoryTheory.CategoryStruct.comp F'.e F'.m).hom ↑(Classical.indefinit …
      -/
      simp_rw [F'.fac, (Classical.indefiniteDescription (fun z => f z = _) _).2]
      /-
        case a
        R : Type u
        inst✝ : Ring R
        G H : ModuleCat R
        f : Quiver.Hom G H
        F' : CategoryTheory.Limits.MonoFactorisation f
        c : R
        x : Subtype fun x => Membership.mem (LinearMap.range f.hom) x
        ⊢ Eq (↑(HSMul.hSMul c x)) (HSMul.hSMul ((RingHom.id R) c) ↑x)
      -/
      rfl }
      /-
        🎉 no goals
      -/


theorem image.lift_fac (F' : MonoFactorisation f) : image.lift F' ≫ F'.m = image.ι f := by
  /-
    R : Type u
    inst✝ : Ring R
    G H : ModuleCat R
    f : Quiver.Hom G H
    F' : CategoryTheory.Limits.MonoFactorisation f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (ModuleCat.image.lift F') F'.m) (Modu …
  -/
  ext x
  /-
    case hf.h
    R : Type u
    inst✝ : Ring R
    G H : ModuleCat R
    f : Quiver.Hom G H
    F' : CategoryTheory.Limits.MonoFactorisation f
    x : ↑(ModuleCat.image f)
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (ModuleCat.image.lift F') F'.m).hom  …
  -/
  change (F'.e ≫ F'.m) _ = _
  /-
    case hf.h
    R : Type u
    inst✝ : Ring R
    G H : ModuleCat R
    f : Quiver.Hom G H
    F' : CategoryTheory.Limits.MonoFactorisation f
    x : ↑(ModuleCat.image f)
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp F'.e F'.m).hom ↑(Classical.indefinit …
  -/
  rw [F'.fac, (Classical.indefiniteDescription _ x.2).2]
  /-
    case hf.h
    R : Type u
    inst✝ : Ring R
    G H : ModuleCat R
    f : Quiver.Hom G H
    F' : CategoryTheory.Limits.MonoFactorisation f
    x : ↑(ModuleCat.image f)
    ⊢ Eq (↑x) ((ModuleCat.image.ι f).hom x)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The factorisation of any morphism in `ModuleCat R` through a mono. -/
def monoFactorisation : MonoFactorisation f where
  I := image f
  m := image.ι f
  e := factorThruImage f


/-- The factorisation of any morphism in `ModuleCat R` through a mono has the universal property of
the image. -/
noncomputable def isImage : IsImage (monoFactorisation f) where
  lift := image.lift
  lift_fac := image.lift_fac


/-- The categorical image of a morphism in `ModuleCat R` agrees with the linear algebraic range. -/
noncomputable def imageIsoRange {G H : ModuleCat.{v} R} (f : G ⟶ H) :
    Limits.image f ≅ ModuleCat.of R (LinearMap.range f.hom) :=
  IsImage.isoExt (Image.isImage f) (isImage f)


@[simp, reassoc, elementwise]
theorem imageIsoRange_inv_image_ι {G H : ModuleCat.{v} R} (f : G ⟶ H) :
    (imageIsoRange f).inv ≫ Limits.image.ι f = ModuleCat.ofHom f.hom.range.subtype :=
  IsImage.isoExt_inv_m _ _


@[simp, reassoc, elementwise]
theorem imageIsoRange_hom_subtype {G H : ModuleCat.{v} R} (f : G ⟶ H) :
    (imageIsoRange f).hom ≫ ModuleCat.ofHom f.hom.range.subtype = Limits.image.ι f := by
  /-
    R : Type u
    inst✝ : Ring R
    G H : ModuleCat R
    f : Quiver.Hom G H
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (ModuleCat.imageIsoRange f).hom (Modu …
  -/
  rw [← imageIsoRange_inv_image_ι f, Iso.hom_inv_id_assoc]
  /-
    🎉 no goals
  -/


