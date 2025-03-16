theorem injective_object_of_injective_module [inj : Injective R M] :
    CategoryTheory.Injective (ModuleCat.of R M) where
  factors g f m :=
    have ⟨l, h⟩ := inj.out f.hom ((ModuleCat.mono_iff_injective f).mp m) g.hom
                           /-
                             R : Type u
                             M : Type v
                             inst✝² : Ring R
                             inst✝¹ : AddCommGroup M
                             inst✝ : Module R M
                             inj : Module.Injective R M
                             X✝ Y✝ : ModuleCat R
                             g : Quiver.Hom X✝ (ModuleCat.of R M)
                             f : Quiver.Hom X✝ Y✝
                             m : CategoryTheory.Mono f
                             l : LinearMap (RingHom.id R) (↑Y✝) M
                             h : ∀ (x : ↑X✝), Eq (l (f.hom x)) (g.hom x)
                             ⊢ Eq (CategoryTheory.CategoryStruct.comp f (ModuleCat.ofHom l)) g
                           -/
    ⟨ModuleCat.ofHom l, by ext x; simpa using h x⟩
                                  /-
                                    🎉 no goals
                                  -/


theorem injective_module_of_injective_object
    [inj : CategoryTheory.Injective <| ModuleCat.of R M] :
    Module.Injective R M where
  out X Y _ _ _ _ f hf g := by
    /-
      R : Type u
      M : Type v
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      inj : CategoryTheory.Injective (ModuleCat.of R M)
      X Y : Type v
      x✝³ : AddCommGroup X
      x✝² : AddCommGroup Y
      x✝¹ : Module R X
      x✝ : Module R Y
      f : LinearMap (RingHom.id R) X Y
      hf : Function.Injective ⇑f
      g : LinearMap (RingHom.id R) X M
      ⊢ Exists fun h => ∀ (x : X), Eq (h (f x)) (g x)
    -/
    have : CategoryTheory.Mono (ModuleCat.ofHom f) := (ModuleCat.mono_iff_injective _).mpr hf
    /-
      R : Type u
      M : Type v
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      inj : CategoryTheory.Injective (ModuleCat.of R M)
      X Y : Type v
      x✝³ : AddCommGroup X
      x✝² : AddCommGroup Y
      x✝¹ : Module R X
      x✝ : Module R Y
      f : LinearMap (RingHom.id R) X Y
      hf : Function.Injective ⇑f
      g : LinearMap (RingHom.id R) X M
      this : CategoryTheory.Mono (ModuleCat.ofHom f)
      ⊢ Exists fun h => ∀ (x : X), Eq (h (f x)) (g x)
    -/
    obtain ⟨l, h⟩ := inj.factors (ModuleCat.ofHom g) (ModuleCat.ofHom f)
    /-
      case intro
      R : Type u
      M : Type v
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      inj : CategoryTheory.Injective (ModuleCat.of R M)
      X Y : Type v
      x✝³ : AddCommGroup X
      x✝² : AddCommGroup Y
      x✝¹ : Module R X
      x✝ : Module R Y
      f : LinearMap (RingHom.id R) X Y
      hf : Function.Injective ⇑f
      g : LinearMap (RingHom.id R) X M
      this : CategoryTheory.Mono (ModuleCat.ofHom f)
      l : Quiver.Hom (ModuleCat.of R Y) (ModuleCat.of R M)
      h : Eq (CategoryTheory.CategoryStruct.comp (ModuleCat.ofHom f) l) (ModuleCat.o …
      ⊢ Exists fun h => ∀ (x : X), Eq (h (f x)) (g x)
    -/
    obtain rfl := ModuleCat.hom_ext_iff.mp h
    /-
      case intro
      R : Type u
      M : Type v
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      inj : CategoryTheory.Injective (ModuleCat.of R M)
      X Y : Type v
      x✝³ : AddCommGroup X
      x✝² : AddCommGroup Y
      x✝¹ : Module R X
      x✝ : Module R Y
      f : LinearMap (RingHom.id R) X Y
      hf : Function.Injective ⇑f
      this : CategoryTheory.Mono (ModuleCat.ofHom f)
      l : Quiver.Hom (ModuleCat.of R Y) (ModuleCat.of R M)
      h : Eq (CategoryTheory.CategoryStruct.comp (ModuleCat.ofHom f) l) (ModuleCat.o …
      ⊢ Exists fun h => ∀ (x : X), Eq (h (f x)) ((CategoryTheory.CategoryStruct.comp …
    -/
    exact ⟨l.hom, fun _ => rfl⟩
    /-
      🎉 no goals
    -/


theorem injective_iff_injective_object :
    Module.Injective R M ↔
    CategoryTheory.Injective (ModuleCat.of R M) :=
  ⟨fun _ => injective_object_of_injective_module R M,
   fun _ => injective_module_of_injective_object R M⟩


instance ModuleCat.ulift_injective_of_injective.{v'}
    [Small.{v} R] [AddCommGroup M] [Module R M]
    [CategoryTheory.Injective <| ModuleCat.of R M] :
    CategoryTheory.Injective <| ModuleCat.of R (ULift.{v'} M) :=
  Module.injective_object_of_injective_module
    (inj := Module.ulift_injective_of_injective
      (inj := Module.injective_module_of_injective_object _ _))

