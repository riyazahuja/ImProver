/-- The kernel of the cokernel of `f` is called the (abelian) image of `f`. -/
protected abbrev image : C :=
  kernel (cokernel.π f)


/-- The inclusion of the image into the codomain. -/
protected abbrev image.ι : Abelian.image f ⟶ Q :=
  kernel.ι (cokernel.π f)


/-- There is a canonical epimorphism `p : P ⟶ image f` for every `f`. -/
protected abbrev factorThruImage : P ⟶ Abelian.image f :=
  kernel.lift (cokernel.π f) f <| cokernel.condition f


/-- `f` factors through its image via the canonical morphism `p`. -/
protected theorem image.fac : Abelian.factorThruImage f ≫ image.ι f = f :=
  kernel.lift_ι _ _ _


instance mono_factorThruImage [Mono f] : Mono (Abelian.factorThruImage f) :=
  mono_of_mono_fac <| image.fac f


/-- The cokernel of the kernel of `f` is called the (abelian) coimage of `f`. -/
protected abbrev coimage : C :=
  cokernel (kernel.ι f)


/-- The projection onto the coimage. -/
protected abbrev coimage.π : P ⟶ Abelian.coimage f :=
  cokernel.π (kernel.ι f)


/-- There is a canonical monomorphism `i : coimage f ⟶ Q`. -/
protected abbrev factorThruCoimage : Abelian.coimage f ⟶ Q :=
  cokernel.desc (kernel.ι f) f <| kernel.condition f


/-- `f` factors through its coimage via the canonical morphism `p`. -/
protected theorem coimage.fac : coimage.π f ≫ Abelian.factorThruCoimage f = f :=
  cokernel.π_desc _ _ _


instance epi_factorThruCoimage [Epi f] : Epi (Abelian.factorThruCoimage f) :=
  epi_of_epi_fac <| coimage.fac f


/-- The canonical map from the abelian coimage to the abelian image.
In any abelian category this is an isomorphism.

Conversely, any additive category with kernels and cokernels and
in which this is always an isomorphism, is abelian.

See <https://stacks.math.columbia.edu/tag/0107>
-/
def coimageImageComparison : Abelian.coimage f ⟶ Abelian.image f :=
                                                               /-
                                                                 C : Type u
                                                                 inst✝³ : CategoryTheory.Category.{v, u} C
                                                                 inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
                                                                 inst✝¹ : CategoryTheory.Limits.HasKernels C
                                                                 inst✝ : CategoryTheory.Limits.HasCokernels C
                                                                 P Q : C
                                                                 f : Quiver.Hom P Q
                                                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.cokernel.π f …
                                                               -/
                                                               /-
                                                                 🎉 no goals
                                                               -/
  cokernel.desc (kernel.ι f) (kernel.lift (cokernel.π f) f (by simp)) (by ext; simp)
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


/-- An alternative formulation of the canonical map from the abelian coimage to the abelian image.
-/
def coimageImageComparison' : Abelian.coimage f ⟶ Abelian.image f :=
                                                               /-
                                                                 C : Type u
                                                                 inst✝³ : CategoryTheory.Category.{v, u} C
                                                                 inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
                                                                 inst✝¹ : CategoryTheory.Limits.HasKernels C
                                                                 inst✝ : CategoryTheory.Limits.HasCokernels C
                                                                 P Q : C
                                                                 f : Quiver.Hom P Q
                                                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernel.ι f) f) 0
                                                               -/
                                                               /-
                                                                 🎉 no goals
                                                               -/
  kernel.lift (cokernel.π f) (cokernel.desc (kernel.ι f) f (by simp)) (by ext; simp)
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


theorem coimageImageComparison_eq_coimageImageComparison' :
    coimageImageComparison f = coimageImageComparison' f := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.Limits.HasKernels C
    inst✝ : CategoryTheory.Limits.HasCokernels C
    P Q : C
    f : Quiver.Hom P Q
    ⊢ Eq (CategoryTheory.Abelian.coimageImageComparison f) (CategoryTheory.Abelian …
  -/
  ext
  /-
    case h.h
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.Limits.HasKernels C
    inst✝ : CategoryTheory.Limits.HasCokernels C
    P Q : C
    f : Quiver.Hom P Q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp [coimageImageComparison, coimageImageComparison']
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem coimage_image_factorisation : coimage.π f ≫ coimageImageComparison f ≫ image.ι f = f := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝¹ : CategoryTheory.Limits.HasKernels C
    inst✝ : CategoryTheory.Limits.HasCokernels C
    P Q : C
    f : Quiver.Hom P Q
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Abelian.coimage.π f)  …
  -/
  simp [coimageImageComparison]
  /-
    🎉 no goals
  -/


