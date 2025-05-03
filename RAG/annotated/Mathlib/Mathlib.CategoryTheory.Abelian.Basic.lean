/-- A (preadditive) category `C` is called abelian if it has all finite products,
all kernels and cokernels, and if every monomorphism is the kernel of some morphism
and every epimorphism is the cokernel of some morphism.

(This definition implies the existence of zero objects:
finite products give a terminal object, and in a preadditive category
any terminal object is a zero object.)
-/
class Abelian extends Preadditive C, NormalMonoCategory C, NormalEpiCategory C where
  [has_finite_products : HasFiniteProducts C]
  [has_kernels : HasKernels C]
  [has_cokernels : HasCokernels C]


/-- The factorisation of a morphism through its abelian image. -/
@[simps]
def imageMonoFactorisation {X Y : C} (f : X ⟶ Y) : MonoFactorisation f where
  I := Abelian.image f
  m := kernel.ι _
  m_mono := inferInstance
  e := kernel.lift _ f (cokernel.condition _)
  fac := kernel.lift_ι _ _ _


theorem imageMonoFactorisation_e' {X Y : C} (f : X ⟶ Y) :
    (imageMonoFactorisation f).e = cokernel.π _ ≫ Abelian.coimageImageComparison f := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.Limits.HasKernels C
    inst✝ : CategoryTheory.Limits.HasCokernels C
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.Abelian.OfCoimageImageComparisonIsIso.imageMonoFactorisat …
  -/
  dsimp
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.Limits.HasKernels C
    inst✝ : CategoryTheory.Limits.HasCokernels C
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.Limits.kernel.lift (CategoryTheory.Limits.cokernel.π f) f …
  -/
  ext
  simp only [Abelian.coimageImageComparison, imageMonoFactorisation_e, Category.assoc,
    cokernel.π_desc_assoc]


/-- If the coimage-image comparison morphism for a morphism `f` is an isomorphism,
we obtain an image factorisation of `f`. -/
def imageFactorisation {X Y : C} (f : X ⟶ Y) [IsIso (Abelian.coimageImageComparison f)] :
    ImageFactorisation f where
  F := imageMonoFactorisation f
  isImage :=
    { lift := fun F => inv (Abelian.coimageImageComparison f) ≫ cokernel.desc _ F.e F.kernel_ι_comp
      lift_fac := fun F => by
        /-
          C : Type u
          inst✝⁴ : CategoryTheory.Category.{v, u} C
          inst✝³ : CategoryTheory.Preadditive C
          inst✝² : CategoryTheory.Limits.HasKernels C
          inst✝¹ : CategoryTheory.Limits.HasCokernels C
          X Y : C
          f : Quiver.Hom X Y
          inst✝ : CategoryTheory.IsIso (CategoryTheory.Abelian.coimageImageComparison f)
          F : CategoryTheory.Limits.MonoFactorisation f
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun F => CategoryTheory.CategoryStr …
        -/
        rw [imageMonoFactorisation_m]
        /-
          C : Type u
          inst✝⁴ : CategoryTheory.Category.{v, u} C
          inst✝³ : CategoryTheory.Preadditive C
          inst✝² : CategoryTheory.Limits.HasKernels C
          inst✝¹ : CategoryTheory.Limits.HasCokernels C
          X Y : C
          f : Quiver.Hom X Y
          inst✝ : CategoryTheory.IsIso (CategoryTheory.Abelian.coimageImageComparison f)
          F : CategoryTheory.Limits.MonoFactorisation f
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun F => CategoryTheory.CategoryStr …
        -/
        simp only [Category.assoc]
        /-
          C : Type u
          inst✝⁴ : CategoryTheory.Category.{v, u} C
          inst✝³ : CategoryTheory.Preadditive C
          inst✝² : CategoryTheory.Limits.HasKernels C
          inst✝¹ : CategoryTheory.Limits.HasCokernels C
          X Y : C
          f : Quiver.Hom X Y
          inst✝ : CategoryTheory.IsIso (CategoryTheory.Abelian.coimageImageComparison f)
          F : CategoryTheory.Limits.MonoFactorisation f
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.inv (CategoryTheory.A …
        -/
        rw [IsIso.inv_comp_eq]
        /-
          C : Type u
          inst✝⁴ : CategoryTheory.Category.{v, u} C
          inst✝³ : CategoryTheory.Preadditive C
          inst✝² : CategoryTheory.Limits.HasKernels C
          inst✝¹ : CategoryTheory.Limits.HasCokernels C
          X Y : C
          f : Quiver.Hom X Y
          inst✝ : CategoryTheory.IsIso (CategoryTheory.Abelian.coimageImageComparison f)
          F : CategoryTheory.Limits.MonoFactorisation f
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.cokernel.desc  …
        -/
        ext
        /-
          case h
          C : Type u
          inst✝⁴ : CategoryTheory.Category.{v, u} C
          inst✝³ : CategoryTheory.Preadditive C
          inst✝² : CategoryTheory.Limits.HasKernels C
          inst✝¹ : CategoryTheory.Limits.HasCokernels C
          X Y : C
          f : Quiver.Hom X Y
          inst✝ : CategoryTheory.IsIso (CategoryTheory.Abelian.coimageImageComparison f)
          F : CategoryTheory.Limits.MonoFactorisation f
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequalizer.π  …
        -/
        simp }
        /-
          🎉 no goals
        -/


instance [HasZeroObject C] {X Y : C} (f : X ⟶ Y) [Mono f]
    [IsIso (Abelian.coimageImageComparison f)] : IsIso (imageMonoFactorisation f).e := by
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : CategoryTheory.Limits.HasKernels C
    inst✝³ : CategoryTheory.Limits.HasCokernels C
    inst✝² : CategoryTheory.Limits.HasZeroObject C
    X Y : C
    f : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Mono f
    inst✝ : CategoryTheory.IsIso (CategoryTheory.Abelian.coimageImageComparison f)
    ⊢ CategoryTheory.IsIso (CategoryTheory.Abelian.OfCoimageImageComparisonIsIso.i …
  -/
  rw [imageMonoFactorisation_e']
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : CategoryTheory.Limits.HasKernels C
    inst✝³ : CategoryTheory.Limits.HasCokernels C
    inst✝² : CategoryTheory.Limits.HasZeroObject C
    X Y : C
    f : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Mono f
    inst✝ : CategoryTheory.IsIso (CategoryTheory.Abelian.coimageImageComparison f)
    ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (CategoryTheory.Lim …
  -/
  exact IsIso.comp_isIso
  /-
    🎉 no goals
  -/


instance [HasZeroObject C] {X Y : C} (f : X ⟶ Y) [Epi f] : IsIso (imageMonoFactorisation f).m := by
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.Limits.HasKernels C
    inst✝² : CategoryTheory.Limits.HasCokernels C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Epi f
    ⊢ CategoryTheory.IsIso (CategoryTheory.Abelian.OfCoimageImageComparisonIsIso.i …
  -/
  dsimp
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.Limits.HasKernels C
    inst✝² : CategoryTheory.Limits.HasCokernels C
    inst✝¹ : CategoryTheory.Limits.HasZeroObject C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Epi f
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.kernel.ι (CategoryTheory.Limits. …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- A category in which coimage-image comparisons are all isomorphisms has images. -/
theorem hasImages : HasImages C :=
  { has_image := fun {_} {_} f => { exists_image := ⟨imageFactorisation f⟩ } }


/-- A category with finite products in which coimage-image comparisons are all isomorphisms
is a normal mono category.
-/
def normalMonoCategory : NormalMonoCategory C where
  normalMonoOfMono f m :=
    { Z := _
      g := cokernel.π f
              /-
                C : Type u
                inst✝⁵ : CategoryTheory.Category.{v, u} C
                inst✝⁴ : CategoryTheory.Preadditive C
                inst✝³ : CategoryTheory.Limits.HasKernels C
                inst✝² : CategoryTheory.Limits.HasCokernels C
                inst✝¹ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.IsIso (CategoryTheor …
                inst✝ : CategoryTheory.Limits.HasFiniteProducts C
                X✝ Y✝ : C
                f : Quiver.Hom X✝ Y✝
                m : CategoryTheory.Mono f
                ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.cokernel.π f …
              -/
      w := by simp
              /-
                🎉 no goals
              -/
      isLimit := by
        /-
          C : Type u
          inst✝⁵ : CategoryTheory.Category.{v, u} C
          inst✝⁴ : CategoryTheory.Preadditive C
          inst✝³ : CategoryTheory.Limits.HasKernels C
          inst✝² : CategoryTheory.Limits.HasCokernels C
          inst✝¹ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.IsIso (CategoryTheor …
          inst✝ : CategoryTheory.Limits.HasFiniteProducts C
          X✝ Y✝ : C
          f : Quiver.Hom X✝ Y✝
          m : CategoryTheory.Mono f
          ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι f ⋯)
        -/
        haveI : Limits.HasImages C := hasImages
        /-
          C : Type u
          inst✝⁵ : CategoryTheory.Category.{v, u} C
          inst✝⁴ : CategoryTheory.Preadditive C
          inst✝³ : CategoryTheory.Limits.HasKernels C
          inst✝² : CategoryTheory.Limits.HasCokernels C
          inst✝¹ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.IsIso (CategoryTheor …
          inst✝ : CategoryTheory.Limits.HasFiniteProducts C
          X✝ Y✝ : C
          f : Quiver.Hom X✝ Y✝
          m : CategoryTheory.Mono f
          this : CategoryTheory.Limits.HasImages C
          ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι f ⋯)
        -/
        haveI : HasEqualizers C := Preadditive.hasEqualizers_of_hasKernels
        /-
          C : Type u
          inst✝⁵ : CategoryTheory.Category.{v, u} C
          inst✝⁴ : CategoryTheory.Preadditive C
          inst✝³ : CategoryTheory.Limits.HasKernels C
          inst✝² : CategoryTheory.Limits.HasCokernels C
          inst✝¹ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.IsIso (CategoryTheor …
          inst✝ : CategoryTheory.Limits.HasFiniteProducts C
          X✝ Y✝ : C
          f : Quiver.Hom X✝ Y✝
          m : CategoryTheory.Mono f
          this✝ : CategoryTheory.Limits.HasImages C
          this : CategoryTheory.Limits.HasEqualizers C
          ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι f ⋯)
        -/
        haveI : HasZeroObject C := Limits.hasZeroObject_of_hasFiniteBiproducts _
        have aux : ∀ (s : KernelFork (cokernel.π f)), (limit.lift (parallelPair (cokernel.π f) 0) s
          ≫ inv (imageMonoFactorisation f).e) ≫ Fork.ι (KernelFork.ofι f (by simp))
            = Fork.ι s := ?_
          /-
            case refine_2
            C : Type u
            inst✝⁵ : CategoryTheory.Category.{v, u} C
            inst✝⁴ : CategoryTheory.Preadditive C
            inst✝³ : CategoryTheory.Limits.HasKernels C
            inst✝² : CategoryTheory.Limits.HasCokernels C
            inst✝¹ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.IsIso (CategoryTheor …
            inst✝ : CategoryTheory.Limits.HasFiniteProducts C
            X✝ Y✝ : C
            f : Quiver.Hom X✝ Y✝
            m : CategoryTheory.Mono f
            this✝¹ : CategoryTheory.Limits.HasImages C
            this✝ : CategoryTheory.Limits.HasEqualizers C
            this : CategoryTheory.Limits.HasZeroObject C
            aux : ∀ (s : CategoryTheory.Limits.KernelFork (CategoryTheory.Limits.cokernel. …
            ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι f ⋯)
          -/
        · refine isLimitAux _ (fun A => limit.lift _ _ ≫ inv (imageMonoFactorisation f).e) aux ?_
          /-
            case refine_2
            C : Type u
            inst✝⁵ : CategoryTheory.Category.{v, u} C
            inst✝⁴ : CategoryTheory.Preadditive C
            inst✝³ : CategoryTheory.Limits.HasKernels C
            inst✝² : CategoryTheory.Limits.HasCokernels C
            inst✝¹ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.IsIso (CategoryTheor …
            inst✝ : CategoryTheory.Limits.HasFiniteProducts C
            X✝ Y✝ : C
            f : Quiver.Hom X✝ Y✝
            m : CategoryTheory.Mono f
            this✝¹ : CategoryTheory.Limits.HasImages C
            this✝ : CategoryTheory.Limits.HasEqualizers C
            this : CategoryTheory.Limits.HasZeroObject C
            aux : ∀ (s : CategoryTheory.Limits.KernelFork (CategoryTheory.Limits.cokernel. …
            ⊢ ∀ (s : CategoryTheory.Limits.KernelFork (CategoryTheory.Limits.cokernel.π f) …
          -/
          intro A g hg
          /-
            case refine_2
            C : Type u
            inst✝⁵ : CategoryTheory.Category.{v, u} C
            inst✝⁴ : CategoryTheory.Preadditive C
            inst✝³ : CategoryTheory.Limits.HasKernels C
            inst✝² : CategoryTheory.Limits.HasCokernels C
            inst✝¹ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.IsIso (CategoryTheor …
            inst✝ : CategoryTheory.Limits.HasFiniteProducts C
            X✝ Y✝ : C
            f : Quiver.Hom X✝ Y✝
            m : CategoryTheory.Mono f
            this✝¹ : CategoryTheory.Limits.HasImages C
            this✝ : CategoryTheory.Limits.HasEqualizers C
            this : CategoryTheory.Limits.HasZeroObject C
            aux : ∀ (s : CategoryTheory.Limits.KernelFork (CategoryTheory.Limits.cokernel. …
            A : CategoryTheory.Limits.KernelFork (CategoryTheory.Limits.cokernel.π f)
            g : Quiver.Hom A.pt (CategoryTheory.Limits.KernelFork.ofι f ⋯).pt
            hg : Eq (CategoryTheory.CategoryStruct.comp g (CategoryTheory.Limits.Fork.ι (C …
            ⊢ Eq g ((fun A => CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.li …
          -/
          rw [KernelFork.ι_ofι] at hg
          /-
            case refine_2
            C : Type u
            inst✝⁵ : CategoryTheory.Category.{v, u} C
            inst✝⁴ : CategoryTheory.Preadditive C
            inst✝³ : CategoryTheory.Limits.HasKernels C
            inst✝² : CategoryTheory.Limits.HasCokernels C
            inst✝¹ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.IsIso (CategoryTheor …
            inst✝ : CategoryTheory.Limits.HasFiniteProducts C
            X✝ Y✝ : C
            f : Quiver.Hom X✝ Y✝
            m : CategoryTheory.Mono f
            this✝¹ : CategoryTheory.Limits.HasImages C
            this✝ : CategoryTheory.Limits.HasEqualizers C
            this : CategoryTheory.Limits.HasZeroObject C
            aux : ∀ (s : CategoryTheory.Limits.KernelFork (CategoryTheory.Limits.cokernel. …
            A : CategoryTheory.Limits.KernelFork (CategoryTheory.Limits.cokernel.π f)
            g : Quiver.Hom A.pt (CategoryTheory.Limits.KernelFork.ofι f ⋯).pt
            hg : Eq (CategoryTheory.CategoryStruct.comp g f) (CategoryTheory.Limits.Fork.ι …
            ⊢ Eq g ((fun A => CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.li …
          -/
          rw [← cancel_mono f, hg, ← aux, KernelFork.ι_ofι]
          /-
            🎉 no goals
          -/
          /-
            case refine_1
            C : Type u
            inst✝⁵ : CategoryTheory.Category.{v, u} C
            inst✝⁴ : CategoryTheory.Preadditive C
            inst✝³ : CategoryTheory.Limits.HasKernels C
            inst✝² : CategoryTheory.Limits.HasCokernels C
            inst✝¹ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.IsIso (CategoryTheor …
            inst✝ : CategoryTheory.Limits.HasFiniteProducts C
            X✝ Y✝ : C
            f : Quiver.Hom X✝ Y✝
            m : CategoryTheory.Mono f
            this✝¹ : CategoryTheory.Limits.HasImages C
            this✝ : CategoryTheory.Limits.HasEqualizers C
            this : CategoryTheory.Limits.HasZeroObject C
            ⊢ ∀ (s : CategoryTheory.Limits.KernelFork (CategoryTheory.Limits.cokernel.π f) …
          -/
        · intro A
          /-
            case refine_1
            C : Type u
            inst✝⁵ : CategoryTheory.Category.{v, u} C
            inst✝⁴ : CategoryTheory.Preadditive C
            inst✝³ : CategoryTheory.Limits.HasKernels C
            inst✝² : CategoryTheory.Limits.HasCokernels C
            inst✝¹ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.IsIso (CategoryTheor …
            inst✝ : CategoryTheory.Limits.HasFiniteProducts C
            X✝ Y✝ : C
            f : Quiver.Hom X✝ Y✝
            m : CategoryTheory.Mono f
            this✝¹ : CategoryTheory.Limits.HasImages C
            this✝ : CategoryTheory.Limits.HasEqualizers C
            this : CategoryTheory.Limits.HasZeroObject C
            A : CategoryTheory.Limits.KernelFork (CategoryTheory.Limits.cokernel.π f)
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
          -/
          simp only [KernelFork.ι_ofι, Category.assoc]
          /-
            case refine_1
            C : Type u
            inst✝⁵ : CategoryTheory.Category.{v, u} C
            inst✝⁴ : CategoryTheory.Preadditive C
            inst✝³ : CategoryTheory.Limits.HasKernels C
            inst✝² : CategoryTheory.Limits.HasCokernels C
            inst✝¹ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.IsIso (CategoryTheor …
            inst✝ : CategoryTheory.Limits.HasFiniteProducts C
            X✝ Y✝ : C
            f : Quiver.Hom X✝ Y✝
            m : CategoryTheory.Mono f
            this✝¹ : CategoryTheory.Limits.HasImages C
            this✝ : CategoryTheory.Limits.HasEqualizers C
            this : CategoryTheory.Limits.HasZeroObject C
            A : CategoryTheory.Limits.KernelFork (CategoryTheory.Limits.cokernel.π f)
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.limit.lift (Ca …
          -/
          convert limit.lift_π A WalkingParallelPair.zero using 2
          /-
            case h.e'_2.h.e'_7
            C : Type u
            inst✝⁵ : CategoryTheory.Category.{v, u} C
            inst✝⁴ : CategoryTheory.Preadditive C
            inst✝³ : CategoryTheory.Limits.HasKernels C
            inst✝² : CategoryTheory.Limits.HasCokernels C
            inst✝¹ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.IsIso (CategoryTheor …
            inst✝ : CategoryTheory.Limits.HasFiniteProducts C
            X✝ Y✝ : C
            f : Quiver.Hom X✝ Y✝
            m : CategoryTheory.Mono f
            this✝¹ : CategoryTheory.Limits.HasImages C
            this✝ : CategoryTheory.Limits.HasEqualizers C
            this : CategoryTheory.Limits.HasZeroObject C
            A : CategoryTheory.Limits.KernelFork (CategoryTheory.Limits.cokernel.π f)
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.inv (CategoryTheory.A …
          -/
          rw [IsIso.inv_comp_eq, eq_comm]
          /-
            case h.e'_2.h.e'_7
            C : Type u
            inst✝⁵ : CategoryTheory.Category.{v, u} C
            inst✝⁴ : CategoryTheory.Preadditive C
            inst✝³ : CategoryTheory.Limits.HasKernels C
            inst✝² : CategoryTheory.Limits.HasCokernels C
            inst✝¹ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.IsIso (CategoryTheor …
            inst✝ : CategoryTheory.Limits.HasFiniteProducts C
            X✝ Y✝ : C
            f : Quiver.Hom X✝ Y✝
            m : CategoryTheory.Mono f
            this✝¹ : CategoryTheory.Limits.HasImages C
            this✝ : CategoryTheory.Limits.HasEqualizers C
            this : CategoryTheory.Limits.HasZeroObject C
            A : CategoryTheory.Limits.KernelFork (CategoryTheory.Limits.cokernel.π f)
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Abelian.OfCoimageImag …
          -/
          exact (imageMonoFactorisation f).fac }
          /-
            🎉 no goals
          -/


/-- A category with finite products in which coimage-image comparisons are all isomorphisms
is a normal epi category.
-/
def normalEpiCategory : NormalEpiCategory C where
  normalEpiOfEpi f m :=
    { W := kernel f
      g := kernel.ι _
      w := kernel.condition _
      isColimit := by
        /-
          C : Type u
          inst✝⁵ : CategoryTheory.Category.{v, u} C
          inst✝⁴ : CategoryTheory.Preadditive C
          inst✝³ : CategoryTheory.Limits.HasKernels C
          inst✝² : CategoryTheory.Limits.HasCokernels C
          inst✝¹ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.IsIso (CategoryTheor …
          inst✝ : CategoryTheory.Limits.HasFiniteProducts C
          X✝ Y✝ : C
          f : Quiver.Hom X✝ Y✝
          m : CategoryTheory.Epi f
          ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ f ⋯)
        -/
        haveI : Limits.HasImages C := hasImages
        /-
          C : Type u
          inst✝⁵ : CategoryTheory.Category.{v, u} C
          inst✝⁴ : CategoryTheory.Preadditive C
          inst✝³ : CategoryTheory.Limits.HasKernels C
          inst✝² : CategoryTheory.Limits.HasCokernels C
          inst✝¹ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.IsIso (CategoryTheor …
          inst✝ : CategoryTheory.Limits.HasFiniteProducts C
          X✝ Y✝ : C
          f : Quiver.Hom X✝ Y✝
          m : CategoryTheory.Epi f
          this : CategoryTheory.Limits.HasImages C
          ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ f ⋯)
        -/
        haveI : HasEqualizers C := Preadditive.hasEqualizers_of_hasKernels
        /-
          C : Type u
          inst✝⁵ : CategoryTheory.Category.{v, u} C
          inst✝⁴ : CategoryTheory.Preadditive C
          inst✝³ : CategoryTheory.Limits.HasKernels C
          inst✝² : CategoryTheory.Limits.HasCokernels C
          inst✝¹ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.IsIso (CategoryTheor …
          inst✝ : CategoryTheory.Limits.HasFiniteProducts C
          X✝ Y✝ : C
          f : Quiver.Hom X✝ Y✝
          m : CategoryTheory.Epi f
          this✝ : CategoryTheory.Limits.HasImages C
          this : CategoryTheory.Limits.HasEqualizers C
          ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ f ⋯)
        -/
        haveI : HasZeroObject C := Limits.hasZeroObject_of_hasFiniteBiproducts _
        have aux : ∀ (s : CokernelCofork (kernel.ι f)), Cofork.π (CokernelCofork.ofπ f (by simp)) ≫
          inv (imageMonoFactorisation f).m ≫ inv (Abelian.coimageImageComparison f) ≫
          colimit.desc (parallelPair (kernel.ι f) 0) s = Cofork.π s := ?_
        · refine isColimitAux _ (fun A => inv (imageMonoFactorisation f).m ≫
                  inv (Abelian.coimageImageComparison f) ≫ colimit.desc _ _) aux ?_
          /-
            case refine_2
            C : Type u
            inst✝⁵ : CategoryTheory.Category.{v, u} C
            inst✝⁴ : CategoryTheory.Preadditive C
            inst✝³ : CategoryTheory.Limits.HasKernels C
            inst✝² : CategoryTheory.Limits.HasCokernels C
            inst✝¹ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.IsIso (CategoryTheor …
            inst✝ : CategoryTheory.Limits.HasFiniteProducts C
            X✝ Y✝ : C
            f : Quiver.Hom X✝ Y✝
            m : CategoryTheory.Epi f
            this✝¹ : CategoryTheory.Limits.HasImages C
            this✝ : CategoryTheory.Limits.HasEqualizers C
            this : CategoryTheory.Limits.HasZeroObject C
            aux : ∀ (s : CategoryTheory.Limits.CokernelCofork (CategoryTheory.Limits.kerne …
            ⊢ ∀ (s : CategoryTheory.Limits.CokernelCofork (CategoryTheory.Limits.kernel.ι  …
          -/
          intro A g hg
          /-
            case refine_2
            C : Type u
            inst✝⁵ : CategoryTheory.Category.{v, u} C
            inst✝⁴ : CategoryTheory.Preadditive C
            inst✝³ : CategoryTheory.Limits.HasKernels C
            inst✝² : CategoryTheory.Limits.HasCokernels C
            inst✝¹ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.IsIso (CategoryTheor …
            inst✝ : CategoryTheory.Limits.HasFiniteProducts C
            X✝ Y✝ : C
            f : Quiver.Hom X✝ Y✝
            m : CategoryTheory.Epi f
            this✝¹ : CategoryTheory.Limits.HasImages C
            this✝ : CategoryTheory.Limits.HasEqualizers C
            this : CategoryTheory.Limits.HasZeroObject C
            aux : ∀ (s : CategoryTheory.Limits.CokernelCofork (CategoryTheory.Limits.kerne …
            A : CategoryTheory.Limits.CokernelCofork (CategoryTheory.Limits.kernel.ι f)
            g : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ f ⋯).pt A.pt
            hg : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π (C …
            ⊢ Eq g ((fun A => CategoryTheory.CategoryStruct.comp (CategoryTheory.inv (Cate …
          -/
          rw [CokernelCofork.π_ofπ] at hg
          /-
            case refine_2
            C : Type u
            inst✝⁵ : CategoryTheory.Category.{v, u} C
            inst✝⁴ : CategoryTheory.Preadditive C
            inst✝³ : CategoryTheory.Limits.HasKernels C
            inst✝² : CategoryTheory.Limits.HasCokernels C
            inst✝¹ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.IsIso (CategoryTheor …
            inst✝ : CategoryTheory.Limits.HasFiniteProducts C
            X✝ Y✝ : C
            f : Quiver.Hom X✝ Y✝
            m : CategoryTheory.Epi f
            this✝¹ : CategoryTheory.Limits.HasImages C
            this✝ : CategoryTheory.Limits.HasEqualizers C
            this : CategoryTheory.Limits.HasZeroObject C
            aux : ∀ (s : CategoryTheory.Limits.CokernelCofork (CategoryTheory.Limits.kerne …
            A : CategoryTheory.Limits.CokernelCofork (CategoryTheory.Limits.kernel.ι f)
            g : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ f ⋯).pt A.pt
            hg : Eq (CategoryTheory.CategoryStruct.comp f g) (CategoryTheory.Limits.Cofork …
            ⊢ Eq g ((fun A => CategoryTheory.CategoryStruct.comp (CategoryTheory.inv (Cate …
          -/
          rw [← cancel_epi f, hg, ← aux, CokernelCofork.π_ofπ]
          /-
            🎉 no goals
          -/
          /-
            case refine_1
            C : Type u
            inst✝⁵ : CategoryTheory.Category.{v, u} C
            inst✝⁴ : CategoryTheory.Preadditive C
            inst✝³ : CategoryTheory.Limits.HasKernels C
            inst✝² : CategoryTheory.Limits.HasCokernels C
            inst✝¹ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.IsIso (CategoryTheor …
            inst✝ : CategoryTheory.Limits.HasFiniteProducts C
            X✝ Y✝ : C
            f : Quiver.Hom X✝ Y✝
            m : CategoryTheory.Epi f
            this✝¹ : CategoryTheory.Limits.HasImages C
            this✝ : CategoryTheory.Limits.HasEqualizers C
            this : CategoryTheory.Limits.HasZeroObject C
            ⊢ ∀ (s : CategoryTheory.Limits.CokernelCofork (CategoryTheory.Limits.kernel.ι  …
          -/
        · intro A
          /-
            case refine_1
            C : Type u
            inst✝⁵ : CategoryTheory.Category.{v, u} C
            inst✝⁴ : CategoryTheory.Preadditive C
            inst✝³ : CategoryTheory.Limits.HasKernels C
            inst✝² : CategoryTheory.Limits.HasCokernels C
            inst✝¹ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.IsIso (CategoryTheor …
            inst✝ : CategoryTheory.Limits.HasFiniteProducts C
            X✝ Y✝ : C
            f : Quiver.Hom X✝ Y✝
            m : CategoryTheory.Epi f
            this✝¹ : CategoryTheory.Limits.HasImages C
            this✝ : CategoryTheory.Limits.HasEqualizers C
            this : CategoryTheory.Limits.HasZeroObject C
            A : CategoryTheory.Limits.CokernelCofork (CategoryTheory.Limits.kernel.ι f)
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π (Cate …
          -/
          simp only [CokernelCofork.π_ofπ, ← Category.assoc]
          /-
            case refine_1
            C : Type u
            inst✝⁵ : CategoryTheory.Category.{v, u} C
            inst✝⁴ : CategoryTheory.Preadditive C
            inst✝³ : CategoryTheory.Limits.HasKernels C
            inst✝² : CategoryTheory.Limits.HasCokernels C
            inst✝¹ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.IsIso (CategoryTheor …
            inst✝ : CategoryTheory.Limits.HasFiniteProducts C
            X✝ Y✝ : C
            f : Quiver.Hom X✝ Y✝
            m : CategoryTheory.Epi f
            this✝¹ : CategoryTheory.Limits.HasImages C
            this✝ : CategoryTheory.Limits.HasEqualizers C
            this : CategoryTheory.Limits.HasZeroObject C
            A : CategoryTheory.Limits.CokernelCofork (CategoryTheory.Limits.kernel.ι f)
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
          -/
          convert colimit.ι_desc A WalkingParallelPair.one using 2
          /-
            case h.e'_2.h.e'_6
            C : Type u
            inst✝⁵ : CategoryTheory.Category.{v, u} C
            inst✝⁴ : CategoryTheory.Preadditive C
            inst✝³ : CategoryTheory.Limits.HasKernels C
            inst✝² : CategoryTheory.Limits.HasCokernels C
            inst✝¹ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.IsIso (CategoryTheor …
            inst✝ : CategoryTheory.Limits.HasFiniteProducts C
            X✝ Y✝ : C
            f : Quiver.Hom X✝ Y✝
            m : CategoryTheory.Epi f
            this✝¹ : CategoryTheory.Limits.HasImages C
            this✝ : CategoryTheory.Limits.HasEqualizers C
            this : CategoryTheory.Limits.HasZeroObject C
            A : CategoryTheory.Limits.CokernelCofork (CategoryTheory.Limits.kernel.ι f)
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
          -/
          rw [IsIso.comp_inv_eq, IsIso.comp_inv_eq, eq_comm, ← imageMonoFactorisation_e']
          /-
            case h.e'_2.h.e'_6
            C : Type u
            inst✝⁵ : CategoryTheory.Category.{v, u} C
            inst✝⁴ : CategoryTheory.Preadditive C
            inst✝³ : CategoryTheory.Limits.HasKernels C
            inst✝² : CategoryTheory.Limits.HasCokernels C
            inst✝¹ : ∀ {X Y : C} (f : Quiver.Hom X Y), CategoryTheory.IsIso (CategoryTheor …
            inst✝ : CategoryTheory.Limits.HasFiniteProducts C
            X✝ Y✝ : C
            f : Quiver.Hom X✝ Y✝
            m : CategoryTheory.Epi f
            this✝¹ : CategoryTheory.Limits.HasImages C
            this✝ : CategoryTheory.Limits.HasEqualizers C
            this : CategoryTheory.Limits.HasZeroObject C
            A : CategoryTheory.Limits.CokernelCofork (CategoryTheory.Limits.kernel.ι f)
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Abelian.OfCoimageImag …
          -/
          exact (imageMonoFactorisation f).fac }
          /-
            🎉 no goals
          -/


/-- A preadditive category with kernels, cokernels, and finite products,
in which the coimage-image comparison morphism is always an isomorphism,
is an abelian category.

The Stacks project uses this characterisation at the definition of an abelian category.
See <https://stacks.math.columbia.edu/tag/0109>.
-/
def ofCoimageImageComparisonIsIso : Abelian C where


/-- An abelian category has finite biproducts. -/
theorem hasFiniteBiproducts : HasFiniteBiproducts C :=
  Limits.HasFiniteBiproducts.of_hasFiniteProducts


instance (priority := 100) hasBinaryBiproducts : HasBinaryBiproducts C :=
  Limits.hasBinaryBiproducts_of_finite_biproducts _


instance (priority := 100) hasZeroObject : HasZeroObject C :=
  hasZeroObject_of_hasInitial_object


/-- Every abelian category is, in particular, `NonPreadditiveAbelian`. -/
def nonPreadditiveAbelian : NonPreadditiveAbelian C :=
  { ‹Abelian C› with }


/-- The map `p : P ⟶ image f` is an epimorphism -/
                                                 /-
                                                   C : Type u
                                                   inst✝¹ : CategoryTheory.Category.{v, u} C
                                                   inst✝ : CategoryTheory.Abelian C
                                                   P Q : C
                                                   f : Quiver.Hom P Q
                                                   ⊢ CategoryTheory.Epi (CategoryTheory.Abelian.factorThruImage f)
                                                 -/
instance : Epi (Abelian.factorThruImage f) := by infer_instance
                                                 /-
                                                   🎉 no goals
                                                 -/


                                                                                  /-
                                                                                    C : Type u
                                                                                    inst✝² : CategoryTheory.Category.{v, u} C
                                                                                    inst✝¹ : CategoryTheory.Abelian C
                                                                                    P Q : C
                                                                                    f : Quiver.Hom P Q
                                                                                    inst✝ : CategoryTheory.Mono f
                                                                                    ⊢ CategoryTheory.IsIso (CategoryTheory.Abelian.factorThruImage f)
                                                                                  -/
instance isIso_factorThruImage [Mono f] : IsIso (Abelian.factorThruImage f) := by infer_instance
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


/-- The canonical morphism `i : coimage f ⟶ Q` is a monomorphism -/
                                                    /-
                                                      C : Type u
                                                      inst✝¹ : CategoryTheory.Category.{v, u} C
                                                      inst✝ : CategoryTheory.Abelian C
                                                      P Q : C
                                                      f : Quiver.Hom P Q
                                                      ⊢ CategoryTheory.Mono (CategoryTheory.Abelian.factorThruCoimage f)
                                                    -/
instance : Mono (Abelian.factorThruCoimage f) := by infer_instance
                                                    /-
                                                      🎉 no goals
                                                    -/


                                                                                     /-
                                                                                       C : Type u
                                                                                       inst✝² : CategoryTheory.Category.{v, u} C
                                                                                       inst✝¹ : CategoryTheory.Abelian C
                                                                                       P Q : C
                                                                                       f : Quiver.Hom P Q
                                                                                       inst✝ : CategoryTheory.Epi f
                                                                                       ⊢ CategoryTheory.IsIso (CategoryTheory.Abelian.factorThruCoimage f)
                                                                                     -/
instance isIso_factorThruCoimage [Epi f] : IsIso (Abelian.factorThruCoimage f) := by infer_instance
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


theorem mono_of_kernel_ι_eq_zero (h : kernel.ι f = 0) : Mono f :=
  mono_of_kernel_zero h


theorem epi_of_cokernel_π_eq_zero (h : cokernel.π f = 0) : Epi f := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    P Q : C
    f : Quiver.Hom P Q
    h : Eq (CategoryTheory.Limits.cokernel.π f) 0
    ⊢ CategoryTheory.Epi f
  -/
  apply NormalMonoCategory.epi_of_zero_cokernel _ (cokernel f)
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    P Q : C
    f : Quiver.Hom P Q
    h : Eq (CategoryTheory.Limits.cokernel.π f) 0
    ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ 0 ⋯)
  -/
  simp_rw [← h]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    P Q : C
    f : Quiver.Hom P Q
    h : Eq (CategoryTheory.Limits.cokernel.π f) 0
    ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ (C …
  -/
  exact IsColimit.ofIsoColimit (colimit.isColimit (parallelPair f 0)) (isoOfπ _)
  /-
    🎉 no goals
  -/


theorem image_ι_comp_eq_zero {R : C} {g : Q ⟶ R} (h : f ≫ g = 0) : Abelian.image.ι f ≫ g = 0 :=
                                                     /-
                                                       C : Type u
                                                       inst✝¹ : CategoryTheory.Category.{v, u} C
                                                       inst✝ : CategoryTheory.Abelian C
                                                       P Q : C
                                                       f : Quiver.Hom P Q
                                                       R : C
                                                       g : Quiver.Hom Q R
                                                       h : Eq (CategoryTheory.CategoryStruct.comp f g) 0
                                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Abelian.factorThruIma …
                                                     -/
  zero_of_epi_comp (Abelian.factorThruImage f) <| by simp [h]
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem comp_coimage_π_eq_zero {R : C} {g : Q ⟶ R} (h : f ≫ g = 0) : f ≫ Abelian.coimage.π g = 0 :=
                                                        /-
                                                          C : Type u
                                                          inst✝¹ : CategoryTheory.Category.{v, u} C
                                                          inst✝ : CategoryTheory.Abelian C
                                                          P Q : C
                                                          f : Quiver.Hom P Q
                                                          R : C
                                                          g : Quiver.Hom Q R
                                                          h : Eq (CategoryTheory.CategoryStruct.comp f g) 0
                                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
                                                        -/
  zero_of_comp_mono (Abelian.factorThruCoimage g) <| by simp [h]
                                                        /-
                                                          🎉 no goals
                                                        -/


/-- Factoring through the image is a strong epi-mono factorisation. -/
@[simps]
def imageStrongEpiMonoFactorisation : StrongEpiMonoFactorisation f where
  I := Abelian.image f
  m := image.ι f
               /-
                 C : Type u
                 inst✝¹ : CategoryTheory.Category.{v, u} C
                 inst✝ : CategoryTheory.Abelian C
                 P Q : C
                 f : Quiver.Hom P Q
                 ⊢ CategoryTheory.Mono (CategoryTheory.Abelian.image.ι f)
               -/
  m_mono := by infer_instance
               /-
                 🎉 no goals
               -/
  e := Abelian.factorThruImage f
  e_strong_epi := strongEpi_of_epi _


/-- Factoring through the coimage is a strong epi-mono factorisation. -/
@[simps]
def coimageStrongEpiMonoFactorisation : StrongEpiMonoFactorisation f where
  I := Abelian.coimage f
  m := Abelian.factorThruCoimage f
               /-
                 C : Type u
                 inst✝¹ : CategoryTheory.Category.{v, u} C
                 inst✝ : CategoryTheory.Abelian C
                 P Q : C
                 f : Quiver.Hom P Q
                 ⊢ CategoryTheory.Mono (CategoryTheory.Abelian.factorThruCoimage f)
               -/
  m_mono := by infer_instance
               /-
                 🎉 no goals
               -/
  e := coimage.π f
  e_strong_epi := strongEpi_of_epi _


/-- An abelian category has strong epi-mono factorisations. -/
instance (priority := 100) : HasStrongEpiMonoFactorisations C :=
  HasStrongEpiMonoFactorisations.mk fun f => imageStrongEpiMonoFactorisation f

-- In particular, this means that it has well-behaved images.

/-- The coimage-image comparison morphism is always an isomorphism in an abelian category.
See `CategoryTheory.Abelian.ofCoimageImageComparisonIsIso` for the converse.
-/
instance : IsIso (coimageImageComparison f) := by
  convert
    Iso.isIso_hom
      (IsImage.isoExt (coimageStrongEpiMonoFactorisation f).toMonoIsImage
        (imageStrongEpiMonoFactorisation f).toMonoIsImage)
  /-
    case h.e'_5.h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    X Y : C
    f : Quiver.Hom X Y
    e_3✝ : Eq (CategoryTheory.Abelian.coimage f) (CategoryTheory.Abelian.coimageSt …
    e_4✝ : Eq (CategoryTheory.Abelian.image f) (CategoryTheory.Abelian.imageStrong …
    ⊢ Eq (CategoryTheory.Abelian.coimageImageComparison f) ((CategoryTheory.Abelia …
  -/
  ext
  /-
    case h.e'_5.h.h.h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    X Y : C
    f : Quiver.Hom X Y
    e_3✝ : Eq (CategoryTheory.Abelian.coimage f) (CategoryTheory.Abelian.coimageSt …
    e_4✝ : Eq (CategoryTheory.Abelian.image f) (CategoryTheory.Abelian.imageStrong …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  change _ = _ ≫ (imageStrongEpiMonoFactorisation f).m
  /-
    case h.e'_5.h.h.h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    X Y : C
    f : Quiver.Hom X Y
    e_3✝ : Eq (CategoryTheory.Abelian.coimage f) (CategoryTheory.Abelian.coimageSt …
    e_4✝ : Eq (CategoryTheory.Abelian.image f) (CategoryTheory.Abelian.imageStrong …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp [-imageStrongEpiMonoFactorisation_m]
  /-
    🎉 no goals
  -/


/-- There is a canonical isomorphism between the abelian coimage and the abelian image of a
    morphism. -/
abbrev coimageIsoImage : Abelian.coimage f ≅ Abelian.image f :=
  asIso (coimageImageComparison f)


/-- There is a canonical isomorphism between the abelian coimage and the categorical image of a
    morphism. -/
abbrev coimageIsoImage' : Abelian.coimage f ≅ image f :=
  IsImage.isoExt (coimageStrongEpiMonoFactorisation f).toMonoIsImage (Image.isImage f)


theorem coimageIsoImage'_hom :
    (coimageIsoImage' f).hom =
                                              /-
                                                C : Type u
                                                inst✝¹ : CategoryTheory.Category.{v, u} C
                                                inst✝ : CategoryTheory.Abelian C
                                                X Y : C
                                                f : Quiver.Hom X Y
                                                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernel.ι f) (C …
                                              -/
      cokernel.desc _ (factorThruImage f) (by simp [← cancel_mono (Limits.image.ι f)]) := by
                                              /-
                                                🎉 no goals
                                              -/
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.Abelian.coimageIsoImage' f).hom (CategoryTheory.Limits.co …
  -/
  ext
  simp only [← cancel_mono (Limits.image.ι f), IsImage.isoExt_hom, cokernel.π_desc,
    Category.assoc, IsImage.lift_ι, coimageStrongEpiMonoFactorisation_m,
    Limits.image.fac]


theorem factorThruImage_comp_coimageIsoImage'_inv :
    factorThruImage f ≫ (coimageIsoImage' f).inv = cokernel.π _ := by
  simp only [IsImage.isoExt_inv, image.isImage_lift, image.fac_lift,
    coimageStrongEpiMonoFactorisation_e]


/-- There is a canonical isomorphism between the abelian image and the categorical image of a
    morphism. -/
abbrev imageIsoImage : Abelian.image f ≅ image f :=
  IsImage.isoExt (imageStrongEpiMonoFactorisation f).toMonoIsImage (Image.isImage f)


theorem imageIsoImage_hom_comp_image_ι : (imageIsoImage f).hom ≫ Limits.image.ι _ = kernel.ι _ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Abelian.imageIsoImage …
  -/
  simp only [IsImage.isoExt_hom, IsImage.lift_ι, imageStrongEpiMonoFactorisation_m]
  /-
    🎉 no goals
  -/


theorem imageIsoImage_inv :
    (imageIsoImage f).inv =
                                           /-
                                             C : Type u
                                             inst✝¹ : CategoryTheory.Category.{v, u} C
                                             inst✝ : CategoryTheory.Abelian C
                                             X Y : C
                                             f : Quiver.Hom X Y
                                             ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.image.ι f) (Ca …
                                           -/
      kernel.lift _ (Limits.image.ι f) (by simp [← cancel_epi (factorThruImage f)]) := by
                                           /-
                                             🎉 no goals
                                           -/
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Abelian C
    X Y : C
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.Abelian.imageIsoImage f).inv (CategoryTheory.Limits.kerne …
  -/
  ext
  rw [IsImage.isoExt_inv, image.isImage_lift, Limits.image.fac_lift,
    imageStrongEpiMonoFactorisation_e, Category.assoc, kernel.lift_ι, equalizer_as_kernel,
    kernel.lift_ι, Limits.image.fac]


/-- In an abelian category, an epi is the cokernel of its kernel. More precisely:
    If `f` is an epimorphism and `s` is some limit kernel cone on `f`, then `f` is a cokernel
    of `fork.ι s`. -/
def epiIsCokernelOfKernel [Epi f] (s : Fork f 0) (h : IsLimit s) :
    IsColimit (CokernelCofork.ofπ f (KernelFork.condition s)) :=
  NonPreadditiveAbelian.epiIsCokernelOfKernel s h


/-- In an abelian category, a mono is the kernel of its cokernel. More precisely:
    If `f` is a monomorphism and `s` is some colimit cokernel cocone on `f`, then `f` is a kernel
    of `cofork.π s`. -/
def monoIsKernelOfCokernel [Mono f] (s : Cofork f 0) (h : IsColimit s) :
    IsLimit (KernelFork.ofι f (CokernelCofork.condition s)) :=
  NonPreadditiveAbelian.monoIsKernelOfCokernel s h


/-- In an abelian category, any morphism that turns to zero when precomposed with the kernel of an
    epimorphism factors through that epimorphism. -/
def epiDesc [Epi f] {T : C} (g : X ⟶ T) (hg : kernel.ι f ≫ g = 0) : Y ⟶ T :=
  (epiIsCokernelOfKernel _ (limit.isLimit _)).desc (CokernelCofork.ofπ _ hg)


@[reassoc (attr := simp)]
theorem comp_epiDesc [Epi f] {T : C} (g : X ⟶ T) (hg : kernel.ι f ≫ g = 0) :
    f ≫ epiDesc f g hg = g :=
  (epiIsCokernelOfKernel _ (limit.isLimit _)).fac (CokernelCofork.ofπ _ hg) WalkingParallelPair.one


/-- In an abelian category, any morphism that turns to zero when postcomposed with the cokernel of a
    monomorphism factors through that monomorphism. -/
def monoLift [Mono f] {T : C} (g : T ⟶ Y) (hg : g ≫ cokernel.π f = 0) : T ⟶ X :=
  (monoIsKernelOfCokernel _ (colimit.isColimit _)).lift (KernelFork.ofι _ hg)


@[reassoc (attr := simp)]
theorem monoLift_comp [Mono f] {T : C} (g : T ⟶ Y) (hg : g ≫ cokernel.π f = 0) :
    monoLift f g hg ≫ f = g :=
  (monoIsKernelOfCokernel _ (colimit.isColimit _)).fac (KernelFork.ofι _ hg)
    WalkingParallelPair.zero


/-- If `F : D ⥤ C` is a functor to an abelian category, `i : X ⟶ Y` is a morphism
admitting a cokernel such that `F` preserves this cokernel and `F.map i` is a mono,
then `F.map X` identifies to the kernel of `F.map (cokernel.π i)`. -/
noncomputable def isLimitMapConeOfKernelForkOfι
    {X Y : D} (i : X ⟶ Y) [HasCokernel i] (F : D ⥤ C)
    [F.PreservesZeroMorphisms] [Mono (F.map i)]
    [PreservesColimit (parallelPair i 0) F] :
    IsLimit (F.mapCone (KernelFork.ofι i (cokernel.condition i))) := by
  let e : parallelPair (cokernel.π (F.map i)) 0 ≅ parallelPair (cokernel.π i) 0 ⋙ F :=
    parallelPair.ext (Iso.refl _) (asIso (cokernelComparison i F)) (by simp) (by simp)
  /-
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    inst✝⁶ : CategoryTheory.Abelian C
    X✝ Y✝ : C
    f : Quiver.Hom X✝ Y✝
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{?u.114194, u_1} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms D
    X Y : D
    i : Quiver.Hom X Y
    inst✝³ : CategoryTheory.Limits.HasCokernel i
    F : CategoryTheory.Functor D C
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : CategoryTheory.Mono (F.map i)
    inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallel …
    e : CategoryTheory.Iso (CategoryTheory.Limits.parallelPair (CategoryTheory.Lim …
    ⊢ CategoryTheory.Limits.IsLimit (F.mapCone (CategoryTheory.Limits.KernelFork.o …
  -/
  refine IsLimit.postcomposeInvEquiv e _ ?_
  /-
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    inst✝⁶ : CategoryTheory.Abelian C
    X✝ Y✝ : C
    f : Quiver.Hom X✝ Y✝
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{?u.114194, u_1} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms D
    X Y : D
    i : Quiver.Hom X Y
    inst✝³ : CategoryTheory.Limits.HasCokernel i
    F : CategoryTheory.Functor D C
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : CategoryTheory.Mono (F.map i)
    inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallel …
    e : CategoryTheory.Iso (CategoryTheory.Limits.parallelPair (CategoryTheory.Lim …
    ⊢ CategoryTheory.Limits.IsLimit ((CategoryTheory.Limits.Cones.postcompose e.in …
  -/
  let hi := Abelian.monoIsKernelOfCokernel _ (cokernelIsCokernel (F.map i))
  /-
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    inst✝⁶ : CategoryTheory.Abelian C
    X✝ Y✝ : C
    f : Quiver.Hom X✝ Y✝
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{?u.114194, u_1} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms D
    X Y : D
    i : Quiver.Hom X Y
    inst✝³ : CategoryTheory.Limits.HasCokernel i
    F : CategoryTheory.Functor D C
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : CategoryTheory.Mono (F.map i)
    inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallel …
    e : CategoryTheory.Iso (CategoryTheory.Limits.parallelPair (CategoryTheory.Lim …
    hi : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι (F.ma …
    ⊢ CategoryTheory.Limits.IsLimit ((CategoryTheory.Limits.Cones.postcompose e.in …
  -/
  refine IsLimit.ofIsoLimit hi (Fork.ext (Iso.refl _) ?_)
  /-
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    inst✝⁶ : CategoryTheory.Abelian C
    X✝ Y✝ : C
    f : Quiver.Hom X✝ Y✝
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{?u.114194, u_1} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms D
    X Y : D
    i : Quiver.Hom X Y
    inst✝³ : CategoryTheory.Limits.HasCokernel i
    F : CategoryTheory.Functor D C
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : CategoryTheory.Mono (F.map i)
    inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallel …
    e : CategoryTheory.Iso (CategoryTheory.Limits.parallelPair (CategoryTheory.Lim …
    hi : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι (F.ma …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Iso.refl (CategoryThe …
  -/
  change 𝟙 _ ≫ F.map i ≫ 𝟙 _ = F.map i
  /-
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    inst✝⁶ : CategoryTheory.Abelian C
    X✝ Y✝ : C
    f : Quiver.Hom X✝ Y✝
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{?u.114194, u_1} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms D
    X Y : D
    i : Quiver.Hom X Y
    inst✝³ : CategoryTheory.Limits.HasCokernel i
    F : CategoryTheory.Functor D C
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : CategoryTheory.Mono (F.map i)
    inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallel …
    e : CategoryTheory.Iso (CategoryTheory.Limits.parallelPair (CategoryTheory.Lim …
    hi : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι (F.ma …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id (F. …
  -/
  rw [Category.comp_id, Category.id_comp]
  /-
    🎉 no goals
  -/


/-- If `F : D ⥤ C` is a functor to an abelian category, `p : X ⟶ Y` is a morphisms
admitting a kernel such that `F` preserves this kernel and `F.map p` is an epi,
then `F.map Y` identifies to the cokernel of `F.map (kernel.ι p)`. -/
noncomputable def isColimitMapCoconeOfCokernelCoforkOfπ
    {X Y : D} (p : X ⟶ Y) [HasKernel p] (F : D ⥤ C)
    [F.PreservesZeroMorphisms] [Epi (F.map p)]
    [PreservesLimit (parallelPair p 0) F] :
    IsColimit (F.mapCocone (CokernelCofork.ofπ p (kernel.condition p))) := by
  let e : parallelPair (kernel.ι p) 0 ⋙ F ≅ parallelPair (kernel.ι (F.map p)) 0 :=
    parallelPair.ext (asIso (kernelComparison p F)) (Iso.refl _) (by simp) (by simp)
  /-
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    inst✝⁶ : CategoryTheory.Abelian C
    X✝ Y✝ : C
    f : Quiver.Hom X✝ Y✝
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{?u.124026, u_1} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms D
    X Y : D
    p : Quiver.Hom X Y
    inst✝³ : CategoryTheory.Limits.HasKernel p
    F : CategoryTheory.Functor D C
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : CategoryTheory.Epi (F.map p)
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPa …
    e : CategoryTheory.Iso ((CategoryTheory.Limits.parallelPair (CategoryTheory.Li …
    ⊢ CategoryTheory.Limits.IsColimit (F.mapCocone (CategoryTheory.Limits.Cokernel …
  -/
  refine IsColimit.precomposeInvEquiv e _ ?_
  /-
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    inst✝⁶ : CategoryTheory.Abelian C
    X✝ Y✝ : C
    f : Quiver.Hom X✝ Y✝
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{?u.124026, u_1} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms D
    X Y : D
    p : Quiver.Hom X Y
    inst✝³ : CategoryTheory.Limits.HasKernel p
    F : CategoryTheory.Functor D C
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : CategoryTheory.Epi (F.map p)
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPa …
    e : CategoryTheory.Iso ((CategoryTheory.Limits.parallelPair (CategoryTheory.Li …
    ⊢ CategoryTheory.Limits.IsColimit ((CategoryTheory.Limits.Cocones.precompose e …
  -/
  let hp := Abelian.epiIsCokernelOfKernel _ (kernelIsKernel (F.map p))
  /-
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    inst✝⁶ : CategoryTheory.Abelian C
    X✝ Y✝ : C
    f : Quiver.Hom X✝ Y✝
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{?u.124026, u_1} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms D
    X Y : D
    p : Quiver.Hom X Y
    inst✝³ : CategoryTheory.Limits.HasKernel p
    F : CategoryTheory.Functor D C
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : CategoryTheory.Epi (F.map p)
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPa …
    e : CategoryTheory.Iso ((CategoryTheory.Limits.parallelPair (CategoryTheory.Li …
    hp : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ …
    ⊢ CategoryTheory.Limits.IsColimit ((CategoryTheory.Limits.Cocones.precompose e …
  -/
  refine IsColimit.ofIsoColimit hp (Cofork.ext (Iso.refl _) ?_)
  /-
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    inst✝⁶ : CategoryTheory.Abelian C
    X✝ Y✝ : C
    f : Quiver.Hom X✝ Y✝
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{?u.124026, u_1} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms D
    X Y : D
    p : Quiver.Hom X Y
    inst✝³ : CategoryTheory.Limits.HasKernel p
    F : CategoryTheory.Functor D C
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : CategoryTheory.Epi (F.map p)
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPa …
    e : CategoryTheory.Iso ((CategoryTheory.Limits.parallelPair (CategoryTheory.Li …
    hp : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π (Cate …
  -/
  change F.map p ≫ 𝟙 _ = 𝟙 _ ≫ F.map p
  /-
    C : Type u
    inst✝⁷ : CategoryTheory.Category.{v, u} C
    inst✝⁶ : CategoryTheory.Abelian C
    X✝ Y✝ : C
    f : Quiver.Hom X✝ Y✝
    D : Type u_1
    inst✝⁵ : CategoryTheory.Category.{?u.124026, u_1} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms D
    X Y : D
    p : Quiver.Hom X Y
    inst✝³ : CategoryTheory.Limits.HasKernel p
    F : CategoryTheory.Functor D C
    inst✝² : F.PreservesZeroMorphisms
    inst✝¹ : CategoryTheory.Epi (F.map p)
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPa …
    e : CategoryTheory.Iso ((CategoryTheory.Limits.parallelPair (CategoryTheory.Li …
    hp : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (F.map p) (CategoryTheory.CategoryStr …
  -/
  rw [Category.comp_id, Category.id_comp]
  /-
    🎉 no goals
  -/


instance (priority := 100) hasEqualizers : HasEqualizers C :=
  Preadditive.hasEqualizers_of_hasKernels


/-- Any abelian category has pullbacks -/
instance (priority := 100) hasPullbacks : HasPullbacks C :=
  hasPullbacks_of_hasBinaryProducts_of_hasEqualizers C


instance (priority := 100) hasCoequalizers : HasCoequalizers C :=
  Preadditive.hasCoequalizers_of_hasCokernels


/-- Any abelian category has pushouts -/
instance (priority := 100) hasPushouts : HasPushouts C :=
  hasPushouts_of_hasBinaryCoproducts_of_hasCoequalizers C


instance (priority := 100) hasFiniteLimits : HasFiniteLimits C :=
  Limits.hasFiniteLimits_of_hasEqualizers_and_finite_products


instance (priority := 100) hasFiniteColimits : HasFiniteColimits C :=
  Limits.hasFiniteColimits_of_hasCoequalizers_and_finite_coproducts


/-- The canonical map `pullback f g ⟶ X ⊞ Y` -/
abbrev pullbackToBiproduct : pullback f g ⟶ X ⊞ Y :=
  biprod.lift (pullback.fst f g) (pullback.snd f g)


/-- The canonical map `pullback f g ⟶ X ⊞ Y` induces a kernel cone on the map
    `biproduct X Y ⟶ Z` induced by `f` and `g`. A slightly more intuitive way to think of
    this may be that it induces an equalizer fork on the maps induced by `(f, 0)` and
    `(0, g)`. -/
abbrev pullbackToBiproductFork : KernelFork (biprod.desc f (-g)) :=
  KernelFork.ofι (pullbackToBiproduct f g) <| by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      X Y Z : C
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Abelian.PullbackToBip …
    -/
    rw [biprod.lift_desc, comp_neg, pullback.condition, add_neg_cancel]
    /-
      🎉 no goals
    -/


/-- The canonical map `pullback f g ⟶ X ⊞ Y` is a kernel of the map induced by
    `(f, -g)`. -/
def isLimitPullbackToBiproduct : IsLimit (pullbackToBiproductFork f g) :=
  Fork.IsLimit.mk _
    (fun s =>
      pullback.lift (Fork.ι s ≫ biprod.fst) (Fork.ι s ≫ biprod.snd) <|
        sub_eq_zero.1 <| by
          rw [Category.assoc, Category.assoc, ← comp_sub, sub_eq_add_neg, ← comp_neg, ←
            biprod.desc_eq, KernelFork.condition s])
    (fun s => by
      /-
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.Abelian C
        inst✝ : CategoryTheory.Limits.HasPullbacks C
        X Y Z : C
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        s : CategoryTheory.Limits.Fork (CategoryTheory.Limits.biprod.desc f (Neg.neg g …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s => CategoryTheory.Limits.pull …
      -/
      apply biprod.hom_ext <;> rw [Fork.ι_ofι, Category.assoc]
        /-
          case h₀
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Abelian C
          inst✝ : CategoryTheory.Limits.HasPullbacks C
          X Y Z : C
          f : Quiver.Hom X Z
          g : Quiver.Hom Y Z
          s : CategoryTheory.Limits.Fork (CategoryTheory.Limits.biprod.desc f (Neg.neg g …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s => CategoryTheory.Limits.pull …
        -/
      · rw [biprod.lift_fst, pullback.lift_fst]
        /-
          🎉 no goals
        -/
        /-
          case h₁
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Abelian C
          inst✝ : CategoryTheory.Limits.HasPullbacks C
          X Y Z : C
          f : Quiver.Hom X Z
          g : Quiver.Hom Y Z
          s : CategoryTheory.Limits.Fork (CategoryTheory.Limits.biprod.desc f (Neg.neg g …
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s => CategoryTheory.Limits.pull …
        -/
      · rw [biprod.lift_snd, pullback.lift_snd])
        /-
          🎉 no goals
        -/
                    /-
                      C : Type u
                      inst✝² : CategoryTheory.Category.{v, u} C
                      inst✝¹ : CategoryTheory.Abelian C
                      inst✝ : CategoryTheory.Limits.HasPullbacks C
                      X Y Z : C
                      f : Quiver.Hom X Z
                      g : Quiver.Hom Y Z
                      s : CategoryTheory.Limits.Fork (CategoryTheory.Limits.biprod.desc f (Neg.neg g …
                      m : Quiver.Hom s.pt (CategoryTheory.Abelian.PullbackToBiproductIsKernel.pullba …
                      h : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.Fork.ι (Ca …
                      ⊢ Eq m ((fun s => CategoryTheory.Limits.pullback.lift (CategoryTheory.Category …
                    -/
                                               /-
                                                 🎉 no goals
                                               -/
    fun s m h => by apply pullback.hom_ext <;> simp [← h]
                                               /-
                                                 🎉 no goals
                                               -/


/-- The canonical map `Y ⊞ Z ⟶ pushout f g` -/
abbrev biproductToPushout : Y ⊞ Z ⟶ pushout f g :=
  biprod.desc (pushout.inl _ _) (pushout.inr _ _)


/-- The canonical map `Y ⊞ Z ⟶ pushout f g` induces a cokernel cofork on the map
    `X ⟶ Y ⊞ Z` induced by `f` and `-g`. -/
abbrev biproductToPushoutCofork : CokernelCofork (biprod.lift f (-g)) :=
  CokernelCofork.ofπ (biproductToPushout f g) <| by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : CategoryTheory.Limits.HasPushouts C
      W X Y Z : C
      f : Quiver.Hom X Y
      g : Quiver.Hom X Z
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biprod.lift f  …
    -/
    rw [biprod.lift_desc, neg_comp, pushout.condition, add_neg_cancel]
    /-
      🎉 no goals
    -/


/-- The cofork induced by the canonical map `Y ⊞ Z ⟶ pushout f g` is in fact a colimit cokernel
    cofork. -/
def isColimitBiproductToPushout : IsColimit (biproductToPushoutCofork f g) :=
  Cofork.IsColimit.mk _
    (fun s =>
      pushout.desc (biprod.inl ≫ Cofork.π s) (biprod.inr ≫ Cofork.π s) <|
        sub_eq_zero.1 <| by
          rw [← Category.assoc, ← Category.assoc, ← sub_comp, sub_eq_add_neg, ← neg_comp, ←
            biprod.lift_eq, Cofork.condition s, zero_comp])
                 /-
                   C : Type u
                   inst✝² : CategoryTheory.Category.{v, u} C
                   inst✝¹ : CategoryTheory.Abelian C
                   inst✝ : CategoryTheory.Limits.HasPushouts C
                   W X Y Z : C
                   f : Quiver.Hom X Y
                   g : Quiver.Hom X Z
                   s : CategoryTheory.Limits.Cofork (CategoryTheory.Limits.biprod.lift f (Neg.neg …
                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π (Cate …
                 -/
                                           /-
                                             🎉 no goals
                                           -/
    (fun s => by apply biprod.hom_ext' <;> simp)
                                           /-
                                             🎉 no goals
                                           -/
                    /-
                      C : Type u
                      inst✝² : CategoryTheory.Category.{v, u} C
                      inst✝¹ : CategoryTheory.Abelian C
                      inst✝ : CategoryTheory.Limits.HasPushouts C
                      W X Y Z : C
                      f : Quiver.Hom X Y
                      g : Quiver.Hom X Z
                      s : CategoryTheory.Limits.Cofork (CategoryTheory.Limits.biprod.lift f (Neg.neg …
                      m : Quiver.Hom (CategoryTheory.Abelian.BiproductToPushoutIsCokernel.biproductT …
                      h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π (Ca …
                      ⊢ Eq m ((fun s => CategoryTheory.Limits.pushout.desc (CategoryTheory.CategoryS …
                    -/
                                              /-
                                                🎉 no goals
                                              -/
    fun s m h => by apply pushout.hom_ext <;> simp [← h]
                                              /-
                                                🎉 no goals
                                              -/


/-- In an abelian category, the pullback of an epimorphism is an epimorphism.
    Proof from [aluffi2016, IX.2.3], cf. [borceux-vol2, 1.7.6] -/
instance epi_pullback_of_epi_f [Epi f] : Epi (pullback.snd f g) :=
  -- It will suffice to consider some morphism e : Y ⟶ R such that
    -- pullback.snd f g ≫ e = 0 and show that e = 0.
    epi_of_cancel_zero _ fun {R} e h => by
    -- Consider the morphism u := (0, e) : X ⊞ Y⟶ R.
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Abelian C
      inst✝¹ : CategoryTheory.Limits.HasPullbacks C
      W X Y Z : C
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : CategoryTheory.Epi f
      R : C
      e : Quiver.Hom Y R
      h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.snd …
      ⊢ Eq e 0
    -/
    let u := biprod.desc (0 : X ⟶ R) e
    -- The composite pullback f g ⟶ X ⊞ Y ⟶ R is zero by assumption.
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Abelian C
      inst✝¹ : CategoryTheory.Limits.HasPullbacks C
      W X Y Z : C
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : CategoryTheory.Epi f
      R : C
      e : Quiver.Hom Y R
      h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.snd …
      u : Quiver.Hom (CategoryTheory.Limits.biprod X Y) R := CategoryTheory.Limits.b …
      ⊢ Eq e 0
    -/
    have hu : PullbackToBiproductIsKernel.pullbackToBiproduct f g ≫ u = 0 := by simpa [u]
    -- pullbackToBiproduct f g is a kernel of (f, -g), so (f, -g) is a
    -- cokernel of pullbackToBiproduct f g
    have :=
      epiIsCokernelOfKernel _
        (PullbackToBiproductIsKernel.isLimitPullbackToBiproduct f g)
    -- We use this fact to obtain a factorization of u through (f, -g) via some d : Z ⟶ R.
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Abelian C
      inst✝¹ : CategoryTheory.Limits.HasPullbacks C
      W X Y Z : C
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : CategoryTheory.Epi f
      R : C
      e : Quiver.Hom Y R
      h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.snd …
      u : Quiver.Hom (CategoryTheory.Limits.biprod X Y) R := CategoryTheory.Limits.b …
      hu : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Abelian.PullbackTo …
      this : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.o …
      ⊢ Eq e 0
    -/
    obtain ⟨d, hd⟩ := CokernelCofork.IsColimit.desc' this u hu
    /-
      case mk
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Abelian C
      inst✝¹ : CategoryTheory.Limits.HasPullbacks C
      W X Y Z : C
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : CategoryTheory.Epi f
      R : C
      e : Quiver.Hom Y R
      h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.snd …
      u : Quiver.Hom (CategoryTheory.Limits.biprod X Y) R := CategoryTheory.Limits.b …
      hu : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Abelian.PullbackTo …
      this : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.o …
      d : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ (CategoryTheory.Limit …
      hd : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π (C …
      ⊢ Eq e 0
    -/
    dsimp at d; dsimp [u] at hd
    -- But then f ≫ d = 0:
    have : f ≫ d = 0 := calc
      f ≫ d = (biprod.inl ≫ biprod.desc f (-g)) ≫ d := by rw [biprod.inl_desc]
      _ = biprod.inl ≫ u := by rw [Category.assoc, hd]
      _ = 0 := biprod.inl_desc _ _
    -- But f is an epimorphism, so d = 0...
    /-
      case mk
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Abelian C
      inst✝¹ : CategoryTheory.Limits.HasPullbacks C
      W X Y Z : C
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : CategoryTheory.Epi f
      R : C
      e : Quiver.Hom Y R
      h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.snd …
      u : Quiver.Hom (CategoryTheory.Limits.biprod X Y) R := CategoryTheory.Limits.b …
      hu : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Abelian.PullbackTo …
      this✝ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork. …
      d : Quiver.Hom Z R
      hd : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biprod.desc …
      this : Eq (CategoryTheory.CategoryStruct.comp f d) 0
      ⊢ Eq e 0
    -/
    have : d = 0 := (cancel_epi f).1 (by simpa)
    -- ...or, in other words, e = 0.
    calc
      e = biprod.inr ≫ biprod.desc (0 : X ⟶ R) e := by rw [biprod.inr_desc]
      _ = biprod.inr ≫ biprod.desc f (-g) ≫ d := by rw [← hd]
      _ = biprod.inr ≫ biprod.desc f (-g) ≫ 0 := by rw [this]
      _ = (biprod.inr ≫ biprod.desc f (-g)) ≫ 0 := by rw [← Category.assoc]
      _ = 0 := HasZeroMorphisms.comp_zero _ _


/-- In an abelian category, the pullback of an epimorphism is an epimorphism. -/
instance epi_pullback_of_epi_g [Epi g] : Epi (pullback.fst f g) :=
  -- It will suffice to consider some morphism e : X ⟶ R such that
  -- pullback.fst f g ≫ e = 0 and show that e = 0.
  epi_of_cancel_zero _ fun {R} e h => by
    -- Consider the morphism u := (e, 0) : X ⊞ Y ⟶ R.
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Abelian C
      inst✝¹ : CategoryTheory.Limits.HasPullbacks C
      W X Y Z : C
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : CategoryTheory.Epi g
      R : C
      e : Quiver.Hom X R
      h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.fst …
      ⊢ Eq e 0
    -/
    let u := biprod.desc e (0 : Y ⟶ R)
    -- The composite pullback f g ⟶ X ⊞ Y ⟶ R is zero by assumption.
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Abelian C
      inst✝¹ : CategoryTheory.Limits.HasPullbacks C
      W X Y Z : C
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : CategoryTheory.Epi g
      R : C
      e : Quiver.Hom X R
      h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.fst …
      u : Quiver.Hom (CategoryTheory.Limits.biprod X Y) R := CategoryTheory.Limits.b …
      ⊢ Eq e 0
    -/
    have hu : PullbackToBiproductIsKernel.pullbackToBiproduct f g ≫ u = 0 := by simpa [u]
    -- pullbackToBiproduct f g is a kernel of (f, -g), so (f, -g) is a
    -- cokernel of pullbackToBiproduct f g
    have :=
      epiIsCokernelOfKernel _
        (PullbackToBiproductIsKernel.isLimitPullbackToBiproduct f g)
    -- We use this fact to obtain a factorization of u through (f, -g) via some d : Z ⟶ R.
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Abelian C
      inst✝¹ : CategoryTheory.Limits.HasPullbacks C
      W X Y Z : C
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : CategoryTheory.Epi g
      R : C
      e : Quiver.Hom X R
      h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.fst …
      u : Quiver.Hom (CategoryTheory.Limits.biprod X Y) R := CategoryTheory.Limits.b …
      hu : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Abelian.PullbackTo …
      this : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.o …
      ⊢ Eq e 0
    -/
    obtain ⟨d, hd⟩ := CokernelCofork.IsColimit.desc' this u hu
    /-
      case mk
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Abelian C
      inst✝¹ : CategoryTheory.Limits.HasPullbacks C
      W X Y Z : C
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : CategoryTheory.Epi g
      R : C
      e : Quiver.Hom X R
      h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.fst …
      u : Quiver.Hom (CategoryTheory.Limits.biprod X Y) R := CategoryTheory.Limits.b …
      hu : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Abelian.PullbackTo …
      this : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.o …
      d : Quiver.Hom (CategoryTheory.Limits.CokernelCofork.ofπ (CategoryTheory.Limit …
      hd : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π (C …
      ⊢ Eq e 0
    -/
    dsimp at d; dsimp [u] at hd
    -- But then (-g) ≫ d = 0:
    have : (-g) ≫ d = 0 := calc
      (-g) ≫ d = (biprod.inr ≫ biprod.desc f (-g)) ≫ d := by rw [biprod.inr_desc]
      _ = biprod.inr ≫ u := by rw [Category.assoc, hd]
      _ = 0 := biprod.inr_desc _ _
    -- But g is an epimorphism, thus so is -g, so d = 0...
    /-
      case mk
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Abelian C
      inst✝¹ : CategoryTheory.Limits.HasPullbacks C
      W X Y Z : C
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : CategoryTheory.Epi g
      R : C
      e : Quiver.Hom X R
      h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.fst …
      u : Quiver.Hom (CategoryTheory.Limits.biprod X Y) R := CategoryTheory.Limits.b …
      hu : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Abelian.PullbackTo …
      this✝ : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork. …
      d : Quiver.Hom Z R
      hd : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.biprod.desc …
      this : Eq (CategoryTheory.CategoryStruct.comp (Neg.neg g) d) 0
      ⊢ Eq e 0
    -/
    have : d = 0 := (cancel_epi (-g)).1 (by simpa)
    -- ...or, in other words, e = 0.
    calc
      e = biprod.inl ≫ biprod.desc e (0 : Y ⟶ R) := by rw [biprod.inl_desc]
      _ = biprod.inl ≫ biprod.desc f (-g) ≫ d := by rw [← hd]
      _ = biprod.inl ≫ biprod.desc f (-g) ≫ 0 := by rw [this]
      _ = (biprod.inl ≫ biprod.desc f (-g)) ≫ 0 := by rw [← Category.assoc]
      _ = 0 := HasZeroMorphisms.comp_zero _ _


theorem epi_snd_of_isLimit [Epi f] {s : PullbackCone f g} (hs : IsLimit s) : Epi s.snd := by
  haveI : Epi (NatTrans.app (limit.cone (cospan f g)).π WalkingCospan.right) :=
    Abelian.epi_pullback_of_epi_f f g
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : CategoryTheory.Limits.HasPullbacks C
    X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : CategoryTheory.Epi f
    s : CategoryTheory.Limits.PullbackCone f g
    hs : CategoryTheory.Limits.IsLimit s
    this : CategoryTheory.Epi ((CategoryTheory.Limits.limit.cone (CategoryTheory.L …
    ⊢ CategoryTheory.Epi s.snd
  -/
  apply epi_of_epi_fac (IsLimit.conePointUniqueUpToIso_hom_comp (limit.isLimit _) hs _)
  /-
    🎉 no goals
  -/


theorem epi_fst_of_isLimit [Epi g] {s : PullbackCone f g} (hs : IsLimit s) : Epi s.fst := by
  haveI : Epi (NatTrans.app (limit.cone (cospan f g)).π WalkingCospan.left) :=
    Abelian.epi_pullback_of_epi_g f g
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : CategoryTheory.Limits.HasPullbacks C
    X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : CategoryTheory.Epi g
    s : CategoryTheory.Limits.PullbackCone f g
    hs : CategoryTheory.Limits.IsLimit s
    this : CategoryTheory.Epi ((CategoryTheory.Limits.limit.cone (CategoryTheory.L …
    ⊢ CategoryTheory.Epi s.fst
  -/
  apply epi_of_epi_fac (IsLimit.conePointUniqueUpToIso_hom_comp (limit.isLimit _) hs _)
  /-
    🎉 no goals
  -/


/-- Suppose `f` and `g` are two morphisms with a common codomain and suppose we have written `g` as
    an epimorphism followed by a monomorphism. If `f` factors through the mono part of this
    factorization, then any pullback of `g` along `f` is an epimorphism. -/
theorem epi_fst_of_factor_thru_epi_mono_factorization (g₁ : Y ⟶ W) [Epi g₁] (g₂ : W ⟶ Z) [Mono g₂]
    (hg : g₁ ≫ g₂ = g) (f' : X ⟶ W) (hf : f' ≫ g₂ = f) (t : PullbackCone f g) (ht : IsLimit t) :
    Epi t.fst := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Abelian C
    inst✝² : CategoryTheory.Limits.HasPullbacks C
    W X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    g₁ : Quiver.Hom Y W
    inst✝¹ : CategoryTheory.Epi g₁
    g₂ : Quiver.Hom W Z
    inst✝ : CategoryTheory.Mono g₂
    hg : Eq (CategoryTheory.CategoryStruct.comp g₁ g₂) g
    f' : Quiver.Hom X W
    hf : Eq (CategoryTheory.CategoryStruct.comp f' g₂) f
    t : CategoryTheory.Limits.PullbackCone f g
    ht : CategoryTheory.Limits.IsLimit t
    ⊢ CategoryTheory.Epi t.fst
  -/
  apply epi_fst_of_isLimit _ _ (PullbackCone.isLimitOfFactors f g g₂ f' g₁ hf hg t ht)
  /-
    🎉 no goals
  -/


instance mono_pushout_of_mono_f [Mono f] : Mono (pushout.inr _ _ : Z ⟶ pushout f g) :=
  mono_of_cancel_zero _ fun {R} e h => by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Abelian C
      inst✝¹ : CategoryTheory.Limits.HasPushouts C
      W X Y Z : C
      f : Quiver.Hom X Y
      g : Quiver.Hom X Z
      inst✝ : CategoryTheory.Mono f
      R : C
      e : Quiver.Hom R Z
      h : Eq (CategoryTheory.CategoryStruct.comp e (CategoryTheory.Limits.pushout.in …
      ⊢ Eq e 0
    -/
    let u := biprod.lift (0 : R ⟶ Y) e
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Abelian C
      inst✝¹ : CategoryTheory.Limits.HasPushouts C
      W X Y Z : C
      f : Quiver.Hom X Y
      g : Quiver.Hom X Z
      inst✝ : CategoryTheory.Mono f
      R : C
      e : Quiver.Hom R Z
      h : Eq (CategoryTheory.CategoryStruct.comp e (CategoryTheory.Limits.pushout.in …
      u : Quiver.Hom R (CategoryTheory.Limits.biprod Y Z) := CategoryTheory.Limits.b …
      ⊢ Eq e 0
    -/
    have hu : u ≫ BiproductToPushoutIsCokernel.biproductToPushout f g = 0 := by simpa [u]
    have :=
      monoIsKernelOfCokernel _
        (BiproductToPushoutIsCokernel.isColimitBiproductToPushout f g)
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Abelian C
      inst✝¹ : CategoryTheory.Limits.HasPushouts C
      W X Y Z : C
      f : Quiver.Hom X Y
      g : Quiver.Hom X Z
      inst✝ : CategoryTheory.Mono f
      R : C
      e : Quiver.Hom R Z
      h : Eq (CategoryTheory.CategoryStruct.comp e (CategoryTheory.Limits.pushout.in …
      u : Quiver.Hom R (CategoryTheory.Limits.biprod Y Z) := CategoryTheory.Limits.b …
      hu : Eq (CategoryTheory.CategoryStruct.comp u (CategoryTheory.Abelian.Biproduc …
      this : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι (Ca …
      ⊢ Eq e 0
    -/
    obtain ⟨d, hd⟩ := KernelFork.IsLimit.lift' this u hu
    /-
      case mk
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Abelian C
      inst✝¹ : CategoryTheory.Limits.HasPushouts C
      W X Y Z : C
      f : Quiver.Hom X Y
      g : Quiver.Hom X Z
      inst✝ : CategoryTheory.Mono f
      R : C
      e : Quiver.Hom R Z
      h : Eq (CategoryTheory.CategoryStruct.comp e (CategoryTheory.Limits.pushout.in …
      u : Quiver.Hom R (CategoryTheory.Limits.biprod Y Z) := CategoryTheory.Limits.b …
      hu : Eq (CategoryTheory.CategoryStruct.comp u (CategoryTheory.Abelian.Biproduc …
      this : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι (Ca …
      d : Quiver.Hom R (CategoryTheory.Limits.KernelFork.ofι (CategoryTheory.Limits. …
      hd : Eq (CategoryTheory.CategoryStruct.comp d (CategoryTheory.Limits.Fork.ι (C …
      ⊢ Eq e 0
    -/
    dsimp at d
    /-
      case mk
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Abelian C
      inst✝¹ : CategoryTheory.Limits.HasPushouts C
      W X Y Z : C
      f : Quiver.Hom X Y
      g : Quiver.Hom X Z
      inst✝ : CategoryTheory.Mono f
      R : C
      e : Quiver.Hom R Z
      h : Eq (CategoryTheory.CategoryStruct.comp e (CategoryTheory.Limits.pushout.in …
      u : Quiver.Hom R (CategoryTheory.Limits.biprod Y Z) := CategoryTheory.Limits.b …
      hu : Eq (CategoryTheory.CategoryStruct.comp u (CategoryTheory.Abelian.Biproduc …
      this : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι (Ca …
      d : Quiver.Hom R X
      hd : Eq (CategoryTheory.CategoryStruct.comp d (CategoryTheory.Limits.Fork.ι (C …
      ⊢ Eq e 0
    -/
    dsimp [u] at hd
    have : d ≫ f = 0 := calc
      d ≫ f = d ≫ biprod.lift f (-g) ≫ biprod.fst := by rw [biprod.lift_fst]
      _ = u ≫ biprod.fst := by rw [← Category.assoc, hd]
      _ = 0 := biprod.lift_fst _ _
    /-
      case mk
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Abelian C
      inst✝¹ : CategoryTheory.Limits.HasPushouts C
      W X Y Z : C
      f : Quiver.Hom X Y
      g : Quiver.Hom X Z
      inst✝ : CategoryTheory.Mono f
      R : C
      e : Quiver.Hom R Z
      h : Eq (CategoryTheory.CategoryStruct.comp e (CategoryTheory.Limits.pushout.in …
      u : Quiver.Hom R (CategoryTheory.Limits.biprod Y Z) := CategoryTheory.Limits.b …
      hu : Eq (CategoryTheory.CategoryStruct.comp u (CategoryTheory.Abelian.Biproduc …
      this✝ : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι (C …
      d : Quiver.Hom R X
      hd : Eq (CategoryTheory.CategoryStruct.comp d (CategoryTheory.Limits.biprod.li …
      this : Eq (CategoryTheory.CategoryStruct.comp d f) 0
      ⊢ Eq e 0
    -/
    have : d = 0 := (cancel_mono f).1 (by simpa)
    calc
      e = biprod.lift (0 : R ⟶ Y) e ≫ biprod.snd := by rw [biprod.lift_snd]
      _ = (d ≫ biprod.lift f (-g)) ≫ biprod.snd := by rw [← hd]
      _ = (0 ≫ biprod.lift f (-g)) ≫ biprod.snd := by rw [this]
      _ = 0 ≫ biprod.lift f (-g) ≫ biprod.snd := by rw [Category.assoc]
      _ = 0 := zero_comp


instance mono_pushout_of_mono_g [Mono g] : Mono (pushout.inl f g) :=
  mono_of_cancel_zero _ fun {R} e h => by
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Abelian C
      inst✝¹ : CategoryTheory.Limits.HasPushouts C
      W X Y Z : C
      f : Quiver.Hom X Y
      g : Quiver.Hom X Z
      inst✝ : CategoryTheory.Mono g
      R : C
      e : Quiver.Hom R Y
      h : Eq (CategoryTheory.CategoryStruct.comp e (CategoryTheory.Limits.pushout.in …
      ⊢ Eq e 0
    -/
    let u := biprod.lift e (0 : R ⟶ Z)
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Abelian C
      inst✝¹ : CategoryTheory.Limits.HasPushouts C
      W X Y Z : C
      f : Quiver.Hom X Y
      g : Quiver.Hom X Z
      inst✝ : CategoryTheory.Mono g
      R : C
      e : Quiver.Hom R Y
      h : Eq (CategoryTheory.CategoryStruct.comp e (CategoryTheory.Limits.pushout.in …
      u : Quiver.Hom R (CategoryTheory.Limits.biprod Y Z) := CategoryTheory.Limits.b …
      ⊢ Eq e 0
    -/
    have hu : u ≫ BiproductToPushoutIsCokernel.biproductToPushout f g = 0 := by simpa [u]
    have :=
      monoIsKernelOfCokernel _
        (BiproductToPushoutIsCokernel.isColimitBiproductToPushout f g)
    /-
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Abelian C
      inst✝¹ : CategoryTheory.Limits.HasPushouts C
      W X Y Z : C
      f : Quiver.Hom X Y
      g : Quiver.Hom X Z
      inst✝ : CategoryTheory.Mono g
      R : C
      e : Quiver.Hom R Y
      h : Eq (CategoryTheory.CategoryStruct.comp e (CategoryTheory.Limits.pushout.in …
      u : Quiver.Hom R (CategoryTheory.Limits.biprod Y Z) := CategoryTheory.Limits.b …
      hu : Eq (CategoryTheory.CategoryStruct.comp u (CategoryTheory.Abelian.Biproduc …
      this : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι (Ca …
      ⊢ Eq e 0
    -/
    obtain ⟨d, hd⟩ := KernelFork.IsLimit.lift' this u hu
    /-
      case mk
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Abelian C
      inst✝¹ : CategoryTheory.Limits.HasPushouts C
      W X Y Z : C
      f : Quiver.Hom X Y
      g : Quiver.Hom X Z
      inst✝ : CategoryTheory.Mono g
      R : C
      e : Quiver.Hom R Y
      h : Eq (CategoryTheory.CategoryStruct.comp e (CategoryTheory.Limits.pushout.in …
      u : Quiver.Hom R (CategoryTheory.Limits.biprod Y Z) := CategoryTheory.Limits.b …
      hu : Eq (CategoryTheory.CategoryStruct.comp u (CategoryTheory.Abelian.Biproduc …
      this : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι (Ca …
      d : Quiver.Hom R (CategoryTheory.Limits.KernelFork.ofι (CategoryTheory.Limits. …
      hd : Eq (CategoryTheory.CategoryStruct.comp d (CategoryTheory.Limits.Fork.ι (C …
      ⊢ Eq e 0
    -/
    dsimp at d
    /-
      case mk
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Abelian C
      inst✝¹ : CategoryTheory.Limits.HasPushouts C
      W X Y Z : C
      f : Quiver.Hom X Y
      g : Quiver.Hom X Z
      inst✝ : CategoryTheory.Mono g
      R : C
      e : Quiver.Hom R Y
      h : Eq (CategoryTheory.CategoryStruct.comp e (CategoryTheory.Limits.pushout.in …
      u : Quiver.Hom R (CategoryTheory.Limits.biprod Y Z) := CategoryTheory.Limits.b …
      hu : Eq (CategoryTheory.CategoryStruct.comp u (CategoryTheory.Abelian.Biproduc …
      this : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι (Ca …
      d : Quiver.Hom R X
      hd : Eq (CategoryTheory.CategoryStruct.comp d (CategoryTheory.Limits.Fork.ι (C …
      ⊢ Eq e 0
    -/
    dsimp [u] at hd
    have : d ≫ (-g) = 0 := calc
      d ≫ (-g) = d ≫ biprod.lift f (-g) ≫ biprod.snd := by rw [biprod.lift_snd]
      _ = biprod.lift e (0 : R ⟶ Z) ≫ biprod.snd := by rw [← Category.assoc, hd]
      _ = 0 := biprod.lift_snd _ _
    /-
      case mk
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Abelian C
      inst✝¹ : CategoryTheory.Limits.HasPushouts C
      W X Y Z : C
      f : Quiver.Hom X Y
      g : Quiver.Hom X Z
      inst✝ : CategoryTheory.Mono g
      R : C
      e : Quiver.Hom R Y
      h : Eq (CategoryTheory.CategoryStruct.comp e (CategoryTheory.Limits.pushout.in …
      u : Quiver.Hom R (CategoryTheory.Limits.biprod Y Z) := CategoryTheory.Limits.b …
      hu : Eq (CategoryTheory.CategoryStruct.comp u (CategoryTheory.Abelian.Biproduc …
      this✝ : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι (C …
      d : Quiver.Hom R X
      hd : Eq (CategoryTheory.CategoryStruct.comp d (CategoryTheory.Limits.biprod.li …
      this : Eq (CategoryTheory.CategoryStruct.comp d (Neg.neg g)) 0
      ⊢ Eq e 0
    -/
    have : d = 0 := (cancel_mono (-g)).1 (by simpa)
    calc
      e = biprod.lift e (0 : R ⟶ Z) ≫ biprod.fst := by rw [biprod.lift_fst]
      _ = (d ≫ biprod.lift f (-g)) ≫ biprod.fst := by rw [← hd]
      _ = (0 ≫ biprod.lift f (-g)) ≫ biprod.fst := by rw [this]
      _ = 0 ≫ biprod.lift f (-g) ≫ biprod.fst := by rw [Category.assoc]
      _ = 0 := zero_comp


theorem mono_inr_of_isColimit [Mono f] {s : PushoutCocone f g} (hs : IsColimit s) : Mono s.inr := by
  haveI : Mono (NatTrans.app (colimit.cocone (span f g)).ι WalkingCospan.right) :=
    Abelian.mono_pushout_of_mono_f f g
  apply
    mono_of_mono_fac (IsColimit.comp_coconePointUniqueUpToIso_hom hs (colimit.isColimit _) _)


theorem mono_inl_of_isColimit [Mono g] {s : PushoutCocone f g} (hs : IsColimit s) : Mono s.inl := by
  haveI : Mono (NatTrans.app (colimit.cocone (span f g)).ι WalkingCospan.left) :=
    Abelian.mono_pushout_of_mono_g f g
  apply
    mono_of_mono_fac (IsColimit.comp_coconePointUniqueUpToIso_hom hs (colimit.isColimit _) _)


/-- Suppose `f` and `g` are two morphisms with a common domain and suppose we have written `g` as
    an epimorphism followed by a monomorphism. If `f` factors through the epi part of this
    factorization, then any pushout of `g` along `f` is a monomorphism. -/
theorem mono_inl_of_factor_thru_epi_mono_factorization (f : X ⟶ Y) (g : X ⟶ Z) (g₁ : X ⟶ W) [Epi g₁]
    (g₂ : W ⟶ Z) [Mono g₂] (hg : g₁ ≫ g₂ = g) (f' : W ⟶ Y) (hf : g₁ ≫ f' = f)
    (t : PushoutCocone f g) (ht : IsColimit t) : Mono t.inl := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Abelian C
    inst✝² : CategoryTheory.Limits.HasPushouts C
    W X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom X Z
    g₁ : Quiver.Hom X W
    inst✝¹ : CategoryTheory.Epi g₁
    g₂ : Quiver.Hom W Z
    inst✝ : CategoryTheory.Mono g₂
    hg : Eq (CategoryTheory.CategoryStruct.comp g₁ g₂) g
    f' : Quiver.Hom W Y
    hf : Eq (CategoryTheory.CategoryStruct.comp g₁ f') f
    t : CategoryTheory.Limits.PushoutCocone f g
    ht : CategoryTheory.Limits.IsColimit t
    ⊢ CategoryTheory.Mono t.inl
  -/
  apply mono_inl_of_isColimit _ _ (PushoutCocone.isColimitOfFactors _ _ _ _ _ hf hg t ht)
  /-
    🎉 no goals
  -/


/-- Every NonPreadditiveAbelian category can be promoted to an abelian category. -/
def abelian : Abelian C :=
  {/- We need the `convert`s here because the instances we have are slightly different from the
       instances we need: `HasKernels` depends on an instance of `HasZeroMorphisms`. In the
       case of `NonPreadditiveAbelian`, this instance is an explicit argument. However, in the case
       of `abelian`, the `HasZeroMorphisms` instance is derived from `Preadditive`. So we need to
       transform an instance of "has kernels with NonPreadditiveAbelian.HasZeroMorphisms" to an
       instance of "has kernels with NonPreadditiveAbelian.Preadditive.HasZeroMorphisms".
       Luckily, we have a `subsingleton` instance for `HasZeroMorphisms`, so `convert` can
       immediately close the goal it creates for the two instances of `HasZeroMorphisms`,
       and the proof is complete. -/
    NonPreadditiveAbelian.preadditive with
                              /-
                                C : Type u
                                inst✝¹ : CategoryTheory.Category.{v, u} C
                                inst✝ : CategoryTheory.NonPreadditiveAbelian C
                                ⊢ CategoryTheory.Limits.HasFiniteProducts C
                              -/
    has_finite_products := by infer_instance
                              /-
                                🎉 no goals
                              -/
                      /-
                        C : Type u
                        inst✝¹ : CategoryTheory.Category.{v, u} C
                        inst✝ : CategoryTheory.NonPreadditiveAbelian C
                        ⊢ CategoryTheory.Limits.HasKernels C
                      -/
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.NonPreadditiveAbelian C
        ⊢ {X Y : C} → (f : Quiver.Hom X Y) → [inst : CategoryTheory.Mono f] → Category …
      -/
    has_kernels := by convert (by infer_instance : Limits.HasKernels C)
      /-
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.NonPreadditiveAbelian C
        X✝ Y✝ : C
        f : Quiver.Hom X✝ Y✝
        inst✝ : CategoryTheory.Mono f
        ⊢ CategoryTheory.NormalMono f
      -/
                      /-
                        🎉 no goals
                      -/
      /-
        🎉 no goals
      -/
                        /-
                          C : Type u
                          inst✝¹ : CategoryTheory.Category.{v, u} C
                          inst✝ : CategoryTheory.NonPreadditiveAbelian C
                          ⊢ CategoryTheory.Limits.HasCokernels C
                        -/
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.NonPreadditiveAbelian C
        ⊢ {X Y : C} → (f : Quiver.Hom X Y) → [inst : CategoryTheory.Epi f] → CategoryT …
      -/
    has_cokernels := by convert (by infer_instance : Limits.HasCokernels C)
      /-
        C : Type u
        inst✝² : CategoryTheory.Category.{v, u} C
        inst✝¹ : CategoryTheory.NonPreadditiveAbelian C
        X✝ Y✝ : C
        f : Quiver.Hom X✝ Y✝
        inst✝ : CategoryTheory.Epi f
        ⊢ CategoryTheory.NormalEpi f
      -/
                        /-
                          🎉 no goals
                        -/
      /-
        🎉 no goals
      -/
    normalMonoOfMono := by
      intro _ _ f _
      convert normalMonoOfMono f
    normalEpiOfEpi := by
      intro _ _ f _
      convert normalEpiOfEpi f }


