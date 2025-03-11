@[reassoc (attr := simp)]
lemma map_condition : G.map c.ι ≫ G.map f = 0 := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
    X Y : C
    f : Quiver.Hom X Y
    c : CategoryTheory.Limits.KernelFork f
    G : CategoryTheory.Functor C D
    inst✝ : G.PreservesZeroMorphisms
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory.Limits.Fork.ι  …
  -/
  rw [← G.map_comp, c.condition, G.map_zero]
  /-
    🎉 no goals
  -/


/-- A kernel fork for `f` is mapped to a kernel fork for `G.map f` if `G` is a functor
which preserves zero morphisms. -/
def map : KernelFork (G.map f) :=
  KernelFork.ofι (G.map c.ι) (c.map_condition G)


@[simp]
lemma map_ι : (c.map G).ι = G.map c.ι := rfl


/-- The underlying cone of a kernel fork is mapped to a limit cone if and only if
the mapped kernel fork is limit. -/
def isLimitMapConeEquiv :
    IsLimit (G.mapCone c) ≃ IsLimit (c.map G) := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
    X Y : C
    f : Quiver.Hom X Y
    c : CategoryTheory.Limits.KernelFork f
    G : CategoryTheory.Functor C D
    inst✝ : G.PreservesZeroMorphisms
    ⊢ Equiv (CategoryTheory.Limits.IsLimit (G.mapCone c)) (CategoryTheory.Limits.I …
  -/
  refine (IsLimit.postcomposeHomEquiv ?_ _).symm.trans (IsLimit.equivIsoLimit ?_)
  /-
    case refine_1
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
    X Y : C
    f : Quiver.Hom X Y
    c : CategoryTheory.Limits.KernelFork f
    G : CategoryTheory.Functor C D
    inst✝ : G.PreservesZeroMorphisms
    ⊢ CategoryTheory.Iso ((CategoryTheory.Limits.parallelPair f 0).comp G) (Catego …
  -/
                                                              /-
                                                                🎉 no goals
                                                              -/
  refine parallelPair.ext (Iso.refl _) (Iso.refl _) ?_ ?_ <;> simp
                                                              /-
                                                                🎉 no goals
                                                              -/
  /-
    case refine_2
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
    X Y : C
    f : Quiver.Hom X Y
    c : CategoryTheory.Limits.KernelFork f
    G : CategoryTheory.Functor C D
    inst✝ : G.PreservesZeroMorphisms
    ⊢ CategoryTheory.Iso ((CategoryTheory.Limits.Cones.postcompose (CategoryTheory …
  -/
  exact Cones.ext (Iso.refl _) (by rintro (_|_) <;> aesop_cat)
  /-
    🎉 no goals
  -/


/-- A limit kernel fork is mapped to a limit kernel fork by a functor `G` when this functor
preserves the corresponding limit. -/
def mapIsLimit (hc : IsLimit c) (G : C ⥤ D)
    [Functor.PreservesZeroMorphisms G] [PreservesLimit (parallelPair f 0) G] :
    IsLimit (c.map G) :=
  c.isLimitMapConeEquiv G (isLimitOfPreserves G hc)


/-- The map of a kernel fork is a limit iff
the kernel fork consisting of the mapped morphisms is a limit.
This essentially lets us commute `KernelFork.ofι` with `Functor.mapCone`.

This is a variant of `isLimitMapConeForkEquiv` for equalizers,
which we can't use directly between `G.map 0 = 0` does not hold definitionally.
-/
def isLimitMapConeForkEquiv' :
    IsLimit (G.mapCone (KernelFork.ofι h w)) ≃
      IsLimit
                                      /-
                                        C : Type u₁
                                        inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
                                        inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                                        D : Type u₂
                                        inst✝² : CategoryTheory.Category.{v₂, u₂} D
                                        inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
                                        G : CategoryTheory.Functor C D
                                        inst✝ : G.PreservesZeroMorphisms
                                        X Y Z : C
                                        f : Quiver.Hom X Y
                                        h : Quiver.Hom Z X
                                        w : Eq (CategoryTheory.CategoryStruct.comp h f) 0
                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map h) (G.map f)) 0
                                      -/
        (KernelFork.ofι (G.map h) (by simp only [← G.map_comp, w, Functor.map_zero]) :
                                      /-
                                        🎉 no goals
                                      -/
          Fork (G.map f) 0) :=
  KernelFork.isLimitMapConeEquiv _ _


/-- The property of preserving kernels expressed in terms of kernel forks.

This is a variant of `isLimitForkMapOfIsLimit` for equalizers,
which we can't use directly between `G.map 0 = 0` does not hold definitionally.
-/
def isLimitForkMapOfIsLimit' [PreservesLimit (parallelPair f 0) G]
    (l : IsLimit (KernelFork.ofι h w)) :
    IsLimit
                                    /-
                                      C : Type u₁
                                      inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
                                      inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
                                      D : Type u₂
                                      inst✝³ : CategoryTheory.Category.{v₂, u₂} D
                                      inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
                                      G : CategoryTheory.Functor C D
                                      inst✝¹ : G.PreservesZeroMorphisms
                                      X Y Z : C
                                      f : Quiver.Hom X Y
                                      h : Quiver.Hom Z X
                                      w : Eq (CategoryTheory.CategoryStruct.comp h f) 0
                                      inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPa …
                                      l : CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι h w)
                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map h) (G.map f)) 0
                                    -/
      (KernelFork.ofι (G.map h) (by simp only [← G.map_comp, w, Functor.map_zero]) :
                                    /-
                                      🎉 no goals
                                    -/
        Fork (G.map f) 0) :=
  isLimitMapConeForkEquiv' G w (isLimitOfPreserves G l)


/-- If `G` preserves kernels and `C` has them, then the fork constructed of the mapped morphisms of
a kernel fork is a limit.
-/
def isLimitOfHasKernelOfPreservesLimit [PreservesLimit (parallelPair f 0) G] :
    IsLimit
      (Fork.ofι (G.map (kernel.ι f))
              /-
                C : Type u₁
                inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
                inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
                D : Type u₂
                inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
                inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
                G : CategoryTheory.Functor C D
                inst✝² : G.PreservesZeroMorphisms
                X Y Z : C
                f : Quiver.Hom X Y
                h : Quiver.Hom Z X
                w : Eq (CategoryTheory.CategoryStruct.comp h f) 0
                inst✝¹ : CategoryTheory.Limits.HasKernel f
                inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPa …
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map (CategoryTheory.Limits.kernel. …
              -/
          (by simp only [← G.map_comp, kernel.condition, comp_zero, Functor.map_zero]) :
              /-
                🎉 no goals
              -/
        Fork (G.map f) 0) :=
  isLimitForkMapOfIsLimit' G (kernel.condition f) (kernelIsKernel f)


instance [PreservesLimit (parallelPair f 0) G] : HasKernel (G.map f) where
  exists_limit := ⟨⟨_, isLimitOfHasKernelOfPreservesLimit G f⟩⟩


/-- If the kernel comparison map for `G` at `f` is an isomorphism, then `G` preserves the
kernel of `f`.
-/
lemma PreservesKernel.of_iso_comparison [i : IsIso (kernelComparison f G)] :
    PreservesLimit (parallelPair f 0) G := by
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    G : CategoryTheory.Functor C D
    inst✝² : G.PreservesZeroMorphisms
    X Y : C
    f : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.HasKernel f
    inst✝ : CategoryTheory.Limits.HasKernel (G.map f)
    i : CategoryTheory.IsIso (CategoryTheory.Limits.kernelComparison f G)
    ⊢ CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPair f 0 …
  -/
  apply preservesLimit_of_preserves_limit_cone (kernelIsKernel f)
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    G : CategoryTheory.Functor C D
    inst✝² : G.PreservesZeroMorphisms
    X Y : C
    f : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.HasKernel f
    inst✝ : CategoryTheory.Limits.HasKernel (G.map f)
    i : CategoryTheory.IsIso (CategoryTheory.Limits.kernelComparison f G)
    ⊢ CategoryTheory.Limits.IsLimit (G.mapCone (CategoryTheory.Limits.Fork.ofι (Ca …
  -/
  apply (isLimitMapConeForkEquiv' G (kernel.condition f)).symm _
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    G : CategoryTheory.Functor C D
    inst✝² : G.PreservesZeroMorphisms
    X Y : C
    f : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.HasKernel f
    inst✝ : CategoryTheory.Limits.HasKernel (G.map f)
    i : CategoryTheory.IsIso (CategoryTheory.Limits.kernelComparison f G)
    ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι (G.map ( …
  -/
  exact @IsLimit.ofPointIso _ _ _ _ _ _ _ (kernelIsKernel (G.map f)) i
  /-
    🎉 no goals
  -/


/-- If `G` preserves the kernel of `f`, then the kernel comparison map for `G` at `f` is
an isomorphism.
-/
def PreservesKernel.iso : G.obj (kernel f) ≅ kernel (G.map f) :=
  IsLimit.conePointUniqueUpToIso (isLimitOfHasKernelOfPreservesLimit G f) (limit.isLimit _)


@[simp]
theorem PreservesKernel.iso_hom : (PreservesKernel.iso G f).hom = kernelComparison f G := by
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁶ : CategoryTheory.Limits.HasZeroMorphisms C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms D
    G : CategoryTheory.Functor C D
    inst✝³ : G.PreservesZeroMorphisms
    X Y : C
    f : Quiver.Hom X Y
    inst✝² : CategoryTheory.Limits.HasKernel f
    inst✝¹ : CategoryTheory.Limits.HasKernel (G.map f)
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPa …
    ⊢ Eq (CategoryTheory.Limits.PreservesKernel.iso G f).hom (CategoryTheory.Limit …
  -/
  rw [← cancel_mono (kernel.ι _)]
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁶ : CategoryTheory.Limits.HasZeroMorphisms C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms D
    G : CategoryTheory.Functor C D
    inst✝³ : G.PreservesZeroMorphisms
    X Y : C
    f : Quiver.Hom X Y
    inst✝² : CategoryTheory.Limits.HasKernel f
    inst✝¹ : CategoryTheory.Limits.HasKernel (G.map f)
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPa …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.PreservesKerne …
  -/
  simp [PreservesKernel.iso]
  /-
    🎉 no goals
  -/


instance : IsIso (kernelComparison f G) := by
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁶ : CategoryTheory.Limits.HasZeroMorphisms C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms D
    G : CategoryTheory.Functor C D
    inst✝³ : G.PreservesZeroMorphisms
    X Y Z : C
    f : Quiver.Hom X Y
    h : Quiver.Hom Z X
    w : Eq (CategoryTheory.CategoryStruct.comp h f) 0
    inst✝² : CategoryTheory.Limits.HasKernel f
    inst✝¹ : CategoryTheory.Limits.HasKernel (G.map f)
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPa …
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.kernelComparison f G)
  -/
  rw [← PreservesKernel.iso_hom]
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁶ : CategoryTheory.Limits.HasZeroMorphisms C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms D
    G : CategoryTheory.Functor C D
    inst✝³ : G.PreservesZeroMorphisms
    X Y Z : C
    f : Quiver.Hom X Y
    h : Quiver.Hom Z X
    w : Eq (CategoryTheory.CategoryStruct.comp h f) 0
    inst✝² : CategoryTheory.Limits.HasKernel f
    inst✝¹ : CategoryTheory.Limits.HasKernel (G.map f)
    inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPa …
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.PreservesKernel.iso G f).hom
  -/
  infer_instance
  /-
    🎉 no goals
  -/


@[reassoc]
theorem kernel_map_comp_preserves_kernel_iso_inv {X' Y' : C} (g : X' ⟶ Y') [HasKernel g]
    [HasKernel (G.map g)] [PreservesLimit (parallelPair g 0) G] (p : X ⟶ X') (q : Y ⟶ Y')
    (hpq : f ≫ q = p ≫ g) :
                                                           /-
                                                             C : Type u₁
                                                             inst✝¹⁰ : CategoryTheory.Category.{v₁, u₁} C
                                                             inst✝⁹ : CategoryTheory.Limits.HasZeroMorphisms C
                                                             D : Type u₂
                                                             inst✝⁸ : CategoryTheory.Category.{v₂, u₂} D
                                                             inst✝⁷ : CategoryTheory.Limits.HasZeroMorphisms D
                                                             G : CategoryTheory.Functor C D
                                                             inst✝⁶ : G.PreservesZeroMorphisms
                                                             X Y Z : C
                                                             f : Quiver.Hom X Y
                                                             h : Quiver.Hom Z X
                                                             w : Eq (CategoryTheory.CategoryStruct.comp h f) 0
                                                             inst✝⁵ : CategoryTheory.Limits.HasKernel f
                                                             inst✝⁴ : CategoryTheory.Limits.HasKernel (G.map f)
                                                             inst✝³ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelP …
                                                             X' Y' : C
                                                             g : Quiver.Hom X' Y'
                                                             inst✝² : CategoryTheory.Limits.HasKernel g
                                                             inst✝¹ : CategoryTheory.Limits.HasKernel (G.map g)
                                                             inst✝ : CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPa …
                                                             p : Quiver.Hom X X'
                                                             q : Quiver.Hom Y Y'
                                                             hpq : Eq (CategoryTheory.CategoryStruct.comp f q) (CategoryTheory.CategoryStru …
                                                             ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) (G.map q)) (CategoryTheory. …
                                                           -/
    kernel.map (G.map f) (G.map g) (G.map p) (G.map q) (by rw [← G.map_comp, hpq, G.map_comp]) ≫
                                                           /-
                                                             🎉 no goals
                                                           -/
        (PreservesKernel.iso G _).inv =
      (PreservesKernel.iso G _).inv ≫ G.map (kernel.map f g p q hpq) := by
  rw [Iso.comp_inv_eq, Category.assoc, PreservesKernel.iso_hom, Iso.eq_inv_comp,
    PreservesKernel.iso_hom, kernelComparison_comp_kernel_map]


@[reassoc (attr := simp)]
lemma map_condition : G.map f ≫ G.map c.π = 0 := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
    X Y : C
    f : Quiver.Hom X Y
    c : CategoryTheory.Limits.CokernelCofork f
    G : CategoryTheory.Functor C D
    inst✝ : G.PreservesZeroMorphisms
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) (G.map (CategoryTheory.Limi …
  -/
  rw [← G.map_comp, c.condition, G.map_zero]
  /-
    🎉 no goals
  -/


/-- A cokernel cofork for `f` is mapped to a cokernel cofork for `G.map f` if `G` is a functor
which preserves zero morphisms. -/
def map : CokernelCofork (G.map f) :=
  CokernelCofork.ofπ (G.map c.π) (c.map_condition G)


@[simp]
lemma map_π : (c.map G).π = G.map c.π := rfl


/-- The underlying cocone of a cokernel cofork is mapped to a colimit cocone if and only if
the mapped cokernel cofork is colimit. -/
def isColimitMapCoconeEquiv :
    IsColimit (G.mapCocone c) ≃ IsColimit (c.map G) := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
    X Y : C
    f : Quiver.Hom X Y
    c : CategoryTheory.Limits.CokernelCofork f
    G : CategoryTheory.Functor C D
    inst✝ : G.PreservesZeroMorphisms
    ⊢ Equiv (CategoryTheory.Limits.IsColimit (G.mapCocone c)) (CategoryTheory.Limi …
  -/
  refine (IsColimit.precomposeHomEquiv ?_ _).symm.trans (IsColimit.equivIsoColimit ?_)
  /-
    case refine_1
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
    X Y : C
    f : Quiver.Hom X Y
    c : CategoryTheory.Limits.CokernelCofork f
    G : CategoryTheory.Functor C D
    inst✝ : G.PreservesZeroMorphisms
    ⊢ CategoryTheory.Iso (CategoryTheory.Limits.parallelPair (G.map f) 0) ((Catego …
  -/
                                                              /-
                                                                🎉 no goals
                                                              -/
  refine parallelPair.ext (Iso.refl _) (Iso.refl _) ?_ ?_ <;> simp
                                                              /-
                                                                🎉 no goals
                                                              -/
  /-
    case refine_2
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
    X Y : C
    f : Quiver.Hom X Y
    c : CategoryTheory.Limits.CokernelCofork f
    G : CategoryTheory.Functor C D
    inst✝ : G.PreservesZeroMorphisms
    ⊢ CategoryTheory.Iso ((CategoryTheory.Limits.Cocones.precompose (CategoryTheor …
  -/
  exact Cocones.ext (Iso.refl _) (by rintro (_|_) <;> aesop_cat)
  /-
    🎉 no goals
  -/


/-- A colimit cokernel cofork is mapped to a colimit cokernel cofork by a functor `G`
when this functor preserves the corresponding colimit. -/
def mapIsColimit (hc : IsColimit c) (G : C ⥤ D)
    [Functor.PreservesZeroMorphisms G] [PreservesColimit (parallelPair f 0) G] :
    IsColimit (c.map G) :=
  c.isColimitMapCoconeEquiv G (isColimitOfPreserves G hc)


/-- The map of a cokernel cofork is a colimit iff
the cokernel cofork consisting of the mapped morphisms is a colimit.
This essentially lets us commute `CokernelCofork.ofπ` with `Functor.mapCocone`.

This is a variant of `isColimitMapCoconeCoforkEquiv` for equalizers,
which we can't use directly between `G.map 0 = 0` does not hold definitionally.
-/
def isColimitMapCoconeCoforkEquiv' :
    IsColimit (G.mapCocone (CokernelCofork.ofπ h w)) ≃
      IsColimit
                                          /-
                                            C : Type u₁
                                            inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
                                            inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
                                            D : Type u₂
                                            inst✝² : CategoryTheory.Category.{v₂, u₂} D
                                            inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
                                            G : CategoryTheory.Functor C D
                                            inst✝ : G.PreservesZeroMorphisms
                                            X Y Z : C
                                            f : Quiver.Hom X Y
                                            h : Quiver.Hom Y Z
                                            w : Eq (CategoryTheory.CategoryStruct.comp f h) 0
                                            ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) (G.map h)) 0
                                          -/
        (CokernelCofork.ofπ (G.map h) (by simp only [← G.map_comp, w, Functor.map_zero]) :
                                          /-
                                            🎉 no goals
                                          -/
          Cofork (G.map f) 0) :=
  CokernelCofork.isColimitMapCoconeEquiv _ _


/-- The property of preserving cokernels expressed in terms of cokernel coforks.

This is a variant of `isColimitCoforkMapOfIsColimit` for equalizers,
which we can't use directly between `G.map 0 = 0` does not hold definitionally.
-/
def isColimitCoforkMapOfIsColimit' [PreservesColimit (parallelPair f 0) G]
    (l : IsColimit (CokernelCofork.ofπ h w)) :
    IsColimit
                                        /-
                                          C : Type u₁
                                          inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
                                          inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms C
                                          D : Type u₂
                                          inst✝³ : CategoryTheory.Category.{v₂, u₂} D
                                          inst✝² : CategoryTheory.Limits.HasZeroMorphisms D
                                          G : CategoryTheory.Functor C D
                                          inst✝¹ : G.PreservesZeroMorphisms
                                          X Y Z : C
                                          f : Quiver.Hom X Y
                                          h : Quiver.Hom Y Z
                                          w : Eq (CategoryTheory.CategoryStruct.comp f h) 0
                                          inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallel …
                                          l : CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ  …
                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) (G.map h)) 0
                                        -/
      (CokernelCofork.ofπ (G.map h) (by simp only [← G.map_comp, w, Functor.map_zero]) :
                                        /-
                                          🎉 no goals
                                        -/
        Cofork (G.map f) 0) :=
  isColimitMapCoconeCoforkEquiv' G w (isColimitOfPreserves G l)


/--
If `G` preserves cokernels and `C` has them, then the cofork constructed of the mapped morphisms of
a cokernel cofork is a colimit.
-/
def isColimitOfHasCokernelOfPreservesColimit [PreservesColimit (parallelPair f 0) G] :
    IsColimit
      (Cofork.ofπ (G.map (cokernel.π f))
              /-
                C : Type u₁
                inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
                inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
                D : Type u₂
                inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
                inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
                G : CategoryTheory.Functor C D
                inst✝² : G.PreservesZeroMorphisms
                X Y Z : C
                f : Quiver.Hom X Y
                h : Quiver.Hom Y Z
                w : Eq (CategoryTheory.CategoryStruct.comp f h) 0
                inst✝¹ : CategoryTheory.Limits.HasCokernel f
                inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallel …
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) (G.map (CategoryTheory.Limi …
              -/
          (by simp only [← G.map_comp, cokernel.condition, zero_comp, Functor.map_zero]) :
              /-
                🎉 no goals
              -/
        Cofork (G.map f) 0) :=
  isColimitCoforkMapOfIsColimit' G (cokernel.condition f) (cokernelIsCokernel f)


instance [PreservesColimit (parallelPair f 0) G] : HasCokernel (G.map f) where
  exists_colimit := ⟨⟨_, isColimitOfHasCokernelOfPreservesColimit G f⟩⟩


/-- If the cokernel comparison map for `G` at `f` is an isomorphism, then `G` preserves the
cokernel of `f`.
-/
lemma PreservesCokernel.of_iso_comparison [i : IsIso (cokernelComparison f G)] :
    PreservesColimit (parallelPair f 0) G := by
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    G : CategoryTheory.Functor C D
    inst✝² : G.PreservesZeroMorphisms
    X Y : C
    f : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.HasCokernel f
    inst✝ : CategoryTheory.Limits.HasCokernel (G.map f)
    i : CategoryTheory.IsIso (CategoryTheory.Limits.cokernelComparison f G)
    ⊢ CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallelPair f …
  -/
  apply preservesColimit_of_preserves_colimit_cocone (cokernelIsCokernel f)
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    G : CategoryTheory.Functor C D
    inst✝² : G.PreservesZeroMorphisms
    X Y : C
    f : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.HasCokernel f
    inst✝ : CategoryTheory.Limits.HasCokernel (G.map f)
    i : CategoryTheory.IsIso (CategoryTheory.Limits.cokernelComparison f G)
    ⊢ CategoryTheory.Limits.IsColimit (G.mapCocone (CategoryTheory.Limits.Cofork.o …
  -/
  apply (isColimitMapCoconeCoforkEquiv' G (cokernel.condition f)).symm _
  /-
    C : Type u₁
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁵ : CategoryTheory.Limits.HasZeroMorphisms C
    D : Type u₂
    inst✝⁴ : CategoryTheory.Category.{v₂, u₂} D
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms D
    G : CategoryTheory.Functor C D
    inst✝² : G.PreservesZeroMorphisms
    X Y : C
    f : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.HasCokernel f
    inst✝ : CategoryTheory.Limits.HasCokernel (G.map f)
    i : CategoryTheory.IsIso (CategoryTheory.Limits.cokernelComparison f G)
    ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ (G …
  -/
  exact @IsColimit.ofPointIso _ _ _ _ _ _ _ (cokernelIsCokernel (G.map f)) i
  /-
    🎉 no goals
  -/


/-- If `G` preserves the cokernel of `f`, then the cokernel comparison map for `G` at `f` is
an isomorphism.
-/
def PreservesCokernel.iso : G.obj (cokernel f) ≅ cokernel (G.map f) :=
  IsColimit.coconePointUniqueUpToIso (isColimitOfHasCokernelOfPreservesColimit G f)
    (colimit.isColimit _)


@[simp]
theorem PreservesCokernel.iso_inv : (PreservesCokernel.iso G f).inv = cokernelComparison f G := by
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁶ : CategoryTheory.Limits.HasZeroMorphisms C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms D
    G : CategoryTheory.Functor C D
    inst✝³ : G.PreservesZeroMorphisms
    X Y : C
    f : Quiver.Hom X Y
    inst✝² : CategoryTheory.Limits.HasCokernel f
    inst✝¹ : CategoryTheory.Limits.HasCokernel (G.map f)
    inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallel …
    ⊢ Eq (CategoryTheory.Limits.PreservesCokernel.iso G f).inv (CategoryTheory.Lim …
  -/
  rw [← cancel_epi (cokernel.π _)]
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁶ : CategoryTheory.Limits.HasZeroMorphisms C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms D
    G : CategoryTheory.Functor C D
    inst✝³ : G.PreservesZeroMorphisms
    X Y : C
    f : Quiver.Hom X Y
    inst✝² : CategoryTheory.Limits.HasCokernel f
    inst✝¹ : CategoryTheory.Limits.HasCokernel (G.map f)
    inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallel …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.cokernel.π (G. …
  -/
  simp [PreservesCokernel.iso]
  /-
    🎉 no goals
  -/


instance : IsIso (cokernelComparison f G) := by
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁶ : CategoryTheory.Limits.HasZeroMorphisms C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms D
    G : CategoryTheory.Functor C D
    inst✝³ : G.PreservesZeroMorphisms
    X Y Z : C
    f : Quiver.Hom X Y
    h : Quiver.Hom Y Z
    w : Eq (CategoryTheory.CategoryStruct.comp f h) 0
    inst✝² : CategoryTheory.Limits.HasCokernel f
    inst✝¹ : CategoryTheory.Limits.HasCokernel (G.map f)
    inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallel …
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.cokernelComparison f G)
  -/
  rw [← PreservesCokernel.iso_inv]
  /-
    C : Type u₁
    inst✝⁷ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁶ : CategoryTheory.Limits.HasZeroMorphisms C
    D : Type u₂
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} D
    inst✝⁴ : CategoryTheory.Limits.HasZeroMorphisms D
    G : CategoryTheory.Functor C D
    inst✝³ : G.PreservesZeroMorphisms
    X Y Z : C
    f : Quiver.Hom X Y
    h : Quiver.Hom Y Z
    w : Eq (CategoryTheory.CategoryStruct.comp f h) 0
    inst✝² : CategoryTheory.Limits.HasCokernel f
    inst✝¹ : CategoryTheory.Limits.HasCokernel (G.map f)
    inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallel …
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.PreservesCokernel.iso G f).inv
  -/
  infer_instance
  /-
    🎉 no goals
  -/


@[reassoc]
theorem preserves_cokernel_iso_comp_cokernel_map {X' Y' : C} (g : X' ⟶ Y') [HasCokernel g]
    [HasCokernel (G.map g)] [PreservesColimit (parallelPair g 0) G] (p : X ⟶ X') (q : Y ⟶ Y')
    (hpq : f ≫ q = p ≫ g) :
    (PreservesCokernel.iso G _).hom ≫
        cokernel.map (G.map f) (G.map g) (G.map p) (G.map q)
              /-
                C : Type u₁
                inst✝¹⁰ : CategoryTheory.Category.{v₁, u₁} C
                inst✝⁹ : CategoryTheory.Limits.HasZeroMorphisms C
                D : Type u₂
                inst✝⁸ : CategoryTheory.Category.{v₂, u₂} D
                inst✝⁷ : CategoryTheory.Limits.HasZeroMorphisms D
                G : CategoryTheory.Functor C D
                inst✝⁶ : G.PreservesZeroMorphisms
                X Y Z : C
                f : Quiver.Hom X Y
                h : Quiver.Hom Y Z
                w : Eq (CategoryTheory.CategoryStruct.comp f h) 0
                inst✝⁵ : CategoryTheory.Limits.HasCokernel f
                inst✝⁴ : CategoryTheory.Limits.HasCokernel (G.map f)
                inst✝³ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.paralle …
                X' Y' : C
                g : Quiver.Hom X' Y'
                inst✝² : CategoryTheory.Limits.HasCokernel g
                inst✝¹ : CategoryTheory.Limits.HasCokernel (G.map g)
                inst✝ : CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallel …
                p : Quiver.Hom X X'
                q : Quiver.Hom Y Y'
                hpq : Eq (CategoryTheory.CategoryStruct.comp f q) (CategoryTheory.CategoryStru …
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (G.map f) (G.map q)) (CategoryTheory. …
              -/
          (by rw [← G.map_comp, hpq, G.map_comp]) =
              /-
                🎉 no goals
              -/
      G.map (cokernel.map f g p q hpq) ≫ (PreservesCokernel.iso G _).hom := by
  rw [← Iso.comp_inv_eq, Category.assoc, ← Iso.eq_inv_comp, PreservesCokernel.iso_inv,
    cokernel_map_comp_cokernelComparison, PreservesCokernel.iso_inv]


instance preservesKernel_zero :
    PreservesLimit (parallelPair (0 : X ⟶ Y) 0) G where
  preserves {c} hc := ⟨by
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
      X Y : C
      G : CategoryTheory.Functor C D
      inst✝ : G.PreservesZeroMorphisms
      c : CategoryTheory.Limits.Cone (CategoryTheory.Limits.parallelPair 0 0)
      hc : CategoryTheory.Limits.IsLimit c
      ⊢ CategoryTheory.Limits.IsLimit (G.mapCone c)
    -/
    have := KernelFork.IsLimit.isIso_ι c hc rfl
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
      X Y : C
      G : CategoryTheory.Functor C D
      inst✝ : G.PreservesZeroMorphisms
      c : CategoryTheory.Limits.Cone (CategoryTheory.Limits.parallelPair 0 0)
      hc : CategoryTheory.Limits.IsLimit c
      this : CategoryTheory.IsIso (CategoryTheory.Limits.Fork.ι c)
      ⊢ CategoryTheory.Limits.IsLimit (G.mapCone c)
    -/
    refine (KernelFork.isLimitMapConeEquiv c G).symm ?_
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
      X Y : C
      G : CategoryTheory.Functor C D
      inst✝ : G.PreservesZeroMorphisms
      c : CategoryTheory.Limits.Cone (CategoryTheory.Limits.parallelPair 0 0)
      hc : CategoryTheory.Limits.IsLimit c
      this : CategoryTheory.IsIso (CategoryTheory.Limits.Fork.ι c)
      ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.map c G)
    -/
    refine IsLimit.ofIsoLimit (KernelFork.IsLimit.ofId _ (G.map_zero _ _)) ?_
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
      X Y : C
      G : CategoryTheory.Functor C D
      inst✝ : G.PreservesZeroMorphisms
      c : CategoryTheory.Limits.Cone (CategoryTheory.Limits.parallelPair 0 0)
      hc : CategoryTheory.Limits.IsLimit c
      this : CategoryTheory.IsIso (CategoryTheory.Limits.Fork.ι c)
      ⊢ CategoryTheory.Iso (CategoryTheory.Limits.KernelFork.ofι (CategoryTheory.Cat …
    -/
    exact (Fork.ext (G.mapIso (asIso (Fork.ι c))).symm (by simp))⟩
    /-
      🎉 no goals
    -/


noncomputable instance preservesCokernel_zero :
    PreservesColimit (parallelPair (0 : X ⟶ Y) 0) G where
  preserves {c} hc := ⟨by
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
      X Y : C
      G : CategoryTheory.Functor C D
      inst✝ : G.PreservesZeroMorphisms
      c : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.parallelPair 0 0)
      hc : CategoryTheory.Limits.IsColimit c
      ⊢ CategoryTheory.Limits.IsColimit (G.mapCocone c)
    -/
    have := CokernelCofork.IsColimit.isIso_π c hc rfl
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
      X Y : C
      G : CategoryTheory.Functor C D
      inst✝ : G.PreservesZeroMorphisms
      c : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.parallelPair 0 0)
      hc : CategoryTheory.Limits.IsColimit c
      this : CategoryTheory.IsIso (CategoryTheory.Limits.Cofork.π c)
      ⊢ CategoryTheory.Limits.IsColimit (G.mapCocone c)
    -/
    refine (CokernelCofork.isColimitMapCoconeEquiv c G).symm ?_
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
      X Y : C
      G : CategoryTheory.Functor C D
      inst✝ : G.PreservesZeroMorphisms
      c : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.parallelPair 0 0)
      hc : CategoryTheory.Limits.IsColimit c
      this : CategoryTheory.IsIso (CategoryTheory.Limits.Cofork.π c)
      ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.map c G)
    -/
    refine IsColimit.ofIsoColimit (CokernelCofork.IsColimit.ofId _ (G.map_zero _ _)) ?_
    /-
      C : Type u₁
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      D : Type u₂
      inst✝² : CategoryTheory.Category.{v₂, u₂} D
      inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
      X Y : C
      G : CategoryTheory.Functor C D
      inst✝ : G.PreservesZeroMorphisms
      c : CategoryTheory.Limits.Cocone (CategoryTheory.Limits.parallelPair 0 0)
      hc : CategoryTheory.Limits.IsColimit c
      this : CategoryTheory.IsIso (CategoryTheory.Limits.Cofork.π c)
      ⊢ CategoryTheory.Iso (CategoryTheory.Limits.CokernelCofork.ofπ (CategoryTheory …
    -/
    exact (Cofork.ext (G.mapIso (asIso (Cofork.π c))) (by simp))⟩
    /-
      🎉 no goals
    -/


/-- The kernel of a zero map is preserved by any functor which preserves zero morphisms. -/
lemma preservesKernel_zero' (f : X ⟶ Y) (hf : f = 0) :
    PreservesLimit (parallelPair f 0) G := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
    X Y : C
    G : CategoryTheory.Functor C D
    inst✝ : G.PreservesZeroMorphisms
    f : Quiver.Hom X Y
    hf : Eq f 0
    ⊢ CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPair f 0 …
  -/
  rw [hf]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
    X Y : C
    G : CategoryTheory.Functor C D
    inst✝ : G.PreservesZeroMorphisms
    f : Quiver.Hom X Y
    hf : Eq f 0
    ⊢ CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPair 0 0 …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The cokernel of a zero map is preserved by any functor which preserves zero morphisms. -/
lemma preservesCokernel_zero' (f : X ⟶ Y) (hf : f = 0) :
    PreservesColimit (parallelPair f 0) G := by
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
    X Y : C
    G : CategoryTheory.Functor C D
    inst✝ : G.PreservesZeroMorphisms
    f : Quiver.Hom X Y
    hf : Eq f 0
    ⊢ CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallelPair f …
  -/
  rw [hf]
  /-
    C : Type u₁
    inst✝⁴ : CategoryTheory.Category.{v₁, u₁} C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    D : Type u₂
    inst✝² : CategoryTheory.Category.{v₂, u₂} D
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
    X Y : C
    G : CategoryTheory.Functor C D
    inst✝ : G.PreservesZeroMorphisms
    f : Quiver.Hom X Y
    hf : Eq f 0
    ⊢ CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallelPair 0 …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


