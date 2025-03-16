theorem exact_iff_epi_imageToKernel' : S.Exact ↔ Epi (imageToKernel' S.f S.g S.zero) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex C
    ⊢ Iff S.Exact (CategoryTheory.Epi (imageToKernel' S.f S.g ⋯))
  -/
  rw [S.exact_iff_epi_kernel_lift]
  have : factorThruImage S.f ≫ imageToKernel' S.f S.g S.zero = kernel.lift S.g S.f S.zero := by
    simp only [← cancel_mono (kernel.ι _), kernel.lift_ι, imageToKernel',
      Category.assoc, image.fac]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex C
    this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.factorThr …
    ⊢ Iff (CategoryTheory.Epi (CategoryTheory.Limits.kernel.lift S.g S.f ⋯)) (Cate …
  -/
  constructor
    /-
      case mp
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Abelian C
      S : CategoryTheory.ShortComplex C
      this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.factorThr …
      ⊢ CategoryTheory.Epi (CategoryTheory.Limits.kernel.lift S.g S.f ⋯) → CategoryT …
    -/
  · intro
    /-
      case mp
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Abelian C
      S : CategoryTheory.ShortComplex C
      this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.factorThr …
      a✝ : CategoryTheory.Epi (CategoryTheory.Limits.kernel.lift S.g S.f ⋯)
      ⊢ CategoryTheory.Epi (imageToKernel' S.f S.g ⋯)
    -/
    exact epi_of_epi_fac this
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Abelian C
      S : CategoryTheory.ShortComplex C
      this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.factorThr …
      ⊢ CategoryTheory.Epi (imageToKernel' S.f S.g ⋯) → CategoryTheory.Epi (Category …
    -/
  · intro
    /-
      case mpr
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Abelian C
      S : CategoryTheory.ShortComplex C
      this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.factorThr …
      a✝ : CategoryTheory.Epi (imageToKernel' S.f S.g ⋯)
      ⊢ CategoryTheory.Epi (CategoryTheory.Limits.kernel.lift S.g S.f ⋯)
    -/
    rw [← this]
    /-
      case mpr
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Abelian C
      S : CategoryTheory.ShortComplex C
      this : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.factorThr …
      a✝ : CategoryTheory.Epi (imageToKernel' S.f S.g ⋯)
      ⊢ CategoryTheory.Epi (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limit …
    -/
    apply epi_comp
    /-
      🎉 no goals
    -/


theorem exact_iff_epi_imageToKernel : S.Exact ↔ Epi (imageToKernel S.f S.g S.zero) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex C
    ⊢ Iff S.Exact (CategoryTheory.Epi (imageToKernel S.f S.g ⋯))
  -/
  rw [S.exact_iff_epi_imageToKernel']
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex C
    ⊢ Iff (CategoryTheory.Epi (imageToKernel' S.f S.g ⋯)) (CategoryTheory.Epi (ima …
  -/
  apply (MorphismProperty.epimorphisms C).arrow_mk_iso_iff
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex C
    ⊢ CategoryTheory.Iso (CategoryTheory.Arrow.mk (imageToKernel' S.f S.g ⋯)) (Cat …
  -/
  exact Arrow.isoMk (imageSubobjectIso S.f).symm (kernelSubobjectIso S.g).symm
  /-
    🎉 no goals
  -/


theorem exact_iff_isIso_imageToKernel : S.Exact ↔ IsIso (imageToKernel S.f S.g S.zero) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex C
    ⊢ Iff S.Exact (CategoryTheory.IsIso (imageToKernel S.f S.g ⋯))
  -/
  rw [S.exact_iff_epi_imageToKernel]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex C
    ⊢ Iff (CategoryTheory.Epi (imageToKernel S.f S.g ⋯)) (CategoryTheory.IsIso (im …
  -/
  constructor
    /-
      case mp
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Abelian C
      S : CategoryTheory.ShortComplex C
      ⊢ CategoryTheory.Epi (imageToKernel S.f S.g ⋯) → CategoryTheory.IsIso (imageTo …
    -/
  · intro
    /-
      case mp
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Abelian C
      S : CategoryTheory.ShortComplex C
      a✝ : CategoryTheory.Epi (imageToKernel S.f S.g ⋯)
      ⊢ CategoryTheory.IsIso (imageToKernel S.f S.g ⋯)
    -/
    apply isIso_of_mono_of_epi
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Abelian C
      S : CategoryTheory.ShortComplex C
      ⊢ CategoryTheory.IsIso (imageToKernel S.f S.g ⋯) → CategoryTheory.Epi (imageTo …
    -/
  · intro
    /-
      case mpr
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Abelian C
      S : CategoryTheory.ShortComplex C
      a✝ : CategoryTheory.IsIso (imageToKernel S.f S.g ⋯)
      ⊢ CategoryTheory.Epi (imageToKernel S.f S.g ⋯)
    -/
    infer_instance
    /-
      🎉 no goals
    -/


/-- In an abelian category, a short complex `S` is exact
iff `imageSubobject S.f = kernelSubobject S.g`.
-/
theorem exact_iff_image_eq_kernel : S.Exact ↔ imageSubobject S.f = kernelSubobject S.g := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex C
    ⊢ Iff S.Exact (Eq (CategoryTheory.Limits.imageSubobject S.f) (CategoryTheory.L …
  -/
  rw [exact_iff_isIso_imageToKernel]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex C
    ⊢ Iff (CategoryTheory.IsIso (imageToKernel S.f S.g ⋯)) (Eq (CategoryTheory.Lim …
  -/
  constructor
    /-
      case mp
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Abelian C
      S : CategoryTheory.ShortComplex C
      ⊢ CategoryTheory.IsIso (imageToKernel S.f S.g ⋯) → Eq (CategoryTheory.Limits.i …
    -/
  · intro
    /-
      case mp
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Abelian C
      S : CategoryTheory.ShortComplex C
      a✝ : CategoryTheory.IsIso (imageToKernel S.f S.g ⋯)
      ⊢ Eq (CategoryTheory.Limits.imageSubobject S.f) (CategoryTheory.Limits.kernelS …
    -/
    exact Subobject.eq_of_comm (asIso (imageToKernel _ _ S.zero)) (by simp)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Abelian C
      S : CategoryTheory.ShortComplex C
      ⊢ Eq (CategoryTheory.Limits.imageSubobject S.f) (CategoryTheory.Limits.kernelS …
    -/
  · intro h
    /-
      case mpr
      C : Type u₁
      inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
      inst✝ : CategoryTheory.Abelian C
      S : CategoryTheory.ShortComplex C
      h : Eq (CategoryTheory.Limits.imageSubobject S.f) (CategoryTheory.Limits.kerne …
      ⊢ CategoryTheory.IsIso (imageToKernel S.f S.g ⋯)
    -/
    exact ⟨Subobject.ofLE _ _ h.ge, by ext; simp, by ext; simp⟩
    /-
      🎉 no goals
    -/


theorem exact_iff_of_forks {cg : KernelFork S.g} (hg : IsLimit cg) {cf : CokernelCofork S.f}
    (hf : IsColimit cf) : S.Exact ↔ cg.ι ≫ cf.π = 0 := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex C
    cg : CategoryTheory.Limits.KernelFork S.g
    hg : CategoryTheory.Limits.IsLimit cg
    cf : CategoryTheory.Limits.CokernelCofork S.f
    hf : CategoryTheory.Limits.IsColimit cf
    ⊢ Iff S.Exact (Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.F …
  -/
  rw [exact_iff_kernel_ι_comp_cokernel_π_zero]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex C
    cg : CategoryTheory.Limits.KernelFork S.g
    hg : CategoryTheory.Limits.IsLimit cg
    cf : CategoryTheory.Limits.CokernelCofork S.f
    hf : CategoryTheory.Limits.IsColimit cf
    ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernel.ι  …
  -/
  let e₁ := IsLimit.conePointUniqueUpToIso (kernelIsKernel S.g) hg
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex C
    cg : CategoryTheory.Limits.KernelFork S.g
    hg : CategoryTheory.Limits.IsLimit cg
    cf : CategoryTheory.Limits.CokernelCofork S.f
    hf : CategoryTheory.Limits.IsColimit cf
    e₁ : CategoryTheory.Iso (CategoryTheory.Limits.Fork.ofι (CategoryTheory.Limits …
    ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernel.ι  …
  -/
  let e₂ := IsColimit.coconePointUniqueUpToIso (cokernelIsCokernel S.f) hf
  have : cg.ι ≫ cf.π = e₁.inv ≫ kernel.ι S.g ≫ cokernel.π S.f ≫ e₂.hom := by
    have eq₁ := IsLimit.conePointUniqueUpToIso_inv_comp (kernelIsKernel S.g) hg (.zero)
    have eq₂ := IsColimit.comp_coconePointUniqueUpToIso_hom (cokernelIsCokernel S.f) hf (.one)
    dsimp at eq₁ eq₂
    rw [← eq₁, ← eq₂, Category.assoc]
  rw [this, IsIso.comp_left_eq_zero e₁.inv, ← Category.assoc,
    IsIso.comp_right_eq_zero _ e₂.hom]


/-- If `(f, g)` is exact, then `Abelian.image.ι S.f` is a kernel of `S.g`. -/
def Exact.isLimitImage (h : S.Exact) :
    IsLimit (KernelFork.ofι (Abelian.image.ι S.f)
      (Abelian.image_ι_comp_eq_zero S.zero) : KernelFork S.g) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex C
    h : S.Exact
    ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι (Categor …
  -/
  rw [exact_iff_kernel_ι_comp_cokernel_π_zero] at h
  exact KernelFork.IsLimit.ofι _ _
    (fun u hu ↦ kernel.lift (cokernel.π S.f) u
      (by rw [← kernel.lift_ι S.g u hu, Category.assoc, h, comp_zero])) (by aesop_cat)
    (fun _ _ _ hm => by rw [← cancel_mono (Abelian.image.ι S.f), hm, kernel.lift_ι])


/-- If `(f, g)` is exact, then `image.ι f` is a kernel of `g`. -/
def Exact.isLimitImage' (h : S.Exact) :
    IsLimit (KernelFork.ofι (Limits.image.ι S.f)
      (image_ι_comp_eq_zero S.zero) : KernelFork S.g) :=
  IsKernel.isoKernel _ _ h.isLimitImage (Abelian.imageIsoImage S.f).symm <| IsImage.lift_fac _ _


/-- If `(f, g)` is exact, then `Abelian.coimage.π g` is a cokernel of `f`. -/
def Exact.isColimitCoimage (h : S.Exact) :
    IsColimit
      (CokernelCofork.ofπ (Abelian.coimage.π S.g) (Abelian.comp_coimage_π_eq_zero S.zero) :
        CokernelCofork S.f) := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex C
    h : S.Exact
    ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ (C …
  -/
  rw [exact_iff_kernel_ι_comp_cokernel_π_zero] at h
  refine CokernelCofork.IsColimit.ofπ _ _
    (fun u hu => cokernel.desc (kernel.ι S.g) u
      (by rw [← cokernel.π_desc S.f u hu, ← Category.assoc, h, zero_comp]))
    (by aesop_cat) ?_
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex C
    h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernel.ι S.g …
    ⊢ ∀ {Z' : C} (g' : Quiver.Hom S.X₂ Z') (eq' : Eq (CategoryTheory.CategoryStruc …
  -/
  intros _ _ _ _ hm
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex C
    h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernel.ι S.g …
    Z'✝ : C
    g'✝ : Quiver.Hom S.X₂ Z'✝
    eq'✝ : Eq (CategoryTheory.CategoryStruct.comp S.f g'✝) 0
    m✝ : Quiver.Hom (CategoryTheory.Abelian.coimage S.g) Z'✝
    hm : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Abelian.coimage.π  …
    ⊢ Eq m✝ ((fun {Z'} u hu => CategoryTheory.Limits.cokernel.desc (CategoryTheory …
  -/
  ext
  /-
    case h
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex C
    h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernel.ι S.g …
    Z'✝ : C
    g'✝ : Quiver.Hom S.X₂ Z'✝
    eq'✝ : Eq (CategoryTheory.CategoryStruct.comp S.f g'✝) 0
    m✝ : Quiver.Hom (CategoryTheory.Abelian.coimage S.g) Z'✝
    hm : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Abelian.coimage.π  …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.coequalizer.π  …
  -/
  rw [hm, cokernel.π_desc]
  /-
    🎉 no goals
  -/


/-- If `(f, g)` is exact, then `factorThruImage g` is a cokernel of `f`. -/
def Exact.isColimitImage (h : S.Exact) :
    IsColimit (CokernelCofork.ofπ (Limits.factorThruImage S.g)
        (comp_factorThruImage_eq_zero S.zero)) :=
  IsCokernel.cokernelIso _ _ h.isColimitCoimage (Abelian.coimageIsoImage' S.g) <|
                                               /-
                                                 C : Type u₁
                                                 inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                                 inst✝ : CategoryTheory.Abelian C
                                                 S : CategoryTheory.ShortComplex C
                                                 h : S.Exact
                                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
                                               -/
    (cancel_mono (Limits.image.ι S.g)).1 <| by simp
                                               /-
                                                 🎉 no goals
                                               -/


theorem exact_kernel {X Y : C} (f : X ⟶ Y) :
                                        /-
                                          C : Type u₁
                                          inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                          inst✝ : CategoryTheory.Abelian C
                                          S : CategoryTheory.ShortComplex C
                                          X Y : C
                                          f : Quiver.Hom X Y
                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernel.ι f) f) 0
                                        -/
    (ShortComplex.mk (kernel.ι f) f (by simp)).Exact :=
                                        /-
                                          🎉 no goals
                                        -/
  exact_of_f_is_kernel _ (kernelIsKernel f)


theorem exact_cokernel {X Y : C} (f : X ⟶ Y) :
                                          /-
                                            C : Type u₁
                                            inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
                                            inst✝ : CategoryTheory.Abelian C
                                            S : CategoryTheory.ShortComplex C
                                            X Y : C
                                            f : Quiver.Hom X Y
                                            ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.cokernel.π f …
                                          -/
    (ShortComplex.mk f (cokernel.π f) (by simp)).Exact :=
                                          /-
                                            🎉 no goals
                                          -/
  exact_of_g_is_cokernel _ (cokernelIsCokernel f)


theorem exact_iff_exact_image_ι :
    S.Exact ↔ (ShortComplex.mk (Abelian.image.ι S.f) S.g
      (Abelian.image_ι_comp_eq_zero S.zero)).Exact :=
  ShortComplex.exact_iff_of_epi_of_isIso_of_mono
    { τ₁ := Abelian.factorThruImage S.f
      τ₂ := 𝟙 _
      τ₃ := 𝟙 _ }


theorem exact_iff_exact_coimage_π :
    S.Exact ↔ (ShortComplex.mk S.f (Abelian.coimage.π S.g)
      (Abelian.comp_coimage_π_eq_zero S.zero)).Exact := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Abelian C
    S : CategoryTheory.ShortComplex C
    ⊢ Iff S.Exact (CategoryTheory.ShortComplex.mk S.f (CategoryTheory.Abelian.coim …
  -/
  symm
  exact ShortComplex.exact_iff_of_epi_of_isIso_of_mono
    { τ₁ := 𝟙 _
      τ₂ := 𝟙 _
      τ₃ := Abelian.factorThruCoimage S.g }


open List in
theorem Abelian.tfae_mono {X Y : C} (f : X ⟶ Y) (Z : C) :
    TFAE [Mono f, kernel.ι f = 0, (ShortComplex.mk (0 : Z ⟶ X) f zero_comp).Exact] := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Abelian C
    X Y : C
    f : Quiver.Hom X Y
    Z : C
    ⊢ (List.cons (CategoryTheory.Mono f) (List.cons (Eq (CategoryTheory.Limits.ker …
  -/
  tfae_have 2 → 1 := mono_of_kernel_ι_eq_zero _
  tfae_have 1 → 2
  | _ => by rw [← cancel_mono f, kernel.condition, zero_comp]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Abelian C
    X Y : C
    f : Quiver.Hom X Y
    Z : C
    tfae_2_to_1 : Eq (CategoryTheory.Limits.kernel.ι f) 0 → CategoryTheory.Mono f
    tfae_1_to_2 : CategoryTheory.Mono f → Eq (CategoryTheory.Limits.kernel.ι f) 0
    ⊢ (List.cons (CategoryTheory.Mono f) (List.cons (Eq (CategoryTheory.Limits.ker …
  -/
  tfae_have 3 ↔ 1 := ShortComplex.exact_iff_mono _ (by simp)
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Abelian C
    X Y : C
    f : Quiver.Hom X Y
    Z : C
    tfae_2_to_1 : Eq (CategoryTheory.Limits.kernel.ι f) 0 → CategoryTheory.Mono f
    tfae_1_to_2 : CategoryTheory.Mono f → Eq (CategoryTheory.Limits.kernel.ι f) 0
    tfae_3_iff_1 : Iff (CategoryTheory.ShortComplex.mk 0 f ⋯).Exact (CategoryTheor …
    ⊢ (List.cons (CategoryTheory.Mono f) (List.cons (Eq (CategoryTheory.Limits.ker …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


open List in
theorem Abelian.tfae_epi {X Y : C} (f : X ⟶ Y) (Z : C ) :
    TFAE [Epi f, cokernel.π f = 0, (ShortComplex.mk f (0 : Y ⟶ Z) comp_zero).Exact] := by
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Abelian C
    X Y : C
    f : Quiver.Hom X Y
    Z : C
    ⊢ (List.cons (CategoryTheory.Epi f) (List.cons (Eq (CategoryTheory.Limits.coke …
  -/
  tfae_have 2 → 1 := epi_of_cokernel_π_eq_zero _
  tfae_have 1 → 2
  | _ => by rw [← cancel_epi f, cokernel.condition, comp_zero]
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Abelian C
    X Y : C
    f : Quiver.Hom X Y
    Z : C
    tfae_2_to_1 : Eq (CategoryTheory.Limits.cokernel.π f) 0 → CategoryTheory.Epi f
    tfae_1_to_2 : CategoryTheory.Epi f → Eq (CategoryTheory.Limits.cokernel.π f) 0
    ⊢ (List.cons (CategoryTheory.Epi f) (List.cons (Eq (CategoryTheory.Limits.coke …
  -/
  tfae_have 3 ↔ 1 := ShortComplex.exact_iff_epi _ (by simp)
  /-
    C : Type u₁
    inst✝¹ : CategoryTheory.Category.{v₁, u₁} C
    inst✝ : CategoryTheory.Abelian C
    X Y : C
    f : Quiver.Hom X Y
    Z : C
    tfae_2_to_1 : Eq (CategoryTheory.Limits.cokernel.π f) 0 → CategoryTheory.Epi f
    tfae_1_to_2 : CategoryTheory.Epi f → Eq (CategoryTheory.Limits.cokernel.π f) 0
    tfae_3_iff_1 : Iff (CategoryTheory.ShortComplex.mk f 0 ⋯).Exact (CategoryTheor …
    ⊢ (List.cons (CategoryTheory.Epi f) (List.cons (Eq (CategoryTheory.Limits.coke …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


lemma reflects_exact_of_faithful [F.Faithful] (S : ShortComplex C) (hS : (S.map F).Exact) :
    S.Exact := by
  /-
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁴ : CategoryTheory.Abelian C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    inst✝² : CategoryTheory.Abelian D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.PreservesZeroMorphisms
    inst✝ : F.Faithful
    S : CategoryTheory.ShortComplex C
    hS : (S.map F).Exact
    ⊢ S.Exact
  -/
  rw [ShortComplex.exact_iff_kernel_ι_comp_cokernel_π_zero] at hS ⊢
  /-
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁴ : CategoryTheory.Abelian C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    inst✝² : CategoryTheory.Abelian D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.PreservesZeroMorphisms
    inst✝ : F.Faithful
    S : CategoryTheory.ShortComplex C
    hS : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernel.ι (S …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernel.ι S.g)  …
  -/
  dsimp at hS
  /-
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁴ : CategoryTheory.Abelian C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    inst✝² : CategoryTheory.Abelian D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.PreservesZeroMorphisms
    inst✝ : F.Faithful
    S : CategoryTheory.ShortComplex C
    hS : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernel.ι (F …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernel.ι S.g)  …
  -/
  apply F.zero_of_map_zero
  obtain ⟨k, hk⟩ :=
    kernel.lift' (F.map S.g) (F.map (kernel.ι S.g))
      (by simp only [← F.map_comp, kernel.condition, CategoryTheory.Functor.map_zero])
  obtain ⟨l, hl⟩ :=
    cokernel.desc' (F.map S.f) (F.map (cokernel.π S.f))
      (by simp only [← F.map_comp, cokernel.condition, CategoryTheory.Functor.map_zero])
  /-
    case h.mk.mk
    C : Type u₁
    inst✝⁵ : CategoryTheory.Category.{v₁, u₁} C
    inst✝⁴ : CategoryTheory.Abelian C
    D : Type u₂
    inst✝³ : CategoryTheory.Category.{v₂, u₂} D
    inst✝² : CategoryTheory.Abelian D
    F : CategoryTheory.Functor C D
    inst✝¹ : F.PreservesZeroMorphisms
    inst✝ : F.Faithful
    S : CategoryTheory.ShortComplex C
    hS : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernel.ι (F …
    k : Quiver.Hom (F.obj (CategoryTheory.Limits.kernel S.g)) (CategoryTheory.Limi …
    hk : Eq (CategoryTheory.CategoryStruct.comp k (CategoryTheory.Limits.kernel.ι  …
    l : Quiver.Hom (CategoryTheory.Limits.cokernel (F.map S.f)) (F.obj (CategoryTh …
    hl : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.cokernel.π  …
    ⊢ Eq (F.map (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernel. …
  -/
  rw [F.map_comp, ← hl, ← hk, Category.assoc, reassoc_of% hS, zero_comp, comp_zero]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-09")] alias CategoryTheory.Functor.map_exact :=
  ShortComplex.Exact.map


/-- A functor which preserves exactness preserves monomorphisms. -/
theorem preservesMonomorphisms_of_map_exact : L.PreservesMonomorphisms where
  preserves f hf := by
    /-
      A : Type u₁
      B : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} A
      inst✝³ : CategoryTheory.Category.{v₂, u₂} B
      inst✝² : CategoryTheory.Abelian A
      inst✝¹ : CategoryTheory.Abelian B
      L : CategoryTheory.Functor A B
      inst✝ : L.PreservesZeroMorphisms
      hL : ∀ (S : CategoryTheory.ShortComplex A), S.Exact → (S.map L).Exact
      X✝ Y✝ : A
      f : Quiver.Hom X✝ Y✝
      hf : CategoryTheory.Mono f
      ⊢ CategoryTheory.Mono (L.map f)
    -/
    apply ((Abelian.tfae_mono (L.map f) (L.obj 0)).out 2 0).mp
    /-
      A : Type u₁
      B : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} A
      inst✝³ : CategoryTheory.Category.{v₂, u₂} B
      inst✝² : CategoryTheory.Abelian A
      inst✝¹ : CategoryTheory.Abelian B
      L : CategoryTheory.Functor A B
      inst✝ : L.PreservesZeroMorphisms
      hL : ∀ (S : CategoryTheory.ShortComplex A), S.Exact → (S.map L).Exact
      X✝ Y✝ : A
      f : Quiver.Hom X✝ Y✝
      hf : CategoryTheory.Mono f
      ⊢ (CategoryTheory.ShortComplex.mk 0 (L.map f) ⋯).Exact
    -/
    refine ShortComplex.exact_of_iso ?_ (hL _ (((tfae_mono f 0).out 0 2).mp hf))
    /-
      A : Type u₁
      B : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} A
      inst✝³ : CategoryTheory.Category.{v₂, u₂} B
      inst✝² : CategoryTheory.Abelian A
      inst✝¹ : CategoryTheory.Abelian B
      L : CategoryTheory.Functor A B
      inst✝ : L.PreservesZeroMorphisms
      hL : ∀ (S : CategoryTheory.ShortComplex A), S.Exact → (S.map L).Exact
      X✝ Y✝ : A
      f : Quiver.Hom X✝ Y✝
      hf : CategoryTheory.Mono f
      ⊢ CategoryTheory.Iso ((CategoryTheory.ShortComplex.mk 0 f ⋯).map L) (CategoryT …
    -/
    exact ShortComplex.isoMk (Iso.refl _) (Iso.refl _) (Iso.refl _)
    /-
      🎉 no goals
    -/


/-- A functor which preserves exactness preserves epimorphisms. -/
theorem preservesEpimorphisms_of_map_exact : L.PreservesEpimorphisms where
  preserves f hf := by
    /-
      A : Type u₁
      B : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} A
      inst✝³ : CategoryTheory.Category.{v₂, u₂} B
      inst✝² : CategoryTheory.Abelian A
      inst✝¹ : CategoryTheory.Abelian B
      L : CategoryTheory.Functor A B
      inst✝ : L.PreservesZeroMorphisms
      hL : ∀ (S : CategoryTheory.ShortComplex A), S.Exact → (S.map L).Exact
      X✝ Y✝ : A
      f : Quiver.Hom X✝ Y✝
      hf : CategoryTheory.Epi f
      ⊢ CategoryTheory.Epi (L.map f)
    -/
    apply ((Abelian.tfae_epi (L.map f) (L.obj 0)).out 2 0).mp
    /-
      A : Type u₁
      B : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} A
      inst✝³ : CategoryTheory.Category.{v₂, u₂} B
      inst✝² : CategoryTheory.Abelian A
      inst✝¹ : CategoryTheory.Abelian B
      L : CategoryTheory.Functor A B
      inst✝ : L.PreservesZeroMorphisms
      hL : ∀ (S : CategoryTheory.ShortComplex A), S.Exact → (S.map L).Exact
      X✝ Y✝ : A
      f : Quiver.Hom X✝ Y✝
      hf : CategoryTheory.Epi f
      ⊢ (CategoryTheory.ShortComplex.mk (L.map f) 0 ⋯).Exact
    -/
    refine ShortComplex.exact_of_iso ?_ (hL _ (((tfae_epi f 0).out 0 2).mp hf))
    /-
      A : Type u₁
      B : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} A
      inst✝³ : CategoryTheory.Category.{v₂, u₂} B
      inst✝² : CategoryTheory.Abelian A
      inst✝¹ : CategoryTheory.Abelian B
      L : CategoryTheory.Functor A B
      inst✝ : L.PreservesZeroMorphisms
      hL : ∀ (S : CategoryTheory.ShortComplex A), S.Exact → (S.map L).Exact
      X✝ Y✝ : A
      f : Quiver.Hom X✝ Y✝
      hf : CategoryTheory.Epi f
      ⊢ CategoryTheory.Iso ((CategoryTheory.ShortComplex.mk f 0 ⋯).map L) (CategoryT …
    -/
    exact ShortComplex.isoMk (Iso.refl _) (Iso.refl _) (Iso.refl _)
    /-
      🎉 no goals
    -/


/-- A functor which preserves the exactness of short complexes preserves homology. -/
lemma preservesHomology_of_map_exact : L.PreservesHomology where
  preservesCokernels X Y f := by
    /-
      A : Type u₁
      B : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} A
      inst✝³ : CategoryTheory.Category.{v₂, u₂} B
      inst✝² : CategoryTheory.Abelian A
      inst✝¹ : CategoryTheory.Abelian B
      L : CategoryTheory.Functor A B
      inst✝ : L.PreservesZeroMorphisms
      hL : ∀ (S : CategoryTheory.ShortComplex A), S.Exact → (S.map L).Exact
      X Y : A
      f : Quiver.Hom X Y
      ⊢ CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallelPair f …
    -/
    have := preservesEpimorphisms_of_map_exact _ hL
    /-
      A : Type u₁
      B : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} A
      inst✝³ : CategoryTheory.Category.{v₂, u₂} B
      inst✝² : CategoryTheory.Abelian A
      inst✝¹ : CategoryTheory.Abelian B
      L : CategoryTheory.Functor A B
      inst✝ : L.PreservesZeroMorphisms
      hL : ∀ (S : CategoryTheory.ShortComplex A), S.Exact → (S.map L).Exact
      X Y : A
      f : Quiver.Hom X Y
      this : L.PreservesEpimorphisms
      ⊢ CategoryTheory.Limits.PreservesColimit (CategoryTheory.Limits.parallelPair f …
    -/
    apply preservesColimit_of_preserves_colimit_cocone (cokernelIsCokernel f)
    /-
      A : Type u₁
      B : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} A
      inst✝³ : CategoryTheory.Category.{v₂, u₂} B
      inst✝² : CategoryTheory.Abelian A
      inst✝¹ : CategoryTheory.Abelian B
      L : CategoryTheory.Functor A B
      inst✝ : L.PreservesZeroMorphisms
      hL : ∀ (S : CategoryTheory.ShortComplex A), S.Exact → (S.map L).Exact
      X Y : A
      f : Quiver.Hom X Y
      this : L.PreservesEpimorphisms
      ⊢ CategoryTheory.Limits.IsColimit (L.mapCocone (CategoryTheory.Limits.Cofork.o …
    -/
    apply (CokernelCofork.isColimitMapCoconeEquiv _ L).2
    have : Epi ((ShortComplex.mk _ _ (cokernel.condition f)).map L).g := by
      dsimp
      infer_instance
    /-
      A : Type u₁
      B : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} A
      inst✝³ : CategoryTheory.Category.{v₂, u₂} B
      inst✝² : CategoryTheory.Abelian A
      inst✝¹ : CategoryTheory.Abelian B
      L : CategoryTheory.Functor A B
      inst✝ : L.PreservesZeroMorphisms
      hL : ∀ (S : CategoryTheory.ShortComplex A), S.Exact → (S.map L).Exact
      X Y : A
      f : Quiver.Hom X Y
      ⊢ CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPair f 0 …
    -/
    exact (hL (ShortComplex.mk _ _ (cokernel.condition f))
    /-
      A : Type u₁
      B : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} A
      inst✝³ : CategoryTheory.Category.{v₂, u₂} B
      inst✝² : CategoryTheory.Abelian A
      inst✝¹ : CategoryTheory.Abelian B
      L : CategoryTheory.Functor A B
      inst✝ : L.PreservesZeroMorphisms
      hL : ∀ (S : CategoryTheory.ShortComplex A), S.Exact → (S.map L).Exact
      X Y : A
      f : Quiver.Hom X Y
      this : L.PreservesMonomorphisms
      ⊢ CategoryTheory.Limits.PreservesLimit (CategoryTheory.Limits.parallelPair f 0 …
    -/
      (ShortComplex.exact_of_g_is_cokernel _ (cokernelIsCokernel f))).gIsCokernel
    /-
      A : Type u₁
      B : Type u₂
      inst✝⁴ : CategoryTheory.Category.{v₁, u₁} A
      inst✝³ : CategoryTheory.Category.{v₂, u₂} B
      inst✝² : CategoryTheory.Abelian A
      inst✝¹ : CategoryTheory.Abelian B
      L : CategoryTheory.Functor A B
      inst✝ : L.PreservesZeroMorphisms
      hL : ∀ (S : CategoryTheory.ShortComplex A), S.Exact → (S.map L).Exact
      X Y : A
      f : Quiver.Hom X Y
      this : L.PreservesMonomorphisms
      ⊢ CategoryTheory.Limits.IsLimit (L.mapCone (CategoryTheory.Limits.Fork.ofι (Ca …
    -/
  preservesKernels X Y f := by
    have := preservesMonomorphisms_of_map_exact _ hL
    apply preservesLimit_of_preserves_limit_cone (kernelIsKernel f)
    apply (KernelFork.isLimitMapConeEquiv _ L).2
    have : Mono ((ShortComplex.mk _ _ (kernel.condition f)).map L).f := by
      dsimp
      infer_instance
    exact (hL (ShortComplex.mk _ _ (kernel.condition f))
      (ShortComplex.exact_of_f_is_kernel _ (kernelIsKernel f))).fIsKernel


@[deprecated (since := "2024-07-09")] alias preservesKernelsOfMapExact :=
  PreservesHomology.preservesKernels

@[deprecated (since := "2024-07-09")] alias preservesCokernelsOfMapExact :=
  PreservesHomology.preservesCokernels


/-- A functor preserving zero morphisms, monos, and cokernels preserves homology. -/
lemma preservesHomology_of_preservesMonos_and_cokernels [PreservesZeroMorphisms L]
    [PreservesMonomorphisms L] [∀ {X Y} (f : X ⟶ Y), PreservesColimit (parallelPair f 0) L] :
    PreservesHomology L := by
  /-
    A : Type u₁
    B : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} A
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} B
    inst✝⁴ : CategoryTheory.Abelian A
    inst✝³ : CategoryTheory.Abelian B
    L : CategoryTheory.Functor A B
    inst✝² : L.PreservesZeroMorphisms
    inst✝¹ : L.PreservesMonomorphisms
    inst✝ : ∀ {X Y : A} (f : Quiver.Hom X Y), CategoryTheory.Limits.PreservesColim …
    ⊢ L.PreservesHomology
  -/
  apply preservesHomology_of_map_exact
  /-
    case hL
    A : Type u₁
    B : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} A
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} B
    inst✝⁴ : CategoryTheory.Abelian A
    inst✝³ : CategoryTheory.Abelian B
    L : CategoryTheory.Functor A B
    inst✝² : L.PreservesZeroMorphisms
    inst✝¹ : L.PreservesMonomorphisms
    inst✝ : ∀ {X Y : A} (f : Quiver.Hom X Y), CategoryTheory.Limits.PreservesColim …
    ⊢ ∀ (S : CategoryTheory.ShortComplex A), S.Exact → (S.map L).Exact
  -/
  intro S hS
  let φ : (ShortComplex.mk _ _ (Abelian.comp_coimage_π_eq_zero S.zero)).map L ⟶ S.map L :=
    { τ₁ := 𝟙 _
      τ₂ := 𝟙 _
      τ₃ := L.map (Abelian.factorThruCoimage S.g)
      comm₂₃ := by
        dsimp
        rw [Category.id_comp, ← L.map_comp, cokernel.π_desc] }
  /-
    case hL
    A : Type u₁
    B : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} A
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} B
    inst✝⁴ : CategoryTheory.Abelian A
    inst✝³ : CategoryTheory.Abelian B
    L : CategoryTheory.Functor A B
    inst✝² : L.PreservesZeroMorphisms
    inst✝¹ : L.PreservesMonomorphisms
    inst✝ : ∀ {X Y : A} (f : Quiver.Hom X Y), CategoryTheory.Limits.PreservesColim …
    S : CategoryTheory.ShortComplex A
    hS : S.Exact
    φ : Quiver.Hom ((CategoryTheory.ShortComplex.mk S.f (CategoryTheory.Abelian.co …
    ⊢ (S.map L).Exact
  -/
  apply (ShortComplex.exact_iff_of_epi_of_isIso_of_mono φ).1
  /-
    case hL
    A : Type u₁
    B : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} A
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} B
    inst✝⁴ : CategoryTheory.Abelian A
    inst✝³ : CategoryTheory.Abelian B
    L : CategoryTheory.Functor A B
    inst✝² : L.PreservesZeroMorphisms
    inst✝¹ : L.PreservesMonomorphisms
    inst✝ : ∀ {X Y : A} (f : Quiver.Hom X Y), CategoryTheory.Limits.PreservesColim …
    S : CategoryTheory.ShortComplex A
    hS : S.Exact
    φ : Quiver.Hom ((CategoryTheory.ShortComplex.mk S.f (CategoryTheory.Abelian.co …
    ⊢ ((CategoryTheory.ShortComplex.mk S.f (CategoryTheory.Abelian.coimage.π S.g)  …
  -/
  apply ShortComplex.exact_of_g_is_cokernel
  /-
    case hL.hS
    A : Type u₁
    B : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} A
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} B
    inst✝⁴ : CategoryTheory.Abelian A
    inst✝³ : CategoryTheory.Abelian B
    L : CategoryTheory.Functor A B
    inst✝² : L.PreservesZeroMorphisms
    inst✝¹ : L.PreservesMonomorphisms
    inst✝ : ∀ {X Y : A} (f : Quiver.Hom X Y), CategoryTheory.Limits.PreservesColim …
    S : CategoryTheory.ShortComplex A
    hS : S.Exact
    φ : Quiver.Hom ((CategoryTheory.ShortComplex.mk S.f (CategoryTheory.Abelian.co …
    ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.CokernelCofork.ofπ (( …
  -/
  exact CokernelCofork.mapIsColimit _ ((S.exact_iff_exact_coimage_π).1 hS).gIsCokernel L
  /-
    🎉 no goals
  -/


/-- A functor preserving zero morphisms, epis, and kernels preserves homology. -/
lemma preservesHomology_of_preservesEpis_and_kernels [PreservesZeroMorphisms L]
    [PreservesEpimorphisms L] [∀ {X Y} (f : X ⟶ Y), PreservesLimit (parallelPair f 0) L] :
    PreservesHomology L := by
  /-
    A : Type u₁
    B : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} A
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} B
    inst✝⁴ : CategoryTheory.Abelian A
    inst✝³ : CategoryTheory.Abelian B
    L : CategoryTheory.Functor A B
    inst✝² : L.PreservesZeroMorphisms
    inst✝¹ : L.PreservesEpimorphisms
    inst✝ : ∀ {X Y : A} (f : Quiver.Hom X Y), CategoryTheory.Limits.PreservesLimit …
    ⊢ L.PreservesHomology
  -/
  apply preservesHomology_of_map_exact
  /-
    case hL
    A : Type u₁
    B : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} A
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} B
    inst✝⁴ : CategoryTheory.Abelian A
    inst✝³ : CategoryTheory.Abelian B
    L : CategoryTheory.Functor A B
    inst✝² : L.PreservesZeroMorphisms
    inst✝¹ : L.PreservesEpimorphisms
    inst✝ : ∀ {X Y : A} (f : Quiver.Hom X Y), CategoryTheory.Limits.PreservesLimit …
    ⊢ ∀ (S : CategoryTheory.ShortComplex A), S.Exact → (S.map L).Exact
  -/
  intro S hS
  let φ : S.map L ⟶ (ShortComplex.mk _ _ (Abelian.image_ι_comp_eq_zero S.zero)).map L :=
    { τ₁ := L.map (Abelian.factorThruImage S.f)
      τ₂ := 𝟙 _
      τ₃ := 𝟙 _
      comm₁₂ := by
        dsimp
        rw [Category.comp_id, ← L.map_comp, kernel.lift_ι] }
  /-
    case hL
    A : Type u₁
    B : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} A
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} B
    inst✝⁴ : CategoryTheory.Abelian A
    inst✝³ : CategoryTheory.Abelian B
    L : CategoryTheory.Functor A B
    inst✝² : L.PreservesZeroMorphisms
    inst✝¹ : L.PreservesEpimorphisms
    inst✝ : ∀ {X Y : A} (f : Quiver.Hom X Y), CategoryTheory.Limits.PreservesLimit …
    S : CategoryTheory.ShortComplex A
    hS : S.Exact
    φ : Quiver.Hom (S.map L) ((CategoryTheory.ShortComplex.mk (CategoryTheory.Abel …
    ⊢ (S.map L).Exact
  -/
  apply (ShortComplex.exact_iff_of_epi_of_isIso_of_mono φ).2
  /-
    case hL
    A : Type u₁
    B : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} A
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} B
    inst✝⁴ : CategoryTheory.Abelian A
    inst✝³ : CategoryTheory.Abelian B
    L : CategoryTheory.Functor A B
    inst✝² : L.PreservesZeroMorphisms
    inst✝¹ : L.PreservesEpimorphisms
    inst✝ : ∀ {X Y : A} (f : Quiver.Hom X Y), CategoryTheory.Limits.PreservesLimit …
    S : CategoryTheory.ShortComplex A
    hS : S.Exact
    φ : Quiver.Hom (S.map L) ((CategoryTheory.ShortComplex.mk (CategoryTheory.Abel …
    ⊢ ((CategoryTheory.ShortComplex.mk (CategoryTheory.Abelian.image.ι S.f) S.g ⋯) …
  -/
  apply ShortComplex.exact_of_f_is_kernel
  /-
    case hL.hS
    A : Type u₁
    B : Type u₂
    inst✝⁶ : CategoryTheory.Category.{v₁, u₁} A
    inst✝⁵ : CategoryTheory.Category.{v₂, u₂} B
    inst✝⁴ : CategoryTheory.Abelian A
    inst✝³ : CategoryTheory.Abelian B
    L : CategoryTheory.Functor A B
    inst✝² : L.PreservesZeroMorphisms
    inst✝¹ : L.PreservesEpimorphisms
    inst✝ : ∀ {X Y : A} (f : Quiver.Hom X Y), CategoryTheory.Limits.PreservesLimit …
    S : CategoryTheory.ShortComplex A
    hS : S.Exact
    φ : Quiver.Hom (S.map L) ((CategoryTheory.ShortComplex.mk (CategoryTheory.Abel …
    ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.KernelFork.ofι ((Catego …
  -/
  exact KernelFork.mapIsLimit _ ((S.exact_iff_exact_image_ι).1 hS).fIsKernel L
  /-
    🎉 no goals
  -/


