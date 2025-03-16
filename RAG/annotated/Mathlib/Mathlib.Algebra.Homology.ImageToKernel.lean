theorem image_le_kernel (w : f ≫ g = 0) : imageSubobject f ≤ kernelSubobject g :=
                                                   /-
                                                     V : Type u
                                                     inst✝³ : CategoryTheory.Category.{v, u} V
                                                     inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
                                                     A B C : V
                                                     f : Quiver.Hom A B
                                                     inst✝¹ : CategoryTheory.Limits.HasImage f
                                                     g : Quiver.Hom B C
                                                     inst✝ : CategoryTheory.Limits.HasKernel g
                                                     w : Eq (CategoryTheory.CategoryStruct.comp f g) 0
                                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernel.lift g  …
                                                   -/
  imageSubobject_le_mk _ _ (kernel.lift _ _ w) (by simp)
                                                   /-
                                                     🎉 no goals
                                                   -/


/-- The canonical morphism `imageSubobject f ⟶ kernelSubobject g` when `f ≫ g = 0`.
-/
def imageToKernel (w : f ≫ g = 0) : (imageSubobject f : V) ⟶ (kernelSubobject g : V) :=
  Subobject.ofLE _ _ (image_le_kernel _ _ w)


instance (w : f ≫ g = 0) : Mono (imageToKernel f g w) := by
  /-
    ι : Type u_1
    V : Type u
    inst✝³ : CategoryTheory.Category.{v, u} V
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
    A B C : V
    f : Quiver.Hom A B
    inst✝¹ : CategoryTheory.Limits.HasImage f
    g : Quiver.Hom B C
    inst✝ : CategoryTheory.Limits.HasKernel g
    w : Eq (CategoryTheory.CategoryStruct.comp f g) 0
    ⊢ CategoryTheory.Mono (imageToKernel f g w)
  -/
  dsimp only [imageToKernel]
  /-
    ι : Type u_1
    V : Type u
    inst✝³ : CategoryTheory.Category.{v, u} V
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
    A B C : V
    f : Quiver.Hom A B
    inst✝¹ : CategoryTheory.Limits.HasImage f
    g : Quiver.Hom B C
    inst✝ : CategoryTheory.Limits.HasKernel g
    w : Eq (CategoryTheory.CategoryStruct.comp f g) 0
    ⊢ CategoryTheory.Mono ((CategoryTheory.Limits.imageSubobject f).ofLE (Category …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- Prefer `imageToKernel`. -/
@[simp]
theorem subobject_ofLE_as_imageToKernel (w : f ≫ g = 0) (h) :
    Subobject.ofLE (imageSubobject f) (kernelSubobject g) h = imageToKernel f g w :=
  rfl


@[reassoc (attr := simp)]
theorem imageToKernel_arrow (w : f ≫ g = 0) :
    imageToKernel f g w ≫ (kernelSubobject g).arrow = (imageSubobject f).arrow := by
  /-
    V : Type u
    inst✝³ : CategoryTheory.Category.{v, u} V
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
    A B C : V
    f : Quiver.Hom A B
    inst✝¹ : CategoryTheory.Limits.HasImage f
    g : Quiver.Hom B C
    inst✝ : CategoryTheory.Limits.HasKernel g
    w : Eq (CategoryTheory.CategoryStruct.comp f g) 0
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (imageToKernel f g w) (CategoryTheory …
  -/
  simp [imageToKernel]
  /-
    🎉 no goals
  -/


@[simp]
lemma imageToKernel_arrow_apply [ConcreteCategory V] (w : f ≫ g = 0)
    (x : (forget V).obj (Subobject.underlying.obj (imageSubobject f))) :
    (kernelSubobject g).arrow (imageToKernel f g w x) =
      (imageSubobject f).arrow x := by
  /-
    V : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} V
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms V
    A B C : V
    f : Quiver.Hom A B
    inst✝² : CategoryTheory.Limits.HasImage f
    g : Quiver.Hom B C
    inst✝¹ : CategoryTheory.Limits.HasKernel g
    inst✝ : CategoryTheory.ConcreteCategory V
    w : Eq (CategoryTheory.CategoryStruct.comp f g) 0
    x : (CategoryTheory.forget V).obj (CategoryTheory.Subobject.underlying.obj (Ca …
    ⊢ Eq ((CategoryTheory.Limits.kernelSubobject g).arrow ((imageToKernel f g w) x …
  -/
  rw [← comp_apply, imageToKernel_arrow]
  /-
    🎉 no goals
  -/

-- This is less useful as a `simp` lemma than it initially appears,
-- as it "loses" the information the morphism factors through the image.

theorem factorThruImageSubobject_comp_imageToKernel (w : f ≫ g = 0) :
    factorThruImageSubobject f ≫ imageToKernel f g w = factorThruKernelSubobject g f w := by
  /-
    V : Type u
    inst✝³ : CategoryTheory.Category.{v, u} V
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
    A B C : V
    f : Quiver.Hom A B
    inst✝¹ : CategoryTheory.Limits.HasImage f
    g : Quiver.Hom B C
    inst✝ : CategoryTheory.Limits.HasKernel g
    w : Eq (CategoryTheory.CategoryStruct.comp f g) 0
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.factorThruImag …
  -/
  ext
  /-
    case h
    V : Type u
    inst✝³ : CategoryTheory.Category.{v, u} V
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
    A B C : V
    f : Quiver.Hom A B
    inst✝¹ : CategoryTheory.Limits.HasImage f
    g : Quiver.Hom B C
    inst✝ : CategoryTheory.Limits.HasKernel g
    w : Eq (CategoryTheory.CategoryStruct.comp f g) 0
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem imageToKernel_zero_left [HasKernels V] [HasZeroObject V] {w} :
    imageToKernel (0 : A ⟶ B) g w = 0 := by
  /-
    V : Type u
    inst✝³ : CategoryTheory.Category.{v, u} V
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
    A B C : V
    g : Quiver.Hom B C
    inst✝¹ : CategoryTheory.Limits.HasKernels V
    inst✝ : CategoryTheory.Limits.HasZeroObject V
    w : Eq (CategoryTheory.CategoryStruct.comp 0 g) 0
    ⊢ Eq (imageToKernel 0 g w) 0
  -/
  ext
  /-
    case h
    V : Type u
    inst✝³ : CategoryTheory.Category.{v, u} V
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
    A B C : V
    g : Quiver.Hom B C
    inst✝¹ : CategoryTheory.Limits.HasKernels V
    inst✝ : CategoryTheory.Limits.HasZeroObject V
    w : Eq (CategoryTheory.CategoryStruct.comp 0 g) 0
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (imageToKernel 0 g w) (CategoryTheory …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem imageToKernel_zero_right [HasImages V] {w} :
    imageToKernel f (0 : B ⟶ C) w =
      (imageSubobject f).arrow ≫ inv (kernelSubobject (0 : B ⟶ C)).arrow := by
  /-
    V : Type u
    inst✝² : CategoryTheory.Category.{v, u} V
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
    A B C : V
    f : Quiver.Hom A B
    inst✝ : CategoryTheory.Limits.HasImages V
    w : Eq (CategoryTheory.CategoryStruct.comp f 0) 0
    ⊢ Eq (imageToKernel f 0 w) (CategoryTheory.CategoryStruct.comp (CategoryTheory …
  -/
  ext
  /-
    case h
    V : Type u
    inst✝² : CategoryTheory.Category.{v, u} V
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms V
    A B C : V
    f : Quiver.Hom A B
    inst✝ : CategoryTheory.Limits.HasImages V
    w : Eq (CategoryTheory.CategoryStruct.comp f 0) 0
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (imageToKernel f 0 w) (CategoryTheory …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem imageToKernel_comp_right {D : V} (h : C ⟶ D) (w : f ≫ g = 0) :
                                /-
                                  ι : Type u_1
                                  V : Type u
                                  inst✝³ : CategoryTheory.Category.{v, u} V
                                  inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
                                  A B C : V
                                  f : Quiver.Hom A B
                                  g : Quiver.Hom B C
                                  inst✝¹ : CategoryTheory.Limits.HasKernels V
                                  inst✝ : CategoryTheory.Limits.HasImages V
                                  D : V
                                  h : Quiver.Hom C D
                                  w : Eq (CategoryTheory.CategoryStruct.comp f g) 0
                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.comp …
                                -/
    imageToKernel f (g ≫ h) (by simp [reassoc_of% w]) =
                                /-
                                  🎉 no goals
                                -/
      imageToKernel f g w ≫ Subobject.ofLE _ _ (kernelSubobject_comp_le g h) := by
  /-
    V : Type u
    inst✝³ : CategoryTheory.Category.{v, u} V
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
    A B C : V
    f : Quiver.Hom A B
    g : Quiver.Hom B C
    inst✝¹ : CategoryTheory.Limits.HasKernels V
    inst✝ : CategoryTheory.Limits.HasImages V
    D : V
    h : Quiver.Hom C D
    w : Eq (CategoryTheory.CategoryStruct.comp f g) 0
    ⊢ Eq (imageToKernel f (CategoryTheory.CategoryStruct.comp g h) ⋯) (CategoryThe …
  -/
  ext
  /-
    case h
    V : Type u
    inst✝³ : CategoryTheory.Category.{v, u} V
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
    A B C : V
    f : Quiver.Hom A B
    g : Quiver.Hom B C
    inst✝¹ : CategoryTheory.Limits.HasKernels V
    inst✝ : CategoryTheory.Limits.HasImages V
    D : V
    h : Quiver.Hom C D
    w : Eq (CategoryTheory.CategoryStruct.comp f g) 0
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (imageToKernel f (CategoryTheory.Cate …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem imageToKernel_comp_left {Z : V} (h : Z ⟶ A) (w : f ≫ g = 0) :
                                /-
                                  ι : Type u_1
                                  V : Type u
                                  inst✝³ : CategoryTheory.Category.{v, u} V
                                  inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
                                  A B C : V
                                  f : Quiver.Hom A B
                                  g : Quiver.Hom B C
                                  inst✝¹ : CategoryTheory.Limits.HasKernels V
                                  inst✝ : CategoryTheory.Limits.HasImages V
                                  Z : V
                                  h : Quiver.Hom Z A
                                  w : Eq (CategoryTheory.CategoryStruct.comp f g) 0
                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp h …
                                -/
    imageToKernel (h ≫ f) g (by simp [w]) =
                                /-
                                  🎉 no goals
                                -/
      Subobject.ofLE _ _ (imageSubobject_comp_le h f) ≫ imageToKernel f g w := by
  /-
    V : Type u
    inst✝³ : CategoryTheory.Category.{v, u} V
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
    A B C : V
    f : Quiver.Hom A B
    g : Quiver.Hom B C
    inst✝¹ : CategoryTheory.Limits.HasKernels V
    inst✝ : CategoryTheory.Limits.HasImages V
    Z : V
    h : Quiver.Hom Z A
    w : Eq (CategoryTheory.CategoryStruct.comp f g) 0
    ⊢ Eq (imageToKernel (CategoryTheory.CategoryStruct.comp h f) g ⋯) (CategoryThe …
  -/
  ext
  /-
    case h
    V : Type u
    inst✝³ : CategoryTheory.Category.{v, u} V
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
    A B C : V
    f : Quiver.Hom A B
    g : Quiver.Hom B C
    inst✝¹ : CategoryTheory.Limits.HasKernels V
    inst✝ : CategoryTheory.Limits.HasImages V
    Z : V
    h : Quiver.Hom Z A
    w : Eq (CategoryTheory.CategoryStruct.comp f g) 0
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (imageToKernel (CategoryTheory.Catego …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem imageToKernel_comp_mono {D : V} (h : C ⟶ D) [Mono h] (w) :
    imageToKernel f (g ≫ h) w =
                                                /-
                                                  ι : Type u_1
                                                  V : Type u
                                                  inst✝⁴ : CategoryTheory.Category.{v, u} V
                                                  inst✝³ : CategoryTheory.Limits.HasZeroMorphisms V
                                                  A B C : V
                                                  f : Quiver.Hom A B
                                                  g : Quiver.Hom B C
                                                  inst✝² : CategoryTheory.Limits.HasKernels V
                                                  inst✝¹ : CategoryTheory.Limits.HasImages V
                                                  D : V
                                                  h : Quiver.Hom C D
                                                  inst✝ : CategoryTheory.Mono h
                                                  w : Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.co …
                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
                                                -/
      imageToKernel f g ((cancel_mono h).mp (by simpa using w : (f ≫ g) ≫ h = 0 ≫ h)) ≫
                                                /-
                                                  🎉 no goals
                                                -/
        (Subobject.isoOfEq _ _ (kernelSubobject_comp_mono g h)).inv := by
  /-
    V : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} V
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms V
    A B C : V
    f : Quiver.Hom A B
    g : Quiver.Hom B C
    inst✝² : CategoryTheory.Limits.HasKernels V
    inst✝¹ : CategoryTheory.Limits.HasImages V
    D : V
    h : Quiver.Hom C D
    inst✝ : CategoryTheory.Mono h
    w : Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.co …
    ⊢ Eq (imageToKernel f (CategoryTheory.CategoryStruct.comp g h) w) (CategoryThe …
  -/
  ext
  /-
    case h
    V : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} V
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms V
    A B C : V
    f : Quiver.Hom A B
    g : Quiver.Hom B C
    inst✝² : CategoryTheory.Limits.HasKernels V
    inst✝¹ : CategoryTheory.Limits.HasImages V
    D : V
    h : Quiver.Hom C D
    inst✝ : CategoryTheory.Mono h
    w : Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.CategoryStruct.co …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (imageToKernel f (CategoryTheory.Cate …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem imageToKernel_epi_comp {Z : V} (h : Z ⟶ A) [Epi h] (w) :
    imageToKernel (h ≫ f) g w =
      Subobject.ofLE _ _ (imageSubobject_comp_le h f) ≫
                                                 /-
                                                   ι : Type u_1
                                                   V : Type u
                                                   inst✝⁴ : CategoryTheory.Category.{v, u} V
                                                   inst✝³ : CategoryTheory.Limits.HasZeroMorphisms V
                                                   A B C : V
                                                   f : Quiver.Hom A B
                                                   g : Quiver.Hom B C
                                                   inst✝² : CategoryTheory.Limits.HasKernels V
                                                   inst✝¹ : CategoryTheory.Limits.HasImages V
                                                   Z : V
                                                   h : Quiver.Hom Z A
                                                   inst✝ : CategoryTheory.Epi h
                                                   w : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp …
                                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp h (CategoryTheory.CategoryStruct.comp …
                                                 -/
        imageToKernel f g ((cancel_epi h).mp (by simpa using w : h ≫ f ≫ g = h ≫ 0)) := by
                                                 /-
                                                   🎉 no goals
                                                 -/
  /-
    V : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} V
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms V
    A B C : V
    f : Quiver.Hom A B
    g : Quiver.Hom B C
    inst✝² : CategoryTheory.Limits.HasKernels V
    inst✝¹ : CategoryTheory.Limits.HasImages V
    Z : V
    h : Quiver.Hom Z A
    inst✝ : CategoryTheory.Epi h
    w : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp …
    ⊢ Eq (imageToKernel (CategoryTheory.CategoryStruct.comp h f) g w) (CategoryThe …
  -/
  ext
  /-
    case h
    V : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} V
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms V
    A B C : V
    f : Quiver.Hom A B
    g : Quiver.Hom B C
    inst✝² : CategoryTheory.Limits.HasKernels V
    inst✝¹ : CategoryTheory.Limits.HasImages V
    Z : V
    h : Quiver.Hom Z A
    inst✝ : CategoryTheory.Epi h
    w : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (imageToKernel (CategoryTheory.Catego …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem imageToKernel_comp_hom_inv_comp [HasEqualizers V] [HasImages V] {Z : V} {i : B ≅ Z} (w) :
    imageToKernel (f ≫ i.hom) (i.inv ≫ g) w =
      (imageSubobjectCompIso _ _).hom ≫
                              /-
                                ι : Type u_1
                                V : Type u
                                inst✝³ : CategoryTheory.Category.{v, u} V
                                inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
                                A B C : V
                                f : Quiver.Hom A B
                                g : Quiver.Hom B C
                                inst✝¹ : CategoryTheory.Limits.HasEqualizers V
                                inst✝ : CategoryTheory.Limits.HasImages V
                                Z : V
                                i : CategoryTheory.Iso B Z
                                w : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp …
                                ⊢ Eq (CategoryTheory.CategoryStruct.comp f g) 0
                              -/
        imageToKernel f g (by simpa using w) ≫ (kernelSubobjectIsoComp i.inv g).inv := by
                              /-
                                🎉 no goals
                              -/
  /-
    V : Type u
    inst✝³ : CategoryTheory.Category.{v, u} V
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
    A B C : V
    f : Quiver.Hom A B
    g : Quiver.Hom B C
    inst✝¹ : CategoryTheory.Limits.HasEqualizers V
    inst✝ : CategoryTheory.Limits.HasImages V
    Z : V
    i : CategoryTheory.Iso B Z
    w : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp …
    ⊢ Eq (imageToKernel (CategoryTheory.CategoryStruct.comp f i.hom) (CategoryTheo …
  -/
  ext
  /-
    case h
    V : Type u
    inst✝³ : CategoryTheory.Category.{v, u} V
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
    A B C : V
    f : Quiver.Hom A B
    g : Quiver.Hom B C
    inst✝¹ : CategoryTheory.Limits.HasEqualizers V
    inst✝ : CategoryTheory.Limits.HasImages V
    Z : V
    i : CategoryTheory.Iso B Z
    w : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (imageToKernel (CategoryTheory.Catego …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- `imageToKernel` for `A --0--> B --g--> C`, where `g` is a mono is itself an epi
(i.e. the sequence is exact at `B`).
-/
instance imageToKernel_epi_of_zero_of_mono [HasKernels V] [HasZeroObject V] [Mono g] :
                                         /-
                                           ι : Type u_1
                                           V : Type u
                                           inst✝⁴ : CategoryTheory.Category.{v, u} V
                                           inst✝³ : CategoryTheory.Limits.HasZeroMorphisms V
                                           A B C : V
                                           f : Quiver.Hom A B
                                           g : Quiver.Hom B C
                                           inst✝² : CategoryTheory.Limits.HasKernels V
                                           inst✝¹ : CategoryTheory.Limits.HasZeroObject V
                                           inst✝ : CategoryTheory.Mono g
                                           ⊢ Eq (CategoryTheory.CategoryStruct.comp 0 g) 0
                                         -/
    Epi (imageToKernel (0 : A ⟶ B) g (by simp)) :=
                                         /-
                                           🎉 no goals
                                         -/
  epi_of_target_iso_zero _ (kernelSubobjectIso g ≪≫ kernel.ofMono g)


/-- `imageToKernel` for `A --f--> B --0--> C`, where `g` is an epi is itself an epi
(i.e. the sequence is exact at `B`).
-/
instance imageToKernel_epi_of_epi_of_zero [HasImages V] [Epi f] :
                                         /-
                                           ι : Type u_1
                                           V : Type u
                                           inst✝³ : CategoryTheory.Category.{v, u} V
                                           inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
                                           A B C : V
                                           f : Quiver.Hom A B
                                           g : Quiver.Hom B C
                                           inst✝¹ : CategoryTheory.Limits.HasImages V
                                           inst✝ : CategoryTheory.Epi f
                                           ⊢ Eq (CategoryTheory.CategoryStruct.comp f 0) 0
                                         -/
    Epi (imageToKernel f (0 : B ⟶ C) (by simp)) := by
                                         /-
                                           🎉 no goals
                                         -/
  /-
    ι : Type u_1
    V : Type u
    inst✝³ : CategoryTheory.Category.{v, u} V
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
    A B C : V
    f : Quiver.Hom A B
    g : Quiver.Hom B C
    inst✝¹ : CategoryTheory.Limits.HasImages V
    inst✝ : CategoryTheory.Epi f
    ⊢ CategoryTheory.Epi (imageToKernel f 0 ⋯)
  -/
  simp only [imageToKernel_zero_right]
  /-
    ι : Type u_1
    V : Type u
    inst✝³ : CategoryTheory.Category.{v, u} V
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
    A B C : V
    f : Quiver.Hom A B
    g : Quiver.Hom B C
    inst✝¹ : CategoryTheory.Limits.HasImages V
    inst✝ : CategoryTheory.Epi f
    ⊢ CategoryTheory.Epi (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limit …
  -/
  haveI := epi_image_of_epi f
  /-
    ι : Type u_1
    V : Type u
    inst✝³ : CategoryTheory.Category.{v, u} V
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
    A B C : V
    f : Quiver.Hom A B
    g : Quiver.Hom B C
    inst✝¹ : CategoryTheory.Limits.HasImages V
    inst✝ : CategoryTheory.Epi f
    this : CategoryTheory.Epi (CategoryTheory.Limits.image.ι f)
    ⊢ CategoryTheory.Epi (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limit …
  -/
  rw [← imageSubobject_arrow]
  /-
    ι : Type u_1
    V : Type u
    inst✝³ : CategoryTheory.Category.{v, u} V
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
    A B C : V
    f : Quiver.Hom A B
    g : Quiver.Hom B C
    inst✝¹ : CategoryTheory.Limits.HasImages V
    inst✝ : CategoryTheory.Epi f
    this : CategoryTheory.Epi (CategoryTheory.Limits.image.ι f)
    ⊢ CategoryTheory.Epi (CategoryTheory.CategoryStruct.comp (CategoryTheory.Categ …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- While `imageToKernel f g w` provides a morphism
`imageSubobject f ⟶ kernelSubobject g`
in terms of the subobject API,
this variant provides a morphism
`image f ⟶ kernel g`,
which is sometimes more convenient.
-/
def imageToKernel' (w : f ≫ g = 0) : image f ⟶ kernel g :=
  kernel.lift g (image.ι f) <| by
    /-
      ι : Type u_1
      V : Type u
      inst✝³ : CategoryTheory.Category.{v, u} V
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
      A B C : V
      f : Quiver.Hom A B
      g : Quiver.Hom B C
      w✝ : Eq (CategoryTheory.CategoryStruct.comp f g) 0
      inst✝¹ : CategoryTheory.Limits.HasKernels V
      inst✝ : CategoryTheory.Limits.HasImages V
      w : Eq (CategoryTheory.CategoryStruct.comp f g) 0
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.image.ι f) g) 0
    -/
    ext
    /-
      case w
      ι : Type u_1
      V : Type u
      inst✝³ : CategoryTheory.Category.{v, u} V
      inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
      A B C : V
      f : Quiver.Hom A B
      g : Quiver.Hom B C
      w✝ : Eq (CategoryTheory.CategoryStruct.comp f g) 0
      inst✝¹ : CategoryTheory.Limits.HasKernels V
      inst✝ : CategoryTheory.Limits.HasImages V
      w : Eq (CategoryTheory.CategoryStruct.comp f g) 0
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.factorThruImag …
    -/
    simpa using w
    /-
      🎉 no goals
    -/


@[simp]
theorem imageSubobjectIso_imageToKernel' (w : f ≫ g = 0) :
    (imageSubobjectIso f).hom ≫ imageToKernel' f g w =
      imageToKernel f g w ≫ (kernelSubobjectIso g).hom := by
  /-
    V : Type u
    inst✝³ : CategoryTheory.Category.{v, u} V
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
    A B C : V
    f : Quiver.Hom A B
    g : Quiver.Hom B C
    inst✝¹ : CategoryTheory.Limits.HasKernels V
    inst✝ : CategoryTheory.Limits.HasImages V
    w : Eq (CategoryTheory.CategoryStruct.comp f g) 0
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.imageSubobject …
  -/
  ext
  /-
    case h
    V : Type u
    inst✝³ : CategoryTheory.Category.{v, u} V
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
    A B C : V
    f : Quiver.Hom A B
    g : Quiver.Hom B C
    inst✝¹ : CategoryTheory.Limits.HasKernels V
    inst✝ : CategoryTheory.Limits.HasImages V
    w : Eq (CategoryTheory.CategoryStruct.comp f g) 0
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp [imageToKernel']
  /-
    🎉 no goals
  -/


@[simp]
theorem imageToKernel'_kernelSubobjectIso (w : f ≫ g = 0) :
    imageToKernel' f g w ≫ (kernelSubobjectIso g).inv =
      (imageSubobjectIso f).inv ≫ imageToKernel f g w := by
  /-
    V : Type u
    inst✝³ : CategoryTheory.Category.{v, u} V
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
    A B C : V
    f : Quiver.Hom A B
    g : Quiver.Hom B C
    inst✝¹ : CategoryTheory.Limits.HasKernels V
    inst✝ : CategoryTheory.Limits.HasImages V
    w : Eq (CategoryTheory.CategoryStruct.comp f g) 0
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (imageToKernel' f g w) (CategoryTheor …
  -/
  ext
  /-
    case h
    V : Type u
    inst✝³ : CategoryTheory.Category.{v, u} V
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms V
    A B C : V
    f : Quiver.Hom A B
    g : Quiver.Hom B C
    inst✝¹ : CategoryTheory.Limits.HasKernels V
    inst✝ : CategoryTheory.Limits.HasImages V
    w : Eq (CategoryTheory.CategoryStruct.comp f g) 0
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp [imageToKernel']
  /-
    🎉 no goals
  -/


