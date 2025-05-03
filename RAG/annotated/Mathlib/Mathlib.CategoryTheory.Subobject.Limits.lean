/-- The equalizer of morphisms `f g : X ⟶ Y` as a `Subobject X`. -/
abbrev equalizerSubobject : Subobject X :=
  Subobject.mk (equalizer.ι f g)


/-- The underlying object of `equalizerSubobject f g` is (up to isomorphism!)
the same as the chosen object `equalizer f g`. -/
def equalizerSubobjectIso : (equalizerSubobject f g : C) ≅ equalizer f g :=
  Subobject.underlyingIso (equalizer.ι f g)


@[reassoc (attr := simp)]
theorem equalizerSubobject_arrow :
    (equalizerSubobjectIso f g).hom ≫ equalizer.ι f g = (equalizerSubobject f g).arrow := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y : C
    f g : Quiver.Hom X Y
    inst✝ : CategoryTheory.Limits.HasEqualizer f g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.equalizerSubob …
  -/
  simp [equalizerSubobjectIso]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem equalizerSubobject_arrow' :
    (equalizerSubobjectIso f g).inv ≫ (equalizerSubobject f g).arrow = equalizer.ι f g := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y : C
    f g : Quiver.Hom X Y
    inst✝ : CategoryTheory.Limits.HasEqualizer f g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.equalizerSubob …
  -/
  simp [equalizerSubobjectIso]
  /-
    🎉 no goals
  -/


@[reassoc]
theorem equalizerSubobject_arrow_comp :
    (equalizerSubobject f g).arrow ≫ f = (equalizerSubobject f g).arrow ≫ g := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y : C
    f g : Quiver.Hom X Y
    inst✝ : CategoryTheory.Limits.HasEqualizer f g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.equalizerSubob …
  -/
  rw [← equalizerSubobject_arrow, Category.assoc, Category.assoc, equalizer.condition]
  /-
    🎉 no goals
  -/


theorem equalizerSubobject_factors {W : C} (h : W ⟶ X) (w : h ≫ f = h ≫ g) :
    (equalizerSubobject f g).Factors h :=
                          /-
                            C : Type u
                            inst✝¹ : CategoryTheory.Category.{v, u} C
                            X Y : C
                            f g : Quiver.Hom X Y
                            inst✝ : CategoryTheory.Limits.HasEqualizer f g
                            W : C
                            h : Quiver.Hom W X
                            w : Eq (CategoryTheory.CategoryStruct.comp h f) (CategoryTheory.CategoryStruct …
                            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.equalizer.lift …
                          -/
  ⟨equalizer.lift h w, by simp⟩
                          /-
                            🎉 no goals
                          -/


theorem equalizerSubobject_factors_iff {W : C} (h : W ⟶ X) :
    (equalizerSubobject f g).Factors h ↔ h ≫ f = h ≫ g :=
  ⟨fun w => by
    rw [← Subobject.factorThru_arrow _ _ w, Category.assoc, equalizerSubobject_arrow_comp,
      Category.assoc],
    equalizerSubobject_factors f g h⟩


/-- The kernel of a morphism `f : X ⟶ Y` as a `Subobject X`. -/
abbrev kernelSubobject : Subobject X :=
  Subobject.mk (kernel.ι f)


/-- The underlying object of `kernelSubobject f` is (up to isomorphism!)
the same as the chosen object `kernel f`. -/
def kernelSubobjectIso : (kernelSubobject f : C) ≅ kernel f :=
  Subobject.underlyingIso (kernel.ι f)


@[reassoc (attr := simp), elementwise (attr := simp)]
theorem kernelSubobject_arrow :
    (kernelSubobjectIso f).hom ≫ kernel.ι f = (kernelSubobject f).arrow := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Limits.HasKernel f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernelSubobjec …
  -/
  simp [kernelSubobjectIso]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp), elementwise (attr := simp)]
theorem kernelSubobject_arrow' :
    (kernelSubobjectIso f).inv ≫ (kernelSubobject f).arrow = kernel.ι f := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Limits.HasKernel f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernelSubobjec …
  -/
  simp [kernelSubobjectIso]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp), elementwise (attr := simp)]
theorem kernelSubobject_arrow_comp : (kernelSubobject f).arrow ≫ f = 0 := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Limits.HasKernel f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernelSubobjec …
  -/
  rw [← kernelSubobject_arrow]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Limits.HasKernel f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [Category.assoc, kernel.condition, comp_zero]
  /-
    🎉 no goals
  -/


theorem kernelSubobject_factors {W : C} (h : W ⟶ X) (w : h ≫ f = 0) :
    (kernelSubobject f).Factors h :=
                         /-
                           C : Type u
                           inst✝² : CategoryTheory.Category.{v, u} C
                           X Y : C
                           inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                           f : Quiver.Hom X Y
                           inst✝ : CategoryTheory.Limits.HasKernel f
                           W : C
                           h : Quiver.Hom W X
                           w : Eq (CategoryTheory.CategoryStruct.comp h f) 0
                           ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernel.lift f  …
                         -/
  ⟨kernel.lift _ h w, by simp⟩
                         /-
                           🎉 no goals
                         -/


theorem kernelSubobject_factors_iff {W : C} (h : W ⟶ X) :
    (kernelSubobject f).Factors h ↔ h ≫ f = 0 :=
  ⟨fun w => by
    rw [← Subobject.factorThru_arrow _ _ w, Category.assoc, kernelSubobject_arrow_comp,
      comp_zero],
    kernelSubobject_factors f h⟩


/-- A factorisation of `h : W ⟶ X` through `kernelSubobject f`, assuming `h ≫ f = 0`. -/
def factorThruKernelSubobject {W : C} (h : W ⟶ X) (w : h ≫ f = 0) : W ⟶ kernelSubobject f :=
  (kernelSubobject f).factorThru h (kernelSubobject_factors f h w)


@[simp]
theorem factorThruKernelSubobject_comp_arrow {W : C} (h : W ⟶ X) (w : h ≫ f = 0) :
    factorThruKernelSubobject f h w ≫ (kernelSubobject f).arrow = h := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Limits.HasKernel f
    W : C
    h : Quiver.Hom W X
    w : Eq (CategoryTheory.CategoryStruct.comp h f) 0
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.factorThruKern …
  -/
  dsimp [factorThruKernelSubobject]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Limits.HasKernel f
    W : C
    h : Quiver.Hom W X
    w : Eq (CategoryTheory.CategoryStruct.comp h f) 0
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.kernelSubobje …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem factorThruKernelSubobject_comp_kernelSubobjectIso {W : C} (h : W ⟶ X) (w : h ≫ f = 0) :
    factorThruKernelSubobject f h w ≫ (kernelSubobjectIso f).hom = kernel.lift f h w :=
                                     /-
                                       C : Type u
                                       inst✝² : CategoryTheory.Category.{v, u} C
                                       X Y : C
                                       inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                                       f : Quiver.Hom X Y
                                       inst✝ : CategoryTheory.Limits.HasKernel f
                                       W : C
                                       h : Quiver.Hom W X
                                       w : Eq (CategoryTheory.CategoryStruct.comp h f) 0
                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
                                     -/
  (cancel_mono (kernel.ι f)).1 <| by simp
                                     /-
                                       🎉 no goals
                                     -/


/-- A commuting square induces a morphism between the kernel subobjects. -/
def kernelSubobjectMap (sq : Arrow.mk f ⟶ Arrow.mk f') :
    (kernelSubobject f : C) ⟶ (kernelSubobject f' : C) :=
  Subobject.factorThru _ ((kernelSubobject f).arrow ≫ sq.left)
                                     /-
                                       C : Type u
                                       inst✝³ : CategoryTheory.Category.{v, u} C
                                       X Y Z : C
                                       inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
                                       f : Quiver.Hom X Y
                                       inst✝¹ : CategoryTheory.Limits.HasKernel f
                                       X' Y' : C
                                       f' : Quiver.Hom X' Y'
                                       inst✝ : CategoryTheory.Limits.HasKernel f'
                                       sq : Quiver.Hom (CategoryTheory.Arrow.mk f) (CategoryTheory.Arrow.mk f')
                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
                                     -/
    (kernelSubobject_factors _ _ (by simp [sq.w]))
                                     /-
                                       🎉 no goals
                                     -/


@[reassoc (attr := simp), elementwise (attr := simp)]
theorem kernelSubobjectMap_arrow (sq : Arrow.mk f ⟶ Arrow.mk f') :
    kernelSubobjectMap sq ≫ (kernelSubobject f').arrow = (kernelSubobject f).arrow ≫ sq.left := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    X Y : C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    f : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.HasKernel f
    X' Y' : C
    f' : Quiver.Hom X' Y'
    inst✝ : CategoryTheory.Limits.HasKernel f'
    sq : Quiver.Hom (CategoryTheory.Arrow.mk f) (CategoryTheory.Arrow.mk f')
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernelSubobjec …
  -/
  simp [kernelSubobjectMap]
  /-
    🎉 no goals
  -/


@[simp]
                                                                                /-
                                                                                  C : Type u
                                                                                  inst✝² : CategoryTheory.Category.{v, u} C
                                                                                  X Y : C
                                                                                  inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                                  f : Quiver.Hom X Y
                                                                                  inst✝ : CategoryTheory.Limits.HasKernel f
                                                                                  ⊢ Eq (CategoryTheory.Limits.kernelSubobjectMap (CategoryTheory.CategoryStruct. …
                                                                                -/
theorem kernelSubobjectMap_id : kernelSubobjectMap (𝟙 (Arrow.mk f)) = 𝟙 _ := by aesop_cat
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


@[simp]
theorem kernelSubobjectMap_comp {X'' Y'' : C} {f'' : X'' ⟶ Y''} [HasKernel f'']
    (sq : Arrow.mk f ⟶ Arrow.mk f') (sq' : Arrow.mk f' ⟶ Arrow.mk f'') :
    kernelSubobjectMap (sq ≫ sq') = kernelSubobjectMap sq ≫ kernelSubobjectMap sq' := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    X Y : C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    f : Quiver.Hom X Y
    inst✝² : CategoryTheory.Limits.HasKernel f
    X' Y' : C
    f' : Quiver.Hom X' Y'
    inst✝¹ : CategoryTheory.Limits.HasKernel f'
    X'' Y'' : C
    f'' : Quiver.Hom X'' Y''
    inst✝ : CategoryTheory.Limits.HasKernel f''
    sq : Quiver.Hom (CategoryTheory.Arrow.mk f) (CategoryTheory.Arrow.mk f')
    sq' : Quiver.Hom (CategoryTheory.Arrow.mk f') (CategoryTheory.Arrow.mk f'')
    ⊢ Eq (CategoryTheory.Limits.kernelSubobjectMap (CategoryTheory.CategoryStruct. …
  -/
  aesop_cat
  /-
    🎉 no goals
  -/


@[reassoc]
theorem kernel_map_comp_kernelSubobjectIso_inv (sq : Arrow.mk f ⟶ Arrow.mk f') :
    kernel.map f f' sq.1 sq.2 sq.3.symm ≫ (kernelSubobjectIso _).inv =
                                                               /-
                                                                 C : Type u
                                                                 inst✝³ : CategoryTheory.Category.{v, u} C
                                                                 X Y : C
                                                                 inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
                                                                 f : Quiver.Hom X Y
                                                                 inst✝¹ : CategoryTheory.Limits.HasKernel f
                                                                 X' Y' : C
                                                                 f' : Quiver.Hom X' Y'
                                                                 inst✝ : CategoryTheory.Limits.HasKernel f'
                                                                 sq : Quiver.Hom (CategoryTheory.Arrow.mk f) (CategoryTheory.Arrow.mk f')
                                                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernel.map f f …
                                                               -/
      (kernelSubobjectIso _).inv ≫ kernelSubobjectMap sq := by aesop_cat
                                                               /-
                                                                 🎉 no goals
                                                               -/


@[reassoc]
theorem kernelSubobjectIso_comp_kernel_map (sq : Arrow.mk f ⟶ Arrow.mk f') :
    (kernelSubobjectIso _).hom ≫ kernel.map f f' sq.1 sq.2 sq.3.symm =
      kernelSubobjectMap sq ≫ (kernelSubobjectIso _).hom := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    X Y : C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    f : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.HasKernel f
    X' Y' : C
    f' : Quiver.Hom X' Y'
    inst✝ : CategoryTheory.Limits.HasKernel f'
    sq : Quiver.Hom (CategoryTheory.Arrow.mk f) (CategoryTheory.Arrow.mk f')
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernelSubobjec …
  -/
  simp [← Iso.comp_inv_eq, kernel_map_comp_kernelSubobjectIso_inv]
  /-
    🎉 no goals
  -/


@[simp]
theorem kernelSubobject_zero {A B : C} : kernelSubobject (0 : A ⟶ B) = ⊤ :=
                                 /-
                                   C : Type u
                                   inst✝¹ : CategoryTheory.Category.{v, u} C
                                   inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                   A B : C
                                   ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.kernel.ι 0)
                                 -/
  (isIso_iff_mk_eq_top _).mp (by infer_instance)
                                 /-
                                   🎉 no goals
                                 -/


instance isIso_kernelSubobject_zero_arrow : IsIso (kernelSubobject (0 : X ⟶ Y)).arrow :=
  (isIso_arrow_iff_eq_top _).mpr kernelSubobject_zero


theorem le_kernelSubobject (A : Subobject X) (h : A.arrow ≫ f = 0) : A ≤ kernelSubobject f :=
                                                        /-
                                                          C : Type u
                                                          inst✝² : CategoryTheory.Category.{v, u} C
                                                          X Y : C
                                                          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                                                          f : Quiver.Hom X Y
                                                          inst✝ : CategoryTheory.Limits.HasKernel f
                                                          A : CategoryTheory.Subobject X
                                                          h : Eq (CategoryTheory.CategoryStruct.comp A.arrow f) 0
                                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernel.lift f  …
                                                        -/
  Subobject.le_mk_of_comm (kernel.lift f A.arrow h) (by simp)
                                                        /-
                                                          🎉 no goals
                                                        -/


/-- The isomorphism between the kernel of `f ≫ g` and the kernel of `g`,
when `f` is an isomorphism.
-/
def kernelSubobjectIsoComp {X' : C} (f : X' ⟶ X) [IsIso f] (g : X ⟶ Y) [HasKernel g] :
    (kernelSubobject (f ≫ g) : C) ≅ (kernelSubobject g : C) :=
  kernelSubobjectIso _ ≪≫ kernelIsIsoComp f g ≪≫ (kernelSubobjectIso _).symm


@[simp]
theorem kernelSubobjectIsoComp_hom_arrow {X' : C} (f : X' ⟶ X) [IsIso f] (g : X ⟶ Y) [HasKernel g] :
    (kernelSubobjectIsoComp f g).hom ≫ (kernelSubobject g).arrow =
      (kernelSubobject (f ≫ g)).arrow ≫ f := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    X Y : C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    X' : C
    f : Quiver.Hom X' X
    inst✝¹ : CategoryTheory.IsIso f
    g : Quiver.Hom X Y
    inst✝ : CategoryTheory.Limits.HasKernel g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernelSubobjec …
  -/
  simp [kernelSubobjectIsoComp]
  /-
    🎉 no goals
  -/


@[simp]
theorem kernelSubobjectIsoComp_inv_arrow {X' : C} (f : X' ⟶ X) [IsIso f] (g : X ⟶ Y) [HasKernel g] :
    (kernelSubobjectIsoComp f g).inv ≫ (kernelSubobject (f ≫ g)).arrow =
      (kernelSubobject g).arrow ≫ inv f := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    X Y : C
    inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
    X' : C
    f : Quiver.Hom X' X
    inst✝¹ : CategoryTheory.IsIso f
    g : Quiver.Hom X Y
    inst✝ : CategoryTheory.Limits.HasKernel g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernelSubobjec …
  -/
  simp [kernelSubobjectIsoComp]
  /-
    🎉 no goals
  -/


/-- The kernel of `f` is always a smaller subobject than the kernel of `f ≫ h`. -/
theorem kernelSubobject_comp_le (f : X ⟶ Y) [HasKernel f] {Z : C} (h : Y ⟶ Z) [HasKernel (f ≫ h)] :
    kernelSubobject f ≤ kernelSubobject (f ≫ h) :=
                             /-
                               C : Type u
                               inst✝³ : CategoryTheory.Category.{v, u} C
                               X Y : C
                               inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
                               f : Quiver.Hom X Y
                               inst✝¹ : CategoryTheory.Limits.HasKernel f
                               Z : C
                               h : Quiver.Hom Y Z
                               inst✝ : CategoryTheory.Limits.HasKernel (CategoryTheory.CategoryStruct.comp f h)
                               ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernelSubobjec …
                             -/
  le_kernelSubobject _ _ (by simp)
                             /-
                               🎉 no goals
                             -/


/-- Postcomposing by a monomorphism does not change the kernel subobject. -/
@[simp]
theorem kernelSubobject_comp_mono (f : X ⟶ Y) [HasKernel f] {Z : C} (h : Y ⟶ Z) [Mono h] :
    kernelSubobject (f ≫ h) = kernelSubobject f :=
                                                              /-
                                                                C : Type u
                                                                inst✝³ : CategoryTheory.Category.{v, u} C
                                                                X Y : C
                                                                inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
                                                                f : Quiver.Hom X Y
                                                                inst✝¹ : CategoryTheory.Limits.HasKernel f
                                                                Z : C
                                                                h : Quiver.Hom Y Z
                                                                inst✝ : CategoryTheory.Mono h
                                                                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
                                                              -/
  le_antisymm (le_kernelSubobject _ _ ((cancel_mono h).mp (by simp))) (kernelSubobject_comp_le f h)
                                                              /-
                                                                🎉 no goals
                                                              -/


instance kernelSubobject_comp_mono_isIso (f : X ⟶ Y) [HasKernel f] {Z : C} (h : Y ⟶ Z) [Mono h] :
    IsIso (Subobject.ofLE _ _ (kernelSubobject_comp_le f h)) := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    X Y Z✝ : C
    inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
    f✝ : Quiver.Hom X Y
    inst✝² : CategoryTheory.Limits.HasKernel f✝
    f : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.HasKernel f
    Z : C
    h : Quiver.Hom Y Z
    inst✝ : CategoryTheory.Mono h
    ⊢ CategoryTheory.IsIso ((CategoryTheory.Limits.kernelSubobject f).ofLE (Catego …
  -/
  rw [ofLE_mk_le_mk_of_comm (kernelCompMono f h).inv]
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      X Y Z✝ : C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      f✝ : Quiver.Hom X Y
      inst✝² : CategoryTheory.Limits.HasKernel f✝
      f : Quiver.Hom X Y
      inst✝¹ : CategoryTheory.Limits.HasKernel f
      Z : C
      h : Quiver.Hom Y Z
      inst✝ : CategoryTheory.Mono h
      ⊢ CategoryTheory.IsIso (CategoryTheory.CategoryStruct.comp (CategoryTheory.Sub …
    -/
  · infer_instance
    /-
      🎉 no goals
    -/
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      X Y Z✝ : C
      inst✝³ : CategoryTheory.Limits.HasZeroMorphisms C
      f✝ : Quiver.Hom X Y
      inst✝² : CategoryTheory.Limits.HasKernel f✝
      f : Quiver.Hom X Y
      inst✝¹ : CategoryTheory.Limits.HasKernel f
      Z : C
      h : Quiver.Hom Y Z
      inst✝ : CategoryTheory.Mono h
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernelCompMono …
    -/
  · simp
    /-
      🎉 no goals
    -/


/-- Taking cokernels is an order-reversing map from the subobjects of `X` to the quotient objects
    of `X`. -/
@[simps]
def cokernelOrderHom [HasCokernels C] (X : C) : Subobject X →o (Subobject (op X))ᵒᵈ where
  toFun :=
    Subobject.lift (fun _ f _ => Subobject.mk (cokernel.π f).op)
      (by
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          X✝ Y Z : C
          inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
          f : Quiver.Hom X✝ Y
          inst✝¹ : CategoryTheory.Limits.HasKernel f
          inst✝ : CategoryTheory.Limits.HasCokernels C
          X : C
          ⊢ ∀ ⦃A B : C⦄ (f : Quiver.Hom A X) (g : Quiver.Hom B X) [inst : CategoryTheory …
        -/
        rintro A B f g hf hg i rfl
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          X✝ Y Z : C
          inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
          f : Quiver.Hom X✝ Y
          inst✝¹ : CategoryTheory.Limits.HasKernel f
          inst✝ : CategoryTheory.Limits.HasCokernels C
          X A B : C
          g : Quiver.Hom B X
          hg : CategoryTheory.Mono g
          i : CategoryTheory.Iso A B
          hf : CategoryTheory.Mono (CategoryTheory.CategoryStruct.comp i.hom g)
          ⊢ Eq ((fun x f x_1 => CategoryTheory.Subobject.mk (CategoryTheory.Limits.coker …
        -/
        refine Subobject.mk_eq_mk_of_comm _ _ (Iso.op ?_) (Quiver.Hom.unop_inj ?_)
        · exact (IsColimit.coconePointUniqueUpToIso (colimit.isColimit _)
            (isCokernelEpiComp (colimit.isColimit _) i.hom rfl)).symm
        · simp only [Iso.comp_inv_eq, Iso.op_hom, Iso.symm_hom, unop_comp, Quiver.Hom.unop_op,
            colimit.comp_coconePointUniqueUpToIso_hom, Cofork.ofπ_ι_app,
            coequalizer.cofork_π])
  monotone' :=
    Subobject.ind₂ _ <| by
      /-
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        X✝ Y Z : C
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
        f : Quiver.Hom X✝ Y
        inst✝¹ : CategoryTheory.Limits.HasKernel f
        inst✝ : CategoryTheory.Limits.HasCokernels C
        X : C
        ⊢ ∀ ⦃A B : C⦄ (f : Quiver.Hom A X) (g : Quiver.Hom B X) [inst : CategoryTheory …
      -/
      intro A B f g hf hg h
      /-
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        X✝ Y Z : C
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
        f✝ : Quiver.Hom X✝ Y
        inst✝¹ : CategoryTheory.Limits.HasKernel f✝
        inst✝ : CategoryTheory.Limits.HasCokernels C
        X A B : C
        f : Quiver.Hom A X
        g : Quiver.Hom B X
        hf : CategoryTheory.Mono f
        hg : CategoryTheory.Mono g
        h : LE.le (CategoryTheory.Subobject.mk f) (CategoryTheory.Subobject.mk g)
        ⊢ LE.le (CategoryTheory.Subobject.lift (fun x f x_1 => CategoryTheory.Subobjec …
      -/
      dsimp only [Subobject.lift_mk]
      /-
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        X✝ Y Z : C
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
        f✝ : Quiver.Hom X✝ Y
        inst✝¹ : CategoryTheory.Limits.HasKernel f✝
        inst✝ : CategoryTheory.Limits.HasCokernels C
        X A B : C
        f : Quiver.Hom A X
        g : Quiver.Hom B X
        hf : CategoryTheory.Mono f
        hg : CategoryTheory.Mono g
        h : LE.le (CategoryTheory.Subobject.mk f) (CategoryTheory.Subobject.mk g)
        ⊢ LE.le (CategoryTheory.Subobject.mk (CategoryTheory.Limits.cokernel.π f).op)  …
      -/
      refine Subobject.mk_le_mk_of_comm (cokernel.desc f (cokernel.π g) ?_).op ?_
        /-
          case refine_1
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          X✝ Y Z : C
          inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
          f✝ : Quiver.Hom X✝ Y
          inst✝¹ : CategoryTheory.Limits.HasKernel f✝
          inst✝ : CategoryTheory.Limits.HasCokernels C
          X A B : C
          f : Quiver.Hom A X
          g : Quiver.Hom B X
          hf : CategoryTheory.Mono f
          hg : CategoryTheory.Mono g
          h : LE.le (CategoryTheory.Subobject.mk f) (CategoryTheory.Subobject.mk g)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.cokernel.π g …
        -/
      · rw [← Subobject.ofMkLEMk_comp h, Category.assoc, cokernel.condition, comp_zero]
        /-
          🎉 no goals
        -/
        /-
          case refine_2
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          X✝ Y Z : C
          inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
          f✝ : Quiver.Hom X✝ Y
          inst✝¹ : CategoryTheory.Limits.HasKernel f✝
          inst✝ : CategoryTheory.Limits.HasCokernels C
          X A B : C
          f : Quiver.Hom A X
          g : Quiver.Hom B X
          hf : CategoryTheory.Mono f
          hg : CategoryTheory.Mono g
          h : LE.le (CategoryTheory.Subobject.mk f) (CategoryTheory.Subobject.mk g)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.cokernel.desc  …
        -/
      · exact Quiver.Hom.unop_inj (cokernel.π_desc _ _ _)
        /-
          🎉 no goals
        -/


/-- Taking kernels is an order-reversing map from the quotient objects of `X` to the subobjects of
    `X`. -/
@[simps]
def kernelOrderHom [HasKernels C] (X : C) : (Subobject (op X))ᵒᵈ →o Subobject X where
  toFun :=
    Subobject.lift (fun _ f _ => Subobject.mk (kernel.ι f.unop))
      (by
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          X✝ Y Z : C
          inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
          f : Quiver.Hom X✝ Y
          inst✝¹ : CategoryTheory.Limits.HasKernel f
          inst✝ : CategoryTheory.Limits.HasKernels C
          X : C
          ⊢ ∀ ⦃A B : Opposite C⦄ (f : Quiver.Hom A { unop := X }) (g : Quiver.Hom B { un …
        -/
        rintro A B f g hf hg i rfl
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          X✝ Y Z : C
          inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
          f : Quiver.Hom X✝ Y
          inst✝¹ : CategoryTheory.Limits.HasKernel f
          inst✝ : CategoryTheory.Limits.HasKernels C
          X : C
          A B : Opposite C
          g : Quiver.Hom B { unop := X }
          hg : CategoryTheory.Mono g
          i : CategoryTheory.Iso A B
          hf : CategoryTheory.Mono (CategoryTheory.CategoryStruct.comp i.hom g)
          ⊢ Eq ((fun x f x_1 => CategoryTheory.Subobject.mk (CategoryTheory.Limits.kerne …
        -/
        refine Subobject.mk_eq_mk_of_comm _ _ ?_ ?_
        · exact
            IsLimit.conePointUniqueUpToIso (limit.isLimit _)
              (isKernelCompMono (limit.isLimit (parallelPair g.unop 0)) i.unop.hom rfl)
          /-
            case refine_2
            C : Type u
            inst✝³ : CategoryTheory.Category.{v, u} C
            X✝ Y Z : C
            inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
            f : Quiver.Hom X✝ Y
            inst✝¹ : CategoryTheory.Limits.HasKernel f
            inst✝ : CategoryTheory.Limits.HasKernels C
            X : C
            A B : Opposite C
            g : Quiver.Hom B { unop := X }
            hg : CategoryTheory.Mono g
            i : CategoryTheory.Iso A B
            hf : CategoryTheory.Mono (CategoryTheory.CategoryStruct.comp i.hom g)
            ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.limit.isLimit …
          -/
        · dsimp
          simp only [← Iso.eq_inv_comp, limit.conePointUniqueUpToIso_inv_comp,
            Fork.ofι_π_app])
  monotone' :=
    Subobject.ind₂ _ <| by
      /-
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        X✝ Y Z : C
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
        f : Quiver.Hom X✝ Y
        inst✝¹ : CategoryTheory.Limits.HasKernel f
        inst✝ : CategoryTheory.Limits.HasKernels C
        X : C
        ⊢ ∀ ⦃A B : Opposite C⦄ (f : Quiver.Hom A { unop := X }) (g : Quiver.Hom B { un …
      -/
      intro A B f g hf hg h
      /-
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        X✝ Y Z : C
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
        f✝ : Quiver.Hom X✝ Y
        inst✝¹ : CategoryTheory.Limits.HasKernel f✝
        inst✝ : CategoryTheory.Limits.HasKernels C
        X : C
        A B : Opposite C
        f : Quiver.Hom A { unop := X }
        g : Quiver.Hom B { unop := X }
        hf : CategoryTheory.Mono f
        hg : CategoryTheory.Mono g
        h : LE.le (CategoryTheory.Subobject.mk f) (CategoryTheory.Subobject.mk g)
        ⊢ LE.le (CategoryTheory.Subobject.lift (fun x f x_1 => CategoryTheory.Subobjec …
      -/
      dsimp only [Subobject.lift_mk]
      /-
        C : Type u
        inst✝³ : CategoryTheory.Category.{v, u} C
        X✝ Y Z : C
        inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
        f✝ : Quiver.Hom X✝ Y
        inst✝¹ : CategoryTheory.Limits.HasKernel f✝
        inst✝ : CategoryTheory.Limits.HasKernels C
        X : C
        A B : Opposite C
        f : Quiver.Hom A { unop := X }
        g : Quiver.Hom B { unop := X }
        hf : CategoryTheory.Mono f
        hg : CategoryTheory.Mono g
        h : LE.le (CategoryTheory.Subobject.mk f) (CategoryTheory.Subobject.mk g)
        ⊢ LE.le (CategoryTheory.Subobject.mk (CategoryTheory.Limits.kernel.ι f.unop))  …
      -/
      refine Subobject.mk_le_mk_of_comm (kernel.lift g.unop (kernel.ι f.unop) ?_) ?_
        /-
          case refine_1
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          X✝ Y Z : C
          inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
          f✝ : Quiver.Hom X✝ Y
          inst✝¹ : CategoryTheory.Limits.HasKernel f✝
          inst✝ : CategoryTheory.Limits.HasKernels C
          X : C
          A B : Opposite C
          f : Quiver.Hom A { unop := X }
          g : Quiver.Hom B { unop := X }
          hf : CategoryTheory.Mono f
          hg : CategoryTheory.Mono g
          h : LE.le (CategoryTheory.Subobject.mk f) (CategoryTheory.Subobject.mk g)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernel.ι f.uno …
        -/
      · rw [← Subobject.ofMkLEMk_comp h, unop_comp, kernel.condition_assoc, zero_comp]
        /-
          🎉 no goals
        -/
        /-
          case refine_2
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          X✝ Y Z : C
          inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
          f✝ : Quiver.Hom X✝ Y
          inst✝¹ : CategoryTheory.Limits.HasKernel f✝
          inst✝ : CategoryTheory.Limits.HasKernels C
          X : C
          A B : Opposite C
          f : Quiver.Hom A { unop := X }
          g : Quiver.Hom B { unop := X }
          hf : CategoryTheory.Mono f
          hg : CategoryTheory.Mono g
          h : LE.le (CategoryTheory.Subobject.mk f) (CategoryTheory.Subobject.mk g)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.kernel.lift g. …
        -/
      · exact Quiver.Hom.op_inj (by simp)
        /-
          🎉 no goals
        -/


/-- The image of a morphism `f g : X ⟶ Y` as a `Subobject Y`. -/
abbrev imageSubobject : Subobject Y :=
  Subobject.mk (image.ι f)


/-- The underlying object of `imageSubobject f` is (up to isomorphism!)
the same as the chosen object `image f`. -/
def imageSubobjectIso : (imageSubobject f : C) ≅ image f :=
  Subobject.underlyingIso (image.ι f)


@[reassoc (attr := simp)]
theorem imageSubobject_arrow :
                                                                           /-
                                                                             C : Type u
                                                                             inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                             X Y : C
                                                                             f : Quiver.Hom X Y
                                                                             inst✝ : CategoryTheory.Limits.HasImage f
                                                                             ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.imageSubobject …
                                                                           -/
    (imageSubobjectIso f).hom ≫ image.ι f = (imageSubobject f).arrow := by simp [imageSubobjectIso]
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


@[reassoc (attr := simp)]
theorem imageSubobject_arrow' :
                                                                           /-
                                                                             C : Type u
                                                                             inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                             X Y : C
                                                                             f : Quiver.Hom X Y
                                                                             inst✝ : CategoryTheory.Limits.HasImage f
                                                                             ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.imageSubobject …
                                                                           -/
    (imageSubobjectIso f).inv ≫ (imageSubobject f).arrow = image.ι f := by simp [imageSubobjectIso]
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


/-- A factorisation of `f : X ⟶ Y` through `imageSubobject f`. -/
def factorThruImageSubobject : X ⟶ imageSubobject f :=
  factorThruImage f ≫ (imageSubobjectIso f).inv


instance [HasEqualizers C] : Epi (factorThruImageSubobject f) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.HasImage f
    inst✝ : CategoryTheory.Limits.HasEqualizers C
    ⊢ CategoryTheory.Epi (CategoryTheory.Limits.factorThruImageSubobject f)
  -/
  dsimp [factorThruImageSubobject]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.HasImage f
    inst✝ : CategoryTheory.Limits.HasEqualizers C
    ⊢ CategoryTheory.Epi (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limit …
  -/
  apply epi_comp
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp), elementwise (attr := simp)]
theorem imageSubobject_arrow_comp : factorThruImageSubobject f ≫ (imageSubobject f).arrow = f := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Limits.HasImage f
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.factorThruImag …
  -/
  simp [factorThruImageSubobject, imageSubobject_arrow]
  /-
    🎉 no goals
  -/


theorem imageSubobject_arrow_comp_eq_zero [HasZeroMorphisms C] {X Y Z : C} {f : X ⟶ Y} {g : Y ⟶ Z}
    [HasImage f] [Epi (factorThruImageSubobject f)] (h : f ≫ g = 0) :
    (imageSubobject f).arrow ≫ g = 0 :=
                                                      /-
                                                        C : Type u
                                                        inst✝³ : CategoryTheory.Category.{v, u} C
                                                        inst✝² : CategoryTheory.Limits.HasZeroMorphisms C
                                                        X Y Z : C
                                                        f : Quiver.Hom X Y
                                                        g : Quiver.Hom Y Z
                                                        inst✝¹ : CategoryTheory.Limits.HasImage f
                                                        inst✝ : CategoryTheory.Epi (CategoryTheory.Limits.factorThruImageSubobject f)
                                                        h : Eq (CategoryTheory.CategoryStruct.comp f g) 0
                                                        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.factorThruImag …
                                                      -/
  zero_of_epi_comp (factorThruImageSubobject f) <| by simp [h]
                                                      /-
                                                        🎉 no goals
                                                      -/


theorem imageSubobject_factors_comp_self {W : C} (k : W ⟶ X) : (imageSubobject f).Factors (k ≫ f) :=
                             /-
                               C : Type u
                               inst✝¹ : CategoryTheory.Category.{v, u} C
                               X Y : C
                               f : Quiver.Hom X Y
                               inst✝ : CategoryTheory.Limits.HasImage f
                               W : C
                               k : Quiver.Hom W X
                               ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp k …
                             -/
  ⟨k ≫ factorThruImage f, by simp⟩
                             /-
                               🎉 no goals
                             -/


@[simp]
theorem factorThruImageSubobject_comp_self {W : C} (k : W ⟶ X) (h) :
    (imageSubobject f).factorThru (k ≫ f) h = k ≫ factorThruImageSubobject f := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Limits.HasImage f
    W : C
    k : Quiver.Hom W X
    h : (CategoryTheory.Limits.imageSubobject f).Factors (CategoryTheory.CategoryS …
    ⊢ Eq ((CategoryTheory.Limits.imageSubobject f).factorThru (CategoryTheory.Cate …
  -/
  ext
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Limits.HasImage f
    W : C
    k : Quiver.Hom W X
    h : (CategoryTheory.Limits.imageSubobject f).Factors (CategoryTheory.CategoryS …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.imageSubobjec …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem factorThruImageSubobject_comp_self_assoc {W W' : C} (k : W ⟶ W') (k' : W' ⟶ X) (h) :
    (imageSubobject f).factorThru (k ≫ k' ≫ f) h = k ≫ k' ≫ factorThruImageSubobject f := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Limits.HasImage f
    W W' : C
    k : Quiver.Hom W W'
    k' : Quiver.Hom W' X
    h : (CategoryTheory.Limits.imageSubobject f).Factors (CategoryTheory.CategoryS …
    ⊢ Eq ((CategoryTheory.Limits.imageSubobject f).factorThru (CategoryTheory.Cate …
  -/
  ext
  /-
    case h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y : C
    f : Quiver.Hom X Y
    inst✝ : CategoryTheory.Limits.HasImage f
    W W' : C
    k : Quiver.Hom W W'
    k' : Quiver.Hom W' X
    h : (CategoryTheory.Limits.imageSubobject f).Factors (CategoryTheory.CategoryS …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.imageSubobjec …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The image of `h ≫ f` is always a smaller subobject than the image of `f`. -/
theorem imageSubobject_comp_le {X' : C} (h : X' ⟶ X) (f : X ⟶ Y) [HasImage f] [HasImage (h ≫ f)] :
    imageSubobject (h ≫ f) ≤ imageSubobject f :=
                                                     /-
                                                       C : Type u
                                                       inst✝² : CategoryTheory.Category.{v, u} C
                                                       X Y X' : C
                                                       h : Quiver.Hom X' X
                                                       f : Quiver.Hom X Y
                                                       inst✝¹ : CategoryTheory.Limits.HasImage f
                                                       inst✝ : CategoryTheory.Limits.HasImage (CategoryTheory.CategoryStruct.comp h f)
                                                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.image.preComp  …
                                                     -/
  Subobject.mk_le_mk_of_comm (image.preComp h f) (by simp)
                                                     /-
                                                       🎉 no goals
                                                     -/


@[simp]
theorem imageSubobject_zero_arrow : (imageSubobject (0 : X ⟶ Y)).arrow = 0 := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    ⊢ Eq (CategoryTheory.Limits.imageSubobject 0).arrow 0
  -/
  rw [← imageSubobject_arrow]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
    inst✝ : CategoryTheory.Limits.HasZeroObject C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.imageSubobject …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem imageSubobject_zero {A B : C} : imageSubobject (0 : A ⟶ B) = ⊥ :=
                                                                                              /-
                                                                                                C : Type u
                                                                                                inst✝² : CategoryTheory.Category.{v, u} C
                                                                                                inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
                                                                                                inst✝ : CategoryTheory.Limits.HasZeroObject C
                                                                                                A B : C
                                                                                                ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.imageSubobjec …
                                                                                              -/
  Subobject.eq_of_comm (imageSubobjectIso _ ≪≫ imageZero ≪≫ Subobject.botCoeIsoZero.symm) (by simp)
                                                                                              /-
                                                                                                🎉 no goals
                                                                                              -/


/-- The morphism `imageSubobject (h ≫ f) ⟶ imageSubobject f`
is an epimorphism when `h` is an epimorphism.
In general this does not imply that `imageSubobject (h ≫ f) = imageSubobject f`,
although it will when the ambient category is abelian.
 -/
instance imageSubobject_comp_le_epi_of_epi {X' : C} (h : X' ⟶ X) [Epi h] (f : X ⟶ Y) [HasImage f]
    [HasImage (h ≫ f)] : Epi (Subobject.ofLE _ _ (imageSubobject_comp_le h f)) := by
  /-
    C : Type u
    inst✝⁵ : CategoryTheory.Category.{v, u} C
    X Y Z : C
    f✝ : Quiver.Hom X Y
    inst✝⁴ : CategoryTheory.Limits.HasImage f✝
    inst✝³ : CategoryTheory.Limits.HasEqualizers C
    X' : C
    h : Quiver.Hom X' X
    inst✝² : CategoryTheory.Epi h
    f : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.HasImage f
    inst✝ : CategoryTheory.Limits.HasImage (CategoryTheory.CategoryStruct.comp h f)
    ⊢ CategoryTheory.Epi ((CategoryTheory.Limits.imageSubobject (CategoryTheory.Ca …
  -/
  rw [ofLE_mk_le_mk_of_comm (image.preComp h f)]
    /-
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      X Y Z : C
      f✝ : Quiver.Hom X Y
      inst✝⁴ : CategoryTheory.Limits.HasImage f✝
      inst✝³ : CategoryTheory.Limits.HasEqualizers C
      X' : C
      h : Quiver.Hom X' X
      inst✝² : CategoryTheory.Epi h
      f : Quiver.Hom X Y
      inst✝¹ : CategoryTheory.Limits.HasImage f
      inst✝ : CategoryTheory.Limits.HasImage (CategoryTheory.CategoryStruct.comp h f)
      ⊢ CategoryTheory.Epi (CategoryTheory.CategoryStruct.comp (CategoryTheory.Subob …
    -/
  · infer_instance
    /-
      🎉 no goals
    -/
    /-
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      X Y Z : C
      f✝ : Quiver.Hom X Y
      inst✝⁴ : CategoryTheory.Limits.HasImage f✝
      inst✝³ : CategoryTheory.Limits.HasEqualizers C
      X' : C
      h : Quiver.Hom X' X
      inst✝² : CategoryTheory.Epi h
      f : Quiver.Hom X Y
      inst✝¹ : CategoryTheory.Limits.HasImage f
      inst✝ : CategoryTheory.Limits.HasImage (CategoryTheory.CategoryStruct.comp h f)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.image.preComp  …
    -/
  · simp
    /-
      🎉 no goals
    -/


/-- Postcomposing by an isomorphism gives an isomorphism between image subobjects. -/
def imageSubobjectCompIso (f : X ⟶ Y) [HasImage f] {Y' : C} (h : Y ⟶ Y') [IsIso h] :
    (imageSubobject (f ≫ h) : C) ≅ (imageSubobject f : C) :=
  imageSubobjectIso _ ≪≫ (image.compIso _ _).symm ≪≫ (imageSubobjectIso _).symm


@[reassoc (attr := simp)]
theorem imageSubobjectCompIso_hom_arrow (f : X ⟶ Y) [HasImage f] {Y' : C} (h : Y ⟶ Y') [IsIso h] :
    (imageSubobjectCompIso f h).hom ≫ (imageSubobject f).arrow =
      (imageSubobject (f ≫ h)).arrow ≫ inv h := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    X Y : C
    inst✝² : CategoryTheory.Limits.HasEqualizers C
    f : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.HasImage f
    Y' : C
    h : Quiver.Hom Y Y'
    inst✝ : CategoryTheory.IsIso h
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.imageSubobject …
  -/
  simp [imageSubobjectCompIso]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem imageSubobjectCompIso_inv_arrow (f : X ⟶ Y) [HasImage f] {Y' : C} (h : Y ⟶ Y') [IsIso h] :
    (imageSubobjectCompIso f h).inv ≫ (imageSubobject (f ≫ h)).arrow =
      (imageSubobject f).arrow ≫ h := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    X Y : C
    inst✝² : CategoryTheory.Limits.HasEqualizers C
    f : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.HasImage f
    Y' : C
    h : Quiver.Hom Y Y'
    inst✝ : CategoryTheory.IsIso h
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.imageSubobject …
  -/
  simp [imageSubobjectCompIso]
  /-
    🎉 no goals
  -/


theorem imageSubobject_mono (f : X ⟶ Y) [Mono f] : imageSubobject f = Subobject.mk f :=
                                                                                         /-
                                                                                           C : Type u
                                                                                           inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                                           X Y : C
                                                                                           f : Quiver.Hom X Y
                                                                                           inst✝ : CategoryTheory.Mono f
                                                                                           ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.imageSubobjec …
                                                                                         -/
  eq_of_comm (imageSubobjectIso f ≪≫ imageMonoIsoSource f ≪≫ (underlyingIso f).symm) (by simp)
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/


/-- Precomposing by an isomorphism does not change the image subobject. -/
theorem imageSubobject_iso_comp [HasEqualizers C] {X' : C} (h : X' ⟶ X) [IsIso h] (f : X ⟶ Y)
    [HasImage f] : imageSubobject (h ≫ f) = imageSubobject f :=
  le_antisymm (imageSubobject_comp_le h f)
                                                              /-
                                                                C : Type u
                                                                inst✝³ : CategoryTheory.Category.{v, u} C
                                                                X Y : C
                                                                inst✝² : CategoryTheory.Limits.HasEqualizers C
                                                                X' : C
                                                                h : Quiver.Hom X' X
                                                                inst✝¹ : CategoryTheory.IsIso h
                                                                f : Quiver.Hom X Y
                                                                inst✝ : CategoryTheory.Limits.HasImage f
                                                                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.inv (CategoryTheory.L …
                                                              -/
    (Subobject.mk_le_mk_of_comm (inv (image.preComp h f)) (by simp))
                                                              /-
                                                                🎉 no goals
                                                              -/


theorem imageSubobject_le {A B : C} {X : Subobject B} (f : A ⟶ B) [HasImage f] (h : A ⟶ X)
    (w : h ≫ X.arrow = f) : imageSubobject f ≤ X :=
  Subobject.le_of_comm
    ((imageSubobjectIso f).hom ≫
      image.lift
        { I := (X : C)
          e := h
          m := X.arrow })
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          A B : C
          X : CategoryTheory.Subobject B
          f : Quiver.Hom A B
          inst✝ : CategoryTheory.Limits.HasImage f
          h : Quiver.Hom A (CategoryTheory.Subobject.underlying.obj X)
          w : Eq (CategoryTheory.CategoryStruct.comp h X.arrow) f
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
    (by rw [assoc, image.lift_fac, imageSubobject_arrow])
        /-
          🎉 no goals
        -/


theorem imageSubobject_le_mk {A B : C} {X : C} (g : X ⟶ B) [Mono g] (f : A ⟶ B) [HasImage f]
    (h : A ⟶ X) (w : h ≫ g = f) : imageSubobject f ≤ Subobject.mk g :=
                                                                /-
                                                                  C : Type u
                                                                  inst✝² : CategoryTheory.Category.{v, u} C
                                                                  A B X : C
                                                                  g : Quiver.Hom X B
                                                                  inst✝¹ : CategoryTheory.Mono g
                                                                  f : Quiver.Hom A B
                                                                  inst✝ : CategoryTheory.Limits.HasImage f
                                                                  h : Quiver.Hom A X
                                                                  w : Eq (CategoryTheory.CategoryStruct.comp h g) f
                                                                  ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp h …
                                                                -/
  imageSubobject_le f (h ≫ (Subobject.underlyingIso g).inv) (by simp [w])
                                                                /-
                                                                  🎉 no goals
                                                                -/


/-- Given a commutative square between morphisms `f` and `g`,
we have a morphism in the category from `imageSubobject f` to `imageSubobject g`. -/
def imageSubobjectMap {W X Y Z : C} {f : W ⟶ X} [HasImage f] {g : Y ⟶ Z} [HasImage g]
    (sq : Arrow.mk f ⟶ Arrow.mk g) [HasImageMap sq] :
    (imageSubobject f : C) ⟶ (imageSubobject g : C) :=
  (imageSubobjectIso f).hom ≫ image.map sq ≫ (imageSubobjectIso g).inv


@[reassoc (attr := simp)]
theorem imageSubobjectMap_arrow {W X Y Z : C} {f : W ⟶ X} [HasImage f] {g : Y ⟶ Z} [HasImage g]
    (sq : Arrow.mk f ⟶ Arrow.mk g) [HasImageMap sq] :
    imageSubobjectMap sq ≫ (imageSubobject g).arrow = (imageSubobject f).arrow ≫ sq.right := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    W X Y Z : C
    f : Quiver.Hom W X
    inst✝² : CategoryTheory.Limits.HasImage f
    g : Quiver.Hom Y Z
    inst✝¹ : CategoryTheory.Limits.HasImage g
    sq : Quiver.Hom (CategoryTheory.Arrow.mk f) (CategoryTheory.Arrow.mk g)
    inst✝ : CategoryTheory.Limits.HasImageMap sq
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.imageSubobject …
  -/
  simp only [imageSubobjectMap, Category.assoc, imageSubobject_arrow']
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    W X Y Z : C
    f : Quiver.Hom W X
    inst✝² : CategoryTheory.Limits.HasImage f
    g : Quiver.Hom Y Z
    inst✝¹ : CategoryTheory.Limits.HasImage g
    sq : Quiver.Hom (CategoryTheory.Arrow.mk f) (CategoryTheory.Arrow.mk g)
    inst✝ : CategoryTheory.Limits.HasImageMap sq
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.imageSubobject …
  -/
  erw [image.map_ι, ← Category.assoc, imageSubobject_arrow]
  /-
    🎉 no goals
  -/


theorem image_map_comp_imageSubobjectIso_inv {W X Y Z : C} {f : W ⟶ X} [HasImage f] {g : Y ⟶ Z}
    [HasImage g] (sq : Arrow.mk f ⟶ Arrow.mk g) [HasImageMap sq] :
    image.map sq ≫ (imageSubobjectIso _).inv =
      (imageSubobjectIso _).inv ≫ imageSubobjectMap sq := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    W X Y Z : C
    f : Quiver.Hom W X
    inst✝² : CategoryTheory.Limits.HasImage f
    g : Quiver.Hom Y Z
    inst✝¹ : CategoryTheory.Limits.HasImage g
    sq : Quiver.Hom (CategoryTheory.Arrow.mk f) (CategoryTheory.Arrow.mk g)
    inst✝ : CategoryTheory.Limits.HasImageMap sq
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.image.map sq)  …
  -/
  ext
  /-
    case h
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    W X Y Z : C
    f : Quiver.Hom W X
    inst✝² : CategoryTheory.Limits.HasImage f
    g : Quiver.Hom Y Z
    inst✝¹ : CategoryTheory.Limits.HasImage g
    sq : Quiver.Hom (CategoryTheory.Arrow.mk f) (CategoryTheory.Arrow.mk g)
    inst✝ : CategoryTheory.Limits.HasImageMap sq
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simpa using image.map_ι sq
  /-
    🎉 no goals
  -/


theorem imageSubobjectIso_comp_image_map {W X Y Z : C} {f : W ⟶ X} [HasImage f] {g : Y ⟶ Z}
    [HasImage g] (sq : Arrow.mk f ⟶ Arrow.mk g) [HasImageMap sq] :
    (imageSubobjectIso _).hom ≫ image.map sq =
      imageSubobjectMap sq ≫ (imageSubobjectIso _).hom := by
  erw [← Iso.comp_inv_eq, Category.assoc, ← (imageSubobjectIso f).eq_inv_comp,
    image_map_comp_imageSubobjectIso_inv sq]


