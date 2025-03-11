/-- A category is called preadditive if `P ⟶ Q` is an abelian group such that composition is
    linear in both variables. -/
class Preadditive where
  homGroup : ∀ P Q : C, AddCommGroup (P ⟶ Q) := by infer_instance
  add_comp : ∀ (P Q R : C) (f f' : P ⟶ Q) (g : Q ⟶ R), (f + f') ≫ g = f ≫ g + f' ≫ g := by
    aesop_cat
  comp_add : ∀ (P Q R : C) (f : P ⟶ Q) (g g' : Q ⟶ R), f ≫ (g + g') = f ≫ g + f ≫ g' := by
    aesop_cat


attribute [reassoc, simp] Preadditive.add_comp


attribute [reassoc] Preadditive.comp_add

-- (the linter doesn't like `simp` on this lemma)

instance inducedCategory : Preadditive.{v} (InducedCategory C F) where
  homGroup P Q := @Preadditive.homGroup C _ _ (F P) (F Q)
  add_comp _ _ _ _ _ _ := add_comp _ _ _ _ _ _
  comp_add _ _ _ _ _ _ := comp_add _ _ _ _ _ _


instance fullSubcategory (Z : C → Prop) : Preadditive.{v} (FullSubcategory Z) where
  homGroup P Q := @Preadditive.homGroup C _ _ P.obj Q.obj
  add_comp _ _ _ _ _ _ := add_comp _ _ _ _ _ _
  comp_add _ _ _ _ _ _ := comp_add _ _ _ _ _ _


instance (X : C) : AddCommGroup (End X) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    X : C
    ⊢ AddCommGroup (CategoryTheory.End X)
  -/
  dsimp [End]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    X : C
    ⊢ AddCommGroup (Quiver.Hom X X)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- Composition by a fixed left argument as a group homomorphism -/
def leftComp {P Q : C} (R : C) (f : P ⟶ Q) : (Q ⟶ R) →+ (P ⟶ R) :=
                                      /-
                                        C : Type u
                                        inst✝¹ : CategoryTheory.Category.{v, u} C
                                        inst✝ : CategoryTheory.Preadditive C
                                        P Q R : C
                                        f : Quiver.Hom P Q
                                        g g' : Quiver.Hom Q R
                                        ⊢ Eq ((fun g => CategoryTheory.CategoryStruct.comp f g) (HAdd.hAdd g g')) (HAd …
                                      -/
  mk' (fun g => f ≫ g) fun g g' => by simp
                                      /-
                                        🎉 no goals
                                      -/


/-- Composition by a fixed right argument as a group homomorphism -/
def rightComp (P : C) {Q R : C} (g : Q ⟶ R) : (P ⟶ Q) →+ (P ⟶ R) :=
                                      /-
                                        C : Type u
                                        inst✝¹ : CategoryTheory.Category.{v, u} C
                                        inst✝ : CategoryTheory.Preadditive C
                                        P Q R : C
                                        g : Quiver.Hom Q R
                                        f f' : Quiver.Hom P Q
                                        ⊢ Eq ((fun f => CategoryTheory.CategoryStruct.comp f g) (HAdd.hAdd f f')) (HAd …
                                      -/
  mk' (fun f => f ≫ g) fun f f' => by simp
                                      /-
                                        🎉 no goals
                                      -/


/-- Composition as a bilinear group homomorphism -/
def compHom : (P ⟶ Q) →+ (Q ⟶ R) →+ (P ⟶ R) :=
  AddMonoidHom.mk' (fun f => leftComp _ f) fun f₁ f₂ =>
    AddMonoidHom.ext fun g => (rightComp _ g).map_add f₁ f₂

-- Porting note: simp can prove the reassoc version

@[reassoc, simp]
theorem sub_comp : (f - f') ≫ g = f ≫ g - f' ≫ g :=
  map_sub (rightComp P g) f f'

-- The redundant simp lemma linter says that simp can prove the reassoc version of this lemma.

@[reassoc, simp]
theorem comp_sub : f ≫ (g - g') = f ≫ g - f ≫ g' :=
  map_sub (leftComp R f) g g'

-- Porting note: simp can prove the reassoc version

@[reassoc, simp]
theorem neg_comp : (-f) ≫ g = -f ≫ g :=
  map_neg (rightComp P g) f

-- The redundant simp lemma linter says that simp can prove the reassoc version of this lemma.

@[reassoc, simp]
theorem comp_neg : f ≫ (-g) = -f ≫ g :=
  map_neg (leftComp R f) g


@[reassoc]
                                                 /-
                                                   C : Type u
                                                   inst✝¹ : CategoryTheory.Category.{v, u} C
                                                   inst✝ : CategoryTheory.Preadditive C
                                                   P Q R : C
                                                   f : Quiver.Hom P Q
                                                   g : Quiver.Hom Q R
                                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (Neg.neg f) (Neg.neg g)) (CategoryThe …
                                                 -/
theorem neg_comp_neg : (-f) ≫ (-g) = f ≫ g := by simp
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem nsmul_comp (n : ℕ) : (n • f) ≫ g = n • f ≫ g :=
  map_nsmul (rightComp P g) n f


theorem comp_nsmul (n : ℕ) : f ≫ (n • g) = n • f ≫ g :=
  map_nsmul (leftComp R f) n g


theorem zsmul_comp (n : ℤ) : (n • f) ≫ g = n • f ≫ g :=
  map_zsmul (rightComp P g) n f


theorem comp_zsmul (n : ℤ) : f ≫ (n • g) = n • f ≫ g :=
  map_zsmul (leftComp R f) n g


@[reassoc]
theorem comp_sum {P Q R : C} {J : Type*} (s : Finset J) (f : P ⟶ Q) (g : J → (Q ⟶ R)) :
    (f ≫ ∑ j ∈ s, g j) = ∑ j ∈ s, f ≫ g j :=
  map_sum (leftComp R f) _ _


@[reassoc]
theorem sum_comp {P Q R : C} {J : Type*} (s : Finset J) (f : J → (P ⟶ Q)) (g : Q ⟶ R) :
    (∑ j ∈ s, f j) ≫ g = ∑ j ∈ s, f j ≫ g :=
  map_sum (rightComp P g) _ _


instance {P Q : C} {f : P ⟶ Q} [Epi f] : Epi (-f) :=
                    /-
                      C : Type u
                      inst✝² : CategoryTheory.Category.{v, u} C
                      inst✝¹ : CategoryTheory.Preadditive C
                      P✝ Q✝ R : C
                      f✝ f' : Quiver.Hom P✝ Q✝
                      g✝ g'✝ : Quiver.Hom Q✝ R
                      P Q : C
                      f : Quiver.Hom P Q
                      inst✝ : CategoryTheory.Epi f
                      Z✝ : C
                      g g' : Quiver.Hom Q Z✝
                      H : Eq (CategoryTheory.CategoryStruct.comp (Neg.neg f) g) (CategoryTheory.Cate …
                      ⊢ Eq g g'
                    -/
  ⟨fun g g' H => by rwa [neg_comp, neg_comp, ← comp_neg, ← comp_neg, cancel_epi, neg_inj] at H⟩
                    /-
                      🎉 no goals
                    -/


instance {P Q : C} {f : P ⟶ Q} [Mono f] : Mono (-f) :=
                    /-
                      C : Type u
                      inst✝² : CategoryTheory.Category.{v, u} C
                      inst✝¹ : CategoryTheory.Preadditive C
                      P✝ Q✝ R : C
                      f✝ f' : Quiver.Hom P✝ Q✝
                      g✝ g'✝ : Quiver.Hom Q✝ R
                      P Q : C
                      f : Quiver.Hom P Q
                      inst✝ : CategoryTheory.Mono f
                      Z✝ : C
                      g g' : Quiver.Hom Z✝ P
                      H : Eq (CategoryTheory.CategoryStruct.comp g (Neg.neg f)) (CategoryTheory.Cate …
                      ⊢ Eq g g'
                    -/
  ⟨fun g g' H => by rwa [comp_neg, comp_neg, ← neg_comp, ← neg_comp, cancel_mono, neg_inj] at H⟩
                    /-
                      🎉 no goals
                    -/


instance (priority := 100) preadditiveHasZeroMorphisms : HasZeroMorphisms C where
  zero := inferInstance
  comp_zero f R := show leftComp R f 0 = 0 from map_zero _
  zero_comp P _ _ f := show rightComp P f 0 = 0 from map_zero _


/-- Porting note: adding this before the ring instance allowed moduleEndRight to find
the correct Monoid structure on End. Moved both down after preadditiveHasZeroMorphisms
to make use of them -/
instance {X : C} : Semiring (End X) :=
  { End.monoid with
                            /-
                              C : Type u
                              inst✝¹ : CategoryTheory.Category.{v, u} C
                              inst✝ : CategoryTheory.Preadditive C
                              P Q R : C
                              f✝ f' : Quiver.Hom P Q
                              g g' : Quiver.Hom Q R
                              X : C
                              f : CategoryTheory.End X
                              ⊢ Eq (HMul.hMul 0 f) 0
                            -/
    zero_mul := fun f => by dsimp [mul]; exact HasZeroMorphisms.comp_zero f _
                                         /-
                                           🎉 no goals
                                         -/
                            /-
                              C : Type u
                              inst✝¹ : CategoryTheory.Category.{v, u} C
                              inst✝ : CategoryTheory.Preadditive C
                              P Q R : C
                              f✝ f' : Quiver.Hom P Q
                              g g' : Quiver.Hom Q R
                              X : C
                              f : CategoryTheory.End X
                              ⊢ Eq (HMul.hMul f 0) 0
                            -/
    mul_zero := fun f => by dsimp [mul]; exact HasZeroMorphisms.zero_comp _ f
                                         /-
                                           🎉 no goals
                                         -/
    left_distrib := fun f g h => Preadditive.add_comp X X X g h f
    right_distrib := fun f g h => Preadditive.comp_add X X X h f g }


/-- Porting note: It looks like Ring's parent classes changed in
Lean 4 so the previous instance needed modification. Was following my nose here. -/
instance {X : C} : Ring (End X) :=
  { (inferInstance : Semiring (End X)),
    (inferInstance : AddCommGroup (End X)) with
    neg_add_cancel := neg_add_cancel }


instance moduleEndRight {X Y : C} : Module (End Y) (X ⟶ Y) where
  smul_add _ _ _ := add_comp _ _ _ _ _ _
  smul_zero _ := zero_comp
  add_smul _ _ _ := comp_add _ _ _ _ _ _
  zero_smul _ := comp_zero


theorem mono_of_cancel_zero {Q R : C} (f : Q ⟶ R) (h : ∀ {P : C} (g : P ⟶ Q), g ≫ f = 0 → g = 0) :
    Mono f where
  right_cancellation := fun {Z} g₁ g₂ hg =>
    sub_eq_zero.1 <| h _ <| (map_sub (rightComp Z f) g₁ g₂).trans <| sub_eq_zero.2 hg


theorem mono_iff_cancel_zero {Q R : C} (f : Q ⟶ R) :
    Mono f ↔ ∀ (P : C) (g : P ⟶ Q), g ≫ f = 0 → g = 0 :=
  ⟨fun _ _ _ => zero_of_comp_mono _, mono_of_cancel_zero f⟩


theorem mono_of_kernel_zero {X Y : C} {f : X ⟶ Y} [HasLimit (parallelPair f 0)]
    (w : kernel.ι f = 0) : Mono f :=
                                      /-
                                        C : Type u
                                        inst✝² : CategoryTheory.Category.{v, u} C
                                        inst✝¹ : CategoryTheory.Preadditive C
                                        X Y : C
                                        f : Quiver.Hom X Y
                                        inst✝ : CategoryTheory.Limits.HasLimit (CategoryTheory.Limits.parallelPair f 0)
                                        w : Eq (CategoryTheory.Limits.kernel.ι f) 0
                                        P✝ : C
                                        g : Quiver.Hom P✝ X
                                        h : Eq (CategoryTheory.CategoryStruct.comp g f) 0
                                        ⊢ Eq g 0
                                      -/
  mono_of_cancel_zero f fun g h => by rw [← kernel.lift_ι f g h, w, Limits.comp_zero]
                                      /-
                                        🎉 no goals
                                      -/


lemma mono_of_isZero_kernel' {X Y : C} {f : X ⟶ Y} (c : KernelFork f) (hc : IsLimit c)
    (h : IsZero c.pt) : Mono f := mono_of_cancel_zero _ (fun g hg => by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    X Y : C
    f : Quiver.Hom X Y
    c : CategoryTheory.Limits.KernelFork f
    hc : CategoryTheory.Limits.IsLimit c
    h : CategoryTheory.Limits.IsZero c.pt
    P✝ : C
    g : Quiver.Hom P✝ X
    hg : Eq (CategoryTheory.CategoryStruct.comp g f) 0
    ⊢ Eq g 0
  -/
  obtain ⟨a, ha⟩ := KernelFork.IsLimit.lift' hc _ hg
  /-
    case mk
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    X Y : C
    f : Quiver.Hom X Y
    c : CategoryTheory.Limits.KernelFork f
    hc : CategoryTheory.Limits.IsLimit c
    h : CategoryTheory.Limits.IsZero c.pt
    P✝ : C
    g : Quiver.Hom P✝ X
    hg : Eq (CategoryTheory.CategoryStruct.comp g f) 0
    a : Quiver.Hom P✝ c.pt
    ha : Eq (CategoryTheory.CategoryStruct.comp a (CategoryTheory.Limits.Fork.ι c) …
    ⊢ Eq g 0
  -/
  rw [← ha, h.eq_of_tgt a 0, Limits.zero_comp])
  /-
    🎉 no goals
  -/


lemma mono_of_isZero_kernel {X Y : C} (f : X ⟶ Y) [HasKernel f] (h : IsZero (kernel f)) :
    Mono f :=
  mono_of_isZero_kernel' _ (kernelIsKernel _) h


theorem epi_of_cancel_zero {P Q : C} (f : P ⟶ Q) (h : ∀ {R : C} (g : Q ⟶ R), f ≫ g = 0 → g = 0) :
    Epi f :=
  ⟨fun {Z} g g' hg =>
    sub_eq_zero.1 <| h _ <| (map_sub (leftComp Z f) g g').trans <| sub_eq_zero.2 hg⟩


theorem epi_iff_cancel_zero {P Q : C} (f : P ⟶ Q) :
    Epi f ↔ ∀ (R : C) (g : Q ⟶ R), f ≫ g = 0 → g = 0 :=
  ⟨fun _ _ _ => zero_of_epi_comp _, epi_of_cancel_zero f⟩


theorem epi_of_cokernel_zero {X Y : C} {f : X ⟶ Y} [HasColimit (parallelPair f 0)]
    (w : cokernel.π f = 0) : Epi f :=
                                     /-
                                       C : Type u
                                       inst✝² : CategoryTheory.Category.{v, u} C
                                       inst✝¹ : CategoryTheory.Preadditive C
                                       X Y : C
                                       f : Quiver.Hom X Y
                                       inst✝ : CategoryTheory.Limits.HasColimit (CategoryTheory.Limits.parallelPair f …
                                       w : Eq (CategoryTheory.Limits.cokernel.π f) 0
                                       R✝ : C
                                       g : Quiver.Hom Y R✝
                                       h : Eq (CategoryTheory.CategoryStruct.comp f g) 0
                                       ⊢ Eq g 0
                                     -/
  epi_of_cancel_zero f fun g h => by rw [← cokernel.π_desc f g h, w, Limits.zero_comp]
                                     /-
                                       🎉 no goals
                                     -/


lemma epi_of_isZero_cokernel' {X Y : C} {f : X ⟶ Y} (c : CokernelCofork f) (hc : IsColimit c)
    (h : IsZero c.pt) : Epi f := epi_of_cancel_zero _ (fun g hg => by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    X Y : C
    f : Quiver.Hom X Y
    c : CategoryTheory.Limits.CokernelCofork f
    hc : CategoryTheory.Limits.IsColimit c
    h : CategoryTheory.Limits.IsZero c.pt
    R✝ : C
    g : Quiver.Hom Y R✝
    hg : Eq (CategoryTheory.CategoryStruct.comp f g) 0
    ⊢ Eq g 0
  -/
  obtain ⟨a, ha⟩ := CokernelCofork.IsColimit.desc' hc _ hg
  /-
    case mk
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.Preadditive C
    X Y : C
    f : Quiver.Hom X Y
    c : CategoryTheory.Limits.CokernelCofork f
    hc : CategoryTheory.Limits.IsColimit c
    h : CategoryTheory.Limits.IsZero c.pt
    R✝ : C
    g : Quiver.Hom Y R✝
    hg : Eq (CategoryTheory.CategoryStruct.comp f g) 0
    a : Quiver.Hom c.pt R✝
    ha : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π c) …
    ⊢ Eq g 0
  -/
  rw [← ha, h.eq_of_src a 0, Limits.comp_zero])
  /-
    🎉 no goals
  -/


lemma epi_of_isZero_cokernel {X Y : C} (f : X ⟶ Y) [HasCokernel f] (h : IsZero (cokernel f)) :
    Epi f :=
  epi_of_isZero_cokernel' _ (cokernelIsCokernel _) h


@[simp]
theorem comp_left_eq_zero [IsIso f] : f ≫ g = 0 ↔ g = 0 := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Preadditive C
    P Q R : C
    f : Quiver.Hom P Q
    g : Quiver.Hom Q R
    inst✝ : CategoryTheory.IsIso f
    ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp f g) 0) (Eq g 0)
  -/
  rw [← IsIso.eq_inv_comp, Limits.comp_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem comp_right_eq_zero [IsIso g] : f ≫ g = 0 ↔ f = 0 := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Preadditive C
    P Q R : C
    f : Quiver.Hom P Q
    g : Quiver.Hom Q R
    inst✝ : CategoryTheory.IsIso g
    ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp f g) 0) (Eq f 0)
  -/
  rw [← IsIso.eq_comp_inv, Limits.zero_comp]
  /-
    🎉 no goals
  -/


theorem mono_of_kernel_iso_zero {X Y : C} {f : X ⟶ Y} [HasLimit (parallelPair f 0)]
    (w : kernel f ≅ 0) : Mono f :=
  mono_of_kernel_zero (zero_of_source_iso_zero _ w)


theorem epi_of_cokernel_iso_zero {X Y : C} {f : X ⟶ Y} [HasColimit (parallelPair f 0)]
    (w : cokernel f ≅ 0) : Epi f :=
  epi_of_cokernel_zero (zero_of_target_iso_zero _ w)


/-- Map a kernel cone on the difference of two morphisms to the equalizer fork. -/
@[simps! pt]
def forkOfKernelFork (c : KernelFork (f - g)) : Fork f g :=
                     /-
                       C : Type u
                       inst✝¹ : CategoryTheory.Category.{v, u} C
                       inst✝ : CategoryTheory.Preadditive C
                       X Y : C
                       f g : Quiver.Hom X Y
                       c : CategoryTheory.Limits.KernelFork (HSub.hSub f g)
                       ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Fork.ι c) f) ( …
                     -/
  Fork.ofι c.ι <| by rw [← sub_eq_zero, ← comp_sub, c.condition]
                     /-
                       🎉 no goals
                     -/


@[simp]
theorem forkOfKernelFork_ι (c : KernelFork (f - g)) : (forkOfKernelFork c).ι = c.ι :=
  rfl


/-- Map any equalizer fork to a cone on the difference of the two morphisms. -/
def kernelForkOfFork (c : Fork f g) : KernelFork (f - g) :=
                     /-
                       C : Type u
                       inst✝¹ : CategoryTheory.Category.{v, u} C
                       inst✝ : CategoryTheory.Preadditive C
                       X Y : C
                       f g : Quiver.Hom X Y
                       c : CategoryTheory.Limits.Fork f g
                       ⊢ Eq (CategoryTheory.CategoryStruct.comp c.ι (HSub.hSub f g)) (CategoryTheory. …
                     -/
  Fork.ofι c.ι <| by rw [comp_sub, comp_zero, sub_eq_zero, c.condition]
                     /-
                       🎉 no goals
                     -/


@[simp]
theorem kernelForkOfFork_ι (c : Fork f g) : (kernelForkOfFork c).ι = c.ι :=
  rfl


@[simp]
theorem kernelForkOfFork_ofι {P : C} (ι : P ⟶ X) (w : ι ≫ f = ι ≫ g) :
                                                           /-
                                                             C : Type u
                                                             inst✝¹ : CategoryTheory.Category.{v, u} C
                                                             inst✝ : CategoryTheory.Preadditive C
                                                             X Y : C
                                                             f g : Quiver.Hom X Y
                                                             P : C
                                                             ι : Quiver.Hom P X
                                                             w : Eq (CategoryTheory.CategoryStruct.comp ι f) (CategoryTheory.CategoryStruct …
                                                             ⊢ Eq (CategoryTheory.CategoryStruct.comp ι (HSub.hSub f g)) 0
                                                           -/
    kernelForkOfFork (Fork.ofι ι w) = KernelFork.ofι ι (by simp [w]) :=
                                                           /-
                                                             🎉 no goals
                                                           -/
  rfl


/-- A kernel of `f - g` is an equalizer of `f` and `g`. -/
def isLimitForkOfKernelFork {c : KernelFork (f - g)} (i : IsLimit c) :
    IsLimit (forkOfKernelFork c) :=
  Fork.IsLimit.mk' _ fun s =>
                                                         /-
                                                           C : Type u
                                                           inst✝¹ : CategoryTheory.Category.{v, u} C
                                                           inst✝ : CategoryTheory.Preadditive C
                                                           X Y : C
                                                           f g : Quiver.Hom X Y
                                                           c : CategoryTheory.Limits.KernelFork (HSub.hSub f g)
                                                           i : CategoryTheory.Limits.IsLimit c
                                                           s : CategoryTheory.Limits.Fork f g
                                                           m✝ : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingP …
                                                           h : Eq (CategoryTheory.CategoryStruct.comp m✝ (CategoryTheory.Preadditive.fork …
                                                           ⊢ Eq m✝ (i.lift (CategoryTheory.Preadditive.kernelForkOfFork s))
                                                         -/
    ⟨i.lift (kernelForkOfFork s), i.fac _ _, fun h => by apply Fork.IsLimit.hom_ext i; aesop_cat⟩
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/


@[simp]
theorem isLimitForkOfKernelFork_lift {c : KernelFork (f - g)} (i : IsLimit c) (s : Fork f g) :
    (isLimitForkOfKernelFork i).lift s = i.lift (kernelForkOfFork s) :=
  rfl


/-- An equalizer of `f` and `g` is a kernel of `f - g`. -/
def isLimitKernelForkOfFork {c : Fork f g} (i : IsLimit c) : IsLimit (kernelForkOfFork c) :=
  Fork.IsLimit.mk' _ fun s =>
                                                         /-
                                                           C : Type u
                                                           inst✝¹ : CategoryTheory.Category.{v, u} C
                                                           inst✝ : CategoryTheory.Preadditive C
                                                           X Y : C
                                                           f g : Quiver.Hom X Y
                                                           c : CategoryTheory.Limits.Fork f g
                                                           i : CategoryTheory.Limits.IsLimit c
                                                           s : CategoryTheory.Limits.Fork (HSub.hSub f g) 0
                                                           m✝ : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingP …
                                                           h : Eq (CategoryTheory.CategoryStruct.comp m✝ (CategoryTheory.Limits.Fork.ι (C …
                                                           ⊢ Eq m✝ (i.lift (CategoryTheory.Preadditive.forkOfKernelFork s))
                                                         -/
    ⟨i.lift (forkOfKernelFork s), i.fac _ _, fun h => by apply Fork.IsLimit.hom_ext i; aesop_cat⟩
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/


/-- A preadditive category has an equalizer for `f` and `g` if it has a kernel for `f - g`. -/
theorem hasEqualizer_of_hasKernel [HasKernel (f - g)] : HasEqualizer f g :=
  HasLimit.mk
    { cone := forkOfKernelFork _
      isLimit := isLimitForkOfKernelFork (equalizerIsEqualizer (f - g) 0) }


/-- A preadditive category has a kernel for `f - g` if it has an equalizer for `f` and `g`. -/
theorem hasKernel_of_hasEqualizer [HasEqualizer f g] : HasKernel (f - g) :=
  HasLimit.mk
    { cone := kernelForkOfFork (equalizer.fork f g)
      isLimit := isLimitKernelForkOfFork (limit.isLimit (parallelPair f g)) }


/-- Map a cokernel cocone on the difference of two morphisms to the coequalizer cofork. -/
@[simps! pt]
def coforkOfCokernelCofork (c : CokernelCofork (f - g)) : Cofork f g :=
                       /-
                         C : Type u
                         inst✝¹ : CategoryTheory.Category.{v, u} C
                         inst✝ : CategoryTheory.Preadditive C
                         X Y : C
                         f g : Quiver.Hom X Y
                         c : CategoryTheory.Limits.CokernelCofork (HSub.hSub f g)
                         ⊢ Eq (CategoryTheory.CategoryStruct.comp f (CategoryTheory.Limits.Cofork.π c)) …
                       -/
  Cofork.ofπ c.π <| by rw [← sub_eq_zero, ← sub_comp, c.condition]
                       /-
                         🎉 no goals
                       -/


@[simp]
theorem coforkOfCokernelCofork_π (c : CokernelCofork (f - g)) :
    (coforkOfCokernelCofork c).π = c.π :=
  rfl


/-- Map any coequalizer cofork to a cocone on the difference of the two morphisms. -/
def cokernelCoforkOfCofork (c : Cofork f g) : CokernelCofork (f - g) :=
                       /-
                         C : Type u
                         inst✝¹ : CategoryTheory.Category.{v, u} C
                         inst✝ : CategoryTheory.Preadditive C
                         X Y : C
                         f g : Quiver.Hom X Y
                         c : CategoryTheory.Limits.Cofork f g
                         ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSub.hSub f g) c.π) (CategoryTheory. …
                       -/
  Cofork.ofπ c.π <| by rw [sub_comp, zero_comp, sub_eq_zero, c.condition]
                       /-
                         🎉 no goals
                       -/


@[simp]
theorem cokernelCoforkOfCofork_π (c : Cofork f g) : (cokernelCoforkOfCofork c).π = c.π :=
  rfl


@[simp]
theorem cokernelCoforkOfCofork_ofπ {P : C} (π : Y ⟶ P) (w : f ≫ π = g ≫ π) :
                                                                       /-
                                                                         C : Type u
                                                                         inst✝¹ : CategoryTheory.Category.{v, u} C
                                                                         inst✝ : CategoryTheory.Preadditive C
                                                                         X Y : C
                                                                         f g : Quiver.Hom X Y
                                                                         P : C
                                                                         π : Quiver.Hom Y P
                                                                         w : Eq (CategoryTheory.CategoryStruct.comp f π) (CategoryTheory.CategoryStruct …
                                                                         ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSub.hSub f g) π) 0
                                                                       -/
    cokernelCoforkOfCofork (Cofork.ofπ π w) = CokernelCofork.ofπ π (by simp [w]) :=
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
  rfl


/-- A cokernel of `f - g` is a coequalizer of `f` and `g`. -/
def isColimitCoforkOfCokernelCofork {c : CokernelCofork (f - g)} (i : IsColimit c) :
    IsColimit (coforkOfCokernelCofork c) :=
  Cofork.IsColimit.mk' _ fun s =>
    ⟨i.desc (cokernelCoforkOfCofork s), i.fac _ _, fun h => by
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Preadditive C
        X Y : C
        f g : Quiver.Hom X Y
        c : CategoryTheory.Limits.CokernelCofork (HSub.hSub f g)
        i : CategoryTheory.Limits.IsColimit c
        s : CategoryTheory.Limits.Cofork f g
        m✝ : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingP …
        h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Preadditive.coforkO …
        ⊢ Eq m✝ (i.desc (CategoryTheory.Preadditive.cokernelCoforkOfCofork s))
      -/
      apply Cofork.IsColimit.hom_ext i; aesop_cat⟩
                                        /-
                                          🎉 no goals
                                        -/


@[simp]
theorem isColimitCoforkOfCokernelCofork_desc {c : CokernelCofork (f - g)} (i : IsColimit c)
    (s : Cofork f g) :
    (isColimitCoforkOfCokernelCofork i).desc s = i.desc (cokernelCoforkOfCofork s) :=
  rfl


/-- A coequalizer of `f` and `g` is a cokernel of `f - g`. -/
def isColimitCokernelCoforkOfCofork {c : Cofork f g} (i : IsColimit c) :
    IsColimit (cokernelCoforkOfCofork c) :=
  Cofork.IsColimit.mk' _ fun s =>
    ⟨i.desc (coforkOfCokernelCofork s), i.fac _ _, fun h => by
      /-
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        inst✝ : CategoryTheory.Preadditive C
        X Y : C
        f g : Quiver.Hom X Y
        c : CategoryTheory.Limits.Cofork f g
        i : CategoryTheory.Limits.IsColimit c
        s : CategoryTheory.Limits.Cofork (HSub.hSub f g) 0
        m✝ : Quiver.Hom (((CategoryTheory.Functor.const CategoryTheory.Limits.WalkingP …
        h : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Cofork.π (Ca …
        ⊢ Eq m✝ (i.desc (CategoryTheory.Preadditive.coforkOfCokernelCofork s))
      -/
      apply Cofork.IsColimit.hom_ext i; aesop_cat⟩
                                        /-
                                          🎉 no goals
                                        -/


/-- A preadditive category has a coequalizer for `f` and `g` if it has a cokernel for `f - g`. -/
theorem hasCoequalizer_of_hasCokernel [HasCokernel (f - g)] : HasCoequalizer f g :=
  HasColimit.mk
    { cocone := coforkOfCokernelCofork _
      isColimit := isColimitCoforkOfCokernelCofork (coequalizerIsCoequalizer (f - g) 0) }


/-- A preadditive category has a cokernel for `f - g` if it has a coequalizer for `f` and `g`. -/
theorem hasCokernel_of_hasCoequalizer [HasCoequalizer f g] : HasCokernel (f - g) :=
  HasColimit.mk
    { cocone := cokernelCoforkOfCofork (coequalizer.cofork f g)
      isColimit := isColimitCokernelCoforkOfCofork (colimit.isColimit (parallelPair f g)) }


/-- If a preadditive category has all kernels, then it also has all equalizers. -/
theorem hasEqualizers_of_hasKernels [HasKernels C] : HasEqualizers C :=
  @hasEqualizers_of_hasLimit_parallelPair _ _ fun {_} {_} f g => hasEqualizer_of_hasKernel f g


/-- If a preadditive category has all cokernels, then it also has all coequalizers. -/
theorem hasCoequalizers_of_hasCokernels [HasCokernels C] : HasCoequalizers C :=
  @hasCoequalizers_of_hasColimit_parallelPair _ _ fun {_} {_} f g =>
    hasCoequalizer_of_hasCokernel f g


instance : SMul (Units ℤ) (X ≅ Y) where
  smul a e :=
    { hom := (a : ℤ) • e.hom
      inv := ((a⁻¹ : Units ℤ) : ℤ) • e.inv
      hom_inv_id := by
        /-
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.84315, u_1} C
          inst✝ : CategoryTheory.Preadditive C
          X Y : C
          a : Units Int
          e : CategoryTheory.Iso X Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSMul.hSMul (↑a) e.hom) (HSMul.hSMul …
        -/
        simp only [comp_zsmul, zsmul_comp, smul_smul, Units.inv_mul, one_smul, e.hom_inv_id]
        /-
          🎉 no goals
        -/
      inv_hom_id := by
        /-
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.84315, u_1} C
          inst✝ : CategoryTheory.Preadditive C
          X Y : C
          a : Units Int
          e : CategoryTheory.Iso X Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSMul.hSMul (↑(Inv.inv a)) e.inv) (H …
        -/
        simp only [comp_zsmul, zsmul_comp, smul_smul, Units.mul_inv, one_smul, e.inv_hom_id] }
        /-
          🎉 no goals
        -/


@[simp]
lemma smul_iso_hom (a : Units ℤ) (e : X ≅ Y) : (a • e).hom = a • e.hom := rfl


@[simp]
lemma smul_iso_inv (a : Units ℤ) (e : X ≅ Y) : (a • e).inv = a⁻¹ • e.inv := rfl


instance : Neg (X ≅ Y) where
  neg e :=
    { hom := -e.hom
      inv := -e.inv }


@[simp]
lemma neg_iso_hom (e : X ≅ Y) : (-e).hom = -e.hom := rfl


@[simp]
lemma neg_iso_inv (e : X ≅ Y) : (-e).inv = -e.inv := rfl


