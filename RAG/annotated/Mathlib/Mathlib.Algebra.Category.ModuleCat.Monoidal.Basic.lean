/-- (implementation) tensor product of R-modules -/
def tensorObj (M N : ModuleCat R) : ModuleCat R :=
  ModuleCat.of R (M ⊗[R] N)


/-- (implementation) tensor product of morphisms R-modules -/
def tensorHom {M N M' N' : ModuleCat R} (f : M ⟶ N) (g : M' ⟶ N') :
    tensorObj M M' ⟶ tensorObj N N' :=
  ofHom <| TensorProduct.map f.hom g.hom


/-- (implementation) left whiskering for R-modules -/
def whiskerLeft (M : ModuleCat R) {N₁ N₂ : ModuleCat R} (f : N₁ ⟶ N₂) :
    tensorObj M N₁ ⟶ tensorObj M N₂ :=
  ofHom <| f.hom.lTensor M


/-- (implementation) right whiskering for R-modules -/
def whiskerRight {M₁ M₂ : ModuleCat R} (f : M₁ ⟶ M₂) (N : ModuleCat R) :
    tensorObj M₁ N ⟶ tensorObj M₂ N :=
  ofHom <| f.hom.rTensor N


theorem tensor_id (M N : ModuleCat R) : tensorHom (𝟙 M) (𝟙 N) = 𝟙 (ModuleCat.of R (M ⊗ N)) := by
  /-
    R : Type u
    inst✝ : CommRing R
    M : ModuleCat R
    N : ModuleCat R
    ⊢ Eq (ModuleCat.MonoidalCategory.tensorHom (CategoryTheory.CategoryStruct.id M …
  -/
  ext : 1
  -- Porting note (https://github.com/leanprover-community/mathlib4/pull/11041): even with high priority `ext` fails to find this.
  /-
    case hf
    R : Type u
    inst✝ : CommRing R
    M : ModuleCat R
    N : ModuleCat R
    ⊢ Eq (ModuleCat.MonoidalCategory.tensorHom (CategoryTheory.CategoryStruct.id M …
  -/
  apply TensorProduct.ext
  /-
    case hf.H
    R : Type u
    inst✝ : CommRing R
    M : ModuleCat R
    N : ModuleCat R
    ⊢ Eq ((TensorProduct.mk R ↑M ↑N).compr₂ (ModuleCat.MonoidalCategory.tensorHom  …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem tensor_comp {X₁ Y₁ Z₁ X₂ Y₂ Z₂ : ModuleCat R} (f₁ : X₁ ⟶ Y₁) (f₂ : X₂ ⟶ Y₂) (g₁ : Y₁ ⟶ Z₁)
    (g₂ : Y₂ ⟶ Z₂) : tensorHom (f₁ ≫ g₁) (f₂ ≫ g₂) = tensorHom f₁ f₂ ≫ tensorHom g₁ g₂ := by
  /-
    R : Type u
    inst✝ : CommRing R
    X₁ Y₁ Z₁ : ModuleCat R
    X₂ Y₂ Z₂ : ModuleCat R
    f₁ : Quiver.Hom X₁ Y₁
    f₂ : Quiver.Hom X₂ Y₂
    g₁ : Quiver.Hom Y₁ Z₁
    g₂ : Quiver.Hom Y₂ Z₂
    ⊢ Eq (ModuleCat.MonoidalCategory.tensorHom (CategoryTheory.CategoryStruct.comp …
  -/
  ext : 1
  -- Porting note (https://github.com/leanprover-community/mathlib4/pull/11041): even with high priority `ext` fails to find this.
  /-
    case hf
    R : Type u
    inst✝ : CommRing R
    X₁ Y₁ Z₁ : ModuleCat R
    X₂ Y₂ Z₂ : ModuleCat R
    f₁ : Quiver.Hom X₁ Y₁
    f₂ : Quiver.Hom X₂ Y₂
    g₁ : Quiver.Hom Y₁ Z₁
    g₂ : Quiver.Hom Y₂ Z₂
    ⊢ Eq (ModuleCat.MonoidalCategory.tensorHom (CategoryTheory.CategoryStruct.comp …
  -/
  apply TensorProduct.ext
  /-
    case hf.H
    R : Type u
    inst✝ : CommRing R
    X₁ Y₁ Z₁ : ModuleCat R
    X₂ Y₂ Z₂ : ModuleCat R
    f₁ : Quiver.Hom X₁ Y₁
    f₂ : Quiver.Hom X₂ Y₂
    g₁ : Quiver.Hom Y₁ Z₁
    g₂ : Quiver.Hom Y₂ Z₂
    ⊢ Eq ((TensorProduct.mk R ↑X₁ ↑X₂).compr₂ (ModuleCat.MonoidalCategory.tensorHo …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- (implementation) the associator for R-modules -/
def associator (M : ModuleCat.{v} R) (N : ModuleCat.{w} R) (K : ModuleCat.{x} R) :
    tensorObj (tensorObj M N) K ≅ tensorObj M (tensorObj N K) :=
  (TensorProduct.assoc R M N K).toModuleIso


/-- (implementation) the left unitor for R-modules -/
def leftUnitor (M : ModuleCat.{u} R) : ModuleCat.of R (R ⊗[R] M) ≅ M :=
  (LinearEquiv.toModuleIso (TensorProduct.lid R M) : of R (R ⊗ M) ≅ of R M).trans (ofSelfIso M)


/-- (implementation) the right unitor for R-modules -/
def rightUnitor (M : ModuleCat.{u} R) : ModuleCat.of R (M ⊗[R] R) ≅ M :=
  (LinearEquiv.toModuleIso (TensorProduct.rid R M) : of R (M ⊗ R) ≅ of R M).trans (ofSelfIso M)


@[simps (config := .lemmasOnly)]
instance instMonoidalCategoryStruct : MonoidalCategoryStruct (ModuleCat.{u} R) where
  tensorObj := tensorObj
  whiskerLeft := whiskerLeft
  whiskerRight := whiskerRight
  tensorHom f g := ofHom <| TensorProduct.map f.hom g.hom
  tensorUnit := ModuleCat.of R R
  associator := associator
  leftUnitor := leftUnitor
  rightUnitor := rightUnitor


theorem associator_naturality {X₁ X₂ X₃ Y₁ Y₂ Y₃ : ModuleCat R} (f₁ : X₁ ⟶ Y₁) (f₂ : X₂ ⟶ Y₂)
    (f₃ : X₃ ⟶ Y₃) :
    tensorHom (tensorHom f₁ f₂) f₃ ≫ (associator Y₁ Y₂ Y₃).hom =
      (associator X₁ X₂ X₃).hom ≫ tensorHom f₁ (tensorHom f₂ f₃) := by
  /-
    R : Type u
    inst✝ : CommRing R
    X₁ : ModuleCat R
    X₂ : ModuleCat R
    X₃ : ModuleCat R
    Y₁ : ModuleCat R
    Y₂ : ModuleCat R
    Y₃ : ModuleCat R
    f₁ : Quiver.Hom X₁ Y₁
    f₂ : Quiver.Hom X₂ Y₂
    f₃ : Quiver.Hom X₃ Y₃
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (ModuleCat.MonoidalCategory.tensorHom …
  -/
  ext : 1
  /-
    case hf
    R : Type u
    inst✝ : CommRing R
    X₁ : ModuleCat R
    X₂ : ModuleCat R
    X₃ : ModuleCat R
    Y₁ : ModuleCat R
    Y₂ : ModuleCat R
    Y₃ : ModuleCat R
    f₁ : Quiver.Hom X₁ Y₁
    f₂ : Quiver.Hom X₂ Y₂
    f₃ : Quiver.Hom X₃ Y₃
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (ModuleCat.MonoidalCategory.tensorHom …
  -/
  apply TensorProduct.ext_threefold
  /-
    case hf.H
    R : Type u
    inst✝ : CommRing R
    X₁ : ModuleCat R
    X₂ : ModuleCat R
    X₃ : ModuleCat R
    Y₁ : ModuleCat R
    Y₂ : ModuleCat R
    Y₃ : ModuleCat R
    f₁ : Quiver.Hom X₁ Y₁
    f₂ : Quiver.Hom X₂ Y₂
    f₃ : Quiver.Hom X₃ Y₃
    ⊢ ∀ (x : ↑X₁) (y : ↑X₂) (z : ↑X₃), Eq ((CategoryTheory.CategoryStruct.comp (Mo …
  -/
  intro x y z
  /-
    case hf.H
    R : Type u
    inst✝ : CommRing R
    X₁ : ModuleCat R
    X₂ : ModuleCat R
    X₃ : ModuleCat R
    Y₁ : ModuleCat R
    Y₂ : ModuleCat R
    Y₃ : ModuleCat R
    f₁ : Quiver.Hom X₁ Y₁
    f₂ : Quiver.Hom X₂ Y₂
    f₃ : Quiver.Hom X₃ Y₃
    x : ↑X₁
    y : ↑X₂
    z : ↑X₃
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (ModuleCat.MonoidalCategory.tensorHo …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem pentagon (W X Y Z : ModuleCat R) :
    whiskerRight (associator W X Y).hom Z ≫
        (associator W (tensorObj X Y) Z).hom ≫ whiskerLeft W (associator X Y Z).hom =
      (associator (tensorObj W X) Y Z).hom ≫ (associator W X (tensorObj Y Z)).hom := by
  /-
    R : Type u
    inst✝ : CommRing R
    W : ModuleCat R
    X : ModuleCat R
    Y : ModuleCat R
    Z : ModuleCat R
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (ModuleCat.MonoidalCategory.whiskerRi …
  -/
  ext : 1
  /-
    case hf
    R : Type u
    inst✝ : CommRing R
    W : ModuleCat R
    X : ModuleCat R
    Y : ModuleCat R
    Z : ModuleCat R
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (ModuleCat.MonoidalCategory.whiskerRi …
  -/
  apply TensorProduct.ext_fourfold
  /-
    case hf.H
    R : Type u
    inst✝ : CommRing R
    W : ModuleCat R
    X : ModuleCat R
    Y : ModuleCat R
    Z : ModuleCat R
    ⊢ ∀ (w : ↑W) (x : ↑X) (y : ↑Y) (z : ↑Z), Eq ((CategoryTheory.CategoryStruct.co …
  -/
  intro w x y z
  /-
    case hf.H
    R : Type u
    inst✝ : CommRing R
    W : ModuleCat R
    X : ModuleCat R
    Y : ModuleCat R
    Z : ModuleCat R
    w : ↑W
    x : ↑X
    y : ↑Y
    z : ↑Z
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (ModuleCat.MonoidalCategory.whiskerR …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem leftUnitor_naturality {M N : ModuleCat R} (f : M ⟶ N) :
    tensorHom (𝟙 (ModuleCat.of R R)) f ≫ (leftUnitor N).hom = (leftUnitor M).hom ≫ f := by
  /-
    R : Type u
    inst✝ : CommRing R
    M N : ModuleCat R
    f : Quiver.Hom M N
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (ModuleCat.MonoidalCategory.tensorHom …
  -/
  ext : 1
  -- Porting note (https://github.com/leanprover-community/mathlib4/pull/11041): broken ext
  /-
    case hf
    R : Type u
    inst✝ : CommRing R
    M N : ModuleCat R
    f : Quiver.Hom M N
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (ModuleCat.MonoidalCategory.tensorHom …
  -/
  apply TensorProduct.ext
  /-
    case hf.H
    R : Type u
    inst✝ : CommRing R
    M N : ModuleCat R
    f : Quiver.Hom M N
    ⊢ Eq ((TensorProduct.mk R ↑(ModuleCat.of R R) ↑M).compr₂ (CategoryTheory.Categ …
  -/
  ext x
  -- Porting note (https://github.com/leanprover-community/mathlib4/pull/10934): used to be dsimp
  change ((leftUnitor N).hom) ((tensorHom (𝟙 (of R R)) f) ((1 : R) ⊗ₜ[R] x)) =
    f (((leftUnitor M).hom) (1 ⊗ₜ[R] x))
  /-
    case hf.H.h.h
    R : Type u
    inst✝ : CommRing R
    M N : ModuleCat R
    f : Quiver.Hom M N
    x : ↑M
    ⊢ Eq ((ModuleCat.MonoidalCategory.leftUnitor N).hom.hom ((ModuleCat.MonoidalCa …
  -/
  erw [TensorProduct.lid_tmul, TensorProduct.lid_tmul]
  /-
    case hf.H.h.h
    R : Type u
    inst✝ : CommRing R
    M N : ModuleCat R
    f : Quiver.Hom M N
    x : ↑M
    ⊢ Eq (HSMul.hSMul ((CategoryTheory.CategoryStruct.id (ModuleCat.of R R)).hom { …
  -/
  rw [LinearMap.map_smul]
  /-
    case hf.H.h.h
    R : Type u
    inst✝ : CommRing R
    M N : ModuleCat R
    f : Quiver.Hom M N
    x : ↑M
    ⊢ Eq (HSMul.hSMul ((CategoryTheory.CategoryStruct.id (ModuleCat.of R R)).hom { …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem rightUnitor_naturality {M N : ModuleCat R} (f : M ⟶ N) :
    tensorHom f (𝟙 (ModuleCat.of R R)) ≫ (rightUnitor N).hom = (rightUnitor M).hom ≫ f := by
  /-
    R : Type u
    inst✝ : CommRing R
    M N : ModuleCat R
    f : Quiver.Hom M N
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (ModuleCat.MonoidalCategory.tensorHom …
  -/
  ext : 1
  -- Porting note (https://github.com/leanprover-community/mathlib4/pull/11041): broken ext
  /-
    case hf
    R : Type u
    inst✝ : CommRing R
    M N : ModuleCat R
    f : Quiver.Hom M N
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (ModuleCat.MonoidalCategory.tensorHom …
  -/
  apply TensorProduct.ext
  /-
    case hf.H
    R : Type u
    inst✝ : CommRing R
    M N : ModuleCat R
    f : Quiver.Hom M N
    ⊢ Eq ((TensorProduct.mk R ↑M ↑(ModuleCat.of R R)).compr₂ (CategoryTheory.Categ …
  -/
  ext x
  /-
    case hf.H.h.h
    R : Type u
    inst✝ : CommRing R
    M N : ModuleCat R
    f : Quiver.Hom M N
    x : ↑M
    ⊢ Eq ((((TensorProduct.mk R ↑M ↑(ModuleCat.of R R)).compr₂ (CategoryTheory.Cat …
  -/
  dsimp
  /-
    case hf.H.h.h
    R : Type u
    inst✝ : CommRing R
    M N : ModuleCat R
    f : Quiver.Hom M N
    x : ↑M
    ⊢ Eq ((ModuleCat.MonoidalCategory.rightUnitor N).hom.hom ((ModuleCat.MonoidalC …
  -/
  erw [TensorProduct.rid_tmul, TensorProduct.rid_tmul]
  /-
    case hf.H.h.h
    R : Type u
    inst✝ : CommRing R
    M N : ModuleCat R
    f : Quiver.Hom M N
    x : ↑M
    ⊢ Eq (HSMul.hSMul ((CategoryTheory.CategoryStruct.id (ModuleCat.of R R)).hom { …
  -/
  rw [LinearMap.map_smul]
  /-
    case hf.H.h.h
    R : Type u
    inst✝ : CommRing R
    M N : ModuleCat R
    f : Quiver.Hom M N
    x : ↑M
    ⊢ Eq (HSMul.hSMul ((CategoryTheory.CategoryStruct.id (ModuleCat.of R R)).hom { …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem triangle (M N : ModuleCat.{u} R) :
    (associator M (ModuleCat.of R R) N).hom ≫ tensorHom (𝟙 M) (leftUnitor N).hom =
      tensorHom (rightUnitor M).hom (𝟙 N) := by
  /-
    R : Type u
    inst✝ : CommRing R
    M N : ModuleCat R
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (ModuleCat.MonoidalCategory.associato …
  -/
  ext : 1
  /-
    case hf
    R : Type u
    inst✝ : CommRing R
    M N : ModuleCat R
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (ModuleCat.MonoidalCategory.associato …
  -/
  apply TensorProduct.ext_threefold
  /-
    case hf.H
    R : Type u
    inst✝ : CommRing R
    M N : ModuleCat R
    ⊢ ∀ (x : ↑M) (y : ↑(ModuleCat.of R R)) (z : ↑N), Eq ((CategoryTheory.CategoryS …
  -/
  intro x y z
  -- Porting note (https://github.com/leanprover-community/mathlib4/pull/10934): used to be dsimp [tensorHom, associator]
  /-
    case hf.H
    R : Type u
    inst✝ : CommRing R
    M N : ModuleCat R
    x : ↑M
    y : ↑(ModuleCat.of R R)
    z : ↑N
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (ModuleCat.MonoidalCategory.associat …
  -/
  change x ⊗ₜ[R] ((leftUnitor N).hom) (y ⊗ₜ[R] z) = ((rightUnitor M).hom) (x ⊗ₜ[R] y) ⊗ₜ[R] z
  /-
    case hf.H
    R : Type u
    inst✝ : CommRing R
    M N : ModuleCat R
    x : ↑M
    y : ↑(ModuleCat.of R R)
    z : ↑N
    ⊢ Eq (TensorProduct.tmul R x ((ModuleCat.MonoidalCategory.leftUnitor N).hom.ho …
  -/
  erw [TensorProduct.lid_tmul, TensorProduct.rid_tmul]
  /-
    case hf.H
    R : Type u
    inst✝ : CommRing R
    M N : ModuleCat R
    x : ↑M
    y : ↑(ModuleCat.of R R)
    z : ↑N
    ⊢ Eq (TensorProduct.tmul R x (HSMul.hSMul y z)) (TensorProduct.tmul R (HSMul.h …
  -/
  exact (TensorProduct.smul_tmul _ _ _).symm
  /-
    🎉 no goals
  -/


                                                                  /-
                                                                    R : Type u
                                                                    inst✝ : CommRing R
                                                                    ⊢ ∀ (X : ModuleCat R) {Y₁ Y₂ : ModuleCat R} (f : Quiver.Hom Y₁ Y₂), Eq (Catego …
                                                                  -/
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
instance monoidalCategory : MonoidalCategory (ModuleCat.{u} R) := MonoidalCategory.ofTensorHom
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
  (tensor_id := fun M N ↦ tensor_id M N)
  (tensor_comp := fun f g h ↦ MonoidalCategory.tensor_comp f g h)
  (associator_naturality := fun f g h ↦ MonoidalCategory.associator_naturality f g h)
  (leftUnitor_naturality := fun f ↦ MonoidalCategory.leftUnitor_naturality f)
  (rightUnitor_naturality := fun f ↦ rightUnitor_naturality f)
  (pentagon := fun M N K L ↦ pentagon M N K L)
  (triangle := fun M N ↦ triangle M N)


/-- Remind ourselves that the monoidal unit, being just `R`, is still a commutative ring. -/
instance : CommRing ((𝟙_ (ModuleCat.{u} R) : ModuleCat.{u} R) : Type u) :=
  inferInstanceAs <| CommRing R


@[simp]
theorem tensorHom_tmul {K L M N : ModuleCat.{u} R} (f : K ⟶ L) (g : M ⟶ N) (k : K) (m : M) :
    (f ⊗ g) (k ⊗ₜ m) = f k ⊗ₜ g m :=
  rfl


@[deprecated (since := "2024-09-30")] alias hom_apply := tensorHom_tmul



@[simp]
theorem whiskerLeft_apply (L : ModuleCat.{u} R) {M N : ModuleCat.{u} R} (f : M ⟶ N)
    (l : L) (m : M) :
    (L ◁ f) (l ⊗ₜ m) = l ⊗ₜ f m :=
  rfl


@[simp]
theorem whiskerRight_apply {L M : ModuleCat.{u} R} (f : L ⟶ M) (N : ModuleCat.{u} R)
    (l : L) (n : N) :
    (f ▷ N) (l ⊗ₜ n) = f l ⊗ₜ n :=
  rfl


@[simp]
theorem leftUnitor_hom_apply {M : ModuleCat.{u} R} (r : R) (m : M) :
    ((λ_ M).hom : 𝟙_ (ModuleCat R) ⊗ M ⟶ M) (r ⊗ₜ[R] m) = r • m :=
  TensorProduct.lid_tmul m r


@[simp]
theorem leftUnitor_inv_apply {M : ModuleCat.{u} R} (m : M) :
    ((λ_ M).inv : M ⟶ 𝟙_ (ModuleCat.{u} R) ⊗ M) m = 1 ⊗ₜ[R] m :=
  TensorProduct.lid_symm_apply m


@[simp]
theorem rightUnitor_hom_apply {M : ModuleCat.{u} R} (m : M) (r : R) :
    ((ρ_ M).hom : M ⊗ 𝟙_ (ModuleCat R) ⟶ M) (m ⊗ₜ r) = r • m :=
  TensorProduct.rid_tmul m r


@[simp]
theorem rightUnitor_inv_apply {M : ModuleCat.{u} R} (m : M) :
    ((ρ_ M).inv : M ⟶ M ⊗ 𝟙_ (ModuleCat.{u} R)) m = m ⊗ₜ[R] 1 :=
  TensorProduct.rid_symm_apply m


@[simp]
theorem associator_hom_apply {M N K : ModuleCat.{u} R} (m : M) (n : N) (k : K) :
    ((α_ M N K).hom : (M ⊗ N) ⊗ K ⟶ M ⊗ N ⊗ K) (m ⊗ₜ n ⊗ₜ k) = m ⊗ₜ (n ⊗ₜ k) :=
  rfl


@[simp]
theorem associator_inv_apply {M N K : ModuleCat.{u} R} (m : M) (n : N) (k : K) :
    ((α_ M N K).inv : M ⊗ N ⊗ K ⟶ (M ⊗ N) ⊗ K) (m ⊗ₜ (n ⊗ₜ k)) = m ⊗ₜ n ⊗ₜ k :=
  rfl


/-- Construct for morphisms from the tensor product of two objects in `ModuleCat`. -/
noncomputable def tensorLift : M₁ ⊗ M₂ ⟶ M₃ :=
  ofHom <| TensorProduct.lift (LinearMap.mk₂ R f h₁ h₂ h₃ h₄)


@[simp]
lemma tensorLift_tmul (m : M₁) (n : M₂) :
    tensorLift f h₁ h₂ h₃ h₄ (m ⊗ₜ n) = f m n := rfl


lemma tensor_ext {f g : M₁ ⊗ M₂ ⟶ M₃} (h : ∀ m n, f (m ⊗ₜ n) = g (m ⊗ₜ n)) :
    f = g :=
                                   /-
                                     R : Type u
                                     inst✝ : CommRing R
                                     M₁ M₂ M₃ : ModuleCat R
                                     f g : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj M₁ M₂) M₃
                                     h : ∀ (m : ↑M₁) (n : ↑M₂), Eq (f.hom (TensorProduct.tmul R m n)) (g.hom (Tenso …
                                     ⊢ Eq ((TensorProduct.mk R ↑M₁ ↑M₂).compr₂ f.hom) ((TensorProduct.mk R ↑M₁ ↑M₂) …
                                   -/
  hom_ext <| TensorProduct.ext (by ext; apply h)
                                        /-
                                          🎉 no goals
                                        -/


/-- Extensionality lemma for morphisms from a module of the form `(M₁ ⊗ M₂) ⊗ M₃`. -/
lemma tensor_ext₃' {f g : (M₁ ⊗ M₂) ⊗ M₃ ⟶ M₄}
    (h : ∀ m₁ m₂ m₃, f (m₁ ⊗ₜ m₂ ⊗ₜ m₃) = g (m₁ ⊗ₜ m₂ ⊗ₜ m₃)) :
    f = g :=
  hom_ext <| TensorProduct.ext_threefold h


/-- Extensionality lemma for morphisms from a module of the form `M₁ ⊗ (M₂ ⊗ M₃)`. -/
lemma tensor_ext₃ {f g : M₁ ⊗ (M₂ ⊗ M₃) ⟶ M₄}
    (h : ∀ m₁ m₂ m₃, f (m₁ ⊗ₜ (m₂ ⊗ₜ m₃)) = g (m₁ ⊗ₜ (m₂ ⊗ₜ m₃))) :
    f = g := by
  /-
    R : Type u
    inst✝ : CommRing R
    M₁ M₂ M₃ M₄ : ModuleCat R
    f g : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj M₁ (Category …
    h : ∀ (m₁ : ↑M₁) (m₂ : ↑M₂) (m₃ : ↑M₃), Eq (f.hom (TensorProduct.tmul R m₁ (Te …
    ⊢ Eq f g
  -/
  rw [← cancel_epi (α_ _ _ _).hom]
  /-
    R : Type u
    inst✝ : CommRing R
    M₁ M₂ M₃ M₄ : ModuleCat R
    f g : Quiver.Hom (CategoryTheory.MonoidalCategoryStruct.tensorObj M₁ (Category …
    h : ∀ (m₁ : ↑M₁) (m₂ : ↑M₂) (m₃ : ↑M₃), Eq (f.hom (TensorProduct.tmul R m₁ (Te …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MonoidalCategoryStruc …
  -/
  exact tensor_ext₃' h
  /-
    🎉 no goals
  -/


instance : MonoidalPreadditive (ModuleCat.{u} R) := by
  /-
    R : Type u
    inst✝ : CommRing R
    ⊢ CategoryTheory.MonoidalPreadditive (ModuleCat R)
  -/
  refine ⟨?_, ?_, ?_, ?_⟩
    /-
      case refine_1
      R : Type u
      inst✝ : CommRing R
      ⊢ ∀ {X Y Z : ModuleCat R}, Eq (CategoryTheory.MonoidalCategoryStruct.whiskerLe …
    -/
  · intros
    /-
      case refine_1
      R : Type u
      inst✝ : CommRing R
      X✝ Y✝ Z✝ : ModuleCat R
      ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerLeft X✝ 0) 0
    -/
    ext : 1
    /-
      case refine_1.hf
      R : Type u
      inst✝ : CommRing R
      X✝ Y✝ Z✝ : ModuleCat R
      ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerLeft X✝ 0).hom (ModuleCat.H …
    -/
    refine TensorProduct.ext (LinearMap.ext fun x => LinearMap.ext fun y => ?_)
    /-
      case refine_1.hf
      R : Type u
      inst✝ : CommRing R
      X✝ Y✝ Z✝ : ModuleCat R
      x : ↑X✝
      y : ↑Y✝
      ⊢ Eq ((((TensorProduct.mk R ↑X✝ ↑Y✝).compr₂ (CategoryTheory.MonoidalCategorySt …
    -/
    simp only [LinearMap.compr₂_apply, TensorProduct.mk_apply, hom_zero, LinearMap.zero_apply]
    -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
    /-
      case refine_1.hf
      R : Type u
      inst✝ : CommRing R
      X✝ Y✝ Z✝ : ModuleCat R
      x : ↑X✝
      y : ↑Y✝
      ⊢ Eq ((CategoryTheory.MonoidalCategoryStruct.whiskerLeft X✝ 0).hom (TensorProd …
    -/
    erw [MonoidalCategory.whiskerLeft_apply]
    /-
      case refine_1.hf
      R : Type u
      inst✝ : CommRing R
      X✝ Y✝ Z✝ : ModuleCat R
      x : ↑X✝
      y : ↑Y✝
      ⊢ Eq (TensorProduct.tmul R x ((ModuleCat.Hom.hom 0) y)) 0
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u
      inst✝ : CommRing R
      ⊢ ∀ {X Y Z : ModuleCat R}, Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRi …
    -/
  · intros
    /-
      case refine_2
      R : Type u
      inst✝ : CommRing R
      X✝ Y✝ Z✝ : ModuleCat R
      ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight 0 X✝) 0
    -/
    ext : 1
    /-
      case refine_2.hf
      R : Type u
      inst✝ : CommRing R
      X✝ Y✝ Z✝ : ModuleCat R
      ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight 0 X✝).hom (ModuleCat. …
    -/
    refine TensorProduct.ext (LinearMap.ext fun x => LinearMap.ext fun y => ?_)
    /-
      case refine_2.hf
      R : Type u
      inst✝ : CommRing R
      X✝ Y✝ Z✝ : ModuleCat R
      x : ↑Y✝
      y : ↑X✝
      ⊢ Eq ((((TensorProduct.mk R ↑Y✝ ↑X✝).compr₂ (CategoryTheory.MonoidalCategorySt …
    -/
    simp only [LinearMap.compr₂_apply, TensorProduct.mk_apply, hom_zero, LinearMap.zero_apply, ]
    -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
    /-
      case refine_2.hf
      R : Type u
      inst✝ : CommRing R
      X✝ Y✝ Z✝ : ModuleCat R
      x : ↑Y✝
      y : ↑X✝
      ⊢ Eq ((CategoryTheory.MonoidalCategoryStruct.whiskerRight 0 X✝).hom (TensorPro …
    -/
    erw [MonoidalCategory.whiskerRight_apply]
    /-
      case refine_2.hf
      R : Type u
      inst✝ : CommRing R
      X✝ Y✝ Z✝ : ModuleCat R
      x : ↑Y✝
      y : ↑X✝
      ⊢ Eq (TensorProduct.tmul R ((ModuleCat.Hom.hom 0) x) y) 0
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      R : Type u
      inst✝ : CommRing R
      ⊢ ∀ {X Y Z : ModuleCat R} (f g : Quiver.Hom Y Z), Eq (CategoryTheory.MonoidalC …
    -/
  · intros
    /-
      case refine_3
      R : Type u
      inst✝ : CommRing R
      X✝ Y✝ Z✝ : ModuleCat R
      f✝ g✝ : Quiver.Hom Y✝ Z✝
      ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerLeft X✝ (HAdd.hAdd f✝ g✝))  …
    -/
    ext : 1
    /-
      case refine_3.hf
      R : Type u
      inst✝ : CommRing R
      X✝ Y✝ Z✝ : ModuleCat R
      f✝ g✝ : Quiver.Hom Y✝ Z✝
      ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerLeft X✝ (HAdd.hAdd f✝ g✝)). …
    -/
    refine TensorProduct.ext (LinearMap.ext fun x => LinearMap.ext fun y => ?_)
    /-
      case refine_3.hf
      R : Type u
      inst✝ : CommRing R
      X✝ Y✝ Z✝ : ModuleCat R
      f✝ g✝ : Quiver.Hom Y✝ Z✝
      x : ↑X✝
      y : ↑Y✝
      ⊢ Eq ((((TensorProduct.mk R ↑X✝ ↑Y✝).compr₂ (CategoryTheory.MonoidalCategorySt …
    -/
    simp only [LinearMap.compr₂_apply, TensorProduct.mk_apply, hom_add, LinearMap.add_apply]
    -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
    /-
      case refine_3.hf
      R : Type u
      inst✝ : CommRing R
      X✝ Y✝ Z✝ : ModuleCat R
      f✝ g✝ : Quiver.Hom Y✝ Z✝
      x : ↑X✝
      y : ↑Y✝
      ⊢ Eq ((CategoryTheory.MonoidalCategoryStruct.whiskerLeft X✝ (HAdd.hAdd f✝ g✝)) …
    -/
    erw [MonoidalCategory.whiskerLeft_apply, MonoidalCategory.whiskerLeft_apply]
    /-
      case refine_3.hf
      R : Type u
      inst✝ : CommRing R
      X✝ Y✝ Z✝ : ModuleCat R
      f✝ g✝ : Quiver.Hom Y✝ Z✝
      x : ↑X✝
      y : ↑Y✝
      ⊢ Eq (TensorProduct.tmul R x ((HAdd.hAdd f✝ g✝).hom y)) (HAdd.hAdd (TensorProd …
    -/
    erw [MonoidalCategory.whiskerLeft_apply]
    /-
      case refine_3.hf
      R : Type u
      inst✝ : CommRing R
      X✝ Y✝ Z✝ : ModuleCat R
      f✝ g✝ : Quiver.Hom Y✝ Z✝
      x : ↑X✝
      y : ↑Y✝
      ⊢ Eq (TensorProduct.tmul R x ((HAdd.hAdd f✝ g✝).hom y)) (HAdd.hAdd (TensorProd …
    -/
    simp [TensorProduct.tmul_add]
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      R : Type u
      inst✝ : CommRing R
      ⊢ ∀ {X Y Z : ModuleCat R} (f g : Quiver.Hom Y Z), Eq (CategoryTheory.MonoidalC …
    -/
  · intros
    /-
      case refine_4
      R : Type u
      inst✝ : CommRing R
      X✝ Y✝ Z✝ : ModuleCat R
      f✝ g✝ : Quiver.Hom Y✝ Z✝
      ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight (HAdd.hAdd f✝ g✝) X✝) …
    -/
    ext : 1
    /-
      case refine_4.hf
      R : Type u
      inst✝ : CommRing R
      X✝ Y✝ Z✝ : ModuleCat R
      f✝ g✝ : Quiver.Hom Y✝ Z✝
      ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight (HAdd.hAdd f✝ g✝) X✝) …
    -/
    refine TensorProduct.ext (LinearMap.ext fun x => LinearMap.ext fun y => ?_)
    /-
      case refine_4.hf
      R : Type u
      inst✝ : CommRing R
      X✝ Y✝ Z✝ : ModuleCat R
      f✝ g✝ : Quiver.Hom Y✝ Z✝
      x : ↑Y✝
      y : ↑X✝
      ⊢ Eq ((((TensorProduct.mk R ↑Y✝ ↑X✝).compr₂ (CategoryTheory.MonoidalCategorySt …
    -/
    simp only [LinearMap.compr₂_apply, TensorProduct.mk_apply, hom_add, LinearMap.add_apply]
    -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
    /-
      case refine_4.hf
      R : Type u
      inst✝ : CommRing R
      X✝ Y✝ Z✝ : ModuleCat R
      f✝ g✝ : Quiver.Hom Y✝ Z✝
      x : ↑Y✝
      y : ↑X✝
      ⊢ Eq ((CategoryTheory.MonoidalCategoryStruct.whiskerRight (HAdd.hAdd f✝ g✝) X✝ …
    -/
    erw [MonoidalCategory.whiskerRight_apply, MonoidalCategory.whiskerRight_apply]
    /-
      case refine_4.hf
      R : Type u
      inst✝ : CommRing R
      X✝ Y✝ Z✝ : ModuleCat R
      f✝ g✝ : Quiver.Hom Y✝ Z✝
      x : ↑Y✝
      y : ↑X✝
      ⊢ Eq (TensorProduct.tmul R ((HAdd.hAdd f✝ g✝).hom x) y) (HAdd.hAdd (TensorProd …
    -/
    erw [MonoidalCategory.whiskerRight_apply]
    /-
      case refine_4.hf
      R : Type u
      inst✝ : CommRing R
      X✝ Y✝ Z✝ : ModuleCat R
      f✝ g✝ : Quiver.Hom Y✝ Z✝
      x : ↑Y✝
      y : ↑X✝
      ⊢ Eq (TensorProduct.tmul R ((HAdd.hAdd f✝ g✝).hom x) y) (HAdd.hAdd (TensorProd …
    -/
    simp [TensorProduct.add_tmul]
    /-
      🎉 no goals
    -/


instance : MonoidalLinear R (ModuleCat.{u} R) := by
  /-
    R : Type u
    inst✝ : CommRing R
    ⊢ CategoryTheory.MonoidalLinear R (ModuleCat R)
  -/
  refine ⟨?_, ?_⟩
    /-
      case refine_1
      R : Type u
      inst✝ : CommRing R
      ⊢ ∀ (X : ModuleCat R) {Y Z : ModuleCat R} (r : R) (f : Quiver.Hom Y Z), Eq (Ca …
    -/
  · intros
    /-
      case refine_1
      R : Type u
      inst✝ : CommRing R
      X✝ Y✝ Z✝ : ModuleCat R
      r✝ : R
      f✝ : Quiver.Hom Y✝ Z✝
      ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerLeft X✝ (HSMul.hSMul r✝ f✝) …
    -/
    ext : 1
    /-
      case refine_1.hf
      R : Type u
      inst✝ : CommRing R
      X✝ Y✝ Z✝ : ModuleCat R
      r✝ : R
      f✝ : Quiver.Hom Y✝ Z✝
      ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerLeft X✝ (HSMul.hSMul r✝ f✝) …
    -/
    refine TensorProduct.ext (LinearMap.ext fun x => LinearMap.ext fun y => ?_)
    /-
      case refine_1.hf
      R : Type u
      inst✝ : CommRing R
      X✝ Y✝ Z✝ : ModuleCat R
      r✝ : R
      f✝ : Quiver.Hom Y✝ Z✝
      x : ↑X✝
      y : ↑Y✝
      ⊢ Eq ((((TensorProduct.mk R ↑X✝ ↑Y✝).compr₂ (CategoryTheory.MonoidalCategorySt …
    -/
    simp only [LinearMap.compr₂_apply, TensorProduct.mk_apply, hom_smul, LinearMap.smul_apply]
    -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
    /-
      case refine_1.hf
      R : Type u
      inst✝ : CommRing R
      X✝ Y✝ Z✝ : ModuleCat R
      r✝ : R
      f✝ : Quiver.Hom Y✝ Z✝
      x : ↑X✝
      y : ↑Y✝
      ⊢ Eq ((CategoryTheory.MonoidalCategoryStruct.whiskerLeft X✝ (HSMul.hSMul r✝ f✝ …
    -/
    erw [MonoidalCategory.whiskerLeft_apply, MonoidalCategory.whiskerLeft_apply]
    /-
      case refine_1.hf
      R : Type u
      inst✝ : CommRing R
      X✝ Y✝ Z✝ : ModuleCat R
      r✝ : R
      f✝ : Quiver.Hom Y✝ Z✝
      x : ↑X✝
      y : ↑Y✝
      ⊢ Eq (TensorProduct.tmul R x ((HSMul.hSMul r✝ f✝).hom y)) (HSMul.hSMul r✝ (Ten …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u
      inst✝ : CommRing R
      ⊢ ∀ (r : R) {Y Z : ModuleCat R} (f : Quiver.Hom Y Z) (X : ModuleCat R), Eq (Ca …
    -/
  · intros
    /-
      case refine_2
      R : Type u
      inst✝ : CommRing R
      r✝ : R
      Y✝ Z✝ : ModuleCat R
      f✝ : Quiver.Hom Y✝ Z✝
      X✝ : ModuleCat R
      ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight (HSMul.hSMul r✝ f✝) X …
    -/
    ext : 1
    /-
      case refine_2.hf
      R : Type u
      inst✝ : CommRing R
      r✝ : R
      Y✝ Z✝ : ModuleCat R
      f✝ : Quiver.Hom Y✝ Z✝
      X✝ : ModuleCat R
      ⊢ Eq (CategoryTheory.MonoidalCategoryStruct.whiskerRight (HSMul.hSMul r✝ f✝) X …
    -/
    refine TensorProduct.ext (LinearMap.ext fun x => LinearMap.ext fun y => ?_)
    /-
      case refine_2.hf
      R : Type u
      inst✝ : CommRing R
      r✝ : R
      Y✝ Z✝ : ModuleCat R
      f✝ : Quiver.Hom Y✝ Z✝
      X✝ : ModuleCat R
      x : ↑Y✝
      y : ↑X✝
      ⊢ Eq ((((TensorProduct.mk R ↑Y✝ ↑X✝).compr₂ (CategoryTheory.MonoidalCategorySt …
    -/
    simp only [LinearMap.compr₂_apply, TensorProduct.mk_apply, hom_smul, LinearMap.smul_apply]
    -- This used to be `rw`, but we need `erw` after https://github.com/leanprover/lean4/pull/2644
    /-
      case refine_2.hf
      R : Type u
      inst✝ : CommRing R
      r✝ : R
      Y✝ Z✝ : ModuleCat R
      f✝ : Quiver.Hom Y✝ Z✝
      X✝ : ModuleCat R
      x : ↑Y✝
      y : ↑X✝
      ⊢ Eq ((CategoryTheory.MonoidalCategoryStruct.whiskerRight (HSMul.hSMul r✝ f✝)  …
    -/
    erw [MonoidalCategory.whiskerRight_apply, MonoidalCategory.whiskerRight_apply]
    /-
      case refine_2.hf
      R : Type u
      inst✝ : CommRing R
      r✝ : R
      Y✝ Z✝ : ModuleCat R
      f✝ : Quiver.Hom Y✝ Z✝
      X✝ : ModuleCat R
      x : ↑Y✝
      y : ↑X✝
      ⊢ Eq (TensorProduct.tmul R ((HSMul.hSMul r✝ f✝).hom x) y) (HSMul.hSMul r✝ (Ten …
    -/
    simp [TensorProduct.smul_tmul, TensorProduct.tmul_smul]
    /-
      🎉 no goals
    -/


@[simp] lemma ofHom₂_compr₂ {M N P Q : ModuleCat.{u} R} (f : M →ₗ[R] N →ₗ[R] P) (g : P →ₗ[R] Q):
    ofHom₂ (f.compr₂ g) = ofHom₂ f ≫ ofHom (Linear.rightComp R _ (ofHom g)) := rfl


