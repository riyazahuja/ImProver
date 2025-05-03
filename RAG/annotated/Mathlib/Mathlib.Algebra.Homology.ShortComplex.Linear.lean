instance : SMul R (S₁ ⟶ S₂) where
  smul a φ :=
    { τ₁ := a • φ.τ₁
      τ₂ := a • φ.τ₂
      τ₃ := a • φ.τ₃ }


@[simp] lemma smul_τ₁ (a : R) (φ : S₁ ⟶ S₂) : (a • φ).τ₁ = a • φ.τ₁ := rfl

@[simp] lemma smul_τ₂ (a : R) (φ : S₁ ⟶ S₂) : (a • φ).τ₂ = a • φ.τ₂ := rfl

@[simp] lemma smul_τ₃ (a : R) (φ : S₁ ⟶ S₂) : (a • φ).τ₃ = a • φ.τ₃ := rfl


instance : Module R (S₁ ⟶ S₂) where
                  /-
                    R : Type u_1
                    C : Type u_2
                    inst✝³ : Semiring R
                    inst✝² : CategoryTheory.Category.{?u.3563, u_2} C
                    inst✝¹ : CategoryTheory.Preadditive C
                    inst✝ : CategoryTheory.Linear R C
                    S₁ S₂ : CategoryTheory.ShortComplex C
                    ⊢ ∀ (x : Quiver.Hom S₁ S₂), Eq (HSMul.hSMul 0 x) 0
                  -/
                 /-
                   R : Type u_1
                   C : Type u_2
                   inst✝³ : Semiring R
                   inst✝² : CategoryTheory.Category.{?u.3563, u_2} C
                   inst✝¹ : CategoryTheory.Preadditive C
                   inst✝ : CategoryTheory.Linear R C
                   S₁ S₂ : CategoryTheory.ShortComplex C
                   ⊢ ∀ (b : Quiver.Hom S₁ S₂), Eq (HSMul.hSMul 1 b) b
                 -/
  zero_smul := by aesop_cat
                 /-
                   🎉 no goals
                 -/
                  /-
                    R : Type u_1
                    C : Type u_2
                    inst✝³ : Semiring R
                    inst✝² : CategoryTheory.Category.{?u.3563, u_2} C
                    inst✝¹ : CategoryTheory.Preadditive C
                    inst✝ : CategoryTheory.Linear R C
                    S₁ S₂ : CategoryTheory.ShortComplex C
                    ⊢ ∀ (a : R), Eq (HSMul.hSMul a 0) 0
                  -/
                  /-
                    🎉 no goals
                  -/
                  /-
                    🎉 no goals
                  -/
                 /-
                   R : Type u_1
                   C : Type u_2
                   inst✝³ : Semiring R
                   inst✝² : CategoryTheory.Category.{?u.3563, u_2} C
                   inst✝¹ : CategoryTheory.Preadditive C
                   inst✝ : CategoryTheory.Linear R C
                   S₁ S₂ : CategoryTheory.ShortComplex C
                   ⊢ ∀ (x y : R) (b : Quiver.Hom S₁ S₂), Eq (HSMul.hSMul (HMul.hMul x y) b) (HSMu …
                 -/
                 /-
                   R : Type u_1
                   C : Type u_2
                   inst✝³ : Semiring R
                   inst✝² : CategoryTheory.Category.{?u.3563, u_2} C
                   inst✝¹ : CategoryTheory.Preadditive C
                   inst✝ : CategoryTheory.Linear R C
                   S₁ S₂ : CategoryTheory.ShortComplex C
                   ⊢ ∀ (a : R) (x y : Quiver.Hom S₁ S₂), Eq (HSMul.hSMul a (HAdd.hAdd x y)) (HAdd …
                 -/
                 /-
                   🎉 no goals
                 -/
  one_smul := by aesop_cat
                 /-
                   🎉 no goals
                 -/
                 /-
                   R : Type u_1
                   C : Type u_2
                   inst✝³ : Semiring R
                   inst✝² : CategoryTheory.Category.{?u.3563, u_2} C
                   inst✝¹ : CategoryTheory.Preadditive C
                   inst✝ : CategoryTheory.Linear R C
                   S₁ S₂ : CategoryTheory.ShortComplex C
                   ⊢ ∀ (r s : R) (x : Quiver.Hom S₁ S₂), Eq (HSMul.hSMul (HAdd.hAdd r s) x) (HAdd …
                 -/
  smul_zero := by aesop_cat
                 /-
                   🎉 no goals
                 -/
  smul_add := by aesop_cat
  add_smul := by aesop_cat
  mul_smul := by aesop_cat


instance : Linear R (ShortComplex C) where


/-- Given a left homology map data for morphism `φ`, this is the induced left homology
map data for `a • φ`. -/
@[simps]
def smul (a : R) : LeftHomologyMapData (a • φ) h₁ h₂ where
  φK := a • γ.φK
  φH := a • γ.φH


@[simp]
lemma leftHomologyMap'_smul :
    leftHomologyMap' (a • φ) h₁ h₂ = a • leftHomologyMap' φ h₁ h₂ := by
  /-
    R : Type u_1
    C : Type u_2
    inst✝³ : Semiring R
    inst✝² : CategoryTheory.Category.{u_3, u_2} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Linear R C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.LeftHomologyData
    h₂ : S₂.LeftHomologyData
    a : R
    ⊢ Eq (CategoryTheory.ShortComplex.leftHomologyMap' (HSMul.hSMul a φ) h₁ h₂) (H …
  -/
  have γ : LeftHomologyMapData φ h₁ h₂ := default
  /-
    R : Type u_1
    C : Type u_2
    inst✝³ : Semiring R
    inst✝² : CategoryTheory.Category.{u_3, u_2} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Linear R C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.LeftHomologyData
    h₂ : S₂.LeftHomologyData
    a : R
    γ : CategoryTheory.ShortComplex.LeftHomologyMapData φ h₁ h₂
    ⊢ Eq (CategoryTheory.ShortComplex.leftHomologyMap' (HSMul.hSMul a φ) h₁ h₂) (H …
  -/
  simp only [(γ.smul a).leftHomologyMap'_eq, LeftHomologyMapData.smul_φH, γ.leftHomologyMap'_eq]
  /-
    🎉 no goals
  -/


@[simp]
lemma cyclesMap'_smul :
    cyclesMap' (a • φ) h₁ h₂ = a • cyclesMap' φ h₁ h₂ := by
  /-
    R : Type u_1
    C : Type u_2
    inst✝³ : Semiring R
    inst✝² : CategoryTheory.Category.{u_3, u_2} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Linear R C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.LeftHomologyData
    h₂ : S₂.LeftHomologyData
    a : R
    ⊢ Eq (CategoryTheory.ShortComplex.cyclesMap' (HSMul.hSMul a φ) h₁ h₂) (HSMul.h …
  -/
  have γ : LeftHomologyMapData φ h₁ h₂ := default
  /-
    R : Type u_1
    C : Type u_2
    inst✝³ : Semiring R
    inst✝² : CategoryTheory.Category.{u_3, u_2} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Linear R C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.LeftHomologyData
    h₂ : S₂.LeftHomologyData
    a : R
    γ : CategoryTheory.ShortComplex.LeftHomologyMapData φ h₁ h₂
    ⊢ Eq (CategoryTheory.ShortComplex.cyclesMap' (HSMul.hSMul a φ) h₁ h₂) (HSMul.h …
  -/
  simp only [(γ.smul a).cyclesMap'_eq, LeftHomologyMapData.smul_φK, γ.cyclesMap'_eq]
  /-
    🎉 no goals
  -/


@[simp]
lemma leftHomologyMap_smul : leftHomologyMap (a • φ) = a • leftHomologyMap φ :=
  leftHomologyMap'_smul _ _ _ _


@[simp]
lemma cyclesMap_smul : cyclesMap (a • φ) = a • cyclesMap φ :=
  cyclesMap'_smul _ _ _ _


instance leftHomologyFunctor_linear [HasKernels C] [HasCokernels C] :
    Functor.Linear R (leftHomologyFunctor C) where


instance cyclesFunctor_linear [HasKernels C] [HasCokernels C] :
    Functor.Linear R (cyclesFunctor C) where


/-- Given a right homology map data for morphism `φ`, this is the induced right homology
map data for `a • φ`. -/
@[simps]
def smul (a : R) : RightHomologyMapData (a • φ) h₁ h₂ where
  φQ := a • γ.φQ
  φH := a • γ.φH


@[simp]
lemma rightHomologyMap'_smul :
    rightHomologyMap' (a • φ) h₁ h₂ = a • rightHomologyMap' φ h₁ h₂ := by
  /-
    R : Type u_1
    C : Type u_2
    inst✝³ : Semiring R
    inst✝² : CategoryTheory.Category.{u_3, u_2} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Linear R C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.RightHomologyData
    h₂ : S₂.RightHomologyData
    a : R
    ⊢ Eq (CategoryTheory.ShortComplex.rightHomologyMap' (HSMul.hSMul a φ) h₁ h₂) ( …
  -/
  have γ : RightHomologyMapData φ h₁ h₂ := default
  /-
    R : Type u_1
    C : Type u_2
    inst✝³ : Semiring R
    inst✝² : CategoryTheory.Category.{u_3, u_2} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Linear R C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.RightHomologyData
    h₂ : S₂.RightHomologyData
    a : R
    γ : CategoryTheory.ShortComplex.RightHomologyMapData φ h₁ h₂
    ⊢ Eq (CategoryTheory.ShortComplex.rightHomologyMap' (HSMul.hSMul a φ) h₁ h₂) ( …
  -/
  simp only [(γ.smul a).rightHomologyMap'_eq, RightHomologyMapData.smul_φH, γ.rightHomologyMap'_eq]
  /-
    🎉 no goals
  -/


@[simp]
lemma opcyclesMap'_smul :
    opcyclesMap' (a • φ) h₁ h₂ = a • opcyclesMap' φ h₁ h₂ := by
  /-
    R : Type u_1
    C : Type u_2
    inst✝³ : Semiring R
    inst✝² : CategoryTheory.Category.{u_3, u_2} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Linear R C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.RightHomologyData
    h₂ : S₂.RightHomologyData
    a : R
    ⊢ Eq (CategoryTheory.ShortComplex.opcyclesMap' (HSMul.hSMul a φ) h₁ h₂) (HSMul …
  -/
  have γ : RightHomologyMapData φ h₁ h₂ := default
  /-
    R : Type u_1
    C : Type u_2
    inst✝³ : Semiring R
    inst✝² : CategoryTheory.Category.{u_3, u_2} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.Linear R C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.RightHomologyData
    h₂ : S₂.RightHomologyData
    a : R
    γ : CategoryTheory.ShortComplex.RightHomologyMapData φ h₁ h₂
    ⊢ Eq (CategoryTheory.ShortComplex.opcyclesMap' (HSMul.hSMul a φ) h₁ h₂) (HSMul …
  -/
  simp only [(γ.smul a).opcyclesMap'_eq, RightHomologyMapData.smul_φQ, γ.opcyclesMap'_eq]
  /-
    🎉 no goals
  -/


@[simp]
lemma rightHomologyMap_smul : rightHomologyMap (a • φ) = a • rightHomologyMap φ :=
  rightHomologyMap'_smul _ _ _ _


@[simp]
lemma opcyclesMap_smul : opcyclesMap (a • φ) = a • opcyclesMap φ :=
  opcyclesMap'_smul _ _ _ _


instance rightHomologyFunctor_linear [HasKernels C] [HasCokernels C] :
    Functor.Linear R (rightHomologyFunctor C) where


instance opcyclesFunctor_linear [HasKernels C] [HasCokernels C] :
    Functor.Linear R (opcyclesFunctor C) where


/-- Given a homology map data for a morphism `φ`, this is the induced homology
map data for `a • φ`. -/
@[simps]
def smul (a : R) : HomologyMapData (a • φ) h₁ h₂ where
  left := γ.left.smul a
  right := γ.right.smul a


@[simp]
lemma homologyMap'_smul :
    homologyMap' (a • φ) h₁ h₂ = a • homologyMap' φ h₁ h₂ :=
  leftHomologyMap'_smul _ _ _ _


@[simp]
lemma homologyMap_smul [S₁.HasHomology] [S₂.HasHomology] :
    homologyMap (a • φ) = a • homologyMap φ :=
  homologyMap'_smul _ _ _


instance homologyFunctor_linear [CategoryWithHomology C] :
    Functor.Linear R (homologyFunctor C) where


/-- Homotopy between morphisms of short complexes is compatible with the scalar multiplication. -/
@[simps]
def Homotopy.smul {φ₁ φ₂ : S₁ ⟶ S₂} (h : Homotopy φ₁ φ₂) (a : R) :
    Homotopy (a • φ₁) (a • φ₂) where
  h₀ := a • h.h₀
  h₁ := a • h.h₁
  h₂ := a • h.h₂
  h₃ := a • h.h₃
  comm₁ := by
    /-
      R : Type u_1
      C : Type u_2
      inst✝³ : Semiring R
      inst✝² : CategoryTheory.Category.{?u.75975, u_2} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Linear R C
      S₁ S₂ : CategoryTheory.ShortComplex C
      φ₁ φ₂ : Quiver.Hom S₁ S₂
      h : CategoryTheory.ShortComplex.Homotopy φ₁ φ₂
      a : R
      ⊢ Eq (HSMul.hSMul a φ₁).τ₁ (HAdd.hAdd (HAdd.hAdd (CategoryTheory.CategoryStruc …
    -/
    dsimp
    /-
      R : Type u_1
      C : Type u_2
      inst✝³ : Semiring R
      inst✝² : CategoryTheory.Category.{?u.75975, u_2} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Linear R C
      S₁ S₂ : CategoryTheory.ShortComplex C
      φ₁ φ₂ : Quiver.Hom S₁ S₂
      h : CategoryTheory.ShortComplex.Homotopy φ₁ φ₂
      a : R
      ⊢ Eq (HSMul.hSMul a φ₁.τ₁) (HAdd.hAdd (HAdd.hAdd (CategoryTheory.CategoryStruc …
    -/
    rw [h.comm₁]
    /-
      R : Type u_1
      C : Type u_2
      inst✝³ : Semiring R
      inst✝² : CategoryTheory.Category.{?u.75975, u_2} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Linear R C
      S₁ S₂ : CategoryTheory.ShortComplex C
      φ₁ φ₂ : Quiver.Hom S₁ S₂
      h : CategoryTheory.ShortComplex.Homotopy φ₁ φ₂
      a : R
      ⊢ Eq (HSMul.hSMul a (HAdd.hAdd (HAdd.hAdd (CategoryTheory.CategoryStruct.comp  …
    -/
    simp only [smul_add, Linear.comp_smul]
    /-
      🎉 no goals
    -/
  comm₂ := by
    /-
      R : Type u_1
      C : Type u_2
      inst✝³ : Semiring R
      inst✝² : CategoryTheory.Category.{?u.75975, u_2} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Linear R C
      S₁ S₂ : CategoryTheory.ShortComplex C
      φ₁ φ₂ : Quiver.Hom S₁ S₂
      h : CategoryTheory.ShortComplex.Homotopy φ₁ φ₂
      a : R
      ⊢ Eq (HSMul.hSMul a φ₁).τ₂ (HAdd.hAdd (HAdd.hAdd (CategoryTheory.CategoryStruc …
    -/
    dsimp
    /-
      R : Type u_1
      C : Type u_2
      inst✝³ : Semiring R
      inst✝² : CategoryTheory.Category.{?u.75975, u_2} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Linear R C
      S₁ S₂ : CategoryTheory.ShortComplex C
      φ₁ φ₂ : Quiver.Hom S₁ S₂
      h : CategoryTheory.ShortComplex.Homotopy φ₁ φ₂
      a : R
      ⊢ Eq (HSMul.hSMul a φ₁.τ₂) (HAdd.hAdd (HAdd.hAdd (CategoryTheory.CategoryStruc …
    -/
    rw [h.comm₂]
    /-
      R : Type u_1
      C : Type u_2
      inst✝³ : Semiring R
      inst✝² : CategoryTheory.Category.{?u.75975, u_2} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Linear R C
      S₁ S₂ : CategoryTheory.ShortComplex C
      φ₁ φ₂ : Quiver.Hom S₁ S₂
      h : CategoryTheory.ShortComplex.Homotopy φ₁ φ₂
      a : R
      ⊢ Eq (HSMul.hSMul a (HAdd.hAdd (HAdd.hAdd (CategoryTheory.CategoryStruct.comp  …
    -/
    simp only [smul_add, Linear.comp_smul, Linear.smul_comp]
    /-
      🎉 no goals
    -/
  comm₃ := by
    /-
      R : Type u_1
      C : Type u_2
      inst✝³ : Semiring R
      inst✝² : CategoryTheory.Category.{?u.75975, u_2} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Linear R C
      S₁ S₂ : CategoryTheory.ShortComplex C
      φ₁ φ₂ : Quiver.Hom S₁ S₂
      h : CategoryTheory.ShortComplex.Homotopy φ₁ φ₂
      a : R
      ⊢ Eq (HSMul.hSMul a φ₁).τ₃ (HAdd.hAdd (HAdd.hAdd (HSMul.hSMul a h.h₃) (Categor …
    -/
    dsimp
    /-
      R : Type u_1
      C : Type u_2
      inst✝³ : Semiring R
      inst✝² : CategoryTheory.Category.{?u.75975, u_2} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Linear R C
      S₁ S₂ : CategoryTheory.ShortComplex C
      φ₁ φ₂ : Quiver.Hom S₁ S₂
      h : CategoryTheory.ShortComplex.Homotopy φ₁ φ₂
      a : R
      ⊢ Eq (HSMul.hSMul a φ₁.τ₃) (HAdd.hAdd (HAdd.hAdd (HSMul.hSMul a h.h₃) (Categor …
    -/
    rw [h.comm₃]
    /-
      R : Type u_1
      C : Type u_2
      inst✝³ : Semiring R
      inst✝² : CategoryTheory.Category.{?u.75975, u_2} C
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : CategoryTheory.Linear R C
      S₁ S₂ : CategoryTheory.ShortComplex C
      φ₁ φ₂ : Quiver.Hom S₁ S₂
      h : CategoryTheory.ShortComplex.Homotopy φ₁ φ₂
      a : R
      ⊢ Eq (HSMul.hSMul a (HAdd.hAdd (HAdd.hAdd h.h₃ (CategoryTheory.CategoryStruct. …
    -/
    simp only [smul_add, Linear.smul_comp]
    /-
      🎉 no goals
    -/


