/-- The property that morphisms between single complexes in arbitrary degrees are `w`-small
in the derived category. -/
abbrev HasExt : Prop :=
  ∀ (X Y : C), HasSmallLocalizedShiftedHom.{w} (HomologicalComplex.quasiIso C (ComplexShape.up ℤ)) ℤ
    ((CochainComplex.singleFunctor C 0).obj X) ((CochainComplex.singleFunctor C 0).obj Y)

-- TODO: when the canonical t-structure is formalized, replace `n : ℤ` by `n : ℕ`

lemma hasExt_iff [HasDerivedCategory.{w'} C] :
    HasExt.{w} C ↔ ∀ (X Y : C) (n : ℤ), Small.{w}
      ((singleFunctor C 0).obj X ⟶
        (((singleFunctor C 0).obj Y)⟦n⟧)) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : HasDerivedCategory C
    ⊢ Iff (CategoryTheory.HasExt C) (∀ (X Y : C) (n : Int), Small.{w, w'} (Quiver. …
  -/
  dsimp [HasExt]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : HasDerivedCategory C
    ⊢ Iff (∀ (X Y : C), CategoryTheory.Localization.HasSmallLocalizedShiftedHom (H …
  -/
  simp only [hasSmallLocalizedShiftedHom_iff _ _ Q]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : HasDerivedCategory C
    ⊢ Iff (∀ (X Y : C) (a b : Int), Small.{w, w'} (Quiver.Hom ((CategoryTheory.shi …
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : HasDerivedCategory C
      ⊢ (∀ (X Y : C) (a b : Int), Small.{w, w'} (Quiver.Hom ((CategoryTheory.shiftFu …
    -/
  · intro h X Y n
    exact (small_congr ((shiftFunctorZero _ ℤ).app
      ((singleFunctor C 0).obj X)).homFromEquiv).1 (h X Y 0 n)
    /-
      case mpr
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : HasDerivedCategory C
      ⊢ (∀ (X Y : C) (n : Int), Small.{w, w'} (Quiver.Hom ((DerivedCategory.singleFu …
    -/
  · intro h X Y a b
    /-
      case mpr
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : HasDerivedCategory C
      h : ∀ (X Y : C) (n : Int), Small.{w, w'} (Quiver.Hom ((DerivedCategory.singleF …
      X Y : C
      a b : Int
      ⊢ Small.{w, w'} (Quiver.Hom ((CategoryTheory.shiftFunctor (DerivedCategory C)  …
    -/
    refine (small_congr ?_).1 (h X Y (b - a))
    exact (Functor.FullyFaithful.ofFullyFaithful
      (shiftFunctor _ a)).homEquiv.trans
      ((shiftFunctorAdd' _ _ _ _ (Int.sub_add_cancel b a)).symm.app _).homToEquiv


lemma hasExt_of_hasDerivedCategory [HasDerivedCategory.{w} C] : HasExt.{w} C := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : HasDerivedCategory C
    ⊢ CategoryTheory.HasExt C
  -/
  rw [hasExt_iff.{w}]
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : HasDerivedCategory C
    ⊢ ∀ (X Y : C) (n : Int), Small.{w, w} (Quiver.Hom ((DerivedCategory.singleFunc …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- A Ext-group in an abelian category `C`, defined as a `Type w` when `[HasExt.{w} C]`. -/
def Ext (X Y : C) (n : ℕ) : Type w :=
  SmallShiftedHom.{w} (HomologicalComplex.quasiIso C (ComplexShape.up ℤ))
    ((CochainComplex.singleFunctor C 0).obj X)
    ((CochainComplex.singleFunctor C 0).obj Y) (n : ℤ)


/-- The composition of `Ext`. -/
noncomputable def comp {a b : ℕ} (α : Ext X Y a) (β : Ext Y Z b) {c : ℕ} (h : a + b = c) :
    Ext X Z c :=
                               /-
                                 C : Type u
                                 inst✝² : CategoryTheory.Category.{v, u} C
                                 inst✝¹ : CategoryTheory.Abelian C
                                 inst✝ : CategoryTheory.HasExt C
                                 X Y Z T : C
                                 a b : Nat
                                 α : CategoryTheory.Abelian.Ext X Y a
                                 β : CategoryTheory.Abelian.Ext Y Z b
                                 c : Nat
                                 h : Eq (HAdd.hAdd a b) c
                                 ⊢ Eq (HAdd.hAdd ↑b ↑a) ↑c
                               -/
  SmallShiftedHom.comp α β (by omega)
                               /-
                                 🎉 no goals
                               -/


lemma comp_assoc {a₁ a₂ a₃ a₁₂ a₂₃ a : ℕ} (α : Ext X Y a₁) (β : Ext Y Z a₂) (γ : Ext Z T a₃)
    (h₁₂ : a₁ + a₂ = a₁₂) (h₂₃ : a₂ + a₃ = a₂₃) (h : a₁ + a₂ + a₃ = a) :
                                                /-
                                                  C : Type u
                                                  inst✝² : CategoryTheory.Category.{v, u} C
                                                  inst✝¹ : CategoryTheory.Abelian C
                                                  inst✝ : CategoryTheory.HasExt C
                                                  X Y Z T : C
                                                  a₁ a₂ a₃ a₁₂ a₂₃ a : Nat
                                                  α : CategoryTheory.Abelian.Ext X Y a₁
                                                  β : CategoryTheory.Abelian.Ext Y Z a₂
                                                  γ : CategoryTheory.Abelian.Ext Z T a₃
                                                  h₁₂ : Eq (HAdd.hAdd a₁ a₂) a₁₂
                                                  h₂₃ : Eq (HAdd.hAdd a₂ a₃) a₂₃
                                                  h : Eq (HAdd.hAdd (HAdd.hAdd a₁ a₂) a₃) a
                                                  ⊢ Eq (HAdd.hAdd a₁₂ a₃) a
                                                -/
    (α.comp β h₁₂).comp γ (show a₁₂ + a₃ = a by omega) =
                                                /-
                                                  🎉 no goals
                                                -/
                                /-
                                  C : Type u
                                  inst✝² : CategoryTheory.Category.{v, u} C
                                  inst✝¹ : CategoryTheory.Abelian C
                                  inst✝ : CategoryTheory.HasExt C
                                  X Y Z T : C
                                  a₁ a₂ a₃ a₁₂ a₂₃ a : Nat
                                  α : CategoryTheory.Abelian.Ext X Y a₁
                                  β : CategoryTheory.Abelian.Ext Y Z a₂
                                  γ : CategoryTheory.Abelian.Ext Z T a₃
                                  h₁₂ : Eq (HAdd.hAdd a₁ a₂) a₁₂
                                  h₂₃ : Eq (HAdd.hAdd a₂ a₃) a₂₃
                                  h : Eq (HAdd.hAdd (HAdd.hAdd a₁ a₂) a₃) a
                                  ⊢ Eq (HAdd.hAdd a₁ a₂₃) a
                                -/
      α.comp (β.comp γ h₂₃) (by omega) :=
                                /-
                                  🎉 no goals
                                -/
                                             /-
                                               C : Type u
                                               inst✝² : CategoryTheory.Category.{v, u} C
                                               inst✝¹ : CategoryTheory.Abelian C
                                               inst✝ : CategoryTheory.HasExt C
                                               X Y Z T : C
                                               a₁ a₂ a₃ a₁₂ a₂₃ a : Nat
                                               α : CategoryTheory.Abelian.Ext X Y a₁
                                               β : CategoryTheory.Abelian.Ext Y Z a₂
                                               γ : CategoryTheory.Abelian.Ext Z T a₃
                                               h₁₂ : Eq (HAdd.hAdd a₁ a₂) a₁₂
                                               h₂₃ : Eq (HAdd.hAdd a₂ a₃) a₂₃
                                               h : Eq (HAdd.hAdd (HAdd.hAdd a₁ a₂) a₃) a
                                               ⊢ Eq (HAdd.hAdd (HAdd.hAdd ↑a₃ ↑a₂) ↑a₁) ↑a
                                             -/
  SmallShiftedHom.comp_assoc _ _ _ _ _ _ (by omega)
                                             /-
                                               🎉 no goals
                                             -/


@[simp]
lemma comp_assoc_of_second_deg_zero
    {a₁ a₃ a₁₃ : ℕ} (α : Ext X Y a₁) (β : Ext Y Z 0) (γ : Ext Z T a₃)
    (h₁₃ : a₁ + a₃ = a₁₃) :
    (α.comp β (add_zero _)).comp γ h₁₃ = α.comp (β.comp γ (zero_add _)) h₁₃ := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    X Y Z T : C
    a₁ a₃ a₁₃ : Nat
    α : CategoryTheory.Abelian.Ext X Y a₁
    β : CategoryTheory.Abelian.Ext Y Z 0
    γ : CategoryTheory.Abelian.Ext Z T a₃
    h₁₃ : Eq (HAdd.hAdd a₁ a₃) a₁₃
    ⊢ Eq ((α.comp β ⋯).comp γ h₁₃) (α.comp (β.comp γ ⋯) h₁₃)
  -/
  apply comp_assoc
  /-
    case h
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    X Y Z T : C
    a₁ a₃ a₁₃ : Nat
    α : CategoryTheory.Abelian.Ext X Y a₁
    β : CategoryTheory.Abelian.Ext Y Z 0
    γ : CategoryTheory.Abelian.Ext Z T a₃
    h₁₃ : Eq (HAdd.hAdd a₁ a₃) a₁₃
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd a₁ 0) a₃) a₁₃
  -/
  omega
  /-
    🎉 no goals
  -/


@[simp]
lemma comp_assoc_of_third_deg_zero
    {a₁ a₂ a₁₂ : ℕ} (α : Ext X Y a₁) (β : Ext Y Z a₂) (γ : Ext Z T 0)
    (h₁₂ : a₁ + a₂ = a₁₂) :
    (α.comp β h₁₂).comp γ (add_zero _) = α.comp (β.comp γ (add_zero _)) h₁₂ := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    X Y Z T : C
    a₁ a₂ a₁₂ : Nat
    α : CategoryTheory.Abelian.Ext X Y a₁
    β : CategoryTheory.Abelian.Ext Y Z a₂
    γ : CategoryTheory.Abelian.Ext Z T 0
    h₁₂ : Eq (HAdd.hAdd a₁ a₂) a₁₂
    ⊢ Eq ((α.comp β h₁₂).comp γ ⋯) (α.comp (β.comp γ ⋯) h₁₂)
  -/
  apply comp_assoc
  /-
    case h
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    X Y Z T : C
    a₁ a₂ a₁₂ : Nat
    α : CategoryTheory.Abelian.Ext X Y a₁
    β : CategoryTheory.Abelian.Ext Y Z a₂
    γ : CategoryTheory.Abelian.Ext Z T 0
    h₁₂ : Eq (HAdd.hAdd a₁ a₂) a₁₂
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd a₁ a₂) 0) a₁₂
  -/
  omega
  /-
    🎉 no goals
  -/


/-- When an instance of `[HasDerivedCategory.{w'} C]` is available, this is the bijection
between `Ext.{w} X Y n` and a type of morphisms in the derived category. -/
noncomputable def homEquiv {n : ℕ} :
    Ext.{w} X Y n ≃ ShiftedHom ((singleFunctor C 0).obj X)
      ((singleFunctor C 0).obj Y) (n : ℤ) :=
  SmallShiftedHom.equiv (HomologicalComplex.quasiIso C (ComplexShape.up ℤ)) Q


/-- The morphism in the derived category which corresponds to an element in `Ext X Y a`. -/
noncomputable abbrev hom {a : ℕ} (α : Ext X Y a) :
    ShiftedHom ((singleFunctor C 0).obj X) ((singleFunctor C 0).obj Y) (a : ℤ) :=
  homEquiv α


@[simp]
lemma comp_hom {a b : ℕ} (α : Ext X Y a) (β : Ext Y Z b) {c : ℕ} (h : a + b = c) :
                                            /-
                                              C : Type u
                                              inst✝³ : CategoryTheory.Category.{v, u} C
                                              inst✝² : CategoryTheory.Abelian C
                                              inst✝¹ : CategoryTheory.HasExt C
                                              X Y Z T : C
                                              inst✝ : HasDerivedCategory C
                                              a b : Nat
                                              α : CategoryTheory.Abelian.Ext X Y a
                                              β : CategoryTheory.Abelian.Ext Y Z b
                                              c : Nat
                                              h : Eq (HAdd.hAdd a b) c
                                              ⊢ Eq (HAdd.hAdd ↑b ↑a) ↑c
                                            -/
    (α.comp β h).hom = α.hom.comp β.hom (by omega) := by
                                            /-
                                              🎉 no goals
                                            -/
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : CategoryTheory.HasExt C
    X Y Z : C
    inst✝ : HasDerivedCategory C
    a b : Nat
    α : CategoryTheory.Abelian.Ext X Y a
    β : CategoryTheory.Abelian.Ext Y Z b
    c : Nat
    h : Eq (HAdd.hAdd a b) c
    ⊢ Eq (α.comp β h).hom (α.hom.comp β.hom ⋯)
  -/
  apply SmallShiftedHom.equiv_comp
  /-
    🎉 no goals
  -/


@[ext]
lemma ext {n : ℕ} {α β : Ext X Y n} (h : α.hom = β.hom) : α = β :=
  homEquiv.injective h


/-- The canonical map `(X ⟶ Y) → Ext X Y 0`. -/
                                                                             /-
                                                                               C : Type u
                                                                               inst✝² : CategoryTheory.Category.{v, u} C
                                                                               inst✝¹ : CategoryTheory.Abelian C
                                                                               inst✝ : CategoryTheory.HasExt C
                                                                               X Y Z T : C
                                                                               f : Quiver.Hom X Y
                                                                               ⊢ Eq (↑0) 0
                                                                             -/
noncomputable def mk₀ (f : X ⟶ Y) : Ext X Y 0 := SmallShiftedHom.mk₀ _ _ (by simp)
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
  ((CochainComplex.singleFunctor C 0).map f)


@[simp]
lemma mk₀_hom [HasDerivedCategory.{w'} C] (f : X ⟶ Y) :
                                       /-
                                         C : Type u
                                         inst✝³ : CategoryTheory.Category.{v, u} C
                                         inst✝² : CategoryTheory.Abelian C
                                         inst✝¹ : CategoryTheory.HasExt C
                                         X Y Z T : C
                                         inst✝ : HasDerivedCategory C
                                         f : Quiver.Hom X Y
                                         ⊢ Eq (↑0) 0
                                       -/
    (mk₀ f).hom = ShiftedHom.mk₀ _ (by simp) ((singleFunctor C 0).map f) := by
                                       /-
                                         🎉 no goals
                                       -/
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : CategoryTheory.HasExt C
    X Y : C
    inst✝ : HasDerivedCategory C
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.Abelian.Ext.mk₀ f).hom (CategoryTheory.ShiftedHom.mk₀ ↑0  …
  -/
  apply SmallShiftedHom.equiv_mk₀
  /-
    🎉 no goals
  -/


@[simp 1100]
lemma mk₀_comp_mk₀ (f : X ⟶ Y) (g : Y ⟶ Z) :
    (mk₀ f).comp (mk₀ g) (zero_add 0) = mk₀ (f ≫ g) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    X Y Z : C
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    ⊢ Eq ((CategoryTheory.Abelian.Ext.mk₀ f).comp (CategoryTheory.Abelian.Ext.mk₀  …
  -/
  letI := HasDerivedCategory.standard C; ext; simp
                                              /-
                                                🎉 no goals
                                              -/


@[simp 1100]
lemma mk₀_comp_mk₀_assoc (f : X ⟶ Y) (g : Y ⟶ Z) {n : ℕ} (α : Ext Z T n) :
    (mk₀ f).comp ((mk₀ g).comp α (zero_add n)) (zero_add n) =
      (mk₀ (f ≫ g)).comp α (zero_add n) := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    X Y Z T : C
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    n : Nat
    α : CategoryTheory.Abelian.Ext Z T n
    ⊢ Eq ((CategoryTheory.Abelian.Ext.mk₀ f).comp ((CategoryTheory.Abelian.Ext.mk₀ …
  -/
  rw [← mk₀_comp_mk₀, comp_assoc]
  /-
    case h
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    X Y Z T : C
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    n : Nat
    α : CategoryTheory.Abelian.Ext Z T n
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd 0 0) n) n
  -/
  omega
  /-
    🎉 no goals
  -/


noncomputable instance : AddCommGroup (Ext X Y n) :=
  letI := HasDerivedCategory.standard C
  homEquiv.addCommGroup


/-- The map from `Ext X Y n` to a `ShiftedHom` type in the *constructed* derived
category given by `HasDerivedCategory.standard`: this definition is introduced
only in order to prove properties of the abelian group structure on `Ext`-groups.
Do not use this definition: use the more general `hom` instead. -/
noncomputable abbrev hom' (α : Ext X Y n) :
  letI := HasDerivedCategory.standard C
  ShiftedHom ((singleFunctor C 0).obj X) ((singleFunctor C 0).obj Y) (n : ℤ) :=
  letI := HasDerivedCategory.standard C
  α.hom


private lemma add_hom' (α β : Ext X Y n) : (α + β).hom' = α.hom' + β.hom' :=
  letI := HasDerivedCategory.standard C
  homEquiv.symm.injective (Equiv.symm_apply_apply _ _)


private lemma neg_hom' (α : Ext X Y n) : (-α).hom' = -α.hom' :=
  letI := HasDerivedCategory.standard C
  homEquiv.symm.injective (Equiv.symm_apply_apply _ _)


variable (X Y n) in
private lemma zero_hom' : (0 : Ext X Y n).hom' = 0 :=
  letI := HasDerivedCategory.standard C
  homEquiv.symm.injective (Equiv.symm_apply_apply _ _)


@[simp]
lemma add_comp (α₁ α₂ : Ext X Y n) {m : ℕ} (β : Ext Y Z m) {p : ℕ} (h : n + m = p) :
    (α₁ + α₂).comp β h = α₁.comp β h + α₂.comp β h := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    X Y Z : C
    n : Nat
    α₁ α₂ : CategoryTheory.Abelian.Ext X Y n
    m : Nat
    β : CategoryTheory.Abelian.Ext Y Z m
    p : Nat
    h : Eq (HAdd.hAdd n m) p
    ⊢ Eq ((HAdd.hAdd α₁ α₂).comp β h) (HAdd.hAdd (α₁.comp β h) (α₂.comp β h))
  -/
  letI := HasDerivedCategory.standard C; ext; simp [this, add_hom']
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
lemma comp_add (α : Ext X Y n) {m : ℕ} (β₁ β₂ : Ext Y Z m) {p : ℕ} (h : n + m = p) :
    α.comp (β₁ + β₂) h = α.comp β₁ h + α.comp β₂ h := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    X Y Z : C
    n : Nat
    α : CategoryTheory.Abelian.Ext X Y n
    m : Nat
    β₁ β₂ : CategoryTheory.Abelian.Ext Y Z m
    p : Nat
    h : Eq (HAdd.hAdd n m) p
    ⊢ Eq (α.comp (HAdd.hAdd β₁ β₂) h) (HAdd.hAdd (α.comp β₁ h) (α.comp β₂ h))
  -/
  letI := HasDerivedCategory.standard C; ext; simp [this, add_hom']
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
lemma neg_comp (α : Ext X Y n) {m : ℕ} (β : Ext Y Z m) {p : ℕ} (h : n + m = p) :
    (-α).comp β h = -α.comp β h := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    X Y Z : C
    n : Nat
    α : CategoryTheory.Abelian.Ext X Y n
    m : Nat
    β : CategoryTheory.Abelian.Ext Y Z m
    p : Nat
    h : Eq (HAdd.hAdd n m) p
    ⊢ Eq ((Neg.neg α).comp β h) (Neg.neg (α.comp β h))
  -/
  letI := HasDerivedCategory.standard C; ext; simp [this, neg_hom']
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
lemma comp_neg (α : Ext X Y n) {m : ℕ} (β : Ext Y Z m) {p : ℕ} (h : n + m = p) :
    α.comp (-β) h = -α.comp β h := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    X Y Z : C
    n : Nat
    α : CategoryTheory.Abelian.Ext X Y n
    m : Nat
    β : CategoryTheory.Abelian.Ext Y Z m
    p : Nat
    h : Eq (HAdd.hAdd n m) p
    ⊢ Eq (α.comp (Neg.neg β) h) (Neg.neg (α.comp β h))
  -/
  letI := HasDerivedCategory.standard C; ext; simp [this, neg_hom']
                                              /-
                                                🎉 no goals
                                              -/


variable (X n) in
@[simp]
lemma zero_comp {m : ℕ} (β : Ext Y Z m) (p : ℕ) (h : n + m = p) :
    (0 : Ext X Y n).comp β h = 0 := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    X Y Z : C
    n m : Nat
    β : CategoryTheory.Abelian.Ext Y Z m
    p : Nat
    h : Eq (HAdd.hAdd n m) p
    ⊢ Eq (CategoryTheory.Abelian.Ext.comp 0 β h) 0
  -/
  letI := HasDerivedCategory.standard C; ext; simp [this, zero_hom']
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
lemma comp_zero (α : Ext X Y n) (Z : C) (m : ℕ) (p : ℕ) (h : n + m = p) :
    α.comp (0 : Ext Y Z m) h = 0 := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    X Y : C
    n : Nat
    α : CategoryTheory.Abelian.Ext X Y n
    Z : C
    m p : Nat
    h : Eq (HAdd.hAdd n m) p
    ⊢ Eq (α.comp 0 h) 0
  -/
  letI := HasDerivedCategory.standard C; ext; simp [this, zero_hom']
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
lemma mk₀_id_comp (α : Ext X Y n) :
    (mk₀ (𝟙 X)).comp α (zero_add n) = α := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    X Y : C
    n : Nat
    α : CategoryTheory.Abelian.Ext X Y n
    ⊢ Eq ((CategoryTheory.Abelian.Ext.mk₀ (CategoryTheory.CategoryStruct.id X)).co …
  -/
  letI := HasDerivedCategory.standard C; ext; simp
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
lemma comp_mk₀_id (α : Ext X Y n) :
    α.comp (mk₀ (𝟙 Y)) (add_zero n) = α := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    X Y : C
    n : Nat
    α : CategoryTheory.Abelian.Ext X Y n
    ⊢ Eq (α.comp (CategoryTheory.Abelian.Ext.mk₀ (CategoryTheory.CategoryStruct.id …
  -/
  letI := HasDerivedCategory.standard C; ext; simp
                                              /-
                                                🎉 no goals
                                              -/


variable (X Y) in
@[simp]
lemma mk₀_zero : mk₀ (0 : X ⟶ Y) = 0 := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    X Y : C
    ⊢ Eq (CategoryTheory.Abelian.Ext.mk₀ 0) 0
  -/
  letI := HasDerivedCategory.standard C; ext; simp [this, zero_hom']
                                              /-
                                                🎉 no goals
                                              -/


attribute [local instance] preservesBinaryBiproducts_of_preservesBiproducts in
lemma biprod_ext {X₁ X₂ : C} {α β : Ext (X₁ ⊞ X₂) Y n}
    (h₁ : (mk₀ biprod.inl).comp α (zero_add n) = (mk₀ biprod.inl).comp β (zero_add n))
    (h₂ : (mk₀ biprod.inr).comp α (zero_add n) = (mk₀ biprod.inr).comp β (zero_add n)) :
    α = β := by
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    Y : C
    n : Nat
    X₁ X₂ : C
    α β : CategoryTheory.Abelian.Ext (CategoryTheory.Limits.biprod X₁ X₂) Y n
    h₁ : Eq ((CategoryTheory.Abelian.Ext.mk₀ CategoryTheory.Limits.biprod.inl).com …
    h₂ : Eq ((CategoryTheory.Abelian.Ext.mk₀ CategoryTheory.Limits.biprod.inr).com …
    ⊢ Eq α β
  -/
  letI := HasDerivedCategory.standard C
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    Y : C
    n : Nat
    X₁ X₂ : C
    α β : CategoryTheory.Abelian.Ext (CategoryTheory.Limits.biprod X₁ X₂) Y n
    h₁ : Eq ((CategoryTheory.Abelian.Ext.mk₀ CategoryTheory.Limits.biprod.inl).com …
    h₂ : Eq ((CategoryTheory.Abelian.Ext.mk₀ CategoryTheory.Limits.biprod.inr).com …
    this : HasDerivedCategory C := HasDerivedCategory.standard C
    ⊢ Eq α β
  -/
  rw [Ext.ext_iff] at h₁ h₂ ⊢
  /-
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    Y : C
    n : Nat
    X₁ X₂ : C
    α β : CategoryTheory.Abelian.Ext (CategoryTheory.Limits.biprod X₁ X₂) Y n
    this : HasDerivedCategory C := HasDerivedCategory.standard C
    h₂ : Eq ((CategoryTheory.Abelian.Ext.mk₀ CategoryTheory.Limits.biprod.inr).com …
    h₁ : Eq ((CategoryTheory.Abelian.Ext.mk₀ CategoryTheory.Limits.biprod.inl).com …
    ⊢ Eq α.hom β.hom
  -/
  simp only [comp_hom, mk₀_hom, ShiftedHom.mk₀_comp] at h₁ h₂
  apply BinaryCofan.IsColimit.hom_ext
    (isBinaryBilimitOfPreserves (singleFunctor C 0)
      (BinaryBiproduct.isBilimit X₁ X₂)).isColimit
  /-
    case h₁
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    inst✝¹ : CategoryTheory.Abelian C
    inst✝ : CategoryTheory.HasExt C
    Y : C
    n : Nat
    X₁ X₂ : C
    α β : CategoryTheory.Abelian.Ext (CategoryTheory.Limits.biprod X₁ X₂) Y n
    this : HasDerivedCategory C := HasDerivedCategory.standard C
    h₁ : Eq (CategoryTheory.CategoryStruct.comp ((DerivedCategory.singleFunctor C  …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp ((DerivedCategory.singleFunctor C  …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.BinaryCofan.in …
  -/
  all_goals assumption
  /-
    🎉 no goals
  -/


variable (X Y n) in
@[simp]
lemma zero_hom : (0 : Ext X Y n).hom = 0 := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : CategoryTheory.HasExt C
    X Y : C
    n : Nat
    inst✝ : HasDerivedCategory C
    ⊢ Eq (CategoryTheory.Abelian.Ext.hom 0) 0
  -/
  let β : Ext 0 Y n := 0
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : CategoryTheory.HasExt C
    X Y : C
    n : Nat
    inst✝ : HasDerivedCategory C
    β : CategoryTheory.Abelian.Ext 0 Y n := 0
    ⊢ Eq (CategoryTheory.Abelian.Ext.hom 0) 0
  -/
  have hβ : β.hom = 0 := by apply (Functor.map_isZero _ (isZero_zero C)).eq_of_src
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : CategoryTheory.HasExt C
    X Y : C
    n : Nat
    inst✝ : HasDerivedCategory C
    β : CategoryTheory.Abelian.Ext 0 Y n := 0
    hβ : Eq β.hom 0
    ⊢ Eq (CategoryTheory.Abelian.Ext.hom 0) 0
  -/
  have : (0 : Ext X Y n) = (0 : Ext X 0 0).comp β (zero_add n) := by simp [β]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : CategoryTheory.HasExt C
    X Y : C
    n : Nat
    inst✝ : HasDerivedCategory C
    β : CategoryTheory.Abelian.Ext 0 Y n := 0
    hβ : Eq β.hom 0
    this : Eq 0 (CategoryTheory.Abelian.Ext.comp 0 β ⋯)
    ⊢ Eq (CategoryTheory.Abelian.Ext.hom 0) 0
  -/
  rw [this, comp_hom, hβ, ShiftedHom.comp_zero]
  /-
    🎉 no goals
  -/


@[simp]
lemma add_hom (α β : Ext X Y n) : (α + β).hom = α.hom + β.hom := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : CategoryTheory.HasExt C
    X Y : C
    n : Nat
    inst✝ : HasDerivedCategory C
    α β : CategoryTheory.Abelian.Ext X Y n
    ⊢ Eq (HAdd.hAdd α β).hom (HAdd.hAdd α.hom β.hom)
  -/
  let α' : Ext (X ⊞ X) Y n := (mk₀ biprod.fst).comp α (zero_add n)
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : CategoryTheory.HasExt C
    X Y : C
    n : Nat
    inst✝ : HasDerivedCategory C
    α β : CategoryTheory.Abelian.Ext X Y n
    α' : CategoryTheory.Abelian.Ext (CategoryTheory.Limits.biprod X X) Y n := (Cat …
    ⊢ Eq (HAdd.hAdd α β).hom (HAdd.hAdd α.hom β.hom)
  -/
  let β' : Ext (X ⊞ X) Y n := (mk₀ biprod.snd).comp β (zero_add n)
  have eq₁ : α + β = (mk₀ (biprod.lift (𝟙 X) (𝟙 X))).comp (α' + β') (zero_add n) := by
    simp [α', β']
  have eq₂ : α' + β' = homEquiv.symm (α'.hom + β'.hom) := by
    apply biprod_ext
    all_goals ext; simp [α', β', ← Functor.map_comp]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : CategoryTheory.HasExt C
    X Y : C
    n : Nat
    inst✝ : HasDerivedCategory C
    α β : CategoryTheory.Abelian.Ext X Y n
    α' : CategoryTheory.Abelian.Ext (CategoryTheory.Limits.biprod X X) Y n := (Cat …
    β' : CategoryTheory.Abelian.Ext (CategoryTheory.Limits.biprod X X) Y n := (Cat …
    eq₁ : Eq (HAdd.hAdd α β) ((CategoryTheory.Abelian.Ext.mk₀ (CategoryTheory.Limi …
    eq₂ : Eq (HAdd.hAdd α' β') (CategoryTheory.Abelian.Ext.homEquiv.symm (HAdd.hAd …
    ⊢ Eq (HAdd.hAdd α β).hom (HAdd.hAdd α.hom β.hom)
  -/
  simp only [eq₁, eq₂, comp_hom, Equiv.apply_symm_apply, ShiftedHom.comp_add]
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : CategoryTheory.HasExt C
    X Y : C
    n : Nat
    inst✝ : HasDerivedCategory C
    α β : CategoryTheory.Abelian.Ext X Y n
    α' : CategoryTheory.Abelian.Ext (CategoryTheory.Limits.biprod X X) Y n := (Cat …
    β' : CategoryTheory.Abelian.Ext (CategoryTheory.Limits.biprod X X) Y n := (Cat …
    eq₁ : Eq (HAdd.hAdd α β) ((CategoryTheory.Abelian.Ext.mk₀ (CategoryTheory.Limi …
    eq₂ : Eq (HAdd.hAdd α' β') (CategoryTheory.Abelian.Ext.homEquiv.symm (HAdd.hAd …
    ⊢ Eq (HAdd.hAdd ((CategoryTheory.Abelian.Ext.mk₀ (CategoryTheory.Limits.biprod …
  -/
  congr
    /-
      case e_a
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Abelian C
      inst✝¹ : CategoryTheory.HasExt C
      X Y : C
      n : Nat
      inst✝ : HasDerivedCategory C
      α β : CategoryTheory.Abelian.Ext X Y n
      α' : CategoryTheory.Abelian.Ext (CategoryTheory.Limits.biprod X X) Y n := (Cat …
      β' : CategoryTheory.Abelian.Ext (CategoryTheory.Limits.biprod X X) Y n := (Cat …
      eq₁ : Eq (HAdd.hAdd α β) ((CategoryTheory.Abelian.Ext.mk₀ (CategoryTheory.Limi …
      eq₂ : Eq (HAdd.hAdd α' β') (CategoryTheory.Abelian.Ext.homEquiv.symm (HAdd.hAd …
      ⊢ Eq ((CategoryTheory.Abelian.Ext.mk₀ (CategoryTheory.Limits.biprod.lift (Cate …
    -/
  · dsimp [α']
    /-
      case e_a
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Abelian C
      inst✝¹ : CategoryTheory.HasExt C
      X Y : C
      n : Nat
      inst✝ : HasDerivedCategory C
      α β : CategoryTheory.Abelian.Ext X Y n
      α' : CategoryTheory.Abelian.Ext (CategoryTheory.Limits.biprod X X) Y n := (Cat …
      β' : CategoryTheory.Abelian.Ext (CategoryTheory.Limits.biprod X X) Y n := (Cat …
      eq₁ : Eq (HAdd.hAdd α β) ((CategoryTheory.Abelian.Ext.mk₀ (CategoryTheory.Limi …
      eq₂ : Eq (HAdd.hAdd α' β') (CategoryTheory.Abelian.Ext.homEquiv.symm (HAdd.hAd …
      ⊢ Eq ((CategoryTheory.Abelian.Ext.mk₀ (CategoryTheory.Limits.biprod.lift (Cate …
    -/
    rw [comp_hom, mk₀_hom, mk₀_hom]
    /-
      case e_a
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Abelian C
      inst✝¹ : CategoryTheory.HasExt C
      X Y : C
      n : Nat
      inst✝ : HasDerivedCategory C
      α β : CategoryTheory.Abelian.Ext X Y n
      α' : CategoryTheory.Abelian.Ext (CategoryTheory.Limits.biprod X X) Y n := (Cat …
      β' : CategoryTheory.Abelian.Ext (CategoryTheory.Limits.biprod X X) Y n := (Cat …
      eq₁ : Eq (HAdd.hAdd α β) ((CategoryTheory.Abelian.Ext.mk₀ (CategoryTheory.Limi …
      eq₂ : Eq (HAdd.hAdd α' β') (CategoryTheory.Abelian.Ext.homEquiv.symm (HAdd.hAd …
      ⊢ Eq ((CategoryTheory.ShiftedHom.mk₀ ↑0 ⋯ ((DerivedCategory.singleFunctor C 0) …
    -/
    dsimp
    rw [ShiftedHom.mk₀_comp_mk₀_assoc, ← Functor.map_comp,
      biprod.lift_fst, Functor.map_id, ShiftedHom.mk₀_id_comp]
    /-
      case e_a
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Abelian C
      inst✝¹ : CategoryTheory.HasExt C
      X Y : C
      n : Nat
      inst✝ : HasDerivedCategory C
      α β : CategoryTheory.Abelian.Ext X Y n
      α' : CategoryTheory.Abelian.Ext (CategoryTheory.Limits.biprod X X) Y n := (Cat …
      β' : CategoryTheory.Abelian.Ext (CategoryTheory.Limits.biprod X X) Y n := (Cat …
      eq₁ : Eq (HAdd.hAdd α β) ((CategoryTheory.Abelian.Ext.mk₀ (CategoryTheory.Limi …
      eq₂ : Eq (HAdd.hAdd α' β') (CategoryTheory.Abelian.Ext.homEquiv.symm (HAdd.hAd …
      ⊢ Eq ((CategoryTheory.Abelian.Ext.mk₀ (CategoryTheory.Limits.biprod.lift (Cate …
    -/
  · dsimp [β']
    /-
      case e_a
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Abelian C
      inst✝¹ : CategoryTheory.HasExt C
      X Y : C
      n : Nat
      inst✝ : HasDerivedCategory C
      α β : CategoryTheory.Abelian.Ext X Y n
      α' : CategoryTheory.Abelian.Ext (CategoryTheory.Limits.biprod X X) Y n := (Cat …
      β' : CategoryTheory.Abelian.Ext (CategoryTheory.Limits.biprod X X) Y n := (Cat …
      eq₁ : Eq (HAdd.hAdd α β) ((CategoryTheory.Abelian.Ext.mk₀ (CategoryTheory.Limi …
      eq₂ : Eq (HAdd.hAdd α' β') (CategoryTheory.Abelian.Ext.homEquiv.symm (HAdd.hAd …
      ⊢ Eq ((CategoryTheory.Abelian.Ext.mk₀ (CategoryTheory.Limits.biprod.lift (Cate …
    -/
    rw [comp_hom, mk₀_hom, mk₀_hom]
    /-
      case e_a
      C : Type u
      inst✝³ : CategoryTheory.Category.{v, u} C
      inst✝² : CategoryTheory.Abelian C
      inst✝¹ : CategoryTheory.HasExt C
      X Y : C
      n : Nat
      inst✝ : HasDerivedCategory C
      α β : CategoryTheory.Abelian.Ext X Y n
      α' : CategoryTheory.Abelian.Ext (CategoryTheory.Limits.biprod X X) Y n := (Cat …
      β' : CategoryTheory.Abelian.Ext (CategoryTheory.Limits.biprod X X) Y n := (Cat …
      eq₁ : Eq (HAdd.hAdd α β) ((CategoryTheory.Abelian.Ext.mk₀ (CategoryTheory.Limi …
      eq₂ : Eq (HAdd.hAdd α' β') (CategoryTheory.Abelian.Ext.homEquiv.symm (HAdd.hAd …
      ⊢ Eq ((CategoryTheory.ShiftedHom.mk₀ ↑0 ⋯ ((DerivedCategory.singleFunctor C 0) …
    -/
    dsimp
    rw [ShiftedHom.mk₀_comp_mk₀_assoc, ← Functor.map_comp,
      biprod.lift_snd, Functor.map_id, ShiftedHom.mk₀_id_comp]


lemma neg_hom (α : Ext X Y n) : (-α).hom = -α.hom := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    inst✝² : CategoryTheory.Abelian C
    inst✝¹ : CategoryTheory.HasExt C
    X Y : C
    n : Nat
    inst✝ : HasDerivedCategory C
    α : CategoryTheory.Abelian.Ext X Y n
    ⊢ Eq (Neg.neg α).hom (Neg.neg α.hom)
  -/
  rw [← add_right_inj α.hom, ← add_hom, add_neg_cancel, add_neg_cancel, zero_hom]
  /-
    🎉 no goals
  -/


/-- When an instance of `[HasDerivedCategory.{w'} C]` is available, this is the additive
bijection between `Ext.{w} X Y n` and a type of morphisms in the derived category. -/
noncomputable def homAddEquiv {n : ℕ} :
    Ext.{w} X Y n ≃+
      ShiftedHom ((singleFunctor C 0).obj X) ((singleFunctor C 0).obj Y) (n : ℤ) where
  toEquiv := homEquiv
                 /-
                   C : Type u
                   inst✝³ : CategoryTheory.Category.{v, u} C
                   inst✝² : CategoryTheory.Abelian C
                   inst✝¹ : CategoryTheory.HasExt C
                   X Y Z T : C
                   n✝ : Nat
                   inst✝ : HasDerivedCategory C
                   n : Nat
                   ⊢ ∀ (x y : CategoryTheory.Abelian.Ext X Y n), Eq (CategoryTheory.Abelian.Ext.h …
                 -/
  map_add' := by simp
                 /-
                   🎉 no goals
                 -/


@[simp]
lemma homAddEquiv_apply (α : Ext X Y n) : homAddEquiv α = α.hom := rfl


variable (X Y Z) in
/-- The composition of `Ext`, as a bilinear map. -/
@[simps!]
noncomputable def bilinearComp (a b c : ℕ) (h : a + b = c) :
    Ext X Y a →+ Ext Y Z b →+ Ext X Z c :=
                                                                      /-
                                                                        C : Type u
                                                                        inst✝² : CategoryTheory.Category.{v, u} C
                                                                        inst✝¹ : CategoryTheory.Abelian C
                                                                        inst✝ : CategoryTheory.HasExt C
                                                                        X Y Z T : C
                                                                        n a b c : Nat
                                                                        h : Eq (HAdd.hAdd a b) c
                                                                        α : CategoryTheory.Abelian.Ext X Y a
                                                                        ⊢ ∀ (a_1 b_1 : CategoryTheory.Abelian.Ext Y Z b), Eq ((fun β => α.comp β h) (H …
                                                                      -/
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
  AddMonoidHom.mk' (fun α ↦ AddMonoidHom.mk' (fun β ↦ α.comp β h) (by simp)) (by aesop)
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


/-- The postcomposition `Ext X Y a →+ Ext X Z b` with `β : Ext Y Z n` when `a + n = b`. -/
noncomputable abbrev postcomp (β : Ext Y Z n) (X : C) {a b : ℕ} (h : a + n = b) :
    Ext X Y a →+ Ext X Z b :=
  (bilinearComp X Y Z a n b h).flip β


/-- The precomposition `Ext Y Z a →+ Ext X Z b` with `α : Ext X Y n` when `n + a = b`. -/
noncomputable abbrev precomp (α : Ext X Y n) (Z : C) {a b : ℕ} (h : n + a = b) :
    Ext Y Z a →+ Ext X Z b :=
  bilinearComp X Y Z n a b h α


/-- Auxiliary definition for `extFunctor`. -/
@[simps]
noncomputable def extFunctorObj (X : C) (n : ℕ) : C ⥤ AddCommGrp.{w} where
  obj Y := AddCommGrp.of (Ext X Y n)
  map f := AddCommGrp.ofHom ((Ext.mk₀ f).postcomp _ (add_zero n))
  map_comp f f' := by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : CategoryTheory.HasExt C
      X : C
      n : Nat
      X✝ Y✝ Z✝ : C
      f : Quiver.Hom X✝ Y✝
      f' : Quiver.Hom Y✝ Z✝
      ⊢ Eq ({ obj := fun Y => AddCommGrp.of (CategoryTheory.Abelian.Ext X Y n), map  …
    -/
    ext α
    /-
      case w
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : CategoryTheory.HasExt C
      X : C
      n : Nat
      X✝ Y✝ Z✝ : C
      f : Quiver.Hom X✝ Y✝
      f' : Quiver.Hom Y✝ Z✝
      α : ↑({ obj := fun Y => AddCommGrp.of (CategoryTheory.Abelian.Ext X Y n), map  …
      ⊢ Eq (({ obj := fun Y => AddCommGrp.of (CategoryTheory.Abelian.Ext X Y n), map …
    -/
    dsimp [AddCommGrp.ofHom]
    /-
      case w
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : CategoryTheory.HasExt C
      X : C
      n : Nat
      X✝ Y✝ Z✝ : C
      f : Quiver.Hom X✝ Y✝
      f' : Quiver.Hom Y✝ Z✝
      α : ↑({ obj := fun Y => AddCommGrp.of (CategoryTheory.Abelian.Ext X Y n), map  …
      ⊢ Eq (CategoryTheory.Abelian.Ext.comp α (CategoryTheory.Abelian.Ext.mk₀ (Categ …
    -/
    rw [← Ext.mk₀_comp_mk₀]
    /-
      case w
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : CategoryTheory.HasExt C
      X : C
      n : Nat
      X✝ Y✝ Z✝ : C
      f : Quiver.Hom X✝ Y✝
      f' : Quiver.Hom Y✝ Z✝
      α : ↑({ obj := fun Y => AddCommGrp.of (CategoryTheory.Abelian.Ext X Y n), map  …
      ⊢ Eq (CategoryTheory.Abelian.Ext.comp α ((CategoryTheory.Abelian.Ext.mk₀ f).co …
    -/
    symm
    /-
      case w
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : CategoryTheory.HasExt C
      X : C
      n : Nat
      X✝ Y✝ Z✝ : C
      f : Quiver.Hom X✝ Y✝
      f' : Quiver.Hom Y✝ Z✝
      α : ↑({ obj := fun Y => AddCommGrp.of (CategoryTheory.Abelian.Ext X Y n), map  …
      ⊢ Eq ((CategoryTheory.Abelian.Ext.comp α (CategoryTheory.Abelian.Ext.mk₀ f) ⋯) …
    -/
    apply Ext.comp_assoc
    /-
      case w.h
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : CategoryTheory.HasExt C
      X : C
      n : Nat
      X✝ Y✝ Z✝ : C
      f : Quiver.Hom X✝ Y✝
      f' : Quiver.Hom Y✝ Z✝
      α : ↑({ obj := fun Y => AddCommGrp.of (CategoryTheory.Abelian.Ext X Y n), map  …
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd n 0) 0) n
    -/
    omega
    /-
      🎉 no goals
    -/


/-- The functor `Cᵒᵖ ⥤ C ⥤ AddCommGrp` which sends `X : C` and `Y : C`
to `Ext X Y n`. -/
@[simps]
noncomputable def extFunctor (n : ℕ) : Cᵒᵖ ⥤ C ⥤ AddCommGrp.{w} where
  obj X := extFunctorObj X.unop n
  map {X₁ X₂} f :=
    { app := fun Y ↦ AddCommGrp.ofHom (AddMonoidHom.mk'
                                                           /-
                                                             C : Type u
                                                             inst✝² : CategoryTheory.Category.{v, u} C
                                                             inst✝¹ : CategoryTheory.Abelian C
                                                             inst✝ : CategoryTheory.HasExt C
                                                             n : Nat
                                                             X₁ X₂ : Opposite C
                                                             f : Quiver.Hom X₁ X₂
                                                             Y : C
                                                             ⊢ ∀ (a b : CategoryTheory.Abelian.Ext (Opposite.unop X₁) Y n), Eq ((fun α => ( …
                                                           -/
        (fun α ↦ (Ext.mk₀ f.unop).comp α (zero_add _)) (by simp))
                                                           /-
                                                             🎉 no goals
                                                           -/
      naturality := fun {Y₁ Y₂} g ↦ by
        /-
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Abelian C
          inst✝ : CategoryTheory.HasExt C
          n : Nat
          X₁ X₂ : Opposite C
          f : Quiver.Hom X₁ X₂
          Y₁ Y₂ : C
          g : Quiver.Hom Y₁ Y₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((fun X => CategoryTheory.Abelian.ex …
        -/
        ext α
        /-
          case w
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Abelian C
          inst✝ : CategoryTheory.HasExt C
          n : Nat
          X₁ X₂ : Opposite C
          f : Quiver.Hom X₁ X₂
          Y₁ Y₂ : C
          g : Quiver.Hom Y₁ Y₂
          α : ↑(((fun X => CategoryTheory.Abelian.extFunctorObj (Opposite.unop X) n) X₁) …
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp (((fun X => CategoryTheory.Abelian.e …
        -/
        dsimp
        /-
          case w
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Abelian C
          inst✝ : CategoryTheory.HasExt C
          n : Nat
          X₁ X₂ : Opposite C
          f : Quiver.Hom X₁ X₂
          Y₁ Y₂ : C
          g : Quiver.Hom Y₁ Y₂
          α : ↑(((fun X => CategoryTheory.Abelian.extFunctorObj (Opposite.unop X) n) X₁) …
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp (AddCommGrp.ofHom ((CategoryTheory.A …
        -/
        symm
        /-
          case w
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Abelian C
          inst✝ : CategoryTheory.HasExt C
          n : Nat
          X₁ X₂ : Opposite C
          f : Quiver.Hom X₁ X₂
          Y₁ Y₂ : C
          g : Quiver.Hom Y₁ Y₂
          α : ↑(((fun X => CategoryTheory.Abelian.extFunctorObj (Opposite.unop X) n) X₁) …
          ⊢ Eq ((CategoryTheory.CategoryStruct.comp (AddCommGrp.ofHom (AddMonoidHom.mk'  …
        -/
        apply Ext.comp_assoc
        /-
          case w.h₁₂
          C : Type u
          inst✝² : CategoryTheory.Category.{v, u} C
          inst✝¹ : CategoryTheory.Abelian C
          inst✝ : CategoryTheory.HasExt C
          n : Nat
          X₁ X₂ : Opposite C
          f : Quiver.Hom X₁ X₂
          Y₁ Y₂ : C
          g : Quiver.Hom Y₁ Y₂
          α : ↑(((fun X => CategoryTheory.Abelian.extFunctorObj (Opposite.unop X) n) X₁) …
          ⊢ Eq (HAdd.hAdd 0 n) n
        -/
        all_goals omega }
        /-
          🎉 no goals
        -/
  map_comp {X₁ X₂ X₃} f f'  := by
    /-
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : CategoryTheory.HasExt C
      n : Nat
      X₁ X₂ X₃ : Opposite C
      f : Quiver.Hom X₁ X₂
      f' : Quiver.Hom X₂ X₃
      ⊢ Eq ({ obj := fun X => CategoryTheory.Abelian.extFunctorObj (Opposite.unop X) …
    -/
    ext Y α
    /-
      case w.h.w
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : CategoryTheory.HasExt C
      n : Nat
      X₁ X₂ X₃ : Opposite C
      f : Quiver.Hom X₁ X₂
      f' : Quiver.Hom X₂ X₃
      Y : C
      α : ↑(({ obj := fun X => CategoryTheory.Abelian.extFunctorObj (Opposite.unop X …
      ⊢ Eq ((({ obj := fun X => CategoryTheory.Abelian.extFunctorObj (Opposite.unop  …
    -/
    dsimp
    /-
      case w.h.w
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : CategoryTheory.HasExt C
      n : Nat
      X₁ X₂ X₃ : Opposite C
      f : Quiver.Hom X₁ X₂
      f' : Quiver.Hom X₂ X₃
      Y : C
      α : ↑(({ obj := fun X => CategoryTheory.Abelian.extFunctorObj (Opposite.unop X …
      ⊢ Eq ((CategoryTheory.Abelian.Ext.mk₀ (CategoryTheory.CategoryStruct.comp f'.u …
    -/
    rw [← Ext.mk₀_comp_mk₀]
    /-
      case w.h.w
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : CategoryTheory.HasExt C
      n : Nat
      X₁ X₂ X₃ : Opposite C
      f : Quiver.Hom X₁ X₂
      f' : Quiver.Hom X₂ X₃
      Y : C
      α : ↑(({ obj := fun X => CategoryTheory.Abelian.extFunctorObj (Opposite.unop X …
      ⊢ Eq (((CategoryTheory.Abelian.Ext.mk₀ f'.unop).comp (CategoryTheory.Abelian.E …
    -/
    apply Ext.comp_assoc
    /-
      case w.h.w.h₂₃
      C : Type u
      inst✝² : CategoryTheory.Category.{v, u} C
      inst✝¹ : CategoryTheory.Abelian C
      inst✝ : CategoryTheory.HasExt C
      n : Nat
      X₁ X₂ X₃ : Opposite C
      f : Quiver.Hom X₁ X₂
      f' : Quiver.Hom X₂ X₃
      Y : C
      α : ↑(({ obj := fun X => CategoryTheory.Abelian.extFunctorObj (Opposite.unop X …
      ⊢ Eq (HAdd.hAdd 0 n) n
    -/
    all_goals omega
    /-
      🎉 no goals
    -/


/-- Up to an equivalence, the type `Ext.{w} X Y n` does not depend on the universe `w`. -/
noncomputable def chgUniv : Ext.{w} X Y n ≃ Ext.{w'} X Y n :=
  SmallShiftedHom.chgUniv.{w', w}


lemma homEquiv_chgUniv [HasDerivedCategory.{w''} C] (e : Ext.{w} X Y n) :
    homEquiv.{w'', w'} (chgUniv.{w'} e) = homEquiv.{w'', w} e := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Abelian C
    inst✝² : CategoryTheory.HasExt C
    inst✝¹ : CategoryTheory.HasExt C
    X Y : C
    n : Nat
    inst✝ : HasDerivedCategory C
    e : CategoryTheory.Abelian.Ext X Y n
    ⊢ Eq (CategoryTheory.Abelian.Ext.homEquiv (CategoryTheory.Abelian.Ext.chgUniv  …
  -/
  apply SmallShiftedHom.equiv_chgUniv
  /-
    🎉 no goals
  -/


