instance : Add (S₁ ⟶ S₂) where
  add φ φ' :=
    { τ₁ := φ.τ₁ + φ'.τ₁
      τ₂ := φ.τ₂ + φ'.τ₂
      τ₃ := φ.τ₃ + φ'.τ₃ }


instance : Sub (S₁ ⟶ S₂) where
  sub φ φ' :=
    { τ₁ := φ.τ₁ - φ'.τ₁
      τ₂ := φ.τ₂ - φ'.τ₂
      τ₃ := φ.τ₃ - φ'.τ₃ }


instance : Neg (S₁ ⟶ S₂) where
  neg φ :=
    { τ₁ := -φ.τ₁
      τ₂ := -φ.τ₂
      τ₃ := -φ.τ₃ }


instance : AddCommGroup (S₁ ⟶ S₂) where
                               /-
                                 C : Type u_1
                                 inst✝¹ : CategoryTheory.Category.{?u.4884, u_1} C
                                 inst✝ : CategoryTheory.Preadditive C
                                 S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                                 a b c : Quiver.Hom S₁ S₂
                                 ⊢ Eq (HAdd.hAdd (HAdd.hAdd a b) c) (HAdd.hAdd a (HAdd.hAdd b c))
                               -/
                                       /-
                                         🎉 no goals
                                       -/
                                       /-
                                         🎉 no goals
                                       -/
  add_assoc := fun a b c => by ext <;> apply add_assoc
                                       /-
                                         🎉 no goals
                                       -/
                          /-
                            C : Type u_1
                            inst✝¹ : CategoryTheory.Category.{?u.4884, u_1} C
                            inst✝ : CategoryTheory.Preadditive C
                            S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                            a : Quiver.Hom S₁ S₂
                            ⊢ Eq (HAdd.hAdd a 0) a
                          -/
                          /-
                            C : Type u_1
                            inst✝¹ : CategoryTheory.Category.{?u.4884, u_1} C
                            inst✝ : CategoryTheory.Preadditive C
                            S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                            a : Quiver.Hom S₁ S₂
                            ⊢ Eq (HAdd.hAdd 0 a) a
                          -/
                                  /-
                                    🎉 no goals
                                  -/
                                  /-
                                    🎉 no goals
                                  -/
                                  /-
                                    🎉 no goals
                                  -/
                                  /-
                                    🎉 no goals
                                  -/
                                  /-
                                    🎉 no goals
                                  -/
  add_zero := fun a => by ext <;> apply add_zero
                                  /-
                                    🎉 no goals
                                  -/
  zero_add := fun a => by ext <;> apply zero_add
                                /-
                                  C : Type u_1
                                  inst✝¹ : CategoryTheory.Category.{?u.4884, u_1} C
                                  inst✝ : CategoryTheory.Preadditive C
                                  S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                                  a : Quiver.Hom S₁ S₂
                                  ⊢ Eq (HAdd.hAdd (Neg.neg a) a) 0
                                -/
                                        /-
                                          🎉 no goals
                                        -/
                                  /-
                                    C : Type u_1
                                    inst✝¹ : CategoryTheory.Category.{?u.4884, u_1} C
                                    inst✝ : CategoryTheory.Preadditive C
                                    S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                                    a b : Quiver.Hom S₁ S₂
                                    ⊢ Eq (HSub.hSub a b) (HAdd.hAdd a (Neg.neg b))
                                  -/
                                          /-
                                            🎉 no goals
                                          -/
                                          /-
                                            🎉 no goals
                                          -/
                                        /-
                                          🎉 no goals
                                        -/
                                          /-
                                            🎉 no goals
                                          -/
  neg_add_cancel := fun a => by ext <;> apply neg_add_cancel
                                        /-
                                          🎉 no goals
                                        -/
                            /-
                              C : Type u_1
                              inst✝¹ : CategoryTheory.Category.{?u.4884, u_1} C
                              inst✝ : CategoryTheory.Preadditive C
                              S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                              a b : Quiver.Hom S₁ S₂
                              ⊢ Eq (HAdd.hAdd a b) (HAdd.hAdd b a)
                            -/
                                    /-
                                      🎉 no goals
                                    -/
                                    /-
                                      🎉 no goals
                                    -/
  add_comm := fun a b => by ext <;> apply add_comm
                                    /-
                                      🎉 no goals
                                    -/
  sub_eq_add_neg := fun a b => by ext <;> apply sub_eq_add_neg
  nsmul := nsmulRec
  zsmul := zsmulRec


@[simp] lemma add_τ₁ (φ φ' : S₁ ⟶ S₂) : (φ + φ').τ₁ = φ.τ₁ + φ'.τ₁ := rfl

@[simp] lemma add_τ₂ (φ φ' : S₁ ⟶ S₂) : (φ + φ').τ₂ = φ.τ₂ + φ'.τ₂ := rfl

@[simp] lemma add_τ₃ (φ φ' : S₁ ⟶ S₂) : (φ + φ').τ₃ = φ.τ₃ + φ'.τ₃ := rfl

@[simp] lemma sub_τ₁ (φ φ' : S₁ ⟶ S₂) : (φ - φ').τ₁ = φ.τ₁ - φ'.τ₁ := rfl

@[simp] lemma sub_τ₂ (φ φ' : S₁ ⟶ S₂) : (φ - φ').τ₂ = φ.τ₂ - φ'.τ₂ := rfl

@[simp] lemma sub_τ₃ (φ φ' : S₁ ⟶ S₂) : (φ - φ').τ₃ = φ.τ₃ - φ'.τ₃ := rfl

@[simp] lemma neg_τ₁ (φ : S₁ ⟶ S₂) : (-φ).τ₁ = -φ.τ₁ := rfl

@[simp] lemma neg_τ₂ (φ : S₁ ⟶ S₂) : (-φ).τ₂ = -φ.τ₂ := rfl

@[simp] lemma neg_τ₃ (φ : S₁ ⟶ S₂) : (-φ).τ₃ = -φ.τ₃ := rfl


instance : Preadditive (ShortComplex C) where


/-- Given a left homology map data for morphism `φ`, this is the induced left homology
map data for `-φ`. -/
@[simps]
def neg : LeftHomologyMapData (-φ) h₁ h₂ where
  φK := -γ.φK
  φH := -γ.φH


/-- Given left homology map data for morphisms `φ` and `φ'`, this is
the induced left homology map data for `φ + φ'`. -/
@[simps]
def add : LeftHomologyMapData (φ + φ') h₁ h₂ where
  φK := γ.φK + γ'.φK
  φH := γ.φH + γ'.φH


@[simp]
lemma leftHomologyMap'_neg :
    leftHomologyMap' (-φ) h₁ h₂ = -leftHomologyMap' φ h₁ h₂ := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.LeftHomologyData
    h₂ : S₂.LeftHomologyData
    ⊢ Eq (CategoryTheory.ShortComplex.leftHomologyMap' (Neg.neg φ) h₁ h₂) (Neg.neg …
  -/
  have γ : LeftHomologyMapData φ h₁ h₂ := default
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.LeftHomologyData
    h₂ : S₂.LeftHomologyData
    γ : CategoryTheory.ShortComplex.LeftHomologyMapData φ h₁ h₂
    ⊢ Eq (CategoryTheory.ShortComplex.leftHomologyMap' (Neg.neg φ) h₁ h₂) (Neg.neg …
  -/
  simp only [γ.leftHomologyMap'_eq, γ.neg.leftHomologyMap'_eq, LeftHomologyMapData.neg_φH]
  /-
    🎉 no goals
  -/


@[simp]
lemma cyclesMap'_neg :
    cyclesMap' (-φ) h₁ h₂ = -cyclesMap' φ h₁ h₂ := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.LeftHomologyData
    h₂ : S₂.LeftHomologyData
    ⊢ Eq (CategoryTheory.ShortComplex.cyclesMap' (Neg.neg φ) h₁ h₂) (Neg.neg (Cate …
  -/
  have γ : LeftHomologyMapData φ h₁ h₂ := default
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.LeftHomologyData
    h₂ : S₂.LeftHomologyData
    γ : CategoryTheory.ShortComplex.LeftHomologyMapData φ h₁ h₂
    ⊢ Eq (CategoryTheory.ShortComplex.cyclesMap' (Neg.neg φ) h₁ h₂) (Neg.neg (Cate …
  -/
  simp only [γ.cyclesMap'_eq, γ.neg.cyclesMap'_eq, LeftHomologyMapData.neg_φK]
  /-
    🎉 no goals
  -/


@[simp]
lemma leftHomologyMap'_add :
    leftHomologyMap' (φ + φ') h₁ h₂ = leftHomologyMap' φ h₁ h₂ +
      leftHomologyMap' φ' h₁ h₂ := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ φ' : Quiver.Hom S₁ S₂
    h₁ : S₁.LeftHomologyData
    h₂ : S₂.LeftHomologyData
    ⊢ Eq (CategoryTheory.ShortComplex.leftHomologyMap' (HAdd.hAdd φ φ') h₁ h₂) (HA …
  -/
  have γ : LeftHomologyMapData φ h₁ h₂ := default
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ φ' : Quiver.Hom S₁ S₂
    h₁ : S₁.LeftHomologyData
    h₂ : S₂.LeftHomologyData
    γ : CategoryTheory.ShortComplex.LeftHomologyMapData φ h₁ h₂
    ⊢ Eq (CategoryTheory.ShortComplex.leftHomologyMap' (HAdd.hAdd φ φ') h₁ h₂) (HA …
  -/
  have γ' : LeftHomologyMapData φ' h₁ h₂ := default
  simp only [γ.leftHomologyMap'_eq, γ'.leftHomologyMap'_eq,
    (γ.add γ').leftHomologyMap'_eq, LeftHomologyMapData.add_φH]


@[simp]
lemma cyclesMap'_add :
    cyclesMap' (φ + φ') h₁ h₂ = cyclesMap' φ h₁ h₂ +
      cyclesMap' φ' h₁ h₂ := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ φ' : Quiver.Hom S₁ S₂
    h₁ : S₁.LeftHomologyData
    h₂ : S₂.LeftHomologyData
    ⊢ Eq (CategoryTheory.ShortComplex.cyclesMap' (HAdd.hAdd φ φ') h₁ h₂) (HAdd.hAd …
  -/
  have γ : LeftHomologyMapData φ h₁ h₂ := default
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ φ' : Quiver.Hom S₁ S₂
    h₁ : S₁.LeftHomologyData
    h₂ : S₂.LeftHomologyData
    γ : CategoryTheory.ShortComplex.LeftHomologyMapData φ h₁ h₂
    ⊢ Eq (CategoryTheory.ShortComplex.cyclesMap' (HAdd.hAdd φ φ') h₁ h₂) (HAdd.hAd …
  -/
  have γ' : LeftHomologyMapData φ' h₁ h₂ := default
  simp only [γ.cyclesMap'_eq, γ'.cyclesMap'_eq,
    (γ.add γ').cyclesMap'_eq, LeftHomologyMapData.add_φK]


@[simp]
lemma leftHomologyMap'_sub :
    leftHomologyMap' (φ - φ') h₁ h₂ = leftHomologyMap' φ h₁ h₂ -
      leftHomologyMap' φ' h₁ h₂ := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ φ' : Quiver.Hom S₁ S₂
    h₁ : S₁.LeftHomologyData
    h₂ : S₂.LeftHomologyData
    ⊢ Eq (CategoryTheory.ShortComplex.leftHomologyMap' (HSub.hSub φ φ') h₁ h₂) (HS …
  -/
  simp only [sub_eq_add_neg, leftHomologyMap'_add, leftHomologyMap'_neg]
  /-
    🎉 no goals
  -/


@[simp]
lemma cyclesMap'_sub :
    cyclesMap' (φ - φ') h₁ h₂ = cyclesMap' φ h₁ h₂ -
      cyclesMap' φ' h₁ h₂ := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ φ' : Quiver.Hom S₁ S₂
    h₁ : S₁.LeftHomologyData
    h₂ : S₂.LeftHomologyData
    ⊢ Eq (CategoryTheory.ShortComplex.cyclesMap' (HSub.hSub φ φ') h₁ h₂) (HSub.hSu …
  -/
  simp only [sub_eq_add_neg, cyclesMap'_add, cyclesMap'_neg]
  /-
    🎉 no goals
  -/


@[simp]
lemma leftHomologyMap_neg : leftHomologyMap (-φ) = -leftHomologyMap φ :=
  leftHomologyMap'_neg _ _


@[simp]
lemma cyclesMap_neg : cyclesMap (-φ) = -cyclesMap φ :=
  cyclesMap'_neg _ _


@[simp]
lemma leftHomologyMap_add : leftHomologyMap (φ + φ') = leftHomologyMap φ + leftHomologyMap φ' :=
  leftHomologyMap'_add _ _


@[simp]
lemma cyclesMap_add : cyclesMap (φ + φ') = cyclesMap φ + cyclesMap φ' :=
  cyclesMap'_add _ _


@[simp]
lemma leftHomologyMap_sub : leftHomologyMap (φ - φ') = leftHomologyMap φ - leftHomologyMap φ' :=
  leftHomologyMap'_sub _ _


@[simp]
lemma cyclesMap_sub : cyclesMap (φ - φ') = cyclesMap φ - cyclesMap φ' :=
  cyclesMap'_sub _ _


instance leftHomologyFunctor_additive [HasKernels C] [HasCokernels C] :
  (leftHomologyFunctor C).Additive where


instance cyclesFunctor_additive [HasKernels C] [HasCokernels C] :
  (cyclesFunctor C).Additive where


/-- Given a right homology map data for morphism `φ`, this is the induced right homology
map data for `-φ`. -/
@[simps]
def neg : RightHomologyMapData (-φ) h₁ h₂ where
  φQ := -γ.φQ
  φH := -γ.φH


/-- Given right homology map data for morphisms `φ` and `φ'`, this is the induced
right homology map data for `φ + φ'`. -/
@[simps]
def add : RightHomologyMapData (φ + φ') h₁ h₂ where
  φQ := γ.φQ + γ'.φQ
  φH := γ.φH + γ'.φH


@[simp]
lemma rightHomologyMap'_neg :
    rightHomologyMap' (-φ) h₁ h₂ = -rightHomologyMap' φ h₁ h₂ := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.RightHomologyData
    h₂ : S₂.RightHomologyData
    ⊢ Eq (CategoryTheory.ShortComplex.rightHomologyMap' (Neg.neg φ) h₁ h₂) (Neg.ne …
  -/
  have γ : RightHomologyMapData φ h₁ h₂ := default
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.RightHomologyData
    h₂ : S₂.RightHomologyData
    γ : CategoryTheory.ShortComplex.RightHomologyMapData φ h₁ h₂
    ⊢ Eq (CategoryTheory.ShortComplex.rightHomologyMap' (Neg.neg φ) h₁ h₂) (Neg.ne …
  -/
  simp only [γ.rightHomologyMap'_eq, γ.neg.rightHomologyMap'_eq, RightHomologyMapData.neg_φH]
  /-
    🎉 no goals
  -/


@[simp]
lemma opcyclesMap'_neg :
    opcyclesMap' (-φ) h₁ h₂ = -opcyclesMap' φ h₁ h₂ := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.RightHomologyData
    h₂ : S₂.RightHomologyData
    ⊢ Eq (CategoryTheory.ShortComplex.opcyclesMap' (Neg.neg φ) h₁ h₂) (Neg.neg (Ca …
  -/
  have γ : RightHomologyMapData φ h₁ h₂ := default
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ : Quiver.Hom S₁ S₂
    h₁ : S₁.RightHomologyData
    h₂ : S₂.RightHomologyData
    γ : CategoryTheory.ShortComplex.RightHomologyMapData φ h₁ h₂
    ⊢ Eq (CategoryTheory.ShortComplex.opcyclesMap' (Neg.neg φ) h₁ h₂) (Neg.neg (Ca …
  -/
  simp only [γ.opcyclesMap'_eq, γ.neg.opcyclesMap'_eq, RightHomologyMapData.neg_φQ]
  /-
    🎉 no goals
  -/


@[simp]
lemma rightHomologyMap'_add :
    rightHomologyMap' (φ + φ') h₁ h₂ = rightHomologyMap' φ h₁ h₂ +
      rightHomologyMap' φ' h₁ h₂ := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ φ' : Quiver.Hom S₁ S₂
    h₁ : S₁.RightHomologyData
    h₂ : S₂.RightHomologyData
    ⊢ Eq (CategoryTheory.ShortComplex.rightHomologyMap' (HAdd.hAdd φ φ') h₁ h₂) (H …
  -/
  have γ : RightHomologyMapData φ h₁ h₂ := default
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ φ' : Quiver.Hom S₁ S₂
    h₁ : S₁.RightHomologyData
    h₂ : S₂.RightHomologyData
    γ : CategoryTheory.ShortComplex.RightHomologyMapData φ h₁ h₂
    ⊢ Eq (CategoryTheory.ShortComplex.rightHomologyMap' (HAdd.hAdd φ φ') h₁ h₂) (H …
  -/
  have γ' : RightHomologyMapData φ' h₁ h₂ := default
  simp only [γ.rightHomologyMap'_eq, γ'.rightHomologyMap'_eq,
    (γ.add γ').rightHomologyMap'_eq, RightHomologyMapData.add_φH]


@[simp]
lemma opcyclesMap'_add :
    opcyclesMap' (φ + φ') h₁ h₂ = opcyclesMap' φ h₁ h₂ +
      opcyclesMap' φ' h₁ h₂ := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ φ' : Quiver.Hom S₁ S₂
    h₁ : S₁.RightHomologyData
    h₂ : S₂.RightHomologyData
    ⊢ Eq (CategoryTheory.ShortComplex.opcyclesMap' (HAdd.hAdd φ φ') h₁ h₂) (HAdd.h …
  -/
  have γ : RightHomologyMapData φ h₁ h₂ := default
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ φ' : Quiver.Hom S₁ S₂
    h₁ : S₁.RightHomologyData
    h₂ : S₂.RightHomologyData
    γ : CategoryTheory.ShortComplex.RightHomologyMapData φ h₁ h₂
    ⊢ Eq (CategoryTheory.ShortComplex.opcyclesMap' (HAdd.hAdd φ φ') h₁ h₂) (HAdd.h …
  -/
  have γ' : RightHomologyMapData φ' h₁ h₂ := default
  simp only [γ.opcyclesMap'_eq, γ'.opcyclesMap'_eq,
    (γ.add γ').opcyclesMap'_eq, RightHomologyMapData.add_φQ]


@[simp]
lemma rightHomologyMap'_sub :
    rightHomologyMap' (φ - φ') h₁ h₂ = rightHomologyMap' φ h₁ h₂ -
      rightHomologyMap' φ' h₁ h₂ := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ φ' : Quiver.Hom S₁ S₂
    h₁ : S₁.RightHomologyData
    h₂ : S₂.RightHomologyData
    ⊢ Eq (CategoryTheory.ShortComplex.rightHomologyMap' (HSub.hSub φ φ') h₁ h₂) (H …
  -/
  simp only [sub_eq_add_neg, rightHomologyMap'_add, rightHomologyMap'_neg]
  /-
    🎉 no goals
  -/


@[simp]
lemma opcyclesMap'_sub :
    opcyclesMap' (φ - φ') h₁ h₂ = opcyclesMap' φ h₁ h₂ -
      opcyclesMap' φ' h₁ h₂ := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ φ' : Quiver.Hom S₁ S₂
    h₁ : S₁.RightHomologyData
    h₂ : S₂.RightHomologyData
    ⊢ Eq (CategoryTheory.ShortComplex.opcyclesMap' (HSub.hSub φ φ') h₁ h₂) (HSub.h …
  -/
  simp only [sub_eq_add_neg, opcyclesMap'_add, opcyclesMap'_neg]
  /-
    🎉 no goals
  -/


@[simp]
lemma rightHomologyMap_neg : rightHomologyMap (-φ) = -rightHomologyMap φ :=
  rightHomologyMap'_neg _ _


@[simp]
lemma opcyclesMap_neg : opcyclesMap (-φ) = -opcyclesMap φ :=
  opcyclesMap'_neg _ _


@[simp]
lemma rightHomologyMap_add :
    rightHomologyMap (φ + φ') = rightHomologyMap φ + rightHomologyMap φ' :=
  rightHomologyMap'_add _ _


@[simp]
lemma opcyclesMap_add : opcyclesMap (φ + φ') = opcyclesMap φ + opcyclesMap φ' :=
  opcyclesMap'_add _ _


@[simp]
lemma rightHomologyMap_sub :
    rightHomologyMap (φ - φ') = rightHomologyMap φ - rightHomologyMap φ' :=
  rightHomologyMap'_sub _ _


@[simp]
lemma opcyclesMap_sub : opcyclesMap (φ - φ') = opcyclesMap φ - opcyclesMap φ' :=
  opcyclesMap'_sub _ _


instance rightHomologyFunctor_additive [HasKernels C] [HasCokernels C] :
  (rightHomologyFunctor C).Additive where


instance opcyclesFunctor_additive [HasKernels C] [HasCokernels C] :
  (opcyclesFunctor C).Additive where


/-- Given a homology map data for a morphism `φ`, this is the induced homology
map data for `-φ`. -/
@[simps]
def neg : HomologyMapData (-φ) h₁ h₂ where
  left := γ.left.neg
  right := γ.right.neg


/-- Given homology map data for morphisms `φ` and `φ'`, this is the induced homology
map data for `φ + φ'`. -/
@[simps]
def add : HomologyMapData (φ + φ') h₁ h₂ where
  left := γ.left.add γ'.left
  right := γ.right.add γ'.right


@[simp]
lemma homologyMap'_neg :
    homologyMap' (-φ) h₁ h₂ = -homologyMap' φ h₁ h₂ :=
  leftHomologyMap'_neg _ _


@[simp]
lemma homologyMap'_add :
    homologyMap' (φ + φ') h₁ h₂ = homologyMap' φ h₁ h₂ + homologyMap' φ' h₁ h₂ :=
  leftHomologyMap'_add _ _


@[simp]
lemma homologyMap'_sub :
    homologyMap' (φ - φ') h₁ h₂ = homologyMap' φ h₁ h₂ - homologyMap' φ' h₁ h₂ :=
  leftHomologyMap'_sub _ _


@[simp]
lemma homologyMap_neg : homologyMap (-φ) = -homologyMap φ :=
  homologyMap'_neg _ _


@[simp]
lemma homologyMap_add : homologyMap (φ + φ') = homologyMap φ + homologyMap φ' :=
  homologyMap'_add _ _


@[simp]
lemma homologyMap_sub : homologyMap (φ - φ') = homologyMap φ - homologyMap φ' :=
  homologyMap'_sub _ _


instance homologyFunctor_additive [CategoryWithHomology C] :
  (homologyFunctor C).Additive where


/-- A homotopy between two morphisms of short complexes `S₁ ⟶ S₂` consists of various
maps and conditions which will be sufficient to show that they induce the same morphism
in homology. -/
@[ext]
structure Homotopy where
  /-- a morphism `S₁.X₁ ⟶ S₂.X₁` -/
  h₀ : S₁.X₁ ⟶ S₂.X₁
  h₀_f : h₀ ≫ S₂.f = 0 := by aesop_cat
  /-- a morphism `S₁.X₂ ⟶ S₂.X₁` -/
  h₁ : S₁.X₂ ⟶ S₂.X₁
  /-- a morphism `S₁.X₃ ⟶ S₂.X₂` -/
  h₂ : S₁.X₃ ⟶ S₂.X₂
  /-- a morphism `S₁.X₃ ⟶ S₂.X₃` -/
  h₃ : S₁.X₃ ⟶ S₂.X₃
  g_h₃ : S₁.g ≫ h₃ = 0 := by aesop_cat
  comm₁ : φ₁.τ₁ = S₁.f ≫ h₁ + h₀ + φ₂.τ₁ := by aesop_cat
  comm₂ : φ₁.τ₂ = S₁.g ≫ h₂ + h₁ ≫ S₂.f + φ₂.τ₂ := by aesop_cat
  comm₃ : φ₁.τ₃ = h₃ + h₂ ≫ S₂.g + φ₂.τ₃ := by aesop_cat


attribute [reassoc (attr := simp)] Homotopy.h₀_f Homotopy.g_h₃


/-- Constructor for null homotopic morphisms, see also `Homotopy.ofNullHomotopic`
and `Homotopy.eq_add_nullHomotopic`. -/
@[simps]
def nullHomotopic (h₀ : S₁.X₁ ⟶ S₂.X₁) (h₀_f : h₀ ≫ S₂.f = 0)
    (h₁ : S₁.X₂ ⟶ S₂.X₁) (h₂ : S₁.X₃ ⟶ S₂.X₂) (h₃ : S₁.X₃ ⟶ S₂.X₃) (g_h₃ : S₁.g ≫ h₃ = 0) :
    S₁ ⟶ S₂ where
  τ₁ := h₀ + S₁.f ≫ h₁
  τ₂ := h₁ ≫ S₂.f + S₁.g ≫ h₂
  τ₃ := h₂ ≫ S₂.g + h₃


/-- The obvious homotopy between two equal morphisms of short complexes. -/
@[simps]
def ofEq (h : φ₁ = φ₂) : Homotopy φ₁ φ₂ where
  h₀ := 0
  h₁ := 0
  h₂ := 0
  h₃ := 0


/-- The obvious homotopy between a morphism of short complexes and itself. -/
@[simps!]
def refl (φ : S₁ ⟶ S₂) : Homotopy φ φ := ofEq rfl


/-- The symmetry of homotopy between morphisms of short complexes. -/
@[simps]
def symm (h : Homotopy φ₁ φ₂) : Homotopy φ₂ φ₁ where
  h₀ := -h.h₀
  h₁ := -h.h₁
  h₂ := -h.h₂
  h₃ := -h.h₃
              /-
                C : Type u_1
                inst✝¹ : CategoryTheory.Category.{?u.99183, u_1} C
                inst✝ : CategoryTheory.Preadditive C
                S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                φ₁ φ₂ φ₃ φ₄ : Quiver.Hom S₁ S₂
                h : CategoryTheory.ShortComplex.Homotopy φ₁ φ₂
                ⊢ Eq φ₂.τ₁ (HAdd.hAdd (HAdd.hAdd (CategoryTheory.CategoryStruct.comp S₁.f (Neg …
              -/
                                      /-
                                        🎉 no goals
                                      -/
  comm₁ := by rw [h.comm₁, comp_neg]; abel
                                      /-
                                        🎉 no goals
                                      -/
              /-
                C : Type u_1
                inst✝¹ : CategoryTheory.Category.{?u.99183, u_1} C
                inst✝ : CategoryTheory.Preadditive C
                S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                φ₁ φ₂ φ₃ φ₄ : Quiver.Hom S₁ S₂
                h : CategoryTheory.ShortComplex.Homotopy φ₁ φ₂
                ⊢ Eq φ₂.τ₂ (HAdd.hAdd (HAdd.hAdd (CategoryTheory.CategoryStruct.comp S₁.g (Neg …
              -/
                                                /-
                                                  🎉 no goals
                                                -/
  comm₂ := by rw [h.comm₂, comp_neg, neg_comp]; abel
                                                /-
                                                  🎉 no goals
                                                -/
              /-
                C : Type u_1
                inst✝¹ : CategoryTheory.Category.{?u.99183, u_1} C
                inst✝ : CategoryTheory.Preadditive C
                S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                φ₁ φ₂ φ₃ φ₄ : Quiver.Hom S₁ S₂
                h : CategoryTheory.ShortComplex.Homotopy φ₁ φ₂
                ⊢ Eq φ₂.τ₃ (HAdd.hAdd (HAdd.hAdd (Neg.neg h.h₃) (CategoryTheory.CategoryStruct …
              -/
                                      /-
                                        🎉 no goals
                                      -/
  comm₃ := by rw [h.comm₃, neg_comp]; abel
                                      /-
                                        🎉 no goals
                                      -/


/-- If two maps of short complexes are homotopic, their opposites also are. -/
@[simps]
def neg (h : Homotopy φ₁ φ₂) : Homotopy (-φ₁) (-φ₂) where
  h₀ := -h.h₀
  h₁ := -h.h₁
  h₂ := -h.h₂
  h₃ := -h.h₃
              /-
                C : Type u_1
                inst✝¹ : CategoryTheory.Category.{?u.104002, u_1} C
                inst✝ : CategoryTheory.Preadditive C
                S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                φ₁ φ₂ φ₃ φ₄ : Quiver.Hom S₁ S₂
                h : CategoryTheory.ShortComplex.Homotopy φ₁ φ₂
                ⊢ Eq (Neg.neg φ₁).τ₁ (HAdd.hAdd (HAdd.hAdd (CategoryTheory.CategoryStruct.comp …
              -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  comm₁ := by rw [neg_τ₁, neg_τ₁, h.comm₁, neg_add_rev, comp_neg]; abel
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
              /-
                C : Type u_1
                inst✝¹ : CategoryTheory.Category.{?u.104002, u_1} C
                inst✝ : CategoryTheory.Preadditive C
                S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                φ₁ φ₂ φ₃ φ₄ : Quiver.Hom S₁ S₂
                h : CategoryTheory.ShortComplex.Homotopy φ₁ φ₂
                ⊢ Eq (Neg.neg φ₁).τ₂ (HAdd.hAdd (HAdd.hAdd (CategoryTheory.CategoryStruct.comp …
              -/
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
  comm₂ := by rw [neg_τ₂, neg_τ₂, h.comm₂, neg_add_rev, comp_neg, neg_comp]; abel
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
              /-
                C : Type u_1
                inst✝¹ : CategoryTheory.Category.{?u.104002, u_1} C
                inst✝ : CategoryTheory.Preadditive C
                S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                φ₁ φ₂ φ₃ φ₄ : Quiver.Hom S₁ S₂
                h : CategoryTheory.ShortComplex.Homotopy φ₁ φ₂
                ⊢ Eq (Neg.neg φ₁).τ₃ (HAdd.hAdd (HAdd.hAdd (Neg.neg h.h₃) (CategoryTheory.Cate …
              -/
                                                      /-
                                                        🎉 no goals
                                                      -/
  comm₃ := by rw [neg_τ₃, neg_τ₃, h.comm₃, neg_comp]; abel
                                                      /-
                                                        🎉 no goals
                                                      -/


/-- The transitivity of homotopy between morphisms of short complexes. -/
@[simps]
def trans (h₁₂ : Homotopy φ₁ φ₂) (h₂₃ : Homotopy φ₂ φ₃) : Homotopy φ₁ φ₃ where
  h₀ := h₁₂.h₀ + h₂₃.h₀
  h₁ := h₁₂.h₁ + h₂₃.h₁
  h₂ := h₁₂.h₂ + h₂₃.h₂
  h₃ := h₁₂.h₃ + h₂₃.h₃
              /-
                C : Type u_1
                inst✝¹ : CategoryTheory.Category.{?u.112447, u_1} C
                inst✝ : CategoryTheory.Preadditive C
                S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                φ₁ φ₂ φ₃ φ₄ : Quiver.Hom S₁ S₂
                h₁₂ : CategoryTheory.ShortComplex.Homotopy φ₁ φ₂
                h₂₃ : CategoryTheory.ShortComplex.Homotopy φ₂ φ₃
                ⊢ Eq φ₁.τ₁ (HAdd.hAdd (HAdd.hAdd (CategoryTheory.CategoryStruct.comp S₁.f (HAd …
              -/
                                                   /-
                                                     🎉 no goals
                                                   -/
  comm₁ := by rw [h₁₂.comm₁, h₂₃.comm₁, comp_add]; abel
                                                   /-
                                                     🎉 no goals
                                                   -/
              /-
                C : Type u_1
                inst✝¹ : CategoryTheory.Category.{?u.112447, u_1} C
                inst✝ : CategoryTheory.Preadditive C
                S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                φ₁ φ₂ φ₃ φ₄ : Quiver.Hom S₁ S₂
                h₁₂ : CategoryTheory.ShortComplex.Homotopy φ₁ φ₂
                h₂₃ : CategoryTheory.ShortComplex.Homotopy φ₂ φ₃
                ⊢ Eq φ₁.τ₂ (HAdd.hAdd (HAdd.hAdd (CategoryTheory.CategoryStruct.comp S₁.g (HAd …
              -/
                                                             /-
                                                               🎉 no goals
                                                             -/
  comm₂ := by rw [h₁₂.comm₂, h₂₃.comm₂, comp_add, add_comp]; abel
                                                             /-
                                                               🎉 no goals
                                                             -/
              /-
                C : Type u_1
                inst✝¹ : CategoryTheory.Category.{?u.112447, u_1} C
                inst✝ : CategoryTheory.Preadditive C
                S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                φ₁ φ₂ φ₃ φ₄ : Quiver.Hom S₁ S₂
                h₁₂ : CategoryTheory.ShortComplex.Homotopy φ₁ φ₂
                h₂₃ : CategoryTheory.ShortComplex.Homotopy φ₂ φ₃
                ⊢ Eq φ₁.τ₃ (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd h₁₂.h₃ h₂₃.h₃) (CategoryTheory.Cat …
              -/
                                                   /-
                                                     🎉 no goals
                                                   -/
  comm₃ := by rw [h₁₂.comm₃, h₂₃.comm₃, add_comp]; abel
                                                   /-
                                                     🎉 no goals
                                                   -/


/-- Homotopy between morphisms of short complexes is compatible with addition. -/
@[simps]
def add (h : Homotopy φ₁ φ₂) (h' : Homotopy φ₃ φ₄) : Homotopy (φ₁ + φ₃) (φ₂ + φ₄) where
  h₀ := h.h₀ + h'.h₀
  h₁ := h.h₁ + h'.h₁
  h₂ := h.h₂ + h'.h₂
  h₃ := h.h₃ + h'.h₃
              /-
                C : Type u_1
                inst✝¹ : CategoryTheory.Category.{?u.121308, u_1} C
                inst✝ : CategoryTheory.Preadditive C
                S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                φ₁ φ₂ φ₃ φ₄ : Quiver.Hom S₁ S₂
                h : CategoryTheory.ShortComplex.Homotopy φ₁ φ₂
                h' : CategoryTheory.ShortComplex.Homotopy φ₃ φ₄
                ⊢ Eq (HAdd.hAdd φ₁ φ₃).τ₁ (HAdd.hAdd (HAdd.hAdd (CategoryTheory.CategoryStruct …
              -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
  comm₁ := by rw [add_τ₁, add_τ₁, h.comm₁, h'.comm₁, comp_add]; abel
                                                                /-
                                                                  🎉 no goals
                                                                -/
              /-
                C : Type u_1
                inst✝¹ : CategoryTheory.Category.{?u.121308, u_1} C
                inst✝ : CategoryTheory.Preadditive C
                S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                φ₁ φ₂ φ₃ φ₄ : Quiver.Hom S₁ S₂
                h : CategoryTheory.ShortComplex.Homotopy φ₁ φ₂
                h' : CategoryTheory.ShortComplex.Homotopy φ₃ φ₄
                ⊢ Eq (HAdd.hAdd φ₁ φ₃).τ₂ (HAdd.hAdd (HAdd.hAdd (CategoryTheory.CategoryStruct …
              -/
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
  comm₂ := by rw [add_τ₂, add_τ₂, h.comm₂, h'.comm₂, comp_add, add_comp]; abel
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
              /-
                C : Type u_1
                inst✝¹ : CategoryTheory.Category.{?u.121308, u_1} C
                inst✝ : CategoryTheory.Preadditive C
                S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                φ₁ φ₂ φ₃ φ₄ : Quiver.Hom S₁ S₂
                h : CategoryTheory.ShortComplex.Homotopy φ₁ φ₂
                h' : CategoryTheory.ShortComplex.Homotopy φ₃ φ₄
                ⊢ Eq (HAdd.hAdd φ₁ φ₃).τ₃ (HAdd.hAdd (HAdd.hAdd (HAdd.hAdd h.h₃ h'.h₃) (Catego …
              -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
  comm₃ := by rw [add_τ₃, add_τ₃, h.comm₃, h'.comm₃, add_comp]; abel
                                                                /-
                                                                  🎉 no goals
                                                                -/


/-- Homotopy between morphisms of short complexes is compatible with subtraction. -/
@[simps]
def sub (h : Homotopy φ₁ φ₂) (h' : Homotopy φ₃ φ₄) : Homotopy (φ₁ - φ₃) (φ₂ - φ₄) where
  h₀ := h.h₀ - h'.h₀
  h₁ := h.h₁ - h'.h₁
  h₂ := h.h₂ - h'.h₂
  h₃ := h.h₃ - h'.h₃
              /-
                C : Type u_1
                inst✝¹ : CategoryTheory.Category.{?u.132317, u_1} C
                inst✝ : CategoryTheory.Preadditive C
                S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                φ₁ φ₂ φ₃ φ₄ : Quiver.Hom S₁ S₂
                h : CategoryTheory.ShortComplex.Homotopy φ₁ φ₂
                h' : CategoryTheory.ShortComplex.Homotopy φ₃ φ₄
                ⊢ Eq (HSub.hSub φ₁ φ₃).τ₁ (HAdd.hAdd (HAdd.hAdd (CategoryTheory.CategoryStruct …
              -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
  comm₁ := by rw [sub_τ₁, sub_τ₁, h.comm₁, h'.comm₁, comp_sub]; abel
                                                                /-
                                                                  🎉 no goals
                                                                -/
              /-
                C : Type u_1
                inst✝¹ : CategoryTheory.Category.{?u.132317, u_1} C
                inst✝ : CategoryTheory.Preadditive C
                S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                φ₁ φ₂ φ₃ φ₄ : Quiver.Hom S₁ S₂
                h : CategoryTheory.ShortComplex.Homotopy φ₁ φ₂
                h' : CategoryTheory.ShortComplex.Homotopy φ₃ φ₄
                ⊢ Eq (HSub.hSub φ₁ φ₃).τ₂ (HAdd.hAdd (HAdd.hAdd (CategoryTheory.CategoryStruct …
              -/
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
  comm₂ := by rw [sub_τ₂, sub_τ₂, h.comm₂, h'.comm₂, comp_sub, sub_comp]; abel
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
              /-
                C : Type u_1
                inst✝¹ : CategoryTheory.Category.{?u.132317, u_1} C
                inst✝ : CategoryTheory.Preadditive C
                S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                φ₁ φ₂ φ₃ φ₄ : Quiver.Hom S₁ S₂
                h : CategoryTheory.ShortComplex.Homotopy φ₁ φ₂
                h' : CategoryTheory.ShortComplex.Homotopy φ₃ φ₄
                ⊢ Eq (HSub.hSub φ₁ φ₃).τ₃ (HAdd.hAdd (HAdd.hAdd (HSub.hSub h.h₃ h'.h₃) (Catego …
              -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
  comm₃ := by rw [sub_τ₃, sub_τ₃, h.comm₃, h'.comm₃, sub_comp]; abel
                                                                /-
                                                                  🎉 no goals
                                                                -/


/-- Homotopy between morphisms of short complexes is compatible with precomposition. -/
@[simps]
def compLeft (h : Homotopy φ₁ φ₂) (ψ : S₃ ⟶ S₁) : Homotopy (ψ ≫ φ₁) (ψ ≫ φ₂) where
  h₀ := ψ.τ₁ ≫ h.h₀
  h₁ := ψ.τ₂ ≫ h.h₁
  h₂ := ψ.τ₃ ≫ h.h₂
  h₃ := ψ.τ₃ ≫ h.h₃
             /-
               C : Type u_1
               inst✝¹ : CategoryTheory.Category.{?u.145795, u_1} C
               inst✝ : CategoryTheory.Preadditive C
               S₁ S₂ S₃ : CategoryTheory.ShortComplex C
               φ₁ φ₂ φ₃ φ₄ : Quiver.Hom S₁ S₂
               h : CategoryTheory.ShortComplex.Homotopy φ₁ φ₂
               ψ : Quiver.Hom S₃ S₁
               ⊢ Eq (CategoryTheory.CategoryStruct.comp S₃.g (CategoryTheory.CategoryStruct.c …
             -/
  g_h₃ := by rw [← ψ.comm₂₃_assoc, h.g_h₃, comp_zero]
             /-
               🎉 no goals
             -/
              /-
                C : Type u_1
                inst✝¹ : CategoryTheory.Category.{?u.145795, u_1} C
                inst✝ : CategoryTheory.Preadditive C
                S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                φ₁ φ₂ φ₃ φ₄ : Quiver.Hom S₁ S₂
                h : CategoryTheory.ShortComplex.Homotopy φ₁ φ₂
                ψ : Quiver.Hom S₃ S₁
                ⊢ Eq (CategoryTheory.CategoryStruct.comp ψ φ₁).τ₁ (HAdd.hAdd (HAdd.hAdd (Categ …
              -/
  comm₁ := by rw [comp_τ₁, comp_τ₁, h.comm₁, comp_add, comp_add, add_left_inj, ψ.comm₁₂_assoc]
              /-
                🎉 no goals
              -/
              /-
                C : Type u_1
                inst✝¹ : CategoryTheory.Category.{?u.145795, u_1} C
                inst✝ : CategoryTheory.Preadditive C
                S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                φ₁ φ₂ φ₃ φ₄ : Quiver.Hom S₁ S₂
                h : CategoryTheory.ShortComplex.Homotopy φ₁ φ₂
                ψ : Quiver.Hom S₃ S₁
                ⊢ Eq (CategoryTheory.CategoryStruct.comp ψ φ₁).τ₂ (HAdd.hAdd (HAdd.hAdd (Categ …
              -/
  comm₂ := by rw [comp_τ₂, comp_τ₂, h.comm₂, comp_add, comp_add, assoc, ψ.comm₂₃_assoc]
              /-
                🎉 no goals
              -/
              /-
                C : Type u_1
                inst✝¹ : CategoryTheory.Category.{?u.145795, u_1} C
                inst✝ : CategoryTheory.Preadditive C
                S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                φ₁ φ₂ φ₃ φ₄ : Quiver.Hom S₁ S₂
                h : CategoryTheory.ShortComplex.Homotopy φ₁ φ₂
                ψ : Quiver.Hom S₃ S₁
                ⊢ Eq (CategoryTheory.CategoryStruct.comp ψ φ₁).τ₃ (HAdd.hAdd (HAdd.hAdd (Categ …
              -/
  comm₃ := by rw [comp_τ₃, comp_τ₃, h.comm₃, comp_add, comp_add, assoc]
              /-
                🎉 no goals
              -/


/-- Homotopy between morphisms of short complexes is compatible with postcomposition. -/
@[simps]
def compRight (h : Homotopy φ₁ φ₂) (ψ : S₂ ⟶ S₃) : Homotopy (φ₁ ≫ ψ) (φ₂ ≫ ψ) where
  h₀ := h.h₀ ≫ ψ.τ₁
  h₁ := h.h₁ ≫ ψ.τ₁
  h₂ := h.h₂ ≫ ψ.τ₂
  h₃ := h.h₃ ≫ ψ.τ₃
              /-
                C : Type u_1
                inst✝¹ : CategoryTheory.Category.{?u.154804, u_1} C
                inst✝ : CategoryTheory.Preadditive C
                S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                φ₁ φ₂ φ₃ φ₄ : Quiver.Hom S₁ S₂
                h : CategoryTheory.ShortComplex.Homotopy φ₁ φ₂
                ψ : Quiver.Hom S₂ S₃
                ⊢ Eq (CategoryTheory.CategoryStruct.comp φ₁ ψ).τ₁ (HAdd.hAdd (HAdd.hAdd (Categ …
              -/
  comm₁ := by rw [comp_τ₁, comp_τ₁, h.comm₁, add_comp, add_comp, assoc]
              /-
                🎉 no goals
              -/
              /-
                C : Type u_1
                inst✝¹ : CategoryTheory.Category.{?u.154804, u_1} C
                inst✝ : CategoryTheory.Preadditive C
                S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                φ₁ φ₂ φ₃ φ₄ : Quiver.Hom S₁ S₂
                h : CategoryTheory.ShortComplex.Homotopy φ₁ φ₂
                ψ : Quiver.Hom S₂ S₃
                ⊢ Eq (CategoryTheory.CategoryStruct.comp φ₁ ψ).τ₂ (HAdd.hAdd (HAdd.hAdd (Categ …
              -/
  comm₂ := by rw [comp_τ₂, comp_τ₂, h.comm₂, add_comp, add_comp, assoc, assoc, assoc, ψ.comm₁₂]
              /-
                🎉 no goals
              -/
              /-
                C : Type u_1
                inst✝¹ : CategoryTheory.Category.{?u.154804, u_1} C
                inst✝ : CategoryTheory.Preadditive C
                S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                φ₁ φ₂ φ₃ φ₄ : Quiver.Hom S₁ S₂
                h : CategoryTheory.ShortComplex.Homotopy φ₁ φ₂
                ψ : Quiver.Hom S₂ S₃
                ⊢ Eq (CategoryTheory.CategoryStruct.comp φ₁ ψ).τ₃ (HAdd.hAdd (HAdd.hAdd (Categ …
              -/
  comm₃ := by rw [comp_τ₃, comp_τ₃, h.comm₃, add_comp, add_comp, assoc, assoc, ψ.comm₂₃]
              /-
                🎉 no goals
              -/


/-- Homotopy between morphisms of short complexes is compatible with composition. -/
@[simps!]
def comp (h : Homotopy φ₁ φ₂) {ψ₁ ψ₂ : S₂ ⟶ S₃} (h' : Homotopy ψ₁ ψ₂) :
    Homotopy (φ₁ ≫ ψ₁) (φ₂ ≫ ψ₂) :=
  (h.compRight ψ₁).trans (h'.compLeft φ₂)


/-- The homotopy between morphisms in `ShortComplex Cᵒᵖ` that is induced by a homotopy
between morphisms in `ShortComplex C`. -/
@[simps]
def op (h : Homotopy φ₁ φ₂) : Homotopy (opMap φ₁) (opMap φ₂) where
  h₀ := h.h₃.op
  h₁ := h.h₂.op
  h₂ := h.h₁.op
  h₃ := h.h₀.op
  h₀_f := Quiver.Hom.unop_inj h.g_h₃
  g_h₃ := Quiver.Hom.unop_inj h.h₀_f
                                   /-
                                     C : Type u_1
                                     inst✝¹ : CategoryTheory.Category.{?u.167499, u_1} C
                                     inst✝ : CategoryTheory.Preadditive C
                                     S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                                     φ₁ φ₂ φ₃ φ₄ : Quiver.Hom S₁ S₂
                                     h : CategoryTheory.ShortComplex.Homotopy φ₁ φ₂
                                     ⊢ Eq (CategoryTheory.ShortComplex.opMap φ₁).τ₁.unop (HAdd.hAdd (HAdd.hAdd (Cat …
                                   -/
                                                        /-
                                                          🎉 no goals
                                                        -/
  comm₁ := Quiver.Hom.unop_inj (by dsimp; rw [h.comm₃]; abel)
                                                        /-
                                                          🎉 no goals
                                                        -/
                                   /-
                                     C : Type u_1
                                     inst✝¹ : CategoryTheory.Category.{?u.167499, u_1} C
                                     inst✝ : CategoryTheory.Preadditive C
                                     S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                                     φ₁ φ₂ φ₃ φ₄ : Quiver.Hom S₁ S₂
                                     h : CategoryTheory.ShortComplex.Homotopy φ₁ φ₂
                                     ⊢ Eq (CategoryTheory.ShortComplex.opMap φ₁).τ₂.unop (HAdd.hAdd (HAdd.hAdd (Cat …
                                   -/
                                                        /-
                                                          🎉 no goals
                                                        -/
  comm₂ := Quiver.Hom.unop_inj (by dsimp; rw [h.comm₂]; abel)
                                                        /-
                                                          🎉 no goals
                                                        -/
                                   /-
                                     C : Type u_1
                                     inst✝¹ : CategoryTheory.Category.{?u.167499, u_1} C
                                     inst✝ : CategoryTheory.Preadditive C
                                     S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                                     φ₁ φ₂ φ₃ φ₄ : Quiver.Hom S₁ S₂
                                     h : CategoryTheory.ShortComplex.Homotopy φ₁ φ₂
                                     ⊢ Eq (CategoryTheory.ShortComplex.opMap φ₁).τ₃.unop (HAdd.hAdd (HAdd.hAdd h.h₀ …
                                   -/
                                                        /-
                                                          🎉 no goals
                                                        -/
  comm₃ := Quiver.Hom.unop_inj (by dsimp; rw [h.comm₁]; abel)
                                                        /-
                                                          🎉 no goals
                                                        -/


/-- The homotopy between morphisms in `ShortComplex C` that is induced by a homotopy
between morphisms in `ShortComplex Cᵒᵖ`. -/
@[simps]
def unop {S₁ S₂ : ShortComplex Cᵒᵖ} {φ₁ φ₂ : S₁ ⟶ S₂} (h : Homotopy φ₁ φ₂) :
    Homotopy (unopMap φ₁) (unopMap φ₂) where
  h₀ := h.h₃.unop
  h₁ := h.h₂.unop
  h₂ := h.h₁.unop
  h₃ := h.h₀.unop
  h₀_f := Quiver.Hom.op_inj h.g_h₃
  g_h₃ := Quiver.Hom.op_inj h.h₀_f
                                 /-
                                   C : Type u_1
                                   inst✝¹ : CategoryTheory.Category.{?u.172508, u_1} C
                                   inst✝ : CategoryTheory.Preadditive C
                                   S₁✝ S₂✝ S₃ : CategoryTheory.ShortComplex C
                                   φ₁✝ φ₂✝ φ₃ φ₄ : Quiver.Hom S₁✝ S₂✝
                                   S₁ S₂ : CategoryTheory.ShortComplex (Opposite C)
                                   φ₁ φ₂ : Quiver.Hom S₁ S₂
                                   h : CategoryTheory.ShortComplex.Homotopy φ₁ φ₂
                                   ⊢ Eq (CategoryTheory.ShortComplex.unopMap φ₁).τ₁.op (HAdd.hAdd (HAdd.hAdd (Cat …
                                 -/
                                                      /-
                                                        🎉 no goals
                                                      -/
  comm₁ := Quiver.Hom.op_inj (by dsimp; rw [h.comm₃]; abel)
                                                      /-
                                                        🎉 no goals
                                                      -/
                                 /-
                                   C : Type u_1
                                   inst✝¹ : CategoryTheory.Category.{?u.172508, u_1} C
                                   inst✝ : CategoryTheory.Preadditive C
                                   S₁✝ S₂✝ S₃ : CategoryTheory.ShortComplex C
                                   φ₁✝ φ₂✝ φ₃ φ₄ : Quiver.Hom S₁✝ S₂✝
                                   S₁ S₂ : CategoryTheory.ShortComplex (Opposite C)
                                   φ₁ φ₂ : Quiver.Hom S₁ S₂
                                   h : CategoryTheory.ShortComplex.Homotopy φ₁ φ₂
                                   ⊢ Eq (CategoryTheory.ShortComplex.unopMap φ₁).τ₂.op (HAdd.hAdd (HAdd.hAdd (Cat …
                                 -/
                                                      /-
                                                        🎉 no goals
                                                      -/
  comm₂ := Quiver.Hom.op_inj (by dsimp; rw [h.comm₂]; abel)
                                                      /-
                                                        🎉 no goals
                                                      -/
                                 /-
                                   C : Type u_1
                                   inst✝¹ : CategoryTheory.Category.{?u.172508, u_1} C
                                   inst✝ : CategoryTheory.Preadditive C
                                   S₁✝ S₂✝ S₃ : CategoryTheory.ShortComplex C
                                   φ₁✝ φ₂✝ φ₃ φ₄ : Quiver.Hom S₁✝ S₂✝
                                   S₁ S₂ : CategoryTheory.ShortComplex (Opposite C)
                                   φ₁ φ₂ : Quiver.Hom S₁ S₂
                                   h : CategoryTheory.ShortComplex.Homotopy φ₁ φ₂
                                   ⊢ Eq (CategoryTheory.ShortComplex.unopMap φ₁).τ₃.op (HAdd.hAdd (HAdd.hAdd h.h₀ …
                                 -/
                                                      /-
                                                        🎉 no goals
                                                      -/
  comm₃ := Quiver.Hom.op_inj (by dsimp; rw [h.comm₁]; abel)
                                                      /-
                                                        🎉 no goals
                                                      -/


/-- Equivalence expressing that two morphisms are homotopic iff
their difference is homotopic to zero. -/
@[simps]
def equivSubZero : Homotopy φ₁ φ₂ ≃ Homotopy (φ₁ - φ₂) 0 where
  toFun h := (h.sub (refl φ₂)).trans (ofEq (sub_self φ₂))
  invFun h := ((ofEq (sub_add_cancel φ₁ φ₂).symm).trans
    (h.add (refl φ₂))).trans (ofEq (zero_add φ₂))
                 /-
                   C : Type u_1
                   inst✝¹ : CategoryTheory.Category.{?u.177613, u_1} C
                   inst✝ : CategoryTheory.Preadditive C
                   S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                   φ₁ φ₂ φ₃ φ₄ : Quiver.Hom S₁ S₂
                   ⊢ Function.LeftInverse (fun h => ((CategoryTheory.ShortComplex.Homotopy.ofEq ⋯ …
                 -/
  left_inv := by aesop_cat
                 /-
                   🎉 no goals
                 -/
                  /-
                    C : Type u_1
                    inst✝¹ : CategoryTheory.Category.{?u.177613, u_1} C
                    inst✝ : CategoryTheory.Preadditive C
                    S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                    φ₁ φ₂ φ₃ φ₄ : Quiver.Hom S₁ S₂
                    ⊢ Function.RightInverse (fun h => ((CategoryTheory.ShortComplex.Homotopy.ofEq  …
                  -/
  right_inv := by aesop_cat
                  /-
                    🎉 no goals
                  -/


lemma eq_add_nullHomotopic (h : Homotopy φ₁ φ₂) :
    φ₁ = φ₂ + nullHomotopic _ _ h.h₀ h.h₀_f h.h₁ h.h₂ h.h₃ h.g_h₃ := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ₁ φ₂ : Quiver.Hom S₁ S₂
    h : CategoryTheory.ShortComplex.Homotopy φ₁ φ₂
    ⊢ Eq φ₁ (HAdd.hAdd φ₂ (S₁.nullHomotopic S₂ h.h₀ ⋯ h.h₁ h.h₂ h.h₃ ⋯))
  -/
  ext
    /-
      case h₁
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      S₁ S₂ : CategoryTheory.ShortComplex C
      φ₁ φ₂ : Quiver.Hom S₁ S₂
      h : CategoryTheory.ShortComplex.Homotopy φ₁ φ₂
      ⊢ Eq φ₁.τ₁ (HAdd.hAdd φ₂ (S₁.nullHomotopic S₂ h.h₀ ⋯ h.h₁ h.h₂ h.h₃ ⋯)).τ₁
    -/
                         /-
                           🎉 no goals
                         -/
  · dsimp; rw [h.comm₁]; abel
                         /-
                           🎉 no goals
                         -/
    /-
      case h₂
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      S₁ S₂ : CategoryTheory.ShortComplex C
      φ₁ φ₂ : Quiver.Hom S₁ S₂
      h : CategoryTheory.ShortComplex.Homotopy φ₁ φ₂
      ⊢ Eq φ₁.τ₂ (HAdd.hAdd φ₂ (S₁.nullHomotopic S₂ h.h₀ ⋯ h.h₁ h.h₂ h.h₃ ⋯)).τ₂
    -/
                         /-
                           🎉 no goals
                         -/
  · dsimp; rw [h.comm₂]; abel
                         /-
                           🎉 no goals
                         -/
    /-
      case h₃
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      S₁ S₂ : CategoryTheory.ShortComplex C
      φ₁ φ₂ : Quiver.Hom S₁ S₂
      h : CategoryTheory.ShortComplex.Homotopy φ₁ φ₂
      ⊢ Eq φ₁.τ₃ (HAdd.hAdd φ₂ (S₁.nullHomotopic S₂ h.h₀ ⋯ h.h₁ h.h₂ h.h₃ ⋯)).τ₃
    -/
                         /-
                           🎉 no goals
                         -/
  · dsimp; rw [h.comm₃]; abel
                         /-
                           🎉 no goals
                         -/


/-- A morphism constructed with `nullHomotopic` is homotopic to zero. -/
@[simps]
def ofNullHomotopic (h₀ : S₁.X₁ ⟶ S₂.X₁) (h₀_f : h₀ ≫ S₂.f = 0)
    (h₁ : S₁.X₂ ⟶ S₂.X₁) (h₂ : S₁.X₃ ⟶ S₂.X₂) (h₃ : S₁.X₃ ⟶ S₂.X₃) (g_h₃ : S₁.g ≫ h₃ = 0) :
  Homotopy (nullHomotopic _ _ h₀ h₀_f h₁ h₂ h₃ g_h₃) 0 where
  h₀ := h₀
  h₁ := h₁
  h₂ := h₂
  h₃ := h₃
  h₀_f := h₀_f
  g_h₃ := g_h₃
              /-
                C : Type u_1
                inst✝¹ : CategoryTheory.Category.{?u.188205, u_1} C
                inst✝ : CategoryTheory.Preadditive C
                S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                φ₁ φ₂ φ₃ φ₄ : Quiver.Hom S₁ S₂
                h₀ : Quiver.Hom S₁.X₁ S₂.X₁
                h₀_f : Eq (CategoryTheory.CategoryStruct.comp h₀ S₂.f) 0
                h₁ : Quiver.Hom S₁.X₂ S₂.X₁
                h₂ : Quiver.Hom S₁.X₃ S₂.X₂
                h₃ : Quiver.Hom S₁.X₃ S₂.X₃
                g_h₃ : Eq (CategoryTheory.CategoryStruct.comp S₁.g h₃) 0
                ⊢ Eq (S₁.nullHomotopic S₂ h₀ h₀_f h₁ h₂ h₃ g_h₃).τ₁ (HAdd.hAdd (HAdd.hAdd (Cat …
              -/
                                                        /-
                                                          🎉 no goals
                                                        -/
  comm₁ := by rw [nullHomotopic_τ₁, zero_τ₁, add_zero]; abel
                                                        /-
                                                          🎉 no goals
                                                        -/
              /-
                C : Type u_1
                inst✝¹ : CategoryTheory.Category.{?u.188205, u_1} C
                inst✝ : CategoryTheory.Preadditive C
                S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                φ₁ φ₂ φ₃ φ₄ : Quiver.Hom S₁ S₂
                h₀ : Quiver.Hom S₁.X₁ S₂.X₁
                h₀_f : Eq (CategoryTheory.CategoryStruct.comp h₀ S₂.f) 0
                h₁ : Quiver.Hom S₁.X₂ S₂.X₁
                h₂ : Quiver.Hom S₁.X₃ S₂.X₂
                h₃ : Quiver.Hom S₁.X₃ S₂.X₃
                g_h₃ : Eq (CategoryTheory.CategoryStruct.comp S₁.g h₃) 0
                ⊢ Eq (S₁.nullHomotopic S₂ h₀ h₀_f h₁ h₂ h₃ g_h₃).τ₂ (HAdd.hAdd (HAdd.hAdd (Cat …
              -/
                                                        /-
                                                          🎉 no goals
                                                        -/
  comm₂ := by rw [nullHomotopic_τ₂, zero_τ₂, add_zero]; abel
                                                        /-
                                                          🎉 no goals
                                                        -/
              /-
                C : Type u_1
                inst✝¹ : CategoryTheory.Category.{?u.188205, u_1} C
                inst✝ : CategoryTheory.Preadditive C
                S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                φ₁ φ₂ φ₃ φ₄ : Quiver.Hom S₁ S₂
                h₀ : Quiver.Hom S₁.X₁ S₂.X₁
                h₀_f : Eq (CategoryTheory.CategoryStruct.comp h₀ S₂.f) 0
                h₁ : Quiver.Hom S₁.X₂ S₂.X₁
                h₂ : Quiver.Hom S₁.X₃ S₂.X₂
                h₃ : Quiver.Hom S₁.X₃ S₂.X₃
                g_h₃ : Eq (CategoryTheory.CategoryStruct.comp S₁.g h₃) 0
                ⊢ Eq (S₁.nullHomotopic S₂ h₀ h₀_f h₁ h₂ h₃ g_h₃).τ₃ (HAdd.hAdd (HAdd.hAdd h₃ ( …
              -/
                                                        /-
                                                          🎉 no goals
                                                        -/
  comm₃ := by rw [nullHomotopic_τ₃, zero_τ₃, add_zero]; abel
                                                        /-
                                                          🎉 no goals
                                                        -/


/-- The left homology map data expressing that null homotopic maps induce the zero
morphism in left homology. -/
def LeftHomologyMapData.ofNullHomotopic
    (H₁ : S₁.LeftHomologyData) (H₂ : S₂.LeftHomologyData)
    (h₀ : S₁.X₁ ⟶ S₂.X₁) (h₀_f : h₀ ≫ S₂.f = 0)
    (h₁ : S₁.X₂ ⟶ S₂.X₁) (h₂ : S₁.X₃ ⟶ S₂.X₂) (h₃ : S₁.X₃ ⟶ S₂.X₃) (g_h₃ : S₁.g ≫ h₃ = 0) :
    LeftHomologyMapData (nullHomotopic _ _ h₀ h₀_f h₁ h₂ h₃ g_h₃) H₁ H₂ where
                                        /-
                                          C : Type u_1
                                          inst✝¹ : CategoryTheory.Category.{?u.193559, u_1} C
                                          inst✝ : CategoryTheory.Preadditive C
                                          S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                                          φ₁ φ₂ φ₃ φ₄ : Quiver.Hom S₁ S₂
                                          H₁ : S₁.LeftHomologyData
                                          H₂ : S₂.LeftHomologyData
                                          h₀ : Quiver.Hom S₁.X₁ S₂.X₁
                                          h₀_f : Eq (CategoryTheory.CategoryStruct.comp h₀ S₂.f) 0
                                          h₁ : Quiver.Hom S₁.X₂ S₂.X₁
                                          h₂ : Quiver.Hom S₁.X₃ S₂.X₂
                                          h₃ : Quiver.Hom S₁.X₃ S₂.X₃
                                          g_h₃ : Eq (CategoryTheory.CategoryStruct.comp S₁.g h₃) 0
                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp H …
                                        -/
  φK := H₂.liftK (H₁.i ≫ h₁ ≫ S₂.f) (by simp)
                                        /-
                                          🎉 no goals
                                        -/
  φH := 0
  commf' := by
    rw [← cancel_mono H₂.i, assoc, LeftHomologyData.liftK_i, LeftHomologyData.f'_i_assoc,
      nullHomotopic_τ₁, add_comp, add_comp, assoc, assoc, assoc, LeftHomologyData.f'_i,
      self_eq_add_left, h₀_f]
  commπ := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.193559, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      S₁ S₂ S₃ : CategoryTheory.ShortComplex C
      φ₁ φ₂ φ₃ φ₄ : Quiver.Hom S₁ S₂
      H₁ : S₁.LeftHomologyData
      H₂ : S₂.LeftHomologyData
      h₀ : Quiver.Hom S₁.X₁ S₂.X₁
      h₀_f : Eq (CategoryTheory.CategoryStruct.comp h₀ S₂.f) 0
      h₁ : Quiver.Hom S₁.X₂ S₂.X₁
      h₂ : Quiver.Hom S₁.X₃ S₂.X₂
      h₃ : Quiver.Hom S₁.X₃ S₂.X₃
      g_h₃ : Eq (CategoryTheory.CategoryStruct.comp S₁.g h₃) 0
      ⊢ Eq (CategoryTheory.CategoryStruct.comp H₁.π 0) (CategoryTheory.CategoryStruc …
    -/
    rw [H₂.liftK_π_eq_zero_of_boundary (H₁.i ≫ h₁ ≫ S₂.f) (H₁.i ≫ h₁) (by rw [assoc]), comp_zero]
    /-
      🎉 no goals
    -/


/-- The right homology map data expressing that null homotopic maps induce the zero
morphism in right homology. -/
def RightHomologyMapData.ofNullHomotopic
    (H₁ : S₁.RightHomologyData) (H₂ : S₂.RightHomologyData)
    (h₀ : S₁.X₁ ⟶ S₂.X₁) (h₀_f : h₀ ≫ S₂.f = 0)
    (h₁ : S₁.X₂ ⟶ S₂.X₁) (h₂ : S₁.X₃ ⟶ S₂.X₂) (h₃ : S₁.X₃ ⟶ S₂.X₃) (g_h₃ : S₁.g ≫ h₃ = 0) :
    RightHomologyMapData (nullHomotopic _ _ h₀ h₀_f h₁ h₂ h₃ g_h₃) H₁ H₂ where
                                        /-
                                          C : Type u_1
                                          inst✝¹ : CategoryTheory.Category.{?u.205199, u_1} C
                                          inst✝ : CategoryTheory.Preadditive C
                                          S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                                          φ₁ φ₂ φ₃ φ₄ : Quiver.Hom S₁ S₂
                                          H₁ : S₁.RightHomologyData
                                          H₂ : S₂.RightHomologyData
                                          h₀ : Quiver.Hom S₁.X₁ S₂.X₁
                                          h₀_f : Eq (CategoryTheory.CategoryStruct.comp h₀ S₂.f) 0
                                          h₁ : Quiver.Hom S₁.X₂ S₂.X₁
                                          h₂ : Quiver.Hom S₁.X₃ S₂.X₂
                                          h₃ : Quiver.Hom S₁.X₃ S₂.X₃
                                          g_h₃ : Eq (CategoryTheory.CategoryStruct.comp S₁.g h₃) 0
                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp S₁.f (CategoryTheory.CategoryStruct.c …
                                        -/
  φQ := H₁.descQ (S₁.g ≫ h₂ ≫ H₂.p) (by simp)
                                        /-
                                          🎉 no goals
                                        -/
  φH := 0
  commg' := by
    rw [← cancel_epi H₁.p, RightHomologyData.p_descQ_assoc, RightHomologyData.p_g'_assoc,
      nullHomotopic_τ₃, comp_add, assoc, assoc, RightHomologyData.p_g', g_h₃, add_zero]
  commι := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.205199, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      S₁ S₂ S₃ : CategoryTheory.ShortComplex C
      φ₁ φ₂ φ₃ φ₄ : Quiver.Hom S₁ S₂
      H₁ : S₁.RightHomologyData
      H₂ : S₂.RightHomologyData
      h₀ : Quiver.Hom S₁.X₁ S₂.X₁
      h₀_f : Eq (CategoryTheory.CategoryStruct.comp h₀ S₂.f) 0
      h₁ : Quiver.Hom S₁.X₂ S₂.X₁
      h₂ : Quiver.Hom S₁.X₃ S₂.X₂
      h₃ : Quiver.Hom S₁.X₃ S₂.X₃
      g_h₃ : Eq (CategoryTheory.CategoryStruct.comp S₁.g h₃) 0
      ⊢ Eq (CategoryTheory.CategoryStruct.comp 0 H₂.ι) (CategoryTheory.CategoryStruc …
    -/
    rw [H₁.ι_descQ_eq_zero_of_boundary (S₁.g ≫ h₂ ≫ H₂.p) (h₂ ≫ H₂.p) rfl, zero_comp]
    /-
      🎉 no goals
    -/


@[simp]
lemma leftHomologyMap'_nullHomotopic
    (H₁ : S₁.LeftHomologyData) (H₂ : S₂.LeftHomologyData)
    (h₀ : S₁.X₁ ⟶ S₂.X₁) (h₀_f : h₀ ≫ S₂.f = 0)
    (h₁ : S₁.X₂ ⟶ S₂.X₁) (h₂ : S₁.X₃ ⟶ S₂.X₂) (h₃ : S₁.X₃ ⟶ S₂.X₃) (g_h₃ : S₁.g ≫ h₃ = 0) :
    leftHomologyMap' (nullHomotopic _ _ h₀ h₀_f h₁ h₂ h₃ g_h₃) H₁ H₂ = 0 :=
  (LeftHomologyMapData.ofNullHomotopic H₁ H₂ h₀ h₀_f h₁ h₂ h₃ g_h₃).leftHomologyMap'_eq


@[simp]
lemma rightHomologyMap'_nullHomotopic
    (H₁ : S₁.RightHomologyData) (H₂ : S₂.RightHomologyData)
    (h₀ : S₁.X₁ ⟶ S₂.X₁) (h₀_f : h₀ ≫ S₂.f = 0)
    (h₁ : S₁.X₂ ⟶ S₂.X₁) (h₂ : S₁.X₃ ⟶ S₂.X₂) (h₃ : S₁.X₃ ⟶ S₂.X₃) (g_h₃ : S₁.g ≫ h₃ = 0) :
    rightHomologyMap' (nullHomotopic _ _ h₀ h₀_f h₁ h₂ h₃ g_h₃) H₁ H₂ = 0 :=
  (RightHomologyMapData.ofNullHomotopic H₁ H₂ h₀ h₀_f h₁ h₂ h₃ g_h₃).rightHomologyMap'_eq


@[simp]
lemma homologyMap'_nullHomotopic
    (H₁ : S₁.HomologyData) (H₂ : S₂.HomologyData)
    (h₀ : S₁.X₁ ⟶ S₂.X₁) (h₀_f : h₀ ≫ S₂.f = 0)
    (h₁ : S₁.X₂ ⟶ S₂.X₁) (h₂ : S₁.X₃ ⟶ S₂.X₂) (h₃ : S₁.X₃ ⟶ S₂.X₃) (g_h₃ : S₁.g ≫ h₃ = 0) :
    homologyMap' (nullHomotopic _ _ h₀ h₀_f h₁ h₂ h₃ g_h₃) H₁ H₂ = 0 := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S₁ S₂ : CategoryTheory.ShortComplex C
    H₁ : S₁.HomologyData
    H₂ : S₂.HomologyData
    h₀ : Quiver.Hom S₁.X₁ S₂.X₁
    h₀_f : Eq (CategoryTheory.CategoryStruct.comp h₀ S₂.f) 0
    h₁ : Quiver.Hom S₁.X₂ S₂.X₁
    h₂ : Quiver.Hom S₁.X₃ S₂.X₂
    h₃ : Quiver.Hom S₁.X₃ S₂.X₃
    g_h₃ : Eq (CategoryTheory.CategoryStruct.comp S₁.g h₃) 0
    ⊢ Eq (CategoryTheory.ShortComplex.homologyMap' (S₁.nullHomotopic S₂ h₀ h₀_f h₁ …
  -/
  apply leftHomologyMap'_nullHomotopic
  /-
    🎉 no goals
  -/


@[simp]
lemma leftHomologyMap_nullHomotopic [S₁.HasLeftHomology] [S₂.HasLeftHomology]
    (h₀ : S₁.X₁ ⟶ S₂.X₁) (h₀_f : h₀ ≫ S₂.f = 0)
    (h₁ : S₁.X₂ ⟶ S₂.X₁) (h₂ : S₁.X₃ ⟶ S₂.X₂) (h₃ : S₁.X₃ ⟶ S₂.X₃) (g_h₃ : S₁.g ≫ h₃ = 0) :
    leftHomologyMap (nullHomotopic _ _ h₀ h₀_f h₁ h₂ h₃ g_h₃) = 0 := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    S₁ S₂ : CategoryTheory.ShortComplex C
    inst✝¹ : S₁.HasLeftHomology
    inst✝ : S₂.HasLeftHomology
    h₀ : Quiver.Hom S₁.X₁ S₂.X₁
    h₀_f : Eq (CategoryTheory.CategoryStruct.comp h₀ S₂.f) 0
    h₁ : Quiver.Hom S₁.X₂ S₂.X₁
    h₂ : Quiver.Hom S₁.X₃ S₂.X₂
    h₃ : Quiver.Hom S₁.X₃ S₂.X₃
    g_h₃ : Eq (CategoryTheory.CategoryStruct.comp S₁.g h₃) 0
    ⊢ Eq (CategoryTheory.ShortComplex.leftHomologyMap (S₁.nullHomotopic S₂ h₀ h₀_f …
  -/
  apply leftHomologyMap'_nullHomotopic
  /-
    🎉 no goals
  -/


@[simp]
lemma rightHomologyMap_nullHomotopic [S₁.HasRightHomology] [S₂.HasRightHomology]
    (h₀ : S₁.X₁ ⟶ S₂.X₁) (h₀_f : h₀ ≫ S₂.f = 0)
    (h₁ : S₁.X₂ ⟶ S₂.X₁) (h₂ : S₁.X₃ ⟶ S₂.X₂) (h₃ : S₁.X₃ ⟶ S₂.X₃) (g_h₃ : S₁.g ≫ h₃ = 0) :
    rightHomologyMap (nullHomotopic _ _ h₀ h₀_f h₁ h₂ h₃ g_h₃) = 0 := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    S₁ S₂ : CategoryTheory.ShortComplex C
    inst✝¹ : S₁.HasRightHomology
    inst✝ : S₂.HasRightHomology
    h₀ : Quiver.Hom S₁.X₁ S₂.X₁
    h₀_f : Eq (CategoryTheory.CategoryStruct.comp h₀ S₂.f) 0
    h₁ : Quiver.Hom S₁.X₂ S₂.X₁
    h₂ : Quiver.Hom S₁.X₃ S₂.X₂
    h₃ : Quiver.Hom S₁.X₃ S₂.X₃
    g_h₃ : Eq (CategoryTheory.CategoryStruct.comp S₁.g h₃) 0
    ⊢ Eq (CategoryTheory.ShortComplex.rightHomologyMap (S₁.nullHomotopic S₂ h₀ h₀_ …
  -/
  apply rightHomologyMap'_nullHomotopic
  /-
    🎉 no goals
  -/


@[simp]
lemma homologyMap_nullHomotopic [S₁.HasHomology] [S₂.HasHomology]
    (h₀ : S₁.X₁ ⟶ S₂.X₁) (h₀_f : h₀ ≫ S₂.f = 0)
    (h₁ : S₁.X₂ ⟶ S₂.X₁) (h₂ : S₁.X₃ ⟶ S₂.X₂) (h₃ : S₁.X₃ ⟶ S₂.X₃) (g_h₃ : S₁.g ≫ h₃ = 0) :
    homologyMap (nullHomotopic _ _ h₀ h₀_f h₁ h₂ h₃ g_h₃) = 0 := by
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    S₁ S₂ : CategoryTheory.ShortComplex C
    inst✝¹ : S₁.HasHomology
    inst✝ : S₂.HasHomology
    h₀ : Quiver.Hom S₁.X₁ S₂.X₁
    h₀_f : Eq (CategoryTheory.CategoryStruct.comp h₀ S₂.f) 0
    h₁ : Quiver.Hom S₁.X₂ S₂.X₁
    h₂ : Quiver.Hom S₁.X₃ S₂.X₂
    h₃ : Quiver.Hom S₁.X₃ S₂.X₃
    g_h₃ : Eq (CategoryTheory.CategoryStruct.comp S₁.g h₃) 0
    ⊢ Eq (CategoryTheory.ShortComplex.homologyMap (S₁.nullHomotopic S₂ h₀ h₀_f h₁  …
  -/
  apply homologyMap'_nullHomotopic
  /-
    🎉 no goals
  -/


lemma leftHomologyMap'_congr (h : Homotopy φ₁ φ₂) (h₁ : S₁.LeftHomologyData)
    (h₂ : S₂.LeftHomologyData) : leftHomologyMap' φ₁ h₁ h₂ = leftHomologyMap' φ₂ h₁ h₂ := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ₁ φ₂ : Quiver.Hom S₁ S₂
    h : CategoryTheory.ShortComplex.Homotopy φ₁ φ₂
    h₁ : S₁.LeftHomologyData
    h₂ : S₂.LeftHomologyData
    ⊢ Eq (CategoryTheory.ShortComplex.leftHomologyMap' φ₁ h₁ h₂) (CategoryTheory.S …
  -/
  rw [h.eq_add_nullHomotopic, leftHomologyMap'_add, leftHomologyMap'_nullHomotopic, add_zero]
  /-
    🎉 no goals
  -/


lemma rightHomologyMap'_congr (h : Homotopy φ₁ φ₂) (h₁ : S₁.RightHomologyData)
    (h₂ : S₂.RightHomologyData) : rightHomologyMap' φ₁ h₁ h₂ = rightHomologyMap' φ₂ h₁ h₂ := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ₁ φ₂ : Quiver.Hom S₁ S₂
    h : CategoryTheory.ShortComplex.Homotopy φ₁ φ₂
    h₁ : S₁.RightHomologyData
    h₂ : S₂.RightHomologyData
    ⊢ Eq (CategoryTheory.ShortComplex.rightHomologyMap' φ₁ h₁ h₂) (CategoryTheory. …
  -/
  rw [h.eq_add_nullHomotopic, rightHomologyMap'_add, rightHomologyMap'_nullHomotopic, add_zero]
  /-
    🎉 no goals
  -/


lemma homologyMap'_congr (h : Homotopy φ₁ φ₂) (h₁ : S₁.HomologyData)
    (h₂ : S₂.HomologyData) : homologyMap' φ₁ h₁ h₂ = homologyMap' φ₂ h₁ h₂ := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    S₁ S₂ : CategoryTheory.ShortComplex C
    φ₁ φ₂ : Quiver.Hom S₁ S₂
    h : CategoryTheory.ShortComplex.Homotopy φ₁ φ₂
    h₁ : S₁.HomologyData
    h₂ : S₂.HomologyData
    ⊢ Eq (CategoryTheory.ShortComplex.homologyMap' φ₁ h₁ h₂) (CategoryTheory.Short …
  -/
  rw [h.eq_add_nullHomotopic, homologyMap'_add, homologyMap'_nullHomotopic, add_zero]
  /-
    🎉 no goals
  -/


lemma leftHomologyMap_congr (h : Homotopy φ₁ φ₂) [S₁.HasLeftHomology] [S₂.HasLeftHomology] :
    leftHomologyMap φ₁ = leftHomologyMap φ₂ :=
  h.leftHomologyMap'_congr _ _


lemma rightHomologyMap_congr (h : Homotopy φ₁ φ₂) [S₁.HasRightHomology] [S₂.HasRightHomology] :
    rightHomologyMap φ₁ = rightHomologyMap φ₂ :=
  h.rightHomologyMap'_congr _ _


lemma homologyMap_congr (h : Homotopy φ₁ φ₂) [S₁.HasHomology] [S₂.HasHomology] :
    homologyMap φ₁ = homologyMap φ₂ :=
  h.homologyMap'_congr _ _


/-- An homotopy equivalence between two short complexes `S₁` and `S₂` consists
of morphisms `hom : S₁ ⟶ S₂` and `inv : S₂ ⟶ S₁` such that both compositions
`hom ≫ inv` and `inv ≫ hom` are homotopic to the identity. -/
@[ext]
structure HomotopyEquiv where
  /-- the forward direction of a homotopy equivalence. -/
  hom : S₁ ⟶ S₂
  /-- the backwards direction of a homotopy equivalence. -/
  inv : S₂ ⟶ S₁
  /-- the composition of the two directions of a homotopy equivalence is
  homotopic to the identity of the source -/
  homotopyHomInvId : Homotopy (hom ≫ inv) (𝟙 S₁)
  /-- the composition of the two directions of a homotopy equivalence is
  homotopic to the identity of the target -/
  homotopyInvHomId : Homotopy (inv ≫ hom) (𝟙 S₂)


/-- The homotopy equivalence from a short complex to itself that is induced
by the identity. -/
@[simps]
def refl (S : ShortComplex C) : HomotopyEquiv S S where
  hom := 𝟙 S
  inv := 𝟙 S
                                        /-
                                          C : Type u_1
                                          inst✝¹ : CategoryTheory.Category.{?u.232558, u_1} C
                                          inst✝ : CategoryTheory.Preadditive C
                                          S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                                          φ₁ φ₂ φ₃ φ₄ : Quiver.Hom S₁ S₂
                                          S : CategoryTheory.ShortComplex C
                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id S)  …
                                        -/
  homotopyHomInvId := Homotopy.ofEq (by simp)
                                        /-
                                          🎉 no goals
                                        -/
                                        /-
                                          C : Type u_1
                                          inst✝¹ : CategoryTheory.Category.{?u.232558, u_1} C
                                          inst✝ : CategoryTheory.Preadditive C
                                          S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                                          φ₁ φ₂ φ₃ φ₄ : Quiver.Hom S₁ S₂
                                          S : CategoryTheory.ShortComplex C
                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id S)  …
                                        -/
  homotopyInvHomId := Homotopy.ofEq (by simp)
                                        /-
                                          🎉 no goals
                                        -/


/-- The inverse of a homotopy equivalence. -/
@[simps]
def symm (e : HomotopyEquiv S₁ S₂) : HomotopyEquiv S₂ S₁ where
  hom := e.inv
  inv := e.hom
  homotopyHomInvId := e.homotopyInvHomId
  homotopyInvHomId := e.homotopyHomInvId


/-- The composition of homotopy equivalences. -/
@[simps]
def trans (e : HomotopyEquiv S₁ S₂) (e' : HomotopyEquiv S₂ S₃) :
    HomotopyEquiv S₁ S₃ where
  hom := e.hom ≫ e'.hom
  inv := e'.inv ≫ e.inv
                                         /-
                                           C : Type u_1
                                           inst✝¹ : CategoryTheory.Category.{?u.234576, u_1} C
                                           inst✝ : CategoryTheory.Preadditive C
                                           S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                                           φ₁ φ₂ φ₃ φ₄ : Quiver.Hom S₁ S₂
                                           e : S₁.HomotopyEquiv S₂
                                           e' : S₂.HomotopyEquiv S₃
                                           ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp e …
                                         -/
  homotopyHomInvId := (Homotopy.ofEq (by simp)).trans
                                         /-
                                           🎉 no goals
                                         -/
    (((e'.homotopyHomInvId.compRight e.inv).compLeft e.hom).trans
                          /-
                            C : Type u_1
                            inst✝¹ : CategoryTheory.Category.{?u.234576, u_1} C
                            inst✝ : CategoryTheory.Preadditive C
                            S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                            φ₁ φ₂ φ₃ φ₄ : Quiver.Hom S₁ S₂
                            e : S₁.HomotopyEquiv S₂
                            e' : S₂.HomotopyEquiv S₃
                            ⊢ Eq (CategoryTheory.CategoryStruct.comp e.hom (CategoryTheory.CategoryStruct. …
                          -/
      ((Homotopy.ofEq (by simp)).trans e.homotopyHomInvId))
                          /-
                            🎉 no goals
                          -/
                                         /-
                                           C : Type u_1
                                           inst✝¹ : CategoryTheory.Category.{?u.234576, u_1} C
                                           inst✝ : CategoryTheory.Preadditive C
                                           S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                                           φ₁ φ₂ φ₃ φ₄ : Quiver.Hom S₁ S₂
                                           e : S₁.HomotopyEquiv S₂
                                           e' : S₂.HomotopyEquiv S₃
                                           ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp e …
                                         -/
  homotopyInvHomId := (Homotopy.ofEq (by simp)).trans
                                         /-
                                           🎉 no goals
                                         -/
    (((e.homotopyInvHomId.compRight e'.hom).compLeft e'.inv).trans
                          /-
                            C : Type u_1
                            inst✝¹ : CategoryTheory.Category.{?u.234576, u_1} C
                            inst✝ : CategoryTheory.Preadditive C
                            S₁ S₂ S₃ : CategoryTheory.ShortComplex C
                            φ₁ φ₂ φ₃ φ₄ : Quiver.Hom S₁ S₂
                            e : S₁.HomotopyEquiv S₂
                            e' : S₂.HomotopyEquiv S₃
                            ⊢ Eq (CategoryTheory.CategoryStruct.comp e'.inv (CategoryTheory.CategoryStruct …
                          -/
      ((Homotopy.ofEq (by simp)).trans e'.homotopyInvHomId))
                          /-
                            🎉 no goals
                          -/


