/-- Given `T : MorphismProperty C`, this is the class of morphisms that have the
left lifting property (llp) with respect to `T`. -/
def llp : MorphismProperty C := fun _ _ f ↦
  ∀ ⦃X Y : C⦄ (g : X ⟶ Y) (_ : T g), HasLiftingProperty f g


/-- Given `T : MorphismProperty C`, this is the class of morphisms that have the
right lifting property (rlp) with respect to `T`. -/
def rlp : MorphismProperty C := fun _ _ f ↦
  ∀ ⦃X Y : C⦄ (g : X ⟶ Y) (_ : T g), HasLiftingProperty g f


lemma llp_isStableUnderRetracts : T.llp.IsStableUnderRetracts where
  of_retract h hg _ _ f hf :=
    letI := hg _ hf
    h.leftLiftingProperty f


lemma rlp_isStableUnderRetracts : T.rlp.IsStableUnderRetracts where
  of_retract h hf _ _ g hg :=
    letI := hf _ hg
    h.rightLiftingProperty g


instance llp_isStableUnderCobaseChange : T.llp.IsStableUnderCobaseChange where
  of_isPushout h hf _ _ g' hg' :=
    letI := hf _ hg'
    h.hasLiftingProperty g'


open IsPullback in
instance rlp_isStableUnderBaseChange : T.rlp.IsStableUnderBaseChange where
  of_isPullback h hf _ _ f' hf' :=
    letI := hf _ hf'
    h.hasLiftingProperty f'


instance llp_isMultiplicative : T.llp.IsMultiplicative where
                          /-
                            C : Type u
                            inst✝ : CategoryTheory.Category.{v, u} C
                            T : CategoryTheory.MorphismProperty C
                            X x✝¹ x✝ : C
                            p : Quiver.Hom x✝¹ x✝
                            hp : T p
                            ⊢ CategoryTheory.HasLiftingProperty (CategoryTheory.CategoryStruct.id X) p
                          -/
  id_mem X _ _ p hp := by infer_instance
                          /-
                            🎉 no goals
                          -/
  comp_mem i j hi hj _ _ p hp := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      T : CategoryTheory.MorphismProperty C
      X✝ Y✝ Z✝ : C
      i : Quiver.Hom X✝ Y✝
      j : Quiver.Hom Y✝ Z✝
      hi : T.llp i
      hj : T.llp j
      x✝¹ x✝ : C
      p : Quiver.Hom x✝¹ x✝
      hp : T p
      ⊢ CategoryTheory.HasLiftingProperty (CategoryTheory.CategoryStruct.comp i j) p
    -/
    have := hi _ hp
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      T : CategoryTheory.MorphismProperty C
      X✝ Y✝ Z✝ : C
      i : Quiver.Hom X✝ Y✝
      j : Quiver.Hom Y✝ Z✝
      hi : T.llp i
      hj : T.llp j
      x✝¹ x✝ : C
      p : Quiver.Hom x✝¹ x✝
      hp : T p
      this : CategoryTheory.HasLiftingProperty i p
      ⊢ CategoryTheory.HasLiftingProperty (CategoryTheory.CategoryStruct.comp i j) p
    -/
    have := hj _ hp
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      T : CategoryTheory.MorphismProperty C
      X✝ Y✝ Z✝ : C
      i : Quiver.Hom X✝ Y✝
      j : Quiver.Hom Y✝ Z✝
      hi : T.llp i
      hj : T.llp j
      x✝¹ x✝ : C
      p : Quiver.Hom x✝¹ x✝
      hp : T p
      this✝ : CategoryTheory.HasLiftingProperty i p
      this : CategoryTheory.HasLiftingProperty j p
      ⊢ CategoryTheory.HasLiftingProperty (CategoryTheory.CategoryStruct.comp i j) p
    -/
    infer_instance
    /-
      🎉 no goals
    -/


instance rlp_isMultiplicative : T.rlp.IsMultiplicative where
                          /-
                            C : Type u
                            inst✝ : CategoryTheory.Category.{v, u} C
                            T : CategoryTheory.MorphismProperty C
                            X x✝¹ x✝ : C
                            p : Quiver.Hom x✝¹ x✝
                            hp : T p
                            ⊢ CategoryTheory.HasLiftingProperty p (CategoryTheory.CategoryStruct.id X)
                          -/
  id_mem X _ _ p hp := by infer_instance
                          /-
                            🎉 no goals
                          -/
  comp_mem i j hi hj _ _ p hp := by
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      T : CategoryTheory.MorphismProperty C
      X✝ Y✝ Z✝ : C
      i : Quiver.Hom X✝ Y✝
      j : Quiver.Hom Y✝ Z✝
      hi : T.rlp i
      hj : T.rlp j
      x✝¹ x✝ : C
      p : Quiver.Hom x✝¹ x✝
      hp : T p
      ⊢ CategoryTheory.HasLiftingProperty p (CategoryTheory.CategoryStruct.comp i j)
    -/
    have := hi _ hp
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      T : CategoryTheory.MorphismProperty C
      X✝ Y✝ Z✝ : C
      i : Quiver.Hom X✝ Y✝
      j : Quiver.Hom Y✝ Z✝
      hi : T.rlp i
      hj : T.rlp j
      x✝¹ x✝ : C
      p : Quiver.Hom x✝¹ x✝
      hp : T p
      this : CategoryTheory.HasLiftingProperty p i
      ⊢ CategoryTheory.HasLiftingProperty p (CategoryTheory.CategoryStruct.comp i j)
    -/
    have := hj _ hp
    /-
      C : Type u
      inst✝ : CategoryTheory.Category.{v, u} C
      T : CategoryTheory.MorphismProperty C
      X✝ Y✝ Z✝ : C
      i : Quiver.Hom X✝ Y✝
      j : Quiver.Hom Y✝ Z✝
      hi : T.rlp i
      hj : T.rlp j
      x✝¹ x✝ : C
      p : Quiver.Hom x✝¹ x✝
      hp : T p
      this✝ : CategoryTheory.HasLiftingProperty p i
      this : CategoryTheory.HasLiftingProperty p j
      ⊢ CategoryTheory.HasLiftingProperty p (CategoryTheory.CategoryStruct.comp i j)
    -/
    infer_instance
    /-
      🎉 no goals
    -/


lemma llp_IsStableUnderCoproductsOfShape (J : Type*) :
    T.llp.IsStableUnderCoproductsOfShape J := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    T : CategoryTheory.MorphismProperty C
    J : Type u_1
    ⊢ T.llp.IsStableUnderCoproductsOfShape J
  -/
  apply IsStableUnderCoproductsOfShape.mk
  /-
    case hW
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    T : CategoryTheory.MorphismProperty C
    J : Type u_1
    ⊢ ∀ (X₁ X₂ : J → C) [inst : CategoryTheory.Limits.HasCoproduct X₁] [inst_1 : C …
  -/
  intro A B _ _ f hf X Y p hp
  /-
    case hW
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    T : CategoryTheory.MorphismProperty C
    J : Type u_1
    A B : J → C
    inst✝¹ : CategoryTheory.Limits.HasCoproduct A
    inst✝ : CategoryTheory.Limits.HasCoproduct B
    f : (j : J) → Quiver.Hom (A j) (B j)
    hf : ∀ (j : J), T.llp (f j)
    X Y : C
    p : Quiver.Hom X Y
    hp : T p
    ⊢ CategoryTheory.HasLiftingProperty (CategoryTheory.Limits.Sigma.map f) p
  -/
  have := fun j ↦ hf j _ hp
  /-
    case hW
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    T : CategoryTheory.MorphismProperty C
    J : Type u_1
    A B : J → C
    inst✝¹ : CategoryTheory.Limits.HasCoproduct A
    inst✝ : CategoryTheory.Limits.HasCoproduct B
    f : (j : J) → Quiver.Hom (A j) (B j)
    hf : ∀ (j : J), T.llp (f j)
    X Y : C
    p : Quiver.Hom X Y
    hp : T p
    this : ∀ (j : J), CategoryTheory.HasLiftingProperty (f j) p
    ⊢ CategoryTheory.HasLiftingProperty (CategoryTheory.Limits.Sigma.map f) p
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma rlp_IsStableUnderProductsOfShape (J : Type*) :
    T.rlp.IsStableUnderProductsOfShape J := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    T : CategoryTheory.MorphismProperty C
    J : Type u_1
    ⊢ T.rlp.IsStableUnderProductsOfShape J
  -/
  apply IsStableUnderProductsOfShape.mk
  /-
    case hW
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    T : CategoryTheory.MorphismProperty C
    J : Type u_1
    ⊢ ∀ (X₁ X₂ : J → C) [inst : CategoryTheory.Limits.HasProduct X₁] [inst_1 : Cat …
  -/
  intro A B _ _ f hf X Y p hp
  /-
    case hW
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    T : CategoryTheory.MorphismProperty C
    J : Type u_1
    A B : J → C
    inst✝¹ : CategoryTheory.Limits.HasProduct A
    inst✝ : CategoryTheory.Limits.HasProduct B
    f : (j : J) → Quiver.Hom (A j) (B j)
    hf : ∀ (j : J), T.rlp (f j)
    X Y : C
    p : Quiver.Hom X Y
    hp : T p
    ⊢ CategoryTheory.HasLiftingProperty p (CategoryTheory.Limits.Pi.map f)
  -/
  have := fun j ↦ hf j _ hp
  /-
    case hW
    C : Type u
    inst✝² : CategoryTheory.Category.{v, u} C
    T : CategoryTheory.MorphismProperty C
    J : Type u_1
    A B : J → C
    inst✝¹ : CategoryTheory.Limits.HasProduct A
    inst✝ : CategoryTheory.Limits.HasProduct B
    f : (j : J) → Quiver.Hom (A j) (B j)
    hf : ∀ (j : J), T.rlp (f j)
    X Y : C
    p : Quiver.Hom X Y
    hp : T p
    this : ∀ (j : J), CategoryTheory.HasLiftingProperty p (f j)
    ⊢ CategoryTheory.HasLiftingProperty p (CategoryTheory.Limits.Pi.map f)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


