/-- Given a category `C` and two complex shapes `c₁` and `c₂` on types `I₁` and `I₂`,
the associated type of bicomplexes `HomologicalComplex₂ C c₁ c₂` is
`K : HomologicalComplex (HomologicalComplex C c₂) c₁`. Then, the object in
position `⟨i₁, i₂⟩` can be obtained as `(K.X i₁).X i₂`. -/
abbrev HomologicalComplex₂ :=
  HomologicalComplex (HomologicalComplex C c₂) c₁


/-- The graded object indexed by `I₁ × I₂` induced by a bicomplex. -/
def toGradedObject (K : HomologicalComplex₂ C c₁ c₂) :
    GradedObject (I₁ × I₂) C :=
  fun ⟨i₁, i₂⟩ => (K.X i₁).X i₂


/-- The morphism of graded objects induced by a morphism of bicomplexes. -/
def toGradedObjectMap {K L : HomologicalComplex₂ C c₁ c₂} (φ : K ⟶ L) :
    K.toGradedObject ⟶ L.toGradedObject :=
  fun ⟨i₁, i₂⟩ => (φ.f i₁).f i₂


@[simp]
lemma toGradedObjectMap_apply {K L : HomologicalComplex₂ C c₁ c₂} (φ : K ⟶ L) (i₁ : I₁) (i₂ : I₂) :
    toGradedObjectMap φ ⟨i₁, i₂⟩ = (φ.f i₁).f i₂ := rfl


variable (C c₁ c₂) in
/-- The functor which sends a bicomplex to its associated graded object. -/
@[simps]
def toGradedObjectFunctor : HomologicalComplex₂ C c₁ c₂ ⥤ GradedObject (I₁ × I₂) C where
  obj K := K.toGradedObject
  map φ := toGradedObjectMap φ


instance : (toGradedObjectFunctor C c₁ c₂).Faithful where
  map_injective {_ _ φ₁ φ₂} h := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      I₁ : Type u_2
      I₂ : Type u_3
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      x✝¹ x✝ : HomologicalComplex₂ C c₁ c₂
      φ₁ φ₂ : Quiver.Hom x✝¹ x✝
      h : Eq ((HomologicalComplex₂.toGradedObjectFunctor C c₁ c₂).map φ₁) ((Homologi …
      ⊢ Eq φ₁ φ₂
    -/
    ext i₁ i₂
    /-
      case h.h
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      I₁ : Type u_2
      I₂ : Type u_3
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      x✝¹ x✝ : HomologicalComplex₂ C c₁ c₂
      φ₁ φ₂ : Quiver.Hom x✝¹ x✝
      h : Eq ((HomologicalComplex₂.toGradedObjectFunctor C c₁ c₂).map φ₁) ((Homologi …
      i₁ : I₁
      i₂ : I₂
      ⊢ Eq ((φ₁.f i₁).f i₂) ((φ₂.f i₁).f i₂)
    -/
    exact congr_fun h ⟨i₁, i₂⟩
    /-
      🎉 no goals
    -/


/-- Constructor for bicomplexes taking as inputs a graded object, horizontal differentials
and vertical differentials satisfying suitable relations. -/
@[simps]
def ofGradedObject :
    HomologicalComplex₂ C c₁ c₂ where
  X i₁ :=
    { X := fun i₂ => X ⟨i₁, i₂⟩
      d := fun i₂ i₂' => d₂ i₁ i₂ i₂'
      shape := shape₂ i₁
                      /-
                        C : Type u_1
                        inst✝¹ : CategoryTheory.Category.{?u.4065, u_1} C
                        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                        I₁ : Type u_2
                        I₂ : Type u_3
                        c₁ : ComplexShape I₁
                        c₂ : ComplexShape I₂
                        X : CategoryTheory.GradedObject (Prod I₁ I₂) C
                        d₁ : (i₁ i₁' : I₁) → (i₂ : I₂) → Quiver.Hom (X { fst := i₁, snd := i₂ }) (X {  …
                        d₂ : (i₁ : I₁) → (i₂ i₂' : I₂) → Quiver.Hom (X { fst := i₁, snd := i₂ }) (X {  …
                        shape₁ : ∀ (i₁ i₁' : I₁), Not (c₁.Rel i₁ i₁') → ∀ (i₂ : I₂), Eq (d₁ i₁ i₁' i₂) 0
                        shape₂ : ∀ (i₁ : I₁) (i₂ i₂' : I₂), Not (c₂.Rel i₂ i₂') → Eq (d₂ i₁ i₂ i₂') 0
                        d₁_comp_d₁ : ∀ (i₁ i₁' i₁'' : I₁) (i₂ : I₂), Eq (CategoryTheory.CategoryStruct …
                        d₂_comp_d₂ : ∀ (i₁ : I₁) (i₂ i₂' i₂'' : I₂), Eq (CategoryTheory.CategoryStruct …
                        comm : ∀ (i₁ i₁' : I₁) (i₂ i₂' : I₂), Eq (CategoryTheory.CategoryStruct.comp ( …
                        i₁ : I₁
                        ⊢ ∀ (i j k : I₂), c₂.Rel i j → c₂.Rel j k → Eq (CategoryTheory.CategoryStruct. …
                      -/
      d_comp_d' := by intros; apply d₂_comp_d₂ }
                              /-
                                🎉 no goals
                              -/
  d i₁ i₁' :=
    { f := fun i₂ => d₁ i₁ i₁' i₂
                  /-
                    C : Type u_1
                    inst✝¹ : CategoryTheory.Category.{?u.4065, u_1} C
                    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                    I₁ : Type u_2
                    I₂ : Type u_3
                    c₁ : ComplexShape I₁
                    c₂ : ComplexShape I₂
                    X : CategoryTheory.GradedObject (Prod I₁ I₂) C
                    d₁ : (i₁ i₁' : I₁) → (i₂ : I₂) → Quiver.Hom (X { fst := i₁, snd := i₂ }) (X {  …
                    d₂ : (i₁ : I₁) → (i₂ i₂' : I₂) → Quiver.Hom (X { fst := i₁, snd := i₂ }) (X {  …
                    shape₁ : ∀ (i₁ i₁' : I₁), Not (c₁.Rel i₁ i₁') → ∀ (i₂ : I₂), Eq (d₁ i₁ i₁' i₂) 0
                    shape₂ : ∀ (i₁ : I₁) (i₂ i₂' : I₂), Not (c₂.Rel i₂ i₂') → Eq (d₂ i₁ i₂ i₂') 0
                    d₁_comp_d₁ : ∀ (i₁ i₁' i₁'' : I₁) (i₂ : I₂), Eq (CategoryTheory.CategoryStruct …
                    d₂_comp_d₂ : ∀ (i₁ : I₁) (i₂ i₂' i₂'' : I₂), Eq (CategoryTheory.CategoryStruct …
                    comm : ∀ (i₁ i₁' : I₁) (i₂ i₂' : I₂), Eq (CategoryTheory.CategoryStruct.comp ( …
                    i₁ i₁' : I₁
                    ⊢ ∀ (i j : I₂), c₂.Rel i j → Eq (CategoryTheory.CategoryStruct.comp ((fun i₂ = …
                  -/
      comm' := by intros; apply comm }
                          /-
                            🎉 no goals
                          -/
  shape i₁ i₁' h := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.4065, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      I₁ : Type u_2
      I₂ : Type u_3
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      X : CategoryTheory.GradedObject (Prod I₁ I₂) C
      d₁ : (i₁ i₁' : I₁) → (i₂ : I₂) → Quiver.Hom (X { fst := i₁, snd := i₂ }) (X {  …
      d₂ : (i₁ : I₁) → (i₂ i₂' : I₂) → Quiver.Hom (X { fst := i₁, snd := i₂ }) (X {  …
      shape₁ : ∀ (i₁ i₁' : I₁), Not (c₁.Rel i₁ i₁') → ∀ (i₂ : I₂), Eq (d₁ i₁ i₁' i₂) 0
      shape₂ : ∀ (i₁ : I₁) (i₂ i₂' : I₂), Not (c₂.Rel i₂ i₂') → Eq (d₂ i₁ i₂ i₂') 0
      d₁_comp_d₁ : ∀ (i₁ i₁' i₁'' : I₁) (i₂ : I₂), Eq (CategoryTheory.CategoryStruct …
      d₂_comp_d₂ : ∀ (i₁ : I₁) (i₂ i₂' i₂'' : I₂), Eq (CategoryTheory.CategoryStruct …
      comm : ∀ (i₁ i₁' : I₁) (i₂ i₂' : I₂), Eq (CategoryTheory.CategoryStruct.comp ( …
      i₁ i₁' : I₁
      h : Not (c₁.Rel i₁ i₁')
      ⊢ Eq ((fun i₁ i₁' => { f := fun i₂ => d₁ i₁ i₁' i₂, comm' := ⋯ }) i₁ i₁') 0
    -/
    ext i₂
    /-
      case h
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.4065, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      I₁ : Type u_2
      I₂ : Type u_3
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      X : CategoryTheory.GradedObject (Prod I₁ I₂) C
      d₁ : (i₁ i₁' : I₁) → (i₂ : I₂) → Quiver.Hom (X { fst := i₁, snd := i₂ }) (X {  …
      d₂ : (i₁ : I₁) → (i₂ i₂' : I₂) → Quiver.Hom (X { fst := i₁, snd := i₂ }) (X {  …
      shape₁ : ∀ (i₁ i₁' : I₁), Not (c₁.Rel i₁ i₁') → ∀ (i₂ : I₂), Eq (d₁ i₁ i₁' i₂) 0
      shape₂ : ∀ (i₁ : I₁) (i₂ i₂' : I₂), Not (c₂.Rel i₂ i₂') → Eq (d₂ i₁ i₂ i₂') 0
      d₁_comp_d₁ : ∀ (i₁ i₁' i₁'' : I₁) (i₂ : I₂), Eq (CategoryTheory.CategoryStruct …
      d₂_comp_d₂ : ∀ (i₁ : I₁) (i₂ i₂' i₂'' : I₂), Eq (CategoryTheory.CategoryStruct …
      comm : ∀ (i₁ i₁' : I₁) (i₂ i₂' : I₂), Eq (CategoryTheory.CategoryStruct.comp ( …
      i₁ i₁' : I₁
      h : Not (c₁.Rel i₁ i₁')
      i₂ : I₂
      ⊢ Eq (((fun i₁ i₁' => { f := fun i₂ => d₁ i₁ i₁' i₂, comm' := ⋯ }) i₁ i₁').f i …
    -/
    exact shape₁ i₁ i₁' h i₂
    /-
      🎉 no goals
    -/
                                  /-
                                    C : Type u_1
                                    inst✝¹ : CategoryTheory.Category.{?u.4065, u_1} C
                                    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                                    I₁ : Type u_2
                                    I₂ : Type u_3
                                    c₁ : ComplexShape I₁
                                    c₂ : ComplexShape I₂
                                    X : CategoryTheory.GradedObject (Prod I₁ I₂) C
                                    d₁ : (i₁ i₁' : I₁) → (i₂ : I₂) → Quiver.Hom (X { fst := i₁, snd := i₂ }) (X {  …
                                    d₂ : (i₁ : I₁) → (i₂ i₂' : I₂) → Quiver.Hom (X { fst := i₁, snd := i₂ }) (X {  …
                                    shape₁ : ∀ (i₁ i₁' : I₁), Not (c₁.Rel i₁ i₁') → ∀ (i₂ : I₂), Eq (d₁ i₁ i₁' i₂) 0
                                    shape₂ : ∀ (i₁ : I₁) (i₂ i₂' : I₂), Not (c₂.Rel i₂ i₂') → Eq (d₂ i₁ i₂ i₂') 0
                                    d₁_comp_d₁ : ∀ (i₁ i₁' i₁'' : I₁) (i₂ : I₂), Eq (CategoryTheory.CategoryStruct …
                                    d₂_comp_d₂ : ∀ (i₁ : I₁) (i₂ i₂' i₂'' : I₂), Eq (CategoryTheory.CategoryStruct …
                                    comm : ∀ (i₁ i₁' : I₁) (i₂ i₂' : I₂), Eq (CategoryTheory.CategoryStruct.comp ( …
                                    i₁ i₁' i₁'' : I₁
                                    x✝¹ : c₁.Rel i₁ i₁'
                                    x✝ : c₁.Rel i₁' i₁''
                                    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i₁ i₁' => { f := fun i₂ => d₁ i …
                                  -/
  d_comp_d' i₁ i₁' i₁'' _ _ := by ext i₂; apply d₁_comp_d₁
                                          /-
                                            🎉 no goals
                                          -/


@[simp]
lemma ofGradedObject_toGradedObject :
    (ofGradedObject c₁ c₂ X d₁ d₂ shape₁ shape₂ d₁_comp_d₁ d₂_comp_d₂ comm).toGradedObject = X :=
  rfl


/-- Constructor for a morphism `K ⟶ L` in the category `HomologicalComplex₂ C c₁ c₂` which
takes as inputs a morphism `f : K.toGradedObject ⟶ L.toGradedObject` and
the compatibilites with both horizontal and vertical differentials. -/
@[simps!]
def homMk {K L : HomologicalComplex₂ C c₁ c₂}
    (f : K.toGradedObject ⟶ L.toGradedObject)
    (comm₁ : ∀ i₁ i₁' i₂, c₁.Rel i₁ i₁' →
      f ⟨i₁, i₂⟩ ≫ (L.d i₁ i₁').f i₂ = (K.d i₁ i₁').f i₂ ≫ f ⟨i₁', i₂⟩)
    (comm₂ : ∀ i₁ i₂ i₂', c₂.Rel i₂ i₂' →
      f ⟨i₁, i₂⟩ ≫ (L.X i₁).d i₂ i₂' = (K.X i₁).d i₂ i₂' ≫ f ⟨i₁, i₂'⟩) : K ⟶ L where
  f i₁ :=
    { f := fun i₂ => f ⟨i₁, i₂⟩
      comm' := comm₂ i₁ }
  comm' i₁ i₁' h₁ := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.6616, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      I₁ : Type u_2
      I₂ : Type u_3
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K L : HomologicalComplex₂ C c₁ c₂
      f : Quiver.Hom K.toGradedObject L.toGradedObject
      comm₁ : ∀ (i₁ i₁' : I₁) (i₂ : I₂), c₁.Rel i₁ i₁' → Eq (CategoryTheory.Category …
      comm₂ : ∀ (i₁ : I₁) (i₂ i₂' : I₂), c₂.Rel i₂ i₂' → Eq (CategoryTheory.Category …
      i₁ i₁' : I₁
      h₁ : c₁.Rel i₁ i₁'
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i₁ => { f := fun i₂ => f { fst  …
    -/
    ext i₂
    /-
      case h
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.6616, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      I₁ : Type u_2
      I₂ : Type u_3
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K L : HomologicalComplex₂ C c₁ c₂
      f : Quiver.Hom K.toGradedObject L.toGradedObject
      comm₁ : ∀ (i₁ i₁' : I₁) (i₂ : I₂), c₁.Rel i₁ i₁' → Eq (CategoryTheory.Category …
      comm₂ : ∀ (i₁ : I₁) (i₂ i₂' : I₂), c₂.Rel i₂ i₂' → Eq (CategoryTheory.Category …
      i₁ i₁' : I₁
      h₁ : c₁.Rel i₁ i₁'
      i₂ : I₂
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((fun i₁ => { f := fun i₂ => f { fst …
    -/
    exact comm₁ i₁ i₁' i₂ h₁
    /-
      🎉 no goals
    -/


lemma shape_f (K : HomologicalComplex₂ C c₁ c₂) (i₁ i₁' : I₁) (h : ¬ c₁.Rel i₁ i₁') (i₂ : I₂) :
    (K.d i₁ i₁').f i₂ = 0 := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    I₁ : Type u_2
    I₂ : Type u_3
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    i₁ i₁' : I₁
    h : Not (c₁.Rel i₁ i₁')
    i₂ : I₂
    ⊢ Eq ((K.d i₁ i₁').f i₂) 0
  -/
  rw [K.shape _ _ h, zero_f]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma d_f_comp_d_f (K : HomologicalComplex₂ C c₁ c₂)
    (i₁ i₁' i₁'' : I₁) (i₂ : I₂) :
    (K.d i₁ i₁').f i₂ ≫ (K.d i₁' i₁'').f i₂ = 0 := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    I₁ : Type u_2
    I₂ : Type u_3
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    i₁ i₁' i₁'' : I₁
    i₂ : I₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((K.d i₁ i₁').f i₂) ((K.d i₁' i₁'').f …
  -/
  rw [← comp_f, d_comp_d, zero_f]
  /-
    🎉 no goals
  -/


@[reassoc]
lemma d_comm (K : HomologicalComplex₂ C c₁ c₂) (i₁ i₁' : I₁) (i₂ i₂' : I₂) :
    (K.d i₁ i₁').f i₂ ≫ (K.X i₁').d i₂ i₂' = (K.X i₁).d i₂ i₂' ≫ (K.d i₁ i₁').f i₂' := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_4, u_1} C
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    I₁ : Type u_2
    I₂ : Type u_3
    c₁ : ComplexShape I₁
    c₂ : ComplexShape I₂
    K : HomologicalComplex₂ C c₁ c₂
    i₁ i₁' : I₁
    i₂ i₂' : I₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((K.d i₁ i₁').f i₂) ((K.X i₁').d i₂ i …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma comm_f {K L : HomologicalComplex₂ C c₁ c₂} (f : K ⟶ L) (i₁ i₁' : I₁) (i₂ : I₂) :
    (f.f i₁).f i₂ ≫ (L.d i₁ i₁').f i₂ = (K.d i₁ i₁').f i₂ ≫ (f.f i₁').f i₂ :=
  congr_hom (f.comm i₁ i₁') i₂


/-- Flip a complex of complexes over the diagonal,
exchanging the horizontal and vertical directions.
-/
@[simps]
def flip (K : HomologicalComplex₂ C c₁ c₂) : HomologicalComplex₂ C c₂ c₁ where
  X i :=
    { X := fun j => (K.X j).X i
      d := fun j j' => (K.d j j').f i
      shape := fun _ _ w => K.shape_f _ _ w i }
  d i i' := { f := fun j => (K.X j).d i i' }
  shape i i' w := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.12445, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      I₁ : Type u_2
      I₂ : Type u_3
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K : HomologicalComplex₂ C c₁ c₂
      i i' : I₂
      w : Not (c₂.Rel i i')
      ⊢ Eq ((fun i i' => { f := fun j => (K.X j).d i i', comm' := ⋯ }) i i') 0
    -/
    ext j
    /-
      case h
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.12445, u_1} C
      inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
      I₁ : Type u_2
      I₂ : Type u_3
      c₁ : ComplexShape I₁
      c₂ : ComplexShape I₂
      K : HomologicalComplex₂ C c₁ c₂
      i i' : I₂
      w : Not (c₂.Rel i i')
      j : I₁
      ⊢ Eq (((fun i i' => { f := fun j => (K.X j).d i i', comm' := ⋯ }) i i').f j) ( …
    -/
    exact (K.X j).shape i i' w
    /-
      🎉 no goals
    -/


@[simp]
lemma flip_flip (K : HomologicalComplex₂ C c₁ c₂) : K.flip.flip = K := rfl


/-- Flipping a complex of complexes over the diagonal, as a functor. -/
@[simps]
def flipFunctor :
    HomologicalComplex₂ C c₁ c₂ ⥤ HomologicalComplex₂ C c₂ c₁ where
  obj K := K.flip
  map {K L} f :=
    { f := fun i =>
        { f := fun j => (f.f j).f i
                      /-
                        C : Type u_1
                        inst✝¹ : CategoryTheory.Category.{?u.17527, u_1} C
                        inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                        I₁ : Type u_2
                        I₂ : Type u_3
                        c₁ : ComplexShape I₁
                        c₂ : ComplexShape I₂
                        K L : HomologicalComplex₂ C c₁ c₂
                        f : Quiver.Hom K L
                        i : I₂
                        ⊢ ∀ (i_1 j : I₁), c₁.Rel i_1 j → Eq (CategoryTheory.CategoryStruct.comp ((fun  …
                      -/
          comm' := by intros; simp }
                              /-
                                🎉 no goals
                              -/
                  /-
                    C : Type u_1
                    inst✝¹ : CategoryTheory.Category.{?u.17527, u_1} C
                    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                    I₁ : Type u_2
                    I₂ : Type u_3
                    c₁ : ComplexShape I₁
                    c₂ : ComplexShape I₂
                    K L : HomologicalComplex₂ C c₁ c₂
                    f : Quiver.Hom K L
                    ⊢ ∀ (i j : I₂), c₂.Rel i j → Eq (CategoryTheory.CategoryStruct.comp ((fun i => …
                  -/
      comm' := by intros; ext; simp }
                               /-
                                 🎉 no goals
                               -/


/-- Auxiliary definition for `HomologicalComplex₂.flipEquivalence`. -/
@[simps!]
def flipEquivalenceUnitIso :
    𝟭 (HomologicalComplex₂ C c₁ c₂) ≅ flipFunctor C c₁ c₂ ⋙ flipFunctor C c₂ c₁ :=
  NatIso.ofComponents (fun K => HomologicalComplex.Hom.isoOfComponents (fun i₁ =>
    HomologicalComplex.Hom.isoOfComponents (fun _ => Iso.refl _)
        /-
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.21599, u_1} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          I₁ : Type u_2
          I₂ : Type u_3
          c₁ : ComplexShape I₁
          c₂ : ComplexShape I₂
          K : HomologicalComplex₂ C c₁ c₂
          i₁ : I₁
          ⊢ ∀ (i j : I₂), c₂.Rel i j → Eq (CategoryTheory.CategoryStruct.comp ((fun x => …
        -/
        /-
          🎉 no goals
        -/
                        /-
                          🎉 no goals
                        -/
    (by aesop_cat)) (by aesop_cat)) (by aesop_cat)
                                        /-
                                          🎉 no goals
                                        -/


/-- Auxiliary definition for `HomologicalComplex₂.flipEquivalence`. -/
@[simps!]
def flipEquivalenceCounitIso :
    flipFunctor C c₂ c₁ ⋙ flipFunctor C c₁ c₂ ≅ 𝟭 (HomologicalComplex₂ C c₂ c₁) :=
  NatIso.ofComponents (fun K => HomologicalComplex.Hom.isoOfComponents (fun i₂ =>
    HomologicalComplex.Hom.isoOfComponents (fun _ => Iso.refl _)
        /-
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.33796, u_1} C
          inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
          I₁ : Type u_2
          I₂ : Type u_3
          c₁ : ComplexShape I₁
          c₂ : ComplexShape I₂
          K : HomologicalComplex₂ C c₂ c₁
          i₂ : I₂
          ⊢ ∀ (i j : I₁), c₁.Rel i j → Eq (CategoryTheory.CategoryStruct.comp ((fun x => …
        -/
        /-
          🎉 no goals
        -/
                        /-
                          🎉 no goals
                        -/
    (by aesop_cat)) (by aesop_cat)) (by aesop_cat)
                                        /-
                                          🎉 no goals
                                        -/


/-- Flipping a complex of complexes over the diagonal, as an equivalence of categories. -/
@[simps]
def flipEquivalence :
    HomologicalComplex₂ C c₁ c₂ ≌ HomologicalComplex₂ C c₂ c₁ where
  functor := flipFunctor C c₁ c₂
  inverse := flipFunctor C c₂ c₁
  unitIso := flipEquivalenceUnitIso C c₁ c₂
  counitIso := flipEquivalenceCounitIso C c₁ c₂


/-- The obvious isomorphism `(K.X x₁).X x₂ ≅ (K.X y₁).X y₂` when `x₁ = y₁` and `x₂ = y₂`. -/
def XXIsoOfEq {x₁ y₁ : I₁} (h₁ : x₁ = y₁) {x₂ y₂ : I₂} (h₂ : x₂ = y₂) :
    (K.X x₁).X x₂ ≅ (K.X y₁).X y₂ :=
              /-
                C : Type u_1
                inst✝¹ : CategoryTheory.Category.{?u.51791, u_1} C
                inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
                I₁ : Type u_2
                I₂ : Type u_3
                c₁ : ComplexShape I₁
                c₂ : ComplexShape I₂
                K : HomologicalComplex₂ C c₁ c₂
                x₁ y₁ : I₁
                h₁ : Eq x₁ y₁
                x₂ y₂ : I₂
                h₂ : Eq x₂ y₂
                ⊢ Eq ((K.X x₁).X x₂) ((K.X y₁).X y₂)
              -/
  eqToIso (by subst h₁ h₂; rfl)
                           /-
                             🎉 no goals
                           -/


@[simp]
lemma XXIsoOfEq_rfl (i₁ : I₁) (i₂ : I₂) :
    K.XXIsoOfEq _ _ _ (rfl : i₁ = i₁) (rfl : i₂ = i₂) = Iso.refl _ := rfl



