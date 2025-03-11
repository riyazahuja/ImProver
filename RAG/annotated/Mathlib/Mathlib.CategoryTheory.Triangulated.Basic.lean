/-- A triangle in `C` is a sextuple `(X,Y,Z,f,g,h)` where `X,Y,Z` are objects of `C`,
and `f : X ⟶ Y`, `g : Y ⟶ Z`, `h : Z ⟶ X⟦1⟧` are morphisms in `C`.
See <https://stacks.math.columbia.edu/tag/0144>.
-/
structure Triangle where mk' ::
  /-- the first object of a triangle -/
  obj₁ : C
  /-- the second object of a triangle -/
  obj₂ : C
  /-- the third object of a triangle -/
  obj₃ : C
  /-- the first morphism of a triangle -/
  mor₁ : obj₁ ⟶ obj₂
  /-- the second morphism of a triangle -/
  mor₂ : obj₂ ⟶ obj₃
  /-- the third morphism of a triangle -/
  mor₃ : obj₃ ⟶ obj₁⟦(1 : ℤ)⟧


/-- A triangle `(X,Y,Z,f,g,h)` in `C` is defined by the morphisms `f : X ⟶ Y`, `g : Y ⟶ Z`
and `h : Z ⟶ X⟦1⟧`.
-/
@[simps]
def Triangle.mk {X Y Z : C} (f : X ⟶ Y) (g : Y ⟶ Z) (h : Z ⟶ X⟦(1 : ℤ)⟧) : Triangle C where
  obj₁ := X
  obj₂ := Y
  obj₃ := Z
  mor₁ := f
  mor₂ := g
  mor₃ := h


instance : Inhabited (Triangle C) :=
  ⟨⟨0, 0, 0, 0, 0, 0⟩⟩


/-- For each object in `C`, there is a triangle of the form `(X,X,0,𝟙 X,0,0)`
-/
@[simps!]
def contractibleTriangle (X : C) : Triangle C :=
  Triangle.mk (𝟙 X) (0 : X ⟶ 0) 0


/-- A morphism of triangles `(X,Y,Z,f,g,h) ⟶ (X',Y',Z',f',g',h')` in `C` is a triple of morphisms
`a : X ⟶ X'`, `b : Y ⟶ Y'`, `c : Z ⟶ Z'` such that
`a ≫ f' = f ≫ b`, `b ≫ g' = g ≫ c`, and `a⟦1⟧' ≫ h = h' ≫ c`.
In other words, we have a commutative diagram:
```
     f      g      h
  X  ───> Y  ───> Z  ───> X⟦1⟧
  │       │       │        │
  │a      │b      │c       │a⟦1⟧'
  V       V       V        V
  X' ───> Y' ───> Z' ───> X'⟦1⟧
     f'     g'     h'
```
See <https://stacks.math.columbia.edu/tag/0144>.
-/
@[ext]
structure TriangleMorphism (T₁ : Triangle C) (T₂ : Triangle C) where
  /-- the first morphism in a triangle morphism -/
  hom₁ : T₁.obj₁ ⟶ T₂.obj₁
  /-- the second morphism in a triangle morphism -/
  hom₂ : T₁.obj₂ ⟶ T₂.obj₂
  /-- the third morphism in a triangle morphism -/
  hom₃ : T₁.obj₃ ⟶ T₂.obj₃
  /-- the first commutative square of a triangle morphism -/
  comm₁ : T₁.mor₁ ≫ hom₂ = hom₁ ≫ T₂.mor₁ := by aesop_cat
  /-- the second commutative square of a triangle morphism -/
  comm₂ : T₁.mor₂ ≫ hom₃ = hom₂ ≫ T₂.mor₂ := by aesop_cat
  /-- the third commutative square of a triangle morphism -/
  comm₃ : T₁.mor₃ ≫ hom₁⟦1⟧' = hom₃ ≫ T₂.mor₃ := by aesop_cat


attribute [reassoc (attr := simp)] TriangleMorphism.comm₁ TriangleMorphism.comm₂
  TriangleMorphism.comm₃


/-- The identity triangle morphism.
-/
@[simps]
def triangleMorphismId (T : Triangle C) : TriangleMorphism T T where
  hom₁ := 𝟙 T.obj₁
  hom₂ := 𝟙 T.obj₂
  hom₃ := 𝟙 T.obj₃


instance (T : Triangle C) : Inhabited (TriangleMorphism T T) :=
  ⟨triangleMorphismId T⟩


/-- Composition of triangle morphisms gives a triangle morphism.
-/
@[simps]
def TriangleMorphism.comp (f : TriangleMorphism T₁ T₂) (g : TriangleMorphism T₂ T₃) :
    TriangleMorphism T₁ T₃ where
  hom₁ := f.hom₁ ≫ g.hom₁
  hom₂ := f.hom₂ ≫ g.hom₂
  hom₃ := f.hom₃ ≫ g.hom₃


/-- Triangles with triangle morphisms form a category.
-/
@[simps]
instance triangleCategory : Category (Triangle C) where
  Hom A B := TriangleMorphism A B
  id A := triangleMorphismId A
  comp f g := f.comp g


@[ext]
lemma Triangle.hom_ext {A B : Triangle C} (f g : A ⟶ B)
    (h₁ : f.hom₁ = g.hom₁) (h₂ : f.hom₂ = g.hom₂) (h₃ : f.hom₃ = g.hom₃) : f = g :=
  TriangleMorphism.ext h₁ h₂ h₃


@[simp]
lemma id_hom₁ (A : Triangle C) : TriangleMorphism.hom₁ (𝟙 A) = 𝟙 _ := rfl

@[simp]
lemma id_hom₂ (A : Triangle C) : TriangleMorphism.hom₂ (𝟙 A) = 𝟙 _ := rfl

@[simp]
lemma id_hom₃ (A : Triangle C) : TriangleMorphism.hom₃ (𝟙 A) = 𝟙 _ := rfl


@[simp, reassoc]
lemma comp_hom₁ {X Y Z : Triangle C} (f : X ⟶ Y) (g : Y ⟶ Z) :
    (f ≫ g).hom₁ = f.hom₁ ≫ g.hom₁ := rfl

@[simp, reassoc]
lemma comp_hom₂ {X Y Z : Triangle C} (f : X ⟶ Y) (g : Y ⟶ Z) :
    (f ≫ g).hom₂ = f.hom₂ ≫ g.hom₂ := rfl

@[simp, reassoc]
lemma comp_hom₃ {X Y Z : Triangle C} (f : X ⟶ Y) (g : Y ⟶ Z) :
    (f ≫ g).hom₃ = f.hom₃ ≫ g.hom₃ := rfl


@[simps]
def Triangle.homMk (A B : Triangle C)
    (hom₁ : A.obj₁ ⟶ B.obj₁) (hom₂ : A.obj₂ ⟶ B.obj₂) (hom₃ : A.obj₃ ⟶ B.obj₃)
    (comm₁ : A.mor₁ ≫ hom₂ = hom₁ ≫ B.mor₁ := by aesop_cat)
    (comm₂ : A.mor₂ ≫ hom₃ = hom₂ ≫ B.mor₂ := by aesop_cat)
    (comm₃ : A.mor₃ ≫ hom₁⟦1⟧' = hom₃ ≫ B.mor₃ := by aesop_cat) :
    A ⟶ B where
  hom₁ := hom₁
  hom₂ := hom₂
  hom₃ := hom₃
  comm₁ := comm₁
  comm₂ := comm₂
  comm₃ := comm₃


@[simps]
def Triangle.isoMk (A B : Triangle C)
    (iso₁ : A.obj₁ ≅ B.obj₁) (iso₂ : A.obj₂ ≅ B.obj₂) (iso₃ : A.obj₃ ≅ B.obj₃)
    (comm₁ : A.mor₁ ≫ iso₂.hom = iso₁.hom ≫ B.mor₁ := by aesop_cat)
    (comm₂ : A.mor₂ ≫ iso₃.hom = iso₂.hom ≫ B.mor₂ := by aesop_cat)
    (comm₃ : A.mor₃ ≫ iso₁.hom⟦1⟧' = iso₃.hom ≫ B.mor₃ := by aesop_cat) : A ≅ B where
  hom := Triangle.homMk _ _ iso₁.hom iso₂.hom iso₃.hom comm₁ comm₂ comm₃
  inv := Triangle.homMk _ _ iso₁.inv iso₂.inv iso₃.inv
    (by simp only [← cancel_mono iso₂.hom, assoc, Iso.inv_hom_id, comp_id,
      comm₁, Iso.inv_hom_id_assoc])
    (by simp only [← cancel_mono iso₃.hom, assoc, Iso.inv_hom_id, comp_id,
      comm₂, Iso.inv_hom_id_assoc])
    (by simp only [← cancel_mono (iso₁.hom⟦(1 : ℤ)⟧'), Category.assoc, comm₃,
      Iso.inv_hom_id_assoc, ← Functor.map_comp, Iso.inv_hom_id,
      Functor.map_id, Category.comp_id])


lemma Triangle.isIso_of_isIsos {A B : Triangle C} (f : A ⟶ B)
    (h₁ : IsIso f.hom₁) (h₂ : IsIso f.hom₂) (h₃ : IsIso f.hom₃) : IsIso f := by
  let e := Triangle.isoMk A B (asIso f.hom₁) (asIso f.hom₂) (asIso f.hom₃)
    (by simp) (by simp) (by simp)
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    inst✝ : CategoryTheory.HasShift C Int
    A B : CategoryTheory.Pretriangulated.Triangle C
    f : Quiver.Hom A B
    h₁ : CategoryTheory.IsIso f.hom₁
    h₂ : CategoryTheory.IsIso f.hom₂
    h₃ : CategoryTheory.IsIso f.hom₃
    e : CategoryTheory.Iso A B := A.isoMk B (CategoryTheory.asIso f.hom₁) (Categor …
    ⊢ CategoryTheory.IsIso f
  -/
  exact (inferInstance : IsIso e.hom)
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
lemma _root_.CategoryTheory.Iso.hom_inv_id_triangle_hom₁ {A B : Triangle C} (e : A ≅ B) :
                                        /-
                                          C : Type u
                                          inst✝¹ : CategoryTheory.Category.{v, u} C
                                          inst✝ : CategoryTheory.HasShift C Int
                                          A B : CategoryTheory.Pretriangulated.Triangle C
                                          e : CategoryTheory.Iso A B
                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp e.hom.hom₁ e.inv.hom₁) (CategoryTheor …
                                        -/
    e.hom.hom₁ ≫ e.inv.hom₁ = 𝟙 _ := by rw [← comp_hom₁, e.hom_inv_id, id_hom₁]
                                        /-
                                          🎉 no goals
                                        -/

@[reassoc (attr := simp)]
lemma _root_.CategoryTheory.Iso.hom_inv_id_triangle_hom₂ {A B : Triangle C} (e : A ≅ B) :
                                        /-
                                          C : Type u
                                          inst✝¹ : CategoryTheory.Category.{v, u} C
                                          inst✝ : CategoryTheory.HasShift C Int
                                          A B : CategoryTheory.Pretriangulated.Triangle C
                                          e : CategoryTheory.Iso A B
                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp e.hom.hom₂ e.inv.hom₂) (CategoryTheor …
                                        -/
    e.hom.hom₂ ≫ e.inv.hom₂ = 𝟙 _ := by rw [← comp_hom₂, e.hom_inv_id, id_hom₂]
                                        /-
                                          🎉 no goals
                                        -/

@[reassoc (attr := simp)]
lemma _root_.CategoryTheory.Iso.hom_inv_id_triangle_hom₃ {A B : Triangle C} (e : A ≅ B) :
                                        /-
                                          C : Type u
                                          inst✝¹ : CategoryTheory.Category.{v, u} C
                                          inst✝ : CategoryTheory.HasShift C Int
                                          A B : CategoryTheory.Pretriangulated.Triangle C
                                          e : CategoryTheory.Iso A B
                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp e.hom.hom₃ e.inv.hom₃) (CategoryTheor …
                                        -/
    e.hom.hom₃ ≫ e.inv.hom₃ = 𝟙 _ := by rw [← comp_hom₃, e.hom_inv_id, id_hom₃]
                                        /-
                                          🎉 no goals
                                        -/


@[reassoc (attr := simp)]
lemma _root_.CategoryTheory.Iso.inv_hom_id_triangle_hom₁ {A B : Triangle C} (e : A ≅ B) :
                                        /-
                                          C : Type u
                                          inst✝¹ : CategoryTheory.Category.{v, u} C
                                          inst✝ : CategoryTheory.HasShift C Int
                                          A B : CategoryTheory.Pretriangulated.Triangle C
                                          e : CategoryTheory.Iso A B
                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp e.inv.hom₁ e.hom.hom₁) (CategoryTheor …
                                        -/
    e.inv.hom₁ ≫ e.hom.hom₁ = 𝟙 _ := by rw [← comp_hom₁, e.inv_hom_id, id_hom₁]
                                        /-
                                          🎉 no goals
                                        -/

@[reassoc (attr := simp)]
lemma _root_.CategoryTheory.Iso.inv_hom_id_triangle_hom₂ {A B : Triangle C} (e : A ≅ B) :
                                        /-
                                          C : Type u
                                          inst✝¹ : CategoryTheory.Category.{v, u} C
                                          inst✝ : CategoryTheory.HasShift C Int
                                          A B : CategoryTheory.Pretriangulated.Triangle C
                                          e : CategoryTheory.Iso A B
                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp e.inv.hom₂ e.hom.hom₂) (CategoryTheor …
                                        -/
    e.inv.hom₂ ≫ e.hom.hom₂ = 𝟙 _ := by rw [← comp_hom₂, e.inv_hom_id, id_hom₂]
                                        /-
                                          🎉 no goals
                                        -/

@[reassoc (attr := simp)]
lemma _root_.CategoryTheory.Iso.inv_hom_id_triangle_hom₃ {A B : Triangle C} (e : A ≅ B) :
                                        /-
                                          C : Type u
                                          inst✝¹ : CategoryTheory.Category.{v, u} C
                                          inst✝ : CategoryTheory.HasShift C Int
                                          A B : CategoryTheory.Pretriangulated.Triangle C
                                          e : CategoryTheory.Iso A B
                                          ⊢ Eq (CategoryTheory.CategoryStruct.comp e.inv.hom₃ e.hom.hom₃) (CategoryTheor …
                                        -/
    e.inv.hom₃ ≫ e.hom.hom₃ = 𝟙 _ := by rw [← comp_hom₃, e.inv_hom_id, id_hom₃]
                                        /-
                                          🎉 no goals
                                        -/


lemma Triangle.eqToHom_hom₁ {A B : Triangle C} (h : A = B) :
                                   /-
                                     C : Type u
                                     inst✝¹ : CategoryTheory.Category.{v, u} C
                                     inst✝ : CategoryTheory.HasShift C Int
                                     T₁ T₂ T₃ A B : CategoryTheory.Pretriangulated.Triangle C
                                     h : Eq A B
                                     ⊢ Eq A.obj₁ B.obj₁
                                   -/
                                            /-
                                              🎉 no goals
                                            -/
    (eqToHom h).hom₁ = eqToHom (by subst h; rfl) := by subst h; rfl
                                                                /-
                                                                  🎉 no goals
                                                                -/

lemma Triangle.eqToHom_hom₂ {A B : Triangle C} (h : A = B) :
                                   /-
                                     C : Type u
                                     inst✝¹ : CategoryTheory.Category.{v, u} C
                                     inst✝ : CategoryTheory.HasShift C Int
                                     T₁ T₂ T₃ A B : CategoryTheory.Pretriangulated.Triangle C
                                     h : Eq A B
                                     ⊢ Eq A.obj₂ B.obj₂
                                   -/
                                            /-
                                              🎉 no goals
                                            -/
    (eqToHom h).hom₂ = eqToHom (by subst h; rfl) := by subst h; rfl
                                                                /-
                                                                  🎉 no goals
                                                                -/

lemma Triangle.eqToHom_hom₃ {A B : Triangle C} (h : A = B) :
                                   /-
                                     C : Type u
                                     inst✝¹ : CategoryTheory.Category.{v, u} C
                                     inst✝ : CategoryTheory.HasShift C Int
                                     T₁ T₂ T₃ A B : CategoryTheory.Pretriangulated.Triangle C
                                     h : Eq A B
                                     ⊢ Eq A.obj₃ B.obj₃
                                   -/
                                            /-
                                              🎉 no goals
                                            -/
    (eqToHom h).hom₃ = eqToHom (by subst h; rfl) := by subst h; rfl
                                                                /-
                                                                  🎉 no goals
                                                                -/


/-- The obvious triangle `X₁ ⟶ X₁ ⊞ X₂ ⟶ X₂ ⟶ X₁⟦1⟧`. -/
@[simps!]
def binaryBiproductTriangle (X₁ X₂ : C) [HasZeroMorphisms C] [HasBinaryBiproduct X₁ X₂] :
    Triangle C :=
  Triangle.mk biprod.inl (Limits.biprod.snd : X₁ ⊞ X₂ ⟶ _) 0


/-- The obvious triangle `X₁ ⟶ X₁ ⨯ X₂ ⟶ X₂ ⟶ X₁⟦1⟧`. -/
@[simps!]
def binaryProductTriangle (X₁ X₂ : C) [HasZeroMorphisms C] [HasBinaryProduct X₁ X₂] :
    Triangle C :=
  Triangle.mk ((Limits.prod.lift (𝟙 X₁) 0)) (Limits.prod.snd : X₁ ⨯ X₂ ⟶ _) 0


/-- The canonical isomorphism of triangles
`binaryProductTriangle X₁ X₂ ≅ binaryBiproductTriangle X₁ X₂`. -/
@[simps!]
def binaryProductTriangleIsoBinaryBiproductTriangle
    (X₁ X₂ : C) [HasZeroMorphisms C] [HasBinaryBiproduct X₁ X₂] :
    binaryProductTriangle X₁ X₂ ≅ binaryBiproductTriangle X₁ X₂ :=
  Triangle.isoMk _ _ (Iso.refl _) (biprod.isoProd X₁ X₂).symm (Iso.refl _)
        /-
          C : Type u
          inst✝³ : CategoryTheory.Category.{v, u} C
          inst✝² : CategoryTheory.HasShift C Int
          T₁ T₂ T₃ : CategoryTheory.Pretriangulated.Triangle C
          X₁ X₂ : C
          inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms C
          inst✝ : CategoryTheory.Limits.HasBinaryBiproduct X₁ X₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated.binar …
        -/
        /-
          🎉 no goals
        -/
                       /-
                         🎉 no goals
                       -/
    (by aesop_cat) (by aesop_cat) (by aesop_cat)
                                      /-
                                        🎉 no goals
                                      -/


/-- The product of a family of triangles. -/
@[simps!]
def productTriangle : Triangle C :=
  Triangle.mk (Limits.Pi.map (fun j => (T j).mor₁))
    (Limits.Pi.map (fun j => (T j).mor₂))
    (Limits.Pi.map (fun j => (T j).mor₃) ≫ inv (piComparison _ _))


/-- A projection from the product of a family of triangles. -/
@[simps]
def productTriangle.π (j : J) :
    productTriangle T ⟶ T j where
  hom₁ := Pi.π _ j
  hom₂ := Pi.π _ j
  hom₃ := Pi.π _ j
  comm₃ := by
    /-
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      inst✝⁴ : CategoryTheory.HasShift C Int
      T₁ T₂ T₃ : CategoryTheory.Pretriangulated.Triangle C
      J : Type u_1
      T : J → CategoryTheory.Pretriangulated.Triangle C
      inst✝³ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₁
      inst✝² : CategoryTheory.Limits.HasProduct fun j => (T j).obj₂
      inst✝¹ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₃
      inst✝ : CategoryTheory.Limits.HasProduct fun j => (CategoryTheory.shiftFunctor …
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated.produ …
    -/
    dsimp
    /-
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      inst✝⁴ : CategoryTheory.HasShift C Int
      T₁ T₂ T₃ : CategoryTheory.Pretriangulated.Triangle C
      J : Type u_1
      T : J → CategoryTheory.Pretriangulated.Triangle C
      inst✝³ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₁
      inst✝² : CategoryTheory.Limits.HasProduct fun j => (T j).obj₂
      inst✝¹ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₃
      inst✝ : CategoryTheory.Limits.HasProduct fun j => (CategoryTheory.shiftFunctor …
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    rw [← piComparison_comp_π, assoc, IsIso.inv_hom_id_assoc]
    /-
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      inst✝⁴ : CategoryTheory.HasShift C Int
      T₁ T₂ T₃ : CategoryTheory.Pretriangulated.Triangle C
      J : Type u_1
      T : J → CategoryTheory.Pretriangulated.Triangle C
      inst✝³ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₁
      inst✝² : CategoryTheory.Limits.HasProduct fun j => (T j).obj₂
      inst✝¹ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₃
      inst✝ : CategoryTheory.Limits.HasProduct fun j => (CategoryTheory.shiftFunctor …
      j : J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Pi.map fun j = …
    -/
    simp only [limMap_π, Discrete.natTrans_app]
    /-
      🎉 no goals
    -/


/-- The fan given by `productTriangle T`. -/
@[simp]
def productTriangle.fan : Fan T := Fan.mk (productTriangle T) (productTriangle.π T)


/-- A family of morphisms `T' ⟶ T j` lifts to a morphism `T' ⟶ productTriangle T`. -/
@[simps]
def productTriangle.lift {T' : Triangle C} (φ : ∀ j, T' ⟶ T j) :
    T' ⟶ productTriangle T where
  hom₁ := Pi.lift (fun j => (φ j).hom₁)
  hom₂ := Pi.lift (fun j => (φ j).hom₂)
  hom₃ := Pi.lift (fun j => (φ j).hom₃)
  comm₃ := by
    /-
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      inst✝⁴ : CategoryTheory.HasShift C Int
      T₁ T₂ T₃ : CategoryTheory.Pretriangulated.Triangle C
      J : Type u_1
      T : J → CategoryTheory.Pretriangulated.Triangle C
      inst✝³ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₁
      inst✝² : CategoryTheory.Limits.HasProduct fun j => (T j).obj₂
      inst✝¹ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₃
      inst✝ : CategoryTheory.Limits.HasProduct fun j => (CategoryTheory.shiftFunctor …
      T' : CategoryTheory.Pretriangulated.Triangle C
      φ : (j : J) → Quiver.Hom T' (T j)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp T'.mor₃ ((CategoryTheory.shiftFunctor …
    -/
    dsimp
    /-
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      inst✝⁴ : CategoryTheory.HasShift C Int
      T₁ T₂ T₃ : CategoryTheory.Pretriangulated.Triangle C
      J : Type u_1
      T : J → CategoryTheory.Pretriangulated.Triangle C
      inst✝³ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₁
      inst✝² : CategoryTheory.Limits.HasProduct fun j => (T j).obj₂
      inst✝¹ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₃
      inst✝ : CategoryTheory.Limits.HasProduct fun j => (CategoryTheory.shiftFunctor …
      T' : CategoryTheory.Pretriangulated.Triangle C
      φ : (j : J) → Quiver.Hom T' (T j)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp T'.mor₃ ((CategoryTheory.shiftFunctor …
    -/
    rw [← cancel_mono (piComparison _ _), assoc, assoc, assoc, IsIso.inv_hom_id, comp_id]
    /-
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      inst✝⁴ : CategoryTheory.HasShift C Int
      T₁ T₂ T₃ : CategoryTheory.Pretriangulated.Triangle C
      J : Type u_1
      T : J → CategoryTheory.Pretriangulated.Triangle C
      inst✝³ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₁
      inst✝² : CategoryTheory.Limits.HasProduct fun j => (T j).obj₂
      inst✝¹ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₃
      inst✝ : CategoryTheory.Limits.HasProduct fun j => (CategoryTheory.shiftFunctor …
      T' : CategoryTheory.Pretriangulated.Triangle C
      φ : (j : J) → Quiver.Hom T' (T j)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp T'.mor₃ (CategoryTheory.CategoryStruc …
    -/
    aesop_cat
    /-
      🎉 no goals
    -/


/-- The triangle `productTriangle T` satisfies the universal property of the categorical
product of the triangles `T`. -/
def productTriangle.isLimitFan : IsLimit (productTriangle.fan T) :=
                                                                       /-
                                                                         C : Type u
                                                                         inst✝⁵ : CategoryTheory.Category.{v, u} C
                                                                         inst✝⁴ : CategoryTheory.HasShift C Int
                                                                         T₁ T₂ T₃ : CategoryTheory.Pretriangulated.Triangle C
                                                                         J : Type u_1
                                                                         T : J → CategoryTheory.Pretriangulated.Triangle C
                                                                         inst✝³ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₁
                                                                         inst✝² : CategoryTheory.Limits.HasProduct fun j => (T j).obj₂
                                                                         inst✝¹ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₃
                                                                         inst✝ : CategoryTheory.Limits.HasProduct fun j => (CategoryTheory.shiftFunctor …
                                                                         s : CategoryTheory.Limits.Fan T
                                                                         j : J
                                                                         ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun s => CategoryTheory.Pretriangul …
                                                                       -/
  mkFanLimit _ (fun s => productTriangle.lift T s.proj) (fun s j => by aesop_cat) (by
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
    /-
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      inst✝⁴ : CategoryTheory.HasShift C Int
      T₁ T₂ T₃ : CategoryTheory.Pretriangulated.Triangle C
      J : Type u_1
      T : J → CategoryTheory.Pretriangulated.Triangle C
      inst✝³ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₁
      inst✝² : CategoryTheory.Limits.HasProduct fun j => (T j).obj₂
      inst✝¹ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₃
      inst✝ : CategoryTheory.Limits.HasProduct fun j => (CategoryTheory.shiftFunctor …
      ⊢ ∀ (s : CategoryTheory.Limits.Fan T) (m : Quiver.Hom s.pt (CategoryTheory.Pre …
    -/
    intro s m hm
    /-
      C : Type u
      inst✝⁵ : CategoryTheory.Category.{v, u} C
      inst✝⁴ : CategoryTheory.HasShift C Int
      T₁ T₂ T₃ : CategoryTheory.Pretriangulated.Triangle C
      J : Type u_1
      T : J → CategoryTheory.Pretriangulated.Triangle C
      inst✝³ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₁
      inst✝² : CategoryTheory.Limits.HasProduct fun j => (T j).obj₂
      inst✝¹ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₃
      inst✝ : CategoryTheory.Limits.HasProduct fun j => (CategoryTheory.shiftFunctor …
      s : CategoryTheory.Limits.Fan T
      m : Quiver.Hom s.pt (CategoryTheory.Pretriangulated.productTriangle.fan T).pt
      hm : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp m ((CategoryTheory.Pret …
      ⊢ Eq m ((fun s => CategoryTheory.Pretriangulated.productTriangle.lift T s.proj …
    -/
    ext1
    all_goals
      exact Pi.hom_ext _ _ (fun j => (by simp [← hm])))


lemma productTriangle.zero₃₁ [HasZeroMorphisms C]
    (h : ∀ j, (T j).mor₃ ≫ (T j).mor₁⟦(1 : ℤ)⟧' = 0) :
    (productTriangle T).mor₃ ≫ (productTriangle T).mor₁⟦1⟧' = 0 := by
  have : HasProduct (fun j => (T j).obj₂⟦(1 : ℤ)⟧) :=
    ⟨_, isLimitFanMkObjOfIsLimit (shiftFunctor C (1 : ℤ)) _ _
      (productIsProduct (fun j => (T j).obj₂))⟩
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.HasShift C Int
    J : Type u_1
    T : J → CategoryTheory.Pretriangulated.Triangle C
    inst✝⁴ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₁
    inst✝³ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₂
    inst✝² : CategoryTheory.Limits.HasProduct fun j => (T j).obj₃
    inst✝¹ : CategoryTheory.Limits.HasProduct fun j => (CategoryTheory.shiftFuncto …
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    h : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (T j).mor₃ ((CategoryThe …
    this : CategoryTheory.Limits.HasProduct fun j => (CategoryTheory.shiftFunctor  …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated.produ …
  -/
  dsimp
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.HasShift C Int
    J : Type u_1
    T : J → CategoryTheory.Pretriangulated.Triangle C
    inst✝⁴ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₁
    inst✝³ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₂
    inst✝² : CategoryTheory.Limits.HasProduct fun j => (T j).obj₃
    inst✝¹ : CategoryTheory.Limits.HasProduct fun j => (CategoryTheory.shiftFuncto …
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    h : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (T j).mor₃ ((CategoryThe …
    this : CategoryTheory.Limits.HasProduct fun j => (CategoryTheory.shiftFunctor  …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  change _ ≫ (Pi.lift (fun j => Pi.π _ j ≫ (T j).mor₁))⟦(1 : ℤ)⟧' = 0
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.HasShift C Int
    J : Type u_1
    T : J → CategoryTheory.Pretriangulated.Triangle C
    inst✝⁴ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₁
    inst✝³ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₂
    inst✝² : CategoryTheory.Limits.HasProduct fun j => (T j).obj₃
    inst✝¹ : CategoryTheory.Limits.HasProduct fun j => (CategoryTheory.shiftFuncto …
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    h : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (T j).mor₃ ((CategoryThe …
    this : CategoryTheory.Limits.HasProduct fun j => (CategoryTheory.shiftFunctor  …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  rw [assoc, ← cancel_mono (piComparison _ _), zero_comp, assoc, assoc]
  /-
    C : Type u
    inst✝⁶ : CategoryTheory.Category.{v, u} C
    inst✝⁵ : CategoryTheory.HasShift C Int
    J : Type u_1
    T : J → CategoryTheory.Pretriangulated.Triangle C
    inst✝⁴ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₁
    inst✝³ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₂
    inst✝² : CategoryTheory.Limits.HasProduct fun j => (T j).obj₃
    inst✝¹ : CategoryTheory.Limits.HasProduct fun j => (CategoryTheory.shiftFuncto …
    inst✝ : CategoryTheory.Limits.HasZeroMorphisms C
    h : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (T j).mor₃ ((CategoryThe …
    this : CategoryTheory.Limits.HasProduct fun j => (CategoryTheory.shiftFunctor  …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Pi.map fun j = …
  -/
  ext j
  simp only [map_lift_piComparison, assoc, limit.lift_π, Fan.mk_π_app, zero_comp,
    Functor.map_comp, ← piComparison_comp_π_assoc, IsIso.inv_hom_id_assoc,
    limMap_π_assoc, Discrete.natTrans_app, h j, comp_zero]


variable (C) in
/-- The functor `C ⥤ Triangle C` which sends `X` to `contractibleTriangle X`. -/
@[simps]
def contractibleTriangleFunctor [HasZeroObject C] [HasZeroMorphisms C] : C ⥤ Triangle C where
  obj X := contractibleTriangle X
  map f :=
    { hom₁ := f
      hom₂ := f
      hom₃ := 0 }


/-- The first projection `Triangle C ⥤ C`. -/
@[simps]
def π₁ : Triangle C ⥤ C where
  obj T := T.obj₁
  map f := f.hom₁


/-- The second projection `Triangle C ⥤ C`. -/
@[simps]
def π₂ : Triangle C ⥤ C where
  obj T := T.obj₂
  map f := f.hom₂


/-- The third projection `Triangle C ⥤ C`. -/
@[simps]
def π₃ : Triangle C ⥤ C where
  obj T := T.obj₃
  map f := f.hom₃


instance : IsIso φ.hom₁ := (inferInstance : IsIso (π₁.map φ))

instance : IsIso φ.hom₂ := (inferInstance : IsIso (π₂.map φ))

instance : IsIso φ.hom₃ := (inferInstance : IsIso (π₃.map φ))


