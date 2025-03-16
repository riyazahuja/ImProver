/-- A preadditive category `C` with an additive shift, and a class of "distinguished triangles"
relative to that shift is called pretriangulated if the following hold:
* Any triangle that is isomorphic to a distinguished triangle is also distinguished.
* Any triangle of the form `(X,X,0,id,0,0)` is distinguished.
* For any morphism `f : X ⟶ Y` there exists a distinguished triangle of the form `(X,Y,Z,f,g,h)`.
* The triangle `(X,Y,Z,f,g,h)` is distinguished if and only if `(Y,Z,X⟦1⟧,g,h,-f⟦1⟧)` is.
* Given a diagram:
  ```
        f       g       h
    X  ───> Y  ───> Z  ───> X⟦1⟧
    │       │                │
    │a      │b               │a⟦1⟧'
    V       V                V
    X' ───> Y' ───> Z' ───> X'⟦1⟧
        f'      g'      h'
  ```
  where the left square commutes, and whose rows are distinguished triangles,
  there exists a morphism `c : Z ⟶ Z'` such that `(a,b,c)` is a triangle morphism.

See <https://stacks.math.columbia.edu/tag/0145>
-/
class Pretriangulated [∀ n : ℤ, Functor.Additive (shiftFunctor C n)] where
  /-- a class of triangle which are called `distinguished` -/
  distinguishedTriangles : Set (Triangle C)
  /-- a triangle that is isomorphic to a distinguished triangle is distinguished -/
  isomorphic_distinguished :
    ∀ T₁ ∈ distinguishedTriangles, ∀ (T₂) (_ : T₂ ≅ T₁), T₂ ∈ distinguishedTriangles
  /-- obvious triangles `X ⟶ X ⟶ 0 ⟶ X⟦1⟧` are distinguished -/
  contractible_distinguished : ∀ X : C, contractibleTriangle X ∈ distinguishedTriangles
  /-- any morphism `X ⟶ Y` is part of a distinguished triangle `X ⟶ Y ⟶ Z ⟶ X⟦1⟧` -/
  distinguished_cocone_triangle :
    ∀ {X Y : C} (f : X ⟶ Y),
      ∃ (Z : C) (g : Y ⟶ Z) (h : Z ⟶ X⟦(1 : ℤ)⟧), Triangle.mk f g h ∈ distinguishedTriangles
  /-- a triangle is distinguished iff it is so after rotating it -/
  rotate_distinguished_triangle :
    ∀ T : Triangle C, T ∈ distinguishedTriangles ↔ T.rotate ∈ distinguishedTriangles
  /-- given two distinguished triangle, a commutative square
        can be extended as morphism of triangles -/
  complete_distinguished_triangle_morphism :
    ∀ (T₁ T₂ : Triangle C) (_ : T₁ ∈ distinguishedTriangles) (_ : T₂ ∈ distinguishedTriangles)
      (a : T₁.obj₁ ⟶ T₂.obj₁) (b : T₁.obj₂ ⟶ T₂.obj₂) (_ : T₁.mor₁ ≫ b = a ≫ T₂.mor₁),
      ∃ c : T₁.obj₃ ⟶ T₂.obj₃, T₁.mor₂ ≫ c = b ≫ T₂.mor₂ ∧ T₁.mor₃ ≫ a⟦1⟧' = c ≫ T₂.mor₃



/-- distinguished triangles in a pretriangulated category -/
notation:60 "distTriang " C => @distinguishedTriangles C _ _ _ _ _ _


lemma distinguished_iff_of_iso {T₁ T₂ : Triangle C} (e : T₁ ≅ T₂) :
    (T₁ ∈ distTriang C) ↔ T₂ ∈ distTriang C :=
  ⟨fun hT₁ => isomorphic_distinguished _ hT₁ _ e.symm,
    fun hT₂ => isomorphic_distinguished _ hT₂ _ e⟩


/-- Given any distinguished triangle `T`, then we know `T.rotate` is also distinguished.
-/
theorem rot_of_distTriang (T : Triangle C) (H : T ∈ distTriang C) : T.rotate ∈ distTriang C :=
  (rotate_distinguished_triangle T).mp H


/-- Given any distinguished triangle `T`, then we know `T.inv_rotate` is also distinguished.
-/
theorem inv_rot_of_distTriang (T : Triangle C) (H : T ∈ distTriang C) :
    T.invRotate ∈ distTriang C :=
  (rotate_distinguished_triangle T.invRotate).mpr
    (isomorphic_distinguished T H T.invRotate.rotate (invRotCompRot.app T))


/-- Given any distinguished triangle
```
      f       g       h
  X  ───> Y  ───> Z  ───> X⟦1⟧
```
the composition `f ≫ g = 0`.
See <https://stacks.math.columbia.edu/tag/0146>
-/
@[reassoc]
theorem comp_distTriang_mor_zero₁₂ (T) (H : T ∈ (distTriang C)) : T.mor₁ ≫ T.mor₂ = 0 := by
  obtain ⟨c, hc⟩ :=
    complete_distinguished_triangle_morphism _ _ (contractible_distinguished T.obj₁) H (𝟙 T.obj₁)
      T.mor₁ rfl
  /-
    case intro
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle C
    H : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    c : Quiver.Hom (CategoryTheory.Pretriangulated.contractibleTriangle T.obj₁).ob …
    hc : And (Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulat …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp T.mor₁ T.mor₂) 0
  -/
  simpa only [contractibleTriangle_mor₂, zero_comp] using hc.left.symm
  /-
    🎉 no goals
  -/


/-- Given any distinguished triangle
```
      f       g       h
  X  ───> Y  ───> Z  ───> X⟦1⟧
```
the composition `g ≫ h = 0`.
See <https://stacks.math.columbia.edu/tag/0146>
-/
@[reassoc]
theorem comp_distTriang_mor_zero₂₃ (T : Triangle C) (H : T ∈ distTriang C) :
    T.mor₂ ≫ T.mor₃ = 0 :=
  comp_distTriang_mor_zero₁₂ T.rotate (rot_of_distTriang T H)


/-- Given any distinguished triangle
```
      f       g       h
  X  ───> Y  ───> Z  ───> X⟦1⟧
```
the composition `h ≫ f⟦1⟧ = 0`.
See <https://stacks.math.columbia.edu/tag/0146>
-/
@[reassoc]
theorem comp_distTriang_mor_zero₃₁ (T : Triangle C) (H : T ∈ distTriang C) :
    T.mor₃ ≫ T.mor₁⟦1⟧' = 0 := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle C
    H : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    ⊢ Eq (CategoryTheory.CategoryStruct.comp T.mor₃ ((CategoryTheory.shiftFunctor  …
  -/
  have H₂ := rot_of_distTriang T.rotate (rot_of_distTriang T H)
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle C
    H : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    H₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T.ro …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp T.mor₃ ((CategoryTheory.shiftFunctor  …
  -/
  simpa using comp_distTriang_mor_zero₁₂ T.rotate.rotate H₂
  /-
    🎉 no goals
  -/


/-- The short complex `T.obj₁ ⟶ T.obj₂ ⟶ T.obj₃` attached to a distinguished triangle. -/
@[simps]
def shortComplexOfDistTriangle (T : Triangle C) (hT : T ∈ distTriang C) : ShortComplex C :=
  ShortComplex.mk T.mor₁ T.mor₂ (comp_distTriang_mor_zero₁₂ _ hT)


/-- The isomorphism between the short complex attached to
two isomorphic distinguished triangles. -/
@[simps!]
def shortComplexOfDistTriangleIsoOfIso {T T' : Triangle C} (e : T ≅ T') (hT : T ∈ distTriang C) :
    shortComplexOfDistTriangle T hT ≅ shortComplexOfDistTriangle T'
      (isomorphic_distinguished _ hT _ e.symm) :=
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T T' : CategoryTheory.Pretriangulated.Triangle C
    e : CategoryTheory.Iso T T'
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated.Trian …
  -/
  /-
    🎉 no goals
  -/
  ShortComplex.isoMk (Triangle.π₁.mapIso e) (Triangle.π₂.mapIso e) (Triangle.π₃.mapIso e)
  /-
    🎉 no goals
  -/


/-- Any morphism `Y ⟶ Z` is part of a distinguished triangle `X ⟶ Y ⟶ Z ⟶ X⟦1⟧` -/
lemma distinguished_cocone_triangle₁ {Y Z : C} (g : Y ⟶ Z) :
    ∃ (X : C) (f : X ⟶ Y) (h : Z ⟶ X⟦(1 : ℤ)⟧), Triangle.mk f g h ∈ distTriang C := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    Y Z : C
    g : Quiver.Hom Y Z
    ⊢ Exists fun X => Exists fun f => Exists fun h => Membership.mem CategoryTheor …
  -/
  obtain ⟨X', f', g', mem⟩ := distinguished_cocone_triangle g
  /-
    case intro.intro.intro
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    Y Z : C
    g : Quiver.Hom Y Z
    X' : C
    f' : Quiver.Hom Z X'
    g' : Quiver.Hom X' ((CategoryTheory.shiftFunctor C 1).obj Y)
    mem : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    ⊢ Exists fun X => Exists fun f => Exists fun h => Membership.mem CategoryTheor …
  -/
  exact ⟨_, _, _, inv_rot_of_distTriang _ mem⟩
  /-
    🎉 no goals
  -/


/-- Any morphism `Z ⟶ X⟦1⟧` is part of a distinguished triangle `X ⟶ Y ⟶ Z ⟶ X⟦1⟧` -/
lemma distinguished_cocone_triangle₂ {Z X : C} (h : Z ⟶ X⟦(1 : ℤ)⟧) :
    ∃ (Y : C) (f : X ⟶ Y) (g : Y ⟶ Z), Triangle.mk f g h ∈ distTriang C := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    Z X : C
    h : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj X)
    ⊢ Exists fun Y => Exists fun f => Exists fun g => Membership.mem CategoryTheor …
  -/
  obtain ⟨Y', f', g', mem⟩ := distinguished_cocone_triangle h
  /-
    case intro.intro.intro
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    Z X : C
    h : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj X)
    Y' : C
    f' : Quiver.Hom ((CategoryTheory.shiftFunctor C 1).obj X) Y'
    g' : Quiver.Hom Y' ((CategoryTheory.shiftFunctor C 1).obj Z)
    mem : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    ⊢ Exists fun Y => Exists fun f => Exists fun g => Membership.mem CategoryTheor …
  -/
  let T' := (Triangle.mk h f' g').invRotate.invRotate
  refine ⟨T'.obj₂, ((shiftEquiv C (1 : ℤ)).unitIso.app X).hom ≫ T'.mor₁, T'.mor₂,
    isomorphic_distinguished _ (inv_rot_of_distTriang _ (inv_rot_of_distTriang _ mem)) _ ?_⟩
  exact Triangle.isoMk _ _ ((shiftEquiv C (1 : ℤ)).unitIso.app X) (Iso.refl _) (Iso.refl _)
    (by aesop_cat) (by aesop_cat)
    (by dsimp; simp only [shift_shiftFunctorCompIsoId_inv_app, id_comp])


/-- A commutative square involving the morphisms `mor₂` of two distinguished triangles
can be extended as morphism of triangles -/
lemma complete_distinguished_triangle_morphism₁ (T₁ T₂ : Triangle C)
    (hT₁ : T₁ ∈ distTriang C) (hT₂ : T₂ ∈ distTriang C) (b : T₁.obj₂ ⟶ T₂.obj₂)
    (c : T₁.obj₃ ⟶ T₂.obj₃) (comm : T₁.mor₂ ≫ c = b ≫ T₂.mor₂) :
    ∃ (a : T₁.obj₁ ⟶ T₂.obj₁), T₁.mor₁ ≫ b = a ≫ T₂.mor₁ ∧
      T₁.mor₃ ≫ a⟦(1 : ℤ)⟧' = c ≫ T₂.mor₃ := by
  obtain ⟨a, ⟨ha₁, ha₂⟩⟩ := complete_distinguished_triangle_morphism _ _
    (rot_of_distTriang _ hT₁) (rot_of_distTriang _ hT₂) b c comm
  /-
    case intro.intro
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle C
    hT₁ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁
    hT₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₂
    b : Quiver.Hom T₁.obj₂ T₂.obj₂
    c : Quiver.Hom T₁.obj₃ T₂.obj₃
    comm : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₂ c) (CategoryTheory.Categ …
    a : Quiver.Hom T₁.rotate.obj₃ T₂.rotate.obj₃
    ha₁ : Eq (CategoryTheory.CategoryStruct.comp T₁.rotate.mor₂ a) (CategoryTheory …
    ha₂ : Eq (CategoryTheory.CategoryStruct.comp T₁.rotate.mor₃ ((CategoryTheory.s …
    ⊢ Exists fun a => And (Eq (CategoryTheory.CategoryStruct.comp T₁.mor₁ b) (Cate …
  -/
  refine ⟨(shiftFunctor C (1 : ℤ)).preimage a, ⟨?_, ?_⟩⟩
    /-
      case intro.intro.refine_1
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T₁ T₂ : CategoryTheory.Pretriangulated.Triangle C
      hT₁ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁
      hT₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₂
      b : Quiver.Hom T₁.obj₂ T₂.obj₂
      c : Quiver.Hom T₁.obj₃ T₂.obj₃
      comm : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₂ c) (CategoryTheory.Categ …
      a : Quiver.Hom T₁.rotate.obj₃ T₂.rotate.obj₃
      ha₁ : Eq (CategoryTheory.CategoryStruct.comp T₁.rotate.mor₂ a) (CategoryTheory …
      ha₂ : Eq (CategoryTheory.CategoryStruct.comp T₁.rotate.mor₃ ((CategoryTheory.s …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp T₁.mor₁ b) (CategoryTheory.CategorySt …
    -/
  · apply (shiftFunctor C (1 : ℤ)).map_injective
    /-
      case intro.intro.refine_1.a
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T₁ T₂ : CategoryTheory.Pretriangulated.Triangle C
      hT₁ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁
      hT₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₂
      b : Quiver.Hom T₁.obj₂ T₂.obj₂
      c : Quiver.Hom T₁.obj₃ T₂.obj₃
      comm : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₂ c) (CategoryTheory.Categ …
      a : Quiver.Hom T₁.rotate.obj₃ T₂.rotate.obj₃
      ha₁ : Eq (CategoryTheory.CategoryStruct.comp T₁.rotate.mor₂ a) (CategoryTheory …
      ha₂ : Eq (CategoryTheory.CategoryStruct.comp T₁.rotate.mor₃ ((CategoryTheory.s …
      ⊢ Eq ((CategoryTheory.shiftFunctor C 1).map (CategoryTheory.CategoryStruct.com …
    -/
    dsimp at ha₂
    /-
      case intro.intro.refine_1.a
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T₁ T₂ : CategoryTheory.Pretriangulated.Triangle C
      hT₁ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁
      hT₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₂
      b : Quiver.Hom T₁.obj₂ T₂.obj₂
      c : Quiver.Hom T₁.obj₃ T₂.obj₃
      comm : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₂ c) (CategoryTheory.Categ …
      a : Quiver.Hom T₁.rotate.obj₃ T₂.rotate.obj₃
      ha₁ : Eq (CategoryTheory.CategoryStruct.comp T₁.rotate.mor₂ a) (CategoryTheory …
      ha₂ : Eq (CategoryTheory.CategoryStruct.comp (Neg.neg ((CategoryTheory.shiftFu …
      ⊢ Eq ((CategoryTheory.shiftFunctor C 1).map (CategoryTheory.CategoryStruct.com …
    -/
    rw [neg_comp, comp_neg, neg_inj] at ha₂
    /-
      case intro.intro.refine_1.a
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T₁ T₂ : CategoryTheory.Pretriangulated.Triangle C
      hT₁ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁
      hT₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₂
      b : Quiver.Hom T₁.obj₂ T₂.obj₂
      c : Quiver.Hom T₁.obj₃ T₂.obj₃
      comm : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₂ c) (CategoryTheory.Categ …
      a : Quiver.Hom T₁.rotate.obj₃ T₂.rotate.obj₃
      ha₁ : Eq (CategoryTheory.CategoryStruct.comp T₁.rotate.mor₂ a) (CategoryTheory …
      ha₂ : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctor C 1 …
      ⊢ Eq ((CategoryTheory.shiftFunctor C 1).map (CategoryTheory.CategoryStruct.com …
    -/
    simpa only [Functor.map_comp, Functor.map_preimage] using ha₂
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T₁ T₂ : CategoryTheory.Pretriangulated.Triangle C
      hT₁ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁
      hT₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₂
      b : Quiver.Hom T₁.obj₂ T₂.obj₂
      c : Quiver.Hom T₁.obj₃ T₂.obj₃
      comm : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₂ c) (CategoryTheory.Categ …
      a : Quiver.Hom T₁.rotate.obj₃ T₂.rotate.obj₃
      ha₁ : Eq (CategoryTheory.CategoryStruct.comp T₁.rotate.mor₂ a) (CategoryTheory …
      ha₂ : Eq (CategoryTheory.CategoryStruct.comp T₁.rotate.mor₃ ((CategoryTheory.s …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp T₁.mor₃ ((CategoryTheory.shiftFunctor …
    -/
  · simpa only [Functor.map_preimage] using ha₁
    /-
      🎉 no goals
    -/


/-- A commutative square involving the morphisms `mor₃` of two distinguished triangles
can be extended as morphism of triangles -/
lemma complete_distinguished_triangle_morphism₂ (T₁ T₂ : Triangle C)
    (hT₁ : T₁ ∈ distTriang C) (hT₂ : T₂ ∈ distTriang C) (a : T₁.obj₁ ⟶ T₂.obj₁)
    (c : T₁.obj₃ ⟶ T₂.obj₃) (comm : T₁.mor₃ ≫ a⟦(1 : ℤ)⟧' = c ≫ T₂.mor₃) :
    ∃ (b : T₁.obj₂ ⟶ T₂.obj₂), T₁.mor₁ ≫ b = a ≫ T₂.mor₁ ∧ T₁.mor₂ ≫ c = b ≫ T₂.mor₂ := by
  obtain ⟨a, ⟨ha₁, ha₂⟩⟩ := complete_distinguished_triangle_morphism _ _
    (inv_rot_of_distTriang _ hT₁) (inv_rot_of_distTriang _ hT₂) (c⟦(-1 : ℤ)⟧') a (by
    dsimp
    simp only [neg_comp, comp_neg, ← Functor.map_comp_assoc, ← comm,
      Functor.map_comp, shift_shift_neg', Functor.id_obj, assoc, Iso.inv_hom_id_app, comp_id])
  /-
    case intro.intro
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle C
    hT₁ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁
    hT₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₂
    a✝ : Quiver.Hom T₁.obj₁ T₂.obj₁
    c : Quiver.Hom T₁.obj₃ T₂.obj₃
    comm : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₃ ((CategoryTheory.shiftFu …
    a : Quiver.Hom T₁.invRotate.obj₃ T₂.invRotate.obj₃
    ha₁ : Eq (CategoryTheory.CategoryStruct.comp T₁.invRotate.mor₂ a) (CategoryThe …
    ha₂ : Eq (CategoryTheory.CategoryStruct.comp T₁.invRotate.mor₃ ((CategoryTheor …
    ⊢ Exists fun b => And (Eq (CategoryTheory.CategoryStruct.comp T₁.mor₁ b) (Cate …
  -/
  refine ⟨a, ⟨ha₁, ?_⟩⟩
  /-
    case intro.intro
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle C
    hT₁ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁
    hT₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₂
    a✝ : Quiver.Hom T₁.obj₁ T₂.obj₁
    c : Quiver.Hom T₁.obj₃ T₂.obj₃
    comm : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₃ ((CategoryTheory.shiftFu …
    a : Quiver.Hom T₁.invRotate.obj₃ T₂.invRotate.obj₃
    ha₁ : Eq (CategoryTheory.CategoryStruct.comp T₁.invRotate.mor₂ a) (CategoryThe …
    ha₂ : Eq (CategoryTheory.CategoryStruct.comp T₁.invRotate.mor₃ ((CategoryTheor …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp T₁.mor₂ c) (CategoryTheory.CategorySt …
  -/
  dsimp only [Triangle.invRotate, Triangle.mk] at ha₂
  /-
    case intro.intro
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle C
    hT₁ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁
    hT₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₂
    a✝ : Quiver.Hom T₁.obj₁ T₂.obj₁
    c : Quiver.Hom T₁.obj₃ T₂.obj₃
    comm : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₃ ((CategoryTheory.shiftFu …
    a : Quiver.Hom T₁.invRotate.obj₃ T₂.invRotate.obj₃
    ha₁ : Eq (CategoryTheory.CategoryStruct.comp T₁.invRotate.mor₂ a) (CategoryThe …
    ha₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.co …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp T₁.mor₂ c) (CategoryTheory.CategorySt …
  -/
  rw [← cancel_mono ((shiftEquiv C (1 : ℤ)).counitIso.inv.app T₂.obj₃), assoc, assoc, ← ha₂]
  /-
    case intro.intro
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle C
    hT₁ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁
    hT₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₂
    a✝ : Quiver.Hom T₁.obj₁ T₂.obj₁
    c : Quiver.Hom T₁.obj₃ T₂.obj₃
    comm : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₃ ((CategoryTheory.shiftFu …
    a : Quiver.Hom T₁.invRotate.obj₃ T₂.invRotate.obj₃
    ha₁ : Eq (CategoryTheory.CategoryStruct.comp T₁.invRotate.mor₂ a) (CategoryThe …
    ha₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.co …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp T₁.mor₂ (CategoryTheory.CategoryStruc …
  -/
  simp only [shiftEquiv'_counitIso, shift_neg_shift', assoc, Iso.inv_hom_id_app_assoc]
  /-
    🎉 no goals
  -/


/-- Obvious triangles `0 ⟶ X ⟶ X ⟶ 0⟦1⟧` are distinguished -/
lemma contractible_distinguished₁ (X : C) :
    Triangle.mk (0 : 0 ⟶ X) (𝟙 X) 0 ∈ distTriang C := by
  refine isomorphic_distinguished _
    (inv_rot_of_distTriang _ (contractible_distinguished X)) _ ?_
  exact Triangle.isoMk _ _ (Functor.mapZeroObject _).symm (Iso.refl _) (Iso.refl _)
    (by aesop_cat) (by aesop_cat) (by aesop_cat)


/-- Obvious triangles `X ⟶ 0 ⟶ X⟦1⟧ ⟶ X⟦1⟧` are distinguished -/
lemma contractible_distinguished₂ (X : C) :
    Triangle.mk (0 : X ⟶ 0) 0 (𝟙 (X⟦1⟧)) ∈ distTriang C := by
  refine isomorphic_distinguished _
    (inv_rot_of_distTriang _ (contractible_distinguished₁ (X⟦(1 : ℤ)⟧))) _ ?_
  exact Triangle.isoMk _ _ ((shiftEquiv C (1 : ℤ)).unitIso.app X) (Iso.refl _) (Iso.refl _)
    (by aesop_cat) (by aesop_cat)
    (by dsimp; simp only [shift_shiftFunctorCompIsoId_inv_app, id_comp])


lemma yoneda_exact₂ {X : C} (f : T.obj₂ ⟶ X) (hf : T.mor₁ ≫ f = 0) :
    ∃ (g : T.obj₃ ⟶ X), f = T.mor₂ ≫ g := by
  obtain ⟨g, ⟨hg₁, _⟩⟩ := complete_distinguished_triangle_morphism T _ hT
    (contractible_distinguished₁ X) 0 f (by aesop_cat)
  /-
    case intro.intro
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    X : C
    f : Quiver.Hom T.obj₂ X
    hf : Eq (CategoryTheory.CategoryStruct.comp T.mor₁ f) 0
    g : Quiver.Hom T.obj₃ (CategoryTheory.Pretriangulated.Triangle.mk 0 (CategoryT …
    hg₁ : Eq (CategoryTheory.CategoryStruct.comp T.mor₂ g) (CategoryTheory.Categor …
    right✝ : Eq (CategoryTheory.CategoryStruct.comp T.mor₃ ((CategoryTheory.shiftF …
    ⊢ Exists fun g => Eq f (CategoryTheory.CategoryStruct.comp T.mor₂ g)
  -/
  exact ⟨g, by simpa using hg₁.symm⟩
  /-
    🎉 no goals
  -/


lemma yoneda_exact₃ {X : C} (f : T.obj₃ ⟶ X) (hf : T.mor₂ ≫ f = 0) :
    ∃ (g : T.obj₁⟦(1 : ℤ)⟧ ⟶ X), f = T.mor₃ ≫ g :=
  yoneda_exact₂ _ (rot_of_distTriang _ hT) f hf


lemma coyoneda_exact₂ {X : C} (f : X ⟶ T.obj₂) (hf : f ≫ T.mor₂ = 0) :
    ∃ (g : X ⟶ T.obj₁), f = g ≫ T.mor₁ := by
  obtain ⟨a, ⟨ha₁, _⟩⟩ := complete_distinguished_triangle_morphism₁ _ T
    (contractible_distinguished X) hT f 0 (by aesop_cat)
  /-
    case intro.intro
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    X : C
    f : Quiver.Hom X T.obj₂
    hf : Eq (CategoryTheory.CategoryStruct.comp f T.mor₂) 0
    a : Quiver.Hom (CategoryTheory.Pretriangulated.contractibleTriangle X).obj₁ T. …
    ha₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated.c …
    right✝ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulate …
    ⊢ Exists fun g => Eq f (CategoryTheory.CategoryStruct.comp g T.mor₁)
  -/
  exact ⟨a, by simpa using ha₁⟩
  /-
    🎉 no goals
  -/


lemma coyoneda_exact₁ {X : C} (f : X ⟶ T.obj₁⟦(1 : ℤ)⟧) (hf : f ≫ T.mor₁⟦1⟧' = 0) :
    ∃ (g : X ⟶ T.obj₃), f = g ≫ T.mor₃ :=
                                                                         /-
                                                                           C : Type u
                                                                           inst✝⁴ : CategoryTheory.Category.{v, u} C
                                                                           inst✝³ : CategoryTheory.Limits.HasZeroObject C
                                                                           inst✝² : CategoryTheory.HasShift C Int
                                                                           inst✝¹ : CategoryTheory.Preadditive C
                                                                           inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
                                                                           hC : CategoryTheory.Pretriangulated C
                                                                           T : CategoryTheory.Pretriangulated.Triangle C
                                                                           hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
                                                                           X : C
                                                                           f : Quiver.Hom X ((CategoryTheory.shiftFunctor C 1).obj T.obj₁)
                                                                           hf : Eq (CategoryTheory.CategoryStruct.comp f ((CategoryTheory.shiftFunctor C  …
                                                                           ⊢ Eq (CategoryTheory.CategoryStruct.comp f T.rotate.rotate.mor₂) 0
                                                                         -/
  coyoneda_exact₂ _ (rot_of_distTriang _ (rot_of_distTriang _ hT)) f (by aesop_cat)
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


lemma coyoneda_exact₃ {X : C} (f : X ⟶ T.obj₃) (hf : f ≫ T.mor₃ = 0) :
    ∃ (g : X ⟶ T.obj₂), f = g ≫ T.mor₂ :=
  coyoneda_exact₂ _ (rot_of_distTriang _ hT) f hf


lemma mor₃_eq_zero_iff_epi₂ : T.mor₃ = 0 ↔ Epi T.mor₂ := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    ⊢ Iff (Eq T.mor₃ 0) (CategoryTheory.Epi T.mor₂)
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      ⊢ Eq T.mor₃ 0 → CategoryTheory.Epi T.mor₂
    -/
  · intro h
    /-
      case mp
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      h : Eq T.mor₃ 0
      ⊢ CategoryTheory.Epi T.mor₂
    -/
    rw [epi_iff_cancel_zero]
    /-
      case mp
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      h : Eq T.mor₃ 0
      ⊢ ∀ (R : C) (g : Quiver.Hom T.obj₃ R), Eq (CategoryTheory.CategoryStruct.comp  …
    -/
    intro X g hg
    /-
      case mp
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      h : Eq T.mor₃ 0
      X : C
      g : Quiver.Hom T.obj₃ X
      hg : Eq (CategoryTheory.CategoryStruct.comp T.mor₂ g) 0
      ⊢ Eq g 0
    -/
    obtain ⟨f, rfl⟩ := yoneda_exact₃ T hT g hg
    /-
      case mp.intro
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      h : Eq T.mor₃ 0
      X : C
      f : Quiver.Hom ((CategoryTheory.shiftFunctor C 1).obj T.obj₁) X
      hg : Eq (CategoryTheory.CategoryStruct.comp T.mor₂ (CategoryTheory.CategoryStr …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp T.mor₃ f) 0
    -/
    rw [h, zero_comp]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      ⊢ CategoryTheory.Epi T.mor₂ → Eq T.mor₃ 0
    -/
  · intro
    /-
      case mpr
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      a✝ : CategoryTheory.Epi T.mor₂
      ⊢ Eq T.mor₃ 0
    -/
    rw [← cancel_epi T.mor₂, comp_distTriang_mor_zero₂₃ _ hT, comp_zero]
    /-
      🎉 no goals
    -/


lemma mor₂_eq_zero_iff_epi₁ : T.mor₂ = 0 ↔ Epi T.mor₁ := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    ⊢ Iff (Eq T.mor₂ 0) (CategoryTheory.Epi T.mor₁)
  -/
  have h := mor₃_eq_zero_iff_epi₂ _ (inv_rot_of_distTriang _ hT)
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    h : Iff (Eq T.invRotate.mor₃ 0) (CategoryTheory.Epi T.invRotate.mor₂)
    ⊢ Iff (Eq T.mor₂ 0) (CategoryTheory.Epi T.mor₁)
  -/
  dsimp at h
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    h : Iff (Eq (CategoryTheory.CategoryStruct.comp T.mor₂ ((CategoryTheory.shiftF …
    ⊢ Iff (Eq T.mor₂ 0) (CategoryTheory.Epi T.mor₁)
  -/
  rw [← h, IsIso.comp_right_eq_zero]
  /-
    🎉 no goals
  -/


lemma mor₁_eq_zero_iff_epi₃ : T.mor₁ = 0 ↔ Epi T.mor₃ := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    ⊢ Iff (Eq T.mor₁ 0) (CategoryTheory.Epi T.mor₃)
  -/
  have h := mor₃_eq_zero_iff_epi₂ _ (rot_of_distTriang _ hT)
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    h : Iff (Eq T.rotate.mor₃ 0) (CategoryTheory.Epi T.rotate.mor₂)
    ⊢ Iff (Eq T.mor₁ 0) (CategoryTheory.Epi T.mor₃)
  -/
  dsimp at h
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    h : Iff (Eq (Neg.neg ((CategoryTheory.shiftFunctor C 1).map T.mor₁)) 0) (Categ …
    ⊢ Iff (Eq T.mor₁ 0) (CategoryTheory.Epi T.mor₃)
  -/
  rw [← h, neg_eq_zero]
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    h : Iff (Eq (Neg.neg ((CategoryTheory.shiftFunctor C 1).map T.mor₁)) 0) (Categ …
    ⊢ Iff (Eq T.mor₁ 0) (Eq ((CategoryTheory.shiftFunctor C 1).map T.mor₁) 0)
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      h : Iff (Eq (Neg.neg ((CategoryTheory.shiftFunctor C 1).map T.mor₁)) 0) (Categ …
      ⊢ Eq T.mor₁ 0 → Eq ((CategoryTheory.shiftFunctor C 1).map T.mor₁) 0
    -/
  · intro h
    /-
      case mp
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      h✝ : Iff (Eq (Neg.neg ((CategoryTheory.shiftFunctor C 1).map T.mor₁)) 0) (Cate …
      h : Eq T.mor₁ 0
      ⊢ Eq ((CategoryTheory.shiftFunctor C 1).map T.mor₁) 0
    -/
    simp only [h, Functor.map_zero]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      h : Iff (Eq (Neg.neg ((CategoryTheory.shiftFunctor C 1).map T.mor₁)) 0) (Categ …
      ⊢ Eq ((CategoryTheory.shiftFunctor C 1).map T.mor₁) 0 → Eq T.mor₁ 0
    -/
  · intro h
    /-
      case mpr
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      h✝ : Iff (Eq (Neg.neg ((CategoryTheory.shiftFunctor C 1).map T.mor₁)) 0) (Cate …
      h : Eq ((CategoryTheory.shiftFunctor C 1).map T.mor₁) 0
      ⊢ Eq T.mor₁ 0
    -/
    rw [← (CategoryTheory.shiftFunctor C (1 : ℤ)).map_eq_zero_iff, h]
    /-
      🎉 no goals
    -/


lemma mor₃_eq_zero_of_epi₂ (h : Epi T.mor₂) : T.mor₃ = 0 := (T.mor₃_eq_zero_iff_epi₂ hT).2 h

lemma mor₂_eq_zero_of_epi₁ (h : Epi T.mor₁) : T.mor₂ = 0 := (T.mor₂_eq_zero_iff_epi₁ hT).2 h

lemma mor₁_eq_zero_of_epi₃ (h : Epi T.mor₃) : T.mor₁ = 0 := (T.mor₁_eq_zero_iff_epi₃ hT).2 h


lemma epi₂ (h : T.mor₃ = 0) : Epi T.mor₂ := (T.mor₃_eq_zero_iff_epi₂ hT).1 h

lemma epi₁ (h : T.mor₂ = 0) : Epi T.mor₁ := (T.mor₂_eq_zero_iff_epi₁ hT).1 h

lemma epi₃ (h : T.mor₁ = 0) : Epi T.mor₃ := (T.mor₁_eq_zero_iff_epi₃ hT).1 h


lemma mor₁_eq_zero_iff_mono₂ : T.mor₁ = 0 ↔ Mono T.mor₂ := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    ⊢ Iff (Eq T.mor₁ 0) (CategoryTheory.Mono T.mor₂)
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      ⊢ Eq T.mor₁ 0 → CategoryTheory.Mono T.mor₂
    -/
  · intro h
    /-
      case mp
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      h : Eq T.mor₁ 0
      ⊢ CategoryTheory.Mono T.mor₂
    -/
    rw [mono_iff_cancel_zero]
    /-
      case mp
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      h : Eq T.mor₁ 0
      ⊢ ∀ (P : C) (g : Quiver.Hom P T.obj₂), Eq (CategoryTheory.CategoryStruct.comp  …
    -/
    intro X g hg
    /-
      case mp
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      h : Eq T.mor₁ 0
      X : C
      g : Quiver.Hom X T.obj₂
      hg : Eq (CategoryTheory.CategoryStruct.comp g T.mor₂) 0
      ⊢ Eq g 0
    -/
    obtain ⟨f, rfl⟩ := coyoneda_exact₂ T hT g hg
    /-
      case mp.intro
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      h : Eq T.mor₁ 0
      X : C
      f : Quiver.Hom X T.obj₁
      hg : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp f T.mor₁) 0
    -/
    rw [h, comp_zero]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      ⊢ CategoryTheory.Mono T.mor₂ → Eq T.mor₁ 0
    -/
  · intro
    /-
      case mpr
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      a✝ : CategoryTheory.Mono T.mor₂
      ⊢ Eq T.mor₁ 0
    -/
    rw [← cancel_mono T.mor₂, comp_distTriang_mor_zero₁₂ _ hT, zero_comp]
    /-
      🎉 no goals
    -/


lemma mor₂_eq_zero_iff_mono₃ : T.mor₂ = 0 ↔ Mono T.mor₃ :=
  mor₁_eq_zero_iff_mono₂ _ (rot_of_distTriang _ hT)


lemma mor₃_eq_zero_iff_mono₁ : T.mor₃ = 0 ↔ Mono T.mor₁ := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    ⊢ Iff (Eq T.mor₃ 0) (CategoryTheory.Mono T.mor₁)
  -/
  have h := mor₁_eq_zero_iff_mono₂ _ (inv_rot_of_distTriang _ hT)
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    h : Iff (Eq T.invRotate.mor₁ 0) (CategoryTheory.Mono T.invRotate.mor₂)
    ⊢ Iff (Eq T.mor₃ 0) (CategoryTheory.Mono T.mor₁)
  -/
  dsimp at h
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    h : Iff (Eq (Neg.neg (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shif …
    ⊢ Iff (Eq T.mor₃ 0) (CategoryTheory.Mono T.mor₁)
  -/
  rw [← h, neg_eq_zero, IsIso.comp_right_eq_zero]
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    h : Iff (Eq (Neg.neg (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shif …
    ⊢ Iff (Eq T.mor₃ 0) (Eq ((CategoryTheory.shiftFunctor C (-1)).map T.mor₃) 0)
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      h : Iff (Eq (Neg.neg (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shif …
      ⊢ Eq T.mor₃ 0 → Eq ((CategoryTheory.shiftFunctor C (-1)).map T.mor₃) 0
    -/
  · intro h
    /-
      case mp
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      h✝ : Iff (Eq (Neg.neg (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shi …
      h : Eq T.mor₃ 0
      ⊢ Eq ((CategoryTheory.shiftFunctor C (-1)).map T.mor₃) 0
    -/
    simp only [h, Functor.map_zero]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      h : Iff (Eq (Neg.neg (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shif …
      ⊢ Eq ((CategoryTheory.shiftFunctor C (-1)).map T.mor₃) 0 → Eq T.mor₃ 0
    -/
  · intro h
    /-
      case mpr
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      h✝ : Iff (Eq (Neg.neg (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shi …
      h : Eq ((CategoryTheory.shiftFunctor C (-1)).map T.mor₃) 0
      ⊢ Eq T.mor₃ 0
    -/
    rw [← (CategoryTheory.shiftFunctor C (-1 : ℤ)).map_eq_zero_iff, h]
    /-
      🎉 no goals
    -/


lemma mor₁_eq_zero_of_mono₂ (h : Mono T.mor₂) : T.mor₁ = 0 := (T.mor₁_eq_zero_iff_mono₂ hT).2 h

lemma mor₂_eq_zero_of_mono₃ (h : Mono T.mor₃) : T.mor₂ = 0 := (T.mor₂_eq_zero_iff_mono₃ hT).2 h

lemma mor₃_eq_zero_of_mono₁ (h : Mono T.mor₁) : T.mor₃ = 0 := (T.mor₃_eq_zero_iff_mono₁ hT).2 h


lemma mono₂ (h : T.mor₁ = 0) : Mono T.mor₂ := (T.mor₁_eq_zero_iff_mono₂ hT).1 h

lemma mono₃ (h : T.mor₂ = 0) : Mono T.mor₃ := (T.mor₂_eq_zero_iff_mono₃ hT).1 h

lemma mono₁ (h : T.mor₃ = 0) : Mono T.mor₁ := (T.mor₃_eq_zero_iff_mono₁ hT).1 h


lemma isZero₂_iff : IsZero T.obj₂ ↔ (T.mor₁ = 0 ∧ T.mor₂ = 0) := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    ⊢ Iff (CategoryTheory.Limits.IsZero T.obj₂) (And (Eq T.mor₁ 0) (Eq T.mor₂ 0))
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      ⊢ CategoryTheory.Limits.IsZero T.obj₂ → And (Eq T.mor₁ 0) (Eq T.mor₂ 0)
    -/
  · intro h
    /-
      case mp
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      h : CategoryTheory.Limits.IsZero T.obj₂
      ⊢ And (Eq T.mor₁ 0) (Eq T.mor₂ 0)
    -/
    exact ⟨h.eq_of_tgt _ _, h.eq_of_src _ _⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      ⊢ And (Eq T.mor₁ 0) (Eq T.mor₂ 0) → CategoryTheory.Limits.IsZero T.obj₂
    -/
  · intro ⟨h₁, h₂⟩
    /-
      case mpr
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      h₁ : Eq T.mor₁ 0
      h₂ : Eq T.mor₂ 0
      ⊢ CategoryTheory.Limits.IsZero T.obj₂
    -/
    obtain ⟨f, hf⟩ := coyoneda_exact₂ T hT (𝟙 _) (by rw [h₂, comp_zero])
    /-
      case mpr.intro
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      h₁ : Eq T.mor₁ 0
      h₂ : Eq T.mor₂ 0
      f : Quiver.Hom T.obj₂ T.obj₁
      hf : Eq (CategoryTheory.CategoryStruct.id T.obj₂) (CategoryTheory.CategoryStru …
      ⊢ CategoryTheory.Limits.IsZero T.obj₂
    -/
    rw [IsZero.iff_id_eq_zero, hf, h₁, comp_zero]
    /-
      🎉 no goals
    -/


lemma isZero₁_iff : IsZero T.obj₁ ↔ (T.mor₁ = 0 ∧ T.mor₃ = 0) := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    ⊢ Iff (CategoryTheory.Limits.IsZero T.obj₁) (And (Eq T.mor₁ 0) (Eq T.mor₃ 0))
  -/
  refine (isZero₂_iff _ (inv_rot_of_distTriang _ hT)).trans ?_
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    ⊢ Iff (And (Eq T.invRotate.mor₁ 0) (Eq T.invRotate.mor₂ 0)) (And (Eq T.mor₁ 0) …
  -/
  dsimp
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    ⊢ Iff (And (Eq (Neg.neg (CategoryTheory.CategoryStruct.comp ((CategoryTheory.s …
  -/
  simp only [neg_eq_zero, IsIso.comp_right_eq_zero, Functor.map_eq_zero_iff]
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    ⊢ Iff (And (Eq T.mor₃ 0) (Eq T.mor₁ 0)) (And (Eq T.mor₁ 0) (Eq T.mor₃ 0))
  -/
  tauto
  /-
    🎉 no goals
  -/


lemma isZero₃_iff : IsZero T.obj₃ ↔ (T.mor₂ = 0 ∧ T.mor₃ = 0) := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    ⊢ Iff (CategoryTheory.Limits.IsZero T.obj₃) (And (Eq T.mor₂ 0) (Eq T.mor₃ 0))
  -/
  refine (isZero₂_iff _ (rot_of_distTriang _ hT)).trans ?_
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    ⊢ Iff (And (Eq T.rotate.mor₁ 0) (Eq T.rotate.mor₂ 0)) (And (Eq T.mor₂ 0) (Eq T …
  -/
  dsimp
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    ⊢ Iff (And (Eq T.mor₂ 0) (Eq T.mor₃ 0)) (And (Eq T.mor₂ 0) (Eq T.mor₃ 0))
  -/
  tauto
  /-
    🎉 no goals
  -/


lemma isZero₁_of_isZero₂₃ (h₂ : IsZero T.obj₂) (h₃ : IsZero T.obj₃) : IsZero T.obj₁ := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    h₂ : CategoryTheory.Limits.IsZero T.obj₂
    h₃ : CategoryTheory.Limits.IsZero T.obj₃
    ⊢ CategoryTheory.Limits.IsZero T.obj₁
  -/
  rw [T.isZero₁_iff hT]
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    h₂ : CategoryTheory.Limits.IsZero T.obj₂
    h₃ : CategoryTheory.Limits.IsZero T.obj₃
    ⊢ And (Eq T.mor₁ 0) (Eq T.mor₃ 0)
  -/
  exact ⟨h₂.eq_of_tgt _ _, h₃.eq_of_src _ _⟩
  /-
    🎉 no goals
  -/


lemma isZero₂_of_isZero₁₃ (h₁ : IsZero T.obj₁) (h₃ : IsZero T.obj₃) : IsZero T.obj₂ := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    h₁ : CategoryTheory.Limits.IsZero T.obj₁
    h₃ : CategoryTheory.Limits.IsZero T.obj₃
    ⊢ CategoryTheory.Limits.IsZero T.obj₂
  -/
  rw [T.isZero₂_iff hT]
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    h₁ : CategoryTheory.Limits.IsZero T.obj₁
    h₃ : CategoryTheory.Limits.IsZero T.obj₃
    ⊢ And (Eq T.mor₁ 0) (Eq T.mor₂ 0)
  -/
  exact ⟨h₁.eq_of_src _ _, h₃.eq_of_tgt _ _⟩
  /-
    🎉 no goals
  -/


lemma isZero₃_of_isZero₁₂ (h₁ : IsZero T.obj₁) (h₂ : IsZero T.obj₂) : IsZero T.obj₃ :=
  isZero₂_of_isZero₁₃ _ (rot_of_distTriang _ hT) h₂ (by
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      h₁ : CategoryTheory.Limits.IsZero T.obj₁
      h₂ : CategoryTheory.Limits.IsZero T.obj₂
      ⊢ CategoryTheory.Limits.IsZero T.rotate.obj₃
    -/
    dsimp
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      h₁ : CategoryTheory.Limits.IsZero T.obj₁
      h₂ : CategoryTheory.Limits.IsZero T.obj₂
      ⊢ CategoryTheory.Limits.IsZero ((CategoryTheory.shiftFunctor C 1).obj T.obj₁)
    -/
    simp only [IsZero.iff_id_eq_zero] at h₁ ⊢
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      h₂ : CategoryTheory.Limits.IsZero T.obj₂
      h₁ : Eq (CategoryTheory.CategoryStruct.id T.obj₁) 0
      ⊢ Eq (CategoryTheory.CategoryStruct.id ((CategoryTheory.shiftFunctor C 1).obj  …
    -/
    rw [← Functor.map_id, h₁, Functor.map_zero])
    /-
      🎉 no goals
    -/


lemma isZero₁_iff_isIso₂ :
    IsZero T.obj₁ ↔ IsIso T.mor₂ := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    ⊢ Iff (CategoryTheory.Limits.IsZero T.obj₁) (CategoryTheory.IsIso T.mor₂)
  -/
  rw [T.isZero₁_iff hT]
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    ⊢ Iff (And (Eq T.mor₁ 0) (Eq T.mor₃ 0)) (CategoryTheory.IsIso T.mor₂)
  -/
  constructor
    /-
      case mp
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      ⊢ And (Eq T.mor₁ 0) (Eq T.mor₃ 0) → CategoryTheory.IsIso T.mor₂
    -/
  · intro ⟨h₁, h₃⟩
    /-
      case mp
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      h₁ : Eq T.mor₁ 0
      h₃ : Eq T.mor₃ 0
      ⊢ CategoryTheory.IsIso T.mor₂
    -/
    have := T.epi₂ hT h₃
    /-
      case mp
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      h₁ : Eq T.mor₁ 0
      h₃ : Eq T.mor₃ 0
      this : CategoryTheory.Epi T.mor₂
      ⊢ CategoryTheory.IsIso T.mor₂
    -/
    obtain ⟨f, hf⟩ := yoneda_exact₂ T hT (𝟙 _) (by rw [h₁, zero_comp])
    /-
      case mp.intro
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      h₁ : Eq T.mor₁ 0
      h₃ : Eq T.mor₃ 0
      this : CategoryTheory.Epi T.mor₂
      f : Quiver.Hom T.obj₃ T.obj₂
      hf : Eq (CategoryTheory.CategoryStruct.id T.obj₂) (CategoryTheory.CategoryStru …
      ⊢ CategoryTheory.IsIso T.mor₂
    -/
    exact ⟨f, hf.symm, by rw [← cancel_epi T.mor₂, comp_id, ← reassoc_of% hf]⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      ⊢ CategoryTheory.IsIso T.mor₂ → And (Eq T.mor₁ 0) (Eq T.mor₃ 0)
    -/
  · intro
    /-
      case mpr
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      a✝ : CategoryTheory.IsIso T.mor₂
      ⊢ And (Eq T.mor₁ 0) (Eq T.mor₃ 0)
    -/
    rw [T.mor₁_eq_zero_iff_mono₂ hT, T.mor₃_eq_zero_iff_epi₂ hT]
    /-
      case mpr
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      a✝ : CategoryTheory.IsIso T.mor₂
      ⊢ And (CategoryTheory.Mono T.mor₂) (CategoryTheory.Epi T.mor₂)
    -/
                    /-
                      🎉 no goals
                    -/
    constructor <;> infer_instance
                    /-
                      🎉 no goals
                    -/


lemma isZero₂_iff_isIso₃ : IsZero T.obj₂ ↔ IsIso T.mor₃ :=
  isZero₁_iff_isIso₂ _ (rot_of_distTriang _ hT)


lemma isZero₃_iff_isIso₁ : IsZero T.obj₃ ↔ IsIso T.mor₁ := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    ⊢ Iff (CategoryTheory.Limits.IsZero T.obj₃) (CategoryTheory.IsIso T.mor₁)
  -/
  refine Iff.trans ?_ (Triangle.isZero₁_iff_isIso₂ _ (inv_rot_of_distTriang _ hT))
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    ⊢ Iff (CategoryTheory.Limits.IsZero T.obj₃) (CategoryTheory.Limits.IsZero T.in …
  -/
  dsimp
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    ⊢ Iff (CategoryTheory.Limits.IsZero T.obj₃) (CategoryTheory.Limits.IsZero ((Ca …
  -/
  simp only [IsZero.iff_id_eq_zero, ← Functor.map_id, Functor.map_eq_zero_iff]
  /-
    🎉 no goals
  -/


lemma isZero₁_of_isIso₂ (h : IsIso T.mor₂) : IsZero T.obj₁ := (T.isZero₁_iff_isIso₂ hT).2 h

lemma isZero₂_of_isIso₃ (h : IsIso T.mor₃) : IsZero T.obj₂ := (T.isZero₂_iff_isIso₃ hT).2 h

lemma isZero₃_of_isIso₁ (h : IsIso T.mor₁) : IsZero T.obj₃ := (T.isZero₃_iff_isIso₁ hT).2 h


lemma shift_distinguished (n : ℤ) :
    (CategoryTheory.shiftFunctor (Triangle C) n).obj T ∈ distTriang C := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    n : Int
    ⊢ Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles ((Categ …
  -/
  revert T hT
  let H : ℤ → Prop := fun n => ∀ (T : Triangle C) (_ : T ∈ distTriang C),
    (Triangle.shiftFunctor C n).obj T ∈ distTriang C
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    n : Int
    H : Int → Prop := fun n => ∀ (T : CategoryTheory.Pretriangulated.Triangle C),  …
    ⊢ ∀ (T : CategoryTheory.Pretriangulated.Triangle C), Membership.mem CategoryTh …
  -/
  change H n
  have H_zero : H 0 := fun T hT =>
    isomorphic_distinguished _ hT _ ((Triangle.shiftFunctorZero C).app T)
  have H_one : H 1 := fun T hT =>
    isomorphic_distinguished _ (rot_of_distTriang _
      (rot_of_distTriang _ (rot_of_distTriang _ hT))) _
        ((rotateRotateRotateIso C).symm.app T)
  have H_neg_one : H (-1) := fun T hT =>
    isomorphic_distinguished _ (inv_rot_of_distTriang _
      (inv_rot_of_distTriang _ (inv_rot_of_distTriang _ hT))) _
        ((invRotateInvRotateInvRotateIso C).symm.app T)
  have H_add : ∀ {a b c : ℤ}, H a → H b → a + b = c → H c := fun {a b c} ha hb hc T hT =>
    isomorphic_distinguished _ (hb _ (ha _ hT)) _
      ((Triangle.shiftFunctorAdd' C _ _ _ hc).app T)
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    n : Int
    H : Int → Prop := fun n => ∀ (T : CategoryTheory.Pretriangulated.Triangle C),  …
    H_zero : H 0
    H_one : H 1
    H_neg_one : H (-1)
    H_add : ∀ {a b c : Int}, H a → H b → Eq (HAdd.hAdd a b) c → H c
    ⊢ H n
  -/
  obtain (n|n) := n
  · induction n with
    | zero =>  exact H_zero
    | succ n hn => exact H_add hn H_one rfl
  · induction n with
    | zero => exact H_neg_one
    | succ n hn => exact H_add hn H_neg_one rfl


instance : SplitEpiCategory C where
  isSplitEpi_of_epi f hf := by
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      X✝ Y✝ : C
      f : Quiver.Hom X✝ Y✝
      hf : CategoryTheory.Epi f
      ⊢ CategoryTheory.IsSplitEpi f
    -/
    obtain ⟨Z, g, h, hT⟩ := distinguished_cocone_triangle f
    obtain ⟨r, hr⟩ := Triangle.coyoneda_exact₂ _ hT (𝟙 _)
      (by rw [Triangle.mor₂_eq_zero_of_epi₁ _ hT hf, comp_zero])
    /-
      case intro.intro.intro.intro
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      X✝ Y✝ : C
      f : Quiver.Hom X✝ Y✝
      hf : CategoryTheory.Epi f
      Z : C
      g : Quiver.Hom Y✝ Z
      h : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj X✝)
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Cat …
      r : Quiver.Hom (CategoryTheory.Pretriangulated.Triangle.mk f g h).obj₂ (Catego …
      hr : Eq (CategoryTheory.CategoryStruct.id (CategoryTheory.Pretriangulated.Tria …
      ⊢ CategoryTheory.IsSplitEpi f
    -/
    exact ⟨r, hr.symm⟩
    /-
      🎉 no goals
    -/


instance : SplitMonoCategory C where
  isSplitMono_of_mono f hf := by
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      X✝ Y✝ : C
      f : Quiver.Hom X✝ Y✝
      hf : CategoryTheory.Mono f
      ⊢ CategoryTheory.IsSplitMono f
    -/
    obtain ⟨X, g, h, hT⟩ := distinguished_cocone_triangle₁ f
    obtain ⟨r, hr⟩ := Triangle.yoneda_exact₂ _ hT (𝟙 _) (by
      rw [Triangle.mor₁_eq_zero_of_mono₂ _ hT hf, zero_comp])
    /-
      case intro.intro.intro.intro
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      X✝ Y✝ : C
      f : Quiver.Hom X✝ Y✝
      hf : CategoryTheory.Mono f
      X : C
      g : Quiver.Hom X X✝
      h : Quiver.Hom Y✝ ((CategoryTheory.shiftFunctor C 1).obj X)
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Cat …
      r : Quiver.Hom (CategoryTheory.Pretriangulated.Triangle.mk g f h).obj₃ (Catego …
      hr : Eq (CategoryTheory.CategoryStruct.id (CategoryTheory.Pretriangulated.Tria …
      ⊢ CategoryTheory.IsSplitMono f
    -/
    exact ⟨r, hr.symm⟩
    /-
      🎉 no goals
    -/


lemma isIso₂_of_isIso₁₃ {T T' : Triangle C} (φ : T ⟶ T') (hT : T ∈ distTriang C)
    (hT' : T' ∈ distTriang C) (h₁ : IsIso φ.hom₁) (h₃ : IsIso φ.hom₃) : IsIso φ.hom₂ := by
  have : Mono φ.hom₂ := by
    rw [mono_iff_cancel_zero]
    intro A f hf
    obtain ⟨g, rfl⟩ := Triangle.coyoneda_exact₂ _ hT f
      (by rw [← cancel_mono φ.hom₃, assoc, φ.comm₂, reassoc_of% hf, zero_comp, zero_comp])
    rw [assoc] at hf
    obtain ⟨h, hh⟩ := Triangle.coyoneda_exact₂ T'.invRotate (inv_rot_of_distTriang _ hT')
      (g ≫ φ.hom₁) (by dsimp; rw [assoc, ← φ.comm₁, hf])
    obtain ⟨k, rfl⟩ : ∃ (k : A ⟶ T.invRotate.obj₁), k ≫ T.invRotate.mor₁ = g := by
      refine ⟨h ≫ inv (φ.hom₃⟦(-1 : ℤ)⟧'), ?_⟩
      have eq := ((invRotate C).map φ).comm₁
      dsimp only [invRotate] at eq
      rw [← cancel_mono φ.hom₁, assoc, assoc, eq, IsIso.inv_hom_id_assoc, hh]
    erw [assoc, comp_distTriang_mor_zero₁₂ _ (inv_rot_of_distTriang _ hT), comp_zero]
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T T' : CategoryTheory.Pretriangulated.Triangle C
    φ : Quiver.Hom T T'
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    hT' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T'
    h₁ : CategoryTheory.IsIso φ.hom₁
    h₃ : CategoryTheory.IsIso φ.hom₃
    this : CategoryTheory.Mono φ.hom₂
    ⊢ CategoryTheory.IsIso φ.hom₂
  -/
  refine isIso_of_yoneda_map_bijective _ (fun A => ⟨?_, ?_⟩)
    /-
      case refine_1
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T T' : CategoryTheory.Pretriangulated.Triangle C
      φ : Quiver.Hom T T'
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      hT' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T'
      h₁ : CategoryTheory.IsIso φ.hom₁
      h₃ : CategoryTheory.IsIso φ.hom₃
      this : CategoryTheory.Mono φ.hom₂
      A : C
      ⊢ Function.Injective fun x => CategoryTheory.CategoryStruct.comp x φ.hom₂
    -/
  · intro f₁ f₂ h
    /-
      case refine_1
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T T' : CategoryTheory.Pretriangulated.Triangle C
      φ : Quiver.Hom T T'
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      hT' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T'
      h₁ : CategoryTheory.IsIso φ.hom₁
      h₃ : CategoryTheory.IsIso φ.hom₃
      this : CategoryTheory.Mono φ.hom₂
      A : C
      f₁ f₂ : Quiver.Hom A T.obj₂
      h : Eq ((fun x => CategoryTheory.CategoryStruct.comp x φ.hom₂) f₁) ((fun x =>  …
      ⊢ Eq f₁ f₂
    -/
    simpa only [← cancel_mono φ.hom₂] using h
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T T' : CategoryTheory.Pretriangulated.Triangle C
      φ : Quiver.Hom T T'
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      hT' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T'
      h₁ : CategoryTheory.IsIso φ.hom₁
      h₃ : CategoryTheory.IsIso φ.hom₃
      this : CategoryTheory.Mono φ.hom₂
      A : C
      ⊢ Function.Surjective fun x => CategoryTheory.CategoryStruct.comp x φ.hom₂
    -/
  · intro y₂
    obtain ⟨x₃, hx₃⟩ : ∃ (x₃ : A ⟶ T.obj₃), x₃ ≫ φ.hom₃ = y₂ ≫ T'.mor₂ :=
      ⟨y₂ ≫ T'.mor₂ ≫ inv φ.hom₃, by simp⟩
    obtain ⟨x₂, hx₂⟩ := Triangle.coyoneda_exact₃ _ hT x₃
      (by rw [← cancel_mono (φ.hom₁⟦(1 : ℤ)⟧'), assoc, zero_comp, φ.comm₃, reassoc_of% hx₃,
        comp_distTriang_mor_zero₂₃ _ hT', comp_zero])
    obtain ⟨y₁, hy₁⟩ := Triangle.coyoneda_exact₂ _ hT' (y₂ - x₂ ≫ φ.hom₂)
      (by rw [sub_comp, assoc, ← φ.comm₂, ← reassoc_of% hx₂, hx₃, sub_self])
    /-
      case refine_2.intro.intro.intro
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T T' : CategoryTheory.Pretriangulated.Triangle C
      φ : Quiver.Hom T T'
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      hT' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T'
      h₁ : CategoryTheory.IsIso φ.hom₁
      h₃ : CategoryTheory.IsIso φ.hom₃
      this : CategoryTheory.Mono φ.hom₂
      A : C
      y₂ : Quiver.Hom A T'.obj₂
      x₃ : Quiver.Hom A T.obj₃
      hx₃ : Eq (CategoryTheory.CategoryStruct.comp x₃ φ.hom₃) (CategoryTheory.Catego …
      x₂ : Quiver.Hom A T.obj₂
      hx₂ : Eq x₃ (CategoryTheory.CategoryStruct.comp x₂ T.mor₂)
      y₁ : Quiver.Hom A T'.obj₁
      hy₁ : Eq (HSub.hSub y₂ (CategoryTheory.CategoryStruct.comp x₂ φ.hom₂)) (Catego …
      ⊢ Exists fun a => Eq ((fun x => CategoryTheory.CategoryStruct.comp x φ.hom₂) a …
    -/
    obtain ⟨x₁, hx₁⟩ : ∃ (x₁ : A ⟶ T.obj₁), x₁ ≫ φ.hom₁ = y₁ := ⟨y₁ ≫ inv φ.hom₁, by simp⟩
    /-
      case refine_2.intro.intro.intro.intro
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T T' : CategoryTheory.Pretriangulated.Triangle C
      φ : Quiver.Hom T T'
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      hT' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T'
      h₁ : CategoryTheory.IsIso φ.hom₁
      h₃ : CategoryTheory.IsIso φ.hom₃
      this : CategoryTheory.Mono φ.hom₂
      A : C
      y₂ : Quiver.Hom A T'.obj₂
      x₃ : Quiver.Hom A T.obj₃
      hx₃ : Eq (CategoryTheory.CategoryStruct.comp x₃ φ.hom₃) (CategoryTheory.Catego …
      x₂ : Quiver.Hom A T.obj₂
      hx₂ : Eq x₃ (CategoryTheory.CategoryStruct.comp x₂ T.mor₂)
      y₁ : Quiver.Hom A T'.obj₁
      hy₁ : Eq (HSub.hSub y₂ (CategoryTheory.CategoryStruct.comp x₂ φ.hom₂)) (Catego …
      x₁ : Quiver.Hom A T.obj₁
      hx₁ : Eq (CategoryTheory.CategoryStruct.comp x₁ φ.hom₁) y₁
      ⊢ Exists fun a => Eq ((fun x => CategoryTheory.CategoryStruct.comp x φ.hom₂) a …
    -/
    refine ⟨x₂ + x₁ ≫ T.mor₁, ?_⟩
    /-
      case refine_2.intro.intro.intro.intro
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T T' : CategoryTheory.Pretriangulated.Triangle C
      φ : Quiver.Hom T T'
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      hT' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T'
      h₁ : CategoryTheory.IsIso φ.hom₁
      h₃ : CategoryTheory.IsIso φ.hom₃
      this : CategoryTheory.Mono φ.hom₂
      A : C
      y₂ : Quiver.Hom A T'.obj₂
      x₃ : Quiver.Hom A T.obj₃
      hx₃ : Eq (CategoryTheory.CategoryStruct.comp x₃ φ.hom₃) (CategoryTheory.Catego …
      x₂ : Quiver.Hom A T.obj₂
      hx₂ : Eq x₃ (CategoryTheory.CategoryStruct.comp x₂ T.mor₂)
      y₁ : Quiver.Hom A T'.obj₁
      hy₁ : Eq (HSub.hSub y₂ (CategoryTheory.CategoryStruct.comp x₂ φ.hom₂)) (Catego …
      x₁ : Quiver.Hom A T.obj₁
      hx₁ : Eq (CategoryTheory.CategoryStruct.comp x₁ φ.hom₁) y₁
      ⊢ Eq ((fun x => CategoryTheory.CategoryStruct.comp x φ.hom₂) (HAdd.hAdd x₂ (Ca …
    -/
    dsimp
    /-
      case refine_2.intro.intro.intro.intro
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T T' : CategoryTheory.Pretriangulated.Triangle C
      φ : Quiver.Hom T T'
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      hT' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T'
      h₁ : CategoryTheory.IsIso φ.hom₁
      h₃ : CategoryTheory.IsIso φ.hom₃
      this : CategoryTheory.Mono φ.hom₂
      A : C
      y₂ : Quiver.Hom A T'.obj₂
      x₃ : Quiver.Hom A T.obj₃
      hx₃ : Eq (CategoryTheory.CategoryStruct.comp x₃ φ.hom₃) (CategoryTheory.Catego …
      x₂ : Quiver.Hom A T.obj₂
      hx₂ : Eq x₃ (CategoryTheory.CategoryStruct.comp x₂ T.mor₂)
      y₁ : Quiver.Hom A T'.obj₁
      hy₁ : Eq (HSub.hSub y₂ (CategoryTheory.CategoryStruct.comp x₂ φ.hom₂)) (Catego …
      x₁ : Quiver.Hom A T.obj₁
      hx₁ : Eq (CategoryTheory.CategoryStruct.comp x₁ φ.hom₁) y₁
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HAdd.hAdd x₂ (CategoryTheory.Categor …
    -/
    rw [add_comp, assoc, φ.comm₁, reassoc_of% hx₁, ← hy₁, add_sub_cancel]
    /-
      🎉 no goals
    -/


lemma isIso₃_of_isIso₁₂ {T T' : Triangle C} (φ : T ⟶ T') (hT : T ∈ distTriang C)
    (hT' : T' ∈ distTriang C) (h₁ : IsIso φ.hom₁) (h₂ : IsIso φ.hom₂) : IsIso φ.hom₃ :=
  isIso₂_of_isIso₁₃ ((rotate C).map φ) (rot_of_distTriang _ hT)
                                     /-
                                       C : Type u
                                       inst✝⁴ : CategoryTheory.Category.{v, u} C
                                       inst✝³ : CategoryTheory.Limits.HasZeroObject C
                                       inst✝² : CategoryTheory.HasShift C Int
                                       inst✝¹ : CategoryTheory.Preadditive C
                                       inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
                                       hC : CategoryTheory.Pretriangulated C
                                       T T' : CategoryTheory.Pretriangulated.Triangle C
                                       φ : Quiver.Hom T T'
                                       hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
                                       hT' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T'
                                       h₁ : CategoryTheory.IsIso φ.hom₁
                                       h₂ : CategoryTheory.IsIso φ.hom₂
                                       ⊢ CategoryTheory.IsIso ((CategoryTheory.Pretriangulated.rotate C).map φ).hom₃
                                     -/
    (rot_of_distTriang _ hT') h₂ (by dsimp; infer_instance)
                                            /-
                                              🎉 no goals
                                            -/


lemma isIso₁_of_isIso₂₃ {T T' : Triangle C} (φ : T ⟶ T') (hT : T ∈ distTriang C)
    (hT' : T' ∈ distTriang C) (h₂ : IsIso φ.hom₂) (h₃ : IsIso φ.hom₃) : IsIso φ.hom₁ :=
  isIso₂_of_isIso₁₃ ((invRotate C).map φ) (inv_rot_of_distTriang _ hT)
                                      /-
                                        C : Type u
                                        inst✝⁴ : CategoryTheory.Category.{v, u} C
                                        inst✝³ : CategoryTheory.Limits.HasZeroObject C
                                        inst✝² : CategoryTheory.HasShift C Int
                                        inst✝¹ : CategoryTheory.Preadditive C
                                        inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
                                        hC : CategoryTheory.Pretriangulated C
                                        T T' : CategoryTheory.Pretriangulated.Triangle C
                                        φ : Quiver.Hom T T'
                                        hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
                                        hT' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T'
                                        h₂ : CategoryTheory.IsIso φ.hom₂
                                        h₃ : CategoryTheory.IsIso φ.hom₃
                                        ⊢ CategoryTheory.IsIso ((CategoryTheory.Pretriangulated.invRotate C).map φ).hom₁
                                      -/
                                             /-
                                               🎉 no goals
                                             -/
    (inv_rot_of_distTriang _ hT') (by dsimp; infer_instance) (by dsimp; infer_instance)
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


/-- Given a distinguished triangle `T` such that `T.mor₃ = 0` and the datum of morphisms
`inr : T.obj₃ ⟶ T.obj₂` and `fst : T.obj₂ ⟶ T.obj₁` satisfying suitable relations, this
is the binary biproduct data expressing that `T.obj₂` identifies to the binary
biproduct of `T.obj₁` and `T.obj₃`.
See also `exists_iso_binaryBiproduct_of_distTriang`. -/
@[simps]
def binaryBiproductData (T : Triangle C) (hT : T ∈ distTriang C) (hT₀ : T.mor₃ = 0)
    (inr : T.obj₃ ⟶ T.obj₂) (inr_snd : inr ≫ T.mor₂ = 𝟙 _) (fst : T.obj₂ ⟶ T.obj₁)
    (total : fst ≫ T.mor₁ + T.mor₂ ≫ inr = 𝟙 T.obj₂) :
    BinaryBiproductData T.obj₁ T.obj₃ := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    hT₀ : Eq T.mor₃ 0
    inr : Quiver.Hom T.obj₃ T.obj₂
    inr_snd : Eq (CategoryTheory.CategoryStruct.comp inr T.mor₂) (CategoryTheory.C …
    fst : Quiver.Hom T.obj₂ T.obj₁
    total : Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp fst T.mor₁) (Categor …
    ⊢ CategoryTheory.Limits.BinaryBiproductData T.obj₁ T.obj₃
  -/
  have : Mono T.mor₁ := T.mono₁ hT hT₀
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    hT₀ : Eq T.mor₃ 0
    inr : Quiver.Hom T.obj₃ T.obj₂
    inr_snd : Eq (CategoryTheory.CategoryStruct.comp inr T.mor₂) (CategoryTheory.C …
    fst : Quiver.Hom T.obj₂ T.obj₁
    total : Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp fst T.mor₁) (Categor …
    this : CategoryTheory.Mono T.mor₁
    ⊢ CategoryTheory.Limits.BinaryBiproductData T.obj₁ T.obj₃
  -/
  have eq : fst ≫ T.mor₁ = 𝟙 T.obj₂ - T.mor₂ ≫ inr := by rw [← total, add_sub_cancel_right]
  exact
    { bicone :=
      { pt := T.obj₂
        fst := fst
        snd := T.mor₂
        inl := T.mor₁
        inr := inr
        inl_fst := by
          simp only [← cancel_mono T.mor₁, assoc, id_comp, eq, comp_sub, comp_id,
            comp_distTriang_mor_zero₁₂_assoc _ hT, zero_comp, sub_zero]
        inl_snd := comp_distTriang_mor_zero₁₂ _ hT
        inr_fst := by
          simp only [← cancel_mono T.mor₁, assoc, eq, comp_sub, reassoc_of% inr_snd,
            comp_id, sub_self, zero_comp]
        inr_snd := inr_snd }
      isBilimit := isBinaryBilimitOfTotal _ total }


instance : HasBinaryBiproducts C := ⟨fun X₁ X₃ => by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    X₁ X₃ : C
    ⊢ CategoryTheory.Limits.HasBinaryBiproduct X₁ X₃
  -/
  obtain ⟨X₂, inl, snd, mem⟩ := distinguished_cocone_triangle₂ (0 : X₃ ⟶ X₁⟦(1 : ℤ)⟧)
  obtain ⟨inr : X₃ ⟶ X₂, inr_snd : 𝟙 _ = inr ≫ snd⟩ :=
    Triangle.coyoneda_exact₃ _ mem (𝟙 X₃) (by simp)
  obtain ⟨fst : X₂ ⟶ X₁, hfst : 𝟙 X₂ - snd ≫ inr = fst ≫ inl⟩ :=
    Triangle.coyoneda_exact₂ _ mem (𝟙 X₂ - snd ≫ inr) (by
      dsimp
      simp only [sub_comp, assoc, id_comp, ← inr_snd, comp_id, sub_self])
  /-
    case intro.intro.intro.intro.intro
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    X₁ X₃ X₂ : C
    inl : Quiver.Hom X₁ X₂
    snd : Quiver.Hom X₂ X₃
    mem : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    inr : Quiver.Hom X₃ X₂
    inr_snd : Eq (CategoryTheory.CategoryStruct.id X₃) (CategoryTheory.CategoryStr …
    fst : Quiver.Hom X₂ X₁
    hfst : Eq (HSub.hSub (CategoryTheory.CategoryStruct.id X₂) (CategoryTheory.Cat …
    ⊢ CategoryTheory.Limits.HasBinaryBiproduct X₁ X₃
  -/
  refine ⟨⟨binaryBiproductData _ mem rfl inr inr_snd.symm fst ?_⟩⟩
  /-
    case intro.intro.intro.intro.intro
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    X₁ X₃ X₂ : C
    inl : Quiver.Hom X₁ X₂
    snd : Quiver.Hom X₂ X₃
    mem : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    inr : Quiver.Hom X₃ X₂
    inr_snd : Eq (CategoryTheory.CategoryStruct.id X₃) (CategoryTheory.CategoryStr …
    fst : Quiver.Hom X₂ X₁
    hfst : Eq (HSub.hSub (CategoryTheory.CategoryStruct.id X₂) (CategoryTheory.Cat …
    ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp fst (CategoryTheory.Pretri …
  -/
  dsimp
  /-
    case intro.intro.intro.intro.intro
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    X₁ X₃ X₂ : C
    inl : Quiver.Hom X₁ X₂
    snd : Quiver.Hom X₂ X₃
    mem : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    inr : Quiver.Hom X₃ X₂
    inr_snd : Eq (CategoryTheory.CategoryStruct.id X₃) (CategoryTheory.CategoryStr …
    fst : Quiver.Hom X₂ X₁
    hfst : Eq (HSub.hSub (CategoryTheory.CategoryStruct.id X₂) (CategoryTheory.Cat …
    ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp fst inl) (CategoryTheory.C …
  -/
  simp only [← hfst, sub_add_cancel]⟩
  /-
    🎉 no goals
  -/


instance : HasFiniteProducts C := hasFiniteProducts_of_has_binary_and_terminal

instance : HasFiniteCoproducts C := hasFiniteCoproducts_of_has_binary_and_initial

instance : HasFiniteBiproducts C := HasFiniteBiproducts.of_hasFiniteProducts


lemma exists_iso_binaryBiproduct_of_distTriang (T : Triangle C) (hT : T ∈ distTriang C)
    (zero : T.mor₃ = 0) :
    ∃ (e : T.obj₂ ≅ T.obj₁ ⊞ T.obj₃), T.mor₁ ≫ e.hom = biprod.inl ∧
      T.mor₂ = e.hom ≫ biprod.snd := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    zero : Eq T.mor₃ 0
    ⊢ Exists fun e => And (Eq (CategoryTheory.CategoryStruct.comp T.mor₁ e.hom) Ca …
  -/
  have := T.epi₂ hT zero
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    zero : Eq T.mor₃ 0
    this : CategoryTheory.Epi T.mor₂
    ⊢ Exists fun e => And (Eq (CategoryTheory.CategoryStruct.comp T.mor₁ e.hom) Ca …
  -/
  have := isSplitEpi_of_epi T.mor₂
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    zero : Eq T.mor₃ 0
    this✝ : CategoryTheory.Epi T.mor₂
    this : CategoryTheory.IsSplitEpi T.mor₂
    ⊢ Exists fun e => And (Eq (CategoryTheory.CategoryStruct.comp T.mor₁ e.hom) Ca …
  -/
  obtain ⟨fst, hfst⟩ := T.coyoneda_exact₂ hT (𝟙 T.obj₂ - T.mor₂ ≫ section_ T.mor₂) (by simp)
  let d := binaryBiproductData _ hT zero (section_ T.mor₂) (by simp) fst
    (by simp only [← hfst, sub_add_cancel])
  /-
    case intro
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    zero : Eq T.mor₃ 0
    this✝ : CategoryTheory.Epi T.mor₂
    this : CategoryTheory.IsSplitEpi T.mor₂
    fst : Quiver.Hom T.obj₂ T.obj₁
    hfst : Eq (HSub.hSub (CategoryTheory.CategoryStruct.id T.obj₂) (CategoryTheory …
    d : CategoryTheory.Limits.BinaryBiproductData T.obj₁ T.obj₃ := CategoryTheory. …
    ⊢ Exists fun e => And (Eq (CategoryTheory.CategoryStruct.comp T.mor₁ e.hom) Ca …
  -/
  refine ⟨biprod.uniqueUpToIso _ _ d.isBilimit, ⟨?_, by simp [d]⟩⟩
  /-
    case intro
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    zero : Eq T.mor₃ 0
    this✝ : CategoryTheory.Epi T.mor₂
    this : CategoryTheory.IsSplitEpi T.mor₂
    fst : Quiver.Hom T.obj₂ T.obj₁
    hfst : Eq (HSub.hSub (CategoryTheory.CategoryStruct.id T.obj₂) (CategoryTheory …
    d : CategoryTheory.Limits.BinaryBiproductData T.obj₁ T.obj₃ := CategoryTheory. …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp T.mor₁ (CategoryTheory.Limits.biprod. …
  -/
  ext
    /-
      case intro.h₀
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      zero : Eq T.mor₃ 0
      this✝ : CategoryTheory.Epi T.mor₂
      this : CategoryTheory.IsSplitEpi T.mor₂
      fst : Quiver.Hom T.obj₂ T.obj₁
      hfst : Eq (HSub.hSub (CategoryTheory.CategoryStruct.id T.obj₂) (CategoryTheory …
      d : CategoryTheory.Limits.BinaryBiproductData T.obj₁ T.obj₃ := CategoryTheory. …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp T …
    -/
  · simpa [d] using d.bicone.inl_fst
    /-
      🎉 no goals
    -/
    /-
      case intro.h₁
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : CategoryTheory.Preadditive C
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      zero : Eq T.mor₃ 0
      this✝ : CategoryTheory.Epi T.mor₂
      this : CategoryTheory.IsSplitEpi T.mor₂
      fst : Quiver.Hom T.obj₂ T.obj₁
      hfst : Eq (HSub.hSub (CategoryTheory.CategoryStruct.id T.obj₂) (CategoryTheory …
      d : CategoryTheory.Limits.BinaryBiproductData T.obj₁ T.obj₃ := CategoryTheory. …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp T …
    -/
  · simpa [d] using d.bicone.inl_snd
    /-
      🎉 no goals
    -/


lemma binaryBiproductTriangle_distinguished (X₁ X₂ : C) :
    binaryBiproductTriangle X₁ X₂ ∈ distTriang C := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    X₁ X₂ : C
    ⊢ Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Catego …
  -/
  obtain ⟨Y, g, h, mem⟩ := distinguished_cocone_triangle₂ (0 : X₂ ⟶ X₁⟦(1 : ℤ)⟧)
  /-
    case intro.intro.intro
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    X₁ X₂ Y : C
    g : Quiver.Hom X₁ Y
    h : Quiver.Hom Y X₂
    mem : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    ⊢ Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Catego …
  -/
  obtain ⟨e, ⟨he₁, he₂⟩⟩ := exists_iso_binaryBiproduct_of_distTriang _ mem rfl
  /-
    case intro.intro.intro.intro.intro
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    X₁ X₂ Y : C
    g : Quiver.Hom X₁ Y
    h : Quiver.Hom Y X₂
    mem : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    e : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk g h 0).obj₂ …
    he₁ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated.T …
    he₂ : Eq (CategoryTheory.Pretriangulated.Triangle.mk g h 0).mor₂ (CategoryTheo …
    ⊢ Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Catego …
  -/
  dsimp at he₁ he₂
  /-
    case intro.intro.intro.intro.intro
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    X₁ X₂ Y : C
    g : Quiver.Hom X₁ Y
    h : Quiver.Hom Y X₂
    mem : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    e : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk g h 0).obj₂ …
    he₁ : Eq (CategoryTheory.CategoryStruct.comp g e.hom) CategoryTheory.Limits.bi …
    he₂ : Eq h (CategoryTheory.CategoryStruct.comp e.hom CategoryTheory.Limits.bip …
    ⊢ Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Catego …
  -/
  refine isomorphic_distinguished _ mem _ (Iso.symm ?_)
  refine Triangle.isoMk _ _ (Iso.refl _) e (Iso.refl _)
    (by aesop_cat) (by aesop_cat) (by aesop_cat)


lemma binaryProductTriangle_distinguished (X₁ X₂ : C) :
    binaryProductTriangle X₁ X₂ ∈ distTriang C :=
  isomorphic_distinguished _ (binaryBiproductTriangle_distinguished X₁ X₂) _
    (binaryProductTriangleIsoBinaryBiproductTriangle X₁ X₂)


/-- A chosen extension of a commutative square into a morphism of distinguished triangles. -/
@[simps hom₁ hom₂]
def completeDistinguishedTriangleMorphism (T₁ T₂ : Triangle C)
    (hT₁ : T₁ ∈ distTriang C) (hT₂ : T₂ ∈ distTriang C)
    (a : T₁.obj₁ ⟶ T₂.obj₁) (b : T₁.obj₂ ⟶ T₂.obj₂) (comm : T₁.mor₁ ≫ b = a ≫ T₂.mor₁) :
    T₁ ⟶ T₂ :=
    have h := complete_distinguished_triangle_morphism _ _ hT₁ hT₂ a b comm
    { hom₁ := a
      hom₂ := b
      hom₃ := h.choose
      comm₁ := comm
      comm₂ := h.choose_spec.1
      comm₃ := h.choose_spec.2 }


/-- A product of distinguished triangles is distinguished -/
lemma productTriangle_distinguished {J : Type*} (T : J → Triangle C)
    (hT : ∀ j, T j ∈ distTriang C)
    [HasProduct (fun j => (T j).obj₁)] [HasProduct (fun j => (T j).obj₂)]
    [HasProduct (fun j => (T j).obj₃)] [HasProduct (fun j => (T j).obj₁⟦(1 : ℤ)⟧)] :
    productTriangle T ∈ distTriang C := by
  /- The proof proceeds by constructing a morphism of triangles
    `φ' : T' ⟶ productTriangle T` with `T'` distinguished, and such that
    `φ'.hom₁` and `φ'.hom₂` are identities. Then, it suffices to show that
    `φ'.hom₃` is an isomorphism, which is achieved by using Yoneda's lemma
    and diagram chases. -/
  /-
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    inst✝⁷ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁶ : CategoryTheory.HasShift C Int
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    J : Type u_1
    T : J → CategoryTheory.Pretriangulated.Triangle C
    hT : ∀ (j : J), Membership.mem CategoryTheory.Pretriangulated.distinguishedTri …
    inst✝³ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₁
    inst✝² : CategoryTheory.Limits.HasProduct fun j => (T j).obj₂
    inst✝¹ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₃
    inst✝ : CategoryTheory.Limits.HasProduct fun j => (CategoryTheory.shiftFunctor …
    ⊢ Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Catego …
  -/
  let f₁ := Limits.Pi.map (fun j => (T j).mor₁)
  /-
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    inst✝⁷ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁶ : CategoryTheory.HasShift C Int
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    J : Type u_1
    T : J → CategoryTheory.Pretriangulated.Triangle C
    hT : ∀ (j : J), Membership.mem CategoryTheory.Pretriangulated.distinguishedTri …
    inst✝³ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₁
    inst✝² : CategoryTheory.Limits.HasProduct fun j => (T j).obj₂
    inst✝¹ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₃
    inst✝ : CategoryTheory.Limits.HasProduct fun j => (CategoryTheory.shiftFunctor …
    f₁ : Quiver.Hom (CategoryTheory.Limits.piObj fun j => (T j).obj₁) (CategoryThe …
    ⊢ Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Catego …
  -/
  obtain ⟨Z, f₂, f₃, hT'⟩ := distinguished_cocone_triangle f₁
  /-
    case intro.intro.intro
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    inst✝⁷ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁶ : CategoryTheory.HasShift C Int
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    J : Type u_1
    T : J → CategoryTheory.Pretriangulated.Triangle C
    hT : ∀ (j : J), Membership.mem CategoryTheory.Pretriangulated.distinguishedTri …
    inst✝³ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₁
    inst✝² : CategoryTheory.Limits.HasProduct fun j => (T j).obj₂
    inst✝¹ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₃
    inst✝ : CategoryTheory.Limits.HasProduct fun j => (CategoryTheory.shiftFunctor …
    f₁ : Quiver.Hom (CategoryTheory.Limits.piObj fun j => (T j).obj₁) (CategoryThe …
    Z : C
    f₂ : Quiver.Hom (CategoryTheory.Limits.piObj fun j => (T j).obj₂) Z
    f₃ : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj (CategoryTheory.Limit …
    hT' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    ⊢ Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Catego …
  -/
  let T' := Triangle.mk f₁ f₂ f₃
  /-
    case intro.intro.intro
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    inst✝⁷ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁶ : CategoryTheory.HasShift C Int
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    J : Type u_1
    T : J → CategoryTheory.Pretriangulated.Triangle C
    hT : ∀ (j : J), Membership.mem CategoryTheory.Pretriangulated.distinguishedTri …
    inst✝³ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₁
    inst✝² : CategoryTheory.Limits.HasProduct fun j => (T j).obj₂
    inst✝¹ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₃
    inst✝ : CategoryTheory.Limits.HasProduct fun j => (CategoryTheory.shiftFunctor …
    f₁ : Quiver.Hom (CategoryTheory.Limits.piObj fun j => (T j).obj₁) (CategoryThe …
    Z : C
    f₂ : Quiver.Hom (CategoryTheory.Limits.piObj fun j => (T j).obj₂) Z
    f₃ : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj (CategoryTheory.Limit …
    hT' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    T' : CategoryTheory.Pretriangulated.Triangle C := CategoryTheory.Pretriangulat …
    ⊢ Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Catego …
  -/
  change T' ∈ distTriang C at hT'
  let φ : ∀ j, T' ⟶ T j := fun j => completeDistinguishedTriangleMorphism _ _
    hT' (hT j) (Pi.π _ j) (Pi.π _ j) (by simp [f₁, T'])
  /-
    case intro.intro.intro
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    inst✝⁷ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁶ : CategoryTheory.HasShift C Int
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    J : Type u_1
    T : J → CategoryTheory.Pretriangulated.Triangle C
    hT : ∀ (j : J), Membership.mem CategoryTheory.Pretriangulated.distinguishedTri …
    inst✝³ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₁
    inst✝² : CategoryTheory.Limits.HasProduct fun j => (T j).obj₂
    inst✝¹ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₃
    inst✝ : CategoryTheory.Limits.HasProduct fun j => (CategoryTheory.shiftFunctor …
    f₁ : Quiver.Hom (CategoryTheory.Limits.piObj fun j => (T j).obj₁) (CategoryThe …
    Z : C
    f₂ : Quiver.Hom (CategoryTheory.Limits.piObj fun j => (T j).obj₂) Z
    f₃ : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj (CategoryTheory.Limit …
    T' : CategoryTheory.Pretriangulated.Triangle C := CategoryTheory.Pretriangulat …
    hT' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T'
    φ : (j : J) → Quiver.Hom T' (T j) := fun j => CategoryTheory.Pretriangulated.c …
    ⊢ Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Catego …
  -/
  let φ' := productTriangle.lift _ φ
  /-
    case intro.intro.intro
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    inst✝⁷ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁶ : CategoryTheory.HasShift C Int
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    J : Type u_1
    T : J → CategoryTheory.Pretriangulated.Triangle C
    hT : ∀ (j : J), Membership.mem CategoryTheory.Pretriangulated.distinguishedTri …
    inst✝³ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₁
    inst✝² : CategoryTheory.Limits.HasProduct fun j => (T j).obj₂
    inst✝¹ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₃
    inst✝ : CategoryTheory.Limits.HasProduct fun j => (CategoryTheory.shiftFunctor …
    f₁ : Quiver.Hom (CategoryTheory.Limits.piObj fun j => (T j).obj₁) (CategoryThe …
    Z : C
    f₂ : Quiver.Hom (CategoryTheory.Limits.piObj fun j => (T j).obj₂) Z
    f₃ : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj (CategoryTheory.Limit …
    T' : CategoryTheory.Pretriangulated.Triangle C := CategoryTheory.Pretriangulat …
    hT' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T'
    φ : (j : J) → Quiver.Hom T' (T j) := fun j => CategoryTheory.Pretriangulated.c …
    φ' : Quiver.Hom T' (CategoryTheory.Pretriangulated.productTriangle T) := Categ …
    ⊢ Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Catego …
  -/
  have h₁ : φ'.hom₁ = 𝟙 _ := by aesop_cat
  /-
    case intro.intro.intro
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    inst✝⁷ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁶ : CategoryTheory.HasShift C Int
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    J : Type u_1
    T : J → CategoryTheory.Pretriangulated.Triangle C
    hT : ∀ (j : J), Membership.mem CategoryTheory.Pretriangulated.distinguishedTri …
    inst✝³ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₁
    inst✝² : CategoryTheory.Limits.HasProduct fun j => (T j).obj₂
    inst✝¹ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₃
    inst✝ : CategoryTheory.Limits.HasProduct fun j => (CategoryTheory.shiftFunctor …
    f₁ : Quiver.Hom (CategoryTheory.Limits.piObj fun j => (T j).obj₁) (CategoryThe …
    Z : C
    f₂ : Quiver.Hom (CategoryTheory.Limits.piObj fun j => (T j).obj₂) Z
    f₃ : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj (CategoryTheory.Limit …
    T' : CategoryTheory.Pretriangulated.Triangle C := CategoryTheory.Pretriangulat …
    hT' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T'
    φ : (j : J) → Quiver.Hom T' (T j) := fun j => CategoryTheory.Pretriangulated.c …
    φ' : Quiver.Hom T' (CategoryTheory.Pretriangulated.productTriangle T) := Categ …
    h₁ : Eq φ'.hom₁ (CategoryTheory.CategoryStruct.id T'.obj₁)
    ⊢ Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Catego …
  -/
  have h₂ : φ'.hom₂ = 𝟙 _ := by aesop_cat
  /-
    case intro.intro.intro
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    inst✝⁷ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁶ : CategoryTheory.HasShift C Int
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    J : Type u_1
    T : J → CategoryTheory.Pretriangulated.Triangle C
    hT : ∀ (j : J), Membership.mem CategoryTheory.Pretriangulated.distinguishedTri …
    inst✝³ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₁
    inst✝² : CategoryTheory.Limits.HasProduct fun j => (T j).obj₂
    inst✝¹ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₃
    inst✝ : CategoryTheory.Limits.HasProduct fun j => (CategoryTheory.shiftFunctor …
    f₁ : Quiver.Hom (CategoryTheory.Limits.piObj fun j => (T j).obj₁) (CategoryThe …
    Z : C
    f₂ : Quiver.Hom (CategoryTheory.Limits.piObj fun j => (T j).obj₂) Z
    f₃ : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj (CategoryTheory.Limit …
    T' : CategoryTheory.Pretriangulated.Triangle C := CategoryTheory.Pretriangulat …
    hT' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T'
    φ : (j : J) → Quiver.Hom T' (T j) := fun j => CategoryTheory.Pretriangulated.c …
    φ' : Quiver.Hom T' (CategoryTheory.Pretriangulated.productTriangle T) := Categ …
    h₁ : Eq φ'.hom₁ (CategoryTheory.CategoryStruct.id T'.obj₁)
    h₂ : Eq φ'.hom₂ (CategoryTheory.CategoryStruct.id T'.obj₂)
    ⊢ Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Catego …
  -/
  have : IsIso φ'.hom₁ := by rw [h₁]; infer_instance
  /-
    case intro.intro.intro
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    inst✝⁷ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁶ : CategoryTheory.HasShift C Int
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    J : Type u_1
    T : J → CategoryTheory.Pretriangulated.Triangle C
    hT : ∀ (j : J), Membership.mem CategoryTheory.Pretriangulated.distinguishedTri …
    inst✝³ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₁
    inst✝² : CategoryTheory.Limits.HasProduct fun j => (T j).obj₂
    inst✝¹ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₃
    inst✝ : CategoryTheory.Limits.HasProduct fun j => (CategoryTheory.shiftFunctor …
    f₁ : Quiver.Hom (CategoryTheory.Limits.piObj fun j => (T j).obj₁) (CategoryThe …
    Z : C
    f₂ : Quiver.Hom (CategoryTheory.Limits.piObj fun j => (T j).obj₂) Z
    f₃ : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj (CategoryTheory.Limit …
    T' : CategoryTheory.Pretriangulated.Triangle C := CategoryTheory.Pretriangulat …
    hT' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T'
    φ : (j : J) → Quiver.Hom T' (T j) := fun j => CategoryTheory.Pretriangulated.c …
    φ' : Quiver.Hom T' (CategoryTheory.Pretriangulated.productTriangle T) := Categ …
    h₁ : Eq φ'.hom₁ (CategoryTheory.CategoryStruct.id T'.obj₁)
    h₂ : Eq φ'.hom₂ (CategoryTheory.CategoryStruct.id T'.obj₂)
    this : CategoryTheory.IsIso φ'.hom₁
    ⊢ Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Catego …
  -/
  have : IsIso φ'.hom₂ := by rw [h₂]; infer_instance
  suffices IsIso φ'.hom₃ by
    have : IsIso φ' := by
      apply Triangle.isIso_of_isIsos
      all_goals infer_instance
    exact isomorphic_distinguished _ hT' _ (asIso φ').symm
  /-
    case intro.intro.intro
    C : Type u
    inst✝⁸ : CategoryTheory.Category.{v, u} C
    inst✝⁷ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁶ : CategoryTheory.HasShift C Int
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    J : Type u_1
    T : J → CategoryTheory.Pretriangulated.Triangle C
    hT : ∀ (j : J), Membership.mem CategoryTheory.Pretriangulated.distinguishedTri …
    inst✝³ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₁
    inst✝² : CategoryTheory.Limits.HasProduct fun j => (T j).obj₂
    inst✝¹ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₃
    inst✝ : CategoryTheory.Limits.HasProduct fun j => (CategoryTheory.shiftFunctor …
    f₁ : Quiver.Hom (CategoryTheory.Limits.piObj fun j => (T j).obj₁) (CategoryThe …
    Z : C
    f₂ : Quiver.Hom (CategoryTheory.Limits.piObj fun j => (T j).obj₂) Z
    f₃ : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj (CategoryTheory.Limit …
    T' : CategoryTheory.Pretriangulated.Triangle C := CategoryTheory.Pretriangulat …
    hT' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T'
    φ : (j : J) → Quiver.Hom T' (T j) := fun j => CategoryTheory.Pretriangulated.c …
    φ' : Quiver.Hom T' (CategoryTheory.Pretriangulated.productTriangle T) := Categ …
    h₁ : Eq φ'.hom₁ (CategoryTheory.CategoryStruct.id T'.obj₁)
    h₂ : Eq φ'.hom₂ (CategoryTheory.CategoryStruct.id T'.obj₂)
    this✝ : CategoryTheory.IsIso φ'.hom₁
    this : CategoryTheory.IsIso φ'.hom₂
    ⊢ CategoryTheory.IsIso φ'.hom₃
  -/
  refine isIso_of_yoneda_map_bijective _ (fun A => ⟨?_, ?_⟩)
  /- the proofs by diagram chase start here -/
  · suffices Mono φ'.hom₃ by
      intro a₁ a₂ ha
      simpa only [← cancel_mono φ'.hom₃] using ha
    /-
      case intro.intro.intro.refine_1
      C : Type u
      inst✝⁸ : CategoryTheory.Category.{v, u} C
      inst✝⁷ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁶ : CategoryTheory.HasShift C Int
      inst✝⁵ : CategoryTheory.Preadditive C
      inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      J : Type u_1
      T : J → CategoryTheory.Pretriangulated.Triangle C
      hT : ∀ (j : J), Membership.mem CategoryTheory.Pretriangulated.distinguishedTri …
      inst✝³ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₁
      inst✝² : CategoryTheory.Limits.HasProduct fun j => (T j).obj₂
      inst✝¹ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₃
      inst✝ : CategoryTheory.Limits.HasProduct fun j => (CategoryTheory.shiftFunctor …
      f₁ : Quiver.Hom (CategoryTheory.Limits.piObj fun j => (T j).obj₁) (CategoryThe …
      Z : C
      f₂ : Quiver.Hom (CategoryTheory.Limits.piObj fun j => (T j).obj₂) Z
      f₃ : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj (CategoryTheory.Limit …
      T' : CategoryTheory.Pretriangulated.Triangle C := CategoryTheory.Pretriangulat …
      hT' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T'
      φ : (j : J) → Quiver.Hom T' (T j) := fun j => CategoryTheory.Pretriangulated.c …
      φ' : Quiver.Hom T' (CategoryTheory.Pretriangulated.productTriangle T) := Categ …
      h₁ : Eq φ'.hom₁ (CategoryTheory.CategoryStruct.id T'.obj₁)
      h₂ : Eq φ'.hom₂ (CategoryTheory.CategoryStruct.id T'.obj₂)
      this✝ : CategoryTheory.IsIso φ'.hom₁
      this : CategoryTheory.IsIso φ'.hom₂
      A : C
      ⊢ CategoryTheory.Mono φ'.hom₃
    -/
    rw [mono_iff_cancel_zero]
    /-
      case intro.intro.intro.refine_1
      C : Type u
      inst✝⁸ : CategoryTheory.Category.{v, u} C
      inst✝⁷ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁶ : CategoryTheory.HasShift C Int
      inst✝⁵ : CategoryTheory.Preadditive C
      inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      J : Type u_1
      T : J → CategoryTheory.Pretriangulated.Triangle C
      hT : ∀ (j : J), Membership.mem CategoryTheory.Pretriangulated.distinguishedTri …
      inst✝³ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₁
      inst✝² : CategoryTheory.Limits.HasProduct fun j => (T j).obj₂
      inst✝¹ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₃
      inst✝ : CategoryTheory.Limits.HasProduct fun j => (CategoryTheory.shiftFunctor …
      f₁ : Quiver.Hom (CategoryTheory.Limits.piObj fun j => (T j).obj₁) (CategoryThe …
      Z : C
      f₂ : Quiver.Hom (CategoryTheory.Limits.piObj fun j => (T j).obj₂) Z
      f₃ : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj (CategoryTheory.Limit …
      T' : CategoryTheory.Pretriangulated.Triangle C := CategoryTheory.Pretriangulat …
      hT' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T'
      φ : (j : J) → Quiver.Hom T' (T j) := fun j => CategoryTheory.Pretriangulated.c …
      φ' : Quiver.Hom T' (CategoryTheory.Pretriangulated.productTriangle T) := Categ …
      h₁ : Eq φ'.hom₁ (CategoryTheory.CategoryStruct.id T'.obj₁)
      h₂ : Eq φ'.hom₂ (CategoryTheory.CategoryStruct.id T'.obj₂)
      this✝ : CategoryTheory.IsIso φ'.hom₁
      this : CategoryTheory.IsIso φ'.hom₂
      A : C
      ⊢ ∀ (P : C) (g : Quiver.Hom P T'.obj₃), Eq (CategoryTheory.CategoryStruct.comp …
    -/
    intro A f hf
    have hf' : f ≫ T'.mor₃ = 0 := by
      rw [← cancel_mono (φ'.hom₁⟦1⟧'), zero_comp, assoc, φ'.comm₃, reassoc_of% hf, zero_comp]
    /-
      case intro.intro.intro.refine_1
      C : Type u
      inst✝⁸ : CategoryTheory.Category.{v, u} C
      inst✝⁷ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁶ : CategoryTheory.HasShift C Int
      inst✝⁵ : CategoryTheory.Preadditive C
      inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      J : Type u_1
      T : J → CategoryTheory.Pretriangulated.Triangle C
      hT : ∀ (j : J), Membership.mem CategoryTheory.Pretriangulated.distinguishedTri …
      inst✝³ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₁
      inst✝² : CategoryTheory.Limits.HasProduct fun j => (T j).obj₂
      inst✝¹ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₃
      inst✝ : CategoryTheory.Limits.HasProduct fun j => (CategoryTheory.shiftFunctor …
      f₁ : Quiver.Hom (CategoryTheory.Limits.piObj fun j => (T j).obj₁) (CategoryThe …
      Z : C
      f₂ : Quiver.Hom (CategoryTheory.Limits.piObj fun j => (T j).obj₂) Z
      f₃ : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj (CategoryTheory.Limit …
      T' : CategoryTheory.Pretriangulated.Triangle C := CategoryTheory.Pretriangulat …
      hT' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T'
      φ : (j : J) → Quiver.Hom T' (T j) := fun j => CategoryTheory.Pretriangulated.c …
      φ' : Quiver.Hom T' (CategoryTheory.Pretriangulated.productTriangle T) := Categ …
      h₁ : Eq φ'.hom₁ (CategoryTheory.CategoryStruct.id T'.obj₁)
      h₂ : Eq φ'.hom₂ (CategoryTheory.CategoryStruct.id T'.obj₂)
      this✝ : CategoryTheory.IsIso φ'.hom₁
      this : CategoryTheory.IsIso φ'.hom₂
      A✝ A : C
      f : Quiver.Hom A T'.obj₃
      hf : Eq (CategoryTheory.CategoryStruct.comp f φ'.hom₃) 0
      hf' : Eq (CategoryTheory.CategoryStruct.comp f T'.mor₃) 0
      ⊢ Eq f 0
    -/
    obtain ⟨g, hg⟩ := T'.coyoneda_exact₃ hT' f hf'
    have hg' : ∀ j, (g ≫ Pi.π _ j) ≫ (T j).mor₂ = 0 := fun j => by
      have : g ≫ T'.mor₂ ≫ φ'.hom₃ ≫ Pi.π _ j = 0 := by
        rw [← reassoc_of% hg, reassoc_of% hf, zero_comp]
      rw [φ'.comm₂_assoc, h₂, id_comp] at this
      simpa using this
    /-
      case intro.intro.intro.refine_1.intro
      C : Type u
      inst✝⁸ : CategoryTheory.Category.{v, u} C
      inst✝⁷ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁶ : CategoryTheory.HasShift C Int
      inst✝⁵ : CategoryTheory.Preadditive C
      inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      J : Type u_1
      T : J → CategoryTheory.Pretriangulated.Triangle C
      hT : ∀ (j : J), Membership.mem CategoryTheory.Pretriangulated.distinguishedTri …
      inst✝³ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₁
      inst✝² : CategoryTheory.Limits.HasProduct fun j => (T j).obj₂
      inst✝¹ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₃
      inst✝ : CategoryTheory.Limits.HasProduct fun j => (CategoryTheory.shiftFunctor …
      f₁ : Quiver.Hom (CategoryTheory.Limits.piObj fun j => (T j).obj₁) (CategoryThe …
      Z : C
      f₂ : Quiver.Hom (CategoryTheory.Limits.piObj fun j => (T j).obj₂) Z
      f₃ : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj (CategoryTheory.Limit …
      T' : CategoryTheory.Pretriangulated.Triangle C := CategoryTheory.Pretriangulat …
      hT' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T'
      φ : (j : J) → Quiver.Hom T' (T j) := fun j => CategoryTheory.Pretriangulated.c …
      φ' : Quiver.Hom T' (CategoryTheory.Pretriangulated.productTriangle T) := Categ …
      h₁ : Eq φ'.hom₁ (CategoryTheory.CategoryStruct.id T'.obj₁)
      h₂ : Eq φ'.hom₂ (CategoryTheory.CategoryStruct.id T'.obj₂)
      this✝ : CategoryTheory.IsIso φ'.hom₁
      this : CategoryTheory.IsIso φ'.hom₂
      A✝ A : C
      f : Quiver.Hom A T'.obj₃
      hf : Eq (CategoryTheory.CategoryStruct.comp f φ'.hom₃) 0
      hf' : Eq (CategoryTheory.CategoryStruct.comp f T'.mor₃) 0
      g : Quiver.Hom A T'.obj₂
      hg : Eq f (CategoryTheory.CategoryStruct.comp g T'.mor₂)
      hg' : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Catego …
      ⊢ Eq f 0
    -/
    have hg'' := fun j => (T j).coyoneda_exact₂ (hT j) _ (hg' j)
    /-
      case intro.intro.intro.refine_1.intro
      C : Type u
      inst✝⁸ : CategoryTheory.Category.{v, u} C
      inst✝⁷ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁶ : CategoryTheory.HasShift C Int
      inst✝⁵ : CategoryTheory.Preadditive C
      inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      J : Type u_1
      T : J → CategoryTheory.Pretriangulated.Triangle C
      hT : ∀ (j : J), Membership.mem CategoryTheory.Pretriangulated.distinguishedTri …
      inst✝³ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₁
      inst✝² : CategoryTheory.Limits.HasProduct fun j => (T j).obj₂
      inst✝¹ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₃
      inst✝ : CategoryTheory.Limits.HasProduct fun j => (CategoryTheory.shiftFunctor …
      f₁ : Quiver.Hom (CategoryTheory.Limits.piObj fun j => (T j).obj₁) (CategoryThe …
      Z : C
      f₂ : Quiver.Hom (CategoryTheory.Limits.piObj fun j => (T j).obj₂) Z
      f₃ : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj (CategoryTheory.Limit …
      T' : CategoryTheory.Pretriangulated.Triangle C := CategoryTheory.Pretriangulat …
      hT' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T'
      φ : (j : J) → Quiver.Hom T' (T j) := fun j => CategoryTheory.Pretriangulated.c …
      φ' : Quiver.Hom T' (CategoryTheory.Pretriangulated.productTriangle T) := Categ …
      h₁ : Eq φ'.hom₁ (CategoryTheory.CategoryStruct.id T'.obj₁)
      h₂ : Eq φ'.hom₂ (CategoryTheory.CategoryStruct.id T'.obj₂)
      this✝ : CategoryTheory.IsIso φ'.hom₁
      this : CategoryTheory.IsIso φ'.hom₂
      A✝ A : C
      f : Quiver.Hom A T'.obj₃
      hf : Eq (CategoryTheory.CategoryStruct.comp f φ'.hom₃) 0
      hf' : Eq (CategoryTheory.CategoryStruct.comp f T'.mor₃) 0
      g : Quiver.Hom A T'.obj₂
      hg : Eq f (CategoryTheory.CategoryStruct.comp g T'.mor₂)
      hg' : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Catego …
      hg'' : ∀ (j : J), Exists fun g_1 => Eq (CategoryTheory.CategoryStruct.comp g ( …
      ⊢ Eq f 0
    -/
    let α := fun j => (hg'' j).choose
    /-
      case intro.intro.intro.refine_1.intro
      C : Type u
      inst✝⁸ : CategoryTheory.Category.{v, u} C
      inst✝⁷ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁶ : CategoryTheory.HasShift C Int
      inst✝⁵ : CategoryTheory.Preadditive C
      inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      J : Type u_1
      T : J → CategoryTheory.Pretriangulated.Triangle C
      hT : ∀ (j : J), Membership.mem CategoryTheory.Pretriangulated.distinguishedTri …
      inst✝³ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₁
      inst✝² : CategoryTheory.Limits.HasProduct fun j => (T j).obj₂
      inst✝¹ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₃
      inst✝ : CategoryTheory.Limits.HasProduct fun j => (CategoryTheory.shiftFunctor …
      f₁ : Quiver.Hom (CategoryTheory.Limits.piObj fun j => (T j).obj₁) (CategoryThe …
      Z : C
      f₂ : Quiver.Hom (CategoryTheory.Limits.piObj fun j => (T j).obj₂) Z
      f₃ : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj (CategoryTheory.Limit …
      T' : CategoryTheory.Pretriangulated.Triangle C := CategoryTheory.Pretriangulat …
      hT' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T'
      φ : (j : J) → Quiver.Hom T' (T j) := fun j => CategoryTheory.Pretriangulated.c …
      φ' : Quiver.Hom T' (CategoryTheory.Pretriangulated.productTriangle T) := Categ …
      h₁ : Eq φ'.hom₁ (CategoryTheory.CategoryStruct.id T'.obj₁)
      h₂ : Eq φ'.hom₂ (CategoryTheory.CategoryStruct.id T'.obj₂)
      this✝ : CategoryTheory.IsIso φ'.hom₁
      this : CategoryTheory.IsIso φ'.hom₂
      A✝ A : C
      f : Quiver.Hom A T'.obj₃
      hf : Eq (CategoryTheory.CategoryStruct.comp f φ'.hom₃) 0
      hf' : Eq (CategoryTheory.CategoryStruct.comp f T'.mor₃) 0
      g : Quiver.Hom A T'.obj₂
      hg : Eq f (CategoryTheory.CategoryStruct.comp g T'.mor₂)
      hg' : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Catego …
      hg'' : ∀ (j : J), Exists fun g_1 => Eq (CategoryTheory.CategoryStruct.comp g ( …
      α : (j : J) → Quiver.Hom A (T j).obj₁ := fun j => ⋯.choose
      ⊢ Eq f 0
    -/
    have hα : ∀ j, _ = α j ≫ _ := fun j => (hg'' j).choose_spec
    /-
      case intro.intro.intro.refine_1.intro
      C : Type u
      inst✝⁸ : CategoryTheory.Category.{v, u} C
      inst✝⁷ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁶ : CategoryTheory.HasShift C Int
      inst✝⁵ : CategoryTheory.Preadditive C
      inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      J : Type u_1
      T : J → CategoryTheory.Pretriangulated.Triangle C
      hT : ∀ (j : J), Membership.mem CategoryTheory.Pretriangulated.distinguishedTri …
      inst✝³ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₁
      inst✝² : CategoryTheory.Limits.HasProduct fun j => (T j).obj₂
      inst✝¹ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₃
      inst✝ : CategoryTheory.Limits.HasProduct fun j => (CategoryTheory.shiftFunctor …
      f₁ : Quiver.Hom (CategoryTheory.Limits.piObj fun j => (T j).obj₁) (CategoryThe …
      Z : C
      f₂ : Quiver.Hom (CategoryTheory.Limits.piObj fun j => (T j).obj₂) Z
      f₃ : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj (CategoryTheory.Limit …
      T' : CategoryTheory.Pretriangulated.Triangle C := CategoryTheory.Pretriangulat …
      hT' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T'
      φ : (j : J) → Quiver.Hom T' (T j) := fun j => CategoryTheory.Pretriangulated.c …
      φ' : Quiver.Hom T' (CategoryTheory.Pretriangulated.productTriangle T) := Categ …
      h₁ : Eq φ'.hom₁ (CategoryTheory.CategoryStruct.id T'.obj₁)
      h₂ : Eq φ'.hom₂ (CategoryTheory.CategoryStruct.id T'.obj₂)
      this✝ : CategoryTheory.IsIso φ'.hom₁
      this : CategoryTheory.IsIso φ'.hom₂
      A✝ A : C
      f : Quiver.Hom A T'.obj₃
      hf : Eq (CategoryTheory.CategoryStruct.comp f φ'.hom₃) 0
      hf' : Eq (CategoryTheory.CategoryStruct.comp f T'.mor₃) 0
      g : Quiver.Hom A T'.obj₂
      hg : Eq f (CategoryTheory.CategoryStruct.comp g T'.mor₂)
      hg' : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Catego …
      hg'' : ∀ (j : J), Exists fun g_1 => Eq (CategoryTheory.CategoryStruct.comp g ( …
      α : (j : J) → Quiver.Hom A (T j).obj₁ := fun j => ⋯.choose
      hα : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp g (CategoryTheory.Limit …
      ⊢ Eq f 0
    -/
    have hg''' : g = Pi.lift α ≫ T'.mor₁ := by dsimp [f₁, T']; ext j; rw [hα]; simp
    /-
      case intro.intro.intro.refine_1.intro
      C : Type u
      inst✝⁸ : CategoryTheory.Category.{v, u} C
      inst✝⁷ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁶ : CategoryTheory.HasShift C Int
      inst✝⁵ : CategoryTheory.Preadditive C
      inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      J : Type u_1
      T : J → CategoryTheory.Pretriangulated.Triangle C
      hT : ∀ (j : J), Membership.mem CategoryTheory.Pretriangulated.distinguishedTri …
      inst✝³ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₁
      inst✝² : CategoryTheory.Limits.HasProduct fun j => (T j).obj₂
      inst✝¹ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₃
      inst✝ : CategoryTheory.Limits.HasProduct fun j => (CategoryTheory.shiftFunctor …
      f₁ : Quiver.Hom (CategoryTheory.Limits.piObj fun j => (T j).obj₁) (CategoryThe …
      Z : C
      f₂ : Quiver.Hom (CategoryTheory.Limits.piObj fun j => (T j).obj₂) Z
      f₃ : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj (CategoryTheory.Limit …
      T' : CategoryTheory.Pretriangulated.Triangle C := CategoryTheory.Pretriangulat …
      hT' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T'
      φ : (j : J) → Quiver.Hom T' (T j) := fun j => CategoryTheory.Pretriangulated.c …
      φ' : Quiver.Hom T' (CategoryTheory.Pretriangulated.productTriangle T) := Categ …
      h₁ : Eq φ'.hom₁ (CategoryTheory.CategoryStruct.id T'.obj₁)
      h₂ : Eq φ'.hom₂ (CategoryTheory.CategoryStruct.id T'.obj₂)
      this✝ : CategoryTheory.IsIso φ'.hom₁
      this : CategoryTheory.IsIso φ'.hom₂
      A✝ A : C
      f : Quiver.Hom A T'.obj₃
      hf : Eq (CategoryTheory.CategoryStruct.comp f φ'.hom₃) 0
      hf' : Eq (CategoryTheory.CategoryStruct.comp f T'.mor₃) 0
      g : Quiver.Hom A T'.obj₂
      hg : Eq f (CategoryTheory.CategoryStruct.comp g T'.mor₂)
      hg' : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Catego …
      hg'' : ∀ (j : J), Exists fun g_1 => Eq (CategoryTheory.CategoryStruct.comp g ( …
      α : (j : J) → Quiver.Hom A (T j).obj₁ := fun j => ⋯.choose
      hα : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp g (CategoryTheory.Limit …
      hg''' : Eq g (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Pi.lif …
      ⊢ Eq f 0
    -/
    rw [hg, hg''', assoc, comp_distTriang_mor_zero₁₂ _ hT', comp_zero]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.refine_2
      C : Type u
      inst✝⁸ : CategoryTheory.Category.{v, u} C
      inst✝⁷ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁶ : CategoryTheory.HasShift C Int
      inst✝⁵ : CategoryTheory.Preadditive C
      inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      J : Type u_1
      T : J → CategoryTheory.Pretriangulated.Triangle C
      hT : ∀ (j : J), Membership.mem CategoryTheory.Pretriangulated.distinguishedTri …
      inst✝³ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₁
      inst✝² : CategoryTheory.Limits.HasProduct fun j => (T j).obj₂
      inst✝¹ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₃
      inst✝ : CategoryTheory.Limits.HasProduct fun j => (CategoryTheory.shiftFunctor …
      f₁ : Quiver.Hom (CategoryTheory.Limits.piObj fun j => (T j).obj₁) (CategoryThe …
      Z : C
      f₂ : Quiver.Hom (CategoryTheory.Limits.piObj fun j => (T j).obj₂) Z
      f₃ : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj (CategoryTheory.Limit …
      T' : CategoryTheory.Pretriangulated.Triangle C := CategoryTheory.Pretriangulat …
      hT' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T'
      φ : (j : J) → Quiver.Hom T' (T j) := fun j => CategoryTheory.Pretriangulated.c …
      φ' : Quiver.Hom T' (CategoryTheory.Pretriangulated.productTriangle T) := Categ …
      h₁ : Eq φ'.hom₁ (CategoryTheory.CategoryStruct.id T'.obj₁)
      h₂ : Eq φ'.hom₂ (CategoryTheory.CategoryStruct.id T'.obj₂)
      this✝ : CategoryTheory.IsIso φ'.hom₁
      this : CategoryTheory.IsIso φ'.hom₂
      A : C
      ⊢ Function.Surjective fun x => CategoryTheory.CategoryStruct.comp x φ'.hom₃
    -/
  · intro a
    obtain ⟨a', ha'⟩ : ∃ (a' : A ⟶ Z), a' ≫ T'.mor₃ = a ≫ (productTriangle T).mor₃ := by
      have zero : ((productTriangle T).mor₃) ≫ (shiftFunctor C 1).map T'.mor₁ = 0 := by
        rw [← cancel_mono (φ'.hom₂⟦1⟧'), zero_comp, assoc, ← Functor.map_comp, φ'.comm₁, h₁,
          id_comp, productTriangle.zero₃₁]
        intro j
        exact comp_distTriang_mor_zero₃₁ _ (hT j)
      have ⟨g, hg⟩ := T'.coyoneda_exact₁ hT' (a ≫ (productTriangle T).mor₃) (by
        rw [assoc, zero, comp_zero])
      exact ⟨g, hg.symm⟩
    have ha'' := fun (j : J) => (T j).coyoneda_exact₃ (hT j) ((a - a' ≫ φ'.hom₃) ≫ Pi.π _ j) (by
      simp only [sub_comp, assoc]
      erw [← (productTriangle.π T j).comm₃]
      rw [← φ'.comm₃_assoc]
      rw [reassoc_of% ha', sub_eq_zero, h₁, Functor.map_id, id_comp])
    /-
      case intro.intro.intro.refine_2.intro
      C : Type u
      inst✝⁸ : CategoryTheory.Category.{v, u} C
      inst✝⁷ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁶ : CategoryTheory.HasShift C Int
      inst✝⁵ : CategoryTheory.Preadditive C
      inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      J : Type u_1
      T : J → CategoryTheory.Pretriangulated.Triangle C
      hT : ∀ (j : J), Membership.mem CategoryTheory.Pretriangulated.distinguishedTri …
      inst✝³ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₁
      inst✝² : CategoryTheory.Limits.HasProduct fun j => (T j).obj₂
      inst✝¹ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₃
      inst✝ : CategoryTheory.Limits.HasProduct fun j => (CategoryTheory.shiftFunctor …
      f₁ : Quiver.Hom (CategoryTheory.Limits.piObj fun j => (T j).obj₁) (CategoryThe …
      Z : C
      f₂ : Quiver.Hom (CategoryTheory.Limits.piObj fun j => (T j).obj₂) Z
      f₃ : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj (CategoryTheory.Limit …
      T' : CategoryTheory.Pretriangulated.Triangle C := CategoryTheory.Pretriangulat …
      hT' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T'
      φ : (j : J) → Quiver.Hom T' (T j) := fun j => CategoryTheory.Pretriangulated.c …
      φ' : Quiver.Hom T' (CategoryTheory.Pretriangulated.productTriangle T) := Categ …
      h₁ : Eq φ'.hom₁ (CategoryTheory.CategoryStruct.id T'.obj₁)
      h₂ : Eq φ'.hom₂ (CategoryTheory.CategoryStruct.id T'.obj₂)
      this✝ : CategoryTheory.IsIso φ'.hom₁
      this : CategoryTheory.IsIso φ'.hom₂
      A : C
      a : Quiver.Hom A (CategoryTheory.Pretriangulated.productTriangle T).obj₃
      a' : Quiver.Hom A Z
      ha' : Eq (CategoryTheory.CategoryStruct.comp a' T'.mor₃) (CategoryTheory.Categ …
      ha'' : ∀ (j : J), Exists fun g => Eq (CategoryTheory.CategoryStruct.comp (HSub …
      ⊢ Exists fun a_1 => Eq ((fun x => CategoryTheory.CategoryStruct.comp x φ'.hom₃ …
    -/
    let b := fun j => (ha'' j).choose
    /-
      case intro.intro.intro.refine_2.intro
      C : Type u
      inst✝⁸ : CategoryTheory.Category.{v, u} C
      inst✝⁷ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁶ : CategoryTheory.HasShift C Int
      inst✝⁵ : CategoryTheory.Preadditive C
      inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      J : Type u_1
      T : J → CategoryTheory.Pretriangulated.Triangle C
      hT : ∀ (j : J), Membership.mem CategoryTheory.Pretriangulated.distinguishedTri …
      inst✝³ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₁
      inst✝² : CategoryTheory.Limits.HasProduct fun j => (T j).obj₂
      inst✝¹ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₃
      inst✝ : CategoryTheory.Limits.HasProduct fun j => (CategoryTheory.shiftFunctor …
      f₁ : Quiver.Hom (CategoryTheory.Limits.piObj fun j => (T j).obj₁) (CategoryThe …
      Z : C
      f₂ : Quiver.Hom (CategoryTheory.Limits.piObj fun j => (T j).obj₂) Z
      f₃ : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj (CategoryTheory.Limit …
      T' : CategoryTheory.Pretriangulated.Triangle C := CategoryTheory.Pretriangulat …
      hT' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T'
      φ : (j : J) → Quiver.Hom T' (T j) := fun j => CategoryTheory.Pretriangulated.c …
      φ' : Quiver.Hom T' (CategoryTheory.Pretriangulated.productTriangle T) := Categ …
      h₁ : Eq φ'.hom₁ (CategoryTheory.CategoryStruct.id T'.obj₁)
      h₂ : Eq φ'.hom₂ (CategoryTheory.CategoryStruct.id T'.obj₂)
      this✝ : CategoryTheory.IsIso φ'.hom₁
      this : CategoryTheory.IsIso φ'.hom₂
      A : C
      a : Quiver.Hom A (CategoryTheory.Pretriangulated.productTriangle T).obj₃
      a' : Quiver.Hom A Z
      ha' : Eq (CategoryTheory.CategoryStruct.comp a' T'.mor₃) (CategoryTheory.Categ …
      ha'' : ∀ (j : J), Exists fun g => Eq (CategoryTheory.CategoryStruct.comp (HSub …
      b : (j : J) → Quiver.Hom A (T j).obj₂ := fun j => ⋯.choose
      ⊢ Exists fun a_1 => Eq ((fun x => CategoryTheory.CategoryStruct.comp x φ'.hom₃ …
    -/
    have hb : ∀ j, _  = b j ≫ _ := fun j => (ha'' j).choose_spec
    have hb' : a - a' ≫ φ'.hom₃ = Pi.lift b ≫ (productTriangle T).mor₂ :=
      Limits.Pi.hom_ext _ _ (fun j => by rw [hb]; simp)
    have : (a' + (by exact Pi.lift b) ≫ T'.mor₂) ≫ φ'.hom₃ = a := by
      rw [add_comp, assoc, φ'.comm₂, h₂, id_comp, ← hb', add_sub_cancel]
    /-
      case intro.intro.intro.refine_2.intro
      C : Type u
      inst✝⁸ : CategoryTheory.Category.{v, u} C
      inst✝⁷ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁶ : CategoryTheory.HasShift C Int
      inst✝⁵ : CategoryTheory.Preadditive C
      inst✝⁴ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      hC : CategoryTheory.Pretriangulated C
      J : Type u_1
      T : J → CategoryTheory.Pretriangulated.Triangle C
      hT : ∀ (j : J), Membership.mem CategoryTheory.Pretriangulated.distinguishedTri …
      inst✝³ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₁
      inst✝² : CategoryTheory.Limits.HasProduct fun j => (T j).obj₂
      inst✝¹ : CategoryTheory.Limits.HasProduct fun j => (T j).obj₃
      inst✝ : CategoryTheory.Limits.HasProduct fun j => (CategoryTheory.shiftFunctor …
      f₁ : Quiver.Hom (CategoryTheory.Limits.piObj fun j => (T j).obj₁) (CategoryThe …
      Z : C
      f₂ : Quiver.Hom (CategoryTheory.Limits.piObj fun j => (T j).obj₂) Z
      f₃ : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj (CategoryTheory.Limit …
      T' : CategoryTheory.Pretriangulated.Triangle C := CategoryTheory.Pretriangulat …
      hT' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T'
      φ : (j : J) → Quiver.Hom T' (T j) := fun j => CategoryTheory.Pretriangulated.c …
      φ' : Quiver.Hom T' (CategoryTheory.Pretriangulated.productTriangle T) := Categ …
      h₁ : Eq φ'.hom₁ (CategoryTheory.CategoryStruct.id T'.obj₁)
      h₂ : Eq φ'.hom₂ (CategoryTheory.CategoryStruct.id T'.obj₂)
      this✝¹ : CategoryTheory.IsIso φ'.hom₁
      this✝ : CategoryTheory.IsIso φ'.hom₂
      A : C
      a : Quiver.Hom A (CategoryTheory.Pretriangulated.productTriangle T).obj₃
      a' : Quiver.Hom A Z
      ha' : Eq (CategoryTheory.CategoryStruct.comp a' T'.mor₃) (CategoryTheory.Categ …
      ha'' : ∀ (j : J), Exists fun g => Eq (CategoryTheory.CategoryStruct.comp (HSub …
      b : (j : J) → Quiver.Hom A (T j).obj₂ := fun j => ⋯.choose
      hb : ∀ (j : J), Eq (CategoryTheory.CategoryStruct.comp (HSub.hSub a (CategoryT …
      hb' : Eq (HSub.hSub a (CategoryTheory.CategoryStruct.comp a' φ'.hom₃)) (Catego …
      this : Eq (CategoryTheory.CategoryStruct.comp (HAdd.hAdd a' (CategoryTheory.Ca …
      ⊢ Exists fun a_1 => Eq ((fun x => CategoryTheory.CategoryStruct.comp x φ'.hom₃ …
    -/
    exact ⟨_, this⟩
    /-
      🎉 no goals
    -/


lemma exists_iso_of_arrow_iso (T₁ T₂ : Triangle C) (hT₁ : T₁ ∈ distTriang C)
    (hT₂ : T₂ ∈ distTriang C) (e : Arrow.mk T₁.mor₁ ≅ Arrow.mk T₂.mor₁) :
    ∃ (e' : T₁ ≅ T₂), e'.hom.hom₁ = e.hom.left ∧ e'.hom.hom₂ = e.hom.right := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle C
    hT₁ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁
    hT₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₂
    e : CategoryTheory.Iso (CategoryTheory.Arrow.mk T₁.mor₁) (CategoryTheory.Arrow …
    ⊢ Exists fun e' => And (Eq e'.hom.hom₁ e.hom.left) (Eq e'.hom.hom₂ e.hom.right)
  -/
  let φ := completeDistinguishedTriangleMorphism T₁ T₂ hT₁ hT₂ e.hom.left e.hom.right e.hom.w.symm
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle C
    hT₁ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁
    hT₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₂
    e : CategoryTheory.Iso (CategoryTheory.Arrow.mk T₁.mor₁) (CategoryTheory.Arrow …
    φ : Quiver.Hom T₁ T₂ := CategoryTheory.Pretriangulated.completeDistinguishedTr …
    ⊢ Exists fun e' => And (Eq e'.hom.hom₁ e.hom.left) (Eq e'.hom.hom₂ e.hom.right)
  -/
  have : IsIso φ.hom₁ := by dsimp [φ]; infer_instance
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle C
    hT₁ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁
    hT₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₂
    e : CategoryTheory.Iso (CategoryTheory.Arrow.mk T₁.mor₁) (CategoryTheory.Arrow …
    φ : Quiver.Hom T₁ T₂ := CategoryTheory.Pretriangulated.completeDistinguishedTr …
    this : CategoryTheory.IsIso φ.hom₁
    ⊢ Exists fun e' => And (Eq e'.hom.hom₁ e.hom.left) (Eq e'.hom.hom₂ e.hom.right)
  -/
  have : IsIso φ.hom₂ := by dsimp [φ]; infer_instance
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle C
    hT₁ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁
    hT₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₂
    e : CategoryTheory.Iso (CategoryTheory.Arrow.mk T₁.mor₁) (CategoryTheory.Arrow …
    φ : Quiver.Hom T₁ T₂ := CategoryTheory.Pretriangulated.completeDistinguishedTr …
    this✝ : CategoryTheory.IsIso φ.hom₁
    this : CategoryTheory.IsIso φ.hom₂
    ⊢ Exists fun e' => And (Eq e'.hom.hom₁ e.hom.left) (Eq e'.hom.hom₂ e.hom.right)
  -/
  have : IsIso φ.hom₃ := isIso₃_of_isIso₁₂ φ hT₁ hT₂ inferInstance inferInstance
  have : IsIso φ := by
    apply Triangle.isIso_of_isIsos
    all_goals infer_instance
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle C
    hT₁ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁
    hT₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₂
    e : CategoryTheory.Iso (CategoryTheory.Arrow.mk T₁.mor₁) (CategoryTheory.Arrow …
    φ : Quiver.Hom T₁ T₂ := CategoryTheory.Pretriangulated.completeDistinguishedTr …
    this✝² : CategoryTheory.IsIso φ.hom₁
    this✝¹ : CategoryTheory.IsIso φ.hom₂
    this✝ : CategoryTheory.IsIso φ.hom₃
    this : CategoryTheory.IsIso φ
    ⊢ Exists fun e' => And (Eq e'.hom.hom₁ e.hom.left) (Eq e'.hom.hom₂ e.hom.right)
  -/
  exact ⟨asIso φ, by simp [φ], by simp [φ]⟩
  /-
    🎉 no goals
  -/


/-- A choice of isomorphism `T₁ ≅ T₂` between two distinguished triangles
when we are given two isomorphisms `e₁ : T₁.obj₁ ≅ T₂.obj₁` and `e₂ : T₁.obj₂ ≅ T₂.obj₂`. -/
@[simps! hom_hom₁ hom_hom₂ inv_hom₁ inv_hom₂]
def isoTriangleOfIso₁₂ (T₁ T₂ : Triangle C) (hT₁ : T₁ ∈ distTriang C)
    (hT₂ : T₂ ∈ distTriang C) (e₁ : T₁.obj₁ ≅ T₂.obj₁) (e₂ : T₁.obj₂ ≅ T₂.obj₂)
    (comm : T₁.mor₁ ≫ e₂.hom = e₁.hom ≫ T₂.mor₁) : T₁ ≅ T₂ := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    hC : CategoryTheory.Pretriangulated C
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle C
    hT₁ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁
    hT₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₂
    e₁ : CategoryTheory.Iso T₁.obj₁ T₂.obj₁
    e₂ : CategoryTheory.Iso T₁.obj₂ T₂.obj₂
    comm : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₁ e₂.hom) (CategoryTheory. …
    ⊢ CategoryTheory.Iso T₁ T₂
  -/
  have h := exists_iso_of_arrow_iso T₁ T₂ hT₁ hT₂ (Arrow.isoMk e₁ e₂ comm.symm)
  exact Triangle.isoMk _ _ e₁ e₂ (Triangle.π₃.mapIso h.choose) comm (by
    have eq := h.choose_spec.2
    dsimp at eq ⊢
    conv_rhs => rw [← eq, ← TriangleMorphism.comm₂]) (by
    have eq := h.choose_spec.1
    dsimp at eq ⊢
    conv_lhs => rw [← eq, TriangleMorphism.comm₃])


