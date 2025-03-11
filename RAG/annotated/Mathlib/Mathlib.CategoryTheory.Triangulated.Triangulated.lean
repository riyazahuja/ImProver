set_option genInjectivity false in
/-- An octahedron is a type of datum whose existence is asserted by
the octahedron axiom (TR 4), see https://stacks.math.columbia.edu/tag/05QK -/
structure Octahedron
  {X₁ X₂ X₃ Z₁₂ Z₂₃ Z₁₃ : C}
  {u₁₂ : X₁ ⟶ X₂} {u₂₃ : X₂ ⟶ X₃} {u₁₃ : X₁ ⟶ X₃} (comm : u₁₂ ≫ u₂₃ = u₁₃)
  {v₁₂ : X₂ ⟶ Z₁₂} {w₁₂ : Z₁₂ ⟶ X₁⟦(1 : ℤ)⟧} (h₁₂ : Triangle.mk u₁₂ v₁₂ w₁₂ ∈ distTriang C)
  {v₂₃ : X₃ ⟶ Z₂₃} {w₂₃ : Z₂₃ ⟶ X₂⟦(1 : ℤ)⟧} (h₂₃ : Triangle.mk u₂₃ v₂₃ w₂₃ ∈ distTriang C)
  {v₁₃ : X₃ ⟶ Z₁₃} {w₁₃ : Z₁₃ ⟶ X₁⟦(1 : ℤ)⟧} (h₁₃ : Triangle.mk u₁₃ v₁₃ w₁₃ ∈ distTriang C) where
  m₁ : Z₁₂ ⟶ Z₁₃
  m₃ : Z₁₃ ⟶ Z₂₃
  comm₁ : v₁₂ ≫ m₁ = u₂₃ ≫ v₁₃
  comm₂ : m₁ ≫ w₁₃ = w₁₂
  comm₃ : v₁₃ ≫ m₃ = v₂₃
  comm₄ : w₁₃ ≫ u₁₂⟦1⟧' = m₃ ≫ w₂₃
  mem : Triangle.mk m₁ m₃ (w₂₃ ≫ v₁₂⟦1⟧') ∈ distTriang C

gen_injective_theorems% Octahedron


instance (X : C) :
    Nonempty (Octahedron (comp_id (𝟙 X)) (contractible_distinguished X)
      (contractible_distinguished X) (contractible_distinguished X)) := by
  refine ⟨⟨0, 0, ?_, ?_, ?_, ?_, isomorphic_distinguished _ (contractible_distinguished (0 : C)) _
    (Triangle.isoMk _ _ (by rfl) (by rfl) (by rfl))⟩⟩
  /-
    case refine_1
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    X : C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp 0 0) (CategoryTheory.CategoryStruct.c …
  -/
  all_goals apply Subsingleton.elim
  /-
    🎉 no goals
  -/


attribute [reassoc] comm₁ comm₂ comm₃ comm₄


/-- The triangle `Z₁₂ ⟶ Z₁₃ ⟶ Z₂₃ ⟶ Z₁₂⟦1⟧` given by an octahedron. -/
@[simps!]
def triangle : Triangle C :=
  Triangle.mk h.m₁ h.m₃ (w₂₃ ≫ v₁₂⟦1⟧')


/-- The first morphism of triangles given by an octahedron. -/
@[simps]
def triangleMorphism₁ : Triangle.mk u₁₂ v₁₂ w₁₂ ⟶ Triangle.mk u₁₃ v₁₃ w₁₃ where
  hom₁ := 𝟙 X₁
  hom₂ := u₂₃
  hom₃ := h.m₁
  comm₁ := by
    /-
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.25003, u_1} C
      inst✝⁴ : CategoryTheory.Preadditive C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      X₁ X₂ X₃ Z₁₂ Z₂₃ Z₁₃ : C
      u₁₂ : Quiver.Hom X₁ X₂
      u₂₃ : Quiver.Hom X₂ X₃
      u₁₃ : Quiver.Hom X₁ X₃
      comm : Eq (CategoryTheory.CategoryStruct.comp u₁₂ u₂₃) u₁₃
      v₁₂ : Quiver.Hom X₂ Z₁₂
      w₁₂ : Quiver.Hom Z₁₂ ((CategoryTheory.shiftFunctor C 1).obj X₁)
      h₁₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      v₂₃ : Quiver.Hom X₃ Z₂₃
      w₂₃ : Quiver.Hom Z₂₃ ((CategoryTheory.shiftFunctor C 1).obj X₂)
      h₂₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      v₁₃ : Quiver.Hom X₃ Z₁₃
      w₁₃ : Quiver.Hom Z₁₃ ((CategoryTheory.shiftFunctor C 1).obj X₁)
      h₁₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      h : CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated.Trian …
    -/
    dsimp
    /-
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.25003, u_1} C
      inst✝⁴ : CategoryTheory.Preadditive C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      X₁ X₂ X₃ Z₁₂ Z₂₃ Z₁₃ : C
      u₁₂ : Quiver.Hom X₁ X₂
      u₂₃ : Quiver.Hom X₂ X₃
      u₁₃ : Quiver.Hom X₁ X₃
      comm : Eq (CategoryTheory.CategoryStruct.comp u₁₂ u₂₃) u₁₃
      v₁₂ : Quiver.Hom X₂ Z₁₂
      w₁₂ : Quiver.Hom Z₁₂ ((CategoryTheory.shiftFunctor C 1).obj X₁)
      h₁₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      v₂₃ : Quiver.Hom X₃ Z₂₃
      w₂₃ : Quiver.Hom Z₂₃ ((CategoryTheory.shiftFunctor C 1).obj X₂)
      h₂₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      v₁₃ : Quiver.Hom X₃ Z₁₃
      w₁₃ : Quiver.Hom Z₁₃ ((CategoryTheory.shiftFunctor C 1).obj X₁)
      h₁₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      h : CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
      ⊢ Eq (CategoryTheory.CategoryStruct.comp u₁₂ u₂₃) (CategoryTheory.CategoryStru …
    -/
    rw [id_comp, comm]
    /-
      🎉 no goals
    -/
  comm₂ := h.comm₁
  comm₃ := by
    /-
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.25003, u_1} C
      inst✝⁴ : CategoryTheory.Preadditive C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      X₁ X₂ X₃ Z₁₂ Z₂₃ Z₁₃ : C
      u₁₂ : Quiver.Hom X₁ X₂
      u₂₃ : Quiver.Hom X₂ X₃
      u₁₃ : Quiver.Hom X₁ X₃
      comm : Eq (CategoryTheory.CategoryStruct.comp u₁₂ u₂₃) u₁₃
      v₁₂ : Quiver.Hom X₂ Z₁₂
      w₁₂ : Quiver.Hom Z₁₂ ((CategoryTheory.shiftFunctor C 1).obj X₁)
      h₁₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      v₂₃ : Quiver.Hom X₃ Z₂₃
      w₂₃ : Quiver.Hom Z₂₃ ((CategoryTheory.shiftFunctor C 1).obj X₂)
      h₂₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      v₁₃ : Quiver.Hom X₃ Z₁₃
      w₁₃ : Quiver.Hom Z₁₃ ((CategoryTheory.shiftFunctor C 1).obj X₁)
      h₁₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      h : CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated.Trian …
    -/
    dsimp
    /-
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.25003, u_1} C
      inst✝⁴ : CategoryTheory.Preadditive C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      X₁ X₂ X₃ Z₁₂ Z₂₃ Z₁₃ : C
      u₁₂ : Quiver.Hom X₁ X₂
      u₂₃ : Quiver.Hom X₂ X₃
      u₁₃ : Quiver.Hom X₁ X₃
      comm : Eq (CategoryTheory.CategoryStruct.comp u₁₂ u₂₃) u₁₃
      v₁₂ : Quiver.Hom X₂ Z₁₂
      w₁₂ : Quiver.Hom Z₁₂ ((CategoryTheory.shiftFunctor C 1).obj X₁)
      h₁₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      v₂₃ : Quiver.Hom X₃ Z₂₃
      w₂₃ : Quiver.Hom Z₂₃ ((CategoryTheory.shiftFunctor C 1).obj X₂)
      h₂₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      v₁₃ : Quiver.Hom X₃ Z₁₃
      w₁₃ : Quiver.Hom Z₁₃ ((CategoryTheory.shiftFunctor C 1).obj X₁)
      h₁₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      h : CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
      ⊢ Eq (CategoryTheory.CategoryStruct.comp w₁₂ ((CategoryTheory.shiftFunctor C 1 …
    -/
    simpa only [Functor.map_id, comp_id] using h.comm₂.symm
    /-
      🎉 no goals
    -/


/-- The second morphism of triangles given an octahedron. -/
@[simps]
def triangleMorphism₂ : Triangle.mk u₁₃ v₁₃ w₁₃ ⟶ Triangle.mk u₂₃ v₂₃ w₂₃ where
  hom₁ := u₁₂
  hom₂ := 𝟙 X₃
  hom₃ := h.m₃
  comm₁ := by
    /-
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.30849, u_1} C
      inst✝⁴ : CategoryTheory.Preadditive C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      X₁ X₂ X₃ Z₁₂ Z₂₃ Z₁₃ : C
      u₁₂ : Quiver.Hom X₁ X₂
      u₂₃ : Quiver.Hom X₂ X₃
      u₁₃ : Quiver.Hom X₁ X₃
      comm : Eq (CategoryTheory.CategoryStruct.comp u₁₂ u₂₃) u₁₃
      v₁₂ : Quiver.Hom X₂ Z₁₂
      w₁₂ : Quiver.Hom Z₁₂ ((CategoryTheory.shiftFunctor C 1).obj X₁)
      h₁₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      v₂₃ : Quiver.Hom X₃ Z₂₃
      w₂₃ : Quiver.Hom Z₂₃ ((CategoryTheory.shiftFunctor C 1).obj X₂)
      h₂₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      v₁₃ : Quiver.Hom X₃ Z₁₃
      w₁₃ : Quiver.Hom Z₁₃ ((CategoryTheory.shiftFunctor C 1).obj X₁)
      h₁₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      h : CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated.Trian …
    -/
    dsimp
    /-
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.30849, u_1} C
      inst✝⁴ : CategoryTheory.Preadditive C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      X₁ X₂ X₃ Z₁₂ Z₂₃ Z₁₃ : C
      u₁₂ : Quiver.Hom X₁ X₂
      u₂₃ : Quiver.Hom X₂ X₃
      u₁₃ : Quiver.Hom X₁ X₃
      comm : Eq (CategoryTheory.CategoryStruct.comp u₁₂ u₂₃) u₁₃
      v₁₂ : Quiver.Hom X₂ Z₁₂
      w₁₂ : Quiver.Hom Z₁₂ ((CategoryTheory.shiftFunctor C 1).obj X₁)
      h₁₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      v₂₃ : Quiver.Hom X₃ Z₂₃
      w₂₃ : Quiver.Hom Z₂₃ ((CategoryTheory.shiftFunctor C 1).obj X₂)
      h₂₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      v₁₃ : Quiver.Hom X₃ Z₁₃
      w₁₃ : Quiver.Hom Z₁₃ ((CategoryTheory.shiftFunctor C 1).obj X₁)
      h₁₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      h : CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
      ⊢ Eq (CategoryTheory.CategoryStruct.comp u₁₃ (CategoryTheory.CategoryStruct.id …
    -/
    rw [comp_id, comm]
    /-
      🎉 no goals
    -/
  comm₂ := by
    /-
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.30849, u_1} C
      inst✝⁴ : CategoryTheory.Preadditive C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      X₁ X₂ X₃ Z₁₂ Z₂₃ Z₁₃ : C
      u₁₂ : Quiver.Hom X₁ X₂
      u₂₃ : Quiver.Hom X₂ X₃
      u₁₃ : Quiver.Hom X₁ X₃
      comm : Eq (CategoryTheory.CategoryStruct.comp u₁₂ u₂₃) u₁₃
      v₁₂ : Quiver.Hom X₂ Z₁₂
      w₁₂ : Quiver.Hom Z₁₂ ((CategoryTheory.shiftFunctor C 1).obj X₁)
      h₁₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      v₂₃ : Quiver.Hom X₃ Z₂₃
      w₂₃ : Quiver.Hom Z₂₃ ((CategoryTheory.shiftFunctor C 1).obj X₂)
      h₂₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      v₁₃ : Quiver.Hom X₃ Z₁₃
      w₁₃ : Quiver.Hom Z₁₃ ((CategoryTheory.shiftFunctor C 1).obj X₁)
      h₁₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      h : CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated.Trian …
    -/
    dsimp
    /-
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.30849, u_1} C
      inst✝⁴ : CategoryTheory.Preadditive C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      X₁ X₂ X₃ Z₁₂ Z₂₃ Z₁₃ : C
      u₁₂ : Quiver.Hom X₁ X₂
      u₂₃ : Quiver.Hom X₂ X₃
      u₁₃ : Quiver.Hom X₁ X₃
      comm : Eq (CategoryTheory.CategoryStruct.comp u₁₂ u₂₃) u₁₃
      v₁₂ : Quiver.Hom X₂ Z₁₂
      w₁₂ : Quiver.Hom Z₁₂ ((CategoryTheory.shiftFunctor C 1).obj X₁)
      h₁₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      v₂₃ : Quiver.Hom X₃ Z₂₃
      w₂₃ : Quiver.Hom Z₂₃ ((CategoryTheory.shiftFunctor C 1).obj X₂)
      h₂₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      v₁₃ : Quiver.Hom X₃ Z₁₃
      w₁₃ : Quiver.Hom Z₁₃ ((CategoryTheory.shiftFunctor C 1).obj X₁)
      h₁₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      h : CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
      ⊢ Eq (CategoryTheory.CategoryStruct.comp v₁₃ h.m₃) (CategoryTheory.CategoryStr …
    -/
    rw [id_comp, h.comm₃]
    /-
      🎉 no goals
    -/
  comm₃ := h.comm₄



/-- When two diagrams are isomorphic, an octahedron for one gives an octahedron for the other. -/
def ofIso {X₁' X₂' X₃' Z₁₂' Z₂₃' Z₁₃' : C} (u₁₂' : X₁' ⟶ X₂') (u₂₃' : X₂' ⟶ X₃') (u₁₃' : X₁' ⟶ X₃')
    (comm' : u₁₂' ≫ u₂₃' = u₁₃')
    (e₁ : X₁ ≅ X₁') (e₂ : X₂ ≅ X₂') (e₃ : X₃ ≅ X₃')
    (comm₁₂ : u₁₂ ≫ e₂.hom = e₁.hom ≫ u₁₂') (comm₂₃ : u₂₃ ≫ e₃.hom = e₂.hom ≫ u₂₃')
    (v₁₂' : X₂' ⟶ Z₁₂') (w₁₂' : Z₁₂' ⟶ X₁'⟦(1 : ℤ)⟧)
    (h₁₂' : Triangle.mk u₁₂' v₁₂' w₁₂' ∈ distTriang C)
    (v₂₃' : X₃' ⟶ Z₂₃') (w₂₃' : Z₂₃' ⟶ X₂'⟦(1 : ℤ)⟧)
    (h₂₃' : Triangle.mk u₂₃' v₂₃' w₂₃' ∈ distTriang C)
    (v₁₃' : X₃' ⟶ Z₁₃') (w₁₃' : Z₁₃' ⟶ X₁'⟦(1 : ℤ)⟧)
    (h₁₃' : Triangle.mk (u₁₃') v₁₃' w₁₃' ∈ distTriang C)
    (H : Octahedron comm' h₁₂' h₂₃' h₁₃') : Octahedron comm h₁₂ h₂₃ h₁₃ := by
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{?u.40535, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    X₁ X₂ X₃ Z₁₂ Z₂₃ Z₁₃ : C
    u₁₂ : Quiver.Hom X₁ X₂
    u₂₃ : Quiver.Hom X₂ X₃
    u₁₃ : Quiver.Hom X₁ X₃
    comm : Eq (CategoryTheory.CategoryStruct.comp u₁₂ u₂₃) u₁₃
    v₁₂ : Quiver.Hom X₂ Z₁₂
    w₁₂ : Quiver.Hom Z₁₂ ((CategoryTheory.shiftFunctor C 1).obj X₁)
    h₁₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    v₂₃ : Quiver.Hom X₃ Z₂₃
    w₂₃ : Quiver.Hom Z₂₃ ((CategoryTheory.shiftFunctor C 1).obj X₂)
    h₂₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    v₁₃ : Quiver.Hom X₃ Z₁₃
    w₁₃ : Quiver.Hom Z₁₃ ((CategoryTheory.shiftFunctor C 1).obj X₁)
    h₁₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    h : CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
    X₁' X₂' X₃' Z₁₂' Z₂₃' Z₁₃' : C
    u₁₂' : Quiver.Hom X₁' X₂'
    u₂₃' : Quiver.Hom X₂' X₃'
    u₁₃' : Quiver.Hom X₁' X₃'
    comm' : Eq (CategoryTheory.CategoryStruct.comp u₁₂' u₂₃') u₁₃'
    e₁ : CategoryTheory.Iso X₁ X₁'
    e₂ : CategoryTheory.Iso X₂ X₂'
    e₃ : CategoryTheory.Iso X₃ X₃'
    comm₁₂ : Eq (CategoryTheory.CategoryStruct.comp u₁₂ e₂.hom) (CategoryTheory.Ca …
    comm₂₃ : Eq (CategoryTheory.CategoryStruct.comp u₂₃ e₃.hom) (CategoryTheory.Ca …
    v₁₂' : Quiver.Hom X₂' Z₁₂'
    w₁₂' : Quiver.Hom Z₁₂' ((CategoryTheory.shiftFunctor C 1).obj X₁')
    h₁₂' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    v₂₃' : Quiver.Hom X₃' Z₂₃'
    w₂₃' : Quiver.Hom Z₂₃' ((CategoryTheory.shiftFunctor C 1).obj X₂')
    h₂₃' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    v₁₃' : Quiver.Hom X₃' Z₁₃'
    w₁₃' : Quiver.Hom Z₁₃' ((CategoryTheory.shiftFunctor C 1).obj X₁')
    h₁₃' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    H : CategoryTheory.Triangulated.Octahedron comm' h₁₂' h₂₃' h₁₃'
    ⊢ CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
  -/
  let iso₁₂ := isoTriangleOfIso₁₂ _ _ h₁₂ h₁₂' e₁ e₂ comm₁₂
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{?u.40535, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    X₁ X₂ X₃ Z₁₂ Z₂₃ Z₁₃ : C
    u₁₂ : Quiver.Hom X₁ X₂
    u₂₃ : Quiver.Hom X₂ X₃
    u₁₃ : Quiver.Hom X₁ X₃
    comm : Eq (CategoryTheory.CategoryStruct.comp u₁₂ u₂₃) u₁₃
    v₁₂ : Quiver.Hom X₂ Z₁₂
    w₁₂ : Quiver.Hom Z₁₂ ((CategoryTheory.shiftFunctor C 1).obj X₁)
    h₁₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    v₂₃ : Quiver.Hom X₃ Z₂₃
    w₂₃ : Quiver.Hom Z₂₃ ((CategoryTheory.shiftFunctor C 1).obj X₂)
    h₂₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    v₁₃ : Quiver.Hom X₃ Z₁₃
    w₁₃ : Quiver.Hom Z₁₃ ((CategoryTheory.shiftFunctor C 1).obj X₁)
    h₁₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    h : CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
    X₁' X₂' X₃' Z₁₂' Z₂₃' Z₁₃' : C
    u₁₂' : Quiver.Hom X₁' X₂'
    u₂₃' : Quiver.Hom X₂' X₃'
    u₁₃' : Quiver.Hom X₁' X₃'
    comm' : Eq (CategoryTheory.CategoryStruct.comp u₁₂' u₂₃') u₁₃'
    e₁ : CategoryTheory.Iso X₁ X₁'
    e₂ : CategoryTheory.Iso X₂ X₂'
    e₃ : CategoryTheory.Iso X₃ X₃'
    comm₁₂ : Eq (CategoryTheory.CategoryStruct.comp u₁₂ e₂.hom) (CategoryTheory.Ca …
    comm₂₃ : Eq (CategoryTheory.CategoryStruct.comp u₂₃ e₃.hom) (CategoryTheory.Ca …
    v₁₂' : Quiver.Hom X₂' Z₁₂'
    w₁₂' : Quiver.Hom Z₁₂' ((CategoryTheory.shiftFunctor C 1).obj X₁')
    h₁₂' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    v₂₃' : Quiver.Hom X₃' Z₂₃'
    w₂₃' : Quiver.Hom Z₂₃' ((CategoryTheory.shiftFunctor C 1).obj X₂')
    h₂₃' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    v₁₃' : Quiver.Hom X₃' Z₁₃'
    w₁₃' : Quiver.Hom Z₁₃' ((CategoryTheory.shiftFunctor C 1).obj X₁')
    h₁₃' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    H : CategoryTheory.Triangulated.Octahedron comm' h₁₂' h₂₃' h₁₃'
    iso₁₂ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₁₂ v₁₂ …
    ⊢ CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
  -/
  let iso₂₃ := isoTriangleOfIso₁₂ _ _ h₂₃ h₂₃' e₂ e₃ comm₂₃
  let iso₁₃ := isoTriangleOfIso₁₂ _ _ h₁₃ h₁₃' e₁ e₃ (by
    dsimp; rw [← comm, assoc, ← comm', ← reassoc_of% comm₁₂, comm₂₃])
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{?u.40535, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    X₁ X₂ X₃ Z₁₂ Z₂₃ Z₁₃ : C
    u₁₂ : Quiver.Hom X₁ X₂
    u₂₃ : Quiver.Hom X₂ X₃
    u₁₃ : Quiver.Hom X₁ X₃
    comm : Eq (CategoryTheory.CategoryStruct.comp u₁₂ u₂₃) u₁₃
    v₁₂ : Quiver.Hom X₂ Z₁₂
    w₁₂ : Quiver.Hom Z₁₂ ((CategoryTheory.shiftFunctor C 1).obj X₁)
    h₁₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    v₂₃ : Quiver.Hom X₃ Z₂₃
    w₂₃ : Quiver.Hom Z₂₃ ((CategoryTheory.shiftFunctor C 1).obj X₂)
    h₂₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    v₁₃ : Quiver.Hom X₃ Z₁₃
    w₁₃ : Quiver.Hom Z₁₃ ((CategoryTheory.shiftFunctor C 1).obj X₁)
    h₁₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    h : CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
    X₁' X₂' X₃' Z₁₂' Z₂₃' Z₁₃' : C
    u₁₂' : Quiver.Hom X₁' X₂'
    u₂₃' : Quiver.Hom X₂' X₃'
    u₁₃' : Quiver.Hom X₁' X₃'
    comm' : Eq (CategoryTheory.CategoryStruct.comp u₁₂' u₂₃') u₁₃'
    e₁ : CategoryTheory.Iso X₁ X₁'
    e₂ : CategoryTheory.Iso X₂ X₂'
    e₃ : CategoryTheory.Iso X₃ X₃'
    comm₁₂ : Eq (CategoryTheory.CategoryStruct.comp u₁₂ e₂.hom) (CategoryTheory.Ca …
    comm₂₃ : Eq (CategoryTheory.CategoryStruct.comp u₂₃ e₃.hom) (CategoryTheory.Ca …
    v₁₂' : Quiver.Hom X₂' Z₁₂'
    w₁₂' : Quiver.Hom Z₁₂' ((CategoryTheory.shiftFunctor C 1).obj X₁')
    h₁₂' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    v₂₃' : Quiver.Hom X₃' Z₂₃'
    w₂₃' : Quiver.Hom Z₂₃' ((CategoryTheory.shiftFunctor C 1).obj X₂')
    h₂₃' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    v₁₃' : Quiver.Hom X₃' Z₁₃'
    w₁₃' : Quiver.Hom Z₁₃' ((CategoryTheory.shiftFunctor C 1).obj X₁')
    h₁₃' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    H : CategoryTheory.Triangulated.Octahedron comm' h₁₂' h₂₃' h₁₃'
    iso₁₂ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₁₂ v₁₂ …
    iso₂₃ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₂₃ v₂₃ …
    iso₁₃ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₁₃ v₁₃ …
    ⊢ CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
  -/
  have eq₁₂ := iso₁₂.hom.comm₂
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{?u.40535, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    X₁ X₂ X₃ Z₁₂ Z₂₃ Z₁₃ : C
    u₁₂ : Quiver.Hom X₁ X₂
    u₂₃ : Quiver.Hom X₂ X₃
    u₁₃ : Quiver.Hom X₁ X₃
    comm : Eq (CategoryTheory.CategoryStruct.comp u₁₂ u₂₃) u₁₃
    v₁₂ : Quiver.Hom X₂ Z₁₂
    w₁₂ : Quiver.Hom Z₁₂ ((CategoryTheory.shiftFunctor C 1).obj X₁)
    h₁₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    v₂₃ : Quiver.Hom X₃ Z₂₃
    w₂₃ : Quiver.Hom Z₂₃ ((CategoryTheory.shiftFunctor C 1).obj X₂)
    h₂₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    v₁₃ : Quiver.Hom X₃ Z₁₃
    w₁₃ : Quiver.Hom Z₁₃ ((CategoryTheory.shiftFunctor C 1).obj X₁)
    h₁₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    h : CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
    X₁' X₂' X₃' Z₁₂' Z₂₃' Z₁₃' : C
    u₁₂' : Quiver.Hom X₁' X₂'
    u₂₃' : Quiver.Hom X₂' X₃'
    u₁₃' : Quiver.Hom X₁' X₃'
    comm' : Eq (CategoryTheory.CategoryStruct.comp u₁₂' u₂₃') u₁₃'
    e₁ : CategoryTheory.Iso X₁ X₁'
    e₂ : CategoryTheory.Iso X₂ X₂'
    e₃ : CategoryTheory.Iso X₃ X₃'
    comm₁₂ : Eq (CategoryTheory.CategoryStruct.comp u₁₂ e₂.hom) (CategoryTheory.Ca …
    comm₂₃ : Eq (CategoryTheory.CategoryStruct.comp u₂₃ e₃.hom) (CategoryTheory.Ca …
    v₁₂' : Quiver.Hom X₂' Z₁₂'
    w₁₂' : Quiver.Hom Z₁₂' ((CategoryTheory.shiftFunctor C 1).obj X₁')
    h₁₂' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    v₂₃' : Quiver.Hom X₃' Z₂₃'
    w₂₃' : Quiver.Hom Z₂₃' ((CategoryTheory.shiftFunctor C 1).obj X₂')
    h₂₃' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    v₁₃' : Quiver.Hom X₃' Z₁₃'
    w₁₃' : Quiver.Hom Z₁₃' ((CategoryTheory.shiftFunctor C 1).obj X₁')
    h₁₃' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    H : CategoryTheory.Triangulated.Octahedron comm' h₁₂' h₂₃' h₁₃'
    iso₁₂ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₁₂ v₁₂ …
    iso₂₃ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₂₃ v₂₃ …
    iso₁₃ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₁₃ v₁₃ …
    eq₁₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated. …
    ⊢ CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
  -/
  have eq₁₂' := iso₁₂.hom.comm₃
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{?u.40535, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    X₁ X₂ X₃ Z₁₂ Z₂₃ Z₁₃ : C
    u₁₂ : Quiver.Hom X₁ X₂
    u₂₃ : Quiver.Hom X₂ X₃
    u₁₃ : Quiver.Hom X₁ X₃
    comm : Eq (CategoryTheory.CategoryStruct.comp u₁₂ u₂₃) u₁₃
    v₁₂ : Quiver.Hom X₂ Z₁₂
    w₁₂ : Quiver.Hom Z₁₂ ((CategoryTheory.shiftFunctor C 1).obj X₁)
    h₁₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    v₂₃ : Quiver.Hom X₃ Z₂₃
    w₂₃ : Quiver.Hom Z₂₃ ((CategoryTheory.shiftFunctor C 1).obj X₂)
    h₂₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    v₁₃ : Quiver.Hom X₃ Z₁₃
    w₁₃ : Quiver.Hom Z₁₃ ((CategoryTheory.shiftFunctor C 1).obj X₁)
    h₁₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    h : CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
    X₁' X₂' X₃' Z₁₂' Z₂₃' Z₁₃' : C
    u₁₂' : Quiver.Hom X₁' X₂'
    u₂₃' : Quiver.Hom X₂' X₃'
    u₁₃' : Quiver.Hom X₁' X₃'
    comm' : Eq (CategoryTheory.CategoryStruct.comp u₁₂' u₂₃') u₁₃'
    e₁ : CategoryTheory.Iso X₁ X₁'
    e₂ : CategoryTheory.Iso X₂ X₂'
    e₃ : CategoryTheory.Iso X₃ X₃'
    comm₁₂ : Eq (CategoryTheory.CategoryStruct.comp u₁₂ e₂.hom) (CategoryTheory.Ca …
    comm₂₃ : Eq (CategoryTheory.CategoryStruct.comp u₂₃ e₃.hom) (CategoryTheory.Ca …
    v₁₂' : Quiver.Hom X₂' Z₁₂'
    w₁₂' : Quiver.Hom Z₁₂' ((CategoryTheory.shiftFunctor C 1).obj X₁')
    h₁₂' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    v₂₃' : Quiver.Hom X₃' Z₂₃'
    w₂₃' : Quiver.Hom Z₂₃' ((CategoryTheory.shiftFunctor C 1).obj X₂')
    h₂₃' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    v₁₃' : Quiver.Hom X₃' Z₁₃'
    w₁₃' : Quiver.Hom Z₁₃' ((CategoryTheory.shiftFunctor C 1).obj X₁')
    h₁₃' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    H : CategoryTheory.Triangulated.Octahedron comm' h₁₂' h₂₃' h₁₃'
    iso₁₂ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₁₂ v₁₂ …
    iso₂₃ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₂₃ v₂₃ …
    iso₁₃ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₁₃ v₁₃ …
    eq₁₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated. …
    eq₁₂' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated …
    ⊢ CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
  -/
  have eq₁₃ := iso₁₃.hom.comm₂
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{?u.40535, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    X₁ X₂ X₃ Z₁₂ Z₂₃ Z₁₃ : C
    u₁₂ : Quiver.Hom X₁ X₂
    u₂₃ : Quiver.Hom X₂ X₃
    u₁₃ : Quiver.Hom X₁ X₃
    comm : Eq (CategoryTheory.CategoryStruct.comp u₁₂ u₂₃) u₁₃
    v₁₂ : Quiver.Hom X₂ Z₁₂
    w₁₂ : Quiver.Hom Z₁₂ ((CategoryTheory.shiftFunctor C 1).obj X₁)
    h₁₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    v₂₃ : Quiver.Hom X₃ Z₂₃
    w₂₃ : Quiver.Hom Z₂₃ ((CategoryTheory.shiftFunctor C 1).obj X₂)
    h₂₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    v₁₃ : Quiver.Hom X₃ Z₁₃
    w₁₃ : Quiver.Hom Z₁₃ ((CategoryTheory.shiftFunctor C 1).obj X₁)
    h₁₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    h : CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
    X₁' X₂' X₃' Z₁₂' Z₂₃' Z₁₃' : C
    u₁₂' : Quiver.Hom X₁' X₂'
    u₂₃' : Quiver.Hom X₂' X₃'
    u₁₃' : Quiver.Hom X₁' X₃'
    comm' : Eq (CategoryTheory.CategoryStruct.comp u₁₂' u₂₃') u₁₃'
    e₁ : CategoryTheory.Iso X₁ X₁'
    e₂ : CategoryTheory.Iso X₂ X₂'
    e₃ : CategoryTheory.Iso X₃ X₃'
    comm₁₂ : Eq (CategoryTheory.CategoryStruct.comp u₁₂ e₂.hom) (CategoryTheory.Ca …
    comm₂₃ : Eq (CategoryTheory.CategoryStruct.comp u₂₃ e₃.hom) (CategoryTheory.Ca …
    v₁₂' : Quiver.Hom X₂' Z₁₂'
    w₁₂' : Quiver.Hom Z₁₂' ((CategoryTheory.shiftFunctor C 1).obj X₁')
    h₁₂' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    v₂₃' : Quiver.Hom X₃' Z₂₃'
    w₂₃' : Quiver.Hom Z₂₃' ((CategoryTheory.shiftFunctor C 1).obj X₂')
    h₂₃' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    v₁₃' : Quiver.Hom X₃' Z₁₃'
    w₁₃' : Quiver.Hom Z₁₃' ((CategoryTheory.shiftFunctor C 1).obj X₁')
    h₁₃' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    H : CategoryTheory.Triangulated.Octahedron comm' h₁₂' h₂₃' h₁₃'
    iso₁₂ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₁₂ v₁₂ …
    iso₂₃ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₂₃ v₂₃ …
    iso₁₃ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₁₃ v₁₃ …
    eq₁₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated. …
    eq₁₂' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated …
    eq₁₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated. …
    ⊢ CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
  -/
  have eq₁₃' := iso₁₃.hom.comm₃
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{?u.40535, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    X₁ X₂ X₃ Z₁₂ Z₂₃ Z₁₃ : C
    u₁₂ : Quiver.Hom X₁ X₂
    u₂₃ : Quiver.Hom X₂ X₃
    u₁₃ : Quiver.Hom X₁ X₃
    comm : Eq (CategoryTheory.CategoryStruct.comp u₁₂ u₂₃) u₁₃
    v₁₂ : Quiver.Hom X₂ Z₁₂
    w₁₂ : Quiver.Hom Z₁₂ ((CategoryTheory.shiftFunctor C 1).obj X₁)
    h₁₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    v₂₃ : Quiver.Hom X₃ Z₂₃
    w₂₃ : Quiver.Hom Z₂₃ ((CategoryTheory.shiftFunctor C 1).obj X₂)
    h₂₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    v₁₃ : Quiver.Hom X₃ Z₁₃
    w₁₃ : Quiver.Hom Z₁₃ ((CategoryTheory.shiftFunctor C 1).obj X₁)
    h₁₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    h : CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
    X₁' X₂' X₃' Z₁₂' Z₂₃' Z₁₃' : C
    u₁₂' : Quiver.Hom X₁' X₂'
    u₂₃' : Quiver.Hom X₂' X₃'
    u₁₃' : Quiver.Hom X₁' X₃'
    comm' : Eq (CategoryTheory.CategoryStruct.comp u₁₂' u₂₃') u₁₃'
    e₁ : CategoryTheory.Iso X₁ X₁'
    e₂ : CategoryTheory.Iso X₂ X₂'
    e₃ : CategoryTheory.Iso X₃ X₃'
    comm₁₂ : Eq (CategoryTheory.CategoryStruct.comp u₁₂ e₂.hom) (CategoryTheory.Ca …
    comm₂₃ : Eq (CategoryTheory.CategoryStruct.comp u₂₃ e₃.hom) (CategoryTheory.Ca …
    v₁₂' : Quiver.Hom X₂' Z₁₂'
    w₁₂' : Quiver.Hom Z₁₂' ((CategoryTheory.shiftFunctor C 1).obj X₁')
    h₁₂' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    v₂₃' : Quiver.Hom X₃' Z₂₃'
    w₂₃' : Quiver.Hom Z₂₃' ((CategoryTheory.shiftFunctor C 1).obj X₂')
    h₂₃' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    v₁₃' : Quiver.Hom X₃' Z₁₃'
    w₁₃' : Quiver.Hom Z₁₃' ((CategoryTheory.shiftFunctor C 1).obj X₁')
    h₁₃' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    H : CategoryTheory.Triangulated.Octahedron comm' h₁₂' h₂₃' h₁₃'
    iso₁₂ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₁₂ v₁₂ …
    iso₂₃ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₂₃ v₂₃ …
    iso₁₃ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₁₃ v₁₃ …
    eq₁₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated. …
    eq₁₂' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated …
    eq₁₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated. …
    eq₁₃' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated …
    ⊢ CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
  -/
  have eq₂₃ := iso₂₃.hom.comm₂
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{?u.40535, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    X₁ X₂ X₃ Z₁₂ Z₂₃ Z₁₃ : C
    u₁₂ : Quiver.Hom X₁ X₂
    u₂₃ : Quiver.Hom X₂ X₃
    u₁₃ : Quiver.Hom X₁ X₃
    comm : Eq (CategoryTheory.CategoryStruct.comp u₁₂ u₂₃) u₁₃
    v₁₂ : Quiver.Hom X₂ Z₁₂
    w₁₂ : Quiver.Hom Z₁₂ ((CategoryTheory.shiftFunctor C 1).obj X₁)
    h₁₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    v₂₃ : Quiver.Hom X₃ Z₂₃
    w₂₃ : Quiver.Hom Z₂₃ ((CategoryTheory.shiftFunctor C 1).obj X₂)
    h₂₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    v₁₃ : Quiver.Hom X₃ Z₁₃
    w₁₃ : Quiver.Hom Z₁₃ ((CategoryTheory.shiftFunctor C 1).obj X₁)
    h₁₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    h : CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
    X₁' X₂' X₃' Z₁₂' Z₂₃' Z₁₃' : C
    u₁₂' : Quiver.Hom X₁' X₂'
    u₂₃' : Quiver.Hom X₂' X₃'
    u₁₃' : Quiver.Hom X₁' X₃'
    comm' : Eq (CategoryTheory.CategoryStruct.comp u₁₂' u₂₃') u₁₃'
    e₁ : CategoryTheory.Iso X₁ X₁'
    e₂ : CategoryTheory.Iso X₂ X₂'
    e₃ : CategoryTheory.Iso X₃ X₃'
    comm₁₂ : Eq (CategoryTheory.CategoryStruct.comp u₁₂ e₂.hom) (CategoryTheory.Ca …
    comm₂₃ : Eq (CategoryTheory.CategoryStruct.comp u₂₃ e₃.hom) (CategoryTheory.Ca …
    v₁₂' : Quiver.Hom X₂' Z₁₂'
    w₁₂' : Quiver.Hom Z₁₂' ((CategoryTheory.shiftFunctor C 1).obj X₁')
    h₁₂' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    v₂₃' : Quiver.Hom X₃' Z₂₃'
    w₂₃' : Quiver.Hom Z₂₃' ((CategoryTheory.shiftFunctor C 1).obj X₂')
    h₂₃' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    v₁₃' : Quiver.Hom X₃' Z₁₃'
    w₁₃' : Quiver.Hom Z₁₃' ((CategoryTheory.shiftFunctor C 1).obj X₁')
    h₁₃' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    H : CategoryTheory.Triangulated.Octahedron comm' h₁₂' h₂₃' h₁₃'
    iso₁₂ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₁₂ v₁₂ …
    iso₂₃ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₂₃ v₂₃ …
    iso₁₃ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₁₃ v₁₃ …
    eq₁₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated. …
    eq₁₂' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated …
    eq₁₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated. …
    eq₁₃' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated …
    eq₂₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated. …
    ⊢ CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
  -/
  have eq₂₃' := iso₂₃.hom.comm₃
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{?u.40535, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    X₁ X₂ X₃ Z₁₂ Z₂₃ Z₁₃ : C
    u₁₂ : Quiver.Hom X₁ X₂
    u₂₃ : Quiver.Hom X₂ X₃
    u₁₃ : Quiver.Hom X₁ X₃
    comm : Eq (CategoryTheory.CategoryStruct.comp u₁₂ u₂₃) u₁₃
    v₁₂ : Quiver.Hom X₂ Z₁₂
    w₁₂ : Quiver.Hom Z₁₂ ((CategoryTheory.shiftFunctor C 1).obj X₁)
    h₁₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    v₂₃ : Quiver.Hom X₃ Z₂₃
    w₂₃ : Quiver.Hom Z₂₃ ((CategoryTheory.shiftFunctor C 1).obj X₂)
    h₂₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    v₁₃ : Quiver.Hom X₃ Z₁₃
    w₁₃ : Quiver.Hom Z₁₃ ((CategoryTheory.shiftFunctor C 1).obj X₁)
    h₁₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    h : CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
    X₁' X₂' X₃' Z₁₂' Z₂₃' Z₁₃' : C
    u₁₂' : Quiver.Hom X₁' X₂'
    u₂₃' : Quiver.Hom X₂' X₃'
    u₁₃' : Quiver.Hom X₁' X₃'
    comm' : Eq (CategoryTheory.CategoryStruct.comp u₁₂' u₂₃') u₁₃'
    e₁ : CategoryTheory.Iso X₁ X₁'
    e₂ : CategoryTheory.Iso X₂ X₂'
    e₃ : CategoryTheory.Iso X₃ X₃'
    comm₁₂ : Eq (CategoryTheory.CategoryStruct.comp u₁₂ e₂.hom) (CategoryTheory.Ca …
    comm₂₃ : Eq (CategoryTheory.CategoryStruct.comp u₂₃ e₃.hom) (CategoryTheory.Ca …
    v₁₂' : Quiver.Hom X₂' Z₁₂'
    w₁₂' : Quiver.Hom Z₁₂' ((CategoryTheory.shiftFunctor C 1).obj X₁')
    h₁₂' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    v₂₃' : Quiver.Hom X₃' Z₂₃'
    w₂₃' : Quiver.Hom Z₂₃' ((CategoryTheory.shiftFunctor C 1).obj X₂')
    h₂₃' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    v₁₃' : Quiver.Hom X₃' Z₁₃'
    w₁₃' : Quiver.Hom Z₁₃' ((CategoryTheory.shiftFunctor C 1).obj X₁')
    h₁₃' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    H : CategoryTheory.Triangulated.Octahedron comm' h₁₂' h₂₃' h₁₃'
    iso₁₂ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₁₂ v₁₂ …
    iso₂₃ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₂₃ v₂₃ …
    iso₁₃ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₁₃ v₁₃ …
    eq₁₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated. …
    eq₁₂' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated …
    eq₁₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated. …
    eq₁₃' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated …
    eq₂₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated. …
    eq₂₃' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated …
    ⊢ CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
  -/
  have rel₁₂ := H.triangleMorphism₁.comm₂
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{?u.40535, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    X₁ X₂ X₃ Z₁₂ Z₂₃ Z₁₃ : C
    u₁₂ : Quiver.Hom X₁ X₂
    u₂₃ : Quiver.Hom X₂ X₃
    u₁₃ : Quiver.Hom X₁ X₃
    comm : Eq (CategoryTheory.CategoryStruct.comp u₁₂ u₂₃) u₁₃
    v₁₂ : Quiver.Hom X₂ Z₁₂
    w₁₂ : Quiver.Hom Z₁₂ ((CategoryTheory.shiftFunctor C 1).obj X₁)
    h₁₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    v₂₃ : Quiver.Hom X₃ Z₂₃
    w₂₃ : Quiver.Hom Z₂₃ ((CategoryTheory.shiftFunctor C 1).obj X₂)
    h₂₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    v₁₃ : Quiver.Hom X₃ Z₁₃
    w₁₃ : Quiver.Hom Z₁₃ ((CategoryTheory.shiftFunctor C 1).obj X₁)
    h₁₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    h : CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
    X₁' X₂' X₃' Z₁₂' Z₂₃' Z₁₃' : C
    u₁₂' : Quiver.Hom X₁' X₂'
    u₂₃' : Quiver.Hom X₂' X₃'
    u₁₃' : Quiver.Hom X₁' X₃'
    comm' : Eq (CategoryTheory.CategoryStruct.comp u₁₂' u₂₃') u₁₃'
    e₁ : CategoryTheory.Iso X₁ X₁'
    e₂ : CategoryTheory.Iso X₂ X₂'
    e₃ : CategoryTheory.Iso X₃ X₃'
    comm₁₂ : Eq (CategoryTheory.CategoryStruct.comp u₁₂ e₂.hom) (CategoryTheory.Ca …
    comm₂₃ : Eq (CategoryTheory.CategoryStruct.comp u₂₃ e₃.hom) (CategoryTheory.Ca …
    v₁₂' : Quiver.Hom X₂' Z₁₂'
    w₁₂' : Quiver.Hom Z₁₂' ((CategoryTheory.shiftFunctor C 1).obj X₁')
    h₁₂' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    v₂₃' : Quiver.Hom X₃' Z₂₃'
    w₂₃' : Quiver.Hom Z₂₃' ((CategoryTheory.shiftFunctor C 1).obj X₂')
    h₂₃' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    v₁₃' : Quiver.Hom X₃' Z₁₃'
    w₁₃' : Quiver.Hom Z₁₃' ((CategoryTheory.shiftFunctor C 1).obj X₁')
    h₁₃' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    H : CategoryTheory.Triangulated.Octahedron comm' h₁₂' h₂₃' h₁₃'
    iso₁₂ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₁₂ v₁₂ …
    iso₂₃ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₂₃ v₂₃ …
    iso₁₃ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₁₃ v₁₃ …
    eq₁₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated. …
    eq₁₂' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated …
    eq₁₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated. …
    eq₁₃' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated …
    eq₂₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated. …
    eq₂₃' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated …
    rel₁₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated …
    ⊢ CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
  -/
  have rel₁₃ := H.triangleMorphism₁.comm₃
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{?u.40535, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    X₁ X₂ X₃ Z₁₂ Z₂₃ Z₁₃ : C
    u₁₂ : Quiver.Hom X₁ X₂
    u₂₃ : Quiver.Hom X₂ X₃
    u₁₃ : Quiver.Hom X₁ X₃
    comm : Eq (CategoryTheory.CategoryStruct.comp u₁₂ u₂₃) u₁₃
    v₁₂ : Quiver.Hom X₂ Z₁₂
    w₁₂ : Quiver.Hom Z₁₂ ((CategoryTheory.shiftFunctor C 1).obj X₁)
    h₁₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    v₂₃ : Quiver.Hom X₃ Z₂₃
    w₂₃ : Quiver.Hom Z₂₃ ((CategoryTheory.shiftFunctor C 1).obj X₂)
    h₂₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    v₁₃ : Quiver.Hom X₃ Z₁₃
    w₁₃ : Quiver.Hom Z₁₃ ((CategoryTheory.shiftFunctor C 1).obj X₁)
    h₁₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    h : CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
    X₁' X₂' X₃' Z₁₂' Z₂₃' Z₁₃' : C
    u₁₂' : Quiver.Hom X₁' X₂'
    u₂₃' : Quiver.Hom X₂' X₃'
    u₁₃' : Quiver.Hom X₁' X₃'
    comm' : Eq (CategoryTheory.CategoryStruct.comp u₁₂' u₂₃') u₁₃'
    e₁ : CategoryTheory.Iso X₁ X₁'
    e₂ : CategoryTheory.Iso X₂ X₂'
    e₃ : CategoryTheory.Iso X₃ X₃'
    comm₁₂ : Eq (CategoryTheory.CategoryStruct.comp u₁₂ e₂.hom) (CategoryTheory.Ca …
    comm₂₃ : Eq (CategoryTheory.CategoryStruct.comp u₂₃ e₃.hom) (CategoryTheory.Ca …
    v₁₂' : Quiver.Hom X₂' Z₁₂'
    w₁₂' : Quiver.Hom Z₁₂' ((CategoryTheory.shiftFunctor C 1).obj X₁')
    h₁₂' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    v₂₃' : Quiver.Hom X₃' Z₂₃'
    w₂₃' : Quiver.Hom Z₂₃' ((CategoryTheory.shiftFunctor C 1).obj X₂')
    h₂₃' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    v₁₃' : Quiver.Hom X₃' Z₁₃'
    w₁₃' : Quiver.Hom Z₁₃' ((CategoryTheory.shiftFunctor C 1).obj X₁')
    h₁₃' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    H : CategoryTheory.Triangulated.Octahedron comm' h₁₂' h₂₃' h₁₃'
    iso₁₂ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₁₂ v₁₂ …
    iso₂₃ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₂₃ v₂₃ …
    iso₁₃ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₁₃ v₁₃ …
    eq₁₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated. …
    eq₁₂' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated …
    eq₁₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated. …
    eq₁₃' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated …
    eq₂₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated. …
    eq₂₃' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated …
    rel₁₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated …
    rel₁₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated …
    ⊢ CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
  -/
  have rel₂₂ := H.triangleMorphism₂.comm₂
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{?u.40535, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    X₁ X₂ X₃ Z₁₂ Z₂₃ Z₁₃ : C
    u₁₂ : Quiver.Hom X₁ X₂
    u₂₃ : Quiver.Hom X₂ X₃
    u₁₃ : Quiver.Hom X₁ X₃
    comm : Eq (CategoryTheory.CategoryStruct.comp u₁₂ u₂₃) u₁₃
    v₁₂ : Quiver.Hom X₂ Z₁₂
    w₁₂ : Quiver.Hom Z₁₂ ((CategoryTheory.shiftFunctor C 1).obj X₁)
    h₁₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    v₂₃ : Quiver.Hom X₃ Z₂₃
    w₂₃ : Quiver.Hom Z₂₃ ((CategoryTheory.shiftFunctor C 1).obj X₂)
    h₂₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    v₁₃ : Quiver.Hom X₃ Z₁₃
    w₁₃ : Quiver.Hom Z₁₃ ((CategoryTheory.shiftFunctor C 1).obj X₁)
    h₁₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    h : CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
    X₁' X₂' X₃' Z₁₂' Z₂₃' Z₁₃' : C
    u₁₂' : Quiver.Hom X₁' X₂'
    u₂₃' : Quiver.Hom X₂' X₃'
    u₁₃' : Quiver.Hom X₁' X₃'
    comm' : Eq (CategoryTheory.CategoryStruct.comp u₁₂' u₂₃') u₁₃'
    e₁ : CategoryTheory.Iso X₁ X₁'
    e₂ : CategoryTheory.Iso X₂ X₂'
    e₃ : CategoryTheory.Iso X₃ X₃'
    comm₁₂ : Eq (CategoryTheory.CategoryStruct.comp u₁₂ e₂.hom) (CategoryTheory.Ca …
    comm₂₃ : Eq (CategoryTheory.CategoryStruct.comp u₂₃ e₃.hom) (CategoryTheory.Ca …
    v₁₂' : Quiver.Hom X₂' Z₁₂'
    w₁₂' : Quiver.Hom Z₁₂' ((CategoryTheory.shiftFunctor C 1).obj X₁')
    h₁₂' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    v₂₃' : Quiver.Hom X₃' Z₂₃'
    w₂₃' : Quiver.Hom Z₂₃' ((CategoryTheory.shiftFunctor C 1).obj X₂')
    h₂₃' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    v₁₃' : Quiver.Hom X₃' Z₁₃'
    w₁₃' : Quiver.Hom Z₁₃' ((CategoryTheory.shiftFunctor C 1).obj X₁')
    h₁₃' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    H : CategoryTheory.Triangulated.Octahedron comm' h₁₂' h₂₃' h₁₃'
    iso₁₂ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₁₂ v₁₂ …
    iso₂₃ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₂₃ v₂₃ …
    iso₁₃ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₁₃ v₁₃ …
    eq₁₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated. …
    eq₁₂' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated …
    eq₁₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated. …
    eq₁₃' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated …
    eq₂₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated. …
    eq₂₃' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated …
    rel₁₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated …
    rel₁₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated …
    rel₂₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated …
    ⊢ CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
  -/
  have rel₂₃ := H.triangleMorphism₂.comm₃
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{?u.40535, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    X₁ X₂ X₃ Z₁₂ Z₂₃ Z₁₃ : C
    u₁₂ : Quiver.Hom X₁ X₂
    u₂₃ : Quiver.Hom X₂ X₃
    u₁₃ : Quiver.Hom X₁ X₃
    comm : Eq (CategoryTheory.CategoryStruct.comp u₁₂ u₂₃) u₁₃
    v₁₂ : Quiver.Hom X₂ Z₁₂
    w₁₂ : Quiver.Hom Z₁₂ ((CategoryTheory.shiftFunctor C 1).obj X₁)
    h₁₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    v₂₃ : Quiver.Hom X₃ Z₂₃
    w₂₃ : Quiver.Hom Z₂₃ ((CategoryTheory.shiftFunctor C 1).obj X₂)
    h₂₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    v₁₃ : Quiver.Hom X₃ Z₁₃
    w₁₃ : Quiver.Hom Z₁₃ ((CategoryTheory.shiftFunctor C 1).obj X₁)
    h₁₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    h : CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
    X₁' X₂' X₃' Z₁₂' Z₂₃' Z₁₃' : C
    u₁₂' : Quiver.Hom X₁' X₂'
    u₂₃' : Quiver.Hom X₂' X₃'
    u₁₃' : Quiver.Hom X₁' X₃'
    comm' : Eq (CategoryTheory.CategoryStruct.comp u₁₂' u₂₃') u₁₃'
    e₁ : CategoryTheory.Iso X₁ X₁'
    e₂ : CategoryTheory.Iso X₂ X₂'
    e₃ : CategoryTheory.Iso X₃ X₃'
    comm₁₂ : Eq (CategoryTheory.CategoryStruct.comp u₁₂ e₂.hom) (CategoryTheory.Ca …
    comm₂₃ : Eq (CategoryTheory.CategoryStruct.comp u₂₃ e₃.hom) (CategoryTheory.Ca …
    v₁₂' : Quiver.Hom X₂' Z₁₂'
    w₁₂' : Quiver.Hom Z₁₂' ((CategoryTheory.shiftFunctor C 1).obj X₁')
    h₁₂' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    v₂₃' : Quiver.Hom X₃' Z₂₃'
    w₂₃' : Quiver.Hom Z₂₃' ((CategoryTheory.shiftFunctor C 1).obj X₂')
    h₂₃' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    v₁₃' : Quiver.Hom X₃' Z₁₃'
    w₁₃' : Quiver.Hom Z₁₃' ((CategoryTheory.shiftFunctor C 1).obj X₁')
    h₁₃' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    H : CategoryTheory.Triangulated.Octahedron comm' h₁₂' h₂₃' h₁₃'
    iso₁₂ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₁₂ v₁₂ …
    iso₂₃ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₂₃ v₂₃ …
    iso₁₃ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₁₃ v₁₃ …
    eq₁₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated. …
    eq₁₂' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated …
    eq₁₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated. …
    eq₁₃' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated …
    eq₂₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated. …
    eq₂₃' : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated …
    rel₁₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated …
    rel₁₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated …
    rel₂₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated …
    rel₂₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated …
    ⊢ CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
  -/
  dsimp [iso₁₂, iso₂₃, iso₁₃] at eq₁₂ eq₁₂' eq₁₃ eq₁₃' eq₂₃ eq₂₃' rel₁₂ rel₁₃ rel₂₂ rel₂₃
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{?u.40535, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    X₁ X₂ X₃ Z₁₂ Z₂₃ Z₁₃ : C
    u₁₂ : Quiver.Hom X₁ X₂
    u₂₃ : Quiver.Hom X₂ X₃
    u₁₃ : Quiver.Hom X₁ X₃
    comm : Eq (CategoryTheory.CategoryStruct.comp u₁₂ u₂₃) u₁₃
    v₁₂ : Quiver.Hom X₂ Z₁₂
    w₁₂ : Quiver.Hom Z₁₂ ((CategoryTheory.shiftFunctor C 1).obj X₁)
    h₁₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    v₂₃ : Quiver.Hom X₃ Z₂₃
    w₂₃ : Quiver.Hom Z₂₃ ((CategoryTheory.shiftFunctor C 1).obj X₂)
    h₂₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    v₁₃ : Quiver.Hom X₃ Z₁₃
    w₁₃ : Quiver.Hom Z₁₃ ((CategoryTheory.shiftFunctor C 1).obj X₁)
    h₁₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    h : CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
    X₁' X₂' X₃' Z₁₂' Z₂₃' Z₁₃' : C
    u₁₂' : Quiver.Hom X₁' X₂'
    u₂₃' : Quiver.Hom X₂' X₃'
    u₁₃' : Quiver.Hom X₁' X₃'
    comm' : Eq (CategoryTheory.CategoryStruct.comp u₁₂' u₂₃') u₁₃'
    e₁ : CategoryTheory.Iso X₁ X₁'
    e₂ : CategoryTheory.Iso X₂ X₂'
    e₃ : CategoryTheory.Iso X₃ X₃'
    comm₁₂ : Eq (CategoryTheory.CategoryStruct.comp u₁₂ e₂.hom) (CategoryTheory.Ca …
    comm₂₃ : Eq (CategoryTheory.CategoryStruct.comp u₂₃ e₃.hom) (CategoryTheory.Ca …
    v₁₂' : Quiver.Hom X₂' Z₁₂'
    w₁₂' : Quiver.Hom Z₁₂' ((CategoryTheory.shiftFunctor C 1).obj X₁')
    h₁₂' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    v₂₃' : Quiver.Hom X₃' Z₂₃'
    w₂₃' : Quiver.Hom Z₂₃' ((CategoryTheory.shiftFunctor C 1).obj X₂')
    h₂₃' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    v₁₃' : Quiver.Hom X₃' Z₁₃'
    w₁₃' : Quiver.Hom Z₁₃' ((CategoryTheory.shiftFunctor C 1).obj X₁')
    h₁₃' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    H : CategoryTheory.Triangulated.Octahedron comm' h₁₂' h₂₃' h₁₃'
    iso₁₂ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₁₂ v₁₂ …
    iso₂₃ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₂₃ v₂₃ …
    iso₁₃ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₁₃ v₁₃ …
    eq₁₂ : Eq (CategoryTheory.CategoryStruct.comp v₁₂ (CategoryTheory.Pretriangula …
    eq₁₂' : Eq (CategoryTheory.CategoryStruct.comp w₁₂ ((CategoryTheory.shiftFunct …
    eq₁₃ : Eq (CategoryTheory.CategoryStruct.comp v₁₃ (CategoryTheory.Pretriangula …
    eq₁₃' : Eq (CategoryTheory.CategoryStruct.comp w₁₃ ((CategoryTheory.shiftFunct …
    eq₂₃ : Eq (CategoryTheory.CategoryStruct.comp v₂₃ (CategoryTheory.Pretriangula …
    eq₂₃' : Eq (CategoryTheory.CategoryStruct.comp w₂₃ ((CategoryTheory.shiftFunct …
    rel₁₂ : Eq (CategoryTheory.CategoryStruct.comp v₁₂' H.m₁) (CategoryTheory.Cate …
    rel₁₃ : Eq (CategoryTheory.CategoryStruct.comp w₁₂' ((CategoryTheory.shiftFunc …
    rel₂₂ : Eq (CategoryTheory.CategoryStruct.comp v₁₃' H.m₃) (CategoryTheory.Cate …
    rel₂₃ : Eq (CategoryTheory.CategoryStruct.comp w₁₃' ((CategoryTheory.shiftFunc …
    ⊢ CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
  -/
  rw [Functor.map_id, comp_id] at rel₁₃
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{?u.40535, u_1} C
    inst✝⁴ : CategoryTheory.Preadditive C
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.HasShift C Int
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    X₁ X₂ X₃ Z₁₂ Z₂₃ Z₁₃ : C
    u₁₂ : Quiver.Hom X₁ X₂
    u₂₃ : Quiver.Hom X₂ X₃
    u₁₃ : Quiver.Hom X₁ X₃
    comm : Eq (CategoryTheory.CategoryStruct.comp u₁₂ u₂₃) u₁₃
    v₁₂ : Quiver.Hom X₂ Z₁₂
    w₁₂ : Quiver.Hom Z₁₂ ((CategoryTheory.shiftFunctor C 1).obj X₁)
    h₁₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    v₂₃ : Quiver.Hom X₃ Z₂₃
    w₂₃ : Quiver.Hom Z₂₃ ((CategoryTheory.shiftFunctor C 1).obj X₂)
    h₂₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    v₁₃ : Quiver.Hom X₃ Z₁₃
    w₁₃ : Quiver.Hom Z₁₃ ((CategoryTheory.shiftFunctor C 1).obj X₁)
    h₁₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    h : CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
    X₁' X₂' X₃' Z₁₂' Z₂₃' Z₁₃' : C
    u₁₂' : Quiver.Hom X₁' X₂'
    u₂₃' : Quiver.Hom X₂' X₃'
    u₁₃' : Quiver.Hom X₁' X₃'
    comm' : Eq (CategoryTheory.CategoryStruct.comp u₁₂' u₂₃') u₁₃'
    e₁ : CategoryTheory.Iso X₁ X₁'
    e₂ : CategoryTheory.Iso X₂ X₂'
    e₃ : CategoryTheory.Iso X₃ X₃'
    comm₁₂ : Eq (CategoryTheory.CategoryStruct.comp u₁₂ e₂.hom) (CategoryTheory.Ca …
    comm₂₃ : Eq (CategoryTheory.CategoryStruct.comp u₂₃ e₃.hom) (CategoryTheory.Ca …
    v₁₂' : Quiver.Hom X₂' Z₁₂'
    w₁₂' : Quiver.Hom Z₁₂' ((CategoryTheory.shiftFunctor C 1).obj X₁')
    h₁₂' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    v₂₃' : Quiver.Hom X₃' Z₂₃'
    w₂₃' : Quiver.Hom Z₂₃' ((CategoryTheory.shiftFunctor C 1).obj X₂')
    h₂₃' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    v₁₃' : Quiver.Hom X₃' Z₁₃'
    w₁₃' : Quiver.Hom Z₁₃' ((CategoryTheory.shiftFunctor C 1).obj X₁')
    h₁₃' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
    H : CategoryTheory.Triangulated.Octahedron comm' h₁₂' h₂₃' h₁₃'
    iso₁₂ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₁₂ v₁₂ …
    iso₂₃ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₂₃ v₂₃ …
    iso₁₃ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₁₃ v₁₃ …
    eq₁₂ : Eq (CategoryTheory.CategoryStruct.comp v₁₂ (CategoryTheory.Pretriangula …
    eq₁₂' : Eq (CategoryTheory.CategoryStruct.comp w₁₂ ((CategoryTheory.shiftFunct …
    eq₁₃ : Eq (CategoryTheory.CategoryStruct.comp v₁₃ (CategoryTheory.Pretriangula …
    eq₁₃' : Eq (CategoryTheory.CategoryStruct.comp w₁₃ ((CategoryTheory.shiftFunct …
    eq₂₃ : Eq (CategoryTheory.CategoryStruct.comp v₂₃ (CategoryTheory.Pretriangula …
    eq₂₃' : Eq (CategoryTheory.CategoryStruct.comp w₂₃ ((CategoryTheory.shiftFunct …
    rel₁₂ : Eq (CategoryTheory.CategoryStruct.comp v₁₂' H.m₁) (CategoryTheory.Cate …
    rel₁₃ : Eq w₁₂' (CategoryTheory.CategoryStruct.comp H.m₁ w₁₃')
    rel₂₂ : Eq (CategoryTheory.CategoryStruct.comp v₁₃' H.m₃) (CategoryTheory.Cate …
    rel₂₃ : Eq (CategoryTheory.CategoryStruct.comp w₁₃' ((CategoryTheory.shiftFunc …
    ⊢ CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
  -/
  rw [id_comp] at rel₂₂
  refine ⟨iso₁₂.hom.hom₃ ≫ H.m₁ ≫ iso₁₃.inv.hom₃,
    iso₁₃.hom.hom₃ ≫ H.m₃ ≫ iso₂₃.inv.hom₃, ?_, ?_, ?_, ?_, ?_⟩
  · rw [reassoc_of% eq₁₂, ← cancel_mono iso₁₃.hom.hom₃, assoc, assoc, assoc, assoc,
      iso₁₃.inv_hom_id_triangle_hom₃, eq₁₃, reassoc_of% comm₂₃, ← rel₁₂]
    /-
      case refine_1
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.40535, u_1} C
      inst✝⁴ : CategoryTheory.Preadditive C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      X₁ X₂ X₃ Z₁₂ Z₂₃ Z₁₃ : C
      u₁₂ : Quiver.Hom X₁ X₂
      u₂₃ : Quiver.Hom X₂ X₃
      u₁₃ : Quiver.Hom X₁ X₃
      comm : Eq (CategoryTheory.CategoryStruct.comp u₁₂ u₂₃) u₁₃
      v₁₂ : Quiver.Hom X₂ Z₁₂
      w₁₂ : Quiver.Hom Z₁₂ ((CategoryTheory.shiftFunctor C 1).obj X₁)
      h₁₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      v₂₃ : Quiver.Hom X₃ Z₂₃
      w₂₃ : Quiver.Hom Z₂₃ ((CategoryTheory.shiftFunctor C 1).obj X₂)
      h₂₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      v₁₃ : Quiver.Hom X₃ Z₁₃
      w₁₃ : Quiver.Hom Z₁₃ ((CategoryTheory.shiftFunctor C 1).obj X₁)
      h₁₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      h : CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
      X₁' X₂' X₃' Z₁₂' Z₂₃' Z₁₃' : C
      u₁₂' : Quiver.Hom X₁' X₂'
      u₂₃' : Quiver.Hom X₂' X₃'
      u₁₃' : Quiver.Hom X₁' X₃'
      comm' : Eq (CategoryTheory.CategoryStruct.comp u₁₂' u₂₃') u₁₃'
      e₁ : CategoryTheory.Iso X₁ X₁'
      e₂ : CategoryTheory.Iso X₂ X₂'
      e₃ : CategoryTheory.Iso X₃ X₃'
      comm₁₂ : Eq (CategoryTheory.CategoryStruct.comp u₁₂ e₂.hom) (CategoryTheory.Ca …
      comm₂₃ : Eq (CategoryTheory.CategoryStruct.comp u₂₃ e₃.hom) (CategoryTheory.Ca …
      v₁₂' : Quiver.Hom X₂' Z₁₂'
      w₁₂' : Quiver.Hom Z₁₂' ((CategoryTheory.shiftFunctor C 1).obj X₁')
      h₁₂' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
      v₂₃' : Quiver.Hom X₃' Z₂₃'
      w₂₃' : Quiver.Hom Z₂₃' ((CategoryTheory.shiftFunctor C 1).obj X₂')
      h₂₃' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
      v₁₃' : Quiver.Hom X₃' Z₁₃'
      w₁₃' : Quiver.Hom Z₁₃' ((CategoryTheory.shiftFunctor C 1).obj X₁')
      h₁₃' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
      H : CategoryTheory.Triangulated.Octahedron comm' h₁₂' h₂₃' h₁₃'
      iso₁₂ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₁₂ v₁₂ …
      iso₂₃ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₂₃ v₂₃ …
      iso₁₃ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₁₃ v₁₃ …
      eq₁₂ : Eq (CategoryTheory.CategoryStruct.comp v₁₂ (CategoryTheory.Pretriangula …
      eq₁₂' : Eq (CategoryTheory.CategoryStruct.comp w₁₂ ((CategoryTheory.shiftFunct …
      eq₁₃ : Eq (CategoryTheory.CategoryStruct.comp v₁₃ (CategoryTheory.Pretriangula …
      eq₁₃' : Eq (CategoryTheory.CategoryStruct.comp w₁₃ ((CategoryTheory.shiftFunct …
      eq₂₃ : Eq (CategoryTheory.CategoryStruct.comp v₂₃ (CategoryTheory.Pretriangula …
      eq₂₃' : Eq (CategoryTheory.CategoryStruct.comp w₂₃ ((CategoryTheory.shiftFunct …
      rel₁₂ : Eq (CategoryTheory.CategoryStruct.comp v₁₂' H.m₁) (CategoryTheory.Cate …
      rel₁₃ : Eq w₁₂' (CategoryTheory.CategoryStruct.comp H.m₁ w₁₃')
      rel₂₂ : Eq (CategoryTheory.CategoryStruct.comp v₁₃' H.m₃) v₂₃'
      rel₂₃ : Eq (CategoryTheory.CategoryStruct.comp w₁₃' ((CategoryTheory.shiftFunc …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp e₂.hom (CategoryTheory.CategoryStruct …
    -/
    dsimp
    /-
      case refine_1
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.40535, u_1} C
      inst✝⁴ : CategoryTheory.Preadditive C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      X₁ X₂ X₃ Z₁₂ Z₂₃ Z₁₃ : C
      u₁₂ : Quiver.Hom X₁ X₂
      u₂₃ : Quiver.Hom X₂ X₃
      u₁₃ : Quiver.Hom X₁ X₃
      comm : Eq (CategoryTheory.CategoryStruct.comp u₁₂ u₂₃) u₁₃
      v₁₂ : Quiver.Hom X₂ Z₁₂
      w₁₂ : Quiver.Hom Z₁₂ ((CategoryTheory.shiftFunctor C 1).obj X₁)
      h₁₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      v₂₃ : Quiver.Hom X₃ Z₂₃
      w₂₃ : Quiver.Hom Z₂₃ ((CategoryTheory.shiftFunctor C 1).obj X₂)
      h₂₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      v₁₃ : Quiver.Hom X₃ Z₁₃
      w₁₃ : Quiver.Hom Z₁₃ ((CategoryTheory.shiftFunctor C 1).obj X₁)
      h₁₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      h : CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
      X₁' X₂' X₃' Z₁₂' Z₂₃' Z₁₃' : C
      u₁₂' : Quiver.Hom X₁' X₂'
      u₂₃' : Quiver.Hom X₂' X₃'
      u₁₃' : Quiver.Hom X₁' X₃'
      comm' : Eq (CategoryTheory.CategoryStruct.comp u₁₂' u₂₃') u₁₃'
      e₁ : CategoryTheory.Iso X₁ X₁'
      e₂ : CategoryTheory.Iso X₂ X₂'
      e₃ : CategoryTheory.Iso X₃ X₃'
      comm₁₂ : Eq (CategoryTheory.CategoryStruct.comp u₁₂ e₂.hom) (CategoryTheory.Ca …
      comm₂₃ : Eq (CategoryTheory.CategoryStruct.comp u₂₃ e₃.hom) (CategoryTheory.Ca …
      v₁₂' : Quiver.Hom X₂' Z₁₂'
      w₁₂' : Quiver.Hom Z₁₂' ((CategoryTheory.shiftFunctor C 1).obj X₁')
      h₁₂' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
      v₂₃' : Quiver.Hom X₃' Z₂₃'
      w₂₃' : Quiver.Hom Z₂₃' ((CategoryTheory.shiftFunctor C 1).obj X₂')
      h₂₃' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
      v₁₃' : Quiver.Hom X₃' Z₁₃'
      w₁₃' : Quiver.Hom Z₁₃' ((CategoryTheory.shiftFunctor C 1).obj X₁')
      h₁₃' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
      H : CategoryTheory.Triangulated.Octahedron comm' h₁₂' h₂₃' h₁₃'
      iso₁₂ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₁₂ v₁₂ …
      iso₂₃ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₂₃ v₂₃ …
      iso₁₃ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₁₃ v₁₃ …
      eq₁₂ : Eq (CategoryTheory.CategoryStruct.comp v₁₂ (CategoryTheory.Pretriangula …
      eq₁₂' : Eq (CategoryTheory.CategoryStruct.comp w₁₂ ((CategoryTheory.shiftFunct …
      eq₁₃ : Eq (CategoryTheory.CategoryStruct.comp v₁₃ (CategoryTheory.Pretriangula …
      eq₁₃' : Eq (CategoryTheory.CategoryStruct.comp w₁₃ ((CategoryTheory.shiftFunct …
      eq₂₃ : Eq (CategoryTheory.CategoryStruct.comp v₂₃ (CategoryTheory.Pretriangula …
      eq₂₃' : Eq (CategoryTheory.CategoryStruct.comp w₂₃ ((CategoryTheory.shiftFunct …
      rel₁₂ : Eq (CategoryTheory.CategoryStruct.comp v₁₂' H.m₁) (CategoryTheory.Cate …
      rel₁₃ : Eq w₁₂' (CategoryTheory.CategoryStruct.comp H.m₁ w₁₃')
      rel₂₂ : Eq (CategoryTheory.CategoryStruct.comp v₁₃' H.m₃) v₂₃'
      rel₂₃ : Eq (CategoryTheory.CategoryStruct.comp w₁₃' ((CategoryTheory.shiftFunc …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp e₂.hom (CategoryTheory.CategoryStruct …
    -/
    rw [comp_id]
    /-
      🎉 no goals
    -/
  · rw [← cancel_mono (e₁.hom⟦(1 : ℤ)⟧'), eq₁₂', assoc, assoc, assoc, eq₁₃',
      iso₁₃.inv_hom_id_triangle_hom₃_assoc, ← rel₁₃]
  · rw [reassoc_of% eq₁₃, reassoc_of% rel₂₂, ← cancel_mono iso₂₃.hom.hom₃, assoc, assoc,
      iso₂₃.inv_hom_id_triangle_hom₃, eq₂₃]
    /-
      case refine_3
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.40535, u_1} C
      inst✝⁴ : CategoryTheory.Preadditive C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      X₁ X₂ X₃ Z₁₂ Z₂₃ Z₁₃ : C
      u₁₂ : Quiver.Hom X₁ X₂
      u₂₃ : Quiver.Hom X₂ X₃
      u₁₃ : Quiver.Hom X₁ X₃
      comm : Eq (CategoryTheory.CategoryStruct.comp u₁₂ u₂₃) u₁₃
      v₁₂ : Quiver.Hom X₂ Z₁₂
      w₁₂ : Quiver.Hom Z₁₂ ((CategoryTheory.shiftFunctor C 1).obj X₁)
      h₁₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      v₂₃ : Quiver.Hom X₃ Z₂₃
      w₂₃ : Quiver.Hom Z₂₃ ((CategoryTheory.shiftFunctor C 1).obj X₂)
      h₂₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      v₁₃ : Quiver.Hom X₃ Z₁₃
      w₁₃ : Quiver.Hom Z₁₃ ((CategoryTheory.shiftFunctor C 1).obj X₁)
      h₁₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      h : CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
      X₁' X₂' X₃' Z₁₂' Z₂₃' Z₁₃' : C
      u₁₂' : Quiver.Hom X₁' X₂'
      u₂₃' : Quiver.Hom X₂' X₃'
      u₁₃' : Quiver.Hom X₁' X₃'
      comm' : Eq (CategoryTheory.CategoryStruct.comp u₁₂' u₂₃') u₁₃'
      e₁ : CategoryTheory.Iso X₁ X₁'
      e₂ : CategoryTheory.Iso X₂ X₂'
      e₃ : CategoryTheory.Iso X₃ X₃'
      comm₁₂ : Eq (CategoryTheory.CategoryStruct.comp u₁₂ e₂.hom) (CategoryTheory.Ca …
      comm₂₃ : Eq (CategoryTheory.CategoryStruct.comp u₂₃ e₃.hom) (CategoryTheory.Ca …
      v₁₂' : Quiver.Hom X₂' Z₁₂'
      w₁₂' : Quiver.Hom Z₁₂' ((CategoryTheory.shiftFunctor C 1).obj X₁')
      h₁₂' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
      v₂₃' : Quiver.Hom X₃' Z₂₃'
      w₂₃' : Quiver.Hom Z₂₃' ((CategoryTheory.shiftFunctor C 1).obj X₂')
      h₂₃' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
      v₁₃' : Quiver.Hom X₃' Z₁₃'
      w₁₃' : Quiver.Hom Z₁₃' ((CategoryTheory.shiftFunctor C 1).obj X₁')
      h₁₃' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
      H : CategoryTheory.Triangulated.Octahedron comm' h₁₂' h₂₃' h₁₃'
      iso₁₂ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₁₂ v₁₂ …
      iso₂₃ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₂₃ v₂₃ …
      iso₁₃ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₁₃ v₁₃ …
      eq₁₂ : Eq (CategoryTheory.CategoryStruct.comp v₁₂ (CategoryTheory.Pretriangula …
      eq₁₂' : Eq (CategoryTheory.CategoryStruct.comp w₁₂ ((CategoryTheory.shiftFunct …
      eq₁₃ : Eq (CategoryTheory.CategoryStruct.comp v₁₃ (CategoryTheory.Pretriangula …
      eq₁₃' : Eq (CategoryTheory.CategoryStruct.comp w₁₃ ((CategoryTheory.shiftFunct …
      eq₂₃ : Eq (CategoryTheory.CategoryStruct.comp v₂₃ (CategoryTheory.Pretriangula …
      eq₂₃' : Eq (CategoryTheory.CategoryStruct.comp w₂₃ ((CategoryTheory.shiftFunct …
      rel₁₂ : Eq (CategoryTheory.CategoryStruct.comp v₁₂' H.m₁) (CategoryTheory.Cate …
      rel₁₃ : Eq w₁₂' (CategoryTheory.CategoryStruct.comp H.m₁ w₁₃')
      rel₂₂ : Eq (CategoryTheory.CategoryStruct.comp v₁₃' H.m₃) v₂₃'
      rel₂₃ : Eq (CategoryTheory.CategoryStruct.comp w₁₃' ((CategoryTheory.shiftFunc …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp e₃.hom (CategoryTheory.CategoryStruct …
    -/
    dsimp
    /-
      case refine_3
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.40535, u_1} C
      inst✝⁴ : CategoryTheory.Preadditive C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      X₁ X₂ X₃ Z₁₂ Z₂₃ Z₁₃ : C
      u₁₂ : Quiver.Hom X₁ X₂
      u₂₃ : Quiver.Hom X₂ X₃
      u₁₃ : Quiver.Hom X₁ X₃
      comm : Eq (CategoryTheory.CategoryStruct.comp u₁₂ u₂₃) u₁₃
      v₁₂ : Quiver.Hom X₂ Z₁₂
      w₁₂ : Quiver.Hom Z₁₂ ((CategoryTheory.shiftFunctor C 1).obj X₁)
      h₁₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      v₂₃ : Quiver.Hom X₃ Z₂₃
      w₂₃ : Quiver.Hom Z₂₃ ((CategoryTheory.shiftFunctor C 1).obj X₂)
      h₂₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      v₁₃ : Quiver.Hom X₃ Z₁₃
      w₁₃ : Quiver.Hom Z₁₃ ((CategoryTheory.shiftFunctor C 1).obj X₁)
      h₁₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      h : CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
      X₁' X₂' X₃' Z₁₂' Z₂₃' Z₁₃' : C
      u₁₂' : Quiver.Hom X₁' X₂'
      u₂₃' : Quiver.Hom X₂' X₃'
      u₁₃' : Quiver.Hom X₁' X₃'
      comm' : Eq (CategoryTheory.CategoryStruct.comp u₁₂' u₂₃') u₁₃'
      e₁ : CategoryTheory.Iso X₁ X₁'
      e₂ : CategoryTheory.Iso X₂ X₂'
      e₃ : CategoryTheory.Iso X₃ X₃'
      comm₁₂ : Eq (CategoryTheory.CategoryStruct.comp u₁₂ e₂.hom) (CategoryTheory.Ca …
      comm₂₃ : Eq (CategoryTheory.CategoryStruct.comp u₂₃ e₃.hom) (CategoryTheory.Ca …
      v₁₂' : Quiver.Hom X₂' Z₁₂'
      w₁₂' : Quiver.Hom Z₁₂' ((CategoryTheory.shiftFunctor C 1).obj X₁')
      h₁₂' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
      v₂₃' : Quiver.Hom X₃' Z₂₃'
      w₂₃' : Quiver.Hom Z₂₃' ((CategoryTheory.shiftFunctor C 1).obj X₂')
      h₂₃' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
      v₁₃' : Quiver.Hom X₃' Z₁₃'
      w₁₃' : Quiver.Hom Z₁₃' ((CategoryTheory.shiftFunctor C 1).obj X₁')
      h₁₃' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
      H : CategoryTheory.Triangulated.Octahedron comm' h₁₂' h₂₃' h₁₃'
      iso₁₂ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₁₂ v₁₂ …
      iso₂₃ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₂₃ v₂₃ …
      iso₁₃ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₁₃ v₁₃ …
      eq₁₂ : Eq (CategoryTheory.CategoryStruct.comp v₁₂ (CategoryTheory.Pretriangula …
      eq₁₂' : Eq (CategoryTheory.CategoryStruct.comp w₁₂ ((CategoryTheory.shiftFunct …
      eq₁₃ : Eq (CategoryTheory.CategoryStruct.comp v₁₃ (CategoryTheory.Pretriangula …
      eq₁₃' : Eq (CategoryTheory.CategoryStruct.comp w₁₃ ((CategoryTheory.shiftFunct …
      eq₂₃ : Eq (CategoryTheory.CategoryStruct.comp v₂₃ (CategoryTheory.Pretriangula …
      eq₂₃' : Eq (CategoryTheory.CategoryStruct.comp w₂₃ ((CategoryTheory.shiftFunct …
      rel₁₂ : Eq (CategoryTheory.CategoryStruct.comp v₁₂' H.m₁) (CategoryTheory.Cate …
      rel₁₃ : Eq w₁₂' (CategoryTheory.CategoryStruct.comp H.m₁ w₁₃')
      rel₂₂ : Eq (CategoryTheory.CategoryStruct.comp v₁₃' H.m₃) v₂₃'
      rel₂₃ : Eq (CategoryTheory.CategoryStruct.comp w₁₃' ((CategoryTheory.shiftFunc …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp e₃.hom (CategoryTheory.CategoryStruct …
    -/
    rw [comp_id]
    /-
      🎉 no goals
    -/
  · rw [← cancel_mono (e₂.hom⟦(1 : ℤ)⟧'), assoc, assoc, assoc,assoc, eq₂₃',
      iso₂₃.inv_hom_id_triangle_hom₃_assoc, ← rel₂₃, ← Functor.map_comp, comm₁₂,
      Functor.map_comp, reassoc_of% eq₁₃']
    /-
      case refine_5
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.40535, u_1} C
      inst✝⁴ : CategoryTheory.Preadditive C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      X₁ X₂ X₃ Z₁₂ Z₂₃ Z₁₃ : C
      u₁₂ : Quiver.Hom X₁ X₂
      u₂₃ : Quiver.Hom X₂ X₃
      u₁₃ : Quiver.Hom X₁ X₃
      comm : Eq (CategoryTheory.CategoryStruct.comp u₁₂ u₂₃) u₁₃
      v₁₂ : Quiver.Hom X₂ Z₁₂
      w₁₂ : Quiver.Hom Z₁₂ ((CategoryTheory.shiftFunctor C 1).obj X₁)
      h₁₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      v₂₃ : Quiver.Hom X₃ Z₂₃
      w₂₃ : Quiver.Hom Z₂₃ ((CategoryTheory.shiftFunctor C 1).obj X₂)
      h₂₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      v₁₃ : Quiver.Hom X₃ Z₁₃
      w₁₃ : Quiver.Hom Z₁₃ ((CategoryTheory.shiftFunctor C 1).obj X₁)
      h₁₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      h : CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
      X₁' X₂' X₃' Z₁₂' Z₂₃' Z₁₃' : C
      u₁₂' : Quiver.Hom X₁' X₂'
      u₂₃' : Quiver.Hom X₂' X₃'
      u₁₃' : Quiver.Hom X₁' X₃'
      comm' : Eq (CategoryTheory.CategoryStruct.comp u₁₂' u₂₃') u₁₃'
      e₁ : CategoryTheory.Iso X₁ X₁'
      e₂ : CategoryTheory.Iso X₂ X₂'
      e₃ : CategoryTheory.Iso X₃ X₃'
      comm₁₂ : Eq (CategoryTheory.CategoryStruct.comp u₁₂ e₂.hom) (CategoryTheory.Ca …
      comm₂₃ : Eq (CategoryTheory.CategoryStruct.comp u₂₃ e₃.hom) (CategoryTheory.Ca …
      v₁₂' : Quiver.Hom X₂' Z₁₂'
      w₁₂' : Quiver.Hom Z₁₂' ((CategoryTheory.shiftFunctor C 1).obj X₁')
      h₁₂' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
      v₂₃' : Quiver.Hom X₃' Z₂₃'
      w₂₃' : Quiver.Hom Z₂₃' ((CategoryTheory.shiftFunctor C 1).obj X₂')
      h₂₃' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
      v₁₃' : Quiver.Hom X₃' Z₁₃'
      w₁₃' : Quiver.Hom Z₁₃' ((CategoryTheory.shiftFunctor C 1).obj X₁')
      h₁₃' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
      H : CategoryTheory.Triangulated.Octahedron comm' h₁₂' h₂₃' h₁₃'
      iso₁₂ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₁₂ v₁₂ …
      iso₂₃ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₂₃ v₂₃ …
      iso₁₃ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₁₃ v₁₃ …
      eq₁₂ : Eq (CategoryTheory.CategoryStruct.comp v₁₂ (CategoryTheory.Pretriangula …
      eq₁₂' : Eq (CategoryTheory.CategoryStruct.comp w₁₂ ((CategoryTheory.shiftFunct …
      eq₁₃ : Eq (CategoryTheory.CategoryStruct.comp v₁₃ (CategoryTheory.Pretriangula …
      eq₁₃' : Eq (CategoryTheory.CategoryStruct.comp w₁₃ ((CategoryTheory.shiftFunct …
      eq₂₃ : Eq (CategoryTheory.CategoryStruct.comp v₂₃ (CategoryTheory.Pretriangula …
      eq₂₃' : Eq (CategoryTheory.CategoryStruct.comp w₂₃ ((CategoryTheory.shiftFunct …
      rel₁₂ : Eq (CategoryTheory.CategoryStruct.comp v₁₂' H.m₁) (CategoryTheory.Cate …
      rel₁₃ : Eq w₁₂' (CategoryTheory.CategoryStruct.comp H.m₁ w₁₃')
      rel₂₂ : Eq (CategoryTheory.CategoryStruct.comp v₁₃' H.m₃) v₂₃'
      rel₂₃ : Eq (CategoryTheory.CategoryStruct.comp w₁₃' ((CategoryTheory.shiftFunc …
      ⊢ Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Catego …
    -/
  · refine isomorphic_distinguished _ H.mem _ ?_
    refine Triangle.isoMk _ _ (Triangle.π₃.mapIso iso₁₂) (Triangle.π₃.mapIso iso₁₃)
      (Triangle.π₃.mapIso iso₂₃) (by simp) (by simp) ?_
    /-
      case refine_5
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.40535, u_1} C
      inst✝⁴ : CategoryTheory.Preadditive C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      X₁ X₂ X₃ Z₁₂ Z₂₃ Z₁₃ : C
      u₁₂ : Quiver.Hom X₁ X₂
      u₂₃ : Quiver.Hom X₂ X₃
      u₁₃ : Quiver.Hom X₁ X₃
      comm : Eq (CategoryTheory.CategoryStruct.comp u₁₂ u₂₃) u₁₃
      v₁₂ : Quiver.Hom X₂ Z₁₂
      w₁₂ : Quiver.Hom Z₁₂ ((CategoryTheory.shiftFunctor C 1).obj X₁)
      h₁₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      v₂₃ : Quiver.Hom X₃ Z₂₃
      w₂₃ : Quiver.Hom Z₂₃ ((CategoryTheory.shiftFunctor C 1).obj X₂)
      h₂₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      v₁₃ : Quiver.Hom X₃ Z₁₃
      w₁₃ : Quiver.Hom Z₁₃ ((CategoryTheory.shiftFunctor C 1).obj X₁)
      h₁₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      h : CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
      X₁' X₂' X₃' Z₁₂' Z₂₃' Z₁₃' : C
      u₁₂' : Quiver.Hom X₁' X₂'
      u₂₃' : Quiver.Hom X₂' X₃'
      u₁₃' : Quiver.Hom X₁' X₃'
      comm' : Eq (CategoryTheory.CategoryStruct.comp u₁₂' u₂₃') u₁₃'
      e₁ : CategoryTheory.Iso X₁ X₁'
      e₂ : CategoryTheory.Iso X₂ X₂'
      e₃ : CategoryTheory.Iso X₃ X₃'
      comm₁₂ : Eq (CategoryTheory.CategoryStruct.comp u₁₂ e₂.hom) (CategoryTheory.Ca …
      comm₂₃ : Eq (CategoryTheory.CategoryStruct.comp u₂₃ e₃.hom) (CategoryTheory.Ca …
      v₁₂' : Quiver.Hom X₂' Z₁₂'
      w₁₂' : Quiver.Hom Z₁₂' ((CategoryTheory.shiftFunctor C 1).obj X₁')
      h₁₂' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
      v₂₃' : Quiver.Hom X₃' Z₂₃'
      w₂₃' : Quiver.Hom Z₂₃' ((CategoryTheory.shiftFunctor C 1).obj X₂')
      h₂₃' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
      v₁₃' : Quiver.Hom X₃' Z₁₃'
      w₁₃' : Quiver.Hom Z₁₃' ((CategoryTheory.shiftFunctor C 1).obj X₁')
      h₁₃' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
      H : CategoryTheory.Triangulated.Octahedron comm' h₁₂' h₂₃' h₁₃'
      iso₁₂ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₁₂ v₁₂ …
      iso₂₃ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₂₃ v₂₃ …
      iso₁₃ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₁₃ v₁₃ …
      eq₁₂ : Eq (CategoryTheory.CategoryStruct.comp v₁₂ (CategoryTheory.Pretriangula …
      eq₁₂' : Eq (CategoryTheory.CategoryStruct.comp w₁₂ ((CategoryTheory.shiftFunct …
      eq₁₃ : Eq (CategoryTheory.CategoryStruct.comp v₁₃ (CategoryTheory.Pretriangula …
      eq₁₃' : Eq (CategoryTheory.CategoryStruct.comp w₁₃ ((CategoryTheory.shiftFunct …
      eq₂₃ : Eq (CategoryTheory.CategoryStruct.comp v₂₃ (CategoryTheory.Pretriangula …
      eq₂₃' : Eq (CategoryTheory.CategoryStruct.comp w₂₃ ((CategoryTheory.shiftFunct …
      rel₁₂ : Eq (CategoryTheory.CategoryStruct.comp v₁₂' H.m₁) (CategoryTheory.Cate …
      rel₁₃ : Eq w₁₂' (CategoryTheory.CategoryStruct.comp H.m₁ w₁₃')
      rel₂₂ : Eq (CategoryTheory.CategoryStruct.comp v₁₃' H.m₃) v₂₃'
      rel₂₃ : Eq (CategoryTheory.CategoryStruct.comp w₁₃' ((CategoryTheory.shiftFunc …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated.Trian …
    -/
    dsimp
    /-
      case refine_5
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.40535, u_1} C
      inst✝⁴ : CategoryTheory.Preadditive C
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.HasShift C Int
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      X₁ X₂ X₃ Z₁₂ Z₂₃ Z₁₃ : C
      u₁₂ : Quiver.Hom X₁ X₂
      u₂₃ : Quiver.Hom X₂ X₃
      u₁₃ : Quiver.Hom X₁ X₃
      comm : Eq (CategoryTheory.CategoryStruct.comp u₁₂ u₂₃) u₁₃
      v₁₂ : Quiver.Hom X₂ Z₁₂
      w₁₂ : Quiver.Hom Z₁₂ ((CategoryTheory.shiftFunctor C 1).obj X₁)
      h₁₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      v₂₃ : Quiver.Hom X₃ Z₂₃
      w₂₃ : Quiver.Hom Z₂₃ ((CategoryTheory.shiftFunctor C 1).obj X₂)
      h₂₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      v₁₃ : Quiver.Hom X₃ Z₁₃
      w₁₃ : Quiver.Hom Z₁₃ ((CategoryTheory.shiftFunctor C 1).obj X₁)
      h₁₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
      h : CategoryTheory.Triangulated.Octahedron comm h₁₂ h₂₃ h₁₃
      X₁' X₂' X₃' Z₁₂' Z₂₃' Z₁₃' : C
      u₁₂' : Quiver.Hom X₁' X₂'
      u₂₃' : Quiver.Hom X₂' X₃'
      u₁₃' : Quiver.Hom X₁' X₃'
      comm' : Eq (CategoryTheory.CategoryStruct.comp u₁₂' u₂₃') u₁₃'
      e₁ : CategoryTheory.Iso X₁ X₁'
      e₂ : CategoryTheory.Iso X₂ X₂'
      e₃ : CategoryTheory.Iso X₃ X₃'
      comm₁₂ : Eq (CategoryTheory.CategoryStruct.comp u₁₂ e₂.hom) (CategoryTheory.Ca …
      comm₂₃ : Eq (CategoryTheory.CategoryStruct.comp u₂₃ e₃.hom) (CategoryTheory.Ca …
      v₁₂' : Quiver.Hom X₂' Z₁₂'
      w₁₂' : Quiver.Hom Z₁₂' ((CategoryTheory.shiftFunctor C 1).obj X₁')
      h₁₂' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
      v₂₃' : Quiver.Hom X₃' Z₂₃'
      w₂₃' : Quiver.Hom Z₂₃' ((CategoryTheory.shiftFunctor C 1).obj X₂')
      h₂₃' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
      v₁₃' : Quiver.Hom X₃' Z₁₃'
      w₁₃' : Quiver.Hom Z₁₃' ((CategoryTheory.shiftFunctor C 1).obj X₁')
      h₁₃' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (C …
      H : CategoryTheory.Triangulated.Octahedron comm' h₁₂' h₂₃' h₁₃'
      iso₁₂ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₁₂ v₁₂ …
      iso₂₃ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₂₃ v₂₃ …
      iso₁₃ : CategoryTheory.Iso (CategoryTheory.Pretriangulated.Triangle.mk u₁₃ v₁₃ …
      eq₁₂ : Eq (CategoryTheory.CategoryStruct.comp v₁₂ (CategoryTheory.Pretriangula …
      eq₁₂' : Eq (CategoryTheory.CategoryStruct.comp w₁₂ ((CategoryTheory.shiftFunct …
      eq₁₃ : Eq (CategoryTheory.CategoryStruct.comp v₁₃ (CategoryTheory.Pretriangula …
      eq₁₃' : Eq (CategoryTheory.CategoryStruct.comp w₁₃ ((CategoryTheory.shiftFunct …
      eq₂₃ : Eq (CategoryTheory.CategoryStruct.comp v₂₃ (CategoryTheory.Pretriangula …
      eq₂₃' : Eq (CategoryTheory.CategoryStruct.comp w₂₃ ((CategoryTheory.shiftFunct …
      rel₁₂ : Eq (CategoryTheory.CategoryStruct.comp v₁₂' H.m₁) (CategoryTheory.Cate …
      rel₁₃ : Eq w₁₂' (CategoryTheory.CategoryStruct.comp H.m₁ w₁₃')
      rel₂₂ : Eq (CategoryTheory.CategoryStruct.comp v₁₃' H.m₃) v₂₃'
      rel₂₃ : Eq (CategoryTheory.CategoryStruct.comp w₁₃' ((CategoryTheory.shiftFunc …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp w …
    -/
    rw [assoc, ← Functor.map_comp, eq₁₂, Functor.map_comp, reassoc_of% eq₂₃']
    /-
      🎉 no goals
    -/


/-- A triangulated category is a pretriangulated category which satisfies
the octahedron axiom (TR 4), see https://stacks.math.columbia.edu/tag/05QK -/
class IsTriangulated : Prop where
  /-- the octahedron axiom (TR 4) -/
  octahedron_axiom :
    ∀ {X₁ X₂ X₃ Z₁₂ Z₂₃ Z₁₃ : C}
      {u₁₂ : X₁ ⟶ X₂} {u₂₃ : X₂ ⟶ X₃} {u₁₃ : X₁ ⟶ X₃} (comm : u₁₂ ≫ u₂₃ = u₁₃)
      {v₁₂ : X₂ ⟶ Z₁₂} {w₁₂ : Z₁₂ ⟶ X₁⟦(1 : ℤ)⟧} (h₁₂ : Triangle.mk u₁₂ v₁₂ w₁₂ ∈ distTriang C)
      {v₂₃ : X₃ ⟶ Z₂₃} {w₂₃ : Z₂₃ ⟶ X₂⟦(1 : ℤ)⟧} (h₂₃ : Triangle.mk u₂₃ v₂₃ w₂₃ ∈ distTriang C)
      {v₁₃ : X₃ ⟶ Z₁₃} {w₁₃ : Z₁₃ ⟶ X₁⟦(1 : ℤ)⟧} (h₁₃ : Triangle.mk u₁₃ v₁₃ w₁₃ ∈ distTriang C),
      Nonempty (Octahedron comm h₁₂ h₂₃ h₁₃)


/-- A choice of octahedron given by the octahedron axiom. -/
def someOctahedron' [IsTriangulated C] : Octahedron comm h₁₂ h₂₃ h₁₃ :=
  (IsTriangulated.octahedron_axiom comm h₁₂ h₂₃ h₁₃).some


/-- A choice of octahedron given by the octahedron axiom. -/
def someOctahedron [IsTriangulated C]
    {X₁ X₂ X₃ Z₁₂ Z₂₃ Z₁₃ : C}
    {u₁₂ : X₁ ⟶ X₂} {u₂₃ : X₂ ⟶ X₃} {u₁₃ : X₁ ⟶ X₃} (comm : u₁₂ ≫ u₂₃ = u₁₃)
    {v₁₂ : X₂ ⟶ Z₁₂} {w₁₂ : Z₁₂ ⟶ X₁⟦(1 : ℤ)⟧} (h₁₂ : Triangle.mk u₁₂ v₁₂ w₁₂ ∈ distTriang C)
    {v₂₃ : X₃ ⟶ Z₂₃} {w₂₃ : Z₂₃ ⟶ X₂⟦(1 : ℤ)⟧} (h₂₃ : Triangle.mk u₂₃ v₂₃ w₂₃ ∈ distTriang C)
    {v₁₃ : X₃ ⟶ Z₁₃} {w₁₃ : Z₁₃ ⟶ X₁⟦(1 : ℤ)⟧} (h₁₃ : Triangle.mk u₁₃ v₁₃ w₁₃ ∈ distTriang C) :
    Octahedron comm h₁₂ h₂₃ h₁₃ :=
  someOctahedron' _


/-- Constructor for `IsTriangulated C` which shows that it suffices to obtain an octahedron
for a suitable isomorphic diagram instead of the given diagram. -/
lemma IsTriangulated.mk' (h : ∀ ⦃X₁' X₂' X₃' : C⦄ (u₁₂' : X₁' ⟶ X₂') (u₂₃' : X₂' ⟶ X₃'),
    ∃ (X₁ X₂ X₃ Z₁₂ Z₂₃ Z₁₃ : C) (u₁₂ : X₁ ⟶ X₂) (u₂₃ : X₂ ⟶ X₃) (e₁ : X₁' ≅ X₁) (e₂ : X₂' ≅ X₂)
    (e₃ : X₃' ≅ X₃) (_ : u₁₂' ≫ e₂.hom = e₁.hom ≫ u₁₂)
    (_ : u₂₃' ≫ e₃.hom = e₂.hom ≫ u₂₃)
    (v₁₂ : X₂ ⟶ Z₁₂) (w₁₂ : Z₁₂ ⟶ X₁⟦1⟧) (h₁₂ : Triangle.mk u₁₂ v₁₂ w₁₂ ∈ distTriang C)
    (v₂₃ : X₃ ⟶ Z₂₃) (w₂₃ : Z₂₃ ⟶ X₂⟦1⟧) (h₂₃ : Triangle.mk u₂₃ v₂₃ w₂₃ ∈ distTriang C)
    (v₁₃ : X₃ ⟶ Z₁₃) (w₁₃ : Z₁₃ ⟶ X₁⟦1⟧)
      (h₁₃ : Triangle.mk (u₁₂ ≫ u₂₃) v₁₃ w₁₃ ∈ distTriang C),
        Nonempty (Octahedron rfl h₁₂ h₂₃ h₁₃)) :
    IsTriangulated C where
  octahedron_axiom {X₁' X₂' X₃' Z₁₂' Z₂₃' Z₁₃' u₁₂' u₂₃' u₁₃'} comm'
    {v₁₂' w₁₂'} h₁₂' {v₂₃' w₂₃'} h₂₃' {v₁₃' w₁₃'} h₁₃' := by
    obtain ⟨X₁, X₂, X₃, Z₁₂, Z₂₃, Z₁₃, u₁₂, u₂₃, e₁, e₂, e₃, comm₁₂, comm₂₃,
      v₁₂, w₁₂, h₁₂, v₂₃, w₂₃, h₂₃, v₁₃, w₁₃, h₁₃, H⟩ := h u₁₂' u₂₃'
    exact ⟨Octahedron.ofIso u₁₂' u₂₃' u₁₃' comm' h₁₂' h₂₃' h₁₃'
      u₁₂ u₂₃ _ rfl e₁ e₂ e₃ comm₁₂ comm₂₃ v₁₂ w₁₂ h₁₂ v₂₃ w₂₃ h₂₃ v₁₃ w₁₃ h₁₃ H.some⟩


