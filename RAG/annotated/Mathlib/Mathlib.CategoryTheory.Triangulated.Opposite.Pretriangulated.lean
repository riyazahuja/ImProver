/-- A triangle in `Cᵒᵖ` shall be distinguished iff it corresponds to a distinguished
triangle in `C` via the equivalence `triangleOpEquivalence C : (Triangle C)ᵒᵖ ≌ Triangle Cᵒᵖ`. -/
def distinguishedTriangles : Set (Triangle Cᵒᵖ) :=
  fun T => ((triangleOpEquivalence C).inverse.obj T).unop ∈ distTriang C


lemma mem_distinguishedTriangles_iff (T : Triangle Cᵒᵖ) :
    T ∈ distinguishedTriangles C ↔
      ((triangleOpEquivalence C).inverse.obj T).unop ∈ distTriang C := by
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.HasShift C Int
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle (Opposite C)
    ⊢ Iff (Membership.mem (CategoryTheory.Pretriangulated.Opposite.distinguishedTr …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma mem_distinguishedTriangles_iff' (T : Triangle Cᵒᵖ) :
    T ∈ distinguishedTriangles C ↔
      ∃ (T' : Triangle C) (_ : T' ∈ distTriang C),
        Nonempty (T ≅ (triangleOpEquivalence C).functor.obj (Opposite.op T')) := by
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.HasShift C Int
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle (Opposite C)
    ⊢ Iff (Membership.mem (CategoryTheory.Pretriangulated.Opposite.distinguishedTr …
  -/
  rw [mem_distinguishedTriangles_iff]
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.HasShift C Int
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle (Opposite C)
    ⊢ Iff (Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (O …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁴ : CategoryTheory.HasShift C Int
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      T : CategoryTheory.Pretriangulated.Triangle (Opposite C)
      ⊢ Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Opposi …
    -/
  · intro hT
    /-
      case mp
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁴ : CategoryTheory.HasShift C Int
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      T : CategoryTheory.Pretriangulated.Triangle (Opposite C)
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Opp …
      ⊢ Exists fun T' => Exists fun x => Nonempty (CategoryTheory.Iso T ((CategoryTh …
    -/
    exact ⟨_ ,hT, ⟨(triangleOpEquivalence C).counitIso.symm.app T⟩⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁴ : CategoryTheory.HasShift C Int
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      T : CategoryTheory.Pretriangulated.Triangle (Opposite C)
      ⊢ (Exists fun T' => Exists fun x => Nonempty (CategoryTheory.Iso T ((CategoryT …
    -/
  · rintro ⟨T', hT', ⟨e⟩⟩
    /-
      case mpr.intro.intro.intro
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁴ : CategoryTheory.HasShift C Int
      inst✝³ : CategoryTheory.Limits.HasZeroObject C
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      T : CategoryTheory.Pretriangulated.Triangle (Opposite C)
      T' : CategoryTheory.Pretriangulated.Triangle C
      hT' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T'
      e : CategoryTheory.Iso T ((CategoryTheory.Pretriangulated.triangleOpEquivalenc …
      ⊢ Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Opposi …
    -/
    refine isomorphic_distinguished _ hT' _ ?_
    exact Iso.unop ((triangleOpEquivalence C).unitIso.app (Opposite.op T') ≪≫
      (triangleOpEquivalence C).inverse.mapIso e.symm)


lemma isomorphic_distinguished (T₁ : Triangle Cᵒᵖ)
    (hT₁ : T₁ ∈ distinguishedTriangles C) (T₂ : Triangle Cᵒᵖ) (e : T₂ ≅ T₁) :
    T₂ ∈ distinguishedTriangles C := by
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.HasShift C Int
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    T₁ : CategoryTheory.Pretriangulated.Triangle (Opposite C)
    hT₁ : Membership.mem (CategoryTheory.Pretriangulated.Opposite.distinguishedTri …
    T₂ : CategoryTheory.Pretriangulated.Triangle (Opposite C)
    e : CategoryTheory.Iso T₂ T₁
    ⊢ Membership.mem (CategoryTheory.Pretriangulated.Opposite.distinguishedTriangl …
  -/
  simp only [mem_distinguishedTriangles_iff] at hT₁ ⊢
  exact Pretriangulated.isomorphic_distinguished _ hT₁ _
    ((triangleOpEquivalence C).inverse.mapIso e).unop.symm


/-- Up to rotation, the contractible triangle `X ⟶ X ⟶ 0 ⟶ X⟦1⟧` for `X : Cᵒᵖ` corresponds
to the contractible triangle for `X.unop` in `C`. -/
@[simps!]
noncomputable def contractibleTriangleIso (X : Cᵒᵖ) :
    contractibleTriangle X ≅ (triangleOpEquivalence C).functor.obj
      (Opposite.op (contractibleTriangle X.unop).invRotate) :=
  Triangle.isoMk _ _ (Iso.refl _) (Iso.refl _)
    (IsZero.iso (isZero_zero _) (by
      /-
        C : Type u_1
        inst✝⁵ : CategoryTheory.Category.{?u.5318, u_1} C
        inst✝⁴ : CategoryTheory.HasShift C Int
        inst✝³ : CategoryTheory.Limits.HasZeroObject C
        inst✝² : CategoryTheory.Preadditive C
        inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
        inst✝ : CategoryTheory.Pretriangulated C
        X : Opposite C
        ⊢ CategoryTheory.Limits.IsZero ((CategoryTheory.Pretriangulated.triangleOpEqui …
      -/
      dsimp
      /-
        C : Type u_1
        inst✝⁵ : CategoryTheory.Category.{?u.5318, u_1} C
        inst✝⁴ : CategoryTheory.HasShift C Int
        inst✝³ : CategoryTheory.Limits.HasZeroObject C
        inst✝² : CategoryTheory.Preadditive C
        inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
        inst✝ : CategoryTheory.Pretriangulated C
        X : Opposite C
        ⊢ CategoryTheory.Limits.IsZero { unop := (CategoryTheory.shiftFunctor C (-1)). …
      -/
      rw [IsZero.iff_id_eq_zero]
      /-
        C : Type u_1
        inst✝⁵ : CategoryTheory.Category.{?u.5318, u_1} C
        inst✝⁴ : CategoryTheory.HasShift C Int
        inst✝³ : CategoryTheory.Limits.HasZeroObject C
        inst✝² : CategoryTheory.Preadditive C
        inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
        inst✝ : CategoryTheory.Pretriangulated C
        X : Opposite C
        ⊢ Eq (CategoryTheory.CategoryStruct.id { unop := (CategoryTheory.shiftFunctor  …
      -/
      change (𝟙 ((0 : C)⟦(-1 : ℤ)⟧)).op = 0
      /-
        C : Type u_1
        inst✝⁵ : CategoryTheory.Category.{?u.5318, u_1} C
        inst✝⁴ : CategoryTheory.HasShift C Int
        inst✝³ : CategoryTheory.Limits.HasZeroObject C
        inst✝² : CategoryTheory.Preadditive C
        inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
        inst✝ : CategoryTheory.Pretriangulated C
        X : Opposite C
        ⊢ Eq (CategoryTheory.CategoryStruct.id ((CategoryTheory.shiftFunctor C (-1)).o …
      -/
      rw [← Functor.map_id, id_zero, Functor.map_zero, op_zero]))
      /-
        🎉 no goals
      -/
        /-
          C : Type u_1
          inst✝⁵ : CategoryTheory.Category.{?u.5318, u_1} C
          inst✝⁴ : CategoryTheory.HasShift C Int
          inst✝³ : CategoryTheory.Limits.HasZeroObject C
          inst✝² : CategoryTheory.Preadditive C
          inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
          inst✝ : CategoryTheory.Pretriangulated C
          X : Opposite C
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated.contr …
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


lemma contractible_distinguished (X : Cᵒᵖ) :
    contractibleTriangle X ∈ distinguishedTriangles C := by
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.HasShift C Int
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    X : Opposite C
    ⊢ Membership.mem (CategoryTheory.Pretriangulated.Opposite.distinguishedTriangl …
  -/
  rw [mem_distinguishedTriangles_iff']
  exact ⟨_, inv_rot_of_distTriang _ (Pretriangulated.contractible_distinguished X.unop),
    ⟨contractibleTriangleIso X⟩⟩


/-- Isomorphism expressing a compatibility of the equivalence `triangleOpEquivalence C`
with the rotation of triangles. -/
noncomputable def rotateTriangleOpEquivalenceInverseObjRotateUnopIso (T : Triangle Cᵒᵖ) :
    ((triangleOpEquivalence C).inverse.obj T.rotate).unop.rotate ≅
      ((triangleOpEquivalence C).inverse.obj T).unop :=
  Triangle.isoMk _ _ (Iso.refl _) (Iso.refl _)
                                                                       /-
                                                                         C : Type u_1
                                                                         inst✝⁵ : CategoryTheory.Category.{?u.41025, u_1} C
                                                                         inst✝⁴ : CategoryTheory.HasShift C Int
                                                                         inst✝³ : CategoryTheory.Limits.HasZeroObject C
                                                                         inst✝² : CategoryTheory.Preadditive C
                                                                         inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
                                                                         inst✝ : CategoryTheory.Pretriangulated C
                                                                         T : CategoryTheory.Pretriangulated.Triangle (Opposite C)
                                                                         ⊢ Eq (CategoryTheory.CategoryStruct.comp (Opposite.unop ((CategoryTheory.Pretr …
                                                                       -/
      (-((opShiftFunctorEquivalence C 1).unitIso.app T.obj₁).unop) (by simp)
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                               /-
                                 C : Type u_1
                                 inst✝⁵ : CategoryTheory.Category.{?u.41025, u_1} C
                                 inst✝⁴ : CategoryTheory.HasShift C Int
                                 inst✝³ : CategoryTheory.Limits.HasZeroObject C
                                 inst✝² : CategoryTheory.Preadditive C
                                 inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
                                 inst✝ : CategoryTheory.Pretriangulated C
                                 T : CategoryTheory.Pretriangulated.Triangle (Opposite C)
                                 ⊢ Eq (CategoryTheory.CategoryStruct.comp (Opposite.unop ((CategoryTheory.Pretr …
                               -/
                               /-
                                 🎉 no goals
                               -/
        (Quiver.Hom.op_inj (by aesop_cat)) (by aesop_cat)
                                               /-
                                                 🎉 no goals
                                               -/


lemma rotate_distinguished_triangle (T : Triangle Cᵒᵖ) :
    T ∈ distinguishedTriangles C ↔ T.rotate ∈ distinguishedTriangles C := by
  simp only [mem_distinguishedTriangles_iff, Pretriangulated.rotate_distinguished_triangle
    ((triangleOpEquivalence C).inverse.obj (T.rotate)).unop]
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.HasShift C Int
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle (Opposite C)
    ⊢ Iff (Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (O …
  -/
  exact distinguished_iff_of_iso (rotateTriangleOpEquivalenceInverseObjRotateUnopIso T).symm
  /-
    🎉 no goals
  -/


lemma distinguished_cocone_triangle {X Y : Cᵒᵖ} (f : X ⟶ Y) :
    ∃ (Z : Cᵒᵖ) (g : Y ⟶ Z) (h : Z ⟶ X⟦(1 : ℤ)⟧),
      Triangle.mk f g h ∈ distinguishedTriangles C := by
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.HasShift C Int
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    X Y : Opposite C
    f : Quiver.Hom X Y
    ⊢ Exists fun Z => Exists fun g => Exists fun h => Membership.mem (CategoryTheo …
  -/
  obtain ⟨Z, g, h, H⟩ := Pretriangulated.distinguished_cocone_triangle₁ f.unop
  refine ⟨_, g.op, (opShiftFunctorEquivalence C 1).counitIso.inv.app (Opposite.op Z) ≫
    (shiftFunctor Cᵒᵖ (1 : ℤ)).map h.op, ?_⟩
  /-
    case intro.intro.intro
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.HasShift C Int
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    X Y : Opposite C
    f : Quiver.Hom X Y
    Z : C
    g : Quiver.Hom Z (Opposite.unop Y)
    h : Quiver.Hom (Opposite.unop X) ((CategoryTheory.shiftFunctor C 1).obj Z)
    H : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Cate …
    ⊢ Membership.mem (CategoryTheory.Pretriangulated.Opposite.distinguishedTriangl …
  -/
  simp only [mem_distinguishedTriangles_iff]
  /-
    case intro.intro.intro
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.HasShift C Int
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    X Y : Opposite C
    f : Quiver.Hom X Y
    Z : C
    g : Quiver.Hom Z (Opposite.unop Y)
    h : Quiver.Hom (Opposite.unop X) ((CategoryTheory.shiftFunctor C 1).obj Z)
    H : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Cate …
    ⊢ Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Opposi …
  -/
  refine Pretriangulated.isomorphic_distinguished _ H _ ?_
  exact Triangle.isoMk _ _ (Iso.refl _) (Iso.refl _) (Iso.refl _) (by aesop_cat) (by aesop_cat)
    (Quiver.Hom.op_inj (by simp [shift_unop_opShiftFunctorEquivalence_counitIso_inv_app]))


lemma complete_distinguished_triangle_morphism (T₁ T₂ : Triangle Cᵒᵖ)
    (hT₁ : T₁ ∈ distinguishedTriangles C) (hT₂ : T₂ ∈ distinguishedTriangles C)
    (a : T₁.obj₁ ⟶ T₂.obj₁) (b : T₁.obj₂ ⟶ T₂.obj₂) (comm : T₁.mor₁ ≫ b = a ≫ T₂.mor₁) :
    ∃ (c : T₁.obj₃ ⟶ T₂.obj₃), T₁.mor₂ ≫ c = b ≫ T₂.mor₂ ∧
      T₁.mor₃ ≫ a⟦1⟧' = c ≫ T₂.mor₃ := by
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.HasShift C Int
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle (Opposite C)
    hT₁ : Membership.mem (CategoryTheory.Pretriangulated.Opposite.distinguishedTri …
    hT₂ : Membership.mem (CategoryTheory.Pretriangulated.Opposite.distinguishedTri …
    a : Quiver.Hom T₁.obj₁ T₂.obj₁
    b : Quiver.Hom T₁.obj₂ T₂.obj₂
    comm : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₁ b) (CategoryTheory.Categ …
    ⊢ Exists fun c => And (Eq (CategoryTheory.CategoryStruct.comp T₁.mor₂ c) (Cate …
  -/
  rw [mem_distinguishedTriangles_iff] at hT₁ hT₂
  obtain ⟨c, hc₁, hc₂⟩ :=
    Pretriangulated.complete_distinguished_triangle_morphism₁ _ _ hT₂ hT₁
      b.unop a.unop (Quiver.Hom.op_inj comm.symm)
  /-
    case intro.intro
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.HasShift C Int
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle (Opposite C)
    hT₁ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Op …
    hT₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Op …
    a : Quiver.Hom T₁.obj₁ T₂.obj₁
    b : Quiver.Hom T₁.obj₂ T₂.obj₂
    comm : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₁ b) (CategoryTheory.Categ …
    c : Quiver.Hom (Opposite.unop ((CategoryTheory.Pretriangulated.triangleOpEquiv …
    hc₁ : Eq (CategoryTheory.CategoryStruct.comp (Opposite.unop ((CategoryTheory.P …
    hc₂ : Eq (CategoryTheory.CategoryStruct.comp (Opposite.unop ((CategoryTheory.P …
    ⊢ Exists fun c => And (Eq (CategoryTheory.CategoryStruct.comp T₁.mor₂ c) (Cate …
  -/
  dsimp at c hc₁ hc₂
  /-
    case intro.intro
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.HasShift C Int
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle (Opposite C)
    hT₁ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Op …
    hT₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Op …
    a : Quiver.Hom T₁.obj₁ T₂.obj₁
    b : Quiver.Hom T₁.obj₂ T₂.obj₂
    comm : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₁ b) (CategoryTheory.Categ …
    c : Quiver.Hom (Opposite.unop T₂.obj₃) (Opposite.unop T₁.obj₃)
    hc₁ : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₂.unop b.unop) (CategoryThe …
    hc₂ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.co …
    ⊢ Exists fun c => And (Eq (CategoryTheory.CategoryStruct.comp T₁.mor₂ c) (Cate …
  -/
  replace hc₂ := ((opShiftFunctorEquivalence C 1).unitIso.hom.app T₂.obj₁).unop ≫= hc₂
  /-
    case intro.intro
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.HasShift C Int
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle (Opposite C)
    hT₁ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Op …
    hT₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Op …
    a : Quiver.Hom T₁.obj₁ T₂.obj₁
    b : Quiver.Hom T₁.obj₂ T₂.obj₂
    comm : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₁ b) (CategoryTheory.Categ …
    c : Quiver.Hom (Opposite.unop T₂.obj₃) (Opposite.unop T₁.obj₃)
    hc₁ : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₂.unop b.unop) (CategoryThe …
    hc₂ : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Pretriangulated. …
    ⊢ Exists fun c => And (Eq (CategoryTheory.CategoryStruct.comp T₁.mor₂ c) (Cate …
  -/
  dsimp at hc₂
  /-
    case intro.intro
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.HasShift C Int
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle (Opposite C)
    hT₁ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Op …
    hT₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Op …
    a : Quiver.Hom T₁.obj₁ T₂.obj₁
    b : Quiver.Hom T₁.obj₂ T₂.obj₂
    comm : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₁ b) (CategoryTheory.Categ …
    c : Quiver.Hom (Opposite.unop T₂.obj₃) (Opposite.unop T₁.obj₃)
    hc₁ : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₂.unop b.unop) (CategoryThe …
    hc₂ : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Pretriangulated. …
    ⊢ Exists fun c => And (Eq (CategoryTheory.CategoryStruct.comp T₁.mor₂ c) (Cate …
  -/
  simp only [assoc, Iso.unop_hom_inv_id_app_assoc] at hc₂
  /-
    case intro.intro
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.HasShift C Int
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle (Opposite C)
    hT₁ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Op …
    hT₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Op …
    a : Quiver.Hom T₁.obj₁ T₂.obj₁
    b : Quiver.Hom T₁.obj₂ T₂.obj₂
    comm : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₁ b) (CategoryTheory.Categ …
    c : Quiver.Hom (Opposite.unop T₂.obj₃) (Opposite.unop T₁.obj₃)
    hc₁ : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₂.unop b.unop) (CategoryThe …
    hc₂ : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctor C 1 …
    ⊢ Exists fun c => And (Eq (CategoryTheory.CategoryStruct.comp T₁.mor₂ c) (Cate …
  -/
  refine ⟨c.op, Quiver.Hom.unop_inj hc₁.symm, Quiver.Hom.unop_inj ?_⟩
  /-
    case intro.intro
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.HasShift C Int
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle (Opposite C)
    hT₁ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Op …
    hT₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Op …
    a : Quiver.Hom T₁.obj₁ T₂.obj₁
    b : Quiver.Hom T₁.obj₂ T₂.obj₂
    comm : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₁ b) (CategoryTheory.Categ …
    c : Quiver.Hom (Opposite.unop T₂.obj₃) (Opposite.unop T₁.obj₃)
    hc₁ : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₂.unop b.unop) (CategoryThe …
    hc₂ : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctor C 1 …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp T₁.mor₃ ((CategoryTheory.shiftFunctor …
  -/
  apply (shiftFunctor C (1 : ℤ)).map_injective
  rw [unop_comp, unop_comp, Functor.map_comp, Functor.map_comp,
    Quiver.Hom.unop_op, hc₂, ← unop_comp_assoc, ← unop_comp_assoc,
    ← opShiftFunctorEquivalence_unitIso_inv_naturality]
  /-
    case intro.intro.a
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.HasShift C Int
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle (Opposite C)
    hT₁ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Op …
    hT₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Op …
    a : Quiver.Hom T₁.obj₁ T₂.obj₁
    b : Quiver.Hom T₁.obj₂ T₂.obj₂
    comm : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₁ b) (CategoryTheory.Categ …
    c : Quiver.Hom (Opposite.unop T₂.obj₃) (Opposite.unop T₁.obj₃)
    hc₁ : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₂.unop b.unop) (CategoryThe …
    hc₂ : Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctor C 1 …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.shiftFunctor C 1).ma …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The pretriangulated structure on the opposite category of
a pretriangulated category. It is a scoped instance, so that we need to
`open CategoryTheory.Pretriangulated.Opposite` in order to be able
to use it: the reason is that it relies on the definition of the shift
on the opposite category `Cᵒᵖ`, for which it is unclear whether it should
be a global instance or not. -/
scoped instance : Pretriangulated Cᵒᵖ where
  distinguishedTriangles := distinguishedTriangles C
  isomorphic_distinguished := isomorphic_distinguished
  contractible_distinguished := contractible_distinguished
  distinguished_cocone_triangle := distinguished_cocone_triangle
  rotate_distinguished_triangle := rotate_distinguished_triangle
  complete_distinguished_triangle_morphism := complete_distinguished_triangle_morphism


lemma mem_distTriang_op_iff (T : Triangle Cᵒᵖ) :
    (T ∈ distTriang Cᵒᵖ) ↔ ((triangleOpEquivalence C).inverse.obj T).unop ∈ distTriang C := by
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.HasShift C Int
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle (Opposite C)
    ⊢ Iff (Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T) …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma mem_distTriang_op_iff' (T : Triangle Cᵒᵖ) :
    (T ∈ distTriang Cᵒᵖ) ↔ ∃ (T' : Triangle C) (_ : T' ∈ distTriang C),
      Nonempty (T ≅ (triangleOpEquivalence C).functor.obj (Opposite.op T')) :=
  Opposite.mem_distinguishedTriangles_iff' T


lemma op_distinguished (T : Triangle C) (hT : T ∈ distTriang C) :
    ((triangleOpEquivalence C).functor.obj (Opposite.op T)) ∈ distTriang Cᵒᵖ := by
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.HasShift C Int
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    ⊢ Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles ((Categ …
  -/
  rw [mem_distTriang_op_iff']
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    inst✝⁴ : CategoryTheory.HasShift C Int
    inst✝³ : CategoryTheory.Limits.HasZeroObject C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝ : CategoryTheory.Pretriangulated C
    T : CategoryTheory.Pretriangulated.Triangle C
    hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    ⊢ Exists fun T' => Exists fun x => Nonempty (CategoryTheory.Iso ((CategoryTheo …
  -/
  exact ⟨T, hT, ⟨Iso.refl _⟩⟩
  /-
    🎉 no goals
  -/


lemma unop_distinguished (T : Triangle Cᵒᵖ) (hT : T ∈ distTriang Cᵒᵖ) :
    ((triangleOpEquivalence C).inverse.obj T).unop ∈ distTriang C := hT


lemma map_distinguished_op_exact {A : Type*} [Category A] [Abelian A] (F : Cᵒᵖ ⥤ A)
    [F.IsHomological] (T : Triangle C) (hT : T ∈ distTriang C) :
    ((shortComplexOfDistTriangle T hT).op.map F).Exact :=
  F.map_distinguished_exact _ (op_distinguished T hT)


