/-- Given `W` is a class of morphisms in a pretriangulated category `C`, this is the condition
that `W` is compatible with the triangulation on `C`. -/
class IsCompatibleWithTriangulation (W : MorphismProperty C)
    extends W.IsCompatibleWithShift ℤ : Prop where
  compatible_with_triangulation (T₁ T₂ : Triangle C)
    (_ : T₁ ∈ distTriang C) (_ : T₂ ∈ distTriang C)
    (a : T₁.obj₁ ⟶ T₂.obj₁) (b : T₁.obj₂ ⟶ T₂.obj₂) (_ : W a) (_ : W b)
    (_ : T₁.mor₁ ≫ b = a ≫ T₂.mor₁) :
      ∃ (c : T₁.obj₃ ⟶ T₂.obj₃) (_ : W c),
        (T₁.mor₂ ≫ c = b ≫ T₂.mor₂) ∧ (T₁.mor₃ ≫ a⟦1⟧' = c ≫ T₂.mor₃)


/-- Given a functor `C ⥤ D` from a pretriangulated category, this is the set of
triangles in `D` that are in the essential image of distinguished triangles of `C`. -/
def essImageDistTriang : Set (Triangle D) :=
  fun T => ∃ (T' : Triangle C) (_ : T ≅ L.mapTriangle.obj T'), T' ∈ distTriang C


lemma essImageDistTriang_mem_of_iso {T₁ T₂ : Triangle D} (e : T₂ ≅ T₁)
    (h : T₁ ∈ L.essImageDistTriang) : T₂ ∈ L.essImageDistTriang := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁸ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁷ : CategoryTheory.Category.{u_3, u_2} D
    L : CategoryTheory.Functor C D
    inst✝⁶ : CategoryTheory.HasShift C Int
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
    inst✝³ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝² : CategoryTheory.Pretriangulated C
    inst✝¹ : CategoryTheory.HasShift D Int
    inst✝ : L.CommShift Int
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle D
    e : CategoryTheory.Iso T₂ T₁
    h : Membership.mem L.essImageDistTriang T₁
    ⊢ Membership.mem L.essImageDistTriang T₂
  -/
  obtain ⟨T', e', hT'⟩ := h
  /-
    case intro.intro
    C : Type u_1
    D : Type u_2
    inst✝⁸ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁷ : CategoryTheory.Category.{u_3, u_2} D
    L : CategoryTheory.Functor C D
    inst✝⁶ : CategoryTheory.HasShift C Int
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
    inst✝³ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝² : CategoryTheory.Pretriangulated C
    inst✝¹ : CategoryTheory.HasShift D Int
    inst✝ : L.CommShift Int
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle D
    e : CategoryTheory.Iso T₂ T₁
    T' : CategoryTheory.Pretriangulated.Triangle C
    e' : CategoryTheory.Iso T₁ (L.mapTriangle.obj T')
    hT' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T'
    ⊢ Membership.mem L.essImageDistTriang T₂
  -/
  exact ⟨T', e ≪≫ e', hT'⟩
  /-
    🎉 no goals
  -/


lemma contractible_mem_essImageDistTriang [EssSurj L] [HasZeroObject D]
    [HasZeroMorphisms D] [L.PreservesZeroMorphisms] (X : D) :
    contractibleTriangle X ∈ L.essImageDistTriang := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹² : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹¹ : CategoryTheory.Category.{u_4, u_2} D
    L : CategoryTheory.Functor C D
    inst✝¹⁰ : CategoryTheory.HasShift C Int
    inst✝⁹ : CategoryTheory.Preadditive C
    inst✝⁸ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁷ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝⁶ : CategoryTheory.Pretriangulated C
    inst✝⁵ : CategoryTheory.HasShift D Int
    inst✝⁴ : L.CommShift Int
    inst✝³ : L.EssSurj
    inst✝² : CategoryTheory.Limits.HasZeroObject D
    inst✝¹ : CategoryTheory.Limits.HasZeroMorphisms D
    inst✝ : L.PreservesZeroMorphisms
    X : D
    ⊢ Membership.mem L.essImageDistTriang (CategoryTheory.Pretriangulated.contract …
  -/
  refine ⟨contractibleTriangle (L.objPreimage X), ?_, contractible_distinguished _⟩
  exact ((contractibleTriangleFunctor D).mapIso (L.objObjPreimageIso X)).symm ≪≫
    Triangle.isoMk _ _ (Iso.refl _) (Iso.refl _) L.mapZeroObject.symm (by simp) (by simp) (by simp)


lemma rotate_essImageDistTriang [Preadditive D] [L.Additive]
    [∀ (n : ℤ), (shiftFunctor D n).Additive] (T : Triangle D) :
  T ∈ L.essImageDistTriang ↔ T.rotate ∈ L.essImageDistTriang := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹¹ : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹⁰ : CategoryTheory.Category.{u_3, u_2} D
    L : CategoryTheory.Functor C D
    inst✝⁹ : CategoryTheory.HasShift C Int
    inst✝⁸ : CategoryTheory.Preadditive C
    inst✝⁷ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝⁵ : CategoryTheory.Pretriangulated C
    inst✝⁴ : CategoryTheory.HasShift D Int
    inst✝³ : L.CommShift Int
    inst✝² : CategoryTheory.Preadditive D
    inst✝¹ : L.Additive
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
    T : CategoryTheory.Pretriangulated.Triangle D
    ⊢ Iff (Membership.mem L.essImageDistTriang T) (Membership.mem L.essImageDistTr …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      D : Type u_2
      inst✝¹¹ : CategoryTheory.Category.{u_4, u_1} C
      inst✝¹⁰ : CategoryTheory.Category.{u_3, u_2} D
      L : CategoryTheory.Functor C D
      inst✝⁹ : CategoryTheory.HasShift C Int
      inst✝⁸ : CategoryTheory.Preadditive C
      inst✝⁷ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝⁵ : CategoryTheory.Pretriangulated C
      inst✝⁴ : CategoryTheory.HasShift D Int
      inst✝³ : L.CommShift Int
      inst✝² : CategoryTheory.Preadditive D
      inst✝¹ : L.Additive
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
      T : CategoryTheory.Pretriangulated.Triangle D
      ⊢ Membership.mem L.essImageDistTriang T → Membership.mem L.essImageDistTriang  …
    -/
  · rintro ⟨T', e', hT'⟩
    exact ⟨T'.rotate, (rotate D).mapIso e' ≪≫ L.mapTriangleRotateIso.app T',
      rot_of_distTriang T' hT'⟩
    /-
      case mpr
      C : Type u_1
      D : Type u_2
      inst✝¹¹ : CategoryTheory.Category.{u_4, u_1} C
      inst✝¹⁰ : CategoryTheory.Category.{u_3, u_2} D
      L : CategoryTheory.Functor C D
      inst✝⁹ : CategoryTheory.HasShift C Int
      inst✝⁸ : CategoryTheory.Preadditive C
      inst✝⁷ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝⁵ : CategoryTheory.Pretriangulated C
      inst✝⁴ : CategoryTheory.HasShift D Int
      inst✝³ : L.CommShift Int
      inst✝² : CategoryTheory.Preadditive D
      inst✝¹ : L.Additive
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
      T : CategoryTheory.Pretriangulated.Triangle D
      ⊢ Membership.mem L.essImageDistTriang T.rotate → Membership.mem L.essImageDist …
    -/
  · rintro ⟨T', e', hT'⟩
    exact ⟨T'.invRotate, (triangleRotation D).unitIso.app T ≪≫ (invRotate D).mapIso e' ≪≫
      L.mapTriangleInvRotateIso.app T', inv_rot_of_distTriang T' hT'⟩


lemma complete_distinguished_essImageDistTriang_morphism
    (H : ∀ (T₁' T₂' : Triangle C) (_ : T₁' ∈ distTriang C) (_ : T₂' ∈ distTriang C)
      (a : L.obj (T₁'.obj₁) ⟶ L.obj (T₂'.obj₁)) (b : L.obj (T₁'.obj₂) ⟶ L.obj (T₂'.obj₂))
      (_ : L.map T₁'.mor₁ ≫ b = a ≫ L.map T₂'.mor₁),
      ∃ (φ : L.mapTriangle.obj T₁' ⟶ L.mapTriangle.obj T₂'), φ.hom₁ = a ∧ φ.hom₂ = b)
    (T₁ T₂ : Triangle D)
    (hT₁ : T₁ ∈ Functor.essImageDistTriang L) (hT₂ : T₂ ∈ L.essImageDistTriang)
    (a : T₁.obj₁ ⟶ T₂.obj₁) (b : T₁.obj₂ ⟶ T₂.obj₂) (fac : T₁.mor₁ ≫ b = a ≫ T₂.mor₁) :
    ∃ c, T₁.mor₂ ≫ c = b ≫ T₂.mor₂ ∧ T₁.mor₃ ≫ a⟦1⟧' = c ≫ T₂.mor₃ := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁷ : CategoryTheory.Category.{u_4, u_2} D
    L : CategoryTheory.Functor C D
    inst✝⁶ : CategoryTheory.HasShift C Int
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
    inst✝³ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝² : CategoryTheory.Pretriangulated C
    inst✝¹ : CategoryTheory.HasShift D Int
    inst✝ : L.CommShift Int
    H : ∀ (T₁' T₂' : CategoryTheory.Pretriangulated.Triangle C), Membership.mem Ca …
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle D
    hT₁ : Membership.mem L.essImageDistTriang T₁
    hT₂ : Membership.mem L.essImageDistTriang T₂
    a : Quiver.Hom T₁.obj₁ T₂.obj₁
    b : Quiver.Hom T₁.obj₂ T₂.obj₂
    fac : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₁ b) (CategoryTheory.Catego …
    ⊢ Exists fun c => And (Eq (CategoryTheory.CategoryStruct.comp T₁.mor₂ c) (Cate …
  -/
  obtain ⟨T₁', e₁, hT₁'⟩ := hT₁
  /-
    case intro.intro
    C : Type u_1
    D : Type u_2
    inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁷ : CategoryTheory.Category.{u_4, u_2} D
    L : CategoryTheory.Functor C D
    inst✝⁶ : CategoryTheory.HasShift C Int
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
    inst✝³ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝² : CategoryTheory.Pretriangulated C
    inst✝¹ : CategoryTheory.HasShift D Int
    inst✝ : L.CommShift Int
    H : ∀ (T₁' T₂' : CategoryTheory.Pretriangulated.Triangle C), Membership.mem Ca …
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle D
    hT₂ : Membership.mem L.essImageDistTriang T₂
    a : Quiver.Hom T₁.obj₁ T₂.obj₁
    b : Quiver.Hom T₁.obj₂ T₂.obj₂
    fac : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₁ b) (CategoryTheory.Catego …
    T₁' : CategoryTheory.Pretriangulated.Triangle C
    e₁ : CategoryTheory.Iso T₁ (L.mapTriangle.obj T₁')
    hT₁' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁'
    ⊢ Exists fun c => And (Eq (CategoryTheory.CategoryStruct.comp T₁.mor₂ c) (Cate …
  -/
  obtain ⟨T₂', e₂, hT₂'⟩ := hT₂
  /-
    case intro.intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁷ : CategoryTheory.Category.{u_4, u_2} D
    L : CategoryTheory.Functor C D
    inst✝⁶ : CategoryTheory.HasShift C Int
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
    inst✝³ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝² : CategoryTheory.Pretriangulated C
    inst✝¹ : CategoryTheory.HasShift D Int
    inst✝ : L.CommShift Int
    H : ∀ (T₁' T₂' : CategoryTheory.Pretriangulated.Triangle C), Membership.mem Ca …
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle D
    a : Quiver.Hom T₁.obj₁ T₂.obj₁
    b : Quiver.Hom T₁.obj₂ T₂.obj₂
    fac : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₁ b) (CategoryTheory.Catego …
    T₁' : CategoryTheory.Pretriangulated.Triangle C
    e₁ : CategoryTheory.Iso T₁ (L.mapTriangle.obj T₁')
    hT₁' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁'
    T₂' : CategoryTheory.Pretriangulated.Triangle C
    e₂ : CategoryTheory.Iso T₂ (L.mapTriangle.obj T₂')
    hT₂' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₂'
    ⊢ Exists fun c => And (Eq (CategoryTheory.CategoryStruct.comp T₁.mor₂ c) (Cate …
  -/
  have comm₁ := e₁.inv.comm₁
  /-
    case intro.intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁷ : CategoryTheory.Category.{u_4, u_2} D
    L : CategoryTheory.Functor C D
    inst✝⁶ : CategoryTheory.HasShift C Int
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
    inst✝³ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝² : CategoryTheory.Pretriangulated C
    inst✝¹ : CategoryTheory.HasShift D Int
    inst✝ : L.CommShift Int
    H : ∀ (T₁' T₂' : CategoryTheory.Pretriangulated.Triangle C), Membership.mem Ca …
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle D
    a : Quiver.Hom T₁.obj₁ T₂.obj₁
    b : Quiver.Hom T₁.obj₂ T₂.obj₂
    fac : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₁ b) (CategoryTheory.Catego …
    T₁' : CategoryTheory.Pretriangulated.Triangle C
    e₁ : CategoryTheory.Iso T₁ (L.mapTriangle.obj T₁')
    hT₁' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁'
    T₂' : CategoryTheory.Pretriangulated.Triangle C
    e₂ : CategoryTheory.Iso T₂ (L.mapTriangle.obj T₂')
    hT₂' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₂'
    comm₁ : Eq (CategoryTheory.CategoryStruct.comp (L.mapTriangle.obj T₁').mor₁ e₁ …
    ⊢ Exists fun c => And (Eq (CategoryTheory.CategoryStruct.comp T₁.mor₂ c) (Cate …
  -/
  have comm₁' := e₂.hom.comm₁
  /-
    case intro.intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁷ : CategoryTheory.Category.{u_4, u_2} D
    L : CategoryTheory.Functor C D
    inst✝⁶ : CategoryTheory.HasShift C Int
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
    inst✝³ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝² : CategoryTheory.Pretriangulated C
    inst✝¹ : CategoryTheory.HasShift D Int
    inst✝ : L.CommShift Int
    H : ∀ (T₁' T₂' : CategoryTheory.Pretriangulated.Triangle C), Membership.mem Ca …
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle D
    a : Quiver.Hom T₁.obj₁ T₂.obj₁
    b : Quiver.Hom T₁.obj₂ T₂.obj₂
    fac : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₁ b) (CategoryTheory.Catego …
    T₁' : CategoryTheory.Pretriangulated.Triangle C
    e₁ : CategoryTheory.Iso T₁ (L.mapTriangle.obj T₁')
    hT₁' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁'
    T₂' : CategoryTheory.Pretriangulated.Triangle C
    e₂ : CategoryTheory.Iso T₂ (L.mapTriangle.obj T₂')
    hT₂' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₂'
    comm₁ : Eq (CategoryTheory.CategoryStruct.comp (L.mapTriangle.obj T₁').mor₁ e₁ …
    comm₁' : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₁ e₂.hom.hom₂) (Category …
    ⊢ Exists fun c => And (Eq (CategoryTheory.CategoryStruct.comp T₁.mor₂ c) (Cate …
  -/
  have comm₂ := e₁.hom.comm₂
  /-
    case intro.intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁷ : CategoryTheory.Category.{u_4, u_2} D
    L : CategoryTheory.Functor C D
    inst✝⁶ : CategoryTheory.HasShift C Int
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
    inst✝³ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝² : CategoryTheory.Pretriangulated C
    inst✝¹ : CategoryTheory.HasShift D Int
    inst✝ : L.CommShift Int
    H : ∀ (T₁' T₂' : CategoryTheory.Pretriangulated.Triangle C), Membership.mem Ca …
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle D
    a : Quiver.Hom T₁.obj₁ T₂.obj₁
    b : Quiver.Hom T₁.obj₂ T₂.obj₂
    fac : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₁ b) (CategoryTheory.Catego …
    T₁' : CategoryTheory.Pretriangulated.Triangle C
    e₁ : CategoryTheory.Iso T₁ (L.mapTriangle.obj T₁')
    hT₁' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁'
    T₂' : CategoryTheory.Pretriangulated.Triangle C
    e₂ : CategoryTheory.Iso T₂ (L.mapTriangle.obj T₂')
    hT₂' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₂'
    comm₁ : Eq (CategoryTheory.CategoryStruct.comp (L.mapTriangle.obj T₁').mor₁ e₁ …
    comm₁' : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₁ e₂.hom.hom₂) (Category …
    comm₂ : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₂ e₁.hom.hom₃) (CategoryT …
    ⊢ Exists fun c => And (Eq (CategoryTheory.CategoryStruct.comp T₁.mor₂ c) (Cate …
  -/
  have comm₂' := e₂.hom.comm₂
  /-
    case intro.intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁷ : CategoryTheory.Category.{u_4, u_2} D
    L : CategoryTheory.Functor C D
    inst✝⁶ : CategoryTheory.HasShift C Int
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
    inst✝³ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝² : CategoryTheory.Pretriangulated C
    inst✝¹ : CategoryTheory.HasShift D Int
    inst✝ : L.CommShift Int
    H : ∀ (T₁' T₂' : CategoryTheory.Pretriangulated.Triangle C), Membership.mem Ca …
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle D
    a : Quiver.Hom T₁.obj₁ T₂.obj₁
    b : Quiver.Hom T₁.obj₂ T₂.obj₂
    fac : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₁ b) (CategoryTheory.Catego …
    T₁' : CategoryTheory.Pretriangulated.Triangle C
    e₁ : CategoryTheory.Iso T₁ (L.mapTriangle.obj T₁')
    hT₁' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁'
    T₂' : CategoryTheory.Pretriangulated.Triangle C
    e₂ : CategoryTheory.Iso T₂ (L.mapTriangle.obj T₂')
    hT₂' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₂'
    comm₁ : Eq (CategoryTheory.CategoryStruct.comp (L.mapTriangle.obj T₁').mor₁ e₁ …
    comm₁' : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₁ e₂.hom.hom₂) (Category …
    comm₂ : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₂ e₁.hom.hom₃) (CategoryT …
    comm₂' : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₂ e₂.hom.hom₃) (Category …
    ⊢ Exists fun c => And (Eq (CategoryTheory.CategoryStruct.comp T₁.mor₂ c) (Cate …
  -/
  have comm₃ := e₁.inv.comm₃
  /-
    case intro.intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁷ : CategoryTheory.Category.{u_4, u_2} D
    L : CategoryTheory.Functor C D
    inst✝⁶ : CategoryTheory.HasShift C Int
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
    inst✝³ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝² : CategoryTheory.Pretriangulated C
    inst✝¹ : CategoryTheory.HasShift D Int
    inst✝ : L.CommShift Int
    H : ∀ (T₁' T₂' : CategoryTheory.Pretriangulated.Triangle C), Membership.mem Ca …
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle D
    a : Quiver.Hom T₁.obj₁ T₂.obj₁
    b : Quiver.Hom T₁.obj₂ T₂.obj₂
    fac : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₁ b) (CategoryTheory.Catego …
    T₁' : CategoryTheory.Pretriangulated.Triangle C
    e₁ : CategoryTheory.Iso T₁ (L.mapTriangle.obj T₁')
    hT₁' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁'
    T₂' : CategoryTheory.Pretriangulated.Triangle C
    e₂ : CategoryTheory.Iso T₂ (L.mapTriangle.obj T₂')
    hT₂' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₂'
    comm₁ : Eq (CategoryTheory.CategoryStruct.comp (L.mapTriangle.obj T₁').mor₁ e₁ …
    comm₁' : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₁ e₂.hom.hom₂) (Category …
    comm₂ : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₂ e₁.hom.hom₃) (CategoryT …
    comm₂' : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₂ e₂.hom.hom₃) (Category …
    comm₃ : Eq (CategoryTheory.CategoryStruct.comp (L.mapTriangle.obj T₁').mor₃ (( …
    ⊢ Exists fun c => And (Eq (CategoryTheory.CategoryStruct.comp T₁.mor₂ c) (Cate …
  -/
  have comm₃' := e₂.hom.comm₃
  /-
    case intro.intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁷ : CategoryTheory.Category.{u_4, u_2} D
    L : CategoryTheory.Functor C D
    inst✝⁶ : CategoryTheory.HasShift C Int
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
    inst✝³ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝² : CategoryTheory.Pretriangulated C
    inst✝¹ : CategoryTheory.HasShift D Int
    inst✝ : L.CommShift Int
    H : ∀ (T₁' T₂' : CategoryTheory.Pretriangulated.Triangle C), Membership.mem Ca …
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle D
    a : Quiver.Hom T₁.obj₁ T₂.obj₁
    b : Quiver.Hom T₁.obj₂ T₂.obj₂
    fac : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₁ b) (CategoryTheory.Catego …
    T₁' : CategoryTheory.Pretriangulated.Triangle C
    e₁ : CategoryTheory.Iso T₁ (L.mapTriangle.obj T₁')
    hT₁' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁'
    T₂' : CategoryTheory.Pretriangulated.Triangle C
    e₂ : CategoryTheory.Iso T₂ (L.mapTriangle.obj T₂')
    hT₂' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₂'
    comm₁ : Eq (CategoryTheory.CategoryStruct.comp (L.mapTriangle.obj T₁').mor₁ e₁ …
    comm₁' : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₁ e₂.hom.hom₂) (Category …
    comm₂ : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₂ e₁.hom.hom₃) (CategoryT …
    comm₂' : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₂ e₂.hom.hom₃) (Category …
    comm₃ : Eq (CategoryTheory.CategoryStruct.comp (L.mapTriangle.obj T₁').mor₃ (( …
    comm₃' : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₃ ((CategoryTheory.shift …
    ⊢ Exists fun c => And (Eq (CategoryTheory.CategoryStruct.comp T₁.mor₂ c) (Cate …
  -/
  dsimp at comm₁ comm₁' comm₂ comm₂' comm₃ comm₃'
  /-
    case intro.intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁷ : CategoryTheory.Category.{u_4, u_2} D
    L : CategoryTheory.Functor C D
    inst✝⁶ : CategoryTheory.HasShift C Int
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
    inst✝³ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝² : CategoryTheory.Pretriangulated C
    inst✝¹ : CategoryTheory.HasShift D Int
    inst✝ : L.CommShift Int
    H : ∀ (T₁' T₂' : CategoryTheory.Pretriangulated.Triangle C), Membership.mem Ca …
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle D
    a : Quiver.Hom T₁.obj₁ T₂.obj₁
    b : Quiver.Hom T₁.obj₂ T₂.obj₂
    fac : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₁ b) (CategoryTheory.Catego …
    T₁' : CategoryTheory.Pretriangulated.Triangle C
    e₁ : CategoryTheory.Iso T₁ (L.mapTriangle.obj T₁')
    hT₁' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁'
    T₂' : CategoryTheory.Pretriangulated.Triangle C
    e₂ : CategoryTheory.Iso T₂ (L.mapTriangle.obj T₂')
    hT₂' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₂'
    comm₁ : Eq (CategoryTheory.CategoryStruct.comp (L.map T₁'.mor₁) e₁.inv.hom₂) ( …
    comm₁' : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₁ e₂.hom.hom₂) (Category …
    comm₂ : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₂ e₁.hom.hom₃) (CategoryT …
    comm₂' : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₂ e₂.hom.hom₃) (Category …
    comm₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct. …
    comm₃' : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₃ ((CategoryTheory.shift …
    ⊢ Exists fun c => And (Eq (CategoryTheory.CategoryStruct.comp T₁.mor₂ c) (Cate …
  -/
  simp only [assoc] at comm₃
  obtain ⟨φ, hφ₁, hφ₂⟩ := H T₁' T₂' hT₁' hT₂' (e₁.inv.hom₁ ≫ a ≫ e₂.hom.hom₁)
    (e₁.inv.hom₂ ≫ b ≫ e₂.hom.hom₂)
    (by simp only [assoc, ← comm₁', ← reassoc_of% fac, ← reassoc_of% comm₁])
  /-
    case intro.intro.intro.intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁷ : CategoryTheory.Category.{u_4, u_2} D
    L : CategoryTheory.Functor C D
    inst✝⁶ : CategoryTheory.HasShift C Int
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
    inst✝³ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝² : CategoryTheory.Pretriangulated C
    inst✝¹ : CategoryTheory.HasShift D Int
    inst✝ : L.CommShift Int
    H : ∀ (T₁' T₂' : CategoryTheory.Pretriangulated.Triangle C), Membership.mem Ca …
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle D
    a : Quiver.Hom T₁.obj₁ T₂.obj₁
    b : Quiver.Hom T₁.obj₂ T₂.obj₂
    fac : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₁ b) (CategoryTheory.Catego …
    T₁' : CategoryTheory.Pretriangulated.Triangle C
    e₁ : CategoryTheory.Iso T₁ (L.mapTriangle.obj T₁')
    hT₁' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁'
    T₂' : CategoryTheory.Pretriangulated.Triangle C
    e₂ : CategoryTheory.Iso T₂ (L.mapTriangle.obj T₂')
    hT₂' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₂'
    comm₁ : Eq (CategoryTheory.CategoryStruct.comp (L.map T₁'.mor₁) e₁.inv.hom₂) ( …
    comm₁' : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₁ e₂.hom.hom₂) (Category …
    comm₂ : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₂ e₁.hom.hom₃) (CategoryT …
    comm₂' : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₂ e₂.hom.hom₃) (Category …
    comm₃' : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₃ ((CategoryTheory.shift …
    comm₃ : Eq (CategoryTheory.CategoryStruct.comp (L.map T₁'.mor₃) (CategoryTheor …
    φ : Quiver.Hom (L.mapTriangle.obj T₁') (L.mapTriangle.obj T₂')
    hφ₁ : Eq φ.hom₁ (CategoryTheory.CategoryStruct.comp e₁.inv.hom₁ (CategoryTheor …
    hφ₂ : Eq φ.hom₂ (CategoryTheory.CategoryStruct.comp e₁.inv.hom₂ (CategoryTheor …
    ⊢ Exists fun c => And (Eq (CategoryTheory.CategoryStruct.comp T₁.mor₂ c) (Cate …
  -/
  have h₂ := φ.comm₂
  /-
    case intro.intro.intro.intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁷ : CategoryTheory.Category.{u_4, u_2} D
    L : CategoryTheory.Functor C D
    inst✝⁶ : CategoryTheory.HasShift C Int
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
    inst✝³ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝² : CategoryTheory.Pretriangulated C
    inst✝¹ : CategoryTheory.HasShift D Int
    inst✝ : L.CommShift Int
    H : ∀ (T₁' T₂' : CategoryTheory.Pretriangulated.Triangle C), Membership.mem Ca …
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle D
    a : Quiver.Hom T₁.obj₁ T₂.obj₁
    b : Quiver.Hom T₁.obj₂ T₂.obj₂
    fac : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₁ b) (CategoryTheory.Catego …
    T₁' : CategoryTheory.Pretriangulated.Triangle C
    e₁ : CategoryTheory.Iso T₁ (L.mapTriangle.obj T₁')
    hT₁' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁'
    T₂' : CategoryTheory.Pretriangulated.Triangle C
    e₂ : CategoryTheory.Iso T₂ (L.mapTriangle.obj T₂')
    hT₂' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₂'
    comm₁ : Eq (CategoryTheory.CategoryStruct.comp (L.map T₁'.mor₁) e₁.inv.hom₂) ( …
    comm₁' : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₁ e₂.hom.hom₂) (Category …
    comm₂ : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₂ e₁.hom.hom₃) (CategoryT …
    comm₂' : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₂ e₂.hom.hom₃) (Category …
    comm₃' : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₃ ((CategoryTheory.shift …
    comm₃ : Eq (CategoryTheory.CategoryStruct.comp (L.map T₁'.mor₃) (CategoryTheor …
    φ : Quiver.Hom (L.mapTriangle.obj T₁') (L.mapTriangle.obj T₂')
    hφ₁ : Eq φ.hom₁ (CategoryTheory.CategoryStruct.comp e₁.inv.hom₁ (CategoryTheor …
    hφ₂ : Eq φ.hom₂ (CategoryTheory.CategoryStruct.comp e₁.inv.hom₂ (CategoryTheor …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp (L.mapTriangle.obj T₁').mor₂ φ.hom …
    ⊢ Exists fun c => And (Eq (CategoryTheory.CategoryStruct.comp T₁.mor₂ c) (Cate …
  -/
  have h₃ := φ.comm₃
  /-
    case intro.intro.intro.intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁷ : CategoryTheory.Category.{u_4, u_2} D
    L : CategoryTheory.Functor C D
    inst✝⁶ : CategoryTheory.HasShift C Int
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
    inst✝³ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝² : CategoryTheory.Pretriangulated C
    inst✝¹ : CategoryTheory.HasShift D Int
    inst✝ : L.CommShift Int
    H : ∀ (T₁' T₂' : CategoryTheory.Pretriangulated.Triangle C), Membership.mem Ca …
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle D
    a : Quiver.Hom T₁.obj₁ T₂.obj₁
    b : Quiver.Hom T₁.obj₂ T₂.obj₂
    fac : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₁ b) (CategoryTheory.Catego …
    T₁' : CategoryTheory.Pretriangulated.Triangle C
    e₁ : CategoryTheory.Iso T₁ (L.mapTriangle.obj T₁')
    hT₁' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁'
    T₂' : CategoryTheory.Pretriangulated.Triangle C
    e₂ : CategoryTheory.Iso T₂ (L.mapTriangle.obj T₂')
    hT₂' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₂'
    comm₁ : Eq (CategoryTheory.CategoryStruct.comp (L.map T₁'.mor₁) e₁.inv.hom₂) ( …
    comm₁' : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₁ e₂.hom.hom₂) (Category …
    comm₂ : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₂ e₁.hom.hom₃) (CategoryT …
    comm₂' : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₂ e₂.hom.hom₃) (Category …
    comm₃' : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₃ ((CategoryTheory.shift …
    comm₃ : Eq (CategoryTheory.CategoryStruct.comp (L.map T₁'.mor₃) (CategoryTheor …
    φ : Quiver.Hom (L.mapTriangle.obj T₁') (L.mapTriangle.obj T₂')
    hφ₁ : Eq φ.hom₁ (CategoryTheory.CategoryStruct.comp e₁.inv.hom₁ (CategoryTheor …
    hφ₂ : Eq φ.hom₂ (CategoryTheory.CategoryStruct.comp e₁.inv.hom₂ (CategoryTheor …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp (L.mapTriangle.obj T₁').mor₂ φ.hom …
    h₃ : Eq (CategoryTheory.CategoryStruct.comp (L.mapTriangle.obj T₁').mor₃ ((Cat …
    ⊢ Exists fun c => And (Eq (CategoryTheory.CategoryStruct.comp T₁.mor₂ c) (Cate …
  -/
  dsimp at h₂ h₃
  /-
    case intro.intro.intro.intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁷ : CategoryTheory.Category.{u_4, u_2} D
    L : CategoryTheory.Functor C D
    inst✝⁶ : CategoryTheory.HasShift C Int
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
    inst✝³ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝² : CategoryTheory.Pretriangulated C
    inst✝¹ : CategoryTheory.HasShift D Int
    inst✝ : L.CommShift Int
    H : ∀ (T₁' T₂' : CategoryTheory.Pretriangulated.Triangle C), Membership.mem Ca …
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle D
    a : Quiver.Hom T₁.obj₁ T₂.obj₁
    b : Quiver.Hom T₁.obj₂ T₂.obj₂
    fac : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₁ b) (CategoryTheory.Catego …
    T₁' : CategoryTheory.Pretriangulated.Triangle C
    e₁ : CategoryTheory.Iso T₁ (L.mapTriangle.obj T₁')
    hT₁' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁'
    T₂' : CategoryTheory.Pretriangulated.Triangle C
    e₂ : CategoryTheory.Iso T₂ (L.mapTriangle.obj T₂')
    hT₂' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₂'
    comm₁ : Eq (CategoryTheory.CategoryStruct.comp (L.map T₁'.mor₁) e₁.inv.hom₂) ( …
    comm₁' : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₁ e₂.hom.hom₂) (Category …
    comm₂ : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₂ e₁.hom.hom₃) (CategoryT …
    comm₂' : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₂ e₂.hom.hom₃) (Category …
    comm₃' : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₃ ((CategoryTheory.shift …
    comm₃ : Eq (CategoryTheory.CategoryStruct.comp (L.map T₁'.mor₃) (CategoryTheor …
    φ : Quiver.Hom (L.mapTriangle.obj T₁') (L.mapTriangle.obj T₂')
    hφ₁ : Eq φ.hom₁ (CategoryTheory.CategoryStruct.comp e₁.inv.hom₁ (CategoryTheor …
    hφ₂ : Eq φ.hom₂ (CategoryTheory.CategoryStruct.comp e₁.inv.hom₂ (CategoryTheor …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp (L.map T₁'.mor₂) φ.hom₃) (Category …
    h₃ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.com …
    ⊢ Exists fun c => And (Eq (CategoryTheory.CategoryStruct.comp T₁.mor₂ c) (Cate …
  -/
  simp only [assoc] at h₃
  /-
    case intro.intro.intro.intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝⁸ : CategoryTheory.Category.{u_3, u_1} C
    inst✝⁷ : CategoryTheory.Category.{u_4, u_2} D
    L : CategoryTheory.Functor C D
    inst✝⁶ : CategoryTheory.HasShift C Int
    inst✝⁵ : CategoryTheory.Preadditive C
    inst✝⁴ : CategoryTheory.Limits.HasZeroObject C
    inst✝³ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝² : CategoryTheory.Pretriangulated C
    inst✝¹ : CategoryTheory.HasShift D Int
    inst✝ : L.CommShift Int
    H : ∀ (T₁' T₂' : CategoryTheory.Pretriangulated.Triangle C), Membership.mem Ca …
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle D
    a : Quiver.Hom T₁.obj₁ T₂.obj₁
    b : Quiver.Hom T₁.obj₂ T₂.obj₂
    fac : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₁ b) (CategoryTheory.Catego …
    T₁' : CategoryTheory.Pretriangulated.Triangle C
    e₁ : CategoryTheory.Iso T₁ (L.mapTriangle.obj T₁')
    hT₁' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁'
    T₂' : CategoryTheory.Pretriangulated.Triangle C
    e₂ : CategoryTheory.Iso T₂ (L.mapTriangle.obj T₂')
    hT₂' : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₂'
    comm₁ : Eq (CategoryTheory.CategoryStruct.comp (L.map T₁'.mor₁) e₁.inv.hom₂) ( …
    comm₁' : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₁ e₂.hom.hom₂) (Category …
    comm₂ : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₂ e₁.hom.hom₃) (CategoryT …
    comm₂' : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₂ e₂.hom.hom₃) (Category …
    comm₃' : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₃ ((CategoryTheory.shift …
    comm₃ : Eq (CategoryTheory.CategoryStruct.comp (L.map T₁'.mor₃) (CategoryTheor …
    φ : Quiver.Hom (L.mapTriangle.obj T₁') (L.mapTriangle.obj T₂')
    hφ₁ : Eq φ.hom₁ (CategoryTheory.CategoryStruct.comp e₁.inv.hom₁ (CategoryTheor …
    hφ₂ : Eq φ.hom₂ (CategoryTheory.CategoryStruct.comp e₁.inv.hom₂ (CategoryTheor …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp (L.map T₁'.mor₂) φ.hom₃) (Category …
    h₃ : Eq (CategoryTheory.CategoryStruct.comp (L.map T₁'.mor₃) (CategoryTheory.C …
    ⊢ Exists fun c => And (Eq (CategoryTheory.CategoryStruct.comp T₁.mor₂ c) (Cate …
  -/
  refine ⟨e₁.hom.hom₃ ≫ φ.hom₃ ≫ e₂.inv.hom₃, ?_, ?_⟩
  · rw [reassoc_of% comm₂, reassoc_of% h₂, hφ₂, assoc, assoc,
      Iso.hom_inv_id_triangle_hom₂_assoc, ← reassoc_of% comm₂',
      Iso.hom_inv_id_triangle_hom₃, comp_id]
  · rw [assoc, assoc, ← cancel_epi e₁.inv.hom₃, ← reassoc_of% comm₃,
      Iso.inv_hom_id_triangle_hom₃_assoc, ← cancel_mono (e₂.hom.hom₁⟦(1 : ℤ)⟧'),
      assoc, assoc, assoc, assoc, assoc, ← Functor.map_comp, ← Functor.map_comp, ← hφ₁,
      h₃, comm₃', Iso.inv_hom_id_triangle_hom₃_assoc]


include W in
lemma distinguished_cocone_triangle {X Y : D} (f : X ⟶ Y) :
    ∃ (Z : D) (g : Y ⟶ Z) (h : Z ⟶ X⟦(1 : ℤ)⟧),
      Triangle.mk f g h ∈ L.essImageDistTriang := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹⁰ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁹ : CategoryTheory.Category.{u_3, u_2} D
    L : CategoryTheory.Functor C D
    inst✝⁸ : CategoryTheory.HasShift C Int
    inst✝⁷ : CategoryTheory.Preadditive C
    inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁵ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝⁴ : CategoryTheory.Pretriangulated C
    inst✝³ : CategoryTheory.HasShift D Int
    inst✝² : L.CommShift Int
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : D
    f : Quiver.Hom X Y
    ⊢ Exists fun Z => Exists fun g => Exists fun h => Membership.mem L.essImageDis …
  -/
  have := essSurj_mapArrow L W
  obtain ⟨φ, ⟨e⟩⟩ : ∃ (φ : Arrow C), Nonempty (L.mapArrow.obj φ ≅ Arrow.mk f) :=
    ⟨_, ⟨Functor.objObjPreimageIso _ _⟩⟩
  /-
    case intro.intro
    C : Type u_1
    D : Type u_2
    inst✝¹⁰ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁹ : CategoryTheory.Category.{u_3, u_2} D
    L : CategoryTheory.Functor C D
    inst✝⁸ : CategoryTheory.HasShift C Int
    inst✝⁷ : CategoryTheory.Preadditive C
    inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁵ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝⁴ : CategoryTheory.Pretriangulated C
    inst✝³ : CategoryTheory.HasShift D Int
    inst✝² : L.CommShift Int
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : D
    f : Quiver.Hom X Y
    this : L.mapArrow.EssSurj
    φ : CategoryTheory.Arrow C
    e : CategoryTheory.Iso (L.mapArrow.obj φ) (CategoryTheory.Arrow.mk f)
    ⊢ Exists fun Z => Exists fun g => Exists fun h => Membership.mem L.essImageDis …
  -/
  obtain ⟨Z, g, h, H⟩ := Pretriangulated.distinguished_cocone_triangle φ.hom
  refine ⟨L.obj Z, e.inv.right ≫ L.map g,
    L.map h ≫ (L.commShiftIso (1 : ℤ)).hom.app _ ≫ e.hom.left⟦(1 : ℤ)⟧', _, ?_, H⟩
  refine Triangle.isoMk _ _ (Arrow.leftFunc.mapIso e.symm) (Arrow.rightFunc.mapIso e.symm)
    (Iso.refl _) e.inv.w.symm (by simp) ?_
  /-
    case intro.intro.intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝¹⁰ : CategoryTheory.Category.{u_4, u_1} C
    inst✝⁹ : CategoryTheory.Category.{u_3, u_2} D
    L : CategoryTheory.Functor C D
    inst✝⁸ : CategoryTheory.HasShift C Int
    inst✝⁷ : CategoryTheory.Preadditive C
    inst✝⁶ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁵ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝⁴ : CategoryTheory.Pretriangulated C
    inst✝³ : CategoryTheory.HasShift D Int
    inst✝² : L.CommShift Int
    W : CategoryTheory.MorphismProperty C
    inst✝¹ : L.IsLocalization W
    inst✝ : W.HasLeftCalculusOfFractions
    X Y : D
    f : Quiver.Hom X Y
    this : L.mapArrow.EssSurj
    φ : CategoryTheory.Arrow C
    e : CategoryTheory.Iso (L.mapArrow.obj φ) (CategoryTheory.Arrow.mk f)
    Z : C
    g : Quiver.Hom ((CategoryTheory.Functor.id C).obj φ.right) Z
    h : Quiver.Hom Z ((CategoryTheory.shiftFunctor C 1).obj ((CategoryTheory.Funct …
    H : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Cate …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated.Trian …
  -/
  dsimp
  simp only [assoc, id_comp, ← Functor.map_comp, ← Arrow.comp_left, e.hom_inv_id, Arrow.id_left,
    Functor.mapArrow_obj_left, Functor.map_id, comp_id]


include W in
lemma complete_distinguished_triangle_morphism (T₁ T₂ : Triangle D)
    (hT₁ : T₁ ∈ L.essImageDistTriang) (hT₂ : T₂ ∈ L.essImageDistTriang)
    (a : T₁.obj₁ ⟶ T₂.obj₁) (b : T₁.obj₂ ⟶ T₂.obj₂) (fac : T₁.mor₁ ≫ b = a ≫ T₂.mor₁) :
    ∃ c, T₁.mor₂ ≫ c = b ≫ T₂.mor₂ ∧ T₁.mor₃ ≫ a⟦1⟧' = c ≫ T₂.mor₃ := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹¹ : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹⁰ : CategoryTheory.Category.{u_3, u_2} D
    L : CategoryTheory.Functor C D
    inst✝⁹ : CategoryTheory.HasShift C Int
    inst✝⁸ : CategoryTheory.Preadditive C
    inst✝⁷ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝⁵ : CategoryTheory.Pretriangulated C
    inst✝⁴ : CategoryTheory.HasShift D Int
    inst✝³ : L.CommShift Int
    W : CategoryTheory.MorphismProperty C
    inst✝² : L.IsLocalization W
    inst✝¹ : W.HasLeftCalculusOfFractions
    inst✝ : W.IsCompatibleWithTriangulation
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle D
    hT₁ : Membership.mem L.essImageDistTriang T₁
    hT₂ : Membership.mem L.essImageDistTriang T₂
    a : Quiver.Hom T₁.obj₁ T₂.obj₁
    b : Quiver.Hom T₁.obj₂ T₂.obj₂
    fac : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₁ b) (CategoryTheory.Catego …
    ⊢ Exists fun c => And (Eq (CategoryTheory.CategoryStruct.comp T₁.mor₂ c) (Cate …
  -/
  refine L.complete_distinguished_essImageDistTriang_morphism ?_ T₁ T₂ hT₁ hT₂ a b fac
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹¹ : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹⁰ : CategoryTheory.Category.{u_3, u_2} D
    L : CategoryTheory.Functor C D
    inst✝⁹ : CategoryTheory.HasShift C Int
    inst✝⁸ : CategoryTheory.Preadditive C
    inst✝⁷ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝⁵ : CategoryTheory.Pretriangulated C
    inst✝⁴ : CategoryTheory.HasShift D Int
    inst✝³ : L.CommShift Int
    W : CategoryTheory.MorphismProperty C
    inst✝² : L.IsLocalization W
    inst✝¹ : W.HasLeftCalculusOfFractions
    inst✝ : W.IsCompatibleWithTriangulation
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle D
    hT₁ : Membership.mem L.essImageDistTriang T₁
    hT₂ : Membership.mem L.essImageDistTriang T₂
    a : Quiver.Hom T₁.obj₁ T₂.obj₁
    b : Quiver.Hom T₁.obj₂ T₂.obj₂
    fac : Eq (CategoryTheory.CategoryStruct.comp T₁.mor₁ b) (CategoryTheory.Catego …
    ⊢ ∀ (T₁' T₂' : CategoryTheory.Pretriangulated.Triangle C), Membership.mem Cate …
  -/
  clear a b fac hT₁ hT₂ T₁ T₂
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹¹ : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹⁰ : CategoryTheory.Category.{u_3, u_2} D
    L : CategoryTheory.Functor C D
    inst✝⁹ : CategoryTheory.HasShift C Int
    inst✝⁸ : CategoryTheory.Preadditive C
    inst✝⁷ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝⁵ : CategoryTheory.Pretriangulated C
    inst✝⁴ : CategoryTheory.HasShift D Int
    inst✝³ : L.CommShift Int
    W : CategoryTheory.MorphismProperty C
    inst✝² : L.IsLocalization W
    inst✝¹ : W.HasLeftCalculusOfFractions
    inst✝ : W.IsCompatibleWithTriangulation
    ⊢ ∀ (T₁' T₂' : CategoryTheory.Pretriangulated.Triangle C), Membership.mem Cate …
  -/
  intro T₁ T₂ hT₁ hT₂ a b fac
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹¹ : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹⁰ : CategoryTheory.Category.{u_3, u_2} D
    L : CategoryTheory.Functor C D
    inst✝⁹ : CategoryTheory.HasShift C Int
    inst✝⁸ : CategoryTheory.Preadditive C
    inst✝⁷ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝⁵ : CategoryTheory.Pretriangulated C
    inst✝⁴ : CategoryTheory.HasShift D Int
    inst✝³ : L.CommShift Int
    W : CategoryTheory.MorphismProperty C
    inst✝² : L.IsLocalization W
    inst✝¹ : W.HasLeftCalculusOfFractions
    inst✝ : W.IsCompatibleWithTriangulation
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle C
    hT₁ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁
    hT₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₂
    a : Quiver.Hom (L.obj T₁.obj₁) (L.obj T₂.obj₁)
    b : Quiver.Hom (L.obj T₁.obj₂) (L.obj T₂.obj₂)
    fac : Eq (CategoryTheory.CategoryStruct.comp (L.map T₁.mor₁) b) (CategoryTheor …
    ⊢ Exists fun φ => And (Eq φ.hom₁ a) (Eq φ.hom₂ b)
  -/
  obtain ⟨α, hα⟩ := exists_leftFraction L W a
  /-
    case intro
    C : Type u_1
    D : Type u_2
    inst✝¹¹ : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹⁰ : CategoryTheory.Category.{u_3, u_2} D
    L : CategoryTheory.Functor C D
    inst✝⁹ : CategoryTheory.HasShift C Int
    inst✝⁸ : CategoryTheory.Preadditive C
    inst✝⁷ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝⁵ : CategoryTheory.Pretriangulated C
    inst✝⁴ : CategoryTheory.HasShift D Int
    inst✝³ : L.CommShift Int
    W : CategoryTheory.MorphismProperty C
    inst✝² : L.IsLocalization W
    inst✝¹ : W.HasLeftCalculusOfFractions
    inst✝ : W.IsCompatibleWithTriangulation
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle C
    hT₁ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁
    hT₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₂
    a : Quiver.Hom (L.obj T₁.obj₁) (L.obj T₂.obj₁)
    b : Quiver.Hom (L.obj T₁.obj₂) (L.obj T₂.obj₂)
    fac : Eq (CategoryTheory.CategoryStruct.comp (L.map T₁.mor₁) b) (CategoryTheor …
    α : W.LeftFraction T₁.obj₁ T₂.obj₁
    hα : Eq a (α.map L ⋯)
    ⊢ Exists fun φ => And (Eq φ.hom₁ a) (Eq φ.hom₂ b)
  -/
  obtain ⟨β, hβ⟩ := (MorphismProperty.RightFraction.mk α.s α.hs T₂.mor₁).exists_leftFraction
  /-
    case intro.intro
    C : Type u_1
    D : Type u_2
    inst✝¹¹ : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹⁰ : CategoryTheory.Category.{u_3, u_2} D
    L : CategoryTheory.Functor C D
    inst✝⁹ : CategoryTheory.HasShift C Int
    inst✝⁸ : CategoryTheory.Preadditive C
    inst✝⁷ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝⁵ : CategoryTheory.Pretriangulated C
    inst✝⁴ : CategoryTheory.HasShift D Int
    inst✝³ : L.CommShift Int
    W : CategoryTheory.MorphismProperty C
    inst✝² : L.IsLocalization W
    inst✝¹ : W.HasLeftCalculusOfFractions
    inst✝ : W.IsCompatibleWithTriangulation
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle C
    hT₁ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁
    hT₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₂
    a : Quiver.Hom (L.obj T₁.obj₁) (L.obj T₂.obj₁)
    b : Quiver.Hom (L.obj T₁.obj₂) (L.obj T₂.obj₂)
    fac : Eq (CategoryTheory.CategoryStruct.comp (L.map T₁.mor₁) b) (CategoryTheor …
    α : W.LeftFraction T₁.obj₁ T₂.obj₁
    hα : Eq a (α.map L ⋯)
    β : W.LeftFraction α.Y' T₂.obj₂
    hβ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismProperty.R …
    ⊢ Exists fun φ => And (Eq φ.hom₁ a) (Eq φ.hom₂ b)
  -/
  obtain ⟨γ, hγ⟩ := exists_leftFraction L W (b ≫ L.map β.s)
  /-
    case intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝¹¹ : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹⁰ : CategoryTheory.Category.{u_3, u_2} D
    L : CategoryTheory.Functor C D
    inst✝⁹ : CategoryTheory.HasShift C Int
    inst✝⁸ : CategoryTheory.Preadditive C
    inst✝⁷ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝⁵ : CategoryTheory.Pretriangulated C
    inst✝⁴ : CategoryTheory.HasShift D Int
    inst✝³ : L.CommShift Int
    W : CategoryTheory.MorphismProperty C
    inst✝² : L.IsLocalization W
    inst✝¹ : W.HasLeftCalculusOfFractions
    inst✝ : W.IsCompatibleWithTriangulation
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle C
    hT₁ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁
    hT₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₂
    a : Quiver.Hom (L.obj T₁.obj₁) (L.obj T₂.obj₁)
    b : Quiver.Hom (L.obj T₁.obj₂) (L.obj T₂.obj₂)
    fac : Eq (CategoryTheory.CategoryStruct.comp (L.map T₁.mor₁) b) (CategoryTheor …
    α : W.LeftFraction T₁.obj₁ T₂.obj₁
    hα : Eq a (α.map L ⋯)
    β : W.LeftFraction α.Y' T₂.obj₂
    hβ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismProperty.R …
    γ : W.LeftFraction T₁.obj₂ β.Y'
    hγ : Eq (CategoryTheory.CategoryStruct.comp b (L.map β.s)) (γ.map L ⋯)
    ⊢ Exists fun φ => And (Eq φ.hom₁ a) (Eq φ.hom₂ b)
  -/
  have := inverts L W β.s β.hs
  /-
    case intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝¹¹ : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹⁰ : CategoryTheory.Category.{u_3, u_2} D
    L : CategoryTheory.Functor C D
    inst✝⁹ : CategoryTheory.HasShift C Int
    inst✝⁸ : CategoryTheory.Preadditive C
    inst✝⁷ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝⁵ : CategoryTheory.Pretriangulated C
    inst✝⁴ : CategoryTheory.HasShift D Int
    inst✝³ : L.CommShift Int
    W : CategoryTheory.MorphismProperty C
    inst✝² : L.IsLocalization W
    inst✝¹ : W.HasLeftCalculusOfFractions
    inst✝ : W.IsCompatibleWithTriangulation
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle C
    hT₁ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁
    hT₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₂
    a : Quiver.Hom (L.obj T₁.obj₁) (L.obj T₂.obj₁)
    b : Quiver.Hom (L.obj T₁.obj₂) (L.obj T₂.obj₂)
    fac : Eq (CategoryTheory.CategoryStruct.comp (L.map T₁.mor₁) b) (CategoryTheor …
    α : W.LeftFraction T₁.obj₁ T₂.obj₁
    hα : Eq a (α.map L ⋯)
    β : W.LeftFraction α.Y' T₂.obj₂
    hβ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismProperty.R …
    γ : W.LeftFraction T₁.obj₂ β.Y'
    hγ : Eq (CategoryTheory.CategoryStruct.comp b (L.map β.s)) (γ.map L ⋯)
    this : CategoryTheory.IsIso (L.map β.s)
    ⊢ Exists fun φ => And (Eq φ.hom₁ a) (Eq φ.hom₂ b)
  -/
  have := inverts L W γ.s γ.hs
  /-
    case intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝¹¹ : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹⁰ : CategoryTheory.Category.{u_3, u_2} D
    L : CategoryTheory.Functor C D
    inst✝⁹ : CategoryTheory.HasShift C Int
    inst✝⁸ : CategoryTheory.Preadditive C
    inst✝⁷ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝⁵ : CategoryTheory.Pretriangulated C
    inst✝⁴ : CategoryTheory.HasShift D Int
    inst✝³ : L.CommShift Int
    W : CategoryTheory.MorphismProperty C
    inst✝² : L.IsLocalization W
    inst✝¹ : W.HasLeftCalculusOfFractions
    inst✝ : W.IsCompatibleWithTriangulation
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle C
    hT₁ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁
    hT₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₂
    a : Quiver.Hom (L.obj T₁.obj₁) (L.obj T₂.obj₁)
    b : Quiver.Hom (L.obj T₁.obj₂) (L.obj T₂.obj₂)
    fac : Eq (CategoryTheory.CategoryStruct.comp (L.map T₁.mor₁) b) (CategoryTheor …
    α : W.LeftFraction T₁.obj₁ T₂.obj₁
    hα : Eq a (α.map L ⋯)
    β : W.LeftFraction α.Y' T₂.obj₂
    hβ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.MorphismProperty.R …
    γ : W.LeftFraction T₁.obj₂ β.Y'
    hγ : Eq (CategoryTheory.CategoryStruct.comp b (L.map β.s)) (γ.map L ⋯)
    this✝ : CategoryTheory.IsIso (L.map β.s)
    this : CategoryTheory.IsIso (L.map γ.s)
    ⊢ Exists fun φ => And (Eq φ.hom₁ a) (Eq φ.hom₂ b)
  -/
  dsimp at hβ
  obtain ⟨Z₂, σ, hσ, fac⟩ := (MorphismProperty.map_eq_iff_postcomp L W
    (α.f ≫ β.f ≫ γ.s) (T₁.mor₁ ≫ γ.f)).1 (by
      rw [← cancel_mono (L.map β.s), assoc, assoc, hγ, ← cancel_mono (L.map γ.s),
        assoc, assoc, assoc, hα, MorphismProperty.LeftFraction.map_comp_map_s,
        ← Functor.map_comp] at fac
      rw [fac, ← Functor.map_comp_assoc, hβ, Functor.map_comp, Functor.map_comp,
        Functor.map_comp, assoc, MorphismProperty.LeftFraction.map_comp_map_s_assoc])
  /-
    case intro.intro.intro.intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝¹¹ : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹⁰ : CategoryTheory.Category.{u_3, u_2} D
    L : CategoryTheory.Functor C D
    inst✝⁹ : CategoryTheory.HasShift C Int
    inst✝⁸ : CategoryTheory.Preadditive C
    inst✝⁷ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝⁵ : CategoryTheory.Pretriangulated C
    inst✝⁴ : CategoryTheory.HasShift D Int
    inst✝³ : L.CommShift Int
    W : CategoryTheory.MorphismProperty C
    inst✝² : L.IsLocalization W
    inst✝¹ : W.HasLeftCalculusOfFractions
    inst✝ : W.IsCompatibleWithTriangulation
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle C
    hT₁ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁
    hT₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₂
    a : Quiver.Hom (L.obj T₁.obj₁) (L.obj T₂.obj₁)
    b : Quiver.Hom (L.obj T₁.obj₂) (L.obj T₂.obj₂)
    fac✝ : Eq (CategoryTheory.CategoryStruct.comp (L.map T₁.mor₁) b) (CategoryTheo …
    α : W.LeftFraction T₁.obj₁ T₂.obj₁
    hα : Eq a (α.map L ⋯)
    β : W.LeftFraction α.Y' T₂.obj₂
    hβ : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₁ β.s) (CategoryTheory.Categ …
    γ : W.LeftFraction T₁.obj₂ β.Y'
    hγ : Eq (CategoryTheory.CategoryStruct.comp b (L.map β.s)) (γ.map L ⋯)
    this✝ : CategoryTheory.IsIso (L.map β.s)
    this : CategoryTheory.IsIso (L.map γ.s)
    Z₂ : C
    σ : Quiver.Hom γ.Y' Z₂
    hσ : W σ
    fac : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.co …
    ⊢ Exists fun φ => And (Eq φ.hom₁ a) (Eq φ.hom₂ b)
  -/
  simp only [assoc] at fac
  /-
    case intro.intro.intro.intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝¹¹ : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹⁰ : CategoryTheory.Category.{u_3, u_2} D
    L : CategoryTheory.Functor C D
    inst✝⁹ : CategoryTheory.HasShift C Int
    inst✝⁸ : CategoryTheory.Preadditive C
    inst✝⁷ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝⁵ : CategoryTheory.Pretriangulated C
    inst✝⁴ : CategoryTheory.HasShift D Int
    inst✝³ : L.CommShift Int
    W : CategoryTheory.MorphismProperty C
    inst✝² : L.IsLocalization W
    inst✝¹ : W.HasLeftCalculusOfFractions
    inst✝ : W.IsCompatibleWithTriangulation
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle C
    hT₁ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁
    hT₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₂
    a : Quiver.Hom (L.obj T₁.obj₁) (L.obj T₂.obj₁)
    b : Quiver.Hom (L.obj T₁.obj₂) (L.obj T₂.obj₂)
    fac✝ : Eq (CategoryTheory.CategoryStruct.comp (L.map T₁.mor₁) b) (CategoryTheo …
    α : W.LeftFraction T₁.obj₁ T₂.obj₁
    hα : Eq a (α.map L ⋯)
    β : W.LeftFraction α.Y' T₂.obj₂
    hβ : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₁ β.s) (CategoryTheory.Categ …
    γ : W.LeftFraction T₁.obj₂ β.Y'
    hγ : Eq (CategoryTheory.CategoryStruct.comp b (L.map β.s)) (γ.map L ⋯)
    this✝ : CategoryTheory.IsIso (L.map β.s)
    this : CategoryTheory.IsIso (L.map γ.s)
    Z₂ : C
    σ : Quiver.Hom γ.Y' Z₂
    hσ : W σ
    fac : Eq (CategoryTheory.CategoryStruct.comp α.f (CategoryTheory.CategoryStruc …
    ⊢ Exists fun φ => And (Eq φ.hom₁ a) (Eq φ.hom₂ b)
  -/
  obtain ⟨Y₃, g, h, hT₃⟩ := Pretriangulated.distinguished_cocone_triangle (β.f ≫ γ.s ≫ σ)
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝¹¹ : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹⁰ : CategoryTheory.Category.{u_3, u_2} D
    L : CategoryTheory.Functor C D
    inst✝⁹ : CategoryTheory.HasShift C Int
    inst✝⁸ : CategoryTheory.Preadditive C
    inst✝⁷ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝⁵ : CategoryTheory.Pretriangulated C
    inst✝⁴ : CategoryTheory.HasShift D Int
    inst✝³ : L.CommShift Int
    W : CategoryTheory.MorphismProperty C
    inst✝² : L.IsLocalization W
    inst✝¹ : W.HasLeftCalculusOfFractions
    inst✝ : W.IsCompatibleWithTriangulation
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle C
    hT₁ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁
    hT₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₂
    a : Quiver.Hom (L.obj T₁.obj₁) (L.obj T₂.obj₁)
    b : Quiver.Hom (L.obj T₁.obj₂) (L.obj T₂.obj₂)
    fac✝ : Eq (CategoryTheory.CategoryStruct.comp (L.map T₁.mor₁) b) (CategoryTheo …
    α : W.LeftFraction T₁.obj₁ T₂.obj₁
    hα : Eq a (α.map L ⋯)
    β : W.LeftFraction α.Y' T₂.obj₂
    hβ : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₁ β.s) (CategoryTheory.Categ …
    γ : W.LeftFraction T₁.obj₂ β.Y'
    hγ : Eq (CategoryTheory.CategoryStruct.comp b (L.map β.s)) (γ.map L ⋯)
    this✝ : CategoryTheory.IsIso (L.map β.s)
    this : CategoryTheory.IsIso (L.map γ.s)
    Z₂ : C
    σ : Quiver.Hom γ.Y' Z₂
    hσ : W σ
    fac : Eq (CategoryTheory.CategoryStruct.comp α.f (CategoryTheory.CategoryStruc …
    Y₃ : C
    g : Quiver.Hom Z₂ Y₃
    h : Quiver.Hom Y₃ ((CategoryTheory.shiftFunctor C 1).obj α.Y')
    hT₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    ⊢ Exists fun φ => And (Eq φ.hom₁ a) (Eq φ.hom₂ b)
  -/
  let T₃ := Triangle.mk (β.f ≫ γ.s ≫ σ) g h
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝¹¹ : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹⁰ : CategoryTheory.Category.{u_3, u_2} D
    L : CategoryTheory.Functor C D
    inst✝⁹ : CategoryTheory.HasShift C Int
    inst✝⁸ : CategoryTheory.Preadditive C
    inst✝⁷ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝⁵ : CategoryTheory.Pretriangulated C
    inst✝⁴ : CategoryTheory.HasShift D Int
    inst✝³ : L.CommShift Int
    W : CategoryTheory.MorphismProperty C
    inst✝² : L.IsLocalization W
    inst✝¹ : W.HasLeftCalculusOfFractions
    inst✝ : W.IsCompatibleWithTriangulation
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle C
    hT₁ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁
    hT₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₂
    a : Quiver.Hom (L.obj T₁.obj₁) (L.obj T₂.obj₁)
    b : Quiver.Hom (L.obj T₁.obj₂) (L.obj T₂.obj₂)
    fac✝ : Eq (CategoryTheory.CategoryStruct.comp (L.map T₁.mor₁) b) (CategoryTheo …
    α : W.LeftFraction T₁.obj₁ T₂.obj₁
    hα : Eq a (α.map L ⋯)
    β : W.LeftFraction α.Y' T₂.obj₂
    hβ : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₁ β.s) (CategoryTheory.Categ …
    γ : W.LeftFraction T₁.obj₂ β.Y'
    hγ : Eq (CategoryTheory.CategoryStruct.comp b (L.map β.s)) (γ.map L ⋯)
    this✝ : CategoryTheory.IsIso (L.map β.s)
    this : CategoryTheory.IsIso (L.map γ.s)
    Z₂ : C
    σ : Quiver.Hom γ.Y' Z₂
    hσ : W σ
    fac : Eq (CategoryTheory.CategoryStruct.comp α.f (CategoryTheory.CategoryStruc …
    Y₃ : C
    g : Quiver.Hom Z₂ Y₃
    h : Quiver.Hom Y₃ ((CategoryTheory.shiftFunctor C 1).obj α.Y')
    hT₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles (Ca …
    T₃ : CategoryTheory.Pretriangulated.Triangle C := CategoryTheory.Pretriangulat …
    ⊢ Exists fun φ => And (Eq φ.hom₁ a) (Eq φ.hom₂ b)
  -/
  change T₃ ∈ distTriang C at hT₃
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    C : Type u_1
    D : Type u_2
    inst✝¹¹ : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹⁰ : CategoryTheory.Category.{u_3, u_2} D
    L : CategoryTheory.Functor C D
    inst✝⁹ : CategoryTheory.HasShift C Int
    inst✝⁸ : CategoryTheory.Preadditive C
    inst✝⁷ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝⁵ : CategoryTheory.Pretriangulated C
    inst✝⁴ : CategoryTheory.HasShift D Int
    inst✝³ : L.CommShift Int
    W : CategoryTheory.MorphismProperty C
    inst✝² : L.IsLocalization W
    inst✝¹ : W.HasLeftCalculusOfFractions
    inst✝ : W.IsCompatibleWithTriangulation
    T₁ T₂ : CategoryTheory.Pretriangulated.Triangle C
    hT₁ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁
    hT₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₂
    a : Quiver.Hom (L.obj T₁.obj₁) (L.obj T₂.obj₁)
    b : Quiver.Hom (L.obj T₁.obj₂) (L.obj T₂.obj₂)
    fac✝ : Eq (CategoryTheory.CategoryStruct.comp (L.map T₁.mor₁) b) (CategoryTheo …
    α : W.LeftFraction T₁.obj₁ T₂.obj₁
    hα : Eq a (α.map L ⋯)
    β : W.LeftFraction α.Y' T₂.obj₂
    hβ : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₁ β.s) (CategoryTheory.Categ …
    γ : W.LeftFraction T₁.obj₂ β.Y'
    hγ : Eq (CategoryTheory.CategoryStruct.comp b (L.map β.s)) (γ.map L ⋯)
    this✝ : CategoryTheory.IsIso (L.map β.s)
    this : CategoryTheory.IsIso (L.map γ.s)
    Z₂ : C
    σ : Quiver.Hom γ.Y' Z₂
    hσ : W σ
    fac : Eq (CategoryTheory.CategoryStruct.comp α.f (CategoryTheory.CategoryStruc …
    Y₃ : C
    g : Quiver.Hom Z₂ Y₃
    h : Quiver.Hom Y₃ ((CategoryTheory.shiftFunctor C 1).obj α.Y')
    T₃ : CategoryTheory.Pretriangulated.Triangle C := CategoryTheory.Pretriangulat …
    hT₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₃
    ⊢ Exists fun φ => And (Eq φ.hom₁ a) (Eq φ.hom₂ b)
  -/
  have hβγσ : W (β.s ≫ γ.s ≫ σ) := W.comp_mem _ _ β.hs (W.comp_mem _ _ γ.hs hσ)
  obtain ⟨ψ₃, hψ₃, hψ₁, hψ₂⟩ := MorphismProperty.compatible_with_triangulation
    T₂ T₃ hT₂ hT₃ α.s (β.s ≫ γ.s ≫ σ) α.hs hβγσ (by dsimp [T₃]; rw [reassoc_of% hβ])
  let ψ : T₂ ⟶ T₃ := Triangle.homMk _ _ α.s (β.s ≫ γ.s ≫ σ) ψ₃
    (by dsimp [T₃]; rw [reassoc_of% hβ]) hψ₁ hψ₂
  have : IsIso (L.mapTriangle.map ψ) := Triangle.isIso_of_isIsos _
    (inverts L W α.s α.hs) (inverts L W _ hβγσ) (inverts L W ψ₃ hψ₃)
  refine ⟨L.mapTriangle.map (completeDistinguishedTriangleMorphism T₁ T₃ hT₁ hT₃ α.f
      (γ.f ≫ σ) fac.symm) ≫ inv (L.mapTriangle.map ψ), ?_, ?_⟩
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.r …
      C : Type u_1
      D : Type u_2
      inst✝¹¹ : CategoryTheory.Category.{u_4, u_1} C
      inst✝¹⁰ : CategoryTheory.Category.{u_3, u_2} D
      L : CategoryTheory.Functor C D
      inst✝⁹ : CategoryTheory.HasShift C Int
      inst✝⁸ : CategoryTheory.Preadditive C
      inst✝⁷ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝⁵ : CategoryTheory.Pretriangulated C
      inst✝⁴ : CategoryTheory.HasShift D Int
      inst✝³ : L.CommShift Int
      W : CategoryTheory.MorphismProperty C
      inst✝² : L.IsLocalization W
      inst✝¹ : W.HasLeftCalculusOfFractions
      inst✝ : W.IsCompatibleWithTriangulation
      T₁ T₂ : CategoryTheory.Pretriangulated.Triangle C
      hT₁ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁
      hT₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₂
      a : Quiver.Hom (L.obj T₁.obj₁) (L.obj T₂.obj₁)
      b : Quiver.Hom (L.obj T₁.obj₂) (L.obj T₂.obj₂)
      fac✝ : Eq (CategoryTheory.CategoryStruct.comp (L.map T₁.mor₁) b) (CategoryTheo …
      α : W.LeftFraction T₁.obj₁ T₂.obj₁
      hα : Eq a (α.map L ⋯)
      β : W.LeftFraction α.Y' T₂.obj₂
      hβ : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₁ β.s) (CategoryTheory.Categ …
      γ : W.LeftFraction T₁.obj₂ β.Y'
      hγ : Eq (CategoryTheory.CategoryStruct.comp b (L.map β.s)) (γ.map L ⋯)
      this✝¹ : CategoryTheory.IsIso (L.map β.s)
      this✝ : CategoryTheory.IsIso (L.map γ.s)
      Z₂ : C
      σ : Quiver.Hom γ.Y' Z₂
      hσ : W σ
      fac : Eq (CategoryTheory.CategoryStruct.comp α.f (CategoryTheory.CategoryStruc …
      Y₃ : C
      g : Quiver.Hom Z₂ Y₃
      h : Quiver.Hom Y₃ ((CategoryTheory.shiftFunctor C 1).obj α.Y')
      T₃ : CategoryTheory.Pretriangulated.Triangle C := CategoryTheory.Pretriangulat …
      hT₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₃
      hβγσ : W (CategoryTheory.CategoryStruct.comp β.s (CategoryTheory.CategoryStruc …
      ψ₃ : Quiver.Hom T₂.obj₃ T₃.obj₃
      hψ₃ : W ψ₃
      hψ₁ : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₂ ψ₃) (CategoryTheory.Categ …
      hψ₂ : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₃ ((CategoryTheory.shiftFun …
      ψ : Quiver.Hom T₂ T₃ := T₂.homMk T₃ α.s (CategoryTheory.CategoryStruct.comp β. …
      this : CategoryTheory.IsIso (L.mapTriangle.map ψ)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (L.mapTriangle.map (CategoryTheory.Pr …
    -/
  · rw [← cancel_mono (L.mapTriangle.map ψ).hom₁, ← comp_hom₁, assoc, IsIso.inv_hom_id, comp_id]
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.r …
      C : Type u_1
      D : Type u_2
      inst✝¹¹ : CategoryTheory.Category.{u_4, u_1} C
      inst✝¹⁰ : CategoryTheory.Category.{u_3, u_2} D
      L : CategoryTheory.Functor C D
      inst✝⁹ : CategoryTheory.HasShift C Int
      inst✝⁸ : CategoryTheory.Preadditive C
      inst✝⁷ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝⁵ : CategoryTheory.Pretriangulated C
      inst✝⁴ : CategoryTheory.HasShift D Int
      inst✝³ : L.CommShift Int
      W : CategoryTheory.MorphismProperty C
      inst✝² : L.IsLocalization W
      inst✝¹ : W.HasLeftCalculusOfFractions
      inst✝ : W.IsCompatibleWithTriangulation
      T₁ T₂ : CategoryTheory.Pretriangulated.Triangle C
      hT₁ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁
      hT₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₂
      a : Quiver.Hom (L.obj T₁.obj₁) (L.obj T₂.obj₁)
      b : Quiver.Hom (L.obj T₁.obj₂) (L.obj T₂.obj₂)
      fac✝ : Eq (CategoryTheory.CategoryStruct.comp (L.map T₁.mor₁) b) (CategoryTheo …
      α : W.LeftFraction T₁.obj₁ T₂.obj₁
      hα : Eq a (α.map L ⋯)
      β : W.LeftFraction α.Y' T₂.obj₂
      hβ : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₁ β.s) (CategoryTheory.Categ …
      γ : W.LeftFraction T₁.obj₂ β.Y'
      hγ : Eq (CategoryTheory.CategoryStruct.comp b (L.map β.s)) (γ.map L ⋯)
      this✝¹ : CategoryTheory.IsIso (L.map β.s)
      this✝ : CategoryTheory.IsIso (L.map γ.s)
      Z₂ : C
      σ : Quiver.Hom γ.Y' Z₂
      hσ : W σ
      fac : Eq (CategoryTheory.CategoryStruct.comp α.f (CategoryTheory.CategoryStruc …
      Y₃ : C
      g : Quiver.Hom Z₂ Y₃
      h : Quiver.Hom Y₃ ((CategoryTheory.shiftFunctor C 1).obj α.Y')
      T₃ : CategoryTheory.Pretriangulated.Triangle C := CategoryTheory.Pretriangulat …
      hT₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₃
      hβγσ : W (CategoryTheory.CategoryStruct.comp β.s (CategoryTheory.CategoryStruc …
      ψ₃ : Quiver.Hom T₂.obj₃ T₃.obj₃
      hψ₃ : W ψ₃
      hψ₁ : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₂ ψ₃) (CategoryTheory.Categ …
      hψ₂ : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₃ ((CategoryTheory.shiftFun …
      ψ : Quiver.Hom T₂ T₃ := T₂.homMk T₃ α.s (CategoryTheory.CategoryStruct.comp β. …
      this : CategoryTheory.IsIso (L.mapTriangle.map ψ)
      ⊢ Eq (L.mapTriangle.map (CategoryTheory.Pretriangulated.completeDistinguishedT …
    -/
    dsimp [ψ]
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.r …
      C : Type u_1
      D : Type u_2
      inst✝¹¹ : CategoryTheory.Category.{u_4, u_1} C
      inst✝¹⁰ : CategoryTheory.Category.{u_3, u_2} D
      L : CategoryTheory.Functor C D
      inst✝⁹ : CategoryTheory.HasShift C Int
      inst✝⁸ : CategoryTheory.Preadditive C
      inst✝⁷ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝⁵ : CategoryTheory.Pretriangulated C
      inst✝⁴ : CategoryTheory.HasShift D Int
      inst✝³ : L.CommShift Int
      W : CategoryTheory.MorphismProperty C
      inst✝² : L.IsLocalization W
      inst✝¹ : W.HasLeftCalculusOfFractions
      inst✝ : W.IsCompatibleWithTriangulation
      T₁ T₂ : CategoryTheory.Pretriangulated.Triangle C
      hT₁ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁
      hT₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₂
      a : Quiver.Hom (L.obj T₁.obj₁) (L.obj T₂.obj₁)
      b : Quiver.Hom (L.obj T₁.obj₂) (L.obj T₂.obj₂)
      fac✝ : Eq (CategoryTheory.CategoryStruct.comp (L.map T₁.mor₁) b) (CategoryTheo …
      α : W.LeftFraction T₁.obj₁ T₂.obj₁
      hα : Eq a (α.map L ⋯)
      β : W.LeftFraction α.Y' T₂.obj₂
      hβ : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₁ β.s) (CategoryTheory.Categ …
      γ : W.LeftFraction T₁.obj₂ β.Y'
      hγ : Eq (CategoryTheory.CategoryStruct.comp b (L.map β.s)) (γ.map L ⋯)
      this✝¹ : CategoryTheory.IsIso (L.map β.s)
      this✝ : CategoryTheory.IsIso (L.map γ.s)
      Z₂ : C
      σ : Quiver.Hom γ.Y' Z₂
      hσ : W σ
      fac : Eq (CategoryTheory.CategoryStruct.comp α.f (CategoryTheory.CategoryStruc …
      Y₃ : C
      g : Quiver.Hom Z₂ Y₃
      h : Quiver.Hom Y₃ ((CategoryTheory.shiftFunctor C 1).obj α.Y')
      T₃ : CategoryTheory.Pretriangulated.Triangle C := CategoryTheory.Pretriangulat …
      hT₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₃
      hβγσ : W (CategoryTheory.CategoryStruct.comp β.s (CategoryTheory.CategoryStruc …
      ψ₃ : Quiver.Hom T₂.obj₃ T₃.obj₃
      hψ₃ : W ψ₃
      hψ₁ : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₂ ψ₃) (CategoryTheory.Categ …
      hψ₂ : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₃ ((CategoryTheory.shiftFun …
      ψ : Quiver.Hom T₂ T₃ := T₂.homMk T₃ α.s (CategoryTheory.CategoryStruct.comp β. …
      this : CategoryTheory.IsIso (L.mapTriangle.map ψ)
      ⊢ Eq (L.map α.f) (CategoryTheory.CategoryStruct.comp a (L.map α.s))
    -/
    rw [hα, MorphismProperty.LeftFraction.map_comp_map_s]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.r …
      C : Type u_1
      D : Type u_2
      inst✝¹¹ : CategoryTheory.Category.{u_4, u_1} C
      inst✝¹⁰ : CategoryTheory.Category.{u_3, u_2} D
      L : CategoryTheory.Functor C D
      inst✝⁹ : CategoryTheory.HasShift C Int
      inst✝⁸ : CategoryTheory.Preadditive C
      inst✝⁷ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝⁵ : CategoryTheory.Pretriangulated C
      inst✝⁴ : CategoryTheory.HasShift D Int
      inst✝³ : L.CommShift Int
      W : CategoryTheory.MorphismProperty C
      inst✝² : L.IsLocalization W
      inst✝¹ : W.HasLeftCalculusOfFractions
      inst✝ : W.IsCompatibleWithTriangulation
      T₁ T₂ : CategoryTheory.Pretriangulated.Triangle C
      hT₁ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁
      hT₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₂
      a : Quiver.Hom (L.obj T₁.obj₁) (L.obj T₂.obj₁)
      b : Quiver.Hom (L.obj T₁.obj₂) (L.obj T₂.obj₂)
      fac✝ : Eq (CategoryTheory.CategoryStruct.comp (L.map T₁.mor₁) b) (CategoryTheo …
      α : W.LeftFraction T₁.obj₁ T₂.obj₁
      hα : Eq a (α.map L ⋯)
      β : W.LeftFraction α.Y' T₂.obj₂
      hβ : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₁ β.s) (CategoryTheory.Categ …
      γ : W.LeftFraction T₁.obj₂ β.Y'
      hγ : Eq (CategoryTheory.CategoryStruct.comp b (L.map β.s)) (γ.map L ⋯)
      this✝¹ : CategoryTheory.IsIso (L.map β.s)
      this✝ : CategoryTheory.IsIso (L.map γ.s)
      Z₂ : C
      σ : Quiver.Hom γ.Y' Z₂
      hσ : W σ
      fac : Eq (CategoryTheory.CategoryStruct.comp α.f (CategoryTheory.CategoryStruc …
      Y₃ : C
      g : Quiver.Hom Z₂ Y₃
      h : Quiver.Hom Y₃ ((CategoryTheory.shiftFunctor C 1).obj α.Y')
      T₃ : CategoryTheory.Pretriangulated.Triangle C := CategoryTheory.Pretriangulat …
      hT₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₃
      hβγσ : W (CategoryTheory.CategoryStruct.comp β.s (CategoryTheory.CategoryStruc …
      ψ₃ : Quiver.Hom T₂.obj₃ T₃.obj₃
      hψ₃ : W ψ₃
      hψ₁ : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₂ ψ₃) (CategoryTheory.Categ …
      hψ₂ : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₃ ((CategoryTheory.shiftFun …
      ψ : Quiver.Hom T₂ T₃ := T₂.homMk T₃ α.s (CategoryTheory.CategoryStruct.comp β. …
      this : CategoryTheory.IsIso (L.mapTriangle.map ψ)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (L.mapTriangle.map (CategoryTheory.Pr …
    -/
  · rw [← cancel_mono (L.mapTriangle.map ψ).hom₂, ← comp_hom₂, assoc, IsIso.inv_hom_id, comp_id]
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.r …
      C : Type u_1
      D : Type u_2
      inst✝¹¹ : CategoryTheory.Category.{u_4, u_1} C
      inst✝¹⁰ : CategoryTheory.Category.{u_3, u_2} D
      L : CategoryTheory.Functor C D
      inst✝⁹ : CategoryTheory.HasShift C Int
      inst✝⁸ : CategoryTheory.Preadditive C
      inst✝⁷ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁶ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝⁵ : CategoryTheory.Pretriangulated C
      inst✝⁴ : CategoryTheory.HasShift D Int
      inst✝³ : L.CommShift Int
      W : CategoryTheory.MorphismProperty C
      inst✝² : L.IsLocalization W
      inst✝¹ : W.HasLeftCalculusOfFractions
      inst✝ : W.IsCompatibleWithTriangulation
      T₁ T₂ : CategoryTheory.Pretriangulated.Triangle C
      hT₁ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₁
      hT₂ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₂
      a : Quiver.Hom (L.obj T₁.obj₁) (L.obj T₂.obj₁)
      b : Quiver.Hom (L.obj T₁.obj₂) (L.obj T₂.obj₂)
      fac✝ : Eq (CategoryTheory.CategoryStruct.comp (L.map T₁.mor₁) b) (CategoryTheo …
      α : W.LeftFraction T₁.obj₁ T₂.obj₁
      hα : Eq a (α.map L ⋯)
      β : W.LeftFraction α.Y' T₂.obj₂
      hβ : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₁ β.s) (CategoryTheory.Categ …
      γ : W.LeftFraction T₁.obj₂ β.Y'
      hγ : Eq (CategoryTheory.CategoryStruct.comp b (L.map β.s)) (γ.map L ⋯)
      this✝¹ : CategoryTheory.IsIso (L.map β.s)
      this✝ : CategoryTheory.IsIso (L.map γ.s)
      Z₂ : C
      σ : Quiver.Hom γ.Y' Z₂
      hσ : W σ
      fac : Eq (CategoryTheory.CategoryStruct.comp α.f (CategoryTheory.CategoryStruc …
      Y₃ : C
      g : Quiver.Hom Z₂ Y₃
      h : Quiver.Hom Y₃ ((CategoryTheory.shiftFunctor C 1).obj α.Y')
      T₃ : CategoryTheory.Pretriangulated.Triangle C := CategoryTheory.Pretriangulat …
      hT₃ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₃
      hβγσ : W (CategoryTheory.CategoryStruct.comp β.s (CategoryTheory.CategoryStruc …
      ψ₃ : Quiver.Hom T₂.obj₃ T₃.obj₃
      hψ₃ : W ψ₃
      hψ₁ : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₂ ψ₃) (CategoryTheory.Categ …
      hψ₂ : Eq (CategoryTheory.CategoryStruct.comp T₂.mor₃ ((CategoryTheory.shiftFun …
      ψ : Quiver.Hom T₂ T₃ := T₂.homMk T₃ α.s (CategoryTheory.CategoryStruct.comp β. …
      this : CategoryTheory.IsIso (L.mapTriangle.map ψ)
      ⊢ Eq (L.mapTriangle.map (CategoryTheory.Pretriangulated.completeDistinguishedT …
    -/
    dsimp [ψ]
    simp only [Functor.map_comp, reassoc_of% hγ,
      MorphismProperty.LeftFraction.map_comp_map_s_assoc]


/-- The pretriangulated structure on the localized category. -/
def pretriangulated : Pretriangulated D where
  distinguishedTriangles := L.essImageDistTriang
  isomorphic_distinguished _ hT₁ _ e := L.essImageDistTriang_mem_of_iso e hT₁
  contractible_distinguished :=
    have := essSurj L W; L.contractible_mem_essImageDistTriang
  distinguished_cocone_triangle f := distinguished_cocone_triangle L W f
  rotate_distinguished_triangle := L.rotate_essImageDistTriang
  complete_distinguished_triangle_morphism := complete_distinguished_triangle_morphism L W


instance isTriangulated_functor :
    letI : Pretriangulated D := pretriangulated L W; L.IsTriangulated :=
  letI : Pretriangulated D := pretriangulated L W
  ⟨fun T hT => ⟨T, Iso.refl _, hT⟩⟩


include W in
lemma isTriangulated [Pretriangulated D] [L.IsTriangulated] [IsTriangulated C] :
    IsTriangulated D := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹⁶ : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹⁵ : CategoryTheory.Category.{u_3, u_2} D
    L : CategoryTheory.Functor C D
    inst✝¹⁴ : CategoryTheory.HasShift C Int
    inst✝¹³ : CategoryTheory.Preadditive C
    inst✝¹² : CategoryTheory.Limits.HasZeroObject C
    inst✝¹¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝¹⁰ : CategoryTheory.Pretriangulated C
    inst✝⁹ : CategoryTheory.HasShift D Int
    inst✝⁸ : L.CommShift Int
    W : CategoryTheory.MorphismProperty C
    inst✝⁷ : L.IsLocalization W
    inst✝⁶ : W.HasLeftCalculusOfFractions
    inst✝⁵ : CategoryTheory.Limits.HasZeroObject D
    inst✝⁴ : CategoryTheory.Preadditive D
    inst✝³ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
    inst✝² : CategoryTheory.Pretriangulated D
    inst✝¹ : L.IsTriangulated
    inst✝ : CategoryTheory.IsTriangulated C
    ⊢ CategoryTheory.IsTriangulated D
  -/
  have := essSurj_mapComposableArrows L W 2
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹⁶ : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹⁵ : CategoryTheory.Category.{u_3, u_2} D
    L : CategoryTheory.Functor C D
    inst✝¹⁴ : CategoryTheory.HasShift C Int
    inst✝¹³ : CategoryTheory.Preadditive C
    inst✝¹² : CategoryTheory.Limits.HasZeroObject C
    inst✝¹¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝¹⁰ : CategoryTheory.Pretriangulated C
    inst✝⁹ : CategoryTheory.HasShift D Int
    inst✝⁸ : L.CommShift Int
    W : CategoryTheory.MorphismProperty C
    inst✝⁷ : L.IsLocalization W
    inst✝⁶ : W.HasLeftCalculusOfFractions
    inst✝⁵ : CategoryTheory.Limits.HasZeroObject D
    inst✝⁴ : CategoryTheory.Preadditive D
    inst✝³ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
    inst✝² : CategoryTheory.Pretriangulated D
    inst✝¹ : L.IsTriangulated
    inst✝ : CategoryTheory.IsTriangulated C
    this : (L.mapComposableArrows 2).EssSurj
    ⊢ CategoryTheory.IsTriangulated D
  -/
  exact isTriangulated_of_essSurj_mapComposableArrows_two L
  /-
    🎉 no goals
  -/


instance (n : ℤ) : (shiftFunctor (W.Localization) n).Additive := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹³ : CategoryTheory.Category.{?u.93926, u_2} D
    L : CategoryTheory.Functor C D
    inst✝¹² : CategoryTheory.HasShift C Int
    inst✝¹¹ : CategoryTheory.Preadditive C
    inst✝¹⁰ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝⁸ : CategoryTheory.Pretriangulated C
    inst✝⁷ : CategoryTheory.HasShift D Int
    inst✝⁶ : L.CommShift Int
    W : CategoryTheory.MorphismProperty C
    inst✝⁵ : L.IsLocalization W
    inst✝⁴ : W.HasLeftCalculusOfFractions
    inst✝³ : CategoryTheory.Limits.HasZeroObject D
    inst✝² : CategoryTheory.Preadditive D
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
    inst✝ : W.IsCompatibleWithTriangulation
    n : Int
    ⊢ (CategoryTheory.shiftFunctor W.Localization n).Additive
  -/
  rw [Localization.functor_additive_iff W.Q W]
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹⁴ : CategoryTheory.Category.{u_3, u_1} C
    inst✝¹³ : CategoryTheory.Category.{?u.93926, u_2} D
    L : CategoryTheory.Functor C D
    inst✝¹² : CategoryTheory.HasShift C Int
    inst✝¹¹ : CategoryTheory.Preadditive C
    inst✝¹⁰ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝⁸ : CategoryTheory.Pretriangulated C
    inst✝⁷ : CategoryTheory.HasShift D Int
    inst✝⁶ : L.CommShift Int
    W : CategoryTheory.MorphismProperty C
    inst✝⁵ : L.IsLocalization W
    inst✝⁴ : W.HasLeftCalculusOfFractions
    inst✝³ : CategoryTheory.Limits.HasZeroObject D
    inst✝² : CategoryTheory.Preadditive D
    inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
    inst✝ : W.IsCompatibleWithTriangulation
    n : Int
    ⊢ (W.Q.comp (CategoryTheory.shiftFunctor W.Localization n)).Additive
  -/
  exact Functor.additive_of_iso (W.Q.commShiftIso n)
  /-
    🎉 no goals
  -/


instance : Pretriangulated W.Localization := pretriangulated W.Q W


instance [IsTriangulated C] : IsTriangulated W.Localization := isTriangulated W.Q W


instance (n : ℤ) : (shiftFunctor (W.Localization') n).Additive := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹⁵ : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹⁴ : CategoryTheory.Category.{?u.100130, u_2} D
    L : CategoryTheory.Functor C D
    inst✝¹³ : CategoryTheory.HasShift C Int
    inst✝¹² : CategoryTheory.Preadditive C
    inst✝¹¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝¹⁰ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝⁹ : CategoryTheory.Pretriangulated C
    inst✝⁸ : CategoryTheory.HasShift D Int
    inst✝⁷ : L.CommShift Int
    W : CategoryTheory.MorphismProperty C
    inst✝⁶ : L.IsLocalization W
    inst✝⁵ : W.HasLeftCalculusOfFractions
    inst✝⁴ : CategoryTheory.Limits.HasZeroObject D
    inst✝³ : CategoryTheory.Preadditive D
    inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
    inst✝¹ : W.IsCompatibleWithTriangulation
    inst✝ : W.HasLocalization
    n : Int
    ⊢ (CategoryTheory.shiftFunctor W.Localization' n).Additive
  -/
  rw [Localization.functor_additive_iff W.Q' W]
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹⁵ : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹⁴ : CategoryTheory.Category.{?u.100130, u_2} D
    L : CategoryTheory.Functor C D
    inst✝¹³ : CategoryTheory.HasShift C Int
    inst✝¹² : CategoryTheory.Preadditive C
    inst✝¹¹ : CategoryTheory.Limits.HasZeroObject C
    inst✝¹⁰ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝⁹ : CategoryTheory.Pretriangulated C
    inst✝⁸ : CategoryTheory.HasShift D Int
    inst✝⁷ : L.CommShift Int
    W : CategoryTheory.MorphismProperty C
    inst✝⁶ : L.IsLocalization W
    inst✝⁵ : W.HasLeftCalculusOfFractions
    inst✝⁴ : CategoryTheory.Limits.HasZeroObject D
    inst✝³ : CategoryTheory.Preadditive D
    inst✝² : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
    inst✝¹ : W.IsCompatibleWithTriangulation
    inst✝ : W.HasLocalization
    n : Int
    ⊢ (W.Q'.comp (CategoryTheory.shiftFunctor W.Localization' n)).Additive
  -/
  exact Functor.additive_of_iso (W.Q'.commShiftIso n)
  /-
    🎉 no goals
  -/


instance : Pretriangulated W.Localization' := pretriangulated W.Q' W


instance [IsTriangulated C] : IsTriangulated W.Localization' := isTriangulated W.Q' W


lemma distTriang_iff (T : Triangle D) :
    (T ∈ distTriang D) ↔ T ∈ L.essImageDistTriang := by
  /-
    C : Type u_1
    D : Type u_2
    inst✝¹⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝¹³ : CategoryTheory.Category.{u_3, u_2} D
    L : CategoryTheory.Functor C D
    inst✝¹² : CategoryTheory.HasShift C Int
    inst✝¹¹ : CategoryTheory.Preadditive C
    inst✝¹⁰ : CategoryTheory.Limits.HasZeroObject C
    inst✝⁹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    inst✝⁸ : CategoryTheory.Pretriangulated C
    inst✝⁷ : CategoryTheory.HasShift D Int
    inst✝⁶ : L.CommShift Int
    inst✝⁵ : CategoryTheory.Limits.HasZeroObject D
    inst✝⁴ : CategoryTheory.Preadditive D
    inst✝³ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
    inst✝² : CategoryTheory.Pretriangulated D
    inst✝¹ : L.mapArrow.EssSurj
    inst✝ : L.IsTriangulated
    T : CategoryTheory.Pretriangulated.Triangle D
    ⊢ Iff (Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T) …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      D : Type u_2
      inst✝¹⁴ : CategoryTheory.Category.{u_4, u_1} C
      inst✝¹³ : CategoryTheory.Category.{u_3, u_2} D
      L : CategoryTheory.Functor C D
      inst✝¹² : CategoryTheory.HasShift C Int
      inst✝¹¹ : CategoryTheory.Preadditive C
      inst✝¹⁰ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝⁸ : CategoryTheory.Pretriangulated C
      inst✝⁷ : CategoryTheory.HasShift D Int
      inst✝⁶ : L.CommShift Int
      inst✝⁵ : CategoryTheory.Limits.HasZeroObject D
      inst✝⁴ : CategoryTheory.Preadditive D
      inst✝³ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
      inst✝² : CategoryTheory.Pretriangulated D
      inst✝¹ : L.mapArrow.EssSurj
      inst✝ : L.IsTriangulated
      T : CategoryTheory.Pretriangulated.Triangle D
      ⊢ Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T → Mem …
    -/
  · intro hT
    /-
      case mp
      C : Type u_1
      D : Type u_2
      inst✝¹⁴ : CategoryTheory.Category.{u_4, u_1} C
      inst✝¹³ : CategoryTheory.Category.{u_3, u_2} D
      L : CategoryTheory.Functor C D
      inst✝¹² : CategoryTheory.HasShift C Int
      inst✝¹¹ : CategoryTheory.Preadditive C
      inst✝¹⁰ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝⁸ : CategoryTheory.Pretriangulated C
      inst✝⁷ : CategoryTheory.HasShift D Int
      inst✝⁶ : L.CommShift Int
      inst✝⁵ : CategoryTheory.Limits.HasZeroObject D
      inst✝⁴ : CategoryTheory.Preadditive D
      inst✝³ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
      inst✝² : CategoryTheory.Pretriangulated D
      inst✝¹ : L.mapArrow.EssSurj
      inst✝ : L.IsTriangulated
      T : CategoryTheory.Pretriangulated.Triangle D
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      ⊢ Membership.mem L.essImageDistTriang T
    -/
    let f := L.mapArrow.objPreimage T.mor₁
    obtain ⟨Z, g : f.right ⟶ Z, h : Z ⟶ f.left⟦(1 : ℤ)⟧, mem⟩ :=
      Pretriangulated.distinguished_cocone_triangle f.hom
    exact ⟨_, (exists_iso_of_arrow_iso T _ hT (L.map_distinguished _ mem)
      (L.mapArrow.objObjPreimageIso T.mor₁).symm).choose, mem⟩
    /-
      case mpr
      C : Type u_1
      D : Type u_2
      inst✝¹⁴ : CategoryTheory.Category.{u_4, u_1} C
      inst✝¹³ : CategoryTheory.Category.{u_3, u_2} D
      L : CategoryTheory.Functor C D
      inst✝¹² : CategoryTheory.HasShift C Int
      inst✝¹¹ : CategoryTheory.Preadditive C
      inst✝¹⁰ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝⁸ : CategoryTheory.Pretriangulated C
      inst✝⁷ : CategoryTheory.HasShift D Int
      inst✝⁶ : L.CommShift Int
      inst✝⁵ : CategoryTheory.Limits.HasZeroObject D
      inst✝⁴ : CategoryTheory.Preadditive D
      inst✝³ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
      inst✝² : CategoryTheory.Pretriangulated D
      inst✝¹ : L.mapArrow.EssSurj
      inst✝ : L.IsTriangulated
      T : CategoryTheory.Pretriangulated.Triangle D
      ⊢ Membership.mem L.essImageDistTriang T → Membership.mem CategoryTheory.Pretri …
    -/
  · rintro ⟨T₀, e, hT₀⟩
    /-
      case mpr.intro.intro
      C : Type u_1
      D : Type u_2
      inst✝¹⁴ : CategoryTheory.Category.{u_4, u_1} C
      inst✝¹³ : CategoryTheory.Category.{u_3, u_2} D
      L : CategoryTheory.Functor C D
      inst✝¹² : CategoryTheory.HasShift C Int
      inst✝¹¹ : CategoryTheory.Preadditive C
      inst✝¹⁰ : CategoryTheory.Limits.HasZeroObject C
      inst✝⁹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝⁸ : CategoryTheory.Pretriangulated C
      inst✝⁷ : CategoryTheory.HasShift D Int
      inst✝⁶ : L.CommShift Int
      inst✝⁵ : CategoryTheory.Limits.HasZeroObject D
      inst✝⁴ : CategoryTheory.Preadditive D
      inst✝³ : ∀ (n : Int), (CategoryTheory.shiftFunctor D n).Additive
      inst✝² : CategoryTheory.Pretriangulated D
      inst✝¹ : L.mapArrow.EssSurj
      inst✝ : L.IsTriangulated
      T : CategoryTheory.Pretriangulated.Triangle D
      T₀ : CategoryTheory.Pretriangulated.Triangle C
      e : CategoryTheory.Iso T (L.mapTriangle.obj T₀)
      hT₀ : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T₀
      ⊢ Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
    -/
    exact isomorphic_distinguished _ (L.map_distinguished _ hT₀) _ e
    /-
      🎉 no goals
    -/


