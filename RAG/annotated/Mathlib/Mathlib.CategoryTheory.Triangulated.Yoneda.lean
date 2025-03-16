instance (A : Cᵒᵖ) : (preadditiveCoyoneda.obj A).IsHomological where
  exact T hT := by
    /-
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁴ : CategoryTheory.Preadditive C
      inst✝³ : CategoryTheory.HasShift C Int
      inst✝² : CategoryTheory.Limits.HasZeroObject C
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      A : Opposite C
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      ⊢ ((CategoryTheory.Pretriangulated.shortComplexOfDistTriangle T hT).map (Categ …
    -/
    rw [ShortComplex.ab_exact_iff]
    /-
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁴ : CategoryTheory.Preadditive C
      inst✝³ : CategoryTheory.HasShift C Int
      inst✝² : CategoryTheory.Limits.HasZeroObject C
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      A : Opposite C
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      ⊢ ∀ (x₂ : ↑((CategoryTheory.Pretriangulated.shortComplexOfDistTriangle T hT).m …
    -/
    intro (x₂ : A.unop ⟶ T.obj₂) (hx₂ : x₂ ≫ T.mor₂ = 0)
    /-
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁴ : CategoryTheory.Preadditive C
      inst✝³ : CategoryTheory.HasShift C Int
      inst✝² : CategoryTheory.Limits.HasZeroObject C
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      A : Opposite C
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      x₂ : Quiver.Hom (Opposite.unop A) T.obj₂
      hx₂ : Eq (CategoryTheory.CategoryStruct.comp x₂ T.mor₂) 0
      ⊢ Exists fun x₁ => Eq (((CategoryTheory.Pretriangulated.shortComplexOfDistTria …
    -/
    obtain ⟨x₁, hx₁⟩ := T.coyoneda_exact₂ hT x₂ hx₂
    /-
      case intro
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁴ : CategoryTheory.Preadditive C
      inst✝³ : CategoryTheory.HasShift C Int
      inst✝² : CategoryTheory.Limits.HasZeroObject C
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      A : Opposite C
      T : CategoryTheory.Pretriangulated.Triangle C
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      x₂ : Quiver.Hom (Opposite.unop A) T.obj₂
      hx₂ : Eq (CategoryTheory.CategoryStruct.comp x₂ T.mor₂) 0
      x₁ : Quiver.Hom (Opposite.unop A) T.obj₁
      hx₁ : Eq x₂ (CategoryTheory.CategoryStruct.comp x₁ T.mor₁)
      ⊢ Exists fun x₁ => Eq (((CategoryTheory.Pretriangulated.shortComplexOfDistTria …
    -/
    exact ⟨x₁, hx₁.symm⟩
    /-
      🎉 no goals
    -/


instance (B : C) : (preadditiveYoneda.obj B).IsHomological where
  exact T hT := by
    /-
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁴ : CategoryTheory.Preadditive C
      inst✝³ : CategoryTheory.HasShift C Int
      inst✝² : CategoryTheory.Limits.HasZeroObject C
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      B : C
      T : CategoryTheory.Pretriangulated.Triangle (Opposite C)
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      ⊢ ((CategoryTheory.Pretriangulated.shortComplexOfDistTriangle T hT).map (Categ …
    -/
    rw [ShortComplex.ab_exact_iff]
    /-
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁴ : CategoryTheory.Preadditive C
      inst✝³ : CategoryTheory.HasShift C Int
      inst✝² : CategoryTheory.Limits.HasZeroObject C
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      B : C
      T : CategoryTheory.Pretriangulated.Triangle (Opposite C)
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      ⊢ ∀ (x₂ : ↑((CategoryTheory.Pretriangulated.shortComplexOfDistTriangle T hT).m …
    -/
    intro (x₂ : T.obj₂.unop ⟶ B) (hx₂ : T.mor₂.unop ≫ x₂ = 0)
    /-
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁴ : CategoryTheory.Preadditive C
      inst✝³ : CategoryTheory.HasShift C Int
      inst✝² : CategoryTheory.Limits.HasZeroObject C
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      B : C
      T : CategoryTheory.Pretriangulated.Triangle (Opposite C)
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      x₂ : Quiver.Hom (Opposite.unop T.obj₂) B
      hx₂ : Eq (CategoryTheory.CategoryStruct.comp T.mor₂.unop x₂) 0
      ⊢ Exists fun x₁ => Eq (((CategoryTheory.Pretriangulated.shortComplexOfDistTria …
    -/
    obtain ⟨x₃, hx₃⟩ := Triangle.yoneda_exact₂ _ (unop_distinguished T hT) x₂ hx₂
    /-
      case intro
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      inst✝⁴ : CategoryTheory.Preadditive C
      inst✝³ : CategoryTheory.HasShift C Int
      inst✝² : CategoryTheory.Limits.HasZeroObject C
      inst✝¹ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      inst✝ : CategoryTheory.Pretriangulated C
      B : C
      T : CategoryTheory.Pretriangulated.Triangle (Opposite C)
      hT : Membership.mem CategoryTheory.Pretriangulated.distinguishedTriangles T
      x₂ : Quiver.Hom (Opposite.unop T.obj₂) B
      hx₂ : Eq (CategoryTheory.CategoryStruct.comp T.mor₂.unop x₂) 0
      x₃ : Quiver.Hom (Opposite.unop ((CategoryTheory.Pretriangulated.triangleOpEqui …
      hx₃ : Eq x₂ (CategoryTheory.CategoryStruct.comp (Opposite.unop ((CategoryTheor …
      ⊢ Exists fun x₁ => Eq (((CategoryTheory.Pretriangulated.shortComplexOfDistTria …
    -/
    exact ⟨x₃, hx₃.symm⟩
    /-
      🎉 no goals
    -/


lemma preadditiveYoneda_map_distinguished
    (T : Triangle C) (hT : T ∈ distTriang C) (B : C) :
    ((shortComplexOfDistTriangle T hT).op.map (preadditiveYoneda.obj B)).Exact :=
  (preadditiveYoneda.obj B).map_distinguished_op_exact T hT


noncomputable instance (A : Cᵒᵖ) : (preadditiveCoyoneda.obj A).ShiftSequence ℤ :=
  Functor.ShiftSequence.tautological _ _


lemma preadditiveCoyoneda_homologySequenceδ_apply
    (T : Triangle C) (n₀ n₁ : ℤ) (h : n₀ + 1 = n₁) {A : Cᵒᵖ} (x : A.unop ⟶ T.obj₃⟦n₀⟧) :
    (preadditiveCoyoneda.obj A).homologySequenceδ T n₀ n₁ h x =
                                                        /-
                                                          C : Type u_1
                                                          inst✝² : CategoryTheory.Category.{?u.7641, u_1} C
                                                          inst✝¹ : CategoryTheory.Preadditive C
                                                          inst✝ : CategoryTheory.HasShift C Int
                                                          T : CategoryTheory.Pretriangulated.Triangle C
                                                          n₀ n₁ : Int
                                                          h : Eq (HAdd.hAdd n₀ 1) n₁
                                                          A : Opposite C
                                                          x : Quiver.Hom (Opposite.unop A) ((CategoryTheory.shiftFunctor C n₀).obj T.obj₃)
                                                          ⊢ Eq (HAdd.hAdd 1 n₀) n₁
                                                        -/
      x ≫ T.mor₃⟦n₀⟧' ≫ (shiftFunctorAdd' C 1 n₀ n₁ (by omega)).inv.app _ := by
                                                        /-
                                                          🎉 no goals
                                                        -/
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Preadditive C
    inst✝ : CategoryTheory.HasShift C Int
    T : CategoryTheory.Pretriangulated.Triangle C
    n₀ n₁ : Int
    h : Eq (HAdd.hAdd n₀ 1) n₁
    A : Opposite C
    x : Quiver.Hom (Opposite.unop A) ((CategoryTheory.shiftFunctor C n₀).obj T.obj₃)
    ⊢ Eq (((CategoryTheory.preadditiveCoyoneda.obj A).homologySequenceδ T n₀ n₁ h) …
  -/
  apply Category.assoc
  /-
    🎉 no goals
  -/


noncomputable instance (B : C) : (preadditiveYoneda.obj B).ShiftSequence ℤ where
  sequence n := preadditiveYoneda.obj (B⟦n⟧)
  isoZero := preadditiveYoneda.mapIso ((shiftFunctorZero C ℤ).app B)
  shiftIso n a a' h := NatIso.ofComponents (fun A ↦ AddEquiv.toAddCommGrpIso
    { toEquiv := Quiver.Hom.opEquiv.trans (ShiftedHom.opEquiv' n a a' h).symm
      map_add' := fun _ _ ↦ ShiftedHom.opEquiv'_symm_add _ _ _ h })
            /-
              C : Type u_1
              inst✝³ : CategoryTheory.Category.{?u.11399, u_1} C
              inst✝² : CategoryTheory.Preadditive C
              inst✝¹ : CategoryTheory.HasShift C Int
              inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
              B : C
              n a a' : Int
              h : Eq (HAdd.hAdd n a) a'
              ⊢ ∀ {X Y : Opposite C} (f : Quiver.Hom X Y), Eq (CategoryTheory.CategoryStruct …
            -/
        (by intros; ext; apply ShiftedHom.opEquiv'_symm_comp _ _ _ h)
                         /-
                           🎉 no goals
                         -/
                        /-
                          C : Type u_1
                          inst✝³ : CategoryTheory.Category.{?u.11399, u_1} C
                          inst✝² : CategoryTheory.Preadditive C
                          inst✝¹ : CategoryTheory.HasShift C Int
                          inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
                          B : C
                          a : Int
                          ⊢ Eq ((fun n a a' h => CategoryTheory.NatIso.ofComponents (fun A => { toEquiv  …
                        -/
  shiftIso_zero a := by ext; apply ShiftedHom.opEquiv'_zero_add_symm
                             /-
                               🎉 no goals
                             -/
  shiftIso_add n m a a' a'' ha' ha'' := by
    /-
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.11399, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.HasShift C Int
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      B : C
      n m a a' a'' : Int
      ha' : Eq (HAdd.hAdd n a) a'
      ha'' : Eq (HAdd.hAdd m a') a''
      ⊢ Eq ((fun n a a' h => CategoryTheory.NatIso.ofComponents (fun A => { toEquiv  …
    -/
    ext _ x
    /-
      case w.w.h.w
      C : Type u_1
      inst✝³ : CategoryTheory.Category.{?u.11399, u_1} C
      inst✝² : CategoryTheory.Preadditive C
      inst✝¹ : CategoryTheory.HasShift C Int
      inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
      B : C
      n m a a' a'' : Int
      ha' : Eq (HAdd.hAdd n a) a'
      ha'' : Eq (HAdd.hAdd m a') a''
      x✝ : Opposite C
      x : ↑(((CategoryTheory.shiftFunctor (Opposite C) (HAdd.hAdd m n)).comp ((fun n …
      ⊢ Eq ((((fun n a a' h => CategoryTheory.NatIso.ofComponents (fun A => { toEqui …
    -/
    exact ShiftedHom.opEquiv'_add_symm n m a a' a'' ha' ha'' x.op
    /-
      🎉 no goals
    -/


lemma preadditiveYoneda_shiftMap_apply (B : C) {X Y : Cᵒᵖ} (n : ℤ) (f : X ⟶ Y⟦n⟧)
    (a a' : ℤ) (h : n + a = a') (z : X.unop ⟶ B⟦a⟧) :
    (preadditiveYoneda.obj B).shiftMap f a a' h z =
                                                                 /-
                                                                   C : Type u_1
                                                                   inst✝³ : CategoryTheory.Category.{?u.22278, u_1} C
                                                                   inst✝² : CategoryTheory.Preadditive C
                                                                   inst✝¹ : CategoryTheory.HasShift C Int
                                                                   inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
                                                                   B : C
                                                                   X Y : Opposite C
                                                                   n : Int
                                                                   f : Quiver.Hom X ((CategoryTheory.shiftFunctor (Opposite C) n).obj Y)
                                                                   a a' : Int
                                                                   h : Eq (HAdd.hAdd n a) a'
                                                                   z : Quiver.Hom (Opposite.unop X) ((CategoryTheory.shiftFunctor C a).obj B)
                                                                   ⊢ Eq (HAdd.hAdd a n) a'
                                                                 -/
      ((ShiftedHom.opEquiv _).symm f).comp z (show a + n = a' by omega) := by
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.HasShift C Int
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    B : C
    X Y : Opposite C
    n : Int
    f : Quiver.Hom X ((CategoryTheory.shiftFunctor (Opposite C) n).obj Y)
    a a' : Int
    h : Eq (HAdd.hAdd n a) a'
    z : Quiver.Hom (Opposite.unop X) ((CategoryTheory.shiftFunctor C a).obj B)
    ⊢ Eq (((CategoryTheory.preadditiveYoneda.obj B).shiftMap f a a' h) z) (((Categ …
  -/
  symm
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.HasShift C Int
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    B : C
    X Y : Opposite C
    n : Int
    f : Quiver.Hom X ((CategoryTheory.shiftFunctor (Opposite C) n).obj Y)
    a a' : Int
    h : Eq (HAdd.hAdd n a) a'
    z : Quiver.Hom (Opposite.unop X) ((CategoryTheory.shiftFunctor C a).obj B)
    ⊢ Eq (((CategoryTheory.ShiftedHom.opEquiv n).symm f).comp z ⋯) (((CategoryTheo …
  -/
  apply ShiftedHom.opEquiv_symm_apply_comp
  /-
    🎉 no goals
  -/


lemma preadditiveYoneda_homologySequenceδ_apply
    (T : Triangle C) (n₀ n₁ : ℤ) (h : n₀ + 1 = n₁) {B : C} (x : T.obj₁ ⟶ B⟦n₀⟧) :
    (preadditiveYoneda.obj B).homologySequenceδ
      ((triangleOpEquivalence _).functor.obj (op T)) n₀ n₁ h x =
      T.mor₃ ≫ x⟦(1 : ℤ)⟧' ≫ (shiftFunctorAdd' C n₀ 1 n₁ h).inv.app B := by
  simp only [Functor.homologySequenceδ, preadditiveYoneda_shiftMap_apply,
    ShiftedHom.comp, ← Category.assoc]
  /-
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.HasShift C Int
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    T : CategoryTheory.Pretriangulated.Triangle C
    n₀ n₁ : Int
    h : Eq (HAdd.hAdd n₀ 1) n₁
    B : C
    x : Quiver.Hom T.obj₁ ((CategoryTheory.shiftFunctor C n₀).obj B)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  congr 2
  /-
    case e_a.e_a
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.HasShift C Int
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    T : CategoryTheory.Pretriangulated.Triangle C
    n₀ n₁ : Int
    h : Eq (HAdd.hAdd n₀ 1) n₁
    B : C
    x : Quiver.Hom T.obj₁ ((CategoryTheory.shiftFunctor C n₀).obj B)
    ⊢ Eq ((CategoryTheory.ShiftedHom.opEquiv 1).symm ((CategoryTheory.Pretriangula …
  -/
  apply (ShiftedHom.opEquiv _).injective
  /-
    case e_a.e_a.a
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.HasShift C Int
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    T : CategoryTheory.Pretriangulated.Triangle C
    n₀ n₁ : Int
    h : Eq (HAdd.hAdd n₀ 1) n₁
    B : C
    x : Quiver.Hom T.obj₁ ((CategoryTheory.shiftFunctor C n₀).obj B)
    ⊢ Eq ((CategoryTheory.ShiftedHom.opEquiv 1) ((CategoryTheory.ShiftedHom.opEqui …
  -/
  rw [Equiv.apply_symm_apply]
  /-
    case e_a.e_a.a
    C : Type u_1
    inst✝³ : CategoryTheory.Category.{u_2, u_1} C
    inst✝² : CategoryTheory.Preadditive C
    inst✝¹ : CategoryTheory.HasShift C Int
    inst✝ : ∀ (n : Int), (CategoryTheory.shiftFunctor C n).Additive
    T : CategoryTheory.Pretriangulated.Triangle C
    n₀ n₁ : Int
    h : Eq (HAdd.hAdd n₀ 1) n₁
    B : C
    x : Quiver.Hom T.obj₁ ((CategoryTheory.shiftFunctor C n₀).obj B)
    ⊢ Eq ((CategoryTheory.Pretriangulated.triangleOpEquivalence C).functor.obj { u …
  -/
  rfl
  /-
    🎉 no goals
  -/


