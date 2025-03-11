theorem P_is_eventually_constant {q n : ℕ} (hqn : n ≤ q) :
    ((P (q + 1)).f n : X _[n] ⟶ _) = (P q).f n := by
  cases n with
  | zero => simp only [P_f_0_eq]
  | succ n =>
    simp only [P_succ, comp_add, comp_id, HomologicalComplex.add_f_apply, HomologicalComplex.comp_f,
      add_right_eq_self]
    exact (HigherFacesVanish.of_P q n).comp_Hσ_eq_zero (Nat.succ_le_iff.mp hqn)


theorem Q_is_eventually_constant {q n : ℕ} (hqn : n ≤ q) :
    ((Q (q + 1)).f n : X _[n] ⟶ _) = (Q q).f n := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    q n : Nat
    hqn : LE.le n q
    ⊢ Eq ((AlgebraicTopology.DoldKan.Q (HAdd.hAdd q 1)).f n) ((AlgebraicTopology.D …
  -/
  simp only [Q, HomologicalComplex.sub_f_apply, P_is_eventually_constant hqn]
  /-
    🎉 no goals
  -/


/-- The endomorphism `PInfty : K[X] ⟶ K[X]` obtained from the `P q` by passing to the limit. -/
noncomputable def PInfty : K[X] ⟶ K[X] :=
  ChainComplex.ofHom _ _ _ _ _ _ (fun n => ((P n).f n : X _[n] ⟶ _)) fun n => by
    simpa only [← P_is_eventually_constant (show n ≤ n by rfl),
      AlternatingFaceMapComplex.obj_d_eq] using (P (n + 1) : K[X] ⟶ _).comm (n + 1) n


/-- The endomorphism `QInfty : K[X] ⟶ K[X]` obtained from the `Q q` by passing to the limit. -/
noncomputable def QInfty : K[X] ⟶ K[X] :=
  𝟙 _ - PInfty


@[simp]
theorem PInfty_f_0 : (PInfty.f 0 : X _[0] ⟶ X _[0]) = 𝟙 _ :=
  rfl


theorem PInfty_f (n : ℕ) : (PInfty.f n : X _[n] ⟶ X _[n]) = (P n).f n :=
  rfl


@[simp]
theorem QInfty_f_0 : (QInfty.f 0 : X _[0] ⟶ X _[0]) = 0 := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    ⊢ Eq (AlgebraicTopology.DoldKan.QInfty.f 0) 0
  -/
  dsimp [QInfty]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    ⊢ Eq (HSub.hSub (CategoryTheory.CategoryStruct.id (X.obj { unop := SimplexCate …
  -/
  simp only [sub_self]
  /-
    🎉 no goals
  -/


theorem QInfty_f (n : ℕ) : (QInfty.f n : X _[n] ⟶ X _[n]) = (Q n).f n :=
  rfl


@[reassoc (attr := simp)]
theorem PInfty_f_naturality (n : ℕ) {X Y : SimplicialObject C} (f : X ⟶ Y) :
    f.app (op [n]) ≫ PInfty.f n = PInfty.f n ≫ f.app (op [n]) :=
  P_f_naturality n n f


@[reassoc (attr := simp)]
theorem QInfty_f_naturality (n : ℕ) {X Y : SimplicialObject C} (f : X ⟶ Y) :
    f.app (op [n]) ≫ QInfty.f n = QInfty.f n ≫ f.app (op [n]) :=
  Q_f_naturality n n f


@[reassoc (attr := simp)]
theorem PInfty_f_idem (n : ℕ) : (PInfty.f n : X _[n] ⟶ _) ≫ PInfty.f n = PInfty.f n := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.PInfty.f n …
  -/
  simp only [PInfty_f, P_f_idem]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem PInfty_idem : (PInfty : K[X] ⟶ _) ≫ PInfty = PInfty := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp AlgebraicTopology.DoldKan.PInfty Alge …
  -/
  ext n
  /-
    case h
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp AlgebraicTopology.DoldKan.PInfty Alg …
  -/
  exact PInfty_f_idem n
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem QInfty_f_idem (n : ℕ) : (QInfty.f n : X _[n] ⟶ _) ≫ QInfty.f n = QInfty.f n :=
  Q_f_idem _ _


@[reassoc (attr := simp)]
theorem QInfty_idem : (QInfty : K[X] ⟶ _) ≫ QInfty = QInfty := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp AlgebraicTopology.DoldKan.QInfty Alge …
  -/
  ext n
  /-
    case h
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp AlgebraicTopology.DoldKan.QInfty Alg …
  -/
  exact QInfty_f_idem n
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem PInfty_f_comp_QInfty_f (n : ℕ) : (PInfty.f n : X _[n] ⟶ _) ≫ QInfty.f n = 0 := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.PInfty.f n …
  -/
  dsimp only [QInfty]
  simp only [HomologicalComplex.sub_f_apply, HomologicalComplex.id_f, comp_sub, comp_id,
    PInfty_f_idem, sub_self]


@[reassoc (attr := simp)]
theorem PInfty_comp_QInfty : (PInfty : K[X] ⟶ _) ≫ QInfty = 0 := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp AlgebraicTopology.DoldKan.PInfty Alge …
  -/
  ext n
  /-
    case h
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp AlgebraicTopology.DoldKan.PInfty Alg …
  -/
  apply PInfty_f_comp_QInfty_f
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem QInfty_f_comp_PInfty_f (n : ℕ) : (QInfty.f n : X _[n] ⟶ _) ≫ PInfty.f n = 0 := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.QInfty.f n …
  -/
  dsimp only [QInfty]
  simp only [HomologicalComplex.sub_f_apply, HomologicalComplex.id_f, sub_comp, id_comp,
    PInfty_f_idem, sub_self]


@[reassoc (attr := simp)]
theorem QInfty_comp_PInfty : (QInfty : K[X] ⟶ _) ≫ PInfty = 0 := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    ⊢ Eq (CategoryTheory.CategoryStruct.comp AlgebraicTopology.DoldKan.QInfty Alge …
  -/
  ext n
  /-
    case h
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp AlgebraicTopology.DoldKan.QInfty Alg …
  -/
  apply QInfty_f_comp_PInfty_f
  /-
    🎉 no goals
  -/


@[simp]
theorem PInfty_add_QInfty : (PInfty : K[X] ⟶ _) + QInfty = 𝟙 _ := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    ⊢ Eq (HAdd.hAdd AlgebraicTopology.DoldKan.PInfty AlgebraicTopology.DoldKan.QIn …
  -/
  dsimp only [QInfty]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    ⊢ Eq (HAdd.hAdd AlgebraicTopology.DoldKan.PInfty (HSub.hSub (CategoryTheory.Ca …
  -/
  simp only [add_sub_cancel]
  /-
    🎉 no goals
  -/


theorem PInfty_f_add_QInfty_f (n : ℕ) : (PInfty.f n : X _[n] ⟶ _) + QInfty.f n = 𝟙 _ :=
  HomologicalComplex.congr_hom PInfty_add_QInfty n


/-- `PInfty` induces a natural transformation, i.e. an endomorphism of
the functor `alternatingFaceMapComplex C`. -/
@[simps]
noncomputable def natTransPInfty : alternatingFaceMapComplex C ⟶ alternatingFaceMapComplex C where
  app _ := PInfty
  naturality X Y f := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.77374, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X✝ X Y : CategoryTheory.SimplicialObject C
      f : Quiver.Hom X Y
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicTopology.alternatingFaceMa …
    -/
    ext n
    /-
      case h
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.77374, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X✝ X Y : CategoryTheory.SimplicialObject C
      f : Quiver.Hom X Y
      n : Nat
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp ((AlgebraicTopology.alternatingFaceM …
    -/
    exact PInfty_f_naturality n f
    /-
      🎉 no goals
    -/


/-- The natural transformation in each degree that is induced by `natTransPInfty`. -/
@[simps!]
noncomputable def natTransPInfty_f (n : ℕ) :=
  natTransPInfty C ◫ 𝟙 (HomologicalComplex.eval _ _ n)


@[simp]
theorem map_PInfty_f {D : Type*} [Category D] [Preadditive D] (G : C ⥤ D) [G.Additive]
    (X : SimplicialObject C) (n : ℕ) :
    (PInfty : K[((whiskering C D).obj G).obj X] ⟶ _).f n =
      G.map ((PInfty : AlternatingFaceMapComplex.obj X ⟶ _).f n) := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
    inst✝³ : CategoryTheory.Preadditive C
    D : Type u_2
    inst✝² : CategoryTheory.Category.{u_3, u_2} D
    inst✝¹ : CategoryTheory.Preadditive D
    G : CategoryTheory.Functor C D
    inst✝ : G.Additive
    X : CategoryTheory.SimplicialObject C
    n : Nat
    ⊢ Eq (AlgebraicTopology.DoldKan.PInfty.f n) (G.map (AlgebraicTopology.DoldKan. …
  -/
  simp only [PInfty_f, map_P]
  /-
    🎉 no goals
  -/


/-- Given an object `Y : Karoubi (SimplicialObject C)`, this lemma
computes `PInfty` for the associated object in `SimplicialObject (Karoubi C)`
in terms of `PInfty` for `Y.X : SimplicialObject C` and `Y.p`. -/
theorem karoubi_PInfty_f {Y : Karoubi (SimplicialObject C)} (n : ℕ) :
    ((PInfty : K[(karoubiFunctorCategoryEmbedding _ _).obj Y] ⟶ _).f n).f =
      Y.p.app (op [n]) ≫ (PInfty : K[Y.X] ⟶ _).f n := by
  -- We introduce P_infty endomorphisms P₁, P₂, P₃, P₄ on various objects Y₁, Y₂, Y₃, Y₄.
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    Y : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
    n : Nat
    ⊢ Eq (AlgebraicTopology.DoldKan.PInfty.f n).f (CategoryTheory.CategoryStruct.c …
  -/
  let Y₁ := (karoubiFunctorCategoryEmbedding _ _).obj Y
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    Y : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
    n : Nat
    Y₁ : CategoryTheory.Functor (Opposite SimplexCategory) (CategoryTheory.Idempot …
    ⊢ Eq (AlgebraicTopology.DoldKan.PInfty.f n).f (CategoryTheory.CategoryStruct.c …
  -/
  let Y₂ := Y.X
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    Y : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
    n : Nat
    Y₁ : CategoryTheory.Functor (Opposite SimplexCategory) (CategoryTheory.Idempot …
    Y₂ : CategoryTheory.SimplicialObject C := Y.X
    ⊢ Eq (AlgebraicTopology.DoldKan.PInfty.f n).f (CategoryTheory.CategoryStruct.c …
  -/
  let Y₃ := ((whiskering _ _).obj (toKaroubi C)).obj Y.X
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    Y : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
    n : Nat
    Y₁ : CategoryTheory.Functor (Opposite SimplexCategory) (CategoryTheory.Idempot …
    Y₂ : CategoryTheory.SimplicialObject C := Y.X
    Y₃ : CategoryTheory.SimplicialObject (CategoryTheory.Idempotents.Karoubi C) := …
    ⊢ Eq (AlgebraicTopology.DoldKan.PInfty.f n).f (CategoryTheory.CategoryStruct.c …
  -/
  let Y₄ := (karoubiFunctorCategoryEmbedding _ _).obj ((toKaroubi _).obj Y.X)
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    Y : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
    n : Nat
    Y₁ : CategoryTheory.Functor (Opposite SimplexCategory) (CategoryTheory.Idempot …
    Y₂ : CategoryTheory.SimplicialObject C := Y.X
    Y₃ : CategoryTheory.SimplicialObject (CategoryTheory.Idempotents.Karoubi C) := …
    Y₄ : CategoryTheory.Functor (Opposite SimplexCategory) (CategoryTheory.Idempot …
    ⊢ Eq (AlgebraicTopology.DoldKan.PInfty.f n).f (CategoryTheory.CategoryStruct.c …
  -/
  let P₁ : K[Y₁] ⟶ _ := PInfty
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    Y : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
    n : Nat
    Y₁ : CategoryTheory.Functor (Opposite SimplexCategory) (CategoryTheory.Idempot …
    Y₂ : CategoryTheory.SimplicialObject C := Y.X
    Y₃ : CategoryTheory.SimplicialObject (CategoryTheory.Idempotents.Karoubi C) := …
    Y₄ : CategoryTheory.Functor (Opposite SimplexCategory) (CategoryTheory.Idempot …
    P₁ : Quiver.Hom (AlgebraicTopology.AlternatingFaceMapComplex.obj Y₁) (Algebrai …
    ⊢ Eq (AlgebraicTopology.DoldKan.PInfty.f n).f (CategoryTheory.CategoryStruct.c …
  -/
  let P₂ : K[Y₂] ⟶ _ := PInfty
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    Y : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
    n : Nat
    Y₁ : CategoryTheory.Functor (Opposite SimplexCategory) (CategoryTheory.Idempot …
    Y₂ : CategoryTheory.SimplicialObject C := Y.X
    Y₃ : CategoryTheory.SimplicialObject (CategoryTheory.Idempotents.Karoubi C) := …
    Y₄ : CategoryTheory.Functor (Opposite SimplexCategory) (CategoryTheory.Idempot …
    P₁ : Quiver.Hom (AlgebraicTopology.AlternatingFaceMapComplex.obj Y₁) (Algebrai …
    P₂ : Quiver.Hom (AlgebraicTopology.AlternatingFaceMapComplex.obj Y₂) (Algebrai …
    ⊢ Eq (AlgebraicTopology.DoldKan.PInfty.f n).f (CategoryTheory.CategoryStruct.c …
  -/
  let P₃ : K[Y₃] ⟶ _ := PInfty
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    Y : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
    n : Nat
    Y₁ : CategoryTheory.Functor (Opposite SimplexCategory) (CategoryTheory.Idempot …
    Y₂ : CategoryTheory.SimplicialObject C := Y.X
    Y₃ : CategoryTheory.SimplicialObject (CategoryTheory.Idempotents.Karoubi C) := …
    Y₄ : CategoryTheory.Functor (Opposite SimplexCategory) (CategoryTheory.Idempot …
    P₁ : Quiver.Hom (AlgebraicTopology.AlternatingFaceMapComplex.obj Y₁) (Algebrai …
    P₂ : Quiver.Hom (AlgebraicTopology.AlternatingFaceMapComplex.obj Y₂) (Algebrai …
    P₃ : Quiver.Hom (AlgebraicTopology.AlternatingFaceMapComplex.obj Y₃) (Algebrai …
    ⊢ Eq (AlgebraicTopology.DoldKan.PInfty.f n).f (CategoryTheory.CategoryStruct.c …
  -/
  let P₄ : K[Y₄] ⟶ _ := PInfty
  -- The statement of lemma relates P₁ and P₂.
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    Y : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
    n : Nat
    Y₁ : CategoryTheory.Functor (Opposite SimplexCategory) (CategoryTheory.Idempot …
    Y₂ : CategoryTheory.SimplicialObject C := Y.X
    Y₃ : CategoryTheory.SimplicialObject (CategoryTheory.Idempotents.Karoubi C) := …
    Y₄ : CategoryTheory.Functor (Opposite SimplexCategory) (CategoryTheory.Idempot …
    P₁ : Quiver.Hom (AlgebraicTopology.AlternatingFaceMapComplex.obj Y₁) (Algebrai …
    P₂ : Quiver.Hom (AlgebraicTopology.AlternatingFaceMapComplex.obj Y₂) (Algebrai …
    P₃ : Quiver.Hom (AlgebraicTopology.AlternatingFaceMapComplex.obj Y₃) (Algebrai …
    P₄ : Quiver.Hom (AlgebraicTopology.AlternatingFaceMapComplex.obj Y₄) (Algebrai …
    ⊢ Eq (AlgebraicTopology.DoldKan.PInfty.f n).f (CategoryTheory.CategoryStruct.c …
  -/
  change (P₁.f n).f = Y.p.app (op [n]) ≫ P₂.f n
  -- The proof proceeds by obtaining relations h₃₂, h₄₃, h₁₄.
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    Y : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
    n : Nat
    Y₁ : CategoryTheory.Functor (Opposite SimplexCategory) (CategoryTheory.Idempot …
    Y₂ : CategoryTheory.SimplicialObject C := Y.X
    Y₃ : CategoryTheory.SimplicialObject (CategoryTheory.Idempotents.Karoubi C) := …
    Y₄ : CategoryTheory.Functor (Opposite SimplexCategory) (CategoryTheory.Idempot …
    P₁ : Quiver.Hom (AlgebraicTopology.AlternatingFaceMapComplex.obj Y₁) (Algebrai …
    P₂ : Quiver.Hom (AlgebraicTopology.AlternatingFaceMapComplex.obj Y₂) (Algebrai …
    P₃ : Quiver.Hom (AlgebraicTopology.AlternatingFaceMapComplex.obj Y₃) (Algebrai …
    P₄ : Quiver.Hom (AlgebraicTopology.AlternatingFaceMapComplex.obj Y₄) (Algebrai …
    ⊢ Eq (P₁.f n).f (CategoryTheory.CategoryStruct.comp (Y.p.app { unop := Simplex …
  -/
  have h₃₂ : (P₃.f n).f = P₂.f n := Karoubi.hom_ext_iff.mp (map_PInfty_f (toKaroubi C) Y₂ n)
  have h₄₃ : P₄.f n = P₃.f n := by
    have h := Functor.congr_obj (toKaroubi_comp_karoubiFunctorCategoryEmbedding _ _) Y₂
    simp only [P₃, P₄, ← natTransPInfty_f_app]
    congr 1
  have h₁₄ := Idempotents.natTrans_eq
    ((𝟙 (karoubiFunctorCategoryEmbedding SimplexCategoryᵒᵖ C)) ◫
      (natTransPInfty_f (Karoubi C) n)) Y
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    Y : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
    n : Nat
    Y₁ : CategoryTheory.Functor (Opposite SimplexCategory) (CategoryTheory.Idempot …
    Y₂ : CategoryTheory.SimplicialObject C := Y.X
    Y₃ : CategoryTheory.SimplicialObject (CategoryTheory.Idempotents.Karoubi C) := …
    Y₄ : CategoryTheory.Functor (Opposite SimplexCategory) (CategoryTheory.Idempot …
    P₁ : Quiver.Hom (AlgebraicTopology.AlternatingFaceMapComplex.obj Y₁) (Algebrai …
    P₂ : Quiver.Hom (AlgebraicTopology.AlternatingFaceMapComplex.obj Y₂) (Algebrai …
    P₃ : Quiver.Hom (AlgebraicTopology.AlternatingFaceMapComplex.obj Y₃) (Algebrai …
    P₄ : Quiver.Hom (AlgebraicTopology.AlternatingFaceMapComplex.obj Y₄) (Algebrai …
    h₃₂ : Eq (P₃.f n).f (P₂.f n)
    h₄₃ : Eq (P₄.f n) (P₃.f n)
    h₁₄ : Eq ((CategoryTheory.NatTrans.hcomp (CategoryTheory.CategoryStruct.id (Ca …
    ⊢ Eq (P₁.f n).f (CategoryTheory.CategoryStruct.comp (Y.p.app { unop := Simplex …
  -/
  dsimp [natTransPInfty_f] at h₁₄
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    Y : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
    n : Nat
    Y₁ : CategoryTheory.Functor (Opposite SimplexCategory) (CategoryTheory.Idempot …
    Y₂ : CategoryTheory.SimplicialObject C := Y.X
    Y₃ : CategoryTheory.SimplicialObject (CategoryTheory.Idempotents.Karoubi C) := …
    Y₄ : CategoryTheory.Functor (Opposite SimplexCategory) (CategoryTheory.Idempot …
    P₁ : Quiver.Hom (AlgebraicTopology.AlternatingFaceMapComplex.obj Y₁) (Algebrai …
    P₂ : Quiver.Hom (AlgebraicTopology.AlternatingFaceMapComplex.obj Y₂) (Algebrai …
    P₃ : Quiver.Hom (AlgebraicTopology.AlternatingFaceMapComplex.obj Y₃) (Algebrai …
    P₄ : Quiver.Hom (AlgebraicTopology.AlternatingFaceMapComplex.obj Y₄) (Algebrai …
    h₃₂ : Eq (P₃.f n).f (P₂.f n)
    h₄₃ : Eq (P₄.f n) (P₃.f n)
    h₁₄ : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.co …
    ⊢ Eq (P₁.f n).f (CategoryTheory.CategoryStruct.comp (Y.p.app { unop := Simplex …
  -/
  rw [id_comp, id_comp, comp_id, comp_id] at h₁₄
  -- We use the three equalities h₃₂, h₄₃, h₁₄.
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    Y : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
    n : Nat
    Y₁ : CategoryTheory.Functor (Opposite SimplexCategory) (CategoryTheory.Idempot …
    Y₂ : CategoryTheory.SimplicialObject C := Y.X
    Y₃ : CategoryTheory.SimplicialObject (CategoryTheory.Idempotents.Karoubi C) := …
    Y₄ : CategoryTheory.Functor (Opposite SimplexCategory) (CategoryTheory.Idempot …
    P₁ : Quiver.Hom (AlgebraicTopology.AlternatingFaceMapComplex.obj Y₁) (Algebrai …
    P₂ : Quiver.Hom (AlgebraicTopology.AlternatingFaceMapComplex.obj Y₂) (Algebrai …
    P₃ : Quiver.Hom (AlgebraicTopology.AlternatingFaceMapComplex.obj Y₃) (Algebrai …
    P₄ : Quiver.Hom (AlgebraicTopology.AlternatingFaceMapComplex.obj Y₄) (Algebrai …
    h₃₂ : Eq (P₃.f n).f (P₂.f n)
    h₄₃ : Eq (P₄.f n) (P₃.f n)
    h₁₄ : Eq (AlgebraicTopology.DoldKan.PInfty.f n) (CategoryTheory.CategoryStruct …
    ⊢ Eq (P₁.f n).f (CategoryTheory.CategoryStruct.comp (Y.p.app { unop := Simplex …
  -/
  rw [← h₃₂, ← h₄₃, h₁₄]
  simp only [KaroubiFunctorCategoryEmbedding.map_app_f, Karoubi.decompId_p_f,
    Karoubi.decompId_i_f, Karoubi.comp_f]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    Y : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
    n : Nat
    Y₁ : CategoryTheory.Functor (Opposite SimplexCategory) (CategoryTheory.Idempot …
    Y₂ : CategoryTheory.SimplicialObject C := Y.X
    Y₃ : CategoryTheory.SimplicialObject (CategoryTheory.Idempotents.Karoubi C) := …
    Y₄ : CategoryTheory.Functor (Opposite SimplexCategory) (CategoryTheory.Idempot …
    P₁ : Quiver.Hom (AlgebraicTopology.AlternatingFaceMapComplex.obj Y₁) (Algebrai …
    P₂ : Quiver.Hom (AlgebraicTopology.AlternatingFaceMapComplex.obj Y₂) (Algebrai …
    P₃ : Quiver.Hom (AlgebraicTopology.AlternatingFaceMapComplex.obj Y₃) (Algebrai …
    P₄ : Quiver.Hom (AlgebraicTopology.AlternatingFaceMapComplex.obj Y₄) (Algebrai …
    h₃₂ : Eq (P₃.f n).f (P₂.f n)
    h₄₃ : Eq (P₄.f n) (P₃.f n)
    h₁₄ : Eq (AlgebraicTopology.DoldKan.PInfty.f n) (CategoryTheory.CategoryStruct …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (Y.p.app { unop := SimplexCategory.mk …
  -/
  let π : Y₄ ⟶ Y₄ := (toKaroubi _ ⋙ karoubiFunctorCategoryEmbedding _ _).map Y.p
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    Y : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
    n : Nat
    Y₁ : CategoryTheory.Functor (Opposite SimplexCategory) (CategoryTheory.Idempot …
    Y₂ : CategoryTheory.SimplicialObject C := Y.X
    Y₃ : CategoryTheory.SimplicialObject (CategoryTheory.Idempotents.Karoubi C) := …
    Y₄ : CategoryTheory.Functor (Opposite SimplexCategory) (CategoryTheory.Idempot …
    P₁ : Quiver.Hom (AlgebraicTopology.AlternatingFaceMapComplex.obj Y₁) (Algebrai …
    P₂ : Quiver.Hom (AlgebraicTopology.AlternatingFaceMapComplex.obj Y₂) (Algebrai …
    P₃ : Quiver.Hom (AlgebraicTopology.AlternatingFaceMapComplex.obj Y₃) (Algebrai …
    P₄ : Quiver.Hom (AlgebraicTopology.AlternatingFaceMapComplex.obj Y₄) (Algebrai …
    h₃₂ : Eq (P₃.f n).f (P₂.f n)
    h₄₃ : Eq (P₄.f n) (P₃.f n)
    h₁₄ : Eq (AlgebraicTopology.DoldKan.PInfty.f n) (CategoryTheory.CategoryStruct …
    π : Quiver.Hom Y₄ Y₄ := ((CategoryTheory.Idempotents.toKaroubi (CategoryTheory …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (Y.p.app { unop := SimplexCategory.mk …
  -/
  have eq := Karoubi.hom_ext_iff.mp (PInfty_f_naturality n π)
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    Y : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
    n : Nat
    Y₁ : CategoryTheory.Functor (Opposite SimplexCategory) (CategoryTheory.Idempot …
    Y₂ : CategoryTheory.SimplicialObject C := Y.X
    Y₃ : CategoryTheory.SimplicialObject (CategoryTheory.Idempotents.Karoubi C) := …
    Y₄ : CategoryTheory.Functor (Opposite SimplexCategory) (CategoryTheory.Idempot …
    P₁ : Quiver.Hom (AlgebraicTopology.AlternatingFaceMapComplex.obj Y₁) (Algebrai …
    P₂ : Quiver.Hom (AlgebraicTopology.AlternatingFaceMapComplex.obj Y₂) (Algebrai …
    P₃ : Quiver.Hom (AlgebraicTopology.AlternatingFaceMapComplex.obj Y₃) (Algebrai …
    P₄ : Quiver.Hom (AlgebraicTopology.AlternatingFaceMapComplex.obj Y₄) (Algebrai …
    h₃₂ : Eq (P₃.f n).f (P₂.f n)
    h₄₃ : Eq (P₄.f n) (P₃.f n)
    h₁₄ : Eq (AlgebraicTopology.DoldKan.PInfty.f n) (CategoryTheory.CategoryStruct …
    π : Quiver.Hom Y₄ Y₄ := ((CategoryTheory.Idempotents.toKaroubi (CategoryTheory …
    eq : Eq (CategoryTheory.CategoryStruct.comp (π.app { unop := SimplexCategory.m …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (Y.p.app { unop := SimplexCategory.mk …
  -/
  simp only [Karoubi.comp_f] at eq
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    Y : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
    n : Nat
    Y₁ : CategoryTheory.Functor (Opposite SimplexCategory) (CategoryTheory.Idempot …
    Y₂ : CategoryTheory.SimplicialObject C := Y.X
    Y₃ : CategoryTheory.SimplicialObject (CategoryTheory.Idempotents.Karoubi C) := …
    Y₄ : CategoryTheory.Functor (Opposite SimplexCategory) (CategoryTheory.Idempot …
    P₁ : Quiver.Hom (AlgebraicTopology.AlternatingFaceMapComplex.obj Y₁) (Algebrai …
    P₂ : Quiver.Hom (AlgebraicTopology.AlternatingFaceMapComplex.obj Y₂) (Algebrai …
    P₃ : Quiver.Hom (AlgebraicTopology.AlternatingFaceMapComplex.obj Y₃) (Algebrai …
    P₄ : Quiver.Hom (AlgebraicTopology.AlternatingFaceMapComplex.obj Y₄) (Algebrai …
    h₃₂ : Eq (P₃.f n).f (P₂.f n)
    h₄₃ : Eq (P₄.f n) (P₃.f n)
    h₁₄ : Eq (AlgebraicTopology.DoldKan.PInfty.f n) (CategoryTheory.CategoryStruct …
    π : Quiver.Hom Y₄ Y₄ := ((CategoryTheory.Idempotents.toKaroubi (CategoryTheory …
    eq : Eq (CategoryTheory.CategoryStruct.comp (π.app { unop := SimplexCategory.m …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (Y.p.app { unop := SimplexCategory.mk …
  -/
  dsimp [π] at eq
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    Y : CategoryTheory.Idempotents.Karoubi (CategoryTheory.SimplicialObject C)
    n : Nat
    Y₁ : CategoryTheory.Functor (Opposite SimplexCategory) (CategoryTheory.Idempot …
    Y₂ : CategoryTheory.SimplicialObject C := Y.X
    Y₃ : CategoryTheory.SimplicialObject (CategoryTheory.Idempotents.Karoubi C) := …
    Y₄ : CategoryTheory.Functor (Opposite SimplexCategory) (CategoryTheory.Idempot …
    P₁ : Quiver.Hom (AlgebraicTopology.AlternatingFaceMapComplex.obj Y₁) (Algebrai …
    P₂ : Quiver.Hom (AlgebraicTopology.AlternatingFaceMapComplex.obj Y₂) (Algebrai …
    P₃ : Quiver.Hom (AlgebraicTopology.AlternatingFaceMapComplex.obj Y₃) (Algebrai …
    P₄ : Quiver.Hom (AlgebraicTopology.AlternatingFaceMapComplex.obj Y₄) (Algebrai …
    h₃₂ : Eq (P₃.f n).f (P₂.f n)
    h₄₃ : Eq (P₄.f n) (P₃.f n)
    h₁₄ : Eq (AlgebraicTopology.DoldKan.PInfty.f n) (CategoryTheory.CategoryStruct …
    π : Quiver.Hom Y₄ Y₄ := ((CategoryTheory.Idempotents.toKaroubi (CategoryTheory …
    eq : Eq (CategoryTheory.CategoryStruct.comp (Y.p.app { unop := SimplexCategory …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (Y.p.app { unop := SimplexCategory.mk …
  -/
  rw [← eq, app_idem_assoc Y (op [n])]
  /-
    🎉 no goals
  -/


