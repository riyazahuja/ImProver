/-- As we are using chain complexes indexed by `ℕ`, we shall need the relation
`c` such `c m n` if and only if `n+1=m`. -/
abbrev c :=
  ComplexShape.down ℕ


/-- Helper when we need some `c.rel i j` (i.e. `ComplexShape.down ℕ`),
e.g. `c_mk n (n+1) rfl` -/
theorem c_mk (i j : ℕ) (h : j + 1 = i) : c.Rel i j :=
  ComplexShape.down_mk i j h


/-- This lemma is meant to be used with `nullHomotopicMap'_f_of_not_rel_left` -/
theorem cs_down_0_not_rel_left (j : ℕ) : ¬c.Rel 0 j := by
  /-
    j : Nat
    ⊢ Not (AlgebraicTopology.DoldKan.c.Rel 0 j)
  -/
  intro hj
  /-
    j : Nat
    hj : AlgebraicTopology.DoldKan.c.Rel 0 j
    ⊢ False
  -/
  dsimp at hj
  /-
    j : Nat
    hj : Eq (HAdd.hAdd j 1) 0
    ⊢ False
  -/
  apply Nat.not_succ_le_zero j
  /-
    j : Nat
    hj : Eq (HAdd.hAdd j 1) 0
    ⊢ LE.le j.succ 0
  -/
  rw [Nat.succ_eq_add_one, hj]
  /-
    🎉 no goals
  -/


/-- The sequence of maps which gives the null homotopic maps `Hσ` that shall be in
the inductive construction of the projections `P q : K[X] ⟶ K[X]` -/
def hσ (q : ℕ) (n : ℕ) : X _[n] ⟶ X _[n + 1] :=
  if n < q then 0 else (-1 : ℤ) ^ (n - q) • X.σ ⟨n - q, Nat.lt_succ_of_le (Nat.sub_le _ _)⟩


/-- We can turn `hσ` into a datum that can be passed to `nullHomotopicMap'`. -/
def hσ' (q : ℕ) : ∀ n m, c.Rel m n → (K[X].X n ⟶ K[X].X m) := fun n m hnm =>
                       /-
                         C : Type u_1
                         inst✝¹ : CategoryTheory.Category.{?u.4760, u_1} C
                         inst✝ : CategoryTheory.Preadditive C
                         X : CategoryTheory.SimplicialObject C
                         q n m : Nat
                         hnm : AlgebraicTopology.DoldKan.c.Rel m n
                         ⊢ Eq (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) }) ((AlgebraicTopolog …
                       -/
  hσ q n ≫ eqToHom (by congr)
                       /-
                         🎉 no goals
                       -/


theorem hσ'_eq_zero {q n m : ℕ} (hnq : n < q) (hnm : c.Rel m n) :
    (hσ' q n m hnm : X _[n] ⟶ X _[m]) = 0 := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    q n m : Nat
    hnq : LT.lt n q
    hnm : AlgebraicTopology.DoldKan.c.Rel m n
    ⊢ Eq (AlgebraicTopology.DoldKan.hσ' q n m hnm) 0
  -/
  simp only [hσ', hσ]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    q n m : Nat
    hnq : LT.lt n q
    hnm : AlgebraicTopology.DoldKan.c.Rel m n
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (ite (LT.lt n q) 0 (HSMul.hSMul (HPow …
  -/
  split_ifs
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    q n m : Nat
    hnq : LT.lt n q
    hnm : AlgebraicTopology.DoldKan.c.Rel m n
    ⊢ Eq (CategoryTheory.CategoryStruct.comp 0 (CategoryTheory.eqToHom ⋯)) 0
  -/
  exact zero_comp
  /-
    🎉 no goals
  -/


theorem hσ'_eq {q n a m : ℕ} (ha : n = a + q) (hnm : c.Rel m n) :
    (hσ' q n m hnm : X _[n] ⟶ X _[m]) =
      ((-1 : ℤ) ^ a • X.σ ⟨a, Nat.lt_succ_iff.mpr (Nat.le.intro (Eq.symm ha))⟩) ≫
                    /-
                      C : Type u_1
                      inst✝¹ : CategoryTheory.Category.{?u.10716, u_1} C
                      inst✝ : CategoryTheory.Preadditive C
                      X : CategoryTheory.SimplicialObject C
                      q n a m : Nat
                      ha : Eq n (HAdd.hAdd a q)
                      hnm : AlgebraicTopology.DoldKan.c.Rel m n
                      ⊢ Eq (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) }) ((AlgebraicTopolog …
                    -/
        eqToHom (by congr) := by
                    /-
                      🎉 no goals
                    -/
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    q n a m : Nat
    ha : Eq n (HAdd.hAdd a q)
    hnm : AlgebraicTopology.DoldKan.c.Rel m n
    ⊢ Eq (AlgebraicTopology.DoldKan.hσ' q n m hnm) (CategoryTheory.CategoryStruct. …
  -/
  simp only [hσ', hσ]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    q n a m : Nat
    ha : Eq n (HAdd.hAdd a q)
    hnm : AlgebraicTopology.DoldKan.c.Rel m n
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (ite (LT.lt n q) 0 (HSMul.hSMul (HPow …
  -/
  split_ifs
    /-
      case pos
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      q n a m : Nat
      ha : Eq n (HAdd.hAdd a q)
      hnm : AlgebraicTopology.DoldKan.c.Rel m n
      h✝ : LT.lt n q
      ⊢ Eq (CategoryTheory.CategoryStruct.comp 0 (CategoryTheory.eqToHom ⋯)) (Catego …
    -/
  · omega
    /-
      🎉 no goals
    -/
    /-
      case neg
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      q n a m : Nat
      ha : Eq n (HAdd.hAdd a q)
      hnm : AlgebraicTopology.DoldKan.c.Rel m n
      h✝ : Not (LT.lt n q)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSMul.hSMul (HPow.hPow (-1) (HSub.hS …
    -/
  · have h' := tsub_eq_of_eq_add ha
    /-
      case neg
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      q n a m : Nat
      ha : Eq n (HAdd.hAdd a q)
      hnm : AlgebraicTopology.DoldKan.c.Rel m n
      h✝ : Not (LT.lt n q)
      h' : Eq (HSub.hSub n q) a
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSMul.hSMul (HPow.hPow (-1) (HSub.hS …
    -/
    congr
    /-
      🎉 no goals
    -/


theorem hσ'_eq' {q n a : ℕ} (ha : n = a + q) :
    (hσ' q n (n + 1) rfl : X _[n] ⟶ X _[n + 1]) =
      (-1 : ℤ) ^ a • X.σ ⟨a, Nat.lt_succ_iff.mpr (Nat.le.intro (Eq.symm ha))⟩ := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    q n a : Nat
    ha : Eq n (HAdd.hAdd a q)
    ⊢ Eq (AlgebraicTopology.DoldKan.hσ' q n (HAdd.hAdd n 1) ⋯) (HSMul.hSMul (HPow. …
  -/
  rw [hσ'_eq ha rfl, eqToHom_refl, comp_id]
  /-
    🎉 no goals
  -/


/-- The null homotopic map $(hσ q) ∘ d + d ∘ (hσ q)$ -/
def Hσ (q : ℕ) : K[X] ⟶ K[X] :=
  nullHomotopicMap' (hσ' q)


/-- `Hσ` is null homotopic -/
def homotopyHσToZero (q : ℕ) : Homotopy (Hσ q : K[X] ⟶ K[X]) 0 :=
  nullHomotopy' (hσ' q)


/-- In degree `0`, the null homotopic map `Hσ` is zero. -/
theorem Hσ_eq_zero (q : ℕ) : (Hσ q : K[X] ⟶ K[X]).f 0 = 0 := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    q : Nat
    ⊢ Eq ((AlgebraicTopology.DoldKan.Hσ q).f 0) 0
  -/
  unfold Hσ
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    q : Nat
    ⊢ Eq ((Homotopy.nullHomotopicMap' (AlgebraicTopology.DoldKan.hσ' q)).f 0) 0
  -/
  rw [nullHomotopicMap'_f_of_not_rel_left (c_mk 1 0 rfl) cs_down_0_not_rel_left]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    q : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.hσ' q 0 1  …
  -/
  rcases q with (_|q)
    /-
      case zero
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.hσ' 0 0 1  …
    -/
  · rw [hσ'_eq (show 0 = 0 + 0 by rfl) (c_mk 1 0 rfl)]
    /-
      case zero
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp only [pow_zero, Fin.mk_zero, one_zsmul, eqToHom_refl, Category.comp_id]
    /-
      case zero
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.σ 0) ((AlgebraicTopology.Alternati …
    -/
    erw [ChainComplex.of_d]
    rw [AlternatingFaceMapComplex.objD, Fin.sum_univ_two, Fin.val_zero, Fin.val_one, pow_zero,
      pow_one, one_smul, neg_smul, one_smul, comp_add, comp_neg, add_neg_eq_zero]
    /-
      case zero
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.σ 0) (X.δ 0)) (CategoryTheory.Cate …
    -/
    erw [δ_comp_σ_self, δ_comp_σ_succ]
    /-
      🎉 no goals
    -/
    /-
      case succ
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      q : Nat
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.hσ' (HAdd. …
    -/
  · rw [hσ'_eq_zero (Nat.succ_pos q) (c_mk 1 0 rfl), zero_comp]
    /-
      🎉 no goals
    -/


/-- The maps `hσ' q n m hnm` are natural on the simplicial object -/
theorem hσ'_naturality (q : ℕ) (n m : ℕ) (hnm : c.Rel m n) {X Y : SimplicialObject C} (f : X ⟶ Y) :
    f.app (op [n]) ≫ hσ' q n m hnm = hσ' q n m hnm ≫ f.app (op [m]) := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    q n m : Nat
    hnm : AlgebraicTopology.DoldKan.c.Rel m n
    X Y : CategoryTheory.SimplicialObject C
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.app { unop := SimplexCategory.mk n …
  -/
  have h : n + 1 = m := hnm
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    q n m : Nat
    hnm : AlgebraicTopology.DoldKan.c.Rel m n
    X Y : CategoryTheory.SimplicialObject C
    f : Quiver.Hom X Y
    h : Eq (HAdd.hAdd n 1) m
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.app { unop := SimplexCategory.mk n …
  -/
  subst h
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    q n : Nat
    X Y : CategoryTheory.SimplicialObject C
    f : Quiver.Hom X Y
    hnm : AlgebraicTopology.DoldKan.c.Rel (HAdd.hAdd n 1) n
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.app { unop := SimplexCategory.mk n …
  -/
  simp only [hσ', eqToHom_refl, comp_id]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    q n : Nat
    X Y : CategoryTheory.SimplicialObject C
    f : Quiver.Hom X Y
    hnm : AlgebraicTopology.DoldKan.c.Rel (HAdd.hAdd n 1) n
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.app { unop := SimplexCategory.mk n …
  -/
  unfold hσ
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    q n : Nat
    X Y : CategoryTheory.SimplicialObject C
    f : Quiver.Hom X Y
    hnm : AlgebraicTopology.DoldKan.c.Rel (HAdd.hAdd n 1) n
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.app { unop := SimplexCategory.mk n …
  -/
  split_ifs
    /-
      case pos
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      q n : Nat
      X Y : CategoryTheory.SimplicialObject C
      f : Quiver.Hom X Y
      hnm : AlgebraicTopology.DoldKan.c.Rel (HAdd.hAdd n 1) n
      h✝ : LT.lt n q
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.app { unop := SimplexCategory.mk n …
    -/
  · rw [zero_comp, comp_zero]
    /-
      🎉 no goals
    -/
    /-
      case neg
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      q n : Nat
      X Y : CategoryTheory.SimplicialObject C
      f : Quiver.Hom X Y
      hnm : AlgebraicTopology.DoldKan.c.Rel (HAdd.hAdd n 1) n
      h✝ : Not (LT.lt n q)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.app { unop := SimplexCategory.mk n …
    -/
  · simp only [zsmul_comp, comp_zsmul]
    /-
      case neg
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      q n : Nat
      X Y : CategoryTheory.SimplicialObject C
      f : Quiver.Hom X Y
      hnm : AlgebraicTopology.DoldKan.c.Rel (HAdd.hAdd n 1) n
      h✝ : Not (LT.lt n q)
      ⊢ Eq (HSMul.hSMul (HPow.hPow (-1) (HSub.hSub n q)) (CategoryTheory.CategoryStr …
    -/
    erw [f.naturality]
    /-
      case neg
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      q n : Nat
      X Y : CategoryTheory.SimplicialObject C
      f : Quiver.Hom X Y
      hnm : AlgebraicTopology.DoldKan.c.Rel (HAdd.hAdd n 1) n
      h✝ : Not (LT.lt n q)
      ⊢ Eq (HSMul.hSMul (HPow.hPow (-1) (HSub.hSub n q)) (CategoryTheory.CategoryStr …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- For each q, `Hσ q` is a natural transformation. -/
def natTransHσ (q : ℕ) : alternatingFaceMapComplex C ⟶ alternatingFaceMapComplex C where
  app _ := Hσ q
  naturality _ _ f := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.42208, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      q : Nat
      x✝¹ x✝ : CategoryTheory.SimplicialObject C
      f : Quiver.Hom x✝¹ x✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicTopology.alternatingFaceMa …
    -/
    unfold Hσ
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.42208, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      q : Nat
      x✝¹ x✝ : CategoryTheory.SimplicialObject C
      f : Quiver.Hom x✝¹ x✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicTopology.alternatingFaceMa …
    -/
    rw [nullHomotopicMap'_comp, comp_nullHomotopicMap']
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.42208, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      q : Nat
      x✝¹ x✝ : CategoryTheory.SimplicialObject C
      f : Quiver.Hom x✝¹ x✝
      ⊢ Eq (Homotopy.nullHomotopicMap' fun i j hij => CategoryTheory.CategoryStruct. …
    -/
    congr
    /-
      case e_h
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.42208, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      q : Nat
      x✝¹ x✝ : CategoryTheory.SimplicialObject C
      f : Quiver.Hom x✝¹ x✝
      ⊢ Eq (fun i j hij => CategoryTheory.CategoryStruct.comp (((AlgebraicTopology.a …
    -/
    ext n m hnm
    /-
      case e_h.h.h.h
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.42208, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      q : Nat
      x✝¹ x✝ : CategoryTheory.SimplicialObject C
      f : Quiver.Hom x✝¹ x✝
      n m : Nat
      hnm : (ComplexShape.down Nat).Rel m n
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (((AlgebraicTopology.alternatingFaceM …
    -/
    simp only [alternatingFaceMapComplex_map_f, hσ'_naturality]
    /-
      🎉 no goals
    -/


/-- The maps `hσ' q n m hnm` are compatible with the application of additive functors. -/
theorem map_hσ' {D : Type*} [Category D] [Preadditive D] (G : C ⥤ D) [G.Additive]
    (X : SimplicialObject C) (q n m : ℕ) (hnm : c.Rel m n) :
    (hσ' q n m hnm : K[((whiskering _ _).obj G).obj X].X n ⟶ _) =
      G.map (hσ' q n m hnm : K[X].X n ⟶ _) := by
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
    q n m : Nat
    hnm : AlgebraicTopology.DoldKan.c.Rel m n
    ⊢ Eq (AlgebraicTopology.DoldKan.hσ' q n m hnm) (G.map (AlgebraicTopology.DoldK …
  -/
  unfold hσ' hσ
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
    q n m : Nat
    hnm : AlgebraicTopology.DoldKan.c.Rel m n
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (ite (LT.lt n q) 0 (HSMul.hSMul (HPow …
  -/
  split_ifs
    /-
      case pos
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
      inst✝³ : CategoryTheory.Preadditive C
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_2} D
      inst✝¹ : CategoryTheory.Preadditive D
      G : CategoryTheory.Functor C D
      inst✝ : G.Additive
      X : CategoryTheory.SimplicialObject C
      q n m : Nat
      hnm : AlgebraicTopology.DoldKan.c.Rel m n
      h✝ : LT.lt n q
      ⊢ Eq (CategoryTheory.CategoryStruct.comp 0 (CategoryTheory.eqToHom ⋯)) (G.map  …
    -/
  · simp only [Functor.map_zero, zero_comp]
    /-
      🎉 no goals
    -/
    /-
      case neg
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
      inst✝³ : CategoryTheory.Preadditive C
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_2} D
      inst✝¹ : CategoryTheory.Preadditive D
      G : CategoryTheory.Functor C D
      inst✝ : G.Additive
      X : CategoryTheory.SimplicialObject C
      q n m : Nat
      hnm : AlgebraicTopology.DoldKan.c.Rel m n
      h✝ : Not (LT.lt n q)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSMul.hSMul (HPow.hPow (-1) (HSub.hS …
    -/
  · simp only [eqToHom_map, Functor.map_comp, Functor.map_zsmul]
    /-
      case neg
      C : Type u_1
      inst✝⁴ : CategoryTheory.Category.{u_4, u_1} C
      inst✝³ : CategoryTheory.Preadditive C
      D : Type u_2
      inst✝² : CategoryTheory.Category.{u_3, u_2} D
      inst✝¹ : CategoryTheory.Preadditive D
      G : CategoryTheory.Functor C D
      inst✝ : G.Additive
      X : CategoryTheory.SimplicialObject C
      q n m : Nat
      hnm : AlgebraicTopology.DoldKan.c.Rel m n
      h✝ : Not (LT.lt n q)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (HSMul.hSMul (HPow.hPow (-1) (HSub.hS …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- The null homotopic maps `Hσ` are compatible with the application of additive functors. -/
theorem map_Hσ {D : Type*} [Category D] [Preadditive D] (G : C ⥤ D) [G.Additive]
    (X : SimplicialObject C) (q n : ℕ) :
    (Hσ q : K[((whiskering C D).obj G).obj X] ⟶ _).f n = G.map ((Hσ q : K[X] ⟶ _).f n) := by
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
    q n : Nat
    ⊢ Eq ((AlgebraicTopology.DoldKan.Hσ q).f n) (G.map ((AlgebraicTopology.DoldKan …
  -/
  unfold Hσ
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
    q n : Nat
    ⊢ Eq ((Homotopy.nullHomotopicMap' (AlgebraicTopology.DoldKan.hσ' q)).f n) (G.m …
  -/
  have eq := HomologicalComplex.congr_hom (map_nullHomotopicMap' G (@hσ' _ _ _ X q)) n
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
    q n : Nat
    eq : Eq (((G.mapHomologicalComplex AlgebraicTopology.DoldKan.c).map (Homotopy. …
    ⊢ Eq ((Homotopy.nullHomotopicMap' (AlgebraicTopology.DoldKan.hσ' q)).f n) (G.m …
  -/
  simp only [Functor.mapHomologicalComplex_map_f, ← map_hσ'] at eq
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
    q n : Nat
    eq : Eq (G.map ((Homotopy.nullHomotopicMap' (AlgebraicTopology.DoldKan.hσ' q)) …
    ⊢ Eq ((Homotopy.nullHomotopicMap' (AlgebraicTopology.DoldKan.hσ' q)).f n) (G.m …
  -/
  rw [eq]
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
    q n : Nat
    eq : Eq (G.map ((Homotopy.nullHomotopicMap' (AlgebraicTopology.DoldKan.hσ' q)) …
    ⊢ Eq ((Homotopy.nullHomotopicMap' (AlgebraicTopology.DoldKan.hσ' q)).f n) ((Ho …
  -/
  let h := (Functor.congr_obj (map_alternatingFaceMapComplex G) X).symm
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
    q n : Nat
    eq : Eq (G.map ((Homotopy.nullHomotopicMap' (AlgebraicTopology.DoldKan.hσ' q)) …
    h : Eq ((((CategoryTheory.SimplicialObject.whiskering C D).obj G).comp (Algebr …
    ⊢ Eq ((Homotopy.nullHomotopicMap' (AlgebraicTopology.DoldKan.hσ' q)).f n) ((Ho …
  -/
  congr
  /-
    🎉 no goals
  -/


