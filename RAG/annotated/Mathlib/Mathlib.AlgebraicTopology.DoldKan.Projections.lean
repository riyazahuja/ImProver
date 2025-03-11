/-- This is the inductive definition of the projections `P q : K[X] ⟶ K[X]`,
with `P 0 := 𝟙 _` and `P (q+1) := P q ≫ (𝟙 _ + Hσ q)`. -/
noncomputable def P : ℕ → (K[X] ⟶ K[X])
  | 0 => 𝟙 _
  | q + 1 => P q ≫ (𝟙 _ + Hσ q)

-- Porting note: `P_zero` and `P_succ` have been added to ease the port, because
-- `unfold P` would sometimes unfold to a `match` rather than the induction formula

lemma P_zero : (P 0 : K[X] ⟶ K[X]) = 𝟙 _ := rfl

lemma P_succ (q : ℕ) : (P (q+1) : K[X] ⟶ K[X]) = P q ≫ (𝟙 _ + Hσ q) := rfl


/-- All the `P q` coincide with `𝟙 _` in degree 0. -/
@[simp]
theorem P_f_0_eq (q : ℕ) : ((P q).f 0 : X _[0] ⟶ X _[0]) = 𝟙 _ := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    q : Nat
    ⊢ Eq ((AlgebraicTopology.DoldKan.P q).f 0) (CategoryTheory.CategoryStruct.id ( …
  -/
  induction' q with q hq
    /-
      case zero
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      ⊢ Eq ((AlgebraicTopology.DoldKan.P 0).f 0) (CategoryTheory.CategoryStruct.id ( …
    -/
  · rfl
    /-
      🎉 no goals
    -/
  · simp only [P_succ, HomologicalComplex.add_f_apply, HomologicalComplex.comp_f,
      HomologicalComplex.id_f, id_comp, hq, Hσ_eq_zero, add_zero]


/-- `Q q` is the complement projection associated to `P q` -/
def Q (q : ℕ) : K[X] ⟶ K[X] :=
  𝟙 _ - P q


theorem P_add_Q (q : ℕ) : P q + Q q = 𝟙 K[X] := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    q : Nat
    ⊢ Eq (HAdd.hAdd (AlgebraicTopology.DoldKan.P q) (AlgebraicTopology.DoldKan.Q q …
  -/
  rw [Q]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    q : Nat
    ⊢ Eq (HAdd.hAdd (AlgebraicTopology.DoldKan.P q) (HSub.hSub (CategoryTheory.Cat …
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/


theorem P_add_Q_f (q n : ℕ) : (P q).f n + (Q q).f n = 𝟙 (X _[n]) :=
  HomologicalComplex.congr_hom (P_add_Q q) n


@[simp]
theorem Q_zero : (Q 0 : K[X] ⟶ _) = 0 :=
  sub_self _


theorem Q_succ (q : ℕ) : (Q (q + 1) : K[X] ⟶ _) = Q q - P q ≫ Hσ q := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    q : Nat
    ⊢ Eq (AlgebraicTopology.DoldKan.Q (HAdd.hAdd q 1)) (HSub.hSub (AlgebraicTopolo …
  -/
  simp only [Q, P_succ, comp_add, comp_id]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    q : Nat
    ⊢ Eq (HSub.hSub (CategoryTheory.CategoryStruct.id (AlgebraicTopology.Alternati …
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/


/-- All the `Q q` coincide with `0` in degree 0. -/
@[simp]
theorem Q_f_0_eq (q : ℕ) : ((Q q).f 0 : X _[0] ⟶ X _[0]) = 0 := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    q : Nat
    ⊢ Eq ((AlgebraicTopology.DoldKan.Q q).f 0) 0
  -/
  simp only [HomologicalComplex.sub_f_apply, HomologicalComplex.id_f, Q, P_f_0_eq, sub_self]
  /-
    🎉 no goals
  -/


/-- This lemma expresses the vanishing of
`(P q).f (n+1) ≫ X.δ k : X _[n+1] ⟶ X _[n]` when `k≠0` and `k≥n-q+2` -/
theorem of_P : ∀ q n : ℕ, HigherFacesVanish q ((P q).f (n + 1) : X _[n + 1] ⟶ X _[n + 1])
                           /-
                             C : Type u_1
                             inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
                             inst✝ : CategoryTheory.Preadditive C
                             X : CategoryTheory.SimplicialObject C
                             n : Nat
                             j : Fin (HAdd.hAdd n 1)
                             hj₁ : LE.le (HAdd.hAdd n 1) (HAdd.hAdd (↑j) 0)
                             ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicTopology.DoldKan.P 0).f (H …
                           -/
  | 0 => fun n j hj₁ => by omega
                           /-
                             🎉 no goals
                           -/
  | q + 1 => fun n => by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      q n : Nat
      ⊢ AlgebraicTopology.DoldKan.HigherFacesVanish (HAdd.hAdd q 1) ((AlgebraicTopol …
    -/
    simp only [P_succ]
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      q n : Nat
      ⊢ AlgebraicTopology.DoldKan.HigherFacesVanish (HAdd.hAdd q 1) ((CategoryTheory …
    -/
    exact (of_P q n).induction
    /-
      🎉 no goals
    -/


@[reassoc]
theorem comp_P_eq_self {Y : C} {n q : ℕ} {φ : Y ⟶ X _[n + 1]} (v : HigherFacesVanish q φ) :
    φ ≫ (P q).f (n + 1) = φ := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    Y : C
    n q : Nat
    φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
    v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
    ⊢ Eq (CategoryTheory.CategoryStruct.comp φ ((AlgebraicTopology.DoldKan.P q).f  …
  -/
  induction' q with q hq
    /-
      case zero
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      Y : C
      n : Nat
      φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
      v : AlgebraicTopology.DoldKan.HigherFacesVanish 0 φ
      ⊢ Eq (CategoryTheory.CategoryStruct.comp φ ((AlgebraicTopology.DoldKan.P 0).f  …
    -/
  · simp only [P_zero]
    /-
      case zero
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      Y : C
      n : Nat
      φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
      v : AlgebraicTopology.DoldKan.HigherFacesVanish 0 φ
      ⊢ Eq (CategoryTheory.CategoryStruct.comp φ ((CategoryTheory.CategoryStruct.id  …
    -/
    apply comp_id
    /-
      🎉 no goals
    -/
  · simp only [P_succ, comp_add, HomologicalComplex.comp_f, HomologicalComplex.add_f_apply,
      comp_id, ← assoc, hq v.of_succ, add_right_eq_self]
    /-
      case succ
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      Y : C
      n : Nat
      φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
      q : Nat
      hq : AlgebraicTopology.DoldKan.HigherFacesVanish q φ → Eq (CategoryTheory.Cate …
      v : AlgebraicTopology.DoldKan.HigherFacesVanish (HAdd.hAdd q 1) φ
      ⊢ Eq (CategoryTheory.CategoryStruct.comp φ ((AlgebraicTopology.DoldKan.Hσ q).f …
    -/
    by_cases hqn : n < q
      /-
        case pos
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        Y : C
        n : Nat
        φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
        q : Nat
        hq : AlgebraicTopology.DoldKan.HigherFacesVanish q φ → Eq (CategoryTheory.Cate …
        v : AlgebraicTopology.DoldKan.HigherFacesVanish (HAdd.hAdd q 1) φ
        hqn : LT.lt n q
        ⊢ Eq (CategoryTheory.CategoryStruct.comp φ ((AlgebraicTopology.DoldKan.Hσ q).f …
      -/
    · exact v.of_succ.comp_Hσ_eq_zero hqn
      /-
        🎉 no goals
      -/
      /-
        case neg
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        Y : C
        n : Nat
        φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
        q : Nat
        hq : AlgebraicTopology.DoldKan.HigherFacesVanish q φ → Eq (CategoryTheory.Cate …
        v : AlgebraicTopology.DoldKan.HigherFacesVanish (HAdd.hAdd q 1) φ
        hqn : Not (LT.lt n q)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp φ ((AlgebraicTopology.DoldKan.Hσ q).f …
      -/
    · obtain ⟨a, ha⟩ := Nat.le.dest (not_lt.mp hqn)
      /-
        case neg.intro
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        Y : C
        n : Nat
        φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
        q : Nat
        hq : AlgebraicTopology.DoldKan.HigherFacesVanish q φ → Eq (CategoryTheory.Cate …
        v : AlgebraicTopology.DoldKan.HigherFacesVanish (HAdd.hAdd q 1) φ
        hqn : Not (LT.lt n q)
        a : Nat
        ha : Eq (HAdd.hAdd q a) n
        ⊢ Eq (CategoryTheory.CategoryStruct.comp φ ((AlgebraicTopology.DoldKan.Hσ q).f …
      -/
      have hnaq : n = a + q := by omega
      /-
        case neg.intro
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        Y : C
        n : Nat
        φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
        q : Nat
        hq : AlgebraicTopology.DoldKan.HigherFacesVanish q φ → Eq (CategoryTheory.Cate …
        v : AlgebraicTopology.DoldKan.HigherFacesVanish (HAdd.hAdd q 1) φ
        hqn : Not (LT.lt n q)
        a : Nat
        ha : Eq (HAdd.hAdd q a) n
        hnaq : Eq n (HAdd.hAdd a q)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp φ ((AlgebraicTopology.DoldKan.Hσ q).f …
      -/
      simp only [v.of_succ.comp_Hσ_eq hnaq, neg_eq_zero, ← assoc]
      have eq := v ⟨a, by omega⟩ (by
        simp only [hnaq, Nat.succ_eq_add_one, add_assoc]
        rfl)
      /-
        case neg.intro
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        Y : C
        n : Nat
        φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
        q : Nat
        hq : AlgebraicTopology.DoldKan.HigherFacesVanish q φ → Eq (CategoryTheory.Cate …
        v : AlgebraicTopology.DoldKan.HigherFacesVanish (HAdd.hAdd q 1) φ
        hqn : Not (LT.lt n q)
        a : Nat
        ha : Eq (HAdd.hAdd q a) n
        hnaq : Eq n (HAdd.hAdd a q)
        eq : Eq (CategoryTheory.CategoryStruct.comp φ (X.δ ⟨a, ⋯⟩.succ)) 0
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp φ …
      -/
      simp only [Fin.succ_mk] at eq
      /-
        case neg.intro
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        Y : C
        n : Nat
        φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
        q : Nat
        hq : AlgebraicTopology.DoldKan.HigherFacesVanish q φ → Eq (CategoryTheory.Cate …
        v : AlgebraicTopology.DoldKan.HigherFacesVanish (HAdd.hAdd q 1) φ
        hqn : Not (LT.lt n q)
        a : Nat
        ha : Eq (HAdd.hAdd q a) n
        hnaq : Eq n (HAdd.hAdd a q)
        eq : Eq (CategoryTheory.CategoryStruct.comp φ (X.δ ⟨HAdd.hAdd a 1, ⋯⟩)) 0
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp φ …
      -/
      simp only [eq, zero_comp]
      /-
        🎉 no goals
      -/


theorem comp_P_eq_self_iff {Y : C} {n q : ℕ} {φ : Y ⟶ X _[n + 1]} :
    φ ≫ (P q).f (n + 1) = φ ↔ HigherFacesVanish q φ := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    Y : C
    n q : Nat
    φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
    ⊢ Iff (Eq (CategoryTheory.CategoryStruct.comp φ ((AlgebraicTopology.DoldKan.P  …
  -/
  constructor
    /-
      case mp
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      Y : C
      n q : Nat
      φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
      ⊢ Eq (CategoryTheory.CategoryStruct.comp φ ((AlgebraicTopology.DoldKan.P q).f  …
    -/
  · intro hφ
    /-
      case mp
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      Y : C
      n q : Nat
      φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
      hφ : Eq (CategoryTheory.CategoryStruct.comp φ ((AlgebraicTopology.DoldKan.P q) …
      ⊢ AlgebraicTopology.DoldKan.HigherFacesVanish q φ
    -/
    rw [← hφ]
    /-
      case mp
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      Y : C
      n q : Nat
      φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
      hφ : Eq (CategoryTheory.CategoryStruct.comp φ ((AlgebraicTopology.DoldKan.P q) …
      ⊢ AlgebraicTopology.DoldKan.HigherFacesVanish q (CategoryTheory.CategoryStruct …
    -/
    apply HigherFacesVanish.of_comp
    /-
      case mp.v
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      Y : C
      n q : Nat
      φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
      hφ : Eq (CategoryTheory.CategoryStruct.comp φ ((AlgebraicTopology.DoldKan.P q) …
      ⊢ AlgebraicTopology.DoldKan.HigherFacesVanish q ((AlgebraicTopology.DoldKan.P  …
    -/
    apply HigherFacesVanish.of_P
    /-
      🎉 no goals
    -/
    /-
      case mpr
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      Y : C
      n q : Nat
      φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
      ⊢ AlgebraicTopology.DoldKan.HigherFacesVanish q φ → Eq (CategoryTheory.Categor …
    -/
  · exact HigherFacesVanish.comp_P_eq_self
    /-
      🎉 no goals
    -/


@[reassoc (attr := simp)]
theorem P_f_idem (q n : ℕ) : ((P q).f n : X _[n] ⟶ _) ≫ (P q).f n = (P q).f n := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    q n : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicTopology.DoldKan.P q).f n) …
  -/
  rcases n with (_|n)
    /-
      case zero
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      q : Nat
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicTopology.DoldKan.P q).f 0) …
    -/
  · rw [P_f_0_eq q, comp_id]
    /-
      🎉 no goals
    -/
    /-
      case succ
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      q n : Nat
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicTopology.DoldKan.P q).f (H …
    -/
  · exact (HigherFacesVanish.of_P q n).comp_P_eq_self
    /-
      🎉 no goals
    -/


@[reassoc (attr := simp)]
theorem Q_f_idem (q n : ℕ) : ((Q q).f n : X _[n] ⟶ _) ≫ (Q q).f n = (Q q).f n :=
  idem_of_id_sub_idem _ (P_f_idem q n)


@[reassoc (attr := simp)]
theorem P_idem (q : ℕ) : (P q : K[X] ⟶ K[X]) ≫ P q = P q := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    q : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.P q) (Alge …
  -/
  ext n
  /-
    case h
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    q n : Nat
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.P q) (Alg …
  -/
  exact P_f_idem q n
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem Q_idem (q : ℕ) : (Q q : K[X] ⟶ K[X]) ≫ Q q = Q q := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    q : Nat
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.Q q) (Alge …
  -/
  ext n
  /-
    case h
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    q n : Nat
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.Q q) (Alg …
  -/
  exact Q_f_idem q n
  /-
    🎉 no goals
  -/


/-- For each `q`, `P q` is a natural transformation. -/
@[simps]
def natTransP (q : ℕ) : alternatingFaceMapComplex C ⟶ alternatingFaceMapComplex C where
  app _ := P q
  naturality _ _ f := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.72576, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      q : Nat
      x✝¹ x✝ : CategoryTheory.SimplicialObject C
      f : Quiver.Hom x✝¹ x✝
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicTopology.alternatingFaceMa …
    -/
    induction' q with q hq
      /-
        case zero
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.72576, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X x✝¹ x✝ : CategoryTheory.SimplicialObject C
        f : Quiver.Hom x✝¹ x✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicTopology.alternatingFaceMa …
      -/
    · dsimp [alternatingFaceMapComplex]
      /-
        case zero
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.72576, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X x✝¹ x✝ : CategoryTheory.SimplicialObject C
        f : Quiver.Hom x✝¹ x✝
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.AlternatingFaceMap …
      -/
      simp only [P_zero, id_comp, comp_id]
      /-
        🎉 no goals
      -/
      /-
        case succ
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.72576, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X x✝¹ x✝ : CategoryTheory.SimplicialObject C
        f : Quiver.Hom x✝¹ x✝
        q : Nat
        hq : Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicTopology.alternatingFac …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicTopology.alternatingFaceMa …
      -/
    · simp only [P_succ, add_comp, comp_add, assoc, comp_id, hq, reassoc_of% hq]
      /-
        case succ
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.72576, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X x✝¹ x✝ : CategoryTheory.SimplicialObject C
        f : Quiver.Hom x✝¹ x✝
        q : Nat
        hq : Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicTopology.alternatingFac …
        ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan …
      -/
      erw [(natTransHσ q).naturality f]
      /-
        case succ
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.72576, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X x✝¹ x✝ : CategoryTheory.SimplicialObject C
        f : Quiver.Hom x✝¹ x✝
        q : Nat
        hq : Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicTopology.alternatingFac …
        ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan …
      -/
      rfl
      /-
        🎉 no goals
      -/


@[reassoc (attr := simp)]
theorem P_f_naturality (q n : ℕ) {X Y : SimplicialObject C} (f : X ⟶ Y) :
    f.app (op [n]) ≫ (P q).f n = (P q).f n ≫ f.app (op [n]) :=
  HomologicalComplex.congr_hom ((natTransP q).naturality f) n


@[reassoc (attr := simp)]
theorem Q_f_naturality (q n : ℕ) {X Y : SimplicialObject C} (f : X ⟶ Y) :
    f.app (op [n]) ≫ (Q q).f n = (Q q).f n ≫ f.app (op [n]) := by
  simp only [Q, HomologicalComplex.sub_f_apply, HomologicalComplex.id_f, comp_sub, P_f_naturality,
    sub_comp, sub_left_inj]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    q n : Nat
    X Y : CategoryTheory.SimplicialObject C
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.app { unop := SimplexCategory.mk n …
  -/
  dsimp
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    q n : Nat
    X Y : CategoryTheory.SimplicialObject C
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (f.app { unop := SimplexCategory.mk n …
  -/
  simp only [comp_id, id_comp]
  /-
    🎉 no goals
  -/


/-- For each `q`, `Q q` is a natural transformation. -/
@[simps]
def natTransQ (q : ℕ) : alternatingFaceMapComplex C ⟶ alternatingFaceMapComplex C where
  app _ := Q q


theorem map_P {D : Type*} [Category D] [Preadditive D] (G : C ⥤ D) [G.Additive]
    (X : SimplicialObject C) (q n : ℕ) :
    G.map ((P q : K[X] ⟶ _).f n) = (P q : K[((whiskering C D).obj G).obj X] ⟶ _).f n := by
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
    ⊢ Eq (G.map ((AlgebraicTopology.DoldKan.P q).f n)) ((AlgebraicTopology.DoldKan …
  -/
  induction' q with q hq
    /-
      case zero
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
      ⊢ Eq (G.map ((AlgebraicTopology.DoldKan.P 0).f n)) ((AlgebraicTopology.DoldKan …
    -/
  · simp only [P_zero]
    /-
      case zero
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
      ⊢ Eq (G.map ((CategoryTheory.CategoryStruct.id (AlgebraicTopology.AlternatingF …
    -/
    apply G.map_id
    /-
      🎉 no goals
    -/
  · simp only [P_succ, comp_add, HomologicalComplex.comp_f, HomologicalComplex.add_f_apply,
      comp_id, Functor.map_add, Functor.map_comp, hq, map_Hσ]


theorem map_Q {D : Type*} [Category D] [Preadditive D] (G : C ⥤ D) [G.Additive]
    (X : SimplicialObject C) (q n : ℕ) :
    G.map ((Q q : K[X] ⟶ _).f n) = (Q q : K[((whiskering C D).obj G).obj X] ⟶ _).f n := by
  rw [← add_right_inj (G.map ((P q : K[X] ⟶ _).f n)), ← G.map_add, map_P G X q n, P_add_Q_f,
    P_add_Q_f]
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
    ⊢ Eq (G.map (CategoryTheory.CategoryStruct.id (X.obj { unop := SimplexCategory …
  -/
  apply G.map_id
  /-
    🎉 no goals
  -/


