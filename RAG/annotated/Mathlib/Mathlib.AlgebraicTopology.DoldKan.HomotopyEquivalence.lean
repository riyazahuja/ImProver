/-- Inductive construction of homotopies from `P q` to `𝟙 _` -/
noncomputable def homotopyPToId : ∀ q : ℕ, Homotopy (P q : K[X] ⟶ _) (𝟙 _)
  | 0 => Homotopy.refl _
  | q + 1 => by
    refine
      Homotopy.trans (Homotopy.ofEq ?_)
        (Homotopy.trans
          (Homotopy.add (homotopyPToId q) (Homotopy.compLeft (homotopyHσToZero q) (P q)))
          (Homotopy.ofEq ?_))
      /-
        case refine_1
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.43, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        q : Nat
        ⊢ Eq (AlgebraicTopology.DoldKan.P (HAdd.hAdd q 1)) (HAdd.hAdd (AlgebraicTopolo …
      -/
    · simp only [P_succ, comp_add, comp_id]
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.43, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        q : Nat
        ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.id (AlgebraicTopology.Alternati …
      -/
    · simp only [add_zero, comp_zero]
      /-
        🎉 no goals
      -/


/-- The complement projection `Q q` to `P q` is homotopic to zero. -/
def homotopyQToZero (q : ℕ) : Homotopy (Q q : K[X] ⟶ _) 0 :=
  Homotopy.equivSubZero.toFun (homotopyPToId X q).symm


theorem homotopyPToId_eventually_constant {q n : ℕ} (hqn : n < q) :
    ((homotopyPToId X (q + 1)).hom n (n + 1) : X _[n] ⟶ X _[n + 1]) =
      (homotopyPToId X q).hom n (n + 1) := by
  simp only [homotopyHσToZero, AlternatingFaceMapComplex.obj_X, Nat.add_eq, Homotopy.trans_hom,
    Homotopy.ofEq_hom, Pi.zero_apply, Homotopy.add_hom, Homotopy.compLeft_hom, add_zero,
    Homotopy.nullHomotopy'_hom, ComplexShape.down_Rel, hσ'_eq_zero hqn (c_mk (n + 1) n rfl),
    dite_eq_ite, ite_self, comp_zero, zero_add, homotopyPToId]


/-- Construction of the homotopy from `PInfty` to the identity using eventually
(termwise) constant homotopies from `P q` to the identity for all `q` -/
@[simps]
def homotopyPInftyToId : Homotopy (PInfty : K[X] ⟶ _) (𝟙 _) where
  hom i j := (homotopyPToId X (j + 1)).hom i j
  zero i j hij := Homotopy.zero _ i j hij
  comm n := by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.13846, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      n : Nat
      ⊢ Eq (AlgebraicTopology.DoldKan.PInfty.f n) (HAdd.hAdd (HAdd.hAdd ((dNext n) f …
    -/
    rcases n with _|n
    · simpa only [Homotopy.dNext_zero_chainComplex, Homotopy.prevD_chainComplex,
        PInfty_f, P_f_0_eq, zero_add] using (homotopyPToId X 2).comm 0
    · simp only [Homotopy.dNext_succ_chainComplex, Homotopy.prevD_chainComplex,
        HomologicalComplex.id_f, PInfty_f, ← P_is_eventually_constant (le_refl <| n + 1)]
      -- Porting note(lean4/2146): remaining proof was
      -- `simpa only [homotopyPToId_eventually_constant X (lt_add_one (Nat.succ n))]
      -- using (homotopyPToId X (n + 2)).comm (n + 1)`;
      -- fails since leanprover/lean4:nightly-2023-05-16; `erw` below clunkily works around this.
      /-
        case succ
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.13846, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        n : Nat
        ⊢ Eq ((AlgebraicTopology.DoldKan.P (HAdd.hAdd (HAdd.hAdd n 1) 1)).f (HAdd.hAdd …
      -/
      erw [homotopyPToId_eventually_constant X (lt_add_one (Nat.succ n))]
      /-
        case succ
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.13846, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        n : Nat
        ⊢ Eq ((AlgebraicTopology.DoldKan.P (HAdd.hAdd (HAdd.hAdd n 1) 1)).f (HAdd.hAdd …
      -/
      have := (homotopyPToId X (n + 2)).comm (n + 1)
      /-
        case succ
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.13846, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        n : Nat
        this : Eq ((AlgebraicTopology.DoldKan.P (HAdd.hAdd n 2)).f (HAdd.hAdd n 1)) (H …
        ⊢ Eq ((AlgebraicTopology.DoldKan.P (HAdd.hAdd (HAdd.hAdd n 1) 1)).f (HAdd.hAdd …
      -/
      rw [Homotopy.dNext_succ_chainComplex, Homotopy.prevD_chainComplex] at this
      /-
        case succ
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.13846, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        n : Nat
        this : Eq ((AlgebraicTopology.DoldKan.P (HAdd.hAdd n 2)).f (HAdd.hAdd n 1)) (H …
        ⊢ Eq ((AlgebraicTopology.DoldKan.P (HAdd.hAdd (HAdd.hAdd n 1) 1)).f (HAdd.hAdd …
      -/
      exact this
      /-
        🎉 no goals
      -/


/-- The inclusion of the Moore complex in the alternating face map complex
is a homotopy equivalence -/
@[simps]
def homotopyEquivNormalizedMooreComplexAlternatingFaceMapComplex {A : Type*} [Category A]
    [Abelian A] {Y : SimplicialObject A} :
    HomotopyEquiv ((normalizedMooreComplex A).obj Y) ((alternatingFaceMapComplex A).obj Y) where
  hom := inclusionOfMooreComplexMap Y
  inv := PInftyToNormalizedMooreComplex Y
  homotopyHomInvId := Homotopy.ofEq (splitMonoInclusionOfMooreComplexMap Y).id
  homotopyInvHomId := Homotopy.trans
      (Homotopy.ofEq (PInftyToNormalizedMooreComplex_comp_inclusionOfMooreComplexMap Y))
      (homotopyPInftyToId Y)


