/-- In each positive degree, this lemma decomposes the idempotent endomorphism
`Q q` as a sum of morphisms which are postcompositions with suitable degeneracies.
As `Q q` is the complement projection to `P q`, this implies that in the case of
simplicial abelian groups, any $(n+1)$-simplex $x$ can be decomposed as
$x = x' + \sum (i=0}^{q-1} σ_{n-i}(y_i)$ where $x'$ is in the image of `P q` and
the $y_i$ are in degree $n$. -/
theorem decomposition_Q (n q : ℕ) :
    ((Q q).f (n + 1) : X _[n + 1] ⟶ X _[n + 1]) =
      ∑ i ∈ Finset.filter (fun i : Fin (n + 1) => (i : ℕ) < q) Finset.univ,
        (P i).f (n + 1) ≫ X.δ i.rev.succ ≫ X.σ (Fin.rev i) := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    n q : Nat
    ⊢ Eq ((AlgebraicTopology.DoldKan.Q q).f (HAdd.hAdd n 1)) ((Finset.filter (fun  …
  -/
  induction' q with q hq
  · simp only [Q_zero, HomologicalComplex.zero_f_apply, Nat.not_lt_zero,
      Finset.filter_False, Finset.sum_empty]
    /-
      case succ
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      n q : Nat
      hq : Eq ((AlgebraicTopology.DoldKan.Q q).f (HAdd.hAdd n 1)) ((Finset.filter (f …
      ⊢ Eq ((AlgebraicTopology.DoldKan.Q (HAdd.hAdd q 1)).f (HAdd.hAdd n 1)) ((Finse …
    -/
  · by_cases hqn : q + 1 ≤ n + 1
    /-
      case pos
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      n q : Nat
      hq : Eq ((AlgebraicTopology.DoldKan.Q q).f (HAdd.hAdd n 1)) ((Finset.filter (f …
      hqn : LE.le (HAdd.hAdd q 1) (HAdd.hAdd n 1)
      ⊢ Eq ((AlgebraicTopology.DoldKan.Q (HAdd.hAdd q 1)).f (HAdd.hAdd n 1)) ((Finse …
    -/
    swap
      /-
        case neg
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        n q : Nat
        hq : Eq ((AlgebraicTopology.DoldKan.Q q).f (HAdd.hAdd n 1)) ((Finset.filter (f …
        hqn : Not (LE.le (HAdd.hAdd q 1) (HAdd.hAdd n 1))
        ⊢ Eq ((AlgebraicTopology.DoldKan.Q (HAdd.hAdd q 1)).f (HAdd.hAdd n 1)) ((Finse …
      -/
    · rw [Q_is_eventually_constant (show n + 1 ≤ q by omega), hq]
      /-
        case neg
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        n q : Nat
        hq : Eq ((AlgebraicTopology.DoldKan.Q q).f (HAdd.hAdd n 1)) ((Finset.filter (f …
        hqn : Not (LE.le (HAdd.hAdd q 1) (HAdd.hAdd n 1))
        ⊢ Eq ((Finset.filter (fun i => LT.lt (↑i) q) Finset.univ).sum fun i => Categor …
      -/
      congr 1
      /-
        case neg.e_s
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        n q : Nat
        hq : Eq ((AlgebraicTopology.DoldKan.Q q).f (HAdd.hAdd n 1)) ((Finset.filter (f …
        hqn : Not (LE.le (HAdd.hAdd q 1) (HAdd.hAdd n 1))
        ⊢ Eq (Finset.filter (fun i => LT.lt (↑i) q) Finset.univ) (Finset.filter (fun i …
      -/
      ext ⟨x, hx⟩
      /-
        case neg.e_s.h.mk
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        n q : Nat
        hq : Eq ((AlgebraicTopology.DoldKan.Q q).f (HAdd.hAdd n 1)) ((Finset.filter (f …
        hqn : Not (LE.le (HAdd.hAdd q 1) (HAdd.hAdd n 1))
        x : Nat
        hx : LT.lt x (HAdd.hAdd n 1)
        ⊢ Iff (Membership.mem (Finset.filter (fun i => LT.lt (↑i) q) Finset.univ) ⟨x,  …
      -/
      simp only [Nat.succ_eq_add_one, Finset.mem_filter, Finset.mem_univ, true_and]
      /-
        case neg.e_s.h.mk
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        n q : Nat
        hq : Eq ((AlgebraicTopology.DoldKan.Q q).f (HAdd.hAdd n 1)) ((Finset.filter (f …
        hqn : Not (LE.le (HAdd.hAdd q 1) (HAdd.hAdd n 1))
        x : Nat
        hx : LT.lt x (HAdd.hAdd n 1)
        ⊢ Iff (LT.lt x q) (LT.lt x (HAdd.hAdd q 1))
      -/
      omega
      /-
        🎉 no goals
      -/
      /-
        case pos
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        n q : Nat
        hq : Eq ((AlgebraicTopology.DoldKan.Q q).f (HAdd.hAdd n 1)) ((Finset.filter (f …
        hqn : LE.le (HAdd.hAdd q 1) (HAdd.hAdd n 1)
        ⊢ Eq ((AlgebraicTopology.DoldKan.Q (HAdd.hAdd q 1)).f (HAdd.hAdd n 1)) ((Finse …
      -/
    · cases' Nat.le.dest (Nat.succ_le_succ_iff.mp hqn) with a ha
      /-
        case pos.intro
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        n q : Nat
        hq : Eq ((AlgebraicTopology.DoldKan.Q q).f (HAdd.hAdd n 1)) ((Finset.filter (f …
        hqn : LE.le (HAdd.hAdd q 1) (HAdd.hAdd n 1)
        a : Nat
        ha : Eq (HAdd.hAdd q a) n
        ⊢ Eq ((AlgebraicTopology.DoldKan.Q (HAdd.hAdd q 1)).f (HAdd.hAdd n 1)) ((Finse …
      -/
      rw [Q_succ, HomologicalComplex.sub_f_apply, HomologicalComplex.comp_f, hq]
      /-
        case pos.intro
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        n q : Nat
        hq : Eq ((AlgebraicTopology.DoldKan.Q q).f (HAdd.hAdd n 1)) ((Finset.filter (f …
        hqn : LE.le (HAdd.hAdd q 1) (HAdd.hAdd n 1)
        a : Nat
        ha : Eq (HAdd.hAdd q a) n
        ⊢ Eq (HSub.hSub ((Finset.filter (fun i => LT.lt (↑i) q) Finset.univ).sum fun i …
      -/
      symm
      /-
        case pos.intro
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        n q : Nat
        hq : Eq ((AlgebraicTopology.DoldKan.Q q).f (HAdd.hAdd n 1)) ((Finset.filter (f …
        hqn : LE.le (HAdd.hAdd q 1) (HAdd.hAdd n 1)
        a : Nat
        ha : Eq (HAdd.hAdd q a) n
        ⊢ Eq ((Finset.filter (fun i => LT.lt (↑i) (HAdd.hAdd q 1)) Finset.univ).sum fu …
      -/
      conv_rhs => rw [sub_eq_add_neg, add_comm]
      /-
        case pos.intro
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        n q : Nat
        hq : Eq ((AlgebraicTopology.DoldKan.Q q).f (HAdd.hAdd n 1)) ((Finset.filter (f …
        hqn : LE.le (HAdd.hAdd q 1) (HAdd.hAdd n 1)
        a : Nat
        ha : Eq (HAdd.hAdd q a) n
        ⊢ Eq ((Finset.filter (fun i => LT.lt (↑i) (HAdd.hAdd q 1)) Finset.univ).sum fu …
      -/
      let q' : Fin (n + 1) := ⟨q, Nat.succ_le_iff.mp hqn⟩
      /-
        case pos.intro
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        n q : Nat
        hq : Eq ((AlgebraicTopology.DoldKan.Q q).f (HAdd.hAdd n 1)) ((Finset.filter (f …
        hqn : LE.le (HAdd.hAdd q 1) (HAdd.hAdd n 1)
        a : Nat
        ha : Eq (HAdd.hAdd q a) n
        q' : Fin (HAdd.hAdd n 1) := ⟨q, ⋯⟩
        ⊢ Eq ((Finset.filter (fun i => LT.lt (↑i) (HAdd.hAdd q 1)) Finset.univ).sum fu …
      -/
      rw [← @Finset.add_sum_erase _ _ _ _ _ _ q' (by simp [q'])]
      /-
        case pos.intro
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        n q : Nat
        hq : Eq ((AlgebraicTopology.DoldKan.Q q).f (HAdd.hAdd n 1)) ((Finset.filter (f …
        hqn : LE.le (HAdd.hAdd q 1) (HAdd.hAdd n 1)
        a : Nat
        ha : Eq (HAdd.hAdd q a) n
        q' : Fin (HAdd.hAdd n 1) := ⟨q, ⋯⟩
        ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp ((AlgebraicTopology.DoldKa …
      -/
      congr
        /-
          case pos.intro.e_a
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
          inst✝ : CategoryTheory.Preadditive C
          X : CategoryTheory.SimplicialObject C
          n q : Nat
          hq : Eq ((AlgebraicTopology.DoldKan.Q q).f (HAdd.hAdd n 1)) ((Finset.filter (f …
          hqn : LE.le (HAdd.hAdd q 1) (HAdd.hAdd n 1)
          a : Nat
          ha : Eq (HAdd.hAdd q a) n
          q' : Fin (HAdd.hAdd n 1) := ⟨q, ⋯⟩
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicTopology.DoldKan.P ↑q').f  …
        -/
      · have hnaq' : n = a + q := by omega
        simp only [Fin.val_mk, (HigherFacesVanish.of_P q n).comp_Hσ_eq hnaq',
          q'.rev_eq hnaq', neg_neg]
        /-
          case pos.intro.e_a
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
          inst✝ : CategoryTheory.Preadditive C
          X : CategoryTheory.SimplicialObject C
          n q : Nat
          hq : Eq ((AlgebraicTopology.DoldKan.Q q).f (HAdd.hAdd n 1)) ((Finset.filter (f …
          hqn : LE.le (HAdd.hAdd q 1) (HAdd.hAdd n 1)
          a : Nat
          ha : Eq (HAdd.hAdd q a) n
          q' : Fin (HAdd.hAdd n 1) := ⟨q, ⋯⟩
          hnaq' : Eq n (HAdd.hAdd a q)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicTopology.DoldKan.P q).f (H …
        -/
        rfl
        /-
          🎉 no goals
        -/
        /-
          case pos.intro.e_a.e_s
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
          inst✝ : CategoryTheory.Preadditive C
          X : CategoryTheory.SimplicialObject C
          n q : Nat
          hq : Eq ((AlgebraicTopology.DoldKan.Q q).f (HAdd.hAdd n 1)) ((Finset.filter (f …
          hqn : LE.le (HAdd.hAdd q 1) (HAdd.hAdd n 1)
          a : Nat
          ha : Eq (HAdd.hAdd q a) n
          q' : Fin (HAdd.hAdd n 1) := ⟨q, ⋯⟩
          ⊢ Eq ((Finset.filter (fun i => LT.lt (↑i) (HAdd.hAdd q 1)) Finset.univ).erase  …
        -/
      · ext ⟨i, hi⟩
        simp only [q', Nat.succ_eq_add_one, Nat.lt_succ_iff_lt_or_eq, Finset.mem_univ,
          forall_true_left, Finset.mem_filter, lt_self_iff_false, or_true, and_self, not_true,
          Finset.mem_erase, ne_eq, Fin.mk.injEq, true_and]
        /-
          case pos.intro.e_a.e_s.h.mk
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
          inst✝ : CategoryTheory.Preadditive C
          X : CategoryTheory.SimplicialObject C
          n q : Nat
          hq : Eq ((AlgebraicTopology.DoldKan.Q q).f (HAdd.hAdd n 1)) ((Finset.filter (f …
          hqn : LE.le (HAdd.hAdd q 1) (HAdd.hAdd n 1)
          a : Nat
          ha : Eq (HAdd.hAdd q a) n
          q' : Fin (HAdd.hAdd n 1) := ⟨q, ⋯⟩
          i : Nat
          hi : LT.lt i (HAdd.hAdd n 1)
          ⊢ Iff (And (Not (Eq i q)) (Or (LT.lt i q) (Eq i q))) (LT.lt i q)
        -/
        aesop
        /-
          🎉 no goals
        -/


/-- The structure `MorphComponents` is an ad hoc structure that is used in
the proof that `N₁ : SimplicialObject C ⥤ Karoubi (ChainComplex C ℕ))`
reflects isomorphisms. The fields are the data that are needed in order to
construct a morphism `X _[n+1] ⟶ Z` (see `φ`) using the decomposition of the
identity given by `decomposition_Q n (n+1)`. -/
@[ext]
structure MorphComponents (n : ℕ) (Z : C) where
  a : X _[n + 1] ⟶ Z
  b : Fin (n + 1) → (X _[n] ⟶ Z)


/-- The morphism `X _[n+1] ⟶ Z` associated to `f : MorphComponents X n Z`. -/
def φ {Z : C} (f : MorphComponents X n Z) : X _[n + 1] ⟶ Z :=
  PInfty.f (n + 1) ≫ f.a + ∑ i : Fin (n + 1), (P i).f (n + 1) ≫ X.δ i.rev.succ ≫
    f.b (Fin.rev i)


/-- the canonical `MorphComponents` whose associated morphism is the identity
(see `F_id`) thanks to `decomposition_Q n (n+1)` -/
@[simps]
def id : MorphComponents X n (X _[n + 1]) where
  a := PInfty.f (n + 1)
  b i := X.σ i


@[simp]
theorem id_φ : (id X n).φ = 𝟙 _ := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    ⊢ Eq (AlgebraicTopology.DoldKan.MorphComponents.id X n).φ (CategoryTheory.Cate …
  -/
  simp only [← P_add_Q_f (n + 1) (n + 1), φ]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan …
  -/
  congr 1
    /-
      case e_a
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      n : Nat
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.PInfty.f ( …
    -/
  · simp only [id, PInfty_f, P_f_idem]
    /-
      🎉 no goals
    -/
    /-
      case e_a
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      n : Nat
      ⊢ Eq (Finset.univ.sum fun i => CategoryTheory.CategoryStruct.comp ((AlgebraicT …
    -/
  · exact Eq.trans (by congr; simp) (decomposition_Q n (n + 1)).symm
    /-
      🎉 no goals
    -/


/-- A `MorphComponents` can be postcomposed with a morphism. -/
@[simps]
def postComp : MorphComponents X n Z' where
  a := f.a ≫ h
  b i := f.b i ≫ h


@[simp]
theorem postComp_φ : (f.postComp h).φ = f.φ ≫ h := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    Z Z' : C
    f : AlgebraicTopology.DoldKan.MorphComponents X n Z
    h : Quiver.Hom Z Z'
    ⊢ Eq (f.postComp h).φ (CategoryTheory.CategoryStruct.comp f.φ h)
  -/
  unfold φ postComp
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    Z Z' : C
    f : AlgebraicTopology.DoldKan.MorphComponents X n Z
    h : Quiver.Hom Z Z'
    ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan …
  -/
  simp only [add_comp, sum_comp, assoc]
  /-
    🎉 no goals
  -/


/-- A `MorphComponents` can be precomposed with a morphism of simplicial objects. -/
@[simps]
def preComp : MorphComponents X' n Z where
  a := g.app (op [n + 1]) ≫ f.a
  b i := g.app (op [n]) ≫ f.b i


@[simp]
theorem preComp_φ : (f.preComp g).φ = g.app (op [n + 1]) ≫ f.φ := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X X' : CategoryTheory.SimplicialObject C
    n : Nat
    Z : C
    f : AlgebraicTopology.DoldKan.MorphComponents X n Z
    g : Quiver.Hom X' X
    ⊢ Eq (f.preComp g).φ (CategoryTheory.CategoryStruct.comp (g.app { unop := Simp …
  -/
  unfold φ preComp
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X X' : CategoryTheory.SimplicialObject C
    n : Nat
    Z : C
    f : AlgebraicTopology.DoldKan.MorphComponents X n Z
    g : Quiver.Hom X' X
    ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan …
  -/
  simp only [PInfty_f, comp_add]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X X' : CategoryTheory.SimplicialObject C
    n : Nat
    Z : C
    f : AlgebraicTopology.DoldKan.MorphComponents X n Z
    g : Quiver.Hom X' X
    ⊢ Eq (HAdd.hAdd (CategoryTheory.CategoryStruct.comp ((AlgebraicTopology.DoldKa …
  -/
  congr 1
    /-
      case e_a
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X X' : CategoryTheory.SimplicialObject C
      n : Nat
      Z : C
      f : AlgebraicTopology.DoldKan.MorphComponents X n Z
      g : Quiver.Hom X' X
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicTopology.DoldKan.P (HAdd.h …
    -/
  · simp only [P_f_naturality_assoc]
    /-
      🎉 no goals
    -/
    /-
      case e_a
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X X' : CategoryTheory.SimplicialObject C
      n : Nat
      Z : C
      f : AlgebraicTopology.DoldKan.MorphComponents X n Z
      g : Quiver.Hom X' X
      ⊢ Eq (Finset.univ.sum fun x => CategoryTheory.CategoryStruct.comp ((AlgebraicT …
    -/
  · simp only [comp_sum, P_f_naturality_assoc, SimplicialObject.δ_naturality_assoc]
    /-
      🎉 no goals
    -/


