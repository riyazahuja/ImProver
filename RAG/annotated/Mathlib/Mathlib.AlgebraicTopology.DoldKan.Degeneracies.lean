theorem HigherFacesVanish.comp_σ {Y : C} {X : SimplicialObject C} {n b q : ℕ} {φ : Y ⟶ X _[n + 1]}
    (v : HigherFacesVanish q φ) (hnbq : n + 1 = b + q) :
    HigherFacesVanish q
      (φ ≫
        X.σ ⟨b, by
          /-
            C : Type u_1
            inst✝¹ : CategoryTheory.Category.{?u.29, u_1} C
            inst✝ : CategoryTheory.Preadditive C
            Y : C
            X : CategoryTheory.SimplicialObject C
            n b q : Nat
            φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
            v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
            hnbq : Eq (HAdd.hAdd n 1) (HAdd.hAdd b q)
            ⊢ LT.lt b (HAdd.hAdd (HAdd.hAdd n 1) 1)
          -/
          simp only [hnbq, Nat.lt_add_one_iff, le_add_iff_nonneg_right, zero_le]⟩) :=
          /-
            🎉 no goals
          -/
  fun j hj => by
  rw [assoc, SimplicialObject.δ_comp_σ_of_gt', Fin.pred_succ, v.comp_δ_eq_zero_assoc _ _ hj,
    zero_comp]
    /-
      case H
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      Y : C
      X : CategoryTheory.SimplicialObject C
      n b q : Nat
      φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
      v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
      hnbq : Eq (HAdd.hAdd n 1) (HAdd.hAdd b q)
      j : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)
      hj : LE.le (HAdd.hAdd (HAdd.hAdd n 1) 1) (HAdd.hAdd (↑j) q)
      ⊢ LT.lt ⟨b, ⋯⟩.succ j.succ
    -/
  · dsimp
    /-
      case H
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      Y : C
      X : CategoryTheory.SimplicialObject C
      n b q : Nat
      φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
      v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
      hnbq : Eq (HAdd.hAdd n 1) (HAdd.hAdd b q)
      j : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)
      hj : LE.le (HAdd.hAdd (HAdd.hAdd n 1) 1) (HAdd.hAdd (↑j) q)
      ⊢ LT.lt ⟨HAdd.hAdd b 1, ⋯⟩ j.succ
    -/
    rw [Fin.lt_iff_val_lt_val, Fin.val_succ]
    /-
      case H
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      Y : C
      X : CategoryTheory.SimplicialObject C
      n b q : Nat
      φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
      v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
      hnbq : Eq (HAdd.hAdd n 1) (HAdd.hAdd b q)
      j : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)
      hj : LE.le (HAdd.hAdd (HAdd.hAdd n 1) 1) (HAdd.hAdd (↑j) q)
      ⊢ LT.lt (↑⟨HAdd.hAdd b 1, ⋯⟩) (HAdd.hAdd (↑j) 1)
    -/
    linarith
    /-
      🎉 no goals
    -/
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      Y : C
      X : CategoryTheory.SimplicialObject C
      n b q : Nat
      φ : Quiver.Hom Y (X.obj { unop := SimplexCategory.mk (HAdd.hAdd n 1) })
      v : AlgebraicTopology.DoldKan.HigherFacesVanish q φ
      hnbq : Eq (HAdd.hAdd n 1) (HAdd.hAdd b q)
      j : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)
      hj : LE.le (HAdd.hAdd (HAdd.hAdd n 1) 1) (HAdd.hAdd (↑j) q)
      ⊢ Ne j 0
    -/
  · intro hj'
    simp only [hnbq, add_comm b, add_assoc, hj', Fin.val_zero, zero_add, add_le_iff_nonpos_right,
      nonpos_iff_eq_zero, add_eq_zero, false_and, reduceCtorEq] at hj


theorem σ_comp_P_eq_zero (X : SimplicialObject C) {n q : ℕ} (i : Fin (n + 1)) (hi : n + 1 ≤ i + q) :
    X.σ i ≫ (P q).f (n + 1) = 0 := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    n q : Nat
    i : Fin (HAdd.hAdd n 1)
    hi : LE.le (HAdd.hAdd n 1) (HAdd.hAdd (↑i) q)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.σ i) ((AlgebraicTopology.DoldKan.P …
  -/
  revert i hi
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    n q : Nat
    ⊢ ∀ (i : Fin (HAdd.hAdd n 1)), LE.le (HAdd.hAdd n 1) (HAdd.hAdd (↑i) q) → Eq ( …
  -/
  induction' q with q hq
    /-
      case zero
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      n : Nat
      ⊢ ∀ (i : Fin (HAdd.hAdd n 1)), LE.le (HAdd.hAdd n 1) (HAdd.hAdd (↑i) 0) → Eq ( …
    -/
  · intro i (hi : n + 1 ≤ i)
    /-
      case zero
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      hi : LE.le (HAdd.hAdd n 1) ↑i
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.σ i) ((AlgebraicTopology.DoldKan.P …
    -/
    omega
    /-
      🎉 no goals
    -/
    /-
      case succ
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      n q : Nat
      hq : ∀ (i : Fin (HAdd.hAdd n 1)), LE.le (HAdd.hAdd n 1) (HAdd.hAdd (↑i) q) → E …
      ⊢ ∀ (i : Fin (HAdd.hAdd n 1)), LE.le (HAdd.hAdd n 1) (HAdd.hAdd (↑i) (HAdd.hAd …
    -/
  · intro i (hi : n + 1 ≤ i + q + 1)
    /-
      case succ
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      n q : Nat
      hq : ∀ (i : Fin (HAdd.hAdd n 1)), LE.le (HAdd.hAdd n 1) (HAdd.hAdd (↑i) q) → E …
      i : Fin (HAdd.hAdd n 1)
      hi : LE.le (HAdd.hAdd n 1) (HAdd.hAdd (HAdd.hAdd (↑i) q) 1)
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.σ i) ((AlgebraicTopology.DoldKan.P …
    -/
    by_cases h : n + 1 ≤ (i : ℕ) + q
      /-
        case pos
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        n q : Nat
        hq : ∀ (i : Fin (HAdd.hAdd n 1)), LE.le (HAdd.hAdd n 1) (HAdd.hAdd (↑i) q) → E …
        i : Fin (HAdd.hAdd n 1)
        hi : LE.le (HAdd.hAdd n 1) (HAdd.hAdd (HAdd.hAdd (↑i) q) 1)
        h : LE.le (HAdd.hAdd n 1) (HAdd.hAdd (↑i) q)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.σ i) ((AlgebraicTopology.DoldKan.P …
      -/
    · rw [P_succ, HomologicalComplex.comp_f, ← assoc, hq i h, zero_comp]
      /-
        🎉 no goals
      -/
    · replace hi : n = i + q := by
        obtain ⟨j, hj⟩ := le_iff_exists_add.mp hi
        rw [← Nat.lt_succ_iff, Nat.succ_eq_add_one, hj, not_lt, add_le_iff_nonpos_right,
          nonpos_iff_eq_zero] at h
        rw [← add_left_inj 1, hj, self_eq_add_right, h]
      /-
        case neg
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
        inst✝ : CategoryTheory.Preadditive C
        X : CategoryTheory.SimplicialObject C
        n q : Nat
        hq : ∀ (i : Fin (HAdd.hAdd n 1)), LE.le (HAdd.hAdd n 1) (HAdd.hAdd (↑i) q) → E …
        i : Fin (HAdd.hAdd n 1)
        h : Not (LE.le (HAdd.hAdd n 1) (HAdd.hAdd (↑i) q))
        hi : Eq n (HAdd.hAdd (↑i) q)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.σ i) ((AlgebraicTopology.DoldKan.P …
      -/
      rcases n with _|n
        /-
          case neg.zero
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
          inst✝ : CategoryTheory.Preadditive C
          X : CategoryTheory.SimplicialObject C
          q : Nat
          hq : ∀ (i : Fin (HAdd.hAdd 0 1)), LE.le (HAdd.hAdd 0 1) (HAdd.hAdd (↑i) q) → E …
          i : Fin (HAdd.hAdd 0 1)
          h : Not (LE.le (HAdd.hAdd 0 1) (HAdd.hAdd (↑i) q))
          hi : Eq 0 (HAdd.hAdd (↑i) q)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.σ i) ((AlgebraicTopology.DoldKan.P …
        -/
      · fin_cases i
        /-
          case neg.zero.«_@»._hyg.320.«0»
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
          inst✝ : CategoryTheory.Preadditive C
          X : CategoryTheory.SimplicialObject C
          q : Nat
          hq : ∀ (i : Fin (HAdd.hAdd 0 1)), LE.le (HAdd.hAdd 0 1) (HAdd.hAdd (↑i) q) → E …
          h : Not (LE.le (HAdd.hAdd 0 1) (HAdd.hAdd (↑((fun i => i) ⟨0, ⋯⟩)) q))
          hi : Eq 0 (HAdd.hAdd (↑((fun i => i) ⟨0, ⋯⟩)) q)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.σ ((fun i => i) ⟨0, ⋯⟩)) ((Algebra …
        -/
        dsimp at h hi
        /-
          case neg.zero.«_@»._hyg.320.«0»
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
          inst✝ : CategoryTheory.Preadditive C
          X : CategoryTheory.SimplicialObject C
          q : Nat
          hq : ∀ (i : Fin (HAdd.hAdd 0 1)), LE.le (HAdd.hAdd 0 1) (HAdd.hAdd (↑i) q) → E …
          h : Not (LE.le 1 (HAdd.hAdd 0 q))
          hi : Eq 0 (HAdd.hAdd 0 q)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.σ ((fun i => i) ⟨0, ⋯⟩)) ((Algebra …
        -/
        rw [show q = 0 by omega]
        /-
          case neg.zero.«_@»._hyg.320.«0»
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
          inst✝ : CategoryTheory.Preadditive C
          X : CategoryTheory.SimplicialObject C
          q : Nat
          hq : ∀ (i : Fin (HAdd.hAdd 0 1)), LE.le (HAdd.hAdd 0 1) (HAdd.hAdd (↑i) q) → E …
          h : Not (LE.le 1 (HAdd.hAdd 0 q))
          hi : Eq 0 (HAdd.hAdd 0 q)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.σ ((fun i => i) ⟨0, ⋯⟩)) ((Algebra …
        -/
        change X.σ 0 ≫ (P 1).f 1 = 0
        simp only [P_succ, HomologicalComplex.add_f_apply, comp_add,
          HomologicalComplex.id_f, AlternatingFaceMapComplex.obj_d_eq, Hσ,
          HomologicalComplex.comp_f, Homotopy.nullHomotopicMap'_f (c_mk 2 1 rfl) (c_mk 1 0 rfl),
          comp_id]
        erw [hσ'_eq' (zero_add 0).symm, hσ'_eq' (add_zero 1).symm, comp_id, Fin.sum_univ_two,
          Fin.sum_univ_succ, Fin.sum_univ_two]
        simp only [Fin.val_zero, pow_zero, pow_one, pow_add, one_smul, neg_smul, Fin.mk_one,
          Fin.val_succ, Fin.val_one, Fin.succ_one_eq_two, P_zero, HomologicalComplex.id_f,
          Fin.val_two, pow_two, mul_neg, one_mul, neg_mul, neg_neg, id_comp, add_comp,
          comp_add, Fin.mk_zero, neg_comp, comp_neg, Fin.succ_zero_eq_one]
        erw [SimplicialObject.δ_comp_σ_self, SimplicialObject.δ_comp_σ_self_assoc,
          SimplicialObject.δ_comp_σ_succ, comp_id,
          SimplicialObject.δ_comp_σ_of_le X
            (show (0 : Fin 2) ≤ Fin.castSucc 0 by rw [Fin.castSucc_zero]),
          SimplicialObject.δ_comp_σ_self_assoc, SimplicialObject.δ_comp_σ_succ_assoc]
        /-
          case neg.zero.«_@»._hyg.320.«0»
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
          inst✝ : CategoryTheory.Preadditive C
          X : CategoryTheory.SimplicialObject C
          q : Nat
          hq : ∀ (i : Fin (HAdd.hAdd 0 1)), LE.le (HAdd.hAdd 0 1) (HAdd.hAdd (↑i) q) → E …
          h : Not (LE.le 1 (HAdd.hAdd 0 q))
          hi : Eq 0 (HAdd.hAdd 0 q)
          ⊢ Eq (HAdd.hAdd (X.σ 0) (HAdd.hAdd (HAdd.hAdd (X.σ 0) (Neg.neg (X.σ 0))) (HAdd …
        -/
        simp only [add_neg_cancel, add_zero, zero_add]
        /-
          🎉 no goals
        -/
      · rw [← id_comp (X.σ i), ← (P_add_Q_f q n.succ : _ = 𝟙 (X.obj _)), add_comp, add_comp,
          P_succ]
        have v : HigherFacesVanish q ((P q).f n.succ ≫ X.σ i) :=
          (HigherFacesVanish.of_P q n).comp_σ hi
        erw [← assoc, v.comp_P_eq_self, HomologicalComplex.add_f_apply, Preadditive.comp_add,
          comp_id, v.comp_Hσ_eq hi, assoc, SimplicialObject.δ_comp_σ_succ_assoc, Fin.eta,
          decomposition_Q n q, sum_comp, sum_comp, Finset.sum_eq_zero, add_zero, add_neg_eq_zero]
        /-
          case neg.succ
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
          inst✝ : CategoryTheory.Preadditive C
          X : CategoryTheory.SimplicialObject C
          q n : Nat
          hq : ∀ (i : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)), LE.le (HAdd.hAdd (HAdd.hAdd n  …
          i : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)
          h : Not (LE.le (HAdd.hAdd (HAdd.hAdd n 1) 1) (HAdd.hAdd (↑i) q))
          hi : Eq (HAdd.hAdd n 1) (HAdd.hAdd (↑i) q)
          v : AlgebraicTopology.DoldKan.HigherFacesVanish q (CategoryTheory.CategoryStru …
          ⊢ ∀ (x : Fin (HAdd.hAdd n 1)), Membership.mem (Finset.filter (fun i => LT.lt ( …
        -/
        intro j hj
        /-
          case neg.succ
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
          inst✝ : CategoryTheory.Preadditive C
          X : CategoryTheory.SimplicialObject C
          q n : Nat
          hq : ∀ (i : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)), LE.le (HAdd.hAdd (HAdd.hAdd n  …
          i : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)
          h : Not (LE.le (HAdd.hAdd (HAdd.hAdd n 1) 1) (HAdd.hAdd (↑i) q))
          hi : Eq (HAdd.hAdd n 1) (HAdd.hAdd (↑i) q)
          v : AlgebraicTopology.DoldKan.HigherFacesVanish q (CategoryTheory.CategoryStru …
          j : Fin (HAdd.hAdd n 1)
          hj : Membership.mem (Finset.filter (fun i => LT.lt (↑i) q) Finset.univ) j
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        simp only [Finset.mem_univ, Finset.mem_filter] at hj
        /-
          case neg.succ
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
          inst✝ : CategoryTheory.Preadditive C
          X : CategoryTheory.SimplicialObject C
          q n : Nat
          hq : ∀ (i : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)), LE.le (HAdd.hAdd (HAdd.hAdd n  …
          i : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)
          h : Not (LE.le (HAdd.hAdd (HAdd.hAdd n 1) 1) (HAdd.hAdd (↑i) q))
          hi : Eq (HAdd.hAdd n 1) (HAdd.hAdd (↑i) q)
          v : AlgebraicTopology.DoldKan.HigherFacesVanish q (CategoryTheory.CategoryStru …
          j : Fin (HAdd.hAdd n 1)
          hj : And True (LT.lt (↑j) q)
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        obtain ⟨k, hk⟩ := Nat.le.dest (Nat.lt_succ_iff.mp (Fin.is_lt j))
        /-
          case neg.succ.intro
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
          inst✝ : CategoryTheory.Preadditive C
          X : CategoryTheory.SimplicialObject C
          q n : Nat
          hq : ∀ (i : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)), LE.le (HAdd.hAdd (HAdd.hAdd n  …
          i : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)
          h : Not (LE.le (HAdd.hAdd (HAdd.hAdd n 1) 1) (HAdd.hAdd (↑i) q))
          hi : Eq (HAdd.hAdd n 1) (HAdd.hAdd (↑i) q)
          v : AlgebraicTopology.DoldKan.HigherFacesVanish q (CategoryTheory.CategoryStru …
          j : Fin (HAdd.hAdd n 1)
          hj : And True (LT.lt (↑j) q)
          k : Nat
          hk : Eq (HAdd.hAdd (↑j) k) n
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        rw [add_comm] at hk
        have hi' : i = Fin.castSucc ⟨i, by omega⟩ := by
          ext
          simp only [Fin.castSucc_mk, Fin.eta]
        have eq := hq j.rev.succ (by
          simp only [← hk, Fin.rev_eq j hk.symm, Nat.succ_eq_add_one, Fin.succ_mk, Fin.val_mk]
          omega)
        rw [HomologicalComplex.comp_f, assoc, assoc, assoc, hi',
          SimplicialObject.σ_comp_σ_assoc, reassoc_of% eq, zero_comp, comp_zero, comp_zero,
          comp_zero]
        /-
          case neg.succ.intro.H
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
          inst✝ : CategoryTheory.Preadditive C
          X : CategoryTheory.SimplicialObject C
          q n : Nat
          hq : ∀ (i : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)), LE.le (HAdd.hAdd (HAdd.hAdd n  …
          i : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)
          h : Not (LE.le (HAdd.hAdd (HAdd.hAdd n 1) 1) (HAdd.hAdd (↑i) q))
          hi : Eq (HAdd.hAdd n 1) (HAdd.hAdd (↑i) q)
          v : AlgebraicTopology.DoldKan.HigherFacesVanish q (CategoryTheory.CategoryStru …
          j : Fin (HAdd.hAdd n 1)
          hj : And True (LT.lt (↑j) q)
          k : Nat
          hk : Eq (HAdd.hAdd k ↑j) n
          hi' : Eq i ⟨↑i, ⋯⟩.castSucc
          eq : Eq (CategoryTheory.CategoryStruct.comp (X.σ j.rev.succ) ((AlgebraicTopolo …
          ⊢ LE.le ⟨↑i, ⋯⟩ j.rev
        -/
        simp only [Fin.rev_eq j hk.symm, Fin.le_iff_val_le_val, Fin.val_mk]
        /-
          case neg.succ.intro.H
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
          inst✝ : CategoryTheory.Preadditive C
          X : CategoryTheory.SimplicialObject C
          q n : Nat
          hq : ∀ (i : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)), LE.le (HAdd.hAdd (HAdd.hAdd n  …
          i : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1)
          h : Not (LE.le (HAdd.hAdd (HAdd.hAdd n 1) 1) (HAdd.hAdd (↑i) q))
          hi : Eq (HAdd.hAdd n 1) (HAdd.hAdd (↑i) q)
          v : AlgebraicTopology.DoldKan.HigherFacesVanish q (CategoryTheory.CategoryStru …
          j : Fin (HAdd.hAdd n 1)
          hj : And True (LT.lt (↑j) q)
          k : Nat
          hk : Eq (HAdd.hAdd k ↑j) n
          hi' : Eq i ⟨↑i, ⋯⟩.castSucc
          eq : Eq (CategoryTheory.CategoryStruct.comp (X.σ j.rev.succ) ((AlgebraicTopolo …
          ⊢ LE.le (↑i) k
        -/
        omega
        /-
          🎉 no goals
        -/


@[reassoc (attr := simp)]
theorem σ_comp_PInfty (X : SimplicialObject C) {n : ℕ} (i : Fin (n + 1)) :
    X.σ i ≫ PInfty.f (n + 1) = 0 := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.σ i) (AlgebraicTopology.DoldKan.PI …
  -/
  rw [PInfty_f, σ_comp_P_eq_zero X i]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    i : Fin (HAdd.hAdd n 1)
    ⊢ LE.le (HAdd.hAdd n 1) (HAdd.hAdd (↑i) (HAdd.hAdd n 1))
  -/
  simp only [le_add_iff_nonneg_left, zero_le]
  /-
    🎉 no goals
  -/


@[reassoc]
theorem degeneracy_comp_PInfty (X : SimplicialObject C) (n : ℕ) {Δ' : SimplexCategory}
    (θ : ([n] : SimplexCategory) ⟶ Δ') (hθ : ¬Mono θ) : X.map θ.op ≫ PInfty.f n = 0 := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    Δ' : SimplexCategory
    θ : Quiver.Hom (SimplexCategory.mk n) Δ'
    hθ : Not (CategoryTheory.Mono θ)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.map θ.op) (AlgebraicTopology.DoldK …
  -/
  rw [SimplexCategory.mono_iff_injective] at hθ
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Preadditive C
    X : CategoryTheory.SimplicialObject C
    n : Nat
    Δ' : SimplexCategory
    θ : Quiver.Hom (SimplexCategory.mk n) Δ'
    hθ : Not (Function.Injective ⇑(SimplexCategory.Hom.toOrderHom θ))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.map θ.op) (AlgebraicTopology.DoldK …
  -/
  cases n
    /-
      case zero
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      Δ' : SimplexCategory
      θ : Quiver.Hom (SimplexCategory.mk 0) Δ'
      hθ : Not (Function.Injective ⇑(SimplexCategory.Hom.toOrderHom θ))
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.map θ.op) (AlgebraicTopology.DoldK …
    -/
  · exfalso
    /-
      case zero
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      Δ' : SimplexCategory
      θ : Quiver.Hom (SimplexCategory.mk 0) Δ'
      hθ : Not (Function.Injective ⇑(SimplexCategory.Hom.toOrderHom θ))
      ⊢ False
    -/
    apply hθ
    /-
      case zero
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      Δ' : SimplexCategory
      θ : Quiver.Hom (SimplexCategory.mk 0) Δ'
      hθ : Not (Function.Injective ⇑(SimplexCategory.Hom.toOrderHom θ))
      ⊢ Function.Injective ⇑(SimplexCategory.Hom.toOrderHom θ)
    -/
    intro x y h
    /-
      case zero
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      Δ' : SimplexCategory
      θ : Quiver.Hom (SimplexCategory.mk 0) Δ'
      hθ : Not (Function.Injective ⇑(SimplexCategory.Hom.toOrderHom θ))
      x y : Fin (HAdd.hAdd (SimplexCategory.mk 0).len 1)
      h : Eq ((SimplexCategory.Hom.toOrderHom θ) x) ((SimplexCategory.Hom.toOrderHom …
      ⊢ Eq x y
    -/
    fin_cases x
    /-
      case zero.«0»
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      Δ' : SimplexCategory
      θ : Quiver.Hom (SimplexCategory.mk 0) Δ'
      hθ : Not (Function.Injective ⇑(SimplexCategory.Hom.toOrderHom θ))
      y : Fin (HAdd.hAdd (SimplexCategory.mk 0).len 1)
      h : Eq ((SimplexCategory.Hom.toOrderHom θ) ((fun i => i) ⟨0, ⋯⟩)) ((SimplexCat …
      ⊢ Eq ((fun i => i) ⟨0, ⋯⟩) y
    -/
    fin_cases y
    /-
      case zero.«0».«0»
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      Δ' : SimplexCategory
      θ : Quiver.Hom (SimplexCategory.mk 0) Δ'
      hθ : Not (Function.Injective ⇑(SimplexCategory.Hom.toOrderHom θ))
      h : Eq ((SimplexCategory.Hom.toOrderHom θ) ((fun i => i) ⟨0, ⋯⟩)) ((SimplexCat …
      ⊢ Eq ((fun i => i) ⟨0, ⋯⟩) ((fun i => i) ⟨0, ⋯⟩)
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case succ
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      inst✝ : CategoryTheory.Preadditive C
      X : CategoryTheory.SimplicialObject C
      Δ' : SimplexCategory
      n✝ : Nat
      θ : Quiver.Hom (SimplexCategory.mk (HAdd.hAdd n✝ 1)) Δ'
      hθ : Not (Function.Injective ⇑(SimplexCategory.Hom.toOrderHom θ))
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (X.map θ.op) (AlgebraicTopology.DoldK …
    -/
  · obtain ⟨i, α, h⟩ := SimplexCategory.eq_σ_comp_of_not_injective θ hθ
    rw [h, op_comp, X.map_comp, assoc, show X.map (SimplexCategory.σ i).op = X.σ i by rfl,
      σ_comp_PInfty, comp_zero]


