theorem natDegree_det_X_add_C_le (A B : Matrix n n α) :
    natDegree (det ((X : α[X]) • A.map C + B.map C : Matrix n n α[X])) ≤ Fintype.card n := by
  /-
    n : Type u_1
    α : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A B : Matrix n n α
    ⊢ LE.le (HAdd.hAdd (HSMul.hSMul Polynomial.X (A.map ⇑Polynomial.C)) (B.map ⇑Po …
  -/
  rw [det_apply]
  /-
    n : Type u_1
    α : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A B : Matrix n n α
    ⊢ LE.le (Finset.univ.sum fun σ => HSMul.hSMul (Equiv.Perm.sign σ) (Finset.univ …
  -/
  refine (natDegree_sum_le _ _).trans ?_
  /-
    n : Type u_1
    α : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A B : Matrix n n α
    ⊢ LE.le (Finset.fold Max.max 0 (Function.comp Polynomial.natDegree fun σ => HS …
  -/
  refine Multiset.max_le_of_forall_le _ _ ?_
  simp only [forall_apply_eq_imp_iff, true_and, Function.comp_apply, Multiset.map_map,
    Multiset.mem_map, exists_imp, Finset.mem_univ_val]
  /-
    n : Type u_1
    α : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A B : Matrix n n α
    ⊢ ∀ (a : Equiv.Perm n), LE.le (HSMul.hSMul (Equiv.Perm.sign a) (Finset.univ.pr …
  -/
  intro g
  calc
    natDegree (sign g • ∏ i : n, (X • A.map C + B.map C : Matrix n n α[X]) (g i) i) ≤
        natDegree (∏ i : n, (X • A.map C + B.map C : Matrix n n α[X]) (g i) i) := by
      cases' Int.units_eq_one_or (sign g) with sg sg
      · rw [sg, one_smul]
      · rw [sg, Units.neg_smul, one_smul, natDegree_neg]
    _ ≤ ∑ i : n, natDegree (((X : α[X]) • A.map C + B.map C : Matrix n n α[X]) (g i) i) :=
      (natDegree_prod_le (Finset.univ : Finset n) fun i : n =>
        (X • A.map C + B.map C : Matrix n n α[X]) (g i) i)
    _ ≤ Finset.univ.card • 1 := (Finset.sum_le_card_nsmul _ _ 1 fun (i : n) _ => ?_)
    _ ≤ Fintype.card n := by simp [mul_one, Algebra.id.smul_eq_mul, Finset.card_univ]
  /-
    n : Type u_1
    α : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A B : Matrix n n α
    g : Equiv.Perm n
    i : n
    x✝ : Membership.mem Finset.univ i
    ⊢ LE.le (HAdd.hAdd (HSMul.hSMul Polynomial.X (A.map ⇑Polynomial.C)) (B.map ⇑Po …
  -/
  dsimp only [add_apply, smul_apply, map_apply, smul_eq_mul]
  /-
    n : Type u_1
    α : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A B : Matrix n n α
    g : Equiv.Perm n
    i : n
    x✝ : Membership.mem Finset.univ i
    ⊢ LE.le (HAdd.hAdd (HMul.hMul Polynomial.X (Polynomial.C (A (g i) i))) (Polyno …
  -/
  compute_degree
  /-
    🎉 no goals
  -/


theorem coeff_det_X_add_C_zero (A B : Matrix n n α) :
    coeff (det ((X : α[X]) • A.map C + B.map C)) 0 = det B := by
  /-
    n : Type u_1
    α : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A B : Matrix n n α
    ⊢ Eq ((HAdd.hAdd (HSMul.hSMul Polynomial.X (A.map ⇑Polynomial.C)) (B.map ⇑Poly …
  -/
  rw [det_apply, finset_sum_coeff, det_apply]
  /-
    n : Type u_1
    α : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A B : Matrix n n α
    ⊢ Eq (Finset.univ.sum fun b => (HSMul.hSMul (Equiv.Perm.sign b) (Finset.univ.p …
  -/
  refine Finset.sum_congr rfl ?_
  /-
    n : Type u_1
    α : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A B : Matrix n n α
    ⊢ ∀ (x : Equiv.Perm n), Membership.mem Finset.univ x → Eq ((HSMul.hSMul (Equiv …
  -/
  rintro g -
  /-
    n : Type u_1
    α : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A B : Matrix n n α
    g : Equiv.Perm n
    ⊢ Eq ((HSMul.hSMul (Equiv.Perm.sign g) (Finset.univ.prod fun i => HAdd.hAdd (H …
  -/
  convert coeff_smul (R := α) (sign g) _ 0
  /-
    case h.e'_3.h.e'_6
    n : Type u_1
    α : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A B : Matrix n n α
    g : Equiv.Perm n
    ⊢ Eq (Finset.univ.prod fun i => B (g i) i) ((Finset.univ.prod fun i => HAdd.hA …
  -/
  rw [coeff_zero_prod]
  /-
    case h.e'_3.h.e'_6
    n : Type u_1
    α : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A B : Matrix n n α
    g : Equiv.Perm n
    ⊢ Eq (Finset.univ.prod fun i => B (g i) i) (Finset.univ.prod fun i => (HAdd.hA …
  -/
  refine Finset.prod_congr rfl ?_
  /-
    case h.e'_3.h.e'_6
    n : Type u_1
    α : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A B : Matrix n n α
    g : Equiv.Perm n
    ⊢ ∀ (x : n), Membership.mem Finset.univ x → Eq (B (g x) x) ((HAdd.hAdd (HSMul. …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem coeff_det_X_add_C_card (A B : Matrix n n α) :
    coeff (det ((X : α[X]) • A.map C + B.map C)) (Fintype.card n) = det A := by
  /-
    n : Type u_1
    α : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A B : Matrix n n α
    ⊢ Eq ((HAdd.hAdd (HSMul.hSMul Polynomial.X (A.map ⇑Polynomial.C)) (B.map ⇑Poly …
  -/
  rw [det_apply, det_apply, finset_sum_coeff]
  /-
    n : Type u_1
    α : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A B : Matrix n n α
    ⊢ Eq (Finset.univ.sum fun b => (HSMul.hSMul (Equiv.Perm.sign b) (Finset.univ.p …
  -/
  refine Finset.sum_congr rfl ?_
  simp only [Algebra.id.smul_eq_mul, Finset.mem_univ, RingHom.mapMatrix_apply, forall_true_left,
    map_apply, Pi.smul_apply]
  /-
    n : Type u_1
    α : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A B : Matrix n n α
    ⊢ ∀ (x : Equiv.Perm n), Eq ((HSMul.hSMul (Equiv.Perm.sign x) (Finset.univ.prod …
  -/
  intro g
  /-
    n : Type u_1
    α : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A B : Matrix n n α
    g : Equiv.Perm n
    ⊢ Eq ((HSMul.hSMul (Equiv.Perm.sign g) (Finset.univ.prod fun i => HAdd.hAdd (H …
  -/
  convert coeff_smul (R := α) (sign g) _ _
  /-
    case h.e'_3.h.e'_6
    n : Type u_1
    α : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A B : Matrix n n α
    g : Equiv.Perm n
    ⊢ Eq (Finset.univ.prod fun i => A (g i) i) ((Finset.univ.prod fun i => HAdd.hA …
  -/
  rw [← mul_one (Fintype.card n)]
  /-
    case h.e'_3.h.e'_6
    n : Type u_1
    α : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A B : Matrix n n α
    g : Equiv.Perm n
    ⊢ Eq (Finset.univ.prod fun i => A (g i) i) ((Finset.univ.prod fun i => HAdd.hA …
  -/
  convert (coeff_prod_of_natDegree_le (R := α) _ _ _ _).symm
    /-
      case h.e'_2.a
      n : Type u_1
      α : Type u_2
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      A B : Matrix n n α
      g : Equiv.Perm n
      x✝ : n
      a✝ : Membership.mem Finset.univ x✝
      ⊢ Eq (A (g x✝) x✝) ((HAdd.hAdd (HSMul.hSMul Polynomial.X (A.map ⇑Polynomial.C) …
    -/
  · simp [coeff_C]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3.h.e'_6.convert_5
      n : Type u_1
      α : Type u_2
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      A B : Matrix n n α
      g : Equiv.Perm n
      ⊢ ∀ (p : n), Membership.mem Finset.univ p → LE.le (HAdd.hAdd (HSMul.hSMul Poly …
    -/
  · rintro p -
    /-
      case h.e'_3.h.e'_6.convert_5
      n : Type u_1
      α : Type u_2
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      A B : Matrix n n α
      g : Equiv.Perm n
      p : n
      ⊢ LE.le (HAdd.hAdd (HSMul.hSMul Polynomial.X (A.map ⇑Polynomial.C)) (B.map ⇑Po …
    -/
    dsimp only [add_apply, smul_apply, map_apply, smul_eq_mul]
    /-
      case h.e'_3.h.e'_6.convert_5
      n : Type u_1
      α : Type u_2
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      A B : Matrix n n α
      g : Equiv.Perm n
      p : n
      ⊢ LE.le (HAdd.hAdd (HMul.hMul Polynomial.X (Polynomial.C (A (g p) p))) (Polyno …
    -/
    compute_degree
    /-
      🎉 no goals
    -/


theorem leadingCoeff_det_X_one_add_C (A : Matrix n n α) :
    leadingCoeff (det ((X : α[X]) • (1 : Matrix n n α[X]) + A.map C)) = 1 := by
  /-
    n : Type u_1
    α : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    ⊢ Eq (HAdd.hAdd (HSMul.hSMul Polynomial.X 1) (A.map ⇑Polynomial.C)).det.leadin …
  -/
  cases subsingleton_or_nontrivial α
    /-
      case inl
      n : Type u_1
      α : Type u_2
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      A : Matrix n n α
      h✝ : Subsingleton α
      ⊢ Eq (HAdd.hAdd (HSMul.hSMul Polynomial.X 1) (A.map ⇑Polynomial.C)).det.leadin …
    -/
  · simp [eq_iff_true_of_subsingleton]
    /-
      🎉 no goals
    -/
  /-
    case inr
    n : Type u_1
    α : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    h✝ : Nontrivial α
    ⊢ Eq (HAdd.hAdd (HSMul.hSMul Polynomial.X 1) (A.map ⇑Polynomial.C)).det.leadin …
  -/
  rw [← @det_one n, ← coeff_det_X_add_C_card _ A, leadingCoeff]
  /-
    case inr
    n : Type u_1
    α : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    h✝ : Nontrivial α
    ⊢ Eq ((HAdd.hAdd (HSMul.hSMul Polynomial.X 1) (A.map ⇑Polynomial.C)).det.coeff …
  -/
  simp only [Matrix.map_one, C_eq_zero, RingHom.map_one]
  /-
    case inr
    n : Type u_1
    α : Type u_2
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : CommRing α
    A : Matrix n n α
    h✝ : Nontrivial α
    ⊢ Eq ((HAdd.hAdd (HSMul.hSMul Polynomial.X 1) (A.map ⇑Polynomial.C)).det.coeff …
  -/
  rcases (natDegree_det_X_add_C_le 1 A).eq_or_lt with h | h
    /-
      case inr.inl
      n : Type u_1
      α : Type u_2
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      A : Matrix n n α
      h✝ : Nontrivial α
      h : Eq (HAdd.hAdd (HSMul.hSMul Polynomial.X (Matrix.map 1 ⇑Polynomial.C)) (A.m …
      ⊢ Eq ((HAdd.hAdd (HSMul.hSMul Polynomial.X 1) (A.map ⇑Polynomial.C)).det.coeff …
    -/
  · simp only [RingHom.map_one, Matrix.map_one, C_eq_zero] at h
    /-
      case inr.inl
      n : Type u_1
      α : Type u_2
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      A : Matrix n n α
      h✝ : Nontrivial α
      h : Eq (HAdd.hAdd (HSMul.hSMul Polynomial.X 1) (A.map ⇑Polynomial.C)).det.natD …
      ⊢ Eq ((HAdd.hAdd (HSMul.hSMul Polynomial.X 1) (A.map ⇑Polynomial.C)).det.coeff …
    -/
    rw [h]
    /-
      🎉 no goals
    -/
  · -- contradiction. we have a hypothesis that the degree is less than |n|
    -- but we know that coeff _ n = 1
    /-
      case inr.inr
      n : Type u_1
      α : Type u_2
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      A : Matrix n n α
      h✝ : Nontrivial α
      h : LT.lt (HAdd.hAdd (HSMul.hSMul Polynomial.X (Matrix.map 1 ⇑Polynomial.C)) ( …
      ⊢ Eq ((HAdd.hAdd (HSMul.hSMul Polynomial.X 1) (A.map ⇑Polynomial.C)).det.coeff …
    -/
    have H := coeff_eq_zero_of_natDegree_lt h
    /-
      case inr.inr
      n : Type u_1
      α : Type u_2
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      A : Matrix n n α
      h✝ : Nontrivial α
      h : LT.lt (HAdd.hAdd (HSMul.hSMul Polynomial.X (Matrix.map 1 ⇑Polynomial.C)) ( …
      H : Eq ((HAdd.hAdd (HSMul.hSMul Polynomial.X (Matrix.map 1 ⇑Polynomial.C)) (A. …
      ⊢ Eq ((HAdd.hAdd (HSMul.hSMul Polynomial.X 1) (A.map ⇑Polynomial.C)).det.coeff …
    -/
    rw [coeff_det_X_add_C_card] at H
    /-
      case inr.inr
      n : Type u_1
      α : Type u_2
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : CommRing α
      A : Matrix n n α
      h✝ : Nontrivial α
      h : LT.lt (HAdd.hAdd (HSMul.hSMul Polynomial.X (Matrix.map 1 ⇑Polynomial.C)) ( …
      H : Eq (Matrix.det 1) 0
      ⊢ Eq ((HAdd.hAdd (HSMul.hSMul Polynomial.X 1) (A.map ⇑Polynomial.C)).det.coeff …
    -/
    simp at H
    /-
      🎉 no goals
    -/


