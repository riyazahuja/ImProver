/-- An action of a group `G` on an extension of rings `B/A` is invariant if every fixed point of
`B` lies in the image of `A`. The converse statement that every point in the image of `A` is fixed
by `G` is `smul_algebraMap` (assuming `SMulCommClass A B G`). -/
@[mk_iff] class IsInvariant : Prop where
  isInvariant : ∀ b : B, (∀ g : G, g • b = b) → ∃ a : A, algebraMap A B a = b


/-- Characteristic polynomial of a finite group action on a ring. -/
noncomputable def charpoly (b : B) : B[X] := ∏ g : G, (X - C (g • b))


theorem charpoly_eq (b : B) : charpoly G b = ∏ g : G, (X - C (g • b)) := rfl


theorem charpoly_eq_prod_smul (b : B) : charpoly G b = ∏ g : G, g • (X - C b) := by
  /-
    B : Type u_2
    G : Type u_3
    inst✝³ : CommRing B
    inst✝² : Group G
    inst✝¹ : MulSemiringAction G B
    inst✝ : Fintype G
    b : B
    ⊢ Eq (MulSemiringAction.charpoly G b) (Finset.univ.prod fun g => HSMul.hSMul g …
  -/
  simp only [smul_sub, smul_C, smul_X, charpoly_eq]
  /-
    🎉 no goals
  -/


theorem monic_charpoly (b : B) : (charpoly G b).Monic :=
  monic_prod_of_monic _ _ (fun _ _ ↦ monic_X_sub_C _)


theorem eval_charpoly (b : B) : (charpoly G b).eval b = 0 := by
  /-
    B : Type u_2
    G : Type u_3
    inst✝³ : CommRing B
    inst✝² : Group G
    inst✝¹ : MulSemiringAction G B
    inst✝ : Fintype G
    b : B
    ⊢ Eq (Polynomial.eval b (MulSemiringAction.charpoly G b)) 0
  -/
  rw [charpoly_eq, eval_prod]
  /-
    B : Type u_2
    G : Type u_3
    inst✝³ : CommRing B
    inst✝² : Group G
    inst✝¹ : MulSemiringAction G B
    inst✝ : Fintype G
    b : B
    ⊢ Eq (Finset.univ.prod fun j => Polynomial.eval b (HSub.hSub Polynomial.X (Pol …
  -/
  apply Finset.prod_eq_zero (Finset.mem_univ (1 : G))
  /-
    B : Type u_2
    G : Type u_3
    inst✝³ : CommRing B
    inst✝² : Group G
    inst✝¹ : MulSemiringAction G B
    inst✝ : Fintype G
    b : B
    ⊢ Eq (Polynomial.eval b (HSub.hSub Polynomial.X (Polynomial.C (HSMul.hSMul 1 b …
  -/
  rw [one_smul, eval_sub, eval_C, eval_X, sub_self]
  /-
    🎉 no goals
  -/


theorem smul_charpoly (b : B) (g : G) : g • (charpoly G b) = charpoly G b := by
  /-
    B : Type u_2
    G : Type u_3
    inst✝³ : CommRing B
    inst✝² : Group G
    inst✝¹ : MulSemiringAction G B
    inst✝ : Fintype G
    b : B
    g : G
    ⊢ Eq (HSMul.hSMul g (MulSemiringAction.charpoly G b)) (MulSemiringAction.charp …
  -/
  rw [charpoly_eq_prod_smul, Finset.smul_prod_perm]
  /-
    🎉 no goals
  -/


theorem smul_coeff_charpoly (b : B) (n : ℕ) (g : G) :
    g • (charpoly G b).coeff n = (charpoly G b).coeff n := by
  /-
    B : Type u_2
    G : Type u_3
    inst✝³ : CommRing B
    inst✝² : Group G
    inst✝¹ : MulSemiringAction G B
    inst✝ : Fintype G
    b : B
    n : Nat
    g : G
    ⊢ Eq (HSMul.hSMul g ((MulSemiringAction.charpoly G b).coeff n)) ((MulSemiringA …
  -/
  rw [← coeff_smul, smul_charpoly]
  /-
    🎉 no goals
  -/


theorem charpoly_mem_lifts [Fintype G] (b : B) :
    charpoly G b ∈ Polynomial.lifts (algebraMap A B) :=
  (charpoly G b).lifts_iff_coeff_lifts.mpr fun n ↦ isInvariant _ (smul_coeff_charpoly b n)


theorem isIntegral [Finite G] : Algebra.IsIntegral A B := by
  /-
    A : Type u_1
    B : Type u_2
    G : Type u_3
    inst✝⁶ : CommRing A
    inst✝⁵ : CommRing B
    inst✝⁴ : Algebra A B
    inst✝³ : Group G
    inst✝² : MulSemiringAction G B
    inst✝¹ : Algebra.IsInvariant A B G
    inst✝ : Finite G
    ⊢ Algebra.IsIntegral A B
  -/
  cases nonempty_fintype G
  /-
    case intro
    A : Type u_1
    B : Type u_2
    G : Type u_3
    inst✝⁶ : CommRing A
    inst✝⁵ : CommRing B
    inst✝⁴ : Algebra A B
    inst✝³ : Group G
    inst✝² : MulSemiringAction G B
    inst✝¹ : Algebra.IsInvariant A B G
    inst✝ : Finite G
    val✝ : Fintype G
    ⊢ Algebra.IsIntegral A B
  -/
  refine ⟨fun b ↦ ?_⟩
  obtain ⟨p, hp1, -, hp2⟩ := Polynomial.lifts_and_natDegree_eq_and_monic
    (charpoly_mem_lifts A B G b) (monic_charpoly G b)
  /-
    case intro.intro.intro.intro
    A : Type u_1
    B : Type u_2
    G : Type u_3
    inst✝⁶ : CommRing A
    inst✝⁵ : CommRing B
    inst✝⁴ : Algebra A B
    inst✝³ : Group G
    inst✝² : MulSemiringAction G B
    inst✝¹ : Algebra.IsInvariant A B G
    inst✝ : Finite G
    val✝ : Fintype G
    b : B
    p : Polynomial A
    hp1 : Eq (Polynomial.map (algebraMap A B) p) (MulSemiringAction.charpoly G b)
    hp2 : p.Monic
    ⊢ IsIntegral A b
  -/
  exact ⟨p, hp2, by rw [← eval_map, hp1, eval_charpoly]⟩
  /-
    🎉 no goals
  -/


/-- `G` acts transitively on the prime ideals of `B` above a given prime ideal of `A`. -/
theorem exists_smul_of_under_eq [Finite G] [SMulCommClass G A B]
    (P Q : Ideal B) [hP : P.IsPrime] [hQ : Q.IsPrime]
    (hPQ : P.under A = Q.under A) :
    ∃ g : G, Q = g • P := by
  /-
    A : Type u_1
    B : Type u_2
    G : Type u_3
    inst✝⁷ : CommRing A
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra A B
    inst✝⁴ : Group G
    inst✝³ : MulSemiringAction G B
    inst✝² : Algebra.IsInvariant A B G
    inst✝¹ : Finite G
    inst✝ : SMulCommClass G A B
    P Q : Ideal B
    hP : P.IsPrime
    hQ : Q.IsPrime
    hPQ : Eq (Ideal.under A P) (Ideal.under A Q)
    ⊢ Exists fun g => Eq Q (HSMul.hSMul g P)
  -/
  cases nonempty_fintype G
  have : ∀ (P Q : Ideal B) [P.IsPrime] [Q.IsPrime], P.under A = Q.under A →
      ∃ g ∈ (⊤ : Finset G), Q ≤ g • P := by
    intro P Q hP hQ hPQ
    rw [← Ideal.subset_union_prime 1 1 (fun _ _ _ _ ↦ hP.smul _)]
    intro b hb
    suffices h : ∃ g ∈ Finset.univ, g • b ∈ P by
      obtain ⟨g, -, hg⟩ := h
      apply Set.mem_biUnion (Finset.mem_univ g⁻¹) (Ideal.mem_inv_pointwise_smul_iff.mpr hg)
    obtain ⟨a, ha⟩ := isInvariant (A := A) (∏ g : G, g • b) (Finset.smul_prod_perm b)
    rw [← hP.prod_mem_iff, ← ha, ← P.mem_comap, ← P.under_def A,
      hPQ, Q.mem_comap, ha, hQ.prod_mem_iff]
    exact ⟨1, Finset.mem_univ 1, (one_smul G b).symm ▸ hb⟩
  /-
    case intro
    A : Type u_1
    B : Type u_2
    G : Type u_3
    inst✝⁷ : CommRing A
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra A B
    inst✝⁴ : Group G
    inst✝³ : MulSemiringAction G B
    inst✝² : Algebra.IsInvariant A B G
    inst✝¹ : Finite G
    inst✝ : SMulCommClass G A B
    P Q : Ideal B
    hP : P.IsPrime
    hQ : Q.IsPrime
    hPQ : Eq (Ideal.under A P) (Ideal.under A Q)
    val✝ : Fintype G
    this : ∀ (P Q : Ideal B) [inst : P.IsPrime] [inst : Q.IsPrime], Eq (Ideal.unde …
    ⊢ Exists fun g => Eq Q (HSMul.hSMul g P)
  -/
  obtain ⟨g, -, hg⟩ := this P Q hPQ
  /-
    case intro.intro.intro
    A : Type u_1
    B : Type u_2
    G : Type u_3
    inst✝⁷ : CommRing A
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra A B
    inst✝⁴ : Group G
    inst✝³ : MulSemiringAction G B
    inst✝² : Algebra.IsInvariant A B G
    inst✝¹ : Finite G
    inst✝ : SMulCommClass G A B
    P Q : Ideal B
    hP : P.IsPrime
    hQ : Q.IsPrime
    hPQ : Eq (Ideal.under A P) (Ideal.under A Q)
    val✝ : Fintype G
    this : ∀ (P Q : Ideal B) [inst : P.IsPrime] [inst : Q.IsPrime], Eq (Ideal.unde …
    g : G
    hg : LE.le Q (HSMul.hSMul g P)
    ⊢ Exists fun g => Eq Q (HSMul.hSMul g P)
  -/
  obtain ⟨g', -, hg'⟩ := this Q (g • P) ((P.under_smul A g).trans hPQ).symm
  /-
    case intro.intro.intro.intro.intro
    A : Type u_1
    B : Type u_2
    G : Type u_3
    inst✝⁷ : CommRing A
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra A B
    inst✝⁴ : Group G
    inst✝³ : MulSemiringAction G B
    inst✝² : Algebra.IsInvariant A B G
    inst✝¹ : Finite G
    inst✝ : SMulCommClass G A B
    P Q : Ideal B
    hP : P.IsPrime
    hQ : Q.IsPrime
    hPQ : Eq (Ideal.under A P) (Ideal.under A Q)
    val✝ : Fintype G
    this : ∀ (P Q : Ideal B) [inst : P.IsPrime] [inst : Q.IsPrime], Eq (Ideal.unde …
    g : G
    hg : LE.le Q (HSMul.hSMul g P)
    g' : G
    hg' : LE.le (HSMul.hSMul g P) (HSMul.hSMul g' Q)
    ⊢ Exists fun g => Eq Q (HSMul.hSMul g P)
  -/
  exact ⟨g, le_antisymm hg (smul_eq_of_le_smul (hg.trans hg') ▸ hg')⟩
  /-
    🎉 no goals
  -/


/-- A technical lemma for `fixed_of_fixed1`. -/
private theorem fixed_of_fixed1_aux1 [DecidableEq (Ideal B)] :
    ∃ a b : B, (∀ g : G, g • a = a) ∧ a ∉ Q ∧
    ∀ g : G, algebraMap B (B ⧸ Q) (g • b) = algebraMap B (B ⧸ Q) (if g • Q = Q then a else 0) := by
  /-
    B : Type u_2
    inst✝⁵ : CommRing B
    G : Type u_3
    inst✝⁴ : Group G
    inst✝³ : Finite G
    inst✝² : MulSemiringAction G B
    Q : Ideal B
    inst✝¹ : Q.IsPrime
    inst✝ : DecidableEq (Ideal B)
    ⊢ Exists fun a => Exists fun b => And (∀ (g : G), Eq (HSMul.hSMul g a) a) (And …
  -/
  obtain ⟨_⟩ := nonempty_fintype G
  /-
    case intro
    B : Type u_2
    inst✝⁵ : CommRing B
    G : Type u_3
    inst✝⁴ : Group G
    inst✝³ : Finite G
    inst✝² : MulSemiringAction G B
    Q : Ideal B
    inst✝¹ : Q.IsPrime
    inst✝ : DecidableEq (Ideal B)
    val✝ : Fintype G
    ⊢ Exists fun a => Exists fun b => And (∀ (g : G), Eq (HSMul.hSMul g a) a) (And …
  -/
  let P := ((Finset.univ : Finset G).filter (fun g ↦ g • Q ≠ Q)).inf (fun g ↦ g • Q)
  have h1 : ¬ P ≤ Q := by
    rw [Ideal.IsPrime.inf_le' inferInstance]
    rintro ⟨g, hg1, hg2⟩
    exact (Finset.mem_filter.mp hg1).2 (smul_eq_of_smul_le hg2)
  /-
    case intro
    B : Type u_2
    inst✝⁵ : CommRing B
    G : Type u_3
    inst✝⁴ : Group G
    inst✝³ : Finite G
    inst✝² : MulSemiringAction G B
    Q : Ideal B
    inst✝¹ : Q.IsPrime
    inst✝ : DecidableEq (Ideal B)
    val✝ : Fintype G
    P : Ideal B := (Finset.filter (fun g => Ne (HSMul.hSMul g Q) Q) Finset.univ).i …
    h1 : Not (LE.le P Q)
    ⊢ Exists fun a => Exists fun b => And (∀ (g : G), Eq (HSMul.hSMul g a) a) (And …
  -/
  obtain ⟨b, hbP, hbQ⟩ := SetLike.not_le_iff_exists.mp h1
  replace hbP : ∀ g : G, g • Q ≠ Q → b ∈ g • Q :=
    fun g hg ↦ (Finset.inf_le (Finset.mem_filter.mpr ⟨Finset.mem_univ g, hg⟩) : P ≤ g • Q) hbP
  /-
    case intro.intro.intro
    B : Type u_2
    inst✝⁵ : CommRing B
    G : Type u_3
    inst✝⁴ : Group G
    inst✝³ : Finite G
    inst✝² : MulSemiringAction G B
    Q : Ideal B
    inst✝¹ : Q.IsPrime
    inst✝ : DecidableEq (Ideal B)
    val✝ : Fintype G
    P : Ideal B := (Finset.filter (fun g => Ne (HSMul.hSMul g Q) Q) Finset.univ).i …
    h1 : Not (LE.le P Q)
    b : B
    hbQ : Not (Membership.mem Q b)
    hbP : ∀ (g : G), Ne (HSMul.hSMul g Q) Q → Membership.mem (HSMul.hSMul g Q) b
    ⊢ Exists fun a => Exists fun b => And (∀ (g : G), Eq (HSMul.hSMul g a) a) (And …
  -/
  let f := MulSemiringAction.charpoly G b
  obtain ⟨q, hq, hq0⟩ :=
    (f.map (algebraMap B (B ⧸ Q))).exists_eq_pow_rootMultiplicity_mul_and_not_dvd
      (Polynomial.map_monic_ne_zero (MulSemiringAction.monic_charpoly G b)) 0
  /-
    case intro.intro.intro.intro.intro
    B : Type u_2
    inst✝⁵ : CommRing B
    G : Type u_3
    inst✝⁴ : Group G
    inst✝³ : Finite G
    inst✝² : MulSemiringAction G B
    Q : Ideal B
    inst✝¹ : Q.IsPrime
    inst✝ : DecidableEq (Ideal B)
    val✝ : Fintype G
    P : Ideal B := (Finset.filter (fun g => Ne (HSMul.hSMul g Q) Q) Finset.univ).i …
    h1 : Not (LE.le P Q)
    b : B
    hbQ : Not (Membership.mem Q b)
    hbP : ∀ (g : G), Ne (HSMul.hSMul g Q) Q → Membership.mem (HSMul.hSMul g Q) b
    f : Polynomial B := MulSemiringAction.charpoly G b
    q : Polynomial (HasQuotient.Quotient B Q)
    hq : Eq (Polynomial.map (algebraMap B (HasQuotient.Quotient B Q)) f) (HMul.hMu …
    hq0 : Not (Dvd.dvd (HSub.hSub Polynomial.X (Polynomial.C 0)) q)
    ⊢ Exists fun a => Exists fun b => And (∀ (g : G), Eq (HSMul.hSMul g a) a) (And …
  -/
  rw [map_zero, sub_zero] at hq hq0
  /-
    case intro.intro.intro.intro.intro
    B : Type u_2
    inst✝⁵ : CommRing B
    G : Type u_3
    inst✝⁴ : Group G
    inst✝³ : Finite G
    inst✝² : MulSemiringAction G B
    Q : Ideal B
    inst✝¹ : Q.IsPrime
    inst✝ : DecidableEq (Ideal B)
    val✝ : Fintype G
    P : Ideal B := (Finset.filter (fun g => Ne (HSMul.hSMul g Q) Q) Finset.univ).i …
    h1 : Not (LE.le P Q)
    b : B
    hbQ : Not (Membership.mem Q b)
    hbP : ∀ (g : G), Ne (HSMul.hSMul g Q) Q → Membership.mem (HSMul.hSMul g Q) b
    f : Polynomial B := MulSemiringAction.charpoly G b
    q : Polynomial (HasQuotient.Quotient B Q)
    hq : Eq (Polynomial.map (algebraMap B (HasQuotient.Quotient B Q)) f) (HMul.hMu …
    hq0 : Not (Dvd.dvd Polynomial.X q)
    ⊢ Exists fun a => Exists fun b => And (∀ (g : G), Eq (HSMul.hSMul g a) a) (And …
  -/
  let j := (f.map (algebraMap B (B ⧸ Q))).rootMultiplicity 0
  /-
    case intro.intro.intro.intro.intro
    B : Type u_2
    inst✝⁵ : CommRing B
    G : Type u_3
    inst✝⁴ : Group G
    inst✝³ : Finite G
    inst✝² : MulSemiringAction G B
    Q : Ideal B
    inst✝¹ : Q.IsPrime
    inst✝ : DecidableEq (Ideal B)
    val✝ : Fintype G
    P : Ideal B := (Finset.filter (fun g => Ne (HSMul.hSMul g Q) Q) Finset.univ).i …
    h1 : Not (LE.le P Q)
    b : B
    hbQ : Not (Membership.mem Q b)
    hbP : ∀ (g : G), Ne (HSMul.hSMul g Q) Q → Membership.mem (HSMul.hSMul g Q) b
    f : Polynomial B := MulSemiringAction.charpoly G b
    q : Polynomial (HasQuotient.Quotient B Q)
    hq : Eq (Polynomial.map (algebraMap B (HasQuotient.Quotient B Q)) f) (HMul.hMu …
    hq0 : Not (Dvd.dvd Polynomial.X q)
    j : Nat := Polynomial.rootMultiplicity 0 (Polynomial.map (algebraMap B (HasQuo …
    ⊢ Exists fun a => Exists fun b => And (∀ (g : G), Eq (HSMul.hSMul g a) a) (And …
  -/
  let k := q.natDegree
  /-
    case intro.intro.intro.intro.intro
    B : Type u_2
    inst✝⁵ : CommRing B
    G : Type u_3
    inst✝⁴ : Group G
    inst✝³ : Finite G
    inst✝² : MulSemiringAction G B
    Q : Ideal B
    inst✝¹ : Q.IsPrime
    inst✝ : DecidableEq (Ideal B)
    val✝ : Fintype G
    P : Ideal B := (Finset.filter (fun g => Ne (HSMul.hSMul g Q) Q) Finset.univ).i …
    h1 : Not (LE.le P Q)
    b : B
    hbQ : Not (Membership.mem Q b)
    hbP : ∀ (g : G), Ne (HSMul.hSMul g Q) Q → Membership.mem (HSMul.hSMul g Q) b
    f : Polynomial B := MulSemiringAction.charpoly G b
    q : Polynomial (HasQuotient.Quotient B Q)
    hq : Eq (Polynomial.map (algebraMap B (HasQuotient.Quotient B Q)) f) (HMul.hMu …
    hq0 : Not (Dvd.dvd Polynomial.X q)
    j : Nat := Polynomial.rootMultiplicity 0 (Polynomial.map (algebraMap B (HasQuo …
    k : Nat := q.natDegree
    ⊢ Exists fun a => Exists fun b => And (∀ (g : G), Eq (HSMul.hSMul g a) a) (And …
  -/
  let r := ∑ i ∈ Finset.range (k + 1), Polynomial.monomial i (f.coeff (i + j))
  have hr : r.map (algebraMap B (B ⧸ Q)) = q := by
    ext n
    rw [Polynomial.coeff_map, Polynomial.finset_sum_coeff]
    simp only [Polynomial.coeff_monomial, Finset.sum_ite_eq', Finset.mem_range_succ_iff]
    split_ifs with hn
    · rw [← Polynomial.coeff_map, hq, Polynomial.coeff_X_pow_mul]
    · rw [map_zero, eq_comm, Polynomial.coeff_eq_zero_of_natDegree_lt (lt_of_not_le hn)]
  /-
    case intro.intro.intro.intro.intro
    B : Type u_2
    inst✝⁵ : CommRing B
    G : Type u_3
    inst✝⁴ : Group G
    inst✝³ : Finite G
    inst✝² : MulSemiringAction G B
    Q : Ideal B
    inst✝¹ : Q.IsPrime
    inst✝ : DecidableEq (Ideal B)
    val✝ : Fintype G
    P : Ideal B := (Finset.filter (fun g => Ne (HSMul.hSMul g Q) Q) Finset.univ).i …
    h1 : Not (LE.le P Q)
    b : B
    hbQ : Not (Membership.mem Q b)
    hbP : ∀ (g : G), Ne (HSMul.hSMul g Q) Q → Membership.mem (HSMul.hSMul g Q) b
    f : Polynomial B := MulSemiringAction.charpoly G b
    q : Polynomial (HasQuotient.Quotient B Q)
    hq : Eq (Polynomial.map (algebraMap B (HasQuotient.Quotient B Q)) f) (HMul.hMu …
    hq0 : Not (Dvd.dvd Polynomial.X q)
    j : Nat := Polynomial.rootMultiplicity 0 (Polynomial.map (algebraMap B (HasQuo …
    k : Nat := q.natDegree
    r : Polynomial B := (Finset.range (HAdd.hAdd k 1)).sum fun i => (Polynomial.mo …
    hr : Eq (Polynomial.map (algebraMap B (HasQuotient.Quotient B Q)) r) q
    ⊢ Exists fun a => Exists fun b => And (∀ (g : G), Eq (HSMul.hSMul g a) a) (And …
  -/
  have hf : f.eval b = 0 := MulSemiringAction.eval_charpoly G b
  have hr : r.eval b ∈ Q := by
    rw [← Ideal.Quotient.eq_zero_iff_mem, ← Ideal.Quotient.algebraMap_eq] at hbQ ⊢
    replace hf := congrArg (algebraMap B (B ⧸ Q)) hf
    rw [← Polynomial.eval₂_at_apply, ← Polynomial.eval_map] at hf ⊢
    rwa [map_zero, hq, ← hr, Polynomial.eval_mul, Polynomial.eval_pow, Polynomial.eval_X,
      mul_eq_zero, or_iff_right (pow_ne_zero _ hbQ)] at hf
  /-
    case intro.intro.intro.intro.intro
    B : Type u_2
    inst✝⁵ : CommRing B
    G : Type u_3
    inst✝⁴ : Group G
    inst✝³ : Finite G
    inst✝² : MulSemiringAction G B
    Q : Ideal B
    inst✝¹ : Q.IsPrime
    inst✝ : DecidableEq (Ideal B)
    val✝ : Fintype G
    P : Ideal B := (Finset.filter (fun g => Ne (HSMul.hSMul g Q) Q) Finset.univ).i …
    h1 : Not (LE.le P Q)
    b : B
    hbQ : Not (Membership.mem Q b)
    hbP : ∀ (g : G), Ne (HSMul.hSMul g Q) Q → Membership.mem (HSMul.hSMul g Q) b
    f : Polynomial B := MulSemiringAction.charpoly G b
    q : Polynomial (HasQuotient.Quotient B Q)
    hq : Eq (Polynomial.map (algebraMap B (HasQuotient.Quotient B Q)) f) (HMul.hMu …
    hq0 : Not (Dvd.dvd Polynomial.X q)
    j : Nat := Polynomial.rootMultiplicity 0 (Polynomial.map (algebraMap B (HasQuo …
    k : Nat := q.natDegree
    r : Polynomial B := (Finset.range (HAdd.hAdd k 1)).sum fun i => (Polynomial.mo …
    hr✝ : Eq (Polynomial.map (algebraMap B (HasQuotient.Quotient B Q)) r) q
    hf : Eq (Polynomial.eval b f) 0
    hr : Membership.mem Q (Polynomial.eval b r)
    ⊢ Exists fun a => Exists fun b => And (∀ (g : G), Eq (HSMul.hSMul g a) a) (And …
  -/
  let a := f.coeff j
  /-
    case intro.intro.intro.intro.intro
    B : Type u_2
    inst✝⁵ : CommRing B
    G : Type u_3
    inst✝⁴ : Group G
    inst✝³ : Finite G
    inst✝² : MulSemiringAction G B
    Q : Ideal B
    inst✝¹ : Q.IsPrime
    inst✝ : DecidableEq (Ideal B)
    val✝ : Fintype G
    P : Ideal B := (Finset.filter (fun g => Ne (HSMul.hSMul g Q) Q) Finset.univ).i …
    h1 : Not (LE.le P Q)
    b : B
    hbQ : Not (Membership.mem Q b)
    hbP : ∀ (g : G), Ne (HSMul.hSMul g Q) Q → Membership.mem (HSMul.hSMul g Q) b
    f : Polynomial B := MulSemiringAction.charpoly G b
    q : Polynomial (HasQuotient.Quotient B Q)
    hq : Eq (Polynomial.map (algebraMap B (HasQuotient.Quotient B Q)) f) (HMul.hMu …
    hq0 : Not (Dvd.dvd Polynomial.X q)
    j : Nat := Polynomial.rootMultiplicity 0 (Polynomial.map (algebraMap B (HasQuo …
    k : Nat := q.natDegree
    r : Polynomial B := (Finset.range (HAdd.hAdd k 1)).sum fun i => (Polynomial.mo …
    hr✝ : Eq (Polynomial.map (algebraMap B (HasQuotient.Quotient B Q)) r) q
    hf : Eq (Polynomial.eval b f) 0
    hr : Membership.mem Q (Polynomial.eval b r)
    a : B := f.coeff j
    ⊢ Exists fun a => Exists fun b => And (∀ (g : G), Eq (HSMul.hSMul g a) a) (And …
  -/
  have ha : ∀ g : G, g • a = a := MulSemiringAction.smul_coeff_charpoly b j
  have hr' : ∀ g : G, g • Q ≠ Q → a - r.eval b ∈ g • Q := by
    intro g hg
    have hr : r = ∑ i ∈ Finset.range (k + 1), Polynomial.monomial i (f.coeff (i + j)) := rfl
    rw [← Ideal.neg_mem_iff, neg_sub, hr, Finset.sum_range_succ', Polynomial.eval_add,
        Polynomial.eval_monomial, zero_add, pow_zero, mul_one, add_sub_cancel_right]
    simp only [ ← Polynomial.monomial_mul_X]
    rw [← Finset.sum_mul, Polynomial.eval_mul_X]
    exact Ideal.mul_mem_left (g • Q) _ (hbP g hg)
  /-
    case intro.intro.intro.intro.intro
    B : Type u_2
    inst✝⁵ : CommRing B
    G : Type u_3
    inst✝⁴ : Group G
    inst✝³ : Finite G
    inst✝² : MulSemiringAction G B
    Q : Ideal B
    inst✝¹ : Q.IsPrime
    inst✝ : DecidableEq (Ideal B)
    val✝ : Fintype G
    P : Ideal B := (Finset.filter (fun g => Ne (HSMul.hSMul g Q) Q) Finset.univ).i …
    h1 : Not (LE.le P Q)
    b : B
    hbQ : Not (Membership.mem Q b)
    hbP : ∀ (g : G), Ne (HSMul.hSMul g Q) Q → Membership.mem (HSMul.hSMul g Q) b
    f : Polynomial B := MulSemiringAction.charpoly G b
    q : Polynomial (HasQuotient.Quotient B Q)
    hq : Eq (Polynomial.map (algebraMap B (HasQuotient.Quotient B Q)) f) (HMul.hMu …
    hq0 : Not (Dvd.dvd Polynomial.X q)
    j : Nat := Polynomial.rootMultiplicity 0 (Polynomial.map (algebraMap B (HasQuo …
    k : Nat := q.natDegree
    r : Polynomial B := (Finset.range (HAdd.hAdd k 1)).sum fun i => (Polynomial.mo …
    hr✝ : Eq (Polynomial.map (algebraMap B (HasQuotient.Quotient B Q)) r) q
    hf : Eq (Polynomial.eval b f) 0
    hr : Membership.mem Q (Polynomial.eval b r)
    a : B := f.coeff j
    ha : ∀ (g : G), Eq (HSMul.hSMul g a) a
    hr' : ∀ (g : G), Ne (HSMul.hSMul g Q) Q → Membership.mem (HSMul.hSMul g Q) (HS …
    ⊢ Exists fun a => Exists fun b => And (∀ (g : G), Eq (HSMul.hSMul g a) a) (And …
  -/
  refine ⟨a, a - r.eval b, ha, ?_, fun h ↦ ?_⟩
  · rwa [← Ideal.Quotient.eq_zero_iff_mem, ← Ideal.Quotient.algebraMap_eq, ← Polynomial.coeff_map,
      ← zero_add j, hq, Polynomial.coeff_X_pow_mul, ← Polynomial.X_dvd_iff]
  · rw [← sub_eq_zero, ← map_sub, Ideal.Quotient.algebraMap_eq, Ideal.Quotient.eq_zero_iff_mem,
      ← Ideal.smul_mem_pointwise_smul_iff (a := h⁻¹), smul_sub, inv_smul_smul]
    /-
      case intro.intro.intro.intro.intro.refine_2
      B : Type u_2
      inst✝⁵ : CommRing B
      G : Type u_3
      inst✝⁴ : Group G
      inst✝³ : Finite G
      inst✝² : MulSemiringAction G B
      Q : Ideal B
      inst✝¹ : Q.IsPrime
      inst✝ : DecidableEq (Ideal B)
      val✝ : Fintype G
      P : Ideal B := (Finset.filter (fun g => Ne (HSMul.hSMul g Q) Q) Finset.univ).i …
      h1 : Not (LE.le P Q)
      b : B
      hbQ : Not (Membership.mem Q b)
      hbP : ∀ (g : G), Ne (HSMul.hSMul g Q) Q → Membership.mem (HSMul.hSMul g Q) b
      f : Polynomial B := MulSemiringAction.charpoly G b
      q : Polynomial (HasQuotient.Quotient B Q)
      hq : Eq (Polynomial.map (algebraMap B (HasQuotient.Quotient B Q)) f) (HMul.hMu …
      hq0 : Not (Dvd.dvd Polynomial.X q)
      j : Nat := Polynomial.rootMultiplicity 0 (Polynomial.map (algebraMap B (HasQuo …
      k : Nat := q.natDegree
      r : Polynomial B := (Finset.range (HAdd.hAdd k 1)).sum fun i => (Polynomial.mo …
      hr✝ : Eq (Polynomial.map (algebraMap B (HasQuotient.Quotient B Q)) r) q
      hf : Eq (Polynomial.eval b f) 0
      hr : Membership.mem Q (Polynomial.eval b r)
      a : B := f.coeff j
      ha : ∀ (g : G), Eq (HSMul.hSMul g a) a
      hr' : ∀ (g : G), Ne (HSMul.hSMul g Q) Q → Membership.mem (HSMul.hSMul g Q) (HS …
      h : G
      ⊢ Membership.mem (HSMul.hSMul (Inv.inv h) Q) (HSub.hSub (HSub.hSub a (Polynomi …
    -/
    simp only [← eq_inv_smul_iff (g := h), eq_comm (a := Q)]
    /-
      case intro.intro.intro.intro.intro.refine_2
      B : Type u_2
      inst✝⁵ : CommRing B
      G : Type u_3
      inst✝⁴ : Group G
      inst✝³ : Finite G
      inst✝² : MulSemiringAction G B
      Q : Ideal B
      inst✝¹ : Q.IsPrime
      inst✝ : DecidableEq (Ideal B)
      val✝ : Fintype G
      P : Ideal B := (Finset.filter (fun g => Ne (HSMul.hSMul g Q) Q) Finset.univ).i …
      h1 : Not (LE.le P Q)
      b : B
      hbQ : Not (Membership.mem Q b)
      hbP : ∀ (g : G), Ne (HSMul.hSMul g Q) Q → Membership.mem (HSMul.hSMul g Q) b
      f : Polynomial B := MulSemiringAction.charpoly G b
      q : Polynomial (HasQuotient.Quotient B Q)
      hq : Eq (Polynomial.map (algebraMap B (HasQuotient.Quotient B Q)) f) (HMul.hMu …
      hq0 : Not (Dvd.dvd Polynomial.X q)
      j : Nat := Polynomial.rootMultiplicity 0 (Polynomial.map (algebraMap B (HasQuo …
      k : Nat := q.natDegree
      r : Polynomial B := (Finset.range (HAdd.hAdd k 1)).sum fun i => (Polynomial.mo …
      hr✝ : Eq (Polynomial.map (algebraMap B (HasQuotient.Quotient B Q)) r) q
      hf : Eq (Polynomial.eval b f) 0
      hr : Membership.mem Q (Polynomial.eval b r)
      a : B := f.coeff j
      ha : ∀ (g : G), Eq (HSMul.hSMul g a) a
      hr' : ∀ (g : G), Ne (HSMul.hSMul g Q) Q → Membership.mem (HSMul.hSMul g Q) (HS …
      h : G
      ⊢ Membership.mem (HSMul.hSMul (Inv.inv h) Q) (HSub.hSub (HSub.hSub a (Polynomi …
    -/
    split_ifs with hh
      /-
        case pos
        B : Type u_2
        inst✝⁵ : CommRing B
        G : Type u_3
        inst✝⁴ : Group G
        inst✝³ : Finite G
        inst✝² : MulSemiringAction G B
        Q : Ideal B
        inst✝¹ : Q.IsPrime
        inst✝ : DecidableEq (Ideal B)
        val✝ : Fintype G
        P : Ideal B := (Finset.filter (fun g => Ne (HSMul.hSMul g Q) Q) Finset.univ).i …
        h1 : Not (LE.le P Q)
        b : B
        hbQ : Not (Membership.mem Q b)
        hbP : ∀ (g : G), Ne (HSMul.hSMul g Q) Q → Membership.mem (HSMul.hSMul g Q) b
        f : Polynomial B := MulSemiringAction.charpoly G b
        q : Polynomial (HasQuotient.Quotient B Q)
        hq : Eq (Polynomial.map (algebraMap B (HasQuotient.Quotient B Q)) f) (HMul.hMu …
        hq0 : Not (Dvd.dvd Polynomial.X q)
        j : Nat := Polynomial.rootMultiplicity 0 (Polynomial.map (algebraMap B (HasQuo …
        k : Nat := q.natDegree
        r : Polynomial B := (Finset.range (HAdd.hAdd k 1)).sum fun i => (Polynomial.mo …
        hr✝ : Eq (Polynomial.map (algebraMap B (HasQuotient.Quotient B Q)) r) q
        hf : Eq (Polynomial.eval b f) 0
        hr : Membership.mem Q (Polynomial.eval b r)
        a : B := f.coeff j
        ha : ∀ (g : G), Eq (HSMul.hSMul g a) a
        hr' : ∀ (g : G), Ne (HSMul.hSMul g Q) Q → Membership.mem (HSMul.hSMul g Q) (HS …
        h : G
        hh : Eq (HSMul.hSMul (Inv.inv h) Q) Q
        ⊢ Membership.mem (HSMul.hSMul (Inv.inv h) Q) (HSub.hSub (HSub.hSub a (Polynomi …
      -/
    · rwa [ha, sub_sub_cancel_left, hh, Q.neg_mem_iff]
      /-
        🎉 no goals
      -/
      /-
        case neg
        B : Type u_2
        inst✝⁵ : CommRing B
        G : Type u_3
        inst✝⁴ : Group G
        inst✝³ : Finite G
        inst✝² : MulSemiringAction G B
        Q : Ideal B
        inst✝¹ : Q.IsPrime
        inst✝ : DecidableEq (Ideal B)
        val✝ : Fintype G
        P : Ideal B := (Finset.filter (fun g => Ne (HSMul.hSMul g Q) Q) Finset.univ).i …
        h1 : Not (LE.le P Q)
        b : B
        hbQ : Not (Membership.mem Q b)
        hbP : ∀ (g : G), Ne (HSMul.hSMul g Q) Q → Membership.mem (HSMul.hSMul g Q) b
        f : Polynomial B := MulSemiringAction.charpoly G b
        q : Polynomial (HasQuotient.Quotient B Q)
        hq : Eq (Polynomial.map (algebraMap B (HasQuotient.Quotient B Q)) f) (HMul.hMu …
        hq0 : Not (Dvd.dvd Polynomial.X q)
        j : Nat := Polynomial.rootMultiplicity 0 (Polynomial.map (algebraMap B (HasQuo …
        k : Nat := q.natDegree
        r : Polynomial B := (Finset.range (HAdd.hAdd k 1)).sum fun i => (Polynomial.mo …
        hr✝ : Eq (Polynomial.map (algebraMap B (HasQuotient.Quotient B Q)) r) q
        hf : Eq (Polynomial.eval b f) 0
        hr : Membership.mem Q (Polynomial.eval b r)
        a : B := f.coeff j
        ha : ∀ (g : G), Eq (HSMul.hSMul g a) a
        hr' : ∀ (g : G), Ne (HSMul.hSMul g Q) Q → Membership.mem (HSMul.hSMul g Q) (HS …
        h : G
        hh : Not (Eq (HSMul.hSMul (Inv.inv h) Q) Q)
        ⊢ Membership.mem (HSMul.hSMul (Inv.inv h) Q) (HSub.hSub (HSub.hSub a (Polynomi …
      -/
    · rw [smul_zero, sub_zero]
      /-
        case neg
        B : Type u_2
        inst✝⁵ : CommRing B
        G : Type u_3
        inst✝⁴ : Group G
        inst✝³ : Finite G
        inst✝² : MulSemiringAction G B
        Q : Ideal B
        inst✝¹ : Q.IsPrime
        inst✝ : DecidableEq (Ideal B)
        val✝ : Fintype G
        P : Ideal B := (Finset.filter (fun g => Ne (HSMul.hSMul g Q) Q) Finset.univ).i …
        h1 : Not (LE.le P Q)
        b : B
        hbQ : Not (Membership.mem Q b)
        hbP : ∀ (g : G), Ne (HSMul.hSMul g Q) Q → Membership.mem (HSMul.hSMul g Q) b
        f : Polynomial B := MulSemiringAction.charpoly G b
        q : Polynomial (HasQuotient.Quotient B Q)
        hq : Eq (Polynomial.map (algebraMap B (HasQuotient.Quotient B Q)) f) (HMul.hMu …
        hq0 : Not (Dvd.dvd Polynomial.X q)
        j : Nat := Polynomial.rootMultiplicity 0 (Polynomial.map (algebraMap B (HasQuo …
        k : Nat := q.natDegree
        r : Polynomial B := (Finset.range (HAdd.hAdd k 1)).sum fun i => (Polynomial.mo …
        hr✝ : Eq (Polynomial.map (algebraMap B (HasQuotient.Quotient B Q)) r) q
        hf : Eq (Polynomial.eval b f) 0
        hr : Membership.mem Q (Polynomial.eval b r)
        a : B := f.coeff j
        ha : ∀ (g : G), Eq (HSMul.hSMul g a) a
        hr' : ∀ (g : G), Ne (HSMul.hSMul g Q) Q → Membership.mem (HSMul.hSMul g Q) (HS …
        h : G
        hh : Not (Eq (HSMul.hSMul (Inv.inv h) Q) Q)
        ⊢ Membership.mem (HSMul.hSMul (Inv.inv h) Q) (HSub.hSub a (Polynomial.eval b r))
      -/
      exact hr' h⁻¹ hh
      /-
        🎉 no goals
      -/


/-- A technical lemma for `fixed_of_fixed1`. -/
private theorem fixed_of_fixed1_aux2 [DecidableEq (Ideal B)] (b₀ : B)
    (hx : ∀ g : G, g • Q = Q → algebraMap B (B ⧸ Q) (g • b₀) = algebraMap B (B ⧸ Q) b₀) :
    ∃ a b : B, (∀ g : G, g • a = a) ∧ a ∉ Q ∧
    (∀ g : G, algebraMap B (B ⧸ Q) (g • b) =
      algebraMap B (B ⧸ Q) (if g • Q = Q then a * b₀ else 0)) := by
  /-
    B : Type u_2
    inst✝⁵ : CommRing B
    G : Type u_3
    inst✝⁴ : Group G
    inst✝³ : Finite G
    inst✝² : MulSemiringAction G B
    Q : Ideal B
    inst✝¹ : Q.IsPrime
    inst✝ : DecidableEq (Ideal B)
    b₀ : B
    hx : ∀ (g : G), Eq (HSMul.hSMul g Q) Q → Eq ((algebraMap B (HasQuotient.Quotie …
    ⊢ Exists fun a => Exists fun b => And (∀ (g : G), Eq (HSMul.hSMul g a) a) (And …
  -/
  obtain ⟨a, b, ha1, ha2, hb⟩ := fixed_of_fixed1_aux1 G Q
  /-
    case intro.intro.intro.intro
    B : Type u_2
    inst✝⁵ : CommRing B
    G : Type u_3
    inst✝⁴ : Group G
    inst✝³ : Finite G
    inst✝² : MulSemiringAction G B
    Q : Ideal B
    inst✝¹ : Q.IsPrime
    inst✝ : DecidableEq (Ideal B)
    b₀ : B
    hx : ∀ (g : G), Eq (HSMul.hSMul g Q) Q → Eq ((algebraMap B (HasQuotient.Quotie …
    a b : B
    ha1 : ∀ (g : G), Eq (HSMul.hSMul g a) a
    ha2 : Not (Membership.mem Q a)
    hb : ∀ (g : G), Eq ((algebraMap B (HasQuotient.Quotient B Q)) (HSMul.hSMul g b …
    ⊢ Exists fun a => Exists fun b => And (∀ (g : G), Eq (HSMul.hSMul g a) a) (And …
  -/
  refine ⟨a, b * b₀, ha1, ha2, fun g ↦ ?_⟩
  /-
    case intro.intro.intro.intro
    B : Type u_2
    inst✝⁵ : CommRing B
    G : Type u_3
    inst✝⁴ : Group G
    inst✝³ : Finite G
    inst✝² : MulSemiringAction G B
    Q : Ideal B
    inst✝¹ : Q.IsPrime
    inst✝ : DecidableEq (Ideal B)
    b₀ : B
    hx : ∀ (g : G), Eq (HSMul.hSMul g Q) Q → Eq ((algebraMap B (HasQuotient.Quotie …
    a b : B
    ha1 : ∀ (g : G), Eq (HSMul.hSMul g a) a
    ha2 : Not (Membership.mem Q a)
    hb : ∀ (g : G), Eq ((algebraMap B (HasQuotient.Quotient B Q)) (HSMul.hSMul g b …
    g : G
    ⊢ Eq ((algebraMap B (HasQuotient.Quotient B Q)) (HSMul.hSMul g (HMul.hMul b b₀ …
  -/
  rw [smul_mul', map_mul, hb]
  /-
    case intro.intro.intro.intro
    B : Type u_2
    inst✝⁵ : CommRing B
    G : Type u_3
    inst✝⁴ : Group G
    inst✝³ : Finite G
    inst✝² : MulSemiringAction G B
    Q : Ideal B
    inst✝¹ : Q.IsPrime
    inst✝ : DecidableEq (Ideal B)
    b₀ : B
    hx : ∀ (g : G), Eq (HSMul.hSMul g Q) Q → Eq ((algebraMap B (HasQuotient.Quotie …
    a b : B
    ha1 : ∀ (g : G), Eq (HSMul.hSMul g a) a
    ha2 : Not (Membership.mem Q a)
    hb : ∀ (g : G), Eq ((algebraMap B (HasQuotient.Quotient B Q)) (HSMul.hSMul g b …
    g : G
    ⊢ Eq (HMul.hMul ((algebraMap B (HasQuotient.Quotient B Q)) (ite (Eq (HSMul.hSM …
  -/
  specialize hb g
  /-
    case intro.intro.intro.intro
    B : Type u_2
    inst✝⁵ : CommRing B
    G : Type u_3
    inst✝⁴ : Group G
    inst✝³ : Finite G
    inst✝² : MulSemiringAction G B
    Q : Ideal B
    inst✝¹ : Q.IsPrime
    inst✝ : DecidableEq (Ideal B)
    b₀ : B
    hx : ∀ (g : G), Eq (HSMul.hSMul g Q) Q → Eq ((algebraMap B (HasQuotient.Quotie …
    a b : B
    ha1 : ∀ (g : G), Eq (HSMul.hSMul g a) a
    ha2 : Not (Membership.mem Q a)
    g : G
    hb : Eq ((algebraMap B (HasQuotient.Quotient B Q)) (HSMul.hSMul g b)) ((algebr …
    ⊢ Eq (HMul.hMul ((algebraMap B (HasQuotient.Quotient B Q)) (ite (Eq (HSMul.hSM …
  -/
  split_ifs with hg
    /-
      case pos
      B : Type u_2
      inst✝⁵ : CommRing B
      G : Type u_3
      inst✝⁴ : Group G
      inst✝³ : Finite G
      inst✝² : MulSemiringAction G B
      Q : Ideal B
      inst✝¹ : Q.IsPrime
      inst✝ : DecidableEq (Ideal B)
      b₀ : B
      hx : ∀ (g : G), Eq (HSMul.hSMul g Q) Q → Eq ((algebraMap B (HasQuotient.Quotie …
      a b : B
      ha1 : ∀ (g : G), Eq (HSMul.hSMul g a) a
      ha2 : Not (Membership.mem Q a)
      g : G
      hb : Eq ((algebraMap B (HasQuotient.Quotient B Q)) (HSMul.hSMul g b)) ((algebr …
      hg : Eq (HSMul.hSMul g Q) Q
      ⊢ Eq (HMul.hMul ((algebraMap B (HasQuotient.Quotient B Q)) a) ((algebraMap B ( …
    -/
  · rw [map_mul, hx g hg]
    /-
      🎉 no goals
    -/
    /-
      case neg
      B : Type u_2
      inst✝⁵ : CommRing B
      G : Type u_3
      inst✝⁴ : Group G
      inst✝³ : Finite G
      inst✝² : MulSemiringAction G B
      Q : Ideal B
      inst✝¹ : Q.IsPrime
      inst✝ : DecidableEq (Ideal B)
      b₀ : B
      hx : ∀ (g : G), Eq (HSMul.hSMul g Q) Q → Eq ((algebraMap B (HasQuotient.Quotie …
      a b : B
      ha1 : ∀ (g : G), Eq (HSMul.hSMul g a) a
      ha2 : Not (Membership.mem Q a)
      g : G
      hb : Eq ((algebraMap B (HasQuotient.Quotient B Q)) (HSMul.hSMul g b)) ((algebr …
      hg : Not (Eq (HSMul.hSMul g Q) Q)
      ⊢ Eq (HMul.hMul ((algebraMap B (HasQuotient.Quotient B Q)) 0) ((algebraMap B ( …
    -/
  · rw [map_zero, zero_mul]
    /-
      🎉 no goals
    -/


/-- A technical lemma for `fixed_of_fixed1`. -/
private theorem fixed_of_fixed1_aux3 [NoZeroDivisors B] {b : B} {i j : ℕ} {p : Polynomial A}
    (h : p.map (algebraMap A B) = (X - C b) ^ i * X ^ j) (f : B ≃ₐ[A] B) (hi : i ≠ 0) :
    f b = b := by
  /-
    A : Type u_1
    B : Type u_2
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : Algebra A B
    inst✝ : NoZeroDivisors B
    b : B
    i j : Nat
    p : Polynomial A
    h : Eq (Polynomial.map (algebraMap A B) p) (HMul.hMul (HPow.hPow (HSub.hSub Po …
    f : AlgEquiv A B B
    hi : Ne i 0
    ⊢ Eq (f b) b
  -/
  by_cases ha : b = 0
    /-
      case pos
      A : Type u_1
      B : Type u_2
      inst✝³ : CommRing A
      inst✝² : CommRing B
      inst✝¹ : Algebra A B
      inst✝ : NoZeroDivisors B
      b : B
      i j : Nat
      p : Polynomial A
      h : Eq (Polynomial.map (algebraMap A B) p) (HMul.hMul (HPow.hPow (HSub.hSub Po …
      f : AlgEquiv A B B
      hi : Ne i 0
      ha : Eq b 0
      ⊢ Eq (f b) b
    -/
  · rw [ha, map_zero]
    /-
      🎉 no goals
    -/
  /-
    case neg
    A : Type u_1
    B : Type u_2
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : Algebra A B
    inst✝ : NoZeroDivisors B
    b : B
    i j : Nat
    p : Polynomial A
    h : Eq (Polynomial.map (algebraMap A B) p) (HMul.hMul (HPow.hPow (HSub.hSub Po …
    f : AlgEquiv A B B
    hi : Ne i 0
    ha : Not (Eq b 0)
    ⊢ Eq (f b) b
  -/
  have hf := congrArg (eval b) (congrArg (Polynomial.mapAlgHom f.toAlgHom) h)
  /-
    case neg
    A : Type u_1
    B : Type u_2
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : Algebra A B
    inst✝ : NoZeroDivisors B
    b : B
    i j : Nat
    p : Polynomial A
    h : Eq (Polynomial.map (algebraMap A B) p) (HMul.hMul (HPow.hPow (HSub.hSub Po …
    f : AlgEquiv A B B
    hi : Ne i 0
    ha : Not (Eq b 0)
    hf : Eq (Polynomial.eval b ((Polynomial.mapAlgHom ↑f) (Polynomial.map (algebra …
    ⊢ Eq (f b) b
  -/
  rw [coe_mapAlgHom, map_map, f.toAlgHom.comp_algebraMap, h] at hf
  simp_rw [Polynomial.map_mul, Polynomial.map_pow, Polynomial.map_sub, map_X, map_C,
    eval_mul, eval_pow, eval_sub, eval_X, eval_C, sub_self, zero_pow hi, zero_mul,
    zero_eq_mul, or_iff_left (pow_ne_zero j ha), pow_eq_zero_iff hi, sub_eq_zero] at hf
  /-
    case neg
    A : Type u_1
    B : Type u_2
    inst✝³ : CommRing A
    inst✝² : CommRing B
    inst✝¹ : Algebra A B
    inst✝ : NoZeroDivisors B
    b : B
    i j : Nat
    p : Polynomial A
    h : Eq (Polynomial.map (algebraMap A B) p) (HMul.hMul (HPow.hPow (HSub.hSub Po …
    f : AlgEquiv A B B
    hi : Ne i 0
    ha : Not (Eq b 0)
    hf : Eq b (↑↑f b)
    ⊢ Eq (f b) b
  -/
  exact hf.symm
  /-
    🎉 no goals
  -/


/-- This theorem will be made redundant by `IsFractionRing.stabilizerHom_surjective`. -/
private theorem fixed_of_fixed1 [NoZeroSMulDivisors (B ⧸ Q) L] (f : L ≃ₐ[K] L) (b : B ⧸ Q)
    (hx : ∀ g : MulAction.stabilizer G Q, Ideal.Quotient.stabilizerHom Q P G g b = b) :
    f (algebraMap (B ⧸ Q) L b) = (algebraMap (B ⧸ Q) L b) := by
  classical
  cases nonempty_fintype G
  obtain ⟨b₀, rfl⟩ := Ideal.Quotient.mk_surjective b
  rw [← Ideal.Quotient.algebraMap_eq]
  obtain ⟨a, b, ha1, ha2, hb⟩ := fixed_of_fixed1_aux2 G Q b₀ (fun g hg ↦ hx ⟨g, hg⟩)
  obtain ⟨M, key⟩ := (mem_lifts _).mp (Algebra.IsInvariant.charpoly_mem_lifts A B G b)
  replace key := congrArg (map (algebraMap B (B ⧸ Q))) key
  rw [map_map, ← algebraMap_eq, algebraMap_eq A (A ⧸ P) (B ⧸ Q),
      ← map_map, MulSemiringAction.charpoly, Polynomial.map_prod] at key
  have key₀ : ∀ g : G, (X - C (g • b)).map (algebraMap B (B ⧸ Q)) =
      if g • Q = Q then X - C (algebraMap B (B ⧸ Q) (a * b₀)) else X := by
    intro g
    rw [Polynomial.map_sub, map_X, map_C, hb]
    split_ifs
    · rfl
    · rw [map_zero, map_zero, sub_zero]
  simp only [key₀, Finset.prod_ite, Finset.prod_const] at key
  replace key := congrArg (map (algebraMap (B ⧸ Q) L)) key
  rw [map_map, ← algebraMap_eq, algebraMap_eq (A ⧸ P) K L,
      ← map_map, Polynomial.map_mul, Polynomial.map_pow, Polynomial.map_pow, Polynomial.map_sub,
      map_X, map_C] at key
  replace key := fixed_of_fixed1_aux3 key f (Finset.card_ne_zero_of_mem
    (Finset.mem_filter.mpr ⟨Finset.mem_univ 1, one_smul G Q⟩))
  simp only [map_mul] at key
  obtain ⟨a, rfl⟩ := Algebra.IsInvariant.isInvariant (A := A) a ha1
  rwa [← algebraMap_apply A B (B ⧸ Q), algebraMap_apply A (A ⧸ P) (B ⧸ Q),
      ← algebraMap_apply, algebraMap_apply (A ⧸ P) K L, f.commutes, mul_right_inj'] at key
  rwa [← algebraMap_apply, algebraMap_apply (A ⧸ P) (B ⧸ Q) L,
      ← algebraMap_apply A (A ⧸ P) (B ⧸ Q), algebraMap_apply A B (B ⧸ Q),
      Ne, algebraMap_eq_zero_iff, Ideal.Quotient.algebraMap_eq, Ideal.Quotient.eq_zero_iff_mem]


/-- If `Q` lies over `P`, then the stabilizer of `Q` acts on `Frac(B/Q)/Frac(A/P)`. -/
noncomputable def IsFractionRing.stabilizerHom : MulAction.stabilizer G Q →* (L ≃ₐ[K] L) :=
  MonoidHom.comp (IsFractionRing.fieldEquivOfAlgEquivHom K L) (Ideal.Quotient.stabilizerHom Q P G)


/-- This theorem will be made redundant by `IsFractionRing.stabilizerHom_surjective`. -/
private theorem fixed_of_fixed2 (f : L ≃ₐ[K] L) (x : L)
    (hx : ∀ g : MulAction.stabilizer G Q, IsFractionRing.stabilizerHom G P Q K L g x = x) :
    f x = x := by
  /-
    A : Type u_1
    B : Type u_2
    inst✝¹⁹ : CommRing A
    inst✝¹⁸ : CommRing B
    inst✝¹⁷ : Algebra A B
    G : Type u_3
    inst✝¹⁶ : Group G
    inst✝¹⁵ : Finite G
    inst✝¹⁴ : MulSemiringAction G B
    inst✝¹³ : SMulCommClass G A B
    P : Ideal A
    Q : Ideal B
    inst✝¹² : Q.IsPrime
    inst✝¹¹ : Q.LiesOver P
    K : Type u_4
    L : Type u_5
    inst✝¹⁰ : Field K
    inst✝⁹ : Field L
    inst✝⁸ : Algebra (HasQuotient.Quotient A P) K
    inst✝⁷ : Algebra (HasQuotient.Quotient B Q) L
    inst✝⁶ : Algebra (HasQuotient.Quotient A P) L
    inst✝⁵ : IsScalarTower (HasQuotient.Quotient A P) (HasQuotient.Quotient B Q) L
    inst✝⁴ : Algebra K L
    inst✝³ : IsScalarTower (HasQuotient.Quotient A P) K L
    inst✝² : Algebra.IsInvariant A B G
    inst✝¹ : IsFractionRing (HasQuotient.Quotient A P) K
    inst✝ : IsFractionRing (HasQuotient.Quotient B Q) L
    f : AlgEquiv K L L
    x : L
    hx : ∀ (g : Subtype fun x => Membership.mem (MulAction.stabilizer G Q) x), Eq  …
    ⊢ Eq (f x) x
  -/
  obtain ⟨_⟩ := nonempty_fintype G
  /-
    case intro
    A : Type u_1
    B : Type u_2
    inst✝¹⁹ : CommRing A
    inst✝¹⁸ : CommRing B
    inst✝¹⁷ : Algebra A B
    G : Type u_3
    inst✝¹⁶ : Group G
    inst✝¹⁵ : Finite G
    inst✝¹⁴ : MulSemiringAction G B
    inst✝¹³ : SMulCommClass G A B
    P : Ideal A
    Q : Ideal B
    inst✝¹² : Q.IsPrime
    inst✝¹¹ : Q.LiesOver P
    K : Type u_4
    L : Type u_5
    inst✝¹⁰ : Field K
    inst✝⁹ : Field L
    inst✝⁸ : Algebra (HasQuotient.Quotient A P) K
    inst✝⁷ : Algebra (HasQuotient.Quotient B Q) L
    inst✝⁶ : Algebra (HasQuotient.Quotient A P) L
    inst✝⁵ : IsScalarTower (HasQuotient.Quotient A P) (HasQuotient.Quotient B Q) L
    inst✝⁴ : Algebra K L
    inst✝³ : IsScalarTower (HasQuotient.Quotient A P) K L
    inst✝² : Algebra.IsInvariant A B G
    inst✝¹ : IsFractionRing (HasQuotient.Quotient A P) K
    inst✝ : IsFractionRing (HasQuotient.Quotient B Q) L
    f : AlgEquiv K L L
    x : L
    hx : ∀ (g : Subtype fun x => Membership.mem (MulAction.stabilizer G Q) x), Eq  …
    val✝ : Fintype G
    ⊢ Eq (f x) x
  -/
  have : P.IsPrime := Ideal.over_def Q P ▸ Ideal.IsPrime.under A Q
  /-
    case intro
    A : Type u_1
    B : Type u_2
    inst✝¹⁹ : CommRing A
    inst✝¹⁸ : CommRing B
    inst✝¹⁷ : Algebra A B
    G : Type u_3
    inst✝¹⁶ : Group G
    inst✝¹⁵ : Finite G
    inst✝¹⁴ : MulSemiringAction G B
    inst✝¹³ : SMulCommClass G A B
    P : Ideal A
    Q : Ideal B
    inst✝¹² : Q.IsPrime
    inst✝¹¹ : Q.LiesOver P
    K : Type u_4
    L : Type u_5
    inst✝¹⁰ : Field K
    inst✝⁹ : Field L
    inst✝⁸ : Algebra (HasQuotient.Quotient A P) K
    inst✝⁷ : Algebra (HasQuotient.Quotient B Q) L
    inst✝⁶ : Algebra (HasQuotient.Quotient A P) L
    inst✝⁵ : IsScalarTower (HasQuotient.Quotient A P) (HasQuotient.Quotient B Q) L
    inst✝⁴ : Algebra K L
    inst✝³ : IsScalarTower (HasQuotient.Quotient A P) K L
    inst✝² : Algebra.IsInvariant A B G
    inst✝¹ : IsFractionRing (HasQuotient.Quotient A P) K
    inst✝ : IsFractionRing (HasQuotient.Quotient B Q) L
    f : AlgEquiv K L L
    x : L
    hx : ∀ (g : Subtype fun x => Membership.mem (MulAction.stabilizer G Q) x), Eq  …
    val✝ : Fintype G
    this : P.IsPrime
    ⊢ Eq (f x) x
  -/
  have : Algebra.IsIntegral A B := Algebra.IsInvariant.isIntegral A B G
  /-
    case intro
    A : Type u_1
    B : Type u_2
    inst✝¹⁹ : CommRing A
    inst✝¹⁸ : CommRing B
    inst✝¹⁷ : Algebra A B
    G : Type u_3
    inst✝¹⁶ : Group G
    inst✝¹⁵ : Finite G
    inst✝¹⁴ : MulSemiringAction G B
    inst✝¹³ : SMulCommClass G A B
    P : Ideal A
    Q : Ideal B
    inst✝¹² : Q.IsPrime
    inst✝¹¹ : Q.LiesOver P
    K : Type u_4
    L : Type u_5
    inst✝¹⁰ : Field K
    inst✝⁹ : Field L
    inst✝⁸ : Algebra (HasQuotient.Quotient A P) K
    inst✝⁷ : Algebra (HasQuotient.Quotient B Q) L
    inst✝⁶ : Algebra (HasQuotient.Quotient A P) L
    inst✝⁵ : IsScalarTower (HasQuotient.Quotient A P) (HasQuotient.Quotient B Q) L
    inst✝⁴ : Algebra K L
    inst✝³ : IsScalarTower (HasQuotient.Quotient A P) K L
    inst✝² : Algebra.IsInvariant A B G
    inst✝¹ : IsFractionRing (HasQuotient.Quotient A P) K
    inst✝ : IsFractionRing (HasQuotient.Quotient B Q) L
    f : AlgEquiv K L L
    x : L
    hx : ∀ (g : Subtype fun x => Membership.mem (MulAction.stabilizer G Q) x), Eq  …
    val✝ : Fintype G
    this✝ : P.IsPrime
    this : Algebra.IsIntegral A B
    ⊢ Eq (f x) x
  -/
  obtain ⟨x, y, hy, rfl⟩ := IsFractionRing.div_surjective (A := B ⧸ Q) x
  /-
    case intro.intro.intro.intro
    A : Type u_1
    B : Type u_2
    inst✝¹⁹ : CommRing A
    inst✝¹⁸ : CommRing B
    inst✝¹⁷ : Algebra A B
    G : Type u_3
    inst✝¹⁶ : Group G
    inst✝¹⁵ : Finite G
    inst✝¹⁴ : MulSemiringAction G B
    inst✝¹³ : SMulCommClass G A B
    P : Ideal A
    Q : Ideal B
    inst✝¹² : Q.IsPrime
    inst✝¹¹ : Q.LiesOver P
    K : Type u_4
    L : Type u_5
    inst✝¹⁰ : Field K
    inst✝⁹ : Field L
    inst✝⁸ : Algebra (HasQuotient.Quotient A P) K
    inst✝⁷ : Algebra (HasQuotient.Quotient B Q) L
    inst✝⁶ : Algebra (HasQuotient.Quotient A P) L
    inst✝⁵ : IsScalarTower (HasQuotient.Quotient A P) (HasQuotient.Quotient B Q) L
    inst✝⁴ : Algebra K L
    inst✝³ : IsScalarTower (HasQuotient.Quotient A P) K L
    inst✝² : Algebra.IsInvariant A B G
    inst✝¹ : IsFractionRing (HasQuotient.Quotient A P) K
    inst✝ : IsFractionRing (HasQuotient.Quotient B Q) L
    f : AlgEquiv K L L
    val✝ : Fintype G
    this✝ : P.IsPrime
    this : Algebra.IsIntegral A B
    x y : HasQuotient.Quotient B Q
    hy : Membership.mem (nonZeroDivisors (HasQuotient.Quotient B Q)) y
    hx : ∀ (g : Subtype fun x => Membership.mem (MulAction.stabilizer G Q) x), Eq  …
    ⊢ Eq (f (HDiv.hDiv ((algebraMap (HasQuotient.Quotient B Q) L) x) ((algebraMap  …
  -/
  obtain ⟨b, a, ha, h⟩ := (Algebra.IsAlgebraic.isAlgebraic (R := A ⧸ P) y).exists_smul_eq_mul x hy
  replace ha : algebraMap (A ⧸ P) L a ≠ 0 := by
    rwa [Ne, algebraMap_apply (A ⧸ P) K L, algebraMap_eq_zero_iff, algebraMap_eq_zero_iff]
  replace hy : algebraMap (B ⧸ Q) L y ≠ 0 :=
    mt (algebraMap_eq_zero_iff (B ⧸ Q) L).mp (nonZeroDivisors.ne_zero hy)
  replace h : algebraMap (B ⧸ Q) L x / algebraMap (B ⧸ Q) L y =
      algebraMap (B ⧸ Q) L b / algebraMap (A ⧸ P) L a := by
    rw [mul_comm, Algebra.smul_def, mul_comm] at h
    rw [div_eq_div_iff hy ha, ← map_mul, ← h, map_mul, ← algebraMap_apply]
  /-
    case intro.intro.intro.intro.intro.intro.intro
    A : Type u_1
    B : Type u_2
    inst✝¹⁹ : CommRing A
    inst✝¹⁸ : CommRing B
    inst✝¹⁷ : Algebra A B
    G : Type u_3
    inst✝¹⁶ : Group G
    inst✝¹⁵ : Finite G
    inst✝¹⁴ : MulSemiringAction G B
    inst✝¹³ : SMulCommClass G A B
    P : Ideal A
    Q : Ideal B
    inst✝¹² : Q.IsPrime
    inst✝¹¹ : Q.LiesOver P
    K : Type u_4
    L : Type u_5
    inst✝¹⁰ : Field K
    inst✝⁹ : Field L
    inst✝⁸ : Algebra (HasQuotient.Quotient A P) K
    inst✝⁷ : Algebra (HasQuotient.Quotient B Q) L
    inst✝⁶ : Algebra (HasQuotient.Quotient A P) L
    inst✝⁵ : IsScalarTower (HasQuotient.Quotient A P) (HasQuotient.Quotient B Q) L
    inst✝⁴ : Algebra K L
    inst✝³ : IsScalarTower (HasQuotient.Quotient A P) K L
    inst✝² : Algebra.IsInvariant A B G
    inst✝¹ : IsFractionRing (HasQuotient.Quotient A P) K
    inst✝ : IsFractionRing (HasQuotient.Quotient B Q) L
    f : AlgEquiv K L L
    val✝ : Fintype G
    this✝ : P.IsPrime
    this : Algebra.IsIntegral A B
    x y : HasQuotient.Quotient B Q
    hx : ∀ (g : Subtype fun x => Membership.mem (MulAction.stabilizer G Q) x), Eq  …
    b : HasQuotient.Quotient B Q
    a : HasQuotient.Quotient A P
    ha : Ne ((algebraMap (HasQuotient.Quotient A P) L) a) 0
    hy : Ne ((algebraMap (HasQuotient.Quotient B Q) L) y) 0
    h : Eq (HDiv.hDiv ((algebraMap (HasQuotient.Quotient B Q) L) x) ((algebraMap ( …
    ⊢ Eq (f (HDiv.hDiv ((algebraMap (HasQuotient.Quotient B Q) L) x) ((algebraMap  …
  -/
  simp only [h, map_div₀, algebraMap_apply (A ⧸ P) K L, AlgEquiv.commutes] at hx ⊢
  /-
    case intro.intro.intro.intro.intro.intro.intro
    A : Type u_1
    B : Type u_2
    inst✝¹⁹ : CommRing A
    inst✝¹⁸ : CommRing B
    inst✝¹⁷ : Algebra A B
    G : Type u_3
    inst✝¹⁶ : Group G
    inst✝¹⁵ : Finite G
    inst✝¹⁴ : MulSemiringAction G B
    inst✝¹³ : SMulCommClass G A B
    P : Ideal A
    Q : Ideal B
    inst✝¹² : Q.IsPrime
    inst✝¹¹ : Q.LiesOver P
    K : Type u_4
    L : Type u_5
    inst✝¹⁰ : Field K
    inst✝⁹ : Field L
    inst✝⁸ : Algebra (HasQuotient.Quotient A P) K
    inst✝⁷ : Algebra (HasQuotient.Quotient B Q) L
    inst✝⁶ : Algebra (HasQuotient.Quotient A P) L
    inst✝⁵ : IsScalarTower (HasQuotient.Quotient A P) (HasQuotient.Quotient B Q) L
    inst✝⁴ : Algebra K L
    inst✝³ : IsScalarTower (HasQuotient.Quotient A P) K L
    inst✝² : Algebra.IsInvariant A B G
    inst✝¹ : IsFractionRing (HasQuotient.Quotient A P) K
    inst✝ : IsFractionRing (HasQuotient.Quotient B Q) L
    f : AlgEquiv K L L
    val✝ : Fintype G
    this✝ : P.IsPrime
    this : Algebra.IsIntegral A B
    x y b : HasQuotient.Quotient B Q
    a : HasQuotient.Quotient A P
    ha : Ne ((algebraMap (HasQuotient.Quotient A P) L) a) 0
    hy : Ne ((algebraMap (HasQuotient.Quotient B Q) L) y) 0
    h : Eq (HDiv.hDiv ((algebraMap (HasQuotient.Quotient B Q) L) x) ((algebraMap ( …
    hx : ∀ (g : Subtype fun x => Membership.mem (MulAction.stabilizer G Q) x), Eq  …
    ⊢ Eq (HDiv.hDiv (f ((algebraMap (HasQuotient.Quotient B Q) L) b)) ((algebraMap …
  -/
  simp only [← algebraMap_apply, div_left_inj' ha] at hx ⊢
  exact fixed_of_fixed1 G P Q K L f b (fun g ↦ IsFractionRing.injective (B ⧸ Q) L
    ((IsFractionRing.fieldEquivOfAlgEquiv_algebraMap K L L
      (Ideal.Quotient.stabilizerHom Q P G g) b).symm.trans (hx g)))


/-- The stabilizer subgroup of `Q` surjects onto `Aut(Frac(B/Q)/Frac(A/P))`. -/
theorem IsFractionRing.stabilizerHom_surjective :
    Function.Surjective (stabilizerHom G P Q K L) := by
  /-
    A : Type u_1
    B : Type u_2
    inst✝¹⁹ : CommRing A
    inst✝¹⁸ : CommRing B
    inst✝¹⁷ : Algebra A B
    G : Type u_3
    inst✝¹⁶ : Group G
    inst✝¹⁵ : Finite G
    inst✝¹⁴ : MulSemiringAction G B
    inst✝¹³ : SMulCommClass G A B
    P : Ideal A
    Q : Ideal B
    inst✝¹² : Q.IsPrime
    inst✝¹¹ : Q.LiesOver P
    K : Type u_4
    L : Type u_5
    inst✝¹⁰ : Field K
    inst✝⁹ : Field L
    inst✝⁸ : Algebra (HasQuotient.Quotient A P) K
    inst✝⁷ : Algebra (HasQuotient.Quotient B Q) L
    inst✝⁶ : Algebra (HasQuotient.Quotient A P) L
    inst✝⁵ : IsScalarTower (HasQuotient.Quotient A P) (HasQuotient.Quotient B Q) L
    inst✝⁴ : Algebra K L
    inst✝³ : IsScalarTower (HasQuotient.Quotient A P) K L
    inst✝² : Algebra.IsInvariant A B G
    inst✝¹ : IsFractionRing (HasQuotient.Quotient A P) K
    inst✝ : IsFractionRing (HasQuotient.Quotient B Q) L
    ⊢ Function.Surjective ⇑(IsFractionRing.stabilizerHom G P Q K L)
  -/
  let _ := MulSemiringAction.compHom L (stabilizerHom G P Q K L)
  /-
    A : Type u_1
    B : Type u_2
    inst✝¹⁹ : CommRing A
    inst✝¹⁸ : CommRing B
    inst✝¹⁷ : Algebra A B
    G : Type u_3
    inst✝¹⁶ : Group G
    inst✝¹⁵ : Finite G
    inst✝¹⁴ : MulSemiringAction G B
    inst✝¹³ : SMulCommClass G A B
    P : Ideal A
    Q : Ideal B
    inst✝¹² : Q.IsPrime
    inst✝¹¹ : Q.LiesOver P
    K : Type u_4
    L : Type u_5
    inst✝¹⁰ : Field K
    inst✝⁹ : Field L
    inst✝⁸ : Algebra (HasQuotient.Quotient A P) K
    inst✝⁷ : Algebra (HasQuotient.Quotient B Q) L
    inst✝⁶ : Algebra (HasQuotient.Quotient A P) L
    inst✝⁵ : IsScalarTower (HasQuotient.Quotient A P) (HasQuotient.Quotient B Q) L
    inst✝⁴ : Algebra K L
    inst✝³ : IsScalarTower (HasQuotient.Quotient A P) K L
    inst✝² : Algebra.IsInvariant A B G
    inst✝¹ : IsFractionRing (HasQuotient.Quotient A P) K
    inst✝ : IsFractionRing (HasQuotient.Quotient B Q) L
    x✝ : MulSemiringAction (Subtype fun x => Membership.mem (MulAction.stabilizer  …
    ⊢ Function.Surjective ⇑(IsFractionRing.stabilizerHom G P Q K L)
  -/
  intro f
  obtain ⟨g, hg⟩ := FixedPoints.toAlgAut_surjective (MulAction.stabilizer G Q) L
    (AlgEquiv.ofRingEquiv (f := f) (fun x ↦ fixed_of_fixed2 G P Q K L f x x.2))
  /-
    case intro
    A : Type u_1
    B : Type u_2
    inst✝¹⁹ : CommRing A
    inst✝¹⁸ : CommRing B
    inst✝¹⁷ : Algebra A B
    G : Type u_3
    inst✝¹⁶ : Group G
    inst✝¹⁵ : Finite G
    inst✝¹⁴ : MulSemiringAction G B
    inst✝¹³ : SMulCommClass G A B
    P : Ideal A
    Q : Ideal B
    inst✝¹² : Q.IsPrime
    inst✝¹¹ : Q.LiesOver P
    K : Type u_4
    L : Type u_5
    inst✝¹⁰ : Field K
    inst✝⁹ : Field L
    inst✝⁸ : Algebra (HasQuotient.Quotient A P) K
    inst✝⁷ : Algebra (HasQuotient.Quotient B Q) L
    inst✝⁶ : Algebra (HasQuotient.Quotient A P) L
    inst✝⁵ : IsScalarTower (HasQuotient.Quotient A P) (HasQuotient.Quotient B Q) L
    inst✝⁴ : Algebra K L
    inst✝³ : IsScalarTower (HasQuotient.Quotient A P) K L
    inst✝² : Algebra.IsInvariant A B G
    inst✝¹ : IsFractionRing (HasQuotient.Quotient A P) K
    inst✝ : IsFractionRing (HasQuotient.Quotient B Q) L
    x✝ : MulSemiringAction (Subtype fun x => Membership.mem (MulAction.stabilizer  …
    f : AlgEquiv K L L
    g : Subtype fun x => Membership.mem (MulAction.stabilizer G Q) x
    hg : Eq ((MulSemiringAction.toAlgAut (Subtype fun x => Membership.mem (MulActi …
    ⊢ Exists fun a => Eq ((IsFractionRing.stabilizerHom G P Q K L) a) f
  -/
  exact ⟨g, by rwa [AlgEquiv.ext_iff] at hg ⊢⟩
  /-
    🎉 no goals
  -/


/-- The stabilizer subgroup of `Q` surjects onto `Aut((B/Q)/(A/P))`. -/
theorem Ideal.Quotient.stabilizerHom_surjective :
    Function.Surjective (Ideal.Quotient.stabilizerHom Q P G) := by
  /-
    A : Type u_1
    B : Type u_2
    inst✝⁹ : CommRing A
    inst✝⁸ : CommRing B
    inst✝⁷ : Algebra A B
    G : Type u_3
    inst✝⁶ : Group G
    inst✝⁵ : Finite G
    inst✝⁴ : MulSemiringAction G B
    inst✝³ : SMulCommClass G A B
    P : Ideal A
    Q : Ideal B
    inst✝² : Q.IsPrime
    inst✝¹ : Q.LiesOver P
    inst✝ : Algebra.IsInvariant A B G
    ⊢ Function.Surjective ⇑(Ideal.Quotient.stabilizerHom Q P G)
  -/
  have : P.IsPrime := Ideal.over_def Q P ▸ Ideal.IsPrime.under A Q
  /-
    A : Type u_1
    B : Type u_2
    inst✝⁹ : CommRing A
    inst✝⁸ : CommRing B
    inst✝⁷ : Algebra A B
    G : Type u_3
    inst✝⁶ : Group G
    inst✝⁵ : Finite G
    inst✝⁴ : MulSemiringAction G B
    inst✝³ : SMulCommClass G A B
    P : Ideal A
    Q : Ideal B
    inst✝² : Q.IsPrime
    inst✝¹ : Q.LiesOver P
    inst✝ : Algebra.IsInvariant A B G
    this : P.IsPrime
    ⊢ Function.Surjective ⇑(Ideal.Quotient.stabilizerHom Q P G)
  -/
  let _ := FractionRing.liftAlgebra (A ⧸ P) (FractionRing (B ⧸ Q))
  have key := IsFractionRing.stabilizerHom_surjective G P Q
    (FractionRing (A ⧸ P)) (FractionRing (B ⧸ Q))
  /-
    A : Type u_1
    B : Type u_2
    inst✝⁹ : CommRing A
    inst✝⁸ : CommRing B
    inst✝⁷ : Algebra A B
    G : Type u_3
    inst✝⁶ : Group G
    inst✝⁵ : Finite G
    inst✝⁴ : MulSemiringAction G B
    inst✝³ : SMulCommClass G A B
    P : Ideal A
    Q : Ideal B
    inst✝² : Q.IsPrime
    inst✝¹ : Q.LiesOver P
    inst✝ : Algebra.IsInvariant A B G
    this : P.IsPrime
    x✝ : Algebra (FractionRing (HasQuotient.Quotient A P)) (FractionRing (HasQuoti …
    key : Function.Surjective ⇑(IsFractionRing.stabilizerHom G P Q (FractionRing ( …
    ⊢ Function.Surjective ⇑(Ideal.Quotient.stabilizerHom Q P G)
  -/
  rw [IsFractionRing.stabilizerHom, MonoidHom.coe_comp] at key
  exact key.of_comp_left (IsFractionRing.fieldEquivOfAlgEquivHom_injective (A ⧸ P) (B ⧸ Q)
    (FractionRing (A ⧸ P)) (FractionRing (B ⧸ Q)))


