local notation "deg("p")" => natDegree p

local notation3 "coeffs("p")" => Set.range (coeff p)

local notation3 "spanCoeffs("p")" => 1 ⊔ Submodule.span R coeffs(p)


open Submodule Set in
lemma coeff_divModByMonicAux_mem_span_pow_mul_span : ∀ (p q : S[X]) (hq : q.Monic) (i),
    (p.divModByMonicAux hq).1.coeff i ∈ spanCoeffs(q) ^ deg(p) * spanCoeffs(p) ∧
    (p.divModByMonicAux hq).2.coeff i ∈ spanCoeffs(q) ^ deg(p) * spanCoeffs(p)
  | p, q, hq, i => by
    /-
      R : Type u_2
      S : Type u_3
      inst✝² : CommRing R
      inst✝¹ : Ring S
      inst✝ : Algebra R S
      p q : Polynomial S
      hq : q.Monic
      i : Nat
      ⊢ And (Membership.mem (HMul.hMul (HPow.hPow (Max.max 1 (Submodule.span R (Set. …
    -/
    rw [divModByMonicAux]
    have H₀ (i) : p.coeff i ∈ spanCoeffs(q) ^ deg(p) * spanCoeffs(p) := by
      refine Submodule.mul_le_mul_left (pow_le_pow_left' le_sup_left _) ?_
      simp only [one_pow, one_mul]
      exact SetLike.le_def.mp le_sup_right (subset_span (mem_range_self i))
    /-
      R : Type u_2
      S : Type u_3
      inst✝² : CommRing R
      inst✝¹ : Ring S
      inst✝ : Algebra R S
      p q : Polynomial S
      hq : q.Monic
      i : Nat
      H₀ : ∀ (i : Nat), Membership.mem (HMul.hMul (HPow.hPow (Max.max 1 (Submodule.s …
      ⊢ And
          (Membership.mem (HMul.hMul (HPow.hPow (Max.max 1 (Submodule.span R (Set.ra …
            ((dite (And (LE.le q.degree p.degree) (Ne p 0))
                    (fun h =>
                      let z := HMul.hMul (Polynomial.C p.leadingCoeff) (HPow.hPow Po …
                      letFun ⋯ fun _wf =>
                        let dm := (HSub.hSub p (HMul.hMul q z)).divModByMonicAux hq;
                        { fst := HAdd.hAdd z dm.1, snd := dm.2 })
                    fun h => { fst := 0, snd := p }).1.coeff
              i))
          (Membership.mem (HMul.hMul (HPow.hPow (Max.max 1 (Submodule.span R (Set.ra …
            ((dite (And (LE.le q.degree p.degree) (Ne p 0))
                    (fun h =>
                      let z := HMul.hMul (Polynomial.C p.leadingCoeff) (HPow.hPow Po …
                      letFun ⋯ fun _wf =>
                        let dm := (HSub.hSub p (HMul.hMul q z)).divModByMonicAux hq;
                        { fst := HAdd.hAdd z dm.1, snd := dm.2 })
                    fun h => { fst := 0, snd := p }).2.coeff
              i))
    -/
    split_ifs with hpq; swap
      /-
        case neg
        R : Type u_2
        S : Type u_3
        inst✝² : CommRing R
        inst✝¹ : Ring S
        inst✝ : Algebra R S
        p q : Polynomial S
        hq : q.Monic
        i : Nat
        H₀ : ∀ (i : Nat), Membership.mem (HMul.hMul (HPow.hPow (Max.max 1 (Submodule.s …
        hpq : Not (And (LE.le q.degree p.degree) (Ne p 0))
        ⊢ And (Membership.mem (HMul.hMul (HPow.hPow (Max.max 1 (Submodule.span R (Set. …
      -/
    · simpa using H₀ _
      /-
        🎉 no goals
      -/
    /-
      case pos
      R : Type u_2
      S : Type u_3
      inst✝² : CommRing R
      inst✝¹ : Ring S
      inst✝ : Algebra R S
      p q : Polynomial S
      hq : q.Monic
      i : Nat
      H₀ : ∀ (i : Nat), Membership.mem (HMul.hMul (HPow.hPow (Max.max 1 (Submodule.s …
      hpq : And (LE.le q.degree p.degree) (Ne p 0)
      ⊢ And
          (Membership.mem (HMul.hMul (HPow.hPow (Max.max 1 (Submodule.span R (Set.ra …
            ((let z := HMul.hMul (Polynomial.C p.leadingCoeff) (HPow.hPow Polynomial …
                  let dm := (HSub.hSub p (HMul.hMul q z)).divModByMonicAux hq;
                  { fst := HAdd.hAdd z dm.1, snd := dm.2 }).1.coeff
              i))
          (Membership.mem (HMul.hMul (HPow.hPow (Max.max 1 (Submodule.span R (Set.ra …
            ((let z := HMul.hMul (Polynomial.C p.leadingCoeff) (HPow.hPow Polynomial …
                  let dm := (HSub.hSub p (HMul.hMul q z)).divModByMonicAux hq;
                  { fst := HAdd.hAdd z dm.1, snd := dm.2 }).2.coeff
              i))
    -/
    simp only [coeff_add, coeff_C_mul, coeff_X_pow]
    /-
      case pos
      R : Type u_2
      S : Type u_3
      inst✝² : CommRing R
      inst✝¹ : Ring S
      inst✝ : Algebra R S
      p q : Polynomial S
      hq : q.Monic
      i : Nat
      H₀ : ∀ (i : Nat), Membership.mem (HMul.hMul (HPow.hPow (Max.max 1 (Submodule.s …
      hpq : And (LE.le q.degree p.degree) (Ne p 0)
      ⊢ And (Membership.mem (HMul.hMul (HPow.hPow (Max.max 1 (Submodule.span R (Set. …
    -/
    generalize hr : (p - q * (C p.leadingCoeff * X ^ (deg(p) - deg(q)))) = r
    /-
      case pos
      R : Type u_2
      S : Type u_3
      inst✝² : CommRing R
      inst✝¹ : Ring S
      inst✝ : Algebra R S
      p q : Polynomial S
      hq : q.Monic
      i : Nat
      H₀ : ∀ (i : Nat), Membership.mem (HMul.hMul (HPow.hPow (Max.max 1 (Submodule.s …
      hpq : And (LE.le q.degree p.degree) (Ne p 0)
      r : Polynomial S
      hr : Eq (HSub.hSub p (HMul.hMul q (HMul.hMul (Polynomial.C p.leadingCoeff) (HP …
      ⊢ And (Membership.mem (HMul.hMul (HPow.hPow (Max.max 1 (Submodule.span R (Set. …
    -/
    by_cases hr' : r = 0
    · simp only [mul_ite, mul_one, mul_zero, hr', divModByMonicAux, degree_zero, le_bot_iff,
        degree_eq_bot, ne_eq, not_true_eq_false, and_false, ↓reduceDIte, Prod.mk_zero_zero,
        Prod.fst_zero, coeff_zero, add_zero, Prod.snd_zero, Submodule.zero_mem, and_true]
      /-
        case pos
        R : Type u_2
        S : Type u_3
        inst✝² : CommRing R
        inst✝¹ : Ring S
        inst✝ : Algebra R S
        p q : Polynomial S
        hq : q.Monic
        i : Nat
        H₀ : ∀ (i : Nat), Membership.mem (HMul.hMul (HPow.hPow (Max.max 1 (Submodule.s …
        hpq : And (LE.le q.degree p.degree) (Ne p 0)
        r : Polynomial S
        hr : Eq (HSub.hSub p (HMul.hMul q (HMul.hMul (Polynomial.C p.leadingCoeff) (HP …
        hr' : Eq r 0
        ⊢ Membership.mem (HMul.hMul (HPow.hPow (Max.max 1 (Submodule.span R (Set.range …
      -/
      split_ifs
      /-
        case pos
        R : Type u_2
        S : Type u_3
        inst✝² : CommRing R
        inst✝¹ : Ring S
        inst✝ : Algebra R S
        p q : Polynomial S
        hq : q.Monic
        i : Nat
        H₀ : ∀ (i : Nat), Membership.mem (HMul.hMul (HPow.hPow (Max.max 1 (Submodule.s …
        hpq : And (LE.le q.degree p.degree) (Ne p 0)
        r : Polynomial S
        hr : Eq (HSub.hSub p (HMul.hMul q (HMul.hMul (Polynomial.C p.leadingCoeff) (HP …
        hr' : Eq r 0
        h✝ : Eq i (HSub.hSub p.natDegree q.natDegree)
        ⊢ Membership.mem (HMul.hMul (HPow.hPow (Max.max 1 (Submodule.span R (Set.range …
      -/
      exacts [H₀ _, zero_mem _]
      /-
        🎉 no goals
      -/
    have H : span R coeffs(r) ≤ span R coeffs(p) ⊔ span R coeffs(q) * span R coeffs(p) := by
      rw [span_le, ← hr]
      rintro _ ⟨i, rfl⟩
      rw [coeff_sub, ← mul_assoc, coeff_mul_X_pow', coeff_mul_C]
      apply sub_mem
      · exact SetLike.le_def.mp le_sup_left (subset_span (mem_range_self _))
      · split_ifs
        · refine SetLike.le_def.mp le_sup_right (mul_mem_mul ?_ ?_) <;> exact subset_span ⟨_, rfl⟩
        · exact zero_mem _
    /-
      case neg
      R : Type u_2
      S : Type u_3
      inst✝² : CommRing R
      inst✝¹ : Ring S
      inst✝ : Algebra R S
      p q : Polynomial S
      hq : q.Monic
      i : Nat
      H₀ : ∀ (i : Nat), Membership.mem (HMul.hMul (HPow.hPow (Max.max 1 (Submodule.s …
      hpq : And (LE.le q.degree p.degree) (Ne p 0)
      r : Polynomial S
      hr : Eq (HSub.hSub p (HMul.hMul q (HMul.hMul (Polynomial.C p.leadingCoeff) (HP …
      hr' : Not (Eq r 0)
      H : LE.le (Submodule.span R (Set.range r.coeff)) (Max.max (Submodule.span R (S …
      ⊢ And (Membership.mem (HMul.hMul (HPow.hPow (Max.max 1 (Submodule.span R (Set. …
    -/
    have deg_r_lt_deg_p : deg(r) < deg(p) := natDegree_lt_natDegree hr' (hr ▸ div_wf_lemma hpq hq)
    have H'' := calc
      spanCoeffs(q) ^ deg(r) * spanCoeffs(r)
      _ ≤ spanCoeffs(q) ^ deg(r) *
          (1 ⊔ (span R coeffs(p) ⊔ span R coeffs(q) * span R coeffs(p))) := by gcongr
      _ ≤ spanCoeffs(q) ^ deg(r) * (spanCoeffs(q) * spanCoeffs(p)) := by
        gcongr
        simp only [sup_le_iff]
        refine ⟨one_le_mul le_sup_left le_sup_left, ?_, mul_le_mul' le_sup_right le_sup_right⟩
        rw [Submodule.sup_mul, one_mul]
        exact le_sup_of_le_left le_sup_right
      _ = spanCoeffs(q) ^ (deg(r) + 1) * spanCoeffs(p) := by rw [pow_succ, mul_assoc]
      _ ≤ spanCoeffs(q) ^ deg(p) * spanCoeffs(p) := by gcongr; exacts [le_sup_left, deg_r_lt_deg_p]
    /-
      case neg
      R : Type u_2
      S : Type u_3
      inst✝² : CommRing R
      inst✝¹ : Ring S
      inst✝ : Algebra R S
      p q : Polynomial S
      hq : q.Monic
      i : Nat
      H₀ : ∀ (i : Nat), Membership.mem (HMul.hMul (HPow.hPow (Max.max 1 (Submodule.s …
      hpq : And (LE.le q.degree p.degree) (Ne p 0)
      r : Polynomial S
      hr : Eq (HSub.hSub p (HMul.hMul q (HMul.hMul (Polynomial.C p.leadingCoeff) (HP …
      hr' : Not (Eq r 0)
      H : LE.le (Submodule.span R (Set.range r.coeff)) (Max.max (Submodule.span R (S …
      deg_r_lt_deg_p : LT.lt r.natDegree p.natDegree
      H'' : LE.le (HMul.hMul (HPow.hPow (Max.max 1 (Submodule.span R (Set.range q.co …
      ⊢ And (Membership.mem (HMul.hMul (HPow.hPow (Max.max 1 (Submodule.span R (Set. …
    -/
    refine ⟨add_mem ?_ ?_, ?_⟩
      /-
        case neg.refine_1
        R : Type u_2
        S : Type u_3
        inst✝² : CommRing R
        inst✝¹ : Ring S
        inst✝ : Algebra R S
        p q : Polynomial S
        hq : q.Monic
        i : Nat
        H₀ : ∀ (i : Nat), Membership.mem (HMul.hMul (HPow.hPow (Max.max 1 (Submodule.s …
        hpq : And (LE.le q.degree p.degree) (Ne p 0)
        r : Polynomial S
        hr : Eq (HSub.hSub p (HMul.hMul q (HMul.hMul (Polynomial.C p.leadingCoeff) (HP …
        hr' : Not (Eq r 0)
        H : LE.le (Submodule.span R (Set.range r.coeff)) (Max.max (Submodule.span R (S …
        deg_r_lt_deg_p : LT.lt r.natDegree p.natDegree
        H'' : LE.le (HMul.hMul (HPow.hPow (Max.max 1 (Submodule.span R (Set.range q.co …
        ⊢ Membership.mem (HMul.hMul (HPow.hPow (Max.max 1 (Submodule.span R (Set.range …
      -/
    · split_ifs <;> simp only [mul_one, mul_zero]
      /-
        case pos
        R : Type u_2
        S : Type u_3
        inst✝² : CommRing R
        inst✝¹ : Ring S
        inst✝ : Algebra R S
        p q : Polynomial S
        hq : q.Monic
        i : Nat
        H₀ : ∀ (i : Nat), Membership.mem (HMul.hMul (HPow.hPow (Max.max 1 (Submodule.s …
        hpq : And (LE.le q.degree p.degree) (Ne p 0)
        r : Polynomial S
        hr : Eq (HSub.hSub p (HMul.hMul q (HMul.hMul (Polynomial.C p.leadingCoeff) (HP …
        hr' : Not (Eq r 0)
        H : LE.le (Submodule.span R (Set.range r.coeff)) (Max.max (Submodule.span R (S …
        deg_r_lt_deg_p : LT.lt r.natDegree p.natDegree
        H'' : LE.le (HMul.hMul (HPow.hPow (Max.max 1 (Submodule.span R (Set.range q.co …
        h✝ : Eq i (HSub.hSub p.natDegree q.natDegree)
        ⊢ Membership.mem (HMul.hMul (HPow.hPow (Max.max 1 (Submodule.span R (Set.range …
      -/
      exacts [H₀ _, zero_mem _]
      /-
        🎉 no goals
      -/
      /-
        case neg.refine_2
        R : Type u_2
        S : Type u_3
        inst✝² : CommRing R
        inst✝¹ : Ring S
        inst✝ : Algebra R S
        p q : Polynomial S
        hq : q.Monic
        i : Nat
        H₀ : ∀ (i : Nat), Membership.mem (HMul.hMul (HPow.hPow (Max.max 1 (Submodule.s …
        hpq : And (LE.le q.degree p.degree) (Ne p 0)
        r : Polynomial S
        hr : Eq (HSub.hSub p (HMul.hMul q (HMul.hMul (Polynomial.C p.leadingCoeff) (HP …
        hr' : Not (Eq r 0)
        H : LE.le (Submodule.span R (Set.range r.coeff)) (Max.max (Submodule.span R (S …
        deg_r_lt_deg_p : LT.lt r.natDegree p.natDegree
        H'' : LE.le (HMul.hMul (HPow.hPow (Max.max 1 (Submodule.span R (Set.range q.co …
        ⊢ Membership.mem (HMul.hMul (HPow.hPow (Max.max 1 (Submodule.span R (Set.range …
      -/
    · exact H'' (coeff_divModByMonicAux_mem_span_pow_mul_span r _ hq i).1
      /-
        🎉 no goals
      -/
      /-
        case neg.refine_3
        R : Type u_2
        S : Type u_3
        inst✝² : CommRing R
        inst✝¹ : Ring S
        inst✝ : Algebra R S
        p q : Polynomial S
        hq : q.Monic
        i : Nat
        H₀ : ∀ (i : Nat), Membership.mem (HMul.hMul (HPow.hPow (Max.max 1 (Submodule.s …
        hpq : And (LE.le q.degree p.degree) (Ne p 0)
        r : Polynomial S
        hr : Eq (HSub.hSub p (HMul.hMul q (HMul.hMul (Polynomial.C p.leadingCoeff) (HP …
        hr' : Not (Eq r 0)
        H : LE.le (Submodule.span R (Set.range r.coeff)) (Max.max (Submodule.span R (S …
        deg_r_lt_deg_p : LT.lt r.natDegree p.natDegree
        H'' : LE.le (HMul.hMul (HPow.hPow (Max.max 1 (Submodule.span R (Set.range q.co …
        ⊢ Membership.mem (HMul.hMul (HPow.hPow (Max.max 1 (Submodule.span R (Set.range …
      -/
    · exact H'' (coeff_divModByMonicAux_mem_span_pow_mul_span _ _ hq i).2
      /-
        🎉 no goals
      -/
  termination_by p => deg(p)


/-- For polynomials `p q : R[X]`, the coefficients of `p %ₘ q` can be written as sums of products of
coefficients of `p` and `q`.

Precisely, each summand needs at most one coefficient of `p` and `deg p` coefficients of `q`. -/
lemma coeff_modByMonic_mem_pow_natDegree_mul (p q : S[X])
    (Mp : Submodule R S) (hp : ∀ i, p.coeff i ∈ Mp) (hp' : 1 ∈ Mp)
    (Mq : Submodule R S) (hq : ∀ i, q.coeff i ∈ Mq) (hq' : 1 ∈ Mq) (i : ℕ) :
    (p %ₘ q).coeff i ∈ Mq ^ p.natDegree * Mp := by
  /-
    R : Type u_2
    S : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    p q : Polynomial S
    Mp : Submodule R S
    hp : ∀ (i : Nat), Membership.mem Mp (p.coeff i)
    hp' : Membership.mem Mp 1
    Mq : Submodule R S
    hq : ∀ (i : Nat), Membership.mem Mq (q.coeff i)
    hq' : Membership.mem Mq 1
    i : Nat
    ⊢ Membership.mem (HMul.hMul (HPow.hPow Mq p.natDegree) Mp) ((p.modByMonic q).c …
  -/
  delta modByMonic
  /-
    R : Type u_2
    S : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    p q : Polynomial S
    Mp : Submodule R S
    hp : ∀ (i : Nat), Membership.mem Mp (p.coeff i)
    hp' : Membership.mem Mp 1
    Mq : Submodule R S
    hq : ∀ (i : Nat), Membership.mem Mq (q.coeff i)
    hq' : Membership.mem Mq 1
    i : Nat
    ⊢ Membership.mem (HMul.hMul (HPow.hPow Mq p.natDegree) Mp) ((dite q.Monic (fun …
  -/
  split_ifs with H
    /-
      case pos
      R : Type u_2
      S : Type u_3
      inst✝² : CommRing R
      inst✝¹ : Ring S
      inst✝ : Algebra R S
      p q : Polynomial S
      Mp : Submodule R S
      hp : ∀ (i : Nat), Membership.mem Mp (p.coeff i)
      hp' : Membership.mem Mp 1
      Mq : Submodule R S
      hq : ∀ (i : Nat), Membership.mem Mq (q.coeff i)
      hq' : Membership.mem Mq 1
      i : Nat
      H : q.Monic
      ⊢ Membership.mem (HMul.hMul (HPow.hPow Mq p.natDegree) Mp) ((p.divModByMonicAu …
    -/
  · refine SetLike.le_def.mp ?_ (coeff_divModByMonicAux_mem_span_pow_mul_span (R := R) p q H i).2
    /-
      case pos
      R : Type u_2
      S : Type u_3
      inst✝² : CommRing R
      inst✝¹ : Ring S
      inst✝ : Algebra R S
      p q : Polynomial S
      Mp : Submodule R S
      hp : ∀ (i : Nat), Membership.mem Mp (p.coeff i)
      hp' : Membership.mem Mp 1
      Mq : Submodule R S
      hq : ∀ (i : Nat), Membership.mem Mq (q.coeff i)
      hq' : Membership.mem Mq 1
      i : Nat
      H : q.Monic
      ⊢ LE.le (HMul.hMul (HPow.hPow (Max.max 1 (Submodule.span R (Set.range q.coeff) …
    -/
               /-
                 🎉 no goals
               -/
    gcongr <;> exact sup_le (by simpa) (by simpa [Submodule.span_le, Set.range_subset_iff])
               /-
                 🎉 no goals
               -/
    /-
      case neg
      R : Type u_2
      S : Type u_3
      inst✝² : CommRing R
      inst✝¹ : Ring S
      inst✝ : Algebra R S
      p q : Polynomial S
      Mp : Submodule R S
      hp : ∀ (i : Nat), Membership.mem Mp (p.coeff i)
      hp' : Membership.mem Mp 1
      Mq : Submodule R S
      hq : ∀ (i : Nat), Membership.mem Mq (q.coeff i)
      hq' : Membership.mem Mq 1
      i : Nat
      H : Not q.Monic
      ⊢ Membership.mem (HMul.hMul (HPow.hPow Mq p.natDegree) Mp) (p.coeff i)
    -/
  · rw [← one_mul (p.coeff i), ← one_pow p.natDegree]
    /-
      case neg
      R : Type u_2
      S : Type u_3
      inst✝² : CommRing R
      inst✝¹ : Ring S
      inst✝ : Algebra R S
      p q : Polynomial S
      Mp : Submodule R S
      hp : ∀ (i : Nat), Membership.mem Mp (p.coeff i)
      hp' : Membership.mem Mp 1
      Mq : Submodule R S
      hq : ∀ (i : Nat), Membership.mem Mq (q.coeff i)
      hq' : Membership.mem Mq 1
      i : Nat
      H : Not q.Monic
      ⊢ Membership.mem (HMul.hMul (HPow.hPow Mq p.natDegree) Mp) (HMul.hMul (HPow.hP …
    -/
    exact Submodule.mul_mem_mul (Submodule.pow_mem_pow Mq hq' _) (hp i)
    /-
      🎉 no goals
    -/


/-- For polynomials `p q : R[X]`, the coefficients of `p /ₘ q` can be written as sums of products of
coefficients of `p` and `q`.

Precisely, each summand needs at most one coefficient of `p` and `deg p` coefficients of `q`. -/
lemma coeff_divByMonic_mem_pow_natDegree_mul (p q : S[X])
    (Mp : Submodule R S) (hp : ∀ i, p.coeff i ∈ Mp) (hp' : 1 ∈ Mp)
    (Mq : Submodule R S) (hq : ∀ i, q.coeff i ∈ Mq) (hq' : 1 ∈ Mq) (i : ℕ) :
    (p /ₘ q).coeff i ∈ Mq ^ p.natDegree * Mp := by
  /-
    R : Type u_2
    S : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    p q : Polynomial S
    Mp : Submodule R S
    hp : ∀ (i : Nat), Membership.mem Mp (p.coeff i)
    hp' : Membership.mem Mp 1
    Mq : Submodule R S
    hq : ∀ (i : Nat), Membership.mem Mq (q.coeff i)
    hq' : Membership.mem Mq 1
    i : Nat
    ⊢ Membership.mem (HMul.hMul (HPow.hPow Mq p.natDegree) Mp) ((p.divByMonic q).c …
  -/
  delta divByMonic
  /-
    R : Type u_2
    S : Type u_3
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    p q : Polynomial S
    Mp : Submodule R S
    hp : ∀ (i : Nat), Membership.mem Mp (p.coeff i)
    hp' : Membership.mem Mp 1
    Mq : Submodule R S
    hq : ∀ (i : Nat), Membership.mem Mq (q.coeff i)
    hq' : Membership.mem Mq 1
    i : Nat
    ⊢ Membership.mem (HMul.hMul (HPow.hPow Mq p.natDegree) Mp) ((dite q.Monic (fun …
  -/
  split_ifs with H
    /-
      case pos
      R : Type u_2
      S : Type u_3
      inst✝² : CommRing R
      inst✝¹ : Ring S
      inst✝ : Algebra R S
      p q : Polynomial S
      Mp : Submodule R S
      hp : ∀ (i : Nat), Membership.mem Mp (p.coeff i)
      hp' : Membership.mem Mp 1
      Mq : Submodule R S
      hq : ∀ (i : Nat), Membership.mem Mq (q.coeff i)
      hq' : Membership.mem Mq 1
      i : Nat
      H : q.Monic
      ⊢ Membership.mem (HMul.hMul (HPow.hPow Mq p.natDegree) Mp) ((p.divModByMonicAu …
    -/
  · refine SetLike.le_def.mp ?_ (coeff_divModByMonicAux_mem_span_pow_mul_span (R := R) p q H i).1
    /-
      case pos
      R : Type u_2
      S : Type u_3
      inst✝² : CommRing R
      inst✝¹ : Ring S
      inst✝ : Algebra R S
      p q : Polynomial S
      Mp : Submodule R S
      hp : ∀ (i : Nat), Membership.mem Mp (p.coeff i)
      hp' : Membership.mem Mp 1
      Mq : Submodule R S
      hq : ∀ (i : Nat), Membership.mem Mq (q.coeff i)
      hq' : Membership.mem Mq 1
      i : Nat
      H : q.Monic
      ⊢ LE.le (HMul.hMul (HPow.hPow (Max.max 1 (Submodule.span R (Set.range q.coeff) …
    -/
               /-
                 🎉 no goals
               -/
    gcongr <;> exact sup_le (by simpa) (by simpa [Submodule.span_le, Set.range_subset_iff])
               /-
                 🎉 no goals
               -/
    /-
      case neg
      R : Type u_2
      S : Type u_3
      inst✝² : CommRing R
      inst✝¹ : Ring S
      inst✝ : Algebra R S
      p q : Polynomial S
      Mp : Submodule R S
      hp : ∀ (i : Nat), Membership.mem Mp (p.coeff i)
      hp' : Membership.mem Mp 1
      Mq : Submodule R S
      hq : ∀ (i : Nat), Membership.mem Mq (q.coeff i)
      hq' : Membership.mem Mq 1
      i : Nat
      H : Not q.Monic
      ⊢ Membership.mem (HMul.hMul (HPow.hPow Mq p.natDegree) Mp) (Polynomial.coeff 0 …
    -/
  · simp
    /-
      🎉 no goals
    -/


open Function Ideal in
lemma idealSpan_range_update_divByMonic (hij : i ≠ j) (v : ι → R[X]) (hi : (v i).Monic) :
    span (Set.range (Function.update v j (v j %ₘ v i))) = span (Set.range v) := by
  rw [modByMonic_eq_sub_mul_div _ hi, mul_comm, ← smul_eq_mul, Ideal.span, Ideal.span,
    Submodule.span_range_update_sub_smul hij]


