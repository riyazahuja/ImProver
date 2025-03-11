theorem natDegree_comp_le : natDegree (p.comp q) ≤ natDegree p * natDegree q :=
  letI := Classical.decEq R
                               /-
                                 R : Type u
                                 inst✝ : Semiring R
                                 p q : Polynomial R
                                 this : DecidableEq R := Classical.decEq R
                                 h0 : Eq (p.comp q) 0
                                 ⊢ LE.le (p.comp q).natDegree (HMul.hMul p.natDegree q.natDegree)
                               -/
  if h0 : p.comp q = 0 then by rw [h0, natDegree_zero]; exact Nat.zero_le _
                                                        /-
                                                          🎉 no goals
                                                        -/
  else
    WithBot.coe_le_coe.1 <|
      calc
        ↑(natDegree (p.comp q)) = degree (p.comp q) := (degree_eq_natDegree h0).symm
        _ = _ := congr_arg degree comp_eq_sum_left
        _ ≤ _ := degree_sum_le _ _
        _ ≤ _ :=
          Finset.sup_le fun n hn =>
            calc
              degree (C (coeff p n) * q ^ n) ≤ degree (C (coeff p n)) + degree (q ^ n) :=
                degree_mul_le _ _
              _ ≤ natDegree (C (coeff p n)) + n • degree q :=
                (add_le_add degree_le_natDegree (degree_pow_le _ _))
              _ ≤ natDegree (C (coeff p n)) + n • ↑(natDegree q) :=
                (add_le_add_left (nsmul_le_nsmul_right (@degree_le_natDegree _ _ q) n) _)
              _ = (n * natDegree q : ℕ) := by
                /-
                  R : Type u
                  inst✝ : Semiring R
                  p q : Polynomial R
                  this : DecidableEq R := Classical.decEq R
                  h0 : Not (Eq (p.comp q) 0)
                  n : Nat
                  hn : Membership.mem p.support n
                  ⊢ Eq (HAdd.hAdd (↑(Polynomial.C (p.coeff n)).natDegree) (HSMul.hSMul n ↑q.natD …
                -/
                rw [natDegree_C, Nat.cast_zero, zero_add, nsmul_eq_mul]
                /-
                  R : Type u
                  inst✝ : Semiring R
                  p q : Polynomial R
                  this : DecidableEq R := Classical.decEq R
                  h0 : Not (Eq (p.comp q) 0)
                  n : Nat
                  hn : Membership.mem p.support n
                  ⊢ Eq (HMul.hMul ↑n ↑q.natDegree) ↑(HMul.hMul n q.natDegree)
                -/
                simp
                /-
                  🎉 no goals
                -/
              _ ≤ (natDegree p * natDegree q : ℕ) :=
                WithBot.coe_le_coe.2 <|
                  mul_le_mul_of_nonneg_right (le_natDegree_of_ne_zero (mem_support_iff.1 hn))
                    (Nat.zero_le _)


theorem natDegree_comp_eq_of_mul_ne_zero (h : p.leadingCoeff * q.leadingCoeff ^ p.natDegree ≠ 0) :
    natDegree (p.comp q) = natDegree p * natDegree q := by
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    h : Ne (HMul.hMul p.leadingCoeff (HPow.hPow q.leadingCoeff p.natDegree)) 0
    ⊢ Eq (p.comp q).natDegree (HMul.hMul p.natDegree q.natDegree)
  -/
  by_cases hq : natDegree q = 0
    /-
      case pos
      R : Type u
      inst✝ : Semiring R
      p q : Polynomial R
      h : Ne (HMul.hMul p.leadingCoeff (HPow.hPow q.leadingCoeff p.natDegree)) 0
      hq : Eq q.natDegree 0
      ⊢ Eq (p.comp q).natDegree (HMul.hMul p.natDegree q.natDegree)
    -/
  · exact le_antisymm natDegree_comp_le (by simp [hq])
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    h : Ne (HMul.hMul p.leadingCoeff (HPow.hPow q.leadingCoeff p.natDegree)) 0
    hq : Not (Eq q.natDegree 0)
    ⊢ Eq (p.comp q).natDegree (HMul.hMul p.natDegree q.natDegree)
  -/
  apply natDegree_eq_of_le_of_coeff_ne_zero natDegree_comp_le
  /-
    case neg
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    h : Ne (HMul.hMul p.leadingCoeff (HPow.hPow q.leadingCoeff p.natDegree)) 0
    hq : Not (Eq q.natDegree 0)
    ⊢ Ne ((p.comp q).coeff (HMul.hMul p.natDegree q.natDegree)) 0
  -/
  rwa [coeff_comp_degree_mul_degree hq]
  /-
    🎉 no goals
  -/


theorem degree_pos_of_root {p : R[X]} (hp : p ≠ 0) (h : IsRoot p a) : 0 < degree p :=
  lt_of_not_ge fun hlt => by
    /-
      R : Type u
      a : R
      inst✝ : Semiring R
      p : Polynomial R
      hp : Ne p 0
      h : p.IsRoot a
      hlt : GE.ge 0 p.degree
      ⊢ False
    -/
    have := eq_C_of_degree_le_zero hlt
    /-
      R : Type u
      a : R
      inst✝ : Semiring R
      p : Polynomial R
      hp : Ne p 0
      h : p.IsRoot a
      hlt : GE.ge 0 p.degree
      this : Eq p (Polynomial.C (p.coeff 0))
      ⊢ False
    -/
    rw [IsRoot, this, eval_C] at h
    /-
      R : Type u
      a : R
      inst✝ : Semiring R
      p : Polynomial R
      hp : Ne p 0
      h : Eq (p.coeff 0) 0
      hlt : GE.ge 0 p.degree
      this : Eq p (Polynomial.C (p.coeff 0))
      ⊢ False
    -/
    simp only [h, RingHom.map_zero] at this
    /-
      R : Type u
      a : R
      inst✝ : Semiring R
      p : Polynomial R
      hp : Ne p 0
      h : Eq (p.coeff 0) 0
      hlt : GE.ge 0 p.degree
      this : Eq p 0
      ⊢ False
    -/
    exact hp this
    /-
      🎉 no goals
    -/


theorem natDegree_le_iff_coeff_eq_zero : p.natDegree ≤ n ↔ ∀ N : ℕ, n < N → p.coeff N = 0 := by
  /-
    R : Type u
    n : Nat
    inst✝ : Semiring R
    p : Polynomial R
    ⊢ Iff (LE.le p.natDegree n) (∀ (N : Nat), LT.lt n N → Eq (p.coeff N) 0)
  -/
  simp_rw [natDegree_le_iff_degree_le, degree_le_iff_coeff_zero, Nat.cast_lt]
  /-
    🎉 no goals
  -/


theorem natDegree_add_le_iff_left {n : ℕ} (p q : R[X]) (qn : q.natDegree ≤ n) :
    (p + q).natDegree ≤ n ↔ p.natDegree ≤ n := by
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    p q : Polynomial R
    qn : LE.le q.natDegree n
    ⊢ Iff (LE.le (HAdd.hAdd p q).natDegree n) (LE.le p.natDegree n)
  -/
  refine ⟨fun h => ?_, fun h => natDegree_add_le_of_degree_le h qn⟩
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    p q : Polynomial R
    qn : LE.le q.natDegree n
    h : LE.le (HAdd.hAdd p q).natDegree n
    ⊢ LE.le p.natDegree n
  -/
  refine natDegree_le_iff_coeff_eq_zero.mpr fun m hm => ?_
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    p q : Polynomial R
    qn : LE.le q.natDegree n
    h : LE.le (HAdd.hAdd p q).natDegree n
    m : Nat
    hm : LT.lt n m
    ⊢ Eq (p.coeff m) 0
  -/
  convert natDegree_le_iff_coeff_eq_zero.mp h m hm using 1
  /-
    case h.e'_2
    R : Type u
    inst✝ : Semiring R
    n : Nat
    p q : Polynomial R
    qn : LE.le q.natDegree n
    h : LE.le (HAdd.hAdd p q).natDegree n
    m : Nat
    hm : LT.lt n m
    ⊢ Eq (p.coeff m) ((HAdd.hAdd p q).coeff m)
  -/
  rw [coeff_add, natDegree_le_iff_coeff_eq_zero.mp qn _ hm, add_zero]
  /-
    🎉 no goals
  -/


theorem natDegree_add_le_iff_right {n : ℕ} (p q : R[X]) (pn : p.natDegree ≤ n) :
    (p + q).natDegree ≤ n ↔ q.natDegree ≤ n := by
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    p q : Polynomial R
    pn : LE.le p.natDegree n
    ⊢ Iff (LE.le (HAdd.hAdd p q).natDegree n) (LE.le q.natDegree n)
  -/
  rw [add_comm]
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    p q : Polynomial R
    pn : LE.le p.natDegree n
    ⊢ Iff (LE.le (HAdd.hAdd q p).natDegree n) (LE.le q.natDegree n)
  -/
  exact natDegree_add_le_iff_left _ _ pn
  /-
    🎉 no goals
  -/

-- TODO: Do we really want the following two lemmas? They are straightforward consequences of a
-- more atomic lemma

theorem natDegree_C_mul_le (a : R) (f : R[X]) : (C a * f).natDegree ≤ f.natDegree := by
  /-
    R : Type u
    inst✝ : Semiring R
    a : R
    f : Polynomial R
    ⊢ LE.le (HMul.hMul (Polynomial.C a) f).natDegree f.natDegree
  -/
  simpa using natDegree_mul_le (p := C a)
  /-
    🎉 no goals
  -/


theorem natDegree_mul_C_le (f : R[X]) (a : R) : (f * C a).natDegree ≤ f.natDegree := by
  /-
    R : Type u
    inst✝ : Semiring R
    f : Polynomial R
    a : R
    ⊢ LE.le (HMul.hMul f (Polynomial.C a)).natDegree f.natDegree
  -/
  simpa using natDegree_mul_le (q := C a)
  /-
    🎉 no goals
  -/


theorem eq_natDegree_of_le_mem_support (pn : p.natDegree ≤ n) (ns : n ∈ p.support) :
    p.natDegree = n :=
  le_antisymm pn (le_natDegree_of_mem_supp _ ns)


theorem natDegree_C_mul_eq_of_mul_eq_one {ai : R} (au : ai * a = 1) :
    (C a * p).natDegree = p.natDegree :=
  le_antisymm (natDegree_C_mul_le a p)
    (calc
                                            /-
                                              R : Type u
                                              a : R
                                              inst✝ : Semiring R
                                              p : Polynomial R
                                              ai : R
                                              au : Eq (HMul.hMul ai a) 1
                                              ⊢ Eq p.natDegree (HMul.hMul 1 p).natDegree
                                            -/
      p.natDegree = (1 * p).natDegree := by nth_rw 1 [← one_mul p]
                                            /-
                                              🎉 no goals
                                            -/
                                             /-
                                               R : Type u
                                               a : R
                                               inst✝ : Semiring R
                                               p : Polynomial R
                                               ai : R
                                               au : Eq (HMul.hMul ai a) 1
                                               ⊢ Eq (HMul.hMul 1 p).natDegree (HMul.hMul (Polynomial.C ai) (HMul.hMul (Polyno …
                                             -/
      _ = (C ai * (C a * p)).natDegree := by rw [← C_1, ← au, RingHom.map_mul, ← mul_assoc]
                                             /-
                                               🎉 no goals
                                             -/
      _ ≤ (C a * p).natDegree := natDegree_C_mul_le ai (C a * p))


theorem natDegree_mul_C_eq_of_mul_eq_one {ai : R} (au : a * ai = 1) :
    (p * C a).natDegree = p.natDegree :=
  le_antisymm (natDegree_mul_C_le p a)
    (calc
                                            /-
                                              R : Type u
                                              a : R
                                              inst✝ : Semiring R
                                              p : Polynomial R
                                              ai : R
                                              au : Eq (HMul.hMul a ai) 1
                                              ⊢ Eq p.natDegree (HMul.hMul p 1).natDegree
                                            -/
      p.natDegree = (p * 1).natDegree := by nth_rw 1 [← mul_one p]
                                            /-
                                              🎉 no goals
                                            -/
                                           /-
                                             R : Type u
                                             a : R
                                             inst✝ : Semiring R
                                             p : Polynomial R
                                             ai : R
                                             au : Eq (HMul.hMul a ai) 1
                                             ⊢ Eq (HMul.hMul p 1).natDegree (HMul.hMul (HMul.hMul p (Polynomial.C a)) (Poly …
                                           -/
      _ = (p * C a * C ai).natDegree := by rw [← C_1, ← au, RingHom.map_mul, ← mul_assoc]
                                           /-
                                             🎉 no goals
                                           -/
      _ ≤ (p * C a).natDegree := natDegree_mul_C_le (p * C a) ai)


/-- Although not explicitly stated, the assumptions of lemma `nat_degree_mul_C_eq_of_mul_ne_zero`
force the polynomial `p` to be non-zero, via `p.leading_coeff ≠ 0`.
-/
theorem natDegree_mul_C_eq_of_mul_ne_zero (h : p.leadingCoeff * a ≠ 0) :
    (p * C a).natDegree = p.natDegree := by
  /-
    R : Type u
    a : R
    inst✝ : Semiring R
    p : Polynomial R
    h : Ne (HMul.hMul p.leadingCoeff a) 0
    ⊢ Eq (HMul.hMul p (Polynomial.C a)).natDegree p.natDegree
  -/
  refine eq_natDegree_of_le_mem_support (natDegree_mul_C_le p a) ?_
  /-
    R : Type u
    a : R
    inst✝ : Semiring R
    p : Polynomial R
    h : Ne (HMul.hMul p.leadingCoeff a) 0
    ⊢ Membership.mem (HMul.hMul p (Polynomial.C a)).support p.natDegree
  -/
  refine mem_support_iff.mpr ?_
  /-
    R : Type u
    a : R
    inst✝ : Semiring R
    p : Polynomial R
    h : Ne (HMul.hMul p.leadingCoeff a) 0
    ⊢ Ne ((HMul.hMul p (Polynomial.C a)).coeff p.natDegree) 0
  -/
  rwa [coeff_mul_C]
  /-
    🎉 no goals
  -/


/-- Although not explicitly stated, the assumptions of lemma `nat_degree_C_mul_eq_of_mul_ne_zero`
force the polynomial `p` to be non-zero, via `p.leading_coeff ≠ 0`.
-/
theorem natDegree_C_mul_of_mul_ne_zero (h : a * p.leadingCoeff ≠ 0) :
    (C a * p).natDegree = p.natDegree := by
  /-
    R : Type u
    a : R
    inst✝ : Semiring R
    p : Polynomial R
    h : Ne (HMul.hMul a p.leadingCoeff) 0
    ⊢ Eq (HMul.hMul (Polynomial.C a) p).natDegree p.natDegree
  -/
  refine eq_natDegree_of_le_mem_support (natDegree_C_mul_le a p) ?_
  /-
    R : Type u
    a : R
    inst✝ : Semiring R
    p : Polynomial R
    h : Ne (HMul.hMul a p.leadingCoeff) 0
    ⊢ Membership.mem (HMul.hMul (Polynomial.C a) p).support p.natDegree
  -/
  refine mem_support_iff.mpr ?_
  /-
    R : Type u
    a : R
    inst✝ : Semiring R
    p : Polynomial R
    h : Ne (HMul.hMul a p.leadingCoeff) 0
    ⊢ Ne ((HMul.hMul (Polynomial.C a) p).coeff p.natDegree) 0
  -/
  rwa [coeff_C_mul]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2025-01-03")]
alias natDegree_C_mul_eq_of_mul_ne_zero := natDegree_C_mul_of_mul_ne_zero


lemma degree_C_mul_of_mul_ne_zero (h : a * p.leadingCoeff ≠ 0) : (C a * p).degree = p.degree := by
  /-
    R : Type u
    a : R
    inst✝ : Semiring R
    p : Polynomial R
    h : Ne (HMul.hMul a p.leadingCoeff) 0
    ⊢ Eq (HMul.hMul (Polynomial.C a) p).degree p.degree
  -/
  rw [degree_mul' (by simpa)]; simp [left_ne_zero_of_mul h]
                               /-
                                 🎉 no goals
                               -/


theorem natDegree_add_coeff_mul (f g : R[X]) :
    (f * g).coeff (f.natDegree + g.natDegree) = f.coeff f.natDegree * g.coeff g.natDegree := by
  /-
    R : Type u
    inst✝ : Semiring R
    f g : Polynomial R
    ⊢ Eq ((HMul.hMul f g).coeff (HAdd.hAdd f.natDegree g.natDegree)) (HMul.hMul (f …
  -/
  simp only [coeff_natDegree, coeff_mul_degree_add_degree]
  /-
    🎉 no goals
  -/


theorem natDegree_lt_coeff_mul (h : p.natDegree + q.natDegree < m + n) :
    (p * q).coeff (m + n) = 0 :=
  coeff_eq_zero_of_natDegree_lt (natDegree_mul_le.trans_lt h)


theorem coeff_mul_of_natDegree_le (pm : p.natDegree ≤ m) (qn : q.natDegree ≤ n) :
    (p * q).coeff (m + n) = p.coeff m * q.coeff n := by
  /-
    R : Type u
    m n : Nat
    inst✝ : Semiring R
    p q : Polynomial R
    pm : LE.le p.natDegree m
    qn : LE.le q.natDegree n
    ⊢ Eq ((HMul.hMul p q).coeff (HAdd.hAdd m n)) (HMul.hMul (p.coeff m) (q.coeff n))
  -/
  simp_rw [← Polynomial.toFinsupp_apply, toFinsupp_mul]
  /-
    R : Type u
    m n : Nat
    inst✝ : Semiring R
    p q : Polynomial R
    pm : LE.le p.natDegree m
    qn : LE.le q.natDegree n
    ⊢ Eq ((HMul.hMul p.toFinsupp q.toFinsupp) (HAdd.hAdd m n)) (HMul.hMul (p.toFin …
  -/
  refine AddMonoidAlgebra.apply_add_of_supDegree_le ?_ Function.injective_id ?_ ?_
    /-
      case refine_1
      R : Type u
      m n : Nat
      inst✝ : Semiring R
      p q : Polynomial R
      pm : LE.le p.natDegree m
      qn : LE.le q.natDegree n
      ⊢ ∀ (a1 a2 : Nat), Eq (id (HAdd.hAdd a1 a2)) (HAdd.hAdd (id a1) (id a2))
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u
      m n : Nat
      inst✝ : Semiring R
      p q : Polynomial R
      pm : LE.le p.natDegree m
      qn : LE.le q.natDegree n
      ⊢ LE.le (AddMonoidAlgebra.supDegree id p.toFinsupp) (id m)
    -/
  · rwa [supDegree_eq_natDegree, id_eq]
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      R : Type u
      m n : Nat
      inst✝ : Semiring R
      p q : Polynomial R
      pm : LE.le p.natDegree m
      qn : LE.le q.natDegree n
      ⊢ LE.le (AddMonoidAlgebra.supDegree id q.toFinsupp) (id n)
    -/
  · rwa [supDegree_eq_natDegree, id_eq]
    /-
      🎉 no goals
    -/


theorem coeff_pow_of_natDegree_le (pn : p.natDegree ≤ n) :
    (p ^ m).coeff (m * n) = p.coeff n ^ m := by
  /-
    R : Type u
    m n : Nat
    inst✝ : Semiring R
    p : Polynomial R
    pn : LE.le p.natDegree n
    ⊢ Eq ((HPow.hPow p m).coeff (HMul.hMul m n)) (HPow.hPow (p.coeff n) m)
  -/
  induction' m with m hm
    /-
      case zero
      R : Type u
      m n : Nat
      inst✝ : Semiring R
      p : Polynomial R
      pn : LE.le p.natDegree n
      ⊢ Eq ((HPow.hPow p 0).coeff (HMul.hMul 0 n)) (HPow.hPow (p.coeff n) 0)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case succ
      R : Type u
      m✝ n : Nat
      inst✝ : Semiring R
      p : Polynomial R
      pn : LE.le p.natDegree n
      m : Nat
      hm : Eq ((HPow.hPow p m).coeff (HMul.hMul m n)) (HPow.hPow (p.coeff n) m)
      ⊢ Eq ((HPow.hPow p (HAdd.hAdd m 1)).coeff (HMul.hMul (HAdd.hAdd m 1) n)) (HPow …
    -/
  · rw [pow_succ, pow_succ, ← hm, Nat.succ_mul, coeff_mul_of_natDegree_le _ pn]
    /-
      R : Type u
      m✝ n : Nat
      inst✝ : Semiring R
      p : Polynomial R
      pn : LE.le p.natDegree n
      m : Nat
      hm : Eq ((HPow.hPow p m).coeff (HMul.hMul m n)) (HPow.hPow (p.coeff n) m)
      ⊢ LE.le (HPow.hPow p m).natDegree (HMul.hMul m n)
    -/
    refine natDegree_pow_le.trans (le_trans ?_ (le_refl _))
    /-
      R : Type u
      m✝ n : Nat
      inst✝ : Semiring R
      p : Polynomial R
      pn : LE.le p.natDegree n
      m : Nat
      hm : Eq ((HPow.hPow p m).coeff (HMul.hMul m n)) (HPow.hPow (p.coeff n) m)
      ⊢ LE.le (HMul.hMul m p.natDegree) (HMul.hMul m n)
    -/
    exact mul_le_mul_of_nonneg_left pn m.zero_le
    /-
      🎉 no goals
    -/


theorem coeff_pow_eq_ite_of_natDegree_le_of_le {o : ℕ}
    (pn : natDegree p ≤ n) (mno : m * n ≤ o) :
    coeff (p ^ m) o = if o = m * n then (coeff p n) ^ m else 0 := by
  /-
    R : Type u
    m n : Nat
    inst✝ : Semiring R
    p : Polynomial R
    o : Nat
    pn : LE.le p.natDegree n
    mno : LE.le (HMul.hMul m n) o
    ⊢ Eq ((HPow.hPow p m).coeff o) (ite (Eq o (HMul.hMul m n)) (HPow.hPow (p.coeff …
  -/
  rcases eq_or_ne o (m * n) with rfl | h
    /-
      case inl
      R : Type u
      m n : Nat
      inst✝ : Semiring R
      p : Polynomial R
      pn : LE.le p.natDegree n
      mno : LE.le (HMul.hMul m n) (HMul.hMul m n)
      ⊢ Eq ((HPow.hPow p m).coeff (HMul.hMul m n)) (ite (Eq (HMul.hMul m n) (HMul.hM …
    -/
  · simpa only [ite_true] using coeff_pow_of_natDegree_le pn
    /-
      🎉 no goals
    -/
  · simpa only [h, ite_false] using coeff_eq_zero_of_natDegree_lt <|
      lt_of_le_of_lt (natDegree_pow_le_of_le m pn) (lt_of_le_of_ne mno h.symm)


theorem coeff_add_eq_left_of_lt (qn : q.natDegree < n) : (p + q).coeff n = p.coeff n :=
  (coeff_add _ _ _).trans <|
    (congr_arg _ <| coeff_eq_zero_of_natDegree_lt <| qn).trans <| add_zero _


theorem coeff_add_eq_right_of_lt (pn : p.natDegree < n) : (p + q).coeff n = q.coeff n := by
  /-
    R : Type u
    n : Nat
    inst✝ : Semiring R
    p q : Polynomial R
    pn : LT.lt p.natDegree n
    ⊢ Eq ((HAdd.hAdd p q).coeff n) (q.coeff n)
  -/
  rw [add_comm]
  /-
    R : Type u
    n : Nat
    inst✝ : Semiring R
    p q : Polynomial R
    pn : LT.lt p.natDegree n
    ⊢ Eq ((HAdd.hAdd q p).coeff n) (q.coeff n)
  -/
  exact coeff_add_eq_left_of_lt pn
  /-
    🎉 no goals
  -/


theorem degree_sum_eq_of_disjoint (f : S → R[X]) (s : Finset S)
    (h : Set.Pairwise { i | i ∈ s ∧ f i ≠ 0 } (Ne on degree ∘ f)) :
    degree (s.sum f) = s.sup fun i => degree (f i) := by
  classical
  induction' s using Finset.induction_on with x s hx IH
  · simp
  · simp only [hx, Finset.sum_insert, not_false_iff, Finset.sup_insert]
    specialize IH (h.mono fun _ => by simp +contextual)
    rcases lt_trichotomy (degree (f x)) (degree (s.sum f)) with (H | H | H)
    · rw [← IH, sup_eq_right.mpr H.le, degree_add_eq_right_of_degree_lt H]
    · rcases s.eq_empty_or_nonempty with (rfl | hs)
      · simp
      obtain ⟨y, hy, hy'⟩ := Finset.exists_mem_eq_sup s hs fun i => degree (f i)
      rw [IH, hy'] at H
      by_cases hx0 : f x = 0
      · simp [hx0, IH]
      have hy0 : f y ≠ 0 := by
        contrapose! H
        simpa [H, degree_eq_bot] using hx0
      refine absurd H (h ?_ ?_ fun H => hx ?_)
      · simp [hx0]
      · simp [hy, hy0]
      · exact H.symm ▸ hy
    · rw [← IH, sup_eq_left.mpr H.le, degree_add_eq_left_of_degree_lt H]


theorem natDegree_sum_eq_of_disjoint (f : S → R[X]) (s : Finset S)
    (h : Set.Pairwise { i | i ∈ s ∧ f i ≠ 0 } (Ne on natDegree ∘ f)) :
    natDegree (s.sum f) = s.sup fun i => natDegree (f i) := by
  /-
    R : Type u
    S : Type v
    inst✝ : Semiring R
    f : S → Polynomial R
    s : Finset S
    h : (setOf fun i => And (Membership.mem s i) (Ne (f i) 0)).Pairwise (Function. …
    ⊢ Eq (s.sum f).natDegree (s.sup fun i => (f i).natDegree)
  -/
  by_cases H : ∃ x ∈ s, f x ≠ 0
    /-
      case pos
      R : Type u
      S : Type v
      inst✝ : Semiring R
      f : S → Polynomial R
      s : Finset S
      h : (setOf fun i => And (Membership.mem s i) (Ne (f i) 0)).Pairwise (Function. …
      H : Exists fun x => And (Membership.mem s x) (Ne (f x) 0)
      ⊢ Eq (s.sum f).natDegree (s.sup fun i => (f i).natDegree)
    -/
  · obtain ⟨x, hx, hx'⟩ := H
    /-
      case pos.intro.intro
      R : Type u
      S : Type v
      inst✝ : Semiring R
      f : S → Polynomial R
      s : Finset S
      h : (setOf fun i => And (Membership.mem s i) (Ne (f i) 0)).Pairwise (Function. …
      x : S
      hx : Membership.mem s x
      hx' : Ne (f x) 0
      ⊢ Eq (s.sum f).natDegree (s.sup fun i => (f i).natDegree)
    -/
    have hs : s.Nonempty := ⟨x, hx⟩
    /-
      case pos.intro.intro
      R : Type u
      S : Type v
      inst✝ : Semiring R
      f : S → Polynomial R
      s : Finset S
      h : (setOf fun i => And (Membership.mem s i) (Ne (f i) 0)).Pairwise (Function. …
      x : S
      hx : Membership.mem s x
      hx' : Ne (f x) 0
      hs : s.Nonempty
      ⊢ Eq (s.sum f).natDegree (s.sup fun i => (f i).natDegree)
    -/
    refine natDegree_eq_of_degree_eq_some ?_
    /-
      case pos.intro.intro
      R : Type u
      S : Type v
      inst✝ : Semiring R
      f : S → Polynomial R
      s : Finset S
      h : (setOf fun i => And (Membership.mem s i) (Ne (f i) 0)).Pairwise (Function. …
      x : S
      hx : Membership.mem s x
      hx' : Ne (f x) 0
      hs : s.Nonempty
      ⊢ Eq (s.sum f).degree ↑(s.sup fun i => (f i).natDegree)
    -/
    rw [degree_sum_eq_of_disjoint]
    · rw [← Finset.sup'_eq_sup hs, ← Finset.sup'_eq_sup hs,
        Nat.cast_withBot, Finset.coe_sup' hs, ←
        Finset.sup'_eq_sup hs]
      /-
        case pos.intro.intro
        R : Type u
        S : Type v
        inst✝ : Semiring R
        f : S → Polynomial R
        s : Finset S
        h : (setOf fun i => And (Membership.mem s i) (Ne (f i) 0)).Pairwise (Function. …
        x : S
        hx : Membership.mem s x
        hx' : Ne (f x) 0
        hs : s.Nonempty
        ⊢ Eq (s.sup' hs fun i => (f i).degree) (s.sup' hs (Function.comp WithBot.some  …
      -/
      refine le_antisymm ?_ ?_
        /-
          case pos.intro.intro.refine_1
          R : Type u
          S : Type v
          inst✝ : Semiring R
          f : S → Polynomial R
          s : Finset S
          h : (setOf fun i => And (Membership.mem s i) (Ne (f i) 0)).Pairwise (Function. …
          x : S
          hx : Membership.mem s x
          hx' : Ne (f x) 0
          hs : s.Nonempty
          ⊢ LE.le (s.sup' hs fun i => (f i).degree) (s.sup' hs (Function.comp WithBot.so …
        -/
      · rw [Finset.sup'_le_iff]
        /-
          case pos.intro.intro.refine_1
          R : Type u
          S : Type v
          inst✝ : Semiring R
          f : S → Polynomial R
          s : Finset S
          h : (setOf fun i => And (Membership.mem s i) (Ne (f i) 0)).Pairwise (Function. …
          x : S
          hx : Membership.mem s x
          hx' : Ne (f x) 0
          hs : s.Nonempty
          ⊢ ∀ (b : S), Membership.mem s b → LE.le (f b).degree (s.sup' hs (Function.comp …
        -/
        intro b hb
        /-
          case pos.intro.intro.refine_1
          R : Type u
          S : Type v
          inst✝ : Semiring R
          f : S → Polynomial R
          s : Finset S
          h : (setOf fun i => And (Membership.mem s i) (Ne (f i) 0)).Pairwise (Function. …
          x : S
          hx : Membership.mem s x
          hx' : Ne (f x) 0
          hs : s.Nonempty
          b : S
          hb : Membership.mem s b
          ⊢ LE.le (f b).degree (s.sup' hs (Function.comp WithBot.some fun i => (f i).nat …
        -/
        by_cases hb' : f b = 0
          /-
            case pos
            R : Type u
            S : Type v
            inst✝ : Semiring R
            f : S → Polynomial R
            s : Finset S
            h : (setOf fun i => And (Membership.mem s i) (Ne (f i) 0)).Pairwise (Function. …
            x : S
            hx : Membership.mem s x
            hx' : Ne (f x) 0
            hs : s.Nonempty
            b : S
            hb : Membership.mem s b
            hb' : Eq (f b) 0
            ⊢ LE.le (f b).degree (s.sup' hs (Function.comp WithBot.some fun i => (f i).nat …
          -/
        · simpa [hb'] using hs
          /-
            🎉 no goals
          -/
        /-
          case neg
          R : Type u
          S : Type v
          inst✝ : Semiring R
          f : S → Polynomial R
          s : Finset S
          h : (setOf fun i => And (Membership.mem s i) (Ne (f i) 0)).Pairwise (Function. …
          x : S
          hx : Membership.mem s x
          hx' : Ne (f x) 0
          hs : s.Nonempty
          b : S
          hb : Membership.mem s b
          hb' : Not (Eq (f b) 0)
          ⊢ LE.le (f b).degree (s.sup' hs (Function.comp WithBot.some fun i => (f i).nat …
        -/
        rw [degree_eq_natDegree hb', Nat.cast_withBot]
        /-
          case neg
          R : Type u
          S : Type v
          inst✝ : Semiring R
          f : S → Polynomial R
          s : Finset S
          h : (setOf fun i => And (Membership.mem s i) (Ne (f i) 0)).Pairwise (Function. …
          x : S
          hx : Membership.mem s x
          hx' : Ne (f x) 0
          hs : s.Nonempty
          b : S
          hb : Membership.mem s b
          hb' : Not (Eq (f b) 0)
          ⊢ LE.le (↑(f b).natDegree) (s.sup' hs (Function.comp WithBot.some fun i => (f  …
        -/
        exact Finset.le_sup' (fun i : S => (natDegree (f i) : WithBot ℕ)) hb
        /-
          🎉 no goals
        -/
        /-
          case pos.intro.intro.refine_2
          R : Type u
          S : Type v
          inst✝ : Semiring R
          f : S → Polynomial R
          s : Finset S
          h : (setOf fun i => And (Membership.mem s i) (Ne (f i) 0)).Pairwise (Function. …
          x : S
          hx : Membership.mem s x
          hx' : Ne (f x) 0
          hs : s.Nonempty
          ⊢ LE.le (s.sup' hs (Function.comp WithBot.some fun i => (f i).natDegree)) (s.s …
        -/
      · rw [Finset.sup'_le_iff]
        /-
          case pos.intro.intro.refine_2
          R : Type u
          S : Type v
          inst✝ : Semiring R
          f : S → Polynomial R
          s : Finset S
          h : (setOf fun i => And (Membership.mem s i) (Ne (f i) 0)).Pairwise (Function. …
          x : S
          hx : Membership.mem s x
          hx' : Ne (f x) 0
          hs : s.Nonempty
          ⊢ ∀ (b : S), Membership.mem s b → LE.le (Function.comp WithBot.some (fun i =>  …
        -/
        intro b hb
        /-
          case pos.intro.intro.refine_2
          R : Type u
          S : Type v
          inst✝ : Semiring R
          f : S → Polynomial R
          s : Finset S
          h : (setOf fun i => And (Membership.mem s i) (Ne (f i) 0)).Pairwise (Function. …
          x : S
          hx : Membership.mem s x
          hx' : Ne (f x) 0
          hs : s.Nonempty
          b : S
          hb : Membership.mem s b
          ⊢ LE.le (Function.comp WithBot.some (fun i => (f i).natDegree) b) (s.sup' hs f …
        -/
        simp only [Finset.le_sup'_iff, exists_prop, Function.comp_apply]
        /-
          case pos.intro.intro.refine_2
          R : Type u
          S : Type v
          inst✝ : Semiring R
          f : S → Polynomial R
          s : Finset S
          h : (setOf fun i => And (Membership.mem s i) (Ne (f i) 0)).Pairwise (Function. …
          x : S
          hx : Membership.mem s x
          hx' : Ne (f x) 0
          hs : s.Nonempty
          b : S
          hb : Membership.mem s b
          ⊢ Exists fun b_1 => And (Membership.mem s b_1) (LE.le (↑(f b).natDegree) (f b_ …
        -/
        by_cases hb' : f b = 0
          /-
            case pos
            R : Type u
            S : Type v
            inst✝ : Semiring R
            f : S → Polynomial R
            s : Finset S
            h : (setOf fun i => And (Membership.mem s i) (Ne (f i) 0)).Pairwise (Function. …
            x : S
            hx : Membership.mem s x
            hx' : Ne (f x) 0
            hs : s.Nonempty
            b : S
            hb : Membership.mem s b
            hb' : Eq (f b) 0
            ⊢ Exists fun b_1 => And (Membership.mem s b_1) (LE.le (↑(f b).natDegree) (f b_ …
          -/
        · refine ⟨x, hx, ?_⟩
          /-
            case pos
            R : Type u
            S : Type v
            inst✝ : Semiring R
            f : S → Polynomial R
            s : Finset S
            h : (setOf fun i => And (Membership.mem s i) (Ne (f i) 0)).Pairwise (Function. …
            x : S
            hx : Membership.mem s x
            hx' : Ne (f x) 0
            hs : s.Nonempty
            b : S
            hb : Membership.mem s b
            hb' : Eq (f b) 0
            ⊢ LE.le (↑(f b).natDegree) (f x).degree
          -/
          contrapose! hx'
          /-
            case pos
            R : Type u
            S : Type v
            inst✝ : Semiring R
            f : S → Polynomial R
            s : Finset S
            h : (setOf fun i => And (Membership.mem s i) (Ne (f i) 0)).Pairwise (Function. …
            x : S
            hx : Membership.mem s x
            hs : s.Nonempty
            b : S
            hb : Membership.mem s b
            hb' : Eq (f b) 0
            hx' : LT.lt (f x).degree ↑(f b).natDegree
            ⊢ Eq (f x) 0
          -/
          simpa [← Nat.cast_withBot, hb', degree_eq_bot] using hx'
          /-
            🎉 no goals
          -/
        /-
          case neg
          R : Type u
          S : Type v
          inst✝ : Semiring R
          f : S → Polynomial R
          s : Finset S
          h : (setOf fun i => And (Membership.mem s i) (Ne (f i) 0)).Pairwise (Function. …
          x : S
          hx : Membership.mem s x
          hx' : Ne (f x) 0
          hs : s.Nonempty
          b : S
          hb : Membership.mem s b
          hb' : Not (Eq (f b) 0)
          ⊢ Exists fun b_1 => And (Membership.mem s b_1) (LE.le (↑(f b).natDegree) (f b_ …
        -/
        exact ⟨b, hb, (degree_eq_natDegree hb').ge⟩
        /-
          🎉 no goals
        -/
      /-
        case pos.intro.intro.h
        R : Type u
        S : Type v
        inst✝ : Semiring R
        f : S → Polynomial R
        s : Finset S
        h : (setOf fun i => And (Membership.mem s i) (Ne (f i) 0)).Pairwise (Function. …
        x : S
        hx : Membership.mem s x
        hx' : Ne (f x) 0
        hs : s.Nonempty
        ⊢ (setOf fun i => And (Membership.mem s i) (Ne (f i) 0)).Pairwise (Function.on …
      -/
    · exact h.imp fun x y hxy hxy' => hxy (natDegree_eq_of_degree_eq hxy')
      /-
        🎉 no goals
      -/
    /-
      case neg
      R : Type u
      S : Type v
      inst✝ : Semiring R
      f : S → Polynomial R
      s : Finset S
      h : (setOf fun i => And (Membership.mem s i) (Ne (f i) 0)).Pairwise (Function. …
      H : Not (Exists fun x => And (Membership.mem s x) (Ne (f x) 0))
      ⊢ Eq (s.sum f).natDegree (s.sup fun i => (f i).natDegree)
    -/
  · push_neg at H
    /-
      case neg
      R : Type u
      S : Type v
      inst✝ : Semiring R
      f : S → Polynomial R
      s : Finset S
      h : (setOf fun i => And (Membership.mem s i) (Ne (f i) 0)).Pairwise (Function. …
      H : ∀ (x : S), Membership.mem s x → Eq (f x) 0
      ⊢ Eq (s.sum f).natDegree (s.sup fun i => (f i).natDegree)
    -/
    rw [Finset.sum_eq_zero H, natDegree_zero, eq_comm, show 0 = ⊥ from rfl, Finset.sup_eq_bot_iff]
    /-
      case neg
      R : Type u
      S : Type v
      inst✝ : Semiring R
      f : S → Polynomial R
      s : Finset S
      h : (setOf fun i => And (Membership.mem s i) (Ne (f i) 0)).Pairwise (Function. …
      H : ∀ (x : S), Membership.mem s x → Eq (f x) 0
      ⊢ ∀ (s_1 : S), Membership.mem s s_1 → Eq (f s_1).natDegree Bot.bot
    -/
    intro x hx
    /-
      case neg
      R : Type u
      S : Type v
      inst✝ : Semiring R
      f : S → Polynomial R
      s : Finset S
      h : (setOf fun i => And (Membership.mem s i) (Ne (f i) 0)).Pairwise (Function. …
      H : ∀ (x : S), Membership.mem s x → Eq (f x) 0
      x : S
      hx : Membership.mem s x
      ⊢ Eq (f x).natDegree Bot.bot
    -/
    simp [H x hx]
    /-
      🎉 no goals
    -/


theorem natDegree_pos_of_eval₂_root {p : R[X]} (hp : p ≠ 0) (f : R →+* S) {z : S}
    (hz : eval₂ f z p = 0) (inj : ∀ x : R, f x = 0 → x = 0) : 0 < natDegree p :=
  lt_of_not_ge fun hlt => by
    /-
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      inst✝ : Semiring S
      p : Polynomial R
      hp : Ne p 0
      f : RingHom R S
      z : S
      hz : Eq (Polynomial.eval₂ f z p) 0
      inj : ∀ (x : R), Eq (f x) 0 → Eq x 0
      hlt : GE.ge 0 p.natDegree
      ⊢ False
    -/
    have A : p = C (p.coeff 0) := eq_C_of_natDegree_le_zero hlt
    /-
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      inst✝ : Semiring S
      p : Polynomial R
      hp : Ne p 0
      f : RingHom R S
      z : S
      hz : Eq (Polynomial.eval₂ f z p) 0
      inj : ∀ (x : R), Eq (f x) 0 → Eq x 0
      hlt : GE.ge 0 p.natDegree
      A : Eq p (Polynomial.C (p.coeff 0))
      ⊢ False
    -/
    rw [A, eval₂_C] at hz
    /-
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      inst✝ : Semiring S
      p : Polynomial R
      hp : Ne p 0
      f : RingHom R S
      z : S
      hz : Eq (f (p.coeff 0)) 0
      inj : ∀ (x : R), Eq (f x) 0 → Eq x 0
      hlt : GE.ge 0 p.natDegree
      A : Eq p (Polynomial.C (p.coeff 0))
      ⊢ False
    -/
    simp only [inj (p.coeff 0) hz, RingHom.map_zero] at A
    /-
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      inst✝ : Semiring S
      p : Polynomial R
      hp : Ne p 0
      f : RingHom R S
      z : S
      hz : Eq (f (p.coeff 0)) 0
      inj : ∀ (x : R), Eq (f x) 0 → Eq x 0
      hlt : GE.ge 0 p.natDegree
      A : Eq p 0
      ⊢ False
    -/
    exact hp A
    /-
      🎉 no goals
    -/


theorem degree_pos_of_eval₂_root {p : R[X]} (hp : p ≠ 0) (f : R →+* S) {z : S}
    (hz : eval₂ f z p = 0) (inj : ∀ x : R, f x = 0 → x = 0) : 0 < degree p :=
  natDegree_pos_iff_degree_pos.mp (natDegree_pos_of_eval₂_root hp f hz inj)


@[simp]
theorem coe_lt_degree {p : R[X]} {n : ℕ} : (n : WithBot ℕ) < degree p ↔ n < natDegree p := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    ⊢ Iff (LT.lt (↑n) p.degree) (LT.lt n p.natDegree)
  -/
  by_cases h : p = 0
    /-
      case pos
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      n : Nat
      h : Eq p 0
      ⊢ Iff (LT.lt (↑n) p.degree) (LT.lt n p.natDegree)
    -/
  · simp [h]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    h : Not (Eq p 0)
    ⊢ Iff (LT.lt (↑n) p.degree) (LT.lt n p.natDegree)
  -/
  simp [degree_eq_natDegree h, Nat.cast_lt]
  /-
    🎉 no goals
  -/


@[simp]
theorem degree_map_eq_iff {f : R →+* S} {p : Polynomial R} :
    degree (map f p) = degree p ↔ f (leadingCoeff p) ≠ 0 ∨ p = 0 := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial R
    ⊢ Iff (Eq (Polynomial.map f p).degree p.degree) (Or (Ne (f p.leadingCoeff) 0)  …
  -/
  rcases eq_or_ne p 0 with h|h
    /-
      case inl
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      inst✝ : Semiring S
      f : RingHom R S
      p : Polynomial R
      h : Eq p 0
      ⊢ Iff (Eq (Polynomial.map f p).degree p.degree) (Or (Ne (f p.leadingCoeff) 0)  …
    -/
  · simp [h]
    /-
      🎉 no goals
    -/
  /-
    case inr
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial R
    h : Ne p 0
    ⊢ Iff (Eq (Polynomial.map f p).degree p.degree) (Or (Ne (f p.leadingCoeff) 0)  …
  -/
  simp only [h, or_false]
  /-
    case inr
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial R
    h : Ne p 0
    ⊢ Iff (Eq (Polynomial.map f p).degree p.degree) (Ne (f p.leadingCoeff) 0)
  -/
  refine ⟨fun h2 ↦ ?_, degree_map_eq_of_leadingCoeff_ne_zero f⟩
  /-
    case inr
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial R
    h : Ne p 0
    h2 : Eq (Polynomial.map f p).degree p.degree
    ⊢ Ne (f p.leadingCoeff) 0
  -/
  have h3 : natDegree (map f p) = natDegree p := by simp_rw [natDegree, h2]
  have h4 : map f p ≠ 0 := by
    rwa [ne_eq, ← degree_eq_bot, h2, degree_eq_bot]
  /-
    case inr
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial R
    h : Ne p 0
    h2 : Eq (Polynomial.map f p).degree p.degree
    h3 : Eq (Polynomial.map f p).natDegree p.natDegree
    h4 : Ne (Polynomial.map f p) 0
    ⊢ Ne (f p.leadingCoeff) 0
  -/
  rwa [← coeff_natDegree, ← coeff_map, ← h3, coeff_natDegree, ne_eq, leadingCoeff_eq_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem natDegree_map_eq_iff {f : R →+* S} {p : Polynomial R} :
    natDegree (map f p) = natDegree p ↔ f (p.leadingCoeff) ≠ 0 ∨ natDegree p = 0 := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial R
    ⊢ Iff (Eq (Polynomial.map f p).natDegree p.natDegree) (Or (Ne (f p.leadingCoef …
  -/
  rcases eq_or_ne (natDegree p) 0 with h|h
    /-
      case inl
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      inst✝ : Semiring S
      f : RingHom R S
      p : Polynomial R
      h : Eq p.natDegree 0
      ⊢ Iff (Eq (Polynomial.map f p).natDegree p.natDegree) (Or (Ne (f p.leadingCoef …
    -/
  · simp_rw [h, ne_eq, or_true, iff_true, ← Nat.le_zero, ← h, natDegree_map_le]
    /-
      🎉 no goals
    -/
  /-
    case inr
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial R
    h : Ne p.natDegree 0
    ⊢ Iff (Eq (Polynomial.map f p).natDegree p.natDegree) (Or (Ne (f p.leadingCoef …
  -/
  have h2 : p ≠ 0 := by rintro rfl; simp at h
  /-
    case inr
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial R
    h : Ne p.natDegree 0
    h2 : Ne p 0
    ⊢ Iff (Eq (Polynomial.map f p).natDegree p.natDegree) (Or (Ne (f p.leadingCoef …
  -/
  have h3 : degree p ≠ (0 : ℕ)  := degree_ne_of_natDegree_ne h
  /-
    case inr
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial R
    h : Ne p.natDegree 0
    h2 : Ne p 0
    h3 : Ne p.degree ↑0
    ⊢ Iff (Eq (Polynomial.map f p).natDegree p.natDegree) (Or (Ne (f p.leadingCoef …
  -/
  simp_rw [h, or_false, natDegree, WithBot.unbot'_eq_unbot'_iff, degree_map_eq_iff]
  /-
    case inr
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial R
    h : Ne p.natDegree 0
    h2 : Ne p 0
    h3 : Ne p.degree ↑0
    ⊢ Iff (Or (Or (Ne (f p.leadingCoeff) 0) (Eq p 0)) (Or (And (Eq (Polynomial.map …
  -/
  simp [h, h2, h3] -- simp doesn't rewrite in the hypothesis for some reason
  /-
    case inr
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial R
    h : Ne p.natDegree 0
    h2 : Ne p 0
    h3 : Ne p.degree ↑0
    ⊢ Eq (Polynomial.map f p) 0 → Eq p.degree 0 → Not (Eq (f p.leadingCoeff) 0)
  -/
  tauto
  /-
    🎉 no goals
  -/


theorem natDegree_pos_of_nextCoeff_ne_zero (h : p.nextCoeff ≠ 0) : 0 < p.natDegree := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    h : Ne p.nextCoeff 0
    ⊢ LT.lt 0 p.natDegree
  -/
  rw [nextCoeff] at h
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    h : Ne (ite (Eq p.natDegree 0) 0 (p.coeff (HSub.hSub p.natDegree 1))) 0
    ⊢ LT.lt 0 p.natDegree
  -/
  by_cases hpz : p.natDegree = 0
    /-
      case pos
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      h : Ne (ite (Eq p.natDegree 0) 0 (p.coeff (HSub.hSub p.natDegree 1))) 0
      hpz : Eq p.natDegree 0
      ⊢ LT.lt 0 p.natDegree
    -/
  · simp_all only [ne_eq, zero_le, ite_true, not_true_eq_false]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      h : Ne (ite (Eq p.natDegree 0) 0 (p.coeff (HSub.hSub p.natDegree 1))) 0
      hpz : Not (Eq p.natDegree 0)
      ⊢ LT.lt 0 p.natDegree
    -/
  · apply Nat.zero_lt_of_ne_zero hpz
    /-
      🎉 no goals
    -/


                                                                    /-
                                                                      R : Type u
                                                                      inst✝ : Ring R
                                                                      p q : Polynomial R
                                                                      ⊢ Eq (HSub.hSub p q).natDegree (HSub.hSub q p).natDegree
                                                                    -/
theorem natDegree_sub : (p - q).natDegree = (q - p).natDegree := by rw [← natDegree_neg, neg_sub]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem natDegree_sub_le_iff_left (qn : q.natDegree ≤ n) :
    (p - q).natDegree ≤ n ↔ p.natDegree ≤ n := by
  /-
    R : Type u
    n : Nat
    inst✝ : Ring R
    p q : Polynomial R
    qn : LE.le q.natDegree n
    ⊢ Iff (LE.le (HSub.hSub p q).natDegree n) (LE.le p.natDegree n)
  -/
  rw [← natDegree_neg] at qn
  /-
    R : Type u
    n : Nat
    inst✝ : Ring R
    p q : Polynomial R
    qn : LE.le (Neg.neg q).natDegree n
    ⊢ Iff (LE.le (HSub.hSub p q).natDegree n) (LE.le p.natDegree n)
  -/
  rw [sub_eq_add_neg, natDegree_add_le_iff_left _ _ qn]
  /-
    🎉 no goals
  -/


theorem natDegree_sub_le_iff_right (pn : p.natDegree ≤ n) :
                                                  /-
                                                    R : Type u
                                                    n : Nat
                                                    inst✝ : Ring R
                                                    p q : Polynomial R
                                                    pn : LE.le p.natDegree n
                                                    ⊢ Iff (LE.le (HSub.hSub p q).natDegree n) (LE.le q.natDegree n)
                                                  -/
    (p - q).natDegree ≤ n ↔ q.natDegree ≤ n := by rwa [natDegree_sub, natDegree_sub_le_iff_left]
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem coeff_sub_eq_left_of_lt (dg : q.natDegree < n) : (p - q).coeff n = p.coeff n := by
  /-
    R : Type u
    n : Nat
    inst✝ : Ring R
    p q : Polynomial R
    dg : LT.lt q.natDegree n
    ⊢ Eq ((HSub.hSub p q).coeff n) (p.coeff n)
  -/
  rw [← natDegree_neg] at dg
  /-
    R : Type u
    n : Nat
    inst✝ : Ring R
    p q : Polynomial R
    dg : LT.lt (Neg.neg q).natDegree n
    ⊢ Eq ((HSub.hSub p q).coeff n) (p.coeff n)
  -/
  rw [sub_eq_add_neg, coeff_add_eq_left_of_lt dg]
  /-
    🎉 no goals
  -/


theorem coeff_sub_eq_neg_right_of_lt (df : p.natDegree < n) : (p - q).coeff n = -q.coeff n := by
  /-
    R : Type u
    n : Nat
    inst✝ : Ring R
    p q : Polynomial R
    df : LT.lt p.natDegree n
    ⊢ Eq ((HSub.hSub p q).coeff n) (Neg.neg (q.coeff n))
  -/
  rwa [sub_eq_add_neg, coeff_add_eq_right_of_lt, coeff_neg]
  /-
    🎉 no goals
  -/


@[simp]
lemma nextCoeff_C_mul_X_add_C (ha : a ≠ 0) (c : R) : nextCoeff (C a * X + C c) = c := by
  /-
    R : Type u
    inst✝ : Semiring R
    a : R
    ha : Ne a 0
    c : R
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Polynomial.C a) Polynomial.X) (Polynomial.C c)).ne …
  -/
                                      /-
                                        🎉 no goals
                                      -/
  rw [nextCoeff_of_natDegree_pos] <;> simp [ha]
                                      /-
                                        🎉 no goals
                                      -/


lemma natDegree_eq_one : p.natDegree = 1 ↔ ∃ a ≠ 0, ∃ b, C a * X + C b = p := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    ⊢ Iff (Eq p.natDegree 1) (Exists fun a => And (Ne a 0) (Exists fun b => Eq (HA …
  -/
  refine ⟨fun hp ↦ ⟨p.coeff 1, fun h ↦ ?_, p.coeff 0, ?_⟩, ?_⟩
    /-
      case refine_1
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      hp : Eq p.natDegree 1
      h : Eq (p.coeff 1) 0
      ⊢ False
    -/
  · rw [← hp, coeff_natDegree, leadingCoeff_eq_zero] at h
    /-
      case refine_1
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      hp : Eq p.natDegree 1
      h : Eq p 0
      ⊢ False
    -/
    aesop
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      hp : Eq p.natDegree 1
      ⊢ Eq (HAdd.hAdd (HMul.hMul (Polynomial.C (p.coeff 1)) Polynomial.X) (Polynomia …
    -/
  · ext n
    /-
      case refine_2.a
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      hp : Eq p.natDegree 1
      n : Nat
      ⊢ Eq ((HAdd.hAdd (HMul.hMul (Polynomial.C (p.coeff 1)) Polynomial.X) (Polynomi …
    -/
    obtain _ | _ | n := n
      /-
        case refine_2.a.zero
        R : Type u
        inst✝ : Semiring R
        p : Polynomial R
        hp : Eq p.natDegree 1
        ⊢ Eq ((HAdd.hAdd (HMul.hMul (Polynomial.C (p.coeff 1)) Polynomial.X) (Polynomi …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case refine_2.a.succ.zero
        R : Type u
        inst✝ : Semiring R
        p : Polynomial R
        hp : Eq p.natDegree 1
        ⊢ Eq ((HAdd.hAdd (HMul.hMul (Polynomial.C (p.coeff 1)) Polynomial.X) (Polynomi …
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case refine_2.a.succ.succ
        R : Type u
        inst✝ : Semiring R
        p : Polynomial R
        hp : Eq p.natDegree 1
        n : Nat
        ⊢ Eq ((HAdd.hAdd (HMul.hMul (Polynomial.C (p.coeff 1)) Polynomial.X) (Polynomi …
      -/
    · simp only [coeff_add, coeff_mul_X, coeff_C_succ, add_zero]
      /-
        case refine_2.a.succ.succ
        R : Type u
        inst✝ : Semiring R
        p : Polynomial R
        hp : Eq p.natDegree 1
        n : Nat
        ⊢ Eq 0 (p.coeff (HAdd.hAdd (HAdd.hAdd n 1) 1))
      -/
      rw [coeff_eq_zero_of_natDegree_lt]
      /-
        case refine_2.a.succ.succ
        R : Type u
        inst✝ : Semiring R
        p : Polynomial R
        hp : Eq p.natDegree 1
        n : Nat
        ⊢ LT.lt p.natDegree (HAdd.hAdd (HAdd.hAdd n 1) 1)
      -/
      simp [hp]
      /-
        🎉 no goals
      -/
    /-
      case refine_3
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      ⊢ (Exists fun a => And (Ne a 0) (Exists fun b => Eq (HAdd.hAdd (HMul.hMul (Pol …
    -/
  · rintro ⟨a, ha, b, rfl⟩
    /-
      case refine_3.intro.intro.intro
      R : Type u
      inst✝ : Semiring R
      a : R
      ha : Ne a 0
      b : R
      ⊢ Eq (HAdd.hAdd (HMul.hMul (Polynomial.C a) Polynomial.X) (Polynomial.C b)).na …
    -/
    simp [ha]
    /-
      🎉 no goals
    -/


theorem degree_mul_C (a0 : a ≠ 0) : (p * C a).degree = p.degree := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    p : Polynomial R
    a : R
    inst✝ : NoZeroDivisors R
    a0 : Ne a 0
    ⊢ Eq (HMul.hMul p (Polynomial.C a)).degree p.degree
  -/
  rw [degree_mul, degree_C a0, add_zero]
  /-
    🎉 no goals
  -/


theorem degree_C_mul (a0 : a ≠ 0) : (C a * p).degree = p.degree := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    p : Polynomial R
    a : R
    inst✝ : NoZeroDivisors R
    a0 : Ne a 0
    ⊢ Eq (HMul.hMul (Polynomial.C a) p).degree p.degree
  -/
  rw [degree_mul, degree_C a0, zero_add]
  /-
    🎉 no goals
  -/


theorem natDegree_mul_C (a0 : a ≠ 0) : (p * C a).natDegree = p.natDegree := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    p : Polynomial R
    a : R
    inst✝ : NoZeroDivisors R
    a0 : Ne a 0
    ⊢ Eq (HMul.hMul p (Polynomial.C a)).natDegree p.natDegree
  -/
  simp only [natDegree, degree_mul_C a0]
  /-
    🎉 no goals
  -/


theorem natDegree_C_mul (a0 : a ≠ 0) : (C a * p).natDegree = p.natDegree := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    p : Polynomial R
    a : R
    inst✝ : NoZeroDivisors R
    a0 : Ne a 0
    ⊢ Eq (HMul.hMul (Polynomial.C a) p).natDegree p.natDegree
  -/
  simp only [natDegree, degree_C_mul a0]
  /-
    🎉 no goals
  -/


theorem natDegree_comp : natDegree (p.comp q) = natDegree p * natDegree q := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    p q : Polynomial R
    inst✝ : NoZeroDivisors R
    ⊢ Eq (p.comp q).natDegree (HMul.hMul p.natDegree q.natDegree)
  -/
  by_cases q0 : q.natDegree = 0
  · rw [degree_le_zero_iff.mp (natDegree_eq_zero_iff_degree_le_zero.mp q0), comp_C, natDegree_C,
      natDegree_C, mul_zero]
    /-
      case neg
      R : Type u
      inst✝¹ : Semiring R
      p q : Polynomial R
      inst✝ : NoZeroDivisors R
      q0 : Not (Eq q.natDegree 0)
      ⊢ Eq (p.comp q).natDegree (HMul.hMul p.natDegree q.natDegree)
    -/
  · by_cases p0 : p = 0
      /-
        case pos
        R : Type u
        inst✝¹ : Semiring R
        p q : Polynomial R
        inst✝ : NoZeroDivisors R
        q0 : Not (Eq q.natDegree 0)
        p0 : Eq p 0
        ⊢ Eq (p.comp q).natDegree (HMul.hMul p.natDegree q.natDegree)
      -/
    · simp only [p0, zero_comp, natDegree_zero, zero_mul]
      /-
        🎉 no goals
      -/
    · simp only [Ne, mul_eq_zero, leadingCoeff_eq_zero, p0, natDegree_comp_eq_of_mul_ne_zero,
        ne_zero_of_natDegree_gt (Nat.pos_of_ne_zero q0), not_false_eq_true, pow_ne_zero, or_self]


@[simp]
theorem natDegree_iterate_comp (k : ℕ) :
    (p.comp^[k] q).natDegree = p.natDegree ^ k * q.natDegree := by
  induction k with
  | zero => simp
  | succ k IH => rw [Function.iterate_succ_apply', natDegree_comp, IH, pow_succ', mul_assoc]


theorem leadingCoeff_comp (hq : natDegree q ≠ 0) :
    leadingCoeff (p.comp q) = leadingCoeff p * leadingCoeff q ^ natDegree p := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    p q : Polynomial R
    inst✝ : NoZeroDivisors R
    hq : Ne q.natDegree 0
    ⊢ Eq (p.comp q).leadingCoeff (HMul.hMul p.leadingCoeff (HPow.hPow q.leadingCoe …
  -/
  rw [← coeff_comp_degree_mul_degree hq, ← natDegree_comp, coeff_natDegree]
  /-
    🎉 no goals
  -/


@[simp] lemma comp_neg_X_leadingCoeff_eq [Ring R] (p : R[X]) :
    (p.comp (-X)).leadingCoeff = (-1) ^ p.natDegree * p.leadingCoeff := by
  /-
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    ⊢ Eq (p.comp (Neg.neg Polynomial.X)).leadingCoeff (HMul.hMul (HPow.hPow (-1) p …
  -/
  nontriviality R
  /-
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    a✝ : Nontrivial R
    ⊢ Eq (p.comp (Neg.neg Polynomial.X)).leadingCoeff (HMul.hMul (HPow.hPow (-1) p …
  -/
  by_cases h : p = 0
    /-
      case pos
      R : Type u
      inst✝ : Ring R
      p : Polynomial R
      a✝ : Nontrivial R
      h : Eq p 0
      ⊢ Eq (p.comp (Neg.neg Polynomial.X)).leadingCoeff (HMul.hMul (HPow.hPow (-1) p …
    -/
  · simp [h]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    a✝ : Nontrivial R
    h : Not (Eq p 0)
    ⊢ Eq (p.comp (Neg.neg Polynomial.X)).leadingCoeff (HMul.hMul (HPow.hPow (-1) p …
  -/
  rw [Polynomial.leadingCoeff, natDegree_comp_eq_of_mul_ne_zero, coeff_comp_degree_mul_degree] <;>
  /-
    case neg
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    a✝ : Nontrivial R
    h : Not (Eq p 0)
    ⊢ Eq (HMul.hMul p.leadingCoeff (HPow.hPow (Neg.neg Polynomial.X).leadingCoeff  …
  -/
  /-
    🎉 no goals
  -/
  /-
    🎉 no goals
  -/
  simp [((Commute.neg_one_left _).pow_left _).eq, h]
  /-
    🎉 no goals
  -/


lemma comp_eq_zero_iff [Semiring R] [NoZeroDivisors R] {p q : R[X]} :
    p.comp q = 0 ↔ p = 0 ∨ p.eval (q.coeff 0) = 0 ∧ q = C (q.coeff 0) := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : NoZeroDivisors R
    p q : Polynomial R
    ⊢ Iff (Eq (p.comp q) 0) (Or (Eq p 0) (And (Eq (Polynomial.eval (q.coeff 0) p)  …
  -/
  refine ⟨fun h ↦ ?_, Or.rec (fun h ↦ by simp [h]) fun h ↦ by rw [h.2, comp_C, h.1, C_0]⟩
  have key : p.natDegree = 0 ∨ q.natDegree = 0 := by
    rw [← mul_eq_zero, ← natDegree_comp, h, natDegree_zero]
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : NoZeroDivisors R
    p q : Polynomial R
    h : Eq (p.comp q) 0
    key : Or (Eq p.natDegree 0) (Eq q.natDegree 0)
    ⊢ Or (Eq p 0) (And (Eq (Polynomial.eval (q.coeff 0) p) 0) (Eq q (Polynomial.C  …
  -/
  obtain key | key := Or.imp eq_C_of_natDegree_eq_zero eq_C_of_natDegree_eq_zero key
    /-
      case inl
      R : Type u
      inst✝¹ : Semiring R
      inst✝ : NoZeroDivisors R
      p q : Polynomial R
      h : Eq (p.comp q) 0
      key✝ : Or (Eq p.natDegree 0) (Eq q.natDegree 0)
      key : Eq p (Polynomial.C (p.coeff 0))
      ⊢ Or (Eq p 0) (And (Eq (Polynomial.eval (q.coeff 0) p) 0) (Eq q (Polynomial.C  …
    -/
  · rw [key, C_comp] at h
    /-
      case inl
      R : Type u
      inst✝¹ : Semiring R
      inst✝ : NoZeroDivisors R
      p q : Polynomial R
      h : Eq (Polynomial.C (p.coeff 0)) 0
      key✝ : Or (Eq p.natDegree 0) (Eq q.natDegree 0)
      key : Eq p (Polynomial.C (p.coeff 0))
      ⊢ Or (Eq p 0) (And (Eq (Polynomial.eval (q.coeff 0) p) 0) (Eq q (Polynomial.C  …
    -/
    exact Or.inl (key.trans h)
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u
      inst✝¹ : Semiring R
      inst✝ : NoZeroDivisors R
      p q : Polynomial R
      h : Eq (p.comp q) 0
      key✝ : Or (Eq p.natDegree 0) (Eq q.natDegree 0)
      key : Eq q (Polynomial.C (q.coeff 0))
      ⊢ Or (Eq p 0) (And (Eq (Polynomial.eval (q.coeff 0) p) 0) (Eq q (Polynomial.C  …
    -/
  · rw [key, comp_C, C_eq_zero] at h
    /-
      case inr
      R : Type u
      inst✝¹ : Semiring R
      inst✝ : NoZeroDivisors R
      p q : Polynomial R
      h : Eq (Polynomial.eval (q.coeff 0) p) 0
      key✝ : Or (Eq p.natDegree 0) (Eq q.natDegree 0)
      key : Eq q (Polynomial.C (q.coeff 0))
      ⊢ Or (Eq p 0) (And (Eq (Polynomial.eval (q.coeff 0) p) 0) (Eq q (Polynomial.C  …
    -/
    exact Or.inr ⟨h, key⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem irreducible_mul_leadingCoeff_inv {p : K[X]} :
    Irreducible (p * C (leadingCoeff p)⁻¹) ↔ Irreducible p := by
  /-
    K : Type u_1
    inst✝ : DivisionRing K
    p : Polynomial K
    ⊢ Iff (Irreducible (HMul.hMul p (Polynomial.C (Inv.inv p.leadingCoeff)))) (Irr …
  -/
  by_cases hp0 : p = 0
    /-
      case pos
      K : Type u_1
      inst✝ : DivisionRing K
      p : Polynomial K
      hp0 : Eq p 0
      ⊢ Iff (Irreducible (HMul.hMul p (Polynomial.C (Inv.inv p.leadingCoeff)))) (Irr …
    -/
  · simp [hp0]
    /-
      🎉 no goals
    -/
  exact irreducible_mul_isUnit
    (isUnit_C.mpr (IsUnit.mk0 _ (inv_ne_zero (leadingCoeff_ne_zero.mpr hp0))))


@[simp] lemma dvd_mul_leadingCoeff_inv {p q : K[X]} (hp0 : p ≠ 0) :
    q ∣ p * C (leadingCoeff p)⁻¹ ↔ q ∣ p :=
  IsUnit.dvd_mul_right <| isUnit_C.mpr <| IsUnit.mk0 _ <|
    inv_ne_zero <| leadingCoeff_ne_zero.mpr hp0


theorem monic_mul_leadingCoeff_inv {p : K[X]} (h : p ≠ 0) : Monic (p * C (leadingCoeff p)⁻¹) := by
  rw [Monic, leadingCoeff_mul, leadingCoeff_C,
    mul_inv_cancel₀ (show leadingCoeff p ≠ 0 from mt leadingCoeff_eq_zero.1 h)]

-- `simp` normal form of `degree_mul_leadingCoeff_inv`

@[simp] lemma degree_leadingCoeff_inv {p : K[X]} (hp0 : p ≠ 0) :
    degree (C (leadingCoeff p)⁻¹) = 0 :=
  degree_C (inv_ne_zero <| leadingCoeff_ne_zero.mpr hp0)


theorem degree_mul_leadingCoeff_inv (p : K[X]) {q : K[X]} (h : q ≠ 0) :
    degree (p * C (leadingCoeff q)⁻¹) = degree p := by
  /-
    K : Type u_1
    inst✝ : DivisionRing K
    p q : Polynomial K
    h : Ne q 0
    ⊢ Eq (HMul.hMul p (Polynomial.C (Inv.inv q.leadingCoeff))).degree p.degree
  -/
  have h₁ : (leadingCoeff q)⁻¹ ≠ 0 := inv_ne_zero (mt leadingCoeff_eq_zero.1 h)
  /-
    K : Type u_1
    inst✝ : DivisionRing K
    p q : Polynomial K
    h : Ne q 0
    h₁ : Ne (Inv.inv q.leadingCoeff) 0
    ⊢ Eq (HMul.hMul p (Polynomial.C (Inv.inv q.leadingCoeff))).degree p.degree
  -/
  rw [degree_mul_C h₁]
  /-
    🎉 no goals
  -/


theorem natDegree_mul_leadingCoeff_inv (p : K[X]) {q : K[X]} (h : q ≠ 0) :
    natDegree (p * C (leadingCoeff q)⁻¹) = natDegree p :=
  natDegree_eq_of_degree_eq (degree_mul_leadingCoeff_inv _ h)


theorem degree_mul_leadingCoeff_self_inv (p : K[X]) :
    degree (p * C (leadingCoeff p)⁻¹) = degree p := by
  /-
    K : Type u_1
    inst✝ : DivisionRing K
    p : Polynomial K
    ⊢ Eq (HMul.hMul p (Polynomial.C (Inv.inv p.leadingCoeff))).degree p.degree
  -/
  by_cases hp : p = 0
    /-
      case pos
      K : Type u_1
      inst✝ : DivisionRing K
      p : Polynomial K
      hp : Eq p 0
      ⊢ Eq (HMul.hMul p (Polynomial.C (Inv.inv p.leadingCoeff))).degree p.degree
    -/
  · simp [hp]
    /-
      🎉 no goals
    -/
  /-
    case neg
    K : Type u_1
    inst✝ : DivisionRing K
    p : Polynomial K
    hp : Not (Eq p 0)
    ⊢ Eq (HMul.hMul p (Polynomial.C (Inv.inv p.leadingCoeff))).degree p.degree
  -/
  exact degree_mul_leadingCoeff_inv _ hp
  /-
    🎉 no goals
  -/


theorem natDegree_mul_leadingCoeff_self_inv (p : K[X]) :
    natDegree (p * C (leadingCoeff p)⁻¹) = natDegree p :=
  natDegree_eq_of_degree_eq (degree_mul_leadingCoeff_self_inv _)

-- `simp` normal form of `degree_mul_leadingCoeff_self_inv`

@[simp] lemma degree_add_degree_leadingCoeff_inv (p : K[X]) :
    degree p + degree (C (leadingCoeff p)⁻¹) = degree p := by
  /-
    K : Type u_1
    inst✝ : DivisionRing K
    p : Polynomial K
    ⊢ Eq (HAdd.hAdd p.degree (Polynomial.C (Inv.inv p.leadingCoeff)).degree) p.deg …
  -/
  rw [← degree_mul, degree_mul_leadingCoeff_self_inv]
  /-
    🎉 no goals
  -/


