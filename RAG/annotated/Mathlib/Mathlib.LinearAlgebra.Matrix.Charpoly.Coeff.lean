theorem charmatrix_apply_natDegree [Nontrivial R] (i j : n) :
    (charmatrix M i j).natDegree = ite (i = j) 1 0 := by
  /-
    R : Type u
    inst✝³ : CommRing R
    n : Type v
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    M : Matrix n n R
    inst✝ : Nontrivial R
    i j : n
    ⊢ Eq (M.charmatrix i j).natDegree (ite (Eq i j) 1 0)
  -/
                         /-
                           🎉 no goals
                         -/
  by_cases h : i = j <;> simp [h, ← degree_eq_iff_natDegree_eq_of_pos (Nat.succ_pos 0)]
                         /-
                           🎉 no goals
                         -/


theorem charmatrix_apply_natDegree_le (i j : n) :
    (charmatrix M i j).natDegree ≤ ite (i = j) 1 0 := by
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    i j : n
    ⊢ LE.le (M.charmatrix i j).natDegree (ite (Eq i j) 1 0)
  -/
                       /-
                         🎉 no goals
                       -/
  split_ifs with h <;> simp [h, natDegree_X_le]
                       /-
                         🎉 no goals
                       -/


theorem charpoly_sub_diagonal_degree_lt :
    (M.charpoly - ∏ i : n, (X - C (M i i))).degree < ↑(Fintype.card n - 1) := by
  rw [charpoly, det_apply', ← insert_erase (mem_univ (Equiv.refl n)),
    sum_insert (not_mem_erase (Equiv.refl n) univ), add_comm]
  simp only [charmatrix_apply_eq, one_mul, Equiv.Perm.sign_refl, id, Int.cast_one,
    Units.val_one, add_sub_cancel_right, Equiv.coe_refl]
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    ⊢ LT.lt ((Finset.univ.erase (Equiv.refl n)).sum fun x => HMul.hMul (↑↑(Equiv.P …
  -/
  rw [← mem_degreeLT]
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    ⊢ Membership.mem (Polynomial.degreeLT R (HSub.hSub (Fintype.card n) 1)) ((Fins …
  -/
  apply Submodule.sum_mem (degreeLT R (Fintype.card n - 1))
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    ⊢ ∀ (c : Equiv n n), Membership.mem (Finset.univ.erase (Equiv.refl n)) c → Mem …
  -/
  intro c hc; rw [← C_eq_intCast, C_mul']
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    c : Equiv n n
    hc : Membership.mem (Finset.univ.erase (Equiv.refl n)) c
    ⊢ Membership.mem (Polynomial.degreeLT R (HSub.hSub (Fintype.card n) 1)) (HSMul …
  -/
  apply Submodule.smul_mem (degreeLT R (Fintype.card n - 1)) ↑↑(Equiv.Perm.sign c)
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    c : Equiv n n
    hc : Membership.mem (Finset.univ.erase (Equiv.refl n)) c
    ⊢ Membership.mem (Polynomial.degreeLT R (HSub.hSub (Fintype.card n) 1)) (Finse …
  -/
  rw [mem_degreeLT]
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    c : Equiv n n
    hc : Membership.mem (Finset.univ.erase (Equiv.refl n)) c
    ⊢ LT.lt (Finset.univ.prod fun i => M.charmatrix (c i) i).degree ↑(HSub.hSub (F …
  -/
  apply lt_of_le_of_lt degree_le_natDegree _
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    c : Equiv n n
    hc : Membership.mem (Finset.univ.erase (Equiv.refl n)) c
    ⊢ LT.lt ↑(Finset.univ.prod fun i => M.charmatrix (c i) i).natDegree ↑(HSub.hSu …
  -/
  rw [Nat.cast_lt]
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    c : Equiv n n
    hc : Membership.mem (Finset.univ.erase (Equiv.refl n)) c
    ⊢ LT.lt (Finset.univ.prod fun i => M.charmatrix (c i) i).natDegree (HSub.hSub  …
  -/
  apply lt_of_le_of_lt _ (Equiv.Perm.fixed_point_card_lt_of_ne_one (ne_of_mem_erase hc))
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    c : Equiv n n
    hc : Membership.mem (Finset.univ.erase (Equiv.refl n)) c
    ⊢ LE.le (Finset.univ.prod fun i => M.charmatrix (c i) i).natDegree (Finset.fil …
  -/
  apply le_trans (Polynomial.natDegree_prod_le univ fun i : n => charmatrix M (c i) i) _
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    c : Equiv n n
    hc : Membership.mem (Finset.univ.erase (Equiv.refl n)) c
    ⊢ LE.le (Finset.univ.sum fun i => (M.charmatrix (c i) i).natDegree) (Finset.fi …
  -/
  rw [card_eq_sum_ones]; rw [sum_filter]; apply sum_le_sum
  /-
    case h
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    c : Equiv n n
    hc : Membership.mem (Finset.univ.erase (Equiv.refl n)) c
    ⊢ ∀ (i : n), Membership.mem Finset.univ i → LE.le (M.charmatrix (c i) i).natDe …
  -/
  intros
  /-
    case h
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    c : Equiv n n
    hc : Membership.mem (Finset.univ.erase (Equiv.refl n)) c
    i✝ : n
    a✝ : Membership.mem Finset.univ i✝
    ⊢ LE.le (M.charmatrix (c i✝) i✝).natDegree (ite (Eq (c i✝) i✝) 1 0)
  -/
  apply charmatrix_apply_natDegree_le
  /-
    🎉 no goals
  -/


theorem charpoly_coeff_eq_prod_coeff_of_le {k : ℕ} (h : Fintype.card n - 1 ≤ k) :
    M.charpoly.coeff k = (∏ i : n, (X - C (M i i))).coeff k := by
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    k : Nat
    h : LE.le (HSub.hSub (Fintype.card n) 1) k
    ⊢ Eq (M.charpoly.coeff k) ((Finset.univ.prod fun i => HSub.hSub Polynomial.X ( …
  -/
  apply eq_of_sub_eq_zero; rw [← coeff_sub]
  /-
    case h
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    k : Nat
    h : LE.le (HSub.hSub (Fintype.card n) 1) k
    ⊢ Eq ((HSub.hSub M.charpoly (Finset.univ.prod fun i => HSub.hSub Polynomial.X  …
  -/
  apply Polynomial.coeff_eq_zero_of_degree_lt
  /-
    case h.h
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    k : Nat
    h : LE.le (HSub.hSub (Fintype.card n) 1) k
    ⊢ LT.lt (HSub.hSub M.charpoly (Finset.univ.prod fun i => HSub.hSub Polynomial. …
  -/
  apply lt_of_lt_of_le (charpoly_sub_diagonal_degree_lt M) ?_
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    k : Nat
    h : LE.le (HSub.hSub (Fintype.card n) 1) k
    ⊢ LE.le ↑(HSub.hSub (Fintype.card n) 1) ↑k
  -/
  rw [Nat.cast_le]; apply h
                    /-
                      🎉 no goals
                    -/


theorem det_of_card_zero (h : Fintype.card n = 0) (M : Matrix n n R) : M.det = 1 := by
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    h : Eq (Fintype.card n) 0
    M : Matrix n n R
    ⊢ Eq M.det 1
  -/
  rw [Fintype.card_eq_zero_iff] at h
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    h : IsEmpty n
    M : Matrix n n R
    ⊢ Eq M.det 1
  -/
  suffices M = 1 by simp [this]
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    h : IsEmpty n
    M : Matrix n n R
    ⊢ Eq M 1
  -/
  ext i
  /-
    case a
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    h : IsEmpty n
    M : Matrix n n R
    i j✝ : n
    ⊢ Eq (M i j✝) (1 i j✝)
  -/
  exact h.elim i
  /-
    🎉 no goals
  -/


theorem charpoly_degree_eq_dim [Nontrivial R] (M : Matrix n n R) :
    M.charpoly.degree = Fintype.card n := by
  /-
    R : Type u
    inst✝³ : CommRing R
    n : Type v
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : Nontrivial R
    M : Matrix n n R
    ⊢ Eq M.charpoly.degree ↑(Fintype.card n)
  -/
  by_cases h : Fintype.card n = 0
    /-
      case pos
      R : Type u
      inst✝³ : CommRing R
      n : Type v
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : Nontrivial R
      M : Matrix n n R
      h : Eq (Fintype.card n) 0
      ⊢ Eq M.charpoly.degree ↑(Fintype.card n)
    -/
  · rw [h]
    /-
      case pos
      R : Type u
      inst✝³ : CommRing R
      n : Type v
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : Nontrivial R
      M : Matrix n n R
      h : Eq (Fintype.card n) 0
      ⊢ Eq M.charpoly.degree ↑0
    -/
    unfold charpoly
    /-
      case pos
      R : Type u
      inst✝³ : CommRing R
      n : Type v
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : Nontrivial R
      M : Matrix n n R
      h : Eq (Fintype.card n) 0
      ⊢ Eq M.charmatrix.det.degree ↑0
    -/
    rw [det_of_card_zero]
      /-
        case pos
        R : Type u
        inst✝³ : CommRing R
        n : Type v
        inst✝² : DecidableEq n
        inst✝¹ : Fintype n
        inst✝ : Nontrivial R
        M : Matrix n n R
        h : Eq (Fintype.card n) 0
        ⊢ Eq (Polynomial.degree 1) ↑0
      -/
    · simp
      /-
        🎉 no goals
      -/
      /-
        case pos.h
        R : Type u
        inst✝³ : CommRing R
        n : Type v
        inst✝² : DecidableEq n
        inst✝¹ : Fintype n
        inst✝ : Nontrivial R
        M : Matrix n n R
        h : Eq (Fintype.card n) 0
        ⊢ Eq (Fintype.card n) 0
      -/
    · assumption
      /-
        🎉 no goals
      -/
  /-
    case neg
    R : Type u
    inst✝³ : CommRing R
    n : Type v
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : Nontrivial R
    M : Matrix n n R
    h : Not (Eq (Fintype.card n) 0)
    ⊢ Eq M.charpoly.degree ↑(Fintype.card n)
  -/
  rw [← sub_add_cancel M.charpoly (∏ i : n, (X - C (M i i)))]
  -- Porting note: added `↑` in front of `Fintype.card n`
  have h1 : (∏ i : n, (X - C (M i i))).degree = ↑(Fintype.card n) := by
    rw [degree_eq_iff_natDegree_eq_of_pos (Nat.pos_of_ne_zero h), natDegree_prod']
    · simp_rw [natDegree_X_sub_C]
      rw [← Finset.card_univ, sum_const, smul_eq_mul, mul_one]
    simp_rw [(monic_X_sub_C _).leadingCoeff]
    simp
  /-
    case neg
    R : Type u
    inst✝³ : CommRing R
    n : Type v
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : Nontrivial R
    M : Matrix n n R
    h : Not (Eq (Fintype.card n) 0)
    h1 : Eq (Finset.univ.prod fun i => HSub.hSub Polynomial.X (Polynomial.C (M i i …
    ⊢ Eq (HAdd.hAdd (HSub.hSub M.charpoly (Finset.univ.prod fun i => HSub.hSub Pol …
  -/
  rw [degree_add_eq_right_of_degree_lt]
    /-
      case neg
      R : Type u
      inst✝³ : CommRing R
      n : Type v
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      inst✝ : Nontrivial R
      M : Matrix n n R
      h : Not (Eq (Fintype.card n) 0)
      h1 : Eq (Finset.univ.prod fun i => HSub.hSub Polynomial.X (Polynomial.C (M i i …
      ⊢ Eq (Finset.univ.prod fun i => HSub.hSub Polynomial.X (Polynomial.C (M i i))) …
    -/
  · exact h1
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u
    inst✝³ : CommRing R
    n : Type v
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : Nontrivial R
    M : Matrix n n R
    h : Not (Eq (Fintype.card n) 0)
    h1 : Eq (Finset.univ.prod fun i => HSub.hSub Polynomial.X (Polynomial.C (M i i …
    ⊢ LT.lt (HSub.hSub M.charpoly (Finset.univ.prod fun i => HSub.hSub Polynomial. …
  -/
  rw [h1]
  /-
    case neg
    R : Type u
    inst✝³ : CommRing R
    n : Type v
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : Nontrivial R
    M : Matrix n n R
    h : Not (Eq (Fintype.card n) 0)
    h1 : Eq (Finset.univ.prod fun i => HSub.hSub Polynomial.X (Polynomial.C (M i i …
    ⊢ LT.lt (HSub.hSub M.charpoly (Finset.univ.prod fun i => HSub.hSub Polynomial. …
  -/
  apply lt_trans (charpoly_sub_diagonal_degree_lt M)
  /-
    case neg
    R : Type u
    inst✝³ : CommRing R
    n : Type v
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : Nontrivial R
    M : Matrix n n R
    h : Not (Eq (Fintype.card n) 0)
    h1 : Eq (Finset.univ.prod fun i => HSub.hSub Polynomial.X (Polynomial.C (M i i …
    ⊢ LT.lt ↑(HSub.hSub (Fintype.card n) 1) ↑(Fintype.card n)
  -/
  rw [Nat.cast_lt]
  /-
    case neg
    R : Type u
    inst✝³ : CommRing R
    n : Type v
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : Nontrivial R
    M : Matrix n n R
    h : Not (Eq (Fintype.card n) 0)
    h1 : Eq (Finset.univ.prod fun i => HSub.hSub Polynomial.X (Polynomial.C (M i i …
    ⊢ LT.lt (HSub.hSub (Fintype.card n) 1) (Fintype.card n)
  -/
  rw [← Nat.pred_eq_sub_one]
  /-
    case neg
    R : Type u
    inst✝³ : CommRing R
    n : Type v
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : Nontrivial R
    M : Matrix n n R
    h : Not (Eq (Fintype.card n) 0)
    h1 : Eq (Finset.univ.prod fun i => HSub.hSub Polynomial.X (Polynomial.C (M i i …
    ⊢ LT.lt (Fintype.card n).pred (Fintype.card n)
  -/
  apply Nat.pred_lt
  /-
    case neg.a
    R : Type u
    inst✝³ : CommRing R
    n : Type v
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : Nontrivial R
    M : Matrix n n R
    h : Not (Eq (Fintype.card n) 0)
    h1 : Eq (Finset.univ.prod fun i => HSub.hSub Polynomial.X (Polynomial.C (M i i …
    ⊢ Ne (Fintype.card n) 0
  -/
  apply h
  /-
    🎉 no goals
  -/


@[simp] theorem charpoly_natDegree_eq_dim [Nontrivial R] (M : Matrix n n R) :
    M.charpoly.natDegree = Fintype.card n :=
  natDegree_eq_of_degree_eq_some (charpoly_degree_eq_dim M)


theorem charpoly_monic (M : Matrix n n R) : M.charpoly.Monic := by
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    ⊢ M.charpoly.Monic
  -/
  nontriviality R -- Porting note: was simply `nontriviality`
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    a✝ : Nontrivial R
    ⊢ M.charpoly.Monic
  -/
  by_cases h : Fintype.card n = 0
    /-
      case pos
      R : Type u
      inst✝² : CommRing R
      n : Type v
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      M : Matrix n n R
      a✝ : Nontrivial R
      h : Eq (Fintype.card n) 0
      ⊢ M.charpoly.Monic
    -/
  · rw [charpoly, det_of_card_zero h]
    /-
      case pos
      R : Type u
      inst✝² : CommRing R
      n : Type v
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      M : Matrix n n R
      a✝ : Nontrivial R
      h : Eq (Fintype.card n) 0
      ⊢ Polynomial.Monic 1
    -/
    apply monic_one
    /-
      🎉 no goals
    -/
  have mon : (∏ i : n, (X - C (M i i))).Monic := by
    apply monic_prod_of_monic univ fun i : n => X - C (M i i)
    simp [monic_X_sub_C]
  /-
    case neg
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    a✝ : Nontrivial R
    h : Not (Eq (Fintype.card n) 0)
    mon : (Finset.univ.prod fun i => HSub.hSub Polynomial.X (Polynomial.C (M i i)) …
    ⊢ M.charpoly.Monic
  -/
  rw [← sub_add_cancel (∏ i : n, (X - C (M i i))) M.charpoly] at mon
  /-
    case neg
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    a✝ : Nontrivial R
    h : Not (Eq (Fintype.card n) 0)
    mon : (HAdd.hAdd (HSub.hSub (Finset.univ.prod fun i => HSub.hSub Polynomial.X  …
    ⊢ M.charpoly.Monic
  -/
  rw [Monic] at *
  /-
    case neg
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    a✝ : Nontrivial R
    h : Not (Eq (Fintype.card n) 0)
    mon : Eq (HAdd.hAdd (HSub.hSub (Finset.univ.prod fun i => HSub.hSub Polynomial …
    ⊢ Eq M.charpoly.leadingCoeff 1
  -/
  rwa [leadingCoeff_add_of_degree_lt] at mon
  /-
    case neg
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    a✝ : Nontrivial R
    h : Not (Eq (Fintype.card n) 0)
    mon : Eq (HAdd.hAdd (HSub.hSub (Finset.univ.prod fun i => HSub.hSub Polynomial …
    ⊢ LT.lt (HSub.hSub (Finset.univ.prod fun i => HSub.hSub Polynomial.X (Polynomi …
  -/
  rw [charpoly_degree_eq_dim]
  /-
    case neg
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    a✝ : Nontrivial R
    h : Not (Eq (Fintype.card n) 0)
    mon : Eq (HAdd.hAdd (HSub.hSub (Finset.univ.prod fun i => HSub.hSub Polynomial …
    ⊢ LT.lt (HSub.hSub (Finset.univ.prod fun i => HSub.hSub Polynomial.X (Polynomi …
  -/
  rw [← neg_sub]
  /-
    case neg
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    a✝ : Nontrivial R
    h : Not (Eq (Fintype.card n) 0)
    mon : Eq (HAdd.hAdd (HSub.hSub (Finset.univ.prod fun i => HSub.hSub Polynomial …
    ⊢ LT.lt (Neg.neg (HSub.hSub M.charpoly (Finset.univ.prod fun i => HSub.hSub Po …
  -/
  rw [degree_neg]
  /-
    case neg
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    a✝ : Nontrivial R
    h : Not (Eq (Fintype.card n) 0)
    mon : Eq (HAdd.hAdd (HSub.hSub (Finset.univ.prod fun i => HSub.hSub Polynomial …
    ⊢ LT.lt (HSub.hSub M.charpoly (Finset.univ.prod fun i => HSub.hSub Polynomial. …
  -/
  apply lt_trans (charpoly_sub_diagonal_degree_lt M)
  /-
    case neg
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    a✝ : Nontrivial R
    h : Not (Eq (Fintype.card n) 0)
    mon : Eq (HAdd.hAdd (HSub.hSub (Finset.univ.prod fun i => HSub.hSub Polynomial …
    ⊢ LT.lt ↑(HSub.hSub (Fintype.card n) 1) ↑(Fintype.card n)
  -/
  rw [Nat.cast_lt]
  /-
    case neg
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    a✝ : Nontrivial R
    h : Not (Eq (Fintype.card n) 0)
    mon : Eq (HAdd.hAdd (HSub.hSub (Finset.univ.prod fun i => HSub.hSub Polynomial …
    ⊢ LT.lt (HSub.hSub (Fintype.card n) 1) (Fintype.card n)
  -/
  rw [← Nat.pred_eq_sub_one]
  /-
    case neg
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    a✝ : Nontrivial R
    h : Not (Eq (Fintype.card n) 0)
    mon : Eq (HAdd.hAdd (HSub.hSub (Finset.univ.prod fun i => HSub.hSub Polynomial …
    ⊢ LT.lt (Fintype.card n).pred (Fintype.card n)
  -/
  apply Nat.pred_lt
  /-
    case neg.a
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    a✝ : Nontrivial R
    h : Not (Eq (Fintype.card n) 0)
    mon : Eq (HAdd.hAdd (HSub.hSub (Finset.univ.prod fun i => HSub.hSub Polynomial …
    ⊢ Ne (Fintype.card n) 0
  -/
  apply h
  /-
    🎉 no goals
  -/


/-- See also `Matrix.coeff_charpolyRev_eq_neg_trace`. -/
theorem trace_eq_neg_charpoly_coeff [Nonempty n] (M : Matrix n n R) :
    trace M = -M.charpoly.coeff (Fintype.card n - 1) := by
  rw [charpoly_coeff_eq_prod_coeff_of_le _ le_rfl, Fintype.card,
    prod_X_sub_C_coeff_card_pred univ (fun i : n => M i i) Fintype.card_pos, neg_neg, trace]
  /-
    R : Type u
    inst✝³ : CommRing R
    n : Type v
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    inst✝ : Nonempty n
    M : Matrix n n R
    ⊢ Eq (Finset.univ.sum fun i => M.diag i) (Finset.univ.sum fun i => M i i)
  -/
  simp_rw [diag_apply]
  /-
    🎉 no goals
  -/


theorem matPolyEquiv_symm_map_eval (M : (Matrix n n R)[X]) (r : R) :
    (matPolyEquiv.symm M).map (eval r) = M.eval (scalar n r) := by
  suffices ((aeval r).mapMatrix.comp matPolyEquiv.symm.toAlgHom : (Matrix n n R)[X] →ₐ[R] _) =
      (eval₂AlgHom' (AlgHom.id R _) (scalar n r)
        fun x => (scalar_commute _ (Commute.all _) _).symm) from
    DFunLike.congr_fun this M
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Polynomial (Matrix n n R)
    r : R
    ⊢ Eq ((Polynomial.aeval r).mapMatrix.comp ↑matPolyEquiv.symm) (Polynomial.eval …
  -/
  ext : 1
    /-
      case hC
      R : Type u
      inst✝² : CommRing R
      n : Type v
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      M : Polynomial (Matrix n n R)
      r : R
      ⊢ Eq (((Polynomial.aeval r).mapMatrix.comp ↑matPolyEquiv.symm).comp Polynomial …
    -/
  · ext M : 1
    /-
      case hC.H
      R : Type u
      inst✝² : CommRing R
      n : Type v
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      M✝ : Polynomial (Matrix n n R)
      r : R
      M : Matrix n n R
      ⊢ Eq ((((Polynomial.aeval r).mapMatrix.comp ↑matPolyEquiv.symm).comp Polynomia …
    -/
    simp [Function.comp_def]
    /-
      🎉 no goals
    -/
    /-
      case hX
      R : Type u
      inst✝² : CommRing R
      n : Type v
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      M : Polynomial (Matrix n n R)
      r : R
      ⊢ Eq (((Polynomial.aeval r).mapMatrix.comp ↑matPolyEquiv.symm) Polynomial.X) ( …
    -/
  · simp [smul_eq_diagonal_mul]
    /-
      🎉 no goals
    -/


theorem matPolyEquiv_eval_eq_map (M : Matrix n n R[X]) (r : R) :
    (matPolyEquiv M).eval (scalar n r) = M.map (eval r) := by
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n (Polynomial R)
    r : R
    ⊢ Eq (Polynomial.eval ((Matrix.scalar n) r) (matPolyEquiv M)) (M.map (Polynomi …
  -/
  simpa only [AlgEquiv.symm_apply_apply] using (matPolyEquiv_symm_map_eval (matPolyEquiv M) r).symm
  /-
    🎉 no goals
  -/

-- I feel like this should use `Polynomial.algHom_eval₂_algebraMap`

theorem matPolyEquiv_eval (M : Matrix n n R[X]) (r : R) (i j : n) :
    (matPolyEquiv M).eval (scalar n r) i j = (M i j).eval r := by
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n (Polynomial R)
    r : R
    i j : n
    ⊢ Eq (Polynomial.eval ((Matrix.scalar n) r) (matPolyEquiv M) i j) (Polynomial. …
  -/
  rw [matPolyEquiv_eval_eq_map, map_apply]
  /-
    🎉 no goals
  -/


theorem eval_det (M : Matrix n n R[X]) (r : R) :
    Polynomial.eval r M.det = (Polynomial.eval (scalar n r) (matPolyEquiv M)).det := by
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n (Polynomial R)
    r : R
    ⊢ Eq (Polynomial.eval r M.det) (Polynomial.eval ((Matrix.scalar n) r) (matPoly …
  -/
  rw [Polynomial.eval, ← coe_eval₂RingHom, RingHom.map_det]
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n (Polynomial R)
    r : R
    ⊢ Eq ((Polynomial.eval₂RingHom (RingHom.id R) r).mapMatrix M).det (Polynomial. …
  -/
  apply congr_arg det
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n (Polynomial R)
    r : R
    ⊢ Eq ((Polynomial.eval₂RingHom (RingHom.id R) r).mapMatrix M) (Polynomial.eval …
  -/
  ext
  /-
    case a
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n (Polynomial R)
    r : R
    i✝ j✝ : n
    ⊢ Eq ((Polynomial.eval₂RingHom (RingHom.id R) r).mapMatrix M i✝ j✝) (Polynomia …
  -/
  symm
  -- Porting note: `exact` was `convert`
  /-
    case a
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n (Polynomial R)
    r : R
    i✝ j✝ : n
    ⊢ Eq (Polynomial.eval ((Matrix.scalar n) r) (matPolyEquiv M) i✝ j✝) ((Polynomi …
  -/
  exact matPolyEquiv_eval _ _ _ _
  /-
    🎉 no goals
  -/


theorem det_eq_sign_charpoly_coeff (M : Matrix n n R) :
    M.det = (-1) ^ Fintype.card n * M.charpoly.coeff 0 := by
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    ⊢ Eq M.det (HMul.hMul (HPow.hPow (-1) (Fintype.card n)) (M.charpoly.coeff 0))
  -/
  rw [coeff_zero_eq_eval_zero, charpoly, eval_det, matPolyEquiv_charmatrix, ← det_smul]
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    ⊢ Eq M.det (HSMul.hSMul (-1) (Polynomial.eval ((Matrix.scalar n) 0) (HSub.hSub …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma eval_det_add_X_smul (A : Matrix n n R[X]) (M : Matrix n n R) :
    (det (A + (X : R[X]) • M.map C)).eval 0 = (det A).eval 0 := by
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    A : Matrix n n (Polynomial R)
    M : Matrix n n R
    ⊢ Eq (Polynomial.eval 0 (HAdd.hAdd A (HSMul.hSMul Polynomial.X (M.map ⇑Polynom …
  -/
  simp only [eval_det, map_zero, map_add, eval_add, Algebra.smul_def, _root_.map_mul]
  simp only [Algebra.algebraMap_eq_smul_one, matPolyEquiv_smul_one, map_X, X_mul, eval_mul_X,
    mul_zero, add_zero]


lemma derivative_det_one_add_X_smul_aux {n} (M : Matrix (Fin n) (Fin n) R) :
    (derivative <| det (1 + (X : R[X]) • M.map C)).eval 0 = trace M := by
  induction n with
  | zero => simp
  | succ n IH =>
    rw [det_succ_row_zero, map_sum, eval_finset_sum]
    simp only [add_apply, smul_apply, map_apply, smul_eq_mul, X_mul_C, submatrix_add,
      submatrix_smul, Pi.add_apply, Pi.smul_apply, submatrix_map, derivative_mul, map_add,
      derivative_C, zero_mul, derivative_X, mul_one, zero_add, eval_add, eval_mul, eval_C, eval_X,
      mul_zero, add_zero, eval_det_add_X_smul, eval_pow, eval_neg, eval_one]
    rw [Finset.sum_eq_single 0]
    · simp only [Fin.val_zero, pow_zero, derivative_one, eval_zero, one_apply_eq, eval_one,
        mul_one, zero_add, one_mul, Fin.succAbove_zero, submatrix_one _ (Fin.succ_injective _),
        det_one, IH, trace_submatrix_succ]
    · intro i _ hi
      cases n with
      | zero => exact (hi (Subsingleton.elim i 0)).elim
      | succ n =>
        simp only [one_apply_ne' hi, eval_zero, mul_zero, zero_add, zero_mul, add_zero]
        rw [det_eq_zero_of_column_eq_zero 0, eval_zero, mul_zero]
        intro j
        rw [submatrix_apply, Fin.succAbove_of_castSucc_lt, one_apply_ne]
        · exact (bne_iff_ne (a := Fin.succ j) (b := Fin.castSucc 0)).mp rfl
        · rw [Fin.castSucc_zero]; exact lt_of_le_of_ne (Fin.zero_le _) hi.symm
    · exact fun H ↦ (H <| Finset.mem_univ _).elim


/-- The derivative of `det (1 + M X)` at `0` is the trace of `M`. -/
lemma derivative_det_one_add_X_smul (M : Matrix n n R) :
    (derivative <| det (1 + (X : R[X]) • M.map C)).eval 0 = trace M := by
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    ⊢ Eq (Polynomial.eval 0 (Polynomial.derivative (HAdd.hAdd 1 (HSMul.hSMul Polyn …
  -/
  let e := Matrix.reindexLinearEquiv R R (Fintype.equivFin n) (Fintype.equivFin n)
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    e : LinearEquiv (RingHom.id R) (Matrix n n R) (Matrix (Fin (Fintype.card n)) ( …
    ⊢ Eq (Polynomial.eval 0 (Polynomial.derivative (HAdd.hAdd 1 (HSMul.hSMul Polyn …
  -/
  rw [← Matrix.det_reindexLinearEquiv_self R[X] (Fintype.equivFin n)]
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    e : LinearEquiv (RingHom.id R) (Matrix n n R) (Matrix (Fin (Fintype.card n)) ( …
    ⊢ Eq (Polynomial.eval 0 (Polynomial.derivative ((Matrix.reindexLinearEquiv (Po …
  -/
  convert derivative_det_one_add_X_smul_aux (e M)
    /-
      case h.e'_2.h.e'_4.h.e'_6.h.e'_6
      R : Type u
      inst✝² : CommRing R
      n : Type v
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      M : Matrix n n R
      e : LinearEquiv (RingHom.id R) (Matrix n n R) (Matrix (Fin (Fintype.card n)) ( …
      ⊢ Eq ((Matrix.reindexLinearEquiv (Polynomial R) (Polynomial R) (Fintype.equivF …
    -/
  · ext; simp [map_add, e]
         /-
           🎉 no goals
         -/
    /-
      case h.e'_3
      R : Type u
      inst✝² : CommRing R
      n : Type v
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      M : Matrix n n R
      e : LinearEquiv (RingHom.id R) (Matrix n n R) (Matrix (Fin (Fintype.card n)) ( …
      ⊢ Eq M.trace (e M).trace
    -/
  · delta trace
    /-
      case h.e'_3
      R : Type u
      inst✝² : CommRing R
      n : Type v
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      M : Matrix n n R
      e : LinearEquiv (RingHom.id R) (Matrix n n R) (Matrix (Fin (Fintype.card n)) ( …
      ⊢ Eq (Finset.univ.sum fun i => M.diag i) (Finset.univ.sum fun i => (e M).diag i)
    -/
    rw [← (Fintype.equivFin n).symm.sum_comp]
    /-
      case h.e'_3
      R : Type u
      inst✝² : CommRing R
      n : Type v
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      M : Matrix n n R
      e : LinearEquiv (RingHom.id R) (Matrix n n R) (Matrix (Fin (Fintype.card n)) ( …
      ⊢ Eq (Finset.univ.sum fun i => M.diag ((Fintype.equivFin n).symm i)) (Finset.u …
    -/
    simp_rw [e, reindexLinearEquiv_apply, reindex_apply, diag_apply, submatrix_apply]
    /-
      🎉 no goals
    -/


lemma coeff_det_one_add_X_smul_one (M : Matrix n n R) :
    (det (1 + (X : R[X]) • M.map C)).coeff 1 = trace M := by
  simp only [← derivative_det_one_add_X_smul, ← coeff_zero_eq_eval_zero,
    coeff_derivative, zero_add, Nat.cast_zero, mul_one]


lemma det_one_add_X_smul (M : Matrix n n R) :
    det (1 + (X : R[X]) • M.map C) =
      (1 : R[X]) + trace M • X + (det (1 + (X : R[X]) • M.map C)).divX.divX * X ^ 2 := by
  rw [Algebra.smul_def (trace M), ← C_eq_algebraMap, pow_two, ← mul_assoc, add_assoc,
    ← add_mul, ← coeff_det_one_add_X_smul_one, ← coeff_divX, add_comm (C _), divX_mul_X_add,
    add_comm (1 : R[X]), ← C.map_one]
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    ⊢ Eq (HAdd.hAdd 1 (HSMul.hSMul Polynomial.X (M.map ⇑Polynomial.C))).det (HAdd. …
  -/
  convert (divX_mul_X_add _).symm
  /-
    case h.e'_3.h.e'_6.h.e'_6
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    ⊢ Eq 1 ((HAdd.hAdd 1 (HSMul.hSMul Polynomial.X (M.map ⇑Polynomial.C))).det.coe …
  -/
  rw [coeff_zero_eq_eval_zero, eval_det_add_X_smul, det_one, eval_one]
  /-
    🎉 no goals
  -/


/-- The first two terms of the taylor expansion of `det (1 + r • M)` at `r = 0`. -/
lemma det_one_add_smul (r : R) (M : Matrix n n R) :
    det (1 + r • M) =
      1 + trace M * r + (det (1 + (X : R[X]) • M.map C)).divX.divX.eval r * r ^ 2 := by
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    r : R
    M : Matrix n n R
    ⊢ Eq (HAdd.hAdd 1 (HSMul.hSMul r M)).det (HAdd.hAdd (HAdd.hAdd 1 (HMul.hMul M. …
  -/
  simpa [eval_det, ← smul_eq_mul_diagonal] using congr_arg (eval r) (Matrix.det_one_add_X_smul M)
  /-
    🎉 no goals
  -/


theorem matPolyEquiv_eq_X_pow_sub_C {K : Type*} (k : ℕ) [Field K] (M : Matrix n n K) :
    matPolyEquiv ((expand K k : K[X] →+* K[X]).mapMatrix (charmatrix (M ^ k))) =
      X ^ k - C (M ^ k) := by
  -- Porting note: `i` and `j` are used later on, but were not mentioned in mathlib3
  /-
    n : Type v
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    K : Type u_1
    k : Nat
    inst✝ : Field K
    M : Matrix n n K
    ⊢ Eq (matPolyEquiv ((↑(Polynomial.expand K k)).mapMatrix (HPow.hPow M k).charm …
  -/
  ext m i j
  rw [coeff_sub, coeff_C, matPolyEquiv_coeff_apply, RingHom.mapMatrix_apply, Matrix.map_apply,
    AlgHom.coe_toRingHom, DMatrix.sub_apply, coeff_X_pow]
  /-
    case a.a
    n : Type v
    inst✝² : DecidableEq n
    inst✝¹ : Fintype n
    K : Type u_1
    k : Nat
    inst✝ : Field K
    M : Matrix n n K
    m : Nat
    i j : n
    ⊢ Eq (((Polynomial.expand K k) ((HPow.hPow M k).charmatrix i j)).coeff m) (HSu …
  -/
  by_cases hij : i = j
    /-
      case pos
      n : Type v
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      K : Type u_1
      k : Nat
      inst✝ : Field K
      M : Matrix n n K
      m : Nat
      i j : n
      hij : Eq i j
      ⊢ Eq (((Polynomial.expand K k) ((HPow.hPow M k).charmatrix i j)).coeff m) (HSu …
    -/
  · rw [hij, charmatrix_apply_eq, map_sub, expand_C, expand_X, coeff_sub, coeff_X_pow, coeff_C]
                             -- Porting note: the second `Matrix.` was `DMatrix.`
    /-
      case pos
      n : Type v
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      K : Type u_1
      k : Nat
      inst✝ : Field K
      M : Matrix n n K
      m : Nat
      i j : n
      hij : Eq i j
      ⊢ Eq (HSub.hSub (ite (Eq m k) 1 0) (ite (Eq m 0) (HPow.hPow M k j j) 0)) (HSub …
    -/
                             /-
                               🎉 no goals
                             -/
                             /-
                               🎉 no goals
                             -/
                             /-
                               🎉 no goals
                             -/
    split_ifs with mp m0 <;> simp only [Matrix.one_apply_eq, Matrix.zero_apply]
                             /-
                               🎉 no goals
                             -/
    /-
      case neg
      n : Type v
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      K : Type u_1
      k : Nat
      inst✝ : Field K
      M : Matrix n n K
      m : Nat
      i j : n
      hij : Not (Eq i j)
      ⊢ Eq (((Polynomial.expand K k) ((HPow.hPow M k).charmatrix i j)).coeff m) (HSu …
    -/
  · rw [charmatrix_apply_ne _ _ _ hij, map_neg, expand_C, coeff_neg, coeff_C]
    /-
      case neg
      n : Type v
      inst✝² : DecidableEq n
      inst✝¹ : Fintype n
      K : Type u_1
      k : Nat
      inst✝ : Field K
      M : Matrix n n K
      m : Nat
      i j : n
      hij : Not (Eq i j)
      ⊢ Eq (Neg.neg (ite (Eq m 0) (HPow.hPow M k i j) 0)) (HSub.hSub (ite (Eq m k) 1 …
    -/
    split_ifs with m0 mp <;>
      -- Porting note: again, the first `Matrix.` that was `DMatrix.`
      simp only [hij, zero_sub, Matrix.zero_apply, sub_zero, neg_zero, Matrix.one_apply_ne, Ne,
        not_false_iff]


/-- Any matrix polynomial `p` is equivalent under evaluation to `p %ₘ M.charpoly`; that is, `p`
is equivalent to a polynomial with degree less than the dimension of the matrix. -/
theorem aeval_eq_aeval_mod_charpoly (M : Matrix n n R) (p : R[X]) :
    aeval M p = aeval M (p %ₘ M.charpoly) :=
  (aeval_modByMonic_eq_self_of_root M.charpoly_monic M.aeval_self_charpoly).symm


/-- Any matrix power can be computed as the sum of matrix powers less than `Fintype.card n`.

TODO: add the statement for negative powers phrased with `zpow`. -/
theorem pow_eq_aeval_mod_charpoly (M : Matrix n n R) (k : ℕ) :
                                                /-
                                                  R : Type u
                                                  inst✝² : CommRing R
                                                  n : Type v
                                                  inst✝¹ : DecidableEq n
                                                  inst✝ : Fintype n
                                                  M : Matrix n n R
                                                  k : Nat
                                                  ⊢ Eq (HPow.hPow M k) ((Polynomial.aeval M) ((HPow.hPow Polynomial.X k).modByMo …
                                                -/
    M ^ k = aeval M (X ^ k %ₘ M.charpoly) := by rw [← aeval_eq_aeval_mod_charpoly, map_pow, aeval_X]
                                                /-
                                                  🎉 no goals
                                                -/


theorem coeff_charpoly_mem_ideal_pow {I : Ideal R} (h : ∀ i j, M i j ∈ I) (k : ℕ) :
    M.charpoly.coeff k ∈ I ^ (Fintype.card n - k) := by
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    I : Ideal R
    h : ∀ (i j : n), Membership.mem I (M i j)
    k : Nat
    ⊢ Membership.mem (HPow.hPow I (HSub.hSub (Fintype.card n) k)) (M.charpoly.coef …
  -/
  delta charpoly
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    I : Ideal R
    h : ∀ (i j : n), Membership.mem I (M i j)
    k : Nat
    ⊢ Membership.mem (HPow.hPow I (HSub.hSub (Fintype.card n) k)) (M.charmatrix.de …
  -/
  rw [Matrix.det_apply, finset_sum_coeff]
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    I : Ideal R
    h : ∀ (i j : n), Membership.mem I (M i j)
    k : Nat
    ⊢ Membership.mem (HPow.hPow I (HSub.hSub (Fintype.card n) k)) (Finset.univ.sum …
  -/
  apply sum_mem
  /-
    case h
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    I : Ideal R
    h : ∀ (i j : n), Membership.mem I (M i j)
    k : Nat
    ⊢ ∀ (c : Equiv.Perm n), Membership.mem Finset.univ c → Membership.mem (HPow.hP …
  -/
  rintro c -
  /-
    case h
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    I : Ideal R
    h : ∀ (i j : n), Membership.mem I (M i j)
    k : Nat
    c : Equiv.Perm n
    ⊢ Membership.mem (HPow.hPow I (HSub.hSub (Fintype.card n) k)) ((HSMul.hSMul (E …
  -/
  rw [coeff_smul, Submodule.smul_mem_iff']
  /-
    case h
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    I : Ideal R
    h : ∀ (i j : n), Membership.mem I (M i j)
    k : Nat
    c : Equiv.Perm n
    ⊢ Membership.mem (HPow.hPow I (HSub.hSub (Fintype.card n) k)) ((Finset.univ.pr …
  -/
  have : ∑ x : n, 1 = Fintype.card n := by rw [Finset.sum_const, card_univ, smul_eq_mul, mul_one]
  /-
    case h
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    I : Ideal R
    h : ∀ (i j : n), Membership.mem I (M i j)
    k : Nat
    c : Equiv.Perm n
    this : Eq (Finset.univ.sum fun x => 1) (Fintype.card n)
    ⊢ Membership.mem (HPow.hPow I (HSub.hSub (Fintype.card n) k)) ((Finset.univ.pr …
  -/
  rw [← this]
  /-
    case h
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    I : Ideal R
    h : ∀ (i j : n), Membership.mem I (M i j)
    k : Nat
    c : Equiv.Perm n
    this : Eq (Finset.univ.sum fun x => 1) (Fintype.card n)
    ⊢ Membership.mem (HPow.hPow I (HSub.hSub (Finset.univ.sum fun x => 1) k)) ((Fi …
  -/
  apply coeff_prod_mem_ideal_pow_tsub
  /-
    case h.h
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    I : Ideal R
    h : ∀ (i j : n), Membership.mem I (M i j)
    k : Nat
    c : Equiv.Perm n
    this : Eq (Finset.univ.sum fun x => 1) (Fintype.card n)
    ⊢ ∀ (i : n), Membership.mem Finset.univ i → ∀ (k : Nat), Membership.mem (HPow. …
  -/
  rintro i - (_ | k)
  · rw [tsub_zero, pow_one, charmatrix_apply, coeff_sub, ← smul_one_eq_diagonal, smul_apply,
      smul_eq_mul, coeff_X_mul_zero, coeff_C_zero, zero_sub]
    /-
      case h.h.zero
      R : Type u
      inst✝² : CommRing R
      n : Type v
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      M : Matrix n n R
      I : Ideal R
      h : ∀ (i j : n), Membership.mem I (M i j)
      k : Nat
      c : Equiv.Perm n
      this : Eq (Finset.univ.sum fun x => 1) (Fintype.card n)
      i : n
      ⊢ Membership.mem I (Neg.neg (M (c i) i))
    -/
    apply neg_mem  -- Porting note: was `rw [neg_mem_iff]`, but Lean could not synth `NegMemClass`
    /-
      case h.h.zero.a
      R : Type u
      inst✝² : CommRing R
      n : Type v
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      M : Matrix n n R
      I : Ideal R
      h : ∀ (i j : n), Membership.mem I (M i j)
      k : Nat
      c : Equiv.Perm n
      this : Eq (Finset.univ.sum fun x => 1) (Fintype.card n)
      i : n
      ⊢ Membership.mem I (M (c i) i)
    -/
    exact h (c i) i
    /-
      🎉 no goals
    -/
    /-
      case h.h.succ
      R : Type u
      inst✝² : CommRing R
      n : Type v
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      M : Matrix n n R
      I : Ideal R
      h : ∀ (i j : n), Membership.mem I (M i j)
      k✝ : Nat
      c : Equiv.Perm n
      this : Eq (Finset.univ.sum fun x => 1) (Fintype.card n)
      i : n
      k : Nat
      ⊢ Membership.mem (HPow.hPow I (HSub.hSub 1 (HAdd.hAdd k 1))) ((M.charmatrix (c …
    -/
  · rw [add_comm, tsub_self_add, pow_zero, Ideal.one_eq_top]
    /-
      case h.h.succ
      R : Type u
      inst✝² : CommRing R
      n : Type v
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      M : Matrix n n R
      I : Ideal R
      h : ∀ (i j : n), Membership.mem I (M i j)
      k✝ : Nat
      c : Equiv.Perm n
      this : Eq (Finset.univ.sum fun x => 1) (Fintype.card n)
      i : n
      k : Nat
      ⊢ Membership.mem Top.top ((M.charmatrix (c i) i).coeff (HAdd.hAdd 1 k))
    -/
    exact Submodule.mem_top
    /-
      🎉 no goals
    -/


/-- The reverse of the characteristic polynomial of a matrix.

It has some advantages over the characteristic polynomial, including the fact that it can be
extended to infinite dimensions (for appropriate operators). In such settings it is known as the
"characteristic power series". -/
def charpolyRev (M : Matrix n n R) : R[X] := det (1 - (X : R[X]) • M.map C)


lemma reverse_charpoly (M : Matrix n n R) :
    M.charpoly.reverse = M.charpolyRev := by
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    ⊢ Eq M.charpoly.reverse M.charpolyRev
  -/
  nontriviality R
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    a✝ : Nontrivial R
    ⊢ Eq M.charpoly.reverse M.charpolyRev
  -/
  let t : R[T;T⁻¹] := T 1
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    a✝ : Nontrivial R
    t : LaurentPolynomial R := LaurentPolynomial.T 1
    ⊢ Eq M.charpoly.reverse M.charpolyRev
  -/
  let t_inv : R[T;T⁻¹] := T (-1)
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    a✝ : Nontrivial R
    t : LaurentPolynomial R := LaurentPolynomial.T 1
    t_inv : LaurentPolynomial R := LaurentPolynomial.T (-1)
    ⊢ Eq M.charpoly.reverse M.charpolyRev
  -/
  let p : R[T;T⁻¹] := det (scalar n t - M.map LaurentPolynomial.C)
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    a✝ : Nontrivial R
    t : LaurentPolynomial R := LaurentPolynomial.T 1
    t_inv : LaurentPolynomial R := LaurentPolynomial.T (-1)
    p : LaurentPolynomial R := (HSub.hSub ((Matrix.scalar n) t) (M.map ⇑LaurentPol …
    ⊢ Eq M.charpoly.reverse M.charpolyRev
  -/
  let q : R[T;T⁻¹] := det (1 - scalar n t * M.map LaurentPolynomial.C)
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    a✝ : Nontrivial R
    t : LaurentPolynomial R := LaurentPolynomial.T 1
    t_inv : LaurentPolynomial R := LaurentPolynomial.T (-1)
    p : LaurentPolynomial R := (HSub.hSub ((Matrix.scalar n) t) (M.map ⇑LaurentPol …
    q : LaurentPolynomial R := (HSub.hSub 1 (HMul.hMul ((Matrix.scalar n) t) (M.ma …
    ⊢ Eq M.charpoly.reverse M.charpolyRev
  -/
  have ht : t_inv * t = 1 := by rw [← T_add, neg_add_cancel, T_zero]
  have hp : toLaurentAlg M.charpoly = p := by
    simp [p, t, charpoly, charmatrix, AlgHom.map_det, map_sub, map_smul']
  have hq : toLaurentAlg M.charpolyRev = q := by
    simp [q, t, charpolyRev, AlgHom.map_det, map_sub, map_smul', smul_eq_diagonal_mul]
  suffices t_inv ^ Fintype.card n * p = invert q by
    apply toLaurent_injective
    rwa [toLaurent_reverse, ← coe_toLaurentAlg, hp, hq, ← involutive_invert.injective.eq_iff,
      _root_.map_mul, involutive_invert p, charpoly_natDegree_eq_dim,
      ← mul_one (Fintype.card n : ℤ), ← T_pow, map_pow, invert_T, mul_comm]
  rw [← det_smul, smul_sub, scalar_apply, ← diagonal_smul, Pi.smul_def, smul_eq_mul, ht,
    diagonal_one, invert.map_det]
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    a✝ : Nontrivial R
    t : LaurentPolynomial R := LaurentPolynomial.T 1
    t_inv : LaurentPolynomial R := LaurentPolynomial.T (-1)
    p : LaurentPolynomial R := (HSub.hSub ((Matrix.scalar n) t) (M.map ⇑LaurentPol …
    q : LaurentPolynomial R := (HSub.hSub 1 (HMul.hMul ((Matrix.scalar n) t) (M.ma …
    ht : Eq (HMul.hMul t_inv t) 1
    hp : Eq (Polynomial.toLaurentAlg M.charpoly) p
    hq : Eq (Polynomial.toLaurentAlg M.charpolyRev) q
    ⊢ Eq (HSub.hSub 1 (HSMul.hSMul t_inv (M.map ⇑LaurentPolynomial.C))).det (Laure …
  -/
  simp [t_inv, map_sub, _root_.map_one, _root_.map_mul, t, map_smul', smul_eq_diagonal_mul]
  /-
    🎉 no goals
  -/



@[simp] lemma eval_charpolyRev :
    eval 0 M.charpolyRev = 1 := by
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    ⊢ Eq (Polynomial.eval 0 M.charpolyRev) 1
  -/
  rw [charpolyRev, ← coe_evalRingHom, RingHom.map_det, ← det_one (R := R) (n := n)]
  have : (1 - (X : R[X]) • M.map C).map (eval 0) = 1 := by
    ext i j; rcases eq_or_ne i j with hij | hij <;> simp [hij, one_apply]
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    this : Eq ((HSub.hSub 1 (HSMul.hSMul Polynomial.X (M.map ⇑Polynomial.C))).map  …
    ⊢ Eq ((Polynomial.evalRingHom 0).mapMatrix (HSub.hSub 1 (HSMul.hSMul Polynomia …
  -/
  congr
  /-
    🎉 no goals
  -/


@[simp] lemma coeff_charpolyRev_eq_neg_trace (M : Matrix n n R) :
    coeff M.charpolyRev 1 = - trace M := by
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    ⊢ Eq (M.charpolyRev.coeff 1) (Neg.neg M.trace)
  -/
  nontriviality R
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    a✝ : Nontrivial R
    ⊢ Eq (M.charpolyRev.coeff 1) (Neg.neg M.trace)
  -/
  cases isEmpty_or_nonempty n
    /-
      case inl
      R : Type u
      inst✝² : CommRing R
      n : Type v
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      M : Matrix n n R
      a✝ : Nontrivial R
      h✝ : IsEmpty n
      ⊢ Eq (M.charpolyRev.coeff 1) (Neg.neg M.trace)
    -/
  · simp [charpolyRev, coeff_one]
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u
      inst✝² : CommRing R
      n : Type v
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      M : Matrix n n R
      a✝ : Nontrivial R
      h✝ : Nonempty n
      ⊢ Eq (M.charpolyRev.coeff 1) (Neg.neg M.trace)
    -/
  · simp [trace_eq_neg_charpoly_coeff M, ← M.reverse_charpoly, nextCoeff]
    /-
      🎉 no goals
    -/


lemma isUnit_charpolyRev_of_isNilpotent (hM : IsNilpotent M) :
    IsUnit M.charpolyRev := by
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    hM : IsNilpotent M
    ⊢ IsUnit M.charpolyRev
  -/
  obtain ⟨k, hk⟩ := hM
  replace hk : 1 - (X : R[X]) • M.map C ∣ 1 := by
    convert one_sub_dvd_one_sub_pow ((X : R[X]) • M.map C) k
    rw [← C.mapMatrix_apply, smul_pow, ← map_pow, hk, map_zero, smul_zero, sub_zero]
  /-
    case intro
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    k : Nat
    hk : Dvd.dvd (HSub.hSub 1 (HSMul.hSMul Polynomial.X (M.map ⇑Polynomial.C))) 1
    ⊢ IsUnit M.charpolyRev
  -/
  apply isUnit_of_dvd_one
  /-
    case intro.h
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    k : Nat
    hk : Dvd.dvd (HSub.hSub 1 (HSMul.hSMul Polynomial.X (M.map ⇑Polynomial.C))) 1
    ⊢ Dvd.dvd M.charpolyRev 1
  -/
  rw [← det_one (R := R[X]) (n := n)]
  /-
    case intro.h
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    k : Nat
    hk : Dvd.dvd (HSub.hSub 1 (HSMul.hSMul Polynomial.X (M.map ⇑Polynomial.C))) 1
    ⊢ Dvd.dvd M.charpolyRev (Matrix.det 1)
  -/
  exact map_dvd detMonoidHom hk
  /-
    🎉 no goals
  -/


lemma isNilpotent_trace_of_isNilpotent (hM : IsNilpotent M) :
    IsNilpotent (trace M) := by
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    hM : IsNilpotent M
    ⊢ IsNilpotent M.trace
  -/
  cases isEmpty_or_nonempty n
    /-
      case inl
      R : Type u
      inst✝² : CommRing R
      n : Type v
      inst✝¹ : DecidableEq n
      inst✝ : Fintype n
      M : Matrix n n R
      hM : IsNilpotent M
      h✝ : IsEmpty n
      ⊢ IsNilpotent M.trace
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    hM : IsNilpotent M
    h✝ : Nonempty n
    ⊢ IsNilpotent M.trace
  -/
  suffices IsNilpotent (coeff (charpolyRev M) 1) by simpa using this
  exact (isUnit_iff_coeff_isUnit_isNilpotent.mp (isUnit_charpolyRev_of_isNilpotent hM)).2
    _ one_ne_zero


lemma isNilpotent_charpoly_sub_pow_of_isNilpotent (hM : IsNilpotent M) :
    IsNilpotent (M.charpoly - X ^ (Fintype.card n)) := by
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    hM : IsNilpotent M
    ⊢ IsNilpotent (HSub.hSub M.charpoly (HPow.hPow Polynomial.X (Fintype.card n)))
  -/
  nontriviality R
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    hM : IsNilpotent M
    a✝ : Nontrivial R
    ⊢ IsNilpotent (HSub.hSub M.charpoly (HPow.hPow Polynomial.X (Fintype.card n)))
  -/
  let p : R[X] := M.charpolyRev
  have hp : p - 1 = X * (p /ₘ X) := by
    conv_lhs => rw [← modByMonic_add_div p monic_X]
    simp [p, modByMonic_X]
  have : IsNilpotent (p /ₘ X) :=
    (Polynomial.isUnit_iff'.mp (isUnit_charpolyRev_of_isNilpotent hM)).2
  have aux : (M.charpoly - X ^ (Fintype.card n)).natDegree ≤ M.charpoly.natDegree :=
    le_trans (natDegree_sub_le _ _) (by simp)
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    hM : IsNilpotent M
    a✝ : Nontrivial R
    p : Polynomial R := M.charpolyRev
    hp : Eq (HSub.hSub p 1) (HMul.hMul Polynomial.X (p.divByMonic Polynomial.X))
    this : IsNilpotent (p.divByMonic Polynomial.X)
    aux : LE.le (HSub.hSub M.charpoly (HPow.hPow Polynomial.X (Fintype.card n))).n …
    ⊢ IsNilpotent (HSub.hSub M.charpoly (HPow.hPow Polynomial.X (Fintype.card n)))
  -/
  rw [← isNilpotent_reflect_iff aux, reflect_sub, ← reverse, M.reverse_charpoly]
  /-
    R : Type u
    inst✝² : CommRing R
    n : Type v
    inst✝¹ : DecidableEq n
    inst✝ : Fintype n
    M : Matrix n n R
    hM : IsNilpotent M
    a✝ : Nontrivial R
    p : Polynomial R := M.charpolyRev
    hp : Eq (HSub.hSub p 1) (HMul.hMul Polynomial.X (p.divByMonic Polynomial.X))
    this : IsNilpotent (p.divByMonic Polynomial.X)
    aux : LE.le (HSub.hSub M.charpoly (HPow.hPow Polynomial.X (Fintype.card n))).n …
    ⊢ IsNilpotent (HSub.hSub M.charpolyRev (Polynomial.reflect M.charpoly.natDegre …
  -/
  simpa [p, hp]
  /-
    🎉 no goals
  -/


