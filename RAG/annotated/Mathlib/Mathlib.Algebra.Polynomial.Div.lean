theorem X_dvd_iff {f : R[X]} : X ∣ f ↔ f.coeff 0 = 0 :=
                      /-
                        R : Type u
                        inst✝ : Semiring R
                        f : Polynomial R
                        x✝ : Dvd.dvd Polynomial.X f
                        g : Polynomial R
                        hfg : Eq f (HMul.hMul Polynomial.X g)
                        ⊢ Eq (f.coeff 0) 0
                      -/
  ⟨fun ⟨g, hfg⟩ => by rw [hfg, coeff_X_mul_zero], fun hf =>
                      /-
                        🎉 no goals
                      -/
                /-
                  R : Type u
                  inst✝ : Semiring R
                  f : Polynomial R
                  hf : Eq (f.coeff 0) 0
                  ⊢ Eq f (HMul.hMul Polynomial.X f.divX)
                -/
    ⟨f.divX, by rw [← add_zero (X * f.divX), ← C_0, ← hf, X_mul_divX_add]⟩⟩
                /-
                  🎉 no goals
                -/


theorem X_pow_dvd_iff {f : R[X]} {n : ℕ} : X ^ n ∣ f ↔ ∀ d < n, f.coeff d = 0 :=
  ⟨fun ⟨g, hgf⟩ d hd => by
    /-
      R : Type u
      inst✝ : Semiring R
      f : Polynomial R
      n : Nat
      x✝ : Dvd.dvd (HPow.hPow Polynomial.X n) f
      d : Nat
      hd : LT.lt d n
      g : Polynomial R
      hgf : Eq f (HMul.hMul (HPow.hPow Polynomial.X n) g)
      ⊢ Eq (f.coeff d) 0
    -/
    simp only [hgf, coeff_X_pow_mul', ite_eq_right_iff, not_le_of_lt hd, IsEmpty.forall_iff],
    /-
      🎉 no goals
    -/
    fun hd => by
    induction n with
    | zero => simp [pow_zero, one_dvd]
    | succ n hn =>
      obtain ⟨g, hgf⟩ := hn fun d : ℕ => fun H : d < n => hd _ (Nat.lt_succ_of_lt H)
      have := coeff_X_pow_mul g n 0
      rw [zero_add, ← hgf, hd n (Nat.lt_succ_self n)] at this
      obtain ⟨k, hgk⟩ := Polynomial.X_dvd_iff.mpr this.symm
      use k
      rwa [pow_succ, mul_assoc, ← hgk]⟩


theorem finiteMultiplicity_of_degree_pos_of_monic (hp : (0 : WithBot ℕ) < degree p) (hmp : Monic p)
    (hq : q ≠ 0) : FiniteMultiplicity p q :=
  have zn0 : (0 : R) ≠ 1 :=
    haveI := Nontrivial.of_polynomial_ne hq
    zero_ne_one
  ⟨natDegree q, fun ⟨r, hr⟩ => by
    /-
      R : Type u
      inst✝ : Semiring R
      p q : Polynomial R
      hp : LT.lt 0 p.degree
      hmp : p.Monic
      hq : Ne q 0
      zn0 : Ne 0 1
      x✝ : Dvd.dvd (HPow.hPow p (HAdd.hAdd q.natDegree 1)) q
      r : Polynomial R
      hr : Eq q (HMul.hMul (HPow.hPow p (HAdd.hAdd q.natDegree 1)) r)
      ⊢ False
    -/
    have hp0 : p ≠ 0 := fun hp0 => by simp [hp0] at hp
    /-
      R : Type u
      inst✝ : Semiring R
      p q : Polynomial R
      hp : LT.lt 0 p.degree
      hmp : p.Monic
      hq : Ne q 0
      zn0 : Ne 0 1
      x✝ : Dvd.dvd (HPow.hPow p (HAdd.hAdd q.natDegree 1)) q
      r : Polynomial R
      hr : Eq q (HMul.hMul (HPow.hPow p (HAdd.hAdd q.natDegree 1)) r)
      hp0 : Ne p 0
      ⊢ False
    -/
    have hr0 : r ≠ 0 := fun hr0 => by subst hr0; simp [hq] at hr
    /-
      R : Type u
      inst✝ : Semiring R
      p q : Polynomial R
      hp : LT.lt 0 p.degree
      hmp : p.Monic
      hq : Ne q 0
      zn0 : Ne 0 1
      x✝ : Dvd.dvd (HPow.hPow p (HAdd.hAdd q.natDegree 1)) q
      r : Polynomial R
      hr : Eq q (HMul.hMul (HPow.hPow p (HAdd.hAdd q.natDegree 1)) r)
      hp0 : Ne p 0
      hr0 : Ne r 0
      ⊢ False
    -/
    have hpn1 : leadingCoeff p ^ (natDegree q + 1) = 1 := by simp [show _ = _ from hmp]
    /-
      R : Type u
      inst✝ : Semiring R
      p q : Polynomial R
      hp : LT.lt 0 p.degree
      hmp : p.Monic
      hq : Ne q 0
      zn0 : Ne 0 1
      x✝ : Dvd.dvd (HPow.hPow p (HAdd.hAdd q.natDegree 1)) q
      r : Polynomial R
      hr : Eq q (HMul.hMul (HPow.hPow p (HAdd.hAdd q.natDegree 1)) r)
      hp0 : Ne p 0
      hr0 : Ne r 0
      hpn1 : Eq (HPow.hPow p.leadingCoeff (HAdd.hAdd q.natDegree 1)) 1
      ⊢ False
    -/
    have hpn0' : leadingCoeff p ^ (natDegree q + 1) ≠ 0 := hpn1.symm ▸ zn0.symm
    have hpnr0 : leadingCoeff (p ^ (natDegree q + 1)) * leadingCoeff r ≠ 0 := by
      simp only [leadingCoeff_pow' hpn0', leadingCoeff_eq_zero, hpn1, one_pow, one_mul, Ne,
          hr0, not_false_eq_true]
    have hnp : 0 < natDegree p := Nat.cast_lt.1 <| by
      rw [← degree_eq_natDegree hp0]; exact hp
    /-
      R : Type u
      inst✝ : Semiring R
      p q : Polynomial R
      hp : LT.lt 0 p.degree
      hmp : p.Monic
      hq : Ne q 0
      zn0 : Ne 0 1
      x✝ : Dvd.dvd (HPow.hPow p (HAdd.hAdd q.natDegree 1)) q
      r : Polynomial R
      hr : Eq q (HMul.hMul (HPow.hPow p (HAdd.hAdd q.natDegree 1)) r)
      hp0 : Ne p 0
      hr0 : Ne r 0
      hpn1 : Eq (HPow.hPow p.leadingCoeff (HAdd.hAdd q.natDegree 1)) 1
      hpn0' : Ne (HPow.hPow p.leadingCoeff (HAdd.hAdd q.natDegree 1)) 0
      hpnr0 : Ne (HMul.hMul (HPow.hPow p (HAdd.hAdd q.natDegree 1)).leadingCoeff r.l …
      hnp : LT.lt 0 p.natDegree
      ⊢ False
    -/
    have := congr_arg natDegree hr
    /-
      R : Type u
      inst✝ : Semiring R
      p q : Polynomial R
      hp : LT.lt 0 p.degree
      hmp : p.Monic
      hq : Ne q 0
      zn0 : Ne 0 1
      x✝ : Dvd.dvd (HPow.hPow p (HAdd.hAdd q.natDegree 1)) q
      r : Polynomial R
      hr : Eq q (HMul.hMul (HPow.hPow p (HAdd.hAdd q.natDegree 1)) r)
      hp0 : Ne p 0
      hr0 : Ne r 0
      hpn1 : Eq (HPow.hPow p.leadingCoeff (HAdd.hAdd q.natDegree 1)) 1
      hpn0' : Ne (HPow.hPow p.leadingCoeff (HAdd.hAdd q.natDegree 1)) 0
      hpnr0 : Ne (HMul.hMul (HPow.hPow p (HAdd.hAdd q.natDegree 1)).leadingCoeff r.l …
      hnp : LT.lt 0 p.natDegree
      this : Eq q.natDegree (HMul.hMul (HPow.hPow p (HAdd.hAdd q.natDegree 1)) r).na …
      ⊢ False
    -/
    rw [natDegree_mul' hpnr0, natDegree_pow' hpn0', add_mul, add_assoc] at this
    exact
      ne_of_lt
        (lt_add_of_le_of_pos (le_mul_of_one_le_right (Nat.zero_le _) hnp)
          (add_pos_of_pos_of_nonneg (by rwa [one_mul]) (Nat.zero_le _)))
        this⟩


@[deprecated (since := "2024-11-30")]
alias multiplicity_finite_of_degree_pos_of_monic := finiteMultiplicity_of_degree_pos_of_monic


theorem div_wf_lemma (h : degree q ≤ degree p ∧ p ≠ 0) (hq : Monic q) :
    degree (p - q * (C (leadingCoeff p) * X ^ (natDegree p - natDegree q))) < degree p :=
  have hp : leadingCoeff p ≠ 0 := mt leadingCoeff_eq_zero.1 h.2
  have hq0 : q ≠ 0 := hq.ne_zero_of_polynomial_ne h.2
  have hlt : natDegree q ≤ natDegree p :=
    (Nat.cast_le (α := WithBot ℕ)).1
          /-
            R : Type u
            inst✝ : Ring R
            p q : Polynomial R
            h : And (LE.le q.degree p.degree) (Ne p 0)
            hq : q.Monic
            hp : Ne p.leadingCoeff 0
            hq0 : Ne q 0
            ⊢ LE.le ↑q.natDegree ↑p.natDegree
          -/
      (by rw [← degree_eq_natDegree h.2, ← degree_eq_natDegree hq0]; exact h.1)
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
  degree_sub_lt
    (by
      rw [hq.degree_mul_comm, hq.degree_mul, degree_C_mul_X_pow _ hp, degree_eq_natDegree h.2,
        degree_eq_natDegree hq0, ← Nat.cast_add, tsub_add_cancel_of_le hlt])
            /-
              R : Type u
              inst✝ : Ring R
              p q : Polynomial R
              h : And (LE.le q.degree p.degree) (Ne p 0)
              hq : q.Monic
              hp : Ne p.leadingCoeff 0
              hq0 : Ne q 0
              hlt : LE.le q.natDegree p.natDegree
              ⊢ Eq p.leadingCoeff (HMul.hMul q (HMul.hMul (Polynomial.C p.leadingCoeff) (HPo …
            -/
    h.2 (by rw [leadingCoeff_monic_mul hq, leadingCoeff_mul_X_pow, leadingCoeff_C])
            /-
              🎉 no goals
            -/


/-- See `divByMonic`. -/
noncomputable def divModByMonicAux : ∀ (_p : R[X]) {q : R[X]}, Monic q → R[X] × R[X]
  | p, q, hq =>
    letI := Classical.decEq R
    if h : degree q ≤ degree p ∧ p ≠ 0 then
      let z := C (leadingCoeff p) * X ^ (natDegree p - natDegree q)
      have _wf := div_wf_lemma h hq
      let dm := divModByMonicAux (p - q * z) hq
      ⟨z + dm.1, dm.2⟩
    else ⟨0, p⟩
  termination_by p => p


/-- `divByMonic` gives the quotient of `p` by a monic polynomial `q`. -/
def divByMonic (p q : R[X]) : R[X] :=
  letI := Classical.decEq R
  if hq : Monic q then (divModByMonicAux p hq).1 else 0


/-- `modByMonic` gives the remainder of `p` by a monic polynomial `q`. -/
def modByMonic (p q : R[X]) : R[X] :=
  letI := Classical.decEq R
  if hq : Monic q then (divModByMonicAux p hq).2 else p


@[inherit_doc]
infixl:70 " /ₘ " => divByMonic


@[inherit_doc]
infixl:70 " %ₘ " => modByMonic


theorem degree_modByMonic_lt [Nontrivial R] :
    ∀ (p : R[X]) {q : R[X]} (_hq : Monic q), degree (p %ₘ q) < degree q
  | p, q, hq =>
    letI := Classical.decEq R
    if h : degree q ≤ degree p ∧ p ≠ 0 then by
      /-
        R : Type u
        inst✝¹ : Ring R
        inst✝ : Nontrivial R
        p q : Polynomial R
        hq : q.Monic
        this : DecidableEq R := Classical.decEq R
        h : And (LE.le q.degree p.degree) (Ne p 0)
        ⊢ LT.lt (p.modByMonic q).degree q.degree
      -/
      have _wf := div_wf_lemma ⟨h.1, h.2⟩ hq
      have :=
        degree_modByMonic_lt (p - q * (C (leadingCoeff p) * X ^ (natDegree p - natDegree q))) hq
      /-
        R : Type u
        inst✝¹ : Ring R
        inst✝ : Nontrivial R
        p q : Polynomial R
        hq : q.Monic
        this✝ : DecidableEq R := Classical.decEq R
        h : And (LE.le q.degree p.degree) (Ne p 0)
        _wf : LT.lt (HSub.hSub p (HMul.hMul q (HMul.hMul (Polynomial.C p.leadingCoeff) …
        this : LT.lt ((HSub.hSub p (HMul.hMul q (HMul.hMul (Polynomial.C p.leadingCoef …
        ⊢ LT.lt (p.modByMonic q).degree q.degree
      -/
      unfold modByMonic at this ⊢
      /-
        R : Type u
        inst✝¹ : Ring R
        inst✝ : Nontrivial R
        p q : Polynomial R
        hq : q.Monic
        this✝ : DecidableEq R := Classical.decEq R
        h : And (LE.le q.degree p.degree) (Ne p 0)
        _wf : LT.lt (HSub.hSub p (HMul.hMul q (HMul.hMul (Polynomial.C p.leadingCoeff) …
        this : LT.lt (dite q.Monic (fun hq => ((HSub.hSub p (HMul.hMul q (HMul.hMul (P …
        ⊢ LT.lt (dite q.Monic (fun hq => (p.divModByMonicAux hq).2) fun hq => p).degre …
      -/
      unfold divModByMonicAux
      /-
        R : Type u
        inst✝¹ : Ring R
        inst✝ : Nontrivial R
        p q : Polynomial R
        hq : q.Monic
        this✝ : DecidableEq R := Classical.decEq R
        h : And (LE.le q.degree p.degree) (Ne p 0)
        _wf : LT.lt (HSub.hSub p (HMul.hMul q (HMul.hMul (Polynomial.C p.leadingCoeff) …
        this : LT.lt (dite q.Monic (fun hq => ((HSub.hSub p (HMul.hMul q (HMul.hMul (P …
        ⊢ LT.lt
            (dite q.Monic
                (fun h =>
                  (dite (And (LE.le q.degree p.degree) (Ne p 0))
                      (fun h_1 =>
                        let z := HMul.hMul (Polynomial.C p.leadingCoeff) (HPow.hPow Po …
                        letFun ⋯ fun _wf =>
                          let dm := (HSub.hSub p (HMul.hMul q z)).divModByMonicAux ⋯;
                          { fst := HAdd.hAdd z dm.1, snd := dm.2 })
                      fun h => { fst := 0, snd := p }).2)
                fun h => p).degree
            q.degree
      -/
      dsimp
      /-
        R : Type u
        inst✝¹ : Ring R
        inst✝ : Nontrivial R
        p q : Polynomial R
        hq : q.Monic
        this✝ : DecidableEq R := Classical.decEq R
        h : And (LE.le q.degree p.degree) (Ne p 0)
        _wf : LT.lt (HSub.hSub p (HMul.hMul q (HMul.hMul (Polynomial.C p.leadingCoeff) …
        this : LT.lt (dite q.Monic (fun hq => ((HSub.hSub p (HMul.hMul q (HMul.hMul (P …
        ⊢ LT.lt (dite q.Monic (fun h => (ite (And (LE.le q.degree p.degree) (Not (Eq p …
      -/
      rw [dif_pos hq] at this ⊢
      /-
        R : Type u
        inst✝¹ : Ring R
        inst✝ : Nontrivial R
        p q : Polynomial R
        hq : q.Monic
        this✝ : DecidableEq R := Classical.decEq R
        h : And (LE.le q.degree p.degree) (Ne p 0)
        _wf : LT.lt (HSub.hSub p (HMul.hMul q (HMul.hMul (Polynomial.C p.leadingCoeff) …
        this : LT.lt ((HSub.hSub p (HMul.hMul q (HMul.hMul (Polynomial.C p.leadingCoef …
        ⊢ LT.lt (ite (And (LE.le q.degree p.degree) (Not (Eq p 0))) { fst := HAdd.hAdd …
      -/
      rw [if_pos h]
      /-
        R : Type u
        inst✝¹ : Ring R
        inst✝ : Nontrivial R
        p q : Polynomial R
        hq : q.Monic
        this✝ : DecidableEq R := Classical.decEq R
        h : And (LE.le q.degree p.degree) (Ne p 0)
        _wf : LT.lt (HSub.hSub p (HMul.hMul q (HMul.hMul (Polynomial.C p.leadingCoeff) …
        this : LT.lt ((HSub.hSub p (HMul.hMul q (HMul.hMul (Polynomial.C p.leadingCoef …
        ⊢ LT.lt { fst := HAdd.hAdd (HMul.hMul (Polynomial.C p.leadingCoeff) (HPow.hPow …
      -/
      exact this
      /-
        🎉 no goals
      -/
    else
      Or.casesOn (not_and_or.1 h)
        (by
          /-
            R : Type u
            inst✝¹ : Ring R
            inst✝ : Nontrivial R
            p q : Polynomial R
            hq : q.Monic
            this : DecidableEq R := Classical.decEq R
            h : Not (And (LE.le q.degree p.degree) (Ne p 0))
            ⊢ Not (LE.le q.degree p.degree) → LT.lt (p.modByMonic q).degree q.degree
          -/
          unfold modByMonic divModByMonicAux
          /-
            R : Type u
            inst✝¹ : Ring R
            inst✝ : Nontrivial R
            p q : Polynomial R
            hq : q.Monic
            this : DecidableEq R := Classical.decEq R
            h : Not (And (LE.le q.degree p.degree) (Ne p 0))
            ⊢ Not (LE.le q.degree p.degree) →
                LT.lt
                  (dite q.Monic
                      (fun h =>
                        (dite (And (LE.le q.degree p.degree) (Ne p 0))
                            (fun h_1 =>
                              let z := HMul.hMul (Polynomial.C p.leadingCoeff) (HPow.hPow  …
                              letFun ⋯ fun _wf =>
                                let dm := (HSub.hSub p (HMul.hMul q z)).divModByMonicAux ⋯;
                                { fst := HAdd.hAdd z dm.1, snd := dm.2 })
                            fun h => { fst := 0, snd := p }).2)
                      fun h => p).degree
                  q.degree
          -/
          dsimp
          /-
            R : Type u
            inst✝¹ : Ring R
            inst✝ : Nontrivial R
            p q : Polynomial R
            hq : q.Monic
            this : DecidableEq R := Classical.decEq R
            h : Not (And (LE.le q.degree p.degree) (Ne p 0))
            ⊢ Not (LE.le q.degree p.degree) → LT.lt (dite q.Monic (fun h => (ite (And (LE. …
          -/
          rw [dif_pos hq, if_neg h]
          /-
            R : Type u
            inst✝¹ : Ring R
            inst✝ : Nontrivial R
            p q : Polynomial R
            hq : q.Monic
            this : DecidableEq R := Classical.decEq R
            h : Not (And (LE.le q.degree p.degree) (Ne p 0))
            ⊢ Not (LE.le q.degree p.degree) → LT.lt { fst := 0, snd := p }.2.degree q.degree
          -/
          exact lt_of_not_ge)
          /-
            🎉 no goals
          -/
        (by
          /-
            R : Type u
            inst✝¹ : Ring R
            inst✝ : Nontrivial R
            p q : Polynomial R
            hq : q.Monic
            this : DecidableEq R := Classical.decEq R
            h : Not (And (LE.le q.degree p.degree) (Ne p 0))
            ⊢ Not (Ne p 0) → LT.lt (p.modByMonic q).degree q.degree
          -/
          intro hp
          /-
            R : Type u
            inst✝¹ : Ring R
            inst✝ : Nontrivial R
            p q : Polynomial R
            hq : q.Monic
            this : DecidableEq R := Classical.decEq R
            h : Not (And (LE.le q.degree p.degree) (Ne p 0))
            hp : Not (Ne p 0)
            ⊢ LT.lt (p.modByMonic q).degree q.degree
          -/
          unfold modByMonic divModByMonicAux
          /-
            R : Type u
            inst✝¹ : Ring R
            inst✝ : Nontrivial R
            p q : Polynomial R
            hq : q.Monic
            this : DecidableEq R := Classical.decEq R
            h : Not (And (LE.le q.degree p.degree) (Ne p 0))
            hp : Not (Ne p 0)
            ⊢ LT.lt
                (dite q.Monic
                    (fun h =>
                      (dite (And (LE.le q.degree p.degree) (Ne p 0))
                          (fun h_1 =>
                            let z := HMul.hMul (Polynomial.C p.leadingCoeff) (HPow.hPow Po …
                            letFun ⋯ fun _wf =>
                              let dm := (HSub.hSub p (HMul.hMul q z)).divModByMonicAux ⋯;
                              { fst := HAdd.hAdd z dm.1, snd := dm.2 })
                          fun h => { fst := 0, snd := p }).2)
                    fun h => p).degree
                q.degree
          -/
          dsimp
          /-
            R : Type u
            inst✝¹ : Ring R
            inst✝ : Nontrivial R
            p q : Polynomial R
            hq : q.Monic
            this : DecidableEq R := Classical.decEq R
            h : Not (And (LE.le q.degree p.degree) (Ne p 0))
            hp : Not (Ne p 0)
            ⊢ LT.lt (dite q.Monic (fun h => (ite (And (LE.le q.degree p.degree) (Not (Eq p …
          -/
          rw [dif_pos hq, if_neg h, Classical.not_not.1 hp]
          /-
            R : Type u
            inst✝¹ : Ring R
            inst✝ : Nontrivial R
            p q : Polynomial R
            hq : q.Monic
            this : DecidableEq R := Classical.decEq R
            h : Not (And (LE.le q.degree p.degree) (Ne p 0))
            hp : Not (Ne p 0)
            ⊢ LT.lt { fst := 0, snd := 0 }.2.degree q.degree
          -/
          exact lt_of_le_of_ne bot_le (Ne.symm (mt degree_eq_bot.1 hq.ne_zero)))
          /-
            🎉 no goals
          -/
  termination_by p => p


theorem natDegree_modByMonic_lt (p : R[X]) {q : R[X]} (hmq : Monic q) (hq : q ≠ 1) :
    natDegree (p %ₘ q) < q.natDegree := by
  /-
    R : Type u
    inst✝ : Ring R
    p q : Polynomial R
    hmq : q.Monic
    hq : Ne q 1
    ⊢ LT.lt (p.modByMonic q).natDegree q.natDegree
  -/
  by_cases hpq : p %ₘ q = 0
    /-
      case pos
      R : Type u
      inst✝ : Ring R
      p q : Polynomial R
      hmq : q.Monic
      hq : Ne q 1
      hpq : Eq (p.modByMonic q) 0
      ⊢ LT.lt (p.modByMonic q).natDegree q.natDegree
    -/
  · rw [hpq, natDegree_zero, Nat.pos_iff_ne_zero]
    /-
      case pos
      R : Type u
      inst✝ : Ring R
      p q : Polynomial R
      hmq : q.Monic
      hq : Ne q 1
      hpq : Eq (p.modByMonic q) 0
      ⊢ Ne q.natDegree 0
    -/
    contrapose! hq
    /-
      case pos
      R : Type u
      inst✝ : Ring R
      p q : Polynomial R
      hmq : q.Monic
      hpq : Eq (p.modByMonic q) 0
      hq : Eq q.natDegree 0
      ⊢ Eq q 1
    -/
    exact eq_one_of_monic_natDegree_zero hmq hq
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u
      inst✝ : Ring R
      p q : Polynomial R
      hmq : q.Monic
      hq : Ne q 1
      hpq : Not (Eq (p.modByMonic q) 0)
      ⊢ LT.lt (p.modByMonic q).natDegree q.natDegree
    -/
  · haveI := Nontrivial.of_polynomial_ne hpq
    /-
      case neg
      R : Type u
      inst✝ : Ring R
      p q : Polynomial R
      hmq : q.Monic
      hq : Ne q 1
      hpq : Not (Eq (p.modByMonic q) 0)
      this : Nontrivial R
      ⊢ LT.lt (p.modByMonic q).natDegree q.natDegree
    -/
    exact natDegree_lt_natDegree hpq (degree_modByMonic_lt p hmq)
    /-
      🎉 no goals
    -/


@[simp]
theorem zero_modByMonic (p : R[X]) : 0 %ₘ p = 0 := by
  classical
  unfold modByMonic divModByMonicAux
  dsimp
  by_cases hp : Monic p
  · rw [dif_pos hp, if_neg (mt And.right (not_not_intro rfl)), Prod.snd_zero]
  · rw [dif_neg hp]


@[simp]
theorem zero_divByMonic (p : R[X]) : 0 /ₘ p = 0 := by
  classical
  unfold divByMonic divModByMonicAux
  dsimp
  by_cases hp : Monic p
  · rw [dif_pos hp, if_neg (mt And.right (not_not_intro rfl)), Prod.fst_zero]
  · rw [dif_neg hp]


@[simp]
theorem modByMonic_zero (p : R[X]) : p %ₘ 0 = p :=
  letI := Classical.decEq R
  if h : Monic (0 : R[X]) then by
    /-
      R : Type u
      inst✝ : Ring R
      p : Polynomial R
      this : DecidableEq R := Classical.decEq R
      h : Polynomial.Monic 0
      ⊢ Eq (p.modByMonic 0) p
    -/
    haveI := monic_zero_iff_subsingleton.mp h
    /-
      R : Type u
      inst✝ : Ring R
      p : Polynomial R
      this✝ : DecidableEq R := Classical.decEq R
      h : Polynomial.Monic 0
      this : Subsingleton R
      ⊢ Eq (p.modByMonic 0) p
    -/
    simp [eq_iff_true_of_subsingleton]
    /-
      🎉 no goals
    -/
          /-
            R : Type u
            inst✝ : Ring R
            p : Polynomial R
            this : DecidableEq R := Classical.decEq R
            h : Not (Polynomial.Monic 0)
            ⊢ Eq (p.modByMonic 0) p
          -/
  else by unfold modByMonic divModByMonicAux; rw [dif_neg h]
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
theorem divByMonic_zero (p : R[X]) : p /ₘ 0 = 0 :=
  letI := Classical.decEq R
  if h : Monic (0 : R[X]) then by
    /-
      R : Type u
      inst✝ : Ring R
      p : Polynomial R
      this : DecidableEq R := Classical.decEq R
      h : Polynomial.Monic 0
      ⊢ Eq (p.divByMonic 0) 0
    -/
    haveI := monic_zero_iff_subsingleton.mp h
    /-
      R : Type u
      inst✝ : Ring R
      p : Polynomial R
      this✝ : DecidableEq R := Classical.decEq R
      h : Polynomial.Monic 0
      this : Subsingleton R
      ⊢ Eq (p.divByMonic 0) 0
    -/
    simp [eq_iff_true_of_subsingleton]
    /-
      🎉 no goals
    -/
          /-
            R : Type u
            inst✝ : Ring R
            p : Polynomial R
            this : DecidableEq R := Classical.decEq R
            h : Not (Polynomial.Monic 0)
            ⊢ Eq (p.divByMonic 0) 0
          -/
  else by unfold divByMonic divModByMonicAux; rw [dif_neg h]
                                              /-
                                                🎉 no goals
                                              -/


theorem divByMonic_eq_of_not_monic (p : R[X]) (hq : ¬Monic q) : p /ₘ q = 0 :=
  dif_neg hq


theorem modByMonic_eq_of_not_monic (p : R[X]) (hq : ¬Monic q) : p %ₘ q = p :=
  dif_neg hq


theorem modByMonic_eq_self_iff [Nontrivial R] (hq : Monic q) : p %ₘ q = p ↔ degree p < degree q :=
  ⟨fun h => h ▸ degree_modByMonic_lt _ hq, fun h => by
    classical
    have : ¬degree q ≤ degree p := not_le_of_gt h
    unfold modByMonic divModByMonicAux; dsimp; rw [dif_pos hq, if_neg (mt And.left this)]⟩


theorem degree_modByMonic_le (p : R[X]) {q : R[X]} (hq : Monic q) : degree (p %ₘ q) ≤ degree q := by
  /-
    R : Type u
    inst✝ : Ring R
    p q : Polynomial R
    hq : q.Monic
    ⊢ LE.le (p.modByMonic q).degree q.degree
  -/
  nontriviality R
  /-
    R : Type u
    inst✝ : Ring R
    p q : Polynomial R
    hq : q.Monic
    a✝ : Nontrivial R
    ⊢ LE.le (p.modByMonic q).degree q.degree
  -/
  exact (degree_modByMonic_lt _ hq).le
  /-
    🎉 no goals
  -/


theorem degree_modByMonic_le_left : degree (p %ₘ q) ≤ degree p := by
  /-
    R : Type u
    inst✝ : Ring R
    p q : Polynomial R
    ⊢ LE.le (p.modByMonic q).degree p.degree
  -/
  nontriviality R
  /-
    R : Type u
    inst✝ : Ring R
    p q : Polynomial R
    a✝ : Nontrivial R
    ⊢ LE.le (p.modByMonic q).degree p.degree
  -/
  by_cases hq : q.Monic
    /-
      case pos
      R : Type u
      inst✝ : Ring R
      p q : Polynomial R
      a✝ : Nontrivial R
      hq : q.Monic
      ⊢ LE.le (p.modByMonic q).degree p.degree
    -/
  · cases lt_or_ge (degree p) (degree q)
      /-
        case pos.inl
        R : Type u
        inst✝ : Ring R
        p q : Polynomial R
        a✝ : Nontrivial R
        hq : q.Monic
        h✝ : LT.lt p.degree q.degree
        ⊢ LE.le (p.modByMonic q).degree p.degree
      -/
    · rw [(modByMonic_eq_self_iff hq).mpr ‹_›]
      /-
        🎉 no goals
      -/
      /-
        case pos.inr
        R : Type u
        inst✝ : Ring R
        p q : Polynomial R
        a✝ : Nontrivial R
        hq : q.Monic
        h✝ : GE.ge p.degree q.degree
        ⊢ LE.le (p.modByMonic q).degree p.degree
      -/
    · exact (degree_modByMonic_le p hq).trans ‹_›
      /-
        🎉 no goals
      -/
    /-
      case neg
      R : Type u
      inst✝ : Ring R
      p q : Polynomial R
      a✝ : Nontrivial R
      hq : Not q.Monic
      ⊢ LE.le (p.modByMonic q).degree p.degree
    -/
  · rw [modByMonic_eq_of_not_monic p hq]
    /-
      🎉 no goals
    -/


theorem natDegree_modByMonic_le (p : Polynomial R) {g : Polynomial R} (hg : g.Monic) :
    natDegree (p %ₘ g) ≤ g.natDegree :=
  natDegree_le_natDegree (degree_modByMonic_le p hg)


theorem natDegree_modByMonic_le_left : natDegree (p %ₘ q) ≤ natDegree p :=
  natDegree_le_natDegree degree_modByMonic_le_left


theorem X_dvd_sub_C : X ∣ p - C (p.coeff 0) := by
  /-
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    ⊢ Dvd.dvd Polynomial.X (HSub.hSub p (Polynomial.C (p.coeff 0)))
  -/
  simp [X_dvd_iff, coeff_C]
  /-
    🎉 no goals
  -/


theorem modByMonic_eq_sub_mul_div :
    ∀ (p : R[X]) {q : R[X]} (_hq : Monic q), p %ₘ q = p - q * (p /ₘ q)
  | p, q, hq =>
    letI := Classical.decEq R
    if h : degree q ≤ degree p ∧ p ≠ 0 then by
      /-
        R : Type u
        inst✝ : Ring R
        p q : Polynomial R
        hq : q.Monic
        this : DecidableEq R := Classical.decEq R
        h : And (LE.le q.degree p.degree) (Ne p 0)
        ⊢ Eq (p.modByMonic q) (HSub.hSub p (HMul.hMul q (p.divByMonic q)))
      -/
      have _wf := div_wf_lemma h hq
      have ih := modByMonic_eq_sub_mul_div
        (p - q * (C (leadingCoeff p) * X ^ (natDegree p - natDegree q))) hq
      /-
        R : Type u
        inst✝ : Ring R
        p q : Polynomial R
        hq : q.Monic
        this : DecidableEq R := Classical.decEq R
        h : And (LE.le q.degree p.degree) (Ne p 0)
        _wf : LT.lt (HSub.hSub p (HMul.hMul q (HMul.hMul (Polynomial.C p.leadingCoeff) …
        ih : Eq ((HSub.hSub p (HMul.hMul q (HMul.hMul (Polynomial.C p.leadingCoeff) (H …
        ⊢ Eq (p.modByMonic q) (HSub.hSub p (HMul.hMul q (p.divByMonic q)))
      -/
      unfold modByMonic divByMonic divModByMonicAux
      /-
        R : Type u
        inst✝ : Ring R
        p q : Polynomial R
        hq : q.Monic
        this : DecidableEq R := Classical.decEq R
        h : And (LE.le q.degree p.degree) (Ne p 0)
        _wf : LT.lt (HSub.hSub p (HMul.hMul q (HMul.hMul (Polynomial.C p.leadingCoeff) …
        ih : Eq ((HSub.hSub p (HMul.hMul q (HMul.hMul (Polynomial.C p.leadingCoeff) (H …
        ⊢ Eq
            (dite q.Monic
              (fun h =>
                (dite (And (LE.le q.degree p.degree) (Ne p 0))
                    (fun h_1 =>
                      let z := HMul.hMul (Polynomial.C p.leadingCoeff) (HPow.hPow Poly …
                      letFun ⋯ fun _wf =>
                        let dm := (HSub.hSub p (HMul.hMul q z)).divModByMonicAux ⋯;
                        { fst := HAdd.hAdd z dm.1, snd := dm.2 })
                    fun h => { fst := 0, snd := p }).2)
              fun h => p)
            (HSub.hSub p
              (HMul.hMul q
                (dite q.Monic
                  (fun h =>
                    (dite (And (LE.le q.degree p.degree) (Ne p 0))
                        (fun h_1 =>
                          let z := HMul.hMul (Polynomial.C p.leadingCoeff) (HPow.hPow  …
                          letFun ⋯ fun _wf =>
                            let dm := (HSub.hSub p (HMul.hMul q z)).divModByMonicAux ⋯;
                            { fst := HAdd.hAdd z dm.1, snd := dm.2 })
                        fun h => { fst := 0, snd := p }).1)
                  fun h => 0)))
      -/
      dsimp
      /-
        R : Type u
        inst✝ : Ring R
        p q : Polynomial R
        hq : q.Monic
        this : DecidableEq R := Classical.decEq R
        h : And (LE.le q.degree p.degree) (Ne p 0)
        _wf : LT.lt (HSub.hSub p (HMul.hMul q (HMul.hMul (Polynomial.C p.leadingCoeff) …
        ih : Eq ((HSub.hSub p (HMul.hMul q (HMul.hMul (Polynomial.C p.leadingCoeff) (H …
        ⊢ Eq (dite q.Monic (fun h => (ite (And (LE.le q.degree p.degree) (Not (Eq p 0) …
      -/
      rw [dif_pos hq, if_pos h]
      /-
        R : Type u
        inst✝ : Ring R
        p q : Polynomial R
        hq : q.Monic
        this : DecidableEq R := Classical.decEq R
        h : And (LE.le q.degree p.degree) (Ne p 0)
        _wf : LT.lt (HSub.hSub p (HMul.hMul q (HMul.hMul (Polynomial.C p.leadingCoeff) …
        ih : Eq ((HSub.hSub p (HMul.hMul q (HMul.hMul (Polynomial.C p.leadingCoeff) (H …
        ⊢ Eq { fst := HAdd.hAdd (HMul.hMul (Polynomial.C p.leadingCoeff) (HPow.hPow Po …
      -/
      rw [modByMonic, dif_pos hq] at ih
      /-
        R : Type u
        inst✝ : Ring R
        p q : Polynomial R
        hq : q.Monic
        this : DecidableEq R := Classical.decEq R
        h : And (LE.le q.degree p.degree) (Ne p 0)
        _wf : LT.lt (HSub.hSub p (HMul.hMul q (HMul.hMul (Polynomial.C p.leadingCoeff) …
        ih : Eq ((HSub.hSub p (HMul.hMul q (HMul.hMul (Polynomial.C p.leadingCoeff) (H …
        ⊢ Eq { fst := HAdd.hAdd (HMul.hMul (Polynomial.C p.leadingCoeff) (HPow.hPow Po …
      -/
      refine ih.trans ?_
      /-
        R : Type u
        inst✝ : Ring R
        p q : Polynomial R
        hq : q.Monic
        this : DecidableEq R := Classical.decEq R
        h : And (LE.le q.degree p.degree) (Ne p 0)
        _wf : LT.lt (HSub.hSub p (HMul.hMul q (HMul.hMul (Polynomial.C p.leadingCoeff) …
        ih : Eq ((HSub.hSub p (HMul.hMul q (HMul.hMul (Polynomial.C p.leadingCoeff) (H …
        ⊢ Eq (HSub.hSub (HSub.hSub p (HMul.hMul q (HMul.hMul (Polynomial.C p.leadingCo …
      -/
      unfold divByMonic
      /-
        R : Type u
        inst✝ : Ring R
        p q : Polynomial R
        hq : q.Monic
        this : DecidableEq R := Classical.decEq R
        h : And (LE.le q.degree p.degree) (Ne p 0)
        _wf : LT.lt (HSub.hSub p (HMul.hMul q (HMul.hMul (Polynomial.C p.leadingCoeff) …
        ih : Eq ((HSub.hSub p (HMul.hMul q (HMul.hMul (Polynomial.C p.leadingCoeff) (H …
        ⊢ Eq (HSub.hSub (HSub.hSub p (HMul.hMul q (HMul.hMul (Polynomial.C p.leadingCo …
      -/
      rw [dif_pos hq, dif_pos hq, if_pos h, mul_add, sub_add_eq_sub_sub]
      /-
        🎉 no goals
      -/
    else by
      /-
        R : Type u
        inst✝ : Ring R
        p q : Polynomial R
        hq : q.Monic
        this : DecidableEq R := Classical.decEq R
        h : Not (And (LE.le q.degree p.degree) (Ne p 0))
        ⊢ Eq (p.modByMonic q) (HSub.hSub p (HMul.hMul q (p.divByMonic q)))
      -/
      unfold modByMonic divByMonic divModByMonicAux
      /-
        R : Type u
        inst✝ : Ring R
        p q : Polynomial R
        hq : q.Monic
        this : DecidableEq R := Classical.decEq R
        h : Not (And (LE.le q.degree p.degree) (Ne p 0))
        ⊢ Eq
            (dite q.Monic
              (fun h =>
                (dite (And (LE.le q.degree p.degree) (Ne p 0))
                    (fun h_1 =>
                      let z := HMul.hMul (Polynomial.C p.leadingCoeff) (HPow.hPow Poly …
                      letFun ⋯ fun _wf =>
                        let dm := (HSub.hSub p (HMul.hMul q z)).divModByMonicAux ⋯;
                        { fst := HAdd.hAdd z dm.1, snd := dm.2 })
                    fun h => { fst := 0, snd := p }).2)
              fun h => p)
            (HSub.hSub p
              (HMul.hMul q
                (dite q.Monic
                  (fun h =>
                    (dite (And (LE.le q.degree p.degree) (Ne p 0))
                        (fun h_1 =>
                          let z := HMul.hMul (Polynomial.C p.leadingCoeff) (HPow.hPow  …
                          letFun ⋯ fun _wf =>
                            let dm := (HSub.hSub p (HMul.hMul q z)).divModByMonicAux ⋯;
                            { fst := HAdd.hAdd z dm.1, snd := dm.2 })
                        fun h => { fst := 0, snd := p }).1)
                  fun h => 0)))
      -/
      dsimp
      /-
        R : Type u
        inst✝ : Ring R
        p q : Polynomial R
        hq : q.Monic
        this : DecidableEq R := Classical.decEq R
        h : Not (And (LE.le q.degree p.degree) (Ne p 0))
        ⊢ Eq (dite q.Monic (fun h => (ite (And (LE.le q.degree p.degree) (Not (Eq p 0) …
      -/
      rw [dif_pos hq, if_neg h, dif_pos hq, if_neg h, mul_zero, sub_zero]
      /-
        🎉 no goals
      -/
  termination_by p => p


theorem modByMonic_add_div (p : R[X]) {q : R[X]} (hq : Monic q) : p %ₘ q + q * (p /ₘ q) = p :=
  eq_sub_iff_add_eq.1 (modByMonic_eq_sub_mul_div p hq)


theorem divByMonic_eq_zero_iff [Nontrivial R] (hq : Monic q) : p /ₘ q = 0 ↔ degree p < degree q :=
  ⟨fun h => by
    /-
      R : Type u
      inst✝¹ : Ring R
      p q : Polynomial R
      inst✝ : Nontrivial R
      hq : q.Monic
      h : Eq (p.divByMonic q) 0
      ⊢ LT.lt p.degree q.degree
    -/
    have := modByMonic_add_div p hq
    /-
      R : Type u
      inst✝¹ : Ring R
      p q : Polynomial R
      inst✝ : Nontrivial R
      hq : q.Monic
      h : Eq (p.divByMonic q) 0
      this : Eq (HAdd.hAdd (p.modByMonic q) (HMul.hMul q (p.divByMonic q))) p
      ⊢ LT.lt p.degree q.degree
    -/
    rwa [h, mul_zero, add_zero, modByMonic_eq_self_iff hq] at this,
    /-
      🎉 no goals
    -/
  fun h => by
    classical
    have : ¬degree q ≤ degree p := not_le_of_gt h
    unfold divByMonic divModByMonicAux; dsimp; rw [dif_pos hq, if_neg (mt And.left this)]⟩


theorem degree_add_divByMonic (hq : Monic q) (h : degree q ≤ degree p) :
    degree q + degree (p /ₘ q) = degree p := by
  /-
    R : Type u
    inst✝ : Ring R
    p q : Polynomial R
    hq : q.Monic
    h : LE.le q.degree p.degree
    ⊢ Eq (HAdd.hAdd q.degree (p.divByMonic q).degree) p.degree
  -/
  nontriviality R
  /-
    R : Type u
    inst✝ : Ring R
    p q : Polynomial R
    hq : q.Monic
    h : LE.le q.degree p.degree
    a✝ : Nontrivial R
    ⊢ Eq (HAdd.hAdd q.degree (p.divByMonic q).degree) p.degree
  -/
  have hdiv0 : p /ₘ q ≠ 0 := by rwa [Ne, divByMonic_eq_zero_iff hq, not_lt]
  have hlc : leadingCoeff q * leadingCoeff (p /ₘ q) ≠ 0 := by
    rwa [Monic.def.1 hq, one_mul, Ne, leadingCoeff_eq_zero]
  have hmod : degree (p %ₘ q) < degree (q * (p /ₘ q)) :=
    calc
      degree (p %ₘ q) < degree q := degree_modByMonic_lt _ hq
      _ ≤ _ := by
        rw [degree_mul' hlc, degree_eq_natDegree hq.ne_zero, degree_eq_natDegree hdiv0, ←
            Nat.cast_add, Nat.cast_le]
        exact Nat.le_add_right _ _
  calc
    degree q + degree (p /ₘ q) = degree (q * (p /ₘ q)) := Eq.symm (degree_mul' hlc)
    _ = degree (p %ₘ q + q * (p /ₘ q)) := (degree_add_eq_right_of_degree_lt hmod).symm
    _ = _ := congr_arg _ (modByMonic_add_div _ hq)


theorem degree_divByMonic_le (p q : R[X]) : degree (p /ₘ q) ≤ degree p :=
  letI := Classical.decEq R
                         /-
                           R : Type u
                           inst✝ : Ring R
                           p q : Polynomial R
                           this : DecidableEq R := Classical.decEq R
                           hp0 : Eq p 0
                           ⊢ LE.le (p.divByMonic q).degree p.degree
                         -/
  if hp0 : p = 0 then by simp only [hp0, zero_divByMonic, le_refl]
                         /-
                           🎉 no goals
                         -/
  else
    if hq : Monic q then
      if h : degree q ≤ degree p then by
        /-
          R : Type u
          inst✝ : Ring R
          p q : Polynomial R
          this : DecidableEq R := Classical.decEq R
          hp0 : Not (Eq p 0)
          hq : q.Monic
          h : LE.le q.degree p.degree
          ⊢ LE.le (p.divByMonic q).degree p.degree
        -/
        haveI := Nontrivial.of_polynomial_ne hp0
        rw [← degree_add_divByMonic hq h, degree_eq_natDegree hq.ne_zero,
          degree_eq_natDegree (mt (divByMonic_eq_zero_iff hq).1 (not_lt.2 h))]
        /-
          R : Type u
          inst✝ : Ring R
          p q : Polynomial R
          this✝ : DecidableEq R := Classical.decEq R
          hp0 : Not (Eq p 0)
          hq : q.Monic
          h : LE.le q.degree p.degree
          this : Nontrivial R
          ⊢ LE.le (↑(p.divByMonic q).natDegree) (HAdd.hAdd ↑q.natDegree ↑(p.divByMonic q …
        -/
        exact WithBot.coe_le_coe.2 (Nat.le_add_left _ _)
        /-
          🎉 no goals
        -/
      else by
        /-
          R : Type u
          inst✝ : Ring R
          p q : Polynomial R
          this : DecidableEq R := Classical.decEq R
          hp0 : Not (Eq p 0)
          hq : q.Monic
          h : Not (LE.le q.degree p.degree)
          ⊢ LE.le (p.divByMonic q).degree p.degree
        -/
        unfold divByMonic divModByMonicAux
        /-
          R : Type u
          inst✝ : Ring R
          p q : Polynomial R
          this : DecidableEq R := Classical.decEq R
          hp0 : Not (Eq p 0)
          hq : q.Monic
          h : Not (LE.le q.degree p.degree)
          ⊢ LE.le
              (dite q.Monic
                  (fun h =>
                    (dite (And (LE.le q.degree p.degree) (Ne p 0))
                        (fun h_1 =>
                          let z := HMul.hMul (Polynomial.C p.leadingCoeff) (HPow.hPow Po …
                          letFun ⋯ fun _wf =>
                            let dm := (HSub.hSub p (HMul.hMul q z)).divModByMonicAux ⋯;
                            { fst := HAdd.hAdd z dm.1, snd := dm.2 })
                        fun h => { fst := 0, snd := p }).1)
                  fun h => 0).degree
              p.degree
        -/
        simp [dif_pos hq, h, if_false, degree_zero, bot_le]
        /-
          🎉 no goals
        -/
    else (divByMonic_eq_of_not_monic p hq).symm ▸ bot_le


theorem degree_divByMonic_lt (p : R[X]) {q : R[X]} (hq : Monic q) (hp0 : p ≠ 0)
    (h0q : 0 < degree q) : degree (p /ₘ q) < degree p :=
  if hpq : degree p < degree q then by
    /-
      R : Type u
      inst✝ : Ring R
      p q : Polynomial R
      hq : q.Monic
      hp0 : Ne p 0
      h0q : LT.lt 0 q.degree
      hpq : LT.lt p.degree q.degree
      ⊢ LT.lt (p.divByMonic q).degree p.degree
    -/
    haveI := Nontrivial.of_polynomial_ne hp0
    /-
      R : Type u
      inst✝ : Ring R
      p q : Polynomial R
      hq : q.Monic
      hp0 : Ne p 0
      h0q : LT.lt 0 q.degree
      hpq : LT.lt p.degree q.degree
      this : Nontrivial R
      ⊢ LT.lt (p.divByMonic q).degree p.degree
    -/
    rw [(divByMonic_eq_zero_iff hq).2 hpq, degree_eq_natDegree hp0]
    /-
      R : Type u
      inst✝ : Ring R
      p q : Polynomial R
      hq : q.Monic
      hp0 : Ne p 0
      h0q : LT.lt 0 q.degree
      hpq : LT.lt p.degree q.degree
      this : Nontrivial R
      ⊢ LT.lt (Polynomial.degree 0) ↑p.natDegree
    -/
    exact WithBot.bot_lt_coe _
    /-
      🎉 no goals
    -/
  else by
    /-
      R : Type u
      inst✝ : Ring R
      p q : Polynomial R
      hq : q.Monic
      hp0 : Ne p 0
      h0q : LT.lt 0 q.degree
      hpq : Not (LT.lt p.degree q.degree)
      ⊢ LT.lt (p.divByMonic q).degree p.degree
    -/
    haveI := Nontrivial.of_polynomial_ne hp0
    rw [← degree_add_divByMonic hq (not_lt.1 hpq), degree_eq_natDegree hq.ne_zero,
      degree_eq_natDegree (mt (divByMonic_eq_zero_iff hq).1 hpq)]
    exact
      Nat.cast_lt.2
        (Nat.lt_add_of_pos_left (Nat.cast_lt.1 <|
          by simpa [degree_eq_natDegree hq.ne_zero] using h0q))


theorem natDegree_divByMonic (f : R[X]) {g : R[X]} (hg : g.Monic) :
    natDegree (f /ₘ g) = natDegree f - natDegree g := by
  /-
    R : Type u
    inst✝ : Ring R
    f g : Polynomial R
    hg : g.Monic
    ⊢ Eq (f.divByMonic g).natDegree (HSub.hSub f.natDegree g.natDegree)
  -/
  nontriviality R
  /-
    R : Type u
    inst✝ : Ring R
    f g : Polynomial R
    hg : g.Monic
    a✝ : Nontrivial R
    ⊢ Eq (f.divByMonic g).natDegree (HSub.hSub f.natDegree g.natDegree)
  -/
  by_cases hfg : f /ₘ g = 0
    /-
      case pos
      R : Type u
      inst✝ : Ring R
      f g : Polynomial R
      hg : g.Monic
      a✝ : Nontrivial R
      hfg : Eq (f.divByMonic g) 0
      ⊢ Eq (f.divByMonic g).natDegree (HSub.hSub f.natDegree g.natDegree)
    -/
  · rw [hfg, natDegree_zero]
    /-
      case pos
      R : Type u
      inst✝ : Ring R
      f g : Polynomial R
      hg : g.Monic
      a✝ : Nontrivial R
      hfg : Eq (f.divByMonic g) 0
      ⊢ Eq 0 (HSub.hSub f.natDegree g.natDegree)
    -/
    rw [divByMonic_eq_zero_iff hg] at hfg
    /-
      case pos
      R : Type u
      inst✝ : Ring R
      f g : Polynomial R
      hg : g.Monic
      a✝ : Nontrivial R
      hfg : LT.lt f.degree g.degree
      ⊢ Eq 0 (HSub.hSub f.natDegree g.natDegree)
    -/
    rw [tsub_eq_zero_iff_le.mpr (natDegree_le_natDegree <| le_of_lt hfg)]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u
    inst✝ : Ring R
    f g : Polynomial R
    hg : g.Monic
    a✝ : Nontrivial R
    hfg : Not (Eq (f.divByMonic g) 0)
    ⊢ Eq (f.divByMonic g).natDegree (HSub.hSub f.natDegree g.natDegree)
  -/
  have hgf := hfg
  /-
    case neg
    R : Type u
    inst✝ : Ring R
    f g : Polynomial R
    hg : g.Monic
    a✝ : Nontrivial R
    hfg hgf : Not (Eq (f.divByMonic g) 0)
    ⊢ Eq (f.divByMonic g).natDegree (HSub.hSub f.natDegree g.natDegree)
  -/
  rw [divByMonic_eq_zero_iff hg] at hgf
  /-
    case neg
    R : Type u
    inst✝ : Ring R
    f g : Polynomial R
    hg : g.Monic
    a✝ : Nontrivial R
    hfg : Not (Eq (f.divByMonic g) 0)
    hgf : Not (LT.lt f.degree g.degree)
    ⊢ Eq (f.divByMonic g).natDegree (HSub.hSub f.natDegree g.natDegree)
  -/
  push_neg at hgf
  /-
    case neg
    R : Type u
    inst✝ : Ring R
    f g : Polynomial R
    hg : g.Monic
    a✝ : Nontrivial R
    hfg : Not (Eq (f.divByMonic g) 0)
    hgf : LE.le g.degree f.degree
    ⊢ Eq (f.divByMonic g).natDegree (HSub.hSub f.natDegree g.natDegree)
  -/
  have := degree_add_divByMonic hg hgf
  have hf : f ≠ 0 := by
    intro hf
    apply hfg
    rw [hf, zero_divByMonic]
  rw [degree_eq_natDegree hf, degree_eq_natDegree hg.ne_zero, degree_eq_natDegree hfg,
    ← Nat.cast_add, Nat.cast_inj] at this
  /-
    case neg
    R : Type u
    inst✝ : Ring R
    f g : Polynomial R
    hg : g.Monic
    a✝ : Nontrivial R
    hfg : Not (Eq (f.divByMonic g) 0)
    hgf : LE.le g.degree f.degree
    this : Eq (HAdd.hAdd g.natDegree (f.divByMonic g).natDegree) f.natDegree
    hf : Ne f 0
    ⊢ Eq (f.divByMonic g).natDegree (HSub.hSub f.natDegree g.natDegree)
  -/
  rw [← this, add_tsub_cancel_left]
  /-
    🎉 no goals
  -/


theorem div_modByMonic_unique {f g} (q r : R[X]) (hg : Monic g)
    (h : r + g * q = f ∧ degree r < degree g) : f /ₘ g = q ∧ f %ₘ g = r := by
  /-
    R : Type u
    inst✝ : Ring R
    f g q r : Polynomial R
    hg : g.Monic
    h : And (Eq (HAdd.hAdd r (HMul.hMul g q)) f) (LT.lt r.degree g.degree)
    ⊢ And (Eq (f.divByMonic g) q) (Eq (f.modByMonic g) r)
  -/
  nontriviality R
  have h₁ : r - f %ₘ g = -g * (q - f /ₘ g) :=
    eq_of_sub_eq_zero
      (by
        rw [← sub_eq_zero_of_eq (h.1.trans (modByMonic_add_div f hg).symm)]
        simp [mul_add, mul_comm, sub_eq_add_neg, add_comm, add_left_comm, add_assoc])
  /-
    R : Type u
    inst✝ : Ring R
    f g q r : Polynomial R
    hg : g.Monic
    h : And (Eq (HAdd.hAdd r (HMul.hMul g q)) f) (LT.lt r.degree g.degree)
    a✝ : Nontrivial R
    h₁ : Eq (HSub.hSub r (f.modByMonic g)) (HMul.hMul (Neg.neg g) (HSub.hSub q (f. …
    ⊢ And (Eq (f.divByMonic g) q) (Eq (f.modByMonic g) r)
  -/
  have h₂ : degree (r - f %ₘ g) = degree (g * (q - f /ₘ g)) := by simp [h₁]
  have h₄ : degree (r - f %ₘ g) < degree g :=
    calc
      degree (r - f %ₘ g) ≤ max (degree r) (degree (f %ₘ g)) := degree_sub_le _ _
      _ < degree g := max_lt_iff.2 ⟨h.2, degree_modByMonic_lt _ hg⟩
  have h₅ : q - f /ₘ g = 0 :=
    _root_.by_contradiction fun hqf =>
      not_le_of_gt h₄ <|
        calc
          degree g ≤ degree g + degree (q - f /ₘ g) := by
            erw [degree_eq_natDegree hg.ne_zero, degree_eq_natDegree hqf, WithBot.coe_le_coe]
            exact Nat.le_add_right _ _
          _ = degree (r - f %ₘ g) := by rw [h₂, degree_mul']; simpa [Monic.def.1 hg]
  /-
    R : Type u
    inst✝ : Ring R
    f g q r : Polynomial R
    hg : g.Monic
    h : And (Eq (HAdd.hAdd r (HMul.hMul g q)) f) (LT.lt r.degree g.degree)
    a✝ : Nontrivial R
    h₁ : Eq (HSub.hSub r (f.modByMonic g)) (HMul.hMul (Neg.neg g) (HSub.hSub q (f. …
    h₂ : Eq (HSub.hSub r (f.modByMonic g)).degree (HMul.hMul g (HSub.hSub q (f.div …
    h₄ : LT.lt (HSub.hSub r (f.modByMonic g)).degree g.degree
    h₅ : Eq (HSub.hSub q (f.divByMonic g)) 0
    ⊢ And (Eq (f.divByMonic g) q) (Eq (f.modByMonic g) r)
  -/
  exact ⟨Eq.symm <| eq_of_sub_eq_zero h₅, Eq.symm <| eq_of_sub_eq_zero <| by simpa [h₅] using h₁⟩
  /-
    🎉 no goals
  -/


theorem map_mod_divByMonic [Ring S] (f : R →+* S) (hq : Monic q) :
    (p /ₘ q).map f = p.map f /ₘ q.map f ∧ (p %ₘ q).map f = p.map f %ₘ q.map f := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Ring R
    p q : Polynomial R
    inst✝ : Ring S
    f : RingHom R S
    hq : q.Monic
    ⊢ And (Eq (Polynomial.map f (p.divByMonic q)) ((Polynomial.map f p).divByMonic …
  -/
  nontriviality S
  /-
    R : Type u
    S : Type v
    inst✝¹ : Ring R
    p q : Polynomial R
    inst✝ : Ring S
    f : RingHom R S
    hq : q.Monic
    a✝ : Nontrivial S
    ⊢ And (Eq (Polynomial.map f (p.divByMonic q)) ((Polynomial.map f p).divByMonic …
  -/
  haveI : Nontrivial R := f.domain_nontrivial
  have : map f p /ₘ map f q = map f (p /ₘ q) ∧ map f p %ₘ map f q = map f (p %ₘ q) :=
    div_modByMonic_unique ((p /ₘ q).map f) _ (hq.map f)
      ⟨Eq.symm <| by rw [← Polynomial.map_mul, ← Polynomial.map_add, modByMonic_add_div _ hq],
        calc
          _ ≤ degree (p %ₘ q) := degree_map_le
          _ < degree q := degree_modByMonic_lt _ hq
          _ = _ :=
            Eq.symm <|
              degree_map_eq_of_leadingCoeff_ne_zero _
                (by rw [Monic.def.1 hq, f.map_one]; exact one_ne_zero)⟩
  /-
    R : Type u
    S : Type v
    inst✝¹ : Ring R
    p q : Polynomial R
    inst✝ : Ring S
    f : RingHom R S
    hq : q.Monic
    a✝ : Nontrivial S
    this✝ : Nontrivial R
    this : And (Eq ((Polynomial.map f p).divByMonic (Polynomial.map f q)) (Polynom …
    ⊢ And (Eq (Polynomial.map f (p.divByMonic q)) ((Polynomial.map f p).divByMonic …
  -/
  exact ⟨this.1.symm, this.2.symm⟩
  /-
    🎉 no goals
  -/


theorem map_divByMonic [Ring S] (f : R →+* S) (hq : Monic q) :
    (p /ₘ q).map f = p.map f /ₘ q.map f :=
  (map_mod_divByMonic f hq).1


theorem map_modByMonic [Ring S] (f : R →+* S) (hq : Monic q) :
    (p %ₘ q).map f = p.map f %ₘ q.map f :=
  (map_mod_divByMonic f hq).2


theorem modByMonic_eq_zero_iff_dvd (hq : Monic q) : p %ₘ q = 0 ↔ q ∣ p :=
               /-
                 R : Type u
                 inst✝ : Ring R
                 p q : Polynomial R
                 hq : q.Monic
                 h : Eq (p.modByMonic q) 0
                 ⊢ Dvd.dvd q p
               -/
  ⟨fun h => by rw [← modByMonic_add_div p hq, h, zero_add]; exact dvd_mul_right _ _, fun h => by
                                                            /-
                                                              🎉 no goals
                                                            -/
    /-
      R : Type u
      inst✝ : Ring R
      p q : Polynomial R
      hq : q.Monic
      h : Dvd.dvd q p
      ⊢ Eq (p.modByMonic q) 0
    -/
    nontriviality R
    /-
      R : Type u
      inst✝ : Ring R
      p q : Polynomial R
      hq : q.Monic
      h : Dvd.dvd q p
      a✝ : Nontrivial R
      ⊢ Eq (p.modByMonic q) 0
    -/
    obtain ⟨r, hr⟩ := exists_eq_mul_right_of_dvd h
    /-
      case intro
      R : Type u
      inst✝ : Ring R
      p q : Polynomial R
      hq : q.Monic
      h : Dvd.dvd q p
      a✝ : Nontrivial R
      r : Polynomial R
      hr : Eq p (HMul.hMul q r)
      ⊢ Eq (p.modByMonic q) 0
    -/
    by_contra hpq0
    /-
      case intro
      R : Type u
      inst✝ : Ring R
      p q : Polynomial R
      hq : q.Monic
      h : Dvd.dvd q p
      a✝ : Nontrivial R
      r : Polynomial R
      hr : Eq p (HMul.hMul q r)
      hpq0 : Not (Eq (p.modByMonic q) 0)
      ⊢ False
    -/
    have hmod : p %ₘ q = q * (r - p /ₘ q) := by rw [modByMonic_eq_sub_mul_div _ hq, mul_sub, ← hr]
    /-
      case intro
      R : Type u
      inst✝ : Ring R
      p q : Polynomial R
      hq : q.Monic
      h : Dvd.dvd q p
      a✝ : Nontrivial R
      r : Polynomial R
      hr : Eq p (HMul.hMul q r)
      hpq0 : Not (Eq (p.modByMonic q) 0)
      hmod : Eq (p.modByMonic q) (HMul.hMul q (HSub.hSub r (p.divByMonic q)))
      ⊢ False
    -/
    have : degree (q * (r - p /ₘ q)) < degree q := hmod ▸ degree_modByMonic_lt _ hq
    have hrpq0 : leadingCoeff (r - p /ₘ q) ≠ 0 := fun h =>
      hpq0 <|
        leadingCoeff_eq_zero.1
          (by rw [hmod, leadingCoeff_eq_zero.1 h, mul_zero, leadingCoeff_zero])
    /-
      case intro
      R : Type u
      inst✝ : Ring R
      p q : Polynomial R
      hq : q.Monic
      h : Dvd.dvd q p
      a✝ : Nontrivial R
      r : Polynomial R
      hr : Eq p (HMul.hMul q r)
      hpq0 : Not (Eq (p.modByMonic q) 0)
      hmod : Eq (p.modByMonic q) (HMul.hMul q (HSub.hSub r (p.divByMonic q)))
      this : LT.lt (HMul.hMul q (HSub.hSub r (p.divByMonic q))).degree q.degree
      hrpq0 : Ne (HSub.hSub r (p.divByMonic q)).leadingCoeff 0
      ⊢ False
    -/
    have hlc : leadingCoeff q * leadingCoeff (r - p /ₘ q) ≠ 0 := by rwa [Monic.def.1 hq, one_mul]
    rw [degree_mul' hlc, degree_eq_natDegree hq.ne_zero,
      degree_eq_natDegree (mt leadingCoeff_eq_zero.2 hrpq0)] at this
    /-
      case intro
      R : Type u
      inst✝ : Ring R
      p q : Polynomial R
      hq : q.Monic
      h : Dvd.dvd q p
      a✝ : Nontrivial R
      r : Polynomial R
      hr : Eq p (HMul.hMul q r)
      hpq0 : Not (Eq (p.modByMonic q) 0)
      hmod : Eq (p.modByMonic q) (HMul.hMul q (HSub.hSub r (p.divByMonic q)))
      this : LT.lt (HAdd.hAdd ↑q.natDegree ↑(HSub.hSub r (p.divByMonic q)).natDegree …
      hrpq0 : Ne (HSub.hSub r (p.divByMonic q)).leadingCoeff 0
      hlc : Ne (HMul.hMul q.leadingCoeff (HSub.hSub r (p.divByMonic q)).leadingCoeff …
      ⊢ False
    -/
    exact not_lt_of_ge (Nat.le_add_right _ _) (WithBot.coe_lt_coe.1 this)⟩
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-03-23")] alias dvd_iff_modByMonic_eq_zero := modByMonic_eq_zero_iff_dvd


/-- See `Polynomial.mul_left_modByMonic` for the other multiplication order. That version, unlike
this one, requires commutativity. -/
@[simp]
lemma self_mul_modByMonic (hq : q.Monic) : (q * p) %ₘ q = 0 := by
  /-
    R : Type u
    inst✝ : Ring R
    p q : Polynomial R
    hq : q.Monic
    ⊢ Eq ((HMul.hMul q p).modByMonic q) 0
  -/
  rw [modByMonic_eq_zero_iff_dvd hq]
  /-
    R : Type u
    inst✝ : Ring R
    p q : Polynomial R
    hq : q.Monic
    ⊢ Dvd.dvd q (HMul.hMul q p)
  -/
  exact dvd_mul_right q p
  /-
    🎉 no goals
  -/


theorem map_dvd_map [Ring S] (f : R →+* S) (hf : Function.Injective f) {x y : R[X]}
    (hx : x.Monic) : x.map f ∣ y.map f ↔ x ∣ y := by
  rw [← modByMonic_eq_zero_iff_dvd hx, ← modByMonic_eq_zero_iff_dvd (hx.map f), ←
    map_modByMonic f hx]
  exact
    ⟨fun H => map_injective f hf <| by rw [H, Polynomial.map_zero], fun H => by
      rw [H, Polynomial.map_zero]⟩


@[simp]
theorem modByMonic_one (p : R[X]) : p %ₘ 1 = 0 :=
                                  /-
                                    R : Type u
                                    inst✝ : Ring R
                                    p : Polynomial R
                                    ⊢ Polynomial.Monic 1
                                  -/
  (modByMonic_eq_zero_iff_dvd (by convert monic_one (R := R))).2 (one_dvd _)
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
theorem divByMonic_one (p : R[X]) : p /ₘ 1 = p := by
  /-
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    ⊢ Eq (p.divByMonic 1) p
  -/
  conv_rhs => rw [← modByMonic_add_div p monic_one]; simp
  /-
    🎉 no goals
  -/


theorem sum_modByMonic_coeff (hq : q.Monic) {n : ℕ} (hn : q.degree ≤ n) :
    (∑ i : Fin n, monomial i ((p %ₘ q).coeff i)) = p %ₘ q := by
  /-
    R : Type u
    inst✝ : Ring R
    p q : Polynomial R
    hq : q.Monic
    n : Nat
    hn : LE.le q.degree ↑n
    ⊢ Eq (Finset.univ.sum fun i => (Polynomial.monomial ↑i) ((p.modByMonic q).coef …
  -/
  nontriviality R
  exact
    (sum_fin (fun i c => monomial i c) (by simp) ((degree_modByMonic_lt _ hq).trans_le hn)).trans
      (sum_monomial_eq _)


theorem mul_divByMonic_cancel_left (p : R[X]) {q : R[X]} (hmo : q.Monic) :
    q * p /ₘ q = p := by
  /-
    R : Type u
    inst✝ : Ring R
    p q : Polynomial R
    hmo : q.Monic
    ⊢ Eq ((HMul.hMul q p).divByMonic q) p
  -/
  nontriviality R
  /-
    R : Type u
    inst✝ : Ring R
    p q : Polynomial R
    hmo : q.Monic
    a✝ : Nontrivial R
    ⊢ Eq ((HMul.hMul q p).divByMonic q) p
  -/
  refine (div_modByMonic_unique _ 0 hmo ⟨by rw [zero_add], ?_⟩).1
  /-
    R : Type u
    inst✝ : Ring R
    p q : Polynomial R
    hmo : q.Monic
    a✝ : Nontrivial R
    ⊢ LT.lt (Polynomial.degree 0) q.degree
  -/
  rw [degree_zero]
  /-
    R : Type u
    inst✝ : Ring R
    p q : Polynomial R
    hmo : q.Monic
    a✝ : Nontrivial R
    ⊢ LT.lt Bot.bot q.degree
  -/
  exact Ne.bot_lt fun h => hmo.ne_zero (degree_eq_bot.1 h)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-06-30")]
alias mul_div_mod_by_monic_cancel_left := mul_divByMonic_cancel_left


lemma coeff_divByMonic_X_sub_C_rec (p : R[X]) (a : R) (n : ℕ) :
    (p /ₘ (X - C a)).coeff n = coeff p (n + 1) + a * (p /ₘ (X - C a)).coeff (n + 1) := by
  /-
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    a : R
    n : Nat
    ⊢ Eq ((p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C a))).coeff n) (HAdd. …
  -/
  nontriviality R
  /-
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    a : R
    n : Nat
    a✝ : Nontrivial R
    ⊢ Eq ((p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C a))).coeff n) (HAdd. …
  -/
  have := monic_X_sub_C a
  /-
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    a : R
    n : Nat
    a✝ : Nontrivial R
    this : (HSub.hSub Polynomial.X (Polynomial.C a)).Monic
    ⊢ Eq ((p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C a))).coeff n) (HAdd. …
  -/
  set q := p /ₘ (X - C a)
  /-
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    a : R
    n : Nat
    a✝ : Nontrivial R
    this : (HSub.hSub Polynomial.X (Polynomial.C a)).Monic
    q : Polynomial R := p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C a))
    ⊢ Eq (q.coeff n) (HAdd.hAdd (p.coeff (HAdd.hAdd n 1)) (HMul.hMul a (q.coeff (H …
  -/
  rw [← p.modByMonic_add_div this]
  have : degree (p %ₘ (X - C a)) < ↑(n + 1) := degree_X_sub_C a ▸ p.degree_modByMonic_lt this
    |>.trans_le <| WithBot.coe_le_coe.mpr le_add_self
  /-
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    a : R
    n : Nat
    a✝ : Nontrivial R
    this✝ : (HSub.hSub Polynomial.X (Polynomial.C a)).Monic
    q : Polynomial R := p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C a))
    this : LT.lt (p.modByMonic (HSub.hSub Polynomial.X (Polynomial.C a))).degree ↑ …
    ⊢ Eq (q.coeff n) (HAdd.hAdd ((HAdd.hAdd (p.modByMonic (HSub.hSub Polynomial.X  …
  -/
  simp [q, sub_mul, add_sub, coeff_eq_zero_of_degree_lt this]
  /-
    🎉 no goals
  -/


theorem coeff_divByMonic_X_sub_C (p : R[X]) (a : R) (n : ℕ) :
    (p /ₘ (X - C a)).coeff n = ∑ i ∈ Icc (n + 1) p.natDegree, a ^ (i - (n + 1)) * p.coeff i := by
  /-
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    a : R
    n : Nat
    ⊢ Eq ((p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C a))).coeff n) ((Fins …
  -/
  wlog h : p.natDegree ≤ n generalizing n
    /-
      case inr
      R : Type u
      inst✝ : Ring R
      p : Polynomial R
      a : R
      n : Nat
      this : ∀ (n : Nat), LE.le p.natDegree n → Eq ((p.divByMonic (HSub.hSub Polynom …
      h : Not (LE.le p.natDegree n)
      ⊢ Eq ((p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C a))).coeff n) ((Fins …
    -/
  · refine Nat.decreasingInduction' (fun n hn _ ih ↦ ?_) (le_of_not_le h) ?_
    · rw [coeff_divByMonic_X_sub_C_rec, ih, eq_comm, Icc_eq_cons_Ioc (Nat.succ_le.mpr hn),
          sum_cons, Nat.sub_self, pow_zero, one_mul, mul_sum]
      /-
        case inr.refine_1
        R : Type u
        inst✝ : Ring R
        p : Polynomial R
        a : R
        n✝ : Nat
        this : ∀ (n : Nat), LE.le p.natDegree n → Eq ((p.divByMonic (HSub.hSub Polynom …
        h : Not (LE.le p.natDegree n✝)
        n : Nat
        hn : LT.lt n p.natDegree
        x✝ : LE.le n✝ n
        ih : Eq ((p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C a))).coeff (HAdd. …
        ⊢ Eq (HAdd.hAdd (p.coeff n.succ) ((Finset.Ioc n.succ p.natDegree).sum fun x => …
      -/
      congr 1; refine sum_congr ?_ fun i hi ↦ ?_
        /-
          case inr.refine_1.e_a.refine_1
          R : Type u
          inst✝ : Ring R
          p : Polynomial R
          a : R
          n✝ : Nat
          this : ∀ (n : Nat), LE.le p.natDegree n → Eq ((p.divByMonic (HSub.hSub Polynom …
          h : Not (LE.le p.natDegree n✝)
          n : Nat
          hn : LT.lt n p.natDegree
          x✝ : LE.le n✝ n
          ih : Eq ((p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C a))).coeff (HAdd. …
          ⊢ Eq (Finset.Ioc n.succ p.natDegree) (Finset.Icc (HAdd.hAdd (HAdd.hAdd n 1) 1) …
        -/
      · ext; simp [Nat.succ_le]
             /-
               🎉 no goals
             -/
      /-
        case inr.refine_1.e_a.refine_2
        R : Type u
        inst✝ : Ring R
        p : Polynomial R
        a : R
        n✝ : Nat
        this : ∀ (n : Nat), LE.le p.natDegree n → Eq ((p.divByMonic (HSub.hSub Polynom …
        h : Not (LE.le p.natDegree n✝)
        n : Nat
        hn : LT.lt n p.natDegree
        x✝ : LE.le n✝ n
        ih : Eq ((p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C a))).coeff (HAdd. …
        i : Nat
        hi : Membership.mem (Finset.Icc (HAdd.hAdd (HAdd.hAdd n 1) 1) p.natDegree) i
        ⊢ Eq (HMul.hMul (HPow.hPow a (HSub.hSub i (HAdd.hAdd n 1))) (p.coeff i)) (HMul …
      -/
      rw [← mul_assoc, ← pow_succ', eq_comm, i.sub_succ', Nat.sub_add_cancel]
      /-
        case inr.refine_1.e_a.refine_2
        R : Type u
        inst✝ : Ring R
        p : Polynomial R
        a : R
        n✝ : Nat
        this : ∀ (n : Nat), LE.le p.natDegree n → Eq ((p.divByMonic (HSub.hSub Polynom …
        h : Not (LE.le p.natDegree n✝)
        n : Nat
        hn : LT.lt n p.natDegree
        x✝ : LE.le n✝ n
        ih : Eq ((p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C a))).coeff (HAdd. …
        i : Nat
        hi : Membership.mem (Finset.Icc (HAdd.hAdd (HAdd.hAdd n 1) 1) p.natDegree) i
        ⊢ LE.le 1 (HSub.hSub i (HAdd.hAdd n 1))
      -/
      apply Nat.le_sub_of_add_le
      /-
        case inr.refine_1.e_a.refine_2.h
        R : Type u
        inst✝ : Ring R
        p : Polynomial R
        a : R
        n✝ : Nat
        this : ∀ (n : Nat), LE.le p.natDegree n → Eq ((p.divByMonic (HSub.hSub Polynom …
        h : Not (LE.le p.natDegree n✝)
        n : Nat
        hn : LT.lt n p.natDegree
        x✝ : LE.le n✝ n
        ih : Eq ((p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C a))).coeff (HAdd. …
        i : Nat
        hi : Membership.mem (Finset.Icc (HAdd.hAdd (HAdd.hAdd n 1) 1) p.natDegree) i
        ⊢ LE.le (HAdd.hAdd 1 (HAdd.hAdd n 1)) i
      -/
      rw [add_comm]; exact (mem_Icc.mp hi).1
                     /-
                       🎉 no goals
                     -/
      /-
        case inr.refine_2
        R : Type u
        inst✝ : Ring R
        p : Polynomial R
        a : R
        n : Nat
        this : ∀ (n : Nat), LE.le p.natDegree n → Eq ((p.divByMonic (HSub.hSub Polynom …
        h : Not (LE.le p.natDegree n)
        ⊢ Eq ((p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C a))).coeff p.natDegr …
      -/
    · exact this _ le_rfl
      /-
        🎉 no goals
      -/
  /-
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    a : R
    n : Nat
    h : LE.le p.natDegree n
    ⊢ Eq ((p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C a))).coeff n) ((Fins …
  -/
  rw [Icc_eq_empty (Nat.lt_succ.mpr h).not_le, sum_empty]
  /-
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    a : R
    n : Nat
    h : LE.le p.natDegree n
    ⊢ Eq ((p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C a))).coeff n) 0
  -/
  nontriviality R
  /-
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    a : R
    n : Nat
    h : LE.le p.natDegree n
    a✝ : Nontrivial R
    ⊢ Eq ((p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C a))).coeff n) 0
  -/
  by_cases hp : p.natDegree = 0
    /-
      case pos
      R : Type u
      inst✝ : Ring R
      p : Polynomial R
      a : R
      n : Nat
      h : LE.le p.natDegree n
      a✝ : Nontrivial R
      hp : Eq p.natDegree 0
      ⊢ Eq ((p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C a))).coeff n) 0
    -/
  · rw [(divByMonic_eq_zero_iff <| monic_X_sub_C a).mpr, coeff_zero]
    /-
      case pos
      R : Type u
      inst✝ : Ring R
      p : Polynomial R
      a : R
      n : Nat
      h : LE.le p.natDegree n
      a✝ : Nontrivial R
      hp : Eq p.natDegree 0
      ⊢ LT.lt p.degree (HSub.hSub Polynomial.X (Polynomial.C a)).degree
    -/
    apply degree_lt_degree; rw [hp, natDegree_X_sub_C]; norm_num
                                                        /-
                                                          🎉 no goals
                                                        -/
    /-
      case neg
      R : Type u
      inst✝ : Ring R
      p : Polynomial R
      a : R
      n : Nat
      h : LE.le p.natDegree n
      a✝ : Nontrivial R
      hp : Not (Eq p.natDegree 0)
      ⊢ Eq ((p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C a))).coeff n) 0
    -/
  · apply coeff_eq_zero_of_natDegree_lt
    /-
      case neg.h
      R : Type u
      inst✝ : Ring R
      p : Polynomial R
      a : R
      n : Nat
      h : LE.le p.natDegree n
      a✝ : Nontrivial R
      hp : Not (Eq p.natDegree 0)
      ⊢ LT.lt (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C a))).natDegree n
    -/
    rw [natDegree_divByMonic p (monic_X_sub_C a), natDegree_X_sub_C]
    /-
      case neg.h
      R : Type u
      inst✝ : Ring R
      p : Polynomial R
      a : R
      n : Nat
      h : LE.le p.natDegree n
      a✝ : Nontrivial R
      hp : Not (Eq p.natDegree 0)
      ⊢ LT.lt (HSub.hSub p.natDegree 1) n
    -/
    exact (Nat.pred_lt hp).trans_le h
    /-
      🎉 no goals
    -/


variable (R) in
theorem not_isField : ¬IsField R[X] := by
  /-
    R : Type u
    inst✝ : Ring R
    ⊢ Not (IsField (Polynomial R))
  -/
  nontriviality R
  /-
    R : Type u
    inst✝ : Ring R
    a✝ : Nontrivial R
    ⊢ Not (IsField (Polynomial R))
  -/
  intro h
  /-
    R : Type u
    inst✝ : Ring R
    a✝ : Nontrivial R
    h : IsField (Polynomial R)
    ⊢ False
  -/
  letI := h.toField
  /-
    R : Type u
    inst✝ : Ring R
    a✝ : Nontrivial R
    h : IsField (Polynomial R)
    this : Field (Polynomial R) := h.toField
    ⊢ False
  -/
  simpa using congr_arg natDegree (monic_X.eq_one_of_isUnit <| monic_X (R := R).ne_zero.isUnit)
  /-
    🎉 no goals
  -/


/-- An algorithm for deciding polynomial divisibility.
The algorithm is "compute `p %ₘ q` and compare to `0`".
See `polynomial.modByMonic` for the algorithm that computes `%ₘ`.
-/
def decidableDvdMonic [DecidableEq R] (p : R[X]) (hq : Monic q) : Decidable (q ∣ p) :=
  decidable_of_iff (p %ₘ q = 0) (modByMonic_eq_zero_iff_dvd hq)


theorem finiteMultiplicity_X_sub_C (a : R) (h0 : p ≠ 0) : FiniteMultiplicity (X - C a) p := by
  /-
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    a : R
    h0 : Ne p 0
    ⊢ FiniteMultiplicity (HSub.hSub Polynomial.X (Polynomial.C a)) p
  -/
  haveI := Nontrivial.of_polynomial_ne h0
  /-
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    a : R
    h0 : Ne p 0
    this : Nontrivial R
    ⊢ FiniteMultiplicity (HSub.hSub Polynomial.X (Polynomial.C a)) p
  -/
  refine finiteMultiplicity_of_degree_pos_of_monic ?_ (monic_X_sub_C _) h0
  /-
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    a : R
    h0 : Ne p 0
    this : Nontrivial R
    ⊢ LT.lt 0 (HSub.hSub Polynomial.X (Polynomial.C a)).degree
  -/
  rw [degree_X_sub_C]
  /-
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    a : R
    h0 : Ne p 0
    this : Nontrivial R
    ⊢ LT.lt 0 1
  -/
  decide
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-30")]
alias multiplicity_X_sub_C_finite := finiteMultiplicity_X_sub_C

/- Porting note: stripping out classical for decidability instance parameter might
make for better ergonomics -/

/-- The largest power of `X - C a` which divides `p`.
This *could be* computable via the divisibility algorithm `Polynomial.decidableDvdMonic`,
as shown by `Polynomial.rootMultiplicity_eq_nat_find_of_nonzero` which has a computable RHS. -/
def rootMultiplicity (a : R) (p : R[X]) : ℕ :=
  letI := Classical.decEq R
  if h0 : p = 0 then 0
  else
    let _ : DecidablePred fun n : ℕ => ¬(X - C a) ^ (n + 1) ∣ p := fun n =>
      have := decidableDvdMonic p ((monic_X_sub_C a).pow (n + 1))
      inferInstanceAs (Decidable ¬_)
    Nat.find (finiteMultiplicity_X_sub_C a h0)

/- Porting note: added the following due to diamond with decidableProp and
decidableDvdMonic see also [Zulip]
(https://leanprover.zulipchat.com/#narrow/stream/287929-mathlib4/topic/non-defeq.20aliased.20instance) -/

theorem rootMultiplicity_eq_nat_find_of_nonzero [DecidableEq R] {p : R[X]} (p0 : p ≠ 0) {a : R} :
    letI : DecidablePred fun n : ℕ => ¬(X - C a) ^ (n + 1) ∣ p := fun n =>
      have := decidableDvdMonic p ((monic_X_sub_C a).pow (n + 1))
      inferInstanceAs (Decidable ¬_)
    rootMultiplicity a p = Nat.find (finiteMultiplicity_X_sub_C a p0) := by
  /-
    R : Type u
    inst✝¹ : Ring R
    inst✝ : DecidableEq R
    p : Polynomial R
    p0 : Ne p 0
    a : R
    ⊢ Eq (Polynomial.rootMultiplicity a p) (Nat.find ⋯)
  -/
  dsimp [rootMultiplicity]
  /-
    R : Type u
    inst✝¹ : Ring R
    inst✝ : DecidableEq R
    p : Polynomial R
    p0 : Ne p 0
    a : R
    ⊢ Eq (dite (Eq p 0) (fun h0 => 0) fun h0 => Nat.find ⋯) (Nat.find ⋯)
  -/
  cases Subsingleton.elim ‹DecidableEq R› (Classical.decEq R)
  /-
    case refl
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    p0 : Ne p 0
    a : R
    ⊢ Eq (dite (Eq p 0) (fun h0 => 0) fun h0 => Nat.find ⋯) (Nat.find ⋯)
  -/
  rw [dif_neg p0]
  /-
    🎉 no goals
  -/


theorem rootMultiplicity_eq_multiplicity [DecidableEq R]
    (p : R[X]) (a : R) :
    rootMultiplicity a p =
      if p = 0 then 0 else multiplicity (X - C a) p := by
  /-
    R : Type u
    inst✝¹ : Ring R
    inst✝ : DecidableEq R
    p : Polynomial R
    a : R
    ⊢ Eq (Polynomial.rootMultiplicity a p) (ite (Eq p 0) 0 (multiplicity (HSub.hSu …
  -/
  simp only [rootMultiplicity, multiplicity, emultiplicity]
  /-
    R : Type u
    inst✝¹ : Ring R
    inst✝ : DecidableEq R
    p : Polynomial R
    a : R
    ⊢ Eq (dite (Eq p 0) (fun h => 0) fun h => Nat.find ⋯) (ite (Eq p 0) 0 (WithTop …
  -/
  split
    /-
      case isTrue
      R : Type u
      inst✝¹ : Ring R
      inst✝ : DecidableEq R
      p : Polynomial R
      a : R
      h✝ : Eq p 0
      ⊢ Eq 0 0
    -/
  · rfl
    /-
      🎉 no goals
    -/
  /-
    case isFalse
    R : Type u
    inst✝¹ : Ring R
    inst✝ : DecidableEq R
    p : Polynomial R
    a : R
    h✝ : Not (Eq p 0)
    ⊢ Eq (Nat.find ⋯) (WithTop.untop' 1 (dite (FiniteMultiplicity (HSub.hSub Polyn …
  -/
  rename_i h
  /-
    case isFalse
    R : Type u
    inst✝¹ : Ring R
    inst✝ : DecidableEq R
    p : Polynomial R
    a : R
    h : Not (Eq p 0)
    ⊢ Eq (Nat.find ⋯) (WithTop.untop' 1 (dite (FiniteMultiplicity (HSub.hSub Polyn …
  -/
  simp only [finiteMultiplicity_X_sub_C a h, ↓reduceDIte]
  /-
    case isFalse
    R : Type u
    inst✝¹ : Ring R
    inst✝ : DecidableEq R
    p : Polynomial R
    a : R
    h : Not (Eq p 0)
    ⊢ Eq (Nat.find ⋯) (WithTop.untop' 1 ↑(Nat.find ⋯))
  -/
  rw [← ENat.some_eq_coe, WithTop.untop'_coe]
  /-
    case isFalse
    R : Type u
    inst✝¹ : Ring R
    inst✝ : DecidableEq R
    p : Polynomial R
    a : R
    h : Not (Eq p 0)
    ⊢ Eq (Nat.find ⋯) (Nat.find ⋯)
  -/
  congr
  /-
    🎉 no goals
  -/


@[simp]
theorem rootMultiplicity_zero {x : R} : rootMultiplicity x 0 = 0 :=
  dif_pos rfl


@[simp]
theorem rootMultiplicity_C (r a : R) : rootMultiplicity a (C r) = 0 := by
  /-
    R : Type u
    inst✝ : Ring R
    r a : R
    ⊢ Eq (Polynomial.rootMultiplicity a (Polynomial.C r)) 0
  -/
  cases subsingleton_or_nontrivial R
    /-
      case inl
      R : Type u
      inst✝ : Ring R
      r a : R
      h✝ : Subsingleton R
      ⊢ Eq (Polynomial.rootMultiplicity a (Polynomial.C r)) 0
    -/
  · rw [Subsingleton.elim (C r) 0, rootMultiplicity_zero]
    /-
      🎉 no goals
    -/
  classical
  rw [rootMultiplicity_eq_multiplicity]
  split_ifs with hr
  · rfl
  have h : natDegree (C r) < natDegree (X - C a) := by simp
  simp_rw [multiplicity_eq_zero.mpr ((monic_X_sub_C a).not_dvd_of_natDegree_lt hr h)]


theorem pow_rootMultiplicity_dvd (p : R[X]) (a : R) : (X - C a) ^ rootMultiplicity a p ∣ p :=
  letI := Classical.decEq R
                       /-
                         R : Type u
                         inst✝ : Ring R
                         p : Polynomial R
                         a : R
                         this : DecidableEq R := Classical.decEq R
                         h : Eq p 0
                         ⊢ Dvd.dvd (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C a)) (Polynomial.roo …
                       -/
  if h : p = 0 then by simp [h]
                       /-
                         🎉 no goals
                       -/
  else by
    classical
    rw [rootMultiplicity_eq_multiplicity, if_neg h]; apply pow_multiplicity_dvd


theorem pow_mul_divByMonic_rootMultiplicity_eq (p : R[X]) (a : R) :
    (X - C a) ^ rootMultiplicity a p * (p /ₘ (X - C a) ^ rootMultiplicity a p) = p := by
  /-
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    a : R
    ⊢ Eq (HMul.hMul (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C a)) (Polynomi …
  -/
  have : Monic ((X - C a) ^ rootMultiplicity a p) := (monic_X_sub_C _).pow _
  conv_rhs =>
      rw [← modByMonic_add_div p this,
        (modByMonic_eq_zero_iff_dvd this).2 (pow_rootMultiplicity_dvd _ _)]
  /-
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    a : R
    this : (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C a)) (Polynomial.rootMu …
    ⊢ Eq (HMul.hMul (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C a)) (Polynomi …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem exists_eq_pow_rootMultiplicity_mul_and_not_dvd (p : R[X]) (hp : p ≠ 0) (a : R) :
    ∃ q : R[X], p = (X - C a) ^ p.rootMultiplicity a * q ∧ ¬ (X - C a) ∣ q := by
  classical
  rw [rootMultiplicity_eq_multiplicity, if_neg hp]
  apply (finiteMultiplicity_X_sub_C a hp).exists_eq_pow_mul_and_not_dvd


@[simp]
theorem modByMonic_X_sub_C_eq_C_eval (p : R[X]) (a : R) : p %ₘ (X - C a) = C (p.eval a) := by
  /-
    R : Type u
    inst✝ : CommRing R
    p : Polynomial R
    a : R
    ⊢ Eq (p.modByMonic (HSub.hSub Polynomial.X (Polynomial.C a))) (Polynomial.C (P …
  -/
  nontriviality R
  have h : (p %ₘ (X - C a)).eval a = p.eval a := by
    rw [modByMonic_eq_sub_mul_div _ (monic_X_sub_C a), eval_sub, eval_mul, eval_sub, eval_X,
      eval_C, sub_self, zero_mul, sub_zero]
  have : degree (p %ₘ (X - C a)) < 1 :=
    degree_X_sub_C a ▸ degree_modByMonic_lt p (monic_X_sub_C a)
  have : degree (p %ₘ (X - C a)) ≤ 0 := by
    revert this
    cases degree (p %ₘ (X - C a))
    · exact fun _ => bot_le
    · exact fun h => WithBot.coe_le_coe.2 (Nat.le_of_lt_succ (WithBot.coe_lt_coe.1 h))
  /-
    R : Type u
    inst✝ : CommRing R
    p : Polynomial R
    a : R
    a✝ : Nontrivial R
    h : Eq (Polynomial.eval a (p.modByMonic (HSub.hSub Polynomial.X (Polynomial.C  …
    this✝ : LT.lt (p.modByMonic (HSub.hSub Polynomial.X (Polynomial.C a))).degree 1
    this : LE.le (p.modByMonic (HSub.hSub Polynomial.X (Polynomial.C a))).degree 0
    ⊢ Eq (p.modByMonic (HSub.hSub Polynomial.X (Polynomial.C a))) (Polynomial.C (P …
  -/
  rw [eq_C_of_degree_le_zero this, eval_C] at h
  /-
    R : Type u
    inst✝ : CommRing R
    p : Polynomial R
    a : R
    a✝ : Nontrivial R
    h : Eq ((p.modByMonic (HSub.hSub Polynomial.X (Polynomial.C a))).coeff 0) (Pol …
    this✝ : LT.lt (p.modByMonic (HSub.hSub Polynomial.X (Polynomial.C a))).degree 1
    this : LE.le (p.modByMonic (HSub.hSub Polynomial.X (Polynomial.C a))).degree 0
    ⊢ Eq (p.modByMonic (HSub.hSub Polynomial.X (Polynomial.C a))) (Polynomial.C (P …
  -/
  rw [eq_C_of_degree_le_zero this, h]
  /-
    🎉 no goals
  -/


theorem mul_divByMonic_eq_iff_isRoot : (X - C a) * (p /ₘ (X - C a)) = p ↔ IsRoot p a :=
  .trans
                 /-
                   R : Type u
                   a : R
                   inst✝ : CommRing R
                   p : Polynomial R
                   h : Eq (HMul.hMul (HSub.hSub Polynomial.X (Polynomial.C a)) (p.divByMonic (HSu …
                   ⊢ Eq (Polynomial.eval a p) 0
                 -/
    ⟨fun h => by rw [← h, eval_mul, eval_sub, eval_X, eval_C, sub_self, zero_mul],
                 /-
                   🎉 no goals
                 -/
    fun h => by
      conv_rhs =>
        rw [← modByMonic_add_div p (monic_X_sub_C a)]
        rw [modByMonic_X_sub_C_eq_C_eval, h, C_0, zero_add]⟩
    IsRoot.def.symm


theorem dvd_iff_isRoot : X - C a ∣ p ↔ IsRoot p a :=
  ⟨fun h => by
    rwa [← modByMonic_eq_zero_iff_dvd (monic_X_sub_C _), modByMonic_X_sub_C_eq_C_eval, ← C_0,
      C_inj] at h,
                                 /-
                                   R : Type u
                                   a : R
                                   inst✝ : CommRing R
                                   p : Polynomial R
                                   h : p.IsRoot a
                                   ⊢ Eq p (HMul.hMul (HSub.hSub Polynomial.X (Polynomial.C a)) (p.divByMonic (HSu …
                                 -/
    fun h => ⟨p /ₘ (X - C a), by rw [mul_divByMonic_eq_iff_isRoot.2 h]⟩⟩
                                 /-
                                   🎉 no goals
                                 -/


theorem X_sub_C_dvd_sub_C_eval : X - C a ∣ p - C (p.eval a) := by
  /-
    R : Type u
    a : R
    inst✝ : CommRing R
    p : Polynomial R
    ⊢ Dvd.dvd (HSub.hSub Polynomial.X (Polynomial.C a)) (HSub.hSub p (Polynomial.C …
  -/
  rw [dvd_iff_isRoot, IsRoot, eval_sub, eval_C, sub_self]
  /-
    🎉 no goals
  -/

-- TODO: generalize this to Ring. In general, 0 can be replaced by any element in the center of R.

theorem modByMonic_X (p : R[X]) : p %ₘ X = C (p.eval 0) := by
  /-
    R : Type u
    inst✝ : CommRing R
    p : Polynomial R
    ⊢ Eq (p.modByMonic Polynomial.X) (Polynomial.C (Polynomial.eval 0 p))
  -/
  rw [← modByMonic_X_sub_C_eq_C_eval, C_0, sub_zero]
  /-
    🎉 no goals
  -/


theorem eval₂_modByMonic_eq_self_of_root [CommRing S] {f : R →+* S} {p q : R[X]} (hq : q.Monic)
    {x : S} (hx : q.eval₂ f x = 0) : (p %ₘ q).eval₂ f x = p.eval₂ f x := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : CommRing R
    inst✝ : CommRing S
    f : RingHom R S
    p q : Polynomial R
    hq : q.Monic
    x : S
    hx : Eq (Polynomial.eval₂ f x q) 0
    ⊢ Eq (Polynomial.eval₂ f x (p.modByMonic q)) (Polynomial.eval₂ f x p)
  -/
  rw [modByMonic_eq_sub_mul_div p hq, eval₂_sub, eval₂_mul, hx, zero_mul, sub_zero]
  /-
    🎉 no goals
  -/


theorem sub_dvd_eval_sub (a b : R) (p : R[X]) : a - b ∣ p.eval a - p.eval b := by
  suffices X - C b ∣ p - C (p.eval b) by
    simpa only [coe_evalRingHom, eval_sub, eval_X, eval_C] using (evalRingHom a).map_dvd this
  /-
    R : Type u
    inst✝ : CommRing R
    a b : R
    p : Polynomial R
    ⊢ Dvd.dvd (HSub.hSub Polynomial.X (Polynomial.C b)) (HSub.hSub p (Polynomial.C …
  -/
  simp [dvd_iff_isRoot]
  /-
    🎉 no goals
  -/


@[simp]
theorem rootMultiplicity_eq_zero_iff {p : R[X]} {x : R} :
    rootMultiplicity x p = 0 ↔ IsRoot p x → p = 0 := by
  classical
  simp only [rootMultiplicity_eq_multiplicity, ite_eq_left_iff,
    Nat.cast_zero, multiplicity_eq_zero, dvd_iff_isRoot, not_imp_not]


theorem rootMultiplicity_eq_zero {p : R[X]} {x : R} (h : ¬IsRoot p x) : rootMultiplicity x p = 0 :=
  rootMultiplicity_eq_zero_iff.2 fun h' => (h h').elim


@[simp]
theorem rootMultiplicity_pos' {p : R[X]} {x : R} :
    0 < rootMultiplicity x p ↔ p ≠ 0 ∧ IsRoot p x := by
  /-
    R : Type u
    inst✝ : CommRing R
    p : Polynomial R
    x : R
    ⊢ Iff (LT.lt 0 (Polynomial.rootMultiplicity x p)) (And (Ne p 0) (p.IsRoot x))
  -/
  rw [pos_iff_ne_zero, Ne, rootMultiplicity_eq_zero_iff, Classical.not_imp, and_comm]
  /-
    🎉 no goals
  -/


theorem rootMultiplicity_pos {p : R[X]} (hp : p ≠ 0) {x : R} :
    0 < rootMultiplicity x p ↔ IsRoot p x :=
  rootMultiplicity_pos'.trans (and_iff_right hp)


theorem eval_divByMonic_pow_rootMultiplicity_ne_zero {p : R[X]} (a : R) (hp : p ≠ 0) :
    eval a (p /ₘ (X - C a) ^ rootMultiplicity a p) ≠ 0 := by
  classical
  haveI : Nontrivial R := Nontrivial.of_polynomial_ne hp
  rw [Ne, ← IsRoot, ← dvd_iff_isRoot]
  rintro ⟨q, hq⟩
  have := pow_mul_divByMonic_rootMultiplicity_eq p a
  rw [hq, ← mul_assoc, ← pow_succ, rootMultiplicity_eq_multiplicity, if_neg hp] at this
  exact
    (finiteMultiplicity_of_degree_pos_of_monic
      (show (0 : WithBot ℕ) < degree (X - C a) by rw [degree_X_sub_C]; decide)
      (monic_X_sub_C _) hp).not_pow_dvd_of_multiplicity_lt
      (Nat.lt_succ_self _) (dvd_of_mul_right_eq _ this)


/-- See `Polynomial.mul_right_modByMonic` for the other multiplication order. This version, unlike
that one, requires commutativity. -/
@[simp]
lemma mul_self_modByMonic (hq : q.Monic) : (p * q) %ₘ q = 0 := by
  /-
    R : Type u
    inst✝ : CommRing R
    p q : Polynomial R
    hq : q.Monic
    ⊢ Eq ((HMul.hMul p q).modByMonic q) 0
  -/
  rw [modByMonic_eq_zero_iff_dvd hq]
  /-
    R : Type u
    inst✝ : CommRing R
    p q : Polynomial R
    hq : q.Monic
    ⊢ Dvd.dvd q (HMul.hMul p q)
  -/
  exact dvd_mul_left q p
  /-
    🎉 no goals
  -/


lemma modByMonic_eq_of_dvd_sub (hq : q.Monic) (h : q ∣ p₁ - p₂) : p₁ %ₘ q = p₂ %ₘ q := by
  /-
    R : Type u
    inst✝ : CommRing R
    p₁ p₂ q : Polynomial R
    hq : q.Monic
    h : Dvd.dvd q (HSub.hSub p₁ p₂)
    ⊢ Eq (p₁.modByMonic q) (p₂.modByMonic q)
  -/
  nontriviality R
  /-
    R : Type u
    inst✝ : CommRing R
    p₁ p₂ q : Polynomial R
    hq : q.Monic
    h : Dvd.dvd q (HSub.hSub p₁ p₂)
    a✝ : Nontrivial R
    ⊢ Eq (p₁.modByMonic q) (p₂.modByMonic q)
  -/
  obtain ⟨f, sub_eq⟩ := h
  /-
    case intro
    R : Type u
    inst✝ : CommRing R
    p₁ p₂ q : Polynomial R
    hq : q.Monic
    a✝ : Nontrivial R
    f : Polynomial R
    sub_eq : Eq (HSub.hSub p₁ p₂) (HMul.hMul q f)
    ⊢ Eq (p₁.modByMonic q) (p₂.modByMonic q)
  -/
  refine (div_modByMonic_unique (p₂ /ₘ q + f) _ hq ⟨?_, degree_modByMonic_lt _ hq⟩).2
  /-
    case intro
    R : Type u
    inst✝ : CommRing R
    p₁ p₂ q : Polynomial R
    hq : q.Monic
    a✝ : Nontrivial R
    f : Polynomial R
    sub_eq : Eq (HSub.hSub p₁ p₂) (HMul.hMul q f)
    ⊢ Eq (HAdd.hAdd (p₂.modByMonic q) (HMul.hMul q (HAdd.hAdd (p₂.divByMonic q) f) …
  -/
  rw [sub_eq_iff_eq_add.mp sub_eq, mul_add, ← add_assoc, modByMonic_add_div _ hq, add_comm]
  /-
    🎉 no goals
  -/


lemma add_modByMonic (p₁ p₂ : R[X]) : (p₁ + p₂) %ₘ q = p₁ %ₘ q + p₂ %ₘ q := by
  /-
    R : Type u
    inst✝ : CommRing R
    q p₁ p₂ : Polynomial R
    ⊢ Eq ((HAdd.hAdd p₁ p₂).modByMonic q) (HAdd.hAdd (p₁.modByMonic q) (p₂.modByMo …
  -/
  by_cases hq : q.Monic
    /-
      case pos
      R : Type u
      inst✝ : CommRing R
      q p₁ p₂ : Polynomial R
      hq : q.Monic
      ⊢ Eq ((HAdd.hAdd p₁ p₂).modByMonic q) (HAdd.hAdd (p₁.modByMonic q) (p₂.modByMo …
    -/
  · cases' subsingleton_or_nontrivial R with hR hR
      /-
        case pos.inl
        R : Type u
        inst✝ : CommRing R
        q p₁ p₂ : Polynomial R
        hq : q.Monic
        hR : Subsingleton R
        ⊢ Eq ((HAdd.hAdd p₁ p₂).modByMonic q) (HAdd.hAdd (p₁.modByMonic q) (p₂.modByMo …
      -/
    · simp only [eq_iff_true_of_subsingleton]
      /-
        🎉 no goals
      -/
    · exact
      (div_modByMonic_unique (p₁ /ₘ q + p₂ /ₘ q) _ hq
          ⟨by
            rw [mul_add, add_left_comm, add_assoc, modByMonic_add_div _ hq, ← add_assoc,
              add_comm (q * _), modByMonic_add_div _ hq],
            (degree_add_le _ _).trans_lt
              (max_lt (degree_modByMonic_lt _ hq) (degree_modByMonic_lt _ hq))⟩).2
    /-
      case neg
      R : Type u
      inst✝ : CommRing R
      q p₁ p₂ : Polynomial R
      hq : Not q.Monic
      ⊢ Eq ((HAdd.hAdd p₁ p₂).modByMonic q) (HAdd.hAdd (p₁.modByMonic q) (p₂.modByMo …
    -/
  · simp_rw [modByMonic_eq_of_not_monic _ hq]
    /-
      🎉 no goals
    -/


lemma neg_modByMonic (p q : R[X]) : (-p) %ₘ q = - (p %ₘ q) := by
  /-
    R : Type u
    inst✝ : CommRing R
    p q : Polynomial R
    ⊢ Eq ((Neg.neg p).modByMonic q) (Neg.neg (p.modByMonic q))
  -/
  rw [eq_neg_iff_add_eq_zero, ← add_modByMonic, neg_add_cancel, zero_modByMonic]
  /-
    🎉 no goals
  -/


lemma sub_modByMonic (p₁ p₂ q : R[X]) : (p₁ - p₂) %ₘ q = p₁ %ₘ q - p₂ %ₘ q := by
  /-
    R : Type u
    inst✝ : CommRing R
    p₁ p₂ q : Polynomial R
    ⊢ Eq ((HSub.hSub p₁ p₂).modByMonic q) (HSub.hSub (p₁.modByMonic q) (p₂.modByMo …
  -/
  simp [sub_eq_add_neg, add_modByMonic, neg_modByMonic]
  /-
    🎉 no goals
  -/


lemma eval_divByMonic_eq_trailingCoeff_comp {p : R[X]} {t : R} :
    (p /ₘ (X - C t) ^ p.rootMultiplicity t).eval t = (p.comp (X + C t)).trailingCoeff := by
  /-
    R : Type u
    inst✝ : CommRing R
    p : Polynomial R
    t : R
    ⊢ Eq (Polynomial.eval t (p.divByMonic (HPow.hPow (HSub.hSub Polynomial.X (Poly …
  -/
  obtain rfl | hp := eq_or_ne p 0
    /-
      case inl
      R : Type u
      inst✝ : CommRing R
      t : R
      ⊢ Eq (Polynomial.eval t (Polynomial.divByMonic 0 (HPow.hPow (HSub.hSub Polynom …
    -/
  · rw [zero_divByMonic, eval_zero, zero_comp, trailingCoeff_zero]
    /-
      🎉 no goals
    -/
  /-
    case inr
    R : Type u
    inst✝ : CommRing R
    p : Polynomial R
    t : R
    hp : Ne p 0
    ⊢ Eq (Polynomial.eval t (p.divByMonic (HPow.hPow (HSub.hSub Polynomial.X (Poly …
  -/
  have mul_eq := p.pow_mul_divByMonic_rootMultiplicity_eq t
  /-
    case inr
    R : Type u
    inst✝ : CommRing R
    p : Polynomial R
    t : R
    hp : Ne p 0
    mul_eq : Eq (HMul.hMul (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C t)) (P …
    ⊢ Eq (Polynomial.eval t (p.divByMonic (HPow.hPow (HSub.hSub Polynomial.X (Poly …
  -/
  set m := p.rootMultiplicity t
  /-
    case inr
    R : Type u
    inst✝ : CommRing R
    p : Polynomial R
    t : R
    hp : Ne p 0
    m : Nat := Polynomial.rootMultiplicity t p
    mul_eq : Eq (HMul.hMul (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C t)) m) …
    ⊢ Eq (Polynomial.eval t (p.divByMonic (HPow.hPow (HSub.hSub Polynomial.X (Poly …
  -/
  set g := p /ₘ (X - C t) ^ m
  have : (g.comp (X + C t)).coeff 0 = g.eval t := by
    rw [coeff_zero_eq_eval_zero, eval_comp, eval_add, eval_X, eval_C, zero_add]
  rw [← congr_arg (comp · <| X + C t) mul_eq, mul_comp, pow_comp, sub_comp, X_comp, C_comp,
    add_sub_cancel_right, ← reverse_leadingCoeff, reverse_X_pow_mul, reverse_leadingCoeff,
    trailingCoeff, Nat.le_zero.1 (natTrailingDegree_le_of_ne_zero <|
      this ▸ eval_divByMonic_pow_rootMultiplicity_ne_zero t hp), this]

/- Porting note: the ML3 proof no longer worked because of a conflict in the
inferred type and synthesized type for `DecidableRel` when using `Nat.le_find_iff` from
`Mathlib.Algebra.Polynomial.Div` After some discussion on [Zulip]
(https://leanprover.zulipchat.com/#narrow/stream/287929-mathlib4/topic/decidability.20leakage)
introduced `Polynomial.rootMultiplicity_eq_nat_find_of_nonzero` to contain the issue
-/

/-- The multiplicity of `a` as root of a nonzero polynomial `p` is at least `n` iff
`(X - a) ^ n` divides `p`. -/
lemma le_rootMultiplicity_iff (p0 : p ≠ 0) {a : R} {n : ℕ} :
    n ≤ rootMultiplicity a p ↔ (X - C a) ^ n ∣ p := by
  classical
  rw [rootMultiplicity_eq_nat_find_of_nonzero p0, @Nat.le_find_iff _ (_)]
  simp_rw [Classical.not_not]
  refine ⟨fun h => ?_, fun h m hm => (pow_dvd_pow _ hm).trans h⟩
  cases' n with n
  · rw [pow_zero]
    apply one_dvd
  · exact h n n.lt_succ_self


lemma rootMultiplicity_le_iff (p0 : p ≠ 0) (a : R) (n : ℕ) :
    rootMultiplicity a p ≤ n ↔ ¬(X - C a) ^ (n + 1) ∣ p := by
  /-
    R : Type u
    inst✝ : CommRing R
    p : Polynomial R
    p0 : Ne p 0
    a : R
    n : Nat
    ⊢ Iff (LE.le (Polynomial.rootMultiplicity a p) n) (Not (Dvd.dvd (HPow.hPow (HS …
  -/
  rw [← (le_rootMultiplicity_iff p0).not, not_le, Nat.lt_add_one_iff]
  /-
    🎉 no goals
  -/


/-- The multiplicity of `p + q` is at least the minimum of the multiplicities. -/
lemma rootMultiplicity_add {p q : R[X]} (a : R) (hzero : p + q ≠ 0) :
    min (rootMultiplicity a p) (rootMultiplicity a q) ≤ rootMultiplicity a (p + q) := by
  /-
    R : Type u
    inst✝ : CommRing R
    p q : Polynomial R
    a : R
    hzero : Ne (HAdd.hAdd p q) 0
    ⊢ LE.le (Min.min (Polynomial.rootMultiplicity a p) (Polynomial.rootMultiplicit …
  -/
  rw [le_rootMultiplicity_iff hzero]
  /-
    R : Type u
    inst✝ : CommRing R
    p q : Polynomial R
    a : R
    hzero : Ne (HAdd.hAdd p q) 0
    ⊢ Dvd.dvd (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C a)) (Min.min (Polyn …
  -/
  exact min_pow_dvd_add (pow_rootMultiplicity_dvd p a) (pow_rootMultiplicity_dvd q a)
  /-
    🎉 no goals
  -/


lemma le_rootMultiplicity_mul {p q : R[X]} (x : R) (hpq : p * q ≠ 0) :
    rootMultiplicity x p + rootMultiplicity x q ≤ rootMultiplicity x (p * q) := by
  /-
    R : Type u
    inst✝ : CommRing R
    p q : Polynomial R
    x : R
    hpq : Ne (HMul.hMul p q) 0
    ⊢ LE.le (HAdd.hAdd (Polynomial.rootMultiplicity x p) (Polynomial.rootMultiplic …
  -/
  rw [le_rootMultiplicity_iff hpq, pow_add]
  /-
    R : Type u
    inst✝ : CommRing R
    p q : Polynomial R
    x : R
    hpq : Ne (HMul.hMul p q) 0
    ⊢ Dvd.dvd (HMul.hMul (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C x)) (Pol …
  -/
  exact mul_dvd_mul (pow_rootMultiplicity_dvd p x) (pow_rootMultiplicity_dvd q x)
  /-
    🎉 no goals
  -/


lemma pow_rootMultiplicity_not_dvd (p0 : p ≠ 0) (a : R) :
                                                      /-
                                                        R : Type u
                                                        inst✝ : CommRing R
                                                        p : Polynomial R
                                                        p0 : Ne p 0
                                                        a : R
                                                        ⊢ Not (Dvd.dvd (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C a)) (HAdd.hAdd …
                                                      -/
    ¬(X - C a) ^ (rootMultiplicity a p + 1) ∣ p := by rw [← rootMultiplicity_le_iff p0]
                                                      /-
                                                        🎉 no goals
                                                      -/


/-- See `Polynomial.rootMultiplicity_eq_natTrailingDegree` for the general case. -/
lemma rootMultiplicity_eq_natTrailingDegree' : p.rootMultiplicity 0 = p.natTrailingDegree := by
  /-
    R : Type u
    inst✝ : CommRing R
    p : Polynomial R
    ⊢ Eq (Polynomial.rootMultiplicity 0 p) p.natTrailingDegree
  -/
  by_cases h : p = 0
    /-
      case pos
      R : Type u
      inst✝ : CommRing R
      p : Polynomial R
      h : Eq p 0
      ⊢ Eq (Polynomial.rootMultiplicity 0 p) p.natTrailingDegree
    -/
  · simp only [h, rootMultiplicity_zero, natTrailingDegree_zero]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u
    inst✝ : CommRing R
    p : Polynomial R
    h : Not (Eq p 0)
    ⊢ Eq (Polynomial.rootMultiplicity 0 p) p.natTrailingDegree
  -/
  refine le_antisymm ?_ ?_
    /-
      case neg.refine_1
      R : Type u
      inst✝ : CommRing R
      p : Polynomial R
      h : Not (Eq p 0)
      ⊢ LE.le (Polynomial.rootMultiplicity 0 p) p.natTrailingDegree
    -/
  · rw [rootMultiplicity_le_iff h, map_zero, sub_zero, X_pow_dvd_iff, not_forall]
    exact ⟨p.natTrailingDegree,
      fun h' ↦ trailingCoeff_nonzero_iff_nonzero.2 h <| h' <| Nat.lt.base _⟩
    /-
      case neg.refine_2
      R : Type u
      inst✝ : CommRing R
      p : Polynomial R
      h : Not (Eq p 0)
      ⊢ LE.le p.natTrailingDegree (Polynomial.rootMultiplicity 0 p)
    -/
  · rw [le_rootMultiplicity_iff h, map_zero, sub_zero, X_pow_dvd_iff]
    /-
      case neg.refine_2
      R : Type u
      inst✝ : CommRing R
      p : Polynomial R
      h : Not (Eq p 0)
      ⊢ ∀ (d : Nat), LT.lt d p.natTrailingDegree → Eq (p.coeff d) 0
    -/
    exact fun _ ↦ coeff_eq_zero_of_lt_natTrailingDegree
    /-
      🎉 no goals
    -/


/-- Division by a monic polynomial doesn't change the leading coefficient. -/
lemma leadingCoeff_divByMonic_of_monic (hmonic : q.Monic)
    (hdegree : q.degree ≤ p.degree) : (p /ₘ q).leadingCoeff = p.leadingCoeff := by
  /-
    R : Type u
    inst✝ : CommRing R
    p q : Polynomial R
    hmonic : q.Monic
    hdegree : LE.le q.degree p.degree
    ⊢ Eq (p.divByMonic q).leadingCoeff p.leadingCoeff
  -/
  nontriviality
  have h : q.leadingCoeff * (p /ₘ q).leadingCoeff ≠ 0 := by
    simpa [divByMonic_eq_zero_iff hmonic, hmonic.leadingCoeff,
      Nat.WithBot.one_le_iff_zero_lt] using hdegree
  /-
    R : Type u
    inst✝ : CommRing R
    p q : Polynomial R
    hmonic : q.Monic
    hdegree : LE.le q.degree p.degree
    a✝ : Nontrivial R
    h : Ne (HMul.hMul q.leadingCoeff (p.divByMonic q).leadingCoeff) 0
    ⊢ Eq (p.divByMonic q).leadingCoeff p.leadingCoeff
  -/
  nth_rw 2 [← modByMonic_add_div p hmonic]
  /-
    R : Type u
    inst✝ : CommRing R
    p q : Polynomial R
    hmonic : q.Monic
    hdegree : LE.le q.degree p.degree
    a✝ : Nontrivial R
    h : Ne (HMul.hMul q.leadingCoeff (p.divByMonic q).leadingCoeff) 0
    ⊢ Eq (p.divByMonic q).leadingCoeff (HAdd.hAdd (p.modByMonic q) (HMul.hMul q (p …
  -/
  rw [leadingCoeff_add_of_degree_lt, leadingCoeff_monic_mul hmonic]
  /-
    R : Type u
    inst✝ : CommRing R
    p q : Polynomial R
    hmonic : q.Monic
    hdegree : LE.le q.degree p.degree
    a✝ : Nontrivial R
    h : Ne (HMul.hMul q.leadingCoeff (p.divByMonic q).leadingCoeff) 0
    ⊢ LT.lt (p.modByMonic q).degree (HMul.hMul q (p.divByMonic q)).degree
  -/
  rw [degree_mul' h, degree_add_divByMonic hmonic hdegree]
  /-
    R : Type u
    inst✝ : CommRing R
    p q : Polynomial R
    hmonic : q.Monic
    hdegree : LE.le q.degree p.degree
    a✝ : Nontrivial R
    h : Ne (HMul.hMul q.leadingCoeff (p.divByMonic q).leadingCoeff) 0
    ⊢ LT.lt (p.modByMonic q).degree p.degree
  -/
  exact (degree_modByMonic_lt p hmonic).trans_le hdegree
  /-
    🎉 no goals
  -/


lemma degree_eq_one_of_irreducible_of_root (hi : Irreducible p) {x : R} (hx : IsRoot p x) :
    degree p = 1 :=
  let ⟨g, hg⟩ := dvd_iff_isRoot.2 hx
  have : IsUnit (X - C x) ∨ IsUnit g := hi.isUnit_or_isUnit hg
  this.elim
    (fun h => by
      /-
        R : Type u
        inst✝¹ : CommRing R
        p : Polynomial R
        inst✝ : IsDomain R
        hi : Irreducible p
        x : R
        hx : p.IsRoot x
        g : Polynomial R
        hg : Eq p (HMul.hMul (HSub.hSub Polynomial.X (Polynomial.C x)) g)
        this : Or (IsUnit (HSub.hSub Polynomial.X (Polynomial.C x))) (IsUnit g)
        h : IsUnit (HSub.hSub Polynomial.X (Polynomial.C x))
        ⊢ Eq p.degree 1
      -/
      have h₁ : degree (X - C x) = 1 := degree_X_sub_C x
      /-
        R : Type u
        inst✝¹ : CommRing R
        p : Polynomial R
        inst✝ : IsDomain R
        hi : Irreducible p
        x : R
        hx : p.IsRoot x
        g : Polynomial R
        hg : Eq p (HMul.hMul (HSub.hSub Polynomial.X (Polynomial.C x)) g)
        this : Or (IsUnit (HSub.hSub Polynomial.X (Polynomial.C x))) (IsUnit g)
        h : IsUnit (HSub.hSub Polynomial.X (Polynomial.C x))
        h₁ : Eq (HSub.hSub Polynomial.X (Polynomial.C x)).degree 1
        ⊢ Eq p.degree 1
      -/
      have h₂ : degree (X - C x) = 0 := degree_eq_zero_of_isUnit h
      /-
        R : Type u
        inst✝¹ : CommRing R
        p : Polynomial R
        inst✝ : IsDomain R
        hi : Irreducible p
        x : R
        hx : p.IsRoot x
        g : Polynomial R
        hg : Eq p (HMul.hMul (HSub.hSub Polynomial.X (Polynomial.C x)) g)
        this : Or (IsUnit (HSub.hSub Polynomial.X (Polynomial.C x))) (IsUnit g)
        h : IsUnit (HSub.hSub Polynomial.X (Polynomial.C x))
        h₁ : Eq (HSub.hSub Polynomial.X (Polynomial.C x)).degree 1
        h₂ : Eq (HSub.hSub Polynomial.X (Polynomial.C x)).degree 0
        ⊢ Eq p.degree 1
      -/
      rw [h₁] at h₂; exact absurd h₂ (by decide))
                     /-
                       🎉 no goals
                     -/
                  /-
                    R : Type u
                    inst✝¹ : CommRing R
                    p : Polynomial R
                    inst✝ : IsDomain R
                    hi : Irreducible p
                    x : R
                    hx : p.IsRoot x
                    g : Polynomial R
                    hg : Eq p (HMul.hMul (HSub.hSub Polynomial.X (Polynomial.C x)) g)
                    this : Or (IsUnit (HSub.hSub Polynomial.X (Polynomial.C x))) (IsUnit g)
                    hgu : IsUnit g
                    ⊢ Eq p.degree 1
                  -/
    fun hgu => by rw [hg, degree_mul, degree_X_sub_C, degree_eq_zero_of_isUnit hgu, add_zero]
                  /-
                    🎉 no goals
                  -/


lemma leadingCoeff_divByMonic_X_sub_C (p : R[X]) (hp : degree p ≠ 0) (a : R) :
    leadingCoeff (p /ₘ (X - C a)) = leadingCoeff p := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    p : Polynomial R
    hp : Ne p.degree 0
    a : R
    ⊢ Eq (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C a))).leadingCoeff p.l …
  -/
  nontriviality
  /-
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    p : Polynomial R
    hp : Ne p.degree 0
    a : R
    inst✝ : Nontrivial R
    ⊢ Eq (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C a))).leadingCoeff p.l …
  -/
  cases' hp.lt_or_lt with hd hd
    /-
      case inl
      R : Type u
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      p : Polynomial R
      hp : Ne p.degree 0
      a : R
      inst✝ : Nontrivial R
      hd : LT.lt p.degree 0
      ⊢ Eq (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C a))).leadingCoeff p.l …
    -/
  · rw [degree_eq_bot.mp <| Nat.WithBot.lt_zero_iff.mp hd, zero_divByMonic]
    /-
      🎉 no goals
    -/
  /-
    case inr
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    p : Polynomial R
    hp : Ne p.degree 0
    a : R
    inst✝ : Nontrivial R
    hd : LT.lt 0 p.degree
    ⊢ Eq (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C a))).leadingCoeff p.l …
  -/
  refine leadingCoeff_divByMonic_of_monic (monic_X_sub_C a) ?_
  /-
    case inr
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    p : Polynomial R
    hp : Ne p.degree 0
    a : R
    inst✝ : Nontrivial R
    hd : LT.lt 0 p.degree
    ⊢ LE.le (HSub.hSub Polynomial.X (Polynomial.C a)).degree p.degree
  -/
  rwa [degree_X_sub_C, Nat.WithBot.one_le_iff_zero_lt]
  /-
    🎉 no goals
  -/


lemma eq_of_dvd_of_natDegree_le_of_leadingCoeff {p q : R[X]} (hpq : p ∣ q)
    (h₁ : q.natDegree ≤ p.natDegree) (h₂ : p.leadingCoeff = q.leadingCoeff) :
    p = q := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    p q : Polynomial R
    hpq : Dvd.dvd p q
    h₁ : LE.le q.natDegree p.natDegree
    h₂ : Eq p.leadingCoeff q.leadingCoeff
    ⊢ Eq p q
  -/
  by_cases hq : q = 0
    /-
      case pos
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      p q : Polynomial R
      hpq : Dvd.dvd p q
      h₁ : LE.le q.natDegree p.natDegree
      h₂ : Eq p.leadingCoeff q.leadingCoeff
      hq : Eq q 0
      ⊢ Eq p q
    -/
  · rwa [hq, leadingCoeff_zero, leadingCoeff_eq_zero, ← hq] at h₂
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    p q : Polynomial R
    hpq : Dvd.dvd p q
    h₁ : LE.le q.natDegree p.natDegree
    h₂ : Eq p.leadingCoeff q.leadingCoeff
    hq : Not (Eq q 0)
    ⊢ Eq p q
  -/
  replace h₁ := (natDegree_le_of_dvd hpq hq).antisymm h₁
  /-
    case neg
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    p q : Polynomial R
    hpq : Dvd.dvd p q
    h₂ : Eq p.leadingCoeff q.leadingCoeff
    hq : Not (Eq q 0)
    h₁ : Eq p.natDegree q.natDegree
    ⊢ Eq p q
  -/
  obtain ⟨u, rfl⟩ := hpq
  /-
    case neg.intro
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    p u : Polynomial R
    h₂ : Eq p.leadingCoeff (HMul.hMul p u).leadingCoeff
    hq : Not (Eq (HMul.hMul p u) 0)
    h₁ : Eq p.natDegree (HMul.hMul p u).natDegree
    ⊢ Eq p (HMul.hMul p u)
  -/
  replace hq := mul_ne_zero_iff.mp hq
  /-
    case neg.intro
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    p u : Polynomial R
    h₂ : Eq p.leadingCoeff (HMul.hMul p u).leadingCoeff
    h₁ : Eq p.natDegree (HMul.hMul p u).natDegree
    hq : And (Ne p 0) (Ne u 0)
    ⊢ Eq p (HMul.hMul p u)
  -/
  rw [natDegree_mul hq.1 hq.2, self_eq_add_right] at h₁
  rw [eq_C_of_natDegree_eq_zero h₁, leadingCoeff_mul, leadingCoeff_C,
    eq_comm, mul_eq_left₀ (leadingCoeff_ne_zero.mpr hq.1)] at h₂
  /-
    case neg.intro
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    p u : Polynomial R
    h₂ : Eq (u.coeff 0) 1
    h₁ : Eq u.natDegree 0
    hq : And (Ne p 0) (Ne u 0)
    ⊢ Eq p (HMul.hMul p u)
  -/
  rw [eq_C_of_natDegree_eq_zero h₁, h₂, map_one, mul_one]
  /-
    🎉 no goals
  -/


lemma associated_of_dvd_of_natDegree_le_of_leadingCoeff {p q : R[X]} (hpq : p ∣ q)
    (h₁ : q.natDegree ≤ p.natDegree) (h₂ : q.leadingCoeff ∣ p.leadingCoeff) :
    Associated p q :=
  have ⟨r, hr⟩ := hpq
  have ⟨u, hu⟩ := associated_of_dvd_dvd ⟨leadingCoeff r, hr ▸ leadingCoeff_mul p r⟩ h₂
  ⟨Units.map C.toMonoidHom u, eq_of_dvd_of_natDegree_le_of_leadingCoeff
        /-
          R : Type u
          inst✝¹ : CommRing R
          inst✝ : IsDomain R
          p q : Polynomial R
          hpq : Dvd.dvd p q
          h₁ : LE.le q.natDegree p.natDegree
          h₂ : Dvd.dvd q.leadingCoeff p.leadingCoeff
          r : Polynomial R
          hr : Eq q (HMul.hMul p r)
          u : Units R
          hu : Eq (HMul.hMul p.leadingCoeff ↑u) q.leadingCoeff
          ⊢ Dvd.dvd (HMul.hMul p ↑((Units.map ↑Polynomial.C) u)) q
        -/
        /-
          🎉 no goals
        -/
                                       /-
                                         🎉 no goals
                                       -/
    (by rwa [Units.mul_right_dvd]) (by simpa [natDegree_mul_C] using h₁) (by simpa using hu)⟩
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


lemma associated_of_dvd_of_natDegree_le {K} [Field K] {p q : K[X]} (hpq : p ∣ q) (hq : q ≠ 0)
    (h₁ : q.natDegree ≤ p.natDegree) : Associated p q :=
  associated_of_dvd_of_natDegree_le_of_leadingCoeff hpq h₁
                    /-
                      K : Type u_1
                      inst✝ : Field K
                      p q : Polynomial K
                      hpq : Dvd.dvd p q
                      hq : Ne q 0
                      h₁ : LE.le q.natDegree p.natDegree
                      ⊢ IsUnit q.leadingCoeff
                    -/
    (IsUnit.dvd (by rwa [← leadingCoeff_ne_zero, ← isUnit_iff_ne_zero] at hq))
                    /-
                      🎉 no goals
                    -/


lemma associated_of_dvd_of_degree_eq {K} [Field K] {p q : K[X]} (hpq : p ∣ q)
    (h₁ : p.degree = q.degree) : Associated p q :=
                                                       /-
                                                         K : Type u_1
                                                         inst✝ : Field K
                                                         p q : Polynomial K
                                                         hpq : Dvd.dvd p q
                                                         h₁ : Eq p.degree q.degree
                                                         hq : Eq q 0
                                                         ⊢ Eq p q
                                                       -/
  (Classical.em (q = 0)).elim (fun hq ↦ (show p = q by simpa [hq] using h₁) ▸ Associated.refl p)
                                                       /-
                                                         🎉 no goals
                                                       -/
    (associated_of_dvd_of_natDegree_le hpq · (natDegree_le_natDegree h₁.ge))


lemma eq_leadingCoeff_mul_of_monic_of_dvd_of_natDegree_le {R} [CommRing R] {p q : R[X]}
    (hp : p.Monic) (hdiv : p ∣ q) (hdeg : q.natDegree ≤ p.natDegree) :
    q = C q.leadingCoeff * p := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    p q : Polynomial R
    hp : p.Monic
    hdiv : Dvd.dvd p q
    hdeg : LE.le q.natDegree p.natDegree
    ⊢ Eq q (HMul.hMul (Polynomial.C q.leadingCoeff) p)
  -/
  obtain ⟨r, hr⟩ := hdiv
  /-
    case intro
    R : Type u_1
    inst✝ : CommRing R
    p q : Polynomial R
    hp : p.Monic
    hdeg : LE.le q.natDegree p.natDegree
    r : Polynomial R
    hr : Eq q (HMul.hMul p r)
    ⊢ Eq q (HMul.hMul (Polynomial.C q.leadingCoeff) p)
  -/
  obtain rfl | hq := eq_or_ne q 0; · simp
                                     /-
                                       🎉 no goals
                                     -/
  /-
    case intro.inr
    R : Type u_1
    inst✝ : CommRing R
    p q : Polynomial R
    hp : p.Monic
    hdeg : LE.le q.natDegree p.natDegree
    r : Polynomial R
    hr : Eq q (HMul.hMul p r)
    hq : Ne q 0
    ⊢ Eq q (HMul.hMul (Polynomial.C q.leadingCoeff) p)
  -/
  have rzero : r ≠ 0 := fun h => by simp [h, hq] at hr
  /-
    case intro.inr
    R : Type u_1
    inst✝ : CommRing R
    p q : Polynomial R
    hp : p.Monic
    hdeg : LE.le q.natDegree p.natDegree
    r : Polynomial R
    hr : Eq q (HMul.hMul p r)
    hq : Ne q 0
    rzero : Ne r 0
    ⊢ Eq q (HMul.hMul (Polynomial.C q.leadingCoeff) p)
  -/
  rw [hr, natDegree_mul'] at hdeg; swap
    /-
      case intro.inr
      R : Type u_1
      inst✝ : CommRing R
      p q : Polynomial R
      hp : p.Monic
      r : Polynomial R
      hdeg : LE.le (HMul.hMul p r).natDegree p.natDegree
      hr : Eq q (HMul.hMul p r)
      hq : Ne q 0
      rzero : Ne r 0
      ⊢ Ne (HMul.hMul p.leadingCoeff r.leadingCoeff) 0
    -/
  · rw [hp.leadingCoeff, one_mul, leadingCoeff_ne_zero]
    /-
      case intro.inr
      R : Type u_1
      inst✝ : CommRing R
      p q : Polynomial R
      hp : p.Monic
      r : Polynomial R
      hdeg : LE.le (HMul.hMul p r).natDegree p.natDegree
      hr : Eq q (HMul.hMul p r)
      hq : Ne q 0
      rzero : Ne r 0
      ⊢ Ne r 0
    -/
    exact rzero
    /-
      🎉 no goals
    -/
  /-
    case intro.inr
    R : Type u_1
    inst✝ : CommRing R
    p q : Polynomial R
    hp : p.Monic
    r : Polynomial R
    hdeg : LE.le (HAdd.hAdd p.natDegree r.natDegree) p.natDegree
    hr : Eq q (HMul.hMul p r)
    hq : Ne q 0
    rzero : Ne r 0
    ⊢ Eq q (HMul.hMul (Polynomial.C q.leadingCoeff) p)
  -/
  rw [mul_comm, @eq_C_of_natDegree_eq_zero _ _ r] at hr
    /-
      case intro.inr
      R : Type u_1
      inst✝ : CommRing R
      p q : Polynomial R
      hp : p.Monic
      r : Polynomial R
      hdeg : LE.le (HAdd.hAdd p.natDegree r.natDegree) p.natDegree
      hr : Eq q (HMul.hMul (Polynomial.C (r.coeff 0)) p)
      hq : Ne q 0
      rzero : Ne r 0
      ⊢ Eq q (HMul.hMul (Polynomial.C q.leadingCoeff) p)
    -/
  · convert hr
    /-
      case h.e'_3.h.e'_5.h.e'_6
      R : Type u_1
      inst✝ : CommRing R
      p q : Polynomial R
      hp : p.Monic
      r : Polynomial R
      hdeg : LE.le (HAdd.hAdd p.natDegree r.natDegree) p.natDegree
      hr : Eq q (HMul.hMul (Polynomial.C (r.coeff 0)) p)
      hq : Ne q 0
      rzero : Ne r 0
      ⊢ Eq q.leadingCoeff (r.coeff 0)
    -/
    convert leadingCoeff_C (coeff r 0) using 1
    /-
      case h.e'_2
      R : Type u_1
      inst✝ : CommRing R
      p q : Polynomial R
      hp : p.Monic
      r : Polynomial R
      hdeg : LE.le (HAdd.hAdd p.natDegree r.natDegree) p.natDegree
      hr : Eq q (HMul.hMul (Polynomial.C (r.coeff 0)) p)
      hq : Ne q 0
      rzero : Ne r 0
      ⊢ Eq q.leadingCoeff (Polynomial.C (r.coeff 0)).leadingCoeff
    -/
    rw [hr, leadingCoeff_mul_monic hp]
    /-
      🎉 no goals
    -/
    /-
      case intro.inr
      R : Type u_1
      inst✝ : CommRing R
      p q : Polynomial R
      hp : p.Monic
      r : Polynomial R
      hdeg : LE.le (HAdd.hAdd p.natDegree r.natDegree) p.natDegree
      hr : Eq q (HMul.hMul r p)
      hq : Ne q 0
      rzero : Ne r 0
      ⊢ Eq r.natDegree 0
    -/
  · exact (add_right_inj _).1 (le_antisymm hdeg <| Nat.le.intro rfl)
    /-
      🎉 no goals
    -/


lemma eq_of_monic_of_dvd_of_natDegree_le {R} [CommRing R] {p q : R[X]} (hp : p.Monic)
    (hq : q.Monic) (hdiv : p ∣ q) (hdeg : q.natDegree ≤ p.natDegree) : q = p := by
  /-
    R : Type u_1
    inst✝ : CommRing R
    p q : Polynomial R
    hp : p.Monic
    hq : q.Monic
    hdiv : Dvd.dvd p q
    hdeg : LE.le q.natDegree p.natDegree
    ⊢ Eq q p
  -/
  convert eq_leadingCoeff_mul_of_monic_of_dvd_of_natDegree_le hp hdiv hdeg
  /-
    case h.e'_3
    R : Type u_1
    inst✝ : CommRing R
    p q : Polynomial R
    hp : p.Monic
    hq : q.Monic
    hdiv : Dvd.dvd p q
    hdeg : LE.le q.natDegree p.natDegree
    ⊢ Eq p (HMul.hMul (Polynomial.C q.leadingCoeff) p)
  -/
  rw [hq.leadingCoeff, C_1, one_mul]
  /-
    🎉 no goals
  -/


