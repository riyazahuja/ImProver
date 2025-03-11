private theorem abc_subcall {a b c w : k[X]} {hw : w ≠ 0} (wab : w = wronskian a b) (ha : a ≠ 0)
    (hb : b ≠ 0) (hc : c ≠ 0) (abc_dr_dvd_w : divRadical (a * b * c) ∣ w) :
      c.natDegree + 1 ≤ (radical (a * b * c)).natDegree := by
  /-
    k : Type u_1
    inst✝¹ : Field k
    inst✝ : DecidableEq k
    a b c w : Polynomial k
    hw : Ne w 0
    wab : Eq w (a.wronskian b)
    ha : Ne a 0
    hb : Ne b 0
    hc : Ne c 0
    abc_dr_dvd_w : Dvd.dvd (EuclideanDomain.divRadical (HMul.hMul (HMul.hMul a b)  …
    ⊢ LE.le (HAdd.hAdd c.natDegree 1) (UniqueFactorizationMonoid.radical (HMul.hMu …
  -/
  have ab_nz := mul_ne_zero ha hb
  /-
    k : Type u_1
    inst✝¹ : Field k
    inst✝ : DecidableEq k
    a b c w : Polynomial k
    hw : Ne w 0
    wab : Eq w (a.wronskian b)
    ha : Ne a 0
    hb : Ne b 0
    hc : Ne c 0
    abc_dr_dvd_w : Dvd.dvd (EuclideanDomain.divRadical (HMul.hMul (HMul.hMul a b)  …
    ab_nz : Ne (HMul.hMul a b) 0
    ⊢ LE.le (HAdd.hAdd c.natDegree 1) (UniqueFactorizationMonoid.radical (HMul.hMu …
  -/
  have abc_nz := mul_ne_zero ab_nz hc
  -- bound the degree of `divRadical (a * b * c)` using Wronskian `w`
  /-
    k : Type u_1
    inst✝¹ : Field k
    inst✝ : DecidableEq k
    a b c w : Polynomial k
    hw : Ne w 0
    wab : Eq w (a.wronskian b)
    ha : Ne a 0
    hb : Ne b 0
    hc : Ne c 0
    abc_dr_dvd_w : Dvd.dvd (EuclideanDomain.divRadical (HMul.hMul (HMul.hMul a b)  …
    ab_nz : Ne (HMul.hMul a b) 0
    abc_nz : Ne (HMul.hMul (HMul.hMul a b) c) 0
    ⊢ LE.le (HAdd.hAdd c.natDegree 1) (UniqueFactorizationMonoid.radical (HMul.hMu …
  -/
  set abc_dr := divRadical (a * b * c)
  have abc_dr_ndeg_lt : abc_dr.natDegree < a.natDegree + b.natDegree := by
    calc
      abc_dr.natDegree ≤ w.natDegree := Polynomial.natDegree_le_of_dvd abc_dr_dvd_w hw
      _ < a.natDegree + b.natDegree := by rw [wab] at hw ⊢; exact natDegree_wronskian_lt_add hw
  -- add the degree of `radical (a * b * c)` to both sides and rearrange
  /-
    k : Type u_1
    inst✝¹ : Field k
    inst✝ : DecidableEq k
    a b c w : Polynomial k
    hw : Ne w 0
    wab : Eq w (a.wronskian b)
    ha : Ne a 0
    hb : Ne b 0
    hc : Ne c 0
    ab_nz : Ne (HMul.hMul a b) 0
    abc_nz : Ne (HMul.hMul (HMul.hMul a b) c) 0
    abc_dr : Polynomial k := EuclideanDomain.divRadical (HMul.hMul (HMul.hMul a b) …
    abc_dr_dvd_w : Dvd.dvd abc_dr w
    abc_dr_ndeg_lt : LT.lt abc_dr.natDegree (HAdd.hAdd a.natDegree b.natDegree)
    ⊢ LE.le (HAdd.hAdd c.natDegree 1) (UniqueFactorizationMonoid.radical (HMul.hMu …
  -/
  set abc_r := radical (a * b * c)
  /-
    k : Type u_1
    inst✝¹ : Field k
    inst✝ : DecidableEq k
    a b c w : Polynomial k
    hw : Ne w 0
    wab : Eq w (a.wronskian b)
    ha : Ne a 0
    hb : Ne b 0
    hc : Ne c 0
    ab_nz : Ne (HMul.hMul a b) 0
    abc_nz : Ne (HMul.hMul (HMul.hMul a b) c) 0
    abc_dr : Polynomial k := EuclideanDomain.divRadical (HMul.hMul (HMul.hMul a b) …
    abc_dr_dvd_w : Dvd.dvd abc_dr w
    abc_dr_ndeg_lt : LT.lt abc_dr.natDegree (HAdd.hAdd a.natDegree b.natDegree)
    abc_r : Polynomial k := UniqueFactorizationMonoid.radical (HMul.hMul (HMul.hMu …
    ⊢ LE.le (HAdd.hAdd c.natDegree 1) abc_r.natDegree
  -/
  apply Nat.lt_of_add_lt_add_left
  calc
    a.natDegree + b.natDegree + c.natDegree = (a * b * c).natDegree := by
      rw [Polynomial.natDegree_mul ab_nz hc, Polynomial.natDegree_mul ha hb]
    _ = ((divRadical (a * b * c)) * (radical (a * b * c))).natDegree := by
      rw [mul_comm _ (radical _), radical_mul_divRadical (a * b * c)]
    _ = abc_dr.natDegree + abc_r.natDegree := by
      rw [← Polynomial.natDegree_mul (divRadical_ne_zero abc_nz) (radical_ne_zero (a * b * c))]
    _ < a.natDegree + b.natDegree + abc_r.natDegree := by
      exact Nat.add_lt_add_right abc_dr_ndeg_lt _


/-- **Polynomial ABC theorem.** -/
theorem Polynomial.abc {a b c : k[X]} (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hab : IsCoprime a b)
    (hbc : IsCoprime b c) (hca : IsCoprime c a) (hsum : a + b + c = 0) :
    ( natDegree a + 1 ≤ (radical (a * b * c)).natDegree ∧
      natDegree b + 1 ≤ (radical (a * b * c)).natDegree ∧
      natDegree c + 1 ≤ (radical (a * b * c)).natDegree ) ∨
      derivative a = 0 ∧ derivative b = 0 ∧ derivative c = 0 := by
  /-
    k : Type u_1
    inst✝¹ : Field k
    inst✝ : DecidableEq k
    a b c : Polynomial k
    ha : Ne a 0
    hb : Ne b 0
    hc : Ne c 0
    hab : IsCoprime a b
    hbc : IsCoprime b c
    hca : IsCoprime c a
    hsum : Eq (HAdd.hAdd (HAdd.hAdd a b) c) 0
    ⊢ Or (And (LE.le (HAdd.hAdd a.natDegree 1) (UniqueFactorizationMonoid.radical  …
  -/
  set w := wronskian a b with wab
  /-
    k : Type u_1
    inst✝¹ : Field k
    inst✝ : DecidableEq k
    a b c : Polynomial k
    ha : Ne a 0
    hb : Ne b 0
    hc : Ne c 0
    hab : IsCoprime a b
    hbc : IsCoprime b c
    hca : IsCoprime c a
    hsum : Eq (HAdd.hAdd (HAdd.hAdd a b) c) 0
    w : Polynomial k := a.wronskian b
    wab : Eq w (a.wronskian b)
    ⊢ Or (And (LE.le (HAdd.hAdd a.natDegree 1) (UniqueFactorizationMonoid.radical  …
  -/
  have wbc : w = wronskian b c := wronskian_eq_of_sum_zero hsum
  have wca : w = wronskian c a := by
    rw [add_rotate] at hsum
    simpa only [← wbc] using wronskian_eq_of_sum_zero hsum
  -- have `divRadical x` dividing `w` for `x = a, b, c`, and use coprimality
  have abc_dr_dvd_w : divRadical (a * b * c) ∣ w := by
    have adr_dvd_w := divRadical_dvd_wronskian_left a b
    have bdr_dvd_w := divRadical_dvd_wronskian_right a b
    have cdr_dvd_w := divRadical_dvd_wronskian_right b c
    rw [← wab] at adr_dvd_w bdr_dvd_w
    rw [← wbc] at cdr_dvd_w
    rw [divRadical_mul (hca.symm.mul_left hbc), divRadical_mul hab]
    exact (hca.divRadical.symm.mul_left hbc.divRadical).mul_dvd
      (hab.divRadical.mul_dvd adr_dvd_w bdr_dvd_w) cdr_dvd_w
  /-
    k : Type u_1
    inst✝¹ : Field k
    inst✝ : DecidableEq k
    a b c : Polynomial k
    ha : Ne a 0
    hb : Ne b 0
    hc : Ne c 0
    hab : IsCoprime a b
    hbc : IsCoprime b c
    hca : IsCoprime c a
    hsum : Eq (HAdd.hAdd (HAdd.hAdd a b) c) 0
    w : Polynomial k := a.wronskian b
    wab : Eq w (a.wronskian b)
    wbc : Eq w (b.wronskian c)
    wca : Eq w (c.wronskian a)
    abc_dr_dvd_w : Dvd.dvd (EuclideanDomain.divRadical (HMul.hMul (HMul.hMul a b)  …
    ⊢ Or (And (LE.le (HAdd.hAdd a.natDegree 1) (UniqueFactorizationMonoid.radical  …
  -/
  by_cases hw : w = 0
    /-
      case pos
      k : Type u_1
      inst✝¹ : Field k
      inst✝ : DecidableEq k
      a b c : Polynomial k
      ha : Ne a 0
      hb : Ne b 0
      hc : Ne c 0
      hab : IsCoprime a b
      hbc : IsCoprime b c
      hca : IsCoprime c a
      hsum : Eq (HAdd.hAdd (HAdd.hAdd a b) c) 0
      w : Polynomial k := a.wronskian b
      wab : Eq w (a.wronskian b)
      wbc : Eq w (b.wronskian c)
      wca : Eq w (c.wronskian a)
      abc_dr_dvd_w : Dvd.dvd (EuclideanDomain.divRadical (HMul.hMul (HMul.hMul a b)  …
      hw : Eq w 0
      ⊢ Or (And (LE.le (HAdd.hAdd a.natDegree 1) (UniqueFactorizationMonoid.radical  …
    -/
  · right
    /-
      case pos.h
      k : Type u_1
      inst✝¹ : Field k
      inst✝ : DecidableEq k
      a b c : Polynomial k
      ha : Ne a 0
      hb : Ne b 0
      hc : Ne c 0
      hab : IsCoprime a b
      hbc : IsCoprime b c
      hca : IsCoprime c a
      hsum : Eq (HAdd.hAdd (HAdd.hAdd a b) c) 0
      w : Polynomial k := a.wronskian b
      wab : Eq w (a.wronskian b)
      wbc : Eq w (b.wronskian c)
      wca : Eq w (c.wronskian a)
      abc_dr_dvd_w : Dvd.dvd (EuclideanDomain.divRadical (HMul.hMul (HMul.hMul a b)  …
      hw : Eq w 0
      ⊢ And (Eq (Polynomial.derivative a) 0) (And (Eq (Polynomial.derivative b) 0) ( …
    -/
    rw [hw] at wab wbc
    /-
      case pos.h
      k : Type u_1
      inst✝¹ : Field k
      inst✝ : DecidableEq k
      a b c : Polynomial k
      ha : Ne a 0
      hb : Ne b 0
      hc : Ne c 0
      hab : IsCoprime a b
      hbc : IsCoprime b c
      hca : IsCoprime c a
      hsum : Eq (HAdd.hAdd (HAdd.hAdd a b) c) 0
      w : Polynomial k := a.wronskian b
      wab : Eq 0 (a.wronskian b)
      wbc : Eq 0 (b.wronskian c)
      wca : Eq w (c.wronskian a)
      abc_dr_dvd_w : Dvd.dvd (EuclideanDomain.divRadical (HMul.hMul (HMul.hMul a b)  …
      hw : Eq w 0
      ⊢ And (Eq (Polynomial.derivative a) 0) (And (Eq (Polynomial.derivative b) 0) ( …
    -/
    cases' hab.wronskian_eq_zero_iff.mp wab.symm with ga gb
    /-
      case pos.h.intro
      k : Type u_1
      inst✝¹ : Field k
      inst✝ : DecidableEq k
      a b c : Polynomial k
      ha : Ne a 0
      hb : Ne b 0
      hc : Ne c 0
      hab : IsCoprime a b
      hbc : IsCoprime b c
      hca : IsCoprime c a
      hsum : Eq (HAdd.hAdd (HAdd.hAdd a b) c) 0
      w : Polynomial k := a.wronskian b
      wab : Eq 0 (a.wronskian b)
      wbc : Eq 0 (b.wronskian c)
      wca : Eq w (c.wronskian a)
      abc_dr_dvd_w : Dvd.dvd (EuclideanDomain.divRadical (HMul.hMul (HMul.hMul a b)  …
      hw : Eq w 0
      ga : Eq (Polynomial.derivative a) 0
      gb : Eq (Polynomial.derivative b) 0
      ⊢ And (Eq (Polynomial.derivative a) 0) (And (Eq (Polynomial.derivative b) 0) ( …
    -/
    cases' hbc.wronskian_eq_zero_iff.mp wbc.symm with _ gc
    /-
      case pos.h.intro.intro
      k : Type u_1
      inst✝¹ : Field k
      inst✝ : DecidableEq k
      a b c : Polynomial k
      ha : Ne a 0
      hb : Ne b 0
      hc : Ne c 0
      hab : IsCoprime a b
      hbc : IsCoprime b c
      hca : IsCoprime c a
      hsum : Eq (HAdd.hAdd (HAdd.hAdd a b) c) 0
      w : Polynomial k := a.wronskian b
      wab : Eq 0 (a.wronskian b)
      wbc : Eq 0 (b.wronskian c)
      wca : Eq w (c.wronskian a)
      abc_dr_dvd_w : Dvd.dvd (EuclideanDomain.divRadical (HMul.hMul (HMul.hMul a b)  …
      hw : Eq w 0
      ga : Eq (Polynomial.derivative a) 0
      gb left✝ : Eq (Polynomial.derivative b) 0
      gc : Eq (Polynomial.derivative c) 0
      ⊢ And (Eq (Polynomial.derivative a) 0) (And (Eq (Polynomial.derivative b) 0) ( …
    -/
    exact ⟨ga, gb, gc⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      k : Type u_1
      inst✝¹ : Field k
      inst✝ : DecidableEq k
      a b c : Polynomial k
      ha : Ne a 0
      hb : Ne b 0
      hc : Ne c 0
      hab : IsCoprime a b
      hbc : IsCoprime b c
      hca : IsCoprime c a
      hsum : Eq (HAdd.hAdd (HAdd.hAdd a b) c) 0
      w : Polynomial k := a.wronskian b
      wab : Eq w (a.wronskian b)
      wbc : Eq w (b.wronskian c)
      wca : Eq w (c.wronskian a)
      abc_dr_dvd_w : Dvd.dvd (EuclideanDomain.divRadical (HMul.hMul (HMul.hMul a b)  …
      hw : Not (Eq w 0)
      ⊢ Or (And (LE.le (HAdd.hAdd a.natDegree 1) (UniqueFactorizationMonoid.radical  …
    -/
  · left
    -- use the subcall three times, using the symmetry in `a, b, c`
    /-
      case neg.h
      k : Type u_1
      inst✝¹ : Field k
      inst✝ : DecidableEq k
      a b c : Polynomial k
      ha : Ne a 0
      hb : Ne b 0
      hc : Ne c 0
      hab : IsCoprime a b
      hbc : IsCoprime b c
      hca : IsCoprime c a
      hsum : Eq (HAdd.hAdd (HAdd.hAdd a b) c) 0
      w : Polynomial k := a.wronskian b
      wab : Eq w (a.wronskian b)
      wbc : Eq w (b.wronskian c)
      wca : Eq w (c.wronskian a)
      abc_dr_dvd_w : Dvd.dvd (EuclideanDomain.divRadical (HMul.hMul (HMul.hMul a b)  …
      hw : Not (Eq w 0)
      ⊢ And (LE.le (HAdd.hAdd a.natDegree 1) (UniqueFactorizationMonoid.radical (HMu …
    -/
    refine ⟨?_, ?_, ?_⟩
      /-
        case neg.h.refine_1
        k : Type u_1
        inst✝¹ : Field k
        inst✝ : DecidableEq k
        a b c : Polynomial k
        ha : Ne a 0
        hb : Ne b 0
        hc : Ne c 0
        hab : IsCoprime a b
        hbc : IsCoprime b c
        hca : IsCoprime c a
        hsum : Eq (HAdd.hAdd (HAdd.hAdd a b) c) 0
        w : Polynomial k := a.wronskian b
        wab : Eq w (a.wronskian b)
        wbc : Eq w (b.wronskian c)
        wca : Eq w (c.wronskian a)
        abc_dr_dvd_w : Dvd.dvd (EuclideanDomain.divRadical (HMul.hMul (HMul.hMul a b)  …
        hw : Not (Eq w 0)
        ⊢ LE.le (HAdd.hAdd a.natDegree 1) (UniqueFactorizationMonoid.radical (HMul.hMu …
      -/
    · rw [mul_rotate] at abc_dr_dvd_w ⊢
      /-
        case neg.h.refine_1
        k : Type u_1
        inst✝¹ : Field k
        inst✝ : DecidableEq k
        a b c : Polynomial k
        ha : Ne a 0
        hb : Ne b 0
        hc : Ne c 0
        hab : IsCoprime a b
        hbc : IsCoprime b c
        hca : IsCoprime c a
        hsum : Eq (HAdd.hAdd (HAdd.hAdd a b) c) 0
        w : Polynomial k := a.wronskian b
        wab : Eq w (a.wronskian b)
        wbc : Eq w (b.wronskian c)
        wca : Eq w (c.wronskian a)
        abc_dr_dvd_w : Dvd.dvd (EuclideanDomain.divRadical (HMul.hMul (HMul.hMul b c)  …
        hw : Not (Eq w 0)
        ⊢ LE.le (HAdd.hAdd a.natDegree 1) (UniqueFactorizationMonoid.radical (HMul.hMu …
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
                                /-
                                  🎉 no goals
                                -/
      apply abc_subcall wbc <;> assumption
                                /-
                                  🎉 no goals
                                -/
      /-
        case neg.h.refine_2
        k : Type u_1
        inst✝¹ : Field k
        inst✝ : DecidableEq k
        a b c : Polynomial k
        ha : Ne a 0
        hb : Ne b 0
        hc : Ne c 0
        hab : IsCoprime a b
        hbc : IsCoprime b c
        hca : IsCoprime c a
        hsum : Eq (HAdd.hAdd (HAdd.hAdd a b) c) 0
        w : Polynomial k := a.wronskian b
        wab : Eq w (a.wronskian b)
        wbc : Eq w (b.wronskian c)
        wca : Eq w (c.wronskian a)
        abc_dr_dvd_w : Dvd.dvd (EuclideanDomain.divRadical (HMul.hMul (HMul.hMul a b)  …
        hw : Not (Eq w 0)
        ⊢ LE.le (HAdd.hAdd b.natDegree 1) (UniqueFactorizationMonoid.radical (HMul.hMu …
      -/
    · rw [← mul_rotate] at abc_dr_dvd_w ⊢
      /-
        case neg.h.refine_2
        k : Type u_1
        inst✝¹ : Field k
        inst✝ : DecidableEq k
        a b c : Polynomial k
        ha : Ne a 0
        hb : Ne b 0
        hc : Ne c 0
        hab : IsCoprime a b
        hbc : IsCoprime b c
        hca : IsCoprime c a
        hsum : Eq (HAdd.hAdd (HAdd.hAdd a b) c) 0
        w : Polynomial k := a.wronskian b
        wab : Eq w (a.wronskian b)
        wbc : Eq w (b.wronskian c)
        wca : Eq w (c.wronskian a)
        abc_dr_dvd_w : Dvd.dvd (EuclideanDomain.divRadical (HMul.hMul (HMul.hMul c a)  …
        hw : Not (Eq w 0)
        ⊢ LE.le (HAdd.hAdd b.natDegree 1) (UniqueFactorizationMonoid.radical (HMul.hMu …
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
                                /-
                                  🎉 no goals
                                -/
      apply abc_subcall wca <;> assumption
                                /-
                                  🎉 no goals
                                -/
      /-
        case neg.h.refine_3
        k : Type u_1
        inst✝¹ : Field k
        inst✝ : DecidableEq k
        a b c : Polynomial k
        ha : Ne a 0
        hb : Ne b 0
        hc : Ne c 0
        hab : IsCoprime a b
        hbc : IsCoprime b c
        hca : IsCoprime c a
        hsum : Eq (HAdd.hAdd (HAdd.hAdd a b) c) 0
        w : Polynomial k := a.wronskian b
        wab : Eq w (a.wronskian b)
        wbc : Eq w (b.wronskian c)
        wca : Eq w (c.wronskian a)
        abc_dr_dvd_w : Dvd.dvd (EuclideanDomain.divRadical (HMul.hMul (HMul.hMul a b)  …
        hw : Not (Eq w 0)
        ⊢ LE.le (HAdd.hAdd c.natDegree 1) (UniqueFactorizationMonoid.radical (HMul.hMu …
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
                                /-
                                  🎉 no goals
                                -/
    · apply abc_subcall wab <;> assumption
                                /-
                                  🎉 no goals
                                -/

