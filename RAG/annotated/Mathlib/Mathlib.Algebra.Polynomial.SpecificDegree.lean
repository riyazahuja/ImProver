/-- A polynomial of degree 2 or 3 is irreducible iff it doesn't have roots. -/
theorem Monic.irreducible_iff_roots_eq_zero_of_degree_le_three {p : R[X]} (hp : p.Monic)
    (hp2 : 2 ≤ p.natDegree) (hp3 : p.natDegree ≤ 3) : Irreducible p ↔ p.roots = 0 := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    p : Polynomial R
    hp : p.Monic
    hp2 : LE.le 2 p.natDegree
    hp3 : LE.le p.natDegree 3
    ⊢ Iff (Irreducible p) (Eq p.roots 0)
  -/
  have hp0 : p ≠ 0 := hp.ne_zero
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    p : Polynomial R
    hp : p.Monic
    hp2 : LE.le 2 p.natDegree
    hp3 : LE.le p.natDegree 3
    hp0 : Ne p 0
    ⊢ Iff (Irreducible p) (Eq p.roots 0)
  -/
  have hp1 : p ≠ 1 := by rintro rfl; rw [natDegree_one] at hp2; cases hp2
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    p : Polynomial R
    hp : p.Monic
    hp2 : LE.le 2 p.natDegree
    hp3 : LE.le p.natDegree 3
    hp0 : Ne p 0
    hp1 : Ne p 1
    ⊢ Iff (Irreducible p) (Eq p.roots 0)
  -/
  rw [hp.irreducible_iff_lt_natDegree_lt hp1]
  simp_rw [show p.natDegree / 2 = 1 from
      (Nat.div_le_div_right hp3).antisymm
        (by apply Nat.div_le_div_right (c := 2) hp2),
    show Finset.Ioc 0 1 = {1} from rfl,
    Finset.mem_singleton, Multiset.eq_zero_iff_forall_not_mem, mem_roots hp0, ← dvd_iff_isRoot]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    p : Polynomial R
    hp : p.Monic
    hp2 : LE.le 2 p.natDegree
    hp3 : LE.le p.natDegree 3
    hp0 : Ne p 0
    hp1 : Ne p 1
    ⊢ Iff (∀ (q : Polynomial R), q.Monic → Eq q.natDegree 1 → Not (Dvd.dvd q p)) ( …
  -/
  refine ⟨fun h r ↦ h _ (monic_X_sub_C r) (natDegree_X_sub_C r), fun h q hq hq1 ↦ ?_⟩
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    p : Polynomial R
    hp : p.Monic
    hp2 : LE.le 2 p.natDegree
    hp3 : LE.le p.natDegree 3
    hp0 : Ne p 0
    hp1 : Ne p 1
    h : ∀ (a : R), Not (Dvd.dvd (HSub.hSub Polynomial.X (Polynomial.C a)) p)
    q : Polynomial R
    hq : q.Monic
    hq1 : Eq q.natDegree 1
    ⊢ Not (Dvd.dvd q p)
  -/
  rw [hq.eq_X_add_C hq1, ← sub_neg_eq_add, ← C_neg]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    p : Polynomial R
    hp : p.Monic
    hp2 : LE.le 2 p.natDegree
    hp3 : LE.le p.natDegree 3
    hp0 : Ne p 0
    hp1 : Ne p 1
    h : ∀ (a : R), Not (Dvd.dvd (HSub.hSub Polynomial.X (Polynomial.C a)) p)
    q : Polynomial R
    hq : q.Monic
    hq1 : Eq q.natDegree 1
    ⊢ Not (Dvd.dvd (HSub.hSub Polynomial.X (Polynomial.C (Neg.neg (q.coeff 0)))) p)
  -/
  apply h
  /-
    🎉 no goals
  -/


/-- A polynomial of degree 2 or 3 is irreducible iff it doesn't have roots. -/
theorem irreducible_iff_roots_eq_zero_of_degree_le_three
    {p : K[X]} (hp2 : 2 ≤ p.natDegree) (hp3 : p.natDegree ≤ 3) : Irreducible p ↔ p.roots = 0 := by
  /-
    K : Type u_1
    inst✝ : Field K
    p : Polynomial K
    hp2 : LE.le 2 p.natDegree
    hp3 : LE.le p.natDegree 3
    ⊢ Iff (Irreducible p) (Eq p.roots 0)
  -/
  have hp0 : p ≠ 0 := by rintro rfl; rw [natDegree_zero] at hp2; cases hp2
  rw [← irreducible_mul_leadingCoeff_inv,
      (monic_mul_leadingCoeff_inv hp0).irreducible_iff_roots_eq_zero_of_degree_le_three,
      mul_comm, roots_C_mul]
    /-
      case ha
      K : Type u_1
      inst✝ : Field K
      p : Polynomial K
      hp2 : LE.le 2 p.natDegree
      hp3 : LE.le p.natDegree 3
      hp0 : Ne p 0
      ⊢ Ne (Inv.inv p.leadingCoeff) 0
    -/
  · exact inv_ne_zero (leadingCoeff_ne_zero.mpr hp0)
    /-
      🎉 no goals
    -/
    /-
      case hp2
      K : Type u_1
      inst✝ : Field K
      p : Polynomial K
      hp2 : LE.le 2 p.natDegree
      hp3 : LE.le p.natDegree 3
      hp0 : Ne p 0
      ⊢ LE.le 2 (HMul.hMul p (Polynomial.C (Inv.inv p.leadingCoeff))).natDegree
    -/
  · rwa [natDegree_mul_leadingCoeff_inv _ hp0]
    /-
      🎉 no goals
    -/
    /-
      case hp3
      K : Type u_1
      inst✝ : Field K
      p : Polynomial K
      hp2 : LE.le 2 p.natDegree
      hp3 : LE.le p.natDegree 3
      hp0 : Ne p 0
      ⊢ LE.le (HMul.hMul p (Polynomial.C (Inv.inv p.leadingCoeff))).natDegree 3
    -/
  · rwa [natDegree_mul_leadingCoeff_inv _ hp0]
    /-
      🎉 no goals
    -/


