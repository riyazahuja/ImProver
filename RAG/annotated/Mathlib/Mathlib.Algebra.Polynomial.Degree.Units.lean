lemma natDegree_eq_zero_of_isUnit (h : IsUnit p) : natDegree p = 0 := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : NoZeroDivisors R
    p : Polynomial R
    h : IsUnit p
    ⊢ Eq p.natDegree 0
  -/
  nontriviality R
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : NoZeroDivisors R
    p : Polynomial R
    h : IsUnit p
    a✝ : Nontrivial R
    ⊢ Eq p.natDegree 0
  -/
  obtain ⟨q, hq⟩ := h.exists_right_inv
  /-
    case intro
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : NoZeroDivisors R
    p : Polynomial R
    h : IsUnit p
    a✝ : Nontrivial R
    q : Polynomial R
    hq : Eq (HMul.hMul p q) 1
    ⊢ Eq p.natDegree 0
  -/
  have := natDegree_mul (left_ne_zero_of_mul_eq_one hq) (right_ne_zero_of_mul_eq_one hq)
  /-
    case intro
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : NoZeroDivisors R
    p : Polynomial R
    h : IsUnit p
    a✝ : Nontrivial R
    q : Polynomial R
    hq : Eq (HMul.hMul p q) 1
    this : Eq (HMul.hMul p q).natDegree (HAdd.hAdd p.natDegree q.natDegree)
    ⊢ Eq p.natDegree 0
  -/
  rw [hq, natDegree_one, eq_comm, add_eq_zero] at this
  /-
    case intro
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : NoZeroDivisors R
    p : Polynomial R
    h : IsUnit p
    a✝ : Nontrivial R
    q : Polynomial R
    hq : Eq (HMul.hMul p q) 1
    this : And (Eq p.natDegree 0) (Eq q.natDegree 0)
    ⊢ Eq p.natDegree 0
  -/
  exact this.1
  /-
    🎉 no goals
  -/


lemma degree_eq_zero_of_isUnit [Nontrivial R] (h : IsUnit p) : degree p = 0 :=
  (natDegree_eq_zero_iff_degree_le_zero.mp <| natDegree_eq_zero_of_isUnit h).antisymm
    (zero_le_degree_iff.mpr h.ne_zero)


@[simp]
lemma degree_coe_units [Nontrivial R] (u : R[X]ˣ) : degree (u : R[X]) = 0 :=
  degree_eq_zero_of_isUnit ⟨u, rfl⟩


/-- Characterization of a unit of a polynomial ring over an integral domain `R`.
See `Polynomial.isUnit_iff_coeff_isUnit_isNilpotent` when `R` is a commutative ring. -/
lemma isUnit_iff : IsUnit p ↔ ∃ r : R, IsUnit r ∧ C r = p :=
  ⟨fun hp =>
    ⟨p.coeff 0,
      let h := eq_C_of_natDegree_eq_zero (natDegree_eq_zero_of_isUnit hp)
      ⟨isUnit_C.1 (h ▸ hp), h.symm⟩⟩,
    fun ⟨_, hr, hrp⟩ => hrp ▸ isUnit_C.2 hr⟩


lemma not_isUnit_of_degree_pos (p : R[X]) (hpl : 0 < p.degree) : ¬ IsUnit p := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : NoZeroDivisors R
    p : Polynomial R
    hpl : LT.lt 0 p.degree
    ⊢ Not (IsUnit p)
  -/
  cases subsingleton_or_nontrivial R
    /-
      case inl
      R : Type u
      inst✝¹ : Semiring R
      inst✝ : NoZeroDivisors R
      p : Polynomial R
      hpl : LT.lt 0 p.degree
      h✝ : Subsingleton R
      ⊢ Not (IsUnit p)
    -/
  · simp [Subsingleton.elim p 0] at hpl
    /-
      🎉 no goals
    -/
  /-
    case inr
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : NoZeroDivisors R
    p : Polynomial R
    hpl : LT.lt 0 p.degree
    h✝ : Nontrivial R
    ⊢ Not (IsUnit p)
  -/
  intro h
  /-
    case inr
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : NoZeroDivisors R
    p : Polynomial R
    hpl : LT.lt 0 p.degree
    h✝ : Nontrivial R
    h : IsUnit p
    ⊢ False
  -/
  simp [degree_eq_zero_of_isUnit h] at hpl
  /-
    🎉 no goals
  -/


lemma not_isUnit_of_natDegree_pos (p : R[X]) (hpl : 0 < p.natDegree) : ¬ IsUnit p :=
  not_isUnit_of_degree_pos _ (natDegree_pos_iff_degree_pos.mp hpl)


@[simp] lemma natDegree_coe_units (u : R[X]ˣ) : natDegree (u : R[X]) = 0 := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : NoZeroDivisors R
    u : Units (Polynomial R)
    ⊢ Eq (↑u).natDegree 0
  -/
  nontriviality R
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : NoZeroDivisors R
    u : Units (Polynomial R)
    a✝ : Nontrivial R
    ⊢ Eq (↑u).natDegree 0
  -/
  exact natDegree_eq_of_degree_eq_some (degree_coe_units u)
  /-
    🎉 no goals
  -/


theorem coeff_coe_units_zero_ne_zero [Nontrivial R] (u : R[X]ˣ) : coeff (u : R[X]) 0 ≠ 0 := by
  /-
    R : Type u
    inst✝² : Semiring R
    inst✝¹ : NoZeroDivisors R
    inst✝ : Nontrivial R
    u : Units (Polynomial R)
    ⊢ Ne ((↑u).coeff 0) 0
  -/
  conv in 0 => rw [← natDegree_coe_units u]
  /-
    R : Type u
    inst✝² : Semiring R
    inst✝¹ : NoZeroDivisors R
    inst✝ : Nontrivial R
    u : Units (Polynomial R)
    ⊢ Ne ((↑u).coeff (↑u).natDegree) 0
  -/
  rw [← leadingCoeff, Ne, leadingCoeff_eq_zero]
  /-
    R : Type u
    inst✝² : Semiring R
    inst✝¹ : NoZeroDivisors R
    inst✝ : Nontrivial R
    u : Units (Polynomial R)
    ⊢ Not (Eq (↑u) 0)
  -/
  exact Units.ne_zero _
  /-
    🎉 no goals
  -/


lemma Monic.C_dvd_iff_isUnit {a : R} : C a ∣ p ↔ IsUnit a where
  mp h := isUnit_iff_dvd_one.mpr <| hp.coeff_natDegree ▸ (C_dvd_iff_dvd_coeff _ _).mp h p.natDegree
  mpr ha := (ha.map C).dvd


lemma Monic.degree_pos_of_not_isUnit (hu : ¬IsUnit p) : 0 < degree p :=
  hp.degree_pos.mpr fun hp' ↦ (hp' ▸ hu) isUnit_one


lemma Monic.natDegree_pos_of_not_isUnit (hu : ¬IsUnit p) : 0 < natDegree p :=
  hp.natDegree_pos.mpr fun hp' ↦ (hp' ▸ hu) isUnit_one


lemma degree_pos_of_not_isUnit_of_dvd_monic (ha : ¬IsUnit a) (hap : a ∣ p) : 0 < degree a := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    a p : Polynomial R
    hp : p.Monic
    ha : Not (IsUnit a)
    hap : Dvd.dvd a p
    ⊢ LT.lt 0 a.degree
  -/
  contrapose! ha with h
  /-
    R : Type u
    inst✝ : CommSemiring R
    a p : Polynomial R
    hp : p.Monic
    hap : Dvd.dvd a p
    h : LE.le a.degree 0
    ⊢ IsUnit a
  -/
  rw [Polynomial.eq_C_of_degree_le_zero h] at hap ⊢
  /-
    R : Type u
    inst✝ : CommSemiring R
    a p : Polynomial R
    hp : p.Monic
    hap : Dvd.dvd (Polynomial.C (a.coeff 0)) p
    h : LE.le a.degree 0
    ⊢ IsUnit (Polynomial.C (a.coeff 0))
  -/
  simpa [hp.C_dvd_iff_isUnit, isUnit_C] using hap
  /-
    🎉 no goals
  -/


lemma natDegree_pos_of_not_isUnit_of_dvd_monic (ha : ¬IsUnit a) (hap : a ∣ p) : 0 < natDegree a :=
  natDegree_pos_iff_degree_pos.mpr <| degree_pos_of_not_isUnit_of_dvd_monic hp ha hap


