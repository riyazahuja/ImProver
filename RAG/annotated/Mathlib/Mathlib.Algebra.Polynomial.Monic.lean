theorem monic_zero_iff_subsingleton : Monic (0 : R[X]) ↔ Subsingleton R :=
  subsingleton_iff_zero_eq_one


theorem not_monic_zero_iff : ¬Monic (0 : R[X]) ↔ (0 : R) ≠ 1 :=
  (monic_zero_iff_subsingleton.trans subsingleton_iff_zero_eq_one.symm).not


theorem monic_zero_iff_subsingleton' :
    Monic (0 : R[X]) ↔ (∀ f g : R[X], f = g) ∧ ∀ a b : R, a = b :=
  Polynomial.monic_zero_iff_subsingleton.trans
    ⟨by
      /-
        R : Type u
        inst✝ : Semiring R
        ⊢ Subsingleton R → And (∀ (f g : Polynomial R), Eq f g) (∀ (a b : R), Eq a b)
      -/
      intro
      /-
        R : Type u
        inst✝ : Semiring R
        a✝ : Subsingleton R
        ⊢ And (∀ (f g : Polynomial R), Eq f g) (∀ (a b : R), Eq a b)
      -/
      simp [eq_iff_true_of_subsingleton], fun h => subsingleton_iff.mpr h.2⟩
      /-
        🎉 no goals
      -/


theorem Monic.as_sum (hp : p.Monic) :
    p = X ^ p.natDegree + ∑ i ∈ range p.natDegree, C (p.coeff i) * X ^ i := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : p.Monic
    ⊢ Eq p (HAdd.hAdd (HPow.hPow Polynomial.X p.natDegree) ((Finset.range p.natDeg …
  -/
  conv_lhs => rw [p.as_sum_range_C_mul_X_pow, sum_range_succ_comm]
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : p.Monic
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Polynomial.C (p.coeff p.natDegree)) (HPow.hPow Pol …
  -/
  suffices C (p.coeff p.natDegree) = 1 by rw [this, one_mul]
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : p.Monic
    ⊢ Eq (Polynomial.C (p.coeff p.natDegree)) 1
  -/
  exact congr_arg C hp
  /-
    🎉 no goals
  -/


theorem ne_zero_of_ne_zero_of_monic (hp : p ≠ 0) (hq : Monic q) : q ≠ 0 := by
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    hp : Ne p 0
    hq : q.Monic
    ⊢ Ne q 0
  -/
  rintro rfl
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : Ne p 0
    hq : Polynomial.Monic 0
    ⊢ False
  -/
  rw [Monic.def, leadingCoeff_zero] at hq
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : Ne p 0
    hq : Eq 0 1
    ⊢ False
  -/
  rw [← mul_one p, ← C_1, ← hq, C_0, mul_zero] at hp
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : Ne 0 0
    hq : Eq 0 1
    ⊢ False
  -/
  exact hp rfl
  /-
    🎉 no goals
  -/


theorem Monic.map [Semiring S] (f : R →+* S) (hp : Monic p) : Monic (p.map f) := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    p : Polynomial R
    inst✝ : Semiring S
    f : RingHom R S
    hp : p.Monic
    ⊢ (Polynomial.map f p).Monic
  -/
  unfold Monic
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    p : Polynomial R
    inst✝ : Semiring S
    f : RingHom R S
    hp : p.Monic
    ⊢ Eq (Polynomial.map f p).leadingCoeff 1
  -/
  nontriviality
  have : f p.leadingCoeff ≠ 0 := by
    rw [show _ = _ from hp, f.map_one]
    exact one_ne_zero
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    p : Polynomial R
    inst✝ : Semiring S
    f : RingHom R S
    hp : p.Monic
    a✝ : Nontrivial S
    this : Ne (f p.leadingCoeff) 0
    ⊢ Eq (Polynomial.map f p).leadingCoeff 1
  -/
  rw [Polynomial.leadingCoeff, coeff_map]
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    p : Polynomial R
    inst✝ : Semiring S
    f : RingHom R S
    hp : p.Monic
    a✝ : Nontrivial S
    this : Ne (f p.leadingCoeff) 0
    ⊢ Eq (f (p.coeff (Polynomial.map f p).natDegree)) 1
  -/
  suffices p.coeff (p.map f).natDegree = 1 by simp [this]
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    p : Polynomial R
    inst✝ : Semiring S
    f : RingHom R S
    hp : p.Monic
    a✝ : Nontrivial S
    this : Ne (f p.leadingCoeff) 0
    ⊢ Eq (p.coeff (Polynomial.map f p).natDegree) 1
  -/
  rwa [natDegree_eq_of_degree_eq (degree_map_eq_of_leadingCoeff_ne_zero f this)]
  /-
    🎉 no goals
  -/


theorem monic_C_mul_of_mul_leadingCoeff_eq_one {b : R} (hp : b * p.leadingCoeff = 1) :
    Monic (C b * p) := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    b : R
    hp : Eq (HMul.hMul b p.leadingCoeff) 1
    ⊢ (HMul.hMul (Polynomial.C b) p).Monic
  -/
  unfold Monic
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    b : R
    hp : Eq (HMul.hMul b p.leadingCoeff) 1
    ⊢ Eq (HMul.hMul (Polynomial.C b) p).leadingCoeff 1
  -/
  nontriviality
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    b : R
    hp : Eq (HMul.hMul b p.leadingCoeff) 1
    a✝ : Nontrivial R
    ⊢ Eq (HMul.hMul (Polynomial.C b) p).leadingCoeff 1
  -/
                               /-
                                 🎉 no goals
                               -/
  rw [leadingCoeff_mul' _] <;> simp [leadingCoeff_C b, hp]
                               /-
                                 🎉 no goals
                               -/


theorem monic_mul_C_of_leadingCoeff_mul_eq_one {b : R} (hp : p.leadingCoeff * b = 1) :
    Monic (p * C b) := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    b : R
    hp : Eq (HMul.hMul p.leadingCoeff b) 1
    ⊢ (HMul.hMul p (Polynomial.C b)).Monic
  -/
  unfold Monic
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    b : R
    hp : Eq (HMul.hMul p.leadingCoeff b) 1
    ⊢ Eq (HMul.hMul p (Polynomial.C b)).leadingCoeff 1
  -/
  nontriviality
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    b : R
    hp : Eq (HMul.hMul p.leadingCoeff b) 1
    a✝ : Nontrivial R
    ⊢ Eq (HMul.hMul p (Polynomial.C b)).leadingCoeff 1
  -/
                               /-
                                 🎉 no goals
                               -/
  rw [leadingCoeff_mul' _] <;> simp [leadingCoeff_C b, hp]
                               /-
                                 🎉 no goals
                               -/


theorem monic_of_degree_le (n : ℕ) (H1 : degree p ≤ n) (H2 : coeff p n = 1) : Monic p :=
  Decidable.byCases
    (fun H : degree p < n => eq_of_zero_eq_one (H2 ▸ (coeff_eq_zero_of_degree_lt H).symm) _ _)
    fun H : ¬degree p < n => by
    /-
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      n : Nat
      H1 : LE.le p.degree ↑n
      H2 : Eq (p.coeff n) 1
      H : Not (LT.lt p.degree ↑n)
      ⊢ p.Monic
    -/
    rwa [Monic, Polynomial.leadingCoeff, natDegree, (lt_or_eq_of_le H1).resolve_left H]
    /-
      🎉 no goals
    -/


theorem monic_X_pow_add {n : ℕ} (H : degree p < n) : Monic (X ^ n + p) :=
  monic_of_degree_le n
    (le_trans (degree_add_le _ _) (max_le (degree_X_pow_le _) (le_of_lt H)))
        /-
          R : Type u
          inst✝ : Semiring R
          p : Polynomial R
          n : Nat
          H : LT.lt p.degree ↑n
          ⊢ Eq ((HAdd.hAdd (HPow.hPow Polynomial.X n) p).coeff n) 1
        -/
    (by rw [coeff_add, coeff_X_pow, if_pos rfl, coeff_eq_zero_of_degree_lt H, add_zero])
        /-
          🎉 no goals
        -/


variable (a) in
theorem monic_X_pow_add_C {n : ℕ} (h : n ≠ 0) : (X ^ n + C a).Monic :=
   monic_X_pow_add <| (lt_of_le_of_lt degree_C_le
         /-
           R : Type u
           a : R
           inst✝ : Semiring R
           n : Nat
           h : Ne n 0
           ⊢ LT.lt 0 ↑n
         -/
     (by simp only [Nat.cast_pos, Nat.pos_iff_ne_zero, ne_eq, h, not_false_eq_true]))
         /-
           🎉 no goals
         -/


theorem monic_X_add_C (x : R) : Monic (X + C x) :=
  pow_one (X : R[X]) ▸ monic_X_pow_add_C x one_ne_zero


theorem Monic.mul (hp : Monic p) (hq : Monic q) : Monic (p * q) :=
  letI := Classical.decEq R
  if h0 : (0 : R) = 1 then
    haveI := subsingleton_of_zero_eq_one h0
    Subsingleton.elim _ _
  else by
    have : p.leadingCoeff * q.leadingCoeff ≠ 0 := by
      simp [Monic.def.1 hp, Monic.def.1 hq, Ne.symm h0]
    /-
      R : Type u
      inst✝ : Semiring R
      p q : Polynomial R
      hp : p.Monic
      hq : q.Monic
      this✝ : DecidableEq R := Classical.decEq R
      h0 : Not (Eq 0 1)
      this : Ne (HMul.hMul p.leadingCoeff q.leadingCoeff) 0
      ⊢ (HMul.hMul p q).Monic
    -/
    rw [Monic.def, leadingCoeff_mul' this, Monic.def.1 hp, Monic.def.1 hq, one_mul]
    /-
      🎉 no goals
    -/


theorem Monic.pow (hp : Monic p) : ∀ n : ℕ, Monic (p ^ n)
  | 0 => monic_one
  | n + 1 => by
    /-
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      hp : p.Monic
      n : Nat
      ⊢ (HPow.hPow p (HAdd.hAdd n 1)).Monic
    -/
    rw [pow_succ]
    /-
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      hp : p.Monic
      n : Nat
      ⊢ (HMul.hMul (HPow.hPow p n) p).Monic
    -/
    exact (Monic.pow hp n).mul hp
    /-
      🎉 no goals
    -/


theorem Monic.add_of_left (hp : Monic p) (hpq : degree q < degree p) : Monic (p + q) := by
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    hp : p.Monic
    hpq : LT.lt q.degree p.degree
    ⊢ (HAdd.hAdd p q).Monic
  -/
  rwa [Monic, add_comm, leadingCoeff_add_of_degree_lt hpq]
  /-
    🎉 no goals
  -/


theorem Monic.add_of_right (hq : Monic q) (hpq : degree p < degree q) : Monic (p + q) := by
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    hq : q.Monic
    hpq : LT.lt p.degree q.degree
    ⊢ (HAdd.hAdd p q).Monic
  -/
  rwa [Monic, leadingCoeff_add_of_degree_lt hpq]
  /-
    🎉 no goals
  -/


theorem Monic.of_mul_monic_left (hp : p.Monic) (hpq : (p * q).Monic) : q.Monic := by
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    hp : p.Monic
    hpq : (HMul.hMul p q).Monic
    ⊢ q.Monic
  -/
  contrapose! hpq
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    hp : p.Monic
    hpq : Not q.Monic
    ⊢ Not (HMul.hMul p q).Monic
  -/
  rw [Monic.def] at hpq ⊢
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    hp : p.Monic
    hpq : Not (Eq q.leadingCoeff 1)
    ⊢ Not (Eq (HMul.hMul p q).leadingCoeff 1)
  -/
  rwa [leadingCoeff_monic_mul hp]
  /-
    🎉 no goals
  -/


theorem Monic.of_mul_monic_right (hq : q.Monic) (hpq : (p * q).Monic) : p.Monic := by
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    hq : q.Monic
    hpq : (HMul.hMul p q).Monic
    ⊢ p.Monic
  -/
  contrapose! hpq
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    hq : q.Monic
    hpq : Not p.Monic
    ⊢ Not (HMul.hMul p q).Monic
  -/
  rw [Monic.def] at hpq ⊢
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    hq : q.Monic
    hpq : Not (Eq p.leadingCoeff 1)
    ⊢ Not (Eq (HMul.hMul p q).leadingCoeff 1)
  -/
  rwa [leadingCoeff_mul_monic hq]
  /-
    🎉 no goals
  -/


lemma comp (hp : p.Monic) (hq : q.Monic) (h : q.natDegree ≠ 0) : (p.comp q).Monic := by
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    hp : p.Monic
    hq : q.Monic
    h : Ne q.natDegree 0
    ⊢ (p.comp q).Monic
  -/
  nontriviality R
  have : (p.comp q).natDegree = p.natDegree * q.natDegree :=
    natDegree_comp_eq_of_mul_ne_zero <| by simp [hp.leadingCoeff, hq.leadingCoeff]
  rw [Monic.def, Polynomial.leadingCoeff, this, coeff_comp_degree_mul_degree h, hp.leadingCoeff,
    hq.leadingCoeff, one_pow, mul_one]


lemma comp_X_add_C (hp : p.Monic) (r : R) : (p.comp (X + C r)).Monic := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : p.Monic
    r : R
    ⊢ (p.comp (HAdd.hAdd Polynomial.X (Polynomial.C r))).Monic
  -/
  nontriviality R
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : p.Monic
    r : R
    a✝ : Nontrivial R
    ⊢ (p.comp (HAdd.hAdd Polynomial.X (Polynomial.C r))).Monic
  -/
  refine hp.comp (monic_X_add_C _) fun ha ↦ ?_
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : p.Monic
    r : R
    a✝ : Nontrivial R
    ha : Eq (HAdd.hAdd Polynomial.X (Polynomial.C r)).natDegree 0
    ⊢ False
  -/
  rw [natDegree_X_add_C] at ha
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : p.Monic
    r : R
    a✝ : Nontrivial R
    ha : Eq 1 0
    ⊢ False
  -/
  exact one_ne_zero ha
  /-
    🎉 no goals
  -/


@[simp]
theorem natDegree_eq_zero_iff_eq_one (hp : p.Monic) : p.natDegree = 0 ↔ p = 1 := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : p.Monic
    ⊢ Iff (Eq p.natDegree 0) (Eq p 1)
  -/
  constructor <;> intro h
  /-
    case mp
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : p.Monic
    h : Eq p.natDegree 0
    ⊢ Eq p 1
  -/
  swap
    /-
      case mpr
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      hp : p.Monic
      h : Eq p 1
      ⊢ Eq p.natDegree 0
    -/
  · rw [h]
    /-
      case mpr
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      hp : p.Monic
      h : Eq p 1
      ⊢ Eq (Polynomial.natDegree 1) 0
    -/
    exact natDegree_one
    /-
      🎉 no goals
    -/
  have : p = C (p.coeff 0) := by
    rw [← Polynomial.degree_le_zero_iff]
    rwa [Polynomial.natDegree_eq_zero_iff_degree_le_zero] at h
  /-
    case mp
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : p.Monic
    h : Eq p.natDegree 0
    this : Eq p (Polynomial.C (p.coeff 0))
    ⊢ Eq p 1
  -/
  rw [this]
  /-
    case mp
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : p.Monic
    h : Eq p.natDegree 0
    this : Eq p (Polynomial.C (p.coeff 0))
    ⊢ Eq (Polynomial.C (p.coeff 0)) 1
  -/
  rw [← h, ← Polynomial.leadingCoeff, Monic.def.1 hp, C_1]
  /-
    🎉 no goals
  -/


@[simp]
theorem degree_le_zero_iff_eq_one (hp : p.Monic) : p.degree ≤ 0 ↔ p = 1 := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : p.Monic
    ⊢ Iff (LE.le p.degree 0) (Eq p 1)
  -/
  rw [← hp.natDegree_eq_zero_iff_eq_one, natDegree_eq_zero_iff_degree_le_zero]
  /-
    🎉 no goals
  -/


theorem natDegree_mul (hp : p.Monic) (hq : q.Monic) :
    (p * q).natDegree = p.natDegree + q.natDegree := by
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    hp : p.Monic
    hq : q.Monic
    ⊢ Eq (HMul.hMul p q).natDegree (HAdd.hAdd p.natDegree q.natDegree)
  -/
  nontriviality R
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    hp : p.Monic
    hq : q.Monic
    a✝ : Nontrivial R
    ⊢ Eq (HMul.hMul p q).natDegree (HAdd.hAdd p.natDegree q.natDegree)
  -/
  apply natDegree_mul'
  /-
    case h
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    hp : p.Monic
    hq : q.Monic
    a✝ : Nontrivial R
    ⊢ Ne (HMul.hMul p.leadingCoeff q.leadingCoeff) 0
  -/
  simp [hp.leadingCoeff, hq.leadingCoeff]
  /-
    🎉 no goals
  -/


theorem degree_mul_comm (hp : p.Monic) (q : R[X]) : (p * q).degree = (q * p).degree := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : p.Monic
    q : Polynomial R
    ⊢ Eq (HMul.hMul p q).degree (HMul.hMul q p).degree
  -/
  by_cases h : q = 0
    /-
      case pos
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      hp : p.Monic
      q : Polynomial R
      h : Eq q 0
      ⊢ Eq (HMul.hMul p q).degree (HMul.hMul q p).degree
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
    hp : p.Monic
    q : Polynomial R
    h : Not (Eq q 0)
    ⊢ Eq (HMul.hMul p q).degree (HMul.hMul q p).degree
  -/
  rw [degree_mul', hp.degree_mul]
    /-
      case neg
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      hp : p.Monic
      q : Polynomial R
      h : Not (Eq q 0)
      ⊢ Eq (HAdd.hAdd p.degree q.degree) (HAdd.hAdd q.degree p.degree)
    -/
  · exact add_comm _ _
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      hp : p.Monic
      q : Polynomial R
      h : Not (Eq q 0)
      ⊢ Ne (HMul.hMul p.leadingCoeff q.leadingCoeff) 0
    -/
  · rwa [hp.leadingCoeff, one_mul, leadingCoeff_ne_zero]
    /-
      🎉 no goals
    -/


nonrec theorem natDegree_mul' (hp : p.Monic) (hq : q ≠ 0) :
    (p * q).natDegree = p.natDegree + q.natDegree := by
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    hp : p.Monic
    hq : Ne q 0
    ⊢ Eq (HMul.hMul p q).natDegree (HAdd.hAdd p.natDegree q.natDegree)
  -/
  rw [natDegree_mul']
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    hp : p.Monic
    hq : Ne q 0
    ⊢ Ne (HMul.hMul p.leadingCoeff q.leadingCoeff) 0
  -/
  simpa [hp.leadingCoeff, leadingCoeff_ne_zero]
  /-
    🎉 no goals
  -/


theorem natDegree_mul_comm (hp : p.Monic) (q : R[X]) : (p * q).natDegree = (q * p).natDegree := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : p.Monic
    q : Polynomial R
    ⊢ Eq (HMul.hMul p q).natDegree (HMul.hMul q p).natDegree
  -/
  by_cases h : q = 0
    /-
      case pos
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      hp : p.Monic
      q : Polynomial R
      h : Eq q 0
      ⊢ Eq (HMul.hMul p q).natDegree (HMul.hMul q p).natDegree
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
    hp : p.Monic
    q : Polynomial R
    h : Not (Eq q 0)
    ⊢ Eq (HMul.hMul p q).natDegree (HMul.hMul q p).natDegree
  -/
  rw [hp.natDegree_mul' h, Polynomial.natDegree_mul', add_comm]
  /-
    case neg
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : p.Monic
    q : Polynomial R
    h : Not (Eq q 0)
    ⊢ Ne (HMul.hMul q.leadingCoeff p.leadingCoeff) 0
  -/
  simpa [hp.leadingCoeff, leadingCoeff_ne_zero]
  /-
    🎉 no goals
  -/


theorem not_dvd_of_natDegree_lt (hp : Monic p) (h0 : q ≠ 0) (hl : natDegree q < natDegree p) :
    ¬p ∣ q := by
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    hp : p.Monic
    h0 : Ne q 0
    hl : LT.lt q.natDegree p.natDegree
    ⊢ Not (Dvd.dvd p q)
  -/
  rintro ⟨r, rfl⟩
  /-
    case intro
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : p.Monic
    r : Polynomial R
    h0 : Ne (HMul.hMul p r) 0
    hl : LT.lt (HMul.hMul p r).natDegree p.natDegree
    ⊢ False
  -/
  rw [hp.natDegree_mul' <| right_ne_zero_of_mul h0] at hl
  /-
    case intro
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : p.Monic
    r : Polynomial R
    h0 : Ne (HMul.hMul p r) 0
    hl : LT.lt (HAdd.hAdd p.natDegree r.natDegree) p.natDegree
    ⊢ False
  -/
  exact hl.not_le (Nat.le_add_right _ _)
  /-
    🎉 no goals
  -/


theorem not_dvd_of_degree_lt (hp : Monic p) (h0 : q ≠ 0) (hl : degree q < degree p) : ¬p ∣ q :=
  Monic.not_dvd_of_natDegree_lt hp h0 <| natDegree_lt_natDegree h0 hl


theorem nextCoeff_mul (hp : Monic p) (hq : Monic q) :
    nextCoeff (p * q) = nextCoeff p + nextCoeff q := by
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    hp : p.Monic
    hq : q.Monic
    ⊢ Eq (HMul.hMul p q).nextCoeff (HAdd.hAdd p.nextCoeff q.nextCoeff)
  -/
  nontriviality
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    hp : p.Monic
    hq : q.Monic
    a✝ : Nontrivial R
    ⊢ Eq (HMul.hMul p q).nextCoeff (HAdd.hAdd p.nextCoeff q.nextCoeff)
  -/
  simp only [← coeff_one_reverse]
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    hp : p.Monic
    hq : q.Monic
    a✝ : Nontrivial R
    ⊢ Eq ((HMul.hMul p q).reverse.coeff 1) (HAdd.hAdd (p.reverse.coeff 1) (q.rever …
  -/
                       /-
                         🎉 no goals
                       -/
  rw [reverse_mul] <;> simp [hp.leadingCoeff, hq.leadingCoeff, mul_coeff_one, add_comm]
                       /-
                         🎉 no goals
                       -/


theorem nextCoeff_pow (hp : p.Monic) (n : ℕ) : (p ^ n).nextCoeff = n • p.nextCoeff := by
  induction n with
  | zero => rw [pow_zero, zero_smul, ← map_one (f := C), nextCoeff_C_eq_zero]
  | succ n ih => rw [pow_succ, (hp.pow n).nextCoeff_mul hp, ih, succ_nsmul]


theorem eq_one_of_map_eq_one {S : Type*} [Semiring S] [Nontrivial S] (f : R →+* S) (hp : p.Monic)
    (map_eq : p.map f = 1) : p = 1 := by
  /-
    R : Type u
    inst✝² : Semiring R
    p : Polynomial R
    S : Type u_1
    inst✝¹ : Semiring S
    inst✝ : Nontrivial S
    f : RingHom R S
    hp : p.Monic
    map_eq : Eq (Polynomial.map f p) 1
    ⊢ Eq p 1
  -/
  nontriviality R
  have hdeg : p.degree = 0 := by
    rw [← degree_map_eq_of_leadingCoeff_ne_zero f _, map_eq, degree_one]
    · rw [hp.leadingCoeff, f.map_one]
      exact one_ne_zero
  have hndeg : p.natDegree = 0 :=
    WithBot.coe_eq_coe.mp ((degree_eq_natDegree hp.ne_zero).symm.trans hdeg)
  /-
    R : Type u
    inst✝² : Semiring R
    p : Polynomial R
    S : Type u_1
    inst✝¹ : Semiring S
    inst✝ : Nontrivial S
    f : RingHom R S
    hp : p.Monic
    map_eq : Eq (Polynomial.map f p) 1
    a✝ : Nontrivial R
    hdeg : Eq p.degree 0
    hndeg : Eq p.natDegree 0
    ⊢ Eq p 1
  -/
  convert eq_C_of_degree_eq_zero hdeg
  /-
    case h.e'_3
    R : Type u
    inst✝² : Semiring R
    p : Polynomial R
    S : Type u_1
    inst✝¹ : Semiring S
    inst✝ : Nontrivial S
    f : RingHom R S
    hp : p.Monic
    map_eq : Eq (Polynomial.map f p) 1
    a✝ : Nontrivial R
    hdeg : Eq p.degree 0
    hndeg : Eq p.natDegree 0
    ⊢ Eq 1 (Polynomial.C (p.coeff 0))
  -/
  rw [← hndeg, ← Polynomial.leadingCoeff, hp.leadingCoeff, C.map_one]
  /-
    🎉 no goals
  -/


theorem natDegree_pow (hp : p.Monic) (n : ℕ) : (p ^ n).natDegree = n * p.natDegree := by
  induction n with
  | zero => simp
  | succ n hn => rw [pow_succ, (hp.pow n).natDegree_mul hp, hn, Nat.succ_mul, add_comm]


@[simp]
theorem natDegree_pow_X_add_C [Nontrivial R] (n : ℕ) (r : R) : ((X + C r) ^ n).natDegree = n := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : Nontrivial R
    n : Nat
    r : R
    ⊢ Eq (HPow.hPow (HAdd.hAdd Polynomial.X (Polynomial.C r)) n).natDegree n
  -/
  rw [(monic_X_add_C r).natDegree_pow, natDegree_X_add_C, mul_one]
  /-
    🎉 no goals
  -/


theorem Monic.eq_one_of_isUnit (hm : Monic p) (hpu : IsUnit p) : p = 1 := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hm : p.Monic
    hpu : IsUnit p
    ⊢ Eq p 1
  -/
  nontriviality R
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hm : p.Monic
    hpu : IsUnit p
    a✝ : Nontrivial R
    ⊢ Eq p 1
  -/
  obtain ⟨q, h⟩ := hpu.exists_right_inv
  /-
    case intro
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hm : p.Monic
    hpu : IsUnit p
    a✝ : Nontrivial R
    q : Polynomial R
    h : Eq (HMul.hMul p q) 1
    ⊢ Eq p 1
  -/
  have := hm.natDegree_mul' (right_ne_zero_of_mul_eq_one h)
  /-
    case intro
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hm : p.Monic
    hpu : IsUnit p
    a✝ : Nontrivial R
    q : Polynomial R
    h : Eq (HMul.hMul p q) 1
    this : Eq (HMul.hMul p q).natDegree (HAdd.hAdd p.natDegree q.natDegree)
    ⊢ Eq p 1
  -/
  rw [h, natDegree_one, eq_comm, add_eq_zero] at this
  /-
    case intro
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hm : p.Monic
    hpu : IsUnit p
    a✝ : Nontrivial R
    q : Polynomial R
    h : Eq (HMul.hMul p q) 1
    this : And (Eq p.natDegree 0) (Eq q.natDegree 0)
    ⊢ Eq p 1
  -/
  exact hm.natDegree_eq_zero_iff_eq_one.mp this.1
  /-
    🎉 no goals
  -/


theorem Monic.isUnit_iff (hm : p.Monic) : IsUnit p ↔ p = 1 :=
  ⟨hm.eq_one_of_isUnit, fun h => h.symm ▸ isUnit_one⟩


theorem eq_of_monic_of_associated (hp : p.Monic) (hq : q.Monic) (hpq : Associated p q) : p = q := by
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    hp : p.Monic
    hq : q.Monic
    hpq : Associated p q
    ⊢ Eq p q
  -/
  obtain ⟨u, rfl⟩ := hpq
  /-
    case intro
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : p.Monic
    u : Units (Polynomial R)
    hq : (HMul.hMul p ↑u).Monic
    ⊢ Eq p (HMul.hMul p ↑u)
  -/
  rw [(hp.of_mul_monic_left hq).eq_one_of_isUnit u.isUnit, mul_one]
  /-
    🎉 no goals
  -/


theorem monic_multiset_prod_of_monic (t : Multiset ι) (f : ι → R[X]) (ht : ∀ i ∈ t, Monic (f i)) :
    Monic (t.map f).prod := by
  /-
    R : Type u
    ι : Type y
    inst✝ : CommSemiring R
    t : Multiset ι
    f : ι → Polynomial R
    ht : ∀ (i : ι), Membership.mem t i → (f i).Monic
    ⊢ (Multiset.map f t).prod.Monic
  -/
  revert ht
  /-
    R : Type u
    ι : Type y
    inst✝ : CommSemiring R
    t : Multiset ι
    f : ι → Polynomial R
    ⊢ (∀ (i : ι), Membership.mem t i → (f i).Monic) → (Multiset.map f t).prod.Monic
  -/
  refine t.induction_on ?_ ?_; · simp
                                 /-
                                   🎉 no goals
                                 -/
  /-
    case refine_2
    R : Type u
    ι : Type y
    inst✝ : CommSemiring R
    t : Multiset ι
    f : ι → Polynomial R
    ⊢ ∀ (a : ι) (s : Multiset ι), ((∀ (i : ι), Membership.mem s i → (f i).Monic) → …
  -/
  intro a t ih ht
  /-
    case refine_2
    R : Type u
    ι : Type y
    inst✝ : CommSemiring R
    t✝ : Multiset ι
    f : ι → Polynomial R
    a : ι
    t : Multiset ι
    ih : (∀ (i : ι), Membership.mem t i → (f i).Monic) → (Multiset.map f t).prod.M …
    ht : ∀ (i : ι), Membership.mem (Multiset.cons a t) i → (f i).Monic
    ⊢ (Multiset.map f (Multiset.cons a t)).prod.Monic
  -/
  rw [Multiset.map_cons, Multiset.prod_cons]
  /-
    case refine_2
    R : Type u
    ι : Type y
    inst✝ : CommSemiring R
    t✝ : Multiset ι
    f : ι → Polynomial R
    a : ι
    t : Multiset ι
    ih : (∀ (i : ι), Membership.mem t i → (f i).Monic) → (Multiset.map f t).prod.M …
    ht : ∀ (i : ι), Membership.mem (Multiset.cons a t) i → (f i).Monic
    ⊢ (HMul.hMul (f a) (Multiset.map f t).prod).Monic
  -/
  exact (ht _ (Multiset.mem_cons_self _ _)).mul (ih fun _ hi => ht _ (Multiset.mem_cons_of_mem hi))
  /-
    🎉 no goals
  -/


theorem monic_prod_of_monic (s : Finset ι) (f : ι → R[X]) (hs : ∀ i ∈ s, Monic (f i)) :
    Monic (∏ i ∈ s, f i) :=
  monic_multiset_prod_of_monic s.1 f hs


theorem Monic.nextCoeff_multiset_prod (t : Multiset ι) (f : ι → R[X]) (h : ∀ i ∈ t, Monic (f i)) :
    nextCoeff (t.map f).prod = (t.map fun i => nextCoeff (f i)).sum := by
  /-
    R : Type u
    ι : Type y
    inst✝ : CommSemiring R
    t : Multiset ι
    f : ι → Polynomial R
    h : ∀ (i : ι), Membership.mem t i → (f i).Monic
    ⊢ Eq (Multiset.map f t).prod.nextCoeff (Multiset.map (fun i => (f i).nextCoeff …
  -/
  revert h
  /-
    R : Type u
    ι : Type y
    inst✝ : CommSemiring R
    t : Multiset ι
    f : ι → Polynomial R
    ⊢ (∀ (i : ι), Membership.mem t i → (f i).Monic) → Eq (Multiset.map f t).prod.n …
  -/
  refine Multiset.induction_on t ?_ fun a t ih ht => ?_
  · simp only [Multiset.not_mem_zero, forall_prop_of_true, forall_prop_of_false, Multiset.map_zero,
      Multiset.prod_zero, Multiset.sum_zero, not_false_iff, forall_true_iff]
    /-
      case refine_1
      R : Type u
      ι : Type y
      inst✝ : CommSemiring R
      t : Multiset ι
      f : ι → Polynomial R
      ⊢ Eq (Polynomial.nextCoeff 1) 0
    -/
    rw [← C_1]
    /-
      case refine_1
      R : Type u
      ι : Type y
      inst✝ : CommSemiring R
      t : Multiset ι
      f : ι → Polynomial R
      ⊢ Eq (Polynomial.C 1).nextCoeff 0
    -/
    rw [nextCoeff_C_eq_zero]
    /-
      🎉 no goals
    -/
  · rw [Multiset.map_cons, Multiset.prod_cons, Multiset.map_cons, Multiset.sum_cons,
      Monic.nextCoeff_mul, ih]
    exacts [fun i hi => ht i (Multiset.mem_cons_of_mem hi), ht a (Multiset.mem_cons_self _ _),
      monic_multiset_prod_of_monic _ _ fun b bs => ht _ (Multiset.mem_cons_of_mem bs)]


theorem Monic.nextCoeff_prod (s : Finset ι) (f : ι → R[X]) (h : ∀ i ∈ s, Monic (f i)) :
    nextCoeff (∏ i ∈ s, f i) = ∑ i ∈ s, nextCoeff (f i) :=
  Monic.nextCoeff_multiset_prod s.1 f h


lemma irreducible_of_monic (hp : p.Monic) (hp1 : p ≠ 1) :
    Irreducible p ↔ ∀ f g : R[X], f.Monic → g.Monic → f * g = p → f = 1 ∨ g = 1 := by
  refine
    ⟨fun h f g hf hg hp => (h.2 f g hp.symm).imp hf.eq_one_of_isUnit hg.eq_one_of_isUnit, fun h =>
      ⟨hp1 ∘ hp.eq_one_of_isUnit, fun f g hfg =>
        (h (g * C f.leadingCoeff) (f * C g.leadingCoeff) ?_ ?_ ?_).symm.imp
          (isUnit_of_mul_eq_one f _)
          (isUnit_of_mul_eq_one g _)⟩⟩
    /-
      case refine_1
      R : Type u
      inst✝¹ : CommSemiring R
      inst✝ : NoZeroDivisors R
      p : Polynomial R
      hp : p.Monic
      hp1 : Ne p 1
      h : ∀ (f g : Polynomial R), f.Monic → g.Monic → Eq (HMul.hMul f g) p → Or (Eq  …
      f g : Polynomial R
      hfg : Eq p (HMul.hMul f g)
      ⊢ (HMul.hMul g (Polynomial.C f.leadingCoeff)).Monic
    -/
  · rwa [Monic, leadingCoeff_mul, leadingCoeff_C, ← leadingCoeff_mul, mul_comm, ← hfg, ← Monic]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u
      inst✝¹ : CommSemiring R
      inst✝ : NoZeroDivisors R
      p : Polynomial R
      hp : p.Monic
      hp1 : Ne p 1
      h : ∀ (f g : Polynomial R), f.Monic → g.Monic → Eq (HMul.hMul f g) p → Or (Eq  …
      f g : Polynomial R
      hfg : Eq p (HMul.hMul f g)
      ⊢ (HMul.hMul f (Polynomial.C g.leadingCoeff)).Monic
    -/
  · rwa [Monic, leadingCoeff_mul, leadingCoeff_C, ← leadingCoeff_mul, ← hfg, ← Monic]
    /-
      🎉 no goals
    -/
  · rw [mul_mul_mul_comm, ← C_mul, ← leadingCoeff_mul, ← hfg, hp.leadingCoeff, C_1, mul_one,
      mul_comm, ← hfg]



lemma Monic.irreducible_iff_natDegree (hp : p.Monic) :
    Irreducible p ↔
      p ≠ 1 ∧ ∀ f g : R[X], f.Monic → g.Monic → f * g = p → f.natDegree = 0 ∨ g.natDegree = 0 := by
  /-
    R : Type u
    inst✝¹ : CommSemiring R
    inst✝ : NoZeroDivisors R
    p : Polynomial R
    hp : p.Monic
    ⊢ Iff (Irreducible p) (And (Ne p 1) (∀ (f g : Polynomial R), f.Monic → g.Monic …
  -/
  by_cases hp1 : p = 1; · simp [hp1]
                          /-
                            🎉 no goals
                          -/
  /-
    case neg
    R : Type u
    inst✝¹ : CommSemiring R
    inst✝ : NoZeroDivisors R
    p : Polynomial R
    hp : p.Monic
    hp1 : Not (Eq p 1)
    ⊢ Iff (Irreducible p) (And (Ne p 1) (∀ (f g : Polynomial R), f.Monic → g.Monic …
  -/
  rw [irreducible_of_monic hp hp1, and_iff_right hp1]
  /-
    case neg
    R : Type u
    inst✝¹ : CommSemiring R
    inst✝ : NoZeroDivisors R
    p : Polynomial R
    hp : p.Monic
    hp1 : Not (Eq p 1)
    ⊢ Iff (∀ (f g : Polynomial R), f.Monic → g.Monic → Eq (HMul.hMul f g) p → Or ( …
  -/
  refine forall₄_congr fun a b ha hb => ?_
  /-
    case neg
    R : Type u
    inst✝¹ : CommSemiring R
    inst✝ : NoZeroDivisors R
    p : Polynomial R
    hp : p.Monic
    hp1 : Not (Eq p 1)
    a b : Polynomial R
    ha : a.Monic
    hb : b.Monic
    ⊢ Iff (Eq (HMul.hMul a b) p → Or (Eq a 1) (Eq b 1)) (Eq (HMul.hMul a b) p → Or …
  -/
  rw [ha.natDegree_eq_zero_iff_eq_one, hb.natDegree_eq_zero_iff_eq_one]
  /-
    🎉 no goals
  -/


lemma Monic.irreducible_iff_natDegree' (hp : p.Monic) : Irreducible p ↔ p ≠ 1 ∧
    ∀ f g : R[X], f.Monic → g.Monic → f * g = p → g.natDegree ∉ Ioc 0 (p.natDegree / 2) := by
  /-
    R : Type u
    inst✝¹ : CommSemiring R
    inst✝ : NoZeroDivisors R
    p : Polynomial R
    hp : p.Monic
    ⊢ Iff (Irreducible p) (And (Ne p 1) (∀ (f g : Polynomial R), f.Monic → g.Monic …
  -/
  simp_rw [hp.irreducible_iff_natDegree, mem_Ioc, Nat.le_div_iff_mul_le zero_lt_two, mul_two]
  /-
    R : Type u
    inst✝¹ : CommSemiring R
    inst✝ : NoZeroDivisors R
    p : Polynomial R
    hp : p.Monic
    ⊢ Iff (And (Ne p 1) (∀ (f g : Polynomial R), f.Monic → g.Monic → Eq (HMul.hMul …
  -/
  apply and_congr_right'
  /-
    case h
    R : Type u
    inst✝¹ : CommSemiring R
    inst✝ : NoZeroDivisors R
    p : Polynomial R
    hp : p.Monic
    ⊢ Iff (∀ (f g : Polynomial R), f.Monic → g.Monic → Eq (HMul.hMul f g) p → Or ( …
  -/
  constructor <;> intro h f g hf hg he <;> subst he
    /-
      case h.mp
      R : Type u
      inst✝¹ : CommSemiring R
      inst✝ : NoZeroDivisors R
      f g : Polynomial R
      hf : f.Monic
      hg : g.Monic
      hp : (HMul.hMul f g).Monic
      h : ∀ (f_1 g_1 : Polynomial R), f_1.Monic → g_1.Monic → Eq (HMul.hMul f_1 g_1) …
      ⊢ Not (And (LT.lt 0 g.natDegree) (LE.le (HAdd.hAdd g.natDegree g.natDegree) (H …
    -/
  · rw [hf.natDegree_mul hg, add_le_add_iff_right]
    /-
      case h.mp
      R : Type u
      inst✝¹ : CommSemiring R
      inst✝ : NoZeroDivisors R
      f g : Polynomial R
      hf : f.Monic
      hg : g.Monic
      hp : (HMul.hMul f g).Monic
      h : ∀ (f_1 g_1 : Polynomial R), f_1.Monic → g_1.Monic → Eq (HMul.hMul f_1 g_1) …
      ⊢ Not (And (LT.lt 0 g.natDegree) (LE.le g.natDegree f.natDegree))
    -/
    exact fun ha => (h f g hf hg rfl).elim (ha.1.trans_le ha.2).ne' ha.1.ne'
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      R : Type u
      inst✝¹ : CommSemiring R
      inst✝ : NoZeroDivisors R
      f g : Polynomial R
      hf : f.Monic
      hg : g.Monic
      hp : (HMul.hMul f g).Monic
      h : ∀ (f_1 g_1 : Polynomial R), f_1.Monic → g_1.Monic → Eq (HMul.hMul f_1 g_1) …
      ⊢ Or (Eq f.natDegree 0) (Eq g.natDegree 0)
    -/
  · simp_rw [hf.natDegree_mul hg, pos_iff_ne_zero] at h
    /-
      case h.mpr
      R : Type u
      inst✝¹ : CommSemiring R
      inst✝ : NoZeroDivisors R
      f g : Polynomial R
      hf : f.Monic
      hg : g.Monic
      hp : (HMul.hMul f g).Monic
      h : ∀ (f_1 g_1 : Polynomial R), f_1.Monic → g_1.Monic → Eq (HMul.hMul f_1 g_1) …
      ⊢ Or (Eq f.natDegree 0) (Eq g.natDegree 0)
    -/
    contrapose! h
    /-
      case h.mpr
      R : Type u
      inst✝¹ : CommSemiring R
      inst✝ : NoZeroDivisors R
      f g : Polynomial R
      hf : f.Monic
      hg : g.Monic
      hp : (HMul.hMul f g).Monic
      h : And (Ne f.natDegree 0) (Ne g.natDegree 0)
      ⊢ Exists fun f_1 => Exists fun g_1 => And f_1.Monic (And g_1.Monic (And (Eq (H …
    -/
    obtain hl | hl := le_total f.natDegree g.natDegree
      /-
        case h.mpr.inl
        R : Type u
        inst✝¹ : CommSemiring R
        inst✝ : NoZeroDivisors R
        f g : Polynomial R
        hf : f.Monic
        hg : g.Monic
        hp : (HMul.hMul f g).Monic
        h : And (Ne f.natDegree 0) (Ne g.natDegree 0)
        hl : LE.le f.natDegree g.natDegree
        ⊢ Exists fun f_1 => Exists fun g_1 => And f_1.Monic (And g_1.Monic (And (Eq (H …
      -/
    · exact ⟨g, f, hg, hf, mul_comm g f, h.1, add_le_add_left hl _⟩
      /-
        🎉 no goals
      -/
      /-
        case h.mpr.inr
        R : Type u
        inst✝¹ : CommSemiring R
        inst✝ : NoZeroDivisors R
        f g : Polynomial R
        hf : f.Monic
        hg : g.Monic
        hp : (HMul.hMul f g).Monic
        h : And (Ne f.natDegree 0) (Ne g.natDegree 0)
        hl : LE.le g.natDegree f.natDegree
        ⊢ Exists fun f_1 => Exists fun g_1 => And f_1.Monic (And g_1.Monic (And (Eq (H …
      -/
    · exact ⟨f, g, hf, hg, rfl, h.2, add_le_add_right hl _⟩
      /-
        🎉 no goals
      -/


/-- Alternate phrasing of `Polynomial.Monic.irreducible_iff_natDegree'` where we only have to check
one divisor at a time. -/
lemma Monic.irreducible_iff_lt_natDegree_lt {p : R[X]} (hp : p.Monic) (hp1 : p ≠ 1) :
    Irreducible p ↔ ∀ q, Monic q → natDegree q ∈ Finset.Ioc 0 (natDegree p / 2) → ¬ q ∣ p := by
  /-
    R : Type u
    inst✝¹ : CommSemiring R
    inst✝ : NoZeroDivisors R
    p : Polynomial R
    hp : p.Monic
    hp1 : Ne p 1
    ⊢ Iff (Irreducible p) (∀ (q : Polynomial R), q.Monic → Membership.mem (Finset. …
  -/
  rw [hp.irreducible_iff_natDegree', and_iff_right hp1]
  /-
    R : Type u
    inst✝¹ : CommSemiring R
    inst✝ : NoZeroDivisors R
    p : Polynomial R
    hp : p.Monic
    hp1 : Ne p 1
    ⊢ Iff (∀ (f g : Polynomial R), f.Monic → g.Monic → Eq (HMul.hMul f g) p → Not  …
  -/
  constructor
    /-
      case mp
      R : Type u
      inst✝¹ : CommSemiring R
      inst✝ : NoZeroDivisors R
      p : Polynomial R
      hp : p.Monic
      hp1 : Ne p 1
      ⊢ (∀ (f g : Polynomial R), f.Monic → g.Monic → Eq (HMul.hMul f g) p → Not (Mem …
    -/
  · rintro h g hg hdg ⟨f, rfl⟩
    /-
      case mp.intro
      R : Type u
      inst✝¹ : CommSemiring R
      inst✝ : NoZeroDivisors R
      g : Polynomial R
      hg : g.Monic
      f : Polynomial R
      hp : (HMul.hMul g f).Monic
      hp1 : Ne (HMul.hMul g f) 1
      h : ∀ (f_1 g_1 : Polynomial R), f_1.Monic → g_1.Monic → Eq (HMul.hMul f_1 g_1) …
      hdg : Membership.mem (Finset.Ioc 0 (HDiv.hDiv (HMul.hMul g f).natDegree 2)) g. …
      ⊢ False
    -/
    exact h f g (hg.of_mul_monic_left hp) hg (mul_comm f g) hdg
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u
      inst✝¹ : CommSemiring R
      inst✝ : NoZeroDivisors R
      p : Polynomial R
      hp : p.Monic
      hp1 : Ne p 1
      ⊢ (∀ (q : Polynomial R), q.Monic → Membership.mem (Finset.Ioc 0 (HDiv.hDiv p.n …
    -/
  · rintro h f g - hg rfl hdg
    /-
      case mpr
      R : Type u
      inst✝¹ : CommSemiring R
      inst✝ : NoZeroDivisors R
      f g : Polynomial R
      hg : g.Monic
      hp : (HMul.hMul f g).Monic
      hp1 : Ne (HMul.hMul f g) 1
      h : ∀ (q : Polynomial R), q.Monic → Membership.mem (Finset.Ioc 0 (HDiv.hDiv (H …
      hdg : Membership.mem (Finset.Ioc 0 (HDiv.hDiv (HMul.hMul f g).natDegree 2)) g. …
      ⊢ False
    -/
    exact h g hg hdg (dvd_mul_left g f)
    /-
      🎉 no goals
    -/


lemma Monic.not_irreducible_iff_exists_add_mul_eq_coeff (hm : p.Monic) (hnd : p.natDegree = 2) :
    ¬Irreducible p ↔ ∃ c₁ c₂, p.coeff 0 = c₁ * c₂ ∧ p.coeff 1 = c₁ + c₂ := by
  /-
    R : Type u
    inst✝¹ : CommSemiring R
    inst✝ : NoZeroDivisors R
    p : Polynomial R
    hm : p.Monic
    hnd : Eq p.natDegree 2
    ⊢ Iff (Not (Irreducible p)) (Exists fun c₁ => Exists fun c₂ => And (Eq (p.coef …
  -/
  cases subsingleton_or_nontrivial R
    /-
      case inl
      R : Type u
      inst✝¹ : CommSemiring R
      inst✝ : NoZeroDivisors R
      p : Polynomial R
      hm : p.Monic
      hnd : Eq p.natDegree 2
      h✝ : Subsingleton R
      ⊢ Iff (Not (Irreducible p)) (Exists fun c₁ => Exists fun c₂ => And (Eq (p.coef …
    -/
  · simp [natDegree_of_subsingleton] at hnd
    /-
      🎉 no goals
    -/
  /-
    case inr
    R : Type u
    inst✝¹ : CommSemiring R
    inst✝ : NoZeroDivisors R
    p : Polynomial R
    hm : p.Monic
    hnd : Eq p.natDegree 2
    h✝ : Nontrivial R
    ⊢ Iff (Not (Irreducible p)) (Exists fun c₁ => Exists fun c₂ => And (Eq (p.coef …
  -/
  rw [hm.irreducible_iff_natDegree', and_iff_right, hnd]
    /-
      case inr
      R : Type u
      inst✝¹ : CommSemiring R
      inst✝ : NoZeroDivisors R
      p : Polynomial R
      hm : p.Monic
      hnd : Eq p.natDegree 2
      h✝ : Nontrivial R
      ⊢ Iff (Not (∀ (f g : Polynomial R), f.Monic → g.Monic → Eq (HMul.hMul f g) p → …
    -/
  · push_neg
    /-
      case inr
      R : Type u
      inst✝¹ : CommSemiring R
      inst✝ : NoZeroDivisors R
      p : Polynomial R
      hm : p.Monic
      hnd : Eq p.natDegree 2
      h✝ : Nontrivial R
      ⊢ Iff (Exists fun f => Exists fun g => And f.Monic (And g.Monic (And (Eq (HMul …
    -/
    constructor
      /-
        case inr.mp
        R : Type u
        inst✝¹ : CommSemiring R
        inst✝ : NoZeroDivisors R
        p : Polynomial R
        hm : p.Monic
        hnd : Eq p.natDegree 2
        h✝ : Nontrivial R
        ⊢ (Exists fun f => Exists fun g => And f.Monic (And g.Monic (And (Eq (HMul.hMu …
      -/
    · rintro ⟨a, b, ha, hb, rfl, hdb⟩
      /-
        case inr.mp.intro.intro.intro.intro.intro
        R : Type u
        inst✝¹ : CommSemiring R
        inst✝ : NoZeroDivisors R
        h✝ : Nontrivial R
        a b : Polynomial R
        ha : a.Monic
        hb : b.Monic
        hdb : Membership.mem (Finset.Ioc 0 (2 / 2)) b.natDegree
        hm : (HMul.hMul a b).Monic
        hnd : Eq (HMul.hMul a b).natDegree 2
        ⊢ Exists fun c₁ => Exists fun c₂ => And (Eq ((HMul.hMul a b).coeff 0) (HMul.hM …
      -/
      simp only [zero_lt_two, Nat.div_self, Nat.Ioc_succ_singleton, zero_add, mem_singleton] at hdb
      /-
        case inr.mp.intro.intro.intro.intro.intro
        R : Type u
        inst✝¹ : CommSemiring R
        inst✝ : NoZeroDivisors R
        h✝ : Nontrivial R
        a b : Polynomial R
        ha : a.Monic
        hb : b.Monic
        hm : (HMul.hMul a b).Monic
        hnd : Eq (HMul.hMul a b).natDegree 2
        hdb : Eq b.natDegree 1
        ⊢ Exists fun c₁ => Exists fun c₂ => And (Eq ((HMul.hMul a b).coeff 0) (HMul.hM …
      -/
      have hda := hnd
      /-
        case inr.mp.intro.intro.intro.intro.intro
        R : Type u
        inst✝¹ : CommSemiring R
        inst✝ : NoZeroDivisors R
        h✝ : Nontrivial R
        a b : Polynomial R
        ha : a.Monic
        hb : b.Monic
        hm : (HMul.hMul a b).Monic
        hnd : Eq (HMul.hMul a b).natDegree 2
        hdb : Eq b.natDegree 1
        hda : Eq (HMul.hMul a b).natDegree 2
        ⊢ Exists fun c₁ => Exists fun c₂ => And (Eq ((HMul.hMul a b).coeff 0) (HMul.hM …
      -/
      rw [ha.natDegree_mul hb, hdb] at hda
      /-
        case inr.mp.intro.intro.intro.intro.intro
        R : Type u
        inst✝¹ : CommSemiring R
        inst✝ : NoZeroDivisors R
        h✝ : Nontrivial R
        a b : Polynomial R
        ha : a.Monic
        hb : b.Monic
        hm : (HMul.hMul a b).Monic
        hnd : Eq (HMul.hMul a b).natDegree 2
        hdb : Eq b.natDegree 1
        hda : Eq (HAdd.hAdd a.natDegree 1) 2
        ⊢ Exists fun c₁ => Exists fun c₂ => And (Eq ((HMul.hMul a b).coeff 0) (HMul.hM …
      -/
      use a.coeff 0, b.coeff 0, mul_coeff_zero a b
      /-
        case right
        R : Type u
        inst✝¹ : CommSemiring R
        inst✝ : NoZeroDivisors R
        h✝ : Nontrivial R
        a b : Polynomial R
        ha : a.Monic
        hb : b.Monic
        hm : (HMul.hMul a b).Monic
        hnd : Eq (HMul.hMul a b).natDegree 2
        hdb : Eq b.natDegree 1
        hda : Eq (HAdd.hAdd a.natDegree 1) 2
        ⊢ Eq ((HMul.hMul a b).coeff 1) (HAdd.hAdd (a.coeff 0) (b.coeff 0))
      -/
      simpa only [nextCoeff, hnd, add_right_cancel hda, hdb] using ha.nextCoeff_mul hb
      /-
        🎉 no goals
      -/
      /-
        case inr.mpr
        R : Type u
        inst✝¹ : CommSemiring R
        inst✝ : NoZeroDivisors R
        p : Polynomial R
        hm : p.Monic
        hnd : Eq p.natDegree 2
        h✝ : Nontrivial R
        ⊢ (Exists fun c₁ => Exists fun c₂ => And (Eq (p.coeff 0) (HMul.hMul c₁ c₂)) (E …
      -/
    · rintro ⟨c₁, c₂, hmul, hadd⟩
      refine
        ⟨X + C c₁, X + C c₂, monic_X_add_C _, monic_X_add_C _, ?_, ?_⟩
      · rw [p.as_sum_range_C_mul_X_pow, hnd, Finset.sum_range_succ, Finset.sum_range_succ,
          Finset.sum_range_one, ← hnd, hm.coeff_natDegree, hnd, hmul, hadd, C_mul, C_add, C_1]
        /-
          case inr.mpr.intro.intro.intro.refine_1
          R : Type u
          inst✝¹ : CommSemiring R
          inst✝ : NoZeroDivisors R
          p : Polynomial R
          hm : p.Monic
          hnd : Eq p.natDegree 2
          h✝ : Nontrivial R
          c₁ c₂ : R
          hmul : Eq (p.coeff 0) (HMul.hMul c₁ c₂)
          hadd : Eq (p.coeff 1) (HAdd.hAdd c₁ c₂)
          ⊢ Eq (HMul.hMul (HAdd.hAdd Polynomial.X (Polynomial.C c₁)) (HAdd.hAdd Polynomi …
        -/
        ring
        /-
          🎉 no goals
        -/
        /-
          case inr.mpr.intro.intro.intro.refine_2
          R : Type u
          inst✝¹ : CommSemiring R
          inst✝ : NoZeroDivisors R
          p : Polynomial R
          hm : p.Monic
          hnd : Eq p.natDegree 2
          h✝ : Nontrivial R
          c₁ c₂ : R
          hmul : Eq (p.coeff 0) (HMul.hMul c₁ c₂)
          hadd : Eq (p.coeff 1) (HAdd.hAdd c₁ c₂)
          ⊢ Membership.mem (Finset.Ioc 0 (2 / 2)) (HAdd.hAdd Polynomial.X (Polynomial.C  …
        -/
      · rw [mem_Ioc, natDegree_X_add_C _]
        /-
          case inr.mpr.intro.intro.intro.refine_2
          R : Type u
          inst✝¹ : CommSemiring R
          inst✝ : NoZeroDivisors R
          p : Polynomial R
          hm : p.Monic
          hnd : Eq p.natDegree 2
          h✝ : Nontrivial R
          c₁ c₂ : R
          hmul : Eq (p.coeff 0) (HMul.hMul c₁ c₂)
          hadd : Eq (p.coeff 1) (HAdd.hAdd c₁ c₂)
          ⊢ And (LT.lt 0 1) (LE.le 1 (2 / 2))
        -/
        simp
        /-
          🎉 no goals
        -/
    /-
      case inr
      R : Type u
      inst✝¹ : CommSemiring R
      inst✝ : NoZeroDivisors R
      p : Polynomial R
      hm : p.Monic
      hnd : Eq p.natDegree 2
      h✝ : Nontrivial R
      ⊢ Ne p 1
    -/
  · rintro rfl
    /-
      case inr
      R : Type u
      inst✝¹ : CommSemiring R
      inst✝ : NoZeroDivisors R
      h✝ : Nontrivial R
      hm : Polynomial.Monic 1
      hnd : Eq (Polynomial.natDegree 1) 2
      ⊢ False
    -/
    simp [natDegree_one] at hnd
    /-
      🎉 no goals
    -/


@[simp]
theorem Monic.natDegree_map [Semiring S] [Nontrivial S] {P : R[X]} (hmo : P.Monic) (f : R →+* S) :
    (P.map f).natDegree = P.natDegree := by
  /-
    R : Type u
    S : Type v
    inst✝² : Semiring R
    inst✝¹ : Semiring S
    inst✝ : Nontrivial S
    P : Polynomial R
    hmo : P.Monic
    f : RingHom R S
    ⊢ Eq (Polynomial.map f P).natDegree P.natDegree
  -/
  refine le_antisymm natDegree_map_le (le_natDegree_of_ne_zero ?_)
  /-
    R : Type u
    S : Type v
    inst✝² : Semiring R
    inst✝¹ : Semiring S
    inst✝ : Nontrivial S
    P : Polynomial R
    hmo : P.Monic
    f : RingHom R S
    ⊢ Ne ((Polynomial.map f P).coeff P.natDegree) 0
  -/
  rw [coeff_map, Monic.coeff_natDegree hmo, RingHom.map_one]
  /-
    R : Type u
    S : Type v
    inst✝² : Semiring R
    inst✝¹ : Semiring S
    inst✝ : Nontrivial S
    P : Polynomial R
    hmo : P.Monic
    f : RingHom R S
    ⊢ Ne 1 0
  -/
  exact one_ne_zero
  /-
    🎉 no goals
  -/


@[simp]
theorem Monic.degree_map [Semiring S] [Nontrivial S] {P : R[X]} (hmo : P.Monic) (f : R →+* S) :
    (P.map f).degree = P.degree := by
  /-
    R : Type u
    S : Type v
    inst✝² : Semiring R
    inst✝¹ : Semiring S
    inst✝ : Nontrivial S
    P : Polynomial R
    hmo : P.Monic
    f : RingHom R S
    ⊢ Eq (Polynomial.map f P).degree P.degree
  -/
  by_cases hP : P = 0
    /-
      case pos
      R : Type u
      S : Type v
      inst✝² : Semiring R
      inst✝¹ : Semiring S
      inst✝ : Nontrivial S
      P : Polynomial R
      hmo : P.Monic
      f : RingHom R S
      hP : Eq P 0
      ⊢ Eq (Polynomial.map f P).degree P.degree
    -/
  · simp [hP]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u
      S : Type v
      inst✝² : Semiring R
      inst✝¹ : Semiring S
      inst✝ : Nontrivial S
      P : Polynomial R
      hmo : P.Monic
      f : RingHom R S
      hP : Not (Eq P 0)
      ⊢ Eq (Polynomial.map f P).degree P.degree
    -/
  · refine le_antisymm degree_map_le ?_
    /-
      case neg
      R : Type u
      S : Type v
      inst✝² : Semiring R
      inst✝¹ : Semiring S
      inst✝ : Nontrivial S
      P : Polynomial R
      hmo : P.Monic
      f : RingHom R S
      hP : Not (Eq P 0)
      ⊢ LE.le P.degree (Polynomial.map f P).degree
    -/
    rw [degree_eq_natDegree hP]
    /-
      case neg
      R : Type u
      S : Type v
      inst✝² : Semiring R
      inst✝¹ : Semiring S
      inst✝ : Nontrivial S
      P : Polynomial R
      hmo : P.Monic
      f : RingHom R S
      hP : Not (Eq P 0)
      ⊢ LE.le (↑P.natDegree) (Polynomial.map f P).degree
    -/
    refine le_degree_of_ne_zero ?_
    /-
      case neg
      R : Type u
      S : Type v
      inst✝² : Semiring R
      inst✝¹ : Semiring S
      inst✝ : Nontrivial S
      P : Polynomial R
      hmo : P.Monic
      f : RingHom R S
      hP : Not (Eq P 0)
      ⊢ Ne ((Polynomial.map f P).coeff P.natDegree) 0
    -/
    rw [coeff_map, Monic.coeff_natDegree hmo, RingHom.map_one]
    /-
      case neg
      R : Type u
      S : Type v
      inst✝² : Semiring R
      inst✝¹ : Semiring S
      inst✝ : Nontrivial S
      P : Polynomial R
      hmo : P.Monic
      f : RingHom R S
      hP : Not (Eq P 0)
      ⊢ Ne 1 0
    -/
    exact one_ne_zero
    /-
      🎉 no goals
    -/


theorem degree_map_eq_of_injective (hf : Injective f) (p : R[X]) : degree (p.map f) = degree p :=
  letI := Classical.decEq R
                       /-
                         R : Type u
                         S : Type v
                         inst✝¹ : Semiring R
                         inst✝ : Semiring S
                         f : RingHom R S
                         hf : Function.Injective ⇑f
                         p : Polynomial R
                         this : DecidableEq R := Classical.decEq R
                         h : Eq p 0
                         ⊢ Eq (Polynomial.map f p).degree p.degree
                       -/
  if h : p = 0 then by simp [h]
                       /-
                         🎉 no goals
                       -/
  else
    degree_map_eq_of_leadingCoeff_ne_zero _
          /-
            R : Type u
            S : Type v
            inst✝¹ : Semiring R
            inst✝ : Semiring S
            f : RingHom R S
            hf : Function.Injective ⇑f
            p : Polynomial R
            this : DecidableEq R := Classical.decEq R
            h : Not (Eq p 0)
            ⊢ Ne (f p.leadingCoeff) 0
          -/
      (by rw [← f.map_zero]; exact mt hf.eq_iff.1 (mt leadingCoeff_eq_zero.1 h))
                             /-
                               🎉 no goals
                             -/


theorem natDegree_map_eq_of_injective (hf : Injective f) (p : R[X]) :
    natDegree (p.map f) = natDegree p :=
  natDegree_eq_of_degree_eq (degree_map_eq_of_injective hf p)


theorem leadingCoeff_map' (hf : Injective f) (p : R[X]) :
    leadingCoeff (p.map f) = f (leadingCoeff p) := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    hf : Function.Injective ⇑f
    p : Polynomial R
    ⊢ Eq (Polynomial.map f p).leadingCoeff (f p.leadingCoeff)
  -/
  unfold leadingCoeff
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    hf : Function.Injective ⇑f
    p : Polynomial R
    ⊢ Eq ((Polynomial.map f p).coeff (Polynomial.map f p).natDegree) (f (p.coeff p …
  -/
  rw [coeff_map, natDegree_map_eq_of_injective hf p]
  /-
    🎉 no goals
  -/


theorem nextCoeff_map (hf : Injective f) (p : R[X]) : (p.map f).nextCoeff = f p.nextCoeff := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    hf : Function.Injective ⇑f
    p : Polynomial R
    ⊢ Eq (Polynomial.map f p).nextCoeff (f p.nextCoeff)
  -/
  unfold nextCoeff
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    hf : Function.Injective ⇑f
    p : Polynomial R
    ⊢ Eq (ite (Eq (Polynomial.map f p).natDegree 0) 0 ((Polynomial.map f p).coeff  …
  -/
  rw [natDegree_map_eq_of_injective hf]
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    hf : Function.Injective ⇑f
    p : Polynomial R
    ⊢ Eq (ite (Eq p.natDegree 0) 0 ((Polynomial.map f p).coeff (HSub.hSub p.natDeg …
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> simp [*]
                /-
                  🎉 no goals
                -/


theorem leadingCoeff_of_injective (hf : Injective f) (p : R[X]) :
    leadingCoeff (p.map f) = f (leadingCoeff p) := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    hf : Function.Injective ⇑f
    p : Polynomial R
    ⊢ Eq (Polynomial.map f p).leadingCoeff (f p.leadingCoeff)
  -/
  delta leadingCoeff
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    hf : Function.Injective ⇑f
    p : Polynomial R
    ⊢ Eq ((Polynomial.map f p).coeff (Polynomial.map f p).natDegree) (f (p.coeff p …
  -/
  rw [coeff_map f, natDegree_map_eq_of_injective hf p]
  /-
    🎉 no goals
  -/


theorem monic_of_injective (hf : Injective f) {p : R[X]} (hp : (p.map f).Monic) : p.Monic := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    hf : Function.Injective ⇑f
    p : Polynomial R
    hp : (Polynomial.map f p).Monic
    ⊢ p.Monic
  -/
  apply hf
  /-
    case a
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    hf : Function.Injective ⇑f
    p : Polynomial R
    hp : (Polynomial.map f p).Monic
    ⊢ Eq (f p.leadingCoeff) (f 1)
  -/
  rw [← leadingCoeff_of_injective hf, hp.leadingCoeff, f.map_one]
  /-
    🎉 no goals
  -/


theorem _root_.Function.Injective.monic_map_iff (hf : Injective f) {p : R[X]} :
    p.Monic ↔ (p.map f).Monic :=
  ⟨Monic.map _, Polynomial.monic_of_injective hf⟩


theorem monic_X_sub_C (x : R) : Monic (X - C x) := by
  /-
    R : Type u
    inst✝ : Ring R
    x : R
    ⊢ (HSub.hSub Polynomial.X (Polynomial.C x)).Monic
  -/
  simpa only [sub_eq_add_neg, C_neg] using monic_X_add_C (-x)
  /-
    🎉 no goals
  -/


theorem monic_X_pow_sub {n : ℕ} (H : degree p < n) : Monic (X ^ n - p) := by
  /-
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    n : Nat
    H : LT.lt p.degree ↑n
    ⊢ (HSub.hSub (HPow.hPow Polynomial.X n) p).Monic
  -/
  simpa [sub_eq_add_neg] using monic_X_pow_add (show degree (-p) < n by rwa [← degree_neg p] at H)
  /-
    🎉 no goals
  -/


/-- `X ^ n - a` is monic. -/
theorem monic_X_pow_sub_C {R : Type u} [Ring R] (a : R) {n : ℕ} (h : n ≠ 0) :
    (X ^ n - C a).Monic := by
  /-
    R : Type u
    inst✝ : Ring R
    a : R
    n : Nat
    h : Ne n 0
    ⊢ (HSub.hSub (HPow.hPow Polynomial.X n) (Polynomial.C a)).Monic
  -/
  simpa only [map_neg, ← sub_eq_add_neg] using monic_X_pow_add_C (-a) h
  /-
    🎉 no goals
  -/


theorem not_isUnit_X_pow_sub_one (R : Type*) [CommRing R] [Nontrivial R] (n : ℕ) :
    ¬IsUnit (X ^ n - 1 : R[X]) := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Nontrivial R
    n : Nat
    ⊢ Not (IsUnit (HSub.hSub (HPow.hPow Polynomial.X n) 1))
  -/
  intro h
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Nontrivial R
    n : Nat
    h : IsUnit (HSub.hSub (HPow.hPow Polynomial.X n) 1)
    ⊢ False
  -/
  rcases eq_or_ne n 0 with (rfl | hn)
    /-
      case inl
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : Nontrivial R
      h : IsUnit (HSub.hSub (HPow.hPow Polynomial.X 0) 1)
      ⊢ False
    -/
  · simp at h
    /-
      🎉 no goals
    -/
  /-
    case inr
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Nontrivial R
    n : Nat
    h : IsUnit (HSub.hSub (HPow.hPow Polynomial.X n) 1)
    hn : Ne n 0
    ⊢ False
  -/
  apply hn
  /-
    case inr
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : Nontrivial R
    n : Nat
    h : IsUnit (HSub.hSub (HPow.hPow Polynomial.X n) 1)
    hn : Ne n 0
    ⊢ Eq n 0
  -/
  rw [← @natDegree_one R, ← (monic_X_pow_sub_C _ hn).eq_one_of_isUnit h, natDegree_X_pow_sub_C]
  /-
    🎉 no goals
  -/


lemma Monic.comp_X_sub_C {p : R[X]} (hp : p.Monic) (r : R) : (p.comp (X - C r)).Monic := by
  /-
    R : Type u
    inst✝ : Ring R
    p : Polynomial R
    hp : p.Monic
    r : R
    ⊢ (p.comp (HSub.hSub Polynomial.X (Polynomial.C r))).Monic
  -/
  simpa using hp.comp_X_add_C (-r)
  /-
    🎉 no goals
  -/


theorem Monic.sub_of_left {p q : R[X]} (hp : Monic p) (hpq : degree q < degree p) :
    Monic (p - q) := by
  /-
    R : Type u
    inst✝ : Ring R
    p q : Polynomial R
    hp : p.Monic
    hpq : LT.lt q.degree p.degree
    ⊢ (HSub.hSub p q).Monic
  -/
  rw [sub_eq_add_neg]
  /-
    R : Type u
    inst✝ : Ring R
    p q : Polynomial R
    hp : p.Monic
    hpq : LT.lt q.degree p.degree
    ⊢ (HAdd.hAdd p (Neg.neg q)).Monic
  -/
  apply hp.add_of_left
  /-
    R : Type u
    inst✝ : Ring R
    p q : Polynomial R
    hp : p.Monic
    hpq : LT.lt q.degree p.degree
    ⊢ LT.lt (Neg.neg q).degree p.degree
  -/
  rwa [degree_neg]
  /-
    🎉 no goals
  -/


theorem Monic.sub_of_right {p q : R[X]} (hq : q.leadingCoeff = -1) (hpq : degree p < degree q) :
    Monic (p - q) := by
  have : (-q).coeff (-q).natDegree = 1 := by
    rw [natDegree_neg, coeff_neg, show q.coeff q.natDegree = -1 from hq, neg_neg]
  /-
    R : Type u
    inst✝ : Ring R
    p q : Polynomial R
    hq : Eq q.leadingCoeff (-1)
    hpq : LT.lt p.degree q.degree
    this : Eq ((Neg.neg q).coeff (Neg.neg q).natDegree) 1
    ⊢ (HSub.hSub p q).Monic
  -/
  rw [sub_eq_add_neg]
  /-
    R : Type u
    inst✝ : Ring R
    p q : Polynomial R
    hq : Eq q.leadingCoeff (-1)
    hpq : LT.lt p.degree q.degree
    this : Eq ((Neg.neg q).coeff (Neg.neg q).natDegree) 1
    ⊢ (HAdd.hAdd p (Neg.neg q)).Monic
  -/
  apply Monic.add_of_right this
  /-
    R : Type u
    inst✝ : Ring R
    p q : Polynomial R
    hq : Eq q.leadingCoeff (-1)
    hpq : LT.lt p.degree q.degree
    this : Eq ((Neg.neg q).coeff (Neg.neg q).natDegree) 1
    ⊢ LT.lt p.degree (Neg.neg q).degree
  -/
  rwa [degree_neg]
  /-
    🎉 no goals
  -/


@[simp]
theorem not_monic_zero : ¬Monic (0 : R[X]) :=
  not_monic_zero_iff.mp zero_ne_one


theorem Monic.mul_left_ne_zero (hp : Monic p) {q : R[X]} (hq : q ≠ 0) : q * p ≠ 0 := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : p.Monic
    q : Polynomial R
    hq : Ne q 0
    ⊢ Ne (HMul.hMul q p) 0
  -/
  by_cases h : p = 1
    /-
      case pos
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      hp : p.Monic
      q : Polynomial R
      hq : Ne q 0
      h : Eq p 1
      ⊢ Ne (HMul.hMul q p) 0
    -/
  · simpa [h]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : p.Monic
    q : Polynomial R
    hq : Ne q 0
    h : Not (Eq p 1)
    ⊢ Ne (HMul.hMul q p) 0
  -/
  rw [Ne, ← degree_eq_bot, hp.degree_mul, WithBot.add_eq_bot, not_or, degree_eq_bot]
  /-
    case neg
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : p.Monic
    q : Polynomial R
    hq : Ne q 0
    h : Not (Eq p 1)
    ⊢ And (Not (Eq q 0)) (Not (Eq p.degree Bot.bot))
  -/
  refine ⟨hq, ?_⟩
  /-
    case neg
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : p.Monic
    q : Polynomial R
    hq : Ne q 0
    h : Not (Eq p 1)
    ⊢ Not (Eq p.degree Bot.bot)
  -/
  rw [← hp.degree_le_zero_iff_eq_one, not_le] at h
  /-
    case neg
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : p.Monic
    q : Polynomial R
    hq : Ne q 0
    h : LT.lt 0 p.degree
    ⊢ Not (Eq p.degree Bot.bot)
  -/
  refine (lt_trans ?_ h).ne'
  /-
    case neg
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : p.Monic
    q : Polynomial R
    hq : Ne q 0
    h : LT.lt 0 p.degree
    ⊢ LT.lt Bot.bot 0
  -/
  simp
  /-
    🎉 no goals
  -/


theorem Monic.mul_right_ne_zero (hp : Monic p) {q : R[X]} (hq : q ≠ 0) : p * q ≠ 0 := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : p.Monic
    q : Polynomial R
    hq : Ne q 0
    ⊢ Ne (HMul.hMul p q) 0
  -/
  by_cases h : p = 1
    /-
      case pos
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      hp : p.Monic
      q : Polynomial R
      hq : Ne q 0
      h : Eq p 1
      ⊢ Ne (HMul.hMul p q) 0
    -/
  · simpa [h]
    /-
      🎉 no goals
    -/
  rw [Ne, ← degree_eq_bot, hp.degree_mul_comm, hp.degree_mul, WithBot.add_eq_bot, not_or,
    degree_eq_bot]
  /-
    case neg
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : p.Monic
    q : Polynomial R
    hq : Ne q 0
    h : Not (Eq p 1)
    ⊢ And (Not (Eq q 0)) (Not (Eq p.degree Bot.bot))
  -/
  refine ⟨hq, ?_⟩
  /-
    case neg
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : p.Monic
    q : Polynomial R
    hq : Ne q 0
    h : Not (Eq p 1)
    ⊢ Not (Eq p.degree Bot.bot)
  -/
  rw [← hp.degree_le_zero_iff_eq_one, not_le] at h
  /-
    case neg
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : p.Monic
    q : Polynomial R
    hq : Ne q 0
    h : LT.lt 0 p.degree
    ⊢ Not (Eq p.degree Bot.bot)
  -/
  refine (lt_trans ?_ h).ne'
  /-
    case neg
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    hp : p.Monic
    q : Polynomial R
    hq : Ne q 0
    h : LT.lt 0 p.degree
    ⊢ LT.lt Bot.bot 0
  -/
  simp
  /-
    🎉 no goals
  -/


theorem Monic.mul_natDegree_lt_iff (h : Monic p) {q : R[X]} :
    (p * q).natDegree < p.natDegree ↔ p ≠ 1 ∧ q = 0 := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    h : p.Monic
    q : Polynomial R
    ⊢ Iff (LT.lt (HMul.hMul p q).natDegree p.natDegree) (And (Ne p 1) (Eq q 0))
  -/
  by_cases hq : q = 0
    /-
      case pos
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      h : p.Monic
      q : Polynomial R
      hq : Eq q 0
      ⊢ Iff (LT.lt (HMul.hMul p q).natDegree p.natDegree) (And (Ne p 1) (Eq q 0))
    -/
  · suffices 0 < p.natDegree ↔ p.natDegree ≠ 0 by simpa [hq, ← h.natDegree_eq_zero_iff_eq_one]
    /-
      case pos
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      h : p.Monic
      q : Polynomial R
      hq : Eq q 0
      ⊢ Iff (LT.lt 0 p.natDegree) (Ne p.natDegree 0)
    -/
    exact ⟨fun h => h.ne', fun h => lt_of_le_of_ne (Nat.zero_le _) h.symm⟩
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      h : p.Monic
      q : Polynomial R
      hq : Not (Eq q 0)
      ⊢ Iff (LT.lt (HMul.hMul p q).natDegree p.natDegree) (And (Ne p 1) (Eq q 0))
    -/
  · simp [h.natDegree_mul', hq]
    /-
      🎉 no goals
    -/


theorem Monic.mul_right_eq_zero_iff (h : Monic p) {q : R[X]} : p * q = 0 ↔ q = 0 := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    h : p.Monic
    q : Polynomial R
    ⊢ Iff (Eq (HMul.hMul p q) 0) (Eq q 0)
  -/
                          /-
                            🎉 no goals
                          -/
  by_cases hq : q = 0 <;> simp [h.mul_right_ne_zero, hq]
                          /-
                            🎉 no goals
                          -/


theorem Monic.mul_left_eq_zero_iff (h : Monic p) {q : R[X]} : q * p = 0 ↔ q = 0 := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    h : p.Monic
    q : Polynomial R
    ⊢ Iff (Eq (HMul.hMul q p) 0) (Eq q 0)
  -/
                          /-
                            🎉 no goals
                          -/
  by_cases hq : q = 0 <;> simp [h.mul_left_ne_zero, hq]
                          /-
                            🎉 no goals
                          -/


theorem Monic.isRegular {R : Type*} [Ring R] {p : R[X]} (hp : Monic p) : IsRegular p := by
  /-
    R : Type u_1
    inst✝ : Ring R
    p : Polynomial R
    hp : p.Monic
    ⊢ IsRegular p
  -/
  constructor
    /-
      case left
      R : Type u_1
      inst✝ : Ring R
      p : Polynomial R
      hp : p.Monic
      ⊢ IsLeftRegular p
    -/
  · intro q r h
    /-
      case left
      R : Type u_1
      inst✝ : Ring R
      p : Polynomial R
      hp : p.Monic
      q r : Polynomial R
      h : Eq ((fun x => HMul.hMul p x) q) ((fun x => HMul.hMul p x) r)
      ⊢ Eq q r
    -/
    dsimp only at h
    /-
      case left
      R : Type u_1
      inst✝ : Ring R
      p : Polynomial R
      hp : p.Monic
      q r : Polynomial R
      h : Eq (HMul.hMul p q) (HMul.hMul p r)
      ⊢ Eq q r
    -/
    rw [← sub_eq_zero, ← hp.mul_right_eq_zero_iff, mul_sub, h, sub_self]
    /-
      🎉 no goals
    -/
    /-
      case right
      R : Type u_1
      inst✝ : Ring R
      p : Polynomial R
      hp : p.Monic
      ⊢ IsRightRegular p
    -/
  · intro q r h
    /-
      case right
      R : Type u_1
      inst✝ : Ring R
      p : Polynomial R
      hp : p.Monic
      q r : Polynomial R
      h : Eq ((fun x => HMul.hMul x p) q) ((fun x => HMul.hMul x p) r)
      ⊢ Eq q r
    -/
    simp only at h
    /-
      case right
      R : Type u_1
      inst✝ : Ring R
      p : Polynomial R
      hp : p.Monic
      q r : Polynomial R
      h : Eq (HMul.hMul q p) (HMul.hMul r p)
      ⊢ Eq q r
    -/
    rw [← sub_eq_zero, ← hp.mul_left_eq_zero_iff, sub_mul, h, sub_self]
    /-
      🎉 no goals
    -/


theorem degree_smul_of_smul_regular {S : Type*} [Monoid S] [DistribMulAction S R] {k : S}
    (p : R[X]) (h : IsSMulRegular R k) : (k • p).degree = p.degree := by
  /-
    R : Type u
    inst✝² : Semiring R
    S : Type u_1
    inst✝¹ : Monoid S
    inst✝ : DistribMulAction S R
    k : S
    p : Polynomial R
    h : IsSMulRegular R k
    ⊢ Eq (HSMul.hSMul k p).degree p.degree
  -/
  refine le_antisymm ?_ ?_
    /-
      case refine_1
      R : Type u
      inst✝² : Semiring R
      S : Type u_1
      inst✝¹ : Monoid S
      inst✝ : DistribMulAction S R
      k : S
      p : Polynomial R
      h : IsSMulRegular R k
      ⊢ LE.le (HSMul.hSMul k p).degree p.degree
    -/
  · rw [degree_le_iff_coeff_zero]
    /-
      case refine_1
      R : Type u
      inst✝² : Semiring R
      S : Type u_1
      inst✝¹ : Monoid S
      inst✝ : DistribMulAction S R
      k : S
      p : Polynomial R
      h : IsSMulRegular R k
      ⊢ ∀ (m : Nat), LT.lt p.degree ↑m → Eq ((HSMul.hSMul k p).coeff m) 0
    -/
    intro m hm
    /-
      case refine_1
      R : Type u
      inst✝² : Semiring R
      S : Type u_1
      inst✝¹ : Monoid S
      inst✝ : DistribMulAction S R
      k : S
      p : Polynomial R
      h : IsSMulRegular R k
      m : Nat
      hm : LT.lt p.degree ↑m
      ⊢ Eq ((HSMul.hSMul k p).coeff m) 0
    -/
    rw [degree_lt_iff_coeff_zero] at hm
    /-
      case refine_1
      R : Type u
      inst✝² : Semiring R
      S : Type u_1
      inst✝¹ : Monoid S
      inst✝ : DistribMulAction S R
      k : S
      p : Polynomial R
      h : IsSMulRegular R k
      m : Nat
      hm : ∀ (m_1 : Nat), LE.le m m_1 → Eq (p.coeff m_1) 0
      ⊢ Eq ((HSMul.hSMul k p).coeff m) 0
    -/
    simp [hm m le_rfl]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u
      inst✝² : Semiring R
      S : Type u_1
      inst✝¹ : Monoid S
      inst✝ : DistribMulAction S R
      k : S
      p : Polynomial R
      h : IsSMulRegular R k
      ⊢ LE.le p.degree (HSMul.hSMul k p).degree
    -/
  · rw [degree_le_iff_coeff_zero]
    /-
      case refine_2
      R : Type u
      inst✝² : Semiring R
      S : Type u_1
      inst✝¹ : Monoid S
      inst✝ : DistribMulAction S R
      k : S
      p : Polynomial R
      h : IsSMulRegular R k
      ⊢ ∀ (m : Nat), LT.lt (HSMul.hSMul k p).degree ↑m → Eq (p.coeff m) 0
    -/
    intro m hm
    /-
      case refine_2
      R : Type u
      inst✝² : Semiring R
      S : Type u_1
      inst✝¹ : Monoid S
      inst✝ : DistribMulAction S R
      k : S
      p : Polynomial R
      h : IsSMulRegular R k
      m : Nat
      hm : LT.lt (HSMul.hSMul k p).degree ↑m
      ⊢ Eq (p.coeff m) 0
    -/
    rw [degree_lt_iff_coeff_zero] at hm
    /-
      case refine_2
      R : Type u
      inst✝² : Semiring R
      S : Type u_1
      inst✝¹ : Monoid S
      inst✝ : DistribMulAction S R
      k : S
      p : Polynomial R
      h : IsSMulRegular R k
      m : Nat
      hm : ∀ (m_1 : Nat), LE.le m m_1 → Eq ((HSMul.hSMul k p).coeff m_1) 0
      ⊢ Eq (p.coeff m) 0
    -/
    refine h ?_
    /-
      case refine_2
      R : Type u
      inst✝² : Semiring R
      S : Type u_1
      inst✝¹ : Monoid S
      inst✝ : DistribMulAction S R
      k : S
      p : Polynomial R
      h : IsSMulRegular R k
      m : Nat
      hm : ∀ (m_1 : Nat), LE.le m m_1 → Eq ((HSMul.hSMul k p).coeff m_1) 0
      ⊢ Eq ((fun x => HSMul.hSMul k x) (p.coeff m)) ((fun x => HSMul.hSMul k x) 0)
    -/
    simpa using hm m le_rfl
    /-
      🎉 no goals
    -/


theorem natDegree_smul_of_smul_regular {S : Type*} [Monoid S] [DistribMulAction S R] {k : S}
    (p : R[X]) (h : IsSMulRegular R k) : (k • p).natDegree = p.natDegree := by
  /-
    R : Type u
    inst✝² : Semiring R
    S : Type u_1
    inst✝¹ : Monoid S
    inst✝ : DistribMulAction S R
    k : S
    p : Polynomial R
    h : IsSMulRegular R k
    ⊢ Eq (HSMul.hSMul k p).natDegree p.natDegree
  -/
  by_cases hp : p = 0
    /-
      case pos
      R : Type u
      inst✝² : Semiring R
      S : Type u_1
      inst✝¹ : Monoid S
      inst✝ : DistribMulAction S R
      k : S
      p : Polynomial R
      h : IsSMulRegular R k
      hp : Eq p 0
      ⊢ Eq (HSMul.hSMul k p).natDegree p.natDegree
    -/
  · simp [hp]
    /-
      🎉 no goals
    -/
  rw [← Nat.cast_inj (R := WithBot ℕ), ← degree_eq_natDegree hp, ← degree_eq_natDegree,
    degree_smul_of_smul_regular p h]
  /-
    case neg
    R : Type u
    inst✝² : Semiring R
    S : Type u_1
    inst✝¹ : Monoid S
    inst✝ : DistribMulAction S R
    k : S
    p : Polynomial R
    h : IsSMulRegular R k
    hp : Not (Eq p 0)
    ⊢ Ne (HSMul.hSMul k p) 0
  -/
  contrapose! hp
  /-
    case neg
    R : Type u
    inst✝² : Semiring R
    S : Type u_1
    inst✝¹ : Monoid S
    inst✝ : DistribMulAction S R
    k : S
    p : Polynomial R
    h : IsSMulRegular R k
    hp : Eq (HSMul.hSMul k p) 0
    ⊢ Eq p 0
  -/
  rw [← smul_zero k] at hp
  /-
    case neg
    R : Type u
    inst✝² : Semiring R
    S : Type u_1
    inst✝¹ : Monoid S
    inst✝ : DistribMulAction S R
    k : S
    p : Polynomial R
    h : IsSMulRegular R k
    hp : Eq (HSMul.hSMul k p) (HSMul.hSMul k 0)
    ⊢ Eq p 0
  -/
  exact h.polynomial hp
  /-
    🎉 no goals
  -/


theorem leadingCoeff_smul_of_smul_regular {S : Type*} [Monoid S] [DistribMulAction S R] {k : S}
    (p : R[X]) (h : IsSMulRegular R k) : (k • p).leadingCoeff = k • p.leadingCoeff := by
  rw [Polynomial.leadingCoeff, Polynomial.leadingCoeff, coeff_smul,
    natDegree_smul_of_smul_regular p h]


theorem monic_of_isUnit_leadingCoeff_inv_smul (h : IsUnit p.leadingCoeff) :
    Monic (h.unit⁻¹ • p) := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    h : IsUnit p.leadingCoeff
    ⊢ (HSMul.hSMul (Inv.inv h.unit) p).Monic
  -/
  rw [Monic.def, leadingCoeff_smul_of_smul_regular _ (isSMulRegular_of_group _), Units.smul_def]
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    h : IsUnit p.leadingCoeff
    ⊢ Eq (HSMul.hSMul (↑(Inv.inv h.unit)) p.leadingCoeff) 1
  -/
  obtain ⟨k, hk⟩ := h
  /-
    case intro
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    k : Units R
    hk : Eq (↑k) p.leadingCoeff
    ⊢ Eq (HSMul.hSMul (↑(Inv.inv (IsUnit.unit ⋯))) p.leadingCoeff) 1
  -/
  simp only [← hk, smul_eq_mul, ← Units.val_mul, Units.val_eq_one, inv_mul_eq_iff_eq_mul]
  /-
    case intro
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    k : Units R
    hk : Eq (↑k) p.leadingCoeff
    ⊢ Eq k (HMul.hMul ⋯.unit 1)
  -/
  simp [Units.ext_iff, IsUnit.unit_spec]
  /-
    🎉 no goals
  -/


theorem isUnit_leadingCoeff_mul_right_eq_zero_iff (h : IsUnit p.leadingCoeff) {q : R[X]} :
    p * q = 0 ↔ q = 0 := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    h : IsUnit p.leadingCoeff
    q : Polynomial R
    ⊢ Iff (Eq (HMul.hMul p q) 0) (Eq q 0)
  -/
  constructor
    /-
      case mp
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      h : IsUnit p.leadingCoeff
      q : Polynomial R
      ⊢ Eq (HMul.hMul p q) 0 → Eq q 0
    -/
  · intro hp
    /-
      case mp
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      h : IsUnit p.leadingCoeff
      q : Polynomial R
      hp : Eq (HMul.hMul p q) 0
      ⊢ Eq q 0
    -/
    rw [← smul_eq_zero_iff_eq h.unit⁻¹] at hp
    have : h.unit⁻¹ • (p * q) = h.unit⁻¹ • p * q := by
      ext
      simp only [Units.smul_def, coeff_smul, coeff_mul, smul_eq_mul, mul_sum]
      refine sum_congr rfl fun x _ => ?_
      rw [← mul_assoc]
    /-
      case mp
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      h : IsUnit p.leadingCoeff
      q : Polynomial R
      hp : Eq (HSMul.hSMul (Inv.inv h.unit) (HMul.hMul p q)) 0
      this : Eq (HSMul.hSMul (Inv.inv h.unit) (HMul.hMul p q)) (HMul.hMul (HSMul.hSM …
      ⊢ Eq q 0
    -/
    rwa [this, Monic.mul_right_eq_zero_iff] at hp
    /-
      case mp.h
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      h : IsUnit p.leadingCoeff
      q : Polynomial R
      hp : Eq (HMul.hMul (HSMul.hSMul (Inv.inv h.unit) p) q) 0
      this : Eq (HSMul.hSMul (Inv.inv h.unit) (HMul.hMul p q)) (HMul.hMul (HSMul.hSM …
      ⊢ (HSMul.hSMul (Inv.inv h.unit) p).Monic
    -/
    exact monic_of_isUnit_leadingCoeff_inv_smul _
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      h : IsUnit p.leadingCoeff
      q : Polynomial R
      ⊢ Eq q 0 → Eq (HMul.hMul p q) 0
    -/
  · rintro rfl
    /-
      case mpr
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      h : IsUnit p.leadingCoeff
      ⊢ Eq (HMul.hMul p 0) 0
    -/
    simp
    /-
      🎉 no goals
    -/


theorem isUnit_leadingCoeff_mul_left_eq_zero_iff (h : IsUnit p.leadingCoeff) {q : R[X]} :
    q * p = 0 ↔ q = 0 := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    h : IsUnit p.leadingCoeff
    q : Polynomial R
    ⊢ Iff (Eq (HMul.hMul q p) 0) (Eq q 0)
  -/
  constructor
    /-
      case mp
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      h : IsUnit p.leadingCoeff
      q : Polynomial R
      ⊢ Eq (HMul.hMul q p) 0 → Eq q 0
    -/
  · intro hp
    /-
      case mp
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      h : IsUnit p.leadingCoeff
      q : Polynomial R
      hp : Eq (HMul.hMul q p) 0
      ⊢ Eq q 0
    -/
    replace hp := congr_arg (· * C ↑h.unit⁻¹) hp
    /-
      case mp
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      h : IsUnit p.leadingCoeff
      q : Polynomial R
      hp : Eq ((fun x => HMul.hMul x (Polynomial.C ↑(Inv.inv h.unit))) (HMul.hMul q  …
      ⊢ Eq q 0
    -/
    simp only [zero_mul] at hp
    /-
      case mp
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      h : IsUnit p.leadingCoeff
      q : Polynomial R
      hp : Eq (HMul.hMul (HMul.hMul q p) (Polynomial.C ↑(Inv.inv h.unit))) 0
      ⊢ Eq q 0
    -/
    rwa [mul_assoc, Monic.mul_left_eq_zero_iff] at hp
    /-
      case mp.h
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      h : IsUnit p.leadingCoeff
      q : Polynomial R
      hp : Eq (HMul.hMul q (HMul.hMul p (Polynomial.C ↑(Inv.inv h.unit)))) 0
      ⊢ (HMul.hMul p (Polynomial.C ↑(Inv.inv h.unit))).Monic
    -/
    refine monic_mul_C_of_leadingCoeff_mul_eq_one ?_
    /-
      case mp.h
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      h : IsUnit p.leadingCoeff
      q : Polynomial R
      hp : Eq (HMul.hMul q (HMul.hMul p (Polynomial.C ↑(Inv.inv h.unit)))) 0
      ⊢ Eq (HMul.hMul p.leadingCoeff ↑(Inv.inv h.unit)) 1
    -/
    simp [Units.mul_inv_eq_iff_eq_mul, IsUnit.unit_spec]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      h : IsUnit p.leadingCoeff
      q : Polynomial R
      ⊢ Eq q 0 → Eq (HMul.hMul q p) 0
    -/
  · rintro rfl
    /-
      case mpr
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      h : IsUnit p.leadingCoeff
      ⊢ Eq (HMul.hMul 0 p) 0
    -/
    rw [zero_mul]
    /-
      🎉 no goals
    -/


