/-- If `A` is a family of enough low-degree polynomials over a finite semiring, there is a
pair of equal elements in `A`. -/
theorem exists_eq_polynomial [Semiring Fq] {d : ℕ} {m : ℕ} (hm : Fintype.card Fq ^ d ≤ m)
    (b : Fq[X]) (hb : natDegree b ≤ d) (A : Fin m.succ → Fq[X])
    (hA : ∀ i, degree (A i) < degree b) : ∃ i₀ i₁, i₀ ≠ i₁ ∧ A i₁ = A i₀ := by
  -- Since there are > q^d elements of A, and only q^d choices for the highest `d` coefficients,
  -- there must be two elements of A with the same coefficients at
  -- `0`, ... `degree b - 1` ≤ `d - 1`.
  -- In other words, the following map is not injective:
  /-
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Semiring Fq
    d m : Nat
    hm : LE.le (HPow.hPow (Fintype.card Fq) d) m
    b : Polynomial Fq
    hb : LE.le b.natDegree d
    A : Fin m.succ → Polynomial Fq
    hA : ∀ (i : Fin m.succ), LT.lt (A i).degree b.degree
    ⊢ Exists fun i₀ => Exists fun i₁ => And (Ne i₀ i₁) (Eq (A i₁) (A i₀))
  -/
  set f : Fin m.succ → Fin d → Fq := fun i j => (A i).coeff j
  have : Fintype.card (Fin d → Fq) < Fintype.card (Fin m.succ) := by
    simpa using lt_of_le_of_lt hm (Nat.lt_succ_self m)
  -- Therefore, the differences have all coefficients higher than `deg b - d` equal.
  /-
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Semiring Fq
    d m : Nat
    hm : LE.le (HPow.hPow (Fintype.card Fq) d) m
    b : Polynomial Fq
    hb : LE.le b.natDegree d
    A : Fin m.succ → Polynomial Fq
    hA : ∀ (i : Fin m.succ), LT.lt (A i).degree b.degree
    f : Fin m.succ → Fin d → Fq := fun i j => (A i).coeff ↑j
    this : LT.lt (Fintype.card (Fin d → Fq)) (Fintype.card (Fin m.succ))
    ⊢ Exists fun i₀ => Exists fun i₁ => And (Ne i₀ i₁) (Eq (A i₁) (A i₀))
  -/
  obtain ⟨i₀, i₁, i_ne, i_eq⟩ := Fintype.exists_ne_map_eq_of_card_lt f this
  /-
    case intro.intro.intro
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Semiring Fq
    d m : Nat
    hm : LE.le (HPow.hPow (Fintype.card Fq) d) m
    b : Polynomial Fq
    hb : LE.le b.natDegree d
    A : Fin m.succ → Polynomial Fq
    hA : ∀ (i : Fin m.succ), LT.lt (A i).degree b.degree
    f : Fin m.succ → Fin d → Fq := fun i j => (A i).coeff ↑j
    this : LT.lt (Fintype.card (Fin d → Fq)) (Fintype.card (Fin m.succ))
    i₀ i₁ : Fin m.succ
    i_ne : Ne i₀ i₁
    i_eq : Eq (f i₀) (f i₁)
    ⊢ Exists fun i₀ => Exists fun i₁ => And (Ne i₀ i₁) (Eq (A i₁) (A i₀))
  -/
  use i₀, i₁, i_ne
  /-
    case right
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Semiring Fq
    d m : Nat
    hm : LE.le (HPow.hPow (Fintype.card Fq) d) m
    b : Polynomial Fq
    hb : LE.le b.natDegree d
    A : Fin m.succ → Polynomial Fq
    hA : ∀ (i : Fin m.succ), LT.lt (A i).degree b.degree
    f : Fin m.succ → Fin d → Fq := fun i j => (A i).coeff ↑j
    this : LT.lt (Fintype.card (Fin d → Fq)) (Fintype.card (Fin m.succ))
    i₀ i₁ : Fin m.succ
    i_ne : Ne i₀ i₁
    i_eq : Eq (f i₀) (f i₁)
    ⊢ Eq (A i₁) (A i₀)
  -/
  ext j
  -- The coefficients higher than `deg b` are the same because they are equal to 0.
  /-
    case right.a
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Semiring Fq
    d m : Nat
    hm : LE.le (HPow.hPow (Fintype.card Fq) d) m
    b : Polynomial Fq
    hb : LE.le b.natDegree d
    A : Fin m.succ → Polynomial Fq
    hA : ∀ (i : Fin m.succ), LT.lt (A i).degree b.degree
    f : Fin m.succ → Fin d → Fq := fun i j => (A i).coeff ↑j
    this : LT.lt (Fintype.card (Fin d → Fq)) (Fintype.card (Fin m.succ))
    i₀ i₁ : Fin m.succ
    i_ne : Ne i₀ i₁
    i_eq : Eq (f i₀) (f i₁)
    j : Nat
    ⊢ Eq ((A i₁).coeff j) ((A i₀).coeff j)
  -/
  by_cases hbj : degree b ≤ j
  · rw [coeff_eq_zero_of_degree_lt (lt_of_lt_of_le (hA _) hbj),
      coeff_eq_zero_of_degree_lt (lt_of_lt_of_le (hA _) hbj)]
  -- So we only need to look for the coefficients between `0` and `deg b`.
  /-
    case neg
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Semiring Fq
    d m : Nat
    hm : LE.le (HPow.hPow (Fintype.card Fq) d) m
    b : Polynomial Fq
    hb : LE.le b.natDegree d
    A : Fin m.succ → Polynomial Fq
    hA : ∀ (i : Fin m.succ), LT.lt (A i).degree b.degree
    f : Fin m.succ → Fin d → Fq := fun i j => (A i).coeff ↑j
    this : LT.lt (Fintype.card (Fin d → Fq)) (Fintype.card (Fin m.succ))
    i₀ i₁ : Fin m.succ
    i_ne : Ne i₀ i₁
    i_eq : Eq (f i₀) (f i₁)
    j : Nat
    hbj : Not (LE.le b.degree ↑j)
    ⊢ Eq ((A i₁).coeff j) ((A i₀).coeff j)
  -/
  rw [not_le] at hbj
  /-
    case neg
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Semiring Fq
    d m : Nat
    hm : LE.le (HPow.hPow (Fintype.card Fq) d) m
    b : Polynomial Fq
    hb : LE.le b.natDegree d
    A : Fin m.succ → Polynomial Fq
    hA : ∀ (i : Fin m.succ), LT.lt (A i).degree b.degree
    f : Fin m.succ → Fin d → Fq := fun i j => (A i).coeff ↑j
    this : LT.lt (Fintype.card (Fin d → Fq)) (Fintype.card (Fin m.succ))
    i₀ i₁ : Fin m.succ
    i_ne : Ne i₀ i₁
    i_eq : Eq (f i₀) (f i₁)
    j : Nat
    hbj : LT.lt (↑j) b.degree
    ⊢ Eq ((A i₁).coeff j) ((A i₀).coeff j)
  -/
  apply congr_fun i_eq.symm ⟨j, _⟩
  /-
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Semiring Fq
    d m : Nat
    hm : LE.le (HPow.hPow (Fintype.card Fq) d) m
    b : Polynomial Fq
    hb : LE.le b.natDegree d
    A : Fin m.succ → Polynomial Fq
    hA : ∀ (i : Fin m.succ), LT.lt (A i).degree b.degree
    f : Fin m.succ → Fin d → Fq := fun i j => (A i).coeff ↑j
    this : LT.lt (Fintype.card (Fin d → Fq)) (Fintype.card (Fin m.succ))
    i₀ i₁ : Fin m.succ
    i_ne : Ne i₀ i₁
    i_eq : Eq (f i₀) (f i₁)
    j : Nat
    hbj : LT.lt (↑j) b.degree
    ⊢ LT.lt j d
  -/
  exact lt_of_lt_of_le (coe_lt_degree.mp hbj) hb
  /-
    🎉 no goals
  -/


/-- If `A` is a family of enough low-degree polynomials over a finite ring,
there is a pair of elements in `A` (with different indices but not necessarily
distinct), such that their difference has small degree. -/
theorem exists_approx_polynomial_aux [Ring Fq] {d : ℕ} {m : ℕ} (hm : Fintype.card Fq ^ d ≤ m)
    (b : Fq[X]) (A : Fin m.succ → Fq[X]) (hA : ∀ i, degree (A i) < degree b) :
    ∃ i₀ i₁, i₀ ≠ i₁ ∧ degree (A i₁ - A i₀) < ↑(natDegree b - d) := by
  have hb : b ≠ 0 := by
    rintro rfl
    specialize hA 0
    rw [degree_zero] at hA
    exact not_lt_of_le bot_le hA
  -- Since there are > q^d elements of A, and only q^d choices for the highest `d` coefficients,
  -- there must be two elements of A with the same coefficients at
  -- `degree b - 1`, ... `degree b - d`.
  -- In other words, the following map is not injective:
  /-
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Ring Fq
    d m : Nat
    hm : LE.le (HPow.hPow (Fintype.card Fq) d) m
    b : Polynomial Fq
    A : Fin m.succ → Polynomial Fq
    hA : ∀ (i : Fin m.succ), LT.lt (A i).degree b.degree
    hb : Ne b 0
    ⊢ Exists fun i₀ => Exists fun i₁ => And (Ne i₀ i₁) (LT.lt (HSub.hSub (A i₁) (A …
  -/
  set f : Fin m.succ → Fin d → Fq := fun i j => (A i).coeff (natDegree b - j.succ)
  have : Fintype.card (Fin d → Fq) < Fintype.card (Fin m.succ) := by
    simpa using lt_of_le_of_lt hm (Nat.lt_succ_self m)
  -- Therefore, the differences have all coefficients higher than `deg b - d` equal.
  /-
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Ring Fq
    d m : Nat
    hm : LE.le (HPow.hPow (Fintype.card Fq) d) m
    b : Polynomial Fq
    A : Fin m.succ → Polynomial Fq
    hA : ∀ (i : Fin m.succ), LT.lt (A i).degree b.degree
    hb : Ne b 0
    f : Fin m.succ → Fin d → Fq := fun i j => (A i).coeff (HSub.hSub b.natDegree ↑ …
    this : LT.lt (Fintype.card (Fin d → Fq)) (Fintype.card (Fin m.succ))
    ⊢ Exists fun i₀ => Exists fun i₁ => And (Ne i₀ i₁) (LT.lt (HSub.hSub (A i₁) (A …
  -/
  obtain ⟨i₀, i₁, i_ne, i_eq⟩ := Fintype.exists_ne_map_eq_of_card_lt f this
  /-
    case intro.intro.intro
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Ring Fq
    d m : Nat
    hm : LE.le (HPow.hPow (Fintype.card Fq) d) m
    b : Polynomial Fq
    A : Fin m.succ → Polynomial Fq
    hA : ∀ (i : Fin m.succ), LT.lt (A i).degree b.degree
    hb : Ne b 0
    f : Fin m.succ → Fin d → Fq := fun i j => (A i).coeff (HSub.hSub b.natDegree ↑ …
    this : LT.lt (Fintype.card (Fin d → Fq)) (Fintype.card (Fin m.succ))
    i₀ i₁ : Fin m.succ
    i_ne : Ne i₀ i₁
    i_eq : Eq (f i₀) (f i₁)
    ⊢ Exists fun i₀ => Exists fun i₁ => And (Ne i₀ i₁) (LT.lt (HSub.hSub (A i₁) (A …
  -/
  use i₀, i₁, i_ne
  /-
    case right
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Ring Fq
    d m : Nat
    hm : LE.le (HPow.hPow (Fintype.card Fq) d) m
    b : Polynomial Fq
    A : Fin m.succ → Polynomial Fq
    hA : ∀ (i : Fin m.succ), LT.lt (A i).degree b.degree
    hb : Ne b 0
    f : Fin m.succ → Fin d → Fq := fun i j => (A i).coeff (HSub.hSub b.natDegree ↑ …
    this : LT.lt (Fintype.card (Fin d → Fq)) (Fintype.card (Fin m.succ))
    i₀ i₁ : Fin m.succ
    i_ne : Ne i₀ i₁
    i_eq : Eq (f i₀) (f i₁)
    ⊢ LT.lt (HSub.hSub (A i₁) (A i₀)).degree ↑(HSub.hSub b.natDegree d)
  -/
  refine (degree_lt_iff_coeff_zero _ _).mpr fun j hj => ?_
  -- The coefficients higher than `deg b` are the same because they are equal to 0.
  /-
    case right
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Ring Fq
    d m : Nat
    hm : LE.le (HPow.hPow (Fintype.card Fq) d) m
    b : Polynomial Fq
    A : Fin m.succ → Polynomial Fq
    hA : ∀ (i : Fin m.succ), LT.lt (A i).degree b.degree
    hb : Ne b 0
    f : Fin m.succ → Fin d → Fq := fun i j => (A i).coeff (HSub.hSub b.natDegree ↑ …
    this : LT.lt (Fintype.card (Fin d → Fq)) (Fintype.card (Fin m.succ))
    i₀ i₁ : Fin m.succ
    i_ne : Ne i₀ i₁
    i_eq : Eq (f i₀) (f i₁)
    j : Nat
    hj : LE.le (HSub.hSub b.natDegree d) j
    ⊢ Eq ((HSub.hSub (A i₁) (A i₀)).coeff j) 0
  -/
  by_cases hbj : degree b ≤ j
    /-
      case pos
      Fq : Type u_1
      inst✝¹ : Fintype Fq
      inst✝ : Ring Fq
      d m : Nat
      hm : LE.le (HPow.hPow (Fintype.card Fq) d) m
      b : Polynomial Fq
      A : Fin m.succ → Polynomial Fq
      hA : ∀ (i : Fin m.succ), LT.lt (A i).degree b.degree
      hb : Ne b 0
      f : Fin m.succ → Fin d → Fq := fun i j => (A i).coeff (HSub.hSub b.natDegree ↑ …
      this : LT.lt (Fintype.card (Fin d → Fq)) (Fintype.card (Fin m.succ))
      i₀ i₁ : Fin m.succ
      i_ne : Ne i₀ i₁
      i_eq : Eq (f i₀) (f i₁)
      j : Nat
      hj : LE.le (HSub.hSub b.natDegree d) j
      hbj : LE.le b.degree ↑j
      ⊢ Eq ((HSub.hSub (A i₁) (A i₀)).coeff j) 0
    -/
  · refine coeff_eq_zero_of_degree_lt (lt_of_lt_of_le ?_ hbj)
    /-
      case pos
      Fq : Type u_1
      inst✝¹ : Fintype Fq
      inst✝ : Ring Fq
      d m : Nat
      hm : LE.le (HPow.hPow (Fintype.card Fq) d) m
      b : Polynomial Fq
      A : Fin m.succ → Polynomial Fq
      hA : ∀ (i : Fin m.succ), LT.lt (A i).degree b.degree
      hb : Ne b 0
      f : Fin m.succ → Fin d → Fq := fun i j => (A i).coeff (HSub.hSub b.natDegree ↑ …
      this : LT.lt (Fintype.card (Fin d → Fq)) (Fintype.card (Fin m.succ))
      i₀ i₁ : Fin m.succ
      i_ne : Ne i₀ i₁
      i_eq : Eq (f i₀) (f i₁)
      j : Nat
      hj : LE.le (HSub.hSub b.natDegree d) j
      hbj : LE.le b.degree ↑j
      ⊢ LT.lt (HSub.hSub (A i₁) (A i₀)).degree b.degree
    -/
    exact lt_of_le_of_lt (degree_sub_le _ _) (max_lt (hA _) (hA _))
    /-
      🎉 no goals
    -/
  -- So we only need to look for the coefficients between `deg b - d` and `deg b`.
  /-
    case neg
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Ring Fq
    d m : Nat
    hm : LE.le (HPow.hPow (Fintype.card Fq) d) m
    b : Polynomial Fq
    A : Fin m.succ → Polynomial Fq
    hA : ∀ (i : Fin m.succ), LT.lt (A i).degree b.degree
    hb : Ne b 0
    f : Fin m.succ → Fin d → Fq := fun i j => (A i).coeff (HSub.hSub b.natDegree ↑ …
    this : LT.lt (Fintype.card (Fin d → Fq)) (Fintype.card (Fin m.succ))
    i₀ i₁ : Fin m.succ
    i_ne : Ne i₀ i₁
    i_eq : Eq (f i₀) (f i₁)
    j : Nat
    hj : LE.le (HSub.hSub b.natDegree d) j
    hbj : Not (LE.le b.degree ↑j)
    ⊢ Eq ((HSub.hSub (A i₁) (A i₀)).coeff j) 0
  -/
  rw [coeff_sub, sub_eq_zero]
  /-
    case neg
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Ring Fq
    d m : Nat
    hm : LE.le (HPow.hPow (Fintype.card Fq) d) m
    b : Polynomial Fq
    A : Fin m.succ → Polynomial Fq
    hA : ∀ (i : Fin m.succ), LT.lt (A i).degree b.degree
    hb : Ne b 0
    f : Fin m.succ → Fin d → Fq := fun i j => (A i).coeff (HSub.hSub b.natDegree ↑ …
    this : LT.lt (Fintype.card (Fin d → Fq)) (Fintype.card (Fin m.succ))
    i₀ i₁ : Fin m.succ
    i_ne : Ne i₀ i₁
    i_eq : Eq (f i₀) (f i₁)
    j : Nat
    hj : LE.le (HSub.hSub b.natDegree d) j
    hbj : Not (LE.le b.degree ↑j)
    ⊢ Eq ((A i₁).coeff j) ((A i₀).coeff j)
  -/
  rw [not_le, degree_eq_natDegree hb] at hbj
  /-
    case neg
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Ring Fq
    d m : Nat
    hm : LE.le (HPow.hPow (Fintype.card Fq) d) m
    b : Polynomial Fq
    A : Fin m.succ → Polynomial Fq
    hA : ∀ (i : Fin m.succ), LT.lt (A i).degree b.degree
    hb : Ne b 0
    f : Fin m.succ → Fin d → Fq := fun i j => (A i).coeff (HSub.hSub b.natDegree ↑ …
    this : LT.lt (Fintype.card (Fin d → Fq)) (Fintype.card (Fin m.succ))
    i₀ i₁ : Fin m.succ
    i_ne : Ne i₀ i₁
    i_eq : Eq (f i₀) (f i₁)
    j : Nat
    hj : LE.le (HSub.hSub b.natDegree d) j
    hbj : LT.lt ↑j ↑b.natDegree
    ⊢ Eq ((A i₁).coeff j) ((A i₀).coeff j)
  -/
  have hbj : j < natDegree b := (@WithBot.coe_lt_coe _ _ _).mp hbj
  have hj : natDegree b - j.succ < d := by
    by_cases hd : natDegree b < d
    · exact lt_of_le_of_lt tsub_le_self hd
    · rw [not_lt] at hd
      have := lt_of_le_of_lt hj (Nat.lt_succ_self j)
      rwa [tsub_lt_iff_tsub_lt hd hbj] at this
  have : j = b.natDegree - (natDegree b - j.succ).succ := by
    rw [← Nat.succ_sub hbj, Nat.succ_sub_succ, tsub_tsub_cancel_of_le hbj.le]
  /-
    case neg
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Ring Fq
    d m : Nat
    hm : LE.le (HPow.hPow (Fintype.card Fq) d) m
    b : Polynomial Fq
    A : Fin m.succ → Polynomial Fq
    hA : ∀ (i : Fin m.succ), LT.lt (A i).degree b.degree
    hb : Ne b 0
    f : Fin m.succ → Fin d → Fq := fun i j => (A i).coeff (HSub.hSub b.natDegree ↑ …
    this✝ : LT.lt (Fintype.card (Fin d → Fq)) (Fintype.card (Fin m.succ))
    i₀ i₁ : Fin m.succ
    i_ne : Ne i₀ i₁
    i_eq : Eq (f i₀) (f i₁)
    j : Nat
    hj✝ : LE.le (HSub.hSub b.natDegree d) j
    hbj✝ : LT.lt ↑j ↑b.natDegree
    hbj : LT.lt j b.natDegree
    hj : LT.lt (HSub.hSub b.natDegree j.succ) d
    this : Eq j (HSub.hSub b.natDegree (HSub.hSub b.natDegree j.succ).succ)
    ⊢ Eq ((A i₁).coeff j) ((A i₀).coeff j)
  -/
  convert congr_fun i_eq.symm ⟨natDegree b - j.succ, hj⟩
  /-
    🎉 no goals
  -/


/-- If `A` is a family of enough low-degree polynomials over a finite field,
there is a pair of elements in `A` (with different indices but not necessarily
distinct), such that the difference of their remainders is close together. -/
theorem exists_approx_polynomial {b : Fq[X]} (hb : b ≠ 0) {ε : ℝ} (hε : 0 < ε)
    (A : Fin (Fintype.card Fq ^ ⌈-log ε / log (Fintype.card Fq)⌉₊).succ → Fq[X]) :
    ∃ i₀ i₁, i₀ ≠ i₁ ∧ (cardPowDegree (A i₁ % b - A i₀ % b) : ℝ) < cardPowDegree b • ε := by
  have hbε : 0 < cardPowDegree b • ε := by
    rw [Algebra.smul_def, eq_intCast]
    exact mul_pos (Int.cast_pos.mpr (AbsoluteValue.pos _ hb)) hε
  /-
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Field Fq
    b : Polynomial Fq
    hb : Ne b 0
    ε : Real
    hε : LT.lt 0 ε
    A : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε …
    hbε : LT.lt 0 (HSMul.hSMul (Polynomial.cardPowDegree b) ε)
    ⊢ Exists fun i₀ => Exists fun i₁ => And (Ne i₀ i₁) (LT.lt (↑(Polynomial.cardPo …
  -/
  have one_lt_q : 1 < Fintype.card Fq := Fintype.one_lt_card
  /-
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Field Fq
    b : Polynomial Fq
    hb : Ne b 0
    ε : Real
    hε : LT.lt 0 ε
    A : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε …
    hbε : LT.lt 0 (HSMul.hSMul (Polynomial.cardPowDegree b) ε)
    one_lt_q : LT.lt 1 (Fintype.card Fq)
    ⊢ Exists fun i₀ => Exists fun i₁ => And (Ne i₀ i₁) (LT.lt (↑(Polynomial.cardPo …
  -/
  have one_lt_q' : (1 : ℝ) < Fintype.card Fq := by assumption_mod_cast
  /-
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Field Fq
    b : Polynomial Fq
    hb : Ne b 0
    ε : Real
    hε : LT.lt 0 ε
    A : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε …
    hbε : LT.lt 0 (HSMul.hSMul (Polynomial.cardPowDegree b) ε)
    one_lt_q : LT.lt 1 (Fintype.card Fq)
    one_lt_q' : LT.lt 1 ↑(Fintype.card Fq)
    ⊢ Exists fun i₀ => Exists fun i₁ => And (Ne i₀ i₁) (LT.lt (↑(Polynomial.cardPo …
  -/
  have q_pos : 0 < Fintype.card Fq := by omega
  /-
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Field Fq
    b : Polynomial Fq
    hb : Ne b 0
    ε : Real
    hε : LT.lt 0 ε
    A : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε …
    hbε : LT.lt 0 (HSMul.hSMul (Polynomial.cardPowDegree b) ε)
    one_lt_q : LT.lt 1 (Fintype.card Fq)
    one_lt_q' : LT.lt 1 ↑(Fintype.card Fq)
    q_pos : LT.lt 0 (Fintype.card Fq)
    ⊢ Exists fun i₀ => Exists fun i₁ => And (Ne i₀ i₁) (LT.lt (↑(Polynomial.cardPo …
  -/
  have q_pos' : (0 : ℝ) < Fintype.card Fq := by assumption_mod_cast
  -- If `b` is already small enough, then the remainders are equal and we are done.
  /-
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Field Fq
    b : Polynomial Fq
    hb : Ne b 0
    ε : Real
    hε : LT.lt 0 ε
    A : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε …
    hbε : LT.lt 0 (HSMul.hSMul (Polynomial.cardPowDegree b) ε)
    one_lt_q : LT.lt 1 (Fintype.card Fq)
    one_lt_q' : LT.lt 1 ↑(Fintype.card Fq)
    q_pos : LT.lt 0 (Fintype.card Fq)
    q_pos' : LT.lt 0 ↑(Fintype.card Fq)
    ⊢ Exists fun i₀ => Exists fun i₁ => And (Ne i₀ i₁) (LT.lt (↑(Polynomial.cardPo …
  -/
  by_cases le_b : b.natDegree ≤ ⌈-log ε / log (Fintype.card Fq)⌉₊
  · obtain ⟨i₀, i₁, i_ne, mod_eq⟩ :=
      exists_eq_polynomial le_rfl b le_b (fun i => A i % b) fun i => EuclideanDomain.mod_lt (A i) hb
    /-
      case pos.intro.intro.intro
      Fq : Type u_1
      inst✝¹ : Fintype Fq
      inst✝ : Field Fq
      b : Polynomial Fq
      hb : Ne b 0
      ε : Real
      hε : LT.lt 0 ε
      A : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε …
      hbε : LT.lt 0 (HSMul.hSMul (Polynomial.cardPowDegree b) ε)
      one_lt_q : LT.lt 1 (Fintype.card Fq)
      one_lt_q' : LT.lt 1 ↑(Fintype.card Fq)
      q_pos : LT.lt 0 (Fintype.card Fq)
      q_pos' : LT.lt 0 ↑(Fintype.card Fq)
      le_b : LE.le b.natDegree (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε)) (Real.log …
      i₀ i₁ : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.l …
      i_ne : Ne i₀ i₁
      mod_eq : Eq (HMod.hMod (A i₁) b) (HMod.hMod (A i₀) b)
      ⊢ Exists fun i₀ => Exists fun i₁ => And (Ne i₀ i₁) (LT.lt (↑(Polynomial.cardPo …
    -/
    refine ⟨i₀, i₁, i_ne, ?_⟩
    /-
      case pos.intro.intro.intro
      Fq : Type u_1
      inst✝¹ : Fintype Fq
      inst✝ : Field Fq
      b : Polynomial Fq
      hb : Ne b 0
      ε : Real
      hε : LT.lt 0 ε
      A : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε …
      hbε : LT.lt 0 (HSMul.hSMul (Polynomial.cardPowDegree b) ε)
      one_lt_q : LT.lt 1 (Fintype.card Fq)
      one_lt_q' : LT.lt 1 ↑(Fintype.card Fq)
      q_pos : LT.lt 0 (Fintype.card Fq)
      q_pos' : LT.lt 0 ↑(Fintype.card Fq)
      le_b : LE.le b.natDegree (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε)) (Real.log …
      i₀ i₁ : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.l …
      i_ne : Ne i₀ i₁
      mod_eq : Eq (HMod.hMod (A i₁) b) (HMod.hMod (A i₀) b)
      ⊢ LT.lt (↑(Polynomial.cardPowDegree (HSub.hSub (HMod.hMod (A i₁) b) (HMod.hMod …
    -/
    rwa [mod_eq, sub_self, map_zero, Int.cast_zero]
    /-
      🎉 no goals
    -/
  -- Otherwise, it suffices to choose two elements whose difference is of small enough degree.
  /-
    case neg
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Field Fq
    b : Polynomial Fq
    hb : Ne b 0
    ε : Real
    hε : LT.lt 0 ε
    A : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε …
    hbε : LT.lt 0 (HSMul.hSMul (Polynomial.cardPowDegree b) ε)
    one_lt_q : LT.lt 1 (Fintype.card Fq)
    one_lt_q' : LT.lt 1 ↑(Fintype.card Fq)
    q_pos : LT.lt 0 (Fintype.card Fq)
    q_pos' : LT.lt 0 ↑(Fintype.card Fq)
    le_b : Not (LE.le b.natDegree (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε)) (Rea …
    ⊢ Exists fun i₀ => Exists fun i₁ => And (Ne i₀ i₁) (LT.lt (↑(Polynomial.cardPo …
  -/
  rw [not_le] at le_b
  obtain ⟨i₀, i₁, i_ne, deg_lt⟩ := exists_approx_polynomial_aux le_rfl b (fun i => A i % b) fun i =>
    EuclideanDomain.mod_lt (A i) hb
  /-
    case neg.intro.intro.intro
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Field Fq
    b : Polynomial Fq
    hb : Ne b 0
    ε : Real
    hε : LT.lt 0 ε
    A : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε …
    hbε : LT.lt 0 (HSMul.hSMul (Polynomial.cardPowDegree b) ε)
    one_lt_q : LT.lt 1 (Fintype.card Fq)
    one_lt_q' : LT.lt 1 ↑(Fintype.card Fq)
    q_pos : LT.lt 0 (Fintype.card Fq)
    q_pos' : LT.lt 0 ↑(Fintype.card Fq)
    le_b : LT.lt (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε)) (Real.log ↑(Fintype.c …
    i₀ i₁ : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.l …
    i_ne : Ne i₀ i₁
    deg_lt : LT.lt (HSub.hSub (HMod.hMod (A i₁) b) (HMod.hMod (A i₀) b)).degree ↑( …
    ⊢ Exists fun i₀ => Exists fun i₁ => And (Ne i₀ i₁) (LT.lt (↑(Polynomial.cardPo …
  -/
  use i₀, i₁, i_ne
  -- Again, if the remainders are equal we are done.
  /-
    case right
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Field Fq
    b : Polynomial Fq
    hb : Ne b 0
    ε : Real
    hε : LT.lt 0 ε
    A : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε …
    hbε : LT.lt 0 (HSMul.hSMul (Polynomial.cardPowDegree b) ε)
    one_lt_q : LT.lt 1 (Fintype.card Fq)
    one_lt_q' : LT.lt 1 ↑(Fintype.card Fq)
    q_pos : LT.lt 0 (Fintype.card Fq)
    q_pos' : LT.lt 0 ↑(Fintype.card Fq)
    le_b : LT.lt (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε)) (Real.log ↑(Fintype.c …
    i₀ i₁ : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.l …
    i_ne : Ne i₀ i₁
    deg_lt : LT.lt (HSub.hSub (HMod.hMod (A i₁) b) (HMod.hMod (A i₀) b)).degree ↑( …
    ⊢ LT.lt (↑(Polynomial.cardPowDegree (HSub.hSub (HMod.hMod (A i₁) b) (HMod.hMod …
  -/
  by_cases h : A i₁ % b = A i₀ % b
    /-
      case pos
      Fq : Type u_1
      inst✝¹ : Fintype Fq
      inst✝ : Field Fq
      b : Polynomial Fq
      hb : Ne b 0
      ε : Real
      hε : LT.lt 0 ε
      A : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε …
      hbε : LT.lt 0 (HSMul.hSMul (Polynomial.cardPowDegree b) ε)
      one_lt_q : LT.lt 1 (Fintype.card Fq)
      one_lt_q' : LT.lt 1 ↑(Fintype.card Fq)
      q_pos : LT.lt 0 (Fintype.card Fq)
      q_pos' : LT.lt 0 ↑(Fintype.card Fq)
      le_b : LT.lt (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε)) (Real.log ↑(Fintype.c …
      i₀ i₁ : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.l …
      i_ne : Ne i₀ i₁
      deg_lt : LT.lt (HSub.hSub (HMod.hMod (A i₁) b) (HMod.hMod (A i₀) b)).degree ↑( …
      h : Eq (HMod.hMod (A i₁) b) (HMod.hMod (A i₀) b)
      ⊢ LT.lt (↑(Polynomial.cardPowDegree (HSub.hSub (HMod.hMod (A i₁) b) (HMod.hMod …
    -/
  · rwa [h, sub_self, map_zero, Int.cast_zero]
    /-
      🎉 no goals
    -/
  /-
    case neg
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Field Fq
    b : Polynomial Fq
    hb : Ne b 0
    ε : Real
    hε : LT.lt 0 ε
    A : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε …
    hbε : LT.lt 0 (HSMul.hSMul (Polynomial.cardPowDegree b) ε)
    one_lt_q : LT.lt 1 (Fintype.card Fq)
    one_lt_q' : LT.lt 1 ↑(Fintype.card Fq)
    q_pos : LT.lt 0 (Fintype.card Fq)
    q_pos' : LT.lt 0 ↑(Fintype.card Fq)
    le_b : LT.lt (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε)) (Real.log ↑(Fintype.c …
    i₀ i₁ : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.l …
    i_ne : Ne i₀ i₁
    deg_lt : LT.lt (HSub.hSub (HMod.hMod (A i₁) b) (HMod.hMod (A i₀) b)).degree ↑( …
    h : Not (Eq (HMod.hMod (A i₁) b) (HMod.hMod (A i₀) b))
    ⊢ LT.lt (↑(Polynomial.cardPowDegree (HSub.hSub (HMod.hMod (A i₁) b) (HMod.hMod …
  -/
  have h' : A i₁ % b - A i₀ % b ≠ 0 := mt sub_eq_zero.mp h
  -- If the remainders are not equal, we'll show their difference is of small degree.
  -- In particular, we'll show the degree is less than the following:
  suffices (natDegree (A i₁ % b - A i₀ % b) : ℝ) < b.natDegree + log ε / log (Fintype.card Fq) by
    rwa [← Real.log_lt_log_iff (Int.cast_pos.mpr (cardPowDegree.pos h')) hbε,
      cardPowDegree_nonzero _ h', cardPowDegree_nonzero _ hb, Algebra.smul_def, eq_intCast,
      Int.cast_pow, Int.cast_natCast, Int.cast_pow, Int.cast_natCast,
      log_mul (pow_ne_zero _ q_pos'.ne') hε.ne', ← rpow_natCast, ← rpow_natCast, log_rpow q_pos',
      log_rpow q_pos', ← lt_div_iff₀ (log_pos one_lt_q'), add_div,
      mul_div_cancel_right₀ _ (log_pos one_lt_q').ne']
  -- And that result follows from manipulating the result from `exists_approx_polynomial_aux`
  -- to turn the `-⌈-stuff⌉₊` into `+ stuff`.
  /-
    case neg
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Field Fq
    b : Polynomial Fq
    hb : Ne b 0
    ε : Real
    hε : LT.lt 0 ε
    A : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε …
    hbε : LT.lt 0 (HSMul.hSMul (Polynomial.cardPowDegree b) ε)
    one_lt_q : LT.lt 1 (Fintype.card Fq)
    one_lt_q' : LT.lt 1 ↑(Fintype.card Fq)
    q_pos : LT.lt 0 (Fintype.card Fq)
    q_pos' : LT.lt 0 ↑(Fintype.card Fq)
    le_b : LT.lt (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε)) (Real.log ↑(Fintype.c …
    i₀ i₁ : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.l …
    i_ne : Ne i₀ i₁
    deg_lt : LT.lt (HSub.hSub (HMod.hMod (A i₁) b) (HMod.hMod (A i₀) b)).degree ↑( …
    h : Not (Eq (HMod.hMod (A i₁) b) (HMod.hMod (A i₀) b))
    h' : Ne (HSub.hSub (HMod.hMod (A i₁) b) (HMod.hMod (A i₀) b)) 0
    ⊢ LT.lt (↑(HSub.hSub (HMod.hMod (A i₁) b) (HMod.hMod (A i₀) b)).natDegree) (HA …
  -/
  apply lt_of_lt_of_le (Nat.cast_lt.mpr (WithBot.coe_lt_coe.mp _)) _
  /-
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Field Fq
    b : Polynomial Fq
    hb : Ne b 0
    ε : Real
    hε : LT.lt 0 ε
    A : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε …
    hbε : LT.lt 0 (HSMul.hSMul (Polynomial.cardPowDegree b) ε)
    one_lt_q : LT.lt 1 (Fintype.card Fq)
    one_lt_q' : LT.lt 1 ↑(Fintype.card Fq)
    q_pos : LT.lt 0 (Fintype.card Fq)
    q_pos' : LT.lt 0 ↑(Fintype.card Fq)
    le_b : LT.lt (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε)) (Real.log ↑(Fintype.c …
    i₀ i₁ : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.l …
    i_ne : Ne i₀ i₁
    deg_lt : LT.lt (HSub.hSub (HMod.hMod (A i₁) b) (HMod.hMod (A i₀) b)).degree ↑( …
    h : Not (Eq (HMod.hMod (A i₁) b) (HMod.hMod (A i₀) b))
    h' : Ne (HSub.hSub (HMod.hMod (A i₁) b) (HMod.hMod (A i₀) b)) 0
    ⊢ Nat
  -/
  swap
    /-
      Fq : Type u_1
      inst✝¹ : Fintype Fq
      inst✝ : Field Fq
      b : Polynomial Fq
      hb : Ne b 0
      ε : Real
      hε : LT.lt 0 ε
      A : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε …
      hbε : LT.lt 0 (HSMul.hSMul (Polynomial.cardPowDegree b) ε)
      one_lt_q : LT.lt 1 (Fintype.card Fq)
      one_lt_q' : LT.lt 1 ↑(Fintype.card Fq)
      q_pos : LT.lt 0 (Fintype.card Fq)
      q_pos' : LT.lt 0 ↑(Fintype.card Fq)
      le_b : LT.lt (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε)) (Real.log ↑(Fintype.c …
      i₀ i₁ : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.l …
      i_ne : Ne i₀ i₁
      deg_lt : LT.lt (HSub.hSub (HMod.hMod (A i₁) b) (HMod.hMod (A i₀) b)).degree ↑( …
      h : Not (Eq (HMod.hMod (A i₁) b) (HMod.hMod (A i₀) b))
      h' : Ne (HSub.hSub (HMod.hMod (A i₁) b) (HMod.hMod (A i₀) b)) 0
      ⊢ LT.lt ↑(HSub.hSub (HMod.hMod (A i₁) b) (HMod.hMod (A i₀) b)).natDegree ↑?m.4 …
    -/
  · convert deg_lt
    /-
      case h.e'_3
      Fq : Type u_1
      inst✝¹ : Fintype Fq
      inst✝ : Field Fq
      b : Polynomial Fq
      hb : Ne b 0
      ε : Real
      hε : LT.lt 0 ε
      A : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε …
      hbε : LT.lt 0 (HSMul.hSMul (Polynomial.cardPowDegree b) ε)
      one_lt_q : LT.lt 1 (Fintype.card Fq)
      one_lt_q' : LT.lt 1 ↑(Fintype.card Fq)
      q_pos : LT.lt 0 (Fintype.card Fq)
      q_pos' : LT.lt 0 ↑(Fintype.card Fq)
      le_b : LT.lt (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε)) (Real.log ↑(Fintype.c …
      i₀ i₁ : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.l …
      i_ne : Ne i₀ i₁
      deg_lt : LT.lt (HSub.hSub (HMod.hMod (A i₁) b) (HMod.hMod (A i₀) b)).degree ↑( …
      h : Not (Eq (HMod.hMod (A i₁) b) (HMod.hMod (A i₀) b))
      h' : Ne (HSub.hSub (HMod.hMod (A i₁) b) (HMod.hMod (A i₀) b)) 0
      ⊢ Eq (↑(HSub.hSub (HMod.hMod (A i₁) b) (HMod.hMod (A i₀) b)).natDegree) (HSub. …
    -/
    rw [degree_eq_natDegree h']; rfl
                                 /-
                                   🎉 no goals
                                 -/
  /-
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Field Fq
    b : Polynomial Fq
    hb : Ne b 0
    ε : Real
    hε : LT.lt 0 ε
    A : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε …
    hbε : LT.lt 0 (HSMul.hSMul (Polynomial.cardPowDegree b) ε)
    one_lt_q : LT.lt 1 (Fintype.card Fq)
    one_lt_q' : LT.lt 1 ↑(Fintype.card Fq)
    q_pos : LT.lt 0 (Fintype.card Fq)
    q_pos' : LT.lt 0 ↑(Fintype.card Fq)
    le_b : LT.lt (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε)) (Real.log ↑(Fintype.c …
    i₀ i₁ : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.l …
    i_ne : Ne i₀ i₁
    deg_lt : LT.lt (HSub.hSub (HMod.hMod (A i₁) b) (HMod.hMod (A i₀) b)).degree ↑( …
    h : Not (Eq (HMod.hMod (A i₁) b) (HMod.hMod (A i₀) b))
    h' : Ne (HSub.hSub (HMod.hMod (A i₁) b) (HMod.hMod (A i₀) b)) 0
    ⊢ LE.le (↑(HSub.hSub b.natDegree (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε)) ( …
  -/
  rw [← sub_neg_eq_add, neg_div]
  /-
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Field Fq
    b : Polynomial Fq
    hb : Ne b 0
    ε : Real
    hε : LT.lt 0 ε
    A : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε …
    hbε : LT.lt 0 (HSMul.hSMul (Polynomial.cardPowDegree b) ε)
    one_lt_q : LT.lt 1 (Fintype.card Fq)
    one_lt_q' : LT.lt 1 ↑(Fintype.card Fq)
    q_pos : LT.lt 0 (Fintype.card Fq)
    q_pos' : LT.lt 0 ↑(Fintype.card Fq)
    le_b : LT.lt (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε)) (Real.log ↑(Fintype.c …
    i₀ i₁ : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.l …
    i_ne : Ne i₀ i₁
    deg_lt : LT.lt (HSub.hSub (HMod.hMod (A i₁) b) (HMod.hMod (A i₀) b)).degree ↑( …
    h : Not (Eq (HMod.hMod (A i₁) b) (HMod.hMod (A i₀) b))
    h' : Ne (HSub.hSub (HMod.hMod (A i₁) b) (HMod.hMod (A i₀) b)) 0
    ⊢ LE.le (↑(HSub.hSub b.natDegree (Nat.ceil (Neg.neg (HDiv.hDiv (Real.log ε) (R …
  -/
  refine le_trans ?_ (sub_le_sub_left (Nat.le_ceil _) (b.natDegree : ℝ))
  /-
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Field Fq
    b : Polynomial Fq
    hb : Ne b 0
    ε : Real
    hε : LT.lt 0 ε
    A : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε …
    hbε : LT.lt 0 (HSMul.hSMul (Polynomial.cardPowDegree b) ε)
    one_lt_q : LT.lt 1 (Fintype.card Fq)
    one_lt_q' : LT.lt 1 ↑(Fintype.card Fq)
    q_pos : LT.lt 0 (Fintype.card Fq)
    q_pos' : LT.lt 0 ↑(Fintype.card Fq)
    le_b : LT.lt (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε)) (Real.log ↑(Fintype.c …
    i₀ i₁ : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.l …
    i_ne : Ne i₀ i₁
    deg_lt : LT.lt (HSub.hSub (HMod.hMod (A i₁) b) (HMod.hMod (A i₀) b)).degree ↑( …
    h : Not (Eq (HMod.hMod (A i₁) b) (HMod.hMod (A i₀) b))
    h' : Ne (HSub.hSub (HMod.hMod (A i₁) b) (HMod.hMod (A i₀) b)) 0
    ⊢ LE.le (↑(HSub.hSub b.natDegree (Nat.ceil (Neg.neg (HDiv.hDiv (Real.log ε) (R …
  -/
  rw [← neg_div]
  /-
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Field Fq
    b : Polynomial Fq
    hb : Ne b 0
    ε : Real
    hε : LT.lt 0 ε
    A : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε …
    hbε : LT.lt 0 (HSMul.hSMul (Polynomial.cardPowDegree b) ε)
    one_lt_q : LT.lt 1 (Fintype.card Fq)
    one_lt_q' : LT.lt 1 ↑(Fintype.card Fq)
    q_pos : LT.lt 0 (Fintype.card Fq)
    q_pos' : LT.lt 0 ↑(Fintype.card Fq)
    le_b : LT.lt (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε)) (Real.log ↑(Fintype.c …
    i₀ i₁ : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.l …
    i_ne : Ne i₀ i₁
    deg_lt : LT.lt (HSub.hSub (HMod.hMod (A i₁) b) (HMod.hMod (A i₀) b)).degree ↑( …
    h : Not (Eq (HMod.hMod (A i₁) b) (HMod.hMod (A i₀) b))
    h' : Ne (HSub.hSub (HMod.hMod (A i₁) b) (HMod.hMod (A i₀) b)) 0
    ⊢ LE.le (↑(HSub.hSub b.natDegree (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε)) ( …
  -/
  exact le_of_eq (Nat.cast_sub le_b.le)
  /-
    🎉 no goals
  -/


/-- If `x` is close to `y` and `y` is close to `z`, then `x` and `z` are at least as close. -/
theorem cardPowDegree_anti_archimedean {x y z : Fq[X]} {a : ℤ} (hxy : cardPowDegree (x - y) < a)
    (hyz : cardPowDegree (y - z) < a) : cardPowDegree (x - z) < a := by
  /-
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Field Fq
    x y z : Polynomial Fq
    a : Int
    hxy : LT.lt (Polynomial.cardPowDegree (HSub.hSub x y)) a
    hyz : LT.lt (Polynomial.cardPowDegree (HSub.hSub y z)) a
    ⊢ LT.lt (Polynomial.cardPowDegree (HSub.hSub x z)) a
  -/
  have ha : 0 < a := lt_of_le_of_lt (AbsoluteValue.nonneg _ _) hxy
  /-
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Field Fq
    x y z : Polynomial Fq
    a : Int
    hxy : LT.lt (Polynomial.cardPowDegree (HSub.hSub x y)) a
    hyz : LT.lt (Polynomial.cardPowDegree (HSub.hSub y z)) a
    ha : LT.lt 0 a
    ⊢ LT.lt (Polynomial.cardPowDegree (HSub.hSub x z)) a
  -/
  by_cases hxy' : x = y
    /-
      case pos
      Fq : Type u_1
      inst✝¹ : Fintype Fq
      inst✝ : Field Fq
      x y z : Polynomial Fq
      a : Int
      hxy : LT.lt (Polynomial.cardPowDegree (HSub.hSub x y)) a
      hyz : LT.lt (Polynomial.cardPowDegree (HSub.hSub y z)) a
      ha : LT.lt 0 a
      hxy' : Eq x y
      ⊢ LT.lt (Polynomial.cardPowDegree (HSub.hSub x z)) a
    -/
  · rwa [hxy']
    /-
      🎉 no goals
    -/
  /-
    case neg
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Field Fq
    x y z : Polynomial Fq
    a : Int
    hxy : LT.lt (Polynomial.cardPowDegree (HSub.hSub x y)) a
    hyz : LT.lt (Polynomial.cardPowDegree (HSub.hSub y z)) a
    ha : LT.lt 0 a
    hxy' : Not (Eq x y)
    ⊢ LT.lt (Polynomial.cardPowDegree (HSub.hSub x z)) a
  -/
  by_cases hyz' : y = z
    /-
      case pos
      Fq : Type u_1
      inst✝¹ : Fintype Fq
      inst✝ : Field Fq
      x y z : Polynomial Fq
      a : Int
      hxy : LT.lt (Polynomial.cardPowDegree (HSub.hSub x y)) a
      hyz : LT.lt (Polynomial.cardPowDegree (HSub.hSub y z)) a
      ha : LT.lt 0 a
      hxy' : Not (Eq x y)
      hyz' : Eq y z
      ⊢ LT.lt (Polynomial.cardPowDegree (HSub.hSub x z)) a
    -/
  · rwa [← hyz']
    /-
      🎉 no goals
    -/
  /-
    case neg
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Field Fq
    x y z : Polynomial Fq
    a : Int
    hxy : LT.lt (Polynomial.cardPowDegree (HSub.hSub x y)) a
    hyz : LT.lt (Polynomial.cardPowDegree (HSub.hSub y z)) a
    ha : LT.lt 0 a
    hxy' : Not (Eq x y)
    hyz' : Not (Eq y z)
    ⊢ LT.lt (Polynomial.cardPowDegree (HSub.hSub x z)) a
  -/
  by_cases hxz' : x = z
    /-
      case pos
      Fq : Type u_1
      inst✝¹ : Fintype Fq
      inst✝ : Field Fq
      x y z : Polynomial Fq
      a : Int
      hxy : LT.lt (Polynomial.cardPowDegree (HSub.hSub x y)) a
      hyz : LT.lt (Polynomial.cardPowDegree (HSub.hSub y z)) a
      ha : LT.lt 0 a
      hxy' : Not (Eq x y)
      hyz' : Not (Eq y z)
      hxz' : Eq x z
      ⊢ LT.lt (Polynomial.cardPowDegree (HSub.hSub x z)) a
    -/
  · rwa [hxz', sub_self, map_zero]
    /-
      🎉 no goals
    -/
  /-
    case neg
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Field Fq
    x y z : Polynomial Fq
    a : Int
    hxy : LT.lt (Polynomial.cardPowDegree (HSub.hSub x y)) a
    hyz : LT.lt (Polynomial.cardPowDegree (HSub.hSub y z)) a
    ha : LT.lt 0 a
    hxy' : Not (Eq x y)
    hyz' : Not (Eq y z)
    hxz' : Not (Eq x z)
    ⊢ LT.lt (Polynomial.cardPowDegree (HSub.hSub x z)) a
  -/
  rw [← Ne, ← sub_ne_zero] at hxy' hyz' hxz'
  /-
    case neg
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Field Fq
    x y z : Polynomial Fq
    a : Int
    hxy : LT.lt (Polynomial.cardPowDegree (HSub.hSub x y)) a
    hyz : LT.lt (Polynomial.cardPowDegree (HSub.hSub y z)) a
    ha : LT.lt 0 a
    hxy' : Ne (HSub.hSub x y) 0
    hyz' : Ne (HSub.hSub y z) 0
    hxz' : Ne (HSub.hSub x z) 0
    ⊢ LT.lt (Polynomial.cardPowDegree (HSub.hSub x z)) a
  -/
  refine lt_of_le_of_lt ?_ (max_lt hxy hyz)
  rw [cardPowDegree_nonzero _ hxz', cardPowDegree_nonzero _ hxy',
    cardPowDegree_nonzero _ hyz']
  /-
    case neg
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Field Fq
    x y z : Polynomial Fq
    a : Int
    hxy : LT.lt (Polynomial.cardPowDegree (HSub.hSub x y)) a
    hyz : LT.lt (Polynomial.cardPowDegree (HSub.hSub y z)) a
    ha : LT.lt 0 a
    hxy' : Ne (HSub.hSub x y) 0
    hyz' : Ne (HSub.hSub y z) 0
    hxz' : Ne (HSub.hSub x z) 0
    ⊢ LE.le (HPow.hPow (↑(Fintype.card Fq)) (HSub.hSub x z).natDegree) (Max.max (H …
  -/
  have : (1 : ℤ) ≤ Fintype.card Fq := mod_cast (@Fintype.one_lt_card Fq _ _).le
  /-
    case neg
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Field Fq
    x y z : Polynomial Fq
    a : Int
    hxy : LT.lt (Polynomial.cardPowDegree (HSub.hSub x y)) a
    hyz : LT.lt (Polynomial.cardPowDegree (HSub.hSub y z)) a
    ha : LT.lt 0 a
    hxy' : Ne (HSub.hSub x y) 0
    hyz' : Ne (HSub.hSub y z) 0
    hxz' : Ne (HSub.hSub x z) 0
    this : LE.le 1 ↑(Fintype.card Fq)
    ⊢ LE.le (HPow.hPow (↑(Fintype.card Fq)) (HSub.hSub x z).natDegree) (Max.max (H …
  -/
  simp only [Int.cast_pow, Int.cast_natCast, le_max_iff]
  /-
    case neg
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Field Fq
    x y z : Polynomial Fq
    a : Int
    hxy : LT.lt (Polynomial.cardPowDegree (HSub.hSub x y)) a
    hyz : LT.lt (Polynomial.cardPowDegree (HSub.hSub y z)) a
    ha : LT.lt 0 a
    hxy' : Ne (HSub.hSub x y) 0
    hyz' : Ne (HSub.hSub y z) 0
    hxz' : Ne (HSub.hSub x z) 0
    this : LE.le 1 ↑(Fintype.card Fq)
    ⊢ Or (LE.le (HPow.hPow (↑(Fintype.card Fq)) (HSub.hSub x z).natDegree) (HPow.h …
  -/
  refine Or.imp (pow_le_pow_right₀ this) (pow_le_pow_right₀ this) ?_
  rw [natDegree_le_iff_degree_le, natDegree_le_iff_degree_le, ← le_max_iff, ←
    degree_eq_natDegree hxy', ← degree_eq_natDegree hyz']
  /-
    case neg
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Field Fq
    x y z : Polynomial Fq
    a : Int
    hxy : LT.lt (Polynomial.cardPowDegree (HSub.hSub x y)) a
    hyz : LT.lt (Polynomial.cardPowDegree (HSub.hSub y z)) a
    ha : LT.lt 0 a
    hxy' : Ne (HSub.hSub x y) 0
    hyz' : Ne (HSub.hSub y z) 0
    hxz' : Ne (HSub.hSub x z) 0
    this : LE.le 1 ↑(Fintype.card Fq)
    ⊢ LE.le (HSub.hSub x z).degree (Max.max (HSub.hSub x y).degree (HSub.hSub y z) …
  -/
  convert degree_add_le (x - y) (y - z) using 2
  /-
    case h.e'_3.h.e'_3
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Field Fq
    x y z : Polynomial Fq
    a : Int
    hxy : LT.lt (Polynomial.cardPowDegree (HSub.hSub x y)) a
    hyz : LT.lt (Polynomial.cardPowDegree (HSub.hSub y z)) a
    ha : LT.lt 0 a
    hxy' : Ne (HSub.hSub x y) 0
    hyz' : Ne (HSub.hSub y z) 0
    hxz' : Ne (HSub.hSub x z) 0
    this : LE.le 1 ↑(Fintype.card Fq)
    ⊢ Eq (HSub.hSub x z) (HAdd.hAdd (HSub.hSub x y) (HSub.hSub y z))
  -/
  exact (sub_add_sub_cancel _ _ _).symm
  /-
    🎉 no goals
  -/


/-- A slightly stronger version of `exists_partition` on which we perform induction on `n`:
for all `ε > 0`, we can partition the remainders of any family of polynomials `A`
into equivalence classes, where the equivalence(!) relation is "closer than `ε`". -/
theorem exists_partition_polynomial_aux (n : ℕ) {ε : ℝ} (hε : 0 < ε) {b : Fq[X]} (hb : b ≠ 0)
    (A : Fin n → Fq[X]) : ∃ t : Fin n → Fin (Fintype.card Fq ^ ⌈-log ε / log (Fintype.card Fq)⌉₊),
      ∀ i₀ i₁ : Fin n, t i₀ = t i₁ ↔
        (cardPowDegree (A i₁ % b - A i₀ % b) : ℝ) < cardPowDegree b • ε := by
  have hbε : 0 < cardPowDegree b • ε := by
    rw [Algebra.smul_def, eq_intCast]
    exact mul_pos (Int.cast_pos.mpr (AbsoluteValue.pos _ hb)) hε
  -- We go by induction on the size `A`.
  /-
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Field Fq
    n : Nat
    ε : Real
    hε : LT.lt 0 ε
    b : Polynomial Fq
    hb : Ne b 0
    A : Fin n → Polynomial Fq
    hbε : LT.lt 0 (HSMul.hSMul (Polynomial.cardPowDegree b) ε)
    ⊢ Exists fun t => ∀ (i₀ i₁ : Fin n), Iff (Eq (t i₀) (t i₁)) (LT.lt (↑(Polynomi …
  -/
  induction' n with n ih
    /-
      case zero
      Fq : Type u_1
      inst✝¹ : Fintype Fq
      inst✝ : Field Fq
      ε : Real
      hε : LT.lt 0 ε
      b : Polynomial Fq
      hb : Ne b 0
      hbε : LT.lt 0 (HSMul.hSMul (Polynomial.cardPowDegree b) ε)
      A : Fin 0 → Polynomial Fq
      ⊢ Exists fun t => ∀ (i₀ i₁ : Fin 0), Iff (Eq (t i₀) (t i₁)) (LT.lt (↑(Polynomi …
    -/
  · refine ⟨finZeroElim, finZeroElim⟩
    /-
      🎉 no goals
    -/
  -- Show `anti_archimedean` also holds for real distances.
  have anti_archim' : ∀ {i j k} {ε : ℝ},
    (cardPowDegree (A i % b - A j % b) : ℝ) < ε →
      (cardPowDegree (A j % b - A k % b) : ℝ) < ε →
        (cardPowDegree (A i % b - A k % b) : ℝ) < ε := by
    intro i j k ε
    simp_rw [← Int.lt_ceil]
    exact cardPowDegree_anti_archimedean
  /-
    case succ
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Field Fq
    ε : Real
    hε : LT.lt 0 ε
    b : Polynomial Fq
    hb : Ne b 0
    hbε : LT.lt 0 (HSMul.hSMul (Polynomial.cardPowDegree b) ε)
    n : Nat
    ih : ∀ (A : Fin n → Polynomial Fq), Exists fun t => ∀ (i₀ i₁ : Fin n), Iff (Eq …
    A : Fin (HAdd.hAdd n 1) → Polynomial Fq
    anti_archim' : ∀ {i j k : Fin (HAdd.hAdd n 1)} {ε : Real}, LT.lt (↑(Polynomial …
    ⊢ Exists fun t => ∀ (i₀ i₁ : Fin (HAdd.hAdd n 1)), Iff (Eq (t i₀) (t i₁)) (LT. …
  -/
  obtain ⟨t', ht'⟩ := ih (Fin.tail A)
  -- We got rid of `A 0`, so determine the index `j` of the partition we'll re-add it to.
  rsuffices ⟨j, hj⟩ :
    ∃ j, ∀ i, t' i = j ↔ (cardPowDegree (A 0 % b - A i.succ % b) : ℝ) < cardPowDegree b • ε
    /-
      case succ.intro.intro
      Fq : Type u_1
      inst✝¹ : Fintype Fq
      inst✝ : Field Fq
      ε : Real
      hε : LT.lt 0 ε
      b : Polynomial Fq
      hb : Ne b 0
      hbε : LT.lt 0 (HSMul.hSMul (Polynomial.cardPowDegree b) ε)
      n : Nat
      ih : ∀ (A : Fin n → Polynomial Fq), Exists fun t => ∀ (i₀ i₁ : Fin n), Iff (Eq …
      A : Fin (HAdd.hAdd n 1) → Polynomial Fq
      anti_archim' : ∀ {i j k : Fin (HAdd.hAdd n 1)} {ε : Real}, LT.lt (↑(Polynomial …
      t' : Fin n → Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (R …
      ht' : ∀ (i₀ i₁ : Fin n), Iff (Eq (t' i₀) (t' i₁)) (LT.lt (↑(Polynomial.cardPow …
      j : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε …
      hj : ∀ (i : Fin n), Iff (Eq (t' i) j) (LT.lt (↑(Polynomial.cardPowDegree (HSub …
      ⊢ Exists fun t => ∀ (i₀ i₁ : Fin (HAdd.hAdd n 1)), Iff (Eq (t i₀) (t i₁)) (LT. …
    -/
  · refine ⟨Fin.cons j t', fun i₀ i₁ => ?_⟩
    /-
      case succ.intro.intro
      Fq : Type u_1
      inst✝¹ : Fintype Fq
      inst✝ : Field Fq
      ε : Real
      hε : LT.lt 0 ε
      b : Polynomial Fq
      hb : Ne b 0
      hbε : LT.lt 0 (HSMul.hSMul (Polynomial.cardPowDegree b) ε)
      n : Nat
      ih : ∀ (A : Fin n → Polynomial Fq), Exists fun t => ∀ (i₀ i₁ : Fin n), Iff (Eq …
      A : Fin (HAdd.hAdd n 1) → Polynomial Fq
      anti_archim' : ∀ {i j k : Fin (HAdd.hAdd n 1)} {ε : Real}, LT.lt (↑(Polynomial …
      t' : Fin n → Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (R …
      ht' : ∀ (i₀ i₁ : Fin n), Iff (Eq (t' i₀) (t' i₁)) (LT.lt (↑(Polynomial.cardPow …
      j : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε …
      hj : ∀ (i : Fin n), Iff (Eq (t' i) j) (LT.lt (↑(Polynomial.cardPowDegree (HSub …
      i₀ i₁ : Fin (HAdd.hAdd n 1)
      ⊢ Iff (Eq (Fin.cons j t' i₀) (Fin.cons j t' i₁)) (LT.lt (↑(Polynomial.cardPowD …
    -/
    refine Fin.cases ?_ (fun i₀ => ?_) i₀ <;> refine Fin.cases ?_ (fun i₁ => ?_) i₁
      /-
        case succ.intro.intro.refine_1.refine_1
        Fq : Type u_1
        inst✝¹ : Fintype Fq
        inst✝ : Field Fq
        ε : Real
        hε : LT.lt 0 ε
        b : Polynomial Fq
        hb : Ne b 0
        hbε : LT.lt 0 (HSMul.hSMul (Polynomial.cardPowDegree b) ε)
        n : Nat
        ih : ∀ (A : Fin n → Polynomial Fq), Exists fun t => ∀ (i₀ i₁ : Fin n), Iff (Eq …
        A : Fin (HAdd.hAdd n 1) → Polynomial Fq
        anti_archim' : ∀ {i j k : Fin (HAdd.hAdd n 1)} {ε : Real}, LT.lt (↑(Polynomial …
        t' : Fin n → Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (R …
        ht' : ∀ (i₀ i₁ : Fin n), Iff (Eq (t' i₀) (t' i₁)) (LT.lt (↑(Polynomial.cardPow …
        j : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε …
        hj : ∀ (i : Fin n), Iff (Eq (t' i) j) (LT.lt (↑(Polynomial.cardPowDegree (HSub …
        i₀ i₁ : Fin (HAdd.hAdd n 1)
        ⊢ Iff (Eq (Fin.cons j t' 0) (Fin.cons j t' 0)) (LT.lt (↑(Polynomial.cardPowDeg …
      -/
    · simpa using hbε
      /-
        🎉 no goals
      -/
      /-
        case succ.intro.intro.refine_1.refine_2
        Fq : Type u_1
        inst✝¹ : Fintype Fq
        inst✝ : Field Fq
        ε : Real
        hε : LT.lt 0 ε
        b : Polynomial Fq
        hb : Ne b 0
        hbε : LT.lt 0 (HSMul.hSMul (Polynomial.cardPowDegree b) ε)
        n : Nat
        ih : ∀ (A : Fin n → Polynomial Fq), Exists fun t => ∀ (i₀ i₁ : Fin n), Iff (Eq …
        A : Fin (HAdd.hAdd n 1) → Polynomial Fq
        anti_archim' : ∀ {i j k : Fin (HAdd.hAdd n 1)} {ε : Real}, LT.lt (↑(Polynomial …
        t' : Fin n → Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (R …
        ht' : ∀ (i₀ i₁ : Fin n), Iff (Eq (t' i₀) (t' i₁)) (LT.lt (↑(Polynomial.cardPow …
        j : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε …
        hj : ∀ (i : Fin n), Iff (Eq (t' i) j) (LT.lt (↑(Polynomial.cardPowDegree (HSub …
        i₀ i₁✝ : Fin (HAdd.hAdd n 1)
        i₁ : Fin n
        ⊢ Iff (Eq (Fin.cons j t' 0) (Fin.cons j t' i₁.succ)) (LT.lt (↑(Polynomial.card …
      -/
    · rw [Fin.cons_succ, Fin.cons_zero, eq_comm, AbsoluteValue.map_sub]
      /-
        case succ.intro.intro.refine_1.refine_2
        Fq : Type u_1
        inst✝¹ : Fintype Fq
        inst✝ : Field Fq
        ε : Real
        hε : LT.lt 0 ε
        b : Polynomial Fq
        hb : Ne b 0
        hbε : LT.lt 0 (HSMul.hSMul (Polynomial.cardPowDegree b) ε)
        n : Nat
        ih : ∀ (A : Fin n → Polynomial Fq), Exists fun t => ∀ (i₀ i₁ : Fin n), Iff (Eq …
        A : Fin (HAdd.hAdd n 1) → Polynomial Fq
        anti_archim' : ∀ {i j k : Fin (HAdd.hAdd n 1)} {ε : Real}, LT.lt (↑(Polynomial …
        t' : Fin n → Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (R …
        ht' : ∀ (i₀ i₁ : Fin n), Iff (Eq (t' i₀) (t' i₁)) (LT.lt (↑(Polynomial.cardPow …
        j : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε …
        hj : ∀ (i : Fin n), Iff (Eq (t' i) j) (LT.lt (↑(Polynomial.cardPowDegree (HSub …
        i₀ i₁✝ : Fin (HAdd.hAdd n 1)
        i₁ : Fin n
        ⊢ Iff (Eq (t' i₁) j) (LT.lt (↑(Polynomial.cardPowDegree (HSub.hSub (HMod.hMod  …
      -/
      exact hj i₁
      /-
        🎉 no goals
      -/
      /-
        case succ.intro.intro.refine_2.refine_1
        Fq : Type u_1
        inst✝¹ : Fintype Fq
        inst✝ : Field Fq
        ε : Real
        hε : LT.lt 0 ε
        b : Polynomial Fq
        hb : Ne b 0
        hbε : LT.lt 0 (HSMul.hSMul (Polynomial.cardPowDegree b) ε)
        n : Nat
        ih : ∀ (A : Fin n → Polynomial Fq), Exists fun t => ∀ (i₀ i₁ : Fin n), Iff (Eq …
        A : Fin (HAdd.hAdd n 1) → Polynomial Fq
        anti_archim' : ∀ {i j k : Fin (HAdd.hAdd n 1)} {ε : Real}, LT.lt (↑(Polynomial …
        t' : Fin n → Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (R …
        ht' : ∀ (i₀ i₁ : Fin n), Iff (Eq (t' i₀) (t' i₁)) (LT.lt (↑(Polynomial.cardPow …
        j : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε …
        hj : ∀ (i : Fin n), Iff (Eq (t' i) j) (LT.lt (↑(Polynomial.cardPowDegree (HSub …
        i₀✝ i₁ : Fin (HAdd.hAdd n 1)
        i₀ : Fin n
        ⊢ Iff (Eq (Fin.cons j t' i₀.succ) (Fin.cons j t' 0)) (LT.lt (↑(Polynomial.card …
      -/
    · rw [Fin.cons_succ, Fin.cons_zero]
      /-
        case succ.intro.intro.refine_2.refine_1
        Fq : Type u_1
        inst✝¹ : Fintype Fq
        inst✝ : Field Fq
        ε : Real
        hε : LT.lt 0 ε
        b : Polynomial Fq
        hb : Ne b 0
        hbε : LT.lt 0 (HSMul.hSMul (Polynomial.cardPowDegree b) ε)
        n : Nat
        ih : ∀ (A : Fin n → Polynomial Fq), Exists fun t => ∀ (i₀ i₁ : Fin n), Iff (Eq …
        A : Fin (HAdd.hAdd n 1) → Polynomial Fq
        anti_archim' : ∀ {i j k : Fin (HAdd.hAdd n 1)} {ε : Real}, LT.lt (↑(Polynomial …
        t' : Fin n → Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (R …
        ht' : ∀ (i₀ i₁ : Fin n), Iff (Eq (t' i₀) (t' i₁)) (LT.lt (↑(Polynomial.cardPow …
        j : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε …
        hj : ∀ (i : Fin n), Iff (Eq (t' i) j) (LT.lt (↑(Polynomial.cardPowDegree (HSub …
        i₀✝ i₁ : Fin (HAdd.hAdd n 1)
        i₀ : Fin n
        ⊢ Iff (Eq (t' i₀) j) (LT.lt (↑(Polynomial.cardPowDegree (HSub.hSub (HMod.hMod  …
      -/
      exact hj i₀
      /-
        🎉 no goals
      -/
      /-
        case succ.intro.intro.refine_2.refine_2
        Fq : Type u_1
        inst✝¹ : Fintype Fq
        inst✝ : Field Fq
        ε : Real
        hε : LT.lt 0 ε
        b : Polynomial Fq
        hb : Ne b 0
        hbε : LT.lt 0 (HSMul.hSMul (Polynomial.cardPowDegree b) ε)
        n : Nat
        ih : ∀ (A : Fin n → Polynomial Fq), Exists fun t => ∀ (i₀ i₁ : Fin n), Iff (Eq …
        A : Fin (HAdd.hAdd n 1) → Polynomial Fq
        anti_archim' : ∀ {i j k : Fin (HAdd.hAdd n 1)} {ε : Real}, LT.lt (↑(Polynomial …
        t' : Fin n → Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (R …
        ht' : ∀ (i₀ i₁ : Fin n), Iff (Eq (t' i₀) (t' i₁)) (LT.lt (↑(Polynomial.cardPow …
        j : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε …
        hj : ∀ (i : Fin n), Iff (Eq (t' i) j) (LT.lt (↑(Polynomial.cardPowDegree (HSub …
        i₀✝ i₁✝ : Fin (HAdd.hAdd n 1)
        i₀ i₁ : Fin n
        ⊢ Iff (Eq (Fin.cons j t' i₀.succ) (Fin.cons j t' i₁.succ)) (LT.lt (↑(Polynomia …
      -/
    · rw [Fin.cons_succ, Fin.cons_succ]
      /-
        case succ.intro.intro.refine_2.refine_2
        Fq : Type u_1
        inst✝¹ : Fintype Fq
        inst✝ : Field Fq
        ε : Real
        hε : LT.lt 0 ε
        b : Polynomial Fq
        hb : Ne b 0
        hbε : LT.lt 0 (HSMul.hSMul (Polynomial.cardPowDegree b) ε)
        n : Nat
        ih : ∀ (A : Fin n → Polynomial Fq), Exists fun t => ∀ (i₀ i₁ : Fin n), Iff (Eq …
        A : Fin (HAdd.hAdd n 1) → Polynomial Fq
        anti_archim' : ∀ {i j k : Fin (HAdd.hAdd n 1)} {ε : Real}, LT.lt (↑(Polynomial …
        t' : Fin n → Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (R …
        ht' : ∀ (i₀ i₁ : Fin n), Iff (Eq (t' i₀) (t' i₁)) (LT.lt (↑(Polynomial.cardPow …
        j : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε …
        hj : ∀ (i : Fin n), Iff (Eq (t' i) j) (LT.lt (↑(Polynomial.cardPowDegree (HSub …
        i₀✝ i₁✝ : Fin (HAdd.hAdd n 1)
        i₀ i₁ : Fin n
        ⊢ Iff (Eq (t' i₀) (t' i₁)) (LT.lt (↑(Polynomial.cardPowDegree (HSub.hSub (HMod …
      -/
      exact ht' i₀ i₁
      /-
        🎉 no goals
      -/
  -- `exists_approx_polynomial` guarantees that we can insert `A 0` into some partition `j`,
  -- but not that `j` is uniquely defined (which is needed to keep the induction going).
  obtain ⟨j, hj⟩ : ∃ j, ∀ i : Fin n,
      t' i = j → (cardPowDegree (A 0 % b - A i.succ % b) : ℝ) < cardPowDegree b • ε := by
    by_contra! hg
    obtain ⟨j₀, j₁, j_ne, approx⟩ := exists_approx_polynomial hb hε
      (Fin.cons (A 0) fun j => A (Fin.succ (Classical.choose (hg j))))
    revert j_ne approx
    refine Fin.cases ?_ (fun j₀ => ?_) j₀ <;>
      refine Fin.cases (fun j_ne approx => ?_) (fun j₁ j_ne approx => ?_) j₁
    · exact absurd rfl j_ne
    · rw [Fin.cons_succ, Fin.cons_zero, ← not_le, AbsoluteValue.map_sub] at approx
      have := (Classical.choose_spec (hg j₁)).2
      contradiction
    · rw [Fin.cons_succ, Fin.cons_zero, ← not_le] at approx
      have := (Classical.choose_spec (hg j₀)).2
      contradiction
    · rw [Fin.cons_succ, Fin.cons_succ] at approx
      rw [Ne, Fin.succ_inj] at j_ne
      have : j₀ = j₁ := (Classical.choose_spec (hg j₀)).1.symm.trans
        (((ht' (Classical.choose (hg j₀)) (Classical.choose (hg j₁))).mpr approx).trans
          (Classical.choose_spec (hg j₁)).1)
      contradiction
  -- However, if one of those partitions `j` is inhabited by some `i`, then this `j` works.
  by_cases exists_nonempty_j : ∃ j, (∃ i, t' i = j) ∧
      ∀ i, t' i = j → (cardPowDegree (A 0 % b - A i.succ % b) : ℝ) < cardPowDegree b • ε
    /-
      case pos
      Fq : Type u_1
      inst✝¹ : Fintype Fq
      inst✝ : Field Fq
      ε : Real
      hε : LT.lt 0 ε
      b : Polynomial Fq
      hb : Ne b 0
      hbε : LT.lt 0 (HSMul.hSMul (Polynomial.cardPowDegree b) ε)
      n : Nat
      ih : ∀ (A : Fin n → Polynomial Fq), Exists fun t => ∀ (i₀ i₁ : Fin n), Iff (Eq …
      A : Fin (HAdd.hAdd n 1) → Polynomial Fq
      anti_archim' : ∀ {i j k : Fin (HAdd.hAdd n 1)} {ε : Real}, LT.lt (↑(Polynomial …
      t' : Fin n → Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (R …
      ht' : ∀ (i₀ i₁ : Fin n), Iff (Eq (t' i₀) (t' i₁)) (LT.lt (↑(Polynomial.cardPow …
      j : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε …
      hj : ∀ (i : Fin n), Eq (t' i) j → LT.lt (↑(Polynomial.cardPowDegree (HSub.hSub …
      exists_nonempty_j : Exists fun j => And (Exists fun i => Eq (t' i) j) (∀ (i :  …
      ⊢ Exists fun j => ∀ (i : Fin n), Iff (Eq (t' i) j) (LT.lt (↑(Polynomial.cardPo …
    -/
  · obtain ⟨j, ⟨i, hi⟩, hj⟩ := exists_nonempty_j
    /-
      case pos.intro.intro.intro
      Fq : Type u_1
      inst✝¹ : Fintype Fq
      inst✝ : Field Fq
      ε : Real
      hε : LT.lt 0 ε
      b : Polynomial Fq
      hb : Ne b 0
      hbε : LT.lt 0 (HSMul.hSMul (Polynomial.cardPowDegree b) ε)
      n : Nat
      ih : ∀ (A : Fin n → Polynomial Fq), Exists fun t => ∀ (i₀ i₁ : Fin n), Iff (Eq …
      A : Fin (HAdd.hAdd n 1) → Polynomial Fq
      anti_archim' : ∀ {i j k : Fin (HAdd.hAdd n 1)} {ε : Real}, LT.lt (↑(Polynomial …
      t' : Fin n → Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (R …
      ht' : ∀ (i₀ i₁ : Fin n), Iff (Eq (t' i₀) (t' i₁)) (LT.lt (↑(Polynomial.cardPow …
      j✝ : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log  …
      hj✝ : ∀ (i : Fin n), Eq (t' i) j✝ → LT.lt (↑(Polynomial.cardPowDegree (HSub.hS …
      j : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε …
      hj : ∀ (i : Fin n), Eq (t' i) j → LT.lt (↑(Polynomial.cardPowDegree (HSub.hSub …
      i : Fin n
      hi : Eq (t' i) j
      ⊢ Exists fun j => ∀ (i : Fin n), Iff (Eq (t' i) j) (LT.lt (↑(Polynomial.cardPo …
    -/
    refine ⟨j, fun i' => ⟨hj i', fun hi' => _root_.trans ((ht' _ _).mpr ?_) hi⟩⟩
    /-
      case pos.intro.intro.intro
      Fq : Type u_1
      inst✝¹ : Fintype Fq
      inst✝ : Field Fq
      ε : Real
      hε : LT.lt 0 ε
      b : Polynomial Fq
      hb : Ne b 0
      hbε : LT.lt 0 (HSMul.hSMul (Polynomial.cardPowDegree b) ε)
      n : Nat
      ih : ∀ (A : Fin n → Polynomial Fq), Exists fun t => ∀ (i₀ i₁ : Fin n), Iff (Eq …
      A : Fin (HAdd.hAdd n 1) → Polynomial Fq
      anti_archim' : ∀ {i j k : Fin (HAdd.hAdd n 1)} {ε : Real}, LT.lt (↑(Polynomial …
      t' : Fin n → Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (R …
      ht' : ∀ (i₀ i₁ : Fin n), Iff (Eq (t' i₀) (t' i₁)) (LT.lt (↑(Polynomial.cardPow …
      j✝ : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log  …
      hj✝ : ∀ (i : Fin n), Eq (t' i) j✝ → LT.lt (↑(Polynomial.cardPowDegree (HSub.hS …
      j : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε …
      hj : ∀ (i : Fin n), Eq (t' i) j → LT.lt (↑(Polynomial.cardPowDegree (HSub.hSub …
      i : Fin n
      hi : Eq (t' i) j
      i' : Fin n
      hi' : LT.lt (↑(Polynomial.cardPowDegree (HSub.hSub (HMod.hMod (A 0) b) (HMod.h …
      ⊢ LT.lt (↑(Polynomial.cardPowDegree (HSub.hSub (HMod.hMod (Fin.tail A i) b) (H …
    -/
    apply anti_archim' _ hi'
    /-
      Fq : Type u_1
      inst✝¹ : Fintype Fq
      inst✝ : Field Fq
      ε : Real
      hε : LT.lt 0 ε
      b : Polynomial Fq
      hb : Ne b 0
      hbε : LT.lt 0 (HSMul.hSMul (Polynomial.cardPowDegree b) ε)
      n : Nat
      ih : ∀ (A : Fin n → Polynomial Fq), Exists fun t => ∀ (i₀ i₁ : Fin n), Iff (Eq …
      A : Fin (HAdd.hAdd n 1) → Polynomial Fq
      anti_archim' : ∀ {i j k : Fin (HAdd.hAdd n 1)} {ε : Real}, LT.lt (↑(Polynomial …
      t' : Fin n → Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (R …
      ht' : ∀ (i₀ i₁ : Fin n), Iff (Eq (t' i₀) (t' i₁)) (LT.lt (↑(Polynomial.cardPow …
      j✝ : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log  …
      hj✝ : ∀ (i : Fin n), Eq (t' i) j✝ → LT.lt (↑(Polynomial.cardPowDegree (HSub.hS …
      j : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε …
      hj : ∀ (i : Fin n), Eq (t' i) j → LT.lt (↑(Polynomial.cardPowDegree (HSub.hSub …
      i : Fin n
      hi : Eq (t' i) j
      i' : Fin n
      hi' : LT.lt (↑(Polynomial.cardPowDegree (HSub.hSub (HMod.hMod (A 0) b) (HMod.h …
      ⊢ LT.lt (↑(Polynomial.cardPowDegree (HSub.hSub (HMod.hMod (A i.succ) b) (HMod. …
    -/
    rw [AbsoluteValue.map_sub]
    /-
      Fq : Type u_1
      inst✝¹ : Fintype Fq
      inst✝ : Field Fq
      ε : Real
      hε : LT.lt 0 ε
      b : Polynomial Fq
      hb : Ne b 0
      hbε : LT.lt 0 (HSMul.hSMul (Polynomial.cardPowDegree b) ε)
      n : Nat
      ih : ∀ (A : Fin n → Polynomial Fq), Exists fun t => ∀ (i₀ i₁ : Fin n), Iff (Eq …
      A : Fin (HAdd.hAdd n 1) → Polynomial Fq
      anti_archim' : ∀ {i j k : Fin (HAdd.hAdd n 1)} {ε : Real}, LT.lt (↑(Polynomial …
      t' : Fin n → Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (R …
      ht' : ∀ (i₀ i₁ : Fin n), Iff (Eq (t' i₀) (t' i₁)) (LT.lt (↑(Polynomial.cardPow …
      j✝ : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log  …
      hj✝ : ∀ (i : Fin n), Eq (t' i) j✝ → LT.lt (↑(Polynomial.cardPowDegree (HSub.hS …
      j : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε …
      hj : ∀ (i : Fin n), Eq (t' i) j → LT.lt (↑(Polynomial.cardPowDegree (HSub.hSub …
      i : Fin n
      hi : Eq (t' i) j
      i' : Fin n
      hi' : LT.lt (↑(Polynomial.cardPowDegree (HSub.hSub (HMod.hMod (A 0) b) (HMod.h …
      ⊢ LT.lt (↑(Polynomial.cardPowDegree (HSub.hSub (HMod.hMod (A 0) b) (HMod.hMod  …
    -/
    exact hj _ hi
    /-
      🎉 no goals
    -/
  -- And otherwise, we can just take any `j`, since those are empty.
  /-
    case neg
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Field Fq
    ε : Real
    hε : LT.lt 0 ε
    b : Polynomial Fq
    hb : Ne b 0
    hbε : LT.lt 0 (HSMul.hSMul (Polynomial.cardPowDegree b) ε)
    n : Nat
    ih : ∀ (A : Fin n → Polynomial Fq), Exists fun t => ∀ (i₀ i₁ : Fin n), Iff (Eq …
    A : Fin (HAdd.hAdd n 1) → Polynomial Fq
    anti_archim' : ∀ {i j k : Fin (HAdd.hAdd n 1)} {ε : Real}, LT.lt (↑(Polynomial …
    t' : Fin n → Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (R …
    ht' : ∀ (i₀ i₁ : Fin n), Iff (Eq (t' i₀) (t' i₁)) (LT.lt (↑(Polynomial.cardPow …
    j : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε …
    hj : ∀ (i : Fin n), Eq (t' i) j → LT.lt (↑(Polynomial.cardPowDegree (HSub.hSub …
    exists_nonempty_j : Not (Exists fun j => And (Exists fun i => Eq (t' i) j) (∀  …
    ⊢ Exists fun j => ∀ (i : Fin n), Iff (Eq (t' i) j) (LT.lt (↑(Polynomial.cardPo …
  -/
  refine ⟨j, fun i => ⟨hj i, fun hi => ?_⟩⟩
  /-
    case neg
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Field Fq
    ε : Real
    hε : LT.lt 0 ε
    b : Polynomial Fq
    hb : Ne b 0
    hbε : LT.lt 0 (HSMul.hSMul (Polynomial.cardPowDegree b) ε)
    n : Nat
    ih : ∀ (A : Fin n → Polynomial Fq), Exists fun t => ∀ (i₀ i₁ : Fin n), Iff (Eq …
    A : Fin (HAdd.hAdd n 1) → Polynomial Fq
    anti_archim' : ∀ {i j k : Fin (HAdd.hAdd n 1)} {ε : Real}, LT.lt (↑(Polynomial …
    t' : Fin n → Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (R …
    ht' : ∀ (i₀ i₁ : Fin n), Iff (Eq (t' i₀) (t' i₁)) (LT.lt (↑(Polynomial.cardPow …
    j : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε …
    hj : ∀ (i : Fin n), Eq (t' i) j → LT.lt (↑(Polynomial.cardPowDegree (HSub.hSub …
    exists_nonempty_j : Not (Exists fun j => And (Exists fun i => Eq (t' i) j) (∀  …
    i : Fin n
    hi : LT.lt (↑(Polynomial.cardPowDegree (HSub.hSub (HMod.hMod (A 0) b) (HMod.hM …
    ⊢ Eq (t' i) j
  -/
  have := exists_nonempty_j ⟨t' i, ⟨i, rfl⟩, fun i' hi' => anti_archim' hi ((ht' _ _).mp hi')⟩
  /-
    case neg
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Field Fq
    ε : Real
    hε : LT.lt 0 ε
    b : Polynomial Fq
    hb : Ne b 0
    hbε : LT.lt 0 (HSMul.hSMul (Polynomial.cardPowDegree b) ε)
    n : Nat
    ih : ∀ (A : Fin n → Polynomial Fq), Exists fun t => ∀ (i₀ i₁ : Fin n), Iff (Eq …
    A : Fin (HAdd.hAdd n 1) → Polynomial Fq
    anti_archim' : ∀ {i j k : Fin (HAdd.hAdd n 1)} {ε : Real}, LT.lt (↑(Polynomial …
    t' : Fin n → Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (R …
    ht' : ∀ (i₀ i₁ : Fin n), Iff (Eq (t' i₀) (t' i₁)) (LT.lt (↑(Polynomial.cardPow …
    j : Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Real.log ε …
    hj : ∀ (i : Fin n), Eq (t' i) j → LT.lt (↑(Polynomial.cardPowDegree (HSub.hSub …
    exists_nonempty_j : Not (Exists fun j => And (Exists fun i => Eq (t' i) j) (∀  …
    i : Fin n
    hi : LT.lt (↑(Polynomial.cardPowDegree (HSub.hSub (HMod.hMod (A 0) b) (HMod.hM …
    this : False
    ⊢ Eq (t' i) j
  -/
  contradiction
  /-
    🎉 no goals
  -/


/-- For all `ε > 0`, we can partition the remainders of any family of polynomials `A`
into classes, where all remainders in a class are close together. -/
theorem exists_partition_polynomial (n : ℕ) {ε : ℝ} (hε : 0 < ε) {b : Fq[X]} (hb : b ≠ 0)
    (A : Fin n → Fq[X]) : ∃ t : Fin n → Fin (Fintype.card Fq ^ ⌈-log ε / log (Fintype.card Fq)⌉₊),
      ∀ i₀ i₁ : Fin n, t i₀ = t i₁ →
        (cardPowDegree (A i₁ % b - A i₀ % b) : ℝ) < cardPowDegree b • ε := by
  /-
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Field Fq
    n : Nat
    ε : Real
    hε : LT.lt 0 ε
    b : Polynomial Fq
    hb : Ne b 0
    A : Fin n → Polynomial Fq
    ⊢ Exists fun t => ∀ (i₀ i₁ : Fin n), Eq (t i₀) (t i₁) → LT.lt (↑(Polynomial.ca …
  -/
  obtain ⟨t, ht⟩ := exists_partition_polynomial_aux n hε hb A
  /-
    case intro
    Fq : Type u_1
    inst✝¹ : Fintype Fq
    inst✝ : Field Fq
    n : Nat
    ε : Real
    hε : LT.lt 0 ε
    b : Polynomial Fq
    hb : Ne b 0
    A : Fin n → Polynomial Fq
    t : Fin n → Fin (HPow.hPow (Fintype.card Fq) (Nat.ceil (HDiv.hDiv (Neg.neg (Re …
    ht : ∀ (i₀ i₁ : Fin n), Iff (Eq (t i₀) (t i₁)) (LT.lt (↑(Polynomial.cardPowDeg …
    ⊢ Exists fun t => ∀ (i₀ i₁ : Fin n), Eq (t i₀) (t i₁) → LT.lt (↑(Polynomial.ca …
  -/
  exact ⟨t, fun i₀ i₁ hi => (ht i₀ i₁).mp hi⟩
  /-
    🎉 no goals
  -/


/-- `fun p => Fintype.card Fq ^ degree p` is an admissible absolute value.
We set `q ^ degree 0 = 0`. -/
noncomputable def cardPowDegreeIsAdmissible :
    IsAdmissible (cardPowDegree : AbsoluteValue Fq[X] ℤ) :=
  { @cardPowDegree_isEuclidean Fq _
      _ with
    card := fun ε => Fintype.card Fq ^ ⌈-log ε / log (Fintype.card Fq)⌉₊
    exists_partition' := fun n _ hε _ hb => exists_partition_polynomial n hε hb }


