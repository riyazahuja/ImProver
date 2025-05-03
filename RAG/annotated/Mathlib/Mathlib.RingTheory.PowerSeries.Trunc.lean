/-- The `n`th truncation of a formal power series to a polynomial -/
def trunc (n : ℕ) (φ : R⟦X⟧) : R[X] :=
  ∑ m ∈ Ico 0 n, Polynomial.monomial m (coeff R m φ)


theorem coeff_trunc (m) (n) (φ : R⟦X⟧) :
    (trunc n φ).coeff m = if m < n then coeff R m φ else 0 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    m n : Nat
    φ : PowerSeries R
    ⊢ Eq ((PowerSeries.trunc n φ).coeff m) (ite (LT.lt m n) ((PowerSeries.coeff R  …
  -/
  simp [trunc, Polynomial.coeff_sum, Polynomial.coeff_monomial, Nat.lt_succ_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem trunc_zero (n) : trunc n (0 : R⟦X⟧) = 0 :=
  Polynomial.ext fun m => by
    /-
      R : Type u_1
      inst✝ : Semiring R
      n m : Nat
      ⊢ Eq ((PowerSeries.trunc n 0).coeff m) (Polynomial.coeff 0 m)
    -/
    rw [coeff_trunc, LinearMap.map_zero, Polynomial.coeff_zero]
    /-
      R : Type u_1
      inst✝ : Semiring R
      n m : Nat
      ⊢ Eq (ite (LT.lt m n) 0 0) 0
    -/
                  /-
                    🎉 no goals
                  -/
    split_ifs <;> rfl
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem trunc_one (n) : trunc (n + 1) (1 : R⟦X⟧) = 1 :=
  Polynomial.ext fun m => by
    /-
      R : Type u_1
      inst✝ : Semiring R
      n m : Nat
      ⊢ Eq ((PowerSeries.trunc (HAdd.hAdd n 1) 1).coeff m) (Polynomial.coeff 1 m)
    -/
    rw [coeff_trunc, coeff_one, Polynomial.coeff_one]
    /-
      R : Type u_1
      inst✝ : Semiring R
      n m : Nat
      ⊢ Eq (ite (LT.lt m (HAdd.hAdd n 1)) (ite (Eq m 0) 1 0) 0) (ite (Eq m 0) 1 0)
    -/
    split_ifs with h _ h'
      /-
        case pos
        R : Type u_1
        inst✝ : Semiring R
        n m : Nat
        h : LT.lt m (HAdd.hAdd n 1)
        _ : Eq m 0
        ⊢ Eq 1 1
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case neg
        R : Type u_1
        inst✝ : Semiring R
        n m : Nat
        h : LT.lt m (HAdd.hAdd n 1)
        _ : Not (Eq m 0)
        ⊢ Eq 0 0
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case pos
        R : Type u_1
        inst✝ : Semiring R
        n m : Nat
        h : Not (LT.lt m (HAdd.hAdd n 1))
        h' : Eq m 0
        ⊢ Eq 0 1
      -/
    · subst h'; simp at h
                /-
                  🎉 no goals
                -/
      /-
        case neg
        R : Type u_1
        inst✝ : Semiring R
        n m : Nat
        h : Not (LT.lt m (HAdd.hAdd n 1))
        h' : Not (Eq m 0)
        ⊢ Eq 0 0
      -/
    · rfl
      /-
        🎉 no goals
      -/


@[simp]
theorem trunc_C (n) (a : R) : trunc (n + 1) (C R a) = Polynomial.C a :=
  Polynomial.ext fun m => by
    /-
      R : Type u_1
      inst✝ : Semiring R
      n : Nat
      a : R
      m : Nat
      ⊢ Eq ((PowerSeries.trunc (HAdd.hAdd n 1) ((PowerSeries.C R) a)).coeff m) ((Pol …
    -/
    rw [coeff_trunc, coeff_C, Polynomial.coeff_C]
    /-
      R : Type u_1
      inst✝ : Semiring R
      n : Nat
      a : R
      m : Nat
      ⊢ Eq (ite (LT.lt m (HAdd.hAdd n 1)) (ite (Eq m 0) a 0) 0) (ite (Eq m 0) a 0)
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
    split_ifs with H <;> first |rfl|try simp_all
                         /-
                           🎉 no goals
                         -/


@[simp]
theorem trunc_add (n) (φ ψ : R⟦X⟧) : trunc n (φ + ψ) = trunc n φ + trunc n ψ :=
  Polynomial.ext fun m => by
    /-
      R : Type u_1
      inst✝ : Semiring R
      n : Nat
      φ ψ : PowerSeries R
      m : Nat
      ⊢ Eq ((PowerSeries.trunc n (HAdd.hAdd φ ψ)).coeff m) ((HAdd.hAdd (PowerSeries. …
    -/
    simp only [coeff_trunc, AddMonoidHom.map_add, Polynomial.coeff_add]
    /-
      R : Type u_1
      inst✝ : Semiring R
      n : Nat
      φ ψ : PowerSeries R
      m : Nat
      ⊢ Eq (ite (LT.lt m n) ((PowerSeries.coeff R m) (HAdd.hAdd φ ψ)) 0) (HAdd.hAdd  …
    -/
    split_ifs with H
      /-
        case pos
        R : Type u_1
        inst✝ : Semiring R
        n : Nat
        φ ψ : PowerSeries R
        m : Nat
        H : LT.lt m n
        ⊢ Eq ((PowerSeries.coeff R m) (HAdd.hAdd φ ψ)) (HAdd.hAdd ((PowerSeries.coeff  …
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case neg
        R : Type u_1
        inst✝ : Semiring R
        n : Nat
        φ ψ : PowerSeries R
        m : Nat
        H : Not (LT.lt m n)
        ⊢ Eq 0 (HAdd.hAdd 0 0)
      -/
    · rw [zero_add]
      /-
        🎉 no goals
      -/


theorem trunc_succ (f : R⟦X⟧) (n : ℕ) :
    trunc n.succ f = trunc n f + Polynomial.monomial n (coeff R n f) := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : PowerSeries R
    n : Nat
    ⊢ Eq (PowerSeries.trunc n.succ f) (HAdd.hAdd (PowerSeries.trunc n f) ((Polynom …
  -/
  rw [trunc, Ico_zero_eq_range, sum_range_succ, trunc, Ico_zero_eq_range]
  /-
    🎉 no goals
  -/


theorem natDegree_trunc_lt (f : R⟦X⟧) (n) : (trunc (n + 1) f).natDegree < n + 1 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : PowerSeries R
    n : Nat
    ⊢ LT.lt (PowerSeries.trunc (HAdd.hAdd n 1) f).natDegree (HAdd.hAdd n 1)
  -/
  rw [Nat.lt_succ_iff, natDegree_le_iff_coeff_eq_zero]
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : PowerSeries R
    n : Nat
    ⊢ ∀ (N : Nat), LT.lt n N → Eq ((PowerSeries.trunc (HAdd.hAdd n 1) f).coeff N) 0
  -/
  intros
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : PowerSeries R
    n N✝ : Nat
    a✝ : LT.lt n N✝
    ⊢ Eq ((PowerSeries.trunc (HAdd.hAdd n 1) f).coeff N✝) 0
  -/
  rw [coeff_trunc]
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : PowerSeries R
    n N✝ : Nat
    a✝ : LT.lt n N✝
    ⊢ Eq (ite (LT.lt N✝ (HAdd.hAdd n 1)) ((PowerSeries.coeff R N✝) f) 0) 0
  -/
  split_ifs with h
    /-
      case pos
      R : Type u_1
      inst✝ : Semiring R
      f : PowerSeries R
      n N✝ : Nat
      a✝ : LT.lt n N✝
      h : LT.lt N✝ (HAdd.hAdd n 1)
      ⊢ Eq ((PowerSeries.coeff R N✝) f) 0
    -/
  · rw [lt_succ, ← not_lt] at h
    /-
      case pos
      R : Type u_1
      inst✝ : Semiring R
      f : PowerSeries R
      n N✝ : Nat
      a✝ : LT.lt n N✝
      h : Not (LT.lt n N✝)
      ⊢ Eq ((PowerSeries.coeff R N✝) f) 0
    -/
    contradiction
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝ : Semiring R
      f : PowerSeries R
      n N✝ : Nat
      a✝ : LT.lt n N✝
      h : Not (LT.lt N✝ (HAdd.hAdd n 1))
      ⊢ Eq 0 0
    -/
  · rfl
    /-
      🎉 no goals
    -/


@[simp] lemma trunc_zero' {f : R⟦X⟧} : trunc 0 f = 0 := rfl


theorem degree_trunc_lt (f : R⟦X⟧) (n) : (trunc n f).degree < n := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : PowerSeries R
    n : Nat
    ⊢ LT.lt (PowerSeries.trunc n f).degree ↑n
  -/
  rw [degree_lt_iff_coeff_zero]
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : PowerSeries R
    n : Nat
    ⊢ ∀ (m : Nat), LE.le n m → Eq ((PowerSeries.trunc n f).coeff m) 0
  -/
  intros
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : PowerSeries R
    n m✝ : Nat
    a✝ : LE.le n m✝
    ⊢ Eq ((PowerSeries.trunc n f).coeff m✝) 0
  -/
  rw [coeff_trunc]
  /-
    R : Type u_1
    inst✝ : Semiring R
    f : PowerSeries R
    n m✝ : Nat
    a✝ : LE.le n m✝
    ⊢ Eq (ite (LT.lt m✝ n) ((PowerSeries.coeff R m✝) f) 0) 0
  -/
  split_ifs with h
    /-
      case pos
      R : Type u_1
      inst✝ : Semiring R
      f : PowerSeries R
      n m✝ : Nat
      a✝ : LE.le n m✝
      h : LT.lt m✝ n
      ⊢ Eq ((PowerSeries.coeff R m✝) f) 0
    -/
  · rw [← not_le] at h
    /-
      case pos
      R : Type u_1
      inst✝ : Semiring R
      f : PowerSeries R
      n m✝ : Nat
      a✝ : LE.le n m✝
      h : Not (LE.le n m✝)
      ⊢ Eq ((PowerSeries.coeff R m✝) f) 0
    -/
    contradiction
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝ : Semiring R
      f : PowerSeries R
      n m✝ : Nat
      a✝ : LE.le n m✝
      h : Not (LT.lt m✝ n)
      ⊢ Eq 0 0
    -/
  · rfl
    /-
      🎉 no goals
    -/


theorem eval₂_trunc_eq_sum_range {S : Type*} [Semiring S] (s : S) (G : R →+* S) (n) (f : R⟦X⟧) :
    (trunc n f).eval₂ G s = ∑ i ∈ range n, G (coeff R i f) * s ^ i := by
  cases n with
  | zero =>
    rw [trunc_zero', range_zero, sum_empty, eval₂_zero]
  | succ n =>
    have := natDegree_trunc_lt f n
    rw [eval₂_eq_sum_range' (hn := this)]
    apply sum_congr rfl
    intro _ h
    rw [mem_range] at h
    congr
    rw [coeff_trunc, if_pos h]


@[simp] theorem trunc_X (n) : trunc (n + 2) X = (Polynomial.X : R[X]) := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    n : Nat
    ⊢ Eq (PowerSeries.trunc (HAdd.hAdd n 2) PowerSeries.X) Polynomial.X
  -/
  ext d
  /-
    case a
    R : Type u_1
    inst✝ : Semiring R
    n d : Nat
    ⊢ Eq ((PowerSeries.trunc (HAdd.hAdd n 2) PowerSeries.X).coeff d) (Polynomial.X …
  -/
  rw [coeff_trunc, coeff_X]
  /-
    case a
    R : Type u_1
    inst✝ : Semiring R
    n d : Nat
    ⊢ Eq (ite (LT.lt d (HAdd.hAdd n 2)) (ite (Eq d 1) 1 0) 0) (Polynomial.X.coeff d)
  -/
  split_ifs with h₁ h₂
    /-
      case pos
      R : Type u_1
      inst✝ : Semiring R
      n d : Nat
      h₁ : LT.lt d (HAdd.hAdd n 2)
      h₂ : Eq d 1
      ⊢ Eq 1 (Polynomial.X.coeff d)
    -/
  · rw [h₂, coeff_X_one]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝ : Semiring R
      n d : Nat
      h₁ : LT.lt d (HAdd.hAdd n 2)
      h₂ : Not (Eq d 1)
      ⊢ Eq 0 (Polynomial.X.coeff d)
    -/
  · rw [coeff_X_of_ne_one h₂]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝ : Semiring R
      n d : Nat
      h₁ : Not (LT.lt d (HAdd.hAdd n 2))
      ⊢ Eq 0 (Polynomial.X.coeff d)
    -/
  · rw [coeff_X_of_ne_one]
    /-
      case neg
      R : Type u_1
      inst✝ : Semiring R
      n d : Nat
      h₁ : Not (LT.lt d (HAdd.hAdd n 2))
      ⊢ Ne d 1
    -/
    intro hd
    /-
      case neg
      R : Type u_1
      inst✝ : Semiring R
      n d : Nat
      h₁ : Not (LT.lt d (HAdd.hAdd n 2))
      hd : Eq d 1
      ⊢ False
    -/
    apply h₁
    /-
      case neg
      R : Type u_1
      inst✝ : Semiring R
      n d : Nat
      h₁ : Not (LT.lt d (HAdd.hAdd n 2))
      hd : Eq d 1
      ⊢ LT.lt d (HAdd.hAdd n 2)
    -/
    rw [hd]
    /-
      case neg
      R : Type u_1
      inst✝ : Semiring R
      n d : Nat
      h₁ : Not (LT.lt d (HAdd.hAdd n 2))
      hd : Eq d 1
      ⊢ LT.lt 1 (HAdd.hAdd n 2)
    -/
    exact n.one_lt_succ_succ
    /-
      🎉 no goals
    -/


lemma trunc_X_of {n : ℕ} (hn : 2 ≤ n) : trunc n X = (Polynomial.X : R[X]) := by
  cases n with
  | zero => contradiction
  | succ n =>
    cases n with
    | zero => contradiction
    | succ n => exact trunc_X n


theorem trunc_trunc_of_le {n m} (f : R⟦X⟧) (hnm : n ≤ m := by rfl) :
    trunc n ↑(trunc m f) = trunc n f := by
  /-
    R : Type u_2
    inst✝ : CommSemiring R
    n m : Nat
    f : PowerSeries R
    hnm : autoParam (LE.le n m) _auto✝
    ⊢ Eq (PowerSeries.trunc n ↑(PowerSeries.trunc m f)) (PowerSeries.trunc n f)
  -/
  ext d
  /-
    case a
    R : Type u_2
    inst✝ : CommSemiring R
    n m : Nat
    f : PowerSeries R
    hnm : autoParam (LE.le n m) _auto✝
    d : Nat
    ⊢ Eq ((PowerSeries.trunc n ↑(PowerSeries.trunc m f)).coeff d) ((PowerSeries.tr …
  -/
  rw [coeff_trunc, coeff_trunc, coeff_coe]
  /-
    case a
    R : Type u_2
    inst✝ : CommSemiring R
    n m : Nat
    f : PowerSeries R
    hnm : autoParam (LE.le n m) _auto✝
    d : Nat
    ⊢ Eq (ite (LT.lt d n) ((PowerSeries.trunc m f).coeff d) 0) (ite (LT.lt d n) (( …
  -/
  split_ifs with h
    /-
      case pos
      R : Type u_2
      inst✝ : CommSemiring R
      n m : Nat
      f : PowerSeries R
      hnm : autoParam (LE.le n m) _auto✝
      d : Nat
      h : LT.lt d n
      ⊢ Eq ((PowerSeries.trunc m f).coeff d) ((PowerSeries.coeff R d) f)
    -/
  · rw [coeff_trunc, if_pos <| lt_of_lt_of_le h hnm]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_2
      inst✝ : CommSemiring R
      n m : Nat
      f : PowerSeries R
      hnm : autoParam (LE.le n m) _auto✝
      d : Nat
      h : Not (LT.lt d n)
      ⊢ Eq 0 0
    -/
  · rfl
    /-
      🎉 no goals
    -/


@[simp] theorem trunc_trunc {n} (f : R⟦X⟧) : trunc n ↑(trunc n f) = trunc n f :=
  /-
    R : Type u_2
    inst✝ : CommSemiring R
    n : Nat
    f : PowerSeries R
    ⊢ LE.le n n
  -/
  trunc_trunc_of_le f
  /-
    🎉 no goals
  -/


@[simp] theorem trunc_trunc_mul {n} (f g : R ⟦X⟧) :
    trunc n ((trunc n f) * g : R⟦X⟧) = trunc n (f * g) := by
  /-
    R : Type u_2
    inst✝ : CommSemiring R
    n : Nat
    f g : PowerSeries R
    ⊢ Eq (PowerSeries.trunc n (HMul.hMul (↑(PowerSeries.trunc n f)) g)) (PowerSeri …
  -/
  ext m
  /-
    case a
    R : Type u_2
    inst✝ : CommSemiring R
    n : Nat
    f g : PowerSeries R
    m : Nat
    ⊢ Eq ((PowerSeries.trunc n (HMul.hMul (↑(PowerSeries.trunc n f)) g)).coeff m)  …
  -/
  rw [coeff_trunc, coeff_trunc]
  /-
    case a
    R : Type u_2
    inst✝ : CommSemiring R
    n : Nat
    f g : PowerSeries R
    m : Nat
    ⊢ Eq (ite (LT.lt m n) ((PowerSeries.coeff R m) (HMul.hMul (↑(PowerSeries.trunc …
  -/
  split_ifs with h
    /-
      case pos
      R : Type u_2
      inst✝ : CommSemiring R
      n : Nat
      f g : PowerSeries R
      m : Nat
      h : LT.lt m n
      ⊢ Eq ((PowerSeries.coeff R m) (HMul.hMul (↑(PowerSeries.trunc n f)) g)) ((Powe …
    -/
  · rw [coeff_mul, coeff_mul, sum_congr rfl]
    /-
      case pos
      R : Type u_2
      inst✝ : CommSemiring R
      n : Nat
      f g : PowerSeries R
      m : Nat
      h : LT.lt m n
      ⊢ ∀ (x : Prod Nat Nat), Membership.mem (Finset.HasAntidiagonal.antidiagonal m) …
    -/
    intro _ hab
    /-
      case pos
      R : Type u_2
      inst✝ : CommSemiring R
      n : Nat
      f g : PowerSeries R
      m : Nat
      h : LT.lt m n
      x✝ : Prod Nat Nat
      hab : Membership.mem (Finset.HasAntidiagonal.antidiagonal m) x✝
      ⊢ Eq (HMul.hMul ((PowerSeries.coeff R x✝.1) ↑(PowerSeries.trunc n f)) ((PowerS …
    -/
    have ha := lt_of_le_of_lt (antidiagonal.fst_le hab) h
    /-
      case pos
      R : Type u_2
      inst✝ : CommSemiring R
      n : Nat
      f g : PowerSeries R
      m : Nat
      h : LT.lt m n
      x✝ : Prod Nat Nat
      hab : Membership.mem (Finset.HasAntidiagonal.antidiagonal m) x✝
      ha : LT.lt x✝.1 n
      ⊢ Eq (HMul.hMul ((PowerSeries.coeff R x✝.1) ↑(PowerSeries.trunc n f)) ((PowerS …
    -/
    rw [coeff_coe, coeff_trunc, if_pos ha]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_2
      inst✝ : CommSemiring R
      n : Nat
      f g : PowerSeries R
      m : Nat
      h : Not (LT.lt m n)
      ⊢ Eq 0 0
    -/
  · rfl
    /-
      🎉 no goals
    -/


@[simp] theorem trunc_mul_trunc {n} (f g : R ⟦X⟧) :
    trunc n (f * (trunc n g) : R⟦X⟧) = trunc n (f * g) := by
  /-
    R : Type u_2
    inst✝ : CommSemiring R
    n : Nat
    f g : PowerSeries R
    ⊢ Eq (PowerSeries.trunc n (HMul.hMul f ↑(PowerSeries.trunc n g))) (PowerSeries …
  -/
  rw [mul_comm, trunc_trunc_mul, mul_comm]
  /-
    🎉 no goals
  -/


theorem trunc_trunc_mul_trunc {n} (f g : R⟦X⟧) :
    trunc n (trunc n f * trunc n g : R⟦X⟧) = trunc n (f * g) := by
  /-
    R : Type u_2
    inst✝ : CommSemiring R
    n : Nat
    f g : PowerSeries R
    ⊢ Eq (PowerSeries.trunc n (HMul.hMul ↑(PowerSeries.trunc n f) ↑(PowerSeries.tr …
  -/
  rw [trunc_trunc_mul, trunc_mul_trunc]
  /-
    🎉 no goals
  -/


@[simp] theorem trunc_trunc_pow (f : R⟦X⟧) (n a : ℕ) :
    trunc n ((trunc n f : R⟦X⟧) ^ a) = trunc n (f ^ a) := by
  induction a with
  | zero =>
    rw [pow_zero, pow_zero]
  | succ a ih =>
    rw [_root_.pow_succ', _root_.pow_succ', trunc_trunc_mul,
      ← trunc_trunc_mul_trunc, ih, trunc_trunc_mul_trunc]


theorem trunc_coe_eq_self {n} {f : R[X]} (hn : natDegree f < n) : trunc n (f : R⟦X⟧) = f := by
  /-
    R : Type u_2
    inst✝ : CommSemiring R
    n : Nat
    f : Polynomial R
    hn : LT.lt f.natDegree n
    ⊢ Eq (PowerSeries.trunc n ↑f) f
  -/
  rw [← Polynomial.coe_inj]
  /-
    R : Type u_2
    inst✝ : CommSemiring R
    n : Nat
    f : Polynomial R
    hn : LT.lt f.natDegree n
    ⊢ Eq ↑(PowerSeries.trunc n ↑f) ↑f
  -/
  ext m
  /-
    case h
    R : Type u_2
    inst✝ : CommSemiring R
    n : Nat
    f : Polynomial R
    hn : LT.lt f.natDegree n
    m : Nat
    ⊢ Eq ((PowerSeries.coeff R m) ↑(PowerSeries.trunc n ↑f)) ((PowerSeries.coeff R …
  -/
  rw [coeff_coe, coeff_trunc]
  /-
    case h
    R : Type u_2
    inst✝ : CommSemiring R
    n : Nat
    f : Polynomial R
    hn : LT.lt f.natDegree n
    m : Nat
    ⊢ Eq (ite (LT.lt m n) ((PowerSeries.coeff R m) ↑f) 0) ((PowerSeries.coeff R m) …
  -/
  split
  /-
    case h.isTrue
    R : Type u_2
    inst✝ : CommSemiring R
    n : Nat
    f : Polynomial R
    hn : LT.lt f.natDegree n
    m : Nat
    h✝ : LT.lt m n
    ⊢ Eq ((PowerSeries.coeff R m) ↑f) ((PowerSeries.coeff R m) ↑f)
  -/
  case isTrue h => rfl
  case isFalse h =>
    rw [not_lt] at h
    rw [coeff_coe]; symm
    exact coeff_eq_zero_of_natDegree_lt <| lt_of_lt_of_le hn h


/-- The function `coeff n : R⟦X⟧ → R` is continuous. I.e. `coeff n f` depends only on a sufficiently
long truncation of the power series `f`. -/
theorem coeff_coe_trunc_of_lt {n m} {f : R⟦X⟧} (h : n < m) :
    coeff R n (trunc m f) = coeff R n f := by
  /-
    R : Type u_2
    inst✝ : CommSemiring R
    n m : Nat
    f : PowerSeries R
    h : LT.lt n m
    ⊢ Eq ((PowerSeries.coeff R n) ↑(PowerSeries.trunc m f)) ((PowerSeries.coeff R  …
  -/
  rwa [coeff_coe, coeff_trunc, if_pos]
  /-
    🎉 no goals
  -/


/-- The `n`-th coefficient of `f*g` may be calculated
from the truncations of `f` and `g`. -/
theorem coeff_mul_eq_coeff_trunc_mul_trunc₂ {n a b} (f g) (ha : n < a) (hb : n < b) :
    coeff R n (f * g) = coeff R n (trunc a f * trunc b g) := by
  /-
    R : Type u_2
    inst✝ : CommSemiring R
    n a b : Nat
    f g : PowerSeries R
    ha : LT.lt n a
    hb : LT.lt n b
    ⊢ Eq ((PowerSeries.coeff R n) (HMul.hMul f g)) ((PowerSeries.coeff R n) (HMul. …
  -/
  symm
  rw [← coeff_coe_trunc_of_lt n.lt_succ_self, ← trunc_trunc_mul_trunc, trunc_trunc_of_le f ha,
    trunc_trunc_of_le g hb, trunc_trunc_mul_trunc, coeff_coe_trunc_of_lt n.lt_succ_self]


theorem coeff_mul_eq_coeff_trunc_mul_trunc {d n} (f g) (h : d < n) :
    coeff R d (f * g) = coeff R d (trunc n f * trunc n g) :=
  coeff_mul_eq_coeff_trunc_mul_trunc₂ f g h h


