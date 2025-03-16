theorem rootMultiplicity_sub_one_le_derivative_rootMultiplicity_of_ne_zero
    (p : R[X]) (t : R) (hnezero : derivative p ≠ 0) :
    p.rootMultiplicity t - 1 ≤ p.derivative.rootMultiplicity t :=
  (le_rootMultiplicity_iff hnezero).2 <|
    pow_sub_one_dvd_derivative_of_pow_dvd (p.pow_rootMultiplicity_dvd t)


theorem derivative_rootMultiplicity_of_root_of_mem_nonZeroDivisors
    {p : R[X]} {t : R} (hpt : Polynomial.IsRoot p t)
    (hnzd : (p.rootMultiplicity t : R) ∈ nonZeroDivisors R) :
    (derivative p).rootMultiplicity t = p.rootMultiplicity t - 1 := by
  /-
    R : Type u
    inst✝ : CommRing R
    p : Polynomial R
    t : R
    hpt : p.IsRoot t
    hnzd : Membership.mem (nonZeroDivisors R) ↑(Polynomial.rootMultiplicity t p)
    ⊢ Eq (Polynomial.rootMultiplicity t (Polynomial.derivative p)) (HSub.hSub (Pol …
  -/
  by_cases h : p = 0
    /-
      case pos
      R : Type u
      inst✝ : CommRing R
      p : Polynomial R
      t : R
      hpt : p.IsRoot t
      hnzd : Membership.mem (nonZeroDivisors R) ↑(Polynomial.rootMultiplicity t p)
      h : Eq p 0
      ⊢ Eq (Polynomial.rootMultiplicity t (Polynomial.derivative p)) (HSub.hSub (Pol …
    -/
  · simp only [h, map_zero, rootMultiplicity_zero]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u
    inst✝ : CommRing R
    p : Polynomial R
    t : R
    hpt : p.IsRoot t
    hnzd : Membership.mem (nonZeroDivisors R) ↑(Polynomial.rootMultiplicity t p)
    h : Not (Eq p 0)
    ⊢ Eq (Polynomial.rootMultiplicity t (Polynomial.derivative p)) (HSub.hSub (Pol …
  -/
  obtain ⟨g, hp, hndvd⟩ := p.exists_eq_pow_rootMultiplicity_mul_and_not_dvd h t
  /-
    case neg.intro.intro
    R : Type u
    inst✝ : CommRing R
    p : Polynomial R
    t : R
    hpt : p.IsRoot t
    hnzd : Membership.mem (nonZeroDivisors R) ↑(Polynomial.rootMultiplicity t p)
    h : Not (Eq p 0)
    g : Polynomial R
    hp : Eq p (HMul.hMul (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C t)) (Pol …
    hndvd : Not (Dvd.dvd (HSub.hSub Polynomial.X (Polynomial.C t)) g)
    ⊢ Eq (Polynomial.rootMultiplicity t (Polynomial.derivative p)) (HSub.hSub (Pol …
  -/
  set m := p.rootMultiplicity t
  /-
    case neg.intro.intro
    R : Type u
    inst✝ : CommRing R
    p : Polynomial R
    t : R
    hpt : p.IsRoot t
    h : Not (Eq p 0)
    g : Polynomial R
    hndvd : Not (Dvd.dvd (HSub.hSub Polynomial.X (Polynomial.C t)) g)
    m : Nat := Polynomial.rootMultiplicity t p
    hnzd : Membership.mem (nonZeroDivisors R) ↑m
    hp : Eq p (HMul.hMul (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C t)) m) g)
    ⊢ Eq (Polynomial.rootMultiplicity t (Polynomial.derivative p)) (HSub.hSub m 1)
  -/
  have hm : m - 1 + 1 = m := Nat.sub_add_cancel <| (rootMultiplicity_pos h).2 hpt
  have hndvd : ¬(X - C t) ^ m ∣ derivative p := by
    rw [hp, derivative_mul, dvd_add_left (dvd_mul_right _ _),
      derivative_X_sub_C_pow, ← hm, pow_succ, hm, mul_comm (C _), mul_assoc,
      dvd_cancel_left_mem_nonZeroDivisors (monic_X_sub_C t |>.pow _ |>.mem_nonZeroDivisors)]
    rw [dvd_iff_isRoot, IsRoot] at hndvd ⊢
    rwa [eval_mul, eval_C, mul_left_mem_nonZeroDivisors_eq_zero_iff hnzd]
  /-
    case neg.intro.intro
    R : Type u
    inst✝ : CommRing R
    p : Polynomial R
    t : R
    hpt : p.IsRoot t
    h : Not (Eq p 0)
    g : Polynomial R
    hndvd✝ : Not (Dvd.dvd (HSub.hSub Polynomial.X (Polynomial.C t)) g)
    m : Nat := Polynomial.rootMultiplicity t p
    hnzd : Membership.mem (nonZeroDivisors R) ↑m
    hp : Eq p (HMul.hMul (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C t)) m) g)
    hm : Eq (HAdd.hAdd (HSub.hSub m 1) 1) m
    hndvd : Not (Dvd.dvd (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C t)) m) ( …
    ⊢ Eq (Polynomial.rootMultiplicity t (Polynomial.derivative p)) (HSub.hSub m 1)
  -/
  have hnezero : derivative p ≠ 0 := fun h ↦ hndvd (by rw [h]; exact dvd_zero _)
  exact le_antisymm (by rwa [rootMultiplicity_le_iff hnezero, hm])
    (rootMultiplicity_sub_one_le_derivative_rootMultiplicity_of_ne_zero _ t hnezero)


theorem isRoot_iterate_derivative_of_lt_rootMultiplicity {p : R[X]} {t : R} {n : ℕ}
    (hn : n < p.rootMultiplicity t) : (derivative^[n] p).IsRoot t :=
  dvd_iff_isRoot.mp <| (dvd_pow_self _ <| Nat.sub_ne_zero_of_lt hn).trans
    (pow_sub_dvd_iterate_derivative_of_pow_dvd _ <| p.pow_rootMultiplicity_dvd t)


open Finset in
theorem eval_iterate_derivative_rootMultiplicity {p : R[X]} {t : R} :
    (derivative^[p.rootMultiplicity t] p).eval t =
      (p.rootMultiplicity t).factorial • (p /ₘ (X - C t) ^ p.rootMultiplicity t).eval t := by
  /-
    R : Type u
    inst✝ : CommRing R
    p : Polynomial R
    t : R
    ⊢ Eq (Polynomial.eval t (Nat.iterate (⇑Polynomial.derivative) (Polynomial.root …
  -/
  set m := p.rootMultiplicity t with hm
  /-
    R : Type u
    inst✝ : CommRing R
    p : Polynomial R
    t : R
    m : Nat := Polynomial.rootMultiplicity t p
    hm : Eq m (Polynomial.rootMultiplicity t p)
    ⊢ Eq (Polynomial.eval t (Nat.iterate (⇑Polynomial.derivative) m p)) (HSMul.hSM …
  -/
  conv_lhs => rw [← p.pow_mul_divByMonic_rootMultiplicity_eq t, ← hm]
  /-
    R : Type u
    inst✝ : CommRing R
    p : Polynomial R
    t : R
    m : Nat := Polynomial.rootMultiplicity t p
    hm : Eq m (Polynomial.rootMultiplicity t p)
    ⊢ Eq (Polynomial.eval t (Nat.iterate (⇑Polynomial.derivative) m (HMul.hMul (HP …
  -/
  rw [iterate_derivative_mul, eval_finset_sum, sum_eq_single_of_mem _ (mem_range.mpr m.succ_pos)]
  · rw [m.choose_zero_right, one_smul, eval_mul, m.sub_zero, iterate_derivative_X_sub_pow_self,
                                   /-
                                     R : Type u
                                     inst✝ : CommRing R
                                     p : Polynomial R
                                     t : R
                                     m : Nat := Polynomial.rootMultiplicity t p
                                     hm : Eq m (Polynomial.rootMultiplicity t p)
                                     ⊢ Eq (HMul.hMul (↑m.factorial) (Polynomial.eval t (Nat.iterate (⇑Polynomial.de …
                                   -/
      eval_natCast, nsmul_eq_mul]; rfl
                                   /-
                                     🎉 no goals
                                   -/
    /-
      R : Type u
      inst✝ : CommRing R
      p : Polynomial R
      t : R
      m : Nat := Polynomial.rootMultiplicity t p
      hm : Eq m (Polynomial.rootMultiplicity t p)
      ⊢ ∀ (b : Nat), Membership.mem (Finset.range m.succ) b → Ne b 0 → Eq (Polynomia …
    -/
  · intro b hb hb0
    rw [iterate_derivative_X_sub_pow, eval_smul, eval_mul, eval_smul, eval_pow,
      Nat.sub_sub_self (mem_range_succ_iff.mp hb), eval_sub, eval_X, eval_C, sub_self,
      zero_pow hb0, smul_zero, zero_mul, smul_zero]


theorem lt_rootMultiplicity_of_isRoot_iterate_derivative_of_mem_nonZeroDivisors
    {p : R[X]} {t : R} {n : ℕ} (h : p ≠ 0)
    (hroot : ∀ m ≤ n, (derivative^[m] p).IsRoot t)
    (hnzd : (n.factorial : R) ∈ nonZeroDivisors R) :
    n < p.rootMultiplicity t := by
  /-
    R : Type u
    inst✝ : CommRing R
    p : Polynomial R
    t : R
    n : Nat
    h : Ne p 0
    hroot : ∀ (m : Nat), LE.le m n → (Nat.iterate (⇑Polynomial.derivative) m p).Is …
    hnzd : Membership.mem (nonZeroDivisors R) ↑n.factorial
    ⊢ LT.lt n (Polynomial.rootMultiplicity t p)
  -/
  by_contra! h'
  /-
    R : Type u
    inst✝ : CommRing R
    p : Polynomial R
    t : R
    n : Nat
    h : Ne p 0
    hroot : ∀ (m : Nat), LE.le m n → (Nat.iterate (⇑Polynomial.derivative) m p).Is …
    hnzd : Membership.mem (nonZeroDivisors R) ↑n.factorial
    h' : LE.le (Polynomial.rootMultiplicity t p) n
    ⊢ False
  -/
  replace hroot := hroot _ h'
  /-
    R : Type u
    inst✝ : CommRing R
    p : Polynomial R
    t : R
    n : Nat
    h : Ne p 0
    hnzd : Membership.mem (nonZeroDivisors R) ↑n.factorial
    h' : LE.le (Polynomial.rootMultiplicity t p) n
    hroot : (Nat.iterate (⇑Polynomial.derivative) (Polynomial.rootMultiplicity t p …
    ⊢ False
  -/
  simp only [IsRoot, eval_iterate_derivative_rootMultiplicity] at hroot
  /-
    R : Type u
    inst✝ : CommRing R
    p : Polynomial R
    t : R
    n : Nat
    h : Ne p 0
    hnzd : Membership.mem (nonZeroDivisors R) ↑n.factorial
    h' : LE.le (Polynomial.rootMultiplicity t p) n
    hroot : Eq (HSMul.hSMul (Polynomial.rootMultiplicity t p).factorial (Polynomia …
    ⊢ False
  -/
  obtain ⟨q, hq⟩ := Nat.cast_dvd_cast (α := R) <| Nat.factorial_dvd_factorial h'
  /-
    case intro
    R : Type u
    inst✝ : CommRing R
    p : Polynomial R
    t : R
    n : Nat
    h : Ne p 0
    hnzd : Membership.mem (nonZeroDivisors R) ↑n.factorial
    h' : LE.le (Polynomial.rootMultiplicity t p) n
    hroot : Eq (HSMul.hSMul (Polynomial.rootMultiplicity t p).factorial (Polynomia …
    q : R
    hq : Eq (↑n.factorial) (HMul.hMul (↑(Polynomial.rootMultiplicity t p).factoria …
    ⊢ False
  -/
  rw [hq, mul_mem_nonZeroDivisors] at hnzd
  /-
    case intro
    R : Type u
    inst✝ : CommRing R
    p : Polynomial R
    t : R
    n : Nat
    h : Ne p 0
    h' : LE.le (Polynomial.rootMultiplicity t p) n
    hroot : Eq (HSMul.hSMul (Polynomial.rootMultiplicity t p).factorial (Polynomia …
    q : R
    hnzd : And (Membership.mem (nonZeroDivisors R) ↑(Polynomial.rootMultiplicity t …
    hq : Eq (↑n.factorial) (HMul.hMul (↑(Polynomial.rootMultiplicity t p).factoria …
    ⊢ False
  -/
  rw [nsmul_eq_mul, mul_left_mem_nonZeroDivisors_eq_zero_iff hnzd.1] at hroot
  /-
    case intro
    R : Type u
    inst✝ : CommRing R
    p : Polynomial R
    t : R
    n : Nat
    h : Ne p 0
    h' : LE.le (Polynomial.rootMultiplicity t p) n
    hroot : Eq (Polynomial.eval t (p.divByMonic (HPow.hPow (HSub.hSub Polynomial.X …
    q : R
    hnzd : And (Membership.mem (nonZeroDivisors R) ↑(Polynomial.rootMultiplicity t …
    hq : Eq (↑n.factorial) (HMul.hMul (↑(Polynomial.rootMultiplicity t p).factoria …
    ⊢ False
  -/
  exact eval_divByMonic_pow_rootMultiplicity_ne_zero t h hroot
  /-
    🎉 no goals
  -/


theorem lt_rootMultiplicity_of_isRoot_iterate_derivative_of_mem_nonZeroDivisors'
    {p : R[X]} {t : R} {n : ℕ} (h : p ≠ 0)
    (hroot : ∀ m ≤ n, (derivative^[m] p).IsRoot t)
    (hnzd : ∀ m ≤ n, m ≠ 0 → (m : R) ∈ nonZeroDivisors R) :
    n < p.rootMultiplicity t := by
  /-
    R : Type u
    inst✝ : CommRing R
    p : Polynomial R
    t : R
    n : Nat
    h : Ne p 0
    hroot : ∀ (m : Nat), LE.le m n → (Nat.iterate (⇑Polynomial.derivative) m p).Is …
    hnzd : ∀ (m : Nat), LE.le m n → Ne m 0 → Membership.mem (nonZeroDivisors R) ↑m
    ⊢ LT.lt n (Polynomial.rootMultiplicity t p)
  -/
  apply lt_rootMultiplicity_of_isRoot_iterate_derivative_of_mem_nonZeroDivisors h hroot
  /-
    R : Type u
    inst✝ : CommRing R
    p : Polynomial R
    t : R
    n : Nat
    h : Ne p 0
    hroot : ∀ (m : Nat), LE.le m n → (Nat.iterate (⇑Polynomial.derivative) m p).Is …
    hnzd : ∀ (m : Nat), LE.le m n → Ne m 0 → Membership.mem (nonZeroDivisors R) ↑m
    ⊢ Membership.mem (nonZeroDivisors R) ↑n.factorial
  -/
  clear hroot
  induction n with
  | zero =>
    simp only [Nat.factorial_zero, Nat.cast_one]
    exact Submonoid.one_mem _
  | succ n ih =>
    rw [Nat.factorial_succ, Nat.cast_mul, mul_mem_nonZeroDivisors]
    exact ⟨hnzd _ le_rfl n.succ_ne_zero, ih fun m h ↦ hnzd m (h.trans n.le_succ)⟩


theorem lt_rootMultiplicity_iff_isRoot_iterate_derivative_of_mem_nonZeroDivisors
    {p : R[X]} {t : R} {n : ℕ} (h : p ≠ 0)
    (hnzd : (n.factorial : R) ∈ nonZeroDivisors R) :
    n < p.rootMultiplicity t ↔ ∀ m ≤ n, (derivative^[m] p).IsRoot t :=
  ⟨fun hn _ hm ↦ isRoot_iterate_derivative_of_lt_rootMultiplicity <| hm.trans_lt hn,
    fun hr ↦ lt_rootMultiplicity_of_isRoot_iterate_derivative_of_mem_nonZeroDivisors h hr hnzd⟩


theorem lt_rootMultiplicity_iff_isRoot_iterate_derivative_of_mem_nonZeroDivisors'
    {p : R[X]} {t : R} {n : ℕ} (h : p ≠ 0)
    (hnzd : ∀ m ≤ n, m ≠ 0 → (m : R) ∈ nonZeroDivisors R) :
    n < p.rootMultiplicity t ↔ ∀ m ≤ n, (derivative^[m] p).IsRoot t :=
  ⟨fun hn _ hm ↦ isRoot_iterate_derivative_of_lt_rootMultiplicity <| Nat.lt_of_le_of_lt hm hn,
    fun hr ↦ lt_rootMultiplicity_of_isRoot_iterate_derivative_of_mem_nonZeroDivisors' h hr hnzd⟩


theorem one_lt_rootMultiplicity_iff_isRoot_iterate_derivative
    {p : R[X]} {t : R} (h : p ≠ 0) :
    1 < p.rootMultiplicity t ↔ ∀ m ≤ 1, (derivative^[m] p).IsRoot t :=
  lt_rootMultiplicity_iff_isRoot_iterate_derivative_of_mem_nonZeroDivisors h
        /-
          R : Type u
          inst✝ : CommRing R
          p : Polynomial R
          t : R
          h : Ne p 0
          ⊢ Membership.mem (nonZeroDivisors R) ↑(Nat.factorial 1)
        -/
    (by rw [Nat.factorial_one, Nat.cast_one]; exact Submonoid.one_mem _)
                                              /-
                                                🎉 no goals
                                              -/


theorem one_lt_rootMultiplicity_iff_isRoot
    {p : R[X]} {t : R} (h : p ≠ 0) :
    1 < p.rootMultiplicity t ↔ p.IsRoot t ∧ (derivative p).IsRoot t := by
  /-
    R : Type u
    inst✝ : CommRing R
    p : Polynomial R
    t : R
    h : Ne p 0
    ⊢ Iff (LT.lt 1 (Polynomial.rootMultiplicity t p)) (And (p.IsRoot t) ((Polynomi …
  -/
  rw [one_lt_rootMultiplicity_iff_isRoot_iterate_derivative h]
  /-
    R : Type u
    inst✝ : CommRing R
    p : Polynomial R
    t : R
    h : Ne p 0
    ⊢ Iff (∀ (m : Nat), LE.le m 1 → (Nat.iterate (⇑Polynomial.derivative) m p).IsR …
  -/
  refine ⟨fun h ↦ ⟨h 0 (by norm_num), h 1 (by norm_num)⟩, fun ⟨h0, h1⟩ m hm ↦ ?_⟩
  /-
    R : Type u
    inst✝ : CommRing R
    p : Polynomial R
    t : R
    h : Ne p 0
    x✝ : And (p.IsRoot t) ((Polynomial.derivative p).IsRoot t)
    m : Nat
    hm : LE.le m 1
    h0 : p.IsRoot t
    h1 : (Polynomial.derivative p).IsRoot t
    ⊢ (Nat.iterate (⇑Polynomial.derivative) m p).IsRoot t
  -/
  obtain (_|_|m) := m
  /-
    case zero
    R : Type u
    inst✝ : CommRing R
    p : Polynomial R
    t : R
    h : Ne p 0
    x✝ : And (p.IsRoot t) ((Polynomial.derivative p).IsRoot t)
    h0 : p.IsRoot t
    h1 : (Polynomial.derivative p).IsRoot t
    hm : LE.le 0 1
    ⊢ (Nat.iterate (⇑Polynomial.derivative) 0 p).IsRoot t
  -/
  exacts [h0, h1, by omega]
  /-
    🎉 no goals
  -/


theorem one_lt_rootMultiplicity_iff_isRoot_gcd
    [GCDMonoid R[X]] {p : R[X]} {t : R} (h : p ≠ 0) :
    1 < p.rootMultiplicity t ↔ (gcd p (derivative p)).IsRoot t := by
  /-
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : GCDMonoid (Polynomial R)
    p : Polynomial R
    t : R
    h : Ne p 0
    ⊢ Iff (LT.lt 1 (Polynomial.rootMultiplicity t p)) ((GCDMonoid.gcd p (Polynomia …
  -/
  simp_rw [one_lt_rootMultiplicity_iff_isRoot h, ← dvd_iff_isRoot, dvd_gcd_iff]
  /-
    🎉 no goals
  -/


theorem derivative_rootMultiplicity_of_root [CharZero R] {p : R[X]} {t : R} (hpt : p.IsRoot t) :
    p.derivative.rootMultiplicity t = p.rootMultiplicity t - 1 := by
  /-
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : CharZero R
    p : Polynomial R
    t : R
    hpt : p.IsRoot t
    ⊢ Eq (Polynomial.rootMultiplicity t (Polynomial.derivative p)) (HSub.hSub (Pol …
  -/
  by_cases h : p = 0
    /-
      case pos
      R : Type u
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : CharZero R
      p : Polynomial R
      t : R
      hpt : p.IsRoot t
      h : Eq p 0
      ⊢ Eq (Polynomial.rootMultiplicity t (Polynomial.derivative p)) (HSub.hSub (Pol …
    -/
  · rw [h, map_zero, rootMultiplicity_zero]
    /-
      🎉 no goals
    -/
  exact derivative_rootMultiplicity_of_root_of_mem_nonZeroDivisors hpt <|
    mem_nonZeroDivisors_of_ne_zero <| Nat.cast_ne_zero.2 ((rootMultiplicity_pos h).2 hpt).ne'


theorem rootMultiplicity_sub_one_le_derivative_rootMultiplicity [CharZero R] (p : R[X]) (t : R) :
    p.rootMultiplicity t - 1 ≤ p.derivative.rootMultiplicity t := by
  /-
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : CharZero R
    p : Polynomial R
    t : R
    ⊢ LE.le (HSub.hSub (Polynomial.rootMultiplicity t p) 1) (Polynomial.rootMultip …
  -/
  by_cases h : p.IsRoot t
    /-
      case pos
      R : Type u
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : CharZero R
      p : Polynomial R
      t : R
      h : p.IsRoot t
      ⊢ LE.le (HSub.hSub (Polynomial.rootMultiplicity t p) 1) (Polynomial.rootMultip …
    -/
  · exact (derivative_rootMultiplicity_of_root h).symm.le
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : CharZero R
      p : Polynomial R
      t : R
      h : Not (p.IsRoot t)
      ⊢ LE.le (HSub.hSub (Polynomial.rootMultiplicity t p) 1) (Polynomial.rootMultip …
    -/
  · rw [rootMultiplicity_eq_zero h, zero_tsub]
    /-
      case neg
      R : Type u
      inst✝² : CommRing R
      inst✝¹ : IsDomain R
      inst✝ : CharZero R
      p : Polynomial R
      t : R
      h : Not (p.IsRoot t)
      ⊢ LE.le 0 (Polynomial.rootMultiplicity t (Polynomial.derivative p))
    -/
    exact zero_le _
    /-
      🎉 no goals
    -/


theorem lt_rootMultiplicity_of_isRoot_iterate_derivative
    [CharZero R] {p : R[X]} {t : R} {n : ℕ} (h : p ≠ 0)
    (hroot : ∀ m ≤ n, (derivative^[m] p).IsRoot t) :
    n < p.rootMultiplicity t :=
  lt_rootMultiplicity_of_isRoot_iterate_derivative_of_mem_nonZeroDivisors h hroot <|
    mem_nonZeroDivisors_of_ne_zero <| Nat.cast_ne_zero.2 <| Nat.factorial_ne_zero n


theorem lt_rootMultiplicity_iff_isRoot_iterate_derivative
    [CharZero R] {p : R[X]} {t : R} {n : ℕ} (h : p ≠ 0) :
    n < p.rootMultiplicity t ↔ ∀ m ≤ n, (derivative^[m] p).IsRoot t :=
  ⟨fun hn _ hm ↦ isRoot_iterate_derivative_of_lt_rootMultiplicity <| Nat.lt_of_le_of_lt hm hn,
    fun hr ↦ lt_rootMultiplicity_of_isRoot_iterate_derivative h hr⟩


/-- A sufficient condition for the set of roots of a nonzero polynomial `f` to be a subset of the
set of roots of `g` is that `f` divides `f.derivative * g`. Over an algebraically closed field of
characteristic zero, this is also a necessary condition.
See `isRoot_of_isRoot_iff_dvd_derivative_mul` -/
theorem isRoot_of_isRoot_of_dvd_derivative_mul [CharZero R] {f g : R[X]} (hf0 : f ≠ 0)
    (hfd : f ∣ f.derivative * g) {a : R} (haf : f.IsRoot a) : g.IsRoot a := by
  /-
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : CharZero R
    f g : Polynomial R
    hf0 : Ne f 0
    hfd : Dvd.dvd f (HMul.hMul (Polynomial.derivative f) g)
    a : R
    haf : f.IsRoot a
    ⊢ g.IsRoot a
  -/
  rcases hfd with ⟨r, hr⟩
  have hdf0 : derivative f ≠ 0 := by
    contrapose! haf
    rw [eq_C_of_derivative_eq_zero haf] at hf0 ⊢
    exact not_isRoot_C _ _ <| C_ne_zero.mp hf0
  /-
    case intro
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : CharZero R
    f g : Polynomial R
    hf0 : Ne f 0
    a : R
    haf : f.IsRoot a
    r : Polynomial R
    hr : Eq (HMul.hMul (Polynomial.derivative f) g) (HMul.hMul f r)
    hdf0 : Ne (Polynomial.derivative f) 0
    ⊢ g.IsRoot a
  -/
  by_contra hg
  /-
    case intro
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : CharZero R
    f g : Polynomial R
    hf0 : Ne f 0
    a : R
    haf : f.IsRoot a
    r : Polynomial R
    hr : Eq (HMul.hMul (Polynomial.derivative f) g) (HMul.hMul f r)
    hdf0 : Ne (Polynomial.derivative f) 0
    hg : Not (g.IsRoot a)
    ⊢ False
  -/
  have hdfg0 : f.derivative * g ≠ 0 := mul_ne_zero hdf0 (by rintro rfl; simp at hg)
  /-
    case intro
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : CharZero R
    f g : Polynomial R
    hf0 : Ne f 0
    a : R
    haf : f.IsRoot a
    r : Polynomial R
    hr : Eq (HMul.hMul (Polynomial.derivative f) g) (HMul.hMul f r)
    hdf0 : Ne (Polynomial.derivative f) 0
    hg : Not (g.IsRoot a)
    hdfg0 : Ne (HMul.hMul (Polynomial.derivative f) g) 0
    ⊢ False
  -/
  have hr' := congr_arg (rootMultiplicity a) hr
  rw [rootMultiplicity_mul hdfg0, derivative_rootMultiplicity_of_root haf,
    rootMultiplicity_eq_zero hg, add_zero, rootMultiplicity_mul (hr ▸ hdfg0), add_comm,
    Nat.sub_eq_iff_eq_add (Nat.succ_le_iff.2 ((rootMultiplicity_pos hf0).2 haf))] at hr'
  /-
    case intro
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : CharZero R
    f g : Polynomial R
    hf0 : Ne f 0
    a : R
    haf : f.IsRoot a
    r : Polynomial R
    hr : Eq (HMul.hMul (Polynomial.derivative f) g) (HMul.hMul f r)
    hdf0 : Ne (Polynomial.derivative f) 0
    hg : Not (g.IsRoot a)
    hdfg0 : Ne (HMul.hMul (Polynomial.derivative f) g) 0
    hr' : Eq (Polynomial.rootMultiplicity a f) (HAdd.hAdd (HAdd.hAdd (Polynomial.r …
    ⊢ False
  -/
  refine lt_irrefl (rootMultiplicity a f) ?_
  refine lt_of_lt_of_le (Nat.lt_succ_self _)
    (le_trans (le_add_of_nonneg_left (Nat.zero_le (rootMultiplicity a r))) ?_)
  /-
    case intro
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : CharZero R
    f g : Polynomial R
    hf0 : Ne f 0
    a : R
    haf : f.IsRoot a
    r : Polynomial R
    hr : Eq (HMul.hMul (Polynomial.derivative f) g) (HMul.hMul f r)
    hdf0 : Ne (Polynomial.derivative f) 0
    hg : Not (g.IsRoot a)
    hdfg0 : Ne (HMul.hMul (Polynomial.derivative f) g) 0
    hr' : Eq (Polynomial.rootMultiplicity a f) (HAdd.hAdd (HAdd.hAdd (Polynomial.r …
    ⊢ LE.le (HAdd.hAdd (Polynomial.rootMultiplicity a r) (Polynomial.rootMultiplic …
  -/
  conv_rhs => rw [hr']
  /-
    case intro
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : CharZero R
    f g : Polynomial R
    hf0 : Ne f 0
    a : R
    haf : f.IsRoot a
    r : Polynomial R
    hr : Eq (HMul.hMul (Polynomial.derivative f) g) (HMul.hMul f r)
    hdf0 : Ne (Polynomial.derivative f) 0
    hg : Not (g.IsRoot a)
    hdfg0 : Ne (HMul.hMul (Polynomial.derivative f) g) 0
    hr' : Eq (Polynomial.rootMultiplicity a f) (HAdd.hAdd (HAdd.hAdd (Polynomial.r …
    ⊢ LE.le (HAdd.hAdd (Polynomial.rootMultiplicity a r) (Polynomial.rootMultiplic …
  -/
  simp [add_assoc]
  /-
    🎉 no goals
  -/


instance instNormalizationMonoid : NormalizationMonoid R[X] where
  normUnit p :=
    ⟨C ↑(normUnit p.leadingCoeff), C ↑(normUnit p.leadingCoeff)⁻¹, by
      /-
        R : Type u
        S : Type v
        k : Type y
        A : Type z
        a b : R
        n : Nat
        inst✝² : CommRing R
        inst✝¹ : IsDomain R
        inst✝ : NormalizationMonoid R
        p : Polynomial R
        ⊢ Eq (HMul.hMul (Polynomial.C ↑(NormalizationMonoid.normUnit p.leadingCoeff))  …
      -/
      /-
        🎉 no goals
      -/
      rw [← RingHom.map_mul, Units.mul_inv, C_1], by rw [← RingHom.map_mul, Units.inv_mul, C_1]⟩
                                                     /-
                                                       🎉 no goals
                                                     -/
                                 /-
                                   R : Type u
                                   S : Type v
                                   k : Type y
                                   A : Type z
                                   a b : R
                                   n : Nat
                                   inst✝² : CommRing R
                                   inst✝¹ : IsDomain R
                                   inst✝ : NormalizationMonoid R
                                   ⊢ Eq ↑((fun p => { val := Polynomial.C ↑(NormalizationMonoid.normUnit p.leadin …
                                 -/
  normUnit_zero := Units.ext (by simp)
                                 /-
                                   🎉 no goals
                                 -/
  normUnit_mul hp0 hq0 :=
    Units.ext
      (by
        /-
          R : Type u
          S : Type v
          k : Type y
          A : Type z
          a b : R
          n : Nat
          inst✝² : CommRing R
          inst✝¹ : IsDomain R
          inst✝ : NormalizationMonoid R
          a✝ b✝ : Polynomial R
          hp0 : Ne a✝ 0
          hq0 : Ne b✝ 0
          ⊢ Eq ↑((fun p => { val := Polynomial.C ↑(NormalizationMonoid.normUnit p.leadin …
        -/
        dsimp
        /-
          R : Type u
          S : Type v
          k : Type y
          A : Type z
          a b : R
          n : Nat
          inst✝² : CommRing R
          inst✝¹ : IsDomain R
          inst✝ : NormalizationMonoid R
          a✝ b✝ : Polynomial R
          hp0 : Ne a✝ 0
          hq0 : Ne b✝ 0
          ⊢ Eq (Polynomial.C ↑(NormalizationMonoid.normUnit (HMul.hMul a✝ b✝).leadingCoe …
        -/
        rw [Ne, ← leadingCoeff_eq_zero] at *
        /-
          R : Type u
          S : Type v
          k : Type y
          A : Type z
          a b : R
          n : Nat
          inst✝² : CommRing R
          inst✝¹ : IsDomain R
          inst✝ : NormalizationMonoid R
          a✝ b✝ : Polynomial R
          hp0 : Not (Eq a✝.leadingCoeff 0)
          hq0 : Not (Eq b✝.leadingCoeff 0)
          ⊢ Eq (Polynomial.C ↑(NormalizationMonoid.normUnit (HMul.hMul a✝ b✝).leadingCoe …
        -/
        rw [leadingCoeff_mul, normUnit_mul hp0 hq0, Units.val_mul, C_mul])
        /-
          🎉 no goals
        -/
  normUnit_coe_units u :=
    Units.ext
      (by
        /-
          R : Type u
          S : Type v
          k : Type y
          A : Type z
          a b : R
          n : Nat
          inst✝² : CommRing R
          inst✝¹ : IsDomain R
          inst✝ : NormalizationMonoid R
          u : Units (Polynomial R)
          ⊢ Eq ↑((fun p => { val := Polynomial.C ↑(NormalizationMonoid.normUnit p.leadin …
        -/
        dsimp
        /-
          R : Type u
          S : Type v
          k : Type y
          A : Type z
          a b : R
          n : Nat
          inst✝² : CommRing R
          inst✝¹ : IsDomain R
          inst✝ : NormalizationMonoid R
          u : Units (Polynomial R)
          ⊢ Eq (Polynomial.C ↑(NormalizationMonoid.normUnit (↑u).leadingCoeff)) ↑(Inv.in …
        -/
        rw [← mul_one u⁻¹, Units.val_mul, Units.eq_inv_mul_iff_mul_eq]
        /-
          R : Type u
          S : Type v
          k : Type y
          A : Type z
          a b : R
          n : Nat
          inst✝² : CommRing R
          inst✝¹ : IsDomain R
          inst✝ : NormalizationMonoid R
          u : Units (Polynomial R)
          ⊢ Eq (HMul.hMul (↑u) (Polynomial.C ↑(NormalizationMonoid.normUnit (↑u).leading …
        -/
        rcases Polynomial.isUnit_iff.1 ⟨u, rfl⟩ with ⟨_, ⟨w, rfl⟩, h2⟩
        /-
          case intro.intro.intro
          R : Type u
          S : Type v
          k : Type y
          A : Type z
          a b : R
          n : Nat
          inst✝² : CommRing R
          inst✝¹ : IsDomain R
          inst✝ : NormalizationMonoid R
          u : Units (Polynomial R)
          w : Units R
          h2 : Eq (Polynomial.C ↑w) ↑u
          ⊢ Eq (HMul.hMul (↑u) (Polynomial.C ↑(NormalizationMonoid.normUnit (↑u).leading …
        -/
        rw [← h2, leadingCoeff_C, normUnit_coe_units, ← C_mul, Units.mul_inv, C_1]
        /-
          case intro.intro.intro
          R : Type u
          S : Type v
          k : Type y
          A : Type z
          a b : R
          n : Nat
          inst✝² : CommRing R
          inst✝¹ : IsDomain R
          inst✝ : NormalizationMonoid R
          u : Units (Polynomial R)
          w : Units R
          h2 : Eq (Polynomial.C ↑w) ↑u
          ⊢ Eq 1 ↑1
        -/
        rfl)
        /-
          🎉 no goals
        -/


@[simp]
theorem coe_normUnit {p : R[X]} : (normUnit p : R[X]) = C ↑(normUnit p.leadingCoeff) := by
  /-
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizationMonoid R
    p : Polynomial R
    ⊢ Eq (↑(NormalizationMonoid.normUnit p)) (Polynomial.C ↑(NormalizationMonoid.n …
  -/
  simp [normUnit]
  /-
    🎉 no goals
  -/


@[simp]
theorem leadingCoeff_normalize (p : R[X]) :
                                                                  /-
                                                                    R : Type u
                                                                    inst✝² : CommRing R
                                                                    inst✝¹ : IsDomain R
                                                                    inst✝ : NormalizationMonoid R
                                                                    p : Polynomial R
                                                                    ⊢ Eq (normalize p).leadingCoeff (normalize p.leadingCoeff)
                                                                  -/
    leadingCoeff (normalize p) = normalize (leadingCoeff p) := by simp [normalize_apply]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


theorem Monic.normalize_eq_self {p : R[X]} (hp : p.Monic) : normalize p = p := by
  simp only [Polynomial.coe_normUnit, normalize_apply, hp.leadingCoeff, normUnit_one,
    Units.val_one, Polynomial.C.map_one, mul_one]


@[deprecated Polynomial.Monic.normalize_eq_self (since := "2024-10-21")]
alias normalize_monic := Monic.normalize_eq_self


theorem roots_normalize {p : R[X]} : (normalize p).roots = p.roots := by
  /-
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizationMonoid R
    p : Polynomial R
    ⊢ Eq (normalize p).roots p.roots
  -/
  rw [normalize_apply, mul_comm, coe_normUnit, roots_C_mul _ (normUnit (leadingCoeff p)).ne_zero]
  /-
    🎉 no goals
  -/


theorem normUnit_X : normUnit (X : Polynomial R) = 1 := by
  /-
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizationMonoid R
    ⊢ Eq (NormalizationMonoid.normUnit Polynomial.X) 1
  -/
  have := coe_normUnit (R := R) (p := X)
  /-
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizationMonoid R
    this : Eq (↑(NormalizationMonoid.normUnit Polynomial.X)) (Polynomial.C ↑(Norma …
    ⊢ Eq (NormalizationMonoid.normUnit Polynomial.X) 1
  -/
  rwa [leadingCoeff_X, normUnit_one, Units.val_one, map_one, Units.val_eq_one] at this
  /-
    🎉 no goals
  -/


theorem X_eq_normalize : (X : Polynomial R) = normalize X := by
  /-
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : NormalizationMonoid R
    ⊢ Eq Polynomial.X (normalize Polynomial.X)
  -/
  simp only [normalize_apply, normUnit_X, Units.val_one, mul_one]
  /-
    🎉 no goals
  -/


theorem degree_pos_of_ne_zero_of_nonunit (hp0 : p ≠ 0) (hp : ¬IsUnit p) : 0 < degree p :=
  lt_of_not_ge fun h => by
    /-
      R : Type u
      inst✝ : DivisionRing R
      p : Polynomial R
      hp0 : Ne p 0
      hp : Not (IsUnit p)
      h : GE.ge 0 p.degree
      ⊢ False
    -/
    rw [eq_C_of_degree_le_zero h] at hp0 hp
    /-
      R : Type u
      inst✝ : DivisionRing R
      p : Polynomial R
      hp0 : Ne (Polynomial.C (p.coeff 0)) 0
      hp : Not (IsUnit (Polynomial.C (p.coeff 0)))
      h : GE.ge 0 p.degree
      ⊢ False
    -/
    exact hp (IsUnit.map C (IsUnit.mk0 (coeff p 0) (mt C_inj.2 (by simpa using hp0))))
    /-
      🎉 no goals
    -/


@[simp]
protected theorem map_eq_zero [Semiring S] [Nontrivial S] (f : R →+* S) : p.map f = 0 ↔ p = 0 := by
  /-
    R : Type u
    S : Type v
    inst✝² : DivisionRing R
    p : Polynomial R
    inst✝¹ : Semiring S
    inst✝ : Nontrivial S
    f : RingHom R S
    ⊢ Iff (Eq (Polynomial.map f p) 0) (Eq p 0)
  -/
  simp only [Polynomial.ext_iff]
  /-
    R : Type u
    S : Type v
    inst✝² : DivisionRing R
    p : Polynomial R
    inst✝¹ : Semiring S
    inst✝ : Nontrivial S
    f : RingHom R S
    ⊢ Iff (∀ (n : Nat), Eq ((Polynomial.map f p).coeff n) (Polynomial.coeff 0 n))  …
  -/
  congr!
  /-
    case a.h.a
    R : Type u
    S : Type v
    inst✝² : DivisionRing R
    p : Polynomial R
    inst✝¹ : Semiring S
    inst✝ : Nontrivial S
    f : RingHom R S
    a✝ : Nat
    ⊢ Iff (Eq ((Polynomial.map f p).coeff a✝) (Polynomial.coeff 0 a✝)) (Eq (p.coef …
  -/
  simp [map_eq_zero, coeff_map, coeff_zero]
  /-
    🎉 no goals
  -/


theorem map_ne_zero [Semiring S] [Nontrivial S] {f : R →+* S} (hp : p ≠ 0) : p.map f ≠ 0 :=
  mt (Polynomial.map_eq_zero f).1 hp


@[simp]
theorem degree_map [Semiring S] [Nontrivial S] (p : R[X]) (f : R →+* S) :
    degree (p.map f) = degree p :=
  p.degree_map_eq_of_injective f.injective


@[simp]
theorem natDegree_map [Semiring S] [Nontrivial S] (f : R →+* S) :
    natDegree (p.map f) = natDegree p :=
  natDegree_eq_of_degree_eq (degree_map _ f)


@[simp]
theorem leadingCoeff_map [Semiring S] [Nontrivial S] (f : R →+* S) :
    leadingCoeff (p.map f) = f (leadingCoeff p) := by
  /-
    R : Type u
    S : Type v
    inst✝² : DivisionRing R
    p : Polynomial R
    inst✝¹ : Semiring S
    inst✝ : Nontrivial S
    f : RingHom R S
    ⊢ Eq (Polynomial.map f p).leadingCoeff (f p.leadingCoeff)
  -/
  simp only [← coeff_natDegree, coeff_map f, natDegree_map]
  /-
    🎉 no goals
  -/


theorem monic_map_iff [Semiring S] [Nontrivial S] {f : R →+* S} {p : R[X]} :
    (p.map f).Monic ↔ p.Monic := by
  /-
    R : Type u
    S : Type v
    inst✝² : DivisionRing R
    inst✝¹ : Semiring S
    inst✝ : Nontrivial S
    f : RingHom R S
    p : Polynomial R
    ⊢ Iff (Polynomial.map f p).Monic p.Monic
  -/
  rw [Monic, leadingCoeff_map, ← f.map_one, Function.Injective.eq_iff f.injective, Monic]
  /-
    🎉 no goals
  -/


theorem isUnit_iff_degree_eq_zero : IsUnit p ↔ degree p = 0 :=
  ⟨degree_eq_zero_of_isUnit, fun h =>
                              /-
                                R : Type u
                                inst✝ : Field R
                                p : Polynomial R
                                h : Eq p.degree 0
                                ⊢ LE.le p.degree 0
                              -/
    have : degree p ≤ 0 := by simp [*, le_refl]
                              /-
                                🎉 no goals
                              -/
    have hc : coeff p 0 ≠ 0 := fun hc => by
      /-
        R : Type u
        inst✝ : Field R
        p : Polynomial R
        h : Eq p.degree 0
        this : LE.le p.degree 0
        hc : Eq (p.coeff 0) 0
        ⊢ False
      -/
      rw [eq_C_of_degree_le_zero this, hc] at h; simp only [map_zero] at h; contradiction
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
    isUnit_iff_dvd_one.2
      ⟨C (coeff p 0)⁻¹, by
        /-
          R : Type u
          inst✝ : Field R
          p : Polynomial R
          h : Eq p.degree 0
          this : LE.le p.degree 0
          hc : Ne (p.coeff 0) 0
          ⊢ Eq 1 (HMul.hMul p (Polynomial.C (Inv.inv (p.coeff 0))))
        -/
        conv in p => rw [eq_C_of_degree_le_zero this]
        /-
          R : Type u
          inst✝ : Field R
          p : Polynomial R
          h : Eq p.degree 0
          this : LE.le p.degree 0
          hc : Ne (p.coeff 0) 0
          ⊢ Eq 1 (HMul.hMul (Polynomial.C (p.coeff 0)) (Polynomial.C (Inv.inv (p.coeff 0 …
        -/
        rw [← C_mul, mul_inv_cancel₀ hc, C_1]⟩⟩
        /-
          🎉 no goals
        -/


/-- Division of polynomials. See `Polynomial.divByMonic` for more details. -/
def div (p q : R[X]) :=
  C (leadingCoeff q)⁻¹ * (p /ₘ (q * C (leadingCoeff q)⁻¹))


/-- Remainder of polynomial division. See `Polynomial.modByMonic` for more details. -/
def mod (p q : R[X]) :=
  p %ₘ (q * C (leadingCoeff q)⁻¹)


private theorem quotient_mul_add_remainder_eq_aux (p q : R[X]) : q * div p q + mod p q = p := by
  /-
    R : Type u
    inst✝ : Field R
    p q : Polynomial R
    ⊢ Eq (HAdd.hAdd (HMul.hMul q (p.div q)) (p.mod q)) p
  -/
  by_cases h : q = 0
    /-
      case pos
      R : Type u
      inst✝ : Field R
      p q : Polynomial R
      h : Eq q 0
      ⊢ Eq (HAdd.hAdd (HMul.hMul q (p.div q)) (p.mod q)) p
    -/
  · simp only [h, zero_mul, mod, modByMonic_zero, zero_add]
    /-
      🎉 no goals
    -/
  · conv =>
      rhs
      rw [← modByMonic_add_div p (monic_mul_leadingCoeff_inv h)]
    /-
      case neg
      R : Type u
      inst✝ : Field R
      p q : Polynomial R
      h : Not (Eq q 0)
      ⊢ Eq (HAdd.hAdd (HMul.hMul q (p.div q)) (p.mod q)) (HAdd.hAdd (p.modByMonic (H …
    -/
    rw [div, mod, add_comm, mul_assoc]
    /-
      🎉 no goals
    -/


private theorem remainder_lt_aux (p : R[X]) (hq : q ≠ 0) : degree (mod p q) < degree q := by
  /-
    R : Type u
    inst✝ : Field R
    q p : Polynomial R
    hq : Ne q 0
    ⊢ LT.lt (p.mod q).degree q.degree
  -/
  rw [← degree_mul_leadingCoeff_inv q hq]
  /-
    R : Type u
    inst✝ : Field R
    q p : Polynomial R
    hq : Ne q 0
    ⊢ LT.lt (p.mod q).degree (HMul.hMul q (Polynomial.C (Inv.inv q.leadingCoeff))) …
  -/
  exact degree_modByMonic_lt p (monic_mul_leadingCoeff_inv hq)
  /-
    🎉 no goals
  -/


instance : Div R[X] :=
  ⟨div⟩


instance : Mod R[X] :=
  ⟨mod⟩


theorem div_def : p / q = C (leadingCoeff q)⁻¹ * (p /ₘ (q * C (leadingCoeff q)⁻¹)) :=
  rfl


theorem mod_def : p % q = p %ₘ (q * C (leadingCoeff q)⁻¹) := rfl


theorem modByMonic_eq_mod (p : R[X]) (hq : Monic q) : p %ₘ q = p % q :=
  show p %ₘ q = p %ₘ (q * C (leadingCoeff q)⁻¹) by
    /-
      R : Type u
      inst✝ : Field R
      q p : Polynomial R
      hq : q.Monic
      ⊢ Eq (p.modByMonic q) (p.modByMonic (HMul.hMul q (Polynomial.C (Inv.inv q.lead …
    -/
    simp only [Monic.def.1 hq, inv_one, mul_one, C_1]
    /-
      🎉 no goals
    -/


theorem divByMonic_eq_div (p : R[X]) (hq : Monic q) : p /ₘ q = p / q :=
  show p /ₘ q = C (leadingCoeff q)⁻¹ * (p /ₘ (q * C (leadingCoeff q)⁻¹)) by
    /-
      R : Type u
      inst✝ : Field R
      q p : Polynomial R
      hq : q.Monic
      ⊢ Eq (p.divByMonic q) (HMul.hMul (Polynomial.C (Inv.inv q.leadingCoeff)) (p.di …
    -/
    simp only [Monic.def.1 hq, inv_one, C_1, one_mul, mul_one]
    /-
      🎉 no goals
    -/


theorem mod_X_sub_C_eq_C_eval (p : R[X]) (a : R) : p % (X - C a) = C (p.eval a) :=
  modByMonic_eq_mod p (monic_X_sub_C a) ▸ modByMonic_X_sub_C_eq_C_eval _ _


theorem mul_div_eq_iff_isRoot : (X - C a) * (p / (X - C a)) = p ↔ IsRoot p a :=
  divByMonic_eq_div p (monic_X_sub_C a) ▸ mul_divByMonic_eq_iff_isRoot


instance instEuclideanDomain : EuclideanDomain R[X] :=
  { Polynomial.commRing,
    Polynomial.nontrivial with
    quotient := (· / ·)
                        /-
                          R : Type u
                          S : Type v
                          k : Type y
                          A : Type z
                          a b : R
                          n : Nat
                          inst✝ : Field R
                          p q : Polynomial R
                          ⊢ ∀ (a : Polynomial R), Eq ((fun x1 x2 => HDiv.hDiv x1 x2) a 0) 0
                        -/
    quotient_zero := by simp [div_def]
                        /-
                          🎉 no goals
                        -/
    remainder := (· % ·)
    r := _
    r_wellFounded := degree_lt_wf
    quotient_mul_add_remainder_eq := quotient_mul_add_remainder_eq_aux
    remainder_lt := fun _ _ hq => remainder_lt_aux _ hq
    mul_left_not_lt := fun _ _ hq => not_lt_of_ge (degree_le_mul_left _ hq) }


theorem mod_eq_self_iff (hq0 : q ≠ 0) : p % q = p ↔ degree p < degree q :=
  ⟨fun h => h ▸ EuclideanDomain.mod_lt _ hq0, fun h => by
    classical
    have : ¬degree (q * C (leadingCoeff q)⁻¹) ≤ degree p :=
      not_le_of_gt <| by rwa [degree_mul_leadingCoeff_inv q hq0]
    rw [mod_def, modByMonic, dif_pos (monic_mul_leadingCoeff_inv hq0)]
    unfold divModByMonicAux
    dsimp
    simp only [this, false_and, if_false]⟩


theorem div_eq_zero_iff (hq0 : q ≠ 0) : p / q = 0 ↔ degree p < degree q :=
  ⟨fun h => by
    /-
      R : Type u
      inst✝ : Field R
      p q : Polynomial R
      hq0 : Ne q 0
      h : Eq (HDiv.hDiv p q) 0
      ⊢ LT.lt p.degree q.degree
    -/
    have := EuclideanDomain.div_add_mod p q
    /-
      R : Type u
      inst✝ : Field R
      p q : Polynomial R
      hq0 : Ne q 0
      h : Eq (HDiv.hDiv p q) 0
      this : Eq (HAdd.hAdd (HMul.hMul q (HDiv.hDiv p q)) (HMod.hMod p q)) p
      ⊢ LT.lt p.degree q.degree
    -/
    rwa [h, mul_zero, zero_add, mod_eq_self_iff hq0] at this,
    /-
      🎉 no goals
    -/
  fun h => by
    have hlt : degree p < degree (q * C (leadingCoeff q)⁻¹) := by
      rwa [degree_mul_leadingCoeff_inv q hq0]
    /-
      R : Type u
      inst✝ : Field R
      p q : Polynomial R
      hq0 : Ne q 0
      h : LT.lt p.degree q.degree
      hlt : LT.lt p.degree (HMul.hMul q (Polynomial.C (Inv.inv q.leadingCoeff))).deg …
      ⊢ Eq (HDiv.hDiv p q) 0
    -/
    have hm : Monic (q * C (leadingCoeff q)⁻¹) := monic_mul_leadingCoeff_inv hq0
    /-
      R : Type u
      inst✝ : Field R
      p q : Polynomial R
      hq0 : Ne q 0
      h : LT.lt p.degree q.degree
      hlt : LT.lt p.degree (HMul.hMul q (Polynomial.C (Inv.inv q.leadingCoeff))).deg …
      hm : (HMul.hMul q (Polynomial.C (Inv.inv q.leadingCoeff))).Monic
      ⊢ Eq (HDiv.hDiv p q) 0
    -/
    rw [div_def, (divByMonic_eq_zero_iff hm).2 hlt, mul_zero]⟩
    /-
      🎉 no goals
    -/


theorem degree_add_div (hq0 : q ≠ 0) (hpq : degree q ≤ degree p) :
    degree q + degree (p / q) = degree p := by
  have : degree (p % q) < degree (q * (p / q)) :=
    calc
      degree (p % q) < degree q := EuclideanDomain.mod_lt _ hq0
      _ ≤ _ := degree_le_mul_left _ (mt (div_eq_zero_iff hq0).1 (not_lt_of_ge hpq))

  conv_rhs =>
    rw [← EuclideanDomain.div_add_mod p q, degree_add_eq_left_of_degree_lt this, degree_mul]


theorem degree_div_le (p q : R[X]) : degree (p / q) ≤ degree p := by
  /-
    R : Type u
    inst✝ : Field R
    p q : Polynomial R
    ⊢ LE.le (HDiv.hDiv p q).degree p.degree
  -/
  by_cases hq : q = 0
    /-
      case pos
      R : Type u
      inst✝ : Field R
      p q : Polynomial R
      hq : Eq q 0
      ⊢ LE.le (HDiv.hDiv p q).degree p.degree
    -/
  · simp [hq]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u
      inst✝ : Field R
      p q : Polynomial R
      hq : Not (Eq q 0)
      ⊢ LE.le (HDiv.hDiv p q).degree p.degree
    -/
  · rw [div_def, mul_comm, degree_mul_leadingCoeff_inv _ hq]; exact degree_divByMonic_le _ _
                                                              /-
                                                                🎉 no goals
                                                              -/


theorem degree_div_lt (hp : p ≠ 0) (hq : 0 < degree q) : degree (p / q) < degree p := by
  /-
    R : Type u
    inst✝ : Field R
    p q : Polynomial R
    hp : Ne p 0
    hq : LT.lt 0 q.degree
    ⊢ LT.lt (HDiv.hDiv p q).degree p.degree
  -/
  have hq0 : q ≠ 0 := fun hq0 => by simp [hq0] at hq
  /-
    R : Type u
    inst✝ : Field R
    p q : Polynomial R
    hp : Ne p 0
    hq : LT.lt 0 q.degree
    hq0 : Ne q 0
    ⊢ LT.lt (HDiv.hDiv p q).degree p.degree
  -/
  rw [div_def, mul_comm, degree_mul_leadingCoeff_inv _ hq0]
  exact degree_divByMonic_lt _ (monic_mul_leadingCoeff_inv hq0) hp
    (by rw [degree_mul_leadingCoeff_inv _ hq0]; exact hq)


theorem isUnit_map [Field k] (f : R →+* k) : IsUnit (p.map f) ↔ IsUnit p := by
  /-
    R : Type u
    k : Type y
    inst✝¹ : Field R
    p : Polynomial R
    inst✝ : Field k
    f : RingHom R k
    ⊢ Iff (IsUnit (Polynomial.map f p)) (IsUnit p)
  -/
  simp_rw [isUnit_iff_degree_eq_zero, degree_map]
  /-
    🎉 no goals
  -/


theorem map_div [Field k] (f : R →+* k) : (p / q).map f = p.map f / q.map f := by
  if hq0 : q = 0 then simp [hq0]
  else
    rw [div_def, div_def, Polynomial.map_mul, map_divByMonic f (monic_mul_leadingCoeff_inv hq0),
      Polynomial.map_mul, map_C, leadingCoeff_map, map_inv₀]


theorem map_mod [Field k] (f : R →+* k) : (p % q).map f = p.map f % q.map f := by
  /-
    R : Type u
    k : Type y
    inst✝¹ : Field R
    p q : Polynomial R
    inst✝ : Field k
    f : RingHom R k
    ⊢ Eq (Polynomial.map f (HMod.hMod p q)) (HMod.hMod (Polynomial.map f p) (Polyn …
  -/
  by_cases hq0 : q = 0
    /-
      case pos
      R : Type u
      k : Type y
      inst✝¹ : Field R
      p q : Polynomial R
      inst✝ : Field k
      f : RingHom R k
      hq0 : Eq q 0
      ⊢ Eq (Polynomial.map f (HMod.hMod p q)) (HMod.hMod (Polynomial.map f p) (Polyn …
    -/
  · simp [hq0]
    /-
      🎉 no goals
    -/
  · rw [mod_def, mod_def, leadingCoeff_map f, ← map_inv₀ f, ← map_C f, ← Polynomial.map_mul f,
      map_modByMonic f (monic_mul_leadingCoeff_inv hq0)]


lemma natDegree_mod_lt [Field k] (p : k[X]) {q : k[X]} (hq : q.natDegree ≠ 0) :
    (p % q).natDegree < q.natDegree := by
  have hq' : q.leadingCoeff ≠ 0 := by
    rw [leadingCoeff_ne_zero]
    contrapose! hq
    simp [hq]
  /-
    k : Type y
    inst✝ : Field k
    p q : Polynomial k
    hq : Ne q.natDegree 0
    hq' : Ne q.leadingCoeff 0
    ⊢ LT.lt (HMod.hMod p q).natDegree q.natDegree
  -/
  rw [mod_def]
  /-
    k : Type y
    inst✝ : Field k
    p q : Polynomial k
    hq : Ne q.natDegree 0
    hq' : Ne q.leadingCoeff 0
    ⊢ LT.lt (p.modByMonic (HMul.hMul q (Polynomial.C (Inv.inv q.leadingCoeff)))).n …
  -/
  refine (natDegree_modByMonic_lt p ?_ ?_).trans_le ?_
    /-
      case refine_1
      k : Type y
      inst✝ : Field k
      p q : Polynomial k
      hq : Ne q.natDegree 0
      hq' : Ne q.leadingCoeff 0
      ⊢ (HMul.hMul q (Polynomial.C (Inv.inv q.leadingCoeff))).Monic
    -/
  · refine monic_mul_C_of_leadingCoeff_mul_eq_one ?_
    /-
      case refine_1
      k : Type y
      inst✝ : Field k
      p q : Polynomial k
      hq : Ne q.natDegree 0
      hq' : Ne q.leadingCoeff 0
      ⊢ Eq (HMul.hMul q.leadingCoeff (Inv.inv q.leadingCoeff)) 1
    -/
    rw [mul_inv_eq_one₀ hq']
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      k : Type y
      inst✝ : Field k
      p q : Polynomial k
      hq : Ne q.natDegree 0
      hq' : Ne q.leadingCoeff 0
      ⊢ Ne (HMul.hMul q (Polynomial.C (Inv.inv q.leadingCoeff))) 1
    -/
  · contrapose! hq
    /-
      case refine_2
      k : Type y
      inst✝ : Field k
      p q : Polynomial k
      hq' : Ne q.leadingCoeff 0
      hq : Eq (HMul.hMul q (Polynomial.C (Inv.inv q.leadingCoeff))) 1
      ⊢ Eq q.natDegree 0
    -/
    rw [← natDegree_mul_C_eq_of_mul_eq_one ((inv_mul_eq_one₀ hq').mpr rfl)]
    /-
      case refine_2
      k : Type y
      inst✝ : Field k
      p q : Polynomial k
      hq' : Ne q.leadingCoeff 0
      hq : Eq (HMul.hMul q (Polynomial.C (Inv.inv q.leadingCoeff))) 1
      ⊢ Eq (HMul.hMul q (Polynomial.C (Inv.inv q.leadingCoeff))).natDegree 0
    -/
    simp [hq]
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      k : Type y
      inst✝ : Field k
      p q : Polynomial k
      hq : Ne q.natDegree 0
      hq' : Ne q.leadingCoeff 0
      ⊢ LE.le (HMul.hMul q (Polynomial.C (Inv.inv q.leadingCoeff))).natDegree q.natD …
    -/
  · exact natDegree_mul_C_le q q.leadingCoeff⁻¹
    /-
      🎉 no goals
    -/


theorem gcd_map [Field k] [DecidableEq R] [DecidableEq k] (f : R →+* k) :
    gcd (p.map f) (q.map f) = (gcd p q).map f :=
                                 /-
                                   R : Type u
                                   k : Type y
                                   inst✝³ : Field R
                                   p q : Polynomial R
                                   inst✝² : Field k
                                   inst✝¹ : DecidableEq R
                                   inst✝ : DecidableEq k
                                   f : RingHom R k
                                   x : Polynomial R
                                   ⊢ Eq (EuclideanDomain.gcd (Polynomial.map f 0) (Polynomial.map f x)) (Polynomi …
                                 -/
  GCD.induction p q (fun x => by simp_rw [Polynomial.map_zero, EuclideanDomain.gcd_zero_left])
                                 /-
                                   🎉 no goals
                                 -/
                       /-
                         R : Type u
                         k : Type y
                         inst✝³ : Field R
                         p q : Polynomial R
                         inst✝² : Field k
                         inst✝¹ : DecidableEq R
                         inst✝ : DecidableEq k
                         f : RingHom R k
                         x y : Polynomial R
                         x✝ : Ne x 0
                         ih : Eq (EuclideanDomain.gcd (Polynomial.map f (HMod.hMod y x)) (Polynomial.ma …
                         ⊢ Eq (EuclideanDomain.gcd (Polynomial.map f x) (Polynomial.map f y)) (Polynomi …
                       -/
    fun x y _ ih => by rw [gcd_val, ← map_mod, ih, ← gcd_val]
                       /-
                         🎉 no goals
                       -/


theorem eval₂_gcd_eq_zero [CommSemiring k] [DecidableEq R]
    {ϕ : R →+* k} {f g : R[X]} {α : k} (hf : f.eval₂ ϕ α = 0)
    (hg : g.eval₂ ϕ α = 0) : (EuclideanDomain.gcd f g).eval₂ ϕ α = 0 := by
  rw [EuclideanDomain.gcd_eq_gcd_ab f g, Polynomial.eval₂_add, Polynomial.eval₂_mul,
    Polynomial.eval₂_mul, hf, hg, zero_mul, zero_mul, zero_add]


theorem eval_gcd_eq_zero [DecidableEq R] {f g : R[X]} {α : R}
    (hf : f.eval α = 0) (hg : g.eval α = 0) : (EuclideanDomain.gcd f g).eval α = 0 :=
  eval₂_gcd_eq_zero hf hg


theorem root_left_of_root_gcd [CommSemiring k] [DecidableEq R] {ϕ : R →+* k} {f g : R[X]} {α : k}
    (hα : (EuclideanDomain.gcd f g).eval₂ ϕ α = 0) : f.eval₂ ϕ α = 0 := by
  /-
    R : Type u
    k : Type y
    inst✝² : Field R
    inst✝¹ : CommSemiring k
    inst✝ : DecidableEq R
    ϕ : RingHom R k
    f g : Polynomial R
    α : k
    hα : Eq (Polynomial.eval₂ ϕ α (EuclideanDomain.gcd f g)) 0
    ⊢ Eq (Polynomial.eval₂ ϕ α f) 0
  -/
  cases' EuclideanDomain.gcd_dvd_left f g with p hp
  /-
    case intro
    R : Type u
    k : Type y
    inst✝² : Field R
    inst✝¹ : CommSemiring k
    inst✝ : DecidableEq R
    ϕ : RingHom R k
    f g : Polynomial R
    α : k
    hα : Eq (Polynomial.eval₂ ϕ α (EuclideanDomain.gcd f g)) 0
    p : Polynomial R
    hp : Eq f (HMul.hMul (EuclideanDomain.gcd f g) p)
    ⊢ Eq (Polynomial.eval₂ ϕ α f) 0
  -/
  rw [hp, Polynomial.eval₂_mul, hα, zero_mul]
  /-
    🎉 no goals
  -/


theorem root_right_of_root_gcd [CommSemiring k] [DecidableEq R] {ϕ : R →+* k} {f g : R[X]} {α : k}
    (hα : (EuclideanDomain.gcd f g).eval₂ ϕ α = 0) : g.eval₂ ϕ α = 0 := by
  /-
    R : Type u
    k : Type y
    inst✝² : Field R
    inst✝¹ : CommSemiring k
    inst✝ : DecidableEq R
    ϕ : RingHom R k
    f g : Polynomial R
    α : k
    hα : Eq (Polynomial.eval₂ ϕ α (EuclideanDomain.gcd f g)) 0
    ⊢ Eq (Polynomial.eval₂ ϕ α g) 0
  -/
  cases' EuclideanDomain.gcd_dvd_right f g with p hp
  /-
    case intro
    R : Type u
    k : Type y
    inst✝² : Field R
    inst✝¹ : CommSemiring k
    inst✝ : DecidableEq R
    ϕ : RingHom R k
    f g : Polynomial R
    α : k
    hα : Eq (Polynomial.eval₂ ϕ α (EuclideanDomain.gcd f g)) 0
    p : Polynomial R
    hp : Eq g (HMul.hMul (EuclideanDomain.gcd f g) p)
    ⊢ Eq (Polynomial.eval₂ ϕ α g) 0
  -/
  rw [hp, Polynomial.eval₂_mul, hα, zero_mul]
  /-
    🎉 no goals
  -/


theorem root_gcd_iff_root_left_right [CommSemiring k] [DecidableEq R]
    {ϕ : R →+* k} {f g : R[X]} {α : k} :
    (EuclideanDomain.gcd f g).eval₂ ϕ α = 0 ↔ f.eval₂ ϕ α = 0 ∧ g.eval₂ ϕ α = 0 :=
  ⟨fun h => ⟨root_left_of_root_gcd h, root_right_of_root_gcd h⟩, fun h => eval₂_gcd_eq_zero h.1 h.2⟩


theorem isRoot_gcd_iff_isRoot_left_right [DecidableEq R] {f g : R[X]} {α : R} :
    (EuclideanDomain.gcd f g).IsRoot α ↔ f.IsRoot α ∧ g.IsRoot α :=
  root_gcd_iff_root_left_right


theorem isCoprime_map [Field k] (f : R →+* k) : IsCoprime (p.map f) (q.map f) ↔ IsCoprime p q := by
  classical
  rw [← EuclideanDomain.gcd_isUnit_iff, ← EuclideanDomain.gcd_isUnit_iff, gcd_map, isUnit_map]


theorem mem_roots_map [CommRing k] [IsDomain k] {f : R →+* k} {x : k} (hp : p ≠ 0) :
    x ∈ (p.map f).roots ↔ p.eval₂ f x = 0 := by
  /-
    R : Type u
    k : Type y
    inst✝² : Field R
    p : Polynomial R
    inst✝¹ : CommRing k
    inst✝ : IsDomain k
    f : RingHom R k
    x : k
    hp : Ne p 0
    ⊢ Iff (Membership.mem (Polynomial.map f p).roots x) (Eq (Polynomial.eval₂ f x  …
  -/
  rw [mem_roots (map_ne_zero hp), IsRoot, Polynomial.eval_map]
  /-
    🎉 no goals
  -/


theorem rootSet_monomial [CommRing S] [IsDomain S] [Algebra R S] {n : ℕ} (hn : n ≠ 0) {a : R}
    (ha : a ≠ 0) : (monomial n a).rootSet S = {0} := by
  classical
  rw [rootSet, aroots_monomial ha,
    Multiset.toFinset_nsmul _ _ hn, Multiset.toFinset_singleton, Finset.coe_singleton]


theorem rootSet_C_mul_X_pow [CommRing S] [IsDomain S] [Algebra R S] {n : ℕ} (hn : n ≠ 0) {a : R}
    (ha : a ≠ 0) : rootSet (C a * X ^ n) S = {0} := by
  /-
    R : Type u
    S : Type v
    inst✝³ : Field R
    inst✝² : CommRing S
    inst✝¹ : IsDomain S
    inst✝ : Algebra R S
    n : Nat
    hn : Ne n 0
    a : R
    ha : Ne a 0
    ⊢ Eq ((HMul.hMul (Polynomial.C a) (HPow.hPow Polynomial.X n)).rootSet S) (Sing …
  -/
  rw [C_mul_X_pow_eq_monomial, rootSet_monomial hn ha]
  /-
    🎉 no goals
  -/


theorem rootSet_X_pow [CommRing S] [IsDomain S] [Algebra R S] {n : ℕ} (hn : n ≠ 0) :
    (X ^ n : R[X]).rootSet S = {0} := by
  /-
    R : Type u
    S : Type v
    inst✝³ : Field R
    inst✝² : CommRing S
    inst✝¹ : IsDomain S
    inst✝ : Algebra R S
    n : Nat
    hn : Ne n 0
    ⊢ Eq ((HPow.hPow Polynomial.X n).rootSet S) (Singleton.singleton 0)
  -/
  rw [← one_mul (X ^ n : R[X]), ← C_1, rootSet_C_mul_X_pow hn]
  /-
    R : Type u
    S : Type v
    inst✝³ : Field R
    inst✝² : CommRing S
    inst✝¹ : IsDomain S
    inst✝ : Algebra R S
    n : Nat
    hn : Ne n 0
    ⊢ Ne 1 0
  -/
  exact one_ne_zero
  /-
    🎉 no goals
  -/


theorem rootSet_prod [CommRing S] [IsDomain S] [Algebra R S] {ι : Type*} (f : ι → R[X])
    (s : Finset ι) (h : s.prod f ≠ 0) : (s.prod f).rootSet S = ⋃ i ∈ s, (f i).rootSet S := by
  classical
  simp only [rootSet, aroots, ← Finset.mem_coe]
  rw [Polynomial.map_prod, roots_prod, Finset.bind_toFinset, s.val_toFinset, Finset.coe_biUnion]
  rwa [← Polynomial.map_prod, Ne, Polynomial.map_eq_zero]


theorem roots_C_mul_X_sub_C (b : R) (ha : a ≠ 0) : (C a * X - C b).roots = {a⁻¹ * b} := by
  /-
    R : Type u
    a : R
    inst✝ : Field R
    b : R
    ha : Ne a 0
    ⊢ Eq (HSub.hSub (HMul.hMul (Polynomial.C a) Polynomial.X) (Polynomial.C b)).ro …
  -/
  simp [roots_C_mul_X_sub_C_of_IsUnit b ⟨a, a⁻¹, mul_inv_cancel₀ ha, inv_mul_cancel₀ ha⟩]
  /-
    🎉 no goals
  -/


theorem roots_C_mul_X_add_C (b : R) (ha : a ≠ 0) : (C a * X + C b).roots = {-(a⁻¹ * b)} := by
  /-
    R : Type u
    a : R
    inst✝ : Field R
    b : R
    ha : Ne a 0
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Polynomial.C a) Polynomial.X) (Polynomial.C b)).ro …
  -/
  simp [roots_C_mul_X_add_C_of_IsUnit b ⟨a, a⁻¹, mul_inv_cancel₀ ha, inv_mul_cancel₀ ha⟩]
  /-
    🎉 no goals
  -/


theorem roots_degree_eq_one (h : degree p = 1) : p.roots = {-((p.coeff 1)⁻¹ * p.coeff 0)} := by
  /-
    R : Type u
    inst✝ : Field R
    p : Polynomial R
    h : Eq p.degree 1
    ⊢ Eq p.roots (Singleton.singleton (Neg.neg (HMul.hMul (Inv.inv (p.coeff 1)) (p …
  -/
  rw [eq_X_add_C_of_degree_le_one (show degree p ≤ 1 by rw [h])]
  /-
    R : Type u
    inst✝ : Field R
    p : Polynomial R
    h : Eq p.degree 1
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Polynomial.C (p.coeff 1)) Polynomial.X) (Polynomia …
  -/
  have : p.coeff 1 ≠ 0 := coeff_ne_zero_of_eq_degree h
  /-
    R : Type u
    inst✝ : Field R
    p : Polynomial R
    h : Eq p.degree 1
    this : Ne (p.coeff 1) 0
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Polynomial.C (p.coeff 1)) Polynomial.X) (Polynomia …
  -/
  simp [roots_C_mul_X_add_C _ this]
  /-
    🎉 no goals
  -/


theorem exists_root_of_degree_eq_one (h : degree p = 1) : ∃ x, IsRoot p x :=
  ⟨-((p.coeff 1)⁻¹ * p.coeff 0), by
    /-
      R : Type u
      inst✝ : Field R
      p : Polynomial R
      h : Eq p.degree 1
      ⊢ p.IsRoot (Neg.neg (HMul.hMul (Inv.inv (p.coeff 1)) (p.coeff 0)))
    -/
    rw [← mem_roots (by simp [← zero_le_degree_iff, h])]
    /-
      R : Type u
      inst✝ : Field R
      p : Polynomial R
      h : Eq p.degree 1
      ⊢ Membership.mem p.roots (Neg.neg (HMul.hMul (Inv.inv (p.coeff 1)) (p.coeff 0)))
    -/
    simp [roots_degree_eq_one h]⟩
    /-
      🎉 no goals
    -/


theorem coeff_inv_units (u : R[X]ˣ) (n : ℕ) : ((↑u : R[X]).coeff n)⁻¹ = (↑u⁻¹ : R[X]).coeff n := by
  rw [eq_C_of_degree_eq_zero (degree_coe_units u), eq_C_of_degree_eq_zero (degree_coe_units u⁻¹),
    coeff_C, coeff_C, inv_eq_one_div]
  /-
    R : Type u
    inst✝ : Field R
    u : Units (Polynomial R)
    n : Nat
    ⊢ Eq (HDiv.hDiv 1 (ite (Eq n 0) ((↑u).coeff 0) 0)) (ite (Eq n 0) ((↑(Inv.inv u …
  -/
  split_ifs
  · rw [div_eq_iff_mul_eq (coeff_coe_units_zero_ne_zero u), coeff_zero_eq_eval_zero,
        coeff_zero_eq_eval_zero, ← eval_mul, ← Units.val_mul, inv_mul_cancel]
    /-
      case pos
      R : Type u
      inst✝ : Field R
      u : Units (Polynomial R)
      n : Nat
      h✝ : Eq n 0
      ⊢ Eq (Polynomial.eval 0 ↑1) 1
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u
      inst✝ : Field R
      u : Units (Polynomial R)
      n : Nat
      h✝ : Not (Eq n 0)
      ⊢ Eq (1 / 0) 0
    -/
  · simp
    /-
      🎉 no goals
    -/


theorem monic_normalize [DecidableEq R] (hp0 : p ≠ 0) : Monic (normalize p) := by
  /-
    R : Type u
    inst✝¹ : Field R
    p : Polynomial R
    inst✝ : DecidableEq R
    hp0 : Ne p 0
    ⊢ (normalize p).Monic
  -/
  rw [Ne, ← leadingCoeff_eq_zero, ← Ne, ← isUnit_iff_ne_zero] at hp0
  /-
    R : Type u
    inst✝¹ : Field R
    p : Polynomial R
    inst✝ : DecidableEq R
    hp0 : IsUnit p.leadingCoeff
    ⊢ (normalize p).Monic
  -/
  rw [Monic, leadingCoeff_normalize, normalize_eq_one]
  /-
    R : Type u
    inst✝¹ : Field R
    p : Polynomial R
    inst✝ : DecidableEq R
    hp0 : IsUnit p.leadingCoeff
    ⊢ IsUnit p.leadingCoeff
  -/
  apply hp0
  /-
    🎉 no goals
  -/


theorem leadingCoeff_div (hpq : q.degree ≤ p.degree) :
    (p / q).leadingCoeff = p.leadingCoeff / q.leadingCoeff := by
  /-
    R : Type u
    inst✝ : Field R
    p q : Polynomial R
    hpq : LE.le q.degree p.degree
    ⊢ Eq (HDiv.hDiv p q).leadingCoeff (HDiv.hDiv p.leadingCoeff q.leadingCoeff)
  -/
  by_cases hq : q = 0
    /-
      case pos
      R : Type u
      inst✝ : Field R
      p q : Polynomial R
      hpq : LE.le q.degree p.degree
      hq : Eq q 0
      ⊢ Eq (HDiv.hDiv p q).leadingCoeff (HDiv.hDiv p.leadingCoeff q.leadingCoeff)
    -/
  · simp [hq]
    /-
      🎉 no goals
    -/
  rw [div_def, leadingCoeff_mul, leadingCoeff_C,
    leadingCoeff_divByMonic_of_monic (monic_mul_leadingCoeff_inv hq) _, mul_comm,
    div_eq_mul_inv]
  /-
    R : Type u
    inst✝ : Field R
    p q : Polynomial R
    hpq : LE.le q.degree p.degree
    hq : Not (Eq q 0)
    ⊢ LE.le (HMul.hMul q (Polynomial.C (Inv.inv q.leadingCoeff))).degree p.degree
  -/
  rwa [degree_mul_leadingCoeff_inv q hq]
  /-
    🎉 no goals
  -/


theorem div_C_mul : p / (C a * q) = C a⁻¹ * (p / q) := by
  /-
    R : Type u
    a : R
    inst✝ : Field R
    p q : Polynomial R
    ⊢ Eq (HDiv.hDiv p (HMul.hMul (Polynomial.C a) q)) (HMul.hMul (Polynomial.C (In …
  -/
  by_cases ha : a = 0
    /-
      case pos
      R : Type u
      a : R
      inst✝ : Field R
      p q : Polynomial R
      ha : Eq a 0
      ⊢ Eq (HDiv.hDiv p (HMul.hMul (Polynomial.C a) q)) (HMul.hMul (Polynomial.C (In …
    -/
  · simp [ha]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u
    a : R
    inst✝ : Field R
    p q : Polynomial R
    ha : Not (Eq a 0)
    ⊢ Eq (HDiv.hDiv p (HMul.hMul (Polynomial.C a) q)) (HMul.hMul (Polynomial.C (In …
  -/
  simp only [div_def, leadingCoeff_mul, mul_inv, leadingCoeff_C, C.map_mul, mul_assoc]
  /-
    case neg
    R : Type u
    a : R
    inst✝ : Field R
    p q : Polynomial R
    ha : Not (Eq a 0)
    ⊢ Eq (HMul.hMul (Polynomial.C (Inv.inv a)) (HMul.hMul (Polynomial.C (Inv.inv q …
  -/
  congr 3
  /-
    case neg.e_a.e_a.e_q
    R : Type u
    a : R
    inst✝ : Field R
    p q : Polynomial R
    ha : Not (Eq a 0)
    ⊢ Eq (HMul.hMul (Polynomial.C a) (HMul.hMul q (HMul.hMul (Polynomial.C (Inv.in …
  -/
  rw [mul_left_comm q, ← mul_assoc, ← C.map_mul, mul_inv_cancel₀ ha, C.map_one, one_mul]
  /-
    🎉 no goals
  -/


theorem C_mul_dvd (ha : a ≠ 0) : C a * p ∣ q ↔ p ∣ q :=
  ⟨fun h => dvd_trans (dvd_mul_left _ _) h, fun ⟨r, hr⟩ =>
    ⟨C a⁻¹ * r, by
      rw [mul_assoc, mul_left_comm p, ← mul_assoc, ← C.map_mul, mul_inv_cancel₀ ha, C.map_one,
        one_mul, hr]⟩⟩


theorem dvd_C_mul (ha : a ≠ 0) : p ∣ Polynomial.C a * q ↔ p ∣ q :=
  ⟨fun ⟨r, hr⟩ =>
    ⟨C a⁻¹ * r, by
      rw [mul_left_comm p, ← hr, ← mul_assoc, ← C.map_mul, inv_mul_cancel₀ ha, C.map_one,
        one_mul]⟩,
    fun h => dvd_trans h (dvd_mul_left _ _)⟩


theorem coe_normUnit_of_ne_zero [DecidableEq R] (hp : p ≠ 0) :
    (normUnit p : R[X]) = C p.leadingCoeff⁻¹ := by
  /-
    R : Type u
    inst✝¹ : Field R
    p : Polynomial R
    inst✝ : DecidableEq R
    hp : Ne p 0
    ⊢ Eq (↑(NormalizationMonoid.normUnit p)) (Polynomial.C (Inv.inv p.leadingCoeff))
  -/
  have : p.leadingCoeff ≠ 0 := mt leadingCoeff_eq_zero.mp hp
  /-
    R : Type u
    inst✝¹ : Field R
    p : Polynomial R
    inst✝ : DecidableEq R
    hp : Ne p 0
    this : Ne p.leadingCoeff 0
    ⊢ Eq (↑(NormalizationMonoid.normUnit p)) (Polynomial.C (Inv.inv p.leadingCoeff))
  -/
  simp [CommGroupWithZero.coe_normUnit _ this]
  /-
    🎉 no goals
  -/


theorem map_dvd_map' [Field k] (f : R →+* k) {x y : R[X]} : x.map f ∣ y.map f ↔ x ∣ y := by
  /-
    R : Type u
    k : Type y
    inst✝¹ : Field R
    inst✝ : Field k
    f : RingHom R k
    x y : Polynomial R
    ⊢ Iff (Dvd.dvd (Polynomial.map f x) (Polynomial.map f y)) (Dvd.dvd x y)
  -/
  by_cases H : x = 0
    /-
      case pos
      R : Type u
      k : Type y
      inst✝¹ : Field R
      inst✝ : Field k
      f : RingHom R k
      x y : Polynomial R
      H : Eq x 0
      ⊢ Iff (Dvd.dvd (Polynomial.map f x) (Polynomial.map f y)) (Dvd.dvd x y)
    -/
  · rw [H, Polynomial.map_zero, zero_dvd_iff, zero_dvd_iff, Polynomial.map_eq_zero]
    /-
      🎉 no goals
    -/
  · classical
    rw [← normalize_dvd_iff, ← @normalize_dvd_iff R[X], normalize_apply, normalize_apply,
      coe_normUnit_of_ne_zero H, coe_normUnit_of_ne_zero (mt (Polynomial.map_eq_zero f).1 H),
      leadingCoeff_map, ← map_inv₀ f, ← map_C, ← Polynomial.map_mul,
      map_dvd_map _ f.injective (monic_mul_leadingCoeff_inv H)]


@[simp]
theorem degree_normalize [DecidableEq R] : degree (normalize p) = degree p := by
  /-
    R : Type u
    inst✝¹ : Field R
    p : Polynomial R
    inst✝ : DecidableEq R
    ⊢ Eq (normalize p).degree p.degree
  -/
  simp [normalize_apply]
  /-
    🎉 no goals
  -/


theorem prime_of_degree_eq_one (hp1 : degree p = 1) : Prime p := by
  classical
  have : Prime (normalize p) :=
    Monic.prime_of_degree_eq_one (hp1 ▸ degree_normalize)
      (monic_normalize fun hp0 => absurd hp1 (hp0.symm ▸ by simp [degree_zero]))
  exact (normalize_associated _).prime this


theorem irreducible_of_degree_eq_one (hp1 : degree p = 1) : Irreducible p :=
  (prime_of_degree_eq_one hp1).irreducible


theorem not_irreducible_C (x : R) : ¬Irreducible (C x) := by
  /-
    R : Type u
    inst✝ : Field R
    x : R
    ⊢ Not (Irreducible (Polynomial.C x))
  -/
  by_cases H : x = 0
    /-
      case pos
      R : Type u
      inst✝ : Field R
      x : R
      H : Eq x 0
      ⊢ Not (Irreducible (Polynomial.C x))
    -/
  · rw [H, C_0]
    /-
      case pos
      R : Type u
      inst✝ : Field R
      x : R
      H : Eq x 0
      ⊢ Not (Irreducible 0)
    -/
    exact not_irreducible_zero
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u
      inst✝ : Field R
      x : R
      H : Not (Eq x 0)
      ⊢ Not (Irreducible (Polynomial.C x))
    -/
  · exact fun hx => Irreducible.not_unit hx <| isUnit_C.2 <| isUnit_iff_ne_zero.2 H
    /-
      🎉 no goals
    -/


theorem degree_pos_of_irreducible (hp : Irreducible p) : 0 < p.degree :=
  lt_of_not_ge fun hp0 =>
    have := eq_C_of_degree_le_zero hp0
    not_irreducible_C (p.coeff 0) <| this ▸ hp

/- Porting note: factored out a have statement from isCoprime_of_is_root_of_eval_derivative_ne_zero
into multiple decls because the original proof was timing out -/

theorem X_sub_C_mul_divByMonic_eq_sub_modByMonic {K : Type*} [Field K] (f : K[X]) (a : K) :
    (X - C a) * (f /ₘ (X - C a)) = f - f %ₘ (X - C a) := by
  /-
    K : Type u_1
    inst✝ : Field K
    f : Polynomial K
    a : K
    ⊢ Eq (HMul.hMul (HSub.hSub Polynomial.X (Polynomial.C a)) (f.divByMonic (HSub. …
  -/
  rw [eq_sub_iff_add_eq, ← eq_sub_iff_add_eq', modByMonic_eq_sub_mul_div]
  /-
    case _hq
    K : Type u_1
    inst✝ : Field K
    f : Polynomial K
    a : K
    ⊢ (HSub.hSub Polynomial.X (Polynomial.C a)).Monic
  -/
  exact monic_X_sub_C a
  /-
    🎉 no goals
  -/

/- Porting note: factored out a have statement from isCoprime_of_is_root_of_eval_derivative_ne_zero
because the original proof was timing out -/

theorem divByMonic_add_X_sub_C_mul_derivate_divByMonic_eq_derivative
    {K : Type*} [Field K] (f : K[X]) (a : K) :
    f /ₘ (X - C a) + (X - C a) * derivative (f /ₘ (X - C a)) = derivative f := by
  /-
    K : Type u_1
    inst✝ : Field K
    f : Polynomial K
    a : K
    ⊢ Eq (HAdd.hAdd (f.divByMonic (HSub.hSub Polynomial.X (Polynomial.C a))) (HMul …
  -/
  have key := by apply congrArg derivative <| X_sub_C_mul_divByMonic_eq_sub_modByMonic f a
  /-
    K : Type u_1
    inst✝ : Field K
    f : Polynomial K
    a : K
    key : Eq (Polynomial.derivative (HMul.hMul (HSub.hSub Polynomial.X (Polynomial …
    ⊢ Eq (HAdd.hAdd (f.divByMonic (HSub.hSub Polynomial.X (Polynomial.C a))) (HMul …
  -/
  rw [modByMonic_X_sub_C_eq_C_eval] at key
  /-
    K : Type u_1
    inst✝ : Field K
    f : Polynomial K
    a : K
    key : Eq (Polynomial.derivative (HMul.hMul (HSub.hSub Polynomial.X (Polynomial …
    ⊢ Eq (HAdd.hAdd (f.divByMonic (HSub.hSub Polynomial.X (Polynomial.C a))) (HMul …
  -/
  rw [derivative_mul,derivative_sub,derivative_X,derivative_sub] at key
  /-
    K : Type u_1
    inst✝ : Field K
    f : Polynomial K
    a : K
    key : Eq (HAdd.hAdd (HMul.hMul (HSub.hSub 1 (Polynomial.derivative (Polynomial …
    ⊢ Eq (HAdd.hAdd (f.divByMonic (HSub.hSub Polynomial.X (Polynomial.C a))) (HMul …
  -/
  rw [derivative_C,sub_zero,one_mul] at key
  /-
    K : Type u_1
    inst✝ : Field K
    f : Polynomial K
    a : K
    key : Eq (HAdd.hAdd (f.divByMonic (HSub.hSub Polynomial.X (Polynomial.C a))) ( …
    ⊢ Eq (HAdd.hAdd (f.divByMonic (HSub.hSub Polynomial.X (Polynomial.C a))) (HMul …
  -/
  rw [derivative_C,sub_zero] at key
  /-
    K : Type u_1
    inst✝ : Field K
    f : Polynomial K
    a : K
    key : Eq (HAdd.hAdd (f.divByMonic (HSub.hSub Polynomial.X (Polynomial.C a))) ( …
    ⊢ Eq (HAdd.hAdd (f.divByMonic (HSub.hSub Polynomial.X (Polynomial.C a))) (HMul …
  -/
  assumption
  /-
    🎉 no goals
  -/

/- Porting note: factored out another have statement from
isCoprime_of_is_root_of_eval_derivative_ne_zero because the original proof was timing out -/

theorem X_sub_C_dvd_derivative_of_X_sub_C_dvd_divByMonic {K : Type*} [Field K] (f : K[X]) {a : K}
    (hf : (X - C a) ∣ f /ₘ (X - C a)) : X - C a ∣ derivative f := by
  /-
    K : Type u_1
    inst✝ : Field K
    f : Polynomial K
    a : K
    hf : Dvd.dvd (HSub.hSub Polynomial.X (Polynomial.C a)) (f.divByMonic (HSub.hSu …
    ⊢ Dvd.dvd (HSub.hSub Polynomial.X (Polynomial.C a)) (Polynomial.derivative f)
  -/
  have key := divByMonic_add_X_sub_C_mul_derivate_divByMonic_eq_derivative f a
  /-
    K : Type u_1
    inst✝ : Field K
    f : Polynomial K
    a : K
    hf : Dvd.dvd (HSub.hSub Polynomial.X (Polynomial.C a)) (f.divByMonic (HSub.hSu …
    key : Eq (HAdd.hAdd (f.divByMonic (HSub.hSub Polynomial.X (Polynomial.C a))) ( …
    ⊢ Dvd.dvd (HSub.hSub Polynomial.X (Polynomial.C a)) (Polynomial.derivative f)
  -/
  have ⟨u,hu⟩ := hf
  /-
    K : Type u_1
    inst✝ : Field K
    f : Polynomial K
    a : K
    hf : Dvd.dvd (HSub.hSub Polynomial.X (Polynomial.C a)) (f.divByMonic (HSub.hSu …
    key : Eq (HAdd.hAdd (f.divByMonic (HSub.hSub Polynomial.X (Polynomial.C a))) ( …
    u : Polynomial K
    hu : Eq (f.divByMonic (HSub.hSub Polynomial.X (Polynomial.C a))) (HMul.hMul (H …
    ⊢ Dvd.dvd (HSub.hSub Polynomial.X (Polynomial.C a)) (Polynomial.derivative f)
  -/
  rw [← key, hu, ← mul_add (X - C a) u _]
  /-
    K : Type u_1
    inst✝ : Field K
    f : Polynomial K
    a : K
    hf : Dvd.dvd (HSub.hSub Polynomial.X (Polynomial.C a)) (f.divByMonic (HSub.hSu …
    key : Eq (HAdd.hAdd (f.divByMonic (HSub.hSub Polynomial.X (Polynomial.C a))) ( …
    u : Polynomial K
    hu : Eq (f.divByMonic (HSub.hSub Polynomial.X (Polynomial.C a))) (HMul.hMul (H …
    ⊢ Dvd.dvd (HSub.hSub Polynomial.X (Polynomial.C a)) (HMul.hMul (HSub.hSub Poly …
  -/
  use (u + derivative ((X - C a) * u))
  /-
    🎉 no goals
  -/


/-- If `f` is a polynomial over a field, and `a : K` satisfies `f' a ≠ 0`,
then `f / (X - a)` is coprime with `X - a`.
Note that we do not assume `f a = 0`, because `f / (X - a) = (f - f a) / (X - a)`. -/
theorem isCoprime_of_is_root_of_eval_derivative_ne_zero {K : Type*} [Field K] (f : K[X]) (a : K)
    (hf' : f.derivative.eval a ≠ 0) : IsCoprime (X - C a : K[X]) (f /ₘ (X - C a)) := by
  classical
  refine Or.resolve_left
      (EuclideanDomain.dvd_or_coprime (X - C a) (f /ₘ (X - C a))
        (irreducible_of_degree_eq_one (Polynomial.degree_X_sub_C a))) ?_
  contrapose! hf' with h
  have : X - C a ∣ derivative f := X_sub_C_dvd_derivative_of_X_sub_C_dvd_divByMonic f h
  rw [← modByMonic_eq_zero_iff_dvd (monic_X_sub_C _), modByMonic_X_sub_C_eq_C_eval] at this
  rwa [← C_inj, C_0]


/-- To check a polynomial over a field is irreducible, it suffices to check only for
divisors that have smaller degree.

See also: `Polynomial.Monic.irreducible_iff_natDegree`.
-/
theorem irreducible_iff_degree_lt (p : R[X]) (hp0 : p ≠ 0) (hpu : ¬ IsUnit p) :
    Irreducible p ↔ ∀ q, q.degree ≤ ↑(natDegree p / 2) → q ∣ p → IsUnit q := by
  rw [← irreducible_mul_leadingCoeff_inv,
      (monic_mul_leadingCoeff_inv hp0).irreducible_iff_degree_lt]
    /-
      R : Type u
      inst✝ : Field R
      p : Polynomial R
      hp0 : Ne p 0
      hpu : Not (IsUnit p)
      ⊢ Iff (∀ (q : Polynomial R), LE.le q.degree ↑(HDiv.hDiv (HMul.hMul p (Polynomi …
    -/
  · simp [hp0, natDegree_mul_leadingCoeff_inv]
    /-
      🎉 no goals
    -/
    /-
      R : Type u
      inst✝ : Field R
      p : Polynomial R
      hp0 : Ne p 0
      hpu : Not (IsUnit p)
      ⊢ Ne (HMul.hMul p (Polynomial.C (Inv.inv p.leadingCoeff))) 1
    -/
  · contrapose! hpu
    /-
      R : Type u
      inst✝ : Field R
      p : Polynomial R
      hp0 : Ne p 0
      hpu : Eq (HMul.hMul p (Polynomial.C (Inv.inv p.leadingCoeff))) 1
      ⊢ IsUnit p
    -/
    exact isUnit_of_mul_eq_one _ _ hpu
    /-
      🎉 no goals
    -/


/-- To check a polynomial `p` over a field is irreducible, it suffices to check there are no
divisors of degree `0 < d ≤ degree p / 2`.

See also: `Polynomial.Monic.irreducible_iff_natDegree'`.
-/
theorem irreducible_iff_lt_natDegree_lt {p : R[X]} (hp0 : p ≠ 0) (hpu : ¬ IsUnit p) :
    Irreducible p ↔ ∀ q, Monic q → natDegree q ∈ Finset.Ioc 0 (natDegree p / 2) → ¬ q ∣ p := by
  have : p * C (leadingCoeff p)⁻¹ ≠ 1 := by
    contrapose! hpu
    exact isUnit_of_mul_eq_one _ _ hpu
  rw [← irreducible_mul_leadingCoeff_inv,
      (monic_mul_leadingCoeff_inv hp0).irreducible_iff_lt_natDegree_lt this,
      natDegree_mul_leadingCoeff_inv _ hp0]
  simp only [IsUnit.dvd_mul_right
    (isUnit_C.mpr (IsUnit.mk0 (leadingCoeff p)⁻¹ (inv_ne_zero (leadingCoeff_ne_zero.mpr hp0))))]


open UniqueFactorizationMonoid in
/--
The normalized factors of a polynomial over a field times its leading coefficient give
the polynomial.
-/
theorem leadingCoeff_mul_prod_normalizedFactors [DecidableEq R] (a : R[X]) :
    C a.leadingCoeff * (normalizedFactors a).prod = a := by
  /-
    R : Type u
    inst✝¹ : Field R
    inst✝ : DecidableEq R
    a : Polynomial R
    ⊢ Eq (HMul.hMul (Polynomial.C a.leadingCoeff) (UniqueFactorizationMonoid.norma …
  -/
  by_cases ha : a = 0
    /-
      case pos
      R : Type u
      inst✝¹ : Field R
      inst✝ : DecidableEq R
      a : Polynomial R
      ha : Eq a 0
      ⊢ Eq (HMul.hMul (Polynomial.C a.leadingCoeff) (UniqueFactorizationMonoid.norma …
    -/
  · simp [ha]
    /-
      🎉 no goals
    -/
  rw [prod_normalizedFactors_eq, normalize_apply, coe_normUnit, CommGroupWithZero.coe_normUnit,
    mul_comm, mul_assoc, ← map_mul, inv_mul_cancel₀] <;>
  /-
    case neg
    R : Type u
    inst✝¹ : Field R
    inst✝ : DecidableEq R
    a : Polynomial R
    ha : Not (Eq a 0)
    ⊢ Eq (HMul.hMul a (Polynomial.C 1)) a
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
  simp_all
  /-
    🎉 no goals
  -/


/-- An irreducible polynomial over a field must have positive degree. -/
theorem Irreducible.natDegree_pos {F : Type*} [Field F] {f : F[X]} (h : Irreducible f) :
    0 < f.natDegree := Nat.pos_of_ne_zero fun H ↦ by
  /-
    F : Type u_1
    inst✝ : Field F
    f : Polynomial F
    h : Irreducible f
    H : Eq f.natDegree 0
    ⊢ False
  -/
  obtain ⟨x, hf⟩ := natDegree_eq_zero.1 H
  /-
    case intro
    F : Type u_1
    inst✝ : Field F
    f : Polynomial F
    h : Irreducible f
    H : Eq f.natDegree 0
    x : F
    hf : Eq (Polynomial.C x) f
    ⊢ False
  -/
  by_cases hx : x = 0
    /-
      case pos
      F : Type u_1
      inst✝ : Field F
      f : Polynomial F
      h : Irreducible f
      H : Eq f.natDegree 0
      x : F
      hf : Eq (Polynomial.C x) f
      hx : Eq x 0
      ⊢ False
    -/
  · rw [← hf, hx, map_zero] at h; exact not_irreducible_zero h
                                  /-
                                    🎉 no goals
                                  -/
  /-
    case neg
    F : Type u_1
    inst✝ : Field F
    f : Polynomial F
    h : Irreducible f
    H : Eq f.natDegree 0
    x : F
    hf : Eq (Polynomial.C x) f
    hx : Not (Eq x 0)
    ⊢ False
  -/
  exact h.1 (hf ▸ isUnit_C.2 (Ne.isUnit hx))
  /-
    🎉 no goals
  -/

