theorem natDegree_pos_of_aeval_root [Algebra R S] {p : R[X]} (hp : p ≠ 0) {z : S}
    (hz : aeval z p = 0) (inj : ∀ x : R, algebraMap R S x = 0 → x = 0) : 0 < p.natDegree :=
  natDegree_pos_of_eval₂_root hp (algebraMap R S) hz inj


theorem degree_pos_of_aeval_root [Algebra R S] {p : R[X]} (hp : p ≠ 0) {z : S} (hz : aeval z p = 0)
    (inj : ∀ x : R, algebraMap R S x = 0 → x = 0) : 0 < p.degree :=
  natDegree_pos_iff_degree_pos.mp (natDegree_pos_of_aeval_root hp hz inj)


theorem smul_modByMonic (c : R) (p : R[X]) : c • p %ₘ q = c • (p %ₘ q) := by
  /-
    R : Type u
    inst✝ : CommRing R
    q : Polynomial R
    c : R
    p : Polynomial R
    ⊢ Eq ((HSMul.hSMul c p).modByMonic q) (HSMul.hSMul c (p.modByMonic q))
  -/
  by_cases hq : q.Monic
    /-
      case pos
      R : Type u
      inst✝ : CommRing R
      q : Polynomial R
      c : R
      p : Polynomial R
      hq : q.Monic
      ⊢ Eq ((HSMul.hSMul c p).modByMonic q) (HSMul.hSMul c (p.modByMonic q))
    -/
  · cases' subsingleton_or_nontrivial R with hR hR
      /-
        case pos.inl
        R : Type u
        inst✝ : CommRing R
        q : Polynomial R
        c : R
        p : Polynomial R
        hq : q.Monic
        hR : Subsingleton R
        ⊢ Eq ((HSMul.hSMul c p).modByMonic q) (HSMul.hSMul c (p.modByMonic q))
      -/
    · simp only [eq_iff_true_of_subsingleton]
      /-
        🎉 no goals
      -/
    · exact
      (div_modByMonic_unique (c • (p /ₘ q)) (c • (p %ₘ q)) hq
          ⟨by rw [mul_smul_comm, ← smul_add, modByMonic_add_div p hq],
            (degree_smul_le _ _).trans_lt (degree_modByMonic_lt _ hq)⟩).2
    /-
      case neg
      R : Type u
      inst✝ : CommRing R
      q : Polynomial R
      c : R
      p : Polynomial R
      hq : Not q.Monic
      ⊢ Eq ((HSMul.hSMul c p).modByMonic q) (HSMul.hSMul c (p.modByMonic q))
    -/
  · simp_rw [modByMonic_eq_of_not_monic _ hq]
    /-
      🎉 no goals
    -/


/-- `_ %ₘ q` as an `R`-linear map. -/
@[simps]
def modByMonicHom (q : R[X]) : R[X] →ₗ[R] R[X] where
  toFun p := p %ₘ q
  map_add' := add_modByMonic
  map_smul' := smul_modByMonic


theorem mem_ker_modByMonic (hq : q.Monic) {p : R[X]} :
    p ∈ LinearMap.ker (modByMonicHom q) ↔ q ∣ p :=
  LinearMap.mem_ker.trans (modByMonic_eq_zero_iff_dvd hq)


theorem aeval_modByMonic_eq_self_of_root [Algebra R S] {p q : R[X]} (hq : q.Monic) {x : S}
    (hx : aeval x q = 0) : aeval x (p %ₘ q) = aeval x p := by
    --`eval₂_modByMonic_eq_self_of_root` doesn't work here as it needs commutativity
  rw [modByMonic_eq_sub_mul_div p hq, _root_.map_sub, _root_.map_mul, hx, zero_mul,
    sub_zero]


theorem trailingDegree_mul : (p * q).trailingDegree = p.trailingDegree + q.trailingDegree := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : NoZeroDivisors R
    p q : Polynomial R
    ⊢ Eq (HMul.hMul p q).trailingDegree (HAdd.hAdd p.trailingDegree q.trailingDegr …
  -/
  by_cases hp : p = 0
    /-
      case pos
      R : Type u
      inst✝¹ : Semiring R
      inst✝ : NoZeroDivisors R
      p q : Polynomial R
      hp : Eq p 0
      ⊢ Eq (HMul.hMul p q).trailingDegree (HAdd.hAdd p.trailingDegree q.trailingDegr …
    -/
  · rw [hp, zero_mul, trailingDegree_zero, top_add]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u
    inst✝¹ : Semiring R
    inst✝ : NoZeroDivisors R
    p q : Polynomial R
    hp : Not (Eq p 0)
    ⊢ Eq (HMul.hMul p q).trailingDegree (HAdd.hAdd p.trailingDegree q.trailingDegr …
  -/
  by_cases hq : q = 0
    /-
      case pos
      R : Type u
      inst✝¹ : Semiring R
      inst✝ : NoZeroDivisors R
      p q : Polynomial R
      hp : Not (Eq p 0)
      hq : Eq q 0
      ⊢ Eq (HMul.hMul p q).trailingDegree (HAdd.hAdd p.trailingDegree q.trailingDegr …
    -/
  · rw [hq, mul_zero, trailingDegree_zero, add_top]
    /-
      🎉 no goals
    -/
  · rw [trailingDegree_eq_natTrailingDegree hp, trailingDegree_eq_natTrailingDegree hq,
    trailingDegree_eq_natTrailingDegree (mul_ne_zero hp hq), natTrailingDegree_mul hp hq]
    /-
      case neg
      R : Type u
      inst✝¹ : Semiring R
      inst✝ : NoZeroDivisors R
      p q : Polynomial R
      hp : Not (Eq p 0)
      hq : Not (Eq q 0)
      ⊢ Eq (↑(HAdd.hAdd p.natTrailingDegree q.natTrailingDegree)) (HAdd.hAdd ↑p.natT …
    -/
    apply WithTop.coe_add
    /-
      🎉 no goals
    -/


theorem rootMultiplicity_eq_rootMultiplicity {p : R[X]} {t : R} :
    p.rootMultiplicity t = (p.comp (X + C t)).rootMultiplicity 0 := by
  classical
  simp_rw [rootMultiplicity_eq_multiplicity, comp_X_add_C_eq_zero_iff]
  congr 1
  rw [C_0, sub_zero]
  convert (multiplicity_map_eq <| algEquivAevalXAddC t).symm using 2
  simp [C_eq_algebraMap]


/-- See `Polynomial.rootMultiplicity_eq_natTrailingDegree'` for the special case of `t = 0`. -/
theorem rootMultiplicity_eq_natTrailingDegree {p : R[X]} {t : R} :
    p.rootMultiplicity t = (p.comp (X + C t)).natTrailingDegree :=
  rootMultiplicity_eq_rootMultiplicity.trans rootMultiplicity_eq_natTrailingDegree'


theorem Monic.mem_nonZeroDivisors {p : R[X]} (h : p.Monic) : p ∈ R[X]⁰ :=
  mem_nonZeroDivisors_iff.2 fun _ hx ↦ (mul_left_eq_zero_iff h).1 hx


theorem mem_nonZeroDivisors_of_leadingCoeff {p : R[X]} (h : p.leadingCoeff ∈ R⁰) : p ∈ R[X]⁰ := by
  /-
    R : Type u
    inst✝ : CommRing R
    p : Polynomial R
    h : Membership.mem (nonZeroDivisors R) p.leadingCoeff
    ⊢ Membership.mem (nonZeroDivisors (Polynomial R)) p
  -/
  refine mem_nonZeroDivisors_iff.2 fun x hx ↦ leadingCoeff_eq_zero.1 ?_
  /-
    R : Type u
    inst✝ : CommRing R
    p : Polynomial R
    h : Membership.mem (nonZeroDivisors R) p.leadingCoeff
    x : Polynomial R
    hx : Eq (HMul.hMul x p) 0
    ⊢ Eq x.leadingCoeff 0
  -/
  by_contra hx'
  /-
    R : Type u
    inst✝ : CommRing R
    p : Polynomial R
    h : Membership.mem (nonZeroDivisors R) p.leadingCoeff
    x : Polynomial R
    hx : Eq (HMul.hMul x p) 0
    hx' : Not (Eq x.leadingCoeff 0)
    ⊢ False
  -/
  rw [← mul_right_mem_nonZeroDivisors_eq_zero_iff h] at hx'
  /-
    R : Type u
    inst✝ : CommRing R
    p : Polynomial R
    h : Membership.mem (nonZeroDivisors R) p.leadingCoeff
    x : Polynomial R
    hx : Eq (HMul.hMul x p) 0
    hx' : Not (Eq (HMul.hMul x.leadingCoeff p.leadingCoeff) 0)
    ⊢ False
  -/
  simp only [← leadingCoeff_mul' hx', hx, leadingCoeff_zero, not_true] at hx'
  /-
    🎉 no goals
  -/


theorem rootMultiplicity_mul_X_sub_C_pow {p : R[X]} {a : R} {n : ℕ} (h : p ≠ 0) :
    (p * (X - C a) ^ n).rootMultiplicity a = p.rootMultiplicity a + n := by
  /-
    R : Type u
    inst✝ : CommRing R
    p : Polynomial R
    a : R
    n : Nat
    h : Ne p 0
    ⊢ Eq (Polynomial.rootMultiplicity a (HMul.hMul p (HPow.hPow (HSub.hSub Polynom …
  -/
  have h2 := monic_X_sub_C a |>.pow n |>.mul_left_ne_zero h
  /-
    R : Type u
    inst✝ : CommRing R
    p : Polynomial R
    a : R
    n : Nat
    h : Ne p 0
    h2 : Ne (HMul.hMul p (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C a)) n)) 0
    ⊢ Eq (Polynomial.rootMultiplicity a (HMul.hMul p (HPow.hPow (HSub.hSub Polynom …
  -/
  refine le_antisymm ?_ ?_
  · rw [rootMultiplicity_le_iff h2, add_assoc, add_comm n, ← add_assoc, pow_add,
      dvd_cancel_right_mem_nonZeroDivisors (monic_X_sub_C a |>.pow n |>.mem_nonZeroDivisors)]
    /-
      case refine_1
      R : Type u
      inst✝ : CommRing R
      p : Polynomial R
      a : R
      n : Nat
      h : Ne p 0
      h2 : Ne (HMul.hMul p (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C a)) n)) 0
      ⊢ Not (Dvd.dvd (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C a)) (HAdd.hAdd …
    -/
    exact pow_rootMultiplicity_not_dvd h a
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u
      inst✝ : CommRing R
      p : Polynomial R
      a : R
      n : Nat
      h : Ne p 0
      h2 : Ne (HMul.hMul p (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C a)) n)) 0
      ⊢ LE.le (HAdd.hAdd (Polynomial.rootMultiplicity a p) n) (Polynomial.rootMultip …
    -/
  · rw [le_rootMultiplicity_iff h2, pow_add]
    /-
      case refine_2
      R : Type u
      inst✝ : CommRing R
      p : Polynomial R
      a : R
      n : Nat
      h : Ne p 0
      h2 : Ne (HMul.hMul p (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C a)) n)) 0
      ⊢ Dvd.dvd (HMul.hMul (HPow.hPow (HSub.hSub Polynomial.X (Polynomial.C a)) (Pol …
    -/
    exact mul_dvd_mul_right (pow_rootMultiplicity_dvd p a) _
    /-
      🎉 no goals
    -/


/-- The multiplicity of `a` as root of `(X - a) ^ n` is `n`. -/
theorem rootMultiplicity_X_sub_C_pow [Nontrivial R] (a : R) (n : ℕ) :
    rootMultiplicity a ((X - C a) ^ n) = n := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : Nontrivial R
    a : R
    n : Nat
    ⊢ Eq (Polynomial.rootMultiplicity a (HPow.hPow (HSub.hSub Polynomial.X (Polyno …
  -/
  have := rootMultiplicity_mul_X_sub_C_pow (a := a) (n := n) C.map_one_ne_zero
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : Nontrivial R
    a : R
    n : Nat
    this : Eq (Polynomial.rootMultiplicity a (HMul.hMul (Polynomial.C 1) (HPow.hPo …
    ⊢ Eq (Polynomial.rootMultiplicity a (HPow.hPow (HSub.hSub Polynomial.X (Polyno …
  -/
  rwa [rootMultiplicity_C, map_one, one_mul, zero_add] at this
  /-
    🎉 no goals
  -/


theorem rootMultiplicity_X_sub_C_self [Nontrivial R] {x : R} :
    rootMultiplicity x (X - C x) = 1 :=
  pow_one (X - C x) ▸ rootMultiplicity_X_sub_C_pow x 1

-- Porting note: swapped instance argument order

theorem rootMultiplicity_X_sub_C [Nontrivial R] [DecidableEq R] {x y : R} :
    rootMultiplicity x (X - C y) = if x = y then 1 else 0 := by
  /-
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : Nontrivial R
    inst✝ : DecidableEq R
    x y : R
    ⊢ Eq (Polynomial.rootMultiplicity x (HSub.hSub Polynomial.X (Polynomial.C y))) …
  -/
  split_ifs with hxy
    /-
      case pos
      R : Type u
      inst✝² : CommRing R
      inst✝¹ : Nontrivial R
      inst✝ : DecidableEq R
      x y : R
      hxy : Eq x y
      ⊢ Eq (Polynomial.rootMultiplicity x (HSub.hSub Polynomial.X (Polynomial.C y))) 1
    -/
  · rw [hxy]
    /-
      case pos
      R : Type u
      inst✝² : CommRing R
      inst✝¹ : Nontrivial R
      inst✝ : DecidableEq R
      x y : R
      hxy : Eq x y
      ⊢ Eq (Polynomial.rootMultiplicity y (HSub.hSub Polynomial.X (Polynomial.C y))) 1
    -/
    exact rootMultiplicity_X_sub_C_self
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u
    inst✝² : CommRing R
    inst✝¹ : Nontrivial R
    inst✝ : DecidableEq R
    x y : R
    hxy : Not (Eq x y)
    ⊢ Eq (Polynomial.rootMultiplicity x (HSub.hSub Polynomial.X (Polynomial.C y))) 0
  -/
  exact rootMultiplicity_eq_zero (mt root_X_sub_C.mp (Ne.symm hxy))
  /-
    🎉 no goals
  -/


theorem rootMultiplicity_mul' {p q : R[X]} {x : R}
    (hpq : (p /ₘ (X - C x) ^ p.rootMultiplicity x).eval x *
      (q /ₘ (X - C x) ^ q.rootMultiplicity x).eval x ≠ 0) :
    rootMultiplicity x (p * q) = rootMultiplicity x p + rootMultiplicity x q := by
  /-
    R : Type u
    inst✝ : CommRing R
    p q : Polynomial R
    x : R
    hpq : Ne (HMul.hMul (Polynomial.eval x (p.divByMonic (HPow.hPow (HSub.hSub Pol …
    ⊢ Eq (Polynomial.rootMultiplicity x (HMul.hMul p q)) (HAdd.hAdd (Polynomial.ro …
  -/
  simp_rw [eval_divByMonic_eq_trailingCoeff_comp] at hpq
  /-
    R : Type u
    inst✝ : CommRing R
    p q : Polynomial R
    x : R
    hpq : Ne (HMul.hMul (p.comp (HAdd.hAdd Polynomial.X (Polynomial.C x))).trailin …
    ⊢ Eq (Polynomial.rootMultiplicity x (HMul.hMul p q)) (HAdd.hAdd (Polynomial.ro …
  -/
  simp_rw [rootMultiplicity_eq_natTrailingDegree, mul_comp, natTrailingDegree_mul' hpq]
  /-
    🎉 no goals
  -/


theorem Monic.neg_one_pow_natDegree_mul_comp_neg_X {p : R[X]} (hp : p.Monic) :
    ((-1) ^ p.natDegree * p.comp (-X)).Monic := by
  /-
    R : Type u
    inst✝ : CommRing R
    p : Polynomial R
    hp : p.Monic
    ⊢ (HMul.hMul (HPow.hPow (-1) p.natDegree) (p.comp (Neg.neg Polynomial.X))).Monic
  -/
  simp only [Monic]
  calc
    ((-1) ^ p.natDegree * p.comp (-X)).leadingCoeff =
        (p.comp (-X) * C ((-1) ^ p.natDegree)).leadingCoeff := by
      simp [mul_comm]
    _ = 1 := by
      apply monic_mul_C_of_leadingCoeff_mul_eq_one
      simp [← pow_add, hp]


theorem degree_eq_degree_of_associated (h : Associated p q) : degree p = degree q := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    p q : Polynomial R
    h : Associated p q
    ⊢ Eq p.degree q.degree
  -/
  let ⟨u, hu⟩ := h
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    p q : Polynomial R
    h : Associated p q
    u : Units (Polynomial R)
    hu : Eq (HMul.hMul p ↑u) q
    ⊢ Eq p.degree q.degree
  -/
  simp [hu.symm]
  /-
    🎉 no goals
  -/


theorem prime_X_sub_C (r : R) : Prime (X - C r) :=
  ⟨X_sub_C_ne_zero r, not_isUnit_X_sub_C r, fun _ _ => by
    /-
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      r : R
      x✝¹ x✝ : Polynomial R
      ⊢ Dvd.dvd (HSub.hSub Polynomial.X (Polynomial.C r)) (HMul.hMul x✝¹ x✝) → Or (D …
    -/
    simp_rw [dvd_iff_isRoot, IsRoot.def, eval_mul, mul_eq_zero]
    /-
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      r : R
      x✝¹ x✝ : Polynomial R
      ⊢ Or (Eq (Polynomial.eval r x✝¹) 0) (Eq (Polynomial.eval r x✝) 0) → Or (Eq (Po …
    -/
    exact id⟩
    /-
      🎉 no goals
    -/


theorem prime_X : Prime (X : R[X]) := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    ⊢ Prime Polynomial.X
  -/
  convert prime_X_sub_C (0 : R)
  /-
    case h.e'_3
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    ⊢ Eq Polynomial.X (HSub.hSub Polynomial.X (Polynomial.C 0))
  -/
  simp
  /-
    🎉 no goals
  -/


theorem Monic.prime_of_degree_eq_one (hp1 : degree p = 1) (hm : Monic p) : Prime p :=
                                      /-
                                        R : Type u
                                        inst✝¹ : CommRing R
                                        inst✝ : IsDomain R
                                        p : Polynomial R
                                        hp1 : Eq p.degree 1
                                        hm : p.Monic
                                        ⊢ Eq p (HSub.hSub Polynomial.X (Polynomial.C (Neg.neg (p.coeff 0))))
                                      -/
  have : p = X - C (-p.coeff 0) := by simpa [hm.leadingCoeff] using eq_X_add_C_of_degree_eq_one hp1
                                      /-
                                        🎉 no goals
                                      -/
  this.symm ▸ prime_X_sub_C _


theorem irreducible_X_sub_C (r : R) : Irreducible (X - C r) :=
  (prime_X_sub_C r).irreducible


theorem irreducible_X : Irreducible (X : R[X]) :=
  Prime.irreducible prime_X


theorem Monic.irreducible_of_degree_eq_one (hp1 : degree p = 1) (hm : Monic p) : Irreducible p :=
  (hm.prime_of_degree_eq_one hp1).irreducible


lemma aeval_ne_zero_of_isCoprime {R} [CommSemiring R] [Nontrivial S] [Semiring S] [Algebra R S]
    {p q : R[X]} (h : IsCoprime p q) (s : S) : aeval s p ≠ 0 ∨ aeval s q ≠ 0 := by
  /-
    S : Type v
    R : Type u_1
    inst✝³ : CommSemiring R
    inst✝² : Nontrivial S
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    p q : Polynomial R
    h : IsCoprime p q
    s : S
    ⊢ Or (Ne ((Polynomial.aeval s) p) 0) (Ne ((Polynomial.aeval s) q) 0)
  -/
  by_contra! hpq
  /-
    S : Type v
    R : Type u_1
    inst✝³ : CommSemiring R
    inst✝² : Nontrivial S
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    p q : Polynomial R
    h : IsCoprime p q
    s : S
    hpq : And (Eq ((Polynomial.aeval s) p) 0) (Eq ((Polynomial.aeval s) q) 0)
    ⊢ False
  -/
  rcases h with ⟨_, _, h⟩
  /-
    case intro.intro
    S : Type v
    R : Type u_1
    inst✝³ : CommSemiring R
    inst✝² : Nontrivial S
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    p q : Polynomial R
    s : S
    hpq : And (Eq ((Polynomial.aeval s) p) 0) (Eq ((Polynomial.aeval s) q) 0)
    w✝¹ w✝ : Polynomial R
    h : Eq (HAdd.hAdd (HMul.hMul w✝¹ p) (HMul.hMul w✝ q)) 1
    ⊢ False
  -/
  apply_fun aeval s at h
  /-
    case intro.intro
    S : Type v
    R : Type u_1
    inst✝³ : CommSemiring R
    inst✝² : Nontrivial S
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    p q : Polynomial R
    s : S
    hpq : And (Eq ((Polynomial.aeval s) p) 0) (Eq ((Polynomial.aeval s) q) 0)
    w✝¹ w✝ : Polynomial R
    h : Eq ((Polynomial.aeval s) (HAdd.hAdd (HMul.hMul w✝¹ p) (HMul.hMul w✝ q))) ( …
    ⊢ False
  -/
  simp only [map_add, map_mul, map_one, hpq.left, hpq.right, mul_zero, add_zero, zero_ne_one] at h
  /-
    🎉 no goals
  -/


theorem isCoprime_X_sub_C_of_isUnit_sub {R} [CommRing R] {a b : R} (h : IsUnit (a - b)) :
    IsCoprime (X - C a) (X - C b) :=
  ⟨-C h.unit⁻¹.val, C h.unit⁻¹.val, by
    /-
      R : Type u_1
      inst✝ : CommRing R
      a b : R
      h : IsUnit (HSub.hSub a b)
      ⊢ Eq (HAdd.hAdd (HMul.hMul (Neg.neg (Polynomial.C ↑(Inv.inv h.unit))) (HSub.hS …
    -/
    rw [neg_mul_comm, ← left_distrib, neg_add_eq_sub, sub_sub_sub_cancel_left, ← C_sub, ← C_mul]
    /-
      R : Type u_1
      inst✝ : CommRing R
      a b : R
      h : IsUnit (HSub.hSub a b)
      ⊢ Eq (Polynomial.C (HMul.hMul (↑(Inv.inv h.unit)) (HSub.hSub a b))) 1
    -/
    rw [← C_1]
    /-
      R : Type u_1
      inst✝ : CommRing R
      a b : R
      h : IsUnit (HSub.hSub a b)
      ⊢ Eq (Polynomial.C (HMul.hMul (↑(Inv.inv h.unit)) (HSub.hSub a b))) (Polynomia …
    -/
    congr
    /-
      case h.e_6.h
      R : Type u_1
      inst✝ : CommRing R
      a b : R
      h : IsUnit (HSub.hSub a b)
      ⊢ Eq (HMul.hMul (↑(Inv.inv h.unit)) (HSub.hSub a b)) 1
    -/
    exact h.val_inv_mul⟩
    /-
      🎉 no goals
    -/


theorem pairwise_coprime_X_sub_C {K} [Field K] {I : Type v} {s : I → K} (H : Function.Injective s) :
    Pairwise (IsCoprime on fun i : I => X - C (s i)) := fun _ _ hij =>
  isCoprime_X_sub_C_of_isUnit_sub (sub_ne_zero_of_ne <| H.ne hij).isUnit


theorem rootMultiplicity_mul {p q : R[X]} {x : R} (hpq : p * q ≠ 0) :
    rootMultiplicity x (p * q) = rootMultiplicity x p + rootMultiplicity x q := by
  classical
  have hp : p ≠ 0 := left_ne_zero_of_mul hpq
  have hq : q ≠ 0 := right_ne_zero_of_mul hpq
  rw [rootMultiplicity_eq_multiplicity (p * q), if_neg hpq, rootMultiplicity_eq_multiplicity p,
    if_neg hp, rootMultiplicity_eq_multiplicity q, if_neg hq,
    multiplicity_mul (prime_X_sub_C x) (finiteMultiplicity_X_sub_C _ hpq)]


open Multiset in
set_option linter.unusedVariables false in
theorem exists_multiset_roots [DecidableEq R] :
    ∀ {p : R[X]} (_ : p ≠ 0), ∃ s : Multiset R,
      (Multiset.card s : WithBot ℕ) ≤ degree p ∧ ∀ a, s.count a = rootMultiplicity a p
  | p, hp =>
    haveI := Classical.propDecidable (∃ x, IsRoot p x)
    if h : ∃ x, IsRoot p x then
      let ⟨x, hx⟩ := h
      have hpd : 0 < degree p := degree_pos_of_root hp hx
      have hd0 : p /ₘ (X - C x) ≠ 0 := fun h => by
        /-
          R : Type u
          inst✝² : CommRing R
          inst✝¹ : IsDomain R
          inst✝ : DecidableEq R
          p : Polynomial R
          hp : Ne p 0
          this : Decidable (Exists fun x => p.IsRoot x)
          h✝ : Exists fun x => p.IsRoot x
          x : R
          hx : p.IsRoot x
          hpd : LT.lt 0 p.degree
          h : Eq (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C x))) 0
          ⊢ False
        -/
        rw [← mul_divByMonic_eq_iff_isRoot.2 hx, h, mul_zero] at hp; exact hp rfl
                                                                     /-
                                                                       🎉 no goals
                                                                     -/
      #adaptation_note
      /--
      Since https://github.com/leanprover/lean4/pull/5338, this is considered unused,
      because it is only used in the decreasing_by clause.
      -/
      have wf : degree (p /ₘ (X - C x)) < degree p :=
                                                                                  /-
                                                                                    R : Type u
                                                                                    inst✝² : CommRing R
                                                                                    inst✝¹ : IsDomain R
                                                                                    inst✝ : DecidableEq R
                                                                                    p : Polynomial R
                                                                                    hp : Ne p 0
                                                                                    this : Decidable (Exists fun x => p.IsRoot x)
                                                                                    h : Exists fun x => p.IsRoot x
                                                                                    x : R
                                                                                    hx : p.IsRoot x
                                                                                    hpd : LT.lt 0 p.degree
                                                                                    hd0 : Ne (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C x))) 0
                                                                                    ⊢ LT.lt 0 1
                                                                                  -/
        degree_divByMonic_lt _ (monic_X_sub_C x) hp ((degree_X_sub_C x).symm ▸ by decide)
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/
      let ⟨t, htd, htr⟩ := @exists_multiset_roots _ (p /ₘ (X - C x)) hd0
      have hdeg : degree (X - C x) ≤ degree p := by
        /-
          R : Type u
          inst✝² : CommRing R
          inst✝¹ : IsDomain R
          inst✝ : DecidableEq R
          p : Polynomial R
          hp : Ne p 0
          this : Decidable (Exists fun x => p.IsRoot x)
          h : Exists fun x => p.IsRoot x
          x : R
          hx : p.IsRoot x
          hpd : LT.lt 0 p.degree
          hd0 : Ne (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C x))) 0
          wf : LT.lt (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C x))).degree p.d …
          t : Multiset R
          htd : LE.le (↑t.card) (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C x))) …
          htr : ∀ (a : R), Eq (Multiset.count a t) (Polynomial.rootMultiplicity a (p.div …
          ⊢ LE.le (HSub.hSub Polynomial.X (Polynomial.C x)).degree p.degree
        -/
        rw [degree_X_sub_C, degree_eq_natDegree hp]
        /-
          R : Type u
          inst✝² : CommRing R
          inst✝¹ : IsDomain R
          inst✝ : DecidableEq R
          p : Polynomial R
          hp : Ne p 0
          this : Decidable (Exists fun x => p.IsRoot x)
          h : Exists fun x => p.IsRoot x
          x : R
          hx : p.IsRoot x
          hpd : LT.lt 0 p.degree
          hd0 : Ne (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C x))) 0
          wf : LT.lt (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C x))).degree p.d …
          t : Multiset R
          htd : LE.le (↑t.card) (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C x))) …
          htr : ∀ (a : R), Eq (Multiset.count a t) (Polynomial.rootMultiplicity a (p.div …
          ⊢ LE.le 1 ↑p.natDegree
        -/
        rw [degree_eq_natDegree hp] at hpd
        /-
          R : Type u
          inst✝² : CommRing R
          inst✝¹ : IsDomain R
          inst✝ : DecidableEq R
          p : Polynomial R
          hp : Ne p 0
          this : Decidable (Exists fun x => p.IsRoot x)
          h : Exists fun x => p.IsRoot x
          x : R
          hx : p.IsRoot x
          hpd : LT.lt 0 ↑p.natDegree
          hd0 : Ne (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C x))) 0
          wf : LT.lt (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C x))).degree p.d …
          t : Multiset R
          htd : LE.le (↑t.card) (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C x))) …
          htr : ∀ (a : R), Eq (Multiset.count a t) (Polynomial.rootMultiplicity a (p.div …
          ⊢ LE.le 1 ↑p.natDegree
        -/
        exact WithBot.coe_le_coe.2 (WithBot.coe_lt_coe.1 hpd)
        /-
          🎉 no goals
        -/
      have hdiv0 : p /ₘ (X - C x) ≠ 0 :=
        mt (divByMonic_eq_zero_iff (monic_X_sub_C x)).1 <| not_lt.2 hdeg
      ⟨x ::ₘ t,
        calc
          (card (x ::ₘ t) : WithBot ℕ) = Multiset.card t + 1 := by
            /-
              R : Type u
              inst✝² : CommRing R
              inst✝¹ : IsDomain R
              inst✝ : DecidableEq R
              p : Polynomial R
              hp : Ne p 0
              this : Decidable (Exists fun x => p.IsRoot x)
              h : Exists fun x => p.IsRoot x
              x : R
              hx : p.IsRoot x
              hpd : LT.lt 0 p.degree
              hd0 : Ne (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C x))) 0
              wf : LT.lt (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C x))).degree p.d …
              t : Multiset R
              htd : LE.le (↑t.card) (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C x))) …
              htr : ∀ (a : R), Eq (Multiset.count a t) (Polynomial.rootMultiplicity a (p.div …
              hdeg : LE.le (HSub.hSub Polynomial.X (Polynomial.C x)).degree p.degree
              hdiv0 : Ne (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C x))) 0
              ⊢ Eq (↑(Multiset.cons x t).card) (HAdd.hAdd (↑t.card) 1)
            -/
            congr
            /-
              case e_a
              R : Type u
              inst✝² : CommRing R
              inst✝¹ : IsDomain R
              inst✝ : DecidableEq R
              p : Polynomial R
              hp : Ne p 0
              this : Decidable (Exists fun x => p.IsRoot x)
              h : Exists fun x => p.IsRoot x
              x : R
              hx : p.IsRoot x
              hpd : LT.lt 0 p.degree
              hd0 : Ne (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C x))) 0
              wf : LT.lt (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C x))).degree p.d …
              t : Multiset R
              htd : LE.le (↑t.card) (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C x))) …
              htr : ∀ (a : R), Eq (Multiset.count a t) (Polynomial.rootMultiplicity a (p.div …
              hdeg : LE.le (HSub.hSub Polynomial.X (Polynomial.C x)).degree p.degree
              hdiv0 : Ne (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C x))) 0
              ⊢ Eq (Multiset.cons x t).card (Add.add (↑t.card) 1)
            -/
            exact mod_cast Multiset.card_cons _ _
            /-
              🎉 no goals
            -/
          _ ≤ degree p := by
            /-
              R : Type u
              inst✝² : CommRing R
              inst✝¹ : IsDomain R
              inst✝ : DecidableEq R
              p : Polynomial R
              hp : Ne p 0
              this : Decidable (Exists fun x => p.IsRoot x)
              h : Exists fun x => p.IsRoot x
              x : R
              hx : p.IsRoot x
              hpd : LT.lt 0 p.degree
              hd0 : Ne (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C x))) 0
              wf : LT.lt (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C x))).degree p.d …
              t : Multiset R
              htd : LE.le (↑t.card) (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C x))) …
              htr : ∀ (a : R), Eq (Multiset.count a t) (Polynomial.rootMultiplicity a (p.div …
              hdeg : LE.le (HSub.hSub Polynomial.X (Polynomial.C x)).degree p.degree
              hdiv0 : Ne (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C x))) 0
              ⊢ LE.le (HAdd.hAdd (↑t.card) 1) p.degree
            -/
            rw [← degree_add_divByMonic (monic_X_sub_C x) hdeg, degree_X_sub_C, add_comm]
            /-
              R : Type u
              inst✝² : CommRing R
              inst✝¹ : IsDomain R
              inst✝ : DecidableEq R
              p : Polynomial R
              hp : Ne p 0
              this : Decidable (Exists fun x => p.IsRoot x)
              h : Exists fun x => p.IsRoot x
              x : R
              hx : p.IsRoot x
              hpd : LT.lt 0 p.degree
              hd0 : Ne (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C x))) 0
              wf : LT.lt (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C x))).degree p.d …
              t : Multiset R
              htd : LE.le (↑t.card) (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C x))) …
              htr : ∀ (a : R), Eq (Multiset.count a t) (Polynomial.rootMultiplicity a (p.div …
              hdeg : LE.le (HSub.hSub Polynomial.X (Polynomial.C x)).degree p.degree
              hdiv0 : Ne (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C x))) 0
              ⊢ LE.le (HAdd.hAdd 1 ↑t.card) (HAdd.hAdd 1 (p.divByMonic (HSub.hSub Polynomial …
            -/
            exact add_le_add (le_refl (1 : WithBot ℕ)) htd,
            /-
              🎉 no goals
            -/
        by
          /-
            R : Type u
            inst✝² : CommRing R
            inst✝¹ : IsDomain R
            inst✝ : DecidableEq R
            p : Polynomial R
            hp : Ne p 0
            this : Decidable (Exists fun x => p.IsRoot x)
            h : Exists fun x => p.IsRoot x
            x : R
            hx : p.IsRoot x
            hpd : LT.lt 0 p.degree
            hd0 : Ne (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C x))) 0
            wf : LT.lt (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C x))).degree p.d …
            t : Multiset R
            htd : LE.le (↑t.card) (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C x))) …
            htr : ∀ (a : R), Eq (Multiset.count a t) (Polynomial.rootMultiplicity a (p.div …
            hdeg : LE.le (HSub.hSub Polynomial.X (Polynomial.C x)).degree p.degree
            hdiv0 : Ne (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C x))) 0
            ⊢ ∀ (a : R), Eq (Multiset.count a (Multiset.cons x t)) (Polynomial.rootMultipl …
          -/
          intro a
          /-
            R : Type u
            inst✝² : CommRing R
            inst✝¹ : IsDomain R
            inst✝ : DecidableEq R
            p : Polynomial R
            hp : Ne p 0
            this : Decidable (Exists fun x => p.IsRoot x)
            h : Exists fun x => p.IsRoot x
            x : R
            hx : p.IsRoot x
            hpd : LT.lt 0 p.degree
            hd0 : Ne (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C x))) 0
            wf : LT.lt (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C x))).degree p.d …
            t : Multiset R
            htd : LE.le (↑t.card) (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C x))) …
            htr : ∀ (a : R), Eq (Multiset.count a t) (Polynomial.rootMultiplicity a (p.div …
            hdeg : LE.le (HSub.hSub Polynomial.X (Polynomial.C x)).degree p.degree
            hdiv0 : Ne (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C x))) 0
            a : R
            ⊢ Eq (Multiset.count a (Multiset.cons x t)) (Polynomial.rootMultiplicity a p)
          -/
          conv_rhs => rw [← mul_divByMonic_eq_iff_isRoot.mpr hx]
          rw [rootMultiplicity_mul (mul_ne_zero (X_sub_C_ne_zero x) hdiv0),
            rootMultiplicity_X_sub_C, ← htr a]
          /-
            R : Type u
            inst✝² : CommRing R
            inst✝¹ : IsDomain R
            inst✝ : DecidableEq R
            p : Polynomial R
            hp : Ne p 0
            this : Decidable (Exists fun x => p.IsRoot x)
            h : Exists fun x => p.IsRoot x
            x : R
            hx : p.IsRoot x
            hpd : LT.lt 0 p.degree
            hd0 : Ne (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C x))) 0
            wf : LT.lt (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C x))).degree p.d …
            t : Multiset R
            htd : LE.le (↑t.card) (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C x))) …
            htr : ∀ (a : R), Eq (Multiset.count a t) (Polynomial.rootMultiplicity a (p.div …
            hdeg : LE.le (HSub.hSub Polynomial.X (Polynomial.C x)).degree p.degree
            hdiv0 : Ne (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C x))) 0
            a : R
            ⊢ Eq (Multiset.count a (Multiset.cons x t)) (HAdd.hAdd (ite (Eq a x) 1 0) (Mul …
          -/
          split_ifs with ha
            /-
              case pos
              R : Type u
              inst✝² : CommRing R
              inst✝¹ : IsDomain R
              inst✝ : DecidableEq R
              p : Polynomial R
              hp : Ne p 0
              this : Decidable (Exists fun x => p.IsRoot x)
              h : Exists fun x => p.IsRoot x
              x : R
              hx : p.IsRoot x
              hpd : LT.lt 0 p.degree
              hd0 : Ne (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C x))) 0
              wf : LT.lt (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C x))).degree p.d …
              t : Multiset R
              htd : LE.le (↑t.card) (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C x))) …
              htr : ∀ (a : R), Eq (Multiset.count a t) (Polynomial.rootMultiplicity a (p.div …
              hdeg : LE.le (HSub.hSub Polynomial.X (Polynomial.C x)).degree p.degree
              hdiv0 : Ne (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C x))) 0
              a : R
              ha : Eq a x
              ⊢ Eq (Multiset.count a (Multiset.cons x t)) (HAdd.hAdd 1 (Multiset.count a t))
            -/
          · rw [ha, count_cons_self, add_comm]
            /-
              🎉 no goals
            -/
            /-
              case neg
              R : Type u
              inst✝² : CommRing R
              inst✝¹ : IsDomain R
              inst✝ : DecidableEq R
              p : Polynomial R
              hp : Ne p 0
              this : Decidable (Exists fun x => p.IsRoot x)
              h : Exists fun x => p.IsRoot x
              x : R
              hx : p.IsRoot x
              hpd : LT.lt 0 p.degree
              hd0 : Ne (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C x))) 0
              wf : LT.lt (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C x))).degree p.d …
              t : Multiset R
              htd : LE.le (↑t.card) (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C x))) …
              htr : ∀ (a : R), Eq (Multiset.count a t) (Polynomial.rootMultiplicity a (p.div …
              hdeg : LE.le (HSub.hSub Polynomial.X (Polynomial.C x)).degree p.degree
              hdiv0 : Ne (p.divByMonic (HSub.hSub Polynomial.X (Polynomial.C x))) 0
              a : R
              ha : Not (Eq a x)
              ⊢ Eq (Multiset.count a (Multiset.cons x t)) (HAdd.hAdd 0 (Multiset.count a t))
            -/
          · rw [count_cons_of_ne ha, zero_add]⟩
            /-
              🎉 no goals
            -/
    else
      ⟨0, (degree_eq_natDegree hp).symm ▸ WithBot.coe_le_coe.2 (Nat.zero_le _), by
        /-
          R : Type u
          inst✝² : CommRing R
          inst✝¹ : IsDomain R
          inst✝ : DecidableEq R
          p : Polynomial R
          hp : Ne p 0
          this : Decidable (Exists fun x => p.IsRoot x)
          h : Not (Exists fun x => p.IsRoot x)
          ⊢ ∀ (a : R), Eq (Multiset.count a 0) (Polynomial.rootMultiplicity a p)
        -/
        intro a
        /-
          R : Type u
          inst✝² : CommRing R
          inst✝¹ : IsDomain R
          inst✝ : DecidableEq R
          p : Polynomial R
          hp : Ne p 0
          this : Decidable (Exists fun x => p.IsRoot x)
          h : Not (Exists fun x => p.IsRoot x)
          a : R
          ⊢ Eq (Multiset.count a 0) (Polynomial.rootMultiplicity a p)
        -/
        rw [count_zero, rootMultiplicity_eq_zero (not_exists.mp h a)]⟩
        /-
          🎉 no goals
        -/
termination_by p => natDegree p
decreasing_by {
  simp_wf
  apply (Nat.cast_lt (α := WithBot ℕ)).mp
  simp only [degree_eq_natDegree hp, degree_eq_natDegree hd0] at wf
  assumption}


