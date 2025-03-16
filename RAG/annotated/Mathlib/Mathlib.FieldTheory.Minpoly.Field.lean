/-- If an element `x` is a root of a nonzero polynomial `p`, then the degree of `p` is at least the
degree of the minimal polynomial of `x`. See also `minpoly.IsIntegrallyClosed.degree_le_of_ne_zero`
which relaxes the assumptions on `A` in exchange for stronger assumptions on `B`. -/
theorem degree_le_of_ne_zero {p : A[X]} (pnz : p ≠ 0) (hp : Polynomial.aeval x p = 0) :
    degree (minpoly A x) ≤ degree p :=
  calc
    degree (minpoly A x) ≤ degree (p * C (leadingCoeff p)⁻¹) :=
                                                   /-
                                                     A : Type u_1
                                                     B : Type u_2
                                                     inst✝² : Field A
                                                     inst✝¹ : Ring B
                                                     inst✝ : Algebra A B
                                                     x : B
                                                     p : Polynomial A
                                                     pnz : Ne p 0
                                                     hp : Eq ((Polynomial.aeval x) p) 0
                                                     ⊢ Eq ((Polynomial.aeval x) (HMul.hMul p (Polynomial.C (Inv.inv p.leadingCoeff) …
                                                   -/
      min A x (monic_mul_leadingCoeff_inv pnz) (by simp [hp])
                                                   /-
                                                     🎉 no goals
                                                   -/
    _ = degree p := degree_mul_leadingCoeff_inv p pnz


theorem ne_zero_of_finite (e : B) [FiniteDimensional A B] : minpoly A e ≠ 0 :=
  minpoly.ne_zero <| .of_finite A _


/-- The minimal polynomial of an element `x` is uniquely characterized by its defining property:
if there is another monic polynomial of minimal degree that has `x` as a root, then this polynomial
is equal to the minimal polynomial of `x`. See also `minpoly.IsIntegrallyClosed.Minpoly.unique`
which relaxes the assumptions on `A` in exchange for stronger assumptions on `B`. -/
theorem unique {p : A[X]} (pmonic : p.Monic) (hp : Polynomial.aeval x p = 0)
    (pmin : ∀ q : A[X], q.Monic → Polynomial.aeval x q = 0 → degree p ≤ degree q) :
    p = minpoly A x := by
  /-
    A : Type u_1
    B : Type u_2
    inst✝² : Field A
    inst✝¹ : Ring B
    inst✝ : Algebra A B
    x : B
    p : Polynomial A
    pmonic : p.Monic
    hp : Eq ((Polynomial.aeval x) p) 0
    pmin : ∀ (q : Polynomial A), q.Monic → Eq ((Polynomial.aeval x) q) 0 → LE.le p …
    ⊢ Eq p (minpoly A x)
  -/
  have hx : IsIntegral A x := ⟨p, pmonic, hp⟩
  /-
    A : Type u_1
    B : Type u_2
    inst✝² : Field A
    inst✝¹ : Ring B
    inst✝ : Algebra A B
    x : B
    p : Polynomial A
    pmonic : p.Monic
    hp : Eq ((Polynomial.aeval x) p) 0
    pmin : ∀ (q : Polynomial A), q.Monic → Eq ((Polynomial.aeval x) q) 0 → LE.le p …
    hx : IsIntegral A x
    ⊢ Eq p (minpoly A x)
  -/
  symm; apply eq_of_sub_eq_zero
  /-
    case h
    A : Type u_1
    B : Type u_2
    inst✝² : Field A
    inst✝¹ : Ring B
    inst✝ : Algebra A B
    x : B
    p : Polynomial A
    pmonic : p.Monic
    hp : Eq ((Polynomial.aeval x) p) 0
    pmin : ∀ (q : Polynomial A), q.Monic → Eq ((Polynomial.aeval x) q) 0 → LE.le p …
    hx : IsIntegral A x
    ⊢ Eq (HSub.hSub (minpoly A x) p) 0
  -/
  by_contra hnz
  /-
    case h
    A : Type u_1
    B : Type u_2
    inst✝² : Field A
    inst✝¹ : Ring B
    inst✝ : Algebra A B
    x : B
    p : Polynomial A
    pmonic : p.Monic
    hp : Eq ((Polynomial.aeval x) p) 0
    pmin : ∀ (q : Polynomial A), q.Monic → Eq ((Polynomial.aeval x) q) 0 → LE.le p …
    hx : IsIntegral A x
    hnz : Not (Eq (HSub.hSub (minpoly A x) p) 0)
    ⊢ False
  -/
  apply degree_le_of_ne_zero A x hnz (by simp [hp]) |>.not_lt
  /-
    case h
    A : Type u_1
    B : Type u_2
    inst✝² : Field A
    inst✝¹ : Ring B
    inst✝ : Algebra A B
    x : B
    p : Polynomial A
    pmonic : p.Monic
    hp : Eq ((Polynomial.aeval x) p) 0
    pmin : ∀ (q : Polynomial A), q.Monic → Eq ((Polynomial.aeval x) q) 0 → LE.le p …
    hx : IsIntegral A x
    hnz : Not (Eq (HSub.hSub (minpoly A x) p) 0)
    ⊢ LT.lt (HSub.hSub (minpoly A x) p).degree (minpoly A x).degree
  -/
  apply degree_sub_lt _ (minpoly.ne_zero hx)
    /-
      case h
      A : Type u_1
      B : Type u_2
      inst✝² : Field A
      inst✝¹ : Ring B
      inst✝ : Algebra A B
      x : B
      p : Polynomial A
      pmonic : p.Monic
      hp : Eq ((Polynomial.aeval x) p) 0
      pmin : ∀ (q : Polynomial A), q.Monic → Eq ((Polynomial.aeval x) q) 0 → LE.le p …
      hx : IsIntegral A x
      hnz : Not (Eq (HSub.hSub (minpoly A x) p) 0)
      ⊢ Eq (minpoly A x).leadingCoeff p.leadingCoeff
    -/
  · rw [(monic hx).leadingCoeff, pmonic.leadingCoeff]
    /-
      🎉 no goals
    -/
    /-
      A : Type u_1
      B : Type u_2
      inst✝² : Field A
      inst✝¹ : Ring B
      inst✝ : Algebra A B
      x : B
      p : Polynomial A
      pmonic : p.Monic
      hp : Eq ((Polynomial.aeval x) p) 0
      pmin : ∀ (q : Polynomial A), q.Monic → Eq ((Polynomial.aeval x) q) 0 → LE.le p …
      hx : IsIntegral A x
      hnz : Not (Eq (HSub.hSub (minpoly A x) p) 0)
      ⊢ Eq (minpoly A x).degree p.degree
    -/
  · exact le_antisymm (min A x pmonic hp) (pmin (minpoly A x) (monic hx) (aeval A x))
    /-
      🎉 no goals
    -/


/-- If an element `x` is a root of a polynomial `p`, then the minimal polynomial of `x` divides `p`.
See also `minpoly.isIntegrallyClosed_dvd` which relaxes the assumptions on `A` in exchange for
stronger assumptions on `B`. -/
theorem dvd {p : A[X]} (hp : Polynomial.aeval x p = 0) : minpoly A x ∣ p := by
  /-
    A : Type u_1
    B : Type u_2
    inst✝² : Field A
    inst✝¹ : Ring B
    inst✝ : Algebra A B
    x : B
    p : Polynomial A
    hp : Eq ((Polynomial.aeval x) p) 0
    ⊢ Dvd.dvd (minpoly A x) p
  -/
  by_cases hp0 : p = 0
    /-
      case pos
      A : Type u_1
      B : Type u_2
      inst✝² : Field A
      inst✝¹ : Ring B
      inst✝ : Algebra A B
      x : B
      p : Polynomial A
      hp : Eq ((Polynomial.aeval x) p) 0
      hp0 : Eq p 0
      ⊢ Dvd.dvd (minpoly A x) p
    -/
  · simp only [hp0, dvd_zero]
    /-
      🎉 no goals
    -/
  /-
    case neg
    A : Type u_1
    B : Type u_2
    inst✝² : Field A
    inst✝¹ : Ring B
    inst✝ : Algebra A B
    x : B
    p : Polynomial A
    hp : Eq ((Polynomial.aeval x) p) 0
    hp0 : Not (Eq p 0)
    ⊢ Dvd.dvd (minpoly A x) p
  -/
  have hx : IsIntegral A x := IsAlgebraic.isIntegral ⟨p, hp0, hp⟩
  /-
    case neg
    A : Type u_1
    B : Type u_2
    inst✝² : Field A
    inst✝¹ : Ring B
    inst✝ : Algebra A B
    x : B
    p : Polynomial A
    hp : Eq ((Polynomial.aeval x) p) 0
    hp0 : Not (Eq p 0)
    hx : IsIntegral A x
    ⊢ Dvd.dvd (minpoly A x) p
  -/
  rw [← modByMonic_eq_zero_iff_dvd (monic hx)]
  /-
    case neg
    A : Type u_1
    B : Type u_2
    inst✝² : Field A
    inst✝¹ : Ring B
    inst✝ : Algebra A B
    x : B
    p : Polynomial A
    hp : Eq ((Polynomial.aeval x) p) 0
    hp0 : Not (Eq p 0)
    hx : IsIntegral A x
    ⊢ Eq (p.modByMonic (minpoly A x)) 0
  -/
  by_contra hnz
  apply degree_le_of_ne_zero A x hnz
    ((aeval_modByMonic_eq_self_of_root (monic hx) (aeval _ _)).trans hp) |>.not_lt
  /-
    case neg
    A : Type u_1
    B : Type u_2
    inst✝² : Field A
    inst✝¹ : Ring B
    inst✝ : Algebra A B
    x : B
    p : Polynomial A
    hp : Eq ((Polynomial.aeval x) p) 0
    hp0 : Not (Eq p 0)
    hx : IsIntegral A x
    hnz : Not (Eq (p.modByMonic (minpoly A x)) 0)
    ⊢ LT.lt (p.modByMonic (minpoly A x)).degree (minpoly A x).degree
  -/
  exact degree_modByMonic_lt _ (monic hx)
  /-
    🎉 no goals
  -/


variable {A x} in
lemma dvd_iff {p : A[X]} : minpoly A x ∣ p ↔ Polynomial.aeval x p = 0 :=
                    /-
                      A : Type u_1
                      B : Type u_2
                      inst✝² : Field A
                      inst✝¹ : Ring B
                      inst✝ : Algebra A B
                      x : B
                      p : Polynomial A
                      x✝ : Dvd.dvd (minpoly A x) p
                      q : Polynomial A
                      hq : Eq p (HMul.hMul (minpoly A x) q)
                      ⊢ Eq ((Polynomial.aeval x) p) 0
                    -/
  ⟨fun ⟨q, hq⟩ ↦ by rw [hq, map_mul, aeval, zero_mul], minpoly.dvd A x⟩
                    /-
                      🎉 no goals
                    -/


theorem isRadical [IsReduced B] : IsRadical (minpoly A x) := fun n p dvd ↦ by
  /-
    A : Type u_1
    B : Type u_2
    inst✝³ : Field A
    inst✝² : Ring B
    inst✝¹ : Algebra A B
    x : B
    inst✝ : IsReduced B
    n : Nat
    p : Polynomial A
    dvd : Dvd.dvd (minpoly A x) (HPow.hPow p n)
    ⊢ Dvd.dvd (minpoly A x) p
  -/
  rw [dvd_iff] at dvd ⊢; rw [map_pow] at dvd; exact IsReduced.eq_zero _ ⟨n, dvd⟩
                                              /-
                                                🎉 no goals
                                              -/


theorem dvd_map_of_isScalarTower (A K : Type*) {R : Type*} [CommRing A] [Field K] [Ring R]
    [Algebra A K] [Algebra A R] [Algebra K R] [IsScalarTower A K R] (x : R) :
    minpoly K x ∣ (minpoly A x).map (algebraMap A K) := by
  /-
    A : Type u_3
    K : Type u_4
    R : Type u_5
    inst✝⁶ : CommRing A
    inst✝⁵ : Field K
    inst✝⁴ : Ring R
    inst✝³ : Algebra A K
    inst✝² : Algebra A R
    inst✝¹ : Algebra K R
    inst✝ : IsScalarTower A K R
    x : R
    ⊢ Dvd.dvd (minpoly K x) (Polynomial.map (algebraMap A K) (minpoly A x))
  -/
  refine minpoly.dvd K x ?_
  /-
    A : Type u_3
    K : Type u_4
    R : Type u_5
    inst✝⁶ : CommRing A
    inst✝⁵ : Field K
    inst✝⁴ : Ring R
    inst✝³ : Algebra A K
    inst✝² : Algebra A R
    inst✝¹ : Algebra K R
    inst✝ : IsScalarTower A K R
    x : R
    ⊢ Eq ((Polynomial.aeval x) (Polynomial.map (algebraMap A K) (minpoly A x))) 0
  -/
  rw [aeval_map_algebraMap, minpoly.aeval]
  /-
    🎉 no goals
  -/


theorem dvd_map_of_isScalarTower' (R : Type*) {S : Type*} (K L : Type*) [CommRing R]
    [CommRing S] [Field K] [CommRing L] [Algebra R S] [Algebra R K] [Algebra S L] [Algebra K L]
    [Algebra R L] [IsScalarTower R K L] [IsScalarTower R S L] (s : S) :
    minpoly K (algebraMap S L s) ∣ map (algebraMap R K) (minpoly R s) := by
  /-
    R : Type u_3
    S : Type u_4
    K : Type u_5
    L : Type u_6
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : Field K
    inst✝⁷ : CommRing L
    inst✝⁶ : Algebra R S
    inst✝⁵ : Algebra R K
    inst✝⁴ : Algebra S L
    inst✝³ : Algebra K L
    inst✝² : Algebra R L
    inst✝¹ : IsScalarTower R K L
    inst✝ : IsScalarTower R S L
    s : S
    ⊢ Dvd.dvd (minpoly K ((algebraMap S L) s)) (Polynomial.map (algebraMap R K) (m …
  -/
  apply minpoly.dvd K (algebraMap S L s)
  /-
    R : Type u_3
    S : Type u_4
    K : Type u_5
    L : Type u_6
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : Field K
    inst✝⁷ : CommRing L
    inst✝⁶ : Algebra R S
    inst✝⁵ : Algebra R K
    inst✝⁴ : Algebra S L
    inst✝³ : Algebra K L
    inst✝² : Algebra R L
    inst✝¹ : IsScalarTower R K L
    inst✝ : IsScalarTower R S L
    s : S
    ⊢ Eq ((Polynomial.aeval ((algebraMap S L) s)) (Polynomial.map (algebraMap R K) …
  -/
  rw [← map_aeval_eq_aeval_map, minpoly.aeval, map_zero]
  /-
    case h
    R : Type u_3
    S : Type u_4
    K : Type u_5
    L : Type u_6
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : Field K
    inst✝⁷ : CommRing L
    inst✝⁶ : Algebra R S
    inst✝⁵ : Algebra R K
    inst✝⁴ : Algebra S L
    inst✝³ : Algebra K L
    inst✝² : Algebra R L
    inst✝¹ : IsScalarTower R K L
    inst✝ : IsScalarTower R S L
    s : S
    ⊢ Eq ((algebraMap K L).comp (algebraMap R K)) ((algebraMap S L).comp (algebraM …
  -/
  rw [← IsScalarTower.algebraMap_eq, ← IsScalarTower.algebraMap_eq]
  /-
    🎉 no goals
  -/


/-- If `y` is a conjugate of `x` over a field `K`, then it is a conjugate over a subring `R`. -/
theorem aeval_of_isScalarTower (R : Type*) {K T U : Type*} [CommRing R] [Field K] [CommRing T]
    [Algebra R K] [Algebra K T] [Algebra R T] [IsScalarTower R K T] [CommSemiring U] [Algebra K U]
    [Algebra R U] [IsScalarTower R K U] (x : T) (y : U)
    (hy : Polynomial.aeval y (minpoly K x) = 0) : Polynomial.aeval y (minpoly R x) = 0 :=
  aeval_map_algebraMap K y (minpoly R x) ▸
    eval₂_eq_zero_of_dvd_of_eval₂_eq_zero (algebraMap K U) y
      (minpoly.dvd_map_of_isScalarTower R K x) hy


/-- See also `minpoly.ker_eval` which relaxes the assumptions on `A` in exchange for
stronger assumptions on `B`. -/
@[simp]
lemma ker_aeval_eq_span_minpoly :
    RingHom.ker (Polynomial.aeval x) = A[X] ∙ minpoly A x := by
  /-
    A : Type u_1
    B : Type u_2
    inst✝² : Field A
    inst✝¹ : Ring B
    inst✝ : Algebra A B
    x : B
    ⊢ Eq (RingHom.ker (Polynomial.aeval x)) (Submodule.span (Polynomial A) (Single …
  -/
  ext p
  simp_rw [RingHom.mem_ker, ← minpoly.dvd_iff, Submodule.mem_span_singleton,
    dvd_iff_exists_eq_mul_left, smul_eq_mul, eq_comm (a := p)]


theorem eq_of_irreducible_of_monic [Nontrivial B] {p : A[X]} (hp1 : Irreducible p)
    (hp2 : Polynomial.aeval x p = 0) (hp3 : p.Monic) : p = minpoly A x :=
  let ⟨_, hq⟩ := dvd A x hp2
  eq_of_monic_of_associated hp3 (monic ⟨p, ⟨hp3, hp2⟩⟩) <|
    mul_one (minpoly A x) ▸ hq.symm ▸ Associated.mul_left _
      (associated_one_iff_isUnit.2 <| (hp1.isUnit_or_isUnit hq).resolve_left <| not_isUnit A x)


theorem eq_iff_aeval_eq_zero [Nontrivial B] {p : A[X]} (irr: Irreducible p) (monic: p.Monic) :
    p = minpoly A x ↔ Polynomial.aeval x p = 0 :=
  ⟨(· ▸ aeval A x), (eq_of_irreducible_of_monic irr · monic)⟩


theorem eq_iff_aeval_minpoly_eq_zero [IsDomain B] {C} [Ring C] [Algebra A C] [Nontrivial C]
    {b : B} (h : IsIntegral A b) {c : C} :
    minpoly A b = minpoly A c ↔ Polynomial.aeval c (minpoly A b) = 0 :=
  eq_iff_aeval_eq_zero (irreducible h) (monic h)


theorem eq_of_irreducible [Nontrivial B] {p : A[X]} (hp1 : Irreducible p)
    (hp2 : Polynomial.aeval x p = 0) : p * C p.leadingCoeff⁻¹ = minpoly A x := by
  /-
    A : Type u_1
    B : Type u_2
    inst✝³ : Field A
    inst✝² : Ring B
    inst✝¹ : Algebra A B
    x : B
    inst✝ : Nontrivial B
    p : Polynomial A
    hp1 : Irreducible p
    hp2 : Eq ((Polynomial.aeval x) p) 0
    ⊢ Eq (HMul.hMul p (Polynomial.C (Inv.inv p.leadingCoeff))) (minpoly A x)
  -/
  have : p.leadingCoeff ≠ 0 := leadingCoeff_ne_zero.mpr hp1.ne_zero
  /-
    A : Type u_1
    B : Type u_2
    inst✝³ : Field A
    inst✝² : Ring B
    inst✝¹ : Algebra A B
    x : B
    inst✝ : Nontrivial B
    p : Polynomial A
    hp1 : Irreducible p
    hp2 : Eq ((Polynomial.aeval x) p) 0
    this : Ne p.leadingCoeff 0
    ⊢ Eq (HMul.hMul p (Polynomial.C (Inv.inv p.leadingCoeff))) (minpoly A x)
  -/
  apply eq_of_irreducible_of_monic
  · exact Associated.irreducible ⟨⟨C p.leadingCoeff⁻¹, C p.leadingCoeff,
      by rwa [← C_mul, inv_mul_cancel₀, C_1], by rwa [← C_mul, mul_inv_cancel₀, C_1]⟩, rfl⟩ hp1
    /-
      case hp2
      A : Type u_1
      B : Type u_2
      inst✝³ : Field A
      inst✝² : Ring B
      inst✝¹ : Algebra A B
      x : B
      inst✝ : Nontrivial B
      p : Polynomial A
      hp1 : Irreducible p
      hp2 : Eq ((Polynomial.aeval x) p) 0
      this : Ne p.leadingCoeff 0
      ⊢ Eq ((Polynomial.aeval x) (HMul.hMul p (Polynomial.C (Inv.inv p.leadingCoeff) …
    -/
  · rw [aeval_mul, hp2, zero_mul]
    /-
      🎉 no goals
    -/
    /-
      case hp3
      A : Type u_1
      B : Type u_2
      inst✝³ : Field A
      inst✝² : Ring B
      inst✝¹ : Algebra A B
      x : B
      inst✝ : Nontrivial B
      p : Polynomial A
      hp1 : Irreducible p
      hp2 : Eq ((Polynomial.aeval x) p) 0
      this : Ne p.leadingCoeff 0
      ⊢ (HMul.hMul p (Polynomial.C (Inv.inv p.leadingCoeff))).Monic
    -/
  · rwa [Polynomial.Monic, leadingCoeff_mul, leadingCoeff_C, mul_inv_cancel₀]
    /-
      🎉 no goals
    -/


theorem add_algebraMap {B : Type*} [CommRing B] [Algebra A B] (x : B)
    (a : A) : minpoly A (x + algebraMap A B a) = (minpoly A x).comp (X - C a) := by
  /-
    A : Type u_1
    inst✝² : Field A
    B : Type u_3
    inst✝¹ : CommRing B
    inst✝ : Algebra A B
    x : B
    a : A
    ⊢ Eq (minpoly A (HAdd.hAdd x ((algebraMap A B) a))) ((minpoly A x).comp (HSub. …
  -/
  by_cases hx : IsIntegral A x
    /-
      case pos
      A : Type u_1
      inst✝² : Field A
      B : Type u_3
      inst✝¹ : CommRing B
      inst✝ : Algebra A B
      x : B
      a : A
      hx : IsIntegral A x
      ⊢ Eq (minpoly A (HAdd.hAdd x ((algebraMap A B) a))) ((minpoly A x).comp (HSub. …
    -/
  · refine (minpoly.unique _ _ ((minpoly.monic hx).comp_X_sub_C _) ?_ fun q qmo hq => ?_).symm
      /-
        case pos.refine_1
        A : Type u_1
        inst✝² : Field A
        B : Type u_3
        inst✝¹ : CommRing B
        inst✝ : Algebra A B
        x : B
        a : A
        hx : IsIntegral A x
        ⊢ Eq ((Polynomial.aeval (HAdd.hAdd x ((algebraMap A B) a))) ((minpoly A x).com …
      -/
    · simp [aeval_comp]
      /-
        🎉 no goals
      -/
      /-
        case pos.refine_2
        A : Type u_1
        inst✝² : Field A
        B : Type u_3
        inst✝¹ : CommRing B
        inst✝ : Algebra A B
        x : B
        a : A
        hx : IsIntegral A x
        q : Polynomial A
        qmo : q.Monic
        hq : Eq ((Polynomial.aeval (HAdd.hAdd x ((algebraMap A B) a))) q) 0
        ⊢ LE.le ((minpoly A x).comp (HSub.hSub Polynomial.X (Polynomial.C a))).degree  …
      -/
    · have : (Polynomial.aeval x) (q.comp (X + C a)) = 0 := by simpa [aeval_comp] using hq
      /-
        case pos.refine_2
        A : Type u_1
        inst✝² : Field A
        B : Type u_3
        inst✝¹ : CommRing B
        inst✝ : Algebra A B
        x : B
        a : A
        hx : IsIntegral A x
        q : Polynomial A
        qmo : q.Monic
        hq : Eq ((Polynomial.aeval (HAdd.hAdd x ((algebraMap A B) a))) q) 0
        this : Eq ((Polynomial.aeval x) (q.comp (HAdd.hAdd Polynomial.X (Polynomial.C  …
        ⊢ LE.le ((minpoly A x).comp (HSub.hSub Polynomial.X (Polynomial.C a))).degree  …
      -/
      have H := minpoly.min A x (qmo.comp_X_add_C _) this
      rw [degree_eq_natDegree qmo.ne_zero,
        degree_eq_natDegree ((minpoly.monic hx).comp_X_sub_C _).ne_zero, natDegree_comp,
        natDegree_X_sub_C, mul_one]
      rwa [degree_eq_natDegree (minpoly.ne_zero hx),
        degree_eq_natDegree (qmo.comp_X_add_C _).ne_zero, natDegree_comp,
        natDegree_X_add_C, mul_one] at H
    /-
      case neg
      A : Type u_1
      inst✝² : Field A
      B : Type u_3
      inst✝¹ : CommRing B
      inst✝ : Algebra A B
      x : B
      a : A
      hx : Not (IsIntegral A x)
      ⊢ Eq (minpoly A (HAdd.hAdd x ((algebraMap A B) a))) ((minpoly A x).comp (HSub. …
    -/
  · rw [minpoly.eq_zero hx, minpoly.eq_zero, zero_comp]
    /-
      case neg
      A : Type u_1
      inst✝² : Field A
      B : Type u_3
      inst✝¹ : CommRing B
      inst✝ : Algebra A B
      x : B
      a : A
      hx : Not (IsIntegral A x)
      ⊢ Not (IsIntegral A (HAdd.hAdd x ((algebraMap A B) a)))
    -/
    refine fun h ↦ hx ?_
    /-
      case neg
      A : Type u_1
      inst✝² : Field A
      B : Type u_3
      inst✝¹ : CommRing B
      inst✝ : Algebra A B
      x : B
      a : A
      hx : Not (IsIntegral A x)
      h : IsIntegral A (HAdd.hAdd x ((algebraMap A B) a))
      ⊢ IsIntegral A x
    -/
    simpa only [add_sub_cancel_right] using IsIntegral.sub h (isIntegral_algebraMap (x := a))
    /-
      🎉 no goals
    -/


theorem sub_algebraMap {B : Type*} [CommRing B] [Algebra A B] (x : B)
    (a : A) : minpoly A (x - algebraMap A B a) = (minpoly A x).comp (X + C a) := by
  /-
    A : Type u_1
    inst✝² : Field A
    B : Type u_3
    inst✝¹ : CommRing B
    inst✝ : Algebra A B
    x : B
    a : A
    ⊢ Eq (minpoly A (HSub.hSub x ((algebraMap A B) a))) ((minpoly A x).comp (HAdd. …
  -/
  simpa [sub_eq_add_neg] using add_algebraMap x (-a)
  /-
    🎉 no goals
  -/


theorem neg {B : Type*} [CommRing B] [Algebra A B] (x : B) :
    minpoly A (- x) = (-1) ^ (natDegree (minpoly A x)) * (minpoly A x).comp (- X) := by
  /-
    A : Type u_1
    inst✝² : Field A
    B : Type u_3
    inst✝¹ : CommRing B
    inst✝ : Algebra A B
    x : B
    ⊢ Eq (minpoly A (Neg.neg x)) (HMul.hMul (HPow.hPow (-1) (minpoly A x).natDegre …
  -/
  by_cases hx : IsIntegral A x
  · refine (minpoly.unique _ _ ((minpoly.monic hx).neg_one_pow_natDegree_mul_comp_neg_X)
        ?_ fun q qmo hq => ?_).symm
      /-
        case pos.refine_1
        A : Type u_1
        inst✝² : Field A
        B : Type u_3
        inst✝¹ : CommRing B
        inst✝ : Algebra A B
        x : B
        hx : IsIntegral A x
        ⊢ Eq ((Polynomial.aeval (Neg.neg x)) (HMul.hMul (HPow.hPow (-1) (minpoly A x). …
      -/
    · simp [aeval_comp]
      /-
        🎉 no goals
      -/
    · have : (Polynomial.aeval x) ((-1) ^ q.natDegree * q.comp (- X)) = 0 := by
        simpa [aeval_comp] using hq
      /-
        case pos.refine_2
        A : Type u_1
        inst✝² : Field A
        B : Type u_3
        inst✝¹ : CommRing B
        inst✝ : Algebra A B
        x : B
        hx : IsIntegral A x
        q : Polynomial A
        qmo : q.Monic
        hq : Eq ((Polynomial.aeval (Neg.neg x)) q) 0
        this : Eq ((Polynomial.aeval x) (HMul.hMul (HPow.hPow (-1) q.natDegree) (q.com …
        ⊢ LE.le (HMul.hMul (HPow.hPow (-1) (minpoly A x).natDegree) ((minpoly A x).com …
      -/
      have H := minpoly.min A x qmo.neg_one_pow_natDegree_mul_comp_neg_X this
      /-
        case pos.refine_2
        A : Type u_1
        inst✝² : Field A
        B : Type u_3
        inst✝¹ : CommRing B
        inst✝ : Algebra A B
        x : B
        hx : IsIntegral A x
        q : Polynomial A
        qmo : q.Monic
        hq : Eq ((Polynomial.aeval (Neg.neg x)) q) 0
        this : Eq ((Polynomial.aeval x) (HMul.hMul (HPow.hPow (-1) q.natDegree) (q.com …
        H : LE.le (minpoly A x).degree (HMul.hMul (HPow.hPow (-1) q.natDegree) (q.comp …
        ⊢ LE.le (HMul.hMul (HPow.hPow (-1) (minpoly A x).natDegree) ((minpoly A x).com …
      -/
      have n1 := ((minpoly.monic hx).neg_one_pow_natDegree_mul_comp_neg_X).ne_zero
      /-
        case pos.refine_2
        A : Type u_1
        inst✝² : Field A
        B : Type u_3
        inst✝¹ : CommRing B
        inst✝ : Algebra A B
        x : B
        hx : IsIntegral A x
        q : Polynomial A
        qmo : q.Monic
        hq : Eq ((Polynomial.aeval (Neg.neg x)) q) 0
        this : Eq ((Polynomial.aeval x) (HMul.hMul (HPow.hPow (-1) q.natDegree) (q.com …
        H : LE.le (minpoly A x).degree (HMul.hMul (HPow.hPow (-1) q.natDegree) (q.comp …
        n1 : Ne (HMul.hMul (HPow.hPow (-1) (minpoly A x).natDegree) ((minpoly A x).com …
        ⊢ LE.le (HMul.hMul (HPow.hPow (-1) (minpoly A x).natDegree) ((minpoly A x).com …
      -/
      have n2 := qmo.neg_one_pow_natDegree_mul_comp_neg_X.ne_zero
      rw [degree_eq_natDegree qmo.ne_zero,
        degree_eq_natDegree n1, natDegree_mul (by simp) (right_ne_zero_of_mul n1), natDegree_comp]
      rw [degree_eq_natDegree (minpoly.ne_zero hx),
        degree_eq_natDegree qmo.neg_one_pow_natDegree_mul_comp_neg_X.ne_zero,
        natDegree_mul (by simp) (right_ne_zero_of_mul n2), natDegree_comp] at H
      /-
        case pos.refine_2
        A : Type u_1
        inst✝² : Field A
        B : Type u_3
        inst✝¹ : CommRing B
        inst✝ : Algebra A B
        x : B
        hx : IsIntegral A x
        q : Polynomial A
        qmo : q.Monic
        hq : Eq ((Polynomial.aeval (Neg.neg x)) q) 0
        this : Eq ((Polynomial.aeval x) (HMul.hMul (HPow.hPow (-1) q.natDegree) (q.com …
        H : LE.le ↑(minpoly A x).natDegree ↑(HAdd.hAdd (HPow.hPow (-1) q.natDegree).na …
        n1 : Ne (HMul.hMul (HPow.hPow (-1) (minpoly A x).natDegree) ((minpoly A x).com …
        n2 : Ne (HMul.hMul (HPow.hPow (-1) q.natDegree) (q.comp (Neg.neg Polynomial.X) …
        ⊢ LE.le ↑(HAdd.hAdd (HPow.hPow (-1) (minpoly A x).natDegree).natDegree (HMul.h …
      -/
      simpa using H
      /-
        🎉 no goals
      -/
    /-
      case neg
      A : Type u_1
      inst✝² : Field A
      B : Type u_3
      inst✝¹ : CommRing B
      inst✝ : Algebra A B
      x : B
      hx : Not (IsIntegral A x)
      ⊢ Eq (minpoly A (Neg.neg x)) (HMul.hMul (HPow.hPow (-1) (minpoly A x).natDegre …
    -/
  · rw [minpoly.eq_zero hx, minpoly.eq_zero, zero_comp]
      /-
        case neg
        A : Type u_1
        inst✝² : Field A
        B : Type u_3
        inst✝¹ : CommRing B
        inst✝ : Algebra A B
        x : B
        hx : Not (IsIntegral A x)
        ⊢ Eq 0 (HMul.hMul (HPow.hPow (-1) (Polynomial.natDegree 0)) 0)
      -/
    · simp only [natDegree_zero, pow_zero, mul_zero]
      /-
        🎉 no goals
      -/
      /-
        case neg
        A : Type u_1
        inst✝² : Field A
        B : Type u_3
        inst✝¹ : CommRing B
        inst✝ : Algebra A B
        x : B
        hx : Not (IsIntegral A x)
        ⊢ Not (IsIntegral A (Neg.neg x))
      -/
    · exact IsIntegral.neg_iff.not.mpr hx
      /-
        🎉 no goals
      -/


/-- A technical finiteness result. -/
noncomputable def Fintype.subtypeProd {E : Type*} {X : Set E} (hX : X.Finite) {L : Type*}
    (F : E → Multiset L) : Fintype (∀ x : X, { l : L // l ∈ F x }) :=
  @Pi.instFintype _ _ _ (Finite.fintype hX) _


/-- Function from Hom_K(E,L) to pi type Π (x : basis), roots of min poly of x -/
def rootsOfMinPolyPiType (φ : E →ₐ[F] K)
    (x : range (Module.finBasis F E : _ → E)) :
    { l : K // l ∈ (minpoly F x.1).aroots K } :=
  ⟨φ x, by
    rw [mem_roots_map (minpoly.ne_zero_of_finite F x.val),
      ← aeval_def, aeval_algHom_apply, minpoly.aeval, map_zero]⟩


theorem aux_inj_roots_of_min_poly : Injective (rootsOfMinPolyPiType F E K) := by
  /-
    F : Type u_3
    E : Type u_4
    K : Type u_5
    inst✝⁶ : Field F
    inst✝⁵ : Ring E
    inst✝⁴ : CommRing K
    inst✝³ : IsDomain K
    inst✝² : Algebra F E
    inst✝¹ : Algebra F K
    inst✝ : FiniteDimensional F E
    ⊢ Function.Injective (minpoly.rootsOfMinPolyPiType F E K)
  -/
  intro f g h
  -- needs explicit coercion on the RHS
  /-
    F : Type u_3
    E : Type u_4
    K : Type u_5
    inst✝⁶ : Field F
    inst✝⁵ : Ring E
    inst✝⁴ : CommRing K
    inst✝³ : IsDomain K
    inst✝² : Algebra F E
    inst✝¹ : Algebra F K
    inst✝ : FiniteDimensional F E
    f g : AlgHom F E K
    h : Eq (minpoly.rootsOfMinPolyPiType F E K f) (minpoly.rootsOfMinPolyPiType F  …
    ⊢ Eq f g
  -/
  suffices (f : E →ₗ[F] K) = (g : E →ₗ[F] K) by rwa [DFunLike.ext'_iff] at this ⊢
  /-
    F : Type u_3
    E : Type u_4
    K : Type u_5
    inst✝⁶ : Field F
    inst✝⁵ : Ring E
    inst✝⁴ : CommRing K
    inst✝³ : IsDomain K
    inst✝² : Algebra F E
    inst✝¹ : Algebra F K
    inst✝ : FiniteDimensional F E
    f g : AlgHom F E K
    h : Eq (minpoly.rootsOfMinPolyPiType F E K f) (minpoly.rootsOfMinPolyPiType F  …
    ⊢ Eq ↑f ↑g
  -/
  rw [funext_iff] at h
  exact LinearMap.ext_on (Module.finBasis F E).span_eq fun e he =>
    Subtype.ext_iff.mp (h ⟨e, he⟩)


/-- Given field extensions `E/F` and `K/F`, with `E/F` finite, there are finitely many `F`-algebra
  homomorphisms `E →ₐ[K] K`. -/
noncomputable instance AlgHom.fintype : Fintype (E →ₐ[F] K) :=
  @Fintype.ofInjective _ _
    (Fintype.subtypeProd (finite_range (Module.finBasis F E)) fun e =>
      (minpoly F e).aroots K)
    _ (aux_inj_roots_of_min_poly F E K)


/-- If `B/K` is a nontrivial algebra over a field, and `x` is an element of `K`,
then the minimal polynomial of `algebraMap K B x` is `X - C x`. -/
theorem eq_X_sub_C (a : A) : minpoly A (algebraMap A B a) = X - C a :=
  eq_X_sub_C_of_algebraMap_inj a (algebraMap A B).injective


theorem eq_X_sub_C' (a : A) : minpoly A a = X - C a :=
  eq_X_sub_C A a


/-- The minimal polynomial of `0` is `X`. -/
@[simp]
theorem zero : minpoly A (0 : B) = X := by
  /-
    A : Type u_1
    B : Type u_2
    inst✝³ : Field A
    inst✝² : Ring B
    inst✝¹ : Algebra A B
    inst✝ : Nontrivial B
    ⊢ Eq (minpoly A 0) Polynomial.X
  -/
  simpa only [add_zero, C_0, sub_eq_add_neg, neg_zero, RingHom.map_zero] using eq_X_sub_C B (0 : A)
  /-
    🎉 no goals
  -/


/-- The minimal polynomial of `1` is `X - 1`. -/
@[simp]
theorem one : minpoly A (1 : B) = X - 1 := by
  /-
    A : Type u_1
    B : Type u_2
    inst✝³ : Field A
    inst✝² : Ring B
    inst✝¹ : Algebra A B
    inst✝ : Nontrivial B
    ⊢ Eq (minpoly A 1) (HSub.hSub Polynomial.X 1)
  -/
  simpa only [RingHom.map_one, C_1, sub_eq_add_neg] using eq_X_sub_C B (1 : A)
  /-
    🎉 no goals
  -/


/-- A minimal polynomial is prime. -/
theorem prime (hx : IsIntegral A x) : Prime (minpoly A x) := by
  /-
    A : Type u_1
    B : Type u_2
    inst✝³ : Field A
    inst✝² : Ring B
    inst✝¹ : IsDomain B
    inst✝ : Algebra A B
    x : B
    hx : IsIntegral A x
    ⊢ Prime (minpoly A x)
  -/
  refine ⟨minpoly.ne_zero hx, not_isUnit A x, ?_⟩
  /-
    A : Type u_1
    B : Type u_2
    inst✝³ : Field A
    inst✝² : Ring B
    inst✝¹ : IsDomain B
    inst✝ : Algebra A B
    x : B
    hx : IsIntegral A x
    ⊢ ∀ (a b : Polynomial A), Dvd.dvd (minpoly A x) (HMul.hMul a b) → Or (Dvd.dvd  …
  -/
  rintro p q ⟨d, h⟩
  /-
    case intro
    A : Type u_1
    B : Type u_2
    inst✝³ : Field A
    inst✝² : Ring B
    inst✝¹ : IsDomain B
    inst✝ : Algebra A B
    x : B
    hx : IsIntegral A x
    p q d : Polynomial A
    h : Eq (HMul.hMul p q) (HMul.hMul (minpoly A x) d)
    ⊢ Or (Dvd.dvd (minpoly A x) p) (Dvd.dvd (minpoly A x) q)
  -/
  have : Polynomial.aeval x (p * q) = 0 := by simp [h, aeval A x]
  /-
    case intro
    A : Type u_1
    B : Type u_2
    inst✝³ : Field A
    inst✝² : Ring B
    inst✝¹ : IsDomain B
    inst✝ : Algebra A B
    x : B
    hx : IsIntegral A x
    p q d : Polynomial A
    h : Eq (HMul.hMul p q) (HMul.hMul (minpoly A x) d)
    this : Eq ((Polynomial.aeval x) (HMul.hMul p q)) 0
    ⊢ Or (Dvd.dvd (minpoly A x) p) (Dvd.dvd (minpoly A x) q)
  -/
  replace : Polynomial.aeval x p = 0 ∨ Polynomial.aeval x q = 0 := by simpa
  /-
    case intro
    A : Type u_1
    B : Type u_2
    inst✝³ : Field A
    inst✝² : Ring B
    inst✝¹ : IsDomain B
    inst✝ : Algebra A B
    x : B
    hx : IsIntegral A x
    p q d : Polynomial A
    h : Eq (HMul.hMul p q) (HMul.hMul (minpoly A x) d)
    this : Or (Eq ((Polynomial.aeval x) p) 0) (Eq ((Polynomial.aeval x) q) 0)
    ⊢ Or (Dvd.dvd (minpoly A x) p) (Dvd.dvd (minpoly A x) q)
  -/
  exact Or.imp (dvd A x) (dvd A x) this
  /-
    🎉 no goals
  -/


/-- If `L/K` is a field extension and an element `y` of `K` is a root of the minimal polynomial
of an element `x ∈ L`, then `y` maps to `x` under the field embedding. -/
theorem root {x : B} (hx : IsIntegral A x) {y : A} (h : IsRoot (minpoly A x) y) :
    algebraMap A B y = x := by
  have key : minpoly A x = X - C y := eq_of_monic_of_associated (monic hx) (monic_X_sub_C y)
    (associated_of_dvd_dvd ((irreducible_X_sub_C y).dvd_symm (irreducible hx) (dvd_iff_isRoot.2 h))
      (dvd_iff_isRoot.2 h))
  /-
    A : Type u_1
    B : Type u_2
    inst✝³ : Field A
    inst✝² : Ring B
    inst✝¹ : IsDomain B
    inst✝ : Algebra A B
    x : B
    hx : IsIntegral A x
    y : A
    h : (minpoly A x).IsRoot y
    key : Eq (minpoly A x) (HSub.hSub Polynomial.X (Polynomial.C y))
    ⊢ Eq ((algebraMap A B) y) x
  -/
  have := aeval A x
  /-
    A : Type u_1
    B : Type u_2
    inst✝³ : Field A
    inst✝² : Ring B
    inst✝¹ : IsDomain B
    inst✝ : Algebra A B
    x : B
    hx : IsIntegral A x
    y : A
    h : (minpoly A x).IsRoot y
    key : Eq (minpoly A x) (HSub.hSub Polynomial.X (Polynomial.C y))
    this : Eq ((Polynomial.aeval x) (minpoly A x)) 0
    ⊢ Eq ((algebraMap A B) y) x
  -/
  rwa [key, map_sub, aeval_X, aeval_C, sub_eq_zero, eq_comm] at this
  /-
    🎉 no goals
  -/


/-- The constant coefficient of the minimal polynomial of `x` is `0` if and only if `x = 0`. -/
@[simp]
theorem coeff_zero_eq_zero (hx : IsIntegral A x) : coeff (minpoly A x) 0 = 0 ↔ x = 0 := by
  /-
    A : Type u_1
    B : Type u_2
    inst✝³ : Field A
    inst✝² : Ring B
    inst✝¹ : IsDomain B
    inst✝ : Algebra A B
    x : B
    hx : IsIntegral A x
    ⊢ Iff (Eq ((minpoly A x).coeff 0) 0) (Eq x 0)
  -/
  constructor
    /-
      case mp
      A : Type u_1
      B : Type u_2
      inst✝³ : Field A
      inst✝² : Ring B
      inst✝¹ : IsDomain B
      inst✝ : Algebra A B
      x : B
      hx : IsIntegral A x
      ⊢ Eq ((minpoly A x).coeff 0) 0 → Eq x 0
    -/
  · intro h
    /-
      case mp
      A : Type u_1
      B : Type u_2
      inst✝³ : Field A
      inst✝² : Ring B
      inst✝¹ : IsDomain B
      inst✝ : Algebra A B
      x : B
      hx : IsIntegral A x
      h : Eq ((minpoly A x).coeff 0) 0
      ⊢ Eq x 0
    -/
    have zero_root := zero_isRoot_of_coeff_zero_eq_zero h
    /-
      case mp
      A : Type u_1
      B : Type u_2
      inst✝³ : Field A
      inst✝² : Ring B
      inst✝¹ : IsDomain B
      inst✝ : Algebra A B
      x : B
      hx : IsIntegral A x
      h : Eq ((minpoly A x).coeff 0) 0
      zero_root : (minpoly A x).IsRoot 0
      ⊢ Eq x 0
    -/
    rw [← root hx zero_root]
    /-
      case mp
      A : Type u_1
      B : Type u_2
      inst✝³ : Field A
      inst✝² : Ring B
      inst✝¹ : IsDomain B
      inst✝ : Algebra A B
      x : B
      hx : IsIntegral A x
      h : Eq ((minpoly A x).coeff 0) 0
      zero_root : (minpoly A x).IsRoot 0
      ⊢ Eq ((algebraMap A B) 0) 0
    -/
    exact RingHom.map_zero _
    /-
      🎉 no goals
    -/
    /-
      case mpr
      A : Type u_1
      B : Type u_2
      inst✝³ : Field A
      inst✝² : Ring B
      inst✝¹ : IsDomain B
      inst✝ : Algebra A B
      x : B
      hx : IsIntegral A x
      ⊢ Eq x 0 → Eq ((minpoly A x).coeff 0) 0
    -/
  · rintro rfl
    /-
      case mpr
      A : Type u_1
      B : Type u_2
      inst✝³ : Field A
      inst✝² : Ring B
      inst✝¹ : IsDomain B
      inst✝ : Algebra A B
      hx : IsIntegral A 0
      ⊢ Eq ((minpoly A 0).coeff 0) 0
    -/
    simp
    /-
      🎉 no goals
    -/


/-- The minimal polynomial of a nonzero element has nonzero constant coefficient. -/
theorem coeff_zero_ne_zero (hx : IsIntegral A x) (h : x ≠ 0) : coeff (minpoly A x) 0 ≠ 0 := by
  /-
    A : Type u_1
    B : Type u_2
    inst✝³ : Field A
    inst✝² : Ring B
    inst✝¹ : IsDomain B
    inst✝ : Algebra A B
    x : B
    hx : IsIntegral A x
    h : Ne x 0
    ⊢ Ne ((minpoly A x).coeff 0) 0
  -/
  contrapose! h
  /-
    A : Type u_1
    B : Type u_2
    inst✝³ : Field A
    inst✝² : Ring B
    inst✝¹ : IsDomain B
    inst✝ : Algebra A B
    x : B
    hx : IsIntegral A x
    h : Eq ((minpoly A x).coeff 0) 0
    ⊢ Eq x 0
  -/
  simpa only [hx, coeff_zero_eq_zero] using h
  /-
    🎉 no goals
  -/


/-- The minimal polynomial (over `K`) of `σ : Gal(L/K)` is `X ^ (orderOf σ) - 1`. -/
lemma minpoly_algEquiv_toLinearMap (σ : L ≃ₐ[K] L) (hσ : IsOfFinOrder σ) :
    minpoly K σ.toLinearMap = X ^ (orderOf σ) - C 1 := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : CommRing L
    inst✝¹ : IsDomain L
    inst✝ : Algebra K L
    σ : AlgEquiv K L L
    hσ : IsOfFinOrder σ
    ⊢ Eq (minpoly K σ.toLinearMap) (HSub.hSub (HPow.hPow Polynomial.X (orderOf σ)) …
  -/
  refine (minpoly.unique _ _ (monic_X_pow_sub_C _ hσ.orderOf_pos.ne.symm) ?_ ?_).symm
    /-
      case refine_1
      K : Type u_1
      L : Type u_2
      inst✝³ : Field K
      inst✝² : CommRing L
      inst✝¹ : IsDomain L
      inst✝ : Algebra K L
      σ : AlgEquiv K L L
      hσ : IsOfFinOrder σ
      ⊢ Eq ((Polynomial.aeval σ.toLinearMap) (HSub.hSub (HPow.hPow Polynomial.X (ord …
    -/
  · rw [map_sub]
    /-
      case refine_1
      K : Type u_1
      L : Type u_2
      inst✝³ : Field K
      inst✝² : CommRing L
      inst✝¹ : IsDomain L
      inst✝ : Algebra K L
      σ : AlgEquiv K L L
      hσ : IsOfFinOrder σ
      ⊢ Eq (HSub.hSub ((Polynomial.aeval σ.toLinearMap) (HPow.hPow Polynomial.X (ord …
    -/
    simp [← AlgEquiv.pow_toLinearMap, pow_orderOf_eq_one]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      K : Type u_1
      L : Type u_2
      inst✝³ : Field K
      inst✝² : CommRing L
      inst✝¹ : IsDomain L
      inst✝ : Algebra K L
      σ : AlgEquiv K L L
      hσ : IsOfFinOrder σ
      ⊢ ∀ (q : Polynomial K), q.Monic → Eq ((Polynomial.aeval σ.toLinearMap) q) 0 →  …
    -/
  · intros q hq hs
    /-
      case refine_2
      K : Type u_1
      L : Type u_2
      inst✝³ : Field K
      inst✝² : CommRing L
      inst✝¹ : IsDomain L
      inst✝ : Algebra K L
      σ : AlgEquiv K L L
      hσ : IsOfFinOrder σ
      q : Polynomial K
      hq : q.Monic
      hs : Eq ((Polynomial.aeval σ.toLinearMap) q) 0
      ⊢ LE.le (HSub.hSub (HPow.hPow Polynomial.X (orderOf σ)) (Polynomial.C 1)).degr …
    -/
    rw [degree_eq_natDegree hq.ne_zero, degree_X_pow_sub_C hσ.orderOf_pos, Nat.cast_le, ← not_lt]
    /-
      case refine_2
      K : Type u_1
      L : Type u_2
      inst✝³ : Field K
      inst✝² : CommRing L
      inst✝¹ : IsDomain L
      inst✝ : Algebra K L
      σ : AlgEquiv K L L
      hσ : IsOfFinOrder σ
      q : Polynomial K
      hq : q.Monic
      hs : Eq ((Polynomial.aeval σ.toLinearMap) q) 0
      ⊢ Not (LT.lt q.natDegree (orderOf σ))
    -/
    intro H
    /-
      case refine_2
      K : Type u_1
      L : Type u_2
      inst✝³ : Field K
      inst✝² : CommRing L
      inst✝¹ : IsDomain L
      inst✝ : Algebra K L
      σ : AlgEquiv K L L
      hσ : IsOfFinOrder σ
      q : Polynomial K
      hq : q.Monic
      hs : Eq ((Polynomial.aeval σ.toLinearMap) q) 0
      H : LT.lt q.natDegree (orderOf σ)
      ⊢ False
    -/
    rw [aeval_eq_sum_range' H, ← Fin.sum_univ_eq_sum_range] at hs
    /-
      case refine_2
      K : Type u_1
      L : Type u_2
      inst✝³ : Field K
      inst✝² : CommRing L
      inst✝¹ : IsDomain L
      inst✝ : Algebra K L
      σ : AlgEquiv K L L
      hσ : IsOfFinOrder σ
      q : Polynomial K
      hq : q.Monic
      hs : Eq (Finset.univ.sum fun i => HSMul.hSMul (q.coeff ↑i) (HPow.hPow σ.toLine …
      H : LT.lt q.natDegree (orderOf σ)
      ⊢ False
    -/
    simp_rw [← AlgEquiv.pow_toLinearMap] at hs
    /-
      case refine_2
      K : Type u_1
      L : Type u_2
      inst✝³ : Field K
      inst✝² : CommRing L
      inst✝¹ : IsDomain L
      inst✝ : Algebra K L
      σ : AlgEquiv K L L
      hσ : IsOfFinOrder σ
      q : Polynomial K
      hq : q.Monic
      H : LT.lt q.natDegree (orderOf σ)
      hs : Eq (Finset.univ.sum fun x => HSMul.hSMul (q.coeff ↑x) (HPow.hPow σ ↑x).to …
      ⊢ False
    -/
    apply hq.ne_zero
    simpa using Fintype.linearIndependent_iff.mp
      (((linearIndependent_algHom_toLinearMap' K L L).comp _ AlgEquiv.coe_algHom_injective).comp _
        (Subtype.val_injective.comp ((finEquivPowers σ hσ).injective)))
      (q.coeff ∘ (↑)) hs ⟨_, H⟩


/-- The minimal polynomial (over `K`) of `σ : Gal(L/K)` is `X ^ (orderOf σ) - 1`. -/
lemma minpoly_algHom_toLinearMap (σ : L →ₐ[K] L) (hσ : IsOfFinOrder σ) :
    minpoly K σ.toLinearMap = X ^ (orderOf σ) - C 1 := by
  have : orderOf σ = orderOf (AlgEquiv.algHomUnitsEquiv _ _ hσ.unit) := by
    rw [← MonoidHom.coe_coe, orderOf_injective, ← orderOf_units, IsOfFinOrder.val_unit]
    exact (AlgEquiv.algHomUnitsEquiv K L).injective
  /-
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : CommRing L
    inst✝¹ : IsDomain L
    inst✝ : Algebra K L
    σ : AlgHom K L L
    hσ : IsOfFinOrder σ
    this : Eq (orderOf σ) (orderOf ((AlgEquiv.algHomUnitsEquiv K L) hσ.unit))
    ⊢ Eq (minpoly K σ.toLinearMap) (HSub.hSub (HPow.hPow Polynomial.X (orderOf σ)) …
  -/
  rw [this, ← minpoly_algEquiv_toLinearMap]
    /-
      K : Type u_1
      L : Type u_2
      inst✝³ : Field K
      inst✝² : CommRing L
      inst✝¹ : IsDomain L
      inst✝ : Algebra K L
      σ : AlgHom K L L
      hσ : IsOfFinOrder σ
      this : Eq (orderOf σ) (orderOf ((AlgEquiv.algHomUnitsEquiv K L) hσ.unit))
      ⊢ Eq (minpoly K σ.toLinearMap) (minpoly K ((AlgEquiv.algHomUnitsEquiv K L) hσ. …
    -/
  · apply congr_arg
    /-
      case h
      K : Type u_1
      L : Type u_2
      inst✝³ : Field K
      inst✝² : CommRing L
      inst✝¹ : IsDomain L
      inst✝ : Algebra K L
      σ : AlgHom K L L
      hσ : IsOfFinOrder σ
      this : Eq (orderOf σ) (orderOf ((AlgEquiv.algHomUnitsEquiv K L) hσ.unit))
      ⊢ Eq σ.toLinearMap ((AlgEquiv.algHomUnitsEquiv K L) hσ.unit).toLinearMap
    -/
    ext
    /-
      case h.h
      K : Type u_1
      L : Type u_2
      inst✝³ : Field K
      inst✝² : CommRing L
      inst✝¹ : IsDomain L
      inst✝ : Algebra K L
      σ : AlgHom K L L
      hσ : IsOfFinOrder σ
      this : Eq (orderOf σ) (orderOf ((AlgEquiv.algHomUnitsEquiv K L) hσ.unit))
      x✝ : L
      ⊢ Eq (σ.toLinearMap x✝) (((AlgEquiv.algHomUnitsEquiv K L) hσ.unit).toLinearMap …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case hσ
      K : Type u_1
      L : Type u_2
      inst✝³ : Field K
      inst✝² : CommRing L
      inst✝¹ : IsDomain L
      inst✝ : Algebra K L
      σ : AlgHom K L L
      hσ : IsOfFinOrder σ
      this : Eq (orderOf σ) (orderOf ((AlgEquiv.algHomUnitsEquiv K L) hσ.unit))
      ⊢ IsOfFinOrder ((AlgEquiv.algHomUnitsEquiv K L) hσ.unit)
    -/
  · rwa [← orderOf_pos_iff, ← this, orderOf_pos_iff]
    /-
      🎉 no goals
    -/


