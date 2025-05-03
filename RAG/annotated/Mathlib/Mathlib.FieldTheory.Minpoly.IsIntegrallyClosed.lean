/-- For integrally closed domains, the minimal polynomial over the ring is the same as the minimal
polynomial over the fraction field. See `minpoly.isIntegrallyClosed_eq_field_fractions'` if
`S` is already a `K`-algebra. -/
theorem isIntegrallyClosed_eq_field_fractions [IsDomain S] {s : S} (hs : IsIntegral R s) :
    minpoly K (algebraMap S L s) = (minpoly R s).map (algebraMap R K) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹⁵ : CommRing R
    inst✝¹⁴ : CommRing S
    inst✝¹³ : IsDomain R
    inst✝¹² : Algebra R S
    K : Type u_3
    L : Type u_4
    inst✝¹¹ : Field K
    inst✝¹⁰ : Algebra R K
    inst✝⁹ : IsFractionRing R K
    inst✝⁸ : CommRing L
    inst✝⁷ : Nontrivial L
    inst✝⁶ : Algebra R L
    inst✝⁵ : Algebra S L
    inst✝⁴ : Algebra K L
    inst✝³ : IsScalarTower R K L
    inst✝² : IsScalarTower R S L
    inst✝¹ : IsIntegrallyClosed R
    inst✝ : IsDomain S
    s : S
    hs : IsIntegral R s
    ⊢ Eq (minpoly K ((algebraMap S L) s)) (Polynomial.map (algebraMap R K) (minpol …
  -/
  refine (eq_of_irreducible_of_monic ?_ ?_ ?_).symm
    /-
      case refine_1
      R : Type u_1
      S : Type u_2
      inst✝¹⁵ : CommRing R
      inst✝¹⁴ : CommRing S
      inst✝¹³ : IsDomain R
      inst✝¹² : Algebra R S
      K : Type u_3
      L : Type u_4
      inst✝¹¹ : Field K
      inst✝¹⁰ : Algebra R K
      inst✝⁹ : IsFractionRing R K
      inst✝⁸ : CommRing L
      inst✝⁷ : Nontrivial L
      inst✝⁶ : Algebra R L
      inst✝⁵ : Algebra S L
      inst✝⁴ : Algebra K L
      inst✝³ : IsScalarTower R K L
      inst✝² : IsScalarTower R S L
      inst✝¹ : IsIntegrallyClosed R
      inst✝ : IsDomain S
      s : S
      hs : IsIntegral R s
      ⊢ Irreducible (Polynomial.map (algebraMap R K) (minpoly R s))
    -/
  · exact ((monic hs).irreducible_iff_irreducible_map_fraction_map).1 (irreducible hs)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      S : Type u_2
      inst✝¹⁵ : CommRing R
      inst✝¹⁴ : CommRing S
      inst✝¹³ : IsDomain R
      inst✝¹² : Algebra R S
      K : Type u_3
      L : Type u_4
      inst✝¹¹ : Field K
      inst✝¹⁰ : Algebra R K
      inst✝⁹ : IsFractionRing R K
      inst✝⁸ : CommRing L
      inst✝⁷ : Nontrivial L
      inst✝⁶ : Algebra R L
      inst✝⁵ : Algebra S L
      inst✝⁴ : Algebra K L
      inst✝³ : IsScalarTower R K L
      inst✝² : IsScalarTower R S L
      inst✝¹ : IsIntegrallyClosed R
      inst✝ : IsDomain S
      s : S
      hs : IsIntegral R s
      ⊢ Eq ((Polynomial.aeval ((algebraMap S L) s)) (Polynomial.map (algebraMap R K) …
    -/
  · rw [aeval_map_algebraMap, aeval_algebraMap_apply, aeval, map_zero]
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      R : Type u_1
      S : Type u_2
      inst✝¹⁵ : CommRing R
      inst✝¹⁴ : CommRing S
      inst✝¹³ : IsDomain R
      inst✝¹² : Algebra R S
      K : Type u_3
      L : Type u_4
      inst✝¹¹ : Field K
      inst✝¹⁰ : Algebra R K
      inst✝⁹ : IsFractionRing R K
      inst✝⁸ : CommRing L
      inst✝⁷ : Nontrivial L
      inst✝⁶ : Algebra R L
      inst✝⁵ : Algebra S L
      inst✝⁴ : Algebra K L
      inst✝³ : IsScalarTower R K L
      inst✝² : IsScalarTower R S L
      inst✝¹ : IsIntegrallyClosed R
      inst✝ : IsDomain S
      s : S
      hs : IsIntegral R s
      ⊢ (Polynomial.map (algebraMap R K) (minpoly R s)).Monic
    -/
  · exact (monic hs).map _
    /-
      🎉 no goals
    -/


/-- For integrally closed domains, the minimal polynomial over the ring is the same as the minimal
polynomial over the fraction field. Compared to `minpoly.isIntegrallyClosed_eq_field_fractions`,
this version is useful if the element is in a ring that is already a `K`-algebra. -/
theorem isIntegrallyClosed_eq_field_fractions' [IsDomain S] [Algebra K S] [IsScalarTower R K S]
    {s : S} (hs : IsIntegral R s) : minpoly K s = (minpoly R s).map (algebraMap R K) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : IsDomain R
    inst✝⁷ : Algebra R S
    K : Type u_3
    inst✝⁶ : Field K
    inst✝⁵ : Algebra R K
    inst✝⁴ : IsFractionRing R K
    inst✝³ : IsIntegrallyClosed R
    inst✝² : IsDomain S
    inst✝¹ : Algebra K S
    inst✝ : IsScalarTower R K S
    s : S
    hs : IsIntegral R s
    ⊢ Eq (minpoly K s) (Polynomial.map (algebraMap R K) (minpoly R s))
  -/
  let L := FractionRing S
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : IsDomain R
    inst✝⁷ : Algebra R S
    K : Type u_3
    inst✝⁶ : Field K
    inst✝⁵ : Algebra R K
    inst✝⁴ : IsFractionRing R K
    inst✝³ : IsIntegrallyClosed R
    inst✝² : IsDomain S
    inst✝¹ : Algebra K S
    inst✝ : IsScalarTower R K S
    s : S
    hs : IsIntegral R s
    L : Type u_2 := FractionRing S
    ⊢ Eq (minpoly K s) (Polynomial.map (algebraMap R K) (minpoly R s))
  -/
  rw [← isIntegrallyClosed_eq_field_fractions K L hs, algebraMap_eq (IsFractionRing.injective S L)]
  /-
    🎉 no goals
  -/


/-- For integrally closed rings, the minimal polynomial divides any polynomial that has the
  integral element as root. See also `minpoly.dvd` which relaxes the assumptions on `S`
  in exchange for stronger assumptions on `R`. -/
theorem isIntegrallyClosed_dvd {s : S} (hs : IsIntegral R s) {p : R[X]}
    (hp : Polynomial.aeval s p = 0) : minpoly R s ∣ p := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : IsDomain R
    inst✝³ : Algebra R S
    inst✝² : IsDomain S
    inst✝¹ : NoZeroSMulDivisors R S
    inst✝ : IsIntegrallyClosed R
    s : S
    hs : IsIntegral R s
    p : Polynomial R
    hp : Eq ((Polynomial.aeval s) p) 0
    ⊢ Dvd.dvd (minpoly R s) p
  -/
  let K := FractionRing R
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : IsDomain R
    inst✝³ : Algebra R S
    inst✝² : IsDomain S
    inst✝¹ : NoZeroSMulDivisors R S
    inst✝ : IsIntegrallyClosed R
    s : S
    hs : IsIntegral R s
    p : Polynomial R
    hp : Eq ((Polynomial.aeval s) p) 0
    K : Type u_1 := FractionRing R
    ⊢ Dvd.dvd (minpoly R s) p
  -/
  let L := FractionRing S
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : IsDomain R
    inst✝³ : Algebra R S
    inst✝² : IsDomain S
    inst✝¹ : NoZeroSMulDivisors R S
    inst✝ : IsIntegrallyClosed R
    s : S
    hs : IsIntegral R s
    p : Polynomial R
    hp : Eq ((Polynomial.aeval s) p) 0
    K : Type u_1 := FractionRing R
    L : Type u_2 := FractionRing S
    ⊢ Dvd.dvd (minpoly R s) p
  -/
  let _ : Algebra K L := FractionRing.liftAlgebra R L
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : IsDomain R
    inst✝³ : Algebra R S
    inst✝² : IsDomain S
    inst✝¹ : NoZeroSMulDivisors R S
    inst✝ : IsIntegrallyClosed R
    s : S
    hs : IsIntegral R s
    p : Polynomial R
    hp : Eq ((Polynomial.aeval s) p) 0
    K : Type u_1 := FractionRing R
    L : Type u_2 := FractionRing S
    x✝ : Algebra K L := FractionRing.liftAlgebra R L
    ⊢ Dvd.dvd (minpoly R s) p
  -/
  have := FractionRing.isScalarTower_liftAlgebra R L
  have : minpoly K (algebraMap S L s) ∣ map (algebraMap R K) (p %ₘ minpoly R s) := by
    rw [map_modByMonic _ (minpoly.monic hs), modByMonic_eq_sub_mul_div]
    · refine dvd_sub (minpoly.dvd K (algebraMap S L s) ?_) ?_
      · rw [← map_aeval_eq_aeval_map, hp, map_zero]
        rw [← IsScalarTower.algebraMap_eq, ← IsScalarTower.algebraMap_eq]
      apply dvd_mul_of_dvd_left
      rw [isIntegrallyClosed_eq_field_fractions K L hs]
    exact Monic.map _ (minpoly.monic hs)
  rw [isIntegrallyClosed_eq_field_fractions _ _ hs,
    map_dvd_map (algebraMap R K) (IsFractionRing.injective R K) (minpoly.monic hs)] at this
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : IsDomain R
    inst✝³ : Algebra R S
    inst✝² : IsDomain S
    inst✝¹ : NoZeroSMulDivisors R S
    inst✝ : IsIntegrallyClosed R
    s : S
    hs : IsIntegral R s
    p : Polynomial R
    hp : Eq ((Polynomial.aeval s) p) 0
    K : Type u_1 := FractionRing R
    L : Type u_2 := FractionRing S
    x✝ : Algebra K L := FractionRing.liftAlgebra R L
    this✝ : IsScalarTower R (FractionRing R) L
    this : Dvd.dvd (minpoly R s) (p.modByMonic (minpoly R s))
    ⊢ Dvd.dvd (minpoly R s) p
  -/
  rw [← modByMonic_eq_zero_iff_dvd (minpoly.monic hs)]
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : IsDomain R
    inst✝³ : Algebra R S
    inst✝² : IsDomain S
    inst✝¹ : NoZeroSMulDivisors R S
    inst✝ : IsIntegrallyClosed R
    s : S
    hs : IsIntegral R s
    p : Polynomial R
    hp : Eq ((Polynomial.aeval s) p) 0
    K : Type u_1 := FractionRing R
    L : Type u_2 := FractionRing S
    x✝ : Algebra K L := FractionRing.liftAlgebra R L
    this✝ : IsScalarTower R (FractionRing R) L
    this : Dvd.dvd (minpoly R s) (p.modByMonic (minpoly R s))
    ⊢ Eq (p.modByMonic (minpoly R s)) 0
  -/
  exact Polynomial.eq_zero_of_dvd_of_degree_lt this (degree_modByMonic_lt p <| minpoly.monic hs)
  /-
    🎉 no goals
  -/


theorem isIntegrallyClosed_dvd_iff {s : S} (hs : IsIntegral R s) (p : R[X]) :
    Polynomial.aeval s p = 0 ↔ minpoly R s ∣ p :=
  ⟨fun hp => isIntegrallyClosed_dvd hs hp, fun hp => by
    simpa only [RingHom.mem_ker, RingHom.coe_comp, coe_evalRingHom, coe_mapRingHom,
      Function.comp_apply, eval_map, ← aeval_def] using
      aeval_eq_zero_of_dvd_aeval_eq_zero hp (minpoly.aeval R s)⟩


theorem ker_eval {s : S} (hs : IsIntegral R s) :
    RingHom.ker ((Polynomial.aeval s).toRingHom : R[X] →+* S) =
    Ideal.span ({minpoly R s} : Set R[X]) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : IsDomain R
    inst✝³ : Algebra R S
    inst✝² : IsDomain S
    inst✝¹ : NoZeroSMulDivisors R S
    inst✝ : IsIntegrallyClosed R
    s : S
    hs : IsIntegral R s
    ⊢ Eq (RingHom.ker (Polynomial.aeval s).toRingHom) (Ideal.span (Singleton.singl …
  -/
  ext p
  simp_rw [RingHom.mem_ker, AlgHom.toRingHom_eq_coe, AlgHom.coe_toRingHom,
    isIntegrallyClosed_dvd_iff hs, ← Ideal.mem_span_singleton]


/-- If an element `x` is a root of a nonzero polynomial `p`, then the degree of `p` is at least the
degree of the minimal polynomial of `x`. See also `minpoly.degree_le_of_ne_zero` which relaxes the
assumptions on `S` in exchange for stronger assumptions on `R`. -/
theorem IsIntegrallyClosed.degree_le_of_ne_zero {s : S} (hs : IsIntegral R s) {p : R[X]}
    (hp0 : p ≠ 0) (hp : Polynomial.aeval s p = 0) : degree (minpoly R s) ≤ degree p := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : IsDomain R
    inst✝³ : Algebra R S
    inst✝² : IsDomain S
    inst✝¹ : NoZeroSMulDivisors R S
    inst✝ : IsIntegrallyClosed R
    s : S
    hs : IsIntegral R s
    p : Polynomial R
    hp0 : Ne p 0
    hp : Eq ((Polynomial.aeval s) p) 0
    ⊢ LE.le (minpoly R s).degree p.degree
  -/
  rw [degree_eq_natDegree (minpoly.ne_zero hs), degree_eq_natDegree hp0]
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : IsDomain R
    inst✝³ : Algebra R S
    inst✝² : IsDomain S
    inst✝¹ : NoZeroSMulDivisors R S
    inst✝ : IsIntegrallyClosed R
    s : S
    hs : IsIntegral R s
    p : Polynomial R
    hp0 : Ne p 0
    hp : Eq ((Polynomial.aeval s) p) 0
    ⊢ LE.le ↑(minpoly R s).natDegree ↑p.natDegree
  -/
  norm_cast
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : IsDomain R
    inst✝³ : Algebra R S
    inst✝² : IsDomain S
    inst✝¹ : NoZeroSMulDivisors R S
    inst✝ : IsIntegrallyClosed R
    s : S
    hs : IsIntegral R s
    p : Polynomial R
    hp0 : Ne p 0
    hp : Eq ((Polynomial.aeval s) p) 0
    ⊢ LE.le (minpoly R s).natDegree p.natDegree
  -/
  exact natDegree_le_of_dvd ((isIntegrallyClosed_dvd_iff hs _).mp hp) hp0
  /-
    🎉 no goals
  -/


/-- The minimal polynomial of an element `x` is uniquely characterized by its defining property:
if there is another monic polynomial of minimal degree that has `x` as a root, then this polynomial
is equal to the minimal polynomial of `x`. See also `minpoly.unique` which relaxes the
assumptions on `S` in exchange for stronger assumptions on `R`. -/
theorem _root_.IsIntegrallyClosed.minpoly.unique {s : S} {P : R[X]} (hmo : P.Monic)
    (hP : Polynomial.aeval s P = 0)
    (Pmin : ∀ Q : R[X], Q.Monic → Polynomial.aeval s Q = 0 → degree P ≤ degree Q) :
    P = minpoly R s := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : IsDomain R
    inst✝³ : Algebra R S
    inst✝² : IsDomain S
    inst✝¹ : NoZeroSMulDivisors R S
    inst✝ : IsIntegrallyClosed R
    s : S
    P : Polynomial R
    hmo : P.Monic
    hP : Eq ((Polynomial.aeval s) P) 0
    Pmin : ∀ (Q : Polynomial R), Q.Monic → Eq ((Polynomial.aeval s) Q) 0 → LE.le P …
    ⊢ Eq P (minpoly R s)
  -/
  have hs : IsIntegral R s := ⟨P, hmo, hP⟩
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : IsDomain R
    inst✝³ : Algebra R S
    inst✝² : IsDomain S
    inst✝¹ : NoZeroSMulDivisors R S
    inst✝ : IsIntegrallyClosed R
    s : S
    P : Polynomial R
    hmo : P.Monic
    hP : Eq ((Polynomial.aeval s) P) 0
    Pmin : ∀ (Q : Polynomial R), Q.Monic → Eq ((Polynomial.aeval s) Q) 0 → LE.le P …
    hs : IsIntegral R s
    ⊢ Eq P (minpoly R s)
  -/
  symm; apply eq_of_sub_eq_zero
  /-
    case h
    R : Type u_1
    S : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : IsDomain R
    inst✝³ : Algebra R S
    inst✝² : IsDomain S
    inst✝¹ : NoZeroSMulDivisors R S
    inst✝ : IsIntegrallyClosed R
    s : S
    P : Polynomial R
    hmo : P.Monic
    hP : Eq ((Polynomial.aeval s) P) 0
    Pmin : ∀ (Q : Polynomial R), Q.Monic → Eq ((Polynomial.aeval s) Q) 0 → LE.le P …
    hs : IsIntegral R s
    ⊢ Eq (HSub.hSub (minpoly R s) P) 0
  -/
  by_contra hnz
  /-
    case h
    R : Type u_1
    S : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : IsDomain R
    inst✝³ : Algebra R S
    inst✝² : IsDomain S
    inst✝¹ : NoZeroSMulDivisors R S
    inst✝ : IsIntegrallyClosed R
    s : S
    P : Polynomial R
    hmo : P.Monic
    hP : Eq ((Polynomial.aeval s) P) 0
    Pmin : ∀ (Q : Polynomial R), Q.Monic → Eq ((Polynomial.aeval s) Q) 0 → LE.le P …
    hs : IsIntegral R s
    hnz : Not (Eq (HSub.hSub (minpoly R s) P) 0)
    ⊢ False
  -/
  refine IsIntegrallyClosed.degree_le_of_ne_zero hs hnz (by simp [hP]) |>.not_lt ?_
  /-
    case h
    R : Type u_1
    S : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : IsDomain R
    inst✝³ : Algebra R S
    inst✝² : IsDomain S
    inst✝¹ : NoZeroSMulDivisors R S
    inst✝ : IsIntegrallyClosed R
    s : S
    P : Polynomial R
    hmo : P.Monic
    hP : Eq ((Polynomial.aeval s) P) 0
    Pmin : ∀ (Q : Polynomial R), Q.Monic → Eq ((Polynomial.aeval s) Q) 0 → LE.le P …
    hs : IsIntegral R s
    hnz : Not (Eq (HSub.hSub (minpoly R s) P) 0)
    ⊢ LT.lt (HSub.hSub (minpoly R s) P).degree (minpoly R s).degree
  -/
  refine degree_sub_lt ?_ (ne_zero hs) ?_
    /-
      case h.refine_1
      R : Type u_1
      S : Type u_2
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : IsDomain R
      inst✝³ : Algebra R S
      inst✝² : IsDomain S
      inst✝¹ : NoZeroSMulDivisors R S
      inst✝ : IsIntegrallyClosed R
      s : S
      P : Polynomial R
      hmo : P.Monic
      hP : Eq ((Polynomial.aeval s) P) 0
      Pmin : ∀ (Q : Polynomial R), Q.Monic → Eq ((Polynomial.aeval s) Q) 0 → LE.le P …
      hs : IsIntegral R s
      hnz : Not (Eq (HSub.hSub (minpoly R s) P) 0)
      ⊢ Eq (minpoly R s).degree P.degree
    -/
  · exact le_antisymm (min R s hmo hP) (Pmin (minpoly R s) (monic hs) (aeval R s))
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      R : Type u_1
      S : Type u_2
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing S
      inst✝⁴ : IsDomain R
      inst✝³ : Algebra R S
      inst✝² : IsDomain S
      inst✝¹ : NoZeroSMulDivisors R S
      inst✝ : IsIntegrallyClosed R
      s : S
      P : Polynomial R
      hmo : P.Monic
      hP : Eq ((Polynomial.aeval s) P) 0
      Pmin : ∀ (Q : Polynomial R), Q.Monic → Eq ((Polynomial.aeval s) Q) 0 → LE.le P …
      hs : IsIntegral R s
      hnz : Not (Eq (HSub.hSub (minpoly R s) P) 0)
      ⊢ Eq (minpoly R s).leadingCoeff P.leadingCoeff
    -/
  · rw [(monic hs).leadingCoeff, hmo.leadingCoeff]
    /-
      🎉 no goals
    -/


theorem prime_of_isIntegrallyClosed {x : S} (hx : IsIntegral R x) : Prime (minpoly R x) := by
  refine
    ⟨(minpoly.monic hx).ne_zero,
      ⟨fun h_contra => (ne_of_lt (minpoly.degree_pos hx)) (degree_eq_zero_of_isUnit h_contra).symm,
        fun a b h => or_iff_not_imp_left.mpr fun h' => ?_⟩⟩
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : IsDomain R
    inst✝³ : Algebra R S
    inst✝² : IsDomain S
    inst✝¹ : NoZeroSMulDivisors R S
    inst✝ : IsIntegrallyClosed R
    x : S
    hx : IsIntegral R x
    a b : Polynomial R
    h : Dvd.dvd (minpoly R x) (HMul.hMul a b)
    h' : Not (Dvd.dvd (minpoly R x) a)
    ⊢ Dvd.dvd (minpoly R x) b
  -/
  rw [← minpoly.isIntegrallyClosed_dvd_iff hx] at h' h ⊢
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : IsDomain R
    inst✝³ : Algebra R S
    inst✝² : IsDomain S
    inst✝¹ : NoZeroSMulDivisors R S
    inst✝ : IsIntegrallyClosed R
    x : S
    hx : IsIntegral R x
    a b : Polynomial R
    h : Eq ((Polynomial.aeval x) (HMul.hMul a b)) 0
    h' : Not (Eq ((Polynomial.aeval x) a) 0)
    ⊢ Eq ((Polynomial.aeval x) b) 0
  -/
  rw [aeval_mul] at h
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : IsDomain R
    inst✝³ : Algebra R S
    inst✝² : IsDomain S
    inst✝¹ : NoZeroSMulDivisors R S
    inst✝ : IsIntegrallyClosed R
    x : S
    hx : IsIntegral R x
    a b : Polynomial R
    h : Eq (HMul.hMul ((Polynomial.aeval x) a) ((Polynomial.aeval x) b)) 0
    h' : Not (Eq ((Polynomial.aeval x) a) 0)
    ⊢ Eq ((Polynomial.aeval x) b) 0
  -/
  exact eq_zero_of_ne_zero_of_mul_left_eq_zero h' h
  /-
    🎉 no goals
  -/


theorem ToAdjoin.injective (hx : IsIntegral R x) : Function.Injective (Minpoly.toAdjoin R x) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : IsDomain R
    inst✝³ : Algebra R S
    inst✝² : IsDomain S
    inst✝¹ : NoZeroSMulDivisors R S
    inst✝ : IsIntegrallyClosed R
    x : S
    hx : IsIntegral R x
    ⊢ Function.Injective ⇑(AdjoinRoot.Minpoly.toAdjoin R x)
  -/
  refine (injective_iff_map_eq_zero _).2 fun P₁ hP₁ => ?_
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : IsDomain R
    inst✝³ : Algebra R S
    inst✝² : IsDomain S
    inst✝¹ : NoZeroSMulDivisors R S
    inst✝ : IsIntegrallyClosed R
    x : S
    hx : IsIntegral R x
    P₁ : AdjoinRoot (minpoly R x)
    hP₁ : Eq ((AdjoinRoot.Minpoly.toAdjoin R x) P₁) 0
    ⊢ Eq P₁ 0
  -/
  obtain ⟨P, rfl⟩ := mk_surjective P₁
  rwa [Minpoly.toAdjoin_apply', liftHom_mk, ← Subalgebra.coe_eq_zero, aeval_subalgebra_coe,
    isIntegrallyClosed_dvd_iff hx, ← AdjoinRoot.mk_eq_zero] at hP₁


/-- The algebra isomorphism `AdjoinRoot (minpoly R x) ≃ₐ[R] adjoin R x` -/
@[simps!]
def equivAdjoin (hx : IsIntegral R x) : AdjoinRoot (minpoly R x) ≃ₐ[R] adjoin R ({x} : Set S) :=
  AlgEquiv.ofBijective (Minpoly.toAdjoin R x)
    ⟨minpoly.ToAdjoin.injective hx, Minpoly.toAdjoin.surjective R x⟩


/-- The `PowerBasis` of `adjoin R {x}` given by `x`. See `Algebra.adjoin.powerBasis` for a version
over a field. -/
def _root_.Algebra.adjoin.powerBasis' (hx : IsIntegral R x) :
    PowerBasis R (Algebra.adjoin R ({x} : Set S)) :=
  PowerBasis.map (AdjoinRoot.powerBasis' (minpoly.monic hx)) (minpoly.equivAdjoin hx)


@[simp]
theorem _root_.Algebra.adjoin.powerBasis'_dim (hx : IsIntegral R x) :
    (Algebra.adjoin.powerBasis' hx).dim = (minpoly R x).natDegree := rfl


@[simp]
theorem _root_.Algebra.adjoin.powerBasis'_gen (hx : IsIntegral R x) :
    (adjoin.powerBasis' hx).gen = ⟨x, SetLike.mem_coe.1 <| subset_adjoin <| mem_singleton x⟩ := by
  rw [Algebra.adjoin.powerBasis', PowerBasis.map_gen, AdjoinRoot.powerBasis'_gen, equivAdjoin,
    AlgEquiv.ofBijective_apply, Minpoly.toAdjoin, liftHom_root]


/-- The power basis given by `x` if `B.gen ∈ adjoin R {x}`. -/
noncomputable def _root_.PowerBasis.ofGenMemAdjoin' (B : PowerBasis R S) (hint : IsIntegral R x)
    (hx : B.gen ∈ adjoin R ({x} : Set S)) : PowerBasis R S :=
  (Algebra.adjoin.powerBasis' hint).map <|
    (Subalgebra.equivOfEq _ _ <| PowerBasis.adjoin_eq_top_of_gen_mem_adjoin hx).trans
      Subalgebra.topEquiv


@[simp]
theorem _root_.PowerBasis.ofGenMemAdjoin'_dim (B : PowerBasis R S) (hint : IsIntegral R x)
    (hx : B.gen ∈ adjoin R ({x} : Set S)) :
    (B.ofGenMemAdjoin' hint hx).dim = (minpoly R x).natDegree := rfl


@[simp]
theorem _root_.PowerBasis.ofGenMemAdjoin'_gen (B : PowerBasis R S) (hint : IsIntegral R x)
    (hx : B.gen ∈ adjoin R ({x} : Set S)) :
    (B.ofGenMemAdjoin' hint hx).gen = x := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : CommRing S
    inst✝⁴ : IsDomain R
    inst✝³ : Algebra R S
    inst✝² : IsDomain S
    inst✝¹ : NoZeroSMulDivisors R S
    inst✝ : IsIntegrallyClosed R
    x : S
    B : PowerBasis R S
    hint : IsIntegral R x
    hx : Membership.mem (Algebra.adjoin R (Singleton.singleton x)) B.gen
    ⊢ Eq (B.ofGenMemAdjoin' hint hx).gen x
  -/
  simp [PowerBasis.ofGenMemAdjoin']
  /-
    🎉 no goals
  -/


instance : Algebra A (integralClosure A L) := Subalgebra.algebra (integralClosure A L)

instance : SMul A (integralClosure A L) := Algebra.toSMul

instance : IsScalarTower A ((integralClosure A L)) L :=
  IsScalarTower.subalgebra' A L L (integralClosure A L)


/-- The minimal polynomial of `x : L` over `K` agrees with its minimal polynomial over the
integrally closed subring `A`. -/
theorem ofSubring (x : integralClosure A L) :
    Polynomial.map (algebraMap A K) (minpoly A x) = minpoly K (x : L) :=
  eq_comm.mpr (isIntegrallyClosed_eq_field_fractions K L (IsIntegralClosure.isIntegral A L x))


