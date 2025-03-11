theorem integralClosure.mem_lifts_of_monic_of_dvd_map {f : R[X]} (hf : f.Monic) {g : K[X]}
    (hg : g.Monic) (hd : g ∣ f.map (algebraMap R K)) :
    g ∈ lifts (algebraMap (integralClosure R K) K) := by
  have := mem_lift_of_splits_of_roots_mem_range (integralClosure R g.SplittingField)
    ((splits_id_iff_splits _).2 <| SplittingField.splits g) (hg.map _) fun a ha =>
      (SetLike.ext_iff.mp (integralClosure R g.SplittingField).range_algebraMap _).mpr <|
        roots_mem_integralClosure hf ?_
    /-
      case refine_2
      R : Type u_1
      inst✝² : CommRing R
      K : Type u_2
      inst✝¹ : Field K
      inst✝ : Algebra R K
      f : Polynomial R
      hf : f.Monic
      g : Polynomial K
      hg : g.Monic
      hd : Dvd.dvd g (Polynomial.map (algebraMap R K) f)
      this : Membership.mem (Polynomial.lifts (algebraMap (Subtype fun x => Membersh …
      ⊢ Membership.mem (Polynomial.lifts (algebraMap (Subtype fun x => Membership.me …
    -/
  · rw [lifts_iff_coeff_lifts, ← RingHom.coe_range, Subalgebra.range_algebraMap] at this
    /-
      case refine_2
      R : Type u_1
      inst✝² : CommRing R
      K : Type u_2
      inst✝¹ : Field K
      inst✝ : Algebra R K
      f : Polynomial R
      hf : f.Monic
      g : Polynomial K
      hg : g.Monic
      hd : Dvd.dvd g (Polynomial.map (algebraMap R K) f)
      this : ∀ (n : Nat), Membership.mem (↑(integralClosure R g.SplittingField).toSu …
      ⊢ Membership.mem (Polynomial.lifts (algebraMap (Subtype fun x => Membership.me …
    -/
    refine (lifts_iff_coeff_lifts _).2 fun n => ?_
    /-
      case refine_2
      R : Type u_1
      inst✝² : CommRing R
      K : Type u_2
      inst✝¹ : Field K
      inst✝ : Algebra R K
      f : Polynomial R
      hf : f.Monic
      g : Polynomial K
      hg : g.Monic
      hd : Dvd.dvd g (Polynomial.map (algebraMap R K) f)
      this : ∀ (n : Nat), Membership.mem (↑(integralClosure R g.SplittingField).toSu …
      n : Nat
      ⊢ Membership.mem (Set.range ⇑(algebraMap (Subtype fun x => Membership.mem (int …
    -/
    rw [← RingHom.coe_range, Subalgebra.range_algebraMap]
    /-
      case refine_2
      R : Type u_1
      inst✝² : CommRing R
      K : Type u_2
      inst✝¹ : Field K
      inst✝ : Algebra R K
      f : Polynomial R
      hf : f.Monic
      g : Polynomial K
      hg : g.Monic
      hd : Dvd.dvd g (Polynomial.map (algebraMap R K) f)
      this : ∀ (n : Nat), Membership.mem (↑(integralClosure R g.SplittingField).toSu …
      n : Nat
      ⊢ Membership.mem (↑(integralClosure R K).toSubring) (g.coeff n)
    -/
    obtain ⟨p, hp, he⟩ := SetLike.mem_coe.mp (this n); use p, hp
    /-
      case right
      R : Type u_1
      inst✝² : CommRing R
      K : Type u_2
      inst✝¹ : Field K
      inst✝ : Algebra R K
      f : Polynomial R
      hf : f.Monic
      g : Polynomial K
      hg : g.Monic
      hd : Dvd.dvd g (Polynomial.map (algebraMap R K) f)
      this : ∀ (n : Nat), Membership.mem (↑(integralClosure R g.SplittingField).toSu …
      n : Nat
      p : Polynomial R
      hp : p.Monic
      he : Eq (Polynomial.eval₂ (algebraMap R g.SplittingField) ((Polynomial.map (al …
      ⊢ Eq (Polynomial.eval₂ (algebraMap R K) (g.coeff n) p) 0
    -/
    rw [IsScalarTower.algebraMap_eq R K, coeff_map, ← eval₂_map, eval₂_at_apply] at he
    /-
      case right
      R : Type u_1
      inst✝² : CommRing R
      K : Type u_2
      inst✝¹ : Field K
      inst✝ : Algebra R K
      f : Polynomial R
      hf : f.Monic
      g : Polynomial K
      hg : g.Monic
      hd : Dvd.dvd g (Polynomial.map (algebraMap R K) f)
      this : ∀ (n : Nat), Membership.mem (↑(integralClosure R g.SplittingField).toSu …
      n : Nat
      p : Polynomial R
      hp : p.Monic
      he : Eq ((algebraMap K g.SplittingField) (Polynomial.eval (g.coeff n) (Polynom …
      ⊢ Eq (Polynomial.eval₂ (algebraMap R K) (g.coeff n) p) 0
    -/
    rw [eval₂_eq_eval_map]; apply (injective_iff_map_eq_zero _).1 _ _ he
    /-
      R : Type u_1
      inst✝² : CommRing R
      K : Type u_2
      inst✝¹ : Field K
      inst✝ : Algebra R K
      f : Polynomial R
      hf : f.Monic
      g : Polynomial K
      hg : g.Monic
      hd : Dvd.dvd g (Polynomial.map (algebraMap R K) f)
      this : ∀ (n : Nat), Membership.mem (↑(integralClosure R g.SplittingField).toSu …
      n : Nat
      p : Polynomial R
      hp : p.Monic
      he : Eq ((algebraMap K g.SplittingField) (Polynomial.eval (g.coeff n) (Polynom …
      ⊢ Function.Injective ⇑(algebraMap K g.SplittingField)
    -/
    apply RingHom.injective
    /-
      🎉 no goals
    -/
  /-
    case refine_1
    R : Type u_1
    inst✝² : CommRing R
    K : Type u_2
    inst✝¹ : Field K
    inst✝ : Algebra R K
    f : Polynomial R
    hf : f.Monic
    g : Polynomial K
    hg : g.Monic
    hd : Dvd.dvd g (Polynomial.map (algebraMap R K) f)
    a : g.SplittingField
    ha : Membership.mem (Polynomial.map (algebraMap K g.SplittingField) g).roots a
    ⊢ Membership.mem (f.aroots g.SplittingField) a
  -/
  rw [aroots_def, IsScalarTower.algebraMap_eq R K _, ← map_map]
  /-
    case refine_1
    R : Type u_1
    inst✝² : CommRing R
    K : Type u_2
    inst✝¹ : Field K
    inst✝ : Algebra R K
    f : Polynomial R
    hf : f.Monic
    g : Polynomial K
    hg : g.Monic
    hd : Dvd.dvd g (Polynomial.map (algebraMap R K) f)
    a : g.SplittingField
    ha : Membership.mem (Polynomial.map (algebraMap K g.SplittingField) g).roots a
    ⊢ Membership.mem (Polynomial.map (algebraMap K g.SplittingField) (Polynomial.m …
  -/
  refine Multiset.mem_of_le (roots.le_of_dvd ((hf.map _).map _).ne_zero ?_) ha
  /-
    case refine_1
    R : Type u_1
    inst✝² : CommRing R
    K : Type u_2
    inst✝¹ : Field K
    inst✝ : Algebra R K
    f : Polynomial R
    hf : f.Monic
    g : Polynomial K
    hg : g.Monic
    hd : Dvd.dvd g (Polynomial.map (algebraMap R K) f)
    a : g.SplittingField
    ha : Membership.mem (Polynomial.map (algebraMap K g.SplittingField) g).roots a
    ⊢ Dvd.dvd (Polynomial.map (algebraMap K g.SplittingField) g) (Polynomial.map ( …
  -/
  exact map_dvd (algebraMap K g.SplittingField) hd
  /-
    🎉 no goals
  -/


/-- If `K = Frac(R)` and `g : K[X]` divides a monic polynomial with coefficients in `R`, then
    `g * (C g.leadingCoeff⁻¹)` has coefficients in `R` -/
theorem IsIntegrallyClosed.eq_map_mul_C_of_dvd [IsIntegrallyClosed R] {f : R[X]} (hf : f.Monic)
    {g : K[X]} (hg : g ∣ f.map (algebraMap R K)) :
    ∃ g' : R[X], g'.map (algebraMap R K) * (C <| leadingCoeff g) = g := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsIntegrallyClosed R
    f : Polynomial R
    hf : f.Monic
    g : Polynomial K
    hg : Dvd.dvd g (Polynomial.map (algebraMap R K) f)
    ⊢ Exists fun g' => Eq (HMul.hMul (Polynomial.map (algebraMap R K) g') (Polynom …
  -/
  have g_ne_0 : g ≠ 0 := ne_zero_of_dvd_ne_zero (Monic.ne_zero <| hf.map (algebraMap R K)) hg
  suffices lem : ∃ g' : R[X], g'.map (algebraMap R K) = g * C g.leadingCoeff⁻¹ by
    obtain ⟨g', hg'⟩ := lem
    use g'
    rw [hg', mul_assoc, ← C_mul, inv_mul_cancel₀ (leadingCoeff_ne_zero.mpr g_ne_0), C_1, mul_one]
  have g_mul_dvd : g * C g.leadingCoeff⁻¹ ∣ f.map (algebraMap R K) := by
    rwa [Associated.dvd_iff_dvd_left (show Associated (g * C g.leadingCoeff⁻¹) g from _)]
    rw [associated_mul_isUnit_left_iff]
    exact isUnit_C.mpr (inv_ne_zero <| leadingCoeff_ne_zero.mpr g_ne_0).isUnit
  let algeq :=
    (Subalgebra.equivOfEq _ _ <| integralClosure_eq_bot R _).trans
      (Algebra.botEquivOfInjective <| IsFractionRing.injective R <| K)
  have :
    (algebraMap R _).comp algeq.toAlgHom.toRingHom = (integralClosure R _).toSubring.subtype := by
    ext x; (conv_rhs => rw [← algeq.symm_apply_apply x]); rfl
  have H :=
    (mem_lifts _).1
      (integralClosure.mem_lifts_of_monic_of_dvd_map K hf (monic_mul_leadingCoeff_inv g_ne_0)
        g_mul_dvd)
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsIntegrallyClosed R
    f : Polynomial R
    hf : f.Monic
    g : Polynomial K
    hg : Dvd.dvd g (Polynomial.map (algebraMap R K) f)
    g_ne_0 : Ne g 0
    g_mul_dvd : Dvd.dvd (HMul.hMul g (Polynomial.C (Inv.inv g.leadingCoeff))) (Pol …
    algeq : AlgEquiv R (Subtype fun x => Membership.mem (integralClosure R K) x) R …
    this : Eq ((algebraMap R K).comp (↑algeq).toRingHom) (integralClosure R K).toS …
    H : Exists fun q => Eq (Polynomial.map (algebraMap (Subtype fun x => Membershi …
    ⊢ Exists fun g' => Eq (Polynomial.map (algebraMap R K) g') (HMul.hMul g (Polyn …
  -/
  refine ⟨map algeq.toAlgHom.toRingHom ?_, ?_⟩
    /-
      case refine_1
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsIntegrallyClosed R
      f : Polynomial R
      hf : f.Monic
      g : Polynomial K
      hg : Dvd.dvd g (Polynomial.map (algebraMap R K) f)
      g_ne_0 : Ne g 0
      g_mul_dvd : Dvd.dvd (HMul.hMul g (Polynomial.C (Inv.inv g.leadingCoeff))) (Pol …
      algeq : AlgEquiv R (Subtype fun x => Membership.mem (integralClosure R K) x) R …
      this : Eq ((algebraMap R K).comp (↑algeq).toRingHom) (integralClosure R K).toS …
      H : Exists fun q => Eq (Polynomial.map (algebraMap (Subtype fun x => Membershi …
      ⊢ Polynomial (Subtype fun x => Membership.mem (integralClosure R K) x)
    -/
  · use! Classical.choose H
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsIntegrallyClosed R
      f : Polynomial R
      hf : f.Monic
      g : Polynomial K
      hg : Dvd.dvd g (Polynomial.map (algebraMap R K) f)
      g_ne_0 : Ne g 0
      g_mul_dvd : Dvd.dvd (HMul.hMul g (Polynomial.C (Inv.inv g.leadingCoeff))) (Pol …
      algeq : AlgEquiv R (Subtype fun x => Membership.mem (integralClosure R K) x) R …
      this : Eq ((algebraMap R K).comp (↑algeq).toRingHom) (integralClosure R K).toS …
      H : Exists fun q => Eq (Polynomial.map (algebraMap (Subtype fun x => Membershi …
      ⊢ Eq (Polynomial.map (algebraMap R K) (Polynomial.map (↑algeq).toRingHom (Clas …
    -/
  · rw [map_map, this]
    /-
      case refine_2
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsIntegrallyClosed R
      f : Polynomial R
      hf : f.Monic
      g : Polynomial K
      hg : Dvd.dvd g (Polynomial.map (algebraMap R K) f)
      g_ne_0 : Ne g 0
      g_mul_dvd : Dvd.dvd (HMul.hMul g (Polynomial.C (Inv.inv g.leadingCoeff))) (Pol …
      algeq : AlgEquiv R (Subtype fun x => Membership.mem (integralClosure R K) x) R …
      this : Eq ((algebraMap R K).comp (↑algeq).toRingHom) (integralClosure R K).toS …
      H : Exists fun q => Eq (Polynomial.map (algebraMap (Subtype fun x => Membershi …
      ⊢ Eq (Polynomial.map (integralClosure R K).toSubring.subtype (Classical.choose …
    -/
    exact Classical.choose_spec H
    /-
      🎉 no goals
    -/


theorem IsPrimitive.isUnit_iff_isUnit_map_of_injective : IsUnit f ↔ IsUnit (map φ f) := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    S : Type u_2
    inst✝¹ : CommRing S
    inst✝ : IsDomain S
    φ : RingHom R S
    hinj : Function.Injective ⇑φ
    f : Polynomial R
    hf : f.IsPrimitive
    ⊢ Iff (IsUnit f) (IsUnit (Polynomial.map φ f))
  -/
  refine ⟨(mapRingHom φ).isUnit_map, fun h => ?_⟩
  /-
    R : Type u_1
    inst✝² : CommRing R
    S : Type u_2
    inst✝¹ : CommRing S
    inst✝ : IsDomain S
    φ : RingHom R S
    hinj : Function.Injective ⇑φ
    f : Polynomial R
    hf : f.IsPrimitive
    h : IsUnit (Polynomial.map φ f)
    ⊢ IsUnit f
  -/
  rcases isUnit_iff.1 h with ⟨_, ⟨u, rfl⟩, hu⟩
  /-
    case intro.intro.intro
    R : Type u_1
    inst✝² : CommRing R
    S : Type u_2
    inst✝¹ : CommRing S
    inst✝ : IsDomain S
    φ : RingHom R S
    hinj : Function.Injective ⇑φ
    f : Polynomial R
    hf : f.IsPrimitive
    h : IsUnit (Polynomial.map φ f)
    u : Units S
    hu : Eq (Polynomial.C ↑u) (Polynomial.map φ f)
    ⊢ IsUnit f
  -/
  have hdeg := degree_C u.ne_zero
  /-
    case intro.intro.intro
    R : Type u_1
    inst✝² : CommRing R
    S : Type u_2
    inst✝¹ : CommRing S
    inst✝ : IsDomain S
    φ : RingHom R S
    hinj : Function.Injective ⇑φ
    f : Polynomial R
    hf : f.IsPrimitive
    h : IsUnit (Polynomial.map φ f)
    u : Units S
    hu : Eq (Polynomial.C ↑u) (Polynomial.map φ f)
    hdeg : Eq (Polynomial.C ↑u).degree 0
    ⊢ IsUnit f
  -/
  rw [hu, degree_map_eq_of_injective hinj] at hdeg
  /-
    case intro.intro.intro
    R : Type u_1
    inst✝² : CommRing R
    S : Type u_2
    inst✝¹ : CommRing S
    inst✝ : IsDomain S
    φ : RingHom R S
    hinj : Function.Injective ⇑φ
    f : Polynomial R
    hf : f.IsPrimitive
    h : IsUnit (Polynomial.map φ f)
    u : Units S
    hu : Eq (Polynomial.C ↑u) (Polynomial.map φ f)
    hdeg : Eq f.degree 0
    ⊢ IsUnit f
  -/
  rw [eq_C_of_degree_eq_zero hdeg] at hf ⊢
  /-
    case intro.intro.intro
    R : Type u_1
    inst✝² : CommRing R
    S : Type u_2
    inst✝¹ : CommRing S
    inst✝ : IsDomain S
    φ : RingHom R S
    hinj : Function.Injective ⇑φ
    f : Polynomial R
    hf : (Polynomial.C (f.coeff 0)).IsPrimitive
    h : IsUnit (Polynomial.map φ f)
    u : Units S
    hu : Eq (Polynomial.C ↑u) (Polynomial.map φ f)
    hdeg : Eq f.degree 0
    ⊢ IsUnit (Polynomial.C (f.coeff 0))
  -/
  exact isUnit_C.mpr (isPrimitive_iff_isUnit_of_C_dvd.mp hf (f.coeff 0) dvd_rfl)
  /-
    🎉 no goals
  -/


theorem IsPrimitive.irreducible_of_irreducible_map_of_injective (h_irr : Irreducible (map φ f)) :
    Irreducible f := by
  refine
    ⟨fun h => h_irr.not_unit (IsUnit.map (mapRingHom φ) h), fun a b h =>
      (h_irr.isUnit_or_isUnit <| by rw [h, Polynomial.map_mul]).imp ?_ ?_⟩
  /-
    case refine_1
    R : Type u_1
    inst✝² : CommRing R
    S : Type u_2
    inst✝¹ : CommRing S
    inst✝ : IsDomain S
    φ : RingHom R S
    hinj : Function.Injective ⇑φ
    f : Polynomial R
    hf : f.IsPrimitive
    h_irr : Irreducible (Polynomial.map φ f)
    a b : Polynomial R
    h : Eq f (HMul.hMul a b)
    ⊢ IsUnit (Polynomial.map φ a) → IsUnit a
  -/
  all_goals apply ((isPrimitive_of_dvd hf _).isUnit_iff_isUnit_map_of_injective hinj).mpr
  /-
    R : Type u_1
    inst✝² : CommRing R
    S : Type u_2
    inst✝¹ : CommRing S
    inst✝ : IsDomain S
    φ : RingHom R S
    hinj : Function.Injective ⇑φ
    f : Polynomial R
    hf : f.IsPrimitive
    h_irr : Irreducible (Polynomial.map φ f)
    a b : Polynomial R
    h : Eq f (HMul.hMul a b)
    ⊢ Dvd.dvd a f
  -/
  exacts [Dvd.intro _ h.symm, Dvd.intro_left _ h.symm]
  /-
    🎉 no goals
  -/


theorem IsPrimitive.isUnit_iff_isUnit_map {p : R[X]} (hp : p.IsPrimitive) :
    IsUnit p ↔ IsUnit (p.map (algebraMap R K)) :=
  hp.isUnit_iff_isUnit_map_of_injective (IsFractionRing.injective _ _)


/-- **Gauss's Lemma** for integrally closed domains states that a monic polynomial is irreducible
  iff it is irreducible in the fraction field. -/
theorem Monic.irreducible_iff_irreducible_map_fraction_map [IsIntegrallyClosed R] {p : R[X]}
    (h : p.Monic) : Irreducible p ↔ Irreducible (p.map <| algebraMap R K) := by
  /- The ← direction follows from `IsPrimitive.irreducible_of_irreducible_map_of_injective`.
       For the → direction, it is enough to show that if `(p.map <| algebraMap R K) = a * b` and
       `a` is not a unit then `b` is a unit -/
  refine
    ⟨fun hp =>
      irreducible_iff.mpr
        ⟨hp.not_unit.imp h.isPrimitive.isUnit_iff_isUnit_map.mpr, fun a b H =>
          or_iff_not_imp_left.mpr fun hₐ => ?_⟩,
      fun hp =>
      h.isPrimitive.irreducible_of_irreducible_map_of_injective (IsFractionRing.injective R K) hp⟩
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsIntegrallyClosed R
    p : Polynomial R
    h : p.Monic
    hp : Irreducible p
    a b : Polynomial K
    H : Eq (Polynomial.map (algebraMap R K) p) (HMul.hMul a b)
    hₐ : Not (IsUnit a)
    ⊢ IsUnit b
  -/
  obtain ⟨a', ha⟩ := eq_map_mul_C_of_dvd K h (dvd_of_mul_right_eq b H.symm)
  /-
    case intro
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsIntegrallyClosed R
    p : Polynomial R
    h : p.Monic
    hp : Irreducible p
    a b : Polynomial K
    H : Eq (Polynomial.map (algebraMap R K) p) (HMul.hMul a b)
    hₐ : Not (IsUnit a)
    a' : Polynomial R
    ha : Eq (HMul.hMul (Polynomial.map (algebraMap R K) a') (Polynomial.C a.leadin …
    ⊢ IsUnit b
  -/
  obtain ⟨b', hb⟩ := eq_map_mul_C_of_dvd K h (dvd_of_mul_left_eq a H.symm)
  have : a.leadingCoeff * b.leadingCoeff = 1 := by
    rw [← leadingCoeff_mul, ← H, Monic.leadingCoeff (h.map <| algebraMap R K)]
  rw [← ha, ← hb, mul_comm _ (C b.leadingCoeff), mul_assoc, ← mul_assoc (C a.leadingCoeff), ←
    C_mul, this, C_1, one_mul, ← Polynomial.map_mul] at H
  /-
    case intro.intro
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsIntegrallyClosed R
    p : Polynomial R
    h : p.Monic
    hp : Irreducible p
    a b : Polynomial K
    hₐ : Not (IsUnit a)
    a' : Polynomial R
    ha : Eq (HMul.hMul (Polynomial.map (algebraMap R K) a') (Polynomial.C a.leadin …
    b' : Polynomial R
    H : Eq (Polynomial.map (algebraMap R K) p) (Polynomial.map (algebraMap R K) (H …
    hb : Eq (HMul.hMul (Polynomial.map (algebraMap R K) b') (Polynomial.C b.leadin …
    this : Eq (HMul.hMul a.leadingCoeff b.leadingCoeff) 1
    ⊢ IsUnit b
  -/
  rw [← hb, ← Polynomial.coe_mapRingHom]
  refine
    IsUnit.mul (IsUnit.map _ (Or.resolve_left (hp.isUnit_or_isUnit ?_) (show ¬IsUnit a' from ?_)))
      (isUnit_iff_exists_inv'.mpr
        -- Porting note(https://github.com/leanprover-community/mathlib4/issues/5073): was `rwa`
        (Exists.intro (C a.leadingCoeff) <| by rw [← C_mul, this, C_1]))
    /-
      case intro.intro.refine_1
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsIntegrallyClosed R
      p : Polynomial R
      h : p.Monic
      hp : Irreducible p
      a b : Polynomial K
      hₐ : Not (IsUnit a)
      a' : Polynomial R
      ha : Eq (HMul.hMul (Polynomial.map (algebraMap R K) a') (Polynomial.C a.leadin …
      b' : Polynomial R
      H : Eq (Polynomial.map (algebraMap R K) p) (Polynomial.map (algebraMap R K) (H …
      hb : Eq (HMul.hMul (Polynomial.map (algebraMap R K) b') (Polynomial.C b.leadin …
      this : Eq (HMul.hMul a.leadingCoeff b.leadingCoeff) 1
      ⊢ Eq p (HMul.hMul a' b')
    -/
  · exact Polynomial.map_injective _ (IsFractionRing.injective R K) H
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsIntegrallyClosed R
      p : Polynomial R
      h : p.Monic
      hp : Irreducible p
      a b : Polynomial K
      hₐ : Not (IsUnit a)
      a' : Polynomial R
      ha : Eq (HMul.hMul (Polynomial.map (algebraMap R K) a') (Polynomial.C a.leadin …
      b' : Polynomial R
      H : Eq (Polynomial.map (algebraMap R K) p) (Polynomial.map (algebraMap R K) (H …
      hb : Eq (HMul.hMul (Polynomial.map (algebraMap R K) b') (Polynomial.C b.leadin …
      this : Eq (HMul.hMul a.leadingCoeff b.leadingCoeff) 1
      ⊢ Not (IsUnit a')
    -/
  · by_contra h_contra
    /-
      case intro.intro.refine_2
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsIntegrallyClosed R
      p : Polynomial R
      h : p.Monic
      hp : Irreducible p
      a b : Polynomial K
      hₐ : Not (IsUnit a)
      a' : Polynomial R
      ha : Eq (HMul.hMul (Polynomial.map (algebraMap R K) a') (Polynomial.C a.leadin …
      b' : Polynomial R
      H : Eq (Polynomial.map (algebraMap R K) p) (Polynomial.map (algebraMap R K) (H …
      hb : Eq (HMul.hMul (Polynomial.map (algebraMap R K) b') (Polynomial.C b.leadin …
      this : Eq (HMul.hMul a.leadingCoeff b.leadingCoeff) 1
      h_contra : IsUnit a'
      ⊢ False
    -/
    refine hₐ ?_
    /-
      case intro.intro.refine_2
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsIntegrallyClosed R
      p : Polynomial R
      h : p.Monic
      hp : Irreducible p
      a b : Polynomial K
      hₐ : Not (IsUnit a)
      a' : Polynomial R
      ha : Eq (HMul.hMul (Polynomial.map (algebraMap R K) a') (Polynomial.C a.leadin …
      b' : Polynomial R
      H : Eq (Polynomial.map (algebraMap R K) p) (Polynomial.map (algebraMap R K) (H …
      hb : Eq (HMul.hMul (Polynomial.map (algebraMap R K) b') (Polynomial.C b.leadin …
      this : Eq (HMul.hMul a.leadingCoeff b.leadingCoeff) 1
      h_contra : IsUnit a'
      ⊢ IsUnit a
    -/
    rw [← ha, ← Polynomial.coe_mapRingHom]
    exact
      IsUnit.mul (IsUnit.map _ h_contra)
        (isUnit_iff_exists_inv.mpr
          -- Porting note(https://github.com/leanprover-community/mathlib4/issues/5073): was `rwa`
          (Exists.intro (C b.leadingCoeff) <| by rw [← C_mul, this, C_1]))


/-- Integrally closed domains are precisely the domains for in which Gauss's lemma holds
    for monic polynomials -/
theorem isIntegrallyClosed_iff' [IsDomain R] :
    IsIntegrallyClosed R ↔
      ∀ p : R[X], p.Monic → (Irreducible p ↔ Irreducible (p.map <| algebraMap R K)) := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsDomain R
    ⊢ Iff (IsIntegrallyClosed R) (∀ (p : Polynomial R), p.Monic → Iff (Irreducible …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDomain R
      ⊢ IsIntegrallyClosed R → ∀ (p : Polynomial R), p.Monic → Iff (Irreducible p) ( …
    -/
  · intro hR p hp; exact Monic.irreducible_iff_irreducible_map_fraction_map hp
                   /-
                     🎉 no goals
                   -/
    /-
      case mpr
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDomain R
      ⊢ (∀ (p : Polynomial R), p.Monic → Iff (Irreducible p) (Irreducible (Polynomia …
    -/
  · intro H
    refine
      (isIntegrallyClosed_iff K).mpr fun {x} hx =>
        RingHom.mem_range.mp <| minpoly.mem_range_of_degree_eq_one R x ?_
    /-
      case mpr
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDomain R
      H : ∀ (p : Polynomial R), p.Monic → Iff (Irreducible p) (Irreducible (Polynomi …
      x : K
      hx : IsIntegral R x
      ⊢ Eq (minpoly R x).degree 1
    -/
    rw [← Monic.degree_map (minpoly.monic hx) (algebraMap R K)]
    apply
      degree_eq_one_of_irreducible_of_root ((H _ <| minpoly.monic hx).mp (minpoly.irreducible hx))
    /-
      case mpr
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsDomain R
      H : ∀ (p : Polynomial R), p.Monic → Iff (Irreducible p) (Irreducible (Polynomi …
      x : K
      hx : IsIntegral R x
      ⊢ (Polynomial.map (algebraMap R K) (minpoly R x)).IsRoot ?m.43386
    -/
    rw [IsRoot, eval_map, ← aeval_def, minpoly.aeval R x]
    /-
      🎉 no goals
    -/


theorem Monic.dvd_of_fraction_map_dvd_fraction_map [IsIntegrallyClosed R] {p q : R[X]}
    (hp : p.Monic) (hq : q.Monic)
    (h : q.map (algebraMap R K) ∣ p.map (algebraMap R K)) : q ∣ p := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsIntegrallyClosed R
    p q : Polynomial R
    hp : p.Monic
    hq : q.Monic
    h : Dvd.dvd (Polynomial.map (algebraMap R K) q) (Polynomial.map (algebraMap R  …
    ⊢ Dvd.dvd q p
  -/
  obtain ⟨r, hr⟩ := h
  /-
    case intro
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsIntegrallyClosed R
    p q : Polynomial R
    hp : p.Monic
    hq : q.Monic
    r : Polynomial K
    hr : Eq (Polynomial.map (algebraMap R K) p) (HMul.hMul (Polynomial.map (algebr …
    ⊢ Dvd.dvd q p
  -/
  obtain ⟨d', hr'⟩ := IsIntegrallyClosed.eq_map_mul_C_of_dvd K hp (dvd_of_mul_left_eq _ hr.symm)
  /-
    case intro.intro
    R : Type u_1
    inst✝⁴ : CommRing R
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra R K
    inst✝¹ : IsFractionRing R K
    inst✝ : IsIntegrallyClosed R
    p q : Polynomial R
    hp : p.Monic
    hq : q.Monic
    r : Polynomial K
    hr : Eq (Polynomial.map (algebraMap R K) p) (HMul.hMul (Polynomial.map (algebr …
    d' : Polynomial R
    hr' : Eq (HMul.hMul (Polynomial.map (algebraMap R K) d') (Polynomial.C r.leadi …
    ⊢ Dvd.dvd q p
  -/
  rw [Monic.leadingCoeff, C_1, mul_one] at hr'
    /-
      case intro.intro
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsIntegrallyClosed R
      p q : Polynomial R
      hp : p.Monic
      hq : q.Monic
      r : Polynomial K
      hr : Eq (Polynomial.map (algebraMap R K) p) (HMul.hMul (Polynomial.map (algebr …
      d' : Polynomial R
      hr' : Eq (Polynomial.map (algebraMap R K) d') r
      ⊢ Dvd.dvd q p
    -/
  · rw [← hr', ← Polynomial.map_mul] at hr
    /-
      case intro.intro
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsIntegrallyClosed R
      p q : Polynomial R
      hp : p.Monic
      hq : q.Monic
      r : Polynomial K
      d' : Polynomial R
      hr : Eq (Polynomial.map (algebraMap R K) p) (Polynomial.map (algebraMap R K) ( …
      hr' : Eq (Polynomial.map (algebraMap R K) d') r
      ⊢ Dvd.dvd q p
    -/
    exact dvd_of_mul_right_eq _ (Polynomial.map_injective _ (IsFractionRing.injective R K) hr.symm)
    /-
      🎉 no goals
    -/
    /-
      case intro.intro
      R : Type u_1
      inst✝⁴ : CommRing R
      K : Type u_2
      inst✝³ : Field K
      inst✝² : Algebra R K
      inst✝¹ : IsFractionRing R K
      inst✝ : IsIntegrallyClosed R
      p q : Polynomial R
      hp : p.Monic
      hq : q.Monic
      r : Polynomial K
      hr : Eq (Polynomial.map (algebraMap R K) p) (HMul.hMul (Polynomial.map (algebr …
      d' : Polynomial R
      hr' : Eq (HMul.hMul (Polynomial.map (algebraMap R K) d') (Polynomial.C r.leadi …
      ⊢ r.Monic
    -/
  · exact Monic.of_mul_monic_left (hq.map (algebraMap R K)) (by simpa [← hr] using hp.map _)
    /-
      🎉 no goals
    -/


theorem Monic.dvd_iff_fraction_map_dvd_fraction_map [IsIntegrallyClosed R] {p q : R[X]}
    (hp : p.Monic) (hq : q.Monic) : q.map (algebraMap R K) ∣ p.map (algebraMap R K) ↔ q ∣ p :=
  ⟨fun h => hp.dvd_of_fraction_map_dvd_fraction_map hq h, fun ⟨a, b⟩ =>
    ⟨a.map (algebraMap R K), b.symm ▸ Polynomial.map_mul (algebraMap R K)⟩⟩


theorem isUnit_or_eq_zero_of_isUnit_integerNormalization_primPart {p : K[X]} (h0 : p ≠ 0)
    (h : IsUnit (integerNormalization R⁰ p).primPart) : IsUnit p := by
  /-
    R : Type u_1
    inst✝⁵ : CommRing R
    K : Type u_2
    inst✝⁴ : Field K
    inst✝³ : Algebra R K
    inst✝² : IsFractionRing R K
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial K
    h0 : Ne p 0
    h : IsUnit (IsLocalization.integerNormalization (nonZeroDivisors R) p).primPart
    ⊢ IsUnit p
  -/
  rcases isUnit_iff.1 h with ⟨_, ⟨u, rfl⟩, hu⟩
  /-
    case intro.intro.intro
    R : Type u_1
    inst✝⁵ : CommRing R
    K : Type u_2
    inst✝⁴ : Field K
    inst✝³ : Algebra R K
    inst✝² : IsFractionRing R K
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial K
    h0 : Ne p 0
    h : IsUnit (IsLocalization.integerNormalization (nonZeroDivisors R) p).primPart
    u : Units R
    hu : Eq (Polynomial.C ↑u) (IsLocalization.integerNormalization (nonZeroDivisor …
    ⊢ IsUnit p
  -/
  obtain ⟨⟨c, c0⟩, hc⟩ := integerNormalization_map_to_map R⁰ p
  /-
    case intro.intro.intro.intro.mk
    R : Type u_1
    inst✝⁵ : CommRing R
    K : Type u_2
    inst✝⁴ : Field K
    inst✝³ : Algebra R K
    inst✝² : IsFractionRing R K
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial K
    h0 : Ne p 0
    h : IsUnit (IsLocalization.integerNormalization (nonZeroDivisors R) p).primPart
    u : Units R
    hu : Eq (Polynomial.C ↑u) (IsLocalization.integerNormalization (nonZeroDivisor …
    c : R
    c0 : Membership.mem (nonZeroDivisors R) c
    hc : Eq (Polynomial.map (algebraMap R K) (IsLocalization.integerNormalization  …
    ⊢ IsUnit p
  -/
  rw [Subtype.coe_mk, Algebra.smul_def, algebraMap_apply] at hc
  /-
    case intro.intro.intro.intro.mk
    R : Type u_1
    inst✝⁵ : CommRing R
    K : Type u_2
    inst✝⁴ : Field K
    inst✝³ : Algebra R K
    inst✝² : IsFractionRing R K
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial K
    h0 : Ne p 0
    h : IsUnit (IsLocalization.integerNormalization (nonZeroDivisors R) p).primPart
    u : Units R
    hu : Eq (Polynomial.C ↑u) (IsLocalization.integerNormalization (nonZeroDivisor …
    c : R
    c0 : Membership.mem (nonZeroDivisors R) c
    hc : Eq (Polynomial.map (algebraMap R K) (IsLocalization.integerNormalization  …
    ⊢ IsUnit p
  -/
  apply isUnit_of_mul_isUnit_right
  rw [← hc, (integerNormalization R⁰ p).eq_C_content_mul_primPart, ← hu, ← RingHom.map_mul,
    isUnit_iff]
  refine
    ⟨algebraMap R K ((integerNormalization R⁰ p).content * ↑u), isUnit_iff_ne_zero.2 fun con => ?_,
      by simp⟩
  /-
    case intro.intro.intro.intro.mk.hu
    R : Type u_1
    inst✝⁵ : CommRing R
    K : Type u_2
    inst✝⁴ : Field K
    inst✝³ : Algebra R K
    inst✝² : IsFractionRing R K
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial K
    h0 : Ne p 0
    h : IsUnit (IsLocalization.integerNormalization (nonZeroDivisors R) p).primPart
    u : Units R
    hu : Eq (Polynomial.C ↑u) (IsLocalization.integerNormalization (nonZeroDivisor …
    c : R
    c0 : Membership.mem (nonZeroDivisors R) c
    hc : Eq (Polynomial.map (algebraMap R K) (IsLocalization.integerNormalization  …
    con : Eq ((algebraMap R K) (HMul.hMul (IsLocalization.integerNormalization (no …
    ⊢ False
  -/
  replace con := (injective_iff_map_eq_zero (algebraMap R K)).1 (IsFractionRing.injective _ _) _ con
  /-
    case intro.intro.intro.intro.mk.hu
    R : Type u_1
    inst✝⁵ : CommRing R
    K : Type u_2
    inst✝⁴ : Field K
    inst✝³ : Algebra R K
    inst✝² : IsFractionRing R K
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial K
    h0 : Ne p 0
    h : IsUnit (IsLocalization.integerNormalization (nonZeroDivisors R) p).primPart
    u : Units R
    hu : Eq (Polynomial.C ↑u) (IsLocalization.integerNormalization (nonZeroDivisor …
    c : R
    c0 : Membership.mem (nonZeroDivisors R) c
    hc : Eq (Polynomial.map (algebraMap R K) (IsLocalization.integerNormalization  …
    con : Eq (HMul.hMul (IsLocalization.integerNormalization (nonZeroDivisors R) p …
    ⊢ False
  -/
  rw [mul_eq_zero, content_eq_zero_iff, IsFractionRing.integerNormalization_eq_zero_iff] at con
  /-
    case intro.intro.intro.intro.mk.hu
    R : Type u_1
    inst✝⁵ : CommRing R
    K : Type u_2
    inst✝⁴ : Field K
    inst✝³ : Algebra R K
    inst✝² : IsFractionRing R K
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial K
    h0 : Ne p 0
    h : IsUnit (IsLocalization.integerNormalization (nonZeroDivisors R) p).primPart
    u : Units R
    hu : Eq (Polynomial.C ↑u) (IsLocalization.integerNormalization (nonZeroDivisor …
    c : R
    c0 : Membership.mem (nonZeroDivisors R) c
    hc : Eq (Polynomial.map (algebraMap R K) (IsLocalization.integerNormalization  …
    con : Or (Eq p 0) (Eq (↑u) 0)
    ⊢ False
  -/
  rcases con with (con | con)
    /-
      case intro.intro.intro.intro.mk.hu.inl
      R : Type u_1
      inst✝⁵ : CommRing R
      K : Type u_2
      inst✝⁴ : Field K
      inst✝³ : Algebra R K
      inst✝² : IsFractionRing R K
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      p : Polynomial K
      h0 : Ne p 0
      h : IsUnit (IsLocalization.integerNormalization (nonZeroDivisors R) p).primPart
      u : Units R
      hu : Eq (Polynomial.C ↑u) (IsLocalization.integerNormalization (nonZeroDivisor …
      c : R
      c0 : Membership.mem (nonZeroDivisors R) c
      hc : Eq (Polynomial.map (algebraMap R K) (IsLocalization.integerNormalization  …
      con : Eq p 0
      ⊢ False
    -/
  · apply h0 con
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.mk.hu.inr
      R : Type u_1
      inst✝⁵ : CommRing R
      K : Type u_2
      inst✝⁴ : Field K
      inst✝³ : Algebra R K
      inst✝² : IsFractionRing R K
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      p : Polynomial K
      h0 : Ne p 0
      h : IsUnit (IsLocalization.integerNormalization (nonZeroDivisors R) p).primPart
      u : Units R
      hu : Eq (Polynomial.C ↑u) (IsLocalization.integerNormalization (nonZeroDivisor …
      c : R
      c0 : Membership.mem (nonZeroDivisors R) c
      hc : Eq (Polynomial.map (algebraMap R K) (IsLocalization.integerNormalization  …
      con : Eq (↑u) 0
      ⊢ False
    -/
  · apply Units.ne_zero _ con
    /-
      🎉 no goals
    -/


/-- **Gauss's Lemma** for GCD domains states that a primitive polynomial is irreducible iff it is
  irreducible in the fraction field. -/
theorem IsPrimitive.irreducible_iff_irreducible_map_fraction_map {p : R[X]} (hp : p.IsPrimitive) :
    Irreducible p ↔ Irreducible (p.map (algebraMap R K)) := by
  -- Porting note: was `(IsFractionRing.injective _ _)`
  refine
    ⟨fun hi => ⟨fun h => hi.not_unit (hp.isUnit_iff_isUnit_map.2 h), fun a b hab => ?_⟩,
      hp.irreducible_of_irreducible_map_of_injective (IsFractionRing.injective R K)⟩
  /-
    R : Type u_1
    inst✝⁵ : CommRing R
    K : Type u_2
    inst✝⁴ : Field K
    inst✝³ : Algebra R K
    inst✝² : IsFractionRing R K
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial R
    hp : p.IsPrimitive
    hi : Irreducible p
    a b : Polynomial K
    hab : Eq (Polynomial.map (algebraMap R K) p) (HMul.hMul a b)
    ⊢ Or (IsUnit a) (IsUnit b)
  -/
  obtain ⟨⟨c, c0⟩, hc⟩ := integerNormalization_map_to_map R⁰ a
  /-
    case intro.mk
    R : Type u_1
    inst✝⁵ : CommRing R
    K : Type u_2
    inst✝⁴ : Field K
    inst✝³ : Algebra R K
    inst✝² : IsFractionRing R K
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial R
    hp : p.IsPrimitive
    hi : Irreducible p
    a b : Polynomial K
    hab : Eq (Polynomial.map (algebraMap R K) p) (HMul.hMul a b)
    c : R
    c0 : Membership.mem (nonZeroDivisors R) c
    hc : Eq (Polynomial.map (algebraMap R K) (IsLocalization.integerNormalization  …
    ⊢ Or (IsUnit a) (IsUnit b)
  -/
  obtain ⟨⟨d, d0⟩, hd⟩ := integerNormalization_map_to_map R⁰ b
  /-
    case intro.mk.intro.mk
    R : Type u_1
    inst✝⁵ : CommRing R
    K : Type u_2
    inst✝⁴ : Field K
    inst✝³ : Algebra R K
    inst✝² : IsFractionRing R K
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial R
    hp : p.IsPrimitive
    hi : Irreducible p
    a b : Polynomial K
    hab : Eq (Polynomial.map (algebraMap R K) p) (HMul.hMul a b)
    c : R
    c0 : Membership.mem (nonZeroDivisors R) c
    hc : Eq (Polynomial.map (algebraMap R K) (IsLocalization.integerNormalization  …
    d : R
    d0 : Membership.mem (nonZeroDivisors R) d
    hd : Eq (Polynomial.map (algebraMap R K) (IsLocalization.integerNormalization  …
    ⊢ Or (IsUnit a) (IsUnit b)
  -/
  rw [Algebra.smul_def, algebraMap_apply, Subtype.coe_mk] at hc hd
  /-
    case intro.mk.intro.mk
    R : Type u_1
    inst✝⁵ : CommRing R
    K : Type u_2
    inst✝⁴ : Field K
    inst✝³ : Algebra R K
    inst✝² : IsFractionRing R K
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial R
    hp : p.IsPrimitive
    hi : Irreducible p
    a b : Polynomial K
    hab : Eq (Polynomial.map (algebraMap R K) p) (HMul.hMul a b)
    c : R
    c0 : Membership.mem (nonZeroDivisors R) c
    hc : Eq (Polynomial.map (algebraMap R K) (IsLocalization.integerNormalization  …
    d : R
    d0 : Membership.mem (nonZeroDivisors R) d
    hd : Eq (Polynomial.map (algebraMap R K) (IsLocalization.integerNormalization  …
    ⊢ Or (IsUnit a) (IsUnit b)
  -/
  rw [mem_nonZeroDivisors_iff_ne_zero] at c0 d0
  /-
    case intro.mk.intro.mk
    R : Type u_1
    inst✝⁵ : CommRing R
    K : Type u_2
    inst✝⁴ : Field K
    inst✝³ : Algebra R K
    inst✝² : IsFractionRing R K
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial R
    hp : p.IsPrimitive
    hi : Irreducible p
    a b : Polynomial K
    hab : Eq (Polynomial.map (algebraMap R K) p) (HMul.hMul a b)
    c : R
    c0 : Ne c 0
    hc : Eq (Polynomial.map (algebraMap R K) (IsLocalization.integerNormalization  …
    d : R
    d0 : Ne d 0
    hd : Eq (Polynomial.map (algebraMap R K) (IsLocalization.integerNormalization  …
    ⊢ Or (IsUnit a) (IsUnit b)
  -/
  have hcd0 : c * d ≠ 0 := mul_ne_zero c0 d0
  /-
    case intro.mk.intro.mk
    R : Type u_1
    inst✝⁵ : CommRing R
    K : Type u_2
    inst✝⁴ : Field K
    inst✝³ : Algebra R K
    inst✝² : IsFractionRing R K
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial R
    hp : p.IsPrimitive
    hi : Irreducible p
    a b : Polynomial K
    hab : Eq (Polynomial.map (algebraMap R K) p) (HMul.hMul a b)
    c : R
    c0 : Ne c 0
    hc : Eq (Polynomial.map (algebraMap R K) (IsLocalization.integerNormalization  …
    d : R
    d0 : Ne d 0
    hd : Eq (Polynomial.map (algebraMap R K) (IsLocalization.integerNormalization  …
    hcd0 : Ne (HMul.hMul c d) 0
    ⊢ Or (IsUnit a) (IsUnit b)
  -/
  rw [Ne, ← C_eq_zero] at hcd0
  have h1 : C c * C d * p = integerNormalization R⁰ a * integerNormalization R⁰ b := by
    apply map_injective (algebraMap R K) (IsFractionRing.injective _ _) _
    rw [Polynomial.map_mul, Polynomial.map_mul, Polynomial.map_mul, hc, hd, map_C, map_C, hab]
    ring
  obtain ⟨u, hu⟩ :
    Associated (c * d)
      (content (integerNormalization R⁰ a) * content (integerNormalization R⁰ b)) := by
    rw [← dvd_dvd_iff_associated, ← normalize_eq_normalize_iff, normalize.map_mul,
      normalize.map_mul, normalize_content, normalize_content, ←
      mul_one (normalize c * normalize d), ← hp.content_eq_one, ← content_C, ← content_C, ←
      content_mul, ← content_mul, ← content_mul, h1]
  rw [← RingHom.map_mul, eq_comm, (integerNormalization R⁰ a).eq_C_content_mul_primPart,
    (integerNormalization R⁰ b).eq_C_content_mul_primPart, mul_assoc, mul_comm _ (C _ * _), ←
    mul_assoc, ← mul_assoc, ← RingHom.map_mul, ← hu, RingHom.map_mul, mul_assoc, mul_assoc, ←
    mul_assoc (C (u : R))] at h1
  have h0 : a ≠ 0 ∧ b ≠ 0 := by
    classical
    rw [Ne, Ne, ← not_or, ← mul_eq_zero, ← hab]
    intro con
    apply hp.ne_zero (map_injective (algebraMap R K) (IsFractionRing.injective _ _) _)
    simp [con]
  /-
    case intro.mk.intro.mk.intro
    R : Type u_1
    inst✝⁵ : CommRing R
    K : Type u_2
    inst✝⁴ : Field K
    inst✝³ : Algebra R K
    inst✝² : IsFractionRing R K
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p : Polynomial R
    hp : p.IsPrimitive
    hi : Irreducible p
    a b : Polynomial K
    hab : Eq (Polynomial.map (algebraMap R K) p) (HMul.hMul a b)
    c : R
    c0 : Ne c 0
    hc : Eq (Polynomial.map (algebraMap R K) (IsLocalization.integerNormalization  …
    d : R
    d0 : Ne d 0
    hd : Eq (Polynomial.map (algebraMap R K) (IsLocalization.integerNormalization  …
    hcd0 : Not (Eq (Polynomial.C (HMul.hMul c d)) 0)
    u : Units R
    h1 : Eq (HMul.hMul (Polynomial.C (HMul.hMul c d)) (HMul.hMul (HMul.hMul (Polyn …
    hu : Eq (HMul.hMul (HMul.hMul c d) ↑u) (HMul.hMul (IsLocalization.integerNorma …
    h0 : And (Ne a 0) (Ne b 0)
    ⊢ Or (IsUnit a) (IsUnit b)
  -/
  rcases hi.isUnit_or_isUnit (mul_left_cancel₀ hcd0 h1).symm with (h | h)
    /-
      case intro.mk.intro.mk.intro.inl
      R : Type u_1
      inst✝⁵ : CommRing R
      K : Type u_2
      inst✝⁴ : Field K
      inst✝³ : Algebra R K
      inst✝² : IsFractionRing R K
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      p : Polynomial R
      hp : p.IsPrimitive
      hi : Irreducible p
      a b : Polynomial K
      hab : Eq (Polynomial.map (algebraMap R K) p) (HMul.hMul a b)
      c : R
      c0 : Ne c 0
      hc : Eq (Polynomial.map (algebraMap R K) (IsLocalization.integerNormalization  …
      d : R
      d0 : Ne d 0
      hd : Eq (Polynomial.map (algebraMap R K) (IsLocalization.integerNormalization  …
      hcd0 : Not (Eq (Polynomial.C (HMul.hMul c d)) 0)
      u : Units R
      h1 : Eq (HMul.hMul (Polynomial.C (HMul.hMul c d)) (HMul.hMul (HMul.hMul (Polyn …
      hu : Eq (HMul.hMul (HMul.hMul c d) ↑u) (HMul.hMul (IsLocalization.integerNorma …
      h0 : And (Ne a 0) (Ne b 0)
      h : IsUnit (HMul.hMul (Polynomial.C ↑u) (IsLocalization.integerNormalization ( …
      ⊢ Or (IsUnit a) (IsUnit b)
    -/
  · right
    apply
      isUnit_or_eq_zero_of_isUnit_integerNormalization_primPart h0.2
        (isUnit_of_mul_isUnit_right h)
    /-
      case intro.mk.intro.mk.intro.inr
      R : Type u_1
      inst✝⁵ : CommRing R
      K : Type u_2
      inst✝⁴ : Field K
      inst✝³ : Algebra R K
      inst✝² : IsFractionRing R K
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      p : Polynomial R
      hp : p.IsPrimitive
      hi : Irreducible p
      a b : Polynomial K
      hab : Eq (Polynomial.map (algebraMap R K) p) (HMul.hMul a b)
      c : R
      c0 : Ne c 0
      hc : Eq (Polynomial.map (algebraMap R K) (IsLocalization.integerNormalization  …
      d : R
      d0 : Ne d 0
      hd : Eq (Polynomial.map (algebraMap R K) (IsLocalization.integerNormalization  …
      hcd0 : Not (Eq (Polynomial.C (HMul.hMul c d)) 0)
      u : Units R
      h1 : Eq (HMul.hMul (Polynomial.C (HMul.hMul c d)) (HMul.hMul (HMul.hMul (Polyn …
      hu : Eq (HMul.hMul (HMul.hMul c d) ↑u) (HMul.hMul (IsLocalization.integerNorma …
      h0 : And (Ne a 0) (Ne b 0)
      h : IsUnit (IsLocalization.integerNormalization (nonZeroDivisors R) a).primPart
      ⊢ Or (IsUnit a) (IsUnit b)
    -/
  · left
    /-
      case intro.mk.intro.mk.intro.inr.h
      R : Type u_1
      inst✝⁵ : CommRing R
      K : Type u_2
      inst✝⁴ : Field K
      inst✝³ : Algebra R K
      inst✝² : IsFractionRing R K
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      p : Polynomial R
      hp : p.IsPrimitive
      hi : Irreducible p
      a b : Polynomial K
      hab : Eq (Polynomial.map (algebraMap R K) p) (HMul.hMul a b)
      c : R
      c0 : Ne c 0
      hc : Eq (Polynomial.map (algebraMap R K) (IsLocalization.integerNormalization  …
      d : R
      d0 : Ne d 0
      hd : Eq (Polynomial.map (algebraMap R K) (IsLocalization.integerNormalization  …
      hcd0 : Not (Eq (Polynomial.C (HMul.hMul c d)) 0)
      u : Units R
      h1 : Eq (HMul.hMul (Polynomial.C (HMul.hMul c d)) (HMul.hMul (HMul.hMul (Polyn …
      hu : Eq (HMul.hMul (HMul.hMul c d) ↑u) (HMul.hMul (IsLocalization.integerNorma …
      h0 : And (Ne a 0) (Ne b 0)
      h : IsUnit (IsLocalization.integerNormalization (nonZeroDivisors R) a).primPart
      ⊢ IsUnit a
    -/
    apply isUnit_or_eq_zero_of_isUnit_integerNormalization_primPart h0.1 h
    /-
      🎉 no goals
    -/


theorem IsPrimitive.dvd_of_fraction_map_dvd_fraction_map {p q : R[X]} (hp : p.IsPrimitive)
    (hq : q.IsPrimitive) (h_dvd : p.map (algebraMap R K) ∣ q.map (algebraMap R K)) : p ∣ q := by
  /-
    R : Type u_1
    inst✝⁵ : CommRing R
    K : Type u_2
    inst✝⁴ : Field K
    inst✝³ : Algebra R K
    inst✝² : IsFractionRing R K
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p q : Polynomial R
    hp : p.IsPrimitive
    hq : q.IsPrimitive
    h_dvd : Dvd.dvd (Polynomial.map (algebraMap R K) p) (Polynomial.map (algebraMa …
    ⊢ Dvd.dvd p q
  -/
  rcases h_dvd with ⟨r, hr⟩
  /-
    case intro
    R : Type u_1
    inst✝⁵ : CommRing R
    K : Type u_2
    inst✝⁴ : Field K
    inst✝³ : Algebra R K
    inst✝² : IsFractionRing R K
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p q : Polynomial R
    hp : p.IsPrimitive
    hq : q.IsPrimitive
    r : Polynomial K
    hr : Eq (Polynomial.map (algebraMap R K) q) (HMul.hMul (Polynomial.map (algebr …
    ⊢ Dvd.dvd p q
  -/
  obtain ⟨⟨s, s0⟩, hs⟩ := integerNormalization_map_to_map R⁰ r
  /-
    case intro.intro.mk
    R : Type u_1
    inst✝⁵ : CommRing R
    K : Type u_2
    inst✝⁴ : Field K
    inst✝³ : Algebra R K
    inst✝² : IsFractionRing R K
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p q : Polynomial R
    hp : p.IsPrimitive
    hq : q.IsPrimitive
    r : Polynomial K
    hr : Eq (Polynomial.map (algebraMap R K) q) (HMul.hMul (Polynomial.map (algebr …
    s : R
    s0 : Membership.mem (nonZeroDivisors R) s
    hs : Eq (Polynomial.map (algebraMap R K) (IsLocalization.integerNormalization  …
    ⊢ Dvd.dvd p q
  -/
  rw [Subtype.coe_mk, Algebra.smul_def, algebraMap_apply] at hs
  have h : p ∣ q * C s := by
    use integerNormalization R⁰ r
    apply map_injective (algebraMap R K) (IsFractionRing.injective _ _)
    rw [Polynomial.map_mul, Polynomial.map_mul, hs, hr, mul_assoc, mul_comm r]
    simp
  /-
    case intro.intro.mk
    R : Type u_1
    inst✝⁵ : CommRing R
    K : Type u_2
    inst✝⁴ : Field K
    inst✝³ : Algebra R K
    inst✝² : IsFractionRing R K
    inst✝¹ : IsDomain R
    inst✝ : NormalizedGCDMonoid R
    p q : Polynomial R
    hp : p.IsPrimitive
    hq : q.IsPrimitive
    r : Polynomial K
    hr : Eq (Polynomial.map (algebraMap R K) q) (HMul.hMul (Polynomial.map (algebr …
    s : R
    s0 : Membership.mem (nonZeroDivisors R) s
    hs : Eq (Polynomial.map (algebraMap R K) (IsLocalization.integerNormalization  …
    h : Dvd.dvd p (HMul.hMul q (Polynomial.C s))
    ⊢ Dvd.dvd p q
  -/
  rw [← hp.dvd_primPart_iff_dvd, primPart_mul, hq.primPart_eq, Associated.dvd_iff_dvd_right] at h
    /-
      case intro.intro.mk
      R : Type u_1
      inst✝⁵ : CommRing R
      K : Type u_2
      inst✝⁴ : Field K
      inst✝³ : Algebra R K
      inst✝² : IsFractionRing R K
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      p q : Polynomial R
      hp : p.IsPrimitive
      hq : q.IsPrimitive
      r : Polynomial K
      hr : Eq (Polynomial.map (algebraMap R K) q) (HMul.hMul (Polynomial.map (algebr …
      s : R
      s0 : Membership.mem (nonZeroDivisors R) s
      hs : Eq (Polynomial.map (algebraMap R K) (IsLocalization.integerNormalization  …
      h✝ : Dvd.dvd p (HMul.hMul q (Polynomial.C s).primPart)
      h : Dvd.dvd p ?m.84174
      ⊢ Dvd.dvd p q
    -/
  · exact h
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.mk
      R : Type u_1
      inst✝⁵ : CommRing R
      K : Type u_2
      inst✝⁴ : Field K
      inst✝³ : Algebra R K
      inst✝² : IsFractionRing R K
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      p q : Polynomial R
      hp : p.IsPrimitive
      hq : q.IsPrimitive
      r : Polynomial K
      hr : Eq (Polynomial.map (algebraMap R K) q) (HMul.hMul (Polynomial.map (algebr …
      s : R
      s0 : Membership.mem (nonZeroDivisors R) s
      hs : Eq (Polynomial.map (algebraMap R K) (IsLocalization.integerNormalization  …
      h : Dvd.dvd p (HMul.hMul q (Polynomial.C s).primPart)
      ⊢ Associated (HMul.hMul q (Polynomial.C s).primPart) q
    -/
  · symm
    /-
      case intro.intro.mk
      R : Type u_1
      inst✝⁵ : CommRing R
      K : Type u_2
      inst✝⁴ : Field K
      inst✝³ : Algebra R K
      inst✝² : IsFractionRing R K
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      p q : Polynomial R
      hp : p.IsPrimitive
      hq : q.IsPrimitive
      r : Polynomial K
      hr : Eq (Polynomial.map (algebraMap R K) q) (HMul.hMul (Polynomial.map (algebr …
      s : R
      s0 : Membership.mem (nonZeroDivisors R) s
      hs : Eq (Polynomial.map (algebraMap R K) (IsLocalization.integerNormalization  …
      h : Dvd.dvd p (HMul.hMul q (Polynomial.C s).primPart)
      ⊢ Associated q (HMul.hMul q (Polynomial.C s).primPart)
    -/
    rcases isUnit_primPart_C s with ⟨u, hu⟩
    /-
      case intro.intro.mk.intro
      R : Type u_1
      inst✝⁵ : CommRing R
      K : Type u_2
      inst✝⁴ : Field K
      inst✝³ : Algebra R K
      inst✝² : IsFractionRing R K
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      p q : Polynomial R
      hp : p.IsPrimitive
      hq : q.IsPrimitive
      r : Polynomial K
      hr : Eq (Polynomial.map (algebraMap R K) q) (HMul.hMul (Polynomial.map (algebr …
      s : R
      s0 : Membership.mem (nonZeroDivisors R) s
      hs : Eq (Polynomial.map (algebraMap R K) (IsLocalization.integerNormalization  …
      h : Dvd.dvd p (HMul.hMul q (Polynomial.C s).primPart)
      u : Units (Polynomial R)
      hu : Eq (↑u) (Polynomial.C s).primPart
      ⊢ Associated q (HMul.hMul q (Polynomial.C s).primPart)
    -/
    use u
    /-
      case h
      R : Type u_1
      inst✝⁵ : CommRing R
      K : Type u_2
      inst✝⁴ : Field K
      inst✝³ : Algebra R K
      inst✝² : IsFractionRing R K
      inst✝¹ : IsDomain R
      inst✝ : NormalizedGCDMonoid R
      p q : Polynomial R
      hp : p.IsPrimitive
      hq : q.IsPrimitive
      r : Polynomial K
      hr : Eq (Polynomial.map (algebraMap R K) q) (HMul.hMul (Polynomial.map (algebr …
      s : R
      s0 : Membership.mem (nonZeroDivisors R) s
      hs : Eq (Polynomial.map (algebraMap R K) (IsLocalization.integerNormalization  …
      h : Dvd.dvd p (HMul.hMul q (Polynomial.C s).primPart)
      u : Units (Polynomial R)
      hu : Eq (↑u) (Polynomial.C s).primPart
      ⊢ Eq (HMul.hMul q ↑u) (HMul.hMul q (Polynomial.C s).primPart)
    -/
    rw [hu]
    /-
      🎉 no goals
    -/
  iterate 2
    apply mul_ne_zero hq.ne_zero
    rw [Ne, C_eq_zero]
    contrapose! s0
    simp [s0, mem_nonZeroDivisors_iff_ne_zero]


theorem IsPrimitive.dvd_iff_fraction_map_dvd_fraction_map {p q : R[X]} (hp : p.IsPrimitive)
    (hq : q.IsPrimitive) : p ∣ q ↔ p.map (algebraMap R K) ∣ q.map (algebraMap R K) :=
  ⟨fun ⟨a, b⟩ => ⟨a.map (algebraMap R K), b.symm ▸ Polynomial.map_mul (algebraMap R K)⟩, fun h =>
    hp.dvd_of_fraction_map_dvd_fraction_map hq h⟩


/-- **Gauss's Lemma** for `ℤ` states that a primitive integer polynomial is irreducible iff it is
  irreducible over `ℚ`. -/
theorem IsPrimitive.Int.irreducible_iff_irreducible_map_cast {p : ℤ[X]} (hp : p.IsPrimitive) :
    Irreducible p ↔ Irreducible (p.map (Int.castRingHom ℚ)) :=
  hp.irreducible_iff_irreducible_map_fraction_map


theorem IsPrimitive.Int.dvd_iff_map_cast_dvd_map_cast (p q : ℤ[X]) (hp : p.IsPrimitive)
    (hq : q.IsPrimitive) : p ∣ q ↔ p.map (Int.castRingHom ℚ) ∣ q.map (Int.castRingHom ℚ) :=
  hp.dvd_iff_fraction_map_dvd_fraction_map ℚ hq


