/-- Given an ideal `𝓟` of a commutative semiring `R`, we say that a polynomial `f : R[X]`
is *weakly Eisenstein at `𝓟`* if `∀ n, n < f.natDegree → f.coeff n ∈ 𝓟`. -/
@[mk_iff]
structure IsWeaklyEisensteinAt [CommSemiring R] (f : R[X]) (𝓟 : Ideal R) : Prop where
  mem : ∀ {n}, n < f.natDegree → f.coeff n ∈ 𝓟


/-- Given an ideal `𝓟` of a commutative semiring `R`, we say that a polynomial `f : R[X]`
is *Eisenstein at `𝓟`* if `f.leadingCoeff ∉ 𝓟`, `∀ n, n < f.natDegree → f.coeff n ∈ 𝓟` and
`f.coeff 0 ∉ 𝓟 ^ 2`. -/
@[mk_iff]
structure IsEisensteinAt [CommSemiring R] (f : R[X]) (𝓟 : Ideal R) : Prop where
  leading : f.leadingCoeff ∉ 𝓟
  mem : ∀ {n}, n < f.natDegree → f.coeff n ∈ 𝓟
  not_mem : f.coeff 0 ∉ 𝓟 ^ 2


theorem map (hf : f.IsWeaklyEisensteinAt 𝓟) {A : Type v} [CommRing A] (φ : R →+* A) :
    (f.map φ).IsWeaklyEisensteinAt (𝓟.map φ) := by
  /-
    R : Type u
    inst✝¹ : CommSemiring R
    𝓟 : Ideal R
    f : Polynomial R
    hf : f.IsWeaklyEisensteinAt 𝓟
    A : Type v
    inst✝ : CommRing A
    φ : RingHom R A
    ⊢ (Polynomial.map φ f).IsWeaklyEisensteinAt (Ideal.map φ 𝓟)
  -/
  refine (isWeaklyEisensteinAt_iff _ _).2 fun hn => ?_
  /-
    R : Type u
    inst✝¹ : CommSemiring R
    𝓟 : Ideal R
    f : Polynomial R
    hf : f.IsWeaklyEisensteinAt 𝓟
    A : Type v
    inst✝ : CommRing A
    φ : RingHom R A
    n✝ : Nat
    hn : LT.lt n✝ (Polynomial.map φ f).natDegree
    ⊢ Membership.mem (Ideal.map φ 𝓟) ((Polynomial.map φ f).coeff n✝)
  -/
  rw [coeff_map]
  /-
    R : Type u
    inst✝¹ : CommSemiring R
    𝓟 : Ideal R
    f : Polynomial R
    hf : f.IsWeaklyEisensteinAt 𝓟
    A : Type v
    inst✝ : CommRing A
    φ : RingHom R A
    n✝ : Nat
    hn : LT.lt n✝ (Polynomial.map φ f).natDegree
    ⊢ Membership.mem (Ideal.map φ 𝓟) (φ (f.coeff n✝))
  -/
  exact mem_map_of_mem _ (hf.mem (lt_of_lt_of_le hn natDegree_map_le))
  /-
    🎉 no goals
  -/


theorem exists_mem_adjoin_mul_eq_pow_natDegree {x : S} (hx : aeval x f = 0) (hmo : f.Monic)
    (hf : f.IsWeaklyEisensteinAt (Submodule.span R {p})) : ∃ y ∈ adjoin R ({x} : Set S),
    (algebraMap R S) p * y = x ^ (f.map (algebraMap R S)).natDegree := by
  rw [aeval_def, Polynomial.eval₂_eq_eval_map, eval_eq_sum_range, range_add_one,
    sum_insert not_mem_range_self, sum_range, (hmo.map (algebraMap R S)).coeff_natDegree,
    one_mul] at hx
  /-
    R : Type u
    inst✝² : CommRing R
    f : Polynomial R
    S : Type v
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    p : R
    x : S
    hx : Eq (HAdd.hAdd (HPow.hPow x (Polynomial.map (algebraMap R S) f).natDegree) …
    hmo : f.Monic
    hf : f.IsWeaklyEisensteinAt (Submodule.span R (Singleton.singleton p))
    ⊢ Exists fun y => And (Membership.mem (Algebra.adjoin R (Singleton.singleton x …
  -/
  replace hx := eq_neg_of_add_eq_zero_left hx
  have : ∀ n < f.natDegree, p ∣ f.coeff n := by
    intro n hn
    exact mem_span_singleton.1 (by simpa using hf.mem hn)
  /-
    R : Type u
    inst✝² : CommRing R
    f : Polynomial R
    S : Type v
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    p : R
    x : S
    hmo : f.Monic
    hf : f.IsWeaklyEisensteinAt (Submodule.span R (Singleton.singleton p))
    hx : Eq (HPow.hPow x (Polynomial.map (algebraMap R S) f).natDegree) (Neg.neg ( …
    this : ∀ (n : Nat), LT.lt n f.natDegree → Dvd.dvd p (f.coeff n)
    ⊢ Exists fun y => And (Membership.mem (Algebra.adjoin R (Singleton.singleton x …
  -/
  choose! φ hφ using this
  conv_rhs at hx =>
    congr
    congr
    · skip
    ext i
    rw [coeff_map, hφ i.1 (lt_of_lt_of_le i.2 natDegree_map_le),
      RingHom.map_mul, mul_assoc]
  /-
    R : Type u
    inst✝² : CommRing R
    f : Polynomial R
    S : Type v
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    p : R
    x : S
    hmo : f.Monic
    hf : f.IsWeaklyEisensteinAt (Submodule.span R (Singleton.singleton p))
    φ : Nat → R
    hx : Eq (HPow.hPow x (Polynomial.map (algebraMap R S) f).natDegree) (Neg.neg ( …
    hφ : ∀ (n : Nat), LT.lt n f.natDegree → Eq (f.coeff n) (HMul.hMul p (φ n))
    ⊢ Exists fun y => And (Membership.mem (Algebra.adjoin R (Singleton.singleton x …
  -/
  rw [hx, ← mul_sum, neg_eq_neg_one_mul, ← mul_assoc (-1 : S), mul_comm (-1 : S), mul_assoc]
  refine
    ⟨-1 * ∑ i : Fin (f.map (algebraMap R S)).natDegree, (algebraMap R S) (φ i.1) * x ^ i.1, ?_, rfl⟩
  exact
    Subalgebra.mul_mem _ (Subalgebra.neg_mem _ (Subalgebra.one_mem _))
      (Subalgebra.sum_mem _ fun i _ =>
        Subalgebra.mul_mem _ (Subalgebra.algebraMap_mem _ _)
          (Subalgebra.pow_mem _ (subset_adjoin (Set.mem_singleton x)) _))


theorem exists_mem_adjoin_mul_eq_pow_natDegree_le {x : S} (hx : aeval x f = 0) (hmo : f.Monic)
    (hf : f.IsWeaklyEisensteinAt (Submodule.span R {p})) :
    ∀ i, (f.map (algebraMap R S)).natDegree ≤ i →
        ∃ y ∈ adjoin R ({x} : Set S), (algebraMap R S) p * y = x ^ i := by
  /-
    R : Type u
    inst✝² : CommRing R
    f : Polynomial R
    S : Type v
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    p : R
    x : S
    hx : Eq ((Polynomial.aeval x) f) 0
    hmo : f.Monic
    hf : f.IsWeaklyEisensteinAt (Submodule.span R (Singleton.singleton p))
    ⊢ ∀ (i : Nat), LE.le (Polynomial.map (algebraMap R S) f).natDegree i → Exists  …
  -/
  intro i hi
  /-
    R : Type u
    inst✝² : CommRing R
    f : Polynomial R
    S : Type v
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    p : R
    x : S
    hx : Eq ((Polynomial.aeval x) f) 0
    hmo : f.Monic
    hf : f.IsWeaklyEisensteinAt (Submodule.span R (Singleton.singleton p))
    i : Nat
    hi : LE.le (Polynomial.map (algebraMap R S) f).natDegree i
    ⊢ Exists fun y => And (Membership.mem (Algebra.adjoin R (Singleton.singleton x …
  -/
  obtain ⟨k, hk⟩ := exists_add_of_le hi
  /-
    case intro
    R : Type u
    inst✝² : CommRing R
    f : Polynomial R
    S : Type v
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    p : R
    x : S
    hx : Eq ((Polynomial.aeval x) f) 0
    hmo : f.Monic
    hf : f.IsWeaklyEisensteinAt (Submodule.span R (Singleton.singleton p))
    i : Nat
    hi : LE.le (Polynomial.map (algebraMap R S) f).natDegree i
    k : Nat
    hk : Eq i (HAdd.hAdd (Polynomial.map (algebraMap R S) f).natDegree k)
    ⊢ Exists fun y => And (Membership.mem (Algebra.adjoin R (Singleton.singleton x …
  -/
  rw [hk, pow_add]
  /-
    case intro
    R : Type u
    inst✝² : CommRing R
    f : Polynomial R
    S : Type v
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    p : R
    x : S
    hx : Eq ((Polynomial.aeval x) f) 0
    hmo : f.Monic
    hf : f.IsWeaklyEisensteinAt (Submodule.span R (Singleton.singleton p))
    i : Nat
    hi : LE.le (Polynomial.map (algebraMap R S) f).natDegree i
    k : Nat
    hk : Eq i (HAdd.hAdd (Polynomial.map (algebraMap R S) f).natDegree k)
    ⊢ Exists fun y => And (Membership.mem (Algebra.adjoin R (Singleton.singleton x …
  -/
  obtain ⟨y, hy, H⟩ := exists_mem_adjoin_mul_eq_pow_natDegree hx hmo hf
  /-
    case intro.intro.intro
    R : Type u
    inst✝² : CommRing R
    f : Polynomial R
    S : Type v
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    p : R
    x : S
    hx : Eq ((Polynomial.aeval x) f) 0
    hmo : f.Monic
    hf : f.IsWeaklyEisensteinAt (Submodule.span R (Singleton.singleton p))
    i : Nat
    hi : LE.le (Polynomial.map (algebraMap R S) f).natDegree i
    k : Nat
    hk : Eq i (HAdd.hAdd (Polynomial.map (algebraMap R S) f).natDegree k)
    y : S
    hy : Membership.mem (Algebra.adjoin R (Singleton.singleton x)) y
    H : Eq (HMul.hMul ((algebraMap R S) p) y) (HPow.hPow x (Polynomial.map (algebr …
    ⊢ Exists fun y => And (Membership.mem (Algebra.adjoin R (Singleton.singleton x …
  -/
  refine ⟨y * x ^ k, ?_, ?_⟩
    /-
      case intro.intro.intro.refine_1
      R : Type u
      inst✝² : CommRing R
      f : Polynomial R
      S : Type v
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      p : R
      x : S
      hx : Eq ((Polynomial.aeval x) f) 0
      hmo : f.Monic
      hf : f.IsWeaklyEisensteinAt (Submodule.span R (Singleton.singleton p))
      i : Nat
      hi : LE.le (Polynomial.map (algebraMap R S) f).natDegree i
      k : Nat
      hk : Eq i (HAdd.hAdd (Polynomial.map (algebraMap R S) f).natDegree k)
      y : S
      hy : Membership.mem (Algebra.adjoin R (Singleton.singleton x)) y
      H : Eq (HMul.hMul ((algebraMap R S) p) y) (HPow.hPow x (Polynomial.map (algebr …
      ⊢ Membership.mem (Algebra.adjoin R (Singleton.singleton x)) (HMul.hMul y (HPow …
    -/
  · exact Subalgebra.mul_mem _ hy (Subalgebra.pow_mem _ (subset_adjoin (Set.mem_singleton x)) _)
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.refine_2
      R : Type u
      inst✝² : CommRing R
      f : Polynomial R
      S : Type v
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      p : R
      x : S
      hx : Eq ((Polynomial.aeval x) f) 0
      hmo : f.Monic
      hf : f.IsWeaklyEisensteinAt (Submodule.span R (Singleton.singleton p))
      i : Nat
      hi : LE.le (Polynomial.map (algebraMap R S) f).natDegree i
      k : Nat
      hk : Eq i (HAdd.hAdd (Polynomial.map (algebraMap R S) f).natDegree k)
      y : S
      hy : Membership.mem (Algebra.adjoin R (Singleton.singleton x)) y
      H : Eq (HMul.hMul ((algebraMap R S) p) y) (HPow.hPow x (Polynomial.map (algebr …
      ⊢ Eq (HMul.hMul ((algebraMap R S) p) (HMul.hMul y (HPow.hPow x k))) (HMul.hMul …
    -/
  · rw [← mul_assoc _ y, H]
    /-
      🎉 no goals
    -/


theorem pow_natDegree_le_of_root_of_monic_mem (hf : f.IsWeaklyEisensteinAt 𝓟)
    {x : R} (hroot : IsRoot f x) (hmo : f.Monic) :
    ∀ i, f.natDegree ≤ i → x ^ i ∈ 𝓟 := by
  /-
    R : Type u
    inst✝ : CommRing R
    𝓟 : Ideal R
    f : Polynomial R
    hf : f.IsWeaklyEisensteinAt 𝓟
    x : R
    hroot : f.IsRoot x
    hmo : f.Monic
    ⊢ ∀ (i : Nat), LE.le f.natDegree i → Membership.mem 𝓟 (HPow.hPow x i)
  -/
  intro i hi
  /-
    R : Type u
    inst✝ : CommRing R
    𝓟 : Ideal R
    f : Polynomial R
    hf : f.IsWeaklyEisensteinAt 𝓟
    x : R
    hroot : f.IsRoot x
    hmo : f.Monic
    i : Nat
    hi : LE.le f.natDegree i
    ⊢ Membership.mem 𝓟 (HPow.hPow x i)
  -/
  obtain ⟨k, hk⟩ := exists_add_of_le hi
  /-
    case intro
    R : Type u
    inst✝ : CommRing R
    𝓟 : Ideal R
    f : Polynomial R
    hf : f.IsWeaklyEisensteinAt 𝓟
    x : R
    hroot : f.IsRoot x
    hmo : f.Monic
    i : Nat
    hi : LE.le f.natDegree i
    k : Nat
    hk : Eq i (HAdd.hAdd f.natDegree k)
    ⊢ Membership.mem 𝓟 (HPow.hPow x i)
  -/
  rw [hk, pow_add]
  /-
    case intro
    R : Type u
    inst✝ : CommRing R
    𝓟 : Ideal R
    f : Polynomial R
    hf : f.IsWeaklyEisensteinAt 𝓟
    x : R
    hroot : f.IsRoot x
    hmo : f.Monic
    i : Nat
    hi : LE.le f.natDegree i
    k : Nat
    hk : Eq i (HAdd.hAdd f.natDegree k)
    ⊢ Membership.mem 𝓟 (HMul.hMul (HPow.hPow x f.natDegree) (HPow.hPow x k))
  -/
  suffices x ^ f.natDegree ∈ 𝓟 by exact mul_mem_right (x ^ k) 𝓟 this
  rw [IsRoot.def, eval_eq_sum_range, Finset.range_add_one,
    Finset.sum_insert Finset.not_mem_range_self, Finset.sum_range, hmo.coeff_natDegree, one_mul] at
    *
  /-
    case intro
    R : Type u
    inst✝ : CommRing R
    𝓟 : Ideal R
    f : Polynomial R
    hf : f.IsWeaklyEisensteinAt 𝓟
    x : R
    hroot : Eq (HAdd.hAdd (HPow.hPow x f.natDegree) (Finset.univ.sum fun i => HMul …
    hmo : f.Monic
    i : Nat
    hi : LE.le f.natDegree i
    k : Nat
    hk : Eq i (HAdd.hAdd f.natDegree k)
    ⊢ Membership.mem 𝓟 (HPow.hPow x f.natDegree)
  -/
  rw [eq_neg_of_add_eq_zero_left hroot, Ideal.neg_mem_iff]
  /-
    case intro
    R : Type u
    inst✝ : CommRing R
    𝓟 : Ideal R
    f : Polynomial R
    hf : f.IsWeaklyEisensteinAt 𝓟
    x : R
    hroot : Eq (HAdd.hAdd (HPow.hPow x f.natDegree) (Finset.univ.sum fun i => HMul …
    hmo : f.Monic
    i : Nat
    hi : LE.le f.natDegree i
    k : Nat
    hk : Eq i (HAdd.hAdd f.natDegree k)
    ⊢ Membership.mem 𝓟 (Finset.univ.sum fun i => HMul.hMul (f.coeff ↑i) (HPow.hPow …
  -/
  exact Submodule.sum_mem _ fun i _ => mul_mem_right _ _ (hf.mem (Fin.is_lt i))
  /-
    🎉 no goals
  -/


theorem pow_natDegree_le_of_aeval_zero_of_monic_mem_map (hf : f.IsWeaklyEisensteinAt 𝓟)
    {x : S} (hx : aeval x f = 0) (hmo : f.Monic) :
    ∀ i, (f.map (algebraMap R S)).natDegree ≤ i → x ^ i ∈ 𝓟.map (algebraMap R S) := by
  suffices x ^ (f.map (algebraMap R S)).natDegree ∈ 𝓟.map (algebraMap R S) by
    intro i hi
    obtain ⟨k, hk⟩ := exists_add_of_le hi
    rw [hk, pow_add]
    exact mul_mem_right _ _ this
  /-
    R : Type u
    inst✝² : CommRing R
    𝓟 : Ideal R
    f : Polynomial R
    S : Type v
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    hf : f.IsWeaklyEisensteinAt 𝓟
    x : S
    hx : Eq ((Polynomial.aeval x) f) 0
    hmo : f.Monic
    ⊢ Membership.mem (Ideal.map (algebraMap R S) 𝓟) (HPow.hPow x (Polynomial.map ( …
  -/
  rw [aeval_def, eval₂_eq_eval_map, ← IsRoot.def] at hx
  /-
    R : Type u
    inst✝² : CommRing R
    𝓟 : Ideal R
    f : Polynomial R
    S : Type v
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    hf : f.IsWeaklyEisensteinAt 𝓟
    x : S
    hx : (Polynomial.map (algebraMap R S) f).IsRoot x
    hmo : f.Monic
    ⊢ Membership.mem (Ideal.map (algebraMap R S) 𝓟) (HPow.hPow x (Polynomial.map ( …
  -/
  exact pow_natDegree_le_of_root_of_monic_mem (hf.map _) hx (hmo.map _) _ rfl.le
  /-
    🎉 no goals
  -/


theorem scaleRoots.isWeaklyEisensteinAt (p : R[X]) {x : R} {P : Ideal R} (hP : x ∈ P) :
    (scaleRoots p x).IsWeaklyEisensteinAt P := by
  /-
    R : Type u
    inst✝ : CommRing R
    p : Polynomial R
    x : R
    P : Ideal R
    hP : Membership.mem P x
    ⊢ (p.scaleRoots x).IsWeaklyEisensteinAt P
  -/
  refine ⟨fun i => ?_⟩
  /-
    R : Type u
    inst✝ : CommRing R
    p : Polynomial R
    x : R
    P : Ideal R
    hP : Membership.mem P x
    n✝ : Nat
    i : LT.lt n✝ (p.scaleRoots x).natDegree
    ⊢ Membership.mem P ((p.scaleRoots x).coeff n✝)
  -/
  rw [coeff_scaleRoots]
  /-
    R : Type u
    inst✝ : CommRing R
    p : Polynomial R
    x : R
    P : Ideal R
    hP : Membership.mem P x
    n✝ : Nat
    i : LT.lt n✝ (p.scaleRoots x).natDegree
    ⊢ Membership.mem P (HMul.hMul (p.coeff n✝) (HPow.hPow x (HSub.hSub p.natDegree …
  -/
  rw [natDegree_scaleRoots, ← tsub_pos_iff_lt] at i
  /-
    R : Type u
    inst✝ : CommRing R
    p : Polynomial R
    x : R
    P : Ideal R
    hP : Membership.mem P x
    n✝ : Nat
    i : LT.lt 0 (HSub.hSub p.natDegree n✝)
    ⊢ Membership.mem P (HMul.hMul (p.coeff n✝) (HPow.hPow x (HSub.hSub p.natDegree …
  -/
  exact Ideal.mul_mem_left _ _ (Ideal.pow_mem_of_mem P hP _ i)
  /-
    🎉 no goals
  -/


theorem dvd_pow_natDegree_of_eval₂_eq_zero {f : R →+* A} (hf : Function.Injective f) {p : R[X]}
    (hp : p.Monic) (x y : R) (z : A) (h : p.eval₂ f z = 0) (hz : f x * z = f y) :
    x ∣ y ^ p.natDegree := by
  /-
    R : Type u
    A : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CommRing A
    f : RingHom R A
    hf : Function.Injective ⇑f
    p : Polynomial R
    hp : p.Monic
    x y : R
    z : A
    h : Eq (Polynomial.eval₂ f z p) 0
    hz : Eq (HMul.hMul (f x) z) (f y)
    ⊢ Dvd.dvd x (HPow.hPow y p.natDegree)
  -/
  rw [← natDegree_scaleRoots p x, ← Ideal.mem_span_singleton]
  refine
    (scaleRoots.isWeaklyEisensteinAt _
          (Ideal.mem_span_singleton.mpr <| dvd_refl x)).pow_natDegree_le_of_root_of_monic_mem
      ?_ ((monic_scaleRoots_iff x).mpr hp) _ le_rfl
  /-
    R : Type u
    A : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CommRing A
    f : RingHom R A
    hf : Function.Injective ⇑f
    p : Polynomial R
    hp : p.Monic
    x y : R
    z : A
    h : Eq (Polynomial.eval₂ f z p) 0
    hz : Eq (HMul.hMul (f x) z) (f y)
    ⊢ (p.scaleRoots x).IsRoot y
  -/
  rw [injective_iff_map_eq_zero'] at hf
  /-
    R : Type u
    A : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CommRing A
    f : RingHom R A
    hf : ∀ (a : R), Iff (Eq (f a) 0) (Eq a 0)
    p : Polynomial R
    hp : p.Monic
    x y : R
    z : A
    h : Eq (Polynomial.eval₂ f z p) 0
    hz : Eq (HMul.hMul (f x) z) (f y)
    ⊢ (p.scaleRoots x).IsRoot y
  -/
  have : eval₂ f _ (p.scaleRoots x) = 0 := scaleRoots_eval₂_eq_zero f h
  /-
    R : Type u
    A : Type u_1
    inst✝¹ : CommRing R
    inst✝ : CommRing A
    f : RingHom R A
    hf : ∀ (a : R), Iff (Eq (f a) 0) (Eq a 0)
    p : Polynomial R
    hp : p.Monic
    x y : R
    z : A
    h : Eq (Polynomial.eval₂ f z p) 0
    hz : Eq (HMul.hMul (f x) z) (f y)
    this : Eq (Polynomial.eval₂ f (HMul.hMul (f x) z) (p.scaleRoots x)) 0
    ⊢ (p.scaleRoots x).IsRoot y
  -/
  rwa [hz, Polynomial.eval₂_at_apply, hf] at this
  /-
    🎉 no goals
  -/


theorem dvd_pow_natDegree_of_aeval_eq_zero [Algebra R A] [Nontrivial A] [NoZeroSMulDivisors R A]
    {p : R[X]} (hp : p.Monic) (x y : R) (z : A) (h : Polynomial.aeval z p = 0)
    (hz : z * algebraMap R A x = algebraMap R A y) : x ∣ y ^ p.natDegree :=
  dvd_pow_natDegree_of_eval₂_eq_zero (NoZeroSMulDivisors.algebraMap_injective R A) hp x y z h
    ((mul_comm _ _).trans hz)


theorem _root_.Polynomial.Monic.leadingCoeff_not_mem (hf : f.Monic) (h : 𝓟 ≠ ⊤) :
    ¬f.leadingCoeff ∈ 𝓟 := hf.leadingCoeff.symm ▸ (Ideal.ne_top_iff_one _).1 h


theorem _root_.Polynomial.Monic.isEisensteinAt_of_mem_of_not_mem (hf : f.Monic) (h : 𝓟 ≠ ⊤)
    (hmem : ∀ {n}, n < f.natDegree → f.coeff n ∈ 𝓟) (hnot_mem : f.coeff 0 ∉ 𝓟 ^ 2) :
    f.IsEisensteinAt 𝓟 :=
  { leading := Polynomial.Monic.leadingCoeff_not_mem hf h
    mem := fun hn => hmem hn
    not_mem := hnot_mem }


theorem isWeaklyEisensteinAt (hf : f.IsEisensteinAt 𝓟) : IsWeaklyEisensteinAt f 𝓟 :=
  ⟨fun h => hf.mem h⟩


theorem coeff_mem (hf : f.IsEisensteinAt 𝓟) {n : ℕ} (hn : n ≠ f.natDegree) : f.coeff n ∈ 𝓟 := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    𝓟 : Ideal R
    f : Polynomial R
    hf : f.IsEisensteinAt 𝓟
    n : Nat
    hn : Ne n f.natDegree
    ⊢ Membership.mem 𝓟 (f.coeff n)
  -/
  cases' ne_iff_lt_or_gt.1 hn with h₁ h₂
    /-
      case inl
      R : Type u
      inst✝ : CommSemiring R
      𝓟 : Ideal R
      f : Polynomial R
      hf : f.IsEisensteinAt 𝓟
      n : Nat
      hn : Ne n f.natDegree
      h₁ : LT.lt n f.natDegree
      ⊢ Membership.mem 𝓟 (f.coeff n)
    -/
  · exact hf.mem h₁
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u
      inst✝ : CommSemiring R
      𝓟 : Ideal R
      f : Polynomial R
      hf : f.IsEisensteinAt 𝓟
      n : Nat
      hn : Ne n f.natDegree
      h₂ : GT.gt n f.natDegree
      ⊢ Membership.mem 𝓟 (f.coeff n)
    -/
  · rw [coeff_eq_zero_of_natDegree_lt h₂]
    /-
      case inr
      R : Type u
      inst✝ : CommSemiring R
      𝓟 : Ideal R
      f : Polynomial R
      hf : f.IsEisensteinAt 𝓟
      n : Nat
      hn : Ne n f.natDegree
      h₂ : GT.gt n f.natDegree
      ⊢ Membership.mem 𝓟 0
    -/
    exact Ideal.zero_mem _
    /-
      🎉 no goals
    -/


/-- If a primitive `f` satisfies `f.IsEisensteinAt 𝓟`, where `𝓟.IsPrime`,
then `f` is irreducible. -/
theorem irreducible (hf : f.IsEisensteinAt 𝓟) (hprime : 𝓟.IsPrime) (hu : f.IsPrimitive)
    (hfd0 : 0 < f.natDegree) : Irreducible f :=
  irreducible_of_eisenstein_criterion hprime hf.leading (fun _ hn => hf.mem (coe_lt_degree.1 hn))
    (natDegree_pos_iff_degree_pos.1 hfd0) hf.not_mem hu


