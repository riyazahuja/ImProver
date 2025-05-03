theorem mem_map_rangeS {p : S[X]} : p ∈ (mapRingHom f).rangeS ↔ ∀ n, p.coeff n ∈ f.rangeS := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial S
    ⊢ Iff (Membership.mem (Polynomial.mapRingHom f).rangeS p) (∀ (n : Nat), Member …
  -/
  constructor
    /-
      case mp
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      inst✝ : Semiring S
      f : RingHom R S
      p : Polynomial S
      ⊢ Membership.mem (Polynomial.mapRingHom f).rangeS p → ∀ (n : Nat), Membership. …
    -/
  · rintro ⟨p, rfl⟩ n
    /-
      case mp.intro
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      inst✝ : Semiring S
      f : RingHom R S
      p : Polynomial R
      n : Nat
      ⊢ Membership.mem f.rangeS (((Polynomial.mapRingHom f) p).coeff n)
    -/
    rw [coe_mapRingHom, coeff_map]
    /-
      case mp.intro
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      inst✝ : Semiring S
      f : RingHom R S
      p : Polynomial R
      n : Nat
      ⊢ Membership.mem f.rangeS (f (p.coeff n))
    -/
    exact Set.mem_range_self _
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      inst✝ : Semiring S
      f : RingHom R S
      p : Polynomial S
      ⊢ (∀ (n : Nat), Membership.mem f.rangeS (p.coeff n)) → Membership.mem (Polynom …
    -/
  · intro h
    /-
      case mpr
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      inst✝ : Semiring S
      f : RingHom R S
      p : Polynomial S
      h : ∀ (n : Nat), Membership.mem f.rangeS (p.coeff n)
      ⊢ Membership.mem (Polynomial.mapRingHom f).rangeS p
    -/
    rw [p.as_sum_range_C_mul_X_pow]
    /-
      case mpr
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      inst✝ : Semiring S
      f : RingHom R S
      p : Polynomial S
      h : ∀ (n : Nat), Membership.mem f.rangeS (p.coeff n)
      ⊢ Membership.mem (Polynomial.mapRingHom f).rangeS ((Finset.range (HAdd.hAdd p. …
    -/
    refine (mapRingHom f).rangeS.sum_mem ?_
    /-
      case mpr
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      inst✝ : Semiring S
      f : RingHom R S
      p : Polynomial S
      h : ∀ (n : Nat), Membership.mem f.rangeS (p.coeff n)
      ⊢ ∀ (c : Nat), Membership.mem (Finset.range (HAdd.hAdd p.natDegree 1)) c → Mem …
    -/
    intro i _hi
    /-
      case mpr
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      inst✝ : Semiring S
      f : RingHom R S
      p : Polynomial S
      h : ∀ (n : Nat), Membership.mem f.rangeS (p.coeff n)
      i : Nat
      _hi : Membership.mem (Finset.range (HAdd.hAdd p.natDegree 1)) i
      ⊢ Membership.mem (Polynomial.mapRingHom f).rangeS (HMul.hMul (Polynomial.C (p. …
    -/
    rcases h i with ⟨c, hc⟩
    /-
      case mpr.intro
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      inst✝ : Semiring S
      f : RingHom R S
      p : Polynomial S
      h : ∀ (n : Nat), Membership.mem f.rangeS (p.coeff n)
      i : Nat
      _hi : Membership.mem (Finset.range (HAdd.hAdd p.natDegree 1)) i
      c : R
      hc : Eq (f c) (p.coeff i)
      ⊢ Membership.mem (Polynomial.mapRingHom f).rangeS (HMul.hMul (Polynomial.C (p. …
    -/
    use C c * X ^ i
    /-
      case h
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      inst✝ : Semiring S
      f : RingHom R S
      p : Polynomial S
      h : ∀ (n : Nat), Membership.mem f.rangeS (p.coeff n)
      i : Nat
      _hi : Membership.mem (Finset.range (HAdd.hAdd p.natDegree 1)) i
      c : R
      hc : Eq (f c) (p.coeff i)
      ⊢ Eq ((Polynomial.mapRingHom f) (HMul.hMul (Polynomial.C c) (HPow.hPow Polynom …
    -/
    rw [coe_mapRingHom, Polynomial.map_mul, map_C, hc, Polynomial.map_pow, map_X]
    /-
      🎉 no goals
    -/


theorem mem_map_range {R S : Type*} [Ring R] [Ring S] (f : R →+* S) {p : S[X]} :
    p ∈ (mapRingHom f).range ↔ ∀ n, p.coeff n ∈ f.range :=
  mem_map_rangeS f


