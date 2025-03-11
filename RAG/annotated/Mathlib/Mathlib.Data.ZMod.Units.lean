/-- `unitsMap` is a group homomorphism that maps units of `ZMod m` to units of `ZMod n` when `n`
divides `m`. -/
def unitsMap (hm : n ∣ m) : (ZMod m)ˣ →* (ZMod n)ˣ := Units.map (castHom hm (ZMod n))


lemma unitsMap_def (hm : n ∣ m) : unitsMap hm = Units.map (castHom hm (ZMod n)) := rfl


lemma unitsMap_comp {d : ℕ} (hm : n ∣ m) (hd : m ∣ d) :
    (unitsMap hm).comp (unitsMap hd) = unitsMap (dvd_trans hm hd) := by
  /-
    n m d : Nat
    hm : Dvd.dvd n m
    hd : Dvd.dvd m d
    ⊢ Eq ((ZMod.unitsMap hm).comp (ZMod.unitsMap hd)) (ZMod.unitsMap ⋯)
  -/
  simp only [unitsMap_def]
  /-
    n m d : Nat
    hm : Dvd.dvd n m
    hd : Dvd.dvd m d
    ⊢ Eq ((Units.map ↑(ZMod.castHom hm (ZMod n))).comp (Units.map ↑(ZMod.castHom h …
  -/
  rw [← Units.map_comp]
  /-
    n m d : Nat
    hm : Dvd.dvd n m
    hd : Dvd.dvd m d
    ⊢ Eq (Units.map ((↑(ZMod.castHom hm (ZMod n))).comp ↑(ZMod.castHom hd (ZMod m) …
  -/
  exact congr_arg Units.map <| congr_arg RingHom.toMonoidHom <| castHom_comp hm hd
  /-
    🎉 no goals
  -/


@[simp]
lemma unitsMap_self (n : ℕ) : unitsMap (dvd_refl n) = MonoidHom.id _ := by
  /-
    n : Nat
    ⊢ Eq (ZMod.unitsMap ⋯) (MonoidHom.id (Units (ZMod n)))
  -/
  simp [unitsMap, castHom_self]
  /-
    🎉 no goals
  -/


lemma isUnit_cast_of_dvd (hm : n ∣ m) (a : Units (ZMod m)) : IsUnit (cast (a : ZMod m) : ZMod n) :=
  Units.isUnit (unitsMap hm a)

@[deprecated (since := "2024-12-16")] alias IsUnit_cast_of_dvd := isUnit_cast_of_dvd


theorem unitsMap_surjective [hm : NeZero m] (h : n ∣ m) :
    Function.Surjective (unitsMap h) := by
  suffices ∀ x : ℕ, x.Coprime n → ∃ k : ℕ, (x + k * n).Coprime m by
    intro x
    have ⟨k, hk⟩ := this x.val.val (val_coe_unit_coprime x)
    refine ⟨unitOfCoprime _ hk, Units.ext ?_⟩
    have : NeZero n := ⟨fun hn ↦ hm.out (eq_zero_of_zero_dvd (hn ▸ h))⟩
    simp [unitsMap_def, - castHom_apply]
  /-
    n m : Nat
    hm : NeZero m
    h : Dvd.dvd n m
    ⊢ ∀ (x : Nat), x.Coprime n → Exists fun k => (HAdd.hAdd x (HMul.hMul k n)).Cop …
  -/
  intro x hx
  /-
    n m : Nat
    hm : NeZero m
    h : Dvd.dvd n m
    x : Nat
    hx : x.Coprime n
    ⊢ Exists fun k => (HAdd.hAdd x (HMul.hMul k n)).Coprime m
  -/
  let ps := m.primeFactors.filter (fun p ↦ ¬p ∣ x)
  /-
    n m : Nat
    hm : NeZero m
    h : Dvd.dvd n m
    x : Nat
    hx : x.Coprime n
    ps : Finset Nat := Finset.filter (fun p => Not (Dvd.dvd p x)) m.primeFactors
    ⊢ Exists fun k => (HAdd.hAdd x (HMul.hMul k n)).Coprime m
  -/
  use ps.prod id
  /-
    case h
    n m : Nat
    hm : NeZero m
    h : Dvd.dvd n m
    x : Nat
    hx : x.Coprime n
    ps : Finset Nat := Finset.filter (fun p => Not (Dvd.dvd p x)) m.primeFactors
    ⊢ (HAdd.hAdd x (HMul.hMul (ps.prod id) n)).Coprime m
  -/
  apply Nat.coprime_of_dvd
  /-
    case h.H
    n m : Nat
    hm : NeZero m
    h : Dvd.dvd n m
    x : Nat
    hx : x.Coprime n
    ps : Finset Nat := Finset.filter (fun p => Not (Dvd.dvd p x)) m.primeFactors
    ⊢ ∀ (k : Nat), Nat.Prime k → Dvd.dvd k (HAdd.hAdd x (HMul.hMul (ps.prod id) n) …
  -/
  intro p pp hp hpn
  /-
    case h.H
    n m : Nat
    hm : NeZero m
    h : Dvd.dvd n m
    x : Nat
    hx : x.Coprime n
    ps : Finset Nat := Finset.filter (fun p => Not (Dvd.dvd p x)) m.primeFactors
    p : Nat
    pp : Nat.Prime p
    hp : Dvd.dvd p (HAdd.hAdd x (HMul.hMul (ps.prod id) n))
    hpn : Dvd.dvd p m
    ⊢ False
  -/
  by_cases hpx : p ∣ x
    /-
      case pos
      n m : Nat
      hm : NeZero m
      h : Dvd.dvd n m
      x : Nat
      hx : x.Coprime n
      ps : Finset Nat := Finset.filter (fun p => Not (Dvd.dvd p x)) m.primeFactors
      p : Nat
      pp : Nat.Prime p
      hp : Dvd.dvd p (HAdd.hAdd x (HMul.hMul (ps.prod id) n))
      hpn : Dvd.dvd p m
      hpx : Dvd.dvd p x
      ⊢ False
    -/
  · have h := Nat.dvd_sub' hp hpx
    /-
      case pos
      n m : Nat
      hm : NeZero m
      h✝ : Dvd.dvd n m
      x : Nat
      hx : x.Coprime n
      ps : Finset Nat := Finset.filter (fun p => Not (Dvd.dvd p x)) m.primeFactors
      p : Nat
      pp : Nat.Prime p
      hp : Dvd.dvd p (HAdd.hAdd x (HMul.hMul (ps.prod id) n))
      hpn : Dvd.dvd p m
      hpx : Dvd.dvd p x
      h : Dvd.dvd p (HSub.hSub (HAdd.hAdd x (HMul.hMul (ps.prod id) n)) x)
      ⊢ False
    -/
    rw [add_comm, Nat.add_sub_cancel] at h
    /-
      case pos
      n m : Nat
      hm : NeZero m
      h✝ : Dvd.dvd n m
      x : Nat
      hx : x.Coprime n
      ps : Finset Nat := Finset.filter (fun p => Not (Dvd.dvd p x)) m.primeFactors
      p : Nat
      pp : Nat.Prime p
      hp : Dvd.dvd p (HAdd.hAdd x (HMul.hMul (ps.prod id) n))
      hpn : Dvd.dvd p m
      hpx : Dvd.dvd p x
      h : Dvd.dvd p (HMul.hMul (ps.prod id) n)
      ⊢ False
    -/
    rcases pp.dvd_mul.mp h with h | h
      /-
        case pos.inl
        n m : Nat
        hm : NeZero m
        h✝¹ : Dvd.dvd n m
        x : Nat
        hx : x.Coprime n
        ps : Finset Nat := Finset.filter (fun p => Not (Dvd.dvd p x)) m.primeFactors
        p : Nat
        pp : Nat.Prime p
        hp : Dvd.dvd p (HAdd.hAdd x (HMul.hMul (ps.prod id) n))
        hpn : Dvd.dvd p m
        hpx : Dvd.dvd p x
        h✝ : Dvd.dvd p (HMul.hMul (ps.prod id) n)
        h : Dvd.dvd p (ps.prod id)
        ⊢ False
      -/
    · have ⟨q, hq, hq'⟩ := (pp.prime.dvd_finset_prod_iff id).mp h
      rw [Finset.mem_filter, Nat.mem_primeFactors,
        ← (Nat.prime_dvd_prime_iff_eq pp hq.1.1).mp hq'] at hq
      /-
        case pos.inl
        n m : Nat
        hm : NeZero m
        h✝¹ : Dvd.dvd n m
        x : Nat
        hx : x.Coprime n
        ps : Finset Nat := Finset.filter (fun p => Not (Dvd.dvd p x)) m.primeFactors
        p : Nat
        pp : Nat.Prime p
        hp : Dvd.dvd p (HAdd.hAdd x (HMul.hMul (ps.prod id) n))
        hpn : Dvd.dvd p m
        hpx : Dvd.dvd p x
        h✝ : Dvd.dvd p (HMul.hMul (ps.prod id) n)
        h : Dvd.dvd p (ps.prod id)
        q : Nat
        hq : And (And (Nat.Prime p) (And (Dvd.dvd p m) (Ne m 0))) (Not (Dvd.dvd p x))
        hq' : Dvd.dvd p (id q)
        ⊢ False
      -/
      exact hq.2 hpx
      /-
        🎉 no goals
      -/
      /-
        case pos.inr
        n m : Nat
        hm : NeZero m
        h✝¹ : Dvd.dvd n m
        x : Nat
        hx : x.Coprime n
        ps : Finset Nat := Finset.filter (fun p => Not (Dvd.dvd p x)) m.primeFactors
        p : Nat
        pp : Nat.Prime p
        hp : Dvd.dvd p (HAdd.hAdd x (HMul.hMul (ps.prod id) n))
        hpn : Dvd.dvd p m
        hpx : Dvd.dvd p x
        h✝ : Dvd.dvd p (HMul.hMul (ps.prod id) n)
        h : Dvd.dvd p n
        ⊢ False
      -/
    · exact Nat.Prime.not_coprime_iff_dvd.mpr ⟨p, pp, hpx, h⟩ hx
      /-
        🎉 no goals
      -/
    /-
      case neg
      n m : Nat
      hm : NeZero m
      h : Dvd.dvd n m
      x : Nat
      hx : x.Coprime n
      ps : Finset Nat := Finset.filter (fun p => Not (Dvd.dvd p x)) m.primeFactors
      p : Nat
      pp : Nat.Prime p
      hp : Dvd.dvd p (HAdd.hAdd x (HMul.hMul (ps.prod id) n))
      hpn : Dvd.dvd p m
      hpx : Not (Dvd.dvd p x)
      ⊢ False
    -/
  · have pps : p ∈ ps := Finset.mem_filter.mpr ⟨Nat.mem_primeFactors.mpr ⟨pp, hpn, hm.out⟩, hpx⟩
    /-
      case neg
      n m : Nat
      hm : NeZero m
      h : Dvd.dvd n m
      x : Nat
      hx : x.Coprime n
      ps : Finset Nat := Finset.filter (fun p => Not (Dvd.dvd p x)) m.primeFactors
      p : Nat
      pp : Nat.Prime p
      hp : Dvd.dvd p (HAdd.hAdd x (HMul.hMul (ps.prod id) n))
      hpn : Dvd.dvd p m
      hpx : Not (Dvd.dvd p x)
      pps : Membership.mem ps p
      ⊢ False
    -/
    have h := Nat.dvd_sub' hp ((Finset.dvd_prod_of_mem id pps).mul_right n)
    /-
      case neg
      n m : Nat
      hm : NeZero m
      h✝ : Dvd.dvd n m
      x : Nat
      hx : x.Coprime n
      ps : Finset Nat := Finset.filter (fun p => Not (Dvd.dvd p x)) m.primeFactors
      p : Nat
      pp : Nat.Prime p
      hp : Dvd.dvd p (HAdd.hAdd x (HMul.hMul (ps.prod id) n))
      hpn : Dvd.dvd p m
      hpx : Not (Dvd.dvd p x)
      pps : Membership.mem ps p
      h : Dvd.dvd p (HSub.hSub (HAdd.hAdd x (HMul.hMul (ps.prod id) n)) (HMul.hMul ( …
      ⊢ False
    -/
    rw [Nat.add_sub_cancel] at h
    /-
      case neg
      n m : Nat
      hm : NeZero m
      h✝ : Dvd.dvd n m
      x : Nat
      hx : x.Coprime n
      ps : Finset Nat := Finset.filter (fun p => Not (Dvd.dvd p x)) m.primeFactors
      p : Nat
      pp : Nat.Prime p
      hp : Dvd.dvd p (HAdd.hAdd x (HMul.hMul (ps.prod id) n))
      hpn : Dvd.dvd p m
      hpx : Not (Dvd.dvd p x)
      pps : Membership.mem ps p
      h : Dvd.dvd p x
      ⊢ False
    -/
    contradiction
    /-
      🎉 no goals
    -/

-- This needs `Nat.primeFactors`, so cannot go into `Mathlib.Data.ZMod.Basic`.

open Nat in
lemma not_isUnit_of_mem_primeFactors {n p : ℕ} (h : p ∈ n.primeFactors) :
    ¬ IsUnit (p : ZMod n) := by
  /-
    n p : Nat
    h : Membership.mem n.primeFactors p
    ⊢ Not (IsUnit ↑p)
  -/
  rw [isUnit_iff_coprime]
  /-
    n p : Nat
    h : Membership.mem n.primeFactors p
    ⊢ Not (p.Coprime n)
  -/
  exact (Prime.dvd_iff_not_coprime <| prime_of_mem_primeFactors h).mp <| dvd_of_mem_primeFactors h
  /-
    🎉 no goals
  -/


/-- Any element of `ZMod N` has the form `u * d` where `u` is a unit and `d` is a divisor of `N`. -/
lemma eq_unit_mul_divisor {N : ℕ} (a : ZMod N) :
    ∃ d : ℕ, d ∣ N ∧ ∃ (u : ZMod N), IsUnit u ∧ a = u * d := by
  /-
    N : Nat
    a : ZMod N
    ⊢ Exists fun d => And (Dvd.dvd d N) (Exists fun u => And (IsUnit u) (Eq a (HMu …
  -/
  rcases eq_or_ne N 0 with rfl | hN
  -- Silly special case : N = 0. Of no mathematical interest, but true, so let's prove it.
    /-
      case inl
      a : ZMod 0
      ⊢ Exists fun d => And (Dvd.dvd d 0) (Exists fun u => And (IsUnit u) (Eq a (HMu …
    -/
  · change ℤ at a
    /-
      case inl
      a : Int
      ⊢ Exists fun d => And (Dvd.dvd d 0) (Exists fun u => And (IsUnit u) (Eq a (HMu …
    -/
    rcases eq_or_ne a 0 with rfl | ha
      /-
        case inl.inl
        ⊢ Exists fun d => And (Dvd.dvd d 0) (Exists fun u => And (IsUnit u) (Eq 0 (HMu …
      -/
    · refine ⟨0, dvd_zero _, 1, isUnit_one, by rw [Nat.cast_zero, mul_zero]⟩
      /-
        🎉 no goals
      -/
    /-
      case inl.inr
      a : Int
      ha : Ne a 0
      ⊢ Exists fun d => And (Dvd.dvd d 0) (Exists fun u => And (IsUnit u) (Eq a (HMu …
    -/
    refine ⟨a.natAbs, dvd_zero _, Int.sign a, ?_, (Int.sign_mul_natAbs a).symm⟩
    /-
      case inl.inr
      a : Int
      ha : Ne a 0
      ⊢ IsUnit a.sign
    -/
    rcases lt_or_gt_of_ne ha with h | h
      /-
        case inl.inr.inl
        a : Int
        ha : Ne a 0
        h : LT.lt a 0
        ⊢ IsUnit a.sign
      -/
    · simp only [Int.sign_eq_neg_one_of_neg h, IsUnit.neg_iff, isUnit_one]
      /-
        🎉 no goals
      -/
      /-
        case inl.inr.inr
        a : Int
        ha : Ne a 0
        h : GT.gt a 0
        ⊢ IsUnit a.sign
      -/
    · simp only [Int.sign_eq_one_of_pos h, isUnit_one]
      /-
        🎉 no goals
      -/
  -- now the interesting case
  /-
    case inr
    N : Nat
    a : ZMod N
    hN : Ne N 0
    ⊢ Exists fun d => And (Dvd.dvd d N) (Exists fun u => And (IsUnit u) (Eq a (HMu …
  -/
  have : NeZero N := ⟨hN⟩
  -- Define `d` as the GCD of a lift of `a` and `N`.
  /-
    case inr
    N : Nat
    a : ZMod N
    hN : Ne N 0
    this : NeZero N
    ⊢ Exists fun d => And (Dvd.dvd d N) (Exists fun u => And (IsUnit u) (Eq a (HMu …
  -/
  let d := a.val.gcd N
  /-
    case inr
    N : Nat
    a : ZMod N
    hN : Ne N 0
    this : NeZero N
    d : Nat := a.val.gcd N
    ⊢ Exists fun d => And (Dvd.dvd d N) (Exists fun u => And (IsUnit u) (Eq a (HMu …
  -/
  have hd : d ≠ 0 := Nat.gcd_ne_zero_right hN
  /-
    case inr
    N : Nat
    a : ZMod N
    hN : Ne N 0
    this : NeZero N
    d : Nat := a.val.gcd N
    hd : Ne d 0
    ⊢ Exists fun d => And (Dvd.dvd d N) (Exists fun u => And (IsUnit u) (Eq a (HMu …
  -/
  obtain ⟨a₀, (ha₀ : _ = d * _)⟩ := a.val.gcd_dvd_left N
  /-
    case inr.intro
    N : Nat
    a : ZMod N
    hN : Ne N 0
    this : NeZero N
    d : Nat := a.val.gcd N
    hd : Ne d 0
    a₀ : Nat
    ha₀ : Eq a.val (HMul.hMul d a₀)
    ⊢ Exists fun d => And (Dvd.dvd d N) (Exists fun u => And (IsUnit u) (Eq a (HMu …
  -/
  obtain ⟨N₀, (hN₀ : _ = d * _)⟩ := a.val.gcd_dvd_right N
  /-
    case inr.intro.intro
    N : Nat
    a : ZMod N
    hN : Ne N 0
    this : NeZero N
    d : Nat := a.val.gcd N
    hd : Ne d 0
    a₀ : Nat
    ha₀ : Eq a.val (HMul.hMul d a₀)
    N₀ : Nat
    hN₀ : Eq N (HMul.hMul d N₀)
    ⊢ Exists fun d => And (Dvd.dvd d N) (Exists fun u => And (IsUnit u) (Eq a (HMu …
  -/
  refine ⟨d, ⟨N₀, hN₀⟩, ?_⟩
  -- Show `a` is a unit mod `N / d`.
  have hu₀ : IsUnit (a₀ : ZMod N₀) := by
    refine (isUnit_iff_coprime _ _).mpr (Nat.isCoprime_iff_coprime.mp ?_)
    obtain ⟨p, q, hpq⟩ : ∃ (p q : ℤ), d = a.val * p + N * q := ⟨_, _, Nat.gcd_eq_gcd_ab _ _⟩
    rw [ha₀, hN₀, Nat.cast_mul, Nat.cast_mul, mul_assoc, mul_assoc, ← mul_add, eq_comm,
      mul_comm _ p, mul_comm _ q] at hpq
    exact ⟨p, q, Int.eq_one_of_mul_eq_self_right (Nat.cast_ne_zero.mpr hd) hpq⟩
  -- Lift it arbitrarily to a unit mod `N`.
  /-
    case inr.intro.intro
    N : Nat
    a : ZMod N
    hN : Ne N 0
    this : NeZero N
    d : Nat := a.val.gcd N
    hd : Ne d 0
    a₀ : Nat
    ha₀ : Eq a.val (HMul.hMul d a₀)
    N₀ : Nat
    hN₀ : Eq N (HMul.hMul d N₀)
    hu₀ : IsUnit ↑a₀
    ⊢ Exists fun u => And (IsUnit u) (Eq a (HMul.hMul u ↑d))
  -/
  obtain ⟨u, hu⟩ := (ZMod.unitsMap_surjective (⟨d, mul_comm d N₀ ▸ hN₀⟩ : N₀ ∣ N)) hu₀.unit
  /-
    case inr.intro.intro.intro
    N : Nat
    a : ZMod N
    hN : Ne N 0
    this : NeZero N
    d : Nat := a.val.gcd N
    hd : Ne d 0
    a₀ : Nat
    ha₀ : Eq a.val (HMul.hMul d a₀)
    N₀ : Nat
    hN₀ : Eq N (HMul.hMul d N₀)
    hu₀ : IsUnit ↑a₀
    u : Units (ZMod N)
    hu : Eq ((ZMod.unitsMap ⋯) u) hu₀.unit
    ⊢ Exists fun u => And (IsUnit u) (Eq a (HMul.hMul u ↑d))
  -/
  rw [unitsMap_def, ← Units.eq_iff, Units.coe_map, IsUnit.unit_spec, MonoidHom.coe_coe] at hu
  /-
    case inr.intro.intro.intro
    N : Nat
    a : ZMod N
    hN : Ne N 0
    this : NeZero N
    d : Nat := a.val.gcd N
    hd : Ne d 0
    a₀ : Nat
    ha₀ : Eq a.val (HMul.hMul d a₀)
    N₀ : Nat
    hN₀ : Eq N (HMul.hMul d N₀)
    hu₀ : IsUnit ↑a₀
    u : Units (ZMod N)
    hu : Eq ((ZMod.castHom ⋯ (ZMod N₀)) ↑u) ↑a₀
    ⊢ Exists fun u => And (IsUnit u) (Eq a (HMul.hMul u ↑d))
  -/
  refine ⟨u.val, u.isUnit, ?_⟩
  rw [← ZMod.natCast_zmod_val a, ← ZMod.natCast_zmod_val u.1, ha₀, ← Nat.cast_mul,
    ZMod.natCast_eq_natCast_iff, mul_comm _ d, Nat.ModEq]
  /-
    case inr.intro.intro.intro
    N : Nat
    a : ZMod N
    hN : Ne N 0
    this : NeZero N
    d : Nat := a.val.gcd N
    hd : Ne d 0
    a₀ : Nat
    ha₀ : Eq a.val (HMul.hMul d a₀)
    N₀ : Nat
    hN₀ : Eq N (HMul.hMul d N₀)
    hu₀ : IsUnit ↑a₀
    u : Units (ZMod N)
    hu : Eq ((ZMod.castHom ⋯ (ZMod N₀)) ↑u) ↑a₀
    ⊢ Eq (HMod.hMod (HMul.hMul d a₀) N) (HMod.hMod (HMul.hMul d (↑u).val) N)
  -/
  simp only [hN₀, Nat.mul_mod_mul_left, Nat.mul_right_inj hd]
  /-
    case inr.intro.intro.intro
    N : Nat
    a : ZMod N
    hN : Ne N 0
    this : NeZero N
    d : Nat := a.val.gcd N
    hd : Ne d 0
    a₀ : Nat
    ha₀ : Eq a.val (HMul.hMul d a₀)
    N₀ : Nat
    hN₀ : Eq N (HMul.hMul d N₀)
    hu₀ : IsUnit ↑a₀
    u : Units (ZMod N)
    hu : Eq ((ZMod.castHom ⋯ (ZMod N₀)) ↑u) ↑a₀
    ⊢ Eq (HMod.hMod a₀ N₀) (HMod.hMod (↑u).val N₀)
  -/
  rw [← Nat.ModEq, ← ZMod.natCast_eq_natCast_iff, ← hu, natCast_val, castHom_apply]
  /-
    🎉 no goals
  -/


