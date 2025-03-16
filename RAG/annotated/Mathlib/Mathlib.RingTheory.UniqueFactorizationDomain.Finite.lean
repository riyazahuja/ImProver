local infixl:50 " ~ᵤ " => Associated


/-- If `y` is a nonzero element of a unique factorization monoid with finitely
many units (e.g. `ℤ`, `Ideal (ring_of_integers K)`), it has finitely many divisors. -/
noncomputable def fintypeSubtypeDvd {M : Type*} [CancelCommMonoidWithZero M]
    [UniqueFactorizationMonoid M] [Fintype Mˣ] (y : M) (hy : y ≠ 0) : Fintype { x // x ∣ y } := by
  /-
    α : Type u_1
    M : Type u_2
    inst✝² : CancelCommMonoidWithZero M
    inst✝¹ : UniqueFactorizationMonoid M
    inst✝ : Fintype (Units M)
    y : M
    hy : Ne y 0
    ⊢ Fintype (Subtype fun x => Dvd.dvd x y)
  -/
  haveI : Nontrivial M := ⟨⟨y, 0, hy⟩⟩
  /-
    α : Type u_1
    M : Type u_2
    inst✝² : CancelCommMonoidWithZero M
    inst✝¹ : UniqueFactorizationMonoid M
    inst✝ : Fintype (Units M)
    y : M
    hy : Ne y 0
    this : Nontrivial M
    ⊢ Fintype (Subtype fun x => Dvd.dvd x y)
  -/
  haveI : NormalizationMonoid M := UniqueFactorizationMonoid.normalizationMonoid
  /-
    α : Type u_1
    M : Type u_2
    inst✝² : CancelCommMonoidWithZero M
    inst✝¹ : UniqueFactorizationMonoid M
    inst✝ : Fintype (Units M)
    y : M
    hy : Ne y 0
    this✝ : Nontrivial M
    this : NormalizationMonoid M
    ⊢ Fintype (Subtype fun x => Dvd.dvd x y)
  -/
  haveI := Classical.decEq M
  /-
    α : Type u_1
    M : Type u_2
    inst✝² : CancelCommMonoidWithZero M
    inst✝¹ : UniqueFactorizationMonoid M
    inst✝ : Fintype (Units M)
    y : M
    hy : Ne y 0
    this✝¹ : Nontrivial M
    this✝ : NormalizationMonoid M
    this : DecidableEq M
    ⊢ Fintype (Subtype fun x => Dvd.dvd x y)
  -/
  haveI := Classical.decEq (Associates M)
  -- We'll show `fun (u : Mˣ) (f ⊆ factors y) ↦ u * Π f` is injective
  -- and has image exactly the divisors of `y`.
  refine
    Fintype.ofFinset
      (((normalizedFactors y).powerset.toFinset ×ˢ (Finset.univ : Finset Mˣ)).image fun s =>
        (s.snd : M) * s.fst.prod)
      fun x => ?_
  simp only [exists_prop, Finset.mem_image, Finset.mem_product, Finset.mem_univ, and_true,
    Multiset.mem_toFinset, Multiset.mem_powerset, exists_eq_right, Multiset.mem_map]
  /-
    α : Type u_1
    M : Type u_2
    inst✝² : CancelCommMonoidWithZero M
    inst✝¹ : UniqueFactorizationMonoid M
    inst✝ : Fintype (Units M)
    y : M
    hy : Ne y 0
    this✝² : Nontrivial M
    this✝¹ : NormalizationMonoid M
    this✝ : DecidableEq M
    this : DecidableEq (Associates M)
    x : M
    ⊢ Iff (Exists fun a => And (LE.le a.1 (UniqueFactorizationMonoid.normalizedFac …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      M : Type u_2
      inst✝² : CancelCommMonoidWithZero M
      inst✝¹ : UniqueFactorizationMonoid M
      inst✝ : Fintype (Units M)
      y : M
      hy : Ne y 0
      this✝² : Nontrivial M
      this✝¹ : NormalizationMonoid M
      this✝ : DecidableEq M
      this : DecidableEq (Associates M)
      x : M
      ⊢ (Exists fun a => And (LE.le a.1 (UniqueFactorizationMonoid.normalizedFactors …
    -/
  · rintro ⟨s, hs, rfl⟩
    /-
      case mp.intro.intro
      α : Type u_1
      M : Type u_2
      inst✝² : CancelCommMonoidWithZero M
      inst✝¹ : UniqueFactorizationMonoid M
      inst✝ : Fintype (Units M)
      y : M
      hy : Ne y 0
      this✝² : Nontrivial M
      this✝¹ : NormalizationMonoid M
      this✝ : DecidableEq M
      this : DecidableEq (Associates M)
      s : Prod (Multiset M) (Units M)
      hs : LE.le s.1 (UniqueFactorizationMonoid.normalizedFactors y)
      ⊢ Membership.mem (fun x => Exists fun c => Eq y (HMul.hMul x c)) (HMul.hMul (↑ …
    -/
    show (s.snd : M) * s.fst.prod ∣ y
    rw [(unit_associated_one.mul_right s.fst.prod).dvd_iff_dvd_left, one_mul,
      ← (prod_normalizedFactors hy).dvd_iff_dvd_right]
    /-
      case mp.intro.intro
      α : Type u_1
      M : Type u_2
      inst✝² : CancelCommMonoidWithZero M
      inst✝¹ : UniqueFactorizationMonoid M
      inst✝ : Fintype (Units M)
      y : M
      hy : Ne y 0
      this✝² : Nontrivial M
      this✝¹ : NormalizationMonoid M
      this✝ : DecidableEq M
      this : DecidableEq (Associates M)
      s : Prod (Multiset M) (Units M)
      hs : LE.le s.1 (UniqueFactorizationMonoid.normalizedFactors y)
      ⊢ Dvd.dvd s.1.prod (UniqueFactorizationMonoid.normalizedFactors y).prod
    -/
    exact Multiset.prod_dvd_prod_of_le hs
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      M : Type u_2
      inst✝² : CancelCommMonoidWithZero M
      inst✝¹ : UniqueFactorizationMonoid M
      inst✝ : Fintype (Units M)
      y : M
      hy : Ne y 0
      this✝² : Nontrivial M
      this✝¹ : NormalizationMonoid M
      this✝ : DecidableEq M
      this : DecidableEq (Associates M)
      x : M
      ⊢ Membership.mem (fun x => Exists fun c => Eq y (HMul.hMul x c)) x → Exists fu …
    -/
  · rintro (h : x ∣ y)
    have hx : x ≠ 0 := by
      refine mt (fun hx => ?_) hy
      rwa [hx, zero_dvd_iff] at h
    /-
      case mpr
      α : Type u_1
      M : Type u_2
      inst✝² : CancelCommMonoidWithZero M
      inst✝¹ : UniqueFactorizationMonoid M
      inst✝ : Fintype (Units M)
      y : M
      hy : Ne y 0
      this✝² : Nontrivial M
      this✝¹ : NormalizationMonoid M
      this✝ : DecidableEq M
      this : DecidableEq (Associates M)
      x : M
      h : Dvd.dvd x y
      hx : Ne x 0
      ⊢ Exists fun a => And (LE.le a.1 (UniqueFactorizationMonoid.normalizedFactors  …
    -/
    obtain ⟨u, hu⟩ := prod_normalizedFactors hx
    /-
      case mpr.intro
      α : Type u_1
      M : Type u_2
      inst✝² : CancelCommMonoidWithZero M
      inst✝¹ : UniqueFactorizationMonoid M
      inst✝ : Fintype (Units M)
      y : M
      hy : Ne y 0
      this✝² : Nontrivial M
      this✝¹ : NormalizationMonoid M
      this✝ : DecidableEq M
      this : DecidableEq (Associates M)
      x : M
      h : Dvd.dvd x y
      hx : Ne x 0
      u : Units M
      hu : Eq (HMul.hMul (UniqueFactorizationMonoid.normalizedFactors x).prod ↑u) x
      ⊢ Exists fun a => And (LE.le a.1 (UniqueFactorizationMonoid.normalizedFactors  …
    -/
    refine ⟨⟨normalizedFactors x, u⟩, ?_, (mul_comm _ _).trans hu⟩
    /-
      case mpr.intro
      α : Type u_1
      M : Type u_2
      inst✝² : CancelCommMonoidWithZero M
      inst✝¹ : UniqueFactorizationMonoid M
      inst✝ : Fintype (Units M)
      y : M
      hy : Ne y 0
      this✝² : Nontrivial M
      this✝¹ : NormalizationMonoid M
      this✝ : DecidableEq M
      this : DecidableEq (Associates M)
      x : M
      h : Dvd.dvd x y
      hx : Ne x 0
      u : Units M
      hu : Eq (HMul.hMul (UniqueFactorizationMonoid.normalizedFactors x).prod ↑u) x
      ⊢ LE.le { fst := UniqueFactorizationMonoid.normalizedFactors x, snd := u }.1 ( …
    -/
    exact (dvd_iff_normalizedFactors_le_normalizedFactors hx hy).mp h
    /-
      🎉 no goals
    -/


