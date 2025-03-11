/-- A linear equivalence which preserves a finite spanning set must have finite order. -/
lemma LinearEquiv.isOfFinOrder_of_finite_of_span_eq_top_of_mapsTo
    {R M : Type*} [CommSemiring R] [AddCommMonoid M] [Module R M]
    {Φ : Set M} (hΦ₁ : Φ.Finite) (hΦ₂ : span R Φ = ⊤) {e : M ≃ₗ[R] M} (he : MapsTo e Φ Φ) :
    IsOfFinOrder e := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    Φ : Set M
    hΦ₁ : Φ.Finite
    hΦ₂ : Eq (Submodule.span R Φ) Top.top
    e : LinearEquiv (RingHom.id R) M M
    he : Set.MapsTo (⇑e) Φ Φ
    ⊢ IsOfFinOrder e
  -/
  replace he : BijOn e Φ Φ := (hΦ₁.injOn_iff_bijOn_of_mapsTo he).mp e.injective.injOn
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    Φ : Set M
    hΦ₁ : Φ.Finite
    hΦ₂ : Eq (Submodule.span R Φ) Top.top
    e : LinearEquiv (RingHom.id R) M M
    he : Set.BijOn (⇑e) Φ Φ
    ⊢ IsOfFinOrder e
  -/
  let e' := he.equiv
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    Φ : Set M
    hΦ₁ : Φ.Finite
    hΦ₂ : Eq (Submodule.span R Φ) Top.top
    e : LinearEquiv (RingHom.id R) M M
    he : Set.BijOn (⇑e) Φ Φ
    e' : Equiv ↑Φ ↑Φ := Set.BijOn.equiv (⇑e) he
    ⊢ IsOfFinOrder e
  -/
  have : Finite Φ := finite_coe_iff.mpr hΦ₁
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    Φ : Set M
    hΦ₁ : Φ.Finite
    hΦ₂ : Eq (Submodule.span R Φ) Top.top
    e : LinearEquiv (RingHom.id R) M M
    he : Set.BijOn (⇑e) Φ Φ
    e' : Equiv ↑Φ ↑Φ := Set.BijOn.equiv (⇑e) he
    this : Finite ↑Φ
    ⊢ IsOfFinOrder e
  -/
  obtain ⟨k, hk₀, hk⟩ := isOfFinOrder_of_finite e'
  /-
    case intro.intro
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    Φ : Set M
    hΦ₁ : Φ.Finite
    hΦ₂ : Eq (Submodule.span R Φ) Top.top
    e : LinearEquiv (RingHom.id R) M M
    he : Set.BijOn (⇑e) Φ Φ
    e' : Equiv ↑Φ ↑Φ := Set.BijOn.equiv (⇑e) he
    this : Finite ↑Φ
    k : Nat
    hk₀ : GT.gt k 0
    hk : Function.IsPeriodicPt (fun x => HMul.hMul e' x) k 1
    ⊢ IsOfFinOrder e
  -/
  refine ⟨k, hk₀, ?_⟩
  /-
    case intro.intro
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    Φ : Set M
    hΦ₁ : Φ.Finite
    hΦ₂ : Eq (Submodule.span R Φ) Top.top
    e : LinearEquiv (RingHom.id R) M M
    he : Set.BijOn (⇑e) Φ Φ
    e' : Equiv ↑Φ ↑Φ := Set.BijOn.equiv (⇑e) he
    this : Finite ↑Φ
    k : Nat
    hk₀ : GT.gt k 0
    hk : Function.IsPeriodicPt (fun x => HMul.hMul e' x) k 1
    ⊢ Function.IsPeriodicPt (fun x => HMul.hMul e x) k 1
  -/
  ext m
  /-
    case intro.intro.h
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    Φ : Set M
    hΦ₁ : Φ.Finite
    hΦ₂ : Eq (Submodule.span R Φ) Top.top
    e : LinearEquiv (RingHom.id R) M M
    he : Set.BijOn (⇑e) Φ Φ
    e' : Equiv ↑Φ ↑Φ := Set.BijOn.equiv (⇑e) he
    this : Finite ↑Φ
    k : Nat
    hk₀ : GT.gt k 0
    hk : Function.IsPeriodicPt (fun x => HMul.hMul e' x) k 1
    m : M
    ⊢ Eq ((Nat.iterate (fun x => HMul.hMul e x) k 1) m) (1 m)
  -/
  have hm : m ∈ span R Φ := hΦ₂ ▸ Submodule.mem_top
  /-
    case intro.intro.h
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    Φ : Set M
    hΦ₁ : Φ.Finite
    hΦ₂ : Eq (Submodule.span R Φ) Top.top
    e : LinearEquiv (RingHom.id R) M M
    he : Set.BijOn (⇑e) Φ Φ
    e' : Equiv ↑Φ ↑Φ := Set.BijOn.equiv (⇑e) he
    this : Finite ↑Φ
    k : Nat
    hk₀ : GT.gt k 0
    hk : Function.IsPeriodicPt (fun x => HMul.hMul e' x) k 1
    m : M
    hm : Membership.mem (Submodule.span R Φ) m
    ⊢ Eq ((Nat.iterate (fun x => HMul.hMul e x) k 1) m) (1 m)
  -/
  simp only [mul_left_iterate, mul_one, LinearEquiv.coe_one, id_eq]
  refine Submodule.span_induction (fun x hx ↦ ?_) (by simp)
    (fun x y _ _ hx hy ↦ by simp [map_add, hx, hy]) (fun t x _ hx ↦ by simp [map_smul, hx]) hm
  /-
    case intro.intro.h
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    Φ : Set M
    hΦ₁ : Φ.Finite
    hΦ₂ : Eq (Submodule.span R Φ) Top.top
    e : LinearEquiv (RingHom.id R) M M
    he : Set.BijOn (⇑e) Φ Φ
    e' : Equiv ↑Φ ↑Φ := Set.BijOn.equiv (⇑e) he
    this : Finite ↑Φ
    k : Nat
    hk₀ : GT.gt k 0
    hk : Function.IsPeriodicPt (fun x => HMul.hMul e' x) k 1
    m : M
    hm : Membership.mem (Submodule.span R Φ) m
    x : M
    hx : Membership.mem Φ x
    ⊢ Eq ((HPow.hPow e k) x) x
  -/
  rw [LinearEquiv.pow_apply, ← he.1.coe_iterate_restrict ⟨x, hx⟩ k]
  /-
    case intro.intro.h
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    Φ : Set M
    hΦ₁ : Φ.Finite
    hΦ₂ : Eq (Submodule.span R Φ) Top.top
    e : LinearEquiv (RingHom.id R) M M
    he : Set.BijOn (⇑e) Φ Φ
    e' : Equiv ↑Φ ↑Φ := Set.BijOn.equiv (⇑e) he
    this : Finite ↑Φ
    k : Nat
    hk₀ : GT.gt k 0
    hk : Function.IsPeriodicPt (fun x => HMul.hMul e' x) k 1
    m : M
    hm : Membership.mem (Submodule.span R Φ) m
    x : M
    hx : Membership.mem Φ x
    ⊢ Eq (↑(Nat.iterate (Set.MapsTo.restrict (⇑e) Φ Φ ⋯) k ⟨x, hx⟩)) x
  -/
  replace hk : (e') ^ k = 1 := by simpa [IsPeriodicPt, IsFixedPt] using hk
  /-
    case intro.intro.h
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    Φ : Set M
    hΦ₁ : Φ.Finite
    hΦ₂ : Eq (Submodule.span R Φ) Top.top
    e : LinearEquiv (RingHom.id R) M M
    he : Set.BijOn (⇑e) Φ Φ
    e' : Equiv ↑Φ ↑Φ := Set.BijOn.equiv (⇑e) he
    this : Finite ↑Φ
    k : Nat
    hk₀ : GT.gt k 0
    m : M
    hm : Membership.mem (Submodule.span R Φ) m
    x : M
    hx : Membership.mem Φ x
    hk : Eq (HPow.hPow e' k) 1
    ⊢ Eq (↑(Nat.iterate (Set.MapsTo.restrict (⇑e) Φ Φ ⋯) k ⟨x, hx⟩)) x
  -/
  replace hk := Equiv.congr_fun hk ⟨x, hx⟩
  /-
    case intro.intro.h
    R : Type u_1
    M : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    Φ : Set M
    hΦ₁ : Φ.Finite
    hΦ₂ : Eq (Submodule.span R Φ) Top.top
    e : LinearEquiv (RingHom.id R) M M
    he : Set.BijOn (⇑e) Φ Φ
    e' : Equiv ↑Φ ↑Φ := Set.BijOn.equiv (⇑e) he
    this : Finite ↑Φ
    k : Nat
    hk₀ : GT.gt k 0
    m : M
    hm : Membership.mem (Submodule.span R Φ) m
    x : M
    hx : Membership.mem Φ x
    hk : Eq ((HPow.hPow e' k) ⟨x, hx⟩) (1 ⟨x, hx⟩)
    ⊢ Eq (↑(Nat.iterate (Set.MapsTo.restrict (⇑e) Φ Φ ⋯) k ⟨x, hx⟩)) x
  -/
  rwa [Equiv.Perm.coe_one, id_eq, Subtype.ext_iff, Equiv.Perm.coe_pow] at hk
  /-
    🎉 no goals
  -/

