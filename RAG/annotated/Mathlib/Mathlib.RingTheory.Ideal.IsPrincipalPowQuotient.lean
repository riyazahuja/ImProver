/-- For a principal ideal `I`, `R ⧸ I ≃ₗ[R] I ^ n ⧸ I ^ (n + 1)`. To convert into a form
that uses the ideal of `R ⧸ I ^ (n + 1)`, compose with
`Ideal.powQuotPowSuccLinearEquivMapMkPowSuccPow`. -/
noncomputable
def quotEquivPowQuotPowSucc (h : I.IsPrincipal) (h': I ≠ ⊥) (n : ℕ) :
    (R ⧸ I) ≃ₗ[R] (I ^ n : Ideal R) ⧸ (I • ⊤ : Submodule R (I ^ n : Ideal R)) := by
  let f : (I ^ n : Ideal R) →ₗ[R] (I ^ n : Ideal R) ⧸ (I • ⊤ : Submodule R (I ^ n : Ideal R)) :=
    Submodule.mkQ _
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    I : Ideal R
    h : Submodule.IsPrincipal I
    h' : Ne I Bot.bot
    n : Nat
    f : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem (HPow.hPow I n)  …
    ⊢ LinearEquiv (RingHom.id R) (HasQuotient.Quotient R I) (HasQuotient.Quotient  …
  -/
  let ϖ := h.principal'.choose
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    I : Ideal R
    h : Submodule.IsPrincipal I
    h' : Ne I Bot.bot
    n : Nat
    f : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem (HPow.hPow I n)  …
    ϖ : R := ⋯.choose
    ⊢ LinearEquiv (RingHom.id R) (HasQuotient.Quotient R I) (HasQuotient.Quotient  …
  -/
  have hI : I = Ideal.span {ϖ} := h.principal'.choose_spec
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    I : Ideal R
    h : Submodule.IsPrincipal I
    h' : Ne I Bot.bot
    n : Nat
    f : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem (HPow.hPow I n)  …
    ϖ : R := ⋯.choose
    hI : Eq I (Ideal.span (Singleton.singleton ϖ))
    ⊢ LinearEquiv (RingHom.id R) (HasQuotient.Quotient R I) (HasQuotient.Quotient  …
  -/
  have hϖ : ϖ ^ n ∈ I ^ n := hI ▸ (Ideal.pow_mem_pow (Ideal.mem_span_singleton_self _) n)
  let g : R →ₗ[R] (I ^ n : Ideal R) := (LinearMap.mulRight R ϖ^n).codRestrict _ fun x ↦ by
    simp only [LinearMap.pow_mulRight, LinearMap.mulRight_apply, Ideal.submodule_span_eq]
    -- TODO: change argument of Ideal.pow_mem_of_mem
    exact Ideal.mul_mem_left _ _ hϖ
  have : I = LinearMap.ker (f.comp g) := by
    ext x
    simp only [LinearMap.codRestrict, LinearMap.pow_mulRight, LinearMap.mulRight_apply,
      LinearMap.mem_ker, LinearMap.coe_comp, LinearMap.coe_mk, AddHom.coe_mk, Function.comp_apply,
      Submodule.mkQ_apply, Submodule.Quotient.mk_eq_zero, Submodule.mem_smul_top_iff, smul_eq_mul,
      f, g]
    constructor <;> intro hx
    · exact Submodule.mul_mem_mul hx hϖ
    · rw [← pow_succ', hI, Ideal.span_singleton_pow, Ideal.mem_span_singleton] at hx
      obtain ⟨y, hy⟩ := hx
      rw [mul_comm, pow_succ, mul_assoc, mul_right_inj' (pow_ne_zero _ _)] at hy
      · rw [hI, Ideal.mem_span_singleton]
        exact ⟨y, hy⟩
      · contrapose! h'
        rw [hI, h', Ideal.span_singleton_eq_bot]
  let e : (R ⧸ I) ≃ₗ[R] R ⧸ (LinearMap.ker (f.comp g)) :=
    Submodule.quotEquivOfEq I (LinearMap.ker (f ∘ₗ g)) this
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    I : Ideal R
    h : Submodule.IsPrincipal I
    h' : Ne I Bot.bot
    n : Nat
    f : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem (HPow.hPow I n)  …
    ϖ : R := ⋯.choose
    hI : Eq I (Ideal.span (Singleton.singleton ϖ))
    hϖ : Membership.mem (HPow.hPow I n) (HPow.hPow ϖ n)
    g : LinearMap (RingHom.id R) R (Subtype fun x => Membership.mem (HPow.hPow I n …
    this : Eq I (LinearMap.ker (f.comp g))
    e : LinearEquiv (RingHom.id R) (HasQuotient.Quotient R I) (HasQuotient.Quotien …
    ⊢ LinearEquiv (RingHom.id R) (HasQuotient.Quotient R I) (HasQuotient.Quotient  …
  -/
  refine e.trans ((f.comp g).quotKerEquivOfSurjective ?_)
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    I : Ideal R
    h : Submodule.IsPrincipal I
    h' : Ne I Bot.bot
    n : Nat
    f : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem (HPow.hPow I n)  …
    ϖ : R := ⋯.choose
    hI : Eq I (Ideal.span (Singleton.singleton ϖ))
    hϖ : Membership.mem (HPow.hPow I n) (HPow.hPow ϖ n)
    g : LinearMap (RingHom.id R) R (Subtype fun x => Membership.mem (HPow.hPow I n …
    this : Eq I (LinearMap.ker (f.comp g))
    e : LinearEquiv (RingHom.id R) (HasQuotient.Quotient R I) (HasQuotient.Quotien …
    ⊢ Function.Surjective ⇑(f.comp g)
  -/
  refine (Submodule.mkQ_surjective _).comp ?_
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    I : Ideal R
    h : Submodule.IsPrincipal I
    h' : Ne I Bot.bot
    n : Nat
    f : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem (HPow.hPow I n)  …
    ϖ : R := ⋯.choose
    hI : Eq I (Ideal.span (Singleton.singleton ϖ))
    hϖ : Membership.mem (HPow.hPow I n) (HPow.hPow ϖ n)
    g : LinearMap (RingHom.id R) R (Subtype fun x => Membership.mem (HPow.hPow I n …
    this : Eq I (LinearMap.ker (f.comp g))
    e : LinearEquiv (RingHom.id R) (HasQuotient.Quotient R I) (HasQuotient.Quotien …
    ⊢ Function.Surjective ⇑g
  -/
  rintro ⟨x, hx⟩
  /-
    case mk
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    I : Ideal R
    h : Submodule.IsPrincipal I
    h' : Ne I Bot.bot
    n : Nat
    f : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem (HPow.hPow I n)  …
    ϖ : R := ⋯.choose
    hI : Eq I (Ideal.span (Singleton.singleton ϖ))
    hϖ : Membership.mem (HPow.hPow I n) (HPow.hPow ϖ n)
    g : LinearMap (RingHom.id R) R (Subtype fun x => Membership.mem (HPow.hPow I n …
    this : Eq I (LinearMap.ker (f.comp g))
    e : LinearEquiv (RingHom.id R) (HasQuotient.Quotient R I) (HasQuotient.Quotien …
    x : R
    hx : Membership.mem (HPow.hPow I n) x
    ⊢ Exists fun a => Eq (g a) ⟨x, hx⟩
  -/
  rw [hI, Ideal.span_singleton_pow, Ideal.mem_span_singleton] at hx
  /-
    case mk
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    I : Ideal R
    h : Submodule.IsPrincipal I
    h' : Ne I Bot.bot
    n : Nat
    f : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem (HPow.hPow I n)  …
    ϖ : R := ⋯.choose
    hI : Eq I (Ideal.span (Singleton.singleton ϖ))
    hϖ : Membership.mem (HPow.hPow I n) (HPow.hPow ϖ n)
    g : LinearMap (RingHom.id R) R (Subtype fun x => Membership.mem (HPow.hPow I n …
    this : Eq I (LinearMap.ker (f.comp g))
    e : LinearEquiv (RingHom.id R) (HasQuotient.Quotient R I) (HasQuotient.Quotien …
    x : R
    hx✝ : Membership.mem (HPow.hPow I n) x
    hx : Dvd.dvd (HPow.hPow ϖ n) x
    ⊢ Exists fun a => Eq (g a) ⟨x, hx✝⟩
  -/
  refine hx.imp ?_
  /-
    case mk
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    I : Ideal R
    h : Submodule.IsPrincipal I
    h' : Ne I Bot.bot
    n : Nat
    f : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem (HPow.hPow I n)  …
    ϖ : R := ⋯.choose
    hI : Eq I (Ideal.span (Singleton.singleton ϖ))
    hϖ : Membership.mem (HPow.hPow I n) (HPow.hPow ϖ n)
    g : LinearMap (RingHom.id R) R (Subtype fun x => Membership.mem (HPow.hPow I n …
    this : Eq I (LinearMap.ker (f.comp g))
    e : LinearEquiv (RingHom.id R) (HasQuotient.Quotient R I) (HasQuotient.Quotien …
    x : R
    hx✝ : Membership.mem (HPow.hPow I n) x
    hx : Dvd.dvd (HPow.hPow ϖ n) x
    ⊢ ∀ (a : R), Eq x (HMul.hMul (HPow.hPow ϖ n) a) → Eq (g a) ⟨x, hx✝⟩
  -/
  simp [g, LinearMap.codRestrict, eq_comm, mul_comm]
  /-
    🎉 no goals
  -/


/-- For a principal ideal `I`, `R ⧸ I ≃ I ^ n ⧸ I ^ (n + 1)`. Supplied as a plain equiv to bypass
typeclass synthesis issues on complex `Module` goals.  To convert into a form
that uses the ideal of `R ⧸ I ^ (n + 1)`, compose with
`Ideal.powQuotPowSuccEquivMapMkPowSuccPow`. -/
noncomputable
def quotEquivPowQuotPowSuccEquiv (h : I.IsPrincipal) (h': I ≠ ⊥) (n : ℕ) :
    (R ⧸ I) ≃ (I ^ n : Ideal R) ⧸ (I • ⊤ : Submodule R (I ^ n : Ideal R)) :=
  quotEquivPowQuotPowSucc h h' n


