private noncomputable def mapPreimageDelta (hf : Function.Surjective f) (x : AdicCauchySequence I N)
    {n : ℕ} {y yₙ : M} (hy : f y = x (n + 1)) (hyₙ : f yₙ = x n) :
    {d : (I ^ n • ⊤ : Submodule R M) | f d = f (yₙ - y) } :=
  have h : f (yₙ - y) ∈ Submodule.map f (I ^ n • ⊤ : Submodule R M) := by
    rw [Submodule.map_smul'', Submodule.map_top, LinearMap.range_eq_top.2 hf,
      map_sub, hyₙ, hy, ← Submodule.neg_mem_iff, neg_sub, ← SModEq.sub_mem]
    /-
      R : Type u
      inst✝⁴ : CommRing R
      I : Ideal R
      M : Type v
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      N : Type w
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      f : LinearMap (RingHom.id R) M N
      hf : Function.Surjective ⇑f
      x : AdicCompletion.AdicCauchySequence I N
      n : Nat
      y yₙ : M
      hy : Eq (f y) (↑x (HAdd.hAdd n 1))
      hyₙ : Eq (f yₙ) (↑x n)
      ⊢ SModEq (HSMul.hSMul (HPow.hPow I n) Top.top) (↑x (HAdd.hAdd n 1)) (↑x n)
    -/
    exact AdicCauchySequence.mk_eq_mk (Nat.le_succ n) x
    /-
      🎉 no goals
    -/
  ⟨⟨h.choose, h.choose_spec.1⟩, h.choose_spec.2⟩

/- Inductively construct preimage of cauchy sequence. -/

private noncomputable def mapPreimage (hf : Function.Surjective f) (x : AdicCauchySequence I N) :
    (n : ℕ) → f ⁻¹' {x n}
  | .zero => ⟨(hf (x 0)).choose, (hf (x 0)).choose_spec⟩
  | .succ n =>
      let y := (hf (x (n + 1))).choose
      have hy := (hf (x (n + 1))).choose_spec
      let ⟨yₙ, (hyₙ : f yₙ = x n)⟩ := mapPreimage hf x n
      let ⟨⟨d, _⟩, (p : f d = f (yₙ - y))⟩ := mapPreimageDelta hf x hy hyₙ
                  /-
                    R : Type u
                    inst✝⁴ : CommRing R
                    I : Ideal R
                    M : Type v
                    inst✝³ : AddCommGroup M
                    inst✝² : Module R M
                    N : Type w
                    inst✝¹ : AddCommGroup N
                    inst✝ : Module R N
                    f : LinearMap (RingHom.id R) M N
                    hf : Function.Surjective ⇑f
                    x : AdicCompletion.AdicCauchySequence I N
                    n : Nat
                    y : M := ⋯.choose
                    hy : Eq (f ⋯.choose) (↑x (HAdd.hAdd n 1))
                    yₙ : M
                    hyₙ : Eq (f yₙ) (↑x n)
                    d : M
                    property✝ : Membership.mem (HSMul.hSMul (HPow.hPow I n) Top.top) d
                    p : Eq (f d) (f (HSub.hSub yₙ y))
                    ⊢ Membership.mem (Set.preimage (⇑f) (Singleton.singleton (↑x n.succ))) (HSub.h …
                  -/
      ⟨yₙ - d, by simpa [p]⟩
                  /-
                    🎉 no goals
                  -/


variable (I) in
/-- Adic completion preserves surjectivity -/
theorem map_surjective (hf : Function.Surjective f) : Function.Surjective (map I f) := fun y ↦ by
  /-
    R : Type u
    inst✝⁴ : CommRing R
    I : Ideal R
    M : Type v
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type w
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) M N
    hf : Function.Surjective ⇑f
    y : AdicCompletion I N
    ⊢ Exists fun a => Eq ((AdicCompletion.map I f) a) y
  -/
  apply AdicCompletion.induction_on I N y (fun b ↦ ?_)
  /-
    R : Type u
    inst✝⁴ : CommRing R
    I : Ideal R
    M : Type v
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type w
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) M N
    hf : Function.Surjective ⇑f
    y : AdicCompletion I N
    b : AdicCompletion.AdicCauchySequence I N
    ⊢ Exists fun a => Eq ((AdicCompletion.map I f) a) ((AdicCompletion.mk I N) b)
  -/
  let a := mapPreimage hf b
  /-
    R : Type u
    inst✝⁴ : CommRing R
    I : Ideal R
    M : Type v
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    N : Type w
    inst✝¹ : AddCommGroup N
    inst✝ : Module R N
    f : LinearMap (RingHom.id R) M N
    hf : Function.Surjective ⇑f
    y : AdicCompletion I N
    b : AdicCompletion.AdicCauchySequence I N
    a : (n : Nat) → ↑(Set.preimage (⇑f) (Singleton.singleton (↑b n))) := AdicCompl …
    ⊢ Exists fun a => Eq ((AdicCompletion.map I f) a) ((AdicCompletion.mk I N) b)
  -/
  refine ⟨AdicCompletion.mk I M (AdicCauchySequence.mk I M (fun n ↦ (a n : M)) ?_), ?_⟩
    /-
      case refine_1
      R : Type u
      inst✝⁴ : CommRing R
      I : Ideal R
      M : Type v
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      N : Type w
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      f : LinearMap (RingHom.id R) M N
      hf : Function.Surjective ⇑f
      y : AdicCompletion I N
      b : AdicCompletion.AdicCauchySequence I N
      a : (n : Nat) → ↑(Set.preimage (⇑f) (Singleton.singleton (↑b n))) := AdicCompl …
      ⊢ ∀ (n : Nat), SModEq (HSMul.hSMul (HPow.hPow I n) Top.top) ((fun n => ↑(a n)) …
    -/
  · refine fun n ↦ SModEq.symm ?_
    simp only [SModEq.symm, SModEq, mapPreimage, Submodule.Quotient.mk_sub,
      sub_eq_self, Submodule.Quotient.mk_eq_zero, SetLike.coe_mem, a]
    /-
      case refine_2
      R : Type u
      inst✝⁴ : CommRing R
      I : Ideal R
      M : Type v
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      N : Type w
      inst✝¹ : AddCommGroup N
      inst✝ : Module R N
      f : LinearMap (RingHom.id R) M N
      hf : Function.Surjective ⇑f
      y : AdicCompletion I N
      b : AdicCompletion.AdicCauchySequence I N
      a : (n : Nat) → ↑(Set.preimage (⇑f) (Singleton.singleton (↑b n))) := AdicCompl …
      ⊢ Eq ((AdicCompletion.map I f) ((AdicCompletion.mk I M) (AdicCompletion.AdicCa …
    -/
  · exact _root_.AdicCompletion.ext fun n ↦ congrArg _ ((a n).property)
    /-
      🎉 no goals
    -/


/-- Adic completion preserves injectivity of finite modules over a Noetherian ring. -/
theorem map_injective {f : M →ₗ[R] N} (hf : Function.Injective f) :
    Function.Injective (map I f) := by
  /-
    R : Type u
    inst✝⁶ : CommRing R
    I : Ideal R
    M : Type u
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    N : Type u
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    inst✝¹ : IsNoetherianRing R
    inst✝ : Module.Finite R N
    f : LinearMap (RingHom.id R) M N
    hf : Function.Injective ⇑f
    ⊢ Function.Injective ⇑(AdicCompletion.map I f)
  -/
  obtain ⟨k, hk⟩ := Ideal.exists_pow_inf_eq_pow_smul I (range f)
  /-
    case intro
    R : Type u
    inst✝⁶ : CommRing R
    I : Ideal R
    M : Type u
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    N : Type u
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    inst✝¹ : IsNoetherianRing R
    inst✝ : Module.Finite R N
    f : LinearMap (RingHom.id R) M N
    hf : Function.Injective ⇑f
    k : Nat
    hk : ∀ (n : Nat), GE.ge n k → Eq (Min.min (HSMul.hSMul (HPow.hPow I n) Top.top …
    ⊢ Function.Injective ⇑(AdicCompletion.map I f)
  -/
  rw [← LinearMap.ker_eq_bot, LinearMap.ker_eq_bot']
  /-
    case intro
    R : Type u
    inst✝⁶ : CommRing R
    I : Ideal R
    M : Type u
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    N : Type u
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    inst✝¹ : IsNoetherianRing R
    inst✝ : Module.Finite R N
    f : LinearMap (RingHom.id R) M N
    hf : Function.Injective ⇑f
    k : Nat
    hk : ∀ (n : Nat), GE.ge n k → Eq (Min.min (HSMul.hSMul (HPow.hPow I n) Top.top …
    ⊢ ∀ (m : AdicCompletion I M), Eq ((AdicCompletion.map I f) m) 0 → Eq m 0
  -/
  intro x
  /-
    case intro
    R : Type u
    inst✝⁶ : CommRing R
    I : Ideal R
    M : Type u
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    N : Type u
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    inst✝¹ : IsNoetherianRing R
    inst✝ : Module.Finite R N
    f : LinearMap (RingHom.id R) M N
    hf : Function.Injective ⇑f
    k : Nat
    hk : ∀ (n : Nat), GE.ge n k → Eq (Min.min (HSMul.hSMul (HPow.hPow I n) Top.top …
    x : AdicCompletion I M
    ⊢ Eq ((AdicCompletion.map I f) x) 0 → Eq x 0
  -/
  apply AdicCompletion.induction_on I M x (fun a ↦ ?_)
  /-
    R : Type u
    inst✝⁶ : CommRing R
    I : Ideal R
    M : Type u
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    N : Type u
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    inst✝¹ : IsNoetherianRing R
    inst✝ : Module.Finite R N
    f : LinearMap (RingHom.id R) M N
    hf : Function.Injective ⇑f
    k : Nat
    hk : ∀ (n : Nat), GE.ge n k → Eq (Min.min (HSMul.hSMul (HPow.hPow I n) Top.top …
    x : AdicCompletion I M
    a : AdicCompletion.AdicCauchySequence I M
    ⊢ Eq ((AdicCompletion.map I f) ((AdicCompletion.mk I M) a)) 0 → Eq ((AdicCompl …
  -/
  intro hx
  /-
    R : Type u
    inst✝⁶ : CommRing R
    I : Ideal R
    M : Type u
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    N : Type u
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    inst✝¹ : IsNoetherianRing R
    inst✝ : Module.Finite R N
    f : LinearMap (RingHom.id R) M N
    hf : Function.Injective ⇑f
    k : Nat
    hk : ∀ (n : Nat), GE.ge n k → Eq (Min.min (HSMul.hSMul (HPow.hPow I n) Top.top …
    x : AdicCompletion I M
    a : AdicCompletion.AdicCauchySequence I M
    hx : Eq ((AdicCompletion.map I f) ((AdicCompletion.mk I M) a)) 0
    ⊢ Eq ((AdicCompletion.mk I M) a) 0
  -/
  refine AdicCompletion.mk_zero_of _ _ _ ⟨42, fun n _ ↦ ⟨n + k, by omega, n, by omega, ?_⟩⟩
  rw [← Submodule.comap_map_eq_of_injective hf (I ^ n • ⊤ : Submodule R M),
    Submodule.map_smul'', Submodule.map_top]
  /-
    R : Type u
    inst✝⁶ : CommRing R
    I : Ideal R
    M : Type u
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    N : Type u
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    inst✝¹ : IsNoetherianRing R
    inst✝ : Module.Finite R N
    f : LinearMap (RingHom.id R) M N
    hf : Function.Injective ⇑f
    k : Nat
    hk : ∀ (n : Nat), GE.ge n k → Eq (Min.min (HSMul.hSMul (HPow.hPow I n) Top.top …
    x : AdicCompletion I M
    a : AdicCompletion.AdicCauchySequence I M
    hx : Eq ((AdicCompletion.map I f) ((AdicCompletion.mk I M) a)) 0
    n : Nat
    x✝ : GE.ge n 42
    ⊢ Membership.mem (Submodule.comap f (HSMul.hSMul (HPow.hPow I n) (LinearMap.ra …
  -/
  apply (smul_mono_right _ inf_le_right : I ^ n • (I ^ k • ⊤ ⊓ (range f)) ≤ _)
  /-
    case a
    R : Type u
    inst✝⁶ : CommRing R
    I : Ideal R
    M : Type u
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    N : Type u
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    inst✝¹ : IsNoetherianRing R
    inst✝ : Module.Finite R N
    f : LinearMap (RingHom.id R) M N
    hf : Function.Injective ⇑f
    k : Nat
    hk : ∀ (n : Nat), GE.ge n k → Eq (Min.min (HSMul.hSMul (HPow.hPow I n) Top.top …
    x : AdicCompletion I M
    a : AdicCompletion.AdicCauchySequence I M
    hx : Eq ((AdicCompletion.map I f) ((AdicCompletion.mk I M) a)) 0
    n : Nat
    x✝ : GE.ge n 42
    ⊢ Membership.mem (HSMul.hSMul (HPow.hPow I n) (Min.min (HSMul.hSMul (HPow.hPow …
  -/
  nth_rw 1 [show n = n + k - k by omega]
  /-
    case a
    R : Type u
    inst✝⁶ : CommRing R
    I : Ideal R
    M : Type u
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    N : Type u
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    inst✝¹ : IsNoetherianRing R
    inst✝ : Module.Finite R N
    f : LinearMap (RingHom.id R) M N
    hf : Function.Injective ⇑f
    k : Nat
    hk : ∀ (n : Nat), GE.ge n k → Eq (Min.min (HSMul.hSMul (HPow.hPow I n) Top.top …
    x : AdicCompletion I M
    a : AdicCompletion.AdicCauchySequence I M
    hx : Eq ((AdicCompletion.map I f) ((AdicCompletion.mk I M) a)) 0
    n : Nat
    x✝ : GE.ge n 42
    ⊢ Membership.mem (HSMul.hSMul (HPow.hPow I (HSub.hSub (HAdd.hAdd n k) k)) (Min …
  -/
  rw [← hk (n + k) (show n + k ≥ k by omega)]
  /-
    case a
    R : Type u
    inst✝⁶ : CommRing R
    I : Ideal R
    M : Type u
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Module R M
    N : Type u
    inst✝³ : AddCommGroup N
    inst✝² : Module R N
    inst✝¹ : IsNoetherianRing R
    inst✝ : Module.Finite R N
    f : LinearMap (RingHom.id R) M N
    hf : Function.Injective ⇑f
    k : Nat
    hk : ∀ (n : Nat), GE.ge n k → Eq (Min.min (HSMul.hSMul (HPow.hPow I n) Top.top …
    x : AdicCompletion I M
    a : AdicCompletion.AdicCauchySequence I M
    hx : Eq ((AdicCompletion.map I f) ((AdicCompletion.mk I M) a)) 0
    n : Nat
    x✝ : GE.ge n 42
    ⊢ Membership.mem (Min.min (HSMul.hSMul (HPow.hPow I (HAdd.hAdd n k)) Top.top)  …
  -/
  exact ⟨by simpa using congrArg (fun x ↦ x.val (n + k)) hx, ⟨a (n + k), rfl⟩⟩
  /-
    🎉 no goals
  -/


private noncomputable def mapExactAuxDelta {n : ℕ} {d : N}
    (hdmem : d ∈ (I ^ (k + n + 1) • ⊤ : Submodule R N)) {y yₙ : M}
    (hd : f y = x (k + n + 1) - d) (hyₙ : f yₙ - x (k + n) ∈ (I ^ (k + n) • ⊤ : Submodule R N)) :
    { d : (I ^ n • ⊤ : Submodule R M)
      | f (yₙ + d) - x (k + n + 1) ∈ (I ^ (k + n + 1) • ⊤ : Submodule R N) } :=
  have h : f (y - yₙ) ∈ (I ^ (k + n) • ⊤ : Submodule R N) := by
    /-
      R : Type u
      inst✝⁸ : CommRing R
      I : Ideal R
      M : Type u
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      N : Type u
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R N
      P : Type u
      inst✝³ : AddCommGroup P
      inst✝² : Module R P
      inst✝¹ : IsNoetherianRing R
      inst✝ : Module.Finite R N
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      hf : Function.Injective ⇑f
      hfg : Function.Exact ⇑f ⇑g
      hg : Function.Surjective ⇑g
      k : Nat
      hkn : ∀ (n : Nat), GE.ge n k → Eq (Min.min (HSMul.hSMul (HPow.hPow I n) Top.to …
      x : AdicCompletion.AdicCauchySequence I N
      hker : ∀ (n : Nat), Membership.mem (HSMul.hSMul (HPow.hPow I n) Top.top) (g (↑ …
      n : Nat
      d : N
      hdmem : Membership.mem (HSMul.hSMul (HPow.hPow I (HAdd.hAdd (HAdd.hAdd k n) 1) …
      y yₙ : M
      hd : Eq (f y) (HSub.hSub (↑x (HAdd.hAdd (HAdd.hAdd k n) 1)) d)
      hyₙ : Membership.mem (HSMul.hSMul (HPow.hPow I (HAdd.hAdd k n)) Top.top) (HSub …
      ⊢ Membership.mem (HSMul.hSMul (HPow.hPow I (HAdd.hAdd k n)) Top.top) (f (HSub. …
    -/
    simp only [map_sub, hd]
    /-
      R : Type u
      inst✝⁸ : CommRing R
      I : Ideal R
      M : Type u
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      N : Type u
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R N
      P : Type u
      inst✝³ : AddCommGroup P
      inst✝² : Module R P
      inst✝¹ : IsNoetherianRing R
      inst✝ : Module.Finite R N
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      hf : Function.Injective ⇑f
      hfg : Function.Exact ⇑f ⇑g
      hg : Function.Surjective ⇑g
      k : Nat
      hkn : ∀ (n : Nat), GE.ge n k → Eq (Min.min (HSMul.hSMul (HPow.hPow I n) Top.to …
      x : AdicCompletion.AdicCauchySequence I N
      hker : ∀ (n : Nat), Membership.mem (HSMul.hSMul (HPow.hPow I n) Top.top) (g (↑ …
      n : Nat
      d : N
      hdmem : Membership.mem (HSMul.hSMul (HPow.hPow I (HAdd.hAdd (HAdd.hAdd k n) 1) …
      y yₙ : M
      hd : Eq (f y) (HSub.hSub (↑x (HAdd.hAdd (HAdd.hAdd k n) 1)) d)
      hyₙ : Membership.mem (HSMul.hSMul (HPow.hPow I (HAdd.hAdd k n)) Top.top) (HSub …
      ⊢ Membership.mem (HSMul.hSMul (HPow.hPow I (HAdd.hAdd k n)) Top.top) (HSub.hSu …
    -/
    convert_to x (k + n + 1) - x (k + n) - d - (f yₙ - x (k + n)) ∈ I ^ (k + n) • ⊤
      /-
        case h.e'_1
        R : Type u
        inst✝⁸ : CommRing R
        I : Ideal R
        M : Type u
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : Module R M
        N : Type u
        inst✝⁵ : AddCommGroup N
        inst✝⁴ : Module R N
        P : Type u
        inst✝³ : AddCommGroup P
        inst✝² : Module R P
        inst✝¹ : IsNoetherianRing R
        inst✝ : Module.Finite R N
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        hf : Function.Injective ⇑f
        hfg : Function.Exact ⇑f ⇑g
        hg : Function.Surjective ⇑g
        k : Nat
        hkn : ∀ (n : Nat), GE.ge n k → Eq (Min.min (HSMul.hSMul (HPow.hPow I n) Top.to …
        x : AdicCompletion.AdicCauchySequence I N
        hker : ∀ (n : Nat), Membership.mem (HSMul.hSMul (HPow.hPow I n) Top.top) (g (↑ …
        n : Nat
        d : N
        hdmem : Membership.mem (HSMul.hSMul (HPow.hPow I (HAdd.hAdd (HAdd.hAdd k n) 1) …
        y yₙ : M
        hd : Eq (f y) (HSub.hSub (↑x (HAdd.hAdd (HAdd.hAdd k n) 1)) d)
        hyₙ : Membership.mem (HSMul.hSMul (HPow.hPow I (HAdd.hAdd k n)) Top.top) (HSub …
        ⊢ Eq (HSub.hSub (HSub.hSub (↑x (HAdd.hAdd (HAdd.hAdd k n) 1)) d) (f yₙ)) (HSub …
      -/
      /-
        🎉 no goals
      -/
    · abel
      /-
        🎉 no goals
      -/
      /-
        case convert_6
        R : Type u
        inst✝⁸ : CommRing R
        I : Ideal R
        M : Type u
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : Module R M
        N : Type u
        inst✝⁵ : AddCommGroup N
        inst✝⁴ : Module R N
        P : Type u
        inst✝³ : AddCommGroup P
        inst✝² : Module R P
        inst✝¹ : IsNoetherianRing R
        inst✝ : Module.Finite R N
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        hf : Function.Injective ⇑f
        hfg : Function.Exact ⇑f ⇑g
        hg : Function.Surjective ⇑g
        k : Nat
        hkn : ∀ (n : Nat), GE.ge n k → Eq (Min.min (HSMul.hSMul (HPow.hPow I n) Top.to …
        x : AdicCompletion.AdicCauchySequence I N
        hker : ∀ (n : Nat), Membership.mem (HSMul.hSMul (HPow.hPow I n) Top.top) (g (↑ …
        n : Nat
        d : N
        hdmem : Membership.mem (HSMul.hSMul (HPow.hPow I (HAdd.hAdd (HAdd.hAdd k n) 1) …
        y yₙ : M
        hd : Eq (f y) (HSub.hSub (↑x (HAdd.hAdd (HAdd.hAdd k n) 1)) d)
        hyₙ : Membership.mem (HSMul.hSMul (HPow.hPow I (HAdd.hAdd k n)) Top.top) (HSub …
        ⊢ Membership.mem (HSMul.hSMul (HPow.hPow I (HAdd.hAdd k n)) Top.top) (HSub.hSu …
      -/
    · refine Submodule.sub_mem _ (Submodule.sub_mem _ ?_ ?_) hyₙ
        /-
          case convert_6.refine_1
          R : Type u
          inst✝⁸ : CommRing R
          I : Ideal R
          M : Type u
          inst✝⁷ : AddCommGroup M
          inst✝⁶ : Module R M
          N : Type u
          inst✝⁵ : AddCommGroup N
          inst✝⁴ : Module R N
          P : Type u
          inst✝³ : AddCommGroup P
          inst✝² : Module R P
          inst✝¹ : IsNoetherianRing R
          inst✝ : Module.Finite R N
          f : LinearMap (RingHom.id R) M N
          g : LinearMap (RingHom.id R) N P
          hf : Function.Injective ⇑f
          hfg : Function.Exact ⇑f ⇑g
          hg : Function.Surjective ⇑g
          k : Nat
          hkn : ∀ (n : Nat), GE.ge n k → Eq (Min.min (HSMul.hSMul (HPow.hPow I n) Top.to …
          x : AdicCompletion.AdicCauchySequence I N
          hker : ∀ (n : Nat), Membership.mem (HSMul.hSMul (HPow.hPow I n) Top.top) (g (↑ …
          n : Nat
          d : N
          hdmem : Membership.mem (HSMul.hSMul (HPow.hPow I (HAdd.hAdd (HAdd.hAdd k n) 1) …
          y yₙ : M
          hd : Eq (f y) (HSub.hSub (↑x (HAdd.hAdd (HAdd.hAdd k n) 1)) d)
          hyₙ : Membership.mem (HSMul.hSMul (HPow.hPow I (HAdd.hAdd k n)) Top.top) (HSub …
          ⊢ Membership.mem (HSMul.hSMul (HPow.hPow I (HAdd.hAdd k n)) Top.top) (HSub.hSu …
        -/
      · rw [← Submodule.Quotient.eq]
        /-
          case convert_6.refine_1
          R : Type u
          inst✝⁸ : CommRing R
          I : Ideal R
          M : Type u
          inst✝⁷ : AddCommGroup M
          inst✝⁶ : Module R M
          N : Type u
          inst✝⁵ : AddCommGroup N
          inst✝⁴ : Module R N
          P : Type u
          inst✝³ : AddCommGroup P
          inst✝² : Module R P
          inst✝¹ : IsNoetherianRing R
          inst✝ : Module.Finite R N
          f : LinearMap (RingHom.id R) M N
          g : LinearMap (RingHom.id R) N P
          hf : Function.Injective ⇑f
          hfg : Function.Exact ⇑f ⇑g
          hg : Function.Surjective ⇑g
          k : Nat
          hkn : ∀ (n : Nat), GE.ge n k → Eq (Min.min (HSMul.hSMul (HPow.hPow I n) Top.to …
          x : AdicCompletion.AdicCauchySequence I N
          hker : ∀ (n : Nat), Membership.mem (HSMul.hSMul (HPow.hPow I n) Top.top) (g (↑ …
          n : Nat
          d : N
          hdmem : Membership.mem (HSMul.hSMul (HPow.hPow I (HAdd.hAdd (HAdd.hAdd k n) 1) …
          y yₙ : M
          hd : Eq (f y) (HSub.hSub (↑x (HAdd.hAdd (HAdd.hAdd k n) 1)) d)
          hyₙ : Membership.mem (HSMul.hSMul (HPow.hPow I (HAdd.hAdd k n)) Top.top) (HSub …
          ⊢ Eq (Submodule.Quotient.mk (↑x (HAdd.hAdd (HAdd.hAdd k n) 1))) (Submodule.Quo …
        -/
        exact AdicCauchySequence.mk_eq_mk (by omega) _
        /-
          🎉 no goals
        -/
        /-
          case convert_6.refine_2
          R : Type u
          inst✝⁸ : CommRing R
          I : Ideal R
          M : Type u
          inst✝⁷ : AddCommGroup M
          inst✝⁶ : Module R M
          N : Type u
          inst✝⁵ : AddCommGroup N
          inst✝⁴ : Module R N
          P : Type u
          inst✝³ : AddCommGroup P
          inst✝² : Module R P
          inst✝¹ : IsNoetherianRing R
          inst✝ : Module.Finite R N
          f : LinearMap (RingHom.id R) M N
          g : LinearMap (RingHom.id R) N P
          hf : Function.Injective ⇑f
          hfg : Function.Exact ⇑f ⇑g
          hg : Function.Surjective ⇑g
          k : Nat
          hkn : ∀ (n : Nat), GE.ge n k → Eq (Min.min (HSMul.hSMul (HPow.hPow I n) Top.to …
          x : AdicCompletion.AdicCauchySequence I N
          hker : ∀ (n : Nat), Membership.mem (HSMul.hSMul (HPow.hPow I n) Top.top) (g (↑ …
          n : Nat
          d : N
          hdmem : Membership.mem (HSMul.hSMul (HPow.hPow I (HAdd.hAdd (HAdd.hAdd k n) 1) …
          y yₙ : M
          hd : Eq (f y) (HSub.hSub (↑x (HAdd.hAdd (HAdd.hAdd k n) 1)) d)
          hyₙ : Membership.mem (HSMul.hSMul (HPow.hPow I (HAdd.hAdd k n)) Top.top) (HSub …
          ⊢ Membership.mem (HSMul.hSMul (HPow.hPow I (HAdd.hAdd k n)) Top.top) d
        -/
      · exact (Submodule.smul_mono_left (Ideal.pow_le_pow_right (by omega))) hdmem
        /-
          🎉 no goals
        -/
  have hincl : I ^ (k + n - k) • (I ^ k • ⊤ ⊓ range f) ≤ I ^ (k + n - k) • (range f) :=
    smul_mono_right _ inf_le_right
  have hyyₙ : y - yₙ ∈ (I ^ n • ⊤ : Submodule R M) := by
    /-
      R : Type u
      inst✝⁸ : CommRing R
      I : Ideal R
      M : Type u
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      N : Type u
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R N
      P : Type u
      inst✝³ : AddCommGroup P
      inst✝² : Module R P
      inst✝¹ : IsNoetherianRing R
      inst✝ : Module.Finite R N
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      hf : Function.Injective ⇑f
      hfg : Function.Exact ⇑f ⇑g
      hg : Function.Surjective ⇑g
      k : Nat
      hkn : ∀ (n : Nat), GE.ge n k → Eq (Min.min (HSMul.hSMul (HPow.hPow I n) Top.to …
      x : AdicCompletion.AdicCauchySequence I N
      hker : ∀ (n : Nat), Membership.mem (HSMul.hSMul (HPow.hPow I n) Top.top) (g (↑ …
      n : Nat
      d : N
      hdmem : Membership.mem (HSMul.hSMul (HPow.hPow I (HAdd.hAdd (HAdd.hAdd k n) 1) …
      y yₙ : M
      hd : Eq (f y) (HSub.hSub (↑x (HAdd.hAdd (HAdd.hAdd k n) 1)) d)
      hyₙ : Membership.mem (HSMul.hSMul (HPow.hPow I (HAdd.hAdd k n)) Top.top) (HSub …
      h : Membership.mem (HSMul.hSMul (HPow.hPow I (HAdd.hAdd k n)) Top.top) (f (HSu …
      hincl : LE.le (HSMul.hSMul (HPow.hPow I (HSub.hSub (HAdd.hAdd k n) k)) (Min.mi …
      ⊢ Membership.mem (HSMul.hSMul (HPow.hPow I n) Top.top) (HSub.hSub y yₙ)
    -/
    convert_to y - yₙ ∈ (I ^ (k + n - k) • ⊤ : Submodule R M)
      /-
        case h.e'_4
        R : Type u
        inst✝⁸ : CommRing R
        I : Ideal R
        M : Type u
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : Module R M
        N : Type u
        inst✝⁵ : AddCommGroup N
        inst✝⁴ : Module R N
        P : Type u
        inst✝³ : AddCommGroup P
        inst✝² : Module R P
        inst✝¹ : IsNoetherianRing R
        inst✝ : Module.Finite R N
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        hf : Function.Injective ⇑f
        hfg : Function.Exact ⇑f ⇑g
        hg : Function.Surjective ⇑g
        k : Nat
        hkn : ∀ (n : Nat), GE.ge n k → Eq (Min.min (HSMul.hSMul (HPow.hPow I n) Top.to …
        x : AdicCompletion.AdicCauchySequence I N
        hker : ∀ (n : Nat), Membership.mem (HSMul.hSMul (HPow.hPow I n) Top.top) (g (↑ …
        n : Nat
        d : N
        hdmem : Membership.mem (HSMul.hSMul (HPow.hPow I (HAdd.hAdd (HAdd.hAdd k n) 1) …
        y yₙ : M
        hd : Eq (f y) (HSub.hSub (↑x (HAdd.hAdd (HAdd.hAdd k n) 1)) d)
        hyₙ : Membership.mem (HSMul.hSMul (HPow.hPow I (HAdd.hAdd k n)) Top.top) (HSub …
        h : Membership.mem (HSMul.hSMul (HPow.hPow I (HAdd.hAdd k n)) Top.top) (f (HSu …
        hincl : LE.le (HSMul.hSMul (HPow.hPow I (HSub.hSub (HAdd.hAdd k n) k)) (Min.mi …
        ⊢ Eq (HSMul.hSMul (HPow.hPow I n) Top.top) (HSMul.hSMul (HPow.hPow I (HSub.hSu …
      -/
    · simp
      /-
        🎉 no goals
      -/
    · rw [← Submodule.comap_map_eq_of_injective hf (I ^ (k + n - k) • ⊤ : Submodule R M),
        Submodule.map_smul'', Submodule.map_top]
      /-
        R : Type u
        inst✝⁸ : CommRing R
        I : Ideal R
        M : Type u
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : Module R M
        N : Type u
        inst✝⁵ : AddCommGroup N
        inst✝⁴ : Module R N
        P : Type u
        inst✝³ : AddCommGroup P
        inst✝² : Module R P
        inst✝¹ : IsNoetherianRing R
        inst✝ : Module.Finite R N
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        hf : Function.Injective ⇑f
        hfg : Function.Exact ⇑f ⇑g
        hg : Function.Surjective ⇑g
        k : Nat
        hkn : ∀ (n : Nat), GE.ge n k → Eq (Min.min (HSMul.hSMul (HPow.hPow I n) Top.to …
        x : AdicCompletion.AdicCauchySequence I N
        hker : ∀ (n : Nat), Membership.mem (HSMul.hSMul (HPow.hPow I n) Top.top) (g (↑ …
        n : Nat
        d : N
        hdmem : Membership.mem (HSMul.hSMul (HPow.hPow I (HAdd.hAdd (HAdd.hAdd k n) 1) …
        y yₙ : M
        hd : Eq (f y) (HSub.hSub (↑x (HAdd.hAdd (HAdd.hAdd k n) 1)) d)
        hyₙ : Membership.mem (HSMul.hSMul (HPow.hPow I (HAdd.hAdd k n)) Top.top) (HSub …
        h : Membership.mem (HSMul.hSMul (HPow.hPow I (HAdd.hAdd k n)) Top.top) (f (HSu …
        hincl : LE.le (HSMul.hSMul (HPow.hPow I (HSub.hSub (HAdd.hAdd k n) k)) (Min.mi …
        ⊢ Membership.mem (Submodule.comap f (HSMul.hSMul (HPow.hPow I (HSub.hSub (HAdd …
      -/
      apply hincl
      /-
        case a
        R : Type u
        inst✝⁸ : CommRing R
        I : Ideal R
        M : Type u
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : Module R M
        N : Type u
        inst✝⁵ : AddCommGroup N
        inst✝⁴ : Module R N
        P : Type u
        inst✝³ : AddCommGroup P
        inst✝² : Module R P
        inst✝¹ : IsNoetherianRing R
        inst✝ : Module.Finite R N
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        hf : Function.Injective ⇑f
        hfg : Function.Exact ⇑f ⇑g
        hg : Function.Surjective ⇑g
        k : Nat
        hkn : ∀ (n : Nat), GE.ge n k → Eq (Min.min (HSMul.hSMul (HPow.hPow I n) Top.to …
        x : AdicCompletion.AdicCauchySequence I N
        hker : ∀ (n : Nat), Membership.mem (HSMul.hSMul (HPow.hPow I n) Top.top) (g (↑ …
        n : Nat
        d : N
        hdmem : Membership.mem (HSMul.hSMul (HPow.hPow I (HAdd.hAdd (HAdd.hAdd k n) 1) …
        y yₙ : M
        hd : Eq (f y) (HSub.hSub (↑x (HAdd.hAdd (HAdd.hAdd k n) 1)) d)
        hyₙ : Membership.mem (HSMul.hSMul (HPow.hPow I (HAdd.hAdd k n)) Top.top) (HSub …
        h : Membership.mem (HSMul.hSMul (HPow.hPow I (HAdd.hAdd k n)) Top.top) (f (HSu …
        hincl : LE.le (HSMul.hSMul (HPow.hPow I (HSub.hSub (HAdd.hAdd k n) k)) (Min.mi …
        ⊢ Membership.mem (HSMul.hSMul (HPow.hPow I (HSub.hSub (HAdd.hAdd k n) k)) (Min …
      -/
      rw [← hkn (k + n) (by omega)]
      /-
        case a
        R : Type u
        inst✝⁸ : CommRing R
        I : Ideal R
        M : Type u
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : Module R M
        N : Type u
        inst✝⁵ : AddCommGroup N
        inst✝⁴ : Module R N
        P : Type u
        inst✝³ : AddCommGroup P
        inst✝² : Module R P
        inst✝¹ : IsNoetherianRing R
        inst✝ : Module.Finite R N
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        hf : Function.Injective ⇑f
        hfg : Function.Exact ⇑f ⇑g
        hg : Function.Surjective ⇑g
        k : Nat
        hkn : ∀ (n : Nat), GE.ge n k → Eq (Min.min (HSMul.hSMul (HPow.hPow I n) Top.to …
        x : AdicCompletion.AdicCauchySequence I N
        hker : ∀ (n : Nat), Membership.mem (HSMul.hSMul (HPow.hPow I n) Top.top) (g (↑ …
        n : Nat
        d : N
        hdmem : Membership.mem (HSMul.hSMul (HPow.hPow I (HAdd.hAdd (HAdd.hAdd k n) 1) …
        y yₙ : M
        hd : Eq (f y) (HSub.hSub (↑x (HAdd.hAdd (HAdd.hAdd k n) 1)) d)
        hyₙ : Membership.mem (HSMul.hSMul (HPow.hPow I (HAdd.hAdd k n)) Top.top) (HSub …
        h : Membership.mem (HSMul.hSMul (HPow.hPow I (HAdd.hAdd k n)) Top.top) (f (HSu …
        hincl : LE.le (HSMul.hSMul (HPow.hPow I (HSub.hSub (HAdd.hAdd k n) k)) (Min.mi …
        ⊢ Membership.mem (Min.min (HSMul.hSMul (HPow.hPow I (HAdd.hAdd k n)) Top.top)  …
      -/
      exact ⟨h, ⟨y - yₙ, rfl⟩⟩
      /-
        🎉 no goals
      -/
                      /-
                        R : Type u
                        inst✝⁸ : CommRing R
                        I : Ideal R
                        M : Type u
                        inst✝⁷ : AddCommGroup M
                        inst✝⁶ : Module R M
                        N : Type u
                        inst✝⁵ : AddCommGroup N
                        inst✝⁴ : Module R N
                        P : Type u
                        inst✝³ : AddCommGroup P
                        inst✝² : Module R P
                        inst✝¹ : IsNoetherianRing R
                        inst✝ : Module.Finite R N
                        f : LinearMap (RingHom.id R) M N
                        g : LinearMap (RingHom.id R) N P
                        hf : Function.Injective ⇑f
                        hfg : Function.Exact ⇑f ⇑g
                        hg : Function.Surjective ⇑g
                        k : Nat
                        hkn : ∀ (n : Nat), GE.ge n k → Eq (Min.min (HSMul.hSMul (HPow.hPow I n) Top.to …
                        x : AdicCompletion.AdicCauchySequence I N
                        hker : ∀ (n : Nat), Membership.mem (HSMul.hSMul (HPow.hPow I n) Top.top) (g (↑ …
                        n : Nat
                        d : N
                        hdmem : Membership.mem (HSMul.hSMul (HPow.hPow I (HAdd.hAdd (HAdd.hAdd k n) 1) …
                        y yₙ : M
                        hd : Eq (f y) (HSub.hSub (↑x (HAdd.hAdd (HAdd.hAdd k n) 1)) d)
                        hyₙ : Membership.mem (HSMul.hSMul (HPow.hPow I (HAdd.hAdd k n)) Top.top) (HSub …
                        h : Membership.mem (HSMul.hSMul (HPow.hPow I (HAdd.hAdd k n)) Top.top) (f (HSu …
                        hincl : LE.le (HSMul.hSMul (HPow.hPow I (HSub.hSub (HAdd.hAdd k n) k)) (Min.mi …
                        hyyₙ : Membership.mem (HSMul.hSMul (HPow.hPow I n) Top.top) (HSub.hSub y yₙ)
                        ⊢ Membership.mem (setOf fun d => Membership.mem (HSMul.hSMul (HPow.hPow I (HAd …
                      -/
  ⟨⟨y - yₙ, hyyₙ⟩, by simpa [hd, Nat.succ_eq_add_one, Nat.add_assoc]⟩
                      /-
                        🎉 no goals
                      -/


include hfg in
/- Inductively construct preimage of cauchy sequence in kernel of `g.adicCompletion I`. -/
private noncomputable def mapExactAux :
    (n : ℕ) → { a : M | f a - x (k + n) ∈ (I ^ (k + n) • ⊤ : Submodule R N) }
  | .zero =>
      let d := (h2 0).choose
      let y := (h2 0).choose_spec.choose
      have hdy : f y = x (k + 0) - d := (h2 0).choose_spec.choose_spec.right
      have hdmem := (h2 0).choose_spec.choose_spec.left
             /-
               R : Type u
               inst✝⁸ : CommRing R
               I : Ideal R
               M : Type u
               inst✝⁷ : AddCommGroup M
               inst✝⁶ : Module R M
               N : Type u
               inst✝⁵ : AddCommGroup N
               inst✝⁴ : Module R N
               P : Type u
               inst✝³ : AddCommGroup P
               inst✝² : Module R P
               inst✝¹ : IsNoetherianRing R
               inst✝ : Module.Finite R N
               f : LinearMap (RingHom.id R) M N
               g : LinearMap (RingHom.id R) N P
               hf : Function.Injective ⇑f
               hfg : Function.Exact ⇑f ⇑g
               hg : Function.Surjective ⇑g
               k : Nat
               hkn : ∀ (n : Nat), GE.ge n k → Eq (Min.min (HSMul.hSMul (HPow.hPow I n) Top.to …
               x : AdicCompletion.AdicCauchySequence I N
               hker : ∀ (n : Nat), Membership.mem (HSMul.hSMul (HPow.hPow I n) Top.top) (g (↑ …
               x✝ : Nat
               d : N := ⋯.choose
               y : M := ⋯.choose
               hdy : Eq (f y) (HSub.hSub (↑x (HAdd.hAdd k 0)) d)
               hdmem : Membership.mem (HSMul.hSMul (HPow.hPow I (HAdd.hAdd k 0)) Top.top) ⋯.c …
               ⊢ Membership.mem (setOf fun a => Membership.mem (HSMul.hSMul (HPow.hPow I (HAd …
             -/
      ⟨y, by simpa [hdy]⟩
             /-
               🎉 no goals
             -/
  | .succ n =>
      let d := (h2 <| n + 1).choose
      let y := (h2 <| n + 1).choose_spec.choose
      have hdy : f y = x (k + (n + 1)) - d := (h2 <| n + 1).choose_spec.choose_spec.right
      have hdmem := (h2 <| n + 1).choose_spec.choose_spec.left
      let ⟨yₙ, (hyₙ : f yₙ - x (k + n) ∈ (I ^ (k + n) • ⊤ : Submodule R N))⟩ :=
        mapExactAux n
      let ⟨d, hd⟩ := mapExactAuxDelta hf hkn x hdmem hdy hyₙ
      ⟨yₙ + d, hd⟩
    /-
      R : Type u
      inst✝⁸ : CommRing R
      I : Ideal R
      M : Type u
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      N : Type u
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R N
      P : Type u
      inst✝³ : AddCommGroup P
      inst✝² : Module R P
      inst✝¹ : IsNoetherianRing R
      inst✝ : Module.Finite R N
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      hf : Function.Injective ⇑f
      hfg : Function.Exact ⇑f ⇑g
      hg : Function.Surjective ⇑g
      k : Nat
      hkn : ∀ (n : Nat), GE.ge n k → Eq (Min.min (HSMul.hSMul (HPow.hPow I n) Top.to …
      x : AdicCompletion.AdicCauchySequence I N
      hker : ∀ (n : Nat), Membership.mem (HSMul.hSMul (HPow.hPow I n) Top.top) (g (↑ …
      x✝ n : Nat
      ⊢ Membership.mem (Submodule.map g (HSMul.hSMul (HPow.hPow I (HAdd.hAdd k n)) T …
    -/
 where
    /-
      R : Type u
      inst✝⁸ : CommRing R
      I : Ideal R
      M : Type u
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      N : Type u
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R N
      P : Type u
      inst✝³ : AddCommGroup P
      inst✝² : Module R P
      inst✝¹ : IsNoetherianRing R
      inst✝ : Module.Finite R N
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      hf : Function.Injective ⇑f
      hfg : Function.Exact ⇑f ⇑g
      hg : Function.Surjective ⇑g
      k : Nat
      hkn : ∀ (n : Nat), GE.ge n k → Eq (Min.min (HSMul.hSMul (HPow.hPow I n) Top.to …
      x : AdicCompletion.AdicCauchySequence I N
      hker : ∀ (n : Nat), Membership.mem (HSMul.hSMul (HPow.hPow I n) Top.top) (g (↑ …
      x✝ n : Nat
      ⊢ Membership.mem (HSMul.hSMul (HPow.hPow I (HAdd.hAdd k n)) Top.top) (g (↑x (H …
    -/
  h1 (n : ℕ) : g (x (k + n)) ∈ Submodule.map g (I ^ (k + n) • ⊤ : Submodule R N) := by
    /-
      🎉 no goals
    -/
    rw [map_smul'', Submodule.map_top, range_eq_top.mpr hg]
    exact hker (k + n)
    /-
      R : Type u
      inst✝⁸ : CommRing R
      I : Ideal R
      M : Type u
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      N : Type u
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R N
      P : Type u
      inst✝³ : AddCommGroup P
      inst✝² : Module R P
      inst✝¹ : IsNoetherianRing R
      inst✝ : Module.Finite R N
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      hf : Function.Injective ⇑f
      hfg : Function.Exact ⇑f ⇑g
      hg : Function.Surjective ⇑g
      k : Nat
      hkn : ∀ (n : Nat), GE.ge n k → Eq (Min.min (HSMul.hSMul (HPow.hPow I n) Top.to …
      x : AdicCompletion.AdicCauchySequence I N
      hker : ∀ (n : Nat), Membership.mem (HSMul.hSMul (HPow.hPow I n) Top.top) (g (↑ …
      x✝ n : Nat
      ⊢ Exists fun d => Exists fun y => And (Membership.mem (HSMul.hSMul (HPow.hPow  …
    -/
  h2 (n : ℕ) : ∃ (d : N) (y : M),
    /-
      case intro.intro
      R : Type u
      inst✝⁸ : CommRing R
      I : Ideal R
      M : Type u
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      N : Type u
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R N
      P : Type u
      inst✝³ : AddCommGroup P
      inst✝² : Module R P
      inst✝¹ : IsNoetherianRing R
      inst✝ : Module.Finite R N
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      hf : Function.Injective ⇑f
      hfg : Function.Exact ⇑f ⇑g
      hg : Function.Surjective ⇑g
      k : Nat
      hkn : ∀ (n : Nat), GE.ge n k → Eq (Min.min (HSMul.hSMul (HPow.hPow I n) Top.to …
      x : AdicCompletion.AdicCauchySequence I N
      hker : ∀ (n : Nat), Membership.mem (HSMul.hSMul (HPow.hPow I n) Top.top) (g (↑ …
      x✝ n : Nat
      d : N
      hdmem : Membership.mem (↑(HSMul.hSMul (HPow.hPow I (HAdd.hAdd k n)) Top.top)) d
      hd : Eq (g d) (g (↑x (HAdd.hAdd k n)))
      ⊢ Exists fun d => Exists fun y => And (Membership.mem (HSMul.hSMul (HPow.hPow  …
    -/
      d ∈ (I ^ (k + n) • ⊤ : Submodule R N) ∧ f y = x (k + n) - d := by
    /-
      case intro.intro.intro
      R : Type u
      inst✝⁸ : CommRing R
      I : Ideal R
      M : Type u
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      N : Type u
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R N
      P : Type u
      inst✝³ : AddCommGroup P
      inst✝² : Module R P
      inst✝¹ : IsNoetherianRing R
      inst✝ : Module.Finite R N
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      hf : Function.Injective ⇑f
      hfg : Function.Exact ⇑f ⇑g
      hg : Function.Surjective ⇑g
      k : Nat
      hkn : ∀ (n : Nat), GE.ge n k → Eq (Min.min (HSMul.hSMul (HPow.hPow I n) Top.to …
      x : AdicCompletion.AdicCauchySequence I N
      hker : ∀ (n : Nat), Membership.mem (HSMul.hSMul (HPow.hPow I n) Top.top) (g (↑ …
      x✝ n : Nat
      d : N
      hdmem : Membership.mem (↑(HSMul.hSMul (HPow.hPow I (HAdd.hAdd k n)) Top.top)) d
      hd : Eq (g d) (g (↑x (HAdd.hAdd k n)))
      y : M
      hdy : Eq (f y) (HSub.hSub (↑x (HAdd.hAdd k n)) d)
      ⊢ Exists fun d => Exists fun y => And (Membership.mem (HSMul.hSMul (HPow.hPow  …
    -/
    obtain ⟨d, hdmem, hd⟩ := h1 n
    /-
      🎉 no goals
    -/
    obtain ⟨y, hdy⟩ := (hfg (x (k + n) - d)).mp (by simp [hd])
    exact ⟨d, y, hdmem, hdy⟩


include hf hfg hg in
/-- `AdicCompletion` over a Noetherian ring is exact on finitely generated modules. -/
theorem map_exact : Function.Exact (map I f) (map I g) := by
  /-
    R : Type u
    inst✝⁸ : CommRing R
    I : Ideal R
    M : Type u
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : Module R M
    N : Type u
    inst✝⁵ : AddCommGroup N
    inst✝⁴ : Module R N
    P : Type u
    inst✝³ : AddCommGroup P
    inst✝² : Module R P
    inst✝¹ : IsNoetherianRing R
    inst✝ : Module.Finite R N
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    hf : Function.Injective ⇑f
    hfg : Function.Exact ⇑f ⇑g
    hg : Function.Surjective ⇑g
    ⊢ Function.Exact ⇑(AdicCompletion.map I f) ⇑(AdicCompletion.map I g)
  -/
  refine LinearMap.exact_of_comp_eq_zero_of_ker_le_range ?_ (fun y ↦ ?_)
    /-
      case refine_1
      R : Type u
      inst✝⁸ : CommRing R
      I : Ideal R
      M : Type u
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      N : Type u
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R N
      P : Type u
      inst✝³ : AddCommGroup P
      inst✝² : Module R P
      inst✝¹ : IsNoetherianRing R
      inst✝ : Module.Finite R N
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      hf : Function.Injective ⇑f
      hfg : Function.Exact ⇑f ⇑g
      hg : Function.Surjective ⇑g
      ⊢ Eq ((AdicCompletion.map I g).comp (AdicCompletion.map I f)) 0
    -/
  · rw [map_comp, hfg.linearMap_comp_eq_zero, AdicCompletion.map_zero]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u
      inst✝⁸ : CommRing R
      I : Ideal R
      M : Type u
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      N : Type u
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R N
      P : Type u
      inst✝³ : AddCommGroup P
      inst✝² : Module R P
      inst✝¹ : IsNoetherianRing R
      inst✝ : Module.Finite R N
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      hf : Function.Injective ⇑f
      hfg : Function.Exact ⇑f ⇑g
      hg : Function.Surjective ⇑g
      y : AdicCompletion I N
      ⊢ Membership.mem (LinearMap.ker (AdicCompletion.map I g)) y → Membership.mem ( …
    -/
  · apply AdicCompletion.induction_on I N y (fun b ↦ ?_)
    /-
      R : Type u
      inst✝⁸ : CommRing R
      I : Ideal R
      M : Type u
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      N : Type u
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R N
      P : Type u
      inst✝³ : AddCommGroup P
      inst✝² : Module R P
      inst✝¹ : IsNoetherianRing R
      inst✝ : Module.Finite R N
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      hf : Function.Injective ⇑f
      hfg : Function.Exact ⇑f ⇑g
      hg : Function.Surjective ⇑g
      y : AdicCompletion I N
      b : AdicCompletion.AdicCauchySequence I N
      ⊢ Membership.mem (LinearMap.ker (AdicCompletion.map I g)) ((AdicCompletion.mk  …
    -/
    intro hz
    /-
      R : Type u
      inst✝⁸ : CommRing R
      I : Ideal R
      M : Type u
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      N : Type u
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R N
      P : Type u
      inst✝³ : AddCommGroup P
      inst✝² : Module R P
      inst✝¹ : IsNoetherianRing R
      inst✝ : Module.Finite R N
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      hf : Function.Injective ⇑f
      hfg : Function.Exact ⇑f ⇑g
      hg : Function.Surjective ⇑g
      y : AdicCompletion I N
      b : AdicCompletion.AdicCauchySequence I N
      hz : Membership.mem (LinearMap.ker (AdicCompletion.map I g)) ((AdicCompletion. …
      ⊢ Membership.mem (LinearMap.range (AdicCompletion.map I f)) ((AdicCompletion.m …
    -/
    obtain ⟨k, hk⟩ := Ideal.exists_pow_inf_eq_pow_smul I (LinearMap.range f)
    have hb (n : ℕ) : g (b n) ∈ (I ^ n • ⊤ : Submodule R P) := by
      simpa using congrArg (fun x ↦ x.val n) hz
    /-
      case intro
      R : Type u
      inst✝⁸ : CommRing R
      I : Ideal R
      M : Type u
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      N : Type u
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R N
      P : Type u
      inst✝³ : AddCommGroup P
      inst✝² : Module R P
      inst✝¹ : IsNoetherianRing R
      inst✝ : Module.Finite R N
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      hf : Function.Injective ⇑f
      hfg : Function.Exact ⇑f ⇑g
      hg : Function.Surjective ⇑g
      y : AdicCompletion I N
      b : AdicCompletion.AdicCauchySequence I N
      hz : Membership.mem (LinearMap.ker (AdicCompletion.map I g)) ((AdicCompletion. …
      k : Nat
      hk : ∀ (n : Nat), GE.ge n k → Eq (Min.min (HSMul.hSMul (HPow.hPow I n) Top.top …
      hb : ∀ (n : Nat), Membership.mem (HSMul.hSMul (HPow.hPow I n) Top.top) (g (↑b  …
      ⊢ Membership.mem (LinearMap.range (AdicCompletion.map I f)) ((AdicCompletion.m …
    -/
    let a := mapExactAux hf hfg hg hk b hb
    /-
      case intro
      R : Type u
      inst✝⁸ : CommRing R
      I : Ideal R
      M : Type u
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : Module R M
      N : Type u
      inst✝⁵ : AddCommGroup N
      inst✝⁴ : Module R N
      P : Type u
      inst✝³ : AddCommGroup P
      inst✝² : Module R P
      inst✝¹ : IsNoetherianRing R
      inst✝ : Module.Finite R N
      f : LinearMap (RingHom.id R) M N
      g : LinearMap (RingHom.id R) N P
      hf : Function.Injective ⇑f
      hfg : Function.Exact ⇑f ⇑g
      hg : Function.Surjective ⇑g
      y : AdicCompletion I N
      b : AdicCompletion.AdicCauchySequence I N
      hz : Membership.mem (LinearMap.ker (AdicCompletion.map I g)) ((AdicCompletion. …
      k : Nat
      hk : ∀ (n : Nat), GE.ge n k → Eq (Min.min (HSMul.hSMul (HPow.hPow I n) Top.top …
      hb : ∀ (n : Nat), Membership.mem (HSMul.hSMul (HPow.hPow I n) Top.top) (g (↑b  …
      a : (n : Nat) → ↑(setOf fun a => Membership.mem (HSMul.hSMul (HPow.hPow I (HAd …
      ⊢ Membership.mem (LinearMap.range (AdicCompletion.map I f)) ((AdicCompletion.m …
    -/
    refine ⟨AdicCompletion.mk I M (AdicCauchySequence.mk I M (fun n ↦ (a n : M)) ?_), ?_⟩
      /-
        case intro.refine_1
        R : Type u
        inst✝⁸ : CommRing R
        I : Ideal R
        M : Type u
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : Module R M
        N : Type u
        inst✝⁵ : AddCommGroup N
        inst✝⁴ : Module R N
        P : Type u
        inst✝³ : AddCommGroup P
        inst✝² : Module R P
        inst✝¹ : IsNoetherianRing R
        inst✝ : Module.Finite R N
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        hf : Function.Injective ⇑f
        hfg : Function.Exact ⇑f ⇑g
        hg : Function.Surjective ⇑g
        y : AdicCompletion I N
        b : AdicCompletion.AdicCauchySequence I N
        hz : Membership.mem (LinearMap.ker (AdicCompletion.map I g)) ((AdicCompletion. …
        k : Nat
        hk : ∀ (n : Nat), GE.ge n k → Eq (Min.min (HSMul.hSMul (HPow.hPow I n) Top.top …
        hb : ∀ (n : Nat), Membership.mem (HSMul.hSMul (HPow.hPow I n) Top.top) (g (↑b  …
        a : (n : Nat) → ↑(setOf fun a => Membership.mem (HSMul.hSMul (HPow.hPow I (HAd …
        ⊢ ∀ (n : Nat), SModEq (HSMul.hSMul (HPow.hPow I n) Top.top) ((fun n => ↑(a n)) …
      -/
    · refine fun n ↦ SModEq.symm ?_
      /-
        case intro.refine_1
        R : Type u
        inst✝⁸ : CommRing R
        I : Ideal R
        M : Type u
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : Module R M
        N : Type u
        inst✝⁵ : AddCommGroup N
        inst✝⁴ : Module R N
        P : Type u
        inst✝³ : AddCommGroup P
        inst✝² : Module R P
        inst✝¹ : IsNoetherianRing R
        inst✝ : Module.Finite R N
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        hf : Function.Injective ⇑f
        hfg : Function.Exact ⇑f ⇑g
        hg : Function.Surjective ⇑g
        y : AdicCompletion I N
        b : AdicCompletion.AdicCauchySequence I N
        hz : Membership.mem (LinearMap.ker (AdicCompletion.map I g)) ((AdicCompletion. …
        k : Nat
        hk : ∀ (n : Nat), GE.ge n k → Eq (Min.min (HSMul.hSMul (HPow.hPow I n) Top.top …
        hb : ∀ (n : Nat), Membership.mem (HSMul.hSMul (HPow.hPow I n) Top.top) (g (↑b  …
        a : (n : Nat) → ↑(setOf fun a => Membership.mem (HSMul.hSMul (HPow.hPow I (HAd …
        n : Nat
        ⊢ SModEq (HSMul.hSMul (HPow.hPow I n) Top.top) ((fun n => ↑(a n)) (HAdd.hAdd n …
      -/
      simp [a, mapExactAux, SModEq]
      /-
        🎉 no goals
      -/
      /-
        case intro.refine_2
        R : Type u
        inst✝⁸ : CommRing R
        I : Ideal R
        M : Type u
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : Module R M
        N : Type u
        inst✝⁵ : AddCommGroup N
        inst✝⁴ : Module R N
        P : Type u
        inst✝³ : AddCommGroup P
        inst✝² : Module R P
        inst✝¹ : IsNoetherianRing R
        inst✝ : Module.Finite R N
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        hf : Function.Injective ⇑f
        hfg : Function.Exact ⇑f ⇑g
        hg : Function.Surjective ⇑g
        y : AdicCompletion I N
        b : AdicCompletion.AdicCauchySequence I N
        hz : Membership.mem (LinearMap.ker (AdicCompletion.map I g)) ((AdicCompletion. …
        k : Nat
        hk : ∀ (n : Nat), GE.ge n k → Eq (Min.min (HSMul.hSMul (HPow.hPow I n) Top.top …
        hb : ∀ (n : Nat), Membership.mem (HSMul.hSMul (HPow.hPow I n) Top.top) (g (↑b  …
        a : (n : Nat) → ↑(setOf fun a => Membership.mem (HSMul.hSMul (HPow.hPow I (HAd …
        ⊢ Eq ((AdicCompletion.map I f) ((AdicCompletion.mk I M) (AdicCompletion.AdicCa …
      -/
    · ext n
      suffices h : Submodule.Quotient.mk (p := (I ^ n • ⊤ : Submodule R N)) (f (a n)) =
            Submodule.Quotient.mk (p := (I ^ n • ⊤ : Submodule R N)) (b (k + n)) by
        simp [h, AdicCauchySequence.mk_eq_mk (show n ≤ k + n by omega)]
      /-
        case intro.refine_2.h
        R : Type u
        inst✝⁸ : CommRing R
        I : Ideal R
        M : Type u
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : Module R M
        N : Type u
        inst✝⁵ : AddCommGroup N
        inst✝⁴ : Module R N
        P : Type u
        inst✝³ : AddCommGroup P
        inst✝² : Module R P
        inst✝¹ : IsNoetherianRing R
        inst✝ : Module.Finite R N
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        hf : Function.Injective ⇑f
        hfg : Function.Exact ⇑f ⇑g
        hg : Function.Surjective ⇑g
        y : AdicCompletion I N
        b : AdicCompletion.AdicCauchySequence I N
        hz : Membership.mem (LinearMap.ker (AdicCompletion.map I g)) ((AdicCompletion. …
        k : Nat
        hk : ∀ (n : Nat), GE.ge n k → Eq (Min.min (HSMul.hSMul (HPow.hPow I n) Top.top …
        hb : ∀ (n : Nat), Membership.mem (HSMul.hSMul (HPow.hPow I n) Top.top) (g (↑b  …
        a : (n : Nat) → ↑(setOf fun a => Membership.mem (HSMul.hSMul (HPow.hPow I (HAd …
        n : Nat
        ⊢ Eq (Submodule.Quotient.mk (f ↑(a n))) (Submodule.Quotient.mk (↑b (HAdd.hAdd  …
      -/
      rw [Submodule.Quotient.eq]
      have hle : (I ^ (k + n) • ⊤ : Submodule R N) ≤ (I ^ n • ⊤ : Submodule R N) :=
        Submodule.smul_mono_left (Ideal.pow_le_pow_right (by omega))
      /-
        case intro.refine_2.h
        R : Type u
        inst✝⁸ : CommRing R
        I : Ideal R
        M : Type u
        inst✝⁷ : AddCommGroup M
        inst✝⁶ : Module R M
        N : Type u
        inst✝⁵ : AddCommGroup N
        inst✝⁴ : Module R N
        P : Type u
        inst✝³ : AddCommGroup P
        inst✝² : Module R P
        inst✝¹ : IsNoetherianRing R
        inst✝ : Module.Finite R N
        f : LinearMap (RingHom.id R) M N
        g : LinearMap (RingHom.id R) N P
        hf : Function.Injective ⇑f
        hfg : Function.Exact ⇑f ⇑g
        hg : Function.Surjective ⇑g
        y : AdicCompletion I N
        b : AdicCompletion.AdicCauchySequence I N
        hz : Membership.mem (LinearMap.ker (AdicCompletion.map I g)) ((AdicCompletion. …
        k : Nat
        hk : ∀ (n : Nat), GE.ge n k → Eq (Min.min (HSMul.hSMul (HPow.hPow I n) Top.top …
        hb : ∀ (n : Nat), Membership.mem (HSMul.hSMul (HPow.hPow I n) Top.top) (g (↑b  …
        a : (n : Nat) → ↑(setOf fun a => Membership.mem (HSMul.hSMul (HPow.hPow I (HAd …
        n : Nat
        hle : LE.le (HSMul.hSMul (HPow.hPow I (HAdd.hAdd k n)) Top.top) (HSMul.hSMul ( …
        ⊢ Membership.mem (HSMul.hSMul (HPow.hPow I n) Top.top) (HSub.hSub (f ↑(a n)) ( …
      -/
      exact hle (a n).property
      /-
        🎉 no goals
      -/


