scoped[Isocrystal] notation "K(" p ", " k ")" => FractionRing (WittVector p k)


/-- The Frobenius automorphism of `k` induces an automorphism of `K`. -/
def FractionRing.frobenius : K(p, k) ≃+* K(p, k) :=
  IsFractionRing.ringEquivOfRingEquiv (frobeniusEquiv p k)


/-- The Frobenius automorphism of `k` induces an endomorphism of `K`. For notation purposes. -/
def FractionRing.frobeniusRingHom : K(p, k) →+* K(p, k) :=
  FractionRing.frobenius p k


scoped[Isocrystal] notation "φ(" p ", " k ")" => WittVector.FractionRing.frobeniusRingHom p k


instance inv_pair₁ : RingHomInvPair φ(p, k) (FractionRing.frobenius p k).symm :=
  RingHomInvPair.of_ringEquiv (FractionRing.frobenius p k)


instance inv_pair₂ : RingHomInvPair ((FractionRing.frobenius p k).symm : K(p, k) →+* K(p, k))
    (FractionRing.frobenius p k) :=
  RingHomInvPair.of_ringEquiv (FractionRing.frobenius p k).symm


scoped[Isocrystal]
  notation3:50 M " →ᶠˡ[" p ", " k "] " M₂ =>
    LinearMap (WittVector.FractionRing.frobeniusRingHom p k) M M₂


scoped[Isocrystal]
  notation3:50 M " ≃ᶠˡ[" p ", " k "] " M₂ =>
    LinearEquiv (WittVector.FractionRing.frobeniusRingHom p k) M M₂


/-- An isocrystal is a vector space over the field `K(p, k)` additionally equipped with a
Frobenius-linear automorphism.
-/
class Isocrystal (V : Type*) [AddCommGroup V] extends Module K(p, k) V where
  frob : V ≃ᶠˡ[p, k] V


/--
Project the Frobenius automorphism from an isocrystal. Denoted by `Φ(p, k)` when V can be inferred.
-/
def Isocrystal.frobenius : V ≃ᶠˡ[p, k] V :=
  Isocrystal.frob (p := p) (k := k) (V := V)


scoped[Isocrystal] notation "Φ(" p ", " k ")" => WittVector.Isocrystal.frobenius p k


/-- A homomorphism between isocrystals respects the Frobenius map. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): this linter isn't ported yet. @[nolint has_nonempty_instance]
structure IsocrystalHom extends V →ₗ[K(p, k)] V₂ where
  frob_equivariant : ∀ x : V, Φ(p, k) (toLinearMap x) = toLinearMap (Φ(p, k) x)


/-- An isomorphism between isocrystals respects the Frobenius map. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): this linter isn't ported yet. @[nolint has_nonempty_instance]
structure IsocrystalEquiv extends V ≃ₗ[K(p, k)] V₂ where
  frob_equivariant : ∀ x : V, Φ(p, k) (toLinearEquiv x) = toLinearEquiv (Φ(p, k) x)


scoped[Isocrystal] notation:50 M " →ᶠⁱ[" p ", " k "] " M₂ => WittVector.IsocrystalHom p k M M₂


scoped[Isocrystal] notation:50 M " ≃ᶠⁱ[" p ", " k "] " M₂ => WittVector.IsocrystalEquiv p k M M₂


/-- Type synonym for `K(p, k)` to carry the standard 1-dimensional isocrystal structure
of slope `m : ℤ`.
-/
@[nolint unusedArguments]
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): this linter isn't ported yet. @[nolint has_nonempty_instance]
def StandardOneDimIsocrystal (_m : ℤ) : Type _ :=
  K(p, k)

-- Porting note(https://github.com/leanprover-community/mathlib4/issues/5020): added

instance {m : ℤ} : AddCommGroup (StandardOneDimIsocrystal p k m) :=
  inferInstanceAs (AddCommGroup K(p, k))

instance {m : ℤ} : Module K(p, k) (StandardOneDimIsocrystal p k m) :=
  inferInstanceAs (Module K(p, k) K(p, k))


/-- The standard one-dimensional isocrystal of slope `m : ℤ` is an isocrystal. -/
instance (m : ℤ) : Isocrystal p k (StandardOneDimIsocrystal p k m) where
  frob :=
    (FractionRing.frobenius p k).toSemilinearEquiv.trans
      (LinearEquiv.smulOfNeZero _ _ _ (zpow_ne_zero m (WittVector.FractionRing.p_nonzero p k)))


@[simp]
theorem StandardOneDimIsocrystal.frobenius_apply (m : ℤ) (x : StandardOneDimIsocrystal p k m) :
    Φ(p, k) x = (p : K(p, k)) ^ m • φ(p, k) x := rfl


/-- A one-dimensional isocrystal over an algebraically closed field
admits an isomorphism to one of the standard (indexed by `m : ℤ`) one-dimensional isocrystals. -/
theorem isocrystal_classification (k : Type*) [Field k] [IsAlgClosed k] [CharP k p] (V : Type*)
    [AddCommGroup V] [Isocrystal p k V] (h_dim : finrank K(p, k) V = 1) :
    ∃ m : ℤ, Nonempty (StandardOneDimIsocrystal p k m ≃ᶠⁱ[p, k] V) := by
  /-
    p : Nat
    inst✝⁵ : Fact (Nat.Prime p)
    k : Type u_2
    inst✝⁴ : Field k
    inst✝³ : IsAlgClosed k
    inst✝² : CharP k p
    V : Type u_3
    inst✝¹ : AddCommGroup V
    inst✝ : WittVector.Isocrystal p k V
    h_dim : Eq (Module.finrank (FractionRing (WittVector p k)) V) 1
    ⊢ Exists fun m => Nonempty (WittVector.IsocrystalEquiv p k (WittVector.Standar …
  -/
  haveI : Nontrivial V := Module.nontrivial_of_finrank_eq_succ h_dim
  /-
    p : Nat
    inst✝⁵ : Fact (Nat.Prime p)
    k : Type u_2
    inst✝⁴ : Field k
    inst✝³ : IsAlgClosed k
    inst✝² : CharP k p
    V : Type u_3
    inst✝¹ : AddCommGroup V
    inst✝ : WittVector.Isocrystal p k V
    h_dim : Eq (Module.finrank (FractionRing (WittVector p k)) V) 1
    this : Nontrivial V
    ⊢ Exists fun m => Nonempty (WittVector.IsocrystalEquiv p k (WittVector.Standar …
  -/
  obtain ⟨x, hx⟩ : ∃ x : V, x ≠ 0 := exists_ne 0
  /-
    case intro
    p : Nat
    inst✝⁵ : Fact (Nat.Prime p)
    k : Type u_2
    inst✝⁴ : Field k
    inst✝³ : IsAlgClosed k
    inst✝² : CharP k p
    V : Type u_3
    inst✝¹ : AddCommGroup V
    inst✝ : WittVector.Isocrystal p k V
    h_dim : Eq (Module.finrank (FractionRing (WittVector p k)) V) 1
    this : Nontrivial V
    x : V
    hx : Ne x 0
    ⊢ Exists fun m => Nonempty (WittVector.IsocrystalEquiv p k (WittVector.Standar …
  -/
  have : Φ(p, k) x ≠ 0 := by simpa only [map_zero] using Φ(p, k).injective.ne hx
  obtain ⟨a, ha, hax⟩ : ∃ a : K(p, k), a ≠ 0 ∧ Φ(p, k) x = a • x := by
    rw [finrank_eq_one_iff_of_nonzero' x hx] at h_dim
    obtain ⟨a, ha⟩ := h_dim (Φ(p, k) x)
    refine ⟨a, ?_, ha.symm⟩
    intro ha'
    apply this
    simp only [← ha, ha', zero_smul]
  /-
    case intro.intro.intro
    p : Nat
    inst✝⁵ : Fact (Nat.Prime p)
    k : Type u_2
    inst✝⁴ : Field k
    inst✝³ : IsAlgClosed k
    inst✝² : CharP k p
    V : Type u_3
    inst✝¹ : AddCommGroup V
    inst✝ : WittVector.Isocrystal p k V
    h_dim : Eq (Module.finrank (FractionRing (WittVector p k)) V) 1
    this✝ : Nontrivial V
    x : V
    hx : Ne x 0
    this : Ne ((WittVector.Isocrystal.frobenius p k) x) 0
    a : FractionRing (WittVector p k)
    ha : Ne a 0
    hax : Eq ((WittVector.Isocrystal.frobenius p k) x) (HSMul.hSMul a x)
    ⊢ Exists fun m => Nonempty (WittVector.IsocrystalEquiv p k (WittVector.Standar …
  -/
  obtain ⟨b, hb, m, hmb⟩ := WittVector.exists_frobenius_solution_fractionRing p ha
  /-
    case intro.intro.intro.intro.intro.intro
    p : Nat
    inst✝⁵ : Fact (Nat.Prime p)
    k : Type u_2
    inst✝⁴ : Field k
    inst✝³ : IsAlgClosed k
    inst✝² : CharP k p
    V : Type u_3
    inst✝¹ : AddCommGroup V
    inst✝ : WittVector.Isocrystal p k V
    h_dim : Eq (Module.finrank (FractionRing (WittVector p k)) V) 1
    this✝ : Nontrivial V
    x : V
    hx : Ne x 0
    this : Ne ((WittVector.Isocrystal.frobenius p k) x) 0
    a : FractionRing (WittVector p k)
    ha : Ne a 0
    hax : Eq ((WittVector.Isocrystal.frobenius p k) x) (HSMul.hSMul a x)
    b : FractionRing (WittVector p k)
    hb : Ne b 0
    m : Int
    hmb : Eq (HMul.hMul ((IsFractionRing.ringEquivOfRingEquiv (WittVector.frobeniu …
    ⊢ Exists fun m => Nonempty (WittVector.IsocrystalEquiv p k (WittVector.Standar …
  -/
  replace hmb : φ(p, k) b * a = (p : K(p, k)) ^ m * b := by convert hmb
  /-
    case intro.intro.intro.intro.intro.intro
    p : Nat
    inst✝⁵ : Fact (Nat.Prime p)
    k : Type u_2
    inst✝⁴ : Field k
    inst✝³ : IsAlgClosed k
    inst✝² : CharP k p
    V : Type u_3
    inst✝¹ : AddCommGroup V
    inst✝ : WittVector.Isocrystal p k V
    h_dim : Eq (Module.finrank (FractionRing (WittVector p k)) V) 1
    this✝ : Nontrivial V
    x : V
    hx : Ne x 0
    this : Ne ((WittVector.Isocrystal.frobenius p k) x) 0
    a : FractionRing (WittVector p k)
    ha : Ne a 0
    hax : Eq ((WittVector.Isocrystal.frobenius p k) x) (HSMul.hSMul a x)
    b : FractionRing (WittVector p k)
    hb : Ne b 0
    m : Int
    hmb : Eq (HMul.hMul ((WittVector.FractionRing.frobeniusRingHom p k) b) a) (HMu …
    ⊢ Exists fun m => Nonempty (WittVector.IsocrystalEquiv p k (WittVector.Standar …
  -/
  use m
  /-
    case h
    p : Nat
    inst✝⁵ : Fact (Nat.Prime p)
    k : Type u_2
    inst✝⁴ : Field k
    inst✝³ : IsAlgClosed k
    inst✝² : CharP k p
    V : Type u_3
    inst✝¹ : AddCommGroup V
    inst✝ : WittVector.Isocrystal p k V
    h_dim : Eq (Module.finrank (FractionRing (WittVector p k)) V) 1
    this✝ : Nontrivial V
    x : V
    hx : Ne x 0
    this : Ne ((WittVector.Isocrystal.frobenius p k) x) 0
    a : FractionRing (WittVector p k)
    ha : Ne a 0
    hax : Eq ((WittVector.Isocrystal.frobenius p k) x) (HSMul.hSMul a x)
    b : FractionRing (WittVector p k)
    hb : Ne b 0
    m : Int
    hmb : Eq (HMul.hMul ((WittVector.FractionRing.frobeniusRingHom p k) b) a) (HMu …
    ⊢ Nonempty (WittVector.IsocrystalEquiv p k (WittVector.StandardOneDimIsocrysta …
  -/
  let F₀ : StandardOneDimIsocrystal p k m →ₗ[K(p, k)] V := LinearMap.toSpanSingleton K(p, k) V x
  let F : StandardOneDimIsocrystal p k m ≃ₗ[K(p, k)] V := by
    refine LinearEquiv.ofBijective F₀ ⟨?_, ?_⟩
    · rw [← LinearMap.ker_eq_bot]
      exact LinearMap.ker_toSpanSingleton K(p, k) V hx
    · rw [← LinearMap.range_eq_top]
      rw [← (finrank_eq_one_iff_of_nonzero x hx).mp h_dim]
      rw [LinearMap.span_singleton_eq_range]
  /-
    case h
    p : Nat
    inst✝⁵ : Fact (Nat.Prime p)
    k : Type u_2
    inst✝⁴ : Field k
    inst✝³ : IsAlgClosed k
    inst✝² : CharP k p
    V : Type u_3
    inst✝¹ : AddCommGroup V
    inst✝ : WittVector.Isocrystal p k V
    h_dim : Eq (Module.finrank (FractionRing (WittVector p k)) V) 1
    this✝ : Nontrivial V
    x : V
    hx : Ne x 0
    this : Ne ((WittVector.Isocrystal.frobenius p k) x) 0
    a : FractionRing (WittVector p k)
    ha : Ne a 0
    hax : Eq ((WittVector.Isocrystal.frobenius p k) x) (HSMul.hSMul a x)
    b : FractionRing (WittVector p k)
    hb : Ne b 0
    m : Int
    hmb : Eq (HMul.hMul ((WittVector.FractionRing.frobeniusRingHom p k) b) a) (HMu …
    F₀ : LinearMap (RingHom.id (FractionRing (WittVector p k))) (WittVector.Standa …
    F : LinearEquiv (RingHom.id (FractionRing (WittVector p k))) (WittVector.Stand …
    ⊢ Nonempty (WittVector.IsocrystalEquiv p k (WittVector.StandardOneDimIsocrysta …
  -/
  refine ⟨⟨(LinearEquiv.smulOfNeZero K(p, k) _ _ hb).trans F, fun c ↦ ?_⟩⟩
  rw [LinearEquiv.trans_apply, LinearEquiv.trans_apply, LinearEquiv.smulOfNeZero_apply,
    LinearEquiv.smulOfNeZero_apply, Units.smul_mk0, Units.smul_mk0, LinearEquiv.map_smul,
    LinearEquiv.map_smul]
  -- Porting note: was
  -- simp only [hax, LinearEquiv.ofBijective_apply, LinearMap.toSpanSingleton_apply,
  --   LinearEquiv.map_smulₛₗ, StandardOneDimIsocrystal.frobenius_apply, Algebra.id.smul_eq_mul]
  /-
    case h
    p : Nat
    inst✝⁵ : Fact (Nat.Prime p)
    k : Type u_2
    inst✝⁴ : Field k
    inst✝³ : IsAlgClosed k
    inst✝² : CharP k p
    V : Type u_3
    inst✝¹ : AddCommGroup V
    inst✝ : WittVector.Isocrystal p k V
    h_dim : Eq (Module.finrank (FractionRing (WittVector p k)) V) 1
    this✝ : Nontrivial V
    x : V
    hx : Ne x 0
    this : Ne ((WittVector.Isocrystal.frobenius p k) x) 0
    a : FractionRing (WittVector p k)
    ha : Ne a 0
    hax : Eq ((WittVector.Isocrystal.frobenius p k) x) (HSMul.hSMul a x)
    b : FractionRing (WittVector p k)
    hb : Ne b 0
    m : Int
    hmb : Eq (HMul.hMul ((WittVector.FractionRing.frobeniusRingHom p k) b) a) (HMu …
    F₀ : LinearMap (RingHom.id (FractionRing (WittVector p k))) (WittVector.Standa …
    F : LinearEquiv (RingHom.id (FractionRing (WittVector p k))) (WittVector.Stand …
    c : WittVector.StandardOneDimIsocrystal p k m
    ⊢ Eq ((WittVector.Isocrystal.frobenius p k) (HSMul.hSMul b (F c))) (HSMul.hSMu …
  -/
  rw [LinearEquiv.ofBijective_apply, LinearEquiv.ofBijective_apply]
  /-
    case h
    p : Nat
    inst✝⁵ : Fact (Nat.Prime p)
    k : Type u_2
    inst✝⁴ : Field k
    inst✝³ : IsAlgClosed k
    inst✝² : CharP k p
    V : Type u_3
    inst✝¹ : AddCommGroup V
    inst✝ : WittVector.Isocrystal p k V
    h_dim : Eq (Module.finrank (FractionRing (WittVector p k)) V) 1
    this✝ : Nontrivial V
    x : V
    hx : Ne x 0
    this : Ne ((WittVector.Isocrystal.frobenius p k) x) 0
    a : FractionRing (WittVector p k)
    ha : Ne a 0
    hax : Eq ((WittVector.Isocrystal.frobenius p k) x) (HSMul.hSMul a x)
    b : FractionRing (WittVector p k)
    hb : Ne b 0
    m : Int
    hmb : Eq (HMul.hMul ((WittVector.FractionRing.frobeniusRingHom p k) b) a) (HMu …
    F₀ : LinearMap (RingHom.id (FractionRing (WittVector p k))) (WittVector.Standa …
    F : LinearEquiv (RingHom.id (FractionRing (WittVector p k))) (WittVector.Stand …
    c : WittVector.StandardOneDimIsocrystal p k m
    ⊢ Eq ((WittVector.Isocrystal.frobenius p k) (HSMul.hSMul b (F₀ c))) (HSMul.hSM …
  -/
  erw [LinearMap.toSpanSingleton_apply K(p, k) V x c, LinearMap.toSpanSingleton_apply K(p, k) V x]
  simp only [hax, LinearEquiv.ofBijective_apply, LinearMap.toSpanSingleton_apply,
    LinearEquiv.map_smulₛₗ, StandardOneDimIsocrystal.frobenius_apply, Algebra.id.smul_eq_mul]
  /-
    case h
    p : Nat
    inst✝⁵ : Fact (Nat.Prime p)
    k : Type u_2
    inst✝⁴ : Field k
    inst✝³ : IsAlgClosed k
    inst✝² : CharP k p
    V : Type u_3
    inst✝¹ : AddCommGroup V
    inst✝ : WittVector.Isocrystal p k V
    h_dim : Eq (Module.finrank (FractionRing (WittVector p k)) V) 1
    this✝ : Nontrivial V
    x : V
    hx : Ne x 0
    this : Ne ((WittVector.Isocrystal.frobenius p k) x) 0
    a : FractionRing (WittVector p k)
    ha : Ne a 0
    hax : Eq ((WittVector.Isocrystal.frobenius p k) x) (HSMul.hSMul a x)
    b : FractionRing (WittVector p k)
    hb : Ne b 0
    m : Int
    hmb : Eq (HMul.hMul ((WittVector.FractionRing.frobeniusRingHom p k) b) a) (HMu …
    F₀ : LinearMap (RingHom.id (FractionRing (WittVector p k))) (WittVector.Standa …
    F : LinearEquiv (RingHom.id (FractionRing (WittVector p k))) (WittVector.Stand …
    c : WittVector.StandardOneDimIsocrystal p k m
    ⊢ Eq (HSMul.hSMul ((WittVector.FractionRing.frobeniusRingHom p k) b) (HSMul.hS …
  -/
  simp only [← mul_smul]
  /-
    case h
    p : Nat
    inst✝⁵ : Fact (Nat.Prime p)
    k : Type u_2
    inst✝⁴ : Field k
    inst✝³ : IsAlgClosed k
    inst✝² : CharP k p
    V : Type u_3
    inst✝¹ : AddCommGroup V
    inst✝ : WittVector.Isocrystal p k V
    h_dim : Eq (Module.finrank (FractionRing (WittVector p k)) V) 1
    this✝ : Nontrivial V
    x : V
    hx : Ne x 0
    this : Ne ((WittVector.Isocrystal.frobenius p k) x) 0
    a : FractionRing (WittVector p k)
    ha : Ne a 0
    hax : Eq ((WittVector.Isocrystal.frobenius p k) x) (HSMul.hSMul a x)
    b : FractionRing (WittVector p k)
    hb : Ne b 0
    m : Int
    hmb : Eq (HMul.hMul ((WittVector.FractionRing.frobeniusRingHom p k) b) a) (HMu …
    F₀ : LinearMap (RingHom.id (FractionRing (WittVector p k))) (WittVector.Standa …
    F : LinearEquiv (RingHom.id (FractionRing (WittVector p k))) (WittVector.Stand …
    c : WittVector.StandardOneDimIsocrystal p k m
    ⊢ Eq (HSMul.hSMul (HMul.hMul ((WittVector.FractionRing.frobeniusRingHom p k) b …
  -/
  congr 1
  /-
    case h.e_a
    p : Nat
    inst✝⁵ : Fact (Nat.Prime p)
    k : Type u_2
    inst✝⁴ : Field k
    inst✝³ : IsAlgClosed k
    inst✝² : CharP k p
    V : Type u_3
    inst✝¹ : AddCommGroup V
    inst✝ : WittVector.Isocrystal p k V
    h_dim : Eq (Module.finrank (FractionRing (WittVector p k)) V) 1
    this✝ : Nontrivial V
    x : V
    hx : Ne x 0
    this : Ne ((WittVector.Isocrystal.frobenius p k) x) 0
    a : FractionRing (WittVector p k)
    ha : Ne a 0
    hax : Eq ((WittVector.Isocrystal.frobenius p k) x) (HSMul.hSMul a x)
    b : FractionRing (WittVector p k)
    hb : Ne b 0
    m : Int
    hmb : Eq (HMul.hMul ((WittVector.FractionRing.frobeniusRingHom p k) b) a) (HMu …
    F₀ : LinearMap (RingHom.id (FractionRing (WittVector p k))) (WittVector.Standa …
    F : LinearEquiv (RingHom.id (FractionRing (WittVector p k))) (WittVector.Stand …
    c : WittVector.StandardOneDimIsocrystal p k m
    ⊢ Eq (HMul.hMul ((WittVector.FractionRing.frobeniusRingHom p k) b) (HMul.hMul  …
  -/
  linear_combination φ(p, k) c * hmb
  /-
    🎉 no goals
  -/


