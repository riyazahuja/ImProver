/-- Let `s` be a closed set in a separable normal space. If the induced topology on `s` is discrete,
then `s` has cardinality less than continuum.

The proof follows
https://en.wikipedia.org/wiki/Moore_plane#Proof_that_the_Moore_plane_is_not_normal -/
theorem IsClosed.mk_lt_continuum [NormalSpace X] {s : Set X} (hs : IsClosed s)
    [DiscreteTopology s] : #s < 𝔠 := by
  -- Proof by contradiction: assume `𝔠 ≤ #s`
  /-
    X : Type u
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace.SeparableSpace X
    inst✝¹ : NormalSpace X
    s : Set X
    hs : IsClosed s
    inst✝ : DiscreteTopology ↑s
    ⊢ LT.lt (Cardinal.mk ↑s) Cardinal.continuum
  -/
  by_contra! h
  -- Choose a countable dense set `t : Set X`
  /-
    X : Type u
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace.SeparableSpace X
    inst✝¹ : NormalSpace X
    s : Set X
    hs : IsClosed s
    inst✝ : DiscreteTopology ↑s
    h : LE.le Cardinal.continuum (Cardinal.mk ↑s)
    ⊢ False
  -/
  rcases exists_countable_dense X with ⟨t, htc, htd⟩
  /-
    case intro.intro
    X : Type u
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace.SeparableSpace X
    inst✝¹ : NormalSpace X
    s : Set X
    hs : IsClosed s
    inst✝ : DiscreteTopology ↑s
    h : LE.le Cardinal.continuum (Cardinal.mk ↑s)
    t : Set X
    htc : t.Countable
    htd : Dense t
    ⊢ False
  -/
  haveI := htc.to_subtype
  -- To obtain a contradiction, we will prove `2 ^ 𝔠 ≤ 𝔠`.
  /-
    case intro.intro
    X : Type u
    inst✝³ : TopologicalSpace X
    inst✝² : TopologicalSpace.SeparableSpace X
    inst✝¹ : NormalSpace X
    s : Set X
    hs : IsClosed s
    inst✝ : DiscreteTopology ↑s
    h : LE.le Cardinal.continuum (Cardinal.mk ↑s)
    t : Set X
    htc : t.Countable
    htd : Dense t
    this : Countable ↑t
    ⊢ False
  -/
  refine (Cardinal.cantor 𝔠).not_le ?_
  calc
    -- Any function `s → ℝ` is continuous, hence `2 ^ 𝔠 ≤ #C(s, ℝ)`
    2 ^ 𝔠 ≤ #C(s, ℝ) := by
      rw [(ContinuousMap.equivFnOfDiscrete _ _).cardinal_eq, mk_arrow, mk_real, lift_continuum,
        lift_uzero]
      exact (power_le_power_left two_ne_zero h).trans (power_le_power_right (nat_lt_continuum 2).le)
    -- By the Tietze Extension Theorem, any function `f : C(s, ℝ)` can be extended to `C(X, ℝ)`,
    -- hence `#C(s, ℝ) ≤ #C(X, ℝ)`
    _ ≤ #C(X, ℝ) := by
      choose f hf using ContinuousMap.exists_restrict_eq (Y := ℝ) hs
      have hfi : Injective f := LeftInverse.injective hf
      exact mk_le_of_injective hfi
    -- Since `t` is dense, restriction `C(X, ℝ) → C(t, ℝ)` is injective, hence `#C(X, ℝ) ≤ #C(t, ℝ)`
    _ ≤ #C(t, ℝ) := mk_le_of_injective <| ContinuousMap.injective_restrict htd
    _ ≤ #(t → ℝ) := mk_le_of_injective DFunLike.coe_injective
    -- Since `t` is countable, we have `#(t → ℝ) ≤ 𝔠`
    _ ≤ 𝔠 := by
      rw [mk_arrow, mk_real, lift_uzero, lift_continuum, continuum, ← power_mul]
      exact power_le_power_left two_ne_zero mk_le_aleph0


/-- Let `s` be a closed set in a separable space. If the induced topology on `s` is discrete and `s`
has cardinality at least continuum, then the ambient space is not a normal space. -/
theorem IsClosed.not_normal_of_continuum_le_mk {s : Set X} (hs : IsClosed s) [DiscreteTopology s]
    (hmk : 𝔠 ≤ #s) : ¬NormalSpace X := fun _ ↦ hs.mk_lt_continuum.not_le hmk

