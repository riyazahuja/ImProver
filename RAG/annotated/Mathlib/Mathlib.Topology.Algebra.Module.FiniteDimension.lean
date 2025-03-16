/-- The space of continuous linear maps between finite-dimensional spaces is finite-dimensional. -/
instance [FiniteDimensional 𝕜 E] [FiniteDimensional 𝕜 F] : FiniteDimensional 𝕜 (E →L[𝕜] F) :=
  FiniteDimensional.of_injective (ContinuousLinearMap.coeLM 𝕜 : (E →L[𝕜] F) →ₗ[𝕜] E →ₗ[𝕜] F)
    ContinuousLinearMap.coe_injective


/-- If `𝕜` is a nontrivially normed field, any T2 topology on `𝕜` which makes it a topological
vector space over itself (with the norm topology) is *equal* to the norm topology. -/
theorem unique_topology_of_t2 {t : TopologicalSpace 𝕜} (h₁ : @TopologicalAddGroup 𝕜 t _)
    (h₂ : @ContinuousSMul 𝕜 𝕜 _ hnorm.toUniformSpace.toTopologicalSpace t) (h₃ : @T2Space 𝕜 t) :
    t = hnorm.toUniformSpace.toTopologicalSpace := by
  -- Let `𝓣₀` denote the topology on `𝕜` induced by the norm, and `𝓣` be any T2 vector
  -- topology on `𝕜`. To show that `𝓣₀ = 𝓣`, it suffices to show that they have the same
  -- neighborhoods of 0.
  /-
    𝕜 : Type u
    hnorm : NontriviallyNormedField 𝕜
    t : TopologicalSpace 𝕜
    h₁ : TopologicalAddGroup 𝕜
    h₂ : ContinuousSMul 𝕜 𝕜
    h₃ : T2Space 𝕜
    ⊢ Eq t UniformSpace.toTopologicalSpace
  -/
  refine TopologicalAddGroup.ext h₁ inferInstance (le_antisymm ?_ ?_)
  · -- To show `𝓣 ≤ 𝓣₀`, we have to show that closed balls are `𝓣`-neighborhoods of 0.
    /-
      case refine_1
      𝕜 : Type u
      hnorm : NontriviallyNormedField 𝕜
      t : TopologicalSpace 𝕜
      h₁ : TopologicalAddGroup 𝕜
      h₂ : ContinuousSMul 𝕜 𝕜
      h₃ : T2Space 𝕜
      ⊢ LE.le (nhds 0) (nhds 0)
    -/
    rw [Metric.nhds_basis_closedBall.ge_iff]
    -- Let `ε > 0`. Since `𝕜` is nontrivially normed, we have `0 < ‖ξ₀‖ < ε` for some `ξ₀ : 𝕜`.
    /-
      case refine_1
      𝕜 : Type u
      hnorm : NontriviallyNormedField 𝕜
      t : TopologicalSpace 𝕜
      h₁ : TopologicalAddGroup 𝕜
      h₂ : ContinuousSMul 𝕜 𝕜
      h₃ : T2Space 𝕜
      ⊢ ∀ (i' : Real), LT.lt 0 i' → Membership.mem (nhds 0) (Metric.closedBall 0 i')
    -/
    intro ε hε
    /-
      case refine_1
      𝕜 : Type u
      hnorm : NontriviallyNormedField 𝕜
      t : TopologicalSpace 𝕜
      h₁ : TopologicalAddGroup 𝕜
      h₂ : ContinuousSMul 𝕜 𝕜
      h₃ : T2Space 𝕜
      ε : Real
      hε : LT.lt 0 ε
      ⊢ Membership.mem (nhds 0) (Metric.closedBall 0 ε)
    -/
    rcases NormedField.exists_norm_lt 𝕜 hε with ⟨ξ₀, hξ₀, hξ₀ε⟩
    -- Since `ξ₀ ≠ 0` and `𝓣` is T2, we know that `{ξ₀}ᶜ` is a `𝓣`-neighborhood of 0.
    -- Porting note: added `mem_compl_singleton_iff.mpr`
    have : {ξ₀}ᶜ ∈ @nhds 𝕜 t 0 := IsOpen.mem_nhds isOpen_compl_singleton <|
      mem_compl_singleton_iff.mpr <| Ne.symm <| norm_ne_zero_iff.mp hξ₀.ne.symm
    -- Thus, its balanced core `𝓑` is too. Let's show that the closed ball of radius `ε` contains
    -- `𝓑`, which will imply that the closed ball is indeed a `𝓣`-neighborhood of 0.
    /-
      case refine_1.intro.intro
      𝕜 : Type u
      hnorm : NontriviallyNormedField 𝕜
      t : TopologicalSpace 𝕜
      h₁ : TopologicalAddGroup 𝕜
      h₂ : ContinuousSMul 𝕜 𝕜
      h₃ : T2Space 𝕜
      ε : Real
      hε : LT.lt 0 ε
      ξ₀ : 𝕜
      hξ₀ : LT.lt 0 (Norm.norm ξ₀)
      hξ₀ε : LT.lt (Norm.norm ξ₀) ε
      this : Membership.mem (nhds 0) (HasCompl.compl (Singleton.singleton ξ₀))
      ⊢ Membership.mem (nhds 0) (Metric.closedBall 0 ε)
    -/
    have : balancedCore 𝕜 {ξ₀}ᶜ ∈ @nhds 𝕜 t 0 := balancedCore_mem_nhds_zero this
    /-
      case refine_1.intro.intro
      𝕜 : Type u
      hnorm : NontriviallyNormedField 𝕜
      t : TopologicalSpace 𝕜
      h₁ : TopologicalAddGroup 𝕜
      h₂ : ContinuousSMul 𝕜 𝕜
      h₃ : T2Space 𝕜
      ε : Real
      hε : LT.lt 0 ε
      ξ₀ : 𝕜
      hξ₀ : LT.lt 0 (Norm.norm ξ₀)
      hξ₀ε : LT.lt (Norm.norm ξ₀) ε
      this✝ : Membership.mem (nhds 0) (HasCompl.compl (Singleton.singleton ξ₀))
      this : Membership.mem (nhds 0) (balancedCore 𝕜 (HasCompl.compl (Singleton.sing …
      ⊢ Membership.mem (nhds 0) (Metric.closedBall 0 ε)
    -/
    refine mem_of_superset this fun ξ hξ => ?_
    -- Let `ξ ∈ 𝓑`. We want to show `‖ξ‖ < ε`. If `ξ = 0`, this is trivial.
    /-
      case refine_1.intro.intro
      𝕜 : Type u
      hnorm : NontriviallyNormedField 𝕜
      t : TopologicalSpace 𝕜
      h₁ : TopologicalAddGroup 𝕜
      h₂ : ContinuousSMul 𝕜 𝕜
      h₃ : T2Space 𝕜
      ε : Real
      hε : LT.lt 0 ε
      ξ₀ : 𝕜
      hξ₀ : LT.lt 0 (Norm.norm ξ₀)
      hξ₀ε : LT.lt (Norm.norm ξ₀) ε
      this✝ : Membership.mem (nhds 0) (HasCompl.compl (Singleton.singleton ξ₀))
      this : Membership.mem (nhds 0) (balancedCore 𝕜 (HasCompl.compl (Singleton.sing …
      ξ : 𝕜
      hξ : Membership.mem (balancedCore 𝕜 (HasCompl.compl (Singleton.singleton ξ₀))) ξ
      ⊢ Membership.mem (Metric.closedBall 0 ε) ξ
    -/
    by_cases hξ0 : ξ = 0
      /-
        case pos
        𝕜 : Type u
        hnorm : NontriviallyNormedField 𝕜
        t : TopologicalSpace 𝕜
        h₁ : TopologicalAddGroup 𝕜
        h₂ : ContinuousSMul 𝕜 𝕜
        h₃ : T2Space 𝕜
        ε : Real
        hε : LT.lt 0 ε
        ξ₀ : 𝕜
        hξ₀ : LT.lt 0 (Norm.norm ξ₀)
        hξ₀ε : LT.lt (Norm.norm ξ₀) ε
        this✝ : Membership.mem (nhds 0) (HasCompl.compl (Singleton.singleton ξ₀))
        this : Membership.mem (nhds 0) (balancedCore 𝕜 (HasCompl.compl (Singleton.sing …
        ξ : 𝕜
        hξ : Membership.mem (balancedCore 𝕜 (HasCompl.compl (Singleton.singleton ξ₀))) ξ
        hξ0 : Eq ξ 0
        ⊢ Membership.mem (Metric.closedBall 0 ε) ξ
      -/
    · rw [hξ0]
      /-
        case pos
        𝕜 : Type u
        hnorm : NontriviallyNormedField 𝕜
        t : TopologicalSpace 𝕜
        h₁ : TopologicalAddGroup 𝕜
        h₂ : ContinuousSMul 𝕜 𝕜
        h₃ : T2Space 𝕜
        ε : Real
        hε : LT.lt 0 ε
        ξ₀ : 𝕜
        hξ₀ : LT.lt 0 (Norm.norm ξ₀)
        hξ₀ε : LT.lt (Norm.norm ξ₀) ε
        this✝ : Membership.mem (nhds 0) (HasCompl.compl (Singleton.singleton ξ₀))
        this : Membership.mem (nhds 0) (balancedCore 𝕜 (HasCompl.compl (Singleton.sing …
        ξ : 𝕜
        hξ : Membership.mem (balancedCore 𝕜 (HasCompl.compl (Singleton.singleton ξ₀))) ξ
        hξ0 : Eq ξ 0
        ⊢ Membership.mem (Metric.closedBall 0 ε) 0
      -/
      exact Metric.mem_closedBall_self hε.le
      /-
        🎉 no goals
      -/
      /-
        case neg
        𝕜 : Type u
        hnorm : NontriviallyNormedField 𝕜
        t : TopologicalSpace 𝕜
        h₁ : TopologicalAddGroup 𝕜
        h₂ : ContinuousSMul 𝕜 𝕜
        h₃ : T2Space 𝕜
        ε : Real
        hε : LT.lt 0 ε
        ξ₀ : 𝕜
        hξ₀ : LT.lt 0 (Norm.norm ξ₀)
        hξ₀ε : LT.lt (Norm.norm ξ₀) ε
        this✝ : Membership.mem (nhds 0) (HasCompl.compl (Singleton.singleton ξ₀))
        this : Membership.mem (nhds 0) (balancedCore 𝕜 (HasCompl.compl (Singleton.sing …
        ξ : 𝕜
        hξ : Membership.mem (balancedCore 𝕜 (HasCompl.compl (Singleton.singleton ξ₀))) ξ
        hξ0 : Not (Eq ξ 0)
        ⊢ Membership.mem (Metric.closedBall 0 ε) ξ
      -/
    · rw [mem_closedBall_zero_iff]
      -- Now suppose `ξ ≠ 0`. By contradiction, let's assume `ε < ‖ξ‖`, and show that
      -- `ξ₀ ∈ 𝓑 ⊆ {ξ₀}ᶜ`, which is a contradiction.
      /-
        case neg
        𝕜 : Type u
        hnorm : NontriviallyNormedField 𝕜
        t : TopologicalSpace 𝕜
        h₁ : TopologicalAddGroup 𝕜
        h₂ : ContinuousSMul 𝕜 𝕜
        h₃ : T2Space 𝕜
        ε : Real
        hε : LT.lt 0 ε
        ξ₀ : 𝕜
        hξ₀ : LT.lt 0 (Norm.norm ξ₀)
        hξ₀ε : LT.lt (Norm.norm ξ₀) ε
        this✝ : Membership.mem (nhds 0) (HasCompl.compl (Singleton.singleton ξ₀))
        this : Membership.mem (nhds 0) (balancedCore 𝕜 (HasCompl.compl (Singleton.sing …
        ξ : 𝕜
        hξ : Membership.mem (balancedCore 𝕜 (HasCompl.compl (Singleton.singleton ξ₀))) ξ
        hξ0 : Not (Eq ξ 0)
        ⊢ LE.le (Norm.norm ξ) ε
      -/
      by_contra! h
      suffices (ξ₀ * ξ⁻¹) • ξ ∈ balancedCore 𝕜 {ξ₀}ᶜ by
        rw [smul_eq_mul 𝕜, mul_assoc, inv_mul_cancel₀ hξ0, mul_one] at this
        exact not_mem_compl_iff.mpr (mem_singleton ξ₀) ((balancedCore_subset _) this)
      -- For that, we use that `𝓑` is balanced : since `‖ξ₀‖ < ε < ‖ξ‖`, we have `‖ξ₀ / ξ‖ ≤ 1`,
      -- hence `ξ₀ = (ξ₀ / ξ) • ξ ∈ 𝓑` because `ξ ∈ 𝓑`.
      /-
        case neg
        𝕜 : Type u
        hnorm : NontriviallyNormedField 𝕜
        t : TopologicalSpace 𝕜
        h₁ : TopologicalAddGroup 𝕜
        h₂ : ContinuousSMul 𝕜 𝕜
        h₃ : T2Space 𝕜
        ε : Real
        hε : LT.lt 0 ε
        ξ₀ : 𝕜
        hξ₀ : LT.lt 0 (Norm.norm ξ₀)
        hξ₀ε : LT.lt (Norm.norm ξ₀) ε
        this✝ : Membership.mem (nhds 0) (HasCompl.compl (Singleton.singleton ξ₀))
        this : Membership.mem (nhds 0) (balancedCore 𝕜 (HasCompl.compl (Singleton.sing …
        ξ : 𝕜
        hξ : Membership.mem (balancedCore 𝕜 (HasCompl.compl (Singleton.singleton ξ₀))) ξ
        hξ0 : Not (Eq ξ 0)
        h : LT.lt ε (Norm.norm ξ)
        ⊢ Membership.mem (balancedCore 𝕜 (HasCompl.compl (Singleton.singleton ξ₀))) (H …
      -/
      refine (balancedCore_balanced _).smul_mem ?_ hξ
      /-
        case neg
        𝕜 : Type u
        hnorm : NontriviallyNormedField 𝕜
        t : TopologicalSpace 𝕜
        h₁ : TopologicalAddGroup 𝕜
        h₂ : ContinuousSMul 𝕜 𝕜
        h₃ : T2Space 𝕜
        ε : Real
        hε : LT.lt 0 ε
        ξ₀ : 𝕜
        hξ₀ : LT.lt 0 (Norm.norm ξ₀)
        hξ₀ε : LT.lt (Norm.norm ξ₀) ε
        this✝ : Membership.mem (nhds 0) (HasCompl.compl (Singleton.singleton ξ₀))
        this : Membership.mem (nhds 0) (balancedCore 𝕜 (HasCompl.compl (Singleton.sing …
        ξ : 𝕜
        hξ : Membership.mem (balancedCore 𝕜 (HasCompl.compl (Singleton.singleton ξ₀))) ξ
        hξ0 : Not (Eq ξ 0)
        h : LT.lt ε (Norm.norm ξ)
        ⊢ LE.le (Norm.norm (HMul.hMul ξ₀ (Inv.inv ξ))) 1
      -/
      rw [norm_mul, norm_inv, mul_inv_le_iff₀ (norm_pos_iff.mpr hξ0), one_mul]
      /-
        case neg
        𝕜 : Type u
        hnorm : NontriviallyNormedField 𝕜
        t : TopologicalSpace 𝕜
        h₁ : TopologicalAddGroup 𝕜
        h₂ : ContinuousSMul 𝕜 𝕜
        h₃ : T2Space 𝕜
        ε : Real
        hε : LT.lt 0 ε
        ξ₀ : 𝕜
        hξ₀ : LT.lt 0 (Norm.norm ξ₀)
        hξ₀ε : LT.lt (Norm.norm ξ₀) ε
        this✝ : Membership.mem (nhds 0) (HasCompl.compl (Singleton.singleton ξ₀))
        this : Membership.mem (nhds 0) (balancedCore 𝕜 (HasCompl.compl (Singleton.sing …
        ξ : 𝕜
        hξ : Membership.mem (balancedCore 𝕜 (HasCompl.compl (Singleton.singleton ξ₀))) ξ
        hξ0 : Not (Eq ξ 0)
        h : LT.lt ε (Norm.norm ξ)
        ⊢ LE.le (Norm.norm ξ₀) (Norm.norm ξ)
      -/
      exact (hξ₀ε.trans h).le
      /-
        🎉 no goals
      -/
  · -- Finally, to show `𝓣₀ ≤ 𝓣`, we simply argue that `id = (fun x ↦ x • 1)` is continuous from
    -- `(𝕜, 𝓣₀)` to `(𝕜, 𝓣)` because `(•) : (𝕜, 𝓣₀) × (𝕜, 𝓣) → (𝕜, 𝓣)` is continuous.
    calc
      @nhds 𝕜 hnorm.toUniformSpace.toTopologicalSpace 0 =
          map id (@nhds 𝕜 hnorm.toUniformSpace.toTopologicalSpace 0) :=
        map_id.symm
      _ = map (fun x => id x • (1 : 𝕜)) (@nhds 𝕜 hnorm.toUniformSpace.toTopologicalSpace 0) := by
        conv_rhs =>
          congr
          ext
          rw [smul_eq_mul, mul_one]
      _ ≤ @nhds 𝕜 t ((0 : 𝕜) • (1 : 𝕜)) :=
        (@Tendsto.smul_const _ _ _ hnorm.toUniformSpace.toTopologicalSpace t _ _ _ _ _
          tendsto_id (1 : 𝕜))
      _ = @nhds 𝕜 t 0 := by rw [zero_smul]


/-- Any linear form on a topological vector space over a nontrivially normed field is continuous if
    its kernel is closed. -/
theorem LinearMap.continuous_of_isClosed_ker (l : E →ₗ[𝕜] 𝕜)
    (hl : IsClosed (LinearMap.ker l : Set E)) :
    Continuous l := by
  -- `l` is either constant or surjective. If it is constant, the result is trivial.
  /-
    𝕜 : Type u
    hnorm : NontriviallyNormedField 𝕜
    E : Type v
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul 𝕜 E
    l : LinearMap (RingHom.id 𝕜) E 𝕜
    hl : IsClosed ↑(LinearMap.ker l)
    ⊢ Continuous ⇑l
  -/
  by_cases H : finrank 𝕜 (LinearMap.range l) = 0
    /-
      case pos
      𝕜 : Type u
      hnorm : NontriviallyNormedField 𝕜
      E : Type v
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module 𝕜 E
      inst✝² : TopologicalSpace E
      inst✝¹ : TopologicalAddGroup E
      inst✝ : ContinuousSMul 𝕜 E
      l : LinearMap (RingHom.id 𝕜) E 𝕜
      hl : IsClosed ↑(LinearMap.ker l)
      H : Eq (Module.finrank 𝕜 (Subtype fun x => Membership.mem (LinearMap.range l)  …
      ⊢ Continuous ⇑l
    -/
  · rw [Submodule.finrank_eq_zero, LinearMap.range_eq_bot] at H
    /-
      case pos
      𝕜 : Type u
      hnorm : NontriviallyNormedField 𝕜
      E : Type v
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module 𝕜 E
      inst✝² : TopologicalSpace E
      inst✝¹ : TopologicalAddGroup E
      inst✝ : ContinuousSMul 𝕜 E
      l : LinearMap (RingHom.id 𝕜) E 𝕜
      hl : IsClosed ↑(LinearMap.ker l)
      H : Eq l 0
      ⊢ Continuous ⇑l
    -/
    rw [H]
    /-
      case pos
      𝕜 : Type u
      hnorm : NontriviallyNormedField 𝕜
      E : Type v
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module 𝕜 E
      inst✝² : TopologicalSpace E
      inst✝¹ : TopologicalAddGroup E
      inst✝ : ContinuousSMul 𝕜 E
      l : LinearMap (RingHom.id 𝕜) E 𝕜
      hl : IsClosed ↑(LinearMap.ker l)
      H : Eq l 0
      ⊢ Continuous ⇑0
    -/
    exact continuous_zero
    /-
      🎉 no goals
    -/
  · -- In the case where `l` is surjective, we factor it as `φ : (E ⧸ l.ker) ≃ₗ[𝕜] 𝕜`. Note that
    -- `E ⧸ l.ker` is T2 since `l.ker` is closed.
    have : finrank 𝕜 (LinearMap.range l) = 1 :=
      le_antisymm (finrank_self 𝕜 ▸ l.range.finrank_le) (zero_lt_iff.mpr H)
    have hi : Function.Injective ((LinearMap.ker l).liftQ l (le_refl _)) := by
      rw [← LinearMap.ker_eq_bot]
      exact Submodule.ker_liftQ_eq_bot _ _ _ (le_refl _)
    have hs : Function.Surjective ((LinearMap.ker l).liftQ l (le_refl _)) := by
      rw [← LinearMap.range_eq_top, Submodule.range_liftQ]
      exact Submodule.eq_top_of_finrank_eq ((finrank_self 𝕜).symm ▸ this)
    let φ : (E ⧸ LinearMap.ker l) ≃ₗ[𝕜] 𝕜 :=
      LinearEquiv.ofBijective ((LinearMap.ker l).liftQ l (le_refl _)) ⟨hi, hs⟩
    /-
      case neg
      𝕜 : Type u
      hnorm : NontriviallyNormedField 𝕜
      E : Type v
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module 𝕜 E
      inst✝² : TopologicalSpace E
      inst✝¹ : TopologicalAddGroup E
      inst✝ : ContinuousSMul 𝕜 E
      l : LinearMap (RingHom.id 𝕜) E 𝕜
      hl : IsClosed ↑(LinearMap.ker l)
      H : Not (Eq (Module.finrank 𝕜 (Subtype fun x => Membership.mem (LinearMap.rang …
      this : Eq (Module.finrank 𝕜 (Subtype fun x => Membership.mem (LinearMap.range  …
      hi : Function.Injective ⇑((LinearMap.ker l).liftQ l ⋯)
      hs : Function.Surjective ⇑((LinearMap.ker l).liftQ l ⋯)
      φ : LinearEquiv (RingHom.id 𝕜) (HasQuotient.Quotient E (LinearMap.ker l)) 𝕜 := …
      ⊢ Continuous ⇑l
    -/
    have hlφ : (l : E → 𝕜) = φ ∘ (LinearMap.ker l).mkQ := by ext; rfl
    -- Since the quotient map `E →ₗ[𝕜] (E ⧸ l.ker)` is continuous, the continuity of `l` will follow
    -- form the continuity of `φ`.
    suffices Continuous φ.toEquiv by
      rw [hlφ]
      exact this.comp continuous_quot_mk
    -- The pullback by `φ.symm` of the quotient topology is a T2 topology on `𝕜`, because `φ.symm`
    -- is injective. Since `φ.symm` is linear, it is also a vector space topology.
    -- Hence, we know that it is equal to the topology induced by the norm.
    have : induced φ.toEquiv.symm inferInstance = hnorm.toUniformSpace.toTopologicalSpace := by
      refine unique_topology_of_t2 (topologicalAddGroup_induced φ.symm.toLinearMap)
        (continuousSMul_induced φ.symm.toMulActionHom) ?_
      -- Porting note: was `rw [t2Space_iff]`
      refine (@t2Space_iff 𝕜 (induced (↑(LinearEquiv.toEquiv φ).symm) inferInstance)).mpr ?_
      exact fun x y hxy =>
        @separated_by_continuous _ _ (induced _ _) _ _ _ continuous_induced_dom _ _
          (φ.toEquiv.symm.injective.ne hxy)
    -- Finally, the pullback by `φ.symm` is exactly the pushforward by `φ`, so we have to prove
    -- that `φ` is continuous when `𝕜` is endowed with the pushforward by `φ` of the quotient
    -- topology, which is trivial by definition of the pushforward.
    /-
      case neg
      𝕜 : Type u
      hnorm : NontriviallyNormedField 𝕜
      E : Type v
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module 𝕜 E
      inst✝² : TopologicalSpace E
      inst✝¹ : TopologicalAddGroup E
      inst✝ : ContinuousSMul 𝕜 E
      l : LinearMap (RingHom.id 𝕜) E 𝕜
      hl : IsClosed ↑(LinearMap.ker l)
      H : Not (Eq (Module.finrank 𝕜 (Subtype fun x => Membership.mem (LinearMap.rang …
      this✝ : Eq (Module.finrank 𝕜 (Subtype fun x => Membership.mem (LinearMap.range …
      hi : Function.Injective ⇑((LinearMap.ker l).liftQ l ⋯)
      hs : Function.Surjective ⇑((LinearMap.ker l).liftQ l ⋯)
      φ : LinearEquiv (RingHom.id 𝕜) (HasQuotient.Quotient E (LinearMap.ker l)) 𝕜 := …
      hlφ : Eq (⇑l) (Function.comp ⇑φ ⇑(LinearMap.ker l).mkQ)
      this : Eq (TopologicalSpace.induced (⇑φ.toEquiv.symm) inferInstance) UniformSp …
      ⊢ Continuous ⇑φ.toEquiv
    -/
    rw [this.symm, Equiv.induced_symm]
    /-
      case neg
      𝕜 : Type u
      hnorm : NontriviallyNormedField 𝕜
      E : Type v
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module 𝕜 E
      inst✝² : TopologicalSpace E
      inst✝¹ : TopologicalAddGroup E
      inst✝ : ContinuousSMul 𝕜 E
      l : LinearMap (RingHom.id 𝕜) E 𝕜
      hl : IsClosed ↑(LinearMap.ker l)
      H : Not (Eq (Module.finrank 𝕜 (Subtype fun x => Membership.mem (LinearMap.rang …
      this✝ : Eq (Module.finrank 𝕜 (Subtype fun x => Membership.mem (LinearMap.range …
      hi : Function.Injective ⇑((LinearMap.ker l).liftQ l ⋯)
      hs : Function.Surjective ⇑((LinearMap.ker l).liftQ l ⋯)
      φ : LinearEquiv (RingHom.id 𝕜) (HasQuotient.Quotient E (LinearMap.ker l)) 𝕜 := …
      hlφ : Eq (⇑l) (Function.comp ⇑φ ⇑(LinearMap.ker l).mkQ)
      this : Eq (TopologicalSpace.induced (⇑φ.toEquiv.symm) inferInstance) UniformSp …
      ⊢ Continuous ⇑φ.toEquiv
    -/
    exact continuous_coinduced_rng
    /-
      🎉 no goals
    -/


/-- Any linear form on a topological vector space over a nontrivially normed field is continuous if
    and only if its kernel is closed. -/
theorem LinearMap.continuous_iff_isClosed_ker (l : E →ₗ[𝕜] 𝕜) :
    Continuous l ↔ IsClosed (LinearMap.ker l : Set E) :=
  ⟨fun h => isClosed_singleton.preimage h, l.continuous_of_isClosed_ker⟩


/-- Over a nontrivially normed field, any linear form which is nonzero on a nonempty open set is
    automatically continuous. -/
theorem LinearMap.continuous_of_nonzero_on_open (l : E →ₗ[𝕜] 𝕜) (s : Set E) (hs₁ : IsOpen s)
    (hs₂ : s.Nonempty) (hs₃ : ∀ x ∈ s, l x ≠ 0) : Continuous l := by
  /-
    𝕜 : Type u
    hnorm : NontriviallyNormedField 𝕜
    E : Type v
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul 𝕜 E
    l : LinearMap (RingHom.id 𝕜) E 𝕜
    s : Set E
    hs₁ : IsOpen s
    hs₂ : s.Nonempty
    hs₃ : ∀ (x : E), Membership.mem s x → Ne (l x) 0
    ⊢ Continuous ⇑l
  -/
  refine l.continuous_of_isClosed_ker (l.isClosed_or_dense_ker.resolve_right fun hl => ?_)
  /-
    𝕜 : Type u
    hnorm : NontriviallyNormedField 𝕜
    E : Type v
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul 𝕜 E
    l : LinearMap (RingHom.id 𝕜) E 𝕜
    s : Set E
    hs₁ : IsOpen s
    hs₂ : s.Nonempty
    hs₃ : ∀ (x : E), Membership.mem s x → Ne (l x) 0
    hl : Dense ↑(LinearMap.ker l)
    ⊢ False
  -/
  rcases hs₂ with ⟨x, hx⟩
  have : x ∈ interior (LinearMap.ker l : Set E)ᶜ := by
    rw [mem_interior_iff_mem_nhds]
    exact mem_of_superset (hs₁.mem_nhds hx) hs₃
  /-
    case intro
    𝕜 : Type u
    hnorm : NontriviallyNormedField 𝕜
    E : Type v
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul 𝕜 E
    l : LinearMap (RingHom.id 𝕜) E 𝕜
    s : Set E
    hs₁ : IsOpen s
    hs₃ : ∀ (x : E), Membership.mem s x → Ne (l x) 0
    hl : Dense ↑(LinearMap.ker l)
    x : E
    hx : Membership.mem s x
    this : Membership.mem (interior (HasCompl.compl ↑(LinearMap.ker l))) x
    ⊢ False
  -/
  rwa [hl.interior_compl] at this
  /-
    🎉 no goals
  -/


/-- This version imposes `ι` and `E` to live in the same universe, so you should instead use
`continuous_equivFun_basis` which gives the same result without universe restrictions. -/
private theorem continuous_equivFun_basis_aux [T2Space E] {ι : Type v} [Fintype ι]
    (ξ : Basis ι 𝕜 E) : Continuous ξ.equivFun := by
  /-
    𝕜 : Type u
    hnorm : NontriviallyNormedField 𝕜
    E : Type v
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : Module 𝕜 E
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : TopologicalAddGroup E
    inst✝³ : ContinuousSMul 𝕜 E
    inst✝² : CompleteSpace 𝕜
    inst✝¹ : T2Space E
    ι : Type v
    inst✝ : Fintype ι
    ξ : Basis ι 𝕜 E
    ⊢ Continuous ⇑ξ.equivFun
  -/
  letI : UniformSpace E := TopologicalAddGroup.toUniformSpace E
  /-
    𝕜 : Type u
    hnorm : NontriviallyNormedField 𝕜
    E : Type v
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : Module 𝕜 E
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : TopologicalAddGroup E
    inst✝³ : ContinuousSMul 𝕜 E
    inst✝² : CompleteSpace 𝕜
    inst✝¹ : T2Space E
    ι : Type v
    inst✝ : Fintype ι
    ξ : Basis ι 𝕜 E
    this : UniformSpace E := TopologicalAddGroup.toUniformSpace E
    ⊢ Continuous ⇑ξ.equivFun
  -/
  letI : UniformAddGroup E := comm_topologicalAddGroup_is_uniform
  /-
    𝕜 : Type u
    hnorm : NontriviallyNormedField 𝕜
    E : Type v
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : Module 𝕜 E
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : TopologicalAddGroup E
    inst✝³ : ContinuousSMul 𝕜 E
    inst✝² : CompleteSpace 𝕜
    inst✝¹ : T2Space E
    ι : Type v
    inst✝ : Fintype ι
    ξ : Basis ι 𝕜 E
    this✝ : UniformSpace E := TopologicalAddGroup.toUniformSpace E
    this : UniformAddGroup E := comm_topologicalAddGroup_is_uniform
    ⊢ Continuous ⇑ξ.equivFun
  -/
  induction' hn : Fintype.card ι with n IH generalizing ι E
    /-
      case zero
      𝕜 : Type u
      hnorm : NontriviallyNormedField 𝕜
      inst✝⁷ : CompleteSpace 𝕜
      E : Type v
      inst✝⁶ : AddCommGroup E
      inst✝⁵ : Module 𝕜 E
      inst✝⁴ : TopologicalSpace E
      inst✝³ : TopologicalAddGroup E
      inst✝² : ContinuousSMul 𝕜 E
      inst✝¹ : T2Space E
      ι : Type v
      inst✝ : Fintype ι
      ξ : Basis ι 𝕜 E
      this✝ : UniformSpace E := TopologicalAddGroup.toUniformSpace E
      this : UniformAddGroup E := comm_topologicalAddGroup_is_uniform
      hn : Eq (Fintype.card ι) 0
      ⊢ Continuous ⇑ξ.equivFun
    -/
  · rw [Fintype.card_eq_zero_iff] at hn
    /-
      case zero
      𝕜 : Type u
      hnorm : NontriviallyNormedField 𝕜
      inst✝⁷ : CompleteSpace 𝕜
      E : Type v
      inst✝⁶ : AddCommGroup E
      inst✝⁵ : Module 𝕜 E
      inst✝⁴ : TopologicalSpace E
      inst✝³ : TopologicalAddGroup E
      inst✝² : ContinuousSMul 𝕜 E
      inst✝¹ : T2Space E
      ι : Type v
      inst✝ : Fintype ι
      ξ : Basis ι 𝕜 E
      this✝ : UniformSpace E := TopologicalAddGroup.toUniformSpace E
      this : UniformAddGroup E := comm_topologicalAddGroup_is_uniform
      hn : IsEmpty ι
      ⊢ Continuous ⇑ξ.equivFun
    -/
    exact continuous_of_const fun x y => funext hn.elim
    /-
      🎉 no goals
    -/
    /-
      case succ
      𝕜 : Type u
      hnorm : NontriviallyNormedField 𝕜
      inst✝⁷ : CompleteSpace 𝕜
      n : Nat
      IH :
        ∀ {E : Type v} [inst : AddCommGroup E] [inst_1 : Module 𝕜 E] [inst_2 : Topol …
          let this := TopologicalAddGroup.toUniformSpace E;
          let this := ⋯;
          Eq (Fintype.card ι) n → Continuous ⇑ξ.equivFun
      E : Type v
      inst✝⁶ : AddCommGroup E
      inst✝⁵ : Module 𝕜 E
      inst✝⁴ : TopologicalSpace E
      inst✝³ : TopologicalAddGroup E
      inst✝² : ContinuousSMul 𝕜 E
      inst✝¹ : T2Space E
      ι : Type v
      inst✝ : Fintype ι
      ξ : Basis ι 𝕜 E
      this✝ : UniformSpace E := TopologicalAddGroup.toUniformSpace E
      this : UniformAddGroup E := comm_topologicalAddGroup_is_uniform
      hn : Eq (Fintype.card ι) (HAdd.hAdd n 1)
      ⊢ Continuous ⇑ξ.equivFun
    -/
  · haveI : FiniteDimensional 𝕜 E := .of_fintype_basis ξ
    -- first step: thanks to the induction hypothesis, any n-dimensional subspace is equivalent
    -- to a standard space of dimension n, hence it is complete and therefore closed.
    have H₁ : ∀ s : Submodule 𝕜 E, finrank 𝕜 s = n → IsClosed (s : Set E) := by
      intro s s_dim
      letI : UniformAddGroup s := s.toAddSubgroup.uniformAddGroup
      let b := Basis.ofVectorSpace 𝕜 s
      have U : IsUniformEmbedding b.equivFun.symm.toEquiv := by
        have : Fintype.card (Basis.ofVectorSpaceIndex 𝕜 s) = n := by
          rw [← s_dim]
          exact (finrank_eq_card_basis b).symm
        have : Continuous b.equivFun := IH b this
        exact
          b.equivFun.symm.isUniformEmbedding b.equivFun.symm.toLinearMap.continuous_on_pi this
      have : IsComplete (s : Set E) :=
        completeSpace_coe_iff_isComplete.1 ((completeSpace_congr U).1 inferInstance)
      exact this.isClosed
    -- second step: any linear form is continuous, as its kernel is closed by the first step
    have H₂ : ∀ f : E →ₗ[𝕜] 𝕜, Continuous f := by
      intro f
      by_cases H : finrank 𝕜 (LinearMap.range f) = 0
      · rw [Submodule.finrank_eq_zero, LinearMap.range_eq_bot] at H
        rw [H]
        exact continuous_zero
      · have : finrank 𝕜 (LinearMap.ker f) = n := by
          have Z := f.finrank_range_add_finrank_ker
          rw [finrank_eq_card_basis ξ, hn] at Z
          have : finrank 𝕜 (LinearMap.range f) = 1 :=
            le_antisymm (finrank_self 𝕜 ▸ f.range.finrank_le) (zero_lt_iff.mpr H)
          rw [this, add_comm, Nat.add_one] at Z
          exact Nat.succ.inj Z
        have : IsClosed (LinearMap.ker f : Set E) := H₁ _ this
        exact LinearMap.continuous_of_isClosed_ker f this
    /-
      case succ
      𝕜 : Type u
      hnorm : NontriviallyNormedField 𝕜
      inst✝⁷ : CompleteSpace 𝕜
      n : Nat
      IH :
        ∀ {E : Type v} [inst : AddCommGroup E] [inst_1 : Module 𝕜 E] [inst_2 : Topol …
          let this := TopologicalAddGroup.toUniformSpace E;
          let this := ⋯;
          Eq (Fintype.card ι) n → Continuous ⇑ξ.equivFun
      E : Type v
      inst✝⁶ : AddCommGroup E
      inst✝⁵ : Module 𝕜 E
      inst✝⁴ : TopologicalSpace E
      inst✝³ : TopologicalAddGroup E
      inst✝² : ContinuousSMul 𝕜 E
      inst✝¹ : T2Space E
      ι : Type v
      inst✝ : Fintype ι
      ξ : Basis ι 𝕜 E
      this✝¹ : UniformSpace E := TopologicalAddGroup.toUniformSpace E
      this✝ : UniformAddGroup E := comm_topologicalAddGroup_is_uniform
      hn : Eq (Fintype.card ι) (HAdd.hAdd n 1)
      this : FiniteDimensional 𝕜 E
      H₁ : ∀ (s : Submodule 𝕜 E), Eq (Module.finrank 𝕜 (Subtype fun x => Membership. …
      H₂ : ∀ (f : LinearMap (RingHom.id 𝕜) E 𝕜), Continuous ⇑f
      ⊢ Continuous ⇑ξ.equivFun
    -/
    rw [continuous_pi_iff]
    /-
      case succ
      𝕜 : Type u
      hnorm : NontriviallyNormedField 𝕜
      inst✝⁷ : CompleteSpace 𝕜
      n : Nat
      IH :
        ∀ {E : Type v} [inst : AddCommGroup E] [inst_1 : Module 𝕜 E] [inst_2 : Topol …
          let this := TopologicalAddGroup.toUniformSpace E;
          let this := ⋯;
          Eq (Fintype.card ι) n → Continuous ⇑ξ.equivFun
      E : Type v
      inst✝⁶ : AddCommGroup E
      inst✝⁵ : Module 𝕜 E
      inst✝⁴ : TopologicalSpace E
      inst✝³ : TopologicalAddGroup E
      inst✝² : ContinuousSMul 𝕜 E
      inst✝¹ : T2Space E
      ι : Type v
      inst✝ : Fintype ι
      ξ : Basis ι 𝕜 E
      this✝¹ : UniformSpace E := TopologicalAddGroup.toUniformSpace E
      this✝ : UniformAddGroup E := comm_topologicalAddGroup_is_uniform
      hn : Eq (Fintype.card ι) (HAdd.hAdd n 1)
      this : FiniteDimensional 𝕜 E
      H₁ : ∀ (s : Submodule 𝕜 E), Eq (Module.finrank 𝕜 (Subtype fun x => Membership. …
      H₂ : ∀ (f : LinearMap (RingHom.id 𝕜) E 𝕜), Continuous ⇑f
      ⊢ ∀ (i : ι), Continuous fun a => ξ.equivFun a i
    -/
    intro i
    /-
      case succ
      𝕜 : Type u
      hnorm : NontriviallyNormedField 𝕜
      inst✝⁷ : CompleteSpace 𝕜
      n : Nat
      IH :
        ∀ {E : Type v} [inst : AddCommGroup E] [inst_1 : Module 𝕜 E] [inst_2 : Topol …
          let this := TopologicalAddGroup.toUniformSpace E;
          let this := ⋯;
          Eq (Fintype.card ι) n → Continuous ⇑ξ.equivFun
      E : Type v
      inst✝⁶ : AddCommGroup E
      inst✝⁵ : Module 𝕜 E
      inst✝⁴ : TopologicalSpace E
      inst✝³ : TopologicalAddGroup E
      inst✝² : ContinuousSMul 𝕜 E
      inst✝¹ : T2Space E
      ι : Type v
      inst✝ : Fintype ι
      ξ : Basis ι 𝕜 E
      this✝¹ : UniformSpace E := TopologicalAddGroup.toUniformSpace E
      this✝ : UniformAddGroup E := comm_topologicalAddGroup_is_uniform
      hn : Eq (Fintype.card ι) (HAdd.hAdd n 1)
      this : FiniteDimensional 𝕜 E
      H₁ : ∀ (s : Submodule 𝕜 E), Eq (Module.finrank 𝕜 (Subtype fun x => Membership. …
      H₂ : ∀ (f : LinearMap (RingHom.id 𝕜) E 𝕜), Continuous ⇑f
      i : ι
      ⊢ Continuous fun a => ξ.equivFun a i
    -/
    change Continuous (ξ.coord i)
    /-
      case succ
      𝕜 : Type u
      hnorm : NontriviallyNormedField 𝕜
      inst✝⁷ : CompleteSpace 𝕜
      n : Nat
      IH :
        ∀ {E : Type v} [inst : AddCommGroup E] [inst_1 : Module 𝕜 E] [inst_2 : Topol …
          let this := TopologicalAddGroup.toUniformSpace E;
          let this := ⋯;
          Eq (Fintype.card ι) n → Continuous ⇑ξ.equivFun
      E : Type v
      inst✝⁶ : AddCommGroup E
      inst✝⁵ : Module 𝕜 E
      inst✝⁴ : TopologicalSpace E
      inst✝³ : TopologicalAddGroup E
      inst✝² : ContinuousSMul 𝕜 E
      inst✝¹ : T2Space E
      ι : Type v
      inst✝ : Fintype ι
      ξ : Basis ι 𝕜 E
      this✝¹ : UniformSpace E := TopologicalAddGroup.toUniformSpace E
      this✝ : UniformAddGroup E := comm_topologicalAddGroup_is_uniform
      hn : Eq (Fintype.card ι) (HAdd.hAdd n 1)
      this : FiniteDimensional 𝕜 E
      H₁ : ∀ (s : Submodule 𝕜 E), Eq (Module.finrank 𝕜 (Subtype fun x => Membership. …
      H₂ : ∀ (f : LinearMap (RingHom.id 𝕜) E 𝕜), Continuous ⇑f
      i : ι
      ⊢ Continuous ⇑(ξ.coord i)
    -/
    exact H₂ (ξ.coord i)
    /-
      🎉 no goals
    -/


/-- Any linear map on a finite dimensional space over a complete field is continuous. -/
theorem LinearMap.continuous_of_finiteDimensional [T2Space E] [FiniteDimensional 𝕜 E]
    (f : E →ₗ[𝕜] F') : Continuous f := by
  -- for the proof, go to a model vector space `b → 𝕜` thanks to `continuous_equivFun_basis`, and
  -- argue that all linear maps there are continuous.
  /-
    𝕜 : Type u
    hnorm : NontriviallyNormedField 𝕜
    E : Type v
    inst✝¹² : AddCommGroup E
    inst✝¹¹ : Module 𝕜 E
    inst✝¹⁰ : TopologicalSpace E
    inst✝⁹ : TopologicalAddGroup E
    inst✝⁸ : ContinuousSMul 𝕜 E
    F' : Type x
    inst✝⁷ : AddCommGroup F'
    inst✝⁶ : Module 𝕜 F'
    inst✝⁵ : TopologicalSpace F'
    inst✝⁴ : TopologicalAddGroup F'
    inst✝³ : ContinuousSMul 𝕜 F'
    inst✝² : CompleteSpace 𝕜
    inst✝¹ : T2Space E
    inst✝ : FiniteDimensional 𝕜 E
    f : LinearMap (RingHom.id 𝕜) E F'
    ⊢ Continuous ⇑f
  -/
  let b := Basis.ofVectorSpace 𝕜 E
  /-
    𝕜 : Type u
    hnorm : NontriviallyNormedField 𝕜
    E : Type v
    inst✝¹² : AddCommGroup E
    inst✝¹¹ : Module 𝕜 E
    inst✝¹⁰ : TopologicalSpace E
    inst✝⁹ : TopologicalAddGroup E
    inst✝⁸ : ContinuousSMul 𝕜 E
    F' : Type x
    inst✝⁷ : AddCommGroup F'
    inst✝⁶ : Module 𝕜 F'
    inst✝⁵ : TopologicalSpace F'
    inst✝⁴ : TopologicalAddGroup F'
    inst✝³ : ContinuousSMul 𝕜 F'
    inst✝² : CompleteSpace 𝕜
    inst✝¹ : T2Space E
    inst✝ : FiniteDimensional 𝕜 E
    f : LinearMap (RingHom.id 𝕜) E F'
    b : Basis (↑(Basis.ofVectorSpaceIndex 𝕜 E)) 𝕜 E := Basis.ofVectorSpace 𝕜 E
    ⊢ Continuous ⇑f
  -/
  have A : Continuous b.equivFun := continuous_equivFun_basis_aux b
  have B : Continuous (f.comp (b.equivFun.symm : (Basis.ofVectorSpaceIndex 𝕜 E → 𝕜) →ₗ[𝕜] E)) :=
    LinearMap.continuous_on_pi _
  have :
    Continuous
      (f.comp (b.equivFun.symm : (Basis.ofVectorSpaceIndex 𝕜 E → 𝕜) →ₗ[𝕜] E) ∘ b.equivFun) :=
    B.comp A
  /-
    𝕜 : Type u
    hnorm : NontriviallyNormedField 𝕜
    E : Type v
    inst✝¹² : AddCommGroup E
    inst✝¹¹ : Module 𝕜 E
    inst✝¹⁰ : TopologicalSpace E
    inst✝⁹ : TopologicalAddGroup E
    inst✝⁸ : ContinuousSMul 𝕜 E
    F' : Type x
    inst✝⁷ : AddCommGroup F'
    inst✝⁶ : Module 𝕜 F'
    inst✝⁵ : TopologicalSpace F'
    inst✝⁴ : TopologicalAddGroup F'
    inst✝³ : ContinuousSMul 𝕜 F'
    inst✝² : CompleteSpace 𝕜
    inst✝¹ : T2Space E
    inst✝ : FiniteDimensional 𝕜 E
    f : LinearMap (RingHom.id 𝕜) E F'
    b : Basis (↑(Basis.ofVectorSpaceIndex 𝕜 E)) 𝕜 E := Basis.ofVectorSpace 𝕜 E
    A : Continuous ⇑b.equivFun
    B : Continuous ⇑(f.comp ↑b.equivFun.symm)
    this : Continuous (Function.comp ⇑(f.comp ↑b.equivFun.symm) ⇑b.equivFun)
    ⊢ Continuous ⇑f
  -/
  convert this
  /-
    case h.e'_5
    𝕜 : Type u
    hnorm : NontriviallyNormedField 𝕜
    E : Type v
    inst✝¹² : AddCommGroup E
    inst✝¹¹ : Module 𝕜 E
    inst✝¹⁰ : TopologicalSpace E
    inst✝⁹ : TopologicalAddGroup E
    inst✝⁸ : ContinuousSMul 𝕜 E
    F' : Type x
    inst✝⁷ : AddCommGroup F'
    inst✝⁶ : Module 𝕜 F'
    inst✝⁵ : TopologicalSpace F'
    inst✝⁴ : TopologicalAddGroup F'
    inst✝³ : ContinuousSMul 𝕜 F'
    inst✝² : CompleteSpace 𝕜
    inst✝¹ : T2Space E
    inst✝ : FiniteDimensional 𝕜 E
    f : LinearMap (RingHom.id 𝕜) E F'
    b : Basis (↑(Basis.ofVectorSpaceIndex 𝕜 E)) 𝕜 E := Basis.ofVectorSpace 𝕜 E
    A : Continuous ⇑b.equivFun
    B : Continuous ⇑(f.comp ↑b.equivFun.symm)
    this : Continuous (Function.comp ⇑(f.comp ↑b.equivFun.symm) ⇑b.equivFun)
    ⊢ Eq (⇑f) (Function.comp ⇑(f.comp ↑b.equivFun.symm) ⇑b.equivFun)
  -/
  ext x
  /-
    case h.e'_5.h
    𝕜 : Type u
    hnorm : NontriviallyNormedField 𝕜
    E : Type v
    inst✝¹² : AddCommGroup E
    inst✝¹¹ : Module 𝕜 E
    inst✝¹⁰ : TopologicalSpace E
    inst✝⁹ : TopologicalAddGroup E
    inst✝⁸ : ContinuousSMul 𝕜 E
    F' : Type x
    inst✝⁷ : AddCommGroup F'
    inst✝⁶ : Module 𝕜 F'
    inst✝⁵ : TopologicalSpace F'
    inst✝⁴ : TopologicalAddGroup F'
    inst✝³ : ContinuousSMul 𝕜 F'
    inst✝² : CompleteSpace 𝕜
    inst✝¹ : T2Space E
    inst✝ : FiniteDimensional 𝕜 E
    f : LinearMap (RingHom.id 𝕜) E F'
    b : Basis (↑(Basis.ofVectorSpaceIndex 𝕜 E)) 𝕜 E := Basis.ofVectorSpace 𝕜 E
    A : Continuous ⇑b.equivFun
    B : Continuous ⇑(f.comp ↑b.equivFun.symm)
    this : Continuous (Function.comp ⇑(f.comp ↑b.equivFun.symm) ⇑b.equivFun)
    x : E
    ⊢ Eq (f x) (Function.comp (⇑(f.comp ↑b.equivFun.symm)) (⇑b.equivFun) x)
  -/
  dsimp
  /-
    case h.e'_5.h
    𝕜 : Type u
    hnorm : NontriviallyNormedField 𝕜
    E : Type v
    inst✝¹² : AddCommGroup E
    inst✝¹¹ : Module 𝕜 E
    inst✝¹⁰ : TopologicalSpace E
    inst✝⁹ : TopologicalAddGroup E
    inst✝⁸ : ContinuousSMul 𝕜 E
    F' : Type x
    inst✝⁷ : AddCommGroup F'
    inst✝⁶ : Module 𝕜 F'
    inst✝⁵ : TopologicalSpace F'
    inst✝⁴ : TopologicalAddGroup F'
    inst✝³ : ContinuousSMul 𝕜 F'
    inst✝² : CompleteSpace 𝕜
    inst✝¹ : T2Space E
    inst✝ : FiniteDimensional 𝕜 E
    f : LinearMap (RingHom.id 𝕜) E F'
    b : Basis (↑(Basis.ofVectorSpaceIndex 𝕜 E)) 𝕜 E := Basis.ofVectorSpace 𝕜 E
    A : Continuous ⇑b.equivFun
    B : Continuous ⇑(f.comp ↑b.equivFun.symm)
    this : Continuous (Function.comp ⇑(f.comp ↑b.equivFun.symm) ⇑b.equivFun)
    x : E
    ⊢ Eq (f x) (f (b.equivFun.symm ⇑(b.repr x)))
  -/
  rw [Basis.equivFun_symm_apply, Basis.sum_repr]
  /-
    🎉 no goals
  -/


instance LinearMap.continuousLinearMapClassOfFiniteDimensional [T2Space E] [FiniteDimensional 𝕜 E] :
    ContinuousLinearMapClass (E →ₗ[𝕜] F') 𝕜 E F' :=
  { LinearMap.semilinearMapClass with map_continuous := fun f => f.continuous_of_finiteDimensional }


/-- In finite dimensions over a non-discrete complete normed field, the canonical identification
(in terms of a basis) with `𝕜^n` (endowed with the product topology) is continuous.
This is the key fact which makes all linear maps from a T2 finite dimensional TVS over such a field
continuous (see `LinearMap.continuous_of_finiteDimensional`), which in turn implies that all
norms are equivalent in finite dimensions. -/
theorem continuous_equivFun_basis [T2Space E] {ι : Type*} [Finite ι] (ξ : Basis ι 𝕜 E) :
    Continuous ξ.equivFun :=
  haveI : FiniteDimensional 𝕜 E := .of_fintype_basis ξ
  ξ.equivFun.toLinearMap.continuous_of_finiteDimensional


/-- The continuous linear map induced by a linear map on a finite dimensional space -/
def toContinuousLinearMap : (E →ₗ[𝕜] F') ≃ₗ[𝕜] E →L[𝕜] F' where
  toFun f := ⟨f, f.continuous_of_finiteDimensional⟩
  invFun := (↑)
  map_add' _ _ := rfl
  map_smul' _ _ := rfl
  left_inv _ := rfl
  right_inv _ := ContinuousLinearMap.coe_injective rfl


/-- Algebra equivalence between the linear maps and continuous linear maps on a finite dimensional
    space. -/
def _root_.Module.End.toContinuousLinearMap (E : Type v) [NormedAddCommGroup E]
    [NormedSpace 𝕜 E] [FiniteDimensional 𝕜 E] : (E →ₗ[𝕜] E) ≃ₐ[𝕜] (E →L[𝕜] E) :=
  { LinearMap.toContinuousLinearMap with
    map_mul' := fun _ _ ↦ rfl
    commutes' := fun _ ↦ rfl }


@[simp]
theorem coe_toContinuousLinearMap' (f : E →ₗ[𝕜] F') : ⇑(LinearMap.toContinuousLinearMap f) = f :=
  rfl


@[simp]
theorem coe_toContinuousLinearMap (f : E →ₗ[𝕜] F') :
    ((LinearMap.toContinuousLinearMap f) : E →ₗ[𝕜] F') = f :=
  rfl


@[simp]
theorem coe_toContinuousLinearMap_symm :
    ⇑(toContinuousLinearMap : (E →ₗ[𝕜] F') ≃ₗ[𝕜] E →L[𝕜] F').symm =
      ((↑) : (E →L[𝕜] F') → E →ₗ[𝕜] F') :=
  rfl


@[simp]
theorem det_toContinuousLinearMap (f : E →ₗ[𝕜] E) :
    (LinearMap.toContinuousLinearMap f).det = LinearMap.det f :=
  rfl


@[simp]
theorem ker_toContinuousLinearMap (f : E →ₗ[𝕜] F') :
    ker (LinearMap.toContinuousLinearMap f) = ker f :=
  rfl


@[simp]
theorem range_toContinuousLinearMap (f : E →ₗ[𝕜] F') :
    range (LinearMap.toContinuousLinearMap f) = range f :=
  rfl


/-- A surjective linear map `f` with finite dimensional codomain is an open map. -/
theorem isOpenMap_of_finiteDimensional (f : F →ₗ[𝕜] E) (hf : Function.Surjective f) :
    IsOpenMap f := by
  /-
    𝕜 : Type u
    hnorm : NontriviallyNormedField 𝕜
    E : Type v
    inst✝¹² : AddCommGroup E
    inst✝¹¹ : Module 𝕜 E
    inst✝¹⁰ : TopologicalSpace E
    inst✝⁹ : TopologicalAddGroup E
    inst✝⁸ : ContinuousSMul 𝕜 E
    F : Type w
    inst✝⁷ : AddCommGroup F
    inst✝⁶ : Module 𝕜 F
    inst✝⁵ : TopologicalSpace F
    inst✝⁴ : TopologicalAddGroup F
    inst✝³ : ContinuousSMul 𝕜 F
    inst✝² : CompleteSpace 𝕜
    inst✝¹ : T2Space E
    inst✝ : FiniteDimensional 𝕜 E
    f : LinearMap (RingHom.id 𝕜) F E
    hf : Function.Surjective ⇑f
    ⊢ IsOpenMap ⇑f
  -/
  obtain ⟨g, hg⟩ := f.exists_rightInverse_of_surjective (LinearMap.range_eq_top.2 hf)
  /-
    case intro
    𝕜 : Type u
    hnorm : NontriviallyNormedField 𝕜
    E : Type v
    inst✝¹² : AddCommGroup E
    inst✝¹¹ : Module 𝕜 E
    inst✝¹⁰ : TopologicalSpace E
    inst✝⁹ : TopologicalAddGroup E
    inst✝⁸ : ContinuousSMul 𝕜 E
    F : Type w
    inst✝⁷ : AddCommGroup F
    inst✝⁶ : Module 𝕜 F
    inst✝⁵ : TopologicalSpace F
    inst✝⁴ : TopologicalAddGroup F
    inst✝³ : ContinuousSMul 𝕜 F
    inst✝² : CompleteSpace 𝕜
    inst✝¹ : T2Space E
    inst✝ : FiniteDimensional 𝕜 E
    f : LinearMap (RingHom.id 𝕜) F E
    hf : Function.Surjective ⇑f
    g : LinearMap (RingHom.id 𝕜) E F
    hg : Eq (f.comp g) LinearMap.id
    ⊢ IsOpenMap ⇑f
  -/
  refine IsOpenMap.of_sections fun x => ⟨fun y => g (y - f x) + x, ?_, ?_, fun y => ?_⟩
  · exact
      ((g.continuous_of_finiteDimensional.comp <| continuous_id.sub continuous_const).add
          continuous_const).continuousAt
    /-
      case intro.refine_2
      𝕜 : Type u
      hnorm : NontriviallyNormedField 𝕜
      E : Type v
      inst✝¹² : AddCommGroup E
      inst✝¹¹ : Module 𝕜 E
      inst✝¹⁰ : TopologicalSpace E
      inst✝⁹ : TopologicalAddGroup E
      inst✝⁸ : ContinuousSMul 𝕜 E
      F : Type w
      inst✝⁷ : AddCommGroup F
      inst✝⁶ : Module 𝕜 F
      inst✝⁵ : TopologicalSpace F
      inst✝⁴ : TopologicalAddGroup F
      inst✝³ : ContinuousSMul 𝕜 F
      inst✝² : CompleteSpace 𝕜
      inst✝¹ : T2Space E
      inst✝ : FiniteDimensional 𝕜 E
      f : LinearMap (RingHom.id 𝕜) F E
      hf : Function.Surjective ⇑f
      g : LinearMap (RingHom.id 𝕜) E F
      hg : Eq (f.comp g) LinearMap.id
      x : F
      ⊢ Eq ((fun y => HAdd.hAdd (g (HSub.hSub y (f x))) x) (f x)) x
    -/
  · simp only
    /-
      case intro.refine_2
      𝕜 : Type u
      hnorm : NontriviallyNormedField 𝕜
      E : Type v
      inst✝¹² : AddCommGroup E
      inst✝¹¹ : Module 𝕜 E
      inst✝¹⁰ : TopologicalSpace E
      inst✝⁹ : TopologicalAddGroup E
      inst✝⁸ : ContinuousSMul 𝕜 E
      F : Type w
      inst✝⁷ : AddCommGroup F
      inst✝⁶ : Module 𝕜 F
      inst✝⁵ : TopologicalSpace F
      inst✝⁴ : TopologicalAddGroup F
      inst✝³ : ContinuousSMul 𝕜 F
      inst✝² : CompleteSpace 𝕜
      inst✝¹ : T2Space E
      inst✝ : FiniteDimensional 𝕜 E
      f : LinearMap (RingHom.id 𝕜) F E
      hf : Function.Surjective ⇑f
      g : LinearMap (RingHom.id 𝕜) E F
      hg : Eq (f.comp g) LinearMap.id
      x : F
      ⊢ Eq (HAdd.hAdd (g (HSub.hSub (f x) (f x))) x) x
    -/
    rw [sub_self, map_zero, zero_add]
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_3
      𝕜 : Type u
      hnorm : NontriviallyNormedField 𝕜
      E : Type v
      inst✝¹² : AddCommGroup E
      inst✝¹¹ : Module 𝕜 E
      inst✝¹⁰ : TopologicalSpace E
      inst✝⁹ : TopologicalAddGroup E
      inst✝⁸ : ContinuousSMul 𝕜 E
      F : Type w
      inst✝⁷ : AddCommGroup F
      inst✝⁶ : Module 𝕜 F
      inst✝⁵ : TopologicalSpace F
      inst✝⁴ : TopologicalAddGroup F
      inst✝³ : ContinuousSMul 𝕜 F
      inst✝² : CompleteSpace 𝕜
      inst✝¹ : T2Space E
      inst✝ : FiniteDimensional 𝕜 E
      f : LinearMap (RingHom.id 𝕜) F E
      hf : Function.Surjective ⇑f
      g : LinearMap (RingHom.id 𝕜) E F
      hg : Eq (f.comp g) LinearMap.id
      x : F
      y : E
      ⊢ Eq (f ((fun y => HAdd.hAdd (g (HSub.hSub y (f x))) x) y)) y
    -/
  · simp only [map_sub, map_add, ← comp_apply f g, hg, id_apply, sub_add_cancel]
    /-
      🎉 no goals
    -/


instance canLiftContinuousLinearMap : CanLift (E →ₗ[𝕜] F) (E →L[𝕜] F) (↑) fun _ => True :=
  ⟨fun f _ => ⟨LinearMap.toContinuousLinearMap f, rfl⟩⟩


/-- The continuous linear equivalence induced by a linear equivalence on a finite dimensional
space. -/
def toContinuousLinearEquiv (e : E ≃ₗ[𝕜] F) : E ≃L[𝕜] F :=
  { e with
    continuous_toFun := e.toLinearMap.continuous_of_finiteDimensional
    continuous_invFun :=
      haveI : FiniteDimensional 𝕜 F := e.finiteDimensional
      e.symm.toLinearMap.continuous_of_finiteDimensional }


@[simp]
theorem coe_toContinuousLinearEquiv (e : E ≃ₗ[𝕜] F) : (e.toContinuousLinearEquiv : E →ₗ[𝕜] F) = e :=
  rfl


@[simp]
theorem coe_toContinuousLinearEquiv' (e : E ≃ₗ[𝕜] F) : (e.toContinuousLinearEquiv : E → F) = e :=
  rfl


@[simp]
theorem coe_toContinuousLinearEquiv_symm (e : E ≃ₗ[𝕜] F) :
    (e.toContinuousLinearEquiv.symm : F →ₗ[𝕜] E) = e.symm :=
  rfl


@[simp]
theorem coe_toContinuousLinearEquiv_symm' (e : E ≃ₗ[𝕜] F) :
    (e.toContinuousLinearEquiv.symm : F → E) = e.symm :=
  rfl


@[simp]
theorem toLinearEquiv_toContinuousLinearEquiv (e : E ≃ₗ[𝕜] F) :
    e.toContinuousLinearEquiv.toLinearEquiv = e := by
  /-
    𝕜 : Type u
    hnorm : NontriviallyNormedField 𝕜
    E : Type v
    inst✝¹³ : AddCommGroup E
    inst✝¹² : Module 𝕜 E
    inst✝¹¹ : TopologicalSpace E
    inst✝¹⁰ : TopologicalAddGroup E
    inst✝⁹ : ContinuousSMul 𝕜 E
    F : Type w
    inst✝⁸ : AddCommGroup F
    inst✝⁷ : Module 𝕜 F
    inst✝⁶ : TopologicalSpace F
    inst✝⁵ : TopologicalAddGroup F
    inst✝⁴ : ContinuousSMul 𝕜 F
    inst✝³ : CompleteSpace 𝕜
    inst✝² : T2Space E
    inst✝¹ : T2Space F
    inst✝ : FiniteDimensional 𝕜 E
    e : LinearEquiv (RingHom.id 𝕜) E F
    ⊢ Eq e.toContinuousLinearEquiv.toLinearEquiv e
  -/
  ext x
  /-
    case h
    𝕜 : Type u
    hnorm : NontriviallyNormedField 𝕜
    E : Type v
    inst✝¹³ : AddCommGroup E
    inst✝¹² : Module 𝕜 E
    inst✝¹¹ : TopologicalSpace E
    inst✝¹⁰ : TopologicalAddGroup E
    inst✝⁹ : ContinuousSMul 𝕜 E
    F : Type w
    inst✝⁸ : AddCommGroup F
    inst✝⁷ : Module 𝕜 F
    inst✝⁶ : TopologicalSpace F
    inst✝⁵ : TopologicalAddGroup F
    inst✝⁴ : ContinuousSMul 𝕜 F
    inst✝³ : CompleteSpace 𝕜
    inst✝² : T2Space E
    inst✝¹ : T2Space F
    inst✝ : FiniteDimensional 𝕜 E
    e : LinearEquiv (RingHom.id 𝕜) E F
    x : E
    ⊢ Eq (e.toContinuousLinearEquiv.toLinearEquiv x) (e x)
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem toLinearEquiv_toContinuousLinearEquiv_symm (e : E ≃ₗ[𝕜] F) :
    e.toContinuousLinearEquiv.symm.toLinearEquiv = e.symm := by
  /-
    𝕜 : Type u
    hnorm : NontriviallyNormedField 𝕜
    E : Type v
    inst✝¹³ : AddCommGroup E
    inst✝¹² : Module 𝕜 E
    inst✝¹¹ : TopologicalSpace E
    inst✝¹⁰ : TopologicalAddGroup E
    inst✝⁹ : ContinuousSMul 𝕜 E
    F : Type w
    inst✝⁸ : AddCommGroup F
    inst✝⁷ : Module 𝕜 F
    inst✝⁶ : TopologicalSpace F
    inst✝⁵ : TopologicalAddGroup F
    inst✝⁴ : ContinuousSMul 𝕜 F
    inst✝³ : CompleteSpace 𝕜
    inst✝² : T2Space E
    inst✝¹ : T2Space F
    inst✝ : FiniteDimensional 𝕜 E
    e : LinearEquiv (RingHom.id 𝕜) E F
    ⊢ Eq e.toContinuousLinearEquiv.symm.toLinearEquiv e.symm
  -/
  ext x
  /-
    case h
    𝕜 : Type u
    hnorm : NontriviallyNormedField 𝕜
    E : Type v
    inst✝¹³ : AddCommGroup E
    inst✝¹² : Module 𝕜 E
    inst✝¹¹ : TopologicalSpace E
    inst✝¹⁰ : TopologicalAddGroup E
    inst✝⁹ : ContinuousSMul 𝕜 E
    F : Type w
    inst✝⁸ : AddCommGroup F
    inst✝⁷ : Module 𝕜 F
    inst✝⁶ : TopologicalSpace F
    inst✝⁵ : TopologicalAddGroup F
    inst✝⁴ : ContinuousSMul 𝕜 F
    inst✝³ : CompleteSpace 𝕜
    inst✝² : T2Space E
    inst✝¹ : T2Space F
    inst✝ : FiniteDimensional 𝕜 E
    e : LinearEquiv (RingHom.id 𝕜) E F
    x : F
    ⊢ Eq (e.toContinuousLinearEquiv.symm.toLinearEquiv x) (e.symm x)
  -/
  rfl
  /-
    🎉 no goals
  -/


instance canLiftContinuousLinearEquiv :
    CanLift (E ≃ₗ[𝕜] F) (E ≃L[𝕜] F) ContinuousLinearEquiv.toLinearEquiv fun _ => True :=
  ⟨fun f _ => ⟨_, f.toLinearEquiv_toContinuousLinearEquiv⟩⟩


/-- Two finite-dimensional topological vector spaces over a complete normed field are continuously
linearly equivalent if they have the same (finite) dimension. -/
theorem FiniteDimensional.nonempty_continuousLinearEquiv_of_finrank_eq
    (cond : finrank 𝕜 E = finrank 𝕜 F) : Nonempty (E ≃L[𝕜] F) :=
  (nonempty_linearEquiv_of_finrank_eq cond).map LinearEquiv.toContinuousLinearEquiv


/-- Two finite-dimensional topological vector spaces over a complete normed field are continuously
linearly equivalent if and only if they have the same (finite) dimension. -/
theorem FiniteDimensional.nonempty_continuousLinearEquiv_iff_finrank_eq :
    Nonempty (E ≃L[𝕜] F) ↔ finrank 𝕜 E = finrank 𝕜 F :=
  ⟨fun ⟨h⟩ => h.toLinearEquiv.finrank_eq, fun h =>
    FiniteDimensional.nonempty_continuousLinearEquiv_of_finrank_eq h⟩


/-- A continuous linear equivalence between two finite-dimensional topological vector spaces over a
complete normed field of the same (finite) dimension. -/
def ContinuousLinearEquiv.ofFinrankEq (cond : finrank 𝕜 E = finrank 𝕜 F) : E ≃L[𝕜] F :=
  (LinearEquiv.ofFinrankEq E F cond).toContinuousLinearEquiv


/-- Construct a continuous linear map given the value at a finite basis. -/
def constrL (v : Basis ι 𝕜 E) (f : ι → F) : E →L[𝕜] F :=
  haveI : FiniteDimensional 𝕜 E := FiniteDimensional.of_fintype_basis v
  LinearMap.toContinuousLinearMap (v.constr 𝕜 f)


@[simp] -- Porting note: removed `norm_cast`
theorem coe_constrL (v : Basis ι 𝕜 E) (f : ι → F) : (v.constrL f : E →ₗ[𝕜] F) = v.constr 𝕜 f :=
  rfl


/-- The continuous linear equivalence between a vector space over `𝕜` with a finite basis and
functions from its basis indexing type to `𝕜`. -/
@[simps! apply]
def equivFunL (v : Basis ι 𝕜 E) : E ≃L[𝕜] ι → 𝕜 :=
  { v.equivFun with
    continuous_toFun :=
      haveI : FiniteDimensional 𝕜 E := FiniteDimensional.of_fintype_basis v
      v.equivFun.toLinearMap.continuous_of_finiteDimensional
    continuous_invFun := by
      /-
        𝕜 : Type u
        hnorm : NontriviallyNormedField 𝕜
        E : Type v
        inst✝¹⁷ : AddCommGroup E
        inst✝¹⁶ : Module 𝕜 E
        inst✝¹⁵ : TopologicalSpace E
        inst✝¹⁴ : TopologicalAddGroup E
        inst✝¹³ : ContinuousSMul 𝕜 E
        F : Type w
        inst✝¹² : AddCommGroup F
        inst✝¹¹ : Module 𝕜 F
        inst✝¹⁰ : TopologicalSpace F
        inst✝⁹ : TopologicalAddGroup F
        inst✝⁸ : ContinuousSMul 𝕜 F
        F' : Type x
        inst✝⁷ : AddCommGroup F'
        inst✝⁶ : Module 𝕜 F'
        inst✝⁵ : TopologicalSpace F'
        inst✝⁴ : TopologicalAddGroup F'
        inst✝³ : ContinuousSMul 𝕜 F'
        inst✝² : CompleteSpace 𝕜
        ι : Type u_1
        inst✝¹ : Finite ι
        inst✝ : T2Space E
        v : Basis ι 𝕜 E
        ⊢ Continuous __src✝.invFun
      -/
      change Continuous v.equivFun.symm.toFun
      /-
        𝕜 : Type u
        hnorm : NontriviallyNormedField 𝕜
        E : Type v
        inst✝¹⁷ : AddCommGroup E
        inst✝¹⁶ : Module 𝕜 E
        inst✝¹⁵ : TopologicalSpace E
        inst✝¹⁴ : TopologicalAddGroup E
        inst✝¹³ : ContinuousSMul 𝕜 E
        F : Type w
        inst✝¹² : AddCommGroup F
        inst✝¹¹ : Module 𝕜 F
        inst✝¹⁰ : TopologicalSpace F
        inst✝⁹ : TopologicalAddGroup F
        inst✝⁸ : ContinuousSMul 𝕜 F
        F' : Type x
        inst✝⁷ : AddCommGroup F'
        inst✝⁶ : Module 𝕜 F'
        inst✝⁵ : TopologicalSpace F'
        inst✝⁴ : TopologicalAddGroup F'
        inst✝³ : ContinuousSMul 𝕜 F'
        inst✝² : CompleteSpace 𝕜
        ι : Type u_1
        inst✝¹ : Finite ι
        inst✝ : T2Space E
        v : Basis ι 𝕜 E
        ⊢ Continuous (↑v.equivFun.symm).toFun
      -/
      exact v.equivFun.symm.toLinearMap.continuous_of_finiteDimensional }
      /-
        🎉 no goals
      -/


@[simp]
lemma equivFunL_symm_apply_repr (v : Basis ι 𝕜 E) (x : E) :
    v.equivFunL.symm (v.repr x) = x :=
  v.equivFunL.symm_apply_apply x


@[simp]
theorem constrL_apply {ι : Type*} [Fintype ι] (v : Basis ι 𝕜 E) (f : ι → F) (e : E) :
    v.constrL f e = ∑ i, v.equivFun e i • f i :=
  v.constr_apply_fintype 𝕜 _ _


@[simp 1100]
theorem constrL_basis (v : Basis ι 𝕜 E) (f : ι → F) (i : ι) : v.constrL f (v i) = f i :=
  v.constr_basis 𝕜 _ _


/-- Builds a continuous linear equivalence from a continuous linear map on a finite-dimensional
vector space whose determinant is nonzero. -/
def toContinuousLinearEquivOfDetNeZero (f : E →L[𝕜] E) (hf : f.det ≠ 0) : E ≃L[𝕜] E :=
  ((f : E →ₗ[𝕜] E).equivOfDetNeZero hf).toContinuousLinearEquiv


@[simp]
theorem coe_toContinuousLinearEquivOfDetNeZero (f : E →L[𝕜] E) (hf : f.det ≠ 0) :
    (f.toContinuousLinearEquivOfDetNeZero hf : E →L[𝕜] E) = f := by
  /-
    𝕜 : Type u
    hnorm : NontriviallyNormedField 𝕜
    E : Type v
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : Module 𝕜 E
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : TopologicalAddGroup E
    inst✝³ : ContinuousSMul 𝕜 E
    inst✝² : CompleteSpace 𝕜
    inst✝¹ : T2Space E
    inst✝ : FiniteDimensional 𝕜 E
    f : ContinuousLinearMap (RingHom.id 𝕜) E E
    hf : Ne f.det 0
    ⊢ Eq (↑(f.toContinuousLinearEquivOfDetNeZero hf)) f
  -/
  ext x
  /-
    case h
    𝕜 : Type u
    hnorm : NontriviallyNormedField 𝕜
    E : Type v
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : Module 𝕜 E
    inst✝⁵ : TopologicalSpace E
    inst✝⁴ : TopologicalAddGroup E
    inst✝³ : ContinuousSMul 𝕜 E
    inst✝² : CompleteSpace 𝕜
    inst✝¹ : T2Space E
    inst✝ : FiniteDimensional 𝕜 E
    f : ContinuousLinearMap (RingHom.id 𝕜) E E
    hf : Ne f.det 0
    x : E
    ⊢ Eq (↑(f.toContinuousLinearEquivOfDetNeZero hf) x) (f x)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem toContinuousLinearEquivOfDetNeZero_apply (f : E →L[𝕜] E) (hf : f.det ≠ 0) (x : E) :
    f.toContinuousLinearEquivOfDetNeZero hf x = f x :=
  rfl


theorem _root_.Matrix.toLin_finTwoProd_toContinuousLinearMap (a b c d : 𝕜) :
    LinearMap.toContinuousLinearMap
      (Matrix.toLin (Basis.finTwoProd 𝕜) (Basis.finTwoProd 𝕜) !![a, b; c, d]) =
      (a • ContinuousLinearMap.fst 𝕜 𝕜 𝕜 + b • ContinuousLinearMap.snd 𝕜 𝕜 𝕜).prod
        (c • ContinuousLinearMap.fst 𝕜 𝕜 𝕜 + d • ContinuousLinearMap.snd 𝕜 𝕜 𝕜) :=
  ContinuousLinearMap.ext <| Matrix.toLin_finTwoProd_apply _ _ _ _


include 𝕜 in
theorem FiniteDimensional.complete [FiniteDimensional 𝕜 E] : CompleteSpace E := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : CompleteSpace 𝕜
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : UniformSpace E
    inst✝⁴ : T2Space E
    inst✝³ : UniformAddGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : ContinuousSMul 𝕜 E
    inst✝ : FiniteDimensional 𝕜 E
    ⊢ CompleteSpace E
  -/
  set e := ContinuousLinearEquiv.ofFinrankEq (@finrank_fin_fun 𝕜 _ _ (finrank 𝕜 E)).symm
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : CompleteSpace 𝕜
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : UniformSpace E
    inst✝⁴ : T2Space E
    inst✝³ : UniformAddGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : ContinuousSMul 𝕜 E
    inst✝ : FiniteDimensional 𝕜 E
    e : ContinuousLinearEquiv (RingHom.id 𝕜) E (Fin (Module.finrank 𝕜 E) → 𝕜) := C …
    ⊢ CompleteSpace E
  -/
  have : IsUniformEmbedding e.toEquiv.symm := e.symm.isUniformEmbedding
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁸ : NontriviallyNormedField 𝕜
    inst✝⁷ : CompleteSpace 𝕜
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : UniformSpace E
    inst✝⁴ : T2Space E
    inst✝³ : UniformAddGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : ContinuousSMul 𝕜 E
    inst✝ : FiniteDimensional 𝕜 E
    e : ContinuousLinearEquiv (RingHom.id 𝕜) E (Fin (Module.finrank 𝕜 E) → 𝕜) := C …
    this : IsUniformEmbedding ⇑e.toEquiv.symm
    ⊢ CompleteSpace E
  -/
  exact (completeSpace_congr this).1 inferInstance
  /-
    🎉 no goals
  -/


/-- A finite-dimensional subspace is complete. -/
theorem Submodule.complete_of_finiteDimensional (s : Submodule 𝕜 E) [FiniteDimensional 𝕜 s] :
    IsComplete (s : Set E) :=
  haveI : UniformAddGroup s := s.toAddSubgroup.uniformAddGroup
  completeSpace_coe_iff_isComplete.1 (FiniteDimensional.complete 𝕜 s)


/-- A finite-dimensional subspace is closed. -/
theorem Submodule.closed_of_finiteDimensional
    [T2Space E] (s : Submodule 𝕜 E) [FiniteDimensional 𝕜 s] :
    IsClosed (s : Set E) :=
  letI := TopologicalAddGroup.toUniformSpace E
  haveI : UniformAddGroup E := comm_topologicalAddGroup_is_uniform
  s.complete_of_finiteDimensional.isClosed


/-- An injective linear map with finite-dimensional domain is a closed embedding. -/
theorem LinearMap.isClosedEmbedding_of_injective [T2Space E] [FiniteDimensional 𝕜 E] {f : E →ₗ[𝕜] F}
    (hf : LinearMap.ker f = ⊥) : IsClosedEmbedding f :=
  let g := LinearEquiv.ofInjective f (LinearMap.ker_eq_bot.mp hf)
  { IsEmbedding.subtypeVal.comp g.toContinuousLinearEquiv.toHomeomorph.isEmbedding with
    isClosed_range := by
      /-
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        inst✝¹⁴ : NontriviallyNormedField 𝕜
        inst✝¹³ : CompleteSpace 𝕜
        inst✝¹² : AddCommGroup E
        inst✝¹¹ : TopologicalSpace E
        inst✝¹⁰ : TopologicalAddGroup E
        inst✝⁹ : Module 𝕜 E
        inst✝⁸ : ContinuousSMul 𝕜 E
        inst✝⁷ : AddCommGroup F
        inst✝⁶ : TopologicalSpace F
        inst✝⁵ : T2Space F
        inst✝⁴ : TopologicalAddGroup F
        inst✝³ : Module 𝕜 F
        inst✝² : ContinuousSMul 𝕜 F
        inst✝¹ : T2Space E
        inst✝ : FiniteDimensional 𝕜 E
        f : LinearMap (RingHom.id 𝕜) E F
        hf : Eq (LinearMap.ker f) Bot.bot
        g : LinearEquiv (RingHom.id 𝕜) E (Subtype fun x => Membership.mem (LinearMap.r …
        ⊢ IsClosed (Set.range ⇑f)
      -/
      haveI := f.finiteDimensional_range
      /-
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        inst✝¹⁴ : NontriviallyNormedField 𝕜
        inst✝¹³ : CompleteSpace 𝕜
        inst✝¹² : AddCommGroup E
        inst✝¹¹ : TopologicalSpace E
        inst✝¹⁰ : TopologicalAddGroup E
        inst✝⁹ : Module 𝕜 E
        inst✝⁸ : ContinuousSMul 𝕜 E
        inst✝⁷ : AddCommGroup F
        inst✝⁶ : TopologicalSpace F
        inst✝⁵ : T2Space F
        inst✝⁴ : TopologicalAddGroup F
        inst✝³ : Module 𝕜 F
        inst✝² : ContinuousSMul 𝕜 F
        inst✝¹ : T2Space E
        inst✝ : FiniteDimensional 𝕜 E
        f : LinearMap (RingHom.id 𝕜) E F
        hf : Eq (LinearMap.ker f) Bot.bot
        g : LinearEquiv (RingHom.id 𝕜) E (Subtype fun x => Membership.mem (LinearMap.r …
        this : FiniteDimensional 𝕜 (Subtype fun x => Membership.mem (LinearMap.range f …
        ⊢ IsClosed (Set.range ⇑f)
      -/
      simpa [LinearMap.range_coe f] using f.range.closed_of_finiteDimensional }
      /-
        🎉 no goals
      -/


@[deprecated (since := "2024-10-20")]
alias LinearMap.closedEmbedding_of_injective := LinearMap.isClosedEmbedding_of_injective


theorem isClosedEmbedding_smul_left [T2Space E] {c : E} (hc : c ≠ 0) :
    IsClosedEmbedding fun x : 𝕜 => x • c :=
  LinearMap.isClosedEmbedding_of_injective (LinearMap.ker_toSpanSingleton 𝕜 E hc)


@[deprecated (since := "2024-10-20")]
alias closedEmbedding_smul_left := isClosedEmbedding_smul_left

-- `smul` is a closed map in the first argument.

theorem isClosedMap_smul_left [T2Space E] (c : E) : IsClosedMap fun x : 𝕜 => x • c := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : CompleteSpace 𝕜
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : TopologicalSpace E
    inst✝³ : TopologicalAddGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : ContinuousSMul 𝕜 E
    inst✝ : T2Space E
    c : E
    ⊢ IsClosedMap fun x => HSMul.hSMul x c
  -/
  by_cases hc : c = 0
    /-
      case pos
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁷ : NontriviallyNormedField 𝕜
      inst✝⁶ : CompleteSpace 𝕜
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : TopologicalSpace E
      inst✝³ : TopologicalAddGroup E
      inst✝² : Module 𝕜 E
      inst✝¹ : ContinuousSMul 𝕜 E
      inst✝ : T2Space E
      c : E
      hc : Eq c 0
      ⊢ IsClosedMap fun x => HSMul.hSMul x c
    -/
  · simp_rw [hc, smul_zero]
    /-
      case pos
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁷ : NontriviallyNormedField 𝕜
      inst✝⁶ : CompleteSpace 𝕜
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : TopologicalSpace E
      inst✝³ : TopologicalAddGroup E
      inst✝² : Module 𝕜 E
      inst✝¹ : ContinuousSMul 𝕜 E
      inst✝ : T2Space E
      c : E
      hc : Eq c 0
      ⊢ IsClosedMap fun x => 0
    -/
    exact isClosedMap_const
    /-
      🎉 no goals
    -/
    /-
      case neg
      𝕜 : Type u_1
      E : Type u_2
      inst✝⁷ : NontriviallyNormedField 𝕜
      inst✝⁶ : CompleteSpace 𝕜
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : TopologicalSpace E
      inst✝³ : TopologicalAddGroup E
      inst✝² : Module 𝕜 E
      inst✝¹ : ContinuousSMul 𝕜 E
      inst✝ : T2Space E
      c : E
      hc : Not (Eq c 0)
      ⊢ IsClosedMap fun x => HSMul.hSMul x c
    -/
  · exact (isClosedEmbedding_smul_left hc).isClosedMap
    /-
      🎉 no goals
    -/


theorem ContinuousLinearMap.exists_right_inverse_of_surjective [FiniteDimensional 𝕜 F]
    (f : E →L[𝕜] F) (hf : LinearMap.range f = ⊤) :
    ∃ g : F →L[𝕜] E, f.comp g = ContinuousLinearMap.id 𝕜 F :=
  let ⟨g, hg⟩ := (f : E →ₗ[𝕜] F).exists_rightInverse_of_surjective hf
  ⟨LinearMap.toContinuousLinearMap g, ContinuousLinearMap.coe_inj.1 hg⟩

