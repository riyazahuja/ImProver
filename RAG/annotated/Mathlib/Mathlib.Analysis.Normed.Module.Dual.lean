/-- The topological dual of a seminormed space `E`. -/
abbrev Dual : Type _ := E →L[𝕜] 𝕜

-- TODO: helper instance for elaboration of inclusionInDoubleDual_norm_eq until
-- https://github.com/leanprover/lean4/issues/2522 is resolved; remove once fixed

instance : NormedSpace 𝕜 (Dual 𝕜 E) := inferInstance

-- TODO: helper instance for elaboration of inclusionInDoubleDual_norm_le until
-- https://github.com/leanprover/lean4/issues/2522 is resolved; remove once fixed

instance : SeminormedAddCommGroup (Dual 𝕜 E) := inferInstance


/-- The inclusion of a normed space in its double (topological) dual, considered
   as a bounded linear map. -/
def inclusionInDoubleDual : E →L[𝕜] Dual 𝕜 (Dual 𝕜 E) :=
  ContinuousLinearMap.apply 𝕜 𝕜


@[simp]
theorem dual_def (x : E) (f : Dual 𝕜 E) : inclusionInDoubleDual 𝕜 E x f = f x :=
  rfl


theorem inclusionInDoubleDual_norm_eq :
    ‖inclusionInDoubleDual 𝕜 E‖ = ‖ContinuousLinearMap.id 𝕜 (Dual 𝕜 E)‖ :=
  ContinuousLinearMap.opNorm_flip _


theorem inclusionInDoubleDual_norm_le : ‖inclusionInDoubleDual 𝕜 E‖ ≤ 1 := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    ⊢ LE.le (Norm.norm (NormedSpace.inclusionInDoubleDual 𝕜 E)) 1
  -/
  rw [inclusionInDoubleDual_norm_eq]
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    ⊢ LE.le (Norm.norm (ContinuousLinearMap.id 𝕜 (NormedSpace.Dual 𝕜 E))) 1
  -/
  exact ContinuousLinearMap.norm_id_le
  /-
    🎉 no goals
  -/


theorem double_dual_bound (x : E) : ‖(inclusionInDoubleDual 𝕜 E) x‖ ≤ ‖x‖ := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x : E
    ⊢ LE.le (Norm.norm ((NormedSpace.inclusionInDoubleDual 𝕜 E) x)) (Norm.norm x)
  -/
  simpa using ContinuousLinearMap.le_of_opNorm_le _ (inclusionInDoubleDual_norm_le 𝕜 E) x
  /-
    🎉 no goals
  -/


/-- The dual pairing as a bilinear form. -/
def dualPairing : Dual 𝕜 E →ₗ[𝕜] E →ₗ[𝕜] 𝕜 :=
  ContinuousLinearMap.coeLM 𝕜


@[simp]
theorem dualPairing_apply {v : Dual 𝕜 E} {x : E} : dualPairing 𝕜 E v x = v x :=
  rfl


theorem dualPairing_separatingLeft : (dualPairing 𝕜 E).SeparatingLeft := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    ⊢ (NormedSpace.dualPairing 𝕜 E).SeparatingLeft
  -/
  rw [LinearMap.separatingLeft_iff_ker_eq_bot, LinearMap.ker_eq_bot]
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    ⊢ Function.Injective ⇑(NormedSpace.dualPairing 𝕜 E)
  -/
  exact ContinuousLinearMap.coe_injective
  /-
    🎉 no goals
  -/


/-- If one controls the norm of every `f x`, then one controls the norm of `x`.
    Compare `ContinuousLinearMap.opNorm_le_bound`. -/
theorem norm_le_dual_bound (x : E) {M : ℝ} (hMp : 0 ≤ M) (hM : ∀ f : Dual 𝕜 E, ‖f x‖ ≤ M * ‖f‖) :
    ‖x‖ ≤ M := by
  classical
    by_cases h : x = 0
    · simp only [h, hMp, norm_zero]
    · obtain ⟨f, hf₁, hfx⟩ : ∃ f : E →L[𝕜] 𝕜, ‖f‖ = 1 ∧ f x = ‖x‖ := exists_dual_vector 𝕜 x h
      calc
        ‖x‖ = ‖(‖x‖ : 𝕜)‖ := RCLike.norm_coe_norm.symm
        _ = ‖f x‖ := by rw [hfx]
        _ ≤ M * ‖f‖ := hM f
        _ = M := by rw [hf₁, mul_one]


theorem eq_zero_of_forall_dual_eq_zero {x : E} (h : ∀ f : Dual 𝕜 E, f x = (0 : 𝕜)) : x = 0 :=
                                                                 /-
                                                                   𝕜 : Type v
                                                                   inst✝² : RCLike 𝕜
                                                                   E : Type u
                                                                   inst✝¹ : NormedAddCommGroup E
                                                                   inst✝ : NormedSpace 𝕜 E
                                                                   x : E
                                                                   h : ∀ (f : NormedSpace.Dual 𝕜 E), Eq (f x) 0
                                                                   f : NormedSpace.Dual 𝕜 E
                                                                   ⊢ LE.le (Norm.norm (f x)) (HMul.hMul 0 (Norm.norm f))
                                                                 -/
  norm_le_zero_iff.mp (norm_le_dual_bound 𝕜 x le_rfl fun f => by simp [h f])
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


theorem eq_zero_iff_forall_dual_eq_zero (x : E) : x = 0 ↔ ∀ g : Dual 𝕜 E, g x = 0 :=
                /-
                  𝕜 : Type v
                  inst✝² : RCLike 𝕜
                  E : Type u
                  inst✝¹ : NormedAddCommGroup E
                  inst✝ : NormedSpace 𝕜 E
                  x : E
                  hx : Eq x 0
                  ⊢ ∀ (g : NormedSpace.Dual 𝕜 E), Eq (g x) 0
                -/
  ⟨fun hx => by simp [hx], fun h => eq_zero_of_forall_dual_eq_zero 𝕜 h⟩
                /-
                  🎉 no goals
                -/


/-- See also `geometric_hahn_banach_point_point`. -/
theorem eq_iff_forall_dual_eq {x y : E} : x = y ↔ ∀ g : Dual 𝕜 E, g x = g y := by
  /-
    𝕜 : Type v
    inst✝² : RCLike 𝕜
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x y : E
    ⊢ Iff (Eq x y) (∀ (g : NormedSpace.Dual 𝕜 E), Eq (g x) (g y))
  -/
  rw [← sub_eq_zero, eq_zero_iff_forall_dual_eq_zero 𝕜 (x - y)]
  /-
    𝕜 : Type v
    inst✝² : RCLike 𝕜
    E : Type u
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    x y : E
    ⊢ Iff (∀ (g : NormedSpace.Dual 𝕜 E), Eq (g (HSub.hSub x y)) 0) (∀ (g : NormedS …
  -/
  simp [sub_eq_zero]
  /-
    🎉 no goals
  -/


/-- The inclusion of a normed space in its double dual is an isometry onto its image. -/
def inclusionInDoubleDualLi : E →ₗᵢ[𝕜] Dual 𝕜 (Dual 𝕜 E) :=
  { inclusionInDoubleDual 𝕜 E with
    norm_map' := by
      /-
        𝕜 : Type v
        inst✝² : RCLike 𝕜
        E : Type u
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace 𝕜 E
        ⊢ ∀ (x : E), Eq (Norm.norm (↑__src✝ x)) (Norm.norm x)
      -/
      intro x
      /-
        𝕜 : Type v
        inst✝² : RCLike 𝕜
        E : Type u
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace 𝕜 E
        x : E
        ⊢ Eq (Norm.norm (↑__src✝ x)) (Norm.norm x)
      -/
      apply le_antisymm
        /-
          case a
          𝕜 : Type v
          inst✝² : RCLike 𝕜
          E : Type u
          inst✝¹ : NormedAddCommGroup E
          inst✝ : NormedSpace 𝕜 E
          x : E
          ⊢ LE.le (Norm.norm (↑__src✝ x)) (Norm.norm x)
        -/
      · exact double_dual_bound 𝕜 E x
        /-
          🎉 no goals
        -/
      /-
        case a
        𝕜 : Type v
        inst✝² : RCLike 𝕜
        E : Type u
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace 𝕜 E
        x : E
        ⊢ LE.le (Norm.norm x) (Norm.norm (↑__src✝ x))
      -/
      rw [ContinuousLinearMap.norm_def]
      /-
        case a
        𝕜 : Type v
        inst✝² : RCLike 𝕜
        E : Type u
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace 𝕜 E
        x : E
        ⊢ LE.le (Norm.norm x) (InfSet.sInf (setOf fun c => And (LE.le 0 c) (∀ (x_1 : N …
      -/
      refine le_csInf ContinuousLinearMap.bounds_nonempty ?_
      /-
        case a
        𝕜 : Type v
        inst✝² : RCLike 𝕜
        E : Type u
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace 𝕜 E
        x : E
        ⊢ ∀ (b : Real), Membership.mem (setOf fun c => And (LE.le 0 c) (∀ (x_1 : Norme …
      -/
      rintro c ⟨hc1, hc2⟩
      /-
        case a.intro
        𝕜 : Type v
        inst✝² : RCLike 𝕜
        E : Type u
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace 𝕜 E
        x : E
        c : Real
        hc1 : LE.le 0 c
        hc2 : ∀ (x_1 : NormedSpace.Dual 𝕜 E), LE.le (Norm.norm ((↑__src✝ x) x_1)) (HMu …
        ⊢ LE.le (Norm.norm x) c
      -/
      exact norm_le_dual_bound 𝕜 x hc1 hc2 }
      /-
        🎉 no goals
      -/


/-- Given a subset `s` in a normed space `E` (over a field `𝕜`), the polar
`polar 𝕜 s` is the subset of `Dual 𝕜 E` consisting of those functionals which
evaluate to something of norm at most one at all points `z ∈ s`. -/
def polar (𝕜 : Type*) [NontriviallyNormedField 𝕜] {E : Type*} [SeminormedAddCommGroup E]
    [NormedSpace 𝕜 E] : Set E → Set (Dual 𝕜 E) :=
  (dualPairing 𝕜 E).flip.polar


theorem mem_polar_iff {x' : Dual 𝕜 E} (s : Set E) : x' ∈ polar 𝕜 s ↔ ∀ z ∈ s, ‖x' z‖ ≤ 1 :=
  Iff.rfl


@[simp]
theorem zero_mem_polar (s : Set E) : (0 : Dual 𝕜 E) ∈ polar 𝕜 s :=
  LinearMap.zero_mem_polar _ s


theorem polar_nonempty (s : Set E) : Set.Nonempty (polar 𝕜 s) :=
  LinearMap.polar_nonempty _ _


@[simp]
theorem polar_univ : polar 𝕜 (univ : Set E) = {(0 : Dual 𝕜 E)} :=
  (dualPairing 𝕜 E).flip.polar_univ
    (LinearMap.flip_separatingRight.mpr (dualPairing_separatingLeft 𝕜 E))


theorem isClosed_polar (s : Set E) : IsClosed (polar 𝕜 s) := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    s : Set E
    ⊢ IsClosed (NormedSpace.polar 𝕜 s)
  -/
  dsimp only [NormedSpace.polar]
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    s : Set E
    ⊢ IsClosed ((NormedSpace.dualPairing 𝕜 E).flip.polar s)
  -/
  simp only [LinearMap.polar_eq_iInter, LinearMap.flip_apply]
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    s : Set E
    ⊢ IsClosed (Set.iInter fun x => Set.iInter fun x_1 => setOf fun y => LE.le (No …
  -/
  refine isClosed_biInter fun z _ => ?_
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    s : Set E
    z : E
    x✝ : Membership.mem s z
    ⊢ IsClosed (setOf fun y => LE.le (Norm.norm (((NormedSpace.dualPairing 𝕜 E) y) …
  -/
  exact isClosed_Iic.preimage (ContinuousLinearMap.apply 𝕜 𝕜 z).continuous.norm
  /-
    🎉 no goals
  -/


@[simp]
theorem polar_closure (s : Set E) : polar 𝕜 (closure s) = polar 𝕜 s :=
  ((dualPairing 𝕜 E).flip.polar_antitone subset_closure).antisymm <|
    (dualPairing 𝕜 E).flip.polar_gc.l_le <|
      closure_minimal ((dualPairing 𝕜 E).flip.polar_gc.le_u_l s) <| by
        simpa [LinearMap.flip_flip] using
          (isClosed_polar _ _).preimage (inclusionInDoubleDual 𝕜 E).continuous


/-- If `x'` is a dual element such that the norms `‖x' z‖` are bounded for `z ∈ s`, then a
small scalar multiple of `x'` is in `polar 𝕜 s`. -/
theorem smul_mem_polar {s : Set E} {x' : Dual 𝕜 E} {c : 𝕜} (hc : ∀ z, z ∈ s → ‖x' z‖ ≤ ‖c‖) :
    c⁻¹ • x' ∈ polar 𝕜 s := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    s : Set E
    x' : NormedSpace.Dual 𝕜 E
    c : 𝕜
    hc : ∀ (z : E), Membership.mem s z → LE.le (Norm.norm (x' z)) (Norm.norm c)
    ⊢ Membership.mem (NormedSpace.polar 𝕜 s) (HSMul.hSMul (Inv.inv c) x')
  -/
  by_cases c_zero : c = 0
    /-
      case pos
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      s : Set E
      x' : NormedSpace.Dual 𝕜 E
      c : 𝕜
      hc : ∀ (z : E), Membership.mem s z → LE.le (Norm.norm (x' z)) (Norm.norm c)
      c_zero : Eq c 0
      ⊢ Membership.mem (NormedSpace.polar 𝕜 s) (HSMul.hSMul (Inv.inv c) x')
    -/
  · simp only [c_zero, inv_zero, zero_smul]
    /-
      case pos
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      s : Set E
      x' : NormedSpace.Dual 𝕜 E
      c : 𝕜
      hc : ∀ (z : E), Membership.mem s z → LE.le (Norm.norm (x' z)) (Norm.norm c)
      c_zero : Eq c 0
      ⊢ Membership.mem (NormedSpace.polar 𝕜 s) 0
    -/
    exact (dualPairing 𝕜 E).flip.zero_mem_polar _
    /-
      🎉 no goals
    -/
  /-
    case neg
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    s : Set E
    x' : NormedSpace.Dual 𝕜 E
    c : 𝕜
    hc : ∀ (z : E), Membership.mem s z → LE.le (Norm.norm (x' z)) (Norm.norm c)
    c_zero : Not (Eq c 0)
    ⊢ Membership.mem (NormedSpace.polar 𝕜 s) (HSMul.hSMul (Inv.inv c) x')
  -/
  have eq : ∀ z, ‖c⁻¹ • x' z‖ = ‖c⁻¹‖ * ‖x' z‖ := fun z => norm_smul c⁻¹ _
  have le : ∀ z, z ∈ s → ‖c⁻¹ • x' z‖ ≤ ‖c⁻¹‖ * ‖c‖ := by
    intro z hzs
    rw [eq z]
    apply mul_le_mul (le_of_eq rfl) (hc z hzs) (norm_nonneg _) (norm_nonneg _)
  have cancel : ‖c⁻¹‖ * ‖c‖ = 1 := by
    simp only [c_zero, norm_eq_zero, Ne, not_false_iff, inv_mul_cancel₀, norm_inv]
  /-
    case neg
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    s : Set E
    x' : NormedSpace.Dual 𝕜 E
    c : 𝕜
    hc : ∀ (z : E), Membership.mem s z → LE.le (Norm.norm (x' z)) (Norm.norm c)
    c_zero : Not (Eq c 0)
    eq : ∀ (z : E), Eq (Norm.norm (HSMul.hSMul (Inv.inv c) (x' z))) (HMul.hMul (No …
    le : ∀ (z : E), Membership.mem s z → LE.le (Norm.norm (HSMul.hSMul (Inv.inv c) …
    cancel : Eq (HMul.hMul (Norm.norm (Inv.inv c)) (Norm.norm c)) 1
    ⊢ Membership.mem (NormedSpace.polar 𝕜 s) (HSMul.hSMul (Inv.inv c) x')
  -/
  rwa [cancel] at le
  /-
    🎉 no goals
  -/


theorem polar_ball_subset_closedBall_div {c : 𝕜} (hc : 1 < ‖c‖) {r : ℝ} (hr : 0 < r) :
    polar 𝕜 (ball (0 : E) r) ⊆ closedBall (0 : Dual 𝕜 E) (‖c‖ / r) := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    r : Real
    hr : LT.lt 0 r
    ⊢ HasSubset.Subset (NormedSpace.polar 𝕜 (Metric.ball 0 r)) (Metric.closedBall  …
  -/
  intro x' hx'
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    r : Real
    hr : LT.lt 0 r
    x' : NormedSpace.Dual 𝕜 E
    hx' : Membership.mem (NormedSpace.polar 𝕜 (Metric.ball 0 r)) x'
    ⊢ Membership.mem (Metric.closedBall 0 (HDiv.hDiv (Norm.norm c) r)) x'
  -/
  rw [mem_polar_iff] at hx'
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    r : Real
    hr : LT.lt 0 r
    x' : NormedSpace.Dual 𝕜 E
    hx' : ∀ (z : E), Membership.mem (Metric.ball 0 r) z → LE.le (Norm.norm (x' z)) 1
    ⊢ Membership.mem (Metric.closedBall 0 (HDiv.hDiv (Norm.norm c) r)) x'
  -/
  simp only [polar, mem_setOf, mem_closedBall_zero_iff, mem_ball_zero_iff] at *
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    r : Real
    hr : LT.lt 0 r
    x' : NormedSpace.Dual 𝕜 E
    hx' : ∀ (z : E), LT.lt (Norm.norm z) r → LE.le (Norm.norm (x' z)) 1
    ⊢ LE.le (Norm.norm x') (HDiv.hDiv (Norm.norm c) r)
  -/
  have hcr : 0 < ‖c‖ / r := div_pos (zero_lt_one.trans hc) hr
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    r : Real
    hr : LT.lt 0 r
    x' : NormedSpace.Dual 𝕜 E
    hx' : ∀ (z : E), LT.lt (Norm.norm z) r → LE.le (Norm.norm (x' z)) 1
    hcr : LT.lt 0 (HDiv.hDiv (Norm.norm c) r)
    ⊢ LE.le (Norm.norm x') (HDiv.hDiv (Norm.norm c) r)
  -/
  refine ContinuousLinearMap.opNorm_le_of_shell hr hcr.le hc fun x h₁ h₂ => ?_
  calc
    ‖x' x‖ ≤ 1 := hx' _ h₂
    _ ≤ ‖c‖ / r * ‖x‖ := (inv_le_iff_one_le_mul₀' hcr).1 (by rwa [inv_div])


theorem closedBall_inv_subset_polar_closedBall {r : ℝ} :
    closedBall (0 : Dual 𝕜 E) r⁻¹ ⊆ polar 𝕜 (closedBall (0 : E) r) := fun x' hx' x hx =>
  calc
    ‖x' x‖ ≤ ‖x'‖ * ‖x‖ := x'.le_opNorm x
    _ ≤ r⁻¹ * r :=
      (mul_le_mul (mem_closedBall_zero_iff.1 hx') (mem_closedBall_zero_iff.1 hx) (norm_nonneg _)
        (dist_nonneg.trans hx'))
    _ = r / r := inv_mul_eq_div _ _
    _ ≤ 1 := div_self_le_one r


/-- The `polar` of closed ball in a normed space `E` is the closed ball of the dual with
inverse radius. -/
theorem polar_closedBall {𝕜 E : Type*} [RCLike 𝕜] [NormedAddCommGroup E] [NormedSpace 𝕜 E] {r : ℝ}
    (hr : 0 < r) : polar 𝕜 (closedBall (0 : E) r) = closedBall (0 : Dual 𝕜 E) r⁻¹ := by
  /-
    𝕜 : Type u_3
    E : Type u_4
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    r : Real
    hr : LT.lt 0 r
    ⊢ Eq (NormedSpace.polar 𝕜 (Metric.closedBall 0 r)) (Metric.closedBall 0 (Inv.i …
  -/
  refine Subset.antisymm ?_ (closedBall_inv_subset_polar_closedBall 𝕜)
  /-
    𝕜 : Type u_3
    E : Type u_4
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    r : Real
    hr : LT.lt 0 r
    ⊢ HasSubset.Subset (NormedSpace.polar 𝕜 (Metric.closedBall 0 r)) (Metric.close …
  -/
  intro x' h
  /-
    𝕜 : Type u_3
    E : Type u_4
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    r : Real
    hr : LT.lt 0 r
    x' : NormedSpace.Dual 𝕜 E
    h : Membership.mem (NormedSpace.polar 𝕜 (Metric.closedBall 0 r)) x'
    ⊢ Membership.mem (Metric.closedBall 0 (Inv.inv r)) x'
  -/
  simp only [mem_closedBall_zero_iff]
  /-
    𝕜 : Type u_3
    E : Type u_4
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    r : Real
    hr : LT.lt 0 r
    x' : NormedSpace.Dual 𝕜 E
    h : Membership.mem (NormedSpace.polar 𝕜 (Metric.closedBall 0 r)) x'
    ⊢ LE.le (Norm.norm x') (Inv.inv r)
  -/
  refine ContinuousLinearMap.opNorm_le_of_ball hr (inv_nonneg.mpr hr.le) fun z _ => ?_
  /-
    𝕜 : Type u_3
    E : Type u_4
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    r : Real
    hr : LT.lt 0 r
    x' : NormedSpace.Dual 𝕜 E
    h : Membership.mem (NormedSpace.polar 𝕜 (Metric.closedBall 0 r)) x'
    z : E
    x✝ : Membership.mem (Metric.ball 0 r) z
    ⊢ LE.le (Norm.norm (x' z)) (HMul.hMul (Inv.inv r) (Norm.norm z))
  -/
  simpa only [one_div] using LinearMap.bound_of_ball_bound' hr 1 x'.toLinearMap h z
  /-
    🎉 no goals
  -/


theorem polar_ball {𝕜 E : Type*} [RCLike 𝕜] [NormedAddCommGroup E] [NormedSpace 𝕜 E] {r : ℝ}
    (hr : 0 < r) : polar 𝕜 (ball (0 : E) r) = closedBall (0 : Dual 𝕜 E) r⁻¹ := by
  /-
    𝕜 : Type u_3
    E : Type u_4
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    r : Real
    hr : LT.lt 0 r
    ⊢ Eq (NormedSpace.polar 𝕜 (Metric.ball 0 r)) (Metric.closedBall 0 (Inv.inv r))
  -/
  apply le_antisymm
    /-
      case a
      𝕜 : Type u_3
      E : Type u_4
      inst✝² : RCLike 𝕜
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      r : Real
      hr : LT.lt 0 r
      ⊢ LE.le (NormedSpace.polar 𝕜 (Metric.ball 0 r)) (Metric.closedBall 0 (Inv.inv  …
    -/
  · intro x hx
    /-
      case a
      𝕜 : Type u_3
      E : Type u_4
      inst✝² : RCLike 𝕜
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      r : Real
      hr : LT.lt 0 r
      x : NormedSpace.Dual 𝕜 E
      hx : Membership.mem (NormedSpace.polar 𝕜 (Metric.ball 0 r)) x
      ⊢ Membership.mem (Metric.closedBall 0 (Inv.inv r)) x
    -/
    rw [mem_closedBall_zero_iff]
    /-
      case a
      𝕜 : Type u_3
      E : Type u_4
      inst✝² : RCLike 𝕜
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      r : Real
      hr : LT.lt 0 r
      x : NormedSpace.Dual 𝕜 E
      hx : Membership.mem (NormedSpace.polar 𝕜 (Metric.ball 0 r)) x
      ⊢ LE.le (Norm.norm x) (Inv.inv r)
    -/
    apply le_of_forall_le_of_dense
    /-
      case a.h
      𝕜 : Type u_3
      E : Type u_4
      inst✝² : RCLike 𝕜
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      r : Real
      hr : LT.lt 0 r
      x : NormedSpace.Dual 𝕜 E
      hx : Membership.mem (NormedSpace.polar 𝕜 (Metric.ball 0 r)) x
      ⊢ ∀ (a : Real), LT.lt (Inv.inv r) a → LE.le (Norm.norm x) a
    -/
    intro a ha
    /-
      case a.h
      𝕜 : Type u_3
      E : Type u_4
      inst✝² : RCLike 𝕜
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      r : Real
      hr : LT.lt 0 r
      x : NormedSpace.Dual 𝕜 E
      hx : Membership.mem (NormedSpace.polar 𝕜 (Metric.ball 0 r)) x
      a : Real
      ha : LT.lt (Inv.inv r) a
      ⊢ LE.le (Norm.norm x) a
    -/
    rw [← mem_closedBall_zero_iff, ← (mul_div_cancel_left₀ a (Ne.symm (ne_of_lt hr)))]
    rw [← RCLike.norm_of_nonneg (K := 𝕜) (le_trans zero_le_one
      (le_of_lt ((inv_lt_iff_one_lt_mul₀' hr).mp ha)))]
    /-
      case a.h
      𝕜 : Type u_3
      E : Type u_4
      inst✝² : RCLike 𝕜
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      r : Real
      hr : LT.lt 0 r
      x : NormedSpace.Dual 𝕜 E
      hx : Membership.mem (NormedSpace.polar 𝕜 (Metric.ball 0 r)) x
      a : Real
      ha : LT.lt (Inv.inv r) a
      ⊢ Membership.mem (Metric.closedBall 0 (HDiv.hDiv (Norm.norm ↑(HMul.hMul r a))  …
    -/
    apply polar_ball_subset_closedBall_div _ hr hx
    rw [RCLike.norm_of_nonneg (K := 𝕜) (le_trans zero_le_one
      (le_of_lt ((inv_lt_iff_one_lt_mul₀' hr).mp ha)))]
    /-
      𝕜 : Type u_3
      E : Type u_4
      inst✝² : RCLike 𝕜
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      r : Real
      hr : LT.lt 0 r
      x : NormedSpace.Dual 𝕜 E
      hx : Membership.mem (NormedSpace.polar 𝕜 (Metric.ball 0 r)) x
      a : Real
      ha : LT.lt (Inv.inv r) a
      ⊢ LT.lt 1 (HMul.hMul r a)
    -/
    exact (inv_lt_iff_one_lt_mul₀' hr).mp ha
    /-
      🎉 no goals
    -/
    /-
      case a
      𝕜 : Type u_3
      E : Type u_4
      inst✝² : RCLike 𝕜
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      r : Real
      hr : LT.lt 0 r
      ⊢ LE.le (Metric.closedBall 0 (Inv.inv r)) (NormedSpace.polar 𝕜 (Metric.ball 0  …
    -/
  · rw [← polar_closedBall hr]
    /-
      case a
      𝕜 : Type u_3
      E : Type u_4
      inst✝² : RCLike 𝕜
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      r : Real
      hr : LT.lt 0 r
      ⊢ LE.le (NormedSpace.polar 𝕜 (Metric.closedBall 0 r)) (NormedSpace.polar 𝕜 (Me …
    -/
    exact LinearMap.polar_antitone _ ball_subset_closedBall
    /-
      🎉 no goals
    -/


/-- Given a neighborhood `s` of the origin in a normed space `E`, the dual norms
of all elements of the polar `polar 𝕜 s` are bounded by a constant. -/
theorem isBounded_polar_of_mem_nhds_zero {s : Set E} (s_nhd : s ∈ 𝓝 (0 : E)) :
    IsBounded (polar 𝕜 s) := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    s : Set E
    s_nhd : Membership.mem (nhds 0) s
    ⊢ Bornology.IsBounded (NormedSpace.polar 𝕜 s)
  -/
  obtain ⟨a, ha⟩ : ∃ a : 𝕜, 1 < ‖a‖ := NormedField.exists_one_lt_norm 𝕜
  /-
    case intro
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    s : Set E
    s_nhd : Membership.mem (nhds 0) s
    a : 𝕜
    ha : LT.lt 1 (Norm.norm a)
    ⊢ Bornology.IsBounded (NormedSpace.polar 𝕜 s)
  -/
  obtain ⟨r, r_pos, r_ball⟩ : ∃ r : ℝ, 0 < r ∧ ball 0 r ⊆ s := Metric.mem_nhds_iff.1 s_nhd
  exact isBounded_closedBall.subset
    (((dualPairing 𝕜 E).flip.polar_antitone r_ball).trans <|
      polar_ball_subset_closedBall_div ha r_pos)


@[simp]
theorem polar_empty : polar 𝕜 (∅ : Set E) = Set.univ :=
  LinearMap.polar_empty _


@[simp]
theorem polar_singleton {a : E} : polar 𝕜 {a} = { x | ‖x a‖ ≤ 1 } := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    a : E
    ⊢ Eq (NormedSpace.polar 𝕜 (Singleton.singleton a)) (setOf fun x => LE.le (Norm …
  -/
  simp only [polar, LinearMap.polar_singleton, LinearMap.flip_apply, dualPairing_apply]
  /-
    🎉 no goals
  -/


theorem mem_polar_singleton {a : E} (y : Dual 𝕜 E) : y ∈ polar 𝕜 {a} ↔ ‖y a‖ ≤ 1 := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    a : E
    y : NormedSpace.Dual 𝕜 E
    ⊢ Iff (Membership.mem (NormedSpace.polar 𝕜 (Singleton.singleton a)) y) (LE.le  …
  -/
  simp only [polar_singleton, mem_setOf_eq]
  /-
    🎉 no goals
  -/


theorem polar_zero : polar 𝕜 ({0} : Set E) = Set.univ :=
  LinearMap.polar_zero _


theorem sInter_polar_eq_closedBall {𝕜 E : Type*} [RCLike 𝕜] [NormedAddCommGroup E] [NormedSpace 𝕜 E]
    {r : ℝ} (hr : 0 < r) :
    ⋂₀ (polar 𝕜 '' { F | F.Finite ∧ F ⊆ closedBall (0 : E) r⁻¹ }) = closedBall 0 r := by
  /-
    𝕜 : Type u_3
    E : Type u_4
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    r : Real
    hr : LT.lt 0 r
    ⊢ Eq (Set.image (NormedSpace.polar 𝕜) (setOf fun F => And F.Finite (HasSubset. …
  -/
  conv_rhs => rw [← inv_inv r]
  rw [← polar_closedBall (inv_pos_of_pos hr), polar,
    (dualPairing 𝕜 E).flip.sInter_polar_finite_subset_eq_polar (closedBall (0 : E) r⁻¹)]


