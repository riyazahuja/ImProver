local postfix:1024 "♯" => continuousLinearMapOfBilin (𝕜 := ℝ)


theorem bounded_below (coercive : IsCoercive B) : ∃ C, 0 < C ∧ ∀ v, C * ‖v‖ ≤ ‖B♯ v‖ := by
  /-
    V : Type u
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : CompleteSpace V
    B : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    coercive : IsCoercive B
    ⊢ Exists fun C => And (LT.lt 0 C) (∀ (v : V), LE.le (HMul.hMul C (Norm.norm v) …
  -/
  rcases coercive with ⟨C, C_ge_0, coercivity⟩
  /-
    case intro.intro
    V : Type u
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : CompleteSpace V
    B : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    C : Real
    C_ge_0 : LT.lt 0 C
    coercivity : ∀ (u : V), LE.le (HMul.hMul (HMul.hMul C (Norm.norm u)) (Norm.nor …
    ⊢ Exists fun C => And (LT.lt 0 C) (∀ (v : V), LE.le (HMul.hMul C (Norm.norm v) …
  -/
  refine ⟨C, C_ge_0, ?_⟩
  /-
    case intro.intro
    V : Type u
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : CompleteSpace V
    B : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    C : Real
    C_ge_0 : LT.lt 0 C
    coercivity : ∀ (u : V), LE.le (HMul.hMul (HMul.hMul C (Norm.norm u)) (Norm.nor …
    ⊢ ∀ (v : V), LE.le (HMul.hMul C (Norm.norm v)) (Norm.norm ((InnerProductSpace. …
  -/
  intro v
  /-
    case intro.intro
    V : Type u
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : CompleteSpace V
    B : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    C : Real
    C_ge_0 : LT.lt 0 C
    coercivity : ∀ (u : V), LE.le (HMul.hMul (HMul.hMul C (Norm.norm u)) (Norm.nor …
    v : V
    ⊢ LE.le (HMul.hMul C (Norm.norm v)) (Norm.norm ((InnerProductSpace.continuousL …
  -/
  by_cases h : 0 < ‖v‖
    /-
      case pos
      V : Type u
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : CompleteSpace V
      B : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
      C : Real
      C_ge_0 : LT.lt 0 C
      coercivity : ∀ (u : V), LE.le (HMul.hMul (HMul.hMul C (Norm.norm u)) (Norm.nor …
      v : V
      h : LT.lt 0 (Norm.norm v)
      ⊢ LE.le (HMul.hMul C (Norm.norm v)) (Norm.norm ((InnerProductSpace.continuousL …
    -/
  · refine (mul_le_mul_right h).mp ?_
    calc
      C * ‖v‖ * ‖v‖ ≤ B v v := coercivity v
      _ = ⟪B♯ v, v⟫_ℝ := (continuousLinearMapOfBilin_apply B v v).symm
      _ ≤ ‖B♯ v‖ * ‖v‖ := real_inner_le_norm (B♯ v) v
    /-
      case neg
      V : Type u
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : CompleteSpace V
      B : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
      C : Real
      C_ge_0 : LT.lt 0 C
      coercivity : ∀ (u : V), LE.le (HMul.hMul (HMul.hMul C (Norm.norm u)) (Norm.nor …
      v : V
      h : Not (LT.lt 0 (Norm.norm v))
      ⊢ LE.le (HMul.hMul C (Norm.norm v)) (Norm.norm ((InnerProductSpace.continuousL …
    -/
  · have : v = 0 := by simpa using h
    /-
      case neg
      V : Type u
      inst✝² : NormedAddCommGroup V
      inst✝¹ : InnerProductSpace Real V
      inst✝ : CompleteSpace V
      B : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
      C : Real
      C_ge_0 : LT.lt 0 C
      coercivity : ∀ (u : V), LE.le (HMul.hMul (HMul.hMul C (Norm.norm u)) (Norm.nor …
      v : V
      h : Not (LT.lt 0 (Norm.norm v))
      this : Eq v 0
      ⊢ LE.le (HMul.hMul C (Norm.norm v)) (Norm.norm ((InnerProductSpace.continuousL …
    -/
    simp [this]
    /-
      🎉 no goals
    -/


theorem antilipschitz (coercive : IsCoercive B) : ∃ C : ℝ≥0, 0 < C ∧ AntilipschitzWith C B♯ := by
  /-
    V : Type u
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : CompleteSpace V
    B : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    coercive : IsCoercive B
    ⊢ Exists fun C => And (LT.lt 0 C) (AntilipschitzWith C ⇑(InnerProductSpace.con …
  -/
  rcases coercive.bounded_below with ⟨C, C_pos, below_bound⟩
  /-
    case intro.intro
    V : Type u
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : CompleteSpace V
    B : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    coercive : IsCoercive B
    C : Real
    C_pos : LT.lt 0 C
    below_bound : ∀ (v : V), LE.le (HMul.hMul C (Norm.norm v)) (Norm.norm ((InnerP …
    ⊢ Exists fun C => And (LT.lt 0 C) (AntilipschitzWith C ⇑(InnerProductSpace.con …
  -/
  refine ⟨C⁻¹.toNNReal, Real.toNNReal_pos.mpr (inv_pos.mpr C_pos), ?_⟩
  /-
    case intro.intro
    V : Type u
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : CompleteSpace V
    B : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    coercive : IsCoercive B
    C : Real
    C_pos : LT.lt 0 C
    below_bound : ∀ (v : V), LE.le (HMul.hMul C (Norm.norm v)) (Norm.norm ((InnerP …
    ⊢ AntilipschitzWith (Inv.inv C).toNNReal ⇑(InnerProductSpace.continuousLinearM …
  -/
  refine ContinuousLinearMap.antilipschitz_of_bound B♯ ?_
  simp_rw [Real.coe_toNNReal', max_eq_left_of_lt (inv_pos.mpr C_pos), ←
    inv_mul_le_iff₀ (inv_pos.mpr C_pos)]
  /-
    case intro.intro
    V : Type u
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : CompleteSpace V
    B : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    coercive : IsCoercive B
    C : Real
    C_pos : LT.lt 0 C
    below_bound : ∀ (v : V), LE.le (HMul.hMul C (Norm.norm v)) (Norm.norm ((InnerP …
    ⊢ ∀ (x : V), LE.le (HMul.hMul (Inv.inv (Inv.inv C)) (Norm.norm x)) (Norm.norm  …
  -/
  simpa using below_bound
  /-
    🎉 no goals
  -/


theorem ker_eq_bot (coercive : IsCoercive B) : ker B♯ = ⊥ := by
  /-
    V : Type u
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : CompleteSpace V
    B : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    coercive : IsCoercive B
    ⊢ Eq (LinearMap.ker (InnerProductSpace.continuousLinearMapOfBilin B)) Bot.bot
  -/
  rw [LinearMapClass.ker_eq_bot]
  /-
    V : Type u
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : CompleteSpace V
    B : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    coercive : IsCoercive B
    ⊢ Function.Injective ⇑(InnerProductSpace.continuousLinearMapOfBilin B)
  -/
  rcases coercive.antilipschitz with ⟨_, _, antilipschitz⟩
  /-
    case intro.intro
    V : Type u
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : CompleteSpace V
    B : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    coercive : IsCoercive B
    w✝ : NNReal
    left✝ : LT.lt 0 w✝
    antilipschitz : AntilipschitzWith w✝ ⇑(InnerProductSpace.continuousLinearMapOf …
    ⊢ Function.Injective ⇑(InnerProductSpace.continuousLinearMapOfBilin B)
  -/
  exact antilipschitz.injective
  /-
    🎉 no goals
  -/


theorem isClosed_range (coercive : IsCoercive B) : IsClosed (range B♯ : Set V) := by
  /-
    V : Type u
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : CompleteSpace V
    B : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    coercive : IsCoercive B
    ⊢ IsClosed ↑(LinearMap.range (InnerProductSpace.continuousLinearMapOfBilin B))
  -/
  rcases coercive.antilipschitz with ⟨_, _, antilipschitz⟩
  /-
    case intro.intro
    V : Type u
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : CompleteSpace V
    B : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    coercive : IsCoercive B
    w✝ : NNReal
    left✝ : LT.lt 0 w✝
    antilipschitz : AntilipschitzWith w✝ ⇑(InnerProductSpace.continuousLinearMapOf …
    ⊢ IsClosed ↑(LinearMap.range (InnerProductSpace.continuousLinearMapOfBilin B))
  -/
  exact antilipschitz.isClosed_range B♯.uniformContinuous
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-03-19")] alias closed_range := isClosed_range


theorem range_eq_top (coercive : IsCoercive B) : range B♯ = ⊤ := by
  /-
    V : Type u
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : CompleteSpace V
    B : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    coercive : IsCoercive B
    ⊢ Eq (LinearMap.range (InnerProductSpace.continuousLinearMapOfBilin B)) Top.top
  -/
  haveI := coercive.isClosed_range.completeSpace_coe
  /-
    V : Type u
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : CompleteSpace V
    B : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    coercive : IsCoercive B
    this : CompleteSpace ↑↑(LinearMap.range (InnerProductSpace.continuousLinearMap …
    ⊢ Eq (LinearMap.range (InnerProductSpace.continuousLinearMapOfBilin B)) Top.top
  -/
  rw [← (range B♯).orthogonal_orthogonal]
  /-
    V : Type u
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : CompleteSpace V
    B : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    coercive : IsCoercive B
    this : CompleteSpace ↑↑(LinearMap.range (InnerProductSpace.continuousLinearMap …
    ⊢ Eq (LinearMap.range (InnerProductSpace.continuousLinearMapOfBilin B)).orthog …
  -/
  rw [Submodule.eq_top_iff']
  /-
    V : Type u
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : CompleteSpace V
    B : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    coercive : IsCoercive B
    this : CompleteSpace ↑↑(LinearMap.range (InnerProductSpace.continuousLinearMap …
    ⊢ ∀ (x : V), Membership.mem (LinearMap.range (InnerProductSpace.continuousLine …
  -/
  intro v w mem_w_orthogonal
  /-
    V : Type u
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : CompleteSpace V
    B : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    coercive : IsCoercive B
    this : CompleteSpace ↑↑(LinearMap.range (InnerProductSpace.continuousLinearMap …
    v w : V
    mem_w_orthogonal : Membership.mem (LinearMap.range (InnerProductSpace.continuo …
    ⊢ Eq (Inner.inner w v) 0
  -/
  rcases coercive with ⟨C, C_pos, coercivity⟩
  obtain rfl : w = 0 := by
    rw [← norm_eq_zero, ← mul_self_eq_zero, ← mul_right_inj' C_pos.ne', mul_zero, ←
      mul_assoc]
    apply le_antisymm
    · calc
        C * ‖w‖ * ‖w‖ ≤ B w w := coercivity w
        _ = ⟪B♯ w, w⟫_ℝ := (continuousLinearMapOfBilin_apply B w w).symm
        _ = 0 := mem_w_orthogonal _ ⟨w, rfl⟩
    · positivity
  /-
    case intro.intro
    V : Type u
    inst✝² : NormedAddCommGroup V
    inst✝¹ : InnerProductSpace Real V
    inst✝ : CompleteSpace V
    B : ContinuousLinearMap (RingHom.id Real) V (ContinuousLinearMap (RingHom.id R …
    this : CompleteSpace ↑↑(LinearMap.range (InnerProductSpace.continuousLinearMap …
    v : V
    C : Real
    C_pos : LT.lt 0 C
    coercivity : ∀ (u : V), LE.le (HMul.hMul (HMul.hMul C (Norm.norm u)) (Norm.nor …
    mem_w_orthogonal : Membership.mem (LinearMap.range (InnerProductSpace.continuo …
    ⊢ Eq (Inner.inner 0 v) 0
  -/
  exact inner_zero_left _
  /-
    🎉 no goals
  -/


/-- The Lax-Milgram equivalence of a coercive bounded bilinear operator:
for all `v : V`, `continuousLinearEquivOfBilin B v` is the unique element `V`
such that `continuousLinearEquivOfBilin B v, w⟫ = B v w`.
The Lax-Milgram theorem states that this is a continuous equivalence.
-/
def continuousLinearEquivOfBilin (coercive : IsCoercive B) : V ≃L[ℝ] V :=
  ContinuousLinearEquiv.ofBijective B♯ coercive.ker_eq_bot coercive.range_eq_top


@[simp]
theorem continuousLinearEquivOfBilin_apply (coercive : IsCoercive B) (v w : V) :
    ⟪coercive.continuousLinearEquivOfBilin v, w⟫_ℝ = B v w :=
  continuousLinearMapOfBilin_apply B v w


theorem unique_continuousLinearEquivOfBilin (coercive : IsCoercive B) {v f : V}
    (is_lax_milgram : ∀ w, ⟪f, w⟫_ℝ = B v w) : f = coercive.continuousLinearEquivOfBilin v :=
  unique_continuousLinearMapOfBilin B is_lax_milgram


