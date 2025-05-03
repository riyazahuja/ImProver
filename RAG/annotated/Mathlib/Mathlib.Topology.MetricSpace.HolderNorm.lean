/-- The `r`-Hölder (semi-)norm in `ℝ≥0∞` of a function `f` is the least non-negative real
number `C` for which `f` is `r`-Hölder continuous with constant `C`. This is `∞` if no such
non-negative real exists. -/
noncomputable
def eHolderNorm (r : ℝ≥0) (f : X → Y) : ℝ≥0∞ := ⨅ (C) (_ : HolderWith C r f), C


/-- The `r`-Hölder (semi)norm in `ℝ≥0`. -/
noncomputable
def nnHolderNorm (r : ℝ≥0) (f : X → Y) : ℝ≥0 := (eHolderNorm r f).toNNReal


/-- A function `f` is `MemHolder r f` if it is Hölder continuous. Namely, `f` has a finite
`r`-Hölder constant. This is equivalent to `f` having finite Hölder norm.
c.f. `memHolder_iff`. -/
def MemHolder (r : ℝ≥0) (f : X → Y) : Prop := ∃ C, HolderWith C r f


lemma HolderWith.memHolder {C : ℝ≥0} (hf : HolderWith C r f) : MemHolder r f := ⟨C, hf⟩


@[simp] lemma eHolderNorm_lt_top : eHolderNorm r f < ∞ ↔ MemHolder r f := by
  refine ⟨fun h => ?_,
    fun hf => let ⟨C, hC⟩ := hf; iInf_lt_top.2 ⟨C, iInf_lt_top.2 ⟨hC, coe_lt_top⟩⟩⟩
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : PseudoEMetricSpace X
    inst✝ : PseudoEMetricSpace Y
    r : NNReal
    f : X → Y
    h : LT.lt (eHolderNorm r f) Top.top
    ⊢ MemHolder r f
  -/
  simp_rw [eHolderNorm, iInf_lt_top] at h
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : PseudoEMetricSpace X
    inst✝ : PseudoEMetricSpace Y
    r : NNReal
    f : X → Y
    h : Exists fun i => Exists fun i_1 => LT.lt (↑i) Top.top
    ⊢ MemHolder r f
  -/
  exact let ⟨C, hC, _⟩ := h; ⟨C, hC⟩
  /-
    🎉 no goals
  -/


lemma eHolderNorm_ne_top : eHolderNorm r f ≠ ∞ ↔ MemHolder r f := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : PseudoEMetricSpace X
    inst✝ : PseudoEMetricSpace Y
    r : NNReal
    f : X → Y
    ⊢ Iff (Ne (eHolderNorm r f) Top.top) (MemHolder r f)
  -/
  rw [← eHolderNorm_lt_top, lt_top_iff_ne_top]
  /-
    🎉 no goals
  -/


@[simp] lemma eHolderNorm_eq_top : eHolderNorm r f = ∞ ↔ ¬ MemHolder r f := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : PseudoEMetricSpace X
    inst✝ : PseudoEMetricSpace Y
    r : NNReal
    f : X → Y
    ⊢ Iff (Eq (eHolderNorm r f) Top.top) (Not (MemHolder r f))
  -/
  rw [← eHolderNorm_ne_top, not_not]
  /-
    🎉 no goals
  -/


protected alias ⟨_, MemHolder.eHolderNorm_lt_top⟩ := eHolderNorm_lt_top

protected alias ⟨_, MemHolder.eHolderNorm_ne_top⟩ := eHolderNorm_ne_top


lemma coe_nnHolderNorm_le_eHolderNorm {r : ℝ≥0} {f : X → Y} :
    (nnHolderNorm r f : ℝ≥0∞) ≤ eHolderNorm r f :=
  coe_toNNReal_le_self


variable (X) in
@[simp]
lemma eHolderNorm_const (r : ℝ≥0) (c : Y) : eHolderNorm r (Function.const X c) = 0 := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : PseudoEMetricSpace X
    inst✝ : PseudoEMetricSpace Y
    r : NNReal
    c : Y
    ⊢ Eq (eHolderNorm r (Function.const X c)) 0
  -/
  rw [eHolderNorm, ← ENNReal.bot_eq_zero, iInf₂_eq_bot]
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : PseudoEMetricSpace X
    inst✝ : PseudoEMetricSpace Y
    r : NNReal
    c : Y
    ⊢ ∀ (b : ENNReal), GT.gt b Bot.bot → Exists fun i => Exists fun j => LT.lt (↑i …
  -/
  exact fun C' hC' => ⟨0, .const, hC'⟩
  /-
    🎉 no goals
  -/


variable (X) in
@[simp]
lemma eHolderNorm_zero [Zero Y] (r : ℝ≥0) : eHolderNorm r (0 : X → Y) = 0 :=
  eHolderNorm_const X r 0


variable (X) in
@[simp]
lemma nnHolderNorm_const (r : ℝ≥0) (c : Y) : nnHolderNorm r (Function.const X c) = 0 := by
  refine le_antisymm (ENNReal.coe_le_coe.1 <|
    le_trans coe_nnHolderNorm_le_eHolderNorm ?_) (zero_le _)
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : PseudoEMetricSpace X
    inst✝ : PseudoEMetricSpace Y
    r : NNReal
    c : Y
    ⊢ LE.le (eHolderNorm r (Function.const X c)) ↑0
  -/
  rw [eHolderNorm_const]
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : PseudoEMetricSpace X
    inst✝ : PseudoEMetricSpace Y
    r : NNReal
    c : Y
    ⊢ LE.le 0 ↑0
  -/
  rfl
  /-
    🎉 no goals
  -/


variable (X) in
@[simp]
lemma nnHolderNorm_zero [Zero Y] (r : ℝ≥0) : nnHolderNorm r (0 : X → Y) = 0 :=
  nnHolderNorm_const X r 0


lemma eHolderNorm_of_isEmpty [hX : IsEmpty X] :
    eHolderNorm r f = 0 := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : PseudoEMetricSpace X
    inst✝ : PseudoEMetricSpace Y
    r : NNReal
    f : X → Y
    hX : IsEmpty X
    ⊢ Eq (eHolderNorm r f) 0
  -/
  rw [eHolderNorm, ← ENNReal.bot_eq_zero, iInf₂_eq_bot]
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : PseudoEMetricSpace X
    inst✝ : PseudoEMetricSpace Y
    r : NNReal
    f : X → Y
    hX : IsEmpty X
    ⊢ ∀ (b : ENNReal), GT.gt b Bot.bot → Exists fun i => Exists fun j => LT.lt (↑i …
  -/
  exact fun ε hε => ⟨0, .of_isEmpty, hε⟩
  /-
    🎉 no goals
  -/


lemma HolderWith.eHolderNorm_le {C : ℝ≥0} (hf : HolderWith C r f) :
    eHolderNorm r f ≤ C :=
  iInf₂_le C hf


/-- See also `memHolder_const` for the version with the spelling `fun _ ↦ c`. -/
@[simp]
lemma memHolder_const {c : Y} : MemHolder r (Function.const X c) :=
  (HolderWith.const (C := 0)).memHolder


/-- Version of `memHolder_const` with the spelling `fun _ ↦ c` for the constant function. -/
@[simp]
lemma memHolder_const' {c : Y} : MemHolder r (fun _ ↦ c : X → Y) :=
  memHolder_const


@[simp]
lemma memHolder_zero [Zero Y] : MemHolder r (0 : X → Y) :=
  memHolder_const


lemma eHolderNorm_eq_zero {r : ℝ≥0} {f : X → Y} :
    eHolderNorm r f = 0 ↔ ∀ x₁ x₂, f x₁ = f x₂ := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : MetricSpace X
    inst✝ : EMetricSpace Y
    r : NNReal
    f : X → Y
    ⊢ Iff (Eq (eHolderNorm r f) 0) (∀ (x₁ x₂ : X), Eq (f x₁) (f x₂))
  -/
  constructor
    /-
      case mp
      X : Type u_1
      Y : Type u_2
      inst✝¹ : MetricSpace X
      inst✝ : EMetricSpace Y
      r : NNReal
      f : X → Y
      ⊢ Eq (eHolderNorm r f) 0 → ∀ (x₁ x₂ : X), Eq (f x₁) (f x₂)
    -/
  · refine fun h x₁ x₂ => ?_
    /-
      case mp
      X : Type u_1
      Y : Type u_2
      inst✝¹ : MetricSpace X
      inst✝ : EMetricSpace Y
      r : NNReal
      f : X → Y
      h : Eq (eHolderNorm r f) 0
      x₁ x₂ : X
      ⊢ Eq (f x₁) (f x₂)
    -/
    by_cases hx : x₁ = x₂
      /-
        case pos
        X : Type u_1
        Y : Type u_2
        inst✝¹ : MetricSpace X
        inst✝ : EMetricSpace Y
        r : NNReal
        f : X → Y
        h : Eq (eHolderNorm r f) 0
        x₁ x₂ : X
        hx : Eq x₁ x₂
        ⊢ Eq (f x₁) (f x₂)
      -/
    · rw [hx]
      /-
        🎉 no goals
      -/
      /-
        case neg
        X : Type u_1
        Y : Type u_2
        inst✝¹ : MetricSpace X
        inst✝ : EMetricSpace Y
        r : NNReal
        f : X → Y
        h : Eq (eHolderNorm r f) 0
        x₁ x₂ : X
        hx : Not (Eq x₁ x₂)
        ⊢ Eq (f x₁) (f x₂)
      -/
    · rw [eHolderNorm, ← ENNReal.bot_eq_zero, iInf₂_eq_bot] at h
      /-
        case neg
        X : Type u_1
        Y : Type u_2
        inst✝¹ : MetricSpace X
        inst✝ : EMetricSpace Y
        r : NNReal
        f : X → Y
        h : ∀ (b : ENNReal), GT.gt b Bot.bot → Exists fun i => Exists fun j => LT.lt ( …
        x₁ x₂ : X
        hx : Not (Eq x₁ x₂)
        ⊢ Eq (f x₁) (f x₂)
      -/
      rw [← edist_eq_zero, ← le_zero_iff]
      /-
        case neg
        X : Type u_1
        Y : Type u_2
        inst✝¹ : MetricSpace X
        inst✝ : EMetricSpace Y
        r : NNReal
        f : X → Y
        h : ∀ (b : ENNReal), GT.gt b Bot.bot → Exists fun i => Exists fun j => LT.lt ( …
        x₁ x₂ : X
        hx : Not (Eq x₁ x₂)
        ⊢ LE.le (EDist.edist (f x₁) (f x₂)) 0
      -/
      refine le_of_forall_lt' fun b hb => ?_
      obtain ⟨C, hC, hC'⟩ := h (b / edist x₁ x₂ ^ (r : ℝ))
        (ENNReal.div_pos hb.ne.symm (ENNReal.rpow_lt_top_of_nonneg zero_le_coe
          (edist_lt_top x₁ x₂).ne).ne)
      /-
        case neg.intro.intro
        X : Type u_1
        Y : Type u_2
        inst✝¹ : MetricSpace X
        inst✝ : EMetricSpace Y
        r : NNReal
        f : X → Y
        h : ∀ (b : ENNReal), GT.gt b Bot.bot → Exists fun i => Exists fun j => LT.lt ( …
        x₁ x₂ : X
        hx : Not (Eq x₁ x₂)
        b : ENNReal
        hb : LT.lt 0 b
        C : NNReal
        hC : HolderWith C r f
        hC' : LT.lt (↑C) (HDiv.hDiv b (HPow.hPow (EDist.edist x₁ x₂) ↑r))
        ⊢ LT.lt (EDist.edist (f x₁) (f x₂)) b
      -/
      exact lt_of_le_of_lt (hC x₁ x₂) <| ENNReal.mul_lt_of_lt_div hC'
      /-
        🎉 no goals
      -/
    /-
      case mpr
      X : Type u_1
      Y : Type u_2
      inst✝¹ : MetricSpace X
      inst✝ : EMetricSpace Y
      r : NNReal
      f : X → Y
      ⊢ (∀ (x₁ x₂ : X), Eq (f x₁) (f x₂)) → Eq (eHolderNorm r f) 0
    -/
  · intro h
    /-
      case mpr
      X : Type u_1
      Y : Type u_2
      inst✝¹ : MetricSpace X
      inst✝ : EMetricSpace Y
      r : NNReal
      f : X → Y
      h : ∀ (x₁ x₂ : X), Eq (f x₁) (f x₂)
      ⊢ Eq (eHolderNorm r f) 0
    -/
    cases' isEmpty_or_nonempty X with hX hX
      /-
        case mpr.inl
        X : Type u_1
        Y : Type u_2
        inst✝¹ : MetricSpace X
        inst✝ : EMetricSpace Y
        r : NNReal
        f : X → Y
        h : ∀ (x₁ x₂ : X), Eq (f x₁) (f x₂)
        hX : IsEmpty X
        ⊢ Eq (eHolderNorm r f) 0
      -/
    · haveI := hX
      /-
        case mpr.inl
        X : Type u_1
        Y : Type u_2
        inst✝¹ : MetricSpace X
        inst✝ : EMetricSpace Y
        r : NNReal
        f : X → Y
        h : ∀ (x₁ x₂ : X), Eq (f x₁) (f x₂)
        hX this : IsEmpty X
        ⊢ Eq (eHolderNorm r f) 0
      -/
      exact eHolderNorm_of_isEmpty
      /-
        🎉 no goals
      -/
      /-
        case mpr.inr
        X : Type u_1
        Y : Type u_2
        inst✝¹ : MetricSpace X
        inst✝ : EMetricSpace Y
        r : NNReal
        f : X → Y
        h : ∀ (x₁ x₂ : X), Eq (f x₁) (f x₂)
        hX : Nonempty X
        ⊢ Eq (eHolderNorm r f) 0
      -/
    · rw [← eHolderNorm_const X r (f hX.some)]
      /-
        case mpr.inr
        X : Type u_1
        Y : Type u_2
        inst✝¹ : MetricSpace X
        inst✝ : EMetricSpace Y
        r : NNReal
        f : X → Y
        h : ∀ (x₁ x₂ : X), Eq (f x₁) (f x₂)
        hX : Nonempty X
        ⊢ Eq (eHolderNorm r f) (eHolderNorm r (Function.const X (f hX.some)))
      -/
      congr
      /-
        case mpr.inr.e_f
        X : Type u_1
        Y : Type u_2
        inst✝¹ : MetricSpace X
        inst✝ : EMetricSpace Y
        r : NNReal
        f : X → Y
        h : ∀ (x₁ x₂ : X), Eq (f x₁) (f x₂)
        hX : Nonempty X
        ⊢ Eq f (Function.const X (f hX.some))
      -/
      simp [funext_iff, h _ hX.some]
      /-
        🎉 no goals
      -/


lemma MemHolder.holderWith {r : ℝ≥0} {f : X → Y} (hf : MemHolder r f) :
    HolderWith (nnHolderNorm r f) r f := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : MetricSpace X
    inst✝ : EMetricSpace Y
    r : NNReal
    f : X → Y
    hf : MemHolder r f
    ⊢ HolderWith (nnHolderNorm r f) r f
  -/
  intros x₁ x₂
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : MetricSpace X
    inst✝ : EMetricSpace Y
    r : NNReal
    f : X → Y
    hf : MemHolder r f
    x₁ x₂ : X
    ⊢ LE.le (EDist.edist (f x₁) (f x₂)) (HMul.hMul (↑(nnHolderNorm r f)) (HPow.hPo …
  -/
  by_cases hx : x₁ = x₂
    /-
      case pos
      X : Type u_1
      Y : Type u_2
      inst✝¹ : MetricSpace X
      inst✝ : EMetricSpace Y
      r : NNReal
      f : X → Y
      hf : MemHolder r f
      x₁ x₂ : X
      hx : Eq x₁ x₂
      ⊢ LE.le (EDist.edist (f x₁) (f x₂)) (HMul.hMul (↑(nnHolderNorm r f)) (HPow.hPo …
    -/
  · simp only [hx, edist_self, zero_le]
    /-
      🎉 no goals
    -/
  /-
    case neg
    X : Type u_1
    Y : Type u_2
    inst✝¹ : MetricSpace X
    inst✝ : EMetricSpace Y
    r : NNReal
    f : X → Y
    hf : MemHolder r f
    x₁ x₂ : X
    hx : Not (Eq x₁ x₂)
    ⊢ LE.le (EDist.edist (f x₁) (f x₂)) (HMul.hMul (↑(nnHolderNorm r f)) (HPow.hPo …
  -/
  rw [nnHolderNorm, eHolderNorm, coe_toNNReal]
  /-
    case neg
    X : Type u_1
    Y : Type u_2
    inst✝¹ : MetricSpace X
    inst✝ : EMetricSpace Y
    r : NNReal
    f : X → Y
    hf : MemHolder r f
    x₁ x₂ : X
    hx : Not (Eq x₁ x₂)
    ⊢ LE.le (EDist.edist (f x₁) (f x₂)) (HMul.hMul (iInf fun C => iInf fun x => ↑C …
  -/
  on_goal 2 => exact hf.eHolderNorm_lt_top.ne
  have h₁ : edist x₁ x₂ ^ (r : ℝ) ≠ 0 :=
    (Ne.symm <| ne_of_lt <| ENNReal.rpow_pos (edist_pos.2 hx) (edist_lt_top x₁ x₂).ne)
  have h₂ : edist x₁ x₂ ^ (r : ℝ) ≠ ∞ := by
    simp [(edist_lt_top x₁ x₂).ne]
  /-
    case neg
    X : Type u_1
    Y : Type u_2
    inst✝¹ : MetricSpace X
    inst✝ : EMetricSpace Y
    r : NNReal
    f : X → Y
    hf : MemHolder r f
    x₁ x₂ : X
    hx : Not (Eq x₁ x₂)
    h₁ : Ne (HPow.hPow (EDist.edist x₁ x₂) ↑r) 0
    h₂ : Ne (HPow.hPow (EDist.edist x₁ x₂) ↑r) Top.top
    ⊢ LE.le (EDist.edist (f x₁) (f x₂)) (HMul.hMul (iInf fun C => iInf fun x => ↑C …
  -/
  rw [← ENNReal.div_le_iff h₁ h₂]
  /-
    case neg
    X : Type u_1
    Y : Type u_2
    inst✝¹ : MetricSpace X
    inst✝ : EMetricSpace Y
    r : NNReal
    f : X → Y
    hf : MemHolder r f
    x₁ x₂ : X
    hx : Not (Eq x₁ x₂)
    h₁ : Ne (HPow.hPow (EDist.edist x₁ x₂) ↑r) 0
    h₂ : Ne (HPow.hPow (EDist.edist x₁ x₂) ↑r) Top.top
    ⊢ LE.le (HDiv.hDiv (EDist.edist (f x₁) (f x₂)) (HPow.hPow (EDist.edist x₁ x₂)  …
  -/
  refine le_iInf₂ fun C hC => ?_
  /-
    case neg
    X : Type u_1
    Y : Type u_2
    inst✝¹ : MetricSpace X
    inst✝ : EMetricSpace Y
    r : NNReal
    f : X → Y
    hf : MemHolder r f
    x₁ x₂ : X
    hx : Not (Eq x₁ x₂)
    h₁ : Ne (HPow.hPow (EDist.edist x₁ x₂) ↑r) 0
    h₂ : Ne (HPow.hPow (EDist.edist x₁ x₂) ↑r) Top.top
    C : NNReal
    hC : HolderWith C r f
    ⊢ LE.le (HDiv.hDiv (EDist.edist (f x₁) (f x₂)) (HPow.hPow (EDist.edist x₁ x₂)  …
  -/
  rw [ENNReal.div_le_iff h₁ h₂]
  /-
    case neg
    X : Type u_1
    Y : Type u_2
    inst✝¹ : MetricSpace X
    inst✝ : EMetricSpace Y
    r : NNReal
    f : X → Y
    hf : MemHolder r f
    x₁ x₂ : X
    hx : Not (Eq x₁ x₂)
    h₁ : Ne (HPow.hPow (EDist.edist x₁ x₂) ↑r) 0
    h₂ : Ne (HPow.hPow (EDist.edist x₁ x₂) ↑r) Top.top
    C : NNReal
    hC : HolderWith C r f
    ⊢ LE.le (EDist.edist (f x₁) (f x₂)) (HMul.hMul (↑C) (HPow.hPow (EDist.edist x₁ …
  -/
  exact hC x₁ x₂
  /-
    🎉 no goals
  -/


lemma memHolder_iff_holderWith {r : ℝ≥0} {f : X → Y} :
    MemHolder r f ↔ HolderWith (nnHolderNorm r f) r f :=
  ⟨MemHolder.holderWith, HolderWith.memHolder⟩


lemma MemHolder.coe_nnHolderNorm_eq_eHolderNorm
    {r : ℝ≥0} {f : X → Y} (hf : MemHolder r f) :
    (nnHolderNorm r f : ℝ≥0∞) = eHolderNorm r f := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : MetricSpace X
    inst✝ : EMetricSpace Y
    r : NNReal
    f : X → Y
    hf : MemHolder r f
    ⊢ Eq (↑(nnHolderNorm r f)) (eHolderNorm r f)
  -/
  rw [nnHolderNorm, coe_toNNReal]
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : MetricSpace X
    inst✝ : EMetricSpace Y
    r : NNReal
    f : X → Y
    hf : MemHolder r f
    ⊢ Ne (eHolderNorm r f) Top.top
  -/
  exact ne_of_lt <| lt_of_le_of_lt hf.holderWith.eHolderNorm_le <| coe_lt_top
  /-
    🎉 no goals
  -/


lemma HolderWith.nnholderNorm_le {C r : ℝ≥0} {f : X → Y} (hf : HolderWith C r f) :
    nnHolderNorm r f ≤ C := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : MetricSpace X
    inst✝ : EMetricSpace Y
    C r : NNReal
    f : X → Y
    hf : HolderWith C r f
    ⊢ LE.le (nnHolderNorm r f) C
  -/
  rw [← ENNReal.coe_le_coe, hf.memHolder.coe_nnHolderNorm_eq_eHolderNorm]
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : MetricSpace X
    inst✝ : EMetricSpace Y
    C r : NNReal
    f : X → Y
    hf : HolderWith C r f
    ⊢ LE.le (eHolderNorm r f) ↑C
  -/
  exact hf.eHolderNorm_le
  /-
    🎉 no goals
  -/


lemma MemHolder.comp {r s : ℝ≥0} {Z : Type*} [MetricSpace Z] {f : Z → X} {g : X → Y}
    (hf : MemHolder r f) (hg : MemHolder s g) : MemHolder (s * r) (g ∘ f) :=
  (hg.holderWith.comp hf.holderWith).memHolder


lemma MemHolder.nnHolderNorm_eq_zero {r : ℝ≥0} {f : X → Y} (hf : MemHolder r f) :
    nnHolderNorm r f = 0 ↔ ∀ x₁ x₂, f x₁ = f x₂ := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : MetricSpace X
    inst✝ : EMetricSpace Y
    r : NNReal
    f : X → Y
    hf : MemHolder r f
    ⊢ Iff (Eq (nnHolderNorm r f) 0) (∀ (x₁ x₂ : X), Eq (f x₁) (f x₂))
  -/
  rw [← ENNReal.coe_eq_zero, hf.coe_nnHolderNorm_eq_eHolderNorm, eHolderNorm_eq_zero]
  /-
    🎉 no goals
  -/


lemma MemHolder.add (hf : MemHolder r f) (hg : MemHolder r g) : MemHolder r (f + g) :=
  (hf.holderWith.add hg.holderWith).memHolder


lemma MemHolder.smul {𝕜} [NormedDivisionRing 𝕜] [Module 𝕜 Y] [BoundedSMul 𝕜 Y]
    {c : 𝕜} (hf : MemHolder r f) : MemHolder r (c • f) :=
  (hf.holderWith.smul c).memHolder


lemma MemHolder.nsmul [Module ℝ Y] [BoundedSMul ℝ Y] (n : ℕ) (hf : MemHolder r f) :
    MemHolder r (n • f) := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝³ : MetricSpace X
    inst✝² : NormedAddCommGroup Y
    r : NNReal
    f : X → Y
    inst✝¹ : Module Real Y
    inst✝ : BoundedSMul Real Y
    n : Nat
    hf : MemHolder r f
    ⊢ MemHolder r (HSMul.hSMul n f)
  -/
  simp [← Nat.cast_smul_eq_nsmul (R := ℝ), hf.smul]
  /-
    🎉 no goals
  -/


lemma MemHolder.nnHolderNorm_add_le (hf : MemHolder r f) (hg : MemHolder r g) :
    nnHolderNorm r (f + g) ≤ nnHolderNorm r f + nnHolderNorm r g :=
  (hf.add hg).holderWith.nnholderNorm_le.trans <|
    coe_le_coe.2 (hf.holderWith.add hg.holderWith).nnholderNorm_le


lemma eHolderNorm_add_le :
    eHolderNorm r (f + g) ≤ eHolderNorm r f + eHolderNorm r g := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝¹ : MetricSpace X
    inst✝ : NormedAddCommGroup Y
    r : NNReal
    f g : X → Y
    ⊢ LE.le (eHolderNorm r (HAdd.hAdd f g)) (HAdd.hAdd (eHolderNorm r f) (eHolderN …
  -/
  by_cases hfg : MemHolder r f  ∧ MemHolder r g
    /-
      case pos
      X : Type u_1
      Y : Type u_2
      inst✝¹ : MetricSpace X
      inst✝ : NormedAddCommGroup Y
      r : NNReal
      f g : X → Y
      hfg : And (MemHolder r f) (MemHolder r g)
      ⊢ LE.le (eHolderNorm r (HAdd.hAdd f g)) (HAdd.hAdd (eHolderNorm r f) (eHolderN …
    -/
  · obtain ⟨hf, hg⟩ := hfg
    rw [← hf.coe_nnHolderNorm_eq_eHolderNorm, ← hg.coe_nnHolderNorm_eq_eHolderNorm,
      ← (hf.add hg).coe_nnHolderNorm_eq_eHolderNorm, ← coe_add, ENNReal.coe_le_coe]
    /-
      case pos.intro
      X : Type u_1
      Y : Type u_2
      inst✝¹ : MetricSpace X
      inst✝ : NormedAddCommGroup Y
      r : NNReal
      f g : X → Y
      hf : MemHolder r f
      hg : MemHolder r g
      ⊢ LE.le (nnHolderNorm r (HAdd.hAdd f g)) (HAdd.hAdd (nnHolderNorm r f) (nnHold …
    -/
    exact hf.nnHolderNorm_add_le hg
    /-
      🎉 no goals
    -/
    /-
      case neg
      X : Type u_1
      Y : Type u_2
      inst✝¹ : MetricSpace X
      inst✝ : NormedAddCommGroup Y
      r : NNReal
      f g : X → Y
      hfg : Not (And (MemHolder r f) (MemHolder r g))
      ⊢ LE.le (eHolderNorm r (HAdd.hAdd f g)) (HAdd.hAdd (eHolderNorm r f) (eHolderN …
    -/
  · rw [Classical.not_and_iff_or_not_not, ← eHolderNorm_eq_top, ← eHolderNorm_eq_top] at hfg
    /-
      case neg
      X : Type u_1
      Y : Type u_2
      inst✝¹ : MetricSpace X
      inst✝ : NormedAddCommGroup Y
      r : NNReal
      f g : X → Y
      hfg : Or (Eq (eHolderNorm r f) Top.top) (Eq (eHolderNorm r g) Top.top)
      ⊢ LE.le (eHolderNorm r (HAdd.hAdd f g)) (HAdd.hAdd (eHolderNorm r f) (eHolderN …
    -/
    obtain (h | h) := hfg
    /-
      case neg.inl
      X : Type u_1
      Y : Type u_2
      inst✝¹ : MetricSpace X
      inst✝ : NormedAddCommGroup Y
      r : NNReal
      f g : X → Y
      h : Eq (eHolderNorm r f) Top.top
      ⊢ LE.le (eHolderNorm r (HAdd.hAdd f g)) (HAdd.hAdd (eHolderNorm r f) (eHolderN …
    -/
    all_goals simp [h]
    /-
      🎉 no goals
    -/


lemma eHolderNorm_smul {α} [NormedDivisionRing α] [Module α Y] [BoundedSMul α Y] (c : α) :
    eHolderNorm r (c • f) = ‖c‖₊ * eHolderNorm r f := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝⁴ : MetricSpace X
    inst✝³ : NormedAddCommGroup Y
    r : NNReal
    f : X → Y
    α : Type u_3
    inst✝² : NormedDivisionRing α
    inst✝¹ : Module α Y
    inst✝ : BoundedSMul α Y
    c : α
    ⊢ Eq (eHolderNorm r (HSMul.hSMul c f)) (HMul.hMul (↑(NNNorm.nnnorm c)) (eHolde …
  -/
  by_cases hc : ‖c‖₊ = 0
    /-
      case pos
      X : Type u_1
      Y : Type u_2
      inst✝⁴ : MetricSpace X
      inst✝³ : NormedAddCommGroup Y
      r : NNReal
      f : X → Y
      α : Type u_3
      inst✝² : NormedDivisionRing α
      inst✝¹ : Module α Y
      inst✝ : BoundedSMul α Y
      c : α
      hc : Eq (NNNorm.nnnorm c) 0
      ⊢ Eq (eHolderNorm r (HSMul.hSMul c f)) (HMul.hMul (↑(NNNorm.nnnorm c)) (eHolde …
    -/
  · rw [nnnorm_eq_zero] at hc
    /-
      case pos
      X : Type u_1
      Y : Type u_2
      inst✝⁴ : MetricSpace X
      inst✝³ : NormedAddCommGroup Y
      r : NNReal
      f : X → Y
      α : Type u_3
      inst✝² : NormedDivisionRing α
      inst✝¹ : Module α Y
      inst✝ : BoundedSMul α Y
      c : α
      hc : Eq c 0
      ⊢ Eq (eHolderNorm r (HSMul.hSMul c f)) (HMul.hMul (↑(NNNorm.nnnorm c)) (eHolde …
    -/
    simp [hc]
    /-
      🎉 no goals
    -/
  /-
    case neg
    X : Type u_1
    Y : Type u_2
    inst✝⁴ : MetricSpace X
    inst✝³ : NormedAddCommGroup Y
    r : NNReal
    f : X → Y
    α : Type u_3
    inst✝² : NormedDivisionRing α
    inst✝¹ : Module α Y
    inst✝ : BoundedSMul α Y
    c : α
    hc : Not (Eq (NNNorm.nnnorm c) 0)
    ⊢ Eq (eHolderNorm r (HSMul.hSMul c f)) (HMul.hMul (↑(NNNorm.nnnorm c)) (eHolde …
  -/
  by_cases hf : MemHolder r f
    /-
      case pos
      X : Type u_1
      Y : Type u_2
      inst✝⁴ : MetricSpace X
      inst✝³ : NormedAddCommGroup Y
      r : NNReal
      f : X → Y
      α : Type u_3
      inst✝² : NormedDivisionRing α
      inst✝¹ : Module α Y
      inst✝ : BoundedSMul α Y
      c : α
      hc : Not (Eq (NNNorm.nnnorm c) 0)
      hf : MemHolder r f
      ⊢ Eq (eHolderNorm r (HSMul.hSMul c f)) (HMul.hMul (↑(NNNorm.nnnorm c)) (eHolde …
    -/
  · refine le_antisymm ((hf.holderWith.smul c).eHolderNorm_le.trans ?_) <| mul_le_of_le_div' ?_
      /-
        case pos.refine_1
        X : Type u_1
        Y : Type u_2
        inst✝⁴ : MetricSpace X
        inst✝³ : NormedAddCommGroup Y
        r : NNReal
        f : X → Y
        α : Type u_3
        inst✝² : NormedDivisionRing α
        inst✝¹ : Module α Y
        inst✝ : BoundedSMul α Y
        c : α
        hc : Not (Eq (NNNorm.nnnorm c) 0)
        hf : MemHolder r f
        ⊢ LE.le (↑(HMul.hMul (nnHolderNorm r f) (NNNorm.nnnorm c))) (HMul.hMul (↑(NNNo …
      -/
    · rw [coe_mul, hf.coe_nnHolderNorm_eq_eHolderNorm, mul_comm]
      /-
        🎉 no goals
      -/
      /-
        case pos.refine_2
        X : Type u_1
        Y : Type u_2
        inst✝⁴ : MetricSpace X
        inst✝³ : NormedAddCommGroup Y
        r : NNReal
        f : X → Y
        α : Type u_3
        inst✝² : NormedDivisionRing α
        inst✝¹ : Module α Y
        inst✝ : BoundedSMul α Y
        c : α
        hc : Not (Eq (NNNorm.nnnorm c) 0)
        hf : MemHolder r f
        ⊢ LE.le (eHolderNorm r f) (HDiv.hDiv (eHolderNorm r (HSMul.hSMul c f)) ↑(NNNor …
      -/
    · rw [← (hf.holderWith.smul c).memHolder.coe_nnHolderNorm_eq_eHolderNorm, ← coe_div hc]
      /-
        case pos.refine_2
        X : Type u_1
        Y : Type u_2
        inst✝⁴ : MetricSpace X
        inst✝³ : NormedAddCommGroup Y
        r : NNReal
        f : X → Y
        α : Type u_3
        inst✝² : NormedDivisionRing α
        inst✝¹ : Module α Y
        inst✝ : BoundedSMul α Y
        c : α
        hc : Not (Eq (NNNorm.nnnorm c) 0)
        hf : MemHolder r f
        ⊢ LE.le (eHolderNorm r f) ↑(HDiv.hDiv (nnHolderNorm r (HSMul.hSMul c f)) (NNNo …
      -/
      refine HolderWith.eHolderNorm_le fun x₁ x₂ => ?_
      rw [coe_div hc, ← ENNReal.mul_div_right_comm,
        ENNReal.le_div_iff_mul_le (Or.inl <| coe_ne_zero.2 hc) <| Or.inl coe_ne_top,
        mul_comm, ← smul_eq_mul, ← ENNReal.smul_def, ← edist_smul₀, ← Pi.smul_apply,
        ← Pi.smul_apply]
      /-
        case pos.refine_2
        X : Type u_1
        Y : Type u_2
        inst✝⁴ : MetricSpace X
        inst✝³ : NormedAddCommGroup Y
        r : NNReal
        f : X → Y
        α : Type u_3
        inst✝² : NormedDivisionRing α
        inst✝¹ : Module α Y
        inst✝ : BoundedSMul α Y
        c : α
        hc : Not (Eq (NNNorm.nnnorm c) 0)
        hf : MemHolder r f
        x₁ x₂ : X
        ⊢ LE.le (EDist.edist (HSMul.hSMul c f x₁) (HSMul.hSMul c f x₂)) (HMul.hMul (↑( …
      -/
      exact hf.smul.holderWith x₁ x₂
      /-
        🎉 no goals
      -/
    /-
      case neg
      X : Type u_1
      Y : Type u_2
      inst✝⁴ : MetricSpace X
      inst✝³ : NormedAddCommGroup Y
      r : NNReal
      f : X → Y
      α : Type u_3
      inst✝² : NormedDivisionRing α
      inst✝¹ : Module α Y
      inst✝ : BoundedSMul α Y
      c : α
      hc : Not (Eq (NNNorm.nnnorm c) 0)
      hf : Not (MemHolder r f)
      ⊢ Eq (eHolderNorm r (HSMul.hSMul c f)) (HMul.hMul (↑(NNNorm.nnnorm c)) (eHolde …
    -/
  · rw [← eHolderNorm_eq_top] at hf
    /-
      case neg
      X : Type u_1
      Y : Type u_2
      inst✝⁴ : MetricSpace X
      inst✝³ : NormedAddCommGroup Y
      r : NNReal
      f : X → Y
      α : Type u_3
      inst✝² : NormedDivisionRing α
      inst✝¹ : Module α Y
      inst✝ : BoundedSMul α Y
      c : α
      hc : Not (Eq (NNNorm.nnnorm c) 0)
      hf : Eq (eHolderNorm r f) Top.top
      ⊢ Eq (eHolderNorm r (HSMul.hSMul c f)) (HMul.hMul (↑(NNNorm.nnnorm c)) (eHolde …
    -/
    rw [hf, mul_top <| coe_ne_zero.2 hc, eHolderNorm_eq_top]
    /-
      case neg
      X : Type u_1
      Y : Type u_2
      inst✝⁴ : MetricSpace X
      inst✝³ : NormedAddCommGroup Y
      r : NNReal
      f : X → Y
      α : Type u_3
      inst✝² : NormedDivisionRing α
      inst✝¹ : Module α Y
      inst✝ : BoundedSMul α Y
      c : α
      hc : Not (Eq (NNNorm.nnnorm c) 0)
      hf : Eq (eHolderNorm r f) Top.top
      ⊢ Not (MemHolder r (HSMul.hSMul c f))
    -/
    rw [nnnorm_eq_zero] at hc
    /-
      case neg
      X : Type u_1
      Y : Type u_2
      inst✝⁴ : MetricSpace X
      inst✝³ : NormedAddCommGroup Y
      r : NNReal
      f : X → Y
      α : Type u_3
      inst✝² : NormedDivisionRing α
      inst✝¹ : Module α Y
      inst✝ : BoundedSMul α Y
      c : α
      hc : Not (Eq c 0)
      hf : Eq (eHolderNorm r f) Top.top
      ⊢ Not (MemHolder r (HSMul.hSMul c f))
    -/
    intro h
    /-
      case neg
      X : Type u_1
      Y : Type u_2
      inst✝⁴ : MetricSpace X
      inst✝³ : NormedAddCommGroup Y
      r : NNReal
      f : X → Y
      α : Type u_3
      inst✝² : NormedDivisionRing α
      inst✝¹ : Module α Y
      inst✝ : BoundedSMul α Y
      c : α
      hc : Not (Eq c 0)
      hf : Eq (eHolderNorm r f) Top.top
      h : MemHolder r (HSMul.hSMul c f)
      ⊢ False
    -/
    have := h.smul (c := c⁻¹)
    /-
      case neg
      X : Type u_1
      Y : Type u_2
      inst✝⁴ : MetricSpace X
      inst✝³ : NormedAddCommGroup Y
      r : NNReal
      f : X → Y
      α : Type u_3
      inst✝² : NormedDivisionRing α
      inst✝¹ : Module α Y
      inst✝ : BoundedSMul α Y
      c : α
      hc : Not (Eq c 0)
      hf : Eq (eHolderNorm r f) Top.top
      h : MemHolder r (HSMul.hSMul c f)
      this : MemHolder r (HSMul.hSMul (Inv.inv c) (HSMul.hSMul c f))
      ⊢ False
    -/
    rw [inv_smul_smul₀ hc] at this
    /-
      case neg
      X : Type u_1
      Y : Type u_2
      inst✝⁴ : MetricSpace X
      inst✝³ : NormedAddCommGroup Y
      r : NNReal
      f : X → Y
      α : Type u_3
      inst✝² : NormedDivisionRing α
      inst✝¹ : Module α Y
      inst✝ : BoundedSMul α Y
      c : α
      hc : Not (Eq c 0)
      hf : Eq (eHolderNorm r f) Top.top
      h : MemHolder r (HSMul.hSMul c f)
      this : MemHolder r f
      ⊢ False
    -/
    exact this.eHolderNorm_lt_top.ne hf
    /-
      🎉 no goals
    -/


lemma MemHolder.nnHolderNorm_smul {α} [NormedDivisionRing α] [Module α Y] [BoundedSMul α Y]
    (hf : MemHolder r f) (c : α) :
    nnHolderNorm r (c • f) = ‖c‖₊ * nnHolderNorm r f := by
  rw [← ENNReal.coe_inj, coe_mul, hf.coe_nnHolderNorm_eq_eHolderNorm,
    hf.smul.coe_nnHolderNorm_eq_eHolderNorm, eHolderNorm_smul]


lemma eHolderNorm_nsmul [Module ℝ Y] [BoundedSMul ℝ Y] (n : ℕ) :
    eHolderNorm r (n • f) = n • eHolderNorm r f := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝³ : MetricSpace X
    inst✝² : NormedAddCommGroup Y
    r : NNReal
    f : X → Y
    inst✝¹ : Module Real Y
    inst✝ : BoundedSMul Real Y
    n : Nat
    ⊢ Eq (eHolderNorm r (HSMul.hSMul n f)) (HSMul.hSMul n (eHolderNorm r f))
  -/
  simp [← Nat.cast_smul_eq_nsmul (R := ℝ), eHolderNorm_smul]
  /-
    🎉 no goals
  -/


lemma MemHolder.nnHolderNorm_nsmul [Module ℝ Y] [BoundedSMul ℝ Y] (n : ℕ) (hf : MemHolder r f) :
    nnHolderNorm r (n • f) = n • nnHolderNorm r f := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝³ : MetricSpace X
    inst✝² : NormedAddCommGroup Y
    r : NNReal
    f : X → Y
    inst✝¹ : Module Real Y
    inst✝ : BoundedSMul Real Y
    n : Nat
    hf : MemHolder r f
    ⊢ Eq (nnHolderNorm r (HSMul.hSMul n f)) (HSMul.hSMul n (nnHolderNorm r f))
  -/
  simp [← Nat.cast_smul_eq_nsmul (R := ℝ), hf.nnHolderNorm_smul]
  /-
    🎉 no goals
  -/


