/-- If `f : 𝕜 → E` is bounded in a punctured neighborhood of `a`, then `f(x) = o((x - a)⁻¹)` as
`x → a`, `x ≠ a`. -/
theorem Filter.IsBoundedUnder.isLittleO_sub_self_inv {𝕜 E : Type*} [NormedField 𝕜] [Norm E] {a : 𝕜}
    {f : 𝕜 → E} (h : IsBoundedUnder (· ≤ ·) (𝓝[≠] a) (norm ∘ f)) :
    f =o[𝓝[≠] a] fun x => (x - a)⁻¹ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝¹ : NormedField 𝕜
    inst✝ : Norm E
    a : 𝕜
    f : 𝕜 → E
    h : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) (nhdsWithin a (HasCompl.c …
    ⊢ Asymptotics.IsLittleO (nhdsWithin a (HasCompl.compl (Singleton.singleton a)) …
  -/
  refine (h.isBigO_const (one_ne_zero' ℝ)).trans_isLittleO (isLittleO_const_left.2 <| Or.inr ?_)
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝¹ : NormedField 𝕜
    inst✝ : Norm E
    a : 𝕜
    f : 𝕜 → E
    h : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) (nhdsWithin a (HasCompl.c …
    ⊢ Filter.Tendsto (Function.comp Norm.norm fun x => Inv.inv (HSub.hSub x a)) (n …
  -/
  simp only [Function.comp_def, norm_inv]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝¹ : NormedField 𝕜
    inst✝ : Norm E
    a : 𝕜
    f : 𝕜 → E
    h : Filter.IsBoundedUnder (fun x1 x2 => LE.le x1 x2) (nhdsWithin a (HasCompl.c …
    ⊢ Filter.Tendsto (fun x => Inv.inv (Norm.norm (HSub.hSub x a))) (nhdsWithin a  …
  -/
  exact (tendsto_norm_sub_self_nhdsNE a).inv_tendsto_nhdsGT_zero
  /-
    🎉 no goals
  -/


theorem pow_div_pow_eventuallyEq_atTop {p q : ℕ} :
    (fun x : 𝕜 => x ^ p / x ^ q) =ᶠ[atTop] fun x => x ^ ((p : ℤ) - q) := by
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    p q : Nat
    ⊢ Filter.atTop.EventuallyEq (fun x => HDiv.hDiv (HPow.hPow x p) (HPow.hPow x q …
  -/
  apply (eventually_gt_atTop (0 : 𝕜)).mono fun x hx => _
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    p q : Nat
    ⊢ ∀ (x : 𝕜), LT.lt 0 x → Eq ((fun x => HDiv.hDiv (HPow.hPow x p) (HPow.hPow x  …
  -/
  intro x hx
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    p q : Nat
    x : 𝕜
    hx : LT.lt 0 x
    ⊢ Eq ((fun x => HDiv.hDiv (HPow.hPow x p) (HPow.hPow x q)) x) ((fun x => HPow. …
  -/
  simp [zpow_sub₀ hx.ne']
  /-
    🎉 no goals
  -/


theorem pow_div_pow_eventuallyEq_atBot {p q : ℕ} :
    (fun x : 𝕜 => x ^ p / x ^ q) =ᶠ[atBot] fun x => x ^ ((p : ℤ) - q) := by
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    p q : Nat
    ⊢ Filter.atBot.EventuallyEq (fun x => HDiv.hDiv (HPow.hPow x p) (HPow.hPow x q …
  -/
  apply (eventually_lt_atBot (0 : 𝕜)).mono fun x hx => _
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    p q : Nat
    ⊢ ∀ (x : 𝕜), LT.lt x 0 → Eq ((fun x => HDiv.hDiv (HPow.hPow x p) (HPow.hPow x  …
  -/
  intro x hx
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    p q : Nat
    x : 𝕜
    hx : LT.lt x 0
    ⊢ Eq ((fun x => HDiv.hDiv (HPow.hPow x p) (HPow.hPow x q)) x) ((fun x => HPow. …
  -/
  simp [zpow_sub₀ hx.ne]
  /-
    🎉 no goals
  -/


theorem tendsto_pow_div_pow_atTop_atTop {p q : ℕ} (hpq : q < p) :
    Tendsto (fun x : 𝕜 => x ^ p / x ^ q) atTop atTop := by
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    p q : Nat
    hpq : LT.lt q p
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (HPow.hPow x p) (HPow.hPow x q)) Filter.a …
  -/
  rw [tendsto_congr' pow_div_pow_eventuallyEq_atTop]
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    p q : Nat
    hpq : LT.lt q p
    ⊢ Filter.Tendsto (fun x => HPow.hPow x (HSub.hSub ↑p ↑q)) Filter.atTop Filter. …
  -/
  apply tendsto_zpow_atTop_atTop
  /-
    case hn
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    p q : Nat
    hpq : LT.lt q p
    ⊢ LT.lt 0 (HSub.hSub ↑p ↑q)
  -/
  omega
  /-
    🎉 no goals
  -/


theorem tendsto_pow_div_pow_atTop_zero [TopologicalSpace 𝕜] [OrderTopology 𝕜] {p q : ℕ}
    (hpq : p < q) : Tendsto (fun x : 𝕜 => x ^ p / x ^ q) atTop (𝓝 0) := by
  /-
    𝕜 : Type u_1
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    p q : Nat
    hpq : LT.lt p q
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (HPow.hPow x p) (HPow.hPow x q)) Filter.a …
  -/
  rw [tendsto_congr' pow_div_pow_eventuallyEq_atTop]
  /-
    𝕜 : Type u_1
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    p q : Nat
    hpq : LT.lt p q
    ⊢ Filter.Tendsto (fun x => HPow.hPow x (HSub.hSub ↑p ↑q)) Filter.atTop (nhds 0)
  -/
  apply tendsto_zpow_atTop_zero
  /-
    case hn
    𝕜 : Type u_1
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    p q : Nat
    hpq : LT.lt p q
    ⊢ LT.lt (HSub.hSub ↑p ↑q) 0
  -/
  omega
  /-
    🎉 no goals
  -/


theorem Asymptotics.isLittleO_pow_pow_atTop_of_lt [OrderTopology 𝕜] {p q : ℕ} (hpq : p < q) :
    (fun x : 𝕜 => x ^ p) =o[atTop] fun x => x ^ q := by
  /-
    𝕜 : Type u_1
    inst✝¹ : NormedLinearOrderedField 𝕜
    inst✝ : OrderTopology 𝕜
    p q : Nat
    hpq : LT.lt p q
    ⊢ Asymptotics.IsLittleO Filter.atTop (fun x => HPow.hPow x p) fun x => HPow.hP …
  -/
  refine (isLittleO_iff_tendsto' ?_).mpr (tendsto_pow_div_pow_atTop_zero hpq)
  /-
    𝕜 : Type u_1
    inst✝¹ : NormedLinearOrderedField 𝕜
    inst✝ : OrderTopology 𝕜
    p q : Nat
    hpq : LT.lt p q
    ⊢ Filter.Eventually (fun x => Eq (HPow.hPow x q) 0 → Eq (HPow.hPow x p) 0) Fil …
  -/
  exact (eventually_gt_atTop 0).mono fun x hx hxq => (pow_ne_zero q hx.ne' hxq).elim
  /-
    🎉 no goals
  -/


theorem Asymptotics.IsBigO.trans_tendsto_norm_atTop {α : Type*} {u v : α → 𝕜} {l : Filter α}
    (huv : u =O[l] v) (hu : Tendsto (fun x => ‖u x‖) l atTop) :
    Tendsto (fun x => ‖v x‖) l atTop := by
  /-
    𝕜 : Type u_1
    inst✝ : NormedLinearOrderedField 𝕜
    α : Type u_2
    u v : α → 𝕜
    l : Filter α
    huv : Asymptotics.IsBigO l u v
    hu : Filter.Tendsto (fun x => Norm.norm (u x)) l Filter.atTop
    ⊢ Filter.Tendsto (fun x => Norm.norm (v x)) l Filter.atTop
  -/
  rcases huv.exists_pos with ⟨c, hc, hcuv⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝ : NormedLinearOrderedField 𝕜
    α : Type u_2
    u v : α → 𝕜
    l : Filter α
    huv : Asymptotics.IsBigO l u v
    hu : Filter.Tendsto (fun x => Norm.norm (u x)) l Filter.atTop
    c : Real
    hc : GT.gt c 0
    hcuv : Asymptotics.IsBigOWith c l u v
    ⊢ Filter.Tendsto (fun x => Norm.norm (v x)) l Filter.atTop
  -/
  rw [IsBigOWith] at hcuv
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝ : NormedLinearOrderedField 𝕜
    α : Type u_2
    u v : α → 𝕜
    l : Filter α
    huv : Asymptotics.IsBigO l u v
    hu : Filter.Tendsto (fun x => Norm.norm (u x)) l Filter.atTop
    c : Real
    hc : GT.gt c 0
    hcuv : Filter.Eventually (fun x => LE.le (Norm.norm (u x)) (HMul.hMul c (Norm. …
    ⊢ Filter.Tendsto (fun x => Norm.norm (v x)) l Filter.atTop
  -/
  convert Tendsto.atTop_div_const hc (tendsto_atTop_mono' l hcuv hu)
  /-
    case h.e'_3.h
    𝕜 : Type u_1
    inst✝ : NormedLinearOrderedField 𝕜
    α : Type u_2
    u v : α → 𝕜
    l : Filter α
    huv : Asymptotics.IsBigO l u v
    hu : Filter.Tendsto (fun x => Norm.norm (u x)) l Filter.atTop
    c : Real
    hc : GT.gt c 0
    hcuv : Filter.Eventually (fun x => LE.le (Norm.norm (u x)) (HMul.hMul c (Norm. …
    x✝ : α
    ⊢ Eq (Norm.norm (v x✝)) (HDiv.hDiv (HMul.hMul c (Norm.norm (v x✝))) c)
  -/
  rw [mul_div_cancel_left₀ _ hc.ne.symm]
  /-
    🎉 no goals
  -/


theorem Asymptotics.IsLittleO.sum_range {α : Type*} [NormedAddCommGroup α] {f : ℕ → α} {g : ℕ → ℝ}
    (h : f =o[atTop] g) (hg : 0 ≤ g) (h'g : Tendsto (fun n => ∑ i ∈ range n, g i) atTop atTop) :
    (fun n => ∑ i ∈ range n, f i) =o[atTop] fun n => ∑ i ∈ range n, g i := by
  /-
    α : Type u_1
    inst✝ : NormedAddCommGroup α
    f : Nat → α
    g : Nat → Real
    h : Asymptotics.IsLittleO Filter.atTop f g
    hg : LE.le 0 g
    h'g : Filter.Tendsto (fun n => (Finset.range n).sum fun i => g i) Filter.atTop …
    ⊢ Asymptotics.IsLittleO Filter.atTop (fun n => (Finset.range n).sum fun i => f …
  -/
  have A : ∀ i, ‖g i‖ = g i := fun i => Real.norm_of_nonneg (hg i)
  have B : ∀ n, ‖∑ i ∈ range n, g i‖ = ∑ i ∈ range n, g i := fun n => by
    rwa [Real.norm_eq_abs, abs_sum_of_nonneg']
  /-
    α : Type u_1
    inst✝ : NormedAddCommGroup α
    f : Nat → α
    g : Nat → Real
    h : Asymptotics.IsLittleO Filter.atTop f g
    hg : LE.le 0 g
    h'g : Filter.Tendsto (fun n => (Finset.range n).sum fun i => g i) Filter.atTop …
    A : ∀ (i : Nat), Eq (Norm.norm (g i)) (g i)
    B : ∀ (n : Nat), Eq (Norm.norm ((Finset.range n).sum fun i => g i)) ((Finset.r …
    ⊢ Asymptotics.IsLittleO Filter.atTop (fun n => (Finset.range n).sum fun i => f …
  -/
  apply isLittleO_iff.2 fun ε εpos => _
  /-
    α : Type u_1
    inst✝ : NormedAddCommGroup α
    f : Nat → α
    g : Nat → Real
    h : Asymptotics.IsLittleO Filter.atTop f g
    hg : LE.le 0 g
    h'g : Filter.Tendsto (fun n => (Finset.range n).sum fun i => g i) Filter.atTop …
    A : ∀ (i : Nat), Eq (Norm.norm (g i)) (g i)
    B : ∀ (n : Nat), Eq (Norm.norm ((Finset.range n).sum fun i => g i)) ((Finset.r …
    ⊢ ∀ (ε : Real), LT.lt 0 ε → Filter.Eventually (fun x => LE.le (Norm.norm ((Fin …
  -/
  intro ε εpos
  obtain ⟨N, hN⟩ : ∃ N : ℕ, ∀ b : ℕ, N ≤ b → ‖f b‖ ≤ ε / 2 * g b := by
    simpa only [A, eventually_atTop] using isLittleO_iff.mp h (half_pos εpos)
  have : (fun _ : ℕ => ∑ i ∈ range N, f i) =o[atTop] fun n : ℕ => ∑ i ∈ range n, g i := by
    apply isLittleO_const_left.2
    exact Or.inr (h'g.congr fun n => (B n).symm)
  /-
    case intro
    α : Type u_1
    inst✝ : NormedAddCommGroup α
    f : Nat → α
    g : Nat → Real
    h : Asymptotics.IsLittleO Filter.atTop f g
    hg : LE.le 0 g
    h'g : Filter.Tendsto (fun n => (Finset.range n).sum fun i => g i) Filter.atTop …
    A : ∀ (i : Nat), Eq (Norm.norm (g i)) (g i)
    B : ∀ (n : Nat), Eq (Norm.norm ((Finset.range n).sum fun i => g i)) ((Finset.r …
    ε : Real
    εpos : LT.lt 0 ε
    N : Nat
    hN : ∀ (b : Nat), LE.le N b → LE.le (Norm.norm (f b)) (HMul.hMul (HDiv.hDiv ε  …
    this : Asymptotics.IsLittleO Filter.atTop (fun x => (Finset.range N).sum fun i …
    ⊢ Filter.Eventually (fun x => LE.le (Norm.norm ((Finset.range x).sum fun i =>  …
  -/
  filter_upwards [isLittleO_iff.1 this (half_pos εpos), Ici_mem_atTop N] with n hn Nn
  calc
    ‖∑ i ∈ range n, f i‖ = ‖(∑ i ∈ range N, f i) + ∑ i ∈ Ico N n, f i‖ := by
      rw [sum_range_add_sum_Ico _ Nn]
    _ ≤ ‖∑ i ∈ range N, f i‖ + ‖∑ i ∈ Ico N n, f i‖ := norm_add_le _ _
    _ ≤ ‖∑ i ∈ range N, f i‖ + ∑ i ∈ Ico N n, ε / 2 * g i :=
      (add_le_add le_rfl (norm_sum_le_of_le _ fun i hi => hN _ (mem_Ico.1 hi).1))
    _ ≤ ‖∑ i ∈ range N, f i‖ + ∑ i ∈ range n, ε / 2 * g i := by
      gcongr
      · exact fun i _ _ ↦ mul_nonneg (half_pos εpos).le (hg i)
      · rw [range_eq_Ico]
        exact Ico_subset_Ico (zero_le _) le_rfl
    _ ≤ ε / 2 * ‖∑ i ∈ range n, g i‖ + ε / 2 * ∑ i ∈ range n, g i := by rw [← mul_sum]; gcongr
    _ = ε * ‖∑ i ∈ range n, g i‖ := by
      simp only [B]
      ring


theorem Asymptotics.isLittleO_sum_range_of_tendsto_zero {α : Type*} [NormedAddCommGroup α]
    {f : ℕ → α} (h : Tendsto f atTop (𝓝 0)) :
    (fun n => ∑ i ∈ range n, f i) =o[atTop] fun n => (n : ℝ) := by
  /-
    α : Type u_1
    inst✝ : NormedAddCommGroup α
    f : Nat → α
    h : Filter.Tendsto f Filter.atTop (nhds 0)
    ⊢ Asymptotics.IsLittleO Filter.atTop (fun n => (Finset.range n).sum fun i => f …
  -/
  have := ((isLittleO_one_iff ℝ).2 h).sum_range fun i => zero_le_one
  /-
    α : Type u_1
    inst✝ : NormedAddCommGroup α
    f : Nat → α
    h : Filter.Tendsto f Filter.atTop (nhds 0)
    this : Filter.Tendsto (fun n => (Finset.range n).sum fun i => 1) Filter.atTop  …
    ⊢ Asymptotics.IsLittleO Filter.atTop (fun n => (Finset.range n).sum fun i => f …
  -/
  simp only [sum_const, card_range, Nat.smul_one_eq_cast] at this
  /-
    α : Type u_1
    inst✝ : NormedAddCommGroup α
    f : Nat → α
    h : Filter.Tendsto f Filter.atTop (nhds 0)
    this : Filter.Tendsto (fun n => ↑n) Filter.atTop Filter.atTop → Asymptotics.Is …
    ⊢ Asymptotics.IsLittleO Filter.atTop (fun n => (Finset.range n).sum fun i => f …
  -/
  exact this tendsto_natCast_atTop_atTop
  /-
    🎉 no goals
  -/


/-- The Cesaro average of a converging sequence converges to the same limit. -/
theorem Filter.Tendsto.cesaro_smul {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] {u : ℕ → E}
    {l : E} (h : Tendsto u atTop (𝓝 l)) :
    Tendsto (fun n : ℕ => (n⁻¹ : ℝ) • ∑ i ∈ range n, u i) atTop (𝓝 l) := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    u : Nat → E
    l : E
    h : Filter.Tendsto u Filter.atTop (nhds l)
    ⊢ Filter.Tendsto (fun n => HSMul.hSMul (Inv.inv ↑n) ((Finset.range n).sum fun  …
  -/
  rw [← tendsto_sub_nhds_zero_iff, ← isLittleO_one_iff ℝ]
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    u : Nat → E
    l : E
    h : Filter.Tendsto u Filter.atTop (nhds l)
    ⊢ Asymptotics.IsLittleO Filter.atTop (fun x => HSub.hSub (HSMul.hSMul (Inv.inv …
  -/
  have := Asymptotics.isLittleO_sum_range_of_tendsto_zero (tendsto_sub_nhds_zero_iff.2 h)
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    u : Nat → E
    l : E
    h : Filter.Tendsto u Filter.atTop (nhds l)
    this : Asymptotics.IsLittleO Filter.atTop (fun n => (Finset.range n).sum fun i …
    ⊢ Asymptotics.IsLittleO Filter.atTop (fun x => HSub.hSub (HSMul.hSMul (Inv.inv …
  -/
  apply ((isBigO_refl (fun n : ℕ => (n : ℝ)⁻¹) atTop).smul_isLittleO this).congr' _ _
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      u : Nat → E
      l : E
      h : Filter.Tendsto u Filter.atTop (nhds l)
      this : Asymptotics.IsLittleO Filter.atTop (fun n => (Finset.range n).sum fun i …
      ⊢ Filter.atTop.EventuallyEq (fun x => HSMul.hSMul (Inv.inv ↑x) ((Finset.range  …
    -/
  · filter_upwards [Ici_mem_atTop 1] with n npos
    /-
      case h
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      u : Nat → E
      l : E
      h : Filter.Tendsto u Filter.atTop (nhds l)
      this : Asymptotics.IsLittleO Filter.atTop (fun n => (Finset.range n).sum fun i …
      n : Nat
      npos : Membership.mem (Set.Ici 1) n
      ⊢ Eq (HSMul.hSMul (Inv.inv ↑n) ((Finset.range n).sum fun i => HSub.hSub (u i)  …
    -/
    have nposℝ : (0 : ℝ) < n := Nat.cast_pos.2 npos
    /-
      case h
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      u : Nat → E
      l : E
      h : Filter.Tendsto u Filter.atTop (nhds l)
      this : Asymptotics.IsLittleO Filter.atTop (fun n => (Finset.range n).sum fun i …
      n : Nat
      npos : Membership.mem (Set.Ici 1) n
      nposℝ : LT.lt 0 ↑n
      ⊢ Eq (HSMul.hSMul (Inv.inv ↑n) ((Finset.range n).sum fun i => HSub.hSub (u i)  …
    -/
    simp only [smul_sub, sum_sub_distrib, sum_const, card_range, sub_right_inj]
    /-
      case h
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      u : Nat → E
      l : E
      h : Filter.Tendsto u Filter.atTop (nhds l)
      this : Asymptotics.IsLittleO Filter.atTop (fun n => (Finset.range n).sum fun i …
      n : Nat
      npos : Membership.mem (Set.Ici 1) n
      nposℝ : LT.lt 0 ↑n
      ⊢ Eq (HSMul.hSMul (Inv.inv ↑n) (HSMul.hSMul n l)) l
    -/
    rw [← Nat.cast_smul_eq_nsmul ℝ, smul_smul, inv_mul_cancel₀ nposℝ.ne', one_smul]
    /-
      🎉 no goals
    -/
    /-
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      u : Nat → E
      l : E
      h : Filter.Tendsto u Filter.atTop (nhds l)
      this : Asymptotics.IsLittleO Filter.atTop (fun n => (Finset.range n).sum fun i …
      ⊢ Filter.atTop.EventuallyEq (fun x => HSMul.hSMul (Inv.inv ↑x) ↑x) fun _x => 1
    -/
  · filter_upwards [Ici_mem_atTop 1] with n npos
    /-
      case h
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      u : Nat → E
      l : E
      h : Filter.Tendsto u Filter.atTop (nhds l)
      this : Asymptotics.IsLittleO Filter.atTop (fun n => (Finset.range n).sum fun i …
      n : Nat
      npos : Membership.mem (Set.Ici 1) n
      ⊢ Eq (HSMul.hSMul (Inv.inv ↑n) ↑n) 1
    -/
    have nposℝ : (0 : ℝ) < n := Nat.cast_pos.2 npos
    /-
      case h
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      u : Nat → E
      l : E
      h : Filter.Tendsto u Filter.atTop (nhds l)
      this : Asymptotics.IsLittleO Filter.atTop (fun n => (Finset.range n).sum fun i …
      n : Nat
      npos : Membership.mem (Set.Ici 1) n
      nposℝ : LT.lt 0 ↑n
      ⊢ Eq (HSMul.hSMul (Inv.inv ↑n) ↑n) 1
    -/
    rw [Algebra.id.smul_eq_mul, inv_mul_cancel₀ nposℝ.ne']
    /-
      🎉 no goals
    -/


/-- The Cesaro average of a converging sequence converges to the same limit. -/
theorem Filter.Tendsto.cesaro {u : ℕ → ℝ} {l : ℝ} (h : Tendsto u atTop (𝓝 l)) :
    Tendsto (fun n : ℕ => (n⁻¹ : ℝ) * ∑ i ∈ range n, u i) atTop (𝓝 l) :=
  h.cesaro_smul


