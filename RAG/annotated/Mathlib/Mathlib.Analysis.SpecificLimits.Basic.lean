theorem tendsto_inverse_atTop_nhds_zero_nat : Tendsto (fun n : ℕ ↦ (n : ℝ)⁻¹) atTop (𝓝 0) :=
  tendsto_inv_atTop_zero.comp tendsto_natCast_atTop_atTop


theorem tendsto_const_div_atTop_nhds_zero_nat (C : ℝ) :
    Tendsto (fun n : ℕ ↦ C / n) atTop (𝓝 0) := by
  /-
    C : Real
    ⊢ Filter.Tendsto (fun n => HDiv.hDiv C ↑n) Filter.atTop (nhds 0)
  -/
  simpa only [mul_zero] using tendsto_const_nhds.mul tendsto_inverse_atTop_nhds_zero_nat
  /-
    🎉 no goals
  -/


theorem tendsto_one_div_atTop_nhds_zero_nat : Tendsto (fun n : ℕ ↦ 1/(n : ℝ)) atTop (𝓝 0) :=
  tendsto_const_div_atTop_nhds_zero_nat 1


theorem NNReal.tendsto_inverse_atTop_nhds_zero_nat :
    Tendsto (fun n : ℕ ↦ (n : ℝ≥0)⁻¹) atTop (𝓝 0) := by
  /-
    ⊢ Filter.Tendsto (fun n => Inv.inv ↑n) Filter.atTop (nhds 0)
  -/
  rw [← NNReal.tendsto_coe]
  /-
    ⊢ Filter.Tendsto (fun a => ↑(Inv.inv ↑a)) Filter.atTop (nhds ↑0)
  -/
  exact _root_.tendsto_inverse_atTop_nhds_zero_nat
  /-
    🎉 no goals
  -/


theorem NNReal.tendsto_const_div_atTop_nhds_zero_nat (C : ℝ≥0) :
    Tendsto (fun n : ℕ ↦ C / n) atTop (𝓝 0) := by
  /-
    C : NNReal
    ⊢ Filter.Tendsto (fun n => HDiv.hDiv C ↑n) Filter.atTop (nhds 0)
  -/
  simpa using tendsto_const_nhds.mul NNReal.tendsto_inverse_atTop_nhds_zero_nat
  /-
    🎉 no goals
  -/


theorem EReal.tendsto_const_div_atTop_nhds_zero_nat {C : EReal} (h : C ≠ ⊥) (h' : C ≠ ⊤) :
    Tendsto (fun n : ℕ ↦ C / n) atTop (𝓝 0) := by
  have : (fun n : ℕ ↦ C / n) = fun n : ℕ ↦ ((C.toReal / n : ℝ) : EReal) := by
    ext n
    nth_rw 1 [← coe_toReal h' h, ← coe_coe_eq_natCast n, ← coe_div C.toReal n]
  /-
    C : EReal
    h : Ne C Bot.bot
    h' : Ne C Top.top
    this : Eq (fun n => HDiv.hDiv C ↑n) fun n => ↑(HDiv.hDiv C.toReal ↑n)
    ⊢ Filter.Tendsto (fun n => HDiv.hDiv C ↑n) Filter.atTop (nhds 0)
  -/
  rw [this, ← coe_zero, tendsto_coe]
  /-
    C : EReal
    h : Ne C Bot.bot
    h' : Ne C Top.top
    this : Eq (fun n => HDiv.hDiv C ↑n) fun n => ↑(HDiv.hDiv C.toReal ↑n)
    ⊢ Filter.Tendsto (fun n => HDiv.hDiv C.toReal ↑n) Filter.atTop (nhds 0)
  -/
  exact _root_.tendsto_const_div_atTop_nhds_zero_nat C.toReal
  /-
    🎉 no goals
  -/


theorem tendsto_one_div_add_atTop_nhds_zero_nat :
    Tendsto (fun n : ℕ ↦ 1 / ((n : ℝ) + 1)) atTop (𝓝 0) :=
                                                                   /-
                                                                     this : Filter.Tendsto (fun n => HDiv.hDiv 1 ↑(HAdd.hAdd n 1)) Filter.atTop (nh …
                                                                     ⊢ Filter.Tendsto (fun n => HDiv.hDiv 1 (HAdd.hAdd (↑n) 1)) Filter.atTop (nhds 0)
                                                                   -/
  suffices Tendsto (fun n : ℕ ↦ 1 / (↑(n + 1) : ℝ)) atTop (𝓝 0) by simpa
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
  (tendsto_add_atTop_iff_nat 1).2 (_root_.tendsto_const_div_atTop_nhds_zero_nat 1)


theorem NNReal.tendsto_algebraMap_inverse_atTop_nhds_zero_nat (𝕜 : Type*) [Semiring 𝕜]
    [Algebra ℝ≥0 𝕜] [TopologicalSpace 𝕜] [ContinuousSMul ℝ≥0 𝕜] :
    Tendsto (algebraMap ℝ≥0 𝕜 ∘ fun n : ℕ ↦ (n : ℝ≥0)⁻¹) atTop (𝓝 0) := by
  convert (continuous_algebraMap ℝ≥0 𝕜).continuousAt.tendsto.comp
    tendsto_inverse_atTop_nhds_zero_nat
  /-
    case h.e'_5.h.e'_3
    𝕜 : Type u_4
    inst✝³ : Semiring 𝕜
    inst✝² : Algebra NNReal 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : ContinuousSMul NNReal 𝕜
    ⊢ Eq 0 ((algebraMap NNReal 𝕜) 0)
  -/
  rw [map_zero]
  /-
    🎉 no goals
  -/


theorem tendsto_algebraMap_inverse_atTop_nhds_zero_nat (𝕜 : Type*) [Semiring 𝕜] [Algebra ℝ 𝕜]
    [TopologicalSpace 𝕜] [ContinuousSMul ℝ 𝕜] :
    Tendsto (algebraMap ℝ 𝕜 ∘ fun n : ℕ ↦ (n : ℝ)⁻¹) atTop (𝓝 0) :=
  NNReal.tendsto_algebraMap_inverse_atTop_nhds_zero_nat 𝕜


/-- The limit of `n / (n + x)` is 1, for any constant `x` (valid in `ℝ` or any topological division
algebra over `ℝ`, e.g., `ℂ`).

TODO: introduce a typeclass saying that `1 / n` tends to 0 at top, making it possible to get this
statement simultaneously on `ℚ`, `ℝ` and `ℂ`. -/
theorem tendsto_natCast_div_add_atTop {𝕜 : Type*} [DivisionRing 𝕜] [TopologicalSpace 𝕜]
    [CharZero 𝕜] [Algebra ℝ 𝕜] [ContinuousSMul ℝ 𝕜] [TopologicalDivisionRing 𝕜] (x : 𝕜) :
    Tendsto (fun n : ℕ ↦ (n : 𝕜) / (n + x)) atTop (𝓝 1) := by
  /-
    𝕜 : Type u_4
    inst✝⁵ : DivisionRing 𝕜
    inst✝⁴ : TopologicalSpace 𝕜
    inst✝³ : CharZero 𝕜
    inst✝² : Algebra Real 𝕜
    inst✝¹ : ContinuousSMul Real 𝕜
    inst✝ : TopologicalDivisionRing 𝕜
    x : 𝕜
    ⊢ Filter.Tendsto (fun n => HDiv.hDiv (↑n) (HAdd.hAdd (↑n) x)) Filter.atTop (nh …
  -/
  convert Tendsto.congr' ((eventually_ne_atTop 0).mp (Eventually.of_forall fun n hn ↦ _)) _
    /-
      case convert_2
      𝕜 : Type u_4
      inst✝⁵ : DivisionRing 𝕜
      inst✝⁴ : TopologicalSpace 𝕜
      inst✝³ : CharZero 𝕜
      inst✝² : Algebra Real 𝕜
      inst✝¹ : ContinuousSMul Real 𝕜
      inst✝ : TopologicalDivisionRing 𝕜
      x : 𝕜
      ⊢ Nat → 𝕜
    -/
  · exact fun n : ℕ ↦ 1 / (1 + x / n)
    /-
      🎉 no goals
    -/
    /-
      case convert_5
      𝕜 : Type u_4
      inst✝⁵ : DivisionRing 𝕜
      inst✝⁴ : TopologicalSpace 𝕜
      inst✝³ : CharZero 𝕜
      inst✝² : Algebra Real 𝕜
      inst✝¹ : ContinuousSMul Real 𝕜
      inst✝ : TopologicalDivisionRing 𝕜
      x : 𝕜
      n : Nat
      hn : Ne n 0
      ⊢ Eq (HDiv.hDiv 1 (HAdd.hAdd 1 (HDiv.hDiv x ↑n))) (HDiv.hDiv (↑n) (HAdd.hAdd ( …
    -/
  · field_simp [Nat.cast_ne_zero.mpr hn]
    /-
      🎉 no goals
    -/
  · have : 𝓝 (1 : 𝕜) = 𝓝 (1 / (1 + x * (0 : 𝕜))) := by
      rw [mul_zero, add_zero, div_one]
    /-
      case convert_6
      𝕜 : Type u_4
      inst✝⁵ : DivisionRing 𝕜
      inst✝⁴ : TopologicalSpace 𝕜
      inst✝³ : CharZero 𝕜
      inst✝² : Algebra Real 𝕜
      inst✝¹ : ContinuousSMul Real 𝕜
      inst✝ : TopologicalDivisionRing 𝕜
      x : 𝕜
      this : Eq (nhds 1) (nhds (HDiv.hDiv 1 (HAdd.hAdd 1 (HMul.hMul x 0))))
      ⊢ Filter.Tendsto (fun n => HDiv.hDiv 1 (HAdd.hAdd 1 (HDiv.hDiv x ↑n))) Filter. …
    -/
    rw [this]
    /-
      case convert_6
      𝕜 : Type u_4
      inst✝⁵ : DivisionRing 𝕜
      inst✝⁴ : TopologicalSpace 𝕜
      inst✝³ : CharZero 𝕜
      inst✝² : Algebra Real 𝕜
      inst✝¹ : ContinuousSMul Real 𝕜
      inst✝ : TopologicalDivisionRing 𝕜
      x : 𝕜
      this : Eq (nhds 1) (nhds (HDiv.hDiv 1 (HAdd.hAdd 1 (HMul.hMul x 0))))
      ⊢ Filter.Tendsto (fun n => HDiv.hDiv 1 (HAdd.hAdd 1 (HDiv.hDiv x ↑n))) Filter. …
    -/
    refine tendsto_const_nhds.div (tendsto_const_nhds.add ?_) (by simp)
    /-
      case convert_6
      𝕜 : Type u_4
      inst✝⁵ : DivisionRing 𝕜
      inst✝⁴ : TopologicalSpace 𝕜
      inst✝³ : CharZero 𝕜
      inst✝² : Algebra Real 𝕜
      inst✝¹ : ContinuousSMul Real 𝕜
      inst✝ : TopologicalDivisionRing 𝕜
      x : 𝕜
      this : Eq (nhds 1) (nhds (HDiv.hDiv 1 (HAdd.hAdd 1 (HMul.hMul x 0))))
      ⊢ Filter.Tendsto (fun n => HDiv.hDiv x ↑n) Filter.atTop (nhds (HMul.hMul x 0))
    -/
    simp_rw [div_eq_mul_inv]
    /-
      case convert_6
      𝕜 : Type u_4
      inst✝⁵ : DivisionRing 𝕜
      inst✝⁴ : TopologicalSpace 𝕜
      inst✝³ : CharZero 𝕜
      inst✝² : Algebra Real 𝕜
      inst✝¹ : ContinuousSMul Real 𝕜
      inst✝ : TopologicalDivisionRing 𝕜
      x : 𝕜
      this : Eq (nhds 1) (nhds (HDiv.hDiv 1 (HAdd.hAdd 1 (HMul.hMul x 0))))
      ⊢ Filter.Tendsto (fun n => HMul.hMul x (Inv.inv ↑n)) Filter.atTop (nhds (HMul. …
    -/
    refine tendsto_const_nhds.mul ?_
    /-
      case convert_6
      𝕜 : Type u_4
      inst✝⁵ : DivisionRing 𝕜
      inst✝⁴ : TopologicalSpace 𝕜
      inst✝³ : CharZero 𝕜
      inst✝² : Algebra Real 𝕜
      inst✝¹ : ContinuousSMul Real 𝕜
      inst✝ : TopologicalDivisionRing 𝕜
      x : 𝕜
      this : Eq (nhds 1) (nhds (HDiv.hDiv 1 (HAdd.hAdd 1 (HMul.hMul x 0))))
      ⊢ Filter.Tendsto (fun n => Inv.inv ↑n) Filter.atTop (nhds 0)
    -/
    have := ((continuous_algebraMap ℝ 𝕜).tendsto _).comp tendsto_inverse_atTop_nhds_zero_nat
    /-
      case convert_6
      𝕜 : Type u_4
      inst✝⁵ : DivisionRing 𝕜
      inst✝⁴ : TopologicalSpace 𝕜
      inst✝³ : CharZero 𝕜
      inst✝² : Algebra Real 𝕜
      inst✝¹ : ContinuousSMul Real 𝕜
      inst✝ : TopologicalDivisionRing 𝕜
      x : 𝕜
      this✝ : Eq (nhds 1) (nhds (HDiv.hDiv 1 (HAdd.hAdd 1 (HMul.hMul x 0))))
      this : Filter.Tendsto (Function.comp ⇑(algebraMap Real 𝕜) fun n => Inv.inv ↑n) …
      ⊢ Filter.Tendsto (fun n => Inv.inv ↑n) Filter.atTop (nhds 0)
    -/
    rw [map_zero, Filter.tendsto_atTop'] at this
    /-
      case convert_6
      𝕜 : Type u_4
      inst✝⁵ : DivisionRing 𝕜
      inst✝⁴ : TopologicalSpace 𝕜
      inst✝³ : CharZero 𝕜
      inst✝² : Algebra Real 𝕜
      inst✝¹ : ContinuousSMul Real 𝕜
      inst✝ : TopologicalDivisionRing 𝕜
      x : 𝕜
      this✝ : Eq (nhds 1) (nhds (HDiv.hDiv 1 (HAdd.hAdd 1 (HMul.hMul x 0))))
      this : ∀ (s : Set 𝕜), Membership.mem (nhds 0) s → Exists fun a => ∀ (b : Nat), …
      ⊢ Filter.Tendsto (fun n => Inv.inv ↑n) Filter.atTop (nhds 0)
    -/
    refine Iff.mpr tendsto_atTop' ?_
    /-
      case convert_6
      𝕜 : Type u_4
      inst✝⁵ : DivisionRing 𝕜
      inst✝⁴ : TopologicalSpace 𝕜
      inst✝³ : CharZero 𝕜
      inst✝² : Algebra Real 𝕜
      inst✝¹ : ContinuousSMul Real 𝕜
      inst✝ : TopologicalDivisionRing 𝕜
      x : 𝕜
      this✝ : Eq (nhds 1) (nhds (HDiv.hDiv 1 (HAdd.hAdd 1 (HMul.hMul x 0))))
      this : ∀ (s : Set 𝕜), Membership.mem (nhds 0) s → Exists fun a => ∀ (b : Nat), …
      ⊢ ∀ (s : Set 𝕜), Membership.mem (nhds 0) s → Exists fun a => ∀ (b : Nat), GE.g …
    -/
    intros
    /-
      case convert_6
      𝕜 : Type u_4
      inst✝⁵ : DivisionRing 𝕜
      inst✝⁴ : TopologicalSpace 𝕜
      inst✝³ : CharZero 𝕜
      inst✝² : Algebra Real 𝕜
      inst✝¹ : ContinuousSMul Real 𝕜
      inst✝ : TopologicalDivisionRing 𝕜
      x : 𝕜
      this✝ : Eq (nhds 1) (nhds (HDiv.hDiv 1 (HAdd.hAdd 1 (HMul.hMul x 0))))
      this : ∀ (s : Set 𝕜), Membership.mem (nhds 0) s → Exists fun a => ∀ (b : Nat), …
      s✝ : Set 𝕜
      a✝ : Membership.mem (nhds 0) s✝
      ⊢ Exists fun a => ∀ (b : Nat), GE.ge b a → Membership.mem s✝ (Inv.inv ↑b)
    -/
    simp_all only [comp_apply, map_inv₀, map_natCast]
    /-
      🎉 no goals
    -/


/-- For any positive `m : ℕ`, `((n % m : ℕ) : ℝ) / (n : ℝ)` tends to `0` as `n` tends to `∞`. -/
theorem tendsto_mod_div_atTop_nhds_zero_nat {m : ℕ} (hm : 0 < m) :
    Tendsto (fun n : ℕ => ((n % m : ℕ) : ℝ) / (n : ℝ)) atTop (𝓝 0) := by
  /-
    m : Nat
    hm : LT.lt 0 m
    ⊢ Filter.Tendsto (fun n => HDiv.hDiv ↑(HMod.hMod n m) ↑n) Filter.atTop (nhds 0)
  -/
  have h0 : ∀ᶠ n : ℕ in atTop, 0 ≤ (fun n : ℕ => ((n % m : ℕ) : ℝ)) n := by aesop
  exact tendsto_bdd_div_atTop_nhds_zero h0
    (.of_forall (fun n ↦  cast_le.mpr (mod_lt n hm).le)) tendsto_natCast_atTop_atTop


theorem Filter.EventuallyEq.div_mul_cancel {α G : Type*} [GroupWithZero G] {f g : α → G}
    {l : Filter α} (hg : Tendsto g l (𝓟 {0}ᶜ)) : (fun x ↦ f x / g x * g x) =ᶠ[l] fun x ↦ f x := by
  /-
    α : Type u_4
    G : Type u_5
    inst✝ : GroupWithZero G
    f g : α → G
    l : Filter α
    hg : Filter.Tendsto g l (Filter.principal (HasCompl.compl (Singleton.singleton …
    ⊢ l.EventuallyEq (fun x => HMul.hMul (HDiv.hDiv (f x) (g x)) (g x)) fun x => f x
  -/
  filter_upwards [hg.le_comap <| preimage_mem_comap (m := g) (mem_principal_self {0}ᶜ)] with x hx
  /-
    case h
    α : Type u_4
    G : Type u_5
    inst✝ : GroupWithZero G
    f g : α → G
    l : Filter α
    hg : Filter.Tendsto g l (Filter.principal (HasCompl.compl (Singleton.singleton …
    x : α
    hx : Membership.mem (Set.preimage g (HasCompl.compl (Singleton.singleton 0))) x
    ⊢ Eq (HMul.hMul (HDiv.hDiv (f x) (g x)) (g x)) (f x)
  -/
  aesop
  /-
    🎉 no goals
  -/


/-- If `g` tends to `∞`, then eventually for all `x` we have `(f x / g x) * g x = f x`. -/
theorem Filter.EventuallyEq.div_mul_cancel_atTop {α K : Type*} [LinearOrderedSemifield K]
    {f g : α → K} {l : Filter α} (hg : Tendsto g l atTop) :
    (fun x ↦ f x / g x * g x) =ᶠ[l] fun x ↦ f x :=
  div_mul_cancel <| hg.mono_right <| le_principal_iff.mpr <|
                                            /-
                                              α : Type u_4
                                              K : Type u_5
                                              inst✝ : LinearOrderedSemifield K
                                              f g : α → K
                                              l : Filter α
                                              hg : Filter.Tendsto g l Filter.atTop
                                              ⊢ HasSubset.Subset (Set.Ioi 0) (HasCompl.compl (Singleton.singleton 0))
                                            -/
    mem_of_superset (Ioi_mem_atTop 0) <| by aesop
                                            /-
                                              🎉 no goals
                                            -/


/-- If when `x` tends to `∞`, `g` tends to `∞` and `f x / g x` tends to a positive
  constant, then `f` tends to `∞`. -/
theorem Tendsto.num {α K : Type*} [LinearOrderedField K] [TopologicalSpace K] [OrderTopology K]
    {f g : α → K} {l : Filter α} (hg : Tendsto g l atTop) {a : K} (ha : 0 < a)
    (hlim : Tendsto (fun x => f x / g x) l (𝓝 a)) :
    Tendsto f l atTop :=
  Tendsto.congr' (EventuallyEq.div_mul_cancel_atTop hg) (Tendsto.mul_atTop ha hlim hg)


/-- If when `x` tends to `∞`, `g` tends to `∞` and `f x / g x` tends to a positive
  constant, then `f` tends to `∞`. -/
theorem Tendsto.den {α K : Type*} [LinearOrderedField K] [TopologicalSpace K] [OrderTopology K]
    [ContinuousInv K] {f g : α → K} {l : Filter α} (hf : Tendsto f l atTop) {a : K} (ha : 0 < a)
    (hlim : Tendsto (fun x => f x / g x) l (𝓝 a)) :
    Tendsto g l atTop := by
  have hlim' : Tendsto (fun x => g x / f x) l (𝓝 a⁻¹) := by
    simp_rw [← inv_div (f _)]
    exact Filter.Tendsto.inv (f := fun x => f x / g x) hlim
  apply Tendsto.congr' (EventuallyEq.div_mul_cancel_atTop hf)
    (Tendsto.mul_atTop (inv_pos_of_pos ha) hlim' hf)


/-- If when `x` tends to `∞`, `f x / g x` tends to a positive constant, then `f` tends to `∞` if
  and only if `g` tends to `∞`. -/
theorem Tendsto.num_atTop_iff_den_atTop {α K : Type*} [LinearOrderedField K] [TopologicalSpace K]
    [OrderTopology K] [ContinuousInv K] {f g : α → K} {l : Filter α} {a : K} (ha : 0 < a)
    (hlim : Tendsto (fun x => f x / g x) l (𝓝 a)) :
    Tendsto f l atTop ↔ Tendsto g l atTop :=
  ⟨fun hf ↦ Tendsto.den hf ha hlim, fun hg ↦ Tendsto.num hg ha hlim⟩


theorem tendsto_add_one_pow_atTop_atTop_of_pos [LinearOrderedSemiring α] [Archimedean α] {r : α}
    (h : 0 < r) : Tendsto (fun n : ℕ ↦ (r + 1) ^ n) atTop atTop :=
  tendsto_atTop_atTop_of_monotone' (pow_right_mono₀ <| le_add_of_nonneg_left h.le) <|
    not_bddAbove_iff.2 fun _ ↦ Set.exists_range_iff.2 <| add_one_pow_unbounded_of_pos _ h


theorem tendsto_pow_atTop_atTop_of_one_lt [LinearOrderedRing α] [Archimedean α] {r : α}
    (h : 1 < r) : Tendsto (fun n : ℕ ↦ r ^ n) atTop atTop :=
  sub_add_cancel r 1 ▸ tendsto_add_one_pow_atTop_atTop_of_pos (sub_pos.2 h)


theorem Nat.tendsto_pow_atTop_atTop_of_one_lt {m : ℕ} (h : 1 < m) :
    Tendsto (fun n : ℕ ↦ m ^ n) atTop atTop :=
  tsub_add_cancel_of_le (le_of_lt h) ▸ tendsto_add_one_pow_atTop_atTop_of_pos (tsub_pos_of_lt h)


theorem tendsto_pow_atTop_nhds_zero_of_lt_one {𝕜 : Type*} [LinearOrderedField 𝕜] [Archimedean 𝕜]
    [TopologicalSpace 𝕜] [OrderTopology 𝕜] {r : 𝕜} (h₁ : 0 ≤ r) (h₂ : r < 1) :
    Tendsto (fun n : ℕ ↦ r ^ n) atTop (𝓝 0) :=
  h₁.eq_or_lt.elim
    (fun hr ↦ (tendsto_add_atTop_iff_nat 1).mp <| by
      /-
        𝕜 : Type u_4
        inst✝³ : LinearOrderedField 𝕜
        inst✝² : Archimedean 𝕜
        inst✝¹ : TopologicalSpace 𝕜
        inst✝ : OrderTopology 𝕜
        r : 𝕜
        h₁ : LE.le 0 r
        h₂ : LT.lt r 1
        hr : Eq 0 r
        ⊢ Filter.Tendsto (fun n => HPow.hPow r (HAdd.hAdd n 1)) Filter.atTop (nhds 0)
      -/
      simp [_root_.pow_succ, ← hr, tendsto_const_nhds])
      /-
        🎉 no goals
      -/
    (fun hr ↦
      have := (one_lt_inv₀ hr).2 h₂ |> tendsto_pow_atTop_atTop_of_one_lt
                                                          /-
                                                            𝕜 : Type u_4
                                                            inst✝³ : LinearOrderedField 𝕜
                                                            inst✝² : Archimedean 𝕜
                                                            inst✝¹ : TopologicalSpace 𝕜
                                                            inst✝ : OrderTopology 𝕜
                                                            r : 𝕜
                                                            h₁ : LE.le 0 r
                                                            h₂ : LT.lt r 1
                                                            hr : LT.lt 0 r
                                                            this : Filter.Tendsto (fun n => HPow.hPow (Inv.inv r) n) Filter.atTop Filter.a …
                                                            n : Nat
                                                            ⊢ Eq (Function.comp (fun r => Inv.inv r) (fun n => HPow.hPow (Inv.inv r) n) n) …
                                                          -/
      (tendsto_inv_atTop_zero.comp this).congr fun n ↦ by simp)
                                                          /-
                                                            🎉 no goals
                                                          -/


@[simp] theorem tendsto_pow_atTop_nhds_zero_iff {𝕜 : Type*} [LinearOrderedField 𝕜] [Archimedean 𝕜]
    [TopologicalSpace 𝕜] [OrderTopology 𝕜] {r : 𝕜} :
    Tendsto (fun n : ℕ ↦ r ^ n) atTop (𝓝 0) ↔ |r| < 1 := by
  /-
    𝕜 : Type u_4
    inst✝³ : LinearOrderedField 𝕜
    inst✝² : Archimedean 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    r : 𝕜
    ⊢ Iff (Filter.Tendsto (fun n => HPow.hPow r n) Filter.atTop (nhds 0)) (LT.lt ( …
  -/
  rw [tendsto_zero_iff_abs_tendsto_zero]
  /-
    𝕜 : Type u_4
    inst✝³ : LinearOrderedField 𝕜
    inst✝² : Archimedean 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    r : 𝕜
    ⊢ Iff (Filter.Tendsto (Function.comp abs fun n => HPow.hPow r n) Filter.atTop  …
  -/
  refine ⟨fun h ↦ by_contra (fun hr_le ↦ ?_), fun h ↦ ?_⟩
    /-
      case refine_1
      𝕜 : Type u_4
      inst✝³ : LinearOrderedField 𝕜
      inst✝² : Archimedean 𝕜
      inst✝¹ : TopologicalSpace 𝕜
      inst✝ : OrderTopology 𝕜
      r : 𝕜
      h : Filter.Tendsto (Function.comp abs fun n => HPow.hPow r n) Filter.atTop (nh …
      hr_le : Not (LT.lt (abs r) 1)
      ⊢ False
    -/
  · by_cases hr : 1 = |r|
      /-
        case pos
        𝕜 : Type u_4
        inst✝³ : LinearOrderedField 𝕜
        inst✝² : Archimedean 𝕜
        inst✝¹ : TopologicalSpace 𝕜
        inst✝ : OrderTopology 𝕜
        r : 𝕜
        h : Filter.Tendsto (Function.comp abs fun n => HPow.hPow r n) Filter.atTop (nh …
        hr_le : Not (LT.lt (abs r) 1)
        hr : Eq 1 (abs r)
        ⊢ False
      -/
    · replace h : Tendsto (fun n : ℕ ↦ |r|^n) atTop (𝓝 0) := by simpa only [← abs_pow, h]
      /-
        case pos
        𝕜 : Type u_4
        inst✝³ : LinearOrderedField 𝕜
        inst✝² : Archimedean 𝕜
        inst✝¹ : TopologicalSpace 𝕜
        inst✝ : OrderTopology 𝕜
        r : 𝕜
        hr_le : Not (LT.lt (abs r) 1)
        hr : Eq 1 (abs r)
        h : Filter.Tendsto (fun n => HPow.hPow (abs r) n) Filter.atTop (nhds 0)
        ⊢ False
      -/
      simp only [hr.symm, one_pow] at h
      /-
        case pos
        𝕜 : Type u_4
        inst✝³ : LinearOrderedField 𝕜
        inst✝² : Archimedean 𝕜
        inst✝¹ : TopologicalSpace 𝕜
        inst✝ : OrderTopology 𝕜
        r : 𝕜
        hr_le : Not (LT.lt (abs r) 1)
        hr : Eq 1 (abs r)
        h : Filter.Tendsto (fun n => 1) Filter.atTop (nhds 0)
        ⊢ False
      -/
      exact zero_ne_one <| tendsto_nhds_unique h tendsto_const_nhds
      /-
        🎉 no goals
      -/
      /-
        case neg
        𝕜 : Type u_4
        inst✝³ : LinearOrderedField 𝕜
        inst✝² : Archimedean 𝕜
        inst✝¹ : TopologicalSpace 𝕜
        inst✝ : OrderTopology 𝕜
        r : 𝕜
        h : Filter.Tendsto (Function.comp abs fun n => HPow.hPow r n) Filter.atTop (nh …
        hr_le : Not (LT.lt (abs r) 1)
        hr : Not (Eq 1 (abs r))
        ⊢ False
      -/
    · apply @not_tendsto_nhds_of_tendsto_atTop 𝕜 ℕ _ _ _ _ atTop _ (fun n ↦ |r| ^ n) _ 0 _
      · refine (pow_right_strictMono₀ <| lt_of_le_of_ne (le_of_not_lt hr_le)
          hr).monotone.tendsto_atTop_atTop (fun b ↦ ?_)
        /-
          𝕜 : Type u_4
          inst✝³ : LinearOrderedField 𝕜
          inst✝² : Archimedean 𝕜
          inst✝¹ : TopologicalSpace 𝕜
          inst✝ : OrderTopology 𝕜
          r : 𝕜
          h : Filter.Tendsto (Function.comp abs fun n => HPow.hPow r n) Filter.atTop (nh …
          hr_le : Not (LT.lt (abs r) 1)
          hr : Not (Eq 1 (abs r))
          b : 𝕜
          ⊢ Exists fun a => LE.le b (HPow.hPow (abs r) a)
        -/
        obtain ⟨n, hn⟩ := (pow_unbounded_of_one_lt b (lt_of_le_of_ne (le_of_not_lt hr_le) hr))
        /-
          case intro
          𝕜 : Type u_4
          inst✝³ : LinearOrderedField 𝕜
          inst✝² : Archimedean 𝕜
          inst✝¹ : TopologicalSpace 𝕜
          inst✝ : OrderTopology 𝕜
          r : 𝕜
          h : Filter.Tendsto (Function.comp abs fun n => HPow.hPow r n) Filter.atTop (nh …
          hr_le : Not (LT.lt (abs r) 1)
          hr : Not (Eq 1 (abs r))
          b : 𝕜
          n : Nat
          hn : LT.lt b (HPow.hPow (abs r) n)
          ⊢ Exists fun a => LE.le b (HPow.hPow (abs r) a)
        -/
        exact ⟨n, le_of_lt hn⟩
        /-
          🎉 no goals
        -/
        /-
          𝕜 : Type u_4
          inst✝³ : LinearOrderedField 𝕜
          inst✝² : Archimedean 𝕜
          inst✝¹ : TopologicalSpace 𝕜
          inst✝ : OrderTopology 𝕜
          r : 𝕜
          h : Filter.Tendsto (Function.comp abs fun n => HPow.hPow r n) Filter.atTop (nh …
          hr_le : Not (LT.lt (abs r) 1)
          hr : Not (Eq 1 (abs r))
          ⊢ Filter.Tendsto (fun n => HPow.hPow (abs r) n) Filter.atTop (nhds 0)
        -/
      · simpa only [← abs_pow]
        /-
          🎉 no goals
        -/
    /-
      case refine_2
      𝕜 : Type u_4
      inst✝³ : LinearOrderedField 𝕜
      inst✝² : Archimedean 𝕜
      inst✝¹ : TopologicalSpace 𝕜
      inst✝ : OrderTopology 𝕜
      r : 𝕜
      h : LT.lt (abs r) 1
      ⊢ Filter.Tendsto (Function.comp abs fun n => HPow.hPow r n) Filter.atTop (nhds …
    -/
  · simpa only [← abs_pow] using (tendsto_pow_atTop_nhds_zero_of_lt_one (abs_nonneg r)) h
    /-
      🎉 no goals
    -/


theorem tendsto_pow_atTop_nhdsWithin_zero_of_lt_one {𝕜 : Type*} [LinearOrderedField 𝕜]
    [Archimedean 𝕜] [TopologicalSpace 𝕜] [OrderTopology 𝕜] {r : 𝕜} (h₁ : 0 < r) (h₂ : r < 1) :
    Tendsto (fun n : ℕ ↦ r ^ n) atTop (𝓝[>] 0) :=
  tendsto_inf.2
    ⟨tendsto_pow_atTop_nhds_zero_of_lt_one h₁.le h₂,
      tendsto_principal.2 <| Eventually.of_forall fun _ ↦ pow_pos h₁ _⟩


theorem uniformity_basis_dist_pow_of_lt_one {α : Type*} [PseudoMetricSpace α] {r : ℝ} (h₀ : 0 < r)
    (h₁ : r < 1) :
    (uniformity α).HasBasis (fun _ : ℕ ↦ True) fun k ↦ { p : α × α | dist p.1 p.2 < r ^ k } :=
  Metric.mk_uniformity_basis (fun _ _ ↦ pow_pos h₀ _) fun _ ε0 ↦
    (exists_pow_lt_of_lt_one ε0 h₁).imp fun _ hk ↦ ⟨trivial, hk.le⟩


theorem geom_lt {u : ℕ → ℝ} {c : ℝ} (hc : 0 ≤ c) {n : ℕ} (hn : 0 < n)
    (h : ∀ k < n, c * u k < u (k + 1)) : c ^ n * u 0 < u n := by
  /-
    u : Nat → Real
    c : Real
    hc : LE.le 0 c
    n : Nat
    hn : LT.lt 0 n
    h : ∀ (k : Nat), LT.lt k n → LT.lt (HMul.hMul c (u k)) (u (HAdd.hAdd k 1))
    ⊢ LT.lt (HMul.hMul (HPow.hPow c n) (u 0)) (u n)
  -/
  apply (monotone_mul_left_of_nonneg hc).seq_pos_lt_seq_of_le_of_lt hn _ _ h
    /-
      u : Nat → Real
      c : Real
      hc : LE.le 0 c
      n : Nat
      hn : LT.lt 0 n
      h : ∀ (k : Nat), LT.lt k n → LT.lt (HMul.hMul c (u k)) (u (HAdd.hAdd k 1))
      ⊢ LE.le (HMul.hMul (HPow.hPow c 0) (u 0)) (u 0)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      u : Nat → Real
      c : Real
      hc : LE.le 0 c
      n : Nat
      hn : LT.lt 0 n
      h : ∀ (k : Nat), LT.lt k n → LT.lt (HMul.hMul c (u k)) (u (HAdd.hAdd k 1))
      ⊢ ∀ (k : Nat), LT.lt k n → LE.le (HMul.hMul (HPow.hPow c (HAdd.hAdd k 1)) (u 0 …
    -/
  · simp [_root_.pow_succ', mul_assoc, le_refl]
    /-
      🎉 no goals
    -/


theorem geom_le {u : ℕ → ℝ} {c : ℝ} (hc : 0 ≤ c) (n : ℕ) (h : ∀ k < n, c * u k ≤ u (k + 1)) :
    c ^ n * u 0 ≤ u n := by
  /-
    u : Nat → Real
    c : Real
    hc : LE.le 0 c
    n : Nat
    h : ∀ (k : Nat), LT.lt k n → LE.le (HMul.hMul c (u k)) (u (HAdd.hAdd k 1))
    ⊢ LE.le (HMul.hMul (HPow.hPow c n) (u 0)) (u n)
  -/
  apply (monotone_mul_left_of_nonneg hc).seq_le_seq n _ _ h <;>
    /-
      u : Nat → Real
      c : Real
      hc : LE.le 0 c
      n : Nat
      h : ∀ (k : Nat), LT.lt k n → LE.le (HMul.hMul c (u k)) (u (HAdd.hAdd k 1))
      ⊢ LE.le (HMul.hMul (HPow.hPow c 0) (u 0)) (u 0)
    -/
    /-
      🎉 no goals
    -/
    simp [_root_.pow_succ', mul_assoc, le_refl]
    /-
      🎉 no goals
    -/


theorem lt_geom {u : ℕ → ℝ} {c : ℝ} (hc : 0 ≤ c) {n : ℕ} (hn : 0 < n)
    (h : ∀ k < n, u (k + 1) < c * u k) : u n < c ^ n * u 0 := by
  /-
    u : Nat → Real
    c : Real
    hc : LE.le 0 c
    n : Nat
    hn : LT.lt 0 n
    h : ∀ (k : Nat), LT.lt k n → LT.lt (u (HAdd.hAdd k 1)) (HMul.hMul c (u k))
    ⊢ LT.lt (u n) (HMul.hMul (HPow.hPow c n) (u 0))
  -/
  apply (monotone_mul_left_of_nonneg hc).seq_pos_lt_seq_of_lt_of_le hn _ h _
    /-
      u : Nat → Real
      c : Real
      hc : LE.le 0 c
      n : Nat
      hn : LT.lt 0 n
      h : ∀ (k : Nat), LT.lt k n → LT.lt (u (HAdd.hAdd k 1)) (HMul.hMul c (u k))
      ⊢ LE.le (u 0) (HMul.hMul (HPow.hPow c 0) (u 0))
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      u : Nat → Real
      c : Real
      hc : LE.le 0 c
      n : Nat
      hn : LT.lt 0 n
      h : ∀ (k : Nat), LT.lt k n → LT.lt (u (HAdd.hAdd k 1)) (HMul.hMul c (u k))
      ⊢ ∀ (k : Nat), LT.lt k n → LE.le (HMul.hMul c (HMul.hMul (HPow.hPow c k) (u 0) …
    -/
  · simp [_root_.pow_succ', mul_assoc, le_refl]
    /-
      🎉 no goals
    -/


theorem le_geom {u : ℕ → ℝ} {c : ℝ} (hc : 0 ≤ c) (n : ℕ) (h : ∀ k < n, u (k + 1) ≤ c * u k) :
    u n ≤ c ^ n * u 0 := by
  /-
    u : Nat → Real
    c : Real
    hc : LE.le 0 c
    n : Nat
    h : ∀ (k : Nat), LT.lt k n → LE.le (u (HAdd.hAdd k 1)) (HMul.hMul c (u k))
    ⊢ LE.le (u n) (HMul.hMul (HPow.hPow c n) (u 0))
  -/
  apply (monotone_mul_left_of_nonneg hc).seq_le_seq n _ h _ <;>
    /-
      u : Nat → Real
      c : Real
      hc : LE.le 0 c
      n : Nat
      h : ∀ (k : Nat), LT.lt k n → LE.le (u (HAdd.hAdd k 1)) (HMul.hMul c (u k))
      ⊢ LE.le (u 0) (HMul.hMul (HPow.hPow c 0) (u 0))
    -/
    /-
      🎉 no goals
    -/
    simp [_root_.pow_succ', mul_assoc, le_refl]
    /-
      🎉 no goals
    -/


/-- If a sequence `v` of real numbers satisfies `k * v n ≤ v (n+1)` with `1 < k`,
then it goes to +∞. -/
theorem tendsto_atTop_of_geom_le {v : ℕ → ℝ} {c : ℝ} (h₀ : 0 < v 0) (hc : 1 < c)
    (hu : ∀ n, c * v n ≤ v (n + 1)) : Tendsto v atTop atTop :=
  (tendsto_atTop_mono fun n ↦ geom_le (zero_le_one.trans hc.le) n fun k _ ↦ hu k) <|
    (tendsto_pow_atTop_atTop_of_one_lt hc).atTop_mul_const h₀


theorem NNReal.tendsto_pow_atTop_nhds_zero_of_lt_one {r : ℝ≥0} (hr : r < 1) :
    Tendsto (fun n : ℕ ↦ r ^ n) atTop (𝓝 0) :=
  NNReal.tendsto_coe.1 <| by
    simp only [NNReal.coe_pow, NNReal.coe_zero,
      _root_.tendsto_pow_atTop_nhds_zero_of_lt_one r.coe_nonneg hr]


@[simp]
protected theorem NNReal.tendsto_pow_atTop_nhds_zero_iff {r : ℝ≥0} :
    Tendsto (fun n : ℕ => r ^ n) atTop (𝓝 0) ↔ r < 1 :=
  ⟨fun h => by simpa [coe_pow, coe_zero, abs_eq, coe_lt_one, val_eq_coe] using
    tendsto_pow_atTop_nhds_zero_iff.mp <| tendsto_coe.mpr h, tendsto_pow_atTop_nhds_zero_of_lt_one⟩


theorem ENNReal.tendsto_pow_atTop_nhds_zero_of_lt_one {r : ℝ≥0∞} (hr : r < 1) :
    Tendsto (fun n : ℕ ↦ r ^ n) atTop (𝓝 0) := by
  /-
    r : ENNReal
    hr : LT.lt r 1
    ⊢ Filter.Tendsto (fun n => HPow.hPow r n) Filter.atTop (nhds 0)
  -/
  rcases ENNReal.lt_iff_exists_coe.1 hr with ⟨r, rfl, hr'⟩
  /-
    case intro.intro
    r : NNReal
    hr' : LT.lt (↑r) 1
    hr : LT.lt (↑r) 1
    ⊢ Filter.Tendsto (fun n => HPow.hPow (↑r) n) Filter.atTop (nhds 0)
  -/
  rw [← ENNReal.coe_zero]
  /-
    case intro.intro
    r : NNReal
    hr' : LT.lt (↑r) 1
    hr : LT.lt (↑r) 1
    ⊢ Filter.Tendsto (fun n => HPow.hPow (↑r) n) Filter.atTop (nhds ↑0)
  -/
  norm_cast at *
  /-
    case intro.intro
    r : NNReal
    hr' hr : LT.lt r 1
    ⊢ Filter.Tendsto (HPow.hPow r) Filter.atTop (nhds 0)
  -/
  apply NNReal.tendsto_pow_atTop_nhds_zero_of_lt_one hr
  /-
    🎉 no goals
  -/


@[simp]
protected theorem ENNReal.tendsto_pow_atTop_nhds_zero_iff {r : ℝ≥0∞} :
    Tendsto (fun n : ℕ => r ^ n) atTop (𝓝 0) ↔ r < 1 := by
  /-
    r : ENNReal
    ⊢ Iff (Filter.Tendsto (fun n => HPow.hPow r n) Filter.atTop (nhds 0)) (LT.lt r …
  -/
  refine ⟨fun h ↦ ?_, tendsto_pow_atTop_nhds_zero_of_lt_one⟩
  /-
    r : ENNReal
    h : Filter.Tendsto (fun n => HPow.hPow r n) Filter.atTop (nhds 0)
    ⊢ LT.lt r 1
  -/
  lift r to NNReal
    /-
      r : ENNReal
      h : Filter.Tendsto (fun n => HPow.hPow r n) Filter.atTop (nhds 0)
      ⊢ Ne r Top.top
    -/
  · refine fun hr ↦ top_ne_zero (tendsto_nhds_unique (EventuallyEq.tendsto ?_) (hr ▸ h))
    /-
      r : ENNReal
      h : Filter.Tendsto (fun n => HPow.hPow r n) Filter.atTop (nhds 0)
      hr : Eq r Top.top
      ⊢ Filter.atTop.EventuallyEq (fun n => HPow.hPow Top.top n) fun x => Top.top
    -/
    exact eventually_atTop.mpr ⟨1, fun _ hn ↦ pow_eq_top_iff.mpr ⟨rfl, Nat.pos_iff_ne_zero.mp hn⟩⟩
    /-
      🎉 no goals
    -/
  /-
    case intro
    r : NNReal
    h : Filter.Tendsto (fun n => HPow.hPow (↑r) n) Filter.atTop (nhds 0)
    ⊢ LT.lt (↑r) 1
  -/
  rw [← coe_zero] at h
  /-
    case intro
    r : NNReal
    h : Filter.Tendsto (fun n => HPow.hPow (↑r) n) Filter.atTop (nhds ↑0)
    ⊢ LT.lt (↑r) 1
  -/
  norm_cast at h ⊢
  /-
    case intro
    r : NNReal
    h : Filter.Tendsto (HPow.hPow r) Filter.atTop (nhds 0)
    ⊢ LT.lt r 1
  -/
  exact NNReal.tendsto_pow_atTop_nhds_zero_iff.mp h
  /-
    🎉 no goals
  -/


@[simp]
protected theorem ENNReal.tendsto_pow_atTop_nhds_top_iff {r : ℝ≥0∞} :
    Tendsto (fun n ↦ r^n) atTop (𝓝 ∞) ↔ 1 < r := by
  /-
    r : ENNReal
    ⊢ Iff (Filter.Tendsto (fun n => HPow.hPow r n) Filter.atTop (nhds Top.top)) (L …
  -/
  refine ⟨?_, ?_⟩
    /-
      case refine_1
      r : ENNReal
      ⊢ Filter.Tendsto (fun n => HPow.hPow r n) Filter.atTop (nhds Top.top) → LT.lt  …
    -/
  · contrapose!
    /-
      case refine_1
      r : ENNReal
      ⊢ LE.le r 1 → Not (Filter.Tendsto (fun n => HPow.hPow r n) Filter.atTop (nhds  …
    -/
    intro r_le_one h_tends
    /-
      case refine_1
      r : ENNReal
      r_le_one : LE.le r 1
      h_tends : Filter.Tendsto (fun n => HPow.hPow r n) Filter.atTop (nhds Top.top)
      ⊢ False
    -/
    specialize h_tends (Ioi_mem_nhds one_lt_top)
    /-
      case refine_1
      r : ENNReal
      r_le_one : LE.le r 1
      h_tends : Membership.mem (Filter.map (fun n => HPow.hPow r n) Filter.atTop) (S …
      ⊢ False
    -/
    simp only [Filter.mem_map, mem_atTop_sets, ge_iff_le, Set.mem_preimage, Set.mem_Ioi] at h_tends
    /-
      case refine_1
      r : ENNReal
      r_le_one : LE.le r 1
      h_tends : Exists fun a => ∀ (b : Nat), LE.le a b → LT.lt 1 (HPow.hPow r b)
      ⊢ False
    -/
    obtain ⟨n, hn⟩ := h_tends
    /-
      case refine_1.intro
      r : ENNReal
      r_le_one : LE.le r 1
      n : Nat
      hn : ∀ (b : Nat), LE.le n b → LT.lt 1 (HPow.hPow r b)
      ⊢ False
    -/
    exact lt_irrefl _ <| lt_of_lt_of_le (hn n le_rfl) <| pow_le_one₀ (zero_le _) r_le_one
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      r : ENNReal
      ⊢ LT.lt 1 r → Filter.Tendsto (fun n => HPow.hPow r n) Filter.atTop (nhds Top.t …
    -/
  · intro r_gt_one
    /-
      case refine_2
      r : ENNReal
      r_gt_one : LT.lt 1 r
      ⊢ Filter.Tendsto (fun n => HPow.hPow r n) Filter.atTop (nhds Top.top)
    -/
    have obs := @Tendsto.inv ℝ≥0∞ ℕ _ _ _ (fun n ↦ (r⁻¹)^n) atTop 0
    /-
      case refine_2
      r : ENNReal
      r_gt_one : LT.lt 1 r
      obs : Filter.Tendsto (fun n => HPow.hPow (Inv.inv r) n) Filter.atTop (nhds 0)  …
      ⊢ Filter.Tendsto (fun n => HPow.hPow r n) Filter.atTop (nhds Top.top)
    -/
    simp only [ENNReal.tendsto_pow_atTop_nhds_zero_iff, inv_zero] at obs
    /-
      case refine_2
      r : ENNReal
      r_gt_one : LT.lt 1 r
      obs : LT.lt (Inv.inv r) 1 → Filter.Tendsto (fun x => Inv.inv (HPow.hPow (Inv.i …
      ⊢ Filter.Tendsto (fun n => HPow.hPow r n) Filter.atTop (nhds Top.top)
    -/
    simpa [← ENNReal.inv_pow] using obs <| ENNReal.inv_lt_one.mpr r_gt_one
    /-
      🎉 no goals
    -/


lemma ENNReal.eq_zero_of_le_mul_pow {x r : ℝ≥0∞} {ε : ℝ≥0} (hr : r < 1)
    (h : ∀ n : ℕ, x ≤ ε * r ^ n) : x = 0 := by
  /-
    x r : ENNReal
    ε : NNReal
    hr : LT.lt r 1
    h : ∀ (n : Nat), LE.le x (HMul.hMul (↑ε) (HPow.hPow r n))
    ⊢ Eq x 0
  -/
  rw [← nonpos_iff_eq_zero]
  /-
    x r : ENNReal
    ε : NNReal
    hr : LT.lt r 1
    h : ∀ (n : Nat), LE.le x (HMul.hMul (↑ε) (HPow.hPow r n))
    ⊢ LE.le x 0
  -/
  refine ge_of_tendsto' (f := fun (n : ℕ) ↦ ε * r ^ n) (x := atTop) ?_ h
  /-
    x r : ENNReal
    ε : NNReal
    hr : LT.lt r 1
    h : ∀ (n : Nat), LE.le x (HMul.hMul (↑ε) (HPow.hPow r n))
    ⊢ Filter.Tendsto (fun n => HMul.hMul (↑ε) (HPow.hPow r n)) Filter.atTop (nhds 0)
  -/
  rw [← mul_zero (M₀ := ℝ≥0∞) (a := ε)]
  /-
    x r : ENNReal
    ε : NNReal
    hr : LT.lt r 1
    h : ∀ (n : Nat), LE.le x (HMul.hMul (↑ε) (HPow.hPow r n))
    ⊢ Filter.Tendsto (fun n => HMul.hMul (↑ε) (HPow.hPow r n)) Filter.atTop (nhds  …
  -/
  exact Tendsto.const_mul (tendsto_pow_atTop_nhds_zero_of_lt_one hr) (Or.inr coe_ne_top)
  /-
    🎉 no goals
  -/


theorem hasSum_geometric_of_lt_one {r : ℝ} (h₁ : 0 ≤ r) (h₂ : r < 1) :
    HasSum (fun n : ℕ ↦ r ^ n) (1 - r)⁻¹ :=
  have : r ≠ 1 := ne_of_lt h₂
  have : Tendsto (fun n ↦ (r ^ n - 1) * (r - 1)⁻¹) atTop (𝓝 ((0 - 1) * (r - 1)⁻¹)) :=
    ((tendsto_pow_atTop_nhds_zero_of_lt_one h₁ h₂).sub tendsto_const_nhds).mul tendsto_const_nhds
  (hasSum_iff_tendsto_nat_of_nonneg (pow_nonneg h₁) _).mpr <| by
    /-
      r : Real
      h₁ : LE.le 0 r
      h₂ : LT.lt r 1
      this✝ : Ne r 1
      this : Filter.Tendsto (fun n => HMul.hMul (HSub.hSub (HPow.hPow r n) 1) (Inv.i …
      ⊢ Filter.Tendsto (fun n => (Finset.range n).sum fun i => HPow.hPow r i) Filter …
    -/
    simp_all [neg_inv, geom_sum_eq, div_eq_mul_inv]
    /-
      🎉 no goals
    -/


theorem summable_geometric_of_lt_one {r : ℝ} (h₁ : 0 ≤ r) (h₂ : r < 1) :
    Summable fun n : ℕ ↦ r ^ n :=
  ⟨_, hasSum_geometric_of_lt_one h₁ h₂⟩



theorem tsum_geometric_of_lt_one {r : ℝ} (h₁ : 0 ≤ r) (h₂ : r < 1) : ∑' n : ℕ, r ^ n = (1 - r)⁻¹ :=
  (hasSum_geometric_of_lt_one h₁ h₂).tsum_eq


theorem hasSum_geometric_two : HasSum (fun n : ℕ ↦ ((1 : ℝ) / 2) ^ n) 2 := by
  /-
    ⊢ HasSum (fun n => HPow.hPow (1 / 2) n) 2
  -/
                                             /-
                                               🎉 no goals
                                             -/
                                             /-
                                               🎉 no goals
                                             -/
  convert hasSum_geometric_of_lt_one _ _ <;> norm_num
                                             /-
                                               🎉 no goals
                                             -/


theorem summable_geometric_two : Summable fun n : ℕ ↦ ((1 : ℝ) / 2) ^ n :=
  ⟨_, hasSum_geometric_two⟩


theorem summable_geometric_two_encode {ι : Type*} [Encodable ι] :
    Summable fun i : ι ↦ (1 / 2 : ℝ) ^ Encodable.encode i :=
  summable_geometric_two.comp_injective Encodable.encode_injective


theorem tsum_geometric_two : (∑' n : ℕ, ((1 : ℝ) / 2) ^ n) = 2 :=
  hasSum_geometric_two.tsum_eq


theorem sum_geometric_two_le (n : ℕ) : (∑ i ∈ range n, (1 / (2 : ℝ)) ^ i) ≤ 2 := by
  have : ∀ i, 0 ≤ (1 / (2 : ℝ)) ^ i := by
    intro i
    apply pow_nonneg
    norm_num
  /-
    n : Nat
    this : ∀ (i : Nat), LE.le 0 (HPow.hPow (1 / 2) i)
    ⊢ LE.le ((Finset.range n).sum fun i => HPow.hPow (1 / 2) i) 2
  -/
  convert sum_le_tsum (range n) (fun i _ ↦ this i) summable_geometric_two
  /-
    case h.e'_4
    n : Nat
    this : ∀ (i : Nat), LE.le 0 (HPow.hPow (1 / 2) i)
    ⊢ Eq 2 (tsum fun i => HPow.hPow (1 / 2) i)
  -/
  exact tsum_geometric_two.symm
  /-
    🎉 no goals
  -/


theorem tsum_geometric_inv_two : (∑' n : ℕ, (2 : ℝ)⁻¹ ^ n) = 2 :=
  (inv_eq_one_div (2 : ℝ)).symm ▸ tsum_geometric_two


/-- The sum of `2⁻¹ ^ i` for `n ≤ i` equals `2 * 2⁻¹ ^ n`. -/
theorem tsum_geometric_inv_two_ge (n : ℕ) :
    (∑' i, ite (n ≤ i) ((2 : ℝ)⁻¹ ^ i) 0) = 2 * 2⁻¹ ^ n := by
  have A : Summable fun i : ℕ ↦ ite (n ≤ i) ((2⁻¹ : ℝ) ^ i) 0 := by
    simpa only [← piecewise_eq_indicator, one_div]
      using summable_geometric_two.indicator {i | n ≤ i}
  have B : ((Finset.range n).sum fun i : ℕ ↦ ite (n ≤ i) ((2⁻¹ : ℝ) ^ i) 0) = 0 :=
    Finset.sum_eq_zero fun i hi ↦
      ite_eq_right_iff.2 fun h ↦ (lt_irrefl _ ((Finset.mem_range.1 hi).trans_le h)).elim
  simp only [← _root_.sum_add_tsum_nat_add n A, B, if_true, zero_add, zero_le',
    le_add_iff_nonneg_left, pow_add, _root_.tsum_mul_right, tsum_geometric_inv_two]


theorem hasSum_geometric_two' (a : ℝ) : HasSum (fun n : ℕ ↦ a / 2 / 2 ^ n) a := by
  convert HasSum.mul_left (a / 2)
      (hasSum_geometric_of_lt_one (le_of_lt one_half_pos) one_half_lt_one) using 1
    /-
      case h.e'_5
      a : Real
      ⊢ Eq (fun n => HDiv.hDiv (HDiv.hDiv a 2) (HPow.hPow 2 n)) fun i => HMul.hMul ( …
    -/
  · funext n
    /-
      case h.e'_5.h
      a : Real
      n : Nat
      ⊢ Eq (HDiv.hDiv (HDiv.hDiv a 2) (HPow.hPow 2 n)) (HMul.hMul (HDiv.hDiv a 2) (H …
    -/
    simp only [one_div, inv_pow]
    /-
      case h.e'_5.h
      a : Real
      n : Nat
      ⊢ Eq (HDiv.hDiv (HDiv.hDiv a 2) (HPow.hPow 2 n)) (HMul.hMul (HDiv.hDiv a 2) (I …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case h.e'_6
      a : Real
      ⊢ Eq a (HMul.hMul (HDiv.hDiv a 2) (Inv.inv (HSub.hSub 1 (1 / 2))))
    -/
  · norm_num
    /-
      🎉 no goals
    -/


theorem summable_geometric_two' (a : ℝ) : Summable fun n : ℕ ↦ a / 2 / 2 ^ n :=
  ⟨a, hasSum_geometric_two' a⟩


theorem tsum_geometric_two' (a : ℝ) : ∑' n : ℕ, a / 2 / 2 ^ n = a :=
  (hasSum_geometric_two' a).tsum_eq


/-- **Sum of a Geometric Series** -/
theorem NNReal.hasSum_geometric {r : ℝ≥0} (hr : r < 1) : HasSum (fun n : ℕ ↦ r ^ n) (1 - r)⁻¹ := by
  /-
    r : NNReal
    hr : LT.lt r 1
    ⊢ HasSum (fun n => HPow.hPow r n) (Inv.inv (HSub.hSub 1 r))
  -/
  apply NNReal.hasSum_coe.1
  /-
    r : NNReal
    hr : LT.lt r 1
    ⊢ HasSum (fun a => ↑(HPow.hPow r a)) ↑(Inv.inv (HSub.hSub 1 r))
  -/
  push_cast
  /-
    r : NNReal
    hr : LT.lt r 1
    ⊢ HasSum (fun a => HPow.hPow (↑r) a) (Inv.inv ↑(HSub.hSub 1 r))
  -/
  rw [NNReal.coe_sub (le_of_lt hr)]
  /-
    r : NNReal
    hr : LT.lt r 1
    ⊢ HasSum (fun a => HPow.hPow (↑r) a) (Inv.inv (HSub.hSub ↑1 ↑r))
  -/
  exact hasSum_geometric_of_lt_one r.coe_nonneg hr
  /-
    🎉 no goals
  -/


theorem NNReal.summable_geometric {r : ℝ≥0} (hr : r < 1) : Summable fun n : ℕ ↦ r ^ n :=
  ⟨_, NNReal.hasSum_geometric hr⟩


theorem tsum_geometric_nnreal {r : ℝ≥0} (hr : r < 1) : ∑' n : ℕ, r ^ n = (1 - r)⁻¹ :=
  (NNReal.hasSum_geometric hr).tsum_eq


/-- The series `pow r` converges to `(1-r)⁻¹`. For `r < 1` the RHS is a finite number,
and for `1 ≤ r` the RHS equals `∞`. -/
@[simp]
theorem ENNReal.tsum_geometric (r : ℝ≥0∞) : ∑' n : ℕ, r ^ n = (1 - r)⁻¹ := by
  /-
    r : ENNReal
    ⊢ Eq (tsum fun n => HPow.hPow r n) (Inv.inv (HSub.hSub 1 r))
  -/
  cases' lt_or_le r 1 with hr hr
    /-
      case inl
      r : ENNReal
      hr : LT.lt r 1
      ⊢ Eq (tsum fun n => HPow.hPow r n) (Inv.inv (HSub.hSub 1 r))
    -/
  · rcases ENNReal.lt_iff_exists_coe.1 hr with ⟨r, rfl, hr'⟩
    /-
      case inl.intro.intro
      r : NNReal
      hr' : LT.lt (↑r) 1
      hr : LT.lt (↑r) 1
      ⊢ Eq (tsum fun n => HPow.hPow (↑r) n) (Inv.inv (HSub.hSub 1 ↑r))
    -/
    norm_cast at *
    /-
      case inl.intro.intro
      r : NNReal
      hr' hr : LT.lt r 1
      ⊢ Eq (tsum fun n => ↑(HPow.hPow r n)) (Inv.inv ↑(HSub.hSub 1 r))
    -/
    convert ENNReal.tsum_coe_eq (NNReal.hasSum_geometric hr)
    /-
      case h.e'_3
      r : NNReal
      hr' hr : LT.lt r 1
      ⊢ Eq (Inv.inv ↑(HSub.hSub 1 r)) ↑(Inv.inv (HSub.hSub 1 r))
    -/
    rw [ENNReal.coe_inv <| ne_of_gt <| tsub_pos_iff_lt.2 hr, coe_sub, coe_one]
    /-
      🎉 no goals
    -/
    /-
      case inr
      r : ENNReal
      hr : LE.le 1 r
      ⊢ Eq (tsum fun n => HPow.hPow r n) (Inv.inv (HSub.hSub 1 r))
    -/
  · rw [tsub_eq_zero_iff_le.mpr hr, ENNReal.inv_zero, ENNReal.tsum_eq_iSup_nat, iSup_eq_top]
    refine fun a ha ↦
      (ENNReal.exists_nat_gt (lt_top_iff_ne_top.1 ha)).imp fun n hn ↦ lt_of_lt_of_le hn ?_
    calc
      (n : ℝ≥0∞) = ∑ i ∈ range n, 1 := by rw [sum_const, nsmul_one, card_range]
      _ ≤ ∑ i ∈ range n, r ^ i := by gcongr; apply one_le_pow₀ hr


theorem ENNReal.tsum_geometric_add_one (r : ℝ≥0∞) : ∑' n : ℕ, r ^ (n + 1) = r * (1 - r)⁻¹ := by
  /-
    r : ENNReal
    ⊢ Eq (tsum fun n => HPow.hPow r (HAdd.hAdd n 1)) (HMul.hMul r (Inv.inv (HSub.h …
  -/
  simp only [_root_.pow_succ', ENNReal.tsum_mul_left, ENNReal.tsum_geometric]
  /-
    🎉 no goals
  -/


include hr hC hu in
/-- If `edist (f n) (f (n+1))` is bounded by `C * r^n`, `C ≠ ∞`, `r < 1`,
then `f` is a Cauchy sequence. -/
theorem cauchySeq_of_edist_le_geometric : CauchySeq f := by
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    r C : ENNReal
    hr : LT.lt r 1
    hC : Ne C Top.top
    f : Nat → α
    hu : ∀ (n : Nat), LE.le (EDist.edist (f n) (f (HAdd.hAdd n 1))) (HMul.hMul C ( …
    ⊢ CauchySeq f
  -/
  refine cauchySeq_of_edist_le_of_tsum_ne_top _ hu ?_
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    r C : ENNReal
    hr : LT.lt r 1
    hC : Ne C Top.top
    f : Nat → α
    hu : ∀ (n : Nat), LE.le (EDist.edist (f n) (f (HAdd.hAdd n 1))) (HMul.hMul C ( …
    ⊢ Ne (tsum fun n => HMul.hMul C (HPow.hPow r n)) Top.top
  -/
  rw [ENNReal.tsum_mul_left, ENNReal.tsum_geometric]
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    r C : ENNReal
    hr : LT.lt r 1
    hC : Ne C Top.top
    f : Nat → α
    hu : ∀ (n : Nat), LE.le (EDist.edist (f n) (f (HAdd.hAdd n 1))) (HMul.hMul C ( …
    ⊢ Ne (HMul.hMul C (Inv.inv (HSub.hSub 1 r))) Top.top
  -/
  refine ENNReal.mul_ne_top hC (ENNReal.inv_ne_top.2 ?_)
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    r C : ENNReal
    hr : LT.lt r 1
    hC : Ne C Top.top
    f : Nat → α
    hu : ∀ (n : Nat), LE.le (EDist.edist (f n) (f (HAdd.hAdd n 1))) (HMul.hMul C ( …
    ⊢ Ne (HSub.hSub 1 r) 0
  -/
  exact (tsub_pos_iff_lt.2 hr).ne'
  /-
    🎉 no goals
  -/


include hu in
/-- If `edist (f n) (f (n+1))` is bounded by `C * r^n`, then the distance from
`f n` to the limit of `f` is bounded above by `C * r^n / (1 - r)`. -/
theorem edist_le_of_edist_le_geometric_of_tendsto {a : α} (ha : Tendsto f atTop (𝓝 a)) (n : ℕ) :
    edist (f n) a ≤ C * r ^ n / (1 - r) := by
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    r C : ENNReal
    f : Nat → α
    hu : ∀ (n : Nat), LE.le (EDist.edist (f n) (f (HAdd.hAdd n 1))) (HMul.hMul C ( …
    a : α
    ha : Filter.Tendsto f Filter.atTop (nhds a)
    n : Nat
    ⊢ LE.le (EDist.edist (f n) a) (HDiv.hDiv (HMul.hMul C (HPow.hPow r n)) (HSub.h …
  -/
  convert edist_le_tsum_of_edist_le_of_tendsto _ hu ha _
  /-
    case h.e'_4
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    r C : ENNReal
    f : Nat → α
    hu : ∀ (n : Nat), LE.le (EDist.edist (f n) (f (HAdd.hAdd n 1))) (HMul.hMul C ( …
    a : α
    ha : Filter.Tendsto f Filter.atTop (nhds a)
    n : Nat
    ⊢ Eq (HDiv.hDiv (HMul.hMul C (HPow.hPow r n)) (HSub.hSub 1 r)) (tsum fun m =>  …
  -/
  simp only [pow_add, ENNReal.tsum_mul_left, ENNReal.tsum_geometric, div_eq_mul_inv, mul_assoc]
  /-
    🎉 no goals
  -/


include hu in
/-- If `edist (f n) (f (n+1))` is bounded by `C * r^n`, then the distance from
`f 0` to the limit of `f` is bounded above by `C / (1 - r)`. -/
theorem edist_le_of_edist_le_geometric_of_tendsto₀ {a : α} (ha : Tendsto f atTop (𝓝 a)) :
    edist (f 0) a ≤ C / (1 - r) := by
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    r C : ENNReal
    f : Nat → α
    hu : ∀ (n : Nat), LE.le (EDist.edist (f n) (f (HAdd.hAdd n 1))) (HMul.hMul C ( …
    a : α
    ha : Filter.Tendsto f Filter.atTop (nhds a)
    ⊢ LE.le (EDist.edist (f 0) a) (HDiv.hDiv C (HSub.hSub 1 r))
  -/
  simpa only [_root_.pow_zero, mul_one] using edist_le_of_edist_le_geometric_of_tendsto r C hu ha 0
  /-
    🎉 no goals
  -/


include hC hu in
/-- If `edist (f n) (f (n+1))` is bounded by `C * 2^-n`, then `f` is a Cauchy sequence. -/
theorem cauchySeq_of_edist_le_geometric_two : CauchySeq f := by
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    C : ENNReal
    hC : Ne C Top.top
    f : Nat → α
    hu : ∀ (n : Nat), LE.le (EDist.edist (f n) (f (HAdd.hAdd n 1))) (HDiv.hDiv C ( …
    ⊢ CauchySeq f
  -/
  simp only [div_eq_mul_inv, ENNReal.inv_pow] at hu
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    C : ENNReal
    hC : Ne C Top.top
    f : Nat → α
    hu : ∀ (n : Nat), LE.le (EDist.edist (f n) (f (HAdd.hAdd n 1))) (HMul.hMul C ( …
    ⊢ CauchySeq f
  -/
  refine cauchySeq_of_edist_le_geometric 2⁻¹ C ?_ hC hu
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    C : ENNReal
    hC : Ne C Top.top
    f : Nat → α
    hu : ∀ (n : Nat), LE.le (EDist.edist (f n) (f (HAdd.hAdd n 1))) (HMul.hMul C ( …
    ⊢ LT.lt (Inv.inv 2) 1
  -/
  simp [ENNReal.one_lt_two]
  /-
    🎉 no goals
  -/


include hu ha in
/-- If `edist (f n) (f (n+1))` is bounded by `C * 2^-n`, then the distance from
`f n` to the limit of `f` is bounded above by `2 * C * 2^-n`. -/
theorem edist_le_of_edist_le_geometric_two_of_tendsto (n : ℕ) : edist (f n) a ≤ 2 * C / 2 ^ n := by
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    C : ENNReal
    f : Nat → α
    hu : ∀ (n : Nat), LE.le (EDist.edist (f n) (f (HAdd.hAdd n 1))) (HDiv.hDiv C ( …
    a : α
    ha : Filter.Tendsto f Filter.atTop (nhds a)
    n : Nat
    ⊢ LE.le (EDist.edist (f n) a) (HDiv.hDiv (HMul.hMul 2 C) (HPow.hPow 2 n))
  -/
  simp only [div_eq_mul_inv, ENNReal.inv_pow] at *
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    C : ENNReal
    f : Nat → α
    a : α
    ha : Filter.Tendsto f Filter.atTop (nhds a)
    n : Nat
    hu : ∀ (n : Nat), LE.le (EDist.edist (f n) (f (HAdd.hAdd n 1))) (HMul.hMul C ( …
    ⊢ LE.le (EDist.edist (f n) a) (HMul.hMul (HMul.hMul 2 C) (HPow.hPow (Inv.inv 2 …
  -/
  rw [mul_assoc, mul_comm]
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    C : ENNReal
    f : Nat → α
    a : α
    ha : Filter.Tendsto f Filter.atTop (nhds a)
    n : Nat
    hu : ∀ (n : Nat), LE.le (EDist.edist (f n) (f (HAdd.hAdd n 1))) (HMul.hMul C ( …
    ⊢ LE.le (EDist.edist (f n) a) (HMul.hMul (HMul.hMul C (HPow.hPow (Inv.inv 2) n …
  -/
  convert edist_le_of_edist_le_geometric_of_tendsto 2⁻¹ C hu ha n using 1
  /-
    case h.e'_4
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    C : ENNReal
    f : Nat → α
    a : α
    ha : Filter.Tendsto f Filter.atTop (nhds a)
    n : Nat
    hu : ∀ (n : Nat), LE.le (EDist.edist (f n) (f (HAdd.hAdd n 1))) (HMul.hMul C ( …
    ⊢ Eq (HMul.hMul (HMul.hMul C (HPow.hPow (Inv.inv 2) n)) 2) (HDiv.hDiv (HMul.hM …
  -/
  rw [ENNReal.one_sub_inv_two, div_eq_mul_inv, inv_inv]
  /-
    🎉 no goals
  -/


include hu ha in
/-- If `edist (f n) (f (n+1))` is bounded by `C * 2^-n`, then the distance from
`f 0` to the limit of `f` is bounded above by `2 * C`. -/
theorem edist_le_of_edist_le_geometric_two_of_tendsto₀ : edist (f 0) a ≤ 2 * C := by
  simpa only [_root_.pow_zero, div_eq_mul_inv, inv_one, mul_one] using
    edist_le_of_edist_le_geometric_two_of_tendsto C hu ha 0


/-- If `dist (f n) (f (n+1))` is bounded by `C * r^n`, `r < 1`, then `f` is a Cauchy sequence. -/
theorem aux_hasSum_of_le_geometric : HasSum (fun n : ℕ ↦ C * r ^ n) (C / (1 - r)) := by
  /-
    α : Type u_1
    inst✝ : PseudoMetricSpace α
    r C : Real
    f : Nat → α
    hr : LT.lt r 1
    hu : ∀ (n : Nat), LE.le (Dist.dist (f n) (f (HAdd.hAdd n 1))) (HMul.hMul C (HP …
    ⊢ HasSum (fun n => HMul.hMul C (HPow.hPow r n)) (HDiv.hDiv C (HSub.hSub 1 r))
  -/
  rcases sign_cases_of_C_mul_pow_nonneg fun n ↦ dist_nonneg.trans (hu n) with (rfl | ⟨_, r₀⟩)
    /-
      case inl
      α : Type u_1
      inst✝ : PseudoMetricSpace α
      r : Real
      f : Nat → α
      hr : LT.lt r 1
      hu : ∀ (n : Nat), LE.le (Dist.dist (f n) (f (HAdd.hAdd n 1))) (HMul.hMul 0 (HP …
      ⊢ HasSum (fun n => HMul.hMul 0 (HPow.hPow r n)) (HDiv.hDiv 0 (HSub.hSub 1 r))
    -/
  · simp [hasSum_zero]
    /-
      🎉 no goals
    -/
    /-
      case inr.intro
      α : Type u_1
      inst✝ : PseudoMetricSpace α
      r C : Real
      f : Nat → α
      hr : LT.lt r 1
      hu : ∀ (n : Nat), LE.le (Dist.dist (f n) (f (HAdd.hAdd n 1))) (HMul.hMul C (HP …
      left✝ : LT.lt 0 C
      r₀ : LE.le 0 r
      ⊢ HasSum (fun n => HMul.hMul C (HPow.hPow r n)) (HDiv.hDiv C (HSub.hSub 1 r))
    -/
  · refine HasSum.mul_left C ?_
    /-
      case inr.intro
      α : Type u_1
      inst✝ : PseudoMetricSpace α
      r C : Real
      f : Nat → α
      hr : LT.lt r 1
      hu : ∀ (n : Nat), LE.le (Dist.dist (f n) (f (HAdd.hAdd n 1))) (HMul.hMul C (HP …
      left✝ : LT.lt 0 C
      r₀ : LE.le 0 r
      ⊢ HasSum (HPow.hPow r) (Inv.inv (HSub.hSub 1 r))
    -/
    simpa using hasSum_geometric_of_lt_one r₀ hr
    /-
      🎉 no goals
    -/


/-- If `dist (f n) (f (n+1))` is bounded by `C * r^n`, `r < 1`, then `f` is a Cauchy sequence.
Note that this lemma does not assume `0 ≤ C` or `0 ≤ r`. -/
theorem cauchySeq_of_le_geometric : CauchySeq f :=
  cauchySeq_of_dist_le_of_summable _ hu ⟨_, aux_hasSum_of_le_geometric hr hu⟩


/-- If `dist (f n) (f (n+1))` is bounded by `C * r^n`, `r < 1`, then the distance from
`f n` to the limit of `f` is bounded above by `C * r^n / (1 - r)`. -/
theorem dist_le_of_le_geometric_of_tendsto₀ {a : α} (ha : Tendsto f atTop (𝓝 a)) :
    dist (f 0) a ≤ C / (1 - r) :=
  (aux_hasSum_of_le_geometric hr hu).tsum_eq ▸
    dist_le_tsum_of_dist_le_of_tendsto₀ _ hu ⟨_, aux_hasSum_of_le_geometric hr hu⟩ ha


/-- If `dist (f n) (f (n+1))` is bounded by `C * r^n`, `r < 1`, then the distance from
`f 0` to the limit of `f` is bounded above by `C / (1 - r)`. -/
theorem dist_le_of_le_geometric_of_tendsto {a : α} (ha : Tendsto f atTop (𝓝 a)) (n : ℕ) :
    dist (f n) a ≤ C * r ^ n / (1 - r) := by
  /-
    α : Type u_1
    inst✝ : PseudoMetricSpace α
    r C : Real
    f : Nat → α
    hr : LT.lt r 1
    hu : ∀ (n : Nat), LE.le (Dist.dist (f n) (f (HAdd.hAdd n 1))) (HMul.hMul C (HP …
    a : α
    ha : Filter.Tendsto f Filter.atTop (nhds a)
    n : Nat
    ⊢ LE.le (Dist.dist (f n) a) (HDiv.hDiv (HMul.hMul C (HPow.hPow r n)) (HSub.hSu …
  -/
  have := aux_hasSum_of_le_geometric hr hu
  /-
    α : Type u_1
    inst✝ : PseudoMetricSpace α
    r C : Real
    f : Nat → α
    hr : LT.lt r 1
    hu : ∀ (n : Nat), LE.le (Dist.dist (f n) (f (HAdd.hAdd n 1))) (HMul.hMul C (HP …
    a : α
    ha : Filter.Tendsto f Filter.atTop (nhds a)
    n : Nat
    this : HasSum (fun n => HMul.hMul C (HPow.hPow r n)) (HDiv.hDiv C (HSub.hSub 1 …
    ⊢ LE.le (Dist.dist (f n) a) (HDiv.hDiv (HMul.hMul C (HPow.hPow r n)) (HSub.hSu …
  -/
  convert dist_le_tsum_of_dist_le_of_tendsto _ hu ⟨_, this⟩ ha n
  /-
    case h.e'_4
    α : Type u_1
    inst✝ : PseudoMetricSpace α
    r C : Real
    f : Nat → α
    hr : LT.lt r 1
    hu : ∀ (n : Nat), LE.le (Dist.dist (f n) (f (HAdd.hAdd n 1))) (HMul.hMul C (HP …
    a : α
    ha : Filter.Tendsto f Filter.atTop (nhds a)
    n : Nat
    this : HasSum (fun n => HMul.hMul C (HPow.hPow r n)) (HDiv.hDiv C (HSub.hSub 1 …
    ⊢ Eq (HDiv.hDiv (HMul.hMul C (HPow.hPow r n)) (HSub.hSub 1 r)) (tsum fun m =>  …
  -/
  simp only [pow_add, mul_left_comm C, mul_div_right_comm]
  /-
    case h.e'_4
    α : Type u_1
    inst✝ : PseudoMetricSpace α
    r C : Real
    f : Nat → α
    hr : LT.lt r 1
    hu : ∀ (n : Nat), LE.le (Dist.dist (f n) (f (HAdd.hAdd n 1))) (HMul.hMul C (HP …
    a : α
    ha : Filter.Tendsto f Filter.atTop (nhds a)
    n : Nat
    this : HasSum (fun n => HMul.hMul C (HPow.hPow r n)) (HDiv.hDiv C (HSub.hSub 1 …
    ⊢ Eq (HMul.hMul (HDiv.hDiv C (HSub.hSub 1 r)) (HPow.hPow r n)) (tsum fun m =>  …
  -/
  rw [mul_comm]
  /-
    case h.e'_4
    α : Type u_1
    inst✝ : PseudoMetricSpace α
    r C : Real
    f : Nat → α
    hr : LT.lt r 1
    hu : ∀ (n : Nat), LE.le (Dist.dist (f n) (f (HAdd.hAdd n 1))) (HMul.hMul C (HP …
    a : α
    ha : Filter.Tendsto f Filter.atTop (nhds a)
    n : Nat
    this : HasSum (fun n => HMul.hMul C (HPow.hPow r n)) (HDiv.hDiv C (HSub.hSub 1 …
    ⊢ Eq (HMul.hMul (HPow.hPow r n) (HDiv.hDiv C (HSub.hSub 1 r))) (tsum fun m =>  …
  -/
  exact (this.mul_left _).tsum_eq.symm
  /-
    🎉 no goals
  -/


/-- If `dist (f n) (f (n+1))` is bounded by `(C / 2) / 2^n`, then `f` is a Cauchy sequence. -/
theorem cauchySeq_of_le_geometric_two : CauchySeq f :=
  cauchySeq_of_dist_le_of_summable _ hu₂ <| ⟨_, hasSum_geometric_two' C⟩


/-- If `dist (f n) (f (n+1))` is bounded by `(C / 2) / 2^n`, then the distance from
`f 0` to the limit of `f` is bounded above by `C`. -/
theorem dist_le_of_le_geometric_two_of_tendsto₀ {a : α} (ha : Tendsto f atTop (𝓝 a)) :
    dist (f 0) a ≤ C :=
  tsum_geometric_two' C ▸ dist_le_tsum_of_dist_le_of_tendsto₀ _ hu₂ (summable_geometric_two' C) ha


/-- If `dist (f n) (f (n+1))` is bounded by `(C / 2) / 2^n`, then the distance from
`f n` to the limit of `f` is bounded above by `C / 2^n`. -/
theorem dist_le_of_le_geometric_two_of_tendsto {a : α} (ha : Tendsto f atTop (𝓝 a)) (n : ℕ) :
    dist (f n) a ≤ C / 2 ^ n := by
  /-
    α : Type u_1
    inst✝ : PseudoMetricSpace α
    C : Real
    f : Nat → α
    hu₂ : ∀ (n : Nat), LE.le (Dist.dist (f n) (f (HAdd.hAdd n 1))) (HDiv.hDiv (HDi …
    a : α
    ha : Filter.Tendsto f Filter.atTop (nhds a)
    n : Nat
    ⊢ LE.le (Dist.dist (f n) a) (HDiv.hDiv C (HPow.hPow 2 n))
  -/
  convert dist_le_tsum_of_dist_le_of_tendsto _ hu₂ (summable_geometric_two' C) ha n
  /-
    case h.e'_4
    α : Type u_1
    inst✝ : PseudoMetricSpace α
    C : Real
    f : Nat → α
    hu₂ : ∀ (n : Nat), LE.le (Dist.dist (f n) (f (HAdd.hAdd n 1))) (HDiv.hDiv (HDi …
    a : α
    ha : Filter.Tendsto f Filter.atTop (nhds a)
    n : Nat
    ⊢ Eq (HDiv.hDiv C (HPow.hPow 2 n)) (tsum fun m => HDiv.hDiv (HDiv.hDiv C 2) (H …
  -/
  simp only [add_comm n, pow_add, ← div_div]
  /-
    case h.e'_4
    α : Type u_1
    inst✝ : PseudoMetricSpace α
    C : Real
    f : Nat → α
    hu₂ : ∀ (n : Nat), LE.le (Dist.dist (f n) (f (HAdd.hAdd n 1))) (HDiv.hDiv (HDi …
    a : α
    ha : Filter.Tendsto f Filter.atTop (nhds a)
    n : Nat
    ⊢ Eq (HDiv.hDiv C (HPow.hPow 2 n)) (tsum fun m => HDiv.hDiv (HDiv.hDiv (HDiv.h …
  -/
  symm
  /-
    case h.e'_4
    α : Type u_1
    inst✝ : PseudoMetricSpace α
    C : Real
    f : Nat → α
    hu₂ : ∀ (n : Nat), LE.le (Dist.dist (f n) (f (HAdd.hAdd n 1))) (HDiv.hDiv (HDi …
    a : α
    ha : Filter.Tendsto f Filter.atTop (nhds a)
    n : Nat
    ⊢ Eq (tsum fun m => HDiv.hDiv (HDiv.hDiv (HDiv.hDiv C 2) (HPow.hPow 2 m)) (HPo …
  -/
  exact ((hasSum_geometric_two' C).div_const _).tsum_eq
  /-
    🎉 no goals
  -/


/-- A series whose terms are bounded by the terms of a converging geometric series converges. -/
theorem summable_one_div_pow_of_le {m : ℝ} {f : ℕ → ℕ} (hm : 1 < m) (fi : ∀ i, i ≤ f i) :
    Summable fun i ↦ 1 / m ^ f i := by
  refine .of_nonneg_of_le (fun a ↦ by positivity) (fun a ↦ ?_)
      (summable_geometric_of_lt_one (one_div_nonneg.mpr (zero_le_one.trans hm.le))
        ((one_div_lt (zero_lt_one.trans hm) zero_lt_one).mpr (one_div_one.le.trans_lt hm)))
  /-
    m : Real
    f : Nat → Nat
    hm : LT.lt 1 m
    fi : ∀ (i : Nat), LE.le i (f i)
    a : Nat
    ⊢ LE.le (HDiv.hDiv 1 (HPow.hPow m (f a))) (HPow.hPow (HDiv.hDiv 1 m) a)
  -/
  rw [div_pow, one_pow]
  /-
    m : Real
    f : Nat → Nat
    hm : LT.lt 1 m
    fi : ∀ (i : Nat), LE.le i (f i)
    a : Nat
    ⊢ LE.le (HDiv.hDiv 1 (HPow.hPow m (f a))) (HDiv.hDiv 1 (HPow.hPow m a))
  -/
  refine (one_div_le_one_div ?_ ?_).mpr (pow_right_mono₀ hm.le (fi a)) <;>
    /-
      case refine_1
      m : Real
      f : Nat → Nat
      hm : LT.lt 1 m
      fi : ∀ (i : Nat), LE.le i (f i)
      a : Nat
      ⊢ LT.lt 0 (HPow.hPow m (f a))
    -/
    /-
      🎉 no goals
    -/
    exact pow_pos (zero_lt_one.trans hm) _
    /-
      🎉 no goals
    -/


/-- For any positive `ε`, define on an encodable type a positive sequence with sum less than `ε` -/
def posSumOfEncodable {ε : ℝ} (hε : 0 < ε) (ι) [Encodable ι] :
    { ε' : ι → ℝ // (∀ i, 0 < ε' i) ∧ ∃ c, HasSum ε' c ∧ c ≤ ε } := by
  /-
    α : Type u_1
    β : Type u_2
    ι✝ : Type u_3
    ε : Real
    hε : LT.lt 0 ε
    ι : Type ?u.145107
    inst✝ : Encodable ι
    ⊢ Subtype fun ε' => And (∀ (i : ι), LT.lt 0 (ε' i)) (Exists fun c => And (HasS …
  -/
  let f n := ε / 2 / 2 ^ n
  /-
    α : Type u_1
    β : Type u_2
    ι✝ : Type u_3
    ε : Real
    hε : LT.lt 0 ε
    ι : Type ?u.145107
    inst✝ : Encodable ι
    f : Nat → Real := fun n => HDiv.hDiv (HDiv.hDiv ε 2) (HPow.hPow 2 n)
    ⊢ Subtype fun ε' => And (∀ (i : ι), LT.lt 0 (ε' i)) (Exists fun c => And (HasS …
  -/
  have hf : HasSum f ε := hasSum_geometric_two' _
  /-
    α : Type u_1
    β : Type u_2
    ι✝ : Type u_3
    ε : Real
    hε : LT.lt 0 ε
    ι : Type ?u.145107
    inst✝ : Encodable ι
    f : Nat → Real := fun n => HDiv.hDiv (HDiv.hDiv ε 2) (HPow.hPow 2 n)
    hf : HasSum f ε
    ⊢ Subtype fun ε' => And (∀ (i : ι), LT.lt 0 (ε' i)) (Exists fun c => And (HasS …
  -/
  have f0 : ∀ n, 0 < f n := fun n ↦ div_pos (half_pos hε) (pow_pos zero_lt_two _)
  /-
    α : Type u_1
    β : Type u_2
    ι✝ : Type u_3
    ε : Real
    hε : LT.lt 0 ε
    ι : Type ?u.145107
    inst✝ : Encodable ι
    f : Nat → Real := fun n => HDiv.hDiv (HDiv.hDiv ε 2) (HPow.hPow 2 n)
    hf : HasSum f ε
    f0 : ∀ (n : Nat), LT.lt 0 (f n)
    ⊢ Subtype fun ε' => And (∀ (i : ι), LT.lt 0 (ε' i)) (Exists fun c => And (HasS …
  -/
  refine ⟨f ∘ Encodable.encode, fun i ↦ f0 _, ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    ι✝ : Type u_3
    ε : Real
    hε : LT.lt 0 ε
    ι : Type ?u.145107
    inst✝ : Encodable ι
    f : Nat → Real := fun n => HDiv.hDiv (HDiv.hDiv ε 2) (HPow.hPow 2 n)
    hf : HasSum f ε
    f0 : ∀ (n : Nat), LT.lt 0 (f n)
    ⊢ Exists fun c => And (HasSum (Function.comp f Encodable.encode) c) (LE.le c ε)
  -/
  rcases hf.summable.comp_injective (@Encodable.encode_injective ι _) with ⟨c, hg⟩
  /-
    case intro
    α : Type u_1
    β : Type u_2
    ι✝ : Type u_3
    ε : Real
    hε : LT.lt 0 ε
    ι : Type ?u.145107
    inst✝ : Encodable ι
    f : Nat → Real := fun n => HDiv.hDiv (HDiv.hDiv ε 2) (HPow.hPow 2 n)
    hf : HasSum f ε
    f0 : ∀ (n : Nat), LT.lt 0 (f n)
    c : Real
    hg : HasSum (Function.comp f Encodable.encode) c
    ⊢ Exists fun c => And (HasSum (Function.comp f Encodable.encode) c) (LE.le c ε)
  -/
  refine ⟨c, hg, hasSum_le_inj _ (@Encodable.encode_injective ι _) ?_ ?_ hg hf⟩
    /-
      case intro.refine_1
      α : Type u_1
      β : Type u_2
      ι✝ : Type u_3
      ε : Real
      hε : LT.lt 0 ε
      ι : Type ?u.145107
      inst✝ : Encodable ι
      f : Nat → Real := fun n => HDiv.hDiv (HDiv.hDiv ε 2) (HPow.hPow 2 n)
      hf : HasSum f ε
      f0 : ∀ (n : Nat), LT.lt 0 (f n)
      c : Real
      hg : HasSum (Function.comp f Encodable.encode) c
      ⊢ ∀ (c : Nat), Not (Membership.mem (Set.range Encodable.encode) c) → LE.le 0 ( …
    -/
  · intro i _
    /-
      case intro.refine_1
      α : Type u_1
      β : Type u_2
      ι✝ : Type u_3
      ε : Real
      hε : LT.lt 0 ε
      ι : Type ?u.145107
      inst✝ : Encodable ι
      f : Nat → Real := fun n => HDiv.hDiv (HDiv.hDiv ε 2) (HPow.hPow 2 n)
      hf : HasSum f ε
      f0 : ∀ (n : Nat), LT.lt 0 (f n)
      c : Real
      hg : HasSum (Function.comp f Encodable.encode) c
      i : Nat
      a✝ : Not (Membership.mem (Set.range Encodable.encode) i)
      ⊢ LE.le 0 (f i)
    -/
    exact le_of_lt (f0 _)
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_2
      α : Type u_1
      β : Type u_2
      ι✝ : Type u_3
      ε : Real
      hε : LT.lt 0 ε
      ι : Type ?u.145107
      inst✝ : Encodable ι
      f : Nat → Real := fun n => HDiv.hDiv (HDiv.hDiv ε 2) (HPow.hPow 2 n)
      hf : HasSum f ε
      f0 : ∀ (n : Nat), LT.lt 0 (f n)
      c : Real
      hg : HasSum (Function.comp f Encodable.encode) c
      ⊢ ∀ (i : ι), LE.le (Function.comp f Encodable.encode i) (f (Encodable.encode i))
    -/
  · intro n
    /-
      case intro.refine_2
      α : Type u_1
      β : Type u_2
      ι✝ : Type u_3
      ε : Real
      hε : LT.lt 0 ε
      ι : Type ?u.145107
      inst✝ : Encodable ι
      f : Nat → Real := fun n => HDiv.hDiv (HDiv.hDiv ε 2) (HPow.hPow 2 n)
      hf : HasSum f ε
      f0 : ∀ (n : Nat), LT.lt 0 (f n)
      c : Real
      hg : HasSum (Function.comp f Encodable.encode) c
      n : ι
      ⊢ LE.le (Function.comp f Encodable.encode n) (f (Encodable.encode n))
    -/
    exact le_rfl
    /-
      🎉 no goals
    -/


theorem Set.Countable.exists_pos_hasSum_le {ι : Type*} {s : Set ι} (hs : s.Countable) {ε : ℝ}
    (hε : 0 < ε) : ∃ ε' : ι → ℝ, (∀ i, 0 < ε' i) ∧ ∃ c, HasSum (fun i : s ↦ ε' i) c ∧ c ≤ ε := by
  classical
  haveI := hs.toEncodable
  rcases posSumOfEncodable hε s with ⟨f, hf0, ⟨c, hfc, hcε⟩⟩
  refine ⟨fun i ↦ if h : i ∈ s then f ⟨i, h⟩ else 1, fun i ↦ ?_, ⟨c, ?_, hcε⟩⟩
  · conv_rhs => simp
    split_ifs
    exacts [hf0 _, zero_lt_one]
  · simpa only [Subtype.coe_prop, dif_pos, Subtype.coe_eta]


theorem Set.Countable.exists_pos_forall_sum_le {ι : Type*} {s : Set ι} (hs : s.Countable) {ε : ℝ}
    (hε : 0 < ε) : ∃ ε' : ι → ℝ,
    (∀ i, 0 < ε' i) ∧ ∀ t : Finset ι, ↑t ⊆ s → ∑ i ∈ t, ε' i ≤ ε := by
  classical
  rcases hs.exists_pos_hasSum_le hε with ⟨ε', hpos, c, hε'c, hcε⟩
  refine ⟨ε', hpos, fun t ht ↦ ?_⟩
  rw [← sum_subtype_of_mem _ ht]
  refine (sum_le_hasSum _ ?_ hε'c).trans hcε
  exact fun _ _ ↦ (hpos _).le


theorem exists_pos_sum_of_countable {ε : ℝ≥0} (hε : ε ≠ 0) (ι) [Countable ι] :
    ∃ ε' : ι → ℝ≥0, (∀ i, 0 < ε' i) ∧ ∃ c, HasSum ε' c ∧ c < ε := by
  /-
    ε : NNReal
    hε : Ne ε 0
    ι : Type u_4
    inst✝ : Countable ι
    ⊢ Exists fun ε' => And (∀ (i : ι), LT.lt 0 (ε' i)) (Exists fun c => And (HasSu …
  -/
  cases nonempty_encodable ι
  /-
    case intro
    ε : NNReal
    hε : Ne ε 0
    ι : Type u_4
    inst✝ : Countable ι
    val✝ : Encodable ι
    ⊢ Exists fun ε' => And (∀ (i : ι), LT.lt 0 (ε' i)) (Exists fun c => And (HasSu …
  -/
  obtain ⟨a, a0, aε⟩ := exists_between (pos_iff_ne_zero.2 hε)
  /-
    case intro.intro.intro
    ε : NNReal
    hε : Ne ε 0
    ι : Type u_4
    inst✝ : Countable ι
    val✝ : Encodable ι
    a : NNReal
    a0 : LT.lt 0 a
    aε : LT.lt a ε
    ⊢ Exists fun ε' => And (∀ (i : ι), LT.lt 0 (ε' i)) (Exists fun c => And (HasSu …
  -/
  obtain ⟨ε', hε', c, hc, hcε⟩ := posSumOfEncodable a0 ι
  exact
    ⟨fun i ↦ ⟨ε' i, (hε' i).le⟩, fun i ↦ NNReal.coe_lt_coe.1 <| hε' i,
      ⟨c, hasSum_le (fun i ↦ (hε' i).le) hasSum_zero hc⟩, NNReal.hasSum_coe.1 hc,
      aε.trans_le' <| NNReal.coe_le_coe.1 hcε⟩


theorem exists_pos_sum_of_countable {ε : ℝ≥0∞} (hε : ε ≠ 0) (ι) [Countable ι] :
    ∃ ε' : ι → ℝ≥0, (∀ i, 0 < ε' i) ∧ (∑' i, (ε' i : ℝ≥0∞)) < ε := by
  /-
    ε : ENNReal
    hε : Ne ε 0
    ι : Type u_4
    inst✝ : Countable ι
    ⊢ Exists fun ε' => And (∀ (i : ι), LT.lt 0 (ε' i)) (LT.lt (tsum fun i => ↑(ε'  …
  -/
  rcases exists_between (pos_iff_ne_zero.2 hε) with ⟨r, h0r, hrε⟩
  /-
    case intro.intro
    ε : ENNReal
    hε : Ne ε 0
    ι : Type u_4
    inst✝ : Countable ι
    r : ENNReal
    h0r : LT.lt 0 r
    hrε : LT.lt r ε
    ⊢ Exists fun ε' => And (∀ (i : ι), LT.lt 0 (ε' i)) (LT.lt (tsum fun i => ↑(ε'  …
  -/
  rcases lt_iff_exists_coe.1 hrε with ⟨x, rfl, _⟩
  /-
    case intro.intro.intro.intro
    ε : ENNReal
    hε : Ne ε 0
    ι : Type u_4
    inst✝ : Countable ι
    x : NNReal
    right✝ : LT.lt (↑x) ε
    h0r : LT.lt 0 ↑x
    hrε : LT.lt (↑x) ε
    ⊢ Exists fun ε' => And (∀ (i : ι), LT.lt 0 (ε' i)) (LT.lt (tsum fun i => ↑(ε'  …
  -/
  rcases NNReal.exists_pos_sum_of_countable (coe_pos.1 h0r).ne' ι with ⟨ε', hp, c, hc, hcr⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    ε : ENNReal
    hε : Ne ε 0
    ι : Type u_4
    inst✝ : Countable ι
    x : NNReal
    right✝ : LT.lt (↑x) ε
    h0r : LT.lt 0 ↑x
    hrε : LT.lt (↑x) ε
    ε' : ι → NNReal
    hp : ∀ (i : ι), LT.lt 0 (ε' i)
    c : NNReal
    hc : HasSum ε' c
    hcr : LT.lt c x
    ⊢ Exists fun ε' => And (∀ (i : ι), LT.lt 0 (ε' i)) (LT.lt (tsum fun i => ↑(ε'  …
  -/
  exact ⟨ε', hp, (ENNReal.tsum_coe_eq hc).symm ▸ lt_trans (coe_lt_coe.2 hcr) hrε⟩
  /-
    🎉 no goals
  -/


theorem exists_pos_sum_of_countable' {ε : ℝ≥0∞} (hε : ε ≠ 0) (ι) [Countable ι] :
    ∃ ε' : ι → ℝ≥0∞, (∀ i, 0 < ε' i) ∧ ∑' i, ε' i < ε :=
  let ⟨δ, δpos, hδ⟩ := exists_pos_sum_of_countable hε ι
  ⟨fun i ↦ δ i, fun i ↦ ENNReal.coe_pos.2 (δpos i), hδ⟩


theorem exists_pos_tsum_mul_lt_of_countable {ε : ℝ≥0∞} (hε : ε ≠ 0) {ι} [Countable ι] (w : ι → ℝ≥0∞)
    (hw : ∀ i, w i ≠ ∞) : ∃ δ : ι → ℝ≥0, (∀ i, 0 < δ i) ∧ (∑' i, (w i * δ i : ℝ≥0∞)) < ε := by
  /-
    ε : ENNReal
    hε : Ne ε 0
    ι : Type u_4
    inst✝ : Countable ι
    w : ι → ENNReal
    hw : ∀ (i : ι), Ne (w i) Top.top
    ⊢ Exists fun δ => And (∀ (i : ι), LT.lt 0 (δ i)) (LT.lt (tsum fun i => HMul.hM …
  -/
  lift w to ι → ℝ≥0 using hw
  /-
    case intro
    ε : ENNReal
    hε : Ne ε 0
    ι : Type u_4
    inst✝ : Countable ι
    w : ι → NNReal
    ⊢ Exists fun δ => And (∀ (i : ι), LT.lt 0 (δ i)) (LT.lt (tsum fun i => HMul.hM …
  -/
  rcases exists_pos_sum_of_countable hε ι with ⟨δ', Hpos, Hsum⟩
  /-
    case intro.intro.intro
    ε : ENNReal
    hε : Ne ε 0
    ι : Type u_4
    inst✝ : Countable ι
    w δ' : ι → NNReal
    Hpos : ∀ (i : ι), LT.lt 0 (δ' i)
    Hsum : LT.lt (tsum fun i => ↑(δ' i)) ε
    ⊢ Exists fun δ => And (∀ (i : ι), LT.lt 0 (δ i)) (LT.lt (tsum fun i => HMul.hM …
  -/
  have : ∀ i, 0 < max 1 (w i) := fun i ↦ zero_lt_one.trans_le (le_max_left _ _)
  /-
    case intro.intro.intro
    ε : ENNReal
    hε : Ne ε 0
    ι : Type u_4
    inst✝ : Countable ι
    w δ' : ι → NNReal
    Hpos : ∀ (i : ι), LT.lt 0 (δ' i)
    Hsum : LT.lt (tsum fun i => ↑(δ' i)) ε
    this : ∀ (i : ι), LT.lt 0 (Max.max 1 (w i))
    ⊢ Exists fun δ => And (∀ (i : ι), LT.lt 0 (δ i)) (LT.lt (tsum fun i => HMul.hM …
  -/
  refine ⟨fun i ↦ δ' i / max 1 (w i), fun i ↦ div_pos (Hpos _) (this i), ?_⟩
  /-
    case intro.intro.intro
    ε : ENNReal
    hε : Ne ε 0
    ι : Type u_4
    inst✝ : Countable ι
    w δ' : ι → NNReal
    Hpos : ∀ (i : ι), LT.lt 0 (δ' i)
    Hsum : LT.lt (tsum fun i => ↑(δ' i)) ε
    this : ∀ (i : ι), LT.lt 0 (Max.max 1 (w i))
    ⊢ LT.lt (tsum fun i => HMul.hMul ((fun i => ↑(w i)) i) ↑((fun i => HDiv.hDiv ( …
  -/
  refine lt_of_le_of_lt (ENNReal.tsum_le_tsum fun i ↦ ?_) Hsum
  /-
    case intro.intro.intro
    ε : ENNReal
    hε : Ne ε 0
    ι : Type u_4
    inst✝ : Countable ι
    w δ' : ι → NNReal
    Hpos : ∀ (i : ι), LT.lt 0 (δ' i)
    Hsum : LT.lt (tsum fun i => ↑(δ' i)) ε
    this : ∀ (i : ι), LT.lt 0 (Max.max 1 (w i))
    i : ι
    ⊢ LE.le (HMul.hMul ((fun i => ↑(w i)) i) ↑((fun i => HDiv.hDiv (δ' i) (Max.max …
  -/
  rw [coe_div (this i).ne']
  /-
    case intro.intro.intro
    ε : ENNReal
    hε : Ne ε 0
    ι : Type u_4
    inst✝ : Countable ι
    w δ' : ι → NNReal
    Hpos : ∀ (i : ι), LT.lt 0 (δ' i)
    Hsum : LT.lt (tsum fun i => ↑(δ' i)) ε
    this : ∀ (i : ι), LT.lt 0 (Max.max 1 (w i))
    i : ι
    ⊢ LE.le (HMul.hMul ((fun i => ↑(w i)) i) (HDiv.hDiv ↑(δ' i) ↑(Max.max 1 (w i)) …
  -/
  refine mul_le_of_le_div' (mul_le_mul_left' (ENNReal.inv_le_inv.2 ?_) _)
  /-
    case intro.intro.intro
    ε : ENNReal
    hε : Ne ε 0
    ι : Type u_4
    inst✝ : Countable ι
    w δ' : ι → NNReal
    Hpos : ∀ (i : ι), LT.lt 0 (δ' i)
    Hsum : LT.lt (tsum fun i => ↑(δ' i)) ε
    this : ∀ (i : ι), LT.lt 0 (Max.max 1 (w i))
    i : ι
    ⊢ LE.le ((fun i => ↑(w i)) i) ↑(Max.max 1 (w i))
  -/
  exact coe_le_coe.2 (le_max_right _ _)
  /-
    🎉 no goals
  -/


theorem factorial_tendsto_atTop : Tendsto Nat.factorial atTop atTop :=
  tendsto_atTop_atTop_of_monotone (fun _ _ ↦ Nat.factorial_le) fun n ↦ ⟨n, n.self_le_factorial⟩


theorem tendsto_factorial_div_pow_self_atTop :
    Tendsto (fun n ↦ n ! / (n : ℝ) ^ n : ℕ → ℝ) atTop (𝓝 0) :=
  tendsto_of_tendsto_of_tendsto_of_le_of_le' tendsto_const_nhds
    (tendsto_const_div_atTop_nhds_zero_nat 1)
    (Eventually.of_forall fun n ↦
      div_nonneg (mod_cast n.factorial_pos.le)
        (pow_nonneg (mod_cast n.zero_le) _))
    (by
      /-
        ⊢ Filter.Eventually (fun b => LE.le (HDiv.hDiv (↑b.factorial) (HPow.hPow (↑b)  …
      -/
      refine (eventually_gt_atTop 0).mono fun n hn ↦ ?_
      /-
        n : Nat
        hn : LT.lt 0 n
        ⊢ LE.le (HDiv.hDiv (↑n.factorial) (HPow.hPow (↑n) n)) (HDiv.hDiv 1 ↑n)
      -/
      rcases Nat.exists_eq_succ_of_ne_zero hn.ne.symm with ⟨k, rfl⟩
      rw [← prod_range_add_one_eq_factorial, pow_eq_prod_const, div_eq_mul_inv, ← inv_eq_one_div,
        prod_natCast, Nat.cast_succ, ← Finset.prod_inv_distrib, ← prod_mul_distrib,
        Finset.prod_range_succ']
      /-
        case intro
        k : Nat
        hn : LT.lt 0 k.succ
        ⊢ LE.le (HMul.hMul ((Finset.range k).prod fun k_1 => HMul.hMul (↑(HAdd.hAdd (H …
      -/
      simp only [prod_range_succ', one_mul, Nat.cast_add, zero_add, Nat.cast_one]
      refine
            mul_le_of_le_one_left (inv_nonneg.mpr <| mod_cast hn.le) (prod_le_one ?_ ?_) <;>
          /-
            case intro.refine_1
            k : Nat
            hn : LT.lt 0 k.succ
            ⊢ ∀ (i : Nat), Membership.mem (Finset.range k) i → LE.le 0 (HMul.hMul (HAdd.hA …
          -/
          intro x hx <;>
        /-
          case intro.refine_1
          k : Nat
          hn : LT.lt 0 k.succ
          x : Nat
          hx : Membership.mem (Finset.range k) x
          ⊢ LE.le 0 (HMul.hMul (HAdd.hAdd (HAdd.hAdd (↑x) 1) 1) (Inv.inv (HAdd.hAdd (↑k) …
        -/
        rw [Finset.mem_range] at hx
        /-
          case intro.refine_1
          k : Nat
          hn : LT.lt 0 k.succ
          x : Nat
          hx : LT.lt x k
          ⊢ LE.le 0 (HMul.hMul (HAdd.hAdd (HAdd.hAdd (↑x) 1) 1) (Inv.inv (HAdd.hAdd (↑k) …
        -/
      · positivity
        /-
          🎉 no goals
        -/
        /-
          case intro.refine_2
          k : Nat
          hn : LT.lt 0 k.succ
          x : Nat
          hx : LT.lt x k
          ⊢ LE.le (HMul.hMul (HAdd.hAdd (HAdd.hAdd (↑x) 1) 1) (Inv.inv (HAdd.hAdd (↑k) 1 …
        -/
      · refine (div_le_one <| mod_cast hn).mpr ?_
        /-
          case intro.refine_2
          k : Nat
          hn : LT.lt 0 k.succ
          x : Nat
          hx : LT.lt x k
          ⊢ LE.le (HAdd.hAdd (HAdd.hAdd (↑x) 1) 1) (HAdd.hAdd (↑k) 1)
        -/
        norm_cast
        /-
          case intro.refine_2
          k : Nat
          hn : LT.lt 0 k.succ
          x : Nat
          hx : LT.lt x k
          ⊢ LE.le (HAdd.hAdd (HAdd.hAdd x 1) 1) (HAdd.hAdd k 1)
        -/
        omega)
        /-
          🎉 no goals
        -/


theorem tendsto_nat_floor_atTop {α : Type*} [LinearOrderedSemiring α] [FloorSemiring α] :
    Tendsto (fun x : α ↦ ⌊x⌋₊) atTop atTop :=
                                                                /-
                                                                  α : Type u_4
                                                                  inst✝¹ : LinearOrderedSemiring α
                                                                  inst✝ : FloorSemiring α
                                                                  x : Nat
                                                                  ⊢ LE.le x (Nat.floor (Max.max 0 (HAdd.hAdd (↑x) 1)))
                                                                -/
  Nat.floor_mono.tendsto_atTop_atTop fun x ↦ ⟨max 0 (x + 1), by simp [Nat.le_floor_iff]⟩
                                                                /-
                                                                  🎉 no goals
                                                                -/


lemma tendsto_nat_ceil_atTop {α : Type*} [LinearOrderedSemiring α] [FloorSemiring α] :
    Tendsto (fun x : α ↦ ⌈x⌉₊) atTop atTop := by
  /-
    α : Type u_4
    inst✝¹ : LinearOrderedSemiring α
    inst✝ : FloorSemiring α
    ⊢ Filter.Tendsto (fun x => Nat.ceil x) Filter.atTop Filter.atTop
  -/
  refine Nat.ceil_mono.tendsto_atTop_atTop (fun x ↦ ⟨x, ?_⟩)
  /-
    α : Type u_4
    inst✝¹ : LinearOrderedSemiring α
    inst✝ : FloorSemiring α
    x : Nat
    ⊢ LE.le x (Nat.ceil ↑x)
  -/
  simp only [Nat.ceil_natCast, le_refl]
  /-
    🎉 no goals
  -/


lemma tendsto_nat_floor_mul_atTop {α : Type _} [LinearOrderedSemifield α] [FloorSemiring α]
    [Archimedean α] (a : α) (ha : 0 < a) : Tendsto (fun (x : ℕ) => ⌊a * x⌋₊) atTop atTop :=
  Tendsto.comp tendsto_nat_floor_atTop
    <| Tendsto.const_mul_atTop ha tendsto_natCast_atTop_atTop


theorem tendsto_nat_floor_mul_div_atTop {a : R} (ha : 0 ≤ a) :
    Tendsto (fun x ↦ (⌊a * x⌋₊ : R) / x) atTop (𝓝 a) := by
  have A : Tendsto (fun x : R ↦ a - x⁻¹) atTop (𝓝 (a - 0)) :=
    tendsto_const_nhds.sub tendsto_inv_atTop_zero
  /-
    R : Type u_4
    inst✝³ : TopologicalSpace R
    inst✝² : LinearOrderedField R
    inst✝¹ : OrderTopology R
    inst✝ : FloorRing R
    a : R
    ha : LE.le 0 a
    A : Filter.Tendsto (fun x => HSub.hSub a (Inv.inv x)) Filter.atTop (nhds (HSub …
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (↑(Nat.floor (HMul.hMul a x))) x) Filter. …
  -/
  rw [sub_zero] at A
  /-
    R : Type u_4
    inst✝³ : TopologicalSpace R
    inst✝² : LinearOrderedField R
    inst✝¹ : OrderTopology R
    inst✝ : FloorRing R
    a : R
    ha : LE.le 0 a
    A : Filter.Tendsto (fun x => HSub.hSub a (Inv.inv x)) Filter.atTop (nhds a)
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (↑(Nat.floor (HMul.hMul a x))) x) Filter. …
  -/
  apply tendsto_of_tendsto_of_tendsto_of_le_of_le' A tendsto_const_nhds
    /-
      case hgf
      R : Type u_4
      inst✝³ : TopologicalSpace R
      inst✝² : LinearOrderedField R
      inst✝¹ : OrderTopology R
      inst✝ : FloorRing R
      a : R
      ha : LE.le 0 a
      A : Filter.Tendsto (fun x => HSub.hSub a (Inv.inv x)) Filter.atTop (nhds a)
      ⊢ Filter.Eventually (fun b => LE.le (HSub.hSub a (Inv.inv b)) (HDiv.hDiv (↑(Na …
    -/
  · refine eventually_atTop.2 ⟨1, fun x hx ↦ ?_⟩
    simp only [le_div_iff₀ (zero_lt_one.trans_le hx), _root_.sub_mul,
      inv_mul_cancel₀ (zero_lt_one.trans_le hx).ne']
    /-
      case hgf
      R : Type u_4
      inst✝³ : TopologicalSpace R
      inst✝² : LinearOrderedField R
      inst✝¹ : OrderTopology R
      inst✝ : FloorRing R
      a : R
      ha : LE.le 0 a
      A : Filter.Tendsto (fun x => HSub.hSub a (Inv.inv x)) Filter.atTop (nhds a)
      x : R
      hx : GE.ge x 1
      ⊢ LE.le (HSub.hSub (HMul.hMul a x) 1) ↑(Nat.floor (HMul.hMul a x))
    -/
    have := Nat.lt_floor_add_one (a * x)
    /-
      case hgf
      R : Type u_4
      inst✝³ : TopologicalSpace R
      inst✝² : LinearOrderedField R
      inst✝¹ : OrderTopology R
      inst✝ : FloorRing R
      a : R
      ha : LE.le 0 a
      A : Filter.Tendsto (fun x => HSub.hSub a (Inv.inv x)) Filter.atTop (nhds a)
      x : R
      hx : GE.ge x 1
      this : LT.lt (HMul.hMul a x) (HAdd.hAdd (↑(Nat.floor (HMul.hMul a x))) 1)
      ⊢ LE.le (HSub.hSub (HMul.hMul a x) 1) ↑(Nat.floor (HMul.hMul a x))
    -/
    linarith
    /-
      🎉 no goals
    -/
    /-
      case hfh
      R : Type u_4
      inst✝³ : TopologicalSpace R
      inst✝² : LinearOrderedField R
      inst✝¹ : OrderTopology R
      inst✝ : FloorRing R
      a : R
      ha : LE.le 0 a
      A : Filter.Tendsto (fun x => HSub.hSub a (Inv.inv x)) Filter.atTop (nhds a)
      ⊢ Filter.Eventually (fun b => LE.le (HDiv.hDiv (↑(Nat.floor (HMul.hMul a b)))  …
    -/
  · refine eventually_atTop.2 ⟨1, fun x hx ↦ ?_⟩
    /-
      case hfh
      R : Type u_4
      inst✝³ : TopologicalSpace R
      inst✝² : LinearOrderedField R
      inst✝¹ : OrderTopology R
      inst✝ : FloorRing R
      a : R
      ha : LE.le 0 a
      A : Filter.Tendsto (fun x => HSub.hSub a (Inv.inv x)) Filter.atTop (nhds a)
      x : R
      hx : GE.ge x 1
      ⊢ LE.le (HDiv.hDiv (↑(Nat.floor (HMul.hMul a x))) x) a
    -/
    rw [div_le_iff₀ (zero_lt_one.trans_le hx)]
    /-
      case hfh
      R : Type u_4
      inst✝³ : TopologicalSpace R
      inst✝² : LinearOrderedField R
      inst✝¹ : OrderTopology R
      inst✝ : FloorRing R
      a : R
      ha : LE.le 0 a
      A : Filter.Tendsto (fun x => HSub.hSub a (Inv.inv x)) Filter.atTop (nhds a)
      x : R
      hx : GE.ge x 1
      ⊢ LE.le (↑(Nat.floor (HMul.hMul a x))) (HMul.hMul a x)
    -/
    simp [Nat.floor_le (mul_nonneg ha (zero_le_one.trans hx))]
    /-
      🎉 no goals
    -/


theorem tendsto_nat_floor_div_atTop : Tendsto (fun x ↦ (⌊x⌋₊ : R) / x) atTop (𝓝 1) := by
  /-
    R : Type u_4
    inst✝³ : TopologicalSpace R
    inst✝² : LinearOrderedField R
    inst✝¹ : OrderTopology R
    inst✝ : FloorRing R
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (↑(Nat.floor x)) x) Filter.atTop (nhds 1)
  -/
  simpa using tendsto_nat_floor_mul_div_atTop (zero_le_one' R)
  /-
    🎉 no goals
  -/


theorem tendsto_nat_ceil_mul_div_atTop {a : R} (ha : 0 ≤ a) :
    Tendsto (fun x ↦ (⌈a * x⌉₊ : R) / x) atTop (𝓝 a) := by
  have A : Tendsto (fun x : R ↦ a + x⁻¹) atTop (𝓝 (a + 0)) :=
    tendsto_const_nhds.add tendsto_inv_atTop_zero
  /-
    R : Type u_4
    inst✝³ : TopologicalSpace R
    inst✝² : LinearOrderedField R
    inst✝¹ : OrderTopology R
    inst✝ : FloorRing R
    a : R
    ha : LE.le 0 a
    A : Filter.Tendsto (fun x => HAdd.hAdd a (Inv.inv x)) Filter.atTop (nhds (HAdd …
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (↑(Nat.ceil (HMul.hMul a x))) x) Filter.a …
  -/
  rw [add_zero] at A
  /-
    R : Type u_4
    inst✝³ : TopologicalSpace R
    inst✝² : LinearOrderedField R
    inst✝¹ : OrderTopology R
    inst✝ : FloorRing R
    a : R
    ha : LE.le 0 a
    A : Filter.Tendsto (fun x => HAdd.hAdd a (Inv.inv x)) Filter.atTop (nhds a)
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (↑(Nat.ceil (HMul.hMul a x))) x) Filter.a …
  -/
  apply tendsto_of_tendsto_of_tendsto_of_le_of_le' tendsto_const_nhds A
    /-
      case hgf
      R : Type u_4
      inst✝³ : TopologicalSpace R
      inst✝² : LinearOrderedField R
      inst✝¹ : OrderTopology R
      inst✝ : FloorRing R
      a : R
      ha : LE.le 0 a
      A : Filter.Tendsto (fun x => HAdd.hAdd a (Inv.inv x)) Filter.atTop (nhds a)
      ⊢ Filter.Eventually (fun b => LE.le a (HDiv.hDiv (↑(Nat.ceil (HMul.hMul a b))) …
    -/
  · refine eventually_atTop.2 ⟨1, fun x hx ↦ ?_⟩
    /-
      case hgf
      R : Type u_4
      inst✝³ : TopologicalSpace R
      inst✝² : LinearOrderedField R
      inst✝¹ : OrderTopology R
      inst✝ : FloorRing R
      a : R
      ha : LE.le 0 a
      A : Filter.Tendsto (fun x => HAdd.hAdd a (Inv.inv x)) Filter.atTop (nhds a)
      x : R
      hx : GE.ge x 1
      ⊢ LE.le a (HDiv.hDiv (↑(Nat.ceil (HMul.hMul a x))) x)
    -/
    rw [le_div_iff₀ (zero_lt_one.trans_le hx)]
    /-
      case hgf
      R : Type u_4
      inst✝³ : TopologicalSpace R
      inst✝² : LinearOrderedField R
      inst✝¹ : OrderTopology R
      inst✝ : FloorRing R
      a : R
      ha : LE.le 0 a
      A : Filter.Tendsto (fun x => HAdd.hAdd a (Inv.inv x)) Filter.atTop (nhds a)
      x : R
      hx : GE.ge x 1
      ⊢ LE.le (HMul.hMul a x) ↑(Nat.ceil (HMul.hMul a x))
    -/
    exact Nat.le_ceil _
    /-
      🎉 no goals
    -/
    /-
      case hfh
      R : Type u_4
      inst✝³ : TopologicalSpace R
      inst✝² : LinearOrderedField R
      inst✝¹ : OrderTopology R
      inst✝ : FloorRing R
      a : R
      ha : LE.le 0 a
      A : Filter.Tendsto (fun x => HAdd.hAdd a (Inv.inv x)) Filter.atTop (nhds a)
      ⊢ Filter.Eventually (fun b => LE.le (HDiv.hDiv (↑(Nat.ceil (HMul.hMul a b))) b …
    -/
  · refine eventually_atTop.2 ⟨1, fun x hx ↦ ?_⟩
    simp [div_le_iff₀ (zero_lt_one.trans_le hx), inv_mul_cancel₀ (zero_lt_one.trans_le hx).ne',
      (Nat.ceil_lt_add_one (mul_nonneg ha (zero_le_one.trans hx))).le, add_mul]


theorem tendsto_nat_ceil_div_atTop : Tendsto (fun x ↦ (⌈x⌉₊ : R) / x) atTop (𝓝 1) := by
  /-
    R : Type u_4
    inst✝³ : TopologicalSpace R
    inst✝² : LinearOrderedField R
    inst✝¹ : OrderTopology R
    inst✝ : FloorRing R
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (↑(Nat.ceil x)) x) Filter.atTop (nhds 1)
  -/
  simpa using tendsto_nat_ceil_mul_div_atTop (zero_le_one' R)
  /-
    🎉 no goals
  -/


lemma Nat.tendsto_div_const_atTop {n : ℕ} (hn : n ≠ 0) : Tendsto (· / n) atTop atTop := by
  /-
    n : Nat
    hn : Ne n 0
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv x n) Filter.atTop Filter.atTop
  -/
  rw [Tendsto, map_div_atTop_eq_nat n hn.bot_lt]
  /-
    🎉 no goals
  -/


