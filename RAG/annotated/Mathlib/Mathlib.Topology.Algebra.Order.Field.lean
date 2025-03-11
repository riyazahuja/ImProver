/-- If a (possibly non-unital and/or non-associative) ring `R` admits a submultiplicative
nonnegative norm `norm : R → 𝕜`, where `𝕜` is a linear ordered field, and the open balls
`{ x | norm x < ε }`, `ε > 0`, form a basis of neighborhoods of zero, then `R` is a topological
ring. -/
theorem TopologicalRing.of_norm {R 𝕜 : Type*} [NonUnitalNonAssocRing R] [LinearOrderedField 𝕜]
    [TopologicalSpace R] [TopologicalAddGroup R] (norm : R → 𝕜)
    (norm_nonneg : ∀ x, 0 ≤ norm x) (norm_mul_le : ∀ x y, norm (x * y) ≤ norm x * norm y)
    (nhds_basis : (𝓝 (0 : R)).HasBasis ((0 : 𝕜) < ·) (fun ε ↦ { x | norm x < ε })) :
    TopologicalRing R := by
  have h0 : ∀ f : R → R, ∀ c ≥ (0 : 𝕜), (∀ x, norm (f x) ≤ c * norm x) →
      Tendsto f (𝓝 0) (𝓝 0) := by
    refine fun f c c0 hf ↦ (nhds_basis.tendsto_iff nhds_basis).2 fun ε ε0 ↦ ?_
    rcases exists_pos_mul_lt ε0 c with ⟨δ, δ0, hδ⟩
    refine ⟨δ, δ0, fun x hx ↦ (hf _).trans_lt ?_⟩
    exact (mul_le_mul_of_nonneg_left (le_of_lt hx) c0).trans_lt hδ
  /-
    R : Type u_1
    𝕜 : Type u_2
    inst✝³ : NonUnitalNonAssocRing R
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace R
    inst✝ : TopologicalAddGroup R
    norm : R → 𝕜
    norm_nonneg : ∀ (x : R), LE.le 0 (norm x)
    norm_mul_le : ∀ (x y : R), LE.le (norm (HMul.hMul x y)) (HMul.hMul (norm x) (n …
    nhds_basis : (nhds 0).HasBasis (fun x => LT.lt 0 x) fun ε => setOf fun x => LT …
    h0 : ∀ (f : R → R) (c : 𝕜), GE.ge c 0 → (∀ (x : R), LE.le (norm (f x)) (HMul.h …
    ⊢ TopologicalRing R
  -/
  apply TopologicalRing.of_addGroup_of_nhds_zero
  case hmul =>
    refine ((nhds_basis.prod nhds_basis).tendsto_iff nhds_basis).2 fun ε ε0 ↦ ?_
    refine ⟨(1, ε), ⟨one_pos, ε0⟩, fun (x, y) ⟨hx, hy⟩ => ?_⟩
    simp only [sub_zero] at *
    calc norm (x * y) ≤ norm x * norm y := norm_mul_le _ _
    _ < ε := mul_lt_of_le_one_of_lt_of_nonneg hx.le hy (norm_nonneg _)
  /-
    case hmul_left
    R : Type u_1
    𝕜 : Type u_2
    inst✝³ : NonUnitalNonAssocRing R
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace R
    inst✝ : TopologicalAddGroup R
    norm : R → 𝕜
    norm_nonneg : ∀ (x : R), LE.le 0 (norm x)
    norm_mul_le : ∀ (x y : R), LE.le (norm (HMul.hMul x y)) (HMul.hMul (norm x) (n …
    nhds_basis : (nhds 0).HasBasis (fun x => LT.lt 0 x) fun ε => setOf fun x => LT …
    h0 : ∀ (f : R → R) (c : 𝕜), GE.ge c 0 → (∀ (x : R), LE.le (norm (f x)) (HMul.h …
    ⊢ ∀ (x₀ : R), Filter.Tendsto (fun x => HMul.hMul x₀ x) (nhds 0) (nhds 0)
  -/
  case hmul_left => exact fun x => h0 _ (norm x) (norm_nonneg _) (norm_mul_le x)
  case hmul_right =>
    exact fun y => h0 (· * y) (norm y) (norm_nonneg y) fun x =>
      (norm_mul_le x y).trans_eq (mul_comm _ _)


instance (priority := 100) LinearOrderedField.topologicalRing : TopologicalRing 𝕜 :=
  .of_norm abs abs_nonneg (fun _ _ ↦ (abs_mul _ _).le) <| by
    /-
      𝕜 : Type u_1
      α : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : TopologicalSpace 𝕜
      inst✝ : OrderTopology 𝕜
      l : Filter α
      f g : α → 𝕜
      ⊢ (nhds 0).HasBasis (fun x => LT.lt 0 x) fun ε => setOf fun x => LT.lt (abs x) ε
    -/
    simpa using nhds_basis_abs_sub_lt (0 : 𝕜)
    /-
      🎉 no goals
    -/


/-- In a linearly ordered field with the order topology, if `f` tends to `Filter.atTop` and `g`
tends to a positive constant `C` then `f * g` tends to `Filter.atTop`. -/
theorem Filter.Tendsto.atTop_mul {C : 𝕜} (hC : 0 < C) (hf : Tendsto f l atTop)
    (hg : Tendsto g l (𝓝 C)) : Tendsto (fun x => f x * g x) l atTop := by
  /-
    𝕜 : Type u_1
    α : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    l : Filter α
    f g : α → 𝕜
    C : 𝕜
    hC : LT.lt 0 C
    hf : Filter.Tendsto f l Filter.atTop
    hg : Filter.Tendsto g l (nhds C)
    ⊢ Filter.Tendsto (fun x => HMul.hMul (f x) (g x)) l Filter.atTop
  -/
  refine tendsto_atTop_mono' _ ?_ (hf.atTop_mul_const (half_pos hC))
  filter_upwards [hg.eventually (lt_mem_nhds (half_lt_self hC)), hf.eventually_ge_atTop 0]
    with x hg hf using mul_le_mul_of_nonneg_left hg.le hf


/-- In a linearly ordered field with the order topology, if `f` tends to a positive constant `C` and
`g` tends to `Filter.atTop` then `f * g` tends to `Filter.atTop`. -/
theorem Filter.Tendsto.mul_atTop {C : 𝕜} (hC : 0 < C) (hf : Tendsto f l (𝓝 C))
    (hg : Tendsto g l atTop) : Tendsto (fun x => f x * g x) l atTop := by
  /-
    𝕜 : Type u_1
    α : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    l : Filter α
    f g : α → 𝕜
    C : 𝕜
    hC : LT.lt 0 C
    hf : Filter.Tendsto f l (nhds C)
    hg : Filter.Tendsto g l Filter.atTop
    ⊢ Filter.Tendsto (fun x => HMul.hMul (f x) (g x)) l Filter.atTop
  -/
  simpa only [mul_comm] using hg.atTop_mul hC hf
  /-
    🎉 no goals
  -/


/-- In a linearly ordered field with the order topology, if `f` tends to `Filter.atTop` and `g`
tends to a negative constant `C` then `f * g` tends to `Filter.atBot`. -/
theorem Filter.Tendsto.atTop_mul_neg {C : 𝕜} (hC : C < 0) (hf : Tendsto f l atTop)
    (hg : Tendsto g l (𝓝 C)) : Tendsto (fun x => f x * g x) l atBot := by
  /-
    𝕜 : Type u_1
    α : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    l : Filter α
    f g : α → 𝕜
    C : 𝕜
    hC : LT.lt C 0
    hf : Filter.Tendsto f l Filter.atTop
    hg : Filter.Tendsto g l (nhds C)
    ⊢ Filter.Tendsto (fun x => HMul.hMul (f x) (g x)) l Filter.atBot
  -/
  have := hf.atTop_mul (neg_pos.2 hC) hg.neg
  simpa only [Function.comp_def, neg_mul_eq_mul_neg, neg_neg] using
    tendsto_neg_atTop_atBot.comp this


/-- In a linearly ordered field with the order topology, if `f` tends to a negative constant `C` and
`g` tends to `Filter.atTop` then `f * g` tends to `Filter.atBot`. -/
theorem Filter.Tendsto.neg_mul_atTop {C : 𝕜} (hC : C < 0) (hf : Tendsto f l (𝓝 C))
    (hg : Tendsto g l atTop) : Tendsto (fun x => f x * g x) l atBot := by
  /-
    𝕜 : Type u_1
    α : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    l : Filter α
    f g : α → 𝕜
    C : 𝕜
    hC : LT.lt C 0
    hf : Filter.Tendsto f l (nhds C)
    hg : Filter.Tendsto g l Filter.atTop
    ⊢ Filter.Tendsto (fun x => HMul.hMul (f x) (g x)) l Filter.atBot
  -/
  simpa only [mul_comm] using hg.atTop_mul_neg hC hf
  /-
    🎉 no goals
  -/


/-- In a linearly ordered field with the order topology, if `f` tends to `Filter.atBot` and `g`
tends to a positive constant `C` then `f * g` tends to `Filter.atBot`. -/
theorem Filter.Tendsto.atBot_mul {C : 𝕜} (hC : 0 < C) (hf : Tendsto f l atBot)
    (hg : Tendsto g l (𝓝 C)) : Tendsto (fun x => f x * g x) l atBot := by
  /-
    𝕜 : Type u_1
    α : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    l : Filter α
    f g : α → 𝕜
    C : 𝕜
    hC : LT.lt 0 C
    hf : Filter.Tendsto f l Filter.atBot
    hg : Filter.Tendsto g l (nhds C)
    ⊢ Filter.Tendsto (fun x => HMul.hMul (f x) (g x)) l Filter.atBot
  -/
  have := (tendsto_neg_atBot_atTop.comp hf).atTop_mul hC hg
  /-
    𝕜 : Type u_1
    α : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    l : Filter α
    f g : α → 𝕜
    C : 𝕜
    hC : LT.lt 0 C
    hf : Filter.Tendsto f l Filter.atBot
    hg : Filter.Tendsto g l (nhds C)
    this : Filter.Tendsto (fun x => HMul.hMul (Function.comp Neg.neg f x) (g x)) l …
    ⊢ Filter.Tendsto (fun x => HMul.hMul (f x) (g x)) l Filter.atBot
  -/
  simpa [Function.comp_def] using tendsto_neg_atTop_atBot.comp this
  /-
    🎉 no goals
  -/


/-- In a linearly ordered field with the order topology, if `f` tends to `Filter.atBot` and `g`
tends to a negative constant `C` then `f * g` tends to `Filter.atTop`. -/
theorem Filter.Tendsto.atBot_mul_neg {C : 𝕜} (hC : C < 0) (hf : Tendsto f l atBot)
    (hg : Tendsto g l (𝓝 C)) : Tendsto (fun x => f x * g x) l atTop := by
  /-
    𝕜 : Type u_1
    α : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    l : Filter α
    f g : α → 𝕜
    C : 𝕜
    hC : LT.lt C 0
    hf : Filter.Tendsto f l Filter.atBot
    hg : Filter.Tendsto g l (nhds C)
    ⊢ Filter.Tendsto (fun x => HMul.hMul (f x) (g x)) l Filter.atTop
  -/
  have := (tendsto_neg_atBot_atTop.comp hf).atTop_mul_neg hC hg
  /-
    𝕜 : Type u_1
    α : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    l : Filter α
    f g : α → 𝕜
    C : 𝕜
    hC : LT.lt C 0
    hf : Filter.Tendsto f l Filter.atBot
    hg : Filter.Tendsto g l (nhds C)
    this : Filter.Tendsto (fun x => HMul.hMul (Function.comp Neg.neg f x) (g x)) l …
    ⊢ Filter.Tendsto (fun x => HMul.hMul (f x) (g x)) l Filter.atTop
  -/
  simpa [Function.comp_def] using tendsto_neg_atBot_atTop.comp this
  /-
    🎉 no goals
  -/


/-- In a linearly ordered field with the order topology, if `f` tends to a positive constant `C` and
`g` tends to `Filter.atBot` then `f * g` tends to `Filter.atBot`. -/
theorem Filter.Tendsto.mul_atBot {C : 𝕜} (hC : 0 < C) (hf : Tendsto f l (𝓝 C))
    (hg : Tendsto g l atBot) : Tendsto (fun x => f x * g x) l atBot := by
  /-
    𝕜 : Type u_1
    α : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    l : Filter α
    f g : α → 𝕜
    C : 𝕜
    hC : LT.lt 0 C
    hf : Filter.Tendsto f l (nhds C)
    hg : Filter.Tendsto g l Filter.atBot
    ⊢ Filter.Tendsto (fun x => HMul.hMul (f x) (g x)) l Filter.atBot
  -/
  simpa only [mul_comm] using hg.atBot_mul hC hf
  /-
    🎉 no goals
  -/


/-- In a linearly ordered field with the order topology, if `f` tends to a negative constant `C` and
`g` tends to `Filter.atBot` then `f * g` tends to `Filter.atTop`. -/
theorem Filter.Tendsto.neg_mul_atBot {C : 𝕜} (hC : C < 0) (hf : Tendsto f l (𝓝 C))
    (hg : Tendsto g l atBot) : Tendsto (fun x => f x * g x) l atTop := by
  /-
    𝕜 : Type u_1
    α : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    l : Filter α
    f g : α → 𝕜
    C : 𝕜
    hC : LT.lt C 0
    hf : Filter.Tendsto f l (nhds C)
    hg : Filter.Tendsto g l Filter.atBot
    ⊢ Filter.Tendsto (fun x => HMul.hMul (f x) (g x)) l Filter.atTop
  -/
  simpa only [mul_comm] using hg.atBot_mul_neg hC hf
  /-
    🎉 no goals
  -/


@[simp]
lemma inv_atTop₀ : (atTop : Filter 𝕜)⁻¹ = 𝓝[>] 0 :=
  (((atTop_basis_Ioi' (0 : 𝕜)).map _).comp_surjective inv_surjective).eq_of_same_basis <|
                               /-
                                 𝕜 : Type u_1
                                 inst✝² : LinearOrderedField 𝕜
                                 inst✝¹ : TopologicalSpace 𝕜
                                 inst✝ : OrderTopology 𝕜
                                 ⊢ ∀ (i : 𝕜), Iff (LT.lt 0 i) (Function.comp (fun x => LT.lt 0 x) Inv.inv i)
                               -/
                               /-
                                 🎉 no goals
                               -/
    (nhdsGT_basis _).congr (by simp) fun a ha ↦ by simp [inv_Ioi₀ (inv_pos.2 ha)]
                                                   /-
                                                     🎉 no goals
                                                   -/


@[simp]
                                                       /-
                                                         𝕜 : Type u_1
                                                         inst✝² : LinearOrderedField 𝕜
                                                         inst✝¹ : TopologicalSpace 𝕜
                                                         inst✝ : OrderTopology 𝕜
                                                         ⊢ Eq (Inv.inv (nhdsWithin 0 (Set.Ioi 0))) Filter.atTop
                                                       -/
lemma inv_nhdsGT_zero : (𝓝[>] (0 : 𝕜))⁻¹ = atTop := by rw [← inv_atTop₀, inv_inv]
                                                       /-
                                                         🎉 no goals
                                                       -/


@[deprecated (since := "2024-12-22")] alias inv_nhdsWithin_Ioi_zero := inv_nhdsGT_zero


/-- The function `x ↦ x⁻¹` tends to `+∞` on the right of `0`. -/
theorem tendsto_inv_nhdsGT_zero : Tendsto (fun x : 𝕜 => x⁻¹) (𝓝[>] (0 : 𝕜)) atTop :=
  inv_nhdsGT_zero.le


@[deprecated (since := "2024-12-22")]
alias tendsto_inv_zero_atTop := tendsto_inv_nhdsGT_zero


/-- The function `r ↦ r⁻¹` tends to `0` on the right as `r → +∞`. -/
theorem tendsto_inv_atTop_nhdsGT_zero : Tendsto (fun r : 𝕜 => r⁻¹) atTop (𝓝[>] (0 : 𝕜)) :=
  inv_atTop₀.le


@[deprecated (since := "2024-12-22")]
alias tendsto_inv_atTop_zero' := tendsto_inv_atTop_nhdsGT_zero


theorem tendsto_inv_atTop_zero : Tendsto (fun r : 𝕜 => r⁻¹) atTop (𝓝 0) :=
  tendsto_inv_atTop_nhdsGT_zero.mono_right inf_le_left


theorem Filter.Tendsto.div_atTop {a : 𝕜} (h : Tendsto f l (𝓝 a)) (hg : Tendsto g l atTop) :
    Tendsto (fun x => f x / g x) l (𝓝 0) := by
  /-
    𝕜 : Type u_1
    α : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    l : Filter α
    f g : α → 𝕜
    a : 𝕜
    h : Filter.Tendsto f l (nhds a)
    hg : Filter.Tendsto g l Filter.atTop
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) l (nhds 0)
  -/
  simp only [div_eq_mul_inv]
  /-
    𝕜 : Type u_1
    α : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    l : Filter α
    f g : α → 𝕜
    a : 𝕜
    h : Filter.Tendsto f l (nhds a)
    hg : Filter.Tendsto g l Filter.atTop
    ⊢ Filter.Tendsto (fun x => HMul.hMul (f x) (Inv.inv (g x))) l (nhds 0)
  -/
  exact mul_zero a ▸ h.mul (tendsto_inv_atTop_zero.comp hg)
  /-
    🎉 no goals
  -/


lemma Filter.Tendsto.const_div_atTop (hg : Tendsto g l atTop) (r : 𝕜)  :
    Tendsto (fun n ↦ r / g n) l (𝓝 0) :=
  tendsto_const_nhds.div_atTop hg


theorem Filter.Tendsto.inv_tendsto_atTop (h : Tendsto f l atTop) : Tendsto f⁻¹ l (𝓝 0) :=
  tendsto_inv_atTop_zero.comp h


theorem Filter.Tendsto.inv_tendsto_nhdsGT_zero (h : Tendsto f l (𝓝[>] 0)) : Tendsto f⁻¹ l atTop :=
  tendsto_inv_nhdsGT_zero.comp h


@[deprecated (since := "2024-12-22")]
alias Filter.Tendsto.inv_tendsto_zero := Filter.Tendsto.inv_tendsto_nhdsGT_zero


/-- If `g` tends to zero and there exists a constant `C : 𝕜` such that eventually `|f x| ≤ C`,
  then the product `f * g` tends to zero. -/
theorem bdd_le_mul_tendsto_zero' {f g : α → 𝕜} (C : 𝕜) (hf : ∀ᶠ x in l, |f x| ≤ C)
    (hg : Tendsto g l (𝓝 0)) : Tendsto (fun x ↦ f x * g x) l (𝓝 0) := by
  /-
    𝕜 : Type u_1
    α : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    l : Filter α
    f g : α → 𝕜
    C : 𝕜
    hf : Filter.Eventually (fun x => LE.le (abs (f x)) C) l
    hg : Filter.Tendsto g l (nhds 0)
    ⊢ Filter.Tendsto (fun x => HMul.hMul (f x) (g x)) l (nhds 0)
  -/
  rw [tendsto_zero_iff_abs_tendsto_zero]
  have hC : Tendsto (fun x ↦ |C * g x|) l (𝓝 0) := by
    convert (hg.const_mul C).abs
    simp_rw [mul_zero, abs_zero]
  /-
    𝕜 : Type u_1
    α : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    l : Filter α
    f g : α → 𝕜
    C : 𝕜
    hf : Filter.Eventually (fun x => LE.le (abs (f x)) C) l
    hg : Filter.Tendsto g l (nhds 0)
    hC : Filter.Tendsto (fun x => abs (HMul.hMul C (g x))) l (nhds 0)
    ⊢ Filter.Tendsto (Function.comp abs fun x => HMul.hMul (f x) (g x)) l (nhds 0)
  -/
  apply tendsto_of_tendsto_of_tendsto_of_le_of_le' tendsto_const_nhds hC
    /-
      case hgf
      𝕜 : Type u_1
      α : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : TopologicalSpace 𝕜
      inst✝ : OrderTopology 𝕜
      l : Filter α
      f g : α → 𝕜
      C : 𝕜
      hf : Filter.Eventually (fun x => LE.le (abs (f x)) C) l
      hg : Filter.Tendsto g l (nhds 0)
      hC : Filter.Tendsto (fun x => abs (HMul.hMul C (g x))) l (nhds 0)
      ⊢ Filter.Eventually (fun b => LE.le 0 (Function.comp abs (fun x => HMul.hMul ( …
    -/
  · filter_upwards [hf] with x _ using abs_nonneg _
    /-
      🎉 no goals
    -/
    /-
      case hfh
      𝕜 : Type u_1
      α : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : TopologicalSpace 𝕜
      inst✝ : OrderTopology 𝕜
      l : Filter α
      f g : α → 𝕜
      C : 𝕜
      hf : Filter.Eventually (fun x => LE.le (abs (f x)) C) l
      hg : Filter.Tendsto g l (nhds 0)
      hC : Filter.Tendsto (fun x => abs (HMul.hMul C (g x))) l (nhds 0)
      ⊢ Filter.Eventually (fun b => LE.le (Function.comp abs (fun x => HMul.hMul (f  …
    -/
  · filter_upwards [hf] with x hx
    /-
      case h
      𝕜 : Type u_1
      α : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : TopologicalSpace 𝕜
      inst✝ : OrderTopology 𝕜
      l : Filter α
      f g : α → 𝕜
      C : 𝕜
      hf : Filter.Eventually (fun x => LE.le (abs (f x)) C) l
      hg : Filter.Tendsto g l (nhds 0)
      hC : Filter.Tendsto (fun x => abs (HMul.hMul C (g x))) l (nhds 0)
      x : α
      hx : LE.le (abs (f x)) C
      ⊢ LE.le (Function.comp abs (fun x => HMul.hMul (f x) (g x)) x) (abs (HMul.hMul …
    -/
    simp only [comp_apply, abs_mul]
    /-
      case h
      𝕜 : Type u_1
      α : Type u_2
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : TopologicalSpace 𝕜
      inst✝ : OrderTopology 𝕜
      l : Filter α
      f g : α → 𝕜
      C : 𝕜
      hf : Filter.Eventually (fun x => LE.le (abs (f x)) C) l
      hg : Filter.Tendsto g l (nhds 0)
      hC : Filter.Tendsto (fun x => abs (HMul.hMul C (g x))) l (nhds 0)
      x : α
      hx : LE.le (abs (f x)) C
      ⊢ LE.le (HMul.hMul (abs (f x)) (abs (g x))) (HMul.hMul (abs C) (abs (g x)))
    -/
    exact mul_le_mul_of_nonneg_right (hx.trans (le_abs_self C)) (abs_nonneg _)
    /-
      🎉 no goals
    -/


/-- If `g` tends to zero and there exist constants `b B : 𝕜` such that eventually `b ≤ f x| ≤ B`,
  then the product `f * g` tends to zero. -/
theorem bdd_le_mul_tendsto_zero {f g : α → 𝕜} {b B : 𝕜} (hb : ∀ᶠ x in l, b ≤ f x)
    (hB : ∀ᶠ x in l, f x ≤ B) (hg : Tendsto g l (𝓝 0)) :
    Tendsto (fun x ↦ f x * g x) l (𝓝 0) := by
  /-
    𝕜 : Type u_1
    α : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    l : Filter α
    f g : α → 𝕜
    b B : 𝕜
    hb : Filter.Eventually (fun x => LE.le b (f x)) l
    hB : Filter.Eventually (fun x => LE.le (f x) B) l
    hg : Filter.Tendsto g l (nhds 0)
    ⊢ Filter.Tendsto (fun x => HMul.hMul (f x) (g x)) l (nhds 0)
  -/
  set C := max |b| |B|
  /-
    𝕜 : Type u_1
    α : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    l : Filter α
    f g : α → 𝕜
    b B : 𝕜
    hb : Filter.Eventually (fun x => LE.le b (f x)) l
    hB : Filter.Eventually (fun x => LE.le (f x) B) l
    hg : Filter.Tendsto g l (nhds 0)
    C : 𝕜 := Max.max (abs b) (abs B)
    ⊢ Filter.Tendsto (fun x => HMul.hMul (f x) (g x)) l (nhds 0)
  -/
  have hbC : -C ≤ b := neg_le.mpr (le_max_of_le_left (neg_le_abs b))
  /-
    𝕜 : Type u_1
    α : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    l : Filter α
    f g : α → 𝕜
    b B : 𝕜
    hb : Filter.Eventually (fun x => LE.le b (f x)) l
    hB : Filter.Eventually (fun x => LE.le (f x) B) l
    hg : Filter.Tendsto g l (nhds 0)
    C : 𝕜 := Max.max (abs b) (abs B)
    hbC : LE.le (Neg.neg C) b
    ⊢ Filter.Tendsto (fun x => HMul.hMul (f x) (g x)) l (nhds 0)
  -/
  have hBC : B ≤ C := le_max_of_le_right (le_abs_self B)
  /-
    𝕜 : Type u_1
    α : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    l : Filter α
    f g : α → 𝕜
    b B : 𝕜
    hb : Filter.Eventually (fun x => LE.le b (f x)) l
    hB : Filter.Eventually (fun x => LE.le (f x) B) l
    hg : Filter.Tendsto g l (nhds 0)
    C : 𝕜 := Max.max (abs b) (abs B)
    hbC : LE.le (Neg.neg C) b
    hBC : LE.le B C
    ⊢ Filter.Tendsto (fun x => HMul.hMul (f x) (g x)) l (nhds 0)
  -/
  apply bdd_le_mul_tendsto_zero' C _ hg
  /-
    𝕜 : Type u_1
    α : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    l : Filter α
    f g : α → 𝕜
    b B : 𝕜
    hb : Filter.Eventually (fun x => LE.le b (f x)) l
    hB : Filter.Eventually (fun x => LE.le (f x) B) l
    hg : Filter.Tendsto g l (nhds 0)
    C : 𝕜 := Max.max (abs b) (abs B)
    hbC : LE.le (Neg.neg C) b
    hBC : LE.le B C
    ⊢ Filter.Eventually (fun x => LE.le (abs (f x)) C) l
  -/
  filter_upwards [hb, hB]
  /-
    case h
    𝕜 : Type u_1
    α : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    l : Filter α
    f g : α → 𝕜
    b B : 𝕜
    hb : Filter.Eventually (fun x => LE.le b (f x)) l
    hB : Filter.Eventually (fun x => LE.le (f x) B) l
    hg : Filter.Tendsto g l (nhds 0)
    C : 𝕜 := Max.max (abs b) (abs B)
    hbC : LE.le (Neg.neg C) b
    hBC : LE.le B C
    ⊢ ∀ (a : α), LE.le b (f a) → LE.le (f a) B → LE.le (abs (f a)) C
  -/
  exact fun x hbx hBx ↦ abs_le.mpr ⟨hbC.trans hbx, hBx.trans hBC⟩
  /-
    🎉 no goals
  -/


/-- If `g` tends to `atTop` and there exist constants `b B : 𝕜` such that eventually
  `b ≤ f x| ≤ B`, then the quotient `f / g` tends to zero. -/
theorem tendsto_bdd_div_atTop_nhds_zero {f g : α → 𝕜} {b B : 𝕜}
    (hb : ∀ᶠ x in l, b ≤ f x) (hB : ∀ᶠ x in l, f x ≤ B) (hg : Tendsto g l atTop) :
    Tendsto (fun x => f x / g x) l (𝓝 0) := by
  /-
    𝕜 : Type u_1
    α : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    l : Filter α
    f g : α → 𝕜
    b B : 𝕜
    hb : Filter.Eventually (fun x => LE.le b (f x)) l
    hB : Filter.Eventually (fun x => LE.le (f x) B) l
    hg : Filter.Tendsto g l Filter.atTop
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) l (nhds 0)
  -/
  simp only [div_eq_mul_inv]
  /-
    𝕜 : Type u_1
    α : Type u_2
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    l : Filter α
    f g : α → 𝕜
    b B : 𝕜
    hb : Filter.Eventually (fun x => LE.le b (f x)) l
    hB : Filter.Eventually (fun x => LE.le (f x) B) l
    hg : Filter.Tendsto g l Filter.atTop
    ⊢ Filter.Tendsto (fun x => HMul.hMul (f x) (Inv.inv (g x))) l (nhds 0)
  -/
  exact bdd_le_mul_tendsto_zero hb hB hg.inv_tendsto_atTop
  /-
    🎉 no goals
  -/


/-- The function `x^(-n)` tends to `0` at `+∞` for any positive natural `n`.
A version for positive real powers exists as `tendsto_rpow_neg_atTop`. -/
theorem tendsto_pow_neg_atTop {n : ℕ} (hn : n ≠ 0) :
    Tendsto (fun x : 𝕜 => x ^ (-(n : ℤ))) atTop (𝓝 0) := by
  /-
    𝕜 : Type u_1
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    n : Nat
    hn : Ne n 0
    ⊢ Filter.Tendsto (fun x => HPow.hPow x (Neg.neg ↑n)) Filter.atTop (nhds 0)
  -/
  simpa only [zpow_neg, zpow_natCast] using (@tendsto_pow_atTop 𝕜 _ _ hn).inv_tendsto_atTop
  /-
    🎉 no goals
  -/


theorem tendsto_zpow_atTop_zero {n : ℤ} (hn : n < 0) :
    Tendsto (fun x : 𝕜 => x ^ n) atTop (𝓝 0) := by
  /-
    𝕜 : Type u_1
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    n : Int
    hn : LT.lt n 0
    ⊢ Filter.Tendsto (fun x => HPow.hPow x n) Filter.atTop (nhds 0)
  -/
  lift -n to ℕ using le_of_lt (neg_pos.mpr hn) with N h
  /-
    case intro
    𝕜 : Type u_1
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    n : Int
    hn : LT.lt n 0
    N : Nat
    h : Eq (↑N) (Neg.neg n)
    ⊢ Filter.Tendsto (fun x => HPow.hPow x n) Filter.atTop (nhds 0)
  -/
  rw [← neg_pos, ← h, Nat.cast_pos] at hn
  /-
    case intro
    𝕜 : Type u_1
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    n : Int
    N : Nat
    hn : LT.lt 0 N
    h : Eq (↑N) (Neg.neg n)
    ⊢ Filter.Tendsto (fun x => HPow.hPow x n) Filter.atTop (nhds 0)
  -/
  simpa only [h, neg_neg] using tendsto_pow_neg_atTop hn.ne'
  /-
    🎉 no goals
  -/


theorem tendsto_const_mul_zpow_atTop_zero {n : ℤ} {c : 𝕜} (hn : n < 0) :
    Tendsto (fun x => c * x ^ n) atTop (𝓝 0) :=
  mul_zero c ▸ Filter.Tendsto.const_mul c (tendsto_zpow_atTop_zero hn)


theorem tendsto_const_mul_pow_nhds_iff' {n : ℕ} {c d : 𝕜} :
    Tendsto (fun x : 𝕜 => c * x ^ n) atTop (𝓝 d) ↔ (c = 0 ∨ n = 0) ∧ c = d := by
  /-
    𝕜 : Type u_1
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    n : Nat
    c d : 𝕜
    ⊢ Iff (Filter.Tendsto (fun x => HMul.hMul c (HPow.hPow x n)) Filter.atTop (nhd …
  -/
  rcases eq_or_ne n 0 with (rfl | hn)
    /-
      case inl
      𝕜 : Type u_1
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : TopologicalSpace 𝕜
      inst✝ : OrderTopology 𝕜
      c d : 𝕜
      ⊢ Iff (Filter.Tendsto (fun x => HMul.hMul c (HPow.hPow x 0)) Filter.atTop (nhd …
    -/
  · simp [tendsto_const_nhds_iff]
    /-
      🎉 no goals
    -/
  /-
    case inr
    𝕜 : Type u_1
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    n : Nat
    c d : 𝕜
    hn : Ne n 0
    ⊢ Iff (Filter.Tendsto (fun x => HMul.hMul c (HPow.hPow x n)) Filter.atTop (nhd …
  -/
  rcases lt_trichotomy c 0 with (hc | rfl | hc)
    /-
      case inr.inl
      𝕜 : Type u_1
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : TopologicalSpace 𝕜
      inst✝ : OrderTopology 𝕜
      n : Nat
      c d : 𝕜
      hn : Ne n 0
      hc : LT.lt c 0
      ⊢ Iff (Filter.Tendsto (fun x => HMul.hMul c (HPow.hPow x n)) Filter.atTop (nhd …
    -/
  · have := tendsto_const_mul_pow_atBot_iff.2 ⟨hn, hc⟩
    /-
      case inr.inl
      𝕜 : Type u_1
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : TopologicalSpace 𝕜
      inst✝ : OrderTopology 𝕜
      n : Nat
      c d : 𝕜
      hn : Ne n 0
      hc : LT.lt c 0
      this : Filter.Tendsto (fun x => HMul.hMul c (HPow.hPow x n)) Filter.atTop Filt …
      ⊢ Iff (Filter.Tendsto (fun x => HMul.hMul c (HPow.hPow x n)) Filter.atTop (nhd …
    -/
    simp [not_tendsto_nhds_of_tendsto_atBot this, hc.ne, hn]
    /-
      🎉 no goals
    -/
    /-
      case inr.inr.inl
      𝕜 : Type u_1
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : TopologicalSpace 𝕜
      inst✝ : OrderTopology 𝕜
      n : Nat
      d : 𝕜
      hn : Ne n 0
      ⊢ Iff (Filter.Tendsto (fun x => HMul.hMul 0 (HPow.hPow x n)) Filter.atTop (nhd …
    -/
  · simp [tendsto_const_nhds_iff]
    /-
      🎉 no goals
    -/
    /-
      case inr.inr.inr
      𝕜 : Type u_1
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : TopologicalSpace 𝕜
      inst✝ : OrderTopology 𝕜
      n : Nat
      c d : 𝕜
      hn : Ne n 0
      hc : LT.lt 0 c
      ⊢ Iff (Filter.Tendsto (fun x => HMul.hMul c (HPow.hPow x n)) Filter.atTop (nhd …
    -/
  · have := tendsto_const_mul_pow_atTop_iff.2 ⟨hn, hc⟩
    /-
      case inr.inr.inr
      𝕜 : Type u_1
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : TopologicalSpace 𝕜
      inst✝ : OrderTopology 𝕜
      n : Nat
      c d : 𝕜
      hn : Ne n 0
      hc : LT.lt 0 c
      this : Filter.Tendsto (fun x => HMul.hMul c (HPow.hPow x n)) Filter.atTop Filt …
      ⊢ Iff (Filter.Tendsto (fun x => HMul.hMul c (HPow.hPow x n)) Filter.atTop (nhd …
    -/
    simp [not_tendsto_nhds_of_tendsto_atTop this, hc.ne', hn]
    /-
      🎉 no goals
    -/


theorem tendsto_const_mul_pow_nhds_iff {n : ℕ} {c d : 𝕜} (hc : c ≠ 0) :
    Tendsto (fun x : 𝕜 => c * x ^ n) atTop (𝓝 d) ↔ n = 0 ∧ c = d := by
  /-
    𝕜 : Type u_1
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    n : Nat
    c d : 𝕜
    hc : Ne c 0
    ⊢ Iff (Filter.Tendsto (fun x => HMul.hMul c (HPow.hPow x n)) Filter.atTop (nhd …
  -/
  simp [tendsto_const_mul_pow_nhds_iff', hc]
  /-
    🎉 no goals
  -/


theorem tendsto_const_mul_zpow_atTop_nhds_iff {n : ℤ} {c d : 𝕜} (hc : c ≠ 0) :
    Tendsto (fun x : 𝕜 => c * x ^ n) atTop (𝓝 d) ↔ n = 0 ∧ c = d ∨ n < 0 ∧ d = 0 := by
  /-
    𝕜 : Type u_1
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    n : Int
    c d : 𝕜
    hc : Ne c 0
    ⊢ Iff (Filter.Tendsto (fun x => HMul.hMul c (HPow.hPow x n)) Filter.atTop (nhd …
  -/
  refine ⟨fun h => ?_, fun h => ?_⟩
  · cases n with -- Porting note: Lean 3 proof used `by_cases`, then `lift` but `lift` failed
    | ofNat n =>
      left
      simpa [tendsto_const_mul_pow_nhds_iff hc] using h
    | negSucc n =>
      have hn := Int.negSucc_lt_zero n
      exact Or.inr ⟨hn, tendsto_nhds_unique h (tendsto_const_mul_zpow_atTop_zero hn)⟩
    /-
      case refine_2
      𝕜 : Type u_1
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : TopologicalSpace 𝕜
      inst✝ : OrderTopology 𝕜
      n : Int
      c d : 𝕜
      hc : Ne c 0
      h : Or (And (Eq n 0) (Eq c d)) (And (LT.lt n 0) (Eq d 0))
      ⊢ Filter.Tendsto (fun x => HMul.hMul c (HPow.hPow x n)) Filter.atTop (nhds d)
    -/
  · cases' h with h h
      /-
        case refine_2.inl
        𝕜 : Type u_1
        inst✝² : LinearOrderedField 𝕜
        inst✝¹ : TopologicalSpace 𝕜
        inst✝ : OrderTopology 𝕜
        n : Int
        c d : 𝕜
        hc : Ne c 0
        h : And (Eq n 0) (Eq c d)
        ⊢ Filter.Tendsto (fun x => HMul.hMul c (HPow.hPow x n)) Filter.atTop (nhds d)
      -/
    · simp only [h.left, h.right, zpow_zero, mul_one]
      /-
        case refine_2.inl
        𝕜 : Type u_1
        inst✝² : LinearOrderedField 𝕜
        inst✝¹ : TopologicalSpace 𝕜
        inst✝ : OrderTopology 𝕜
        n : Int
        c d : 𝕜
        hc : Ne c 0
        h : And (Eq n 0) (Eq c d)
        ⊢ Filter.Tendsto (fun x => d) Filter.atTop (nhds d)
      -/
      exact tendsto_const_nhds
      /-
        🎉 no goals
      -/
      /-
        case refine_2.inr
        𝕜 : Type u_1
        inst✝² : LinearOrderedField 𝕜
        inst✝¹ : TopologicalSpace 𝕜
        inst✝ : OrderTopology 𝕜
        n : Int
        c d : 𝕜
        hc : Ne c 0
        h : And (LT.lt n 0) (Eq d 0)
        ⊢ Filter.Tendsto (fun x => HMul.hMul c (HPow.hPow x n)) Filter.atTop (nhds d)
      -/
    · exact h.2.symm ▸ tendsto_const_mul_zpow_atTop_zero h.1
      /-
        🎉 no goals
      -/

-- see Note [lower instance priority]

instance (priority := 100) LinearOrderedSemifield.toHasContinuousInv₀ {𝕜}
    [LinearOrderedSemifield 𝕜] [TopologicalSpace 𝕜] [OrderTopology 𝕜] [ContinuousMul 𝕜] :
    HasContinuousInv₀ 𝕜 := .of_nhds_one <| tendsto_order.2 <| by
  /-
    𝕜✝ : Type u_1
    α : Type u_2
    inst✝⁶ : LinearOrderedField 𝕜✝
    inst✝⁵ : TopologicalSpace 𝕜✝
    inst✝⁴ : OrderTopology 𝕜✝
    l : Filter α
    f g : α → 𝕜✝
    𝕜 : Type u_3
    inst✝³ : LinearOrderedSemifield 𝕜
    inst✝² : TopologicalSpace 𝕜
    inst✝¹ : OrderTopology 𝕜
    inst✝ : ContinuousMul 𝕜
    ⊢ And (∀ (a' : 𝕜), LT.lt a' 1 → Filter.Eventually (fun b => LT.lt a' (Inv.inv  …
  -/
  refine ⟨fun x hx => ?_, fun x hx => ?_⟩
  · obtain ⟨x', h₀, hxx', h₁⟩ : ∃ x', 0 < x' ∧ x ≤ x' ∧ x' < 1 :=
      ⟨max x (1 / 2), one_half_pos.trans_le (le_max_right _ _), le_max_left _ _,
        max_lt hx one_half_lt_one⟩
    /-
      case refine_1.intro.intro.intro
      𝕜✝ : Type u_1
      α : Type u_2
      inst✝⁶ : LinearOrderedField 𝕜✝
      inst✝⁵ : TopologicalSpace 𝕜✝
      inst✝⁴ : OrderTopology 𝕜✝
      l : Filter α
      f g : α → 𝕜✝
      𝕜 : Type u_3
      inst✝³ : LinearOrderedSemifield 𝕜
      inst✝² : TopologicalSpace 𝕜
      inst✝¹ : OrderTopology 𝕜
      inst✝ : ContinuousMul 𝕜
      x : 𝕜
      hx : LT.lt x 1
      x' : 𝕜
      h₀ : LT.lt 0 x'
      hxx' : LE.le x x'
      h₁ : LT.lt x' 1
      ⊢ Filter.Eventually (fun b => LT.lt x (Inv.inv b)) (nhds 1)
    -/
    filter_upwards [Ioo_mem_nhds one_pos ((one_lt_inv₀ h₀).2 h₁)] with y hy
    /-
      case h
      𝕜✝ : Type u_1
      α : Type u_2
      inst✝⁶ : LinearOrderedField 𝕜✝
      inst✝⁵ : TopologicalSpace 𝕜✝
      inst✝⁴ : OrderTopology 𝕜✝
      l : Filter α
      f g : α → 𝕜✝
      𝕜 : Type u_3
      inst✝³ : LinearOrderedSemifield 𝕜
      inst✝² : TopologicalSpace 𝕜
      inst✝¹ : OrderTopology 𝕜
      inst✝ : ContinuousMul 𝕜
      x : 𝕜
      hx : LT.lt x 1
      x' : 𝕜
      h₀ : LT.lt 0 x'
      hxx' : LE.le x x'
      h₁ : LT.lt x' 1
      y : 𝕜
      hy : Membership.mem (Set.Ioo 0 (Inv.inv x')) y
      ⊢ LT.lt x (Inv.inv y)
    -/
    exact hxx'.trans_lt <| lt_inv_of_lt_inv₀ hy.1 hy.2
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      𝕜✝ : Type u_1
      α : Type u_2
      inst✝⁶ : LinearOrderedField 𝕜✝
      inst✝⁵ : TopologicalSpace 𝕜✝
      inst✝⁴ : OrderTopology 𝕜✝
      l : Filter α
      f g : α → 𝕜✝
      𝕜 : Type u_3
      inst✝³ : LinearOrderedSemifield 𝕜
      inst✝² : TopologicalSpace 𝕜
      inst✝¹ : OrderTopology 𝕜
      inst✝ : ContinuousMul 𝕜
      x : 𝕜
      hx : GT.gt x 1
      ⊢ Filter.Eventually (fun b => LT.lt (Inv.inv b) x) (nhds 1)
    -/
  · filter_upwards [Ioi_mem_nhds (inv_lt_one_of_one_lt₀ hx)] with y hy
    /-
      case h
      𝕜✝ : Type u_1
      α : Type u_2
      inst✝⁶ : LinearOrderedField 𝕜✝
      inst✝⁵ : TopologicalSpace 𝕜✝
      inst✝⁴ : OrderTopology 𝕜✝
      l : Filter α
      f g : α → 𝕜✝
      𝕜 : Type u_3
      inst✝³ : LinearOrderedSemifield 𝕜
      inst✝² : TopologicalSpace 𝕜
      inst✝¹ : OrderTopology 𝕜
      inst✝ : ContinuousMul 𝕜
      x : 𝕜
      hx : GT.gt x 1
      y : 𝕜
      hy : Membership.mem (Set.Ioi (Inv.inv x)) y
      ⊢ LT.lt (Inv.inv y) x
    -/
    exact inv_lt_of_inv_lt₀ (by positivity) hy
    /-
      🎉 no goals
    -/


instance (priority := 100) LinearOrderedField.toTopologicalDivisionRing :
    TopologicalDivisionRing 𝕜 := ⟨⟩

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: generalize to a `GroupWithZero`

theorem comap_mulLeft_nhdsGT_zero {x : 𝕜} (hx : 0 < x) : comap (x * ·) (𝓝[>] 0) = 𝓝[>] 0 := by
  /-
    𝕜 : Type u_1
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    x : 𝕜
    hx : LT.lt 0 x
    ⊢ Eq (Filter.comap (fun x_1 => HMul.hMul x x_1) (nhdsWithin 0 (Set.Ioi 0))) (n …
  -/
  rw [nhdsWithin, comap_inf, comap_principal, preimage_const_mul_Ioi _ hx, zero_div]
  /-
    𝕜 : Type u_1
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    x : 𝕜
    hx : LT.lt 0 x
    ⊢ Eq (Min.min (Filter.comap (fun x_1 => HMul.hMul x x_1) (nhds 0)) (Filter.pri …
  -/
  congr 1
  /-
    case e_a
    𝕜 : Type u_1
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    x : 𝕜
    hx : LT.lt 0 x
    ⊢ Eq (Filter.comap (fun x_1 => HMul.hMul x x_1) (nhds 0)) (nhds 0)
  -/
  refine ((Homeomorph.mulLeft₀ x hx.ne').comap_nhds_eq _).trans ?_
  /-
    case e_a
    𝕜 : Type u_1
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    x : 𝕜
    hx : LT.lt 0 x
    ⊢ Eq (nhds ((Homeomorph.mulLeft₀ x ⋯).symm 0)) (nhds 0)
  -/
  simp
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-22")]
alias nhdsWithin_pos_comap_mul_left := comap_mulLeft_nhdsGT_zero


theorem eventually_nhdsGT_zero_mul_left {x : 𝕜} (hx : 0 < x) {p : 𝕜 → Prop}
    (h : ∀ᶠ ε in 𝓝[>] 0, p ε) : ∀ᶠ ε in 𝓝[>] 0, p (x * ε) := by
  /-
    𝕜 : Type u_1
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    x : 𝕜
    hx : LT.lt 0 x
    p : 𝕜 → Prop
    h : Filter.Eventually (fun ε => p ε) (nhdsWithin 0 (Set.Ioi 0))
    ⊢ Filter.Eventually (fun ε => p (HMul.hMul x ε)) (nhdsWithin 0 (Set.Ioi 0))
  -/
  rw [← comap_mulLeft_nhdsGT_zero hx]
  /-
    𝕜 : Type u_1
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : TopologicalSpace 𝕜
    inst✝ : OrderTopology 𝕜
    x : 𝕜
    hx : LT.lt 0 x
    p : 𝕜 → Prop
    h : Filter.Eventually (fun ε => p ε) (nhdsWithin 0 (Set.Ioi 0))
    ⊢ Filter.Eventually (fun ε => p (HMul.hMul x ε)) (Filter.comap (fun x_1 => HMu …
  -/
  exact h.comap fun ε => x * ε
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-22")]
alias eventually_nhdsWithin_pos_mul_left := eventually_nhdsGT_zero_mul_left

