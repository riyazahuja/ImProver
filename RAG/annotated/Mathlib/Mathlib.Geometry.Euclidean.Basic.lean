/-- The midpoint of the segment AB is the same distance from A as it is from B. -/
theorem dist_left_midpoint_eq_dist_right_midpoint (p1 p2 : P) :
    dist p1 (midpoint ℝ p1 p2) = dist p2 (midpoint ℝ p1 p2) := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    p1 p2 : P
    ⊢ Eq (Dist.dist p1 (midpoint Real p1 p2)) (Dist.dist p2 (midpoint Real p1 p2))
  -/
  rw [dist_left_midpoint (𝕜 := ℝ) p1 p2, dist_right_midpoint (𝕜 := ℝ) p1 p2]
  /-
    🎉 no goals
  -/


/-- The inner product of two vectors given with `weightedVSub`, in
terms of the pairwise distances. -/
theorem inner_weightedVSub {ι₁ : Type*} {s₁ : Finset ι₁} {w₁ : ι₁ → ℝ} (p₁ : ι₁ → P)
    (h₁ : ∑ i ∈ s₁, w₁ i = 0) {ι₂ : Type*} {s₂ : Finset ι₂} {w₂ : ι₂ → ℝ} (p₂ : ι₂ → P)
    (h₂ : ∑ i ∈ s₂, w₂ i = 0) :
    ⟪s₁.weightedVSub p₁ w₁, s₂.weightedVSub p₂ w₂⟫ =
      (-∑ i₁ ∈ s₁, ∑ i₂ ∈ s₂, w₁ i₁ * w₂ i₂ * (dist (p₁ i₁) (p₂ i₂) * dist (p₁ i₁) (p₂ i₂))) /
        2 := by
  rw [Finset.weightedVSub_apply, Finset.weightedVSub_apply,
    inner_sum_smul_sum_smul_of_sum_eq_zero _ h₁ _ h₂]
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    ι₁ : Type u_3
    s₁ : Finset ι₁
    w₁ : ι₁ → Real
    p₁ : ι₁ → P
    h₁ : Eq (s₁.sum fun i => w₁ i) 0
    ι₂ : Type u_4
    s₂ : Finset ι₂
    w₂ : ι₂ → Real
    p₂ : ι₂ → P
    h₂ : Eq (s₂.sum fun i => w₂ i) 0
    ⊢ Eq (HDiv.hDiv (Neg.neg (s₁.sum fun i₁ => s₂.sum fun i₂ => HMul.hMul (HMul.hM …
  -/
  simp_rw [vsub_sub_vsub_cancel_right]
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    ι₁ : Type u_3
    s₁ : Finset ι₁
    w₁ : ι₁ → Real
    p₁ : ι₁ → P
    h₁ : Eq (s₁.sum fun i => w₁ i) 0
    ι₂ : Type u_4
    s₂ : Finset ι₂
    w₂ : ι₂ → Real
    p₂ : ι₂ → P
    h₂ : Eq (s₂.sum fun i => w₂ i) 0
    ⊢ Eq (HDiv.hDiv (Neg.neg (s₁.sum fun x => s₂.sum fun x_1 => HMul.hMul (HMul.hM …
  -/
                     /-
                       🎉 no goals
                     -/
  rcongr (i₁ i₂) <;> rw [dist_eq_norm_vsub V (p₁ i₁) (p₂ i₂)]
                     /-
                       🎉 no goals
                     -/


/-- The distance between two points given with `affineCombination`,
in terms of the pairwise distances between the points in that
combination. -/
theorem dist_affineCombination {ι : Type*} {s : Finset ι} {w₁ w₂ : ι → ℝ} (p : ι → P)
    (h₁ : ∑ i ∈ s, w₁ i = 1) (h₂ : ∑ i ∈ s, w₂ i = 1) : by
      /-
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : InnerProductSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        ι : Type u_3
        s : Finset ι
        w₁ w₂ : ι → Real
        p : ι → P
        h₁ : Eq (s.sum fun i => w₁ i) 1
        h₂ : Eq (s.sum fun i => w₂ i) 1
        ⊢ Sort ?u.15967
      -/
      have a₁ := s.affineCombination ℝ p w₁
      /-
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : InnerProductSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        ι : Type u_3
        s : Finset ι
        w₁ w₂ : ι → Real
        p : ι → P
        h₁ : Eq (s.sum fun i => w₁ i) 1
        h₂ : Eq (s.sum fun i => w₂ i) 1
        a₁ : P
        ⊢ Sort ?u.15967
      -/
      have a₂ := s.affineCombination ℝ p w₂
      exact dist a₁ a₂ * dist a₁ a₂ = (-∑ i₁ ∈ s, ∑ i₂ ∈ s,
        (w₁ - w₂) i₁ * (w₁ - w₂) i₂ * (dist (p i₁) (p i₂) * dist (p i₁) (p i₂))) / 2 := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    ι : Type u_3
    s : Finset ι
    w₁ w₂ : ι → Real
    p : ι → P
    h₁ : Eq (s.sum fun i => w₁ i) 1
    h₂ : Eq (s.sum fun i => w₂ i) 1
    ⊢ letFun ((Finset.affineCombination Real s p) w₁) fun a₁ => letFun ((Finset.af …
  -/
  dsimp only
  rw [dist_eq_norm_vsub V (s.affineCombination ℝ p w₁) (s.affineCombination ℝ p w₂), ←
    @inner_self_eq_norm_mul_norm ℝ, Finset.affineCombination_vsub]
  have h : (∑ i ∈ s, (w₁ - w₂) i) = 0 := by
    simp_rw [Pi.sub_apply, Finset.sum_sub_distrib, h₁, h₂, sub_self]
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    ι : Type u_3
    s : Finset ι
    w₁ w₂ : ι → Real
    p : ι → P
    h₁ : Eq (s.sum fun i => w₁ i) 1
    h₂ : Eq (s.sum fun i => w₂ i) 1
    h : Eq (s.sum fun i => HSub.hSub w₁ w₂ i) 0
    ⊢ Eq (RCLike.re (Inner.inner ((s.weightedVSub p) (HSub.hSub w₁ w₂)) ((s.weight …
  -/
  exact inner_weightedVSub p h p h
  /-
    🎉 no goals
  -/

-- Porting note: `inner_vsub_vsub_of_dist_eq_of_dist_eq` moved to `PerpendicularBisector`


/-- The squared distance between points on a line (expressed as a
multiple of a fixed vector added to a point) and another point,
expressed as a quadratic. -/
theorem dist_smul_vadd_sq (r : ℝ) (v : V) (p₁ p₂ : P) :
    dist (r • v +ᵥ p₁) p₂ * dist (r • v +ᵥ p₁) p₂ =
      ⟪v, v⟫ * r * r + 2 * ⟪v, p₁ -ᵥ p₂⟫ * r + ⟪p₁ -ᵥ p₂, p₁ -ᵥ p₂⟫ := by
  rw [dist_eq_norm_vsub V _ p₂, ← real_inner_self_eq_norm_mul_norm, vadd_vsub_assoc,
    real_inner_add_add_self, real_inner_smul_left, real_inner_smul_left, real_inner_smul_right]
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    r : Real
    v : V
    p₁ p₂ : P
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul r (HMul.hMul r (Inner.inner v v))) (HMul …
  -/
  ring
  /-
    🎉 no goals
  -/


/-- The condition for two points on a line to be equidistant from
another point. -/
theorem dist_smul_vadd_eq_dist {v : V} (p₁ p₂ : P) (hv : v ≠ 0) (r : ℝ) :
    dist (r • v +ᵥ p₁) p₂ = dist p₁ p₂ ↔ r = 0 ∨ r = -2 * ⟪v, p₁ -ᵥ p₂⟫ / ⟪v, v⟫ := by
  conv_lhs =>
    rw [← mul_self_inj_of_nonneg dist_nonneg dist_nonneg, dist_smul_vadd_sq, mul_assoc,
      ← sub_eq_zero, add_sub_assoc, dist_eq_norm_vsub V p₁ p₂, ← real_inner_self_eq_norm_mul_norm,
      sub_self]
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    v : V
    p₁ p₂ : P
    hv : Ne v 0
    r : Real
    ⊢ Iff (Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul (Inner.inner v v) (HMul.hMul r r))  …
  -/
  have hvi : ⟪v, v⟫ ≠ 0 := by simpa using hv
  have hd : discrim ⟪v, v⟫ (2 * ⟪v, p₁ -ᵥ p₂⟫) 0 = 2 * ⟪v, p₁ -ᵥ p₂⟫ * (2 * ⟪v, p₁ -ᵥ p₂⟫) := by
    rw [discrim]
    ring
  rw [quadratic_eq_zero_iff hvi hd, neg_add_cancel, zero_div, neg_mul_eq_neg_mul, ←
    mul_sub_right_distrib, sub_eq_add_neg, ← mul_two, mul_assoc, mul_div_assoc, mul_div_mul_left,
    mul_div_assoc]
  /-
    case hc
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : InnerProductSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    v : V
    p₁ p₂ : P
    hv : Ne v 0
    r : Real
    hvi : Ne (Inner.inner v v) 0
    hd : Eq (discrim (Inner.inner v v) (HMul.hMul 2 (Inner.inner v (VSub.vsub p₁ p …
    ⊢ Ne 2 0
  -/
  norm_num
  /-
    🎉 no goals
  -/


/-- Distances `r₁` `r₂` of `p` from two different points `c₁` `c₂` determine at
most two points `p₁` `p₂` in a two-dimensional subspace containing those points
(two circles intersect in at most two points). -/
theorem eq_of_dist_eq_of_dist_eq_of_mem_of_finrank_eq_two {s : AffineSubspace ℝ P}
    [FiniteDimensional ℝ s.direction] (hd : finrank ℝ s.direction = 2) {c₁ c₂ p₁ p₂ p : P}
    (hc₁s : c₁ ∈ s) (hc₂s : c₂ ∈ s) (hp₁s : p₁ ∈ s) (hp₂s : p₂ ∈ s) (hps : p ∈ s) {r₁ r₂ : ℝ}
    (hc : c₁ ≠ c₂) (hp : p₁ ≠ p₂) (hp₁c₁ : dist p₁ c₁ = r₁) (hp₂c₁ : dist p₂ c₁ = r₁)
    (hpc₁ : dist p c₁ = r₁) (hp₁c₂ : dist p₁ c₂ = r₂) (hp₂c₂ : dist p₂ c₂ = r₂)
    (hpc₂ : dist p c₂ = r₂) : p = p₁ ∨ p = p₂ := by
  have ho : ⟪c₂ -ᵥ c₁, p₂ -ᵥ p₁⟫ = 0 :=
    inner_vsub_vsub_of_dist_eq_of_dist_eq (hp₁c₁.trans hp₂c₁.symm) (hp₁c₂.trans hp₂c₂.symm)
  have hop : ⟪c₂ -ᵥ c₁, p -ᵥ p₁⟫ = 0 :=
    inner_vsub_vsub_of_dist_eq_of_dist_eq (hp₁c₁.trans hpc₁.symm) (hp₁c₂.trans hpc₂.symm)
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : MetricSpace P
    inst✝¹ : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝ : FiniteDimensional Real (Subtype fun x => Membership.mem s.direction x)
    hd : Eq (Module.finrank Real (Subtype fun x => Membership.mem s.direction x)) 2
    c₁ c₂ p₁ p₂ p : P
    hc₁s : Membership.mem s c₁
    hc₂s : Membership.mem s c₂
    hp₁s : Membership.mem s p₁
    hp₂s : Membership.mem s p₂
    hps : Membership.mem s p
    r₁ r₂ : Real
    hc : Ne c₁ c₂
    hp : Ne p₁ p₂
    hp₁c₁ : Eq (Dist.dist p₁ c₁) r₁
    hp₂c₁ : Eq (Dist.dist p₂ c₁) r₁
    hpc₁ : Eq (Dist.dist p c₁) r₁
    hp₁c₂ : Eq (Dist.dist p₁ c₂) r₂
    hp₂c₂ : Eq (Dist.dist p₂ c₂) r₂
    hpc₂ : Eq (Dist.dist p c₂) r₂
    ho : Eq (Inner.inner (VSub.vsub c₂ c₁) (VSub.vsub p₂ p₁)) 0
    hop : Eq (Inner.inner (VSub.vsub c₂ c₁) (VSub.vsub p p₁)) 0
    ⊢ Or (Eq p p₁) (Eq p p₂)
  -/
  let b : Fin 2 → V := ![c₂ -ᵥ c₁, p₂ -ᵥ p₁]
  have hb : LinearIndependent ℝ b := by
    refine linearIndependent_of_ne_zero_of_inner_eq_zero ?_ ?_
    · intro i
      fin_cases i <;> simp [b, hc.symm, hp.symm]
    · intro i j hij
      fin_cases i <;> fin_cases j <;> try exact False.elim (hij rfl)
      · exact ho
      · rw [real_inner_comm]
        exact ho
  have hbs : Submodule.span ℝ (Set.range b) = s.direction := by
    refine Submodule.eq_of_le_of_finrank_eq ?_ ?_
    · rw [Submodule.span_le, Set.range_subset_iff]
      intro i
      fin_cases i
      · exact vsub_mem_direction hc₂s hc₁s
      · exact vsub_mem_direction hp₂s hp₁s
    · rw [finrank_span_eq_card hb, Fintype.card_fin, hd]
  have hv : ∀ v ∈ s.direction, ∃ t₁ t₂ : ℝ, v = t₁ • (c₂ -ᵥ c₁) + t₂ • (p₂ -ᵥ p₁) := by
    intro v hv
    have hr : Set.range b = {c₂ -ᵥ c₁, p₂ -ᵥ p₁} := by
      have hu : (Finset.univ : Finset (Fin 2)) = {0, 1} := by decide
      classical
      rw [← Fintype.coe_image_univ, hu]
      simp [b]
    rw [← hbs, hr, Submodule.mem_span_insert] at hv
    rcases hv with ⟨t₁, v', hv', hv⟩
    rw [Submodule.mem_span_singleton] at hv'
    rcases hv' with ⟨t₂, rfl⟩
    exact ⟨t₁, t₂, hv⟩
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : MetricSpace P
    inst✝¹ : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝ : FiniteDimensional Real (Subtype fun x => Membership.mem s.direction x)
    hd : Eq (Module.finrank Real (Subtype fun x => Membership.mem s.direction x)) 2
    c₁ c₂ p₁ p₂ p : P
    hc₁s : Membership.mem s c₁
    hc₂s : Membership.mem s c₂
    hp₁s : Membership.mem s p₁
    hp₂s : Membership.mem s p₂
    hps : Membership.mem s p
    r₁ r₂ : Real
    hc : Ne c₁ c₂
    hp : Ne p₁ p₂
    hp₁c₁ : Eq (Dist.dist p₁ c₁) r₁
    hp₂c₁ : Eq (Dist.dist p₂ c₁) r₁
    hpc₁ : Eq (Dist.dist p c₁) r₁
    hp₁c₂ : Eq (Dist.dist p₁ c₂) r₂
    hp₂c₂ : Eq (Dist.dist p₂ c₂) r₂
    hpc₂ : Eq (Dist.dist p c₂) r₂
    ho : Eq (Inner.inner (VSub.vsub c₂ c₁) (VSub.vsub p₂ p₁)) 0
    hop : Eq (Inner.inner (VSub.vsub c₂ c₁) (VSub.vsub p p₁)) 0
    b : Fin 2 → V := Matrix.vecCons (VSub.vsub c₂ c₁) (Matrix.vecCons (VSub.vsub p …
    hb : LinearIndependent Real b
    hbs : Eq (Submodule.span Real (Set.range b)) s.direction
    hv : ∀ (v : V), Membership.mem s.direction v → Exists fun t₁ => Exists fun t₂  …
    ⊢ Or (Eq p p₁) (Eq p p₂)
  -/
  rcases hv (p -ᵥ p₁) (vsub_mem_direction hps hp₁s) with ⟨t₁, t₂, hpt⟩
  simp only [hpt, inner_add_right, inner_smul_right, ho, mul_zero, add_zero,
    mul_eq_zero, inner_self_eq_zero, vsub_eq_zero_iff_eq, hc.symm, or_false] at hop
  /-
    case intro.intro
    V : Type u_1
    P : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : MetricSpace P
    inst✝¹ : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝ : FiniteDimensional Real (Subtype fun x => Membership.mem s.direction x)
    hd : Eq (Module.finrank Real (Subtype fun x => Membership.mem s.direction x)) 2
    c₁ c₂ p₁ p₂ p : P
    hc₁s : Membership.mem s c₁
    hc₂s : Membership.mem s c₂
    hp₁s : Membership.mem s p₁
    hp₂s : Membership.mem s p₂
    hps : Membership.mem s p
    r₁ r₂ : Real
    hc : Ne c₁ c₂
    hp : Ne p₁ p₂
    hp₁c₁ : Eq (Dist.dist p₁ c₁) r₁
    hp₂c₁ : Eq (Dist.dist p₂ c₁) r₁
    hpc₁ : Eq (Dist.dist p c₁) r₁
    hp₁c₂ : Eq (Dist.dist p₁ c₂) r₂
    hp₂c₂ : Eq (Dist.dist p₂ c₂) r₂
    hpc₂ : Eq (Dist.dist p c₂) r₂
    ho : Eq (Inner.inner (VSub.vsub c₂ c₁) (VSub.vsub p₂ p₁)) 0
    b : Fin 2 → V := Matrix.vecCons (VSub.vsub c₂ c₁) (Matrix.vecCons (VSub.vsub p …
    hb : LinearIndependent Real b
    hbs : Eq (Submodule.span Real (Set.range b)) s.direction
    hv : ∀ (v : V), Membership.mem s.direction v → Exists fun t₁ => Exists fun t₂  …
    t₁ t₂ : Real
    hpt : Eq (VSub.vsub p p₁) (HAdd.hAdd (HSMul.hSMul t₁ (VSub.vsub c₂ c₁)) (HSMul …
    hop : Eq t₁ 0
    ⊢ Or (Eq p p₁) (Eq p p₂)
  -/
  rw [hop, zero_smul, zero_add, ← eq_vadd_iff_vsub_eq] at hpt
  /-
    case intro.intro
    V : Type u_1
    P : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : MetricSpace P
    inst✝¹ : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝ : FiniteDimensional Real (Subtype fun x => Membership.mem s.direction x)
    hd : Eq (Module.finrank Real (Subtype fun x => Membership.mem s.direction x)) 2
    c₁ c₂ p₁ p₂ p : P
    hc₁s : Membership.mem s c₁
    hc₂s : Membership.mem s c₂
    hp₁s : Membership.mem s p₁
    hp₂s : Membership.mem s p₂
    hps : Membership.mem s p
    r₁ r₂ : Real
    hc : Ne c₁ c₂
    hp : Ne p₁ p₂
    hp₁c₁ : Eq (Dist.dist p₁ c₁) r₁
    hp₂c₁ : Eq (Dist.dist p₂ c₁) r₁
    hpc₁ : Eq (Dist.dist p c₁) r₁
    hp₁c₂ : Eq (Dist.dist p₁ c₂) r₂
    hp₂c₂ : Eq (Dist.dist p₂ c₂) r₂
    hpc₂ : Eq (Dist.dist p c₂) r₂
    ho : Eq (Inner.inner (VSub.vsub c₂ c₁) (VSub.vsub p₂ p₁)) 0
    b : Fin 2 → V := Matrix.vecCons (VSub.vsub c₂ c₁) (Matrix.vecCons (VSub.vsub p …
    hb : LinearIndependent Real b
    hbs : Eq (Submodule.span Real (Set.range b)) s.direction
    hv : ∀ (v : V), Membership.mem s.direction v → Exists fun t₁ => Exists fun t₂  …
    t₁ t₂ : Real
    hpt : Eq p (HVAdd.hVAdd (HSMul.hSMul t₂ (VSub.vsub p₂ p₁)) p₁)
    hop : Eq t₁ 0
    ⊢ Or (Eq p p₁) (Eq p p₂)
  -/
  subst hpt
  /-
    case intro.intro
    V : Type u_1
    P : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : MetricSpace P
    inst✝¹ : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝ : FiniteDimensional Real (Subtype fun x => Membership.mem s.direction x)
    hd : Eq (Module.finrank Real (Subtype fun x => Membership.mem s.direction x)) 2
    c₁ c₂ p₁ p₂ : P
    hc₁s : Membership.mem s c₁
    hc₂s : Membership.mem s c₂
    hp₁s : Membership.mem s p₁
    hp₂s : Membership.mem s p₂
    r₁ r₂ : Real
    hc : Ne c₁ c₂
    hp : Ne p₁ p₂
    hp₁c₁ : Eq (Dist.dist p₁ c₁) r₁
    hp₂c₁ : Eq (Dist.dist p₂ c₁) r₁
    hp₁c₂ : Eq (Dist.dist p₁ c₂) r₂
    hp₂c₂ : Eq (Dist.dist p₂ c₂) r₂
    ho : Eq (Inner.inner (VSub.vsub c₂ c₁) (VSub.vsub p₂ p₁)) 0
    b : Fin 2 → V := Matrix.vecCons (VSub.vsub c₂ c₁) (Matrix.vecCons (VSub.vsub p …
    hb : LinearIndependent Real b
    hbs : Eq (Submodule.span Real (Set.range b)) s.direction
    hv : ∀ (v : V), Membership.mem s.direction v → Exists fun t₁ => Exists fun t₂  …
    t₁ t₂ : Real
    hop : Eq t₁ 0
    hps : Membership.mem s (HVAdd.hVAdd (HSMul.hSMul t₂ (VSub.vsub p₂ p₁)) p₁)
    hpc₁ : Eq (Dist.dist (HVAdd.hVAdd (HSMul.hSMul t₂ (VSub.vsub p₂ p₁)) p₁) c₁) r₁
    hpc₂ : Eq (Dist.dist (HVAdd.hVAdd (HSMul.hSMul t₂ (VSub.vsub p₂ p₁)) p₁) c₂) r₂
    ⊢ Or (Eq (HVAdd.hVAdd (HSMul.hSMul t₂ (VSub.vsub p₂ p₁)) p₁) p₁) (Eq (HVAdd.hV …
  -/
  have hp' : (p₂ -ᵥ p₁ : V) ≠ 0 := by simp [hp.symm]
  /-
    case intro.intro
    V : Type u_1
    P : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : MetricSpace P
    inst✝¹ : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝ : FiniteDimensional Real (Subtype fun x => Membership.mem s.direction x)
    hd : Eq (Module.finrank Real (Subtype fun x => Membership.mem s.direction x)) 2
    c₁ c₂ p₁ p₂ : P
    hc₁s : Membership.mem s c₁
    hc₂s : Membership.mem s c₂
    hp₁s : Membership.mem s p₁
    hp₂s : Membership.mem s p₂
    r₁ r₂ : Real
    hc : Ne c₁ c₂
    hp : Ne p₁ p₂
    hp₁c₁ : Eq (Dist.dist p₁ c₁) r₁
    hp₂c₁ : Eq (Dist.dist p₂ c₁) r₁
    hp₁c₂ : Eq (Dist.dist p₁ c₂) r₂
    hp₂c₂ : Eq (Dist.dist p₂ c₂) r₂
    ho : Eq (Inner.inner (VSub.vsub c₂ c₁) (VSub.vsub p₂ p₁)) 0
    b : Fin 2 → V := Matrix.vecCons (VSub.vsub c₂ c₁) (Matrix.vecCons (VSub.vsub p …
    hb : LinearIndependent Real b
    hbs : Eq (Submodule.span Real (Set.range b)) s.direction
    hv : ∀ (v : V), Membership.mem s.direction v → Exists fun t₁ => Exists fun t₂  …
    t₁ t₂ : Real
    hop : Eq t₁ 0
    hps : Membership.mem s (HVAdd.hVAdd (HSMul.hSMul t₂ (VSub.vsub p₂ p₁)) p₁)
    hpc₁ : Eq (Dist.dist (HVAdd.hVAdd (HSMul.hSMul t₂ (VSub.vsub p₂ p₁)) p₁) c₁) r₁
    hpc₂ : Eq (Dist.dist (HVAdd.hVAdd (HSMul.hSMul t₂ (VSub.vsub p₂ p₁)) p₁) c₂) r₂
    hp' : Ne (VSub.vsub p₂ p₁) 0
    ⊢ Or (Eq (HVAdd.hVAdd (HSMul.hSMul t₂ (VSub.vsub p₂ p₁)) p₁) p₁) (Eq (HVAdd.hV …
  -/
  have hp₂ : dist ((1 : ℝ) • (p₂ -ᵥ p₁) +ᵥ p₁) c₁ = r₁ := by simp [hp₂c₁]
  /-
    case intro.intro
    V : Type u_1
    P : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : MetricSpace P
    inst✝¹ : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝ : FiniteDimensional Real (Subtype fun x => Membership.mem s.direction x)
    hd : Eq (Module.finrank Real (Subtype fun x => Membership.mem s.direction x)) 2
    c₁ c₂ p₁ p₂ : P
    hc₁s : Membership.mem s c₁
    hc₂s : Membership.mem s c₂
    hp₁s : Membership.mem s p₁
    hp₂s : Membership.mem s p₂
    r₁ r₂ : Real
    hc : Ne c₁ c₂
    hp : Ne p₁ p₂
    hp₁c₁ : Eq (Dist.dist p₁ c₁) r₁
    hp₂c₁ : Eq (Dist.dist p₂ c₁) r₁
    hp₁c₂ : Eq (Dist.dist p₁ c₂) r₂
    hp₂c₂ : Eq (Dist.dist p₂ c₂) r₂
    ho : Eq (Inner.inner (VSub.vsub c₂ c₁) (VSub.vsub p₂ p₁)) 0
    b : Fin 2 → V := Matrix.vecCons (VSub.vsub c₂ c₁) (Matrix.vecCons (VSub.vsub p …
    hb : LinearIndependent Real b
    hbs : Eq (Submodule.span Real (Set.range b)) s.direction
    hv : ∀ (v : V), Membership.mem s.direction v → Exists fun t₁ => Exists fun t₂  …
    t₁ t₂ : Real
    hop : Eq t₁ 0
    hps : Membership.mem s (HVAdd.hVAdd (HSMul.hSMul t₂ (VSub.vsub p₂ p₁)) p₁)
    hpc₁ : Eq (Dist.dist (HVAdd.hVAdd (HSMul.hSMul t₂ (VSub.vsub p₂ p₁)) p₁) c₁) r₁
    hpc₂ : Eq (Dist.dist (HVAdd.hVAdd (HSMul.hSMul t₂ (VSub.vsub p₂ p₁)) p₁) c₂) r₂
    hp' : Ne (VSub.vsub p₂ p₁) 0
    hp₂ : Eq (Dist.dist (HVAdd.hVAdd (HSMul.hSMul 1 (VSub.vsub p₂ p₁)) p₁) c₁) r₁
    ⊢ Or (Eq (HVAdd.hVAdd (HSMul.hSMul t₂ (VSub.vsub p₂ p₁)) p₁) p₁) (Eq (HVAdd.hV …
  -/
  rw [← hp₁c₁, dist_smul_vadd_eq_dist _ _ hp'] at hpc₁ hp₂
  /-
    case intro.intro
    V : Type u_1
    P : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : MetricSpace P
    inst✝¹ : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝ : FiniteDimensional Real (Subtype fun x => Membership.mem s.direction x)
    hd : Eq (Module.finrank Real (Subtype fun x => Membership.mem s.direction x)) 2
    c₁ c₂ p₁ p₂ : P
    hc₁s : Membership.mem s c₁
    hc₂s : Membership.mem s c₂
    hp₁s : Membership.mem s p₁
    hp₂s : Membership.mem s p₂
    r₁ r₂ : Real
    hc : Ne c₁ c₂
    hp : Ne p₁ p₂
    hp₁c₁ : Eq (Dist.dist p₁ c₁) r₁
    hp₂c₁ : Eq (Dist.dist p₂ c₁) r₁
    hp₁c₂ : Eq (Dist.dist p₁ c₂) r₂
    hp₂c₂ : Eq (Dist.dist p₂ c₂) r₂
    ho : Eq (Inner.inner (VSub.vsub c₂ c₁) (VSub.vsub p₂ p₁)) 0
    b : Fin 2 → V := Matrix.vecCons (VSub.vsub c₂ c₁) (Matrix.vecCons (VSub.vsub p …
    hb : LinearIndependent Real b
    hbs : Eq (Submodule.span Real (Set.range b)) s.direction
    hv : ∀ (v : V), Membership.mem s.direction v → Exists fun t₁ => Exists fun t₂  …
    t₁ t₂ : Real
    hop : Eq t₁ 0
    hps : Membership.mem s (HVAdd.hVAdd (HSMul.hSMul t₂ (VSub.vsub p₂ p₁)) p₁)
    hpc₁ : Or (Eq t₂ 0) (Eq t₂ (HDiv.hDiv (HMul.hMul (-2) (Inner.inner (VSub.vsub  …
    hpc₂ : Eq (Dist.dist (HVAdd.hVAdd (HSMul.hSMul t₂ (VSub.vsub p₂ p₁)) p₁) c₂) r₂
    hp' : Ne (VSub.vsub p₂ p₁) 0
    hp₂ : Or (Eq 1 0) (Eq 1 (HDiv.hDiv (HMul.hMul (-2) (Inner.inner (VSub.vsub p₂  …
    ⊢ Or (Eq (HVAdd.hVAdd (HSMul.hSMul t₂ (VSub.vsub p₂ p₁)) p₁) p₁) (Eq (HVAdd.hV …
  -/
  simp only [one_ne_zero, false_or] at hp₂
  /-
    case intro.intro
    V : Type u_1
    P : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : MetricSpace P
    inst✝¹ : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝ : FiniteDimensional Real (Subtype fun x => Membership.mem s.direction x)
    hd : Eq (Module.finrank Real (Subtype fun x => Membership.mem s.direction x)) 2
    c₁ c₂ p₁ p₂ : P
    hc₁s : Membership.mem s c₁
    hc₂s : Membership.mem s c₂
    hp₁s : Membership.mem s p₁
    hp₂s : Membership.mem s p₂
    r₁ r₂ : Real
    hc : Ne c₁ c₂
    hp : Ne p₁ p₂
    hp₁c₁ : Eq (Dist.dist p₁ c₁) r₁
    hp₂c₁ : Eq (Dist.dist p₂ c₁) r₁
    hp₁c₂ : Eq (Dist.dist p₁ c₂) r₂
    hp₂c₂ : Eq (Dist.dist p₂ c₂) r₂
    ho : Eq (Inner.inner (VSub.vsub c₂ c₁) (VSub.vsub p₂ p₁)) 0
    b : Fin 2 → V := Matrix.vecCons (VSub.vsub c₂ c₁) (Matrix.vecCons (VSub.vsub p …
    hb : LinearIndependent Real b
    hbs : Eq (Submodule.span Real (Set.range b)) s.direction
    hv : ∀ (v : V), Membership.mem s.direction v → Exists fun t₁ => Exists fun t₂  …
    t₁ t₂ : Real
    hop : Eq t₁ 0
    hps : Membership.mem s (HVAdd.hVAdd (HSMul.hSMul t₂ (VSub.vsub p₂ p₁)) p₁)
    hpc₁ : Or (Eq t₂ 0) (Eq t₂ (HDiv.hDiv (HMul.hMul (-2) (Inner.inner (VSub.vsub  …
    hpc₂ : Eq (Dist.dist (HVAdd.hVAdd (HSMul.hSMul t₂ (VSub.vsub p₂ p₁)) p₁) c₂) r₂
    hp' : Ne (VSub.vsub p₂ p₁) 0
    hp₂ : Eq 1 (HDiv.hDiv (HMul.hMul (-2) (Inner.inner (VSub.vsub p₂ p₁) (VSub.vsu …
    ⊢ Or (Eq (HVAdd.hVAdd (HSMul.hSMul t₂ (VSub.vsub p₂ p₁)) p₁) p₁) (Eq (HVAdd.hV …
  -/
  rw [hp₂.symm] at hpc₁
  /-
    case intro.intro
    V : Type u_1
    P : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : InnerProductSpace Real V
    inst✝² : MetricSpace P
    inst✝¹ : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝ : FiniteDimensional Real (Subtype fun x => Membership.mem s.direction x)
    hd : Eq (Module.finrank Real (Subtype fun x => Membership.mem s.direction x)) 2
    c₁ c₂ p₁ p₂ : P
    hc₁s : Membership.mem s c₁
    hc₂s : Membership.mem s c₂
    hp₁s : Membership.mem s p₁
    hp₂s : Membership.mem s p₂
    r₁ r₂ : Real
    hc : Ne c₁ c₂
    hp : Ne p₁ p₂
    hp₁c₁ : Eq (Dist.dist p₁ c₁) r₁
    hp₂c₁ : Eq (Dist.dist p₂ c₁) r₁
    hp₁c₂ : Eq (Dist.dist p₁ c₂) r₂
    hp₂c₂ : Eq (Dist.dist p₂ c₂) r₂
    ho : Eq (Inner.inner (VSub.vsub c₂ c₁) (VSub.vsub p₂ p₁)) 0
    b : Fin 2 → V := Matrix.vecCons (VSub.vsub c₂ c₁) (Matrix.vecCons (VSub.vsub p …
    hb : LinearIndependent Real b
    hbs : Eq (Submodule.span Real (Set.range b)) s.direction
    hv : ∀ (v : V), Membership.mem s.direction v → Exists fun t₁ => Exists fun t₂  …
    t₁ t₂ : Real
    hop : Eq t₁ 0
    hps : Membership.mem s (HVAdd.hVAdd (HSMul.hSMul t₂ (VSub.vsub p₂ p₁)) p₁)
    hpc₁ : Or (Eq t₂ 0) (Eq t₂ 1)
    hpc₂ : Eq (Dist.dist (HVAdd.hVAdd (HSMul.hSMul t₂ (VSub.vsub p₂ p₁)) p₁) c₂) r₂
    hp' : Ne (VSub.vsub p₂ p₁) 0
    hp₂ : Eq 1 (HDiv.hDiv (HMul.hMul (-2) (Inner.inner (VSub.vsub p₂ p₁) (VSub.vsu …
    ⊢ Or (Eq (HVAdd.hVAdd (HSMul.hSMul t₂ (VSub.vsub p₂ p₁)) p₁) p₁) (Eq (HVAdd.hV …
  -/
                                 /-
                                   🎉 no goals
                                 -/
  cases' hpc₁ with hpc₁ hpc₁ <;> simp [hpc₁]
                                 /-
                                   🎉 no goals
                                 -/


/-- Distances `r₁` `r₂` of `p` from two different points `c₁` `c₂` determine at
most two points `p₁` `p₂` in two-dimensional space (two circles intersect in at
most two points). -/
theorem eq_of_dist_eq_of_dist_eq_of_finrank_eq_two [FiniteDimensional ℝ V] (hd : finrank ℝ V = 2)
    {c₁ c₂ p₁ p₂ p : P} {r₁ r₂ : ℝ} (hc : c₁ ≠ c₂) (hp : p₁ ≠ p₂) (hp₁c₁ : dist p₁ c₁ = r₁)
    (hp₂c₁ : dist p₂ c₁ = r₁) (hpc₁ : dist p c₁ = r₁) (hp₁c₂ : dist p₁ c₂ = r₂)
    (hp₂c₂ : dist p₂ c₂ = r₂) (hpc₂ : dist p c₂ = r₂) : p = p₁ ∨ p = p₂ :=
  haveI hd' : finrank ℝ (⊤ : AffineSubspace ℝ P).direction = 2 := by
    /-
      V : Type u_1
      P : Type u_2
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : InnerProductSpace Real V
      inst✝² : MetricSpace P
      inst✝¹ : NormedAddTorsor V P
      inst✝ : FiniteDimensional Real V
      hd : Eq (Module.finrank Real V) 2
      c₁ c₂ p₁ p₂ p : P
      r₁ r₂ : Real
      hc : Ne c₁ c₂
      hp : Ne p₁ p₂
      hp₁c₁ : Eq (Dist.dist p₁ c₁) r₁
      hp₂c₁ : Eq (Dist.dist p₂ c₁) r₁
      hpc₁ : Eq (Dist.dist p c₁) r₁
      hp₁c₂ : Eq (Dist.dist p₁ c₂) r₂
      hp₂c₂ : Eq (Dist.dist p₂ c₂) r₂
      hpc₂ : Eq (Dist.dist p c₂) r₂
      ⊢ Eq (Module.finrank Real (Subtype fun x => Membership.mem Top.top.direction x …
    -/
    rw [direction_top, finrank_top]
    /-
      V : Type u_1
      P : Type u_2
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : InnerProductSpace Real V
      inst✝² : MetricSpace P
      inst✝¹ : NormedAddTorsor V P
      inst✝ : FiniteDimensional Real V
      hd : Eq (Module.finrank Real V) 2
      c₁ c₂ p₁ p₂ p : P
      r₁ r₂ : Real
      hc : Ne c₁ c₂
      hp : Ne p₁ p₂
      hp₁c₁ : Eq (Dist.dist p₁ c₁) r₁
      hp₂c₁ : Eq (Dist.dist p₂ c₁) r₁
      hpc₁ : Eq (Dist.dist p c₁) r₁
      hp₁c₂ : Eq (Dist.dist p₁ c₂) r₂
      hp₂c₂ : Eq (Dist.dist p₂ c₂) r₂
      hpc₂ : Eq (Dist.dist p c₂) r₂
      ⊢ Eq (Module.finrank Real V) 2
    -/
    exact hd
    /-
      🎉 no goals
    -/
  eq_of_dist_eq_of_dist_eq_of_mem_of_finrank_eq_two hd' (mem_top ℝ V _) (mem_top ℝ V _)
    (mem_top ℝ V _) (mem_top ℝ V _) (mem_top ℝ V _) hc hp hp₁c₁ hp₂c₁ hpc₁ hp₁c₂ hp₂c₂ hpc₂


/-- The orthogonal projection of a point onto a nonempty affine
subspace, whose direction is complete, as an unbundled function. This
definition is only intended for use in setting up the bundled version
`orthogonalProjection` and should not be used once that is
defined. -/
def orthogonalProjectionFn (s : AffineSubspace ℝ P) [Nonempty s]
    [HasOrthogonalProjection s.direction] (p : P) : P :=
  Classical.choose <|
    inter_eq_singleton_of_nonempty_of_isCompl (nonempty_subtype.mp ‹_›)
      (mk'_nonempty p s.directionᗮ)
      (by
        /-
          V : Type u_1
          P : Type u_2
          inst✝⁵ : NormedAddCommGroup V
          inst✝⁴ : InnerProductSpace Real V
          inst✝³ : MetricSpace P
          inst✝² : NormedAddTorsor V P
          s : AffineSubspace Real P
          inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
          inst✝ : HasOrthogonalProjection s.direction
          p : P
          ⊢ IsCompl s.direction (AffineSubspace.mk' p s.direction.orthogonal).direction
        -/
        rw [direction_mk' p s.directionᗮ]
        /-
          V : Type u_1
          P : Type u_2
          inst✝⁵ : NormedAddCommGroup V
          inst✝⁴ : InnerProductSpace Real V
          inst✝³ : MetricSpace P
          inst✝² : NormedAddTorsor V P
          s : AffineSubspace Real P
          inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
          inst✝ : HasOrthogonalProjection s.direction
          p : P
          ⊢ IsCompl s.direction s.direction.orthogonal
        -/
        exact Submodule.isCompl_orthogonal_of_completeSpace)
        /-
          🎉 no goals
        -/


/-- The intersection of the subspace and the orthogonal subspace
through the given point is the `orthogonalProjectionFn` of that
point onto the subspace. This lemma is only intended for use in
setting up the bundled version and should not be used once that is
defined. -/
theorem inter_eq_singleton_orthogonalProjectionFn {s : AffineSubspace ℝ P} [Nonempty s]
    [HasOrthogonalProjection s.direction] (p : P) :
    (s : Set P) ∩ mk' p s.directionᗮ = {orthogonalProjectionFn s p} :=
  Classical.choose_spec <|
    inter_eq_singleton_of_nonempty_of_isCompl (nonempty_subtype.mp ‹_›)
      (mk'_nonempty p s.directionᗮ)
      (by
        /-
          V : Type u_1
          P : Type u_2
          inst✝⁵ : NormedAddCommGroup V
          inst✝⁴ : InnerProductSpace Real V
          inst✝³ : MetricSpace P
          inst✝² : NormedAddTorsor V P
          s : AffineSubspace Real P
          inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
          inst✝ : HasOrthogonalProjection s.direction
          p : P
          ⊢ IsCompl s.direction (AffineSubspace.mk' p s.direction.orthogonal).direction
        -/
        rw [direction_mk' p s.directionᗮ]
        /-
          V : Type u_1
          P : Type u_2
          inst✝⁵ : NormedAddCommGroup V
          inst✝⁴ : InnerProductSpace Real V
          inst✝³ : MetricSpace P
          inst✝² : NormedAddTorsor V P
          s : AffineSubspace Real P
          inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
          inst✝ : HasOrthogonalProjection s.direction
          p : P
          ⊢ IsCompl s.direction s.direction.orthogonal
        -/
        exact Submodule.isCompl_orthogonal_of_completeSpace)
        /-
          🎉 no goals
        -/


/-- The `orthogonalProjectionFn` lies in the given subspace. This
lemma is only intended for use in setting up the bundled version and
should not be used once that is defined. -/
theorem orthogonalProjectionFn_mem {s : AffineSubspace ℝ P} [Nonempty s]
    [HasOrthogonalProjection s.direction] (p : P) : orthogonalProjectionFn s p ∈ s := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : InnerProductSpace Real V
    inst✝³ : MetricSpace P
    inst✝² : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
    inst✝ : HasOrthogonalProjection s.direction
    p : P
    ⊢ Membership.mem s (EuclideanGeometry.orthogonalProjectionFn s p)
  -/
  rw [← mem_coe, ← Set.singleton_subset_iff, ← inter_eq_singleton_orthogonalProjectionFn]
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : InnerProductSpace Real V
    inst✝³ : MetricSpace P
    inst✝² : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
    inst✝ : HasOrthogonalProjection s.direction
    p : P
    ⊢ HasSubset.Subset (Inter.inter ↑s ↑(AffineSubspace.mk' p s.direction.orthogon …
  -/
  exact Set.inter_subset_left
  /-
    🎉 no goals
  -/


/-- The `orthogonalProjectionFn` lies in the orthogonal
subspace. This lemma is only intended for use in setting up the
bundled version and should not be used once that is defined. -/
theorem orthogonalProjectionFn_mem_orthogonal {s : AffineSubspace ℝ P} [Nonempty s]
    [HasOrthogonalProjection s.direction] (p : P) :
    orthogonalProjectionFn s p ∈ mk' p s.directionᗮ := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : InnerProductSpace Real V
    inst✝³ : MetricSpace P
    inst✝² : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
    inst✝ : HasOrthogonalProjection s.direction
    p : P
    ⊢ Membership.mem (AffineSubspace.mk' p s.direction.orthogonal) (EuclideanGeome …
  -/
  rw [← mem_coe, ← Set.singleton_subset_iff, ← inter_eq_singleton_orthogonalProjectionFn]
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : InnerProductSpace Real V
    inst✝³ : MetricSpace P
    inst✝² : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
    inst✝ : HasOrthogonalProjection s.direction
    p : P
    ⊢ HasSubset.Subset (Inter.inter ↑s ↑(AffineSubspace.mk' p s.direction.orthogon …
  -/
  exact Set.inter_subset_right
  /-
    🎉 no goals
  -/


/-- Subtracting `p` from its `orthogonalProjectionFn` produces a
result in the orthogonal direction. This lemma is only intended for
use in setting up the bundled version and should not be used once that
is defined. -/
theorem orthogonalProjectionFn_vsub_mem_direction_orthogonal {s : AffineSubspace ℝ P} [Nonempty s]
    [HasOrthogonalProjection s.direction] (p : P) :
    orthogonalProjectionFn s p -ᵥ p ∈ s.directionᗮ :=
  direction_mk' p s.directionᗮ ▸
    vsub_mem_direction (orthogonalProjectionFn_mem_orthogonal p) (self_mem_mk' _ _)


/-- The orthogonal projection of a point onto a nonempty affine
subspace, whose direction is complete. The corresponding linear map
(mapping a vector to the difference between the projections of two
points whose difference is that vector) is the `orthogonalProjection`
for real inner product spaces, onto the direction of the affine
subspace being projected onto. -/
nonrec def orthogonalProjection (s : AffineSubspace ℝ P) [Nonempty s]
    [HasOrthogonalProjection s.direction] : P →ᵃ[ℝ] s where
  toFun p := ⟨orthogonalProjectionFn s p, orthogonalProjectionFn_mem p⟩
  linear := orthogonalProjection s.direction
  map_vadd' p v := by
    have hs : ((orthogonalProjection s.direction) v : V) +ᵥ orthogonalProjectionFn s p ∈ s :=
      vadd_mem_of_mem_direction (orthogonalProjection s.direction v).2
        (orthogonalProjectionFn_mem p)
    have ho :
      ((orthogonalProjection s.direction) v : V) +ᵥ orthogonalProjectionFn s p ∈
        mk' (v +ᵥ p) s.directionᗮ := by
      rw [← vsub_right_mem_direction_iff_mem (self_mem_mk' _ _) _, direction_mk',
        vsub_vadd_eq_vsub_sub, vadd_vsub_assoc, add_comm, add_sub_assoc]
      refine Submodule.add_mem _ (orthogonalProjectionFn_vsub_mem_direction_orthogonal p) ?_
      rw [Submodule.mem_orthogonal']
      intro w hw
      rw [← neg_sub, inner_neg_left, orthogonalProjection_inner_eq_zero _ w hw, neg_zero]
    have hm :
      ((orthogonalProjection s.direction) v : V) +ᵥ orthogonalProjectionFn s p ∈
        ({orthogonalProjectionFn s (v +ᵥ p)} : Set P) := by
      rw [← inter_eq_singleton_orthogonalProjectionFn (v +ᵥ p)]
      exact Set.mem_inter hs ho
    /-
      V : Type u_1
      P : Type u_2
      inst✝⁵ : NormedAddCommGroup V
      inst✝⁴ : InnerProductSpace Real V
      inst✝³ : MetricSpace P
      inst✝² : NormedAddTorsor V P
      s : AffineSubspace Real P
      inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
      inst✝ : HasOrthogonalProjection s.direction
      p : P
      v : V
      hs : Membership.mem s (HVAdd.hVAdd (↑((_root_.orthogonalProjection s.direction …
      ho : Membership.mem (AffineSubspace.mk' (HVAdd.hVAdd v p) s.direction.orthogon …
      hm : Membership.mem (Singleton.singleton (EuclideanGeometry.orthogonalProjecti …
      ⊢ Eq ((fun p => ⟨EuclideanGeometry.orthogonalProjectionFn s p, ⋯⟩) (HVAdd.hVAd …
    -/
    rw [Set.mem_singleton_iff] at hm
    /-
      V : Type u_1
      P : Type u_2
      inst✝⁵ : NormedAddCommGroup V
      inst✝⁴ : InnerProductSpace Real V
      inst✝³ : MetricSpace P
      inst✝² : NormedAddTorsor V P
      s : AffineSubspace Real P
      inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
      inst✝ : HasOrthogonalProjection s.direction
      p : P
      v : V
      hs : Membership.mem s (HVAdd.hVAdd (↑((_root_.orthogonalProjection s.direction …
      ho : Membership.mem (AffineSubspace.mk' (HVAdd.hVAdd v p) s.direction.orthogon …
      hm : Eq (HVAdd.hVAdd (↑((_root_.orthogonalProjection s.direction) v)) (Euclide …
      ⊢ Eq ((fun p => ⟨EuclideanGeometry.orthogonalProjectionFn s p, ⋯⟩) (HVAdd.hVAd …
    -/
    ext
    /-
      case a
      V : Type u_1
      P : Type u_2
      inst✝⁵ : NormedAddCommGroup V
      inst✝⁴ : InnerProductSpace Real V
      inst✝³ : MetricSpace P
      inst✝² : NormedAddTorsor V P
      s : AffineSubspace Real P
      inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
      inst✝ : HasOrthogonalProjection s.direction
      p : P
      v : V
      hs : Membership.mem s (HVAdd.hVAdd (↑((_root_.orthogonalProjection s.direction …
      ho : Membership.mem (AffineSubspace.mk' (HVAdd.hVAdd v p) s.direction.orthogon …
      hm : Eq (HVAdd.hVAdd (↑((_root_.orthogonalProjection s.direction) v)) (Euclide …
      ⊢ Eq ↑((fun p => ⟨EuclideanGeometry.orthogonalProjectionFn s p, ⋯⟩) (HVAdd.hVA …
    -/
    exact hm.symm
    /-
      🎉 no goals
    -/


@[simp]
theorem orthogonalProjectionFn_eq {s : AffineSubspace ℝ P} [Nonempty s]
    [HasOrthogonalProjection s.direction] (p : P) :
    orthogonalProjectionFn s p = orthogonalProjection s p :=
  rfl


/-- The linear map corresponding to `orthogonalProjection`. -/
@[simp]
theorem orthogonalProjection_linear {s : AffineSubspace ℝ P} [Nonempty s]
    [HasOrthogonalProjection s.direction] :
    (orthogonalProjection s).linear = _root_.orthogonalProjection s.direction :=
  rfl


/-- The intersection of the subspace and the orthogonal subspace
through the given point is the `orthogonalProjection` of that point
onto the subspace. -/
theorem inter_eq_singleton_orthogonalProjection {s : AffineSubspace ℝ P} [Nonempty s]
    [HasOrthogonalProjection s.direction] (p : P) :
    (s : Set P) ∩ mk' p s.directionᗮ = {↑(orthogonalProjection s p)} := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : InnerProductSpace Real V
    inst✝³ : MetricSpace P
    inst✝² : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
    inst✝ : HasOrthogonalProjection s.direction
    p : P
    ⊢ Eq (Inter.inter ↑s ↑(AffineSubspace.mk' p s.direction.orthogonal)) (Singleto …
  -/
  rw [← orthogonalProjectionFn_eq]
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : InnerProductSpace Real V
    inst✝³ : MetricSpace P
    inst✝² : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
    inst✝ : HasOrthogonalProjection s.direction
    p : P
    ⊢ Eq (Inter.inter ↑s ↑(AffineSubspace.mk' p s.direction.orthogonal)) (Singleto …
  -/
  exact inter_eq_singleton_orthogonalProjectionFn p
  /-
    🎉 no goals
  -/


/-- The `orthogonalProjection` lies in the given subspace. -/
theorem orthogonalProjection_mem {s : AffineSubspace ℝ P} [Nonempty s]
    [HasOrthogonalProjection s.direction] (p : P) : ↑(orthogonalProjection s p) ∈ s :=
  (orthogonalProjection s p).2


/-- The `orthogonalProjection` lies in the orthogonal subspace. -/
theorem orthogonalProjection_mem_orthogonal (s : AffineSubspace ℝ P) [Nonempty s]
    [HasOrthogonalProjection s.direction] (p : P) :
    ↑(orthogonalProjection s p) ∈ mk' p s.directionᗮ :=
  orthogonalProjectionFn_mem_orthogonal p


/-- Subtracting a point in the given subspace from the
`orthogonalProjection` produces a result in the direction of the
given subspace. -/
theorem orthogonalProjection_vsub_mem_direction {s : AffineSubspace ℝ P} [Nonempty s]
    [HasOrthogonalProjection s.direction] {p1 : P} (p2 : P) (hp1 : p1 ∈ s) :
    ↑(orthogonalProjection s p2 -ᵥ ⟨p1, hp1⟩ : s.direction) ∈ s.direction :=
  (orthogonalProjection s p2 -ᵥ ⟨p1, hp1⟩ : s.direction).2


/-- Subtracting the `orthogonalProjection` from a point in the given
subspace produces a result in the direction of the given subspace. -/
theorem vsub_orthogonalProjection_mem_direction {s : AffineSubspace ℝ P} [Nonempty s]
    [HasOrthogonalProjection s.direction] {p1 : P} (p2 : P) (hp1 : p1 ∈ s) :
    ↑((⟨p1, hp1⟩ : s) -ᵥ orthogonalProjection s p2 : s.direction) ∈ s.direction :=
  ((⟨p1, hp1⟩ : s) -ᵥ orthogonalProjection s p2 : s.direction).2


/-- A point equals its orthogonal projection if and only if it lies in
the subspace. -/
theorem orthogonalProjection_eq_self_iff {s : AffineSubspace ℝ P} [Nonempty s]
    [HasOrthogonalProjection s.direction] {p : P} : ↑(orthogonalProjection s p) = p ↔ p ∈ s := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : InnerProductSpace Real V
    inst✝³ : MetricSpace P
    inst✝² : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
    inst✝ : HasOrthogonalProjection s.direction
    p : P
    ⊢ Iff (Eq (↑((EuclideanGeometry.orthogonalProjection s) p)) p) (Membership.mem …
  -/
  constructor
    /-
      case mp
      V : Type u_1
      P : Type u_2
      inst✝⁵ : NormedAddCommGroup V
      inst✝⁴ : InnerProductSpace Real V
      inst✝³ : MetricSpace P
      inst✝² : NormedAddTorsor V P
      s : AffineSubspace Real P
      inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
      inst✝ : HasOrthogonalProjection s.direction
      p : P
      ⊢ Eq (↑((EuclideanGeometry.orthogonalProjection s) p)) p → Membership.mem s p
    -/
  · exact fun h => h ▸ orthogonalProjection_mem p
    /-
      🎉 no goals
    -/
    /-
      case mpr
      V : Type u_1
      P : Type u_2
      inst✝⁵ : NormedAddCommGroup V
      inst✝⁴ : InnerProductSpace Real V
      inst✝³ : MetricSpace P
      inst✝² : NormedAddTorsor V P
      s : AffineSubspace Real P
      inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
      inst✝ : HasOrthogonalProjection s.direction
      p : P
      ⊢ Membership.mem s p → Eq (↑((EuclideanGeometry.orthogonalProjection s) p)) p
    -/
  · intro h
    /-
      case mpr
      V : Type u_1
      P : Type u_2
      inst✝⁵ : NormedAddCommGroup V
      inst✝⁴ : InnerProductSpace Real V
      inst✝³ : MetricSpace P
      inst✝² : NormedAddTorsor V P
      s : AffineSubspace Real P
      inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
      inst✝ : HasOrthogonalProjection s.direction
      p : P
      h : Membership.mem s p
      ⊢ Eq (↑((EuclideanGeometry.orthogonalProjection s) p)) p
    -/
    have hp : p ∈ (s : Set P) ∩ mk' p s.directionᗮ := ⟨h, self_mem_mk' p _⟩
    /-
      case mpr
      V : Type u_1
      P : Type u_2
      inst✝⁵ : NormedAddCommGroup V
      inst✝⁴ : InnerProductSpace Real V
      inst✝³ : MetricSpace P
      inst✝² : NormedAddTorsor V P
      s : AffineSubspace Real P
      inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
      inst✝ : HasOrthogonalProjection s.direction
      p : P
      h : Membership.mem s p
      hp : Membership.mem (Inter.inter ↑s ↑(AffineSubspace.mk' p s.direction.orthogo …
      ⊢ Eq (↑((EuclideanGeometry.orthogonalProjection s) p)) p
    -/
    rw [inter_eq_singleton_orthogonalProjection p] at hp
    /-
      case mpr
      V : Type u_1
      P : Type u_2
      inst✝⁵ : NormedAddCommGroup V
      inst✝⁴ : InnerProductSpace Real V
      inst✝³ : MetricSpace P
      inst✝² : NormedAddTorsor V P
      s : AffineSubspace Real P
      inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
      inst✝ : HasOrthogonalProjection s.direction
      p : P
      h : Membership.mem s p
      hp : Membership.mem (Singleton.singleton ↑((EuclideanGeometry.orthogonalProjec …
      ⊢ Eq (↑((EuclideanGeometry.orthogonalProjection s) p)) p
    -/
    symm
    /-
      case mpr
      V : Type u_1
      P : Type u_2
      inst✝⁵ : NormedAddCommGroup V
      inst✝⁴ : InnerProductSpace Real V
      inst✝³ : MetricSpace P
      inst✝² : NormedAddTorsor V P
      s : AffineSubspace Real P
      inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
      inst✝ : HasOrthogonalProjection s.direction
      p : P
      h : Membership.mem s p
      hp : Membership.mem (Singleton.singleton ↑((EuclideanGeometry.orthogonalProjec …
      ⊢ Eq p ↑((EuclideanGeometry.orthogonalProjection s) p)
    -/
    exact hp
    /-
      🎉 no goals
    -/


@[simp]
theorem orthogonalProjection_mem_subspace_eq_self {s : AffineSubspace ℝ P} [Nonempty s]
    [HasOrthogonalProjection s.direction] (p : s) : orthogonalProjection s p = p := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : InnerProductSpace Real V
    inst✝³ : MetricSpace P
    inst✝² : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
    inst✝ : HasOrthogonalProjection s.direction
    p : Subtype fun x => Membership.mem s x
    ⊢ Eq ((EuclideanGeometry.orthogonalProjection s) ↑p) p
  -/
  ext
  /-
    case a
    V : Type u_1
    P : Type u_2
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : InnerProductSpace Real V
    inst✝³ : MetricSpace P
    inst✝² : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
    inst✝ : HasOrthogonalProjection s.direction
    p : Subtype fun x => Membership.mem s x
    ⊢ Eq ↑((EuclideanGeometry.orthogonalProjection s) ↑p) ↑p
  -/
  rw [orthogonalProjection_eq_self_iff]
  /-
    case a
    V : Type u_1
    P : Type u_2
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : InnerProductSpace Real V
    inst✝³ : MetricSpace P
    inst✝² : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
    inst✝ : HasOrthogonalProjection s.direction
    p : Subtype fun x => Membership.mem s x
    ⊢ Membership.mem s ↑p
  -/
  exact p.2
  /-
    🎉 no goals
  -/


/-- Orthogonal projection is idempotent. -/
theorem orthogonalProjection_orthogonalProjection (s : AffineSubspace ℝ P) [Nonempty s]
    [HasOrthogonalProjection s.direction] (p : P) :
    orthogonalProjection s (orthogonalProjection s p) = orthogonalProjection s p := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : InnerProductSpace Real V
    inst✝³ : MetricSpace P
    inst✝² : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
    inst✝ : HasOrthogonalProjection s.direction
    p : P
    ⊢ Eq ((EuclideanGeometry.orthogonalProjection s) ↑((EuclideanGeometry.orthogon …
  -/
  ext
  /-
    case a
    V : Type u_1
    P : Type u_2
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : InnerProductSpace Real V
    inst✝³ : MetricSpace P
    inst✝² : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
    inst✝ : HasOrthogonalProjection s.direction
    p : P
    ⊢ Eq ↑((EuclideanGeometry.orthogonalProjection s) ↑((EuclideanGeometry.orthogo …
  -/
  rw [orthogonalProjection_eq_self_iff]
  /-
    case a
    V : Type u_1
    P : Type u_2
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : InnerProductSpace Real V
    inst✝³ : MetricSpace P
    inst✝² : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
    inst✝ : HasOrthogonalProjection s.direction
    p : P
    ⊢ Membership.mem s ↑((EuclideanGeometry.orthogonalProjection s) p)
  -/
  exact orthogonalProjection_mem p
  /-
    🎉 no goals
  -/


theorem eq_orthogonalProjection_of_eq_subspace {s s' : AffineSubspace ℝ P} [Nonempty s]
    [Nonempty s'] [HasOrthogonalProjection s.direction] [HasOrthogonalProjection s'.direction]
    (h : s = s') (p : P) : (orthogonalProjection s p : P) = (orthogonalProjection s' p : P) := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁷ : NormedAddCommGroup V
    inst✝⁶ : InnerProductSpace Real V
    inst✝⁵ : MetricSpace P
    inst✝⁴ : NormedAddTorsor V P
    s s' : AffineSubspace Real P
    inst✝³ : Nonempty (Subtype fun x => Membership.mem s x)
    inst✝² : Nonempty (Subtype fun x => Membership.mem s' x)
    inst✝¹ : HasOrthogonalProjection s.direction
    inst✝ : HasOrthogonalProjection s'.direction
    h : Eq s s'
    p : P
    ⊢ Eq ↑((EuclideanGeometry.orthogonalProjection s) p) ↑((EuclideanGeometry.orth …
  -/
  subst h
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁷ : NormedAddCommGroup V
    inst✝⁶ : InnerProductSpace Real V
    inst✝⁵ : MetricSpace P
    inst✝⁴ : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝³ : Nonempty (Subtype fun x => Membership.mem s x)
    inst✝² : HasOrthogonalProjection s.direction
    p : P
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
    inst✝ : HasOrthogonalProjection s.direction
    ⊢ Eq ↑((EuclideanGeometry.orthogonalProjection s) p) ↑((EuclideanGeometry.orth …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The distance to a point's orthogonal projection is 0 iff it lies in the subspace. -/
theorem dist_orthogonalProjection_eq_zero_iff {s : AffineSubspace ℝ P} [Nonempty s]
    [HasOrthogonalProjection s.direction] {p : P} :
    dist p (orthogonalProjection s p) = 0 ↔ p ∈ s := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : InnerProductSpace Real V
    inst✝³ : MetricSpace P
    inst✝² : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
    inst✝ : HasOrthogonalProjection s.direction
    p : P
    ⊢ Iff (Eq (Dist.dist p ↑((EuclideanGeometry.orthogonalProjection s) p)) 0) (Me …
  -/
  rw [dist_comm, dist_eq_zero, orthogonalProjection_eq_self_iff]
  /-
    🎉 no goals
  -/


/-- The distance between a point and its orthogonal projection is
nonzero if it does not lie in the subspace. -/
theorem dist_orthogonalProjection_ne_zero_of_not_mem {s : AffineSubspace ℝ P} [Nonempty s]
    [HasOrthogonalProjection s.direction] {p : P} (hp : p ∉ s) :
    dist p (orthogonalProjection s p) ≠ 0 :=
  mt dist_orthogonalProjection_eq_zero_iff.mp hp


/-- Subtracting `p` from its `orthogonalProjection` produces a result
in the orthogonal direction. -/
theorem orthogonalProjection_vsub_mem_direction_orthogonal (s : AffineSubspace ℝ P) [Nonempty s]
    [HasOrthogonalProjection s.direction] (p : P) :
    (orthogonalProjection s p : P) -ᵥ p ∈ s.directionᗮ :=
  orthogonalProjectionFn_vsub_mem_direction_orthogonal p


/-- Subtracting the `orthogonalProjection` from `p` produces a result
in the orthogonal direction. -/
theorem vsub_orthogonalProjection_mem_direction_orthogonal (s : AffineSubspace ℝ P) [Nonempty s]
    [HasOrthogonalProjection s.direction] (p : P) : p -ᵥ orthogonalProjection s p ∈ s.directionᗮ :=
  direction_mk' p s.directionᗮ ▸
    vsub_mem_direction (self_mem_mk' _ _) (orthogonalProjection_mem_orthogonal s p)


/-- Subtracting the `orthogonalProjection` from `p` produces a result in the kernel of the linear
part of the orthogonal projection. -/
theorem orthogonalProjection_vsub_orthogonalProjection (s : AffineSubspace ℝ P) [Nonempty s]
    [HasOrthogonalProjection s.direction] (p : P) :
    _root_.orthogonalProjection s.direction (p -ᵥ orthogonalProjection s p) = 0 := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : InnerProductSpace Real V
    inst✝³ : MetricSpace P
    inst✝² : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
    inst✝ : HasOrthogonalProjection s.direction
    p : P
    ⊢ Eq ((_root_.orthogonalProjection s.direction) (VSub.vsub p ↑((EuclideanGeome …
  -/
  apply orthogonalProjection_mem_subspace_orthogonalComplement_eq_zero
  /-
    case hv
    V : Type u_1
    P : Type u_2
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : InnerProductSpace Real V
    inst✝³ : MetricSpace P
    inst✝² : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
    inst✝ : HasOrthogonalProjection s.direction
    p : P
    ⊢ Membership.mem s.direction.orthogonal (VSub.vsub p ↑((EuclideanGeometry.orth …
  -/
  intro c hc
  rw [← neg_vsub_eq_vsub_rev, inner_neg_right,
    orthogonalProjection_vsub_mem_direction_orthogonal s p c hc, neg_zero]


/-- Adding a vector to a point in the given subspace, then taking the
orthogonal projection, produces the original point if the vector was
in the orthogonal direction. -/
theorem orthogonalProjection_vadd_eq_self {s : AffineSubspace ℝ P} [Nonempty s]
    [HasOrthogonalProjection s.direction] {p : P} (hp : p ∈ s) {v : V} (hv : v ∈ s.directionᗮ) :
    orthogonalProjection s (v +ᵥ p) = ⟨p, hp⟩ := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : InnerProductSpace Real V
    inst✝³ : MetricSpace P
    inst✝² : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
    inst✝ : HasOrthogonalProjection s.direction
    p : P
    hp : Membership.mem s p
    v : V
    hv : Membership.mem s.direction.orthogonal v
    ⊢ Eq ((EuclideanGeometry.orthogonalProjection s) (HVAdd.hVAdd v p)) ⟨p, hp⟩
  -/
  have h := vsub_orthogonalProjection_mem_direction_orthogonal s (v +ᵥ p)
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : InnerProductSpace Real V
    inst✝³ : MetricSpace P
    inst✝² : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
    inst✝ : HasOrthogonalProjection s.direction
    p : P
    hp : Membership.mem s p
    v : V
    hv : Membership.mem s.direction.orthogonal v
    h : Membership.mem s.direction.orthogonal (VSub.vsub (HVAdd.hVAdd v p) ↑((Eucl …
    ⊢ Eq ((EuclideanGeometry.orthogonalProjection s) (HVAdd.hVAdd v p)) ⟨p, hp⟩
  -/
  rw [vadd_vsub_assoc, Submodule.add_mem_iff_right _ hv] at h
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : InnerProductSpace Real V
    inst✝³ : MetricSpace P
    inst✝² : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
    inst✝ : HasOrthogonalProjection s.direction
    p : P
    hp : Membership.mem s p
    v : V
    hv : Membership.mem s.direction.orthogonal v
    h : Membership.mem s.direction.orthogonal (VSub.vsub p ↑((EuclideanGeometry.or …
    ⊢ Eq ((EuclideanGeometry.orthogonalProjection s) (HVAdd.hVAdd v p)) ⟨p, hp⟩
  -/
  refine (eq_of_vsub_eq_zero ?_).symm
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : InnerProductSpace Real V
    inst✝³ : MetricSpace P
    inst✝² : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
    inst✝ : HasOrthogonalProjection s.direction
    p : P
    hp : Membership.mem s p
    v : V
    hv : Membership.mem s.direction.orthogonal v
    h : Membership.mem s.direction.orthogonal (VSub.vsub p ↑((EuclideanGeometry.or …
    ⊢ Eq (VSub.vsub ⟨p, hp⟩ ((EuclideanGeometry.orthogonalProjection s) (HVAdd.hVA …
  -/
  ext
  /-
    case a
    V : Type u_1
    P : Type u_2
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : InnerProductSpace Real V
    inst✝³ : MetricSpace P
    inst✝² : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
    inst✝ : HasOrthogonalProjection s.direction
    p : P
    hp : Membership.mem s p
    v : V
    hv : Membership.mem s.direction.orthogonal v
    h : Membership.mem s.direction.orthogonal (VSub.vsub p ↑((EuclideanGeometry.or …
    ⊢ Eq ↑(VSub.vsub ⟨p, hp⟩ ((EuclideanGeometry.orthogonalProjection s) (HVAdd.hV …
  -/
  refine Submodule.disjoint_def.1 s.direction.orthogonal_disjoint _ ?_ h
  /-
    case a
    V : Type u_1
    P : Type u_2
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : InnerProductSpace Real V
    inst✝³ : MetricSpace P
    inst✝² : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
    inst✝ : HasOrthogonalProjection s.direction
    p : P
    hp : Membership.mem s p
    v : V
    hv : Membership.mem s.direction.orthogonal v
    h : Membership.mem s.direction.orthogonal (VSub.vsub p ↑((EuclideanGeometry.or …
    ⊢ Membership.mem s.direction ↑(VSub.vsub ⟨p, hp⟩ ((EuclideanGeometry.orthogona …
  -/
  exact (_ : s.direction).2
  /-
    🎉 no goals
  -/


/-- Adding a vector to a point in the given subspace, then taking the
orthogonal projection, produces the original point if the vector is a
multiple of the result of subtracting a point's orthogonal projection
from that point. -/
theorem orthogonalProjection_vadd_smul_vsub_orthogonalProjection {s : AffineSubspace ℝ P}
    [Nonempty s] [HasOrthogonalProjection s.direction] {p1 : P} (p2 : P) (r : ℝ) (hp : p1 ∈ s) :
    orthogonalProjection s (r • (p2 -ᵥ orthogonalProjection s p2 : V) +ᵥ p1) = ⟨p1, hp⟩ :=
  orthogonalProjection_vadd_eq_self hp
    (Submodule.smul_mem _ _ (vsub_orthogonalProjection_mem_direction_orthogonal s _))


/-- The square of the distance from a point in `s` to `p2` equals the
sum of the squares of the distances of the two points to the
`orthogonalProjection`. -/
theorem dist_sq_eq_dist_orthogonalProjection_sq_add_dist_orthogonalProjection_sq
    {s : AffineSubspace ℝ P} [Nonempty s] [HasOrthogonalProjection s.direction] {p1 : P} (p2 : P)
    (hp1 : p1 ∈ s) :
    dist p1 p2 * dist p1 p2 =
      dist p1 (orthogonalProjection s p2) * dist p1 (orthogonalProjection s p2) +
        dist p2 (orthogonalProjection s p2) * dist p2 (orthogonalProjection s p2) := by
  rw [dist_comm p2 _, dist_eq_norm_vsub V p1 _, dist_eq_norm_vsub V p1 _, dist_eq_norm_vsub V _ p2,
    ← vsub_add_vsub_cancel p1 (orthogonalProjection s p2) p2,
    norm_add_sq_eq_norm_sq_add_norm_sq_iff_real_inner_eq_zero]
  exact Submodule.inner_right_of_mem_orthogonal (vsub_orthogonalProjection_mem_direction p2 hp1)
    (orthogonalProjection_vsub_mem_direction_orthogonal s p2)


/-- The square of the distance between two points constructed by
adding multiples of the same orthogonal vector to points in the same
subspace. -/
theorem dist_sq_smul_orthogonal_vadd_smul_orthogonal_vadd {s : AffineSubspace ℝ P} {p1 p2 : P}
    (hp1 : p1 ∈ s) (hp2 : p2 ∈ s) (r1 r2 : ℝ) {v : V} (hv : v ∈ s.directionᗮ) :
    dist (r1 • v +ᵥ p1) (r2 • v +ᵥ p2) * dist (r1 • v +ᵥ p1) (r2 • v +ᵥ p2) =
      dist p1 p2 * dist p1 p2 + (r1 - r2) * (r1 - r2) * (‖v‖ * ‖v‖) :=
  calc
    dist (r1 • v +ᵥ p1) (r2 • v +ᵥ p2) * dist (r1 • v +ᵥ p1) (r2 • v +ᵥ p2) =
        ‖p1 -ᵥ p2 + (r1 - r2) • v‖ * ‖p1 -ᵥ p2 + (r1 - r2) • v‖ := by
      rw [dist_eq_norm_vsub V (r1 • v +ᵥ p1), vsub_vadd_eq_vsub_sub, vadd_vsub_assoc, sub_smul,
        add_comm, add_sub_assoc]
    _ = ‖p1 -ᵥ p2‖ * ‖p1 -ᵥ p2‖ + ‖(r1 - r2) • v‖ * ‖(r1 - r2) • v‖ :=
      (norm_add_sq_eq_norm_sq_add_norm_sq_real
        (Submodule.inner_right_of_mem_orthogonal (vsub_mem_direction hp1 hp2)
          (Submodule.smul_mem _ _ hv)))
    _ = ‖(p1 -ᵥ p2 : V)‖ * ‖(p1 -ᵥ p2 : V)‖ + |r1 - r2| * |r1 - r2| * ‖v‖ * ‖v‖ := by
      /-
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : InnerProductSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        s : AffineSubspace Real P
        p1 p2 : P
        hp1 : Membership.mem s p1
        hp2 : Membership.mem s p2
        r1 r2 : Real
        v : V
        hv : Membership.mem s.direction.orthogonal v
        ⊢ Eq (HAdd.hAdd (HMul.hMul (Norm.norm (VSub.vsub p1 p2)) (Norm.norm (VSub.vsub …
      -/
      rw [norm_smul, Real.norm_eq_abs]
      /-
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : InnerProductSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        s : AffineSubspace Real P
        p1 p2 : P
        hp1 : Membership.mem s p1
        hp2 : Membership.mem s p2
        r1 r2 : Real
        v : V
        hv : Membership.mem s.direction.orthogonal v
        ⊢ Eq (HAdd.hAdd (HMul.hMul (Norm.norm (VSub.vsub p1 p2)) (Norm.norm (VSub.vsub …
      -/
      ring
      /-
        🎉 no goals
      -/
    _ = dist p1 p2 * dist p1 p2 + (r1 - r2) * (r1 - r2) * (‖v‖ * ‖v‖) := by
      /-
        V : Type u_1
        P : Type u_2
        inst✝³ : NormedAddCommGroup V
        inst✝² : InnerProductSpace Real V
        inst✝¹ : MetricSpace P
        inst✝ : NormedAddTorsor V P
        s : AffineSubspace Real P
        p1 p2 : P
        hp1 : Membership.mem s p1
        hp2 : Membership.mem s p2
        r1 r2 : Real
        v : V
        hv : Membership.mem s.direction.orthogonal v
        ⊢ Eq (HAdd.hAdd (HMul.hMul (Norm.norm (VSub.vsub p1 p2)) (Norm.norm (VSub.vsub …
      -/
      rw [dist_eq_norm_vsub V p1, abs_mul_abs_self, mul_assoc]
      /-
        🎉 no goals
      -/


/-- Reflection in an affine subspace, which is expected to be nonempty
and complete. The word "reflection" is sometimes understood to mean
specifically reflection in a codimension-one subspace, and sometimes
more generally to cover operations such as reflection in a point. The
definition here, of reflection in an affine subspace, is a more
general sense of the word that includes both those common cases. -/
def reflection (s : AffineSubspace ℝ P) [Nonempty s] [HasOrthogonalProjection s.direction] :
    P ≃ᵃⁱ[ℝ] P :=
  AffineIsometryEquiv.mk'
    (fun p => (↑(orthogonalProjection s p) -ᵥ p) +ᵥ (orthogonalProjection s p : P))
    (_root_.reflection s.direction) (↑(Classical.arbitrary s))
    (by
      /-
        V : Type u_1
        P : Type u_2
        inst✝⁵ : NormedAddCommGroup V
        inst✝⁴ : InnerProductSpace Real V
        inst✝³ : MetricSpace P
        inst✝² : NormedAddTorsor V P
        s : AffineSubspace Real P
        inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
        inst✝ : HasOrthogonalProjection s.direction
        ⊢ ∀ (p' : P), Eq ((fun p => HVAdd.hVAdd (VSub.vsub (↑((EuclideanGeometry.ortho …
      -/
      intro p
      /-
        V : Type u_1
        P : Type u_2
        inst✝⁵ : NormedAddCommGroup V
        inst✝⁴ : InnerProductSpace Real V
        inst✝³ : MetricSpace P
        inst✝² : NormedAddTorsor V P
        s : AffineSubspace Real P
        inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
        inst✝ : HasOrthogonalProjection s.direction
        p : P
        ⊢ Eq ((fun p => HVAdd.hVAdd (VSub.vsub (↑((EuclideanGeometry.orthogonalProject …
      -/
      let v := p -ᵥ ↑(Classical.arbitrary s)
      /-
        V : Type u_1
        P : Type u_2
        inst✝⁵ : NormedAddCommGroup V
        inst✝⁴ : InnerProductSpace Real V
        inst✝³ : MetricSpace P
        inst✝² : NormedAddTorsor V P
        s : AffineSubspace Real P
        inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
        inst✝ : HasOrthogonalProjection s.direction
        p : P
        v : V := VSub.vsub p ↑(Classical.arbitrary (Subtype fun x => Membership.mem s  …
        ⊢ Eq ((fun p => HVAdd.hVAdd (VSub.vsub (↑((EuclideanGeometry.orthogonalProject …
      -/
      let a : V := _root_.orthogonalProjection s.direction v
      /-
        V : Type u_1
        P : Type u_2
        inst✝⁵ : NormedAddCommGroup V
        inst✝⁴ : InnerProductSpace Real V
        inst✝³ : MetricSpace P
        inst✝² : NormedAddTorsor V P
        s : AffineSubspace Real P
        inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
        inst✝ : HasOrthogonalProjection s.direction
        p : P
        v : V := VSub.vsub p ↑(Classical.arbitrary (Subtype fun x => Membership.mem s  …
        a : V := ↑((_root_.orthogonalProjection s.direction) v)
        ⊢ Eq ((fun p => HVAdd.hVAdd (VSub.vsub (↑((EuclideanGeometry.orthogonalProject …
      -/
      let b : P := ↑(Classical.arbitrary s)
      have key : ((a +ᵥ b) -ᵥ (v +ᵥ b)) +ᵥ (a +ᵥ b) = (a + a - v) +ᵥ (b -ᵥ b) +ᵥ b := by
        rw [← add_vadd, vsub_vadd_eq_vsub_sub, vsub_vadd, vadd_vsub]
        congr 1
        abel
      /-
        V : Type u_1
        P : Type u_2
        inst✝⁵ : NormedAddCommGroup V
        inst✝⁴ : InnerProductSpace Real V
        inst✝³ : MetricSpace P
        inst✝² : NormedAddTorsor V P
        s : AffineSubspace Real P
        inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
        inst✝ : HasOrthogonalProjection s.direction
        p : P
        v : V := VSub.vsub p ↑(Classical.arbitrary (Subtype fun x => Membership.mem s  …
        a : V := ↑((_root_.orthogonalProjection s.direction) v)
        b : P := ↑(Classical.arbitrary (Subtype fun x => Membership.mem s x))
        key : Eq (HVAdd.hVAdd (VSub.vsub (HVAdd.hVAdd a b) (HVAdd.hVAdd v b)) (HVAdd.h …
        ⊢ Eq ((fun p => HVAdd.hVAdd (VSub.vsub (↑((EuclideanGeometry.orthogonalProject …
      -/
      dsimp only
      rwa [reflection_apply, (vsub_vadd p b).symm, AffineMap.map_vadd, orthogonalProjection_linear,
        vadd_vsub, orthogonalProjection_mem_subspace_eq_self, two_smul])


/-- The result of reflecting. -/
theorem reflection_apply (s : AffineSubspace ℝ P) [Nonempty s] [HasOrthogonalProjection s.direction]
    (p : P) :
    reflection s p = (↑(orthogonalProjection s p) -ᵥ p) +ᵥ (orthogonalProjection s p : P) :=
  rfl


theorem eq_reflection_of_eq_subspace {s s' : AffineSubspace ℝ P} [Nonempty s] [Nonempty s']
    [HasOrthogonalProjection s.direction] [HasOrthogonalProjection s'.direction] (h : s = s')
    (p : P) : (reflection s p : P) = (reflection s' p : P) := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁷ : NormedAddCommGroup V
    inst✝⁶ : InnerProductSpace Real V
    inst✝⁵ : MetricSpace P
    inst✝⁴ : NormedAddTorsor V P
    s s' : AffineSubspace Real P
    inst✝³ : Nonempty (Subtype fun x => Membership.mem s x)
    inst✝² : Nonempty (Subtype fun x => Membership.mem s' x)
    inst✝¹ : HasOrthogonalProjection s.direction
    inst✝ : HasOrthogonalProjection s'.direction
    h : Eq s s'
    p : P
    ⊢ Eq ((EuclideanGeometry.reflection s) p) ((EuclideanGeometry.reflection s') p)
  -/
  subst h
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁷ : NormedAddCommGroup V
    inst✝⁶ : InnerProductSpace Real V
    inst✝⁵ : MetricSpace P
    inst✝⁴ : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝³ : Nonempty (Subtype fun x => Membership.mem s x)
    inst✝² : HasOrthogonalProjection s.direction
    p : P
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
    inst✝ : HasOrthogonalProjection s.direction
    ⊢ Eq ((EuclideanGeometry.reflection s) p) ((EuclideanGeometry.reflection s) p)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Reflecting twice in the same subspace. -/
@[simp]
theorem reflection_reflection (s : AffineSubspace ℝ P) [Nonempty s]
    [HasOrthogonalProjection s.direction] (p : P) : reflection s (reflection s p) = p := by
  have : ∀ a : s, ∀ b : V, (_root_.orthogonalProjection s.direction) b = 0 →
      reflection s (reflection s (b +ᵥ (a : P))) = b +ᵥ (a : P) := by
    intro _ _ h
    simp [reflection, h]
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : InnerProductSpace Real V
    inst✝³ : MetricSpace P
    inst✝² : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
    inst✝ : HasOrthogonalProjection s.direction
    p : P
    this : ∀ (a : Subtype fun x => Membership.mem s x) (b : V), Eq ((_root_.orthog …
    ⊢ Eq ((EuclideanGeometry.reflection s) ((EuclideanGeometry.reflection s) p)) p
  -/
  rw [← vsub_vadd p (orthogonalProjection s p)]
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : InnerProductSpace Real V
    inst✝³ : MetricSpace P
    inst✝² : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
    inst✝ : HasOrthogonalProjection s.direction
    p : P
    this : ∀ (a : Subtype fun x => Membership.mem s x) (b : V), Eq ((_root_.orthog …
    ⊢ Eq ((EuclideanGeometry.reflection s) ((EuclideanGeometry.reflection s) (HVAd …
  -/
  exact this (orthogonalProjection s p) _ (orthogonalProjection_vsub_orthogonalProjection s p)
  /-
    🎉 no goals
  -/


/-- Reflection is its own inverse. -/
@[simp]
theorem reflection_symm (s : AffineSubspace ℝ P) [Nonempty s]
    [HasOrthogonalProjection s.direction] : (reflection s).symm = reflection s := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : InnerProductSpace Real V
    inst✝³ : MetricSpace P
    inst✝² : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
    inst✝ : HasOrthogonalProjection s.direction
    ⊢ Eq (EuclideanGeometry.reflection s).symm (EuclideanGeometry.reflection s)
  -/
  ext
  /-
    case h
    V : Type u_1
    P : Type u_2
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : InnerProductSpace Real V
    inst✝³ : MetricSpace P
    inst✝² : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
    inst✝ : HasOrthogonalProjection s.direction
    x✝ : P
    ⊢ Eq ((EuclideanGeometry.reflection s).symm x✝) ((EuclideanGeometry.reflection …
  -/
  rw [← (reflection s).injective.eq_iff]
  /-
    case h
    V : Type u_1
    P : Type u_2
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : InnerProductSpace Real V
    inst✝³ : MetricSpace P
    inst✝² : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
    inst✝ : HasOrthogonalProjection s.direction
    x✝ : P
    ⊢ Eq ((EuclideanGeometry.reflection s) ((EuclideanGeometry.reflection s).symm  …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Reflection is involutive. -/
theorem reflection_involutive (s : AffineSubspace ℝ P) [Nonempty s]
    [HasOrthogonalProjection s.direction] : Function.Involutive (reflection s) :=
  reflection_reflection s


/-- A point is its own reflection if and only if it is in the subspace. -/
theorem reflection_eq_self_iff {s : AffineSubspace ℝ P} [Nonempty s]
    [HasOrthogonalProjection s.direction] (p : P) : reflection s p = p ↔ p ∈ s := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : InnerProductSpace Real V
    inst✝³ : MetricSpace P
    inst✝² : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
    inst✝ : HasOrthogonalProjection s.direction
    p : P
    ⊢ Iff (Eq ((EuclideanGeometry.reflection s) p) p) (Membership.mem s p)
  -/
  rw [← orthogonalProjection_eq_self_iff, reflection_apply]
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : InnerProductSpace Real V
    inst✝³ : MetricSpace P
    inst✝² : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
    inst✝ : HasOrthogonalProjection s.direction
    p : P
    ⊢ Iff (Eq (HVAdd.hVAdd (VSub.vsub (↑((EuclideanGeometry.orthogonalProjection s …
  -/
  constructor
    /-
      case mp
      V : Type u_1
      P : Type u_2
      inst✝⁵ : NormedAddCommGroup V
      inst✝⁴ : InnerProductSpace Real V
      inst✝³ : MetricSpace P
      inst✝² : NormedAddTorsor V P
      s : AffineSubspace Real P
      inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
      inst✝ : HasOrthogonalProjection s.direction
      p : P
      ⊢ Eq (HVAdd.hVAdd (VSub.vsub (↑((EuclideanGeometry.orthogonalProjection s) p)) …
    -/
  · intro h
    rw [← @vsub_eq_zero_iff_eq V, vadd_vsub_assoc, ← two_smul ℝ (↑(orthogonalProjection s p) -ᵥ p),
      smul_eq_zero] at h
    /-
      case mp
      V : Type u_1
      P : Type u_2
      inst✝⁵ : NormedAddCommGroup V
      inst✝⁴ : InnerProductSpace Real V
      inst✝³ : MetricSpace P
      inst✝² : NormedAddTorsor V P
      s : AffineSubspace Real P
      inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
      inst✝ : HasOrthogonalProjection s.direction
      p : P
      h : Or (Eq 2 0) (Eq (VSub.vsub (↑((EuclideanGeometry.orthogonalProjection s) p …
      ⊢ Eq (↑((EuclideanGeometry.orthogonalProjection s) p)) p
    -/
    norm_num at h
    /-
      case mp
      V : Type u_1
      P : Type u_2
      inst✝⁵ : NormedAddCommGroup V
      inst✝⁴ : InnerProductSpace Real V
      inst✝³ : MetricSpace P
      inst✝² : NormedAddTorsor V P
      s : AffineSubspace Real P
      inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
      inst✝ : HasOrthogonalProjection s.direction
      p : P
      h : Eq (↑((EuclideanGeometry.orthogonalProjection s) p)) p
      ⊢ Eq (↑((EuclideanGeometry.orthogonalProjection s) p)) p
    -/
    exact h
    /-
      🎉 no goals
    -/
    /-
      case mpr
      V : Type u_1
      P : Type u_2
      inst✝⁵ : NormedAddCommGroup V
      inst✝⁴ : InnerProductSpace Real V
      inst✝³ : MetricSpace P
      inst✝² : NormedAddTorsor V P
      s : AffineSubspace Real P
      inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
      inst✝ : HasOrthogonalProjection s.direction
      p : P
      ⊢ Eq (↑((EuclideanGeometry.orthogonalProjection s) p)) p → Eq (HVAdd.hVAdd (VS …
    -/
  · intro h
    /-
      case mpr
      V : Type u_1
      P : Type u_2
      inst✝⁵ : NormedAddCommGroup V
      inst✝⁴ : InnerProductSpace Real V
      inst✝³ : MetricSpace P
      inst✝² : NormedAddTorsor V P
      s : AffineSubspace Real P
      inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
      inst✝ : HasOrthogonalProjection s.direction
      p : P
      h : Eq (↑((EuclideanGeometry.orthogonalProjection s) p)) p
      ⊢ Eq (HVAdd.hVAdd (VSub.vsub (↑((EuclideanGeometry.orthogonalProjection s) p)) …
    -/
    simp [h]
    /-
      🎉 no goals
    -/


/-- Reflecting a point in two subspaces produces the same result if
and only if the point has the same orthogonal projection in each of
those subspaces. -/
theorem reflection_eq_iff_orthogonalProjection_eq (s₁ s₂ : AffineSubspace ℝ P) [Nonempty s₁]
    [Nonempty s₂] [HasOrthogonalProjection s₁.direction] [HasOrthogonalProjection s₂.direction]
    (p : P) :
    reflection s₁ p = reflection s₂ p ↔
      (orthogonalProjection s₁ p : P) = orthogonalProjection s₂ p := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁷ : NormedAddCommGroup V
    inst✝⁶ : InnerProductSpace Real V
    inst✝⁵ : MetricSpace P
    inst✝⁴ : NormedAddTorsor V P
    s₁ s₂ : AffineSubspace Real P
    inst✝³ : Nonempty (Subtype fun x => Membership.mem s₁ x)
    inst✝² : Nonempty (Subtype fun x => Membership.mem s₂ x)
    inst✝¹ : HasOrthogonalProjection s₁.direction
    inst✝ : HasOrthogonalProjection s₂.direction
    p : P
    ⊢ Iff (Eq ((EuclideanGeometry.reflection s₁) p) ((EuclideanGeometry.reflection …
  -/
  rw [reflection_apply, reflection_apply]
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁷ : NormedAddCommGroup V
    inst✝⁶ : InnerProductSpace Real V
    inst✝⁵ : MetricSpace P
    inst✝⁴ : NormedAddTorsor V P
    s₁ s₂ : AffineSubspace Real P
    inst✝³ : Nonempty (Subtype fun x => Membership.mem s₁ x)
    inst✝² : Nonempty (Subtype fun x => Membership.mem s₂ x)
    inst✝¹ : HasOrthogonalProjection s₁.direction
    inst✝ : HasOrthogonalProjection s₂.direction
    p : P
    ⊢ Iff (Eq (HVAdd.hVAdd (VSub.vsub (↑((EuclideanGeometry.orthogonalProjection s …
  -/
  constructor
    /-
      case mp
      V : Type u_1
      P : Type u_2
      inst✝⁷ : NormedAddCommGroup V
      inst✝⁶ : InnerProductSpace Real V
      inst✝⁵ : MetricSpace P
      inst✝⁴ : NormedAddTorsor V P
      s₁ s₂ : AffineSubspace Real P
      inst✝³ : Nonempty (Subtype fun x => Membership.mem s₁ x)
      inst✝² : Nonempty (Subtype fun x => Membership.mem s₂ x)
      inst✝¹ : HasOrthogonalProjection s₁.direction
      inst✝ : HasOrthogonalProjection s₂.direction
      p : P
      ⊢ Eq (HVAdd.hVAdd (VSub.vsub (↑((EuclideanGeometry.orthogonalProjection s₁) p) …
    -/
  · intro h
    rw [← @vsub_eq_zero_iff_eq V, vsub_vadd_eq_vsub_sub, vadd_vsub_assoc, add_comm, add_sub_assoc,
      vsub_sub_vsub_cancel_right, ←
      two_smul ℝ ((orthogonalProjection s₁ p : P) -ᵥ orthogonalProjection s₂ p), smul_eq_zero] at h
    /-
      case mp
      V : Type u_1
      P : Type u_2
      inst✝⁷ : NormedAddCommGroup V
      inst✝⁶ : InnerProductSpace Real V
      inst✝⁵ : MetricSpace P
      inst✝⁴ : NormedAddTorsor V P
      s₁ s₂ : AffineSubspace Real P
      inst✝³ : Nonempty (Subtype fun x => Membership.mem s₁ x)
      inst✝² : Nonempty (Subtype fun x => Membership.mem s₂ x)
      inst✝¹ : HasOrthogonalProjection s₁.direction
      inst✝ : HasOrthogonalProjection s₂.direction
      p : P
      h : Or (Eq 2 0) (Eq (VSub.vsub ↑((EuclideanGeometry.orthogonalProjection s₁) p …
      ⊢ Eq ↑((EuclideanGeometry.orthogonalProjection s₁) p) ↑((EuclideanGeometry.ort …
    -/
    norm_num at h
    /-
      case mp
      V : Type u_1
      P : Type u_2
      inst✝⁷ : NormedAddCommGroup V
      inst✝⁶ : InnerProductSpace Real V
      inst✝⁵ : MetricSpace P
      inst✝⁴ : NormedAddTorsor V P
      s₁ s₂ : AffineSubspace Real P
      inst✝³ : Nonempty (Subtype fun x => Membership.mem s₁ x)
      inst✝² : Nonempty (Subtype fun x => Membership.mem s₂ x)
      inst✝¹ : HasOrthogonalProjection s₁.direction
      inst✝ : HasOrthogonalProjection s₂.direction
      p : P
      h : Eq ↑((EuclideanGeometry.orthogonalProjection s₁) p) ↑((EuclideanGeometry.o …
      ⊢ Eq ↑((EuclideanGeometry.orthogonalProjection s₁) p) ↑((EuclideanGeometry.ort …
    -/
    exact h
    /-
      🎉 no goals
    -/
    /-
      case mpr
      V : Type u_1
      P : Type u_2
      inst✝⁷ : NormedAddCommGroup V
      inst✝⁶ : InnerProductSpace Real V
      inst✝⁵ : MetricSpace P
      inst✝⁴ : NormedAddTorsor V P
      s₁ s₂ : AffineSubspace Real P
      inst✝³ : Nonempty (Subtype fun x => Membership.mem s₁ x)
      inst✝² : Nonempty (Subtype fun x => Membership.mem s₂ x)
      inst✝¹ : HasOrthogonalProjection s₁.direction
      inst✝ : HasOrthogonalProjection s₂.direction
      p : P
      ⊢ Eq ↑((EuclideanGeometry.orthogonalProjection s₁) p) ↑((EuclideanGeometry.ort …
    -/
  · intro h
    /-
      case mpr
      V : Type u_1
      P : Type u_2
      inst✝⁷ : NormedAddCommGroup V
      inst✝⁶ : InnerProductSpace Real V
      inst✝⁵ : MetricSpace P
      inst✝⁴ : NormedAddTorsor V P
      s₁ s₂ : AffineSubspace Real P
      inst✝³ : Nonempty (Subtype fun x => Membership.mem s₁ x)
      inst✝² : Nonempty (Subtype fun x => Membership.mem s₂ x)
      inst✝¹ : HasOrthogonalProjection s₁.direction
      inst✝ : HasOrthogonalProjection s₂.direction
      p : P
      h : Eq ↑((EuclideanGeometry.orthogonalProjection s₁) p) ↑((EuclideanGeometry.o …
      ⊢ Eq (HVAdd.hVAdd (VSub.vsub (↑((EuclideanGeometry.orthogonalProjection s₁) p) …
    -/
    rw [h]
    /-
      🎉 no goals
    -/


/-- The distance between `p₁` and the reflection of `p₂` equals that
between the reflection of `p₁` and `p₂`. -/
theorem dist_reflection (s : AffineSubspace ℝ P) [Nonempty s] [HasOrthogonalProjection s.direction]
    (p₁ p₂ : P) : dist p₁ (reflection s p₂) = dist (reflection s p₁) p₂ := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : InnerProductSpace Real V
    inst✝³ : MetricSpace P
    inst✝² : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
    inst✝ : HasOrthogonalProjection s.direction
    p₁ p₂ : P
    ⊢ Eq (Dist.dist p₁ ((EuclideanGeometry.reflection s) p₂)) (Dist.dist ((Euclide …
  -/
  conv_lhs => rw [← reflection_reflection s p₁]
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : InnerProductSpace Real V
    inst✝³ : MetricSpace P
    inst✝² : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
    inst✝ : HasOrthogonalProjection s.direction
    p₁ p₂ : P
    ⊢ Eq (Dist.dist ((EuclideanGeometry.reflection s) ((EuclideanGeometry.reflecti …
  -/
  exact (reflection s).dist_map _ _
  /-
    🎉 no goals
  -/


/-- A point in the subspace is equidistant from another point and its
reflection. -/
theorem dist_reflection_eq_of_mem (s : AffineSubspace ℝ P) [Nonempty s]
    [HasOrthogonalProjection s.direction] {p₁ : P} (hp₁ : p₁ ∈ s) (p₂ : P) :
    dist p₁ (reflection s p₂) = dist p₁ p₂ := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : InnerProductSpace Real V
    inst✝³ : MetricSpace P
    inst✝² : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
    inst✝ : HasOrthogonalProjection s.direction
    p₁ : P
    hp₁ : Membership.mem s p₁
    p₂ : P
    ⊢ Eq (Dist.dist p₁ ((EuclideanGeometry.reflection s) p₂)) (Dist.dist p₁ p₂)
  -/
  rw [← reflection_eq_self_iff p₁] at hp₁
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : InnerProductSpace Real V
    inst✝³ : MetricSpace P
    inst✝² : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
    inst✝ : HasOrthogonalProjection s.direction
    p₁ : P
    hp₁ : Eq ((EuclideanGeometry.reflection s) p₁) p₁
    p₂ : P
    ⊢ Eq (Dist.dist p₁ ((EuclideanGeometry.reflection s) p₂)) (Dist.dist p₁ p₂)
  -/
  convert (reflection s).dist_map p₁ p₂
  /-
    case h.e'_2.h.e'_3
    V : Type u_1
    P : Type u_2
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : InnerProductSpace Real V
    inst✝³ : MetricSpace P
    inst✝² : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
    inst✝ : HasOrthogonalProjection s.direction
    p₁ : P
    hp₁ : Eq ((EuclideanGeometry.reflection s) p₁) p₁
    p₂ : P
    ⊢ Eq p₁ ((EuclideanGeometry.reflection s) p₁)
  -/
  rw [hp₁]
  /-
    🎉 no goals
  -/


/-- The reflection of a point in a subspace is contained in any larger
subspace containing both the point and the subspace reflected in. -/
theorem reflection_mem_of_le_of_mem {s₁ s₂ : AffineSubspace ℝ P} [Nonempty s₁]
    [HasOrthogonalProjection s₁.direction] (hle : s₁ ≤ s₂) {p : P} (hp : p ∈ s₂) :
    reflection s₁ p ∈ s₂ := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : InnerProductSpace Real V
    inst✝³ : MetricSpace P
    inst✝² : NormedAddTorsor V P
    s₁ s₂ : AffineSubspace Real P
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem s₁ x)
    inst✝ : HasOrthogonalProjection s₁.direction
    hle : LE.le s₁ s₂
    p : P
    hp : Membership.mem s₂ p
    ⊢ Membership.mem s₂ ((EuclideanGeometry.reflection s₁) p)
  -/
  rw [reflection_apply]
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : InnerProductSpace Real V
    inst✝³ : MetricSpace P
    inst✝² : NormedAddTorsor V P
    s₁ s₂ : AffineSubspace Real P
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem s₁ x)
    inst✝ : HasOrthogonalProjection s₁.direction
    hle : LE.le s₁ s₂
    p : P
    hp : Membership.mem s₂ p
    ⊢ Membership.mem s₂ (HVAdd.hVAdd (VSub.vsub (↑((EuclideanGeometry.orthogonalPr …
  -/
  have ho : ↑(orthogonalProjection s₁ p) ∈ s₂ := hle (orthogonalProjection_mem p)
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : InnerProductSpace Real V
    inst✝³ : MetricSpace P
    inst✝² : NormedAddTorsor V P
    s₁ s₂ : AffineSubspace Real P
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem s₁ x)
    inst✝ : HasOrthogonalProjection s₁.direction
    hle : LE.le s₁ s₂
    p : P
    hp : Membership.mem s₂ p
    ho : Membership.mem s₂ ↑((EuclideanGeometry.orthogonalProjection s₁) p)
    ⊢ Membership.mem s₂ (HVAdd.hVAdd (VSub.vsub (↑((EuclideanGeometry.orthogonalPr …
  -/
  exact vadd_mem_of_mem_direction (vsub_mem_direction ho hp) ho
  /-
    🎉 no goals
  -/


/-- Reflecting an orthogonal vector plus a point in the subspace
produces the negation of that vector plus the point. -/
theorem reflection_orthogonal_vadd {s : AffineSubspace ℝ P} [Nonempty s]
    [HasOrthogonalProjection s.direction] {p : P} (hp : p ∈ s) {v : V} (hv : v ∈ s.directionᗮ) :
    reflection s (v +ᵥ p) = -v +ᵥ p := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : InnerProductSpace Real V
    inst✝³ : MetricSpace P
    inst✝² : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
    inst✝ : HasOrthogonalProjection s.direction
    p : P
    hp : Membership.mem s p
    v : V
    hv : Membership.mem s.direction.orthogonal v
    ⊢ Eq ((EuclideanGeometry.reflection s) (HVAdd.hVAdd v p)) (HVAdd.hVAdd (Neg.ne …
  -/
  rw [reflection_apply, orthogonalProjection_vadd_eq_self hp hv, vsub_vadd_eq_vsub_sub]
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁵ : NormedAddCommGroup V
    inst✝⁴ : InnerProductSpace Real V
    inst✝³ : MetricSpace P
    inst✝² : NormedAddTorsor V P
    s : AffineSubspace Real P
    inst✝¹ : Nonempty (Subtype fun x => Membership.mem s x)
    inst✝ : HasOrthogonalProjection s.direction
    p : P
    hp : Membership.mem s p
    v : V
    hv : Membership.mem s.direction.orthogonal v
    ⊢ Eq (HVAdd.hVAdd (HSub.hSub (VSub.vsub (↑⟨p, hp⟩) p) v) ↑⟨p, hp⟩) (HVAdd.hVAd …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Reflecting a vector plus a point in the subspace produces the
negation of that vector plus the point if the vector is a multiple of
the result of subtracting a point's orthogonal projection from that
point. -/
theorem reflection_vadd_smul_vsub_orthogonalProjection {s : AffineSubspace ℝ P} [Nonempty s]
    [HasOrthogonalProjection s.direction] {p₁ : P} (p₂ : P) (r : ℝ) (hp₁ : p₁ ∈ s) :
    reflection s (r • (p₂ -ᵥ orthogonalProjection s p₂) +ᵥ p₁) =
      -(r • (p₂ -ᵥ orthogonalProjection s p₂)) +ᵥ p₁ :=
  reflection_orthogonal_vadd hp₁
    (Submodule.smul_mem _ _ (vsub_orthogonalProjection_mem_direction_orthogonal s _))


