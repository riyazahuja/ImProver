/-- Characterization of equicontinuity for families of functions taking values in a (pseudo) metric
space. -/
theorem equicontinuousAt_iff_right {ι : Type*} [TopologicalSpace β] {F : ι → β → α} {x₀ : β} :
    EquicontinuousAt F x₀ ↔ ∀ ε > 0, ∀ᶠ x in 𝓝 x₀, ∀ i, dist (F i x₀) (F i x) < ε :=
  uniformity_basis_dist.equicontinuousAt_iff_right


/-- Characterization of equicontinuity for families of functions between (pseudo) metric spaces. -/
theorem equicontinuousAt_iff {ι : Type*} [PseudoMetricSpace β] {F : ι → β → α} {x₀ : β} :
    EquicontinuousAt F x₀ ↔ ∀ ε > 0, ∃ δ > 0, ∀ x, dist x x₀ < δ → ∀ i, dist (F i x₀) (F i x) < ε :=
  nhds_basis_ball.equicontinuousAt_iff uniformity_basis_dist


/-- Reformulation of `equicontinuousAt_iff_pair` for families of functions taking values in a
(pseudo) metric space. -/
protected theorem equicontinuousAt_iff_pair {ι : Type*} [TopologicalSpace β] {F : ι → β → α}
    {x₀ : β} :
    EquicontinuousAt F x₀ ↔
      ∀ ε > 0, ∃ U ∈ 𝓝 x₀, ∀ x ∈ U, ∀ x' ∈ U, ∀ i, dist (F i x) (F i x') < ε := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PseudoMetricSpace α
    ι : Type u_4
    inst✝ : TopologicalSpace β
    F : ι → β → α
    x₀ : β
    ⊢ Iff (EquicontinuousAt F x₀) (∀ (ε : Real), GT.gt ε 0 → Exists fun U => And ( …
  -/
  rw [equicontinuousAt_iff_pair]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PseudoMetricSpace α
    ι : Type u_4
    inst✝ : TopologicalSpace β
    F : ι → β → α
    x₀ : β
    ⊢ Iff (∀ (U : Set (Prod α α)), Membership.mem (uniformity α) U → Exists fun V  …
  -/
  constructor <;> intro H
    /-
      case mp
      α : Type u_1
      β : Type u_2
      inst✝¹ : PseudoMetricSpace α
      ι : Type u_4
      inst✝ : TopologicalSpace β
      F : ι → β → α
      x₀ : β
      H : ∀ (U : Set (Prod α α)), Membership.mem (uniformity α) U → Exists fun V =>  …
      ⊢ ∀ (ε : Real), GT.gt ε 0 → Exists fun U => And (Membership.mem (nhds x₀) U) ( …
    -/
  · intro ε hε
    /-
      case mp
      α : Type u_1
      β : Type u_2
      inst✝¹ : PseudoMetricSpace α
      ι : Type u_4
      inst✝ : TopologicalSpace β
      F : ι → β → α
      x₀ : β
      H : ∀ (U : Set (Prod α α)), Membership.mem (uniformity α) U → Exists fun V =>  …
      ε : Real
      hε : GT.gt ε 0
      ⊢ Exists fun U => And (Membership.mem (nhds x₀) U) (∀ (x : β), Membership.mem  …
    -/
    exact H _ (dist_mem_uniformity hε)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      β : Type u_2
      inst✝¹ : PseudoMetricSpace α
      ι : Type u_4
      inst✝ : TopologicalSpace β
      F : ι → β → α
      x₀ : β
      H : ∀ (ε : Real), GT.gt ε 0 → Exists fun U => And (Membership.mem (nhds x₀) U) …
      ⊢ ∀ (U : Set (Prod α α)), Membership.mem (uniformity α) U → Exists fun V => An …
    -/
  · intro U hU
    /-
      case mpr
      α : Type u_1
      β : Type u_2
      inst✝¹ : PseudoMetricSpace α
      ι : Type u_4
      inst✝ : TopologicalSpace β
      F : ι → β → α
      x₀ : β
      H : ∀ (ε : Real), GT.gt ε 0 → Exists fun U => And (Membership.mem (nhds x₀) U) …
      U : Set (Prod α α)
      hU : Membership.mem (uniformity α) U
      ⊢ Exists fun V => And (Membership.mem (nhds x₀) V) (∀ (x : β), Membership.mem  …
    -/
    rcases mem_uniformity_dist.mp hU with ⟨ε, hε, hεU⟩
    /-
      case mpr.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝¹ : PseudoMetricSpace α
      ι : Type u_4
      inst✝ : TopologicalSpace β
      F : ι → β → α
      x₀ : β
      H : ∀ (ε : Real), GT.gt ε 0 → Exists fun U => And (Membership.mem (nhds x₀) U) …
      U : Set (Prod α α)
      hU : Membership.mem (uniformity α) U
      ε : Real
      hε : GT.gt ε 0
      hεU : ∀ ⦃a b : α⦄, LT.lt (Dist.dist a b) ε → Membership.mem U { fst := a, snd  …
      ⊢ Exists fun V => And (Membership.mem (nhds x₀) V) (∀ (x : β), Membership.mem  …
    -/
    refine Exists.imp (fun V => And.imp_right fun h => ?_) (H _ hε)
    /-
      case mpr.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝¹ : PseudoMetricSpace α
      ι : Type u_4
      inst✝ : TopologicalSpace β
      F : ι → β → α
      x₀ : β
      H : ∀ (ε : Real), GT.gt ε 0 → Exists fun U => And (Membership.mem (nhds x₀) U) …
      U : Set (Prod α α)
      hU : Membership.mem (uniformity α) U
      ε : Real
      hε : GT.gt ε 0
      hεU : ∀ ⦃a b : α⦄, LT.lt (Dist.dist a b) ε → Membership.mem U { fst := a, snd  …
      V : Set β
      h : ∀ (x : β), Membership.mem V x → ∀ (x' : β), Membership.mem V x' → ∀ (i : ι …
      ⊢ ∀ (x : β), Membership.mem V x → ∀ (y : β), Membership.mem V y → ∀ (i : ι), M …
    -/
    exact fun x hx x' hx' i => hεU (h _ hx _ hx' i)
    /-
      🎉 no goals
    -/


/-- Characterization of uniform equicontinuity for families of functions taking values in a
(pseudo) metric space. -/
theorem uniformEquicontinuous_iff_right {ι : Type*} [UniformSpace β] {F : ι → β → α} :
    UniformEquicontinuous F ↔ ∀ ε > 0, ∀ᶠ xy : β × β in 𝓤 β, ∀ i, dist (F i xy.1) (F i xy.2) < ε :=
  uniformity_basis_dist.uniformEquicontinuous_iff_right


/-- Characterization of uniform equicontinuity for families of functions between
(pseudo) metric spaces. -/
theorem uniformEquicontinuous_iff {ι : Type*} [PseudoMetricSpace β] {F : ι → β → α} :
    UniformEquicontinuous F ↔
      ∀ ε > 0, ∃ δ > 0, ∀ x y, dist x y < δ → ∀ i, dist (F i x) (F i y) < ε :=
  uniformity_basis_dist.uniformEquicontinuous_iff uniformity_basis_dist


/-- For a family of functions to a (pseudo) metric spaces, a convenient way to prove
equicontinuity at a point is to show that all of the functions share a common *local* continuity
modulus. -/
theorem equicontinuousAt_of_continuity_modulus {ι : Type*} [TopologicalSpace β] {x₀ : β}
    (b : β → ℝ) (b_lim : Tendsto b (𝓝 x₀) (𝓝 0)) (F : ι → β → α)
    (H : ∀ᶠ x in 𝓝 x₀, ∀ i, dist (F i x₀) (F i x) ≤ b x) : EquicontinuousAt F x₀ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PseudoMetricSpace α
    ι : Type u_4
    inst✝ : TopologicalSpace β
    x₀ : β
    b : β → Real
    b_lim : Filter.Tendsto b (nhds x₀) (nhds 0)
    F : ι → β → α
    H : Filter.Eventually (fun x => ∀ (i : ι), LE.le (Dist.dist (F i x₀) (F i x))  …
    ⊢ EquicontinuousAt F x₀
  -/
  rw [Metric.equicontinuousAt_iff_right]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PseudoMetricSpace α
    ι : Type u_4
    inst✝ : TopologicalSpace β
    x₀ : β
    b : β → Real
    b_lim : Filter.Tendsto b (nhds x₀) (nhds 0)
    F : ι → β → α
    H : Filter.Eventually (fun x => ∀ (i : ι), LE.le (Dist.dist (F i x₀) (F i x))  …
    ⊢ ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x => ∀ (i : ι), LT.lt (Dist …
  -/
  intro ε ε0
  -- Porting note: Lean 3 didn't need `Filter.mem_map.mp` here
  filter_upwards [Filter.mem_map.mp <| b_lim (Iio_mem_nhds ε0), H] using
    fun x hx₁ hx₂ i => (hx₂ i).trans_lt hx₁


/-- For a family of functions between (pseudo) metric spaces, a convenient way to prove
uniform equicontinuity is to show that all of the functions share a common *global* continuity
modulus. -/
theorem uniformEquicontinuous_of_continuity_modulus {ι : Type*} [PseudoMetricSpace β] (b : ℝ → ℝ)
    (b_lim : Tendsto b (𝓝 0) (𝓝 0)) (F : ι → β → α)
    (H : ∀ (x y : β) (i), dist (F i x) (F i y) ≤ b (dist x y)) : UniformEquicontinuous F := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PseudoMetricSpace α
    ι : Type u_4
    inst✝ : PseudoMetricSpace β
    b : Real → Real
    b_lim : Filter.Tendsto b (nhds 0) (nhds 0)
    F : ι → β → α
    H : ∀ (x y : β) (i : ι), LE.le (Dist.dist (F i x) (F i y)) (b (Dist.dist x y))
    ⊢ UniformEquicontinuous F
  -/
  rw [Metric.uniformEquicontinuous_iff]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PseudoMetricSpace α
    ι : Type u_4
    inst✝ : PseudoMetricSpace β
    b : Real → Real
    b_lim : Filter.Tendsto b (nhds 0) (nhds 0)
    F : ι → β → α
    H : ∀ (x y : β) (i : ι), LE.le (Dist.dist (F i x) (F i y)) (b (Dist.dist x y))
    ⊢ ∀ (ε : Real), GT.gt ε 0 → Exists fun δ => And (GT.gt δ 0) (∀ (x y : β), LT.l …
  -/
  intro ε ε0
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PseudoMetricSpace α
    ι : Type u_4
    inst✝ : PseudoMetricSpace β
    b : Real → Real
    b_lim : Filter.Tendsto b (nhds 0) (nhds 0)
    F : ι → β → α
    H : ∀ (x y : β) (i : ι), LE.le (Dist.dist (F i x) (F i y)) (b (Dist.dist x y))
    ε : Real
    ε0 : GT.gt ε 0
    ⊢ Exists fun δ => And (GT.gt δ 0) (∀ (x y : β), LT.lt (Dist.dist x y) δ → ∀ (i …
  -/
  rcases tendsto_nhds_nhds.1 b_lim ε ε0 with ⟨δ, δ0, hδ⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    inst✝¹ : PseudoMetricSpace α
    ι : Type u_4
    inst✝ : PseudoMetricSpace β
    b : Real → Real
    b_lim : Filter.Tendsto b (nhds 0) (nhds 0)
    F : ι → β → α
    H : ∀ (x y : β) (i : ι), LE.le (Dist.dist (F i x) (F i y)) (b (Dist.dist x y))
    ε : Real
    ε0 : GT.gt ε 0
    δ : Real
    δ0 : GT.gt δ 0
    hδ : ∀ ⦃x : Real⦄, LT.lt (Dist.dist x 0) δ → LT.lt (Dist.dist (b x) 0) ε
    ⊢ Exists fun δ => And (GT.gt δ 0) (∀ (x y : β), LT.lt (Dist.dist x y) δ → ∀ (i …
  -/
  refine ⟨δ, δ0, fun x y hxy i => ?_⟩
  calc
    dist (F i x) (F i y) ≤ b (dist x y) := H x y i
    _ ≤ |b (dist x y)| := le_abs_self _
    _ = dist (b (dist x y)) 0 := by simp [Real.dist_eq]
    _ < ε := hδ (by simpa only [Real.dist_eq, tsub_zero, abs_dist] using hxy)


/-- For a family of functions between (pseudo) metric spaces, a convenient way to prove
equicontinuity is to show that all of the functions share a common *global* continuity modulus. -/
theorem equicontinuous_of_continuity_modulus {ι : Type*} [PseudoMetricSpace β] (b : ℝ → ℝ)
    (b_lim : Tendsto b (𝓝 0) (𝓝 0)) (F : ι → β → α)
    (H : ∀ (x y : β) (i), dist (F i x) (F i y) ≤ b (dist x y)) : Equicontinuous F :=
  (uniformEquicontinuous_of_continuity_modulus b b_lim F H).equicontinuous


