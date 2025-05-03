/-- The Birkhoff averages of a function `g` over the orbit of a fixed point `x` of `f`
tend to `g x` as `N → ∞`. In fact, they are equal to `g x` for all `N ≠ 0`,
see `Function.IsFixedPt.birkhoffAverage_eq`.

TODO: add a version for a periodic orbit. -/
theorem Function.IsFixedPt.tendsto_birkhoffAverage
    (R : Type*) [DivisionSemiring R] [CharZero R]
    [AddCommMonoid E] [TopologicalSpace E] [Module R E]
    {f : α → α} {x : α} (h : f.IsFixedPt x) (g : α → E) :
    Tendsto (birkhoffAverage R f g · x) atTop (𝓝 (g x)) :=
  tendsto_const_nhds.congr' <| (eventually_ne_atTop 0).mono fun _n hn ↦
    (h.birkhoffAverage_eq R g hn).symm


theorem dist_birkhoffSum_apply_birkhoffSum (f : α → α) (g : α → E) (n : ℕ) (x : α) :
    dist (birkhoffSum f g n (f x)) (birkhoffSum f g n x) = dist (g (f^[n] x)) (g x) := by
  /-
    α : Type u_1
    E : Type u_2
    inst✝ : NormedAddCommGroup E
    f : α → α
    g : α → E
    n : Nat
    x : α
    ⊢ Eq (Dist.dist (birkhoffSum f g n (f x)) (birkhoffSum f g n x)) (Dist.dist (g …
  -/
  simp only [dist_eq_norm, birkhoffSum_apply_sub_birkhoffSum]
  /-
    🎉 no goals
  -/


theorem dist_birkhoffSum_birkhoffSum_le (f : α → α) (g : α → E) (n : ℕ) (x y : α) :
    dist (birkhoffSum f g n x) (birkhoffSum f g n y) ≤
      ∑ k ∈ Finset.range n, dist (g (f^[k] x)) (g (f^[k] y)) :=
  dist_sum_sum_le _ _ _


theorem dist_birkhoffAverage_birkhoffAverage (f : α → α) (g : α → E) (n : ℕ) (x y : α) :
    dist (birkhoffAverage 𝕜 f g n x) (birkhoffAverage 𝕜 f g n y) =
      dist (birkhoffSum f g n x) (birkhoffSum f g n y) / n := by
  /-
    α : Type u_1
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    𝕜 : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : Module 𝕜 E
    inst✝ : BoundedSMul 𝕜 E
    f : α → α
    g : α → E
    n : Nat
    x y : α
    ⊢ Eq (Dist.dist (birkhoffAverage 𝕜 f g n x) (birkhoffAverage 𝕜 f g n y)) (HDiv …
  -/
  simp [birkhoffAverage, dist_smul₀, div_eq_inv_mul]
  /-
    🎉 no goals
  -/


theorem dist_birkhoffAverage_birkhoffAverage_le (f : α → α) (g : α → E) (n : ℕ) (x y : α) :
    dist (birkhoffAverage 𝕜 f g n x) (birkhoffAverage 𝕜 f g n y) ≤
      (∑ k ∈ Finset.range n, dist (g (f^[k] x)) (g (f^[k] y))) / n :=
  (dist_birkhoffAverage_birkhoffAverage _ _ _ _ _ _).trans_le <| by
    /-
      α : Type u_1
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      𝕜 : Type u_3
      inst✝² : RCLike 𝕜
      inst✝¹ : Module 𝕜 E
      inst✝ : BoundedSMul 𝕜 E
      f : α → α
      g : α → E
      n : Nat
      x y : α
      ⊢ LE.le (HDiv.hDiv (Dist.dist (birkhoffSum f g n x) (birkhoffSum f g n y)) ↑n) …
    -/
    gcongr; apply dist_birkhoffSum_birkhoffSum_le
            /-
              🎉 no goals
            -/


theorem dist_birkhoffAverage_apply_birkhoffAverage (f : α → α) (g : α → E) (n : ℕ) (x : α) :
    dist (birkhoffAverage 𝕜 f g n (f x)) (birkhoffAverage 𝕜 f g n x) =
      dist (g (f^[n] x)) (g x) / n := by
  /-
    α : Type u_1
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    𝕜 : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : Module 𝕜 E
    inst✝ : BoundedSMul 𝕜 E
    f : α → α
    g : α → E
    n : Nat
    x : α
    ⊢ Eq (Dist.dist (birkhoffAverage 𝕜 f g n (f x)) (birkhoffAverage 𝕜 f g n x)) ( …
  -/
  simp [dist_birkhoffAverage_birkhoffAverage, dist_birkhoffSum_apply_birkhoffSum]
  /-
    🎉 no goals
  -/


/-- If a function `g` is bounded along the positive orbit of `x` under `f`,
then the difference between Birkhoff averages of `g`
along the orbit of `f x` and along the orbit of `x`
tends to zero.

See also `tendsto_birkhoffAverage_apply_sub_birkhoffAverage'`. -/
theorem tendsto_birkhoffAverage_apply_sub_birkhoffAverage {f : α → α} {g : α → E} {x : α}
    (h : Bornology.IsBounded (range (g <| f^[·] x))) :
    Tendsto (fun n ↦ birkhoffAverage 𝕜 f g n (f x) - birkhoffAverage 𝕜 f g n x) atTop (𝓝 0) := by
  /-
    α : Type u_1
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    𝕜 : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : Module 𝕜 E
    inst✝ : BoundedSMul 𝕜 E
    f : α → α
    g : α → E
    x : α
    h : Bornology.IsBounded (Set.range fun x_1 => g (Nat.iterate f x_1 x))
    ⊢ Filter.Tendsto (fun n => HSub.hSub (birkhoffAverage 𝕜 f g n (f x)) (birkhoff …
  -/
  rcases Metric.isBounded_range_iff.1 h with ⟨C, hC⟩
  have : Tendsto (fun n : ℕ ↦ C / n) atTop (𝓝 0) :=
    tendsto_const_nhds.div_atTop tendsto_natCast_atTop_atTop
  /-
    case intro
    α : Type u_1
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    𝕜 : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : Module 𝕜 E
    inst✝ : BoundedSMul 𝕜 E
    f : α → α
    g : α → E
    x : α
    h : Bornology.IsBounded (Set.range fun x_1 => g (Nat.iterate f x_1 x))
    C : Real
    hC : ∀ (x_1 y : Nat), LE.le (Dist.dist (g (Nat.iterate f x_1 x)) (g (Nat.itera …
    this : Filter.Tendsto (fun n => HDiv.hDiv C ↑n) Filter.atTop (nhds 0)
    ⊢ Filter.Tendsto (fun n => HSub.hSub (birkhoffAverage 𝕜 f g n (f x)) (birkhoff …
  -/
  refine squeeze_zero_norm (fun n ↦ ?_) this
  /-
    case intro
    α : Type u_1
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    𝕜 : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : Module 𝕜 E
    inst✝ : BoundedSMul 𝕜 E
    f : α → α
    g : α → E
    x : α
    h : Bornology.IsBounded (Set.range fun x_1 => g (Nat.iterate f x_1 x))
    C : Real
    hC : ∀ (x_1 y : Nat), LE.le (Dist.dist (g (Nat.iterate f x_1 x)) (g (Nat.itera …
    this : Filter.Tendsto (fun n => HDiv.hDiv C ↑n) Filter.atTop (nhds 0)
    n : Nat
    ⊢ LE.le (Norm.norm (HSub.hSub (birkhoffAverage 𝕜 f g n (f x)) (birkhoffAverage …
  -/
  rw [← dist_eq_norm, dist_birkhoffAverage_apply_birkhoffAverage]
  /-
    case intro
    α : Type u_1
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    𝕜 : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : Module 𝕜 E
    inst✝ : BoundedSMul 𝕜 E
    f : α → α
    g : α → E
    x : α
    h : Bornology.IsBounded (Set.range fun x_1 => g (Nat.iterate f x_1 x))
    C : Real
    hC : ∀ (x_1 y : Nat), LE.le (Dist.dist (g (Nat.iterate f x_1 x)) (g (Nat.itera …
    this : Filter.Tendsto (fun n => HDiv.hDiv C ↑n) Filter.atTop (nhds 0)
    n : Nat
    ⊢ LE.le (HDiv.hDiv (Dist.dist (g (Nat.iterate f n x)) (g x)) ↑n) (HDiv.hDiv C  …
  -/
  gcongr
  /-
    case intro.hab
    α : Type u_1
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    𝕜 : Type u_3
    inst✝² : RCLike 𝕜
    inst✝¹ : Module 𝕜 E
    inst✝ : BoundedSMul 𝕜 E
    f : α → α
    g : α → E
    x : α
    h : Bornology.IsBounded (Set.range fun x_1 => g (Nat.iterate f x_1 x))
    C : Real
    hC : ∀ (x_1 y : Nat), LE.le (Dist.dist (g (Nat.iterate f x_1 x)) (g (Nat.itera …
    this : Filter.Tendsto (fun n => HDiv.hDiv C ↑n) Filter.atTop (nhds 0)
    n : Nat
    ⊢ LE.le (Dist.dist (g (Nat.iterate f n x)) (g x)) C
  -/
  exact hC n 0
  /-
    🎉 no goals
  -/


/-- If a function `g` is bounded,
then the difference between Birkhoff averages of `g`
along the orbit of `f x` and along the orbit of `x`
tends to zero.

See also `tendsto_birkhoffAverage_apply_sub_birkhoffAverage`. -/
theorem tendsto_birkhoffAverage_apply_sub_birkhoffAverage' {g : α → E}
    (h : Bornology.IsBounded (range g)) (f : α → α) (x : α) :
    Tendsto (fun n ↦ birkhoffAverage 𝕜 f g n (f x) - birkhoffAverage 𝕜 f g n x) atTop (𝓝 0) :=
  tendsto_birkhoffAverage_apply_sub_birkhoffAverage _ <| h.subset <| range_comp_subset_range _ _


/-- If `f` is a non-strictly contracting map (i.e., it is Lipschitz with constant `1`)
and `g` is a uniformly continuous, then the Birkhoff averages of `g` along orbits of `f`
is a uniformly equicontinuous family of functions. -/
theorem uniformEquicontinuous_birkhoffAverage (hf : LipschitzWith 1 f) (hg : UniformContinuous g) :
    UniformEquicontinuous (birkhoffAverage 𝕜 f g) := by
  /-
    𝕜 : Type u_1
    X : Type u_2
    E : Type u_3
    inst✝³ : PseudoEMetricSpace X
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : X → X
    g : X → E
    hf : LipschitzWith 1 f
    hg : UniformContinuous g
    ⊢ UniformEquicontinuous (birkhoffAverage 𝕜 f g)
  -/
  refine Metric.uniformity_basis_dist_le.uniformEquicontinuous_iff_right.2 fun ε hε ↦ ?_
  rcases (uniformity_basis_edist_le.uniformContinuous_iff Metric.uniformity_basis_dist_le).1 hg ε hε
    with ⟨δ, hδ₀, hδε⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    X : Type u_2
    E : Type u_3
    inst✝³ : PseudoEMetricSpace X
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : X → X
    g : X → E
    hf : LipschitzWith 1 f
    hg : UniformContinuous g
    ε : Real
    hε : LT.lt 0 ε
    δ : ENNReal
    hδ₀ : LT.lt 0 δ
    hδε : ∀ (x y : X), Membership.mem (setOf fun p => LE.le (EDist.edist p.1 p.2)  …
    ⊢ Filter.Eventually (fun xy => ∀ (i : Nat), Membership.mem (setOf fun p => LE. …
  -/
  refine mem_uniformity_edist.2 ⟨δ, hδ₀, fun {x y} h n ↦ ?_⟩
  calc
    dist (birkhoffAverage 𝕜 f g n x) (birkhoffAverage 𝕜 f g n y)
      ≤ (∑ k ∈ Finset.range n, dist (g (f^[k] x)) (g (f^[k] y))) / n :=
      dist_birkhoffAverage_birkhoffAverage_le ..
    _ ≤ (∑ _k ∈ Finset.range n, ε) / n := by
      gcongr
      refine hδε _ _ ?_
      simpa using (hf.iterate _).edist_le_mul_of_le h.le
    _ = n * ε / n := by simp
    _ ≤ ε := by
      rcases eq_or_ne n 0 with hn | hn <;> field_simp [hn, hε.le, mul_div_cancel_left₀]


/-- If `f : X → X` is a non-strictly contracting map (i.e., it is Lipschitz with constant `1`),
`g : X → E` is a uniformly continuous, and `l : X → E` is a continuous function,
then the set of points `x`
such that the Birkhoff average of `g` along the orbit of `x` tends to `l x`
is a closed set. -/
theorem isClosed_setOf_tendsto_birkhoffAverage
    (hf : LipschitzWith 1 f) (hg : UniformContinuous g) (hl : Continuous l) :
    IsClosed {x | Tendsto (birkhoffAverage 𝕜 f g · x) atTop (𝓝 (l x))} :=
  (uniformEquicontinuous_birkhoffAverage 𝕜 hf hg).equicontinuous.isClosed_setOf_tendsto hl

