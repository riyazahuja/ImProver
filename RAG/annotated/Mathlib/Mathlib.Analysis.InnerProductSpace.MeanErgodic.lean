/-- **Von Neumann Mean Ergodic Theorem**, a version for a normed space.

Let `f : E → E` be a contracting linear self-map of a normed space.
Let `S` be the subspace of fixed points of `f`.
Let `g : E → S` be a continuous linear projection, `g|_S=id`.
If the range of `f - id` is dense in the kernel of `g`,
then for each `x`, the Birkhoff averages
```
birkhoffAverage 𝕜 f id N x = (N : 𝕜)⁻¹ • ∑ n ∈ Finset.range N, f^[n] x
```
converge to `g x` as `N → ∞`.

Usually, this fact is not formulated as a separate lemma.
I chose to do it in order to isolate parts of the proof that do not rely
on the inner product space structure.
-/
theorem LinearMap.tendsto_birkhoffAverage_of_ker_subset_closure [NormedSpace 𝕜 E]
    (f : E →ₗ[𝕜] E) (hf : LipschitzWith 1 f) (g : E →L[𝕜] LinearMap.eqLocus f 1)
    (hg_proj : ∀ x : LinearMap.eqLocus f 1, g x = x)
    (hg_ker : (LinearMap.ker g : Set E) ⊆ closure (LinearMap.range (f - 1))) (x : E) :
    Tendsto (birkhoffAverage 𝕜 f _root_.id · x) atTop (𝓝 (g x)) := by
  /- Any point can be represented as a sum of `y ∈ LinearMap.ker g` and a fixed point `z`. -/
  obtain ⟨y, hy, z, hz, rfl⟩ : ∃ y, g y = 0 ∧ ∃ z, IsFixedPt f z ∧ x = y + z :=
    ⟨x - g x, by simp [hg_proj], g x, (g x).2, by simp⟩
  /- For a fixed point, the theorem is trivial,
  so it suffices to prove it for `y ∈ LinearMap.ker g`. -/
  suffices Tendsto (birkhoffAverage 𝕜 f _root_.id · y) atTop (𝓝 0) by
    have hgz : g z = z := congr_arg Subtype.val (hg_proj ⟨z, hz⟩)
    simpa [hy, hgz, birkhoffAverage, birkhoffSum, Finset.sum_add_distrib, smul_add]
      using this.add (hz.tendsto_birkhoffAverage 𝕜 _root_.id)
  /- By continuity, it suffices to prove the theorem on a dense subset of `LinearMap.ker g`.
  By assumption, `LinearMap.range (f - 1)` is dense in the kernel of `g`,
  so it suffices to prove the theorem for `y = f x - x`. -/
  have : IsClosed {x | Tendsto (birkhoffAverage 𝕜 f _root_.id · x) atTop (𝓝 0)} :=
    isClosed_setOf_tendsto_birkhoffAverage 𝕜 hf uniformContinuous_id continuous_const
  /-
    case intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : LinearMap (RingHom.id 𝕜) E E
    hf : LipschitzWith 1 ⇑f
    g : ContinuousLinearMap (RingHom.id 𝕜) E (Subtype fun x => Membership.mem (Lin …
    hg_proj : ∀ (x : Subtype fun x => Membership.mem (LinearMap.eqLocus f 1) x), E …
    hg_ker : HasSubset.Subset (↑(LinearMap.ker g)) (closure ↑(LinearMap.range (HSu …
    y : E
    hy : Eq (g y) 0
    z : E
    hz : Function.IsFixedPt (⇑f) z
    this : IsClosed (setOf fun x => Filter.Tendsto (fun x_1 => birkhoffAverage 𝕜 ( …
    ⊢ Filter.Tendsto (fun x => birkhoffAverage 𝕜 (⇑f) _root_.id x y) Filter.atTop  …
  -/
  refine closure_minimal (Set.forall_mem_range.2 fun x ↦ ?_) this (hg_ker hy)
  /- Finally, for `y = f x - x` the average is equal to the difference between averages
  along the orbits of `f x` and `x`, and most of the terms cancel. -/
  have : IsBounded (Set.range (_root_.id <| f^[·] x)) :=
    isBounded_iff_forall_norm_le.2 ⟨‖x‖, Set.forall_mem_range.2 fun n ↦ by
      have H : f^[n] 0 = 0 := iterate_map_zero (f : E →+ E) n
      simpa [H] using (hf.iterate n).dist_le_mul x 0⟩
  /-
    case intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    f : LinearMap (RingHom.id 𝕜) E E
    hf : LipschitzWith 1 ⇑f
    g : ContinuousLinearMap (RingHom.id 𝕜) E (Subtype fun x => Membership.mem (Lin …
    hg_proj : ∀ (x : Subtype fun x => Membership.mem (LinearMap.eqLocus f 1) x), E …
    hg_ker : HasSubset.Subset (↑(LinearMap.ker g)) (closure ↑(LinearMap.range (HSu …
    y : E
    hy : Eq (g y) 0
    z : E
    hz : Function.IsFixedPt (⇑f) z
    this✝ : IsClosed (setOf fun x => Filter.Tendsto (fun x_1 => birkhoffAverage 𝕜  …
    x : E
    this : Bornology.IsBounded (Set.range fun x_1 => _root_.id (Nat.iterate (⇑f) x …
    ⊢ Membership.mem (setOf fun x => Filter.Tendsto (fun x_1 => birkhoffAverage 𝕜  …
  -/
  have H : ∀ n x y, f^[n] (x - y) = f^[n] x - f^[n] y := iterate_map_sub (f : E →+ E)
  simpa [birkhoffAverage, birkhoffSum, Finset.sum_sub_distrib, smul_sub, H]
    using tendsto_birkhoffAverage_apply_sub_birkhoffAverage 𝕜 this


local notation "⟪" x ", " y "⟫" => @inner 𝕜 _ _ x y


/-- **Von Neumann Mean Ergodic Theorem** for an operator in a Hilbert space.
For a contracting continuous linear self-map `f : E →L[𝕜] E` of a Hilbert space, `‖f‖ ≤ 1`,
the Birkhoff averages
```
birkhoffAverage 𝕜 f id N x = (N : 𝕜)⁻¹ • ∑ n ∈ Finset.range N, f^[n] x
```
converge to the orthogonal projection of `x` to the subspace of fixed points of `f`. -/
theorem ContinuousLinearMap.tendsto_birkhoffAverage_orthogonalProjection (f : E →L[𝕜] E)
    (hf : ‖f‖ ≤ 1) (x : E) :
    Tendsto (birkhoffAverage 𝕜 f _root_.id · x) atTop
      (𝓝 <| orthogonalProjection (LinearMap.eqLocus f 1) x) := by
  /- Due to the previous theorem, it suffices to verify
  that the range of `f - 1` is dense in the orthogonal complement
  to the submodule of fixed points of `f`. -/
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    f : ContinuousLinearMap (RingHom.id 𝕜) E E
    hf : LE.le (Norm.norm f) 1
    x : E
    ⊢ Filter.Tendsto (fun x_1 => birkhoffAverage 𝕜 (⇑f) _root_.id x_1 x) Filter.at …
  -/
  apply (f : E →ₗ[𝕜] E).tendsto_birkhoffAverage_of_ker_subset_closure (f.lipschitz.weaken hf)
    /-
      case hg_proj
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : RCLike 𝕜
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace 𝕜 E
      inst✝ : CompleteSpace E
      f : ContinuousLinearMap (RingHom.id 𝕜) E E
      hf : LE.le (Norm.norm f) 1
      x : E
      ⊢ ∀ (x : Subtype fun x => Membership.mem (LinearMap.eqLocus (↑f) 1) x), Eq ((o …
    -/
  · exact orthogonalProjection_mem_subspace_eq_self (K := LinearMap.eqLocus f 1)
    /-
      🎉 no goals
    -/
    /-
      case hg_ker
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : RCLike 𝕜
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace 𝕜 E
      inst✝ : CompleteSpace E
      f : ContinuousLinearMap (RingHom.id 𝕜) E E
      hf : LE.le (Norm.norm f) 1
      x : E
      ⊢ HasSubset.Subset (↑(LinearMap.ker (orthogonalProjection (LinearMap.eqLocus f …
    -/
  · clear x
    /- In other words, we need to verify that any vector that is orthogonal to the range of `f - 1`
    is a fixed point of `f`. -/
    rw [ker_orthogonalProjection, ← Submodule.topologicalClosure_coe, SetLike.coe_subset_coe,
      ← Submodule.orthogonal_orthogonal_eq_closure]
    /- To verify this, we verify `‖f x‖ ≤ ‖x‖` (because `‖f‖ ≤ 1`) and `⟪f x, x⟫ = ‖x‖²`. -/
    /-
      case hg_ker
      𝕜 : Type u_1
      E : Type u_2
      inst✝³ : RCLike 𝕜
      inst✝² : NormedAddCommGroup E
      inst✝¹ : InnerProductSpace 𝕜 E
      inst✝ : CompleteSpace E
      f : ContinuousLinearMap (RingHom.id 𝕜) E E
      hf : LE.le (Norm.norm f) 1
      ⊢ LE.le (LinearMap.eqLocus f 1).orthogonal (LinearMap.range (HSub.hSub (↑f) 1) …
    -/
    refine Submodule.orthogonal_le fun x hx ↦ eq_of_norm_le_re_inner_eq_norm_sq (𝕜 := 𝕜) ?_ ?_
      /-
        case hg_ker.refine_1
        𝕜 : Type u_1
        E : Type u_2
        inst✝³ : RCLike 𝕜
        inst✝² : NormedAddCommGroup E
        inst✝¹ : InnerProductSpace 𝕜 E
        inst✝ : CompleteSpace E
        f : ContinuousLinearMap (RingHom.id 𝕜) E E
        hf : LE.le (Norm.norm f) 1
        x : E
        hx : Membership.mem (LinearMap.range (HSub.hSub (↑f) 1)).orthogonal x
        ⊢ LE.le (Norm.norm (f x)) (Norm.norm (1 x))
      -/
    · simpa using f.le_of_opNorm_le hf x
      /-
        🎉 no goals
      -/
    · have : ∀ y, ⟪f y, x⟫ = ⟪y, x⟫ := by
        simpa [Submodule.mem_orthogonal, inner_sub_left, sub_eq_zero] using hx
      /-
        case hg_ker.refine_2
        𝕜 : Type u_1
        E : Type u_2
        inst✝³ : RCLike 𝕜
        inst✝² : NormedAddCommGroup E
        inst✝¹ : InnerProductSpace 𝕜 E
        inst✝ : CompleteSpace E
        f : ContinuousLinearMap (RingHom.id 𝕜) E E
        hf : LE.le (Norm.norm f) 1
        x : E
        hx : Membership.mem (LinearMap.range (HSub.hSub (↑f) 1)).orthogonal x
        this : ∀ (y : E), Eq (Inner.inner (f y) x) (Inner.inner y x)
        ⊢ Eq (RCLike.re (Inner.inner (f x) (1 x))) (HPow.hPow (Norm.norm (1 x)) 2)
      -/
      simp [this, ← norm_sq_eq_inner]
      /-
        🎉 no goals
      -/

