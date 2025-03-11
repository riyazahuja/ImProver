theorem ContinuousSMul.of_nhds_zero [TopologicalRing R] [TopologicalAddGroup M]
    (hmul : Tendsto (fun p : R × M => p.1 • p.2) (𝓝 0 ×ˢ 𝓝 0) (𝓝 0))
    (hmulleft : ∀ m : M, Tendsto (fun a : R => a • m) (𝓝 0) (𝓝 0))
    (hmulright : ∀ a : R, Tendsto (fun m : M => a • m) (𝓝 0) (𝓝 0)) : ContinuousSMul R M where
  continuous_smul := by
    /-
      R : Type u_1
      M : Type u_2
      inst✝⁶ : Ring R
      inst✝⁵ : TopologicalSpace R
      inst✝⁴ : TopologicalSpace M
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : TopologicalRing R
      inst✝ : TopologicalAddGroup M
      hmul : Filter.Tendsto (fun p => HSMul.hSMul p.1 p.2) (SProd.sprod (nhds 0) (nh …
      hmulleft : ∀ (m : M), Filter.Tendsto (fun a => HSMul.hSMul a m) (nhds 0) (nhds …
      hmulright : ∀ (a : R), Filter.Tendsto (fun m => HSMul.hSMul a m) (nhds 0) (nhd …
      ⊢ Continuous fun p => HSMul.hSMul p.1 p.2
    -/
    rw [← nhds_prod_eq] at hmul
    /-
      R : Type u_1
      M : Type u_2
      inst✝⁶ : Ring R
      inst✝⁵ : TopologicalSpace R
      inst✝⁴ : TopologicalSpace M
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : TopologicalRing R
      inst✝ : TopologicalAddGroup M
      hmul : Filter.Tendsto (fun p => HSMul.hSMul p.1 p.2) (nhds { fst := 0, snd :=  …
      hmulleft : ∀ (m : M), Filter.Tendsto (fun a => HSMul.hSMul a m) (nhds 0) (nhds …
      hmulright : ∀ (a : R), Filter.Tendsto (fun m => HSMul.hSMul a m) (nhds 0) (nhd …
      ⊢ Continuous fun p => HSMul.hSMul p.1 p.2
    -/
    refine continuous_of_continuousAt_zero₂ (AddMonoidHom.smul : R →+ M →+ M) ?_ ?_ ?_ <;>
      /-
        case refine_1
        R : Type u_1
        M : Type u_2
        inst✝⁶ : Ring R
        inst✝⁵ : TopologicalSpace R
        inst✝⁴ : TopologicalSpace M
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : TopologicalRing R
        inst✝ : TopologicalAddGroup M
        hmul : Filter.Tendsto (fun p => HSMul.hSMul p.1 p.2) (nhds { fst := 0, snd :=  …
        hmulleft : ∀ (m : M), Filter.Tendsto (fun a => HSMul.hSMul a m) (nhds 0) (nhds …
        hmulright : ∀ (a : R), Filter.Tendsto (fun m => HSMul.hSMul a m) (nhds 0) (nhd …
        ⊢ ContinuousAt (fun x => (AddMonoidHom.smul x.1) x.2) { fst := 0, snd := 0 }
      -/
      /-
        🎉 no goals
      -/
      /-
        🎉 no goals
      -/
      simpa [ContinuousAt]
      /-
        🎉 no goals
      -/


/-- If `M` is a topological module over `R` and `0` is a limit of invertible elements of `R`, then
`⊤` is the only submodule of `M` with a nonempty interior.
This is the case, e.g., if `R` is a nontrivially normed field. -/
theorem Submodule.eq_top_of_nonempty_interior' [NeBot (𝓝[{ x : R | IsUnit x }] 0)]
    (s : Submodule R M) (hs : (interior (s : Set M)).Nonempty) : s = ⊤ := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁷ : Ring R
    inst✝⁶ : TopologicalSpace R
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : AddCommGroup M
    inst✝³ : ContinuousAdd M
    inst✝² : Module R M
    inst✝¹ : ContinuousSMul R M
    inst✝ : (nhdsWithin 0 (setOf fun x => IsUnit x)).NeBot
    s : Submodule R M
    hs : (interior ↑s).Nonempty
    ⊢ Eq s Top.top
  -/
  rcases hs with ⟨y, hy⟩
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝⁷ : Ring R
    inst✝⁶ : TopologicalSpace R
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : AddCommGroup M
    inst✝³ : ContinuousAdd M
    inst✝² : Module R M
    inst✝¹ : ContinuousSMul R M
    inst✝ : (nhdsWithin 0 (setOf fun x => IsUnit x)).NeBot
    s : Submodule R M
    y : M
    hy : Membership.mem (interior ↑s) y
    ⊢ Eq s Top.top
  -/
  refine Submodule.eq_top_iff'.2 fun x => ?_
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝⁷ : Ring R
    inst✝⁶ : TopologicalSpace R
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : AddCommGroup M
    inst✝³ : ContinuousAdd M
    inst✝² : Module R M
    inst✝¹ : ContinuousSMul R M
    inst✝ : (nhdsWithin 0 (setOf fun x => IsUnit x)).NeBot
    s : Submodule R M
    y : M
    hy : Membership.mem (interior ↑s) y
    x : M
    ⊢ Membership.mem s x
  -/
  rw [mem_interior_iff_mem_nhds] at hy
  have : Tendsto (fun c : R => y + c • x) (𝓝[{ x : R | IsUnit x }] 0) (𝓝 (y + (0 : R) • x)) :=
    tendsto_const_nhds.add ((tendsto_nhdsWithin_of_tendsto_nhds tendsto_id).smul tendsto_const_nhds)
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝⁷ : Ring R
    inst✝⁶ : TopologicalSpace R
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : AddCommGroup M
    inst✝³ : ContinuousAdd M
    inst✝² : Module R M
    inst✝¹ : ContinuousSMul R M
    inst✝ : (nhdsWithin 0 (setOf fun x => IsUnit x)).NeBot
    s : Submodule R M
    y : M
    hy : Membership.mem (nhds y) ↑s
    x : M
    this : Filter.Tendsto (fun c => HAdd.hAdd y (HSMul.hSMul c x)) (nhdsWithin 0 ( …
    ⊢ Membership.mem s x
  -/
  rw [zero_smul, add_zero] at this
  obtain ⟨_, hu : y + _ • _ ∈ s, u, rfl⟩ :=
    nonempty_of_mem (inter_mem (Filter.mem_map.1 (this hy)) self_mem_nhdsWithin)
  /-
    case intro.intro.intro.intro
    R : Type u_1
    M : Type u_2
    inst✝⁷ : Ring R
    inst✝⁶ : TopologicalSpace R
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : AddCommGroup M
    inst✝³ : ContinuousAdd M
    inst✝² : Module R M
    inst✝¹ : ContinuousSMul R M
    inst✝ : (nhdsWithin 0 (setOf fun x => IsUnit x)).NeBot
    s : Submodule R M
    y : M
    hy : Membership.mem (nhds y) ↑s
    x : M
    this : Filter.Tendsto (fun c => HAdd.hAdd y (HSMul.hSMul c x)) (nhdsWithin 0 ( …
    u : Units R
    hu : Membership.mem s (HAdd.hAdd y (HSMul.hSMul (↑u) x))
    ⊢ Membership.mem s x
  -/
  have hy' : y ∈ ↑s := mem_of_mem_nhds hy
  /-
    case intro.intro.intro.intro
    R : Type u_1
    M : Type u_2
    inst✝⁷ : Ring R
    inst✝⁶ : TopologicalSpace R
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : AddCommGroup M
    inst✝³ : ContinuousAdd M
    inst✝² : Module R M
    inst✝¹ : ContinuousSMul R M
    inst✝ : (nhdsWithin 0 (setOf fun x => IsUnit x)).NeBot
    s : Submodule R M
    y : M
    hy : Membership.mem (nhds y) ↑s
    x : M
    this : Filter.Tendsto (fun c => HAdd.hAdd y (HSMul.hSMul c x)) (nhdsWithin 0 ( …
    u : Units R
    hu : Membership.mem s (HAdd.hAdd y (HSMul.hSMul (↑u) x))
    hy' : Membership.mem s y
    ⊢ Membership.mem s x
  -/
  rwa [s.add_mem_iff_right hy', ← Units.smul_def, s.smul_mem_iff' u] at hu
  /-
    🎉 no goals
  -/


/-- Let `R` be a topological ring such that zero is not an isolated point (e.g., a nontrivially
normed field, see `NormedField.punctured_nhds_neBot`). Let `M` be a nontrivial module over `R`
such that `c • x = 0` implies `c = 0 ∨ x = 0`. Then `M` has no isolated points. We formulate this
using `NeBot (𝓝[≠] x)`.

This lemma is not an instance because Lean would need to find `[ContinuousSMul ?m_1 M]` with
unknown `?m_1`. We register this as an instance for `R = ℝ` in `Real.punctured_nhds_module_neBot`.
One can also use `haveI := Module.punctured_nhds_neBot R M` in a proof.
-/
theorem Module.punctured_nhds_neBot [Nontrivial M] [NeBot (𝓝[≠] (0 : R))] [NoZeroSMulDivisors R M]
    (x : M) : NeBot (𝓝[≠] x) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁹ : Ring R
    inst✝⁸ : TopologicalSpace R
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : ContinuousAdd M
    inst✝⁴ : Module R M
    inst✝³ : ContinuousSMul R M
    inst✝² : Nontrivial M
    inst✝¹ : (nhdsWithin 0 (HasCompl.compl (Singleton.singleton 0))).NeBot
    inst✝ : NoZeroSMulDivisors R M
    x : M
    ⊢ (nhdsWithin x (HasCompl.compl (Singleton.singleton x))).NeBot
  -/
  rcases exists_ne (0 : M) with ⟨y, hy⟩
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝⁹ : Ring R
    inst✝⁸ : TopologicalSpace R
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : ContinuousAdd M
    inst✝⁴ : Module R M
    inst✝³ : ContinuousSMul R M
    inst✝² : Nontrivial M
    inst✝¹ : (nhdsWithin 0 (HasCompl.compl (Singleton.singleton 0))).NeBot
    inst✝ : NoZeroSMulDivisors R M
    x y : M
    hy : Ne y 0
    ⊢ (nhdsWithin x (HasCompl.compl (Singleton.singleton x))).NeBot
  -/
  suffices Tendsto (fun c : R => x + c • y) (𝓝[≠] 0) (𝓝[≠] x) from this.neBot
  /-
    case intro
    R : Type u_1
    M : Type u_2
    inst✝⁹ : Ring R
    inst✝⁸ : TopologicalSpace R
    inst✝⁷ : TopologicalSpace M
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : ContinuousAdd M
    inst✝⁴ : Module R M
    inst✝³ : ContinuousSMul R M
    inst✝² : Nontrivial M
    inst✝¹ : (nhdsWithin 0 (HasCompl.compl (Singleton.singleton 0))).NeBot
    inst✝ : NoZeroSMulDivisors R M
    x y : M
    hy : Ne y 0
    ⊢ Filter.Tendsto (fun c => HAdd.hAdd x (HSMul.hSMul c y)) (nhdsWithin 0 (HasCo …
  -/
  refine Tendsto.inf ?_ (tendsto_principal_principal.2 <| ?_)
    /-
      case intro.refine_1
      R : Type u_1
      M : Type u_2
      inst✝⁹ : Ring R
      inst✝⁸ : TopologicalSpace R
      inst✝⁷ : TopologicalSpace M
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : ContinuousAdd M
      inst✝⁴ : Module R M
      inst✝³ : ContinuousSMul R M
      inst✝² : Nontrivial M
      inst✝¹ : (nhdsWithin 0 (HasCompl.compl (Singleton.singleton 0))).NeBot
      inst✝ : NoZeroSMulDivisors R M
      x y : M
      hy : Ne y 0
      ⊢ Filter.Tendsto (fun c => HAdd.hAdd x (HSMul.hSMul c y)) (nhds 0) (nhds x)
    -/
  · convert tendsto_const_nhds.add ((@tendsto_id R _).smul_const y)
    /-
      case h.e'_5.h.e'_3
      R : Type u_1
      M : Type u_2
      inst✝⁹ : Ring R
      inst✝⁸ : TopologicalSpace R
      inst✝⁷ : TopologicalSpace M
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : ContinuousAdd M
      inst✝⁴ : Module R M
      inst✝³ : ContinuousSMul R M
      inst✝² : Nontrivial M
      inst✝¹ : (nhdsWithin 0 (HasCompl.compl (Singleton.singleton 0))).NeBot
      inst✝ : NoZeroSMulDivisors R M
      x y : M
      hy : Ne y 0
      ⊢ Eq x (HAdd.hAdd x (HSMul.hSMul 0 y))
    -/
    rw [zero_smul, add_zero]
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_2
      R : Type u_1
      M : Type u_2
      inst✝⁹ : Ring R
      inst✝⁸ : TopologicalSpace R
      inst✝⁷ : TopologicalSpace M
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : ContinuousAdd M
      inst✝⁴ : Module R M
      inst✝³ : ContinuousSMul R M
      inst✝² : Nontrivial M
      inst✝¹ : (nhdsWithin 0 (HasCompl.compl (Singleton.singleton 0))).NeBot
      inst✝ : NoZeroSMulDivisors R M
      x y : M
      hy : Ne y 0
      ⊢ ∀ (a : R), Membership.mem (HasCompl.compl (Singleton.singleton 0)) a → Membe …
    -/
  · intro c hc
    /-
      case intro.refine_2
      R : Type u_1
      M : Type u_2
      inst✝⁹ : Ring R
      inst✝⁸ : TopologicalSpace R
      inst✝⁷ : TopologicalSpace M
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : ContinuousAdd M
      inst✝⁴ : Module R M
      inst✝³ : ContinuousSMul R M
      inst✝² : Nontrivial M
      inst✝¹ : (nhdsWithin 0 (HasCompl.compl (Singleton.singleton 0))).NeBot
      inst✝ : NoZeroSMulDivisors R M
      x y : M
      hy : Ne y 0
      c : R
      hc : Membership.mem (HasCompl.compl (Singleton.singleton 0)) c
      ⊢ Membership.mem (HasCompl.compl (Singleton.singleton x)) (HAdd.hAdd x (HSMul. …
    -/
    simpa [hy] using hc
    /-
      🎉 no goals
    -/


theorem continuousSMul_induced : @ContinuousSMul R M₁ _ u (t.induced f) :=
  let _ : TopologicalSpace M₁ := t.induced f
  IsInducing.continuousSMul ⟨rfl⟩ continuous_id (map_smul f _ _)


/-- The span of a separable subset with respect to a separable scalar ring is again separable. -/
lemma TopologicalSpace.IsSeparable.span {R M : Type*} [AddCommMonoid M] [Semiring R] [Module R M]
    [TopologicalSpace M] [TopologicalSpace R] [SeparableSpace R]
    [ContinuousAdd M] [ContinuousSMul R M] {s : Set M} (hs : IsSeparable s) :
    IsSeparable (Submodule.span R s : Set M) := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : Semiring R
    inst✝⁵ : Module R M
    inst✝⁴ : TopologicalSpace M
    inst✝³ : TopologicalSpace R
    inst✝² : TopologicalSpace.SeparableSpace R
    inst✝¹ : ContinuousAdd M
    inst✝ : ContinuousSMul R M
    s : Set M
    hs : TopologicalSpace.IsSeparable s
    ⊢ TopologicalSpace.IsSeparable ↑(Submodule.span R s)
  -/
  rw [span_eq_iUnion_nat]
  /-
    R : Type u_1
    M : Type u_2
    inst✝⁷ : AddCommMonoid M
    inst✝⁶ : Semiring R
    inst✝⁵ : Module R M
    inst✝⁴ : TopologicalSpace M
    inst✝³ : TopologicalSpace R
    inst✝² : TopologicalSpace.SeparableSpace R
    inst✝¹ : ContinuousAdd M
    inst✝ : ContinuousSMul R M
    s : Set M
    hs : TopologicalSpace.IsSeparable s
    ⊢ TopologicalSpace.IsSeparable (Set.iUnion fun n => Set.image (fun f => Finset …
  -/
  refine .iUnion fun n ↦ .image ?_ ?_
  · have : IsSeparable {f : Fin n → R × M | ∀ (i : Fin n), f i ∈ Set.univ ×ˢ s} := by
      apply isSeparable_pi (fun i ↦ .prod (.of_separableSpace Set.univ) hs)
    /-
      case refine_1
      R : Type u_1
      M : Type u_2
      inst✝⁷ : AddCommMonoid M
      inst✝⁶ : Semiring R
      inst✝⁵ : Module R M
      inst✝⁴ : TopologicalSpace M
      inst✝³ : TopologicalSpace R
      inst✝² : TopologicalSpace.SeparableSpace R
      inst✝¹ : ContinuousAdd M
      inst✝ : ContinuousSMul R M
      s : Set M
      hs : TopologicalSpace.IsSeparable s
      n : Nat
      this : TopologicalSpace.IsSeparable (setOf fun f => ∀ (i : Fin n), Membership. …
      ⊢ TopologicalSpace.IsSeparable (setOf fun f => ∀ (i : Fin n), Membership.mem s …
    -/
    rwa [Set.univ_prod] at this
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      M : Type u_2
      inst✝⁷ : AddCommMonoid M
      inst✝⁶ : Semiring R
      inst✝⁵ : Module R M
      inst✝⁴ : TopologicalSpace M
      inst✝³ : TopologicalSpace R
      inst✝² : TopologicalSpace.SeparableSpace R
      inst✝¹ : ContinuousAdd M
      inst✝ : ContinuousSMul R M
      s : Set M
      hs : TopologicalSpace.IsSeparable s
      n : Nat
      ⊢ Continuous fun f => Finset.univ.sum fun i => HSMul.hSMul (f i).1 (f i).2
    -/
  · apply continuous_finset_sum _ (fun i _ ↦ ?_)
    /-
      R : Type u_1
      M : Type u_2
      inst✝⁷ : AddCommMonoid M
      inst✝⁶ : Semiring R
      inst✝⁵ : Module R M
      inst✝⁴ : TopologicalSpace M
      inst✝³ : TopologicalSpace R
      inst✝² : TopologicalSpace.SeparableSpace R
      inst✝¹ : ContinuousAdd M
      inst✝ : ContinuousSMul R M
      s : Set M
      hs : TopologicalSpace.IsSeparable s
      n : Nat
      i : Fin n
      x✝ : Membership.mem Finset.univ i
      ⊢ Continuous fun a => HSMul.hSMul (a i).1 (a i).2
    -/
    exact (continuous_fst.comp (continuous_apply i)).smul (continuous_snd.comp (continuous_apply i))
    /-
      🎉 no goals
    -/


instance topologicalAddGroup [Ring α] [AddCommGroup β] [Module α β] [TopologicalAddGroup β]
    (S : Submodule α β) : TopologicalAddGroup S :=
  inferInstanceAs (TopologicalAddGroup S.toAddSubgroup)


theorem Submodule.mapsTo_smul_closure (s : Submodule R M) (c : R) :
    Set.MapsTo (c • ·) (closure s : Set M) (closure s) :=
  have : Set.MapsTo (c • ·) (s : Set M) s := fun _ h ↦ s.smul_mem c h
  this.closure (continuous_const_smul c)


theorem Submodule.smul_closure_subset (s : Submodule R M) (c : R) :
    c • closure (s : Set M) ⊆ closure (s : Set M) :=
  (s.mapsTo_smul_closure c).image_subset


/-- The (topological-space) closure of a submodule of a topological `R`-module `M` is itself
a submodule. -/
def Submodule.topologicalClosure (s : Submodule R M) : Submodule R M :=
  { s.toAddSubmonoid.topologicalClosure with
    smul_mem' := s.mapsTo_smul_closure }


@[simp]
theorem Submodule.topologicalClosure_coe (s : Submodule R M) :
    (s.topologicalClosure : Set M) = closure (s : Set M) :=
  rfl


theorem Submodule.le_topologicalClosure (s : Submodule R M) : s ≤ s.topologicalClosure :=
  subset_closure


theorem Submodule.closure_subset_topologicalClosure_span (s : Set M) :
    closure s ⊆ (span R s).topologicalClosure := by
  /-
    R : Type u
    M : Type v
    inst✝⁵ : Semiring R
    inst✝⁴ : TopologicalSpace M
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : ContinuousConstSMul R M
    inst✝ : ContinuousAdd M
    s : Set M
    ⊢ HasSubset.Subset (closure s) ↑(Submodule.span R s).topologicalClosure
  -/
  rw [Submodule.topologicalClosure_coe]
  /-
    R : Type u
    M : Type v
    inst✝⁵ : Semiring R
    inst✝⁴ : TopologicalSpace M
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : ContinuousConstSMul R M
    inst✝ : ContinuousAdd M
    s : Set M
    ⊢ HasSubset.Subset (closure s) (closure ↑(Submodule.span R s))
  -/
  exact closure_mono subset_span
  /-
    🎉 no goals
  -/


theorem Submodule.isClosed_topologicalClosure (s : Submodule R M) :
    IsClosed (s.topologicalClosure : Set M) := isClosed_closure


theorem Submodule.topologicalClosure_minimal (s : Submodule R M) {t : Submodule R M} (h : s ≤ t)
    (ht : IsClosed (t : Set M)) : s.topologicalClosure ≤ t :=
  closure_minimal h ht


theorem Submodule.topologicalClosure_mono {s : Submodule R M} {t : Submodule R M} (h : s ≤ t) :
    s.topologicalClosure ≤ t.topologicalClosure :=
  closure_mono h


/-- The topological closure of a closed submodule `s` is equal to `s`. -/
theorem IsClosed.submodule_topologicalClosure_eq {s : Submodule R M} (hs : IsClosed (s : Set M)) :
    s.topologicalClosure = s :=
  SetLike.ext' hs.closure_eq


/-- A subspace is dense iff its topological closure is the entire space. -/
theorem Submodule.dense_iff_topologicalClosure_eq_top {s : Submodule R M} :
    Dense (s : Set M) ↔ s.topologicalClosure = ⊤ := by
  /-
    R : Type u
    M : Type v
    inst✝⁵ : Semiring R
    inst✝⁴ : TopologicalSpace M
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : ContinuousConstSMul R M
    inst✝ : ContinuousAdd M
    s : Submodule R M
    ⊢ Iff (Dense ↑s) (Eq s.topologicalClosure Top.top)
  -/
  rw [← SetLike.coe_set_eq, dense_iff_closure_eq]
  /-
    R : Type u
    M : Type v
    inst✝⁵ : Semiring R
    inst✝⁴ : TopologicalSpace M
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : ContinuousConstSMul R M
    inst✝ : ContinuousAdd M
    s : Submodule R M
    ⊢ Iff (Eq (closure ↑s) Set.univ) (Eq ↑s.topologicalClosure ↑Top.top)
  -/
  simp
  /-
    🎉 no goals
  -/


instance Submodule.topologicalClosure.completeSpace {M' : Type*} [AddCommMonoid M'] [Module R M']
    [UniformSpace M'] [ContinuousAdd M'] [ContinuousConstSMul R M'] [CompleteSpace M']
    (U : Submodule R M') : CompleteSpace U.topologicalClosure :=
  isClosed_closure.completeSpace_coe


/-- A maximal proper subspace of a topological module (i.e a `Submodule` satisfying `IsCoatom`)
is either closed or dense. -/
theorem Submodule.isClosed_or_dense_of_isCoatom (s : Submodule R M) (hs : IsCoatom s) :
    IsClosed (s : Set M) ∨ Dense (s : Set M) := by
  /-
    R : Type u
    M : Type v
    inst✝⁵ : Semiring R
    inst✝⁴ : TopologicalSpace M
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : ContinuousConstSMul R M
    inst✝ : ContinuousAdd M
    s : Submodule R M
    hs : IsCoatom s
    ⊢ Or (IsClosed ↑s) (Dense ↑s)
  -/
  refine (hs.le_iff.mp s.le_topologicalClosure).symm.imp ?_ dense_iff_topologicalClosure_eq_top.mpr
  /-
    R : Type u
    M : Type v
    inst✝⁵ : Semiring R
    inst✝⁴ : TopologicalSpace M
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    inst✝¹ : ContinuousConstSMul R M
    inst✝ : ContinuousAdd M
    s : Submodule R M
    hs : IsCoatom s
    ⊢ Eq s.topologicalClosure s → IsClosed ↑s
  -/
  exact fun h ↦ h ▸ isClosed_closure
  /-
    🎉 no goals
  -/


theorem LinearMap.continuous_on_pi {ι : Type*} {R : Type*} {M : Type*} [Finite ι] [Semiring R]
    [TopologicalSpace R] [AddCommMonoid M] [Module R M] [TopologicalSpace M] [ContinuousAdd M]
    [ContinuousSMul R M] (f : (ι → R) →ₗ[R] M) : Continuous f := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    inst✝⁷ : Finite ι
    inst✝⁶ : Semiring R
    inst✝⁵ : TopologicalSpace R
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module R M
    inst✝² : TopologicalSpace M
    inst✝¹ : ContinuousAdd M
    inst✝ : ContinuousSMul R M
    f : LinearMap (RingHom.id R) (ι → R) M
    ⊢ Continuous ⇑f
  -/
  cases nonempty_fintype ι
  classical
    -- for the proof, write `f` in the standard basis, and use that each coordinate is a continuous
    -- function.
    have : (f : (ι → R) → M) = fun x => ∑ i : ι, x i • f fun j => if i = j then 1 else 0 := by
      ext x
      exact f.pi_apply_eq_sum_univ x
    rw [this]
    refine continuous_finset_sum _ fun i _ => ?_
    exact (continuous_apply i).smul continuous_const


/-- Constructs a bundled linear map from a function and a proof that this function belongs to the
closure of the set of linear maps. -/
@[simps (config := .asFn)]
def linearMapOfMemClosureRangeCoe (f : M₁ → M₂)
    (hf : f ∈ closure (Set.range ((↑) : (M₁ →ₛₗ[σ] M₂) → M₁ → M₂))) : M₁ →ₛₗ[σ] M₂ :=
  { addMonoidHomOfMemClosureRangeCoe f hf with
    map_smul' := (isClosed_setOf_map_smul M₁ M₂ σ).closure_subset_iff.2
      (Set.range_subset_iff.2 LinearMap.map_smulₛₗ) hf }


/-- Construct a bundled linear map from a pointwise limit of linear maps -/
@[simps! (config := .asFn)]
def linearMapOfTendsto (f : M₁ → M₂) (g : α → M₁ →ₛₗ[σ] M₂) [l.NeBot]
    (h : Tendsto (fun a x => g a x) l (𝓝 f)) : M₁ →ₛₗ[σ] M₂ :=
  linearMapOfMemClosureRangeCoe f <|
    mem_closure_of_tendsto h <| Eventually.of_forall fun _ => Set.mem_range_self _


theorem LinearMap.isClosed_range_coe : IsClosed (Set.range ((↑) : (M₁ →ₛₗ[σ] M₂) → M₁ → M₂)) :=
  isClosed_of_closure_subset fun f hf => ⟨linearMapOfMemClosureRangeCoe f hf, rfl⟩


instance _root_.QuotientModule.Quotient.topologicalSpace : TopologicalSpace (M ⧸ S) :=
  inferInstanceAs (TopologicalSpace (Quotient S.quotientRel))


theorem isOpenMap_mkQ [ContinuousAdd M] : IsOpenMap S.mkQ :=
  QuotientAddGroup.isOpenMap_coe


theorem isOpenQuotientMap_mkQ [ContinuousAdd M] : IsOpenQuotientMap S.mkQ :=
  QuotientAddGroup.isOpenQuotientMap_mk


instance topologicalAddGroup_quotient [TopologicalAddGroup M] : TopologicalAddGroup (M ⧸ S) :=
  inferInstanceAs <| TopologicalAddGroup (M ⧸ S.toAddSubgroup)


instance continuousSMul_quotient [TopologicalSpace R] [TopologicalAddGroup M] [ContinuousSMul R M] :
    ContinuousSMul R (M ⧸ S) where
  continuous_smul := by
    /-
      R : Type u_1
      M : Type u_2
      inst✝⁶ : Ring R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : TopologicalSpace M
      S : Submodule R M
      inst✝² : TopologicalSpace R
      inst✝¹ : TopologicalAddGroup M
      inst✝ : ContinuousSMul R M
      ⊢ Continuous fun p => HSMul.hSMul p.1 p.2
    -/
    rw [← (IsOpenQuotientMap.id.prodMap S.isOpenQuotientMap_mkQ).continuous_comp_iff]
    /-
      R : Type u_1
      M : Type u_2
      inst✝⁶ : Ring R
      inst✝⁵ : AddCommGroup M
      inst✝⁴ : Module R M
      inst✝³ : TopologicalSpace M
      S : Submodule R M
      inst✝² : TopologicalSpace R
      inst✝¹ : TopologicalAddGroup M
      inst✝ : ContinuousSMul R M
      ⊢ Continuous (Function.comp (fun p => HSMul.hSMul p.1 p.2) (Prod.map id ⇑S.mkQ))
    -/
    exact continuous_quot_mk.comp continuous_smul
    /-
      🎉 no goals
    -/


instance t3_quotient_of_isClosed [TopologicalAddGroup M] [IsClosed (S : Set M)] :
    T3Space (M ⧸ S) :=
  letI : IsClosed (S.toAddSubgroup : Set M) := ‹_›
  QuotientAddGroup.instT3Space S.toAddSubgroup


