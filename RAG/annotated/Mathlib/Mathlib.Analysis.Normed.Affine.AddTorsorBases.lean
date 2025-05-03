theorem isOpenMap_barycentric_coord [Nontrivial ι] (b : AffineBasis ι 𝕜 P) (i : ι) :
    IsOpenMap (b.coord i) :=
  AffineMap.isOpenMap_linear_iff.mp <|
    (b.coord i).linear.isOpenMap_of_finiteDimensional <|
      (b.coord i).linear_surjective_iff.mpr (b.surjective_coord i)


@[continuity]
theorem continuous_barycentric_coord (i : ι) : Continuous (b.coord i) :=
  (b.coord i).continuous_of_finiteDimensional


/-- Given a finite-dimensional normed real vector space, the interior of the convex hull of an
affine basis is the set of points whose barycentric coordinates are strictly positive with respect
to this basis.

TODO Restate this result for affine spaces (instead of vector spaces) once the definition of
convexity is generalised to this setting. -/
theorem AffineBasis.interior_convexHull {ι E : Type*} [Finite ι] [NormedAddCommGroup E]
    [NormedSpace ℝ E] (b : AffineBasis ι ℝ E) :
    interior (convexHull ℝ (range b)) = {x | ∀ i, 0 < b.coord i x} := by
  /-
    ι : Type u_1
    E : Type u_2
    inst✝² : Finite ι
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    b : AffineBasis ι Real E
    ⊢ Eq (interior ((convexHull Real) (Set.range ⇑b))) (setOf fun x => ∀ (i : ι),  …
  -/
  cases subsingleton_or_nontrivial ι
  · -- The zero-dimensional case.
    have : range b = univ :=
      AffineSubspace.eq_univ_of_subsingleton_span_eq_top (subsingleton_range _) b.tot
    /-
      case inl
      ι : Type u_1
      E : Type u_2
      inst✝² : Finite ι
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      b : AffineBasis ι Real E
      h✝ : Subsingleton ι
      this : Eq (Set.range ⇑b) Set.univ
      ⊢ Eq (interior ((convexHull Real) (Set.range ⇑b))) (setOf fun x => ∀ (i : ι),  …
    -/
    simp [this]
    /-
      🎉 no goals
    -/
  · -- The positive-dimensional case.
    /-
      case inr
      ι : Type u_1
      E : Type u_2
      inst✝² : Finite ι
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      b : AffineBasis ι Real E
      h✝ : Nontrivial ι
      ⊢ Eq (interior ((convexHull Real) (Set.range ⇑b))) (setOf fun x => ∀ (i : ι),  …
    -/
    haveI : FiniteDimensional ℝ E := b.finiteDimensional
    have : convexHull ℝ (range b) = ⋂ i, b.coord i ⁻¹' Ici 0 := by
      rw [b.convexHull_eq_nonneg_coord, setOf_forall]; rfl
    /-
      case inr
      ι : Type u_1
      E : Type u_2
      inst✝² : Finite ι
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      b : AffineBasis ι Real E
      h✝ : Nontrivial ι
      this✝ : FiniteDimensional Real E
      this : Eq ((convexHull Real) (Set.range ⇑b)) (Set.iInter fun i => Set.preimage …
      ⊢ Eq (interior ((convexHull Real) (Set.range ⇑b))) (setOf fun x => ∀ (i : ι),  …
    -/
    ext
    simp only [this, interior_iInter_of_finite, ←
      IsOpenMap.preimage_interior_eq_interior_preimage (isOpenMap_barycentric_coord b _)
        (continuous_barycentric_coord b _),
      interior_Ici, mem_iInter, mem_setOf_eq, mem_Ioi, mem_preimage]


/-- Given a set `s` of affine-independent points belonging to an open set `u`, we may extend `s` to
an affine basis, all of whose elements belong to `u`. -/
theorem IsOpen.exists_between_affineIndependent_span_eq_top {s u : Set P} (hu : IsOpen u)
    (hsu : s ⊆ u) (hne : s.Nonempty) (h : AffineIndependent ℝ ((↑) : s → P)) :
    ∃ t : Set P, s ⊆ t ∧ t ⊆ u ∧ AffineIndependent ℝ ((↑) : t → P) ∧ affineSpan ℝ t = ⊤ := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s u : Set P
    hu : IsOpen u
    hsu : HasSubset.Subset s u
    hne : s.Nonempty
    h : AffineIndependent Real Subtype.val
    ⊢ Exists fun t => And (HasSubset.Subset s t) (And (HasSubset.Subset t u) (And  …
  -/
  obtain ⟨q, hq⟩ := hne
  /-
    case intro
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s u : Set P
    hu : IsOpen u
    hsu : HasSubset.Subset s u
    h : AffineIndependent Real Subtype.val
    q : P
    hq : Membership.mem s q
    ⊢ Exists fun t => And (HasSubset.Subset s t) (And (HasSubset.Subset t u) (And  …
  -/
  obtain ⟨ε, ε0, hεu⟩ := Metric.nhds_basis_closedBall.mem_iff.1 (hu.mem_nhds <| hsu hq)
  /-
    case intro.intro.intro
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s u : Set P
    hu : IsOpen u
    hsu : HasSubset.Subset s u
    h : AffineIndependent Real Subtype.val
    q : P
    hq : Membership.mem s q
    ε : Real
    ε0 : LT.lt 0 ε
    hεu : HasSubset.Subset (Metric.closedBall q ε) u
    ⊢ Exists fun t => And (HasSubset.Subset s t) (And (HasSubset.Subset t u) (And  …
  -/
  obtain ⟨t, ht₁, ht₂, ht₃⟩ := exists_subset_affineIndependent_affineSpan_eq_top h
  /-
    case intro.intro.intro.intro.intro.intro
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    s u : Set P
    hu : IsOpen u
    hsu : HasSubset.Subset s u
    h : AffineIndependent Real Subtype.val
    q : P
    hq : Membership.mem s q
    ε : Real
    ε0 : LT.lt 0 ε
    hεu : HasSubset.Subset (Metric.closedBall q ε) u
    t : Set P
    ht₁ : HasSubset.Subset s t
    ht₂ : AffineIndependent Real fun p => ↑p
    ht₃ : Eq (affineSpan Real t) Top.top
    ⊢ Exists fun t => And (HasSubset.Subset s t) (And (HasSubset.Subset t u) (And  …
  -/
  let f : P → P := fun y => lineMap q y (ε / dist y q)
  have hf : ∀ y, f y ∈ u := by
    refine fun y => hεu ?_
    simp only [f]
    rw [Metric.mem_closedBall, lineMap_apply, dist_vadd_left, norm_smul, Real.norm_eq_abs,
      dist_eq_norm_vsub V y q, abs_div, abs_of_pos ε0, abs_of_nonneg (norm_nonneg _), div_mul_comm]
    exact mul_le_of_le_one_left ε0.le (div_self_le_one _)
  have hεyq : ∀ y ∉ s, ε / dist y q ≠ 0 := fun y hy =>
    div_ne_zero ε0.ne' (dist_ne_zero.2 (ne_of_mem_of_not_mem hq hy).symm)
  classical
  let w : t → ℝˣ := fun p => if hp : (p : P) ∈ s then 1 else Units.mk0 _ (hεyq (↑p) hp)
  refine ⟨Set.range fun p : t => lineMap q p (w p : ℝ), ?_, ?_, ?_, ?_⟩
  · intro p hp; use ⟨p, ht₁ hp⟩; simp [w, hp]
  · rintro y ⟨⟨p, hp⟩, rfl⟩
    by_cases hps : p ∈ s <;>
    simp only [w, hps, lineMap_apply_one, Units.val_mk0, dif_neg, dif_pos, not_false_iff,
      Units.val_one, Subtype.coe_mk] <;>
    [exact hsu hps; exact hf p]
  · exact (ht₂.units_lineMap ⟨q, ht₁ hq⟩ w).range
  · rw [affineSpan_eq_affineSpan_lineMap_units (ht₁ hq) w, ht₃]


theorem IsOpen.exists_subset_affineIndependent_span_eq_top {u : Set P} (hu : IsOpen u)
    (hne : u.Nonempty) : ∃ s ⊆ u, AffineIndependent ℝ ((↑) : s → P) ∧ affineSpan ℝ s = ⊤ := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    u : Set P
    hu : IsOpen u
    hne : u.Nonempty
    ⊢ Exists fun s => And (HasSubset.Subset s u) (And (AffineIndependent Real Subt …
  -/
  rcases hne with ⟨x, hx⟩
  rcases hu.exists_between_affineIndependent_span_eq_top (singleton_subset_iff.mpr hx)
    (singleton_nonempty _) (affineIndependent_of_subsingleton _ _) with ⟨s, -, hsu, hs⟩
  /-
    case intro.intro.intro.intro
    V : Type u_1
    P : Type u_2
    inst✝³ : NormedAddCommGroup V
    inst✝² : NormedSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    u : Set P
    hu : IsOpen u
    x : P
    hx : Membership.mem u x
    s : Set P
    hsu : HasSubset.Subset s u
    hs : And (AffineIndependent Real Subtype.val) (Eq (affineSpan Real s) Top.top)
    ⊢ Exists fun s => And (HasSubset.Subset s u) (And (AffineIndependent Real Subt …
  -/
  exact ⟨s, hsu, hs⟩
  /-
    🎉 no goals
  -/


/-- The affine span of a nonempty open set is `⊤`. -/
theorem IsOpen.affineSpan_eq_top {u : Set P} (hu : IsOpen u) (hne : u.Nonempty) :
    affineSpan ℝ u = ⊤ :=
  let ⟨_, hsu, _, hs'⟩ := hu.exists_subset_affineIndependent_span_eq_top hne
  top_unique <| hs' ▸ affineSpan_mono _ hsu


theorem affineSpan_eq_top_of_nonempty_interior {s : Set V}
    (hs : (interior <| convexHull ℝ s).Nonempty) : affineSpan ℝ s = ⊤ :=
  top_unique <| isOpen_interior.affineSpan_eq_top hs ▸
    (affineSpan_mono _ interior_subset).trans_eq (affineSpan_convexHull _)


theorem AffineBasis.centroid_mem_interior_convexHull {ι} [Fintype ι] (b : AffineBasis ι ℝ V) :
    Finset.univ.centroid ℝ b ∈ interior (convexHull ℝ (range b)) := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : NormedSpace Real V
    ι : Type u_3
    inst✝ : Fintype ι
    b : AffineBasis ι Real V
    ⊢ Membership.mem (interior ((convexHull Real) (Set.range ⇑b))) (Finset.centroi …
  -/
  haveI := b.nonempty
  simp only [b.interior_convexHull, mem_setOf_eq, b.coord_apply_centroid (Finset.mem_univ _),
    inv_pos, Nat.cast_pos, Finset.card_pos, Finset.univ_nonempty, forall_true_iff]


theorem interior_convexHull_nonempty_iff_affineSpan_eq_top [FiniteDimensional ℝ V] {s : Set V} :
    (interior (convexHull ℝ s)).Nonempty ↔ affineSpan ℝ s = ⊤ := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : NormedSpace Real V
    inst✝ : FiniteDimensional Real V
    s : Set V
    ⊢ Iff (interior ((convexHull Real) s)).Nonempty (Eq (affineSpan Real s) Top.top)
  -/
  refine ⟨affineSpan_eq_top_of_nonempty_interior, fun h => ?_⟩
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : NormedSpace Real V
    inst✝ : FiniteDimensional Real V
    s : Set V
    h : Eq (affineSpan Real s) Top.top
    ⊢ (interior ((convexHull Real) s)).Nonempty
  -/
  obtain ⟨t, hts, b, hb⟩ := AffineBasis.exists_affine_subbasis h
  suffices (interior (convexHull ℝ (range b))).Nonempty by
    rw [hb, Subtype.range_coe_subtype, setOf_mem_eq] at this
    refine this.mono (by gcongr)
  /-
    case intro.intro.intro
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : NormedSpace Real V
    inst✝ : FiniteDimensional Real V
    s : Set V
    h : Eq (affineSpan Real s) Top.top
    t : Set V
    hts : HasSubset.Subset t s
    b : AffineBasis (↑t) Real V
    hb : Eq (⇑b) Subtype.val
    ⊢ (interior ((convexHull Real) (Set.range ⇑b))).Nonempty
  -/
  lift t to Finset V using b.finite_set
  /-
    case intro.intro.intro.intro
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : NormedSpace Real V
    inst✝ : FiniteDimensional Real V
    s : Set V
    h : Eq (affineSpan Real s) Top.top
    t : Finset V
    hts : HasSubset.Subset (↑t) s
    b : AffineBasis (↑↑t) Real V
    hb : Eq (⇑b) Subtype.val
    ⊢ (interior ((convexHull Real) (Set.range ⇑b))).Nonempty
  -/
  exact ⟨_, b.centroid_mem_interior_convexHull⟩
  /-
    🎉 no goals
  -/


theorem Convex.interior_nonempty_iff_affineSpan_eq_top [FiniteDimensional ℝ V] {s : Set V}
    (hs : Convex ℝ s) : (interior s).Nonempty ↔ affineSpan ℝ s = ⊤ := by
  /-
    V : Type u_1
    inst✝² : NormedAddCommGroup V
    inst✝¹ : NormedSpace Real V
    inst✝ : FiniteDimensional Real V
    s : Set V
    hs : Convex Real s
    ⊢ Iff (interior s).Nonempty (Eq (affineSpan Real s) Top.top)
  -/
  rw [← interior_convexHull_nonempty_iff_affineSpan_eq_top, hs.convexHull_eq]
  /-
    🎉 no goals
  -/

